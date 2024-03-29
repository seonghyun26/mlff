"""
Copyright (C) 2023 Samsung Electronics Co. LTD

This software is a property of Samsung Electronics.
No part of this software, either material or conceptual may be copied or distributed, transmitted,
transcribed, stored in a retrieval system or translated into any human or computer language in any form by any means,
electronic, mechanical, manual or otherwise, or disclosed
to third parties without the express written permission of Samsung Electronics.
"""

import json
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from collections import defaultdict

from ase import io
from ase.build.supercells import make_supercell
from FOX import MultiMolecule

from src.common.registry import md_evaluate_registry
from src.md_evaluate.base_evaluator import BaseEvaluator
from src.md_evaluate.utils import calc_error_metric


import wandb
import imageio

@md_evaluate_registry.register_md_evaluate("df")
@md_evaluate_registry.register_md_evaluate("distribution_functions")
class DFEvaluator(BaseEvaluator):

    def calculate_rdf_fox(self, traj_atoms, out_identifier):
        mols = MultiMolecule.from_ase(traj_atoms)
        # assume periodic in all axis as RDF is meaningful only in periodic system...
        rdf = mols.init_rdf(periodic="xyz", dr=self.config["dr_rdf"], r_max=self.config["r_max_rdf"])
        pair_list = rdf.columns.values.tolist()

        filename_rdf = f"RDF_{out_identifier}.csv"
        rdf.to_csv(Path(self.config["res_out_dir"]) / filename_rdf)

        return rdf, pair_list
    
    def calculate_adf_fox(self, traj_atoms, out_identifier):
        mols = MultiMolecule.from_ase(traj_atoms)
        # assume periodic in all axis as ADF is meaningful only in periodic system...
        adf = mols.init_adf(periodic="xyz", r_max=self.config["r_max_adf"])
        triplet_list = adf.columns.values.tolist()

        filename_adf = f"ADF_{out_identifier}.csv"
        adf.to_csv(Path(self.config["res_out_dir"]) / filename_adf)

        return adf, triplet_list

    def generate_comparison_figure(self, distribution_ref, distribution_dict_mlff, combination_list, fig_name):
        x_axis_name = distribution_ref.index.name
        label_list = ["AI_MD"] + list(distribution_dict_mlff.keys())
        for combination in combination_list:
            plt.figure()
            ax = distribution_ref.plot(y=combination, use_index=True)
            for df_mlff in distribution_dict_mlff.values():
                df_mlff.plot(y=combination, use_index=True, ax=ax)

            ax.set_xlabel(x_axis_name)
            ax.set_ylabel('Distrubution Func.')
            ax.set_title(combination)
            ax.legend(label_list)
            plt.savefig(Path(self.config["res_out_dir"]) / f'{fig_name}_{combination}.png')
            
            image_path = Path(self.config["res_out_dir"]) / f'{fig_name}_{combination}.png'
            image_array = imageio.imread(image_path)
            wandb.log(
                {
                    "eval/"+fig_name: wandb.Image(image_array, file_type="png")
                }
            )
    
    def output_error_metrics(self, distribution_ref, distribution_dict_mlff, combination_list, file_name):
        distribution_error_dict = defaultdict(dict)
        for mlff_uid, distribution_df in distribution_dict_mlff.items():
            for combination in combination_list:
                distribution_error_dict[mlff_uid][combination] = \
                    calc_error_metric(distribution_df[combination].values,
                                      distribution_ref[combination].values,
                                      'mae')

            distribution_error_dict[mlff_uid]['average'] = \
                sum(distribution_error_dict[mlff_uid].values()) \
                    / len(distribution_error_dict[mlff_uid])

        out_file_path = Path(self.config["res_out_dir"]) / f"{file_name}.dat"
        with open(out_file_path, 'w') as f:
            json.dump(distribution_error_dict, f, indent=4)

    @staticmethod
    def get_traj_atoms(path, index, format, n_extend):
        traj_atoms = io.read(path, index=index, format=format)

        if n_extend is not None and n_extend != 1:
            traj_atoms_extended = []
            for atoms in traj_atoms:
                atoms.wrap()
                atoms_extended = make_supercell(
                    atoms, [[n_extend, 0, 0], [0, n_extend, 0], [0, 0, n_extend]], wrap=True)
                traj_atoms_extended.append(atoms_extended)
            traj_atoms = traj_atoms_extended

        return traj_atoms

    def evaluate(self):
        Path(self.config["res_out_dir"]).mkdir(parents=True, exist_ok=True)

        if "ai_md_traj" in self.config.keys() and "ai_md_dfs_results" in self.config.keys():
            raise RuntimeError("Only one file for reference is required by 'ai_md_traj' or 'ai_md_dfs_results")
        
        if "ai_md_traj" in self.config.keys():
            trajs_atoms_ai = DFEvaluator.get_traj_atoms(
                self.config["ai_md_traj"]["path"],
                index=self.config["ai_md_traj"].get("index", ":"),
                format=self.config["ai_md_traj"].get("format"),
                n_extend=self.config["ai_md_traj"].get("n_extend")
            )
            out_identifier_ai = f"AIMD_{self.config['ai_md_traj']['out_identifier']}"

            rdf_ref, pair_list_ref = self.calculate_rdf_fox(trajs_atoms_ai, out_identifier_ai)
            adf_ref, triplet_list_ref = self.calculate_adf_fox(trajs_atoms_ai, out_identifier_ai)
        else:
            rdf_ref = pd.read_csv(self.config['ai_md_dfs_results']['rdf_path'], index_col=0)
            pair_list_ref = rdf_ref.columns.values.tolist()
            adf_ref = pd.read_csv(self.config['ai_md_dfs_results']['adf_path'], index_col=0)
            triplet_list_ref = adf_ref.columns.values.tolist()

        rdf_dict_mlff = {}
        adf_dict_mlff = {}
        for mlff_uid, mlff_traj_dict in self.config["mlff_md_traj"].items():
            try:
                traj_atoms_mlff = \
                    DFEvaluator.get_traj_atoms(
                        mlff_traj_dict["path"],
                        index=mlff_traj_dict.get("index", ":"),
                        format=mlff_traj_dict.get("format"),
                        n_extend=mlff_traj_dict.get("n_extend")
                    )
            except:
                self.logger.info(f"trajectory for {mlff_uid} is missing. Skipping calculation for this trajectory")
                continue
            
            out_identifier_mlff = f"{mlff_uid}_{mlff_traj_dict['out_identifier']}"

            self.logger.info(f"Start calculating distrubution functions for model '{mlff_uid}'")
            try:
                rdf_dict_mlff[mlff_uid], pair_list_mlff = self.calculate_rdf_fox(traj_atoms_mlff, out_identifier_mlff)
                assert pair_list_ref == pair_list_mlff, \
                    "Atom types in 'mlff_md_traj' should be the same as those in 'ai_md_traj'"

                adf_dict_mlff[mlff_uid], _ = self.calculate_adf_fox(traj_atoms_mlff, out_identifier_mlff)
            except:
                self.logger.info(f"Failed to calculate rdf, adf for {mlff_uid}.")
                continue
        
        self.generate_comparison_figure(rdf_ref, rdf_dict_mlff, pair_list_ref,
                                        fig_name='RDF_compare')
        self.generate_comparison_figure(adf_ref, adf_dict_mlff, triplet_list_ref,
                                        fig_name='ADF_compare')
        self.output_error_metrics(rdf_ref, rdf_dict_mlff, pair_list_ref,
                                  file_name='RDF_error')
        self.output_error_metrics(adf_ref, adf_dict_mlff, triplet_list_ref,
                                  file_name='ADF_error')
