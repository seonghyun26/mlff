from mp_api.client import MPRester

API_KEY = "LqVBicSVWdpZFSv4YE3H91DGrBtKbLVA"

with MPRester(api_key=API_KEY) as mpr:
    data = mpr.materials.search(material_ids=["mp-5924"])
    print(data)