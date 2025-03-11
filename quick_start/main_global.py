from MolTransformer import *

report_save_path_base = '/work/mech-ai/bella/ChemTransformer/report/GenerativeMethods/global_molecular_generation_500/'

GM = GenerateMethods(save = True,report_save_path = report_save_path)

GM.global_molecular_generation(n_samples=500)


