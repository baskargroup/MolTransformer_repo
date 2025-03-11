from MolTransformer import *

report_save_path_base = '/Users/tcpba/MolTransformer_repo/output/test_output/global_molecular_generation_1000/'

GM = GenerateMethods(save = True,report_save_path = report_save_path_base)

GM.global_molecular_generation(n_samples=1000)






