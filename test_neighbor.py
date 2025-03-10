from MolTransformer import *

save_path_1 = '/work/mech-ai/bella/MolTransformer_repo/output/to_baskar/TCNQ/'
GM = GenerateMethods(report_save_path=save_path_1,save = True)



"""
save_path_2 = '/work/mech-ai/bella/MolTransformer_repo/output/to_baskar/TCNE_4FTCNQ/'
GM.report_save_path = save_path_2
TCNE = "N#CC(C#N)=C(C#N)C#N"
FTCNQ = "FC1=C(F)C(\C(F)=C(F)/C1=C(\C#N)C#N)=C(\C#N)C#N"
TTF = "S1C=CSC1=C2SC=CS2"
GM.molecular_evolution(start_molecule = TCNE, end_molecule = FTCNQ, number = 10000)

save_path_3 = '/work/mech-ai/bella/MolTransformer_repo/output/to_baskar/TTF_4FTCNQ/'
GM.report_save_path = save_path_3
GM.molecular_evolution(start_molecule = TTF, end_molecule = FTCNQ, number = 10000)

"""
GM.report_save_path = save_path_1
TCNQ = 'c1cc(=C(C#N)C#N)ccc1=C(C#N)C#N'
print('Initial SMILE:', TCNQ)
print('Results save to: ', GM.report_save_path)
GM.local_molecular_generation(random_initial_smile = False, initial_smile = TCNQ,search_range = 40, resolution = 0.001,num_vector = 5000,sa_threshold=6)

