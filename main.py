from MolTransformer import *

interest_molecules = {
    'qm9':['C','c1ccccc1', 'NC2NC(=O)c1ncn(COCCO)c1N2',"Cc2c(N(C)C)c(=O)n(c1ccccc1)n2C",'CCNC2C1CCC(C1)C2c3ccccc3'],
	'ocelot': ['c2ccc(c1ccccc1)cc2','c5ccc4cc3cc2cc1ccccc1cc2cc3cc4c5','c1ccc3c(c1)sc4c2ccccc2sc34','CC1CC2CC(C)[Si]12C#Cc8c4cc3ccccc3cc4c(C#C[Si]56C(C)CC5CC6C)c9cc7ccccc7cc89',
			'Oc4ccc(c1cc(O)c(O)cc1c2cc(O)c(O)cc2c3ccc(O)c(O)c3)cc4O']}
dataset = 'qm9'
target_molecule = 4
report_save_path_base = '/work/mech-ai/bella/ChemTransformer/report/GenerativeMethods/optimistic_property_driven_molecules_generation_n800_k100/'
report_save_path = report_save_path_base +'/'+dataset+'/'+str(target_molecule)+'/'
GM = GenerateMethods(save = True,report_save_path = report_save_path)

molecules_generation_record = GM.optimistic_property_driven_molecules_generation(initial_smile = interest_molecules[dataset][target_molecule],dataset = dataset,k = 100,num_vector = 800, sa_threshold = 6,resolution = 0.0001,search_range = 60,max_step = 10,alpha = 0.5) # dataset can be your own csv file, make sure the files contains 'SELFIES' 
print('smiles', molecules_generation_record['SMILES'])
print('properties', molecules_generation_record['Property'])

