from MolTransformer import *

interest_molecules = {
    'qm9':['C','c1ccccc1', 'NC2NC(=O)c1ncn(COCCO)c1N2',"Cc2c(N(C)C)c(=O)n(c1ccccc1)n2C",'CCNC2C1CCC(C1)C2c3ccccc3'],
	'ocelot': ['c2ccc(c1ccccc1)cc2','c5ccc4cc3cc2cc1ccccc1cc2cc3cc4c5','c1ccc3c(c1)sc4c2ccccc2sc34','CC1CC2CC(C)[Si]12C#Cc8c4cc3ccccc3cc4c(C#C[Si]56C(C)CC5CC6C)c9cc7ccccc7cc89',
			'Oc4ccc(c1cc(O)c(O)cc1c2cc(O)c(O)cc2c3ccc(O)c(O)c3)cc4O']}
interest_molecules_step1 = {
    'qm9':['[H]C([N-1])([H])[H]',
    'c1ccccc1', 
    '[H]OC([H])([H])C([H])([H])OC([H])([H])N1C([H])=NC2=C1N([H])C([H])(N([H])[H])N([H])C2[H]',
    "[H]C1=C([H])C([H])=C(N2C(=O)/N(N(C([H])([H])[H])C([H])([H])[H])C(C([H])([H])[H])N2C([H])([H])[H])C([H])=C1[H]",
    '[H]C1=C([H])C([H])=C(C2([H])C([H])(N([H])C([H])([H])C([H])([H])[H])C([H])C([H])([H])C([H])([H])C2([H])C([H])[H])C([H])=C1[H]'],
	'ocelot': [
	'[H]C1=C([H])C([H])=C(C2N([H])C([H])=C([H])C([H])=C2[H])C([H])=C1[H]',
	'[H]C1=C([H])C([H])=C2C([H])=C3C([H])=C4C([H])=C5C([H])=C([H])N([H])C([H])C5=C([H])C4=C([H])C3=C([H])C2=C1[H]',
	'[H]C1=C([H])C([H])=C2C(=C1[H])SC=C2SC(C[H])C[H]',
	'CC1CC2CC(C)[Si]12C#Cc8c4cc3ccccc3cc4c(C#C[Si]56C(C)CC5CC6C)c9cc7ccccc7cc89',
	'[H]OC1=C([H])C([H])=C(C2=C([H])C(O[H])=C(O[H])C([H])=C2C3=C([H])C(ON)=C(O[H])C([H])=C3C4=C([H])C([H])=C(O[H])C(O[H])=C4[H])C([H])=C1O[H]']}
dataset = 'ocelot'
target_molecule = 1
report_save_path_base = '/work/mech-ai/bella/ChemTransformer/report/GenerativeMethods/optimistic_property_driven_molecules_generation_continue_800/'
report_save_path = report_save_path_base +'/'+dataset+'/'+str(target_molecule)+'/'
GM = GenerateMethods(save = True,report_save_path = report_save_path)

molecules_generation_record = GM.optimistic_property_driven_molecules_generation(initial_smile = interest_molecules_step1[dataset][target_molecule],dataset = dataset,k = 100,num_vector = 800, sa_threshold = 6,resolution = 0.0001,search_range = 60,max_step = 10,alpha = 0.5) # dataset can be your own csv file, make sure the files contains 'SELFIES' 
print('smiles', molecules_generation_record['SMILES'])
print('properties', molecules_generation_record['Property'])

