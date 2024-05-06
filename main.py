from MolTransformer import *
import numpy as np
#ex1 
#GM = GenerateMethods()
#smiles_list,selfies_list = GM.global_molecular_generation(n_samples = 100 ) # generate random n_sample of smiles and selfies within the range on latent space, note that the length of unique_smiles_list,unique_selfies_list might not equal to n_samples, since there might be duplicate  generatived molecules, only unique molecules are returned. 

# ex2
# GM = GenerateMethods()
# generated_results,_ = GM.local_molecular_generation(dataset = 'qm9',num_vector = 30)  # generate random num_vector of smiles and selfies from a random selfies in the dataset
# generated_results['SMILES']
# generated_results['SELFIES']
## generated_results sorted by pareto_frontie have saved to report_save_path/local_molecular_generation, default report_save_path is the package location/output/output/GenerateMethods/

#ex3
# GM = GenerateMethods(report_save_path = to your own path)
#initial_smile = GM.random_smile(dataset = 'qm9') # dataset can be your own csv file, make sure the files contains 'SELFIES'
#print('initial_smile: ', initial_smile)
#generated_results,_ = GM.neighboring_search(initial_smile = initial_smile,num_vector= 20)
#print('SMILES: ',generated_results['SMILES'])
#print('SELFIES: ',generated_results['SELFIES'])
interest_molecules = {
    'qm9':['C','c1ccccc1', 'NC2NC(=O)c1ncn(COCCO)c1N2',"Cc2c(N(C)C)c(=O)n(c1ccccc1)n2C",'CCNC2C1CCC(C1)C2c3ccccc3'],
	'ocelot': ['c2ccc(c1ccccc1)cc2','c5ccc4cc3cc2cc1ccccc1cc2cc3cc4c5','c1ccc3c(c1)sc4c2ccccc2sc34','CC1CC2CC(C)[Si]12C#Cc8c4cc3ccccc3cc4c(C#C[Si]56C(C)CC5CC6C)c9cc7ccccc7cc89',
			'Oc4ccc(c1cc(O)c(O)cc1c2cc(O)c(O)cc2c3ccc(O)c(O)c3)cc4O']}
dataset = 'qm9'
target_molecule = 0
report_save_path_base = ''
report_save_path = report_save_path_base +'/'+dataset+'/'+str(target_molecule)+'/'
GM = GenerateMethods(save = True,report_save_path = report_save_path)

molecules_generation_record = GM.optimistic_property_driven_molecules_generation(initial_smile = interest_molecules[dataset][target_molecule],dataset = dataset,k = 100,num_vector = 10000, sa_threshold = 6,initial_smile = '',resolution = 0.0001,search_range = 60,max_step = 10,alpha = 0.5) # dataset can be your own csv file, make sure the files contains 'SELFIES' 
print('smiles', molecules_generation_record['SMILES'])
print('properties', molecules_generation_record['Property'])

