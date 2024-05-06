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

# use the code to form a proper example with good command to be a example at readme.
GM = GenerateMethods()
initial_smile = GM.random_smile(dataset = 'qm9') # dataset can be your own csv file, make sure the files contains 'SELFIES' 
print('initial_smile', initial_smile)
ls = GM.smiles_2_latent_space([initial_smile])
print('ls shape', ls.shape)
## you may do some small change to the ls by pertumate or so
edit_strings = GM.latent_space_2_strings(ls)
selfies = GM.smile_2_selfies(initial_smile)
ls_selfies = GM.selfies_2_latent_space([selfies])
edit_strings_from_selfies = GM.latent_space_2_strings(ls_selfies)
#you can also do some pertubute on the ls and see whether the new ls give you a better property value by the following function 
GM.set_property_model(dataset = 'qm9')
properties_1 = GM.latent_spaces_2_properties(ls)
print('properties_1: ', properties_1)
#also we offer this function
properties_2 = GM.smiles_2_properties(edit_strings_from_selfies['SMILES'])
print('properties_2: ', properties_2)

#we can compare the selfies we get to see whether they are same
print('edit_smile: ',edit_strings['SMILES'][0])
print('edit_smiles_selfies: ',edit_strings_from_selfies['SMILES'][0])






