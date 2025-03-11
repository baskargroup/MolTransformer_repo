from MolTransformer import *
GM = GenerateMethods(report_save_path='/Users/tcpba/MolTransformer_repo/output/test_baskar')
initial_smile = 'c1ccccc1'
print('Initial SMILE:', initial_smile)
generated_results, _ = GM.neighboring_search(initial_smile=initial_smile, num_vector=100)
print('Generated SMILES:', generated_results['SMILES'])
print('Generated SELFIES:', generated_results['SELFIES'])