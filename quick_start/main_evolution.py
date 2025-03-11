from MolTransformer import *

interest_molecules = {
    'qm9':['C','c1ccccc1', 'NC2NC(=O)c1ncn(COCCO)c1N2',"Cc2c(N(C)C)c(=O)n(c1ccccc1)n2C",'CCNC2C1CCC(C1)C2c3ccccc3'],
	'ocelot': ['c2ccc(c1ccccc1)cc2','c5ccc4cc3cc2cc1ccccc1cc2cc3cc4c5','c1ccc3c(c1)sc4c2ccccc2sc34','CC1CC2CC(C)[Si]12C#Cc8c4cc3ccccc3cc4c(C#C[Si]56C(C)CC5CC6C)c9cc7ccccc7cc89',
			'Oc4ccc(c1cc(O)c(O)cc1c2cc(O)c(O)cc2c3ccc(O)c(O)c3)cc4O']}
count = 0
for i in range(5):
	for j in range(5):
		end_molecule =  interest_molecules['qm9'][i]
		start_molecule=  interest_molecules['ocelot'][j]
		report_save_path_base = '/work/mech-ai/bella/ChemTransformer/report/GenerativeMethods/reverse_molecular_evolution_histogram_1k_2/'
		report_save_path = report_save_path_base +'/'+str(count)+'/'
		GM = GenerateMethods(report_save_path = report_save_path)
		GM.molecular_evolution(start_molecule, end_molecule, number = 1000)
		count += 1
		print('done 1 pair')
