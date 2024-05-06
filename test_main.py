from MolTransformer import *

# molecular_evolution search the smiles/ molecules at the line from latent space of star molecules and ls of end_molecules.
GM = GenerateMethods(save = True)
start_molecule = 'c1ccccc1'
end_molecule = 'Oc4ccc(c1cc(O)c(O)cc1c2cc(O)c(O)cc2c3ccc(O)c(O)c3)cc4O'
molecules_df = GM.molecular_evolution( start_molecule, end_molecule, number = 100)
print(' smiles on path is: ', molecules_df['SMILES'])






