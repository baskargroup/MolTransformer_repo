#### Example 4: Custom Latent Space Manipulation

This example shows how to manually manipulate the latent space representation of a molecule to explore structural variations. 

**Important:**  
Before generating or manipulating molecular structures, set the appropriate **property prediction model** using the `set_property_model` method.  
- For the QM9 dataset (`dataset='qm9'`), it will load a model predicting lumo to QM9 molecules.
- For the Ocelot dataset (`dataset='ocelot'`), it will load another model predicting AEA.

**Example usage:**

```python
from MolTransformer import GenerateMethods

# Initialize generator
GM = GenerateMethods()

# Set the property prediction model (required)
GM.set_property_model(dataset='qm9')  # or 'ocelot' depending on your data

# Select or define an initial SMILES molecule
initial_smile = GM.random_smile(dataset='qm9')
print('Initial SMILE:', initial_smile)

# Convert SMILES to latent space
latent_space = GM.smiles_2_latent_space([initial_smile])
print('Latent Space Shape:', latent_space.shape)

# Manually modify latent space here if desired
# Example: latent_space += np.random.normal(0, 0.1, latent_space.shape)

# Convert the modified latent space back to SMILES/SELFIES
edit_results = GM.latent_space_2_strings(latent_space)
print('Edited SMILE:', edit_results['SMILES'][0])
print('Edited SELFIES:', edit_results['SELFIES'][0])

# Compute and print molecular properties of the edited molecule
properties = GM.smiles_2_properties(edit_results['SMILES'])
print('Predicted Property Value:', float(properties[0][0]))
print('Properties Shape:', properties.shape)
```

- **Note**: Omitting `set_property_model()` will result in runtime errors. Ensure it is always called first for smooth execution.


#### Example 5: Optimistic Property-Driven Molecule Generation
This example showcases property-driven generation aimed at optimizing molecule properties through iterative exploration of the latent space. The k parameter selects the top k most similar neighboring molecules in each iteration, from which the molecule showing the most significant improvement in properties is chosen for the next cycle. This process necessitates defining the dataset, the number of latent space vectors to sample, and the k parameter that dictates the extent of the neighborhood search.
```python
from MolTransformer import GenerateMethods
GM = GenerateMethods(save=True)  # Enable saving of results and logs
molecules_generation_record = GM.optimistic_property_driven_molecules_generation(dataset='qm9',k=30,num_vector=100) # you can also specify your initail_smile = 'your interested molecule'
print('Generated SMILES:', molecules_generation_record['SMILES'])
print('Properties:', molecules_generation_record['Property'])
```

