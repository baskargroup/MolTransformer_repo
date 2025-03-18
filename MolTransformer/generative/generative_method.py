from typing import List
from ..model import BuildModel, IndexConvert
from .generative_utils import *
from . import settings 
import torch  # type: ignore
from torch.autograd import Variable  # type: ignore
import logging 
import selfies as sf # type: ignore
from rdkit import Chem # type: ignore
from rdkit.Chem import inchi # type: ignore
from numpy.random import choice
from collections import defaultdict

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config('config.json')

class GenerateMethods(IndexConvert):
    """
    1. local_molecular_generation: v
        It generate closet new valid molecules from normalize random ls vectors both positive and negtive directions.
    2. global_molecular_generation: v
        Randomly sample n number of  LS vectors; a random samples each dimension vector. 
        ! Analysis around ls space needed.  
    3. !  optimistic_property_driven_molecules_generation:
        ! adjust the needed for definition changed of NeighboringSearch
        Iteratively use LocalMolecularGeneration, then use Multi-Fedility model to predict the High-Fidelity lables.
        Then, chose the neighbor molecule with the highest predicted lable to be the new itial molecule for next iteration.
        If predicted HF label of  new initail molecule is not larger than epislon + predicted HF label of  old initail molecule, break the loop.
        # to run the method, we need to load multi-fidelity model...and compute the low-fidelity label
    4. !  molecular_evolution:
        Take LS of the moleular pair: ( start , end ), compute the vector of the LS and then take k points on the vector.
        Decode the k points, record them if they are unique molecules. 
        Plot and solve all the generative molecules. 
    5. smiles_2_latent_space v
    6. latent_space_2_strings v
    7. !  latent_space_2_properties:
        to use the function, make sure you call .set_property_model(dataset = 'qm9') or 'ocelot'
    smiles_2_properties
    8. compute_uniqueness_by_inchi v
    9. random_smiles(dataset = 'ocelot') or 'qm9' or a path v
    10. neighboring_search(smile) v
    11. set_property_model v
    12. sort_pareto_frontier  v
                    """
    def __init__(self,gpu_mode = False,report_save_path = '',save = False):
        super().__init__()  # Initialize the base IndexConvert class
        self.device = device

        

        self.gpu_mode = gpu_mode
        # the gpu mode here means parallel compute or not

        
        self.save = save or bool(report_save_path)
        self.base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../'))
        
        if self.save  and not report_save_path:
            report_save_path = os.path.join(self.base_dir, 'output','GenerateMethods/')
            print('Resault will be save to the following path: ', report_save_path)
        if self.save:
            self.report_save_path = report_save_path
            check_path(self.report_save_path)
            print('Resault will be save to the following path: ', report_save_path)

        build_model_instance = BuildModel(device=device,gpu_mode = self.gpu_mode)
        self.model = build_model_instance.model.to(self.device)

        self.iteration_num = 0

    def molecular_evolution(self, start_molecule, end_molecule, number):
        """
        Generates a linear interpolation in the latent space between two given molecules and analyzes the 
        evolutionary path. This method aims to explore the transformation from the start molecule to the end 
        molecule by generating intermediate molecular representations, assessing their novelty and validity, and 
        visualizing the transition.

        Parameters:
        - start_molecule (str): SMILES string representing the starting molecule.
        - end_molecule (str): SMILES string representing the target end molecule.
        - number (int): Number of intermediate molecules to generate along the path from start to end.

        Steps:
        1. Computes latent space representations (memory vectors) for both start and end molecules.
        2. Interpolates between these vectors to create intermediate representations and calculates their 
           distances and relative distances from the start.
        3. Converts each interpolated representation back to a SMILES string.
        4. Constructs a DataFrame with SMILES strings and their respective distances and ratios.
        5. Applies a filtering process to remove duplicates and validate molecules against PubChem.
        6. Visualizes all unique and valid molecules along the evolutionary path.
        7. Plots the distance ratios of the molecules to visualize the linear progression in the latent space.
        8. Saves the detailed information of the molecular evolution path to a CSV file.

        Returns:
        - A DataFrame containing the filtered and validated list of molecules along with their distances and 
          distance ratios from the start molecule.
        """

        # Step 1: Compute memory vectors for start and end molecules
        start_memory = self.smiles_2_latent_space(start_molecule)  # Ensure this returns a numpy
        end_memory = self.smiles_2_latent_space(end_molecule)      # Ensure this returns a numpy

        # Step 2: Interpolate vectors and compute distances and ratios using numpy
        interpolated_memories = [start_memory + (i / number) * (end_memory - start_memory) for i in range(number)]
        distances = [np.linalg.norm(mem - start_memory) for mem in interpolated_memories]
        max_distance = np.linalg.norm(end_memory - start_memory)  # Distance to the end molecule
        distance_ratios = [dist / max_distance for dist in distances]
        interpolated_memories_array = np.stack(interpolated_memories)
        print('shape of interpolated_memories_array: ', interpolated_memories_array.shape)

        # Step 3: Convert each interpolated memory back to SMILES
        # Adjusting for numpy arrays
        strings_list = self.latent_space_2_strings(interpolated_memories_array)

        # Step 4: Construct DataFrame with additional columns
        df = pd.DataFrame({
            'SMILES': [smiles for smiles in strings_list['SMILES']],
            'distance': distances,
            'distance_ratio': distance_ratios,
            'similarity_start':np.nan,
            'similarity_end':np.nan

        })

        # Step 5: Apply filter before returning
        filtered_df = filter_duplicates(df)  # Assuming filter_duplicates is defined within the class
        # check pubchem_api
        filtered_df = validate_smiles_in_pubchem(filtered_df) 
        filtered_df['similarity_start'] = calculate_tanimoto_similarity_from_target(filtered_df,start_molecule)
        filtered_df['similarity_end'] = calculate_tanimoto_similarity_from_target(filtered_df,end_molecule)

        if self.save:
            molecular_evolution_report_save_path = self.report_save_path + 'molecular_evolution/'
            check_path(molecular_evolution_report_save_path)
            
            # edit plot_tanimoto
            plot_tanimoto_histogram(filtered_df,out_dir = molecular_evolution_report_save_path)
            plot_tanimoto(filtered_df,out_dir = molecular_evolution_report_save_path)

            # step6: Plot all the molecules:
            for i, smiles in enumerate(filtered_df['SMILES']):
                if i == 0:
                    path = os.path.join(molecular_evolution_report_save_path + 'start_molecule')
                elif i == len(filtered_df)-1:
                    path = os.path.join(molecular_evolution_report_save_path + 'end_molecule')
                else:
                    path = os.path.join(molecular_evolution_report_save_path, f'molecule_{i}.png')
                plot_molecules(smiles, path)
            # step7: plot the linear line 
            plot_1d_distance_ratio_line(filtered_df,molecular_evolution_report_save_path,marker_size=10)
            # step 8: save dataframe to csv
            filtered_df.to_csv(os.path.join(molecular_evolution_report_save_path, 'MolecularLinearEvolution.csv'), index=False)

            smiles_list = filtered_df['SMILES']
            draw_all_structures(smiles_list, out_dir = molecular_evolution_report_save_path, mols_per_image = 15, molsPerRow = 5, name_tag = '', file_prefix= 'MolecularLinearEvolution',pop_first = False,molecule_prefix = 'step :  ')
        return filtered_df

    def global_molecular_generation(self, n_samples, sample_type='normal'):
        """
        Generates molecular structures by sampling in either the latent space (LS) 
        and converts those samples into SMILES and SELFIES representations.

        Parameters:
            n_samples (int): The number of samples to generate.
            sample_type (str): The type of sampling to use ('normal' for normal distribution,
                               'random' for uniform random sampling). Defaults to 'random'.

        Depending on the sample_space, the method performs sampling using normal or random distributions,
        then converts the samples to SMILES and SELFIES. The results are saved, and the uniqueness of
        the generated molecules is evaluated.
        """
        if self.save:
            global_molecular_generation_save_path = self.report_save_path + 'global_molecular_generation/'
            check_path(global_molecular_generation_save_path)

        # bounding_box 
        current_dir = os.path.dirname(__file__)
        LS_statistic_path = os.path.join(current_dir, 'LS_statistic')

        n_dimensions = 12030
        np.random.seed(42)  # For reproducible random values
        bounding_box = np.ones((n_dimensions, 2))
        bounding_box[:, 0] = np.load(LS_statistic_path + '/overall_max_vals.npy').squeeze()
        bounding_box[:, 1] = np.load(LS_statistic_path + '/overall_min_vals.npy').squeeze()

        n_components = bounding_box.shape[0]  # Determine the number of latent space components from the bounding box
        mean_matrix = np.load(LS_statistic_path + '/overall_mean_vals.npy').squeeze()  # Example mean, adjust as necessary
        std_dev_matrix = np.load(LS_statistic_path + '/overall_std_vals.npy').squeeze() # Example std dev, adjust as necessary
    
        mean_matrix = mean_matrix.reshape((n_dimensions,1))
        std_dev_matrix = std_dev_matrix.reshape((n_dimensions,1))
        print('bounding_box shape: ',bounding_box.shape)
        print('mean_matrix shape: ',mean_matrix.shape)
        print('std_dev_matrix shape: ',std_dev_matrix.shape)

        sampled_vectors = np.zeros((n_samples, n_components))

        for i in range(n_components):
            #print('components: ',i)
            #print('std_dev_matrix', std_dev_matrix[i])
            #print('min: ', bounding_box[i, 1])
            #print('max: ',bounding_box[i, 0])
            if sample_type == 'normal':
                #print('mean_matrix', mean_matrix[i])
                samples = np.random.normal(loc=mean_matrix[i], scale=np.abs(std_dev_matrix[i]), size=n_samples)
                sampled_vectors[:, i] = samples
            elif sample_type == 'uniform':
                sampled_vectors[:, i] = np.random.uniform(low=bounding_box[i, 0], high=bounding_box[i, 1], size=n_samples)
            else:
                raise ValueError("please choose a valid sample_type")

        smiles_list = []
        selfies_list = []
        n_batch = (n_samples // settings.batch_size) + (1 if n_samples % settings.batch_size != 0 else 0)

        for i in range(n_batch):
            print('in batch: ',i)
            start_idx = i * settings.batch_size
            end_idx = min((i + 1) * settings.batch_size, n_samples)
            # No need to subtract 1 from end_idx due to Python's exclusive range in slicing
            vector = sampled_vectors[start_idx:end_idx].reshape((-1, sampled_vectors.shape[1])).astype(np.float64)
            # The reshape now uses -1 for the first dimension to automatically adjust to the correct batch size

            strings = self.latent_space_2_strings(latent_space = vector)

            smiles_list += strings['SMILES']
            selfies_list += strings['SELFIES']

        uniqueness_ratio, unique_index , rdk_mol = self.compute_uniqueness_by_inchi(smiles_list,return_index = True)  # Ensure this function is defined
        logging.info(f"iteration_num: " + str(self.iteration_num))
        logging.info(f"uniqueness_ratio: " + str(uniqueness_ratio))
        print(f"iteration_num: " + str(self.iteration_num))
        print(f"uniqueness_ratio: " + str(uniqueness_ratio))

        unique_smiles_list = [smiles_list[i] for i in unique_index]
        unique_selfies_list = [selfies_list[i] for i in unique_index]
        unique_rdk_mol_list = [rdk_mol[i] for i in unique_index]

        if self.save:
            # Saving to a CSV file
            csv_file_path = global_molecular_generation_save_path + 'GlobalGeneratedMolecules.csv'
            df = pd.DataFrame({'SMILES': unique_smiles_list, 'SELFIES': unique_selfies_list})
            #df = validate_smiles_in_pubchem(df)  # Ensure this function is defined
            df.to_csv(csv_file_path, index=False)

            diversity_score = Tanimoto_diversity(unique_smiles_list)
            csv_file_path = global_molecular_generation_save_path + 'Analysis.csv'
            df_Analysis = pd.DataFrame({'uniqueness_ratio': [uniqueness_ratio], 'Tanimoto_diversity': [diversity_score]})
            #df = validate_smiles_in_pubchem(df)  # Ensure this function is defined
            df_Analysis.to_csv(csv_file_path, index=False)

            #plotting
            # test
            df['rdk_mol'] = unique_rdk_mol_list
            plot_generative_molecules_analysis(df,save_file = global_molecular_generation_save_path )
        return unique_smiles_list,unique_selfies_list
    
    def latent_space_2_strings(self, latent_space):
        # Reshape the numpy array
        ls = latent_space.reshape(-1, 401, 30)
        
        # Convert numpy array to a torch tensor and move it to the device
        if isinstance(ls, np.ndarray):
            torch_tensor = torch.as_tensor(ls).to(self.device)
        else:
            torch_tensor = ls.to(self.device)
        
        # Assuming _memory_2_representation is a method that takes a torch tensor and returns smiles and selfies
        representation = self._memory_2_representation(torch_tensor)
        return representation
    
    def _memory_2_representation(self,memory_torch):
        # Move input tensor explicitly to the correct device
        memory_torch = memory_torch.to(self.device)

        # Ensure the model itself is on the correct device
        self.model.to(self.device)

        # if the shape of memory_torch is (1, 401, 30) 
        if memory_torch.shape[1] == 401:
            memory = memory_torch.permute(1, 0, 2)
        else:
            memory = memory_torch
        if self.gpu_mode:
            molecule = self.model.module.decoder(memory)
        else:
            molecule = self.model.decoder(memory)

        smiles,selfies = self._idx_2_smiles(molecules_idx = molecule)
        return {'SMILES': smiles, 'SELFIES': selfies}
    
    def _idx_2_smiles(self,molecules_idx):
        selfies_list = self.index_2_selfies(molecules_idx)
        smiles_list = self.selfies_2_smile(selfies_list)
        return smiles_list,selfies_list
    
    def compute_uniqueness_by_inchi(self, smiles_list, return_index=False):
        # Initialize a dictionary to store InChI identifiers and their indices
        inchi_indices = {}
        rdk_mol = []
        
        # Convert each SMILES in the list to an InChI identifier
        for index, smiles in enumerate(smiles_list):
            try:
                mol = Chem.MolFromSmiles(smiles)
                rdk_mol.append(mol)
                if mol:  # Ensure the molecule was successfully created
                    inchi_code = inchi.MolToInchi(mol)
                    if inchi_code not in inchi_indices:
                        inchi_indices[inchi_code] = [index]  # Store the index of the first occurrence
                    else:
                        inchi_indices[inchi_code].append(index)  # Append subsequent occurrences
                else:
                    # Handle cases where the SMILES string could not be converted to a molecule
                    print(f"Warning: Failed to convert SMILES '{smiles}' to a molecule.")
            except:
                rdk_mol.append(None)
                print(f"Warning: Failed to convert SMILES '{smiles}' to a molecule.")

        # Extract indices of unique InChI codes
        unique_indices = [indices[0] for indices in inchi_indices.values()]  # Take the first index of each InChI code
        
        # Calculate uniqueness
        total_inchis = len(smiles_list)
        unique_inchis = len(unique_indices)
        uniqueness_ratio = unique_inchis / total_inchis if total_inchis > 0 else 0
        
        if return_index:
            return uniqueness_ratio, unique_indices ,rdk_mol
        else:
            return uniqueness_ratio
    
    def local_molecular_generation(self,top_k_closest = True, k = 30,alpha_list = [0,0.5,1], random_initial_smile = True,initial_smile = '',dataset = 'ocelot',search_range = 40, resolution = 0.001,num_vector = 150,sa_threshold=6):
        if self.save:
            local_molecular_generation_report_save_path = self.report_save_path + 'local_molecular_generation/'
            check_path(local_molecular_generation_report_save_path)
        # Check if top_k_closest is True and alpha_list is empty
        if top_k_closest and not alpha_list:
            raise ValueError("Alpha list cannot be empty when top_k_closest is True")

        if random_initial_smile:
            if not initial_smile:
                initial_smile = self.random_smile(dataset)
            else:
                raise ValueError("please not define initial_smile or set random_initial_smile to False")
        else:
            if not initial_smile:
                raise ValueError("please define initial_smile or set random to True")
        
        
        generated_results,fail_case_check = self.neighboring_search(initial_smile = initial_smile, search_range=search_range, resolution=resolution, num_vector=num_vector)
        
        uniqueness_ratio = self.compute_uniqueness_by_inchi(generated_results['SMILES'])
        logging.info(f"iteration_num :" + str(self.iteration_num))
        logging.info(f"uniqueness_ratio :" + str(uniqueness_ratio))
        print(f"iteration_num :" + str(self.iteration_num))
        print(f"uniqueness_ratio:" + str(uniqueness_ratio))

        if self.save:
            csv_file_path = local_molecular_generation_report_save_path + 'GenerateClosetMolecules' + '.csv'
            df = pd.DataFrame(generated_results)
            df.to_csv(csv_file_path, index=True)
            
            if top_k_closest:
                for alpha in alpha_list: 
                    self.sort_pareto_frontier(generated_results = df,alpha = alpha, k = k, save = True,folder_path = local_molecular_generation_report_save_path,sa_threshold=sa_threshold)
            
            csv_file_path = local_molecular_generation_report_save_path + 'fail_case_check' + '.csv'
            df = pd.DataFrame(fail_case_check)
            df.to_csv(csv_file_path, index=True)
        return generated_results
    
    def set_property_model(self,dataset):
        if dataset not in ['ocelot','qm9']:
            error_message = "Invalid dataset selection. Please choose from 'ocelot' or 'qm9'."
            print(error_message)
            raise ValueError(error_message)
        build_model_instance = BuildModel(device=device,gpu_mode = self.gpu_mode,dataset = dataset)
        self.property_model = build_model_instance.model.to(self.device)
        self.std_parameter =  defaultdict(float)
        model_folder = 'ocelot_aea' if dataset == 'ocelot' else 'qm9_lumo'
        model_path = os.path.join(self.base_dir,'MolTransformer','model','models','best_models','MultiF_HF',model_folder)
        mean1, std1, constant = np.load(model_path + '/std.npy')
        self.std_parameter.update({'mean': mean1, 'std': std1, 'constant': constant})

    def smiles_2_properties(self, smiles):
        # Check if the property model has been initialized
        if not hasattr(self, 'property_model'):
            error_message = ("Property model not set. Please call set_property_model(dataset) to initialize "
                            "the property model before using latent_space_2_properties.")
            print(error_message)
            raise AttributeError(error_message)
        
        # Prepare model input from smiles
        model_input = self._smile_2_property_model_input(smiles)

        # Explicitly send tensors to GPU
        input_tensor = model_input['input'].to(self.device)
        descriptors_tensor = model_input['descriptors'].to(self.device)
        
        # Predict properties using the initialized model
        
        predicted_property= self.property_model(input_tensor,descriptors_tensor)  # need convert to numpy array
        # Detach and convert tensor to numpy for further processing
        predicted_property_np = predicted_property.detach().cpu().numpy()
        # Convert standardized prediction outputs to actual property values
        properties = self._recover_standardized_data(predicted_property_np)
        return properties
    
    def latent_spaces_2_properties(self, latent_space):
        # Ensure the property model is loaded before using this function
        if not hasattr(self, 'property_model'):
            error_message = ("Property model not set. Please call set_property_model(dataset) to initialize "
                            "the property model before using latent_space_2_properties.")
            print(error_message)
            raise AttributeError(error_message)
        strings = self.latent_space_2_strings(latent_space)
        print('------- smiles: ',strings['SMILES'])
        properties = self.smiles_2_properties(strings['SMILES'])    
        return properties

    def _recover_standardized_data(self, data):
        mean1 = self.std_parameter['mean'] 
        std1 = self.std_parameter['std'] 
        constant = self.std_parameter['constant'] 
        data = data - constant
        data = data * std1 + mean1
        return data
    
    def optimistic_property_driven_molecules_generation(self, dataset = 'qm9',k = 30,num_vector = 100, sa_threshold = 6,initial_smile = '',resolution = 0.001,search_range = 40,max_step = 5,alpha = 0.5):
        #k: top k high score of the neighbors can be candidates for next molecule's next move
        if not hasattr(self, 'property_model'):
            self.set_property_model(dataset = dataset)
        
        # neighbor_search
        # get property
        # if not improve
        # stop
        # plot all molecules and txt the property
        # return {'SMILES','SELFIES', 'properties'}

        if not initial_smile:
            initial_smile = self.random_smile(dataset = dataset)
        initial_selfies = self.smile_2_selfies(initial_smile)
        current_property_ = self.smiles_2_properties(initial_smile)
        current_property = float(current_property_[0][0])
        molecules_generation_record = { 'SMILES':[initial_smile], 'Property':[current_property] , 'SELFIES':[initial_selfies]}
        improvement = True
        step = 0
        if self.save:
            optimistic_property_driven_molecules_generation_report_path = self.report_save_path + 'optimistic_property_driven_molecules_generation/'
            check_path(optimistic_property_driven_molecules_generation_report_path)
            plot_molecules(initial_smile, path=optimistic_property_driven_molecules_generation_report_path + 'initial')
            csv_file_path = optimistic_property_driven_molecules_generation_report_path + 'molecules_generation_record.csv'
            df = pd.DataFrame(molecules_generation_record)
            df = validate_smiles_in_pubchem(df) 
            df.to_csv(csv_file_path, mode='w', header=True, index=False)  # Write initial record

        current_smile = initial_smile
        while improvement and step < max_step:
            print('step: ', str(step))
            generated_results,_ = self.neighboring_search(initial_smile = current_smile, search_range=search_range, resolution=resolution, num_vector=num_vector)
            top_k_neighbors = self.sort_pareto_frontier(generated_results = generated_results,alpha = alpha, k = k, save = False, sa_threshold=sa_threshold)
            neighbor_properties = self.smiles_2_properties(top_k_neighbors['SMILES'])
            max_neighbor_properties = float(np.max(neighbor_properties))
            index_max_neighbor_properties = int(np.argmax(neighbor_properties))

            if max_neighbor_properties > current_property: # if improve
                molecules_generation_record['SMILES'].append(top_k_neighbors['SMILES'][index_max_neighbor_properties])
                molecules_generation_record['Property'].append(max_neighbor_properties)
                molecules_generation_record['SELFIES'].append(top_k_neighbors['SELFIES'][index_max_neighbor_properties])
                current_property = max_neighbor_properties
                current_smile = top_k_neighbors['SMILES'][index_max_neighbor_properties]
                if self.save:
                    plot_molecules(top_k_neighbors['SMILES'][index_max_neighbor_properties], path=optimistic_property_driven_molecules_generation_report_path + 'step_' + str(step + 1))
                    df = pd.DataFrame({'SMILES': [current_smile], 'Property': [current_property], 'SELFIES': [top_k_neighbors['SELFIES'][index_max_neighbor_properties]]})
                    df = validate_smiles_in_pubchem(df) 
                    df.to_csv(csv_file_path, mode='a', header=False, index=False)  # Append new record
                
            else:
                improvement = False
            step += 1

        #save revolution_record to csv file 
        if self.save:
            draw_all_structures(molecules_generation_record['SMILES'], out_dir = optimistic_property_driven_molecules_generation_report_path, mols_per_image = 10, molsPerRow = 5, name_tag = '', file_prefix= 'molecules_generation_',pop_first = False,molecule_prefix = 'step :  ')
        return molecules_generation_record

    def neighboring_search(self, initial_smile, search_range=40, resolution=0.001, num_vector=100):
        """
        Performs a neighboring search around an initial molecule represented by its SMILES string.
        This search aims to find minimal perturbations in the molecule's latent space representation
        that result in different molecules, using binary search to optimize the perturbation magnitude.

        Parameters:
        - initial_smile (str): SMILES representation of the initial molecule.
        - search_range (float): Maximum magnitude for perturbations.
        - resolution (float): The resolution of the search; smaller values yield more precise searches.
                                Must be strictly positive to avoid division by zero errors.
        - num_vector (int): Number of random vectors used for the search directions in the latent space.

        Returns:
        - generated_results (dict): Contains the dimensions, directions, radii, SMILES, and SELFIES of the generated molecules.
        - fail_case_check (dict): Tracks the SMILES and SELFIES that failed to generate valid molecules.
        """
        if self.save:
            neighboring_search_report_save_path = self.report_save_path + 'neighboring_search/'
            check_path(neighboring_search_report_save_path)
        # Validate resolution is strictly positive
        if resolution <= 0:
            raise ValueError("Resolution must be strictly positive to avoid potential division by zero.")

        # Initial setup common to both modes
        # Validate and initialize common parameters
        if search_range <= 0:
            raise ValueError("search_range must be positive.")
        
        ori_mol = Chem.MolFromSmiles(initial_smile)
        if not ori_mol:
            raise ValueError(f"Invalid initial_smile: {initial_smile}")
        
        ori_inchi = inchi.MolToInchi(ori_mol)
        ori_selfies = self.smile_2_selfies(initial_smile)
        fail_count = 0

        # Result and failure case tracking
        generated_results = {'dimension': ['initial'], 'direction': [0], 'radius': [0], 'SMILES': [initial_smile], 'SELFIES': [ori_selfies]}
        fail_case_check = {'SMILES': [], 'SELFIES': []}

        # Mode-specific initializations
        representation = self.smiles_2_latent_space([initial_smile])  # Use latent space representation for 'ls'
        representation = representation.reshape((1,-1))
        vector_generator = self._generate_normalized_vectors(dimensions=representation.shape[1], num_vectors=num_vector)
        directions = [-1,1]
        adjustment_scale = search_range
        print('vector_generator shape ', vector_generator.shape)
        print(f"Initial molecule: {initial_smile}")

        for vector_idx, vector in enumerate(vector_generator):
            print(f"Vector: {vector_idx}")
            vector = vector.reshape(1, -1)
            print('vector shape: ', vector.shape)
            for direction in directions:
                count = 0
                adjusted_representation = representation.copy()
                adjustment_scale = search_range  # Static adjustment scale for 'ls'
                r_left, r_right = 0, adjustment_scale
                min_success_r = adjustment_scale

                while r_left <= (r_right - resolution):
                    adjusted_representation = representation.copy()
                    r_mid = (r_right + r_left) / 2
                    adjusted_representation += (r_mid * direction) * vector
                    new_smiles, new_selfies = None, None
                    try:
                        strings = self.latent_space_2_strings(adjusted_representation)
                        new_smiles, new_selfies  = strings['SMILES'], strings['SELFIES']
                        new_mol = Chem.MolFromSmiles(new_smiles[0])
                        new_inchi = inchi.MolToInchi(new_mol)
                        if new_inchi != ori_inchi:
                            r_right = r_mid
                            min_success_r = r_mid
                        else:
                            r_left = r_mid
                    except Exception as e:
                        # Log error and adjust search as if it's too far
                        logging.info(f"Error for SMILES '{new_smiles[0]}' at radius {r_mid}")
                        fail_case_check['SMILES'].append(new_smiles[0] if new_smiles else "Conversion failed")
                        fail_case_check['SELFIES'].append(new_selfies[0] if new_selfies else "Conversion failed")
                        r_right = r_mid
                    count += 1
                #print('min_success_r', min_success_r)
                adjusted_representation = representation.copy()
                adjusted_representation += min_success_r * direction * vector
                
                final_smiles, final_selfies = None, None
                try:
                    strings = self.latent_space_2_strings(adjusted_representation)
                    final_smiles, final_selfies  = strings['SMILES'],strings['SELFIES']
                except Exception as e:
                    logging.info(f"Final conversion error for adjusted representation at radius {min_success_r}")
                    final_smiles, final_selfies = initial_smile, ori_selfies[0]
                
                try:
                    new_mol = Chem.MolFromSmiles(final_smiles[0])
                    new_inchi = inchi.MolToInchi(new_mol)
                except:
                    logging.info(f"Conversion error for SMILES '{final_smiles[0]}'")
                    logging.info("Current radius: " + str(min_success_r))
                    print('error in translate for: ', final_smiles[0])
                    print('Current radius: ' + str(min_success_r))
                    fail_case_check['SMILES'].append(final_smiles[0])
                    fail_case_check['SELFIES'].append(final_selfies[0])
                    new_inchi = ori_inchi
                    new_smiles = initial_smile

                if new_inchi != ori_inchi:
                    pass
                else:
                    fail_count += 1
                    logging.info( "For vector " + str(vector_idx) +' , the direction:'+ str(direction) + '  failed to find new molecule   ' + ' ,count: ' + str(count))
                generated_results['radius'].append(min_success_r)
                generated_results['dimension'].append('N/A')
                generated_results['direction'].append(direction)
                generated_results['SMILES'].append(final_smiles[0] if final_smiles else initial_smile)
                generated_results['SELFIES'].append(final_selfies[0] if final_selfies else ori_selfies[0])

        # Log final summary and save radius values if needed
        print(f"Number of failures: {fail_count}")
        print(f"Ratio of fail: {fail_count / (num_vector * 2 )}")
        r_list = np.array(generated_results['radius'][1:], dtype=float)  # Exclude the initial '0' radius
        if self.save:
            np.save(neighboring_search_report_save_path + 'radius_values.npy', r_list)
        logging.info(f"Min of r: {np.min(r_list)}")
        logging.info(f"Max of r: {np.max(r_list)}")
        logging.info(f"Mean of r: {np.mean(r_list)}")
        logging.info(f"Standard Deviation of r: {np.std(r_list)}")
        logging.info(f"Ratio of fail: {fail_count / (num_vector * 2 )}")

        return generated_results, fail_case_check
    


    def random_smile(self,dataset):
        #define path
        if dataset == 'qm9':
            file =  os.path.join(self.base_dir,'MolTransformer/model/data/qm9/test/test_qm9.csv')
        elif dataset == 'ocelot':
            file = os.path.join(self.base_dir,'MolTransformer/model/data/ocelot/test/test_ocelot_w_lowf.csv') 
        else:
            message = "you are not entering a valid dataset name but a file path, please make sure it is a valid path to csv file that contains 'SEILES'"
            print(message)
            logging.info(message)
            file = dataset
        SELFIES_list = self._load_selfies_from_file(file)
        random_selfies = choice(SELFIES_list)
        random_initial_smile_list = self.selfies_2_smile([random_selfies])
        return random_initial_smile_list[0]
    
    def _load_selfies_from_file(self,file):
        df = pd.read_csv(file)
        SELFIES_list = np.asanyarray(df.SELFIES).tolist()
        return SELFIES_list
    
    def smile_2_selfies(self, smiles):
        if isinstance(smiles, str):
            # Single SMILES string
            return self._process_smile(smiles)
        elif isinstance(smiles, list):
            # List of SMILES strings
            return [self._process_smile(smile) for smile in smiles if smile]
        else:
            raise ValueError("Input must be a SMILES string or a list of SMILES strings.")

    def _process_smile(self, smile):
        try:
            mol = Chem.MolFromSmiles(smile)
            if mol is None:
                print('Error in translation for:', smile)
                return ''
            hmol = Chem.AddHs(mol)
            k_smile = Chem.MolToSmiles(hmol, kekuleSmiles=True)
            return sf.encoder(k_smile)
        except Exception as e:
            print('Error processing', smile, ":", str(e))
            return ''
    
    def smiles_2_latent_space(self, smiles):
        # Check if the input is a string, and if so, convert it to a list
        if isinstance(smiles, str):
            smiles = [smiles]
        
        # Generate SELFIES from each SMILES string
        selfies_list = [self.smile_2_selfies(smile) for smile in smiles]

        # Convert SELFIES to latent space
        latent_spaces = self.selfies_2_latent_space(selfies_list)
        return latent_spaces

    def selfies_2_latent_space(self, selfies):
        print('selfies ', selfies)
        # If the input is a single SELFIES string, convert it to a list
        if isinstance(selfies, str):
            selfies = [selfies]

        latent_spaces = []
        for selfies_str in selfies:
            inputs = [self.Index.char2ind.get(char, self.Index.char2ind['[nop]']) for char in sf.split_selfies(selfies_str)]
            seq_len = min(len(inputs), settings.max_sequence_length)

            # Initialize tensor clearly on the correct device
            inputs_padd = torch.zeros((1, settings.max_sequence_length + 1), dtype=torch.long, device=self.device)
            inputs_padd[0, 0] = self.Index.char2ind['G']
            inputs_padd[0, 1:seq_len + 1] = torch.tensor(inputs[:seq_len], device=self.device)

            with torch.no_grad():
                # Ensure model is also explicitly on the correct device
                self.model.to(self.device)

                if self.gpu_mode:
                    memory = self.model.module.encoder(inputs_padd)
                else:
                    memory = self.model.encoder(inputs_padd)

                memory_ = memory.permute(1, 0, 2).cpu().numpy()
                latent_spaces.append(memory_)

        # Concatenate all the latent space representations into a single numpy array
        return np.concatenate(latent_spaces, axis=0)

    def _generate_normalized_vectors(self,dimensions, num_vectors):
        # Generate random vectors with values in the range [-1, 1]
        random_vectors = np.random.uniform(low=-1, high=1, size=(num_vectors, dimensions))
        
        # Normalize each vector to have a total length of 1
        norms = np.linalg.norm(random_vectors, ord=2,axis=1, keepdims=True)
        normalized_vectors = random_vectors / norms
        
        return normalized_vectors
    
    def sort_pareto_frontier(self, generated_results,alpha = 0.5, k = 30, mols_per_image = 30 , molsPerRow = 5 , save=False,  folder_path='',sa_threshold=''):
        if isinstance(generated_results, dict):
            df = pd.DataFrame.from_dict(generated_results)
        elif not isinstance(generated_results, pd.DataFrame):
            raise TypeError("generated_results must be a dict or a pandas DataFrame")
        else:
            df = generated_results
        updated_df = self._calculate_memory_l2(df)  # Add 'MemoryDistance'
        updated_df = calculate_tanimoto_similarity(updated_df)  # Add 'TanimotoSimilarity'

        initial_molecule_row = updated_df.iloc[0].copy()
        updated_df_without_initial = updated_df.iloc[1:].copy()


        # Adjusting the calculation for 'TanimotoSimilarity' max to ensure it is more than 0
        memory_distance_sorted = updated_df_without_initial['MemoryDistance'].sort_values(ascending=True)
        memory_distance_min = memory_distance_sorted.iloc[0]
        idx = 0
        while memory_distance_min <= 0 and idx < len(memory_distance_sorted) - 1:
            idx += 1
            memory_distance_min = memory_distance_sorted.iloc[idx]

        print('------------------------memory_distance_min------------------', memory_distance_min)

        
        memory_distance_max = updated_df_without_initial['MemoryDistance'].max()

        tanimoto_similarity_sorted = updated_df_without_initial['TanimotoSimilarity'].sort_values(ascending=False)
        tanimoto_similarity_max = tanimoto_similarity_sorted.iloc[0]
        print('------------------------tanimoto_similarity_max------------------', tanimoto_similarity_max)

        # Calculate the minimum Tanimoto similarity
        tanimoto_similarity_min = updated_df_without_initial['TanimotoSimilarity'].min()

        # Normalize distances
        updated_df_without_initial['distances_norm'] = (updated_df_without_initial['MemoryDistance'] - memory_distance_min) / (memory_distance_max - memory_distance_min)

        # Normalize similarities
        updated_df_without_initial['similarities_norm'] = (updated_df_without_initial['TanimotoSimilarity'] - tanimoto_similarity_min) / (tanimoto_similarity_max - tanimoto_similarity_min)

        # Calculate inverted distances
        updated_df_without_initial['distances_inverted'] = 1 - updated_df_without_initial['distances_norm']

        # Calculate values for Pareto frontier
        updated_df_without_initial['pareto_frontier'] = alpha * updated_df_without_initial['distances_inverted'] + (1 - alpha) * updated_df_without_initial['similarities_norm']

        # Sort the rest of the DataFrame based on 'pareto_frontier'
        sorted_rest_of_df = updated_df_without_initial.sort_values(by='pareto_frontier', ascending=False)
        
        # Prepend the initial molecule row to the sorted DataFrame
        sorted_df_with_initial = pd.concat([pd.DataFrame([initial_molecule_row]), sorted_rest_of_df], ignore_index=True)

        df_filtered = filter_duplicates(sorted_df_with_initial)

        columns_to_save = ['dimension', 'direction', 'radius', 'SMILES', 'SELFIES', 'TanimotoSimilarity', 'MemoryDistance', 'distances_norm', 'similarities_norm', 'distances_inverted', 'pareto_frontier','pubchem']
        if isinstance(sa_threshold, (float,int)):
            count_select = [0]
            sa_values_list = [sa_score(df_filtered['SMILES'].iloc[0])]
            i = 1
            max_num = len(df_filtered)
            while len(count_select) < k + 1 and i < max_num:
                sa_score_value = sa_score(df_filtered['SMILES'].iloc[i])
                if sa_score_value is not None and sa_score_value <= float(sa_threshold):
                    count_select.append(i)
                    sa_values_list.append(sa_score_value)
                i += 1
                
            top_k_1 = df_filtered.iloc[count_select].copy()
            top_k_1['SA_score'] = sa_values_list
            if 'SA_score' not in columns_to_save:
                columns_to_save.append('SA_score')
        else:
            top_k_1 = df_filtered[:min(k+1, len(df_filtered))]

        # check pubchem_api
        top_k_1 = validate_smiles_in_pubchem(top_k_1)

        if save:
            file_name = f"{alpha}_pareto_frontier_top_{k}.csv"
            top_k_1.to_csv(os.path.join(folder_path, file_name), index=False, columns=[col for col in columns_to_save if col in top_k_1.columns])
            if 'SMILES' in top_k_1.columns:
                smiles_list = top_k_1['SMILES'].tolist()
                draw_all_structures(smiles_list, out_dir = folder_path, mols_per_image = mols_per_image, molsPerRow = molsPerRow, name_tag = '', file_prefix=str(alpha) + '_Pareto_Frontier_')
        return top_k_1
    
    def _calculate_memory_l2(self, df):
        # Convert each SELFIES in the dataframe to its latent space representation
        df['Memory'] = df['SELFIES'].apply(lambda selfies: self.selfies_2_latent_space([selfies])[0])
        # Get the memory of the first molecule in the dataframe
        first_memory = df.iloc[0]['Memory']
        # Calculate the L2 distance between the memory of each molecule and the first molecule
        df['MemoryDistance'] = df['Memory'].apply(lambda mem: np.linalg.norm(first_memory - mem))
        return df
    
    def _smile_2_property_model_input(self, smile_list):
        # Ensure the input is in list format
        if isinstance(smile_list, str):
            smile_list = [smile_list]

        # Initialize the container for the input indices
        inputs_idx = []
        # Compute descriptors for the smile list
        model_input = {'descriptors': self._compute_descriptor(smile_list)}

        # Convert each smile in the list to its corresponding input tensor
        for smile in smile_list:
            selfies = self.smile_2_selfies(smile)
            if not selfies:  # Handle possible empty returns from smile_2_selfies
                continue
            # Vectorize the SELFIES and handle sequence length limits
            vectorized_seqs = self._vectorize_sequence(selfies)
            seq_len = min(len(vectorized_seqs), settings.max_sequence_length)
            
            # Prepare padded input tensor
            inputs_padd = torch.zeros((1, settings.max_sequence_length + 1), dtype=torch.long)
            inputs_padd[0, 0] = self.Index.char2ind['G']  # Start token
            inputs_padd[0, 1:seq_len + 1] = torch.LongTensor(vectorized_seqs[:seq_len])

            inputs_idx.append(inputs_padd[0])
        
        # Stack all input tensors and add to model_input dictionary
        if inputs_idx:  # Ensure there is at least one input to stack
            model_input['input'] = torch.stack(inputs_idx, dim=0)
        return model_input

    def _vectorize_sequence(self, selfies):
        # Convert a selfies string to a sequence of indices, handling unknown characters
        return [self.Index.char2ind.get(char, self.Index.char2ind['[nop]']) for char in sf.split_selfies(selfies)]

    def _compute_descriptor(self,smile_list):
        descriptors_list = []
        for smi in smile_list:
            molecule = Chem.MolFromSmiles(smi)
            descriptors = torch.tensor(molecule_descriptors(molecule))   
            descriptors_list.append(descriptors)
        descriptors_tensor = torch.stack(descriptors_list, dim=0)
        return descriptors_tensor
    
    
        
    
    

    
    



    

        
     
          
          

