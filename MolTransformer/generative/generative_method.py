from ..model import BuildModel, IndexConvert
from .generative_utils import *
from . import settings 
import torch  # type: ignore
import logging 
import selfies as sf # type: ignore
from rdkit import Chem # type: ignore
from rdkit.Chem import inchi # type: ignore

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config('config.json')

class GenerateMethods(IndexConvert):
    """
    1. LocalMolecularGeneration: 
        It generate closet new valid molecules from normalize random ls vectors both positive and negtive directions.
    2. global_molecular_generation:
        Randomly sample n number of  LS vectors; a random samples each dimension vector. 
        ! Analysis around ls space needed.  
    3. OptimisticMoleculesRevolution:
        ! adjust the needed for definition changed of NeighboringSearch
        Iteratively use LocalMolecularGeneration, then use Multi-Fedility model to predict the High-Fidelity lables.
        Then, chose the neighbor molecule with the highest predicted lable to be the new itial molecule for next iteration.
        If predicted HF label of  new initail molecule is not larger than epislon + predicted HF label of  old initail molecule, break the loop.
        # to run the method, we need to load multi-fidelity model...and compute the low-fidelity label
    4. MolecularEvolution:
        Take LS of the moleular pair: ( start , end ), compute the vector of the LS and then take k points on the vector.
        Decode the k points, record them if they are unique molecules. 
        Plot and solve all the generative molecules. 
    5. smiles_2_latent_space
    6. latent_space_2_smiles
    7. latent_space_2_properties:
        to use the function, make sure you call .set_property_model(dataset = 'qm9') or 'ocelot'
    8. compute_uniqueness_by_inchi
        """
    def __init__(self,gpu_mode = False,report_save_path = ''):
        super().__init__()  # Initialize the base IndexConvert class
        self.device = device

        self.gpu_mode = gpu_mode
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        if not report_save_path:
            report_save_path = os.path.join(base_dir, 'output','GenerateMethods/')
        self.report_save_path = report_save_path
        check_path(self.report_save_path)

        build_model_instance = BuildModel(device=device,gpu_mode = self.gpu_mode)
        self.model = build_model_instance.model

        self.iteration_num = 0

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
        global_molecular_generation_save_path = self.report_save_path + 'GlobalMolecularGeneration/'
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
            print('components: ',i)
            print('std_dev_matrix', std_dev_matrix[i])
            print('min: ', bounding_box[i, 1])
            print('max: ',bounding_box[i, 0])
            if sample_type == 'normal':
                print('mean_matrix', mean_matrix[i])
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

            smiles_list_, selfies_list_ = self.latent_space_2_smiles(latent_space = vector)

            smiles_list += smiles_list_
            selfies_list += selfies_list_

        uniqueness_ratio, unique_index , rdk_mol = self.compute_uniqueness_by_inchi(smiles_list,return_index = True)  # Ensure this function is defined
        logging.info(f"iteration_num: " + str(self.iteration_num))
        logging.info(f"uniqueness_ratio: " + str(uniqueness_ratio))
        print(f"iteration_num: " + str(self.iteration_num))
        print(f"uniqueness_ratio: " + str(uniqueness_ratio))

        unique_smiles_list = [smiles_list[i] for i in unique_index]
        unique_selfies_list = [selfies_list[i] for i in unique_index]
        unique_rdk_mol_list = [rdk_mol[i] for i in unique_index]

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
    
    def latent_space_2_smiles(self,latent_space):
        # Check if the numpy array has the correct number of elements
        # expect shape: (num,401,30)
        if int(latent_space.size % 12030) != 0 :
            raise ValueError(f"Expected numpy array of size 12030, got {latent_space.size}")
        
        # Reshape the numpy array and convert it to a torch tensor
        torch_tensor = torch.from_numpy(latent_space.reshape(-1, 401, 30)).to(self.device)
        
        smiles,selfies = self._memory_2_smiles(torch_tensor)
        return smiles,selfies
    
    def _memory_2_smiles(self,memory_torch):

        # if the shape of memory_torch is (1, 401, 30) 
        # need to check whether pca version still work
        if memory_torch.shape[1] == 401:
            memory = memory_torch.permute(1, 0, 2)
        else:
            memory = memory_torch
        if self.gpu_mode:
            molecule = self.model.module.decoder(memory)
        else:
            molecule = self.model.decoder(memory)

        smiles,selfies = self._idx_2_smiles(molecules_idx = molecule)
        return smiles,selfies
    
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

    
    



    

        
     
          
          

