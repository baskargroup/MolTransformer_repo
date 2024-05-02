from ..model import BuildModel
from .generative_utils import *
from . import settings 
import torch  # type: ignore
import logging 

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
config = Config('config.json')

class GenerateMolecules():
    def __init__(self):
        build_model_instance = BuildModel(device=device,gpu_mode = config["gpu_mode"])
        self.model = build_model_instance.model

        self.iteration_num = 0

    def GlobalMolecularGeneration(self, n_samples, sample_type='normal'):
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

        # bounding_box 
        current_dir = os.path.dirname(__file__)
        LS_statistic_path = os.path.join(current_dir, 'LS_statistic')

        n_dimensions = 12030
        np.random.seed(42)  # For reproducible random values
        bounding_box = np.ones((n_dimensions, 2))
        bounding_box[:, 0] = np.load(LS_statistic_path + 'overall_max_vals.npy').squeeze()
        bounding_box[:, 1] = np.load(LS_statistic_path + 'overall_min_vals.npy').squeeze()

        n_components = bounding_box.shape[0]  # Determine the number of latent space components from the bounding box
        mean_matrix = np.load(LS_statistic_path + 'overall_mean_vals.npy').squeeze()  # Example mean, adjust as necessary
        std_dev_matrix = np.load(LS_statistic_path + 'overall_std_vals.npy').squeeze() # Example std dev, adjust as necessary
    
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

            smiles_list_, selfies_list_ = self._adjusted_ls_2_smiles(vector)

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
        csv_file_path = self.Args.report_save_path + 'GlobalGeneratedMolecules.csv'
        df = pd.DataFrame({'SMILES': unique_smiles_list, 'SELFIES': unique_selfies_list})
        #df = validate_smiles_in_pubchem(df)  # Ensure this function is defined
        df.to_csv(csv_file_path, index=False)

        diversity_score = Tanimoto_diversity(unique_smiles_list)
        csv_file_path = self.Args.report_save_path + 'Analysis.csv'
        df_Analysis = pd.DataFrame({'uniqueness_ratio': [uniqueness_ratio], 'Tanimoto_diversity': [diversity_score]})
        #df = validate_smiles_in_pubchem(df)  # Ensure this function is defined
        df_Analysis.to_csv(csv_file_path, index=False)

        #plotting
        # test
        df['rdk_mol'] = unique_rdk_mol_list
        plot_generative_molecules_analysis(df,save_file = self.Args.report_save_path )



    

        
     
          
          

