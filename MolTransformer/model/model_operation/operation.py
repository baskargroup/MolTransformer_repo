# Import everything that's included in the __init__.py of the utils package
from tkinter import Variable
from ..utils import *
from ..model_architecture import *
from .build_model import BuildModel
import torch # type: ignore
import torch.nn as nn # type: ignore 
import os
import torch.nn.functional as F # type: ignore 
import torch.optim as optim # type: ignore 
import matplotlib.pyplot as plt # type: ignore 
from sklearn.metrics import r2_score # type: ignore
from sklearn.metrics import mean_absolute_error # type: ignore
from scipy.stats import kurtosis # type: ignore
import numpy as np
import logging 
import torch.distributed as dist # type: ignore 
import torch.multiprocessing as mp # type: ignore  
from torch.distributed import barrier # type: ignore 
import selfies as sf # type: ignore
from rdkit import Chem # type: ignore
from . import settings 
import glob
# add mutiF for HF: to compare with Chad's group paper and QM9
# add lfHF for HF: for compare and prove the low f is well embeeding in the model
# for rl: train decoder only: 128 -> memory -> embedding
# anaysis the symbol set, embedding size and architecture of Transformer  ok

# clear what in_analysis do?

global_config = Config()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ModelOperator():
    def __init__(self):
        self.device = device
        # CHECK PATH 
        check_path(global_config['report_save_path'])
        check_path(global_config['model_save_folder'] )
        logging.info(global_config['model_mode'])

        #set DDP
        if global_config["gpu_mode"]:
            print('gpu mode')
            if not torch.distributed.is_initialized():
                self.gpu_world_size = init_distributed_mode()
                print('self.gpu_world_size: ', self.gpu_world_size)

        build_model_instance = BuildModel(device=device,model_mode=global_config['model_mode'],
            gpu_mode = global_config["gpu_mode"] ,train = True, 
            preload_model=global_config['pretrain_model_type'], pretrain_model_file=global_config['pretrain_model_file'],dataset = global_config['dataset'])
        self.model = build_model_instance.model
        
        ########### define data loader stuff 
        self.generate_data(user_data = global_config['user_data'],dataset = global_config['dataset'])

        self.loss = Loss(batch_size = global_config['batch_size'])
        self.MSE = nn.MSELoss()
        self.MAE = nn.L1Loss()
        
        #loss trace 
        self._init_loss()
        self.model.to(self.device)


    
    def _init_loss(self):
        self.loss_trace = []
        self.test_loss_trace = []
        self.current_best = 1e8
        self.current_best = 1e8
        self.r2 = -100
    
    

    def _load_model(self,path = ''):
        model_path = path if path else global_config['pretrain_model_file']
        if global_config["gpu_mode"]:
            self.model.load_state_dict(torch.load(model_path))
            print('load the gpu model!! ')
        else:
            self.model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
            print('load the cpu model!! ')

    
    def _check_trainable_layer(self):
        if 'SS' in global_config["train_only_lock_layer"]:
            self.lock_Transformer = True
        else:
            self.lock_Transformer = False

        if 'fc1fc2' in global_config["train_only_lock_layer"]:
            self.lock_FirstTop = True
        else:
            self.lock_FirstTop = False


    def train(self,epochs = None,lr = 0.0001):
        print('start training')
        self.count_stop = 0
        self.model.train()
        if global_config['model_mode'] != 'Descriptors':
            self._check_trainable_layer()
            self._freelock_layers()
        if not epochs:
            epochs = global_config['train_only_train_epochs']
        print("global_config['model_mode'] ",global_config['model_mode'])
        if global_config['model_mode'] == 'SS':
            self._train_SS(epochs,lr)
        elif global_config['model_mode'] == 'SS_HF':
            self._train_multi_task(epochs,lr)
        elif global_config['model_mode'] in ['HF','multiF_HF'] :
            self._train_highf(epochs,lr)
        elif global_config['model_mode'] == 'Descriptors':
            self._train_descriptors(epochs,lr) 
        else:
            raise ValueError('please choose a valid model_mode for training')
        
           
    def generate_data(self,data_set_id = 0,user_data = False,dataset = ''):

        #model_mode,data_path,high_fidelity_label,save_path  
        if not user_data:
            if global_config['model_mode'] != 'SS' and not dataset:
                dataset = 'qm9'
            elif global_config['model_mode'] == 'SS':
                dataset = 'SS'
            if dataset not in ['qm9', 'ocelot','SS']:
                raise ValueError("Invalid dataset specified. Please choose either 'qm9' , 'ocelot' or 'SS'.")
            label = 'lumo' if dataset == 'qm9' else 'aea'
            # Path to the 'data' directory relative to the current script
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            base_train_path = os.path.join(base_dir, 'model','data', dataset, 'train')
            base_test_path = os.path.join(base_dir,  'model','data', dataset, 'test')

            # List all CSV files in the training and testing directories
            train_path = glob.glob(os.path.join(base_train_path, '*.csv'))
            test_path = glob.glob(os.path.join(base_test_path, '*.csv'))

        else:
            train_path = global_config['data_path']['train'][data_set_id] 
            test_path = global_config['data_path']['test'][data_set_id] 
            label = global_config['high_fidelity']
        
        data_path = {'train':train_path,'test':test_path}

        Data = DataProcess(model_mode = global_config['model_mode'] ,data_path = data_path,high_fidelity_label =label ,report_save_path = global_config['report_save_path'])
        logging.info("********train size :  " + str(len(Data.dataset_train)) + " ***************")
        logging.info("********test size :  " + str(len(Data.dataset_test)) + " ***************")
        if global_config["gpu_mode"]:
            self.train_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_train)
            self.dataloader_train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=global_config['batch_size'],sampler=self.train_sampler ,num_workers=self.gpu_world_size,pin_memory=True)        
            self.test_sampler = torch.utils.data.distributed.DistributedSampler(Data.dataset_test, shuffle=False)
            self.dataloader_test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=global_config['batch_size'], sampler=self.test_sampler, num_workers=self.gpu_world_size, pin_memory=True)
            
        else:
            self.dataloader_train =  torch.utils.data.DataLoader(Data.dataset_train,batch_size=global_config['batch_size'],pin_memory=True) 
            self.dataloader_test = torch.utils.data.DataLoader(Data.dataset_test, batch_size=global_config['batch_size'],  pin_memory=True)
                          
        if global_config['model_mode'] != 'SS':
            self.std_parameter = Data.std_parameter   
        del Data
    #--------------------_train_SS ------------------------------------------------------------        
    def _train_SS(self,epochs, lr):    
        logging.info("********train_teacher_forcing, with lr =  " + str(lr) + ' , epochs = ' + str(epochs))
        check_path(global_config['model_save_folder'] )
        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()) , lr = lr)
        #self.optimizer = torch.optim.SGD(filter(lambda p: p.requires_grad, self.model.parameters()) , lr = lr)
        if global_config['user_data']: 
            list_id = np.arange(len(global_config["data_path"]['train'])).tolist()
        else:
            list_id = [0]
        for epoch in range(0, epochs):
            print('#####################epoch:',epoch)
            logging.info("#####################epoch: " + str(epoch))
            for data_id in list_id:
                if data_id != 0:
                    self.generate_data(data_id)
                if global_config['gpu_mode']:
                    barrier()
                logging.info("----------- for data set folder list number:" + str(data_id))
                for n_iter in range(settings.loop_number_in_one_epoch):
                    self.model.train()
                    current_loss = 0
                    logging.info("----------- for data set :" + str(data_id) + '--- n_iter:  '+ str(n_iter))
                    print("----------- for data set :" + str(data_id) + '--- n_iter:  '+ str(n_iter))
                    if global_config['gpu_mode']:
                        self.train_sampler.set_epoch(n_iter)
                    for iteration, batch in enumerate(self.dataloader_train):
                        if global_config['gpu_mode']:
                            current_device = torch.cuda.current_device()
                        else:
                            current_device = self.device
                        print('#####################iteration:',iteration)
                        batch = self._batch2tensor(batch)
                        input_idx = batch['input']
                        target = batch['target']
                        ######     Forward pass  ######
                        logp = self.model(input_idx.to(current_device)) #
                        loss = self.loss.TF_logp_loss(logp,target,device = current_device) 
                        self.optimizer.zero_grad()
                        loss.backward()
                        self.optimizer.step()
                        current_loss += loss.item()
                        print('loss: ',loss.item() )
                        if iteration %  settings.test_training_num_iteration == 1:
                            logging.info("epoch: " + str(epoch))
                            logging.info("-----------" + str(iteration))
                            logging.info("loss:      " + str(loss.item()))
                            self.model.eval()
                            test_loss_ = self._test_SS(num_batch = settings.testing_num_batch_SS_in_iteration)

                    self.loss_trace.append(current_loss/(iteration))
                    self.model.eval()
                    test_loss_ = self._test_SS(num_batch = settings.testing_num_batch_SS_in_epoch)
                    self.test_loss_trace.append(test_loss_)
                self.save_model(str(epoch)+'_'+str(data_id)) 
                torch.cuda.empty_cache()
                self._plot_loss()
                self.evaluate_decoder(num_batch = settings.testing_num_batch_evaluate_decoder) #check

            self.save_model(str(epoch))
            self._save_best(test_loss_)
            self._plot_loss()
            torch.cuda.empty_cache()    # Clear the GPU memory
        
    def _test_SS(self,num_batch = 10):
        logging.info("start testing")
        current_loss = 0
        with torch.no_grad():
            for iteration, batch in enumerate(self.dataloader_test):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                target = batch['target']
                ######     Forward pass  ######
                logp = self.model(input_idx.to(current_device)) #
                loss = self.loss.TF_logp_loss(logp,target,device = current_device) 
                current_loss += loss.item()
                if iteration == num_batch-1:
                    break
            loss =  current_loss/(iteration)
            logging.info("test loss:      " + str(loss))
        torch.cuda.empty_cache()
        return loss
    #--------------------------- SS_HF ----------------------------------------------
    def _train_multi_task(self,epochs, lr):
        logging.info("********transformer_dnn, with lr =  " + str(lr) + ' , epochs = ' + str(epochs))
        check_path(global_config['model_save_folder'] )

        self.optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()) , lr = lr)

        for epoch in range(0, epochs):
            print('#####################epoch:',epoch)
            logging.info("#####################epoch: " + str(epoch))
            
            if global_config['gpu_mode']:
                self.train_sampler.set_epoch(epoch)
            
            for iteration, batch in enumerate(self.dataloader_train):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                self.model.train()    
                current_loss = 0
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                target = batch['target']
                ######     Forward pass  ######
                target_regression = batch['high_f']
                predictedProperties, logp ,regularization = self.model(input_idx.to(current_device))    
                loss_tf = self.loss.TF_logp_loss(logp,target,device = current_device) 
                loss_lf = self.MSE(predictedProperties, target_regression)
                loss = loss_tf + loss_lf + regularization

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss += loss.item()
                print('loss: ', loss.item())
                if iteration %  settings.log_training_num_iteration == 1:
                    logging.info("epoch: " + str(epoch))
                    logging.info("-----------" + str(iteration))
                    logging.info("loss:      " + str(loss.item()))
                    logging.info("loss_tf:      " + str(loss_tf.item()))
                    logging.info("loss_lf:      " + str(loss_lf.item()))
                    logging.info("loss_regu:      " + str(regularization.item()))

            self.loss_trace.append(current_loss/(iteration))
            self.model.eval()
            test_loss_,test_lf_loss = self._test_multi_task(num_batch = settings.testing_num_batch_multi_task)
            self.test_loss_trace.append(test_loss_)
            r2 = self.r_square(data_set = 'test',num_batch = settings.testing_num_batch_multi_task,intrain = True)
            if global_config['gpu_mode']:
                barrier()
            self._save_best_r2(r2)
            if global_config['gpu_mode']:
                barrier()
            self._save_best(test_lf_loss)
            self._plot_loss()
            torch.cuda.empty_cache() # Clear the GPU memory
            if global_config['gpu_mode']:
                barrier()
        
        
    def _test_multi_task(self,num_batch = 10):
        logging.info("start testing")
        current_loss = 0
        current_lf_loss = 0
        current_tf_loss = 0
        with torch.no_grad():
            for iteration, batch in enumerate(self.dataloader_test):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                target = batch['target']
                ######     Forward pass  ######
                target_regression = batch['high_f']
                predictedProperties, logp ,regularization = self.model(input_idx.to(current_device)) 
                loss_tf = self.loss.TF_logp_loss(logp,target,device = current_device) 
                loss_lf = self.MSE(predictedProperties, target_regression)
                loss = loss_tf + loss_lf + regularization

                current_loss += loss.item()
                current_lf_loss += loss_lf.item()
                current_tf_loss += loss_tf.item()
                if iteration == num_batch-1:
                    break
            loss =  current_loss/(iteration)
            current_lf_loss = current_lf_loss /(iteration)
            current_tf_loss = current_tf_loss/(iteration)
            logging.info("test loss:      " + str(loss))
            logging.info("test property loss:      " + str(current_lf_loss))
            logging.info("test tf loss:      " + str(current_tf_loss))
        torch.cuda.empty_cache()
        return loss,current_lf_loss
    
    #--------------------train high fidelity--------------------------------------------------------
    def _train_highf(self,epochs, lr):
        logging.info("********train_low_f, with lr =  " + str(lr) + ' , epochs = ' + str(epochs))
        check_path(global_config['model_save_folder'] )

        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = lr)

        for epoch in range(0, epochs):
            logging.info("#####################epoch: " + str(epoch))
            print('#####################epoch:',epoch)
            if global_config['gpu_mode']:
                self.train_sampler.set_epoch(epoch)
            
            current_loss = 0
            for iteration, batch in enumerate(self.dataloader_train):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                self.model.train()
                batch = self._batch2tensor(batch)
                input_idx = batch['input']

                ######     Forward pass  ######
                if global_config['model_mode'] == 'multiF_HF':
                    descriptors = batch['descriptors']
                    predicted_high_F,regularization = self.model(input_idx.to(current_device),descriptors.to(current_device))
                else:
                    predicted_high_F,regularization = self.model(input_idx.to(current_device))
                loss = self.MSE(predicted_high_F, batch['high_f'])
                loss += regularization
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss += loss.item()
                print('loss: ',loss.item() )
                if iteration %  settings.log_training_num_iteration == 1:
                    logging.info("epoch: " + str(epoch))
                    logging.info("-----------" + str(iteration))
                    logging.info("loss:      " + str(loss.item()))
                    logging.info("predicted: " + str(predicted_high_F[0]))
                    logging.info("target:    " + str(batch['high_f'][0]))

            if iteration != 0:
                self.loss_trace.append(current_loss/iteration)
            else:
                self.loss_trace.append(current_loss)
            self.model.eval()
            test_loss_ = self._test_highf(num_batch = settings.testing_num_batch_HF)
            self.test_loss_trace.append(test_loss_)
            r2 = self.r_square(data_set = 'test',num_batch = settings.testing_num_batch_HF,intrain = True)
            logging.info("test R2:      " + str(r2))
            if global_config['gpu_mode']:
                barrier()
            self._save_best_r2(r2)
            if global_config['gpu_mode']:
                barrier()
            self._plot_loss()
            torch.cuda.empty_cache()  # Clear the GPU memory
            if global_config['gpu_mode']:
                barrier()
            if self.count_stop >= settings.early_stop_epochs:
                logging.info("early stop : not improving for last num epochs: " + str(settings.early_stop_epochs))
                break
        
    def _test_highf(self,num_batch = 10):
        logging.info("start testing")
        current_loss = 0
        with torch.no_grad():
            for iteration, batch in enumerate(self.dataloader_test):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                ######     Forward pass  ######
                if global_config['model_mode'] == 'multiF_HF':
                    descriptors = batch['descriptors']
                    predicted_high_F,regularization = self.model(input_idx.to(current_device),descriptors.to(current_device))
                else:
                    predicted_high_F,regularization = self.model(input_idx.to(current_device))
                loss = self.MSE(predicted_high_F, batch['high_f'])
                loss += regularization
                current_loss += loss.item()
                if iteration == num_batch-1:
                    break
            loss =  current_loss/(iteration)
            logging.info("test loss:      " + str(loss))
            logging.info("regularization:      " + str(regularization.item()))
        torch.cuda.empty_cache()
        return loss
    
    #-------------------------------------------------descriptors--------------------------------
    def _train_descriptors(self,epochs, lr):
        check_path(global_config['model_save_folder'] )
        self.optimizer = torch.optim.Adam(self.model.parameters() , lr = lr)
        for epoch in range(0, epochs):
            logging.info("#####################epoch: " + str(epoch))
            print('#####################epoch:',epoch)
            if global_config['gpu_mode']:
                self.train_sampler.set_epoch(epoch)
            current_loss = 0
            for iteration, batch in enumerate(self.dataloader_train):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                self.model.train()
                batch = self._batch2tensor(batch)
                descriptors = batch['descriptors']
                ######     Forward pass  ######
                predicted_high_F = self.model(descriptors.to(current_device))
                loss = self.MSE(predicted_high_F, batch['high_f'])
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                current_loss += loss.item()
                print('loss: ',loss.item() )
                logging.info("loss: " +loss.item())
                if iteration %  settings.log_training_num_iteration == 1:
                    logging.info("epoch: " + str(epoch))
                    logging.info("-----------" + str(iteration))
                    logging.info("loss:      " + str(loss.item()))
                    logging.info("target:    " + str(batch['high_f'][0]))
            if iteration != 0:
                self.loss_trace.append(current_loss/iteration)
            else:
                self.loss_trace.append(current_loss)
            self.model.eval()
            test_loss_ = self._test_descriptors(num_batch = settings.testing_num_batch_descriptor)
            self.test_loss_trace.append(test_loss_)
            r2 = self.r_square(data_set = 'test',num_batch = settings.testing_num_batch_descriptor,intrain = True)
            logging.info("test R2:      " + str(r2))
            if global_config['gpu_mode']:
                barrier()
            self._save_best_r2(r2)
            if global_config['gpu_mode']:
                barrier()
            self._plot_loss()
            torch.cuda.empty_cache()  # Clear the GPU memory
            if global_config['gpu_mode']:
                barrier()
            if self.count_stop >= settings.early_stop_epochs:
                logging.info("early stop : not improving for last epochs: "+ str(settings.early_stop_epochs))
                break
        
    def _test_descriptors(self,num_batch = 10):
        logging.info("start testing")
        current_loss = 0
        with torch.no_grad():
            for iteration, batch in enumerate(self.dataloader_test):
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                descriptors = batch['descriptors']
                ######     Forward pass  ######
                predicted_high_F = self.model(descriptors.to(current_device))    
                loss = self.MSE(predicted_high_F, batch['high_f'])
                current_loss += loss.item()
                if iteration == num_batch-1:
                    break
            loss =  current_loss/(iteration)
            logging.info("test loss:      " + str(loss))
        torch.cuda.empty_cache()
        return loss


    #--------------------analisys -----------------------------------------------------------   
    def evaluate_decoder(self,num_batch = 10):
        if not getattr(self, 'index_convert', None):
            self.index_convert = IndexConvert()
        logging.info("evaluate decoder, num_batch: " + str(num_batch))
        molecular_correct,symbol,symbol_correct,validation_num,test_size = 0,0,0,0,0
        with torch.no_grad():
            for iteration, batch in enumerate(self.dataloader_test):
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device
                ######     Forward pass  ######
                if global_config['gpu_mode']:
                    memory = self.model.module.encoder(input_idx.to(current_device)).to(current_device)
                    inference = self.model.module.decoder(memory) 
                else:
                    memory = self.model.encoder(input_idx).to(current_device)
                    inference = self.model.decoder(memory) 

                molecular_correct_,symbol_,symbol_correct_,test_size_ = self.index_convert.compare_2_idxs(batch['target'],inference) #check
                recons_selfies = self.index_convert.index_2_selfies(inference)
                smiles_list = self.index_convert.selfies_2_smile(recons_selfies)
                validation_num_ = self._validation_num(smiles_list)
                
                molecular_correct += molecular_correct_
                symbol += symbol_
                symbol_correct += symbol_correct_
                validation_num += validation_num_
                test_size += test_size_
                
                if iteration == num_batch-1:
                    break
        #validation_ratio
        (validation_ratio,molecular_acuracy,symbol_accuracy) = (validation_num/test_size,molecular_correct/test_size,symbol_correct/symbol)
        
        print("number of inference:  %i, ' validation ratio' %9.4f, molecular acuracy %9.4f,symbol accuracy %9.4f"
        %(test_size, validation_ratio,molecular_acuracy,symbol_accuracy))
        logging.info("number of inference: " + str(test_size))
        logging.info("validation ratio:    " + str(validation_ratio))
        logging.info("molecular acuracy:   " + str(molecular_acuracy))
        logging.info("symbol accuracy:     " + str(symbol_accuracy))
        return molecular_acuracy,symbol_accuracy

    def compute_memory_statistics(self, id_list=[0]):
        """
        Might move to gen methods
        
        Compute and update memory statistics for each batch in a given list of data chunks, 
        specifically focusing on the minimum, maximum, mean, and standard deviation of memory representations.

        This function processes data in batches to efficiently handle large datasets with minimal memory usage. 
        It computes running statistics (min, max, mean, std) for memory representations, enabling analysis similar to 
        a moving average approach.

        Parameters:
        - id_list: A list of integers representing the identifiers for data chunks to be processed.

        The function updates the statistics batch by batch and saves these statistics into numpy files. 
        This is especially useful for analyzing large datasets or when working with hardware with limited memory resources.

        Returns:
        - None: This function does not return a value. Instead, it saves the computed statistics to disk.
        """
        for data_id in id_list:
            self.generate_data(data_id)  # Assuming this prepares the data based on the given id
            # Initialize running statistics
            min_vals = None
            max_vals = None
            sum_memory = None
            sumsq_memory = None
            n_samples = 0

            for i, batch in enumerate(self.dataloader_train):
                print('batch:',i)
                batch = self._batch2tensor(batch)  # Convert batch to tensor format
                input_idx = batch['input']

                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device

                # Extract memory representation for the batch without training
                memory = self.model.encoder(input_idx.to(current_device)).detach() if not global_config["gpu_mode"] else self.model.module.encoder(input_idx.to(current_device)).detach()
                memory = memory.permute(1, 0, 2).reshape(memory.size(1), -1).to(current_device)
                
                # Update running sums for mean and std dev calculation
                if sum_memory is None:
                    sum_memory = torch.zeros_like(memory.sum(dim=0))
                    sumsq_memory = torch.zeros_like((memory ** 2).sum(dim=0))
                sum_memory += memory.sum(dim=0)
                sumsq_memory += (memory ** 2).sum(dim=0)
                n_samples += memory.size(0)
                
                # Update min and max values with each batch
                if min_vals is None:
                    min_vals = memory.min(dim=0)[0]  # Initial min values for each dimension
                    max_vals = memory.max(dim=0)[0]  # Initial max values for each dimension
                else:
                    min_vals = torch.min(min_vals, memory.min(dim=0)[0])
                    max_vals = torch.max(max_vals, memory.max(dim=0)[0])

            # Finalize statistics
            mean_vals = sum_memory / n_samples
            std_vals = torch.sqrt((sumsq_memory / n_samples) - (mean_vals ** 2))

            # Convert statistics to numpy and reshape for saving
            min_vals = min_vals.numpy().reshape(-1, 1)
            max_vals = max_vals.numpy().reshape(-1, 1)
            mean_vals = mean_vals.numpy().reshape(-1, 1)
            std_vals = std_vals.numpy().reshape(-1, 1)

            # Save statistics to files
            np.save(global_config['report_save_path'] + str(data_id) + '_min_vals.npy', min_vals)
            np.save(global_config['report_save_path'] + str(data_id) + '_max_vals.npy', max_vals)
            np.save(global_config['report_save_path'] + str(data_id) + '_mean_vals.npy', mean_vals)
            np.save(global_config['report_save_path'] + str(data_id) + '_std_vals.npy', std_vals)


    def r_square(self, data_set = 'test',num_batch = 10,intrain = False):
        if global_config['model_mode'] ==  'SS': 
            return 0
        check_path(global_config['report_save_path'])
        if data_set ==  'test':
            data = self.dataloader_test
        else:
            data = self.dataloader_train

        ground_true = torch.empty((1,1),device = self.device)
        predicted = torch.empty((1,1),device = self.device)
        num = 1

        with torch.no_grad():
            for iteration, batch in enumerate(data):
                batch = self._batch2tensor(batch)
                input_idx = batch['input']
                if global_config['gpu_mode']:
                    current_device = torch.cuda.current_device()
                else:
                    current_device = self.device

                if global_config['model_mode'] == 'Descriptors':
                    descriptors = batch['descriptors']
                    predictedProperties = self.model(descriptors.to(current_device))
                elif  global_config['model_mode'] == 'SS_HF':
                    predictedProperties,_,_ = self.model(input_idx.to(current_device))
                elif global_config['model_mode'] == 'multiF_HF':
                    descriptors = batch['descriptors']
                    predictedProperties,_ = self.model(input_idx.to(current_device),descriptors.to(current_device))
                elif global_config['model_mode'] == 'HF':
                    predictedProperties,_ = self.model(input_idx.to(current_device))
                else:
                    raise ValueError('Not a valid model_mode')
                ground_true = torch.cat((ground_true, batch['high_f']), 0)
                predicted = torch.cat((predicted, predictedProperties), 0)
                if iteration == num_batch-1:
                    break
        if global_config['gpu_mode']:
            barrier()
            target  = all_gather_2d(ground_true)
            predict  = all_gather_2d(predicted)
        else:
            target = ground_true
            predict = predicted


        label = global_config['high_fidelity']
        
        r_total = 0 
        r_list = []
        target_ = target[1:,0]
        predict_ = predict[1:,0]
        r2 = r2_score(target_, predict_)
        if intrain:
            return r2
        else:
            target_recover = self.recover_standardized_data(target_, label)
            predict_recover = self.recover_standardized_data(predict_, label)
            # Calculate min/mean/max
            min_target = target_recover.min().item()
            mean_target = target_recover.mean().item()
            max_target = target_recover.max().item()
            # Log the min/mean/max
            logging.info(f'After recovery - Target: Min = {min_target}, Mean = {mean_target}, Max = {max_target}')
            #logging.info(f'After recovery - Predict: Min = {min_predict}, Mean = {mean_predict}, Max = {max_predict}')
            r2_recover  = r2_score(target_recover, predict_recover)
            self.plot_r2(target_recover, predict_recover,r2_recover,property_name = 'recover_'+label + '_' + data_set)   
            self.plot_r2(target_, predict_,r2,property_name =  label + '_' + data_set)
            self.plot_error_histogram(target_recover, predict_recover, 'recover_'+ label + '_' + data_set)
            self.plot_value_histogram(target_recover, predict_recover,'recover_'+ label + '_' + data_set)
            logging.info('For ' + label+ ' , R2 = ' + str(r2))
            mae = mean_absolute_error(target_recover, predict_recover)
            return r2_recover,mae
            
    
    def recover_standardized_data(self, data,name = ''):
        mean1 = self.std_parameter[name]['mean'] 
        std1 = self.std_parameter[name]['std'] 
        constant = self.std_parameter[name]['constant'] 
        data = data - constant
        data = data * std1 + mean1
        return data

    def plot_r2(self, target, predict, r2, property_name):
        if torch.is_tensor(predict):
            predict = predict.cpu().detach().numpy()
        if torch.is_tensor(target):
            target = target.cpu().detach().numpy()
        
        plt.figure(figsize=(5, 5))
        plt.scatter(target, predict, s=60, alpha=0.2, edgecolors="k")
        
        p_lim = max(np.max(predict), np.max(target)) + 0.5
        n_lim = min(np.min(predict), np.min(target)) - 0.5
        xseq = np.linspace(n_lim, p_lim, num=100)
        
        plt.plot(xseq, xseq, label=f"R2: {r2:.2f}", color="gray", alpha=0.5, lw=2.5)
        plt.ylabel("Predicted Label", color="saddlebrown", fontsize=14)
        plt.xlabel("Ground Truth", fontsize=14)
        plt.legend(loc='upper right')
        plt.xlim([n_lim, p_lim])
        plt.ylim([n_lim, p_lim])
        plt.savefig(global_config['report_save_path'] + '/' + property_name + '.png')
        plt.close()

    def plot_error_histogram(self, y_true, y_predict, data_set):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().detach().numpy()
        if torch.is_tensor(y_predict):
            y_predict = y_predict.cpu().detach().numpy()

        plt.figure()
        error = y_true - y_predict
        mae = np.mean(np.abs(error))
        print('----------------------mae: ', mae)
        print('shape of mae', error.shape)
        print('type of mae', type(mae))
        std = np.std(error)
        k_value = kurtosis(error, fisher=False)
        plt.hist(error, bins=20, edgecolor='black')
        plt.title('Histogram of Error (MAE = {:.2f}, STD = {:.2f}, Kurtosis = {:.2f})'.format(mae, std, k_value))
        plt.xlabel('Error (y_true - y_predict)')
        plt.ylabel('Frequency')
        plt.savefig(global_config['report_save_path'] + '/' + data_set + '_Error_histogram.png')
        plt.close()

    def plot_value_histogram(self, y_true, y_predict, data_set):
        if torch.is_tensor(y_true):
            y_true = y_true.cpu().detach().numpy()
        if torch.is_tensor(y_predict):
            y_predict = y_predict.cpu().detach().numpy()

        n_intervals = 25
        plt.figure()
        intervals = np.linspace(start=np.min(y_true), stop=np.max(y_true), num=n_intervals + 1)
        y_true_counts = [(y_true >= intervals[i]) & (y_true < intervals[i + 1]) for i in range(n_intervals)]
        y_predict_counts = [(y_predict >= intervals[i]) & (y_predict < intervals[i + 1]) for i in range(n_intervals)]

        y_true_counts = np.sum(y_true_counts, axis=1)
        y_predict_counts = np.sum(y_predict_counts, axis=1)

        plt.plot(intervals[:-1], y_true_counts, '-', label='y_true')
        plt.plot(intervals[:-1], y_predict_counts, '-', label='y_predict')
        plt.legend()
        plt.title('Histogram of Values')
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.savefig(global_config['report_save_path'] + data_set + '_Value_histogram.png')
        plt.close()

    ###### free for loack the layers
    def _freelock_layers(self):
        
        if self.lock_Transformer:
            self.lock_layers('tansformer_encoder')
            self.lock_layers('embedding')
            self.lock_layers('decoder')
        else:
            self.unlock_layers('tansformer_encoder')
            self.unlock_layers('embedding')
            self.unlock_layers('decoder')
        
        if self.lock_FirstTop:
            self.lock_layers('fc1.weight')
            self.lock_layers('fc1.bias')
            self.lock_layers('fc2.weight')
            self.lock_layers('fc2.bias')
        else:
            self.unlock_layers('fc1.weight')
            self.unlock_layers('fc1.bias')
            self.unlock_layers('fc2.weight')
            self.unlock_layers('fc2.bias')


    def lock_layers(self,layer_name):
        #lock
        for name, param in self.model.named_parameters():
            if param.requires_grad and layer_name in name:
                param.requires_grad = False
    
    def unlock_layers(self,layer_name):
        for name, param in self.model.named_parameters():
            if  not param.requires_grad and layer_name in name:
                param.requires_grad = True
                
    def check_freezed_layers(self):
        logging.info("the following layers are locked")
        for name,param in self.model.named_parameters():
            if  not param.requires_grad:
                logging.info(name)
    
        
    def _save_best(self,current_loss):
        if current_loss < self.current_best:
            self.current_best = current_loss
            torch.save(self.model.state_dict(), global_config['model_save_folder']  + global_config['model_mode'] + '_best.pt')
            logging.info('************save the best ****************************')
            self.count_stop = 0
        else:
            self.count_stop += 1

    def _save_best_r2(self,current_r2):
        if current_r2 > self.r2:
            logging.info('************save the best r2 ****************************')
            self.r2 = current_r2
            self.save_model(name = 'R2_HF_best')
            #torch.save(self.model.state_dict(), global_config['model_save_folder']   + 'R2_HF_best.pt')
            self.count_stop = 0
        else: 
            self.count_stop += 1


    def save_model(self,name = None):
        if name is not None:
            checkpoint_path = global_config['model_save_folder']  + name + ".pt"
            non_ddp_path = global_config['model_save_folder']  + name + "_noneDDP.pt"
        else:
            checkpoint_path = os.path.join(global_config['model_save_folder'] , "model.pt")
            non_ddp_path = os.path.join(global_config['model_save_folder'] , "model_noneDDP.pt")
        torch.save(self.model.state_dict(), checkpoint_path)
        if global_config['train_only_save_non_ddp_model'] and global_config['gpu_mode']:
            state_dict = self.model.module.state_dict()
            # Remove 'module.' prefix from each key
            new_state_dict = {key.replace('module.', ''): value for key, value in state_dict.items()}
            # Save the adjusted state dict
            torch.save(new_state_dict, non_ddp_path)
            #torch.save(self.model.module.state_dict(), PATH)
                
    
    def load_model(self,name = None):
        # if the epoch not provide, the best performance model will be load
        if name is None:
            checkpoint_path = global_config['model_save_folder']  + 'model.pt'
        else:
            checkpoint_path = global_config['model_save_folder']  + name + ".pt"
        self.model.load_state_dict(torch.load(checkpoint_path))
        self.model.eval()
            
            
    #--------------------plot loss -----------------------------------------------------------    
    def _plot_loss(self):
        plt.plot(range(1, len(self.loss_trace)+1), self.loss_trace, color="darkgoldenrod",label="train loss",alpha=0.7)
        plt.plot(range(1, len(self.test_loss_trace)+1), self.test_loss_trace, color="maroon",label="test loss",alpha=0.7)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc = 'upper right')
        check_path(global_config['report_save_path'])
        plt.savefig(global_config['report_save_path'] + global_config['model_mode'] + '_Loss_History' +'.png')
        plt.close()    

    #--------------------chem strings -----------------------------------------------------------
    def _validation_num(self,lists_of_smile:list):
        num = 0
        for s in lists_of_smile:
            if self._is_correct_smiles(s):
                num += 1
        return num

    def _is_correct_smiles(self,smiles):
        m = Chem.MolFromSmiles(smiles,sanitize=False)
        if m is None:
            return False
        return True
    
            
    def _batch2tensor(self,batch):
        for k, v in batch.items():
            if torch.is_tensor(v):
                batch[k] = self._to_var(v)
        return batch 
    def _to_var(self, x):
        return x.to(self.device)

    