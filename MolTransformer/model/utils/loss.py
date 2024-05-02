import torch # type: ignore
import torch.nn as nn # type: ignore
from . import settings 
from .general_utils import LoadIndex

Index = LoadIndex()

class Loss():
    def __init__(self,batch_size):
        self.ignore_index = Index.pad_indx
        self.class_weight = torch.FloatTensor(Index.class_weight)
        self.batch_size = batch_size
        self.mse = torch.nn.MSELoss()
        self.mae = torch.nn.L1Loss()
        self.max_sequence_length = settings.max_sequence_length + 1
    def BT(self,LS,lamda = 1):
        c = LS.T @ LS
        # sum the cross-correlation matrix between all gpus
        c.div_(self.batch_size)
        on_diag = torch.diagonal(c).add_(-1).pow_(2).sum()
        off_diag = self.off_diagonal(c).pow_(2).sum()
        loss = on_diag + lamda * off_diag
        #print('off diag', lamda * off_diag)
        #print('on diag', on_diag)
        return loss
    
    def off_diagonal(self,x):
        # return a flattened view of the off-diagonal elements of a square matrix
        n, m = x.shape
        assert n == m
        return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()
    
    def LS_loss(self,LS,memory,recons_memory,beta = 1,lamda = 1):
        mse_loss = self.mse(recons_memory.permute(1, 0, 2), memory.permute(1, 0, 2))
        BT_loss = beta*self.BT(LS,lamda)
        print('mse_loss',mse_loss)
        print('BT_loss',BT_loss)
        return mse_loss + BT_loss
        
    def r2_loss(self,output, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2
    
    def RMSLE(self,predicted,target):
        return torch.sqrt(self.MSE(torch.log(predicted + 1), torch.log(target + 1)))
    def weighted_mse(self, predicted, target):
        mse = self.mse(predicted, target)
        mean_mse = torch.mean(mse)
        loss = mean_mse * target.sum()
        return loss
    def TF_logp_loss(self,logp,target,device):
        NLL = nn.NLLLoss(weight = self.class_weight.to(device), reduction= 'mean', ignore_index = self.ignore_index)
        target = target[:, :self.max_sequence_length].contiguous().view(-1)
        logp = logp.view(-1, logp.size(2))
        return NLL(logp,target)*1e-4