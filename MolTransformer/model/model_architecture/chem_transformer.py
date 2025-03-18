import torch # type: ignore 
import torch.nn as nn # type: ignore
from .positional_encoding import PositionalEncoding
from . import settings 
import selfies as sf # type: ignore
from ..utils import LoadIndex

Index = LoadIndex()

class ChemTransformer(nn.Module):

    """
    The ChemTransformer is a neural network model for chemical compound representation
    that uses a transformer architecture. It is designed for tasks such as molecule
    generation or property prediction. The model uses positional encoding and separate
    encoder and decoder transformer components.
    
    Attributes:
        
        device (torch.device): The device on which the model will run.
        mode (str): The mode of operation (e.g., 'Na', 'SS', 'multiF_HF').
    """
    def __init__(self,device = '',model_mode = 'SS',train = False,gpu = True): 
        """
        Initialize the ChemTransformer with given arguments, index, device, and mode.
        
        Parameters:
            Index (object): An object containing vocabulary and index mappings.
            device (torch.device): The device on which the model will run.
            mode (str): The mode of operation (e.g., 'Na', 'SS', 'multiF_HF').
        """
        super().__init__()
        # idx for decod back to selfies and smile
        # this will need the adjustment of the operation
        if not device:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        if train:
            self.regularization = True
        else:
            self.regularization  = False
        if gpu:
            self.gpu = True
        else:
            self.gpu = False
        self.sos_idx, self.eos_idx, self.pad_idx = Index.sos_indx, Index.eos_indx, Index.pad_indx
        self.mode = model_mode
        print('self.mode', self.mode)
        
        #test git hub
        # Use settings from settings.py
        self.embedding_size = settings.embedding_size
        self.num_layers = settings.num_layers
        self.hidden_size = settings.hidden_size
        self.num_head = settings.num_head
        self.word_dropout = settings.word_dropout
        self.max_sequence_length = settings.max_sequence_length + 1

        
        
        # for chem and data
        self.vocab_size = Index.vocab_size
        self.char2ind = Index.char2ind
        self.ind2char = Index.ind2char
        
        # definition of layers and embedding
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)#batch, max_length,embedding
        self.position_encoding = PositionalEncoding(self.embedding_size,max_len = self.max_sequence_length,dropout = self.word_dropout)
        
        # encoder/decoder layers
        enc_layer = nn.TransformerEncoderLayer(self.embedding_size, self.num_head, self.hidden_size, self.word_dropout)
        dec_layer = nn.TransformerDecoderLayer(self.embedding_size, self.num_head, self.hidden_size, self.word_dropout)
        self.tansformer_encoder = nn.TransformerEncoder(enc_layer, num_layers = self.num_layers)
        self.tansformer_decoder = nn.TransformerDecoder(dec_layer, num_layers = self.num_layers)
        # final dense layer
        # output of the decoder to be the embedding of the sybols. note output of transformer shape: max_length, batch, embedding
        #                                                         embedding shape: #batch, max_length,embedding
        # the embedding of the symbols to the index 
        self.outputs2vocab = nn.Linear(self.embedding_size, self.vocab_size)
        self.log_softmax = nn.LogSoftmax()

        #for high_fidelity
        self.l1_crit = nn.L1Loss()
        
    def encoder(self,input_idx):
        """
        Encode the input indices using the transformer encoder.
        
        Parameters:
            input_idx (torch.Tensor): The input tensor containing indices of tokens.
            
        Returns:
            torch.Tensor: The memory output from the transformer encoder.
        """
        input_idx = input_idx.to(self.device)
        padding_mask = self.create_pad_mask(input_idx)
        naive_embedding = self.embedding(input_idx).permute(1, 0, 2)
        input_embedding = self.position_encoding(naive_embedding)
        batch_size = input_idx.size(0)
        memory = self.tansformer_encoder(input_embedding,src_key_padding_mask = padding_mask)
        return memory
    

    def teacher_forcing(self,memory,input_idx):
        """
        Apply teacher forcing during training, where the ground truth is provided
        as the next input to the decoder.

        Parameters:
            memory (torch.Tensor): The memory tensor from the encoder.
            input_idx (torch.Tensor): The input indices tensor.

        Returns:
            torch.Tensor: The output from the transformer decoder.
        """
        input_idx = input_idx.to(self.device)  # explicitly move to GPU
        naive_embedding = self.embedding(input_idx).permute(1, 0, 2)
        input_embedding = self.position_encoding(naive_embedding)       
        ipt_mask = self.get_tgt_mask(self.max_sequence_length)
        padding_mask = self.create_pad_mask(input_idx)
        transformer_output = self.tansformer_decoder(input_embedding, memory,tgt_mask = ipt_mask,tgt_key_padding_mask = padding_mask )
        return transformer_output
    
    def TF_2_logp(self,transformer_output):
        transformer_output_ = transformer_output.permute(1, 0, 2)
        logp = nn.functional.log_softmax(self.outputs2vocab(transformer_output_.reshape(-1, self.embedding_size)), dim=-1)
        logp = logp.reshape(-1, self.max_sequence_length , self.vocab_size) #
        return logp

    def decoder(self, memory):
        inference = []
        batch_size = memory.size(1)
        for i in range(batch_size):
            memory_ = memory[:, i, :].reshape((self.max_sequence_length, 1, self.embedding_size))
            init_input = torch.tensor([[self.sos_idx]], dtype=torch.long, device=self.device)
            for _ in range(self.max_sequence_length):
                init_naive_embedding = self.embedding(init_input).permute(1, 0, 2)
                target_embedding = self.position_encoding(init_naive_embedding)
                tgt_mask = self.get_tgt_mask(init_input.size(1))

                padding_mask = self.create_pad_mask(init_input)
                transformer_output = self.tansformer_decoder(target_embedding.float(), memory_.float(), tgt_mask=tgt_mask.float())
                transformer_output_ = transformer_output.permute(1, 0, 2)
                logp = nn.functional.log_softmax(self.outputs2vocab(transformer_output_.reshape(-1, self.embedding_size)), dim=-1)
                logp = logp.reshape(-1, self.vocab_size)
                next_item = torch.argmax(logp, dim=1).view(-1).tolist()[-1]
                next_item = torch.tensor([[next_item]], device=self.device)
                init_input = torch.cat((init_input, next_item), dim=1)
                if next_item.view(-1).item() == self.eos_idx:
                    break
            inference.append(init_input.view(-1).tolist()[1:-1])
        return inference
    
    def get_tgt_mask(self, size) -> torch.tensor:
        # Generates a squeare matrix where the each row allows one word more to be seen
        mask = torch.tril(torch.ones(size, size) == 1) # Lower triangular matrix
        mask = mask.float()
        mask = mask.masked_fill(mask == 0, float('-inf')) # Convert zeros to -inf
        mask = mask.masked_fill(mask == 1, float(0.0)) # Convert ones to 0
        
        return mask.to(self.device)
    
    def create_pad_mask(self, matrix: torch.tensor) -> torch.tensor:
        return (matrix == self.pad_idx)
    
    def index_2_selfies(self,list_of_index_list:list):
        size = len(list_of_index_list)
        ignore_list = [self.sos_idx, self.eos_idx, self.pad_idx]
        selfies_list = []
        for i in range(size):
            selfies_ = ''
            list_ = list_of_index_list[i]
            for v in list_:
                if (v not in ignore_list ):  
                    selfies_ += self.ind2char[v]
            selfies_list.append(selfies_)
        return selfies_list

    def selfies_2_smile(self,selfies_list):
        smiles = []
        for s in selfies_list:
            smiles.append(sf.decoder(s))
        return smiles

    def l1_regularization(self):
        reg_loss = 0
        if self.gpu:
            for param in self.high_fidelity_model.module.fc1.parameters(): ###
                reg_loss += self.l1_crit(param, target=torch.zeros_like(param))
            for param in self.high_fidelity_model.module.fc2.parameters(): ###
                reg_loss += self.l1_crit(param, target=torch.zeros_like(param))
        else:
            for param in self.high_fidelity_model.fc1.parameters(): ###
                reg_loss += self.l1_crit(param, target=torch.zeros_like(param))
            for param in self.high_fidelity_model.fc2.parameters(): ###
                reg_loss += self.l1_crit(param, target=torch.zeros_like(param))

        return settings.regularization_weight * reg_loss
        
   
    def forward(self,*args):
        """
        Defines the forward pass of the ChemTransformer.

        Parameters:
            *args: Variable length argument list containing input tensors.

        Returns:
            The output of the ChemTransformer which varies based on the mode.
        """
        input_idx = args[0].clone().detach().requires_grad_(False).to(torch.int).to(self.device)
    
        memory = self.encoder(input_idx)
        memory_ = memory.permute(1, 0, 2)
        
        if 'SS' in self.mode:
            transformer_output = self.teacher_forcing(memory,input_idx)
            logp = self.TF_2_logp(transformer_output)
            
        if 'HF' in self.mode:
            if self.mode == 'multiF_HF':
                descriptors = args[1].clone().detach().requires_grad_(False).to(torch.float)
                predictions = self.high_fidelity_model(memory_,descriptors)
            else:
                predictions = self.high_fidelity_model(memory_)
            if self.regularization:
                regularization = self.l1_regularization()
                if 'SS' in self.mode:
                    return predictions,logp,regularization
                else: 
                    return  predictions,regularization
        if self.mode == 'SS':
            return logp
        elif self.mode == 'SS_HF':
            return predictions,logp
        else:
            return predictions
        