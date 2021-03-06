from __future__ import print_function
import torch
import torch.nn as nn
import numpy as np
from torch.nn.functional import binary_cross_entropy
from torch.autograd import Variable

class RNNBinaryClassifierModule(nn.Module):
    ''' RNNBinaryClassifierModule

    NeuralNet module for binary classification of sequences/time-series

    Examples
    --------
    # Apply random weights to data
    >>> RNNBinaryClassifierModule('LSTM', 1, 3, 1)

    Args
    ----
    rnn_type : str
        One of ['LSTM', 'GRU', 'ELMAN+relu', 'ELMAN+tanh']
    n_inputs : int
        Number of input features
    n_hiddens : int    
        Number of hidden units in each rnn cell 
    '''
    def __init__(self,
            rnn_type='LSTM', n_inputs=1, n_hiddens=1, n_layers=1,
            dropout_proba=0.0, dropout_proba_non_recurrent=0.0, bidirectional=False):
        super(RNNBinaryClassifierModule, self).__init__()
        self.drop = nn.Dropout(dropout_proba)
        self.dropout_proba_non_recurrent = dropout_proba_non_recurrent
        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(
                n_inputs, n_hiddens, n_layers,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        elif rnn_type in ['ELMAN+tanh', 'ELMAN+relu']:
            nonlinearity = rnn_type.split("+")[1]
            self.rnn = nn.RNN(
                n_inputs, n_hiddens, n_layers,
                nonlinearity=nonlinearity,
                bidirectional=bidirectional,
                batch_first=True,
                dropout=dropout_proba)
        else:
            raise ValueError("Invalid option for --rnn_type: %s" % rnn_type)
        self.output = nn.Linear(n_hiddens, 2) # weights initialized as samples from uniform[-1,1]
        
        # initialize weights of the module as samples from N(0,1)
        #         nn_module = nn.Linear(n_hiddens,2)
        #         nn_module.apply(init_weights)
        #         self.output = nn_module
        self.double()
    
        
    def score(self, X, y, sample_weight=None):
        correct_predictions = 0
        total_predictions = 0

        results = self.forward(torch.DoubleTensor(X))
        for probabilties, outcome in zip(results, y):
            if probabilties[outcome] > 0.5:
                correct_predictions += 1
            total_predictions += 1

        return float(correct_predictions) / total_predictions
    
    def score_bce(self, X, y, sample_weight=None):
        results = self.forward(torch.DoubleTensor(X))
        return binary_cross_entropy(results[:,0],y.double())

    def forward(self, inputs_NTF, seq_lens_N=None, pad_val=0, return_hiddens=False):
        ''' Forward pass of input data through NN module

        Cleanly handles variable-length sequences (though internals a bit messy).

        Args
        ----
        inputs_NTF : 3D array (n_sequences, n_timesteps, n_features)
            Each row is one sequence, padded to length T = n_timesteps
        seq_lens_N : 1D array-like (n_sequences)
            Each entry indicates how many timesteps the n-th sequence has.
            (Remaining entries are all padding and should be ignored).

        Returns
        -------
        yproba_N2 : 2D array (n_sequences, 2)
            Each row gives probability that given sequence is class 0 or 1
            Each row sums to one
        
        hiddens_NTH : 3D array (n_sequences, n_timesteps, n_hiddens)
            Each (n,t) index gives the hidden-state vector at sequence n, timestep t
        '''
        N, T, F = inputs_NTF.shape

        if seq_lens_N is None:
            if T>1:
                seq_lens_N = torch.zeros(N, dtype=torch.int64)
                # account for collapsed features across time
                for n in range(N):
                    bmask_T = torch.all(inputs_NTF[n] == pad_val, dim=-1)
                    seq_lens_N[n] = np.searchsorted(bmask_T, 1)
            else:
                seq_lens_N = torch.ones(N, dtype=torch.int64)

        ## Create PackedSequence representation to handle variable-length sequences
        # Requires sorting all sequences in current batch in descending order by length
        sorted_seq_lens_N, ids_N = seq_lens_N.sort(0, descending=True)
        _, rev_ids_N = ids_N.sort(0, descending=False)
        sorted_inputs_NTF = inputs_NTF[ids_N] 
        packed_inputs_PF = nn.utils.rnn.pack_padded_sequence(sorted_inputs_NTF, sorted_seq_lens_N, batch_first=True)
        
        # Apply dropout to the non-recurrent layer weights between LSTM layers before output ie is weights for h_(l-1)^t
        # See https://pytorch.org/docs/stable/nn.html#torch.nn.LSTM for choosing the right weights
        if (self.dropout_proba_non_recurrent>0.0 and self.rnn.num_layers>1):
            dropout = nn.Dropout(p=self.dropout_proba_non_recurrent)
            self.rnn.weight_ih_l1 = torch.nn.Parameter(dropout(self.rnn.weight_ih_l1), 
                                                                      requires_grad=True)
            self.rnn.bias_ih_l1 = torch.nn.Parameter(dropout(self.rnn.bias_ih_l1), 
                                                                    requires_grad=True)
        
        ## Apply the RNN  
        packed_outputs_PH, _ = self.rnn(packed_inputs_PF)
        ## Unpack to N x T x H padded representation
        outputs_NTH, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs_PH, batch_first=True)
        ## Apply weights + softmax to final timestep of each sequence
        end_hiddens_NH = outputs_NTH[range(N), sorted_seq_lens_N - 1]
        yproba_N2 = nn.functional.softmax(self.output(end_hiddens_NH), dim=-1)
        # yproba_N2 = nn.functional.logsigmoid(self.output(end_hiddens_NH))

        ## Unsort and return
        if return_hiddens:
            return yproba_N2.index_select(0, rev_ids_N), outputs_NTH.index_select(0, rev_ids_N)
        else:
            return yproba_N2.index_select(0, rev_ids_N)

# function to handle weight initialization 
def init_weights(m):
    if type(m) == nn.Linear:
        torch.manual_seed(42)
        m.weight.data = torch.randn(m.weight.shape)
        print(m.weight)        
        
if __name__ == '__main__':
    N = 5   # n_sequences
    T = 10  # n_timesteps
    F = 3   # n_features
    H = 2   # n_hiddens

    np.random.seed(0)
    torch.random.manual_seed(0)

    # Generate random sequence data
    inputs_NTF = np.random.randn(N, T, F)
    y_N_ = torch.rand(N, requires_grad = False)
    y_N = y_N_.detach().numpy().astype(int)
    seq_lens_N = np.random.randint(low=1, high=T, size=N)
    
    # Convert numpy to torch
    inputs_NTF_ = torch.from_numpy(inputs_NTF)
    seq_lens_N_ = torch.from_numpy(seq_lens_N)

    rnn_clf = RNNBinaryClassifierModule('LSTM', n_inputs=F, n_hiddens=H, n_layers=1)
    yproba_N2_, hiddens_NTH_ = rnn_clf.forward(inputs_NTF_, seq_lens_N_, return_hiddens=True)
    yproba_N2 = yproba_N2_.detach().numpy()
    hiddens_NTH = hiddens_NTH_.detach().numpy()
    accuracy_scores_N2 = rnn_clf.score(inputs_NTF, y_N)
    bce_scores_N2 = rnn_clf.score_bce(inputs_NTF_, y_N_)
    
    for n in range(N):
        print("==== Sequence %d" % n)
        print("X:")
        print(inputs_NTF[n, :seq_lens_N[n]])
        print("H:")
        print(hiddens_NTH[n, :seq_lens_N[n]])
        print("yproba:")
        print(yproba_N2[n])
        print("accuracy score:")
        print(accuracy_scores_N2)        
        print("BCE loss:")
        print(bce_scores_N2.detach().numpy())