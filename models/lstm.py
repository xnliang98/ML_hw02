import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from torch.nn import init

from utils import torch_utils




class LSTMClassifier(nn.Module):
    """ A classifier use lstm to extract relations. """
    def __init__(self, opt, emb_matrix=None):
        super(LSTMClassifier, self).__init__()
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.use_cuda = opt['cuda']
        self.topn = self.opt.get('topn', 1e10)

        self.drop = nn.Dropout(opt['dropout'])
        self.emb = nn.Embedding(opt['vocab_size'], opt['emb_dim'], padding_idx=1)

        input_size = opt['emb_dim']
        self.rnn = nn.LSTM(input_size, opt['hidden_dim'], num_layers=opt['num_layers'], \
            bidirectional=opt['bidirectional'], dropout=opt['rnn_dropout'], batch_first=True)
        self.direct = 1
        if opt['bidirectional']:
            self.direct = 2

        self.linear = nn.Linear(opt['hidden_dim'] * self.direct, opt['hidden_dim'])
        self.predict = nn.Linear(opt['hidden_dim'], opt['output_dim'])

        self.init_weights()
    
    def init_weights(self):
        if self.emb_matrix is None:
            # if emb matrix is Noen, random init word vector and keep padding dimension to be 0
            self.emb.weight.data[1:, :].uniform_(-1.0, 1.0)
        else:
            self.emb.weight.data.copy_(self.emb_matrix)

        self.linear.bias.data.fill_(0)
        init.xavier_uniform_(self.linear.weight, gain=1) # initialize linear layer
        init.xavier_uniform_(self.predict.weight, gain=1)
        
        # decide finetuning
        if self.topn <= 0:
            print("Do not finetune word embedding layer.")
            self.emb.weight.requires_grad = False
        elif self.topn < self.opt['vocab_size']:
            print("Finetune top {} word embeddings.".format(self.topn))
            self.emb.weight.register_hook(lambda x: \
                    torch_utils.keep_partial_grad(x, self.topn))
        else:
            print("Finetune all embeddings.")
        
    def forward(self, x):
        # seq_lens = list(masks.data.eq(constant.PAD_ID).long().sum(1).squeeze())
        batch_size = x.size(0)

        # embedding lookup
        x = self.emb(x)
        x = self.drop(x)
        input_size = x.size(2)

        # rnn
        rnn_outputs, (ht, _) = self.rnn(x)

        if self.direct == 2:
            hidden = torch.cat([ht[-1, :, :], ht[-2, :, :]], dim=-1)
        else:
            hidden = ht[-1, :, :]

        hidden = self.drop(hidden)
        # outputs
        outputs = self.linear(hidden)
        outputs = self.drop(outputs)
        outputs = self.predict(outputs)
        return outputs

