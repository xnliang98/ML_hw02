"""
A trainer class.
"""

import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

from models.lstm import LSTMClassifier
from utils import torch_utils, helper

def unpack_batch(batch):
    inputs = batch.text[0]
    labels = batch.label
    return inputs, labels

class Trainer(object):
    """Interface of Trainer of each model training."""
    def __init__(self, opt, emb_matrix=None):
        raise NotImplementedError

    def update(self, batch):
        raise NotImplementedError

    def predict(self, batch):
        raise NotImplementedError

    def update_lr(self, new_lr):
        torch_utils.change_lr(self.optimizer, new_lr)
    
    def load(self, filename):
        try:
            checkpoint = torch.load(filename)
        except BaseException:
            print("Can not load model from {}".format(filename))
            exit()
        self.model.load_state_dict(checkpoint['model'])
        self.opt = checkpoint['config']
    
    def save(self, filename, epoch):
        params = {
                'model': self.model.state_dict(),
                'config': self.opt,
                }
        try:
            torch.save(params, filename)
            print("model saved to {}".format(filename))
        except BaseException:
            print("[Warning: Saving failed... continuing anyway.]")

class LSTMTrainer(Trainer):
    def __init__(self, opt, emb_matrix=None):
        self.opt = opt
        self.emb_matrix = emb_matrix
        self.model = LSTMClassifier(opt, emb_matrix=emb_matrix)
        self.criterion = nn.CrossEntropyLoss()
        self.parameters = [p for p in self.model.parameters() if p.requires_grad]
        if opt['cuda']:
            self.model.cuda()
            self.criterion.cuda()
        self.optimizer = torch_utils.get_optimizer(opt['optim'], self.parameters, opt['lr'])
    
    def update(self, batch):
        inputs, labels = unpack_batch(batch)

        # Step 1 init and forward
        self.model.train()
        self.optimizer.zero_grad()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()

        # Step 2 backward
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.opt['max_grad_norm'])

        # Step 3 update
        self.optimizer.step()
        return loss_val 
    
    def predict(self, batch, unsort=True):
        inputs, labels = unpack_batch(batch)
        
        self.model.eval()

        logits = self.model(inputs)
        loss = self.criterion(logits, labels)
        loss_val = loss.item()
        
        probs = F.softmax(logits, 1).data.cpu().numpy().tolist()
        predictions = np.argmax(logits.data.cpu().numpy(), axis=1).tolist()
        labels = labels.data.cpu().numpy().tolist()
        return predictions, probs, labels, loss_val