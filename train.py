"""
Train a model on TACRED.
"""

import os
import sys
from datetime import datetime
import time
import numpy as np
import random
import argparse
from shutil import copyfile

import torch
import torch.nn as nn
import torch.optim as optim 

from data_loader import load_data
from models.trainer import LSTMTrainer
from utils import torch_utils, helper
from sklearn.metrics import roc_auc_score, accuracy_score

def get_parser():
    parser = argparse.ArgumentParser()

    # Embedding hyper-parameter
    parser.add_argument("--emb_dim", type=int, default=300, help="Word embedding dimension.")

    # model parameter
    parser.add_argument("--hidden_dim", type=int, default=256, help="RNN hidden state size.")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of RNN layers.")
    parser.add_argument('--dropout', type=float, default=0.25, help='Dropout rate.')
    parser.add_argument('--rnn_dropout', type=float, default=0.1, help='RNN dropout rate.')
    # parser.add_argument('--rnn_type', type=str, default="lstm", help='lstm type')
    parser.add_argument('--output_dim', type=int, default=2, help='output dim.')
    parser.add_argument('--bidirectional', dest="bidirectional", action='store_true', help='bidirectional rnn')
    parser.add_argument('--no_bidirectional', dest="bidirectional", action='store_false')
    parser.set_defaults(bidirectional=True)


    parser.add_argument('--lr', type=float, default=1.0, help='Applies to sgd and adagrad.')
    parser.add_argument('--lr_decay', type=float, default=0.9, help='Learning rate decay rate.')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Decay learning rate after this epoch.')
    parser.add_argument('--optim', choices=['sgd', 'adagrad', 'adam', 'adamax'], default='adam', help='Optimizer: sgd, adagrad, adam or adamax.')
    parser.add_argument('--num_epoch', type=int, default=50, help='Number of total training epochs.')
    parser.add_argument('--batch_size', type=int, default=100, help='Training batch size.')
    parser.add_argument('--max_grad_norm', type=float, default=5.0, help='Gradient clipping.')
    parser.add_argument('--log_step', type=int, default=10, help='Print log every k steps.')
    parser.add_argument('--log', type=str, default='logs.txt', help='Write training log to file.')
    parser.add_argument('--save_epoch', type=int, default=5, help='Save model checkpoints every k epochs.')
    parser.add_argument('--save_dir', type=str, default='./saved_models', help='Root dir for saving models.')
    parser.add_argument('--id', type=str, default='00', help='Model ID under which to save models.')
    parser.add_argument('--info', type=str, default='', help='Optional info for the experiment.')

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true', help='Ignore CUDA.')

    parser.add_argument('--topn', type=int, default=1e10, help='Keep topn embedding.')

    parser.add_argument('--load', dest='load', action='store_true', help='Load pretrained model.')
    parser.add_argument('--model_file', type=str, help='Filename of the pretrained model.')

    return parser.parse_args()

def main():
    args = get_parser()

    # set seed and prepare for training
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.cpu:
        args.cuda = False
    elif args.cuda:
        torch.cuda.manual_seed(args.seed)
    init_time = time.time()

    # make opt
    opt = vars(args)
    TEXT, train_batch, dev_batch = load_data(opt['batch_size'], device='cuda:0')
    
    vocab = TEXT.vocab
    opt['vocab_size'] = len(vocab.stoi)
    emb_matrix = vocab.vectors

    assert emb_matrix.shape[0] == opt['vocab_size']
    assert emb_matrix.shape[1] == opt['emb_dim']

    model_id = opt['id'] if len(opt['id']) > 1 else '0' + opt['id']
    model_save_dir = opt['save_dir'] + '/' + str(model_id)
    opt['model_save_dir'] = model_save_dir
    helper.ensure_dir(model_save_dir, verbose=True)
  
    # save config
    path = os.path.join(model_save_dir, 'config.json')
    helper.save_config(opt, path, verbose=True)
    # vocab.save(os.path.join(model_save_dir, 'vocab.pkl'))
    file_logger = helper.FileLogger(os.path.join(model_save_dir, opt['log']), 
                                    header="# epoch\ttrain_loss\tdev_loss\tdev_score\tbest_dev_score")

    
    # print model info
    helper.print_config(opt)

    # Build Model
    if not opt['load']:
        trainer = LSTMTrainer(opt, emb_matrix)
    else:
        model_file = opt['model_file']
        print("Loading model from {}".format(model_file))
        model_opt = torch_utils.load_config(model_file)
        model_opt['optim'] = opt['optim']
        trainer = LSTMTrainer(model_opt)
        trainer.load(model_file)

    dev_score_history = []
    current_lr = opt['lr']

    global_step = 0
    global_start_time = time.time()
    format_str = '{}: step {}/{} (epoch {}/{}), loss = {:.6f} ({:.3f} sec/batch), lr: {:.6f}'
    max_steps = len(train_batch) * opt['num_epoch']

    # start training
    for epoch in range(1, opt['num_epoch'] + 1):
        train_loss = 0
        for i, batch in enumerate(train_batch):
            start_time = time.time()
            global_step += 1
            loss = trainer.update(batch)
            train_loss += loss
            if global_step % opt['log_step'] == 0:
                duration = time.time() - start_time
                print(format_str.format(datetime.now(), global_step, max_steps, epoch, \
                    opt['num_epoch'], loss, duration, current_lr))
        
        # eval on dev
        print("Evaluating on dev set ...")
        predictions = []
        golds = []
        dev_loss = 0.0
        for i, batch in enumerate(dev_batch):
            preds, probs, labels, loss = trainer.predict(batch)
            predictions += preds
            golds += labels
            dev_loss += loss 
        train_loss = train_loss / len(train_batch)
        dev_loss = dev_loss / len(dev_batch)
        # print(golds)
        # print(predictions)
        print(accuracy_score(golds, predictions))
        dev_roc = roc_auc_score(golds, predictions)
        print("epoch {}: train loss = {:.6f}, dev loss = {:.6f}, dev roc = {:.4f}".format(
            epoch, train_loss, dev_loss, dev_roc
        ))
        dev_score = dev_roc
        file_logger.log("{}\t{:.6f}\t{:.6f}\t{:.4f}\t{:.4f}".format(
            epoch, train_loss, dev_loss, dev_score, max([dev_score] + dev_score_history)))
        
        # save model
        model_file = os.path.join(model_save_dir, "checkpoint_epoch_{}.py".format(epoch))
        trainer.save(model_file, epoch)
        if epoch == 1 or dev_score > max(dev_score_history):
            copyfile(model_file, model_save_dir + '/best_model.pt')
            print("new best model saved.")
            file_logger.log("new best model saved at epoch {}: {:.2f}"\
                .format(epoch, dev_score*100))
        if epoch % opt['save_epoch'] != 0:
            os.remove(model_file)
        
        if len(dev_score_history) > opt['decay_epoch'] and dev_score <= dev_score_history[-1] and \
            opt['optim'] in ['sgd', 'adagrad', 'adadelta']:
            current_lr *= opt['lr_decay']
            trainer.update_lr(current_lr)
        
        dev_score_history += [dev_score]
        print("")
    
    print("Training ended with {} epochs.".format(epoch))

if __name__ == "__main__":
    main()