import torch
import torchtext
from torchtext import data, datasets
from torchtext.vocab import Vectors

def load_data(batch_size=32, is_test=False, device='cpu'):
    tokenize = lambda x: x.split()
    LABEL = data.LabelField(dtype=torch.long)
    TEXT = data.Field(sequential=True, tokenize=tokenize, \
        lower=True, include_lengths=True, batch_first=True, fix_length=256)
    print("读取csv数据...")
    train_data, test_data = data.TabularDataset.splits(
        path='dataset/sentiment', train='train.csv', test='test.csv',
        format='csv', skip_header=True, fields=[('text', TEXT), ('label', LABEL)]
    )
    print("构建vocab，并读取Word Embedding...")
    TEXT.build_vocab(train_data, min_freq=1, vectors=Vectors(name='glove.840B.300d.txt', cache='dataset/glove/'))
    LABEL.build_vocab(train_data)
    print("数据读取，词表词向量构建完成...")
    print ("Length of Text Vocabulary: " + str(len(TEXT.vocab)))
    print ("Vector size of Text Vocabulary: ", TEXT.vocab.vectors.size())
    print ("Label Length: " + str(len(LABEL.vocab)))
    train_data, valid_data = train_data.split() 
    # Further splitting of training_data to create new training_data & validation_data
    train_iter, valid_iter = data.BucketIterator.splits(
        (train_data, valid_data), batch_size=batch_size, sort_key=lambda x: len(x.text), \
            repeat=False, shuffle=True, device=device)
    if is_test:
        return TEXT, data.BucketIterator.splits(
            (test_data), batch_size=batch_size, shuffle=False, device=device
        )
    return TEXT, train_iter, valid_iter








