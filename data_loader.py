import torch
import torchtext
from torchtext import data, datasets
from torchtext.vocab import Vectors

def get_data():
    tokenize = lambda x: x.split()
    LABEL = data.LabelField(dtype=torch.float32)
    TEXT = data.Field(sequential=True, tokenize=tokenize, \
        lower=True, include_lengths=True, batch_first=True, fix_length=512)
    print("读取csv数据...")
    train_data, validate_data, test_data = data.TabularDataset.splits(
        path='dataset/sentiment', train='train.csv', validation='dev.csv', test='test.csv',
        format='csv', skip_header=True, fields=[('text', TEXT), ('label', LABEL)]
    )
    print("构建vocab，并读取Word Embedding...")
    TEXT.build_vocab(train_data, min_freq=5, vectors=Vectors(name='glove.6B.300d.txt', cache='dataset/glove/'))
    print("数据读取，词表词向量构建完成...")
    return TEXT, train_data, validate_data, test_data

def load_data(batch_size=32, device='cpu'):
    TEXT, train_data, validate_data, test_data = get_data()

    print('Batch data ...')
    train_iter, validate_iter, test_iter = data.BucketIterator.splits(
        (train_data, validate_data, test_data), batch_size=batch_size, sort=False, device=device
    )

    return TEXT, train_iter, validate_iter, test_iter

if __name__ == "__main__":
    TEXT, train_iter, validate_iter, test_iter = load_data(32)
    print(TEXT.vocab.itos[:10])
    print(train_iter.dataset.example[7].text)





