# BUAA 机器学习作业 -- 团队 情感分类（文本分类）任务，2019 FALL

> 代码结构参照 https://github.com/yuhaozhang/tacred-relation 

## 代码使用指南
### 直接调用 best_model.pt 生成submit文件
```python
python eval.py
```
### 训练GloVe词向量
生成训练词向量的文件：

```python
python word2vec.py
```
GloVe 使用的是Stanford 大学提出的词向量训练算法，具体项目地址：https://github.com/stanfordnlp/GloVe
```shell
cd GloVe
make 
./demo.sh
```

### 获取预处理好的csv文件
```python
python preprocess.py
```

### 训练模型
```python
python train.py
```



