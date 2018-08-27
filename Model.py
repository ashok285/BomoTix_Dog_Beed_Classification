import mxnet as mx
from mxnet import init, gluon, nd, autograd, image
from mxnet.gluon import nn
from mxnet.gluon.data import vision
from mxnet.gluon.model_zoo import vision as models
import numpy as np
import pandas as pd
from tqdm import tqdm
import cv2
import h5py
import os
from glob import glob
import matplotlib.pyplot as plt

ctx = mx.cpu()

data_dir        = '/home/ashokkunam9/dogbeed/data'
batch_size      = 128
learning_rate   = 1e-3
epochs          = 150
lr_decay        = 0.95
lr_decay2       = 0.8
lr_period       = 100
#output file creation

submit_fileName = '/home/ashokkunam9/dogbeed/data/pred.csv'


synset = list(pd.read_csv(os.path.join('.', data_dir, '/home/ashokkunam9/dogbeed/data/sample_submission.csv')).columns[1:])
n = len(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg')))

y = nd.zeros((n,))
for i, file_name in tqdm(enumerate(glob(os.path.join('.', data_dir, 'Images', '*', '*.jpg'))), total=n):
    y[i] = synset.index(file_name.split('/')[3][10:].lower())
    nd.waitall()

features = [nd.load(os.path.join(data_dir, 'features_incep.nd'))[0], \
            nd.load(os.path.join(data_dir, 'features_res.nd'))[0]]
features = nd.concat(*features, dim=1)

features.shape

models   = ['incep', 'res']
features_test = [nd.load(os.path.join(data_dir, 'features_test_%s.nd') % model)[0] for model in models]
features_test = nd.concat(*features_test, dim=1)

print(features_test.shape)

def build_model():
    net = nn.Sequential()
    with net.name_scope():
        net.add(nn.BatchNorm())
        net.add(nn.Dense(1024))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
#         net.add(nn.Dropout(0.5))
        net.add(nn.Dense(512))
        net.add(nn.BatchNorm())
        net.add(nn.Activation('relu'))
#         net.add(nn.Dropout(0.5))
        net.add(nn.Dense(120))
    net.initialize(ctx=ctx)
    return net


def accuracy(output, labels):
    return nd.mean(nd.argmax(output, axis=1) == labels).asscalar()


def evaluate(net, data_iter):
    loss, acc, n = 0., 0., 0.
    steps = len(data_iter)
    for data, label in data_iter:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        output = net(data)
        acc += accuracy(output, label)
        loss += nd.mean(softmax_cross_entropy(output, label)).asscalar()
    return loss/steps, acc/steps
    

data_iter_train       = gluon.data.DataLoader(gluon.data.ArrayDataset(features, y), batch_size, shuffle=True)
softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()
net                   = build_model()
trainer               = gluon.Trainer(net.collect_params(), 'adam', {'learning_rate': learning_rate})

# https://github.com/yinglang/CIFAR10_mxnet/blob/master/CIFAR10_train.md

for epoch in range(epochs):
    if epoch <= lr_period:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay)
    else:
        trainer.set_learning_rate(trainer.learning_rate * lr_decay2)
    train_loss = 0.
    train_acc = 0.
    steps = len(data_iter_train)
    for data, label in data_iter_train:
        data, label = data.as_in_context(ctx), label.as_in_context(ctx)
        with autograd.record():
            output = net(data)
            loss = softmax_cross_entropy(output, label)
        loss.backward()
        trainer.step(batch_size)
        train_loss += nd.mean(loss).asscalar()
        train_acc += accuracy(output, label)

    val_loss, val_acc = evaluate(net, data_iter_train)
 
    if epoch % 10 == 0:
        print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
            epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))

print("Epoch %d. loss: %.4f, acc: %.2f%%, val_loss %.4f, val_acc %.2f%%" % (
    epoch+1, train_loss/steps, train_acc/steps*100, val_loss, val_acc*100))

output = nd.softmax(net(nd.array(features_test).as_in_context(ctx))).asnumpy()

#output file generation
df_pred = pd.read_csv(os.path.join('.', data_dir, 'sample_submission.csv'))

for i, c in enumerate(df_pred.columns[1:]):
    df_pred[c] = output[:,i]

df_pred.to_csv(os.path.join('.', data_dir, submit_fileName), index=None)
