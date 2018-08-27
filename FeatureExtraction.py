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

#ctx = mx.gpu()
#if yous machine not supports gpu use cpu. i used free trail of google cloud ,i dont have gpu i used cpu
ctx = mx.cpu()


#setting parameters
data_dir = "data" 

#288 = 224 + 32 *2, 352 = 224 + 32 * 4
imageSize_resnet = 288  

# 363 = 299 + 32 *2, 427 = 299 + 32 * 4
imageSize_inception = 363


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


n = len(glob(os.path.join('.', data_dir, "Images", "*", "*.jpg")))

mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

#improving resnet50 pretrained model weights
net = models.get_model('resnet152_v1', pretrained=True, ctx=ctx)
features = []
for j in tqdm(range(0,161)):
    i = 0
    temp = nd.zeros((128, 3, imageSize_resnet, imageSize_resnet)) 
    for file_name in glob(os.path.join(data_dir, "Images", "*", "*.jpg"))[128*j:128*(j+1)]:
        img = cv2.imread(file_name)
        img_224 = ((cv2.resize(img, (imageSize_resnet, imageSize_resnet))[:,:,::-1] \
                    / 255.0 - mean) / std).transpose((2, 0, 1))
        temp[i] = nd.array(img_224)
        nd.waitall()
        i += 1
    if j == 160:
        temp = temp[0:100]
    data_iter_224 = gluon.data.DataLoader(gluon.data.ArrayDataset(temp), batch_size=128)
    for data in data_iter_224:
        feature = net.features(data.as_in_context(mx.gpu()))
        feature = gluon.nn.Flatten()(feature)
        features.append(feature.as_in_context(mx.cpu()))
    nd.waitall()
features = nd.concat(*features, dim=0)
print(features.shape)
nd.save(os.path.join(data_dir, 'features_res.nd'), features)
