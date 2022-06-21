#    Mario Valerio Giuffrida - SIngle-modal network training on cvppp dataset
#    Copyright (C) 2018 Mario Valerio Giuffrida
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details. 
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

import argparse
import warnings

parser = argparse.ArgumentParser(description='MultiModal Leaf Counter')
parser.add_argument('--model','-m',dest='model',action='store',type=str,default='CVPPP.npz')
parser.add_argument('--loss','-l',dest='loss',action='store',type=str,default='mse',choices=['mse','mae','huber'])
parser.add_argument('--epochs','-e',dest='epochs',action='store',type=int,default=50)
parser.add_argument('--batch-size','-b',dest='bs',action='store',type=int,default=16)
parser.add_argument('--debug','-d',dest='debug',action='store_true')
parser.add_argument('--train',dest='train',action='store_true')
parser.add_argument('--validate',dest='val',action='store_true')
parser.add_argument('--test',dest='test',action='store_true')
parser.add_argument('--patience',dest='patience',action='store',type=int,default=10)
#parser.add_argument('--resnet',dest='resnet',action='store_true')
#parser.add_argument('--blur',dest='blur',action='store_true')
parser.add_argument('--implementation','-i',dest='impl',action='store',type=str,default='keras',choices=['keras'])
parser.add_argument('--finisher','-f',dest='finisher',action='store',type=str,default='classic',choices=['classic'])
parser.add_argument('--split','-s',dest='split',action='store',type=int,default=0,choices=range(4))
args = parser.parse_args()

print(args)

# if ((args.impl=='keras') and (args.resnet)):
#     warnings.warn('Keras only use Resnet as modality branch. The --resnet parameter can be omitted in this case')
#
#
# if (args.impl =='lasagne'):
#     import network
#     from network import MultiModalLeafDeepCounter, compute_metrics
# elif (args.impl=='keras'):
#     import keras.backend as K
#     #K.set_image_data_format("channels_first")
import network_keras as network
from network_keras import MultiModalLeafDeepCounter, compute_metrics

from dataset import dataset_augmentation
import numpy as np
from dataset import get_data,get_data_testing,get_data_masks,get_data_testing_mask
from PIL import Image,ImageOps
from keras.applications.imagenet_utils import preprocess_input


network.RGB_SHAPE = (320,320,3)

def norm_perturbation(p,sigma):
    def fcn(y):
        if (np.random.binomial(1,p)==1):
            return y + np.random.normal(scale=sigma)
        else:
            return y

    return fcn

def autocontrast(x):
    for k in range(x.shape[0]):
        I  = Image.fromarray(x[k,:,:,:].transpose((1,2,0)).astype('uint8'))
        I  = ImageOps.autocontrast(I)
        y  = np.asarray(I).transpose((2,0,1))
        x[k,:,:,:] = y
    return x

def blur_background(x,m):
	import cv2 

	fg = x * m
	bg = x * (1-m)

	n = np.zeros(x.shape)

	for i in range(n.shape[0]):
		im = bg[i,:,:,:].transpose((1,2,0))

		n[i,:,:,:] = cv2.GaussianBlur(im,(19,19),9).transpose(2,0,1)

	n += fg

	return n






d = MultiModalLeafDeepCounter(rgb=True,fmp=False,ir=False,depth=False,finisher=args.finisher,debug=args.debug,l2_w=0.02)
d.LearingRate=0.0001
d.MiniBatchSize=args.bs
d.compile(args.loss)
d.Epochs = args.epochs
d.EarlyStop=True
d.EarlyStopLookAhead = args.patience

splits =  [[[1,2,3], 4,4],
           [[2,3,4], 1,1],
           [[3,4,1], 2,2],
           [[4,1,2], 3,3]]




fname = args.model

if (args.train):
    x = get_data(splits[args.split],'/media/PHDDATA/Data/Plant Phenotyping/CVPPP_2017_splits/training/TrainingSplits')
    x_m = get_data_masks(splits[args.split], '/media/PHDDATA/Data/Plant Phenotyping/CVPPP_2017_splits/training/TrainingSplits')


    x_train = x[0].astype(float)
    x_val = x[1].astype(float)
    y_train = x[3]
    y_val = x[4]

    # if (args.blur):
    #     x_train_mask = x_m[0].transpose((0, 3, 1, 2)) > 0
    #     x_val_mask = x_m[1].transpose((0, 3, 1, 2)) > 0
    #
    #     x_train = blur_background(x_train,x_train_mask)
    #     x_val   = blur_background(x_val,x_val_mask)

    try:
        d.train([x_train],y_train,[x_val],y_val,fname)
    except KeyboardInterrupt:
        f = fname.replace('.npz','-partial.npz')
        print("Emergency save: {}".format(f))

        d.save(f)
elif (args.val):
    print("Starting validation training only")
    x = get_data(splits[args.split], '/media/PHDDATA/Data/Plant Phenotyping/CVPPP_2017_splits/training/TrainingSplits')
    x_train = np.concatenate((x[0].astype(float),x[1].astype(float)))
    y_train = np.concatenate((x[3],x[4]))

    x_val = x[2]
    y_val = x[5]


    try:
        d.train([x_train], y_train, [x_val], y_val, fname)
    except KeyboardInterrupt:
        f = fname.replace('.npz', '-partial.npz')
        print("Emergency save: {}".format(f))

        d.save(f)
if (args.test):
    import xlsxwriter as xls

    x = get_data_testing('../CVPPP_2017/CVPPP2017_testing/testing')
    x_m = get_data_testing_mask('../CVPPP_2017/CVPPP2017_testing/testing')

    x_test = (x[0].transpose((0,3,1,2)).astype(float))
    y_test = x[1]
    datasets = x[2]

    if (args.blur):
        x_test_mask = x_m[0].transpose((0, 3, 1, 2)) > 0
        x_test = blur_background(x_test,x_test_mask)

    d.load(fname)

    y_test_hat = d.test(x_test)

    # idx_test = splits[args.split][0]
    # x_test = [x[idx_test] for x in xs]
    # x_test = [x_test[i] / max_values[i] for i in range(len(max_values))]
    #
    # y_test = y[idx_test]
    #
    # y_hat={}; metrics = {}
    #
    # y_hat['train'] = d.test(x_train,False)
    # y_hat['val'] = d.test(x_val, False)
    # y_hat['test'] = d.test(x_test, False)
    #
    # metrics['train'] = compute_metrics(y_train, y_hat['train'])
    # metrics['val']   = compute_metrics(y_val,   y_hat['val'])
    # metrics['test']  = compute_metrics(y_test,  y_hat['test'])
    #
    xls_fname = fname.replace('.npz','.xlsx')

    book = xls.Workbook(xls_fname)
    sheet = book.add_worksheet()

    for t in range(len(datasets)):
        sheet.write(t, 0, y_test[t])
        sheet.write(t, 1, y_test_hat[t])
        sheet.write(t, 2, datasets[t])


    book.close()

    #
    # handle.write('-,dic,dic std,|dic|,|dic| std,mse,%,r2\n')
    #
    # for lbl in metrics:
    #     handle.write('%s,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f,%.5f\n' % (lbl,
    #                                                               metrics[lbl]['dic'][0],
    #                                                               metrics[lbl]['dic'][1],
    #                                                               metrics[lbl]['abs_dic'][0],
    #                                                               metrics[lbl]['abs_dic'][1],
    #                                                               metrics[lbl]['mse'],
    #                                                               metrics[lbl]['%'],
    #                                                               metrics[lbl]['r2']
    #                                                               ))
    #
    # handle.write('Train')
    # handle.write('\nY:,'); np.savetxt(handle,y_train.reshape((1,-1)),fmt='%d',delimiter=',')
    # handle.write('\nY_hat:,');np.savetxt(handle,y_hat['train'].reshape((1,-1)),fmt='%d',delimiter=',')
    #
    # handle.write('\nValidation')
    # handle.write('\nY:,');np.savetxt(handle, y_val.reshape((1,-1)), fmt='%d', delimiter=',')
    # handle.write('\nY_hat:,');np.savetxt(handle, y_hat['val'].reshape((1,-1)), fmt='%d', delimiter=',')
    #
    # handle.write('\nTest')
    # handle.write('\nY:,');    np.savetxt(handle, y_test.reshape((1,-1)), fmt='%d', delimiter=',')
    # handle.write('\nY_hat:,'); np.savetxt(handle, y_hat['test'].reshape((1,-1)), fmt='%d', delimiter=',')
    #
    # handle.close()
