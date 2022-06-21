#    Mario Valerio Giuffrida - Multi-modal network training and testing
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

class real_range:
    _status=0

    def __init__(this,a=0,b=1,include_a=True,include_b=True):
        this.a=a
        this.b=b

        this.include_a = include_a
        this.include_b = include_b

    def __contains__(this, x):

        p1 = this.a <= x if this.include_a else this.a < x
        p2 = x <= this.b if this.include_b else x < this.b

        return p1 and p2

    def __iter__(this):
        return this

    def next(this):
        if (this._status==0):
            this._status += 1
            return this.a
        elif (this._status==0):
            this._status += 1
            return this.b
        else:
            this._status=0
            raise StopIteration()




parser = argparse.ArgumentParser(description='MultiModal Leaf Counter')
parser.add_argument('--rgb',dest='rgb',action='store_true')
parser.add_argument('--fmp',dest='fmp',action='store_true')
parser.add_argument('--ir',dest='ir',action='store_true')
parser.add_argument('--depth',dest='depth',action='store_true')
parser.add_argument('--split','-s',dest='split',action='store',type=int,default=0)
parser.add_argument('--split-file',dest='split_file',action='store',type=str,default='splits.npz')
parser.add_argument('--train',dest='train',action='store_true')
parser.add_argument('--test',dest='test',action='store_true')
parser.add_argument('--model','-m',dest='model',action='store',type=str,default=None)
parser.add_argument('--model-branch-rgb',dest='model_branch_rgb',action='store',type=str,default=None)
parser.add_argument('--model-branch-ir',dest='model_branch_ir',action='store',type=str,default=None)
parser.add_argument('--model-branch-fmp',dest='model_branch_fmp',action='store',type=str,default=None)
parser.add_argument('--loss','-l',dest='loss',action='store',type=str,default='mse',choices=['mse','mae','huber'])
parser.add_argument('--epochs','-e',dest='epochs',action='store',type=int,default=50)
parser.add_argument('--batch-size','-b',dest='bs',action='store',type=int,default=64)
parser.add_argument('--debug','-d',dest='debug',action='store_true')
parser.add_argument('--mute',dest='mute',action='store',type=str,default=[],nargs='+')
parser.add_argument('--patience',dest='patience',action='store',type=int,default=5)
parser.add_argument('--drop-prob',metavar='P',dest='drop_probability',action='store',type=float,default=0,choices=real_range(include_b=False))
parser.add_argument('--lambda',metavar='L',dest='l_varreg',action='store',type=float,default=1)
args = parser.parse_args()

if ((args.test == True) and (args.train == False)  and (args.model is None)):
    print "Trained model not set. Please use --model flag to specify a trained model"
    exit()


from network_keras import MultiModalLeafDeepCounter, compute_metrics
from dataset import MultiModalDataset, dataset_augmentation
from keras.applications.imagenet_utils import preprocess_input
from PIL import Image
import xlsxwriter as xls
import numpy as np
import os
import keras
import keras.backend as K

print "Keras Version: %s" % keras.__version__

K.set_image_data_format("channels_first")

def norm_perturbation(p,sigma):
    def fcn(y):
        if (np.random.binomial(1,p)==1):
            return y + np.random.normal(scale=sigma)
        else:
            return y

    return fcn


d = MultiModalLeafDeepCounter(rgb=args.rgb,fmp=args.fmp,ir=args.ir,depth=args.depth,debug=args.debug,finisher='classic',l2_w=0.02,l2var_reg=args.l_varreg)
d.LearingRate=0.001
d.MiniBatchSize=args.bs
d.compile(args.loss)
d.Epochs = args.epochs
d.EarlyStopLookAhead = args.patience
d.drop_probability = args.drop_probability

if (args.model_branch_rgb is not None):
    d.load_branch(args.model_branch_rgb,'rgb')

if (args.model_branch_ir is not None):
    d.load_branch(args.model_branch_ir,'ir')

if (args.model_branch_fmp is not None):
    d.load_branch(args.model_branch_fmp,'fmp')

data = np.load('mm_data.npz')['data'][0]

if ('rgb' in data):
    n = data['rgb'].shape[0]
    newrgb = np.zeros((n,3,240,240))

    for i in range(n):
        img = data['rgb'][i,:,:,:].transpose((1,2,0))
        prev = img.dtype
        img = np.asarray(img, dtype='uint8')
        I = Image.fromarray(img, mode=None)
        img = np.asarray(I.resize((240,240),Image.ANTIALIAS), dtype=prev)
        newrgb[i,:,:,:] = img.transpose((2,0,1))

    data['rgb'] = newrgb

if ('fmp' in data):
    data['fmp'] = np.repeat(data['fmp'],3,axis=1)
if ('ir' in data):
    data['ir'] = np.repeat(data['ir'], 3, axis=1)


splits = np.load(args.split_file)['splits']

xs = []
normaliser = []

max_norm = lambda x : x / x.max() * 255

if (args.rgb):
    xs.append(data['rgb'])    #if 'rgb' in data else np.zeros(shape=data['rgb'].shape))
    normaliser.append(preprocess_input)

if (args.fmp):
    xs.append(data['fmp'])    #   if 'fmp' in data else np.zeros(shape=data['fmp'].shape))
    normaliser.append(max_norm)

if (args.ir):
    xs.append(data['ir'])     #   if 'ir' in data else np.zeros(shape=data['ir'].shape))
    normaliser.append(max_norm)

if (args.depth):
    xs.append(data['depth']/2734.0*255.0)# if 'depth' in data else np.zeros(shape=data['depth'].shape))
    normaliser.append(None)


y = data['count']
N = y.shape[0]

idx_train = splits[args.split][2]
idx_val = splits[args.split][1]
idx_test = splits[args.split][0]

print '# Train: %d' % len(idx_train)
print '# Val: %d' % len(idx_val)
print '# Test: %d' % len(idx_test)


x_train = [x[idx_train] for x in xs]
x_val   = [x[idx_val] for x in xs]
x_test =  [x[idx_test] for x in xs]



x_train     = [ normaliser[i](x_train[i])  if normaliser[i] is not None else x_train[i]    for i in range(len(x_train)) ]
x_val       = [ normaliser[i](x_val[i])    if normaliser[i] is not None else x_val[i]    for i in range(len(x_val)) ]
x_test      = [ normaliser[i](x_test[i])   if normaliser[i] is not None else x_test[i]   for i in range(len(x_test)) ]


y_train = y[idx_train]
y_val   = y[idx_val]
y_test = y[idx_test]



print "Modality RGB: %s" % ('True' if args.rgb else 'False')
print "Modality FMP: %s" % ('True' if args.fmp else 'False')
print "Modality IR.: %s" % ('True' if args.ir else 'False')
print "Modality DPH: %s" % ('True' if args.depth else 'False')
print "Split: %d " % args.split

data = []

if (args.model is not None):
    print "Reading model: %s" % args.model
    fname = args.model
    d.load(fname)

if (args.train==True):
    if (args.model is  None):
        fname = 'results/MM_RGB=%d_FMP=%d_IR=%d_DPH=%d_SPLIT=%d.npz' % (args.rgb,args.fmp,args.ir,args.depth,args.split)

    print "Training model: %s" % fname

    try:
        d.train(x_train,y_train,x_val,y_val,model_fname=fname)
        print 'Saving results: %s' % fname
        #d.save(fname)
    except KeyboardInterrupt:
        f = fname.replace('.npz','-partial.npz')
        print "Emergency save: %s" % f
        d.save('results/'+f)
        exit()

if (args.test==True):
    y_hat={}; metrics = {}

    y_hat['train'] = d.test(x_train)
    y_hat['val'] = d.test(x_val)
    y_hat['test'] = d.test(x_test)

    metrics['train'] = compute_metrics(y_train, y_hat['train'])
    metrics['val']   = compute_metrics(y_val,   y_hat['val'])
    metrics['test']  = compute_metrics(y_test,  y_hat['test'])

    xls_fname = fname.replace('.npz','.xlsx')
    _,xls_fname = os.path.split(xls_fname)

    book = xls.Workbook('results/'+ xls_fname)
    sheet = book.add_worksheet()

    table = [('Y_train',y_train),
             ('Y_hat_train',y_hat['train']),
             ('Y_val', y_val),
             ('Y_hat_val', y_hat['val']),
             ('Y_test', y_test),
             ('Y_hat_test', y_hat['test'])]

    r=c=0

    for t in table:
        r=0
        sheet.write(r,c,t[0])
        for v in t[1]:
            r+=1
            sheet.write(r,c,v)

        c+=1

    book.close()

    # handle = open(csv_fname,'w')
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
