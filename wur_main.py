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
parser.add_argument(metavar='BASEPATH' ,dest='basepath',action='store',type=str)
parser.add_argument('--species',   '-s',dest='species',action='store',type=str,default='all')
parser.add_argument('--model'  ,   '-m',dest='model',action='store',type=str)
parser.add_argument('--output',    '-o',dest='output',action='store',type=str,default='out.npz')
parser.add_argument('--loss',      '-l',dest='loss',action='store',type=str,default='mse',choices=['mse','mae','huber'])
parser.add_argument('--epochs',    '-e',dest='epochs',action='store',type=int,default=50)
parser.add_argument('--batch-size','-b',dest='bs',action='store',type=int,default=16)
parser.add_argument('--debug',     '-d',dest='debug',action='store_true')
parser.add_argument('--train',          dest='train',action='store_true')
parser.add_argument('--val',            dest='val',action='store',type=float,default=0.2)
parser.add_argument('--test',           dest='test',action='store_true')
parser.add_argument('--patience',       dest='patience',action='store',type=int,default=10)

args = parser.parse_args()

print(args)

import network_keras as network
from network_keras import MultiModalLeafDeepCounter, compute_metrics, print_metrics
from dataset import WageningenDataset
import xlsxwriter as xls


network.RGB_SHAPE = (320,320,3)


d = MultiModalLeafDeepCounter(rgb=True,fmp=False,ir=False,depth=False,debug=args.debug,l2_w=0.02)
d.LearingRate=0.0001
d.MiniBatchSize=args.bs
d.compile("mse")
d.Epochs = args.epochs
d.EarlyStop=True
d.EarlyStopLookAhead = args.patience

if (args.model is not None):
    d.load(args.model)


dataset = WageningenDataset(basepath=args.basepath,species=args.species)
dataset.load(resize=network.RGB_SHAPE[0:2],seed=47,val_portion=args.val)


if (args.train):
    model_fname = args.output


    try:
        d.train([dataset.x_train],dataset.y_train,[dataset.x_val],dataset.y_val,model_fname)
    except KeyboardInterrupt:
        f = model_fname.replace('.npz','-partial.npz')
        print("Emergency save: {}".format(f))

        d.save(f)

if (args.test):

    if (args.train == False):
        model_fname = args.model
        d.load(model_fname)
    # else:
    xls_fname = args.output.replace('.npz', '.xlsx')

    y_train_hat = d.test(dataset.x_train)
    y_test_hat = d.test(dataset.x_test)

    metrics = {}
    metrics['train'] = compute_metrics(dataset.y_train, y_train_hat)
    metrics['test']  = compute_metrics(dataset.y_test,  y_test_hat)

    print("==== Training performance ====")
    print_metrics(metrics['train'])

    print("\n==== Testing performance ====")
    print_metrics(metrics['test'])
    #


    book = xls.Workbook(xls_fname)
    sheet_test = book.add_worksheet("Test")
    sheet_test.write_column(0, 0, dataset.y_test.tolist())
    sheet_test.write_column(0, 1, y_test_hat.tolist())

    sheet_train = book.add_worksheet("Train")
    sheet_train.write_column(0, 0, dataset.y_train.tolist())
    sheet_train.write_column(0, 1, y_train_hat.tolist())


    book.close()

