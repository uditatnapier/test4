#    Mario Valerio Giuffrida - Multi-modal network definition
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

import numpy as np
import os
np.random.seed(0)

import Resnet50_counting_keras_vg as resnet50
import tensorflow as tf
import keras.backend.tensorflow_backend as tfback

#from keras import backend as K
from keras import layers as LL
#from keras.regularizers import l1,l2,Regularizer
from keras.regularizers import l2
#from keras.engine import Layer
from keras.engine.saving import load_weights_from_hdf5_group_by_name
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.merge import Maximum#,Concatenate as Concat

#from keras.preprocessing.image import ImageDataGenerator
from keras.models import Model
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,CSVLogger
from dataset import MultiInputImageDataGenerator

from h5py import File
from scipy import stats

def _get_available_gpus():
    """Get a list of available gpu devices (formatted as strings).

    # Returns
        A list of available GPU devices.
    """
    #global _LOCAL_DEVICES
    if tfback._LOCAL_DEVICES is None:
        devices = tf.config.list_logical_devices()
        tfback._LOCAL_DEVICES = [x.name for x in devices]
    return [x for x in tfback._LOCAL_DEVICES if 'device:gpu' in x.lower()]

tfback._get_available_gpus = _get_available_gpus



RGB_SHAPE = (240,240,3)
IR_SHAPE  = (273,243,3)
FMP_SHAPE  = (273,243,3)
DEPTH_SHAPE = (30,29,1)


class MultiModalLeafDeepCounter(object):
    Epochs = 50
    MiniBatchSize = 128
    LearingRate = 0.001
    L2RegulariserLambda = 0.01

    SaveStatus=False
    StatusFilename='Status<T>.npz'
    SaveEvery=10

    EarlyStop=True
    EarlyStopLookAhead=5
    EarlyStopParameters = []
    Finisher = 'classic'

    _validation_mse = []
    _training_mse = []

    _modality_order = ['rgb','ir','fmp','depth']

    _sizes = [1024,512,100]
    #_sizes = [1000, 250, 100]


    def __init__(this,rgb=True,ir=False,fmp=False,depth=False,l2_w=0.001,debug=False,finisher='classic',drop_prob=0,l2var_reg=1,**kwargs):

        branches = []

        this.rgb=rgb; this.ir=ir;this.fmp=fmp;this.depth=depth
        this.debug = debug
        this.Finisher = finisher

        this.L2RegulariserLambda = l2_w

        this._inputs = {}

        this.drop_probability = drop_prob

        if (rgb):

            branches.append(this._rgb_network())


        if (ir):
            branches.append(this._ir_network())


        if (fmp):

            branches.append(this._fmp_network())


        if (depth):
            branches.append(this._depth_network())


        if (len(branches)>1):
            combined = Maximum()(branches)
        else:
            combined=branches[0]


        if (this.Finisher=='classic'):
            fc1 = LL.core.Dense(this._sizes[1], activation='relu', activity_regularizer=l2(this.L2RegulariserLambda))(combined)
            output = LL.core.Dense(1)(fc1)
        elif (this.Finisher is None):
        	output = [combined] + [t for t in branches] 
        else:
            raise ValueError('Finisher %s not implemented'%this.Finisher)

        this._output = output

    def _rgb_network(this):
        layers = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=RGB_SHAPE)
        this._inputs['rgb'] = layers.input

        layers.layers[0].name='rgb_input'
        for i in range(1,len(layers.layers)):
            layers.layers[i].name = 'rgb_' + layers.layers[i].name

        x = layers.output

        x = LL.Flatten(name='rgb_flatten')(x)

        x = LL.core.Dense(this._sizes[0],activation=this._get_activation(),name='rgb_dense')(x)

        return x

    def _ir_network(this):
        layers = resnet50.ResNet50(include_top=False, weights='imagenet',input_shape=IR_SHAPE)
        this._inputs['ir'] = layers.input

        layers.layers[0].name='ir_input'

        for i in range(1,len(layers.layers)):
            layers.layers[i].name = 'ir_' + layers.layers[i].name

        x = layers.output
        x = LL.Flatten(name='ir_flatten')(x)
        x = LL.core.Dense(this._sizes[0], activation=this._get_activation(),name='ir_dense')(x)

        return x

    def _fmp_network(this):
        layers = resnet50.ResNet50(include_top=False,weights='imagenet',input_shape=FMP_SHAPE)
        this._inputs['fmp'] = layers.input

        layers.layers[0].name='fmp_input'

        for i in range(1,len(layers.layers)):
            layers.layers[i].name = 'fmp_' + layers.layers[i].name

        x = layers.output
        x = LL.Flatten(name='fmp_flatten')(x)
        x = LL.core.Dense(this._sizes[0], activation=this._get_activation(),name='fmp_dense')(x)

        return x

    def _depth_network(this):
        inp = LL.Input(shape=DEPTH_SHAPE)
        this._inputs['depth'] = inp

        layer= LL.Conv2D(32,(2,2),strides=(1,1),padding='valid',name='depth_conv')(inp)
        layer = LL.BatchNormalization(name='depth_bn')(layer)
        layer = LeakyReLU(name='depth_lrelu')(layer)
        layer = LL.Flatten(name='depth_flatten')(layer)
        layer = LL.core.Dense(this._sizes[0],activation=this._get_activation(),name='depth_dense')(layer)

        return layer


    def compile(this,loss='mse',**kwargs):
        model = Model(inputs=[this._inputs[x] for x in this._modality_order if x in this._inputs],outputs=this._output)
        model.compile(optimizer=Adam(lr=this.LearingRate),loss=loss)

        if (this.debug):
            model.summary()

        this._model = model

    def _get_activation(this):
        if this.Finisher=='valerio':
            return 'tanh'
        else:
            return 'relu'

    def test(this,x):
        y_hat = this._model.predict(x,this.MiniBatchSize)
        
        if (this.Finisher is not None):
        	return np.round(y_hat.flatten())
        else:
        	return y_hat



    def train(this,x_train,y_train,x_val=None,y_val=None,model_fname=None):
        x_aug = MultiInputImageDataGenerator(len(x_train),
            rotation_range=180,
            width_shift_range=0.1,
            height_shift_range=0.1,
            horizontal_flip=True,
            vertical_flip=True
        )

        x_aug.fit(x_train)

        callbacks = [ EarlyStopping(patience=this.EarlyStopLookAhead,verbose=1) ]

        if (model_fname is not None):
            callbacks.append(ModelCheckpoint(model_fname,verbose=1,save_best_only=True,save_weights_only=True,mode='min'))

            if (this.debug):

                path,fname = os.path.split(model_fname)

                csv_fname,_ = os.path.splitext(fname)
                csv_fname += '.csv'
                callbacks.append(CSVLogger(os.path.join(path,csv_fname)))

        #this._model.fit(x_train,y_train,epochs=this.Epochs,validation_data=(x_val,y_val),batch_size=this.MiniBatchSize,callbacks=[early_stop])

        this._model.fit_generator(x_aug.flow(x_train, y_train, seed=47,batch_size=this.MiniBatchSize),
                                  steps_per_epoch=812,epochs=this.Epochs, validation_data=(x_val, y_val),callbacks=callbacks)


    def save(this,fname):
        this._model.save_weights(fname)

    def load(this,fname,by_name=False):
        if (this.debug):
            print("Loading model {}".format(fname))
        this._model.load_weights(fname,by_name=by_name)

        if (this.debug):
            print("Model {} loaded".format(fname))

    def load_branch(this,filename,branch):
        if (branch not in this._modality_order):
            raise AttributeError('Invalid modality {}'.format(branch))

        f = File(filename,'r')

        wrapper = KerasModelFileWrapper(f,branch)

        load_weights_from_hdf5_group_by_name(wrapper,this._model.layers)





class KerasModelFileWrapper:

    def __init__(this,handle,prefix=None):
        this.handle=handle

        this.attrs = {t:this.handle.attrs[t] for t in this.handle.attrs.keys()}
        this.attrs['layer_names'] = [ t for t in this.attrs['layer_names'] if t.startswith(prefix)]


    def __getitem__(this, name):
        return this.handle[name]



def compute_metrics(Y,Y_hat):
    diff = Y - np.round(Y_hat)

    _, _, r_value, _,_ = stats.linregress(Y, Y_hat)

    res = {
        'dic': (diff.mean(),diff.std()),
        'abs_dic': ( np.abs(diff).mean(),np.abs(diff).std()),
        'mse' : (diff ** 2).mean(),
        '%' : np.mean(Y==Y_hat),
        'r2': r_value ** 2
    }

    return res

def print_metrics(metrics):
    print("DIC ....: {:0.2} ({:0.2})".format(*metrics['dic']))
    print("|DIC| ..: {:0.2} ({:0.2})".format(*metrics['abs_dic']))
    print("MSE ....: {:0.2}".format(metrics['mse']))
    print("Accuracy: {:0.2}".format(metrics['%']))
    print("R2 .....: {:0.2}".format(metrics['r2']))

def _my_print_fn(op, xin):
    for attr in op.attrs:
        temp = getattr(xin, attr)
        if callable(temp):
            pmsg = temp()
        else:
            pmsg = temp
    print(op.message, attr, '=', pmsg)
