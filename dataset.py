#    Mario Valerio Giuffrida - This file defines functions and classes to read data
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
import re
import csv
import glob
import os.path as path
import imageio as misc
import pandas as pd
import numpy as np
from keras.preprocessing.image import * #ImageDataGenerator,NumpyArrayIterator,array_to_img, transform_matrix_offset_center, random_channel_shift
from PIL import Image,ImageOps
from scipy import linalg
import warnings
import re

misc.imresize = lambda arr , sz: np.array(Image.fromarray(arr).resize(sz[0:2]))

class Dataset(object):
    def __init__(this,basepath=""):
        this.basepath = basepath
        this.Images = []

    def __add__(this, other):
        this.Images += other.Images
        return this

    def __len__(this):
        return len(this.Images)

    def __getitem__(this, key):
        return this.Images[key]

    def load(this,**kwargs):
        raise NotImplementedError

    def getData(this,modalities):
        N = len(this)
        idx = np.zeros((N,))

        if type(modalities) is not list:
            modalities = [modalities]

        for i in range(N):
            t = [ x in this[i] for x in modalities ]
            idx[i] = np.all(t)

        res = {x: np.stack((this.Images[i][x] for i in range(N) if idx[i] == True)) for x in modalities}

        return res

    def _pad_images(this,list):
        N = len(list)

        max_shape = max([list[x].shape for x in range(N)])

        for i in range(N):
            shape = list[i].shape

            diff = [ max_shape[k] - shape[k] for k in range(len(max_shape)) ]

            if (np.sum(diff)>0):
                T = np.zeros(shape=max_shape)
                diff = np.floor(np.asarray(diff)/2.).astype('int')
                slices = [slice(diff[k],diff[k]+shape[k]) for k in range(len(shape))]
                T[slices] = list[i]
                list[i] = T

        return list


class WageningenDataset(Dataset):
    def __init__(this,basepath,species='all'):
        super(WageningenDataset,this).__init__(basepath)
        this.__species__=[1,2]
        this.__subset__=['a','b']

        this.species=species
        this.subsets=['a,b']


    @property
    def species(this):
        return this.__species__

    @species.setter
    def species(this,value):
        if (value=='all'):
            value=[1,2]

        if (np.all([int(x) in [1,2] for x in value])):
            this.__species__=value
        else:
            raise ValueError('The value(s) {} is not valid for species'.format(value))

    @property
    def subset(this):
        return [ "" if x == "a" else x for x in this.__subset__ ]

    # @subset.setter
    # def subset(this, value):
    #     if (value == 'all'):
    #         value = ['a','b']
    #
    #     if (np.all([x in ['a', 'b'] for x in value])):
    #         this.__subset__ = value
    #     else:
    #         raise ValueError('The value(s) {} is not valid for species'.format(value))

    def load(this,resize=None,val_portion=0.2,seed=None,csv_format="Kale Species {}{}.csv"):

        np.random.seed(seed)

        stacked_filenames = []
        stacked_counts    = []

        max_id = 0

        this.indices={"train":[], "val":[],"test":[]}

        for s in this.species:
            for ss in this.subset:
                csv_filename = os.path.join(this.basepath,csv_format.format(s,ss))
                df = pd.read_csv(csv_filename,delimiter=',')

                stacked_filenames += [x.lower() for x in df.iloc[:, 0].to_list()]
                stacked_counts    += df.iloc[:, 1].to_list()

        image_fnames = glob.glob(os.path.join(this.basepath,"Day*/*/*.jpg"))

        j = 0

        for fn in image_fnames:
            path,name = os.path.split(fn)
            split = "test" if os.path.split(path)[0][-1] == "b" else "train"

            regex_res = re.search("D([0-9]){1,2}[b]{0,1}S([1-2]) \(([0-9]+)\)",name)
            metadata={'day': int(regex_res[1]), 'species':int(regex_res[2]), 'id': int(regex_res[3])}

            I = Image.open(fn)

            if (resize is not None):
                I = I.resize(resize, Image.ANTIALIAS)

            im = np.asarray(I)


            try:
                idx = stacked_filenames.index(name.lower())
                this.Images.append({'rgb':im,'count':stacked_counts[idx],"split":split,**metadata})
                this.indices[split].append(j)

                max_id = max(max_id,metadata['id'])

                j+=1
            except ValueError:

                warnings.warn("Image {} does not have a leaf count".format(fn))

        perm_idx = np.random.permutation(range(1,max_id+1))
        n = int( np.round(max_id * val_portion))

        val_idx = perm_idx[0:n]

        this.indices['val'] =  [ i for i in this.indices['test'] if this.Images[i]['id'] in val_idx ]
        this.indices['test'] = [ i for i in this.indices['test'] if this.Images[i]['id'] not in val_idx ]

    def __extract_data__(this,what,index_type):
        return np.stack([ this.Images[i][what] for i in this.indices[index_type] ])
    @property
    def x_train(this):
        return this.__extract_data__("rgb",'train')

    @property
    def y_train(this):
        return this.__extract_data__("count", 'train')

    @property
    def x_val(this):
        return this.__extract_data__("rgb", 'val')

    @property
    def y_val(this):
        return this.__extract_data__("count", 'val')

    @property
    def x_test(this):
        return this.__extract_data__("rgb", 'test')

    @property
    def y_test(this):
        return this.__extract_data__("count", 'test')

class CVPPPDataset(Dataset):
    def __init__(this,basepath,folders=range(1,5)):
        super(CVPPPDataset, this).__init__(basepath)
        this.folders=folders

    def load(this,**kwargs):
        new_size = kwargs.get('new_size',(250,250))

        for k in this.folders:
            folder = 'A'+str(k)
            path = os.path.join(this.basepath,folder)

            csv_handle = open(os.path.join(path,folder+'.csv'),'r')
            csv_reader = csv.reader(csv_handle,delimiter=',')

            for row in csv_reader:
                I = Image.open(os.path.join(path,row[0]))
                I.thumbnail(new_size,Image.ANTIALIAS)
                im = np.asarray(I)[:,:,0:3].transpose(2,0,1)
                count = int(row[1])

                this.Images.append({'rgb':im,'count':count})

            csv_handle.close()

        N = len(this.Images)

        for x in ['rgb']:
            l = [this.Images[i][x] for i in range(N)]
            l = this._pad_images(l)

            for i in range(N):
                this.Images[i][x] = l[i]


class MultiModalDataset(Dataset):

    def __init__(this,basepath="",plants="Arabidopsis"):
        super(MultiModalDataset,this).__init__(basepath)
        this.plants = plants

    def load(this,**kwargs):
        map = []
        this.modalities = []
        img_path = os.path.join(this.basepath,'Images',this.plants)
        #lbl_path = os.path.join(this.basepath, 'Labels', this.plants)

        for (path,dirnames,filenames) in os.walk(img_path):
            for f in filenames:
                if f.endswith('.png'):
                    fullfile = os.path.join(path,f)
                    s=f.rfind('_')
                    modality = f[s+1:-4]
                    I = np.array(Image.open(fullfile))

                    shape = I.shape
                    if len(shape)==2:
                        I = I.reshape((shape[0],shape[1],1))

                    I = np.transpose(I,(2,0,1))

                    k = f[0:s]

                    if k in map:
                        idx = map.index(k)
                        this.Images[idx][modality] = I
                    else:
                        m = re.search('day_([0-9]+)_hour_([0-9]+)',f)
                        day = int(m.group(1)); hour = int(m.group(2))/24.0
                        this.Images.append({modality:I, 'time':day+hour})
                        map.append(k)

                    if modality not in this.modalities:
                        this.modalities.append(modality)


                    fullfile = fullfile.replace("Images","Labels").replace(modality,'label_'+modality)
                    if (os.path.isfile(fullfile)):
                        I = Image.open(fullfile)
                        I = np.asarray(I)

                        idx = map.index(k)
                        this.Images[idx]['label_'+modality] = I
                        this.Images[idx]['count'] = len(np.unique(I))-1

        N = len(this.Images)

        for x in this.modalities:
            l = [this.Images[i][x] for i in range(N)]
            l = this._pad_images(l)

            for i in range(N):
                this.Images[i][x] = l[i]

        this.Images = sorted(this.Images,key=lambda x:x['time'])


def dataset_augmentation(X,rotations=range(0,360,90),flip=True):
    X = X.transpose((2,3,1,0))

    N = X.shape[-1] * len(rotations) * (3 if flip else 1)

    X_aug = np.zeros( ( X.shape[0],X.shape[1],X.shape[2], N) )

    flips = (range(3) if flip else range(1))

    k = 0

    for f in flips:
        for theta in rotations:
            for i in range(X.shape[-1]):
                mode = None

                img = X[:, :, :, i]
                prev = img.dtype


                if (theta!=0):

                    if img.shape[-1]==1:
                        mode = 'L'
                        img = img[:,:,0]

                    img = np.asarray(img, dtype='uint8')
                    I = Image.fromarray(img,mode=mode)
                    img = np.asarray(I.rotate(theta),dtype=prev)

                    if (len(img.shape)<3):
                        img = img.reshape((img.shape[0],img.shape[1],1))

                if f==1: # h flip
                    img = img[-1::-1,:,:]
                elif f==2: # v flip
                    img = img[:,-1::-1, :]

                X_aug[:,:,:,k] = img

                k+=1

    return X_aug.transpose((3,2,0,1))


def get_filename(p):
    _, t = path.split(p)
    return t


def get_last_dir(p):
    h, _ = path.split(p)
    _, t = path.split(h)
    return t[:2]


def get_data_masks(split_load, basepath='CVPPP2017_LCC_training/TrainingSplits/'):
        ###############################
        # Getting images (x data)	  #
        ###############################
    imgname_train_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(h) + '/*fg.png')) for h in split_load[0]])
    imgname_train_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(h) + '/*fg.png')) for h in split_load[0]])
    imgname_train_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(h) + '/*fg.png')) for h in split_load[0]])
    imgname_train_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(h) + '/*fg.png')) for h in split_load[0]])

    imgname_val_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(split_load[1])) + '/*fg.png')])
    imgname_val_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(split_load[1])) + '/*fg.png')])
    imgname_val_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(split_load[1])) + '/*fg.png')])
    imgname_val_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(split_load[1])) + '/*fg.png')])

    imgname_test_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(split_load[2])) + '/*fg.png')])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(split_load[2])) + '/*fg.png')])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(split_load[2])) + '/*fg.png')])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(split_load[2])) + '/*fg.png')])

    filelist_train_A1 = list(np.sort(imgname_train_A1.flat))
    filelist_train_A2 = list(np.sort(imgname_train_A2.flat))
    filelist_train_A3 = list(np.sort(imgname_train_A3.flat))
    filelist_train_A4 = list(np.sort(imgname_train_A4.flat))


    filelist_train_A1_img = np.array([np.array(get_filename(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_img = np.array([np.array(get_filename(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_img = np.array([np.array(get_filename(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_img = np.array([np.array(get_filename(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])

    filelist_train_A1_set = np.array([np.array(get_last_dir(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_set = np.array([np.array(get_last_dir(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_set = np.array([np.array(get_last_dir(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_set = np.array([np.array(get_last_dir(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])

    filelist_val_A1 = list(np.sort(imgname_val_A1.flat))
    filelist_val_A2 = list(np.sort(imgname_val_A2.flat))
    filelist_val_A3 = list(np.sort(imgname_val_A3.flat))
    filelist_val_A4 = list(np.sort(imgname_val_A4.flat))
    filelist_val_A1_img = np.array([np.array(get_filename(filelist_val_A1[h])) for h in range(len(filelist_val_A1))])
    filelist_val_A2_img = np.array([np.array(get_filename(filelist_val_A2[h])) for h in range(len(filelist_val_A2))])
    filelist_val_A3_img = np.array([np.array(get_filename(filelist_val_A3[h])) for h in range(len(filelist_val_A3))])
    filelist_val_A4_img = np.array([np.array(get_filename(filelist_val_A4[h])) for h in range(len(filelist_val_A4))])
    filelist_val_A1_set = np.array([np.array(get_last_dir(filelist_val_A1[h])) for h in range(len(filelist_val_A1))])
    filelist_val_A2_set = np.array([np.array(get_last_dir(filelist_val_A2[h])) for h in range(len(filelist_val_A2))])
    filelist_val_A3_set = np.array([np.array(get_last_dir(filelist_val_A3[h])) for h in range(len(filelist_val_A3))])
    filelist_val_A4_set = np.array([np.array(get_last_dir(filelist_val_A4[h])) for h in range(len(filelist_val_A4))])

    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])

    # Read image names into np array train
    x_train_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A1])   
    x_train_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A2])
    x_train_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A3])
    x_train_A3 = x_train_A3[:,:,:,np.newaxis]; x_train_A3 = np.repeat(x_train_A3,3,3)
    x_train_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A4])
    x_train_A4 = x_train_A4[:,:,:,np.newaxis]; x_train_A4 = np.repeat(x_train_A4,3,3)

    # Read image names into np array validation
    x_val_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A1])
    #x_val_A1 = np.delete(x_va283l_A1, 3, 3)
    x_val_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A2])
    #x_val_A2 = np.delete(x_val_A2, 3, 3)
    x_val_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A3])
    x_val_A3 = x_val_A3[:,:,:,np.newaxis]; x_val_A3 = np.repeat(x_val_A3,3,3)
    x_val_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A4])
    x_val_A4 = x_val_A4[:,:,:,np.newaxis]; x_val_A4 = np.repeat(x_val_A4,3,3)

    # Read image names into np array test
    x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
    x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
    x_test_A2 = x_test_A2[:,:,:,np.newaxis]; x_test_A2 = np.repeat(x_test_A2,3,3)
    x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
    x_test_A3 = x_test_A3[:,:,:,np.newaxis]; x_test_A3 = np.repeat(x_test_A3,3,3)
    x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])
    x_test_A4 = x_test_A4[:,:,:,np.newaxis]; x_test_A4 = np.repeat(x_test_A4,3,3)

    x_train_res_A1 = np.array([misc.imresize(x_train_A1[i], [320, 320, 3]) for i in range(0, len(x_train_A1))])
    x_train_res_A2 = np.array([misc.imresize(x_train_A2[i], [320, 320, 3]) for i in range(0, len(x_train_A2))])
    x_train_res_A3 = np.array([misc.imresize(x_train_A3[i], [320, 320, 3]) for i in range(0, len(x_train_A3))])
    x_train_res_A4 = np.array([misc.imresize(x_train_A4[i], [320, 320, 3]) for i in range(0, len(x_train_A4))])

    x_val_res_A1 = np.array([misc.imresize(x_val_A1[i], [320, 320, 3]) for i in range(0, len(x_val_A1))])
    x_val_res_A2 = np.array([misc.imresize(x_val_A2[i], [320, 320, 3]) for i in range(0, len(x_val_A2))])
    x_val_res_A3 = np.array([misc.imresize(x_val_A3[i], [320, 320, 3]) for i in range(0, len(x_val_A3))])
    x_val_res_A4 = np.array([misc.imresize(x_val_A4[i], [320, 320, 3]) for i in range(0, len(x_val_A4))])

    x_test_res_A1 = np.array([misc.imresize(x_test_A1[i], [320, 320, 3]) for i in range(0, len(x_test_A1))])
    x_test_res_A2 = np.array([misc.imresize(x_test_A2[i], [320, 320, 3]) for i in range(0, len(x_test_A2))])
    x_test_res_A3 = np.array([misc.imresize(x_test_A3[i], [320, 320, 3]) for i in range(0, len(x_test_A3))])
    x_test_res_A4 = np.array([misc.imresize(x_test_A4[i], [320, 320, 3]) for i in range(0, len(x_test_A4))])

    x_train_res_A1 = x_train_res_A1[..., np.newaxis].repeat(3, axis=3)
    x_train_res_A2 = x_train_res_A2[..., np.newaxis].repeat(3, axis=3)
    x_val_res_A1   = x_val_res_A1  [..., np.newaxis].repeat(3, axis=3)
    x_val_res_A2   = x_val_res_A2  [..., np.newaxis].repeat(3, axis=3)
    x_test_res_A1  = x_test_res_A1 [..., np.newaxis].repeat(3, axis=3)
    #x_test_res_A2  = x_test_res_A2 [..., np.newaxis].repeat(3, axis=3)

    x_train_all = np.concatenate((x_train_res_A1, x_train_res_A2, x_train_res_A3, x_train_res_A4), axis=0)
    x_val_all = np.concatenate((x_val_res_A1, x_val_res_A2, x_val_res_A3, x_val_res_A4), axis=0)
    x_test_all = np.concatenate((x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4), axis=0)

    # Concatenate the image names
    x_train_img = np.concatenate(
        (filelist_train_A1_img, filelist_train_A2_img, filelist_train_A3_img, filelist_train_A4_img), axis=0)
    x_val_img = np.concatenate((filelist_val_A1_img, filelist_val_A2_img, filelist_val_A3_img, filelist_val_A4_img), axis=0)
    x_test_img = np.concatenate((filelist_test_A1_img, filelist_test_A2_img, filelist_test_A3_img, filelist_test_A4_img),
                                axis=0)

    x_train_set = np.concatenate(
        (filelist_train_A1_set, filelist_train_A2_set, filelist_train_A3_set, filelist_train_A4_set), axis=0)
    x_val_set = np.concatenate((filelist_val_A1_set, filelist_val_A2_set, filelist_val_A3_set, filelist_val_A4_set), axis=0)
    x_test_set = np.concatenate((filelist_test_A1_set, filelist_test_A2_set, filelist_test_A3_set, filelist_test_A4_set),
                                axis=0)



    return x_train_all, x_val_all, x_test_all, x_train_set, x_val_set, x_test_set, x_train_img, x_val_img, x_test_img


def get_data(split_load, basepath='CVPPP2017_LCC_training/TrainingSplits/'):

    imgname_train_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(h) + '/*rgb.png')) for h in split_load[0]])
    imgname_train_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(h) + '/*rgb.png')) for h in split_load[0]])

    imgname_val_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(split_load[1])) + '/*rgb.png')])
    imgname_val_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(split_load[1])) + '/*rgb.png')])
    imgname_val_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(split_load[1])) + '/*rgb.png')])
    imgname_val_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(split_load[1])) + '/*rgb.png')])

    imgname_test_A1 = np.array([glob.glob(path.join(basepath, 'A1' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath, 'A2' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath, 'A3' + str(split_load[2])) + '/*rgb.png')])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath, 'A4' + str(split_load[2])) + '/*rgb.png')])

    filelist_train_A1 = list(np.sort(imgname_train_A1.flat))
    filelist_train_A2 = list(np.sort(imgname_train_A2.flat))
    filelist_train_A3 = list(np.sort(imgname_train_A3.flat))
    filelist_train_A4 = list(np.sort(imgname_train_A4.flat))


    filelist_train_A1_img = np.array([np.array(get_filename(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_img = np.array([np.array(get_filename(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_img = np.array([np.array(get_filename(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_img = np.array([np.array(get_filename(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])

    filelist_train_A1_set = np.array([np.array(get_last_dir(filelist_train_A1[h])) for h in range(len(filelist_train_A1))])
    filelist_train_A2_set = np.array([np.array(get_last_dir(filelist_train_A2[h])) for h in range(len(filelist_train_A2))])
    filelist_train_A3_set = np.array([np.array(get_last_dir(filelist_train_A3[h])) for h in range(len(filelist_train_A3))])
    filelist_train_A4_set = np.array([np.array(get_last_dir(filelist_train_A4[h])) for h in range(len(filelist_train_A4))])

    filelist_val_A1 = list(np.sort(imgname_val_A1.flat))
    filelist_val_A2 = list(np.sort(imgname_val_A2.flat))
    filelist_val_A3 = list(np.sort(imgname_val_A3.flat))
    filelist_val_A4 = list(np.sort(imgname_val_A4.flat))
    filelist_val_A1_img = np.array([np.array(get_filename(filelist_val_A1[h])) for h in range(len(filelist_val_A1))])
    filelist_val_A2_img = np.array([np.array(get_filename(filelist_val_A2[h])) for h in range(len(filelist_val_A2))])
    filelist_val_A3_img = np.array([np.array(get_filename(filelist_val_A3[h])) for h in range(len(filelist_val_A3))])
    filelist_val_A4_img = np.array([np.array(get_filename(filelist_val_A4[h])) for h in range(len(filelist_val_A4))])
    filelist_val_A1_set = np.array([np.array(get_last_dir(filelist_val_A1[h])) for h in range(len(filelist_val_A1))])
    filelist_val_A2_set = np.array([np.array(get_last_dir(filelist_val_A2[h])) for h in range(len(filelist_val_A2))])
    filelist_val_A3_set = np.array([np.array(get_last_dir(filelist_val_A3[h])) for h in range(len(filelist_val_A3))])
    filelist_val_A4_set = np.array([np.array(get_last_dir(filelist_val_A4[h])) for h in range(len(filelist_val_A4))])

    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])

    # Read image names into np array train
    x_train_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A1])
    x_train_A1 = np.delete(x_train_A1, 3, 3)
    x_train_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_train_A2])
    x_train_A2 = np.delete(x_train_A2, 3, 3)
    x_train_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A3])
    x_train_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_train_A4])


    # Read image names into np array validation
    x_val_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A1])
    x_val_A1 = np.delete(x_val_A1, 3, 3)
    x_val_A2 = np.array([np.array(misc.imread(fname)) for fname in filelist_val_A2])
    x_val_A2 = np.delete(x_val_A2, 3, 3)
    x_val_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A3])
    x_val_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_val_A4])

    # Read image names into np array test
    x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
    x_test_A1 = np.delete(x_test_A1, 3, 3)
    x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
    x_test_A2 = np.delete(x_test_A2, 3, 3)
    x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
    x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])

    x_train_res_A1 = np.array([misc.imresize(x_train_A1[i], [320, 320, 3]) for i in range(0, len(x_train_A1))])
    x_train_res_A2 = np.array([misc.imresize(x_train_A2[i], [320, 320, 3]) for i in range(0, len(x_train_A2))])
    x_train_res_A3 = np.array([misc.imresize(x_train_A3[i], [320, 320, 3]) for i in range(0, len(x_train_A3))])
    x_train_res_A4 = np.array([misc.imresize(x_train_A4[i], [320, 320, 3]) for i in range(0, len(x_train_A4))])

    x_val_res_A1 = np.array([misc.imresize(x_val_A1[i], [320, 320, 3]) for i in range(0, len(x_val_A1))])
    x_val_res_A2 = np.array([misc.imresize(x_val_A2[i], [320, 320, 3]) for i in range(0, len(x_val_A2))])
    x_val_res_A3 = np.array([misc.imresize(x_val_A3[i], [320, 320, 3]) for i in range(0, len(x_val_A3))])
    x_val_res_A4 = np.array([misc.imresize(x_val_A4[i], [320, 320, 3]) for i in range(0, len(x_val_A4))])

    x_test_res_A1 = np.array([misc.imresize(x_test_A1[i], [320, 320, 3]) for i in range(0, len(x_test_A1))])
    x_test_res_A2 = np.array([misc.imresize(x_test_A2[i], [320, 320, 3]) for i in range(0, len(x_test_A2))])
    x_test_res_A3 = np.array([misc.imresize(x_test_A3[i], [320, 320, 3]) for i in range(0, len(x_test_A3))])
    x_test_res_A4 = np.array([misc.imresize(x_test_A4[i], [320, 320, 3]) for i in range(0, len(x_test_A4))])

    x_train_all = np.concatenate((x_train_res_A1, x_train_res_A2, x_train_res_A3, x_train_res_A4), axis=0)
    x_val_all = np.concatenate((x_val_res_A1, x_val_res_A2, x_val_res_A3, x_val_res_A4), axis=0)
    x_test_all = np.concatenate((x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4), axis=0)

    # Histogram stretching
    for h in range(0, len(x_train_all)):
        x_img = x_train_all[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_train_all[h] = x_img_ar

    for h in range(0, len(x_val_all)):
        x_img = x_val_all[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_val_all[h] = x_img_ar

    for h in range(0, len(x_test_all)):
        x_img = x_test_all[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_all[h] = x_img_ar

    # Concatenate the image names
    x_train_img = np.concatenate(
        (filelist_train_A1_img, filelist_train_A2_img, filelist_train_A3_img, filelist_train_A4_img), axis=0)
    x_val_img = np.concatenate((filelist_val_A1_img, filelist_val_A2_img, filelist_val_A3_img, filelist_val_A4_img), axis=0)
    x_test_img = np.concatenate((filelist_test_A1_img, filelist_test_A2_img, filelist_test_A3_img, filelist_test_A4_img),
                                axis=0)

    x_train_set = np.concatenate(
        (filelist_train_A1_set, filelist_train_A2_set, filelist_train_A3_set, filelist_train_A4_set), axis=0)
    x_val_set = np.concatenate((filelist_val_A1_set, filelist_val_A2_set, filelist_val_A3_set, filelist_val_A4_set), axis=0)
    x_test_set = np.concatenate((filelist_test_A1_set, filelist_test_A2_set, filelist_test_A3_set, filelist_test_A4_set),
                                axis=0)

    ###############################
    # Getting targets (y data)    #
    ###############################
    counts_A1 = np.array([glob.glob(path.join(basepath, 'A1.xlsx'))])
    counts_A2 = np.array([glob.glob(path.join(basepath, 'A2.xlsx'))])
    counts_A3 = np.array([glob.glob(path.join(basepath, 'A3.xlsx'))])
    counts_A4 = np.array([glob.glob(path.join(basepath, 'A4.xlsx'))])

    counts_train_flat_A1 = list(counts_A1.flat)
    train_labels_A1 = pd.DataFrame()
    y_train_A1_list = []
    y_val_A1_list = []
    y_test_A1_list = []
    for f in counts_train_flat_A1:
        frame = pd.read_excel(f, header=None)
        train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
    all_labels_A1 = np.array(train_labels_A1)

    for j in filelist_train_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_train_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_train_A1_labels = np.concatenate(y_train_A1_list, axis=0)

    for j in filelist_val_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_val_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_val_A1_labels = np.concatenate(y_val_A1_list, axis=0)

    for j in filelist_test_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_test_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)

    counts_train_flat_A2 = list(counts_A2.flat)
    train_labels_A2 = pd.DataFrame()
    y_train_A2_list = []
    y_val_A2_list = []
    y_test_A2_list = []
    for f in counts_train_flat_A2:
        frame = pd.read_excel(f, header=None)
        train_labels_A2 = train_labels_A2.append(frame, ignore_index=False)
    all_labels_A2 = np.array(train_labels_A2)

    for j in filelist_train_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_train_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_train_A2_labels = np.concatenate(y_train_A2_list, axis=0)

    for j in filelist_val_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_val_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_val_A2_labels = np.concatenate(y_val_A2_list, axis=0)

    for j in filelist_test_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_test_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_test_A2_labels = np.concatenate(y_test_A2_list, axis=0)

    counts_train_flat_A3 = list(counts_A3.flat)
    train_labels_A3 = pd.DataFrame()
    y_train_A3_list = []
    y_val_A3_list = []
    y_test_A3_list = []
    for f in counts_train_flat_A3:
        frame = pd.read_excel(f, header=None)
        train_labels_A3 = train_labels_A3.append(frame, ignore_index=False)
    all_labels_A3 = np.array(train_labels_A3)

    for j in filelist_train_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_train_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_train_A3_labels = np.concatenate(y_train_A3_list, axis=0)

    for j in filelist_val_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_val_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_val_A3_labels = np.concatenate(y_val_A3_list, axis=0)

    for j in filelist_test_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_test_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_test_A3_labels = np.concatenate(y_test_A3_list, axis=0)

    counts_train_flat_A4 = list(counts_A4.flat)
    train_labels_A4 = pd.DataFrame()
    y_train_A4_list = []
    y_val_A4_list = []
    y_test_A4_list = []
    for f in counts_train_flat_A4:
        frame = pd.read_excel(f, header=None)
        train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
    all_labels_A4 = np.array(train_labels_A4)

    for j in filelist_train_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_train_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_train_A4_labels = np.concatenate(y_train_A4_list, axis=0)

    for j in filelist_val_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_val_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_val_A4_labels = np.concatenate(y_val_A4_list, axis=0)

    for j in filelist_test_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_test_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)

    y_train_all_labels = np.concatenate((y_train_A1_labels, y_train_A2_labels, y_train_A3_labels, y_train_A4_labels),
                                        axis=0)
    y_val_all_labels = np.concatenate((y_val_A1_labels, y_val_A2_labels, y_val_A3_labels, y_val_A4_labels), axis=0)
    y_test_all_labels = np.concatenate((y_test_A1_labels, y_test_A2_labels, y_test_A3_labels, y_test_A4_labels), axis=0)

    y_train_all = y_train_all_labels[:, 1]
    y_val_all = y_val_all_labels[:, 1]
    y_test_all = y_test_all_labels[:, 1]


    return x_train_all, x_val_all, x_test_all, y_train_all, y_val_all, y_test_all, x_train_set, x_val_set, x_test_set, x_train_img, x_val_img, x_test_img

def get_data_testing_mask(basepath='CVPPP2017_testing/testing'):
    ###############################
    # Getting images (x data)	  #
    ###############################


    imgname_test_A1 = np.array([glob.glob(path.join(basepath,'A1','*fg.png'))])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath,'A2','*fg.png'))])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath,'A3','*fg.png'))])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath,'A4','*fg.png'))])
    imgname_test_A5 = np.array([glob.glob(path.join(basepath,'A5','*fg.png'))])


    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A5 = list(np.sort(imgname_test_A5.flat))


    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_img = np.array([np.array(get_filename(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_set = np.array([np.array(get_last_dir(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])

    x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
    x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
    x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
    x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])
    x_test_A5 = [np.array(Image.open(fname)) for fname in filelist_test_A5]

    x_test_A5 = [x[:,:,np.newaxis] if len(x.shape)==2 else x for x in x_test_A5]
    x_test_A5 = [np.repeat(x,3,2) if x.shape[2] == 1 else x for x in x_test_A5]
    x_test_A5 = [misc.imresize(x_test_A5[i],[320,320,3]) for i in range(len(x_test_A5))]

    x_test_A2 = x_test_A2[:, :, :, np.newaxis]; x_test_A2 = np.repeat(x_test_A2, 3, 3)
    x_test_A3 = x_test_A3[:, :, :, np.newaxis];x_test_A3 = np.repeat(x_test_A3, 3, 3)
    x_test_A4 = x_test_A4[:, :, :, np.newaxis];x_test_A4 = np.repeat(x_test_A4, 3, 3)

    x_test_res_A1 = np.array([misc.imresize(x_test_A1[i],[320,320,3]) for i in range(0,len(x_test_A1))])
    x_test_res_A2 = np.array([misc.imresize(x_test_A2[i],[320,320,3]) for i in range(0,len(x_test_A2))])
    x_test_res_A3 = np.array([misc.imresize(x_test_A3[i],[320,320,3]) for i in range(0,len(x_test_A3))])
    x_test_res_A4 = np.array([misc.imresize(x_test_A4[i],[320,320,3]) for i in range(0,len(x_test_A4))])
    x_test_res_A5 = np.array(x_test_A5)
    #x_test_res_A5 = np.array([misc.imresize(x_test_A5[i],[320,320,3]) for i in range(0,len(x_test_A5))])



    x_test_all = np.concatenate((x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4,x_test_res_A5), axis=0)
    test_sets  = np.concatenate((filelist_test_A1_set,filelist_test_A2_set,filelist_test_A3_set,filelist_test_A4_set,filelist_test_A5_set))



    return x_test_all,test_sets


def get_data_testing(basepath='CVPPP2017_testing/testing'):
    ###############################
    # Getting images (x data)	  #
    ###############################


    imgname_test_A1 = np.array([glob.glob(path.join(basepath,'A1','*rgb.png'))])
    imgname_test_A2 = np.array([glob.glob(path.join(basepath,'A2','*rgb.png'))])
    imgname_test_A3 = np.array([glob.glob(path.join(basepath,'A3','*rgb.png'))])
    imgname_test_A4 = np.array([glob.glob(path.join(basepath,'A4','*rgb.png'))])
    imgname_test_A5 = np.array([glob.glob(path.join(basepath,'A5','*rgb.png'))])


    filelist_test_A1 = list(np.sort(imgname_test_A1.flat))
    filelist_test_A2 = list(np.sort(imgname_test_A2.flat))
    filelist_test_A3 = list(np.sort(imgname_test_A3.flat))
    filelist_test_A4 = list(np.sort(imgname_test_A4.flat))
    filelist_test_A5 = list(np.sort(imgname_test_A5.flat))


    filelist_test_A1_img = np.array([np.array(get_filename(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_img = np.array([np.array(get_filename(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_img = np.array([np.array(get_filename(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_img = np.array([np.array(get_filename(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_img = np.array([np.array(get_filename(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])
    filelist_test_A1_set = np.array([np.array(get_last_dir(filelist_test_A1[h])) for h in range(len(filelist_test_A1))])
    filelist_test_A2_set = np.array([np.array(get_last_dir(filelist_test_A2[h])) for h in range(len(filelist_test_A2))])
    filelist_test_A3_set = np.array([np.array(get_last_dir(filelist_test_A3[h])) for h in range(len(filelist_test_A3))])
    filelist_test_A4_set = np.array([np.array(get_last_dir(filelist_test_A4[h])) for h in range(len(filelist_test_A4))])
    filelist_test_A5_set = np.array([np.array(get_last_dir(filelist_test_A5[h])) for h in range(len(filelist_test_A5))])

    x_test_A1 = np.array([np.array(misc.imread(fname)) for fname in filelist_test_A1])
    x_test_A1 = np.delete(x_test_A1,3,3)
    x_test_A2 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A2])
    x_test_A2 = np.delete(x_test_A2,3,3)
    x_test_A3 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A3])
    #x_test_A3 = np.delete(x_test_A3, 3, 3)
    x_test_A4 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A4])
    #x_test_A4 = np.delete(x_test_A4, 3, 3)
    x_test_A5 = np.array([np.array(Image.open(fname)) for fname in filelist_test_A5])
    #x_test_A5 = np.delete(x_test_A5, 3, 3)

    for i in range(len(x_test_A5)):
        x_A5_img = x_test_A5[i]
        if x_A5_img.shape[2] == 4:
            x_A5_img_del = np.delete(x_A5_img,3,2)
            x_test_A5[i] = x_A5_img_del

    x_test_res_A1 = np.array([misc.imresize(x_test_A1[i],[320,320,3]) for i in range(0,len(x_test_A1))])
    x_test_res_A2 = np.array([misc.imresize(x_test_A2[i],[320,320,3]) for i in range(0,len(x_test_A2))])
    x_test_res_A3 = np.array([misc.imresize(x_test_A3[i],[320,320,3]) for i in range(0,len(x_test_A3))])
    x_test_res_A4 = np.array([misc.imresize(x_test_A4[i],[320,320,3]) for i in range(0,len(x_test_A4))])
    x_test_res_A5 = np.array([misc.imresize(x_test_A5[i],[320,320,3]) for i in range(0,len(x_test_A5))])

    for h in range(0,len(x_test_res_A1)):
        x_img = x_test_res_A1[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_res_A1[h] = x_img_ar

    for h in range(0,len(x_test_res_A2)):
        x_img = x_test_res_A2[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_res_A2[h] = x_img_ar

    for h in range(0,len(x_test_res_A3)):
        x_img = x_test_res_A3[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_res_A3[h] = x_img_ar

    for h in range(0,len(x_test_res_A4)):
        x_img = x_test_res_A4[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_res_A4[h] = x_img_ar

    for h in range(0,len(x_test_res_A5)):
        x_img = x_test_res_A5[h]
        x_img_pil = Image.fromarray(x_img)
        x_img_pil = ImageOps.autocontrast(x_img_pil)
        x_img_ar = np.array(x_img_pil)
        x_test_res_A5[h] = x_img_ar

    ###############################
    # Getting targets (y data)	  #
    ###############################
    counts_A1 = np.array([glob.glob(path.join(basepath, 'A1/A1.xlsx'))])
    counts_A2 = np.array([glob.glob(path.join(basepath, 'A2/A2.xlsx'))])
    counts_A3 = np.array([glob.glob(path.join(basepath, 'A3/A3.xlsx'))])
    counts_A4 = np.array([glob.glob(path.join(basepath, 'A4/A4.xlsx'))])
    counts_A5 = np.array([glob.glob(path.join(basepath, 'A5/A5.xlsx'))])

    counts_train_flat_A1 = list(counts_A1.flat)
    train_labels_A1 = pd.DataFrame()
    y_test_A1_list = []
    for f in counts_train_flat_A1:
        frame = pd.read_excel(f, header=None)
        train_labels_A1 = train_labels_A1.append(frame, ignore_index=False)
    all_labels_A1 = np.array(train_labels_A1)

    for j in filelist_test_A1_img:
        arr_idx = np.where(all_labels_A1 == j)
        y_test_A1_list.append(all_labels_A1[arr_idx[0], :])
    y_test_A1_labels = np.concatenate(y_test_A1_list, axis=0)

    counts_train_flat_A2 = list(counts_A2.flat)
    train_labels_A2 = pd.DataFrame()
    y_test_A2_list = []
    for f in counts_train_flat_A2:
        frame = pd.read_excel(f, header=None)
        train_labels_A2 = train_labels_A2.append(frame, ignore_index=False)
    all_labels_A2 = np.array(train_labels_A2)


    for j in filelist_test_A2_img:
        arr_idx = np.where(all_labels_A2 == j)
        y_test_A2_list.append(all_labels_A2[arr_idx[0], :])
    y_test_A2_labels = np.concatenate(y_test_A2_list, axis=0)

    counts_train_flat_A3 = list(counts_A3.flat)
    train_labels_A3 = pd.DataFrame()
    y_test_A3_list = []
    for f in counts_train_flat_A3:
        frame = pd.read_excel(f, header=None)
        train_labels_A3 = train_labels_A3.append(frame, ignore_index=False)
    all_labels_A3 = np.array(train_labels_A3)


    for j in filelist_test_A3_img:
        arr_idx = np.where(all_labels_A3 == j)
        y_test_A3_list.append(all_labels_A3[arr_idx[0], :])
    y_test_A3_labels = np.concatenate(y_test_A3_list, axis=0)

    counts_train_flat_A4 = list(counts_A4.flat)
    train_labels_A4 = pd.DataFrame()
    y_test_A4_list = []
    for f in counts_train_flat_A4:
        frame = pd.read_excel(f, header=None)
        train_labels_A4 = train_labels_A4.append(frame, ignore_index=False)
    all_labels_A4 = np.array(train_labels_A4)

    for j in filelist_test_A4_img:
        arr_idx = np.where(all_labels_A4 == j)
        y_test_A4_list.append(all_labels_A4[arr_idx[0], :])
    y_test_A4_labels = np.concatenate(y_test_A4_list, axis=0)


    #################
    counts_train_flat_A5 = list(counts_A5.flat)
    train_labels_A5 = pd.DataFrame()
    y_test_A5_list = []
    for f in counts_train_flat_A5:
        frame = pd.read_excel(f, header=None)
        train_labels_A5 = train_labels_A5.append(frame, ignore_index=False)
    all_labels_A5 = np.array(train_labels_A5)

    for j in filelist_test_A5_img:
        arr_idx = np.where(all_labels_A5 == j)
        y_test_A5_list.append(all_labels_A5[arr_idx[0], :])
    y_test_A5_labels = np.concatenate(y_test_A5_list, axis=0)
    #################


    x_test_all = np.concatenate((x_test_res_A1, x_test_res_A2, x_test_res_A3, x_test_res_A4, x_test_res_A5), axis=0)
    y_test_all = np.concatenate((y_test_A1_labels, y_test_A2_labels, y_test_A3_labels, y_test_A4_labels,y_test_A5_labels),axis=0)
    test_sets  = np.concatenate((filelist_test_A1_set,filelist_test_A2_set,filelist_test_A3_set,filelist_test_A4_set,filelist_test_A5_set))


    y_test_all = y_test_all[:, 1]


    return x_test_all,y_test_all,test_sets

# class MultiInputImageDataGenerator(ImageDataGenerator):
#     """Generate minibatches of image data with real-time data augmentation.
#     # Arguments
#         featurewise_center: set input mean to 0 over the dataset.
#         samplewise_center: set each sample mean to 0.
#         featurewise_std_normalization: divide inputs by std of the dataset.
#         samplewise_std_normalization: divide each input by its std.
#         zca_whitening: apply ZCA whitening.
#         zca_epsilon: epsilon for ZCA whitening. Default is 1e-6.
#         rotation_range: degrees (0 to 180).
#         width_shift_range: fraction of total width.
#         height_shift_range: fraction of total height.
#         shear_range: shear intensity (shear angle in radians).
#         zoom_range: amount of zoom. if scalar z, zoom will be randomly picked
#             in the range [1-z, 1+z]. A sequence of two can be passed instead
#             to select this range.
#         channel_shift_range: shift range for each channel.
#         fill_mode: points outside the boundaries are filled according to the
#             given mode ('constant', 'nearest', 'reflect' or 'wrap'). Default
#             is 'nearest'.
#         cval: value used for points outside the boundaries when fill_mode is
#             'constant'. Default is 0.
#         horizontal_flip: whether to randomly flip images horizontally.
#         vertical_flip: whether to randomly flip images vertically.
#         rescale: rescaling factor. If None or 0, no rescaling is applied,
#             otherwise we multiply the data by the value provided. This is
#             applied after the `preprocessing_function` (if any provided)
#             but before any other transformation.
#         preprocessing_function: function that will be implied on each input.
#             The function will run before any other modification on it.
#             The function should take one argument:
#             one image (Numpy tensor with rank 3),
#             and should output a Numpy tensor with the same shape.
#         data_format: 'channels_first' or 'channels_last'. In 'channels_first' mode, the channels dimension
#             (the depth) is at index 1, in 'channels_last' mode it is at index 3.
#             It defaults to the `image_data_format` value found in your
#             Keras config file at `~/.keras/keras.json`.
#             If you never set it, then it will be "channels_last".
#     """
#
#
#
#     def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None,
#              save_to_dir=None, save_prefix='', save_format='png',drop_probability=0):
#         return MultiInputNumpyArrayIterator(
#             x, y, self,
#             batch_size=batch_size,
#             shuffle=shuffle,
#             seed=seed,
#             data_format=self.data_format,
#             save_to_dir=save_to_dir,
#             save_prefix=save_prefix,
#             save_format=save_format,
#             drop_probability=drop_probability)
#
#     def flow_from_directory(self, directory,
#                             target_size=(256, 256), color_mode='rgb',
#                             classes=None, class_mode='categorical',
#                             batch_size=32, shuffle=True, seed=None,
#                             save_to_dir=None,
#                             save_prefix='',
#                             save_format='png',
#                             follow_links=False):
#         raise NotImplementedError()
#         # return DirectoryIterator(
#         #     directory, self,
#         #     target_size=target_size, color_mode=color_mode,
#         #     classes=classes, class_mode=class_mode,
#         #     data_format=self.data_format,
#         #     batch_size=batch_size, shuffle=shuffle, seed=seed,
#         #     save_to_dir=save_to_dir,
#         #     save_prefix=save_prefix,
#         #     save_format=save_format,
#         #     follow_links=follow_links)
#
#     def random_transform(self, x, seed=None):
#         """Randomly augment a single image tensor.
#         # Arguments
#             x: 3D tensor, single image.
#             seed: random seed.
#         # Returns
#             A randomly transformed version of the input (same shape).
#         """
#         # x is a single image, so it doesn't have image number at index 0
#         img_row_axis = self.row_axis - 1
#         img_col_axis = self.col_axis - 1
#         img_channel_axis = self.channel_axis - 1
#
#         n_inputs = len(x)
#
#         if seed is not None:
#             np.random.seed(seed)
#
#         # use composition of homographies
#         # to generate final transform that needs to be applied
#         # if self.rotation_range:
#         #     theta = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
#         # else:
#         #     theta = 0
#         #
#         # if self.height_shift_range:
#         #     r = np.random.uniform(-self.height_shift_range, self.height_shift_range) #* x.shape[img_row_axis]
#         #     tx = [r * img.shape[img_row_axis] for img in x]
#         # else:
#         #     tx = 0
#         #
#         # if self.width_shift_range:
#         #     r = np.random.uniform(-self.width_shift_range, self.width_shift_range) #* x.shape[img_col_axis]
#         #     ty =  [r * img.shape[img_col_axis] for img in x]
#         # else:
#         #     ty = 0
#         #
#         # if self.shear_range:
#         #     shear = np.random.uniform(-self.shear_range, self.shear_range)
#         # else:
#         #     shear = 0
#         #
#         # if self.zoom_range[0] == 1 and self.zoom_range[1] == 1:
#         #     zx, zy = 1, 1
#         # else:
#         #     zx, zy = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#         #
#         # transform_matrix = [None] * n_inputs
#         # if theta != 0:
#         #     rotation_matrix = np.array([[np.cos(theta), -np.sin(theta), 0],
#         #                                 [np.sin(theta), np.cos(theta), 0],
#         #                                 [0, 0, 1]])
#         #     transform_matrix = [rotation_matrix] * n_inputs
#         #
#         # if tx != 0 or ty != 0:
#         #     shift_matrix = [np.array([[1, 0, tx[i]],[0, 1, ty[i]], [0, 0, 1]]) for i in range(n_inputs)]
#         #
#         #     transform_matrix = [ shift_matrix if transform_matrix[j] is None else np.dot(transform_matrix[j], shift_matrix[j]) for j in range(n_inputs) ]
#         #
#         # if shear != 0:
#         #     shear_matrix = np.array([[1, -np.sin(shear), 0],
#         #                              [0, np.cos(shear), 0],
#         #                              [0, 0, 1]])
#         #     transform_matrix = [shear_matrix if tm is None else np.dot(tm, shear_matrix) for tm in transform_matrix]
#         #
#         # if zx != 1 or zy != 1:
#         #     zoom_matrix = np.array([[zx, 0, 0],
#         #                             [0, zy, 0],
#         #                             [0, 0, 1]])
#         #     transform_matrix = [zoom_matrix if tm is None else np.dot(tm, zoom_matrix) for tm in transform_matrix]
#
#
#         transf_params={}
#         for m in range(len(x)):
#             transf_params['theta'] = 0
#
#             if self.rotation_range:
#                 transf_params['theta'] = np.pi / 180 * np.random.uniform(-self.rotation_range, self.rotation_range)
#
#             transf_params['ty'] = 0
#             if self.height_shift_range:
#                 r = np.random.uniform(-self.height_shift_range, self.height_shift_range)
#                 transf_params['ty']  = r * x[m].shape[img_row_axis]
#
#             transf_params['tx'] = 0
#             if self.width_shift_range:
#                 r = np.random.uniform(-self.width_shift_range, self.width_shift_range)
#                 transf_params['tx'] =  r * x[m].shape[img_col_axis]
#
#             transf_params['shear'] = 0
#             if self.shear_range:
#                 transf_params['shear'] = np.random.uniform(-self.shear_range, self.shear_range)
#
#             transf_params['zx'], transf_params['zy'] = np.random.uniform(self.zoom_range[0], self.zoom_range[1], 2)
#
#             flip_prob = np.random.random(size=(2,)) # prob of flipping
#
#             transf_params['flip_horizontal'] = flip_prob[0] < 0.5
#             transf_params['flip_vertical']   = flip_prob[1] < 0.5
#
#             x[m] = self.apply_transform(x[m],transf_params)
#
#
#         return x
#
#     def fit(self, x,
#             augment=False,
#             rounds=1,
#             seed=None):
#         """Fits internal statistics to some sample data.
#         Required for featurewise_center, featurewise_std_normalization
#         and zca_whitening.
#         # Arguments
#             x: Numpy array, the data to fit on. Should have rank 4.
#                 In case of grayscale data,
#                 the channels axis should have value 1, and in case
#                 of RGB data, it should have value 3.
#             augment: Whether to fit on randomly augmented samples
#             rounds: If `augment`,
#                 how many augmentation passes to do over the data
#             seed: random seed.
#         # Raises
#             ValueError: in case of invalid input `x`.
#         """
#
#         x = [np.asarray(xx, dtype=K.floatx()) for xx in x]
#         if any([xx.ndim != 4 for xx in x]):
#             raise ValueError('Input to `.fit()` should have rank 4. '
#                              'Got array with shape: ' + str(x.shape))
#
#         for xx in x:
#             if xx.shape[self.channel_axis] not in {1, 3, 4}:
#                 warnings.warn(
#                     'Expected input to be images (as Numpy array) '
#                     'following the data format convention "' + self.data_format + '" '
#                                                                                   '(channels on axis ' + str(
#                         self.channel_axis) + '), i.e. expected '
#                                              'either 1, 3 or 4 channels on axis ' + str(self.channel_axis) + '. '
#                                                                                                              'However, it was passed an array with shape ' + str(
#                         xx.shape) +
#                     ' (' + str(xx.shape[self.channel_axis]) + ' channels).')
#
#         if seed is not None:
#             np.random.seed(seed)
#
#         x = [np.copy(xx) for xx in x]
#
#         if augment:
#             ax = [np.zeros(tuple([rounds * xx.shape[0]] + list(xx.shape)[1:]), dtype=K.floatx()) for xx in x]
#
#             for m in range(len(ax)):
#                 for r in range(rounds):
#                     for i in range(x[m].shape[0]):
#                         ax[m][i + r * x[m].shape[0]] = self.random_transform(x[m][i])
#             x = ax
#
#         if self.featurewise_center:
#             self.mean = [np.mean(xx, axis=(0, self.row_axis, self.col_axis)) for xx in x]
#             broadcast_shape = [1, 1, 1]
#             broadcast_shape[self.channel_axis - 1] = x[0].shape[self.channel_axis]
#             self.mean = [np.reshape(mu, broadcast_shape) for mu in self.mean]
#             x = [x[m] - self.mean[m] for m in range(len(x))]
#
#         if self.featurewise_std_normalization:
#             self.std = [np.std(xx, axis=(0, self.row_axis, self.col_axis)) for xx in x]
#             broadcast_shape = [1, 1, 1]
#             broadcast_shape[self.channel_axis - 1] = x[0].shape[self.channel_axis]
#             self.std = [np.reshape(sigma, broadcast_shape) for sigma in self.std]
#             x = [x[m]/(self.std[m] + K.epsilon()) for m in range(len(m))]
#
#         if self.zca_whitening:
#             self.principal_components = [None] * len(x)
#             for k in range(len(x)):
#                 flat_x = np.reshape(x[k], (x[k].shape[0], x[k].shape[1] * x[k].shape[2] * x[k].shape[3]))
#                 sigma = np.dot(flat_x.T, flat_x) / flat_x.shape[0]
#                 u, s, _ =  linalg.svd(sigma)
#
#
#                 self.principal_components[k] = np.dot(np.dot(u, np.diag(1. / np.sqrt(s + self.zca_epsilon))), u.T)
#
# class MultiInputNumpyArrayIterator(NumpyArrayIterator):
#     """Iterator yielding data from a Numpy array.
#     # Arguments
#         x: Numpy array of input data.
#         y: Numpy array of targets data.
#         image_data_generator: Instance of `ImageDataGenerator`
#             to use for random transformations and normalization.
#         batch_size: Integer, size of a batch.
#         shuffle: Boolean, whether to shuffle the data between epochs.
#         seed: Random seed for data shuffling.
#         data_format: String, one of `channels_first`, `channels_last`.
#         save_to_dir: Optional directory where to save the pictures
#             being yielded, in a viewable format. This is useful
#             for visualizing the random transformations being
#             applied, for debugging purposes.
#         save_prefix: String prefix to use for saving sample
#             images (if `save_to_dir` is set).
#         save_format: Format to use for saving sample images
#             (if `save_to_dir` is set).
#     """
#
#     _dropping_counter=0
#
#     def __init__(self, x, y, image_data_generator,
#                  batch_size=32, shuffle=False, seed=None,
#                  data_format=None,
#                  save_to_dir=None, save_prefix='', save_format='png',
#                  drop_probability=0):
#
#         self.drop_probability= drop_probability
#         for xx in x:
#             if y is not None and len(xx) != len(y):
#                 raise ValueError('X (images tensor) and y (labels) '
#                                  'should have the same length. '
#                                  'Found: X.shape = %s, y.shape = %s' %
#                                  (np.asarray(xx).shape, np.asarray(y).shape))
#
#         if data_format is None:
#             data_format = K.image_data_format()
#         self.x = [np.asarray(xx, dtype=K.floatx()) for xx in x]
#
#         for xx in self.x:
#             if xx.ndim != 4:
#                 raise ValueError('Input data in `NumpyArrayIterator` '
#                                  'should have rank 4. You passed an array '
#                                  'with shape', xx.shape)
#
#         channels_axis = 3 if data_format == 'channels_last' else 1
#
#         for xx in self.x:
#             if xx.shape[channels_axis] not in {1, 3, 4}:
#                 warnings.warn('NumpyArrayIterator is set to use the '
#                               'data format convention "' + data_format + '" '
#                                                                          '(channels on axis ' + str(
#                     channels_axis) + '), i.e. expected '
#                                      'either 1, 3 or 4 channels on axis ' + str(channels_axis) + '. '
#                                                                                                  'However, it was passed an array with shape ' + str(
#                     xx.shape) +
#                               ' (' + str(xx.shape[channels_axis]) + ' channels).')
#         if y is not None:
#             self.y = np.asarray(y)
#         else:
#             self.y = None
#         self.image_data_generator = image_data_generator
#         self.data_format = data_format
#         self.save_to_dir = save_to_dir
#         self.save_prefix = save_prefix
#         self.save_format = save_format
#         super(NumpyArrayIterator, self).__init__(x,y,image_data_generator, batch_size=batch_size, shuffle=shuffle, seed=seed)
#
#
#
#
#     def next(self):
#         # for python 2.x.
#         # Keeps under lock only the mechanism which advances
#         # the indexing of each batch
#         # see http://anandology.com/blog/using-iterators-and-generators/
#         with self.lock:
#             index_array, current_index, current_batch_size = next(self.index_generator)
#
#         return self._get_batches_of_transformed_samples(index_array)
#
#     def _get_batches_of_transformed_samples(self, index_array):
#         # The transformation of images is not under thread lock so it can be done in parallel
#         batch_x = [np.zeros(tuple([len(index_array)] + list(x.shape)[0:]),
#                            dtype=K.floatx()) for x in self.x]
#         for i, j in enumerate(index_array):
#             x = [m.astype(K.floatx()) for m in self.x]
#             x = self.image_data_generator.random_transform(x)
#             x = self.image_data_generator.standardize(x)
#
#             for k in range(len(self.x)):
#                 batch_x[k][i] = x[k]
#
#         if self.save_to_dir:
#             for i, j in enumerate(index_array):
#                 img = array_to_img(batch_x[i], self.data_format, scale=True)
#                 fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
#                                                                   index=j,
#                                                                   hash=np.random.randint(1e4),
#                                                                   format=self.save_format)
#                 img.save(os.path.join(self.save_to_dir, fname))
#         if self.y is None:
#             return batch_x
#         batch_y = self.y[index_array]
#
#         max_comb = 2**len(self.x)
#
#         for i in range(self.batch_size):
#             k =  self._dropping_counter if np.random.binomial(1,self.drop_probability)==1 else 0
#             for m in range(len(self.x)):
#                 mask = 2**m
#
#                 if ((k & mask) != 0):
#                     batch_x[m][i] = np.zeros_like(batch_x[m][i])
#
#             self._dropping_counter= (self._dropping_counter+1) % max_comb
#
#
#
#
#         #if (self.drop_probability>0):
#         #    for m in range(len(self.x)):
#         #        for i in range(self.batch_size):
#         #            P = np.random.binomial(1,self.drop_probability)
#         #            if (P>0):
#         #                batch_x[m][i] = np.zeros_like(batch_x[m][i])
#
#
#
#         return batch_x, batch_y


class MultiInputImageDataGenerator:

    def __init__(this,n_inputs,*args,**kwargs):
        this.generators = []

        for i in range(n_inputs):
            this.generators.append(ImageDataGenerator(*args,**kwargs))

    def fit(this,x):
        assert len(x) == len(this.generators)

        for i in range(len(x)):
            this.generators[i].fit(x[i])

    def flow(this,x_train,y_train,seed,**kwargs):

        flow1 = this.generators[0].flow(x_train[0],y_train,seed=seed,**kwargs)
        flows = [this.generators[i].flow(x_train[0],x_train[i]) for i in range(1,len(x_train))]

        while True:
            output1 = flow1.next()
            outputs = [f.next() for f in flows]

            yield [output1[0]]+[o[1] for o in outputs], output1[1]











if __name__ == "__main__":
    D = CVPPPDataset("/media/vgiuffrida/PHDDATA/Data/Plant Phenotyping/CVPPP_2017/CVPPP2017_LCC_training/training/",folders=[1, 2, 4])
    D.load(new_size=(320,320))
    data = D.getData(['rgb', 'count'])
    np.savez('cvppp17_data.npz', data=[data])

    D = CVPPPDataset("/media/vgiuffrida/PHDDATA/Data/Plant Phenotyping/CVPPP_2017/CVPPP2017_testing/testing/",folders=[1,2,4])
    D.load(new_size=(320, 320))
    data = D.getData(['rgb','count'])
    np.savez('cvppp17_data_testing.npz',data=[data])


