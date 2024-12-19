import numpy as np
import matplotlib.pyplot as plt
import os
import pandas as pd
import nrrd
from PIL import Image
import re


def get_volumes(datadir, id, csv_data) :    
    df= csv_data[csv_data['patient'] == id]['t']
    volume = []
    volume_l = []
    volume_v = []
    ts = []
    for slicename in df :        
        filename = os.path.join(datadir, 'images', slicename)
        maskname_l = os.path.join(datadir, 'masks', slicename)
        maskname_v = os.path.join(datadir, 'masks_vss', slicename)
        image = np.load(filename)
        mask_l = np.load(maskname_l)
        mask_v = np.load(maskname_v)
        volume = volume + [image]
        volume_l = volume_l + [mask_l]
        volume_v = volume_v + [mask_v]
        match = re.search('_.-(.+?).npy',slicename)
        ts.append(match.group(1))

    #load volume
    volume = np.array(volume)
    volume = np.transpose(volume, [1,2,0])
    volume = np.astype(volume, np.float32)

    #load liver mask volume
    volume_l = np.array(volume_l)
    volume_l = np.transpose(volume_l, [1,2,0])
    volume_l[volume_l > 0] = 1
    x_l, y_l, z_l = np.where(volume_l > 0)
    left = x_l.min()
    right = x_l.max()
    top = y_l.min()
    bottom = y_l.max()

    #load vessel mask volume
    volume_v = np.array(volume_v)
    volume_v = np.transpose(volume_v, [1,2,0])
    volume_v[volume_v > 0] = 1

    volume = volume[left:right, top:bottom, :]
    volume_l = volume_l[left:right, top:bottom, :]
    volume_v = volume_v[left:right, top:bottom, :]
    
    return  np.astype(volume, np.int16), np.astype(volume_l, np.int8),  np.astype(volume_v, np.int8),  ts

def save_volume_unet(voldata, vollabel, unnet_dir, dataset, id, ts, train_or_test = 'train') :
    """
    ts: list of t for each slice
    id: patient id
    """
    datadir = os.path.join(unnet_dir, dataset + '-2D')    

    for i in range(voldata.shape[2]) : 
        im = Image.fromarray(voldata[:,:,i])
        lblim =   Image.fromarray(vollabel[:,:,i])
        filename = str(id) + '-' + ts[i] + '_0000.png'         
        if train_or_test == 'train' :
            im.save(os.path.join(datadir, 'imagesTr', filename), bits = 16)
            lblim.save(os.path.join(datadir, 'labelsTr' , filename), bits = 8)
        else :
            im.save(os.path.join(datadir, 'imagesTs', filename), bits = 16)


# data = np.extract(volume>0, volume)
# print(volume.min(), volume.max())
# hist = np.histogram(data, bins = 10)
# plt.bar(range(len(hist[0])), hist[0])
# plt.xticks(range(len(hist[0])), hist[1][:-1])
# plt.show()
# nrrd.write('/home/jmsaavedrar/Research/git/medical_imaging/seg_l.nrrd', volume, index_order= 'C')


# # volume =  ( volume - min_val )  / ( max_val - min_val)
# # volume = volume * volume_s
# print(volume.dtype)
# print(volume.min(), volume.max())

def create_dataset() :
    unet_home ='/hd_data/nnUnet'
    unet_rawdir = os.path.join(unet_home, 'nnUnet_raw')
    unet_rawdir = os.path.join(unet_home, 'nnUnet_raw') 

    datadir = '/hd_data/colorectal/colorectal'
    folds = np.array([[1,2,3,4,5]])
    folds = np.repeat(folds, 5,axis = 0)
    for i in range(5):
        aux = folds[i][i]        
        folds[i][i] = folds[i][0]
        folds[i][0] = aux 
    print(folds)
    fold_id = 2
    csvfile_test = [os.path.join(datadir, 'folds', 'fold_{}.csv'.format(i)) for i in folds[fold_id][0:1]]
    csvfile_train = [os.path.join(datadir, 'folds', 'fold_{}.csv'.format(i)) for i in folds[fold_id][1:]]
    
    df_list = [pd.read_csv(f) for f in csvfile_train]
    df_train = pd.concat(df_list)

    df_list = [pd.read_csv(f) for f in csvfile_test]
    df_test = pd.concat(df_list)

    ids_train = df_train['patient'].unique()
    ids_test = df_test['patient'].unique()
    
    n_train = 0
    n_test = 0
    for patient_id in  ids_train: 
        volume, volume_l, volume_v, ts = get_volumes(datadir, patient_id, df_train)        
        print(" saving data for patient {}".format(patient_id))
        n_train = n_train + volume.shape[2]
        save_volume_unet(volume, volume_v, unet_rawdir, 'Dataset001_HepaticVessels', patient_id, ts, 'train')
    
    print('end training data')
    for patient_id in  ids_test: 
        volume, volume_l, volume_v, ts = get_volumes(datadir, patient_id, df_test)        
        n_test = n_test + volume.shape[2]
        print(" saving data for patient {}".format(patient_id))
        save_volume_unet(volume, volume_v, unet_rawdir, 'Dataset001_HepaticVessels', patient_id, ts, 'test')
    print('end testing/validation data')
    print('slices-> n_train ({}) n_test ({})'.format(n_train, n_test)) 

def test_images() :
    datadir ='/hd_data/nnUnet/nnUnet_raw/Dataset001_HepaticVessels-2D'
    id = '1003-017_000'
    imname = os.path.join(datadir, 'imagesTr', id + '.png')
    lblname = os.path.join(datadir, 'labelsTr', id + '.png')

    im = np.asarray(Image.open(imname), dtype = np.int16)
    lbl = np.asarray(Image.open(lblname), dtype = np.int8)
    print('im: {} lbl: {}'.format(im.dtype, lbl.dtype))
    fig, ws = plt.subplots(1,2)
    ws[0].imshow(im, cmap = 'jet')
    ws[1].imshow(lbl, cmap = 'jet')
    plt.show()


if __name__ == '__main__' :
    op = 'test'
    if op == 'create' :
        create_dataset() 
    if op == 'test' :
        test_images()
    #save image for each patient  imagesTr and imagesTs labelsTr

    # df = pd.read_csv(csvfile)
    # ids = df['patient'].unique()    
    # list_data = []
    # for id in ids[:5] :
    #     print(id)
    #     volume, volume_l, volume_v = get_volumes(id, df)
    #     data = np.extract(np.where(volume_v == 1), volume) - 1024
    #     min_val = 800
    #     max_val = 1200
    #     volume[volume < min_val] = min_val
    #     volume[volume > max_val] = max_val
    #     volume =  ( volume - min_val )  / ( max_val - min_val)
    #     volume = volume * volume_l
    #     print(volume_v.shape)
    #     #nrrd.write('/home/jmsaavedrar/Research/git/medical_imaging/seg_l_{}.nrrd'.format(id), volume_v, header = header_f,  index_order= 'C')
    #     # fig, ws = plt.subplots(1,2)
    #     # for i in np.arange(volume.shape[2]) :
    #     #     ws[0].cla()
    #     #     ws[1].cla()
    #     #     ws[0].imshow(volume[:,:,i], cmap = 'Spectral', vmin = 0, vmax = 1)    
    #     #     ws[1].imshow(volume_v[:,:,i], cmap = 'Spectral', vmin = 0, vmax = 1)    
    #     #     plt.waitforbuttonpress(0.1)
    #     #     plt.show()

    #     list_data = list_data + data.tolist()
    # all_data = np.array(list_data)    
    # std = all_data.std()
    # mean = all_data.mean()
    # # print('{} +-{}'.format(mean, std))
    # # hist = np.histogram(all_data, bins = 20)
    # # plt.bar(range(len(hist[0])), hist[0])
    # # plt.xticks(range(len(hist[0])), hist[1][:-1])
    # # plt.show()
    