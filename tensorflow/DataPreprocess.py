import tensorflow as tf
import pandas as pd
import numpy as np
from PIL import Image
import glob, re

DATASOURCE = 'modeltraining.csv'
IMAGESOURCE1 = '/images/source1/'
IMAGESOURCE2 = '/images/source2/'
TFRECORDPATH = 'data/tfrecord/'

df = pd.read_csv(DATASOURCE, escapechar='\\', encoding='utf-8')
df_train = df[df['modelgroup']=='TRAIN']
df_test = df[df['modelgroup']=='TEST']

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

## _bytes is used for string/char values

def _bytes_feature(value):
    return tf.train.Feature(byteslist=tf.train.BytesList(value=[value]))

image_path1 = IMAGESOURCE1
image_path2 = IMAGESOURCE2
image1_list = glob.glob(image_path1+'*.jpg')
image2_list = glob.glob(image_path2+'*.jpg')

tfRecordPath = TFRECORDPATH

def writeTFRecords(dataFrame):
    ## tfrecord_filename = tfRecordPath + recordName
    successCnt, errorCnt = 0, 0

    for c1, c2, label, title1, title2, price1, price2, isSingle in zip(dataFrame.id1, dataFrame.id2, dataFrame.label,
                                                                       dataFrame.title1, dataFrame.title2,
                                                                       dataFrame.price1, dataFrame.price2, dataFrame.isSingle):
        c1Path, c2Path = image_path1+str(c1)+'.jpg', image_path2+str(c2)+'.jpg'

        try:
            img1 = Image.open(c1Path).convert('RGB')
            img2 = Image.open(c2Path).convert('RGB')
            rimg1 = img1.resize([299, 299], Image.ANTIALIAS)
            rimg2 = img2.resize([299, 299], Image.ANTIALIAS)

            img1 = np.asarray(img1)
            img2 = np.asarray(img2)
            rimg1 = np.asarray(rimg1)
            rimg2 = np.asarray(rimg2)

            img1Shape = img1.shape
            img2Shape = img2.shape
            tfrecord_filename = tfRecordPath+str(c1)+'_'+str(c2)+'.tfrecords'
            writer = tf.python_io.TFRecordWriter(tfrecord_filename)

            line = title1.encode('utf8')
            cLine = title2.encode('utf8')

            feature = {
                'label': _int64_feature(label),
                'orig_img1_height': _int64_feature(img1Shape[0]),
                'orig_img1_width': _int64_feature(img1Shape[1]),
                'orig_img2_height': _int64_feature(img2Shape[0]),
                'orig_img2_width': _int64_feature(img2Shape[1]),
                'orig_img1': _bytes_feature(img1.toString()),
                'orig_img2': _bytes_feature(img2.toString()),
                'img1': _bytes_feature(rimg1.toString()),
                'img2': _bytes_feature(rimg2.toString()),
                'title1': _bytes_feature(line),
                'title2': _bytes_feature(cLine),
                'price1': _int64_feature(long(price1)),
                'price2': _int64_feature(long(price2)),
                'isSingle': _int64_feature(isSingle)
            }
            ## Create an example protocol buffer
            example = tf.train.Example(features=tf.train.Features(feature=feature))

            #Serialize to string and write on the file
            writer.write(example.SerializeToString())
            writer.close()

            successCnt += 1
            print('count: {}'.format(successCnt))

        except IOError:
            print('No such file')
            errorCnt += 1
            print('error count: {}'.format(errorCnt))

