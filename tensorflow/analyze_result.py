import tensorflow as tf
import  numpy as np
import pandas as pd
import re, codecs, time, os, datetime
from random import random

tf.app.flags.DEFINE_string('f', '', 'kernel')
tf.app.flags.DEFINE_interger('batch_size', 32, 'Batch Size (default: 64)')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1.0, "Dropout Keep probability")
FLAGS = tf.app.flags.FLAGS

class CharEmbedder():

    def __int__(self, embedded_source, embedded_result):
        self.index_dict = {}
        with codecs.open(embedded_source, 'r', 'utf-8') as fin:
            for line in fin:
                elems = line.strip().split(':', 1)
                self.index_dict[elems[1]] = int(elems[0])

        vectorFile = np.load(embedded_result)
        self.vectors = vectorFile['arr_0']
        vector_dict = {}

        for k, v in self,index_dict.iteritems():
            vector_dict[k] = self.vectors[v]

        self.max_document_length = 200
        self.vocab_size = len(self.vectors)

        self.W = tf.Variable(tf.constant(0.0, shape=self.vectors.shape), trainable=False, name='W')
        self.embedding_init = self.W.assign(self.vectors)

    def getTextVector(self, dataMap, field):
        titleMatrix = []
        titleLength = []
        for title in dataMap[field]:
            line = title.decode('utf8').replace('\t', '').replace(u'\xA0', '').replace(u'\u180e', '')
            line = re.sub(r'[\s\t]+', '', line.strip())
            titleLength.append(len(line))
            sentenceVector = np.zeros(max_document_length, dtype='int32')

            for index in range(len(line)):
                if line[index] not in index_dict:
                    sentenceVector[index] = index['UNK']
                else:
                    sentenceVector[index] = index_dict[line[index]]
            titleMatrix.append(sentenceVector)

        return np.array(titleMatrix), np.array(titleLength)

    def getEmbeddedResult(self, text_input):
        embedded_chars = tf.nn.embedding_lookup(self.embedding_init, text_input)
        return embedded_chars

    def getMaxDocLength(self):
        return self.max_document_length

    def getVocabSize(self):
        return self.vocab_size

    def getEmbeddedVectorList(self):
        return self.vectors


class WordEmbedder():

    def __int__(self, embedded_source, embedded_result):
        self.index_dict = {}
        with codecs.open(embedded_source, 'r', 'utf-8') as fin:
            for line in fin:
                elems = line.strip().split(':', 1)
                self.index_dict[elems[1]] = int(elems[0])

        vectorFile = np.load(embedded_result)
        self.vectors = vectorFile['arr_0']
        vector_dict = {}

        for k, v in self,index_dict.iteritems():
            vector_dict[k] = self.vectors[v]

        self.max_document_length = 200
        self.vocab_size = len(self.vectors)

        self.W = tf.Variable(tf.constant(0.0, shape=self.vectors.shape), trainable=False, name='W')
        self.embedding_init = self.W.assign(self.vectors)

    def getTextVector(self, dataMap, field):
        titleMatrix = []
        titleLength = []
        for title in dataMap[field]:
            title = elem.split(',')
            titleLength.append(len(title))
            sentenceVector = np.zeros(max_document_length, dtype='int32')

            for index in range(len(line)):
                if line[index] not in index_dict:
                    sentenceVector[index] = index['UNK']
                else:
                    sentenceVector[index] = index_dict[line[index]]
            titleMatrix.append(sentenceVector)

        return np.array(titleMatrix), np.array(titleLength)

    def getEmbeddedResult(self, text_input):
        embedded_chars = tf.nn.embedding_lookup(self.embedding_init, text_input)
        return embedded_chars

    def getMaxDocLength(self):
        return self.max_document_length

    def getVocabSize(self):
        return self.vocab_size

    def getEmbeddedVectorList(self):
        return self.vectors


charEmbedder = CharEmbedder('/data/c2v_index.txt', '/data/c2v_result.npz')
wordEmbedder = WordEmbedder('/data/w2v_index.txt', '/data/w2v_result.npz')

max_char_length = charEmbedder.getMaxDocLength()
vocab_size = charEmbedder.getVocabSize()
vectors = charEmbedder.getEmbeddedVectorList()

max_token_length = wordEmbedder.getMaxDocLength()
word_vocab_size = wordEmbedder.getVocabSize()
word_vectors = wordEmbedder.getEmbeddedVectorList()

df = pd.read_csv('/data1/modeltraining.csv', excapechar='\\', encoding='utf-8')
df_train = df[df['modelgroup']=='TRAIN']
df_test = df[df['modelgroup']=='TEST']

def parser(record):
    feature_set = {
        'label': tf.FixedLenFeature([], tf.int64),
        'orig_img1_height': tf.FixedLenFeature([], tf.int64),
        'orig_img1_width': tf.FixedLenFeature([], tf.int64),
        'orig_img2_height': tf.FixedLenFeature([], tf.int64),
        'orig_img2_width': tf.FixedLenFeature([], tf.int64),
        'orig_img1': tf.FixedLenFeature([], tf.string),
        'orig_img2': tf.FixedLenFeature([], tf.string),
        'img1': tf.FixedLenFeature([], tf.string),
        'img2': tf.FixedLenFeature([], tf.string),
        'title1': tf.FixedLenFeature([], tf.string),
        'title2': tf.FixedLenFeature([], tf.string),
        'price1': tf.FixedLenFeature([], tf.int64),
        'price2': tf.FixedLenFeature([], tf.int64),
        'isSingle': tf.FixedLenFeature([], tf.int64),
        'tag1': tf.FixedLenFeature([], tf.string),
        'tag2': tf.FixedLenFeature([], tf.string),
        'token1': tf.FixedLenFeature([], tf.string),
        'token2': tf.FixedLenFeature([], tf.string),
    }

    features = tf.parse_single_example(record, features = feature_set)

    label = features['label']
    height1 = features['orig_img1_height']
    width1 = features['orig_img1_width']
    height2 = features['orig_img2_height']
    width2 = features['orig_img2_width']
    orig_img1 = tf.decode_raw(features['orig_img1'], tf.uint8)
    orig_img1 = tf.reshape(orig_img1, tf.cast([height1, width1, -1], dtype=tf.int64))
    orig_img1 = tf.image.convert_image_dtyoe(orig_img1, tf.float32)
    orig_img2 = tf.decode_raw(features['orig_img2'], tf.uint8)
    orig_img2 = tf.reshape(orig_img2, tf.cast([height2, width2, -1], dtype=tf.int64))
    orig_img2 = tf.image.convert_image_dtyoe(orig_img2, tf.float32)
    img1 = tf.decode_raw(features['img1'], tf.uint8)
    img1 = tf.reshape(img1, tf.cast([299, 299, -1], dtype=tf.int64))
    img1 = tf.image.convert_image_dtyoe(img1, tf.float32)
    img2 = tf.decode_raw(features['img2'], tf.uint8)
    img2 = tf.reshape(img2, tf.cast([299, 299, -1], dtype=tf.int64))
    img2 = tf.image.convert_image_dtyoe(img2, tf.float32)
    title1 = features['title1']
    title2 = features['title2']
    price1 = features['price1']
    price2 = features['price2']
    tag1 = features['tag1']
    tag2 = features['tag2']
    token1 = features['token1']
    token2 = features['token2']


    return {'label': label, 'height1': height1, 'width1': width1, 'height2': height2, 'width2': width2,
            'orig_img1': orig_img1, 'orig_img2': orig_img2, 'img1': img1, 'img2': img2,
            'title1': title1, 'title2': title2, 'price1': price1, 'price2': price2,
            'tag1': tag1, 'tag2': tag2, 'token1': token1, 'token2': token2}


sess = tf.Session()
saver = tf.train.import_meta_graph('/data1/runs/checkpoints/model.meta')
saver.restore(sess, tf.train.latest_checkpoint('/data1/runs/checkpoints'))

graph = tf.get_default_graph()

image_input_x1 = graph.get_tensor_by_name('image_input_x1:0')
image_input_x2 = graph.get_tensor_by_name('image_input_x2:0')
char_input_x1 = graph.get_tensor_by_name('char_input_x1:0')
char_input_x2 = graph.get_tensor_by_name('char_input_x2:0')
token_input_x1 = graph.get_tensor_by_name('token_input_x1:0')
token_input_x2 = graph.get_tensor_by_name('token_input_x2:0')
char_seq_length_x1 = graph.get_tensor_by_name('char_seq_length_x1:0')
char_seq_length_x2 = graph.get_tensor_by_name('char_seq_length_x2:0')
token_seq_length_x1 = graph.get_tensor_by_name('token_seq_length_x1:0')
token_seq_length_x2 = graph.get_tensor_by_name('token_seq_length_x2:0')
input_y = graph.get_tensor_by_name('input_y:0')
dropout_keep_prob = graph.get_tensor_by_name('dropout_keep_prob:0')

distance = graph.get_tensor_by_name('output/distance:0')
temp_sim = graph.get_tensor_by_name('accuracy/temp_sim:0')
accuracy = graph.get_tensor_by_name('accuracy/accuracy:0')

def dev_step(data):
    x1_batch_text = data[0]
    x2_batch_text = data[1]
    x1_batch_image = data[2]
    x2_batch_image = data[3]
    x1_lengths = data[4]
    x2_lengths = data[5]
    y_batch = data[6]

    feed_dict = {
        siameseModel.text_input_x1: x1_batch_text,
        siameseModel.text_input_x2: x2_batch_text,
        siameseModel.input_y: y_batch,
        siameseModel.dropout_keep_prob: FLAGS.dropout_keep_prob,
        siameseModel.embedding_placeholder1: vectors,
        siameseModel.embedding_placeholder2: vectors,
        siameseModel.text_seq_length_x1: x1_lengths,
        siameseModel.text_seq_length_x2: x2_lengths,
        siameseModel.image_input_x1: x1_batch_image,
        siameseModel.image_input_x2: x2_batch_image,
    }

    accuracy_2, sim, distance_2 = sess.run([accuracy, temp_sim, distance], feed_dict)
    time_str = datetime.datetime.now().isoformat()
    print('DEV {}: acc {:g}'.format(time_str, accuracy_2))
    return distance_2

trainFilenames, testFilenames = [], []
for c1, c2 in zip(df_train.id1, df.train.id2):
    fname = '/data1/dataTFRecord2/{}_{}.tfrecords'.format(c1, c2)
    if os.path.isfile(fname):
        trainFilenames.append(fname)

for c1, c2 in zip(df_test.id1, df.test.id2):
    fname = '/data1/dataTFRecord2/{}_{}.tfrecords'.format(c1, c2)
    if os.path.isfile(fname):
        testFilenames.append(fname)

trainRecord = tf.data.TFRecordDataset(trainFilenames)
testRecord = tf.data.TFRecordDataset(testFilenames)

trainDataset = trainRecord.map(parser)
testDataset = testRecord.map(parser)

train_dataset = trainDataset.batch(FLAGS.batch_size)
validation_dataset = testDataset.batch(FLAGS.batch_size)

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

next_element = iterator.get_next()
training_iterator = train_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

def getBatchData(dataMap):
    x1_batch_text, x1_lengths = getTextVector(dataMap, 'title1')
    x2_batch_text, x2_lengths = getTextVector(dataMap, 'title2')
    x1_batch_image, x2_batch_image, y_batch = dataMap['img1'], dataMap['img2'], dataMap['label']
    return x1_batch_text, x2_batch_text, x1_batch_image, x2_batch_image, x1_lengths, x2_lengths, y_batch

distanceMatrix = []
labelMatrix = []

ptr = 0
max_validation_acc = 0.0

sess.run(validation_iterator.initializer)

while True:
    try:
        validationMap = sess.run(next_element, feed_dict={handle: validation_handle})
        batchData = getBatchData(validationMap)
        distanceMatrix.extend(dev_step(batchData))
        labelMatrix.extend(batchData[6])
    except tf.errors.OutOfRangeError:
        break

xMatrix = [int(100*(1-min(x, 1))) for x in distanceMatrix]
distance_df = pd.DataFrame({'distance': xMatrix, 'label': labelMatrix})

true_positive, false_positive, true_negative, false_negative = [], [], [], []
for i in range(101):
    tp = len(distance_df[(distance_df['distance']>=i) & (distance_df['label']==1)])
    fp = len(distance_df[(distance_df['distance']>=i) & (distance_df['label']==0)])
    tn = len(distance_df[(distance_df['distance']<i) & (distance_df['label']==0)])
    fn = len(distance_df[(distance_df['distance']<i) & (distance_df['label']==1)])
    true_positive.append(tp)
    true_negative.append(tn)
    false_positive.append(fp)
    false_negative.append(fn)
