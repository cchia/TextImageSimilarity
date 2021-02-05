import tensorflow as tf
import re, codecs, time, os, datetime
import numpy as np, pandas as pd
from Inception_Resnet import inception_resnet_v2, inception_resnet_v2_arg_scope
slim = tf.contrib.slim

tf.app.flags.DEFINE_string('f', '', 'kernel')

#parameters
tf.app.flags.DEFINE_integer('embedding_dim', 128, 'Character_embedding_dimension')
tf.app.flags.DEFINE_float('dropout_keep_prob', 1.0, 'Dropout keep probability')
tf.app.flags.DEFINE_float('l2_reg_lambda', 0.0, 'L2 regularization lambda')
tf.app.flags.DEFINE_float('optimizer_rate', 0.001, 'rate for optimizer')
tf.app.flags.DEFINE_integer('embedding_dim', 128, 'Number of hidden units')

# Training parameters
tf.app.flags.DEFINE_integer('batch_size', 16, 'Batch Size (default: 64)')
tf.app.flags.DEFINE_integer('num_epochs', 200, 'Number of training epochs (default: 200)')
tf.app.flags.DEFINE_integer('evaluate_every', 2000, 'Evaluate model on dev set after this many steps (default: 100)')
tf.app.flags.DEFINE_integer('checkpoint_every', 2000, 'Save model after this many steps (default: 100')

# Misc parameters
tf.app.flags.DEFINE_boolean('allow_soft_placement', True, 'Allow device soft device placement')
tf.app.flags.DEFINE_boolean('log_device_placement', False, 'Log placement of ops on devices')

FLAGS = tf.app.flags.FLAGS
FLAGS.flag_values_dict()

checkpoint_file = '/data1/pretrained/inception_resnet_v2_2016_08_30.ckpt'

log_dir = 'log/'
if not os.path.exisrs(log_dir):
    os.mkdir(log_dir)

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

trainFilenames, testFilenames = [], []

for c1, c2 in zip(df_train.id1, df_train.id2)
    fname = '/data/dataTFRecord/+{}_{}.tfrecords'.format(c1, c2)
    if os.path.isfile(fname):
        trainFilenames.append(fname)

for c1, c2 in zip(df_test.id1, df_test.id2)
    fname = '/data/dataTFRecord/+{}_{}.tfrecords'.format(c1, c2)
    if os.path.isfile(fname):
        testFilenames.append(fname)

trainRecord  = tf.data.TFRecordDataset(trainFilenames)
testRecord  = tf.data.TFRecordDataset(testFilenames)
trainDataset = trainRecord.map(parser)
testDataset = testRecord.map(parser)

train_dataset = trainDataset.batch(FLAGS.batch_size)
validation_dataset = testDataset.batch(FLAGS.batch_size)

class CharMachine():

    def extract_axis(selfself, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def BiRNN(self, x, dropout, hidden_units, seq_length, reuse=False):
        n_hidden = hidden_units
        n_layers = 3
        with tf.name_scope('char_model'):
            with tf.variable_scope('c_fw', reuse = reuse):
                stacked_rnn_fw = []
                for _ in range(n_layers):
                    fw_cell = tf.nn.rnn.cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                    stacked_rnn_fw.append(lstm_fw_cell)
                lstm_fw_cell_m = tf.nn.rnn.cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            with tf.variable_scope('c_bw', reuse = reuse):
                stacked_rnn_bw = []
                for _ in range(n_layers):
                    bw_cell = tf.nn.rnn.cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                    stacked_rnn_bw.append(lstm_bw_cell)
                lstm_bw_cell_m = tf.nn.rnn.cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
            # Get lstm cell output
            with tf.variable_scope('c_biLSTM', reuse = reuse):
                outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs=x,
                                                                  sequence_length = seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        return self.extract_axis(outputs, seq_length-1)

class WordMachine():

    def extract_axis(selfself, data, ind):
        batch_range = tf.range(tf.shape(data)[0])
        indices = tf.stack([batch_range, ind], axis=1)
        res = tf.gather_nd(data, indices)
        return res

    def BiRNN(self, x, dropout, hidden_units, seq_length, reuse=False):
        n_hidden = hidden_units
        n_layers = 3
        with tf.name_scope('word_model'):
            with tf.variable_scope('w_fw', reuse = reuse):
                stacked_rnn_fw = []
                for _ in range(n_layers):
                    fw_cell = tf.nn.rnn.cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_fw_cell = tf.contrib.rnn.DropoutWrapper(fw_cell, output_keep_prob=dropout)
                    stacked_rnn_fw.append(lstm_fw_cell)
                lstm_fw_cell_m = tf.nn.rnn.cell.MultiRNNCell(cells=stacked_rnn_fw, state_is_tuple=True)
            with tf.variable_scope('w_bw', reuse = reuse):
                stacked_rnn_bw = []
                for _ in range(n_layers):
                    bw_cell = tf.nn.rnn.cell.BasicLSTMCell(n_hidden, forget_bias=1.0, state_is_tuple=True)
                    lstm_bw_cell = tf.contrib.rnn.DropoutWrapper(bw_cell, output_keep_prob=dropout)
                    stacked_rnn_bw.append(lstm_bw_cell)
                lstm_bw_cell_m = tf.nn.rnn.cell.MultiRNNCell(cells=stacked_rnn_bw, state_is_tuple=True)
            # Get lstm cell output
            with tf.variable_scope('w_biLSTM', reuse = reuse):
                outputs, states = tf.nn.bidirectional_dynamic_rnn(lstm_fw_cell_m, lstm_bw_cell_m, inputs=x,
                                                                  sequence_length = seq_length, dtype=tf.float32)
            outputs = tf.concat(outputs, 2)
        return self.extract_axis(outputs, seq_length-1)


class ImageMachine():
    def inceptionResnet(self, input, reuse = False):
        with tf.variable_scope('', reuse=reuse):
            with slim.arg_scope(inception_resnet_v2_arg_scope()):
                logits, end_points = inception_resnet_v2(input, is_training=True)
            net = end_points['Conv2d_7b_1x1']
            net = tf.contrib.layes.flatten(net)
            net = tf.layers.dense(inputs = net, units = 256, reuse=reuse)
            net = tf.contrib.layers.batch_norm(net, center=True, scale=True, is_training=True, reuse=reuse, scope='bn')

        return net

    def cnn(self, input, reuse=False):
        net = input
        with tf.name_scope('model'):
            with tf.variable_scope('conv1') as scope:
                net = tf.contrib.layrs.conv2d(net, 32, [5, 5], activation_fn=tf.nn.relu, padding='SAME',
                                              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope('conv2') as scope:
                net = tf.contrib.layrs.conv2d(net, 64, [4, 4], activation_fn=tf.nn.relu, padding='SAME',
                                              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope('conv3') as scope:
                net = tf.contrib.layrs.conv2d(net, 128, [3, 3], activation_fn=tf.nn.relu, padding='SAME',
                                              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            with tf.variable_scope('conv4') as scope:
                net = tf.contrib.layrs.conv2d(net, 2, [2, 2], activation_fn=tf.nn.relu, padding='SAME',
                                              weights_initializer=tf.contrib.layers.xavier_initializer_conv2d(), scope=scope, reuse=reuse)
                net = tf.contrib.layers.max_pool2d(net, [2, 2], padding='SAME')

            net = tf.reshpae(net, [-1, 19*19*2])
            net = tf.layers.dense(inputs=net, units=256, reuse=reuse)

        return net


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

cm = CharMachine()
wm = WordMachine()
im = ImageMachine()
charEmbedder = CharEmbedder('/data/c2v_index.txt', '/data/c2v_result.npz')
wordEmbedder = WordEmbedder('/data/w2v_index.txt', '/data/w2v_result.npz')

max_char_length = charEmbedder.getMaxDocLength()
vocab_size = charEmbedder.getVocabSize()
vectors = charEmbedder.getEmbeddedVectorList()

max_token_length = wordEmbedder.getMaxDocLength()
word_vocab_size = wordEmbedder.getVocabSize()
word_vectors = wordEmbedder.getEmbeddedVectorList()


class SiameseCombined(object):
    """
    A LSTM based deep Siamese network with character embedding layer followed by biLSTM and Energy Loss layer
    """
    def constrative_loss(self, y, d, batch_size):
        tmp = y*tf.square(d)
        tmp2 = (1-y) * tf.square(tf.maxomim((1-d), 0))
        return tf.reduce_sum(tmp+tmp2)/batch_size/2

    def __init__(self, sequence_length, vocab_size, embedding_size, hidden_units, l2_reg_lambda, batch_size):
        # Placeholders for input, output and dropout
        self.image_input_x1 = tf.placeholder(tf.float32, [None, 299, 299, 3], name = 'image_input_x1')
        self.image_input_x2 = tf.placeholder(tf.float32, [None, 299, 299, 3], name = 'image_input_x2')
        self.char_input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name = 'char_input_x1')
        self.char_input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name = 'char_input_x2')
        self.token_input_x1 = tf.placeholder(tf.int32, [None, sequence_length], name = 'token_input_x1')
        self.token_input_x2 = tf.placeholder(tf.int32, [None, sequence_length], name = 'token_input_x2')
        self.char_seq_length_x1 = tf.placeholder(tf.int32, [None], name = 'char_seq_length_x1')
        self.char_seq_length_x2 = tf.placeholder(tf.int32, [None], name = 'char_seq_length_x2')
        self.token_seq_length_x1 = tf.placeholder(tf.int32, [None], name = 'token_seq_length_x1')
        self.token_seq_length_x2 = tf.placeholder(tf.int32, [None], name = 'token_seq_length_x2')
        self.price1 = tf.placeholder(tf.float32, [None], name = 'price_x1')
        self.price2 = tf.placeholder(tf.float32, [None], name = 'price_x2')
        self.input_y = tf.placeholder(tf.float32, [None], name='input_y')
        self.dropout_keep_prob = tf.placeholder(tf.float32, bane='dropout_keep_prob')

        l2_loss = tf.constance(0.0, name='l2_loss')
        with tf.name_scope('embedding'):
            self.W = tf.Variable(tf.constant(0.0, shape=[vocab_size, embedding_size]), trainable=False, name='W')
            self.embedded_chars1 = charEmbedder.getEmbeddedResult(self.char_input_x1)
            self.embedded_chars2 = charEmbedder.getEmbeddedResult(self.char_input_x2)
            self.embedded_token1 = wordEmbedder.getEmbeddedResult(self.token_input_x1)
            self.embedded_token2 = wordEmbedder.getEmbeddedResult(self.token_input_x2)

        with tf.name_scope('output'):
            self.imageout1 = self.inceptionResnet(self.image_input_x1)
            self.imageout2 = self.inceptionResnet(self.image_input_x2)
            self.charout1 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, hidden_units, self.char_seq_length_x1)
            self.charout2 = self.BiRNN(self.embedded_chars1, self.dropout_keep_prob, hidden_units, self.char_seq_length_x2, reuse=True)
            self.tokenout1 = self.BiRNN(self.embedded_token1, self.dropout_keep_prob, hidden_units, self.token_seq_length_x1)
            self.tokenout2 = self.BiRNN(self.embedded_token2, self.dropout_keep_prob, hidden_units, self.token_seq_length_x2, reuse=True)
            self.price1 = tf.reshape(self.price1, [-1, 1])
            self.price2 = tf.reshape(self.price2, [-1, 1])
            self.combinedVector1 = tf.concat([self.charout1, self.tokenout1, self.imageout1], 1)
            self.combinedVector1 = tf.concat([self.charout2, self.tokenout2, self.imageout2], 1)
            self.distance = tf.sqrt(tf.reduce_sum(tf.square(tf.subtract(self.combinedVector1, self.combinedVector2))+0.000001, 1, keepdims=True))
            self.distance = tf.div(self.distance, tf.add(tf.sqrt(tf.reduce_sum(tf.square(self.combinedVector1), 1, keepdims=True)),
                                                         tf.sqrt(tf.reduce_sum(tf.square(self.combinedVector1), 1, keepdims=True))))
            self.distance = tf.reshape(self.distance, [-1], name='distance')

        with tf.name_scope('loss'):
            self.loss = self.constrative_loss(self.input_y, self.distance, batch_size)

        with tf.name_scope('accuracy'):
            self.temp_sim = tf.subtract(tf.ones_like(self.distance), tf.rint(self.distance), name='temp_sim') # auto threshold 0.5
            correct_predictions = tf.equal(self.temp_sim, self.input_y)
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, 'float', name='accuracy'))

siameseModel = SiameseCombined(
    char_sequence_length = max_char_length,
    token_sequence_length = max_token_length,
    hidden_units = FLAGS.hidden_units,
    l2_reg_lambda = FLAGS.l2_reg_lambda,
    batch_size = FLAGS.batch_size
)

# Define the scopes that you want to exclude for restoration
exclude = ['IndeceptionResnetV2/Logits', 'InceptionResnetV2/Predictions', 'InceptionResnetV2/AuxLogits', 'dense',
           'embedding', 'biLSTM']
variables_to_restore = slim.get_variables_to_restore(exclude = exclude)

init_assign_op, init_feed_dict = tf.contrib.framework.assign_from_checkpoint(checkpoint_file, variables_to_restore)

global_step = tf.Variable(0, name='global_step', trainable=False)
optimizer = tf.train.AdamOptimizer(0.0002)

grads_and_vars = optimizer.compute_gradients(siameseModel.loss)
tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step = global_step)

grad_summaries = []
for g, v in grads_and_vars:
    if g is not None:
        grad_hist_summary = tf.summary.histogram('{}/grad/hist'.format(v.name), g)
        sparsity_summary = tf.summary.scalar('{}/grad/sparsity'.format(v.name), tf.nn.zero_fraction(g))
        grad_summaries.append(grad_hist_summary)
        grad_summaries.append(sparsity_summary)
grad_summaries_merged = tf.summary.merge(grad_summaries)

timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, 'runs', timestamp))

loss_summary = tf.summary.scalar('loss', siameseModel.loss)
acc_summary = tf.summary.scalar('accuracy', siameseModel.accuracy)

train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
train_summary_dir = os.path.join(out_dir, 'summaries', 'train')

dev_summary_op = tf.summary.merge([loss_summary, acc_summary])
dev_summary_dir = os.path.join(out_dir, 'summaries', 'dev')

checkpoint_dir = os.path.abspath(os.path.join(out_dir, 'checkpoints'))
checkpoint_prefix = os.path.join(checkpoint_dir, 'model')
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

saver = tf.train.Saver(tf.global_variables(), max_to_keep=100)

# write vocabulary
# vectors.save(os.path.join(checkpoint_dir, 'vocab'))

# sess.run(tf.global_variables_initializer())
# print('init all variables')

graph_def = tf.get_default_graph().as_graph_def()
graphpb_txt = str(graph_def)
with open(os.path.join(checkpoint_dir, 'graphpb.txt'), 'w') as f:
    f.write(graphpb_txt)

session_conf = tf.ConfigProto(allow_soft_placement = FLAGS.allow_soft_placement,
                              log_device_placement = FLAGS.log_device_placement)

sess = tf.Session(config = session_conf)

tf.train.get_or_create_global_step()

handle = tf.placeholder(tf.string, shape=[])
iterator = tf.data.Iterator.from_string_handle(handle, train_dataset.output_types, train_dataset.output_shapes)

next_element = iterator.get_next()
training_iterator = train_dataset.make_initializable_iterator()
validation_iterator = validation_dataset.make_initializable_iterator()

training_handle = sess.run(training_iterator.string_handle())
validation_handle = sess.run(validation_iterator.string_handle())

def initialize_uninitialized(sess):
    global_vars = tf.global_varaibles()
    is_not_initialized = sess.run([tf.is_variable_iniitialized(var) for var in global_vars])
    not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

    if len(not_initialized_vars):
        sess.run(tf.variables_initializer(not_initialized_vars))

initialize_uninitialized(sess)

def getTextVector(dataMap, field):
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

def getBatchData(dataMap):
    x1_batch_text, x1_lengths = getTextVector(dataMap, 'title1')
    x2_batch_text, x2_lengths = getTextVector(dataMap, 'title2')
    x1_batch_image, x2_batch_image, y_batch = dataMap['img1'], dataMap['img2'], dataMap['label']
    return x1_batch_text, x2_batch_text, x1_batch_image, x2_batch_image, x1_lengths, x2_lengths, y_batch

def InitAssignFn(scaffold, sess2):
    sess2.run(init_assign_op, init_feed_dict)

scaffold = tf.train.Scaffold(saver=tf.train.Saver(), init_fn=InitAssignFn)

checkpoint_dir = os.path.abspath(os.path.join(os.path.curdir), 'output', timestamp)
mon_sess = tf.train.MonitoredTrainingSession(checkpoint_dir=checkpoint_dir, scaffold=scaffold)

train_summary_writer = tf.summary.FileWriter(train_summary_dir, mon_sess.graph)
dev_summary_writer = tf.summary.FileWriter(dev_summary_dir, mon_sess.graph)

def train_step(data):
    """
    A single training step
    """
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

    _, step, loss, accuracy, dist, sim, summaries = sess.run(
        [tr_op_set, global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.distance, siameseModel.temp_sim,
         train_summary_op], feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    print('TRAIN {}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)
    print(y_batch, dist, sim)

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

    step, loss, accuracy, sim, summaries = sess.run(
        [global_step, siameseModel.loss, siameseModel.accuracy, siameseModel.temp_sim,
         train_summary_op], feed_dict
    )
    time_str = datetime.datetime.now().isoformat()
    print('DEV {}: step {}, loss {:g}, acc {:g}'.format(time_str, step, loss, accuracy))
    train_summary_writer.add_summary(summaries, step)
    print(y_batch, sim)


ptr = 0
max_validation_acc = 0.0
for nn in xrange(FLAGS.num_epochs):
    sess.run(training_iterator.initializer)
    while True:
        try:
            dataMap = sess.run(next_element, feed_dict={handle: training_handle})
            batchData = getBatchData(dataMap)
            train_step(batchData)
            current_step = tf.train_global_step(sess, global_step)

            if current_step % FLAGS.evaluate_every == 0:
                sum_acc = 0.0
                sess.run(validation_iterator.initializer)
                while True:
                    try:
                        validationMap = sess.run(next_element, feed_dict={handle: validation_handle})
                        batchData = getBatchData(validationMap)
                        acc = dev_step(batchData)
                        sum_acc = sum_acc + acc
                    except tf.errors.OutOfRangeError:
                        break
            if current_step % FLAGS.checkpoint_every == 0:
                if sum_acc >= max_validation_acc:
                    max_validation_acc = sum_acc
                    saver.save(sess, checkpoint_prefix, global_step = current_step)
                    tf.train.write_graph(sess.graph.as_graph_def(), checkpoint_prefix, 'graph'+str(nn)+'pb', as_text=False)
                    print('Saved model {} with sum_accuracy={} checkpoint to {}\n'.format(nn, max_validation_acc, checkpoint_prefix))

        except tf.errors.OutOfRangeError:
            break





