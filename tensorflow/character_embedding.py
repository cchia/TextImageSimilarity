import tensorflow as tf
import collections
import numpy as np
import re, random, math, os, codecs, sys
from tensorflow.contrib.tensorboard.plugins import projector

reload(sys)
sys.setdefaultencoding('utf8')

vocabulary_size = 1600

corpus = []
with codecs.open('/data/doc.txt', 'r', 'utf-8') as fin:
    for line in fin:
        #remove horizontal whitespace character
        # \t\xA0\u180e\u2000-\u200a\u202f\u205f\u3000
        line = line.replace('\t', '').replace(u'\xA0', '').replace(u'\u180e', '')
        line = line.replace(u'\u2000', '').replace(u'\u200a', '').replace(u'\u202f', '')
        line = line.replace(u'\u205f', '').replace(u'\u3000', '')
        chars = ['BEGIN']
        chars.extend([c for c in line])
        chars.append('END')
        corpus.extend(chars)

def build_dataset(words, n_words):
    """Process raw inputs into a dataset"""
    count = [['UNK', -1]]
    count.extend(collections.Counter(words).most_common(n_words-1))
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)
    data = list()
    unk_count = 0
    for word in words:
        index = dictionary.get(word, 0)
        if index == 0: #dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reversed_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return data, count, dictionary, reversed_dictionary

data, count, dictionary, reversed_dictionary = build_dataset(corpus, vocabulary_size)

def generate_batch(batch_size, num_skips, skip_window):
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)
    span = 2 * skip_window + 1 #skip_window target skip_window
    buffer = collections.deque(maxlen=span) #pylint: disable = redefined-builtin
    if data_index + span > len(data):
        data_index = 0
    buffer.extend(data[data_index: data_index + span])
    data_index += span
    for i in range(batch_size // num_skips):
        context_words = [w for w in range(span) if w != skip_window]
        words_to_use = random.sample(context_words, num_skips)
        for j, context_word in enumerate(words_to_use):
            batch[i * num_skips + j] = buffer[skip_window]
            labels[i * num_skips + j, 0] = buffer[context_word]
        if data_index == len(data):
            buffer.extend(data[0: span])
            data_index = span
        else:
            buffer.append(data[data_index])
            data_index += 1
    # Backtrack a little bit to avoid skipping words in the end of a batch
    data_index = (data_index + len(data) - span) % len(data)
    return batch, labels

data_index = 0
#batch, labels = generate_batch(batch_size=8, num_skips=2, skip_window=1)

batch_size = 128
embedding_size = 128 #dimension of the embedding vector
skip_window = 1 # How many words to consider left and right
num_skips = 2 # How many times to reuse an input to generate a label
num_sampled = 64 # Number of negative examples to sample

# We pick a random validation set to sample nearest neighbors. Here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent. These 3 variables are used only for
# displaying model accuracy, they don't affect calculation
valid_size = 16 # Random set of words to evaluate similarity on
valid_window = 100 # Only pick dev samples in the head of the distribution
valid_examples = np.random.choice(valid_window, valid_size, replace=False)

class Flag:
    log_dir = ''

FLAGS = Flag()
timestamp = str(int(time.time()))
out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
print("Writing to {}\n".format(out_dir))

FLAGS.log_dir = out_dir
graph = tf.Graph()

with graph.as_default():

    # Input data
    with tf.name_scope('inputs'):
        train_inputs = tf.placeholder(tf.int32, shape = [batch_size])
        train_labels = tf.placeholder(tf.int32, shape = [batch_size, 1])
        valid_dataset = tf.constant(valid_examples, dtype = tf.int32)

    # Ops and variables pinned to the CPU because of missing GPU implementation
    with tf.device('/cpu:0'):
        # Look up embeddings for inputs
        with tf.name_scope('embeddings'):
            embeddings = tf.Variable(tf.random_uniform([vocabulary_size, embedding_size], -1.0, 1.0))
            embed = tf.nn.embedding_lookup(embeddings, train_inputs)

        # Construct the variables for the NCE loss
        with tf.name_scope('weights'):
            nce_weights = tf.Variable(
                tf.truncated_normal([vocabulary_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
        with tf.name_scope('biases'):
            nce_biases = tf.Variable(tf.zeros([vocabulary_size]))

    # Compute the average NCE loss for the batch.
    # tf.nce_loss automatically draws a new sample of the negative labels each
    # time we evaluate the loss
    # Explanation of the meaning of NCE loss:
    # http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/
    with tf.name_scope('loss'):
        loss = tf.reduce_mean(
            tf.nn.nce_loss(
                weights = nce_weights,
                biases = nce_biases,
                labels = train_labels,
                inputs = embed,
                num_sampled = num_sampled,
                num_classes = vocabulary_size))

    # Add the loss value as a scalar to summary
    tf.summary.scalar('loss', loss)

    # Construct the SGD optimizer using a learning rate of 1.0
    with tf.name_scope('optimizer'):
        optimizer = tf.train.GradientDescentOptimizer(1.0).minimize(loss)

    # Compute the cosine similarity between minibatch examples and all embeddings
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keepdims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    similarity = tf.matmul(valid_embeddings, normalized_embeddings, transpose_b=True)

    # Merge all summaries
    merged = tf.summary.merge_all()

    # Add variable initializer
    init = tf.global_variables_initializer()

    # Create a saver
    saver = tf.train.Saver()

# Step 5: Begin training
num_steps = 100001

with tf.Session(graph=graph) as session:
    # Open a writer to write summaries
    writer = tf.summary.FileWriter(FLAGS.log_dir, session.graph)

    # We must initialize all variables before we use them
    init.run()
    print('Initialized')

    average_loss = 0
    for step in xrange(num_steps):
        batch_inputs, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_inputs: batch_inputs, train_labels: batch_labels}

        # Define metadata variable
        run_metadata = tf.RunMetadata()

        # We perform one update step by evaluating the optimizer op (including it
        # in the list of returned values for session.run()
        # Also, evaluate the merged op to get all summaries from the returned "summary" variable
        # Feed metadata variable to session for visualizing the graph in TensorBoard
        _, summary, loss_val = session.run(
            [optimizer, merged, loss],
            feed_dict = feed_dict,
            run_metadata = run_metadata
        )
        average_loss = loss_val

        # Add returned summaries to writer in each step
        writer.add_summary(summary, step)
        # Add metadata to visualize the graph for the last run
        if step == (num_steps - 1):
            writer.add_run_metadata(run_metadata, step, ': ', average_loss)
            average_loss = 0

        # Note that this is expensive (~20% slow down if computed every 500 steps_
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in xrange(valid_size):
                valid_word = reversed_dictionary[valid_examples[i]]
                top_k = 8 # Number of nearest neighbor
                nearest = (-sim[i, :]).argsort()[1:top_k + 1]
                log_str = 'Nearest to {}'.format(valid_word)
                for k in xrange(top_k):
                    close_word = reversed_dictionary[nearest[k]]
                    log_str = '{} {}'.format(log_str, close_word)
                print(log_str)
        final_embeddings = normalized_embeddings.eval()

        # Write corresponding labels for the embeddings
        with open(FLAGS.log_dir + '/metadata.tsv', 'w') as f:
            for i in xrange(vocabulary_size):
                f.write(reversed_dictionary[i] + '\n')

        # Save the model for checkpoints
        saver.save(session, os.path.join(FLAGS.log_dir, 'model.ckpt'))

        # Create a configuration for visualizing embeddings with the labels in TensorBoard
        config = projector.ProjectorConfig()
        embedding_conf = config.embeddings.add()
        embedding_conf.tensor_name = embeddings.name
        embedding_conf.metadata_path = os.path.join(FLAGS.log_dir, 'metadata.tsv')
        projector.visualize_embeddings(writer, config)

writer.close()


# pylint: disable = missing-docstring
# Function to draw visualization of distance between embeddings

def plot_with_labels(low_dim_embs, labels, filename):
    assert low_dim_embs.shape[0] >= len(labels) # More labels than embeddings
    plt.figure(figsize=(18, 18))
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label, xy=(x, y), xytest = (5, 2), textcoords= 'offset points', ha='right', va='bottom')
    plt.savefig(filename)

# try:
# pylint: disable=g-import-not-at-top
from sklearn.manifold import TSNE
import matplotlib
from matplotlib import pyplot as plt
from matplotlib import font_manager as fm

try:
    font_location = 'nanumgothic-regular.ttf'
    font_name = fm.FontProperties(fname=font_location).get_name()
    matplotlib.rcParams['font.family'] = font_name
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 1000
    low_dim_embs = tsne.fit_transfrom(final_embeddings[:plot_only, :])
    labels = [reversed_dictionary[i] for i in xrange(plot_only)]
    plot_with_labels(low_dim_embs, labels, 'tsne.png')

except ImportError as ex:
    print('Please install sklearn, matplotlib, as scipy to show embeddings.')
    print(ex)

# save embedding result
fout = codecs.open('/data/c2v_index.txt', 'w', 'utf-8')
for i in range(vocabulary_size):
    fout.write(str(i) + ':')
    fout.write(reversed_dictionary[i])
    fout.write('\n')
fout.close()

np.savez('/data/c2v_result', final_embeddings)
