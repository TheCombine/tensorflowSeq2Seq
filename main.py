from DataEngine import *
from Vocabulary import *
from DataIterator import *
import tensorflow as tf
from tensorflow.python.layers.core import Dense
import numpy as np
import os
import logging
import time

de = DataEngine()
vocab = LoadVocabulary()
if vocab == None:
    vocab = Vocabulary()
    load = [o1test, o1train, o1valid, o2test, o2train, o2valid, o3test, o3train, o3valid]
    for l in load:
        TokensObj, TokensSrc = de.LoadData(l)
        vocab.FeedTokensObj(TokensObj)
        vocab.FeedTokensSrc(TokensSrc)
        print ("done inspecting ", l)
    vocab.SaveVocabulary()
# most_lines_enc: 58, longest_line_enc: 14

enc_vocab_size = len(vocab.index_to_token_obj)
dec_vocab_size = len(vocab.index_to_token_src)
state_size = 256

# INPUTS
X = tf.placeholder(tf.int32, [None, None], 'X') # [batch_size, line_length]
X_len = tf.placeholder(tf.int32, [None], 'X_len') # [batch_size]
Y = tf.placeholder(tf.int32, [None, None], 'Y') # [batch_size, line_length]
Y_len = tf.placeholder(tf.int32, [None], 'Y_len') # [batch_size]
Y_targets = tf.placeholder(tf.int32, [None, None], 'Y_targets') # [batch_size, line_length]

# ENCODER embedding layer
enc_init_scale = 1. / np.sqrt(1./enc_vocab_size)
enc_embeddings = tf.Variable(tf.random_uniform([enc_vocab_size, state_size], -1 * enc_init_scale, enc_init_scale))
enc_inputs = tf.nn.embedding_lookup(enc_embeddings, X)

# ENCODER
enc_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
enc_outputs, enc_final_state = tf.nn.dynamic_rnn(cell=enc_cell, 
                                                 inputs=enc_inputs, 
                                                 sequence_length=X_len,
                                                 dtype=tf.float32) # , time_major=True

# ATTENTION
attention_mechanism = tf.contrib.seq2seq.LuongAttention(
    num_units=state_size, 
    memory=enc_outputs, # [batch_size, line_length]
    memory_sequence_length=X_len)

# DECODER embedding layer
dec_init_scale = 1. / np.sqrt(1./dec_vocab_size)
dec_embeddings = tf.Variable(tf.random_uniform([dec_vocab_size, state_size], -1 * dec_init_scale, dec_init_scale))
dec_inputs = tf.nn.embedding_lookup(dec_embeddings, Y)

# DECODER
dec_cell = tf.nn.rnn_cell.BasicLSTMCell(state_size)
dec_cell = tf.contrib.seq2seq.AttentionWrapper(cell=dec_cell, 
                                               attention_mechanism=attention_mechanism, 
                                               attention_layer_size=state_size)
dec_helper = tf.contrib.seq2seq.TrainingHelper(inputs=dec_inputs, 
                                               sequence_length=Y_len)
projection_layer = Dense(dec_vocab_size)
dec_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, 
                                              helper=dec_helper,
                                              initial_state=dec_cell.zero_state(batch_size=tf.size(X_len), dtype=tf.float32).clone(cell_state=enc_final_state), 
                                              output_layer=projection_layer)
dec_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=dec_decoder, 
                                                      impute_finished=True, 
                                                      maximum_iterations=tf.reduce_max(Y_len))
dec_logits = dec_outputs.rnn_output

# LOSS
masks = tf.sequence_mask(lengths=Y_len, 
                         maxlen=tf.reduce_max(Y_len), 
                         dtype=tf.float32)
loss = tf.contrib.seq2seq.sequence_loss(logits=dec_logits, 
                                        targets=Y_targets, 
                                        weights=masks)

# BACKWARD
params = tf.trainable_variables()
gradients = tf.gradients(ys=loss, xs=params)
clipped_gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
train_op = tf.train.AdamOptimizer().apply_gradients(zip(clipped_gradients, params))

# PREDICTION
pred_helper = tf.contrib.seq2seq.GreedyEmbeddingHelper(embedding=dec_embeddings,
                                                       start_tokens=tf.fill([tf.size(X_len)], 1), 
                                                       end_token=2)
pred_decoder = tf.contrib.seq2seq.BasicDecoder(cell=dec_cell, 
                                               helper=pred_helper, 
                                               initial_state=dec_cell.zero_state(batch_size=tf.size(X_len), dtype=tf.float32).clone(cell_state=enc_final_state), 
                                               output_layer=projection_layer)
pred_outputs, _, _ = tf.contrib.seq2seq.dynamic_decode(decoder=pred_decoder, 
                                                       impute_finished=True, 
                                                       maximum_iterations=512) # , maximum_iterations=324 (323+1)

n_epochs = 8
batch_size = 32

def LogInfo(message):
    logging.info("%s | %s" % (time.ctime(), message))
logging.basicConfig(filename='logs.txt',level=logging.INFO)

# Add ops to save and restore all the variables.
saver = tf.train.Saver()
savePath = os.path.dirname(os.path.abspath(__file__)) + "\savesModel\model.ckpt"
print ("Save path:", savePath)
LogInfo("Save path: %s" % savePath)
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    try:
        saver.restore(sess, savePath)
        print("Model restored.")
        LogInfo("Model restored.")
    except:
        print("No model save found.")
        LogInfo("No model save found.")
    print ("Beginning training. Epochs: %s, Batch size: %s" % (n_epochs, batch_size))
    for epoch in range(n_epochs):
        epoch_loss = 0
        LogInfo("Beginning epoch %s out of %s" % (epoch + 1, n_epochs))
        load = [o1train, o2train, o3train]
        for l in load:
            TokensObj, TokensSrc = de.LoadData(l)
            IndicesObj = vocab.TokensToIndicesObj(TokensObj)
            IndicesSrc = vocab.TokensToIndicesSrc(TokensSrc)
            di = DataIterator(IndicesObj, IndicesSrc)
            while True:
                batch = di.GetNextBatchNLP(32)
                if batch == None:
                    break
                _, c = sess.run([train_op, loss], feed_dict={X: batch[0], X_len: batch[1], Y: batch[2], Y_len: batch[3], Y_targets: batch[4]})
                epoch_loss += c
                # LogInfo("Batch loss: %s" % c)
            LogInfo("Epoch loss: %s" % epoch_loss)
            saver.save(sess, savePath)
            LogInfo("Model saved.")

    #di = DataIterator(IndicesObj, IndicesSrc)
    #feed_X, feed_X_len, feed_Y, feed_Y_len, feed_Y_targets = di.GetNextBatchNLP(1)
    #predictions = sess.run([pred_outputs], feed_dict={X: feed_X, X_len: feed_X_len})
    
    #print ("expected:")
    #print (feed_Y_targets[0])
    #print ("prediction:")
    #print (predictions[0].sample_id)

#def length(sequence):
#  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
#  length = tf.reduce_sum(used, 1)
#  length = tf.cast(length, tf.int32)
#  return length

#max_length = 14
#frame_size = vocab.sizeObj()
#num_hidden = 64

#sequence = tf.placeholder(tf.float32, [None, max_length, frame_size])
#output, state = tf.nn.dynamic_rnn(
#    cell=tf.contrib.rnn.BasicLSTMCell(num_hidden),
#    inputs=sequence,
#    sequence_length=length(sequence),
#    dtype=tf.float32
#)

