import tensorflow as tf
import sys
import numpy as np
import nn_functions as funcs

obs_label=0.1
sim_label=0.9
pho_label=0.8
filter_width=10
output_channels=5
stride_length=2
network_layout=[250, 0, 60, 30, 15, 7, 1]
num_strides=(network_layout[0]/5)/stride_length
network_layout[1]=num_strides*output_channels
weights_token='Aug24_med1'

#Loads the three datasets
print('Loading the data this could take a minute...')
OO=np.load('Data/oo_matrix.npy')
SO=np.load('Data/so_matrix.npy')
PHO=np.load('Data/pho_matrix.npy')
print('Done!')

#Limits the amount of observed and simulated
max_obj=int(10)
np.random.shuffle(OO)
np.random.shuffle(SO)
OO=OO[:max_obj]
SO=SO[:max_obj]

#Creates the eval set and labels
num_OO,_=OO.shape
num_SO,_=SO.shape
num_PHO,_=PHO.shape
num_eval=num_OO+num_SO+num_PHO
eval_set=np.concatenate((OO,SO),axis=0)
eval_set=np.concatenate((eval_set,PHO))
eval_labels=np.zeros(num_eval)
for i in range(num_eval):
    if i<num_OO:
        eval_labels[i]=obs_label
    elif i>=num_OO and i<(num_OO+num_SO):
        eval_labels[i]=sim_label
    else:
        eval_labels[i]=pho_label
eval_labels=eval_labels.reshape((num_eval,1))

#Constructs NN
weights, biases = funcs.load_weights_conv(weights_token)
keep_prob = tf.placeholder(tf.float32)
x = tf.placeholder("float", [None, network_layout[0]])
y_ = tf.placeholder("float", [None, network_layout[-1]])
conv_test = funcs.conv_1D(x, weights, biases, num_strides, stride_length, num_eval, keep_prob, network_layout)
variables=conv_test, y_

#Runs the data through the NN
print('Running the data through the NN...')
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
guesses,answers=sess.run(variables, feed_dict={x:eval_set[:,:-1], y_:eval_labels, keep_prob:1.0})
print('Done!')

#Performs some evaluations       		
eval_matrix=np.zeros((guesses.size,3))
eval_matrix[:,0]=guesses.reshape(guesses.size)
eval_matrix[:,1]=eval_labels.reshape(guesses.size)
eval_matrix[:,2]=eval_set[:,-1] #These are the names
hazard_matrix=funcs.eval_function(eval_matrix, obs_label, sim_label, pho_label,eval_set[:,:-1], weights_token)


