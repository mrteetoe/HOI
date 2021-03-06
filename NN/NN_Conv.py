import nn_functions as funcs
import tensorflow as tf
import numpy as np
import math
import datetime
import sys
import time

#For setting printing options
np.set_printoptions(linewidth=400, threshold=int(1e4), edgeitems=6)

###Variables to alter###
obs_train_perc = 0.9
sim_train_perc = 0.9
obs_label=0.1
sim_label=0.9
pho_label=0.8
total_OO=5e4
total_SO=1e5
  
filter_width=10
output_channels=10
stride_length=2
keep_rate=1.0
num_of_epochs=1
learning_rate=0.001
batch_size=100
network_layout=[250, 0, 120, 60, 30, 15, 1]
num_strides=(network_layout[0]/5)/stride_length
network_layout[1]=num_strides*output_channels

########################
print('')
print('The network layout is: %s'%(network_layout))
print('Keep rate: %s'%(keep_rate))
print('Learning rate: %s'%(learning_rate))
print('Batch size: %s'%(batch_size))
print('Filter width: %s'%(filter_width))
print('Output channels: %s'%(output_channels))
print('Stride length: %s'%(stride_length))
print('Number of strides: %s'%(num_strides))
print('')

#Checks to make sure the decider is correctly formatted
#Best to do this first thing...
decider=str(sys.argv[1])
if decider!="new" and decider!="load":
    print('Invalid input')
    raise
weights_token=str(sys.argv[2])

#Loads the three datasets
print('Loading the data this could take a minute...')
OO=np.load('Data/oe_matrices/oo_matrixN.npy')
SO=np.load('Data/oe_matrices/so_MatrixSept.npy')
PHO=np.load('Data/oe_matrices/pho_matrixN.npy')
print('Done!')

#num_PHO,_=PHO.shape
#for i in range(num_PHO):
#    if int(PHO[i,-1])==101955:
#        print(PHO[i,5:10])

#Throws some of the data out (randomly)
np.random.shuffle(OO)
np.random.shuffle(SO)
OOc=OO[:int(total_OO)]
SOc=SO[:int(total_SO)]

#Creates the training and test set
#SO and OO are chosen randomly
#PHO are always in the test set
num_OO,_=OOc.shape
num_SO,_=SOc.shape
num_train_OO=int(num_OO*obs_train_perc)
num_train_SO=int(num_SO*sim_train_perc)
num_test_OO=num_OO-num_train_OO
num_test_SO=num_SO-num_train_SO
train_set=np.concatenate((OOc[:num_train_OO],SOc[:num_train_SO]),axis=0)
test_set=np.concatenate((OOc[num_train_OO:],SOc[num_train_SO:]),axis=0)
test_set=np.concatenate((test_set,PHO),axis=0)
num_train,_=train_set.shape
num_test,_=test_set.shape

#Creates labels for train and test sets
#Assumes sets are stack OO, SO, PHO top from bottom
train_labels=np.zeros(num_train)
for i in range(num_train):
    if i<num_train_OO:
        train_labels[i]=obs_label
    elif i>=num_train_OO:
        train_labels[i]=sim_label
train_labels=train_labels.reshape((num_train,1))
test_labels=np.zeros(num_test)
for i in range(num_test):
    if i<num_test_OO:
        test_labels[i]=obs_label
    elif i>=num_test_OO and i<(num_test_OO+num_test_SO):
        test_labels[i]=sim_label
    else:
        test_labels[i]=pho_label
test_labels=test_labels.reshape((num_test,1))

###########################################
###Creates, trains, and test the network###
###########################################

if decider=="new":
    weights, biases = funcs.initialize_weights_conv(filter_width, network_layout, output_channels)

elif decider=="load":
    weights, biases = funcs.load_weights_conv(weights_token)

# tf Graph input
x = tf.placeholder("float", [None, network_layout[0]])
y_ = tf.placeholder("float", [None, network_layout[-1]])
keep_prob = tf.placeholder(tf.float32)

#Defines the 2 convolution networks for the training and test stages
conv_train = funcs.conv_1D(x, weights, biases, num_strides, stride_length, batch_size, keep_prob, network_layout)
conv_test = funcs.conv_1D(x, weights, biases, num_strides, stride_length, num_test, keep_prob, network_layout)

#Defines the cross entropy function and its optimizer
cross_entropy = tf.nn.weighted_cross_entropy_with_logits(targets=y_,logits=conv_train,pos_weight=0.4)
train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)

#Starts the interactive session
print('\nStarting session...')
start_time=datetime.datetime.now()
sess=tf.InteractiveSession()
sess.run(tf.global_variables_initializer())
end_time=datetime.datetime.now()
elapsed_time=end_time-start_time
print('Done!')
print('%s was spent starting the session.\n'%(elapsed_time))

#Sets the batch size and epochs
num_training_steps=int(num_train/batch_size)

#Trains the model
variables=conv_test, y_
evaluation_frequency=1
epoch_counter=0
save_counter=0
learning_rate_counter=0
loss_counter=0

print('\nTraining commencing...\n')
for epoch in range(num_of_epochs):
    np.random.shuffle(OO)
    np.random.shuffle(SO)
    OOc=OO[:int(total_OO)]
    SOc=SO[:int(total_SO)]
    train_set=np.concatenate((OOc[:num_train_OO],SOc[:num_train_SO]),axis=0)
    train_labels=np.zeros(num_train)
    for i in range(num_train):
        if i<num_train_OO:
            train_labels[i]=obs_label
        elif i>=num_train_OO:
            train_labels[i]=sim_label
    train_labels=train_labels.reshape((num_train,1))
    funcs.shuffle_in_unison(train_set, train_labels)
    total_loss=0
    total_entries=0
    #train_step = tf.train.AdamOptimizer(learning_rate).minimize(cross_entropy)
    for i in range(num_training_steps): 
        _, loss_val = sess.run([train_step, cross_entropy], feed_dict={x:train_set[i*batch_size:(i+1)*batch_size,:-1], y_:train_labels[i*batch_size:(i+1)*batch_size], keep_prob:keep_rate})
        if np.isnan(np.sum(loss_val))*1 == 0:
            total_loss+=np.sum(loss_val)
            total_entries+=loss_val.size
 
    #Prints out the evaluation of the network's performance at a defined interval based off of test set
    epoch_counter+=1
    if epoch%evaluation_frequency==0:
        guesses, answers=sess.run(variables, feed_dict={x:test_set[:,:-1], y_:test_labels, keep_prob:1.0})
        if loss_counter>0:
            last_loss=avg_loss
        elif loss_counter==0:
            last_loss=1
            loss_counter+=1
        avg_loss=total_loss/total_entries
        print('The average loss, per training vector, is %s for the %s epoch'%(avg_loss, epoch))
                         
#Performs some evaluations       		
eval_matrix=np.zeros((guesses.size,3))
eval_matrix[:,0]=guesses.reshape(guesses.size)
eval_matrix[:,1]=test_labels.reshape(guesses.size)
eval_matrix[:,2]=test_set[:,-1] #These are the names
hazard_matrix=funcs.eval_function(eval_matrix, obs_label, sim_label, pho_label, test_set[:,:-1], weights_token)

#Saves the weights for future use
if funcs.query_yes_no('Save weights and biases?') == True:
    funcs.save_weights_conv(sess.run(weights), sess.run(biases), weights_token)

