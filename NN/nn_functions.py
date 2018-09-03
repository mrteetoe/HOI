import numpy as np
import math
import tensorflow as tf
from matplotlib import pyplot as plt
import sys

def eval_function(eval_M, obs_label, sim_label, pho_label, OEs, weight_token):
    fil_pho_list='Data/pho_list.txt'
    pho_list=list(np.genfromtxt(fil_pho_list, delimiter=','))
    threshold=0.5
    num_objects=eval_M.shape[0]
    num_haz_sim=0
    num_haz_obs=0
    num_haz_pho=0
    simulated_ratings=[]
    observed_ratings=[]
    PHO_ratings=[]
    topThree=[410777,101955, 29075]
    print('')
    for i in range(num_objects):
        hazard_rating=eval_M[i,0]
        label=eval_M[i,1]
        name=eval_M[i,2]
        if hazard_rating>=threshold and label==sim_label:
            num_haz_sim+=1
        elif hazard_rating>=threshold and label==obs_label:
            num_haz_obs+=1
        elif hazard_rating>=threshold and label==pho_label:
            num_haz_pho+=1
        if label==sim_label:
            simulated_ratings.append(hazard_rating)
        elif label==obs_label:
            observed_ratings.append(hazard_rating)
        elif label==pho_label:
            PHO_ratings.append(hazard_rating)
        if int(name) in topThree:
            print('The rating for %s was %s.'%(int(name),hazard_rating))
    num_observed=len(observed_ratings)
    num_simulated=len(simulated_ratings)
    num_pho=len(PHO_ratings)
    print('\nTotal objects: %s'%(num_objects))
    print('Out of the %s observed objects, %s were identified as hazardous (%.2f%%).'%(num_observed, num_haz_obs, float(num_haz_obs)/float(num_observed)*100.))
    print(sum(observed_ratings)/len(observed_ratings))
    print('Out of the %s simulated objects, %s were identified as hazardous (%.2f%%).'%(num_simulated, num_haz_sim, float(num_haz_sim)/float(num_simulated)*100.))
    print(sum(simulated_ratings)/len(simulated_ratings))
    print('Out of the %s PHOs, %s were identified as hazardous (%.2f%%).\n'%(num_pho, num_haz_pho, float(num_haz_pho)/float(num_pho)*100.))
    print(sum(PHO_ratings)/len(PHO_ratings))

    print('Creating hazard matrix...')
    hazard_matrix=np.zeros((num_objects, 13))
    #First column is name
    #Second column is hazard rating
    #Third column is the label
    #Fourth column through eighth are OEs means
    #ninth through thirteenth are OE STD
    for i in range(num_objects):
        hazard_rating=eval_M[i,0]
        label=eval_M[i,1]
        name=eval_M[i,2]
        hazard_matrix[i,0]=int(name)
        hazard_matrix[i,1]=label
        hazard_matrix[i,2]=hazard_rating
        OE_Matrix=OEs[i].reshape((50,5))
        
        #Fixes for normalization
        #And converts inclination to degrees
        OE_Matrix[:,0]*=10.
        OE_Matrix[:,2]*=(180./math.pi)
        OE_Matrix[:,3]/=1e7
        OE_Matrix[:,4]/=1e-16

        OE_means=np.mean(OE_Matrix,axis=0)
        OE_stds=np.std(OE_Matrix,axis=0)
        hazard_matrix[i,3:8]=OE_means
        hazard_matrix[i,8:13]=OE_stds
    np.savetxt("Data/Hazard_Matrices/"+weight_token+".csv", hazard_matrix, delimiter=",")    
    print('Done!')    

    return hazard_matrix
    
def shuffle_in_unison(a, b):
    rng_state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(rng_state)
    np.random.shuffle(b)
    
def query_yes_no(question, default="yes"):
    """Ask a yes/no question via raw_input() and return their answer.

    "question" is a string that is presented to the user.
    "default" is the presumed answer if the user just hits <Enter>.
        It must be "yes" (the default), "no" or None (meaning
        an answer is required of the user).

    The "answer" return value is True for "yes" or False for "no".
    """
    valid = {"yes": True, "y": True, "ye": True,
             "no": False, "n": False}
    if default is None:
        prompt = " [y/n] "
    elif default == "yes":
        prompt = " [Y/n] "
    elif default == "no":
        prompt = " [y/N] "
    else:
        raise ValueError("invalid default answer: '%s'" % default)

    while True:
        sys.stdout.write(question + prompt)
        choice = raw_input().lower()
        if default is not None and choice == '':
            return valid[default]
        elif choice in valid:
            return valid[choice]
        else:
            sys.stdout.write("Please respond with 'yes' or 'no' "
                             "(or 'y' or 'n').\n")

#######################     
###Convolution Stuff###
#######################
dir_weights='NN/Weights_Biases/'
def initialize_weights_conv(filter_width, network_layout, output_channels):
    weights = {
	    'h_conv': tf.Variable(tf.truncated_normal([filter_width, 5, output_channels])),
        'h1': tf.Variable(tf.truncated_normal([network_layout[1], network_layout[2]])/math.sqrt(network_layout[1])),
        'h2': tf.Variable(tf.truncated_normal([network_layout[2], network_layout[3]])/math.sqrt(network_layout[2])),
        'h3': tf.Variable(tf.truncated_normal([network_layout[3], network_layout[4]])/math.sqrt(network_layout[3])),
        'h4': tf.Variable(tf.truncated_normal([network_layout[4], network_layout[5]])/math.sqrt(network_layout[4])),
	    'out': tf.Variable(tf.truncated_normal([network_layout[5], network_layout[-1]])/math.sqrt(network_layout[5]))
    }
    biases = {
	    'b_conv': tf.Variable(tf.zeros(output_channels)),
        'b1': tf.Variable(tf.zeros(network_layout[2])),
        'b2': tf.Variable(tf.zeros(network_layout[3])),
        'b3': tf.Variable(tf.zeros(network_layout[4])),
        'b4': tf.Variable(tf.zeros(network_layout[5])),
	    'out': tf.Variable(tf.zeros(network_layout[-1]))
    }  
    return weights, biases

def save_weights_conv(weights, biases, weights_token):
    np.save(dir_weights+'h_conv_weights_'+weights_token, weights['h_conv'])
    np.save(dir_weights+'h1_weights_'+weights_token, weights['h1'])
    np.save(dir_weights+'h2_weights_'+weights_token, weights['h2'])
    np.save(dir_weights+'h3_weights_'+weights_token, weights['h3'])
    np.save(dir_weights+'h4_weights_'+weights_token, weights['h4'])
    np.save(dir_weights+'out_weights_'+weights_token, weights['out'])
    np.save(dir_weights+'b_conv_biases_'+weights_token, biases['b_conv'])
    np.save(dir_weights+'b1_biases_'+weights_token, biases['b1'])
    np.save(dir_weights+'b2_biases_'+weights_token, biases['b2'])
    np.save(dir_weights+'b3_biases_'+weights_token, biases['b3'])
    np.save(dir_weights+'b4_biases_'+weights_token, biases['b4'])
    np.save(dir_weights+'out_biases_'+weights_token, biases['out'])

def load_weights_conv(weights_token):
    weights = {
        'h_conv': tf.Variable(np.load(dir_weights+'h_conv_weights_'+weights_token+'.npy')),
        'h1': tf.Variable(np.load(dir_weights+'h1_weights_'+weights_token+'.npy')),
        'h2': tf.Variable(np.load(dir_weights+'h2_weights_'+weights_token+'.npy')),
        'h3': tf.Variable(np.load(dir_weights+'h3_weights_'+weights_token+'.npy')),
        'h4': tf.Variable(np.load(dir_weights+'h4_weights_'+weights_token+'.npy')),
        'out': tf.Variable(np.load(dir_weights+'out_weights_'+weights_token+'.npy'))
    }
    biases = {
        'b_conv': tf.Variable(np.load(dir_weights+'b_conv_biases_'+weights_token+'.npy')),
        'b1': tf.Variable(np.load(dir_weights+'b1_biases_'+weights_token+'.npy')),
        'b2': tf.Variable(np.load(dir_weights+'b2_biases_'+weights_token+'.npy')),
        'b3': tf.Variable(np.load(dir_weights+'b3_biases_'+weights_token+'.npy')),
        'b4': tf.Variable(np.load(dir_weights+'b4_biases_'+weights_token+'.npy')),
        'out': tf.Variable(np.load(dir_weights+'out_biases_'+weights_token+'.npy'))
    } 
    return weights, biases

def conv_1D(x, weights, biases, num_strides, stride_length, load_amount, keep_prob, network_layout):
    
    #Computes the convolution layer
    x_image = tf.reshape(x, [load_amount, network_layout[0]/5, 5])
    layer_conv=tf.add(tf.nn.conv1d(x_image, weights['h_conv'], padding="SAME", stride=stride_length), biases['b_conv'])
    layer_conv=tf.nn.relu(layer_conv)
    layer_conv_flat = tf.reshape(layer_conv, [load_amount, network_layout[1]])

    #Computes the first fully connected layer
    layer_1=tf.add(tf.matmul(layer_conv_flat, weights['h1']), biases['b1'])
    layer_1=tf.sigmoid(layer_1)
    drop_out_1 = tf.nn.dropout(layer_1, keep_prob)

    #Computes the second fully connected layer
    layer_2=tf.add(tf.matmul(drop_out_1, weights['h2']), biases['b2'])
    layer_2=tf.sigmoid(layer_2)
    drop_out_2 = tf.nn.dropout(layer_2, keep_prob)

    #Computes the third fully connected layer
    layer_3=tf.add(tf.matmul(drop_out_2, weights['h3']), biases['b3'])
    layer_3=tf.sigmoid(layer_3)
    drop_out_3 = tf.nn.dropout(layer_3, keep_prob)

    #Computes the third fully connected layer
    layer_4=tf.add(tf.matmul(drop_out_3, weights['h4']), biases['b4'])
    layer_4=tf.sigmoid(layer_4)
    drop_out_4 = tf.nn.dropout(layer_4, keep_prob)

    #Computes output fully connected layer
    out_layer = tf.add(tf.matmul(drop_out_4, weights['out']), biases['out'])
    out_layer = tf.sigmoid(out_layer)

    return out_layer
