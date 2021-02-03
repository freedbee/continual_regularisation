#!/usr/bin/env python
# coding: utf-8
import os, sys, time
start = time.time()
end = time.time()
import tensorflow as tf
import numpy as np
import keras
from keras.datasets import cifar10, cifar100
import pickle
import utils


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','SI', '0.1', '1.0', '1']
inputs = sys.argv
visible_GPU = inputs[1]
save_outputs_to_log_dir = True

### HYPERPARAMETERS
# for method choose SI, SIU, SIB, SOS
HP = {\
'seed'                  : int(inputs[2]),\
'method'                : inputs[3],\
'c'                     : float(inputs[4]),\
're_init_model'         : bool(float(inputs[5])),\
'rescale'               : float(inputs[6]),\
'evaluate_on_validation': bool(float(inputs[7])),\
'batch_size'            : int(inputs[8]),\
'n_tasks'               : 6,\
'n_epochs_per_task'     : 60,\
'equal_iter'            : bool(float(inputs[9])),\
'SOS_alpha'             : float(inputs[10]),\

}

if HP['rescale']== 1.0: #set damping for division as in SI paper. Changes here will also need changes in HP['c'].
    HP['damp'] = 0.001 
if HP['rescale']== 0.0: #turn of dividing by length.
    HP['damp'] = 1.0 

#############
#OTHER PARAMETERS
#############
train_share = 0.9 #remaing part is validation set
p_c = 0.25 #dropout for convolutional layer
p_l = 0.5 #dropout for fully connected layer
input_dim = 32
n_channel = 3



    
HP_label = 'large'
for item in HP.items():
    HP_label += '__'
    HP_label += item[0]
    HP_label += '='
    HP_label += str(item[1])

# VISIBLE GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=visible_GPU
if save_outputs_to_log_dir:
    orig_stdout = sys.stdout
    f = open('logs/log_'+HP_label+'.txt', 'w')
    sys.stdout = f

# SET UP TF
tf.random.set_random_seed(HP['seed'])
np.random.seed(HP['seed'])
tf.reset_default_graph()



###############################
#IMPORT AND PREPROCESS DATASET
###############################
(cifar10_x, cifar10_y), (cifar10_x_test, cifar10_y_test) = tf.keras.datasets.cifar10.load_data()
cifar10_train = (cifar10_x.astype('float32')/255, cifar10_y)
cifar10_test = (cifar10_x_test.astype('float32')/255, cifar10_y_test)
print('cifar10x shape ', cifar10_x.shape)

(cifar100_x, cifar100_y), (cifar100_x_test, cifar100_y_test) = tf.keras.datasets.cifar100.load_data()
cifar100_train = (cifar100_x.astype('float32')/255, cifar100_y)
cifar100_test = (cifar100_x_test.astype('float32')/255, cifar100_y_test)
print('cifar100x shape ', cifar100_x.shape)

###########
#CREATE SPLIT VERSION OF DATASET
###########
cifar10_split_train, cifar10_split_test = utils.split_dataset(cifar10_train, cifar10_test)
cifar100_split_train, cifar100_split_test = utils.split_dataset(cifar100_train, cifar100_test)

cifar10_grouped_train, cifar10_grouped_test = utils.mix_dataset(cifar10_split_train, cifar10_split_test, group_size=10)
cifar100_grouped_train, cifar100_grouped_test = utils.mix_dataset(cifar100_split_train, cifar100_split_test, group_size=10)

dataset_train = [*cifar10_grouped_train, *cifar100_grouped_train[:HP['n_tasks']-1]]
dataset_test = [*cifar10_grouped_test, *cifar100_grouped_test[:HP['n_tasks']-1]]

output_head_dimensions = [len(x[1][0]) for x in dataset_train] #need to be equal for implementation to work (tf.stack)



#############
#PLACEHOLDERS
##############
#XX = tf.placeholder(tf.float32, [None,28,28])
#X = tf.reshape(XX,[-1,784])
X_ph = tf.placeholder(tf.float32,[None, input_dim, input_dim, n_channel])
Y_ph = tf.placeholder(tf.float32, [None,output_head_dimensions[0]])
p_conv_ph = tf.placeholder(tf.float32)
p_latent_ph = tf.placeholder(tf.float32)
task_id_ph = tf.placeholder(tf.int64)



###########
#VARIABLES
###########
K = 32  # first convolutional layer output depth
L = 32  # second convolutional layer output depth
M = 64  # third convolutional layer
N = 64  #fourth convolution 
flattened = int(N*(input_dim/4)**2)
final = 512
W1 = tf.Variable(tf.random.uniform([3,3,n_channel,K], minval=-tf.sqrt(6/(3*3*(n_channel+K))), \
                                   maxval=tf.sqrt(6/(3*3*(n_channel+K))) ))
b1 = tf.Variable(tf.zeros([K]))
W2 = tf.Variable(tf.random.uniform([3,3,K,L], minval=-tf.sqrt(6/(3*3*(K+L))), \
                                   maxval=tf.sqrt(6/(3*3*(K+L))) ))
b2 = tf.Variable(tf.zeros([L]))
W3 = tf.Variable(tf.random.uniform([3,3,L,M], minval=-tf.sqrt(6/(3*3*(L+M))), \
                                   maxval=tf.sqrt(6/(3*3*(L+M))) ))
b3 = tf.Variable(tf.zeros([M]))
W4 = tf.Variable(tf.random.uniform([3,3,M,N], minval=-tf.sqrt(6/(3*3*(M+N))), \
                                   maxval=tf.sqrt(6/(3*3*(M+N))) ))
b4 = tf.Variable(tf.zeros([N]))
W5 =  tf.Variable(tf.random.uniform([flattened,final], minval=-tf.sqrt(6/(flattened+final)),maxval=tf.sqrt(6/(flattened+final)))) 
b5 = tf.Variable(tf.zeros([final]))

#define multiple outputheads matrices and biases
#algo only works for equal outputhead dimensions, see tf.stack'
out_W = []
out_b = []
for dim in output_head_dimensions:
    out_W.append( tf.Variable(tf.random.uniform([final,dim], minval=-tf.sqrt(6/(final+dim)),maxval=tf.sqrt(6/(final+dim)))) )
    out_b.append(tf.Variable(tf.zeros(dim)))


#################    
#CNN ARCHITECTURE
#################
Y1_cnv = tf.nn.conv2d(X_ph,W1,strides=[1,1,1,1], padding='SAME')
Y1 = tf.nn.relu(Y1_cnv + b1)

Y2_cnv = tf.nn.conv2d(Y1,W2,strides=[1,1,1,1], padding='SAME')
Y2 = tf.nn.relu(Y2_cnv + b2)
Y2_max = tf.nn.dropout(tf.nn.max_pool(Y2, [2,2], [2,2], padding='SAME'), rate=p_conv_ph)

Y3_cnv = tf.nn.conv2d(Y2_max,W3,strides=[1,1,1,1], padding='SAME')
Y3 = tf.nn.relu(Y3_cnv + b3)

Y4_cnv = tf.nn.conv2d(Y3,W4,strides=[1,1,1,1], padding='SAME')
Y4 = tf.nn.relu(Y4_cnv + b4)
Y4_max = tf.nn.dropout(tf.nn.max_pool(Y4, [2,2], [2,2], padding='SAME'), rate=p_conv_ph)

Y_flattened = tf.reshape(Y4_max, [-1,flattened] )
Y_final = tf.nn.dropout(tf.nn.relu(tf.matmul(Y_flattened, W5) + b5), rate=p_latent_ph)

##########
#OUTPUT HEADS
##########
Y_logits = []
Y_pred = []

for task_id in range(len(dataset_train)):
    cc = tf.matmul(Y_final,out_W[task_id]) + out_b[task_id]
    dd = tf.nn.softmax(cc)
    Y_logits.append(cc)
    Y_pred.append(dd)

Y_logits = tf.stack(Y_logits)
Y_pred = tf.stack(Y_pred)






##################
#WEIGHT PROTECTION
##################
variables = [b1, W1, b2, W2, b3, W3, b4, W4, W5, b5] #these are the variables whose importance will be measured
variables_all = [*variables, *out_W, *out_b]

#weight protection
old_variables = []
prev_variables = []
importances = []
contributions_SI = []
contributions_SIU = []
contributions_SIB = []
contributions_SOS = []
lengths = []
length_variables = []
length_eq_new_op = []

old_eq_new_op = []
prev_eq_curr_op = []


for i in range(len(variables)):
    #store old variables
    old_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_eq_new_op.append(old_variables[i].assign(variables[i]))
    #store prev variables
    prev_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    prev_eq_curr_op.append(prev_variables[i].assign(variables[i]))
    #store importances
    importances.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    #store changes
    contributions_SI.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_SIU.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_SIB.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_SOS.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    lengths.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    # store variables needed to calculated update lengths
    length_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    length_eq_new_op.append(length_variables[i].assign(variables[i]))  




########
#LOSS
########
#Cross-entropy
CEL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_ph, logits=Y_logits[task_id_ph]))

#weight protection
consolidation_loss = 0.0
for i in range(len(variables)):
  consolidation_loss += tf.reduce_sum( tf.multiply(importances[i],tf.square(variables[i]-old_variables[i]) ) )
total_loss = CEL + HP['c']*consolidation_loss

#accuracy
correct_a = tf.equal(tf.argmax(Y_ph,axis=1), tf.argmax(Y_pred[task_id_ph],axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_a,tf.float32))

#training and gradients
trainer = tf.train.AdamOptimizer()
train = trainer.minimize(total_loss)

trainer2 = tf.train.GradientDescentOptimizer(1.0) #lr doesnt matter
gradient = trainer2.compute_gradients(CEL, variables)



#define ops to update importances
contributions_to_zero_op = []
lengths_op = []
importances_op = []
old_gradients_var = []
old_gradients_op = []
old_gradients_unbiased_var = []
old_gradients_unbiased_op = []
add_contributions_SI_op = []
add_contributions_SIU_op = []
add_contributions_SIB_op = []
add_contributions_SOS_op = []

for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_SI[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIB[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SOS[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    
    old_gradients_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_op.append(old_gradients_var[i].assign(gradient[i][0]))
    old_gradients_unbiased_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_unbiased_op.append(old_gradients_unbiased_var[i].assign(gradient[i][0]))

    
    add_contributions_SI_op.append( contributions_SI[i].assign_add(   tf.multiply(-old_gradients_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_op.append( contributions_SIU[i].assign_add(   tf.multiply(-old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIB_op.append( contributions_SIB[i].assign_add(   tf.multiply(-old_gradients_var[i]+old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SOS_op.append( contributions_SOS[i].assign(   0.999*contributions_SOS[i] + 0.001*tf.square(old_gradients_var[i]-HP['SOS_alpha']*old_gradients_unbiased_var[i])  ))
    #old_gradients_var without drop out

    lengths_op.append( lengths[i].assign(variables[i] - length_variables[i]) )
    if HP['method'] == 'SI':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SI[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    if HP['method'] == 'SIU':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SIU[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    if HP['method'] == 'SIB':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SIB[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    if HP['method'] == 'SOS':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(tf.sqrt(contributions_SOS[i]),tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) ) #max not needed since contributtions_SOS non-negative

        


#Get Adam Variables
#get momentum vars and such
optimizer_vars = [trainer.get_slot(var, name) for name in trainer.get_slot_names() for var in variables]
#append beta vars
for new_var in trainer._get_beta_accumulators(): 
  optimizer_vars.append(new_var)
#filter for None-vars
optimizer_vars_2 = []
for var in optimizer_vars:
   if var != None:
      optimizer_vars_2.append(var)
#INITIALISE OPT
init_opt_vars_op = tf.variables_initializer(optimizer_vars_2)

#INITIALISE Model Vraibales
init_model_vars_op = tf.variables_initializer(variables)



#initialize the model
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True



#############
#THE SESS
#############

with tf.Session(config=config) as sess:
    sess.run(init)
    for task_id, data in enumerate(dataset_train):
        print("\nSTARTING TASK ",task_id)
        sess.run(init_opt_vars_op)
        if HP['re_init_model']:  
            sess.run(init_model_vars_op) 
        sess.run(length_eq_new_op)
        sess.run(contributions_to_zero_op)
        
        n_iterations = int(data[0].shape[0]*train_share /HP['batch_size']) 
        
        n_ep = HP['n_epochs_per_task']
        if HP['equal_iter'] and task_id>0:
            n_ep *= 10
        for epoch in range(n_ep):
            print("Epoch ", epoch, '    Task ', task_id)
            # SHUFFLE TRAINING DATA
            perm = np.random.permutation(int(train_share*data[0].shape[0]))
            my_data = [[],[]]
            my_data[0] = data[0][perm,:,:,:]
            my_data[1] = data[1][perm,:]
            end = time.time()
            print('TIME: ', end-start, '\n')
            start =time.time()
            
            for j in range(n_iterations):
                bs = HP['batch_size']
                ## IF NEEDED, CALCULATE GRADIENTS ON INDEPENDENT BATCH
                if (HP['method'] == 'SOS' and HP['SOS_alpha']!=0) or HP['method'] == 'SIU' or HP['method'] == 'SIB':
                    X_batch_grad, Y_batch_grad = utils.mini_batch(data, bs, train_share=train_share)
                    sess.run(old_gradients_unbiased_op,feed_dict={X_ph:X_batch_grad, Y_ph:Y_batch_grad, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})      
                ## NOW SAMPLE BATCH FOR OPTIMIZER AND TRAIN
                X_batch = my_data[0][j*bs:(j+1)*bs,:,:,:]
                Y_batch = my_data[1][j*bs:(j+1)*bs,:]
                if not HP['method'] == 'SIU':
                    sess.run(old_gradients_op,feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})
                sess.run(prev_eq_curr_op)    
                sess.run(train,feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:p_c, p_latent_ph:p_l})
                
                #UPDATE RUNNING SUM FOR ALGO
                if HP['method'] == 'SI':
                    sess.run(add_contributions_SI_op)  
                if HP['method'] == 'SIU':
                    sess.run(add_contributions_SIU_op)  
                if HP['method'] == 'SIB':
                    sess.run(add_contributions_SIB_op)  
                if HP['method'] == 'SOS':
                    sess.run(add_contributions_SOS_op) 
            #END OF EPOCH
        #END OF TASK
        #DO SI STUFF
        sess.run(lengths_op)
        sess.run(importances_op)
        sess.run(old_eq_new_op)
        #CALCULATE PREVIOUS TASK PERFORMANCE
        val_acc_s = np.zeros(task_id+1)
        for old_task in range(task_id+1):
            if HP['evaluate_on_validation']:
                X_batch, Y_batch = utils.mini_batch(dataset_train[old_task], 2000, train=False, train_share=train_share) #note that batch size is ignored for train=False
                val_acc, val_loss = sess.run([accuracy, CEL], feed_dict={X_ph:X_batch, Y_ph:Y_batch, p_conv_ph:0.0, p_latent_ph:0.0, task_id_ph:old_task})
            else:
                split = int(dataset_test[old_task][0].shape[0]/2)
                X_batch = dataset_test[old_task][0][:split,:,:,:]
                Y_batch = dataset_test[old_task][1][:split,:]
                val_acc_1, val_loss_1 = sess.run([accuracy, CEL], feed_dict={X_ph:X_batch, Y_ph:Y_batch, p_conv_ph:0.0, p_latent_ph:0.0, task_id_ph:old_task})
                X_batch = dataset_test[old_task][0][split:,:,:,:]
                Y_batch = dataset_test[old_task][1][split:,:]
                val_acc_2, val_loss_2 = sess.run([accuracy, CEL], feed_dict={X_ph:X_batch, Y_ph:Y_batch, p_conv_ph:0.0, p_latent_ph:0.0, task_id_ph:old_task})
                val_acc = (val_acc_1 + val_acc_2)/2
                val_loss = (val_loss_1 + val_loss_2)/2
            val_acc_s[old_task]= val_acc
            print("val acc: "+str(val_acc)+" ********* "+"loss "+str(val_loss))

        print("avg acc on tasks seen so far:", np.mean(val_acc_s))
 
if save_outputs_to_log_dir:
    sys.stdout = orig_stdout
    f.close()


file = open('summary_SI.txt', 'a+')
file.write(str(np.mean(val_acc_s))+' '+HP_label+'\n')
file.close()

    



