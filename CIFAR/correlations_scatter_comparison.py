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
from scipy.stats import pearsonr


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','SI', '0.1', '1.0', '1']
inputs = sys.argv
outer = inputs[1]
visible_GPU = '1'
save_outputs_to_log_dir = False
meta = 'run_new'

### HYPERPARAMETERS
# for method choose SI, SIU, SIB, OnAF
HP = {\
'seed'                  : int(inputs[1]),\
'method'                : 'SI',\
'c'                     : 5.0,\
're_init_model'         : True,\
'rescale'               : 1.0,\
'n_samples'             : 500,\
'evaluate_on_validation': True,\
'batch_size'            : 256,\
'n_tasks'               : 6,\
'n_epochs_per_task'     : 60,\
}
evaluate_fisher = True

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



    
HP_label = 'scatter'
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
my_factor_ph = tf.placeholder(tf.float32)


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

l2_pred = tf.reduce_sum(tf.square(Y_pred[task_id_ph]))
l2_pred_max = tf.square(tf.reduce_max(Y_pred[task_id_ph], axis=1))

l2_logits = tf.reduce_sum(tf.square(Y_logits[task_id_ph]))
l2_logits_max = tf.square(tf.reduce_max(Y_logits[task_id_ph], axis=1))


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
contributions_expSM = []
contributions_expH = []


contributions_MAS = []
contributions_MASX = []

contributions_MAS_2 = []
contributions_MASX_2 = []


contributions_EWC = []
contributions_AF = []
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

    contributions_expSM.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_expH.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))

    contributions_MAS.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MASX.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    
    contributions_MAS_2.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MASX_2.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    
    
    contributions_EWC.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_AF.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
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
gradient_MAS = trainer2.compute_gradients(l2_pred, variables)
gradient_MASX = trainer2.compute_gradients(l2_pred_max, variables)

gradient_MAS_2 = trainer2.compute_gradients(l2_logits, variables)
gradient_MASX_2 = trainer2.compute_gradients(l2_logits_max, variables)



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

add_contributions_expSM_op = []
add_contributions_expH_op = []

add_contributions_MAS_op = []
add_contributions_MASX_op = []

add_contributions_MAS_2_op = []
add_contributions_MASX_2_op = []


add_contributions_EWC_op = []
add_contributions_AF_op = []

for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_SI[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIB[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_expSM[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_expH[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MAS[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MAS_2[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX_2[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_EWC[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_AF[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    
    old_gradients_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_op.append(old_gradients_var[i].assign(gradient[i][0]))
    old_gradients_unbiased_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_unbiased_op.append(old_gradients_unbiased_var[i].assign(gradient[i][0]))

    
    add_contributions_SI_op.append( contributions_SI[i].assign_add(   tf.multiply(-old_gradients_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_op.append( contributions_SIU[i].assign_add(   tf.multiply(-old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIB_op.append( contributions_SIB[i].assign_add(   tf.multiply(-old_gradients_var[i]+old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))

    add_contributions_expSM_op.append( contributions_expSM[i].assign(   0.999*contributions_expSM[i] + 0.001*tf.square(old_gradients_var[i])  ))
    add_contributions_expH_op.append( contributions_expH[i].assign(   0.999*contributions_expH[i] + 0.001*tf.square(old_gradients_var[i]-old_gradients_unbiased_var[i])  ))
    add_contributions_MAS_op.append( contributions_MAS[i].assign_add(   tf.abs(gradient_MAS[i][0])  ))
    add_contributions_MASX_op.append( contributions_MASX[i].assign_add(   tf.abs(gradient_MASX[i][0])  ))
    
    add_contributions_MAS_2_op.append( contributions_MAS_2[i].assign_add(   tf.abs(gradient_MAS_2[i][0])  ))
    add_contributions_MASX_2_op.append( contributions_MASX_2[i].assign_add(   tf.abs(gradient_MASX_2[i][0])  ))
    
    add_contributions_EWC_op.append( contributions_EWC[i].assign_add(   my_factor_ph * tf.square(gradient[i][0])  ))
    add_contributions_AF_op.append( contributions_AF[i].assign_add(   my_factor_ph * tf.abs(gradient[i][0])  ))

    lengths_op.append( lengths[i].assign(variables[i] - length_variables[i]) )
    if HP['method'] == 'SI':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SI[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    

##Prepare lists for storing different importance measures
cont_SI_all = []
cont_SIU_all = []
cont_SIB_all = []

cont_expSM_all = []
cont_expH_all = []

cont_EWC_all = []
cont_AF_all = []
cont_MAS_all = []
cont_MASX_all = []

cont_MAS_2_all = []
cont_MASX_2_all = []


cont_len_all = []


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
        for epoch in range(HP['n_epochs_per_task']):
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
                if True:
                    X_batch_grad, Y_batch_grad = utils.mini_batch(data, bs, train_share=train_share)
                    sess.run(old_gradients_unbiased_op,feed_dict={X_ph:X_batch_grad, Y_ph:Y_batch_grad, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})      
                ## NOW SAMPLE BATCH FOR OPTIMIZER AND TRAIN
                X_batch = my_data[0][j*bs:(j+1)*bs,:,:,:]
                Y_batch = my_data[1][j*bs:(j+1)*bs,:]
                sess.run(old_gradients_op,feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})
                sess.run(prev_eq_curr_op)    
                sess.run(train,feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:p_c, p_latent_ph:p_l})
                
                #UPDATE RUNNING SUM FOR ALGO
                sess.run(add_contributions_SI_op)  
                sess.run(add_contributions_SIU_op)  
                sess.run(add_contributions_SIB_op)  
                sess.run(add_contributions_expSM_op)
                sess.run(add_contributions_expH_op)

            #END OF EPOCH
        #END OF TASK
        #DO SI STUFF
        sess.run(lengths_op)
        cont_len_all.append(utils.flatten_list(sess.run(lengths)))
        sess.run(importances_op)
        sess.run(old_eq_new_op)
        
        #Calcualte Fisher or so
        if evaluate_fisher:
            permut = np.random.permutation(int(train_share*data[0].shape[0]))
            for sam in range(HP['n_samples']):
                X_batch = data[0][permut[sam:sam+1],:,:,:]
                Y_batch = data[1][permut[sam:sam+1],:]
                sess.run(add_contributions_MAS_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})
                sess.run(add_contributions_MASX_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})  
                
                sess.run(add_contributions_MAS_2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})
                sess.run(add_contributions_MASX_2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0})   

                
                predictions = sess.run(Y_pred[task_id],feed_dict={X_ph:X_batch, Y_ph:Y_batch, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0}) 
                for ii in range(10):  
                    Y_fake = np.zeros([1,10])
                    Y_fake[0,ii] = 1
                    my_factor = predictions[0,ii]
                    sess.run(add_contributions_EWC_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0, my_factor_ph:my_factor})
                    sess.run(add_contributions_AF_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, task_id_ph:task_id, p_conv_ph:0.0, p_latent_ph:0.0, my_factor_ph:my_factor})    
        #SOTRE ALL CONTRIBUTIONS
        cont_SI_all.append(   utils.flatten_list(sess.run(contributions_SI))         )
        cont_SIU_all.append(   utils.flatten_list(sess.run(contributions_SIU))         )
        cont_SIB_all.append(   utils.flatten_list(sess.run(contributions_SIB))         )
        #cont_lens_all was handled above

        cont_expSM_all.append(   utils.flatten_list(sess.run(contributions_expSM))         )
        cont_expH_all.append(   utils.flatten_list(sess.run(contributions_expH))         )

        cont_EWC_all.append(   utils.flatten_list(sess.run(contributions_EWC))         )
        cont_AF_all.append(   utils.flatten_list(sess.run(contributions_AF))         )
        cont_MAS_all.append(   utils.flatten_list(sess.run(contributions_MAS))         )
        cont_MASX_all.append(   utils.flatten_list(sess.run(contributions_MASX))         )
        
        cont_MAS_2_all.append(   utils.flatten_list(sess.run(contributions_MAS_2))         )
        cont_MASX_2_all.append(   utils.flatten_list(sess.run(contributions_MASX_2))         )
        
        
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



        
        
my_dict_SI = {\
    'SI':cont_SI_all,\
    'SIU':cont_SIU_all,\
    'SIB':cont_SIB_all,\
    'expSM':cont_expSM_all,\
    'expH':cont_expH_all,\
    'lengths':cont_len_all,\
        }
for key, val in my_dict_SI.items():
    my_dict_SI[key] = np.asarray(val)
if evaluate_fisher:
    my_dict_MAS = {\
        'EWC':cont_EWC_all,\
        'AF':cont_AF_all,\
        'MAS':cont_MAS_all,\
        'MASX':cont_MASX_all,\
        'MAS2':cont_MAS_2_all,\
        'MASX2':cont_MASX_2_all,\
            }
    for key, val in my_dict_MAS.items():
        my_dict_MAS[key] = np.asarray(val)

save_importances = (HP['seed'] == 100)
if save_importances:
    with open('scatter_repetitions/all_importances_SI_'+meta+str(outer)+'.pickle', 'wb') as file:
        pickle.dump(my_dict_SI, file)   
    if evaluate_fisher:
        with open('scatter_repetitions/all_importances_MAS_'+meta+str(outer)+'.pickle', 'wb') as file:
            pickle.dump(my_dict_MAS, file)           
if save_outputs_to_log_dir:
    sys.stdout = orig_stdout
    f.close()

#save all correlations
eps = 1e-8 #for numerical stability

#SI
a = {}
for key, val in my_dict_SI.items():
    a[key] = np.asarray(val) 
a['RSM'] = np.sqrt(a['expSM']) #RSM root-square-mean
a['RH'] = np.sqrt(a['expH']) #RSM root-square-mean lol
a['SI-N'] = a['SI']/(a['lengths']**2+HP['damp'])
a['SIB-N'] = a['SIB']/(a['lengths']**2 + HP['damp'])
a['SIU-N'] = a['SIU']/(a['lengths']**2 + HP['damp'])

a['SI-P'] = np.maximum(0,a['SI'])
a['SIB-P'] = np.maximum(0,a['SIB'])
a['SIU-P'] = np.maximum(0,a['SIU'])


#initialise
all_corrs = {}
for k1 in a.keys():
    all_corrs[k1] = {}
    for k2 in a.keys():
        all_corrs[k1][k2] = []
#compute
for task in range(HP['n_tasks']):
    for k1 in a.keys():
        for k2 in a.keys():
            z = pearsonr(a[k1][task,:], a[k2][task,:])[0]
            all_corrs[k1][k2].append(z)
#save
with open('scatter_repetitions/all_correlations_SI_'+meta+str(outer)+'.pickle', 'wb') as file:
    pickle.dump(all_corrs, file)

    
if evaluate_fisher:    
    #MAS
    a = {}
    for key, val in my_dict_MAS.items():
        a[key] = np.asarray(val) 
    a['rEWC'] = np.sqrt(a['EWC'])
    #initialise
    all_corrs = {}
    for k1 in a.keys():
        all_corrs[k1] = {}
        for k2 in a.keys():
            all_corrs[k1][k2] = []
    #compute
    for task in range(HP['n_tasks']):
        for k1 in a.keys():
            for k2 in a.keys():
                #print(k1, k2, a[k1].shape, a[k2].shape)
                z = pearsonr(a[k1][task,:], a[k2][task,:])[0]
                all_corrs[k1][k2].append(z)
    #save
    with open('scatter_repetitions/all_correlations_MAS_'+meta+str(outer)+'.pickle', 'wb') as file:
        pickle.dump(all_corrs, file)

file = open('scatter_repetitions/summary.txt', 'a+')
file.write(str(np.mean(val_acc_s))+' '+HP_label+'\n')
file.close()




