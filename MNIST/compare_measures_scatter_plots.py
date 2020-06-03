import os, sys, time
start = time.time()
end = time.time()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import pickle
import utils


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','SI', '0.1', '1.0', '1']
inputs = sys.argv
visible_GPU = '0'
save_outputs_to_log_dir = False
meta = 'run_new'
### HYPERPARAMETERS
# for method choose SI, SIU, SIB, OnAF
HP = {\
'seed'              : 1,\
'method'            : 'SI',\
'c'                 : 0.2,\
're_init_model'     : True,\
'rescale'           : 1.0,\
'n_samples'         : 1000,\
'batch_size'        : 256,\
'n_tasks'           : 10,\
'n_epochs_per_task' : 20,\
'first_hidden'      : 2000,\
'second_hidden'     : 2000,\
}

if HP['rescale']== 1.0: #set damping for division as in SI paper. Changes here will also need changes in HP['c'].
    HP['damp'] = 0.1 
if HP['rescale']== 0.0: #turn of dividing by length.
    HP['damp'] = 1.0 
    
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

# CREATE RANDOM PERMUTATIONS FOR P-MNIST TASKS
permutations = []
for i in range(HP['n_tasks']):
    permutations.append(np.random.permutation(784))

#placeholders
X_ph = tf.placeholder(tf.float32,[None, 784])
Y_ph = tf.placeholder(tf.float32, [None,10])
my_factor_ph = tf.placeholder(tf.float32)

#Variables
K = HP['first_hidden']
L = HP['second_hidden']
W0 = tf.Variable(tf.random.uniform([784,K], minval=-tf.sqrt(6/(784+K)),maxval=tf.sqrt(6/(784+K))))
b0 = tf.Variable(tf.ones([K])/10)
W1 = tf.Variable(tf.random.uniform([K,L], minval=-tf.sqrt(6/(L+K)),maxval=tf.sqrt(6/(L+K))))
b1 = tf.Variable(tf.ones([L])/10)
W2 = tf.Variable(tf.random.uniform([L,10], minval=-tf.sqrt(6/(L+10)),maxval=tf.sqrt(6/(10+L))))
b2 = tf.Variable(tf.ones(10)/10)
variables = [b0, W0, b1, W1, b2, W2]

#model with 2 hidden layers, 784-K-L-10
H1 = tf.nn.relu(tf.matmul(X_ph,W0) + b0)
H2 = tf.nn.relu(tf.matmul(H1, W1) + b1)
Y_logits = tf.matmul(H2,W2) + b2
Y_pred = tf.nn.softmax(Y_logits)
l2_pred = tf.reduce_sum(tf.square(Y_pred))
l2_pred_max = tf.square(tf.reduce_max(Y_pred, axis=1))


#weight protection
old_variables = []
prev_variables = []
importances = []
contributions_SI = []
contributions_SIU = []
contributions_SIB = []
contributions_OnAF = []
contributions_MAS = []
contributions_MASX = []
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
    contributions_OnAF.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MAS.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MASX.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_EWC.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_AF.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    lengths.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    # store variables needed to calculated update lengths
    length_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    length_eq_new_op.append(length_variables[i].assign(variables[i]))  


#loss
#Cross-entropy
CEL = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(labels=Y_ph, logits=Y_logits))

#weight protection loss
consolidation_loss = 0.0
for i in range(len(variables)):
    consolidation_loss += tf.reduce_sum( tf.multiply(importances[i],tf.square(variables[i]-old_variables[i]) ) )
total_loss = CEL + HP['c']*consolidation_loss

#accuracy
correct_a = tf.equal(tf.argmax(Y_ph,axis=1), tf.argmax(Y_pred,axis=1))
accuracy = tf.reduce_mean(tf.cast(correct_a,tf.float32))

#training and gradients
trainer = tf.train.AdamOptimizer()
train = trainer.minimize(total_loss)

trainer2 = tf.train.GradientDescentOptimizer(1.0) #lr doesnt matter
gradient = trainer2.compute_gradients(CEL, variables)
gradient_MAS = trainer2.compute_gradients(l2_pred, variables)
gradient_MASX = trainer2.compute_gradients(l2_pred_max, variables)


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
add_contributions_OnAF_op = []
add_contributions_MAS_op = []
add_contributions_MASX_op = []
add_contributions_EWC_op = []
add_contributions_AF_op = []

for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_SI[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIB[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_OnAF[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MAS[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_EWC[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_AF[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    
    old_gradients_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_op.append(old_gradients_var[i].assign(gradient[i][0]))
    old_gradients_unbiased_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_unbiased_op.append(old_gradients_unbiased_var[i].assign(gradient[i][0]))

    
    add_contributions_SI_op.append( contributions_SI[i].assign_add(   tf.multiply(-old_gradients_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_op.append( contributions_SIU[i].assign_add(   tf.multiply(-old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIB_op.append( contributions_SIB[i].assign_add(   tf.multiply(-old_gradients_var[i]+old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_OnAF_op.append( contributions_OnAF[i].assign_add(   tf.abs(old_gradients_var[i])  ))
    add_contributions_MAS_op.append( contributions_MAS[i].assign_add(   tf.abs(gradient_MAS[i][0])  ))
    add_contributions_MASX_op.append( contributions_MASX[i].assign_add(   tf.abs(gradient_MASX[i][0])  ))
    add_contributions_EWC_op.append( contributions_EWC[i].assign_add(   my_factor_ph * tf.square(gradient[i][0])  ))
    add_contributions_AF_op.append( contributions_AF[i].assign_add(   my_factor_ph * tf.abs(gradient[i][0])  ))

    lengths_op.append( lengths[i].assign(variables[i] - length_variables[i]) )
    if HP['method'] == 'SI':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SI[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )

        
        
##Prepare lists for storing different importance measures
cont_SI_all = []
cont_SIU_all = []
cont_SIB_all = []
cont_OnAF_all = []
cont_EWC_all = []
cont_AF_all = []
cont_MAS_all = []
cont_MASX_all = []




        
        
        
        
        
        
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


## DO IT
#initialize the model
init = tf.global_variables_initializer()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
n_iterations = int(60000/HP['batch_size'])



with tf.Session(config=config) as sess:
    sess.run(init)

    for task in range(HP['n_tasks']):
        print("\nStarting task ",task)
        sess.run(contributions_to_zero_op)
        sess.run(init_opt_vars_op)
        if HP['re_init_model']:
            sess.run(init_model_vars_op)
        sess.run(length_eq_new_op)
        
        for epoch in range(HP['n_epochs_per_task']):
            print("Epoch", epoch, '; Task', task)
            end = time.time()
            print('Time ', end-start)
            start = time.time()
            for j in range(n_iterations):
                ## CALCULATE GRADIENTS ON INDEPENDENT BATCH
                X_batch, Y_batch = mnist.train.next_batch(HP['batch_size'])
                X_batch = X_batch[:,permutations[task]]
                sess.run(old_gradients_unbiased_op,feed_dict={X_ph:X_batch, Y_ph:Y_batch})
                
                ## NOW SAMPLE BATCH FOR OPTIMIZER AND TRAIN
                X_batch, Y_batch = mnist.train.next_batch(HP['batch_size'])
                X_batch = X_batch[:,permutations[task]]
                sess.run(old_gradients_op,feed_dict={X_ph:X_batch, Y_ph:Y_batch})
                sess.run(prev_eq_curr_op)    
                sess.run(train,feed_dict={X_ph:X_batch, Y_ph:Y_batch})
                
                #UPDATE RUNNING SUM FOR ALGO
                sess.run(add_contributions_SI_op)  
                sess.run(add_contributions_SIU_op)  
                sess.run(add_contributions_SIB_op)  
                sess.run(add_contributions_OnAF_op)  
            #END OF EPOCH
        #END OF TASK
        # Do what needs to be done for SI
        sess.run(lengths_op)
        sess.run(importances_op)
        sess.run(old_eq_new_op)
        
        #Calculate FISHER and MAS
        for i in range(HP['n_samples']):
              X_batch, Y_batch = mnist.train.next_batch(1)
              X_batch = X_batch[:,permutations[task]]  
              sess.run(add_contributions_MAS_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})
              sess.run(add_contributions_MASX_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})   
              predictions = sess.run(Y_pred,feed_dict={X_ph:X_batch, Y_ph:Y_batch}) 
              for ii in range(10):  
                  Y_fake = np.zeros([1,10])
                  Y_fake[0,ii] = 1
                  my_factor = predictions[0,ii]
                  sess.run(add_contributions_EWC_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, my_factor_ph:my_factor})
                  sess.run(add_contributions_AF_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, my_factor_ph:my_factor})    
        #SOTRE ALL CONTRIBUTIONS
        cont_SI_all.append(   utils.flatten_list(sess.run(contributions_SI))         )
        cont_SIU_all.append(   utils.flatten_list(sess.run(contributions_SIU))         )
        cont_SIB_all.append(   utils.flatten_list(sess.run(contributions_SIB))         )
        cont_OnAF_all.append(   utils.flatten_list(sess.run(contributions_OnAF))         )
        cont_EWC_all.append(   utils.flatten_list(sess.run(contributions_EWC))         )
        cont_AF_all.append(   utils.flatten_list(sess.run(contributions_AF))         )
        cont_MAS_all.append(   utils.flatten_list(sess.run(contributions_MAS))         )
        cont_MASX_all.append(   utils.flatten_list(sess.run(contributions_MASX))         )
        
        
        
        
        #Evaluate performance on previous tasks
        print("\n")
        testing_acc_s = np.zeros(task+1)
        for old_task in range(HP['n_tasks']):
            test_acc, test_loss = sess.run([accuracy, CEL], feed_dict={X_ph:mnist.test.images[:,permutations[old_task]], 
                                                                            Y_ph:mnist.test.labels})
            if old_task < task+1:
                testing_acc_s[old_task]= test_acc
            print("Task:",old_task, "Test acc:", test_acc, "loss: "+str(test_loss))
        print("avgerage acc on tasks seen so far:", np.mean(testing_acc_s))
        print(" ")

my_dict = {\
    'SI':cont_SI_all,\
    'SIU':cont_SIU_all,\
    'SIB':cont_SIB_all,\
    'OnAF':cont_OnAF_all,\
    'EWC':cont_EWC_all,\
    'AF':cont_AF_all,\
    'MAS':cont_MAS_all,\
    'MASX':cont_MASX_all,\
          }

with open('all_measures_'+meta+'.pickle', 'wb') as file:
    pickle.dump(my_dict, file)            

if save_outputs_to_log_dir:
    sys.stdout = orig_stdout
    f.close()

file = open('summary_2.txt', 'a+')
file.write(str(np.mean(testing_acc_s))+' '+HP_label+'\n')
file.close()

