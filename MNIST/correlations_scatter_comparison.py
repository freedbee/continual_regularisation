import os, sys, time
start = time.time()
end = time.time()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import pickle
import utils
from scipy.stats import pearsonr


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','SI', '0.1', '1.0', '1']
inputs = sys.argv
outer = inputs[1]
visible_GPU = '0'
save_outputs_to_log_dir = False
meta = 'run_new'
### HYPERPARAMETERS
# for method choose SI, SIU, SIB, OnAF
HP = {\
    'seed'              : int(inputs[1]),\
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

dtype = 'float32'
    
# VISIBLE GPU
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]=visible_GPU
if save_outputs_to_log_dir:
    orig_stdout = sys.stdout
    f = open('scatter_repetitions/logs/log_'+HP_label+'.txt', 'w')
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
X_ph = tf.placeholder(dtype,[None, 784])
Y_ph = tf.placeholder(dtype, [None,10])
my_factor_ph = tf.placeholder(dtype)

#Variables
K = HP['first_hidden']
L = HP['second_hidden']
W0 = tf.Variable(tf.random.uniform([784,K], minval=tf.cast(-tf.sqrt(6/(784+K)),dtype=dtype),maxval=tf.cast(tf.sqrt(6/(784+K)),dtype=dtype),dtype=dtype),dtype=dtype)
b0 = tf.Variable(tf.cast(tf.ones([K])/10, dtype=dtype),dtype=dtype)
W1 = tf.Variable(tf.random.uniform([K,L], minval=tf.cast(-tf.sqrt(6/(L+K)),dtype=dtype),maxval=tf.cast(tf.sqrt(6/(L+K)),dtype=dtype),dtype=dtype),dtype=dtype)
b1 = tf.Variable(tf.cast(tf.ones([L])/10, dtype=dtype),dtype=dtype)
W2 = tf.Variable(tf.random.uniform([L,10], minval=tf.cast(-tf.sqrt(6/(L+10)),dtype=dtype),maxval=tf.cast(tf.sqrt(6/(L+10)),dtype=dtype),dtype=dtype),dtype=dtype)
b2 = tf.Variable(tf.cast(tf.ones([10])/10, dtype=dtype),dtype=dtype)
variables = [b0, W0, b1, W1, b2, W2]

#model with 2 hidden layers, 784-K-L-10
H1 = tf.nn.relu(tf.matmul(X_ph,W0) + b0)
H2 = tf.nn.relu(tf.matmul(H1, W1) + b1)
Y_logits = tf.matmul(H2,W2) + b2
Y_pred = tf.nn.softmax(Y_logits)
l2_pred = tf.reduce_sum(tf.square(Y_pred))
l2_pred_max = tf.square(tf.reduce_max(Y_pred, axis=1))

l2_logits = tf.reduce_sum(tf.square(Y_logits))
l2_logits_max = tf.square(tf.reduce_max(Y_logits, axis=1))


#weight protection
old_variables = []
prev_variables = []
importances = []
contributions_SI = []
contributions_SIU = []
contributions_SIB = []

contributions_expSM = [] #exponential average second Moment

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
    old_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype) )
    old_eq_new_op.append(old_variables[i].assign(variables[i]))
    #store prev variables
    prev_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype) )
    prev_eq_curr_op.append(prev_variables[i].assign(variables[i]))
    #store importances
    importances.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    #store changes
    contributions_SI.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    contributions_SIU.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    contributions_SIB.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))

    contributions_expSM.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))

    contributions_MAS.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    contributions_MASX.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))

    contributions_MAS_2.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    contributions_MASX_2.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))

    contributions_EWC.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    contributions_AF.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    lengths.append(tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype))
    # store variables needed to calculated update lengths
    length_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype) )
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

add_contributions_MAS_op = []
add_contributions_MASX_op = []

add_contributions_MAS_2_op = []
add_contributions_MASX_2_op = []

add_contributions_EWC_op = []
add_contributions_AF_op = []

for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_SI[i], tf.zeros(tf.shape(contributions_SI[i]), dtype=dtype)) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU[i], tf.zeros(tf.shape(contributions_SI[i]), dtype=dtype)) )
    contributions_to_zero_op.append( tf.assign(contributions_SIB[i], tf.zeros(tf.shape(contributions_SI[i]), dtype=dtype)) )

    contributions_to_zero_op.append( tf.assign(contributions_expSM[i], tf.zeros(tf.shape(contributions_SI[i]), dtype=dtype)) )

    contributions_to_zero_op.append( tf.assign(contributions_MAS[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )

    contributions_to_zero_op.append( tf.assign(contributions_MAS_2[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX_2[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )

    contributions_to_zero_op.append( tf.assign(contributions_EWC[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )
    contributions_to_zero_op.append( tf.assign(contributions_AF[i], tf.zeros(tf.shape(contributions_AF[i]), dtype=dtype)) )

    old_gradients_var.append( tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype) )
    old_gradients_op.append(old_gradients_var[i].assign(gradient[i][0]))
    old_gradients_unbiased_var.append( tf.Variable(tf.zeros(tf.shape(variables[i]), dtype=dtype), trainable=False, dtype=dtype) )
    old_gradients_unbiased_op.append(old_gradients_unbiased_var[i].assign(gradient[i][0]))


    add_contributions_SI_op.append( contributions_SI[i].assign_add(   tf.multiply(-old_gradients_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_op.append( contributions_SIU[i].assign_add(   tf.multiply(-old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIB_op.append( contributions_SIB[i].assign_add(   tf.multiply(-old_gradients_var[i]+old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))

    add_contributions_expSM_op.append( contributions_expSM[i].assign(   0.999*contributions_expSM[i] + 0.001*tf.square(old_gradients_var[i])  ))

    add_contributions_MAS_op.append( contributions_MAS[i].assign_add(   tf.abs(gradient_MAS[i][0])  ))
    add_contributions_MASX_op.append( contributions_MASX[i].assign_add(   tf.abs(gradient_MASX[i][0])  ))

    add_contributions_MAS_2_op.append( contributions_MAS_2[i].assign_add(   tf.abs(gradient_MAS_2[i][0])  ))
    add_contributions_MASX_2_op.append( contributions_MASX_2[i].assign_add(   tf.abs(gradient_MASX_2[i][0])  ))

    add_contributions_EWC_op.append( contributions_EWC[i].assign_add(   my_factor_ph * tf.square(gradient[i][0])  ))
    add_contributions_AF_op.append( contributions_AF[i].assign_add(   my_factor_ph * tf.abs(gradient[i][0])  ))

    lengths_op.append( lengths[i].assign(variables[i] - length_variables[i]) )
    if HP['method'] == 'SI':
        importances_op.append( importances[i].assign_add(tf.maximum(tf.cast(0.0, dtype=dtype),tf.divide(contributions_SI[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )



##Prepare lists for storing different importance measures
cont_SI_all = []
cont_SIU_all = []
cont_SIB_all = []

cont_expSM_all = []

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
                sess.run(add_contributions_expSM_op)
            #END OF EPOCH
        #END OF TASK
        # Do what needs to be done for SI
        sess.run(lengths_op)
        cont_len_all.append(utils.flatten_list(sess.run(lengths)))
        sess.run(importances_op)
        sess.run(old_eq_new_op)

        #Calculate FISHER and MAS
        for i in range(HP['n_samples']):
            if i%100 == 0:
                    print('sample ', i)
            X_batch, Y_batch = mnist.train.next_batch(1)
            X_batch = X_batch[:,permutations[task]]  
            sess.run(add_contributions_MAS_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})
            sess.run(add_contributions_MASX_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})  

            sess.run(add_contributions_MAS_2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})
            sess.run(add_contributions_MASX_2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})    

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
        #cont_lens_all was handled above

        cont_expSM_all.append(   utils.flatten_list(sess.run(contributions_expSM))         )

        cont_EWC_all.append(   utils.flatten_list(sess.run(contributions_EWC))         )
        cont_AF_all.append(   utils.flatten_list(sess.run(contributions_AF))         )
        cont_MAS_all.append(   utils.flatten_list(sess.run(contributions_MAS))         )
        cont_MASX_all.append(   utils.flatten_list(sess.run(contributions_MASX))         )

        cont_MAS_2_all.append(   utils.flatten_list(sess.run(contributions_MAS_2))         )
        cont_MASX_2_all.append(   utils.flatten_list(sess.run(contributions_MASX_2))         )




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


my_dict_SI = {\
    'SI':cont_SI_all,\
    'SIU':cont_SIU_all,\
    'SIB':cont_SIB_all,\
    'expSM':cont_expSM_all,\
    'lengths':cont_len_all,\
        }
for key, val in my_dict_SI.items():
    my_dict_SI[key] = np.asarray(val)

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
a['SI-N'] = np.maximum(0,a['SI'])/(a['lengths']**2+HP['damp'])
a['SIB-N'] = np.maximum(0,a['SIB'])/(a['lengths']**2 + HP['damp'])
a['SIU-N'] = np.maximum(0,a['SIU'])/(a['lengths']**2 + HP['damp'])
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
            z = pearsonr(a[k1][task,:], a[k2][task,:])[0]
            all_corrs[k1][k2].append(z)
#save
with open('scatter_repetitions/all_correlations_MAS_'+meta+str(outer)+'.pickle', 'wb') as file:
    pickle.dump(all_corrs, file)

file = open('scatter_repetitions/summary.txt', 'a+')
file.write(str(np.mean(testing_acc_s))+' '+HP_label+'\n')
file.close()

