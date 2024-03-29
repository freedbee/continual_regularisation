import os, sys, time
start = time.time()
end = time.time()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np
import pickle


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','SI', '0.1', '1.0', '1']
inputs = sys.argv
visible_GPU = '0'
save_outputs_to_log_dir = False

### HYPERPARAMETERS
compute_full_gradients = False             #calculates full training set gradient at each step -- may take a while...
meta = 'run_new'                           #influences name of .pickle file which will store data from this run
HP = {\
'seed'              : 0,\
'method'            : 'SI',\
'c'                 : 0.2,\
're_init_model'     : True,\
'rescale'           : 1.0,\
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
    
HP_label = 'summed_importances'
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


#weight protection
old_variables = []
prev_variables = []
importances = []
contributions_SI = []
contributions_SIU = []
contributions_SIU_full = []
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
    contributions_SIU_full.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
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


#define ops to update importances
contributions_to_zero_op = []
lengths_op = []
importances_op = []
old_gradients_var = []
old_gradients_op = []
old_gradients_unbiased_var = []
old_gradients_unbiased_op = []
old_gradients_full_var = []
old_gradients_full_op = []
add_contributions_SI_op = []
add_contributions_SIU_op = []
add_contributions_SIU_full_op = []



for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_SI[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_SIU_full[i], tf.zeros(tf.shape(contributions_SI[i]))) )
    
    old_gradients_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_op.append(old_gradients_var[i].assign(gradient[i][0]))
    old_gradients_unbiased_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_unbiased_op.append(old_gradients_unbiased_var[i].assign(gradient[i][0]))
    old_gradients_full_var.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_gradients_full_op.append(old_gradients_full_var[i].assign(gradient[i][0]))

    
    add_contributions_SI_op.append( contributions_SI[i].assign_add(   tf.multiply(-old_gradients_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_op.append( contributions_SIU[i].assign_add(   tf.multiply(-old_gradients_unbiased_var[i],variables[i]-prev_variables[i])  ))
    add_contributions_SIU_full_op.append( contributions_SIU_full[i].assign_add(   tf.multiply(-old_gradients_full_var[i],variables[i]-prev_variables[i])  ))
    
    lengths_op.append( lengths[i].assign(variables[i] - length_variables[i]) )
    if HP['method'] == 'SI':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SI[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    if HP['method'] == 'SIU':
        importances_op.append( importances[i].assign_add(tf.maximum(0.0,tf.divide(contributions_SIU[i],tf.square(lengths[i])*HP['rescale'] + HP['damp']) )) )
    

# Summed contributions
cont_biased = 0.0
cont_unbiased = 0.0
cont_full = 0.0
for i in range(len(variables)):
    cont_biased += tf.reduce_sum(contributions_SI[i])
    cont_unbiased += tf.reduce_sum(contributions_SIU[i])
    cont_full += tf.reduce_sum(contributions_SIU_full[i])
#lists to store above quantities
summed_biased = []
summed_unbiased = []
summed_full = []
loss_hist = []    
    
    
    
    
    

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
        summed_biased.append([])
        summed_unbiased.append([])
        summed_full.append([])
        loss_hist.append([])    
        
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
                
                ## NOW SAMPLE BATCH FOR OPTIMIZER 
                X_batch, Y_batch = mnist.train.next_batch(HP['batch_size'])
                X_batch = X_batch[:,permutations[task]]
                t_loss, _ = sess.run([CEL,old_gradients_op],feed_dict={X_ph:X_batch, Y_ph:Y_batch})
                ## CALCULATE GRADIENT ON FULL TRAINING SET, if requested
                if compute_full_gradients:
                    t_loss, _ = sess.run([CEL,old_gradients_full_op],feed_dict={X_ph:mnist.train.images[:,permutations[task]], Y_ph:mnist.train.labels})
                # TRAIN
                sess.run(prev_eq_curr_op)    
                sess.run(train,feed_dict={X_ph:X_batch, Y_ph:Y_batch})
                
                #UPDATE RUNNING SUMs FOR ALGOs
                sess.run(add_contributions_SI_op)  
                sess.run(add_contributions_SIU_op)  
                sess.run(add_contributions_SIU_full_op)  
                # store the summed importances
                sum_b, sum_u, sum_f = sess.run([cont_biased, cont_unbiased, cont_full])
                summed_biased[task].append(sum_b)
                summed_unbiased[task].append(sum_u)
                summed_full[task].append(sum_f) 
                loss_hist[task].append(t_loss) #full training loss if computed. noise training batch otherwise.

            #END OF EPOCH
        #END OF TASK
        # Do what needs to be done for SI
        sess.run(lengths_op)
        sess.run(importances_op)
        sess.run(old_eq_new_op)

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

        
if save_outputs_to_log_dir:
    sys.stdout = orig_stdout
    f.close()

my_dict = {'summed_biased':summed_biased,\
           'summed_unbiased':summed_unbiased,\
           'summed_full':summed_full,\
           'loss_hist': loss_hist,\
          }    
with open('summed_importances_full-grad='+str(compute_full_gradients)+'_'+meta+'.pickle', 'wb') as file:
    pickle.dump(my_dict, file)    
    
    
file = open('summary_2.txt', 'a+')
file.write(str(np.mean(testing_acc_s))+' '+HP_label+'\n')
file.close()

