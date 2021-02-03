import os, sys, time
start = time.time()
end = time.time()
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
import numpy as np


## HYPERPARAMETERS
#inputs = ['dummy', '0', '0','MAS', '0.1', '1.0', '1']
inputs = sys.argv
visible_GPU = inputs[1]
save_outputs_to_log_dir = True

### HYPERPARAMETERS
# for method choose MAS, MASX, AF, EWC
HP = {\
'seed'              : int(inputs[2]),\
'method'            : inputs[3],\
'c'                 : float(inputs[4]),\
're_init_model'     : bool(float(inputs[5])),\
'n_samples'         : int(inputs[6]),\
'batch_size'        : 256,\
'n_tasks'           : 10,\
'n_epochs_per_task' : 20,\
'first_hidden'      : 2000,\
'second_hidden'     : 2000,\
}
    
HP_label = ''
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
l2_logits = tf.reduce_sum(tf.square(Y_logits))
l2_logits_max = tf.square(tf.reduce_max(Y_logits, axis=1))


#weight protection
old_variables = []
importances = []
contributions_MAS = []
contributions_MASX = []
contributions_MAS2 = []
contributions_MASX2 = []
contributions_EWC = []
contributions_AF = []
old_eq_new_op = []


for i in range(len(variables)):
    #store old variables
    old_variables.append( tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False) )
    old_eq_new_op.append(old_variables[i].assign(variables[i]))
    #store importances
    importances.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    #store changes
    contributions_MAS.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MASX.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MAS2.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_MASX2.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_EWC.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))
    contributions_AF.append(tf.Variable(tf.zeros(tf.shape(variables[i])), trainable=False))


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

gradient_MAS2 = trainer2.compute_gradients(l2_logits, variables)
gradient_MASX2 = trainer2.compute_gradients(l2_logits_max, variables)


#define ops to update importances
contributions_to_zero_op = []
importances_op = []
add_contributions_MAS_op = []
add_contributions_MASX_op = []
add_contributions_MAS2_op = []
add_contributions_MASX2_op = []
add_contributions_EWC_op = []
add_contributions_AF_op = []

for i in range(len(variables)):
    contributions_to_zero_op.append( tf.assign(contributions_MAS[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MAS2[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_MASX2[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_EWC[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    contributions_to_zero_op.append( tf.assign(contributions_AF[i], tf.zeros(tf.shape(contributions_AF[i]))) )
    
    add_contributions_MAS_op.append( contributions_MAS[i].assign_add(   tf.abs(gradient_MAS[i][0])  ))
    add_contributions_MASX_op.append( contributions_MASX[i].assign_add(   tf.abs(gradient_MASX[i][0])  ))
    add_contributions_MAS2_op.append( contributions_MAS2[i].assign_add(   tf.abs(gradient_MAS2[i][0])  ))
    add_contributions_MASX2_op.append( contributions_MASX2[i].assign_add(   tf.abs(gradient_MASX2[i][0])  ))
    add_contributions_EWC_op.append( contributions_EWC[i].assign_add(   my_factor_ph * tf.square(gradient[i][0])  ))
    add_contributions_AF_op.append( contributions_AF[i].assign_add(   my_factor_ph * tf.abs(gradient[i][0])  ))

    if HP['method'] == 'MAS':
        importances_op.append( importances[i].assign_add( contributions_MAS[i]/HP['n_samples']) )
    if HP['method'] == 'MASX':
        importances_op.append( importances[i].assign_add( contributions_MASX[i]/HP['n_samples']) )
    if HP['method'] == 'MAS2':
        importances_op.append( importances[i].assign_add( contributions_MAS2[i]/HP['n_samples']) )
    if HP['method'] == 'MASX2':
        importances_op.append( importances[i].assign_add( contributions_MASX2[i]/HP['n_samples']) )
    if HP['method'] == 'EWC':
        importances_op.append( importances[i].assign_add( contributions_EWC[i]/HP['n_samples']) )
    if HP['method'] == 'rEWC':
        importances_op.append( importances[i].assign_add( tf.sqrt(contributions_EWC[i]/HP['n_samples'])) )
    if HP['method'] == 'AF':
        importances_op.append( importances[i].assign_add( contributions_AF[i]/HP['n_samples']) )


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
        
        for epoch in range(HP['n_epochs_per_task']):
            print("Epoch", epoch, '; Task', task)
            end = time.time()
            print('Time ', end-start)
            start = time.time()
            for j in range(n_iterations):
                X_batch, Y_batch = mnist.train.next_batch(HP['batch_size'])
                X_batch = X_batch[:,permutations[task]]
                sess.run(train,feed_dict={X_ph:X_batch, Y_ph:Y_batch})
            #END OF EPOCH
        #END OF TASK
        #Calcualte Fisher or so
        for i in range(HP['n_samples']):
              X_batch, Y_batch = mnist.train.next_batch(1)
              X_batch = X_batch[:,permutations[task]]  
              if HP['method'] == 'MAS':
                  sess.run(add_contributions_MAS_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})
              if HP['method'] == 'MASX':
                  sess.run(add_contributions_MASX_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch}) 
              if HP['method'] == 'MAS2':
                  sess.run(add_contributions_MAS2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})
              if HP['method'] == 'MASX2':
                  sess.run(add_contributions_MASX2_op, feed_dict={X_ph:X_batch, Y_ph:Y_batch})   
              if HP['method'] == 'EWC' or HP['method'] == 'AF' or HP['method'] == 'rEWC':
                  predictions = sess.run(Y_pred,feed_dict={X_ph:X_batch, Y_ph:Y_batch}) 
                  for ii in range(10):  
                      Y_fake = np.zeros([1,10])
                      Y_fake[0,ii] = 1
                      my_factor = predictions[0,ii]
                      if HP['method'] == 'EWC' or HP['method'] == 'rEWC':
                          sess.run(add_contributions_EWC_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, my_factor_ph:my_factor})
                      if HP['method'] == 'AF':
                          sess.run(add_contributions_AF_op, feed_dict={X_ph:X_batch, Y_ph:Y_fake, my_factor_ph:my_factor})      
        
        # Update Importances and Old variables
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

file = open('summary_MAS.txt', 'a+')
file.write(str(np.mean(testing_acc_s))+' '+HP_label+'\n')
file.close()


