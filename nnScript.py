import numpy as np

from scipy.optimize import minimize

from scipy.io import loadmat

from math import sqrt

import pickle

import time

cols_selected = []

def initializeWeights(n_in, n_out):

    """

    # initializeWeights return the random weights for Neural Network given the

    # number of node in the input layer and output layer



    # Input:

    # n_in: number of nodes of the input layer

    # n_out: number of nodes of the output layer

       

    # Output: 

    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""



    epsilon = sqrt(6) / sqrt(n_in + n_out + 1)

    W = (np.random.rand(n_out, n_in + 1) * 2 * epsilon) - epsilon

    return W





def sigmoid(z):

    """# Notice that z can be a scalar, a vector or a matrix

    # return the sigmoid of input z"""



    return 1 / (1 + np.exp(-z))  # your code here





def preprocess_small():

    """ Input:

     Although this function doesn't have any input, you are required to load

     the MNIST data set from file 'mnist_sample.mat'.



     Output:

     train_data: matrix of training set. Each row of train_data contains

       feature vector of a image

     train_label: vector of label corresponding to each image in the training

       set

     validation_data: matrix of training set. Each row of validation_data

       contains feature vector of a image

     validation_label: vector of label corresponding to each image in the

       training set

     test_data: matrix of training set. Each row of test_data contains

       feature vector of a image

     test_label: vector of label corresponding to each image in the testing

       set



     - feature selection"""



    mat = loadmat('mnist_sample.mat')

    # ------------Initialize preprocess arrays----------------------#

    train_preprocess = np.zeros(shape=(4996, 784))

    validation_preprocess = np.zeros(shape=(1000, 784))

    test_preprocess = np.zeros(shape=(996, 784))

    train_label_preprocess = np.zeros(shape=(4996,))

    validation_label_preprocess = np.zeros(shape=(1000,))

    test_label_preprocess = np.zeros(shape=(996,))

    # ------------Initialize flag variables----------------------#

    train_len = 0

    validation_len = 0

    test_len = 0

    train_label_len = 0

    validation_label_len = 0

    # ------------Start to split the data set into 6 arrays-----------#

    for key in mat:

        # -----------when the set is training set--------------------#

        if "train" in key:

            label = key[-1]  # record the corresponding label

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)  # get the length of current training set

            tag_len = tup_len - 100  # defines the number of examples which will be added into the training set



            # ---------------------adding data to training set-------------------------#

            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[100:], :]

            train_len += tag_len



            train_label_preprocess[train_label_len:train_label_len + tag_len] = label

            train_label_len += tag_len



            # ---------------------adding data to validation set-------------------------#

            validation_preprocess[validation_len:validation_len + 100] = tup[tup_perm[0:100], :]

            validation_len += 100



            validation_label_preprocess[validation_label_len:validation_label_len + 100] = label

            validation_label_len += 100



            # ---------------------adding data to test set-------------------------#

        elif "test" in key:

            label = key[-1]

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)

            test_label_preprocess[test_len:test_len + tup_len] = label

            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]

            test_len += tup_len

            # ---------------------Shuffle,double and normalize-------------------------#

    train_size = range(train_preprocess.shape[0])

    train_perm = np.random.permutation(train_size)

    train_data = train_preprocess[train_perm]

    train_data = np.double(train_data)

    train_data = train_data / 255.0

    train_label = train_label_preprocess[train_perm]



    validation_size = range(validation_preprocess.shape[0])

    vali_perm = np.random.permutation(validation_size)

    validation_data = validation_preprocess[vali_perm]

    validation_data = np.double(validation_data)

    validation_data = validation_data / 255.0

    validation_label = validation_label_preprocess[vali_perm]



    test_size = range(test_preprocess.shape[0])

    test_perm = np.random.permutation(test_size)

    test_data = test_preprocess[test_perm]

    test_data = np.double(test_data)

    test_data = test_data / 255.0

    test_label = test_label_preprocess[test_perm]



    # Feature selection

    # Your code here.

    cols_to_del = []

    cols_selected = []

    for i in range(0,len(train_data.T)):

        curr_col = train_data[:,i]

        unique_feature_values = np.unique(curr_col)

        if len(unique_feature_values) == 1:

            cols_to_del.append(i)

        else:

            cols_selected.append(i)

    train_data = np.delete(train_data, cols_to_del, axis=1)

    validation_data = np.delete(validation_data, cols_to_del, axis=1)

    test_data = np.delete(test_data, cols_to_del, axis=1)

    print('preprocess done')



    return train_data, train_label, validation_data, validation_label, test_data, test_label



def preprocess():

    """ Input:

     Although this function doesn't have any input, you are required to load

     the MNIST data set from file 'mnist_all.mat'.



     Output:

     train_data: matrix of training set. Each row of train_data contains 

       feature vector of a image

     train_label: vector of label corresponding to each image in the training

       set

     validation_data: matrix of training set. Each row of validation_data 

       contains feature vector of a image

     validation_label: vector of label corresponding to each image in the 

       training set

     test_data: matrix of training set. Each row of test_data contains 

       feature vector of a image

     test_label: vector of label corresponding to each image in the testing

       set



     Some suggestions for preprocessing step:

     - feature selection"""



    mat = loadmat('mnist_all.mat')  # loads the MAT object as a Dictionary



    # Pick a reasonable size for validation data



    # ------------Initialize preprocess arrays----------------------#

    train_preprocess = np.zeros(shape=(50000, 784))

    validation_preprocess = np.zeros(shape=(10000, 784))

    test_preprocess = np.zeros(shape=(10000, 784))

    train_label_preprocess = np.zeros(shape=(50000,))

    validation_label_preprocess = np.zeros(shape=(10000,))

    test_label_preprocess = np.zeros(shape=(10000,))

    # ------------Initialize flag variables----------------------#

    train_len = 0

    validation_len = 0

    test_len = 0

    train_label_len = 0

    validation_label_len = 0

    # ------------Start to split the data set into 6 arrays-----------#

    for key in mat:

        # -----------when the set is training set--------------------#

        if "train" in key:

            label = key[-1]  # record the corresponding label

            tup = mat.get(key)

     #       print (tup)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)  # get the length of current training set

            tag_len = tup_len - 1000  # defines the number of examples which will be added into the training set



            # ---------------------adding data to training set-------------------------#

            train_preprocess[train_len:train_len + tag_len] = tup[tup_perm[1000:], :]

            train_len += tag_len



            train_label_preprocess[train_label_len:train_label_len + tag_len] = label

            train_label_len += tag_len



            # ---------------------adding data to validation set-------------------------#

            validation_preprocess[validation_len:validation_len + 1000] = tup[tup_perm[0:1000], :]

            validation_len += 1000



            validation_label_preprocess[validation_label_len:validation_label_len + 1000] = label

            validation_label_len += 1000



            # ---------------------adding data to test set-------------------------#

        elif "test" in key:

            label = key[-1]

            tup = mat.get(key)

            sap = range(tup.shape[0])

            tup_perm = np.random.permutation(sap)

            tup_len = len(tup)

            test_label_preprocess[test_len:test_len + tup_len] = label

            test_preprocess[test_len:test_len + tup_len] = tup[tup_perm]

            test_len += tup_len

            # ---------------------Shuffle,double and normalize-------------------------#

    train_size = range(train_preprocess.shape[0])

    train_perm = np.random.permutation(train_size)

    train_data = train_preprocess[train_perm]

    train_data = np.double(train_data)

    train_data = train_data / 255.0

    train_label = train_label_preprocess[train_perm]



    validation_size = range(validation_preprocess.shape[0])

    vali_perm = np.random.permutation(validation_size)

    validation_data = validation_preprocess[vali_perm]

    validation_data = np.double(validation_data)

    validation_data = validation_data / 255.0

    validation_label = validation_label_preprocess[vali_perm]



    test_size = range(test_preprocess.shape[0])

    test_perm = np.random.permutation(test_size)

    test_data = test_preprocess[test_perm]

    test_data = np.double(test_data)

    test_data = test_data / 255.0

    test_label = test_label_preprocess[test_perm]



    # Feature selection

    # Your code here.



    cols_to_del = []

#    cols_selected = []

    for i in range(0,len(train_data.T)):

        curr_col = train_data[:,i]

        unique_feature_values = np.unique(curr_col)

        if len(unique_feature_values) == 1:

            cols_to_del.append(i)

        else:

            cols_selected.append(i)

    train_data = np.delete(train_data, cols_to_del, axis=1)

    validation_data = np.delete(validation_data, cols_to_del, axis=1)

    test_data = np.delete(test_data, cols_to_del, axis=1)

    #print('preprocess done')



    return train_data, train_label, validation_data, validation_label, test_data, test_label





def nnObjFunction(params, *args):

    """% nnObjFunction computes the value of objective function (negative log 

    %   likelihood error function with regularization) given the parameters 

    %   of Neural Networks, thetraining data, their corresponding training 

    %   labels and lambda - regularization hyper-parameter.



    % Input:

    % params: vector of weights of 2 matrices w1 (weights of connections from

    %     input layer to hidden layer) and w2 (weights of connections from

    %     hidden layer to output layer) where all of the weights are contained

    %     in a single vector.

    % n_input: number of node in input layer (not include the bias node)

    % n_hidden: number of node in hidden layer (not include the bias node)

    % n_class: number of node in output layer (number of classes in

    %     classification problem

    % training_data: matrix of training data. Each row of this matrix

    %     represents the feature vector of a particular image

    % training_label: the vector of truth label of training images. Each entry

    %     in the vector represents the truth label of its corresponding image.

    % lambda: regularization hyper-parameter. This value is used for fixing the

    %     overfitting problem.

       

    % Output: 

    % obj_val: a scalar value representing value of error function

    % obj_grad: a SINGLE vector of gradient value of error function

    % NOTE: how to compute obj_grad

    % Use backpropagation algorithm to compute the gradient of error function

    % for each weights in weight matrices.



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % reshape 'params' vector into 2 matrices of weight w1 and w2

    % w1: matrix of weights of connections from input layer to hidden layers.

    %     w1(i, j) represents the weight of connection from unit j in input 

    %     layer to unit i in hidden layer.

    % w2: matrix of weights of connections from hidden layer to output layers.

    %     w2(i, j) represents the weight of connection from unit j in hidden 

    %     layer to unit i in output layer."""



    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args



    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))

    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

    obj_val = 0   #cumulative objective function



    # Your code here

    #print(len(training_data))

    #print(len(training_data.T))]



    training_data1 = np.reshape(training_data, (len(training_data),len(training_data.T)))

    training_data1_biased = np.insert(training_data1, len(training_data.T), 1, axis=1)

    data_size = len(training_data1)





    #calculating obj_val

    a = np.zeros(shape=(n_hidden, data_size))

    b = np.zeros(shape=(n_class, data_size))



    a = np.dot(w1, training_data1_biased.T) #calculated in hidden layer, added bias

    z = sigmoid(a)  #output from hidden layer after activation



    z_biased = np.insert(z, len(z), 1, axis=0)

    b = np.dot(w2, z_biased) #calculated in output layer

    o = sigmoid(b) #output from output layer after activation



    y = np.zeros(shape=(n_class, data_size))

    for i in range(0,data_size):

        y[training_label[i]][i] = 1



    log_o = np.log(o)

    sub_y = np.subtract(1,y)

    log_sub_o = np.log(np.subtract(1,o))



    obj_val_raw = 0 - ((1/data_size) * (np.sum(np.multiply(y,log_o) + np.multiply(sub_y, log_sub_o))))

    w1_sq = np.square(w1)
    
    w2_sq = np.square(w2)
    
    w1_sum = np.sum(w1_sq)

    w2_sum = np.sum(w2_sq)

    obj_val_hyper = np.multiply((lambdaval/(2*data_size)), (w1_sum + w2_sum))

    obj_val = obj_val_raw + obj_val_hyper



    #calculating obj_val

    delta = o - y

    temp1 = np.dot(delta, z_biased.T)

    temp2 = np.multiply(lambdaval, w2)

    grad_w2 = np.multiply(1/(data_size), temp1 + temp2)



    w2_unbiased = np.delete(w2,len(w2.T) - 1, axis = 1)



    temp1 = np.multiply(np.subtract(1,z), z)

    temp2 = np.dot(delta.T, w2_unbiased)

    temp3 = np.multiply(temp1.T, temp2)

    temp4 = np.dot(temp3.T, training_data1_biased)

    grad_w2_hyper = np.multiply(lambdaval, w1)

    grad_w1 = np.multiply(1/(data_size),temp4 + grad_w2_hyper)



    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2

    # you would use code similar to the one below to create a flat array

    # obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()),0)

    obj_grad = np.array([])

    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)



    #print ("objective function value: " + str(obj_val))

    return (obj_val, obj_grad)





def nnPredict(w1, w2, data):

    """% nnPredict predicts the label of data given the parameter w1, w2 of Neural

    % Network.



    % Input:

    % w1: matrix of weights of connections from input layer to hidden layers.

    %     w1(i, j) represents the weight of connection from unit i in input 

    %     layer to unit j in hidden layer.

    % w2: matrix of weights of connections from hidden layer to output layers.

    %     w2(i, j) represents the weight of connection from unit i in input 

    %     layer to unit j in hidden layer.

    % data: matrix of data. Each row of this matrix represents the feature 

    %       vector of a particular image

       

    % Output: 

    % label: a column vector of predicted labels"""



    labels = np.array([])

    # Your code here

    data1 = np.reshape(data, (len(data),len(data.T)))

    data1_biased = np.insert(data1, len(data.T), 1, axis=1)

    data_size = len(data1)





    #calculating obj_val

    a = np.zeros(shape=(n_hidden, data_size))

    b = np.zeros(shape=(n_class, data_size))



    a = np.dot(w1, data1_biased.T) #calculated in hidden layer, added bias

    z = sigmoid(a)  #output from hidden layer after activation



    z_biased = np.insert(z, len(z), 1, axis=0)

    b = np.dot(w2, z_biased) #calculated in output layer

    o = sigmoid(b) #output from output layer after activation



    labels = np.argmax(o, axis = 0)

    return labels





"""**************Neural Network Script Starts here********************************"""



train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()

#train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess_small()



#  Train Neural Network



# set the number of nodes in input unit (not including bias unit)

n_input = train_data.shape[1]



# set the number of nodes in hidden unit (not including bias unit)

n_hidden = 50


# set the number of nodes in output unit

n_class = 10



# initialize the weights into some random matrices

initial_w1 = initializeWeights(n_input, n_hidden)

initial_w2 = initializeWeights(n_hidden, n_class)



# unroll 2 weight matrices into single column vector

initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()), 0)



# set the regularization hyper-parameter

lambdaval = 10



args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)



# Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example



opts = {'maxiter': 50}  # Preferred value.

start_time = time.time()

nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args, method='CG', options=opts)

end_time = time.time()

time_taken = end_time - start_time

print("\n Hidden nodes: ", n_hidden)
print("\n Training time: ", time_taken)

# In Case you want to use fmin_cg, you may have to split the nnObjectFunction to two functions nnObjFunctionVal

# and nnObjGradient. Check documentation for this function before you proceed.

# nn_params, cost = fmin_cg(nnObjFunctionVal, initialWeights, nnObjGradient,args = args, maxiter = 50)





# Reshape nnParams from 1D vector into w1 and w2 matrices

w1 = nn_params.x[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))

w2 = nn_params.x[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))



# Test the computed parameters



predicted_label = nnPredict(w1, w2, train_data)



# find the accuracy on Training Dataset



print('\n Training set Accuracy:' + str(100 * np.mean((predicted_label == train_label).astype(float))) + '%')



predicted_label = nnPredict(w1, w2, validation_data)



# find the accuracy on Validation Dataset



print('\n Validation set Accuracy:' + str(100 * np.mean((predicted_label == validation_label).astype(float))) + '%')



predicted_label = nnPredict(w1, w2, test_data)



# find the accuracy on Validation Dataset



print('\n Test set Accuracy:' + str(100 * np.mean((predicted_label == test_label).astype(float))) + '%')

obj = [cols_selected, n_hidden, w1, w2, lambdaval]
# selected_features is a list of feature indices that you use after removing unwanted features in feature selection step

pickle.dump(obj, open('params.pickle', 'wb'))

