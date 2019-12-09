'''
############THIS FILE IS TO COMPARE THE EFFICIENCIES OF TWO CENTER SELECTION METHODS
'''
import os
import pandas as pd
import numpy as np
import copy
import math
'''################################################################################'''

'''
Generate a small data set to test the following functions
'''
def Small_sample(size = 10):
    att1 = ['A', 'B']
    att2 = [1,2,3]
    att3 = ['D', 'E','F']
    label = ['y1', 'y2']
    # Randomly assign the values
    att1_v = np.random.choice(att1, size)
    att2_v = np.random.choice(att2, size)
    att3_v = np.random.choice(att3, size)
    label_v = np.random.choice(label, size)
    # Combine them into a data frame
    sample_data = pd.DataFrame({'att1':att1_v, 'att2':att2_v, 'att3': att3_v, 'label':label_v})
    return sample_data

'''
##########################################################################################3
****************************************************************************************8
         DEFINE THE RADIAL BASIS FUNCTIONS
         
RBF: the radial basis function
'''
def Thin_Plate_Spline(d):
    design_matrix = np.zeros(d.shape)
    nrow, ncol = d.shape
    for i in range(nrow):
        for j in range(ncol):
            dist = d[i,j]
            if dist != 0:
                design_matrix[i,j] = dist**2 * np.log(dist)
            else:
                design_matrix[i,j] = 0
            
    return design_matrix

def Gaussian(distance, radius):
    return np.exp(-distance**2/radius**2)
    
def Markov(distance, radius):
    return np.exp(-distance/radius)  

def Inverse_Multi_Quadric(distance,c, beta):
    return (distance**2 + c**2)**(-beta) 

def Design_matrix(distance_matrix, RBF = 'Gaussian'):
    
    if RBF == 'Gaussian':
        design_matrix = Gaussian(distance_matrix, radius = 1)
    elif RBF == 'Markov':
        design_matrix = Markov(distance_matrix, radius = 1)
    elif RBF == 'Thin_Plate_Spline':
        design_matrix = Thin_Plate_Spline(distance_matrix)
    elif RBF == 'Inverse_Multi_Quadric':
        design_matrix = Inverse_Multi_Quadric(distance_matrix, c = 0.5, beta=1)
        
    return design_matrix

'''
#######################################################################################
'''
def Frequency_distinct(col):
    distinct_values = list(set(col))
    df_element = pd.DataFrame({'d_values':[], 'freq':[]})
    for i in distinct_values:
        df_element = df_element.append({'d_values':i, 'freq':sum(col == i)},\
                                        ignore_index = True)
    
    freq_col = col.apply(lambda x:np.int(df_element[x==df_element['d_values']]['freq']))
        
    return freq_col
'''
Preparation is for data wrangling and calculate the distance matrix according to the
selected distance
*******************************************************************************
Inputs:
    data_frame: The training data frame
    attributes: a list gives the names of the attributes used as explatary variables, which are consistant
                with  the names of the columns.
    categories: a list contains the names of the columns of the labels.
Outputs:
    different types of distance matrices
'''
class Distance_martix:
    def __init__(self, data_frame, attributes, categories):
        self.att = data_frame[attributes]
        self.cate = data_frame[categories]
        nrow, ncol = self.att.shape
    def Hamming_matrix(self):
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                distance_matrix[i,j] = np.sum(self.att.iloc[i] != self.att.iloc[j])
                distance_matrix[j,i] = distance_matrix[i,j]
        self.hamming_matrix = distance_matrix/ncol
        
    def IOF_matrix(self):
        nrow, ncol = self.att.shape
        freq_frame = self.att.apply(Frequency_distinct, axis = 0)
        freq_frame_log = np.log(freq_frame, dtype = np.float32)
        self.freq_frame = freq_frame
        self.freq_frame_log = freq_frame_log
        
#        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i, nrow):
                indicator =1- self.att.iloc[i].eq(self.att.iloc[j])
                product = np.float32(freq_frame_log.iloc[i]*freq_frame_log.iloc[j])
                distance_matrix[i,j] = np.sum(indicator * product)
                distance_matrix[j,i] = distance_matrix[i,j]
        self.iof_matrix = np.float32(distance_matrix/ncol)

    def OF_matrix(self):
        nrow, ncol = self.att.shape
        logk = np.log(nrow, dtype=np.float32)
        distance_matrix = np.zeros((nrow, nrow))
        frame_log_f_div_k = self.freq_frame_log - logk
        self.frame_log_f_div_k  = frame_log_f_div_k 
        for i in range(nrow):
            for j in range(i, nrow):
                indicator =1- self.att.iloc[i].eq(self.att.iloc[j])
                product = frame_log_f_div_k.iloc[i]*frame_log_f_div_k.iloc[j]
                distance_matrix[i,j] = np.sum(indicator * product)
                distance_matrix[j,i] = distance_matrix[i,j]
                self.of_matrix = np.float32(distance_matrix/ncol)
    '''There should be at least 3 values for each attribute in order to use Lin_matrix'''
    def Lin_matrix(self):
        nrow, ncol = self.att.shape
        distance_matrix = np.zeros((nrow, nrow))
        frame_log_f_div_k = self.frame_log_f_div_k 
        
        A_matrix = np.zeros(distance_matrix.shape)
        for i in range(nrow):
            for j in range(i, nrow):
                A_matrix[i,j] = np.sum(frame_log_f_div_k.iloc[i]+frame_log_f_div_k.iloc[j])
                A_matrix[j,i] = A_matrix[i,j]  
        self.A_matrix = A_matrix

        for i in [1]:
            for j in [1]:
                equal_series = A_matrix[i,j]/(2*frame_log_f_div_k.iloc[i]) - 1
                if self.att.iloc[i].equals(self.att.iloc[j]):
                    distance_matrix[i,j] = np.sum(equal_series)
                else:
                    unequal_series = A_matrix[i,j]/(2*np.log((self.freq_frame.iloc[i]+self.freq_frame.iloc[j])/nrow,  dtype=np.float32))- 1
                    equal_indicator = self.att.iloc[i].eq(self.att.iloc[j])
                    unequal_indicator = 1 - equal_indicator   
                    distance_matrix[i,j] = np.sum(equal_indicator*equal_series + unequal_indicator * unequal_series)

                distance_matrix[j,i] = distance_matrix[i,j]
                
        self.lin_matrix = np.float32(distance_matrix/ncol)

    def Burnaby_matrix(self):
        nrow, ncol = self.att.shape
        # Make sure there are at least two values for each attribute
        distance_matrix = np.zeros((nrow, nrow))
        nrow_minus_freq = nrow - self.freq_frame
        relative_freq = self.freq_frame / nrow
        denorminator_series = pd.DataFrame()
        
        for col in relative_freq.columns:
            column = relative_freq[col]
            s = 0
            for val in set(self.att[col]):
                s += 2 * np.log(1-column[self.att[col] == val].iloc[0])
            denorminator_series[col] = [s]
                
        for i in range(nrow):
            for j in range(i, nrow):
                equal_indicator = (self.att.iloc[i] == self.att.iloc[j])
                unequal_indicator =  1 - equal_indicator
                freq_product = self.freq_frame.iloc[i] * self.freq_frame.iloc[j]
                nrow_minus_freq_product = nrow_minus_freq.iloc[i] * nrow_minus_freq.iloc[j]
                numerator_series =np.log(freq_product / nrow_minus_freq_product, dtype = np.float32)
                
                unequal_series = numerator_series / denorminator_series
                unequal_series /= ncol
                
                distance_matrix[i,j] = np.sum(equal_indicator + unequal_indicator * unequal_series, axis = 1)
                distance_matrix[j,i] = distance_matrix[i,j]
                
        self.burnaby_matrix = distance_matrix

    def Eskin_matrix(self):
        nrow, ncol = self.att.shape
        ni_frame = self.att.apply(set, axis= 0).apply(lambda x:len(x))
        unequal_series = 2/(ni_frame*ni_frame)
        distance_matrix = np.zeros((nrow, nrow))
        for i in range(nrow):
            for j in range(i,nrow):
                unequal_indicator = (self.att.iloc[i] != self.att.iloc[j])
                distance_matrix[i,j] = np.sum(unequal_indicator * unequal_series)
                distance_matrix[j,i] = distance_matrix[i,j]
        self.eskin_matrix = distance_matrix / ncol

    # Define the distances

'''
###################################################################################
'''

'''****************************************************************************************
********************************************************************************************
All the above functions have been verified by  22:00 Aug.10th'''
'''#################################################################################
            CALCULATE THE DESIGN MATRIX FOR CENTERS SELECTED BY COVERAGE METHOD

**************************************************************************************
CS_coverage_all: a function to find the oder of the centers to be eliminated and the corresponding 
                cutoff distances
Inputs:
    distance_matrix
    
Outputs:
    eliminated: a list, gives the indices, corresponding to the rows, to be removed, so that the leftovers
                can be used as centers
    radius: a list, gives the cutoff distances that corresponding to the elements in the eliminated.
    
For example:
    eliminated = [1,2,3], radius = [1, 1, 2]
    Means: to remove center corresponding to row 1, the cutoff distance need to be 1.
           to remove center corresponding to row 2, the cutoff distance need to be 1.
           to remove center corresponding to row 3, the cutoff distance need to be 2.
'''
def CS_coverage_all(distance_matrix):
    l = list(np.ndenumerate(distance_matrix))
    # get rid of the diagonal elements
    l_sub = [x for x in l if x[0][0] != x[0][1]]
    # arrange the distance in the increasing order
    l_sub.sort(key = lambda x:x[1])

    eliminated = []; radius = []
    while l_sub !=[]:
        eliminated.append(l_sub[0][0][1])
        radius.append(l_sub[0][1])
        # update l_sub
        l_sub = [x for x in l_sub if x[0][0] != l_sub[0][0][1] and x[0][1] != l_sub[0][0][1]]
        
    return eliminated, radius

'''
Design_matrix_coverage_by_nCenters: find centers with the size equal to nCenters

Inputs:
        distance_matrix: as above
        eliminated: as above
        nCenters: integer, the number of centers
Outputs:
    design_matrix: a submatrix of the distance matrix, with the rows corresponding to the
                    samples and the columns corresponding to the selected centers
'''
def Design_matrix_coverage_by_nCenters(train_design_matrix, test_design_matrix, eliminated, nCenters):
    
    _, ncol= train_design_matrix.shape
    to_be_del = eliminated[:ncol - nCenters]
    sub_train_dm = np.delete(train_design_matrix, to_be_del, axis=1)
    sub_test_dm = np.delete(test_design_matrix, to_be_del, axis =1)
        
    return sub_train_dm, sub_test_dm

'''
Design_matrix_coverage_by_radius: find centers with the cut off distance cutoff

Inputs:
        distance_matrix: as above
        eliminated: as above
        radius: as above
        cutoff: float, gives the cutoff distance. For each center, any sample with 
            the distance from it no larger than cutoff will not be selected as a center.
Outputs:
    design_matrix: a submatrix of the distance matrix, with the rows corresponding to the
                    samples and the columns corresponding to the selected centers
'''

def Design_matrix_coverage_by_radius(train_design_matrix, test_design_matrix, eliminated, radius, cutoff):
    
    nrow, _ = train_design_matrix.shape
    to_be_del = []
    for ind, v in enumerate(radius):
        if v <= cutoff:
            to_be_del.append(eliminated[ind])
            
    sub_train_dm = np.delete(train_design_matrix, to_be_del, axis=1)
    sub_test_dm = np.delete(test_design_matrix, to_be_del, axis =1)
    
    return sub_train_dm, sub_test_dm
'''We should return the centers which will be used in testing'''


'''#################################################################################
****************************************************************************************
                CALCULATE THE LOSS AND THE GRADIENTS

Suppose: design_matrix is of dimension (m,n)                

Loss_Sigmoid: Use the sigmoid as the loss function
Inputs:
    design_matrix: as above
    labels: an array corresponding to the rows of the design_matrix,
            with values of either 0 or 1. labels.shape = (m, 1)
    coefficients: an array contains the coeffients, coefficients.shape = (n, 1)
    reg: a float, the coefficient of regularization
Outputs:
    loss: float
    grad_coefficients: an array containing all the gradients corresopnding to the coefficients
'''
def Loss_Sigmoid(design_matrix, labels, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    
    logit = design_matrix.dot(coefficients)
    prob = 1/(1+np.exp(-logit))
    loss = np.average(- np.log(prob) * labels - (1 - labels) * np.log(1 - prob))
    # plus the regularization
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradient from the first part of loss
    grad_logit = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_logit)
    grad_coefficients /= nrow
    # Calculate the gradient from the regularizatio part
    grad_coefficients += 2 * reg * coefficients
    
    # return the above results
    return loss, grad_coefficients
'''
Loss_Softmax: this function applies to the case when the output classes are more than 2
Input:
    design_matrix: as above
    labels: a matrix of dimension (m, k), where k is the number of classes. The label of
            each sample is a vector of dimenion (1, k), and the values are either 0 or 1, with
            1 indicate the correct category.
    coefficients: a matrix of dimension (n*k, 1). For the convenience of latter usage,  we don't use the shape(n, k).
                    When (n,k) is reshaped into (n*k,1), we stack column by column.
    reg: as above
Output:
    similar as above
    
THE FLAW OF SOFTMAX: IF THE DIFFERENCES OF THE LOGIT VALUES ARE TWO BIG, THE LOSS FUNCTION MAY BE TOO BIG!
'''
def Loss_Softmax(design_matrix, labels, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    # Reshape the coefficients
    coefficients = coefficients.reshape((-1, ncol)).T
    
    Wx = design_matrix.dot(coefficients)
    # Make sure the elements in Wx is not too big or too small
    Wx -= np.max(Wx, axis = 1, keepdims = True)
    # Calculate the probabilities
    exp = np.exp(Wx)
    prob = exp / np.sum(exp, axis = 1, keepdims = True)
    
    log_prob = np.log(prob)

    # Calculate  the loss
    loss = np.sum(- log_prob * labels)/nrow
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradients
    grad_Wx = prob - labels
    grad_coefficients = (design_matrix.T).dot(grad_Wx)
    grad_coefficients /= nrow
    
    grad_coefficients += 2 * reg * coefficients
    
    grad_coefficients = grad_coefficients.T.reshape((-1, 1))
    
    return loss, grad_coefficients

'''
'''

def Loss_SVM(design_matrix, observed, coefficients, reg):
     
    nrow, ncol = design_matrix.shape
    # Reshape the coefficients
    coefficients = coefficients.reshape((-1, ncol)).T
    # Calculate the loss
    ii = np.zeros((observed.shape[1], observed.shape[1])) + 1
    Wx = design_matrix.dot(coefficients)
    s1 = Wx + 1
    obs = observed * Wx
    obsii = obs.dot(ii)
    ad = s1 - obsii
    d = ad * (1-observed)
    ind = (d>0)
    sd = d * ind
    loss = np.sum(sd)
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradients
    grad_d = ind
    grad_ad = grad_d * (1-observed)
    grad_s1 = grad_ad
    grad_obsii = - grad_ad
    grad_Wx = grad_s1
    grad_obs = grad_obsii.dot(ii)
    grad_Wx += observed * grad_obs
    grad_coeff = (design_matrix.T).dot(grad_Wx)
    
    grad_coeff += 2 * reg * coefficients
    # Reshape the gradient
    grad_coeff = grad_coeff.T.reshape((-1, 1))
    return loss, grad_coeff


'''
Loss_SumSquares: the loss is measured by the sum of the sequares of the difference 
                between the predicted values and the observed values
Inputs:
    observed: an array of shape (m,1). with each element a float.
    design_matrix, coefficients and reg are the same as above
Outputs:
    the same as above
'''    
def Loss_SumSquares(design_matrix, observed, coefficients, reg):
    
    nrow, ncol = design_matrix.shape
    
    # Calculate the loss
    pred = design_matrix.dot(coefficients)
    loss = np.average((pred - observed) * (pred - observed))
    
    loss += reg * np.sum(coefficients * coefficients)
    
    # Calculate the gradient
    
    grad_coefficients = (design_matrix.T).dot(2 * (pred - observed))
    grad_coefficients /= nrow
    grad_coefficients += 2 * reg * coefficients
    
    return loss, grad_coefficients
'''
Integrate the above functions into one function for convenient usage.
Input:
    train_para: a dictionary, contains all the needed values to train a model.
'''
def Loss(train_para):
    design_matrix = train_para['design_matrix'] 
    observed = train_para['observed']
    reg = train_para['reg']
    coefficients = train_para['coefficients']
    loss_type = train_para['loss_type']
    if loss_type == 'Sigmoid':
        loss, grad_coefficients = Loss_Sigmoid(design_matrix, observed, coefficients, reg)
    elif loss_type == 'Softmax':
        loss, grad_coefficients = Loss_Softmax(design_matrix, observed, coefficients, reg)
    elif loss_type == 'SumSquares':
        loss, grad_coefficients = Loss_SumSquares(design_matrix, observed, coefficients, reg)
    elif loss_type == 'SVM':
        loss, grad_coefficients = Loss_SVM(design_matrix, observed, coefficients, reg)
    return loss, grad_coefficients


'''#################################################################################'''
'''
Train_GD: train the model by using gradient descent
Inputs:
    gd_train_para: a dictionary, contains
        reg: coefficients of regularization
        setp_size: a float
        loss_type: string, gives types of different loss functions
        design_matrix:
        observed: observed values, the format of which decides the type of loss functions
Outputs:  
    coefficients: a matrix of shape (design_matrix.shape[0], observed.shape[1])
                    the values of the coefficients after n_iterations training      
'''
def Train_GD(train_para):
    # Take out the parameters
#    design_matrix = train_para['design_matrix'] 
#    observed = train_para['observed']
#    reg = train_para['reg']
    step_size = train_para['step_size']
#    loss_type = gd_train_para['loss_type']
    n_iterations = train_para['n_iterations']    
    
    for i in range(n_iterations):
        loss, grad_coefficients = Loss(train_para)
        # update the coefficients
        train_para['coefficients'] -= step_size * grad_coefficients
        '''Do we print the loss'''
        if i % 100 == 0:
            print(round(loss, 6))
            
    return train_para, loss


def Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-4, termination = 1e-2):   
    
    nrow, _ = np.shape(train_para['coefficients']) 
    max_design = np.max(np.abs(train_para['design_matrix']))       
    # Create an iteration counter
    n_iteration = 0
    # BFGS algorithm
    loss, grad_coeff = Loss(train_para)
    # Initiate H. This H should not be large in case it may destroy the Loss function
    H = np.eye(nrow)
    H *= 1/(np.max(np.abs(grad_coeff)) * max_design)
    ternination_square = termination**2
    grad_square = ternination_square + 1
    while grad_square >= ternination_square:          
        # keep a record of this grad_square for monitoring the efficiency of this process
        n_iteration += 1
      
        p = - H.dot(grad_coeff)        
        # There should be both old and new coefficients in the train_para
        train_para['coefficients_old'] = train_para['coefficients']
        train_para['coefficients'] = p + train_para['coefficients_old']
        # Calculate the loss and gradient
        new_loss, new_grad_coeff = Loss(train_para)        
        # Ramijo Back-tracking
        while new_loss > loss + c * (grad_coeff.T).dot(p):
            p *= rho
            train_para['coefficients'] = p + train_para['coefficients_old']            
            new_loss, new_grad_coeff = Loss(train_para)        
        # update H
        s = p
        y = new_grad_coeff - grad_coeff
        r = (y.T).dot(s)
        I = np.eye(nrow)
        if r != 0:
            r = 1/r            
            H = (I - r*s.dot(y.T)).dot(H).dot(I - r*y.dot(s.T)) + r*s.dot(s.T)# Can be accelerated
        else:
            H = np.diag(np.random.uniform(0.5, 1, nrow))# try to eliminate the periodic dead loop
            H *= 1/(np.max(np.abs(new_grad_coeff))*max_design)# Make sure H is not too large
        # Update loss, grad_square and paramter
        loss = new_loss
        grad_coeff = new_grad_coeff
        grad_square = new_grad_coeff.T.dot(new_grad_coeff)            
        # print some values to monitor the training process 
        if n_iteration % 100 == 0:
            print('loss  ', loss, '    ','grad_square   ', grad_square)
            n_iteration = 0        
        
    return train_para, loss

'''#################################################################################'''
'''
**************THE FOLOWING BLOCK IS TO SELECT CENTER BY THE ABSOLUTE VALUES OF THE COEFFICIENTS
*************The general idea is to generate a list of centers and the corresponding trained 
************** coefficients, which can be applied to the testig set.

One_step_reduce_centers: This function is to reduce the centers on basis of the coefficients after 
                one round of training
Input:
    train_para: as above
    testing_design_matrix:
        a matrix with the rows the testing samples and the columns the centers. 
       the testing design matrix and the design matrix in the train_para should have the columuns corresponding 
       set of centers.
'''
def One_step_reduce_centers(train_para, testing_design_matrix, nCenters):
    
    method = train_para['train_method']
    nrow, ncol = train_para['design_matrix'].shape
    
    if ncol != testing_design_matrix.shape[1]:
        raise Exception('The centers don\'t match')
    
    else:
        m =1
        '''m is the number of centers to be removed at each round of training'''
        for i in range(m):    
            coeff = train_para['coefficients'] 
            # if it is the case of softmax, we have to reshape the coeff
            coeff = coeff.reshape((-1, ncol)).T
            sabs_coeff = np.sum(np.abs(coeff), axis = 1, keepdims = True)
            # find the index of the coefficients with the smallest absolute value
            ind_min = np.argmin(np.abs(sabs_coeff))
            # remove the smallest
            one_less_coeff = np.delete(coeff, ind_min, axis=0)
            train_para['coefficients'] = one_less_coeff.T.reshape((-1,1))
            testing_design_matrix = np.delete(testing_design_matrix, ind_min, axis = 1)
            train_para['design_matrix'] = np.delete(train_para['design_matrix'], ind_min, axis = 1)
            

        termination = 1e-3
        '''Here, we want the coefficient is well trained at the number of centers we need'''
    #    termination =10* len(centers)  
        if method == 'BFGS':      
            train_para, loss = Train_RBFN_BFGS(train_para, rho=0.85, c = 1e-3, termination=termination)
        elif method == 'GD':
            train_para, loss = Train_GD(train_para)           

    return testing_design_matrix

'''
Center_Select_Coef: select center according to the absolute values of the coefficients
Inputs:
    train_para: as above
    testing_design_matrix: as above
    nCenters_list: a list giving the number of centers to be kept
Outputs:
    test_para:
        nCenters_list: as above
        design_matrix_list: a list of testing design matrices, of which the columns
                            of each matrix are correponding to the numbers of the centers
                            given in the nCenters_list.
        coefficients_list: a list of coefficients corresponding to the design_matrix_list
        loss_type: as above
'''

def Center_Select_Coef(train_para, testing_design_matrix):
    nCenters_list = train_para['nCenters_list']
    test_para = {}
    test_para['nCenters_list'] = nCenters_list
    test_para['design_matrix_list'] = []
    test_para['coefficients_list'] = []
    test_para['loss_type'] = train_para['loss_type']
    
    for nCenters in nCenters_list:
        while testing_design_matrix.shape[1] > nCenters:
            testing_design_matrix = One_step_reduce_centers(train_para, testing_design_matrix, nCenters)
            print('\n Number of centers:   ', testing_design_matrix.shape[1], '\n')
            
        test_para['design_matrix_list'].append(copy.deepcopy(testing_design_matrix))
        test_para['coefficients_list'].append(copy.deepcopy(train_para['coefficients']))
        
    return test_para
'''#######################################################################################'''
'''
*************Strategy:
                      1, Calculate a big distance matrix, with all the training and testing samples
******************** included. Then split this matrix into a square matrix with the rows and columns 
********************  the training samples, and a matrix with all the rows testing samples and all the columns
********************  the training samples.
                      2, Calculate the design matrices for the above two distance matrices
                      3, Select centers and sub select the corresponding designmatrics from step 2
'''

'''
##################################################################################################
'''
'''
Denominator_factor: this is a small subfuntion of Test_MCC.
'''
def Denominator_factor(c_matrix):
    denominator_factor = 0
    nclass, _ = c_matrix.shape
    c_rowsum = np.sum(c_matrix, axis = 1, keepdims=True)
    for k in range(nclass):
        for kp in range(nclass):
            if kp != k:
                for lp in range(nclass):
                    denominator_factor += c_rowsum[k]*c_matrix[kp,lp]
    return denominator_factor

'''
Test_MCC:This function is to calculate the MCC. The binary classification should be considered as a 
        special case of the multiclass classification problem, with the number of class be 2.
'''
def Test_MCC(test_para):
    observed = test_para['observed']
    n_class = observed.shape[1]
    design_matrix_list = test_para['design_matrix_list']
    coefficients_list = test_para['coefficients_list']
    nCenters_list = test_para['nCenters_list']
    
    test_para['mcc_list'] = [] 
    test_para['c_matrix_list'] = []

    for ind, nCenters in enumerate(nCenters_list):
        coefficients = coefficients_list[ind]
        design_matrix = design_matrix_list[ind]
        # Set the shape of coefficients
        coefficients = coefficients.reshape((-1, nCenters)).T
        pred_logit = design_matrix.dot(coefficients)
        max_logit = np.max(pred_logit, axis = 1, keepdims=True)
        pred_bool = pred_logit==max_logit
        # Calculate the C matrix
        c_matrix = np.zeros((n_class, n_class))
        for i in range(n_class):
            for j in range(n_class):
                pred_i = observed[pred_bool[:, i] == True, :]
                # pred_i is a subset of observed with the predicted class i
                observe_j = pred_i[pred_i[:, j] == 1, :]
                #observe_j is the set of samples with predicted class i and observed class j
                c_matrix[i,j] = observe_j.shape[0]
        test_para['c_matrix_list'].append(copy.deepcopy(c_matrix))        
        # Calculate the MCC
        numerator = [0]
        for ka in range(n_class):
            for la in range(n_class):
                for ma in range(n_class):
                    numerator += c_matrix[ka, ka]*c_matrix[la, ma] - c_matrix[ka, la]*c_matrix[ma, ka]
                    
                    
        df1 = Denominator_factor(c_matrix)
        df2 = Denominator_factor(c_matrix.T)
        denominator = df1**(0.5)*df2**(0.5)
        if denominator == 0:
            denominator = 1
        # Calculate the MCC
#        print(numerator)
        mcc = numerator/denominator
        test_para['mcc_list'].append(mcc[0])
        
    return test_para
'''##################################################################################''' 
'''
Ceoff_RBFN: to train the model and make prediction by the our newly designed method.
Inputs: 
    train_para: as above, train_para['design_matrix'] should be the initial train_design_matrix
    test_design_matrix: the initial one with the same columns as train_para['design_matrix']
    observed_test: all should be in the multiclass form
    nCenters_list: as above
    reg: a float, a hyperparameter
Output:
    test_para
''' 

def Ceoff_RBFN(train_para, test_design_matrix, observed_test):    
    
    test_para = Center_Select_Coef(train_para, test_design_matrix)    

    test_para['observed'] = observed_test
    
    test_para= Test_MCC(test_para)
    
    return test_para
'''###########################################################################################'''
'''
Coverage_RBFN: this function is to train and test the model by using the meothod in the paper 
    'A Fast and Efficient Method for Training Categorical Radial Basis Function Networks'
Inputs:
    nCenters_list:as above
    train_distance_matrix: a square distance matrix, with both the rows and columns the training samples
    train_design_matrix:  both the rows and columns the training samples
    test_design_matrix: the initial one, with rows the testing samples and the columns training samples
    observed_train: in the multiclass form
    observed_test: in the multiclass form
    reg: a float, this is a hyperparameter
Output:
    test_para_coverage: the same as test_para
    
'''
    
def Coverage_RBFN(train_para, test_design_matrix, observed_test, train_distance_matrix):
    
    train_design_matrix = train_para['design_matrix']
    nCenters_list = train_para['nCenters_list']
    train_method = train_para['train_method']
    
    test_para_coverage = {}
    nCenters_list.sort(reverse=True)
    test_para_coverage['nCenters_list'] = nCenters_list
    test_para_coverage['observed'] = observed_test
    test_para_coverage['design_matrix_list'] = []
    test_para_coverage['coefficients_list'] = []

     
    eliminated, radius = CS_coverage_all(train_distance_matrix)
    for nCenters in nCenters_list:
        
        train_center_dm, test_center_dm = Design_matrix_coverage_by_nCenters\
                                        (train_design_matrix,test_design_matrix, eliminated, nCenters)
                                        
        test_para_coverage['design_matrix_list'].append(copy.deepcopy(test_center_dm))
        # Train the model
        train_para['design_matrix'] = train_center_dm
#        train_para['coefficients'] = np.random.randn(train_center_dm.shape[1]*train_para['observed'].shape[1], 1)
        train_para['coefficients'] = np.zeros((train_center_dm.shape[1]*train_para['observed'].shape[1], 1))
        '''Can start with 0 solve the nan problem?'''
        
        if train_method == 'BFGS':
            train_para, _  = Train_RBFN_BFGS(train_para, rho=0.8, c = 1e-3, termination = 1e-3)
        elif train_method == 'GD':
            train_para, _  = Train_GD(train_para)
            
        test_para_coverage['coefficients_list'].append(copy.deepcopy(train_para['coefficients']))

    test_para_coverage = Test_MCC(test_para_coverage)
    
    return test_para_coverage
''''#########################################################################################'''
'''
'''
def Cross_validation(data_dict, reg_list, nCenters_list, train_method):

    test_design_matrix = copy.deepcopy(data_dict['validation_design_matrix'])
    observed_test = copy.deepcopy(data_dict['observed_validation'])
    train_distance_matrix = copy.deepcopy(data_dict['train_distance_matrix'])
    train_design_matrix = copy.deepcopy(data_dict['train_design_matrix'])
    
    mcc_coeff = np.zeros((len(nCenters_list), len(reg_list)))
    mcc_coverage = np.zeros(mcc_coeff.shape)
    
    train_para = {}
    train_para['design_matrix'] = copy.deepcopy(data_dict['train_design_matrix'])
    train_para['observed'] = copy.deepcopy(data_dict['observed_train'])
#    train_para['coefficients'] = np.random.randn(train_design_matrix.shape[1]*train_para['observed'].shape[1], 1)
    train_para['coefficients'] = np.zeros((train_design_matrix.shape[1]*train_para['observed'].shape[1], 1))
    train_para['loss_type'] = 'Softmax'
    train_para['train_method'] = train_method
    train_para['step_size'] = 1e-6
    train_para['n_iterations'] = 10000
    nCenters_list.sort(reverse=True)
    train_para['nCenters_list'] = nCenters_list
    
    for i, reg in enumerate(reg_list):
        
        train_para['reg'] = reg
        train_para['design_matrix'] = copy.deepcopy(data_dict['train_design_matrix'])
#        train_para['coefficients'] = np.random.randn(train_design_matrix.shape[1]*train_para['observed'].shape[1], 1)
        train_para['coefficients'] = np.zeros((train_design_matrix.shape[1]*train_para['observed'].shape[1], 1))
        test_coeff = Ceoff_RBFN(train_para, test_design_matrix, observed_test)
        #renew
        train_para['design_matrix'] = copy.deepcopy(data_dict['train_design_matrix'])
#        train_para['coefficients'] = np.random.randn(train_design_matrix.shape[1]*train_para['observed'].shape[1], 1)
        train_para['coefficients'] = np.zeros((train_design_matrix.shape[1]*train_para['observed'].shape[1], 1))
        test_coverage = Coverage_RBFN(train_para, test_design_matrix, observed_test, train_distance_matrix)
        
        mcc_coeff[:,i] = np.array(test_coeff['mcc_list'])
        mcc_coverage[:,i] = np.array(test_coverage['mcc_list'])
    
    
    reg_coeff_list = np.array(reg_list)[np.argmax(mcc_coeff, axis = 1)]    
    reg_coverage_list = np.array(reg_list)[np.argmax(mcc_coverage, axis = 1)]  
    
    return reg_coeff_list, reg_coverage_list, mcc_coeff, mcc_coverage

# Load the data of breast cancer and do the data wrangling
def Wrangling_BC():
    os.chdir('/Users/ARAN/Downloads/CenterSelection/BreastCancer')
    data = pd.read_csv('breast-cancer-wisconsin.data')

    col_name = data.columns

    complete = data[:][data[col_name[6]] != '?']
    complete[col_name[6]] = complete[col_name[6]].astype(np.int64)
    
    attributes = list(col_name[1:10])
    categories = [col_name[-1]]
    
    return complete, attributes, categories



def Split_data(distance_matrix, design_matrix, observed_total, hamming_dist_m):
    
    n_total = distance_matrix.shape[0]
    n_train = math.floor(2*n_total/4)
    n_validation = math.floor(n_total/4)
    
    train_ind = list(range(n_train))
    validation_ind = list(range(n_train, n_train+n_validation))
    test_ind = list(range(n_train+n_validation, n_total))    
    train_distance_matrix = distance_matrix[train_ind, :][:, train_ind]
    train_design_matrix = design_matrix[train_ind, :][:, train_ind]
    validation_design_matrix = design_matrix[validation_ind,:][:, train_ind]
    test_design_matrix = design_matrix[test_ind, :][:, train_ind]    
    
    # Remove the duplicates
    train_hamming_dist_m = hamming_dist_m[train_ind, :][:, train_ind]    
    eliminated, radius = CS_coverage_all(train_hamming_dist_m)  
    to_be_del = []
    for ind, v in enumerate(radius):
        if v <= 0:
            to_be_del.append(eliminated[ind])        
    sub_distance_m = np.delete(train_distance_matrix, to_be_del, axis = 0)
    sub_train_distance_m = np.delete(sub_distance_m, to_be_del, axis = 1)    
    sub_train_design_m = np.delete(train_design_matrix, to_be_del, axis=1)
    sub_validation_design_m= np.delete(validation_design_matrix, to_be_del, axis=1)
    sub_test_design_m = np.delete(test_design_matrix, to_be_del, axis=1)

    
    observed_train = observed_total[train_ind, :]
    observed_validation = observed_total[validation_ind,:]
    observed_test = observed_total[test_ind, :]
    
    # Pack the data up into a dictionary
    data_dict = {}
    data_dict['observed_total'] = observed_total
    data_dict['distance_matrix'] = distance_matrix
    data_dict['design_matrix'] = design_matrix
    data_dict['train_distance_matrix'] = sub_train_distance_m
    data_dict['train_design_matrix'] = sub_train_design_m
    data_dict['validation_design_matrix'] = sub_validation_design_m
    data_dict['test_design_matrix'] = sub_test_design_m
    data_dict['observed_train'] = observed_train
    data_dict['observed_validation'] = observed_validation
    data_dict['observed_test'] = observed_test
    
    return data_dict


def Validate_and_Test(data_dict, reg_list, nCenters_list, train_method = 'BFGS'):

    reg_coeff_list, reg_coverage_list, mcc_coeff, mcc_coverage = Cross_validation(data_dict, reg_list, nCenters_list, train_method)
    #data_dict.keys()
    mcc_coeff_list = []; mcc_coverage_list = []
    nCenters_list.sort(reverse=True)
    for i, nCenters in enumerate(nCenters_list): 
   
        reg_coeff = reg_coeff_list[i]
        reg_coverage = reg_coverage_list[i]
        

        test_design_matrix = copy.deepcopy(data_dict['test_design_matrix'])
        observed_train = copy.deepcopy(data_dict['observed_train'])
        observed_test = copy.deepcopy(data_dict['observed_test'])
        observed_validation = copy.deepcopy(data_dict['observed_validation'])
        train_distance_matrix = copy.deepcopy(data_dict['train_distance_matrix'])
        train_design_matrix = copy.deepcopy(data_dict['train_design_matrix'])
        validation_design_matrix = copy.deepcopy(data_dict['validation_design_matrix'])
        # combine the validation and the train
        train_design_matrix_com = np.vstack((train_design_matrix, validation_design_matrix))
        observed_train_com = np.vstack((observed_train, observed_validation))
        
        train_para = {}
        train_para['design_matrix'] = copy.deepcopy(train_design_matrix_com)
        train_para['observed'] = copy.deepcopy(observed_train_com)
        train_para['reg'] = reg_coeff
#        train_para['coefficients'] = np.random.randn(train_design_matrix.shape[1]*train_para['observed'].shape[1], 1)
        train_para['coefficients'] = np.zeros((train_design_matrix.shape[1]*train_para['observed'].shape[1], 1))
        train_para['loss_type'] = 'Softmax'
        train_para['train_method'] = train_method
        train_para['step_size'] = 1e-6
        train_para['n_iterations'] = 10000
        train_para['nCenters_list'] = [nCenters]
        
        #reg = 0.1
        test_coeff = Ceoff_RBFN(train_para, test_design_matrix, observed_test)
        
        # Renew some of the values in train_para
        train_para['design_matrix'] = copy.deepcopy(train_design_matrix_com)
        train_para['reg'] = reg_coverage
#        train_para['coefficients'] = np.random.randn(train_design_matrix.shape[1]*train_para['observed'].shape[1], 1)
        train_para['coefficients'] = np.zeros((train_design_matrix.shape[1]*train_para['observed'].shape[1], 1))
        
        test_coverage = Coverage_RBFN(train_para, test_design_matrix, observed_test, train_distance_matrix)
        
        mcc_coeff_list.append(copy.deepcopy(test_coeff['mcc_list'][0]))
        mcc_coverage_list.append(copy.deepcopy(test_coverage['mcc_list'][0]))
        
    return mcc_coeff_list, mcc_coverage_list


'''#########################################################################'''
'''************************PROCESSING THE DATA******************************'''
def BRE():
    complete, attributes, categories = Wrangling_BC()
    SS = Distance_martix(complete, attributes, categories)
    SS.Hamming_matrix()
    SS.IOF_matrix()
    SS.OF_matrix()
    #SS.Burnaby_matrix()
    SS.Eskin_matrix()# If the distance were not of great contrast. We multiply by a constant
    #SS.Lin_matrix()
    #
    observed = SS.cate[['2.1']]
    observed[observed['2.1'] == 2] = 0
    observed[observed['2.1'] == 4] = 1
    observed_total = np.float64(np.hstack((observed.values, 1 - observed.values)))
    
    dist_type_list = ['Hamming', 'IOF', 'OF', 'Eskin']
    RBF_list = ['Gaussian', 'Markov', 'Thin_Plate_Spline', 'Inverse_Multi_Quadric']
    
    distance = copy.deepcopy(SS.hamming_matrix)
    data_dict = Split_data(distance, distance, observed_total, SS.hamming_matrix) 
    # Generate the nCenters_list
    nCenters_list = []
    nCenters_total = data_dict['train_design_matrix'].shape[1]
    nCenters = math.floor(nCenters_total/2)
    while nCenters > 2:
        nCenters = math.floor(nCenters/2)
        nCenters_list.append(nCenters)
    nCenters_list            
    BRE_results = {}
    
    for dist_type in dist_type_list:
        for RBF in RBF_list:
            if dist_type == 'Hamming':
                distance_matrix = SS.hamming_matrix
            elif dist_type == 'IOF':
                distance_matrix = SS.iof_matrix
            elif dist_type == 'OF':
                distance_matrix =SS.of_matrix
            elif dist_type == 'Eskin':
                distance_matrix = SS.eskin_matrix
            # Amplify the distance            
            train_size = math.floor(distance_matrix.shape[0])
            mean = np.average(np.abs(distance_matrix[:train_size,:train_size]))
            amplified_dist_matrix = distance_matrix / mean
            # Calculate the design matrix
            design_matrix = Design_matrix(amplified_dist_matrix, RBF)         
            # Load the data_dict
            data_dict = Split_data(distance_matrix, design_matrix, observed_total, SS.hamming_matrix) 
            
             # Set the reg list
            reg_list = [0.01, 0.05, 0.1, 0.5, 1]
    #        
            nIter = 5
            mcc_coeff_matrix = np.zeros((len(nCenters_list), nIter))
            mcc_coverage_matrix = np.zeros((len(nCenters_list), nIter))
            for i in range(nIter):
                mcc_coeff_list, mcc_coverage_list = Validate_and_Test(data_dict, reg_list, nCenters_list, train_method = 'BFGS')
                mcc_coeff_matrix[:, i] = copy.deepcopy(mcc_coeff_list)
                mcc_coverage_matrix[:,i] = copy.deepcopy(mcc_coverage_list)
                # Monitor the process
                key = dist_type+'_'+RBF
                print(key)
            BRE_results[dist_type+'_'+RBF] = copy.deepcopy((mcc_coeff_matrix, mcc_coverage_matrix))
            
            # Save the file from time to time
#            os.chdir('/home/leo/Documents/Project_SelectCenters/Code/Results')
            results_frame = pd.DataFrame(BRE_results)
#            results_frame.to_pickle('BRE_results_frame.pkl')
            
    return results_frame
results_frame = BRE()
#os.chdir('/home/leo/Documents/Project_SelectCenters/Code/Results')        
#BRC_results = pd.read_pickle('BRE_results_frame.pkl')        
#keys = BRC_results.keys()
#len(keys)
#keys
#
#i = 16
#BRC_results[keys[i]][0]
#BRC_results[keys[i]][1]
#nCenters_list
#BRC_results['nCenters_reg_lists'] = [nCenters_list, reg_list]
#BRC_results.to_pickle('BRE_results_frame.pkl')   

