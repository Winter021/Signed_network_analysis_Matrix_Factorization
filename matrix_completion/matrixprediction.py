#Prediction framework for global methods
#based on matrix factorization and clustering

import matrix_completion.svpsignprediction as svp
import matrix_completion.matrixfactorization as mf
import logging, time

import numpy as np
import scipy.sparse as sp
import utils.ml_pipeline as pipeline
import analytics.stats as stats



#k fold cross validation for matrix completion problems
# Input: adjacency matrix
# Algorithm to use
# Parameters for that algorithm (loss function, learning rate, etc.)
# (for more details see individual algorithm requirements in matrix_factorization.py)
# Number of folds for cross-valudation

from sklearn.model_selection import KFold

"""
def kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds=10):
    dense_adj_matrix = adj_matrix.toarray()
    nonzero_row_indices, nonzero_col_indices = np.where(dense_adj_matrix != 0)
    data = list(zip(nonzero_row_indices, nonzero_col_indices))
    folds = pipeline.kfold_CV_split(data, num_folds)

    accuracy_fold_data = list()
    false_positive_rate_fold_data = list()
    time_fold_data = list()

    for fold_index in range(num_folds):
        train_points = pipeline.join_folds(folds, fold_index)
        test_points = folds[fold_index]
        test_row_indices, test_col_indices = zip(*test_points)
        test_labels = adj_matrix[test_row_indices, test_col_indices].A[0]
        
        for i in range(len(train_points)):
            train_point = train_points[i]
            train_row_indices, train_col_indices = zip(*train_points)
            train_labels = adj_matrix[train_row_indices, train_col_indices].A[0]
            train_matrix = sp.csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape=adj_matrix.shape)
            
            before_train = time.time()
            train_complet = matrix_completion(train_matrix, alg, alg_params)
            after_train = time.time()
            model_time = after_train - before_train
            
            preds = train_complet[test_row_indices, test_col_indices]
            acc, fpr = pipeline.evaluate(preds, test_labels)
            print(len(preds))
            print("Fold %d, Train point %d: accuracy=%f, computation time=%f" % (fold_index, i, acc, model_time))
            
            accuracy_fold_data.append(acc)
            false_positive_rate_fold_data.append(fpr)
            time_fold_data.append(model_time)
    avg_acc = sum(accuracy_fold_data) / float(len(accuracy_fold_data))
    avg_fpr = sum(false_positive_rate_fold_data) / float(len(false_positive_rate_fold_data))
    avg_time = sum(time_fold_data) / float(len(time_fold_data))
    acc_stderr = stats.error_width(stats.sample_std(accuracy_fold_data), num_folds)
    fpr_stderr = stats.error_width(stats.sample_std(false_positive_rate_fold_data), num_folds)
    time_stderr = stats.error_width(stats.sample_std(time_fold_data), num_folds)
    return avg_acc, acc_stderr, avg_fpr, fpr_stderr, avg_time, time_stderr
"""

def kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds=10):
    dense_adj_matrix = adj_matrix.toarray()
    nonzero_row_indices, nonzero_col_indices = np.where(dense_adj_matrix != 0)
    if len(nonzero_row_indices) != len(nonzero_col_indices):
        raise ValueError("The number of non-zero row indices and non-zero column indices do not match.")
    data = list(zip(nonzero_row_indices, nonzero_col_indices))
    folds = pipeline.kfold_CV_split(data, num_folds)

    accuracy_fold_data = list()
    false_positive_rate_fold_data = list()
    time_fold_data = list()
    for fold_index in range(num_folds):
        print("Fold %d" % (fold_index + 1))
        train_points = pipeline.join_folds(folds, fold_index)
        test_points = folds[fold_index]
        test_row_indices, test_col_indices = zip(*test_points)
        test_labels = adj_matrix[test_row_indices, test_col_indices].A[0]
        train_matrix = sp.csr_matrix(([], ([], [])), shape = adj_matrix.shape)
        print(len(train_points))
        for i in range(1, len(train_points)+1):
            train_points_subset = train_points[:i]
            train_row_indices, train_col_indices = zip(*train_points_subset)
            train_labels = adj_matrix[train_row_indices, train_col_indices].A[0]
            new_train_matrix = sp.csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)
            train_matrix = train_matrix + new_train_matrix
            before_train = time.time()
            train_complet = matrix_completion(train_matrix, alg, alg_params)
            after_train = time.time()
            model_time = after_train - before_train
            
            preds = train_complet[test_row_indices, test_col_indices]
            acc, fpr = pipeline.evaluate(preds, test_labels)
            
            print("Fold %d, Iteration %d, Time: %f" % (fold_index + 1, i, model_time))

            accuracy_fold_data.append(acc)
            false_positive_rate_fold_data.append(fpr)
            time_fold_data.append(model_time)

    avg_acc = sum(accuracy_fold_data) / float(len(accuracy_fold_data))
    avg_fpr = sum(false_positive_rate_fold_data) / float(len(false_positive_rate_fold_data))
    avg_time = sum(time_fold_data) / float(len(time_fold_data))
    acc_stderr = stats.error_width(stats.sample_std(accuracy_fold_data), num_folds)
    fpr_stderr = stats.error_width(stats.sample_std(false_positive_rate_fold_data), num_folds)
    time_stderr = stats.error_width(stats.sample_std(time_fold_data), num_folds)

    return avg_acc, avg_fpr, avg_time, fpr_stderr, avg_time, time_stderr






"""
    

def kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds=10):
  #get folds
  # nonzero_row_indices, nonzero_col_indices = np.nonzero(adj_matrix) # .nonzero() is depreciated
  
  dense_adj_matrix = adj_matrix.toarray()
  nonzero_row_indices, nonzero_col_indices = np.where(dense_adj_matrix != 0)

  
  # print("nonzero row indices",len(nonzero_row_indices))
  # print("nonzero col indices",len(nonzero_col_indices))


  if len(nonzero_row_indices) != len(nonzero_col_indices):
    raise ValueError("The number of non-zero row indices and non-zero column indices do not match.")

  data = zip(nonzero_row_indices, nonzero_col_indices) #TODO maybe should try to keep arrays separate?
  # print("data",data) # zip object at some location


  labels = adj_matrix[nonzero_row_indices, nonzero_col_indices]
  print("labels : ",labels)
  # print("labels : ",labels)
  
  data = list(zip(nonzero_row_indices, nonzero_col_indices))
  folds = pipeline.kfold_CV_split(data, num_folds)
  # folds = pipeline.kfold_CV_split(nonzero_row_indices,nonzero_col_indices, num_folds) #debug

  print("folds ",folds)#all folds are blank 
  print ("got folds")

  #keep track of accuracy, false positive rate
  accuracy_fold_data = list()
  false_positive_rate_fold_data = list()
  time_fold_data = list()

  #perform learning problem on each fold
  for fold_index in range(num_folds):
    
    print("Fold %d" % (fold_index + 1))
    #get train data for learning problem
    
    train_points = pipeline.join_folds(folds, fold_index)
   
    # print("train point lengt : ",len(train_points)) #debug


    train_row_indices, train_col_indices = zip(*train_points)
    
    train_labels = adj_matrix[train_row_indices, train_col_indices].A[0] #array of signs of training edges
    #construct matrix using just training edges
    train_matrix = sp.csr_matrix((train_labels, (train_row_indices, train_col_indices)), shape = adj_matrix.shape)

    #get test data
    test_points = folds[fold_index]
    test_row_indices, test_col_indices = zip(*test_points)
    test_labels = adj_matrix[test_row_indices, test_col_indices].A[0] #array of signs of test edges

    #perform learning on training matrix
    before_train = time.time()
    train_complet = matrix_completion(adj_matrix, alg, alg_params)
    after_train = time.time()
    model_time = after_train - before_train

    #WRITETEST to make sure this is same shape as adj matrix
    print (train_complet.shape)

    preds = train_complet[test_row_indices, test_col_indices]

    acc, fpr = pipeline.evaluate(preds, test_labels)
    accuracy_fold_data.append(acc)
    false_positive_rate_fold_data.append(fpr)
    time_fold_data.append(model_time)

  avg_acc = sum(accuracy_fold_data) / float(len(accuracy_fold_data))
  avg_fpr = sum(false_positive_rate_fold_data) / float(len(false_positive_rate_fold_data))
  avg_time = sum(time_fold_data) / float(len(time_fold_data))
  acc_stderr = stats.error_width(stats.sample_std(accuracy_fold_data), num_folds)
  fpr_stderr = stats.error_width(stats.sample_std(false_positive_rate_fold_data), num_folds)
  time_stderr = stats.error_width(stats.sample_std(time_fold_data), num_folds)
  return avg_acc, acc_stderr, avg_fpr, fpr_stderr, avg_time, time_stderr

"""

#Matrix completion with matrix factorization
#Input: matrix to complete
#       Algorithm (SVP, SGD, or ALS)
#       Tuple of params other than matrix for each algorithm
#         (see relevant methods for details)
#Output: completed matrix
def matrix_completion(matrix, alg, params):
  alg = alg.lower()
  completed_matrix = None
  if alg == "svp":
    try:
      rank, tol, max_iter, step_size = params
      completed_matrix = svp.sign_prediction_SVP(matrix, rank, tol, max_iter, step_size)
    except:
      logging.exception("Exception: ")
      raise ValueError("invalid number or type of input for SVP?")
  elif alg == "sgd":
    try:
      learn_rate, loss_type, tol, max_iter, reg_param, dim = params
      factor1, factor2 = mf.matrix_factor_SGD(matrix, learn_rate, loss_type, tol, max_iter, reg_param, dim)
      completed_matrix = sp.csr_matrix.sign(sp.csr_matrix(factor1*factor2.transpose()))
    except:
      logging.exception("Exception: ")
      raise ValueError("invalid number or type of input for SGD?")
  elif alg == "als":
    try:
      max_iter, dim = params
      factor1, factor2 = mf.matrix_factor_ALS(matrix, dim, max_iter)
      completed_matrix = sp.csr_matrix.sign(sp.csr_matrix(factor1.transpose()*factor2))
    except:
      logging.exception("Exception: ")
      raise ValueError("invalid number or type of input for ALS?")
  else:
    raise ValueError("unrecognized matrix completion algorithm: ", alg)
  return completed_matrix