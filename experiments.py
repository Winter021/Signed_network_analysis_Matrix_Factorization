#Perform experiments
#Select network
#Select methods
#Study performance of all methods on the network

# import numpy as np 
import numpy as np
# save np.load
np_load_old = np.load

# modify the default parameters of np.load
np.load = lambda *a,**k: np_load_old(*a, allow_pickle=True, **k)

import data.simulate_networks as sim
import local_prediction.local_moi as moi 
import local_prediction.hoc_prediction as hoc
import matrix_completion.matrixprediction as mf
# import matrix_completion.matrix_prediction as mf
# from .matrix_completion import matrix_prediction as mf 

import cluster.clustering as clust
import analytics.stats as stats

import time

import networkx as nx
import re
from data.preprocess_ES import preprocess

# Create an empty graph


#Run experiments on different algorithms on the same network

"""
def run_experiment():
  G = nx.Graph()
  simulated = False
  real = True

  use_moi = False #True
  use_hoc = False #True
  use_svp = True
  use_sgd_sh = False 
  use_sgd_sig = False
  use_als = False #True

  adj_matrix = None
  if simulated:
    cluster_sizes = [100,200,300,400]
    sparsity_level = 0.01175
    noise_prob = 0
    print ("creating adjacency matrix...")
    adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)

  elif real:
    data_file_name = "data/Preprocessed Data/small_network.npy"
    # data_file_name = "data/Preprocessed Data/epinions_csr.txt"
    #data_file_name = "data/Preprocessed Data/wiki_elections_csr.npy"
    batch_size = 10000
    
    adj_matrix_list = []
    for data_batch in preprocess(data_file_name, batch_size):
      # convert each data batch to an adjacency matrix
      adj_matrix = nx.adjacency_matrix(data_batch).todense()
      adj_matrix_list.append(adj_matrix)
    adj_matrix = np.concatenate(adj_matrix_list, axis=0)


  if use_moi:
    print ("performing MOI...")
    max_cycle_order_moi = 10
    discount = [0.5**i for i in range(3, max_cycle_order_moi + 1)]
    #max_cycle_order_moi = np.inf
    #discount = 0.0001
    num_folds = 5
    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
          moi.kfoldcv_moi(adj_matrix, discount, max_cycle_order_moi, num_folds)
    print ("MOI results: ")
    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
    print

  if use_hoc:
    print ("performing HOC...")
    max_cycle_order_hoc = 5
    num_folds = 10
    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
              hoc.hoc_learning_pipeline(adj_matrix, max_cycle_order_hoc)
    print ("HOC results:")
    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
    print

  alg = ""
  alg_params = None

  #settings if using SGD
  if use_sgd_sh or use_sgd_sig:
    #Parameters used for this experiment

    #https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
    learning_rate = 1000#0.05 for square hinge
    tol = adj_matrix.nnz/10
    max_iter = 20
    reg_param = 10#0.5 for square hinge
    dim = 100
    num_folds_mf = 10

    #Bundle up these parameters and use this algorithm
    if use_sgd_sh:
      loss_type = "squarehinge" #"sigmoid"
      alg_params = (learning_rate, loss_type, tol, max_iter, reg_param, dim)
      alg = "sgd"
      
      print ("performing SGD with square-hinge loss...")
      avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
              mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
      print ("SGD_SH results:")
      print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
      print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
      print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
      print
    if use_sgd_sig:
      loss_type = "sigmoid"
      alg_params = (learning_rate, loss_type, tol, max_iter, reg_param, dim)
      alg = "sgd"
      
      print ("performing SGD with sigmoid loss...")
      avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
              mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
      print ("SGD_SIG results:")
      print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
      print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
      print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
      print
  #settings if using als debug
  if use_als:
    #Parameters used for this experiment
    max_iter = 2
    dim = 40

    #Bundle up these parameters and use this algorithm
    alg_params = (max_iter, dim)
    alg = "als"

    num_folds_mf = 10

    print ("performing ALS...")
    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
            mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
    print ("ALS results:")
    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
    print

  # settings if using SVP debug
  if use_svp:
    #Parameters used for this experiment
    rank = 40
    tol = 100
    max_iter = 5
    step_size = 1

    #Bundle up these parameters and use this algorithm
    alg_params = (rank, tol, max_iter, step_size)
    alg = "svp"

    num_folds_mf = 10
    
    print ("performing SVP...")
    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
          mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
    print ("SVP results:")
    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
    print

"""





import os

def run_experiment():
    G = nx.Graph()
    simulated = False
    real = True

    use_moi = False
    use_hoc = False
    use_svp = False
    use_sgd_sh = True
    use_sgd_sig = False
    use_als = False

    adj_matrix = None
    if simulated:
        cluster_sizes = [100,200,300,400]
        sparsity_level = 0.01175
        noise_prob = 0
        print ("creating adjacency matrix...")
        adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)

    elif real:
        # data_file_name = "data/Preprocessed Data/epinions_csr.txt"
        # data_file_name = "data/Preprocessed Data/small_network.npy"
        # data_file_name = "data/Preprocessed Data/wiki_elections_csr.npy"
        data_folder = "data/Preprocessed Data/"
        results = []
        for data_file_name in os.listdir(data_folder):
            if data_file_name.endswith(".npy"):
                try:
                    adj_matrix = np.load(os.path.join(data_folder, data_file_name)).item()
                except Exception as e:
                    raise ValueError("could not load adj matrix from file: ", e)
                print(adj_matrix.get_shape())
                if use_svp:
                    # Parameters used for this experiment
                    rank = 40
                    tol = 100
                    max_iter = 5
                    step_size = 1

                    # Bundle up these parameters and use this algorithm
                    alg_params = (rank, tol, max_iter, step_size)
                    alg = "svp"

                    num_folds_mf = 10

                    print(f"performing SVP on {data_file_name}...")
                    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
                        mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
                    print(f"SVP results on {data_file_name}:")
                    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
                    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
                    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
                    results.append((data_file_name, avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time))
                if use_sgd_sh or use_sgd_sig:
                    #Parameters used for this experiment

                    #https://www.cs.uic.edu/~liub/KDD-cup-2007/proceedings/Regular-Paterek.pdf
                    learning_rate = 1000#0.05 for square hinge
                    tol = adj_matrix.nnz/10
                    max_iter = 20
                    reg_param = 10#0.5 for square hinge
                    dim = 100
                    num_folds_mf = 10

                    #Bundle up these parameters and use this algorithm
                    if use_sgd_sh:
                      loss_type = "squarehinge" #"sigmoid"
                      alg_params = (learning_rate, loss_type, tol, max_iter, reg_param, dim)
                      alg = "sgd"
                      
                      print ("performing SGD with square-hinge loss...")
                      avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
                              mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
                      print ("SGD_SH results:")
                      print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
                      print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
                      print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
                      results.append((data_file_name, avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time))
                    if use_sgd_sig:
                      loss_type = "sigmoid"
                      alg_params = (learning_rate, loss_type, tol, max_iter, reg_param, dim)
                      alg = "sgd"
                      
                      print ("performing SGD with sigmoid loss...")
                      avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
                              mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
                      print ("SGD_SIG results:")
                      print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
                      print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
                      print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
                      results.append((data_file_name, avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time))
        return results


if __name__ == "__main__":
  run_experiment()





"""


def run_experiment():
  G = nx.Graph()
  simulated = False
  real = True

  use_moi = False #True
  use_hoc = False #True
  use_svp = True
  use_sgd_sh = False 
  use_sgd_sig = False
  use_als = False #True

  adj_matrix = None
  if simulated:
    cluster_sizes = [100,200,300,400]
    sparsity_level = 0.01175
    noise_prob = 0
    print ("creating adjacency matrix...")
    adj_matrix = sim.sample_network(cluster_sizes, sparsity_level, noise_prob)

  elif real:
    # data_file_name = "data/Preprocessed Data/epinions_csr.txt"
    data_file_name = "data/Preprocessed Data/small_network.npy"
    #data_file_name = "data/Preprocessed Data/wiki_elections_csr.npy"
    try:
      # with open(data_file_name, 'r') as f:
      #   for line in f:
      #       u, v, sign = re.findall(r'\S+', line)
      #       if sign == "1":
      #           G.add_edge(u, v)
      #       else:
      #           G.add_edge(u, v, sign='-')
      adj_matrix = np.load(data_file_name).item()
      # adj_matrix = nx.adjacency_matrix(G)
      # adj_matrix = adj_matrix.todense()
    except Exception as e:
      raise ValueError("could not load adj matrix from file: ", e)

  if use_svp:
    #Parameters used for this experiment
    rank = 40
    tol = 100
    max_iter = 5
    step_size = 1

    #Bundle up these parameters and use this algorithm
    alg_params = (rank, tol, max_iter, step_size)
    alg = "svp"

    num_folds_mf = 10
    
    print ("performing SVP...")
    avg_acc, stderr_acc, avg_fpr, stderr_fpr, avg_time, stderr_time = \
          mf.kfold_CV_pipeline(adj_matrix, alg, alg_params, num_folds_mf)
    print ("SVP results:")
    print("Accuracy: average %f with standard error %f" % (avg_acc, stderr_acc))
    print("False positive rate: average %f with standard error %f" % (avg_fpr, stderr_fpr))
    print("Model running time: average %f with standard error %f" % (avg_time, stderr_time))
    print



"""