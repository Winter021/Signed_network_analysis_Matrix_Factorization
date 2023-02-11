'''Preprocess Epinions and Slashdot data'''
'''Take in text file of data, create and save adjacency matrix'''

import numpy as np, pickle
import scipy.sparse as sp

#Preprocess data
#Optionally run normally or in test mode (when writing tests)

"""

def preprocess(batch_size = 1000, mode = "normal"):
  FILE_PATH = "Raw Data/soc-sign-evomain.txt"
  #FILE_PATH = "Raw Data/soc-sign-epinions.txt"
  #FILE_PATH = "Raw Data/soc-sign-Slashdot090221.txt"

  #Dataset name (for filename of matrix)
  #Split off part right before file extension
  dataset = FILE_PATH.split(".txt")[0].split("-")[-1]

  with open(FILE_PATH, "rb") as data_file:
    data_lines = data_file.readlines()

    #initialize the data matrices
    from_data = list()
    to_data = list()
    labels = list()
    max_id = 0

    #Data format: each line FROM_ID TO_ID LABEL
    for line_index in range(4, len(data_lines), batch_size):
      batch_end = min(line_index + batch_size, len(data_lines))

      for i in range(line_index, batch_end):
        data = data_lines[i].split()
        from_data.append(int(data[0]))
        to_data.append(int(data[1]))
        labels.append(int(data[2]))
        max_id = max(max_id, int(data[0]), int(data[1]))

      #Make a (square) adjacency matrix the size of the number of people 
      #(as given by ID. note: ID starts at 0)

      #Create in sparse row-major format
      #Note: ID starts at 0 so number of people is 1 more than max ID
      data_matrix = sp.csr_matrix((np.array(labels), (np.array(from_data), 
                    np.array(to_data)) ), shape=(max_id + 1, max_id + 1))

      #correction to make data matrix symmetric
      if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
        data_matrix = (data_matrix + data_matrix.transpose()).sign()

      #test data is a valid symmetric signed matrix
      if mode == "test":
        assert data_matrix.min() == -1
        assert data_matrix.max() == 1
        assert (data_matrix != data_matrix.transpose()).nnz == 0

      #Save data
      np.save("Preprocessed Data/" + dataset + "_csr_" + str(line_index), data_matrix)

      #clear the data matrices for the next iteration
      from_data.clear()
      to_data.clear()
      labels.clear()

"""


def preprocess(batch_size = 20000, mode = "normal"):
  FILE_PATH = "Raw Data/soc-sign-evomain.txt"
  #FILE_PATH = "Raw Data/soc-evomain.txt"
  #FILE_PATH = "Raw Data/soc-sign-Slashdot090221.txt"

  #Dataset name (for filename of matrix)
  #Split off part right before file extension
  dataset = FILE_PATH.split(".txt")[0].split("-")[-1]

  with open(FILE_PATH, "rb") as data_file:
    data_lines = data_file.readlines()
    #Save components of data in three lists kept in synchrony
    from_data = list()
    to_data = list()
    labels = list()

    #Data format: each line FROM_ID TO_ID LABEL
    for line_index in range(4, len(data_lines)): #skip first 4 boilerplate lines
      data = data_lines[line_index].split()
      if (int(data[0])<30000) and (int(data[1])<30000):
          from_data.append(int(data[0]))
          to_data.append(int(data[1]))
          labels.append(int(data[2]))
    #Get the number of people (as given by ID. note: ID starts at 0)
    max_id = len(from_data)
    # Use a sliding window approach to handle large data size
    for i in range(0, max_id + 1, batch_size):
        start = i
        end = min(i + batch_size, max_id + 1)

        # Create a (square) adjacency matrix for the current batch
        batch_from_data = [x for j, (x, y) in enumerate(zip(from_data, to_data)) if start <= j < end]
        batch_to_data = [y for j, (x, y) in enumerate(zip(from_data, to_data)) if start <= j < end]
        batch_labels = [label for j, (x, y, label) in enumerate(zip(from_data, to_data, labels)) if start <= j < end]
        max_id = max(max(batch_from_data), max(batch_to_data))

        print(max(batch_to_data))
        print(len(batch_labels))
        data_matrix = sp.csr_matrix((np.array(batch_labels), (np.array(batch_from_data), np.array(batch_to_data)) ),
                                       shape=(max_id + 1, max_id + 1), dtype=np.int8)

        # Correction to make data matrix symmetric
        if (data_matrix != data_matrix.transpose()).nnz > 0:  # data matrix is not symmetric
          data_matrix = (data_matrix + data_matrix.transpose()).sign()

        #test data is a valid symmetric signed matrix
        if mode == "test":
          assert data_matrix.min() == -1
          assert data_matrix.max() == 1
          assert (data_matrix != data_matrix.transpose()).nnz == 0

        # Save the matrix in batch
        np.save("Preprocessed Data/" + dataset + "_batch_" + str(i), data_matrix)



"""
def preprocess(batch_size = 1000, mode = "normal"):
  FILE_PATH = "Raw Data/soc-sign-evomain.txt"
  #FILE_PATH = "Raw Data/soc-evomain.txt"
  #FILE_PATH = "Raw Data/soc-sign-Slashdot090221.txt"

  #Dataset name (for filename of matrix)
  #Split off part right before file extension
  dataset = FILE_PATH.split(".txt")[0].split("-")[-1]

  with open(FILE_PATH, "rb") as data_file:
    data_lines = data_file.readlines()

    #Save components of data in three lists kept in synchrony
    from_data = list()
    to_data = list()
    labels = list()

    #Data format: each line FROM_ID TO_ID LABEL
    for line_index in range(4, len(data_lines)): #skip first 4 boilerplate lines
      data = data_lines[line_index].split()
      from_data.append(int(data[0]))
      to_data.append(int(data[1]))
      labels.append(int(data[2]))

    #Make a (square) adjacency matrix the size of the number of people 
    #(as given by ID. note: ID starts at 0)

    max_id = max(max(from_data), max(to_data)) #aka number of people

    #Create in sparse row-major format
    #Note: ID starts at 0 so number of people is 1 more than max ID
    data_matrix = sp.csr_matrix((np.array(labels), (np.array(from_data), 
                  np.array(to_data)) ), shape=(max_id + 1, max_id + 1))

    #correction to make data matrix symmetric
    if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
      data_matrix = (data_matrix + data_matrix.transpose()).sign()

    #test data is a valid symmetric signed matrix
    if mode == "test":
      assert data_matrix.min() == -1
      assert data_matrix.max() == 1
      assert (data_matrix != data_matrix.transpose()).nnz == 0

    # Save the matrix in batches
    num_batches = int(max_id / batch_size) + 1
    print(num_batches)
    for i in range(num_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, max_id + 1)
        batch = data_matrix[start:end, start:end]
        np.save("Preprocessed Data/" + dataset + "_batch_" + str(i), batch)

"""

if __name__ == "__main__":
  preprocess()



"""
def preprocess(mode = "normal"):
  #FILE_PATH = "Raw Data/soc-sign-evomain.txt"
  FILE_PATH = "Raw Data/soc-sign-epinions.txt"
  #FILE_PATH = "Raw Data/soc-sign-Slashdot090221.txt"

  #Dataset name (for filename of matrix)
  #Split off part right before file extension
  dataset = FILE_PATH.split(".txt")[0].split("-")[-1]

  with open(FILE_PATH, "rb") as data_file:
    data_lines = data_file.readlines()

    #Save components of data in three lists kept in synchrony
    from_data = list()
    to_data = list()
    labels = list()

    #Data format: each line FROM_ID TO_ID LABEL
    for line_index in range(4, len(data_lines)): #skip first 4 boilerplate lines
      data = data_lines[line_index].split()
      from_data.append(int(data[0]))
      to_data.append(int(data[1]))
      labels.append(int(data[2]))

    #Make a (square) adjacency matrix the size of the number of people 
    #(as given by ID. note: ID starts at 0)

    max_id = max(max(from_data), max(to_data)) #aka number of people

    #Create in sparse row-major format
    #Note: ID starts at 0 so number of people is 1 more than max ID
    data_matrix = sp.csr_matrix((np.array(labels), (np.array(from_data), 
                  np.array(to_data)) ), shape=(max_id + 1, max_id + 1))

    #correction to make data matrix symmetric
    if (data_matrix != data_matrix.transpose()).nnz > 0: #data matrix is not symmetric
      data_matrix = (data_matrix + data_matrix.transpose()).sign()

    #test data is a valid symmetric signed matrix
    if mode == "test":
      assert data_matrix.min() == -1
      assert data_matrix.max() == 1
      assert (data_matrix != data_matrix.transpose()).nnz == 0

  #Save data
  np.save("Preprocessed Data/" + dataset + "_csr", data_matrix)
  np.save("Preprocessed Data/small_network", data_matrix[:100,:100])
"""

