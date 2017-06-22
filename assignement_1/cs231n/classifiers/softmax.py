import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
  """
  Softmax loss function, naive implementation (with loops)

  Inputs have dimension D, there are C classes, and we operate on minibatches
  of N examples.

  Inputs:
  - W: A numpy array of shape (D, C) containing weights.
  - X: A numpy array of shape (N, D) containing a minibatch of data.
  - y: A numpy array of shape (N,) containing training labels; y[i] = c means
    that X[i] has label c, where 0 <= c < C.
  - reg: (float) regularization strength

  Returns a tuple of:
  - loss as single float
  - gradient with respect to weights W; an array of same shape as W
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################

  scores = X @ W   # (N, C) shape , N - number of elem in batch, C number of class
  max_score = np.max(scores, axis=1)  # max scores for each elem in batch
  # defining the constant used to limit instability
  C = - max_score
  score_stable = scores - C[:, None]

  loss = 0
  nb_elem_in_training_set = y.shape[0]
  nb_class = W.shape[1]
  for i in xrange(nb_elem_in_training_set):
      v = np.exp(score_stable[i, y[i]]) / (np.sum(np.exp(score_stable[i, :])))
      Li = -np.log(v)
      loss += Li

      for j in range(nb_class):
        dW[:, j] += X[i] * np.exp(score_stable[i, j]) / np.sum(np.exp(score_stable[i, :]), axis=0)
        if j == y[i]:
            dW[:, j] -= X[i]


  loss /= y.shape[0]
  dW = (1.0 / nb_elem_in_training_set)  * dW
  # regularization term
  loss += reg * np.sum(W**2)
  dW += reg * W





  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

def softmax_func(A, y):
    """
    vectorized version of softmax function
    """
    # first we create a mask to select only the predicted score for the real class, it's tricky but works fine
    mask_y = A[np.arange(y.shape[0]), y]
                        
    num =  np.exp(mask_y)
    denum = np.sum(np.exp(A), axis=1)
    return num, denum

def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0.0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  D = X.shape[0]
  scores = X @ W  # so we have D values in z (z is the score)
  
  
  max_score = np.max(scores, axis=1)  # max scores for each elem in batch
  # defining the constant used to limit instability
  C = - max_score
  scores_stable = scores - C[:, np.newaxis]
  
  num, denum = softmax_func(scores_stable, y)
  loss = np.sum(-np.log(num/denum)) / D + reg * np.sum(W**2)
  
  # NOW the gradient..
  a = np.exp(scores_stable)
  b = np.sum(np.exp(scores_stable), axis=1)[:, None]
  c = a/b
  dW = X.transpose() @ c
  #print("resulting shape is {}".format(dW.shape))
  
  
  
  # On cree une matrice avec des 1 dans les cases [i,j] ssi classe(du sample j) == i
  # et on utilise cette matric pour calculer la matrice qui contient les valeurs à soustraire de dW
  # Cf cahier
  Y = np.zeros( (X.shape[0], W.shape[1]) ) 
  Y[ np.arange(y.shape[0]), y] = 1
  res = - np.transpose(X) @ Y
  #print(res.shape)
  dW += res
      
  #temp = num, denum, scores_stable
  dW /=  y.shape[0]
  # regularization term
  dW += reg * W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW






























