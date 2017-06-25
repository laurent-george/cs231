import numpy as np
from random import shuffle
from past.builtins import xrange

def svm_loss_naive(W, X, y, reg):
  """
  Structured SVM loss function, naive implementation (with loops).

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
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  num_classes = W.shape[1]
  num_train = X.shape[0]
  loss = 0.0
  for i in xrange(num_train):
    scores = X[i].dot(W)
    correct_class_score = scores[y[i]]
    count = 0
    for j in xrange(num_classes):
      if j == y[i]:
        continue

      margin = scores[j] - correct_class_score + 1 # note delta = 1
      if margin > 0:
        loss += margin
        count += 1
        dW[:, j] += X[i]
    dW[:, y[i]] += - count * X[i]



  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= num_train
  dW /= num_train
  
  

  # Add regularization to the loss.
  loss += reg * np.sum(W * W)

  # Add regularization to the gradient.
  # TODO

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  dW += reg * W
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0.0
  dW = np.zeros(W.shape) # initialize the gradient as zero

  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  
  number_of_inf_in_W = np.sum(np.isinf(W))
  
  scores = X @ W  # <<-- introduce some nan ? -> yes..
  
  #number_of_inf_in_scores = np.sum(np.isinf(scores))
  #number_of_inf_in_X= np.sum(np.isinf(X))
  #if number_of_inf_in_scores > 0:
  #  print("number of inf in score is {}, W {}, X {}".format(number_of_inf_in_scores, number_of_inf_in_W, number_of_inf_in_X))
  #  np.save("/tmp/X.npy", X)
  #  np.save("/tmp/W.npy", W)
  #  raise ValueError("nan occure")
  
  
  #number_of_nan = np.sum(np.isnan(scores))
  #if  number_of_nan > 0:
    #  print("ERROR !!!! we have {} nan in score".format(number_of_nan))
    
  #if number_of_nan_in_W > 0:
    #print("number of nan in W is {}".format(number_of_nan_in_W))
  

  num_train = X.shape[0]
  correct_class_score = scores[np.arange(num_train), y]
  

  margins = scores[:, :] - correct_class_score[:, None] + 1
  margins[margins<=0] = 0

  # if y[i] ==j -> set margins to 0
  margins[np.arange(num_train), y] = 0

  loss = np.sum(margins) / num_train

  loss += reg * np.sum(W * W)

  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################
  


  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the gradient for the structured SVM     #
  # loss, storing the result in dW.                                           #
  #                                                                           #
  # Hint: Instead of computing the gradient from scratch, it may be easier    #
  # to reuse some of the intermediate values that you used to compute the     #
  # loss.                                                                     #
  #############################################################################
  # For Dw : we build a matrix where value is 1 if > margin, 0 else, and -cur_count when j == [y[i]] 
  #Y = np.zeros( (X.shape[0], W.shape[1]) ) 
  
  # we reuse margins array here.. as Y should be close to margins except for the count part
  #Y = margins.copy()
  #Y[Y>0] = 1
  #Y[Y<=0] = 0
  
  # set -1 when j == y[i] meaning for the correct class
  #cur_count = np.sum(Y >0, axis=1)
  #Y[ np.arange(y.shape[0]), y] = -1 * cur_count
  
  # we use this matrix to build dW
  #dW = np.transpose(X) @ Y
  
  #dW /= num_train
  
  # regularization of dW term
  #dW += reg * W
  
  
  # alternative way
  
  # d(margin)/dscores
  dmargin = margins.copy()
  dmargin[dmargin <=0] = 0   # derivative of maximum between 0, negative value  -> does not depends on the negative value.. so it's 0
  dmargin[dmargin > 0] = 1   # now the derivative is the derivative of the score - a + 1 .. which is 1
  # d(scorers)/dW
  dscores = np.transpose(X)
  
  # dLoss/dW
  dW = dscores @ dmargin
  dW /= num_train
  
  # we add the regulalization term
  dW += reg * W
  
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
