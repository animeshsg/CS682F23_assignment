import numpy as np
from random import shuffle

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
  #print("svm_loss_naive")
  dW = np.zeros(W.shape) # initialize the gradient as zero

  # compute the loss and the gradient
  loss = 0
  for i in range(X.shape[0]):
    vals = X[i]@W # score values for each class and data
    #vals -= np.max(vals) # for numerical stability
    ccvals = vals[y[i]] #correct class values
    for j in range(W.shape[1]):
      if j == y[i]:
        continue
      inner_vals = vals[j] - ccvals + 1 # note delta = 1
      if inner_vals > 0:
        loss += inner_vals
        #Subgradients of multiclass SVM loss is +X for Wj and -X for W_yi and 0 when negative inner_vals
        dW[:,j] += X[i].T
        dW[:,y[i]] -= X[i].T


  # Right now the loss is a sum over all training examples, but we want it
  # to be an average instead so we divide by num_train.
  loss /= X.shape[0]

  # Add regularization to the loss.
  loss += reg * np.sum(W**2)

  #Average the gradient
  dW/=X.shape[0]
  dW += 2*reg*W

  #############################################################################
  # TODO:                                                                     #
  # Compute the gradient of the loss function and store it dW.                #
  # Rather that first computing the loss and then computing the derivative,   #
  # it may be simpler to compute the derivative at the same time that the     #
  # loss is being computed. As a result you may need to modify some of the    #
  # code above to compute the gradient.                                       #
  #############################################################################
  return loss, dW


def svm_loss_vectorized(W, X, y, reg):
  """
  Structured SVM loss function, vectorized implementation.

  Inputs and outputs are the same as svm_loss_naive.
  """
  loss = 0
  dW = np.zeros(W.shape) # initialize the gradient as zero
  vals=X@W # shape(N,C)
  #vals-=np.max(vals,axis=1)[:,np.newaxis] #shape(N,1) for numerical stability
  ccvals=vals[np.arange(vals.shape[0]),y]
  iv=vals-ccvals[:,np.newaxis]+1 #shape(N,C)  inner values iinside the max func
  loss=np.sum(iv[iv>0])/X.shape[0]-1+reg*np.sum(W**2)
  iv[iv<=0]=0
  iv[iv>0]=1
  iv[np.arange(iv.shape[0]),y]-=np.sum(iv,axis=1)
  dW=X.T@iv/X.shape[0]+2*reg*W



  #############################################################################
  # TODO:                                                                     #
  # Implement a vectorized version of the structured SVM loss, storing the    #
  # result in loss.                                                           #
  #############################################################################
  pass
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
  pass
  #############################################################################
  #                             END OF YOUR CODE                              #
  #############################################################################

  return loss, dW
