import numpy as np
from random import shuffle

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
  loss = 0
  dW = np.zeros_like(W)

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using explicit loops.     #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  for i in range(X.shape[0]):
      vals = X[i]@W
      vals -= np.max(vals) #to ensure numerical instability is mitigated. so we subtract the max to stop potential overflow
      #softmax calcs
      vals = np.exp(vals) 
      vals /= np.sum(vals)
      #cross entropy loss
      loss -= np.log(vals[y[i]])
      vals[y[i]] -= 1
      dW += np.outer(X[i], vals)
  
  loss /= X.shape[0]
  loss += reg * np.sum(W**2)
  dW /= X.shape[0]
  dW += 2*reg*W
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW 


def softmax_loss_vectorized(W, X, y, reg):
  """
  Softmax loss function, vectorized version.

  Inputs and outputs are the same as softmax_loss_naive.
  """
  # Initialize the loss and gradient to zero.
  loss = 0
  dW = np.zeros_like(W)

  vals=X@W
  vals -= np.max(vals, axis=1, keepdims=True) #to ensure numerical instability is mitigated. so we subtract the max to stop the overflow 
  #softmax calcs
  vals = np.exp(vals)
  vals /= np.sum(vals, axis=1, keepdims=True)
  #cross entropy loss
  loss -= np.sum(np.log(vals[np.arange(X.shape[0]), y]))

  vals[np.arange(X.shape[0]), y] -= 1 #Softmax_grads: subtract 1 from correct class and 0 from others, its like subtracting probits with one hot encoded Y to help calculate the gradient
  dW += X.T @ vals #calculate the gradients by multipluing the gradient of relu i.e X with the previous vals

  loss /= X.shape[0]
  loss += reg*np.sum(W**2) 
  dW /= X.shape[0]
  dW += 2*reg*W
  

  #############################################################################
  # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
  # Store the loss in loss and the gradient in dW. If you are not careful     #
  # here, it is easy to run into numeric instability. Don't forget the        #
  # regularization!                                                           #
  #############################################################################
  pass
  #############################################################################
  #                          END OF YOUR CODE                                 #
  #############################################################################

  return loss, dW

