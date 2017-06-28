from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data
        - num_filters: Number of filters to use in the convolutional layer
        - filter_size: Size of filters to use in the convolutional layer
        - hidden_dim: Number of units to use in the fully-connected hidden layer
        - num_classes: Number of scores to produce from the final affine layer.
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights.
        - reg: Scalar giving L2 regularization strength
        - dtype: numpy datatype to use for computation.
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian with standard     #
        # deviation equal to weight_scale; biases should be initialized to zero.   #
        # All weights and biases should be stored in the dictionary self.params.   #
        # Store weights and biases for the convolutional layer using the keys 'W1' #
        # and 'b1'; use keys 'W2' and 'b2' for the weights and biases of the       #
        # hidden affine layer, and keys 'W3' and 'b3' for the weights and biases   #
        # of the output affine layer.                                              #
        ############################################################################
        # Calculer le nombre de filtre avec la formule convolution
        
        w1_shape = (num_filters, input_dim[0], filter_size, filter_size)
        self.params['W1'] = (np.random.randn(*w1_shape) * weight_scale)
        self.params['b1'] = np.zeros(num_filters)
        
        # NOT GOOD DUPLICATION of code here
        # pass conv_param to the forward pass for the convolutional layer
        stride = 1
        pool_stride =  2
        pool_height = 2
        pool_width = 2
        
        pad =  (filter_size - 1) // 2
        H = input_dim[1]
        W = input_dim[2]
        H_prime = int(1 + (H + 2 * pad - filter_size) / stride)
        W_prime = int(1 + (W + 2 * pad - filter_size) / stride)
        
        H_pool = int((H_prime - pool_height) / pool_stride + 1)
        W_pool = int((W_prime - pool_width) / pool_stride + 1)
        conv_output_dim = (num_filters* H_pool* W_pool)

        W2 = (np.random.randn(conv_output_dim, hidden_dim) * weight_scale)
        b2 = np.zeros(hidden_dim)
                           
        self.params['W2'] = W2
        self.params['b2'] = b2
        
        W3 = (np.random.randn(hidden_dim, num_classes) * weight_scale)
        b3 = np.zeros(num_classes)
        self.params['W3'] = W3
        self.params['b3'] = b3
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        ############################################################################
        out_layer_1, cache_layer_1 = conv_relu_pool_forward(X, W1, b1, conv_param, pool_param)
        out_layer_2, cache_layer_2 =  affine_relu_forward(out_layer_1, W2, b2)
        out_layer_3, cache_layer_3 = affine_forward(out_layer_2, W3, b3)
        scores = out_layer_3

        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        ############################################################################
        loss, grads_soft_max = softmax_loss(scores, y)
        
        # TODO: finish gradient
        
        layer3_dx, layer3_dw, layer3_db = affine_backward(grads_soft_max, cache_layer_3)
        layer2_dx, layer2_dw, layer2_db = affine_relu_backward(layer3_dx, cache_layer_2)
        layer1_dx, layer1_dw, layer1_db = conv_relu_pool_backward(layer2_dx, cache_layer_1)

        if self.reg != 0:
            # adding L2 regularization
            regularization_factor = 0.5 * self.reg
            loss += np.sum(regularization_factor * self.params['W1'] ** 2)
            loss += np.sum(regularization_factor * self.params['W2'] ** 2)
            loss += np.sum(regularization_factor * self.params['W3'] ** 2)

            layer1_dw += self.reg * self.params['W1']
            layer2_dw += self.reg * self.params['W2']
            layer3_dw += self.reg * self.params['W3']
        
        grads['W2'] = layer2_dw
        grads['b2'] = layer2_db
        
        grads['W1'] = layer1_dw
        grads['b1'] = layer1_db
        
        grads['W3'] = layer3_dw
        grads['b3'] = layer3_db
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
