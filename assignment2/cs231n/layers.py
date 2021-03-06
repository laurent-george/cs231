from builtins import range
import numpy as np


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    x_reshape = x.reshape(x.shape[0], -1)
    out = x_reshape @ w + b
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache
    dx, dw, db = None, None, None

    # just applying chain rules with matrix
    dx = (dout @ w.T).reshape(x.shape)

    x_reshape = x.reshape(x.shape[0], -1)
    dw = x_reshape.T @ dout

    N = dout.shape[0]
    db = (np.ones((1,N)) @ dout).flatten()


    return dx, dw, db


def relu_forward(x):
    """
    Computes the forward pass for a layer of rectified linear units (ReLUs).

    Input:
    - x: Inputs, of any shape

    Returns a tuple of:
    - out: Output, of the same shape as x
    - cache: x
    """
    # relu = max(0, x)
    cache = x
    #out = x.copy()
    #out[out<0] = 0
    out = np.maximum(0, x)
    return out, cache


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    dx, x = None, cache

    # 1 if x > 0

    temp =  np.zeros(x.shape)
    temp[x>0] = 1
    dx = temp * dout
    return dx


def batchnorm_forward(x, gamma, beta, bn_param):
    """
    Forward pass for batch normalization.

    During training the sample mean and (uncorrected) sample variance are
    computed from minibatch statistics and used to normalize the incoming data.
    During training we also keep an exponentially decaying running mean of the
    mean and variance of each feature, and these averages are used to normalize
    data at test-time.

    At each timestep we update the running averages for mean and variance using
    an exponential decay based on the momentum parameter:

    running_mean = momentum * running_mean + (1 - momentum) * sample_mean
    running_var = momentum * running_var + (1 - momentum) * sample_var

    Note that the batch normalization paper suggests a different test-time
    behavior: they compute sample mean and variance for each feature using a
    large number of training images rather than using a running average. For
    this implementation we have chosen to use running averages instead since
    they do not require an additional estimation step; the torch7
    implementation of batch normalization also uses running averages.

    Input:
    - x: Data of shape (N, D)
    - gamma: Scale parameter of shape (D,)
    - beta: Shift paremeter of shape (D,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: of shape (N, D)
    - cache: A tuple of values needed in the backward pass
    """
    mode = bn_param['mode']
    eps = bn_param.get('eps', 1e-5)
    momentum = bn_param.get('momentum', 0.9)

    N, D = x.shape
    running_mean = bn_param.get('running_mean', np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get('running_var', np.zeros(D, dtype=x.dtype))

    out, cache = None, None
    if mode == 'train':
        #######################################################################
        # TODO: Implement the training-time forward pass for batch norm.      #
        # Use minibatch statistics to compute the mean and variance, use      #
        # these statistics to normalize the incoming data, and scale and      #
        # shift the normalized data using gamma and beta.                     #
        #                                                                     #
        # You should store the output in the variable out. Any intermediates  #
        # that you need for the backward pass should be stored in the cache   #
        # variable.                                                           #
        #                                                                     #
        # You should also use your computed sample mean and variance together #
        # with the momentum variable to update the running mean and running   #
        # variance, storing your result in the running_mean and running_var   #
        # variables.                                                          #
        #######################################################################
        #mini_batch_mean = np.mean(x, axis=0)
        #mini_batch_variance = np.var(x, axis=0)
        
        # as we want to do backprop.. we are decomposing the mean/variance into steps that are easily differentiable
        
        mini_batch_mean = np.mean(x, axis=0)
        
        x_centered = x - mini_batch_mean
        
        # to compute variance: E((X-E[X])**2)
        
        x_centered_squared = x_centered * x_centered
        
        mini_batch_variance = np.mean(x_centered_squared, axis=0)
        
        sqrt = np.sqrt(mini_batch_variance + eps)
        
        inverse = 1.0 / sqrt
        
        #x_normalized = (x - mini_batch_mean) / np.sqrt(mini_batch_variance + eps)
        x_normalized = x_centered * inverse
        
        x_normalized_scaled = gamma * x_normalized
        
        x_normalized_scaled_and_shifted = x_normalized_scaled + beta
        
        out = x_normalized_scaled_and_shifted
        
        
        cache = (x, eps, gamma, beta, mini_batch_mean, x_centered, x_centered_squared, mini_batch_variance, sqrt, inverse, x_normalized, x_normalized_scaled, x_normalized_scaled_and_shifted)
        
        running_mean = momentum * running_mean + (1 - momentum) * mini_batch_mean
        running_var = momentum * running_var + (1 - momentum) * mini_batch_variance
        
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test-time forward pass for batch normalization. #
        # Use the running mean and variance to normalize the incoming data,   #
        # then scale and shift the normalized data using gamma and beta.      #
        # Store the result in the out variable.                               #
        #######################################################################
        x_normalized = (x - running_mean) / np.sqrt(running_var + eps)
        out = gamma * x_normalized + beta
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param['running_mean'] = running_mean
    bn_param['running_var'] = running_var


    # TODO: what to put into cache ?? 
    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    
    # le blog: http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html m'a bien aidé a debugger/comprendre
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    ###########################################################################
    x, epsilon, gamma, beta, mini_batch_mean, x_centered, x_centered_squared, mini_batch_variance, sqrt, inverse, x_normalized, x_normalized_scaled, x_normalized_scaled_and_shifted = cache
    
    dbeta = np.sum(dout, axis=0)
    
    temp =  x_normalized * dout
    dgamma = np.sum(temp, axis=0)
    
    # pour dx ca devient plus long
    dx_normalized = gamma * dout
    
    d_x_centered_1 = inverse * dx_normalized
    
    # l'autre branche... 
    d_inverse = np.sum(x_centered * dx_normalized, axis=0)
    
    d_sqrt = -1 / (sqrt**2) * d_inverse 
    
    d_mini_batch_variance = 0.5 / np.sqrt(mini_batch_variance + epsilon) * d_sqrt
    
    #print(d_mini_batch_variance.shape)
    
    # this one is a bit "complicated" see step4 here http://kratzert.github.io/2016/02/12/understanding-the-gradient-flow-through-the-batch-normalization-layer.html
    d_x_centered_square = (np.ones(x_centered_squared.shape) / x_centered_squared.shape[0]) * d_mini_batch_variance
    
    d_x_centered_2 = 2 * x_centered * d_x_centered_square
    
    
    # whenever we have two gradients coming to one node, we simply add them up.
    d_x_1 = 1 * (d_x_centered_2 + d_x_centered_1)
    
    d_mini_batch_mean = -1 * np.sum( (d_x_centered_1 + d_x_centered_2) , axis=0)
    
    d_x_2 = (np.ones(x.shape) / x.shape[0]) * d_mini_batch_mean
    
    # TADA :D
    dx = d_x_1 + d_x_2
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def batchnorm_backward_alt(dout, cache):
    """
    Alternative backward pass for batch normalization.

    For this implementation you should work out the derivatives for the batch
    normalizaton backward pass on paper and simplify as much as possible. You
    should be able to derive a simple expression for the backward pass.

    Note: This implementation should expect to receive the same cache variable
    as batchnorm_backward, but might not use all of the values in the cache.

    Inputs / outputs: Same as batchnorm_backward
    """
    dx, dgamma, dbeta = None, None, None
    ###########################################################################
    # TODO: Implement the backward pass for batch normalization. Store the    #
    # results in the dx, dgamma, and dbeta variables.                         #
    #                                                                         #
    # After computing the gradient with respect to the centered inputs, you   #
    # should be able to compute gradients with respect to the inputs in a     #
    # single statement; our implementation fits on a single 80-character line.#
    ###########################################################################
    pass
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def dropout_forward(x, dropout_param):
    """
    Performs the forward pass for (inverted) dropout.

    Inputs:
    - x: Input data, of any shape
    - dropout_param: A dictionary with the following keys:
      - p: Dropout parameter. We drop each neuron output with probability p.
      - mode: 'test' or 'train'. If the mode is train, then perform dropout;
        if the mode is test, then just return the input.
      - seed: Seed for the random number generator. Passing seed makes this
        function deterministic, which is needed for gradient checking but not
        in real networks.

    Outputs:
    - out: Array of the same shape as x.
    - cache: tuple (dropout_param, mask). In training mode, mask is the dropout
      mask that was used to multiply the input; in test mode, mask is None.
    """
    p, mode = dropout_param['p'], dropout_param['mode']
    if 'seed' in dropout_param:
        np.random.seed(dropout_param['seed'])

    mask = None
    out = None

    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase forward pass for inverted dropout.   #
        # Store the dropout mask in the mask variable.                        #
        #######################################################################
        mask = (np.random.rand(*x.shape) > p) / p
        out = x*mask
        #######################################################################
        #                           END OF YOUR CODE                          #
        #######################################################################
    elif mode == 'test':
        #######################################################################
        # TODO: Implement the test phase forward pass for inverted dropout.   #
        #######################################################################
        out = x
        #######################################################################
        #                            END OF YOUR CODE                         #
        #######################################################################

    cache = (dropout_param, mask)
    out = out.astype(x.dtype, copy=False)

    return out, cache


def dropout_backward(dout, cache):
    """
    Perform the backward pass for (inverted) dropout.

    Inputs:
    - dout: Upstream derivatives, of any shape
    - cache: (dropout_param, mask) from dropout_forward.
    """
    dropout_param, mask = cache
    mode = dropout_param['mode']

    dx = None
    if mode == 'train':
        #######################################################################
        # TODO: Implement training phase backward pass for inverted dropout   #
        #######################################################################
        dx = mask * dout
        #######################################################################
        #                          END OF YOUR CODE                           #
        #######################################################################
    elif mode == 'test':
        dx = dout
    return dx


def conv_forward_naive(x, w, b, conv_param):
    """
    A naive implementation of the forward pass for a convolutional layer.

    The input consists of N data points, each with C channels, height H and
    width W. We convolve each input with F different filters, where each filter
    spans all C channels and has height HH and width HH.

    Input:
    - x: Input data of shape (N, C, H, W)
    - w: Filter weights of shape (F, C, HH, WW)
    - b: Biases, of shape (F,)
    - conv_param: A dictionary with the following keys:
      - 'stride': The number of pixels between adjacent receptive fields in the
        horizontal and vertical directions.
      - 'pad': The number of pixels that will be used to zero-pad the input.

    Returns a tuple of:
    - out: Output data, of shape (N, F, H', W') where H' and W' are given by
      H' = 1 + (H + 2 * pad - HH) / stride
      W' = 1 + (W + 2 * pad - WW) / stride
    - cache: (x, w, b, conv_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the convolutional forward pass.                         #
    # Hint: you can use the function np.pad for padding.                      #
    ###########################################################################
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    stride = conv_param['stride']
    pad = conv_param['pad']
    
    H_prime = int(1 + (H+2*pad -HH) /stride)
    W_prime = int(1 + (H+2*pad -WW) /stride)
    out = np.zeros((N, F, H_prime, W_prime))
    
    for n in range(N):
        for f in range(F):
            for c in range(C):
                cur_X = np.pad(x[n, c, :, :], pad, 'constant')
                filter = w[f, c, :, :]
                
                out_top = 0
                for top in range(0, H, stride):
                    out_left = 0
                    for left in range(0, W, stride):
                        convolution_res = np.sum(filter * cur_X[top:top+HH, left:left+WW])
                        out[n, f, out_top, out_left] += convolution_res
                        out_left += 1
                    out_top += 1
                    
    for n in range(N):
        out[n, :, :, :] += b[:, None, None]
    
    # TODO: revoir ca pour retrouver un produit en utilisant un reshape.. 
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, w, b, conv_param)
    return out, cache


def conv_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a convolutional layer.

    Inputs:
    - dout: Upstream derivatives.
    - cache: A tuple of (x, w, b, conv_param) as in conv_forward_naive

    Returns a tuple of:
    - dx: Gradient with respect to x
    - dw: Gradient with respect to w
    - db: Gradient with respect to b
    """
    dx, dw, db = None, None, None
    ###########################################################################
    # TODO: Implement the convolutional backward pass.                        #
    ###########################################################################
    x, w, b, conv_param = cache
    stride = conv_param['stride']
    N, C, H, W = x.shape
    F, _, HH, WW = w.shape
    
    # easy db:
    db = np.sum(dout, axis=(0, 2, 3))
    
    # dx
    pad = conv_param['pad']
    
    H_prime = int(1 + (H+2*pad -HH) /stride)
    W_prime = int(1 + (H+2*pad -WW) /stride)
    print("N = {}, C={}, H={}, W={}, F={}, HH={}, WW={}, H_prime={}, W_prime={}".format(N, C, H, W, F, HH, WW, H_prime, W_prime))
    
    #dx = np.zeros(x.shape)
    npad = ((0,0), (0,0), (pad,pad), (pad,pad))
    xpad = np.pad(x, pad_width=npad, mode='constant', constant_values=0)
    dx = np.zeros_like(xpad)  # pour l'instant on garde un dx plus grand on cropera apres
    
    # simple version.. we use similar loop as the one in forward
    for n in range(N):
        for f in range(F):
            for c in range(C):
                out_top = 0
                for top in range(0, H, stride):
                    out_left = 0
                    for left in range(0, W, stride):
                        for i in range(top, top+HH):
                            for j in range(left, left+WW):
                                # it's just looking for which pixels contribute with wich weight on output 
                                dx[n, c, i, j] += w[f, c, i-top, j-left] * dout[n, f, out_top, out_left]
                        out_left += 1
                    out_top += 1
    
    dx = dx[:,:, pad:H+pad, pad:W+pad]
    
    
    # dw
    dw = np.zeros_like(w)
    # simple version.. we use similar loop as the one in forward
    for n in range(N):
        for f in range(F):
            for c in range(C):
                out_top = 0
                for top in range(0, H, stride):
                    out_left = 0
                    for left in range(0, W, stride):
                        for i in range(top, top+HH):
                            for j in range(left, left+WW):
                                # it's just looking for which weight contributes with wich pixel to produce output 
                                
                                
                                # we use the xpad.. as we have 0 it just discard the weight that are used on the pad part
                                val = xpad[n, c, i, j]
                                dw[f, c, i-top, j-left] += val * dout[n, f, out_top, out_left]
                        out_left += 1
                    out_top += 1
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx, dw, db


def max_pool_forward_naive(x, pool_param):
    """
    A naive implementation of the forward pass for a max pooling layer.

    Inputs:
    - x: Input data, of shape (N, C, H, W)
    - pool_param: dictionary with the following keys:
      - 'pool_height': The height of each pooling region
      - 'pool_width': The width of each pooling region
      - 'stride': The distance between adjacent pooling regions

    Returns a tuple of:
    - out: Output data
    - cache: (x, pool_param)
    """
    out = None
    ###########################################################################
    # TODO: Implement the max pooling forward pass                            #
    ###########################################################################
    
    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    #x_reshaped = x.reshape(N, C, H/pool_height, W/pool_width, 
    
    stride = pool_param['stride']
    h_out = int((H - pool_height)/stride  + 1)
    w_out = int((W - pool_width)/stride  +  1)
    out = np.zeros((N, C, h_out, w_out))
    
    # let's do it naively
    out_row = 0
    for row in range(0, H, stride):
        out_col = 0
        for col in range(0, W, stride):
            #print("max on {}:{} ,, {}:{}".format(row, row+pool_height, col, col+pool_width))
            for n in range(N):
                for c in range(C):
                    out[n, c, out_row, out_col] = np.max(x[n, c, row:row+pool_height, col:col+pool_width])
            out_col += 1
        out_row += 1
    
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    cache = (x, pool_param)
    return out, cache


def max_pool_backward_naive(dout, cache):
    """
    A naive implementation of the backward pass for a max pooling layer.

    Inputs:
    - dout: Upstream derivatives
    - cache: A tuple of (x, pool_param) as in the forward pass.

    Returns:
    - dx: Gradient with respect to x
    """
    dx = None
    ###########################################################################
    # TODO: Implement the max pooling backward pass                           #
    ###########################################################################
    #pass

# naive approach, we use similar loop to forward
    x, pool_param = cache
    dx = np.zeros_like(x)

    N, C, H, W = x.shape
    pool_height = pool_param['pool_height']
    pool_width = pool_param['pool_width']
    #x_reshaped = x.reshape(N, C, H/pool_height, W/pool_width, 
    
    stride = pool_param['stride']
    h_out = int((H - pool_height)/stride  + 1)
    w_out = int((W - pool_width)/stride  +  1)
    out = np.zeros((N, C, h_out, w_out))
    
    # let's do it naively
    out_row = 0
    for row in range(0, H, stride):
        out_col = 0
        for col in range(0, W, stride):
            #print("max on {}:{} ,, {}:{}".format(row, row+pool_height, col, col+pool_width))
            for n in range(N):
                for channel in range(C):
                    # We are interested only in coordinates where   # < ---- ICI 
                    res = np.argmax(x[n, channel, row:row+pool_height, col:col+pool_width])
                    #maxx = np.max(x[n, channel, row:row+pool_height, col:col+pool_width])
                    pos_x, pos_y = np.unravel_index(res, (pool_height, pool_width))
                    #print("HxW {}x{}".format(pool_height, pool_width))
                    #print("res is {} Pos x, y = {}, {}, max is {}, should be {}".format(res, pos_x, pos_y, x[n, channel, row+pos_x, col+pos_y], maxx))
                    #print("sub tab was {}".format(x[n, channel, row:row+pool_height, col:col+pool_width]))
                    dx[n, channel, row+pos_x, col+pos_y] += dout[n, channel, out_row, out_col]
            out_col += 1
        out_row += 1


    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    return dx


def spatial_batchnorm_forward(x, gamma, beta, bn_param):
    """
    Computes the forward pass for spatial batch normalization.

    Inputs:
    - x: Input data of shape (N, C, H, W)
    - gamma: Scale parameter, of shape (C,)
    - beta: Shift parameter, of shape (C,)
    - bn_param: Dictionary with the following keys:
      - mode: 'train' or 'test'; required
      - eps: Constant for numeric stability
      - momentum: Constant for running mean / variance. momentum=0 means that
        old information is discarded completely at every time step, while
        momentum=1 means that new information is never incorporated. The
        default of momentum=0.9 should work well in most situations.
      - running_mean: Array of shape (D,) giving running mean of features
      - running_var Array of shape (D,) giving running variance of features

    Returns a tuple of:
    - out: Output data, of shape (N, C, H, W)
    - cache: Values needed for the backward pass
    """
    out, cache = None, None

    ###########################################################################
    # TODO: Implement the forward pass for spatial batch normalization.       #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    
    # just flatten over channels
    N, C, H, W = x.shape
    x_reshaped = np.transpose(x, [1, 0, 2, 3])
    x_reshaped = np.transpose(x_reshaped.reshape(C, -1))
    
    temp_out, temp_cache = batchnorm_forward(x_reshaped, gamma, beta, bn_param)
    
    temp_out = np.transpose(temp_out).reshape([C, N, H, W])
    temp_out = np.transpose(temp_out, [1, 0, 2, 3])
    
    out = temp_out
    cache = temp_cache
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################
    

    return out, cache


def spatial_batchnorm_backward(dout, cache):
    """
    Computes the backward pass for spatial batch normalization.

    Inputs:
    - dout: Upstream derivatives, of shape (N, C, H, W)
    - cache: Values from the forward pass

    Returns a tuple of:
    - dx: Gradient with respect to inputs, of shape (N, C, H, W)
    - dgamma: Gradient with respect to scale parameter, of shape (C,)
    - dbeta: Gradient with respect to shift parameter, of shape (C,)
    """
    dx, dgamma, dbeta = None, None, None

    ###########################################################################
    # TODO: Implement the backward pass for spatial batch normalization.      #
    #                                                                         #
    # HINT: You can implement spatial batch normalization using the vanilla   #
    # version of batch normalization defined above. Your implementation should#
    # be very short; ours is less than five lines.                            #
    ###########################################################################
    N, C, H, W = dout.shape
    
    dout_reshaped = np.transpose(dout, [0, 2, 3, 1])  # now having N, H, W, C
    dout_reshaped = dout_reshaped.reshape(N*H*W, C)
    
    temp_dx, temp_dgamma, temp_dbeta = batchnorm_backward(dout_reshaped, cache)
    
    #print(temp_dx.shape)
    #print(temp_dgamma.shape)
    #print(temp_dbeta.shape)
    
    
    dx = temp_dx.reshape((N,H,W,C)).transpose((0,3,1,2))  # now N, C, H, W as expected
    dgamma = temp_dgamma
    dbeta = temp_dbeta
    
    ###########################################################################
    #                             END OF YOUR CODE                            #
    ###########################################################################

    return dx, dgamma, dbeta


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


def softmax_loss(x, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = x - np.max(x, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = x.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx
