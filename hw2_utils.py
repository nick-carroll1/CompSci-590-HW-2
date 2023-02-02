from autodiff import (AutoDiffer, add, sub, mult, div,
                      square, cosh, sinh, tanh, exp, log,
                      mm, T, dimmax, dimsum, bdcast,
                      get)

ad = AutoDiffer() # our global autodiffer object, which will be imported in hw2_main.py

def single_var_func1(w, b, x, y):
    """
    all arguments are numpy scalars.
    returns (cosh(w^2) + sinh(wx+b) - y)^2
    """
    z = add(cosh(square(w)),
            sinh(add(mult(w, x), b)))
    return square(sub(z, y))


def single_var_func2(w, b, x, y):
    """
    all arguments are numpy scalars.
    returns (cosh(w^2) * tanh(wx+b) - y)^2
    """
    raise NotImplementedError()


def single_var_func_opt(lossfunc, args, lr, niter=10):
    """
    lossfunc - a function which returns a scalar
    args - a list of arguments to lossfunc
    lr - a scalar learning rate
    niter - number of gradient descent steps to take
    """
    gvf = ad.make_grad_and_val_func(lossfunc)
    for i in range(niter):
        grad, lossval = gvf(*args)
        print("grad: {}, funcval: {}".format(grad, lossval))
        args[0] = args[0] - lr*grad


def logsumexp(Z, axis):
    """
    Z - M x N
    axis - 0 or 1
    returns:
       M-dimensional array if axis=1; N-dimensional array if axis=0
    """
    maxes = dimmax(Z, axis)
    return add(maxes, 
               log(dimsum(exp(sub(Z, bdcast(maxes, Z.shape, axis))), axis)))


def lin_pred(theta, X):
    """
    theta - K x D_1
    X - B x D_1
    returns:
      a B x K matrix of scores
    """
    return mm(X, T(theta))


def lin_xent(theta, X, Y):
    """
    theta - K x D_1
    X - B x D_1
    Y - B x K
    returns: 1/B \sum_b xent(x_b, y_b)
    """
    bsz = X.shape[0]
    S = lin_pred(theta, X)
    raise NotImplementedError()


def mlp_pred(params, X):
    """
    params - [W2, W1, b], where W2 is K x H_1, W_1 is H_1 x D_1, and b is a H_1-length vector
    X - B x D_1
    returns:
      a B x K matrix of scores computed as: tanh(X W1^T + b) W2^T
    """
    W2, W1, b = get(params, 0), get(params, 1), get(params, 2)
    raise NotImplementedError()


def mlp_xent(params, X, Y):
    """
    params - [W2, W1, b], where W2 is K x H_1, W_1 is H_1 x D_1, and b is a H_1-length vector
    X - B x D_1
    Y - B x K
    returns: 1/B \sum_b xent(x_b, y_b)
    """
    bsz = X.shape[0]
    S = mlp_pred(params, X)
    raise NotImplementedError()
