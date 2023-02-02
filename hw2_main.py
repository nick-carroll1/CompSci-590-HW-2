import argparse
import numpy as np
import hw2_utils as utils


def get_mnist_data():
    loaded = np.load("mnist_dat.npz")
    X, y = loaded["X"], loaded["y"]
    
    # make one-hot
    Y = np.zeros((X.shape[0], 10))
    Y[np.arange(Y.shape[0]), y] = 1

    valperc = 0.1
    nval = int(Y.shape[0] * valperc)

    val_X, val_Y = X[-nval:], Y[-nval:]
    X, Y = X[:-nval], Y[:-nval]
    return X, Y, val_X, val_Y


def eval_perf(theta, X, Y, loss_fn, pred_fn):
    Xs, Ys = batchify(X, Y, 10) # bsz=10 so we don't truncate on val
    total_loss, ncorrect, ntotal = 0, 0, 0
    for i, X_i in enumerate(Xs):
        Y_i = Ys[i]
        scores = pred_fn(theta, X_i)
        preds = scores.argmax(1)
        golds = Y_i.argmax(1)
        ncorrect += (preds == golds).sum()
        mb_loss = loss_fn(theta, X_i, Y_i)
        total_loss += mb_loss
        ntotal += Y_i.shape[0]
    return total_loss/len(Xs), ncorrect/ntotal


def batchify(X, Y, bsz):
    ntrunc = X.shape[0] // bsz * bsz
    nbatch = ntrunc // bsz if bsz > 0 else 1
    Xs, Ys = np.split(X[:ntrunc], nbatch), np.split(Y[:ntrunc], nbatch)
    return Xs, Ys


def train_mnist(theta, loss_fn, pred_fn):
    X, Y, val_X, val_Y = get_mnist_data()

    if loss_fn == utils.lin_xent:
        # add bias terms
        X = np.hstack((X, np.ones((X.shape[0], 1))))
        val_X = np.hstack((val_X, np.ones((val_X.shape[0], 1))))

    gvf = utils.ad.make_grad_and_val_func(loss_fn)
    eta, bsz = args.eta, args.bsz
    for epoch in range(1, args.nepochs+1):
        randperm = np.random.permutation(X.shape[0])
        X, Y = X[randperm], Y[randperm]
        Xs, Ys = batchify(X, Y, bsz)
        total_loss, nexamples = 0, 0
        for i, X_i in enumerate(Xs):
            Y_i = Ys[i]
            grad, mb_loss = gvf(theta, X_i, Y_i)
            total_loss += mb_loss
            if isinstance(theta, list):
                for p in range(len(theta)):
                    theta[p] = theta[p] - eta*grad[p]
            else:
                theta = theta - eta * grad

        print("epoch {} | train_loss {:.4f} | eta {:.5f}".format(
            epoch, total_loss/len(Xs), eta))

        val_loss, val_acc = eval_perf(theta, val_X, val_Y, loss_fn, pred_fn)
        print("epoch {} | val_loss {:.4f} | val_acc {:.3f}".format(
            epoch, val_loss, val_acc))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nepochs", type=int, default=20)
    parser.add_argument("--bsz", type=int, default=32)
    parser.add_argument("--eta", type=float, default=1e-2)
    parser.add_argument("--problem", type=str, default="1.1", 
                        choices=["1.1", "1.2", "2.1", "3.1"])
    args = parser.parse_args()

    np.random.seed(59006)

    x = np.random.randn(1)
    w = np.random.randn(1)
    b = np.random.randn(1)
    y = np.random.randn(1)   

    ########## 1.1 ###########
    if args.problem == "1.1":
        print("Problem 1.1:")
        print("*"*45)
        utils.single_var_func_opt(utils.single_var_func1, [w, b, x, y], 1e-2)

    ########## 1.2 ########### 
    if args.problem == "1.2":
        print("Problem 1.2:")
        print("*"*45)
        utils.single_var_func_opt(utils.single_var_func2, [w, b, x, y], 1e-2)

    ########## 2.1 ########### 
    if args.problem == "2.1":
        print("Problem 2.1:")
        print("*"*45)
        mnistK, mnistD1 = 10, 784
        theta = np.zeros((mnistK, mnistD1+1)) # K x D_1
        train_mnist(theta, utils.lin_xent, utils.lin_pred)

    ########## 3.1 ########### 
    if args.problem == "3.1":
        print("Problem 3.1:")
        print("*"*45)
        mnistK, mnistD1, mnistH1 = 10, 784, 256
        W2 = np.random.randn(mnistK, mnistH1) * (2/(10+256))**0.5
        W1 = np.random.randn(mnistH1, mnistD1) * 1.67 * (2/(784+256))**0.5
        b = np.zeros((mnistH1,))
        train_mnist([W2, W1, b], utils.mlp_xent, utils.mlp_pred)
