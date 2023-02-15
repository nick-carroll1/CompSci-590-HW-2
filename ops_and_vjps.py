import numpy as np


class Result:
    def __init__(self, val, op, parents, trace_levels):
        self.val = val  # the numerical value this Result wraps
        self.op = op  # the op that created this Result
        self.parents = parents  # aka the args to the op that created this Result
        self.trace_levels = trace_levels  # a set of integers

    def __repr__(self):
        return "Result(traces={}): {}".format(self.trace_levels, self.val.__repr__())

    def __str__(self):
        return "Result(traces={}): {}".format(self.trace_levels, self.val.__str__())

    def get_val(self):
        """
        returns unwrapped val
        """
        val = self.val
        while isinstance(val, Result):
            val = val.val
        return val

    def clone(self, remove_level=None):
        """
        makes a copy of self, and optionally removes remove_level from its trace_levels
        """
        resclone = Result(self.val, self.op, self.parents, self.trace_levels.copy())
        if remove_level is not None and remove_level in resclone.trace_levels:
            resclone.trace_levels.remove(remove_level)
        return resclone

    def get_all_parents(self, remove_level):
        """
        returns a list of *all* parents irrespective of level. parents with
        remove_level in their self.trace_levels will have this level removed.
        """
        return [
            par.clone(remove_level) if isinstance(par, Result) else par
            for par in self.parents
        ]

    def get_bprop_parents(self, level):
        """
        return a list of (argument_idx, parent_result) containing each parent result
        with level in its self.trace_levels
        """
        return [
            (i, par)
            for i, par in enumerate(self.parents)
            if isinstance(par, Result) and level in par.trace_levels
        ]

    # convenience methods so we can treat everything like a numpy array
    @property
    def shape(self):
        _val = self.get_val()
        if hasattr(_val, "shape"):
            return _val.shape
        return (1,)

    @property
    def size(self):
        _val = self.get_val()
        if hasattr(_val, "size"):
            return _val.size
        return 1

    @staticmethod
    def result_from_op_if_needed(op, *args):
        """
        makes a Result from applying op to args if necessary; otherwise just returns op(args)
        """
        trace_levels = set()
        for arg in args:
            if isinstance(arg, Result):
                trace_levels.update(arg.trace_levels)

        if len(trace_levels) > 0:
            unwrapped_res = op.compute_op(
                *(arg.get_val() if hasattr(arg, "get_val") else arg for arg in args)
            )
            return Result(unwrapped_res, op, args, trace_levels)
        return op.compute_op(*args)


## -------------------------------------------- OPs utilities ----------------------------------------------- ##
def _get_shape(arg):
    assert not isinstance(
        arg, list
    ), "lists have no shape; arg should be numeric or an ndarray!"
    shape = arg.shape if hasattr(arg, "shape") else (1,)
    if not shape:  # can happen w/ numpy scalars which have an empty shape attribute
        shape = (1,)
    return shape


def _safe_add(arg0, arg1):
    shape0 = _get_shape(arg0)
    shape1 = _get_shape(arg1)
    assert shape0 == shape1
    return arg0 + arg1


def _safe_mult(arg0, arg1):
    shape0 = _get_shape(arg0)
    shape1 = _get_shape(arg1)
    assert shape0 == shape1
    return arg0 * arg1


## -------------------------------------------- OPs definitions --------------------------------------------- ##
class MyOp:
    def compute_op(self, *args):
        raise NotImplementedError()

    def __call__(self, *args):
        """
        by defining the __call__ method, we make MyOp objects callable, like functions.
        For instance if we create a new op with op = MyOp(), we can then do op(array).
        """
        return Result.result_from_op_if_needed(self, *args)


class MyPutAtIdx(MyOp):
    def compute_op(self, alist, idx, thing):
        newlist = alist[:]
        newlist[idx] = thing
        return newlist


class MyGetAtIdx(MyOp):
    def compute_op(self, alist, idx):
        return alist[idx]


class MyAdd(MyOp):
    def compute_op(self, arg0, arg1):
        return _safe_add(arg0, arg1)


class MyAccum(MyOp):
    def compute_op(self, arg0, arg1):
        if isinstance(arg0, list) and isinstance(arg1, list):
            assert len(arg0) == len(arg1)
            return [arg0[i] + arg1[i] for i in range(len(arg0))]
        return arg0 + arg1


class MyMultiply(MyOp):
    def compute_op(self, arg0, arg1):
        return _safe_mult(arg0, arg1)


class MySubtract(MyOp):
    def compute_op(self, arg0, arg1):
        return arg0 - arg1


class MySquare(MyOp):
    def compute_op(self, arg0):
        return arg0**2


class MyNeg(MyOp):
    def compute_op(self, arg0):
        return -arg0


class MyDivide(MyOp):
    def compute_op(self, arg0, arg1):
        return arg0 / arg1


class MySinh(MyOp):
    def compute_op(self, arg0):
        return np.sinh(arg0)


class MyCosh(MyOp):
    def compute_op(self, arg0):
        return np.cosh(arg0)


class MyTanh(MyOp):
    def compute_op(self, arg0):
        return np.tanh(arg0)


class MyExp(MyOp):
    def compute_op(self, arg0):
        return np.exp(arg0)


class MyLog(MyOp):
    def compute_op(self, arg0):
        return np.log(arg0)


class MyBroadcast(MyOp):
    """
    allows broadcasting scalar to vector or to tensor, or vector to matrix
    """

    def compute_op(self, arg0, shape, axis):
        """
        shape is DESIRED shape
        axis is needed to disambiguate when desired nrows equals ncols (for vectors)
        """
        assert len(shape) <= 2, "can broadcast to at most a matrix"
        assert (
            isinstance(arg0, float) or len(arg0.shape) == 1
        ), "can only broadcast floats or vectors"
        # scalar case
        if isinstance(arg0, float) or arg0.size == 1:
            return np.tile(arg0, shape)
        # vector case
        assert len(shape) == 2, "can only broadcast a vector to a matrix"
        ncols = arg0.shape[0]  # np arrays are rows by default
        assert (
            shape[0] == ncols or shape[1] == ncols
        ), "at least one dim must match input dim"
        if shape[1] == ncols and axis == 0:  # broadcast along dim 0
            return np.tile(arg0, (shape[0], 1))
        return np.tile(arg0.reshape(-1, 1), (1, shape[1]))  # broadcast along dim 1


class MyDimSum(MyOp):
    def compute_op(self, arg0, axis):
        if axis is None:
            return np.sum(arg0)
        assert axis == 0 or axis == 1
        assert len(arg0.shape) == 2
        return np.sum(arg0, axis=axis)


class MyDimMax(MyOp):
    def compute_op(self, arg0, axis):
        if axis is None:
            return np.max(arg0)
        assert axis == 0 or axis == 1
        return np.max(arg0, axis=axis)


class MyTranspose(MyOp):
    def compute_op(self, arg0):
        assert len(arg0.shape) == 2
        return arg0.transpose()


class MyMatmul(MyOp):
    """
    allows only Mv, vM, MM, or vv.
    note that outer prod can be implemented as MM
    """

    def compute_op(self, arg0, arg1):
        shape0, shape1 = arg0.shape, arg1.shape
        assert len(shape0) <= 2 and len(shape1) <= 2
        return np.matmul(arg0, arg1)


# instantiate all the preceding ops, so we can use them to compute our VJPs
get, put, accum = MyGetAtIdx(), MyPutAtIdx(), MyAccum()
add, mult, neg, sub, div = MyAdd(), MyMultiply(), MyNeg(), MySubtract(), MyDivide()
square, tanh, cosh, sinh, exp, log = (
    MySquare(),
    MyTanh(),
    MyCosh(),
    MySinh(),
    MyExp(),
    MyLog(),
)
dimsum, dimmax, bdcast = MyDimSum(), MyDimMax(), MyBroadcast()
mm, T = MyMatmul(), MyTranspose()

## ------------------------------------------- VJPs ------------------------------------

op_to_vjp_func = {}  # maps an op to a function that computes the VJP for its args


def get_vjp_for_arg(argidx, gradout, result, args):
    assert argidx == 0
    alist = args[0].get_val() if isinstance(args[0], Result) else args[0]
    # make a zeroed-out list, which we do not need to track
    zerolist = [
        np.zeros(thing.shape) if isinstance(thing, np.ndarray) else 0 for thing in alist
    ]
    return put(zerolist, args[1], gradout)


op_to_vjp_func[type(get)] = get_vjp_for_arg


def put_vjp_for_arg(argidx, gradout, result, args):
    if argidx == 0:
        alist = args[0].get_val() if isinstance(args[0], Result) else args[0]
        zerolist = [
            np.zeros(thing.shape) if isinstance(thing, np.ndarray) else 0
            for thing in alist
        ]
        return zerolist
    assert argidx == 2
    return get(gradout, args[1])


op_to_vjp_func[type(put)] = put_vjp_for_arg


def add_vjp_for_arg(argidx, gradout, result, args):
    return gradout


op_to_vjp_func[type(add)] = add_vjp_for_arg


def accum_vjp_for_arg(argidx, gradout, result, args):
    raise NotImplementedError()


op_to_vjp_func[type(accum)] = accum_vjp_for_arg


def mult_vjp_for_arg(argidx, gradout, result, args):
    if argidx == 0:
        return mult(gradout, args[1])
    return mult(gradout, args[0])


op_to_vjp_func[type(mult)] = mult_vjp_for_arg


def sub_vjp_for_arg(argidx, gradout, result, args):
    if argidx == 0:
        return gradout
    return neg(gradout)


op_to_vjp_func[type(sub)] = sub_vjp_for_arg


def square_vjp_for_arg(argidx, gradout, result, args):
    return mult(gradout, mult(np.full(gradout.shape, 2.0), args[0]))


op_to_vjp_func[type(square)] = square_vjp_for_arg


def neg_vjp_for_arg(argidx, gradout, result, args):
    return neg(gradout)


op_to_vjp_func[type(neg)] = neg_vjp_for_arg


def div_vjp_for_arg(argidx, gradout, result, args):
    if argidx == 0:
        return mult(gradout, div(np.ones(gradout.shape), args[1]))
    return neg(div(mult(gradout, args[0]), square(args[1])))


op_to_vjp_func[type(div)] = div_vjp_for_arg


def sinh_vjp_for_arg(argidx, gradout, result, args):
    return mult(gradout, cosh(args[0]))


op_to_vjp_func[type(sinh)] = sinh_vjp_for_arg


def cosh_vjp_for_arg(argidx, gradout, result, args):
    return mult(gradout, sinh(args[0]))


op_to_vjp_func[type(cosh)] = cosh_vjp_for_arg


def tanh_vjp_for_arg(argidx, gradout, result, args):
    return mult(gradout, div(1, square(cosh(args[0]))))


op_to_vjp_func[type(tanh)] = tanh_vjp_for_arg


def exp_vjp_for_arg(argidx, gradout, result, args):
    return mult(gradout, result)


op_to_vjp_func[type(exp)] = exp_vjp_for_arg


def log_vjp_for_arg(argidx, gradout, result, args):
    return div(gradout, args[0])


op_to_vjp_func[type(log)] = log_vjp_for_arg


def dimsum_vjp_for_arg(argidx, gradout, result, args):
    assert argidx == 0
    arg0, argaxis = args
    argshape, gshape = arg0.shape, gradout.shape
    if argaxis is not None and not isinstance(argaxis, int):
        argaxis = argaxis.get_val()
    if argaxis is None:  # we summed to a scalar
        return bdcast(gradout, argshape, None)  # axis doesn't matter here
    assert len(gshape) == 1
    return bdcast(gradout, argshape, argaxis)


op_to_vjp_func[type(dimsum)] = dimsum_vjp_for_arg


def dimmax_vjp_for_arg(argidx, gradout, result, args):
    assert argidx == 0
    arg0, argaxis = args
    arg0val = arg0.get_val() if hasattr(arg0, "get_val") else arg0
    argshape, gshape = arg0.shape, gradout.shape
    if argaxis is not None and not isinstance(argaxis, int):
        argaxis = argaxis.get_val()
    if argaxis is None:  # we summed to a scalar
        argmax = np.argmax(arg0val)
        mask = np.zeros(argshape)
        np.put(mask, argmax, 1.0)
        return mult(mask, bdcast(gradout, argshape, None))  # axis doesn't matter here
    assert len(gshape) == 1
    argmaxes = np.argmax(arg0val, axis=argaxis)
    mask = np.zeros(argshape)
    if argaxis == 0:
        mask[argmaxes, np.arange(argshape[1])] = 1.0
    else:
        mask[np.arange(argshape[0]), argmaxes] = 1.0
    return mult(mask, bdcast(gradout, argshape, argaxis))


op_to_vjp_func[type(dimmax)] = dimmax_vjp_for_arg


def bdcast_vjp_for_arg(argidx, gradout, result, args):
    assert argidx == 0
    arg0 = args[0]
    assert (
        isinstance(arg0, float)
        or len(arg0.shape) == 1
        or isinstance(arg0.get_val() if hasattr(arg0, "get_val") else arg0, float)
    ), "could've only broadcast floats or vectors"

    if isinstance(arg0, float) or arg0.size == 1:  # arg0 was a scalar
        return dimsum(gradout, None)

    # arg0 was a vector
    assert len(result.shape) <= 2, "could've broadcast to at most a matrix"
    argaxis = args[2] if isinstance(args[2], int) else args[2].get_val()
    assert len(gradout.shape) == 2
    return dimsum(gradout, argaxis)


op_to_vjp_func[type(bdcast)] = bdcast_vjp_for_arg


def T_vjp_for_arg(argidx, gradout, result, args):
    return T(gradout)


op_to_vjp_func[type(T)] = T_vjp_for_arg


def mm_vjp_for_arg(argidx, gradout, result, args):
    arg0, arg1 = args
    shape0, shape1 = arg0.shape, arg1.shape
    if argidx == 0:
        if len(shape0) == 2 and len(shape1) == 1:  # Mv
            return mult(bdcast(gradout, shape0, 1), bdcast(arg1, shape0, 0))
        if len(shape0) == 2 and len(shape1) == 2:  # MM:
            return mm(gradout, T(arg1))
        if len(shape0) == 1 and len(shape1) == 1:  # vv
            return mult(bdcast(gradout, shape1), arg1)
        return mm(gradout, T(arg1))  # vM

    # argidx is 1
    if len(shape0) == 2 and len(shape1) == 1:  # Mv
        return mm(gradout, arg0)
    if len(shape0) == 2 and len(shape1) == 2:  # MM
        return mm(T(arg0), gradout)
    if len(shape0) == 1 and len(shape1) == 1:  # vv
        return mult(bdcast(gradout, shape0), arg0)
    return mult(bdcast(gradout, shape1, 0), bdcast(arg0, shape1, 1))  # vM


op_to_vjp_func[type(mm)] = mm_vjp_for_arg
