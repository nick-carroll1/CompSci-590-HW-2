import numpy as np

from ops_and_vjps import (get, put, accum,
                          add, mult, neg, sub, div,
                          square, tanh, cosh, sinh, exp, log,
                          dimsum, dimmax, bdcast,
                          mm, T,
                          Result,
                          op_to_vjp_func)


class AutoDiffer:
    def __init__(self):
        self.trace_level = -1 # global counter of the number of functions we're currently tracing

    @staticmethod
    def _topo_order_parents(end_result, level):
        """
        iterates over the parent Results of end_result which have level=level,
        in topological order
        """
        child_counts = {} # nchildren of each node
        stack = [end_result]
        while stack:
            node = stack.pop()
            if node in child_counts:
                child_counts[node] += 1
            else:
                child_counts[node] = 1
                stack.extend([par for idx, par in node.get_bprop_parents(level)])

        childless_nodes = [end_result]
        while childless_nodes:
            node = childless_nodes.pop()
            yield node
            for parent in [par for idx, par in node.get_bprop_parents(level)]:
                if child_counts[parent] == 1:
                    childless_nodes.append(parent)
                else:
                    child_counts[parent] -= 1

    def _backward(self, gradout, end_result, level):
        """
        gradout - an ndarray of size (1,)
        end_result - a scalar Result
        level - the trace level at which to backprop
        returns: 
          Result representing the gradient of end_result wrt the first arg of the 
          function that computed it
        """
        gradouts = {end_result: gradout} # maps result -> grad of loss wrt result
        # topologically sort nodes in the computation graph at the same level
        topo_sorted = list(AutoDiffer._topo_order_parents(end_result, level))
        # loop through computation graph in topological order
        for result in topo_sorted:
            gradout = gradouts[result]
            del gradouts[result]
            # decrement this result's level, so downstream results also have a lower level
            result.trace_levels.remove(level)
            # get all the parents (i.e., arguments) that made this result
            # also decrement their levels so downstream results have a lower level
            all_parents = result.get_all_parents(level)
            # update grad of loss wrt each parent/argument, by computing the corresponding VJP
            for par_argidx, par in result.get_bprop_parents(level):
                # compute the VJP for parent par
                par_vjp = op_to_vjp_func[type(result.op)](par_argidx, gradout, result, all_parents)
                # add this vjp to the previously computed vjps for this parent, if any
                if par not in gradouts:
                    gradouts[par] = par_vjp
                else:
                    gradouts[par] = accum(gradouts[par], par_vjp)
        return gradout


    def make_grad_and_val_func(self, f):
        """
        f - a function that computes a scalar value from its args, using ops from ops_and_vjps
        returns:
          a new function g such that g(args) returns the grad of f wrt args[0], and f(args)
        """
        def grad_and_val_func(arg0, *args):
            self.trace_level += 1 # increment global trace to indicate we're tracing a new function
            if isinstance(arg0, Result): # if arg0 is a Result it's already being traced
                init_result = arg0
                init_result.trace_levels.add(self.trace_level) # add new trace to its levels
            else: # need to turn arg0 into a Result so we can trace it
                init_result = Result(arg0, None, [], {-1, self.trace_level})
            # trace through f using init_result
            end_result = f(init_result, *args)
            if end_result is None:
                assert False
            assert end_result.get_val().size == 1, "Can only take grads of scalar functions!!!"
            grad = self._backward(np.ones((1)), end_result, self.trace_level)
            self.trace_level -= 1
            if max(end_result.trace_levels) == -1:
                grad = grad.get_val()
            return grad, end_result.get_val()

        return grad_and_val_func
