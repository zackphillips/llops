import math
import collections
from numpy import prod
import numpy as np

# operators configuration imports

# Llops imports
from llops import max, min, boundingBox, rand, concatenate, setByteOrder, changeBackend, shape, scalar, rand, dcopy, size, ndim, flip, pad, crop, reshape, alloc, zeros, ones, amax, amin, norm, abs, angle, exp, conj, flip, roll, transpose, matmul, tile, precision, getDatatype, getBackend, real, imag, concatenate, vec, astype
from .operators import Operator, _GradientOperator, Identity
from operator import mul
from functools import reduce
import builtins

# Parallel execution
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool
thread_count_default = multiprocessing.cpu_count()

__all__ = ['Hstack', 'Vstack', 'Dstack', 'VecStack', 'compressStack', 'expandStack', 'VecSplit']

# Flag to use parallel processing of stacks, where applicable
use_parallel = False


class Hstack(Operator):
    """
    Horizontal sum outputs of operators.

    Parameters
    ----------
    operators : arrays of operators with same output shape
    normalize :
    label :
    """

    def __init__(self, operator_list, parallelize=True, thread_count=None,
                 normalize=False, label=None):

        self.stack_op_count = len(operator_list)
        self.normalize = normalize
        assert(self.stack_op_count > 0)

        # Get datatype and backend
        dtype = operator_list[0].dtype
        backend = operator_list[0].backend

        # Store the individual operator input and output sizes
        self.m = operator_list[0].M
        self.n = operator_list[0].N

        # Store parallelization state
        self.parallelize = parallelize
        self.thread_count = thread_count if thread_count is not None else thread_count_default

        # Check linearity and smoothness
        linear = all([op.linear for op in operator_list])
        smooth = all([op.smooth for op in operator_list])

        # Assign label
        if label is None:
            label = 'O_{1 \\times ' + str(self.stack_op_count) + '}'

        # Store indicies
        self.idx = [0] * (self.stack_op_count + 1)

        # Pre-compute map objects for forward and inverse operators
        M = self.m
        N = [0, self.n[-1]] # This variable is the horizontally-stacked M, corresponding to the indicies

        # Check array parameters and populate N
        for i in range(self.stack_op_count):
            assert(operator_list[i].M == self.m)
            assert(operator_list[i].N == self.n)
            assert(operator_list[i].dtype == dtype)
            assert(operator_list[i].backend == backend)
            linear = linear and operator_list[i].linear
            self.idx[i + 1] = self.idx[i] + operator_list[i].N[0]
            N[0] += operator_list[i].N[0]

        # Instantiate metaclass
        super().__init__((M, N), 
                         dtype, 
                         backend, 
                         repr_latex=self._latex,
                         smooth=smooth, forward=self._forward_func,
                         gradient=self._gradient_func,
                         adjoint=self.adjoint_func if linear else None,
                         stack_operators=operator_list,
                         label=label)

    def _latex(self, latex_input=None):
        latex_ret = '\\begin{bmatrix}'
        for index, operator in enumerate(self.stack_operators):
            if callable(operator.repr_latex):
                repr_latex = operator.repr_latex()
            else:
                repr_latex = operator.repr_latex
            if index is not 0:
                latex_ret += ' & '
            latex_ret += (repr_latex)

        latex_ret += '\end{bmatrix}'

        if latex_input is not None:
            latex_ret += (' \\times ' + latex_input)

        return latex_ret

    def _forward_func(self, x, y, m=False):

        # Create iterator
        iterator = lambda i: reshape(self.stack_operators[i] * reshape(x[:], self.N)[self.idx[i]:self.idx[i + 1], :], shape(y))

        # Use the map command to expose parallel nature of this problem
        if self.parallelize:

            # Make the Pool of workers
            pool = ThreadPool(self.thread_count)

            # Run parallal map
            result_parallel = pool.map(iterator, range(self.stack_op_count))

            # Unpack result
            y[:] = sum(result_parallel)

            # Close and join workers
            pool.close()
            pool.join()

        else:
            # Run serial map
            result_serial = map(iterator, range(self.stack_op_count))

            # Unpack result
            y[:] = sum(result_serial)

        if self.normalize:
            y[:] /= self.stack_op_count

    def _gradient_func(self, x=None, inner_operator=None):
        op_list = []

        # Pre-compute O * x if necessary
        Ox = x if inner_operator is None else inner_operator * x

        for i in range(self.stack_op_count):
            op_list.append(self.stack_operators[i]._gradient(inner_operator=None, x=reshape(Ox, self.N)[self.idx[i]:self.idx[i + 1], :]))
            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Vstack(op_list))

    def adjoint_func(self, x, y):
        for i in range(self.stack_op_count):
            y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i].H * x

        if self.normalize:
            y /= self.stack_op_count


class Vstack(Operator):
    """
    Vertical stacking of operators.

    Parameters
    ----------
    operators : arrays of operators with same output shape
    normalize :
    label :
    """

    def __init__(self, operators, parallelize=True, thread_count=None,
                 normalize=False, label=None):
        self.stack_op_count = len(operators)
        self.normalize = normalize
        assert(self.stack_op_count > 0)

        # Get datatype and backend
        dtype = operators[0].dtype
        backend = operators[0].backend

        # Store the individual operator input and output sizes
        self.m = operators[0].M
        self.n = operators[0].N

         # Store parallelization config
        self.parallelize = parallelize
        self.thread_count = thread_count if thread_count is not None else thread_count_default

        # Check linearity and smoothness
        linear = all([op.linear for op in operators])
        smooth = all([op.smooth for op in operators])

        # Assign label
        if label is None:
            label = 'O_{' + str(self.stack_op_count) + ' \\times 1}'

        # Store indicies
        self.idx = [0] * (self.stack_op_count + 1)

        # Check if operators have common bases; if so, assign this and singular values
        condition_number = None

        # Determine condition number
        if all([op.unitary for op in operators]):
            # All operator are unitary
            condition_number = 1.0
        else:
            condition_number = np.inf

        # Store stacked sizes (N will be populated in the loop below)
        N = self.n
        M = [0, self.m[1]]  # This variable is the horizontally-stacked M, corresponding to the indicies

        # Check array parameters and populate N
        for i in range(self.stack_op_count):
            assert(operators[i].M == self.m)
            assert(operators[i].N == self.n)
            assert(operators[i].dtype == dtype)
            assert(operators[i].backend == backend)
            linear = linear and operators[i].linear
            self.idx[i + 1] = self.idx[i] + operators[i].M[0]
            M[0] += operators[i].M[0]

        # Define adjoint if all operators are linear
        adjoint_func, inverse_func = None, None

        if linear:
            adjoint_func = self._adjoint_func

        # Define inverse iff operators are non-linear and invertable
        if builtins.all([op.invertable and not op.linear for op in operators]):
            inverse_func = self._inverse_func

        # Instantiate metaclass
        super().__init__((M, N), dtype, backend, repr_latex=self._latex,
                                             forward=self._forward_func, 
                                             adjoint=adjoint_func,
                                             gradient=self._gradient_func,
                                             inverse=inverse_func,
                                             smooth=smooth,
                                             stack_operators=operators,
                                             condition_number=condition_number,
                                             label=label)

    def _latex(self, latex_input=None):
        latex_ret = '\\begin{bmatrix}'
        if len(self.stack_operators) < 5:
            for index, operator in enumerate(self.stack_operators):
                if callable(operator.repr_latex):
                    repr_latex = operator.repr_latex()
                else:
                    repr_latex = operator.repr_latex

                if index is not 0:
                    latex_ret += ' \\cr '
                latex_ret += (repr_latex)
        else:
            for index, operator in enumerate(self.stack_operators[:2]):
                if callable(operator.repr_latex):
                    repr_latex = operator.repr_latex()
                else:
                    repr_latex = operator.repr_latex

                if index is not 0:
                    latex_ret += ' \\cr '
                latex_ret += (repr_latex)

            latex_ret += '\\cr \\vdots \\cr'

            for index, operator in enumerate(self.stack_operators[-2:]):
                if callable(operator.repr_latex):
                    repr_latex = operator.repr_latex()
                else:
                    repr_latex = operator.repr_latex

                if index is not 0:
                    latex_ret += ' \cr '
                latex_ret += (repr_latex)

        latex_ret += '\end{bmatrix}'

        if latex_input is not None:
            latex_ret += (' \\times ' + latex_input)
        return latex_ret

    def _forward_func(self, x, y):

        # Define iterator
        iterator = lambda i: self.stack_operators[i] * dcopy(x)

        # Use the map command to expose parallel nature of this problem
        if self.parallelize:

            # Make the Pool of workers
            pool = ThreadPool(self.thread_count)

            # Run thread across parallel map
            result_parallel = pool.map(iterator, range(self.stack_op_count))

            # Collect result
            for i, result in enumerate(result_parallel):
                y[self.idx[i]:self.idx[i + 1], :] = result

            # Close and join workers
            pool.close()
            pool.join()
        else:
            # Run thread across parallel map
            result_serial = map(iterator, range(self.stack_op_count))

            # Collect result
            for i, result in enumerate(result_serial):
                y[self.idx[i]:self.idx[i + 1], :] = result

        if self.normalize:
            y /= self.stack_op_count

    def _gradient_func(self, x=None, inner_operator=None):
        # Generate gradient operator
        op_list = []
        for i in range(self.stack_op_count):
            op_list.append(self.stack_operators[i]._gradient(inner_operator=inner_operator, x=x))

            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Hstack(op_list))

    def _adjoint_func(self, x, y):
        y[:] = sum([reshape(self.stack_operators[i].H * x[self.idx[i]:self.idx[i + 1], :], shape(y)) for i in range(self.stack_op_count)])

        if self.normalize:
            y[:] /= self.stack_op_count

    def _inverse_func(self, x, y):
        """Inverse (Used on y for non-linear operators)."""
        y[:] = sum([reshape(self.stack_operators[i].inv * x[self.idx[i]:self.idx[i + 1], :], shape(y)) for i in range(self.stack_op_count)])

        if self.normalize:
            y[:] /= self.stack_op_count

class Dstack(Operator):
    """Diagonal stack of operators.
    Parameters
    ----------
    operators : arrays of operators
    """

    def __init__(self, operators, parallelize=True, thread_count=None,
                 normalize=False, label=None):
        self.stack_op_count = len(operators)
        self.normalize = normalize
        assert(self.stack_op_count > 0)

        # Get datatype and backend
        self.stack_operators = operators
        dtype = operators[0].dtype
        backend = operators[0].backend

        # Store the individual operator input and output sizes
        self.m = operators[0].M
        self.n = operators[0].N

        # Store parallelization config
        self.parallelize = parallelize
        self.thread_count = thread_count if thread_count is not None else thread_count_default

        # Check linearity and smoothness
        linear = all([op.linear for op in operators])
        smooth = all([op.smooth for op in operators])
        invertable = all([op.invertable for op in operators])

        if label is None:
            label = 'O_{' + str(self.stack_op_count) + ' \\times ' + str(self.stack_op_count) + '}'

        # Determine input array indexing for diagonal operators
        self.idx = [0] * (self.stack_op_count + 1)

        # Check if operators have common bases; if so, assign this and singular values
        condition_number, invertable = None, None

        # Determine condition number
        if builtins.all([op.unitary for op in operators]):
            # All operator are unitary
            condition_number = 1.0
        elif builtins.all([op.invertable for op in operators]):
            # No valid way to store condition number, but this operator is invertable
            condition_number = None
            invertable = True
        else:
            condition_number = np.inf

        # Store stacked sizes (N will be populated in the loop below)
        M = [0, self.m[1]]  # This variable is the diagonally-stacked M, corresponding to the indicies
        N = [0, self.n[1]]  # This variable is the diagonally-stacked M, corresponding to the indicies

        # Check array parameters and populate N
        for i in range(self.stack_op_count):
            assert(operators[i].M == self.m)
            assert(operators[i].N == self.n)
            assert(operators[i].dtype == dtype)
            assert(operators[i].backend == backend)
            linear = linear and operators[i].linear
            self.idx[i + 1] = self.idx[i] + operators[i].N[0]
            self.idx[i + 1] = self.idx[i] + operators[i].M[0]
            M[0] += operators[i].M[0]
            N[0] += operators[i].N[0]

        # Define adjoint if all operators are linear
        adjoint_func, inverse_func = None, None
        if linear:
            adjoint_func = self._adjoint_func
        if invertable:
            inverse_func = self._inverse_func

        # Instantiate metaclass
        super().__init__((M, N), dtype, backend, repr_latex=self._latex,
                                             smooth=smooth,
                                             forward=self._forward_func,
                                             gradient=self._gradient_func,
                                             adjoint=adjoint_func,
                                             inverse=inverse_func,
                                             stack_operators=operators,
                                             condition_number=condition_number,
                                             label=label)

    def _latex(self, latex_input=None):
        latex_ret = '\\begin{bmatrix}'
        if len(self.stack_operators) < 5:
            for index, operator in enumerate(self.stack_operators):
                if callable(operator.repr_latex):
                    repr_latex = operator.repr_latex()
                else:
                    repr_latex = operator.repr_latex

                for _ in range(index):
                    latex_ret += ' 0 & '

                latex_ret += repr_latex

                for _ in range(self.stack_op_count-index - 1):
                    latex_ret += ' & 0'

                latex_ret += ' \\cr '
        else:
            # Add top row
            operator = self.stack_operators[0]
            if callable(operator.repr_latex):
                repr_latex = operator.repr_latex()
            else:
                repr_latex = operator.repr_latex

            repr_latex += ' & \\ldots & 0 \\cr '
            latex_ret += repr_latex

            # Add middle row
            latex_ret +=  ' \\vdots & \\ddots & \\vdots \\cr '

            # Add bottom row
            operator = self.stack_operators[-1]
            if callable(operator.repr_latex):
                repr_latex = operator.repr_latex()
            else:
                repr_latex = operator.repr_latex

            repr_latex = '0 & \\ldots & ' + repr_latex + ' \\cr '
            latex_ret += repr_latex

        latex_ret += '\\end{bmatrix}'

        if latex_input is not None:
            latex_ret += (' \\times ' + latex_input)

        return latex_ret


    def _forward_func(self, x, y):

        # Define iterator
        iterator = lambda i: self.stack_operators[i] * dcopy(x[self.idx[i]:self.idx[i + 1], :])

        # Use the map command to expose parallel nature of this problem
        if self.parallelize:

            # Make the Pool of workers
            pool = ThreadPool(self.thread_count)

            # Run thread across parallel map
            result_parallel = pool.map(iterator, range(self.stack_op_count))

            # Collect result
            for i, result in enumerate(result_parallel):
                y[self.idx[i]:self.idx[i + 1], :] = result

            # Close and join workers
            pool.close()
            pool.join()

        else:

            # Run thread across serial map
            result_serial = map(iterator, range(self.stack_op_count))

            # Collect result
            for i, result in enumerate(result_serial):
                y[self.idx[i]:self.idx[i + 1], :] = result

        if self.normalize:
            y /= self.stack_op_count

    def _adjoint_func(self, x, y):
        for i in range(self.stack_op_count):
            y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i].H * dcopy(x[self.idx[i]:self.idx[i + 1], :])

        if self.normalize:
            y /= self.stack_op_count

    def _gradient_func(self, x=None, inner_operator=None):
        # Generate gradient operator
        op_list = []

        # Loop over stack operators
        for i in range(self.stack_op_count):
            # Get subarray of input
            _x = x[self.idx[i]:self.idx[i + 1], :]

            # Append gradient to list
            op_list.append(self.stack_operators[i]._gradient(inner_operator=inner_operator, x=_x))

            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Dstack(op_list))

    def _inverse_func(self, x, y):
        # print(shape(y))
        for i in range(self.stack_op_count):
            y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i].inv * dcopy(x[self.idx[i]:self.idx[i + 1], :])

        # Normalize, if requested
        if self.normalize:
            y /= self.stack_op_count

def VecStack(vector_list, axis=0):
    """ This is a helper function to stack vectors """

    # Determine output size
    single_vector_shape = [max([shape(vector)[0] for vector in vector_list]), max([shape(vector)[1] for vector in vector_list])]
    vector_shape = dcopy(single_vector_shape)
    vector_shape[axis] *= len(vector_list)

    # Allocate memory
    vector_full = zeros(vector_shape, getDatatype(vector_list[0]), getBackend(vector_list[0]))

    # Assign vector elements to the output
    for index, vector in enumerate(vector_list):
        vector_full[index * single_vector_shape[0]:(index + 1) * single_vector_shape[0], :] = pad(vector, single_vector_shape)

    # Return
    return vector_full


def VecSplit(vector, count, axis=0):
    _shape = shape(vector)
    _step = _shape[axis] // count

    if axis == 0:
        return [vector[index * _step: (index + 1) * _step, :] for index in range(count)]
    elif axis == 1:
        return [vector[:, index * _step: (index + 1) * _step] for index in range(count)]


def compressStack(A):
    op_list = []
    for operator_index in range(len(A.suboperators[0].stack_operators)):
        op_list.append(reduce(mul, [op.stack_operators[operator_index] for op in A.suboperators]))

    # Form stack and return
    if 'Vstack' in str(A.suboperators):
        return Vstack(op_list)
    elif 'Hstack' in str(A.suboperators):
        return Hstack(op_list)
    else:
        return Dstack(op_list)

def expandStack(A):

    # Get and check stack size to ensure it's compressed
    stack_size = len(A.suboperators)

    # If stack is partially compressed, go ahead and fully compres it
    if stack_size > 1:
        A = compressStack(A)

    assert stack_size == 1

    # Expand Operator lists
    op_list = []
    for op in A.stack_operators:
        op_list.append(op.suboperators)

    # Transpose list
    op_list = list(map(list, zip(*op_list)))

    if 'Vstack' in str(A):
        # Process all but the last element
        D_list = []
        for operators in op_list[:-1]:
            D_list.append(Dstack(operators))
        expanded_op = reduce(mul, D_list)

        # Process last op
        expanded_op *= Vstack(op_list[-1])
    elif 'Hstack' in str(A):
        # Process all but first element
        D_list = []
        for operators in op_list[1:]:
            D_list.append(Dstack(operators))
        expanded_op = reduce(mul, D_list)

        # Process last op
        expanded_op = Hstack(op_list[-1]) * expanded_op
    else:
        # Process all elements (Diagional operator)
        D_list = []
        for operators in op_list:
            D_list.append(Dstack(operators))
        expanded_op = reduce(mul, D_list)

    return expanded_op
