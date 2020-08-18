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

    def __init__(self, operators, normalize=False, label=None):
        self.stack_op_count = len(operators)
        self.normalize = normalize
        assert(self.stack_op_count > 0)

        # Get datatype and backend
        dtype = operators[0].dtype
        backend = operators[0].backend

        # Store the individual operator input and output sizes
        self.m = operators[0].M
        self.n = operators[0].N

        # Check linearity and smoothness
        linear = all([op.linear for op in operators])
        smooth = all([op.smooth for op in operators])

        # Assign label
        if label is None:
            label = 'O_{1 \\times ' + str(self.stack_op_count) + '}'

        # Store indicies
        self.idx = [0] * (self.stack_op_count + 1)

        # Pre-compute map objects for forward and inverse operators
        M = self.m
        N = [0, self.n[-1]] # This variable is the horizontally-stacked M, corresponding to the indicies
        set_argument_function_dict = {}

        # Check array parameters and populate N
        for i in range(self.stack_op_count):
            assert(operators[i].M == self.m)
            assert(operators[i].N == self.n)
            assert(operators[i].dtype == dtype)
            assert(operators[i].backend == backend)
            linear = linear and operators[i].linear
            self.idx[i + 1] = self.idx[i] + operators[i].N[0]
            N[0] += operators[i].N[0]

        # Define adjoint if all operators are linear
        if linear:
            adjoint = self._adjoint
        else:
            adjoint = None

        super().__init__((M, N), dtype, backend, repr_latex=self._latex,
                                             smooth=smooth, forward=self._forward,
                                             gradient=self._gradient,
                                             adjoint=adjoint,
                                             # get_arguments_function=self._getArguments,
                                             # set_arguments_function=self._setArguments,
                                             stack_operators=operators,
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

    def _forward(self, x, y, m=False):
        # Set output to zero
        # y[:] = 0

        # Use the map command to expose parallel nature of this problem
        if m:
            iterator = lambda i: reshape(self.stack_operators[i] * reshape(x[:], self.N)[self.idx[i]:self.idx[i + 1], :], shape(y))
            y[:] = sum(map(iterator, range(self.stack_op_count)))
        else:
            y[:] = sum([reshape(self.stack_operators[i] * reshape(x[:], self.N)[self.idx[i]:self.idx[i + 1], :], shape(y)) for i in range(self.stack_op_count)])

        if self.normalize:
            y[:] /= self.stack_op_count

    def _gradient(self, x=None, inner_operator=None):
        op_list = []

        # Pre-compute O * x if necessary
        Ox = x if inner_operator is None else inner_operator * x

        for i in range(self.stack_op_count):
            op_list.append(self.stack_operators[i]._gradient(inner_operator=None, x=reshape(Ox, self.N)[self.idx[i]:self.idx[i + 1], :]))
            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Vstack(op_list))

    def _adjoint(self, x, y):
        # Parallel examples:
        # https://pythonhosted.org/joblib/parallel.html
        if False: # getBackend(x) == 'arrayfire': # Doesn't work yet due to operator indexing
            import arrayfire as af
            for i in af.index.ParallelRange(self.stack_op_count):
                y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[int(i)].H * x
        else:
            if use_parallel:
                for i in range(self.stack_op_count):
                    y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i].H * x
            else:
                for i in range(self.stack_op_count):
                    y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i].H * x

        if self.normalize:
            y /= self.stack_op_count

    def _getArguments(self):
        return prod([op.arguments for op in self.stack_operators])

    def _setArguments(self, new_arguments):
        for index, argument in new_arguments:
            op_index = np.floor(index / self.stack_op_count)
            arg_index = index % self.stack_op_count
            args = [None] * len(self.stack_operators[op_index].arguments)
            args[arg_index] = argument
            self.stack_operators[op_index] = args


class Vstack(Operator):
    """
    Vertical stacking of operators.

    Parameters
    ----------
    operators : arrays of operators with same output shape
    normalize :
    label :
    """

    def __init__(self, operators, normalize=False, label=None):
        self.stack_op_count = len(operators)
        self.normalize = normalize
        assert(self.stack_op_count > 0)

        # Get datatype and backend
        dtype = operators[0].dtype
        backend = operators[0].backend

        # Store the individual operator input and output sizes
        self.m = operators[0].M
        self.n = operators[0].N

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
        adjoint, inverse = None, None

        if linear:
            adjoint = self._adjoint

        # Define inverse iff operators are non-linear and invertable
        if builtins.all([op.invertable and not op.linear for op in operators]):
            inverse = self._inverse

        # Instantiate metaclass
        super().__init__((M, N), dtype, backend, repr_latex=self._latex,
                                             forward=self._forward, adjoint=adjoint,
                                             gradient=self._gradient, inverse=inverse,
                                             stack_operators=operators,
                                             condition_number=condition_number,
                                             set_arguments_function=self._setArguments,
                                             get_arguments_function=self._getArguments,
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

    def _forward(self, x, y):
        if False: # getBackend(x) == 'arrayfire': # doesnt work yet due to operator indexing
            import arrayfire as af
            for i in af.index.ParallelRange(self.stack_op_count):
                y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[int(i)] * x
        else:
            for i in range(self.stack_op_count):
                y[self.idx[i]:self.idx[i + 1], :] = self.stack_operators[i] * x

        if self.normalize:
            y /= self.stack_op_count

    def _gradient(self, x=None, inner_operator=None):
        # Generate gradient operator
        op_list = []
        for i in range(self.stack_op_count):
            op_list.append(self.stack_operators[i]._gradient(inner_operator=inner_operator, x=x))

            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Hstack(op_list))

    def _adjoint(self, x, y):
        y[:] = sum([reshape(self.stack_operators[i].H * x[self.idx[i]:self.idx[i + 1], :], shape(y)) for i in range(self.stack_op_count)])

        if self.normalize:
            y[:] /= self.stack_op_count

    def _inverse(self, x, y):
        """Inverse (Used on y for non-linear operators)."""
        y[:] = sum([reshape(self.stack_operators[i].inv * x[self.idx[i]:self.idx[i + 1], :], shape(y)) for i in range(self.stack_op_count)])

        if self.normalize:
            y[:] /= self.stack_op_count

    def _getArguments(self):
        arguments = []
        for stack_operator in self.stack_operators:
            arguments.append(stack_operator.arguments)
        return arguments

    def _setArguments(self, new_arguments):

        assert len(new_arguments) == len(self.stack_operators)
        for index, op in enumerate(self.stack_operators):
            if new_arguments[index] is not None:
                op.arguments = new_arguments[index]

class Dstack(Operator):
    """Diagonal stack of operators.
    Parameters
    ----------
    operators : arrays of operators
    """

    def __init__(self, operators, normalize=False, label=None):
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

        # Check linearity and smoothness
        linear = all([op.linear for op in operators])
        smooth = all([op.smooth for op in operators])
        invertable = all([op.invertable for op in operators])

        if label is None:
            label = 'O_{' + str(self.stack_op_count) + ' \\times ' + str(self.stack_op_count) + '}'

        # Determine input array indexing for diagonal operators
        self.idx_in = [0] * (self.stack_op_count + 1)
        self.idx_out = [0] * (self.stack_op_count + 1)

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
            self.idx_in[i + 1] = self.idx_in[i] + operators[i].N[0]
            self.idx_out[i + 1] = self.idx_out[i] + operators[i].M[0]
            M[0] += operators[i].M[0]
            N[0] += operators[i].N[0]

        # Define adjoint if all operators are linear
        adjoint, inverse = None, None
        if linear:
            adjoint = self._adjoint
        if invertable:
            inverse = self._inverse

        # Instantiate metaclass
        super().__init__((M, N), dtype, backend, repr_latex=self._latex,
                                             smooth=smooth,
                                             forward=self._forward,
                                             gradient=self._gradient,
                                             adjoint=adjoint,
                                             inverse=inverse,
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


    def _forward(self, x, y):
        for i in range(self.stack_op_count):
            y[self.idx_out[i]:self.idx_out[i + 1], :] = self.stack_operators[i] * dcopy(x[self.idx_in[i]:self.idx_in[i + 1], :])

        if self.normalize:
            y /= self.stack_op_count

    def _adjoint(self, x, y):
        for i in range(self.stack_op_count):
            y[self.idx_in[i]:self.idx_in[i + 1], :] = self.stack_operators[i].H * dcopy(x[self.idx_out[i]:self.idx_out[i + 1], :])

        if self.normalize:
            y /= self.stack_op_count

    def _gradient(self, x=None, inner_operator=None):
        # Generate gradient operator
        op_list = []

        # Loop over stack operators
        for i in range(self.stack_op_count):
            # Get subarray of input
            _x = x[self.idx_in[i]:self.idx_in[i + 1], :]

            # Append gradient to list
            op_list.append(self.stack_operators[i]._gradient(inner_operator=inner_operator, x=_x))

            if self.normalize:
                op_list[-1] /= self.stack_op_count

        return _GradientOperator(Dstack(op_list))

    def _inverse(self, x, y):
        # print(shape(y))
        for i in range(self.stack_op_count):
            y[self.idx_in[i]:self.idx_in[i + 1], :] = self.stack_operators[i].inv * dcopy(x[self.idx_out[i]:self.idx_out[i + 1], :])

        # Normalize, if requested
        if self.normalize:
            y /= self.stack_op_count

    def _getArguments(self):
        return builtins.sum([op.arguments for op in self.stack_operators])

    def _setArguments(self, new_arguments):
        for index, argument in new_arguments:
            op_index = np.floor(index / self.stack_op_count)
            arg_index = index % self.stack_op_count
            args = [None] * len(self.stack_operators[op_index].arguments)
            args[arg_index] = argument
            self.stack_operators[op_index] = args


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
