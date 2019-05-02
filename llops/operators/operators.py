"""
Copyright 2017 Zachary Phillips, Waller Lab, University of California, Berkeley

Redistribution and use in source and binary forms, with or without modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its contributors may be used to endorse or promote products derived from this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""

import numpy as np
import scipy as sp
from scipy.fftpack import next_fast_len
from numpy import isscalar, prod, inf, ndarray
import math
import random
import pywt
import builtins

# Llops imports
from llops import asarray, max, min, sign, gradientCheck, roll, zeros_like, fftshift, circshift, dealloc, getNativeDatatype, changeBackend, shape, scalar, rand, dcopy, size, ndim, flip, pad, crop, reshape, alloc, zeros, ones, amax, amin, norm, abs, angle, exp, conj, roll, transpose, matmul, sum, tile, precision, getDatatype, getBackend, real, vec, astype, cos, sin, isComplex, crop_roi, Roi, pad_roi
from llops.fft import fftfuncs
from llops.filter import softThreshold
from llops import config
import llops as yp

__all__ = ['Identity', 'Diagonalize', 'Convolution',
           'MatrixMultiply', 'FourierTransform', 'WaveletTransform',
           'Intensity', 'Power', 'Flip', 'Crop', 'Sum', 'Shift', 'Segmentation',
           'Exponential',  'Intensity',  'L2Norm', 'L1Norm', 'Operator', 'PhaseRamp']

explicit_identity = False
WARN_FOR_EXPENSIVE_OPERATIONS = True

class Operator(object):
    """
    Abstraction for a generic operator (linear or non-linear).
    This could include a linear operator, nonlinear operator, or norm.

    This class should not be instantiated directly; it is a metaclass.
    """

    def __init__(self, shape, dtype, backend, repr_str=None, label=None,
                 forward=None, adjoint=None, gradient=None, proximal=None,
                 inverse=None, condition_number=None, cost=None,
                 inner_operator=None, outer_operator=None,
                 condition_number_is_upper_bound=False, convex=False,
                 smooth=True, set_arguments_function=None,
                 get_arguments_function=None,
                 inverse_regularizer=0.0, repr_latex=None,
                 stack_operators=None, non_smooth_part=None,
                 smooth_part=None):

        # Store operator parameters
        self.ndim = len(shape)
        self.dtype = dtype
        self.backend = backend

        # self.shape is the overall shape of the operator (M, N)
        # self.M is the local shape of the output
        # self.N is the local shape of the input
        self.shape = tuple(shape)

        # Store component dimensions
        self.M = shape[0] if not hasattr(shape[0], '__iter__') else tuple(shape[0])
        self.N = shape[1] if not hasattr(shape[1], '__iter__') else tuple(shape[1])

        # Define whether or not a forward model is smooth; if smooth is not provided, set based on the gradient and adjoint
        if smooth is None:
            self.smooth = not (gradient is None and adjoint is None)
        else:
            self.smooth = smooth

        # Set default label representation (may change for each instance of this class)
        if label is None:
            self.label = 'A'
        else:
            self.label = label

        # Set default string representation (for technical printing)
        if repr_str is None:
            self.repr_str = self.__class__.__name__
        else:
            self.repr_str = repr_str

        # Set default string representation (for mathematical printing)
        if repr_latex is None:
            def repr_latex(latex_input=None):
                if not latex_input:
                    return self.label
                else:
                    return self.label + ' \\times ' + latex_input
            self.repr_latex = repr_latex
        else:
            self.repr_latex = repr_latex

        # These will be defined by sub-functions if they're used
        self._forward = forward
        self._adjoint = adjoint
        self._gradient = gradient
        self._proximal = proximal

        # Store condition_number and whether this is an upper bound
        self.condition_number = condition_number
        self.condition_number_is_upper_bound = condition_number_is_upper_bound

        # If system is linear but condition number is not provided,
        # are provided, set conditon number to infinity
        if self._adjoint and self.condition_number is None:
            self.condition_number = np.inf

        # This is used for inverses if the operator is not unitary
        self._inverse_regularizer = inverse_regularizer

        # Store stack operators if provided
        self.stack_operators = stack_operators

        # Store inner operator if one is provided
        # (this includes all operators except for the left-most operator)
        self.inner_operators = inner_operator

        # Store outer operator if one is provided
        # (this is the left-most operator)
        self.outer_operator = outer_operator

        # Store cost
        if cost is None:
            self.cost = prod(self.shape[1])
        else:
            self.cost = cost

        # If the inverse is not specified but the condition number is 1,
        # the adjoint is the inverse since the operator is unitary
        if inverse is None and self._adjoint is not None and self.condition_number == 1.:
            self._inverse = self._adjoint
        elif inverse is not None:
            self._inverse = inverse
        else:
            self._inverse = None

        # If adjoint is specified (function is linear) and gradient is not,
        # set the gradient operator to the adjoint times the input
        if self._adjoint is not None and self._gradient is None:
            self._gradient = self._adjointToGradient
        else:
            self._gradient = gradient

        # Define convexity
        if self.linear:
            self.convex = True
        else:
            self.convex = convex

        # Define set and get arguments functions
        self._setArgumentsFunction = set_arguments_function
        self._getArgumentsFunction = get_arguments_function

        if self._setArgumentsFunction is not None:
            assert self._getArgumentsFunction, "Both set and get arguments functions must be defined!"

        # Define smooth and non-smooth functions
        if (non_smooth_part is not None) and (smooth_part is not None):
            assert smooth_part.smooth, "Provided smooth part is not smooth!"
            assert not non_smooth_part.smooth, "Provided non-smooth part is smooth!"
            self.non_smooth_part = non_smooth_part
            self.smooth_part = smooth_part
        else:
            if self.smooth:
                self.non_smooth_part = None
                self.smooth_part = self._returnSelf()
            else:
                self.smooth_part = None
                self.non_smooth_part = self._returnSelf()

    @property
    def suboperators_with_arguments(self):
        editable_op_list = []
        if len(self.suboperators) > 1:
            for op in self.suboperators:
                if op.arguments is not None:
                    editable_op_list.append(op)
        return editable_op_list

    @property
    def arguments(self):
        if self._getArgumentsFunction:
            return self._getArgumentsFunction()
        elif self.composite and not self.is_stack:
            # print('Operator %s is composite - please use the .suboperators property to index specific operators.' % self.repr_str)
            return None
        elif self.is_stack:
            arguments = []
            for stack_operator in self.stack_operators:
                arguments.append(stack_operator.arguments)
            return arguments
        else:
            return None

    @arguments.setter
    def arguments(self, new_arguments):
        if self._setArgumentsFunction:
            return self._setArgumentsFunction(new_arguments)
        elif self.composite:
            # print('Operator %s is composite - please use the .suboperators property to index specific operators.' % self.repr_str)
            pass
        elif self.is_stack:
            assert len(new_arguments) == len(self.stack_operators)
            for index, op in enumerate(self.stack_operators):
                if new_arguments[index] is not None:
                    op.arguments = new_arguments[index]
        else:
            raise ValueError('Argument not defined for operator %s' % self)

    # Pass-through function to convert the adjoint to a gradient
    def _adjointToGradient(self, x=None, inside_operator=None):

        # Strip the extra methods from the adjoint operator
        G = _GradientOperator(self.adjoint())

        # Process latex rendering
        latex_str = self.repr_latex()
        if '^H' not in latex_str:
            """Current operator is not an adjoint """
            def repr_latex(latex_input=None):
                if not latex_input:
                    return latex_str + '^H'
                else:
                    return latex_str + '^H' + ' \\times ' + latex_input
        else:
            """Current operator is an adjoint """
            def repr_latex(latex_input=None):
                if not latex_input:
                    return latex_str.split('^H')[0]
                else:
                    return latex_str.split('^H')[0] + ' \\times ' + latex_input
        G.repr_latex = repr_latex
        return G

    def _returnSelf(self):
        return self

    @property
    def nops(self):
        return len(self.suboperators)

    # Pass-through function to convert the adjoint to a inverse
    def _adjointToInverse(self, x, y):
        y[:] = self.adjoint() * x

    def latex(self, gradient=False):
        from IPython.display import Latex, display

        if gradient:
            x = zeros(self.N, self.dtype, self.backend)
            display(Latex('$ \\nabla_x' + self.label + '(\\vec{x}) = ' + self.gradient(x, return_op=True).repr_latex('\\vec{x}') + '$'))
        else:
            display(Latex('$' + self.label + '(\\vec{x}) = ' + self.repr_latex('\\vec{x}') + '$'))
        # else:
        #     display(Latex('$' + self.label + '(\\vec{x}) = ' + self.repr_latex + ' \\times \\vec{x} $'))

    def forward(self, x, y=None):
        """
        Evaluate the forward operation, or return a new composite operator if the input is another operator.
        This method is defined for ALL operators.
        """
        if hasattr(x, "forward") and hasattr(x, "adjoint"):
            """Operator Product """
            return _ProductOperator(self, x)

        elif isscalar(x):
            """Scalar Product """
            return _ScaledOperator(self, x)

        elif x is None:
            """Input is None """
            return self  # Treat as identity

        else:
            # Store whether we should return y or not
            return_y = y is None

            """Dense Matrix/Vector Product """
            if getDatatype(x) != self.dtype:
                raise ValueError('Input data type mismatch, for {}, got {}'.format(self, getDatatype(x)))

            if getBackend(x) != self.backend:
                raise ValueError('Input backend mismatch, for {}, got {}'.format(self, getBackend(x)))

            if size(x) != prod(self.shape[1]):
                raise ValueError('Input size mismatch, for {}, got {}'.format(self, shape(x)))

            # Ensure x is the correct size
            x = reshape(x, self.N)

            # Initialize output
            if y is None:
                y = alloc(self.M, self.dtype, self.backend)
            else:
                # Check backend and dtype of provided output
                if getDatatype(y) != self.dtype:
                    raise ValueError('Output dtype mismatch, for {}, got {}'.format(self, getDatatype(x)))
                if getBackend(y) != self.backend:
                    raise ValueError('Output backend mismatch, for {}, got {}'.format(self, getBackend(x)))

            # Call operator forward function
            self._forward(x, y)

            # Optionally return y, if one was not provided as a keyword arguments
            if return_y:
                return y

    def adjoint(self):
        """
        Return a new operator which is the adjoint of the forward operator
        """
        if self._adjoint is not None:
            if '.H' in self.repr_str:
                repr_str = self.repr_str.split('.H')[0]
                def repr_latex(latex_input=None):
                    return self.repr_latex(latex_input=latex_input)
            else:
                repr_str = self.repr_str + '.H'
                def repr_latex(latex_input=None):
                    return self._gradient(None).repr_latex(latex_input=latex_input)

            return Operator((self.N, self.M),
                            self.dtype, self.backend,
                            forward=self._adjoint,
                            adjoint=self._forward,
                            label=self.label,
                            cost=self.cost,
                            get_arguments_function=self._getArgumentsFunction,
                            set_arguments_function=self._setArgumentsFunction,
                            repr_latex=repr_latex,
                            condition_number=self.condition_number,
                            condition_number_is_upper_bound=self.condition_number_is_upper_bound,
                            repr_str=repr_str)
        else:
            raise NotImplementedError('The adjoint operator is not defined for function %s' % self.repr_str)

    def inverse(self, inverse_regularizer=None, force=False):
        """
        Return a new operator which is the inverse of the forward operator
        """
        if self.invertable or force:

            # Set inverse_regularizer if provided
            if inverse_regularizer is not None:
                self.inverse_regularizer = inverse_regularizer

            # Get labels and reverse order for latex printing
            labels = list(reversed(self.repr_latex().split(' \\times ')))

            # Reverse order
            latex = ''
            for index, label in enumerate(reversed(labels)):
                if '^{-1}' in label:
                    label = label.split('^{-1}')[0]
                elif '^H' in label:
                    label = label.split('^H')[0]
                    label += '^{-H}'
                elif '^{-H}' in label:
                    label = label.split('^{-H}')[0]
                    label += '^H'
                else:
                    label += '^{-1}'

                if index > 0:
                    latex += ' \\times '
                latex += label

            # Generate function
            def repr_latex(latex_input=None):
                if not latex_input:
                    return latex
                else:
                    return latex + ' \\times ' + latex_input

            # Generate repr_string
            if '^{-1}' in self.repr_str:
                repr_str = self.repr_str.split('^{-1}')[0]
            else:
                repr_str = self.repr_str + '^{-1}'
            # Generate inverse operator
            if self.unitary:
                # For unitary operators, regardless of composite nature
                op = self.H

            elif self._inverse is not None:
                # If an inverse is specified, use this
                op = Operator((self.N, self.M),
                              self.dtype, self.backend,
                              forward=self._inverse,
                              inverse=self._forward,
                              inverse_regularizer=inverse_regularizer,
                              condition_number=self.condition_number,
                              condition_number_is_upper_bound=self.condition_number_is_upper_bound,
                              stack_operators=self.stack_operators,
                              repr_str=repr_str,
                              repr_latex=repr_latex)

            else:
                raise ValueError('Operator %s has no defined inverse' % self.__str__())

            # Add string and latex formatting
            # op.repr_str = repr_str
            # op.repr_latex = repr_latex

            # Assing stack operators
            op.stack_operators = self.stack_operators

            return op
        else:
            raise ValueError('Operator %s is not invertable!' % self.__str__())

    def inv(self, inverse_regularizer=None):
        """Short-hand operator for inverse """
        return(self.inverse(inverse_regularizer))

    @property
    def inverse_regularizer(self):
        return self._inverse_regularizer

    @inverse_regularizer.setter
    def inverse_regularizer(self, new_inverse_regularizer):
        # Assign global definition
        self._inverse_regularizer = new_inverse_regularizer

        # Assign to all sub-operators
        if self.is_stack:
            for operator in self.stack_operators:
                operator.inverse_regularizer = new_inverse_regularizer
        else:
            for op in self.suboperators:
                if op.is_stack:
                    op.inverse_regularizer = new_inverse_regularizer
                else:
                    op._inverse_regularizer = new_inverse_regularizer

    def gradient(self, x, y=None, return_op=False):
        """
        Return the gradient of the function or operator with respect to x
        """
        if self._gradient is not None:

            # If return_op flag is specified, return an operator instead of the numeric gradient
            if return_op:
                return self._gradient(inside_operator=self.inner_operators, x=x)

            # Else, return the computed gradient
            if y is None:
                return self._gradient(inside_operator=self.inner_operators, x=x) * x
            else:
                y[:] = self._gradient(inside_operator=self.inner_operators, x=x) * x

        else:
            raise NotImplementedError('The gradient function is not defined for operator %s' % self.repr_str)

    def grad(self, x, y=None):
        """
        Short-hand for gradient
        """
        return(self.gradient(x, y=y))

    def inverse_check(self, eps=None):
        # Determine eps
        if eps is None:
            # The extra factor is for non-linearity
            eps = precision(self.dtype) * (self.cost ** 2) * 20

        if self.invertable:
            # Generate test object
            x = yp.rand(self.N)

            # Propagate inverse and forward operator
            x_star = self.inv * self.forward(x)

            # Calculate sse
            sse = yp.sum(yp.abs(x - x_star))

            # Check that sse is bounded
            assert sse < eps, 'SSE (%g) is greater than threshold (%g)' % (sse, eps)

            # Return sse
            return sse
        else:
            raise ValueError('Operator %s is not invertable.' % self.repr_str)

    def gradient_check(self, eps=None, step=1e-4, x=None, direction=None, use_l2=True):
        """This function performs a numeric check on the gradient of each operator."""

        # Determine eps
        if eps is None:
            # The extra factor is for non-linearity
            eps = precision(self.dtype) * (self.cost ** 2) * 20

        if eps == 0:
            eps = 1.0

        # If this operator is linear, calling gradient directly will generate the full, dense matrix of the operator.
        # This will take a very long time. So, we'll add a L2Norm in front to make the operation 20x faster.
        if use_l2 and 'norm' not in str(self.__repr__).lower() and yp.prod(self.shape[0]) > 1:
            L2 = L2Norm(self.M, dtype=self.dtype, backend=self.backend)
            A = L2 * self
            forward = A.forward
            gradient = A.gradient
        else:
            forward = self.forward
            gradient = self.gradient

        return scalar(gradientCheck(forward, gradient, self.N, self.dtype, eps=eps, step=step, x=x, direction=direction, backend=self.backend))

    def proximal(self, x, alpha=1.):
        """
        Return the proximal operator with respect to x.
        """

        if self._proximal is not None:
            # Perform proximal operation
            return self._proximal(x, alpha)
        else:
            raise NotImplementedError('The proximal function is not defined for operator %s' % self.repr_str)

    def prox(self, x, alpha=1.):
        """Helper function (shirt-hand) for the proximal function"""
        return self.proximal(x, alpha=alpha)

    def type(self):
        """Return the the type of the operator as a string"""
        return str(self.__class__).split('.')[-1].split("'>")[0]

    def _isSumOfOperators(self):
        """Returns whether a given operator is the sum of operators."""
        return len(self.suboperators) == 2 and (self.suboperators[0].type == 'Hstack' and self.suboperators[1].type == 'Vstack')

    def _isLinear(self):
        """Returns whether a given operator is linear."""
        return self._adjoint is not None

    def _isUnitary(self):
        """Returns whether a given operator is unitary."""
        if self.condition_number is not None:
            return (builtins.abs(self.condition_number - 1.0) < 1e-4)
        else:
            return False

    def _isSmooth(self):
        """Returns whether a given operator is smooth."""
        return self._proximal is None

    def _isConvex(self):
        """Returns whether a given operator is convex."""
        return self.convex or self._isLinear()

    def _isComposite(self):
        """Returns whether a given operator is a composite operator (consists of several operators)."""
        return self.inner_operators is not None

    def _isStack(self):
        """Returns whether a given operator is a stacked operator."""
        return self.stack_operators is not None

    def isAdjointOf(self, adjoint_to_test):
        """Returns whether a given operator is the adjoint of this operator."""
        return self._forward == adjoint_to_test._adjoint

    def isInverseOf(self, inverse_to_test):
        """Returns whether a given operator is the inverse of this operator."""
        return self._forward == inverse_to_test._inverse

    def _isInvertable(self):
        """Returns whether a given operator is invertable."""
        return self._inverse is not None

    def _isSquare(self):
        """Returns whether a given operator is square."""
        return all([dim == prod(self.shape[0]) for dim in self.shape])

    def _isSimplyDiagonalizable(self):
        """
        The goal of this function is to determine whether all sub-operators but
        one are unitary, and that the one non-unitary function has a condition
        number of < infinity. This indicates the matrix can be very easily
        inverted.
        """

        if self.inner_operators is None:
            # If this is a single operator, just return whether it's invertable
            return self.invertable
        else:
            # Loop through inner operator and count the number of non-unitary
            # operators
            non_unitary_operator_count = builtins.sum([not op.unitary for op in self.suboperators])

            # Check if there is one (or zero) non-unitary operators
            return non_unitary_operator_count <= 1

    # Short-hand property names
    H = property(adjoint)
    inv = property(inverse)
    linear = property(_isLinear)
    unitary = property(_isUnitary)
    type = property(type)
    invertable = property(_isInvertable)
    simply_diagonalizable = property(_isSimplyDiagonalizable)
    composite = property(_isComposite)
    square = property(_isSquare)
    sum = property(_isSumOfOperators)
    is_stack = property(_isStack)

    #
    # def asFullMatrix(self, force=False):
    #     """Return the full matrix representation of an operator.
    #
    #     This function should be used with care as many operators are very large.
    #
    #     Args:
    #         force [False]: Force the return of very large operators. If false,
    #                        this function will raise a ValueError
    #     Returns:
    #         matrix: An numerical representation of the operator
    #     Raises:
    #         ValueError
    #
    #     """
    #     if self._isLinear():
    #         if prod(self.shape[1]) < MAX_FULL_MATRIX_ELEMENT_SIZE or force:
    #             return self.forward(eye(prod(self.shape[1]), dtype=self.dtype))
    #         else:
    #             raise ValueError('Operator size (%d) is greater than max full matrix size (%d). Use force=True to over-ride.' %
    #                              (self.shape[1], MAX_FULL_MATRIX_ELEMENT_SIZE))
    #     else:
    #         raise ValueError('Operator is not linear and cannot be represented as a matrix.')

    @property
    def suboperators(self):
        """This function returns all operators in the composite operator """
        op_list = []
        op = self
        while op.inner_operators:
            op_list.append(op.outer_operator)
            op = op.inner_operators
        op_list.append(op)
        return op_list

    def peel(self, level):
        """This function trims off a range of operators from the left side of a composite operator"""
        assert level < len(self.suboperators), 'Level (%d) is greater than the number of operators (%d)' % (level, len(self.suboperators))

        # Traverse and shrink operator
        op = self
        for _ in range(level):
            op = op.inner_operators

        # Return result
        return op

    def __len__(self):
        if len(self.suboperators) > 1 or not self.is_stack:
            return len(self.suboperators)
        else:
            return len(self.stack_operators)

    def __getitem__(self, index):
        if len(self.suboperators) > 1:
            if index >= len(self.suboperators):
                raise IndexError("Index out of range")
            return self.suboperators[index]
        else:
            if not self.is_stack:
                if index == 0:
                    return self
                else:
                    raise IndexError("Index out of range")
            else:
                if index >= len(self.stack_operators):
                    raise IndexError("Index out of range")
                return self.stack_operators[index]

    def __call__(self, x, y=None):
        return self.forward(x, y)

    def __mul__(self, x):
        return self.forward(x)

    def __rmul__(self, x):
        if isscalar(x):
            return _ScaledOperator(self, x)
        else:
            return NotImplemented

    def __add__(self, x):
        if hasattr(x, "forward") and hasattr(x, "adjoint"):
            return _SumOperator(self, x)
        elif x is None:
            return self
        else:
            return(_VectorSumOperator(self, x))

    def __neg__(self):
        return _ScaledOperator(self, -1)

    def __sub__(self, x):
        if hasattr(x, "forward") and hasattr(x, "adjoint"):
            return _SumOperator(self, -x, subtract_str=True)
        else:
            return _VectorSumOperator(self, -x, subtract_str=True)

    def __eq__(self, other):
        """Override the default Equals behavior"""

        # Deal with the case that other is None
        if other is None:
            return False

        # Ensure operators are of the same class
        if not self.repr_str.split('.H')[0] == other.repr_str.split('.H')[0]:
            return False

        # Ensure these are the same size
        if not (self.M == other.M and self.N == other.N):
            return False

        # Check label
        if not self.label == other.label:
            return False

        # Ensure dtype and backend are the same
        if not self.dtype == other.dtype and self.backend == other.backend:
            return False

        # Ensure all argumentss are the same
        if self.arguments is not None and other.arguments is not None:

            # Check if we have a list of arguments (only used for stack operators)
            if type(self.arguments) in (list, tuple):
                assert self.is_stack and other.is_stack, 'Arguments with lists should only be used for stack operators'

                # Check that the number of arguments is the same
                if not len(self.arguments) == len(other.arguments):
                    return False

                # TODO: Compare arguments

            else:

                # Check that the number of arguments is the same
                if not len(self.arguments) == len(other.arguments):
                    return False

                # Check names
                if not all([arg == other_arg for (arg, other_arg) in zip(self.arguments, other.arguments)]):
                    return False

                # Check argument values
                for argument_label in self.arguments:
                    if yp.isarray(self.arguments[argument_label]):
                        # Check shape
                        if not shape(self.arguments[argument_label]) == shape(other.arguments[argument_label]):
                            return False

                        # Check values
                        if not yp.sum(yp.abs(self.arguments[argument_label] - other.arguments[argument_label])) < 1e-3:
                            return False
                    else:
                        if not self.arguments[argument_label] == other.arguments[argument_label]:
                            return False

        # If we've passed all tests, return True
        return True

    def __repr__(self):
        M, N = self.shape
        dt = 'dtype=' + str(self.dtype)

        return '<{}: {}x{} {} {} operator with {}>'.format(self.label, self.M, self.N, self.repr_str,
                                                           self.backend.upper(), dt)

def _ScaledOperator(A, a):
    """Multiplication of Operator and scalar """

    shape = A.M, A.N
    dtype = A.dtype
    backend = A.backend

    def _forward(x, y):
        A._forward(x, y)
        y[:] = y * a

    if callable(A.repr_latex):
        def repr_latex(latex_input=None): return ('%g \\times ' % a + A.repr_latex(latex_input))
    else:
        def repr_latex(latex_input=None): return ('%g \\times ' % a + A.repr_latex  + ' \\times ' + latex_input)

    # Check if adjoint exists (operator is linear)
    if A._adjoint is not None:
        """Define adjoint (gradient is also defined implicitly) """
        def _adjoint(x, y):
            A._adjoint(x, y)
            y[:] = y * a.conjugate()

        return Operator(shape, dtype, backend,
                        forward=_forward,
                        adjoint=_adjoint,
                        repr_latex=repr_latex,
                        inner_operator=A.inner_operators,
                        condition_number=A.condition_number,
                        set_arguments_function=A._setArgumentsFunction,
                        get_arguments_function=A._getArgumentsFunction,
                        cost=A.cost,
                        convex=True,
                        smooth=A.smooth,
                        label=A.label,
                        repr_str=A.repr_str)

    elif A._gradient is not None:
        """Define gradient only (adjoint is undefined)"""

        def _gradient(x=None, inside_operator=None):
            return a.conjugate() * A._gradient(x=x, inside_operator=inside_operator)

        return Operator(shape, dtype, backend,
                        forward=_forward,
                        gradient=_gradient,
                        repr_latex=repr_latex,
                        inner_operator=A.inner_operators,
                        cost=A.cost,
                        smooth=A.smooth,
                        convex=A.convex,
                        label=A.label,
                        repr_str=A.repr_str)

    elif A._proximal is not None:
        """
        If only one of the functions is smooth, use the proximal form
        from the non-smooth and gradient from the smooth
        """
        def _proximal(x, alpha=1.):
            return A._proximal(x, alpha * a)

        return Operator(shape, dtype, backend,
                        forward=_forward,
                        proximal=_proximal,
                        smooth=A.smooth,
                        smooth_part=A.smooth_part,
                        non_smooth_part=A.non_smooth_part,
                        inner_operator=A.inner_operators,
                        cost=A.cost,
                        repr_latex=repr_latex,
                        convex=A.convex,
                        label=A.label,
                        repr_str=A.repr_str)
    else:
        """
        Both A and B are non-smooth
        """
        return Operator(shape, dtype, backend,
                        forward=_forward,
                        gradient=None,
                        smooth=False,
                        repr_latex=repr_latex,
                        inner_operator=A.inner_operators,
                        condition_number=A.condition_number,
                        cost=A.cost,
                        convex=A.convex,
                        label=A.label,
                        repr_str=A.repr_str)


def _SumOperator(A, B, subtract_str=False):
    """Sum of two operators."""

    if prod(A.shape) != prod(B.shape):
        raise ValueError('cannot add {} and {}: shape mismatch'
                         .format(A, B))

    if A.dtype != B.dtype:
        raise ValueError('cannot add {} and {}: dtype mismatch'
                         .format(A, B))

    if A.backend != B.backend:
        raise ValueError('cannot multiply {} and {}: backend mismatch'
                         .format(A, B))

    # Deal with latex representations
    if callable(B.repr_latex):
        def b_repr_latex(latex_input=None): return B.repr_latex(latex_input=latex_input)
    else:
        def b_repr_latex(latex_input=None): return B.repr_latex + ' \\times ' + (latex_input)

    if callable(A.repr_latex):
        def a_repr_latex(latex_input=None): return A.repr_latex(latex_input=latex_input)
    else:
        def a_repr_latex(latex_input=None): return A.repr_latex + ' \\times ' + (latex_input)

    # Define composite latex function
    def repr_latex(latex_input=None):
        if not subtract_str:
            return a_repr_latex(latex_input=latex_input) + ' + ' + b_repr_latex(latex_input=latex_input)
        else:
            return a_repr_latex(latex_input=latex_input) + ' - ' + b_repr_latex(latex_input=latex_input)

    # Turn an addition into a Multiplication
    I = Identity(A.N, A.dtype, A.backend)

    # Import stack
    from .stack import Hstack, Vstack

    # Create stack operators
    AB = Hstack([A, B])
    II = Vstack([I, I])

    # This is the operator we're going to return - just need to over-ride (monkey-patch) a few methods
    Q = AB * II

    # Monkey-patch latex
    Q.repr_latex = repr_latex

    # Determine if sum is smooth
    Q.smooth = A.smooth and B.smooth

    # Set repr_str
    Q.repr_str = A.repr_str + ' + ' + B.repr_str

    # Deal with smooth and non-smooth parts
    if not Q.smooth:
        if A.smooth or B.smooth:
            Q.smooth_part = A if A.smooth else B
            Q.non_smooth_part = A if not A.smooth else B
            Q._proximal = A._proximal if not A.smooth else B._proximal
        else:
            Q.smooth_part = None
            Q.non_smooth_part = Q
    else:
        Q.smooth_part = Q

    # Return this new operator
    return Q

def _GradientOperator(forward_operator):
    """Gradient operator """
    if hasattr(forward_operator, "forward") and hasattr(forward_operator, "adjoint"):
        """If forward operator is an operator, return more or less the same operator,
            but strip out the gradient and adjoint methods. """
        return Operator((forward_operator.M, forward_operator.N),
                        forward_operator.dtype,
                        forward_operator.backend,
                        forward=forward_operator._forward,
                        adjoint=forward_operator._adjoint,
                        smooth=forward_operator.smooth,
                        smooth_part=forward_operator.smooth_part,
                        non_smooth_part=forward_operator.non_smooth_part,
                        label=forward_operator.label,
                        repr_str='∇(' + forward_operator.repr_str + ')',
                        repr_latex=forward_operator.repr_latex,
                        condition_number=forward_operator.condition_number,
                        convex=forward_operator.convex)
    else:
        raise ValueError('Gradient %s must be an operator!' % forward_operator)

def _ProximalOperator(forward_operator):
    """Proximal operator """
    if hasattr(forward_operator, "forward") and hasattr(forward_operator, "proximal"):
        """If forward operator is an operator, return more or less the same operator,
            but strip out the gradient and adjoint methods. """
        return Operator(forward_operator.shape,
                        forward_operator.dtype,
                        forward_operator.backend,
                        forward=forward_operator._forward,
                        smooth=forward_operator.smooth,
                        smooth_part=forward_operator.smooth_part,
                        non_smooth_part=forward_operator.non_smooth_part,
                        label='prox_{' + forward_operator.label + '}',
                        repr_str='prox_{' + forward_operator.repr_str + '}',
                        repr_latex = forward_operator.repr_latex,
                        condition_number=inf,
                        convex=forward_operator.convex)
    else:
        raise ValueError('Gradient %s must be an operator!' % forward_operator)

class VectorSum(Operator):
    """Vector sum linear operator.
    Parameters
    ----------
    vector : vector to sum
    """

    def __init__(self, vector, dtype=None, backend=None, label="y",  subtract_str=False):

        if dtype is None:
            dtype = getDatatype(vector)

        if backend is None:
            backend = getBackend(vector)

        # Store vector
        self.vector = vector

        # Store label for latex
        self.label = label

        # Store whether this is a subtraction or not
        self.subtract_str = subtract_str

        # Store shape
        sz = shape(vector)

        # Store the gradient operator
        self._gradient_op = _GradientOperator(Identity(shape(vector), dtype=getDatatype(vector), backend=getBackend(vector)))

        # Instantiate metaclass
        super(self.__class__, self).__init__((sz, sz), dtype, backend,
                                             forward=self._forward, adjoint=self._adjoint, gradient=self._gradient, label=label,
                                             condition_number=1., repr_latex=self._latex,
                                             set_arguments_function=self._setArguments,
                                             get_arguments_function=self._getArguments)

    def _forward(self, x, y):
        y[:] = reshape(reshape(x, shape(self.vector)) + self.vector, shape(y))

    def _adjoint(self, x, y):
        y[:] = reshape(x, shape(y))

    def _gradient(self, x=None, inside_operator=None):
        return self._gradient_op

    def _latex(self, latex_input=None):
        if latex_input is not None:
            if not self.subtract_str:
                return "(" + latex_input + " + \\vec{" + self.label + "} )"
            else:
                return "(" + latex_input + " - \\vec{" + self.label + "} )"
        else:
            return "\\vec{" + self.label + "}"

    def _setArguments(self, arguments):
        if 'vector' in arguments:
            vector = arguments['vector']

            # Check parameters of vector
            assert size(vector) == size(self.vector), "Argument of wrong shape (should be %d)" % size(self.vector)
            assert getDatatype(vector) == getDatatype(self.vector), "Argument of wrong dtype (should be %s)" % getDatatype(self.vector)
            assert getBackend(vector) == getBackend(self.vector), "Argument of wrong backend (should be %s)" % getBackend(self.vector)

            self.vector = vector

    def _getArguments(self):
        return {'vector': self.vector}


def _VectorSumOperator(A, b, subtract_str=False):
    """Sum of a vector and the output of an operator."""
    if prod(A.shape[0]) != size(b):
        raise ValueError('Cannot add {} and vector of size {}: shape mismatch'
                         .format(A, shape(b)))

    if A.dtype != getDatatype(b):
        raise ValueError('Cannot add {} and vector of dtype {}: dtype mismatch'
                         .format(A, getDatatype(b)))

    # Generate vector sum operator
    V = VectorSum(b, backend=A.backend, dtype=A.dtype, subtract_str=subtract_str)

    # Return
    return V * A


def _ProductOperator(A, B):
    """Product of two operators """

    # If B is none, just Ignore the product operator call
    if B is None:
        return A

    if prod(A.shape[1]) != prod(B.shape[0]):
        raise ValueError('Cannot operate {} on {}: shape mismatch'
                         .format(A, B))
    _shape = (tuple(A.M), tuple(B.N))

    if A.dtype != B.dtype:
        raise ValueError('Cannot operate {} on {}: dtype mismatch'
                         .format(A, B))
    dtype = A.dtype

    if A.backend != B.backend:
        raise ValueError('Cannot operate {} on {}: backend mismatch'
                         .format(A, B))
    backend = A.backend

    def _forward(x, y):
        if all([B.M[i] == A.M[i] for i in range(len(B.N))]):
            B._forward(x, y)
            A._forward(y, y)
        else:
            tmp = alloc(B.M, dtype, backend)
            B._forward(x, tmp)
            A._forward(tmp, y)
            dealloc(tmp)

    # Initialize condition number
    condition_number, condition_number_is_upper_bound = None, False

    if A.condition_number is not None and B.condition_number is not None:
        # Define condition number
        condition_number = A.condition_number * B.condition_number

        # TODO: Find a fast way of determining if A and B have same singular vectors
        condition_number_is_upper_bound = True

    # Parse latex functions for first (left) operator
    if callable(A.repr_latex):
        def repr_latex(latex_input=None):
            return A.repr_latex(b_repr_latex(latex_input=latex_input))
    else:
        def repr_latex(latex_input=None):
            return A.repr_latex + ' \\times ' + b_repr_latex(latex_input)

    # Parse latex functions for second (right) operator
    if callable(B.repr_latex):
        def b_repr_latex(latex_input=None):
            return B.repr_latex(latex_input=latex_input)
    else:
        def b_repr_latex(latex_input=None):
            if not latex_input:
                return B.repr_latex
            else:
                return B.repr_latex + ' \\times ' + latex_input

    # If B is an identity operator, return A
    if 'Identity' in str(B) and not B.composite and not explicit_identity:
        return A

    # If A is an identity operator, return B
    if 'Identity' in str(A) and not A.composite and not explicit_identity:
        return B

    # Check if this operator is the inverse of the other operator - if so, return identity.
    if A.isInverseOf(B):
        return Identity(A.shape[0], A.dtype, A.backend)

    # Account for parentheses order of operations if necessary
    # This is implicitly recursive due to the '*' which creates a new
    # _ProductOperator
    if A.outer_operator is not None:
        _A = A.outer_operator
        _B = A.inner_operators * B
        A, B = _A, _B

    # If both operators have a well-defined inverse, use this
    _inverse = None
    if A.invertable and B.invertable:
        def _inverse(x, y):
            y[:] = reshape(B.inv * A.inv * x, B.N)

    # Check if outer-most operator is the inverse of the inner-most operator.
    # If this is the case, strip these operators.
    if A.linear and A.suboperators[-1].H == B.suboperators[0] and A.suboperators[-1].unitary:
        if B.composite:
            return yp.prod(A.suboperators[:-1] + [B.inner_operators])

    # Check if adjoint exists (implies that the operator is linear)
    if A._adjoint is not None and B._adjoint is not None:
        """Define adjoint (gradient is also defined implicitly) """

        # Define composite adjoint operator
        def _adjoint(x, y):
            if all([A.N[i] == B.N[i] for i in range(len(B.N))]):
                A._adjoint(x, y)
                B._adjoint(y, y)
            else:
                tmp = alloc(B.M, dtype, backend)
                A._adjoint(x, tmp)
                B._adjoint(tmp, y)
                dealloc(tmp)

        # Also define the gradient
        def _gradient(x=None, inside_operator=None):
            return B._gradient(inside_operator=None, x=x) * A._gradient(inside_operator=B, x=x)

        return Operator(_shape, dtype, backend,
                        forward=_forward,
                        adjoint=_adjoint,
                        gradient=_gradient,
                        inverse=_inverse,
                        condition_number=condition_number,
                        inner_operator=B,
                        outer_operator=A,
                        cost=A.cost + B.cost,
                        repr_latex=repr_latex,  # Operator uses non-linear latex notation
                        convex=(A.convex and B.convex),
                        condition_number_is_upper_bound=condition_number_is_upper_bound,
                        repr_str=A.repr_str + ' * ' + B.repr_str)

    elif A._gradient is not None and B._gradient is not None:
        """Define gradient only (adjoint/inverse are undefined, which is the case for a non-linear operator)"""

        # A potential pitfall of this sequential stacking is when there are parentheses in the operator - python follows
        # the standard order of operations, so the items in parentheses will be COMPLETLY processed first. This ordering
        # messes up the recursion order for the associative processing. To fix this, we use a slightly different gradient

        def _gradient(x=None, inside_operator=None):
            return B._gradient(inside_operator=None, x=x) * A._gradient(inside_operator=B, x=x)

        return Operator(_shape, dtype, backend,
                        forward=_forward,
                        gradient=_gradient,
                        inverse=_inverse,
                        condition_number=condition_number,
                        inner_operator=B,
                        outer_operator=A,
                        cost=A.cost + B.cost,
                        repr_latex=repr_latex,  # Operator uses non-linear latex notation
                        convex=(A.convex and B.convex),
                        condition_number_is_upper_bound=condition_number_is_upper_bound,
                        repr_str=A.repr_str + ' * ' + B.repr_str)

    elif A._gradient is None and B._gradient is not None and not A.smooth and A._proximal is not None:
        """A is non-smooth and B is smooth (proximal)"""

        assert B.condition_number == 1.0, "Inner matrix must be unitary for proximal form!"

        def _proximal(x, alpha=1):
            return B.H * A._proximal(B * x, alpha=alpha)

        return Operator(_shape, dtype, backend,
                        forward=_forward,
                        inverse=_inverse,
                        smooth=False,
                        inner_operator=B,
                        outer_operator=A,
                        cost=A.cost + B.cost,
                        convex=(A.convex and B.convex),
                        repr_latex=repr_latex,
                        proximal=_proximal,
                        repr_str=A.repr_str + ' * ' + B.repr_str)
    else:
        """Just copy forward operators"""
        return Operator(_shape, dtype, backend,
                        forward=_forward,
                        inner_operator=B,
                        outer_operator=A,
                        cost=A.cost + B.cost,
                        smooth=(A.smooth and B.smooth),
                        convex=(A.convex and B.convex),
                        repr_latex=repr_latex,
                        repr_str=A.repr_str + ' * ' + B.repr_str)


class Identity(Operator):
    """Identity Linear Operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, label='I'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype
        super(self.__class__, self).__init__((N, N), dtype, backend,
                                             forward=self._forward, adjoint=self._adjoint,
                                             condition_number=1., label=label)

    def _forward(self, x, y):
        y[:] = reshape(x, shape(y))

    def _adjoint(self, x, y):
        y[:] = reshape(x, shape(y))

class MatrixMultiply(Operator):
    """Matrix Linear Operator.
    Parameters
    --------
    A : 2D array, matrix
    """

    def __init__(self, A, dtype=None, backend=None, label=None):

        # Check size of A
        assert len(A.shape) == 2, "A must be a 2D matrix!"

        # Configure backend and datatype
        backend = backend if backend is not None else getBackend(A)
        dtype = dtype if dtype is not None else getDatatype(A)

        # Store matrix and transpose
        self.A = astype(A, dtype)
        self.AH = transpose(self.A, hermitian=True)

        super(self.__class__, self).__init__(((shape(A)[0], 1), [shape(A)[1], 1]), dtype, backend,
                                             repr_latex=self._latex, label=label, cost=prod(A.shape) ** 2,
                                             forward=self._forward, adjoint=self._adjoint)

    def _latex(self, latex_input=None):
        if latex_input is None:
            return 'M'
        else:
            return 'M' + str(latex_input)

    def _forward(self, x, y):
        # TODO: warnings from inevitable reshape
        y[:] = reshape(matmul(self.A, vec(x)), shape(y))

    def _adjoint(self, x, y):
        y[:] = reshape(matmul(self.AH, vec(x)), shape(y))


class FourierTransform(Operator):
    """FFT Linear Operator
    Parameters
    ----------
    N : tuple of int, input shape
    axes : tuple of int, axes to perform FFT
    fftw_flag : FFT flag
           {'FFTPACK', FFTW_ESTIMATE'}
    center : bool, optional
    num_threads : int, optional
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, fft_backend=None,
                 axes=None, center=True, normalize=False, pad=False,
                 label='F'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype
        fft_backend = fft_backend if fft_backend is not None else config.default_fft_backend

        # Set parameters
        if axes is None: axes = tuple(range(len(N)))
        self.N_pad = list(N)

        # Deal with zero-padding for prime and difficult FFTs
        if pad:
            for ind, d in enumerate(N):
                if next_fast_len(d) != d:
                    self.N_pad[ind] = next_fast_len(d)

        self._pad = tuple(self.N_pad) != tuple(N)

        # Store arguments
        self._center = center
        self._normalize = normalize
        self._fft_backend = fft_backend
        self._normalize = normalize
        self._axes = axes

        # Create new fft functions
        self.fft_fun, self.ifft_fun = fftfuncs(self.N_pad, self._axes,
                                               self._center, self._normalize,
                                               dtype, backend,
                                               self._fft_backend)

        super(self.__class__, self).__init__((N, N), dtype, backend,
                                             condition_number=1.0, label=label, cost=prod(N) * math.log(prod(N)),
                                             forward=self._forward, adjoint=self._adjoint,
                                             get_arguments_function=self._getArguments,
                                             set_arguments_function=self._setArguments)

    def _forward(self, x, y):
        x = reshape(x, self.N)  # Convert back to 2D

        # # Ensure y is complex
        # if 'complex' not in getDatatype(y):
        #     y = y + 0j

        if self._pad:
            y[:] = reshape(crop(self.fft_fun(pad(x, self.N_pad, center=True)), self.N, center=True), shape(y))
        else:
            y[:] = reshape(self.fft_fun(x), shape(y))

    def _adjoint(self, x, y):
        x = reshape(x, self.N)  # Convert back to 2D

        if self._pad:
            y[:] = reshape(crop(self.ifft_fun(pad(x, self.N_pad, center=True)), self.N, center=True), shape(y))
        else:
            y[:] = reshape(self.ifft_fun(x), shape(y))

    def _getArguments(self):
        return {'center': self._center,
                'normalize': self._normalize,
                'fft_backend': self._fft_backend,
                'axes': self._axes,
                'pad': self._pad}

    def _setArguments(self, new_arguments):
        if 'center' in new_arguments:
            self._center = new_arguments['center']

        if 'normalize' in new_arguments:
            self._normalize = new_arguments['normalize']

        if 'fft_backend' in new_arguments:
            self._fft_backend = new_arguments['fft_backend']

        if 'axes' in new_arguments:
            self._axes = new_arguments['axes']

        if 'pad' in new_arguments:
            self._pad = new_arguments['pad']

        # Create new fft functions
        self.fft_fun, self.ifft_fun = fftfuncs(self.N_pad, self._axes,
                                               self._center, self._normalize,
                                               self.dtype, self.backend,
                                               self._fft_backend)

class WaveletTransform(Operator):
    """Wavelet Transform Linear Operator
    Padding/dimensions can be tricky.. see https://github.com/PyWavelets/pywt/pull/168
    Parameters
    ----------
    N : tuple of int, input shape
    axes : tuple of int, axes to perform FFT
    dtype : data-type, optional
    """

    def __init__(self, N, wavelet_type='haar', extention_mode='symmetric', level=None, use_cycle_spinning=False,
                 axes=None, dtype=None, backend=None, label=None):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        if axes is None:
            self.axes = tuple(range(len(N)))
        else:
            self.axes = axes

        self.N = N
        self.M = N
        self.wavelet_type = wavelet_type
        self.mode = extention_mode
        self.level = level
        self.use_cycle_spinning = use_cycle_spinning
        self.coeff_slices = None
        self.wt_size = None

        if label is None:
            label = 'W_{' + str(wavelet_type) + '}'

        # If the object has an odd number of pixels, we'll need to pad it to an even number
        pad_shape = [N[i] if not N[i] % 2 else N[i] + 1 for i in range(len(N))]
        self.pad_op = Crop(pad_shape, N, dtype=dtype, backend=backend).H if prod(tuple(pad_shape)) != prod(N) else None

        # Determine new size based on output of wavelet transform
        shape = [self.M, self.N]
        x = rand(self.N, dtype, backend)
        x = self.pad_op * x if self.pad_op is not None else x

        coeffs = pywt.wavedecn(x, self.wavelet_type, mode=self.mode, level=self.level, axes=self.axes)
        coeffs_2d, self.coeff_slices = pywt.coeffs_to_array(coeffs)

        if (prod(coeffs_2d.shape) != prod(self.N)):
            shape[0] = coeffs_2d.shape
            print('WARNING: Wavelet Transform output cannot be tightly packed, reshaping from (%s) to (%s)' % (self.N, coeffs_2d.shape))

        super(self.__class__, self).__init__(shape, dtype, backend, cost=prod(self.N),
                                             condition_number=1.0, label=label,
                                             forward=self._forward, adjoint=self._adjoint)

    def _forward(self, x, y):

        # Reshape input if necessary
        x = reshape(x, self.N)

        # Pad input if necessary
        x = self.pad_op * x if self.pad_op is not None else x

        # Change to numpy backend (pywavelets)
        x = changeBackend(x, 'numpy')

        if self.use_cycle_spinning:
            # Generate random shift
            self.randshift = [random.randint(0, x.shape[i]) for i in range(ndim(x))]

            # Pre-shift input
            x = circshift(x, self.randshift)

        coeffs = pywt.wavedecn(x, self.wavelet_type, mode=self.mode, level=self.level, axes=self.axes)
        coeffs_2d, self.coeff_slices = pywt.coeffs_to_array(coeffs)
        y[:] = reshape(changeBackend(coeffs_2d, getBackend(y)), shape(y))

    def _adjoint(self, x, y):
        # Change to numpy backend
        x = changeBackend(reshape(x, self.M), 'numpy')

        # Ensure the coefficient slices have
        if self.coeff_slices is not None:
            coeffs = pywt.array_to_coeffs(reshape(x, self.M), self.coeff_slices)
        else:
            raise NotImplementedError(
                'Wavelet recon cannot function until decon has been called first! (this is a bug which will be fixed eventually)')

        # Perform reconstrution
        if self.use_cycle_spinning and self.randshift is not None:
            # Un-shift input if we're using cycle_spinning (using shift defined in forward operator)
            _y = changeBackend(circshift(pywt.waverecn(coeffs, self.wavelet_type, mode=self.mode, axes=self.axes), [-1 * self.randshift[i] for i in range(len(self.randshift))]), getBackend(y))
        else:
            _y = changeBackend(pywt.waverecn(coeffs, self.wavelet_type, mode=self.mode, axes=self.axes), getBackend(y))

        # Unpad input if necessary
        _y = self.pad_op.H * _y if self.pad_op is not None else _y

        y[:] = reshape(_y, shape(y))


class Diagonalize(Operator):
    """Point-wise multiplication linear operator.
    Parameters
    ----------
    M : tuple of int, output shape
    N : tuple of int, input shape
    mult : dtype array, array to multiply
    dtype : data-type, optional
    """

    def __init__(self, mult, dtype=None, backend=None, label='d',
                 inverse_regularizer=0.0, inside_operator=None, **kwargs):

        # Configure backend and datatype
        backend = backend if backend is not None else getBackend(mult)
        dtype = dtype if dtype is not None else getDatatype(mult)

        # Ensure mult is of the correct backend
        mult = yp.cast(mult, dtype, backend)

        # Store label
        self.label = label

        # Store inside operator if one is defined
        self.inside_operator = inside_operator

        # Use non-exposed keyword shallow to indicate whether we should perform a deep copy
        # We would NOT want to perform a deep copy for operations inside gradients, for example.
        if kwargs.get('shallow', False):
            elements = mult
        else:
            elements = dcopy(mult)

        # Set elements. These are what is actually used for the multiplication
        if self.inside_operator is None:
            self._elements = elements
        else:
            self._elements = self.inside_operator * elements

        # Store shape
        N = shape(self._elements)

        # Calculate condition number
        if min(abs(self._elements)) == 0:
            condition_number = np.inf
        else:
            condition_number = max(abs(self._elements)) / min(abs(self._elements))

        super(self.__class__, self).__init__((N, N), dtype, backend, cost=prod(N),
                                             forward=self._forward, adjoint=self._adjoint,
                                             inverse=self._inverse,
                                             repr_latex=self._latex,
                                             inverse_regularizer=inverse_regularizer,
                                             label=label,
                                             condition_number=condition_number,
                                             set_arguments_function=self._setArguments,
                                             get_arguments_function=self._getArguments)

    def _forward(self, x, y):
        y[:] = x[:] * self._elements

    def _adjoint(self, x, y):
        y[:] = reshape(conj(self._elements) * reshape(x, self.N), shape(y))

    def _inverse(self, x, y):
        inv_reg = super(self.__class__, self).inverse_regularizer
        y[:] = reshape(conj(self._elements) / (yp.abs(self._elements) ** 2 + inv_reg) * reshape(x, self.N), shape(y))

    def _setArguments(self, arguments):
        if 'elements' in arguments:
            elements = arguments['elements']

            # Check datatype
            assert getDatatype(elements) == self.dtype, "Argument of wrong datatype (should be %d)" % self.dtype

            # Check size
            if self.inside_operator is None:
                assert size(elements) == size(self._elements), "Argument of wrong shape (should be %d)" % size(self._elements)
            else:
                assert prod(size(elements)) == prod(self.inside_operator.N), "Argument of wrong shape (was %s, should be %s)" % (str(shape(elements)), str(self.inside_operator.N))

            # Assign forward operator and elements
            if self.inside_operator is None:
                self._elements = elements
            else:
                self._elements = self.inside_operator * dcopy(elements)

    def _getArguments(self):
        return {'elements': self._elements}

    def _latex(self, latex_input=None):
        if self.inside_operator is None:
            repr_latex = 'diag(\\vec{' + self.label + '})'
        else:
            repr_latex = 'diag(' + self.inside_operator.repr_latex(self.label) + ')'

        if latex_input:
            repr_latex += ('\\times ' + latex_input)

        return repr_latex


class Crop(Operator):
    """Crop linear operator.
    Parameters
    ----------
    M : tuple of int, output shape
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N=None, M=None, roi=None, crop_start=None, pad_value=0,
                 center=None, out_of_bounds_placeholder=None,
                 dtype=None, backend=None, label='CR'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        # Note that either M and N can be supplied OR roi can provide this
        # information, but only if roi.input_shape is specified.
        assert N is not None or roi is not None, "Either N or a ROI must be provided!"
        assert not ((M is not None) and (roi is not None)), "Both M or a ROI cannot be provided!"
        if roi is not None:
            self._roi = roi

            # Over-ride crop_start in the roi if provided
            if crop_start is not None:
                self._roi.start = crop_start

            # Over-ride output shape if M is provided
            if N is not None:
                self._roi.input_shape = N
        else:
            assert M is not None
            assert N is not None
            crop_start = (0, 0) if crop_start is None else crop_start
            self._roi = yp.Roi(start=crop_start, shape=M, input_shape=N)

        # Store parameters
        self._pad_value = pad_value
        self.label = label
        self._out_of_bounds_placeholder = out_of_bounds_placeholder
        self.M, self.N = tuple(self._roi.shape), tuple(self._roi.input_shape)

        # Parse centering condition
        if center is not None:
            self._roi.start = [int((n - m) / 2) for (m, n) in zip(M, N)]

        # Ensure our crop region (N) is within the FOV (M)
        assert min([int((self.N[i] - self.M[i]) / 2) for i in range(len(self.M))]) >= 0
        assert any([self.N[i] >= self.M[i] for i in range(len(self.M))])

        # Define condition number as infinity
        if min(self._roi.mask) == 0:
            condition_number = np.inf
        else:
            condition_number = 1

        # Instantiate metaclass
        super(self.__class__, self).__init__((self.M, self.N), dtype, backend,
                                             condition_number=condition_number,
                                             set_arguments_function=self._setArguments,
                                             get_arguments_function=self._getArguments,
                                             repr_latex=self._latex, label=label, cost=prod(M),
                                             forward=self._forward, adjoint=self._adjoint)

    def _forward(self, x, y):
        crop_roi(reshape(x, self.N), self._roi, y=y,
                 out_of_bounds_placeholder=self._out_of_bounds_placeholder)

    def _adjoint(self, x, y):
        pad_roi(reshape(x, self.M), self._roi, pad_value=self._pad_value, y=y)

    def _latex(self, latex_input=None):
        repr_latex = self.label
        if '_' not in self.label:
            repr_latex += '_{%s \\longrightarrow %s}' % (str(self.N), str(self.M))

        if latex_input is not None:
            return repr_latex + ' \\times ' + latex_input
        else:
            return repr_latex

    def _setArguments(self, arguments):
        """Set operator arguments which can be set."""
        # Check for pad_value
        if 'pad_value' in arguments:
            self._pad_value = arguments['pad_value']

        # Check for crop_start
        if 'roi' in arguments:

            # Ensure ROI has the correct size
            assert all([sh == roi_sh for (sh, roi_sh) in zip(self.M, arguments['roi'].shape)])

            # Set ROI
            self._roi = arguments['roi']

    def _getArguments(self):
        """Get the operator arguments which can be set."""
        return {'pad_value': self._pad_value,
                'roi': self._roi}


class Flip(Operator):
    """Flip linear operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, axis=None, dtype=None, backend=None, label='L'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        self.N = N
        self.axis = axis

        super(self.__class__, self).__init__((N, N), dtype, backend, label=label,
                                             forward=self._forward, adjoint=self._adjoint)

    def _forward(self, x, y):
        x = reshape(x, self.N)
        y[:] = reshape(flip(x, axis=self.axis), shape(y))

    def _adjoint(self, x, y):
        self._forward(x, y)


class Exponential(Operator):
    """Exponential non-linear operator
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, label='EXP'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        # Generate Gradient operator

        super(self.__class__, self).__init__((N, N), dtype, backend, label=label,
                                             repr_latex=self._latex,
                                             forward=self._forward,
                                             gradient=self._gradient)

    def _forward(self, x, y):
        y[:] = reshape(exp(x), shape(y))

    def _gradient(self, x=None, inside_operator=None):
        assert x is not None, "Exponential operator requires x input"
        if inside_operator is None:
            G = Diagonalize(conj(self.forward(x)), dtype=self.dtype, shallow=True)
        else:
            G = Diagonalize(conj(self.forward(inside_operator * x)), dtype=self.dtype, shallow=True)
        G.label = self.repr_latex('\\vec{x}')

        # Update Label
        G.label = 'D_{e^x}'

        return _GradientOperator(G)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return 'e^{' + latex_input + '}'
        else:
            return 'e^{ [\\cdot] }'


class Intensity(Operator):
    """Intensity non-linear operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, label='I'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        super(self.__class__, self).__init__((N, N), dtype, backend,
                                             repr_latex=self._latex, label=label,
                                             forward=self._forward, gradient=self._gradient)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return '|' + latex_input + '|^2'
        else:
            return '|' + ' \\cdot ' + '|^2'

    def _forward(self, x, y):
        y[:] = conj(x) * x

    def _gradient(self, x=None, inside_operator=None):
        if inside_operator is None:
            return _GradientOperator(Diagonalize(x, dtype=self.dtype, shallow=True))
        else:
            if callable(inside_operator.repr_latex):
                repr = inside_operator.repr_latex(' \\vec{x}')
            else:
                repr = inside_operator.repr_latex + ' \\times \\vec{x}'
            return _GradientOperator(Diagonalize(inside_operator * x, dtype=self.dtype, label=repr, shallow=True))


class Power(Operator):
    """Power non-linear operator, raises each element to the given power.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, power, dtype=None, backend=None, label='P'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        self.power = power

        super(self.__class__, self).__init__((N, N), dtype, backend,
                                             label=label, repr_latex=self._latex,
                                             forward=self._forward, gradient=self._gradient)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return latex_input + '^' + str(self.power)
        else:
            return '[\\cdot]' + '^' + str(self.power)

    def _forward(self, x, y):
        x = reshape(x, self.N)
        y = reshape(y, self.N)
        y[:] = x ** self.power

    def _gradient(self, x=None, inside_operator=None):
        return _GradientOperator((self.power * Power(self.N, self.power - 1., dtype=self.dtype)))


class Sum(Operator):
    """
    Sum along given axes
    Parameters
    ----------
    axes
    """

    def __init__(self, N, dtype=None, backend=None, axes='all', label='Σ'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        M = list(N)
        self.reps = [1, ] * len(N)
        self.axes = list(range(len(N))) if axes == 'all' else axes

        if type(self.axes) not in (tuple, list):
            self.axes = [self.axes]

        # Initialize scaling values
        self.scale_forward = 1
        self.scale_adjoint = 1

        # Calculate repitation and scaling values
        for axis in self.axes:
            M[axis] = 1
            self.reps[axis] = N[axis]
            self.scale_adjoint *= 1 / N[axis]


        super(self.__class__, self).__init__((M, N), dtype, backend, label=label, cost=prod(N),
                                             repr_latex=self._latex, forward=self._forward, adjoint=self._adjoint)

    def _latex(self, latex_input=None):
        repr_latex = '\\sum_{axes=' + str(self.axes)
        repr_latex += '} '
        return repr_latex

    def _forward(self, x, y):
        y[:] = reshape(sum(reshape(x, self.N), self.axes) * self.scale_forward, shape(y))

    def _adjoint(self, x, y):
        y[:] = reshape(tile(reshape(x, self.M), self.reps) * self.scale_adjoint, shape(y))


class L2Norm(Operator):
    """1/2 L2-norm squared operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, label='L2'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        super(self.__class__, self).__init__(((1, 1), N), dtype, backend, smooth=True, label=label,
                                             forward=self._forward, gradient=self._gradient, cost=prod(N) ** 2,
                                             proximal=self._proximal, convex=True, repr_latex=self._latex)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return '\\frac{1}{2} ||' + latex_input + '||_2^2'
        else:
            return '\\frac{1}{2} || \\cdot ||_2^2'

    def _forward(self, x, y):
        y[:] = 0.5 * real(norm(x) ** 2)

    def _gradient(self, x=None, inside_operator=None):
        if inside_operator is not None:
            return _GradientOperator(inside_operator)
        else:
            return _GradientOperator(Identity(self.N, self.dtype, self.backend))

    def _proximal(self, x, y, alpha):
        y[:] = x * max(1 - (alpha / norm(x)), 0)


class L1Norm(Operator):
    """Generic L1-norm non-linear operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, dtype=None, backend=None, label='L1'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        super(self.__class__, self).__init__(((1, 1), N), dtype, backend, smooth=False, label=label,
                                             forward=self._forward, proximal=self._proximal,
                                             convex=True,  repr_latex=self._latex)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return '||' + latex_input + '||_1'
        else:
            return '|| \\cdot ||_1'

    def _forward(self, x, y):
        y[:] = sum(abs(x, return_real=False))

    def _proximal(self, x, alpha):
        return softThreshold(x, alpha)

class PhaseRamp(Operator):

    def __init__(self, M, dtype=None, backend=None, label='R',
                 center=True, axes=None, grid_size=None):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype
        axes = tuple(range(len(M))) if axes is None else axes
        grid_size = (1, 1) if grid_size is None else grid_size

        # Generate phase ramp coordinates
        _phase_ramp_coordinates = np.mgrid[[slice(-M[ax] / 2 / M[ax], M[ax] / 2 / M[ax], 1.0 / M[ax]) for ax in axes]]
        _phase_ramp_coordinates = [-2 * np.pi / pitch * coords for (pitch, coords) in zip(grid_size, _phase_ramp_coordinates)]

        # Convert to the correct dtype and backend
        dtype_np = getNativeDatatype(dtype, 'numpy')
        self._phase_ramp_coordinates = [changeBackend(coords.astype(dtype_np), backend) for coords in _phase_ramp_coordinates]

        super(self.__class__, self).__init__((M, (len(axes), 1)), dtype, backend, smooth=True, label=label,
                                             forward=self._forward, gradient=self._gradient,
                                             convex=False,  repr_latex=self._latex)

    def _latex(self, latex_input=None):
        if latex_input is not None:
            return 'e^{-i2\\pi \\vec{k} ' + latex_input + '}'
        else:
            return 'e^{-i2\\pi \\vec{k} [\\cdot ]}'

    def _forward(self, x, y):
        inner = builtins.sum([coords * scalar(xi) for (coords, xi) in zip(self._phase_ramp_coordinates, x)])
        y[:] = cos(inner) + 1j * sin(inner)

    def _gradient(self, x=None, inside_operator=None):
        from .stack import Vstack
        if inside_operator is None:
            expx = self.forward(x)
        else:
            expx = self.forward(inside_operator * x)

        S = Sum(self.M, self.dtype, self.backend)
        D_list = [S * Diagonalize(conj(1j * coords) * conj(expx), shallow=True) for coords in self._phase_ramp_coordinates]
        return _GradientOperator(Vstack(D_list))


class Shift(Operator):
    """
    Shift linear operator.
    Parameters
    ----------
    N : tuple of int, input shape
    dtype : data-type, optional
    """

    def __init__(self, N, shift, axes=None, dtype=None, backend=None, label='S'):

        # Configure backend and datatype
        backend = backend if backend is not None else config.default_backend
        dtype = dtype if dtype is not None else config.default_dtype

        # Ensure shift is a tuple
        self.shift = tuple(np.real(np.asarray(shift)))

        # Determine which axes to shift on
        if axes is None:
            self.axes = len(np.asarray(shift))
        else:
            self.axes = tuple([axes])

        # Determine if we're performing a subpixel shift
        self.is_subpixel_shift = builtins.any([not float(s).is_integer() for s in self.shift])

        # If we are subpixel, create a convolution operator instead
        if self.is_subpixel_shift:
            # Construct subpixel shift operator
            H = PhaseRamp(N, dtype, backend, center=False)
            F = FourierTransform(N, dtype, backend, center=False, pad=True)
            D = Diagonalize(asarray(shift), dtype, backend, inside_operator=H, label=label.lower())
            self.subpixel_shift_op = F.H * D * F

        # Instantiate Metaclass
        super(self.__class__, self).__init__((N, N), dtype, backend, label=label,
                                             condition_number=1.,
                                             forward=self._forward,
                                             adjoint=self._adjoint,
                                             get_arguments_function=self._getArguments,
                                             set_arguments_function=self._setArguments)

    def _forward(self, x, y):
        if not self.is_subpixel_shift:
            for ax, sh in enumerate(self.shift):
                x = roll(reshape(x, self.N), int(sh), axis=ax)
            y[:] = reshape(x, shape(y))
        else:
            self.subpixel_shift_op._forward(x, y)

    def _adjoint(self, x, y):
        if not self.is_subpixel_shift:
            for ax, sh in enumerate(self.shift):
                x = roll(reshape(x, self.N), int(-1 * sh), axis=ax)
            y[:] = reshape(x, shape(y))
        else:
            self.subpixel_shift_op._adjoint(x, y)

    def _getArguments(self):
        return({'shift': self.shift})

    def _setArguments(self, new_arguments):
        if 'shift' in new_arguments:
            self.shift = tuple(np.asarray(new_arguments['shift']))


class Convolution(Operator):
    """Convolution operator."""

    def __init__(self, kernel, mode='circular', dtype=None, backend=None, label='C',
                 pad_value=0, axis=None, pad_convolution=True, pad_fft=True,
                 invalid_support_value=1, fft_backend=None):

        # Configure backend and datatype
        self.backend = backend if backend is not None else yp.config.default_backend
        self.dtype = dtype if dtype is not None else yp.config.default_dtype
        self.fft_backend = fft_backend if fft_backend is not None else yp.config.default_fft_backend

        # Store the mode
        assert mode in ('valid', 'same', 'full', 'circular'), "Convolution mode %s is not supported (Use 'valid', 'same', 'full', or 'circular')"
        self.mode = mode

        # Store input shape
        self.N = shape(kernel)

        # Ensure kernel has the correct datatype and backend
        kernel = yp.cast(kernel, self.dtype, self.backend)

        # Get convolution functions
        self.conv_func, self.conv_adj_func, self.conv_inv_func, self.M = yp.fft.conv_functions(self.N,
                                                                                               kernel,
                                                                                               mode=mode,
                                                                                               axis=axis,
                                                                                               pad_value=pad_value,
                                                                                               pad_convolution=pad_convolution,
                                                                                               fourier_input=False,
                                                                                               pad_fft=pad_fft,
                                                                                               fft_backend=self.fft_backend)

        # Determine condition number
        if yp.min(yp.abs(yp.Ft(kernel))) > 0:
            condition_number = yp.max(yp.abs(yp.Ft(kernel))) / yp.min(yp.abs(yp.Ft(kernel)))
        else:
            condition_number = np.inf

        # Instantiate metaclass
        super(self.__class__, self).__init__((self.M, self.N),
                                             self.dtype,
                                             self.backend,
                                             smooth=True,
                                             forward=self._forward,
                                             adjoint=self._adjoint,
                                             inverse=self._inverse,
                                             condition_number=condition_number,
                                             # get_arguments_function=self._getArguments,
                                             # set_arguments_function=self._setArguments,
                                             label=label)

    def _forward(self, x, y):
        self.conv_func(x, y=y)

    def _adjoint(self, x, y):
        self.conv_adj_func(x, y=y)

    def _inverse(self, x, y):
        self.conv_inv_func(x, y=y, regularization=super(self.__class__, self).inverse_regularizer)


class Segmentation(Operator):
    """Image Segmentation Operator."""
    def __init__(self, roi_list, N=None, dtype=None, backend=None, label='G',
                 alpha_blend_size=0):

        # Configure backend and datatype
        self.backend = backend if backend is not None else yp.config.default_backend
        self.dtype = dtype if dtype is not None else yp.config.default_dtype

        # Store alpha blend size
        self._alpha_blend_size = alpha_blend_size

        # Determine input size from roi list
        N = builtins.sum(roi_list).shape if N is None else N

        # Parse roi list if necessary
        self._roi_list = []
        for roi in roi_list:
            if 'Roi' in str(type(roi)):
                self._roi_list.append(roi)
            else:
                raise ValueError('Expected ROI object for roi_list element (got %s)' % str(roi))

        # Determine the size of the output
        M = [max([roi.shape[0] for roi in self._roi_list]), max([roi.shape[1] for roi in self._roi_list])]
        for roi in self._roi_list[1:]:
            # Check that all Rois have the same width
            # assert roi.shape[1] == self._roi_list[0].shape[1], "All Roi outputs should have the same width!"

            # Increment size in first dimension
            M[0] += roi.shape[0]

        # Ensure M and N are in the correct format
        self.M = tuple(M)
        self.N = tuple(N)

        # Update masks
        self._updateMask()

        # Instantiate the metalass
        super(self.__class__, self).__init__((self.M, self.N), self.dtype, self.backend,
                                             smooth=True,
                                             forward=self._forward,
                                             adjoint=self._adjoint,
                                             inverse=self._inverse,
                                             repr_latex=self._latex,
                                             get_arguments_function=self._getArguments,
                                             set_arguments_function=self._setArguments,
                                             label=label)

    def _forward(self, x, y):
        yp.fill(y, 0)
        for (index, roi) in enumerate(self._roi_list):
            slice_out = (slice(index * self.N[0], (index + 1) * self.N[0]),
                         slice(0, self.M[1]))
            y[slice_out] = crop_roi(x, roi, out_of_bounds_placeholder=0)

    def _adjoint(self, x, y):
        yp.fill(y, 0)

        # Loop over ROIs and update measurement
        for index, roi in enumerate(self._roi_list):
            # roi is where it's going

            # This ROI corresponds to where
            roi_data = yp.Roi(start=(index * self._roi_list[0].shape[0], 0), shape=roi.shape)

            # Crop x to this region and apply alpha blending mask
            x_single = x[roi_data.slice]

            # Generate valid ROI, which indicates which region of the input we should get
            roi_valid = roi.valid - roi.start

            # Assign valid region and apply alpha blending and inversion mask
            if self._alpha_blend_mask is not None:
                y[roi.slice] += x_single[roi_valid.slice] * self._alpha_blend_mask[roi_valid.slice]
            else:
                y[roi.slice] += x_single[roi_valid.slice]

        # Normalize by alpha mask sum
        if self._alpha_blend_mask_sum is not None:
            y[:] /= self._alpha_blend_mask_sum

    def _inverse(self, x, y):

        # Fill array with zeros
        yp.fill(y, 0)

        # Loop over ROIs and update measurement
        for index, roi in enumerate(self._roi_list):

            # ROI for data location
            roi_data = yp.Roi(start=(index * roi.shape[0], 0), shape=roi.shape)

            # Crop x to this region
            x_single = x[roi_data.slice]

            # Generate valid ROI, which indicates which region of the input we should get
            roi_valid = roi.valid - roi.start

            # Assign valid region and apply alpha blending and inversion mask
            if self._alpha_blend_mask is not None:
                y[roi.slice] += x_single[roi_valid.slice] * self._alpha_blend_mask[roi_valid.slice]
            else:
                y[roi.slice] += x_single[roi_valid.slice]

        # Normalize by alpha mask sum
        if self._alpha_blend_mask_sum is not None:
            y[:] /= self._alpha_blend_mask_sum

        # Normalize by inverse normalization (normalize by eigenvalues)
        y[:] /= self._coverage_mask

    def _getArguments(self):
        args = {'alpha_blend_size': self._alpha_blend_size,
                'roi_list': self._roi_list}
        return args

    def _setArguments(self, new_arguments):
        if 'alpha_blend_size' in new_arguments:
            self._alpha_blend_size = int(new_arguments['alpha_blend_size'])
        if 'roi_list' in new_arguments:
            assert len(new_arguments['roi_list']) == len(self._roi_list), 'Roi list has wrong length!'
            roi_list = new_arguments['roi_list']

            for index, roi in enumerate(roi_list):
                if 'Roi' in str(type(roi)):
                    # If we've passed in Roi objects, just set them directly here
                    self._roi_list[index] = roi
                else:
                    raise ValueError('Must pass Roi Objects!')

            # Update mask
            self._updateMask()

    def _latex(self, latex_input=None):
        latex_ret = self.label + '_{\\begin{bmatrix}'
        if len(self._roi_list) < 5:
            for index, roi in enumerate(self._roi_list):
                repr_latex = '{%s}' % str([(start, end) for (start, end) in zip(roi.start, roi.end)])

                if index is not 0:
                    latex_ret += ' \\cr '
                latex_ret += (repr_latex)
        else:
            for index, roi in enumerate(self._roi_list[:2]):
                repr_latex = '{%s}' % str([(start, end) for (start, end) in zip(roi.start, roi.end)])

                if index is not 0:
                    latex_ret += ' \\cr '
                latex_ret += (repr_latex)

            latex_ret += '\\cr \\vdots \\cr '

            for index, roi in enumerate(self._roi_list[-2:]):
                repr_latex = '{%s}' % str([(start, end) for (start, end) in zip(roi.start, roi.end)])

                if index is not 0:
                    latex_ret += ' \\cr '
                latex_ret += (repr_latex)

        latex_ret += '\\end{bmatrix}}'

        if latex_input is not None:
            latex_ret += (' \\times ' + latex_input)

        return latex_ret

    def _updateMask(self):

        # Generate inverse normalization mask (normalizes by the number of times each frame is measured)
        self._coverage_mask = yp.zeros(self.N, self.dtype, self.backend)

        # Count the number of times each position is covered
        for roi in self._roi_list:
            self._coverage_mask[roi.slice] += 1.0

        # Get rid of zeros in both alpha blend mask and _coverage_mask
        self._coverage_mask[abs(self._coverage_mask) < 1e-2] = 1.0

        # Generate alpha blending parameters
        if self._alpha_blend_size > 0:

            # Generate windows
            alpha = self._alpha_blend_size / np.max(self._roi_list[0].shape)
            windows = [sp.signal.windows.tukey(s, alpha=alpha) for s in self._roi_list[0].shape]

            self._alpha_blend_mask = yp.cast(np.outer(windows[0], windows[1]), self.dtype, self.backend)
            self._alpha_blend_mask -= yp.min(self._alpha_blend_mask)
            self._alpha_blend_mask /= yp.max(self._alpha_blend_mask)
            self._alpha_blend_mask[abs(self._alpha_blend_mask) < 1e-2] = 1e-2

            # Generate sum
            self._alpha_blend_mask_sum = yp.zeros_like(self._coverage_mask)
            for roi in self._roi_list:
                roi_valid = roi.valid - roi.start
                self._alpha_blend_mask_sum[roi.valid.slice] += self._alpha_blend_mask[roi_valid.slice]

            # Get rid of zeros
            self._alpha_blend_mask_sum[abs(self._alpha_blend_mask_sum) < 1e-2] = 1

            # Normalize by coverage
            self._alpha_blend_mask_sum /= self._coverage_mask

        else:
            self._alpha_blend_mask = None
            self._alpha_blend_mask_sum = None
