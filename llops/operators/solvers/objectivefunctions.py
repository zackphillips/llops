import numpy as np
from llops import shape, getDatatype, getBackend, scalar, config, cast, Ft, iFt, conj, argmax, argmin, abs, dcopy
from ndoperators import L2Norm, Vstack, PhaseRamp, Diagonalize, FourierTransform


def L2(A, y, reg=None):
    """ L2 objective function (global solver)"""

    #  assert A.M == shape(y), "Operator first dimension %d is not operator shape %s" % (A.shape[0], shape(y))
    assert A.backend == getBackend(y)
    assert A.dtype == getDatatype(y)

    # Define objective function
    L2_op = L2Norm(A.M, A.dtype, A.backend)
    objective_data_term = L2_op * (A - y)

    # Add L2 regularization if desired
    if reg is not None:
        # Create full objective (for gradient descent)
        objective = objective_data_term + reg * L2Norm(A.shape[1], A.dtype, A.backend)

        # Add regularization to data term (only used for inverse)
        objective_data_term.inverse_regularizer = reg
    else:
        objective = objective_data_term

    # Simply return this operator
    return _addInvertFunction(objective)


def L2Sequential(A_list, y_list, reg=None):
    """ L2 objective function (sequential solver) """

    objective_function_list = []
    for index in range(len(A_list)):

        # Assign local operator and measurement
        A, y = A_list[index], y_list[index]

        # Check backend and datatype
        assert A.backend == getBackend(y)
        assert A.dtype == getDatatype(y)

        # Form local objective function and append
        objective_function_list.append(L2(A, y, reg))

    # Return stacked list
    return Vstack(objective_function_list)


def _addInvertFunction(objective):
    """This function adds the .invert() function to a L2 objective function. Other types of objectives are not supported."""

    # Determine if objective has a regularization term. If so, only inverte data term
    # Regularization has been added in previous step (TODO move that here)
    if objective.sum:
        _objective = objective.suboperators[0].stack_operators[0]
    else:
        _objective = objective

    # Check structure of objective
    assert _objective.suboperators[0].type == 'L2Norm', "Direct inversion is only support for L2 objectives"
    assert _objective.suboperators[1].type == 'VectorSum', "Direct inversion is only support for L2 objectives with a data fidelity term"

    # Get measurement
    y = -1 * _objective.suboperators[1].arguments['vector']

    # Get forward operator
    A = _objective.inner_operators.inner_operators

    def invert():
        if not A.invertable:
            raise ValueError('Operator %s is not invertable!' % A)
        else:
            return A.inv * y

    # Assign inversion operator
    _objective.invert = invert

    return _objective
