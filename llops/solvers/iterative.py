import numpy as np
from . import objectivefunctions
from llops import display
import matplotlib.pyplot as plt
import llops as yp
import time
import math

# Single-precision floating-point error
eps = 1e-15


class NesterovAccelerator():
    def __init__(self, objective, restart_enabled=False):
        self.restart_enabled=restart_enabled
        self.objective = objective

        self.y_prev = None     # Set to initialization
        self.x_prev = None     # Previous result
        self.beta = 0.5      # Momentum term
        self.t = 0.         # Nesterov parameter
        self.t_prev = 1.    # Nesterov parameter

    def iterate(self, y):
        """Nesterov iteration term"""

        # Update step size t
        _t = self.t
        self.t = 1 + math.sqrt(1 + 4 * self.t_prev ** 2) / 2
        self.t_prev = _t

        # Update x
        if self.y_prev is not None:
            if type(self.y_prev) not in (list, tuple):
                x = (1 - self.beta) * y + self.beta * self.y_prev
            else:
                x = [(1 - self.beta) * _y + self.beta * _y_prev for _y, _y_prev in zip(y, self.y_prev)]

            # Update momentum term
            if type(self.objective) in (list, tuple):
                if self.restart_enabled and abs(yp.scalar(self.objective[0](self.x_prev[0]))) < abs(yp.scalar(self.objective[0](x[0]))):
                    self.beta = 0
                    x = y
                else:
                    self.beta = (1 - self.t_prev) / self.t
            else:
                if self.restart_enabled and abs(yp.scalar(self.objective(self.x_prev))) < abs(yp.scalar(self.objective(x))):
                    self.beta = 0
                    x = y
                else:
                    self.beta = (1 - self.t_prev) / self.t
        else:
            x = y

        # Store x, y and t values from previous iteration
        self.x_prev = x
        self.y_prev = y
        self.t_prev = self.t

        return(x)


class IterativeAlgorithm():
    """
    Metaclass for Iterative algorithms
    """

    def __init__(self, objective_function, solver_type,
                 display_type='text',
                 convergence_tol=1e-14, let_diverge=False,
                 use_nesterov_acceleration=False, nesterov_restart_enabled=True,
                 step_size=0.1):
        self.objective = objective_function
        self.type = solver_type
        self.x = None
        self.initialized = False
        self.plot = None
        self.t0 = 0

        # Set default settings from kwargs
        self.display_type = display_type
        self.convergence_tol = convergence_tol
        self.use_nesterov_acceleration = use_nesterov_acceleration
        self.nesterov_restart_enabled = nesterov_restart_enabled
        self.let_diverge = let_diverge or nesterov_restart_enabled
        self.step_size = step_size
        self._default_iteration_count = 10
        self.multi_objective = type(self.objective) in (list, tuple)

    def _initialize(self, initialization=None, **kwargs):

        for keyword in kwargs:
            if hasattr(self, keyword):
                setattr(self, keyword, kwargs[keyword])
            else:
                "Ignoring keyword %s" % keyword

        # Show objective function(s)
        if self.display_type is not None:
            if type(self.objective) in (list, tuple):
                print('Minimizing functions:')
                for objective in self.objective:
                    objective.latex()
            else:
                print('Minimizing function:')
                self.objective.latex()

        # Generate initialization
        if type(self.objective) not in (list, tuple):
            if initialization is None:
                self.x = yp.zeros(self.objective.N, dtype=self.objective.dtype, backend=self.objective.backend)
            else:
                self.x = yp.dcopy(initialization)
        else:
            # Create random initializations for all variables
            self.x = []
            for index, objective in enumerate(self.objective_list):
                if initialization is None:
                    self.x.append(yp.zeros(objective.N, objective.dtype, objective.backend))
                else:
                    assert type(initialization) in (list, tuple)
                    self.x.append(initialization[index])

        # Generate plotting interface
        if self.display_type == 'text':
            # Create text plot object
            self.plot = display.IterationText()

        elif self.display_type == 'plot':
            # Create figure if axis was not provided
            if 'ax' in kwargs:
                self.ax = kwargs['ax']
            else:
                self.fig = plt.figure(figsize=kwargs.pop('figsize', (5, 3)))
                ax = plt.gca()

            # Create iteration plot object
            use_log = (kwargs.pop('use_log_x', False), kwargs.pop('use_log_y', False))
            max_iter = kwargs.pop('max_iter_plot', self._default_iteration_count)
            self.plot = display.IterationPlot(ax, max_iter, use_log=use_log)
            self.fig.canvas.draw()
            plt.tight_layout()

        # Update plot with first value
        if self.plot is not None:
            objective_value = self.objective(self.x) if not self.multi_objective else self.objective[0](self.x[0])

            self.plot.update(0, abs(yp.scalar(objective_value)), new_time=0, step_norm=0)
            self.t0 = time.time()

        # If using Nesterov acceleration, intitalize NesterovAccelerator class
        if self.use_nesterov_acceleration:
            self.nesterov = NesterovAccelerator(self.objective, self.nesterov_restart_enabled)

            # Enable restarting if desired
            if self.nesterov_restart_enabled:
                self.let_diverge = True  # We need to let nesterov diverge a bit

        self.initialized = True

    def solve(self, initialization=None, iteration_count=10, display_iteration_delta=None, **kwargs):

        # Process display iteration delta
        if display_iteration_delta is None:
            display_iteration_delta = iteration_count // 10

        # Try to import arrayfire and call garbage collection to free memory
        try:
            import arrayfire
            arrayfire.device_gc()
        except ImportError:
            pass

        # Initialize solver if it hasn't been already
        if not self.initialized:
            self._initialize(initialization, **kwargs)

        cost = []
        # Run Algorithm
        for iteration in range(iteration_count):

            # Determine step norm
            if self.multi_objective:
                x_prev_norm = sum([yp.norm(x) for x in self.x])
            else:
                x_prev_norm = yp.norm(self.x)

            # Perform iteration
            self.x = self._iteration_function(self.x, iteration, self.step_size)

            # Apply nesterov acceleration if desired
            if self.use_nesterov_acceleration:
                self.x = self.nesterov.iterate(self.x)

            # Store cost
            objective_value = self.objective(self.x) if not self.multi_objective else self.objective[0](self.x[0])
            cost.append(abs(yp.scalar(objective_value)))

            # Determine step norm
            if self.multi_objective:
                step_norm = abs(sum([yp.norm(x) for x in self.x]) - x_prev_norm)
            else:
                step_norm = abs(yp.norm(self.x) - x_prev_norm)

            # Show update
            if self.display_type == 'text':
                if (iteration + 1) % display_iteration_delta == 0:
                    self.plot.update(iteration + 1, cost[-1], time.time() - self.t0, step_norm)
            elif self.display_type == 'plot':
                self.plot.update(iteration, new_cost=cost[-1])
                self.fig.canvas.draw()
            elif self.display_type is not None:
                raise ValueError('display_type %s is not defined!' % self.display_type)

            # Check if converged or diverged
            if len(cost) > 2:
                if self.convergence_tol is not None and (abs(cost[-1] - cost[-2]) / max(cost[-1], 1e-10) < self.convergence_tol or cost[-1] < 1e-20):
                    print("Met convergence requirement (delta < %.2E) at iteration %d" %
                          (self.convergence_tol, iteration + 1))
                    return(self.x)
                elif cost[-1] > cost[-2] and not self.let_diverge:
                    print("Diverged at iteration %d" % (iteration + 1))
                    return(self.x)
        return(self.x)


class ProjectedGradientDescent(IterativeAlgorithm):
    """
    A class defining the standard projected gradient descent algorithm.
    """

    def __init__(self, objective, projection, smooth_only=False, **kwargs):

        # Deal with smooth/non-smooth functions
        if smooth_only:
            self.objective = objective.smooth_part
        else:
            self.objective = objective
        self.projection = projection

        # Instantiate metaclass
        super(self.__class__, self).__init__(self.objective,
                                             solver_type='Gradient Descent',
                                             **kwargs)

    def _iteration_function(self, x, iteration_number, step_size):
        # Perform gradient step
        if step_size is not None:
            if hasattr(step_size, '__call__'):
                x -= step_size(iteration_number) * self.objective.gradient(x)
            else:
                """ Explicit step size is provided """
                x -= step_size * self.objective.gradient(x)
        else:
            """ No explicit step size is provided """
            # If no step size provided, use optimal step size if objective is convex,
            # or backtracking linesearch if not.\
            # TODO incorporate projection to linesearch
            if self.objective.convex:
                g = self.objective.grad(x)
                step_size = yp.norm(g) ** 2 / (yp.norm(g) ** 2 + eps)
                x[:] -= step_size * g
            else:
                # TODO confirm gradient shape
                x,_ = backTrackingStep(x, lambda x: yp.scalar(self.objective(x)), self.objective.grad(x))

        return(self.projection(x))


class GradientDescent(IterativeAlgorithm):
    """
    A class defining the standard gradient descent algorithm.
    """

    def __init__(self, objective, smooth_only=False, use_nesterov_acceleration=True,
                 **kwargs):

        # Deal with smooth/non-smooth functions
        if smooth_only:
            self.objective = objective.smooth_part
        else:
            self.objective = objective

        # Instantiate metaclass
        super(self.__class__, self).__init__(self.objective,
                                             solver_type='Gradient Descent',
                                             use_nesterov_acceleration=use_nesterov_acceleration,
                                             **kwargs)

    def _iteration_function(self, x, iteration_number, step_size):
        # Perform gradient step
        # TODO check gradient shape
        if step_size is not None:
            if hasattr(step_size, '__call__'):
                x[:] -= step_size(iteration_number) * self.objective.gradient(x)
            else:
                """ Explicit step size is provided """
                x[:] -= step_size * self.objective.gradient(x)
        else:
            """ No explicit step size is provided """
            # If no step size provided, use optimal step size if objective is convex,
            # or backtracking linesearch if not.
            if self.objective.convex:
                g = self.objective.grad(x)
                step_size = yp.norm(g) ** 2 / (yp.norm(g) ** 2 + eps)
                x[:] -= step_size * g
            else:
                x[:],_ = backTrackingStep(x.reshape(-1), lambda x: yp.scalar(self.objective(x)), self.objective.grad(x).reshape(-1))

        return(x)


class Ista(IterativeAlgorithm):
    """A class defining the standard gradient descent algorithm."""

    def __init__(self, objective, **kwargs):

        # Create gradient descent solver
        self.gd_solver = GradientDescent(objective, smooth_only=True, **kwargs)

        super(self.__class__, self).__init__(objective, solver_type='ISTA',
                                             use_nesterov_acceleration=False,
                                             nesterov_restart_enabled=False,
                                             **kwargs)

    def _iteration_function(self, x, iteration_number, step_size):

        # Perform gradient iteration
        x = self.gd_solver._iteration_function(x, iteration_number, step_size)

        # Perform ISTA step
        if self.objective.non_smooth_part is not None:
            # Perform iterative shrinkage
            x = self.objective.non_smooth_part.prox(x, alpha=step_size)

        return(x)

class Fista(IterativeAlgorithm):
    """
    A class defining the standard gradient descent algorithm.
    """
    def __init__(self, objective, **kwargs):

        # Create gradient descent solver
        self.gd_solver = GradientDescent(objective, smooth_only=True, **kwargs)

        super(self.__class__, self).__init__(objective, solver_type='FISTA',
                                             use_nesterov_acceleration=True,
                                             nesterov_restart_enabled=True,
                                             let_diverge=True,
                                             **kwargs)

    def _iteration_function(self, x, iteration_number, step_size):

        # Perform gradient iteration
        x = self.gd_solver._iteration_function(x, iteration_number, step_size)

        # Perform ISTA step
        if self.objective.non_smooth_part is not None:
            # Perform iterative shrinkage
            x = self.objective.non_smooth_part.prox(x, alpha=step_size)

        return(x)


class ConjugateGradient(IterativeAlgorithm):
    """
    A class defining the conjugate gradient algorithm.
    """

    def __init__(self, A, y, **kwargs):

        # If A is not square, make is square by multipling by A^H
        if A.shape[0] != A.shape[1]:
            self.A = A.H * A
            self.y = A.H * y
        else:
            self.A = A
            self.y = y

        # Define a L2 objective function for evaulating Cost
        objective = objectivefunctions.L2(A, y)

        super(self.__class__, self).__init__(objective, solver_type='Conjugate Gradient', **kwargs)

    def _iteration_function(self, x, iteration_number, step_size):
        if iteration_number == 0:
            self.r = self.y - self.A * x
            self.p = self.r
        else:

            # Helper variables
            Ap = self.A * self.p
            pAp = yp.sum(yp.real((yp.conj(self.p) * Ap)))
            r2 =  yp.sum(yp.real((yp.conj(self.r) * self.r)))

            # Update alpha
            alpha = r2 / pAp

            # Update x
            x += alpha * self.p

            # Update r
            self.r -= alpha * Ap

            # Update beta
            beta = yp.sum(yp.real((yp.conj(self.r) * self.r))) / r2

            # Update p
            self.p = self.r + beta * self.p

        return(x)


class AlternatingSolver(IterativeAlgorithm):
    """An Alternating Solver
    Args:
        objectives: objective functions to minimize which are functions of different variables
        its_per_step: number of gradient iterations for each variable
        total_it: total number of iterations
        initializations (optional): starting points
    Returns:
        next iterate and function value at iterate

        TODO currently mutates a lot of things (including initializations)
    """
    def __init__(self, objective_function_list, iteration_counts, argument_mask, **kwargs):
        # Store objective function list
        self.objective_list = objective_function_list

        # Store iteration counts
        self.iteration_counts = iteration_counts

        # Store argument_mask
        self.argument_mask = argument_mask

        # Check that objective functions are communative
        self._check_communitivity()

        # Initialize metaclass
        super(self.__class__, self).__init__(self.objective_list,
                                             solver_type='Conjugate Gradient',
                                             **kwargs)

    def _check_communitivity(self):
        """This function simply checks whether a list of objective functions is communative"""
        # Generate test arrays
        test_arrays = []
        for objective in self.objective_list:
            test_arrays.append(yp.rand(objective.N))

        # Loop over each objective, setting arguments and operating on missing one
        objective_value_list = []
        for index, objective in enumerate(self.objective_list):

            # Get sublist with all arguments EXCEPT the index. These will be used to set arguments.
            arguments = [None] * len(objective.arguments)
            for (_index, replacement) in zip(self.argument_mask, test_arrays[:index] + test_arrays[index + 1:]):
                arguments[_index] = replacement
            objective.arguments = arguments

            # Determine value of objective function and append to list
            objective_value_list.append(yp.scalar(objective * test_arrays[index]))

        # Ensure all the objective values are the same
        assert all([value == objective_value_list[0] for value in objective_value_list]), "Objective functions are not commutative"

    def _iteration_function(self, x, iteration_number, step_size):
        assert type(x) in (list, tuple)

        if type(step_size) not in (tuple, list):
            step_size = [step_size] * len(self.objective_list)

        for index in range(len(self.objective_list)):

            # Set arguments in objective
            arguments = [None] * len(self.objective_list[index].arguments)
            for (_index, replacement) in zip(self.argument_mask, x[:index] + x[index + 1:]):
                arguments[_index] = replacement
            self.objective_list[index].arguments = arguments

            # Perform iterations
            if self.iteration_counts[index] <= 0:
                x[index] = self.objective_list[index].invert()
            else:
                # Subiterations
                for __ in range(self.iteration_counts[index]):
                    x[index] -= step_size[index] * self.objective_list[index].gradient(x[index])

            # Take sub-gradient
            # x[index] -= step_size[index] * self.objective_list[index].gradient(x[index])

        return x


def backTrackingStep(x, f, gradfx, max_search_iteration=100):
    """a step method which uses backtracking linesearch
    Args:
        x: current iterate
        f: function
        gradfx: gradient of the function at x
        max_search_iteration: Maximum number of iterations to use for line-search
    Returns:
        next iterate and function value at iterate
    """
    t = 1e-2
    a = 0.9
    step = x - t*gradfx
    i = 0
    fstep = f(step)
    fx = f(x)
    while (np.any(np.isnan(fstep)) or (fstep > fx - 0.0004 * t*gradfx.T.dot(gradfx))) and (i < max_search_iteration):
        t = t*a
        i = i + 1
        step = x - t*gradfx
        fstep = f(step)
    return step, fstep
