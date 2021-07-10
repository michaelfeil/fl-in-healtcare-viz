import numpy as np
from scipy import misc
from typing import List, Union

# global
START_THETA0 = -2.0  # start value for \theta_0
START_THETA1 = 0.1  # start value for \theta_1




# ----------------------------------------------------------------------------------------#
# multivariante gaussian


class MultivarianteGaussian:
    # credits https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    def __init__(self, mu, sigma) -> None:
        assert mu.shape[0] ==sigma.shape[0] == sigma.shape[1]
        self.n = mu.shape[0]
        self.sigma = sigma
        sigma_det = np.linalg.det(sigma)
        self.sigma_inv = np.linalg.inv(sigma)
        self.N = np.sqrt((2 * np.pi) ** self.n * sigma_det)
        self.mu = mu

    def evaluate(
        self, theta0: Union[None, float] = None, theta1: Union[None, float] = None, theta_vec: Union[None, np.ndarray] = None
    ):
        if theta_vec is not None and theta0 is None and theta1 is None:
            pos = theta_vec
        elif theta_vec is None and theta0 is not None and theta1 is not None:
            pos = np.array((theta0, theta1))
        else:
            raise BaseException("either set a theta0/theta0 or theta_vec")
        fac = np.einsum(
            "...k,kl,...l->...", pos - self.mu, self.sigma_inv, pos - self.mu
        )
        return np.exp(-fac / 2) / self.N


# ----------------------------------------------------------------------------------------#
# Gradient Descent
def partial_derivative(func, var=0, thetas=[]):
    """for line search in one-dim parameter space"""
    # credits https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/
    args = thetas[:]

    def wraps(x):
        """pick one of the thetas to optimize"""
        args[var] = x
        return func(*args)

    return misc.derivative(wraps, thetas[var])


def grad_descent(
    function_descent,
    theta0=START_THETA0,
    theta1=START_THETA1,
    eps=1e-07,  # stop condition
    lr_alpha=3e-01,  # learning rate
    nb_max_iter=1000,  # max iterations,
    verbose=0,
):
    """gradient descent
    
    :param function_descent: python function for evaluating the gradient 
    :param theta0/theta1: float, start values of theta
    :param eps: float, stop condition
    :param lr_alpha: float,learning rate
    :param nb_max_iter: int, max iterations,

    :return: history, np.ndarray[3,n]
    """
    # credits https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/
    cond = eps + 10.0
    nb_iter = 0
    tmp_z0 = function_descent(theta0, theta1)
    history = [(theta0, theta1, tmp_z0)]

    while cond > eps and nb_iter < nb_max_iter:
        theta0 = theta0 + lr_alpha * partial_derivative(
            function_descent, 0, [theta0, theta1]
        )
        theta1 = theta1 + lr_alpha * partial_derivative(
            function_descent, 1, [theta0, theta1]
        )
        z0 = function_descent(theta0, theta1)
        nb_iter += 1
        cond = abs(tmp_z0 - z0)
        tmp_z0 = z0
        if nb_iter % 20 == 0 and verbose:
            print(f"iter: {nb_iter}, theta0: {str(theta0)[:5]}, theta1: {str(theta1)[:5]}, gradient: {cond}")
        history.append((theta0, theta1, z0))
    if verbose:
        print(f"stop condition reached after {nb_iter} iterations")
    return np.array(history)


def fedavg(
    function_descent_1,
    function_descent_2,
    function_eval,
    communication_rounds=100,
    gd_steps_local=10,
    theta0=START_THETA0,
    theta1=START_THETA1,
):
    """Federated averaging
    
    :param function_descent_1: python function with inputs (float, float) -> output float
    :param function_descent_2: python function with inputs (float, float) -> output float
    :param function_eval: python function with inputs (float, float) -> output float
    :param communication_rounds: int, number of rounds
    :param gd_steps_local: int, number of gradient decent steps
    :param theta0: model weight, float
    :param theta1: model weight, float

    return history_server, history_client1, history_client2
    """
    x1_x2_z = np.array((theta0, theta1, function_eval(theta0=theta0, theta1=theta1)))
    history_server, history_client1, history_client2 = [], [], []
    history_server.append(x1_x2_z.tolist())

    for i in range(communication_rounds):
        # local epochs, client 1
        t1_t2_z_descent1 = grad_descent(
            function_descent_1,
            theta0=x1_x2_z[0],
            theta1=x1_x2_z[1],
            nb_max_iter=gd_steps_local,
        )
        # local epochs, client 2
        t1_t2_z_descent2 = grad_descent(
            function_descent_2,
            theta0=x1_x2_z[0],
            theta1=x1_x2_z[1],
            nb_max_iter=gd_steps_local,
        )
        # Average Update, server
        x1_x2_z = (t1_t2_z_descent1[-1, :] + t1_t2_z_descent2[-1, :]) / 2
        x1_x2_z[2] = function_eval(theta0=x1_x2_z[0], theta1=x1_x2_z[1])

        history_server.append(x1_x2_z.tolist())
        history_client1.append(t1_t2_z_descent1.tolist())
        history_client2.append(t1_t2_z_descent2.tolist())

    return (
        np.array(history_server),
        np.array(history_client1, dtype=object),
        np.array(history_client2, dtype=object),
    )



# draw the two client distributions
m1 = MultivarianteGaussian(
    mu=np.array([0, 1]), sigma=np.array([[1.0, -0.9], [-0.5, 1.5]])
)
m2 = MultivarianteGaussian(
    np.array([0, -1.2]), sigma=np.array([[1.5, 0.6], [0.9, 1.0]])
)

# draw global distribution (add both )
def f_added(theta0: float = None, theta1: float = None, theta_vec: float = None):
    
    if theta_vec is not None and theta0 is None and theta1 is None:
        pos = theta_vec
    elif theta_vec is None and theta0 is not None and theta1 is not None:
        pos = np.array((theta0, theta1))
    else:
        raise BaseException("either set a theta0/theta0 or theta_vec")
    
    return m1.evaluate(theta_vec = pos) + m2.evaluate(theta_vec = pos)

