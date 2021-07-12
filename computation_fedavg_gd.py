import numpy as np
from scipy import misc
from typing import List, Union
from scipy.integrate import nquad
# credits https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/
# credits https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma

# global
START_THETA0 = -2.0  # start value for \theta_0
START_THETA1 = 0.1  # start value for \theta_1
START_THETA = np.array([-2.0, 0.1])



# ----------------------------------------------------------------------------------------#
# multivariate gaussian

class MultivariateGaussian:
    """Multivariante Gaussian definition and evaluation"""
    def __init__(self, mu, sigma) -> None:
        assert mu.ndim == 1 and sigma.ndim == 2
        assert mu.shape[0] ==sigma.shape[0] == sigma.shape[1]
        
        self.sigma = sigma
        self.mu = mu

        self.sigma_inv = np.linalg.inv(sigma)
        self.N = np.sqrt((2 * np.pi) ** mu.shape[0] * np.linalg.det(sigma))

    def evaluate(
        self, theta_vec: np.ndarray
    ):  
        """evaluate Gaussian at theta position"""
                
        fac = np.einsum(
            "...k,kl,...l->...", theta_vec - self.mu, self.sigma_inv, theta_vec - self.mu
        )
        return np.exp(-fac / 2) / self.N

# Gradient Descent
def partial_derivative(func, var: int, thetas: np.ndarray):
    """for line search in one-dim parameter space"""
    args = thetas.copy()

    def wraps(x):
        """pick one of the thetas to optimize"""
        args[var] = x
        return func(args)

    return misc.derivative(wraps, thetas[var])


def grad_descent(
    function_descent,
    theta = START_THETA,
    eps=1e-07,  
    lr_alpha=3e-01,  
    nb_max_iter=1000, 
    verbose=0,
):
    """gradient descent
    
    :param function_descent: python function for evaluating the gradient 
    :param theta1: float, start values of theta
    :param eps: float, stop condition
    :param lr_alpha: float, learning rate
    :param nb_max_iter: int, max iterations,

    :return: history, np.ndarray[3,n]
    """
    assert len(theta.shape) == 1
    tmp_z0 = function_descent(theta)
    history = [np.append(theta, tmp_z0)]

    for nb_iter in range(nb_max_iter):

        theta = np.fromiter(
            (
                theta_i + lr_alpha * partial_derivative(
                    function_descent, i, theta
                )
                for i, theta_i in enumerate(theta)
            ),
            np.float
        )
        
        z0 = function_descent(theta)
        if abs(tmp_z0 - z0) < eps:
            break # end GD
        if verbose:
            if nb_iter % 50 == 0:
                print(f"iter: {nb_iter}, theta: {str(theta)[:5]}, gradient: {abs(tmp_z0 - z0)}")

        tmp_z0 = z0
        history.append(np.append(theta, tmp_z0))

    if verbose:
        print(f"stop condition reached after {nb_iter} iterations")
    
    return np.array(history)


def fedavg(
    function_clients: list,
    function_eval,
    communication_rounds=100,
    gd_steps_local=10,
    theta: np.ndarray = START_THETA
):
    """Federated averaging
    
    :param function_clients: List of python method with inputs (np.ndarrray) -> output float
    :param function_eval: python method with inputs (np.ndarrray) -> output float
    :param communication_rounds: int, number of rounds
    :param gd_steps_local: int, number of gradient decent steps
    :param theta: model weight, float

    return history_server, history_client1, history_client2
    """
    
    history_server, history_clients = [], []
    history_server.append(theta)

    for i in range(communication_rounds):
        # collect the most updated theta from each client after GD
        client_resposes_theta = [
            # distribute theta to each client and do GD
            grad_descent(
                fct_client, # over the client distribution
                theta = theta, # current theta
                nb_max_iter=gd_steps_local # run for k steps
            )[-1,:]
            for fct_client in function_clients
        ]
        
        # Average update from clients on server
        theta = np.average(client_resposes_theta, axis=0)[:-1]
        
        # done
        history_server.append(theta)
        history_clients.append(client_resposes_theta)

    history_server = np.array(history_server)
    return (
        np.concatenate((history_server, function_eval(history_server)[:, None]), axis=1),
        np.array(history_clients, dtype=object),
    )


# ----------------------------------------------------------------------------------------#



# draw the two client distributions
m1 = MultivariateGaussian(
    mu=np.array([0, 1]), sigma=np.array([[1.0, -0.9], [-0.5, 1.5]])
)
m2 = MultivariateGaussian(
    np.array([0, -1.2]), sigma=np.array([[1.5, 0.6], [0.9, 1.0]])
)



# draw global distribution (multiply both )
class PDFsManipulate:
    def __init__(self, functions, method = np.multiply, shape_theta = (2,), range_l = (-100,100)) -> None:
        self.functions = functions
        self.method = method
        self.shape_theta = shape_theta
        ranges = list([range_l for i in range(shape_theta[0])])
        # integrate pdf by discret integration over ranges range_l of theta possibilities
        self.normalizer_pdf = nquad(func = self._eval_args, ranges=ranges)[0]
        
    def _eval_args(self, *args):
        return self.evaluate(np.fromiter(args, np.float), normalize_pdf=False)

    def evaluate(self, theta_vec: np.ndarray, normalize_pdf=True):
        assert theta_vec.shape[-1] == self.shape_theta[0]
        if normalize_pdf:
            norm =  self.normalizer_pdf 
        else:
            norm = 1
        return self.method(*[fct(theta_vec) for fct in self.functions]) / norm


f_multi = PDFsManipulate(functions=[m1.evaluate, m2.evaluate], method = np.multiply)
f_add = PDFsManipulate(functions=[m1.evaluate, m2.evaluate], method = np.add)
