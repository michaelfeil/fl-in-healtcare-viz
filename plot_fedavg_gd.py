"""
Author: Michael Feil / michaelfeil.eu

python 3.6

"""
import sys
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import matplotlib
from scipy import misc
from scipy.signal import resample_poly
import os
try:
    import plotly.graph_objects as go
except:
    go = None
    print("plotly plots will not be created")
    pass
try:
    import mpld3 # for exporting to html, does not always work for 3d plots
except:
    mpld3 = None
    print("mpld3 html plots will not be created, because the current mpld3 is not working or not installed")
    pass

# global
START_THETA0 = -2.0  # start value for \theta_0
START_THETA1 = 0.1  # start value for \theta_1
OUTPUT_FOLDER = ".\\plot_output"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)
# ----------------------------------------------------------------------------------------#
# multivariante gaussian


class MultivarianteGaussian:
    # credits https://stackoverflow.com/questions/28342968/how-to-plot-a-2d-gaussian-with-different-sigma
    def __init__(self, mu, sigma) -> None:
        self.n = mu.shape[0]
        sigma_det = np.linalg.det(sigma)
        self.sigma_inv = np.linalg.inv(sigma)
        self.N = np.sqrt((2 * np.pi) ** self.n * sigma_det)
        self.mu = mu

    def evaluate(self, pos):
        fac = np.einsum(
            "...k,kl,...l->...", pos - self.mu, self.sigma_inv, pos - self.mu
        )
        return np.exp(-fac / 2) / self.N

    def from_float(self, x0, x1):
        return self.evaluate(np.array((x0, x1)))


# ----------------------------------------------------------------------------------------#
# Gradient Descent
def partial_derivative(func, var=0, point=[]):
    """for line search in one-dim parameter space"""
    # credits https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/
    args = point[:]

    def wraps(x):
        args[var] = x
        return func(*args)

    return misc.derivative(wraps, point[var])


def grad_descent(
    function_descent,
    theta0=START_THETA0,
    theta1=START_THETA1,
    eps=1e-07,  # stop condition
    alpha=3e-01,  # learning rate
    nb_max_iter=1000,  # max iterations,
    verbose=0,
):
    """gradient descent"""
    # credits https://moonbooks.org/Articles/How-to-implement-a-gradient-descent-in-python-to-find-a-local-minimum-/
    cond = eps + 10.0
    nb_iter = 0
    tmp_z0 = function_descent(theta0, theta1)
    history = [(theta0, theta1, tmp_z0)]

    while cond > eps and nb_iter < nb_max_iter:
        theta0 = theta0 + alpha * partial_derivative(
            function_descent, 0, [theta0, theta1]
        )
        theta1 = theta1 + alpha * partial_derivative(
            function_descent, 1, [theta0, theta1]
        )
        z0 = function_descent(theta0, theta1)
        nb_iter += 1
        cond = abs(tmp_z0 - z0)
        tmp_z0 = z0
        if nb_iter % 10 == 0 and verbose:
            print(theta0, theta1, cond)
        history.append((theta0, theta1, z0))
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
        np.array(history_client1),
        np.array(history_client2),
    )


# ----------------------------------------------------------------------------------------#
# plotting utils


def resample_history(history: np.ndarray, function, n_min=32):
    """not used"""
    if len(history) < n_min:
        n = n_min // len(history)
        history = resample_poly(history, up=n, down=1, axis=0)
    # reevaluate z axis
    history_z = function(pos=history[:, 0:2])

    history[:, 2] = history_z
    return history


def mpl_multivariante_3d_sgd(
    filename: str,
    name_labels: List[str],
    colors: List[str],
    functions: list,  # list of funtions
    target_function: List[np.ndarray],
    cmap_target: List[str],
    label_target: List[str],
    fedavg_1=None,
    fedavg_2=None,
    fedavg_eval=None,
    fedavg_communication_rounds=10,
    fedavg_steps_local=10,
    title: str = "",
    hist_slice=slice(None),
):
    """save plot as svg
    
    :param filename: path for writing the plot to
    :param name_labels: List[str], name of the labels for the history,
    :param colors: List[str], list of matplotlib colors
    :param functions: list,  # list of funtions for evaluationg each descent
    :param target_function: List[np.ndarray], # global function to be plotted.
    :param cmap_target: List[str],
    :param label_target: List[str],
    :param fedavg_1: None,
    :param fedavg_2=None,
    :param fedavg_eval=None,
    :param fedavg_communication_rounds=10,
    :param fedavg_steps_local=10,
    :param title: str = "",
    :param hist_slice=slice(None),

    :return: matplotlib.pyplot
    """
    assert (
        len(name_labels) == len(colors) == len(functions)
    ), "incorrect settings. for each function, specify a matplotlib color and label name"

    assert (
        len(target_function) == len(cmap_target) == len(label_target) == 1
    ), "incorrect settings. only one target function can be plotted"

    # 3D Settings
    fig = plt.figure()
    ax = fig.gca(projection="3d")
    ax.set_zlim(-0.15, 0.2)
    ax.set_zticks(np.linspace(0, 0.2, 5))

    offset_contour = -0.15
    ax.view_init(elev=20.0, azim=-125)
    ax.dist = 7.5

    ax.set_xlabel(r"${\theta}_0$")
    ax.set_ylabel(r"${\theta}_1$")
    ax.set_zlabel(r"J(${\theta}_0$, ${\theta}_1$)")

    # history of regular GD
    for i in range(len(name_labels)):
        history = grad_descent(functions[i])
        history = history[hist_slice]
        ax.plot(
            history[:, 0],
            history[:, 1],
            history[:, 2],
            label=name_labels[i],
            c=colors[i],
            lw=5,
            zorder=100,
        )
        ax.plot(
            history[:, 0],
            history[:, 1],
            np.full_like(history[:, 1], offset_contour),
            lw=2,
            c=colors[i],
            zorder=100,
        )

    # history of FedAvg
    if fedavg_1 is not None:
        hs, h1, h2 = fedavg(
            fedavg_1,
            fedavg_2,
            fedavg_eval,
            fedavg_communication_rounds,
            fedavg_steps_local,
        )
        hs = hs[hist_slice]
        ax.plot(hs[:, 0], hs[:, 1], hs[:, 2], label="FedAvg", c="red", lw=5, zorder=100)
        # on contour plot
        ax.plot(
            hs[:, 0],
            hs[:, 1],
            np.full_like(hs[:, 1], offset_contour),
            lw=2,
            c="red",
            zorder=100,
        )

        print(title, hs[-1, :])

    # plot target function
    for i in range(len(target_function)):
        # Create a surface plot and projected filled contour plot under it.
        ax.plot_wireframe(
            X,
            Y,
            target_function[i],
            rstride=3,
            cstride=3,
            linewidth=0.5,
            antialiased=True,
            cmap=cmap_target[i],
            label=label_target[i],
            zorder=1,
        )
        ax.contourf(
            X,
            Y,
            target_function[i],
            zdir="z",
            offset=offset_contour,
            cmap=cmap_target[i],
            zorder=0,
        )

    ax.legend()
    if title:
        ax.set_title(title)

    fig.savefig(f"{filename}.svg")

    if "mpld3" in sys.modules:
        mpld3.save_html(fig, f"{filename}.html")
        # try to creat figure mpld3, but 3d plots are not supported.
    return fig

    # def plotly_multivariante_3d_sgd(
    #     filename: str,
    #     name_labels: List[str],
    #     colors: List[str],
    #     functions: list,  # list of funtions
    #     target_function: List[np.ndarray],
    #     cmap_target: List[str],
    #     label_target: List[str],
    #     fedavg_1 = None,
    #     fedavg_2 = None,
    #     fedavg_eval = None,
    #     fedavg_communication_rounds = 10,
    #     fedavg_steps_local = 10,
    #     title: str = "",
    # ):
    # """save plot via plotly to make and interactive html"""
    # assert (
    #     len(name_labels) == len(colors) == len(functions)
    #     and len(target_function) == len(cmap_target) == len(label_target) == 1
    # ), "incorrect settings"

    # # 3D Settings
    # # fig = plt.figure()
    # # ax = fig.gca(projection='3d')
    # # ax.set_zlim(-0.15,0.2)
    # # #ax.set_zticks(np.linspace(0,0.2,5))
    # # ax.view_init(elev=15., azim=10)
    # # ax.dist = 8

    # # ax.set_xlabel(r"${\theta}_0$")
    # # ax.set_ylabel(r"${\theta}_1$")
    # # ax.set_zlabel(r"J(${\theta}_0$, ${\theta}_1$)")
    # plotly_figures = []

    # for i in range(len(name_labels)):
    #     history = grad_descent(functions[i])
    #     plotly_figures.append(
    #         go.Scatter3d(
    #             x=history[:, 0],
    #             y=history[:, 1],
    #             z=history[:, 2],
    #             marker=dict(
    #                 size=4,
    #                 color=history[:, 2],
    #                 colorscale="Viridis",
    #             ),
    #             line=dict(color="darkblue", width=2),
    #         )
    #     )

    # # plot target function
    # for i in range(len(target_function)):
    #     # Create a surface plot and projected filled contour plot under it.
    #     plotly_figures.append(go.Surface(x=X, y=Y, z=target_function[i], opacity=0.55))

    # if fedavg_1 is not None:
    #     hs, h1, h2 = fedavg(
    #         fedavg_1,
    #         fedavg_2,
    #         fedavg_eval,
    #         fedavg_communication_rounds,
    #         fedavg_steps_local,
    #     )
    #     # hs = resample_history(hs, function=fedavg_eval)
    #     plotly_figures.append(
    #         go.Scatter3d(
    #             x=hs[:, 0],
    #             y=hs[:, 1],
    #             z=hs[:, 2],
    #             marker=dict(
    #                 size=4,
    #                 color=hs[:, 2],
    #                 colorscale="Viridis",
    #             ),
    #             line=dict(color="darkblue", width=2),
    #         )
    #     )

    # fig = go.Figure(data=plotly_figures)

    # # fig.update_traces(contours_z=dict(show=True, usecolormap=True,
    # #                               highlightcolor="limegreen", project_z=True))

    # fig.update_layout(
    #     scene=dict(
    #         xaxis=dict(title="x AXIS TITLE"),
    #         yaxis=dict(title="y AXIS TITLE"),
    #         zaxis=dict(title="z AXIS TITLE"),
    #     ),
    # )

    # fig.show()

    # return fig


# define parameters
# Our 2-dimensional distribution will be over variables X and Y
N = 60 
X = np.linspace(-3, 3, N)
Y = np.linspace(-3, 4, N)
X, Y = np.meshgrid(X, Y) 

# Pack X and Y into a single 3-dimensional array
pos = np.empty(X.shape + (2,))
pos[:, :, 0] = X
pos[:, :, 1] = Y
# sampling grid initialized

# draw the two client distributions
m1 = MultivarianteGaussian(
    mu=np.array([0, 1]), sigma=np.array([[1.0, -0.9], [-0.5, 1.5]])
)
m2 = MultivarianteGaussian(
    np.array([0, -1.2]), sigma=np.array([[1.5, 0.6], [0.9, 1.0]])
)

# draw global distribution (add both )
def f_added(theta0: float = None, theta1: float = None, pos: float = None):
    if pos is not None:
        return m1.evaluate(pos) + m2.evaluate(pos)
    else:
        return m1.from_float(theta0, theta1) + m2.from_float(theta0, theta1)


# for visualization target function over sampling grid, global function == Z, client1 = Z1.
# 
Z = f_added(pos=pos)
Z1 = m1.evaluate(pos)
Z2 = m2.evaluate(pos)


# run some plots

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd_single"),
    name_labels=["GD distr. 1&2"],
    colors=[
        "purple",
    ],
    functions=[
        f_added,
    ],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_eval=f_added,
)

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_added_argmax_single"),
    name_labels=["argmax distr. 1&2"],  
    colors=[
        "purple",
    ],
    functions=[f_added],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_eval=f_added,
    hist_slice=slice(-2, None), # does not actually do argmax, but just plot last step of GD
)

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd"),
    name_labels=["GD distr. 1&2", "GD distr. 2", "GD distr. 1"],
    colors=["purple", "orange", "green"],
    functions=[f_added, m2.from_float, m1.from_float],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_eval=f_added,
)

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_2_gd"),
    name_labels=["GD distr. 2"],
    colors=["orange"],
    functions=[m2.from_float],
    target_function=[Z2],
    cmap_target=[matplotlib.cm.Oranges],
    label_target=["distr. 2 target"],
    fedavg_eval=m2.evaluate,
)

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_1_gd"),
    name_labels=["GD distr. 1"],
    colors=["green"],
    functions=[m1.from_float],
    target_function=[Z1],
    cmap_target=[matplotlib.cm.Greens],
    label_target=["distr. 1 target"],
    fedavg_eval=m1.evaluate,
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_10_100"),
    name_labels=["GD distr. 1&2", "GD distr. 2", "GD distr. 1"],
    colors=["purple", "orange", "green"],
    functions=[f_added, m2.from_float, m1.from_float],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_1=m1.from_float,
    fedavg_2=m2.from_float,
    fedavg_eval=f_added,
    fedavg_communication_rounds=10,
    fedavg_steps_local=100,
    title="FedAvg 10 rounds with 100 gd steps",
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_2_1000"),
    name_labels=["GD distr. 1&2", "GD distr. 2", "GD distr. 1"],
    colors=["purple", "orange", "green"],
    functions=[f_added, m2.from_float, m1.from_float],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_1=m1.from_float,
    fedavg_2=m2.from_float,
    fedavg_eval=f_added,
    fedavg_communication_rounds=2,
    fedavg_steps_local=500,
    title="FedAvg 2 rounds with 1000 gd steps",
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_50_20"),
    name_labels=["GD distr. 1&2", "GD distr. 2", "GD distr. 1"],
    colors=["purple", "orange", "green"],
    functions=[f_added, m2.from_float, m1.from_float],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_1=m1.from_float,
    fedavg_2=m2.from_float,
    fedavg_eval=f_added,
    fedavg_communication_rounds=50,
    fedavg_steps_local=20,
    title="FedAvg 50 rounds with 20 gd steps",
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_1000_1"),
    name_labels=["GD distr. 1&2", "GD distr. 2", "GD distr. 1"],
    colors=["purple", "orange", "green"],
    functions=[f_added, m2.from_float, m1.from_float],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_1=m1.from_float,
    fedavg_2=m2.from_float,
    fedavg_eval=f_added,
    fedavg_communication_rounds=1000,
    fedavg_steps_local=1,
    title="FedAvg 1000 rounds with 1 gd step",
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_1_no-gd"),
    name_labels=[],
    colors=[],
    functions=[],
    target_function=[Z1],
    cmap_target=["Greens"],
    label_target=["distr. 1 target"],
    fedavg_eval=f_added,
)

mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_2_no-gd"),
    name_labels=[],
    colors=[],
    functions=[],
    target_function=[Z2],
    cmap_target=["Oranges"],
    label_target=["distr. 2 target"],
    fedavg_eval=f_added,
)
mpl_multivariante_3d_sgd(
    filename=os.path.join(OUTPUT_FOLDER, "pdf_added_no-gd"),
    name_labels=[],
    colors=[],
    functions=[],
    target_function=[Z],
    cmap_target=["Purples"],
    label_target=["global target"],
    fedavg_eval=f_added,
)

plt.show()
print("all plots created")