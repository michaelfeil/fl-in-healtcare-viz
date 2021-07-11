"""
Author: Michael Feil 
Website: michaelfeil.eu

python 3.6

"""
import sys
from typing import List, Union
import os

from scipy.signal import resample_poly
import numpy as np

import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from computation_fedavg_gd import grad_descent, fedavg
from computation_fedavg_gd import m1, m2, f_added # import gaussian functions
from computation_fedavg_gd import START_THETA0, START_THETA1

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


OUTPUT_FOLDER = ".\\plot_output"

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


def mpl_multivariante_3d_gd(
    filename: str= "",
    name_labels: List[str] = [],
    colors: List[str] = [],
    functions: list = [],  # list of funtions
    target_function: List[np.ndarray] = [],
    cmap_target: List[str] = [],
    label_target: List[str] = [],
    fedavg_1=None,
    fedavg_2=None,
    fedavg_eval=None,
    fedavg_communication_rounds=10,
    fedavg_steps_local=10,
    title: str = "",
    hist_slice=slice(None),
    theta0=START_THETA0,
    theta1=START_THETA1,
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
        len(target_function) == len(cmap_target) == len(label_target)
    ), "incorrect settings. only one target function can be plotted"

    print(
        f"create 3D plot using SGD on {list(zip(name_labels, colors))}"
        f"over distribution {label_target}"
    )

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
        history = grad_descent(functions[i], theta0=theta0, theta1=theta1,)
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
                theta0=theta0,
                theta1=theta1,
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
    if filename:
        fig.savefig(f"{filename}.svg")

    if "mpld3" in sys.modules:
        try:
            mpld3.save_html(fig, f"{filename}.html")
        except:
            pass
        # try to creat figure mpld3, but 3d plots are not supported.
    return fig


def mpld3_multivariante_2d_gd(
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
    theta0=START_THETA0,
    theta1=START_THETA1,
):
    """save plot as html using mpld3
    
    :param filename: path for writing the html plot to
    :param name_labels: List[str], name of the labels for the history,
    :param colors: List[str], list of matplotlib colors
    :param functions: list,  # list of funtions for evaluationg each descent
    :param target_function: List[np.ndarray], # global function to be plotted.
    :param cmap_target: List[str],
    :param label_target: List[str],
    :param fedavg_1=None,
    :param fedavg_2=None,
    :param fedavg_eval=None,
    :param fedavg_communication_rounds=[10],
    :param fedavg_steps_local=[10],
    :param title: str = "",
    :param hist_slice=slice(None),

    :return: matplotlib.pyplot
    """
    assert (
        len(name_labels) == len(colors) == len(functions)
    ), "incorrect settings. for each function, specify a matplotlib color and label name"

    assert (
        len(target_function) == len(cmap_target) == len(label_target) 
    ), "incorrect settings. only one target function can be plotted"

    print(
        f"create 2D plot using SGD on {list(zip(name_labels, colors))}"
        f"over distribution {label_target}"
    )

    # 2D Settings
    fig = plt.figure()
    ax = plt.gca()
    
    offset_contour = -0.15
    labels_all = []
    plots_all = []
    plots_hover = []
    hover_label = []
    visibillity_label = []
    ax.set_xlabel('\u03F4_0', fontdict={'fontsize': 18})
    ax.set_ylabel('\u03F4_1', fontdict={'fontsize': 18})
    

    # plot target function
    for i in range(len(target_function)):
        plotted = ax.contourf(
            X,
            Y,
            target_function[i],
            cmap=cmap_target[i],
            zorder=0,
            alpha=.97,
        ).collections
        plots_all.append(plotted)
        labels_all.append(f"toggle {label_target[i]}")
        if i == 0:
            visibillity_label.append(True)
        else:
            visibillity_label.append(False)
        # hover_label.append("")
    
    # history of regular GD
    for i in range(len(name_labels)):
        history = grad_descent(functions[i], theta0=theta0, theta1=theta1,)
        history = history[hist_slice]
        plotted, = ax.plot(
            history[:, 0],
            history[:, 1],
            label=name_labels[i],
            c=colors[i],
            lw=5,
            zorder=100,
        )
        plots_all.append(plotted)
        labels_all.append(f"toggle {name_labels[i]}")
        plots_hover.append(plotted)
        hover_label.append(f"{name_labels[i]}_acc_{str(history[-1, 2])[:5]}")
        if i == 0:
            visibillity_label.append(True)
        else:
            visibillity_label.append(False)


    # history of FedAvg
    if fedavg_1 is not None:
        for i in range(len(fedavg_communication_rounds)):
            hs, h1, h2 = fedavg(
                fedavg_1,
                fedavg_2,
                fedavg_eval,
                fedavg_communication_rounds[i],
                fedavg_steps_local[i],
                theta0=theta0,
                theta1=theta1,
            )
            hs = hs[hist_slice]
            plotted, = ax.plot(hs[:, 0], hs[:, 1], label="FedAvg", c="red", lw=5, zorder=100)
            # on contour plot
            # ax.plot(
            #     hs[:, 0],
            #     hs[:, 1],
            #     np.full_like(hs[:, 1], offset_contour),
            #     lw=2,
            #     c="red",
            #     zorder=100,
            # )
            plots_all.append(plotted)
            labels_all.append(f"toggle FedAvg_rounds_{fedavg_communication_rounds[i]}_steps_{fedavg_steps_local[i]}")
            plots_hover.append(plotted)
            hover_label.append(f"FedAvg_acc_{str(hs[-1, 2])[:5]}_rounds_{fedavg_communication_rounds[i]}_steps_{fedavg_steps_local[i]}")
            if i == 0:
                visibillity_label.append(True)
            else:
                visibillity_label.append(False)
    top =  3.5 
    height_from_top = top - len(labels_all) * 0.22
    height_to_top = 3.5 - height_from_top
    ax.add_patch(matplotlib.patches.Rectangle((1.8,height_from_top),1.7,height_to_top, facecolor='white', capstyle="round"))
    
    if title:
        ax.set_title(title, fontdict={'fontsize': 18})

    fig.savefig(f"{filename}.svg")

    if "mpld3" in sys.modules:
        interactive_legend = mpld3.plugins.InteractiveLegendPlugin(plots_all, labels_all, alpha_over=1.5, alpha_unsel=0.2,  
                                                   start_visible=visibillity_label, font_size=18, legend_offset=(-255,-5)) # -1050,-5
        
        hover = [mpld3.plugins.LineLabelTooltip(pl,label=lb) for pl, lb in zip( plots_hover, hover_label)]
        mpld3.plugins.connect(fig, interactive_legend, *hover)
        

        # Show the interactive figure in the notebook
        mpld3.display()
        mpld3.save_html(fig, f"{filename}.html")
        
        # try to creat figure mpld3, but 3d plots are not supported.
    return fig

# define parameters
# Our 2-dimensional distribution will be over variables X and Y
N = 71
X = np.linspace(-3.5, 3.5, N)
Y = np.linspace(-3.5, 3.5, N)
X, Y = np.meshgrid(X, Y) 

# Pack X and Y into a single 3-dimensional array
SAMPLE_GRID_VECTOR = np.empty(X.shape + (2,))
SAMPLE_GRID_VECTOR[:, :, 0] = X
SAMPLE_GRID_VECTOR[:, :, 1] = Y
# sampling grid initialized
# for visualization target function over sampling grid, global function == Z, client1 = Z1.

Z = f_added(theta_vec =SAMPLE_GRID_VECTOR)
Z1 = m1.evaluate(theta_vec = SAMPLE_GRID_VECTOR)
Z2 = m2.evaluate(theta_vec = SAMPLE_GRID_VECTOR)


if __name__ == "__main__":
    print(f"output graphs to folder: {OUTPUT_FOLDER}")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    # 2D plots

    mpld3_multivariante_2d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_clients_only"),
        name_labels=[],
        colors=[],
        functions=[],
        target_function=[ Z2, Z1],
        cmap_target=[  matplotlib.cm.Oranges, matplotlib.cm.Greens],
        label_target=[  " client 2 target", " client 1 target"],
        title="Client distributions only",
    )

    mpld3_multivariante_2d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd_2d"),
        name_labels=[" GD client 1&2", " GD client 2", " GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z1, Z2, Z],
        cmap_target=[ matplotlib.cm.Greens, matplotlib.cm.Oranges, "Purples",],
        label_target=[ " client 1 target", " client 2 target", " accumulated targets (1&2)"],
        title="GD on Clients and theoretical accumulated global distributions",
    )

    mpld3_multivariante_2d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd_2d_fedavg"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z1, Z2, Z],
        cmap_target=[ matplotlib.cm.Greens, matplotlib.cm.Oranges, "Purples",],
        label_target=[ "client 1 target", "client 2 target", "accumulated targets (1&2)"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=[20, 5, 10, 100, 1000],
        fedavg_steps_local=[200, 200, 100, 10, 1],
        title="FedAvg from \u03F4 = (-2,0.1)",
    )

    mpld3_multivariante_2d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd_2d_fedavg_start2"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z1, Z2, Z],
        cmap_target=[ matplotlib.cm.Greens, matplotlib.cm.Oranges, "Purples",],
        label_target=[ "client 1 target", "client 2 target", "accumulated targets (1&2)"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=[20, 5, 10, 100, 1000],
        fedavg_steps_local=[200, 200, 100, 10, 1],
        title="FedAvg from \u03F4 = (2,-2)",
        theta0=2.0,
        theta1=-2.
    )


    # run some plots

    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd_single"),
        name_labels=["GD client 1&2"],
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

    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_argmax_single"),
        name_labels=["argmax client 1&2"],  
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

    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_added_gd"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z],
        cmap_target=["Purples"],
        label_target=["global target"],
        fedavg_eval=f_added,
    )



    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_2_gd"),
        name_labels=["GD client 2"],
        colors=["orange"],
        functions=[m2.evaluate],
        target_function=[Z2],
        cmap_target=[matplotlib.cm.Oranges],
        label_target=["client 2 target"],
        fedavg_eval=m2.evaluate,
    )

    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_1_gd"),
        name_labels=["GD client 1"],
        colors=["green"],
        functions=[m1.evaluate],
        target_function=[Z1],
        cmap_target=[matplotlib.cm.Greens],
        label_target=["client 1 target"],
        fedavg_eval=m1.evaluate,
    )
    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_10_100"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z],
        cmap_target=["Purples"],
        label_target=["global target"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=10,
        fedavg_steps_local=100,
        title="FedAvg 10 rounds with 100 gd steps",
    )
    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_2_1000"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z],
        cmap_target=["Purples"],
        label_target=["global target"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=2,
        fedavg_steps_local=500,
        title="FedAvg 2 rounds with 1000 gd steps",
    )
    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_50_20"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z],
        cmap_target=["Purples"],
        label_target=["global target"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=50,
        fedavg_steps_local=20,
        title="FedAvg 50 rounds with 20 gd steps",
    )
    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_addedfedavg_gd_1000_1"),
        name_labels=["GD client 1&2", "GD client 2", "GD client 1"],
        colors=["purple", "orange", "green"],
        functions=[f_added, m2.evaluate, m1.evaluate],
        target_function=[Z],
        cmap_target=["Purples"],
        label_target=["global target"],
        fedavg_1=m1.evaluate,
        fedavg_2=m2.evaluate,
        fedavg_eval=f_added,
        fedavg_communication_rounds=1000,
        fedavg_steps_local=1,
        title="FedAvg 1000 rounds with 1 gd step",
    )
    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_1_no-gd"),
        name_labels=[],
        colors=[],
        functions=[],
        target_function=[Z1],
        cmap_target=["Greens"],
        label_target=["client 1 target"],
        fedavg_eval=f_added,
    )

    mpl_multivariante_3d_gd(
        filename=os.path.join(OUTPUT_FOLDER, "pdf_2_no-gd"),
        name_labels=[],
        colors=[],
        functions=[],
        target_function=[Z2],
        cmap_target=["Oranges"],
        label_target=["client 2 target"],
        fedavg_eval=f_added,
    )
    mpl_multivariante_3d_gd(
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