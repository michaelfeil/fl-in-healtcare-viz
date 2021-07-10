# Visualizations for the Seminar "Federated Learning in Healtcare"
TU Munich, Website https://albarqouni.github.io/courses/flhsose2021/

Visualizes gradient descent (line-search, non-stochastic) for non-iid Gaussian distributions with Federated Averaging.
3D Visualizations using Matplotlib. ![Binder](https://mybinder.org/badge_logo.svg)

# Usage: Run Jupyter Notebook online in binder

You can run the notebook by clicking on the link below.

[![Binder](https://mybinder.org/badge_logo.svg)](https://mybinder.org/v2/gh/michaelfeil/fl-in-healtcare-viz/HEAD?filepath=colab-viz-fedavg-vs-gd.ipynb)

# Usage: local
## installation
Follow, if you wish to install this on yourdevice. 

Tested with this conda env, may run with most versions of numpy, scipy and matplotlib. 
plotly and mpld3 only need for exporting the plots to html. 

```conda env create -f environment.yml```

## run the plots
```python plot_fedavg_gd.py```

## samples
![](/plot_output/pdf_2_gd.svg   " pdf_2_gd.svg ")
![](/plot_output/pdf_addedfedavg_gd_10_100.svg   " pdf_addedfedavg_gd_10_100.svg ")
