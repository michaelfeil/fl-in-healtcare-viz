# Visualizations for the Seminar "Federated Learning in Healtcare"
TU Munich, Website https://albarqouni.github.io/courses/flhsose2021/

Visualizes gradient descent (line-search, non-stochastic) for non-iid Gaussian distributions with Federated Averaging.
3D Visualizations using Matplotlib.

# Usage
## installation: local
Tested with this conda env, may run with most versions of numpy, scipy and matplotlib. 
plotly and mpld3 only need for exporting the plots to html. 

```conda create -n newcondaenv python==3.6.* --file requirements.txt```

## run the plots
```python plot_fedavg_gd.py```

## samples
![](/plot_output/pdf_2_gd.svg   " pdf_2_gd.svg ")
![](/plot_output/pdf_addedfedavg_gd_10_100.svg   " pdf_addedfedavg_gd_10_100.svg ")
