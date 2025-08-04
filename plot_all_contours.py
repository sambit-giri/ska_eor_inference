import numpy as np
import matplotlib.pyplot as plt
import corner
from matplotlib.lines import Line2D  # for custom legend handles

# Load the data
cov1 = np.loadtxt("paramcov_100_AAstar.npy")
cov2 = np.loadtxt("paramcov_1000_AAstar.npy")
cov3 = np.loadtxt("paramcov_1000_AA4.npy")

Fisher_Param = ['$T_{Vir}$','$R_{Max}$','$\zeta$']

# True values (manually specify)
fid = [pow(10,4.7),15,30]

data1 = np.random.multivariate_normal(fid, cov1, size=100000)
data2 = np.random.multivariate_normal(fid, cov2, size=100000)
data3 = np.random.multivariate_normal(fid, cov3, size=100000)

# Define colors for each dataset
colors = ["red", "blue", "green"]

# Initialize the corner plot with the first dataset
fig = corner.corner(data1, 
                    color=colors[0], 
                    plot_density=True, 
                    plot_contours=True, 
                    fill_contours=False,
                    plot_datapoints=False,
                    labels=Fisher_Param, 
                    truths=fid,
                    truth_color='black',
                    levels=(0.68,0.95),
                    label_kwargs={"fontsize": 14},
                    #title_fmt=".2f",
                    #show_titles=True
                    )

# Overlay the next two datasets
corner.corner(data2, fig=fig,
              color=colors[1], 
              plot_density=True, 
              plot_contours=True,
              plot_datapoints=False,
              levels=(0.68,0.95), 
              fill_contours=False)

corner.corner(data3, fig=fig,
              color=colors[2], 
              plot_density=True, 
              plot_contours=True,
              plot_datapoints=False,
              levels=(0.68,0.95), 
              fill_contours=False)

labels = [r'$100$h ${\rm AA}_{\star}$',r'$1000$h ${\rm AA}_{\star}$', r'$1000$h ${\rm AA}4$' ]
# Add manual legend
handles = [Line2D([0], [0], color=c, lw=6) for c in colors]
fig.legend(handles, labels, loc='upper right', fontsize=16, frameon=False)

fig.savefig('corner_all.png', format='png', dpi=300, bbox_inches='tight')
