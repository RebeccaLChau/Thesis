from astropy.io import fits
import numpy as np
import pandas as pd
import datetime
import math
import matplotlib.pyplot as plt
import matplotlib
import matplotlib as mpl
from mpl_toolkits import mplot3d
from mpl_toolkits.mplot3d import Axes3D
import hdbscan
import pickle
from MulticoreTSNE import MulticoreTSNE as TSNE
from os.path import dirname, join as pjoin
import scipy.io as sio
from scipy.io import readsav
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import RobustScaler
from sklearn.neighbors import NearestNeighbors as nn
from sklearn.model_selection import train_test_split
import sys

# Data directory
direct = r'C:\Users\reb\New1\sav'
figs = r'C:\Users\reb\New1\figs'

# Trained models names
FNAME = 'NewBatch32LR54E5000'

print(FNAME)
print('--------------------------------------------------------------')

path = direct
hdu_lvec = fits.open(path + '/LVec_' + FNAME + '.fits')
hdu_mu = fits.open(path + '/Mu_' + FNAME + '.fits')
hdu_std = fits.open(path + '/Std_' + FNAME + '.fits')

# Load latent vectors (X1)
X1 = hdu_lvec[0].data

# ============================================================================ #
# Check t-SNE Data Shape and Values                                            #
# ============================================================================ #

# Ensure the t-SNE result has data
assert X1.shape[0] > 0, "No data points for t-SNE"  # Ensure there are data points
print("Shape of X1:", X1.shape)  # Print the shape to verify it looks correct

# Optionally, print the first few rows to inspect the data values
print("First few rows of X1:\n", X1[:5])

# Check if there are NaN or Inf values in X1
assert not np.any(np.isnan(X1)), "Latent vectors contain NaN values"
assert not np.any(np.isinf(X1)), "Latent vectors contain Inf values"

# ============================================================================ #
#  Perform t-SNE                                                              #
# ============================================================================ #

it = 3000
p = 480
t = 0.50

# Run t-SNE
X_embedded1 = TSNE(
    learning_rate=200,
    metric='euclidean',
    n_components=2,
    n_iter=it,
    n_jobs=60,
    random_state=24,
    perplexity=p,
    angle=t,
    verbose=1).fit_transform(X1)

# ============================================================================ #
# Check t-SNE Results and Plot                                                #
# ============================================================================ #

# Ensure the t-SNE result has valid shape and values
assert X_embedded1.shape[0] > 0, "No data points for t-SNE"  # Ensure there are data points
print("Shape of X_embedded1:", X_embedded1.shape)  # Print the shape to verify it looks correct

# Optionally, print the first few rows to inspect the data values
print("First few rows of X_embedded1:\n", X_embedded1[:5])

# Check if there are NaN or Inf values in X_embedded1
assert not np.any(np.isnan(X_embedded1)), "t-SNE data contains NaN values"
assert not np.any(np.isinf(X_embedded1)), "t-SNE data contains Inf values"

# Get the axis limits for plotting
xmax = X_embedded1[:, 0].max()
xmin = X_embedded1[:, 0].min()
ymax = X_embedded1[:, 1].max()
ymin = X_embedded1[:, 1].min()

# Print values to debug the plot range
print(f"xmin: {xmin}, xmax: {xmax}, ymin: {ymin}, ymax: {ymax}")

# Create the plot
plt.figure(figsize=(8, 8))

# Plotting the points
plt.scatter(X_embedded1[:, 0], X_embedded1[:, 1], s=5, c='blue')  # Use a fixed color for now

plt.title('2D MCTSNE Embedding', fontsize=16)
plt.xlabel('x-coordinate', fontsize=14)
plt.ylabel('y-coordinate', fontsize=14)

# Set axis limits with some margin around the data points
plt.xlim(xmin - 5, xmax + 5)
plt.ylim(ymin - 5, ymax + 5)

# Check if the scatter plot actually has points
print(f"Number of points plotted: {len(X_embedded1)}")

# Save the plot to a file
plot_filename = f'{figs}/tsne_{FNAME}_embedding.png'  # Change extension or format if needed
plt.savefig(plot_filename, bbox_inches='tight')

print(f"Plot saved to {plot_filename}")

# Display the plot (blocking the script until the plot window is closed)
plt.show()

# ============================================================================ #
# Save t-SNE Embedding to FITS File                                           #
# ============================================================================ #

# Save the t-SNE embedding as a FITS file
hdu = fits.PrimaryHDU(data=X_embedded1)
hdu.writeto(path + '/tsne_' + str(p) + '_' + str(it) + '_' + str(t) + '_lvec_' + FNAME + '.fits', overwrite=True)



# ---------------------------------------------------------------------------- #
#  HDBSCAN                                                                     #
# ---------------------------------------------------------------------------- #
# ---------------------------------------------------------------------------- #
#  HDBSCAN                                                                     #
# ---------------------------------------------------------------------------- #
# HAND IT TO DBSCAN
# Assign the t-SNE result to X for HDBSCAN clustering
X = X_embedded1

# Perform HDBSCAN clustering
cluster = hdbscan.HDBSCAN(min_cluster_size=5, min_samples=1).fit(X)

unique_labels = set(cluster.labels_)
print(unique_labels)

# Create a color map for the clusters
colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
labels = cluster.labels_.astype(int)

# Plot the clusters with HDBSCAN labels in the t-SNE 2D space
plt.figure(figsize=(8, 8))
scatter = plt.scatter(X[:, 0], X[:, 1], s=5, c=labels, cmap='hsv')

# Title and axis labels
plt.title('Clusters in a MCTSNE 2D embedding', fontsize=16)
plt.xlabel('x-coordinate', fontsize=14)
plt.ylabel('y-coordinate', fontsize=14)

# Set axis limits for better visualization
plt.xlim(-10., 10.)
plt.ylim(-10., 10.)

# Optional: Add a color bar to show the mapping of labels to colors
plt.colorbar(scatter)

# Save the plot with the clustering information
cluster_plot_filename = f'{figs}/tsne_{FNAME}_embedding_with_clusters.png'
plt.savefig(cluster_plot_filename, bbox_inches='tight')

print(f"Clustered plot saved to {cluster_plot_filename}")

# Show the plot
plt.show()

# Save HDBSCAN cluster labels to FITS file
hdu = fits.PrimaryHDU(data=labels)
hdu.writeto(direct + 'labels_lvec_' + FNAME + '.fits', overwrite=True)

labels_2d = cluster.labels_.reshape(-1, 1)  # Ensure 2D array for FITS
output_file = r'C:\Users\reb\New1\sav\labels_lvec_NewBatch32LR54.fits'

try:
    hdu = fits.PrimaryHDU(data=labels_2d)
    hdu.writeto(output_file, overwrite=True)
    print(f"Labels saved successfully to {output_file}")
except Exception as e:
    print(f"Error saving FITS file: {e}")
# Exit the script
sys.exit()
