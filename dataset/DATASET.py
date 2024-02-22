import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import warnings
import random
import pandas as pd
from typing import List
from sklearn import datasets
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import rand_score
from sklearn.datasets import make_moons
from sklearn.datasets import make_circles

#DATASET(1)
X, y = make_blobs(centers=3, n_samples=300, n_features=2, shuffle=True, random_state=112)

#DATASET(2)
X1, y1 = make_blobs(n_samples=300 , random_state=112)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X1 = np.dot(X1, transformation)

#DATASET(3)
X2, y2 = make_moons(n_samples=300, noise=0.1, random_state=112)

#DATASET(4)
X3, y3 =make_circles(n_samples=300, noise=0.05, factor=0.5, random_state=112)
