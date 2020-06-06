import pyspark.sql.functions as F
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import json as j
import sys
import math
from tqdm import tqdm

# set random seed for numpy
np.random.seed(42)

# get command line arguments
FEATURES_FILE = sys.argv[1]
SAMPLES = int(sys.argv[2])

# read product features
with open(FEATURES_FILE, 'r') as f:
    productFeatures = j.load(f)

# Product features is of
pf = np.array([p[1] for p in productFeatures])
sampleIndices = np.random.randint(len(pf), size=SAMPLES)
sampledFeatures = pf[sampleIndices, :]
# perform tsne
embeddedFeatures = TSNE(n_components=2, verbose=1, learning_rate=50, perplexity=int(math.sqrt(SAMPLES))).fit_transform(sampledFeatures)

# different color for each genre
# uncomment these if table join of features and genres is successfull
# genresColors = {'poetry': '#1f77b4',
#                 'comics, graphic': '#ff7f0e',
#                 'young-adult': '#d62728',
#                 'history, historical fiction, biography': '#2ca02c',
#                 'mystery, thriller, crime': '#9467bd',
#                 'fiction': '#8c564b',
#                 'non-fiction': '#e377c2',
#                 'fantasy, paranormal': '#7f7f7f',
#                 'children': '#bcbd22',
#                 'romance': '#17becf'
#                 }

# read book genre map file
# with open('genre_books2.json', 'r') as f:
#     genre_books = j.load(f)

# convert list to set for faster search
# for k, v in genre_books.items():
#     genre_books[k] = set(v)

fig, ax = plt.subplots()
# plot the features
for idx, x in tqdm(enumerate(embeddedFeatures)):
    # for k, v in genre_books.items():
    #     if productFeatures[sampleIndices[idx]][0] in v:
    ax.scatter(x[0], x[1], c="#0000ff")

plt.show()
