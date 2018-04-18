# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 02:11:00 2016

@author: admin
"""

import time
import datetime as dt
from datetime import datetime, date, time, timedelta
import numpy as np
from sklearn.cluster import KMeans
import matplotlib
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import style
style.use('ggplot')


import pandas as pd
import numpy as np
import datetime as dt
from sklearn import preprocessing

#Import data into pandas dataframe gtd
sdf = pd.DataFrame(pd.read_excel('sales.xlsx'))
sdf = sdf.drop(sdf[sdf.Quantity < 1].index)
sdf = sdf[np.isfinite(sdf['CustomerID'])]
          

sdf['Sales'] = sdf['Quantity']*sdf['UnitPrice']          
apsales = sdf[['CustomerID', 'Sales']]
          
##############################################################################
from bokeh.charts import Bar, output_file, show
from bokeh.charts.attributes import cat, color, CatAttr
from bokeh.models import Axis
from bokeh.palettes import Inferno7
sdf['count'] = 1

#Graph1 : total number of attacks per year per attack type
bar = Bar(sdf,
          values='count',
          label='Description',
          #stack=CatAttr(columns='attacktype1_txt',  sort = True),
          legend='top_left',
          title="Fig 1. Total number of victims per year, stacked by Attack Type", 
          palette=Inferno7, 
          xlabel="Year", ylabel="Total no. of attacks",
          yscale = 'Auto')

output_file("stacked_bar1.html", title="Total number attacks per year, stacked by Attack Type")
show(bar)       
          
################################################################################
NOW = dt.datetime(2011,12,10)


rfmTable = sdf.groupby('CustomerID').agg({'InvoiceDate': lambda x: (NOW.date() - x.max().date()), # Recency
                                        'InvoiceNo': lambda x: len(x),      # Frequency
                                        'Sales': lambda x: x.sum()}) # Monetary Value
cnames = rfmTable.index

def Date_Extract(x):
    d= int(x.days)
    return d
rfmTable['Day'] = rfmTable['InvoiceDate'].apply(Date_Extract)
del rfmTable['InvoiceDate']


rfmTable.rename(columns={'Day': 'Recency', 
                         'InvoiceNo': 'Frequency',
                         'Sales': 'Monetary_value'}, inplace=True)
rfm = rfmTable[['Recency', 'Frequency', 'Monetary_value']]
rfm['Monetary_value'] = rfm['Monetary_value'].astype(int)

x = rfm.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
rfm_norm = pd.DataFrame(x_scaled)

rfm_norm.describe()


from scipy import stats
rfm_norm = rfm_norm[(np.abs(stats.zscore(rfm_norm)) < 3).all(axis=1)]

rfm_norm.columns = ['recency', 'frequency', 'monetary']

rfmmat = rfm_norm.as_matrix()

clustnum=7

kmeans = KMeans(n_clusters=clustnum, max_iter=100)
kmeans.fit(rfm_norm)



centroids = kmeans.cluster_centers_
labels = kmeans.labels_


print ("centroids : ")
print (centroids)
print ("labels : ")
print (labels)



color = ["r", "g", "b", "y", "m", "c", "k"]
from collections import Counter
c = Counter(labels)

for clustnum in range(clustnum):
  print("Cluster {} contains {} samples".format(clustnum, c[clustnum]))


fig = plt.figure()
ax = fig.gca(projection='3d')

for i in range(len(rfmmat)):
    print("coordinate:",rfmmat[i], "label:", rfmmat[i])
    print ("i : ",i)
    print ("color[labels[i]] : ",color[labels[i]])
    ax.scatter(rfmmat[i][0], rfmmat[i][1], rfmmat[i][2], c=color[labels[i]])
    


ax.scatter(centroids[:, 0],centroids[:, 1], centroids[:, 2], marker = "x", s=150, linewidths = 5, zorder = 100, c = color)

plt.show()

#############################################################################

from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(rfm_norm, 'ward')

fig2  = plt.figure()
plt.title('Hierarchical Clustering Dendrogram (truncated)')
plt.xlabel('sample index or (cluster size)')
plt.ylabel('distance')
dendrogram(
    Z,
    truncate_mode='lastp',  # show only the last p merged clusters
    p=7,  # show only the last p merged clusters
    leaf_rotation=90.,
    leaf_font_size=12.,
    show_contracted=True,  # to get a distribution impression in truncated branches
)
plt.show(fig2)

from scipy.cluster.hierarchy import fcluster
k=7
clusters = fcluster(Z, k, criterion='maxclust')
clusters
rfm_norm['hclust'] = clusters

################################################################################


