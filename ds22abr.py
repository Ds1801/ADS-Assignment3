
# -*- coding: utf-8 -*-
"""
Created on Thu Jan  5 17:03:36 2023

@author: ds22abr
"""

"""
CLUSTERING PART STARTS
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
import sklearn.metrics as skmet

agri = pd.read_csv("Agricultural_Land.csv")
print("\nagriulation: \n", agri)

agri = agri.drop(['Country Code', 'Indicator Name', 'Indicator Code', '1960'], axis=1)
print("\nNew agricultural: \n", agri)

agri = agri.fillna(0)
print("\nNew agricultural after filling null values: \n", agri)

agri = pd.DataFrame.transpose(agri)
print("\nTransposed Dataframe: \n",agri)

header = agri.iloc[0].values.tolist()
agri.columns = header
print("\nagricultural Header: \n",agri)

agri= agri.iloc[2:]
print("\nNew Transposed Dataframe: \n",agri)

agri_ex = agri[["India","Mexico"]].copy()

max_val = agri_ex.max()
min_val = agri_ex.min()
agri_ex = (agri_ex - min_val) / (max_val - min_val)
print("\nNew selected columns dataframe: \n", agri_ex)

ncluster = 4
kmeans = cluster.KMeans(n_clusters=ncluster)

kmeans.fit(agri_ex)

labels = kmeans.labels_

# extract the estimated cluster centres
cen = kmeans.cluster_centers_
print(cen)

# calculate the silhoutte score
print(skmet.silhouette_score(agri_ex, labels))

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))
col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
    
for l in range(ncluster): # loop over the different labels
    plt.plot(agri_ex[labels==l]["India"], agri_ex[labels==l]["Mexico"], marker="o", markersize=3, color=col[l])    
    
# show cluster centres
for ic in range(ncluster):
    xc, yc = cen[ic,:]  
    plt.plot(xc, yc, "dk", markersize=10)
plt.xlabel("India")
plt.ylabel("Mexico")
plt.show()    

print(cen)

df_cen = pd.DataFrame(cen, columns=["India", "Mexico"])
print(df_cen)
df_cen = df_cen * (max_val - min_val) + max_val
agri_ex = agri_ex * (max_val - min_val) + max_val
# print(df_ex.min(), df_ex.max())

print(df_cen)

# plot using the labels to select colour
plt.figure(figsize=(10.0, 10.0))

col = ["tab:blue", "tab:orange", "tab:green", "tab:red", "tab:purple", "tab:brown", \
"tab:pink", "tab:gray", "tab:olive", "tab:cyan"]
for l in range(ncluster): # loop over the different labels
    plt.plot(agri_ex[labels==l]["India"], agri_ex[labels==l]["Mexico"], "o", markersize=3, color=col[l])
    
# show cluster centres
plt.plot(df_cen["India"], df_cen["Mexico"], "dk", markersize=10)
plt.xlabel("India")
plt.ylabel("Mexico")
plt.title("Agricultural_Land (% of land area)")
plt.show()
print(cen)    



# In[ ]:
"""
CURVE FIT PART STARTS
"""

import pandas as pd
import scipy.optimize as opt
import matplotlib.pyplot as plt
#import errors as err

# function to read file
def readFile(y):
    '''
        This function is used to read csv file in original form and then 
        filling 'nan' values by '0' to avoid disturbance in data visualization
        and final results.

        Parameters
        ----------
        x : csv filename
    
        Returns
        -------
        gdp_growth : variable for storing csv file

    '''
    grow = pd.read_csv("Population.csv");
    grow = pd.read_csv(y)
    grow = grow.fillna(0.0)
    return grow

grow = pd.read_csv("Population.csv")
print("\nPOP: \n", grow)


grow = pd.DataFrame(grow)


grow = grow.transpose()
print("\nPOP: \n", grow)


header3 = grow.iloc[0].values.tolist()
grow.columns = header3
print("\nPopulation Header: \n",grow)

grow = grow["United Kingdom"]
print("\nPOP after dropping columns: \n", grow)


grow.columns = ["POP"]
print("\nPOP: \n",grow)


grow = grow.iloc[5:]
grow = grow.iloc[:-1]
print("\nPOP: \n",grow)


grow = grow.reset_index()
print("\nPOP index: \n",grow)


grow = grow.rename(columns={"index": "Year", "United Kingdom": "POP"} )
print("\nPOP rename: \n",grow)


print(grow.columns)
grow.plot("Year", "POP",label="POP")
plt.title("Population Growth")
plt.show()


def exponential(s, q0, h):
    """Calculates exponential function with scale factor n0 and growth rate g."""
    s = s - 1970.0
    x = q0 * np.exp(h*s)
    return x

print(type(grow["Year"].iloc[1]))
grow["Year"] = pd.to_numeric(grow["Year"])
print("\nPOP Type: \n", type(grow["Year"].iloc[1]))
param, covar = opt.curve_fit(exponential, grow["Year"], grow["POP"],
p0=(4.978423, 0.03))


grow["fit"] = exponential(grow["Year"], *param)
grow.plot("Year", ["POP", "fit"], label=["POP", "Fit"])
plt.legend()
plt.show()

# predict fit for future years
year = np.arange(1960, 2031)

print("\nForecast Years: \n", year)

forecast = exponential(year, *param)

plt.figure()
plt.plot(grow["Year"], grow["POP"], label="POP")
plt.plot(year, forecast, label="Forecast")
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("POP Growth")
plt.legend()    
plt.show()



def err_ranges(x, exponential, param, sigma):
    """
    Calculates the upper and lower limits for the function, parameters and
    sigmas for single value or array x. Functions values are calculated for 
    all combinations of +/- sigma and the minimum and maximum is determined.
    Can be used for all number of parameters and sigmas >=1.
    
    This routine can be used in assignment programs.
    """
    import itertools as iter
    
    # initiate arrays for lower and upper limits
    lower = exponential(x, *param)
    upper = lower
    
    uplow = []   # list to hold upper and lower limits for parameters
    for p,s in zip(param, sigma):
        pmin = p - s
        pmax = p + s
        uplow.append((pmin, pmax))
        
    pmix = list(iter.product(*uplow))
    
    for p in pmix:
        y = exponential(x, *p)
        lower = np.minimum(lower, y)
        upper = np.maximum(upper, y)
        print("\nLower: \n", lower)
        print("\nUpper: \n", upper)        
    return lower, upper







