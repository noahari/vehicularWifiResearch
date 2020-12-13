#!/usr/bin/env python
# coding: utf-8

# # Variable correlation analysisnotebook
# 
# 

# ## Data set
# 
# 
# 
# ### Node info
# 
# APs are stationary and colocated. Clients are mobile and colocated. Data flows in the AP -> client direction, so clients are also referred to as receivers.
# 
# | 802.11 type | AP lat | AP lon | channel | cntr. freq (MHz) | bw (MHz) |
# |-------------|--------|--------|---------|------------------|----------|
# | n | 41.111879 | -8.631146 | 6 | 2437 | 20 |
# | ac (wave 1) | 41.111879 | -8.631146 | 40 | 5200 | 40 |
# | ad | 41.111879 | -8.631146 | 1 | 60480 | 2160 |
# 
# Main clients, all positioned in the moving vehicle's roof, a VW Golf Mk 3:
# 
# | 802.11 type | radio | nr. antennas |
# |-------------|-----------------------------------|---|
# | n           | csl usb 2.0 wlan Adapter 300 Mbps | 2 |
# | ac          | tp-link archer t4uh               | 2 |
# | ad          | tp-link talon ad7200 (tp-03)      | ? |
# 
# Monitor nodes, used to collect frame parameters unavailable from main clients. They were colocated with the main clients. Details are:
# 
# | 802.11 type | radio | nr. antennas |
# |-------------|-------|--------------|
# | n | csl usb 2.0 wlan adapter 300 Mbps | 2 |
# | ac | tp-link talon ad7200 (tp-02) | 8 |
# | ad | tp-link talon ad7200 (tp-02) | ? |
# 
# Background clients, used to induce channel utilization diversity. They were static, and place near the AP. Details:
# 
# | 802.11 type | radio | nr. antennas | position |
# |-------------|-------|--------------|----------|
# | n | tp-link wn722n | 1 | ~2m away from AP |
# | n | tp-link wn722n | 1 | ~2m away from AP |
# | ac | csl usb 2.0 wlan Adapter 300 Mbps | 2 | ~2m away from AP |
# | ac | csl usb 2.0 wlan Adapter 300 Mbps | 2 | ~2m away from AP |
# | ad | tp-link talon ad7200 (tp-04) | ? | stopped vehicle's roof (so very near AP) |
# 
# 
# ### Trace info
# 
# The data is divided into traces. Each trace represents a uninterrupted experimental period.
# Traces vari
# 
# | trace nr | date | start time | #samples w/ successful reception (802.11n) | #samples w/ successful reception (802.11ac) | #samples w/ successful reception (802.11ad) | AP vehicle | nr. clients |
# |-----|------------|----------|------|------|-----| --- | --- |
# | 302 | 2019-08-20 | 10:28:45 | 3262 | 2787 | 423 | Honda Civic sedan (2001) | 1 |
# | 303 | 2019-08-20 | 11:26:23 | 3374 | 3027 | 312 | Honda Civic sedan (2001) | 2 |
# | 304 | 2019-08-20 | 12:39:46 | 1711 | 216  | 14  | Honda Civic sedan (2001) | 3 (n & ac) 2 (ad) |
# | 401 | 2019-08-22 | 10:19:24 | 1685 | 1681 | 545 | Peugeot Partner van (2002) | 1 |
# | 402 | 2019-08-22 | 10:48:26 | 2859 | 2827 | 764 | Peugeot Partner van (2002) | 2 |
# | 403 | 2019-08-22 | 11:39:36 | 135 | 135 | 116 | Peugeot Partner van (2002) | 2 |
# | 404 | 2019-08-22 | 11:42:50 | 114 | 114 | 53 | Peugeot Partner van (2002) | 2 |
# | 405 | 2019-08-22 | 11:45:07 | 2019 | 2019 | 507 | Peugeot Partner van (2002) | 2 |
# 
# 
# ### Feature info
# 
# Each row represents 1-second worth of data, and is comprised of many features (columns):
# 
# * **systime** : system time (1 Hz resolution) that this row refers to. All node clocks were synchronized through NTP.
# * **traceNr** : nr. of the trace the row belongs to. 
# * **lon** : longitude (in decimal degrees) reported by the client's GPS at `systime`.
# * **lat** : latitude (in decimal degrees) reported by the client's GPS at `systime`.
# * **receiverAlt** : altitude (in meters) reported by the client's GPS at `systime`.
# * **receiverX** : x coordinate of the client's position when space is discretized as a Cartesian plane and the AP is set to be the origin of the coordinate system. The x axis corresponds to east-west (positive values are east, negative values are west). Unit is meters.
# * **receiverY** : y coordinate of the client's position when space is discretized as a Cartesian plane, in meters.
# * **receiverDist** : great-circle distance (in meters) between AP and client, computed by the haversine formula.
# * **receiverSpeed** : speed (in m/s) reported by the client's GPS at `systime`.
# * **receiverId** : system-specific id for the client (in the vehicle).
# * **senderId** : system-specific id for the AP serving the client.
# * **isIperfOn** : 1 if the row's `systime` corresponds to a period where iperf (the app used to send and receive data) is known to have been running on the receiver side, and 0 otherwise. Should always be 1.
# * **isInLap** : 1 if this row's systime has been marked as being part of a time period where clients were doing laps around the APs, and 0 otherwise. **This feature is unreliable and should be ignored.**
# * **rssiMean** : the mean of the RSSI (Received Signal Strength Indicator) values of AP data frames received by the client during the 1-second period the row refers to. Measured in dBm. **For 802.11ad, we retrieve RSSI from sector sweep feedback frames.**
# * **snrMean** : the mean of the SNR (Signal-to-Noise Ratio) values of AP data frames received by the client during the 1-second period the row refers to. Only valid for 802.11ad rows.
# * **channelFreq** : center frequency of the WiFi channel used, in MHz.
# * **channelBw** : channel width of the WiFi channel used, in MHz.
# * **channelUtil** : percentage of time the wireless medium was sensed to be busy during the 1-second period the row refers to. A value of -1 indicates an absense of data. **In traces numbered 40x, the 802.11n and ac routers didn't log channel busy time, and as such we had to approximate channel util. based on x,y coordinates and nr. of active clients.**
# * **dataRateMedian** : the median of the data rate values of AP data frames received by client second period the row refers to, in Mbps.
# * **dataRateMean** : the mean of the data rate values of AP data frames received by the client during the 1-second period the row refers to, in Mbps.
# * **nBytesReceived** : total number of bytes received by the client from the ap during the 1-second period systime period the row refers to, in Bytes.
# * **tghptConsumer**: throughput reported by the receiving end of iperf, during the 1-second period the row refers to, in Mbps.
# * **wifiType** : IEEE 802.11 network type, i.e. n, ac or ad.
# * **nrClients** : total number of clients operating on the same channel and bandwidth as `receiverId` during the 1-second period the row refers to.
# * **nRetries** : total number of frames marked as being retries (i.e. retransmissions due to loss) during the 1-second period the row refers to.
# * **meanBeaconRssi** : the mean of the RSSI (Received Signal Strength Indicator) values of AP beacon frames received by the client during the 1-second period the row refers to. Measured in dBm.
# * **meanInterBeaconTime** : the mean of the interval between AP beacon frames received by the client during the 1-second period the row refers to. Measured in seconds.
# * **nBeacons** : the total number of beacons received during the 1-second period the row refers to.
# 

# ### Dataset loading 
# 

# In[1]:


import warnings
import pandas
import sys
import numpy as np

warnings.filterwarnings('ignore')

dframe = pandas.read_csv("./wifi-exp-log-summary.csv")

dframe = dframe.reset_index(drop = True) # creates index column, numbered from 0 to n-1

print("Loaded data set (%d rows)" % (len(dframe)))

# let's now filter out stuff we don't want
dframe = dframe.loc[dframe['isIperfOn'] == 1] # filter out periods when iperf was off
dframe.drop(['isIperfOn'], axis = 1, inplace = True)


# Split up days
dfday1 = dframe[:26130]
dfday2 = dframe[26130:]
dfday1['systime'] = dfday1['systime'] - dfday1.iloc[0]['systime']
dfday2['systime'] = dfday2['systime'] - dfday2.iloc[0]['systime']
#print(dfday1['systime'])
# print(dfday2['systime'])


dframe = dframe.reset_index(drop = True) # creates index column, numbered from 0 to n-1

dframes = []
for df in [dfday1, dfday2]:
    for wifi in ['n', 'ac', 'ad']: 
        dframeNew = df[df['wifiType'] == wifi]  # focus on a particular type of wifi
        dframes.append(dframeNew)

    
#print("After filtering, left with %d rows\n" % (len(dframe)))

#print("Sample summary:")
#with pandas.option_context('display.max_rows', None, 'display.max_columns', None):
  #print(dframe.groupby(['wifiType']).size().reset_index(drop = False, name = 'nr-samples'))
  #print(dframe['systime'])


# In[16]:


# Plots of different wifi types vs datarate over time
import matplotlib.pyplot as plt

for wifi in dframes:
 for xval in ['systime', 'receiverDist', 'receiverSpeed']:
    fig = plt.figure(figsize=(15, 5))
    plt.title(f"802.11{wifi['wifiType'].iloc[0]} experiment for 2019/08/20")
    plt.xlabel(xval)
    plt.ylabel("Mean data rate (Mbps)")
    plt.grid(True, ls="dotted", lw=0.75)
    plt.xlim([min(wifi[xval]), max(wifi[xval])])
    plt.ylim([min(wifi["dataRateMean"]), max(wifi["dataRateMean"]) + 10])
    #line graph if data is temporal, otherwise scatter plot
    if(xval == "systime"):
        plt.plot(wifi[xval], wifi["dataRateMean"])
    else:
        plt.plot(wifi[xval], wifi["dataRateMean"],'o')
    plt.show()


# In[17]:


for df in dframes:  
    corMat= df.corr(method="kendall")
    display(corMat.style.background_gradient(cmap="coolwarm", axis=None).set_precision(2))


# In[25]:


import sklearn

dframeNew = dframe[dframe['wifiType'] == 'n']
#print(dframeNew)
dframeNew = dframeNew.drop(['wifiType','senderId', 'receiverId'], axis = 1)

#print(dframeNew)

x = dframeNew.values # the values in the data frame
from sklearn.preprocessing import StandardScaler
x = StandardScaler().fit_transform(x) # subtract mean and
# divide by standard deviation (yields mean=0, variance=1)

from sklearn import decomposition
pca = decomposition.PCA(n_components=0.95) # enough comps to explain 95% var
pca.fit(x)

print("Found", pca.n_components_, "PCA components:", pca.explained_variance_)
print("Explained variance ratios:", pca.explained_variance_ratio_)
print("Total:", sum(pca.explained_variance_ratio_))
print("Components:", pca.components_)

xtrans = pca.transform(x) # apply dimensionality reduction to x
pcaDf = pandas.DataFrame(data = xtrans) # create a data frame out of x


# In[8]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

target = 'dataRateMean' # variable we're trying to predict
for wifiType in ('n', 'ac', 'ad'): # separate data by wifi type
    # filter out unwanted rows
    fdf = dfday1.loc[dfday1['wifiType'] == wifiType]
    featureList = ['receiverDist', 'rssiMean', 'receiverSpeed'] # features of interest
    # replace rssi with snr for ad, if needed
    if wifiType == 'ad' and 'rssiMean' in featureList:
        featureList.remove('rssiMean')
        featureList.append('snrMean')



    featDropList = [] # drop the features we won't be using
    for feature in fdf.columns:
        if feature not in featureList and feature != target:
            featDropList.append(feature)
    fdf.drop(featDropList, axis=1, inplace=True)
    fdf.reset_index(inplace=True, drop=True)
    print("Predicting 802.11%s %s using features: %s" %
    (wifiType, target, featureList))
    # split data set into features and target labels
    x = fdf.drop(target, axis=1) # x contains all the features of interest
    y = fdf[target] # y contains only the target label
    # split data into training and test subsets:
    # - xTrain and yTrain contain features and labels for training
    # - xTest and yTest contain features and labels for testing
    # test_size=0.3 means 30% data for testing
    # random_state=1, is the seed value used by the random number generator
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3,
    random_state=1)
    

    regLinr = LinearRegression() # create linear regression model
    regLinr.fit(xTrain, yTrain) # train model
    print("Linear regression, coefficients: \n", regLinr.coef_)
    yPred = regLinr.predict(xTest) # predict on test data
    yPred = yPred.clip(min=0) # data rate can't be negative, so make it >= 0
    print("Mean-Square Error: %.2f" % mean_squared_error(yTest, yPred))
    print("Coefficient of determination: %.2f" %     r2_score(yTest, yPred)) # 1 is perfect prediction
    #plt.scatter(xTest, yTest, color='black')
    #plt.plot(xTest, yPred, color='blue', linewidth=3)
   # plt.xticks(())
    #plt.yticks(())
  #  plt.show()


# In[51]:


from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sn

def classify(fdf):
    # split data set into features and target labels
    x = fdf.drop(target, axis=1) # x contains all the features of interest
    y = fdf[target] # y contains only the target label
    # split data into training and test subsets:
    # - xTrain and yTrain contain features and labels for training
    # - xTest and yTest contain features and labels for testing
    # test_size=0.3 means 30% data for testing
    # random_state=1, is the seed value used by the random number generator
    xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3,
    random_state=1)
    print(f"yTrain: {yTrain}")
    clfKnn = KNeighborsClassifier()
    clfKnn.fit(xTrain, yTrain) # train classifier
    yPred = clfKnn.predict(xTest) # predict on the unseen data
    print("Accuracy of k-nearest neighbors:", accuracy_score(yTest, yPred))
    print(classification_report(yTest, yPred))
    # display a confusion matrix
    print("Confusion matrix:")
    cmat = confusion_matrix(yTest, yPred)
    plt.figure(figsize=(6.5, 6.5)) # new figure
    sn.set(font_scale=1.4) # for label size
    sn.heatmap(cmat, annot=True, annot_kws={"size": 13}, fmt='g')
    plt.show()
    
def exp_binulate(factor, nbins):
    return [0] + [factor**n for n in range(0, nbins - 1)]
    

target = 'dataRateMean' # variable we're trying to predict
for wifiType in ('n', 'ac', 'ad'): # separate data by wifi type
    fdf = dframe.loc[dframe['wifiType'] == wifiType] # filter out unwanted rows
    featureList = ['rssiMean', 'receiverDist', 'receiverSpeed', 'meanBeaconRssi'] # features of interest
    # replace rssi with snr for ad, if needed
    if wifiType == 'ad' and 'rssiMean' in featureList:
        featureList.remove('rssiMean')
        featureList.append('snrMean')
    featDropList = [] # drop the features we won't be using
    for feature in fdf.columns:
        if feature not in featureList and feature != target:
            featDropList.append(feature)
    fdf.drop(featDropList, axis=1, inplace=True)
    fdf.reset_index(inplace=True, drop=True)
    print("Classifying 802.11%s %s using features: %s\n" %
    (wifiType, target, featureList))
    # discretize target variable
    # start by printing some information about the bins
    nquantiles = 4
    uniBins = pandas.qcut(fdf[target], q=nquantiles, duplicates='drop').unique()
    uniBins = uniBins.ravel() # convert data to an array for sorting
#     uniBins.sort()
    print("Discretizing", target, "using", nquantiles,
    "quantiles, yielding bins:\n", uniBins, "\n")
    target_orig = fdf[target]
    
    print("TG ", target_orig[1610])
    
    fdf[target] = pandas.qcut(fdf[target], q=nquantiles, labels=False,
    duplicates='drop')
    
#     classify(fdf)
    
    fdf[target] = pandas.cut(target_orig, bins=exp_binulate(2, 20), duplicates='drop')
    
    print(f"NEW: {fdf[target]}")
    
    # Error here, negative numbers in target_orig? Some columns NAN after cut?
    classify(fdf)
    
    
    
#     # split data set into features and target labels
#     x = fdf.drop(target, axis=1) # x contains all the features of interest
#     y = fdf[target] # y contains only the target label
#     # split data into training and test subsets:
#     # - xTrain and yTrain contain features and labels for training
#     # - xTest and yTest contain features and labels for testing
#     # test_size=0.3 means 30% data for testing
#     # random_state=1, is the seed value used by the random number generator
#     xTrain, xTest, yTrain, yTest = train_test_split(x, y, test_size=0.3,
#     random_state=1)
#     clfKnn = KNeighborsClassifier()
#     clfKnn.fit(xTrain, yTrain) # train classifier
#     yPred = clfKnn.predict(xTest) # predict on the unseen data
#     print("Accuracy of k-nearest neighbors:", accuracy_score(yTest, yPred))
#     print(classification_report(yTest, yPred))
#     # display a confusion matrix
#     print("Confusion matrix:")
#     cmat = confusion_matrix(yTest, yPred)
#     plt.figure(figsize=(6.5, 6.5)) # new figure
#     sn.set(font_scale=1.4) # for label size
#     sn.heatmap(cmat, annot=True, annot_kws={"size": 13}, fmt='g')
#     plt.show()
   
  


# ## NOTES on the Heat Map
# When we run classify over just RSSI mean, the precision, recall, and f1-score are all lower than when we run over distance and speed as well, telling us that these are strong indicators for data rate mean as well.
# 
# Without meanBeaconRssi
#            0       0.93      0.93      0.93      1146
#            1       0.74      0.79      0.77      1158
#            2       0.61      0.59      0.60      1179
#            3       0.75      0.73      0.74      1174
# 
# The final section requires some correction for NaN values, cleaning the data set/pruning prior to runtime.

# In[ ]:




