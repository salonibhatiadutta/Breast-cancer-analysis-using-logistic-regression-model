#!/usr/bin/env python
# coding: utf-8

# In[98]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
plt.rc("font",size=14)
get_ipython().run_line_magic('matplotlib', 'inline')

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score# for tuning parameter
from sklearn.feature_selection import RFE
from sklearn import metrics 
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold

 

import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid",color_codes=True)
data= pd.read_csv(r'C:\Users\Admin\.jupyter\breastCancer_data.csv',header=0)
data.drop('Unnamed: 32', axis=1,inplace=True)    # dropping the last unnamed column from analysis
print(list(data.columns))
print(data.shape)
print("\n")


#defining function to find correlation 
def corr_(x=[], *args):
    correlation= x.corr() # .corr is used for find corelation
    return(correlation)

#a) data cleaning: Exploring the data on the basis of 10 real valued features and categorically dividing them into mean, se, worst
mean_feature = list(data.columns[2:11])
se_feature= list(data.columns[12:21])
worst_feature= list(data.columns[22:31])
print("MEAN FEARTURES:\n",mean_feature)
print("\n")
print("SE FEARTURES:\n",se_feature)
print("\n")
print("WORST FEARTURES:\n",worst_feature)
print("\n")

# b) GRAPH PLOTTING: plotting for malignant and benign class of diagnosis
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
#print(data['diagnosis'])
data.drop('id', axis=1,inplace=True)   # dropping id column as it is not required in analysis
#print(data.describe())
sns.countplot(data['diagnosis'],label='count')
print(data['diagnosis'].value_counts())

#c) Features_selection: choosing features which are different and reflecting more information. For that we will try to find correlation 
#   between the features. If higly correlated then we will use only one of them. values near to 1 show high correlation and near to 0 show no or less correlation
# The heatmap plot shows higher level of correlation between radius, perimeter and area so considering radius as the unique feature
#Also concavity, compactness and concave points also has higher level of correlation so considering compactness
plt.figure(figsize=(12,12))
sns.heatmap(corr_(data[mean_feature]), cbar = True,  square = True, annot=True,  xticklabels=mean_feature , yticklabels= mean_feature,cmap= 'coolwarm') 
#plt.figure(figsize=(12,12))
#sns.heatmap(corr_(data[se_feature]), cbar = True,  square = True, annot=True,  xticklabels= se_feature, yticklabels= se_feature,cmap= 'coolwarm')
#print("SELECTED FEATURES FROM SE: ")
#print(" 1. Radius\n 2. Texture\n 3. Smoothness\n 4. Compactness")
#plt.figure(figsize=(12,12))
#sns.heatmap(corr_(data[worst_feature]), cbar = True,  square = True, annot=True,  xticklabels= worst_feature, yticklabels= worst_feature,cmap= 'coolwarm')
#print("SELECTED FEATURES FROM WORST: ")
#print(" 1. Radius\n 2. Texture\n 3. Smoothness\n 4. Compactness")
# further filtering the data to check which features are best fitted to understand the model and can best describe the two classes of diagnosis
# Plotting a scatter plot 
color_function = {0: "blue", 1: "red"} # 0 is benign and plotted by blue, 1 is maligenen and plotted by red
colors = data["diagnosis"].map(lambda x: color_function.get(x)) # mapping the color function with diagnosis column
pd.plotting.scatter_matrix(data[mean_feature], c=colors, alpha = 0.5, figsize = (10, 10))
#texture_mean, smoothness_mean, symmetry_mean and fractal_dimension_mean cannot be used for classify two category because 
#both category are mixed there is no separable plane So we can remove them from our selected features list
# also area, radius and perimeter shows linear relationship
print("SELECTED FEATURES FROM MEAN: ")
print(" 1. Radius\n 2. Perimeter\n 3. Area\n 4. Compactness\n 5. Concave Points\n")
feature_selected = ['radius_mean','perimeter_mean','area_mean','compactness_mean','concave points_mean','radius_se','perimeter_se','area_se','compactness_se','concave points_se','radius_worst','perimeter_worst','area_worst','compactness_worst','concave points_worst']
#Similar analysis is drawn for se_features and worst_features

# d) Building a model: 
#DIVIDING DATA INTO TEST AND TRAINING PART
x= data[feature_selected]
y= data['diagnosis']           #target column
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.4)
x_train.head()
model=LogisticRegression()
rfe= RFE(model,15)
model.fit(x_train,y_train)

#e) Predicting the results on test data set and calculating accuracy and cross validation score
y_pred= model.predict(x_test)
print("accuracy of logistic regression : {:.2f}".format(model.score(x_test,y_test)))

kfold= KFold(n_splits=10,random_state=9)
modelCV= LogisticRegression()
scoring= 'accuracy'
results= cross_val_score(modelCV, x_train,y_train,cv=kfold,scoring=scoring)
print("10_fold cross validation average accuracy: %.3f"% (results.mean()))

# calculating confusion matrix
print("CONFUSION MATRIX:\n",confusion_matrix(y_test, y_pred))


# In[ ]:





# In[ ]:





# In[ ]:




