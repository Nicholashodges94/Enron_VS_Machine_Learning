#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import all libraries

import sys
import os
import pickle
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import warnings

get_ipython().run_line_magic('pylab', 'inline')
# Change figure size into 8 by 6 inches
matplotlib.rcParams['figure.figsize'] = (8, 6)
warnings.filterwarnings('ignore')


from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit, GridSearchCV
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler


# In[2]:


# Load in the dataset
with open('final_project_dataset.pkl', 'rb') as data_file:
    data_dict = pickle.load(data_file)


# In[8]:


# Create dataframe to replace NaN message corpus values with median values
data_df = pd.DataFrame(data_dict)
data_df = data_df.transpose()


# In[9]:


# Replace string 'NaN' values with NULL
data_df.replace(to_replace='NaN', value=numpy.nan, inplace=True)


# In[12]:


# Replace NaN values for email datapoints with median value of feature
data_df.from_messages.fillna(data_df.from_messages.median(), inplace=True)
data_df.from_poi_to_this_person.fillna(data_df.from_poi_to_this_person.median(), inplace=True)
data_df.from_this_person_to_poi.fillna(data_df.from_this_person_to_poi.median(), inplace=True)
data_df.to_messages.fillna(data_df.to_messages.median(), inplace=True)
data_df.shared_receipt_with_poi.fillna(data_df.shared_receipt_with_poi.median(), inplace=True)


# In[13]:


# Replace remaining financial NULLs with 0
data_df.replace(to_replace=numpy.nan, value=0, inplace=True)


# In[14]:


# Convert back to dict
data_dict = data_df.to_dict(orient = 'index')


# In[15]:


# Create function for easy conversion into Pandas DataFrame with proper format
def allFeaturesFormat(data_dict):
    
    temp_list = []
    
    for name, features_values in data_dict.items():
        temp_dict = {}
        temp_dict['name'] = name     
        for feature, value in features_values.items():
            if feature in ['poi', 'email_address']:
                temp_dict[feature] = value
            else:
                temp_dict[feature] = float(value)
        temp_list.append(temp_dict)
    df = pd.DataFrame(temp_list)
    
    return df


# In[16]:


# Remove employee with no data
data_dict.pop('LOCKHART EUGENE E', 0)


# In[18]:


# Remove the obvious outliers
data_dict.pop('TOTAL', 0)
data_dict.pop("THE TRAVEL AGENCY IN THE PARK", 0)


# In[21]:


# Calculate fraction of messages to/from POIs
def computeFraction(poi_messages, all_messages):
    
    fraction = 0
    
    if poi_messages != 0 and all_messages != 0:
        fraction = poi_messages / all_messages
    
    return fraction


# In[22]:


# Calculate total compensation
def computeTotal(total_payments, total_stock_value):
    
    total = total_payments + total_stock_value
    
    return total


# In[23]:


# Create new features
for data_point in data_dict.values():    
    
    # Create feature fraction_from_poi
    from_poi_to_this_person = data_point['from_poi_to_this_person']
    to_messages = data_point['to_messages']
    fraction_from_poi = computeFraction(from_poi_to_this_person, to_messages)
    data_point['fraction_from_poi'] = fraction_from_poi

    # Create feature fraction_to_poi
    from_this_person_to_poi = data_point['from_this_person_to_poi']
    from_messages = data_point['from_messages']
    fraction_to_poi = computeFraction(from_this_person_to_poi, from_messages)
    data_point['fraction_to_poi'] = fraction_to_poi
    
    # Create feature total_compensation
    total_payments = data_point['total_payments']
    total_stock_value = data_point['total_stock_value']
    data_point['total_compensation'] = computeTotal(total_payments, total_stock_value)


# In[24]:


# Prepare features and labels for machine learning
work_df = allFeaturesFormat(data_dict)

# Subset dataframe with only numeric columns as features
features = work_df.drop(['poi', 'name', 'email_address'], axis=1)


# In[26]:


# Scale features
scaler = MinMaxScaler(copy=False)
scaler.fit(features)
scaler.transform(features)

# Labels are the 'POI' column
labels = work_df.poi


# In[28]:


# Split the whole dataset into stratified training/validation and test sets
# Training/validation set is used for building the model
# Test set is reserved for final model assessment
sss = StratifiedShuffleSplit(n_splits=1000, test_size=0.3, random_state=69)


# In[29]:


# Calculates average feature scores for each feature
def feature_scores(features, labels, cv, selector, score_attr):
    
    feature_names = features.columns.values
    feature_scores = defaultdict(list)
    
    for train_indices, test_indices in cv.split(features, labels):
        
        features_train = features.values[train_indices]
        features_test = features.values[test_indices]
        labels_train = labels.values[train_indices]
        labels_test = labels.values[test_indices]

        slct = selector
        slct.fit(features_train, labels_train)
        
        for feature_name, feature_score in zip(feature_names, getattr(slct, score_attr)):
            feature_scores[feature_name].append(feature_score)
    
    feature_scores = pd.DataFrame(feature_scores)
    feature_scores_avg = feature_scores.mean()
    feature_scores_avg = feature_scores_avg.sort_values(ascending=False)
    
    return feature_scores_avg


# In[30]:




# Feature selection using SelectKBest method
selector = SelectKBest(k='all')
feature_scores_avg = feature_scores(features, labels, sss, selector, 'scores_')
features_selectKBest_all = list(feature_scores_avg.index)


# In[31]:




# Feature selection using DecisionTree method
selector = DecisionTreeClassifier()
feature_scores_avg = feature_scores(features, labels, sss, selector, 'feature_importances_')
features_DecicionTree_all = list(feature_scores_avg.index)


# In[35]:


# Due to running time, 250 shuffles are used instead of 1000 shuffles for algorithm selection
sss_250 = StratifiedShuffleSplit(n_splits=250, test_size=0.3, random_state=69)


# In[43]:


# Create list of used features
features_DecisionTree_5 = features_DecicionTree_all[:5]


# In[44]:


param_grid = dict(base_estimator__max_depth = [None, 1, 3, 5],
                  base_estimator__class_weight = [None, 'balanced'],
                  n_estimators = [25, 50, 100])
gs = GridSearchCV(estimator=AdaBoostClassifier(base_estimator=DecisionTreeClassifier(random_state=69), 
                                                random_state=69),
                   param_grid=param_grid,
                   scoring='f1', 
                   cv=sss_250)
gs.fit(features[features_DecisionTree_5], labels)
clf = gs.best_estimator_


# In[ ]:


# Rename for export
my_classifier = clf
my_feature_list = features_DecisionTree_5
my_feature_list.insert(0, 'poi')
my_dataset = data_dict

# Export to pickle file
pickle.dump( my_classifier, open( "my_classifier.pkl", "wb" ) )
pickle.dump( my_feature_list, open( "my_feature_list.pkl", "wb" ) )
pickle.dump( my_dataset, open( "my_dataset.pkl", "wb" ) )