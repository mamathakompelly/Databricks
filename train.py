#!/usr/bin/env python
# coding: utf-8

# In[1]:

# Import Python libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plot
import seaborn as sns
from scipy.stats import chi2_contingency
from scipy.stats import chi2

from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
encoder=OneHotEncoder(handle_unknown='ignore')
from pandas import to_datetime
import itertools
import warnings
import datetime
warnings.filterwarnings('ignore')

from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, r2_score
from sklearn.model_selection import train_test_split
import shap
import xgboost
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score

import plotly.offline as py
py.init_notebook_mode(connected=False)

import plotly.io as pio
import plotly.express as px

import plotly.graph_objects as go

pio.templates.default = "presentation"

pd.options.plotting.backend = "plotly"

import shap
from numpy import arange
from numpy import argmax
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve

import joblib

# Load user defined helper methods
from myutils import *
from file_methods import *

# Load configurations (filespaths)
from config import Config
conf = Config()


# Data loading

# ELIZA SURVEY DATA

import os
print(os.getcwd())
data=pd.read_csv(Config.raw_data_folder + "eliza20210901.csv")
data.shape
data.FELT_ISOLATED.isnull().sum()

# STRATIFICATION DATA
sd=pd.read_csv(Config.raw_data_folder+"member_strat20210901.csv")
Stratificationdata=sd.rename(columns={"Consistent_member_id":"consistent_member_id"})
sd.shape

# Data Preprocessing
# Dropping of columns having more than 80% of null values from ELIZA Survey Data 
null_cols_list = get_cols_threshold_on_null(data)
data = data.drop(null_cols_list, axis=1)

# Deriving Target Variable
df1=derive_target_variable(data)
# ELIZA SURVEY data ready to merge with stratification data
df1.shape

# Merging of Eliza AND Stratification Data
MERGEDDATA= pd.merge(df1,Stratificationdata, on='MemberNumber')

# Adding column PCP_Visit_Counts- no of times each member visited PCP
MERGEDDATA['PCP_Visit_Counts']=MERGEDDATA.groupby(['MemberNumber'])['PCP_Visit_Dt'].transform('count')

# Important features are in data having less than 23% null values- 
mergeddata_more_than23_nulls=get_cols_threshold_on_null(MERGEDDATA, threshold=23)

testing_modeldata_nona=MERGEDDATA.drop(mergeddata_more_than23_nulls,axis=1)

testing_modeldata_nona.isnull().sum()

# Removing null values from selected features

df_filtered_dropna=testing_modeldata_nona.dropna()

df_filtered_dropna.shape

# Categorical and Numerical Columns

cat_cols = []

for i in df_filtered_dropna.columns:
    if df_filtered_dropna[i].dtype.name == 'object':
             cat_cols.append(i)    
 
(cat_cols)


num_cols = []

for i in df_filtered_dropna.columns:
    if df_filtered_dropna[i].dtype.name == 'int64':
        num_cols.append(i)

len(num_cols)


float_cols = []

for i in df_filtered_dropna.columns:
    if df_filtered_dropna[i].dtype.name == 'float64':
        float_cols.append(i)

len(float_cols)

# List of Categorical columns
coltest1=['Gender_x',
 'CallResult',
 'OVERALL_HEALTH',
 'HEALTH_ASSESSMENT',
 'MENTAL_ASSESSMENT',
 'LIFE_QUALITY',
 'RATE_SLEEP',
 'INTERFERE_PAIN_MUCH',
 'PHYS_ACT',
 'HAD_FLU_SHOT',
 'FELT_DOWN',
 'LACK_INTEREST',
 'FELT_ANXIOUS',
 'FELT_WORRIED',
 'RECENT_FALLING',
 'PROBLEM_BALANCE',
 'BLADDER_CONTROL_HOS',
 'FOOD_MONEY',
 'NUMBER_MEALS',
 'HELP_APPOINTMENTS',
 'HELP_PERSON',
 'INTERNET_ACCESS',
 'AgeCat',
 'Internal_CM',
 'Critical_Care_flag',
 'Fall_Flag',
 'Hospice_Flag',
 'HomeCare_Flag',
 'Frailty_Flag',
 'PRARISK',
 'pcmh_flag',
 'PCP_Prim_Specialty',
 'attribution_method',
 'High_Cost_50K',
 'High_Cost_100K',
 'High_Cost_150K',
 'High_Cost_250K',
 'PART_D_LI_IND',
 'Medicare_Dual',
 'VO_Case',
 'PCP_Visit_Type',
 'PCP_AWV_Flag',
 'SleepApnea_Flag',
 'BH_Risk_Wgt_Range',
 'BH_Risk_Category',
 'COVID_Score', 
 'BMI_Adult',
 'BMI_Adult_Cat',
 'BMI_Obese_Flag',
 'SDoH_Vulnerable_Overall_Label']
len(coltest1)


# CHI SQAURE TEST

mylist=[]
mylist1=[]
for i in cat_cols:
    dataset_table=pd.crosstab(df_filtered_dropna.ISOLATED,df_filtered_dropna[i],margins='True')
    Observed_Values = dataset_table.values 
    val=stats.chi2_contingency(dataset_table)
    Expected_Values=val[3]
    no_of_rows=len(dataset_table.iloc[0:2,0])
    no_of_columns=len(dataset_table.iloc[0,0:2])
    ddof=(no_of_rows-1)*(no_of_columns-1)
    alpha = 0.05
    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    
    critical_value=chi2.ppf(q=1-alpha,df=ddof)
    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
    if chi_square_statistic>=critical_value:
        mylist.append(i)
       # print("Reject H0,There is a relationship between 2 categorical variables-{},{}".format(i,'ISOLATED'))
    else:
        mylist1.append(i)
        print("Retain H0,There is no relationship between 2 categorical variables-{},{}".format(i,'ISOLATED'))
mylist

# Categorical data 
final_cat_data=df_filtered_dropna[coltest1]
final_cat_data


#  One Hot Encoding---Data Transformation: Pre processing some features to prepare data for training
dump_onehot_encoding(df_filtered_dropna,coltest1)

DATA_MODEL = load_onehot_encoding(df_filtered_dropna,coltest1,'onehot_encoding.pickle')

#encoded_categorical_cols=onehot_encoding_categoricalcols(final_cat_data)
#encoded_categorical_cols
#numeric=df_filtered_dropna[num_cols].reset_index(drop=True)
#numeric.drop('File__Date',axis=1,inplace=True)
#DATA_MODEL=pd.concat([encoded_categorical_cols,numeric],axis=1)
#df_filtered_dropna[float_cols].reset_index(drop=True)
#DATA_MODEL=pd.concat([DATA_MODEL,numeric_float],axis=1)


# FINAL DATA FOR MODELLING
DATA_MODEL

# Final list of Features : Selected from Chi-Square Test, Shap values , Feature importance and major chronic conditions

final_features_selection=[ 'LACK_INTEREST_N',
'LACK_INTEREST_Y',
'Rx_allowed',
'RV_Score_Diff_Rx',
'Total_Allowed',
'RV_Potential_Score',
'RV_Current_Score_Rx',
'HELP_PERSON_N',
'HELP_PERSON_Y',
'LIFE_QUALITY_FAIR',
'LIFE_QUALITY_GOOD',
'LIFE_QUALITY_POOR',
'LIFE_QUALITY_VERY_GOOD',
'FELT_ANXIOUS_ALL',
'FELT_ANXIOUS_MOST',
'FELT_ANXIOUS_NONE',
'FELT_ANXIOUS_SOME',
'FELT_WORRIED_ALL',
'FELT_WORRIED_MOST',
'FELT_WORRIED_NONE',
'FELT_WORRIED_SOME',
'FELT_DOWN_N',
'FELT_DOWN_Y',
'RV_Payment_Diff',
'Prof_allowed',
'Age',
'Rx_TotScripts_Cnt',
'RV_Current_Score',
'OP_allowed',
'ISOLATED',
'Gender_x_Female',
'Gender_x_Male',
 'PCP_Visit_Counts',
 'Hypertension',
 'Hyperlipid',
 'LowBackPain',
 'Diabetes',
 'IschemicHD',
 'Asthma',
 'COPD',
 'CHF',
 'Cancer',
 'HIV_AIDS',
 'Depression',
 'PersonalityDisorder',
 'Bipolar',
 'Dementia',
 'Anxiety',
 'Med_Allowed',
 'SDoH_Vulnerable_Overall_Index',
 'Socioeconomic_Index',
 'Household_Composition_Disability',
 'VO_Case_N',
 'VO_Case_Y',
 'BMI_Adult_Cat_2) Healthy',
 'BMI_Adult_Cat_3) Overweight',
 'BMI_Adult_Cat_4) Obese Low Risk',
 'BMI_Adult_Cat_5) Obese Moderate Risk',
 'BMI_Adult_Cat_6) Obese High Risk',
 'BMI_Obese_Flag_N',
 'BMI_Obese_Flag_Y',
'AgeCat_00-17',
'AgeCat_18-39',
'AgeCat_40-64',
'AgeCat_65 and over'
                   
]


len(final_features_selection)

# Selecting final features for training
DATA_MODEL=DATA_MODEL[num_cols]


# Model Building-Decision tree, Randomforest, Xgboost

# TRAIN TEST SPLIT

X_expe = DATA_MODEL.drop('ISOLATED',axis=1)# Input features (attributes)
y_expe = DATA_MODEL['ISOLATED'].values # Target vector

X_train, X_test, y_train, y_test = train_test_split(X_expe, y_expe, train_size = 0.7, test_size=0.3, random_state=42)


# Decision Tree

dttest = DecisionTreeClassifier(criterion='gini', 
                                class_weight='balanced', 
                                max_depth=10, 
                                random_state=42)
dttest.fit(X_train, y_train)

y_train_pred = dttest.predict(X_train)
y_test_pred = dttest.predict(X_test)

print("Training Accuracy is: ", dttest.score(X_train, y_train))

# Accuracy on Train
print("Testing Accuracy is: ", dttest.score(X_test, y_test))


from sklearn.metrics import classification_report


print(classification_report(y_train, y_train_pred))

print(classification_report(y_test, y_test_pred))


#  Random Forest Model

#randomforest_model= build_randomforestmodel(X_train, X_test, y_train, y_test)

# Random forest Metrics:
def get_best_threshold_model(model,X_test,y_test):
    yhat = model.predict_proba(X_test)
    yhat = yhat[:, 1]
    precision, recall, thresholds = precision_recall_curve(y_test, yhat)
    thresholds = np.arange(0, 1, 0.001)
# evaluate each threshold
    scores = [f1_score(y_test, to_labels(yhat, t)) for t in thresholds]
# get best threshold
    ix = argmax(scores)
    print('Threshold=%.3f, F-Score=%.5f' % (thresholds[ix], scores[ix]))
    
#get_best_threshold_model(randomforest_model,X_test,y_test)

#(randomforest_model,X_train, X_test, y_train, y_test,.289)
  
# ### Feature Importance- RF

#from matplotlib.pyplot import figure
#figure(figsize=(8,8),dpi=80)

#figrf=pd.Series(randomforest_model.feature_importances_,index=X_train.columns).nlargest(15).sort_values(ascending=True).plot(kind='barh')

#figrf.update_layout(margin=dict(l=250))


# Xg-Boost Model

xgb1 = XGBClassifier(base_score=0.5, booster='gbtree', colsample_bylevel=0.8,
              colsample_bynode=1, colsample_bytree=0.8, gpu_id=0,
              importance_type='gain', interaction_constraints='',
              learning_rate=0.01, max_delta_step=0, max_depth=5,
              min_child_weight=1, 
              n_estimators=65, n_jobs=0, num_parallel_tree=1, random_state=0,
              reg_alpha=0.5, reg_lambda=0.9, scale_pos_weight=6, subsample=0.8,
              tree_method='exact', validate_parameters=1, verbosity=1)

xgb1.fit(X_train, y_train)

# Save model artifacts as pickle file
import joblib

joblib.dump(xgb1, Config.model_filename)

""" Load saved model artifacts for Inference run.
    Trained model artifacts will be used to predict Obesity condition for new members. """
# Load saved trained model object
#model_savedfile=joblib.load(Config.model_filename)



#with open(Config.model_filename,'wb') as file:
 #        pickle.dump(xgb1,file)
#saving the best model to the directory.
#file_op = file_methods.File_Operation(self.file_object,self.log_writer)
#save_model=file_op.save_model(xgb1,best_model_name+str(i))


get_best_threshold_model(xgb1,X_test,y_test)


model_scores(xgb1,X_train, X_test, y_train, y_test,0.510)

# ### Feature Importance-Xgboost

### from matplotlib.pyplot import figure
#from matplotlib.pyplot import figure
#figure(figsize=(12,12),dpi=80)

#fi = xgb1.get_booster().get_score(importance_type='weight')
#keys= list(fi.keys())
#values = list(fi.values())

#tmp_df = pd.DataFrame(data=values, index=keys, columns=['score']).sort_values(by='score', ascending=False)

#fig = tmp_df.iloc[:15,:].sort_values(by='score').plot(kind='barh')

#fig.update_layout(margin=dict(l=350))


# ### SHAP VALUES -RF

#get_shap_summaryplot_rf(randomforest_model,X_train)

get_shap_summaryplot_xgboost(xgb1,X_train)


# #### Top 10 Columns from Shapvalues- Random forest



#get_shapfeatures_rf(randomforest_model,X_train)


# #### Top 10 Columns from Shapvalues- Xgboost

get_shapfeatures_xgboost(xgb1,X_train)


# ### Final Chi-Square-Test

mylist=[]
mylist1=[]
for i in DATA_MODEL.columns:
    dataset_table=pd.crosstab(DATA_MODEL.ISOLATED,DATA_MODEL[i],margins='True')
    Observed_Values = dataset_table.values 
    val=stats.chi2_contingency(dataset_table)
    Expected_Values=val[3]
    no_of_rows=len(dataset_table.iloc[0:2,0])
    no_of_columns=len(dataset_table.iloc[0,0:2])
    ddof=(no_of_rows-1)*(no_of_columns-1)
    alpha = 0.05
    chi_square=sum([(o-e)**2./e for o,e in zip(Observed_Values,Expected_Values)])
    chi_square_statistic=chi_square[0]+chi_square[1]
    
    critical_value=chi2.ppf(q=1-alpha,df=ddof)
    p_value=1-chi2.cdf(x=chi_square_statistic,df=ddof)
    if chi_square_statistic>=critical_value:
        mylist.append(i)
       # print("Reject H0,There is a relationship between 2 categorical variables-{},{}".format(i,'ISOLATED'))
    else:
        mylist1.append(i)
        print("Retain H0,There is no relationship between 2 categorical variables-{},{}".format(i,'ISOLATED'))



mylist





