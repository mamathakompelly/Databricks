# -*- coding: utf-8 -*-
"""
Purpose: Best baseline Machine Learning model to be deployed
        to predict Isolation of members over a period.

Table of Contents: Perform the following operations over a ML modeling lifecycle -
    Step 1. Import required Python, 3rd-party & user-defined packages & modules
            - Import Python libraries
            - Load user defined myutils(function defs)
            - Load model artifacts & configurations
    Step 2. Perform Data Modeling
            - Extracting MERGEDDATA data from ELIZA SURVEY DATA AND MEMBER STRATIFICATION DATA
            - Data preparation 
            - Data Transformation: Pre processing some features to prepare data for training
    Step 3. Prepare entire dataset as Train set
            - Define base Training dataset & target labels
                Train set - to be used to train the model
                Target labels  - target output for training set (condition_flag ['isolated','Non isolated'] against a member data)
            - Encoding - Prepare dummy dataset from categorical features in Train set
              to make it suitable for training a Tree based classifier model
    Step 4. Reuse best  ML model hyperparameters evaluated during development
            to recreate the model artifacts using entire data as Train set
            - Train the model & get model artifacts
    Step 5. Save model artifacts - to be used to predict Social Isolation condition  for new members
    Step 6. Generate feature importances plot & explainer dashboard with shapely analysis
            - Generate feature importances plot for target - Isolation with shapely analysis
            - Build an interactive dashboard for explaining individual predictions
              and analyzing the evaluation metrics on trained ML model.
            - The dashboard can be viewed by opening the development server URL in the browser at
              the end of the script run if the flag is set
            - Flag for generating the dashboard to be set during Runtime
"""
# Import Python libraries
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification
import scipy.stats as stats
from sklearn.preprocessing import OneHotEncoder
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
import shap
from numpy import arange
from numpy import argmax
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import classification_report
import pickle
from config import Config
import joblib





# Null value handling with threshold of 80% null values in data
def get_cols_threshold_on_null(df, threshold=80):
    
    df = df.drop('FELT_ISOLATED', axis=1)
    tmp = (df.isna().sum()/df.shape[0])*100
    tmp = tmp[tmp>threshold]
    return tmp.index.values
    
# Deriving Target Variable
def derive_target_variable(df):
    TEMP_ISO=df[((df['FELT_ISOLATED']=='MOST')|(df['FELT_ISOLATED']=='ALL')) | ((df.FELT_ISOLATED == 'SOME') & (df.HELP_PERSON=='N')) ]
    TEMP_NONISO=df[((df['FELT_ISOLATED']=='A_LI')) | ((df.FELT_ISOLATED == 'SOME') & (df.HELP_PERSON=='Y')) ]
    TEMP_ISO['ISOLATED']=1
    TEMP_NONISO['ISOLATED']=0
    df1=pd.concat([TEMP_ISO,TEMP_NONISO])
    return df1
    
# Define encoder object
encoder = OneHotEncoder(handle_unknown='ignore')    
def onehot_encoding_categoricalcols(df):
    encoded_featurestest=encoder.fit_transform(df)
    encoded_columnstest=list(encoder.get_feature_names(df.columns))
    df_dummytest=pd.DataFrame(encoded_featurestest.toarray(),columns=encoded_columnstest)
    return df_dummytest 

    
# Define Random forest model to be used, tune different hyperparameters & train the model.
def build_randomforestmodel(X_train, X_test, y_train, y_test):
    # Number of trees in random forest
    n_estimators = [int(x) for x in np.linspace(start = 10, stop = 200, num = 10)]
    # Number of features to consider at every split
    max_features = ['auto', 'sqrt']
    # Maximum number of levels in tree
    max_depth = [2,4]
    # Minimum number of samples required to split a node
    min_samples_split = [2, 5]
    # Minimum number of samples required at each leaf node
    min_samples_leaf = [1, 2, 4]
    # Method of selecting samples for training each tree
    bootstrap = [True, False]

    # Create the param grid
    param_grid = {'n_estimators': n_estimators,
                   'max_features': max_features,
                   'max_depth': max_depth,
                   'min_samples_split': min_samples_split,
                   'min_samples_leaf': min_samples_leaf,
                   'bootstrap': bootstrap}
   

    rf_Model = RandomForestClassifier(random_state=0)


    from sklearn.model_selection import GridSearchCV
    rf_Grid = GridSearchCV(estimator = rf_Model, param_grid = param_grid, cv = 5, verbose=2, n_jobs = -1,scoring='f1')

    rf_Grid.fit(X_train, y_train)

    print('Rf_bestparameters',rf_Grid.best_params_)

   
    
    return rf_Grid.best_estimator_

    
def to_labels(pos_probs, threshold):
    return (pos_probs >= threshold).astype('int')
    
    

# Define model metrics  
def model_scores(model,X_train, X_test, y_train, y_test,threshold):
     
    predicted_proba = model.predict_proba(X_train)
    predicted_ytrain = (predicted_proba [:,1] >= threshold).astype('int')

    predicted_proba = model.predict_proba(X_test)
    predicted_ytest = (predicted_proba [:,1] >= threshold).astype('int')
    
    print (f'Train Accuracy - : {model.score(X_train,y_train):.3f}')
    print (f'Test Accuracy - : {model.score(X_test,y_test):.3f}')

    print("\n", "#"*20, "CLASSICATION REPORT FOR TRAIN DATA", "#"*20, "\n")

    print(classification_report(y_train, predicted_ytrain))
    
    print("\n", "#"*20, "CLASSICATION REPORT FOR TEST DATA", "#"*20, "\n")

    print(classification_report(y_test, predicted_ytest))
    
    print("\n", "#"*20, "CONFUSION MATRIX FOR TRAIN DATA", "#"*20, "\n")
    print(confusion_matrix(y_train, predicted_ytrain))
    print("\n", "#"*20, "CONFUSION MATRIX FOR TEST DATA", "#"*20, "\n")
    print(confusion_matrix(y_test, predicted_ytest))
    
    
# Define shap summary plots for random forest model  
def get_shap_summaryplot_rf(model,X_train):
    shap_values_rf = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values_rf[1], X_train,plot_type='dot')
    shap.summary_plot(shap_values_rf[1], X_train, plot_type="bar")
    
    
 # Define shap summary plots xgboost model     
def get_shap_summaryplot_xgboost(model,X_train):
    shap_values_xgb1 = shap.TreeExplainer(model).shap_values(X_train)
    shap.summary_plot(shap_values_xgb1, X_train)
    shap.summary_plot(shap_values_xgb1, X_train, plot_type="bar")
    
    
 # Define feature importance rf model     
def get_shapfeatures_rf(model,X_train):
    shap_values_rf = shap.TreeExplainer(model).shap_values(X_train)
    vals= np.abs(shap_values_rf).mean(0)
    features=X_train
    feature_importance = pd.DataFrame(list(zip(features.columns, sum(vals))), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False,inplace=True)
    return feature_importance.head(10)
    
    
 # Define feature importance xgboost model    
def get_shapfeatures_xgboost(model,X_train):
    explainer = shap.Explainer(model)
    shap_values_xgb1 = explainer(X_train)

    feature_names = list(X_train.columns.values)
    vals = np.abs(shap_values_xgb1.values).mean(0)
    feature_importance = pd.DataFrame(list(zip(feature_names, vals)), columns=['col_name','feature_importance_vals'])
    feature_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    return feature_importance.head(10)


# One hot encoding
""" Prepare dummy dataset from categorical features for a Tree based classifier model to predict target labels
            encoder - saved encoder object fit on Training dataset
            Returns:
                    - dataset with features encoded into new categorical columns & Binary values
    """
def dump_onehot_encoding(df, ohe_cols):

    ohe = OneHotEncoder(sparse=False)

    ohe.fit(np.array(df[ohe_cols]))
    
    
    joblib.dump(ohe, Config.encoder_filename)
       
    return None


# Method to load one hot encodings
def load_onehot_encoding(df, ohe_cols, filename):
           
    ohe = joblib.load(filename)
    # Encode categorical features
    encoded_cols = list(ohe.get_feature_names())

    col_map = dict()

    for i,j in list(enumerate(ohe_cols)):  
        
        col_map['x' + str(i)] = j
    
    new_cols = [col_map[i[0:2]] + i[2:] for i in encoded_cols]
    # Convert sparse array with encoded features into a dataframe
    df_encoded = pd.DataFrame(ohe.transform(df[ohe_cols]), columns= new_cols)
    
    tmp = df.drop(columns=ohe_cols, axis=1)
    
    tmp = tmp.reset_index(drop=True)
    
    combined_df =  pd.concat([tmp, df_encoded],axis =1)
    
    return combined_df

    
    
       

        
 
      