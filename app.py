import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score
from scipy.stats import chi2_contingency
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report,precision_recall_curve,precision_recall_fscore_support
import warnings
import os

a1= pd.read_excel("case_study1.xlsx")
a2= pd.read_excel("case_study2.xlsx")

d1 = a1.copy()
d2 =a2.copy()

#remove Null value (-9999)
d1 = d1.loc[d1['Age_Oldest_TL'] != -99999]

columns_to_be_removed =[]

for i in d2.columns:
    if d2.loc[d2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)

d2= d2.drop(columns_to_be_removed,axis=1)

for i in d2.columns:
    d2 = d2.loc[d2[i] != -99999]

#check if resolved
d2.isna().sum()
d1.isna().sum()

#checking common id for join
for i in list(d1.columns):
    if i in list(d2.columns):
        print(i)

df= pd.merge(d1,d2,how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])

#separating categorical features


for i in df.columns:
    if df[i].dtype == 'object':
        print(i)


for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    chi2,pval,_,_= chi2_contingency(pd.crosstab(df[i],df['Approved_Flag']))

    print(i,'---',pval)

    #all features have p-value , less than 0.5(5%): all are accepted


#separating numerical features

numerical_columns = []

for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numerical_columns.append(i)


numerical_columns


#VIF sequentially
vif_data = df[numerical_columns]
total_columns= vif_data.shape[1]
columns_to_be_kept = []
column_index =0

for i in range (0,total_columns):
    vif_value = variance_inflation_factor(vif_data, column_index)
    print(column_index, '---', vif_value)


    if vif_value <= 6:
        columns_to_be_kept.append(numerical_columns[i])
        column_index = column_index+1

    else:
        vif_data = vif_data.drop([numerical_columns[i]], axis =1)


#check for Anova for columns_to_be_kept

from scipy.stats import f_oneway

columns_to_be_kept_numerical =[]

for i in columns_to_be_kept:
    a = list(df[i])
    b= list(df['Approved_Flag'])

    group_p1 = [value for value, group in zip(a,b) if group == 'P1']
    group_p2 = [value for value, group in zip(a,b) if group == 'P2']
    group_p3 = [value for value, group in zip(a,b) if group == 'P3']
    group_p4 = [value for value, group in zip(a,b) if group == 'P4']

    f_statistic, p_value = f_oneway(group_p1,group_p2,group_p3,group_p4)

    if p_value <= 0.05:
        columns_to_be_kept_numerical.append(i)




#categorical labels
#['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']

# listing all the final features
features = columns_to_be_kept_numerical + ['MARITALSTATUS', 'EDUCATION', 'GENDER', 'last_prod_enq2', 'first_prod_enq2']
df = df[features + ['Approved_Flag']]

df['MARITALSTATUS'].unique()
df['EDUCATION'].unique()
df['GENDER'].unique()
df['last_prod_enq2'].unique()
df['first_prod_enq2'].unique()


# Ordinal feature -- EDUCATION
# SSC            : 1
# 12TH           : 2
# GRADUATE       : 3
# UNDER GRADUATE : 3
# POST-GRADUATE  : 4
# OTHERS         : 1
# PROFESSIONAL   : 3


# Others has to be verified by the business end user 




df.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3


df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()

df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])


df_encoded.info()
k= df_encoded.describe()


# 2. xgboost

import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

xgb_classifier = xgb.XGBClassifier(objective='multi:softmax',  num_class=4)



y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )


label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)


x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)




xgb_classifier.fit(x_train, y_train)
y_pred = xgb_classifier.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy:.2f}')
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()



# xgboost is giving me best results

# Hyperparameter tuning in xgboost
from sklearn.model_selection import GridSearchCV
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)

#initial hyperparameters
xgb_model = xgb.XGBClassifier(objective='multi:softmax', num_class=4)

#parameter grid

param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
}

grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='accuracy', n_jobs=-1)
grid_search.fit(x_train, y_train)


print("Best Hyperparameters:", grid_search.best_params_)

#evaluate the model with the best hyperparameters
best_model = grid_search.best_estimator_
accuracy = best_model.score(x_test, y_test)
print("Test Accuracy:", accuracy)




import xgboost as xgb
from sklearn.preprocessing import LabelEncoder

a3 = pd.read_excel("Unseen_Dataset.xlsx")

cols_in_df = list(df.columns)
cols_in_df.pop(42)

df_unseen = a3[cols_in_df]

df_unseen['MARITALSTATUS'].unique()
df_unseen['EDUCATION'].unique()
df_unseen['GENDER'].unique()
df_unseen['last_prod_enq2'].unique()
df_unseen['first_prod_enq2'].unique()


df_unseen.loc[df['EDUCATION'] == 'SSC',['EDUCATION']]              = 1
df_unseen.loc[df['EDUCATION'] == '12TH',['EDUCATION']]             = 2
df_unseen.loc[df['EDUCATION'] == 'GRADUATE',['EDUCATION']]         = 3
df_unseen.loc[df['EDUCATION'] == 'UNDER GRADUATE',['EDUCATION']]   = 3
df_unseen.loc[df['EDUCATION'] == 'POST-GRADUATE',['EDUCATION']]    = 4
df_unseen.loc[df['EDUCATION'] == 'OTHERS',['EDUCATION']]           = 1
df_unseen.loc[df['EDUCATION'] == 'PROFESSIONAL',['EDUCATION']]     = 3


df_unseen['EDUCATION'].value_counts()
df_unseen['EDUCATION'] = df['EDUCATION'].astype(int)
df_unseen.info()


df_encoded_unseen = pd.get_dummies(df_unseen, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])

df_encoded_unseen.info()
k= df_encoded.describe()


model = xgb.XGBClassifier(objective='multi:softmax',  num_class=4,colsample_bytree = 0.9,learning_rate = 1,max_depth = 3,alpha = 10,n_estimators = 100)


model.fit(x_train, y_train)

y_pred_unseen = model.predict(df_encoded_unseen)

a3['Target_variable'] = y_pred_unseen

a3.to_excel("Final_Predictions.xlsx",index = False)





