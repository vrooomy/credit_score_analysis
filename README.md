```python
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




```


```python
!pip install statsmodels

```

    Requirement already satisfied: statsmodels in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (0.14.4)
    Requirement already satisfied: numpy<3,>=1.22.3 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from statsmodels) (1.26.4)
    Requirement already satisfied: scipy!=1.9.2,>=1.8 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from statsmodels) (1.14.0)
    Requirement already satisfied: pandas!=2.1.0,>=1.4 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from statsmodels) (2.1.3)
    Requirement already satisfied: patsy>=0.5.6 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from statsmodels) (1.0.1)
    Requirement already satisfied: packaging>=21.3 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from statsmodels) (24.0)
    Requirement already satisfied: python-dateutil>=2.8.2 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2.8.2)
    Requirement already satisfied: pytz>=2020.1 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2023.3.post1)
    Requirement already satisfied: tzdata>=2022.1 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from pandas!=2.1.0,>=1.4->statsmodels) (2023.3)
    Requirement already satisfied: six>=1.5 in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from python-dateutil>=2.8.2->pandas!=2.1.0,>=1.4->statsmodels) (1.16.0)
    


```python
!pip install openpyxl
```

    Requirement already satisfied: openpyxl in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (3.1.5)
    Requirement already satisfied: et-xmlfile in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from openpyxl) (2.0.0)
    


```python
a1= pd.read_excel("case_study1.xlsx")
```


```python
a2= pd.read_excel("case_study2.xlsx")
```


```python
d1 = a1.copy()
d2 =a2.copy()
```


```python
d1.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>PROSPECTID</th>
      <th>Total_TL</th>
      <th>Tot_Closed_TL</th>
      <th>Tot_Active_TL</th>
      <th>Total_TL_opened_L6M</th>
      <th>Tot_TL_closed_L6M</th>
      <th>pct_tl_open_L6M</th>
      <th>pct_tl_closed_L6M</th>
      <th>pct_active_tl</th>
      <th>pct_closed_tl</th>
      <th>...</th>
      <th>CC_TL</th>
      <th>Consumer_TL</th>
      <th>Gold_TL</th>
      <th>Home_TL</th>
      <th>PL_TL</th>
      <th>Secured_TL</th>
      <th>Unsecured_TL</th>
      <th>Other_TL</th>
      <th>Age_Oldest_TL</th>
      <th>Age_Newest_TL</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>5</td>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.200</td>
      <td>0.800</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>4</td>
      <td>1</td>
      <td>4</td>
      <td>0</td>
      <td>72</td>
      <td>18</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>0</td>
      <td>7</td>
      <td>7</td>
    </tr>
    <tr>
      <th>2</th>
      <td>3</td>
      <td>8</td>
      <td>0</td>
      <td>8</td>
      <td>1</td>
      <td>0</td>
      <td>0.125</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0</td>
      <td>6</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>2</td>
      <td>6</td>
      <td>0</td>
      <td>47</td>
      <td>2</td>
    </tr>
    <tr>
      <th>3</th>
      <td>4</td>
      <td>1</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>0</td>
      <td>1.000</td>
      <td>0.0</td>
      <td>1.000</td>
      <td>0.000</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>1</td>
      <td>1</td>
      <td>5</td>
      <td>5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>3</td>
      <td>2</td>
      <td>1</td>
      <td>0</td>
      <td>0</td>
      <td>0.000</td>
      <td>0.0</td>
      <td>0.333</td>
      <td>0.667</td>
      <td>...</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>0</td>
      <td>3</td>
      <td>0</td>
      <td>2</td>
      <td>131</td>
      <td>32</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
d2.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 51336 entries, 0 to 51335
    Data columns (total 62 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   PROSPECTID                    51336 non-null  int64  
     1   time_since_recent_payment     51336 non-null  int64  
     2   time_since_first_deliquency   51336 non-null  int64  
     3   time_since_recent_deliquency  51336 non-null  int64  
     4   num_times_delinquent          51336 non-null  int64  
     5   max_delinquency_level         51336 non-null  int64  
     6   max_recent_level_of_deliq     51336 non-null  int64  
     7   num_deliq_6mts                51336 non-null  int64  
     8   num_deliq_12mts               51336 non-null  int64  
     9   num_deliq_6_12mts             51336 non-null  int64  
     10  max_deliq_6mts                51336 non-null  int64  
     11  max_deliq_12mts               51336 non-null  int64  
     12  num_times_30p_dpd             51336 non-null  int64  
     13  num_times_60p_dpd             51336 non-null  int64  
     14  num_std                       51336 non-null  int64  
     15  num_std_6mts                  51336 non-null  int64  
     16  num_std_12mts                 51336 non-null  int64  
     17  num_sub                       51336 non-null  int64  
     18  num_sub_6mts                  51336 non-null  int64  
     19  num_sub_12mts                 51336 non-null  int64  
     20  num_dbt                       51336 non-null  int64  
     21  num_dbt_6mts                  51336 non-null  int64  
     22  num_dbt_12mts                 51336 non-null  int64  
     23  num_lss                       51336 non-null  int64  
     24  num_lss_6mts                  51336 non-null  int64  
     25  num_lss_12mts                 51336 non-null  int64  
     26  recent_level_of_deliq         51336 non-null  int64  
     27  tot_enq                       51336 non-null  int64  
     28  CC_enq                        51336 non-null  int64  
     29  CC_enq_L6m                    51336 non-null  int64  
     30  CC_enq_L12m                   51336 non-null  int64  
     31  PL_enq                        51336 non-null  int64  
     32  PL_enq_L6m                    51336 non-null  int64  
     33  PL_enq_L12m                   51336 non-null  int64  
     34  time_since_recent_enq         51336 non-null  int64  
     35  enq_L12m                      51336 non-null  int64  
     36  enq_L6m                       51336 non-null  int64  
     37  enq_L3m                       51336 non-null  int64  
     38  MARITALSTATUS                 51336 non-null  object 
     39  EDUCATION                     51336 non-null  object 
     40  AGE                           51336 non-null  int64  
     41  GENDER                        51336 non-null  object 
     42  NETMONTHLYINCOME              51336 non-null  int64  
     43  Time_With_Curr_Empr           51336 non-null  int64  
     44  pct_of_active_TLs_ever        51336 non-null  float64
     45  pct_opened_TLs_L6m_of_L12m    51336 non-null  float64
     46  pct_currentBal_all_TL         51336 non-null  float64
     47  CC_utilization                51336 non-null  float64
     48  CC_Flag                       51336 non-null  int64  
     49  PL_utilization                51336 non-null  float64
     50  PL_Flag                       51336 non-null  int64  
     51  pct_PL_enq_L6m_of_L12m        51336 non-null  float64
     52  pct_CC_enq_L6m_of_L12m        51336 non-null  float64
     53  pct_PL_enq_L6m_of_ever        51336 non-null  float64
     54  pct_CC_enq_L6m_of_ever        51336 non-null  float64
     55  max_unsec_exposure_inPct      51336 non-null  float64
     56  HL_Flag                       51336 non-null  int64  
     57  GL_Flag                       51336 non-null  int64  
     58  last_prod_enq2                51336 non-null  object 
     59  first_prod_enq2               51336 non-null  object 
     60  Credit_Score                  51336 non-null  int64  
     61  Approved_Flag                 51336 non-null  object 
    dtypes: float64(10), int64(46), object(6)
    memory usage: 24.3+ MB
    


```python
#remove Null value (-9999)
d1 = d1.loc[d1['Age_Oldest_TL'] != -99999]

```


```python
columns_to_be_removed =[]

for i in d2.columns:
    if d2.loc[d2[i] == -99999].shape[0] > 10000:
        columns_to_be_removed.append(i)
```


```python

d2= d2.drop(columns_to_be_removed,axis=1)
```


```python
for i in d2.columns:
    d2 = d2.loc[d2[i] != -99999]
```


```python
#check if resolved
d2.isna().sum()
d1.isna().sum()
```




    PROSPECTID              0
    Total_TL                0
    Tot_Closed_TL           0
    Tot_Active_TL           0
    Total_TL_opened_L6M     0
    Tot_TL_closed_L6M       0
    pct_tl_open_L6M         0
    pct_tl_closed_L6M       0
    pct_active_tl           0
    pct_closed_tl           0
    Total_TL_opened_L12M    0
    Tot_TL_closed_L12M      0
    pct_tl_open_L12M        0
    pct_tl_closed_L12M      0
    Tot_Missed_Pmnt         0
    Auto_TL                 0
    CC_TL                   0
    Consumer_TL             0
    Gold_TL                 0
    Home_TL                 0
    PL_TL                   0
    Secured_TL              0
    Unsecured_TL            0
    Other_TL                0
    Age_Oldest_TL           0
    Age_Newest_TL           0
    dtype: int64




```python
#checking common id for join
for i in list(d1.columns):
    if i in list(d2.columns):
        print(i)
```

    PROSPECTID
    


```python
df= pd.merge(d1,d2,how='inner', left_on=['PROSPECTID'], right_on=['PROSPECTID'])
```


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42064 entries, 0 to 42063
    Data columns (total 79 columns):
     #   Column                      Non-Null Count  Dtype  
    ---  ------                      --------------  -----  
     0   PROSPECTID                  42064 non-null  int64  
     1   Total_TL                    42064 non-null  int64  
     2   Tot_Closed_TL               42064 non-null  int64  
     3   Tot_Active_TL               42064 non-null  int64  
     4   Total_TL_opened_L6M         42064 non-null  int64  
     5   Tot_TL_closed_L6M           42064 non-null  int64  
     6   pct_tl_open_L6M             42064 non-null  float64
     7   pct_tl_closed_L6M           42064 non-null  float64
     8   pct_active_tl               42064 non-null  float64
     9   pct_closed_tl               42064 non-null  float64
     10  Total_TL_opened_L12M        42064 non-null  int64  
     11  Tot_TL_closed_L12M          42064 non-null  int64  
     12  pct_tl_open_L12M            42064 non-null  float64
     13  pct_tl_closed_L12M          42064 non-null  float64
     14  Tot_Missed_Pmnt             42064 non-null  int64  
     15  Auto_TL                     42064 non-null  int64  
     16  CC_TL                       42064 non-null  int64  
     17  Consumer_TL                 42064 non-null  int64  
     18  Gold_TL                     42064 non-null  int64  
     19  Home_TL                     42064 non-null  int64  
     20  PL_TL                       42064 non-null  int64  
     21  Secured_TL                  42064 non-null  int64  
     22  Unsecured_TL                42064 non-null  int64  
     23  Other_TL                    42064 non-null  int64  
     24  Age_Oldest_TL               42064 non-null  int64  
     25  Age_Newest_TL               42064 non-null  int64  
     26  time_since_recent_payment   42064 non-null  int64  
     27  num_times_delinquent        42064 non-null  int64  
     28  max_recent_level_of_deliq   42064 non-null  int64  
     29  num_deliq_6mts              42064 non-null  int64  
     30  num_deliq_12mts             42064 non-null  int64  
     31  num_deliq_6_12mts           42064 non-null  int64  
     32  num_times_30p_dpd           42064 non-null  int64  
     33  num_times_60p_dpd           42064 non-null  int64  
     34  num_std                     42064 non-null  int64  
     35  num_std_6mts                42064 non-null  int64  
     36  num_std_12mts               42064 non-null  int64  
     37  num_sub                     42064 non-null  int64  
     38  num_sub_6mts                42064 non-null  int64  
     39  num_sub_12mts               42064 non-null  int64  
     40  num_dbt                     42064 non-null  int64  
     41  num_dbt_6mts                42064 non-null  int64  
     42  num_dbt_12mts               42064 non-null  int64  
     43  num_lss                     42064 non-null  int64  
     44  num_lss_6mts                42064 non-null  int64  
     45  num_lss_12mts               42064 non-null  int64  
     46  recent_level_of_deliq       42064 non-null  int64  
     47  tot_enq                     42064 non-null  int64  
     48  CC_enq                      42064 non-null  int64  
     49  CC_enq_L6m                  42064 non-null  int64  
     50  CC_enq_L12m                 42064 non-null  int64  
     51  PL_enq                      42064 non-null  int64  
     52  PL_enq_L6m                  42064 non-null  int64  
     53  PL_enq_L12m                 42064 non-null  int64  
     54  time_since_recent_enq       42064 non-null  int64  
     55  enq_L12m                    42064 non-null  int64  
     56  enq_L6m                     42064 non-null  int64  
     57  enq_L3m                     42064 non-null  int64  
     58  MARITALSTATUS               42064 non-null  object 
     59  EDUCATION                   42064 non-null  object 
     60  AGE                         42064 non-null  int64  
     61  GENDER                      42064 non-null  object 
     62  NETMONTHLYINCOME            42064 non-null  int64  
     63  Time_With_Curr_Empr         42064 non-null  int64  
     64  pct_of_active_TLs_ever      42064 non-null  float64
     65  pct_opened_TLs_L6m_of_L12m  42064 non-null  float64
     66  pct_currentBal_all_TL       42064 non-null  float64
     67  CC_Flag                     42064 non-null  int64  
     68  PL_Flag                     42064 non-null  int64  
     69  pct_PL_enq_L6m_of_L12m      42064 non-null  float64
     70  pct_CC_enq_L6m_of_L12m      42064 non-null  float64
     71  pct_PL_enq_L6m_of_ever      42064 non-null  float64
     72  pct_CC_enq_L6m_of_ever      42064 non-null  float64
     73  HL_Flag                     42064 non-null  int64  
     74  GL_Flag                     42064 non-null  int64  
     75  last_prod_enq2              42064 non-null  object 
     76  first_prod_enq2             42064 non-null  object 
     77  Credit_Score                42064 non-null  int64  
     78  Approved_Flag               42064 non-null  object 
    dtypes: float64(13), int64(60), object(6)
    memory usage: 25.4+ MB
    


```python
#separating categorical features


for i in df.columns:
    if df[i].dtype == 'object':
        print(i)
```

    MARITALSTATUS
    EDUCATION
    GENDER
    last_prod_enq2
    first_prod_enq2
    Approved_Flag
    


```python
for i in ['MARITALSTATUS','EDUCATION','GENDER','last_prod_enq2','first_prod_enq2']:
    chi2,pval,_,_= chi2_contingency(pd.crosstab(df[i],df['Approved_Flag']))

    print(i,'---',pval)

    #all features have p-value , less than 0.5(5%): all are accepted

```

    MARITALSTATUS --- 3.578180861038862e-233
    EDUCATION --- 2.6942265249737532e-30
    GENDER --- 1.907936100186563e-05
    last_prod_enq2 --- 0.0
    first_prod_enq2 --- 7.84997610555419e-287
    


```python
#separating numerical features

numerical_columns = []

for i in df.columns:
    if df[i].dtype != 'object' and i not in ['PROSPECTID','Approved_Flag']:
        numerical_columns.append(i)


numerical_columns
```




    ['Total_TL',
     'Tot_Closed_TL',
     'Tot_Active_TL',
     'Total_TL_opened_L6M',
     'Tot_TL_closed_L6M',
     'pct_tl_open_L6M',
     'pct_tl_closed_L6M',
     'pct_active_tl',
     'pct_closed_tl',
     'Total_TL_opened_L12M',
     'Tot_TL_closed_L12M',
     'pct_tl_open_L12M',
     'pct_tl_closed_L12M',
     'Tot_Missed_Pmnt',
     'Auto_TL',
     'CC_TL',
     'Consumer_TL',
     'Gold_TL',
     'Home_TL',
     'PL_TL',
     'Secured_TL',
     'Unsecured_TL',
     'Other_TL',
     'Age_Oldest_TL',
     'Age_Newest_TL',
     'time_since_recent_payment',
     'num_times_delinquent',
     'max_recent_level_of_deliq',
     'num_deliq_6mts',
     'num_deliq_12mts',
     'num_deliq_6_12mts',
     'num_times_30p_dpd',
     'num_times_60p_dpd',
     'num_std',
     'num_std_6mts',
     'num_std_12mts',
     'num_sub',
     'num_sub_6mts',
     'num_sub_12mts',
     'num_dbt',
     'num_dbt_6mts',
     'num_dbt_12mts',
     'num_lss',
     'num_lss_6mts',
     'num_lss_12mts',
     'recent_level_of_deliq',
     'tot_enq',
     'CC_enq',
     'CC_enq_L6m',
     'CC_enq_L12m',
     'PL_enq',
     'PL_enq_L6m',
     'PL_enq_L12m',
     'time_since_recent_enq',
     'enq_L12m',
     'enq_L6m',
     'enq_L3m',
     'AGE',
     'NETMONTHLYINCOME',
     'Time_With_Curr_Empr',
     'pct_of_active_TLs_ever',
     'pct_opened_TLs_L6m_of_L12m',
     'pct_currentBal_all_TL',
     'CC_Flag',
     'PL_Flag',
     'pct_PL_enq_L6m_of_L12m',
     'pct_CC_enq_L6m_of_L12m',
     'pct_PL_enq_L6m_of_ever',
     'pct_CC_enq_L6m_of_ever',
     'HL_Flag',
     'GL_Flag',
     'Credit_Score']




```python
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

```

    c:\Users\vjbro\AppData\Local\Programs\Python\Python310\lib\site-packages\statsmodels\stats\outliers_influence.py:197: RuntimeWarning: divide by zero encountered in scalar divide
      vif = 1. / (1. - r_squared_i)
    

    0 --- inf
    

    c:\Users\vjbro\AppData\Local\Programs\Python\Python310\lib\site-packages\statsmodels\stats\outliers_influence.py:197: RuntimeWarning: divide by zero encountered in scalar divide
      vif = 1. / (1. - r_squared_i)
    

    0 --- inf
    0 --- 11.320180023967996
    0 --- 8.363698035000327
    0 --- 6.520647877790928
    0 --- 5.149501618212625
    1 --- 2.611111040579735
    

    c:\Users\vjbro\AppData\Local\Programs\Python\Python310\lib\site-packages\statsmodels\stats\outliers_influence.py:197: RuntimeWarning: divide by zero encountered in scalar divide
      vif = 1. / (1. - r_squared_i)
    

    2 --- inf
    2 --- 1788.7926256209232
    2 --- 8.601028256477228
    2 --- 3.8328007921530785
    3 --- 6.099653381646739
    3 --- 5.5813520096427585
    4 --- 1.985584353098778
    

    c:\Users\vjbro\AppData\Local\Programs\Python\Python310\lib\site-packages\statsmodels\stats\outliers_influence.py:197: RuntimeWarning: divide by zero encountered in scalar divide
      vif = 1. / (1. - r_squared_i)
    

    5 --- inf
    5 --- 4.809538302819343
    6 --- 23.270628983464636
    6 --- 30.595522588100053
    6 --- 4.3843464059655854
    7 --- 3.0646584155234238
    8 --- 2.898639771299252
    9 --- 4.377876915347324
    10 --- 2.2078535836958433
    11 --- 4.916914200506864
    12 --- 5.214702030064725
    13 --- 3.3861625024231476
    14 --- 7.840583309478997
    14 --- 5.255034641721438
    

    c:\Users\vjbro\AppData\Local\Programs\Python\Python310\lib\site-packages\statsmodels\stats\outliers_influence.py:197: RuntimeWarning: divide by zero encountered in scalar divide
      vif = 1. / (1. - r_squared_i)
    

    15 --- inf
    15 --- 7.380634506427232
    15 --- 1.421005001517573
    16 --- 8.083255010190323
    16 --- 1.624122752404011
    17 --- 7.257811920140003
    17 --- 15.59624383268298
    17 --- 1.825857047132431
    18 --- 1.5080839450032664
    19 --- 2.172088834824577
    20 --- 2.623397553527229
    21 --- 2.2959970812106176
    22 --- 7.360578319196439
    22 --- 2.1602387773102554
    23 --- 2.8686288267891467
    24 --- 6.458218003637272
    24 --- 2.8474118865638265
    25 --- 4.7531981562840855
    26 --- 16.22735475594825
    26 --- 6.424377256363877
    26 --- 8.887080381808687
    26 --- 2.3804746142952653
    27 --- 8.60951347651454
    27 --- 13.06755093547673
    27 --- 3.5000400566546555
    28 --- 1.9087955874813773
    29 --- 17.006562234161628
    29 --- 10.730485153719197
    29 --- 2.3538497522950275
    30 --- 22.104855915136433
    30 --- 2.7971639638512906
    31 --- 3.424171203217696
    32 --- 10.175021454450935
    32 --- 6.408710354561301
    32 --- 1.0011511962625619
    33 --- 3.069197305397274
    34 --- 2.8091261600643724
    35 --- 20.249538381980678
    35 --- 15.864576541593774
    35 --- 1.8331649740532163
    36 --- 1.5680839909542037
    37 --- 1.9307572353811677
    38 --- 4.331265056645247
    39 --- 9.390334396150173
    


```python
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





```


```python
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

```




    array(['PL', 'ConsumerLoan', 'others', 'AL', 'HL', 'CC'], dtype=object)




```python
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

```


```python
df['EDUCATION'].value_counts()
df['EDUCATION'] = df['EDUCATION'].astype(int)
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42064 entries, 0 to 42063
    Data columns (total 43 columns):
     #   Column                     Non-Null Count  Dtype  
    ---  ------                     --------------  -----  
     0   pct_tl_open_L6M            42064 non-null  float64
     1   pct_tl_closed_L6M          42064 non-null  float64
     2   Tot_TL_closed_L12M         42064 non-null  int64  
     3   pct_tl_closed_L12M         42064 non-null  float64
     4   Tot_Missed_Pmnt            42064 non-null  int64  
     5   CC_TL                      42064 non-null  int64  
     6   Home_TL                    42064 non-null  int64  
     7   PL_TL                      42064 non-null  int64  
     8   Secured_TL                 42064 non-null  int64  
     9   Unsecured_TL               42064 non-null  int64  
     10  Other_TL                   42064 non-null  int64  
     11  Age_Oldest_TL              42064 non-null  int64  
     12  Age_Newest_TL              42064 non-null  int64  
     13  time_since_recent_payment  42064 non-null  int64  
     14  max_recent_level_of_deliq  42064 non-null  int64  
     15  num_deliq_6_12mts          42064 non-null  int64  
     16  num_times_60p_dpd          42064 non-null  int64  
     17  num_std_12mts              42064 non-null  int64  
     18  num_sub                    42064 non-null  int64  
     19  num_sub_6mts               42064 non-null  int64  
     20  num_sub_12mts              42064 non-null  int64  
     21  num_dbt                    42064 non-null  int64  
     22  num_dbt_12mts              42064 non-null  int64  
     23  num_lss                    42064 non-null  int64  
     24  recent_level_of_deliq      42064 non-null  int64  
     25  CC_enq_L12m                42064 non-null  int64  
     26  PL_enq_L12m                42064 non-null  int64  
     27  time_since_recent_enq      42064 non-null  int64  
     28  enq_L3m                    42064 non-null  int64  
     29  NETMONTHLYINCOME           42064 non-null  int64  
     30  Time_With_Curr_Empr        42064 non-null  int64  
     31  CC_Flag                    42064 non-null  int64  
     32  PL_Flag                    42064 non-null  int64  
     33  pct_PL_enq_L6m_of_ever     42064 non-null  float64
     34  pct_CC_enq_L6m_of_ever     42064 non-null  float64
     35  HL_Flag                    42064 non-null  int64  
     36  GL_Flag                    42064 non-null  int64  
     37  MARITALSTATUS              42064 non-null  object 
     38  EDUCATION                  42064 non-null  int32  
     39  GENDER                     42064 non-null  object 
     40  last_prod_enq2             42064 non-null  object 
     41  first_prod_enq2            42064 non-null  object 
     42  Approved_Flag              42064 non-null  object 
    dtypes: float64(5), int32(1), int64(32), object(5)
    memory usage: 13.6+ MB
    


```python
df_encoded = pd.get_dummies(df, columns=['MARITALSTATUS','GENDER','last_prod_enq2','first_prod_enq2'])
```


```python
df_encoded.info()
k= df_encoded.describe()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 42064 entries, 0 to 42063
    Data columns (total 55 columns):
     #   Column                        Non-Null Count  Dtype  
    ---  ------                        --------------  -----  
     0   pct_tl_open_L6M               42064 non-null  float64
     1   pct_tl_closed_L6M             42064 non-null  float64
     2   Tot_TL_closed_L12M            42064 non-null  int64  
     3   pct_tl_closed_L12M            42064 non-null  float64
     4   Tot_Missed_Pmnt               42064 non-null  int64  
     5   CC_TL                         42064 non-null  int64  
     6   Home_TL                       42064 non-null  int64  
     7   PL_TL                         42064 non-null  int64  
     8   Secured_TL                    42064 non-null  int64  
     9   Unsecured_TL                  42064 non-null  int64  
     10  Other_TL                      42064 non-null  int64  
     11  Age_Oldest_TL                 42064 non-null  int64  
     12  Age_Newest_TL                 42064 non-null  int64  
     13  time_since_recent_payment     42064 non-null  int64  
     14  max_recent_level_of_deliq     42064 non-null  int64  
     15  num_deliq_6_12mts             42064 non-null  int64  
     16  num_times_60p_dpd             42064 non-null  int64  
     17  num_std_12mts                 42064 non-null  int64  
     18  num_sub                       42064 non-null  int64  
     19  num_sub_6mts                  42064 non-null  int64  
     20  num_sub_12mts                 42064 non-null  int64  
     21  num_dbt                       42064 non-null  int64  
     22  num_dbt_12mts                 42064 non-null  int64  
     23  num_lss                       42064 non-null  int64  
     24  recent_level_of_deliq         42064 non-null  int64  
     25  CC_enq_L12m                   42064 non-null  int64  
     26  PL_enq_L12m                   42064 non-null  int64  
     27  time_since_recent_enq         42064 non-null  int64  
     28  enq_L3m                       42064 non-null  int64  
     29  NETMONTHLYINCOME              42064 non-null  int64  
     30  Time_With_Curr_Empr           42064 non-null  int64  
     31  CC_Flag                       42064 non-null  int64  
     32  PL_Flag                       42064 non-null  int64  
     33  pct_PL_enq_L6m_of_ever        42064 non-null  float64
     34  pct_CC_enq_L6m_of_ever        42064 non-null  float64
     35  HL_Flag                       42064 non-null  int64  
     36  GL_Flag                       42064 non-null  int64  
     37  EDUCATION                     42064 non-null  int32  
     38  Approved_Flag                 42064 non-null  object 
     39  MARITALSTATUS_Married         42064 non-null  bool   
     40  MARITALSTATUS_Single          42064 non-null  bool   
     41  GENDER_F                      42064 non-null  bool   
     42  GENDER_M                      42064 non-null  bool   
     43  last_prod_enq2_AL             42064 non-null  bool   
     44  last_prod_enq2_CC             42064 non-null  bool   
     45  last_prod_enq2_ConsumerLoan   42064 non-null  bool   
     46  last_prod_enq2_HL             42064 non-null  bool   
     47  last_prod_enq2_PL             42064 non-null  bool   
     48  last_prod_enq2_others         42064 non-null  bool   
     49  first_prod_enq2_AL            42064 non-null  bool   
     50  first_prod_enq2_CC            42064 non-null  bool   
     51  first_prod_enq2_ConsumerLoan  42064 non-null  bool   
     52  first_prod_enq2_HL            42064 non-null  bool   
     53  first_prod_enq2_PL            42064 non-null  bool   
     54  first_prod_enq2_others        42064 non-null  bool   
    dtypes: bool(16), float64(5), int32(1), int64(32), object(1)
    memory usage: 13.0+ MB
    


```python
#Machine Learning model fitting


#Random Forest

y= df_encoded['Approved_Flag']
x = df_encoded.drop(['Approved_Flag'], axis =1)

x_train , x_test ,y_train ,y_test = train_test_split(x,y, test_size=0.2 , random_state=42)

rf_classifier = RandomForestClassifier(n_estimators = 200, random_state=42)

rf_classifier.fit(x_train , y_train)

y_pred = rf_classifier.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)
print ()
print(f'Accuracy: {accuracy}')
print ()
precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()





```

    
    Accuracy: 0.7636990372043266
    
    Class p1:
    Precision: 0.8370457209847597
    Recall: 0.7041420118343196
    F1 Score: 0.7648634172469202
    
    Class p2:
    Precision: 0.7957519116397621
    Recall: 0.9282457879088206
    F1 Score: 0.856907593778591
    
    Class p3:
    Precision: 0.4423380726698262
    Recall: 0.21132075471698114
    F1 Score: 0.28600612870275793
    
    Class p4:
    Precision: 0.7178502879078695
    Recall: 0.7269193391642371
    F1 Score: 0.7223563495895703
    
    


```python
!pip install xgboost
```

    Requirement already satisfied: xgboost in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (2.1.4)
    Requirement already satisfied: numpy in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from xgboost) (1.26.4)
    Requirement already satisfied: scipy in c:\users\vjbro\appdata\local\programs\python\python310\lib\site-packages (from xgboost) (1.14.0)
    


```python
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
```

    
    Accuracy: 0.78
    
    Class p1:
    Precision: 0.823906083244397
    Recall: 0.7613412228796844
    F1 Score: 0.7913890312660175
    
    Class p2:
    Precision: 0.8255418233924413
    Recall: 0.913577799801784
    F1 Score: 0.8673315769665035
    
    Class p3:
    Precision: 0.4756380510440835
    Recall: 0.30943396226415093
    F1 Score: 0.37494284407864653
    
    Class p4:
    Precision: 0.7342386032977691
    Recall: 0.7356656948493683
    F1 Score: 0.7349514563106796
    
    


```python
# 3. Decision Tree
from sklearn.tree import DecisionTreeClassifier


y = df_encoded['Approved_Flag']
x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


dt_model = DecisionTreeClassifier(max_depth=20, min_samples_split=10)
dt_model.fit(x_train, y_train)
y_pred = dt_model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print ()
print(f"Accuracy: {accuracy:.2f}")
print ()

precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
```

    
    Accuracy: 0.71
    
    Class p1:
    Precision: 0.7207472959685349
    Recall: 0.722879684418146
    F1 Score: 0.7218119153126539
    
    Class p2:
    Precision: 0.809727626459144
    Recall: 0.8249752229930625
    F1 Score: 0.8172803141875307
    
    Class p3:
    Precision: 0.3391304347826087
    Recall: 0.3237735849056604
    F1 Score: 0.3312741312741313
    
    Class p4:
    Precision: 0.6528758829465187
    Recall: 0.6287657920310982
    F1 Score: 0.6405940594059406
    
    


```python
# xgboost is giving me best results
```


```python
# xgboost is giving me best results

# Apply standard scaler 

from sklearn.preprocessing import StandardScaler

columns_to_be_scaled = ['Age_Oldest_TL','Age_Newest_TL','time_since_recent_payment',
'max_recent_level_of_deliq','recent_level_of_deliq',
'time_since_recent_enq','NETMONTHLYINCOME','Time_With_Curr_Empr']

for i in columns_to_be_scaled:
    column_data = df_encoded[i].values.reshape(-1, 1)
    scaler = StandardScaler()
    scaled_column = scaler.fit_transform(column_data)
    df_encoded[i] = scaled_column
```


```python
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
print(f'Accuracy: {accuracy:.2f}')


precision, recall, f1_score, _ = precision_recall_fscore_support(y_test, y_pred)

for i, v in enumerate(['p1', 'p2', 'p3', 'p4']):
    print(f"Class {v}:")
    print(f"Precision: {precision[i]}")
    print(f"Recall: {recall[i]}")
    print(f"F1 Score: {f1_score[i]}")
    print()
    
    
    
# No improvement in metrices
```

    Accuracy: 0.78
    Class p1:
    Precision: 0.823906083244397
    Recall: 0.7613412228796844
    F1 Score: 0.7913890312660175
    
    Class p2:
    Precision: 0.8255418233924413
    Recall: 0.913577799801784
    F1 Score: 0.8673315769665035
    
    Class p3:
    Precision: 0.4756380510440835
    Recall: 0.30943396226415093
    F1 Score: 0.37494284407864653
    
    Class p4:
    Precision: 0.7342386032977691
    Recall: 0.7356656948493683
    F1 Score: 0.7349514563106796
    
    


```python
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
```

    Best Hyperparameters: {'learning_rate': 0.2, 'max_depth': 3, 'n_estimators': 200}
    Test Accuracy: 0.7811719957209081
    


```python
# # Define the hyperparameter grid
# param_grid = {
#   'colsample_bytree': [0.1, 0.3, 0.5, 0.7, 0.9],
#   'learning_rate'   : [0.001, 0.01, 0.1, 1],
#   'max_depth'       : [3, 5, 8, 10],
#   'alpha'           : [1, 10, 100],
#   'n_estimators'    : [10,50,100]
# }

# index = 0

# answers_grid = {
#     'combination'       :[],
#     'train_Accuracy'    :[],
#     'test_Accuracy'     :[],
#     'colsample_bytree'  :[],
#     'learning_rate'     :[],
#     'max_depth'         :[],
#     'alpha'             :[],
#     'n_estimators'      :[]

#     }


# # Loop through each combination of hyperparameters
# for colsample_bytree in param_grid['colsample_bytree']:
#   for learning_rate in param_grid['learning_rate']:
#     for max_depth in param_grid['max_depth']:
#       for alpha in param_grid['alpha']:
#           for n_estimators in param_grid['n_estimators']:
             
#               index = index + 1
             
#               # Define and train the XGBoost model
#               model = xgb.XGBClassifier(objective='multi:softmax',  
#                                        num_class=4,
#                                        colsample_bytree = colsample_bytree,
#                                        learning_rate = learning_rate,
#                                        max_depth = max_depth,
#                                        alpha = alpha,
#                                        n_estimators = n_estimators)
               
       
                     
#               y = df_encoded['Approved_Flag']
#               x = df_encoded. drop ( ['Approved_Flag'], axis = 1 )

#               label_encoder = LabelEncoder()
#               y_encoded = label_encoder.fit_transform(y)


#               x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42)


#               model.fit(x_train, y_train)
  

       
#               # Predict on training and testing sets
#               y_pred_train = model.predict(x_train)
#               y_pred_test = model.predict(x_test)
       
       
#               # Calculate train and test results
              
#               train_accuracy =  accuracy_score (y_train, y_pred_train)
#               test_accuracy  =  accuracy_score (y_test , y_pred_test)
              
              
       
#               # Include into the lists
#               answers_grid ['combination']   .append(index)
#               answers_grid ['train_Accuracy']    .append(train_accuracy)
#               answers_grid ['test_Accuracy']     .append(test_accuracy)
#               answers_grid ['colsample_bytree']   .append(colsample_bytree)
#               answers_grid ['learning_rate']      .append(learning_rate)
#               answers_grid ['max_depth']          .append(max_depth)
#               answers_grid ['alpha']              .append(alpha)
#               answers_grid ['n_estimators']       .append(n_estimators)
       
       
#               # Print results for this combination
#               print(f"Combination {index}")
#               print(f"colsample_bytree: {colsample_bytree}, learning_rate: {learning_rate}, max_depth: {max_depth}, alpha: {alpha}, n_estimators: {n_estimators}")
#               print(f"Train Accuracy: {train_accuracy:.2f}")
#               print(f"Test Accuracy : {test_accuracy :.2f}")
#               print("-" * 30)
```


```python
model = xgb.XGBClassifier(objective='multi:softmax',  num_class=4,colsample_bytree = 0.9,learning_rate = 1,max_depth = 3,alpha = 10,n_estimators = 100)


model.fit(x_train, y_train)
```


```python
# save the model
import pickle
filename = 'scorer.sav'
pickle.dump(model, open(filename,'wb'))


#load in flask
load_model = pickle.load(open(filename,'rb'))


arg = x_train[:2]
load_model.predict(arg)
```
