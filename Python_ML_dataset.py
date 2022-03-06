#!/usr/bin/env python
# coding: utf-8

# In[8]:


import pandas as pd
import numpy as np
import statistics
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime, timedelta
from math import sqrt

from sklearn.model_selection  import train_test_split
from sklearn.model_selection import cross_val_score, KFold, GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn import tree
from sklearn.svm import SVC
from sklearn import metrics
from sklearn.metrics import mean_squared_error
from sklearn.metrics import confusion_matrix, classification_report
#import graphviz

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib as mp
import matplotlib.pyplot as plt
import seaborn as sns 
import scipy.stats as scs
import scipy.stats as stats

import statsmodels.api as sm
from datetime import date, datetime, time


# In[97]:


df=pd.read_csv("C:\\Users\\admin\\Desktop\\python_ml_dataset.csv")


# In[4]:


df.head()


# In[5]:


df.describe()


# In[6]:


df.info()


# In[26]:


df['customer_id'].unique()
df['merchant_id'].unique()


# ## Checking the null values

# In[34]:


df.isnull().sum()


# In[27]:


df.head()


# ## Adding new features like day of week and month

# In[98]:


df['booking_time']=pd.to_datetime(df['booking_time'])
df['book_DOW'] = df['booking_time'].dt.dayofweek
df['book_m'] = df['booking_time'].dt.month
df['book_hr'] = df['booking_time'].dt.hour


# In[33]:


df.head()


# In[51]:


## Visualising

df2=df["book_DOW"].value_counts().to_frame().reset_index()
df2.columns=["book_DOW","count"]


# In[52]:


df2


# In[63]:


sns.barplot(df2['book_DOW'],df2["count"])
plt.xticks(rotation=35)
plt.show()


# In[61]:


df3=df["book_m"].value_counts().to_frame().reset_index()
df3.columns=["book_m","count"]
sns.barplot(df3['book_m'],df3["count"])
plt.xticks(rotation=35)
plt.show()


# In[60]:


df3=df["book_hr"].value_counts().to_frame().reset_index()
df3.columns=["book_hr","count"]
plt.figure(2)
sns.barplot(df3['book_hr'],df3["count"])
plt.xticks(rotation=35)
plt.show()


# In[64]:


df.head()


# In[86]:


df3=df["merchant_id"].value_counts().to_frame().reset_index()
df3.columns=["merchant_id","count"]
plt.figure(2)
sns.barplot(df3['merchant_id'],df3["count"])
plt.xticks(rotation=35)
plt.show()


# ### Findings:
# ### 1. highest number of order during 11-12 th hour and 18th-19th hour of the day
# ### 2. In the given data, highest no. of order in Sep and October

# In[75]:


list(df)


# In[77]:


Numerical_column=[
 'booking_distance',
 'booking_amount',
 'actual_total_time',
 'book_DOW',
 'book_m',
 'book_hr']
for i in (Numerical_column):
    boxplt = df.boxplot(column=[i])
    plt.show()


# ## checking multicollienarity

# In[79]:


df.corr()


# In[99]:


df['customer_id']=df['customer_id'].astype(str)
df['merchant_id']=df['merchant_id'].astype(str)


# In[100]:


df.info()


# In[82]:


df.corr()


# In[85]:


corr1=df.corr()
corr=sns.heatmap(corr1, annot=True)


# ## Finding: variables are not correlated

# ## Merchant id,Payment method and Customer id does not play any role in prediction so we will drop them from  data

# In[116]:


#independent and dependent features
df_new=df.copy()
df_new.drop(['customer_id','customer_id','payment_method_name','booking_time'],axis=1,inplace=True)
x=df_new.drop(['actual_total_time'],axis=1)
y=df_new['actual_total_time']


# In[113]:


y


# ## Linear Regression

# In[118]:


#Data partitioning into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.40,random_state=109) 
model=LinearRegression().fit(X_train,y_train)
# get importance
importance = model.coef_
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[119]:


importance


# In[121]:


out_forecast = model.predict(X_test)
in_forecast = model.predict(X_train)


# In[122]:


out_forecast


# In[123]:


in_forecast


# In[124]:


def mae(actual,forecast):
    result = 0 
    for a,f in zip(actual,forecast):
        result += abs(a-f)
    return result/len(forecast)

def mape(actual, forecast): 
    result = 0 
    for a,f in zip(actual,forecast):
            result += abs(a-f)/a
    return result/len(forecast)
    
def rmse(actual, forecast): 
    rmse = np.sqrt(mean_squared_error(actual, forecast))
    return rmse

def run_error_result(actual,forecast,name):
    def mae(actual,forecast):
        result = 0 
        for a,f in zip(actual,forecast):
            result += abs(a-f)
        return result/len(forecast)

    def mape(actual, forecast): 
        result = 0 
        for a,f in zip(actual,forecast):
                result += abs(a-f)/a
        return result/len(forecast)
    
    def rmse(actual, forecast): 
        rmse = np.sqrt(mean_squared_error(actual, forecast))
        return rmse
    
    return pd.DataFrame({'MAE':[mae(actual, forecast)],'MAPE': [mape(actual,forecast)],'RMSE': [rmse(actual,forecast)]},index=pd.Series([name]))


# In[125]:


print(run_error_result(y_test, out_forecast,'Out-sample'))
print(run_error_result(y_train, in_forecast,'In-sample'))


# ## Random Forest

# In[129]:


#Data partitioning into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.30,random_state=109)
rf = RandomForestRegressor()
rf.fit(X_train, y_train)


# In[130]:


print(rf.feature_importances_)


# In[132]:


f_imp=pd.Series(rf.feature_importances_,index=x.columns)
f_imp.plot(kind='barh')
plt.show()


# In[133]:


## predict test set  (out-sample)
out_forecast = rf.predict(X_test)
## predict train set  (in-sample)
in_forecast = rf.predict(X_train)
print(run_error_result(y_test, out_forecast,'Out-sample'))
print(run_error_result(y_train, in_forecast,'In-sample'))


# ## Random forest gives least error in-sampl so going with it

# In[134]:


X_test['pred_delivery_time'] = rf.predict(X_test)
X_train['pred_delivery_time'] = rf.predict(X_train)


# In[136]:


X_train


# In[140]:


a = pd.DataFrame(y_test)
a.columns = ['actual_delivery_time']
b = pd.DataFrame(y_train)
b.columns = ['actual_delivery_time']
test_result = X_test.join(a)
train_result = X_train.join(b)


# In[141]:


test_result


# In[142]:


train_result


# In[ ]:




