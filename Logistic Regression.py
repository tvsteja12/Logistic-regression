import pandas as pd
from sklearn import linear_model
import numpy as np
from matplotlib import pyplot as plt
b = pd.read_csv("C:\\Users\\tvste\\OneDrive\\Desktop\\HR_comma_sep.csv")
b1 = b[b['left']==1]
b2 = b[b['left']==0]
c= pd.crosstab(b.Department,b.left)
d = pd.pivot_table(b,index = ['Department'],columns=['left'], values=['satisfaction_level'])
df = b[['satisfaction_level','average_montly_hours','promotion_last_5years','salary']]
print(d)
salary_dummies = pd.get_dummies(df['salary'],prefix = 'salary')
df1 = pd.concat([df,salary_dummies],axis = 'columns')
df2 = df1.drop(['salary'],axis='columns')
from sklearn.model_selection import train_test_split
y = b['left']
x_train,x_test,y_train,y_test = train_test_split(df2,y,test_size=0.3)
z = linear_model.LogisticRegression()
z.fit(x_train,y_train)
print(x_test)
print(z.score(x_test,y_test))
ypredict_test = z.predict(x_test)
ypredict_train = z.predict(x_train)
yproba_train = z.predict_proba(x_train)[:,1]
yproba_train.reshape(-1,1)
yproba_test = z.predict_proba(x_test)[:,1]
yproba_test.reshape(-1,1)
import statsmodels.api as sm
logit_model = sm.Logit(y,df2).fit()
from sklearn.metrics import accuracy_score
score =accuracy_score(y_test,ypredict_test)
from sklearn.metrics import confusion_matrix
print(confusion_matrix(y_test,ypredict_test))
from sklearn.metrics import classification_report
print(classification_report(y_test,ypredict_test))
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve,auc
log_curve = roc_auc_score(y_test,ypredict_test) #roc curve for test data sets
frp1,trp1,thresholf = roc_curve(y_test,yproba_test)
roccurve = auc(frp1,trp1)
from matplotlib import pyplot as plt
plt.plot(frp1,trp1,color='blue',label = 'ROC Curve (area = %0.2f)')
print(roccurve)
log1_curve = roc_auc_score(y_train,ypredict_train) # roc curve for training data sets
frp2,tpr2, threshold2 = roc_curve(y_train,yproba_train)
roccurve1= auc(frp2,tpr2)
print(roccurve1)
print(y_test,yproba_test)

