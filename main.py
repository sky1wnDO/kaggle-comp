import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,LabelBinarizer
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score
from catboost import CatBoostClassifier
train = pd.read_csv('E:/PyCharm Projects/Kaggle/flight_delays_train.csv')
test = pd.read_csv('E:/PyCharm Projects/Kaggle/flight_delays_test.csv')

X_train, y_train = train.iloc[:,:-1], train['dep_delayed_15min'].map({'Y': 1, 'N': 0}).values
X_train['route']=X_train['Origin']+'-'+X_train['Dest']
X_train=X_train.drop(['Origin','Dest'],axis=1)
for c in ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier','route']:
    X_train[c].astype('category')

X_test=test
X_test['route']=X_test['Origin']+'-'+X_test['Dest']
X_test=X_test.drop(['Origin','Dest'],axis=1)
for c in ['Month', 'DayofMonth', 'DayOfWeek', 'UniqueCarrier','route']:
    X_test[c].astype('category')

model = CatBoostClassifier(random_seed=17)
model.fit(X_train, y_train, cat_features=[0,1,2,4,6])

test_pred = model.predict_proba(X_test)[:,1]

pd.Series(test_pred, name='dep_delayed_15min').to_csv('new_cat_2feat.csv', index_label='id', header=True)