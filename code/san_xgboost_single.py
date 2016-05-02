import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xgboost as xgb
import time
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize

np.random.seed(13) #1000
# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

def insert_df(df, string, array):
    df.insert(1, string, array)
    
print('FEATURE ENGINEERING START')

features = df_train.columns[1:-1]
df_train.insert(1, 'SumZeros', (df_train[features] == 0).astype(int).sum(axis=1))
df_test.insert(1, 'SumZeros', (df_test[features] == 0).astype(int).sum(axis=1))
df_train = df_train.replace(-999999,2)
df_test = df_test.replace(-999999,2)

# remove constant columns
remove = []
for col in df_train.columns:
    if df_train[col].std() == 0:
        remove.append(col)

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

# remove duplicated columns
remove = []
c = df_train.columns
for i in range(len(c)-1):
    v = df_train[c[i]].values
    for j in range(i+1,len(c)):
        if np.array_equal(v,df_train[c[j]].values):
            remove.append(c[j])

df_train.drop(remove, axis=1, inplace=True)
df_test.drop(remove, axis=1, inplace=True)

features = df_train.columns[1:-1]

# Making input dataset
y_train = df_train['TARGET']
X_train = df_train.drop(['ID','TARGET'], axis=1)
id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1)   
    
print('TRAINING START')

y_preds = []
cv_list =[]
print('dimension of training/test set :{}/{}'.format(X_train.shape, X_test.shape))
print("training a XGBoost classifier\n")

dtrain_sub = xgb.DMatrix(X_train, label=y_train.values)
dtest = xgb.DMatrix(X_test)

#Parameter setting    
#parameter 'eta' should be very small but not too small
num_rounds = 1000
ver_step = 50
num_thread = 24
eta = 0.0102048

# max-depth controls model complexity. Higher is more complex, so it encourages overfitting. 4, 5 are appropriate
param1 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc',
          'nthread' : num_thread, 'subsample': 0.6815, 'colsample_bytree': 0.701, 'eta' : eta, 'silent':1}

paramlist = [param1]
             
for param in paramlist:    
    xgb_cv = xgb.cv(params=param, dtrain=dtrain_sub, num_boost_round=num_rounds,
                    early_stopping_rounds=75, nfold = 10)    
    #xgb_cv = xgb.cv(params=param, dtrain=dtrain_sub, num_boost_round=560, nfold=10)
    cv_list.append(xgb_cv.as_matrix())
    print('Cross Validation: {}'.format(xgb_cv.as_matrix()[-1,:]))
    
    clf = xgb.train(params=param, dtrain=dtrain_sub, verbose_eval = ver_step, num_boost_round=xgb_cv.shape[0])
    y_train_pred = clf.predict(dtrain_sub)
    print('AUC score: {}'.format(roc_auc_score(y_train, y_train_pred)))
    
    y_pred = clf.predict(dtest)
    y_preds.append(y_pred)

y_sub = np.mean(y_preds, axis=0)
submission = pd.DataFrame({"ID":id_test, "TARGET":y_sub})
submission.to_csv("../input/submission_single_model.csv", index=False)

print('Completed!')
#print(np.array(cv_list).round(5))
print(paramlist)