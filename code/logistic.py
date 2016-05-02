import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time
from sklearn.linear_model import LogisticRegression
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn import ensemble
from sklearn import cross_validation
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import normalize
from sklearn.decomposition import PCA

np.random.seed(44) #1000
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
pca = PCA(n_components=2)
x_train_projected = pca.fit_transform(normalize(df_train[features], axis=0))
x_test_projected = pca.transform(normalize(df_test[features], axis=0))

df_train.insert(1, 'PCAOne', x_train_projected[:, 0])
df_train.insert(1, 'PCATwo', x_train_projected[:, 1])
df_test.insert(1, 'PCAOne', x_test_projected[:, 0])
df_test.insert(1, 'PCATwo', x_test_projected[:, 1])

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

clf = LogisticRegression(C=2.0, penalty='l1', tol=0.05)
clf.fit(X_train, y_train)
y_train_pred = clf.predict(X_train)

print('AUC score: {}'.format(roc_auc_score(y_train, y_train_pred)))
    
y_pred = clf.predict(X_test)
y_preds.append(y_pred)
    
y_sub = np.mean(y_preds, axis=0)
submission = pd.DataFrame({"ID":id_test, "TARGET":y_sub})
#submission.to_csv("../input/submission_single_model.csv", index=False)

print('Completed!')
#print(np.array(cv_list).round(5))
#print(paramlist)