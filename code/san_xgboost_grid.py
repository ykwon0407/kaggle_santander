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

FEATURE = False
np.random.seed(44) #1000

# load data
df_train = pd.read_csv('../input/train.csv')
df_test = pd.read_csv('../input/test.csv')

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

# Making input dataset
y_train = df_train['TARGET']
X_train = df_train.drop(['ID','TARGET'], axis=1)
id_test = df_test['ID']
X_test = df_test.drop(['ID'], axis=1)

if FEATURE:
    print('Begin feature extraction')
    #The following feature extraction gives really bad auc score.
    #Feature extraction process
    X_tr, X_val, y_tr, y_val = cross_validation.train_test_split(X_train, y_train, test_size=0.2, random_state=1)
    verySimpleLearner = ensemble.GradientBoostingClassifier(n_estimators=20, max_features=1, max_depth=3, 
                                                      min_samples_leaf=100,learning_rate=0.1, 
                                                      subsample=0.65, loss='deviance', random_state=1)

    singleFeatureTable = pd.DataFrame(index=range(len(X_tr.columns)), columns=['feature','AUC'])
    for k,feature in enumerate(X_tr.columns):
        trainInputFeature = X_tr[feature].values.reshape(-1,1)
        validInputFeature = X_val[feature].values.reshape(-1,1)
        verySimpleLearner.fit(trainInputFeature, y_tr)

        validAUC = roc_auc_score(y_val, verySimpleLearner.predict_proba(validInputFeature)[:,1])
        singleFeatureTable.ix[k,'feature'] = feature
        singleFeatureTable.ix[k,'AUC'] = validAUC

    #%% sort according to AUC and present the table
    singleFeatureTable = singleFeatureTable.sort_values(by='AUC', axis=0, ascending=False).reset_index(drop=True)
    print(singleFeatureTable)
    feature_train_2 = singleFeatureTable.ix[:10,:].feature
    print(feature_train_2)
    feature_train = singleFeatureTable.ix[10:250,:].feature 
    #feature_train_drop = singleFeatureTable.ix[:20,:].feature 

    print('Begin extracting polynomial features')
    poly_2 = PolynomialFeatures(degree=2, include_bias=False)
    #poly_3 = PolynomialFeatures(degree=3, interaction_only=True, include_bias=False)
    X_train_2 = poly_2.fit_transform(X_train[feature_train_2].values)
    X_test_2 = poly_2.fit_transform(X_test[feature_train_2].values)
    #X_train_fea_3 = poly_3.fit_transform(X_train[feature_train_3].values)
    #X_test_fea_3 = poly_3.fit_transform(X_test[feature_train_3].values)
    print('End extracting polynomial features')

    '''
    X_train = X_train[feature_train]
    X_test = X_test[feature_train]
    X_train = X_train.drop(feature_train_drop, axis=1).values
    X_test = X_test.drop(feature_train_drop, axis=1).values
    '''

    X_train = np.hstack((X_train[feature_train].values, X_train_2))
    X_test = np.hstack((X_test[feature_train].values, X_test_2))
else:
    print('No feature extraction')
    
y_preds = []
cv_list =[]
print('dimension of training/test set :{}/{}'.format(X_train.shape, X_test.shape))
print("training a XGBoost classifier\n")

dtrain_sub = xgb.DMatrix(X_train, label=y_train.values)
dtest = xgb.DMatrix(X_test)

#Parameter setting    
#parameter 'eta' should be very small but not too small
num_rounds = 3000
ver_step = 100
num_thread = 30
eta = 0.007

# max-depth controls model complexity. Higher is more complex, so it encourages overfitting. 4, 5 are appropriate
param1 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param2 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param3 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param4 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param5 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param6 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 

param11 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.0, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param21 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.0, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param31 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.0, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 
param41 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param51 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param61 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 2.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 1.0, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 

param12 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param22 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param32 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param42 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param52 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param62 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.05, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 

param13 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param23 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param33 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.10, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 
param43 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}
param53 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1}  
param63 = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc', 'lambda' : 0.05, 'scale_pos_weight' : 1.0,
          'alpha': 0.1, 'nthread' : num_thread, 'subsample': 0.9, 'colsample_bytree': 0.7, 'eta' : eta, 'silent':1} 

param_bench = {'max_depth':5, 'objective':'binary:logistic', 'eval_metric': 'auc',
          'nthread' : num_thread, 'subsample': 0.7, 'colsample_bytree': 0.7, 'eta' : 0.02, 'silent':1}

#paramlist = [param12,param22,param32,param42,param52,param62,param13,param23,param33,param43,param53,param63,
#            param1,param2,param3,param4,param5,param6,param11,param21,param31,param41,param51,param61]

paramlist = [param12,param22,param32,param42,param52,param62,param13,param23,param33,param43,param53,param63]
             
for param in paramlist:    
    #xgb_cv = xgb.cv(params=param, dtrain=dtrain_sub, num_boost_round=num_rounds, nfold = 5,
    #                early_stopping_rounds=500)
    xgb_cv = xgb.cv(params=param, dtrain=dtrain_sub, num_boost_round=num_rounds, nfold = 5)    
    cv_list.append(xgb_cv.as_matrix())
    print('Cross Validation: {}'.format(xgb_cv.as_matrix()[-1,:]))
    
    #clf = xgb.train(params=param, dtrain=dtrain_sub, verbose_eval = ver_step, num_boost_round=(xgb_cv.shape[0]))
    clf = xgb.train(params=param, dtrain=dtrain_sub, verbose_eval = ver_step, num_boost_round=num_rounds)
    y_train_pred = clf.predict(dtrain_sub)
    print('AUC score: {}'.format(roc_auc_score(y_train, y_train_pred)))
    
    y_pred = clf.predict(dtest)
    y_preds.append(y_pred)


xgb_cv = xgb.cv(params=param_bench, dtrain=dtrain_sub, num_boost_round=558, nfold = 5)
cv_list.append(xgb_cv.as_matrix()[-1,:])

clf = xgb.train(params=param_bench, dtrain=dtrain_sub, verbose_eval = ver_step, num_boost_round=558)
y_pred = clf.predict(dtest)

y_sub = np.mean(y_preds, axis=0)
submission = pd.DataFrame({"ID":id_test, "TARGET":y_sub})
submission.to_csv("../input/submission_ensemble_model.csv", index=False)

print('Completed!')
#print(np.array(cv_list).round(5))
print(paramlist)