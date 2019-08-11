import pandas as pd 

#n distribution of data 
dist = df.describe(percentiles=[0.05,.25,.5,.75,.95]).T


# KDE Plot
dfp = pd.DataFrame({'bad':y_test,'scores':pred_test[:,1] })
dfp.groupby('bad')['scores'].plot(kind='kde',legend=True)

# Modeling
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score,precision_recall_curve, roc_curve, auc
from sklearn.model_selection import  train_test_split
from sklearn.externals import joblib

X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)

# model XGBoost
from sklearn.ensemble import GradientBoostingClassifier
clf = GradientBoostingClassifier(learning_rate=0.1,min_samples_leaf=50)
clf.fit(X_train,y_train)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100, max_depth=2,random_state=0)

# Important Variable
important_features = pd.Series(data=clf.feature_importances_,index=X_train.columns)
important_features.sort_values(ascending=False,inplace=True)
important_features.plot(kind='bar', rot=80)


# probability
pred_train = dt.predict_proba(X_train)
print(classification_report(y_train,pred_train[:,1]>0.5))

# Auc score
fpr, tpr, roc_thresholds = roc_curve(y_test, pred_test[:,1])
auc_score = auc(fpr, tpr)





