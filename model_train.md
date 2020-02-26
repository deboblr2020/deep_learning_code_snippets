# Training Model Random Forest:

Step to perform:
1. Building the pipeline to train multiple models at the sametime.  
2. Feature extraction - pending  
3. EDA missing  



```python
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_curve, roc_curve, auc
from sklearn.model_selection import  train_test_split

X = new_df.drop(['pred_ind'],axis=1)
y = new_df['pred_ind']

X_train ,X_test, y_train, y_test = train_test_split(X,y,test_size=0.3)


pipeline = Pipeline([
    ('clf', RandomForestClassifier())
])
parameters = {
    'clf__max_depth': (4,6,8),
    'clf__class_weight':({1:0.9,0:0.1},{1:0.85,0:0.15},{1:0.8,0:0.2}),
    'clf__criterion':('gini','entropy'),
    'clf__min_samples_leaf': (500,800,1000)}

## scoring option: accuracy,auc,roc,f1 etc.
grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1, scoring='accuracy')
grid_search.fit(X_train,y_train)

print('Best score: {}'.format(grid_search.best_score_))
print('Best parameters set:')
best_parameters = grid_search.best_estimator_.get_params()
params = {}
for param_name in sorted(parameters.keys()):
    params[param_name[5:]]=best_parameters[param_name]
    print('\t {}: {}'.format(param_name[5:], best_parameters[param_name]))
params


rf = RandomForestClassifier(**params)


```
