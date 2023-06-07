import pandas as pd
from operator import itemgetter
import joblib
from sklearn.model_selection import GridSearchCV


class EstimatorSelectionHelper:
    
    def __init__(self, models, params):
        self.models = models
        self.params = params
        self.keys = models.keys()
        self.grid_searches = {}
        self.result = []
    
    def fit(self, X, y, **grid_kwargs):
        for key in self.keys:
            print('Running GridSearchCV for %s.' % key)
            model = self.models[key]
            params = self.params[key]
            grid_search = GridSearchCV(model, params, **grid_kwargs)
            grid_search.fit(X, y)
            self.grid_searches[key] = grid_search
            
            self.result.append(
                {
                    'grid': grid_search,
                    'classifier': grid_search.best_estimator_,
                    'best score': grid_search.best_score_,
                    'best params': grid_search.best_params_,
                    'cv': grid_search.cv
                }
            )
        print('Done.')
    
    
    def best_result(self):
        result = sorted(self.result, key=itemgetter('best score'),reverse=True)
        grid = result[0]['grid']
        joblib.dump(grid, 'classifier.pickle')
        return result
    
    def predict_proba_with_threshold(self, X, threshold):
        best_classifier = self.result[0]['classifier']
        y_proba = best_classifier.predict_proba(X)
        y_pred = (y_proba[:, 1] >= threshold).astype(int)
        return y_pred, y_proba
    
    def score_summary(self, sort_by='mean_test_score'):
        frames = []
        for name, grid_search in self.grid_searches.items():
            frame = pd.DataFrame(grid_search.cv_results_)
            frame = frame.filter(regex='^(?!.*param_).*$')
            frame['estimator'] = len(frame)*[name]
            frames.append(frame)
        df = pd.concat(frames)
        
        df = df.sort_values([sort_by], ascending=False)
        df = df.reset_index()
        df = df.drop(['rank_test_score', 'index'], 1)
        
        columns = df.columns.tolist()
        columns.remove('estimator')
        columns = ['estimator']+columns
        df = df[columns]
        return df 
    
    def get_best_score(self):
        result = sorted(self.result, key=itemgetter('best score'), reverse=True)
        return result[0]['best score']
    
    
# models1 = { 
#     'ExtraTreesClassifier': ExtraTreesClassifier(),
#     'RandomForestClassifier': RandomForestClassifier(),
#     'AdaBoostClassifier': AdaBoostClassifier(),
#     'GradientBoostingClassifier': GradientBoostingClassifier(),
#     'DecisionTreeClassifier': DecisionTreeClassifier(),
#     'LogisticRegression' :LogisticRegression()
# }

# params1 = { 
#     'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
#     'RandomForestClassifier': [
#         { 'n_estimators': [16, 32], "min_samples_leaf":[10] },
#         {'criterion': ['gini', 'entropy'], 'n_estimators': [8, 16]}],
#     'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
#     'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
#     'DecisionTreeClassifier': {'max_depth': [1,3,5,], 'min_samples_leaf': [30,35,40]},
#     'LogisticRegression': {'solver' : ['liblinear']}
# }

# helper1 = EstimatorSelectionHelper(models1, params1)
# helper1.fit(X, y, scoring='f1_macro', n_jobs=-1)
# helper1.score_summary()

# # Predict with threshold of 0.4
# threshold = 0.4
# y_pred, y_proba = helper1.predict_proba_with_threshold(X, threshold)

# # Compute the best score
# best_score = helper1.get_best_score()

# print(f"Best Score: {best_score}")




