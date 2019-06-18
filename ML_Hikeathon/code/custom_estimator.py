import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score as scoring_metric
from sklearn.base import clone
from sklearn.preprocessing import RobustScaler
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier, Pool, cv
from sklearn.model_selection import train_test_split
from custom_fold_generator import CustomFolds, FoldScheme
from encoding import get_categorical_column_indexes,LabelEncoding

'''
modify the scoring_metric to use custom metric, for not using roc_auc_score
scoring metric should accept y_true and y_predicted as parameters
add a functionality to give folds as an iterable
'''

class_instance = lambda a, b: eval("{}(**{})".format(a, b if b is not None else {}))

class Estimator(object):
        
    def __init__(self, model, n_splits=5, random_state=100, shuffle=True, validation_scheme=FoldScheme.StratifiedKFold,
                 categorical_features_indices=None, threshold=7,
                 cv_group_col=None, n_jobs=-1, early_stopping_rounds=None, **kwargs):
        try:
            # build model instance from tuple/list of ModelName and params
            # model should be imported before creating the instance
            self.model = class_instance(model[0], model[1])
        except Exception as e:
            # model instance is already passed
            self.model = clone(model)
            

        self.n_splits = n_splits
        self.random_state = random_state
        self.seed = random_state
        self.shuffle = shuffle
        self.n_jobs = n_jobs
        self.early_stopping_rounds = early_stopping_rounds
        self.validation_scheme = FoldScheme(validation_scheme)
        self.cv_group_col = cv_group_col
        
        self.categorical_features_indices=categorical_features_indices
        self.threshold=threshold
        self.encoder=None


    def get_params(self):
        return {
            'model': (self.model.__class__.__name__, self.model.get_params()),
            'n_splits': self.n_splits,
            'random_state': self.random_state,
            'shuffle': self.shuffle,
            'n_jobs': self.n_jobs,
            'early_stopping_rounds': self.early_stopping_rounds,
            'validation_scheme': self.validation_scheme.value,
            'cv_group_col': self.cv_group_col,
            'categorical_features_indices': self.categorical_features_indices,
            'threshold': self.threshold
            
        }

      
    def fit(self, x, y, use_oof=False, n_jobs=-1):
        if not hasattr(self.model, 'fit') :
            raise Exception ("Model/algorithm needs to implement fit()")
        fitted_models = []
        scaler_models=  []

#         if (isinstance(self.model, CatBoostClassifier)):
#             if (self.categorical_features_indices is None):
#                 temp_df=pd.DataFrame(x)
#                 temp_df=temp_df.apply(pd.to_numeric, errors='ignore')
#                 self.categorical_features_indices=get_categorical_column_indexes(temp_df,threshold=self.threshold).tolist()
#             self.encoder = LabelEncoding(categorical_columns=self.categorical_features_indices)
#             x = self.encoder.fit_transform(x)
        if use_oof:
            folds=CustomFolds(num_folds=self.n_splits,random_state=self.random_state,shuffle=self.shuffle,
                        validation_scheme=self.validation_scheme)
            self.indices=folds.split(x,y,group=self.cv_group_col)

            for i, (train_index, test_index) in enumerate(self.indices):
                model = clone(self.model)
                model.n_jobs = n_jobs
                if (isinstance(model, LGBMClassifier) and self.early_stopping_rounds is not None):
                    model.fit(X=x[train_index], y=y[train_index], eval_set=[(x[test_index],y[test_index]),(x[train_index],y[train_index])],
                        verbose=100, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds)
                elif (isinstance(model, XGBClassifier) and self.early_stopping_rounds is not None):
                    model.fit(X=x[train_index], y=y[train_index], eval_set=[(x[test_index],y[test_index])],
                        verbose=50, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds)
                elif (isinstance(model, CatBoostClassifier) and self.early_stopping_rounds is not None):
                    model.od_wait=int(self.early_stopping_rounds)
                    model.fit(x[train_index], y[train_index], cat_features=self.categorical_features_indices,
                             eval_set=(x[test_index],y[test_index]), use_best_model=True,verbose=10)
                elif isinstance(model,LogisticRegression):
                    scaler=RobustScaler()
                    xtrain=scaler.fit_transform(x[train_index])
                    scaler_models.append(scaler)
                    model.fit(xtrain,y[train_index])
                else:
                    model.fit(x[train_index], y[train_index])
                fitted_models.append(model)
        else:
            model = clone(self.model)
            model.n_jobs = n_jobs
            x_train, x_val, y_train, y_val = train_test_split(x, y, test_size =0.2,shuffle=True,random_state=100)
            if isinstance(model, LGBMClassifier):
                if self.early_stopping_rounds is not None:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val,y_val)],
                        verbose=False, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds)

            elif isinstance(model, XGBClassifier):
                if self.early_stopping_rounds is not None:
                    model.fit(X=x_train, y=y_train, eval_set=[(x_val,y_val)],
                        verbose=False, eval_metric='auc', early_stopping_rounds=self.early_stopping_rounds)

            model.fit(x, y)
            fitted_models.append(model)
        self.fitted_models = fitted_models
        self.scaler_models = scaler_models
        return self
    
    def feature_importances(self):
        if not hasattr(self, 'fitted_models') :
            raise Exception ("Model/algorithm needs to implement fit()")
        if isinstance(self.model,LogisticRegression):
            feature_importances = np.column_stack(m.coef_[0] for m in self.fitted_models)
        else:
            feature_importances = np.column_stack(m.feature_importances_ for m in self.fitted_models)
        return np.mean(1.*feature_importances/feature_importances.sum(axis=0), axis=1)

    def transform(self, x):
        if not hasattr(self, 'fitted_models') :
            raise Exception ("Model/algorithm needs to implement fit()")
        if isinstance(self.model, LogisticRegression):
            return np.mean(np.column_stack((est.predict_proba(self.scaler_models[i].transform(x))[:,1] for i, est in
                   enumerate(self.fitted_models))), axis=1)
        else:
            if isinstance(self.model, CatBoostClassifier):
                x = self.encoder.transform(x)
            return np.mean(np.column_stack((est.predict_proba(x)[:,1] for est in self.fitted_models)), axis=1)
                           
    def fit_transform(self, x, y):
        self.fit(x, y, use_oof=True)
        predictions = np.zeros((x.shape[0],))
#         if isinstance(self.model, CatBoostClassifier):
#             x = self.encoder.transform(x)
        for i, (train_index, test_index) in enumerate(self.indices):
            if isinstance(self.model,LogisticRegression):
                predictions[test_index] = self.fitted_models[i].predict_proba(self.scaler_models[i].transform(x[test_index]))[:,1]
            else:
                predictions[test_index] = self.fitted_models[i].predict_proba(x[test_index])[:,1]

        self.cv_scores = [
            scoring_metric(y[test_index], predictions[test_index])
            for i, (train_index, test_index) in enumerate(self.indices)
        ]
        self.avg_cv_score = np.mean(self.cv_scores)
        self.overall_cv_score = scoring_metric(y, predictions)
        return predictions


    def save_model(self):
        pass

    def load_model(self):
        pass

    def predict_proba(self, x):
        return self.transform(x)

    def get_repeated_out_of_folds(self, x, y, num_repeats=1):
        cv_scores = []
        for iteration in range(num_repeats):
            self.random_state = self.seed*(iteration+1)
            predictions = self.fit_transform(x, y)
            cv_scores+=self.cv_scores
            self.random_state = self.seed
        return {
            'cv_scores': cv_scores,
            'avg_cv_score': np.mean(cv_scores),
            'var_scores': np.std(cv_scores),
            'overall_cv_score': self.overall_cv_score,
        }

    def get_nested_scores(self, x, y):
        pass
