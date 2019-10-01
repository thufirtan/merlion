import pickle
import pandas as pd
import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import StandardScaler
from category_encoders.ordinal import OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
import shap, logging

logger = logging.getLogger('merlion')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)
logger.propagate = False

def generate_train_test(original_df, target_col, test_size=0.2):
    df = original_df.copy()
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size)

DEFAULT_PARAMS = {'learning_rate': 0.15, 'metric': 'rmse', 'verbose':-1, 'first_metric_only':True}

FEATURE_SPACE = {
    'num_leaves': (10, 2000),
    'min_data_in_leaf': (10, 500),
    'colsample_bytree': (0.4, 0.99),
    'reg_lambda': (0, 10),
    'max_depth': (2, 64),
    'subsample': (0.4, 1.0),
    'feature_fraction': (0.7, 0.95)
    }

NO_PICKLE_ITEMS = ['lgb_eval', 'best_params', 'maximize', 'lgbBO', 'lgb_train']

class Merlion:
    '''Base Class for Machine Learning using LightGBM and Bayes Optimization'''
    def __init__(self, default_params=DEFAULT_PARAMS, feature_space=FEATURE_SPACE, early_stopping_rounds=100, 
        num_boost_round=100000, n_folds=5, random_seed=0, metric='rmse'):
        default_params['metric'] = metric
        self.default_params = default_params
        self.feature_space = feature_space
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.metric = metric

    def __getstate__(self):
        '''Excluding items which cannot be pickled'''
        return dict((k, v) for (k, v) in self.__dict__.items() if k not in NO_PICKLE_ITEMS)

    def fit_transform(self, X, threshold=100, nan_value_fill=''):
        '''Fit and transform training dataframe using ordinal encoder for category features'''
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        high_cardinality_features = []
        if categorical_features:
            categorical_features_nuniques = X[categorical_features].nunique()
            categorical_features = categorical_features_nuniques[lambda x: x <= threshold].index.tolist()
            high_cardinality_features = categorical_features_nuniques[lambda x: x > threshold].index.tolist()
        numerical_features = X.select_dtypes(exclude='object').columns.tolist()
        logger.info('Categorical Features {}'.format(categorical_features))
        logger.info('Numerical Features {}'.format(numerical_features))
        logger.warn('High Cardinality Features Ignored {}'.format(high_cardinality_features))

        self.preprocess = make_column_transformer(
            (make_pipeline(OrdinalEncoder()), categorical_features),
            (StandardScaler(), numerical_features))
        X_enriched = self.preprocess.fit_transform(X)
        self.feature_names = categorical_features + numerical_features
        self.categorical_features = categorical_features
        return X_enriched

    def transform(self, X):
        '''Transform dataset for testing purposes'''
        return self.preprocess.transform(X)

    def maximize(self, X_train, y_train, init_points=1, n_iter=1, acq='poi', xi=1e-4):
        '''Trigger the optimization function to search the given feature space'''
        self.lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        self.lgbBO = BayesianOptimization(f=self.lgb_eval, pbounds=self.feature_space)
        self.lgbBO.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)
        logger.info('Final result: {}'.format(self.lgbBO.max))
        return

    def validate_params(self, num_leaves, min_data_in_leaf, colsample_bytree, reg_lambda, max_depth, subsample, feature_fraction):
        '''Ensure that parameters are acceptable by lightgbm'''
        params = self.default_params.copy()
        params['num_leaves'] = int(round(num_leaves))
        params['min_data_in_leaf'] = int(round(min_data_in_leaf))
        params['colsample_bytree'] = max(min(colsample_bytree, 1), 0)
        params['reg_lambda'] = max(reg_lambda, 0)
        params['max_depth'] = int(round(max_depth))
        params['subsample'] = subsample
        params['feature_fraction'] = feature_fraction
        return params

    def lgb_eval(self, num_leaves, min_data_in_leaf, colsample_bytree, reg_lambda, max_depth, subsample, feature_fraction):
        '''Optimization function'''
        params = self.validate_params(num_leaves, min_data_in_leaf, 
            colsample_bytree, reg_lambda, max_depth, subsample, feature_fraction)
        cv_result = lgb.cv(params=params, 
                        train_set=self.lgb_train, 
                        num_boost_round=self.num_boost_round, 
                        nfold=self.n_folds, 
                        seed=self.random_seed,
                        stratified=True, 
                        metrics=self.metric, 
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=100)

        metric_value = cv_result['{}-mean'.format(self.metric[0])][-1]
        if self.metric[0] in ['rmse', 'binary_logloss', 'logloss']:
            metric_value = -metric_value
        return metric_value

    def train_single_model(self, X, y, test_size=0.2):
        '''Train a single model based on best parameters identified by optimization search'''
        self.single_model_params = self.validate_params(**self.lgbBO.max['params'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, stratify=y)
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, y_val, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        lgbr = lgb.train(self.single_model_params, lgb_train, num_boost_round=self.num_boost_round,
                        valid_sets=lgb_val, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        self.single_model = lgbr
        return

    def validate_single_model(self, X_test, y_test, metric):
        '''Validate performance of single model'''
        preds = self.generate_single_model_predictions(X_test)
        if metric == 'rmse':
            result = np.sqrt(mean_squared_error(y_test, preds))
        elif metric == 'auc':
            result = roc_auc_score(y_test, preds)
        return result     

    def train_ensemble_models(self, X, y, n=5, n_splits=5):
        '''Train ensemble models based on top N parameter combinations and using cross validation training'''
        self.ensemble_models_params = self.best_params(self.lgbBO, n)
        logger.info('Params of top {} models:'.format(n))
        logger.info(self.ensemble_models_params)
        if y.nunique() <= 5:
            kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        else:
            kf = KFold(n_splits=n_splits, shuffle=True)
        self.ensemble_models = []
        self.feature_importance_df = pd.DataFrame()
        split = 1
        for tidx, vidx in kf.split(X, y):
            X_train, X_val = X[tidx], X[vidx]
            y_train, y_val = y.iloc[tidx], y.iloc[vidx]
            lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
            lgb_val = lgb.Dataset(X_val, y_val, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
            cv_models = []
            model = 1 
            # manually increasing early stopping rounds
            for params in self.ensemble_models_params:
                params['learning_rate'] = params['learning_rate'] 
                lgbr = lgb.train(params, lgb_train, num_boost_round=self.num_boost_round,
                        valid_sets=lgb_val, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
                cv_models.append(lgbr)
                model_importance_df = pd.DataFrame()
                model_importance_df['feature'] = lgbr.feature_name()
                model_importance_df['importance'] = lgbr.feature_importance()
                model_importance_df['split'] = split
                model_importance_df['model'] = model
                self.feature_importance_df = pd.concat([self.feature_importance_df, model_importance_df], axis=0)
                model += 1
            self.ensemble_models.append(cv_models)
            split += 1
        return
    
    def best_params(self, lgbBO, n=5):
        '''Returns best parameters'''
        lgbBO_df = pd.DataFrame([x['params'] for x in lgbBO.res]).join(pd.DataFrame([x['target'] for x in lgbBO.res], columns=['target']))
        top_n = lgbBO_df.sort_values(by='target', ascending=False).head(n).drop('target', axis=1)
        model_params = []
        for record in top_n.to_dict(orient='records'):
            logger.info(record)
            logger.info(self.default_params)
            model_params.append(self.validate_params(**record))
        return model_params

    def validate_ensemble_models(self, X_test, y_test, metric):
        '''Validation of ensemble models performance'''
        average_preds = self.generate_ensemble_models_predictions(X_test)
        if metric == 'rmse':
            result = np.sqrt(mean_squared_error(y_test, average_preds))
        elif metric == 'auc':
            result = roc_auc_score(y_test, average_preds)
        return result

    def generate_shap_values(self, X):
        '''Generate shap values for single model'''
        self.explainer = shap.TreeExplainer(self.single_model)
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
        #shap.summary_plot(self.shap_values, X)

    def generate_single_model_predictions(self, X):
        '''Generate predictions for single model'''
        return self.single_model.predict(X)

    def generate_ensemble_models_predictions(self, X):
        '''Generate predictions for ensemble models'''
        preds = [[x.predict(X) for x in y] for y in self.ensemble_models]
        average_preds = np.mean(np.mean(preds, axis=0), axis=0)
        return average_preds

    def export_single_model(self, filename):
        '''Export single model as pickle file'''
        with open(filename, 'wb') as output_file:
            pickle.dump(self.single_model, output_file)
    
    def export_ensemble_models(self, filename):
        '''Export ensemble models as pickle file'''
        with open(filename, 'wb') as output_file:
            pickle.dump(self.ensemble_models, output_file)

    def export_merlion(self, filename):
        with open(filename, 'wb') as output_file:
            pickle.dump(self, output_file)
