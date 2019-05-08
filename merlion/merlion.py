import pandas as pd
import numpy as np
import lightgbm as lgb
from bayes_opt import BayesianOptimization
from sklearn.model_selection import StratifiedKFold, train_test_split, KFold
from sklearn.metrics import mean_squared_error, roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
import shap, logging

logger = logging.getLogger('merlion')
handler = logging.StreamHandler()
formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)

def generate_train_test(original_df, target_col, test_size=0.2):
    df = original_df.copy()
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    return train_test_split(X, y, test_size=test_size)

# def generate_encoded_labels(df):
#     categories = df.select_dtypes(include='object').columns.tolist()
#     for category in categories:
#         le = LabelEncoder()
#         df[category] = le.fit_transform(df[category].astype(str))
#         df[category] = df[category].astype('category')
#     return df, categories

# def generate_dummies(df):
#     for category in df.select_dtypes(include='object').columns:
#         dummies = pd.get_dummies(df[category], prefix=category)
#         df = pd.concat([df, dummies], axis=1)
#         df.drop(category, axis=1, inplace=True)
#     return df

DEFAULT_PARAMS = {'learning_rate': 0.15, 'metric': 'rmse', 'verbose':-1}
N_FOLDS = 5
RANDOM_SEED = 0
EARLY_STOPPING_ROUNDS = 100
NUM_BOOST_ROUND = 100000

FEATURE_SPACE = {
    'num_leaves': (10, 2000),
    'min_data_in_leaf': (10, 500),
    'colsample_bytree': (0.4, 0.99),
    'reg_lambda': (0, 10),
    'max_depth': (2, 64),
    'subsample': (0.4, 1.0),
    'feature_fraction': (0.7, 0.95)
    }

class Merlion:
    def __init__(self, default_params=DEFAULT_PARAMS, feature_space=FEATURE_SPACE, early_stopping_rounds=EARLY_STOPPING_ROUNDS, 
        num_boost_round=NUM_BOOST_ROUND, n_folds=N_FOLDS, random_seed=RANDOM_SEED, metric='rmse'):
        default_params['metric'] = metric
        self.default_params = default_params
        self.feature_space = feature_space
        self.early_stopping_rounds = early_stopping_rounds
        self.num_boost_round = num_boost_round
        self.n_folds = n_folds
        self.random_seed = random_seed
        self.metric = metric

    def fit_transform(self, X):
        categorical_features = X.select_dtypes(include='object').columns.tolist()
        numerical_features = X.select_dtypes(exclude='object').columns.tolist()
        logger.info(f'Categorical Features {categorical_features}')
        logger.info(f'Numerical Features {numerical_features}')
        self.preprocess = make_column_transformer(
            (OrdinalEncoder(), categorical_features), remainder='passthrough')
        X_enriched = self.preprocess.fit_transform(X)
        self.feature_names = categorical_features + numerical_features
        self.categorical_features = categorical_features
        return X_enriched

    def transform(self, X):
        return self.preprocess.transform(X)

    def maximize(self, X_train, y_train, init_points=1, n_iter=1, acq='poi', xi=1e-4):
        self.lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        self.lgbBO = BayesianOptimization(f=self.lgb_eval, pbounds=self.feature_space)
        self.lgbBO.maximize(init_points=init_points, n_iter=n_iter, acq=acq, xi=xi)
        logger.info(f"Final result: {self.lgbBO.max}")
        return

    def validate_params(self, num_leaves, min_data_in_leaf, colsample_bytree, reg_lambda, max_depth, subsample, feature_fraction):
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
        params = self.validate_params(num_leaves, min_data_in_leaf, 
            colsample_bytree, reg_lambda, max_depth, subsample, feature_fraction)
        cv_result = lgb.cv(params=params, 
                        train_set=self.lgb_train, 
                        num_boost_round=self.num_boost_round, 
                        nfold=self.n_folds, 
                        seed=self.random_seed,
                        stratified=False, 
                        metrics=[self.metric], 
                        early_stopping_rounds=self.early_stopping_rounds,
                        verbose_eval=100)

        metric_value = cv_result[f'{self.metric}-mean'][-1]
        if self.metric == 'rmse':
            metric_value = -metric_value
        return metric_value

    def train_single_model(self, X, y):
        self.single_model_params = self.validate_params(**self.lgbBO.max['params'])
        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
        lgb_train = lgb.Dataset(X_train, y_train, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        lgb_val = lgb.Dataset(X_val, y_val, feature_name=self.feature_names, categorical_feature=self.categorical_features, free_raw_data=False)
        lgbr = lgb.train(self.single_model_params, lgb_train, num_boost_round=self.num_boost_round,
                        valid_sets=lgb_val, early_stopping_rounds=self.early_stopping_rounds, verbose_eval=False)
        self.single_model = lgbr
        return

    def validate_single_model(self, X_test, y_test, metric):
        preds = self.single_model.predict(X_test)
        if metric == 'rmse':
            result = np.sqrt(mean_squared_error(y_test, preds))
        elif metric == 'auc':
            result = roc_auc_score(y_test, preds)
        return result     

    def train_ensemble_models(self, X, y, n=5, n_splits=5):
        self.ensemble_models_params = self.best_params(self.lgbBO, n)
        logger.info(f'Params of top {n} models:')
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
        lgbBO_df = pd.DataFrame([x['params'] for x in lgbBO.res]).join(pd.DataFrame([x['target'] for x in lgbBO.res], columns=['target']))
        top_n = lgbBO_df.sort_values(by='target', ascending=False).head(n).drop('target', axis=1)
        model_params = []
        for record in top_n.to_dict(orient='records'):
            logger.info(record)
            logger.info(self.default_params)
            model_params.append(self.validate_params(**record))
        return model_params

    def validate_ensemble_models(self, X_test, y_test, metric):
        average_preds = self.generate_ensemble_models_predictions(X_test)
        if metric == 'rmse':
            result = np.sqrt(mean_squared_error(y_test, average_preds))
        elif metric == 'auc':
            result = roc_auc_score(y_test, average_preds)
        return result

    def generate_shap_values(self, X):
        self.explainer = shap.TreeExplainer(self.single_model)
        self.shap_values = self.explainer.shap_values(X)
        return self.shap_values
        #shap.summary_plot(self.shap_values, X)

    def generate_single_model_predictions(self, X):
        return self.single_model.predict(X)

    def generate_ensemble_models_predictions(self, X):
        preds = [[x.predict(X) for x in y] for y in self.ensemble_models]
        average_preds = np.mean(np.mean(preds, axis=0), axis=0)
        return average_preds