import functools
from typing import Callable, Tuple, Dict

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
from optuna.trial import Trial


def optimize_model(model_objective: Callable,
                   data: pd.DataFrame or np.ndarray,
                   target: pd.Series or np.ndarray,
                   n_trials: int,
                   cv_random_state: int,
                   verbose: bool = True) -> Tuple[Dict, float, pd.DataFrame]:
    """
    Performs classification model optimization (maximization AUC) using optuna package.
    If we want to compare results from different models, it is important to choose the same
    cv_random_state for all models in order to compare their performance on the same random
    split of data.

    :param model_objective: objective function, which should receive data, target and optuna
                            Trial object and return cross validation score
                            (example function: model.optimize_params.lgb_objective)
    :param data: train data
    :param target: train target
    :param n_trials: the number of trials to perform
    :param verbose: to print the intermediate results
    :param cv_random_state: the integer to define a random state for data split (
    :return: best params dict, best score, trails DataFrame
    """
    study = optuna.create_study(direction='maximize')

    if not verbose:
        optuna.logging.disable_default_handler()

    study.optimize(
        functools.partial(model_objective, data, target, cv_random_state),
        n_trials=n_trials,
        n_jobs=-1
    )

    if verbose:
        print(f'Number of finished trials: {len(study.trials)}')
        print('\nBest trial:')
        print(f'  Value: {study.best_value :.4f}')
        print('  Params: ')
        for key, value in study.best_params.items():
            print(f'    {key}: {value}')

    return study.best_params, study.best_value, study.trials_dataframe()


def lgb_objective(data: pd.DataFrame or np.ndarray,
                  target: pd.Series or np.ndarray,
                  cv_random_state: int,
                  trial: Trial) -> float:
    """
    Provides random run of lightgbm.LGBMClassifier object using optuna package.
    :return: R2 score of cross validation with randomly selected parameters from given space
    """
    train_set = lgb.Dataset(data, label=target)

    param = {
        'objective': 'binary',
        'verbosity': -1,
        'boosting_type': 'gbdt',
        'class_weight': trial.suggest_categorical('class_weight', [None, 'balanced']),
        'n_estimators': trial.suggest_int('n_estimators', 500, 1500),
        'learning_rate': trial.suggest_loguniform('learning_rate', 0.01, 0.5),
        'num_leaves': trial.suggest_int('num_leaves', 2, 127),
        'max_depth': trial.suggest_int('max_depth', 2, 7),
        'lambda_l1': trial.suggest_loguniform('lambda_l1', 1e-8, 10.0),
        'lambda_l2': trial.suggest_loguniform('lambda_l2', 1e-8, 10.0),
        'feature_fraction': trial.suggest_uniform('feature_fraction', 0.4, 1.0),
        'bagging_fraction': trial.suggest_uniform('bagging_fraction', 0.4, 1.0),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 7),
        'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 5, 100),
        'max_bin': trial.suggest_int('max_bin', 50, 500),
    }
    result = lgb.cv(
        param,
        train_set,
        metrics=['AUC'],
        early_stopping_rounds=50,
        nfold=5,
        seed=cv_random_state
    )

    return np.mean(result['auc-mean'])
