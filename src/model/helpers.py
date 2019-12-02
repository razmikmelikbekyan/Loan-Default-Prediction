from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator
from sklearn.model_selection import KFold, cross_validate


def make_cross_validation(
        models: Dict[str, Tuple[BaseEstimator, Dict]],
        data: pd.DataFrame or np.ndarray,
        target: pd.Series or np.ndarray,
        n_splits: int = 3,
        random_state: int = None,
        verbose: int = 0,
        n_jobs: int = -1) -> pd.DataFrame:
    """
    Performs cross validation of given model instances using classification metrics: AUC.
    :return: the DataFrame of cross validation scores
    """

    metrics = {'AUC': 'roc_auc'}

    cv = KFold(n_splits=n_splits, random_state=random_state)

    final_result = {}
    for model_name, (model, model_fit_params) in models.items():
        model_result = cross_validate(
            model,
            data,
            y=target,
            fit_params=model_fit_params,
            scoring=metrics,
            cv=cv,
            verbose=verbose,
            n_jobs=n_jobs
        )

        filtered_result = {}
        for key, value in metrics.items():
            metrics_values = model_result[f'test_{key}']
            mean, std = np.mean(metrics_values), np.std(metrics_values)
            ci = (mean - 1.96 * std, mean + 1.96 * std)
            ci = (f'{ci[0] :.2f}', f'{ci[1] :.2f}')

            filtered_result[f'mean_{key}'] = mean
            filtered_result[f'std_{key}'] = std
            filtered_result[f'CI_{key}'] = ci

        final_result[model_name] = filtered_result.copy()

    columns = [f'{x}_AUC' for x in ('mean', 'std', 'CI')]
    final_result = pd.DataFrame(final_result).T[columns]
    return final_result
