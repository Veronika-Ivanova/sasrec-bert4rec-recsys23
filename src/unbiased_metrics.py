from typing import Callable, List

import numpy as np
import pandas as pd
import scipy.stats as st


def get_metrics(preds_test, preds_val,
                user_col='test_user_idx',
                item_col='item_id',
                relevance_col='rating',
                relevance_threshold=3.5):
    preds_test_pos = preds_test[preds_test[relevance_col] >= relevance_threshold]
    preds_val_neg = preds_val[preds_val[relevance_col] < relevance_threshold]
    preds_val_pos = preds_val[preds_val[relevance_col] >= relevance_threshold]

    tp, fn = confusion_matrix_metrics(preds_val_pos, user_col, item_col)
    fp, tn = confusion_matrix_metrics(preds_val_neg, user_col, item_col)

    #biased_val_precision = tp / (tp + fp)
    beta = tp ** 2 / (tp + fn) / (tp + fp)

    metrics_dict = {'type': ['Biased', 'Unbiased', 'Unbiased_feedback_sampling']}
    metrics = [hr, mrr, ndcg]
    metric_names = ['HR', 'MRR', 'nDCG']
    for metric, metric_name in zip(metrics, metric_names):
        metrics_dict[metric_name] = calculate_metric(metric, preds_test_pos, beta)

    metrics_df = pd.DataFrame.from_dict(metrics_dict)
    return metrics_df, beta


def calculate_metric(metric: Callable,
                     preds_test_pos: pd.DataFrame, 
                     beta: float,
                     user_col: str = 'test_user_idx', 
                     item_col: str = 'item_id') -> List:
    biased_metric = metric(preds_test_pos, user_col, item_col)
    unbiased_metric = metric(preds_test_pos, user_col, item_col, beta)
    unbiased_sampled_feedback_metric = metric(preds_test_pos, user_col, item_col, beta, sample_feedback=True)
    return [biased_metric, unbiased_metric, unbiased_sampled_feedback_metric]


def confusion_matrix_metrics(
    preds: pd.DataFrame,
    user_col='test_user_idx',
    item_col='item_id',
) -> float:
    '''
    returns:
        tp, fn for positive ground_truth
        fp, tn for negative ground_truth
    '''
    positive_values = []
    negative_values = []
    for _, row in preds.iterrows():
        positive_values.append(int(row[item_col] in row['pred_items']))
        negative_values.append(int(row[item_col] not in row['pred_items']))
    return np.sum(positive_values), np.sum(negative_values)


def hr(
    preds: pd.DataFrame,
    user_col: str = 'test_user_idx',
    item_col: str = 'item_id',
    beta: float = .0,
    sample_feedback: bool = False,
    k: int = 10,
    return_confidence_interval=False,
) -> float:   
    if sample_feedback:
        mean_hr_values = []
        for seed in np.linspace(0, 1140, 20, dtype=int):
            np.random.seed(seed) 
            hr_values = []
            for _, row in preds.iterrows():
                biased_user_hr = int(row[item_col] in row['pred_items'])
                hr_values.append(max(biased_user_hr, int(np.any(np.random.binomial(1, beta, k)) > 0)))
            mean_hr_values.append(np.mean(hr_values))
        if return_confidence_interval:
            confidence_interval = st.t.interval(0.95, df=len(mean_hr_values)-1, loc=np.mean(mean_hr_values), scale=st.sem(mean_hr_values))
            return round(np.mean(mean_hr_values), 6), (confidence_interval[1] - confidence_interval[0]) / 2
        else:
            return round(np.mean(mean_hr_values), 6)
            
    else:
        hr_values = []
        for _, row in preds.iterrows():
            biased_user_hr = int(row[item_col] in row['pred_items'])
            hr_values.append(max(biased_user_hr, beta))
        return round(np.mean(hr_values), 6)


def mrr(
    preds: pd.DataFrame,
    user_col: str = 'test_user_idx',
    item_col: str = 'item_id',
    beta: float = .0,
    sample_feedback: bool = False,
    k: int = 10,
    return_confidence_interval=False,
) -> float:
    if sample_feedback:
        mean_mrr_values = []
        for seed in np.linspace(0, 1140, 20, dtype=int):
            np.random.seed(seed) 
            mrr_values = []
            for _, row in preds.iterrows():
                try:
                    biased_user_mrr = 1 / (row['pred_items'].index(row[item_col]) + 1)
                except ValueError:
                    biased_user_mrr = 0
                    
                feedback = np.random.binomial(1, beta, k)
                try:
                    sampled_feedback_user_mrr = 1 / (np.where(feedback == 1)[0][0] + 1)
                except IndexError:
                    sampled_feedback_user_mrr = 0
                mrr_values.append(max(biased_user_mrr, sampled_feedback_user_mrr))            
            mean_mrr_values.append(np.mean(mrr_values))
        if return_confidence_interval:
            confidence_interval = st.t.interval(0.95, df=len(mean_mrr_values)-1, loc=np.mean(mean_mrr_values), scale=st.sem(mean_mrr_values))
            return round(np.mean(mean_mrr_values), 6), (confidence_interval[1] - confidence_interval[0]) / 2
        else:
            return round(np.mean(mean_mrr_values), 6)
            
    else:
        mrr_values = []
        for _, row in preds.iterrows():
            try:
                biased_user_mrr = 1 / (row['pred_items'].index(row[item_col]) + 1)
            except ValueError:
                biased_user_mrr = 0
            mrr_values.append(max(biased_user_mrr, beta))
        return round(np.mean(mrr_values), 6)


def ndcg(
    preds: pd.DataFrame,
    user_col: str = 'test_user_idx',
    item_col: str = 'item_id',
    beta: float = .0,
    sample_feedback: bool = False,
    k: int = 10,
    return_confidence_interval=False,
) -> float:
    # ideal dcg == 1 при стратегии разделения leave-one-out
    if sample_feedback:
        mean_ndcg_values = []
        for seed in np.linspace(0, 1140, 20, dtype=int):
            np.random.seed(seed) 
            ndcg_values = []
            for _, row in preds.iterrows():
                try:
                    biased_user_ndcg = 1 / np.log2(row['pred_items'].index(row[item_col]) + 2)
                except ValueError:
                    biased_user_ndcg = 0
                    
                feedback = np.random.binomial(1, beta, k)
                try:
                    sampled_feedback_user_ndcg = (2 ** beta - 1) / np.log2(np.where(feedback == 1)[0][0] + 2)
                except IndexError:
                    sampled_feedback_user_ndcg = 0
                
                ndcg_values.append(max(biased_user_ndcg, sampled_feedback_user_ndcg))            
            mean_ndcg_values.append(np.mean(ndcg_values))
        if return_confidence_interval:
            confidence_interval = st.t.interval(0.95, df=len(mean_ndcg_values)-1, loc=np.mean(mean_ndcg_values), scale=st.sem(mean_ndcg_values))
            return round(np.mean(mean_ndcg_values), 6), (confidence_interval[1] - confidence_interval[0]) / 2
        else:
            return round(np.mean(mean_ndcg_values), 6)

    else:
        ndcg_values = []
        for _, row in preds.iterrows():
            try:
                biased_user_ndcg = 1 / np.log2(row['pred_items'].index(row[item_col]) + 2)
            except ValueError:
                biased_user_ndcg = 0
            ndcg_values.append(max(biased_user_ndcg, 2 ** beta - 1))
        return round(np.mean(ndcg_values), 6)
