import gc

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags
from scipy.sparse.linalg import svds
from tqdm import tqdm


class SVD:
    """
    Parameters
    ----------
    n_factors: размерность эмбеддингов.
    alpha: степень масштабирования
    """
    def __init__(self,
                 n_factors: int=20,
                 alpha: float=0.5):

        self.n_factors = n_factors
        self.alpha = alpha

    def fit(self, X: pd.DataFrame):
        """Обучает модель.
        """
        X = self._preprocess_data(X)

        self.build_svd_model(X)

    def _preprocess_data(self, X: pd.DataFrame):
        """Заменяет id  пользователей и фильмов на их индексы.
        """
        X = X.copy()

        self.user_ids = X['user_id'].unique().tolist()
        self.item_ids = X['item_id'].unique().tolist()

        self.n_items = len(self.item_ids)
        self.n_users = len(self.user_ids)

        item_idx = range(self.n_items)
        user_idx = range(self.n_users)
        
        self.item_mapping_ = dict(zip(self.item_ids, item_idx))
        self.user_mapping_ = dict(zip(self.user_ids, user_idx))
        
        X['item_id'] = X['item_id'].map(self.item_mapping_)
        X['user_id'] = X['user_id'].map(self.user_mapping_)
        return X

    def generate_interactions_matrix(self, X: pd.DataFrame, n_users: int, user_col: str = 'user_id'):
        '''
        Создает sparse matrix для обучающей выборки
        '''
        item_idx = X['item_id'].values
        user_idx = X[user_col].values
        ratings = X['rating'].values
        return csr_matrix(
            (ratings, (user_idx, item_idx)),
            shape=(n_users, self.n_items),
            dtype=np.int8
            )

    def scale_matrix(self):
        D = diags(self.matrix.getnnz(axis=0) ** (0.5 * (self.alpha - 1)))
        return self.matrix.dot(D)

    def build_svd_model(self, X: pd.DataFrame):
        self.matrix = self.generate_interactions_matrix(X, self.n_users)
        scaled_matrix = self.scale_matrix()
        _, _, vt = svds(scaled_matrix, k=self.n_factors, return_singular_vectors='vh')
        self.item_factors = np.ascontiguousarray(vt[::-1, :].T, dtype=np.float32)

    def predict_folding_in(self, X: pd.DataFrame):
        X = X.copy()
        X['item_id'] = X['item_id'].map(self.item_mapping_)
        
        prediction_matrix = self.generate_interactions_matrix(X, X['test_user_idx'].max() + 1, user_col='test_user_idx')
        return self.predict(prediction_matrix)

    def predict(self, matrix: csr_matrix):
        # обнуление скоров для айтемов из обучающей выборки, чтобы они не попали в рекомендации
        # учтено, что среди скоров будет достаточное количество положительных
        return np.multiply(
            matrix.dot(self.item_factors) @ self.item_factors.T,
            np.invert(matrix.astype(bool).toarray())
            )

    def get_top_k(self, scores: np.array, user_col: str = 'test_user_idx', batch_size: int = 70000, k: int = 10):
        id2item = {v: k for k, v in self.item_mapping_.items()}
        preds = pd.DataFrame()

        for batch_ind_left in tqdm(range(0, scores.shape[0], batch_size)):
            batch_ind_right = min(batch_ind_left + batch_size, scores.shape[0])
            ind_part = np.argpartition(
                scores[batch_ind_left: batch_ind_right],
                -k + 1
                )[:, -k:].copy()
            scores_not_sorted = np.take_along_axis(
                scores[batch_ind_left: batch_ind_right],
                ind_part,
                axis=1
                )
            ind_sorted = np.argsort(scores_not_sorted, axis=1)
            indices = np.take_along_axis(ind_part, ind_sorted, axis=1)
            preds = pd.concat([preds, pd.DataFrame({
                user_col: range(batch_ind_left, batch_ind_right),
                'pred_items': indices.tolist()
                })])

            gc.collect()

        preds['pred_items'] = preds['pred_items'].map(lambda inds: [id2item[i] for i in inds])
        if user_col == 'user_id':
            id2user = {v: k for k, v in self.user_mapping_.items()}
            preds[user_col] = preds[user_col].map(id2user)
        return preds