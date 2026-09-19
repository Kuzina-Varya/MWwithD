"""Пользовательские трансформеры для создания признаков."""

from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


class MissingValueCountTransformer(TransformerMixin, BaseEstimator):
    """Добавить количество пропущенных значений в каждой строке."""

    def fit(
        self, features: pd.DataFrame, target: pd.Series | None = None
    ) -> MissingValueCountTransformer:
        """Запомнить состав и порядок входных признаков."""
        del target
        self._validate_dataframe(features)
        self.feature_names_in_ = features.columns.to_numpy(dtype=object)
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def transform(self, features: pd.DataFrame) -> pd.DataFrame:
        """Создать копию данных с числом пропусков в столбце MissingCount."""
        check_is_fitted(self, "feature_names_in_")
        self._validate_dataframe(features)

        missing_columns = set(self.feature_names_in_).difference(features.columns)
        if missing_columns:
            raise ValueError(
                f"Для подсчёта пропусков отсутствуют столбцы: "
                f"{sorted(missing_columns)}"
            )

        transformed = features.loc[:, self.feature_names_in_].copy()
        transformed["MissingCount"] = transformed.isna().sum(axis=1)
        return transformed

    def get_feature_names_out(self, input_features: Any = None) -> Any:
        """Вернуть имена исходных и созданного признаков."""
        del input_features
        check_is_fitted(self, "feature_names_in_")
        names = [*self.feature_names_in_, "MissingCount"]
        return pd.Index(names).to_numpy(dtype=object)

    @staticmethod
    def _validate_dataframe(features: pd.DataFrame) -> None:
        """Проверить, что трансформер получил таблицу pandas."""
        if not isinstance(features, pd.DataFrame):
            raise TypeError("MissingValueCountTransformer ожидает pandas DataFrame")
