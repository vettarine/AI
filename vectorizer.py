"""
Модуль для векторизации текстов
"""
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import TruncatedSVD, LatentDirichletAllocation
import joblib
import numpy as np


class TextVectorizer:
    def __init__(self, config):
        self.config = config
        self.vectorizer = None
        self.feature_names = None

    def fit_transform(self, texts, method='tfidf'):
        """Обучение векторизатора и преобразование текстов."""
        print(f"Векторизация методом: {method}")

        if method == 'tfidf':
            self.vectorizer = TfidfVectorizer(
                max_features=self.config.TFIDF_MAX_FEATURES,
                max_df=self.config.TFIDF_MAX_DF,
                min_df=self.config.TFIDF_MIN_DF,
                ngram_range=self.config.TFIDF_NGRAM_RANGE,
                stop_words='english'
            )
        elif method == 'bow':
            self.vectorizer = CountVectorizer(
                max_features=self.config.BOW_MAX_FEATURES,
                max_df=self.config.BOW_MAX_DF,
                min_df=self.config.BOW_MIN_DF,
                ngram_range=(1, 2),
                stop_words='english'
            )
        else:
            raise ValueError(f"Неизвестный метод векторизации: {method}")

        # Обучение и преобразование
        X = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        print(f"Создано {X.shape[1]} признаков")
        print(f"Размер матрицы признаков: {X.shape}")

        return X

    def transform(self, texts):
        """Преобразование новых текстов."""
        if self.vectorizer is None:
            raise ValueError("Векторизатор не обучен. Сначала вызовите fit_transform.")

        return self.vectorizer.transform(texts)

    def get_feature_names(self):
        """Получение имен признаков."""
        return self.feature_names

    def reduce_dimensionality(self, X, n_components=100, method='svd'):
        """Уменьшение размерности."""
        print(f"Уменьшение размерности до {n_components} компонент методом {method}")

        if method == 'svd':
            reducer = TruncatedSVD(n_components=n_components, random_state=42)
        elif method == 'lda':
            reducer = LatentDirichletAllocation(n_components=n_components, random_state=42)
        else:
            raise ValueError(f"Неизвестный метод уменьшения размерности: {method}")

        X_reduced = reducer.fit_transform(X)
        print(f"Новая размерность: {X_reduced.shape}")

        return X_reduced, reducer

    def save_vectorizer(self, filepath):
        """Сохранение векторизатора."""
        if self.vectorizer:
            joblib.dump(self.vectorizer, filepath)
            print(f"Векторизатор сохранен в {filepath}")

    def load_vectorizer(self, filepath):
        """Загрузка векторизатора."""
        self.vectorizer = joblib.load(filepath)
        self.feature_names = self.vectorizer.get_feature_names_out()
        print(f"Векторизатор загружен из {filepath}")