"""
Модуль с моделями машинного обучения
"""
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib
import numpy as np


class FakeNewsClassifier:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_params = {}

    def train_logistic_regression(self, X_train, y_train, X_val=None, y_val=None):
        """Обучение логистической регрессии."""
        print("Обучение модели логистической регрессии...")

        # Параметры для GridSearch
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'penalty': ['l1', 'l2'],
            'solver': ['liblinear', 'saga']
        }

        # Базовые параметры
        base_lr = LogisticRegression(
            max_iter=self.config.LR_MAX_ITER,
            class_weight=self.config.LR_CLASS_WEIGHT,
            random_state=self.config.RANDOM_STATE
        )

        # GridSearch для подбора параметров
        grid_search = GridSearchCV(
            base_lr,
            param_grid,
            cv=self.config.CV_FOLDS,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший score: {grid_search.best_score_:.4f}")

        self.models['logistic_regression'] = grid_search.best_estimator_
        self.best_params['logistic_regression'] = grid_search.best_params_

        return grid_search.best_estimator_

    def train_svm(self, X_train, y_train, X_val=None, y_val=None):
        """Обучение SVM."""
        print("Обучение модели SVM...")

        # Для линейного ядра
        param_grid = {
            'C': [0.1, 1.0, 10.0],
            'kernel': ['linear', 'rbf'],
            'gamma': ['scale', 'auto']
        }

        # Базовые параметры
        base_svm = SVC(
            class_weight=self.config.SVM_CLASS_WEIGHT,
            probability=True,
            random_state=self.config.RANDOM_STATE
        )

        # GridSearch для подбора параметров
        grid_search = GridSearchCV(
            base_svm,
            param_grid,
            cv=self.config.CV_FOLDS,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )

        grid_search.fit(X_train, y_train)

        print(f"Лучшие параметры: {grid_search.best_params_}")
        print(f"Лучший score: {grid_search.best_score_:.4f}")

        self.models['svm'] = grid_search.best_estimator_
        self.best_params['svm'] = grid_search.best_params_

        return grid_search.best_estimator_

    def train_random_forest(self, X_train, y_train):
        """Обучение случайного леса (дополнительная модель)."""
        print("Обучение модели случайного леса...")

        from sklearn.ensemble import RandomForestClassifier

        rf = RandomForestClassifier(
            n_estimators=100,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            class_weight='balanced',
            random_state=self.config.RANDOM_STATE,
            n_jobs=-1
        )

        rf.fit(X_train, y_train)
        self.models['random_forest'] = rf

        return rf

    def predict(self, model_name, X):
        """Предсказание с использованием выбранной модели."""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")

        model = self.models[model_name]

        # Для моделей с вероятностями
        if hasattr(model, 'predict_proba'):
            y_pred_proba = model.predict_proba(X)[:, 1]
            y_pred = model.predict(X)
            return y_pred, y_pred_proba
        else:
            y_pred = model.predict(X)
            return y_pred, None

    def evaluate(self, model_name, X_test, y_test):
        """Оценка модели."""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")

        model = self.models[model_name]

        # Предсказания
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

        # Метрики
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        cm = confusion_matrix(y_test, y_pred)

        results = {
            'accuracy': accuracy,
            'precision': report['weighted avg']['precision'],
            'recall': report['weighted avg']['recall'],
            'f1': report['weighted avg']['f1-score'],
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'predictions': y_pred.tolist(),
            'probabilities': y_pred_proba.tolist() if y_pred_proba is not None else None
        }

        # Вывод результатов
        print(f"\nРезультаты для модели {model_name}:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {results['precision']:.4f}")
        print(f"Recall: {results['recall']:.4f}")
        print(f"F1-Score: {results['f1']:.4f}")
        print("\nConfusion Matrix:")
        print(cm)

        return results

    def cross_validate(self, model_name, X, y, cv=5):
        """Кросс-валидация."""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")

        model = self.models[model_name]

        scores = cross_val_score(model, X, y, cv=cv, scoring='f1', n_jobs=-1)

        print(f"\nКросс-валидация для {model_name} (F1-score):")
        print(f"Среднее: {scores.mean():.4f} (±{scores.std():.4f})")
        print(f"Все оценки: {scores}")

        return scores

    def get_feature_importance(self, model_name, feature_names, top_n=20):
        """Получение важности признаков."""
        if model_name not in self.models:
            raise ValueError(f"Модель {model_name} не обучена")

        model = self.models[model_name]

        try:
            if hasattr(model, 'coef_'):
                # Для линейных моделей
                coef = model.coef_
                # Преобразование разреженной матрицы в плотную
                if hasattr(coef, 'toarray'):
                    importance = np.abs(coef.toarray()[0])
                else:
                    importance = np.abs(coef[0])
            elif hasattr(model, 'feature_importances_'):
                # Для Random Forest
                importance = model.feature_importances_
            else:
                print(f"Модель {model_name} не поддерживает извлечение важности признаков")
                return None

            # Сортировка по важности
            indices = np.argsort(importance)[::-1]

            top_features = []
            for i in range(min(top_n, len(feature_names))):
                idx = indices[i]
                top_features.append({
                    'feature': feature_names[idx],
                    'importance': float(importance[idx])
                })

            return top_features
        except Exception as e:
            print(f"Ошибка при извлечении важности признаков: {e}")
            return None

    def save_model(self, model_name, filepath):
        """Сохранение модели."""
        if model_name in self.models:
            joblib.dump(self.models[model_name], filepath)
            print(f"Модель {model_name} сохранена в {filepath}")

    def load_model(self, model_name, filepath):
        """Загрузка модели."""
        self.models[model_name] = joblib.load(filepath)
        print(f"Модель {model_name} загружена из {filepath}")

    def train_model(self, model_name, X_train, y_train):
        """Обучение модели по имени."""
        if model_name == 'logistic_regression':
            return self.train_logistic_regression(X_train, y_train)
        elif model_name == 'svm':
            return self.train_svm(X_train, y_train)
        elif model_name == 'random_forest':
            return self.train_random_forest(X_train, y_train)
        else:
            raise ValueError(f"Неизвестная модель: {model_name}")