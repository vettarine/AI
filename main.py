"""
Основной скрипт для классификации фейковых новостей
"""
import os
import sys
import numpy as np
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from config import *
from data_loader import DataLoader
from vectorizer import TextVectorizer
from models import FakeNewsClassifier
from utils import Visualization, ReportGenerator


def main():
    """Основная функция проекта."""
    print("=" * 80)
    print("КЛАССИФИКАЦИЯ ФЕЙКОВЫХ НОВОСТЕЙ")
    print("Формат: Текстовые файлы в папках")
    print(f"Метод: {VectorizationConfig.VECTORIZATION_METHOD.upper()} + Logistic Regression & SVM")
    print("=" * 80)



    try:
        # Инициализация
        print("\n1. ИНИЦИАЛИЗАЦИЯ")
        print("-" * 40)

        # Создание папок для результатов
        Visualization.create_output_directories(OutputConfig)

        # Загрузка данных
        print("\n2. ЗАГРУЗКА ДАННЫХ")
        print("-" * 40)

        data_loader = DataLoader(PreprocessingConfig)

        # Загрузка тренировочных данных
        train_df = data_loader.load_text_files(
            DataConfig.TRAIN_FAKE_PATHS,
            DataConfig.TRAIN_REAL_PATHS
        )

        if len(train_df) == 0:
            print("ОШИБКА: Не загружены тренировочные данные!")
            return 1

        # Анализ тренировочных данных
        train_info = data_loader.analyze_dataset(train_df, "Тренировочные данные")

        # Загрузка тестовых данных
        test_df = data_loader.load_test_data(
            DataConfig.TEST_FAKE_PATH,
            DataConfig.TEST_REAL_PATH
        )

        # Анализ тестовых данных
        if len(test_df) > 0:
            test_info = data_loader.analyze_dataset(test_df, "Тестовые данные")
        else:
            print("\nТестовые данные не загружены.")
            test_df = None

        # Предобработка данных
        print("\n3. ПРЕДОБРАБОТКА ДАННЫХ")
        print("-" * 40)

        # Предобработка тренировочных данных
        train_df_processed = data_loader.preprocess_dataframe(train_df)

        # Балансировка классов
        train_df_balanced = data_loader.balance_classes(train_df_processed)

        # Предобработка тестовых данных
        if test_df is not None and len(test_df) > 0:
            test_df_processed = data_loader.preprocess_dataframe(test_df)
        else:
            test_df_processed = None

        # Проверка и удаление дубликатов
        print("\n4. ПРОВЕРКА И УДАЛЕНИЕ ДУБЛИКАТОВ")
        print("-" * 40)

        if test_df_processed is not None and len(test_df_processed) > 0:
            print("Проверка дубликатов между train и test...")

            # Создание копии для безопасности
            train_df_final = train_df_balanced.copy()
            test_df_final = test_df_processed.copy()

            # Проверка дубликатов через hash
            train_hashes = [hash(str(text)) for text in train_df_final['processed_text']]
            test_hashes = [hash(str(text)) for text in test_df_final['processed_text']]

            common_hashes = set(train_hashes).intersection(set(test_hashes))

            if common_hashes:
                print(f"Найдено {len(common_hashes)} дубликатов между train и test!")

                # Создание новых DataFrame без дубликатов в test
                duplicate_indices = []
                for idx, test_hash in enumerate(test_hashes):
                    if test_hash in common_hashes:
                        duplicate_indices.append(idx)

                if len(duplicate_indices) > 0:
                    test_df_final = test_df_final.drop(duplicate_indices).reset_index(drop=True)
                    print(f"Удалено {len(duplicate_indices)} дубликатов из тестовых данных")
            else:
                print("Дубликаты не обнаружены")

            # Проверяем, остались ли тестовые данные
            if len(test_df_final) == 0:
                print("\nПосле удаления дубликатов тестовые данные пусты!")
                print("Используем разделение тренировочных данных на train/test")
                use_separate_test = False
            else:
                print(f"Осталось тестовых данных: {len(test_df_final)} записей")
                use_separate_test = True
        else:
            print("Тестовые данные отсутствуют или пусты")
            use_separate_test = False
            train_df_final = train_df_balanced.copy()

        # Подготовка данных для обучения
        print("\n5. ПОДГОТОВКА ДАННЫХ ДЛЯ ОБУЧЕНИЯ")
        print("-" * 40)

        if use_separate_test and test_df_final is not None and len(test_df_final) > 0:
            # Используем отдельные тестовые данные
            X_train = train_df_final['processed_text'].values
            y_train = train_df_final['label'].values
            X_test = test_df_final['processed_text'].values
            y_test = test_df_final['label'].values

            print(f"Используются отдельные тестовые данные:")
            print(f"  Train: {len(X_train)} записей")
            print(f"  Test:  {len(X_test)} записей")
        else:
            # Разделение тренировочных данных на train/test
            print("Используем разделение тренировочных данных на train/test")

            X = train_df_final['processed_text'].values
            y = train_df_final['label'].values

            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size=ModelConfig.TEST_SIZE,
                random_state=ModelConfig.RANDOM_STATE,
                stratify=y
            )

            print(f"Создано разделение:")
            print(f"  Train: {len(X_train)} записей")
            print(f"  Test:  {len(X_test)} записей")

        # Векторизация текстов
        print("\n6. ВЕКТОРИЗАЦИЯ ТЕКСТОВ")
        print("-" * 40)

        vectorizer = TextVectorizer(VectorizationConfig)

        # Выбор метода векторизации
        vec_method = VectorizationConfig.VECTORIZATION_METHOD
        print(f"Используется метод: {vec_method.upper()}")

        # Обучение векторизатора на тренировочных данных
        X_train_vec = vectorizer.fit_transform(X_train, method=vec_method)

        # Преобразование тестовых данных
        X_test_vec = vectorizer.transform(X_test)

        feature_names = vectorizer.get_feature_names()
        print(f"Создано признаков: {len(feature_names)}")
        print(f"Размерность данных: Train={X_train_vec.shape}, Test={X_test_vec.shape}")

        # Обучение моделей
        print("\n7. ОБУЧЕНИЕ МОДЕЛЕЙ")
        print("-" * 40)

        classifier = FakeNewsClassifier(ModelConfig)

        # Обучение Logistic Regression
        print("\nОбучение Logistic Regression...")
        lr_model = classifier.train_logistic_regression(X_train_vec, y_train)

        # Обучение SVM
        print("\nОбучение Support Vector Machine (SVM)...")
        svm_model = classifier.train_svm(X_train_vec, y_train)

        # Оценка моделей
        print("\n8. ОЦЕНКА МОДЕЛЕЙ НА ТЕСТОВЫХ ДАННЫХ")
        print("-" * 40)

        all_metrics = {}
        feature_importance = {}

        models_to_evaluate = ['logistic_regression', 'svm']

        for model_name in models_to_evaluate:
            print(f"\n{'=' * 40}")
            print(f"МОДЕЛЬ: {model_name.upper()}")
            print(f"{'=' * 40}")

            # Получение предсказаний
            y_pred, y_pred_proba = classifier.predict(model_name, X_test_vec)

            # Оценка модели
            metrics = classifier.evaluate(model_name, X_test_vec, y_test)
            all_metrics[model_name] = metrics

            # Получение важности признаков
            importance = classifier.get_feature_importance(
                model_name, feature_names, top_n=15
            )
            feature_importance[model_name] = importance

            # Визуализация результатов
            if y_pred_proba is not None:
                # ROC кривая
                roc_file = os.path.join(OutputConfig.GRAPHS_DIR, f"roc_{model_name}.png")
                auc_score = Visualization.plot_roc_curve(
                    y_test, y_pred_proba, model_name, roc_file
                )
                all_metrics[model_name]['auc'] = float(auc_score)

                # Precision-Recall кривая
                pr_file = os.path.join(OutputConfig.GRAPHS_DIR, f"pr_{model_name}.png")
                ap_score = Visualization.plot_precision_recall_curve(
                    y_test, y_pred_proba, model_name, pr_file
                )
                all_metrics[model_name]['average_precision'] = float(ap_score)

            # Матрица ошибок
            cm_file = os.path.join(OutputConfig.GRAPHS_DIR, f"cm_{model_name}.png")
            Visualization.plot_confusion_matrix(
                np.array(metrics['confusion_matrix']),
                model_name,
                cm_file
            )

            # Важность признаков
            if importance:
                fi_file = os.path.join(OutputConfig.GRAPHS_DIR, f"fi_{model_name}.png")
                Visualization.plot_feature_importance(
                    importance, model_name, fi_file
                )

        # Проверка на переобучение
        print("\n9. ПРОВЕРКА НА ПЕРЕОБУЧЕНИЕ")
        print("-" * 40)

        # Проверка на тренировочных данных
        print("\n9.1. Результаты на тренировочных данных:")
        for model_name in models_to_evaluate:
            y_pred_train, _ = classifier.predict(model_name, X_train_vec)
            accuracy_train = accuracy_score(y_train, y_pred_train)
            print(f"  {model_name}: {accuracy_train:.4f}")

        # Проверка на кросс-валидации
        print("\n9.2. Кросс-валидация (5 фолдов):")
        for model_name in models_to_evaluate:
            scores = classifier.cross_validate(model_name, X_train_vec, y_train, cv=5)
            print(f"  {model_name}: Средний F1 = {scores.mean():.4f} (±{scores.std():.4f})")

        # Проверка на валидационной выборке (новая разбивка)
        print("\n9.3. Проверка на валидационной выборке (train-test split 80/20):")

        # Использование только тренировочных данных для новой проверки
        X_temp, X_val, y_temp, y_val = train_test_split(
            X_train_vec, y_train,
            test_size=0.2,
            random_state=42,
            stratify=y_train
        )

        print(f"  Размеры: Train={X_temp.shape}, Val={X_val.shape}")

        for model_name in models_to_evaluate:
            try:
                # Создание нового классификатора для этой проверки
                temp_classifier = FakeNewsClassifier(ModelConfig)

                if model_name == 'logistic_regression':
                    temp_model = temp_classifier.train_logistic_regression(X_temp, y_temp)
                elif model_name == 'svm':
                    temp_model = temp_classifier.train_svm(X_temp, y_temp)

                y_pred_val = temp_model.predict(X_val)
                accuracy_val = accuracy_score(y_val, y_pred_val)
                print(f"  {model_name}: {accuracy_val:.4f}")
            except Exception as e:
                print(f"  {model_name}: Ошибка - {str(e)}")

        # 10. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ
        print("\n10. ДОПОЛНИТЕЛЬНАЯ ВИЗУАЛИЗАЦИЯ")
        print("-" * 40)
        print("Графики сохранены в папке graphs/")
        print("  • roc_logistic_regression.png - ROC-кривая для Logistic Regression")
        print("  • roc_svm.png - ROC-кривая для SVM")
        print("  • cm_logistic_regression.png - Матрица ошибок для LR")
        print("  • cm_svm.png - Матрица ошибок для SVM")
        print("  • fi_logistic_regression.png - Важные слова для LR")
        print("  • fi_svm.png - Важные слова для SVM")

        # Сохранение результатов
        print("\n11. СОХРАНЕНИЕ РЕЗУЛЬТАТОВ")
        print("-" * 40)

        # Сохранение метрик
        ReportGenerator.save_metrics(all_metrics, OutputConfig.METRICS_FILE)

        # Сохранение важности признаков
        ReportGenerator.save_feature_importance_txt(
            feature_importance, OutputConfig.FEATURE_IMPORTANCE_FILE
        )

        # Сохранение полного отчета
        ReportGenerator.save_text_report(
            all_metrics, feature_importance, train_info,
            VectorizationConfig, ModelConfig, OutputConfig
        )

        # Сохранение моделей и векторизатора
        if LoggingConfig.SAVE_MODELS:
            # Векторизатор
            vec_file = os.path.join(OutputConfig.MODELS_DIR, "vectorizer.pkl")
            vectorizer.save_vectorizer(vec_file)

            # Модели
            for model_name in models_to_evaluate:
                model_file = os.path.join(OutputConfig.MODELS_DIR, f"{model_name}.pkl")
                classifier.save_model(model_name, model_file)

        # Итоговый отчет
        print("\n12. ИТОГИ ЭКСПЕРИМЕНТА")
        print("-" * 40)

        # Находим лучшую модель
        best_model = None
        best_f1 = 0

        for model_name, metrics in all_metrics.items():
            print(f"\n{model_name.upper()}:")
            print(f"  Accuracy:  {metrics['accuracy']:.4f}")
            print(f"  F1-Score:  {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            if 'auc' in metrics:
                print(f"  ROC-AUC:   {metrics['auc']:.4f}")

            if metrics['f1'] > best_f1:
                best_f1 = metrics['f1']
                best_model = model_name

        # Анализ самых важных слов
        print("\n" + "=" * 60)
        print("АНАЛИЗ ВАЖНЫХ СЛОВ (Топ-10 для каждой модели):")
        print("=" * 60)

        for model_name, features in feature_importance.items():
            if features:
                print(f"\n{model_name.upper()}:")
                for i, feat in enumerate(features[:10]):
                    print(f"  {i + 1:2d}. {feat['feature']:20} (важность: {feat['importance']:.4f})")

        print(f"\n{'=' * 60}")
        print(f"ЭКСПЕРИМЕНТ ЗАВЕРШЕН УСПЕШНО!")
        print(f"{'=' * 60}")

        print(f"\nЛУЧШАЯ МОДЕЛЬ: {best_model.upper()}")
        print(f"   F1-Score: {best_f1:.4f}")

        print(f"\nОБЩАЯ СТАТИСТИКА:")
        print(f"   Всего данных: {len(train_df_final)} записей")
        print(f"   Признаков: {len(feature_names)}")
        print(f"   Метод векторизации: {vec_method.upper()}")
        print(f"   Тестовая выборка: {len(X_test)} записей")

        if use_separate_test:
            print(f"   Использованы отдельные тестовые данные")
        else:
            print(f"   Использовано разделение тренировочных данных")

        print(f"\nРЕЗУЛЬТАТЫ СОХРАНЕНЫ В:")
        print(f"   {OutputConfig.BASE_DIR}/")

        print(f"\nДля анализа результатов:")
        print(f"   1. Изучить графики в {OutputConfig.GRAPHS_DIR}/")
        print(f"   2. Прочитать отчеты в {OutputConfig.REPORTS_DIR}/")

        print(f"\n{'=' * 60}")

    except KeyboardInterrupt:
        print("\n\nПрограмма прервана пользователем")
        return 130
    except Exception as e:
        print(f"\nОШИБКА: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    # Добавление пути к текущей директории
    sys.path.append(os.path.dirname(os.path.abspath(__file__)))

    exit_code = main()
    exit(exit_code)