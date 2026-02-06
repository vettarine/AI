"""
Вспомогательные функции
"""
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import json
import os
from datetime import datetime
from sklearn.metrics import roc_curve, auc, precision_recall_curve
from wordcloud import WordCloud


class Visualization:
    @staticmethod
    def create_output_directories(config):
        """Создание папок для результатов."""
        os.makedirs(config.MODELS_DIR, exist_ok=True)
        os.makedirs(config.GRAPHS_DIR, exist_ok=True)
        os.makedirs(config.REPORTS_DIR, exist_ok=True)

        print(f"Созданы папки для результатов:")
        print(f"  • {config.MODELS_DIR}")
        print(f"  • {config.GRAPHS_DIR}")
        print(f"  • {config.REPORTS_DIR}")

    @staticmethod
    def plot_roc_curve(y_test, y_pred_proba, model_name, filepath):
        """Построение ROC кривой."""
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)

        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, color='darkorange', lw=2,
                 label=f'ROC curve (AUC = {roc_auc:.4f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc="lower right")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"ROC кривая сохранена: {filepath}")
        return roc_auc

    @staticmethod
    def plot_confusion_matrix(cm, model_name, filepath):
        """Построение матрицы ошибок."""
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Real', 'Fake'],
                    yticklabels=['Real', 'Fake'])
        plt.title(f'Confusion Matrix - {model_name}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Матрица ошибок сохранена: {filepath}")

    @staticmethod
    def plot_feature_importance(features, model_name, filepath):
        """Построение важности признаков."""
        if not features:
            return

        df = pd.DataFrame(features)

        plt.figure(figsize=(10, 6))
        bars = plt.barh(range(len(df)), df['importance'],
                        color='steelblue', edgecolor='black')
        plt.yticks(range(len(df)), df['feature'], fontsize=9)
        plt.xlabel('Importance')
        plt.title(f'Top Features - {model_name}')

        # Добавление значений на столбцы
        for i, (bar, val) in enumerate(zip(bars, df['importance'])):
            plt.text(val + 0.001, bar.get_y() + bar.get_height() / 2,
                     f'{val:.4f}', va='center', fontsize=8)

        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Важность признаков сохранена: {filepath}")

    @staticmethod
    def plot_precision_recall_curve(y_test, y_pred_proba, model_name, filepath):
        """Построение Precision-Recall кривой."""
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        average_precision = auc(recall, precision)

        plt.figure(figsize=(8, 6))
        plt.plot(recall, precision, color='darkgreen', lw=2,
                 label=f'PR curve (AP = {average_precision:.4f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve - {model_name}')
        plt.legend(loc="lower left")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Precision-Recall кривая сохранена: {filepath}")
        return average_precision

    @staticmethod
    def create_word_cloud(texts, title, filepath):
        """Создание облака слов."""
        if not texts:
            return

        # Объединение всех текстов
        all_text = ' '.join(texts)

        # Создание облака слов
        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color='white',
            max_words=50,
            contour_width=2,
            contour_color='steelblue'
        ).generate(all_text)

        plt.figure(figsize=(10, 5))
        plt.imshow(wordcloud, interpolation='bilinear')
        plt.axis('off')
        plt.title(title)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()

        print(f"Облако слов сохранено: {filepath}")


class ReportGenerator:
    @staticmethod
    def save_metrics(metrics, filepath):
        """Сохранение метрик в JSON."""
        with open(filepath, 'w') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        print(f"Метрики сохранены: {filepath}")

    @staticmethod
    def save_feature_importance_txt(feature_importance, filepath):
        """Сохранение важности признаков в текстовый файл."""
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("ВАЖНОСТЬ ПРИЗНАКОВ (СЛОВ)\n")
            f.write("=" * 60 + "\n\n")

            for model_name, features in feature_importance.items():
                if features:
                    f.write(f"{model_name.upper()}:\n")
                    f.write("-" * 40 + "\n")
                    for i, feat in enumerate(features, 1):
                        f.write(f"{i:2d}. {feat['feature']:20} {feat['importance']:.6f}\n")
                    f.write("\n")

        print(f"Важность признаков сохранена: {filepath}")

    @staticmethod
    def save_text_report(metrics, feature_importance, dataset_info,
                         vec_config, model_config, output_config):
        """Сохранение текстового отчета."""
        filepath = output_config.CLASSIFICATION_REPORT_FILE

        with open(filepath, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("ОТЧЕТ: КЛАССИФИКАЦИЯ ФЕЙКОВЫХ НОВОСТЕЙ\n")
            f.write(f"Дата: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write("=" * 80 + "\n\n")

            f.write("1. ИНФОРМАЦИЯ О ДАТАСЕТЕ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Всего записей: {dataset_info['total_samples']}\n")
            if dataset_info['class_distribution']:
                f.write(f"Реальных новостей: {dataset_info['class_distribution'].get(0, 0)}\n")
                f.write(f"Фейковых новостей: {dataset_info['class_distribution'].get(1, 0)}\n")

            f.write("\n2. КОНФИГУРАЦИЯ\n")
            f.write("-" * 40 + "\n")
            f.write(f"Метод векторизации: {vec_config.VECTORIZATION_METHOD}\n")
            f.write(
                f"Количество признаков: {vec_config.TFIDF_MAX_FEATURES if vec_config.VECTORIZATION_METHOD == 'tfidf' else vec_config.BOW_MAX_FEATURES}\n")
            f.write(f"Тестовая выборка: {model_config.TEST_SIZE * 100}%\n")

            f.write("\n3. РЕЗУЛЬТАТЫ МОДЕЛЕЙ\n")
            f.write("-" * 40 + "\n")

            best_model = None
            best_f1 = 0

            for model_name, model_metrics in metrics.items():
                f.write(f"\n{model_name.upper()}:\n")
                f.write(f"  Accuracy:  {model_metrics['accuracy']:.4f}\n")
                f.write(f"  Precision: {model_metrics['precision']:.4f}\n")
                f.write(f"  Recall:    {model_metrics['recall']:.4f}\n")
                f.write(f"  F1-Score:  {model_metrics['f1']:.4f}\n")

                if 'auc' in model_metrics:
                    f.write(f"  ROC-AUC:   {model_metrics['auc']:.4f}\n")

                # Матрица ошибок
                cm = model_metrics['confusion_matrix']
                f.write(f"  Матрица ошибок:\n")
                f.write(f"    [[{cm[0][0]:4d}  {cm[0][1]:4d}]\n")
                f.write(f"     [{cm[1][0]:4d}  {cm[1][1]:4d}]]\n")

                if model_metrics['f1'] > best_f1:
                    best_f1 = model_metrics['f1']
                    best_model = model_name

            f.write("\n4. ВАЖНЫЕ ПРИЗНАКИ\n")
            f.write("-" * 40 + "\n")

            for model_name, features in feature_importance.items():
                if features:
                    f.write(f"\n{model_name.upper()} (Топ-10):\n")
                    for i, feat in enumerate(features[:10], 1):
                        f.write(f"  {i:2d}. {feat['feature']:20} - {feat['importance']:.6f}\n")

            f.write("\n5. ВЫВОДЫ\n")
            f.write("-" * 40 + "\n")

            if best_model:
                f.write(f"Лучшая модель: {best_model.upper()}\n")
                f.write(f"Лучший F1-Score: {best_f1:.4f}\n")

            f.write("\n" + "=" * 80 + "\n")
            f.write("ОТЧЕТ СФОРМИРОВАН АВТОМАТИЧЕСКИ\n")
            f.write("=" * 80 + "\n")

        print(f"Текстовый отчет сохранен: {filepath}")