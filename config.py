"""
Конфигурация для датасета Fake News
"""
import os

# ============================================
# ПУТИ К ДАННЫМ
# ============================================
class DataConfig:
    # Основной путь к датасету
    DATASET_PATH = r"C:\Users\Ulyana Seredenko\.cache\kagglehub\datasets\sumanthvrao\fakenewsdataset\versions\3"

    # Тренировочные данные
    TRAIN_FAKE_PATHS = [
        os.path.join(DATASET_PATH, "training", "training", "celebrityDataset", "fake"),
        os.path.join(DATASET_PATH, "training", "training", "fakeNewsDataset", "fake"),
        os.path.join(DATASET_PATH, "overall", "overall", "celebrityDataset", "fake"),
        os.path.join(DATASET_PATH, "overall", "overall", "fake"),
    ]

    TRAIN_REAL_PATHS = [
        os.path.join(DATASET_PATH, "training", "training", "celebrityDataset", "legit"),
        os.path.join(DATASET_PATH, "training", "training", "fakeNewsDataset", "legit"),
        os.path.join(DATASET_PATH, "overall", "overall", "celebrityDataset", "legit"),
        os.path.join(DATASET_PATH, "overall", "overall", "real"),
    ]

    # Тестовые данные
    TEST_FAKE_PATH = os.path.join(DATASET_PATH, "Testing_dataset", "testingSet", "fake")
    TEST_REAL_PATH = os.path.join(DATASET_PATH, "Testing_dataset", "testingSet", "real")

    # CSV файлы
    FAKE_CSV = os.path.join(DATASET_PATH, "Testing_dataset", "testingSet", "Catalog - Fake Articles.csv")
    REAL_CSV = os.path.join(DATASET_PATH, "Testing_dataset", "testingSet", "Catalog - Real Articles.csv")

# ============================================
# НАСТРОЙКИ ПРЕДОБРАБОТКИ
# ============================================
class PreprocessingConfig:
    # Очистка текста
    REMOVE_HTML = True
    REMOVE_URLS = True
    REMOVE_SPECIAL_CHARS = True
    REMOVE_NUMBERS = False  # Можно оставить числа, иногда они важны

    # Обработка текста
    USE_LEMMATIZATION = True
    USE_STEMMING = False

    # Стоп-слова
    REMOVE_STOPWORDS = True
    CUSTOM_STOPWORDS = ['said', 'would', 'could', 'also', 'like', 'one', 'two', 'three']

    # Длина слов
    MIN_WORD_LENGTH = 2
    MAX_WORD_LENGTH = 25

    # Максимальное количество документов
    MAX_SAMPLES = None
    MAX_TEST_SAMPLES = 100  # Ограничение тестовых данных для быстрого тестирования

# ============================================
# НАСТРОЙКИ ВЕКТОРИЗАЦИИ
# ============================================
class VectorizationConfig:
    # Выбор метода векторизации
    VECTORIZATION_METHOD = 'tfidf'  # 'tfidf' или 'bow'

    # TF-IDF параметры
    TFIDF_MAX_FEATURES = 5000
    TFIDF_MAX_DF = 0.95
    TFIDF_MIN_DF = 2
    TFIDF_NGRAM_RANGE = (1, 2)  # Униграммы и биграммы
    TFIDF_USE_IDF = True
    TFIDF_SUBlinear_TF = True

    # Bag-of-Words параметры
    BOW_MAX_FEATURES = 5000
    BOW_MAX_DF = 0.95
    BOW_MIN_DF = 2
    BOW_NGRAM_RANGE = (1, 2)

# ============================================
# НАСТРОЙКИ МОДЕЛЕЙ
# ============================================
class ModelConfig:
    # Logistic Regression
    LR_C = 0.1
    LR_PENALTY = 'l2'
    LR_SOLVER = 'liblinear'
    LR_MAX_ITER = 1000
    LR_CLASS_WEIGHT = 'balanced'

    # SVM
    SVM_C = 0.1
    SVM_KERNEL = 'linear'
    SVM_GAMMA = 'scale'
    SVM_CLASS_WEIGHT = 'balanced'
    SVM_PROBABILITY = True

    # Общие параметры
    RANDOM_STATE = 42
    TEST_SIZE = 0.3
    CV_FOLDS = 3

    # GridSearch параметры
    USE_GRID_SEARCH = False
    GRID_SEARCH_CV = 2

# ============================================
# НАСТРОЙКИ ВЫВОДА
# ============================================
class OutputConfig:
    BASE_DIR = 'results_fake_news_text'
    MODELS_DIR = os.path.join(BASE_DIR, 'models')
    GRAPHS_DIR = os.path.join(BASE_DIR, 'graphs')
    REPORTS_DIR = os.path.join(BASE_DIR, 'reports')

    # Файлы результатов
    METRICS_FILE = os.path.join(REPORTS_DIR, 'model_metrics.json')
    FEATURE_IMPORTANCE_FILE = os.path.join(REPORTS_DIR, 'feature_importance.txt')
    CLASSIFICATION_REPORT_FILE = os.path.join(REPORTS_DIR, 'classification_report.txt')
    DATASET_INFO_FILE = os.path.join(REPORTS_DIR, 'dataset_info.txt')

    # Графики
    GRAPH_NAMES = {
        'roc_curve': 'roc_curve.png',
        'confusion_matrix': 'confusion_matrix.png',
        'feature_importance': 'feature_importance.png',
        'precision_recall_curve': 'precision_recall_curve.png',
        'word_cloud_real': 'word_cloud_real.png',
        'word_cloud_fake': 'word_cloud_fake.png',
        'class_distribution': 'class_distribution.png',
        'text_length_distribution': 'text_length_distribution.png'
    }

class LoggingConfig:
    VERBOSE = True
    PRINT_PROGRESS = True
    SAVE_MODELS = True
    SHOW_PLOTS = False