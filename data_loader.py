"""
Загрузка данных из текстовых файлов для Fake News Dataset
"""
import os
import re
import pandas as pd
import numpy as np
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer, PorterStemmer
from nltk.tokenize import word_tokenize
from sklearn.utils import resample


class DataLoader:
    def __init__(self, config):
        self.config = config
        self.lemmatizer = WordNetLemmatizer() if config.USE_LEMMATIZATION else None
        self.stemmer = PorterStemmer() if config.USE_STEMMING else None

        # Скачивание ресурсов NLTK
        self._download_nltk_resources()

    def _download_nltk_resources(self):
        """Скачивание необходимых ресурсов NLTK."""
        resources = ['punkt', 'stopwords', 'wordnet']
        for resource in resources:
            try:
                nltk.data.find(f'tokenizers/punkt' if resource == 'punkt' else f'corpora/{resource}')
            except LookupError:
                print(f"Скачивание ресурса NLTK: {resource}")
                nltk.download(resource)

    def load_text_files(self, fake_paths, real_paths):
        """Загрузка текстовых файлов из указанных путей."""
        data = []

        # Загрузка фейковых новостей
        print("Загрузка фейковых новостей...")
        fake_count = 0
        for fake_path in fake_paths:
            if os.path.exists(fake_path):
                files_loaded = self._load_from_folder(fake_path, label=1, data=data)
                fake_count += files_loaded
                print(f"  Загружено из {os.path.basename(fake_path)}: {files_loaded} файлов")

        # Загрузка реальных новостей
        print("Загрузка реальных новостей...")
        real_count = 0
        for real_path in real_paths:
            if os.path.exists(real_path):
                files_loaded = self._load_from_folder(real_path, label=0, data=data)
                real_count += files_loaded
                print(f"  Загружено из {os.path.basename(real_path)}: {files_loaded} файлов")

        print(f"\nВсего загружено: {fake_count} фейковых, {real_count} реальных новостей")

        # Создание DataFrame
        df = pd.DataFrame(data, columns=['text', 'label'])

        return df

    def _load_from_folder(self, folder_path, label, data):
        """Загрузка всех текстовых файлов из папки."""
        files_loaded = 0

        if not os.path.exists(folder_path):
            return 0

        # Получение списка всех текстовых файлов
        txt_files = [f for f in os.listdir(folder_path) if f.endswith('.txt')]

        if not txt_files:
            return 0

        # Загрузка файлов с прогресс-баром
        for filename in tqdm(txt_files, desc=f"Загрузка {os.path.basename(folder_path)}", leave=False):
            try:
                file_path = os.path.join(folder_path, filename)
                with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                    text = f.read().strip()
                    if text:
                        data.append((text, label))
                        files_loaded += 1
            except Exception as e:
                print(f"Ошибка при чтении {filename}: {e}")

        return files_loaded

    def load_test_data(self, fake_path, real_path):
        """Загрузка тестовых данных."""
        data = []

        print("Загрузка тестовых данных...")

        # Загрузка фейковых новостей
        if os.path.exists(fake_path):
            fake_files = self._load_from_folder(fake_path, label=1, data=data)
            print(f"  Тестовых фейковых: {fake_files}")
        else:
            print(f"  Предупреждение: путь {fake_path} не существует")

        # Загрузка реальных новостей
        if os.path.exists(real_path):
            real_files = self._load_from_folder(real_path, label=0, data=data)
            print(f"  Тестовых реальных: {real_files}")
        else:
            print(f"  Предупреждение: путь {real_path} не существует")

        # Создание DataFrame
        df = pd.DataFrame(data, columns=['text', 'label'])

        # Ограничение количества тестовых данных
        if self.config.MAX_TEST_SAMPLES and len(df) > self.config.MAX_TEST_SAMPLES:
            print(f"Ограничение тестовых данных до {self.config.MAX_TEST_SAMPLES} записей")
            df = df.sample(n=self.config.MAX_TEST_SAMPLES, random_state=42)

        return df

    def clean_text(self, text):
        """Очистка текста."""
        if pd.isna(text):
            return ""

        text = str(text)

        # Приведение к нижнему регистру
        text = text.lower()

        # Удаление HTML тегов
        if self.config.REMOVE_HTML:
            text = re.sub(r'<.*?>', '', text)

        # Удаление URL
        if self.config.REMOVE_URLS:
            text = re.sub(r'https?://\S+|www\.\S+', '', text)

        # Удаление специальных символов
        if self.config.REMOVE_SPECIAL_CHARS:
            # Оставляем базовые знаки препинания
            text = re.sub(r'[^\w\s.,!?\'"-]', ' ', text)

        # Удаление чисел
        if self.config.REMOVE_NUMBERS:
            text = re.sub(r'\d+', '', text)

        # Удаление лишних пробелов
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def tokenize_and_process(self, text):
        """Токенизация и обработка текста."""
        if not text or pd.isna(text):
            return []

        # Токенизация
        tokens = word_tokenize(text)

        # Удаление стоп-слов
        if self.config.REMOVE_STOPWORDS:
            stop_words = set(stopwords.words('english'))
            if self.config.CUSTOM_STOPWORDS:
                stop_words.update(self.config.CUSTOM_STOPWORDS)
            tokens = [word for word in tokens if word not in stop_words]

        # Фильтрация по длине слов
        tokens = [word for word in tokens
                  if self.config.MIN_WORD_LENGTH <= len(word) <= self.config.MAX_WORD_LENGTH]

        # Лемматизация
        if self.config.USE_LEMMATIZATION and self.lemmatizer:
            tokens = [self.lemmatizer.lemmatize(word) for word in tokens]

        # Стемминг
        if self.config.USE_STEMMING and self.stemmer:
            tokens = [self.stemmer.stem(word) for word in tokens]

        return tokens

    def preprocess_dataframe(self, df):
        """Предобработка всего датафрейма."""
        print("\nПредобработка текстов...")

        # Ограничение количества выборок
        if self.config.MAX_SAMPLES and self.config.MAX_SAMPLES < len(df):
            print(f"Ограничение до {self.config.MAX_SAMPLES} выборок...")
            df = df.sample(n=self.config.MAX_SAMPLES, random_state=42)

        # Очистка текстов
        print("Очистка текстов...")
        tqdm.pandas(desc="Очистка")
        df['cleaned_text'] = df['text'].progress_apply(self.clean_text)

        # Токенизация
        print("Токенизация...")
        tqdm.pandas(desc="Токенизация")
        df['tokens'] = df['cleaned_text'].progress_apply(self.tokenize_and_process)

        # Соединение токенов обратно в текст для векторизации
        df['processed_text'] = df['tokens'].apply(lambda x: ' '.join(x))

        # Удаление пустых текстов
        initial_count = len(df)
        df = df[df['processed_text'].str.len() > 0]
        removed_count = initial_count - len(df)

        if removed_count > 0:
            print(f"Удалено {removed_count} пустых текстов")

        print(f"После предобработки: {len(df)} записей")

        return df

    def balance_classes(self, df):
        """Балансировка классов."""
        if 'label' not in df.columns:
            return df

        # Разделение по классам
        df_majority = df[df['label'] == 0]
        df_minority = df[df['label'] == 1]

        print(f"\nДо балансировки:")
        print(f"  Реальных новостей (0): {len(df_majority)}")
        print(f"  Фейковых новостей (1): {len(df_minority)}")

        if len(df_minority) == 0 or len(df_majority) == 0:
            print("Один из классов пуст. Балансировка невозможна.")
            return df

        # Определяем, какой класс больше
        if len(df_minority) < len(df_majority):
            print("Апсемплинг меньшего класса...")
            df_minority_upsampled = resample(df_minority,
                                             replace=True,
                                             n_samples=len(df_majority),
                                             random_state=42)
            df_balanced = pd.concat([df_majority, df_minority_upsampled])
        else:
            print("Даунсемплинг большего класса...")
            df_majority_downsampled = resample(df_majority,
                                               replace=False,
                                               n_samples=len(df_minority),
                                               random_state=42)
            df_balanced = pd.concat([df_majority_downsampled, df_minority])

        # Перемешиваем
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

        print(f"После балансировки:")
        print(f"  Реальных новостей (0): {len(df_balanced[df_balanced['label'] == 0])}")
        print(f"  Фейковых новостей (1): {len(df_balanced[df_balanced['label'] == 1])}")

        return df_balanced

    def analyze_dataset(self, df, name="Датасет"):
        """Анализ датасета."""
        print(f"\n{'=' * 60}")
        print(f"АНАЛИЗ: {name}")
        print(f"{'=' * 60}")

        info = {
            'total_samples': len(df),
            'class_distribution': {},
            'text_stats': {}
        }

        # Распределение классов
        if 'label' in df.columns:
            class_dist = df['label'].value_counts().to_dict()
            info['class_distribution'] = class_dist

            print(f"Общее количество: {len(df)}")
            print(f"Реальных новостей (0): {class_dist.get(0, 0)}")
            print(f"Фейковых новостей (1): {class_dist.get(1, 0)}")

            if len(df) > 0:
                real_percent = class_dist.get(0, 0) / len(df) * 100
                fake_percent = class_dist.get(1, 0) / len(df) * 100
                print(f"Баланс: {real_percent:.1f}% реальных, {fake_percent:.1f}% фейковых")

        # Статистика по текстам
        if 'text' in df.columns:
            df['text_length'] = df['text'].apply(lambda x: len(str(x)))
            info['text_stats'] = {
                'min_length': int(df['text_length'].min()),
                'max_length': int(df['text_length'].max()),
                'mean_length': float(df['text_length'].mean()),
                'median_length': float(df['text_length'].median())
            }

            print(f"\nСтатистика длины текстов:")
            print(f"  Минимальная: {info['text_stats']['min_length']} символов")
            print(f"  Максимальная: {info['text_stats']['max_length']} символов")
            print(f"  Средняя: {info['text_stats']['mean_length']:.1f} символов")
            print(f"  Медиана: {info['text_stats']['median_length']:.1f} символов")

        # Пример текста
        if len(df) > 0:
            print("\nПример текста (первые 200 символов):")
            sample_text = df.iloc[0]['text'][:200] if 'text' in df.columns else ""
            print(f"  '{sample_text}...'")

        return info

    def check_for_duplicates(self, train_df, test_df):
        """Проверка на дубликаты между train и test."""
        print("\nПроверка на дубликаты...")

        # Проверка уникальности текстов
        train_texts = set(train_df['text'].str[:100])  # Первые 100 символов
        test_texts = set(test_df['text'].str[:100])

        duplicates = train_texts.intersection(test_texts)

        if duplicates:
            print(f"Найдено {len(duplicates)} потенциальных дубликатов!")
            print("Примеры совпадений:")
            for i, text in enumerate(list(duplicates)[:3]):
                print(f"  {i + 1}. {text[:50]}...")
            return True
        else:
            print("Дубликаты не обнаружены")
            return False

    def remove_duplicates(self, train_df, test_df):
        """Удаление дубликатов между train и test с проверкой."""
        print("\nУдаление дубликатов между train и test...")

        if len(test_df) == 0:
            print("Тестовые данные пусты, пропускаем проверку дубликатов")
            return train_df, test_df

        # Создание уникальных идентификаторов
        train_df['text_hash'] = train_df['processed_text'].apply(
            lambda x: hash(str(x)) if pd.notna(x) else 0
        )
        test_df['text_hash'] = test_df['processed_text'].apply(
            lambda x: hash(str(x)) if pd.notna(x) else 0
        )

        # Находим дубликаты
        train_hashes = set(train_df['text_hash'])
        original_test_len = len(test_df)

        # Удаление из test_df записей, которые есть в train
        test_df_filtered = test_df[~test_df['text_hash'].isin(train_hashes)].copy()

        removed_count = original_test_len - len(test_df_filtered)
        print(f"Удалено {removed_count} дубликатов из тестовых данных")

        if len(test_df_filtered) == 0:
            print("❌ ВНИМАНИЕ: После удаления дубликатов тестовые данные пусты!")
            print("   Это означает, что тестовый набор полностью дублирует тренировочный")
            print("   Используем разделение тренировочных данных на train/test")

            # Удаление временных колонок
            train_df.drop('text_hash', axis=1, inplace=True)
            test_df.drop('text_hash', axis=1, inplace=True)

            return train_df, None

        # Удаление временных колонок
        train_df.drop('text_hash', axis=1, inplace=True)
        test_df_filtered.drop('text_hash', axis=1, inplace=True)

        print(f"Осталось тестовых данных: {len(test_df_filtered)} записей")

        return train_df, test_df_filtered