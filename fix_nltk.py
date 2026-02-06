"""
Скрипт для скачивания всех необходимых ресурсов NLTK
"""
import nltk

print("Скачивание всех необходимых ресурсов NLTK...")

resources = [
    'punkt',           # Токенизатор предложений
    'punkt_tab',       # Дополнительные данные для токенизатора
    'stopwords',       # Стоп-слова
    'wordnet',         # Лемматизатор
    'averaged_perceptron_tagger',  # Для POS тегов
    'maxent_ne_chunker',           # Для NER
    'words'            # Словарь слов
]

for resource in resources:
    print(f"\nСкачивание {resource}...")
    try:
        nltk.download(resource, quiet=False)
        print(f"✓ {resource} скачан успешно")
    except Exception as e:
        print(f"⚠️ Ошибка при скачивании {resource}: {e}")

print("\n" + "="*60)
print("✅ ВСЕ РЕСУРСЫ NLTK СКАЧАНЫ!")
print("Теперь запустите main.py снова")
print("="*60)