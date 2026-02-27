"""
📥 Скачивание датасета для обучения klenskiy-GPT-1B

Датасет: Russian Wikipedia (полный текст, очищенный)
Источник: https://huggingface.co/datasets/atBuba/ruwiki-dataset

Формат: один .txt файл (~7.9 GB), статьи разделены <|endoftext|>

Использование:
    python download_data.py              # скачать датасет
    python download_data.py --check      # проверить что датасет на месте

После скачивания запусти токенизацию:
    python dataset.py                    # полный датасет
    python dataset.py --test             # тестовый (~10M токенов)
"""

import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).parent
DATA_DIR = REPO_ROOT / "data"
CORPUS_FILE = DATA_DIR / "ruwiki_full.txt"

HF_DATASET_REPO = "atBuba/ruwiki-dataset"


def check_dataset():
    """Проверяет наличие датасета."""
    if CORPUS_FILE.exists():
        size_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"✅ Датасет найден: {CORPUS_FILE}")
        print(f"   Размер: {size_gb:.2f} GB")
        return True
    else:
        print(f"❌ Датасет не найден: {CORPUS_FILE}")
        print(f"   Запусти: python download_data.py")
        return False


def download_dataset():
    """
    Скачивает датасет с HuggingFace.

    Использует huggingface_hub для скачивания файлов из репозитория.
    Датасет хранится как .txt файл — скачиваем его напрямую в data/.
    """
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if CORPUS_FILE.exists():
        size_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"✅ Датасет уже скачан: {CORPUS_FILE} ({size_gb:.2f} GB)")
        print(f"\n   Следующий шаг — токенизация:")
        print(f"   python dataset.py            # полный датасет")
        print(f"   python dataset.py --test     # тестовый (~10M токенов)")
        return

    # Проверяем наличие huggingface_hub
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError:
        print("❌ Библиотека huggingface_hub не установлена!")
        print("   Установи: pip install huggingface_hub")
        sys.exit(1)

    print(f"📥 Скачиваю датасет с HuggingFace...")
    print(f"   Репозиторий: https://huggingface.co/datasets/{HF_DATASET_REPO}")
    print(f"   Это может занять время (~8 GB)...\n")

    # Получаем список файлов в репозитории
    api = HfApi()
    try:
        files = api.list_repo_files(HF_DATASET_REPO, repo_type="dataset")
    except Exception as e:
        print(f"❌ Ошибка при доступе к репозиторию: {e}")
        print(f"   Проверь ссылку: https://huggingface.co/datasets/{HF_DATASET_REPO}")
        sys.exit(1)

    # Ищем .txt файл
    txt_files = [f for f in files if f.endswith(".txt")]

    if not txt_files:
        # Может быть файл в подпапке или другом формате — показываем что есть
        print(f"   Файлы в репозитории: {files}")
        print(f"\n⚠️  .txt файл не найден. Скачиваю все доступные файлы...")

        # Скачиваем всё и ищем текст
        for fname in files:
            if fname.startswith(".") or fname in ("README.md", ".gitattributes"):
                continue
            print(f"   Скачиваю: {fname}")
            downloaded = hf_hub_download(
                repo_id=HF_DATASET_REPO,
                filename=fname,
                repo_type="dataset",
                local_dir=str(DATA_DIR),
            )
            print(f"   → {downloaded}")

        # Проверяем, появился ли нужный файл
        _find_and_rename_corpus()
    else:
        # Скачиваем .txt файл напрямую
        target_file = txt_files[0]
        print(f"   Найден: {target_file}")
        print(f"   Скачиваю...\n")

        downloaded = hf_hub_download(
            repo_id=HF_DATASET_REPO,
            filename=target_file,
            repo_type="dataset",
            local_dir=str(DATA_DIR),
        )

        downloaded_path = Path(downloaded)

        # Переименовываем в ruwiki_full.txt если нужно
        if downloaded_path != CORPUS_FILE:
            if downloaded_path.exists():
                # hf_hub_download может сохранить в подпапку
                downloaded_path.rename(CORPUS_FILE)
                print(f"   Переименовано: {downloaded_path.name} → ruwiki_full.txt")
            else:
                # Файл мог быть сохранён по другому пути
                _find_and_rename_corpus()

    # Финальная проверка
    if CORPUS_FILE.exists():
        size_gb = CORPUS_FILE.stat().st_size / 1024**3
        print(f"\n✅ Датасет скачан: {CORPUS_FILE}")
        print(f"   Размер: {size_gb:.2f} GB")
        print(f"\n   Следующий шаг — токенизация:")
        print(f"   python dataset.py            # полный датасет")
        print(f"   python dataset.py --test     # тестовый (~10M токенов)")
    else:
        print(f"\n⚠️  Файл {CORPUS_FILE} не найден после скачивания.")
        print(f"   Проверь содержимое папки: {DATA_DIR}")
        _list_data_dir()


def _find_and_rename_corpus():
    """Ищет скачанный .txt файл и переименовывает в ruwiki_full.txt."""
    for f in DATA_DIR.rglob("*.txt"):
        if f.name == "ruwiki_full.txt":
            continue
        size_mb = f.stat().st_size / 1024**2
        if size_mb > 100:  # > 100MB — скорее всего это наш датасет
            print(f"   Найден текстовый файл: {f} ({size_mb:.0f} MB)")
            # Перемещаем в корень data/
            f.rename(CORPUS_FILE)
            print(f"   Переименовано → {CORPUS_FILE}")
            return True
    return False


def _list_data_dir():
    """Выводит содержимое data/ для дебага."""
    print(f"\n   Содержимое {DATA_DIR}:")
    if DATA_DIR.exists():
        for f in sorted(DATA_DIR.rglob("*")):
            if f.is_file():
                size_mb = f.stat().st_size / 1024**2
                rel = f.relative_to(DATA_DIR)
                print(f"     {rel} ({size_mb:.1f} MB)")
    else:
        print(f"     (папка не существует)")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check_dataset()
    else:
        print("=" * 60)
        print("📥 Скачивание датасета для klenskiy-GPT-1B")
        print(f"   Источник: https://huggingface.co/datasets/{HF_DATASET_REPO}")
        print("=" * 60 + "\n")
        download_dataset()
