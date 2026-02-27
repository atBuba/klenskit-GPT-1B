# klenskiy-GPT-1B

Русскоязычная языковая модель на ~1B параметров.
Архитектура: Llama-style (RoPE + GQA + SwiGLU + RMSNorm).

## Требования

- GPU с 48+ GB VRAM (Blackwell Pro 6000, A100, H100)
- 50+ GB дискового пространства (данные + чекпоинты)
- Python 3.10+

```bash
pip install torch sentencepiece wikiextractor numpy
# Опционально:
pip install wandb  # для мониторинга обучения
```

## Архитектура

| Параметр | Значение |
|----------|----------|
| Hidden dim | 2048 |
| Layers | 22 |
| Q-heads | 16 |
| KV-heads | 4 (GQA) |
| Context | 2048 токенов |
| Vocab | 32,000 (BPE) |
| FFN | SwiGLU (5,632) |
| Norm | RMSNorm |
| Positions | RoPE |
| **Total params** | **~1B** |

## Запуск (5 шагов)

### 1. Скачать данные (русская Wikipedia)
```bash
python download_data.py
```
Скачает ~4.5GB сжатого дампа, извлечёт ~3-4GB текста.
Дополнительные тексты можно положить в `data/custom/`.

### 2. Обучить токенизатор
```bash
python tokenizer.py data/corpus.txt 32000
```
Обучит BPE-токенизатор на 32K токенов (5-10 минут).

### 3. Подготовить датасет
```bash
python dataset.py data/corpus.txt tokenizer_model/klenskiy_bpe.model
```
Токенизирует весь корпус и сохранит как `data/train.bin` и `data/val.bin`.

### 4. Обучить модель
```bash
python train.py
```
На Blackwell Pro 6000 обучение займёт ~24-48 часов (100K шагов).
Чекпоинты сохраняются каждые 5000 шагов.

### 5. Генерация
```bash
python generate.py --prompt "Однажды в далёком" --length 500
python generate.py --interactive  # интерактивный режим
```

## Мониторинг (Wandb)

Включи `USE_WANDB = True` в `train.py` и:
```bash
wandb login
python train.py
```

## Структура

```
klenskiy-gpt-1b/
├── README.md
├── download_data.py    # Скачивание русской Wikipedia
├── tokenizer.py        # BPE-токенизатор (SentencePiece)
├── dataset.py          # Memory-mapped датасет
├── model.py            # Архитектура (RoPE + GQA + SwiGLU + RMSNorm)
├── train.py            # Тренировочный цикл (bf16, cosine LR, grad clip)
├── generate.py         # Генерация текста
├── data/
│   ├── raw/            # Сырые данные (Wikipedia dump)
│   ├── custom/         # Дополнительные тексты
│   ├── corpus.txt      # Собранный корпус
│   ├── train.bin       # Токенизированные данные (train)
│   └── val.bin         # Токенизированные данные (val)
├── tokenizer_model/
│   ├── klenskiy_bpe.model
│   └── klenskiy_bpe.vocab
└── checkpoints/
    ├── klenskiy-gpt-1b-best.pt
    ├── klenskiy-gpt-1b-step_5000.pt
    └── ...
```
