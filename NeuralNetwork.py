import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import json
import gc

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    # Системные параметры
    system = "colab"
    
    if system == "my":
        dir = "/media/alex/Programs/NEURAL_NETWORKS/"
    else:
        dir = "/content/"

    # Пути
    data_path = dir + "NeuralNetwork-GPT/DataSet.txt"
    model_path = dir + "NeuralNetwork-GPT/Model/text_model.pth"
    vocab_path = dir + "NeuralNetwork-GPT/Model/vocab.json"

    # Параметры модели
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    max_seq_len = 256
    dropout = 0.5

    # Параметры обучения
    batch_size = 32
    lr = 0.0001
    epochs = 100
    accumulation_steps = 4
    early_stop_patience = 3
    min_loss_delta = 0.001
    mixed_precision = True
    factor_lr = 0.5                 # Коэффициент уменьшения learning rate при plateau
    patience_lr = 0                 # Количество эпох без улучшения для снижения learning rate

config = Config()

# ====================== ОБРАБОТКА ДАННЫХ ======================
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        self.unk_token = vocab["<unk>"]
        self.pad_token = vocab["<pad>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()[:self.seq_len]
        n_tokens = len(tokens)

        # Конвертация в индексы
        input_ids = [self.vocab.get(token, self.unk_token) for token in tokens]

        # Добавление паддинга
        if n_tokens < self.seq_len:
            pad_len = self.seq_len - n_tokens
            input_ids += [self.pad_token] * pad_len
            attention_mask = [1] * n_tokens + [0] * pad_len
        else:
            attention_mask = [1] * self.seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
        }

def build_vocab(texts, min_freq=3):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<sos>": 2,
        "<eos>": 3
    }

    idx = 4
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def load_data(val_split=0.1):
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"❌ Файл с данными не найден: {config.data_path}")

    with open(config.data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if len(texts) == 0:
        raise ValueError(f"❌ Файл с данными пуст: {config.data_path}")

    # Разделение на train/val
    val_size = int(len(texts) * val_split)
    train_size = len(texts) - val_size
    train_texts, val_texts = random_split(texts, [train_size, val_size])

    return list(train_texts), list(val_texts)

# ====================== МОДЕЛЬ ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)

# ====================== ОБУЧЕНИЕ ======================
def train_epoch(model, loader, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader)):
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        # Создаем таргеты (сдвинутые на один токен)
        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1]
        mask = mask[:, :-1]

        autocast_context = torch.amp.autocast(
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.float16,
            enabled=scaler is not None and config.mixed_precision
        )

        with autocast_context:
            outputs = model(inputs, src_key_padding_mask=~mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = F.cross_entropy(
                outputs,
                targets,
                ignore_index=0
            ) / config.accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % config.accumulation_steps == 0 or (i + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.accumulation_steps

    return total_loss / len(loader)

@torch.inference_mode()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader):
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1]
        mask = mask[:, :-1]

        autocast_context = torch.amp.autocast(
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.float16,
            enabled=scaler is not None and config.mixed_precision
        )

        with autocast_context:
            outputs = model(inputs, src_key_padding_mask=~mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = F.cross_entropy(
                outputs,
                targets,
                ignore_index=0
            )

        total_loss += loss.item()

    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': len(vocab),
        'd_model': config.d_model,
        'nhead': config.nhead,
        'num_layers': config.num_layers,
        'dim_feedforward': config.dim_feedforward,
        'dropout': config.dropout
    }, path)

def run_training(resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    try:
        # Загрузка данных
        train_texts, val_texts = load_data()
        print(f"Загружено {len(train_texts)} обучающих фраз, {len(val_texts)} валидационных")

        # Построение словаря
        all_texts = train_texts + val_texts  # Оптимизировано
        vocab = build_vocab(all_texts)
        print(f"Создан словарь из {len(vocab)} токенов")

        # Сохранение словаря
        with open(config.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

        # Создание даталоадеров
        train_dataset = TextDataset(train_texts, vocab, config.max_seq_len)
        val_dataset = TextDataset(val_texts, vocab, config.max_seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True  # Оптимизация
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(2, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True  # Оптимизация
        )

        # Инициализация модели
        model = TextTransformer(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(device)

        # Настройки оптимизации
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.factor_lr,
            patience=config.patience_lr,
            verbose=True
        )

        scaler = None
        if config.mixed_precision and device.type == 'cuda':
            scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())

        # Инициализация переменных обучения
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0

        # Загрузка чекпоинта
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Загружен чекпоинт эпохи {checkpoint['epoch']}, "
                  f"Train loss: {checkpoint['train_loss']:.4f}, "
                  f"Val loss: {checkpoint['val_loss']:.4f}")

        # Обучение
        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.time()

            # Очистка памяти
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            # Тренировочная эпоха
            train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, scaler)

            # Валидация
            val_loss = evaluate(model, val_loader, device, scaler)
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']

            print(f"Эпоха {epoch+1}/{config.epochs} | "
                  f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                  f"Время: {epoch_time:.2f}с | LR: {lr:.6f}")

            # Проверка улучшения и сохранение лучшей модели
            if val_loss < best_val_loss - config.min_loss_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), config.model_path)
                print(f"🔥 Лучшая модель сохранена: {config.model_path}")

            # Сохранение чекпоинта каждый эпох
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab, config.model_path)
            print(f"💾 Чекпоинт сохранен: {config.model_path}")

            # Проверка ранней остановки
            if val_loss >= best_val_loss - config.min_loss_delta:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    print(f"🏁 Ранняя остановка на эпохе {epoch+1}")
                    break

        print(f"Обучение завершено! Лучшая Val loss: {best_val_loss:.4f}")

    except Exception as e:
        print(f"❌ Ошибка при обучении: {str(e)}")
        import traceback
        traceback.print_exc()

# ====================== ГЕНЕРАЦИЯ ======================
@torch.inference_mode()
def generate_text(model, vocab, prompt, device, max_length=50, temperature=0.7, top_k=50, stop_tokens=None):
    model.eval()
    rev_vocab = {idx: word for word, idx in vocab.items()}

    # Токенизация промпта
    tokens = prompt.split()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    # Определение стоп-токенов (включая <unk>)
    stop_tokens = stop_tokens or {"<eos>"}
    stop_tokens = stop_tokens | {"<unk>"}  # Объединение множеств
    stop_ids = {vocab[token] for token in stop_tokens if token in vocab}

    # Генерация
    for _ in range(max_length):
        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        mask = torch.ones_like(inputs, dtype=torch.bool)

        with torch.amp.autocast(device_type=device.type, enabled=True):
            outputs = model(inputs, src_key_padding_mask=~mask)
            logits = outputs[0, -1, :] / temperature

            # Фильтрация top-k
            if top_k > 0:
                top_vals, top_idxs = torch.topk(logits, top_k)
                logits[logits < top_vals[-1]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # Прерывание при появлении стоп-токена (включая <unk>)
            if next_token in stop_ids:
                break

            input_ids.append(next_token)
            tokens.append(rev_vocab.get(next_token, "<unk>"))

    return " ".join(tokens)

# ====================== ИНТЕРФЕЙС ======================
def interactive_mode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Загрузка словаря
        if not os.path.exists(config.vocab_path):
            print("❌ Словарь не найден!")
            return

        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # Загрузка модели
        if not os.path.exists(config.model_path):
            print("❌ Модель не найдена!")
            return

        # Загрузка конфигурации модели из чекпоинта (если доступно)
        model_path = config.model_path if os.path.exists(config.model_path) else config.model_path
        checkpoint = torch.load(model_path, map_location=device)

        model = TextTransformer(
            vocab_size=checkpoint.get('vocab_size', len(vocab)),
            d_model=checkpoint.get('d_model', config.d_model),
            nhead=checkpoint.get('nhead', config.nhead),
            num_layers=checkpoint.get('num_layers', config.num_layers),
            dim_feedforward=checkpoint.get('dim_feedforward', config.dim_feedforward),
            dropout=checkpoint.get('dropout', config.dropout)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("Модель загружена. Введите текст для генерации (или 'exit' для выхода)")
        print("Параметры генерации: temperature=0.7, top_k=50")

        while True:
            try:
                prompt = input("\nВаше сообщение: ").strip()
                if not prompt:
                    print("⚠️ Введите текст!")
                    continue
                if prompt.lower() == 'exit':
                    break

                start_time = time.time()
                generated = generate_text(model, vocab, prompt, device)
                gen_time = time.time() - start_time

                print(f"\nСеть: {generated}")
                print(f"Время генерации: {gen_time:.2f}с")

            except KeyboardInterrupt:
                print("\nВыход...")
                break

    except Exception as e:
        print(f"❌ Ошибка в интерактивном режиме: {str(e)}")

def continue_training():
    run_training(resume_checkpoint=config.model_path)

def main_menu():
    while True:
        print("\nМеню текстовой модели:")
        print("1. Обучить модель с нуля")
        print("2. Продолжить обучение с чекпоинта")
        print("3. Интерактивный режим")
        print("4. Выход")
        choice = input("Выбор: ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            continue_training()
        elif choice == '3':
            interactive_mode()
        elif choice == '4':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()"

    if system == "my":
        dir = "/media/alex/Programs/NEURAL_NETWORKS/"
    else:
        dir = "/content/"

    # Пути
    data_path = dir + "NeuralNetwork-GPT/DataSet.txt"
    model_path = dir + "NeuralNetwork-GPT/Model/text_model.pth"
    vocab_path = dir + "NeuralNetwork-GPT/Model/vocab.json"

    # Параметры модели
    d_model = 512
    nhead = 8
    num_layers = 6
    dim_feedforward = 2048
    max_seq_len = 256
    dropout = 0.5

    # Параметры обучения
    batch_size = 32
    lr = 0.0001
    epochs = 100
    accumulation_steps = 4
    early_stop_patience = 3
    min_loss_delta = 0.001
    mixed_precision = True
    factor_lr = 0.5                 # Коэффициент уменьшения learning rate при plateau
    patience_lr = 0                 # Количество эпох без улучшения для снижения learning rate

config = Config()

# ====================== ОБРАБОТКА ДАННЫХ ======================
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        self.unk_token = vocab["<unk>"]
        self.pad_token = vocab["<pad>"]

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()[:self.seq_len]
        n_tokens = len(tokens)

        # Конвертация в индексы
        input_ids = [self.vocab.get(token, self.unk_token) for token in tokens]

        # Добавление паддинга
        if n_tokens < self.seq_len:
            pad_len = self.seq_len - n_tokens
            input_ids += [self.pad_token] * pad_len
            attention_mask = [1] * n_tokens + [0] * pad_len
        else:
            attention_mask = [1] * self.seq_len

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.bool)
        }

def build_vocab(texts, min_freq=3):
    counter = Counter()
    for text in texts:
        counter.update(text.split())

    vocab = {
        "<pad>": 0,
        "<unk>": 1,
        "<sos>": 2,
        "<eos>": 3
    }

    idx = 4
    for word, count in counter.items():
        if count >= min_freq:
            vocab[word] = idx
            idx += 1

    return vocab

def load_data(val_split=0.1):
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"❌ Файл с данными не найден: {config.data_path}")

    with open(config.data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]

    if len(texts) == 0:
        raise ValueError(f"❌ Файл с данными пуст: {config.data_path}")

    # Разделение на train/val
    val_size = int(len(texts) * val_split)
    train_size = len(texts) - val_size
    train_texts, val_texts = random_split(texts, [train_size, val_size])

    return list(train_texts), list(val_texts)

# ====================== МОДЕЛЬ ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1), :].squeeze(1)
        return self.dropout(x)

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout, config.max_seq_len)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, src, src_key_padding_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        output = self.transformer(src, src_key_padding_mask=src_key_padding_mask)
        return self.fc_out(output)

# ====================== ОБУЧЕНИЕ ======================
def train_epoch(model, loader, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0.0
    optimizer.zero_grad()

    for i, batch in enumerate(tqdm(loader)):
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        # Создаем таргеты (сдвинутые на один токен)
        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1]
        mask = mask[:, :-1]

        autocast_context = torch.amp.autocast(
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.float16,
            enabled=scaler is not None and config.mixed_precision
        )

        with autocast_context:
            outputs = model(inputs, src_key_padding_mask=~mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = F.cross_entropy(
                outputs,
                targets,
                ignore_index=0
            ) / config.accumulation_steps

        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()

        if (i + 1) % config.accumulation_steps == 0 or (i + 1) == len(loader):
            if scaler:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()

            optimizer.zero_grad(set_to_none=True)

        total_loss += loss.item() * config.accumulation_steps

    return total_loss / len(loader)

@torch.inference_mode()
def evaluate(model, loader, device, scaler=None):
    model.eval()
    total_loss = 0.0

    for batch in tqdm(loader):
        inputs = batch["input_ids"].to(device, non_blocking=True)
        mask = batch["attention_mask"].to(device, non_blocking=True)

        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1]
        mask = mask[:, :-1]

        autocast_context = torch.amp.autocast(
            device_type='cuda' if device.type == 'cuda' else 'cpu',
            dtype=torch.float16,
            enabled=scaler is not None and config.mixed_precision
        )

        with autocast_context:
            outputs = model(inputs, src_key_padding_mask=~mask)
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)

            loss = F.cross_entropy(
                outputs,
                targets,
                ignore_index=0
            )

        total_loss += loss.item()

    return total_loss / len(loader)

def save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab, path):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
        'vocab_size': len(vocab),
        'd_model': config.d_model,
        'nhead': config.nhead,
        'num_layers': config.num_layers,
        'dim_feedforward': config.dim_feedforward,
        'dropout': config.dropout
    }, path)

def run_training(resume_checkpoint=None):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)

    try:
        # Загрузка данных
        train_texts, val_texts = load_data()
        print(f"Загружено {len(train_texts)} обучающих фраз, {len(val_texts)} валидационных")

        # Построение словаря
        all_texts = train_texts + val_texts  # Оптимизировано
        vocab = build_vocab(all_texts)
        print(f"Создан словарь из {len(vocab)} токенов")

        # Сохранение словаря
        with open(config.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)

        # Создание даталоадеров
        train_dataset = TextDataset(train_texts, vocab, config.max_seq_len)
        val_dataset = TextDataset(val_texts, vocab, config.max_seq_len)

        train_loader = DataLoader(
            train_dataset,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True  # Оптимизация
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=min(2, os.cpu_count()),
            pin_memory=True,
            persistent_workers=True  # Оптимизация
        )

        # Инициализация модели
        model = TextTransformer(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(device)

        # Настройки оптимизации
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=config.factor_lr,
            patience=config.patience_lr,
            verbose=True
        )

        scaler = None
        if config.mixed_precision and device.type == 'cuda':
            scaler = torch.amp.GradScaler('cuda', enabled=config.mixed_precision and torch.cuda.is_available())

        # Инициализация переменных обучения
        start_epoch = 0
        best_val_loss = float('inf')
        patience_counter = 0

        # Загрузка чекпоинта
        if resume_checkpoint and os.path.exists(resume_checkpoint):
            checkpoint = torch.load(resume_checkpoint, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_val_loss = checkpoint['val_loss']
            print(f"Загружен чекпоинт эпохи {checkpoint['epoch']}, "
                  f"Train loss: {checkpoint['train_loss']:.4f}, "
                  f"Val loss: {checkpoint['val_loss']:.4f}")

        # Обучение
        for epoch in range(start_epoch, config.epochs):
            epoch_start = time.time()

            # Очистка памяти
            if device.type == 'cuda':
                torch.cuda.empty_cache()
            gc.collect()

            # Тренировочная эпоха
            train_loss = train_epoch(model, train_loader, optimizer, device, scheduler, scaler)

            # Валидация
            val_loss = evaluate(model, val_loader, device, scaler)
            scheduler.step(val_loss)

            epoch_time = time.time() - epoch_start
            lr = optimizer.param_groups[0]['lr']

            print(f"Эпоха {epoch+1}/{config.epochs} | "
                  f"Train loss: {train_loss:.4f} | Val loss: {val_loss:.4f} | "
                  f"Время: {epoch_time:.2f}с | LR: {lr:.6f}")

            # Проверка улучшения и сохранение лучшей модели
            if val_loss < best_val_loss - config.min_loss_delta:
                best_val_loss = val_loss
                patience_counter = 0
                torch.save(model.state_dict(), config.model_path)
                print(f"🔥 Лучшая модель сохранена: {config.model_path}")

            # Сохранение чекпоинта каждый эпох
            save_checkpoint(model, optimizer, epoch, train_loss, val_loss, vocab, config.model_path)
            print(f"💾 Чекпоинт сохранен: {config.model_path}")

            # Проверка ранней остановки
            if val_loss >= best_val_loss - config.min_loss_delta:
                patience_counter += 1
                if patience_counter >= config.early_stop_patience:
                    print(f"🏁 Ранняя остановка на эпохе {epoch+1}")
                    break

        print(f"Обучение завершено! Лучшая Val loss: {best_val_loss:.4f}")

    except Exception as e:
        print(f"❌ Ошибка при обучении: {str(e)}")
        import traceback
        traceback.print_exc()

# ====================== ГЕНЕРАЦИЯ ======================
@torch.inference_mode()
def generate_text(model, vocab, prompt, device, max_length=50, temperature=0.7, top_k=50, stop_tokens=None):
    model.eval()
    rev_vocab = {idx: word for word, idx in vocab.items()}

    # Токенизация промпта
    tokens = prompt.split()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]

    # Определение стоп-токенов (включая <unk>)
    stop_tokens = stop_tokens or {"<eos>"}
    stop_tokens = stop_tokens | {"<unk>"}  # Объединение множеств
    stop_ids = {vocab[token] for token in stop_tokens if token in vocab}

    # Генерация
    for _ in range(max_length):
        inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
        mask = torch.ones_like(inputs, dtype=torch.bool)

        with torch.amp.autocast(device_type=device.type, enabled=True):
            outputs = model(inputs, src_key_padding_mask=~mask)
            logits = outputs[0, -1, :] / temperature

            # Фильтрация top-k
            if top_k > 0:
                top_vals, top_idxs = torch.topk(logits, top_k)
                logits[logits < top_vals[-1]] = -float('Inf')

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1).item()

            # Прерывание при появлении стоп-токена (включая <unk>)
            if next_token in stop_ids:
                break

            input_ids.append(next_token)
            tokens.append(rev_vocab.get(next_token, "<unk>"))

    return " ".join(tokens)

# ====================== ИНТЕРФЕЙС ======================
def interactive_mode():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        # Загрузка словаря
        if not os.path.exists(config.vocab_path):
            print("❌ Словарь не найден!")
            return

        with open(config.vocab_path, 'r', encoding='utf-8') as f:
            vocab = json.load(f)

        # Загрузка модели
        if not os.path.exists(config.model_path):
            print("❌ Модель не найдена!")
            return

        # Загрузка конфигурации модели из чекпоинта (если доступно)
        model_path = config.model_path if os.path.exists(config.model_path) else config.model_path
        checkpoint = torch.load(model_path, map_location=device)

        model = TextTransformer(
            vocab_size=checkpoint.get('vocab_size', len(vocab)),
            d_model=checkpoint.get('d_model', config.d_model),
            nhead=checkpoint.get('nhead', config.nhead),
            num_layers=checkpoint.get('num_layers', config.num_layers),
            dim_feedforward=checkpoint.get('dim_feedforward', config.dim_feedforward),
            dropout=checkpoint.get('dropout', config.dropout)
        ).to(device)

        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()

        print("Модель загружена. Введите текст для генерации (или 'exit' для выхода)")
        print("Параметры генерации: temperature=0.7, top_k=50")

        while True:
            try:
                prompt = input("\nВаше сообщение: ").strip()
                if not prompt:
                    print("⚠️ Введите текст!")
                    continue
                if prompt.lower() == 'exit':
                    break

                start_time = time.time()
                generated = generate_text(model, vocab, prompt, device)
                gen_time = time.time() - start_time

                print(f"\nСеть: {generated}")
                print(f"Время генерации: {gen_time:.2f}с")

            except KeyboardInterrupt:
                print("\nВыход...")
                break

    except Exception as e:
        print(f"❌ Ошибка в интерактивном режиме: {str(e)}")

def continue_training():
    run_training(resume_checkpoint=config.model_path)

def main_menu():
    while True:
        print("\nМеню текстовой модели:")
        print("1. Обучить модель с нуля")
        print("2. Продолжить обучение с чекпоинта")
        print("3. Интерактивный режим")
        print("4. Выход")
        choice = input("Выбор: ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            continue_training()
        elif choice == '3':
            interactive_mode()
        elif choice == '4':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()