import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import math
import time
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
from collections import Counter
import json

# ====================== КОНФИГУРАЦИЯ ======================
class Config:
    # Системные параметры
    system = "my"
    
    if system == "my":
        dir = "/media/alex/Programs/NEURAL_NETWORKS/"
    else:
        dir = "/content/"
    
    # Пути
    data_path = dir + "NeuralNetwork-GPT/DataSet.txt"    # Текстовый файл с фразами
    model_path = dir + "NeuralNetwork-GPT/Model/text_model.pth"  # Путь для модели
    vocab_path = dir + "NeuralNetwork-GPT/Model/vocab.json"      # Словарь
    
    # Параметры модели
    d_model = 512              # Размерность эмбеддингов
    nhead = 8                  # Количество голов внимания
    num_layers = 6             # Количество слоев трансформера
    dim_feedforward = 2048     # Размер скрытого слоя FFN
    max_seq_len = 256          # Максимальная длина последовательности
    dropout = 0.1              # Dropout
    
    # Параметры обучения
    batch_size = 32            # Размер батча
    lr = 0.0001                # Скорость обучения
    epochs = 20                # Количество эпох
    accumulation_steps = 4     # Шаги накопления градиентов

config = Config()

# ====================== ОБРАБОТКА ДАННЫХ ======================
class TextDataset(Dataset):
    def __init__(self, texts, vocab, seq_len):
        self.texts = texts
        self.vocab = vocab
        self.seq_len = seq_len
        
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        tokens = text.split()[:self.seq_len]
        
        # Конвертация в индексы
        input_ids = [self.vocab.get(token, self.vocab["<unk>"]) for token in tokens]
        
        # Добавление паддинга
        if len(input_ids) < self.seq_len:
            input_ids += [self.vocab["<pad>"]] * (self.seq_len - len(input_ids))
        
        # Создание маски внимания
        attention_mask = [1] * len(tokens) + [0] * (self.seq_len - len(tokens))
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.float)
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

def load_data():
    if not os.path.exists(config.data_path):
        raise FileNotFoundError(f"❌ Файл с данными не найден: {config.data_path}")
    
    with open(config.data_path, 'r', encoding='utf-8') as f:
        texts = [line.strip() for line in f if line.strip()]
    
    if len(texts) == 0:
        raise ValueError(f"❌ Файл с данными пуст: {config.data_path}")
    
    return texts

# ====================== МОДЕЛЬ ======================
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x.transpose(0, 1)
        x = x + self.pe[:x.size(0)]
        return self.dropout(x).transpose(0, 1)

class TextTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, nhead, num_layers, dim_feedforward, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward, dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.fc_out = nn.Linear(d_model, vocab_size)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        src = self.embedding(src)
        src = self.pos_encoder(src)
        
        output = self.transformer(
            src, 
            mask=src_mask,
            src_key_padding_mask=src_key_padding_mask
        )
        
        return self.fc_out(output)

# ====================== ОБУЧЕНИЕ ======================
def train_epoch(model, loader, optimizer, device, scheduler=None, scaler=None):
    model.train()
    total_loss = 0
    optimizer.zero_grad()
    
    for i, batch in enumerate(tqdm(loader)):
        inputs = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        
        # Создаем таргеты (сдвинутые на один токен)
        targets = inputs[:, 1:].contiguous()
        inputs = inputs[:, :-1]
        mask = mask[:, :-1]
        
        with torch.amp.autocast(device_type='cuda', enabled=scaler is not None):
            outputs = model(inputs, src_key_padding_mask=(1 - mask).bool())
            
            # Переформатирование вывода для вычисления потерь
            outputs = outputs.view(-1, outputs.size(-1))
            targets = targets.view(-1)
            
            # Игнорируем паддинг в таргетах
            loss = F.cross_entropy(
                outputs,
                targets,
                ignore_index=0  # Игнорируем паддинг
            ) / config.accumulation_steps
        
        if scaler:
            scaler.scale(loss).backward()
        else:
            loss.backward()
        
        if (i + 1) % config.accumulation_steps == 0 or (i + 1) == len(loader):
            if scaler:
                scaler.step(optimizer)
                scaler.update()
            else:
                optimizer.step()
            optimizer.zero_grad()
            if scheduler:
                scheduler.step()
        
        total_loss += loss.item() * config.accumulation_steps
    
    return total_loss / len(loader)

def run_training():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(os.path.dirname(config.model_path), exist_ok=True)
    
    try:
        # Загрузка данных
        texts = load_data()
        print(f"Загружено {len(texts)} фраз")
        
        # Построение словаря
        vocab = build_vocab(texts)
        print(f"Создан словарь из {len(vocab)} токенов")
        
        # Сохранение словаря
        with open(config.vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab, f, ensure_ascii=False)
        
        dataset = TextDataset(texts, vocab, config.max_seq_len)
        loader = DataLoader(
            dataset, 
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count())
        )
        
        model = TextTransformer(
            vocab_size=len(vocab),
            d_model=config.d_model,
            nhead=config.nhead,
            num_layers=config.num_layers,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        ).to(device)
        
        optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer, 
            max_lr=config.lr,
            steps_per_epoch=len(loader),
            epochs=config.epochs
        )
        scaler = torch.amp.GradScaler('cuda', enabled=True)
        
        # Обучение
        for epoch in range(config.epochs):
            start_time = time.time()
            train_loss = train_epoch(model, loader, optimizer, device, scheduler, scaler)
            epoch_time = time.time() - start_time
            
            print(f"Эпоха {epoch+1}/{config.epochs} | "
                  f"Потери: {train_loss:.4f} | "
                  f"Время: {epoch_time:.2f}с")
        
        # Сохранение модели
        torch.save({
            'model_state_dict': model.state_dict(),
            'vocab_size': len(vocab),
            'd_model': config.d_model,  # Явно сохраняем параметры
            'nhead': config.nhead,
            'num_layers': config.num_layers,
            'dim_feedforward': config.dim_feedforward,
            'dropout': config.dropout
        }, config.model_path)
        print(f"Модель сохранена: {config.model_path}")
        
    except Exception as e:
        print(f"❌ Ошибка при обучении: {str(e)}")

# ====================== ГЕНЕРАЦИЯ ======================
def generate_text(model, vocab, prompt, device, max_length=50, temperature=0.7, top_k=50):
    model.eval()
    rev_vocab = {idx: word for word, idx in vocab.items()}
    
    # Токенизация промпта
    tokens = prompt.split()
    input_ids = [vocab.get(token, vocab["<unk>"]) for token in tokens]
    
    # Генерация
    with torch.no_grad():
        for _ in range(max_length):
            inputs = torch.tensor([input_ids], dtype=torch.long).to(device)
            mask = torch.ones_like(inputs, dtype=torch.float)
            
            outputs = model(inputs, src_key_padding_mask=(1 - mask).bool())
            
            # Получение последнего токена
            logits = outputs[0, -1, :] / temperature
            
            # Фильтрация top-k
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = -float('Inf')
            
            # Применение softmax
            probs = F.softmax(logits, dim=-1)
            
            # Выборка
            next_token = torch.multinomial(probs, 1).item()
            
            # Проверка на конец предложения
            if next_token == vocab["<eos>"]:
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
        
        checkpoint = torch.load(config.model_path, map_location=device)
        model = TextTransformer(
            vocab_size=checkpoint['vocab_size'],
            d_model=checkpoint['d_model'],  # Прямой доступ к параметрам
            nhead=checkpoint['nhead'],
            num_layers=checkpoint['num_layers'],
            dim_feedforward=checkpoint['dim_feedforward'],
            dropout=checkpoint['dropout']
        ).to(device)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        print("Модель загружена. Введите текст для генерации (или 'exit' для выхода)")
        
        while True:
            try:
                prompt = input("\nВаше сообщение: ").strip()
            except KeyboardInterrupt:
                print("\nВыход...")
                break
                
            if prompt.lower() == 'exit':
                break
                
            if not prompt:
                print("⚠️ Введите текст!")
                continue
                
            start_time = time.time()
            generated = generate_text(model, vocab, prompt, device)
            gen_time = time.time() - start_time
            
            print(f"\nБогдан: {generated}")
            print(f"Время генерации: {gen_time:.2f}с")
            
    except Exception as e:
        print(f"❌ Ошибка в интерактивном режиме: {str(e)}")

def main_menu():
    while True:
        print("\nМеню текстовой модели:")
        print("1. Обучить модель")
        print("2. Интерактивный режим")
        print("3. Выход")
        choice = input("Выбор: ").strip()

        if choice == '1':
            run_training()
        elif choice == '2':
            interactive_mode()
        elif choice == '3':
            break
        else:
            print("Неверный ввод!")

if __name__ == "__main__":
    main_menu()