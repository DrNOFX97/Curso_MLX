## Módulo 2: Gestão de Recursos e Limitações

### 2.1 Compreender os 16GB de RAM

#### Unified Memory: Como Funciona

O M1 usa **Unified Memory Architecture (UMA)** - um conceito revolucionário:

**Arquitectura Tradicional (Intel/NVIDIA):**
```
CPU RAM (16GB) ←→ PCIe Bus ←→ GPU VRAM (8GB)
     ↑                              ↑
  Separados!              Cópia necessária!
```

**Arquitectura M1 (Unified):**
```
CPU + GPU + Neural Engine
         ↓
    Unified Memory (16GB)
         ↑
  Partilhado por todos!
```

**Vantagens:**
- ✅ Sem overhead de transferências CPU↔GPU
- ✅ GPU pode aceder a toda a RAM disponível
- ✅ Batch sizes maiores que GPUs com 8GB VRAM
- ✅ Zero-copy entre operações

**Desvantagens:**
- ❌ Total de 16GB é partilhado (sistema + apps + modelo)
- ❌ Sem possibilidade de expandir
- ❌ Swap para disco é MUITO lento para ML

#### Distribuição Típica da Memória

```
┌─────────────────────────────────────┐
│ macOS Sistema          │ 3-4 GB     │
├─────────────────────────────────────┤
│ Apps em Background     │ 1-2 GB     │
├─────────────────────────────────────┤
│ Disponível para ML     │ 10-12 GB   │
└─────────────────────────────────────┘
```

**Realidade:** Tens ~10-12GB úteis para treino de modelos.

#### Monitorização de Uso de Memória

**1. Activity Monitor (GUI):**
- Aplicações → Utilitários → Monitor de Atividade
- Tab "Memória"
- Observa: "Pressão de Memória" (deve estar verde!)

**2. Terminal - Comando `top`:**
```bash
top -l 1 | grep PhysMem
```

**3. htop (mais visual):**
```bash
# Se ainda não instalaste
brew install htop

# Executar
htop
```

**4. Script Python de Monitorização:**

```python
# monitor_memoria.py
import psutil
import subprocess

def get_memory_info():
    """Informação detalhada de memória"""
    mem = psutil.virtual_memory()
    
    print("=" * 50)
    print("MEMÓRIA DO SISTEMA")
    print("=" * 50)
    print(f"Total:       {mem.total / (1024**3):.2f} GB")
    print(f"Disponível:  {mem.available / (1024**3):.2f} GB")
    print(f"Usada:       {mem.used / (1024**3):.2f} GB")
    print(f"Percentagem: {mem.percent}%")
    print(f"Livre:       {mem.free / (1024**3):.2f} GB")
    
    # Pressão de memória (específico macOS)
    try:
        result = subprocess.run(['memory_pressure'], 
                              capture_output=True, text=True)
        if 'System-wide memory free percentage' in result.stdout:
            print("\n" + "=" * 50)
            print("PRESSÃO DE MEMÓRIA")
            print("=" * 50)
            for line in result.stdout.split('\n'):
                if 'percentage' in line or 'level' in line:
                    print(line)
    except:
        pass

if __name__ == "__main__":
    get_memory_info()
```

Instala dependência e executa:
```bash
pip install psutil
python monitor_memoria.py
```

**5. Monitorização Durante Treino:**

```python
# monitor_treino.py
import psutil
import time
import GPUtil  # Para monitorizar GPU

def monitor_recursos(intervalo=2):
    """
    Monitoriza recursos em tempo real
    Útil para executar em paralelo com o treino
    """
    try:
        while True:
            # CPU e RAM
            cpu = psutil.cpu_percent(interval=1)
            mem = psutil.virtual_memory()
            
            print(f"\r CPU: {cpu:5.1f}% | RAM: {mem.percent:5.1f}% "
                  f"({mem.available / (1024**3):.1f}GB livres)", end='')
            
            time.sleep(intervalo)
    except KeyboardInterrupt:
        print("\n\nMonitorização parada.")

if __name__ == "__main__":
    print("Monitorizando recursos (Ctrl+C para parar)...")
    monitor_recursos()
```

---

### 2.2 Otimização de Memória

#### Batch Size Adequado

O batch size é o parâmetro mais crítico para gestão de memória:

**Regra Geral:**
```python
# Memória usada ≈ batch_size × tamanho_input × profundidade_modelo × 4 (float32)

# Exemplo: Imagens 224×224×3, modelo ResNet50
# batch_size=32: ~4GB
# batch_size=64: ~8GB  ← Ideal para M1 16GB
# batch_size=128: ~16GB ← Arriscado!
```

**Encontrar o Batch Size Ideal:**

```python
# encontrar_batch_size.py
import torch
import torch.nn as nn

def find_optimal_batch_size(model, input_shape, device='mps'):
    """
    Encontra o maior batch size que cabe na memória
    """
    model = model.to(device)
    batch_size = 1
    
    print("Testando batch sizes...")
    
    while batch_size <= 512:
        try:
            # Criar batch de teste
            dummy_input = torch.randn(batch_size, *input_shape, device=device)
            
            # Forward pass
            with torch.no_grad():
                output = model(dummy_input)
            
            # Limpar memória
            del dummy_input, output
            torch.mps.empty_cache() if device == 'mps' else torch.cuda.empty_cache()
            
            print(f"✓ Batch size {batch_size:4d} funciona")
            batch_size *= 2
            
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"✗ Batch size {batch_size:4d} - Out of Memory")
                optimal = batch_size // 2
                print(f"\n🎯 Batch size recomendado: {optimal}")
                print(f"   (Usa {optimal // 2} para margem de segurança)")
                return optimal
            else:
                raise e
    
    return batch_size // 2

# Exemplo de uso
if __name__ == "__main__":
    from torchvision.models import resnet50
    
    model = resnet50()
    optimal_bs = find_optimal_batch_size(
        model, 
        input_shape=(3, 224, 224),
        device='mps'
    )
```

#### Gradient Accumulation

Quando precisas de batch size maior que cabe na memória:

```python
# gradient_accumulation.py
def train_with_gradient_accumulation(model, dataloader, optimizer, 
                                     accumulation_steps=4):
    """
    Simula batch_size maior acumulando gradientes
    
    Effective batch size = batch_size × accumulation_steps
    """
    model.train()
    optimizer.zero_grad()
    
    for i, (inputs, targets) in enumerate(dataloader):
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        # Normalizar loss pelo número de acumulações
        loss = loss / accumulation_steps
        
        # Backward pass (acumula gradientes)
        loss.backward()
        
        # Update apenas a cada N steps
        if (i + 1) % accumulation_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
    
    return loss.item()

# Exemplo:
# batch_size=16, accumulation_steps=4
# → Effective batch_size = 64
# Mas usa apenas memória de batch_size=16!
```

#### Mixed Precision Training (FP16)

Usa metade da memória mantendo qualidade:

**TensorFlow:**
```python
import tensorflow as tf

# Ativar mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Criar modelo normalmente
model = tf.keras.applications.ResNet50(weights=None)

# Última camada deve ser float32
outputs = tf.keras.layers.Dense(10, dtype='float32')(model.output)
model = tf.keras.Model(inputs=model.input, outputs=outputs)

print("Mixed precision ativado! ✓")
print(f"Compute dtype: {policy.compute_dtype}")
print(f"Variable dtype: {policy.variable_dtype}")
```

**PyTorch:**
```python
import torch
from torch.cuda.amp import autocast, GradScaler

# Criar modelo e optimizer
model = YourModel().to('mps')
optimizer = torch.optim.Adam(model.parameters())
scaler = GradScaler()  # Para estabilidade numérica

# Loop de treino
for inputs, targets in dataloader:
    optimizer.zero_grad()
    
    # Forward pass em FP16
    with autocast(device_type='mps'):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
    
    # Backward com scaling
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
```

**Economia de Memória:**
- FP32 (float32): 4 bytes por parâmetro
- FP16 (float16): 2 bytes por parâmetro
- **Redução: ~50% de uso de memória!**

#### Gradient Checkpointing

Técnica avançada para modelos muito profundos:

```python
import torch
from torch.utils.checkpoint import checkpoint

class ModelWithCheckpointing(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = torch.nn.Linear(1000, 1000)
        self.layer2 = torch.nn.Linear(1000, 1000)
        self.layer3 = torch.nn.Linear(1000, 1000)
        self.layer4 = torch.nn.Linear(1000, 10)
    
    def forward(self, x):
        # Checkpoint em blocos pesados
        x = checkpoint(self._block1, x, use_reentrant=False)
        x = checkpoint(self._block2, x, use_reentrant=False)
        x = self.layer4(x)
        return x
    
    def _block1(self, x):
        return torch.relu(self.layer2(torch.relu(self.layer1(x))))
    
    def _block2(self, x):
        return torch.relu(self.layer3(x))

# Trade-off: -30% memória, +20% tempo de treino
```

---

### 2.3 Gestão de Datasets

#### Datasets que Cabem na Memória

Para datasets pequenos (<5GB):

```python
import numpy as np
import tensorflow as tf

# Carregar tudo para memória
def load_dataset_to_memory():
    # Exemplo: CIFAR-10
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
    
    # Pré-processar
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    print(f"Dataset carregado: {x_train.nbytes / 1e6:.1f} MB")
    
    return x_train, y_train, x_test, y_test
```

#### Data Generators e Streaming

Para datasets grandes (>5GB):

**TensorFlow:**
```python
def create_data_generator(image_dir, batch_size=32):
    """
    Generator que carrega imagens sob demanda
    """
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest',
        validation_split=0.2
    )
    
    train_generator = datagen.flow_from_directory(
        image_dir,
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    return train_generator

# Uso:
# model.fit(train_generator, epochs=10)
# Apenas batch_size imagens na memória de cada vez!
```

**PyTorch:**
```python
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os

class StreamingImageDataset(Dataset):
    """
    Dataset que carrega imagens sob demanda
    """
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = os.listdir(image_dir)
        self.transform = transform
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Carrega apenas quando necessário
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        # Extrair label do nome do ficheiro (exemplo)
        label = int(self.image_files[idx].split('_')[0])
        
        return image, label

# Criar DataLoader
dataset = StreamingImageDataset('data/images')
dataloader = DataLoader(
    dataset, 
    batch_size=32,
    shuffle=True,
    num_workers=2,  # Carregamento paralelo
    pin_memory=True
)
```

#### Pré-processamento Eficiente

**Cache de pré-processamento:**

```python
# preprocessamento_cache.py
import numpy as np
import os
import pickle

def preprocess_and_cache(data, cache_path='cache/processed.pkl'):
    """
    Pré-processa uma vez e guarda em cache
    """
    if os.path.exists(cache_path):
        print("Carregando de cache...")
        with open(cache_path, 'rb') as f:
            return pickle.load(f)
    
    print("Pré-processando pela primeira vez...")
    # Operações pesadas aqui
    processed = heavy_preprocessing(data)
    
    # Guardar cache
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    with open(cache_path, 'wb') as f:
        pickle.dump(processed, f)
    
    return processed
```

**Data Augmentation on-the-fly:**

```python
# Não aumentas o dataset inteiro na memória
# Aplicas transformações durante o treino

from torchvision import transforms

transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(0.2, 0.2, 0.2),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], 
                        [0.229, 0.224, 0.225])
])

# Cada batch tem augmentation diferente
# Sem usar memória extra!
```

---

### 💡 Dicas Práticas de Gestão de Memória

1. **Fecha aplicações desnecessárias** antes de treinar
2. **Usa `del` e garbage collection** para libertar memória:
   ```python
   import gc
   del large_variable
   gc.collect()
   torch.mps.empty_cache()  # PyTorch
   ```

3. **Monitoriza durante desenvolvimento**, otimiza para production

4. **Testa com subset pequeno** antes de treino completo:
   ```python
   # Testar com 10% dos dados
   x_train_small = x_train[:len(x_train)//10]
   model.fit(x_train_small, epochs=1)  # Quick test
   ```

5. **Prefere batch_size múltiplo de 8** (otimizado para hardware)

---

### ✅ Checklist Módulo 2

- [ ] Compreendo Unified Memory Architecture
- [ ] Sei monitorizar uso de RAM (Activity Monitor, htop, Python)
- [ ] Encontrei batch size ideal para meu modelo
- [ ] Sei usar gradient accumulation
- [ ] Mixed precision configurado (FP16)
- [ ] Data generators implementados para datasets grandes
- [ ] Cache de pré-processamento a funcionar

---

### 🎯 Próximos Passos

No **Módulo 3**, vamos finalmente treinar modelos! Começamos com classificação de imagens, NLP e modelos tabulares - tudo otimizado para o M1 16GB!
