# Uso
profiler = ProfilerM1()

# Profile de carregamento de dados
with profiler.profile_block("Carregamento de dados"):
    # Curso: Como Treinar Modelos de Aprendizagem Automática no MacBook Pro M1 16GB

## Módulo 1: Preparação do Ambiente

### 1.1 Introdução ao chip Apple Silicon M1

#### Arquitetura ARM vs x86
O chip M1 da Apple representa uma mudança fundamental na computação pessoal:

**Arquitetura ARM:**
- O M1 usa arquitetura ARM64, não x86 como os processadores Intel
- Mais eficiente energeticamente (RISC vs CISC)
- Alguns programas precisam de ser recompilados ou usar Rosetta 2 (camada de tradução)
- Para AA: bibliotecas nativas ARM têm desempenho muito superior

**Diferenças práticas:**
- Binários x86 funcionam via Rosetta 2, mas são mais lentos
- Python compilado para ARM é até 2-3x mais rápido
- Nunca uses Python do python.org (é x86) - usa Miniforge!

#### GPU Integrada e Neural Engine

**GPU (7 ou 8 núcleos no M1):**
- Totalmente integrada com a CPU (Unified Memory Architecture)
- Acesso direto à RAM sem transferências CPU↔GPU
- Framework Metal para aceleração GPU
- Ideal para treino de modelos pequenos/médios

**Neural Engine (16 núcleos):**
- Chip dedicado para operações de AA
- 11 TFLOPS de desempenho
- Usado principalmente para inferência (Core ML)
- Menos flexível que GPU, mas muito eficiente

#### Vantagens para Aprendizagem Automática

1. **Unified Memory Architecture (UMA):**
   - CPU e GPU partilham os mesmos 16GB de RAM
   - Sem overhead de transferências de dados
   - Batch sizes maiores que GPUs dedicadas com VRAM limitada

2. **Eficiência Energética:**
   - Treino prolongado sem sobreaquecimento
   - Bateria dura horas mesmo em treino intensivo

3. **Desempenho Surpreendente:**
   - Para modelos até ~7B parâmetros, competitivo com GPUs mid-range
   - Excelente para prototipagem e experimentação

---

### 1.2 Configuração Inicial

#### Passo 1: Instalar Homebrew

Homebrew é o gestor de pacotes essencial para macOS. Abre o Terminal e executa:

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Após instalação, adiciona ao PATH (o instalador mostrará os comandos exatos):

```bash
echo 'eval "$(/opt/homebrew/bin/brew shellenv)"' >> ~/.zshrc
eval "$(/opt/homebrew/bin/brew shellenv)"
```

Verifica a instalação:
```bash
brew --version
```

#### Passo 2: Instalar Miniforge (Python para ARM)

**IMPORTANTE:** Nunca uses o Python do python.org no M1! Usa Miniforge (Conda otimizado para ARM).

```bash
# Download do Miniforge
brew install miniforge

# Inicializar conda
conda init zsh

# Reinicia o terminal ou executa:
source ~/.zshrc
```

Verifica que estás a usar Python ARM:
```bash
python -c "import platform; print(platform.machine())"
# Deve retornar: arm64
```

#### Passo 3: Criar Ambiente Virtual Base

Cria um ambiente dedicado para AA:

```bash
# Criar ambiente com Python 3.11
conda create -n ml-m1 python=3.11 -y

# Ativar ambiente
conda activate ml-m1

# Instalar pacotes base
conda install -c conda-forge numpy pandas matplotlib scikit-learn jupyter -y
```

**Configurar ambiente como padrão:**
```bash
# Adiciona ao ~/.zshrc para ativar automaticamente
echo "conda activate ml-m1" >> ~/.zshrc
```

#### Passo 4: Ferramentas de Desenvolvimento

```bash
# Git (se ainda não tiveres)
brew install git

# Editor de código (opcional)
brew install --cask visual-studio-code

# Ferramentas de monitorização
brew install htop
```

---

### 1.3 Frameworks Otimizados para M1

#### TensorFlow Metal

TensorFlow com aceleração GPU via Metal:

```bash
conda activate ml-m1

# Instalar TensorFlow e plugin Metal
pip install tensorflow==2.16.1
pip install tensorflow-metal==1.1.0
```

**Testar aceleração GPU:**

```python
import tensorflow as tf

# Verificar dispositivos disponíveis
print("Dispositivos disponíveis:")
print(tf.config.list_physical_devices())

# Deve mostrar GPU
print("\nGPU disponível:", len(tf.config.list_physical_devices('GPU')) > 0)

# Teste de performance
import time
import numpy as np

# Criar dados aleatórios
x = tf.random.normal([10000, 10000])

# Operação na GPU
start = time.time()
y = tf.matmul(x, x)
gpu_time = time.time() - start

print(f"\nTempo de multiplicação de matrizes (10000x10000): {gpu_time:.4f}s")
print("✓ TensorFlow Metal está a funcionar!" if gpu_time < 1.0 else "⚠ Performance abaixo do esperado")
```

#### PyTorch com MPS (Metal Performance Shaders)

PyTorch com suporte nativo para GPU do M1:

```bash
conda activate ml-m1

# Instalar PyTorch com suporte MPS
pip install torch torchvision torchaudio
```

**Testar aceleração MPS:**

```python
import torch

# Verificar disponibilidade do MPS
print("MPS disponível:", torch.backends.mps.is_available())
print("MPS construído:", torch.backends.mps.is_built())

# Definir dispositivo
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Teste de performance
import time

x = torch.randn(10000, 10000, device=device)

start = time.time()
y = torch.matmul(x, x)
torch.mps.synchronize()  # Esperar GPU terminar
mps_time = time.time() - start

print(f"\nTempo de multiplicação de matrizes: {mps_time:.4f}s")
print("✓ PyTorch MPS está a funcionar!" if mps_time < 1.0 else "⚠ Performance abaixo do esperado")
```

#### Comparação: TensorFlow vs PyTorch no M1

| Aspecto | TensorFlow Metal | PyTorch MPS |
|---------|-----------------|
| **Maturidade M1** | Muito estável | Estável (melhorou muito) |
| **Desempenho** | Excelente | Excelente |
| **Compatibilidade** | Alta | Algumas operações não suportadas |
| **Comunidade** | Maior | A crescer rapidamente |
| **Recomendação** | Projetos de produção | Pesquisa e prototipagem |

#### JAX com Metal (Opcional - Avançado)

A Apple lançou suporte oficial para JAX com aceleração Metal! Ideal para pesquisa e computação numérica de alto desempenho.

**Instalação:**

```bash
conda activate ml-m1

# Instalar JAX com plugin Metal da Apple
pip install jax-metal
```

Isto instala automaticamente o JAX e o plugin Metal otimizado.

**Testar aceleração Metal:**

```python
import jax
import jax.numpy as jnp

# Verificar dispositivos
print("Dispositivos JAX:", jax.devices())
print("Dispositivo padrão:", jax.default_backend())

# Teste de performance
import time

x = jax.random.normal(jax.random.PRNGKey(0), (10000, 10000))

# JIT compile para otimização
@jax.jit
def matrix_multiply(a):
    return jnp.matmul(a, a)

# Warm-up (primeira execução compila)
_ = matrix_multiply(x)

# Benchmark
start = time.time()
result = matrix_multiply(x)
result.block_until_ready()  # Esperar GPU terminar
jax_time = time.time() - start

print(f"\nTempo de multiplicação de matrizes: {jax_time:.4f}s")
print("✓ JAX Metal está a funcionar!")
```

**Quando usar JAX:**
- Pesquisa que requer diferenciação automática avançada
- Código científico que precisa de ser muito otimizado
- Quando queres controlo fino sobre transformações (vmap, pmap, etc.)
- Treino com XLA (compilação otimizada)

**Documentação oficial:** https://developer.apple.com/metal/jax/

#### Verificação Final do Ambiente

Cria um script de teste completo:

```python
# teste_ambiente.py
import sys
print(f"Python: {sys.version}")
print(f"Arquitetura: {sys.platform}")

try:
    import tensorflow as tf
    print(f"✓ TensorFlow: {tf.__version__}")
    print(f"  GPU disponível: {len(tf.config.list_physical_devices('GPU')) > 0}")
except ImportError:
    print("✗ TensorFlow não instalado")

try:
    import torch
    print(f"✓ PyTorch: {torch.__version__}")
    print(f"  MPS disponível: {torch.backends.mps.is_available()}")
except ImportError:
    print("✗ PyTorch não instalado")

try:
    import numpy as np
    print(f"✓ NumPy: {np.__version__}")
except ImportError:
    print("✗ NumPy não instalado")

try:
    import pandas as pd
    print(f"✓ Pandas: {pd.__version__}")
except ImportError:
    print("✗ Pandas não instalado")

print("\n🎉 Ambiente configurado com sucesso!")
```

Executa:
```bash
python teste_ambiente.py
```

---

### ✅ Checklist Módulo 1

- [ ] Homebrew instalado
- [ ] Miniforge instalado (Python ARM)
- [ ] Ambiente virtual `ml-m1` criado
- [ ] TensorFlow Metal instalado e testado
- [ ] PyTorch MPS instalado e testado
- [ ] Script de teste executado com sucesso
- [ ] GPU/MPS a funcionar corretamente

---

### 🎯 Próximos Passos

No **Módulo 2**, vamos aprender a gerir eficientemente os 16GB de RAM e monitorizar recursos durante o treino!

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

## Módulo 3: Treino de Modelos Pequenos e Médios

### 3.1 Classificação de Imagens

#### O que são CNNs (Convolutional Neural Networks)?

**CNN = Rede Neural Convolucional**

São um tipo especial de rede neural **desenhada especificamente para processar imagens**. São a base de praticamente toda a visão computacional moderna.

**Por que CNNs existem?**

Problema com redes neurais normais em imagens:
- Imagina uma imagem pequena de 224×224 pixels RGB
- 224 × 224 × 3 = **150,528 pixels**
- Rede neural normal: cada neurônio conecta-se a TODOS os pixels
- Primeira camada com 1000 neurônios = **150 milhões de parâmetros!**
- ❌ Impossível de treinar, overfitting garantido

Solução das CNNs:
- ✅ Usa **filtros locais** (olha pequenas regiões de cada vez)
- ✅ Partilha pesos (mesmo filtro para toda a imagem)
- ✅ Aprende hierarquia de features (bordas → formas → objetos)

**Como funcionam as camadas convolucionais:**

Um filtro 3×3 "desliza" pela imagem detectando padrões específicos. Cada filtro aprende a detectar uma característica diferente (bordas verticais, horizontais, curvas, texturas, etc.).

**Hierarquia de aprendizagem:**
```
Camada 1 → Deteta bordas simples (/, \, |, ―)
Camada 2 → Combina bordas em formas (círculos, quadrados)
Camada 3 → Combina formas em partes (olhos, rodas, janelas)
Camada 4 → Combina partes em objetos (gato, carro, casa)
```

**Arquitetura típica de uma CNN:**

```python
# cnn_basica.py
import tensorflow as tf
from tensorflow import keras

# CNN simples para entender a estrutura
model = keras.Sequential([
    # BLOCO 1: Deteta features simples
    keras.layers.Conv2D(32, (3,3), activation='relu', 
                        input_shape=(224,224,3), name='conv1'),
    keras.layers.MaxPooling2D((2,2), name='pool1'),
    
    # BLOCO 2: Features mais complexas
    keras.layers.Conv2D(64, (3,3), activation='relu', name='conv2'),
    keras.layers.MaxPooling2D((2,2), name='pool2'),
    
    # BLOCO 3: Features de alto nível
    keras.layers.Conv2D(128, (3,3), activation='relu', name='conv3'),
    keras.layers.MaxPooling2D((2,2), name='pool3'),
    
    # Achatar para classificação
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10, activation='softmax')  # 10 classes
])

model.summary()

# Mostra a progressão do tamanho:
# Input:  224×224×3    (imagem original)
# Conv1:  222×222×32   (32 filtros aprendidos)
# Pool1:  111×111×32   (redução de tamanho)
# Conv2:  109×109×64   (64 padrões mais complexos)
# Pool2:  54×54×64     (redução)
# Conv3:  52×52×128    (128 features abstratas)
# Pool3:  26×26×128    (redução final)
# Flatten: 86528       (vetor único)
# Dense:  128          (combinação de features)
# Output: 10           (probabilidades das classes)
```

**Visualizar o que uma CNN aprende:**

```python
# visualizar_cnn.py
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

def visualizar_filtros_cnn(caminho_imagem):
    """
    Mostra o que cada camada de uma CNN deteta
    """
    # Carregar modelo pré-treinado
    model = keras.applications.MobileNetV2(weights='imagenet', include_top=False)
    
    # Carregar e preparar imagem
    img = keras.preprocessing.image.load_img(caminho_imagem, target_size=(224, 224))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = keras.applications.mobilenet_v2.preprocess_input(img_array)
    
    # Criar modelo para ver ativações intermediárias
    layer_outputs = [layer.output for layer in model.layers[:8]]
    activation_model = keras.Model(inputs=model.input, outputs=layer_outputs)
    
    # Obter ativações
    activations = activation_model.predict(img_array)
    
    # Visualizar primeira camada (features básicas)
    first_layer = activations[0]
    
    plt.figure(figsize=(15, 10))
    plt.subplot(4, 9, 1)
    plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    # Mostrar 32 filtros diferentes
    for i in range(32):
        plt.subplot(4, 9, i+2)
        plt.imshow(first_layer[0, :, :, i], cmap='viridis')
        plt.axis('off')
        plt.title(f'F{i+1}', fontsize=8)
    
    plt.suptitle('O que a CNN "vê": 32 filtros da primeira camada', fontsize=14)
    plt.tight_layout()
    plt.savefig('cnn_visualizacao.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print("✓ Cada imagem colorida mostra um padrão diferente que a CNN detetou!")
    print("  - Alguns filtros detetam bordas verticais")
    print("  - Outros detetam bordas horizontais")
    print("  - Outros detetam cores ou texturas específicas")

# Uso:
# visualizar_filtros_cnn('gato.jpg')
```

**Tipos de operações em CNNs:**

1. **Convolução**: Deteta padrões locais
2. **Pooling**: Reduz dimensionalidade mantendo informação importante
3. **Batch Normalization**: Estabiliza o treino
4. **Dropout**: Previne overfitting

Agora que entendes CNNs, vamos aos modelos práticos otimizados para o M1!

#### CNNs Compactas (MobileNet, EfficientNet)

Modelos leves e eficientes, perfeitos para o M1 16GB:

**Por que usar modelos compactos?**
- ✅ Treinam mais rápido
- ✅ Usam menos memória
- ✅ Performance surpreendentemente boa
- ✅ Ideais para dispositivos móveis/edge

**Comparação de Modelos:**

| Modelo | Parâmetros | Tamanho | Top-1 Accuracy | Ideal para M1? |
|--------|-----------|---------|----------------|----------------|
| MobileNetV2 | 3.5M | 14MB | 71.3% | ✅ Excelente |
| EfficientNetB0 | 5.3M | 29MB | 77.1% | ✅ Excelente |
| ResNet50 | 25.6M | 98MB | 76.1% | ✅ Bom |
| EfficientNetB4 | 19M | 75MB | 83.0% | ⚠️ Usar batch pequeno |
| ResNet152 | 60.2M | 232MB | 78.3% | ⚠️ Limite do M1 |

#### Exemplo Prático: Classificador de Imagens Custom

**Cenário:** Criar classificador de 10 categorias de animais.

```python
# classificador_animais.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

def criar_modelo_mobilenet(num_classes=10, input_shape=(224, 224, 3)):
    """
    Modelo baseado em MobileNetV2 para classificação custom
    """
    # Base pré-treinada (sem top layer)
    base_model = keras.applications.MobileNetV2(
        input_shape=input_shape,
        include_top=False,
        weights='imagenet'
    )
    
    # Congelar base inicialmente (transfer learning)
    base_model.trainable = False
    
    # Adicionar camadas custom
    inputs = keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    x = keras.layers.Dense(128, activation='relu')(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(num_classes, activation='softmax', dtype='float32')(x)
    
    model = keras.Model(inputs, outputs)
    
    # Compilar
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model, base_model

# Criar modelo
model, base_model = criar_modelo_mobilenet(num_classes=10)
model.summary()
```

**Preparar dados:**

```python
# preparar_dados.py
def preparar_dataset(data_dir, img_size=(224, 224), batch_size=32):
    """
    Prepara dataset com data augmentation
    """
    # Data augmentation para treino
    train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
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
    
    # Apenas rescale para validação
    val_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        validation_split=0.2
    )
    
    train_generator = train_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='training',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        data_dir,
        target_size=img_size,
        batch_size=batch_size,
        class_mode='categorical',
        subset='validation',
        shuffle=False
    )
    
    return train_generator, val_generator

# Estrutura de pastas esperada:
# data/
#   ├── animals/
#   │   ├── cat/
#   │   │   ├── img1.jpg
#   │   │   └── img2.jpg
#   │   ├── dog/
#   │   └── ...
```

#### Transfer Learning com Modelos Pré-treinados

**Fase 1: Treinar apenas o top (rápido)**

```python
# treino_fase1.py
from tensorflow import keras

def treino_fase1(model, train_gen, val_gen, epochs=10):
    """
    Fase 1: Apenas camadas custom (base congelada)
    """
    print("🎯 FASE 1: Treino do classificador (base congelada)")
    print("=" * 60)
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=3,
            restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=2,
            min_lr=1e-7
        ),
        keras.callbacks.ModelCheckpoint(
            'modelo_fase1.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Executar
history1 = treino_fase1(model, train_generator, val_generator, epochs=10)
```

**Fase 2: Fine-tuning (descongelar algumas camadas)**

```python
# treino_fase2.py
def treino_fase2(model, base_model, train_gen, val_gen, epochs=20):
    """
    Fase 2: Fine-tuning de camadas superiores
    """
    print("\n🎯 FASE 2: Fine-tuning (descongelando camadas superiores)")
    print("=" * 60)
    
    # Descongelar últimas 30 camadas da base
    base_model.trainable = True
    for layer in base_model.layers[:-30]:
        layer.trainable = False
    
    # Recompilar com learning rate menor
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-5),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print(f"Camadas treináveis: {sum([1 for l in model.layers if l.trainable])}")
    
    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            restore_best_weights=True
        ),
        keras.callbacks.ModelCheckpoint(
            'modelo_final.keras',
            save_best_only=True,
            monitor='val_accuracy'
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1
    )
    
    return history

# Executar
history2 = treino_fase2(model, base_model, train_generator, val_generator, epochs=20)
```

#### Fine-tuning Eficiente

**Script completo de treino otimizado para M1:**

```python
# treino_completo_m1.py
import tensorflow as tf
from tensorflow import keras
import time

# Ativar mixed precision
policy = keras.mixed_precision.Policy('mixed_float16')
keras.mixed_precision.set_global_policy(policy)

class TreinadorM1:
    def __init__(self, num_classes, modelo_base='mobilenet'):
        self.num_classes = num_classes
        self.modelo_base = modelo_base
        self.model = None
        self.base_model = None
        
    def criar_modelo(self):
        """Cria modelo otimizado para M1"""
        if self.modelo_base == 'mobilenet':
            base = keras.applications.MobileNetV2(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        elif self.modelo_base == 'efficientnet':
            base = keras.applications.EfficientNetB0(
                include_top=False,
                weights='imagenet',
                input_shape=(224, 224, 3)
            )
        
        base.trainable = False
        
        inputs = keras.Input(shape=(224, 224, 3))
        x = base(inputs, training=False)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(128, activation='relu')(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(
            self.num_classes, 
            activation='softmax',
            dtype='float32'  # Importante para mixed precision
        )(x)
        
        model = keras.Model(inputs, outputs)
        self.model = model
        self.base_model = base
        
        return model
    
    def treinar_completo(self, train_gen, val_gen):
        """Pipeline completo de treino"""
        
        # FASE 1: Transfer Learning
        print("🚀 Iniciando Fase 1: Transfer Learning")
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start = time.time()
        hist1 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=10,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True),
                keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=2)
            ]
        )
        tempo_fase1 = time.time() - start
        
        # FASE 2: Fine-tuning
        print("\n🎯 Iniciando Fase 2: Fine-tuning")
        self.base_model.trainable = True
        for layer in self.base_model.layers[:-30]:
            layer.trainable = False
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(1e-5),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        start = time.time()
        hist2 = self.model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=15,
            callbacks=[
                keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True),
                keras.callbacks.ModelCheckpoint('melhor_modelo.keras', save_best_only=True)
            ]
        )
        tempo_fase2 = time.time() - start
        
        print(f"\n✅ Treino completo!")
        print(f"Fase 1: {tempo_fase1/60:.1f} minutos")
        print(f"Fase 2: {tempo_fase2/60:.1f} minutos")
        print(f"Total: {(tempo_fase1+tempo_fase2)/60:.1f} minutos")
        
        return hist1, hist2

# Uso
treinador = TreinadorM1(num_classes=10, modelo_base='mobilenet')
treinador.criar_modelo()
hist1, hist2 = treinador.treinar_completo(train_generator, val_generator)
```

---

### 3.2 Processamento de Linguagem Natural (NLP)

#### Modelos Pequenos (DistilBERT, TinyBERT)

Para NLP no M1 16GB, usa versões destiladas:

| Modelo | Parâmetros | Tamanho | Performance | M1 16GB |
|--------|-----------|---------|-------------|---------|
| BERT-base | 110M | 440MB | 100% | ⚠️ Limite |
| DistilBERT | 66M | 260MB | 97% | ✅ Ideal |
| TinyBERT | 14M | 56MB | 96% | ✅ Excelente |
| MobileBERT | 25M | 100MB | 99% | ✅ Muito bom |

**Instalação:**

```bash
pip install transformers datasets accelerate
```

#### Treino de Embeddings

**Exemplo: Classificação de Sentimentos**

```python
# classificador_sentimentos.py
from transformers import (
    DistilBertTokenizer, 
    TFDistilBertForSequenceClassification,
    DataCollatorWithPadding
)
from datasets import load_dataset
import tensorflow as tf

class ClassificadorSentimentos:
    def __init__(self, num_labels=2):
        self.num_labels = num_labels
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = None
        
    def preparar_modelo(self):
        """Inicializa DistilBERT para classificação"""
        self.model = TFDistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=self.num_labels
        )
        
        # Otimizador com learning rate pequeno
        optimizer = tf.keras.optimizers.Adam(learning_rate=2e-5)
        loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
        self.model.compile(
            optimizer=optimizer,
            loss=loss,
            metrics=['accuracy']
        )
        
        return self.model
    
    def tokenizar_dados(self, exemplos):
        """Tokeniza textos"""
        return self.tokenizer(
            exemplos['text'],
            padding='max_length',
            truncation=True,
            max_length=128  # Reduzir para economizar memória
        )
    
    def preparar_dataset(self, dataset_name='imdb'):
        """
        Carrega e prepara dataset
        """
        # Carregar dataset
        dataset = load_dataset(dataset_name)
        
        # Tokenizar
        tokenized = dataset.map(
            self.tokenizar_dados,
            batched=True,
            remove_columns=dataset['train'].column_names
        )
        
        # Converter para TF format
        train_dataset = tokenized['train'].to_tf_dataset(
            columns=['input_ids', 'attention_mask'],
            label_cols=['label'],
            shuffle=True,
            batch_size=16,  # Batch pequeno para M1
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.tokenizer, 
                return_tensors='tf'
            )
        )
        
        test_dataset = tokenized['test'].to_tf_dataset(
            columns=['input_ids', 'attention_mask'],
            label_cols=['label'],
            shuffle=False,
            batch_size=16,
            collate_fn=DataCollatorWithPadding(
                tokenizer=self.tokenizer,
                return_tensors='tf'
            )
        )
        
        return train_dataset, test_dataset
    
    def treinar(self, train_dataset, val_dataset, epochs=3):
        """
        Treina o modelo
        """
        callbacks = [
            tf.keras.callbacks.EarlyStopping(
                monitor='val_loss',
                patience=2,
                restore_best_weights=True
            ),
            tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=1
            )
        ]
        
        history = self.model.fit(
            train_dataset,
            validation_data=val_dataset,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history

# Uso
classificador = ClassificadorSentimentos(num_labels=2)
classificador.preparar_modelo()
train_ds, test_ds = classificador.preparar_dataset('imdb')
history = classificador.treinar(train_ds, test_ds, epochs=3)
```

#### Classificação de Texto com PyTorch

```python
# classificador_pytorch.py
import torch
from torch.utils.data import DataLoader
from transformers import (
    DistilBertTokenizer,
    DistilBertForSequenceClassification,
    AdamW,
    get_linear_schedule_with_warmup
)
from tqdm import tqdm

class TreinadorNLP:
    def __init__(self, num_labels=2, device='mps'):
        self.device = device
        self.tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        self.model = DistilBertForSequenceClassification.from_pretrained(
            'distilbert-base-uncased',
            num_labels=num_labels
        ).to(device)
        
    def treinar_epoch(self, dataloader, optimizer, scheduler):
        """
        Treina uma epoch
        """
        self.model.train()
        total_loss = 0
        
        progress = tqdm(dataloader, desc="Treino")
        for batch in progress:
            # Move para device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            
            # Forward pass
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            
            loss = outputs.loss
            total_loss += loss.item()
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            optimizer.step()
            scheduler.step()
            
            # Atualizar progress bar
            progress.set_postfix({'loss': loss.item()})
            
        return total_loss / len(dataloader)
    
    def avaliar(self, dataloader):
        """
        Avalia o modelo
        """
        self.model.eval()
        total_correct = 0
        total_samples = 0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Avaliação"):
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask
                )
                
                predictions = torch.argmax(outputs.logits, dim=1)
                total_correct += (predictions == labels).sum().item()
                total_samples += labels.size(0)
        
        accuracy = total_correct / total_samples
        return accuracy
    
    def treinar_completo(self, train_loader, val_loader, epochs=3):
        """
        Pipeline completo
        """
        # Optimizer
        optimizer = AdamW(self.model.parameters(), lr=2e-5)
        
        # Scheduler
        total_steps = len(train_loader) * epochs
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=0,
            num_training_steps=total_steps
        )
        
        # Treino
        for epoch in range(epochs):
            print(f"\n📚 Epoch {epoch+1}/{epochs}")
            
            train_loss = self.treinar_epoch(train_loader, optimizer, scheduler)
            val_accuracy = self.avaliar(val_loader)
            
            print(f"Train Loss: {train_loss:.4f}")
            print(f"Val Accuracy: {val_accuracy:.4f}")
        
        # Salvar modelo
        self.model.save_pretrained('modelo_nlp_final')
        self.tokenizer.save_pretrained('modelo_nlp_final')
        
        print("\n✅ Modelo salvo em 'modelo_nlp_final/'")

# Uso
treinador = TreinadorNLP(num_labels=2, device='mps')
treinador.treinar_completo(train_loader, val_loader, epochs=3)
```

---

### 3.3 Modelos Tabulares

Para dados estruturados/tabulares, o M1 é excelente!

#### XGBoost, LightGBM, CatBoost

**Por que usar gradient boosting no M1?**
- ✅ Extremamente eficientes em memória
- ✅ Não precisam GPU (CPU M1 é muito rápida)
- ✅ Performance state-of-the-art em dados tabulares
- ✅ Treinam muito rápido

**Instalação:**

```bash
pip install xgboost lightgbm catboost
```

**Exemplo completo:**

```python
# modelos_tabulares.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import xgboost as xgb
import lightgbm as lgb
from catboost import CatBoostClassifier

class TreinadorTabular:
    def __init__(self, X, y):
        self.X_train, self.X_test, self.y_train, self.y_test = \
            train_test_split(X, y, test_size=0.2, random_state=42)
    
    def treinar_xgboost(self):
        """XGBoost - Muito popular"""
        print("🚀 Treinando XGBoost...")
        
        model = xgb.XGBClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            tree_method='hist',  # Mais rápido
            early_stopping_rounds=50,
            eval_metric='logloss'
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=50
        )
        
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"✓ XGBoost Accuracy: {acc:.4f}\n")
        
        return model, acc
    
    def treinar_lightgbm(self):
        """LightGBM - Mais rápido"""
        print("⚡ Treinando LightGBM...")
        
        model = lgb.LGBMClassifier(
            n_estimators=1000,
            learning_rate=0.05,
            num_leaves=31,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            early_stopping_rounds=50
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=[(self.X_test, self.y_test)],
            verbose=50
        )
        
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"✓ LightGBM Accuracy: {acc:.4f}\n")
        
        return model, acc
    
    def treinar_catboost(self):
        """CatBoost - Melhor para categóricas"""
        print("🐱 Treinando CatBoost...")
        
        model = CatBoostClassifier(
            iterations=1000,
            learning_rate=0.05,
            depth=6,
            early_stopping_rounds=50,
            verbose=50
        )
        
        model.fit(
            self.X_train, self.y_train,
            eval_set=(self.X_test, self.y_test)
        )
        
        preds = model.predict(self.X_test)
        acc = accuracy_score(self.y_test, preds)
        print(f"✓ CatBoost Accuracy: {acc:.4f}\n")
        
        return model, acc
    
    def comparar_todos(self):
        """Compara os 3 modelos"""
        resultados = {}
        
        resultados['XGBoost'] = self.treinar_xgboost()
        resultados['LightGBM'] = self.treinar_lightgbm()
        resultados['CatBoost'] = self.treinar_catboost()
        
        print("\n" + "="*50)
        print("COMPARAÇÃO FINAL")
        print("="*50)
        for nome, (modelo, acc) in resultados.items():
            print(f"{nome:12s}: {acc:.4f}")
        
        # Melhor modelo
        melhor = max(resultados.items(), key=lambda x: x[1][1])
        print(f"\n🏆 Melhor modelo: {melhor[0]} ({melhor[1][1]:.4f})")
        
        return resultados

# Uso com dataset de exemplo
from sklearn.datasets import make_classification

X, y = make_classification(
    n_samples=10000,
    n_features=20,
    n_informative=15,
    n_redundant=5,
    random_state=42
)

treinador = TreinadorTabular(X, y)
resultados = treinador.comparar_todos()
```

#### Redes Neurais para Dados Tabulares

```python
# nn_tabular.py
import tensorflow as tf
from tensorflow import keras

def criar_nn_tabular(input_dim, num_classes):
    """
    Rede neural simples para dados tabulares
    """
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        
        # BatchNorm ajuda muito em tabulares
        keras.layers.BatchNormalization(),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(0.3),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        
        keras.layers.BatchNormalization(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dropout(0.2),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    return model

# Treinar
model = criar_nn_tabular(input_dim=20, num_classes=2)
history = model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=100,
    batch_size=128,
    callbacks=[
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
    ]
)
```

---

### 💡 Dicas de Performance no M1

**Para Imagens:**
- Use MobileNet/EfficientNet como base
- Batch size 32-64 é ideal
- Mixed precision economiza ~40% memória

**Para NLP:**
- DistilBERT > BERT para M1 16GB
- max_length=128 em vez de 512
- Batch size 16-32

**Para Tabulares:**
- LightGBM geralmente é o mais rápido no M1
- Aproveita CPU multicores muito bem
- Não precisa GPU!

---

### ✅ Checklist Módulo 3

- [ ] Treinei classificador de imagens com transfer learning
- [ ] Entendo diferença entre fase 1 (frozen) e fase 2 (fine-tuning)
- [ ] Consigo treinar modelo NLP com DistilBERT
- [ ] Sei usar XGBoost/LightGBM/CatBoost
- [ ] Mixed precision ativado nos modelos
- [ ] Callbacks implementados (Early Stopping, ReduceLR)

---

### 🎯 Próximos Passos

No **Módulo 4**, vamos aprender técnicas avançadas: quantização, pruning, e como comprimir modelos para ficarem ainda mais eficientes!

## Módulo 4: Técnicas Avançadas de Otimização

### 4.1 Quantização de Modelos

#### O que é Quantização?

**Quantização** = Reduzir a precisão dos números no modelo para usar menos memória.

**Tipos de números em deep learning:**
```
FP32 (Float32):     4 bytes  ← Padrão
FP16 (Float16):     2 bytes  ← Mixed precision
INT8 (Integer 8):   1 byte   ← Quantização
INT4 (Integer 4):   0.5 byte ← Quantização extrema
```

**Economia de Memória:**
- Modelo de 1GB em FP32 → 250MB em INT8 (75% redução!)
- Modelo de 1GB em FP32 → 125MB em INT4 (87.5% redução!)

**Trade-off:**
- ✅ Muito menos memória
- ✅ Inferência mais rápida
- ⚠️ Pequena perda de precisão (1-3% tipicamente)

#### Post-Training Quantization (PTQ)

Quantiza um modelo **já treinado** - mais simples e rápido!

**TensorFlow Lite (excelente para M1):**

```python
# quantizacao_tensorflow.py
import tensorflow as tf
from tensorflow import keras
import numpy as np

def quantizar_modelo_int8(modelo_path, dados_calibracao):
    """
    Quantiza modelo para INT8 usando TensorFlow Lite
    Ideal para deployment no M1
    """
    # Carregar modelo
    model = keras.models.load_model(modelo_path)
    
    # Converter para TFLite com quantização INT8
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    
    # Configurar quantização completa INT8
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.int8]
    
    # Dataset representativo para calibração
    def representative_dataset():
        for data in dados_calibracao:
            yield [data.astype(np.float32)]
    
    converter.representative_dataset = representative_dataset
    
    # Quantizar!
    tflite_model = converter.convert()
    
    # Guardar modelo quantizado
    with open('modelo_int8.tflite', 'wb') as f:
        f.write(tflite_model)
    
    # Comparar tamanhos
    tamanho_original = os.path.getsize(modelo_path) / (1024**2)
    tamanho_quantizado = len(tflite_model) / (1024**2)
    reducao = (1 - tamanho_quantizado/tamanho_original) * 100
    
    print(f"✓ Modelo quantizado com sucesso!")
    print(f"  Original:    {tamanho_original:.2f} MB")
    print(f"  Quantizado:  {tamanho_quantizado:.2f} MB")
    print(f"  Redução:     {reducao:.1f}%")
    
    return tflite_model

# Exemplo de uso
# Preparar dados de calibração (amostra do dataset de treino)
x_train_sample = x_train[:1000]  # 1000 amostras

# Quantizar
modelo_quantizado = quantizar_modelo_int8('modelo_final.keras', x_train_sample)
```

**Inferência com modelo quantizado:**

```python
# inferencia_quantizada.py
import tensorflow as tf
import numpy as np

def inferencia_tflite(modelo_path, input_data):
    """
    Executa inferência com modelo TFLite quantizado
    """
    # Carregar interpretador TFLite
    interpreter = tf.lite.Interpreter(model_path=modelo_path)
    interpreter.allocate_tensors()
    
    # Obter detalhes de input/output
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    # Preparar input
    input_data = input_data.astype(np.float32)
    interpreter.set_tensor(input_details[0]['index'], input_data)
    
    # Executar
    interpreter.invoke()
    
    # Obter resultado
    output_data = interpreter.get_tensor(output_details[0]['index'])
    
    return output_data

# Uso
predictions = inferencia_tflite('modelo_int8.tflite', test_images)
```

**PyTorch - Quantização Dinâmica:**

```python
# quantizacao_pytorch.py
import torch
import torch.quantization

def quantizar_modelo_pytorch(model):
    """
    Quantização dinâmica para PyTorch
    Mais simples, sem necessidade de calibração
    """
    # Modelo para CPU (quantização funciona melhor em CPU)
    model_cpu = model.cpu()
    
    # Quantização dinâmica (pesos INT8, ativações FP32)
    model_quantized = torch.quantization.quantize_dynamic(
        model_cpu,
        {torch.nn.Linear, torch.nn.Conv2d},  # Camadas a quantizar
        dtype=torch.qint8
    )
    
    # Comparar tamanhos
    def get_model_size(model):
        torch.save(model.state_dict(), "temp.p")
        size = os.path.getsize("temp.p") / (1024**2)
        os.remove("temp.p")
        return size
    
    size_original = get_model_size(model_cpu)
    size_quantized = get_model_size(model_quantized)
    reducao = (1 - size_quantized/size_original) * 100
    
    print(f"✓ Modelo quantizado!")
    print(f"  Original:    {size_original:.2f} MB")
    print(f"  Quantizado:  {size_quantized:.2f} MB")
    print(f"  Redução:     {reducao:.1f}%")
    
    return model_quantized

# Uso
model_quantized = quantizar_modelo_pytorch(model)

# Testar velocidade
import time

x = torch.randn(1, 3, 224, 224)

# Original
start = time.time()
_ = model.cpu()(x)
tempo_original = time.time() - start

# Quantizado
start = time.time()
_ = model_quantized(x)
tempo_quantizado = time.time() - start

print(f"\nVelocidade:")
print(f"  Original:    {tempo_original*1000:.2f} ms")
print(f"  Quantizado:  {tempo_quantizado*1000:.2f} ms")
print(f"  Speedup:     {tempo_original/tempo_quantizado:.2f}x")
```

#### Quantization-Aware Training (QAT)

Treinar o modelo **já a simular quantização** - melhor precisão!

```python
# qat_tensorflow.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def treinar_com_qat(model, train_dataset, val_dataset, epochs=10):
    """
    Quantization-Aware Training
    Modelo aprende a compensar quantização durante treino
    """
    # Aplicar quantização ao modelo
    quantize_model = tfmot.quantization.keras.quantize_model
    q_aware_model = quantize_model(model)
    
    # Compilar
    q_aware_model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("🎯 Treinando com Quantization-Aware Training...")
    
    # Treinar
    history = q_aware_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=3),
            tf.keras.callbacks.ReduceLROnPlateau(patience=2)
        ]
    )
    
    # Converter para TFLite INT8
    converter = tf.lite.TFLiteConverter.from_keras_model(q_aware_model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # Guardar
    with open('modelo_qat_int8.tflite', 'wb') as f:
        f.write(tflite_model)
    
    print("✓ Modelo QAT treinado e quantizado!")
    
    return q_aware_model, tflite_model

# Instalação necessária:
# pip install tensorflow-model-optimization
```

#### Comparação: PTQ vs QAT

| Método | Facilidade | Tempo | Precisão | Quando Usar |
|--------|-----------|-------|----------|-------------|
| **PTQ** | ⭐⭐⭐ | Rápido | Boa (-1-2%) | Modelo já treinado |
| **QAT** | ⭐⭐ | Lento | Melhor (-0.5%) | Máxima precisão |

**Recomendação para M1 16GB:**
- Começa com **PTQ** (post-training)
- Se perderes muita precisão, tenta **QAT**

---

### 4.2 Pruning e Compressão

#### O que é Pruning?

**Pruning** = Remover pesos/neurónios menos importantes do modelo.

**Analogia:** É como podar uma árvore - removes ramos que não contribuem muito.

```
Modelo Original:           Modelo com Pruning:
████████████████          ██░░██░░░░██░░██
████████████████    →     ███░░░██░░░░░███
████████████████          ░░██████░░██████
                          (░ = peso removido/zero)
```

**Vantagens:**
- ✅ Modelo mais pequeno
- ✅ Inferência mais rápida
- ✅ Menos memória

#### Magnitude-Based Pruning

Remove pesos com menor valor absoluto:

```python
# pruning_tensorflow.py
import tensorflow as tf
import tensorflow_model_optimization as tfmot

def pruning_progressivo(model, train_dataset, val_dataset):
    """
    Pruning progressivo - remove pesos gradualmente durante treino
    """
    # Definir schedule de pruning
    pruning_params = {
        'pruning_schedule': tfmot.sparsity.keras.PolynomialDecay(
            initial_sparsity=0.0,      # Começa sem pruning
            final_sparsity=0.5,        # Termina com 50% dos pesos removidos
            begin_step=0,
            end_step=1000              # Ao longo de 1000 steps
        )
    }
    
    # Aplicar pruning ao modelo
    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(
        model, 
        **pruning_params
    )
    
    # Compilar
    model_for_pruning.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # Callbacks necessários
    callbacks = [
        tfmot.sparsity.keras.UpdatePruningStep(),  # Atualiza máscara de pruning
        tfmot.sparsity.keras.PruningSummaries(log_dir='logs'),  # Logs
        tf.keras.callbacks.EarlyStopping(patience=3)
    ]
    
    print("✂️ Treinando com Pruning Progressivo...")
    
    # Treinar
    history = model_for_pruning.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=10,
        callbacks=callbacks
    )
    
    # Remover wrappers de pruning
    model_final = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    
    # Estatísticas
    sparsity = calcular_sparsity(model_final)
    print(f"\n✓ Pruning concluído!")
    print(f"  Sparsity: {sparsity:.1f}% (pesos zero)")
    
    return model_final

def calcular_sparsity(model):
    """Calcula percentagem de pesos zero"""
    total_weights = 0
    zero_weights = 0
    
    for layer in model.layers:
        if hasattr(layer, 'kernel'):
            weights = layer.kernel.numpy()
            total_weights += weights.size
            zero_weights += np.sum(weights == 0)
    
    return zero_weights / total_weights if total_weights > 0 else 0
```

**Combinar Pruning + Quantização:**

```python
# pruning_e_quantizacao.py
def otimizar_modelo_completo(model, train_dataset, val_dataset):
    """
    Pipeline completo: Pruning → Fine-tuning → Quantização
    Máxima compressão!
    """
    print("🎯 PASSO 1: Pruning")
    model_pruned = self.aplicar_pruning(train_dataset, val_dataset)
    
    print("\n🎯 PASSO 2: Fine-tuning")
    model_pruned.fit(train_dataset, validation_data=val_dataset, epochs=5, verbose=0)
    
    print("\n🎯 PASSO 3: Quantização")
    # Guardar temporariamente
    model_pruned.save('temp_pruned.keras')
    
    # Quantizar
    model_final = self.aplicar_quantizacao('temp_pruned.keras', train_dataset)
    
    # Comparar
    tamanho_original = get_model_size(model)
    tamanho_final = len(model_final) / (1024**2)
    reducao = (1 - tamanho_final/tamanho_original) * 100
    
    print(f"\n✅ OTIMIZAÇÃO COMPLETA!")
    print(f"  Original:  {tamanho_original:.2f} MB")
    print(f"  Final:     {tamanho_final:.2f} MB")
    print(f"  Redução:   {reducao:.1f}%")
    
    return model_final

# Uso
modelo_otimizado = otimizar_modelo_completo(model, train_ds, val_ds)
```

#### Knowledge Distillation

Transferir conhecimento de um modelo grande (professor) para um pequeno (aluno):

```python
# knowledge_distillation.py
import tensorflow as tf
from tensorflow import keras

class DistillationModel(keras.Model):
    """
    Modelo que aprende tanto com labels reais como com professor
    """
    def __init__(self, aluno, professor, alpha=0.1, temperatura=3):
        super().__init__()
        self.aluno = aluno
        self.professor = professor
        self.alpha = alpha          # Peso do professor
        self.temperatura = temperatura
        
    def compile(self, optimizer, metrics):
        super().compile(optimizer=optimizer, metrics=metrics)
        self.loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
        
    def train_step(self, data):
        x, y = data
        
        # Previsões do professor (soft targets)
        professor_predictions = self.professor(x, training=False)
        
        with tf.GradientTape() as tape:
            # Previsões do aluno
            aluno_predictions = self.aluno(x, training=True)
            
            # Loss com labels reais (hard targets)
            loss_hard = self.loss_fn(y, aluno_predictions)
            
            # Loss com professor (soft targets)
            # Temperatura suaviza probabilidades
            soft_aluno = tf.nn.softmax(aluno_predictions / self.temperatura)
            soft_professor = tf.nn.softmax(professor_predictions / self.temperatura)
            
            loss_soft = tf.reduce_mean(
                tf.keras.losses.categorical_crossentropy(
                    soft_professor, soft_aluno
                )
            ) * (self.temperatura ** 2)
            
            # Loss combinada
            loss_total = self.alpha * loss_soft + (1 - self.alpha) * loss_hard
        
        # Backprop
        gradients = tape.gradient(loss_total, self.aluno.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.aluno.trainable_variables))
        
        # Atualizar métricas
        self.compiled_metrics.update_state(y, aluno_predictions)
        
        return {m.name: m.result() for m in self.metrics}

# Uso
# Professor: Modelo grande e preciso
professor = keras.applications.ResNet50(weights='imagenet', include_top=True)
professor.trainable = False

# Aluno: Modelo pequeno
aluno = keras.applications.MobileNetV2(weights=None, include_top=True)

# Distillation
distiller = DistillationModel(aluno, professor, alpha=0.1, temperatura=3)
distiller.compile(optimizer='adam', metrics=['accuracy'])

# Treinar aluno com conhecimento do professor
distiller.fit(train_dataset, validation_data=val_dataset, epochs=20)

# Guardar apenas o aluno
aluno.save('modelo_aluno.keras')
```

---

### 4.3 Treino Eficiente

#### Learning Rate Scheduling

Ajustar learning rate durante treino para melhor convergência:

```python
# lr_schedulers.py
import tensorflow as tf
from tensorflow import keras

# 1. ReduceLROnPlateau (reduz quando estagnar)
reduce_lr = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,           # Multiplica LR por 0.5
    patience=3,           # Após 3 epochs sem melhoria
    min_lr=1e-7,         # LR mínimo
    verbose=1
)

# 2. CosineDecay (redução suave)
lr_schedule = keras.optimizers.schedules.CosineDecay(
    initial_learning_rate=1e-3,
    decay_steps=1000,
    alpha=0.1  # LR final = initial × alpha
)

optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)

# 3. ExponentialDecay
lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.96  # Multiplica por 0.96 a cada 100 steps
)

# 4. Warm-up + Decay (personalizado)
class WarmUpCosineDecay(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, warmup_steps, total_steps, initial_lr, target_lr):
        super().__init__()
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.initial_lr = initial_lr
        self.target_lr = target_lr
        
    def __call__(self, step):
        if step < self.warmup_steps:
            # Fase warm-up: aumenta LR linearmente
            return (step / self.warmup_steps) * self.initial_lr
        else:
            # Fase decay: reduz com cosine
            progress = (step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            cosine_decay = 0.5 * (1 + tf.cos(tf.constant(3.14159) * progress))
            return self.target_lr + (self.initial_lr - self.target_lr) * cosine_decay

# Uso
lr_schedule = WarmUpCosineDecay(
    warmup_steps=1000,
    total_steps=10000,
    initial_lr=1e-3,
    target_lr=1e-6
)
optimizer = keras.optimizers.Adam(learning_rate=lr_schedule)
```

#### Early Stopping

Para quando o modelo não melhora mais:

```python
# early_stopping.py
early_stop = keras.callbacks.EarlyStopping(
    monitor='val_loss',        # Métrica a monitorizar
    patience=10,               # Espera 10 epochs
    restore_best_weights=True, # Volta aos melhores pesos
    verbose=1,
    min_delta=0.001           # Melhoria mínima considerada
)

# Uso
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=100,  # Máximo (vai parar antes!)
    callbacks=[early_stop]
)
```

#### Checkpointing Estratégico

Guarda apenas os melhores modelos:

```python
# checkpointing.py
# 1. Guardar melhor modelo
checkpoint_best = keras.callbacks.ModelCheckpoint(
    filepath='melhor_modelo.keras',
    monitor='val_accuracy',
    save_best_only=True,
    mode='max',
    verbose=1
)

# 2. Guardar periodicamente
checkpoint_periodic = keras.callbacks.ModelCheckpoint(
    filepath='checkpoint_epoch_{epoch:02d}.keras',
    save_freq='epoch',
    period=5  # A cada 5 epochs
)

# 3. Backup inteligente (guarda top-3)
from tensorflow.keras.callbacks import Callback
import os

class TopKCheckpoint(Callback):
    def __init__(self, filepath, monitor='val_accuracy', k=3):
        super().__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.k = k
        self.best_models = []  # (score, filepath)
        
    def on_epoch_end(self, epoch, logs=None):
        current = logs.get(self.monitor)
        filepath = self.filepath.format(epoch=epoch, **logs)
        
        # Guardar modelo
        self.model.save(filepath)
        
        # Adicionar à lista
        self.best_models.append((current, filepath))
        self.best_models.sort(reverse=True, key=lambda x: x[0])
        
        # Manter apenas top-k
        if len(self.best_models) > self.k:
            _, to_delete = self.best_models.pop()
            if os.path.exists(to_delete):
                os.remove(to_delete)
                print(f"\n🗑️  Removido: {to_delete}")

# Uso
top3 = TopKCheckpoint('modelo_epoch{epoch:02d}_acc{val_accuracy:.4f}.keras', k=3)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=[checkpoint_best, top3]
)
```

---

### 💡 Pipeline Completo de Otimização para M1

```python
# pipeline_otimizacao_m1.py
class PipelineOtimizacaoM1:
    """
    Pipeline completo de otimização para M1 16GB
    """
    def __init__(self, model):
        self.model_original = model
        self.model_otimizado = None
        
    def otimizar_completo(self, train_ds, val_ds):
        """
        Aplica todas as otimizações na sequência ideal
        """
        print("🚀 INICIANDO PIPELINE DE OTIMIZAÇÃO\n")
        
        # 1. Mixed Precision
        print("📊 PASSO 1: Mixed Precision (FP16)")
        policy = keras.mixed_precision.Policy('mixed_float16')
        keras.mixed_precision.set_global_policy(policy)
        print("✓ Ativado\n")
        
        # 2. Pruning
        print("✂️ PASSO 2: Pruning (50% sparsity)")
        model_pruned = self.aplicar_pruning(train_ds, val_ds)
        print("✓ Concluído\n")
        
        # 3. Fine-tuning
        print("🎯 PASSO 3: Fine-tuning")
        model_pruned.fit(train_ds, validation_data=val_ds, epochs=5, verbose=0)
        print("✓ Concluído\n")
        
        # 4. Quantização
        print("🔢 PASSO 4: Quantização INT8")
        model_pruned.save('temp_pruned.keras')
        model_quantizado = self.aplicar_quantizacao('temp_pruned.keras', train_ds)
        print("✓ Concluído\n")
        
        # Relatório final
        self.gerar_relatorio(model_quantizado)
        
        self.model_otimizado = model_quantizado
        return model_quantizado
    
    def aplicar_pruning(self, train_ds, val_ds):
        # ... (código de pruning do exemplo anterior)
        pass
    
    def aplicar_quantizacao(self, model_path, train_ds):
        # ... (código de quantização do exemplo anterior)
        pass
    
    def gerar_relatorio(self, model_final):
        print("\n" + "=" * 60)
        print("RELATÓRIO FINAL DE OTIMIZAÇÃO")
        print("=" * 60)
        # Comparações de tamanho, velocidade, precisão
        pass

# Uso
pipeline = PipelineOtimizacaoM1(model)
modelo_otimizado = pipeline.otimizar_completo(train_dataset, val_dataset)
```

---

### ✅ Checklist Módulo 4

- [ ] Entendo diferença entre FP32, FP16, INT8
- [ ] Consigo fazer quantização post-training (PTQ)
- [ ] Sei aplicar pruning progressivo
- [ ] Conheço knowledge distillation
- [ ] Implementei learning rate scheduling
- [ ] Uso early stopping e checkpointing
- [ ] Consigo combinar técnicas (pruning + quantização)

---

### 🎯 Próximos Passos

No **Módulo 5**, vamos finalmente trabalhar com LLMs (Large Language Models)! Vais aprender a fazer fine-tuning de modelos até 7B parâmetros no M1 16GB usando LoRA e QLoRA!

## Módulo 5: Modelos de Linguagem Grandes (LLMs)

### 5.1 Limitações e Estratégias

#### Por que 16GB é Limitante para LLMs?

**Requisitos típicos de memória para LLMs:**

| Modelo | Parâmetros | FP32 | FP16 | INT8 | INT4 |
|--------|-----------|------|------|------|------|
| GPT-2 Small | 124M | 500MB | 250MB | 125MB | 63MB ✅ |
| GPT-2 Medium | 355M | 1.4GB | 700MB | 350MB | 175MB ✅ |
| LLaMA-2 7B | 7B | 28GB ❌ | 14GB ⚠️ | 7GB ✅ | 3.5GB ✅ |
| LLaMA-2 13B | 13B | 52GB ❌ | 26GB ❌ | 13GB ⚠️ | 6.5GB ✅ |
| Mistral 7B | 7B | 28GB ❌ | 14GB ⚠️ | 7GB ✅ | 3.5GB ✅ |

**Realidade no M1 16GB:**
- Sistema operativo: ~3-4GB
- Apps em background: ~1-2GB
- **Disponível para modelo: ~10-12GB**

**Conclusão:**
- ✅ Modelos até 7B: Viáveis com quantização INT4/INT8
- ⚠️ Modelos 13B: Possível apenas com INT4 + otimizações
- ❌ Modelos 30B+: Impossível carregar completos na memória

#### Estratégias para Trabalhar com LLMs no M1

**1. Quantização Extrema (4-bit, GGUF)**

GGUF = GPT-Generated Unified Format (sucessor de GGML)
- Formato otimizado para inferência em CPU/GPU unificada
- Quantização 4-bit com grupos (mantém qualidade)
- Ideal para Apple Silicon

**2. LoRA (Low-Rank Adaptation)**

Em vez de treinar todos os parâmetros, adiciona matrizes pequenas:
```
Modelo original:  7B parâmetros (frozen)
     +
LoRA adapters:    ~10M parâmetros (treináveis)
     =
Modelo fine-tuned mantendo 99.9% frozen!
```

**Economia de memória:**
- Treino normal de 7B: ~40GB necessários (gradientes + optimizador)
- Treino com LoRA: ~12GB necessários ✅

**3. QLoRA (Quantized LoRA)**

Combina quantização + LoRA:
```
Base model em INT4:  3.5GB
LoRA params em FP16: 20MB
Gradientes:          ~2GB
Total:               ~6GB ✅ Cabe no M1!
```

#### Formato GGUF - Explicação Técnica

```python
# compreender_gguf.py
"""
GGUF (GPT-Generated Unified Format)

Características:
- Formato binário otimizado
- Suporta quantização 2-bit até 8-bit
- Metadados incluídos (arquitetura, vocabulário)
- Carregamento rápido (mmap)
- Ideal para llama.cpp e MLX
"""

# Tipos de quantização GGUF:
quantizacoes = {
    'Q2_K': '2-bit, muito compacto, perda qualidade',
    'Q3_K_S': '3-bit small, bom compromisso',
    'Q4_0': '4-bit, recomendado geral',
    'Q4_K_M': '4-bit medium, melhor qualidade',
    'Q5_0': '5-bit, alta qualidade',
    'Q5_K_M': '5-bit medium, muito boa',
    'Q8_0': '8-bit, quase sem perda',
}

# Escolha para M1 16GB:
# - Q4_K_M: Melhor equilíbrio qualidade/tamanho
# - Q5_K_M: Se tiveres espaço extra
# - Q8_0: Apenas modelos pequenos (<3B)
```

---

### 5.2 Fine-tuning de LLMs Pequenos

#### Cuidados com Datasets e Formatos Ideais

**⚠️ CRÍTICO: A qualidade do dataset é mais importante que a quantidade!**

#### Princípios Fundamentais

**1. Qualidade sobre Quantidade**

```python
# Exemplo de dataset de baixa qualidade:
# Respostas genéricas, erros, etc.

# Exemplo de dataset de alta qualidade:
# Respostas precisas, bem escritas, diversas.
```

**2. Diversidade é Essencial**

```python
# analisar_diversidade.py
from collections import Counter
import numpy as np

def analisar_diversidade(dataset):
    """
    Analisa diversos aspetos do dataset
    """
    # ... (código de análise de comprimento, palavras iniciais, tópicos, duplicados)
    pass

# Se uma palavra aparece >30%, há falta de diversidade!
# Se categorias estão desbalanceadas, usar balancear_categorias()
```

**3. Balanceamento**

```python
# balancear_dataset.py
from collections import Counter
import random

def balancear_categorias(dataset, max_por_categoria=100):
    """
    Balanceia dataset para evitar overfitting em categorias sobre-representadas
    """
    # ... (código para agrupar por categoria e amostrar)
    pass

# Uso
dataset_balanceado = balancear_categorias(dataset, max_por_categoria=200)
```

#### Formatos de Dataset Ideais

**Formato 1: Alpaca (Mais Comum)**

```json
{
  "instruction": "Explica o que é aprendizagem automática",
  "input": "",
  "output": "Aprendizagem automática é um ramo da inteligência artificial que permite aos computadores aprender a partir de dados sem serem explicitamente programados..."
}
```

**Quando usar:**
- Fine-tuning de instruções
- Q&A geral
- Tarefas de seguir comandos

**Formato em texto:**

```python
# formato_alpaca.py
def formatar_prompt(instrucao, resposta, contexto=""):
    """Formato Alpaca padrão"""
    if contexto:
        prompt = f"### Instrução:\n{instrucao}\n\n### Contexto:\n{contexto}\n\n### Resposta:\n{resposta}" 
    else:
        prompt = f"### Instrução:\n{instrucao}\n\n### Resposta:\n{resposta}"
    
    return prompt

# Exemplo
exemplo = formatar_prompt(
    instrucao="Traduz para inglês: Olá, como estás?",
    resposta="Hello, how are you?"
)
print(exemplo)
```

**Formato 2: ChatML (Para Conversas)**

```json
{
  "messages": [
    {"role": "system", "content": "És um assistente prestável em português."},
    {"role": "user", "content": "Como fazer um bolo?"},
    {"role": "assistant", "content": "Para fazer um bolo: 1. ..."}
  ]
}
```

**Quando usar:**
- Conversas multi-turn
- Assistentes com personalidade
- Contexto de sistema importante

**Implementação:**

```python
# formato_chatml.py
def formatar_chatml(mensagens):
    """
    Formato ChatML (usado por GPT, Mistral, etc.)
    """
    prompt = ""
    
    for msg in mensagens:
        role = msg['role']
        content = msg['content']
        
        if role == "system":
            prompt += f"<|im_start|>system\n{content}<|im_end|>\n"
        elif role == "user":
            prompt += f"<|im_start|>user\n{content}<|im_end|>\n"
        elif role == "assistant":
            prompt += f"<|im_start|>assistant\n{content}<|im_end|>\n"
    
    return prompt

# Exemplo de conversa
conversa = [
    {"role": "system", "content": "És um tutor de matemática."},
    {"role": "user", "content": "Quanto é 2+2?"},
    {"role": "assistant", "content": "2+2 = 4"},
    {"role": "user", "content": "E 5*3?"},
    {"role": "assistant", "content": "5*3 = 15"}
]

prompt = formatar_chatml(conversa)
```

**Formato 3: Completion (Mais Simples)**

```json
{
  "text": "Pergunta: O que é Python?\n\nResposta: Python é uma linguagem..."
}
```

**Quando usar:**
- Datasets muito simples
- Continuar texto
- Estilo literário

#### Tamanho Ideal do Dataset

**Regras gerais:**

| Tipo de Fine-tuning | Mínimo | Recomendado | Ideal |
|---------------------|--------|-------------|-------|
| **Adaptação de estilo** | 50 | 200 | 500 |
| **Domínio específico** | 200 | 1.000 | 5.000 |
| **Novo conhecimento** | 1.000 | 5.000 | 10.000+ |
| **Mudança de comportamento** | 500 | 2.000 | 5.000 |

**⚠️ IMPORTANTE: Com LoRA no M1, datasets grandes (>5k) podem ser problemáticos**

```python
# calcular_tempo_treino.py
def estimar_tempo_treino(num_exemplos, epochs=3, batch_size=4):
    """
    Estima tempo de treino no M1 16GB
    """
    steps_por_epoch = num_exemplos // batch_size
    total_steps = steps_por_epoch * epochs
    
    # M1 processa ~2-3 steps/segundo para LLaMA 7B com LoRA
    tempo_segundos = total_steps / 2.5
    
    horas = tempo_segundos / 3600
    
    print(f"📊 Estimativa de Treino")
    print(f"  Exemplos: {num_exemplos}")
    print(f"  Epochs: {epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Total steps: {total_steps}")
    print(f"  Tempo estimado: {horas:.1f} horas")
    
    if horas > 12:
        print(f"\n⚠️  Considera reduzir dataset ou epochs!")

# Testa
estimar_tempo_treino(num_exemplos=5000, epochs=3, batch_size=4)
```

#### Cuidados Críticos

**1. Evitar Data Leakage**

```python
# split_correto.py
from sklearn.model_selection import train_test_split

def split_dataset_correto(dataset):
    """
    Split correto para evitar leakage
    """
    # ❌ ERRADO: Split aleatório se houver conversas multi-turn
    # Uma parte da conversa no treino, outra no teste = leakage!
    
    # ✅ CORRETO: Split por "conversa_id" ou "user_id"
    
    # Agrupar por conversa
    conversas = {}
    for exemplo in dataset:
        conv_id = exemplo.get('conversa_id', exemplo.get('id'))
        if conv_id not in conversas:
            conversas[conv_id] = []
        conversas[conv_id].append(exemplo)
    
    # Split por conversas completas
    conv_ids = list(conversas.keys())
    train_ids, test_ids = train_test_split(conv_ids, test_size=0.2, random_state=42)
    
    train_data = []
    test_data = []
    
    for conv_id in train_ids:
        train_data.extend(conversas[conv_id])
    
    for conv_id in test_ids:
        test_data.extend(conversas[conv_id])
    
    print(f"✓ Split correto:")
    print(f"  Treino: {len(train_data)} exemplos ({len(train_ids)} conversas)")
    print(f"  Teste:  {len(test_data)} exemplos ({len(test_ids)} conversas)")
    
    return train_data, test_data
```

**2. Limpeza de Dados**

```python
# limpar_dataset.py
import re
import unicodedata

def limpar_texto(texto):
    """
    Limpa texto removendo problemas comuns
    """
    # Normalizar unicode
    texto = unicodedata.normalize('NFKC', texto)
    
    # Remover espaços múltiplos
    texto = re.sub(r'\s+', ' ', texto)
    
    # Remover caracteres de controlo (excepto \n)
    texto = ''.join(char for char in texto if char == '\n' or not unicodedata.category(char).startswith('C'))
    
    # Remover linhas vazias múltiplas
    texto = re.sub(r'\n\s*\n', '\n\n', texto)
    
    # Trim
    texto = texto.strip()
    
    return texto

def validar_exemplo(exemplo, min_length=10, max_length=2048):
    """
    Valida se exemplo é adequado
    """
    instrucao = exemplo.get('instrucao', '')
    resposta = exemplo.get('resposta', '')
    
    # Verificações
    if len(resposta) < min_length:
        return False, "Resposta muito curta"
    
    if len(resposta) > max_length:
        return False, "Resposta muito longa"
    
    if not instrucao or not resposta:
        return False, "Campos vazios"
    
    # Verificar se não é só pontuação
    if len(re.findall(r'\w+', resposta)) < 3:
        return False, "Resposta sem conteúdo"
    
    # Verificar URLs suspeitas (possível spam)
    if len(re.findall(r'http[s]?:\/\/(?:[a-zA-Z]|[0-9]|[$-_@.&+])+', resposta)) > 2:
        return False, "Muitas URLs"
    
    return True, "OK"

def limpar_dataset(dataset):
    """
    Limpa e valida dataset completo
    """
    dataset_limpo = []
    estatisticas = {'removidos': 0, 'mantidos': 0, 'motivos': {}}
    
    for exemplo in dataset:
        # Limpar textos
        if 'instrucao' in exemplo:
            exemplo['instrucao'] = limpar_texto(exemplo['instrucao'])
        if 'resposta' in exemplo:
            exemplo['resposta'] = limpar_texto(exemplo['resposta'])
        
        # Validar
        valido, motivo = validar_exemplo(exemplo)
        
        if valido:
            dataset_limpo.append(exemplo)
            estatisticas['mantidos'] += 1
        else:
            estatisticas['removidos'] += 1
            estatisticas['motivos'][motivo] = estatisticas['motivos'].get(motivo, 0) + 1
    
    # Relatório
    print("🧹 LIMPEZA DE DATASET")
    print("=" * 60)
    print(f"Total original: {len(dataset)}")
    print(f"Mantidos:       {estatisticas['mantidos']}")
    print(f"Removidos:      {estatisticas['removidos']}")
    
    if estatisticas['motivos']:
        print(f"\nMotivos de remoção:")
        for motivo, count in estatisticas['motivos'].items():
            print(f"  {motivo}: {count}")
    
    return dataset_limpo

# Uso
dataset_limpo = limpar_dataset(dataset_original)
```

**3. Tokenização e Comprimento**

```python
# analisar_tokens.py
from transformers import AutoTokenizer

def analisar_comprimento_tokens(dataset, model_name="meta-llama/Llama-2-7b-hf"):
    """
    Analisa comprimento em tokens (não palavras!)
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    comprimentos = []
    muito_longos = 0
    max_length = 2048  # Limite típico
    
    for exemplo in dataset:
        texto_completo = formatar_prompt(
            exemplo['instrucao'],
            exemplo['resposta']
        )
        
        tokens = tokenizer.encode(texto_completo)
        comprimentos.append(len(tokens))
        
        if len(tokens) > max_length:
            muito_longos += 1
    
    import numpy as np
    
    print("📏 ANÁLISE DE COMPRIMENTO (TOKENS)")
    print("=" * 60)
    print(f"Mínimo:    {min(comprimentos)} tokens")
    print(f"Máximo:    {max(comprimentos)} tokens")
    print(f"Média:     {np.mean(comprimentos):.0f} tokens")
    print(f"Mediana:   {np.median(comprimentos):.0f} tokens")
    print(f"P95:       {np.percentile(comprimentos, 95):.0f} tokens")
    
    if muito_longos > 0:
        pct = 100 * muito_longos / len(dataset)
        print(f"\n⚠️  {muito_longos} exemplos ({pct:.1f}%) excedem {max_length} tokens")
        print(f"    Considera truncar ou remover estes exemplos")
    
    # Recomendação de max_length
    p95 = int(np.percentile(comprimentos, 95))
    max_recomendado = min(2048, ((p95 // 128) + 1) * 128)  # Arredondar a múltiplo de 128
    
    print(f"\n✅ max_length recomendado: {max_recomendado}")
    
    return comprimentos, max_recomendado

# Uso
comprimentos, max_len = analisar_comprimento_tokens(dataset)
```

#### Template de Preparação Completa

```python
# preparar_dataset_completo.py
"""
Pipeline completo de preparação de dataset
"""

class PreparadorDataset:
    def __init__(self, formato='alpaca'):
        self.formato = formato
        self.estatisticas = {}
    
    def pipeline_completo(self, dados_brutos, output_file="dataset_preparado.jsonl"):
        """
        Pipeline completo: validar → limpar → balancear → analisar → guardar
        """
        print("🚀 INICIANDO PREPARAÇÃO DE DATASET\n")
        
        # 1. Limpeza
        print("PASSO 1: Limpeza")
        dados_limpos = limpar_dataset(dados_brutos)
        print()
        
        # 2. Balanceamento
        print("PASSO 2: Balanceamento")
        dados_balanceados = balancear_categorias(dados_limpos)
        print()
        
        # 3. Análise de diversidade
        print("PASSO 3: Análise de Diversidade")
        metricas = analisar_diversidade(dados_balanceados)
        print()
        
        # 4. Análise de tokens
        print("PASSO 4: Análise de Tokens")
        comprimentos, max_len = analisar_comprimento_tokens(dados_balanceados)
        print()
        
        # 5. Split treino/validação
        print("PASSO 5: Split Treino/Validação")
        treino, validacao = split_dataset_correto(dados_balanceados)
        print()
        
        # 6. Guardar
        print("PASSO 6: Guardar Dataset")
        import json
        
        with open(f"train_{output_file}", 'w', encoding='utf-8') as f:
            for exemplo in treino:
                f.write(json.dumps(exemplo, ensure_ascii=False) + '\n')
        
        with open(f"val_{output_file}", 'w', encoding='utf-8') as f:
            for exemplo in validacao:
                f.write(json.dumps(exemplo, ensure_ascii=False) + '\n')
        
        print(f"✓ Guardado em train_{output_file} e val_{output_file}")
        
        # Relatório final
        self.relatorio_final(treino, validacao, max_len)
        
        return treino, validacao
    
    def relatorio_final(self, treino, validacao, max_len):
        """
        Relatório final com recomendações
        """
        print("\n" + "=" * 60)
        print("📋 RELATÓRIO FINAL")
        print("=" * 60)
        print(f"Dataset de treino:   {len(treino)} exemplos")
        print(f"Dataset de validação: {len(validacao)} exemplos")
        print(f"max_length sugerido: {max_len}")
        
        # Estimativa de tempo
        print("\n⏱️  Estimativa de Treino (3 epochs, batch_size=4):")
        estimar_tempo_treino(len(treino), epochs=3, batch_size=4)
        
        print("\n✅ Dataset pronto para treino!")
        print("   Próximo passo: Fine-tuning com LoRA")

# Uso
preparador = PreparadorDataset(formato='alpaca')
train_data, val_data = preparador.pipeline_completo(dados_brutos)
```

#### Checklist de Qualidade do Dataset

```
✅ QUALIDADE
   [ ] Respostas revistas manualmente (amostra de 10%)
   [ ] Sem erros ortográficos ou gramaticais
   [ ] Informação factualmente correta
   [ ] Tom e estilo consistentes

✅ DIVERSIDADE
   [ ] Tópicos variados
   [ ] Comprimentos diversos (curto, médio, longo)
   [ ] Diferentes tipos de perguntas
   [ ] Vocabulário rico

✅ FORMATO
   [ ] Formato consistente (Alpaca/ChatML)
   [ ] Campos obrigatórios preenchidos
   [ ] Encoding UTF-8
   [ ] Linha por exemplo (JSONL)

✅ TAMANHO
   [ ] Mínimo de exemplos atingido (50-200 dependendo do caso)
   [ ] Não excessivamente grande (>10k pode ser contraproducente)
   [ ] Balanceado entre categorias

✅ TÉCNICO
   [ ] Sem duplicados
   [ ] Split treino/validação correto (sem leakage)
   [ ] Comprimento de tokens analisado
   [ ] max_length definido apropriadamente

✅ ÉTICA
   [ ] Sem conteúdo ofensivo ou discriminatório
   [ ] Sem informação pessoal identificável
   [ ] Sem dados com copyright
   [ ] Consentimento para uso dos dados (se aplicável)
```

---

#### Modelos até 7B parâmetros

```bash
# Instalar dependências
pip install transformers==4.36.0
pip install peft==0.7.1              # Para LoRA
pip install bitsandbytes==0.41.3     # Para quantização
pip install datasets
pip install accelerate
pip install trl                      # Para RLHF/SFT
```

#### Fine-tuning com LoRA - Exemplo Completo

**Dataset: Instruções em Português**

```python
# preparar_dataset_pt.py
from datasets import Dataset
import json

# Exemplo: Dataset de perguntas e respostas
dados_treino = [
    {
        "instrucao": "Explica o que é aprendizagem automática",
        "resposta": "Aprendizagem automática é um ramo da inteligência artificial que permite aos computadores aprender a partir de dados sem serem explicitamente programados..."
    },
    {
        "instrucao": "Como fazer um bolo de chocolate?",
        "resposta": "Para fazer um bolo de chocolate: 1. Pré-aquece o forno a 180°C. 2. Mistura 200g de farinha..."
    },
    # ... mais exemplos
]

# Converter para formato de treino
def formatar_prompt(exemplo):
    """Formato Alpaca"""
    return f"### Instrução:\n{exemplo['instrucao']}\n\n### Resposta:\n{exemplo['resposta']}"

# Criar dataset
dataset = Dataset.from_list(dados_treino)
dataset = dataset.map(lambda x: {"texto": formatar_prompt(x)})

print(f"✓ Dataset criado com {len(dataset)} exemplos")
```

**Fine-tuning com LoRA (7B modelo):**

```python
# lora_finetuning.py
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import load_dataset

class FineTunerLoRA:
    def __init__(self, modelo_base="meta-llama/Llama-2-7b-hf"):
        self.modelo_base = modelo_base
        self.device = "mps"  # Apple Silicon
        
    def carregar_modelo_4bit(self):
        """
        Carrega modelo em 4-bit para economizar memória
        """
        print(f"📥 Carregando {self.modelo_base} em 4-bit...")
        
        # Configuração de quantização
        from transformers import BitsAndBytesConfig
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Carregar modelo
        model = AutoModelForCausalLM.from_pretrained(
            self.modelo_base,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True
        )
        
        # Preparar para treino
        model = prepare_model_for_kbit_training(model)
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(self.modelo_base)
        tokenizer.pad_token = tokenizer.eos_token
        
        print("✓ Modelo carregado!")
        return model, tokenizer
    
    def configurar_lora(self, model):
        """
        Adiciona adaptadores LoRA ao modelo
        """
        print("🔧 Configurando LoRA...")
        
        lora_config = LoraConfig(
            r=16,                    # Rank das matrizes LoRA (8-64)
            lora_alpha=32,           # Scaling factor
            target_modules=[
                "q_proj",
                "k_proj", 
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )
        
        model = get_peft_model(model, lora_config)
        
        # Estatísticas
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in model.parameters())
        
        print(f"✓ LoRA configurado!")
        print(f"  Parâmetros treináveis: {trainable_params:,} ({100*trainable_params/all_params:.2f}%)")
        print(f"  Parâmetros totais: {all_params:,}")
        
        return model
    
    def treinar(self, model, tokenizer, dataset, output_dir="./lora-model"):
        """
        Treina o modelo com LoRA
        """
        print("🚀 Iniciando treino...")
        
        # Tokenizar dataset
        def tokenize(examples):
            return tokenizer(
                examples["texto"],
                truncation=True,
                max_length=512,
                padding="max_length"
            )
        
        tokenized_dataset = dataset.map(
            tokenize,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        # Argumentos de treino otimizados para M1
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=4,      # Batch pequeno para M1
            gradient_accumulation_steps=4,       # Simula batch 16
            learning_rate=2e-4,
            fp16=True,                           # Mixed precision
            logging_steps=10,
            save_strategy="epoch",
            evaluation_strategy="epoch",
            warmup_steps=100,
            lr_scheduler_type="cosine",
            optim="adamw_torch",                 # Optimizador nativo PyTorch
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        # Treinar!
        trainer.train()
        
        # Guardar
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        print(f"✓ Modelo guardado em {output_dir}")
        
        return model

# Uso completo
finetuner = FineTunerLoRA("meta-llama/Llama-2-7b-hf")
model, tokenizer = finetuner.carregar_modelo_4bit()
model = finetuner.configurar_lora(model)

# Carregar teu dataset
dataset = load_dataset("json", data_files="dados_treino.json")

# Treinar
model = finetuner.treinar(model, tokenizer, dataset["train"])
```

#### Testar Modelo Fine-tuned

```python
# testar_modelo.py
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

def testar_modelo_lora(base_model, lora_path, prompt):
    """
    Testa modelo fine-tuned com LoRA
    """
    # Carregar base
    print("📥 Carregando modelo...")
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        device_map="auto",
        torch_dtype=torch.float16
    )
    
    # Carregar adaptadores LoRA
    model = PeftModel.from_pretrained(model, lora_path)
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    
    # Gerar resposta
    print(f"\n🤖 Prompt: {prompt}\n")
    
    inputs = tokenizer(prompt, return_tensors="pt").to("mps")
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            repetition_penalty=1.1
        )
    
    resposta = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print(f"📝 Resposta:\n{resposta}")
    
    return resposta

# Teste
prompt = "### Instrução:\nExplica o que é o chip M1 da Apple\n\n### Resposta:"

resposta = testar_modelo_lora(
    base_model="meta-llama/Llama-2-7b-hf",
    lora_path="./lora-model",
    prompt=prompt
)
```

---

### 5.3 MLX Framework da Apple

#### O que é MLX?

**MLX** = Machine Learning framework nativo da Apple para Apple Silicon
- Desenvolvido pela Apple Research
- Otimizado especificamente para M1/M2/M3
- Unified Memory aproveitada ao máximo
- Sintaxe parecida com NumPy/PyTorch

**Vantagens no M1:**
- ✅ Performance superior ao PyTorch MPS
- ✅ Menos uso de memória
- ✅ Suporte nativo para quantização
- ✅ LLMs otimizados (llama, mistral, phi)

#### Instalação e Setup

```bash
# Instalar MLX
pip install mlx
pip install mlx-lm  # Para modelos de linguagem

# Verificar instalação
python -c "import mlx.core as mx; print(mx.__version__)"
```

#### Fine-tuning com MLX (Mais Eficiente!)

```python
# mlx_finetuning.py
"""
Fine-tuning com MLX - Mais rápido e eficiente que PyTorch no M1
"""

# Exemplo usando mlx-lm (wrapper de alto nível)
from mlx_lm import load, generate
from mlx_lm.tuner import train

# 1. Carregar modelo base
model, tokenizer = load("mlx-community/Mistral-7B-v0.1-4bit")

# 2. Preparar dados (formato JSONL)
"""
# dados.jsonl:
# {"text": "### Instrução: ... ### Resposta: ..."}
# {"text": "### Instrução: ... ### Resposta: ..."}
"""

# 3. Configuração LoRA para MLX
config = {
    "lora_layers": 16,           # Número de camadas LoRA
    "lora_rank": 16,             # Rank
    "lora_alpha": 32,
    "batch_size": 4,
    "iters": 1000,               # Iterações
    "learning_rate": 1e-5,
    "steps_per_eval": 100,
    "save_every": 100,
    "adapter_file": "adapters.npz"
}

# 4. Treinar
print("🚀 Treinando com MLX...")
train(
    model="mlx-community/Mistral-7B-v0.1-4bit",
    data="dados.jsonl",
    valid_data="val.jsonl",
    **config
)

print("✓ Treino completo! Adaptadores guardados em adapters.npz")

# 5. Testar
prompt = "### Instrução: O que é o MLX? ### Resposta:"
resposta = generate(
    model,
    tokenizer,
    prompt=prompt,
    max_tokens=200,
    temp=0.7
)
print(resposta)
```

#### Converter Modelos para MLX

```python
# converter_para_mlx.py
"""
Converter modelos HuggingFace para formato MLX
"""
from mlx_lm import convert

# Converter modelo
convert(
    hf_path="meta-llama/Llama-2-7b-hf",  # Modelo HuggingFace
    mlx_path="./llama-7b-mlx",           # Destino MLX
    quantize=True,                        # Quantizar para 4-bit
    q_group_size=64,                      # Tamanho do grupo
    q_bits=4                              # 4-bit quantization
)

print("✓ Modelo convertido para MLX com quantização 4-bit!")
```

#### MLX vs PyTorch no M1 - Comparação

```python
# benchmark_mlx_vs_pytorch.py
import time
import mlx.core as mx
import torch

def benchmark_inferencia():
    """
    Compara velocidade MLX vs PyTorch
    """
    # MLX
    from mlx_lm import load, generate
    model_mlx, tokenizer_mlx = load("mlx-community/Mistral-7B-v0.1-4bit")
    
    prompt = "Explica o que é inteligência artificial"
    
    # Teste MLX
    start = time.time()
    resposta_mlx = generate(model_mlx, tokenizer_mlx, prompt, max_tokens=100)
    tempo_mlx = time.time() - start
    
    # PyTorch MPS
    from transformers import AutoModelForCausalLM, AutoTokenizer
    model_pt = AutoModelForCausalLM.from_pretrained(
        "mistralai/Mistral-7B-v0.1",
        device_map="mps",
        torch_dtype=torch.float16
    )
    tokenizer_pt = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")
    
    start = time.time()
    inputs = tokenizer_pt(prompt, return_tensors="pt").to("mps")
    outputs = model_pt.generate(**inputs, max_new_tokens=100)
    tempo_pt = time.time() - start
    
    print("=" * 60)
    print("BENCHMARK: MLX vs PyTorch MPS")
    print("=" * 60)
    print(f"MLX:        {tempo_mlx:.2f}s")
    print(f"PyTorch:    {tempo_pt:.2f}s")
    print(f"Speedup:    {tempo_pt/tempo_mlx:.2f}x")
    print("=" * 60)

# Executar
benchmark_inferencia()

# Resultado típico no M1 16GB:
# MLX:        3.2s
# PyTorch:    5.8s
# Speedup:    1.8x ✅
```

---

### 6.2 Core ML

#### O que é Core ML?

**Core ML** = Framework da Apple para executar modelos de AA em dispositivos Apple.

**Vantagens:**
- ✅ Optimizado para todos os chips Apple (M-series)
- ✅ Usa Neural Engine automaticamente
- ✅ Integração perfeita com apps iOS/macOS
- ✅ Privacidade (tudo on-device)
- ✅ Baixo consumo energético

**Quando usar Core ML:**
- Apps iOS/macOS que precisam de AA
- Inferência em produção
- Privacidade é crítica
- Modelos pequenos/médios (<1GB)

#### Converter Modelos para Core ML

```python
# converter_coreml.py
import coremltools as ct
import tensorflow as tf
from tensorflow import keras

class ConversorCoreML:
    """
    Converte modelos TensorFlow/PyTorch para Core ML
    """
    
    def converter_keras(self, modelo_path, output_path="modelo.mlmodel"):
        """
        Converte modelo Keras para Core ML
        """
        print(f"🔄 Convertendo {modelo_path} para Core ML...")
        
        # Carregar modelo
        model = keras.models.load_model(modelo_path)
        
        # Converter
        mlmodel = ct.convert(
            model,
            inputs=[ct.ImageType(shape=(1, 224, 224, 3))],
            convert_to="mlprogram",  # Formato moderno
            compute_precision=ct.precision.FLOAT16  # Mais eficiente
        )
        
        # Adicionar metadados
        mlmodel.author = "Teu Nome"
        mlmodel.short_description = "Classificador de imagens"
        mlmodel.version = "1.0"
        
        # Guardar
        mlmodel.save(output_path)
        
        print(f"✓ Modelo convertido: {output_path}")
        print(f"  Tamanho: {os.path.getsize(output_path) / (1024**2):.1f} MB")
        
        return mlmodel
    
    def converter_pytorch(self, model, exemplo_input, output_path="modelo.mlmodel"):
        """
        Converte modelo PyTorch para Core ML
        """
        import torch
        
        print("🔄 Convertendo PyTorch para Core ML...")
        
        # Colocar em modo eval
        model.eval()
        
        # Traçar modelo
        traced_model = torch.jit.trace(model, exemplo_input)
        
        # Converter
        mlmodel = ct.convert(
            traced_model,
            inputs=[ct.TensorType(shape=exemplo_input.shape)],
            convert_to="mlprogram",
            compute_precision=ct.precision.FLOAT16
        )
        
        mlmodel.save(output_path)
        print(f"✓ Modelo convertido: {output_path}")
        
        return mlmodel
    
    def optimizar_para_neural_engine(self, mlmodel):
        """
        Optimiza modelo para Neural Engine
        """
        print("\n⚡ Optimizando para Neural Engine...")
        
        # Configurações recomendadas
        mlmodel = ct.compression_utils.quantize_weights(
            mlmodel,
            nbits=8,
            quantization_mode="linear"
        )
        
        print("✓ Optimização completa!")
        print("  - Quantização 8-bit aplicada")
        print("  - Preparado para Neural Engine")
        
        return mlmodel

# Instalação necessária:
# pip install coremltools

# Exemplo de uso
conversor = ConversorCoreML()

# Converter modelo Keras
modelo_coreml = conversor.converter_keras(
    "classificador_final.keras",
    "ClassificadorImagens.mlmodel"
)

# Optimizar
modelo_optimizado = conversor.optimizar_para_neural_engine(modelo_coreml)
modelo_optimizado.save("ClassificadorImagens_Optimizado.mlmodel")
```

#### Usar Core ML em Python

```python
# usar_coreml.py
import coremltools as ct
import numpy as np
from PIL import Image

class InferenciaCoreML:
    """
    Executa inferência com modelos Core ML
    """
    def __init__(self, modelo_path):
        self.modelo_path = modelo_path
        self.model = None
    
    def carregar(self):
        """
        Carrega modelo Core ML
        """
        print(f"📥 Carregando {self.modelo_path}...")
        self.model = ct.models.MLModel(self.modelo_path)
        
        # Informação do modelo
        spec = self.model.get_spec()
        print(f"✓ Modelo carregado!")
        print(f"  Descrição: {spec.description}")
        
        return self.model
    
    def prever_imagem(self, imagem_path):
        """
        Faz previsão em imagem
        """
        # Carregar e preparar imagem
        img = Image.open(imagem_path).resize((224, 224))
        img_array = np.array(img).astype(np.float32) / 255.0
        
        # Prever
        resultado = self.model.predict({'image': img_array})
        
        return resultado
    
    def benchmark(self, num_runs=100):
        """
        Mede desempenho do modelo
        """
        import time
        
        print(f"\n⚡ Benchmark ({num_runs} execuções)")
        print("=" * 60)
        
        # Criar input dummy
        dummy_input = np.random.rand(224, 224, 3).astype(np.float32)
        
        # Warm-up
        for _ in range(10):
            _ = self.model.predict({'image': dummy_input})
        
        # Benchmark
        start = time.time()
        for _ in range(num_runs):
            _ = self.model.predict({'image': dummy_input})
        tempo_total = time.time() - start
        
        tempo_medio = (tempo_total / num_runs) * 1000
        fps = num_runs / tempo_total
        
        print(f"Tempo médio: {tempo_medio:.2f} ms")
        print(f"FPS: {fps:.1f}")
        print(f"Throughput: {1000/tempo_medio:.1f} inferências/segundo")

# Uso
inferencia = InferenciaCoreML("ClassificadorImagens.mlmodel")
inferencia.carregar()

# Prever
resultado = inferencia.prever_imagem("teste.jpg")
print(f"\nPrevisão: {resultado}")

# Benchmark
inferencia.benchmark(num_runs=100)
```

---

### 6.3 Monitorização e Debugging

#### TensorBoard

```python
# tensorboard_avancado.py
import tensorflow as tf
from tensorflow import keras
from datetime import datetime

class MonitorizacaoTensorBoard:
    """
    Monitorização avançada com TensorBoard
    """
    def __init__(self, log_dir="logs"):
        self.log_dir = f"{log_dir}/{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        self.callbacks = []
    
    def configurar_callbacks(self):
        """
        Configura callbacks do TensorBoard
        """
        # Callback básico
        tensorboard = keras.callbacks.TensorBoard(
            log_dir=self.log_dir,
            histogram_freq=1,        # Histogramas dos pesos
            write_graph=True,        # Gráfico do modelo
            write_images=True,       # Visualização de features
            update_freq='epoch',     # Frequência de actualização
            profile_batch='10,20'    # Profiling de performance
        )
        
        self.callbacks.append(tensorboard)
        
        # Callback personalizado para métricas extras
        class MetricasPersonalizadas(keras.callbacks.Callback):
            def __init__(self, log_dir):
                super().__init__()
                self.file_writer = tf.summary.create_file_writer(log_dir + '/custom')
            
            def on_epoch_end(self, epoch, logs=None):
                with self.file_writer.as_default():
                    # Learning rate actual
                    lr = self.model.optimizer.learning_rate
                    if hasattr(lr, 'numpy'):
                        tf.summary.scalar('learning_rate', lr.numpy(), step=epoch)
                    
                    # Norma dos gradientes (detectar problemas)
                    if logs and 'gradient_norm' in logs:
                        tf.summary.scalar('gradient_norm', logs['gradient_norm'], step=epoch)
                
                self.file_writer.flush()
        
        self.callbacks.append(MetricasPersonalizadas(self.log_dir))
        
        print(f"✓ TensorBoard configurado")
        print(f"  Log dir: {self.log_dir}")
        print(f"  Para visualizar: tensorboard --logdir={self.log_dir}")
        
        return self.callbacks

# Uso
monitor = MonitorizacaoTensorBoard()
cbks = monitor.configurar_callbacks()

# Treinar modelo
model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=50,
    callbacks=cbks
)

print("\n📊 Para ver os resultados:")
print(f"   tensorboard --logdir={monitor.log_dir}")
```

#### Weights & Biases (wandb)

```python
# wandb_integracao.py
import wandb
from wandb.keras import WandbCallback

class MonitorizacaoWandB:
    """
    Integração com Weights & Biases
    Melhor que TensorBoard para experimentos múltiplos
    """
    def __init__(self, projeto="treino-m1", nome_experimento=None):
        self.projeto = projeto
        self.nome = nome_experimento or f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    def inicializar(self, config):
        """
        Inicializa tracking do wandb
        
        Args:
            config: Dicionário com hiperparâmetros
        """
        wandb.init(
            project=self.projeto,
            name=self.nome,
            config=config
        )
        
        print(f"✓ W&B inicializado")
        print(f"  Projecto: {self.projeto}")
        print(f"  Experimento: {self.nome}")
        print(f"  URL: {wandb.run.url}")
    
    def criar_callback(self):
        """
        Cria callback para Keras
        """
        return WandbCallback(
            save_model=True,
            monitor='val_accuracy',
            mode='max'
        )
    
    def registar_metricas(self, metricas, step=None):
        """
        Regista métricas personalizadas
        """
        wandb.log(metricas, step=step)
    
    def registar_modelo(self, modelo_path):
        """
        Regista modelo final
        """
        artifact = wandb.Artifact('modelo', type='model')
        artifact.add_file(modelo_path)
        wandb.log_artifact(artifact)
        
        print(f"✓ Modelo registado no W&B")

# Instalação: 
# pip install wandb

# Uso
config = {
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'arquitetura': 'MobileNetV2',
    'dataset': 'custom_animals'
}

monitor = MonitorizacaoWandB(projeto="classificador-animais")
monitor.inicializar(config)

# Treinar
history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=config['epochs'],
    callbacks=[monitor.criar_callback()]
)

# Registar modelo final
monitor.registar_modelo('modelo_final.keras')

# Terminar
wandb.finish()
```

#### Profiling de Desempenho

```python
# profiling_m1.py
import time
import psutil
import tensorflow as tf
from contextlib import contextmanager

class ProfilerM1:
    """
    Profile de desempenho específico para M1
    """
    def __init__(self):
        self.metricas = []
    
    @contextmanager
    def profile_block(self, nome):
        """
        Context manager para profile de blocos de código
        """
        print(f"\n⏱️  Profiling: {nome}")
        print("=" * 60)
        
        # Métricas iniciais
        mem_antes = psutil.virtual_memory().used / (1024**3)
        start = time.time()
        
        try:
            yield
        finally:
            # Métricas finais
            tempo = time.time() - start
            mem_depois = psutil.virtual_memory().used / (1024**3)
            mem_usada = mem_depois - mem_antes
            
            resultado = {
                'nome': nome,
                'tempo': tempo,
                'memoria_usada_gb': mem_usada
            }
            
            self.metricas.append(resultado)
            
            print(f"Tempo: {tempo:.2f}s")
            print(f"Memória usada: {mem_usada:.2f} GB")
    
    def relatorio(self):
        """
        Gera relatório final de profiling
        """
        print("\n" + "=" * 60)
        print("📊 RELATÓRIO DE PROFILING")
        print("=" * 60)
        
        for metrica in self.metricas:
            print(f"\n{metrica['nome']}:")
            print(f"  Tempo: {metrica['tempo']:.2f}s")
            print(f"  Memória: {metrica['memoria_usada_gb']:.2f} GB")
        
        # Tempo total
        tempo_total = sum(m['tempo'] for m in self.metricas)
        print(f"\n⏱️  Tempo total: {tempo_total:.2f}s")


# Módulo 7: Projectos Práticos para M1 16GB

## 📋 Visão Geral

| Projecto | Tempo | Dificuldade | Memória | Dataset Mín. |
|----------|-------|-------------|---------|--------------|
| **1. Classificador Imagens** | 1-1.5h | ⭐⭐⭐ | 6-8GB | 50/classe |
| **2. Análise Sentimentos** | 35-40min | ⭐⭐⭐⭐ | 8-10GB | 300 total |
| **3. Fine-tuning LLM** | 1.5-2h | ⭐⭐⭐⭐⭐ | 6-7GB | 100 exemplos |

---

## ⚠️ Preparação Obrigatória (Todos os Projectos)

### Verificações Essenciais

```bash
# 1. Espaço em disco (mín. 20GB)
df -h

# 2. Python ARM64 (CRÍTICO!)
python -c "import platform; print(platform.machine())"
# Deve retornar: arm64

# 3. Ambiente correcto
conda activate ml-m1

# 4. Memória disponível (mín. 8GB)
vm_stat | grep "Pages free"
```

### Regras de Ouro

✅ **SEMPRE:**
- Fecha Chrome, Slack e apps pesadas
- Verifica dataset antes de treinar
- Usa mixed precision (FP16)
- Guarda checkpoints regularmente

❌ **NUNCA:**
- Interrompas treino sem motivo
- Uses Python x86 (só ARM64!)
- Treines com batch_size >64
- Ignores avisos de memória

---

## 🖼️ Projecto 1: Classificador de Imagens

### Objectivo
Classificar 10 raças de cães portugueses com 85%+ accuracy.

### Requisitos Específicos
- Mínimo 50 imagens/classe (ideal: 100+)
- Formatos: JPG ou PNG
- Nomes sem espaços: `cao_da_serra_estrela`

### Script 1: Organizar Dataset

```python
# 1_organizar_dataset.py
from pathlib import Path

class OrganizadorDataset:
    """Cria estrutura: dataset/train/classe/ e dataset/validation/classe/"""
    
    def __init__(self, destino="dataset_caes"):
        self.destino = Path(destino)
        self.racas = [
            "cao_serra_estrela", "cao_agua_portugues", "podengo_portugues",
            "perdigueiro_portugues", "rafeiro_alentejano", "cao_castro_laboreiro",
            "barbado_terceira", "fila_sao_miguel", "transmontano", "serra_aires"
        ]
    
    def criar_estrutura(self):
        for split in ['train', 'validation']:
            for raca in self.racas:
                (self.destino / split / raca).mkdir(parents=True, exist_ok=True)
        print(f"✓ Estrutura criada em {self.destino}")
    
    def verificar(self):
        problemas = []
        for raca in self.racas:
            n_train = len(list((self.destino/'train'/raca).glob('*.jpg')))
            n_val = len(list((self.destino/'validation'/raca).glob('*.jpg')))
            
            print(f"{raca:25s} - Train: {n_train:3d} | Val: {n_val:3d}")
            
            if n_train < 50: problemas.append(f"{raca}: <50 treino")
            if n_val < 10: problemas.append(f"{raca}: <10 validação")
        
        if problemas:
            print("\n⚠️  PROBLEMAS:")
            for p in problemas: print(f"  - {p}")
            return False
        return True

# Executar
org = OrganizadorDataset()
org.criar_estrutura()
if not org.verificar():
    print("\n❌ Corrige problemas antes de continuar!")
    exit(1)
```

**💡 Explicação:**
- `mkdir(exist_ok=True)` - Não dá erro se pasta existe
- `stratify` - Mantém proporção de classes no split
- Verificação obrigatória evita treino em dataset inválido

### Script 2: Treinar Modelo

```python
# 2_treinar.py
import tensorflow as tf
from tensorflow import keras

# CRÍTICO: Mixed precision economiza 50% RAM
keras.mixed_precision.set_global_policy('mixed_float16')

class TreinadorSimples:
    def __init__(self, dataset_path="dataset_caes"):
        self.dataset_path = dataset_path
    
    def carregar_dados(self, batch_size=32):
        """
        batch_size=32 ideal para M1
        Se OOM: reduz para 16
        """
        augmentation = keras.Sequential([
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.2),
            keras.layers.RandomZoom(0.2),
        ])
        
        train_ds = keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/train",
            image_size=(224, 224),
            batch_size=batch_size,
            shuffle=True
        )
        
        val_ds = keras.preprocessing.image_dataset_from_directory(
            f"{self.dataset_path}/validation",
            image_size=(224, 224),
            batch_size=batch_size
        )
        
        # Augmentation + Normalização
        train_ds = train_ds.map(lambda x,y: (augmentation(x, training=True)/255, y))
        val_ds = val_ds.map(lambda x,y: (x/255, y))
        
        return train_ds.prefetch(tf.data.AUTOTUNE), val_ds.prefetch(tf.data.AUTOTUNE)
    
    def criar_modelo(self, num_classes=10):
        """EfficientNetB0: leve e eficiente"""
        base = keras.applications.EfficientNetB0(
            include_top=False, weights='imagenet', input_shape=(224,224,3)
        )
        base.trainable = False  # Fase 1: congelado
        
        model = keras.Sequential([
            base,
            keras.layers.GlobalAveragePooling2D(),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(256, activation='relu'),
            keras.layers.Dense(num_classes, activation='softmax', dtype='float32')
        ])
        
        return model, base
    
    def treinar(self, model, base, train_ds, val_ds):
        """Treino em 2 fases: Transfer Learning → Fine-tuning"""
        
        # FASE 1: Só classificador (10-20min)
        print("\n🎯 FASE 1: Transfer Learning")
        model.compile(
            optimizer=keras.optimizers.Adam(1e-3),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(train_ds, validation_data=val_ds, epochs=20, 
                  callbacks=[keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])
        
        # FASE 2: Fine-tuning últimas 30 camadas (20-40min)
        print("\n🎯 FASE 2: Fine-tuning")
        base.trainable = True
        for layer in base.layers[:-30]:
            layer.trainable = False
        
        model.compile(
            optimizer=keras.optimizers.Adam(1e-5),  # LR 100x menor!
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy']
        )
        model.fit(train_ds, validation_data=val_ds, epochs=30,
                  callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])
        
        return model

# Executar
treinador = TreinadorSimples()
train_ds, val_ds = treinador.carregar_dados()
model, base = treinador.criar_modelo()
model = treinador.treinar(model, base, train_ds, val_ds)
model.save('classificador_caes.keras')
print("✅ Modelo guardado!")
```

**💡 Pontos-chave:**
- **Mixed precision** (FP16): Economiza 50% RAM
- **2 fases**: Transfer learning (rápido) → Fine-tuning (preciso)
- **Learning rates**: 1e-3 fase 1, 1e-5 fase 2 (importante!)
- **Early stopping**: Para automaticamente quando não melhora

### Script 3: API REST

```python
# 3_api.py
from flask import Flask, request, jsonify
from tensorflow import keras
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
model = keras.models.load_model('classificador_caes.keras')

racas = [
    "Cão Serra Estrela", "Cão Água Português", "Podengo Português",
    "Perdigueiro Português", "Rafeiro Alentejano", "Cão Castro Laboreiro",
    "Barbado Terceira", "Fila São Miguel", "Transmontano", "Serra Aires"
]

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'erro': 'Sem imagem'}), 400
    
    img = Image.open(io.BytesIO(request.files['image'].read()))
    if img.mode != 'RGB': img = img.convert('RGB')
    
    img = np.array(img.resize((224,224))) / 255.0
    predictions = model.predict(np.expand_dims(img, 0), verbose=0)[0]
    
    top3 = np.argsort(predictions)[-3:][::-1]
    return jsonify({
        'previsoes': [
            {'raca': racas[i], 'probabilidade': float(predictions[i])}
            for i in top3
        ]
    })

if __name__ == '__main__':
    app.run(port=5000)
```

**Testar:** `curl -X POST -F "image=@teste.jpg" http://localhost:5000/predict`

### ✅ Checklist
- [ ] Dataset ≥50 imgs/classe
- [ ] Treino completo sem erros
- [ ] Accuracy validação >80%
- [ ] API responde correctamente

---

## 📝 Projecto 2: Análise de Sentimentos

### Objectivo
Classificar sentimento (Negativo/Neutro/Positivo) com 85%+ accuracy.

### Script 1: Dataset

```python
# 1_criar_dataset.py
import pandas as pd
import random

def criar_dataset(n=2000):
    exemplos = {
        2: ["Excelente!", "Adorei!", "Recomendo!"],  # Positivo
        1: ["Razoável.", "Está bem.", "Normal."],    # Neutro
        0: ["Péssimo!", "Horrível!", "Não comprem!"] # Negativo
    }
    
    dados = []
    for label, textos in exemplos.items():
        for _ in range(n//3):
            dados.append({'texto': random.choice(textos), 'label': label})
    
    df = pd.DataFrame(dados).sample(frac=1, random_state=42)
    df.to_csv('reviews.csv', index=False)
    print(f"✓ Dataset: {len(df)} exemplos")
    print(df['label'].value_counts())

criar_dataset()
```

### Script 2: Treinar

```python
# 2_treinar_bert.py
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split

# Mixed precision
tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Dados
df = pd.read_csv('reviews.csv')
train_df, val_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

# BERT Português (400MB download primeira vez)
tokenizer = AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")
model = TFAutoModelForSequenceClassification.from_pretrained(
    "neuralmind/bert-base-portuguese-cased", num_labels=3
)

# Tokenizar
def prep_dataset(df, batch_size=16):
    encodings = tokenizer(df['texto'].tolist(), truncation=True, padding=True, 
                          max_length=128, return_tensors='tf')
    return tf.data.Dataset.from_tensor_slices((
        dict(encodings), df['label'].values
    )).batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = prep_dataset(train_df)
val_ds = prep_dataset(val_df)

# Treinar (15-20min, 3 epochs suficientes)
model.compile(
    optimizer=tf.keras.optimizers.Adam(2e-5),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

model.fit(train_ds, validation_data=val_ds, epochs=3)
model.save_pretrained('modelo_sentimentos')
tokenizer.save_pretrained('modelo_sentimentos')
print("✅ Modelo guardado!")
```

**💡 Importante:**
- `max_length=128` (não 512) economiza RAM
- `batch_size=16` para M1 16GB
- `2e-5` learning rate padrão para BERT

### Script 3: Interface Streamlit

```python
# 3_app.py
import streamlit as st
from transformers import AutoTokenizer, TFAutoModelForSequenceClassification
import tensorflow as tf

st.set_page_config(page_title="Sentimentos PT", page_icon="😊")

@st.cache_resource
def carregar():
    tokenizer = AutoTokenizer.from_pretrained('modelo_sentimentos')
    model = TFAutoModelForSequenceClassification.from_pretrained('modelo_sentimentos')
    return tokenizer, model

tokenizer, model = carregar()

st.title("😊 Análise de Sentimentos")
texto = st.text_area("Texto:", height=150)

if st.button("Analisar") and texto:
    inputs = tokenizer(texto, return_tensors='tf', truncation=True, max_length=128)
    outputs = model(**inputs)
    probs = tf.nn.softmax(outputs.logits, axis=-1).numpy()[0]
    
    labels = ['Negativo 😢', 'Neutro 😐', 'Positivo 😊']
    resultado = labels[probs.argmax()]
    
    st.markdown(f"## {resultado}")
    st.markdown(f"**Confiança:** {probs.max()*100:.1f}%")
    
    import plotly.graph_objects as go
    fig = go.Figure(data=[go.Bar(x=labels, y=probs*100)])
    st.plotly_chart(fig)
```

**Executar:** `streamlit run 3_app.py`

### ✅ Checklist
- [ ] Dataset balanceado (≥300)
- [ ] Treino 3 epochs completo
- [ ] Accuracy >80%
- [ ] Interface funcional

---

## 🤖 Projecto 3: Fine-tuning LLM

### Objectivo
Fine-tuning Mistral 7B com LoRA para domínio específico (Python).

### Script 1: Dataset

```python
# 1_dataset.py
import json

exemplos = [
    {"instruction": "Como criar lista?", 
     "output": "Use colchetes:\n```python\nlista = [1, 2, 3]\n```"},
    {"instruction": "O que é dicionário?",
     "output": "Pares chave-valor:\n```python\nd = {'nome': 'João'}\n```"},
    # Adiciona 100+ exemplos!
]

# Guardar JSONL
with open('train.jsonl', 'w', encoding='utf-8') as f:
    for ex in exemplos[:int(len(exemplos)*0.9)]:
        texto = f"### Instrução:\n{ex['instruction']}\n\n### Resposta:\n{ex['output']}"
        f.write(json.dumps({"text": texto}, ensure_ascii=False) + '\n')

with open('val.jsonl', 'w', encoding='utf-8') as f:
    for ex in exemplos[int(len(exemplos)*0.9):]:
        texto = f"### Instrução:\n{ex['instruction']}\n\n### Resposta:\n{ex['output']}"
        f.write(json.dumps({"text": texto}, ensure_ascii=False) + '\n')

print(f"✓ {len(exemplos)} exemplos criados")
```

### Script 2: Fine-tuning MLX

```python
# 2_treinar_mlx.py
from mlx_lm.tuner import train

# Configuração optimizada M1 16GB
config = {
    "lora_layers": 16,      # Camadas a adaptar
    "lora_rank": 16,        # Tamanho adaptadores
    "lora_alpha": 32,       # Scaling
    "batch_size": 2,        # CRÍTICO: não aumentar!
    "iters": 1000,          # ~3 epochs
    "learning_rate": 1e-5,  # Padrão LoRA
    "adapter_file": "adapters.npz"
}

print("🚀 Fine-tuning (30-60min)...")
train(
    model="mlx-community/Mistral-7B-v0.1-4bit",
    data="train.jsonl",
    valid_data="val.jsonl",
    **config
)
print("✅ Adaptadores guardados!")
```

**💡 Memória:**
- Base 4-bit: ~3.5GB
- LoRA params: ~20MB
- Activações: ~2GB
- **Total: ~6GB** ✅ Cabe no M1!

### Script 3: Chat Interface

```python
# 3_chat.py
import streamlit as st
from mlx_lm import load, generate

st.title("🐍 Assistente Python")

@st.cache_resource
def carregar():
    return load("mlx-community/Mistral-7B-v0.1-4bit", adapter_file="adapters.npz")

model, tokenizer = carregar()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if prompt := st.chat_input("Pergunta..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    
    with st.chat_message("user"):
        st.markdown(prompt)
    
    with st.chat_message("assistant"):
        prompt_fmt = f"### Instrução:\n{prompt}\n\n### Resposta:\n"
        resposta = generate(model, tokenizer, prompt=prompt_fmt, max_tokens=200, temp=0.7)
        resposta = resposta.split("### Resposta:\n")[-1].strip()
        st.markdown(resposta)
    
    st.session_state.messages.append({"role": "assistant", "content": resposta})
```

### ✅ Checklist
- [ ] Dataset ≥100 exemplos qualidade
- [ ] Fine-tuning completo
- [ ] Testes mostram melhoria
- [ ] Chat interface funcional

---

## 🚨 Troubleshooting Rápido

### Out of Memory
```python
# Reduz batch_size
batch_size = 16  # era 32
# Ou fecha apps e reinicia
```

### GPU não detectada
```bash
pip uninstall tensorflow-metal
pip install tensorflow-metal==1.1.0
```

### Modelo não guarda
```python
import os
os.makedirs('modelos', exist_ok=True)
model.save('modelos/modelo.keras')
```

### Treino muito lento
```python
# Verifica GPU
import tensorflow as tf
print(len(tf.config.list_physical_devices('GPU')))
# Deve ser >0
```

---

## 📊 Resumo Final

**Completaste os 3 projectos!** 🎉

Agora dominas:
- ✅ Transfer learning e fine-tuning
- ✅ NLP com transformers
- ✅ LLMs com LoRA/QLoRA
- ✅ Optimização para M1 16GB
- ✅ Deployment (API + Streamlit)

**Próximos passos:**
1. Adapta ao teu domínio
2. Experimenta outros modelos
3. Combina técnicas
4. Deploy em produção


## Módulo 8: Boas Práticas e Troubleshooting
### 8.1 Workflows Eficientes
- Experimentação rápida
- Versionamento de modelos
- Reprodutibilidade

### 8.2 Problemas Comuns
- Out of Memory errors
- Lentidão inesperada
- Incompatibilidades de bibliotecas

### 8.3 Quando Usar Cloud Computing
- Limitações do M1 16GB
- Google Colab, Kaggle, AWS
- Estratégia híbrida

## Módulo 9: Recursos Adicionais
### 9.1 Comunidades e Suporte
- Fóruns especializados em M1
- Discord e Slack communities
- Stack Overflow

### 9.2 Leitura Complementar
- Papers importantes
- Blogs técnicos
- Documentação oficial

### 9.3 Atualizações e Futuro
- Novas versões de frameworks
- M2/M3 e diferenças
- Tendências em edge ML

---

## Anexos
- A. Comandos úteis de terminal
- B. Snippets de código reutilizáveis
- C. Checklist de otimização
- D. Recursos de datasets gratuitos