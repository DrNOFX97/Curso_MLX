# Anexos: Referência Rápida

## Índice

### Anexo A: Comandos Úteis de Terminal
### Anexo B: Snippets de Código Reutilizáveis
### Anexo C: Checklist de Optimização
### Anexo D: Recursos de Datasets Gratuitos
### Anexo E: Glossário de Termos
### Anexo F: Tabelas de Referência Rápida

---

## Anexo A: Comandos Úteis de Terminal

### Gestão de Ambiente

```bash
# ═══════════════════════════════════════════════════════════
# CONDA
# ═══════════════════════════════════════════════════════════

# Listar ambientes
conda env list

# Criar ambiente
conda create -n ml-m1 python=3.11 -y

# Activar/Desactivar
conda activate ml-m1
conda deactivate

# Exportar ambiente (para partilhar)
conda env export > environment.yml

# Criar de environment.yml
conda env create -f environment.yml

# Remover ambiente
conda env remove -n ml-m1

# Actualizar conda
conda update -n base conda

# Limpar cache (liberta espaço)
conda clean --all


# ═══════════════════════════════════════════════════════════
# PIP
# ═══════════════════════════════════════════════════════════

# Listar pacotes instalados
pip list

# Instalar versão específica
pip install tensorflow==2.16.1

# Actualizar pacote
pip install --upgrade transformers

# Guardar dependências
pip freeze > requirements.txt

# Instalar de requirements.txt
pip install -r requirements.txt

# Desinstalar
pip uninstall tensorflow

# Ver informação de pacote
pip show tensorflow

# Procurar pacote
pip search "machine learning"
```

### Monitorização do Sistema

```bash
# ═══════════════════════════════════════════════════════════
# MEMÓRIA E CPU
# ═══════════════════════════════════════════════════════════

# Uso de memória (simples)
vm_stat

# Uso de memória (detalhado)
top -l 1 | head -n 10

# Monitorizar em tempo real
htop  # Instalar: brew install htop

# Ver processos Python
ps aux | grep python

# Matar processo
kill -9 <PID>

# Ver espaço em disco
df -h

# Ver tamanho de pasta
du -sh models/

# Ver ficheiros grandes
du -ah . | sort -rh | head -n 20


# ═══════════════════════════════════════════════════════════
# GPU/METAL
# ═══════════════════════════════════════════════════════════

# Ver actividade GPU (Activity Monitor)
open -a "Activity Monitor"

# Stats do sistema
system_profiler SPHardwareDataType

# Temperatura e ventoinhas (requer iStat Menus ou similar)
# Alternativa: usar Activity Monitor → Window → GPU History


# ═══════════════════════════════════════════════════════════
# REDE
# ═══════════════════════════════════════════════════════════

# Testar velocidade download
curl -o /dev/null http://speedtest.wdc01.softlayer.com/downloads/test100.zip

# Ver processos que usam rede
lsof -i

# Download com progresso
curl -L -o model.zip https://exemplo.com/model.zip \
  --progress-bar
```

### Git Essenciais

```bash
# ═══════════════════════════════════════════════════════════
# GIT BASICS
# ═══════════════════════════════════════════════════════════

# Inicializar repo
git init

# Clonar repo
git clone https://github.com/user/repo.git

# Status
git status

# Adicionar ficheiros
git add .
git add src/  # apenas pasta src

# Commit
git commit -m "Adiciona modelo de classificação"

# Push
git push origin main

# Pull
git pull origin main

# Ver histórico
git log --oneline --graph

# Ver diferenças
git diff


# ═══════════════════════════════════════════════════════════
# GIT LFS (para modelos grandes)
# ═══════════════════════════════════════════════════════════

# Instalar
brew install git-lfs
git lfs install

# Trackear ficheiros grandes
git lfs track "*.keras"
git lfs track "*.h5"
git lfs track "*.pth"

# Adicionar .gitattributes
git add .gitattributes
git commit -m "Adiciona LFS tracking"

# Ver ficheiros tracked
git lfs ls-files
```

### Jupyter Notebook

```bash
# Iniciar Jupyter
jupyter notebook

# Jupyter Lab (interface moderna)
jupyter lab

# Listar kernels
jupyter kernelspec list

# Adicionar ambiente conda como kernel
python -m ipykernel install --user --name=ml-m1

# Remover kernel
jupyter kernelspec uninstall ml-m1

# Converter notebook para script
jupyter nbconvert --to script notebook.ipynb

# Converter para HTML
jupyter nbconvert --to html notebook.ipynb
```

### Scripts Úteis One-liners

```bash
# Encontrar ficheiros Python modificados hoje
find . -name "*.py" -mtime -1

# Contar linhas de código Python
find . -name "*.py" | xargs wc -l

# Apagar todos os __pycache__
find . -type d -name __pycache__ -exec rm -r {} +

# Apagar checkpoints Jupyter
find . -name ".ipynb_checkpoints" -type d -exec rm -rf {} +

# Ver tamanho de cada pasta (top 10)
du -sh */ | sort -rh | head -10

# Comprimir modelo
tar -czf modelo.tar.gz models/

# Descomprimir
tar -xzf modelo.tar.gz

# Sync de pasta (backup)
rsync -avz --progress models/ backup/models/
```

---

## Anexo B: Snippets de Código Reutilizáveis

### Setup Básico

```python
# setup.py
"""
Configuração padrão para qualquer projecto ML no M1
Copiar para início de cada script
"""
import os
import random
import numpy as np
import tensorflow as tf

# Seed para reprodutibilidade
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

set_seed(42)

# Mixed precision
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)

# Verificar GPU
gpus = tf.config.list_physical_devices('GPU')
print(f"GPUs disponíveis: {len(gpus)}")
if gpus:
    print(f"✓ GPU detectada: {gpus[0].name}")
else:
    print("⚠️  GPU não detectada!")

# Imports comuns
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt
import pandas as pd

print("✓ Setup completo!")
```

### Data Augmentation Templates

```python
# augmentation_templates.py

# ═══════════════════════════════════════════════════════════
# IMAGENS - Augmentation Leve (Documentos, texto)
# ═══════════════════════════════════════════════════════════
aug_leve = keras.Sequential([
    layers.RandomRotation(0.05),      # ±5%
    layers.RandomTranslation(0.05, 0.05),
    layers.RandomBrightness(0.1),
], name="aug_leve")


# ═══════════════════════════════════════════════════════════
# IMAGENS - Augmentation Médio (Fotografia geral)
# ═══════════════════════════════════════════════════════════
aug_medio = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.15),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.1, 0.1),
    layers.RandomContrast(0.2),
], name="aug_medio")


# ═══════════════════════════════════════════════════════════
# IMAGENS - Augmentation Forte (Datasets pequenos)
# ═══════════════════════════════════════════════════════════
aug_forte = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.3),
    layers.RandomZoom(0.3),
    layers.RandomTranslation(0.2, 0.2),
    layers.RandomContrast(0.3),
    layers.RandomBrightness(0.2),
], name="aug_forte")


# ═══════════════════════════════════════════════════════════
# TEXTO - Augmentation (Back-translation, synonims)
# ═══════════════════════════════════════════════════════════
def text_augmentation(text, prob=0.3):
    """Augmentation simples de texto"""
    import random
    
    words = text.split()
    
    # Shuffle aleatório de palavras (30% chance)
    if random.random() < prob:
        random.shuffle(words)
    
    # Duplicar palavras aleatórias (30% chance)
    if random.random() < prob and len(words) > 3:
        idx = random.randint(0, len(words)-1)
        words.insert(idx, words[idx])
    
    return ' '.join(words)
```

### Callbacks Standard

```python
# callbacks_standard.py

def get_callbacks(
    model_name="model",
    monitor='val_accuracy',
    patience_early=10,
    patience_lr=5
):
    """
    Callbacks padrão para qualquer treino
    
    Returns:
        List de callbacks prontos a usar
    """
    callbacks = [
        # Early stopping
        keras.callbacks.EarlyStopping(
            monitor=monitor,
            patience=patience_early,
            restore_best_weights=True,
            verbose=1
        ),
        
        # Reduce LR on plateau
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience_lr,
            min_lr=1e-7,
            verbose=1
        ),
        
        # Model checkpoint
        keras.callbacks.ModelCheckpoint(
            f'checkpoints/{model_name}_best.keras',
            monitor=monitor,
            save_best_only=True,
            verbose=1
        ),
        
        # TensorBoard
        keras.callbacks.TensorBoard(
            log_dir=f'logs/{model_name}',
            histogram_freq=1
        ),
        
        # CSV Logger (backup)
        keras.callbacks.CSVLogger(
            f'logs/{model_name}_history.csv'
        )
    ]
    
    return callbacks

# Uso:
# callbacks = get_callbacks("classificador_v1")
# model.fit(..., callbacks=callbacks)
```

### Data Loading Templates

```python
# data_loaders.py

# ═══════════════════════════════════════════════════════════
# Carregar Imagens de Directório
# ═══════════════════════════════════════════════════════════
def load_image_dataset(
    data_dir,
    img_size=(224, 224),
    batch_size=32,
    validation_split=0.2
):
    """Template standard para datasets de imagens"""
    
    train_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="training",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=True
    )
    
    val_ds = keras.preprocessing.image_dataset_from_directory(
        data_dir,
        validation_split=validation_split,
        subset="validation",
        seed=42,
        image_size=img_size,
        batch_size=batch_size,
        shuffle=False
    )
    
    # Optimizações
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    return train_ds, val_ds


# ═══════════════════════════════════════════════════════════
# Carregar CSV
# ═══════════════════════════════════════════════════════════
def load_csv_dataset(
    csv_path,
    text_col,
    label_col,
    test_size=0.2
):
    """Template para datasets tabulares/texto"""
    import pandas as pd
    from sklearn.model_selection import train_test_split
    
    df = pd.read_csv(csv_path)
    
    train_df, val_df = train_test_split(
        df,
        test_size=test_size,
        stratify=df[label_col],
        random_state=42
    )
    
    return train_df, val_df


# ═══════════════════════════════════════════════════════════
# TFRecord (grandes datasets)
# ═══════════════════════════════════════════════════════════
def load_tfrecord_dataset(
    tfrecord_path,
    batch_size=32
):
    """Para datasets optimizados"""
    
    dataset = tf.data.TFRecordDataset(tfrecord_path)
    
    def parse_fn(example):
        # Define o teu schema aqui
        features = {
            'image': tf.io.FixedLenFeature([], tf.string),
            'label': tf.io.FixedLenFeature([], tf.int64),
        }
        parsed = tf.io.parse_single_example(example, features)
        image = tf.io.decode_jpeg(parsed['image'])
        label = parsed['label']
        return image, label
    
    dataset = dataset.map(parse_fn, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    
    return dataset
```

### Visualização

```python
# visualization.py

def plot_training_curves(history, save_path='training_curves.png'):
    """Plot bonito de curvas de treino"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train', linewidth=2)
    ax1.plot(history.history['val_accuracy'], label='Validation', linewidth=2)
    ax1.set_title('Model Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Loss
    ax2.plot(history.history['loss'], label='Train', linewidth=2)
    ax2.plot(history.history['val_loss'], label='Validation', linewidth=2)
    ax2.set_title('Model Loss', fontsize=14, fontweight='bold')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Gráfico guardado: {save_path}")


def plot_confusion_matrix(y_true, y_pred, labels, save_path='confusion_matrix.png'):
    """Matriz de confusão bonita"""
    from sklearn.metrics import confusion_matrix
    import seaborn as sns
    import matplotlib.pyplot as plt
    
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        cm, 
        annot=True, 
        fmt='d', 
        cmap='Blues',
        xticklabels=labels,
        yticklabels=labels,
        cbar_kws={'label': 'Count'}
    )
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Matriz guardada: {save_path}")


def plot_samples(dataset, class_names, num_samples=9):
    """Mostra amostras do dataset"""
    import matplotlib.pyplot as plt
    
    plt.figure(figsize=(12, 12))
    
    for images, labels in dataset.take(1):
        for i in range(min(num_samples, len(images))):
            ax = plt.subplot(3, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i]])
            plt.axis("off")
    
    plt.tight_layout()
    plt.show()
```

---

## Anexo C: Checklist de Optimização

### Antes do Treino

```markdown
## 🔍 CHECKLIST PRÉ-TREINO

### Ambiente
- [ ] Python ARM64 verificado (`platform.machine()` → arm64)
- [ ] Ambiente virtual activado
- [ ] Todas as dependências instaladas
- [ ] Versões correctas (TF 2.16.x + Metal 1.1.x)

### Dados
- [ ] Dataset verificado e balanceado
- [ ] Split train/val/test correcto
- [ ] Sem data leakage
- [ ] Augmentation configurado
- [ ] Pipeline optimizado (prefetch, cache se aplicável)

### Modelo
- [ ] Arquitectura apropriada para tarefa
- [ ] Tamanho adequado para M1 16GB
- [ ] Pesos pré-treinados carregados (se aplicável)
- [ ] Mixed precision activado

### Configuração
- [ ] Batch size testado e optimizado
- [ ] Learning rate apropriado
- [ ] Callbacks configurados (Early Stop, ReduceLR)
- [ ] Checkpointing activo
- [ ] Seed definido (reprodutibilidade)

### Sistema
- [ ] Memória disponível >8GB
- [ ] Outras apps fechadas
- [ ] GPU/Metal verificado
- [ ] Espaço em disco >10GB livre
```

### Durante o Treino

```markdown
## ⚙️ MONITORIZAÇÃO ACTIVA

### A Cada Epoch
- [ ] Val accuracy a subir?
- [ ] Gap train-val <15%?
- [ ] Loss a descer consistentemente?
- [ ] Memória <90%?

### Sinais de Problema
- [ ] ⚠️ Val accuracy estável mas train sobe → Overfitting
- [ ] ⚠️ Ambas estáveis e baixas → Underfitting
- [ ] ⚠️ Loss explode (NaN) → LR muito alto
- [ ] ⚠️ Memória a crescer → Memory leak
- [ ] ⚠️ Muito lento → Pipeline de dados lento

### Acções Correctivas
- [ ] Se overfitting → Adiciona regularização/dropout
- [ ] Se underfitting → Modelo maior ou mais epochs
- [ ] Se OOM → Reduz batch_size
- [ ] Se lento → Verifica prefetch/paralelização
```

### Após Treino

```markdown
## ✅ PÓS-TREINO

### Avaliação
- [ ] Testado em test set (não usado no treino)
- [ ] Métricas documentadas
- [ ] Confusion matrix analisada
- [ ] Erros comuns identificados

### Optimização
- [ ] Quantização testada (se deployment)
- [ ] Pruning considerado
- [ ] Tamanho final aceitável?
- [ ] Velocidade de inferência aceitável?

### Documentação
- [ ] Modelo versionado com metadata
- [ ] Hiperparâmetros guardados
- [ ] README actualizado
- [ ] Exemplos de uso criados

### Próximos Passos
- [ ] Deploy planeado
- [ ] Melhorias identificadas
- [ ] Baseline estabelecida para futuras versões
```

---

## Anexo D: Recursos de Datasets Gratuitos

### Imagens

```markdown
## 🖼️ DATASETS DE IMAGENS

### Classificação Geral
🔗 ImageNet (via Kaggle)
   - 1.2M imagens, 1000 classes
   - Benchmark standard
   - kaggle.com/c/imagenet-object-localization-challenge

🔗 CIFAR-10 / CIFAR-100
   - 60K imagens pequenas (32x32)
   - 10/100 classes
   - Disponível via tf.keras.datasets

🔗 Fashion-MNIST
   - 70K imagens de roupa (28x28)
   - 10 classes
   - Disponível via tf.keras.datasets

### Específicos
🔗 Plantas (PlantVillage)
   - 54K imagens de doenças em plantas
   - kaggle.com/datasets/emmarex/plantdisease

🔗 Animais (Animals-10)
   - 28K imagens, 10 animais
   - kaggle.com/datasets/alessiocorrado99/animals10

🔗 Faces (CelebA)
   - 200K faces de celebridades
   - Anotações de atributos
   - kaggle.com/datasets/jessicali9530/celeba-dataset

🔗 Médico (Chest X-Ray)
   - Raios-X de tórax
   - Pneumonia detection
   - kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

### Portugueses
🔗 Azulejos Portugueses
   - github.com/delftrobotics/azulejos-dataset
   
🔗 Vinhos Portugueses (dados tabulares)
   - archive.ics.uci.edu/ml/datasets/wine+quality
```

### Texto/NLP

```markdown
## 📝 DATASETS DE TEXTO

### Português
🔗 ASSIN (Similaridade Semântica)
   - Pares de frases PT-BR e PT-PT
   - kaggle.com/datasets/assin

🔗 Reviews Produtos (PT-BR)
   - Amazon reviews em português
   - kaggle.com/datasets/fredericods/ptbr-sentiment-analysis-datasets

🔗 Notícias Portuguesas
   - Público, Observador, etc.
   - Usar web scraping (respeitar ToS)

### Inglês
🔗 IMDB Reviews
   - 50K reviews de filmes
   - Disponível via tf.keras.datasets

🔗 AG News
   - 120K notícias, 4 categorias
   - huggingface.co/datasets/ag_news

🔗 Wikipedia
   - Dumps completos
   - dumps.wikimedia.org

🔗 Common Crawl
   - Petabytes de texto web
   - commoncrawl.org

### Multilingual
🔗 OSCAR
   - Corpus web multilingual
   - Inclui português
   - huggingface.co/datasets/oscar
```

### Áudio

```markdown
## 🎵 DATASETS DE ÁUDIO

🔗 Speech Commands
   - Comandos de voz curtos
   - tensorflow.org/datasets/catalog/speech_commands

🔗 Common Voice (Mozilla)
   - Voz em múltiplas línguas
   - Inclui português
   - commonvoice.mozilla.org

🔗 LibriSpeech
   - 1000h de audiolivros
   - openslr.org/12
```

### Tabulares

```markdown
## 📊 DATASETS TABULARES

🔗 UCI Machine Learning Repository
   - Centenas de datasets
   - archive.ics.uci.edu/ml

🔗 Kaggle Datasets
   - Milhares de datasets
   - kaggle.com/datasets

🔗 Google Dataset Search
   - Motor de busca de datasets
   - datasetsearch.research.google.com

🔗 Papers With Code Datasets
   - Datasets de papers
   - paperswithcode.com/datasets
```

### Onde Procurar

```markdown
## 🔍 MOTORES DE BUSCA

1. Kaggle (kaggle.com/datasets)
   ✅ Muito datasets prontos
   ✅ Notebooks de exemplo
   ✅ APIs para download

2. Hugging Face (huggingface.co/datasets)
   ✅ Foco em NLP
   ✅ Fácil integração
   ✅ Streaming de grandes datasets

3. TensorFlow Datasets (tensorflow.org/datasets)
   ✅ Prontos para usar
   ✅ Pipeline tf.data
   ✅ Bem documentados

4. Papers With Code
   ✅ Datasets de research
   ✅ Benchmarks incluídos

5. GitHub (github.com/topics/dataset)
   ✅ Datasets especializados
   ✅ Scripts de processamento
```

---

## Anexo E: Glossário de Termos

```markdown
## 📖 GLOSSÁRIO ML/DL

### A
**Accuracy** - Percentagem de previsões correctas
**Activation Function** - Função não-linear (ReLU, sigmoid, etc.)
**Adam** - Optimizador adaptativo popular
**Augmentation** - Criação de variações de dados para treino

### B
**Backpropagation** - Algoritmo para calcular gradientes
**Batch** - Conjunto de exemplos processados juntos
**Batch Size** - Número de exemplos por batch
**Bias** - Parâmetro adicional nos neurónios / Enviesamento nos dados

### C
**Checkpoint** - Snapshot do modelo durante treino
**CNN** - Convolutional Neural Network
**Confusion Matrix** - Tabela de previsões vs realidade

### D
**Dataset** - Conjunto de dados
**Dropout** - Técnica de regularização
**Dense Layer** - Camada totalmente conectada

### E
**Early Stopping** - Para treino quando não melhora
**Embedding** - Representação vectorial de dados
**Epoch** - Uma passagem completa pelo dataset

### F
**Fine-tuning** - Ajuste fino de modelo pré-treinado
**FP16** - Float16, mixed precision
**Frozen Layers** - Camadas não-treináveis

### G
**GPU** - Graphics Processing Unit
**Gradient** - Derivada da loss em relação aos parâmetros
**Gradient Descent** - Algoritmo de optimização

### H
**Hyperparameter** - Parâmetro de configuração (LR, batch size)
**Hidden Layer** - Camada intermediária da rede

### I
**Inference** - Usar modelo treinado para previsões
**Input Shape** - Dimensões do input

### L
**Label** - Valor verdadeiro / target
**Learning Rate (LR)** - Tamanho do passo na optimização
**Loss** - Função que mede erro do modelo
**LoRA** - Low-Rank Adaptation (fine-tuning eficiente)

### M
**MPS** - Metal Performance Shaders (GPU do M1)
**Mixed Precision** - Treino com FP16+FP32

### N
**Neuron** - Unidade básica de rede neural
**Normalization** - Escalar dados para range padrão

### O
**OOM** - Out Of Memory
**Overfitting** - Modelo decora treino, não generaliza
**Optimizer** - Algoritmo que actualiza pesos (Adam, SGD)

### P
**Parameter** - Peso ou bias aprendido
**Pooling** - Redução de dimensionalidade
**Preprocessing** - Preparação de dados

### Q
**Quantization** - Redução de precisão (FP32→INT8)

### R
**ReLU** - Rectified Linear Unit (activação)
**Regularization** - Técnicas para prevenir overfitting

### S
**SGD** - Stochastic Gradient Descent
**Softmax** - Função para probabilidades de classes
**Split** - Divisão de dados (train/val/test)

### T
**Tensor** - Array multidimensional
**Transfer Learning** - Usar modelo pré-treinado
**Training Loop** - Ciclo de treino (forward/backward)

### U
**Underfitting** - Modelo não aprende o suficiente
**UMA** - Unified Memory Architecture (M1)

### V
**Validation Set** - Dados para avaliar durante treino

### W
**Weight** - Parâmetro aprendido
**Weight Decay** - Regularização L2
```

---

## Anexo F: Tabelas de Referência Rápida

### Batch Sizes Recomendados (M1 16GB)

| Modelo | Input Size | Batch Size | Memória ~|
|--------|-----------|------------|----------|
| MobileNetV2 | 224x224 | 64 | 6GB |
| EfficientNetB0 | 224x224 | 32-64 | 7GB |
| ResNet50 | 224x224 | 32 | 8GB |
| EfficientNetB3 | 300x300 | 16 | 9GB |
| ViT-Base | 224x224 | 16 | 10GB |
| DistilBERT | 128 tokens | 16 | 8GB |
| BERT-base | 128 tokens | 8 | 10GB |
| LLaMA 7B (4-bit) | - | 2 | 6GB |

### Learning Rates Típicos

| Situação | Learning Rate | Notas |
|----------|--------------|-------|
| **Treino from scratch** | 1e-3 | Padrão para Adam |
| **Transfer learning (fase 1)** | 1e-3 | Só classificador |
| **Fine-tuning (fase 2)** | 1e-5 a 1e-4 | Toda a rede |
| **BERT fine-tuning** | 2e-5 a 5e-5 | Standard |
| **LoRA LLM** | 1e-4 a 3e-4 | Mais alto que full |
| **Se loss
