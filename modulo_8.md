# Módulo 8: Boas Práticas e Troubleshooting

## 📋 Índice

### 8.1 Workflows Eficientes
- Organização de projectos
- Versionamento de modelos
- Experimentação rápida
- Reprodutibilidade

### 8.2 Problemas Comuns
- Out of Memory (OOM)
- Treino lento
- Overfitting/Underfitting
- Incompatibilidades

### 8.3 Quando Usar Cloud Computing
- Limitações do M1 16GB
- Alternativas cloud
- Estratégia híbrida

---

## 8.1 Workflows Eficientes

### Organização de Projectos

```
projeto_ml/
├── data/
│   ├── raw/              # Dados originais (nunca modificar!)
│   ├── processed/        # Dados processados
│   └── splits/           # train/val/test
├── notebooks/            # Exploração e análise
│   ├── 01_exploracao.ipynb
│   └── 02_experimentos.ipynb
├── src/                  # Código de produção
│   ├── data/
│   │   ├── loader.py
│   │   └── preprocessor.py
│   ├── models/
│   │   ├── base_model.py
│   │   └── custom_model.py
│   ├── training/
│   │   ├── trainer.py
│   │   └── callbacks.py
│   └── utils/
│       └── helpers.py
├── models/               # Modelos treinados
│   ├── checkpoints/
│   └── final/
├── logs/                 # TensorBoard, logs
├── configs/              # Configurações
│   └── config.yaml
├── tests/                # Testes unitários
├── requirements.txt
├── README.md
└── .gitignore
```

**Regras de ouro:**
- ✅ Dados raw nunca são modificados
- ✅ Código em `src/`, exploração em `notebooks/`
- ✅ Configurações em ficheiros separados
- ✅ `.gitignore` para modelos grandes

### Template .gitignore

```bash
# .gitignore
# Dados
data/raw/*
data/processed/*
*.csv
*.json
*.pkl

# Modelos (>100MB)
models/*.keras
models/*.h5
models/*.pth
*.ckpt
*.npz

# Logs
logs/*
*.log

# Python
__pycache__/
*.pyc
.ipynb_checkpoints/

# MacOS
.DS_Store

# Ambientes
venv/
.conda/
```

### Versionamento de Modelos

```python
# model_versioning.py
"""
Sistema simples de versionamento de modelos
"""
import json
from datetime import datetime
from pathlib import Path

class ModelVersioner:
    """Gere versões de modelos com metadata"""
    
    def __init__(self, base_dir="models"):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(exist_ok=True)
        self.metadata_file = self.base_dir / "versions.json"
        
        # Carregar histórico
        if self.metadata_file.exists():
            with open(self.metadata_file) as f:
                self.versions = json.load(f)
        else:
            self.versions = {}
    
    def save_model(self, model, name, metrics, hyperparams, notes=""):
        """
        Guarda modelo com metadata completa
        
        Args:
            model: Modelo treinado
            name: Nome base (ex: "classificador_caes")
            metrics: Dict com métricas (accuracy, loss, etc.)
            hyperparams: Dict com hiperparâmetros
            notes: Notas adicionais
        """
        # Gerar versão única
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"{name}_v{timestamp}"
        
        # Guardar modelo
        model_path = self.base_dir / f"{version_id}.keras"
        model.save(model_path)
        
        # Metadata
        self.versions[version_id] = {
            "name": name,
            "timestamp": timestamp,
            "path": str(model_path),
            "metrics": metrics,
            "hyperparams": hyperparams,
            "notes": notes,
            "size_mb": model_path.stat().st_size / (1024**2)
        }
        
        # Guardar metadata
        with open(self.metadata_file, 'w') as f:
            json.dump(self.versions, f, indent=2)
        
        print(f"✓ Modelo guardado: {version_id}")
        print(f"  Accuracy: {metrics.get('accuracy', 'N/A')}")
        print(f"  Tamanho: {self.versions[version_id]['size_mb']:.1f}MB")
        
        return version_id
    
    def list_versions(self, name=None):
        """Lista todas as versões (ou filtrado por nome)"""
        print("\n📊 VERSÕES DE MODELOS")
        print("="*80)
        
        versions = self.versions.items()
        if name:
            versions = [(k,v) for k,v in versions if v['name'] == name]
        
        for version_id, info in sorted(versions, key=lambda x: x[1]['timestamp'], reverse=True):
            print(f"\n{version_id}")
            print(f"  Data: {info['timestamp']}")
            print(f"  Accuracy: {info['metrics'].get('accuracy', 'N/A'):.4f}")
            print(f"  Tamanho: {info['size_mb']:.1f}MB")
            if info['notes']:
                print(f"  Notas: {info['notes']}")
    
    def load_best(self, name, metric='accuracy'):
        """Carrega melhor modelo baseado numa métrica"""
        import tensorflow as tf
        
        candidates = [(k,v) for k,v in self.versions.items() if v['name'] == name]
        if not candidates:
            raise ValueError(f"Nenhum modelo encontrado com nome: {name}")
        
        best = max(candidates, key=lambda x: x[1]['metrics'].get(metric, 0))
        best_id, best_info = best
        
        print(f"📥 Carregando melhor modelo: {best_id}")
        print(f"  {metric}: {best_info['metrics'][metric]:.4f}")
        
        return tf.keras.models.load_model(best_info['path'])

# Uso
versioner = ModelVersioner()

# Após treinar
versioner.save_model(
    model=model,
    name="classificador_caes",
    metrics={'accuracy': 0.89, 'val_accuracy': 0.85, 'loss': 0.31},
    hyperparams={'lr': 1e-3, 'batch_size': 32, 'epochs': 30},
    notes="Transfer learning + fine-tuning, dataset balanceado"
)

# Listar
versioner.list_versions("classificador_caes")

# Carregar melhor
best_model = versioner.load_best("classificador_caes", metric='val_accuracy')
```

### Configurações Centralizadas

```yaml
# configs/config.yaml
# Centraliza todos os hiperparâmetros

project:
  name: "classificador_caes"
  seed: 42
  
data:
  path: "dataset_caes"
  img_size: [224, 224]
  batch_size: 32
  val_split: 0.2
  
  augmentation:
    horizontal_flip: true
    rotation: 0.2
    zoom: 0.2
    brightness: 0.1

model:
  architecture: "efficientnetb0"
  pretrained: true
  num_classes: 10
  dropout: 0.3
  dense_units: 256

training:
  epochs: 50
  learning_rate: 0.001
  optimizer: "adam"
  loss: "sparse_categorical_crossentropy"
  
  callbacks:
    early_stopping:
      patience: 10
      monitor: "val_accuracy"
    
    reduce_lr:
      factor: 0.5
      patience: 5
      min_lr: 1.0e-7

hardware:
  mixed_precision: true
  gpu_memory_limit: 12  # GB
```

```python
# config_loader.py
import yaml
from pathlib import Path

class Config:
    """Carrega e valida configurações"""
    
    def __init__(self, config_path="configs/config.yaml"):
        with open(config_path) as f:
            self.config = yaml.safe_load(f)
        
        self.validate()
    
    def validate(self):
        """Valida configurações"""
        # Verificar campos obrigatórios
        required = ['project', 'data', 'model', 'training']
        for field in required:
            if field not in self.config:
                raise ValueError(f"Campo obrigatório ausente: {field}")
        
        # Validar valores
        if self.config['data']['batch_size'] > 128:
            print("⚠️  batch_size >128 pode causar OOM no M1")
        
        if self.config['training']['learning_rate'] > 0.01:
            print("⚠️  Learning rate muito alto!")
    
    def get(self, path, default=None):
        """Acede valores com dot notation: config.get('data.batch_size')"""
        keys = path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def __getitem__(self, key):
        return self.config[key]

# Uso
config = Config()
batch_size = config.get('data.batch_size')
lr = config.get('training.learning_rate')
```

### Experimentação Rápida

```python
# experiment_tracker.py
"""
Sistema leve para tracking de experimentos
Alternativa simples ao W&B quando offline
"""
import json
import time
from pathlib import Path
from datetime import datetime

class ExperimentTracker:
    """Regista experimentos localmente"""
    
    def __init__(self, project_name, log_dir="experiments"):
        self.project_name = project_name
        self.log_dir = Path(log_dir) / project_name
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Criar experimento
        self.exp_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.exp_dir = self.log_dir / self.exp_id
        self.exp_dir.mkdir()
        
        self.metrics = []
        self.start_time = time.time()
        
        print(f"🧪 Experimento iniciado: {self.exp_id}")
    
    def log_params(self, params):
        """Regista hiperparâmetros"""
        with open(self.exp_dir / "params.json", 'w') as f:
            json.dump(params, f, indent=2)
    
    def log_metric(self, name, value, step=None):
        """Regista métrica"""
        self.metrics.append({
            'name': name,
            'value': value,
            'step': step,
            'timestamp': time.time() - self.start_time
        })
    
    def log_metrics(self, metrics_dict, step=None):
        """Regista múltiplas métricas"""
        for name, value in metrics_dict.items():
            self.log_metric(name, value, step)
    
    def save_model(self, model, name="model.keras"):
        """Guarda modelo no experimento"""
        model_path = self.exp_dir / name
        model.save(model_path)
        return model_path
    
    def finish(self, status="completed", notes=""):
        """Finaliza experimento"""
        duration = time.time() - self.start_time
        
        summary = {
            'exp_id': self.exp_id,
            'status': status,
            'duration_seconds': duration,
            'notes': notes,
            'final_metrics': {
                name: value 
                for name, value, *_ in 
                [m.values() for m in self.metrics[-10:]]
            }
        }
        
        with open(self.exp_dir / "summary.json", 'w') as f:
            json.dump(summary, f, indent=2)
        
        with open(self.exp_dir / "metrics.json", 'w') as f:
            json.dump(self.metrics, f, indent=2)
        
        print(f"✓ Experimento concluído: {self.exp_id}")
        print(f"  Duração: {duration/60:.1f} min")

# Uso no treino
tracker = ExperimentTracker("classificador_caes")

# Registar hiperparâmetros
tracker.log_params({
    'learning_rate': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'architecture': 'efficientnetb0'
})

# Durante treino
for epoch in range(epochs):
    # ... treino ...
    
    tracker.log_metrics({
        'loss': train_loss,
        'accuracy': train_acc,
        'val_loss': val_loss,
        'val_accuracy': val_acc
    }, step=epoch)

# No fim
tracker.save_model(model)
tracker.finish(status="completed", notes="Melhor modelo até agora!")
```

### Reprodutibilidade

```python
# reproducibility.py
"""
Garante reprodutibilidade de experimentos
"""
import os
import random
import numpy as np
import tensorflow as tf

def set_seed(seed=42):
    """
    Define seed global para reprodutibilidade
    
    ⚠️ IMPORTANTE: Chama isto ANTES de qualquer treino!
    """
    # Python
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # TensorFlow
    tf.random.set_seed(seed)
    
    # Para operações determínicas (mais lento mas reprodutível)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    print(f"✓ Seed global definido: {seed}")
    print("⚠️  Treino será determinístico (mais lento)")

def get_system_info():
    """Regista informação do sistema para debug"""
    import platform
    import tensorflow as tf
    
    info = {
        'python_version': platform.python_version(),
        'platform': platform.platform(),
        'processor': platform.processor(),
        'tensorflow_version': tf.__version__,
        'gpu_available': len(tf.config.list_physical_devices('GPU')) > 0,
        'mixed_precision': tf.keras.mixed_precision.global_policy().name
    }
    
    return info

# Uso no início do script
set_seed(42)
system_info = get_system_info()
print(json.dumps(system_info, indent=2))
```

---

## 8.2 Problemas Comuns

### Out of Memory (OOM)

**Sintomas:**
- Kernel crashes
- Mensagem "ResourceExhaustedError"
- Sistema congela

**Diagnóstico:**
```python
# check_memory.py
import psutil
import tensorflow as tf

def check_memory_usage():
    """Verifica uso actual de memória"""
    mem = psutil.virtual_memory()
    
    print("💾 MEMÓRIA DO SISTEMA")
    print(f"Total:      {mem.total / 1e9:.1f} GB")
    print(f"Disponível: {mem.available / 1e9:.1f} GB")
    print(f"Usada:      {mem.used / 1e9:.1f} GB ({mem.percent}%)")
    
    if mem.available < 8e9:  # <8GB
        print("⚠️  ALERTA: Pouca memória disponível!")
        return False
    
    return True

# Durante treino
class MemoryCallback(tf.keras.callbacks.Callback):
    """Monitoriza memória durante treino"""
    
    def on_epoch_end(self, epoch, logs=None):
        mem = psutil.virtual_memory()
        print(f"\nMemória após epoch {epoch}: {mem.percent}%")
        
        if mem.percent > 90:
            print("⚠️  Memória >90%! Considera parar treino")
```

**Soluções (em ordem de prioridade):**

1. **Reduzir batch_size**
```python
# De 32 para 16
batch_size = 16  

# Ou usar gradient accumulation para simular batch maior
def train_with_grad_accum(model, data, accum_steps=4):
    optimizer.zero_grad()
    for i, batch in enumerate(data):
        loss = model(batch) / accum_steps
        loss.backward()
        
        if (i + 1) % accum_steps == 0:
            optimizer.step()
            optimizer.zero_grad()
```

2. **Mixed Precision**
```python
# Economiza ~50% RAM
policy = tf.keras.mixed_precision.Policy('mixed_float16')
tf.keras.mixed_precision.set_global_policy(policy)
```

3. **Limpar cache**
```python
import gc
import tensorflow as tf

# Após cada epoch
gc.collect()
tf.keras.backend.clear_session()
```

4. **Reduzir tamanho do modelo**
```python
# Usar modelo mais leve
model = tf.keras.applications.MobileNetV2()  # em vez de ResNet50
```

### Treino Lento

**Diagnóstico:**
```python
import time
import tensorflow as tf

def benchmark_pipeline(dataset, num_steps=100):
    """Mede velocidade do pipeline de dados"""
    
    start = time.time()
    for i, batch in enumerate(dataset.take(num_steps)):
        if i == 0:
            first_batch_time = time.time() - start
        pass
    total_time = time.time() - start
    
    print(f"⏱️  BENCHMARK")
    print(f"Primeiro batch: {first_batch_time:.2f}s (compila grafo)")
    print(f"Total {num_steps} batches: {total_time:.2f}s")
    print(f"Média por batch: {total_time/num_steps:.3f}s")
    print(f"Batches/segundo: {num_steps/total_time:.1f}")
```

**Soluções:**

1. **Prefetch e Cache**
```python
# Carrega próximo batch enquanto treina
dataset = dataset.prefetch(tf.data.AUTOTUNE)

# Cache dataset pequenos na RAM
dataset = dataset.cache()
```

2. **Verificar GPU**
```python
# GPU deve estar activa
print(tf.config.list_physical_devices('GPU'))

# Se vazio, reinstala Metal
# pip uninstall tensorflow-metal
# pip install tensorflow-metal==1.1.0
```

3. **Optimizar I/O**
```python
# Paralelizar carregamento
dataset = dataset.map(
    preprocess_fn,
    num_parallel_calls=tf.data.AUTOTUNE
)

# Reduzir tamanho de imagens se possível
img_size = (224, 224)  # em vez de (512, 512)
```

### Overfitting

**Sintomas:**
- Train accuracy alta, val accuracy baixa
- Gap >10% entre train e validation

**Diagnóstico:**
```python
def plot_training_history(history):
    """Visualiza overfitting"""
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # Accuracy
    ax1.plot(history.history['accuracy'], label='Train')
    ax1.plot(history.history['val_accuracy'], label='Validation')
    ax1.set_title('Accuracy')
    ax1.legend()
    ax1.grid(True)
    
    # Loss  
    ax2.plot(history.history['loss'], label='Train')
    ax2.plot(history.history['val_loss'], label='Validation')
    ax2.set_title('Loss')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_curve.png')
    
    # Detectar overfitting
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    gap = final_train_acc - final_val_acc
    
    if gap > 0.1:
        print(f"⚠️  OVERFITTING DETECTADO!")
        print(f"   Gap train-val: {gap:.2%}")
```

**Soluções:**

1. **Mais dados / Data Augmentation**
```python
augmentation = tf.keras.Sequential([
    tf.keras.layers.RandomFlip("horizontal"),
    tf.keras.layers.RandomRotation(0.3),      # Aumenta
    tf.keras.layers.RandomZoom(0.3),          # Aumenta
    tf.keras.layers.RandomTranslation(0.2, 0.2),
    tf.keras.layers.RandomContrast(0.2),
])
```

2. **Regularização**
```python
model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.Dropout(0.5),  # Aumenta dropout
    tf.keras.layers.Dense(256, kernel_regularizer=tf.keras.regularizers.l2(0.01)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(num_classes)
])
```

3. **Early Stopping**
```python
early_stop = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss',
    patience=5,  # Reduz patience
    restore_best_weights=True
)
```

### Underfitting

**Sintomas:**
- Train e val accuracy ambas baixas
- Loss não desce

**Soluções:**

1. **Modelo maior**
```python
# Mais camadas/neurónios
dense_units = 512  # era 256
```

2. **Learning rate**
```python
# Testar LRs diferentes
for lr in [1e-5, 1e-4, 1e-3, 1e-2]:
    model.compile(optimizer=tf.keras.optimizers.Adam(lr))
    # treinar...
```

3. **Mais epochs**
```python
epochs = 100  # em vez de 30
```

---

## 8.3 Quando Usar Cloud Computing

### Limitações do M1 16GB

| Tarefa | M1 16GB | Recomendação |
|--------|---------|--------------|
| **Modelos <1B params** | ✅ Excelente | Local |
| **Fine-tuning 7B (LoRA)** | ✅ Viável | Local |
| **Fine-tuning 13B+** | ⚠️ Difícil | Cloud |
| **Treino from scratch LLM** | ❌ Impossível | Cloud |
| **Datasets >50GB** | ❌ Lento | Cloud |
| **Batch processing grande** | ⚠️ Lento | Cloud |

### Alternativas Cloud

**Google Colab (Gratuito/Pro)**
```python
# Vantagens:
# - GPU T4 gratuita
# - 12GB RAM (gratuito), 25GB (Pro)
# - Setup zero

# Desvantagens:
# - Sessões limitadas (12h gratuito)
# - Pode desligar aleatoriamente
# - Dados perdidos se não guardar

# Quando usar: Experimentação rápida, tutoriais
```

**Kaggle Notebooks (Gratuito)**
```python
# Vantagens:
# - GPU P100/T4 30h/semana
# - 16GB RAM
# - Datasets públicos integrados

# Desvantagens:
# - 9h por sessão
# - Internet limitada

# Quando usar: Competições, datasets públicos
```

**AWS/GCP/Azure**
```python
# Vantagens:
# - GPUs potentes (A100, V100)
# - Escalável
# - Controlo total

# Desvantagens:
# - Pago (caro!)
# - Setup complexo

# Quando usar: Produção, modelos grandes
```

### Estratégia Híbrida

**Desenvolvimento local + Treino cloud:**

```python
# hybrid_workflow.py
"""
Desenvolve local, treina na cloud
"""

# 1. Desenvolver e testar local (M1)
# - Usa subset pequeno (10% dados)
# - Testa pipeline completo
# - Debugging

# 2. Quando pronto, enviar para cloud
# - Código versionado (git)
# - Configurações em YAML
# - Scripts automatizados

# 3. Treinar na cloud
# - Dataset completo
# - Múltiplos GPUs se necessário
# - Checkpoints para S3/GCS

# 4. Download modelo final para M1
# - Inferência local
# - Deployment
```

**Script para sync:**
```bash
#!/bin/bash
# sync_to_cloud.sh

# Upload código
rsync -avz --exclude='data/' --exclude='models/' \
    ./ user@cloud-instance:/home/user/projeto/

# Upload dados (se pequenos)
rsync -avz data/processed/ \
    user@cloud-instance:/home/user/projeto/data/

# Executar treino remoto
ssh user@cloud-instance \
    "cd /home/user/projeto && python train.py --config cloud_config.yaml"

# Download modelo treinado
rsync -avz user@cloud-instance:/home/user/projeto/models/final/ \
    ./models/final/
```

### Decisão: Local vs Cloud

**Usa M1 16GB quando:**
- ✅ Modelo <7B parâmetros
- ✅ Dataset <20GB
- ✅ Fine-tuning com LoRA
- ✅ Prototipagem
- ✅ Inferência
- ✅ Desenvolvimento

**Usa Cloud quando:**
- ✅ Modelo >13B parâmetros
- ✅ Dataset >50GB
- ✅ Treino from scratch
- ✅ Múltiplas experiências paralelas
- ✅ Deadline apertado
- ✅ Produção de alta escala

---

## 📋 Checklist Final de Boas Práticas

### Antes de Começar
- [ ] Ambiente virtual activado
- [ ] Configurações centralizadas (YAML)
- [ ] .gitignore configurado
- [ ] Seed definido para reprodutibilidade

### Durante Desenvolvimento
- [ ] Código organizado (src/ vs notebooks/)
- [ ] Testes com subset pequeno primeiro
- [ ] Versionamento de modelos activo
- [ ] Tracking de experimentos

### Durante Treino
- [ ] Mixed precision activada
- [ ] Callbacks configurados (Early Stop, ReduceLR)
- [ ] Monitorização de recursos
- [ ] Checkpoints a guardar

### Após Treino
- [ ] Modelo versionado com metadata
- [ ] Métricas documentadas
- [ ] Curvas de treino analisadas
- [ ] README actualizado

### Troubleshooting
- [ ] Logs guardados
- [ ] Configurações documentadas
- [ ] Testes de reprodutibilidade
- [ ] Plano B (cloud) se necessário

---

## 🎯 Resumo

Dominas agora:
- ✅ Workflows profissionais
- ✅ Versionamento de modelos
- ✅ Troubleshooting sistemático
- ✅ Quando escalar para cloud

**Próximos passos:**
1. Aplica estas práticas nos teus projectos
2. Cria templates reutilizáveis
3. Automatiza processos repetitivos
4. Documenta aprendizagens

**Lembra-te:**
> "Horas de debugging podem poupar minutos de planeamento" 🙃