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

```