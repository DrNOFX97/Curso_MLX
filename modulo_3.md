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

```