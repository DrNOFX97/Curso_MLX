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

```