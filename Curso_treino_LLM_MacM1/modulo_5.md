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
        
        print("✓ LoRA configurado!")
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