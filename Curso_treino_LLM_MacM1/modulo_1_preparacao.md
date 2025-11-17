# Curso: Como Treinar Modelos de Aprendizagem Automática (Machine Learning) no MacBook Pro M1 16GB

## Módulo 1: Preparação do Ambiente

### 1.1 Introdução ao chip Apple Silicon M1

#### Arquitetura ARM vs x86
O chip M1 da Apple representa uma mudança fundamental na computação pessoal:

**Arquitetura ARM:**
- O M1 usa arquitetura ARM64, não x86 como os processadores Intel
- Mais eficiente energeticamente (RISC vs CISC)
- Alguns programas precisam de ser recompilados ou usar Rosetta 2 (camada de tradução)
- Para ML: bibliotecas nativas ARM têm desempenho muito superior

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
- Chip dedicado para operações de ML
- 11 TFLOPS de desempenho
- Usado principalmente para inferência (Core ML)
- Menos flexível que GPU, mas muito eficiente

#### Vantagens para Machine Learning

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

## Metal: A Base do Aceleramento em Chips Apple Silicon

### Introdução ao Metal

O Metal é a API gráfica e de computação de baixo nível da Apple, lançada em 2014. Ao contrário de APIs de alto nível, o Metal proporciona acesso quase direto à GPU, minimizando overhead e maximizando o desempenho. Para treino de LLMs em Mac M1, o Metal é **fundamental** porque:

- É a única forma de aceder à GPU integrada nos chips Apple Silicon
- Permite aproveitar a arquitetura unificada de memória (Unified Memory Architecture)
- Fornece aceleração hardware para operações de machine learning

## Metal vs. CUDA: Diferenças Chave

| Aspeto | Metal | CUDA |
|--------|-------|------|
| **Fabricante** | Apple | NVIDIA |
| **Plataformas** | macOS, iOS, iPadOS | Windows, Linux |
| **Linguagem Shaders** | Metal Shading Language (MSL) | CUDA C/C++ |
| **Memória** | Unificada (CPU+GPU) | Separada (requer cópias) |

## Metal no Contexto do MLX

O MLX foi desenvolvido especificamente pela Apple para aproveitar o Metal:

```python
import mlx.core as mx

# O MLX usa Metal automaticamente
array = mx.array([1, 2, 3, 4])
# Esta operação executa na GPU via Metal
resultado = array * 2
```

### Vantagens da Arquitetura Unificada

No M1, CPU e GPU partilham a mesma memória RAM. Isto significa:

1. **Zero-copy operations**: Não há necessidade de transferir dados entre memórias
2. **Maior eficiência**: Menos latência e maior bandwidth
3. **Melhor uso de memória**: Os 8-16GB são partilhados inteligentemente

```python
# Com CUDA (NVIDIA):
# 1. Alocar memória na GPU
# 2. Copiar dados CPU → GPU
# 3. Processar
# 4. Copiar resultados GPU → CPU

# Com Metal (M1):
# 1. Processar (dados já acessíveis!)
```

## Metal Performance Shaders (MPS)

O PyTorch e outros frameworks usam MPS, uma camada sobre o Metal otimizada para ML:

```python
import torch

# Verifica se MPS está disponível
if torch.backends.mps.is_available():
    device = torch.device("mps")
    tensor = torch.randn(1000, 1000).to(device)
    # Operações aceleradas por Metal
    resultado = torch.matmul(tensor, tensor.T)
```

## Limitações do Metal para ML

Apesar das vantagens, há limitações a considerar:

1. **Memória partilhada**: Os 8-16GB são divididos entre CPU e GPU
2. **Menos maduro que CUDA**: Ecossistema menor, menos otimizações
3. **Debugging**: Ferramentas menos desenvolvidas
4. **Compatibilidade**: Alguns modelos podem não funcionar otimamente

**Ponto-chave**: O Metal não é apenas uma "alternativa ao CUDA" - é uma arquitectura fundamentalmente diferente que aproveita as características únicas dos chips Apple Silicon. Compreender esta diferença é essencial para otimizar o treino de LLMs no M1.

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
|---------|-----------------|-----------------|
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

```
