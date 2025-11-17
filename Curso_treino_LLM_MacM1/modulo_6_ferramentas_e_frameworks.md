## Módulo 6: Ferramentas e Frameworks Especializados

### 6.1 MLX Framework da Apple (Avançado)

#### Arquitetura e Filosofia do MLX

**MLX** foi criado pela Apple Research especificamente para Apple Silicon. Diferente do PyTorch/TensorFlow que foram adaptados, o MLX foi **desenhado desde o início** para unified memory.

**Princípios de Design:**
1. **Familiar**: API similar a NumPy e PyTorch
2. **Composable**: Fácil adicionar novas operações
3. **Efficient**: Totalmente optimizado para Metal
4. **Unified Memory**: Aproveita UMA ao máximo

**Vantagens Técnicas:**

```python
# comparacao_mlx.py
"""
Por que MLX é mais rápido no M1?

1. LAZY EVALUATION
   - PyTorch: Executa operações imediatamente
   - MLX: Constrói grafo e optimiza antes de executar

2. UNIFIED MEMORY NATIVA
   - PyTorch: Ainda pensa em CPU/GPU separados
   - MLX: Arquitectura unificada desde o início

3. COMPILAÇÃO AUTOMÁTICA
   - PyTorch: Requer torch.compile() explícito
   - MLX: JIT compilation por padrão
"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np

# Exemplo de lazy evaluation
a = mx.array([1, 2, 3])
b = mx.array([4, 5, 6])
c = a + b  # Não executa ainda!

# Força execução
result = mx.eval(c)  # Agora sim executa, optimizado
print(result)
```

#### MLX Core - Operações Fundamentais

```python
# mlx_core_basico.py
import mlx.core as mx

class IntroducaoMLX:
    """
    Introdução às operações básicas do MLX
    """
    
    def __init__(self):
        print("🍎 MLX Core - Apple Silicon Native")
        print("=" * 60)
    
    def arrays_basicos(self):
        """Arrays e operações básicas"""
        print("\n1️⃣ ARRAYS BÁSICOS")
        
        # Criar arrays
        a = mx.array([1, 2, 3, 4])
        b = mx.zeros((3, 3))
        c = mx.ones((2, 2))
        d = mx.random.normal((5, 5))
        
        print(f"Array simples: {a}")
        print(f"Shape: {a.shape}, dtype: {a.dtype}")
        
        # Operações vectorizadas
        resultado = a * 2 + 10
        print(f"Operações: {resultado}")
    
    def operacoes_matriz(self):
        """Operações de álgebra linear"""
        print("\n2️⃣ ÁLGEBRA LINEAR")
        
        # Multiplicação de matrizes
        A = mx.random.normal((100, 100))
        B = mx.random.normal((100, 100))
        
        # @ é matmul em MLX (como NumPy)
        C = A @ B
        mx.eval(C)  # Força computação
        
        print(f"Multiplicação de matrizes: {C.shape}")
        
        # Outras operações
        inversa = mx.linalg.inv(A)
        autovalores = mx.linalg.eigvalsh(A)
        
        print("✓ Operações de álgebra linear disponíveis")
    
    def gradientes(self):
        """Diferenciação automática"""
        print("\n3️⃣ DIFERENCIAÇÃO AUTOMÁTICA")
        
        # Função simples
        def f(x):
            return x ** 2 + 2 * x + 1
        
        # Calcular gradiente
        grad_f = mx.grad(f)
        
        x = mx.array(3.0)
        gradiente = grad_f(x)
        
        print(f"f(x) = x² + 2x + 1")
        print(f"f'(3) = {gradiente}")  # Deve ser 2*3 + 2 = 8
        
        # Gradientes de funções com múltiplos argumentos
        def g(x, y):
            return x ** 2 + y ** 2
        
        # Gradiente em relação a x
        grad_x = mx.grad(g, argnums=0)
        # Gradiente em relação a y
        grad_y = mx.grad(g, argnums=1)
        
        print(f"∂g/∂x(2,3) = {grad_x(mx.array(2.0), mx.array(3.0))}")
        print(f"∂g/∂y(2,3) = {grad_y(mx.array(2.0), mx.array(3.0))}")
    
    def transformacoes(self):
        """Transformações funcionais (vmap, etc.)"""
        print("\n4️⃣ TRANSFORMAÇÕES FUNCIONAIS")
        
        # vmap: vectoriza função automaticamente
        def f(x):
            return x ** 2
        
        # Aplicar a um batch
        batch = mx.array([1, 2, 3, 4, 5])
        
        # Sem vmap (loop manual)
        resultado_loop = mx.array([f(x) for x in batch])
        
        # Com vmap (paralelo e optimizado)
        f_vectorizado = mx.vmap(f)
        resultado_vmap = f_vectorizado(batch)
        
        print(f"Resultado: {resultado_vmap}")
        print("✓ vmap permite vectorização automática")

# Demonstração
intro = IntroducaoMLX()
intro.arrays_basicos()
intro.operacoes_matriz()
intro.gradientes()
intro.transformacoes()
```

#### MLX NN - Redes Neurais

```python
# mlx_nn_exemplo.py
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim

class MLPSimples(nn.Module):
    """
    Multi-Layer Perceptron em MLX
    """
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        
        # Definir camadas
        self.layers = [
            nn.Linear(input_dim, hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Linear(hidden_dim, output_dim)
        ]
    
    def __call__(self, x):
        # Forward pass
        for i, layer in enumerate(self.layers[:-1]):
            x = layer(x)
            x = nn.relu(x)
        
        # Última camada sem activação
        x = self.layers[-1](x)
        return x

class TreinadorMLX:
    """
    Treino de modelo com MLX
    """
    def __init__(self, model):
        self.model = model
        self.optimizer = optim.Adam(learning_rate=0.001)
    
    def loss_fn(self, model, X, y):
        """Função de perda"""
        predictions = model(X)
        return nn.losses.mse_loss(predictions, y)
    
    def step(self, X, y):
        """Um passo de treino"""
        # Calcular loss e gradientes
        loss_and_grad = nn.value_and_grad(self.model, self.loss_fn)
        loss, grads = loss_and_grad(self.model, X, y)
        
        # Actualizar pesos
        self.optimizer.update(self.model, grads)
        
        # Avaliar (forçar computação)
        mx.eval(self.model.parameters(), self.optimizer.state)
        
        return loss
    
    def treinar(self, X_train, y_train, epochs=100):
        """Loop de treino"""
        print("🏋️ Treinando modelo com MLX...")
        
        for epoch in range(epochs):
            loss = self.step(X_train, y_train)
            
            if (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        
        print("✓ Treino concluído!")

# Exemplo de uso
print("📊 Exemplo de Treino com MLX")
print("=" * 60)

# Criar dados sintéticos
n_samples = 1000
X = mx.random.normal((n_samples, 10))
y = mx.random.normal((n_samples, 1))

# Criar modelo
model = MLPSimples(input_dim=10, hidden_dim=64, output_dim=1)

# Treinar
treinador = TreinadorMLX(model)
treinador.treinar(X, y, epochs=50)

# Teste
X_test = mx.random.normal((5, 10))
predictions = model(X_test)
print(f"\nPrevisões de teste: {predictions}")
```

#### MLX LM - Modelos de Linguagem

```python
# mlx_lm_avancado.py
"""
Trabalhar com LLMs usando MLX
"""
from mlx_lm import load, generate
from mlx_lm.utils import load_config
import mlx.core as mx

class GestorLLM_MLX:
    """
    Gestor para trabalhar com LLMs em MLX
    """
    def __init__(self, modelo="mlx-community/Mistral-7B-v0.1-4bit"):
        self.modelo_path = modelo
        self.model = None
        self.tokenizer = None
        self.config = None
    
    def carregar(self):
        """Carrega modelo optimizado para M1"""
        print(f"📥 Carregando {self.modelo_path}...")
        
        # Carregar modelo e tokenizer
        self.model, self.tokenizer = load(self.modelo_path)
        self.config = load_config(self.modelo_path)
        
        # Informação do modelo
        print(f"✓ Modelo carregado!")
        print(f"  Parâmetros: {self.config.get('model_type', 'N/A')}")
        print(f"  Quantização: {self.config.get('quantization', 'N/A')}")
    
    def gerar(self, prompt, max_tokens=200, temperatura=0.7, top_p=0.9):
        """
        Gera texto com controlo fino
        """
        response = generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperatura,
            top_p=top_p,
            verbose=False
        )
        
        return response
    
    def gerar_batch(self, prompts, max_tokens=200):
        """
        Gera múltiplas respostas em paralelo
        Aproveita unified memory!
        """
        respostas = []
        
        for prompt in prompts:
            resposta = self.gerar(prompt, max_tokens=max_tokens)
            respostas.append(resposta)
        
        return respostas
    
    def benchmark_velocidade(self, prompt, num_tokens=100):
        """
        Mede velocidade de geração (tokens/segundo)
        """
        import time
        
        print(f"\n⚡ Benchmark de Velocidade")
        print("=" * 60)
        
        start = time.time()
        resposta = self.gerar(prompt, max_tokens=num_tokens)
        tempo = time.time() - start
        
        # Contar tokens gerados
        tokens_gerados = len(self.tokenizer.encode(resposta)) - len(self.tokenizer.encode(prompt))
        tokens_por_segundo = tokens_gerados / tempo
        
        print(f"Tempo total: {tempo:.2f}s")
        print(f"Tokens gerados: {tokens_gerados}")
        print(f"Velocidade: {tokens_por_segundo:.1f} tokens/s")
        
        return tokens_por_segundo

# Demonstração
print("🍎 MLX LM - Optimizado para Apple Silicon")
print("=" * 60)

gestor = GestorLLM_MLX()
gestor.carregar()

# Teste de geração
prompt = "Explica o que é o MLX framework em 2 frases:"
resposta = gestor.gerar(prompt, max_tokens=100)
print(f"\n📝 Resposta:\n{resposta}\n")

# Benchmark
gestor.benchmark_velocidade(
    "Conta-me uma história sobre",
    num_tokens=100
)
```

#### Conversão de Modelos para MLX

```python
# converter_modelos_mlx.py
"""
Converter modelos HuggingFace para MLX
"""
from mlx_lm import convert
import os

class ConversorMLX:
    """
    Converte e optimiza modelos para MLX
    """
    def __init__(self):
        self.modelos_suportados = [
            "llama", "mistral", "phi", "gemma", 
            "qwen", "stablelm", "mixtral"
        ]
    
    def converter_modelo(
        self, 
        hf_path,
        mlx_path="./modelo_mlx",
        quantize=True,
        q_bits=4,
        q_group_size=64
    ):
        """
        Converte modelo para formato MLX
        
        Args:
            hf_path: Caminho ou ID do modelo HuggingFace
            mlx_path: Onde guardar modelo MLX
            quantize: Se deve quantizar
            q_bits: Bits de quantização (2, 4, 8)
            q_group_size: Tamanho do grupo para quantização
        """
        print(f"🔄 Convertendo {hf_path} para MLX...")
        print("=" * 60)
        
        # Converter
        convert(
            hf_path=hf_path,
            mlx_path=mlx_path,
            quantize=quantize,
            q_group_size=q_group_size,
            q_bits=q_bits
        )
        
        # Verificar tamanho
        tamanho_mb = sum(
            os.path.getsize(os.path.join(mlx_path, f)) 
            for f in os.listdir(mlx_path)
        ) / (1024**2)
        
        print(f"\n✓ Conversão completa!")
        print(f"  Localização: {mlx_path}")
        print(f"  Tamanho: {tamanho_mb:.1f} MB")
        print(f"  Quantização: {q_bits}-bit")
        
        return mlx_path
    
    def optimizar_para_m1(self, modelo_path):
        """
        Recomendações de optimização para M1 16GB
        """
        print(f"\n💡 RECOMENDAÇÕES DE OPTIMIZAÇÃO")
        print("=" * 60)
        
        recomendacoes = {
            "2GB-4GB": {
                "q_bits": 4,
                "q_group_size": 64,
                "desc": "Ideal para M1 16GB - máxima eficiência"
            },
            "4GB-8GB": {
                "q_bits": 4,
                "q_group_size": 128,
                "desc": "Bom balanço qualidade/velocidade"
            },
            "8GB+": {
                "q_bits": 8,
                "q_group_size": 64,
                "desc": "Melhor qualidade (mais lento)"
            }
        }
        
        for tamanho, config in recomendacoes.items():
            print(f"\n{tamanho}:")
            print(f"  q_bits: {config['q_bits']}")
            print(f"  q_group_size: {config['q_group_size']}")
            print(f"  {config['desc']}")

# Exemplo de uso
conversor = ConversorMLX()

# Converter LLaMA 7B para MLX
modelo_mlx = conversor.converter_modelo(
    hf_path="meta-llama/Llama-2-7b-hf",
    mlx_path="./llama2-7b-mlx-4bit",
    quantize=True,
    q_bits=4,
    q_group_size=64
)

# Ver recomendações
conversor.optimizar_para_m1(modelo_mlx)
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
import os # Import os for os.path.getsize

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
# model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=50,
#     callbacks=cbks
# )

print("\n📊 Para ver os resultados:")
print(f"   tensorboard --logdir={monitor.log_dir}")
```

#### Weights & Biases (wandb)

```python
# wandb_integracao.py
import wandb
from wandb.keras import WandbCallback
from datetime import datetime # Already imported above, but good to have here for clarity

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
# config = {
#     'learning_rate': 0.001,
#     'batch_size': 32,
#     'epochs': 50,
#     'arquitetura': 'MobileNetV2',
#     'dataset': 'custom_animals'
# }

# monitor = MonitorizacaoWandB(projeto="classificador-animais")
# monitor.inicializar(config)

# Treinar
# history = model.fit(
#     train_dataset,
#     validation_data=val_dataset,
#     epochs=config['epochs'],
#     callbacks=[monitor.criar_callback()]
# )
# Registar modelo final
# monitor.registar_modelo('modelo_final.keras')

# Terminar
# wandb.finish()
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

```

---