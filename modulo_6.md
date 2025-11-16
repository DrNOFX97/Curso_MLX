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