# Curso Completo: Treinamento de Modelos de Machine Learning em Apple Silicon (M1/M2/M3)

![Apple Silicon](https://img.shields.io/badge/Apple_Silicon-M1%2FM2%2FM3-black?style=for-the-badge&logo=apple)
![Python](https://img.shields.io/badge/Python-3.11%2B-blue?style=for-the-badge&logo=python)
![Frameworks](https://img.shields.io/badge/Frameworks-TensorFlow%20%7C%20PyTorch%20%7C%20MLX-orange?style=for-the-badge)

## 📖 Sobre o Curso

Este repositório contém o material completo para o curso "Como Treinar Modelos de Aprendizagem Automática no MacBook Pro M1". O objetivo é capacitar desenvolvedores e entusiastas de IA a extrair o máximo de performance da arquitetura Apple Silicon para tarefas de Machine Learning, desde a configuração do ambiente até o fine-tuning de Large Language Models (LLMs).

O conteúdo está estruturado em módulos sequenciais, projetados para construir uma base sólida e avançar para tópicos complexos de forma gradual.

## 📚 Tópicos Abordados

- **Configuração de Ambiente:** Preparação otimizada com Homebrew, Miniforge e ambientes virtuais.
- **Frameworks Acelerados:** Uso de TensorFlow (Metal), PyTorch (MPS) e o novo MLX da Apple.
- **Gestão de Memória:** Estratégias para trabalhar com a Arquitetura de Memória Unificada (UMA) de 16GB.
- **Treinamento de Modelos:** Classificação de imagens (CNNs), NLP (Transformers) e modelos tabulares (Gradient Boosting).
- **Otimização Avançada:** Técnicas de quantização (PTQ, QAT), pruning e knowledge distillation.
- **Large Language Models (LLMs):** Fine-tuning de modelos de até 7B de parâmetros com LoRA, QLoRA e o framework MLX.
- **Deployment:** Conversão de modelos para Core ML e criação de APIs REST e interfaces com Streamlit.
- **Projetos Práticos:** Três projetos completos para aplicar todo o conhecimento adquirido.

## 🎯 Público-Alvo

Este curso é ideal para:
- Desenvolvedores com um MacBook (M1, M2, M3) que desejam entrar na área de IA.
- Estudantes de ciência de dados que buscam otimizar seus workflows em hardware local.
- Profissionais de ML que querem aproveitar a eficiência energética e de performance do Apple Silicon.

## ✅ Pré-requisitos

- Um MacBook com chip Apple Silicon (M1, M2, M3).
- Conhecimento básico de Python e da linha de comando.
- Familiaridade com conceitos fundamentais de Machine Learning.

---

## 🚀 Estrutura do Curso

### [Módulo 1: Preparação do Ambiente](./modulo_1.md)
- Introdução à arquitetura Apple Silicon (ARM vs x86, GPU, Neural Engine).
- Configuração inicial com Homebrew e Miniforge.
- Instalação e teste de frameworks otimizados: TensorFlow-metal, PyTorch-MPS e JAX.

### [Módulo 2: Gestão de Recursos e Limitações](./modulo_2.md)
- Compreensão da Memória Unificada (UMA) de 16GB.
- Técnicas de monitoramento de memória e performance.
- Otimização de memória: Batch Size, Gradient Accumulation, Mixed Precision (FP16).
- Gerenciamento eficiente de datasets (Data Generators e Streaming).

### [Módulo 3: Treino de Modelos Pequenos e Médios](./modulo_3.md)
- Classificação de Imagens com CNNs (MobileNet, EfficientNet) e Transfer Learning.
- Processamento de Linguagem Natural (NLP) com modelos compactos (DistilBERT).
- Treinamento de modelos para dados tabulares com XGBoost, LightGBM e CatBoost.

### [Módulo 4: Técnicas Avançadas de Otimização](./modulo_4.md)
- Quantização de modelos (Post-Training e Quantization-Aware Training).
- Pruning (poda) de redes neurais para compressão.
- Knowledge Distillation para transferir conhecimento de modelos grandes para pequenos.
- Estratégias de treino eficiente: Learning Rate Scheduling, Early Stopping e Checkpointing.

### [Módulo 5: Modelos de Linguagem Grandes (LLMs)](./modulo_5.md)
- Estratégias para trabalhar com LLMs em 16GB de RAM.
- Fine-tuning eficiente com LoRA e QLoRA.
- Introdução ao MLX, o framework de ML da Apple.
- Exemplo prático de fine-tuning de um modelo 7B no M1.

### [Módulo 6: Deployment e Produção](./modulo_6.md)
- Introdução ao Core ML para deployment em dispositivos Apple.
- Conversão de modelos Keras e PyTorch para o formato Core ML.
- Monitoramento e debugging com TensorBoard e Weights & Biases (wandb).
- Profiling de performance para identificar gargalos.

### [Módulo 7: Projetos Práticos](./modulo_7.md)
- **Projeto 1:** Classificador de Imagens (10 classes) com deployment via API REST.
- **Projeto 2:** Análise de Sentimentos em português com BERT e interface em Streamlit.
- **Projeto 3:** Fine-tuning de um LLM (Mistral 7B) para um domínio específico e criação de um chatbot.

### [Módulo 8: Boas Práticas e Troubleshooting](./modulo_8.md)
- Workflows eficientes para experimentação e reprodutibilidade.
- Solução de problemas comuns (Out of Memory, lentidão, etc.).
- Estratégias para escalar para a nuvem quando necessário.

### [Módulo 9: Recursos Adicionais](./modulo_9.md)
- Links para comunidades, documentação e leitura complementar.
- Discussão sobre o futuro do ML em edge devices.

### [Anexos](./anexos.md)
- Comandos úteis, snippets de código e checklists de otimização.

---

## 🤝 Contribuições

Contribuições são bem-vindas! Se você encontrar um erro, tiver uma sugestão de melhoria ou quiser adicionar um novo conteúdo, sinta-se à vontade para abrir uma **Issue** ou um **Pull Request**.

## 📄 Licença

Este projeto é distribuído sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.
