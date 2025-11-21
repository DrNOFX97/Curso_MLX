# MÓDULO 1: Fundamentos do Desenvolvimento com IA

⏱️ **Duração:** 20 minutos  
📊 **Formato:** Teórico com exemplos práticos  
🎯 **Objetivo:** Compreender o contexto e possibilidades do desenvolvimento com IA

---

## 1.1 O que é Vibe Coding / Desenvolvimento com IA
**Duração:** 7 minutos

### Definição Prática e Aplicada

**O que É:**
- Método de desenvolvimento onde describes o que queres em linguagem natural
- A IA gera código funcional baseado nas tuas instruções
- Processo iterativo: gera → testa → refina → repete
- Foco no "o quê" em vez do "como"

**O que NÃO É:**
- ❌ Substituição completa de programadores
- ❌ Magia que resolve tudo sem esforço
- ❌ Ferramenta que dispensa pensamento crítico
- ❌ Solução para quem não sabe o que quer construir

### Programação Tradicional vs Desenvolvimento com IA

| Aspecto | Tradicional | Com IA |
|---------|-------------|--------|
| **Conhecimento necessário** | Sintaxe, frameworks, best practices | O que queres construir, lógica básica |
| **Tempo de aprendizagem** | Meses/anos | Dias/semanas |
| **Velocidade inicial** | Lenta (setup, configuração) | Muito rápida (protótipo em minutos) |
| **Controlo** | Total sobre cada linha | Orientas mas não controlas tudo |
| **Debugging** | Manual, linha a linha | Describes o erro à IA |
| **Manutenção** | Precisas entender o código | IA ajuda a entender e modificar |

### Casos de Uso: Quando Usar

✅ **Usar Desenvolvimento com IA quando:**
- Precisas de protótipos rápidos
- Queres automatizar tarefas repetitivas
- Tens uma ideia mas não sabes por onde começar
- Precisas de aprender uma nova tecnologia/framework
- Queres focar no problema de negócio, não na sintaxe
- Projetos pequenos a médios (MVPs, ferramentas internas)

❌ **Evitar quando:**
- Projetos mission-critical (sistemas bancários, saúde)
- Performance extrema é crítica
- Segurança máxima é requisito (dados sensíveis)
- Sistemas legados complexos
- Compliance rigoroso com standards específicos

### Exemplos Reais de Aplicação

**Caso 1: Automatização em RH**
- Problema: Processar 50 CVs mensalmente e extrair info relevante
- Solução IA: Script Python que lê PDFs, extrai dados, organiza em Excel
- Tempo: 30min a criar vs 4h/mês a fazer manualmente

**Caso 2: Dashboard para Loja**
- Problema: Dono de loja quer ver vendas em tempo real
- Solução IA: App web que lê CSV de vendas e mostra gráficos
- Tempo: 2h a criar vs semanas e €€€ a contratar developer

**Caso 3: Ferramenta Interna de Marketing**
- Problema: Equipa precisa gerar relatórios de campanhas
- Solução IA: Ferramenta que agrega dados de várias fontes
- Tempo: 1 dia a criar vs semana a desenvolver manualmente

---

## 1.2 Ecossistema de Ferramentas
**Duração:** 12 minutos

O ecossistema de ferramentas de IA para desenvolvimento explodiu em 2024-2025. Existem agora dezenas de opções, cada uma com os seus pontos fortes. Vamos explorar as principais categorias e ferramentas.

---

### 🤖 CATEGORIA 1: Assistentes Conversacionais (Chat AI)

Estas são plataformas web onde conversas com IA para gerar código, resolver problemas, e aprender.

#### **ChatGPT (OpenAI)** - O Veterano

**O que é:**
- O pioneiro que tornou a IA mainstream
- Modelos atuais: GPT-4o (mais recente), GPT-4.1, o3 (raciocínio)
- Interface conversacional simples e intuitiva

**Pontos Fortes:**
✅ Mais popular e testado (milhões de utilizadores)  
✅ Excelente para explicações e ensino  
✅ Gera código em qualquer linguagem  
✅ Grande comunidade e recursos online  
✅ Plugins e integrações extensas  

**Limitações:**
❌ Conhecimento limitado a dados de treino (cutoff)  
❌ Às vezes "alucina" (inventa informação)  
❌ Versão gratuita limitada  

**Custos (2025):**
- **Gratuito**: GPT-4o-mini com limites de mensagens
- **Plus (€20/mês)**: GPT-4o ilimitado, acesso a o3, imagens
- **Pro (€200/mês)**: Uso intensivo, prioridade máxima

**Melhor para:**
- Iniciantes a aprender programação
- Gerar scripts rápidos
- Brainstorming de ideias
- Explicar código complexo

---

#### **Claude (Anthropic)** - O Pensador

**O que é:**
- Desenvolvido pela Anthropic (fundada por ex-OpenAI)
- Família Claude 4: Opus 4 (mais poderoso), Sonnet 4.5 (equilibrado), Haiku 4.5 (rápido)
- Contexto de 200.000 tokens (equivalente a ~150.000 palavras)

**Pontos Fortes:**
✅ Melhor do mundo em coding (82% no benchmark SWE-bench)  
✅ Contexto gigante (consegue ler livros inteiros)  
✅ Recurso "Artifacts" para criar código interativo  
✅ Modo "extended thinking" para raciocínio profundo  
✅ Claude Code: ferramenta CLI para delegar tarefas de código  
✅ Mais cuidadoso e menos propenso a erros  

**Limitações:**
❌ Menos conhecido que ChatGPT  
❌ Interface mais limitada (sem plugins nativos)  
❌ Às vezes demasiado cauteloso (recusa pedidos benignos)  

**Custos (2025):**
- **Gratuito**: Claude Sonnet 4 com limites
- **Pro (€20/mês)**: Sonnet 4.5 + Opus 4, extended thinking
- **Max (€60/mês)**: Uso muito estendido + features experimentais

**Melhor para:**
- Projetos de código complexos e grandes
- Análise de código existente extenso
- Quando precisas de raciocínio profundo
- Refactoring de aplicações inteiras
- Tarefas autónomas que requerem várias horas de foco

---

#### **Perplexity AI** - O Investigador

**O que é:**
- Motor de busca conversacional com IA que pesquisa a web em tempo real
- Usa múltiplos modelos (GPT-5, Claude 4.5, Gemini, Mistral) dependendo da tarefa
- 780 milhões de queries mensais (Maio 2025)

**Pontos Fortes:**
✅ Respostas com citações de fontes web atualizadas  
✅ Perfeito para pesquisa e fact-checking  
✅ Funcionalidades financeiras (stocks, earnings)  
✅ Integrações: email, calendário, Notion, GitHub  
✅ Browser Comet com IA integrada  
✅ Acesso a informação atualizada (não tem cutoff)  

**Limitações:**
❌ Menos focado em geração de código extenso  
❌ Controvérsias sobre scraping e copyright  
❌ Melhor para pesquisa que para programação pura  

**Custos (2025):**
- **Gratuito**: Acesso a modelos mistos com pesquisa em tempo real
- **Pro (€20/mês)**: Modelos avançados, modo "Deep Research"
- **Enterprise**: Recursos enterprise

**Melhor para:**
- Pesquisa de informação atual
- Verificar factos e tendências
- Encontrar APIs e bibliotecas
- Aprender sobre tecnologias novas
- Análise de notícias tech e mercado

---

#### **Microsoft Copilot** (anteriormente Bing Chat)

**O que é:**
- Assistente IA da Microsoft integrado em Windows, Edge, Microsoft 365
- Baseado em modelos OpenAI (GPT-4)

**Pontos Fortes:**
✅ Grátis e integrado no Windows 11/Edge  
✅ Acesso a web em tempo real  
✅ Integração profunda com Microsoft 365  
✅ Geração de imagens (DALL-E)  

**Limitações:**
❌ Menos focado em programação que outras ferramentas  
❌ Interface menos personalizável  
❌ Ecossistema Microsoft-centric  

**Melhor para:**
- Utilizadores do ecossistema Microsoft
- Pesquisa web com contexto de IA
- Produtividade geral (não só código)

---

#### **Gemini (Google)** - Antes Bard

**O que é:**
- Assistente IA do Google
- Integrado com Google Search e serviços Google

**Pontos Fortes:**
✅ Acesso direto a Google Search  
✅ Integração com Gmail, Docs, Drive  
✅ Multimodal (texto, imagem, vídeo)  
✅ Grátis para uso básico  

**Limitações:**
❌ Historicamente menos forte em código que ChatGPT/Claude  
❌ Privacidade (Google usa dados para treino)  

**Melhor para:**
- Quem usa muito o ecossistema Google
- Pesquisa e produtividade geral
- Análise de imagens e vídeos

---

### 💻 CATEGORIA 2: IDEs com IA Integrada - "Os Pesos Pesados"

Editores de código construídos do zero para IA, não apenas plugins. O mercado explodiu em finais de 2024/2025 com a entrada de gigantes como Google e ByteDance.

---

#### **🆕 Google Antigravity** - O Mais Recente (Nov 2025)

**O que é:**
- Resposta direta do Google ao Cursor
- Lançado há poucos dias (Novembro 2025)
- Abordagem "Agent-first": agentes que operam editor, terminal e browser simultaneamente

**Pontos Fortes:**
✅ **Gemini 3 Pro** - modelo mais recente com desempenho superior em coding  
✅ **Mission Control** - interface para gerir múltiplos agentes a trabalhar em paralelo  
✅ **GRATUITO durante preview** - estratégia agressiva de entrada no mercado  
✅ **Multi-modelo** - surpreendentemente suporta Claude Sonnet 4.5 e GPT-OSS  
✅ Agentes controlam browser/terminal autonomamente  
✅ Poder de infraestrutura do Google  

**Limitações:**
❌ **Muito novo** - lançado há dias, pode ter bugs de estabilidade  
❌ Ainda em "public preview"  
❌ Ecossistema Google-centric (otimizado para infraestrutura Google)  
❌ Documentação e comunidade ainda pequenas  
❌ Desconhecido como será pricing pós-preview  

**Custos (2025):**
- **Preview: GRATUITO** (aproveita agora!)
- Pricing futuro: ainda não anunciado

**Destaque IA:**
🚀 **Agentes autónomos** que operam browser + terminal + editor ao mesmo tempo. Isto é literalmente o futuro - não estás só a escrever código, estás a gerir "workers" de IA.

**Melhor para:**
- Early adopters que querem testar o cutting-edge
- Quem usa ecossistema Google (Cloud, Firebase, etc.)
- Projetos que beneficiam de agentes autónomos
- Experimentação com Gemini 3

**Não é para ti se:**
- ❌ Precisas de estabilidade production-grade já
- ❌ Tens receio de ferramentas muito novas
- ❌ Empresa requer ferramentas maduras e testadas

**📊 Status:** 🔥 **QUENTE** - Acabado de lançar, toda a gente a experimentar

---

#### **Windsurf (Codeium / OpenAI)** - O Adquirido

**O que é:**
- IDE que se solidificou como top choice após aquisição pela OpenAI
- Fork do VS Code com foco em "Flows" e contexto profundo
- Sistema "Cascade" que entende arquitetura de projeto

**Pontos Fortes:**
✅ **Integração OpenAI** - acesso prioritário a modelos next-gen (possivelmente GPT-5)  
✅ **Cascade Flow** - consciência de contexto superior, raramente "alucina"  
✅ Indexa projeto de forma superior ao Cursor  
✅ **Preço mais simples** e competitivo para equipas  
✅ UI limpa, beginner-friendly  
✅ Consciência automática de contexto  
✅ Riptide: pesquisa milhões de linhas em segundos  

**Limitações:**
❌ **Mais lento** que Cursor (prioriza precisão vs velocidade)  
❌ Créditos premium esgotam rápido  
❌ Comunidade menor que Cursor  
❌ Menos "features mágicas" que Antigravity  

**Custos (2025):**
- **Gratuito**: Trial Pro 2 semanas + SWE-1-lite ilimitado
- **Pro (€10-15/mês)**: Acesso a modelos premium
- **Enterprise**: Custom pricing

**Destaque IA:**
🎯 **"Cascade"** - sistema de contexto que raramente se perde na arquitetura do projeto. Melhor escolha para **grandes codebases**.

**Melhor para:**
- Grandes codebases e monorepos
- Equipas que precisam precisão > velocidade
- Onboarding de novos devs
- Quem quer evitar "alucinações" de IA

---

#### **Cursor** - O Pioneiro Líder

**O que é:**
- O que definiu a categoria "AI Editor"
- Referência de velocidade e fluidez
- Avaliado em $9 mil milhões

**Pontos Fortes:**
✅ **Cursor Composer** (Ctrl+I/K) - escreve apps inteiras via prompt  
✅ **"Shadow Cursor"** - previsão de edição (Tab) ainda é a melhor do mercado  
✅ Velocidade imbatível de digitação  
✅ Base de utilizadores enorme (pioneiro)  
✅ Melhor edição multi-ficheiro  
✅ Auto-completion via Supermaven (mais rápido)  
✅ Suporte a múltiplos modelos (GPT-4, Claude, Gemini)  

**Limitações:**
❌ **Custo** - plano Pro essencial (gratuito muito limitado)  
❌ Preocupações de privacidade corporativa (envia código para cloud)  
❌ Modelo muda em todas instâncias abertas  
❌ Mudou para pricing baseado em tokens  

**Custos (2025):**
- **Gratuito**: 500 pedidos/mês → queue lenta
- **Pro (€20/mês)**: ~225 Sonnet 4, ~550 Gemini, ~650 GPT 4.1
- **Business (€40/user/mês)**: Features de equipa

**Destaque IA:**
⚡ **Velocidade de edição preditiva** - a "sensação" de escrever código é incomparável. O Tab do Cursor prevê não só a próxima palavra, mas o próximo bloco de código inteiro.

**Melhor para:**
- Developers experientes que querem velocidade máxima
- Projetos complexos multi-ficheiro
- Quem quer a melhor experiência de "flow" ao codar
- Equipas dispostas a pagar €20/user

---

#### **🆕 Trae (ByteDance/TikTok)** - O Gratuito Agressivo

**O que é:**
- Aposta da ByteDance (dona do TikTok) no mercado de AI IDEs
- Estratégia agressiva: totalmente gratuito com features premium
- Foco em "vibe coding" e Builder Mode

**Pontos Fortes:**
✅ **TOTALMENTE GRATUITO** (atualmente) - até Claude 3.5 Sonnet sem custo!  
✅ **Builder Mode** - transforma requisitos em código estruturado  
✅ Interface visual muito elogiada  
✅ UX limpa e moderna  
✅ Acesso grátis a modelos premium que outros cobram  
✅ Boa para iniciantes (zero investment)  

**Limitações:**
❌ **Ecossistema de extensões menor** que VS Code  
❌ **Desconfiança de privacidade** - ByteDance/China, empresas receosas  
❌ Não se sabe quanto tempo será gratuito  
❌ Menos testado que Cursor/Windsurf  
❌ Incerteza sobre uso de dados  

**Custos (2025):**
- **GRATUITO** (estratégia de entrada no mercado)
- Futuro pricing: desconhecido

**Destaque IA:**
🎁 **Builder Mode gratuito** + acesso a Claude 3.5 Sonnet sem pagar. Isto é insustentável a longo prazo, mas por agora é incrível.

**Melhor para:**
- Quem quer testar AI coding sem investir
- Estudantes e hobbyistas
- Projetos pessoais
- Quem não trabalha com código sensível/corporativo

**Não é para ti se:**
- ❌ Trabalhas com código confidencial de empresa
- ❌ Empresa tem políticas contra ferramentas chinesas
- ❌ Precisas de garantias enterprise

**⚠️ Aviso:** ByteDance = TikTok. Se a tua empresa ou país tem restrições, verifica primeiro.

---

### 🔐 CATEGORIA 2B: IDEs com Foco em Privacidade

Para quem **não quer** enviar código para servidores Google/OpenAI/Microsoft.

#### **Void** - Open Source Total

**O que é:**
- Fork open-source do VS Code
- Alternativa ao Cursor para quem quer privacidade total

**Pontos Fortes:**
✅ **Privacidade 100%** - código nunca sai da tua máquina  
✅ **BYOM** (Bring Your Own Model) - conecta teus próprios LLMs  
✅ Suporta Ollama (modelos locais)  
✅ Ou usa tuas API keys diretas  
✅ **Custo zero** (pagas só API se usares cloud models)  
✅ Open-source e transparente  

**Limitações:**
❌ **Requer configuração técnica** - não é "instalar e usar"  
❌ Recursos de IA dependem do modelo que conectares  
❌ Sem "mágica" proprietária de indexação  
❌ Menos polido que soluções comerciais  

**Melhor para:**
- Developers que valorizam privacidade acima de tudo
- Trabalho com código confidencial
- Empresas com políticas de dados rigorosas
- Quem tem conhecimento técnico para configurar

---

#### **PearAI** - Open Source Comunitário

**O que é:**
- Alternativa open-source apoiada pela Y Combinator
- Desenvolvimento transparente e guiado pela comunidade

**Pontos Fortes:**
✅ Comunitário e transparente  
✅ Facilita uso de modelos open-source  
✅ Y Combinator backing  

**Limitações:**
❌ Ainda em "beta" comparado aos comerciais  
❌ Menos polido  

---

### 💎 CATEGORIA 2C: Os Clássicos

#### **VS Code + GitHub Copilot** - O Padrão Universal

**Status:** A base instalada é gigante - centenas de milhões de developers.

**Pontos Fortes:**
✅ **Ecossistema infinito** de extensões  
✅ Gratuito e suportado pela Microsoft  
✅ Copilot cada vez melhor (agent + multi-file editing)  
✅ Toda a gente conhece  
✅ Estabilidade comprovada  
✅ 50 pedidos/mês grátis (2025)  

**Limitações:**
❌ IA ainda parece "plugin" (não tão fluido como Cursor/Antigravity)  
❌ Não é AI-native como os outros  

**Melhor para:**
- Quem já usa VS Code e não quer mudar
- Equipas conservadoras
- Quando precisas do ecossistema VS Code

---

#### **Zed** - O Velocista

**O que é:**
- Editor focado em velocidade extrema (escrito em Rust)
- Abre projetos gigantes em milissegundos

**Pontos Fortes:**
✅ **INSTANTÂNEO** - performance absurda  
✅ Colaboração multiplayer nativa  
✅ Agora com Zed AI integrado  
✅ Perfeito para projetos gigantes  

**Limitações:**
❌ Ecossistema de plugins menor  
❌ IA menos "agente" e mais "assistente"  

**Melhor para:**
- Quem odeia editores lentos
- Projetos massivos
- Pair programming remoto

---

### 📊 TABELA COMPARATIVA DOS "PESOS PESADOS" (Nov 2025)

| IDE | Criador | Melhor Para | Custo | Destaque IA | Status |
|-----|---------|-------------|-------|-------------|--------|
| **Antigravity** 🆕 | Google | Agentes autónomos | Grátis (preview) | Agentes Browser/Terminal + Gemini 3 | 🔥 Novíssimo |
| **Windsurf** | OpenAI/Codeium | Contexto & Precisão | €15/mês | Cascade Flow & Deep Context | ✅ Sólido |
| **Cursor** | Anysphere | Velocidade de edição | €20/mês | Shadow Cursor (Tab) & Composer | 👑 Líder |
| **Trae** 🆕 | ByteDance | Grátis com features premium | **GRÁTIS** | Builder Mode & Claude grátis | ⚠️ Privacidade? |
| **Void** | Open Source | Privacidade 100% | Grátis | BYOM - teu modelo | 🔐 Privado |
| **VS Code + Copilot** | Microsoft | Estabilidade geral | €10/mês | Universal, testado | 📊 Padrão |
| **Zed** | Zed Industries | Performance | Grátis | Velocidade Rust | ⚡ Rápido |

---

### 🎯 RECOMENDAÇÃO DE ESCOLHA (Nov 2025)

#### **Quer experimentar o futuro AGORA?**
→ **Google Antigravity** (grátis durante preview)
- Agentes controlando browser/terminal é o próximo nível
- Aproveita enquanto é grátis
- Gemini 3 é excelente

#### **Quer estabilidade e precisão de código?**
→ **Windsurf** (€15/mês)
- Melhor para grandes projetos
- Menos bugs que Cursor
- Boa relação qualidade/preço

#### **Quer simplesmente codar RÁPIDO?**
→ **Cursor** (€20/mês)
- Sensação de digitação imbatível
- Tab completion mágico
- Vale cada cêntimo se és dev profissional

#### **Quer experimentar sem gastar nada?**
→ **Trae** (grátis... por agora)
- Todas as features premium grátis
- **MAS**: se trabalhas com código sensível, pensa duas vezes

#### **Privacidade é prioridade absoluta?**
→ **Void** (open-source)
- Código nunca sai da tua máquina
- Requer setup técnico
- Para paranóicos de segurança (no bom sentido!)

---

### ⚡ NOVIDADES DE ÚLTIMA HORA (Nov 2025)

**O mercado está INSANO agora:**
- Google lançou Antigravity há dias
- ByteDance (Trae) está a dar tudo grátis
- OpenAI adquiriu Codeium (Windsurf)
- Cursor continua a dominar mas tem pressão

**Tendência clara:** Agentes autónomos que fazem mais que só sugerir código.

**Próximos 6 meses:** Espera mais features de "agente" em todos. A IA vai começar a executar testes, fazer deploys, e gerir infraestrutura autonomamente.

---

---

### 🔧 CATEGORIA 3: Extensões e Ferramentas Integradas

Ferramentas que se integram no teu editor existente (VS Code, JetBrains, etc.)

#### **GitHub Copilot** - A Extensão Padrão

**O que é:**
- Extensão para VS Code, Visual Studio, JetBrains, Vim/Neovim
- Desenvolvido por GitHub + OpenAI
- Suporta chat, voz, visão (imagens), e capacidades de agente

**Pontos Fortes:**
✅ 50 pedidos/mês gratuitos (2025)  
✅ Integração profunda com GitHub (Issues, PRs, Actions)  
✅ Auto-gera resumos de PR e reviews de código  
✅ Copilot Vision: input de screenshots para debugging  
✅ Funciona no teu editor atual (sem mudar)  
✅ Acesso a GPT-4.5, Claude Opus 4, o3 no tier Pro+  
✅ Estável, testado, enterprise-ready  

**Limitações:**
❌ Menos features de "full agent" que Cursor/Windsurf  
❌ Edição multi-ficheiro ainda básica  
❌ Não é um IDE completo  

**Custos (2025):**
- **Gratuito**: 50 pedidos/mês, grátis para estudantes/open-source
- **Individual (€10/mês)**: Uso pessoal estendido
- **Pro+ (€39/mês)**: Todos os modelos cutting-edge
- **Business (€19/utilizador/mês)**: Features de equipa
- **Enterprise**: Compliance, security avançada

**Melhor para:**
- Equipas já no ecossistema GitHub/Microsoft
- Quem não quer mudar de editor
- Empresas que precisam segurança/compliance provada
- Coding diário com autocomplete inteligente

---

#### **Cline (anteriormente Claude Dev)**

**O que é:**
- Extensão VS Code que usa modelos Claude
- Foco em autonomia e tarefas complexas

**Pontos Fortes:**
✅ Gratuito (usas a tua API key)  
✅ Controlo total sobre modelo usado  
✅ Open-source  

**Limitações:**
❌ Precisa de API key própria (custos variáveis)  
❌ Menos polido que soluções pagas  

**Melhor para:**
- Quem quer máximo controlo
- Developers que preferem pagar API diretamente

---

### 🌐 CATEGORIA 4: Plataformas de "Vibe Coding"

Ferramentas que transformam ideias em apps funcionais através de prompts, com pouco ou zero código manual. Estas são as mais populares em 2025.

---

#### **Bolt.new (StackBlitz)** - Para Developers

**O que é:**
- IDE completo no browser que gera apps full-stack
- Foco em React, Next.js, Tailwind, Node.js
- "Vibe coding" mais técnico e com controlo

**Pontos Fortes:**
✅ **Controlo total sobre código gerado** - podes editar livremente  
✅ UI moderna e limpa com bom design  
✅ Rápido para iterar com mudanças pequenas (diffs)  
✅ Zero setup - tudo no browser  
✅ Suporta frontend + backend (Node.js)  
✅ Preview em tempo real  
✅ Partilha facilmente (link)  
✅ Exporta projeto completo  

**Limitações:**
❌ **Curva de aprendizado técnica** - precisa entender React/Next/Tailwind  
❌ **Gera bugs frequentemente** que precisam depuração manual  
❌ Não resolve sozinho lógica de negócio complexa  
❌ Backend é básico (não substitui arquitetura real)  
❌ Performance limitada para projetos grandes  

**Custos:**
- Gratuito com limites
- Pro para projetos maiores e mais recursos

**Perfil Ideal:**
- ⚡ Developers que querem acelerar prototipagem
- 🎨 Quem conhece React e quer UI rápido
- 🔧 Pessoas que não têm medo de editar código
- 📦 Projetos que precisam de frontend + backend simples

**Não é para ti se:**
- ❌ Nunca programaste (vai ser frustrante)
- ❌ Queres app 100% pronto sem mexer em nada
- ❌ Precisas de backend complexo/escalável

---

#### **Lovable.dev** (anteriormente GPT Engineer) - Para Não-Devs

**O que é:**
- Plataforma de criação de apps para empreendedores e não-programadores
- Experiência guiada com planeamento de produto integrado
- Foco em MVPs e protótipos funcionais rápidos

**Pontos Fortes:**
✅ **Pensado para não-developers** - interface muito intuitiva  
✅ **Experiência guiada** - ajuda a planear o produto  
✅ Prototipagem visual rápida  
✅ **Integrações prontas**: Stripe (pagamentos), Supabase (database)  
✅ **Deploy simplificado** - poucos cliques para publicar  
✅ Gera documentação automática  
✅ Ideal para validar ideias rapidamente  

**Limitações:**
❌ **Menos controlo fino** sobre o código  
❌ Personalização avançada é difícil  
❌ **Qualidade depende muito do prompt** - resultados inconsistentes  
❌ Bugs de código difíceis de corrigir se não sabes programar  
❌ Pode ficar caro para projetos além de protótipo  
❌ Menos flexível que Bolt para devs  

**Custos:**
- Free tier limitado
- Pro: ~€20-40/mês (pricing varia)
- Enterprise: custom

**Perfil Ideal:**
- 💡 Empreendedores que querem validar MVP
- 🚀 Founders não-técnicos
- 🎯 Quem quer app funcional sem aprender código
- ⏱️ Pressa em testar ideias no mercado

**Não é para ti se:**
- ❌ És developer experiente (Bolt/Cursor são melhores)
- ❌ Precisas de controlo total do código
- ❌ Vais escalar para app complexa (melhor código custom)

---

#### **v0.dev (Vercel)** - Para UI/Components

**O que é:**
- Gerador especializado de componentes UI em React/Next.js
- Foco exclusivo em frontend de alta qualidade
- Da mesma empresa que criou Next.js e Vercel

**Pontos Fortes:**
✅ **Qualidade de código superior** - mais limpo que Bolt/Lovable  
✅ Componentes React bem organizados e profissionais  
✅ **Menos "gambiarras"** em CSS/layout  
✅ Export para Shadcn/UI (biblioteca popular)  
✅ Integração perfeita com Next.js/Vercel  
✅ Preview instantâneo e iteração rápida  
✅ Aceita screenshots como input  

**Limitações:**
❌ **Só frontend** - zero backend/database/lógica  
❌ **Requer ambiente de dev** configurado (VS Code, Node.js, Git)  
❌ **Não é "no-code"** - precisas saber usar GitHub/terminal  
❌ Créditos mensais limitados no free tier  
❌ Não cria apps completas, só partes da UI  

**Custos:**
- Freemium com créditos mensais
- Pro: créditos adicionais

**Perfil Ideal:**
- 👨‍💻 Developers frontend (React/Next.js)
- 🎨 Quem quer UI bonita rapidamente
- 🏗️ Projetos que já têm backend definido
- 🔨 Pessoas que vão integrar em codebase existente

**Não é para ti se:**
- ❌ Não sabes programar (precisa setup técnico)
- ❌ Queres app completa (frontend + backend)
- ❌ És iniciante absoluto em dev

---

#### **Banani** - Para Designers (UI/UX)

**O que é:**
- Ferramenta focada em design e prototipagem visual
- Transforma prompts em telas editáveis
- Export para Figma para refinamento

**Pontos Fortes:**
✅ **Excelente para UX/UI** - foco em design, não código  
✅ Gera protótipos visuais muito rápidos  
✅ **Export para Figma** - continua o trabalho lá  
✅ Ideal para designers que não programam  
✅ Boa para explorar variações de design rapidamente  
✅ Interface intuitiva para não-técnicos  

**Limitações:**
❌ **Não gera apps funcionais** - só mockups  
❌ Não é ferramenta de desenvolvimento  
❌ **Depende de outro stack** para virar produto real  
❌ Precisa de developers para implementar o design  
❌ Não cuida de lógica, dados, backend  

**Custos:**
- Varia (verificar site oficial)

**Perfil Ideal:**
- 🎨 Designers UI/UX
- 🖼️ Quem faz protótipos para mostrar a clientes
- 👥 Equipas que separam design de development
- 📱 Validação rápida de conceitos visuais

**Não é para ti se:**
- ❌ Queres app funcional (usa Bolt/Lovable)
- ❌ És developer (podes começar direto em código)
- ❌ Não tens equipa de dev para implementar depois

---

### 🔄 Outras Menções Rápidas

#### **Replit Agent**
- IDE online com IA que cria apps completas
- Colaboração em tempo real
- Deploy com um clique
- Database integrada
- Suporta 50+ linguagens
- **Melhor para:** Aprender, projetos educacionais, colaboração remota

#### **Claude Artifacts**
- Feature dentro do Claude.ai
- Cria HTML/React interativo em janela dedicada
- Preview em tempo real
- **Melhor para:** Protótipos rápidos, demos simples

---

### 📊 COMPARAÇÃO: Escolhe a Certa Para Ti

| Aspecto | Bolt.new | Lovable | v0.dev | Banani |
|---------|----------|---------|--------|--------|
| **Público** | Devs | Não-devs | Devs React | Designers |
| **Curva aprendizado** | Média-Alta | Baixa | Média | Baixa |
| **Controlo código** | ⭐⭐⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ (só design) |
| **Full-stack?** | Sim (básico) | Sim | Não (só UI) | Não |
| **Qualidade código** | ⭐⭐⭐ | ⭐⭐ | ⭐⭐⭐⭐⭐ | N/A |
| **Facilidade uso** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Deploy fácil** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐ (só export) |
| **Preço** | Freemium | €20-40/mês | Freemium | Varia |

---

### 🎯 DECISÃO RÁPIDA: Qual Usar?

#### **Usa Bolt.new se:**
- ✅ Sabes programar (pelo menos básico de React)
- ✅ Queres controlo total sobre o código
- ✅ Vais editar e depurar manualmente
- ✅ Projeto precisa frontend + backend simples

#### **Usa Lovable se:**
- ✅ Não sabes programar
- ✅ Queres MVP funcional rápido
- ✅ Precisas Stripe/Supabase integrado
- ✅ Foco em validar ideia, não em tech perfeita

#### **Usa v0.dev se:**
- ✅ És dev React/Next.js experiente
- ✅ Só precisas de componentes UI de qualidade
- ✅ Já tens backend/infraestrutura definida
- ✅ Queres código limpo para integrar em projeto existente

#### **Usa Banani se:**
- ✅ És designer, não developer
- ✅ Queres protótipos visuais para apresentar
- ✅ Vais passar para Figma depois
- ✅ Tens equipa de dev que vai implementar

---

### ⚠️ REALIDADE CHECK

**Nenhuma destas ferramentas cria apps de produção enterprise-ready sozinha.**

Todas elas são **excelentes para:**
- ✅ Protótipos e MVPs
- ✅ Validar ideias rapidamente
- ✅ Demos para investidores/clientes
- ✅ Aprender e experimentar

Mas **não substituem** (ainda):
- ❌ Arquitetura escalável
- ❌ Segurança robusta
- ❌ Performance otimizada
- ❌ Testes automatizados
- ❌ Manutenção long-term
- ❌ Features complexas de negócio

**Usa-as como acelerador**, não como solução completa. Para produtos sérios, eventualmente precisas de developers reais a refinar e escalar.

---

## 🔌 MCP: Model Context Protocol - Ligar a IA ao Mundo Real

### O Que É e Porquê Importa?

**MCP (Model Context Protocol)** é o standard universal que permite à IA aceder aos teus dados e ferramentas. Pensa nele como o **"USB-C para IA"** - um conector que funciona em todo o lado.

**Criado por:** Anthropic (Nov 2024) | **Status:** Open-source, adotado por Google, OpenAI, Microsoft

**A Analogia:**
```
Antes do USB-C: Cada dispositivo = 1 cabo diferente
Depois do USB-C: 1 cabo para tudo
MCP: 1 protocolo para qualquer IA → qualquer ferramenta
```

---

### O Problema Real Que MCP Resolve

**Cenário:** Queres que a IA aceda Gmail, Drive, GitHub, Postgres e Slack.

**Sem MCP (O Pesadelo):**
```
Claude + Gmail = Código custom #1
Claude + Drive = Código custom #2
ChatGPT + Gmail = Código custom #3 (diferente!)
ChatGPT + Drive = Código custom #4
...

10 IAs × 100 ferramentas = 1000 integrações custom! 
```
Cada atualização quebra tudo. Insustentável.

**Com MCP (A Solução):**
```
Gmail Server (MCP) ←→ Qualquer IA
Drive Server (MCP) ←→ Qualquer IA
...

10 IAs × 100 ferramentas = 110 componentes (10 + 100)
Redução de 90% no trabalho!
```

---

### Como Funciona (Simples)

```
┌──────────────┐         ┌──────────────┐         ┌─────────────┐
│   IA App     │◄───────►│ MCP Server   │◄───────►│ Teus Dados  │
│ (Cursor/     │         │ (Connector)  │         │ (Gmail/DB/  │
│  Claude)     │         │              │         │  GitHub)    │
└──────────────┘         └──────────────┘         └─────────────┘
```

**Fluxo:**
1. Tu (no Cursor): "Lista os meus últimos commits no GitHub"
2. Cursor → MCP Client pede ao GitHub MCP Server
3. GitHub Server → consulta GitHub API
4. Resposta volta pela cadeia
5. Cursor mostra-te os commits

---

### MCP Servers Prontos a Usar

**Produtividade:**
- Google Drive, Gmail, Slack, Notion, Linear

**Desenvolvimento:**  
- GitHub, Git, Postgres, SQLite, Puppeteer

**Outros:**
- Stripe, Filesystem, Browser, Memory

**16,000+ servers** disponíveis: https://github.com/modelcontextprotocol/servers

---

### Usar MCP em IDEs (Super Fácil)

#### **Cursor/Windsurf (1 clique):**

Settings → MCP → Add Server → Escolhe GitHub/Drive/Slack → Done!

#### **Claude Desktop (Manual):**

1. Edita ficheiro config:
```bash
# Mac: ~/Library/Application Support/Claude/claude_desktop_config.json
# Windows: %APPDATA%/Claude/claude_desktop_config.json
```

2. Adiciona servers:
```json
{
  "mcpServers": {
    "github": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-github"],
      "env": { "GITHUB_TOKEN": "ghp_..." }
    }
  }
}
```

3. Restart Claude → Pronto!

---

### Exemplo Prático Real

**Tarefa:** IA lê emails e cria GitHub Issues automaticamente

**Setup (5 min):**
```json
{
  "mcpServers": {
    "gmail": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-gmail"] },
    "github": { "command": "npx", "args": ["-y", "@modelcontextprotocol/server-github"] }
  }
}
```

**Prompt:**
```
"Lê últimos 10 emails não lidos. 
Para cada email sobre bugs, cria GitHub Issue em repo/XYZ com:
- Título = assunto do email
- Corpo = conteúdo do email  
- Label 'bug'"
```

**IA faz tudo sozinha!** Sem código, sem scripts complexos.

---

### IDEs com MCP Nativo (2025)

| IDE | Setup MCP | Facilidade |
|-----|-----------|------------|
| **Cursor** | 1 clique | ⭐⭐⭐⭐⭐ |
| **Windsurf** | 1 clique | ⭐⭐⭐⭐⭐ |
| **Zed** | Config simples | ⭐⭐⭐⭐ |
| **VS Code** | Via extensão | ⭐⭐⭐ |

---

### Casos de Uso Poderosos

**1. Workflow Automation**
```
"Quando PR recebe 'LGTM', faz merge e posta no Slack #eng"
```

**2. Análise de Dados**
```
"Consulta Postgres vendas últimos 3 meses, cria gráficos, envia PDF ao CEO"
```

**3. Gestão de Conteúdo**
```
"Lê artigos no Notion sobre 'IA', faz resumo, guarda em Google Doc"
```

---

### Segurança (Importante!)

**✅ Features de Segurança:**
- OAuth 2.1 (autenticação standard)
- Permissions granulares
- Human-in-the-loop (pede confirmação)
- Filesystem limits (só pastas autorizadas)

**⚠️ Alertas (2025):**
- 2000+ servers públicos sem autenticação
- Over-permissioning é comum
- **Usa só servers de fontes confiáveis**
- Review permissions sempre

---

### Adoção Lightning-Fast

**Timeline:**
- **Nov 2024:** Anthropic lança MCP
- **Mar 2025:** OpenAI adota (ChatGPT + Agents)
- **Abr 2025:** Google anuncia suporte (Gemini)
- **Nov 2025:** 16,000+ servers, standard da indústria

**Por isso importa:** Em <1 ano tornou-se o standard universal. Google, OpenAI, Microsoft - todos adotaram.

---

### Criar Teu Próprio MCP Server

**Exemplo: Weather API**

```typescript
import { Server } from "@modelcontextprotocol/sdk/server/index.js";

const server = new Server({ name: "weather-server", version: "1.0.0" });

// Define tool
server.setRequestHandler("tools/list", async () => ({
  tools: [{
    name: "get_weather",
    description: "Obter tempo para cidade",
    inputSchema: {
      type: "object",
      properties: { city: { type: "string" } },
      required: ["city"]
    }
  }]
}));

// Implementa
server.setRequestHandler("tools/call", async (request) => {
  const city = request.params.arguments.city;
  const response = await fetch(`https://api.weather.com?q=${city}`);
  return { content: [{ type: "text", text: JSON.stringify(await response.json()) }] };
});
```

**SDKs:** Python, TypeScript, Java, C#

---

### MCP vs Alternativas

| | MCP | OpenAI Functions | ChatGPT Plugins |
|---|-----|------------------|-----------------|
| **Standard aberto** | ✅ | ❌ | ❌ |
| **Multi-modelo** | ✅ | ❌ | ❌ |
| **Servers disponíveis** | 16K+ | Limitado | Descontinuado |
| **Self-host** | ✅ | ❌ | ❌ |

**Vencedor:** MCP (por isso até OpenAI adotou!)

---

### 📚 Recursos

**Aprender:**
- 📖 Docs oficiais: https://modelcontextprotocol.io
- 💻 GitHub: https://github.com/modelcontextprotocol
- 🎥 Workshops: "Building Agents with MCP" (Mahesh Murag)

**Comunidade:**
- Discord: MCP Community
- X: #ModelContextProtocol

---

### 🎯 TL;DR - MCP em 3 Pontos

1. **O quê:** Standard para IA aceder a dados/ferramentas (USB-C da IA)
2. **Porquê:** Reduz 1000 integrações para 110 (90% menos trabalho)
3. **Como usar:** Cursor/Windsurf têm setup de 1 clique, ou Claude Desktop com config JSON

**Bottom line:** MCP é o futuro. IDEs modernos já o têm. Se usas IA para desenvolvimento, **já podes beneficiar de MCP hoje**.

---

Linha de comando e automação avançada.

#### **Claude Code (Anthropic)**

**O que é:**
- Ferramenta CLI para delegar tarefas de coding direto do terminal
- Research preview tornado GA em 2025

**Pontos Fortes:**
✅ Trabalha autonomamente em tarefas durante horas  
✅ Checkpoints: guarda progresso e rollback  
✅ Extensão nativa VS Code  
✅ Features de memória e contexto editing  

**Limitações:**
❌ Ainda relativamente novo  
❌ Requer subscrição Anthropic  

**Melhor para:**
- Tarefas complexas de várias horas
- Refactoring de projetos grandes
- Automação de workflows

---

### 📊 QUADRO COMPARATIVO COMPLETO (2025)

| Ferramenta | Tipo | Melhor Para | Preço Base | Modelos | Destaca-se Em |
|------------|------|-------------|------------|---------|---------------|
| **ChatGPT** | Chat | Generalista, learning | Grátis/€20 | GPT-4o, o3 | Explicações, teaching |
| **Claude** | Chat | Código complexo | Grátis/€20 | Opus 4, Sonnet 4.5 | Coding (82% SWE-bench) |
| **Perplexity** | Chat+Search | Pesquisa atual | Grátis/€20 | Múltiplos | Respostas com citações web |
| **Cursor** | IDE | Projetos complexos | Grátis/€20 | Múltiplos | Edição multi-ficheiro |
| **Windsurf** | IDE | Value, beginners | Grátis/€15 | Múltiplos | Contexto automático |
| **Copilot** | Extensão | GitHub integration | €10-39 | GPT-4.5, Claude | Integração GitHub |
| **v0.dev** | Web | UI rápido | Freemium | Próprio | Components React |
| **Bolt.new** | Web | Full-stack rápido | Freemium | Próprio | Zero setup |

---

### 🎯 RECOMENDAÇÕES POR PERFIL

#### **Para Iniciantes Absolutos:**
1. **ChatGPT Free** - aprender conceitos
2. **Bolt.new** - testar código sem instalar nada
3. **Claude Free** - quando precisas de código melhor

**Custo:** €0/mês

---

#### **Para Estudantes/Hobbyistas:**
1. **Claude Pro** (€20) - melhor coding
2. **VS Code** + **GitHub Copilot Free** (50 pedidos/mês)
3. **Perplexity** - pesquisa e learning

**Custo:** €20/mês

---

#### **Para Developers Profissionais:**
1. **Cursor Pro** (€20) OU **Windsurf Pro** (€15) - IDE principal
2. **Claude Pro** (€20) - análise e planeamento
3. **Perplexity Pro** (€20) - research

**Custo:** €40-60/mês

---

#### **Para Equipas/Empresas:**
1. **GitHub Copilot Business** (€19/user) - toda a equipa
2. **Cursor Business** (€40/user) - developers avançados
3. **Claude Enterprise** - projetos críticos

**Custo:** Variável por utilizador

---

### 💡 DICAS DE ESCOLHA

**Escolhe ChatGPT se:**
- És iniciante completo
- Queres a experiência mais mainstream
- Precisas de muitos plugins e integrações

**Escolhe Claude se:**
- Vais trabalhar com código complexo e extenso
- Precisas de contexto gigante (projetos grandes)
- Queres a melhor qualidade de código possível

**Escolhe Perplexity se:**
- Precisas de informação atualizada constantemente
- Trabalhas com pesquisa e análise
- Queres respostas com fontes verificáveis

**Escolhe Cursor se:**
- És developer e queres o melhor IDE com IA
- Trabalhas em projetos multi-ficheiro complexos
- Orçamento permite €20/mês

**Escolhe Windsurf se:**
- Queres performance de Cursor com menos custo
- És beginner e queres UI intuitivo
- Trabalhas com grandes codebases

**Escolhe GitHub Copilot se:**
- Já usas GitHub extensivamente
- Não queres mudar de editor
- Trabalhas em empresa que exige compliance

---

### ⚠️ AVISOS IMPORTANTES

**Não precisas de todas!** Muitos iniciantes cometem o erro de subscrever tudo. Começa com:
1. Uma ferramenta de chat (ChatGPT OU Claude)
2. Uma ferramenta de coding (Cursor OU Windsurf OU Copilot)
3. Experimenta grátis primeiro!

**Custos podem acumular:** €20 + €20 + €20 = €60/mês rapidamente. Define orçamento primeiro.

**A ferramenta não te faz programmer:** Estas ferramentas aceleram, mas ainda precisas de:
- Saber o que queres construir
- Entender lógica básica
- Debugging e troubleshooting
- Pensamento crítico

---

---

## 1.3 Mindset para Trabalhar com IA
**Duração:** 5 minutos

### Pensar em "Conversação" vs "Programar"

**Mindset Tradicional:**
```
"Preciso de aprender sintaxe → Escrever código → Testar → Corrigir erros"
```

**Mindset com IA:**
```
"Preciso descrever o que quero → IA gera → Eu testo → Refino a descrição"
```

### Princípios Fundamentais

#### 1. Clareza > Perfeição
- Não precisas do prompt perfeito à primeira
- Começa simples, refina depois
- A IA entende contexto, não precisa de ser ultra-específico

**Exemplo:**
```
❌ Evita: "Preciso de um algoritmo complexo de machine learning..."
✅ Prefere: "Quero um script que analise este ficheiro Excel e me diga quais produtos vendem mais"
```

#### 2. Iteração é o Caminho
- Raramente acertas à primeira
- Processo normal: gera → avalia → ajusta → repete
- Cada iteração aproxima-te do objetivo

**Ciclo típico:**
1. Primeira versão: 60-70% do que queres
2. Segunda iteração: 85-90%
3. Terceira iteração: 95-100%

#### 3. A Importância de Saber o Que Queres

**IA não substitui:**
- ❌ Clareza de objetivos
- ❌ Conhecimento do problema
- ❌ Tomada de decisões

**IA acelera:**
- ✅ Implementação
- ✅ Prototipagem
- ✅ Aprendizagem de novas tecnologias

### Limitações e Expectativas Realistas

#### O que a IA Faz Bem
✅ Código boilerplate (estrutura repetitiva)  
✅ Transformações de dados comuns  
✅ Interfaces básicas e médias  
✅ Scripts de automação  
✅ Integração de APIs conhecidas  
✅ Explicar código existente  

#### Onde a IA Ainda Falha
❌ Arquitetura de sistemas complexos  
❌ Otimização extrema de performance  
❌ Contexto de negócio específico  
❌ Decisões estratégicas  
❌ Debugging de problemas obscuros  
❌ Código para casos muito específicos/raros  

### Regras de Ouro

**1. Valida Sempre**
- Nunca assumes que o código gerado está 100% correto
- Testa antes de usar em produção
- Compreende o básico do que foi gerado

**2. Contexto é Rei**
- Quanto mais contexto deres, melhor o resultado
- Explica o caso de uso, não só o código
- Menciona limitações ou requisitos específicos

**3. Experimenta e Aprende**
- Não tenhas medo de tentar
- O pior que acontece é não funcionar (e tentas outra vez)
- Cada tentativa ensina-te algo

**4. Documenta o Teu Processo**
- Guarda os prompts que funcionaram bem
- Anota soluções para problemas comuns
- Cria a tua própria biblioteca de templates

### Mudança de Paradigma

**Antes da IA:**
"Não posso fazer isto porque não sei programar"

**Depois da IA:**
"Não sei como se faz, mas sei o que quero. Vou tentar com IA"

**Resultado:**
- Mais pessoas conseguem criar soluções tecnológicas
- Menos tempo entre ideia e protótipo
- Democratização do desenvolvimento

---

## 📝 CHECKPOINT DO MÓDULO 1

### O que aprendeste:
✅ O que é desenvolvimento com IA e quando usar  
✅ As principais ferramentas disponíveis  
✅ Como pensar diferente ao trabalhar com IA  
✅ Limitações e expectativas realistas  

### Próximo passo:
➡️ **MÓDULO 2:** Vais criar o teu primeiro projeto completo com IA!

---

## 💡 EXERCÍCIO PREPARATÓRIO

Antes de avançares para o Módulo 2, faz isto:

1. **Cria uma conta gratuita** em ChatGPT ou Claude
2. **Instala o VS Code** (https://code.visualstudio.com)
3. **Pensa num projeto simples** que gostarias de criar (pode ser automatizar algo no trabalho, uma página web, um script útil)

---

## 🔗 RECURSOS ADICIONAIS

### Links Úteis:
- [ChatGPT](https://chat.openai.com)
- [Claude](https://claude.ai)
- [VS Code Download](https://code.visualstudio.com)
- [Cursor IDE](https://cursor.sh)
- [v0.dev](https://v0.dev)
- [Bolt.new](https://bolt.new)

### Leitura Recomendada:
- Documentação oficial das ferramentas
- Exemplos de prompts eficazes
- Casos de estudo de automação com IA

---

*Continua para: [MÓDULO 2 - Primeiros Passos Práticos](02-MODULO-2-PRIMEIROS-PASSOS.md)*
