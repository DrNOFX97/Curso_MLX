# MÓDULO 4: Workflow Profissional

⏱️ **Duração:** 20 minutos  
📊 **Formato:** Best practices e organização  
🎯 **Objetivo:** Trabalhar como um profissional, mesmo sendo iniciante

---

## 4.1 Organizar Projetos como um Profissional
**Duração:** 8 minutos

### 📁 Estrutura de Pastas

#### Para Projetos Web

```
meu-projeto/
│
├── index.html          # Página principal
├── sobre.html          # Outras páginas
│
├── css/
│   ├── style.css       # Estilos principais
│   └── responsive.css  # Estilos mobile
│
├── js/
│   ├── main.js         # JavaScript principal
│   └── utils.js        # Funções auxiliares
│
├── img/
│   ├── logo.png
│   └── hero-bg.jpg
│
├── assets/
│   ├── fonts/
│   └── icons/
│
├── README.md           # Documentação do projeto
├── .gitignore          # Ficheiros a ignorar no Git
└── LICENSE             # Licença (se aplicável)
```

---

#### Para Scripts Python

```
automacao-vendas/
│
├── main.py                  # Script principal
├── config.py                # Configurações
├── requirements.txt         # Dependências
│
├── src/
│   ├── __init__.py
│   ├── processar.py        # Módulo de processamento
│   └── emails.py           # Módulo de emails
│
├── data/
│   ├── input/              # Dados de entrada
│   └── output/             # Resultados gerados
│
├── tests/
│   └── test_processar.py   # Testes
│
├── logs/
│   └── app.log             # Logs de execução
│
├── README.md
├── .env.example            # Exemplo de variáveis de ambiente
└── .gitignore
```

---

### 📝 README.md Profissional

**Template que podes usar sempre:**

```markdown
# Nome do Projeto

Breve descrição do que o projeto faz (1-2 linhas).

## 🎯 Objetivo

Explicação mais detalhada do problema que resolve.

## ✨ Funcionalidades

- Feature 1
- Feature 2
- Feature 3

## 🚀 Como Usar

### Pré-requisitos
- Python 3.8+ (ou outra tecnologia)
- Bibliotecas necessárias

### Instalação

```bash
# Clonar repositório
git clone https://github.com/teu-user/projeto.git

# Entrar na pasta
cd projeto

# Instalar dependências
pip install -r requirements.txt
```

### Configuração

1. Copiar `.env.example` para `.env`
2. Preencher variáveis de ambiente:
   ```
   API_KEY=tua_chave_aqui
   EMAIL=teu@email.com
   ```

### Executar

```bash
python main.py
```

## 📸 Screenshots

[Inserir imagens do projeto funcionando]

## 🛠️ Tecnologias Usadas

- Python 3.9
- Pandas
- Flask
- etc.

## 📄 Licença

MIT License - Vê ficheiro LICENSE para detalhes

## 👤 Autor

Teu Nome - [LinkedIn](link) - [Email](mailto:)

## 🙏 Agradecimentos

- Inspiração ou recursos usados
```

---

### 🔧 Versionamento com Git (Básico com IA)

**Não sabes Git? Não tem problema. A IA ajuda-te!**

#### Setup Inicial

**Prompt:**
```
Quero começar a usar Git no meu projeto para controlo de versões.
Nunca usei Git antes.

Projeto: [descreve brevemente]
Sistema: [Windows/Mac/Linux]

Dá-me instruções passo a passo:
1. Como instalar Git
2. Como inicializar repositório
3. Comandos básicos que vou usar
4. Como ligar ao GitHub

Explica cada comando de forma simples.
```

---

#### Workflow Básico (com ajuda de IA)

**Quando quiseres guardar alterações:**

```
Fiz alterações no meu projeto e quero guardá-las no Git.

Alterações:
- Adicionei nova funcionalidade de [...]
- Corrigi bug em [...]

Que comandos Git devo usar? Explica o que cada um faz.
```

**A IA vai dar-te algo como:**

```bash
# 1. Ver o que mudou
git status

# 2. Adicionar ficheiros alterados
git add .

# 3. Guardar com mensagem
git commit -m "Adiciona funcionalidade X e corrige bug Y"

# 4. Enviar para GitHub (se configurado)
git push
```

---

### 💾 Backup e Colaboração

#### Estratégias de Backup

**1. GitHub (Recomendado)**
- Código seguro na cloud
- Histórico completo de alterações
- Gratuito para projetos públicos e privados

**2. Google Drive / Dropbox**
- Boa para backups rápidos
- Não tem controlo de versões
- Útil para ficheiros grandes (dados, imagens)

**3. Regra 3-2-1**
- 3 cópias do projeto
- 2 tipos de armazenamento diferentes
- 1 cópia offsite (cloud)

---

#### Colaborar com Outros (ou com IA)

**Cenário:** Precisas de partilhar o projeto com alguém para ajudar

**Prompt:**
```
Quero partilhar o meu projeto no GitHub para alguém poder:
- Ver o código
- Fazer sugestões
- Contribuir com melhorias

Como faço:
1. Upload do projeto para GitHub
2. Dar acesso a outra pessoa
3. Receber e integrar sugestões dessa pessoa

Explica passo a passo para iniciantes.
```

---

### 📦 Gestão de Dependências

#### Python - requirements.txt

**Criar automaticamente:**

```bash
# Gera lista de todas as bibliotecas instaladas
pip freeze > requirements.txt
```

**Conteúdo típico:**
```
pandas==2.0.3
openpyxl==3.1.2
matplotlib==3.7.1
requests==2.31.0
```

**Instalar num novo ambiente:**
```bash
pip install -r requirements.txt
```

---

#### JavaScript - package.json

**Se usares Node.js:**

```bash
npm init  # Cria package.json
npm install biblioteca-x  # Adiciona dependência
```

---

## 4.2 Boas Práticas e Armadilhas Comuns
**Duração:** 8 minutos

### ✅ Checklist de Validação de Código

**Antes de usar código gerado pela IA, verifica:**

#### 🔐 SEGURANÇA

❌ **Nunca faças isto:**
```python
# API key hardcoded no código
API_KEY = "sk_live_123456789abcdef"

# Password visível
email_password = "minhapassword123"

# SQL direto (sujeito a SQL injection)
query = f"SELECT * FROM users WHERE id = {user_input}"
```

✅ **Faz assim:**
```python
# Variáveis de ambiente
import os
API_KEY = os.getenv('API_KEY')

# .env file (nunca commitar!)
# API_KEY=sk_live_123456789abcdef

# Prepared statements
cursor.execute("SELECT * FROM users WHERE id = ?", (user_input,))
```

---

#### 🔒 EXPOSIÇÃO DE DADOS

**Ficheiros a NUNCA partilhar:**
- `.env` - Variáveis de ambiente
- `config.py` com passwords
- Ficheiros com dados pessoais
- API keys, tokens

**Criar `.gitignore`:**
```
# Ambiente
.env
venv/
__pycache__/

# Dados sensíveis
config.py
*.log
data/private/

# Ficheiros do sistema
.DS_Store
Thumbs.db
```

---

#### 🐛 TRATAMENTO DE ERROS

❌ **Código frágil:**
```python
# Assume que tudo vai correr bem
data = open('ficheiro.csv').read()
result = processar(data)
enviar_email(result)
```

✅ **Código robusto:**
```python
try:
    with open('ficheiro.csv', 'r') as f:
        data = f.read()
    
    result = processar(data)
    
    if result:
        enviar_email(result)
        print("✅ Email enviado com sucesso")
    else:
        print("⚠️ Sem dados para enviar")
        
except FileNotFoundError:
    print("❌ Erro: Ficheiro não encontrado")
except Exception as e:
    print(f"❌ Erro inesperado: {e}")
    # Log do erro para debug
    with open('errors.log', 'a') as log:
        log.write(f"{datetime.now()}: {e}\n")
```

---

### ⚡ PERFORMANCE

**Código Lento (evitar):**
```python
# Loop dentro de loop - O(n²)
for cliente in clientes:  # 1000 clientes
    for venda in vendas:  # 10000 vendas
        if venda.cliente_id == cliente.id:
            # processa...
```

**Código Rápido (preferir):**
```python
# Usar dicionário - O(n)
vendas_por_cliente = {}
for venda in vendas:
    if venda.cliente_id not in vendas_por_cliente:
        vendas_por_cliente[venda.cliente_id] = []
    vendas_por_cliente[venda.cliente_id].append(venda)

for cliente in clientes:
    vendas_cliente = vendas_por_cliente.get(cliente.id, [])
    # processa...
```

**Ou deixa o pandas fazer o trabalho:**
```python
# Pandas otimizado
df_resultado = df_vendas.merge(df_clientes, on='cliente_id')
```

---

### 🔟 10 Erros Típicos de Principiantes (e Como Evitar)

#### 1. **Não testar com dados reais**

❌ Problema: Código funciona com 5 linhas, mas quebra com 5000

✅ Solução: Testa sempre com volume real de dados

---

#### 2. **Assumir formato de dados**

❌ Problema: Código espera data em DD/MM/YYYY, recebe DD-MM-YYYY

✅ Solução: Valida e normaliza inputs

```python
# Prompt para IA:
"Adiciona validação que aceite datas em múltiplos formatos:
DD/MM/YYYY, DD-MM-YYYY, YYYY-MM-DD
e converta tudo para formato padrão"
```

---

#### 3. **Ignorar mensagens de erro**

❌ Problema: "Não funciona" mas não lês o erro

✅ Solução: Lê a mensagem de erro completa e pesquisa/pergunta à IA

---

#### 4. **Não fazer backup antes de mudanças grandes**

❌ Problema: Mudas 50% do código, quebra tudo, não consegues voltar atrás

✅ Solução: Git commit antes de mudanças, ou copia pasta inteira

---

#### 5. **Hardcoding (valores fixos no código)**

❌ Problema:
```python
if mes == "Novembro":  # E nos outros meses?
if ficheiro == "vendas.xlsx":  # E se mudar o nome?
```

✅ Solução:
```python
# Usar variáveis
MES_ATUAL = datetime.now().strftime("%B")
ficheiros = glob.glob("*.xlsx")  # Pega todos os .xlsx
```

---

#### 6. **Não comentar código complexo**

❌ Problema: Voltas ao código 1 mês depois e não sabes o que faz

✅ Solução:
```python
# Calcula comissão escalonada:
# 0-1000€: 10%, 1000-5000€: 15%, >5000€: 20%
def calcular_comissao(valor):
    if valor < 1000:
        return valor * 0.10
    # ...
```

---

#### 7. **Expor credenciais**

❌ Problema: Fazer commit de API keys, passwords

✅ Solução: `.env` + `.gitignore`

---

#### 8. **Não validar inputs do utilizador**

❌ Problema:
```python
idade = int(input("Idade: "))  # User escreve "vinte" → CRASH
```

✅ Solução:
```python
try:
    idade = int(input("Idade: "))
    if idade < 0 or idade > 120:
        print("Idade inválida")
except ValueError:
    print("Por favor insere um número")
```

---

#### 9. **Reinventar a roda**

❌ Problema: Gastar 2 horas a criar função que já existe em biblioteca

✅ Solução: Pergunta à IA se já existe solução pronta

```
"Preciso de [funcionalidade X]. Existe alguma biblioteca Python 
que já faça isto?"
```

---

#### 10. **Esquecer de testar em mobile/outros browsers**

❌ Problema: Web app funciona no teu Chrome mas quebra no Safari/mobile

✅ Solução: Testa em múltiplos ambientes

---

### 📋 Checklist Antes de "Produção"

**Uso pessoal:**
- [ ] Código funciona com dados reais
- [ ] Tratamento básico de erros
- [ ] Comentários em partes complexas
- [ ] Backup feito

**Partilhar com outros:**
- [ ] Tudo acima +
- [ ] README com instruções
- [ ] Sem credenciais expostas
- [ ] Testado em ambiente "limpo"
- [ ] requirements.txt ou equivalente

**Uso profissional/cliente:**
- [ ] Tudo acima +
- [ ] Testes automatizados
- [ ] Logs de erro
- [ ] Validação completa de inputs
- [ ] Performance testada com volume real
- [ ] Documentação completa

---

## 4.3 Próximos Passos na Tua Jornada
**Duração:** 4 minutos

### 🎓 Como Continuar a Aprender

#### Prática Deliberada

**Semana 1-2: Consolidar Fundamentos**
- Recria os projetos do curso sem olhar
- Adiciona 1 funcionalidade nova a cada projeto
- Tempo: 1h/dia

**Semana 3-4: Projetos Pessoais**
- Identifica problema real que tens
- Cria solução com IA
- Usa diariamente e itera

**Mês 2: Aumentar Complexidade**
- Integra múltiplas tecnologias
- Adiciona autenticação
- Trabalha com bases de dados

---

### 📚 Recursos Recomendados

#### Gratuitos

**Documentação:**
- MDN Web Docs (HTML/CSS/JS)
- Python.org Documentation
- W3Schools (tutoriais interativos)

**Comunidades:**
- r/learnprogramming (Reddit)
- Stack Overflow (perguntas técnicas)
- Discord communities (Python, WebDev)

**YouTube (PT/BR):**
- Curso em Vídeo (Gustavo Guanabara)
- Programador BR
- Dev Aprender

**Prática:**
- FreeCodeCamp (projetos guiados)
- Exercism.io (exercícios com feedback)
- HackerRank (desafios)

---

#### Pagos (se quiseres investir)

**Cursos:**
- Udemy (promoções frequentes ~10-15€)
- Pluralsight
- Frontend Masters

**Livros:**
- "Automate the Boring Stuff with Python"
- "Eloquent JavaScript"
- "You Don't Know JS"

---

### 💡 Ideias de Projetos Para Praticar

**Nível Iniciante (próximas 2 semanas):**

1. **Portfolio Pessoal**
   - Landing page com teus projetos
   - Formulário de contacto funcional
   - Deploy online

2. **Conversor de Unidades**
   - Temperatura, distância, peso, moeda
   - Interface simples
   - Histórico de conversões

3. **Lista de Compras Inteligente**
   - Adicionar/remover items
   - Marcar como comprado
   - Estimar custo total
   - Partilhar lista (export)

---

**Nível Intermédio (mês 2-3):**

4. **Bot de Telegram/Discord**
   - Responde a comandos
   - Integra com API (tempo, notícias)
   - Notificações automáticas

5. **Dashboard de Produtividade**
   - Timer Pomodoro
   - Tracker de hábitos
   - Estatísticas semanais
   - Integração com Google Calendar

6. **Web Scraper Útil**
   - Monitora preços de produtos
   - Extrai vagas de emprego
   - Alerta quando encontra match

---

**Nível Avançado (mês 4+):**

7. **SaaS Simples**
   - Ferramenta útil para nicho específico
   - Autenticação (login/register)
   - Planos free/paid (conceito)
   - Landing + App

8. **App Mobile (com IA)**
   - Usa Flutter ou React Native com IA
   - CRUD completo
   - Sincronização cloud

9. **Automatização Empresarial**
   - Sistema completo para pequena empresa
   - Múltiplos módulos integrados
   - Relatórios avançados

---

### 🚀 Construir Portfólio

**3 Projetos Essenciais:**

1. **Projeto Showcase (Impressionar)**
   - Visualmente atraente
   - Bem documentado
   - Casos de uso claros

2. **Projeto Técnico (Demonstrar Skills)**
   - Código bem estruturado
   - Testes incluídos
   - README detalhado

3. **Projeto Real (Provar Valor)**
   - Resolve problema real
   - Métricas de impacto
   - Testemunhos se possível

**GitHub Profile:**
- README.md atrativo (usa templates)
- Pins dos melhores projetos
- Contribuições consistentes (green squares)

---

### 💼 Oportunidades Profissionais

**Com estas skills podes:**

**Freelancing:**
- Criar websites para pequenos negócios
- Automatizar processos empresariais
- Desenvolver MVPs para startups
- Scraping e análise de dados

**Emprego:**
- Junior Developer (com portfólio forte)
- Business Analyst (com skills técnicas)
- Product Manager (tech-savvy)
- Automation Specialist

**Empreendedorismo:**
- Criar e vender templates
- SaaS simples (micro-saas)
- Cursos/tutoriais
- Consultoria em automação

---

### 🎯 Meta de 6 Meses

**Objetivo realista:**

Conseguires:
- ✅ Criar qualquer web app simples-média em < 2h
- ✅ Automatizar 90% das tuas tarefas repetitivas
- ✅ Ler e entender código de outros
- ✅ Contribuir em projetos open-source
- ✅ Ter 5-10 projetos no GitHub
- ✅ Sentir confiança em tentar qualquer ideia

**Não vais:**
- ❌ Ser senior developer (leva anos)
- ❌ Saber tudo sobre programação
- ❌ Nunca mais precisar de ajuda

**E está tudo bem!** Ninguém sabe tudo. O importante é:
1. Saberes fazer o suficiente para criar valor
2. Saberes procurar ajuda quando precisas
3. Continuares a aprender

---

### 📧 Manter-te Atualizado

**Newsletters (gratuitas):**
- JavaScript Weekly
- Python Weekly
- Frontend Focus
- TLDR (tech news diária)

**Podcasts:**
- Syntax.fm
- CodeNewbie
- PodProgramar (PT)

**Seguir no Twitter/X:**
- @dan_abramov (React)
- @kentcdodds (JavaScript)
- @unclebobmartin (Clean Code)
- Developers locais PT

---

## 📝 CHECKPOINT DO MÓDULO 4

### O que aprendeste:
✅ Organizar projetos profissionalmente  
✅ Usar Git para controlo de versões  
✅ Boas práticas de segurança  
✅ Evitar armadilhas comuns  
✅ Caminho de aprendizagem contínua  

### Próximo nível:
➡️ **MÓDULO 5:** Vais criar um projeto final completo do zero!

---

## 💪 DESAFIO PRÉ-MÓDULO 5

Antes de avançares:

1. **Organiza** um dos teus projetos anteriores com a estrutura profissional
2. **Cria README.md** completo
3. **Faz upload** para GitHub
4. **Partilha** o link (LinkedIn, amigos, etc.)

Isto prepara-te para criar e apresentar o projeto final com confiança.

---

*Continua para: [MÓDULO 5 - Projeto Final Guiado](05-MODULO-5-PROJETO-FINAL.md)*
