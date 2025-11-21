# MÓDULO 3: Casos de Uso Profissionais

⏱️ **Duração:** 60 minutos  
📊 **Formato:** 100% Hands-on com projetos reais  
🎯 **Objetivo:** Aplicar IA em cenários profissionais e criar ferramentas úteis

---

## 3.1 Automação de Tarefas do Dia-a-Dia
**Duração:** 20 minutos

### 🎯 DEMO 1: Processador de Relatórios Excel (10 min)

#### CENÁRIO REAL
**Situação:** Trabalhas em vendas e recebes 20 ficheiros Excel mensalmente (um por vendedor) que precisas de consolidar num único relatório.

**Problema:** 
- Demora 2-3 horas a fazer manualmente
- Sujeito a erros (copiar/colar)
- Chato e repetitivo

**Solução:** Script Python automático

---

#### PASSO 1: Preparar o Contexto

**Estrutura dos ficheiros que recebes:**

```
vendas_joao_novembro.xlsx
vendas_maria_novembro.xlsx
vendas_carlos_novembro.xlsx
...

Cada ficheiro tem:
| Data | Cliente | Produto | Quantidade | Valor | Comissão |
```

**O que precisas:**
- Consolidar todos numa única folha
- Calcular totais por vendedor
- Identificar top 3 produtos mais vendidos
- Criar gráfico de vendas por vendedor
- Exportar para Excel formatado

---

#### PASSO 2: Prompt Completo

```
CONTEXTO:
Sou gestor de vendas. Recebo mensalmente 20 ficheiros Excel (um por vendedor) 
com dados de vendas que preciso de consolidar num único relatório.

OBJETIVO:
Criar um script Python que automatize este processo completamente, 
poupando 2-3h de trabalho manual.

ESTRUTURA DOS FICHEIROS INPUT:
- Localização: pasta "vendas_mes/"
- Nomes: vendas_[nome]_[mes].xlsx
- Colunas: Data, Cliente, Produto, Quantidade, Valor, Comissão
- Exemplo de dados:
  | Data | Cliente | Produto | Quantidade | Valor | Comissão |
  | 2024-11-01 | Cliente A | Produto X | 5 | 1000 | 100 |
  | 2024-11-02 | Cliente B | Produto Y | 3 | 600 | 60 |

REQUISITOS FUNCIONAIS:
1. Ler todos os ficheiros .xlsx de uma pasta específica
2. Consolidar dados numa única tabela
3. Adicionar coluna "Vendedor" (extraída do nome do ficheiro)
4. Calcular:
   - Total de vendas por vendedor
   - Total de comissões por vendedor
   - Top 3 produtos mais vendidos (quantidade)
   - Média de venda por transação
5. Criar 2 gráficos:
   - Gráfico de barras: Vendas por vendedor
   - Gráfico de pizza: Top 3 produtos
6. Exportar para Excel com 3 folhas:
   - "Dados Consolidados" - Todos os dados
   - "Resumo" - Tabela de totais por vendedor
   - "Análise" - Estatísticas e gráficos

REQUISITOS TÉCNICOS:
- Python 3.8+
- Usar pandas para manipulação
- Usar openpyxl para Excel
- Usar matplotlib/seaborn para gráficos
- Tratamento de erros:
  * Ficheiros corrompidos
  * Colunas em falta
  * Dados inválidos
- Logs no terminal do progresso
- Comentários em português explicando cada passo

FORMATAÇÃO DO OUTPUT:
- Headers em bold, fundo azul escuro, texto branco
- Valores numéricos: formato € (ex: 1.234,56 €)
- Datas: formato DD/MM/YYYY
- Auto-ajustar largura das colunas
- Adicionar filtros automáticos na primeira linha

OUTPUT ESPERADO:
Ficheiro: relatorio_vendas_novembro_2024.xlsx

Gera o código Python completo.
```

---

#### PASSO 3: Executar e Testar

**Setup (se ainda não tens Python):**
```bash
# Instalar Python: python.org

# Instalar bibliotecas necessárias
pip install pandas openpyxl matplotlib seaborn
```

**Testar o script:**
1. Cria pasta `vendas_mes/`
2. Coloca alguns ficheiros Excel de teste
3. Executa: `python processar_vendas.py`
4. Verifica o ficheiro gerado

**Troubleshooting comum:**
```
Erro: "ModuleNotFoundError: No module named 'pandas'"
Solução: pip install pandas

Erro: "FileNotFoundError"
Solução: Verifica que a pasta existe e tem ficheiros .xlsx
```

---

### 🎯 DEMO 2: Automatizar Envio de Emails Personalizados (10 min)

#### CENÁRIO REAL
**Situação:** Precisas de enviar emails personalizados a 50 clientes com propostas comerciais.

**Manual:**
- Abrir Excel com lista de clientes
- Copiar cada nome/email/valor
- Compor email personalizado
- Anexar PDF
- Enviar
- Repetir 50 vezes (2-3 horas)

**Automatizado:**
- Executa script
- 50 emails enviados em 2 minutos

---

#### Prompt para o Script

```
CONTEXTO:
Sou gestor comercial e preciso de enviar propostas comerciais personalizadas 
a 50 clientes. Cada cliente deve receber email com seu nome, produto de interesse 
e valor da proposta.

OBJETIVO:
Script Python que envia emails personalizados automaticamente a partir de ficheiro CSV.

ESTRUTURA DO CSV (clientes.csv):
Nome,Email,Produto,Valor,Data_Reuniao
João Silva,joao@empresa.pt,Software CRM,5000,2024-11-25
Maria Santos,maria@outra.pt,Dashboard Analytics,3000,2024-11-26

TEMPLATE DO EMAIL:
Assunto: Proposta Comercial - {Produto}

Olá {Nome},

Foi um prazer conversar consigo na nossa reunião de {Data_Reuniao}.

Conforme discutido, segue em anexo a proposta comercial para {Produto} 
no valor de {Valor}€.

A proposta inclui:
- Implementação completa
- Formação da equipa
- Suporte durante 12 meses

Aguardo o seu feedback.

Melhores cumprimentos,
[Meu Nome]
[Minha Empresa]

REQUISITOS:
FUNCIONAL:
- Ler CSV com dados dos clientes
- Gerar email personalizado para cada linha
- Anexar ficheiro PDF (mesmo ficheiro para todos): "proposta_comercial.pdf"
- Enviar via Gmail SMTP
- Rate limiting: 1 email por segundo (evitar spam/bloqueio)
- Criar log de emails enviados:
  * Timestamp
  * Destinatário
  * Status (sucesso/erro)
  * Motivo do erro (se aplicável)

TÉCNICO:
- Python 3.8+
- smtplib para envio
- email.mime para composição
- pandas para ler CSV
- Credenciais Gmail via variáveis de ambiente (não hardcoded)
- Tratamento de erros:
  * Email inválido
  * Falha de conexão
  * Falha no envio
- Progress bar (tqdm)

SEGURANÇA:
- Usar App Password do Gmail (não password normal)
- Verificar formato email antes de enviar
- Confirmar antes de enviar em massa (Y/N)

Adiciona instruções de como configurar o Gmail para permitir o script.
```

---

#### PASSO EXTRA: Configurar Gmail

**A IA vai dar-te isto:**

1. Ativar verificação em 2 passos no Gmail
2. Gerar App Password:
   - https://myaccount.google.com/apppasswords
   - Seleciona "Correio" e "Outro"
   - Copia a password gerada

3. Criar ficheiro `.env`:
```
GMAIL_USER=teu@email.com
GMAIL_PASSWORD=password_app_aqui
```

4. Instalar biblioteca:
```bash
pip install python-dotenv
```

---

### 💼 EXERCÍCIO PRÁTICO (Escolhe 1)

**A. Automatizar Relatório de Despesas**
```
Lês múltiplos PDFs de faturas, extrais:
- Data, fornecedor, valor, categoria
- Consolidas em Excel
- Identifica despesas acima de 100€
- Cria gráfico de despesas por categoria
```

**B. Backup Automático de Ficheiros**
```
Script que:
- Copia ficheiros de pasta específica
- Comprime em ZIP com data no nome
- Envia para Google Drive ou Dropbox
- Apaga backups com mais de 30 dias
- Executa automaticamente todas as noites
```

**C. Webscraping de Preços**
```
Monitora preços de produtos em websites:
- Lista de URLs de produtos
- Extrai preço atual
- Compara com preço anterior
- Envia notificação se descida > 10%
- Guarda histórico em Excel
```

---

## 3.2 Criar Aplicações Web Funcionais
**Duração:** 20 minutos

### 🎯 DEMO 3: Dashboard Interativo de Despesas (15 min)

#### O QUE VAMOS CONSTRUIR
App web completa que permite:
- Adicionar despesas (categoria, valor, data)
- Visualizar gráficos (pizza e linha)
- Filtrar por mês/categoria
- Exportar para Excel
- Dados guardados localmente (sem backend)

---

#### PROMPT COMPLETO

```
CONTEXTO:
Quero uma aplicação web para controlo de despesas pessoais que funcione 
completamente no browser (sem backend/servidor).

OBJETIVO:
Dashboard interativo onde posso adicionar despesas, ver gráficos e estatísticas, 
e exportar dados.

FUNCIONALIDADES:
1. ADICIONAR DESPESA:
   - Form com campos:
     * Categoria (dropdown: Alimentação, Transporte, Saúde, Lazer, Outros)
     * Valor (número, €)
     * Data (date picker)
     * Descrição (opcional, texto curto)
   - Botão "Adicionar"
   - Validação: categoria e valor obrigatórios

2. LISTAR DESPESAS:
   - Tabela com todas as despesas
   - Colunas: Data, Categoria, Descrição, Valor
   - Ações: Editar, Eliminar
   - Ordenar por data (mais recente primeiro)

3. VISUALIZAÇÕES:
   - Gráfico Pizza: Despesas por categoria (%)
   - Gráfico Linha: Evolução mensal de despesas
   - Cards com estatísticas:
     * Total do mês atual
     * Média por dia
     * Categoria que mais gastas

4. FILTROS:
   - Por mês (dropdown com últimos 6 meses)
   - Por categoria (multi-select)
   - Botão "Limpar filtros"

5. EXPORTAR:
   - Botão "Exportar Excel"
   - Gera ficheiro .xlsx com todas as despesas

DESIGN:
- Layout moderno e limpo
- Cores: Azul (#4A90E2), Branco, Cinza claro
- Responsivo (funciona bem em mobile)
- Cards com sombras suaves
- Ícones nos botões (Font Awesome)
- Animações suaves (transições)

TÉCNICO:
- HTML + CSS + JavaScript puro (sem frameworks)
- Chart.js para gráficos (CDN)
- SheetJS (xlsx) para export (CDN)
- localStorage para guardar dados
- Tudo num único ficheiro .html

ESTRUTURA DE DADOS:
localStorage: array de objetos
[
  {
    id: 1,
    categoria: "Alimentação",
    valor: 15.50,
    data: "2024-11-20",
    descricao: "Almoço"
  },
  ...
]

COMPORTAMENTO:
- Ao adicionar despesa: limpar form e atualizar tudo
- Ao eliminar: pedir confirmação
- Ao editar: preencher form com dados e mudar botão para "Atualizar"
- Gráficos atualizam automaticamente com filtros
- Dados persistem mesmo após fechar browser

Cria o código HTML completo com tudo integrado.
```

---

#### MELHORIAS INCREMENTAIS

**Depois de ter a base funcional, adiciona:**

**Iteração 1: Temas Escuro/Claro**
```
Adiciona toggle para alternar entre tema claro e escuro.
Guarda a preferência em localStorage.
```

**Iteração 2: Orçamento Mensal**
```
Permite definir orçamento mensal.
Mostra barra de progresso (quanto já gastaste vs orçamento).
Alerta visual se ultrapassar 90% do orçamento.
```

**Iteração 3: Recorrências**
```
Opção de marcar despesa como "Recorrente" (mensal).
Botão para gerar automaticamente despesas recorrentes do próximo mês.
```

---

### 💻 EXERCÍCIO: Adapta para Outro Caso

Usa o dashboard de despesas como base e adapta para:

**Opção A: Gestor de Tarefas (To-Do)**
- Adicionar tarefas com prioridade
- Marcar como concluída
- Filtrar por estado/prioridade
- Estatísticas de produtividade

**Opção B: Tracker de Exercício Físico**
- Registar treinos (tipo, duração, calorias)
- Gráficos de progresso
- Metas semanais
- Exportar histórico

**Opção C: Gestor de Leituras**
- Adicionar livros (título, autor, páginas, status)
- Progresso de leitura
- Gráfico de livros lidos por mês
- Lista de desejos

---

## 3.3 Análise e Visualização de Dados
**Duração:** 10 minutos

### 🎯 DEMO 4: Analisar Dados e Gerar Insights

#### CENÁRIO: Análise de Vendas de E-commerce

**Dados disponíveis:** CSV com 1000 vendas

```
Data,Produto,Categoria,Quantidade,Valor,Região,Canal
2024-01-15,Produto A,Eletrónicos,2,199.90,Norte,Online
...
```

---

#### PROMPT PARA ANÁLISE COMPLETA

```
CONTEXTO:
Tenho um ficheiro CSV com dados de vendas de e-commerce (1000 linhas).
Preciso de analisar estes dados e gerar insights visuais para apresentar à gestão.

OBJETIVO:
Script Python que analisa os dados, cria visualizações profissionais e 
gera relatório em PDF.

ESTRUTURA DO CSV:
Data,Produto,Categoria,Quantidade,Valor,Região,Canal

ANÁLISES NECESSÁRIAS:
1. VENDAS:
   - Total de vendas (€)
   - Número de transações
   - Ticket médio
   - Evolução mensal

2. PRODUTOS:
   - Top 10 produtos mais vendidos
   - Produtos com maior receita
   - Categorias mais populares

3. GEOGRAFIA:
   - Vendas por região
   - Comparação entre regiões

4. CANAIS:
   - Online vs Loja Física
   - Performance por canal

VISUALIZAÇÕES:
1. Gráfico de linha: Evolução de vendas mensais
2. Gráfico de barras: Top 10 produtos
3. Gráfico de pizza: Vendas por categoria
4. Heatmap: Vendas por região e mês
5. Box plot: Distribuição de valores de venda

RELATÓRIO PDF:
Estrutura:
- Página 1: Resumo executivo (texto + principais métricas)
- Página 2-3: Gráficos e análises
- Página 4: Recomendações baseadas nos dados

REQUISITOS TÉCNICOS:
- pandas para análise
- matplotlib/seaborn para gráficos
- reportlab ou fpdf para PDF
- Design profissional:
  * Logo no cabeçalho
  * Cores corporativas
  * Gráficos com títulos claros
- Tratamento de dados:
  * Limpar valores nulos
  * Detectar e remover outliers extremos
  * Converter datas corretamente

Cria o código completo com comentários explicativos.
```

---

### 🔍 TÉCNICAS DE ANÁLISE COMUNS

**1. Análise Descritiva**
```python
# A IA vai gerar algo como:
df.describe()  # Estatísticas básicas
df.groupby('Categoria').sum()  # Totais por grupo
df['Valor'].mean()  # Média
```

**2. Deteção de Padrões**
```python
# Correlações
df.corr()

# Tendências temporais
df.groupby(df['Data'].dt.month).sum()
```

**3. Visualizações Avançadas**
```python
# Gráfico interativo
import plotly.express as px
fig = px.scatter(df, x='Quantidade', y='Valor', color='Categoria')
```

---

## 3.4 Integração com APIs e Serviços
**Duração:** 10 minutos

### 🎯 DEMO 5: App que Integra API Externa

#### PROJETO: Dashboard de Meteorologia Personalizado

**O que faz:**
- Mostra tempo atual da tua cidade
- Previsão para 5 dias
- Alertas de condições extremas
- Recomendações de roupa

---

#### PROMPT

```
CONTEXTO:
Quero criar uma app web que mostra informações meteorológicas de forma visual e útil.

OBJETIVO:
Dashboard interativo que consome API de meteorologia e apresenta dados de forma intuitiva.

FUNCIONALIDADES:
1. TEMPO ATUAL:
   - Localização (cidade)
   - Temperatura atual
   - Sensação térmica
   - Humidade, vento
   - Ícone representativo (sol, nuvens, chuva)

2. PREVISÃO:
   - Próximos 5 dias
   - Temperaturas máx/mín
   - Probabilidade de chuva
   - Ícones

3. ALERTAS:
   - Destaque visual se:
     * Temperatura < 5°C ou > 35°C
     * Vento > 40 km/h
     * Chuva forte
   - Notificação no topo da página

4. RECOMENDAÇÕES:
   - Sugestão de roupa baseada na temperatura
   - Levar guarda-chuva? (sim/não)
   - Melhor hora para atividades outdoor

5. PESQUISA:
   - Input para mudar de cidade
   - Guardar cidades favoritas (localStorage)

API:
- Usar OpenWeatherMap API (gratuita)
- Endpoint: api.openweathermap.org/data/2.5/weather
- Incluir instruções de como obter API key

DESIGN:
- Fundo que muda consoante condição (sol, nuvens, chuva)
- Animações suaves de transição
- Ícones modernos
- Responsivo

TÉCNICO:
- HTML + CSS + JavaScript
- Fetch API para consumir dados
- Async/await para chamadas
- Tratamento de erros (cidade não encontrada, sem internet)
- Loading spinner durante fetch

Cria o código completo.
```

---

### 🌐 OUTRAS APIS ÚTEIS PARA PRATICAR

**1. API de Notícias**
```
NewsAPI - Últimas notícias por categoria/país
Use case: Agregador de notícias personalizado
```

**2. API de Câmbio**
```
ExchangeRate-API - Taxas de câmbio atualizadas
Use case: Conversor de moedas
```

**3. API de Receitas**
```
TheMealDB - Base de dados de receitas
Use case: App de receitas com pesquisa e filtros
```

**4. API de Filmes**
```
OMDB API - Info de filmes e séries
Use case: Catálogo de filmes com ratings
```

**5. API de Mapas**
```
OpenStreetMap - Mapas e localização
Use case: Localizador de pontos de interesse
```

---

### 💡 EXERCÍCIO FINAL DO MÓDULO

**Desafio: Cria um "Super Dashboard"**

Integra 3 das APIs acima num único dashboard:
- Tempo na tua cidade
- Últimas notícias
- Taxas de câmbio do dia

**Bonus:**
- Atualiza automaticamente a cada 10 minutos
- Modo noturno
- Notificações desktop para alertas importantes

---

## 📝 CHECKPOINT DO MÓDULO 3

### Projetos criados:
✅ Processador automático de Excel  
✅ Sistema de envio de emails em massa  
✅ Dashboard interativo de despesas  
✅ Análise de dados com visualizações  
✅ Integração com APIs externas  

### Skills profissionais:
🔓 Automação de tarefas repetitivas  
🔓 Criação de aplicações web completas  
🔓 Análise e visualização de dados  
🔓 Integração de serviços externos  
🔓 Deploy e manutenção de projetos  

### Tempo poupado:
- Processamento de relatórios: 2-3h → 2 min
- Envio de emails: 2h → 5 min
- Análise de dados: 4h → 15 min

**ROI do curso até agora: Horas poupadas = €€€ ganhos**

---

*Continua para: [MÓDULO 4 - Workflow Profissional](04-MODULO-4-WORKFLOW.md)*
