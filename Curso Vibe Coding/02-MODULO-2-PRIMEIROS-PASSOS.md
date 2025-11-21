# MÓDULO 2: Primeiros Passos Práticos ⭐

⏱️ **Duração:** 40 minutos  
📊 **Formato:** 90% Prático, 10% Explicativo  
🎯 **Objetivo:** Criar o teu primeiro projeto completo e dominar os fundamentos práticos

---

## 2.1 O Teu Primeiro Projeto Completo
**Duração:** 15 minutos

### 🎬 DEMO AO VIVO: Landing Page "Café da Esquina"

#### Setup Inicial (1 min)
**O que vais fazer:**
- Abrir ChatGPT ou Claude
- Criar uma landing page profissional do zero
- Sem escrever uma única linha de código manualmente

**Material necessário:**
- Navegador web
- Editor de texto (VS Code ou Notepad)
- Acesso a IA (ChatGPT ou Claude)

---

### FASE 1: Primeiro Prompt - Gerar a Base (3 min)

**O prompt que vais usar:**

```
Quero criar uma landing page para um café chamado "Café da Esquina" em Lisboa. 

Preciso de uma página moderna e acolhedora com:

ESTRUTURA:
- Header fixo com logo (texto) e menu de navegação (Início, Menu, Sobre, Contacto)
- Secção hero com imagem de fundo de café e call-to-action "Reservar Mesa"
- Secção "Sobre Nós" com texto sobre o café
- Galeria com 4 placeholders de imagens
- Secção de horários e localização
- Formulário de contacto (nome, email, mensagem)
- Footer com redes sociais (ícones)

DESIGN:
- Esquema de cores: castanho (#8B4513), bege (#F5DEB3), dourado (#DAA520)
- Tipografia moderna e legível
- Totalmente responsiva (mobile-first)
- Animações suaves no scroll

TÉCNICO:
- Tudo num único ficheiro HTML
- CSS inline ou em <style>
- JavaScript mínimo (só para menu mobile e form validation)
- Usar Font Awesome para ícones (via CDN)

Cria o código completo agora.
```

**O que fazer com a resposta:**

1. **Copiar o código** gerado pela IA
2. **Criar um ficheiro** chamado `cafe-esquina.html`
3. **Colar o código** no ficheiro
4. **Abrir no browser** (duplo clique no ficheiro)

**Resultado esperado:**
✅ Página completa e funcional em 2-3 minutos  
✅ Design profissional  
✅ Responsiva (testa no mobile)  

---

### FASE 2: Primeira Iteração - Refinamento Visual (4 min)

**Problemas típicos que vais encontrar:**
- Imagens placeholder (caixas vazias ou genéricas)
- Botão sem muito destaque
- Cores podem não estar perfeitas

**Prompt de refinamento:**

```
A landing page ficou boa, mas preciso de alguns ajustes:

1. HERO SECTION:
   - Usar uma imagem real do Unsplash de um café acolhedor (pesquisa automática)
   - Adicionar overlay escuro (rgba(0,0,0,0.5)) para destacar o texto
   - Aumentar o tamanho do título
   - O botão "Reservar Mesa" precisa de:
     * Animação hover (scale 1.05)
     * Sombra mais pronunciada
     * Ícone de calendário ao lado do texto

2. GALERIA:
   - Usar imagens reais do Unsplash: café, cappuccino, pastel de nata, interior de café
   - Grid 2x2 em desktop, 1 coluna em mobile
   - Efeito hover: zoom suave (1.1x) com overlay

3. FORMULÁRIO:
   - Adicionar ícones nos campos (pessoa, email, mensagem)
   - Botão de envio mais destacado
   - Estados de hover e focus nos inputs

Atualiza apenas estas secções, mantém o resto igual.
```

**Processo:**
1. IA retorna o código atualizado
2. **Substitui as secções** relevantes no teu ficheiro
3. Refresh no browser
4. Testa as animações e hover effects

---

### FASE 3: Adicionar Funcionalidade - Form Validation (4 min)

**Prompt para validação:**

```
O formulário de contacto precisa de validação funcional:

REQUISITOS:
- Campo Nome: obrigatório, mínimo 3 caracteres
- Campo Email: obrigatório, formato email válido
- Campo Mensagem: obrigatório, mínimo 10 caracteres
- Todos os campos devem mostrar erro visual se inválidos
- Ao submeter com sucesso, mostrar mensagem de confirmação em verde
- Limpar o formulário após submissão bem-sucedida
- NÃO precisa de backend, só validação frontend

FEEDBACK VISUAL:
- Border vermelho se campo inválido
- Mensagem de erro abaixo de cada campo
- Ícone de sucesso quando tudo OK
- Desabilitar botão durante 2 segundos após submissão (evitar spam)

Adiciona o JavaScript necessário.
```

**Testar:**
1. Tentar enviar formulário vazio
2. Colocar email inválido
3. Preencher tudo corretamente
4. Ver mensagem de sucesso

---

### FASE 4: Debugging ao Vivo (3 min)

**Cenário: Algo não funciona**

Vamos simular um erro comum:

**Sintoma:** O menu mobile não abre quando clicas no hamburguer

**Como resolver com IA:**

```
Tenho um problema: quando clico no ícone de menu mobile (hamburguer), nada acontece.

O erro no console do browser é:
"Uncaught TypeError: Cannot read property 'classList' of null"

Aqui está o código do menu:
[cola o código relevante]

Como corrijo isto?
```

**A IA vai:**
1. Identificar que falta um `getElementById` correto
2. Fornecer o código corrigido
3. Explicar o que estava errado

**Tu vais:**
1. Substituir o código
2. Testar novamente
3. ✅ Funciona!

---

### FASE 5: Deploy e Partilha (1 min)

**Opção 1: GitHub Pages (Gratuito)**

```
Como faço deploy desta landing page no GitHub Pages gratuitamente? 
Dá-me os passos exatos, passo a passo.
```

**Opção 2: Netlify Drop**
1. Vai a https://app.netlify.com/drop
2. Arrasta o ficheiro .html
3. ✅ Link público em segundos

**Opção 3: Vercel**
```
Como faço deploy no Vercel? Explica para iniciantes.
```

---

### 📊 RECAP da Demo

**O que fizeste em 15 minutos:**
✅ Landing page profissional completa  
✅ Design responsivo  
✅ Animações e efeitos visuais  
✅ Formulário com validação  
✅ Debug de um erro  
✅ Deploy público  

**Investimento:**
- ⏱️ Tempo: 15 minutos
- 💰 Custo: €0
- 📚 Conhecimento prévio: Zero HTML/CSS/JS

**Resultado:**
- 🌐 Website funcional online
- 📱 Funciona em todos os dispositivos
- 🎨 Design profissional

---

## 2.2 Anatomia de um Bom Prompt
**Duração:** 10 minutos

### Framework C.O.R.E.

Todo bom prompt deve ter:
- **C**ontexto
- **O**bjetivo
- **R**equisitos
- **E**xemplos

---

### 📝 ESTRUTURA DETALHADA

#### C - CONTEXTO
**O que é:**
- Situação ou cenário onde vais usar o código
- Quem vai usar
- Limitações ou restrições

**Exemplo:**
```
❌ Fraco: "Cria um script Python"

✅ Forte: "Trabalho em RH numa empresa com 200 colaboradores. 
Recebo mensalmente ficheiros Excel com dados de assiduidade."
```

---

#### O - OBJETIVO
**O que é:**
- O que precisas de alcançar
- Resultado final esperado
- Problema específico a resolver

**Exemplo:**
```
❌ Fraco: "Para analisar dados"

✅ Forte: "Preciso de automatizar o cálculo de horas trabalhadas por colaborador, 
identificar ausências não justificadas, e gerar relatório mensal para a gestão."
```

---

#### R - REQUISITOS
**O que é:**
- Especificações técnicas
- Funcionalidades obrigatórias
- Restrições ou preferências

**Exemplo:**
```
❌ Fraco: "Com gráficos"

✅ Forte: 
FUNCIONALIDADES:
- Ler múltiplos ficheiros .xlsx de uma pasta
- Calcular total de horas por colaborador
- Identificar quem tem <160h no mês
- Gerar gráfico de barras por departamento
- Exportar para Excel com formatação (headers bold, valores em horas)

TÉCNICO:
- Usar pandas e openpyxl
- Comentários em português
- Tratamento de erros (ficheiros corrompidos, dados em falta)
```

---

#### E - EXEMPLOS
**O que é:**
- Estrutura de dados de input
- Formato esperado de output
- Casos de uso específicos

**Exemplo:**
```
❌ Fraco: [não dá exemplos]

✅ Forte:
ESTRUTURA INPUT (ficheiro Excel):
| Nome | Departamento | Data | HorasEntrada | HorasSaída |
| João Silva | TI | 2024-11-01 | 09:00 | 18:00 |
| Maria Santos | Marketing | 2024-11-01 | 09:30 | 17:30 |

OUTPUT ESPERADO (ficheiro gerado):
- Folha 1: Resumo mensal
  | Colaborador | Total Horas | Status |
  | João Silva | 168 | ✅ OK |
  | Maria Santos | 152 | ⚠️ Abaixo |
  
- Folha 2: Gráfico de barras (horas por departamento)
```

---

### 🎯 EXERCÍCIO COMPARATIVO

Vamos melhorar prompts fracos juntos:

#### CASO 1: Script Python

**❌ PROMPT FRACO:**
```
Cria um script Python para enviar emails
```

**⚠️ PROMPT MÉDIO:**
```
Cria um script Python que envia emails para uma lista de contactos
```

**✅ PROMPT FORTE:**
```
CONTEXTO:
Sou gestor de vendas e preciso de enviar emails personalizados a 50 clientes 
com propostas comerciais. Tenho ficheiro CSV com dados dos clientes.

OBJETIVO:
Automatizar o envio de emails personalizados, onde cada email menciona o nome 
do cliente e o produto que ele demonstrou interesse.

REQUISITOS:
FUNCIONAL:
- Ler CSV com colunas: Nome, Email, Produto, Valor
- Gerar email personalizado usando template
- Enviar via Gmail SMTP
- Adicionar PDF da proposta como anexo (ficheiro único para todos)
- Log de emails enviados (timestamp, destinatário, status)
- Rate limiting: 1 email por segundo (evitar spam)

TÉCNICO:
- Python 3.8+
- Usar smtplib para envio
- Jinja2 para templates
- Tratamento de erros (email inválido, falha no envio)
- Credenciais Gmail via variáveis de ambiente

TEMPLATE DO EMAIL:
Assunto: Proposta Comercial - [Produto]

Olá [Nome],

Conforme conversado, segue em anexo nossa proposta comercial 
para [Produto] no valor de [Valor]€.

[resto do template...]

EXEMPLO CSV:
Nome,Email,Produto,Valor
João Silva,joao@empresa.pt,Software CRM,5000
Maria Santos,maria@outra.pt,Dashboard Analytics,3000
```

**Diferença:** Resposta será 10x mais útil e próxima do que precisas!

---

#### CASO 2: Página Web

**❌ PROMPT FRACO:**
```
Faz uma página de login
```

**✅ PROMPT FORTE:**
```
CONTEXTO:
Aplicação web interna da empresa para gestão de projetos. 
Apenas colaboradores internos têm acesso.

OBJETIVO:
Criar página de login moderna e segura, que valide credenciais 
e redirecione para dashboard.

REQUISITOS:
DESIGN:
- Layout centrado, minimalista
- Logo da empresa no topo
- Campos: email e password
- Checkbox "Lembrar-me"
- Link "Esqueci password"
- Botão "Entrar" destacado
- Cores: Azul (#0066CC), Branco, Cinza claro
- Responsivo (mobile e desktop)

FUNCIONAL:
- Validação frontend:
  * Email formato válido
  * Password mínimo 8 caracteres
- Mostrar/ocultar password (ícone olho)
- Loading spinner durante autenticação
- Mensagens de erro claras
- Segurança:
  * Autocomplete off em password
  * Prevenir SQL injection (prepared statements)
  * Rate limiting (máx 5 tentativas)

TÉCNICO:
- HTML5 + CSS3 + JavaScript vanilla (sem frameworks)
- Integração com backend API (endpoint: POST /api/auth/login)
- Guardar token JWT em localStorage se "Lembrar-me" ativo
- Redirect para /dashboard após sucesso

EXEMPLO DE RESPOSTA DA API:
Success: { "token": "eyJ...", "user": {...} }
Error: { "error": "Credenciais inválidas" }
```

---

### 💡 DICAS PRÁTICAS

**1. Começa Simples, Expande Depois**
```
Iteração 1: "Cria uma calculadora básica em HTML"
↓ testa ↓
Iteração 2: "Adiciona histórico das últimas 5 contas"
↓ testa ↓
Iteração 3: "Melhora o design com cores e botões arredondados"
```

**2. Usa Listas e Bullet Points**
- IA processa melhor informação estruturada
- Mais fácil de ler e validar
- Menos ambiguidade

**3. Sê Específico em Tecnologias**
```
❌ "Usa uma base de dados"
✅ "Usa SQLite (ficheiro local, sem server)"
```

**4. Menciona o que NÃO Queres**
```
"NÃO uses jQuery (só vanilla JS)"
"NÃO precisa de autenticação complexa"
"NÃO é necessário backend (só frontend)"
```

---

### 📋 TEMPLATE REUTILIZÁVEL

Guarda isto para usar sempre:

```
CONTEXTO:
[Quem és, situação, onde vais usar]

OBJETIVO:
[O que precisas de alcançar, resultado final]

REQUISITOS:
FUNCIONAL:
- [Feature 1]
- [Feature 2]
- [Feature 3]

TÉCNICO:
- [Linguagem/Framework]
- [Bibliotecas específicas]
- [Restrições]

DESIGN (se aplicável):
- [Layout]
- [Cores]
- [Responsividade]

EXEMPLOS:
[Estrutura de dados input]
[Formato output esperado]

IMPORTANTE:
- [Qualquer consideração especial]
- [O que evitar]
```

---

## 2.3 Ler e Modificar Código Gerado
**Duração:** 15 minutos

### 🔍 Autópsia de Código

Vamos dissecar a landing page que criaste para perceber a anatomia.

---

### PARTE 1: Estrutura HTML (5 min)

#### O que é ESSENCIAL (não mexer)

```html
<!DOCTYPE html>
<html lang="pt">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Café da Esquina</title>
```

**Por quê?**
- `DOCTYPE`: Diz ao browser que é HTML5
- `charset=UTF-8`: Suporta acentos e caracteres especiais
- `viewport`: Faz a página funcionar bem em mobile

**Regra:** 🔒 Não mexas nisto a menos que saibas exatamente o quê

---

#### O que é CUSTOMIZÁVEL (mexer à vontade)

```html
<title>Café da Esquina</title>
```
➡️ Muda para o nome do teu projeto

```html
<h1>Bem-vindo ao Café da Esquina</h1>
<p>O melhor café de Lisboa</p>
```
➡️ Muda textos livremente

```html
<img src="cafe.jpg" alt="Interior do café">
```
➡️ Muda imagens (certifica-te que o caminho está correto)

---

#### O que é ESTRUTURAL (entender antes de mexer)

```html
<nav>
    <ul>
        <li><a href="#inicio">Início</a></li>
        <li><a href="#menu">Menu</a></li>
    </ul>
</nav>
```

**Estrutura:**
- `<nav>` = contentor de navegação
- `<ul>` = lista não ordenada
- `<li>` = item da lista
- `<a>` = link

**Para adicionar item:**
```html
<li><a href="#galeria">Galeria</a></li>
```

---

### PARTE 2: Estilos CSS (5 min)

#### Onde estão os estilos?

**Opção 1: Inline**
```html
<style>
    body {
        font-family: Arial, sans-serif;
        margin: 0;
        padding: 0;
    }
</style>
```

**Opção 2: No elemento**
```html
<div style="color: red; font-size: 20px;">Texto</div>
```

---

#### Anatomia de uma Regra CSS

```css
.botao-reserva {           /* ← Seletor (quem recebe o estilo) */
    background: #DAA520;   /* ← Propriedade: valor */
    color: white;
    padding: 15px 30px;
    border: none;
    border-radius: 5px;
}
```

**Tipos de Seletores:**
- `.classe` = Todos os elementos com essa classe
- `#id` = Elemento específico com esse ID
- `button` = Todos os botões
- `.classe button` = Botões dentro de elementos com .classe

---

#### Modificações Comuns

**MUDAR CORES:**
```css
/* ANTES */
background: #8B4513;

/* DEPOIS (azul) */
background: #0066CC;
```

**MUDAR TAMANHOS:**
```css
/* ANTES */
font-size: 16px;

/* DEPOIS */
font-size: 20px;
```

**MUDAR ESPAÇAMENTOS:**
```css
/* ANTES */
padding: 10px;

/* DEPOIS (mais espaço) */
padding: 20px;
```

---

### PARTE 3: JavaScript (5 min)

#### Onde está o JavaScript?

```html
<script>
    // Código aqui
</script>
```

Normalmente no final do `<body>`

---

#### Padrões Comuns

**1. Seleção de Elementos**
```javascript
const botao = document.getElementById('meuBotao');
const formulario = document.querySelector('.formulario');
```

**2. Event Listeners (ações ao clicar, etc)**
```javascript
botao.addEventListener('click', function() {
    alert('Botão clicado!');
});
```

**3. Manipulação de Conteúdo**
```javascript
// Mudar texto
elemento.textContent = 'Novo texto';

// Mudar estilo
elemento.style.color = 'red';

// Adicionar classe CSS
elemento.classList.add('ativo');
```

---

### 🛠️ EXERCÍCIO PRÁTICO GUIADO

**Desafio:** Mudar o esquema de cores da landing page de café para restaurante italiano

**Passo 1: Identifica as Cores Atuais**
```
Café: Castanho (#8B4513), Bege (#F5DEB3), Dourado (#DAA520)
↓
Italiano: Verde (#228B22), Vermelho (#DC143C), Branco (#FFFFFF)
```

**Passo 2: Prompt para IA**
```
No código da minha landing page "Café da Esquina", preciso de mudar 
o esquema de cores para um tema de restaurante italiano:

CORES ATUAIS → CORES NOVAS:
- Castanho #8B4513 → Verde #228B22
- Bege #F5DEB3 → Branco #FFFFFF  
- Dourado #DAA520 → Vermelho #DC143C

Dá-me uma lista de todas as linhas CSS que preciso de alterar.
Mostra lado a lado: código antigo | código novo
```

**Passo 3: Aplica as Mudanças**
- Usa Find & Replace no editor (Ctrl+H)
- Substitui cor por cor
- Guarda e testa

---

### 🎯 EXERCÍCIO: Modificação Independente

**Agora é contigo! Escolhe UMA destas modificações:**

**Opção A: Adicionar Nova Secção**
```
Adiciona uma secção "Testemunhos de Clientes" entre 
a galeria e o formulário de contacto.

Deve ter:
- Título "O Que Dizem de Nós"
- 3 cards com: foto (placeholder), nome, depoimento
- Layout em linha em desktop, coluna em mobile
```

**Opção B: Melhorar Formulário**
```
Melhora o formulário de contacto:
- Adiciona campo "Telefone" (opcional)
- Adiciona campo "Data Preferida" (date picker)
- Muda o botão de envio para ter um ícone de avião
- Adiciona contador de caracteres na mensagem (máx 500)
```

**Opção C: Adicionar Menu de Produtos**
```
Cria uma secção "Nosso Menu":
- Grid de produtos (nome, descrição curta, preço, imagem)
- 6 produtos de exemplo
- Filtros: "Café", "Pastelaria", "Bebidas"
- Ao clicar num produto, mostrar modal com detalhes
```

---

### 🔍 COMO DEBUGAR

**Processo em 3 Passos:**

**1. Identifica o Erro**
```
- Browser não mostra nada? → Erro HTML
- Design está quebrado? → Erro CSS  
- Funcionalidade não funciona? → Erro JavaScript
```

**2. Usa Developer Tools**
```
- Windows/Linux: F12
- Mac: Cmd + Option + I

Console → Ver erros JavaScript
Elements → Inspeccionar HTML/CSS
```

**3. Pergunta à IA**
```
Tenho este erro:
[cola mensagem de erro ou descreve o problema]

No código:
[cola a parte relevante]

Como corrijo?
```

---

## 📝 CHECKPOINT DO MÓDULO 2

### O que fizeste:
✅ Criaste uma landing page completa do zero  
✅ Refinaste o design em iterações  
✅ Adicionaste funcionalidades (validação de form)  
✅ Debugaste um erro  
✅ Fizeste deploy online  
✅ Aprendeste a estrutura de prompts eficazes  
✅ Dissecaste código HTML/CSS/JS  
✅ Modificaste código com confiança  

### Skills desbloqueadas:
🔓 Criar projetos web funcionais  
🔓 Iterar e refinar código  
🔓 Debug básico  
🔓 Deploy de websites  
🔓 Leitura de código gerado  
🔓 Modificação segura de código  

---

## 💪 DESAFIO PÓS-MÓDULO

**Cria sozinho:** Uma landing page para um negócio à tua escolha

**Sugestões:**
- Clínica de estética
- Ginásio local
- Loja de roupa
- Escritório de advogados
- Freelancer (tu mesmo!)

**Requisitos mínimos:**
- Header com navegação
- Hero section com CTA
- 3 secções de conteúdo
- Formulário de contacto
- Footer
- Design responsivo
- Deploy online

**Tempo estimado:** 30-45 minutos

---

*Continua para: [MÓDULO 3 - Casos de Uso Profissionais](03-MODULO-3-CASOS-USO.md)*
