# MÓDULO 5: Projeto Final Guiado 🎯

⏱️ **Duração:** 30 minutos  
📊 **Formato:** Projeto completo do conceito ao deploy  
🎯 **Objetivo:** Consolidar tudo o que aprendeste num projeto real

---

## Visão Geral do Módulo

Neste módulo final, vais criar um projeto completo usando tudo o que aprendeste:
- Planeamento com IA
- Desenvolvimento iterativo
- Boas práticas de código
- Organização profissional
- Deploy e documentação

**Escolhe 1 dos 3 projetos abaixo** (ou propõe o teu próprio!)

---

## 🎨 OPÇÃO 1: Sistema de Gestão de Tarefas para Equipas

### Descrição
App web para equipas gerirem projetos e tarefas de forma colaborativa.

### Funcionalidades Core
1. **Gestão de Projetos**
   - Criar/editar/eliminar projetos
   - Atribuir cor e prioridade
   - Ver progresso (% tarefas completas)

2. **Gestão de Tarefas**
   - Criar tarefas dentro de projetos
   - Atribuir a membros da equipa
   - Definir deadline e prioridade
   - Marcar como completa
   - Adicionar comentários/notas

3. **Dashboard**
   - Visão geral de todos os projetos
   - Tarefas por fazer vs completas
   - Tarefas atrasadas (alertas visuais)
   - Gráficos de produtividade

4. **Filtros e Pesquisa**
   - Filtrar por projeto, prioridade, membro
   - Pesquisar tarefas
   - Ver só as minhas tarefas

5. **Exportação**
   - Exportar relatório em PDF
   - Exportar tarefas para Excel

### Stack Técnica
- **Frontend:** HTML, CSS (Tailwind via CDN), JavaScript
- **Dados:** localStorage (sem backend)
- **Charts:** Chart.js
- **PDF:** jsPDF
- **Excel:** SheetJS

---

## 📊 OPÇÃO 2: Dashboard de Análise Financeira Pessoal

### Descrição
Sistema completo de gestão financeira pessoal com análise e previsões.

### Funcionalidades Core
1. **Gestão de Contas**
   - Múltiplas contas (bancária, carteira, investimentos)
   - Saldo atual e histórico
   - Transferências entre contas

2. **Receitas e Despesas**
   - Adicionar transações
   - Categorias customizáveis
   - Transações recorrentes
   - Upload de extractos CSV

3. **Análise e Visualizações**
   - Gráfico de despesas por categoria (pizza)
   - Evolução mensal (linha)
   - Comparação com mês anterior
   - Top 5 maiores despesas

4. **Orçamento**
   - Definir orçamento por categoria
   - Alertas quando ultrapassar
   - Progresso visual (barras)
   - Sugestões de poupança

5. **Objetivos Financeiros**
   - Definir metas (ex: poupar 5000€)
   - Calcular quanto poupar por mês
   - Tracker de progresso
   - Projeção de quando atingir meta

6. **Relatórios**
   - Relatório mensal em PDF
   - Insights automáticos (IA analisa padrões)
   - Exportar dados para Excel

### Stack Técnica
- **Frontend:** HTML, CSS (custom + framework opcional), JavaScript
- **Dados:** localStorage + IndexedDB (grande volume)
- **Charts:** Chart.js + D3.js (opcional)
- **IA:** Integração com Claude/ChatGPT para análise de padrões
- **PDF/Excel:** jsPDF, SheetJS

---

## 🏪 OPÇÃO 3: Sistema de Inventário para Pequeno Negócio

### Descrição
Gestão completa de stock para loja/restaurante/armazém pequeno.

### Funcionalidades Core
1. **Gestão de Produtos**
   - Adicionar produtos (nome, categoria, preço, SKU)
   - Upload de imagem
   - Stock atual e mínimo
   - Alertas quando stock baixo

2. **Movimentações**
   - Entrada de stock (compras)
   - Saída de stock (vendas)
   - Ajustes manuais (quebras, devoluções)
   - Histórico completo

3. **Fornecedores**
   - Cadastro de fornecedores
   - Produtos por fornecedor
   - Histórico de encomendas

4. **Relatórios e Análise**
   - Valor total do inventário
   - Produtos mais/menos vendidos
   - Previsão de quando repor stock
   - Produtos parados (sem movimento há X dias)

5. **Códigos de Barras**
   - Gerar código de barras para cada produto
   - Scanner via câmara (para dar baixa rápida)
   - Impressão de etiquetas

6. **Alertas e Notificações**
   - Email quando stock abaixo do mínimo
   - Lista de compras automática
   - Produtos próximos da validade (se aplicável)

### Stack Técnica
- **Frontend:** HTML, CSS, JavaScript
- **Dados:** localStorage + possível integração com Airtable (grátis)
- **Barcode:** JsBarcode library
- **Camera:** HTML5 getUserMedia
- **Notificações:** EmailJS (grátis)
- **PDF:** jsPDF

---

## 📋 Roadmap de Desenvolvimento (Aplica-se a qualquer opção)

### FASE 1: Planeamento (5 min)

#### 1.1 Definir Scope Claro

**Prompt para IA:**
```
Quero criar [PROJETO ESCOLHIDO] com as seguintes funcionalidades:
[lista as funcionalidades core]

Ajuda-me a criar um plano de desenvolvimento dividido em sprints/fases.
Cada fase deve ter:
- Duração estimada
- Funcionalidades a implementar
- Ordem de prioridade (MVP primeiro)
- Dependências entre funcionalidades

Mantém realista para ser completado em 3-4 horas de desenvolvimento.
```

#### 1.2 Criar Wireframes/Mockups

**Prompt para IA:**
```
Para o projeto [X], preciso de decidir o layout e estrutura de páginas.

Preciso de:
1. Lista de todas as páginas/views necessárias
2. Descrição de cada secção de cada página
3. Sugestão de navegação entre páginas
4. Elementos de UI necessários (botões, forms, tabelas, gráficos)

Não preciso de código ainda, só estrutura e organização.
```

---

### FASE 2: Setup do Projeto (5 min)

#### 2.1 Estrutura de Pastas

**Criar:**
```
projeto-final/
├── index.html
├── css/
│   └── style.css
├── js/
│   ├── main.js
│   ├── data.js
│   └── utils.js
├── img/
├── README.md
└── .gitignore
```

#### 2.2 Boilerplate HTML

**Prompt:**
```
Cria o HTML base para [PROJETO], incluindo:
- DOCTYPE e meta tags corretas
- Links para CSS e JS
- Estrutura semântica (header, nav, main, footer)
- CDNs necessários: [lista as bibliotecas]
- Estilos base reset/normalize

Ainda sem funcionalidade, só estrutura.
```

---

### FASE 3: Desenvolvimento Iterativo (60-90 min)

**Abordagem: Feature por Feature**

#### Iteração 1: Funcionalidade Base (Mínimo Viável)

**Para Task Manager:**
```
Implementa a funcionalidade básica de adicionar e listar tarefas:

FUNCIONAL:
- Form para adicionar tarefa (título, descrição, prioridade)
- Validação de form
- Lista de tarefas adicionadas
- Botão para eliminar tarefa
- Guardar em localStorage
- Carregar ao abrir página

DESIGN:
- Layout simples e limpo
- Cores: [escolhe esquema]
- Responsivo (mobile first)

Cria o código completo para esta funcionalidade.
```

**Testa antes de avançar!**
- Adiciona 5-10 tarefas
- Fecha e reabre o browser
- Elimina algumas
- Testa em mobile

---

#### Iteração 2: Funcionalidade Secundária

**Para Task Manager:**
```
Agora adiciona a funcionalidade de marcar tarefas como completas:

NOVO:
- Checkbox em cada tarefa
- Visual diferente para completas (linha atravessada, opacidade reduzida)
- Filtros: Todas / Ativas / Completas
- Contador: X de Y tarefas completas

Mantém o código anterior intacto, apenas adiciona.
```

---

#### Iteração 3: Melhorias Visuais

```
Melhora o design do projeto:

1. Adiciona animações suaves:
   - Transição ao adicionar/remover tarefa
   - Hover effects nos botões
   - Fade in ao carregar página

2. Melhora os forms:
   - Ícones nos campos
   - Estados de focus
   - Feedback visual de validação

3. Adiciona feedback do utilizador:
   - Toasts/snackbars para ações (adicionado, eliminado)
   - Loading states
   - Estados vazios (quando não há tarefas)
```

---

#### Iteração 4: Funcionalidades Avançadas

```
Adiciona análise de dados e visualizações:

1. Dashboard com estatísticas:
   - Total de tarefas
   - Taxa de conclusão (%)
   - Tarefas por prioridade (gráfico pizza)
   - Evolução semanal (gráfico linha)

2. Usar Chart.js (já incluído via CDN)

3. Atualizar gráficos automaticamente quando dados mudam
```

---

#### Iteração 5: Extras e Polish

```
Funcionalidades nice-to-have:

1. Pesquisa/filtros avançados
2. Drag & drop para reordenar
3. Modo escuro
4. Exportar dados (Excel/PDF)
5. Atalhos de teclado
6. Tutorial de primeira utilização

Implementa 2-3 destas funcionalidades.
```

---

### FASE 4: Debugging e Testes (15 min)

#### Checklist de Testes

**Funcionalidade:**
- [ ] Todas as funcionalidades core funcionam
- [ ] Dados persistem após refresh
- [ ] Validações estão a funcionar
- [ ] Não há erros na consola

**Usabilidade:**
- [ ] Intuitivo de usar (pede a alguém para testar)
- [ ] Feedback claro para todas as ações
- [ ] Mensagens de erro úteis
- [ ] Funciona em mobile

**Performance:**
- [ ] Carrega rápido (<2s)
- [ ] Não trava com 100+ items
- [ ] Animações são suaves

**Compatibilidade:**
- [ ] Chrome ✓
- [ ] Firefox ✓
- [ ] Safari ✓
- [ ] Mobile ✓

---

### FASE 5: Organização e Documentação (10 min)

#### 5.1 Código Limpo

**Prompt para IA:**
```
Revê o código do meu projeto e sugere melhorias:

[cola o código]

Foca em:
1. Nomes de variáveis/funções mais claros
2. Comentários em partes complexas
3. Remover código duplicado
4. Organização lógica
5. Separação de responsabilidades

Mantém a funcionalidade igual, apenas refactoring.
```

#### 5.2 README Completo

**Estrutura:**
```markdown
# [Nome do Projeto]

[Screenshot ou GIF do projeto]

## 🎯 Sobre o Projeto

[Breve descrição do que faz e porquê é útil]

## ✨ Funcionalidades

- Feature 1
- Feature 2
- Feature 3

## 🚀 Demo

[Link para demo online]

## 🛠️ Tecnologias Usadas

- HTML5
- CSS3
- JavaScript ES6
- Chart.js
- LocalStorage API

## 📦 Como Usar

### Online
Acede a [link]

### Local
1. Clone o repositório
bash
git clone [url]


2. Abre `index.html` no browser

Não precisa de instalação!

## 💡 Como Funciona

[Explicação técnica breve]

## 📸 Screenshots

[Várias imagens mostrando diferentes partes]

## 🤝 Contribuições

Sugestões e melhorias são bem-vindas!

## 📄 Licença

MIT

## 👤 Autor

[Teu Nome]
- LinkedIn: [link]
- Email: [email]

---

⭐ Desenvolvido como projeto final do curso "Desenvolvimento com IA" da AQIA
```

---

### FASE 6: Deploy (5 min)

#### Opção 1: GitHub Pages (Recomendado)

**Passo a passo:**

1. Cria repositório no GitHub
2. Upload do projeto
3. Settings → Pages → Source: main branch
4. Aguarda 2-3 minutos
5. ✅ Online em: `username.github.io/projeto`

**Prompt se precisares:**
```
Explica passo a passo como fazer deploy do meu projeto web 
(HTML/CSS/JS) no GitHub Pages. Nunca fiz isto antes.
```

---

#### Opção 2: Netlify Drop

1. Vai a https://app.netlify.com/drop
2. Arrasta a pasta do projeto
3. ✅ Online em segundos
4. Pode customizar URL

---

#### Opção 3: Vercel

**Prompt:**
```
Como faço deploy gratuito de projeto HTML/CSS/JS no Vercel?
Passo a passo para iniciante.
```

---

## 5.2 Apresentação e Documentação

### Preparar Para Mostrar

#### 1. Screenshots de Qualidade

**Dicas:**
- Usa dados realistas (não "teste teste 123")
- Mostra as funcionalidades principais
- Desktop E mobile
- Modo claro E escuro (se tiveres)

**Ferramentas:**
- Screely.com (mockups bonitos)
- Cleanshot (Mac)
- ShareX (Windows)

---

#### 2. GIF ou Vídeo Demonstrativo

**Captura:**
- 30-60 segundos
- Mostra workflow completo
- Silencioso (com legendas) ou com narração

**Ferramentas:**
- ScreenToGif (Windows, grátis)
- Giphy Capture (Mac, grátis)
- Loom (browser, grátis)

---

#### 3. LinkedIn Post

**Template:**
```
🚀 Acabei de lançar [Nome do Projeto]!

[Screenshot ou GIF]

O que faz:
• [Funcionalidade 1]
• [Funcionalidade 2]
• [Funcionalidade 3]

🛠️ Tecnologias: HTML, CSS, JavaScript, [outras]

Criado em X dias usando desenvolvimento com IA. 
Aprender esta abordagem mudou completamente como encaro 
a criação de soluções tecnológicas.

👉 Experimenta: [link]
💻 Código: [GitHub link]

#WebDevelopment #IA #AI #TechPortugal #Programming

[Tag pessoas relevantes]
```

---

#### 4. Portfólio

**Adiciona à tua página pessoal:**

```html
<div class="project-card">
  <img src="projeto-screenshot.png" alt="[Projeto]">
  <h3>[Nome do Projeto]</h3>
  <p>[Descrição curta]</p>
  <div class="tags">
    <span>HTML</span>
    <span>JavaScript</span>
    <span>Chart.js</span>
  </div>
  <div class="links">
    <a href="[demo]">Ver Demo</a>
    <a href="[github]">Ver Código</a>
  </div>
</div>
```

---

### Próximos Passos com o Projeto

#### Versão 2.0 - Funcionalidades Adicionais

**Para Task Manager, considera:**
- [ ] Multi-utilizador (autenticação)
- [ ] Sincronização cloud (Firebase)
- [ ] Notificações push
- [ ] Integração com Calendar
- [ ] App mobile (PWA)
- [ ] Modo offline
- [ ] Subtarefas
- [ ] Time tracking
- [ ] Comentários e menções
- [ ] Anexos de ficheiros

**Não faças tudo de uma vez!** Adiciona 1 feature de cada vez.

---

#### Feedback e Iteração

**Onde conseguir feedback:**
1. Amigos e família (testa usabilidade)
2. Reddit (r/webdev, r/SideProject)
3. Twitter/X (partilha com hashtags)
4. LinkedIn (rede profissional)
5. ProductHunt (se for muito bom!)

**Como processar feedback:**
```
Recebi este feedback: [descreve]

No meu projeto atual: [explica arquitetura]

Como posso implementar esta sugestão? 
Dá-me um plano de implementação e código necessário.
```

---

## 🏆 Parabéns pelo Projeto Final!

### O Que Conquistaste

Ao completares este projeto, demonstraste capacidade de:

✅ **Planeamento:** Do conceito à especificação técnica  
✅ **Desenvolvimento:** Iterativo e organizado  
✅ **Problem-solving:** Debug e troubleshooting  
✅ **Design:** UI/UX funcional e atrativo  
✅ **Deployment:** Publicar online  
✅ **Documentação:** Código e projeto  
✅ **Apresentação:** Comunicar o teu trabalho  

---

### Checklist Final

**Antes de considerar o projeto "completo":**

#### Funcionalidade
- [ ] Todas as funcionalidades core implementadas
- [ ] Testado extensivamente
- [ ] Sem bugs críticos
- [ ] Performance aceitável

#### Código
- [ ] Bem organizado (estrutura de pastas)
- [ ] Comentado onde necessário
- [ ] Sem código morto (comentado, não usado)
- [ ] Credenciais/API keys em `.env`

#### Documentação
- [ ] README completo
- [ ] Screenshots/GIFs
- [ ] Instruções de instalação claras
- [ ] Licença definida

#### Deploy
- [ ] Online e acessível
- [ ] URL memorável (custom domain opcional)
- [ ] Funciona em produção
- [ ] SSL ativado (https)

#### Apresentação
- [ ] GitHub profile atualizado
- [ ] LinkedIn post criado
- [ ] Portfólio atualizado
- [ ] Código no GitHub

---

## 📜 Certificado de Conclusão (Conceito)

```
┌─────────────────────────────────────────────────────┐
│                                                     │
│            AQIA - Academia de IA                   │
│                                                     │
│              Certificado de Conclusão              │
│                                                     │
│                  [TEU NOME]                        │
│                                                     │
│       Completou com sucesso o curso                │
│    "Desenvolvimento com IA - Do Zero à Prática"    │
│                                                     │
│              Projeto Final:                         │
│            [NOME DO PROJETO]                        │
│                                                     │
│         Competências Demonstradas:                  │
│    • Desenvolvimento Web                            │
│    • Automação com Python                           │
│    • Análise de Dados                               │
│    • Integração de APIs                             │
│    • Deploy e Documentação                          │
│                                                     │
│         Data: [DATA]                                │
│                                                     │
│    ___________________                              │
│         [Instrutor]                                 │
│                                                     │
└─────────────────────────────────────────────────────┘
```

---

## 🎯 Agora é Contigo!

### Desafios Pós-Curso

**Semana 1:** Adiciona 3 funcionalidades ao teu projeto final

**Semana 2-3:** Cria um projeto novo (completamente diferente)

**Mês 2:** Contribui para um projeto open-source no GitHub

**Mês 3:** Freelance - consegue o teu primeiro cliente

**Mês 6:** Lança um produto (mesmo que pequeno)

---

### Recursos Finais

**Continua conectado:**
- [Community Discord] (se aplicável)
- [Newsletter AQIA] (se aplicável)
- [Grupo LinkedIn] (se aplicável)

**Próximos cursos AQIA:**
- [Lista de cursos avançados]
- [Especializações]
- [Workshops]

---

### Uma Última Reflexão

**Há 2-3 horas atrás:**
"Não sei programar, mas gostava de criar projetos"

**Agora:**
"Criei X projetos funcionais e tenho-os online para o mundo ver"

**Esta é a magia do desenvolvimento com IA:**
- Democratiza a criação tecnológica
- Reduz dramaticamente a barreira de entrada
- Permite foco no problema, não na sintaxe
- Acelera aprendizagem exponencialmente

**Mas lembra-te:**
IA é uma ferramenta poderosa, mas **TU** és o criador. És tu que tens a visão, defines os requisitos, decides o que é bom ou mau, e fazes acontecer.

---

## 🙏 Obrigado!

Por teres investido o teu tempo neste curso.
Por teres abraçado uma nova forma de criar.
Por provares que qualquer pessoa pode construir tecnologia.

**Agora sai e constrói coisas incríveis! 🚀**

---

## 📞 Contactos e Feedback

**Para dúvidas sobre o curso:**
[Email/Discord da AQIA]

**Para partilhar o teu projeto:**
[Formulário ou hashtag]

**Para feedback sobre o curso:**
[Link para formulário]

---

## 🎁 BONUS: Recursos Extra

### Templates Prontos a Usar

**Prompts reutilizáveis:** (ficheiro separado)
**Estruturas de projeto:** (repo GitHub)
**Cheatsheets:** (PDF)
**Lista de APIs gratuitas:** (documento)
**Checklist de lançamento:** (PDF)

---

*Fim do Curso - Início da Tua Jornada! 🎓*

---

## Apêndice: Prompt Master para o Projeto Final

```
PLANEAMENTO COMPLETO DE PROJETO

Quero criar [DESCREVE O PROJETO] usando desenvolvimento com IA.

OBJECTIVO:
[O que o projeto resolve, para quem é, por que é útil]

FUNCIONALIDADES ESSENCIAIS:
1. [Feature 1]
2. [Feature 2]
3. [Feature 3]
[...]

FUNCIONALIDADES NICE-TO-HAVE:
1. [Feature extra 1]
2. [Feature extra 2]

RESTRIÇÕES:
- Tempo disponível: [X horas]
- Nível de experiência: Iniciante
- Deve funcionar no browser (sem backend complexo)
- Deve ser deployável gratuitamente

TECNOLOGIAS PREFERIDAS:
- [Lista se tiveres preferências]

PRECISO DE:
1. Plano de desenvolvimento dividido em fases realistas
2. Wireframe/estrutura de páginas em texto
3. Lista de tecnologias e bibliotecas necessárias
4. Sugestão de ordem de implementação (MVP first)
5. Estimativa de tempo por fase
6. Potenciais desafios e como ultrapassá-los

Cria um plano completo e detalhado para eu começar já!
```

---

*Boa sorte com o teu projeto! 🍀*
