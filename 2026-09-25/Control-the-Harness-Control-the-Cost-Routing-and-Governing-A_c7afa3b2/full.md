# Control the Harness, Control the Cost Routing and Governing AI Coding Agents in the Enterprise

Arian Abbasi<sup>1</sup> Alan Aqrawi<sup>1</sup> Ted Kwartler<sup>1,2,3</sup>

<sup>1</sup>Accenture Responsible AI <sup>2</sup>Accenture Americas Advanced AI Practice

<sup>3</sup>Harvard Extension School, Harvard University

Preprint, 24 September 2026

## Abstract

Harnesses, the products that run AI coding agents, are multiplying, and enterprises are rolling them out to their employees: what started as pilots with a few hundred seats is scaling to tens of thousands. Most enterprises do not build these harnesses but buy them from large vendors, such as Anthropic’s Claude Code or OpenAI’s Codex. A harness decides which model answers, what the model reads, how the prompt cache is used and which subagents run, so it picks the rate on the price sheet and sets the volume bought at it. Enterprises that keep a proprietary or untuned harness at its defaults inherit these choices and their bill. We build a fast, customisable router in which Jev, a classifier with calibrated probabilities, labels every prompt against a bring-your-own taxonomy of agentic requests. Because one user turn is many requests over a prompt cache that belongs to one model, the router moves work only where no running conversation has to rebuild its cache: at session start, in side lanes and at subagent launch. From the price sheet we derive when a mid-task switch pays back, and a crossover: on long tool-heavy sessions the highest-priced model costs less than the next tier, as repricing about 10,000 real sessions from public datasets confirms. In an emulated enterprise of 10,000 seats with user behaviour taken from these datasets, the router recovers 14 to 21% of model spend at Anthropic’s list prices of 21 September 2026, \$3.3M to \$5.0M a year. The paper also maps the risks across twenty harnesses, prices the dependence on one vendor’s models, and proposes a control plane that enterprises can run from within, starting now, with a ladder for deciding later whether to own the harness.

## Key findings

## 14–21%

\$3.3–5.0M

51–80%

of agentic model spend recovered a year for a 10,000-seat enterprise of long real sessions cost less on by cache-safe routing the top model than on Opus 5

The harness sets the bill, and routing cuts it by 14 to 21%. We emulated an enterprise of 10,000 seats, with corrections, pauses and subagents as measured in public datasets of real sessions. It spends \$23.7M a year on models, \$40 per engineer per active day. Sending each task to the cheapest model that can do it, without disturbing work in progress, saves \$3.3M to \$5.0M a year. A one-line change in a long coding session drops from \$0.88 to \$0.36.

List price is the wrong metric. On long tool-heavy sessions the top model re-reads the conversation at half the price of the next tier and costs less in 51 to 80% of real sessions over \$100. A mid-task switch discards the cached conversation and pays of only after 8 to 291 requests.

Pauses matter as much as the classifier. The API cache lasts five minutes, and 17 to 27% of user turns follow a longer pause and pay for the whole conversation again. The saving runs from 24% without such pauses to 7% for users who pause most, and a one-hour cache adds three points. Move work only when sure. Corrections are over a third of the bill, and a task given to too weak a model triggers more. The router acts only on confident labels and never downgrades a correction.

Take control from within, starting this quarter. Put a cache-safe gateway in front of every harness, choose models at session start and subagent launch, turn on the one-hour cache, report token use only in aggregate, let each function define its own task categories and ship one short bundle of procedures and connectors. Decide later, on purpose, whether to own the harness. Harnesses that accept the enterprise’s own endpoint also reach other vendors’ strong models, at 0.12 to 0.80 times the cost of Opus 5.

## 1 Introduction

Agents are becoming the way enterprises use frontier models. A harness sends the model a task, runs the tools it asks for, feeds the results back and repeats until the work is done. Claude Code, Codex, Gemini CLI, OpenCode and more than a dozen others ship this loop as a product, and enterprises are moving them from pilots to production. Accenture announced training on Claude for about 30,000 professionals and Claude Code for tens of thousands of its developers [1], Cognizant Claude for up to 350,000 employees [2], and PwC a rollout of Claude Code and Cowork that starts with its U.S. teams and expands toward a global workforce of hundreds of thousands [3]. The work grows with them: GitHub’s Copilot coding agent alone opened more than a million pull requests between May and September 2025 [4], and the share of Codex users who sent at least one prompt estimated to need an hour or more of skilled work doubled in five months, from 35 to 70% [5]. In 2025 most developers had yet to adopt agents [6], so the defaults an enterprise accepts now will govern most of its agentic work to come.

Most enterprises do not build these harnesses but buy them as products, and with the product come its defaults. The harness chooses the model for every request, writes the prompts, manages the prompt cache, launches subagents and decides what is logged. The price sheet sets what a token costs, and the harness decides which rate applies and how many tokens are bought at it, so it shapes both what the work costs and who controls it. In Claude Code the defaults are the vendor’s own models, subagents on the parent’s model and a five-minute cache for API customers. Section 3 names the four risks this hands to the enterprise: coupling to one vendor’s prices, closed code, skills and connectors fragmented across harnesses, and dependence on individual user skill.

Cost is where the defaults show first, and routing is the obvious lever: send each task to the cheapest model that can do it. Chat routers decide per request, and that fails inside a harness. There one user turn is many requests over a prompt cache that belongs to one model, and moving a turn to another model pays to rebuild the cache. We make routing work inside the harness. From the price sheet we derive after how many requests a mid-task switch pays back (Section 4.2). Under Anthropic’s list prices of 21 September 2026 the highest-priced model also costs less per session than the next tier for long tool-heavy work, a crossover that repricing about 10,000 real sessions confirms (Section 4.3). The router therefore moves work only where no running conversation has to rebuild its cache: it chooses the model at session start, runs small tasks in separate lanes and routes subagents at launch. Jev, a classifier built for fast structured decisions, labels every prompt against a bring-your-own taxonomy of agentic requests in under half a second, and the router moves a turn only when Jev’s calibrated probabilities say the cheaper model sufices (Section 4).

In an emulation of a 10,000-seat enterprise with user behaviour measured in real sessions, the router recovers 14 to 21% of model spend, \$3.3M to \$5.0M a year, and users’ pauses move the saving as much as the classifier does (Section 5). The harness also decides which vendors are in reach. On the same real sessions, strong models of other vendors cost 0.12 to 0.80 times Opus 5, and Claude Code, which works with Anthropic’s models only, cannot call them. Section 6 turns these results into a control plane that an enterprise can run from within in the short term, and a ladder for deciding later, on purpose, whether to own the harness.

## Contributions.

1. A risk map of third-party agentic harnesses, with a coupling matrix over twenty harnesses, eight of them from Chinese vendors, built from licence files and configuration documents (Table 8).

2. A cost model of agentic turns under model-scoped caching: the payback rule for mid-task switches and the crossover result, checked against real sessions.

3. A cache-safe router: execution modes that route at session start, in side lanes and at subagent launch, and a policy that acts on a fast classifier’s calibrated probabilities over a bring-your-own taxonomy.

4. An emulation method for cost cases before telemetry exists: a tagged catalogue perturbed with behaviour measured in real sessions, a live classifier, misroutes charged as correction chains, population scaling.

5. The case study with its sensitivities, the price of lock-in on real sessions, a control-plane architecture and a sovereignty ladder.

Section 2 explains how harnesses work and who ships them, Section 3 states the problem, Section 4 derives the cost rules and specifies the router and its emulation, Section 5 reports the results, Section 6 turns them into recommendations, Section 7 places the work and Section 8 states its limits.

## 2 Background

## 2.1 Every tool step re-sends the conversation

The loop. A harness holds a conversation with a model on the user’s behalf. Each user message starts a turn. Within the turn the model may call tools (read a file, run tests, search), the harness executes them and sends the results back, and the model continues. A turn with k tool steps is k+1 API requests, each carrying the whole conversation so far. An agentic turn of eighteen tool steps re-sends the conversation nineteen times.

The context window. The conversation grows with every turn and tool step until it meets the model’s context window, the ceiling on what one request can hold, input and output together: 1M tokens for Fable 5.1, Opus 5 and Sonnet 5, and 200k for Haiku 4.5 [7]. A request whose input alone exceeds the window is rejected. Before that point the harness compacts the conversation, replacing its history with a summary and starting its cache afresh [8, 9], or the user starts a new one. A continuing conversation is therefore one whose context keeps filling, and every request in it re-reads more than the last until compaction resets it.

The cache. Vendors bill re-sent context at a discount if the request shares a prefix with a recent request to the same model: the prompt cache. The prefix must match exactly and the cache belongs to one model, so sending the same conversation to another model recomputes everything [8, 10]. Cache reads cost a fraction of the input price, cache writes carry a premium, and entries expire after minutes.

The price sheet. On 21 September 2026 Anthropic listed its four models, dearest first, at \$10 and \$50 per million input and output tokens for Fable 5.1, \$5 and \$25 for Opus 5, \$2 and \$10 for Sonnet 5 and \$1 and \$5 for Haiku 4.5 [11]. Table 1 gives the cache rules the three large vendors apply on top of their prices, and everything that follows turns on them.

Layers, hooks, subagents. The conversation is ordered as system prompt, project instructions, then messages, so a change deep in the conversation leaves the earlier layers cached [8]. Harnesses expose hooks that run at lifecycle events and can rewrite a tool call before it runs [15]. They spawn subagents that start a fresh conversation with their own context [16], read instruction files (AGENTS.md, CLAUDE.md), load skills (SKILL.md folders) and connect tools through the Model Context Protocol (MCP).

Table 1: Prompt-cache price rules by vendor (Anthropic’s of 21 September 2026, OpenAI’s and Google’s of 23 September 2026). Multipliers are of the model’s input price.
<table><tr><td>Vendor</td><td>Cache read</td><td>Cache write</td><td>Lifetime</td><td>Minimum prefix</td></tr><tr><td>Anthropic [10, 11]</td><td>0.025× (Fable 5.1), 0.1× others</td><td>1.25× (5 min), 2× (1 h)</td><td>5 min or 1 h</td><td>512 to 4,096 tokens</td></tr><tr><td>OpenAI, GPT-5.6 and later [12, 13]</td><td>0.1×</td><td>1.25×</td><td>30 min</td><td>1,024 tokens</td></tr><tr><td>Google, Gemini [14]</td><td>0.1×</td><td>input price plus a storage fee per hour (explicit cache)</td><td>not stated (implicit caching on by default)</td><td>2,048 to 4,096 tokens</td></tr></table>

## 2.2 Model coupling is a design choice

Table 8 (Appendix C) grades twenty harnesses on the properties an enterprise inherits, eight from Chinese vendors and twelve from elsewhere. Three facts stand out. First, openness and model coupling are independent axes. Gemini CLI is Apache-2.0 but talks only to Google, while Amp and Copilot CLI are proprietary but multi-model. Claude Code is proprietary and works with Anthropic’s models only: its public repository holds issues and plugins but no source, and its documentation states that routing it to non-Claude models through a gateway is unsupported [17, 18]. Second, where a default is documented, subagents inherit the parent’s model (seven of the twenty), and the rest set the model per agent, leave it undocumented or have no subagents. Third, the harnesses of Chinese vendors, Qwen Code, DeepSeek Harness, Kimi Code, Trae Agent, ZCode and MiniMax Code, are MIT or Apache-2.0 and accept any OpenAIor Anthropic-compatible endpoint, with CodeBuddy and iFlow CLI the closed exceptions. Model coupling is therefore a design choice of each vendor, not a property of the category.

## 3 Problem statement

Buying the harness as a product transfers four risks to the enterprise, and each is set by configuration an enterprise can read today (Table 8).

## 3.1 Coupling ties the enterprise to one vendor’s prices

A harness bound to one vendor’s models binds the enterprise to that vendor’s prices, plans and limits. The binding is graded, not binary. Kiro accepts no other endpoint, and Claude Code no other vendor’s models (a gateway can sit in front of it, for Claude models only). Cursor accepts a few vendors’ keys, and Codex, OpenCode, pi, Goose, Aider, Cline and Copilot CLI accept an enterprise’s own or third-party endpoints. Cloud marketplaces (Bedrock, Vertex, Foundry) change the invoice, not the model or its price structure [19]. The exposure is not hypothetical. Cursor replaced request-based plans with usage-based pricing in June 2025 and apologised and refunded in July [20], Anthropic announced weekly rate limits for its Pro and Max subscriptions in late July, citing demand for Claude Code [21], and GitHub announced in April 2026 that Copilot moves to usage-based billing in June [22]. Developers do not rank lock-in among their top concerns [23], so it lands on procurement and the platform team.

## 3.2 Enterprises pay for behaviour they cannot inspect

Closed harness code means the enterprise cannot read what is sent on its behalf, audit prompt changes between versions, or verify how the cache, retries and subagents behave. The gap is real for Claude Code, Cursor, Kiro, Amp, CodeBuddy and iFlow CLI, partial for Copilot CLI (public repository, proprietary licence), and absent for the Apache-2.0 and MIT harnesses. Opacity compounds the pricing exposure: a bill set by a coupled harness is also a bill the enterprise cannot audit.

## 3.3 Connectors travel, the control plane does not

A workforce split between two harnesses runs two skill bundles, two sets of connectors, two governance configurations and no shared telemetry. Three layers are already portable. MCP carries connectors in every surveyed harness except Aider and pi, which refuses it by design. AGENTS.md carries instructions: multi-vendor since August 2025, it is stewarded by the Linux Foundation and used by over 60,000 projects [24]. Agent Skills, an open standard since December 2025, carries procedures, with oficial adoption by OpenCode, GitHub Copilot, Gemini CLI, Cursor, Junie and Goose [25]. What is not portable is exactly the control plane: model policy, telemetry, hook and permission semantics, subagent behaviour and billing.

## 3.4 Novices get half the results for the same spend

The same harness produces very diferent results for diferent users. Anthropic’s analysis of about 400,000 Claude Code sessions rates each session’s user expertise from the precision of directions, what the user asks to be verified, and who corrects whom, and finds the rating task-specific and “quite diferent from job title” [26]. Novice-rated sessions reach verified success 15% of the time, against 28 to 33% for intermediate users and above, and 19% of them end abandoned, against 5 to 7% for everyone else. A novice prompt sets of about five actions and 600 words of output, an expert prompt twelve actions and 3,200 words [26]. A randomised trial found experienced open-source developers 19% slower with AI tools while they believed they were faster [27]. Anthropic’s own data shows a 10% higher success rate for higher-tenure users that task selection does not explain [28], and DORA summarises the mechanism as “AI doesn’t fix a team; it amplifies what’s already there” [29].

The cost consequence is direct. For a novice-rated user the same spend buys about half the verified successes, and a fifth of the sessions produce no outcome at all. A static instruction file does not close the gap on its own: repository context files do not raise an autonomous agent’s success on well-specified benchmark tasks and add over 20% to inference cost, while explicit instructions are followed and files help for non-standard practices [30, 31]. The lever that reaches every user is the harness itself. Training reaches some users and scales with headcount, while harness configuration reaches every user at no marginal cost. The three expertise signals map onto features a harness can ship: scafolds for precise requests, hooks that verify before a task completes, review gates, and a model policy that lowers the price of the waste that remains.

## 3.5 One layer decides all four risks

Every risk above is decided in the layer between the user and the model vendor: which model, which prompts, which subagents, which telemetry, which floors. Enterprises can own that layer today without owning the harness (Section 6). Governance constrains how. Per-user cost and productivity metrics are the obvious by-product of a routing layer and the most dangerous one: in the EU, employee monitoring by AI falls under the AI Act’s high-risk provisions and, in Germany, under works-council co-determination [32]. The design in Section 4 therefore has no productivity axis and reports aggregates only. Data retention and refusal policies difer by model and exclude some workloads in some organisations, and the routing policy carries those floors. Cost is the risk with a price tag, and the next two sections price it.

## 4 Methodology

Routing inside a harness is a cache problem: the unit to price is the turn, a switch must earn back its cache writes, and a lower list price does not mean a lower session cost. The router and the emulation below are built on these three rules.

## 4.1 Price the turn, not the request

In a harness one user turn is k+1 API requests over a conversation that is cached on one model. That is the unit a router has to price, and it is where routing for chat products and routing for harnesses part ways. The routing literature chooses a model per request from an estimate of dificulty and treats each request as free to place [33–35], but in a harness the request carries the cached conversation. Writing C for the conversation prefix, D for the new tokens a turn adds and k for its tool steps, a warm turn reads roughly $( k { + } 1 ) C + D k / 2$ cached tokens, writes D and produces its output (Appendix B). Moving the turn to another model turns the reads into a cold write of C at the write premium and, when the user continues on the original model, a second cold write there.

## 4.2 A mid-task switch must earn back its cache writes

Let $R _ { \mathrm { h } }$ be the home model’s cache-read price per token, $R _ { \mathrm { t } }$ the target’s read price, $W _ { \mathrm { t } }$ the target’s write price and $W _ { \mathrm { h } }$ the home model’s rewrite price. A mid-task switch that later returns pays only if the cheaper stretch lasts more than

$$
n ^ { * } = { \frac { W _ { \mathrm { t } } + W _ { \mathrm { h } } - R _ { \mathrm { t } } - R _ { \mathrm { h } } } { R _ { \mathrm { h } } - R _ { \mathrm { t } } } } { \mathrm { ~ r e q u e s t s , } }  \quad \quad n _ { \mathrm { n o ~ r e t u r n } } ^ { * } = { \frac { W _ { \mathrm { t } } - R _ { \mathrm { t } } } { R _ { \mathrm { h } } - R _ { \mathrm { t } } } } .\tag{1}
$$

The return request rewrites the home cache instead of reading it, so it costs $W _ { \mathrm { h } } - R _ { \mathrm { h } }$ more than staying. On Anthropic’s sheet, Fable 5.1 to Sonnet 5 gives $n ^ { * } = 2 9 1$ requests with return and 46 without, and Opus 5 to Sonnet 5 gives 27 and 8. A naive switch can therefore pay on an Opus-pinned session over a few consecutive routine turns and never pays on a Fable-pinned session within a task. Figure 1 shows the emulation confirming this on the technical scenarios: the naive per-request switch costs more than pinning once misroutes are charged, and the router beats both. GitHub Copilot’s documentation states the same observation operationally, routing “along natural cache boundaries” because switching mid-session “has shown increased cost” [36]. Equation 1 says when and by how much.

## 4.3 The top model undercuts the next tier on long sessions

Cache reads dominate long agentic sessions, and vendors price them per model. On Anthropic’s sheet of 21 September 2026, Fable 5.1 reads cached context at \$0.25 per million tokens against \$0.50 for Opus 5, while its input and output prices are double [11]. Figure 2 plots the cost of one warm user turn against context size. With eighteen tool steps per turn the highest-priced model becomes cheaper than Opus 5 above 65k tokens of context, and with six steps above 211k, both well inside the 1M-token window (Table 2). A router that ranks models by list price gets long sessions exactly wrong. Real sessions bear this out. Repriced token for token on both models, the Claude Code sessions in two public corpora that cost \$100 or more on Opus 5 are cheaper on Fable 5.1 in 51% (TraceLab) and 80% (SWE-chat) of cases, and spend concentrates in exactly such sessions: the top 1% of sessions carry 31 to 53% of it (Figure 3; our repricing of [37, 38]). This one efect produces three quarters of the saving in the case study (Section 5.2).

![](images/2cb8c69eb24a27c66829f27b00b44d1ac047a79b89382b38486c1d78b7a1406b.jpg)  
Figure 1: Six technical scenarios priced four ways: pinned, with naive per-request switching, and with the router on the live classifier’s labels and on perfect labels. The naive per-request switch pays a cold cache write on every move and, when the tags say the cheaper model was too weak, redoes the turn; the router runs small tasks in a separate lane and never rewrites the main conversation.

6 tool steps per user turn  
![](images/25ee314c299819adf49979fbc7bf0d0a638603e0ca5faf2c1d390dcc2da0fdac.jpg)

18 tool steps per user turn  
![](images/df7b9ff07d219775245cdcb875bfe1eee2447e2e4d4d41def7402a2fcba9a9de.jpg)  
Conversation context (thousand tokens)  
Figure 2: Cost of one user turn with a warm cache against conversation context, for 6 and 18 tool steps per turn (40k new tokens and 6k output per turn; list prices of 21 September 2026). The break-even moves left as tool steps grow.

## 4.4 Route without touching running work

The rules follow from the arithmetic.

R1. Decide the model when the session is small, and never rewrite a running conversation’s model.

R2. Run bounded side requests in a separate conversation seeded with a compact project ledger rather than the transcript, so the side lane writes a few thousand tokens instead of the whole context.

R3. After a side lane returns, refresh the home cache with one empty request, so the next home turn does not pay a full rewrite.

R4. Append the side lane’s result as a note; never edit earlier turns.

R5. Branch only when the home context is large enough that the lane’s setup is cheaper than re-reading in place.

R6. Treat a new task as the moment to re-choose.

R7. Choose by estimated session cost, not by tier.

R8. Verify that any gateway forwards the cache markers.

R9. Route subagents at launch: a subagent is a fresh conversation, so choosing its model costs nothing in cache terms, and fresh short conversations are write-heavy, where the top model is dearest.

Table 2: Context above which Fable 5.1 is cheaper than Opus 5 for one warm turn, by tool steps per turn (assumptions as in Figure 2).
<table><tr><td>Tool steps per user turn</td><td>Context above which Fable 5.1 is cheaper than Opus 5</td></tr><tr><td>3 385k</td><td></td></tr><tr><td>6</td><td>211k</td></tr><tr><td>12</td><td>105k</td></tr><tr><td>18</td><td>65k</td></tr><tr><td>30 32k</td><td></td></tr></table>

TraceLab 5,319 Claude Code sessions, priced on Opus 5  
SWE-chat 4,806 Claude Code sessions, priced on Opus 5

![](images/e39c3df500811df82603e397d84cb90dcd2bd335ec752cc9b6192ecd23c8ccfe.jpg)  
Cumulative share of sessions, most expensive first  
Figure 3: Where real-session spend sits: the Claude Code sessions of TraceLab and SWE-chat priced on Opus 5 (list prices of 21 September 2026), most expensive first. A few long sessions carry most of the spend: the black curve counts all sessions, the purple curve only the sessions that cost less on Fable 5.1.

R10. Never move a correction below the tier of the turn it corrects.

A side lane pays most where the context is largest: a one-line change inside an 805k-token session costs \$0.88 in place on Fable 5.1 and \$0.36 in a ledger side lane on Sonnet 5, the ledger and the home refresh included (Figure 4). Five typical subagents cost \$4.88 when they inherit Fable 5.1 and \$1.27 on Sonnet 5, with the parent’s own turn at \$0.24 either way, and Figure 11 gives one subagent’s cost by model and profile. Seven harnesses in Table 8, Claude Code among them, document that subagents inherit the parent’s model by default. In Claude Code a pre-tool hook can rewrite the launch arguments, including the model, before the subagent starts [15].

## 4.5 Label with a bring-your-own taxonomy, move only when sure

Routing needs labels. We use a bring-your-own taxonomy of four axes (Appendix A): the action requested (thirteen labels from lookup to build and steer), the domain (twelve), an ordinal complexity (trivial, routine, complex, expert) and how much the request depends on context, from none to the conversation history, plus stakes and output-shape modifiers. Complexity maps to a capability tier, from Haiku 4.5 through Sonnet 5 (the mid tier) and Opus 5 to Fable 5.1. Domain sets governance floors: legal, finance and HR never go below the mid tier, and neither do agentic edits. Stakes can raise the tier but never above the user’s own model. There is deliberately no productivity axis. The policy acts only when the classifier’s probability that the needed tier is at or below the chosen one, and that the turn does not depend on history, both clear a threshold. Otherwise it holds the turn on its current model, where it runs exactly as it would have without a router. A hold forgoes a saving, a misroute costs quality and rework, and Section 5.3 measures both. Corrections need care: “no, that is wrong” reads as trivial, but the steer label inherits the task’s tier and the turn depends on the history, so the router holds it, and a separate classifier question asks of every turn whether it corrects the previous output (R10). Repeated corrections signal a task harder than it looked, and that is the moment to consider moving the task up, never down, at the price the payback rule gives.

## 4.6 Jev classifies every prompt in under half a second

Every prompt waits for the routing decision, so the classifier sits on the user’s critical path: a decision that takes two seconds is felt on every turn, one under half a second is not. That rules out a frontier model as the classifier, on latency and on cost, and rules in small purpose-built ones. Once a taxonomy bounds the answers to a small, enumerated set of labels, the classifier only has to score fixed options, a job that trades capacity the task never uses for latency.

The router uses Jev, a model of the class TypeSafe AI calls System One, released in September 2026 [39]. The name draws on the distinction between fast, intuitive System 1 thinking and slow, deliberate System 2 reasoning: Jev makes structured decisions and does not generate text. Its architecture explains its speed. A generative model writes its answer token by token, one forward pass per token, and a probability for each option needs several samples or access to its logits. Jev instead ingests the state once and evaluates every question against it in parallel: a new architecture with a parallel sampler generates all answers in a single query, and because the possible outputs are defined in advance, every answer is a typed value [39, 40]. A yes/no question returns a probability, a choice among up to 255 options returns a probability per option, and a score against a rubric returns a distribution over levels [41]. Jev is trained with reinforcement learning for calibrated decisions (RLCD), so that higher confidence means higher accuracy [39, 40]. The vendor reports 70 to 500 milliseconds end to end, 40 to 200 times faster than frontier models of the same intelligence on decision-shaped queries, and about \$0.04 per million input tokens with output free [39]. A request may carry 64k tokens (32k of state), the model is not trained on customer requests, and enterprises get zero data retention [40].

We verified the interface live: the endpoint answered our four-axis questions in the documented shape (model string jev-1.13.0). An independent test on 60 hand-labelled cases measured 91.7% accuracy, a calibration error near 0.07 and a median latency of about 420 milliseconds, inside the half-second budget, on a diferent task: rating the risk of agent tool calls [42]. In the case study the classifier costs about \$179 a month for 1.3 million routed turns, and the router’s other overhead, keeping caches warm and building ledgers, is \$18k a month (Section 5.2).

Calibrated options beat a single label. The per-option probabilities let the policy compute the probability that the decision is right, that the tier the turn needs is at or below the cheaper model’s and that the turn does not depend on history, and hold the turn when either is low. A single top label cannot express that, and a general model’s stated confidence is not calibrated. Any classifier that returns calibrated per-option probabilities fast enough fits the same interface: small encoder classifiers of the kind vLLM’s semantic router uses (a 307M-parameter encoder with several heads [43]), purpose-trained routing models such as Arch-Router (a 1.5B-parameter model aligned to a user-defined domain-and-action taxonomy [44]), or an in-house encoder fine-tuned on the enterprise’s own tagged corpus. Hosted classification means every prompt leaves the company, and Jev’s weights and size are not published, nor is it ofered for self-hosting. That is why the interface is classifier-agnostic and a self-hosted encoder is the stated fallback for restricted contexts. Figure 4 walks through one real decision from the catalogue: what the classifier received, what it answered, and what the policy did with it.

![](images/5d9d5eaaf3d2cc0495cad910db49670a43ce5c01bc22e810549dbc9e41f22b87.jpg)  
Figure 4: One routing decision end to end: the one-line change inside an 805k-token session. The state and the six typed questions are what the classifier receives; the probabilities are its cached answer; the policy’s arithmetic and the resulting side lane are shown at the bottom.

## 4.7 Reasoning efort is a second dial

Model choice is one dial. The reasoning efort a model spends is another, and it does not always move cost the way the price sheet suggests. On bounded work, lower efort saves: in Anthropic’s published runs on knowledge-work benchmarks, medium efort matched the default’s accuracy at about 70 to 87% of its cost, and “below the model’s ceiling, the highest efort levels pay for depth the task never uses” [45]. On long-horizon, many-step work the direction can reverse. In ARC Prize’s evaluation of GPT-6 Astra on ARC-AGI-3, high efort cost \$40.7k against \$48.1k for medium while scoring 54.8% against 38.6%, and maximum efort was the cheapest setting of all at \$26.1k and 62.7%, “because Astra solves games in fewer actions, reducing the total number of model calls and tokens” [46]. An observational study of 90 agentic coding runs found that raising efort from high to extra-high lifted first-try perfect runs from 28% to 89% and cut corrective prompts about fivefold [47]. Task shape decides, which is exactly what the taxonomy’s complexity and output-shape axes describe, so efort is a natural second output of the same classifier and the next lever to route. Its metric is cost per completed task, not tokens per turn. One constraint carries over from the cache arithmetic: on most models each efort level has its own cache, so changing efort mid-conversation invalidates it like a model switch. Two exceptions exist on 21 September 2026: Fable 5.1 in Claude Code, and a per-message efort change, in beta, on Opus 5 and Fable 5.1 through a system message that leaves the prefix intact [8, 10, 48]. Efort routing therefore belongs at turn boundaries on models that keep the cache.

## 4.8 Eight components, from gateway to governance floors

The router consists of eight components. Our prototype implements components 2 to 6 and 8 as a decision engine and simulator, with the override rule of component 7. Components 1 and 7 connect the engine to live harnesses and their users.

1. An enforcement point: a gateway between every harness and its model endpoint for session-start decisions, cache-marker passthrough and measurement, plus harness hooks for in-session lanes and subagent launches.

2. A classifier adapter with the typed question set derived from the taxonomy (Section 4.6).

3. The bring-your-own taxonomy and policy: tiers, floors, thresholds and the switches for each execution mode (Appendix A).

4. Session state: context size, cache age, whether a task is open, and the builder of the project ledger that seeds side lanes.

5. The execution modes: session-start choice; a detached request (a one-of call to a cheaper model without the conversation); a ledger side lane, followed by one empty request that refreshes the home cache; subagent launch routing; and, behind an evaluation gate, a rebase that restarts the home conversation from the ledger at a task boundary.

6. Counterfactual accounting: what each decision cost against what pinning would have cost, aggregated by unit, never by person.

7. Transparency and override: the user sees which model answered and can pin.

8. Governance floors: domain and data-class floors, a stakes ceiling, no productivity axis.

## 4.9 The emulation runs authored tasks with measured behaviour

The emulation answers the question an enterprise faces before it has telemetry: what would routing save on our work, at our scale?

Tasks. A scenario catalogue of 46 conversations with 109 authored turns across eleven job types (engineering, data, product, marketing, sales, legal, finance, HR, consulting, IT operations, general), each turn tagged with ground-truth labels and an activity profile: tool steps, tool-result tokens, output tokens, pauses. Selected consulting scenarios are shaped by the authors’ own measured sessions: a slide deck built with a skill and a six-scout research run.

Measured behaviour. Each conversation is expanded into 200 Monte Carlo variants that carry behaviour measured in about 10,000 real Claude Code sessions in two public corpora [37, 38]. After every task comes a chain of corrections drawn from SWE-chat’s own pushback labels: 70% of tasks draw none and 3% five or more, and a correction does 5 to 13 tool calls, growing only slowly with the task. Pauses before follow-ups are drawn from SWE-chat (17% longer than five minutes), subagents launch on 4% of turns with tool work at TraceLab’s rate (SWE-chat’s 16% as a sensitivity), and every turn carries the correction chain a misroute of it would cost. Engineering turn lengths follow the Claude Code sessions in TraceLab, whose upper tail they match (51 and 121 requests at the 75th and 90th percentile, weighted by requests, against 56 and 121). Prompt tokens come from a real tokenizer.

Classifier and pricing. The live classifier labels every authored turn and every correction prompt from its text and structural session facts, and answers in a separate question whether a turn corrects the previous output (rule R10). The policy decides, and a cache-exact simulator prices pinned, naive-switch and routed execution from list prices and the vendors’ caching rules: every tool step re-sends the conversation, with cache expiry, compaction, subagents and the side-lane overheads. Misroutes are charged as the correction chains users actually run: a turn moved below the tier its tags require, or away from history it needed, is redone on the home model and followed by further corrections, and a session started on too weak a model pays its first turn twice. Everything runs ofline from the cached labels.

Table 3: Case-study summary (list prices of 21 September 2026; 10,000 seats; live classifier labels, cached).
<table><tr><td>Quantity</td><td>Value</td></tr><tr><td>Scenarios / turns / job types</td><td>46 / 109 / 11</td></tr><tr><td>Classifier accuracy: action / domain / complexity / history</td><td>74% / 80% / 63% / 82%</td></tr><tr><td>False downgrades among moved turns / held in place (of routable)</td><td>12.5% / 42%</td></tr><tr><td>As-is spend, 10,000 seats</td><td>$1,974k a month, $23.7M a year</td></tr><tr><td>Routed, today&#x27;s labels</td><td>$1,701k a month (−13.8%)</td></tr><tr><td>Routed, perfect labels</td><td>$1,558k a month (−21.1%)</td></tr></table>

Population. Results are scaled to a mixed population of 14 segments by function and usage intensity (power, regular and light engineers; heavy and light users in consulting and in data; one segment for each other function), each under its as-is model mix. “Power”, “regular” and “light” are usage intensities (active days a month and sessions a day), not skill levels: a power user is an engineer who works in the harness all day. A seat runs 30 sessions a month on average, from 6 for light engineers to 72 for heavy consulting and research users. The as-is Fable share follows one rule, the top model for sessions that contain expert-level work: 20% of power users’ sessions, 15% of regular users’ and 4% of light users’, with the rest on Opus 5. Other segments use Opus 5, and Section 5.4 lets non-engineers pick Fable 5.1. The emulated engineers spend \$40 of model tokens per active day, between the median (\$15) and the mean (\$58) of TraceLab’s heavy users (our pricing), and read 98.1% of their input from cache, against 95 to 98% in the two corpora.

## 5 Results

## 5.1 A 10,000-seat enterprise recovers 14 to 21%

As-is spend is \$1.97M a month, \$23.7M a year. Routed spend is \$1.70M with today’s labels and \$1.56M with perfect labels: a saving of 14 to 21%, \$3.3M to \$5.0M a year (Table 3). By segment, engineers save 13 to 14%, and the small non-engineering bills save 6 to 35%, most in marketing and IT operations (Figure 5).

Dollars scale with seats and power users. Multiplied across an enterprise, the percentages become budget lines (Figure 6). Dollars scale linearly with seats: the same population at 50,000 seats saves \$16.4M a year with today’s labels, and Figure 13 (Appendix D) gives the saving for other seat counts and spend levels. Power users move the dollars but not the rate: raising their share from 15% to 45% of engineers lifts the as-is bill from \$1.97M to \$2.91M a month and the annual saving at 50,000 seats from \$16.4M to \$24.1M, at 13.8% throughout. Agentic intensity moves the rate: the saving grows from 1.6% at half today’s tool steps and tokens per turn to 24% at double, because the crossover grows with every re-read of a long context, and agentic work is moving towards the long end [5]. In dollars, the router earns most where sessions are tool-heavy and caches stay warm.

## 5.2 Three quarters of the saving comes from the crossover

Figure 7 splits the saving: \$211k a month from choosing the cheapest capable model per session (the crossover of Section 4.3), \$64k from side lanes for small tasks, \$32k from routing subagents at launch, \$28k from right-sizing sessions at their start and \$7k on turns left in place, less \$51k of misroutes and their correction chains and \$18k of router overhead, almost all of it keeping the home conversation’s cache warm across side lanes and building ledgers. The classifier calls

![](images/73b63b42450cfd5e7a1a2fde2098da97d1d9d64e910a4af0c17e135c5728ecd2.jpg)  
Figure 5: Monthly model spend by population segment. Each bar ends at the as-is bill; the green tail is the saving with today’s labels, the tick the routed bill with perfect labels. Engineering carries 86% of the bill; non-engineering segments save 6 to 35% of small bills.

themselves cost about \$179 a month.

Fable 5.1 runs the long builds. Fable 5.1 takes 86% of routed model spend but starts only a fifth of the sessions (Figure 8): the long engineering and data builds that make up most of the bill. The sessions it takes from Opus 5 cost 14% less on it. No session in marketing, sales, legal, HR, finance, product or consulting runs on Fable 5.1: Sonnet 5 is cheaper on every price line, so the router weighs Fable 5.1 only against Opus 5, and only for tool work. The choice is made at session start, before a session’s length is known: 47% of the sessions moved to Fable 5.1 cost more on it, \$23k a month in all, against \$229k saved on the rest, \$206k net. Figure 7 shows \$211k because it books these sessions’ subagents and side lanes under their own mechanisms.

## 5.3 Holds cost money, not quality

The classifier gets the action right 74% of the time, the domain 80%, the complexity level 63% and history dependence 82%. Among the turns the router moved, 12.5% were false downgrades, moved below the tier their tags required. Corrections are a cost of their own: without them the as-is bill would be \$1.26M, so they make up more than a third of it. The classifier flagged every generated correction as one and the router moved none (rule R10). It also flagged 5% of new tasks, which R10 then holds.

A turn is routable when its ground-truth tags allow a cheaper model than the one it ran on. The router holds a routable turn when the classifier’s probability that the cheaper tier sufices, or that the turn does not depend on history, falls below the threshold: the turn stays on its current model and runs exactly as it would have without a router. A hold is not a failed decision, because nothing is lost but the saving. With today’s labels the router held 42% of routable turns. Of the \$144k a month between today’s labels and perfect labels, two thirds (\$93k) are holds, mostly sessions that could have started on a cheaper model, and one third (\$51k) the correction chains of misroutes. It is a deliberate asymmetry: a wrong move costs quality and rework, a hold costs only money, so the policy is tuned to move only when sure. The 42% comes from the classifier’s complexity probabilities, which split between neighbouring levels, most often routine against complex, exactly where the taxonomy’s level definitions are least crisp (63% complexity accuracy, with the errors one level of). The remedy is therefore not a diferent classifier but sharper level definitions and examples in the taxonomy, tagged on the enterprise’s real prompts, and a threshold set on that data. Section 5.4 shows the threshold trade-of.

![](images/803cddeecc855460c40254aec02dad0d3ef48bfb671bf7584ade92024d41340c.jpg)

![](images/7477d63d67668df8062a7da6b8b96768990abed3c8263328fe7cab8cb46722a1.jpg)  
Figure 6: How the saving scales. Left: annual saving against seats for three shares of power users among engineers, at the same saving rate on every line. Right: saving against agentic intensity, tool steps and tool-result tokens scaled together.

![](images/7015cf2cfff3bea6b59d43be684e6181bc00b67125c15d05a341542dc79eefc9.jpg)  
Figure 7: Where the saving comes from (monthly, today’s labels). The vertical axis is cut below the lowest step; the torn columns continue to zero.

## 5.4 Pauses, the price sheet and tool use move the saving most

Figure 9 and Table 4 vary one assumption at a time on the same labels and the same random draws. Four results matter for an enterprise reader. First, pauses. At the five-minute cache lifetime an API key gets, a user who returns after a pause finds the cache expired and pays for the whole conversation again, at the top model’s higher write price. Without such pauses the saving would be 24%, and with TraceLab’s more pause-heavy users it is 7%. A one-hour cache lifetime lowers the as-is bill and lifts the saving to 17%, 19% against today’s bill: the cheapest default to change. Second, the price sheet. Under a uniform 0.1× cache-read regime with the same write premium, OpenAI’s stated rule and Anthropic’s sheet without the Fable discount, the as-is bill is 17% higher (\$2.30M a month) and the saving falls to 5%: the crossover lever disappears, and misroutes eat most of what side lanes and subagent routing save. Without a Fable 5.1 licence the saving falls to 2.3%, because the crossover needs the top model. Third, agentic intensity (Section 5.1). Fourth, the classifier threshold (Figure 12). Lowering it from 0.8 to 0.5 moves 46% more turns, a fifth of them false downgrades whose correction chains cost more than the moves save, and the saving falls to 10%. Raising it to 0.9 saves 14.2% while holding half the routable turns. Moving more turns is not better: on this corpus the best tested threshold is 0.9, just ahead of 0.8. We keep 0.8 as the base, because choosing on this corpus would tune on the test set.

![](images/895d7d9c5532c69b64b0e10e52272175270e26f45149516de56c2dc2d836131e.jpg)  
Figure 8: Who runs on which model after routing (monthly, today’s labels). Left: each function’s sessions by the model a session starts on. Right: its routed model spend by the model that ran, with subagents, side lanes and redone misroutes, without router overhead. Totals at the end of each bar. As-is, every function outside engineering runs on Opus 5.

The behaviour sensitivities are smaller. Without corrections the bill is 36% lower and the saving 14.7%. Subagents at SWE-chat’s rate lift the saving to 15.9%, and without launch routing it falls to 11.7%. Charging a misroute as a single redo would flatter the router by a point, and the as-is Fable share moves the saving between 15.3% (3% of engineering sessions, as observed in public data) and 11.4% (Fable 5.1 for all complex or expert work). Outside engineering the base case keeps everyone on Opus 5, although in the authors’ experience of enterprise rollouts many non-technical users pick the top model even for easy work. If non-engineers pick Fable 5.1 for half their sessions, the as-is bill rises 2.3% and the saving to 15.3%, and for all of them to 16.6%. Their sessions are short chats, so the money stays in engineering’s long sessions.

## 5.5 Strong models of other vendors cost 0.12 to 0.80 times Opus 5

A router behind Claude Code chooses only among Claude models, because the harness does not support other vendors’ models behind a gateway [18]. A harness that accepts the enterprise’s own endpoint (Table 8) can call them, and we priced that option on real sessions: every Claude Code session in TraceLab and SWE-chat, token for token, at each route’s list prices of 23 September 2026 [13, 14, 19, 49–52]. The cache-read price decides again (Figure 10). On routes that discount cache hits, strong models of other vendors cost 0.12 to 0.80 times Opus 5 on the same sessions: DeepSeek V4 Pro 0.12 to 0.14 at peak-hour prices, Kimi K2.6 0.25 to 0.27, Mistral Medium 3.5 0.28 to 0.29, GPT-5.3-Codex 0.35, Gemini 3.1 Pro 0.39 to 0.64 and GPT-5.6 Sol, OpenAI’s counterpart to Opus 5, 0.80. DeepSeek, Kimi and GLM models on Amazon Bedrock, which lists no cache price for them, cost 0.70 to 0.90 times Opus 5. At the top tier OpenAI does not undercut Anthropic: GPT-6 Astra, its counterpart to Fable 5.1, costs 1.70 to 2.12 times Fable 5.1 on the same sessions, because its cache reads cost four times as much. xAI’s Grok 4.7 costs 0.79 to 1.19 times Opus 5. On the emulated bill, other vendors that take only the light work the router already moves down add at most 2.3 points to the saving. Strong models of other vendors that take complex work at session start add 10 to 55 points to the 14% on routes with cache pricing, and 3 to 24 without, at equal quality and token use.

![](images/da2be2a3c6dce8054b94279cfe75d810b06070e02ed479f47e276815c4386566.jpg)  
Figure 9: One-at-a-time sensitivities of the population saving, one row per driver with its low and high case; italic case names mark the base (base 13.8%; same labels and random draws). Orange bars run below the base, green bars above it. Table 4 lists every case.

## 6 Recommendations

Two horizons, two decisions. The first can start this quarter, from within: put a router and a control plane behind the harnesses employees already use, between the harness and the model endpoint. The second is a product decision for the next one to two years: whether to own the harness itself.

## 6.1 Now: put a router behind today’s harnesses

Own the control plane. Put a gateway between every harness and its model endpoint (the harness’s base-URL setting) and verify that it forwards cache markers and beta headers unchanged, because a gateway that strips them bills the whole conversation as uncached input on every turn [8]. Route at session start from the first prompt. Set a fleet default model for subagents today and route them per launch tomorrow. Export telemetry (token usage by type, session and model) and report aggregates only. Run a two-hour tagging workshop per function to build the routing taxonomy from that function’s own prompts. The tags double as the policy review.

Ship one short bundle. The router decides which model runs, and the bundle, shipped with the harness configuration, decides what the model can do and what every request carries. Ten

Table 4: Sensitivities (population, monthly, same labels).
<table><tr><td>Case (population, same labels)</td><td>As-is month</td><td>Routed month</td><td>Saving</td></tr><tr><td>Base case</td><td>$1.97M</td><td>$1.70M</td><td>13.8%</td></tr><tr><td>Ledger rebase at task boundaries</td><td>$1.97M</td><td>$1.70M</td><td>13.9%</td></tr><tr><td>One-hour cache lifetime (as-is and routed)</td><td>$1.93M</td><td>$1.60M</td><td>17.1%</td></tr><tr><td>No Fable 5.1 licence (as-is on Opus 5, router capped at Opus 5)</td><td>$2.01M</td><td>$1.97M</td><td>2.3%</td></tr><tr><td>Everyone as-is on Opus 5 (Fable 5.1 licensed for the router)</td><td>$2.01M</td><td>$1.70M</td><td>15.6%</td></tr><tr><td>Confidence threshold 0.7 (base 0.8)</td><td>$1.97M</td><td>$1.78M</td><td>9.8%</td></tr><tr><td>Confidence threshold 0.9 (base 0.8)</td><td>$1.97M</td><td>$1.69M</td><td>14.2%</td></tr><tr><td>Uniform 0.1× cache reads, writes 1.25× / 2×</td><td>$2.30M</td><td>$2.18M</td><td>5.2%</td></tr><tr><td>Uniform 0.1× cache reads, no write premium</td><td>$2.21M</td><td>$2.10M</td><td>4.9%</td></tr><tr><td>Agentic intensity ×0.5</td><td>$1.05M</td><td>$1.03M</td><td>1.6%</td></tr><tr><td>Agentic intensity ×2</td><td>$4.45M</td><td>$3.38M</td><td>24.0%</td></tr><tr><td>Every answer accepted (no correction turns)</td><td>$1.26M</td><td>$1.08M</td><td>14.7%</td></tr><tr><td>Pauses as authored (almost none over five minutes)</td><td>$1.76M</td><td>$1.34M</td><td>24.0%</td></tr><tr><td>Pauses as in TraceLab (27% over five minutes)</td><td>$2.12M</td><td>$1.96M</td><td>7.4%</td></tr><tr><td>No subagents beyond the authored ones</td><td>$1.91M</td><td>$1.67M</td><td>12.6%</td></tr><tr><td>Subagents at the SWE-chat rate (16% of tool turns)</td><td>$2.10M</td><td>$1.76M</td><td>15.9%</td></tr><tr><td>Misroutes cost a single redo</td><td>$1.97M</td><td>$1.68M</td><td>14.7%</td></tr><tr><td>Subagents inherit the parent&#x27;s model (no launch routing)</td><td>$1.97M</td><td>$1.74M</td><td>11.7%</td></tr><tr><td>As-is Fable 5.1 share as observed (3% of engineering sessions)</td><td>$2.01M</td><td>$1.70M</td><td>15.3%</td></tr><tr><td>As-is Fable 5.1 for any complex or expert work (45 / 35 / 14%)</td><td>$1.92M</td><td>$1.70M</td><td>11.4%</td></tr><tr><td>Non-engineers pick Fable 5.1 for half their sessions</td><td>$2.02M</td><td>$1.71M</td><td>15.3%</td></tr><tr><td>Non-engineers pick Fable 5.1 for all their sessions</td><td>$2.06M</td><td>$1.72M</td><td>16.6%</td></tr></table>

items, in the order they pay of:

1. Model policy: the session-start rule, the subagent default, the domain and data floors, the user override.

2. Skills as procedures: how this enterprise deploys, reviews, names things and handles data, loaded on demand (the open Agent Skills format keeps only a description in context until a skill is used [25]).

3. Connectors: the MCP servers a role needs, scoped by role, no more.

4. Hooks: tests and review before a task may complete, cache-marker checks on the gateway, the subagent launch rule.

5. Permissions and data classes: what tools may run unattended, which data never leaves for which model.

6. The routing taxonomy tagged by the function that uses it, with its thresholds.

7. The project ledger format that seeds side lanes and subagents instead of the transcript.

8. Telemetry export and aggregate reporting, wired to the counterfactual accounting.

9. One short instruction file: procedures and constraints, no repository description (Section 3.4).

10. Context defaults by role: when old reasoning and tool results are cleared, when to compact, when to start a new conversation (below).

Weigh the bundle. Everything in it that is loaded on every request is re-read on every tool step. Table 5 prices the always-loaded prefix for a ten-turn session of eighteen tool steps. The 55k row is real: one of the authors’ own Claude Code sessions, read from the harness’s context view, carried 5.9k tokens of system prompt, 15.8k of tool schemas, 15.4k of MCP tool schemas, 9.8k of skill descriptions, 6.4k describing 72 custom agents from a community plugin bundle, and 2.1k of memory before a single message. That configuration costs three to four times the lean default per session on every model, and it would cost more than ten times the default had the harness not deferred a further 104k tokens of tool schemas until needed. Two lessons follow. Harnesses already expose the accounting an enterprise needs to audit its bundle, per category and per tool: use it. And deferred loading, of tool schemas, skills and agent descriptions, is a cost lever in its own right, as the 104k deferred tokens show: cap what a role declares at start, load the rest when it is used, and measure.

![](images/cf41a00083cf6b356df75ac691d64633f3e4700e276b31e200c3f03c8d8f1906.jpg)  
Figure 10: The price of lock-in. Left: every Claude Code session in TraceLab and SWE-chat repriced token for token at each route’s list prices (23 September 2026), as a multiple of the same sessions on Opus 5; each bar spans the two corpora. Anthropic’s models keep their colours from Figure 8; other vendors routes are grey where they price cache reads and orange where they list no cache price. Right: the saving on the emulated bill when other vendors take only light work, or when strong models of other vendors take complex work at session start, with and without cache pricing. Ceilings: equal quality and the same token use on every model.

Table 5: Cost per session of the always-loaded prefix (ten turns of eighteen tool steps, warm cache; list prices of 21 September 2026). Every token in the prefix is re-read on every request; the 55k row is a measured power-user configuration.
<table><tr><td>Always-loaded prefix per request</td><td>Fable 5.1</td><td>Opus 5</td><td>Sonnet 5</td></tr><tr><td>15k: lean default (system prompt and core tools)</td><td>$0.90</td><td>$1.51</td><td>$0.60</td></tr><tr><td>25k: default plus a 10k instruction and skills bundle</td><td>$1.49</td><td>$2.52</td><td>$1.01</td></tr><tr><td>55k: observed power-user configuration with plugins</td><td>$3.29</td><td>$5.54</td><td>$2.22</td></tr><tr><td>159k: the same if deferred tool schemas were loaded eagerly</td><td>$9.50</td><td>$16.02</td><td>$6.41</td></tr></table>

Set context defaults by role. Harnesses apply one context strategy to everyone. Opus 5, Sonnet 5 and Fable 5.1 keep every earlier turn’s reasoning in context by default [53], and Claude Code compacts every conversation at one threshold [9]. An engineer in the middle of a task may need that history, while a marketer revising a draft rarely needs the reasoning behind the previous version. Clearing old reasoning and tool results rewrites the cache from the cleared point on, so it follows the payback rule of Section 4.2: on a warm cache it rarely pays, on an expired one it is nearly free, because the rewrite is due anyway. Expired caches are common. In two public corpora of Claude Code sessions, 17 to 27% of user turns came after a pause of more than five minutes and 3 to 4% after more than an hour (our analysis of [37, 38]), and Claude Code keeps the cache for an hour on a subscription but for five minutes on API keys and cloud providers [8], which is how an enterprise gateway connects. A one-hour cache is therefore the

Table 6: Returning to a conversation whose cache has expired: 10,000 returns at 300k tokens of context on Opus 5 (list prices of 21 September 2026; 5-minute cache). Pruning clears old reasoning and tool results without a model call; a summary is written by the named model, which reads the full history uncached. The 30k is 15k of system prompt and tools plus a 15k-token summary. “Each later request” is the cache read of the history on every request after the return.
<table><tr><td>10,000 returns at 300k context</td><td>On the return</td><td>Each later request</td><td>Return vs re-send</td></tr><tr><td>Re-send the full history (default)</td><td>$18,750</td><td>$1,500</td><td>+0%</td></tr><tr><td>Prune to 30k first (clear old reasoning and tool results)</td><td>$1,875</td><td>$150</td><td>-90%</td></tr><tr><td>Compact to 30k, summary by Opus 5</td><td>$20,625</td><td>$150</td><td>+10%</td></tr><tr><td>Compact to 30k, summary by Sonnet 5</td><td>$9,375</td><td>$150</td><td>-50%</td></tr></table>

## first default to set.

For the returns that stay cold, Table 6 prices the options. Ten thousand returns to a 300k-token history on Opus 5 cost \$18,750 just to re-send it, and pruning it to 30k first costs \$1,875. A summary by the same model costs more than the re-send on the return itself, because the summariser must read the whole history once, and pays back after two further requests. A summary by Sonnet 5 halves the cost of the return. A standard per role follows. Chat-shaped work: prune or summarise cheaply on every cold return, and start a new conversation per deliverable. Engineering: keep reasoning within a task, clear stale tool results in large batches, compact at task boundaries. Test answer quality on a pilot group before a standard rolls out, then set it centrally: an organisation’s managed settings take precedence over every user’s [54]. In Claude Code only the compaction threshold is such a setting, while clearing old reasoning and tool results needs the API’s context-editing parameters, set by a gateway or an owned harness.

## 6.2 Later: decide on purpose whether to own the harness

Table 7 orders the options by what they buy and what they cost. Rungs one and two are this quarter’s work and deliver the saving of Section 5.1 on the harnesses people already use. Rung three is a diferent decision: an internal harness on a vendor SDK or an open-source base, with pluggable models, native lanes and one bundle for everyone. It is right for enterprises with regulated workloads that no vendor’s retention terms fit, with their own models to serve, or with enough volume that a product team costs less than the dependency. It is wrong as a default, because vendors ship weekly and an open-source base moves the dependency to its maintainers. Decide it on purpose, after rung two has produced the numbers.

Keep the choice of models open. The larger saving of Section 5.5 turns on one decision, which models an enterprise accepts for its complex work, and only a harness that accepts the enterprise’s own endpoint lets it take that decision. Most harnesses in Table 8 accept one, and an owned harness always does.

Ask vendors seven questions. Does the harness forward cache markers through a gateway? Can it call an endpoint we operate? Can we set and enforce the model of subagents? Does it export token telemetry by type? What is the data retention of each model it ofers? Can we pin a harness version and read its prompts? Which instruction, skill and connector formats does it read, and are they the open ones?

Measure cost per verified task. Report it, and its spread across teams and harness configurations, in aggregate. Not tokens per user.

Table 7: The sovereignty ladder.
<table><tr><td>Rung</td><td>What it buys</td><td>What it costs</td></tr><tr><td>0. Vendor defaults</td><td>Nothing to run The bundle above, model defaults,</td><td>Every risk in Section 3; the highest bill Configuration per harness; no</td></tr><tr><td>1. Configured</td><td>subagent default, permissions</td><td>measurement</td></tr><tr><td>2. Controlled</td><td>Gateway, session-start routing, side lanes via hooks, telemetry, policy floors, taxonomy; the 14 to 21% of</td><td>A small platform team; classifier hosting; works-council and legal review of the policy</td></tr><tr><td>3. Owned</td><td>An internal harness on a vendor SDK or an open-source base with pluggable models, native lanes, one bundle for everyone; other vendors&#x27; models (Figure 10)</td><td>A product team; keeping pace with weekly vendor releases; the dependency moves to maintainers</td></tr></table>

## 7 Related work

Routing and cascades. We price the vendor-side prompt cache of an agentic turn and derive when a switch pays, where the routing literature prices single requests. RouteLLM learns win-prediction routers from preference data [33], FrugalGPT cascades models by confidence [34], and recent benchmarks and analyses score routers on independent queries [35, 55, 56], the largest on over 400,000 instances [35]. Multi-turn formulations treat routing as a budgeted sequential decision [57], and agent-serving simulators test policies that keep a program’s successive turns on the same engine [58]. Neither prices a vendor’s prompt cache: the first charges abstract per-token costs, the second works inside the serving engine.

Routers in harnesses. Our payback rule explains what shipped routers do by rule of thumb. Copilot’s auto mode routes along cache boundaries and states that mid-session switching raised cost [36]. OpenRouter’s sticky routing activates only when a provider’s cache reads are cheaper than fresh input [59], and the open-source claude-code-router selects by static keys and has met the cache-prefix problem in the wild [60]. The concept of routing is not new, and independent work applied Jev to it within days of the model’s release: those routers fall back to a safe tier when the classifier is unsure, and one refuses to downgrade long conversations to protect the cache [61, 62]. OpenAI’s GPT-5 router [63] met a user backlash over lost control [64], which is why our design shows the model and allows a pin.

Caching for agents. The nearest academic neighbour evaluates caching strategies for longhorizon agentic tasks across vendors without model switching [65], and Codex’s exact-prefix cache has been measured under model and efort changes [66]. We take the cache as given and ask which model should own it.

Developer productivity and skill. Our skill argument rests on Anthropic’s session-level expertise analysis [26, 28], the METR trial [27, 67] and DORA [29]. The context-file ablations [30, 31] bound what static files do for well-specified tasks and leave the unskilled-user case open (Section 3.4). The dificulty distribution of coding-agent tasks [5] bounds how much work can move to cheaper models at all.

Governance. Employee-monitoring constraints under the EU AI Act and co-determination law [32] shaped the absence of a productivity axis.

## 8 Limitations

The results rest on an emulation, two public corpora and one vendor’s list prices. Every input below can be replaced with an enterprise’s own, and the bring-your-own design exists for that.

• Synthetic scenarios. The 46 conversations and 109 turns are synthetic. The authors wrote them from their experience of leading teams and watching how people work with assistants and agents, and shaped selected scenarios on real sessions of their own. Measured behaviour perturbs them but adds no new tasks, and job types without a public corpus borrow the developers’ behaviour on turns without tool work. Their ground-truth tags come from the authors alone, and future work should rest on real, independently tagged prompts wherever an enterprise can provide them.

• A taxonomy still in development. The routing taxonomy is not final. Its complexity levels are its least crisp part and cause most holds, which is why every result is a band from today’s labels to perfect labels. It has no ablations and no inter-rater agreement yet. The classifier’s accuracy on a given team is known only once that team’s prompts are tagged by two independent raters with their agreement reported.

• Session shape. The engineering scenarios under-represent the lightest turns: a quarter of real requests sit in turns of fewer than seven requests, against 3.8% of ours. They also re-read more of their input from cache (98.1%, against 95.2% in TraceLab and 97.6% in SWE-chat), which favours the crossover. The share of long sessions that are cheaper on Fable 5.1 (51 to 80%) rests on the 102 TraceLab and 30 SWE-chat sessions that cost \$100 or more. Repricing real sessions bounds the crossover at 6.0% (TraceLab) to 12.6% (SWE-chat) of an all-Opus bill, against 12.2% of the Opus spend in the emulation. Sessions shaped like TraceLab’s would bring the 14% to about 8%, in line with its pause sensitivity (7.4%).

• Usage intensity and the dollars. The emulated engineers spend \$40 per active day: between the median (\$15) and the mean (\$58) of TraceLab’s heavy users, and three times the vendor’s reported average of about \$13, with 90% of users below \$30 [68]. Volume scales the dollars and leaves each segment’s percentage unchanged: at \$13 per engineer-day the bill is \$10.0M a year and the saving \$1.3M to \$2.1M. The blended rate moves to 13.3 to 21.1% because engineering’s share of the bill falls. Shorter sessions lower the percentage as well (Figure 6).

• Scale and the cost of building a router. The saving grows with seats (Figure 6), and the cost of building and running a router does not. For a small organisation the saving may not pay for an engineering team to build one. Standard routers that need no engineering of their own are likely to appear and would remove that threshold.

• Model choice outside engineering. Non-engineers start on Opus 5 in the base case. That many pick Fable 5.1 even for easy work is the authors’ experience of enterprise rollouts, not a measurement, and it enters only as a sensitivity (15.3% if they pick it for half their sessions, 16.6% for all).

• Two corpora, developers only. About 10,000 Claude Code sessions from SWE-chat and TraceLab [37, 38]. They disagree on subagent use by a factor of four per tool turn and on pauses by a factor of 1.6.

• List prices, no contract discounts. Prices are the list prices of one vendor family on 21 September 2026. Enterprise contracts are not deducted, so the dollar totals may be lower. A discount that applies to every model lowers every dollar figure by its share and leaves the percentages unchanged, while discounts that difer by model move the percentages as well.

• Prices move fast. Providers change prices and price structures often and at short notice, and every change moves the dollars and can move the percentages. Under a uniform 0.1× cache-read rule the crossover disappears and the saving falls to 5% (Table 4). What transfers is the method: price every candidate on the enterprise’s own sessions at the day’s prices.

• Other vendors’ models. The lock-in ceilings (Figure 10) reprice real sessions at list prices.

They assume equal quality and the same token use on every model, no cross-vendor cache events inside a session, and data terms that allow the route. DeepSeek is priced at peak hours, and its of-peak prices are half. OpenAI’s models are priced at their short-context rates, because its page does not state where the long-context rates begin.

• The perfect-label ceiling. Perfect labels know each turn’s tier and treat capability as a step: a model at the needed tier always succeeds, one below always fails. Real dificulty is only partly visible in the prompt and real capability is a probability, so 21% is a ceiling rather than a target.

• Classifier figures. Speed and calibration are the vendor’s figures, apart from one independent test on 60 hand-labelled cases [42]. A hosted classifier sends every prompt out of the company, which the self-hosted fallback avoids.

• Prototype scope. The decision engine, the policy and the simulator are implemented. The enforcement point (gateway and hooks) and the user’s view of the routing are specified but not built, so the case study runs every decision in the simulator, not behind a live harness.

• Static users. Behaviour does not react to the router: nobody changes how they work because a router exists.

• Levers outside the emulation. Routing of reasoning efort (Section 4.7), the quality efect of clearing old reasoning, for which Claude Code has no setting, and whether procedures and skills lift unskilled users on under-specified requests. The context-file ablations leave the last open, because they have no human in the loop and use well-specified tasks (Section 3.4).

## 9 Conclusion

An enterprise’s AI bill is set in two places: the model’s price sheet and the harness. The price sheet lists the rates, and the harness picks which rate applies to every request and how many tokens are bought at it. It also decides governance and dependence. Enterprises that negotiate only the price sheet leave the harness, and with it much of the bill, to their vendor’s defaults.

Routing is the most direct lever the harness ofers, and inside a harness it obeys the cache. We gave the payback rule for mid-task switches, showed that for long sessions the highest-priced model costs less than the next tier and confirmed it on real sessions, and built a router that acts only where no running conversation rebuilds its cache, with Jev labelling every prompt against a bring-your-own taxonomy in under half a second. At 10,000 seats it recovers 14 to 21% of model spend, \$3.3M to \$5.0M a year. Pauses decide as much as the classifier, because an expired cache turns every return into a full rewrite, which makes the cache lifetime part of the lever. Corrections make up more than a third of the bill and make every misroute expensive, so the router moves work only when it is sure. The harness also decides which vendors are in reach: on the same real sessions, strong models of other vendors cost 0.12 to 0.80 times Opus 5. Enterprises can own the control plane around their harnesses from within, starting this quarter, and decide about owning the harness on purpose rather than by default.

Reproducibility. Appendix A specifies the taxonomy and policy and Appendix B the cost model and its constants, so that any enterprise can rebuild the method with its own inputs. The authors’ simulator, scenario catalogue and cached classifier labels are not released.

Disclaimer. The spend, prices, populations and savings in this paper come from public list prices, public session corpora, the authors’ own sessions and an emulated enterprise. They do not represent Accenture’s spend, prices, contract terms or internal figures.

## References

[1] Anthropic. Accenture and Anthropic launch multi-year partnership to move enterprises from AI pilots to production, December 2025. URL https://www.anthropic.com/news/ anthropic-accenture-partnership. 9 December 2025. Accessed 24 September 2026.

[2] Anthropic. Cognizant will make Claude available to 350,000 employees, accelerating enterprise AI adoption and internal transformation, November 2025. URL https://www. anthropic.com/news/cognizant-partnership. 4 November 2025. Accessed 24 September 2026.

[3] Anthropic. PwC is deploying Claude to build technology, execute deals, and reinvent enterprise functions for clients, May 2026. URL https://www.anthropic.com/news/ pwc-expanded-partnership. 14 May 2026. Accessed 24 September 2026.

[4] GitHub. Octoverse 2025: A new developer joins GitHub every second as AI leads TypeScript to #1, October 2025. URL https://octoverse.github.com/. Accessed 23 September 2026.

[5] Drew Johnston, David Holtz, Alex Martin Richmond, Christopher Ong, Prasanna Tambe, and Aaron Chatterji. The Shift to Agentic AI: Evidence from Codex. arXiv:2606.26959, 2026. URL https://arxiv.org/abs/2606.26959.

[6] Stack Overflow. 2025 Stack Overflow Developer Survey, 2025. URL https://survey. stackoverflow.co/2025/ai. Accessed 23 September 2026.

[7] Anthropic. Context windows, 2026. URL https://platform.claude.com/docs/en/ build-with-claude/context-windows. Accessed 24 September 2026.

[8] Anthropic. How Claude Code uses prompt caching, 2026. URL https://code.claude. com/docs/en/prompt-caching. Accessed 21 September 2026.

[9] Anthropic. Claude Code: Environment variables, 2026. URL https://code.claude.com/ docs/en/env-vars. Accessed 22 September 2026.

[10] Anthropic. Prompt caching, 2026. URL https://platform.claude.com/docs/en/ build-with-claude/prompt-caching. Accessed 21 September 2026.

[11] Anthropic. Claude API pricing, 2026. URL https://platform.claude.com/docs/en/ about-claude/pricing. Accessed 21 September 2026.

[12] OpenAI. Prompt caching guide, 2026. URL https://developers.openai.com/api/docs/ guides/prompt-caching. Accessed 21 September 2026.

[13] OpenAI. OpenAI API pricing, 2026. URL https://developers.openai.com/api/docs/ pricing. Accessed 21 and 23 September 2026.

[14] Google. Gemini API pricing and context caching, 2026. URL https://ai.google.dev/ gemini-api/docs/pricing. Accessed 21 and 23 September 2026.

[15] Anthropic. Claude Code: Hooks reference, 2026. URL https://code.claude.com/docs/ en/hooks. Accessed 21 September 2026.

[16] Anthropic. Claude Code: Subagents, 2026. URL https://code.claude.com/docs/en/ sub-agents. Accessed 21 September 2026.

[17] Anthropic. Claude Code repository licence and npm distribution, 2026. URL https:// github.com/anthropics/claude-code/blob/main/LICENSE.md. Accessed 21 September 2026.

[18] Anthropic. Claude Code: LLM gateway configuration, 2026. URL https://code.claude. com/docs/en/llm-gateway. Accessed 21 September 2026.

[19] Amazon Web Services. Amazon Bedrock pricing, 2026. URL https://aws.amazon.com/ bedrock/pricing/. Prompt caching: https://docs.aws.amazon.com/bedrock/latest/ userguide/prompt-caching.html. Accessed 23 September 2026.

[20] Cursor. Clarifying our pricing, July 2025. URL https://cursor.com/blog/ june-2025-pricing. Accessed 23 September 2026.

[21] Anthropic. We’re rolling out new weekly rate limits for Claude Pro and Max in late August. Post on X (@AnthropicAI), July 2025. URL https://x.com/AnthropicAI/ status/1949898502688903593. Accessed 23 September 2026.

[22] GitHub. GitHub Copilot is moving to usage-based billing, April 2026. URL https://github.blog/news-insights/company-news/ github-copilot-is-moving-to-usage-based-billing/. Accessed 23 September 2026.

[23] JetBrains. State of Developer Ecosystem Report 2025, October 2025. URL https:// devecosystem-2025.jetbrains.com/. Accessed 23 September 2026.

[24] Agentic AI Foundation. AGENTS.md: a simple, open format for guiding coding agents, 2025. URL https://agents.md. Accessed 21 September 2026.

[25] Anthropic. Agent Skills: an open standard, December 2025. URL https://agentskills.io. Accessed 21 September 2026.

[26] Anthropic. How Claude Code is used in practice, 2026. URL https://www.anthropic. com/research/claude-code-expertise. Accessed 21 September 2026.

[27] Joel Becker, Nate Rush, Elizabeth Barnes, and David Rein. Measuring the Impact of Early-2025 AI on Experienced Open-Source Developer Productivity. arXiv:2507.09089, 2025. URL https://arxiv.org/abs/2507.09089.

[28] Anthropic. Anthropic Economic Index, March 2026 report, 2026. URL https://www. anthropic.com/research/economic-index-march-2026-report. Accessed 21 September 2026.

[29] DORA. State of AI-assisted Software Development 2025, September 2025. URL https: //dora.dev/dora-report-2025/. Quoted from the report’s announcement on the Google Cloud blog, 23 September 2025. Accessed 23 September 2026.

[30] Thibaud Gloaguen, Niels Mündler, Mark Müller, Veselin Raychev, and Martin Vechev. Evaluating AGENTS.md: Are Repository-Level Context Files Helpful for Coding Agents? arXiv:2602.11988, 2026. URL https://arxiv.org/abs/2602.11988. MemAgents workshop at ICLR 2026.

[31] Prakhar Khatri. Do Context Files Help Coding Agents? A Two-Agent Ablation Study on Real Repositories. arXiv:2607.27250, 2026. URL https://arxiv.org/abs/2607.27250.

[32] Compound. AI and employee monitoring: the EU AI Act and German works councils, 2026. URL https://compound.law/en-DE/compliance/ai-employee-monitoring/. Accessed 21 September 2026.

[33] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M Waleed Kadous, and Ion Stoica. RouteLLM: Learning to Route LLMs with Preference Data. arXiv:2406.18665, 2024. URL https://arxiv.org/abs/2406.18665.

[34] Lingjiao Chen, Matei Zaharia, and James Zou. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. arXiv:2305.05176, 2023. URL https://arxiv.org/abs/2305.05176.

[35] Hao Li, Yiqun Zhang, Zhaoyan Guo, Chenxu Wang, Shengji Tang, Qiaosheng Zhang, Yang Chen, Biqing Qi, Peng Ye, Lei Bai, Zhen Wang, and Shuyue Hu. LLMRouterBench: A Massive Benchmark and Unified Framework for LLM Routing. arXiv:2601.07206, 2026. URL https://arxiv.org/abs/2601.07206.

[36] GitHub. GitHub Copilot: auto model selection, 2026. URL https://docs.github.com/ en/copilot/concepts/models/auto-model-selection. Accessed 21 September 2026.

[37] Kan Zhu, Mathew Jacob, Chenxi Ma, Yi Pan, Stephanie Wang, Arvind Krishnamurthy, and Baris Kasikci. TraceLab: Characterizing Coding Agent Workloads for LLM Serving. arXiv:2606.30560, 2026. URL https://arxiv.org/abs/2606.30560.

[38] Joachim Baumann, Vishakh Padmakumar, Xiang Li, John Yang, Diyi Yang, and Sanmi Koyejo. SWE-chat: Coding Agent Interactions From Real Users in the Wild. arXiv:2604.20779, 2026. URL https://arxiv.org/abs/2604.20779.

[39] TypeSafe AI. Introducing System One Models and Jev, September 2026. URL https:// typesafe.ai/blog/introducing-system-one-models-and-jev. Accessed 21 September 2026.

[40] TypeSafe AI. TypeSafe AI API reference and model card for Jev, 2026. URL https: //docs.typesafe.ai/models. Accessed 22 September 2026.

[41] Cloudflare. Jev on Cloudflare Workers AI: model page and request schema, 2026. URL https://developers.cloudflare.com/ai/models/typesafe/jev/. Accessed 21 September 2026.

[42] Mike Moore. I Benchmarked Jev on Agent Tool-Call Risk. Calibration Held. DEV Community (@webofmike), September 2026. URL https://dev.to/webofmike/ i-benchmarked-jev-on-agent-tool-call-risk-calibration-held-49i3. Accessed 23 September 2026.

[43] vLLM project. vLLM Semantic Router, 2026. URL https://github.com/vllm-project/ semantic-router. Accessed 21 September 2026.

[44] Co Tran, Salman Paracha, Adil Hafeez, and Shuguang Chen. Arch-Router: Aligning LLM Routing with Human Preferences. arXiv:2506.16655, 2025. URL https://arxiv.org/abs/ 2506.16655.

[45] Anthropic. Optimizing for cost and intelligence, 2026. URL https://platform.claude. com/docs/en/about-claude/models/optimizing-for-cost-and-intelligence. Accessed 22 September 2026.

[46] Greg Kamradt and ARC Prize. OpenAI’s GPT-6 Astra on ARC-AGI-3, September 2026. URL https://arcprize.org/blog/astra. Accessed 22 September 2026.

[47] Achint Mehta. Reasoning efort, not tool access, buys first-try reliability in agentic code generation: an observational study. arXiv:2607.02436, 2026. URL https://arxiv.org/ abs/2607.02436.

[48] Anthropic. Efort, 2026. URL https://platform.claude.com/docs/en/ build-with-claude/effort. Accessed 23 September 2026.

[49] Mistral AI. Mistral AI API pricing and prompt caching, 2026. URL https://mistral.ai/ pricing/api/. Accessed 23 September 2026.

[50] SpaceXAI. xAI API pricing, 2026. URL https://docs.x.ai/developers/pricing. Accessed 23 September 2026.

[51] DeepSeek. DeepSeek API models and pricing, 2026. URL https://api-docs.deepseek. com/quick\_start/pricing. Accessed 23 September 2026.

[52] Moonshot AI. Kimi API pricing, 2026. URL https://platform.kimi.ai/docs/pricing/ chat. Accessed 23 September 2026.

[53] Anthropic. Context editing, 2026. URL https://platform.claude.com/docs/en/ build-with-claude/context-editing. Accessed 22 September 2026.

[54] Anthropic. Claude Code: Settings files and precedence, 2026. URL https://code.claude. com/docs/en/settings. Accessed 22 September 2026.

[55] Guannan Lai and Han-Jia Ye. When Routing Collapses: On the Degenerate Convergence of LLM Routers. arXiv:2602.03478, 2026. URL https://arxiv.org/abs/2602.03478.

[56] Varun Kotte. UCCI: Calibrated Uncertainty for Cost-Optimal LLM Cascade Routing. arXiv:2605.18796, 2026. URL https://arxiv.org/abs/2605.18796.

[57] Zhongling Xu, Shunan Zheng, and Wei Wang. SeqRoute: Global Budget-Aware Sequential LLM Routing via Ofline Reinforcement Learning. arXiv:2605.25424, 2026. URL https: //arxiv.org/abs/2605.25424.

[58] Rakibul Hasan Rajib, Mengxin Zheng, and Qian Lou. AgentServeSim: Serving-System Simulation and Policy Search for LLM Agent Programs. arXiv:2606.09613, 2026. URL https://arxiv.org/abs/2606.09613.

[59] OpenRouter. OpenRouter: prompt caching best practices (provider sticky routing), 2026. URL https://openrouter.ai/docs/guides/best-practices/prompt-caching. Accessed 21 September 2026.

[60] musistudio. claude-code-router, 2026. URL https://github.com/musistudio/ claude-code-router. Accessed 21 September 2026.

[61] gargpratyush. jev-router, 2026. URL https://github.com/gargpratyush/jev-router. Accessed 21 September 2026.

[62] antoniolg. Jev Codex Router (jev\_server.py, GitHub gist), 2026. URL https://gist. github.com/antoniolg/62f82f2a5d191fe074e3f5065501993d. Accessed 21 September 2026.

[63] OpenAI. Introducing GPT-5, August 2025. URL https://openai.com/index/ introducing-gpt-5/. Accessed 23 September 2026.

[64] Sharon Goldman. GPT-5’s model router ignited a user backlash against OpenAI—but it might be the future of AI. Fortune, August 2025. URL https://fortune.com/2025/08/ 12/openai-gpt-5-model-router-backlash-ai-future. Accessed 23 September 2026.

[65] Elias Lumer, Faheem Nizar, Akshaya Jangiti, Kevin Frank, Anmol Gulati, Mandar Phadate, and Vamse Kumar Subbiah. Don’t Break the Cache: An Evaluation of Prompt Caching for Long-Horizon Agentic Tasks. arXiv:2601.06007, 2026. URL https://arxiv.org/abs/ 2601.06007.

[66] Zentrik. Cache is part of the context, 2026. URL https://zentrik.ai/blog/ cache-is-part-of-the-context. Accessed 21 September 2026.

[67] Joel Becker, Nate Rush, Tom Cunningham, David Rein, and Khalid Mahamud. We are Changing our Developer Productivity Experiment Design. METR blog, February 2026. URL https://metr.org/blog/2026-02-24-uplift-update/. Accessed 23 September 2026.

[68] Anthropic. Claude Code: Manage costs efectively, 2026. URL https://code.claude. com/docs/en/costs. Accessed 21 September 2026.

[69] OpenAI. Codex: repository and configuration reference, 2026. URL https://github.com/ openai/codex. Accessed 21 September 2026.

[70] Google. Gemini CLI, 2026. URL https://github.com/google-gemini/gemini-cli. Accessed 21 September 2026.

[71] sst. OpenCode, 2026. URL https://opencode.ai/docs/. Accessed 21 September 2026.

[72] Mario Zechner. pi coding agent, 2026. URL https://pi.dev/docs/latest/providers. Accessed 21 September 2026.

[73] Block and the Agentic AI Foundation. Goose, 2026. URL https://github.com/ aaif-goose/goose. Accessed 21 September 2026.

[74] Aider-AI. Aider, 2026. URL https://aider.chat/docs/. Accessed 21 September 2026.

[75] Cline Bot Inc. Cline, 2026. URL https://docs.cline.bot/. Accessed 21 September 2026.

[76] Cursor. Cursor CLI documentation, 2026. URL https://cursor.com/docs/cli/ installation. Accessed 21 September 2026.

[77] GitHub. GitHub Copilot CLI and coding agent documentation, 2026. URL https:// github.com/github/copilot-cli. Accessed 21 September 2026.

[78] Amazon Web Services. Kiro documentation and licence, 2026. URL https://kiro.dev/ license/. Accessed 21 September 2026.

[79] Sourcegraph. Amp documentation, 2026. URL https://ampcode.com/docs/. Accessed 21 September 2026.

[80] Alibaba QwenLM. Qwen Code, 2026. URL https://github.com/QwenLM/qwen-code. Accessed 22 September 2026.

[81] DeepSeek. DeepSeek Harness, 2026. URL https://github.com/deepseek-ai/ deepseek-harness. Accessed 22 September 2026.

[82] Moonshot AI. Kimi Code, 2026. URL https://github.com/MoonshotAI/kimi-code. Accessed 22 September 2026.

[83] ByteDance. Trae Agent, 2026. URL https://github.com/bytedance/trae-agent. Accessed 22 September 2026.

[84] Z.ai (Zhipu). ZCode, 2026. URL https://github.com/zai-org/ZCode. Accessed 22 September 2026.

[85] MiniMax. MiniMax Code, 2026. URL https://github.com/MiniMax-AI/minimax-code. Accessed 22 September 2026.

[86] Tencent. CodeBuddy Code: settings reference, 2026. URL https://codebuddy.ai/docs/ cli/settings. Accessed 22 September 2026.

[87] iFlow. iFlow CLI, 2026. URL https://github.com/iflow-ai/iflow-cli. Accessed 22 September 2026.

## A The routing taxonomy

The taxonomy is bring-your-own: labels, descriptions and examples live in a configuration file that is sent to the classifier with every request and can be edited per enterprise; only the four context values are fixed, because the execution modes depend on their meaning. The default used in the case study has four axes and two modifiers.

Action (13, domain-neutral verbs). lookup, explain, draft, transform, edit, build, diagnose, review, plan, analyze, operate, steer (a short control message inside an ongoing task, which inherits the task’s tier), converse.

Domain (12). software, data analytics, product and design, marketing and communications, sales and customer, legal and compliance, finance and procurement, people and HR, operations and IT, research and strategy, general or personal, unknown. Domain drives policy floors and the expected session shape, not dificulty.

Complexity (4, ordinal). trivial (one step, one fact or a one-line change; under a minute of skilled work), routine (well specified and bounded; minutes to a quarter of an hour), complex (multi-step, needs synthesis or judgement; a quarter of an hour to hours), expert (novel, ambiguous or high-stakes reasoning; long-horizon agentic work where errors compound). Complexity maps to the capability tiers fast, balanced, strong and frontier.

Context (4, fixed). none (the prompt alone sufices), artifact (needs the current working set but not the conversation’s reasoning; reconstructible from a project ledger plus the files), history (needs prior reasoning, decisions or unfinished state from this conversation), external (needs information outside the session).

Modifiers. stakes (a wrong answer has material consequences; may raise the tier but never above the user’s own model) and output shape (short, medium, long; used for cost estimates).

Decision. The policy computes, from the classifier’s probability vectors, the probability that the tier the turn needs is at or below the tier of the candidate model and the probability that the turn does not depend on history. Both must clear a threshold (0.8 in the case study) for the router to act. The action steer, task-opening actions and turns in a task that is still open stay on the home model, and legal, finance and HR never go below the balanced tier, nor do agentic edits.

## B Cost model and constants

Per-request accounting. Each API request is billed in four categories: uncached input, cache read, cache write (5-minute or 1-hour) and output. For a user turn of k tool steps (k+1 requests) at prefix C with D new tokens spread over the requests and output O, a warm cache gives reads ≈ $( k { + } 1 ) C + D k / 2$ and writes $\approx D ;$ a cold cache writes $C + D$ and reads $k C + D k / 2$ . The simulator tracks per-model cache state, time since last touch against the lifetime, compaction at 85% of the window (a 20k-token summary replaces the history), and charges the side lane’s ledger build, its cache write and the one-request home refresh.

Misroutes. A moved turn whose ground-truth complexity requires a higher tier than the model it ran on is redone on the home model; a turn moved away from history it needed likewise; a session started on too weak a model pays its first turn twice. The naive-switch baseline is charged by the same rule.

Subagents. One subagent with k tool steps, system prompt plus brief P tokens, t tool-result tokens per step, o output tokens per step and a final report r costs reads $k P + t k ( k - 1 ) / 2$ writes $P + k t$ and output $k o + r$ . Profiles: light (5 steps, 2k tokens per step), typical (12 steps, 3k) and heavy (60 steps, 2.5k), with P = 16,200.

Measured behaviour. Every scenario is expanded into 200 variants (seed fixed; one random stream per behaviour and turn position, so a sensitivity changes one behaviour only), each carrying 1/200 of the scenario’s weight. After every task, k corrections are drawn from SWEchat’s distribution for tasks with or without tool work (probabilities for k = 0 to 12, a geometric tail above). A correction has 30 prompt tokens and the measured tool calls for its task’s size (4.9, 7.0, 8.4, 10.5, 13.3 for tasks of 0, 1–5, 6–20, 21–60 and more tool calls), with the corrected turn’s tokens per step. Pauses before follow-ups follow SWE-chat’s quantiles, a turn with tool work launches subagents with probability 0.039 and the measured fan-out, and every turn carries a misroute chain drawn from the corrections distribution conditioned on at least one.

Constants. System prompt and tools 15k tokens, ledger 6k, detached system prompt 1.5k, reintegration note 80. Cache lifetime 5 minutes (API default) or 1 hour (sensitivity). Each classifier call carries the prompt plus 350 tokens of session facts and the taxonomy’s question set of 1,812 tokens, and rule R10 asks its correction question (136 tokens) in a second call with the same state, at \$0.042 per million. Prompt tokens are counted with the o200k\_base tokenizer scaled by 1.25, because the vendor’s token-counting endpoint was unavailable. Prices are the list prices of 21 September 2026 (Section 2.1 and Table 1), and later releases are out of scope. Population: 10,000 seats in 14 segments by function and usage intensity, 301k sessions a month, engineering 86% of spend.

Reproduction. Every number, figure and table is produced ofline by the authors’ scripts (emulation, figures, worked examples, corpus calibration and the real-session repricing) from the catalogue, the cached classifier labels and the two public corpora; an audit script recomputes every printed number from the same files. The code and the catalogue are not released; Appendix A gives the taxonomy and policy and this appendix the cost model, so the method can be rebuilt with an enterprise’s own inputs.

## C The harness landscape

Twenty agentic coding harnesses graded on the properties an enterprise inherits (Section 2.2).

## D Supplementary figures

![](images/486f1a72cc3f1849746c167e4de590b2ded7f4c160374d622d68142871ba11d9.jpg)  
Figure 11: Cost of one subagent by model and activity profile (15k-token system prompt, 1.2k-token brief; tool results and output as in Appendix B). Subagents write their whole context once and read it a dozen times; the gap between the top model and the mid tier is largest for short subagents and narrows for very long ones.

![](images/b9efd34d035e0d82ec9e5d26ef73674c645de26bfb12b990d0be847e09f00aae.jpg)  
Figure 12: Saving, false-downgrade rate and hold rate against the decision-confidence threshold, applied to both of the classifier’s decision questions: the tier a turn needs and its dependence on the history.

<sub>of</sub> <sub>twenty</sub> <sub>agentic</sub> <sub>coding</sub> <sub>harnesses,</sub> <sub>from</sub> <sub>licence</sub> <sub>files</sub> <sub>and</sub> <sub>o</sub>fi<sup>cial</sup> <sup>configuration</sup> <sup>documents</sup> <sup>September</sup> <sup>2026).</sup> <sup>(</sup> <sub>harness</sub> <sub>at</sub> <sub>its</sub> <sub>own</sub> <sub>or</sub> <sub>a</sub> <sub>third-party</sub> <sub>model</sub> <sub>server</sub> <sub>with</sub> <sup>other</sup> <sup>vendors’</sup> <sup>models</sup> <sup>(a</sup> <sup>gateway</sup> <sup>in</sup> <sup>front</sup> <sup>of</sup> <sup>the</sup> <sup>ven</sup> <sub>n/d:</sub> <sub>not</sub> <sub>documented</sub> <sub>in</sub> <sub>o</sub>fi<sup>cial</sup>
<table><tr><td>Harness</td><td>Source</td><td>Model coupling</td><td>Own endpoint</td><td>Subagent model</td><td>Skills / MCP</td><td>Telemetry</td></tr><tr><td>Claude Code [17]</td><td>proprietary, native binary</td><td>Anthropic only</td><td></td><td>field, inherits</td><td>yes / yes</td><td>OpenTelemetry</td></tr><tr><td>Codex [69]</td><td>Apache-2.0</td><td>OpenAI default, custom providers</td><td>yes</td><td>config, per agent</td><td>yes / / client</td><td>opt-in OTel</td></tr><tr><td>Gemini CLI [70]</td><td>Apache-2.0</td><td>Google only</td><td>no</td><td>field, inherits</td><td>yes yes</td><td>OpenTelemetry</td></tr><tr><td>OpenCode [71]</td><td>MIT</td><td>75+ providers</td><td>any</td><td>per agent</td><td>yes / / yes</td><td>dropped in v2</td></tr><tr><td>pi [72]</td><td>MIT</td><td>any supported API</td><td>yes</td><td>no subagents</td><td>yes / refused</td><td>none</td></tr><tr><td>Goose [73]</td><td>Apache-2.0</td><td>15+ providers</td><td>yes</td><td>env, recipe</td><td>yes yes</td><td>OTLP</td></tr><tr><td>Aider [74]</td><td>Apache-2.0</td><td>any via LiteLLM</td><td>yes</td><td>no subagents</td><td>no no</td><td>none</td></tr><tr><td>Cline [75]</td><td>Apache-2.0</td><td>multi, any compatible</td><td>yes</td><td>inherits</td><td>yes / yes</td><td>enterprise</td></tr><tr><td>Cursor [76]</td><td>closed</td><td>own + a few vendors&#x27; keys</td><td>no</td><td>field, inherits</td><td>yes / yes</td><td>enterprise</td></tr><tr><td>Copilot CLI [77]</td><td>public, proprietary</td><td>Copilot + own keys</td><td>yes</td><td>field, inherits</td><td>yes / yes</td><td>announced</td></tr><tr><td>Kiro [78]</td><td>proprietary</td><td>Bedrock only</td><td>no</td><td>field</td><td>yes / / yes</td><td>enterprise</td></tr><tr><td>Amp [79]</td><td>closed</td><td>multi via routing</td><td>no</td><td>per mode</td><td>yes / yes</td><td>analytics API</td></tr><tr><td>Qwen Code [80]</td><td>Apache-2.0 (Gemini CLI fork)</td><td>Qwen default, OpenAI/Anthropic/local</td><td>yes</td><td>field, inherits</td><td>yes / yes</td><td>OTel, off by default</td></tr><tr><td>DeepSeek Harness [81]</td><td>MIT</td><td>model-agnostic (OpenAI, Anthropic protocols)</td><td>yes</td><td>delegate providers</td><td>plugins / plugins</td><td>third-party plugin</td></tr><tr><td>Kimi Code [82]</td><td>MIT</td><td>Kimi default, Anthropic/OpenAI/Google</td><td>yes</td><td>inherits; experimental pool</td><td>marketplace / yes</td><td>n/d</td></tr><tr><td>Trae Agent [83]</td><td>MIT</td><td>multi incl. Doubao, Ollama</td><td>yes</td><td>no subagents</td><td>no / yes</td><td>none</td></tr><tr><td>ZCode [84]</td><td>Apache-2.0</td><td>GLM default, tool-agnostic</td><td>config override</td><td>n/d</td><td>yes / yes</td><td></td></tr><tr><td>MiniMax Code [85]</td><td>MIT</td><td>MiniMax default, OpenAI/Anthropic formats</td><td>yes</td><td>subagents, no override found</td><td></td><td>n/d</td></tr><tr><td>CodeBuddy [86]</td><td>closed, npm</td><td>Tencent models, custom endpoint</td><td>yes</td><td>per-agent overrides</td><td>marketplace / yes no / yes</td><td>n/d none found</td></tr><tr><td>iFlow CLI [87]</td><td>closed, no licence file</td><td>Qwen, Kimi, DeepSeek; custom base URL</td><td>yes</td><td>agents, model n/d</td><td>no / marketplace</td><td>n/d</td></tr></table>

<table><tr><td>Model spend per engineer-day</td><td>5k seats</td><td>10k seats</td><td>25k seats</td><td>50k seats</td></tr><tr><td>$13.00 vendor-reported average</td><td>$0.7-1.1M</td><td>$1.3–2.1M</td><td>$3.3–5.3M</td><td>$6.7–10.6M</td></tr><tr><td>$15.18 TraceLab median</td><td>$0.7-1.2M</td><td>$1.5–2.3M</td><td>$3.7–5.9M</td><td>$7.4–11.7M</td></tr><tr><td>$39.73 as emulated</td><td>$1.6–2.5M</td><td>$3.3–5.0M</td><td>$8.2–12.5M</td><td>$16.4–25.0M</td></tr><tr><td>$58.40 TraceLab mean</td><td>$2.3–3.5M</td><td>$4.6–7.0M</td><td>$11.6–17.5M</td><td>$23.1–35.0M</td></tr></table>

Annual saving, today's labels to perfect labels; darker, more saved with today's labels; outlined, the case study. Engineering spend, as-is and routed, scaled by the anchor ÷ \$39.73 (the emulated engineer-day; volume only); non-engineering spend and every segment's percentage unchanged; all dollars then scaled by seats ÷ 10,000.  
Figure 13: Annual saving by model spend per engineer-day and by seats, from today’s labels to perfect labels; the outlined cell is the case study. Engineering spend scales with the spend level; other spend and every segment’s percentage stay as emulated.