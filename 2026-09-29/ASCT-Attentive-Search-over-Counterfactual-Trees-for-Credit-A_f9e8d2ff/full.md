# ASCT: Attentive Search over Counterfactual Trees for Credit Assignment in Agentic Reinforcement Learning

Yang Li<sup>1</sup> Jinhan Yang<sup>2</sup> hai liu<sup>3</sup> Di Wan

Xiyu Chen<sup>1</sup> Zongsi Xu<sup>1</sup> Tuo Zhou<sup>1</sup> Sheng Zhong<sup>1</sup>

Sergey Volkov<sup>1</sup> Ye Luo<sup>1,\*</sup> Hao Sun<sup>5,\*</sup>

<sup>1</sup>University of Hong Kong (hku.hk)

<sup>2</sup>The Chinese University of Hong Kong (cuhk.edu.hk)

<sup>3</sup>Jiangxi Science and Technology Normal University (jxstnu.edu.cn)

<sup>5</sup>Shenzhen University (szu.edu.cn)

\* Corresponding authors: Ye Luo and Hao Sun.

## Abstract

Terminal utility evaluates a complete agentic workflow, but learning requires credit for the decisions within it. We introduce Attentive Search over Counterfactual Trees (ASCT), a framework that turns training-time multi-step search into local action credit. At actor-visited states, an auxiliary tree evaluates alternative legal actions from the same recoverable prefix. Its action-value table is centered by the frozen actor’s probabilities and supplies credit for PPO on actor-sampled trajectories. This protocol connects counterfactual evaluation to policy learning while deploying the actor alone. Uniform, UCT, and cost-aware AgentUCT instantiate the framework. On HotpotQA agentic retrieval-augmented generation, all three improve mean held-out utility over trajectory-return PPO and workflow-adapted VinePPO. Across three seeds, ASCT-AgentUCT reaches 0.6187 utility versus 0.5939 for VinePPO, with gains in answer F1 and execution cost, and uses 50.3% fewer recorded auxiliary Qwen tokens. Transfer and component-description studies examine the learned policies beyond the training setting.

## 1 Introduction

A workflow actor, the trainable large language model (LLM) policy in this paper, chooses components, gathers evidence, and decides when to stop. Terminal utility combines answer quality and execution cost, but a successful workflow can still contain a poor local choice. Learning requires credit that connects each decision to its downstream consequences.

VinePPO estimates state values from Monte Carlo continuations for step-wise credit (Kazemnejad et al., 2025); branching learners derive updates from collections of search trajectories. We study an action-resolved interface: compare executable alternatives at states the actor visits, then use those comparisons to train its own sampled decisions.

Recoverable prefixes make this possible. From the same observations, an evaluator can execute alternative legal actions through later workflow decisions to terminal outcomes. Their value depends on these continuations: diferent retrieved evidence can change whether another retrieval round is useful. Finite action sets permit root coverage, while shared prefixes permit execution reuse within a limited auxiliary budget.

We introduce Attentive Search over Counterfactual Trees (ASCT). To our knowledge, ASCT is the first agentic RL framework to use multi-step counterfactual tree search as an auxiliary legal-action value estimator for actor-centered credit on separately sampled actor trajectories. Search constructs a local Q table; frozen actor probabilities center its values; PPO updates the actor’s sampled decisions with their recorded behavior probabilities. Deployment uses the actor alone.

All three evaluators—Uniform, UCT, and AgentUCT—improve mean held-out utility over trajectory-return PPO and workflow-adapted VinePPO. Across three seeds, ASCT-AgentUCT reaches 0.6187 versus 0.5939 for VinePPO, combining higher answer F1 with lower execution cost and using 50.3% fewer recorded auxiliary Qwen tokens. Transfer and description-controlled experiments examine the learned policies beyond the training setting.

ASCT connects: multi-step action comparison from the same recoverable state; separate evaluation allocation and behavior sampling, linked by actor-centered credit and consistent PPO records; and training-time information acquisition amortized into an actor-only policy. Uniform, UCT, AgentUCT, and WTB provide inherited search and execution components (Li et al., 2026). We evaluate the resulting learning protocol, acquisition cost, and transfer.

## 2 Related Work

Auxiliary evaluation and credit. PPO updates sampled decisions using advantage estimates (Schulman et al., 2017). VinePPO supplies step credit from Monte Carlo state values (Kazemnejad et al., 2025). It is our closest estimator comparison: both credit the main actor trajectory, allowing us to compare state-value credit with ASCT’s action table. COMA uses a policy-weighted counterfactual baseline with a centralized critic (Foerster et al., 2018); ASCT estimates values through workflow execution.

Tree-based RL for agentic decisions. BPO uses sibling-relative credit at high-entropy branch states; Tree-GRPO uses intra- and inter-tree advantages; TreePS-RAG derives process supervision from descendants (He et al., 2026; Ji et al., 2026; Zhang et al., 2026). ATPO combines dialogue branching with critic-based credit, and AT<sup>2</sup>PO uses turn-level tree credit (Cao et al., 2026; Zong et al., 2026). These learners optimize branched rollout collections. ASCT’s tree estimates legal action values at separately sampled actor states; only actor-trajectory decisions enter PPO. Its multi-step branches execute evidence-dependent component choices, linking workflow consequences to this distinct learning interface.

Search allocation and execution reuse. UCT allocates trials through reward and uncertainty (Auer et al., 2002; Kocsis and Szepesvári, 2006). LATS and related methods search for deployed actions (Zhou et al., 2024; Koh et al., 2024). We use AgentUCT’s reuse-aware allocation and WTB execution infrastructure (Li et al., 2026) to acquire training credit, then assess both acquisition cost and the learned actor.

## 3 Problem Formulation

A state $s _ { t }$ contains a task, its executed workflow prefix, and observations. The actor $\pi _ { \boldsymbol { \theta } } ( \boldsymbol { a } \mid \boldsymbol { s } _ { t } )$ chooses from a finite legal set $\boldsymbol { \mathcal { A } } ( \boldsymbol { s } _ { t } )$ . A terminal workflow w receives

$$
U ( w ) = P ( w ) - \lambda C _ { \mathrm { e x e c } } ( w ) / C _ { 0 } ,\tag{1}
$$

where $P$ is answer quality and $C _ { \mathrm { e x e c } }$ is execution cost. The learning objective is expected terminal utility. Recoverable prefixes permit alternative continuations at actor-visited states. We hold the PPO learner fixed to compare credit estimators, with B terminal trials per evaluated state. Equal logical budgets can require diferent physical work.

Auxiliary work is logged separately from workflow execution utility; RQ2 defines the costadjusted metrics.

## 4 ASCT

An agentic decision example. At a retriever choice, the actor may favor a familiar retriever. Another legal retriever could expose evidence that makes an extra retrieval round useful; its value emerges after continuation, stopping, and reranking. ASCT restores the shared prefix and evaluates these multi-step outcomes. The actor still executes its own sampled action; search supplies the comparison used to train that decision (Figure 1).

Separate behavior from evaluation. At visited state $s ,$ the frozen actor computes ${ p } _ { a } =$ $\pi _ { \mathrm { o l d } } ( \boldsymbol { a } \mid \boldsymbol { s } )$ and samples $a _ { t }$ . An independent random stream evaluates alternatives from recoverable copies of s. Search allocation can change while the actor advances with $a _ { t }$

Retain multi-step action values. Tree nodes are workflow prefixes and edges are legal actions. Trials execute selected paths to completion and back up $U .$ . Root coverage evaluates every action before repeat allocation $( B \geq | { \mathcal { A } } ( s ) | )$ . Each value averages the returns of its root-action trials:

$$
\widehat { Q } _ { E , B } ( s , a ) = \frac { 1 } { | \mathcal { T } _ { E , B } ( s , a ) | } \sum _ { i \in \mathcal { T } _ { E , B } ( s , a ) } U ( w _ { i } ) .\tag{2}
$$

Here $\mathcal { T } _ { E , B } ( s , a )$ indexes trials starting with $a ;$ evaluator $E$ and budget B determine their sufixes. Credit the sampled action relative to the actor.

$$
\widehat { V } _ { E } ( s ) = \sum _ { a \in \mathcal { A } ( s ) } p _ { a } \widehat { Q } _ { E , B } ( s , a ) ,\tag{3}
$$

$$
\begin{array} { r } { \widehat { A } _ { E } ( s , a _ { t } ) = \widehat { Q } _ { E , B } ( s , a _ { t } ) - \widehat { V } _ { E } ( s ) . } \end{array}\tag{4}
$$

For fixed $\widehat { Q } _ { E , B }$ , let $\begin{array} { r } { L _ { s } ( \theta ) = \sum _ { a } \pi _ { \theta } ( a \mid s ) \widehat { Q } _ { E , B } ( s , a ) } \end{array}$ . At $\pi _ { \theta } = \pi _ { \mathrm { o l d } }$ , its derivative with respect to softmax logit $z _ { a }$ is $\partial L _ { s } / \partial z _ { a } = p _ { a } \widehat { A } _ { E } ( s , a )$ . Thus, actor-centered credit provides a local policygradient signal before normalization and clipping.

Reuse execution and update online. Uniform balances visits; UCT uses utility and exploration; AgentUCT also accounts for predicted uncached cost. WTB restores shared prefixes and executes uncached sufixes for all evaluators and VinePPO. Each actor decision contributes $( s , a _ { t } , p _ { a _ { t } } , \widehat { A } _ { E } ( s , a _ { t } ) )$ ). After an iteration, credit is standardized and PPO uses $\rho _ { \theta } = \pi _ { \theta } ( a _ { t } \mid s ) / p _ { a _ { t } }$ (Schulman et al., 2017). One frozen actor defines sampling, baseline weights, and the ratio (a) ASCT: multi-step search supplies local actor credit

![](images/623fbdcd77b6eb129a5adffe41676afacaa30f679987df61e02476c0e286455d.jpg)

Update actor-sampled decisions; use branches only to estimate credit.

Next iteration: refresh actor states and credit. Deployment: actor only.

(b) VinePPO: credit from adjacent MC state values

![](images/36682d6c7984111cc003c3181a9c86848f7ddfe4877c8395976149daf7b98003.jpg)  
Figure 1: ASCT action-value credit and VinePPO state-value credit share actor-trajectory PPO updates. Solid arrows carry behavior records; dashed arrows carry auxiliary information. Frozen probabilities center ASCT’s action table. Values and tree geometry are illustrative; subscripts are suppressed.

denominator; auxiliary branches supply credit without entering the update trajectories. The next iteration refreshes visited states and credit with the updated actor. Deployment runs the actor alone. Algorithm 1 and Appendix A specify the full procedure and objective; Appendix D illustrates evaluator rules.

## 5 Experiments

Four RQs evaluate final-policy efectiveness, auxiliary cost, cross-dataset transfer, and descriptionguided choice. Methods share the workflow and learning protocol. Section 6 relates these policy

results to supplementary evaluation controls.

## 5.1 Agentic workflow and RAGSpace

We instantiate the workflow in RAGSpace on HotpotQA questions and supplied candidate passages (Yang et al., 2018; Li et al., 2026). The actor successively chooses query formulation, retrieval, evidence selection, continuation, and reranking (Table 1). Retrieval and generation consume passage text; answer annotations score completed workflows. The shared generator and retrieval components remain frozen.

The query choices are the original question, Query2Doc-style expansion (Wang et al., 2023), and two-query decomposition. Retrievers are BM25 (Robertson and Zaragoza, 2009), E5-base-v2 (Wang et al., 2022), BGE-M3 hybrid (Chen et al., 2024), Qwen3-Embedding-0.6B (Zhang et al., 2025), and reciprocal-rank fusion of their rankings (Cormack et al., 2009). Passage selection uses relevance or maximal marginal relevance (Carbonell and Goldstein, 1998); optional reranking uses Qwen3-Reranker-0.6B (Zhang et al., 2025).

Table 1: Executable RAG decisions. A workflow uses up to three retrieval rounds. Choices within a row are legal alternatives at that stage.
<table><tr><td>Stage</td><td>Actions</td></tr><tr><td>Query formulation</td><td>Original question; Query2Doc; two-query decomposition</td></tr><tr><td>Retriever</td><td>BM25; E5 dense; BGE-M3 hybrid; Qwen3 embedding; reciprocal-rank fusion</td></tr><tr><td>Passage selection</td><td>Relevance ranking; maximal marginal relevance</td></tr><tr><td>Retrieval width</td><td>3 or 6 passages</td></tr><tr><td>Retrieval control</td><td>Stop; continue while fewer than three rounds have executed</td></tr><tr><td>Reranking</td><td>Preserve order; Qwen3 reranker</td></tr><tr><td>Answer context</td><td>Top 2 or top 4 evidence passages</td></tr></table>

Evidence-dependent control. After retrieval, Stop ends acquisition; Continue refines the query from retrieved evidence, repeats the chosen retrieval rule, and merges passages. The actor then chooses again. At round three only Stop is legal; reranking and context-width choices finish the workflow. Thus earlier choices change later evidence and the value of continuing. Prompts and deterministic refinement are fixed across methods (Appendix A.3).

Actor and observations. The planner sees the question, stage, selected components, round, active queries, evidence titles, and legal labels. It scores legal action responses and normalizes their probabilities; training samples actions and held-out evaluation is greedy. The actor is Qwen3-4B-Instruct-2507 (Qwen Team, 2025) with LoRA (Hu et al., 2022); query and answer generation use the frozen backbone with its adapter disabled. Learning changes workflow decisions while keeping component implementations fixed.

Protocol and comparisons. All methods share the actor, workflow, splits, and PPO configuration (batch size eight; two update epochs per iteration). We train for three iterations on 2,000 questions and evaluate final checkpoints on 400 test questions over seeds 11, 23, and 37. Training uses the repository F1 scorer; endpoint reporting uses oficial F1, with λ = 0.1 and $C _ { 0 } = 4 0 9 6$ . VinePPO estimates actor-continuation state values for one-step, unit-discount credit; PPO uses terminal trajectory credit. ASCT-Uniform, UCT, and AgentUCT vary the evaluator. VinePPO and ASCT share $B = 1 2$ and WTB reuse. Logical budgets and realized work are reported separately. We set $c _ { \mathrm { e x p } } = 1 . 4$ and $c _ { \mathrm { t o k } } = \beta = 1 0 ^ { - 4 }$ ; Appendix A gives the full configuration.

## 5.2 RQ1: Does counterfactual credit improve the final policy?

Table 2: HotpotQA held-out results of the final policies. Workflow utility is the mean over test questions, using oficial F1 and execution words. Trained rows are mean ± sample SD over three seeds.
<table><tr><td>Method</td><td>Workflow utility</td><td></td><td>Answer F1 Execution words</td></tr><tr><td>Base</td><td>0.5648</td><td>0.6954</td><td>5,348.7</td></tr><tr><td>PPO</td><td> $0 . 5 6 5 1 \pm 0 . 0 0 2 1$ </td><td> $0 . 6 9 1 3 \pm 0 . 0 0 2 1$ </td><td> $5 , 1 7 0 . 4 \pm 7 6 . 8$ </td></tr><tr><td>VinePPO</td><td> $0 . 5 9 3 9 \pm 0 . 0 0 6 2$ </td><td> $0 . 7 1 1 5 \pm 0 . 0 0 2 6$ </td><td> $4 , 8 1 7 . 7 \pm 1 7 9 . 7$ </td></tr><tr><td>ASCT-Uniform</td><td> $\mathbf { 0 . 6 1 4 6 \ : \pm { \ : 0 . 0 1 9 1 } }$ </td><td> $\mathbf { 0 . 7 3 1 6 \pm 0 . 0 1 6 8 }$ </td><td> $\mathbf { 4 } , \mathbf { 7 9 2 . 2 \ : \pm { \ : 1 5 1 . 4 } }$ </td></tr><tr><td>ASCT-UCT</td><td> $\mathbf { 0 . 6 0 7 5 \pm 0 . 0 2 4 1 }$ </td><td> $\mathbf { 0 . 7 2 5 7 \pm 0 . 0 2 6 0 }$ </td><td> ${ \bf 4 } , { \bf 8 4 1 . 2 } \pm { \bf 1 1 2 . 8 }$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 6 1 8 7 \ : \pm { \ : 0 . 0 0 7 6 } }$ </td><td> $\mathbf { 0 . 7 2 6 6 \pm 0 . 0 1 2 6 }$ </td><td> $\mathbf { 4 , 4 2 0 . 8 \ : \pm 2 1 3 . 3 }$ </td></tr></table>

![](images/2c51a0d7d23d1c6efe3d2ad4980101df253c4e7a695db8c2245190ee4acdf39e.jpg)  
Figure 2: HotpotQA test utility. Diamonds and dashed guides mark means; bars show ±1 sample SD over three seeds; open circles show seeds. Base is fixed; ASCT labels are bold.

All three ASCT evaluators improve mean held-out utility over VinePPO and trajectory-return PPO (Table 2, Figure 2). ASCT-AgentUCT reaches 0.6187, a gain of 0.0248 over VinePPO: 0.0151 comes from higher answer F1 and 0.0097 from lower execution cost. Execution words decrease by 8.24%. The paired utility diferences are positive at all three seeds: 0.0293, 0.0354, and 0.0098.

ASCT-Uniform reaches utility 0.6146 and the highest mean F1, 0.7316, supporting the interface with balanced evaluation. The smaller AgentUCT–Uniform diference, $0 . 0 0 4 1 \pm 0 . 0 1 1 6$ , changes sign across seeds; acquisition cost also matters to evaluator choice.

Where the gains occur. All ASCT policies improve medium- and hard-question utility over VinePPO and PPO (Figure 3). AgentUCT gains 0.0307 and 0.0339 over VinePPO, respectively;

the easy group is closely matched, with VinePPO slightly higher.

![](images/3569f323cf66ce1eb21f29f23ae7a78336bf00e4fa02e9deb089e1e008abab3c.jpg)  
Figure 3: Test utility by dificulty: mean ± sample SD over three seeds, with open seed markers. Base is fixed. Panels share the utility scale and method order.

## 5.3 RQ2: What does auxiliary evaluation cost, and when is it valuable?

Auxiliary tokens are $T _ { \mathrm { a u x } } = T _ { \mathrm { e n v } } + T _ { \mathrm { a c t o r , a u x } } \colon$ recorded Qwen environment calls and auxiliary actor scoring after reuse; E5, BGE-M3, and index construction are outside this token sum. For K logical terminal trials and an assumed N subsequent policy uses, define

$$
\begin{array} { r } { \overline { { J } } _ { \mathrm { s e a r c h } } = \left[ \sum _ { i = 1 } ^ { K } U ( w _ { i } ) - \beta T _ { \mathrm { a u x } } \right] / K , } \end{array}\tag{5}
$$

$$
J _ { \mathrm { d e p l o y } } ( N ) = \overline { { U } } _ { \mathrm { t e s t } } - \beta T _ { \mathrm { a u x } } / N .\tag{6}
$$

The first measures acquired outcomes after their cost; the second projects auxiliary-cost amortization, assuming held-out utility represents future use. Search backup and policy credit use U. Main-trajectory collection and optimizer scoring have separate counters. The common token weight $\beta$ defines an accounting convention; GPU time or API price requires model-specific calibration (Appendix B).

Auxiliary tokens for the recorded Qwen operations. Table 3 separates environment execution from actor scoring. VinePPO uses 137.09 million environment tokens and 251.29 million tokens to score auxiliary actor continuations, totaling 388.37 million. ASCT obtains continuation decisions from its search rules and incurs no auxiliary actor-scoring calls. AgentUCT uses 193.09 million tokens in this auxiliary ledger, 50.3% below VinePPO and 2.02–2.28% below the other ASCT evaluators. The within-ASCT reductions occur in all three seeds. The 50.3% diference chiefly reflects avoided actor-continuation scoring; VinePPO uses fewer environment tokens. The cost-aware evaluator contributes the smaller within-ASCT reduction.

VinePPO reuses frozen action probabilities within each question; scoring is charged once to the first requesting path.

Cost-adjusted evaluation value. Table 4 aggregates rollout utility, auxiliary tokens, and $\overline { { J } } _ { \mathrm { s e a r c h } }$ over complete runs. Figure 4 shows per-iteration outcomes within ASCT. AgentUCT

Table 3: Combined token ledger for the listed Qwen model operations across three training iterations, in millions (mean ± sample SD over three seeds). The first row sums auxiliary environment execution and auxiliary actor scoring within each seed. Indented rows are components of their section subtotal.
<table><tr><td>Token count (M)</td><td>VinePPO</td><td>ASCT- Uniform</td><td>ASCT- UCT</td><td>ASCT- AgentUCT</td></tr><tr><td>Auxiliary total</td><td> $\mathbf { 3 8 8 . 3 7 4 \pm 4 . 6 9 4 }$ </td><td> $\mathbf { 1 9 7 . 0 7 7 \pm 0 . 0 2 4 }$ </td><td> $\mathbf { 1 9 7 . 5 8 8 \ : \pm { \ : 0 . 0 6 8 } }$ </td><td> $\mathbf { 1 9 3 . 0 8 8 \pm 0 . 4 0 4 }$ </td></tr><tr><td colspan="5">Auxiliary environment execution</td></tr><tr><td>Environment subtotal  $T _ { \mathrm { e n v } }$ </td><td> $1 3 7 . 0 8 8 \pm 1 . 2 3 1$ </td><td> $1 9 7 . 0 7 7 \pm 0 . 0 2 4$ </td><td> $1 9 7 . 5 8 8 \pm 0 . 0 6 8$ </td><td> $1 9 3 . 0 8 8 \pm 0 . 4 0 4$ </td></tr><tr><td>Generator input (4B)</td><td> $8 9 . 9 7 2 \pm 0 . 2 7 8$ </td><td> $1 3 9 . 4 5 2 \pm 0 . 1 6 2$ </td><td> $1 3 9 . 6 1 3 \pm 0 . 1 9 3$ </td><td> $1 3 7 . 7 7 6 \pm 0 . 1 7 4$ </td></tr><tr><td>Generator output (4B)</td><td> $3 . 5 9 0 \pm 0 . 0 0 9$ </td><td> $5 . 8 7 4 \pm 0 . 0 0 3$ </td><td> $5 . 8 8 3 \pm 0 . 0 0 2$ </td><td> $5 . 8 3 3 \pm 0 . 0 0 3$ </td></tr><tr><td>Query embedding (0.6B)</td><td> $0 . 6 1 1 \pm 0 . 0 1 2$ </td><td> $0 . 6 8 9 \pm 0 . 0 0 2$ </td><td> $0 . 6 9 0 \pm 0 . 0 0 1$ </td><td> $0 . 6 8 9 \pm 0 . 0 0 2$ </td></tr><tr><td>Reranker input (0.6B)</td><td> $4 2 . 9 1 6 \pm 1 . 4 5 2$ </td><td> $5 1 . 0 6 2 \pm 0 . 1 5 2$ </td><td> $5 1 . 4 0 1 \pm 0 . 1 4 8$ </td><td> $4 8 . 7 8 9 \pm 0 . 2 2 7$ </td></tr><tr><td>Actor scoring (4B)</td><td></td><td></td><td></td><td></td></tr><tr><td>Actor-scoring subtotal</td><td> $3 0 5 . 2 3 4 \pm 3 . 5 1 5$ </td><td> $7 6 . 3 2 5 \pm 0 . 0 3 8$ </td><td> $7 6 . 2 0 8 \pm 0 . 0 5 1$ </td><td> $7 6 . 1 1 5 \pm 0 . 1 4 8$ </td></tr><tr><td>Auxiliary continuations</td><td> $2 5 1 . 2 8 6 \pm 3 . 4 7 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td>Main-trajectory</td><td> $3 . 1 5 1 \pm 0 . 0 1 3$ </td><td> $2 5 . 4 4 2 \pm 0 . 0 1 3$ </td><td> $2 5 . 4 0 3 \pm 0 . 0 1 7$ </td><td> $2 5 . 3 7 2 \pm 0 . 0 4 9$ </td></tr><tr><td>collection</td><td></td><td></td><td></td><td></td></tr><tr><td>Optimizer forwards</td><td> $5 0 . 7 9 6 \pm 0 . 0 4 4$ </td><td> $5 0 . 8 8 3 \pm 0 . 0 2 5$ </td><td> $5 0 . 8 0 5 \pm 0 . 0 3 4$ </td><td> $5 0 . 7 4 4 \pm 0 . 0 9 8$ </td></tr></table>

The main J metrics use the auxiliary total: environment subtotal plus auxiliary-continuation scoring. Main-trajectory collection and optimizer forwards contribute to the actor-scoring subtotal.  
has the highest mean $J _ { \mathrm { s e a r c h } }$ in each iteration and the lowest full-run auxiliary cost. Both use environment execution plus auxiliary actor scoring.

Table 4: Mean auxiliary rollout utility, recorded auxiliary token cost, and $\overline { { J } } _ { \mathrm { s e a r c h } }$ over complete training runs. Rollout utility uses the training scorer; tokens include the listed Qwen environment operations and auxiliary actor scoring. Values are mean ± sample SD across three seeds.
<table><tr><td>Evaluator</td><td></td><td>Mean rollout U Auxiliary tokens (M)</td><td> $\overline { { J } } _ { \mathrm { s e a r c h } }$ </td></tr><tr><td>VinePPO</td><td> $0 . 5 5 0 7 \pm 0 . 0 0 2 7$ </td><td> $3 8 8 . 3 7 4 \pm 4 . 6 9 4$ </td><td> $0 . 4 8 9 3 \pm 0 . 0 0 1 9$ </td></tr><tr><td>ASCT-Uniform</td><td> $\mathbf { 0 . 5 2 8 3 \pm 0 . 0 0 2 2 }$ </td><td> $\mathbf { 1 9 7 . 0 7 7 \pm 0 . 0 2 4 }$ </td><td> $\mathbf { 0 . 4 9 7 2 \pm 0 . 0 0 2 2 }$ </td></tr><tr><td>ASCT-UCT</td><td> $\mathbf { 0 . 5 5 1 5 \ : \pm { \ : 0 . 0 0 3 1 } }$ </td><td> ${ \bf 1 9 7 . 5 8 8 \pm 0 . 0 6 8 }$ </td><td> $\mathbf { 0 . 5 2 0 3 \pm 0 . 0 0 3 1 }$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 5 5 2 6 \ : \pm { \ : 0 . 0 0 1 6 } }$ </td><td> $\mathbf { 1 9 3 . 0 8 8 \pm 0 . 4 0 4 0 . 5 2 2 0 \pm 0 . 0 0 1 6 }$ </td><td></td></tr></table>

Value after amortizing auxiliary work. Figure 5 projects final-policy value under Equation 6. PPO leads at 100,000 uses; AgentUCT leads at 500,000 and 1,000,000. AgentUCT has higher mean utility and lower auxiliary tokens than VinePPO, giving a mean advantage for every positive N. Its mean curve crosses PPO at approximately 360,358 uses. These thresholds assume the specified token weight and subsequent-use utility; only auxiliary expenditure is amortized.

![](images/d99ae566de3858262a36e2bbfb9d97dd9b89fd8686fe77633a642c77490cc263.jpg)  
Figure 4: Per-iteration auxiliary evaluation within ASCT: mean rollout $U _ { \mathrm { t r a i n } }$ and $J _ { \mathrm { s e a r c h } }$ (2,000 questions; $B = 1 2 )$ Utility uses the training scorer. Bands and bars: ± sample SD across three seeds.

![](images/dbc88d707cead817868e0c03157d8e61e57ea64a3ad894ae27ee857cb8bb6298.jpg)  
Figure 5: Amortized value across assumed policy uses N (Equation $6 ; \beta = 1 0 ^ { - 4 } )$ . Bands: ± sample SD across three seed projections. The guide marks AgentUCT’s mean crossing with PPO. Both J metrics charge the recorded auxiliary ledger.

## 5.4 RQ3: Does the learned policy transfer across question distributions?

New question distributions. Without further updates, we evaluate on 100 unseen 2WikiMultihopQA bridge-comparison questions and 100 MuSiQue questions (Ho et al., 2020; Trivedi et al., 2022). MuSiQue contains 34 two-hop, 33 three-hop, and 33 four-hop items. Supplied passages are mapped to the same RAG environment; policies share questions, frozen components, and alias-aware answer scoring.

Transfer difers by dataset (Table 5). On 2Wiki, AgentUCT improves both F1 (0.6700 versus 0.6517) and execution words (2,939.3 versus 3,069.3) over VinePPO, raising utility from 0.5767 to 0.5982. On MuSiQue, its lower execution cost ofsets slightly lower F1, yielding utility 0.0048 versus −0.0047. These unchanged actors use the same component library under new questions and evidence.

## 5.5 RQ4: Do trained policies retain description-guided component choice?

A newly available component. A fixed retriever A extends Qwen embedding retrieval with a query instruction and metadata rule. In a 100-question constructed probe, annotation-derived tags mark answer-bearing passages and A prioritizes them. Its implementation, name, corpus, actor weights, and all existing component descriptions stay fixed. Four conditions vary only A’s description: type plus the correct evidence relation, type and construction details only, name only,

Table 5: Cross-dataset results on the fixed 100-question cohorts. F1, workflow utility, and execution words are mean ± sample SD over three seeds; Base is deterministic.
<table><tr><td>Dataset</td><td>Method</td><td>F1</td><td></td><td>Utility Execution words</td></tr><tr><td>2Wiki</td><td>Base</td><td>0.5583</td><td>0.4732</td><td>3,488.0</td></tr><tr><td>2Wiki</td><td>PPO</td><td> $0 . 5 8 8 3 \pm 0 . 0 1 0 0$ </td><td> $0 . 5 0 6 7 \pm 0 . 0 1 0 2$ </td><td> $3 , 3 4 2 . 6 \pm 1 3 0 . 9$ </td></tr><tr><td>2Wiki</td><td>VinePPO</td><td> $0 . 6 5 1 7 \pm 0 . 0 3 7 5$ </td><td> $0 . 5 7 6 7 \pm 0 . 0 3 6 0$ </td><td> $3 , 0 6 9 . 3 \pm 1 3 3 . 7$ </td></tr><tr><td>2Wiki</td><td>ASCT-Uniform</td><td> $\mathbf { 0 . 6 5 6 7 \pm 0 . 0 3 2 1 }$ </td><td> $\mathbf { 0 . 5 8 3 2 \ : \pm { \ : 0 . 0 3 2 9 } }$ </td><td> $\mathbf { 3 , 0 0 9 . 9 } \pm \mathbf { 4 7 . 6 }$ </td></tr><tr><td>2Wiki</td><td>ASCT-UCT</td><td> $\mathbf { 0 . 6 5 8 3 \pm 0 . 0 5 2 2 }$ </td><td> $\mathbf { 0 . 5 8 3 7 \ : \pm { \ : 0 . 0 5 0 8 } }$ </td><td> $\mathbf { 3 , 0 5 5 . 2 } \pm \mathbf { 5 9 . 8 }$ </td></tr><tr><td>2Wiki</td><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 6 7 0 0 \pm 0 . 0 3 2 1 }$ </td><td> $\mathbf { 0 . 5 9 8 2 \ : \pm { \bf 0 . 0 3 1 3 } }$ </td><td> $\mathbf { 2 , 9 3 9 . 3 \ : \pm { \ : 7 3 . 7 } }$ </td></tr><tr><td>MuSiQue</td><td>Base</td><td>0.2198</td><td>-0.0215</td><td>9,880.9</td></tr><tr><td> $\mathrm { M u S i Q u e }$ </td><td>PPO</td><td> $0 . 2 1 8 1 \pm 0 . 0 0 2 9$ </td><td> $- 0 . 0 1 8 4 \pm 0 . 0 0 6 1$ </td><td> $9 , 6 8 6 . 6 \pm 1 4 4 . 0$ </td></tr><tr><td> $\mathrm { M u S i Q u e }$ </td><td>VinePPO</td><td> $0 . 2 2 4 8 \pm 0 . 0 1 0 0$ </td><td> $- 0 . 0 0 4 7 \pm 0 . 0 1 2 8$ </td><td> $9 , 3 9 7 . 0 \pm 1 5 6 . 5$ </td></tr><tr><td> $\mathbf { M u S i Q u e }$ </td><td>ASCT-Uniform</td><td> $\mathbf { 0 . 2 1 9 2 \ : \pm 0 . 0 1 9 5 ~ - 0 . 0 0 7 2 \ : \pm 0 . 0 2 3 2 }$ </td><td></td><td> $\mathbf { 9 , 2 7 4 . 5 \ : \pm 1 8 1 . 5 }$ </td></tr><tr><td> $\mathbf { M u S i Q u e }$ </td><td>ASCT-UCT</td><td> $\mathbf { 0 . 2 2 2 5 \ : \pm 0 . 0 0 8 4 ~ - 0 . 0 0 4 1 \ : \pm 0 . 0 0 8 5 }$ </td><td></td><td> $\mathbf { 9 , 2 8 4 . 5 \ : \pm { \ : 2 7 2 . 0 } }$ </td></tr><tr><td> $\mathbf { M u S i Q u e }$ </td><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 2 1 9 2 \ : \pm 0 . 0 1 3 5 }$ </td><td> $\mathbf { 0 . 0 0 4 8 \ : \pm { \ : 0 . 0 0 8 2 } }$ </td><td> $\mathbf { 8 , 7 8 2 . 3 \ : \pm { \ : 3 5 8 . 6 } }$ </td></tr></table>

and type plus the opposite relation. Appendix F summarizes the protocol.

Table 6: Selection rate for the same new retriever under four descriptions. The correct relation states a preference for tagged, answer-bearing evidence; the opposite relation states a preference for untagged passages. Values are mean ± sample SD over three seeds.
<table><tr><td rowspan="2">Planner</td><td rowspan="2"> $\mathbf { T y p e } +$  correct relation</td><td rowspan="2"></td><td rowspan="2"></td><td rowspan="2"> $\mathbf { T y p e } +$  Name only opposite relation</td></tr><tr><td>Type only</td></tr><tr><td>Base</td><td>0.820</td><td>0.010</td><td>0.000</td><td>0.020</td></tr><tr><td>PPO</td><td> $0 . 8 3 0 \pm 0 . 0 4 4$ </td><td> $0 . 0 3 3 \pm 0 . 0 4 2$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 2 0 \pm 0 . 0 0 0$ </td></tr><tr><td>VinePPO</td><td> $0 . 9 4 7 \pm 0 . 0 3 5$ </td><td> $0 . 2 8 0 \pm 0 . 1 0 8$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 4 0 \pm 0 . 0 1 0$ </td></tr><tr><td>ASCT-Uniform</td><td> $\mathbf { 0 . 9 6 0 \pm 0 . 0 1 7 0 . 3 0 0 \pm 0 . 1 1 1 0 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td></td><td></td><td> $\mathbf { 0 . 0 3 7 \pm 0 . 0 0 6 }$ </td></tr><tr><td>ASCT-UCT</td><td> $\mathbf { 0 . 9 3 3 \pm 0 . 0 2 1 0 . 1 9 7 \pm 0 . 0 6 8 0 . 0 0 0 \pm 0 . 0 0 0 }$ </td><td></td><td></td><td> $\mathbf { 0 . 0 2 7 \pm 0 . 0 1 2 }$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 9 5 0 \ : \pm { \ : 0 . 0 3 6 } }$ </td><td>0.373 ± 0.205</td><td> $\mathbf { 0 . 0 0 0 \ : \pm { \ : 0 . 0 0 0 } }$ </td><td> $\mathbf { 0 . 0 5 0 \ : \pm { \ : 0 . 0 2 0 } }$ </td></tr></table>

Selection responds strongly to the description while the component itself stays fixed (Table 6). Under the correct evidence relation, ASCT selects A on 93.3–96.0% of questions, compared with 94.7% for VinePPO and 82.0% for Base. Removing the relation lowers ASCT selection to 19.7–37.3%; name-only descriptions yield zero selection, and the opposite relation yields 2.7–5.0%. The contrasts show that the trained actors retain description-guided choice across this library extension. This is a controlled selection probe with constructed tags; the cross-dataset study separately evaluates answer quality and execution eficiency.

## 6 Discussion

What the learned-policy results establish. The common gain across Uniform, UCT, and AgentUCT supports ASCT’s action-evaluation interface across search rules. Uniform’s efectiveness is evidence that the benefit does not require cost-aware allocation. The methodological contribution is the connection from executable multi-step alternatives to actor-trajectory updates; the following controls characterize how evaluation supplies that connection.

Tree allocation and evaluation budget. On common five-action states from two questions and three frozen actors, root MC and Uniform share root coverage, uniform sufix sampling, and WTB reuse. At $B = 1 2 .$ , Uniform lowers repeat Q SD from 0.1203 to 0.0619 and raises sign agreement from 63.3% to 80.0%, using 1.7% more tokens (Table 15). The Q-SD ordering reverses at $B = 4 8$ . Figure 6 traces information acquisition: at 48 trials, UCT, AgentUCT, and Uniform reach evaluated utilities 0.6464, 0.6584, and 0.5322, with sign agreement 66.7%, 90.0%, and 90.0%. These are repeatability diagnostics; RQ1 tests learned-policy efectiveness. Appendix E supplies costs and full controls.

$$
\mathrm { ~ --- ~ } \mathrm { A S C T - U n i f o r m } \mathrm { ~ \quad ~ } \mathrm { ~ --- ~ } \mathrm { A S C T - U C T } \mathrm { ~ \quad ~ } \mathrm { ~ --- ~ } \mathrm { A S C T - A g e n t U C T }
$$

![](images/0b4e2ab708d802f634d522098014daa7904f77e5ce88c4d7b05c3d04de2a35dc.jpg)

![](images/04208f2b7fee3cec875644e337db83b06227ba6cbdd0e62969ac6c93c4f7c8b6.jpg)  
Figure 6: Fixed-state evaluation on two questions, three frozen actor seeds, and two repeats per state. Lines are seed means; bands span seed means. Credit agreement compares signs across repeats. All budgets from 5 to 48 are retained; the guide marks 24.

Actor-rollout evaluation within the interface. With shared actor-centered credit and PPO settings, Uniform improves test utility over ActorRollout at every seed and uses 60.2% fewer auxiliary tokens (Table 7). ActorRollout balances roots and samples frozen-actor sufixes; this contrast tests the full sufix-evaluation mechanism, including allocation and continuation sampling.

Table 7: ASCT continuation control: shared $B = 1 2 ,$ credit, PPO settings, and 400 test QIDs. Full-run auxiliary tokens; mean ± sample SD over three seeds.
<table><tr><td>Continuation variant</td><td>Test F1</td><td></td><td>Test Utility Aux. tokens (M)</td></tr><tr><td>ASCT-ActorRollout</td><td> $0 . 7 1 4 2 \pm 0 . 0 0 9 0$ </td><td> $0 . 5 9 7 5 \pm 0 . 0 1 2 8$ </td><td> $4 9 5 . 2 4 \pm 1 3 . 2 9$ </td></tr><tr><td>ASCT-Uniform</td><td> $0 . 7 3 1 6 \pm 0 . 0 1 6 8$ </td><td> $0 . 6 1 4 6 \pm 0 . 0 1 9 1$ </td><td> $1 9 7 . 0 8 \pm 0 . 0 2$ </td></tr></table>

Learning from branch collections. Our $\mathrm { A T ^ { 2 } P O }$ adaptation performs close to PPO under the shared configuration (Appendix F). Branch records stay fixed within each update and refresh next iteration. This result concerns that adaptation without method-specific tuning.

## 7 Limitations

Evidence covers one backbone, a finite RAG library, three training seeds, 100-question transfer cohorts, and a constructed description probe. Recoverable execution and finite legal actions delimit applicability. Frozen-state repeatability diagnoses evaluation; it does not isolate each component’s contribution to policy gains. Cost projections use the stated auxiliary-token convention; time and price require calibration.

## 8 Conclusion

ASCT turns multi-step workflow comparisons into actor-centered credit for PPO on actor-sampled decisions. Three evaluators improve mean held-out utility; cost accounting and transfer studies characterize these gains. The learned policy executes without deployment search.

## Reproducibility statement

The experiment package retains per-question results, QID manifests, training ledgers, compact credit records, and fixed-state search traces. Appendices A–G document their scope and link reported quantities to the retained artifacts. Paper sources and vector figures accompany the manuscript. Recomputing retained results and reproducing training are separate procedures; training additionally requires the recorded models, datasets, and runtime dependencies.

## AI use statement

Following the authors’ directions, generative AI tools assisted with (1) identifying related work, (2) implementing experimental methods and code, (3) drafting manuscript sections and explanatory figures, and (4) revising wording. They also provided feedback on experimental methodology and interpretation of results, and assisted with reference, formatting, and code-test checks. The authors take responsibility for the final submission and its claims.

## References

Peter Auer, Nicolò Cesa-Bianchi, and Paul Fischer. Finite-time analysis of the multiarmed bandit problem. Machine Learning, 47:235–256, 2002. doi: 10.1023/A:1013689704352. URL https://homes.di.unimi.it/\~cesabian/Pubblicazioni/ml-02.pdf.

Ruike Cao, Shaojie Bai, Fugen Yao, Liang Dong, Jian Xu, and Li Xiao. ATPO: Adaptive tree policy optimization for multi-turn medical dialogue. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2603.02216.

Jaime Carbonell and Jade Goldstein. The use of MMR, diversity-based reranking for reordering documents and producing summaries. In International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 335–336, 1998. doi: 10.1145/290941.291025.

Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3-Embedding: Multi-linguality, multi-functionality, multi-granularity text embeddings through self-knowledge distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, 2024. URL https://aclanthology.org/2024.findings-acl.137/.

Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal rank fusion outperforms condorcet and individual rank learning methods. In International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759, 2009. doi: 10.1145/ 1571941.1572114.

Jakob Foerster, Gregory Farquhar, Triantafyllos Afouras, Nantas Nardelli, and Shimon Whiteson. Counterfactual multi-agent policy gradients. In AAAI Conference on Artificial Intelligence, 2018. URL https://arxiv.org/abs/1705.08926.

Bowei He, Yankai Chen, Xiaokun Zhang, and Xue Liu. Branching policy optimization: Sandboxnative language agent reinforcement learning. arXiv preprint arXiv:2607.14171, 2026.

Xanh Ho, Anh-Khoa Duong Nguyen, Saku Sugawara, and Akiko Aizawa. Constructing a multi-hop QA dataset for comprehensive evaluation of reasoning steps. In International Conference on Computational Linguistics, 2020. URL https://aclanthology.org/2020.coling-main.580/.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-rank adaptation of large language models. In International Conference on Learning Representations, 2022. URL https://arxiv.org/abs/2106.09685.

Yuxiang Ji, Ziyu Ma, Yong Wang, Guanhua Chen, Xiangxiang Chu, and Liaoni Wu. Tree search for LLM agent reinforcement learning. In International Conference on Learning Representations, 2026. URL https://arxiv.org/abs/2509.21240.

Amirhossein Kazemnejad, Milad Aghajohari, Eva Portelance, Alessandro Sordoni, Siva Reddy, Aaron Courville, and Nicolas Le Roux. VinePPO: Refining credit assignment in RL training of LLMs. In International Conference on Machine Learning, 2025. URL https://proceedings.mlr. press/v267/kazemnejad25a.html.

Levente Kocsis and Csaba Szepesvári. Bandit based Monte-Carlo planning. In European Conference on Machine Learning, pages 282–293, 2006. doi: 10.1007/11871842\_29.

Jing Yu Koh, Stephen McAleer, Daniel Fried, and Ruslan Salakhutdinov. Tree search for language model agents. arXiv preprint arXiv:2407.01476, 2024. URL https://arxiv.org/abs/2407.01476.

Yang Li, Hai Liu, Dian Shao, Yu Wang, Xiyu Chen, Sergey Volkov, Bozhi Wang, Ziyu Sun, Sihang Liu, Ye Luo, and Xiaowei Zhang. Agent-UCT: Upper confidence bounds applied to trees for agentic workflow optimization with cost-awareness. arXiv preprint arXiv:2607.24162v2, 2026.

Qwen Team. Qwen3-4B-Instruct-2507. Oficial model card, 2025. URL https://huggingface.co /Qwen/Qwen3-4B-Instruct-2507.

Stephen Robertson and Hugo Zaragoza. The probabilistic relevance framework: BM25 and beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal policy optimization algorithms. arXiv preprint arXiv:1707.06347, 2017.

Harsh Trivedi, Niranjan Balasubramanian, Tushar Khot, and Ashish Sabharwal. MuSiQue: Multihop questions via single-hop question composition. Transactions of the Association for Computational Linguistics, 2022. URL https://aclanthology.org/2022.tacl-1.31/.

Liang Wang, Nan Yang, Xiaolong Huang, Binxing Jiao, Linjun Yang, Daxin Jiang, Rangan Majumder, and Furu Wei. Text embeddings by weakly-supervised contrastive pre-training. arXiv preprint arXiv:2212.03533, 2022. URL https://arxiv.org/abs/2212.03533.

Liang Wang, Nan Yang, and Furu Wei. Query2doc: Query expansion with large language models. In Conference on Empirical Methods in Natural Language Processing, pages 9414–9423, 2023. URL https://aclanthology.org/2023.emnlp-main.585/.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William W. Cohen, Ruslan Salakhutdinov, and Christopher D. Manning. HotpotQA: A dataset for diverse, explainable multi-hop question answering. In Conference on Empirical Methods in Natural Language Processing, 2018. URL https://aclanthology.org/D18-1259/.

Tianhua Zhang, Kun Li, Junan Li, Yunxiang Li, Hongyin Luo, Xixin Wu, James Glass, and Helen Meng. TreePS-RAG: Tree-based process supervision for reinforcement learning in agentic RAG. arXiv preprint arXiv:2601.06922, 2026.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. URL https://arxiv.org/abs/2506.05176.

Andy Zhou, Kai Yan, Michal Shlapentokh-Rothman, Haohan Wang, and Yu-Xiong Wang. Language agent tree search unifies reasoning, acting, and planning in language models. In International Conference on Machine Learning, 2024. URL https://arxiv.org/abs/2310.04406.

Zefang Zong, Dingwei Chen, Yang Li, Qi Yi, Bo Zhou, Chengming Li, Bo Qian, Peng Chen, and Jie Jiang. AT<sup>2</sup>PO: Agentic turn-based policy optimization via tree search. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 2026. URL https://aclanthology.org/2026.acl-long.1106/.

## A Configuration and execution details

## A.1 Evaluator and update implementation

Search attention specifies how B trials are assigned to root actions and subsequent branches. Uniform selects the least-visited child at fully expanded nodes and samples expansions and continuations uniformly. UCT directs repeat visits using mean utility and an exploration bonus,

retaining uniform expansion and continuation. AgentUCT also accounts for predicted uncached continuation cost. At a fully expanded node $v ,$ its selection score is

$$
S ( v , a ) = \overline { { U } } ( v , a ) + c _ { \mathrm { e x p } } \sqrt { \frac { \log ( n ( v ) + 1 ) } { n ( v , a ) } - c _ { \mathrm { t o k } } \widehat { T } ( v , a ) , }\tag{7}
$$

where $\overline { { U } } ( v , a )$ is the backed-up empirical mean, n denotes visit counts, and $\widehat { T }$ predicts uncached continuation work. The coeficients control exploration and estimated cost. Expansion and continuation sample candidate actions with probability proportional to $\exp [ - c _ { \mathrm { t o k } } \widehat { T } ( v , a ) ]$ . The implementation averages predicted remaining cost across legal terminal sufixes. These inherited search rules determine the information acquired for Equations 3–4.

WTB restores shared materialized prefixes and executes only uncached sufixes, allowing counterfactual comparisons to reuse prior workflow computation. AgentUCT additionally uses cache-conditioned cost estimates to allocate trials; all ASCT evaluators and VinePPO share the reuse infrastructure. A partial hit restores the longest cached prefix and executes the remaining sufix. A terminal hit contributes a logical evaluation with zero new environment tokens. Each method, seed, and worker has an isolated cache namespace. The realized ledger measures executed work; the estimated cost guides allocation.

## A.2 Online PPO training and tree-free deployment

Each iteration collects fresh actor trajectories under $\pi _ { \mathrm { o l d } }$ , constructs their auxiliary credit, and updates the actor. Training standardizes the collected advantages within the iteration. With standardized credit $\widetilde { A } _ { E }$ and ratio $\rho _ { \theta } = \pi _ { \theta } ( a _ { t } \mid s ) / \pi _ { \mathrm { o l d } } ( a _ { t } \mid s )$ , PPO maximizes (Schulman et al., 2017)

$$
L ( \theta ) = \mathbb { E } \Big [ \operatorname* { m i n } \Big ( \rho _ { \theta } \widetilde { A } _ { E } , \mathrm { c l i p } ( \rho _ { \theta } , 1 - \epsilon , 1 + \epsilon ) \widetilde { A } _ { E } \Big ) \Big ] .\tag{8}
$$

The same frozen actor defines action sampling, baseline weights, and the PPO denominator. Thus $\rho _ { \theta } = \pi _ { \theta } ( a _ { t } \mid s ) / p _ { a _ { t } }$ starts at one before updating, avoiding an initial actor–search behavior mismatch. PPO uses actor-sampled trajectories with evaluator-derived credit.

The next iteration evaluates the updated actor’s visited states. Online here refers to this alternating collection-and-update procedure. Deployment selects actions from the trained actor and executes the workflow; auxiliary trees are used during training credit construction.

The workflow stages and legal choices appear in Table 1. The following tables specify the common learner and credit estimators.

The main-task data are mutually disjoint subsets of the oficial HotpotQA training partition: 2,000 training, 400 validation, and 400 test questions. Both held-out sets contain 76 easy, 256 medium, and 68 hard questions to preserve the training-pool dificulty proportions. Final iteration-3 checkpoints are evaluated with greedy actor actions, using the oficial answer scorer. We report means and sample standard deviations over seeds 11, 23, and 37.

Answer scoring. Training computes multiset token-overlap F1 after case folding and extracting lowercase alphanumeric tokens; articles are retained and there is no special yes/no/noanswer rule. Held-out HotpotQA reporting uses the oficial scorer: lowercase, punctuation deletion, article removal, whitespace normalization, and zero credit for a mismatched $\scriptstyle { \mathrm { y e s } } / { \mathrm { n o } } /$ noanswer answer. The reward definition used during collection is fixed across methods; oficial endpoint scores evaluate the resulting answers under the same reporting rule.

```latex
Algorithm 1 ASCT with actor-centered credit and PPO
Inputs: actor $\pi _ { \theta } ,$ training tasks $\mathcal { D } ,$ evaluator $E ,$ budget $B ,$ utility $U .$
1 for each training iteration do
2 Freeze $\pi _ { \mathrm { o l d } }  \pi _ { \theta } ;$ initialize actor-record bufer ${ \mathcal { R } } \gets { \mathcal { D } } .$
3 for each task in D do
4 Initialize workflow state s.
5 while s is nonterminal do
6 Compute $p _ { a } = \pi _ { \mathrm { o l d } } ( a \mid s )$ for $a \in A ( s ) ;$ sample $a _ { t } \sim p .$
7 Initialize an auxiliary tree rooted at a recoverable copy of $s .$
8 Run $B$ tree trials under $E ,$ covering all legal root actions first;
9 after each trial, back $\mathrm { u p }$ terminal $U ( w )$ and log realized auxiliary tokens.
10 Form $\{ \widehat { Q } _ { E , B } ( s , a ) \} _ { a \in \mathcal { A } ( s ) }$ from the per-root trial means.
11 Set $\begin{array} { r } { \widehat { V } _ { E } ( s ) = \sum _ { a } p _ { a } \widehat { Q } _ { E , B } ( s , a ) } \end{array}$ and $\widehat { A } _ { E } ( s , a _ { t } ) = \widehat { Q } _ { E , B } ( s , a _ { t } ) - \widehat { V } _ { E } ( s )$
12 Append $( s , a _ { t } , p _ { a _ { t } } , \widehat { A } _ { E } ( s , a _ { t } ) )$ to $\mathcal { R } ;$ advance s using $a _ { t } .$
13 end while
14 end for
15 Standardize advantages in $\mathcal { R } ;$ update θ with the PPO objective $\left( \operatorname { E q . 8 } \right)$
16 end for; return $\pi _ { \theta } .$
Auxiliary trials use an independent random stream. Policy updates use the sampled actor records in $\mathcal { R } .$
Deployment executes the trained actor.
```

Table 8: Shared training and evaluation configuration. Training uses the repository answer scorer; endpoint reporting uses the corresponding evaluation scorer.  
Setting Value   
Planner Qwen3-4B-Instruct-2507   
Adapter LoRA rank 4, alpha 8, dropout $0 ;$ query and value projections   
Optimizer AdamW; learning rate $1 0 ^ { - 5 }$   
Optimizer batch size 8   
PPO Two update epochs per iteration; clipping $\epsilon = 0 . 2$   
Advantage processing Standardization across collected decisions within each iteration   
Train / validation / test 2,000 / 400 / 400 mutually disjoint QIDs   
Held-out dificulty counts 76 easy, 256 medium, 68 hard in each set   
Seeds and iterations 11, 23, 37; three iterations per trained method   
Endpoint rule Final iteration-3 checkpoint; greedy actor decisions   
Auxiliary budget B = 12 terminal trials per actor-visited nonterminal state   
Workflow utility $\lambda = 0 . 1 , C _ { 0 } = 4 0 9 6$ execution words   
UCT exploration $c _ { \mathrm { e x p } } = 1 . 4$   
AgentUCT allocation cost $c _ { \mathrm { t o k } } = 1 0 ^ { - 4 }$ ; cache-conditioned sufix estimate   
Cost reporting $\beta = 1 0 ^ { - 4 }$ per auxiliary token: environment execution and auxiliary actor   
scoring

Table 9: Experimental methods. All trained rows use the same planner family and PPO update settings.
<table><tr><td>Method</td><td>Credit at a sampled actor decision</td><td>Auxiliary evaluation</td></tr><tr><td>Base</td><td>No policy update</td><td>None</td></tr><tr><td>PPO</td><td>Terminal trajectory utility</td><td>None</td></tr><tr><td>VinePPO</td><td>Adjacent Monte Carlo state values Actor continuations</td><td></td></tr><tr><td>ASCT-Uniform</td><td>Actor-centered action advantage</td><td>Uniform tree evaluator</td></tr><tr><td>ASCT-UCT</td><td>Actor-centered action advantage</td><td>Utility-directed UCT</td></tr><tr><td>ASCT-AgentUCT</td><td>Actor-centered action advantage</td><td>evaluator Cost-aware AgentUCT evaluator</td></tr></table>

The planner scores legal action labels and normalizes their probabilities over the current legal set. Collection samples actions from this distribution; evaluation takes its maximum. Advantages are standardized over each iteration’s collected decisions before PPO updates. Auxiliary samples update the estimator attached to those decisions.

The workflow execution counter uses lowercase alphanumeric word units. Retrieval charges corpus words for each query; reciprocal-rank fusion charges its four constituent rankings. Maximal marginal relevance and model reranking add the work defined by their passage sets. Generation charges the selected answer-context words. This common execution proxy is distinct from tokenizerbased model-call counts.

Workflow-adapted VinePPO. At each actor-visited nonterminal state, the frozen actor supplies B = 12 independently sampled continuations with replacement; their mean terminal utility is the stored state value. Credit is $r _ { t } + \widehat { V } \big ( s _ { t + 1 } \big ) - \widehat { V } \big ( s _ { t } \big )$ with unit discount. Intermediate rewards are zero; the final actor transition receives its workflow utility and has successor value zero. The stored estimate at the next actor-visited state supplies the preceding transition’s successor value, without an additional value-estimation call. Thus B counts auxiliary terminal samples per visited nonterminal state, not per side of a value diference. Main actor execution is separate. Cached terminal outcomes still count as logical samples; cached probabilities are reused, but action draws are fresh. WTB reuse and actor-scoring costs follow the declared ledger.

Legal-action probabilities. For each legal label, the planner scores its JSON action response under the same prompt. The score is the mean conditional log probability over response tokens (excluding prompt and padding tokens). A softmax over these length-normalized scores defines the categorical action distribution. Collection samples this distribution, and PPO uses the corresponding stored normalized action log probability; greedy evaluation selects its maximum. This scoring rule is shared by the compared actors.

## A.3 Fixed prompts and query refinement

Table 10 gives the core instructions used across methods. The actor prompt is fixed while its LoRA weights are trained; query and answer generation disable the adapter. The templates are implemented in code/main/src/atppo\_lab/policy.py. The table omits shared JSON-format reminders and retry messages, which remain in the released code.

Table 10: Fixed LLM instructions and their runtime inputs. Quoted instructions are taken from the implementation; output constraints are summarized.
<table><tr><td>Role / input</td><td>Core instruction</td><td>Output constraint</td></tr><tr><td>Actor / current workflow observation</td><td>“You select the next action in a RAG workflow.&quot;</td><td>JSON with one action field; probabilities normalized over legal labels.</td></tr><tr><td>Query2Doc / question</td><td>“Write a short hypothetical passage that would answer the JSON with one question.&quot;</td><td>pseudo_document string, used as the retrieval query.</td></tr><tr><td>Decomposition/ question</td><td>“Decompose the multi-hop question into exactly two focused search queries.&quot; Answer / question “Answer only from the evidence. Return the shortest</td><td>JSON queries: exactly two nonempty strings. JSON answer and</td></tr><tr><td>and evidence passages</td><td>answer span that satisfies the question.&quot;</td><td>citations; use UNKNOWN if evidence is insufficient and cite relevant evidence.</td></tr></table>

Deterministic refinement. Continue uses BridgeQueryRefiner in code/main/src/atppo\_ lab/rag.py, with no LLM call. It prioritizes evidence whose title occurs in the question, extracts a bridge entity with fixed text patterns, and falls back to an unseen evidence title, then the first title or the original question. Fixed templates target birth, death, government position, or location; otherwise the query is “Facts about {entity} needed to answer: {question}.” Fixed workflow rules implement the original-question action, Stop/Continue limits, retrieval widths, and context cutofs.

## B Token ledger and cost-adjusted value

The main-text auxiliary-token ledger separates environment execution and auxiliary actor scoring:

$$
T _ { \mathrm { a u x } } = T _ { \mathrm { e n v } } + T _ { \mathrm { a c t o r , a u x } } .\tag{9}
$$

The environment term expands as

$$
T _ { \mathrm { e n v } } = T _ { \mathrm { 4 B , i n } } + T _ { \mathrm { 4 B , o u t } } + T _ { \mathrm { q u e r y ~ e m b e d d i n g , 0 . 6 B } } + T _ { \mathrm { r e r a n k e r , 0 . 6 B } } .\tag{10}
$$

These are realized counts after reuse. Auxiliary actor scoring charges the recorded tokens for evaluating continuation-action probabilities. Main-trajectory actor scoring, optimizer forward passes, E5, BGE-M3, and shared index construction have separate counters. AgentUCT’s allocation estimate is a cache-conditioned proxy for remaining generative work; the reporting ledger includes the recorded embedding and reranking calls. All token categories use unit weight under the stated $\beta .$

Table 11: Auxiliary evaluation cost within ASCT. Totals are millions of environment tokens. Per-state values normalize within each seed. Search return charges realized auxiliary work as in Equation 5. Values are mean ± sample SD.
<table><tr><td>Evaluator</td><td></td><td>Total tokens (M) Tokens / actor state</td><td> $\overline { { J } } _ { \mathrm { s e a r c h } }$ </td></tr><tr><td>ASCT-Uniform</td><td> $1 9 7 . 0 7 7 \pm 0 . 0 2 4$ </td><td> $3 , 7 3 2 . 5 \pm 3 . 2$ </td><td> $0 . 4 9 7 2 \pm 0 . 0 0 2 2$ </td></tr><tr><td>ASCT-UCT</td><td> $1 9 7 . 5 8 8 \pm 0 . 0 6 8$ </td><td> $3 , 7 4 9 . 5 \pm 2 . 6$ </td><td> $0 . 5 2 0 3 \pm 0 . 0 0 3 1$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 1 9 3 . 0 8 8 \pm 0 . 4 0 4 }$ </td><td> $\mathbf { 3 , 6 6 4 . 5 \ : \pm { \ : 9 . 2 } }$ </td><td> $\mathbf { 0 . 5 2 2 0 \pm 0 . 0 0 1 6 }$ </td></tr></table>

Table 3 reports the combined ledger in Section 5.3. Figure 7 shows total auxiliary volume first, followed by the four environment categories for VinePPO and ASCT. PPO has zero auxiliary evaluation tokens; its main-trajectory and optimizer scoring remain part of training. The lower panels use a common method order and an independently labeled, zero-based scale for each category.

For method m and seed $^ { r , }$ the projection is computed before aggregation:

$$
J _ { \mathrm { d e p l o y } , m , r } ( N ) = U _ { m , r } ^ { \mathrm { t e s t } } - \beta T _ { m , r } ^ { \mathrm { a u x } } / N .\tag{11}
$$

Reported standard deviations are across these seed-level projections. For a comparison with positive mean utility diference $\Delta \overline { { U } }$ and positive mean token diference $\Delta \overline { { T } }$ , the crossing of mean curves is

$$
N _ { * } = { \beta \Delta \overline { { T } } } / { \Delta \overline { { U } } } .\tag{12}
$$

With total auxiliary cost, ASCT-AgentUCT versus VinePPO has $\Delta \overline { { U } } ~ = ~ 0 . 0 2 4 8 1 8 4 9$ and $\Delta \overline { { T } } = - 1 9 5 , 2 8 6 , 2 5 2 . 3 3$ tokens. Higher mean utility and lower mean cost give a positive projected value diference for every $N > 0$

Table 12: Projected final-policy value after amortizing total auxiliary tokens: $J _ { \mathrm { d e p l o y } } ( N )$ , mean ± sample SD over three seed-level projections. N is the assumed number of uses. These values correspond to Figure 5.
<table><tr><td>Method</td><td> $N = 1 0 0 { , } 0 0 0$ </td><td> $N = 5 0 0 { , } 0 0 0$ </td><td> $N = 1 { , } 0 0 0 { , } 0 0 0$ </td></tr><tr><td>PPO</td><td> $0 . 5 6 5 1 \pm 0 . 0 0 2 1$ </td><td> $0 . 5 6 5 1 \pm 0 . 0 0 2 1$ </td><td> $0 . 5 6 5 1 \pm 0 . 0 0 2 1$ </td></tr><tr><td>VinePPO</td><td> $0 . 2 0 5 5 \pm 0 . 0 0 4 6$ </td><td> $0 . 5 1 6 2 \pm 0 . 0 0 5 6$ </td><td> $0 . 5 5 5 0 \pm 0 . 0 0 5 9$ </td></tr><tr><td>ASCT-Uniform</td><td> $\mathbf { 0 . 4 1 7 5 \ : \pm 0 . 0 1 9 0 }$ </td><td> $\mathbf { 0 . 5 7 5 2 \pm 0 . 0 1 9 1 }$ </td><td> $\mathbf { 0 . 5 9 4 9 \ : \pm 0 . 0 1 9 1 }$ </td></tr><tr><td>ASCT-UCT</td><td> $\mathbf { 0 . 4 0 9 9 \pm 0 . 0 2 4 1 }$ </td><td> $\mathbf { 0 . 5 6 8 0 \ : \pm { \ : 0 . 0 2 4 1 } }$ </td><td> $\mathbf { 0 . 5 8 7 8 \pm 0 . 0 2 4 1 }$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 4 2 5 6 \pm 0 . 0 0 7 5 }$ </td><td> $\mathbf { 0 . 5 8 0 1 \ : \pm { \ : 0 . 0 0 7 6 } }$ </td><td> $\mathbf { 0 . 5 9 9 4 \ : \pm { \ : 0 . 0 0 7 6 } }$ </td></tr></table>

Environment tokens (M)  
(a) Auxiliary total and its two components  
![](images/e11920684568e2bcba39d82f0fcc62bbbfab100fc545123c04a0d9488612cdee.jpg)

(b) Generator input (4B)  
![](images/a1a8c8c99bcc21d62d8b7e967dcdbc2e0430c457b12e908530e2690df186c960.jpg)  
Environment tokens (M)

(c) Generator output (4B)  
![](images/521edf6cc514af7b24f0de24147554838dd96976e4b0e990a72d584a59378429.jpg)  
Environment tokens (M)

(d) Query embedding (0.6B)  
![](images/690e3ef2122d6b9725e6891e203e58a6f59a23925c657a65cfe38a3c2e71e98e.jpg)

(e) Reranker input (0.6B)  
![](images/43af29152a735baebab967a4e8f9b99a991e2bc80b66cc1b2a1d93b7d7b3fcac.jpg)  
Environment tokens (M)  
Figure 7: Auxiliary token volume for the recorded Qwen model operations over complete threeiteration training runs. (a) Stacked means sum environment calls and auxiliary actor scoring; endpoint whiskers give the sample SD of each seed’s summed count. Numbers at the right are totals. (b–e) Environment categories for VinePPO and the three ASCT evaluators, in the same top-to-bottom order. Each panel has an independent zero-based token axis. Diamonds and whiskers show mean ± sample SD across three seeds. M denotes one million tokens. Boldface identifies all ASCT variants.

## C Dificulty-resolved policy results

The test questions are grouped by their supplied HotpotQA dificulty labels. Each method–seed group contains the same 76 easy, 256 medium, and 68 hard questions. Utility uses the oficial answer scorer and the execution-word penalty in Equation 1. We compute each seed’s group mean before reporting the mean and sample SD across seeds.

Table 13: Final-policy utility by HotpotQA test dificulty, using the oficial answer scorer. Values are mean ± sample SD across three training seeds; Base is deterministic. The three dificulty groups partition the same 400 test questions used in Table 2.
<table><tr><td>Method</td><td>Easy (n = 76)</td><td>Medium (n = 256)</td><td>Hard (n = 68)</td></tr><tr><td>Base</td><td>0.6348</td><td>0.5723</td><td>0.4582</td></tr><tr><td>PPO</td><td> $0 . 6 4 0 5 \pm 0 . 0 0 2 3$ </td><td>0.5702 ± 0.0050</td><td> $0 . 4 6 1 6 \pm 0 . 0 0 5 3$ </td></tr><tr><td>VinePPO</td><td> $0 . 6 6 1 3 \pm 0 . 0 0 4 6$ </td><td>0.6031 ± 0.0108</td><td> $0 . 4 8 3 6 \pm 0 . 0 1 9 1$ </td></tr><tr><td>ASCT-Uniform</td><td></td><td>0.6565 ± 0.0045 0.6334 ± 0.0265 0.4968 ± 0.0097</td><td></td></tr><tr><td>ASCT-UCT</td><td></td><td>0.6540 ± 0.0056 0.6222 ± 0.0345</td><td> $\mathbf { 0 . 5 0 0 5 \ : \pm { \ : 0 . 0 0 9 7 } }$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $\mathbf { 0 . 6 5 8 3 \pm 0 . 0 1 1 1 }$ </td><td>0.6338 ± 0.0178</td><td> $\mathbf { 0 . 5 1 7 4 \ : \pm { \ : 0 . 0 1 2 0 } }$ </td></tr></table>

## D Learning interfaces, evaluator rules, and paired records

Figure 1 in Section 4 shows how evaluation enters learning. Figure 8 details the allocation rules.

Seed-level endpoints and paired utility contrasts are retained in the accompanying data files, including paired\_seed\_differences.csv.

The three training seeds are the units for variability reporting. Methods share the same test questions, enabling paired contrasts. The reported sample standard deviations describe observed variability; the smaller evaluator diferences have mixed signs across seeds.

![](images/d3bd9854a073e10482d513d2750a567afb0c68c5cbdae7572d30da719f762b70.jpg)  
Thick arrow: next choice (illustrative); shaded nodes: reusable prefixes

Figure 8: Decision rules within the three ASCT evaluators. UCT and AgentUCT share the depicted tree and prefix; the thick arrow difers only at one circled internal node, illustrating a possible local change in the next choice. Arrow thickness does not encode visit counts or the frequency or magnitude of cost efects. AgentUCT adds predicted uncached cost to selection and uses cost-weighted sampling. All three use WTB prefix reuse, back up terminal utility U, and return root-action means (Li et al., 2026).

## E Training credit and frozen-state allocation controls

Credit on the observed actor trajectories. We analyze the existing ASCT-Uniform training records for all three seeds: 2,000 training questions across three iterations give 6,000 trajectories per seed and 158,402 decision records in total. Each record contains the sampled action, frozen actor probabilities, legal-action Q table and raw advantage. We reconstruct the actor-weighted baseline and sampled-action advantage from these fields. Keeping the recorded Q table fixed, we also compute the sampled action’s advantage under a uniform-action baseline. Table 14 reports the resulting sign diferences on states with at least three legal actions. Both compared advantages must have magnitude above 10<sup>−8</sup> to count a sign change.

Across the three seeds, 3,117 of 36,000 multi-action decisions change sign (8.66%). These are actions that actually entered training, extending the mechanism analysis beyond selected counterfactual examples. The calculation identifies the local comparison supplied by actor-centered credit. Final learning outcomes are evaluated by the complete-method training comparisons in Section 5.2; the alternative baseline here is an ofline recomputation.

Root-stratified controls on common five-action states. The fixed-state follow-up uses the original training split’s question indices 5 and 6. Each completed iteration-3 ASCT-Uniform actor (seeds 11, 23, and 37) supplies its sampled retriever state, with five legal root actions. Evaluators share its materialized prefix, actor probabilities and sampled action. Every run starts with a fresh WTB namespace and reset transient ranking/reranking caches; WTB reuse remains enabled within the run. Prefix reconstruction and index preparation precede measurement. Each evaluator has two repeats of 48 terminal trials, retaining every intermediate Q table and executed token increment. The 12-trial comparison uses the corresponding prefixes.

Table 14: Baseline semantics on actual ASCT-Uniform training decisions. Counts cover three training iterations per seed. Multi-action states have at least three legal actions. Sign changes compare actor-weighted and uniform-action centering of the same recorded Q table at the actual sampled action.
<table><tr><td></td><td></td><td>Seed All decisions Multi-action decisions Sign changes Rate (%)</td><td></td><td></td></tr><tr><td>11</td><td>52,828</td><td>12,000</td><td>1,043</td><td>8.69</td></tr><tr><td>23</td><td>52,833</td><td>12,000</td><td>1,021</td><td>8.51</td></tr><tr><td>37</td><td>52,741</td><td>12,000</td><td>1,053</td><td>8.78</td></tr></table>

Root MC balances root trials and draws each subsequent legal action independently and uniformly. Uniform tree evaluation uses uniform expansion and rollout sampling but balances visits at internal nodes. This comparison tests the allocation structure under the same sufix-sampling rules and cache availability. Internal allocation changes which sufixes are evaluated; it need not estimate the same fixed continuation distribution as independent MC.

Table 15: Internal tree allocation at the training budget (B = 12). Both evaluators cover legal root actions, use uniform expansion/rollout sampling, and retain WTB reuse. Two training questions provide five-action retriever states for each of three frozen actors; two repeats per state. Cells report mean ± sample SD across seed means. Q SD and credit-sign agreement measure repeatability.
<table><tr><td>Evaluator</td><td></td><td>Repeat Q SD Credit agreement (%) Tokens (k)</td><td></td></tr><tr><td>Root-stratified MC</td><td> $0 . 1 2 0 3 \pm 0 . 0 6 5 3$ </td><td> $6 3 . 3 \pm 1 1 . 5$ </td><td> $1 1 . 0 9 \pm 0 . 1 3$ </td></tr><tr><td>ASCT-Uniform</td><td> $0 . 0 6 1 9 \pm 0 . 0 2 9 3$ </td><td> $8 0 . 0 \pm 1 0 . 0$ </td><td> $1 1 . 2 8 \pm 0 . 5 1$ </td></tr></table>

Table 16: Root-stratified MC and Uniform tree evaluation on the same five-action states. All values are mean ± sample SD across three seed means. Each state has two repeats; budgets are prefixes of the completed 48-trial records.
<table><tr><td>B Evaluator</td><td></td><td>Repeat Q SD Credit (%) Tokens (k)</td><td></td></tr><tr><td>12 Root-stratified MC</td><td></td></tr><tr><td> $1 1 . 0 9 \pm 0 . 1 3$ </td><td> $0 . 1 2 0 3 \pm 0 . 0 6 5 3$   $6 3 . 3 \pm 1 1 . 5$ </td></tr><tr><td>12 ASCT-Uniform 48 Root-stratified MC</td><td> $0 . 0 6 1 9 \pm 0 . 0 2 9 3$   $8 0 . 0 \pm 1 0 . 0$   $1 1 . 2 8 \pm 0 . 5 1$ </td></tr><tr><td> $0 . 0 3 6 8 \pm 0 . 0 0 8 8$ </td><td> $8 3 . 3 \pm 5 . 8$   $3 8 . 1 5 \pm 1 . 0 3$ </td></tr></table>

At 12 trials, repeat Q SD decreases by 48.6% and credit-sign repeat agreement increases by

16.7 percentage points with internal allocation, at 1.7% greater executed token cost. At 48 trials, Uniform’s Q SD is 0.0484 versus root MC’s 0.0368, while sign agreement is 90.0% versus 83.3%. The observed repeatability benefit applies to the small budget used in training; the ordering depends on budget. Q SD measures variation across repeated evaluations and credit agreement measures sign repeatability; neither uses an accuracy target.

Four-question binary-state corroboration. A separate completed diagnostic uses the first four training questions and the first control and reranker state visited by each of the same three actors: eight binary-action states per seed, 24 checkpoint–state combinations in total. Five evaluators each make four repeats of 12 trials. Root MC balances root trials with uniform sufixes; Actor MC balances roots with frozen-actor sufixes. Two independent 96-trial Actor-MC batches provide a pooled actor-continuation reference. The resulting 528 evaluations retain complete Q tables, terminal paths, utilities and cost counters.

Table 17: Four-question binary-state diagnostic: eight states per actor seed, four repeats at $B = 1 2$ Order agreement uses two independent 96-trial actor-continuation batches as its reference. Total auxiliary tokens include actor scoring. Values are mean ± sample SD across three seed means.
<table><tr><td>Evaluator</td><td></td><td>Repeat Q SD Ref. order (%) Tokens (k)</td><td></td></tr><tr><td>Root-stratified MC</td><td> $0 . 0 1 4 9 \pm 0 . 0 2 2 1$ </td><td> $9 7 . 9 \pm 3 . 6 $ </td><td> $3 . 9 0 \pm 0 . 3 1$ </td></tr><tr><td>ASCT-Uniform</td><td> $0 . 0 0 6 4 \pm 0 . 0 0 9 3$ </td><td> $9 9 . 0 \pm 1 . 8 $ </td><td> $4 . 1 6 \pm 0 . 3 1$ </td></tr><tr><td>ASCT-UCT</td><td> $0 . 0 0 9 3 \pm 0 . 0 1 4 8$ </td><td> $9 6 . 9 \pm 5 . 4$ </td><td> $4 . 1 8 \pm 0 . 3 2$ </td></tr><tr><td>ASCT-AgentUCT</td><td> $0 . 0 0 8 5 \pm 0 . 0 1 2 1$ </td><td> $9 9 . 0 \pm 1 . 8 $ </td><td> $4 . 0 7 \pm 0 . 3 7$ </td></tr><tr><td>Actor MC</td><td> $0 . 0 0 3 8 \pm 0 . 0 0 4 0$ </td><td> $9 9 . 0 \pm 1 . 8 $ </td><td> $4 . 7 8 \pm 0 . 5 3$ </td></tr></table>

Uniform tree evaluation reduces repeat Q SD relative to root MC in every actor seed, from 0.0149 to 0.0064 on average, using 4,158 versus 3,899 environment tokens. The order-agreement ceiling reflects this binary cohort: both reference batches agree on the action order at every state. For two actions, any nondegenerate convex baseline lies between their Q values; comparing actor and uniform baselines therefore cannot reveal a sign diference. The actual multi-action training decisions above supply the relevant centering evidence.

Functional cost-phase controls. On the five-action follow-up states, eight settings independently enable cost awareness in selection, expansion and rollout. The all-of setting is UCT and the all-on setting is native AgentUCT. With Uniform and root MC as controls, the design contains 120 runs and 5,760 terminal trials. For each state, a separate 240-leaf enumeration weights terminal paths by the product of uniform legal-action probabilities, producing a reference for that named continuation distribution. Six enumerations add 1,440 reference terminals. The earlier one-question pilot retains 30 runs and three further enumerations in the evidence package.

At 48 trials, full AgentUCT reduces mean executed tokens by 3.4% relative to UCT and improves credit-sign repeat agreement from 66.7% to 90.0%. At 12 trials its token count is 3.7% higher. The switches therefore afect the executed search and its local comparisons, with efects that depend on phase combinations and budget. All per-trial records are retained, including reference discrepancies and selection changes. The reference describes uniform-continuation values; discrepancies for adaptive evaluators are reference alignment, not universal Q error. These controls make no policy updates.

Budget curves from completed evaluations. The original training records use 12 terminal evaluations per state and contain endpoint action tables. For longer budget curves, we use the completed 48-trial fixed-state evaluations on two questions already in the same training split (ordered indices 5 and 6). Each of the three iteration-3 actors supplies its observed five-action retriever state. Two repeated searches per method retain every terminal utility, cumulative executed token count, and reconstructed action table. Prefixes of these records provide every budget from 5 to 48, including 24, with no additional environment execution or policy updates. The same materialized prefix and actor probabilities are used across evaluators within each state. Fresh per-evaluation WTB namespaces and reset transient retrieval/reranker caches isolate auxiliary execution; prefix preparation is outside the measured ledger.

Table 18: Cost-aware selection, expansion and rollout $\mathrm { ( S / E / R ) }$ on the same two questions and three actors. Bits enable cost awareness at each phase; 000 is UCT and 111 is native AgentUCT. Cells are mean ± sample SD across seed means, with two repeats per state. WTB reuse is enabled throughout.
<table><tr><td> $\mathrm { S / E / R }$ </td><td>Tokens@12 (k) Tokens@48 (k)</td><td></td><td>Q SD@48 Credit@48 (%)</td><td></td></tr><tr><td>UCT (000)</td><td> $1 1 . 1 4 \pm 0 . 6 3$ </td><td></td><td> $3 5 . 6 6 \pm 1 . 3 6 0 . 0 6 0 9 \pm 0 . 0 1 5 0$ </td><td> $6 6 . 7 \pm 5 . 8$ </td></tr><tr><td>001</td><td> $1 0 . 9 4 \pm 0 . 6 6$ </td><td></td><td> $3 5 . 9 3 \pm 0 . 7 3 0 . 0 5 2 8 \pm 0 . 0 1 2 8$ </td><td> $8 0 . 0 \pm 2 0 . 0$ </td></tr><tr><td>010</td><td> $1 1 . 6 9 \pm 1 . 5 5$ </td><td></td><td> $3 7 . 0 1 \pm 1 . 9 6 0 . 0 5 2 6 \pm 0 . 0 1 2 3$ </td><td> $7 6 . 7 \pm 1 5 . 3$ </td></tr><tr><td>011</td><td> $1 1 . 5 5 \pm 1 . 2 9$ </td><td></td><td> $3 4 . 9 0 \pm 2 . 2 4 0 . 0 4 9 2 \pm 0 . 0 2 1 9$ </td><td> $8 0 . 0 \pm 1 0 . 0$ </td></tr><tr><td>100</td><td> $1 1 . 2 4 \pm 0 . 5 7$ </td><td></td><td> $3 4 . 9 4 \pm 2 . 1 2 0 . 0 4 7 1 \pm 0 . 0 1 9 7$ </td><td> $7 0 . 0 \pm 1 7 . 3$ </td></tr><tr><td>101</td><td> $1 1 . 0 1 \pm 0 . 5 0$ </td><td></td><td> $3 6 . 1 5 \pm 0 . 6 8 0 . 0 4 5 0 \pm 0 . 0 0 7 4$ </td><td> $7 6 . 7 \pm 5 . 8$ </td></tr><tr><td>110</td><td> $1 1 . 7 7 \pm 1 . 5 2$ </td><td></td><td> $3 6 . 2 9 \pm 1 . 1 6 0 . 0 5 1 9 \pm 0 . 0 0 2 9$ </td><td> $7 3 . 3 \pm 2 3 . 1$ </td></tr><tr><td>AgentUCT (111)</td><td> $1 1 . 5 6 \pm 1 . 4 7$ </td><td></td><td> $3 4 . 4 6 \pm 0 . 7 5 0 . 0 5 0 3 \pm 0 . 0 2 3 9$ </td><td> $9 0 . 0 \pm 1 0 . 0$ </td></tr><tr><td>Uniform</td><td> $1 1 . 2 8 \pm 0 . 5 1$ </td><td></td><td> $3 6 . 9 8 \pm 1 . 6 7 0 . 0 4 8 4 \pm 0 . 0 1 5 8$ </td><td> $9 0 . 0 \pm 1 7 . 3$ </td></tr><tr><td>Root MC</td><td> $1 1 . 0 9 \pm 0 . 1 3$ </td><td></td><td> $3 8 . 1 5 \pm 1 . 0 3 0 . 0 3 6 8 \pm 0 . 0 0 8 8$ </td><td> $8 3 . 3 \pm 5 . 8$ </td></tr></table>

At budget K, mean evaluated utility is $\begin{array} { r } { K ^ { - 1 } \sum _ { i = 1 } ^ { K } U ( w _ { i } ) } \end{array}$ . Credit repeat agreement is the fraction of legal actions whose centered-credit signs agree between the two independent searches at that budget. It measures repeatability, with final-policy utility assessed separately in the training experiments. These ASCT evaluators use rule-based continuations and add no auxiliary actorscoring calls, so $T _ { \mathrm { a u x } } ( K ) = T _ { \mathrm { e n v } } ( K )$ . Equation 5 gives $\overline { { J } } _ { \mathrm { s e a r c h } } ( K )$ using $\beta = 1 0 ^ { - 4 }$ . We average repeats and questions within each seed before reporting seed means. The bands in Figures 6 and 9 span the three seed-level means; they are descriptive ranges.

Table 19: The 24-evaluation checkpoint of the full curves. Values average the same two questions, two repeats and three actor seeds. The complete curves retain seed-level variability.
<table><tr><td>Evaluator</td><td>Mean utility</td><td>Tokens</td><td> $\overline { { J } } _ { \mathrm { s e a r c h } }$ </td><td>Credit agreement (%)</td></tr><tr><td>ASCT-Uniform</td><td>0.5449</td><td>21,077.9</td><td>0.4571</td><td>76.7</td></tr><tr><td>ASCT-UCT</td><td>0.5887</td><td>20,788.5</td><td>0.5021</td><td>76.7</td></tr><tr><td>ASCT-AgentUCT</td><td>0.5960</td><td>20,531.3</td><td>0.5105</td><td>73.3</td></tr></table>

Observed relation. At 24 evaluations, ASCT-AgentUCT has the highest mean auxiliary utility and $\overline { { J } } _ { \mathrm { s e a r c h } }$ among the three evaluators, while its credit repeat agreement is lower (Table 19).

![](images/5f9a3e4dc29325b8e4cd1bfb6f8f9c47367a2fe741114d325e6ec50660d5c539.jpg)

![](images/8999b6e160a7f456d4a47eec6db470bdd3dfec44b17251d381bab64988903449.jpg)  
Figure 9: Executed auxiliary tokens and cost-adjusted evaluation value on the same states and budgets as Figure 6. $\overline { { J } } _ { \mathrm { s e a r c h } } ( K ) = \overline { { U } } ( K ) - 1 0 ^ { - 4 } T _ { \mathrm { a u x } } ( K ) / K$ . Auxiliary actor scoring is zero for these three evaluators, so total auxiliary cost equals environment cost. The deployment amortization curve uses Equation 6. Lines and bands follow Figure 6.

At 48 evaluations, UCT and AgentUCT reach mean utilities 0.6464 and 0.6584, compared with 0.5322 for Uniform. Their corresponding $\overline { { J } } _ { \mathrm { s e a r c h } }$ values are 0.5721, 0.5866, and 0.4551. Credit repeat agreement is 66.7%, 90.0%, and 90.0%, respectively. Thus, directing evaluation toward higher-utility continuations and producing repeatable action comparisons are distinct properties. These exploratory curves describe information acquisition on two existing training questions. The main experiments measure the learned policy’s quality and the aggregate execution cost across training.

## F Component protocol and continuation diagnostics

Description controls. Component A prioritizes passages tagged from answer-bearing annotations in a constructed corpus. The four descriptions vary whether its type and evidence preference are stated correctly, only its construction is stated, no description is given, or the opposite preference is stated. Its implementation, name, corpus, actor weights, and existing component descriptions stay fixed. The correct description also explains the tags, so the comparison does not isolate one wording feature. Per-question answer outcomes are retained in the accompanying data package.

What makes the credit useful? For a fixed table, the local surrogate $\begin{array} { r } { L _ { s } ( \theta ) = \sum _ { a } \pi _ { \theta } ( a } \end{array}$ $s ) \widehat { Q } _ { E , B } ( s , a )$ has logit derivative $p _ { a } \widehat { A } _ { E } ( s , a )$ at the behavior policy. This explains the direction of the raw credit before normalization and clipping. It benefits actor utility when evaluator preferences align with outcomes the actor can realize. The continuation rule determines which downstream outcomes contribute to each action value. Online recollection refreshes visited states and actor-weighted baselines after each training iteration.

Continuation control. ASCT-Uniform uses uniform tree expansion and continuation with least-visited selection; ASCT-ActorRollout balances root actions and samples sufix actions from the frozen actor. Both retain root coverage and actor-centered credit. This intervention changes the full continuation mechanism, and each condition trains its own evolving actor. Table 7 reports the three-seed held-out endpoints. The matched-QID ASCT-ActorRollout minus ASCT-Uniform test-utility diferences are −0.0077, −0.0294, and −0.0143 in seeds 11, 23, and 37. All 18,000 Actor-CF training trajectories were retained; reconstructing the Q-table centering for 158,357 actor decisions found no protocol violations. The audit verifies that recorded credits match the action values and actor-weighted baselines.

Across complete training runs, ASCT-ActorRollout uses $4 9 5 . 2 4 \pm 1 3 . 2 9$ million auxiliary tokens: 180.12 million in environment execution and 315.12 million in auxiliary actor scoring on average. Main-trajectory and optimizer actor scoring are excluded, as in the shared ledger. Its mean auxiliary rollout utility is 0.5488.

AT<sup>2</sup>PO workflow adaptation. We adapted the public tree learner (Zong et al., 2026) to categorical workflow actions under the planned training setup, with batch size eight and three collect–update iterations. Each iteration collects trees with the current actor and optimizes over that collection; the next iteration collects again with the updated actor. Each training question produces a 106-leaf tree whose branch records enter optimization. The training trajectories remain fixed during each update and are refreshed at the next collection.

Independent iteration-3 evaluation on the paper’s test QIDs with the oficial scorer gives test utility $0 . 5 6 4 7 { \scriptstyle \pm 0 . 0 0 4 2 }$ and F1 $0 . 6 9 4 5 { \scriptstyle \pm 0 . 0 0 4 1 }$ across three seeds. Relative to $\mathrm { P P O } \left( 0 . 5 6 5 1 \pm 0 . 0 0 2 1 \right)$ , the paired test-utility diference is $- 0 . 0 0 0 4 \pm 0 . 0 0 6 3$ . Auxiliary tokens total $6 9 6 . 1 5 \pm 1 9 . 5 3$ million per run; optimizer scoring is separate. This is an exploratory evaluation of one adaptation without a method-specific tuning study. The adaptation difers from ASCT in its evaluation unit, credit construction, and policy-update records.

## G Retained evidence

The accompanying data package retains matched per-QID endpoints, seed-level auxiliary ledgers, compact credit records, and the component and transfer probes. For the new controls, the compact supplement includes three-seed matched-QID predictions, training metrics, and protocol audits. Full Actor-CF and $\mathrm { A T ^ { 2 } P O }$ trajectory archives are retained separately by the authors. These artifacts support recomputation of reported results; reproducing training additionally requires the recorded models, indices, and runtime.