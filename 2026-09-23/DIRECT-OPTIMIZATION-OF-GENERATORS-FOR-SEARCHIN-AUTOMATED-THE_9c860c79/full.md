# DIRECT OPTIMIZATION OF GENERATORS FOR SEARCHIN AUTOMATED THEOREM PROVING

Adam Ousherovitch Department of Statistics University of Michigan aoushero@umich.edu

Ambuj Tewari Department of Statistics University of Michigan tewaria@umich.edu

## ABSTRACT

Fine-tuned Large Language Models (LLMs) significantly advance Automated Theorem Proving (ATP), but are often deployed as guiding policies within tree search rather than for single-attempt generation. Recent work shows cross entropy is suboptimal for an LLM used in flat search strategies such as aggregation or filtering and that work has developed new loss functions to correct this misalignment. Extending this alignment to tree search is more challenging: proof discovery depends on exploration and recovery through off-trace states that supervised demonstrations do not reveal. We extend Compute-Aligned Training (CAT) to this setting through an abstraction of policy-guided search, deriving tractable, trace-supported losses. Alongside these search-aware losses, we introduce a search-agnostic uniform-allocation (UA) loss that accounts for the budget without specifying the specific search. Both induce scalar weights on per-tactic cross-entropy gradients. We characterize how off-trace behavior affects the search-aware weights, including conditions for vanishing approximation error at large budgets. On a Lean benchmark, both approaches achieve higher observed proof-success rates than cross-entropy across six search strategies, with strong results from a single shared UA adapter. Budget sweeps show larger gains over cross-entropy at 256 than at 16 expansions, implying CAT scales with test time compute.

## 1 INTRODUCTION

Automated Theorem Proving (ATP) is an important frontier in Artificial Intelligence (AI) (Yang et al., 2024), with applications in software and hardware verification (Bengio & Malkin, 2024). Formal verifiers such as Lean 4 (de Moura & Ullrich, 2021) allow generated proofs to be checked, but proof generation remains difficult because rewards are sparse and horizons are long (Poesia et al., 2024). Large Language Models (LLMs) predict proof steps (Polu & Sutskever, 2020), yet a single flawed step can invalidate an entire attempt. Recent approaches therefore pair LLMs with structured search (Lample et al., 2022; Polu et al., 2022; Song et al., 2024), which retains partial proofs and explores alternative continuations rather than requiring every step to succeed on the first attempt.

LLMs driving search are typically trained through Supervised Fine-Tuning (SFT) with Cross-Entropy (CE). Recent work shows that CE can be suboptimal when deployment uses search (Ousherovitch & Tewari, 2026; Chen et al., 2025), but existing corrections focus on flat strategies such as candidate aggregation (Wang et al., 2023) or filtering (Chen et al., 2021; Cobbe et al., 2021). Extending alignment to tree search is fundamentally more difficult because success probability includes unsuccessful exploration followed by recovery, and depends on how compute is allocated across intermediate proof states. An exact objective therefore requires off-trace continuations that demonstrated proof traces do not supply.

We address this gap by formalizing policy-guided search and constructing tractable Compute-Aligned Training (CAT) losses for this kind of search. Given a demonstrated proof trace, we restrict the modeled search to observed states and collapse deviations into a trace-miss event. This yields search-aware objectives that preserve selected aspects of the strategy’s compute allocation. Their gradients are weighted sums of the usual per-tactic CE gradients, emphasizing steps according to their influence on modeled proof success.

Alongside these objectives, we introduce a simple search-agnostic uniform-allocation (UA) loss. UA assigns a uniform local compute budget to each demonstrated step, accounting for repeated sampling without specifying the deployed search rule. It thus provides a shared training objective for use across search algorithms, rather than an approximation tied to one particular deployed strategy.

Trace support introduces approximation error because demonstrations omit alternative proofs and the compute consumed by off-trace exploration. We characterize these effects through the per-tactic gradient weights and identify when search-aware weights scale better than CE with more budget.

On a Lean benchmark, both strategy-specific CAT and UA attain higher accuracy than an epochmatched CE control under all six evaluated strategies (Table 1). A shared UA adapter performs strongly across strategies, while search-aware CAT yields an additional observed gain under Pass@N. Experiments also show larger gains over CE at N = 256 than at N = 16 (Table 2).

## 1.1 RELATED WORK

Automated Theorem Proving. The application of deep learning to ATP has rapidly evolved alongside LLMs (Polu & Sutskever, 2020; Han et al., 2022). Modern neural provers target interactive verification environments like Lean (de Moura & Ullrich, 2021; Zheng et al., 2022), with openweight milestones such as Lean Copilot (Song et al., 2024), Llemma (Azerbayev et al., 2023), and ReProver (Song et al., 2025) powered almost exclusively by SFT. Our work is strictly situated within this SFT paradigm. We explicitly scope our analysis away from systems that interleave informal natural-language reasoning with formal verification (Jiang et al., 2023; Wu et al., 2022), as well as online Reinforcement Learning (RL) pipelines (DeepMind, 2024).

Test-Time Search Algorithms. The use of search in ATP falls in the wider trends of utilizing inference-time compute (Snell et al., 2024). Search in ATP is diverse. It includes uninformed traversal methods like Depth-First Search (DFS) (Pearl, 1984) or Pass@N (Chen et al., 2021), and the more complex policy-guided searches like Best-First Search (BFS) (Hart et al., 1968), most notably Levin Search and its modern neural adaptations (Orseau et al., 2018; Orseau & Lelis, 2021; Xin et al., 2025). While prior literature has extensively benchmarked these algorithms in isolation, it treats the language model as a static, search-agnostic prior. Crucially, the literature has never investigated how the structural assumptions of these distinct search rules interact with the training of the policy itself.

Test-Time Aligned Training. A growing body of work recognizes that standard supervised objectives fail to align models with inference-time search (Ousherovitch & Tewari, 2026; Chen et al., 2025; Chow et al., 2025; Balashankar et al., 2025; Tang et al., 2025). However, these efforts focus overwhelmingly on non-branching strategies where the global search objective can be expressed a an exact function of observed generation probabilities. Within ATP, the sole existing attempt at testtime aligned training is Direct Coverage Optimization (DCO) (Chen et al., 2025), which focused exclusively on Pass@N. Because the global success probability of Pass@N can be expressed as a closed-form function of on-path tactic probabilities, DCO sidesteps the structural approximation problem entirely. By contrast, state-of-the-art theorem provers rely on branching strategies whose objectives cannot be resolved purely from on-path data. This paper bridges that gap.

## 1.2 CONTRIBUTIONS

• A Framework for Search-Aligned Training. We formalize policy-guided search and derive tractable losses for branching strategies. We also develop a budget-based UA loss independent of the specific search’s structure. Both approaches induce scalar weights on per-tactic CE gradients.

• Characterizing the Trace-Supported Approximation. For Pass@N and best-first search, we analyze how unobserved alternative proofs and off-trace compute consumption affect searchaware gradient weights, including their dependence on the test-time budget.

• Empirical Evaluation in Lean. Across six search strategies, strategy-specific CAT and UA each attain higher success rates than CE. We also evaluate gains across deployment budgets.

## 2 A GENERAL OBJECTIVE FOR SEARCH-ALIGNED TRAINING

This section formalizes LLM-guided search and derives a general aligned objective and the searchagnostic UA objective. We assume familiarity with Interactive Theorem Proving (ITP).<sup>1</sup>

## 2.1 SETUP

Let the environment be a state space X (proof states), an action space ${ \mathcal { V } } ,$ and a transition $\tau$ : $\mathcal X \times \mathcal y  \mathcal x \cup \{ x _ { e r r } \}$ , where $x _ { e r r }$ is the state produced by incorrect tactics. The LLM defines a stochastic policy $\pi _ { \theta } ( y \mid x )$ . A dataset D provides valid traces; a single trace is

$$
\tau ^ { * } = ( x _ { 0 } , y _ { 1 } ^ { * } , x _ { 1 } , \ldots , x _ { L - 1 } , y _ { L } ^ { * } , x _ { L } ) , \qquad x _ { L } = x _ { Q E D } ,
$$

where each demonstrated tactic ${ \boldsymbol y } _ { t } ^ { * }$ moves the environment from $x _ { t - 1 } \ 1 0 \ x _ { t }$ . Write $p _ { t } ~ = ~ \pi _ { \theta } ( y _ { t } ^ { * } \mid$ $x _ { t - 1 } )$ . Standard SFT with CE maximizes the likelihood of these traces:

$$
L _ { C E } ( \theta ) = - \sum _ { t = 1 } ^ { L } \log \pi _ { \theta } ( y _ { t } ^ { * } \mid x _ { t - 1 } ) = - \sum _ { t = 1 } ^ { L } \log p _ { t } , \qquad L _ { C E } ^ { ( t ) } : = - \log p _ { t } .\tag{1}
$$

At test time, the policy is deployed inside a search algorithm $\mathcal { A }$ with a compute budget. We wish to optimize the probability that ${ \mathcal { A } } ,$ using $\pi _ { \theta } .$ , finds a proof within that budget.

## 2.2 SEARCH AS CONFIGURATION DYNAMICS

A is defined by its interaction with memory, which we define via configurations: $c = \langle \mathcal { F } , M \rangle$

The frontier F holds the nodes available for expansion with a heuristic score (insertion order, internal confidence, etc.); its elements are tuples $( x , s )$ with $x \in \mathcal { X } , s \in \mathbb { R }$ We may write $x \in { \mathcal { F } }$ for $( x , s ) \in { \mathcal { F } }$ for notational ease. M holds everything else: the policy, the remaining budget, the unavailable nodes, etc. Budget is most naturally interpreted as either tactic expansions or rollouts. Budget handling can be encoded in the specification of the strategy.

$\mathcal { A }$ selects a node $x \in { \mathcal { F } }$ , samples a tactic $y \sim \pi _ { \boldsymbol { \theta } } ( \cdot \mid x )$ , executes it, and updates its configuration from the observation $z = \mathcal { T } ( x , y )$ . Excluding the initialization of $c _ { 0 }$ from $x _ { 0 }$ and the compilation of the returned proof from $M , A$ is fully specified by two operations,

$$
S _ { A } ( x \mid c ) , \qquad U _ { A } ( c , x , z ) ,
$$

a (possibly stochastic) selection rule $S _ { A }$ and a deterministic update rule $U _ { A }$

A configuration is solved if it contains a completed proof, and dead if its budget is exhausted. Let $J ^ { A } ( c )$ be the true probability that A, started from c, ever discovers a proof. It satisfies

$$
J ^ { A } ( c ) = \left\{ \begin{array} { l l } { { \mathrm { 1 i f ~ } c \mathrm { ~ s o l v e d } ; } } & { { \mathrm { 0 ~ i f ~ } c \mathrm { ~ d e a d ~ ( a n d ~ u n s o l v e d ) } , } } \\ { { \sum _ { x \in \mathcal { F } ( c ) } S _ { A } ( x \mid c ) \sum _ { y \in \mathcal { Y } } \pi _ { \theta } ( y \mid x ) J ^ { A } \big ( U _ { A } ( c , x , \mathcal { T } ( x , y ) ) \big ) , } } & { { \mathrm { o t h e r w i s e } . } } \end{array} \right.\tag{2}
$$

and the deployed objective is $\mathcal { T } ^ { A } ( \theta ) = J ^ { A } ( c _ { 0 } )$ , where $c _ { 0 }$ encodes the start state $x _ { 0 }$ and the full budget. Returning any valid trajectory from $x _ { 0 }$ to $x _ { Q E D }$ counts as success.

Equation (2) is exact but cannot be evaluated from a supervised trace. Consider expanding a frontier node on the trace, say $x = x _ { t - 1 }$ . The tactic space partitions into three disjoint sets: the demonstrated tactic $\{ y _ { t } ^ { * } \}$ , the invalid tactics $\mathcal { T } _ { t } : = \{ y : \mathcal { T } ( x _ { t - 1 } , y ) = x _ { e r r } \}$ , and the valid but off-trace tactics $\mathcal { O } _ { t } : = \{ \bar { y } : \mathcal { T } ( x _ { t - 1 } , y ) \notin \{ x _ { e r r } , x _ { t } \} \}$ . With off-trace states defined as $x _ { t , y } = \mathscr { T } ( x _ { t - 1 } , y )$ for $y \in { \mathcal { O } } _ { t }$ , the sum in Equation (2) then expands to:

$$
\begin{array} { r l } & { \displaystyle \sum _ { y } \pi _ { \theta } ( y \mid x _ { t - 1 } ) { \cal J } ^ { \mathcal { A } } \big ( U _ { \cal A } ( c , x _ { t - 1 } , T ( x _ { t - 1 } , y ) ) \big ) = p _ { t } { \cal J } ^ { \mathcal { A } } \big ( U _ { \cal A } ( c , x _ { t - 1 } , x _ { t } ) \big ) } \\ & { \displaystyle + \sum _ { y \in \mathbb { Z } _ { t } } \pi _ { \theta } ( y \mid x _ { t - 1 } ) { \cal J } ^ { \mathcal { A } } \big ( U _ { \cal A } ( c , x _ { t - 1 } , x _ { e r r } ) \big ) + \sum _ { y \in \mathcal { O } _ { t } } \pi _ { \theta } ( y \mid x _ { t - 1 } ) { \cal J } ^ { \mathcal { A } } \big ( U _ { \cal A } ( c , x _ { t - 1 } , x _ { t , y } ) \big ) . } \end{array}\tag{3}
$$

The dataset supplies neither probabilities within $\mathcal { T } _ { t } \cup \mathcal { O } _ { t }$ nor continuations $J ^ { \mathcal { A } } ( U _ { \mathcal { A } } ( c , x _ { t - 1 } , x _ { t , y } ) )$ containing unobserved off-trace states. Off-trace tactics may lead to alternate proofs or rejoin the trace so the exact objective depends on the search space outside of our dataset.

## 2.3 THE TRACE-SUPPORTED SURROGATE

To obtain a tractable objective, we restrict the search to states from the demonstrated trace. Let $\widehat { \mathcal { C } } _ { \tau }$ ∗ be the configurations whose frontier and memory reference only $x _ { 0 } , \ldots , x _ { L }$ . We replace the true transition by an offline version,

$$
\widehat { \mathcal { T } } _ { \tau ^ { * } } ( x _ { t - 1 } , y ) = \left\{ \begin{array} { l l } { x _ { t } , } & { y = y _ { t } ^ { * } , } \\ { \perp _ { \tau } , } & { y \neq y _ { t } ^ { * } , } \end{array} \right.\tag{4}
$$

where $\perp _ { \tau }$ marks any deviation from the trace: valid off-trace and invalid tactics are collapsed into one training-time event. This transition induces two projected update rules,

$$
\widehat { U } _ { A } ^ { + } ( \widehat { c } , x ) , \qquad \widehat { U } _ { A } ^ { - } ( \widehat { c } , x ) ,
$$

the positive update when the selected node advances $x _ { t - 1 } \to x _ { t }$ , and the negative update following a trace miss. Both updates map to $\widehat { \mathcal { C } } _ { \tau ^ { * } }$ . At a demonstrated state, the model either samples ${ \boldsymbol y } _ { t } ^ { * }$ with probability $p _ { t }$ or deviates with probability $1 - p _ { t }$

Trace-supported recurrence. The surrogate success probability is

$$
\begin{array}{c} \widehat { J } _ { A } ( \widehat c ; p ) = \left\{ { 1 \mathrm { i f } \widehat c \mathrm { s o l v e d } } ; \quad 0 \mathrm { i f } \widehat c \mathrm { d e a d } ,  \\ { \sum _ { x \in \widehat { \mathcal { F } } ( \widehat { c } ) } S _ { A } ( x \mid \widehat c ; p ) \Big [ p _ { t ( x ) } \widehat { J } _ { A } ( \widehat { U } _ { A } ^ { + } ( \widehat { c } , x ) ; p ) + ( 1 - p _ { t ( x ) } ) \widehat { J } _ { A } ( \widehat { U } _ { A } ^ { - } ( \widehat { c } , x ) ; p ) \Big ] , } \end{array} \right.\tag{else}
$$

(5)

where $t ( x )$ is the index of x in $\tau ^ { * }$

We call this loss Compute Aligned Training (CAT), following Ousherovitch & Tewari (2026)

$$
\widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } ( \theta ) = \widehat { J } _ { A } \big ( \widehat { c } _ { 0 } ; p ( \theta ) \big ) , \qquad \widehat { L } _ { C A T } ^ { A } ( \theta ) = - \log \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } ( \theta ) ,\tag{6}
$$

where $\widehat { c } _ { 0 }$ encodes $x _ { 0 }$ and the initial budget. We will assume the recursion is finite.<sup>2</sup>

## 2.4 THE CAT GRADIENT

The following Proposition shows search strategy can be completely specified for training by its triple $( S _ { \mathcal { A } } , \widehat { U } _ { \mathcal { A } } ^ { + } , \widehat { U } _ { \mathcal { A } } ^ { - } )$ , and that its loss and per-tactic gradient weights follow mechanically.

Proposition 1 (CAT Loss). Fix a strategy A given by $( S _ { \mathcal { A } } , \widehat { U } _ { \mathcal { A } } ^ { + } , \widehat { U } _ { \mathcal { A } } ^ { - } )$ . Then:

(i) Equation (5) has a unique solution, and ${ \widehat { J } } _ { \tau ^ { * } } ^ { A } ( \theta )$ depends on θ only through $p = ( p _ { 1 } , \ldots , p _ { L } ) . ^ { 3 }$

(ii) The CAT gradient is a weighted sum of per-step cross-entropy gradients,

$$
\nabla _ { \boldsymbol { \theta } } \widehat { L } _ { C A T } ^ { A } ( \boldsymbol { \theta } ) = \sum _ { t = 1 } ^ { L } \widehat { w } _ { t } ^ { A } \nabla _ { \boldsymbol { \theta } } L _ { C E } ^ { ( t ) } , \qquad \widehat { w } _ { t } ^ { A } = \frac { p _ { t } } { \widehat { \mathcal { T } } _ { \boldsymbol { \tau } ^ { * } } ^ { A } } \frac { \partial \widehat { \mathcal { T } } _ { \boldsymbol { \tau } ^ { * } } ^ { A } } { \partial p _ { t } } .\tag{7}
$$

Proof. (i) Under the assumption the budget is finite and decrements with each update, it terminates after finitely many steps and assigns a unique value to each configuration. All θ-dependence enters only through $p _ { t ( v ) }$ or possibly through the scores $s _ { \mathcal { A } } ( \cdot ; p )$

(ii) By (i), $\begin{array} { r } { \nabla _ { \boldsymbol { \theta } } \widehat { \mathcal { I } } _ { \tau ^ { * } } ^ { A } = \sum _ { t } \partial _ { p _ { t } } \widehat { \mathcal { I } } _ { \tau ^ { * } } ^ { A } \nabla _ { \boldsymbol { \theta } } p _ { t } } \end{array}$ . From $L _ { C E } ^ { ( t ) } = - \log p _ { t }$ we have $\nabla _ { \theta } p _ { t } = - p _ { t } \nabla _ { \theta } L _ { C E } ^ { ( t ) } ,$ so

$$
\nabla _ { \theta } \big [ - \log \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } \big ] = - \frac { 1 } { \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } } \sum _ { t = 1 } ^ { L } \frac { \partial \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } } { \partial p _ { t } } \big ( - p _ { t } \nabla _ { \theta } L _ { C E } ^ { ( t ) } \big ) = \sum _ { t = 1 } ^ { L } \frac { p _ { t } } { \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } } \frac { \partial \widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { A } } { \partial p _ { t } } \nabla _ { \theta } L _ { C E } ^ { ( t ) } .\tag{□}
$$

As noted by Chen et al. (2025) for the case of Pass@N, separating $\widehat { w } _ { t } ^ { A }$ lets one filter low-weight gradients before they corrupt batch normalization statistics, stabilizing training.

## 2.5 A SEARCH-AGNOSTIC UNIFORM-ALLOCATION OBJECTIVE

The same gradient form permits a compute-aware objective without specifying A. For an expansion budget $N \overset { \cdot } { \geq } L$ , UA assigns each demonstrated step a uniform local budget $\bar { K } = N / L$ and uses

$$
\widehat { \mathcal { T } } ^ { \mathrm { U A } } ( p ; N ) = \prod _ { t = 1 } ^ { L } \big [ 1 - ( 1 - p _ { t } ) ^ { \bar { K } } \big ] , \qquad \widehat { L } _ { C A T } ^ { \mathrm { U A } } = - \log \widehat { \mathcal { T } } ^ { \mathrm { U A } } .\tag{8}
$$

For integer $\bar { K }$ , this is the probability that every demonstrated step succeeds within its own quota of independent attempts, without transferring unused quota between steps. For noninteger $N / L$ , we use the continuous extension of the formula. This allocation is a modeling choice, not a claim about how a particular deployed algorithm distributes its budget. By Proposition 1,

$$
\widehat { w } _ { t } ^ { \mathrm { U A } } = \frac { \bar { K } p _ { t } ( 1 - p _ { t } ) ^ { \bar { K } - 1 } } { 1 - ( 1 - p _ { t } ) ^ { \bar { K } } } .\tag{9}
$$

The rule depends on $N , L ,$ , and $p _ { t }$ , but not on $\mathcal { A } . \mathrm { A t } N = L$ it recovers CE; additional local attempts reduce the emphasis on tactics already likely to be sampled. Thus UA is search-agnostic, but not compute-agnostic. Its connection to the shared-budget formulation is detailed in Section J.

## 3 STRATEGY-SPECIFIC OBJECTIVES

We instantiate the construction for Pass@N and best-first search using Proposition $1 . ^ { 4 }$

## 3.1 PASS@N

Pass@N uses up to N independent rollouts (Figure 3); here $N$ counts rollouts, not tactic expansions. In the trace-supported model, let $\widehat { \boldsymbol { c } } = \left( \boldsymbol { r } , t \right)$ , where r counts remaining rollouts, including the current one, and t indexes the next demonstrated tactic. Advancement gives $( r , t + 1 )$ ; a miss abandons the rollout and gives $( r - 1 , 1 )$ . Thus Equation (5) becomes

$$
\widehat { J } ( r , t ) = p _ { t } \widehat { J } ( r , t + 1 ) + ( 1 - p _ { t } ) \widehat { J } ( r - 1 , 1 ) ,\tag{10}
$$

with $\widehat { J } ( r , L + 1 ) = 1$ and $\widehat { J } ( 0 , t ) = 0$ for $t \leq L$ . A rollout completes the demonstrated trace with probability $\begin{array} { r } { \pi _ { \mathrm { p r o o f } } : = \prod _ { t = 1 } ^ { L } p _ { t } , { \mathrm { g i v i n g } } } \end{array}$

$$
\widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { \mathrm { P a s s } \ @ N } ( \theta ) = 1 - ( 1 - \pi _ { \mathrm { p r o o f } } ) ^ { N } .\tag{11}
$$

Applying Proposition 1 yields the same weight at every step of the trace:

$$
\widehat { w } _ { t } ^ { \mathrm { P a s s @ } N } = \frac { N \pi _ { \mathrm { p r o o f } } ( 1 - \pi _ { \mathrm { p r o o f } } ) ^ { N - 1 } } { 1 - ( 1 - \pi _ { \mathrm { p r o o f } } ) ^ { N } } .
$$

## 3.2 BEST-FIRST SEARCH

We consider policy-guided best-first search (BFS) (Orseau et al., 2018; Orseau & Lelis, 2021; Xin et al., 2025). A node $x _ { k }$ reached through tactics $y _ { 1 } , \dotsc , y _ { k } , k \geq 1$ , is scored by

$$
s _ { \mathrm { B F S } } ( x _ { k } ; \pi _ { \theta } ) = \left( \prod _ { j = 1 } ^ { k } \pi _ { \theta } ( y _ { j } \mid x _ { j - 1 } ) \right) ^ { 1 / k } .\tag{12}
$$

The algorithm pops the highest-scoring frontier node, samples B tactics there, and inserts valid children (Figure 4). Here $\bar { N }$ counts tactic expansions: each sample costs one unit.

Trace-supported instantiation. We use a retry surrogate with configuration ${ \widehat { c } } = ( b , t )$ , where b is the remaining expansion budget and the only live node is $x _ { t - 1 }$ . Starting from $( N , \dot { 1 } )$ , advancement gives $( b - 1 , \bar { t } + \bar { 1 } )$ and a miss gives $( b - 1 , \dot { t } )$ . Retaining the node after a miss is a modeling choice: this surrogate replaces the deployed finite per-pop sampling with retries until the demonstrated tactic is sampled or the budget is exhausted. Its singleton frontier also removes score-based competition. The recurrence is

$$
\widehat { J } _ { \mathrm { B F S } } ( \boldsymbol { b } , t ) = p _ { t } \widehat { J } _ { \mathrm { B F S } } ( \boldsymbol { b } - 1 , t + 1 ) + ( 1 - p _ { t } ) \widehat { J } _ { \mathrm { B F S } } ( \boldsymbol { b } - 1 , t ) ,
$$

$$
\widehat { J } _ { \mathrm { B F S } } ( b , L + 1 ) = 1 , \qquad \widehat { J } _ { \mathrm { B F S } } ( 0 , t ) = 0 \quad ( t \leq L ) .\tag{13}
$$

Objective. Let $T _ { t } \sim \mathrm { G e o m } ( p _ { t } )$ independently count the samples needed at step t, including the successful sample, and write $\begin{array} { r } { S _ { L } = \sum _ { t = 1 } ^ { L } T _ { t } } \end{array}$ . For $N \geq L$ , the surrogate success probability is

$$
\widehat { \mathcal { T } } _ { \tau ^ { * } } ^ { \mathrm { B F S } } ( \boldsymbol { \theta } ) = \mathbb { P } ( S _ { L } \leq N ) .\tag{14}
$$

The shared budget couples the per-tactic weights, which follow by differentiating Equation (13).   
The connection to $\mathrm { U A } ^ { \prime } \mathrm { s }$ fixed local allocation is detailed in Section J.

## 4 APPROXIMATION ERROR

A demonstrated trace tells us how one proof succeeds, but not what happens when search leaves that path. At state $x _ { t - 1 } ,$ we can compute the probability of missing the demonstrated tactic. This section intends to characterize what is lost in not having information about this event at training. We study this missing information through two channels: alternative proofs (the bypass channel) and compute spent on unsuccessful off-trace exploration (the trap channel). How do these unseen consequences change the importance of sampling the demonstrated tactic? Full derivations are in Section $\overset { \cdot } { \mathrm { C } } . ^ { 5 }$

## 4.1 EFFECTS OF THE CHANNELS ON THE TRUE WEIGHT

Bypass: missing the trace isn’t failing. A different tactic may lead to a valid proof without completing $\tau ^ { * }$ . The surrogate cannot see these successes. For Pass@ $N$ , accounting for alternative proofs reduces the deployed weight relative to the trace-supported weight.

Theorem 1 (Bypass Deflation). Under Assumption 1,for Pass@N with $N > 1$ and every step t,

$$
w _ { t } ^ { \star } \leq \widehat { w } _ { t } ^ { \mathrm { P a s s } @ N } < 1 = w _ { t } ^ { \mathrm { C E } } .\tag{15}
$$

$$
C o n s e q u e n t l y , | \widehat { w } _ { t } ^ { \mathrm { P a s s @ } N } - w _ { t } ^ { \star } | < | 1 - w _ { t } ^ { \star } | .
$$

The proof and decomposition of the missing alternative-success contributions are given in Theorem 4 and eq. (25).

Trap: a miss can cost much more than one attempt. A valid but unhelpful tactic can lead search into a branch that consumes many expansions before the demonstrated state is selected again. Unlike an immediate rejection, this excursion leaves less budget for completing the proof. Our approximation understates the cost of such misses.

To isolate this effect, we censor alternative proofs, so only completion of $\tau ^ { * }$ counts as success while off-trace exploration remains possible (Lemma 4). We quantify the resulting upward pressure on weights in a model with a common deterministic return cost.

Theorem 2 (Shared-Budget Bracketing under Deterministic Excursions). Under Assumption $^ { 2 , }$ let $N \geq L$ be an integer. Every miss costs $1 \leq \kappa \leq \bar { \kappa } < \infty$ expansions before returning to the same trace state; successful trials cost one, and κ is independent of p. Then

$$
w _ { t } ^ { \mathrm { t r a p } } ( N ) \in \left[ \widehat { w } _ { t } ( N ) , \widehat { w } _ { t } \bigg ( L + \frac { N - L } { \bar { \kappa } } \bigg ) \right] \subseteq [ \widehat { w } _ { t } ( N ) , 1 ] ,\tag{16}
$$

where $w _ { t } ^ { \mathrm { t r a p } } = w _ { t } ( J ^ { \mathrm { t r a p } } )$ and noninteger budget arguments are rounded down.

The mechanism is a reduction in effective budget to $N _ { \mathrm { e f f } } = L + ( N - L ) / \kappa$ (Proposition 3). The surrogate assumes $\kappa = 1 \colon$ a miss costs one expansion. At the other extreme, if every miss prevents recovery, success requires following the trace without mistakes, recovering CE’s unit weights. Finite excursion costs interpolate between these cases within this model.

What the trace cannot determine. Two environments can agree on every demonstrated transition, all $p _ { s }$ , the search rule, and the budget, yet differ in what a miss causes. When those off-trace differences produce distinct deployed weights, no rule using only these trace-level inputs can be exact in both environments (Proposition 4). Thus, solving the trace-supported recurrence exactly does not resolve the missing information about recovery.

## 4.2 LARGE-COMPUTE SCALING: CAN SEARCH RECOVER FROM A MISS?

More compute can compensate for a delay, but not for permanent loss of the opportunity to recover. $\mathbf { A } \mathbf { t } \ N = { \ddot { L } }$ in the trap-only model, there is no room for wasted expansions: success requires every demonstrated tactic on its first attempt, and both the deployed and surrogate weights equal one. Larger budgets create opportunities to recover, making the outcome depend on whether excursions eventually return. The following limits hold with the policy and excursion parameters fixed.

Theorem 3 (Large-Budget Limits and Recoverability). (i) Pass@N. Under Assumption $^ { l , }$

$$
w _ { t } ^ { \star } ( N ) \longrightarrow 0 , \qquad \widehat { w } _ { t } ( N ) \longrightarrow 0 .
$$

Hence the search-aware weight error tends to zero, while CE’s weight error tends to one.

(ii) Shared-budget excursions. Under Assumption 3, each miss at step s independently causes permanent absorption with probability $q _ { s } \in [ 0 , 1 )$ ; otherwise it incurs a bounded excursion before returning to the same trace state. Only the demonstrated proof counts as success. Holding $q _ { s }$ and thefinite-excursion lawsfixed under differentiation,

$$
\widehat w _ { t } ( N ) \longrightarrow 0 , \qquad w _ { t } ^ { \mathrm { t r a p } } ( N ) \longrightarrow \omega _ { t } : = \frac { q _ { t } } { p _ { t } + ( 1 - p _ { t } ) q _ { t } } .
$$

See Propositions 5 and 6 for the derivations.

When all excursions are recoverable $( q _ { s } ~ = ~ 0 )$ , both deployed and surrogate weights eventually vanish: avoiding a particular miss matters less when search has enough budget to try again. With permanent absorption, avoiding the miss retains value even with unlimited compute. The retry surrogate overlooks this risk, leaving limiting error ω<sub>t</sub>; CE’s limiting error is $1 - \omega _ { t }$

The analysis therefore identifies what trace-only training leaves unresolved: alternative routes to success, the compute needed to recover, and the possibility of never recovering. These quantities can be modeled, but are not determined by the demonstrated trace. Additional assumptions can specify them, and training-time search offers a way to collect evidence about them. Neither is replaced by evaluating the same trace-supported surrogate more exactly.

## 5 EXPERIMENTS

We test whether accounting for budgeted search during SFT improves proof discovery, whether matching the training objective to the deployed strategy helps, and how the resulting gains vary with test-time budget. We compare search-aware CAT and search-agnostic UA with a CE control.

Setup. We use Lean 4 through LeanDojo (Yang et al., 2023), training on leandojobenchmark-4-random, a random split of mathlib4 (The mathlib Community, 2020). The heldout pool contains 458 theorems whose reference proofs have 2–5 steps; any kernel-verified proof counts as success. We initialize from Qwen2.5-Math-7B-Instruct (Qwen Team, 2024) and fine-tune with LoRA (Hu et al., 2021). Within each experiment, models share a CE warm-up.<sup>6</sup> The control then receives one additional CE epoch, while the search-aware and UA models use one additional epoch with their respective weights. Data, ordering, and optimizer settings are otherwise identical. Evaluation budgets count node expansions, each comprising one tactic sample and its kernel verification.<sup>7</sup> Full details are in Section G.

## 5.1 EXPERIMENT 1: TRAINING FOR SEARCH ACROSS STRATEGIES

At N = 256 expansions, we evaluate Pass@N, BFS, DFS variants (Section D), and Monte Carlo Tree Search (MCTS) (Kocsis & Szepesvari, 2006). ´ <sup>8</sup> For each deployed strategy, we compare CE, the corresponding search-aware weights, and the same single shared UA adapter. This study also applies wall-clock caps; capped runs without a proof count as failures (Section G.3).

Gain For CAT Training  
Table 1: Proofs found (%) at $N = 2 5 6$ on 458 held-out theorems. Columns are deployed strategies. The same UA adapter is used throughout. <sup>∗</sup> marks $p \ < \ 0 . 0 5$ against CE under the same strategy; bold marks the highest reported rate.
<table><tr><td>Training objective</td><td>Pass@N</td><td>BFS</td><td>DFS</td><td>RDFS</td><td>VDFS</td><td>MCTS</td></tr><tr><td>CE</td><td>21.2</td><td>23.8</td><td>19.2</td><td>20.7</td><td>21.2</td><td>20.3</td></tr><tr><td>Search-aware CAT</td><td>26.6*</td><td>25.1</td><td>20.7</td><td>23.6*</td><td>23.8</td><td>22.5</td></tr><tr><td>Shared UA</td><td>25.1*</td><td>26.9</td><td>23.6*</td><td>25.3*</td><td>24.5*</td><td>25.1*</td></tr></table>

![](images/7fce948741040e29531d758dc2fed0a0b129c021134d0660e706caa973d99bf3.jpg)  
Figure 1: Both training families improve accuracy over CE. Proofs found (%) on 458 held-out theorems with a budget of $N = 2 5 6$ tactic expansions. Search-aware CAT uses strategy-matched training objectives. UA uses one adapter across all strategies. UA has the highest accuracy under branching strategies; search-aware CAT leads under Pass@N. Error bars show 95% intervals.

Results Both search-aware CAT and UA have higher observed success rates than CE under all six strategies (Table 1), showing that accounting for deployment-time compute provides substantial gains over standard SFT. The shared UA objective is a strong search-agnostic approach, attaining the highest point estimate under the five branching strategies. At the same time, search-aware CAT achieves the highest observed success under Pass@N (26.6% versus 25.1% for UA), showing modeling the deployed search strategy can yield additional improvement. Consistent with this, within the search-aware family, matching training and deployment gives higher point estimates for the Pass@N/BFS adapter pair: 26.6% versus 23.7% under Pass@N, and 25.1% versus 24.5% under BFS (Table 3). Together, these results support compute-aware training and suggest that modeling the deployed strategy can provide additional benefits.

## 5.2 EXPERIMENT 2: ALIGNMENT GAIN VERSUS TEST-TIME BUDGET

We train a separate search-aware CAT adapter for each strategy (Pass@N or BFS) and budget $N \in$ {16, 64, 256}, and evaluate each adapter at its corresponding budget. This sweep removes the perstrategy and per-theorem wall-clock caps, but each model receives at most 80 hours of evaluation on four A100 GPUs. At each budget, we compare CAT and CE only on theorems for which both completed evaluation; Table 2 reports the paired subset sizes. N = 1024 runs completed only 56–57 paired theorems and are reported separately in Section H.

Gains persist across the evaluated budgets. Search-aware CAT has positive paired gains at every reported budget in Table 2. The observed gains at $N = 2 5 6$ exceed those at $N = 1 6$ for both Pass@N (+4.2 versus +2.0 points) and BFS (+6.9 versus +4.4). However, these are finite-budget performance comparisons, not direct measurements of the weight-error limits in Section 4.

## 6 CONCLUSION

This paper studies how to train an LLM from demonstrations when it will be deployed within policy-guided search under a finite compute budget. We develop two complementary approaches:

Alignment Gain versus Test-Time Budget  
Table 2: Paired search-aware CAT gain over CE. n is the paired subset size; b and c count CATonly and CE-only successes. Gains are in percentage points. The 95% intervals use the variance approximation $( \bar { b + c } ) / n ^ { 2 }$ on the probability scale; p is the exact two-sided McNemar test.
<table><tr><td>Strategy</td><td>Budget N</td><td>n</td><td> $b / c$ </td><td>Gain (pp), 95% interval</td><td>p</td></tr><tr><td rowspan="3">Pass@N</td><td>16</td><td>458</td><td>10/1</td><td>+2.0 [0.5, 3.4]</td><td>0.012</td></tr><tr><td>64</td><td>458</td><td>28/8</td><td>+4.4 [1.8, 6.9]</td><td>0.001</td></tr><tr><td>256</td><td>408</td><td>29/12</td><td>+4.2 [1.1, 7.2]</td><td>0.012</td></tr><tr><td rowspan="3">BFS</td><td>16</td><td>458</td><td>28/8</td><td>+4.4 [1.8, 6.9]</td><td>0.001</td></tr><tr><td>64</td><td>458</td><td>34/13</td><td>+4.6 [1.7, 7.5]</td><td>0.003</td></tr><tr><td>256</td><td>407</td><td>37/9</td><td>+6.9 [3.6, 10.1]</td><td>&lt; 0.001</td></tr></table>

![](images/b6f89c3308b321e0a48f66b6149e7dab842f4db00e92625e9098c3c7f7a1da8c.jpg)  
Figure 2: Paired search-aware CAT gain over CE with budget-matched adapters. Error bars are 95% intervals. Completed subsets differ across budgets, so connecting lines are descriptive rather than a common-cohort comparison.

search-aware CAT objectives that incorporate a model of the search strategy, and a search-agnostic uniform-allocation objective that accounts for compute without specifying the search rule. Both induce scalar weights on the usual per-tactic CE gradients. Although our experiments focus on ATP, the framework provides a starting point for studying other settings in which generative models guide tree search. Our analysis characterizes how alternative proofs and off-trace exploration affect search-aware gradient weights. It identifies conditions under which their approximation error vanishes as the test-time budget grows while CE remains misaligned. Empirically, all objectives attain higher observed proof-success rates than an epoch-matched CE control across all six evaluated strategies. The shared UA objective performs strongly across search algorithms, while searchaware CAT achieves the highest observed success under Pass@N, demonstrating that incorporating strategy-specific structure can, in some cases, provide an additional benefit beyond generic compute awareness. Budget sweeps additionally show larger gains over CE at 256 than at 16 expansions for both tested strategies. Together, these findings show that considering the downstream search during training can improve performance.

Limitations and Future Work. Our experiments use one model and short reference proofs, with training restricted to offline SFT. UA accounts for compute without modeling the search rule; searchaware CAT models selected search dynamics on the demonstrated trace. Both remain trace-only. Ad ditional training-time compute could sample one-step deviations, explore local trees around demonstrations, and increasingly approximate deployment search. One-step samples reveal alternative transitions; deeper searches provide evidence about alternative proofs and recovery costs. Such observations could refine search-aware objectives beyond fixed allocation. Testing whether these refinements improve performance, and how to allocate training compute between supervision and search, remain future work.

## REFERENCES

Zhangir Azerbayev, Hailey Schoelkopf, Keiran Paster, Marco Dos Santos, Stephen McAleer, Albert Q Jiang, Jia Deng Jia, Sean Welleck, et al. Llemma: An open language model for mathematics. arXiv preprint arXiv:2310.10631, 2023.

Ananth Balashankar, Ziteng Sun, Jonathan Berant, Jacob Eisenstein, Michael Collins, Adrian Hutter, Jong Lee, Chirag Nagpal, Flavien Prost, Aradhana Sinha, Ananda Theertha Suresh, and Ahmad Beirami. Infalign: Inference-aware language model alignment, 2025. URL https://arxiv. org/abs/2412.19792.

Yoshua Bengio and Nikolay Malkin. Machine learning and information theory concepts towards an ai mathematician, 2024. URL https://arxiv.org/abs/2403.04571.

Feng Chen, Allan Raventos, Nan Cheng, Surya Ganguli, and Shaul Druckmann. Rethinking fine-´ tuning when scaling test-time compute: Limiting confidence improves mathematical reasoning. In Advances in Neural Information Processing Systems, 2025. URL https://openreview .net/forum?id=jvVQeSMeGM.

Mark Chen, Jerry Tworek, Heewoo Jun, Qiming Yuan, Henrique Ponde de Oliveira Pinto, Jared Kaplan, Harri Edwards, Yuri Burda, Nicholas Joseph, Greg Brockman, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Yinlam Chow, Guy Tennenholtz, et al. Inference-aware fine-tuning for best-of-n sampling in large language models. In International Conference on Learning Representations, 2025.

Karl Cobbe, Vineet Kosaraju, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Remi Coulom. Efficient selectivity and backup operators in monte-carlo tree search. In ´ Computers and games, pp. 72–83. Springer, 2006.

Remi Coulom. Computing elo ratings of move patterns in the game of go. In ´ Computer Games Workshop, 2007.

Tri Dao. FlashAttention-2: Faster attention with better parallelism and work partitioning. In International Conference on Learning Representations (ICLR), 2024.

Leonardo de Moura and Sebastian Ullrich. The lean 4 theorem prover and programming language. In Automated Deduction – CADE 28, pp. 625–635. Springer International Publishing, 2021. doi: 10.1007/978-3-030-79876-5\_37. URL https://doi.org/10.1007/978-3-0 30-79876-5\_37.

Google DeepMind. Ai achieves silver-medal standard solving international mathematical olympiad problems. Google DeepMind Blog, 2024. URL https://deepmind.google/discover /blog/ai-achieves-silver-medal-standard-solving-international-m athematical-olympiad-problems/.

Jesse Michael Han, Jason Razeghi, Igor Babuschkin, Ilya Sutskever, and Stanislas Polu. Proof artifact co-training for theorem proving with language models. In International Conference on Learning Representations, 2022.

Peter E Hart, Nils J Nilsson, and Bertram Raphael. A formal basis for the heuristic determination of minimum cost paths. IEEE Transactions on Systems Science and Cybernetics, 4(2):100–107, 1968.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. Lora: Low-rank adaptation of large language models, 2021. URL https: //arxiv.org/abs/2106.09685.

Albert Q Jiang, Sean Welleck, Jin P Zhou, Timothee Li, Jiacheng Liu, Mateja Jamnik, Simon´ Lacoste-Julien, and Yuhuai Wu. Draft, sketch, and prove: Guiding formal theorem provers with informal proofs. In International Conference on Learning Representations (ICLR), 2023.

Levente Kocsis and Csaba Szepesvari. Bandit based monte-carlo planning. In ´ European Conference on Machine Learning, pp. 282–293. Springer, 2006.

Guillaume Lample, Marie-Anne Lachaux, Thibaut Lavril, Xavier Martinet, Amaury Hayat, Gabriel Ebner, Aurelien Rodriguez, and Timoth ´ ee Lacroix. Hypertree proof search for neural theorem´ proving. arXiv, 2022. doi: 10.48550/arxiv.2205.11491.

Laurent Orseau and Levi H. S. Lelis. Policy-guided heuristic search with guarantees, 2021. URL https://arxiv.org/abs/2103.11505.

Laurent Orseau, Levi H. S. Lelis, Tor Lattimore, and Theophane Weber. Single-agent policy tree ´ search with guarantees, 2018. URL https://arxiv.org/abs/1811.10928.

Adam Ousherovitch and Ambuj Tewari. Test time aligned post training. arXiv preprint, 2026. Forthcoming/Under Review.

Judea Pearl. Heuristics: Intelligent Search Strategies for Computer Problem Solving. Addison-Wesley, 1984.

Gabriel Poesia, David Broman, Nick Haber, and Noah D. Goodman. Learning formal mathematics from intrinsic motivation, 2024. URL https://arxiv.org/abs/2407.00695.

Stanislas Polu and Ilya Sutskever. Generative language modeling for automated theorem proving, 2020. URL https://arxiv.org/abs/2009.03393.

Stanislas Polu, Jesse Michael Han, Kunhao Zheng, Mantas Baksys, Igor Babuschkin, and Ilya Sutskever. Formal mathematics statement curriculum learning. arXiv, 2022. doi: 10.48550 /arxiv.2202.01344.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

Charlie Snell, Jaehoon Lee, Kelvin Xu, and Aviral Kumar. Scaling LLM test-time compute optimally can be more effective than scaling model parameters. arXiv preprint arXiv:2408.03314, 2024. URL https://arxiv.org/abs/2408.03314.

Peiyang Song, Kaiyu Yang, and Anima Anandkumar. Lean copilot: Large language models as copilots for theorem proving in lean. arXiv, 2024. doi: 10.48550/arxiv.2404.12534.

Peiyang Song, Kaiyu Yang, and Anima Anandkumar. Lean copilot: Large language models as copilots for theorem proving in lean, 2025. URL https://arxiv.org/abs/2404.12534.

Yunhao Tang, Kunhao Zheng, Gabriel Synnaeve, and Remi Munos. Optimizing language models for ´ inference time objectives using reinforcement learning. arXiv preprint arXiv:2503.19595, 2025. URL https://arxiv.org/abs/2503.19595.

The mathlib Community. The Lean mathematical library. In Proceedings of the 9th ACM SIGPLAN International Conference on Certified Programs and Proofs (CPP), pp. 367–381. ACM, 2020. doi: 10.1145/3372885.3373824.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. Self-consistency improves chain of thought reasoning in language models. In International Conference on Learning Representations, 2023.

Yuhuai Wu, Albert Q Jiang, Wenda Li, Markus N Rabe, Charles Staats, Mateja Jamnik, and Christian Szegedy. Autoformalization with large language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 35, pp. 32353–32368, 2022.

Ran Xin, Chenguang Xi, Jie Yang, Feng Chen, Hang Wu, Xia Xiao, Yifan Sun, Shen Zheng, and Kai Shen. Bfs-prover: Scalable best-first tree search for llm-based automatic theorem proving, 2025. URL https://arxiv.org/abs/2502.03438.

Kaiyu Yang, Aidan Swope, Alex Gu, Rahul Chalamala, Peiyang Song, Shixing Yu, Saad Godbole, Zhichen Lai, and Anima Anandkumar. LeanDojo: Theorem proving with retrieval-augmented language models. In Advances in Neural Information Processing Systems (NeurIPS), volume 36, pp. 23722–23745, 2023.

Kaiyu Yang, Gabriel Poesia, Jingxuan He, Wenda Li, Kristin Lauter, Swarat Chaudhuri, and Dawn Song. Formal mathematical reasoning: A new frontier in ai, 2024. URL https://arxiv.or g/abs/2412.16075.

Kunhao Zheng, Jesse Michael Han, and Stanislas Polu. minif2f: a cross-system benchmark for formal olympiad-level mathematics. In International Conference on Learning Representations, 2022.

Budget: N = 4 parallel threads  
![](images/d869b274bbbfc24a3792195db2b6ccbe72864e373d580bbf65fc9d419a9142fd.jpg)  
Figure 3: Pass@N Parallel Search. $N = 4$ independent paths explore the tree simultaneously. A single valid path reaching QED constitutes a global success.

![](images/c6295687595774bfb1bcccd924e62e7527371182818936eab04cb22e393ac137.jpg)  
Figure 4: Policy-Guided Best-First Search. The numbered circles indicate chronological node $\mathbf { e x - }$ pansions. Expanding $x _ { 0 }$ yields two valid states on the frontier, $x _ { 1 } \left( S = 0 . 8 \right)$ and $x _ { 1 } ^ { \prime } \ ( \bar { S } = 0 . 6 )$ . The algorithm greedily expands $x _ { 1 }$ (step $2 )$ . However, generating the tactic to reach $x _ { 2 }$ drops the path’s geometric mean to $S \stackrel { - } { = } 0 . 3$ . Because the algorithm evaluates the global frontier, the decision rule abandons the active path and dynamically jumps across the tree to expand $x _ { 1 } ^ { \prime }$ (step 4), ultimately finding the proof.

Global Budget: N = 5 sequential expansions

![](images/78fd2ec1607975c6805055c5c4115a10df1ec22b2c8b5f6a7f41dc2d7261e636.jpg)  
Figure 5: Depth-First Search. The numbered circles indicate the chronological order of node expansions. The algorithm pushes forward, backtracks locally upon failure (e.g., from step 2 back to $x _ { 1 } )$ and terminates the search immediately upon reaching the QED state at step 5.

![](images/2cd233660bdcfd9cb39f328fc135b333193b030c41daed72ac9f15486a4cb662.jpg)  
Figure 6: Randomized Depth-First Search. After expanding a deep, dead-end path (steps 1–3), the algorithm selects a node uniformly at random from the active path $\{ x _ { 0 } , x _ { 1 } , x _ { 2 } \}$ to backtrack to. In this case, it randomly selects $x _ { 1 }$ (step 4), escaping the trap at $x _ { 2 }$ and successfully finding the proof on an alternative branch (step 5).

![](images/45de7ef4fd7109321fc361389cde395adbe4913818216c59b57e9ae6946d5e89.jpg)  
Figure 7: Value Guided Depth-First Search (VDFS). After expanding into a dead end (step 3), the algorithm evaluates the active path prefix $\{ x _ { 0 } , x _ { 1 } , x _ { 2 } \}$ . Instead of backtracking uniformly, it samples an ancestor proportional to its Levin score S. In this case, it randomly jumps to $x _ { 1 }$ (step 4) because its higher score $( S = 0 . 8 )$ makes it a heavily favored target over the deep trap at x<sub>2</sub> $( S = 0 . 3 )$ . This allows it to successfully escape and find the proof on an alternative branch (step 5).

![](images/21c8bd70d0789d4ae284b1592a33c97366f28b30b63d6dfb1076d4db9b0e824e.jpg)  
Figure 8: Deployed Monte Carlo Tree Search. Circled numbers indicate the chronological order of node expansions; a budget of $N = 4$ expansions is exhausted in this run. Selection/Expansion: the root is expanded twice, sampling $y _ { 1 }$ into $x _ { 1 }$ (step 1) and $y _ { 2 }$ into $x _ { 1 } ^ { \prime }$ (step $_ { 2 ) ; }$ each is evaluated once via the policy’s own Levin score, giving $V ( x _ { 1 } ) = 0 . 8 0$ and $V ( \bar { x } _ { 1 } ^ { \prime } ) = 0 . 6 0$ . With both children visited once, UCT (exploration constant $c = 0 . 5 )$ scores $x _ { 1 }$ at 1.22 against $x _ { 1 } ^ { \prime }$ at 1.02, so the search commits to the higher-value child and expands it (step 3), reaching a dead end. Backup: the failure backs up a value of 0 through $x _ { 1 }$ , dropping its running average to $\bar { Q } ( x _ { 1 } ) = 0 . 4 0$ . Selection reverses: recomputing UCT with the updated statistics gives $\breve { x _ { 1 } ^ { \prime } }$ a score of 1.12 against $x _ { 1 } \mathrm { { ' } s }$ now-diminished 0.77 – unlike Policy-Guided Best-First Search (Figure 4), where the frontier is re-ranked by a static score, here it is the shrinking exploration bonus and the value drop from a real backup that overturn the earlier choice. The search abandons $x _ { 1 }$ ’s subtree, expands $x _ { 1 } ^ { \prime }$ (step 4), and reaches QED.

## B FORMAL MDP FORMULATION OF AUTOMATED THEOREM PROVING

We formalize the structure of LLM-guided theorem proving in interactive environments like Lean, where text generation is strictly mediated by a formal verifier (the Lean kernel), as a Markov Decision Process (MDP) (Polu & Sutskever, 2020; Lample et al., 2022; Song et al., 2024). We define this sequential decision-making process as a tuple $\bar { \mathcal { M } } = \langle \mathcal { X } , \mathcal { Y } , \mathcal { T } , \mathcal { R } , x _ { 0 } \rangle \mathrm { : }$

• State Space (X): The discrete set of all valid Lean proof states. This space includes two terminal absorbing states: a universal success state $x _ { Q E D }$ and a failure state $x _ { e r r }$

• Action Space (Y): The space of possible tactics the LLM can generate.

• Transition Function (T): The deterministic update rules governed by the Lean kernel, defined as $\mathcal { T } : \mathcal { X } \times \mathcal { Y }  \mathcal { X }$ . Given a state $x _ { t - 1 }$ and tactic $y _ { t } , \mathcal { T } ( x _ { t - 1 } , y _ { t } ) = x _ { t }$ if the tactic is logically and syntactically valid. If $y _ { t }$ is invalid, the kernel immediately rejects $\mathrm { i t , }$ , and $\mathcal { T } ( x _ { t - 1 } , y _ { t } ) = x _ { e r r }$ . If $y _ { t }$ completes the proof, $\mathcal { T } ( x _ { t - 1 } , y _ { t } ) = x _ { Q E D }$

• Reward Function (R): A sparse terminal reward function mapping to {0, 1}. $\mathcal { R } ( x _ { t - 1 } , y _ { t } ) =$ 1 if $\mathcal { T } ( x _ { t - 1 } , y _ { t } ) = x _ { Q E D }$ , and 0 otherwise.

• Initial State (x<sub>0</sub>): The underlying theorem declaration to be proven.

The probability of executing any valid trajectory of length $K$ , denoted $\tau = ( x _ { 0 } , y _ { 1 } , x _ { 1 } , \dots , y _ { K } , x _ { K } )$ is the product of the model’s autoregressive action probabilities:

$$
P _ { \pi _ { \theta } } ( \tau \mid x _ { 0 } ) = \prod _ { t = 1 } ^ { K } \pi _ { \theta } ( y _ { t } \mid x _ { t - 1 } )\tag{17}
$$

Let $\Gamma _ { s u c c } ( x _ { 0 } )$ denote the intractable set of all valid, successful trajectories originating from $x _ { 0 }$ and terminating in $x _ { Q E D }$ . The true probability of proof discovery without search (Pass@1) is the marginal likelihood over this entire set:

$$
P _ { \mathrm { a n y } } : = \sum _ { \tau \in \Gamma _ { s u c c } ( x _ { 0 } ) } P _ { \pi _ { \theta } } ( \tau \mid x _ { 0 } )\tag{18}
$$

Because marginalizing over $\Gamma _ { s u c c } ( x _ { 0 } )$ is computationally impossible, standard SFT sidesteps this global objective, reducing the objective to maximizing the likelihood of a single optimal trace $\tau ^ { * }$ via behavioral cloning, which yields the standard cross-entropy loss discussed in Section 2.

## C APPROXIMATION ERROR AND SCALING

This appendix develops the approximation analysis for the search-aware objectives introduced in Section 3. These objectives replace the deployed success probability $J ^ { A }$ with a trace-supported proxy $\widehat { J } _ { A }$ . We study the resulting approximation through the per-tactic gradient weights, asking two questions: (i) at a fixed budget, when is the search-aware weight closer to the deployed weight than CE’s constant unit weight? (ii) how does this approximation error change as the test-time budget grows? The search-agnostic UA objective is a separate budget-based construction, discussed in Section J, and need not approximate the dynamics of a particular deployed search.

We study Pass@N (Section C.2) and the shared-budget retry model used for best-first search (Section C.3). For Pass@N, the budget counts independent rollouts, allowing us to isolate the effect of alternative successful trajectories omitted by the demonstrated trace. For the retry model, we examine how unsuccessful off-trace excursions consume a shared expansion budget. The latter results rely on explicit assumptions about excursion costs and recovery, rather than providing a complete characterization of deployed best-first search.

The analysis separates three effects:

• Bypass channel. For Pass@N, under the proportional-reallocation convention and Assumption 1, the deployed weight is no larger than the trace-supported weight, which is no larger than CE’s unit weight (Theorem 4).

• Trap channel. In a trap-only retry model with a common deterministic miss cost, excursions that return to the same trace state reduce the effective budget. The resulting weight lies between the retry-surrogate weight and CE’s unit weight (Proposition 3).

• Budget scaling. In the models studied here, large-budget weight error vanishes for Pass@N and for recoverable shared-budget excursions. Permanent absorption can instead leave a nonzero error floor (Section C.4).

These results identify off-trace quantities that demonstrations alone do not determine. Incorporating them requires additional modeling assumptions or observations of search behavior. Section I explores a simple use of such assumptions through priors on off-trace cost and recovery.

## C.1 WELL-POSED DEPLOYED WEIGHTS

The true objective depends on the entire policy, not only on the demonstrated probabilities $p =$ $( p _ { 1 } , \dots , p _ { L } ) , \mathrm { s o } \ ^ { \prime } \partial J / \partial p _ { t } ^ { \prime }$ is meaningless until we fix how the remaining probability mass moves when $p _ { t }$ does. We adopt the assumption of proportional reallocation: at each demonstrated state, write

$$
\pi _ { \theta } ( y \mid x _ { t - 1 } ) = p _ { t } { \bf 1 } \{ y = y _ { t } ^ { * } \} + ( 1 - p _ { t } ) \rho _ { t } ( y ) , \qquad \rho _ { t } \in \Delta \bigl ( \mathcal { V } \setminus \{ y _ { t } ^ { * } \} \bigr ) ,\tag{19}
$$

and define derivatives in $p _ { t }$ holding $\rho _ { t } ,$ , all other $p _ { s } .$ , and all conditionals at non-demonstrated states fixed. This is not explicitly true, tactics similar to the demonstrated tactic will gain probability mass while distant ones will lose mass; however, this convention makes analysis feasible. This is strictly a convention and not a statement about how changing the gradient redistributes mass to unobserved tactics. All statements about the true weights below are relative to this convention. For any success function $J > 0$ define the elasticity

$$
w _ { t } ( J ) : = { \frac { \partial \log J } { \partial \log p _ { t } } } = { \frac { p _ { t } } { J } } { \frac { \partial J } { \partial p _ { t } } } ,\tag{20}
$$

so that $w _ { t } ^ { \star } : = w _ { t } ( J ^ { A } )$ is the (unknown) deployed weight, $\widehat { w } _ { t } : = w _ { t } ( \widehat { J } _ { A } )$ is the CAT weight of Proposition 1, and CE corresponds to $w _ { t } ^ { \mathrm { { \tiny { C E } } } } \equiv 1$ . Comparing training rules then reduces to a scalar question at each step.

Proposition 2 (When is CAT closer than CE?). Suppose $0 \leq \widehat { w } _ { t } < 1$ . Then $\left| w _ { t } ^ { \star } - \widehat { w } _ { t } \right| \leq \left| w _ { t } ^ { \star } - 1 \right|$ if and only $i f w _ { t } ^ { \star } \leq \frac { 1 } { 2 } ( 1 + \widehat { w } _ { t } ) . \ I f \widehat { w } _ { t } = 1 , C A T$ and CE coincide.

Proof. Square both sides, cancel $( w _ { t } ^ { \star } ) ^ { 2 }$ , and divide by $1 - \widehat { w } _ { t } > 0$

Thus, CAT wins exactly when the deployed search makes the demonstrated tactic less critical than single-shot generation assumes; CE can win only when the tactic remains a near single-shot bottleneck under deployment. The next two subsections determine which side of the crossover each strategy occupies.

## C.2 PASS@N : UNIT ELASTICITY AND A ONE-SIDED SQUEEZE

Let $\begin{array} { r } { a ( p ) = \prod _ { i = 1 } ^ { L } p _ { i } } \end{array}$ be the one-rollout probability of the demonstrated trace, $b ( p )$ the one-rollout probability of proving the theorem by any other successful trajectory, and $P _ { \mathrm { a n y } } = a + b . ^ { 9 }$ With $F ( x ) = 1 - ( 1 - x ) ^ { N }$ , the deployed objective is $J ^ { \mathrm { t r u e } } = F ( P _ { \mathrm { a n y } } )$ and the surrogate is ${ \widehat { J } } = F ( a )$ Define the elasticity of the outer function,

$$
H _ { F } ( x ) = { \frac { x F ^ { \prime } ( x ) } { F ( x ) } } .\tag{21}
$$

Lemma 1 (Properties of $H _ { F } ) .$ . For $F ( x ) = 1 - ( 1 - x ) ^ { N } a n d x \in ( 0 , 1 )$

$$
H _ { F } ( x ) = \frac { N } { \sum _ { j = 0 } ^ { N - 1 } ( 1 - x ) ^ { - j } } ,\tag{22}
$$

so $H _ { F }$ is nonincreasing, $H _ { F } ( x ) \leq 1$ with $H _ { F } ( 0 ^ { + } ) = 1$ , and for fixed x, $H _ { F } ( x )  0$ as $N \to \infty$

Proof. $\begin{array} { r } { F ( x ) = x \sum _ { k = 0 } ^ { N - 1 } ( 1 - x ) ^ { k } } \end{array}$ and $F ^ { \prime } ( x ) = N ( 1 - x ) ^ { N - 1 }$ ; divide and shift the index by $N - 1$ Each summand $( 1 - x ) ^ { - j } \geq 1$ and is nondecreasing in x, giving the bound and monotonicity; the sum diverges geometrically in N for fixed x, giving the limit. □

Assumption 1 (At most one visit). Almost surely, every successful one-rollout trajectory visits each demonstrated proof state $x _ { t - 1 }$ at most once. This holds automatically when states are historyaugmented (the search tree), which is the case in our setting.

Lemma 2 (Unit elasticity). Under Equation (19) and Assumption 1, partition the non-demonstrated success mass at step t as $\dot { b } = b _ { t } ^ { \mathrm { s h } } + b _ { t } ^ { \mathrm { s i b } } + b _ { t } ^ { \mathcal { O } } .$ : trajectories that traverse the demonstrated transition $( x _ { t - 1 } , y _ { t } ^ { * } )$ but are not $\tau ^ { * }$ ; trajectories that visit $x _ { t - 1 }$ and deviate there; and trajectories that never visit $x _ { t - 1 } .$ . Then

$$
p _ { t } \frac { \partial P _ { \mathrm { a n y } } } { \partial p _ { t } } = \left( a + b _ { t } ^ { \mathrm { s h } } \right) - \frac { p _ { t } } { 1 - p _ { t } } b _ { t } ^ { \mathrm { s i b } } \leq P _ { \mathrm { a n y } } , \qquad i . e . \qquad \eta _ { t } : = \frac { p _ { t } \partial _ { p _ { t } } P _ { \mathrm { a n y } } } { P _ { \mathrm { a n y } } } \leq 1 .\tag{23}
$$

No demonstrated tactic can move the total success probability excessively. $( \eta _ { t }$ may be negative when sibling mass dominates.)

Proof. By Assumption 1 each successful trajectory contains the conditional at $x _ { t - 1 }$ at most once. Under Equation (19) its probability is therefore of the form $p _ { t } C , ( 1 - p _ { t } ) \rho _ { t } ( y ) C$ , or C with $C$ free of $p _ { t }$ , according to which subset of b it belongs. Differentiate termwise and multiply by $p _ { t }$ . The inequality follows since the first term is at most $P _ { \mathrm { a n y } }$ and the second is nonpositive. □

Theorem 4 (Weight squeeze for Pass@N). Under Assumption 1, for every step t and budget N,

$$
w _ { t } ^ { \star } \ = \ \eta _ { t } H _ { F } ( P _ { \mathrm { a n y } } ) \ \leq \ H _ { F } ( P _ { \mathrm { a n y } } ) \ \leq \ H _ { F } ( a ) \ = \ { \widehat w } _ { t } \ \leq \ 1 \ = \ w _ { t } ^ { \mathrm { C E } } .\tag{24}
$$

Consequently $\lvert \widehat { w } _ { t } - w _ { t } ^ { \star } \rvert \leq \lvert 1 - w _ { t } ^ { \star } \rvert .$ : CAT weakly dominates CE at every step and every budget, with strict domination whenever any inequality is strict.

Proof. $w _ { t } ^ { \star } = H _ { F } ( P _ { \mathrm { a n y } } ) \eta _ { t }$ and $\widehat { w } _ { t } = H _ { F } ( a ) \cdot 1$ by the chain rule and $p _ { t } \partial _ { p _ { t } } a = a$ . Apply Lemma 2, then $a \le P _ { \mathrm { a n y } }$ with ${ \cal { H } } _ { F }$ nonincreasing, then $H _ { F } \leq 1 ( \mathrm { L e m m a } 1 )$ . Domination: $w _ { t } ^ { \star } \leq \widehat { w } _ { t } \leq 1$ gives $\widehat { w } _ { t } - w _ { t } ^ { \star } \leq 1 ^ { \setminus } w _ { t } ^ { \star }$ □

Error decomposition. The gap localizes into the two ways the surrogate is blind:

$$
\widehat w _ { t } - w _ { t } ^ { \star } = \underbrace { \left[ H _ { F } ( a ) - H _ { F } ( P _ { \mathrm { a n y } } ) \right] } _ { \mathrm { i g n o r e d b y p a s s ~ m a s s } } + \underbrace { \left( 1 - \eta _ { t } \right) H _ { F } ( P _ { \mathrm { a n y } } ) } _ { \mathrm { s h a r i n g / r e a l l o c a t i o n ~ s t r u c t u r e } } .\tag{25}
$$

The first term is the objective-level blindness to b; the second is present even at $N = 1$ and reflects that raising $p _ { t }$ proportionally starves sibling alternatives at $x _ { t - 1 }$ while riding along with sharedprefix alternatives.

Remark. A useful strengthening is assuming $b \leq \varepsilon a .$ , the mean value theorem bounds the first term of Equation (25) by $\varepsilon a \operatorname* { s u p } _ { x \in [ a , a + b ] } | H _ { F } ^ { \prime } ( x ) |$ , giving a quantitative guarantee. This assumption is in a sense that the expert trace takes up a nontrivial chunk of the probability mass of correct proofs.

## C.3 BEST-FIRST SEARCH: CENSORING AND EXCURSIONS

For BFS the surrogate is the retry objective of Equation (14),

$$
{ \widehat { J } } ( N ) = \mathbb { P } ( S _ { L } \leq N ) , \qquad S _ { L } = \sum _ { t = 1 } ^ { L } T _ { t } , \qquad T _ { t } \sim \operatorname { G e o m } ( p _ { t } ) { \mathrm { ~ i n d e p e n d e n t . } }\tag{26}
$$

Unlike Pass@N, a deployed miss does not cost exactly one budget unit. The search may wander off-trace before the trace node is re-selected, or it may find a proof out there without returning. We separate these two events, then approximate each.

## C.3.1 THE CAT WEIGHT IS A CRITICALITY

Lemma 3 (CAT weight as criticality). Let $T _ { t } ^ { \prime }$ be an independent copy of $T _ { t }$ and $p _ { i } \in ( 0 , 1 )$ . For Equation (26),

$$
\widehat { w } _ { t } ( N ) = 1 - \frac { \mathbb { P } ( S _ { L } + T _ { t } ^ { \prime } \leq N ) } { \mathbb { P } ( S _ { L } \leq N ) } = \mathbb { P } ( S _ { L } \leq N < S _ { L } + T _ { t } ^ { \prime } | S _ { L } \leq N ) .\tag{27}
$$

Hence $0 \le \widehat { w } _ { t } ( N ) \le 1 , \widehat { w } _ { t } ( L ) = 1 , \widehat { w } _ { t } ( N ) \to 0$ as $N  \infty ,$ , and $\widehat { w } _ { t } ( N )$ is non-increasing in $N .$

Proof. For $T \sim \operatorname { G e o m } ( p )$ and any bounded ψ, $p \partial _ { p } \mathbb { E } [ \psi ( T ) ] = \mathbb { E } [ \psi ( T ) ] - \mathbb { E } [ \psi ( T + T ^ { \prime } ) ]$ . Apply with $\begin{array} { r } { \psi ( k ) = \mathbb { P } ( R _ { t } + k \le N ) , R _ { t } = \sum _ { i \neq t } T _ { i } } \end{array}$ , and divide by $\widehat { J } ( N )$ . For monotonicity: the pmf of $S _ { L }$ is a convolution of log-concave pmfs, hence log-concave, and so is its CDF $F ;$ thus $F ( n - k ) / F ( n )$ is nondecreasing in n for fixed $\bar { k \ : } \geq 1$ , and $1 - \breve { \widehat { w } } _ { t } ( n ) = \mathbb { E } _ { T _ { t } ^ { \prime } } [ F ( n - T _ { t } ^ { \prime } ) / F ( n ) ]$ is nondecreasing.

## C.3.2 SEPARATING ALTERNATIVE PROOFS: THE BYPASS CENSORED COUPLING

To analyze the impact of off-trace proofs on BFS, consider an artificial ”bypass censored process”. This process behaves exactly like standard, deployed BFS. Except, if it ever samples a tactic which would yield a proof other than the expert proof, we replace the observation of $x _ { Q E D }$ by $x _ { e r r }$ . So it only succeeds if it returns the demonstrated trace but it is allowed to be off path during the search.

Lemma 4 (Bypass Censoring). Let N be the initial budget, $J ^ { \mathrm { t r a p } } ( N )$ be the censored process’ success probability and let $\beta ( N )$ be the probability that the deployed run samples a transition which would have been censored ifwe had instead run the censored process. Then

$$
J ^ { \mathrm { t r a p } } ( N ) \ \leq \ J ^ { \mathrm { t r u e } } ( N ) \ \leq \ J ^ { \mathrm { t r a p } } ( N ) + \beta ( N ) .\tag{28}
$$

Proof. Couple the two runs on the same tactic draws and selector randomness. They are identical until the deployed run first samples a transition which would have been censored: at that moment, the deployed run has succeeded. If no censored transition is ever sampled, outcomes agree. So censored success implies deployed success, and deployed success implies censored success or divergence.

This is just to say that alternate proof paths can only add success. To use this to analyze the weights, split the true success probability by which event occurs first, $J ^ { \mathrm { t r u e } } = J ^ { \dagger } + B \left( J ^ { \dagger } \right.$ : completion of the demonstrated trace; $B \colon$ completion of an alternate trace). From looking at the sum of elasticities,

$$
{ \frac { \partial J ^ { \mathrm { t r u e } } } { \partial p _ { t } } } = { \frac { \partial J ^ { \dagger } } { \partial p _ { t } } } + { \frac { \partial B } { \partial p _ { t } } }
$$

$$
\begin{array} { r l r } & { } & { w _ { t } ^ { \star } = \frac { p _ { t } } { J ^ { \mathrm { t r u e } } } \frac { \partial J ^ { \dagger } } { \partial p _ { t } } + \frac { p _ { t } } { J ^ { \mathrm { t r u e } } } \frac { \partial B } { \partial p _ { t } } \qquad } \\ & { } & { w _ { t } ^ { \star } = \left( \frac { J ^ { \dagger } } { J ^ { \mathrm { t r u e } } } \right) \left( \frac { p _ { t } } { J ^ { \dagger } } \frac { \partial J ^ { \dagger } } { \partial p _ { t } } \right) + \left( \frac { B } { J ^ { \mathrm { t r u e } } } \right) \left( \frac { p _ { t } } { B } \frac { \partial B } { \partial p _ { t } } \right) } \\ & { } & { w _ { t } ^ { \star } = \lambda w _ { t } ( J ^ { \dagger } ) + ( 1 - \lambda ) w _ { t } ( B ) , \qquad \lambda = \frac { J ^ { \dagger } } { J ^ { \mathrm { t r u e } } } , } \end{array}\tag{29}
$$

holds exactly.

This mixture identity reveals that the true weight is subject to a tug-of-war between two competing ”channels” of success.

First, the trap channel $( J ^ { \dagger } )$ captures trajectories that wander into dead ends but eventually return to complete the demonstrated proof. Because these off-trace excursions consume a shared compute budget, they increase the bottleneck on the remaining path. This increased criticality pushes the weight $w _ { t } ( { \dot { J } } ^ { \dagger } )$ up relative to the trace-supported surrogate $\widehat { w } _ { t }$ (formalized in Proposition 3).

Conversely, the bypass channel (B) captures trajectories that successfully discover alternative, unobserved proofs. The existence of these alternative solutions makes the exact demonstrated tactic less critical for global success, which pushes the weight down, mirroring the elasticity squeeze observed in the Pass@N analysis. Thus, we will focus the remaining analysis on the effect of the trap channel.

One structural caveat remains: the true BFS selector’s geometric-mean scores dynamically depend on $p ,$ a dependence the trace-supported surrogate erases to maintain independence between steps. Because Levin scores are cumulative, increasing an early on-trace probability $p _ { s }$ creates a cascading advantage for all subsequent trace nodes. This allows the trace to win frontier comparisons sooner, actively suppressing the length of future off-trace excursions. This positive feedback loop means the true search penalizes on-trace mistakes even more heavily than our surrogate models. Consequently, the trap channel’s upward pressure on the deployed weight $w _ { t } ^ { \star }$ is strictly stronger in reality. We will, however, treat the mixture identity (Equation (29)) as exact.

## C.3.3 THE TRAP CHANNEL: EXPLORATION WASTE AND SHARED-BUDGET BRACKETING

In the trap channel, when BFS samples an off-trace tactic at state $x _ { t - 1 }$ , the algorithm embarks on an off-trace excursion until it returns back to the trace. To quantify the computational toll of deviations, let $E _ { t , j } \in \{ 0 , 1 , . . . \} \cup \{ \infty \}$ denote the number of wasted node expansions spent off-trace following the j-th miss at state $x _ { t - 1 }$ before the search backtracks and re-selects the trace node $( E _ { t , j } = \infty$ if the algorithm is permanently absorbed). The trap-channel success probability is then precisely represented by:

$$
J ^ { \mathrm { t r a p } } ( N ) = \mathbb { P } \left( \sum _ { t = 1 } ^ { L } T _ { t } + \sum _ { t = 1 } ^ { L } \sum _ { j = 1 } ^ { T _ { t } - 1 } E _ { t , j } \ \le \ N \right) ,\tag{30}
$$

where $T _ { t } \sim \mathrm { G e o m } ( p _ { t } )$ tracks on-trace trials. Different choices of weight during training can be viewed as distinct assumptions on $E { : }$

• Minimal waste $( E \equiv 0 ) \colon$ Every off-trace tactic is instantly rejected by the environment $( x _ { e r r } )$ A miss costs exactly 1 expansion, recovering our CAT retry surrogate exactly $( J ^ { \mathrm { t r a p } } ( N ) =$ $\widehat { J } ( N ) )$ .

• Permanent absorption $( E \equiv \infty )$ : Any off-trace mistake traps the search indefinitely. Here, $\begin{array} { r } { J ^ { \mathrm { t r a p } } ( N ) = \prod _ { t = 1 } ^ { L } p _ { t } } \end{array}$ for any budget $N \geq L$ , yielding unit elasticity $( w _ { t } ^ { \star } = 1 )$ , recovering the exact regime where standard SFT weighting is optimal.

Between these two extremes lies the realistic setting where off-trace excursions consume finite, nonzero compute before backtracking. To isolate how this exploration waste alters the true weight, we first examine the case of a fixed return cost.

Assumption 2 (Deterministic excursion cost). In the trap-channel process, every miss incurs a total expenditure of $\kappa \geq 1$ expansions (the initial trial plus an excursion of length $E \equiv \kappa - 1 )$ before returning the search to $x _ { t - 1 } ,$ successful trials cost 1 expansion; κ is independent of the policy probabilities $p .$

Proposition 3 (Shared-budget bracketing). Under Assumption 2, let $N \geq L$ be an integer budget and $p _ { s } \in ( 0 , 1 )$ for all s. Completing the demonstrated path in the bypass-censored process within budget N is equivalent to evaluating the retry surrogate at a reduced effective budget:

$$
J ^ { \mathrm { t r a p } } ( N ) = \widehat { J } ( N _ { \mathrm { e f f } } ) , \qquad N _ { \mathrm { e f f } } = L + \frac { N - L } { \kappa } \leq N .\tag{31}
$$

For noninteger budget arguments $B ,$ interpret ${ \widehat { J } } ( B ) = { \widehat { J } } ( \lfloor B \rfloor )$ and $\widehat { w } _ { t } ( B ) = \widehat { w } _ { t } ( \lfloor B \rfloor )$ .

Define the trap-channel weight by $w _ { t } ^ { \mathrm { t r a p } } ( N ) : = w _ { t } ( J ^ { \mathrm { t r a p } } ( N ) )$ . Because κ is independent of $\cdot _ { p , }$ the effective-budget identity and Lemma 3 give

$$
\widehat { w } _ { t } ( N ) \leq w _ { t } ^ { \mathrm { t r a p } } ( N ) = \widehat { w } _ { t } ( N _ { \mathrm { e f f } } ) \leq 1 = w _ { t } ^ { \mathrm { C E } } .\tag{32}
$$

Proof. Completing step t requires $G _ { t } \sim \mathrm { G e o m } ( p _ { t } )$ trials costing $\kappa ( G _ { t } - 1 ) + 1$ total expansions. Summing over the trace yields a total cost $\kappa S _ { L } - ( \kappa - 1 ) L \leq N \iff S _ { L } \leq N _ { \mathrm { e f f } }$ . The bracket directly follows since $N _ { \mathrm { e f f } } \leq N$ for $\kappa \geq 1$ and $\widehat { w } _ { t } ( \cdot )$ is non-increasing. □

In real policy-guided Best-First Search, excursions do not incur deterministic costs; rather, they exhibit stochastic, bounded trajectories $( E \le \bar { \kappa } - 1 \mathsf { a . s . } )$ . Crucially, we can actively engineer this bound during deployment to prevent permanent absorption.

Enforcing Bounded Excursions via Global Search Restarts. In practical ATP deployments, we actively prevent policies from being trapped in cyclical or deep, unguided proof attempts $( P ( E = \infty ) = 0 )$ by deploying a restart-based Best-First Search heuristic. Specifically, if the global BFS does not discover a complete proof within a step threshold of $M < N$ total node expansions, the active search tree is aborted and the algorithm resets entirely to the initial theorem state $x _ { 0 }$ to begin a fresh run with the remaining compute budget. However, the cost to setting such an M too low is aborting a possibly prominent path so in practice this M is fairly high. This restart mechanism guarantees that no single exploratory trajectory can waste more than M expansions off-trace. Consequently, the exploration waste incurred before re-attempting the demonstrated trajectory is strictly bounded $( E \le M \mathrm { a . s }$ ., enforcing a maximum excursion severity $\bar { \kappa } \leq M + 1 )$ . While enforcing this test-time constraint eliminates the SFT ceiling $( w _ { t } ^ { \star } < 1 )$ , combining it with Proposition 3 reveals an irreducible tension between trace-only supervision and search optimization.

Proposition 4 (Irreducible Ambiguity under Bounded Excursions). Call a per-step weighting rule trace-supported if it depends solely on training-time observables along $\tau ^ { * }$ (specifically, the demonstrated probabilities $\{ \hat { p } _ { s } \} _ { s = 1 } ^ { L }$ , the search strategy ${ \mathcal A } ,$ and the initial budget ${ \mathrm { \bar { \rho } } } _ { N } ) .$ . Suppose test-time search engineering enforces an upper bound on excursion waste such that $\kappa \in [ 1 , \bar { \kappa } ]$ ]for afinite limit $\bar { \kappa } > 1$ . Then:

(i) The true deployed weight w<sup>⋆</sup> is strictly confined to the compressed bracket:

$$
w _ { t } ^ { \star } \in \left[ \widehat { w } _ { t } ( N ) , \widehat { w } _ { t } \biggl ( L + \frac { N - L } { \bar { \kappa } } \biggr ) \right] .\tag{33}
$$

(ii) Because the realized excursion waste κ depends on unobserved off-trace dynamics, no tracesupported objective can uniquely identify $w _ { t } ^ { \star }$ . Any trace-supported rule $w _ { t }$ incurs a worstcase approximation error bounded below by halfthe bracket width:

$$
\operatorname* { s u p } _ { \kappa \in [ 1 , \bar { \kappa } ] } \big | w _ { t } - w _ { t } ^ { \star } ( \kappa ; N ) \big | \geq \frac 1 2 \left( \widehat { w } _ { t } \Big ( L + \frac { N - L } { \bar { \kappa } } \Big ) - \widehat { w } _ { t } ( N ) \right) > 0 .\tag{34}
$$

Proof. By Proposition 3, an environment with effective excursion cost κ realizes the deployed weight $\begin{array} { r } { w _ { t } ^ { \star } = \dot { \widehat { w } } _ { t } ( L + \frac { N - L } { \kappa } ) } \end{array}$ . Because environments with $\kappa = 1$ (instant syntax error) and $\kappa = \bar { \kappa }$ (maximum restart cap $\ddot { M } )$ yield identical training-time observables along $\tau ^ { * }$ , any trace-supported rule must output the same scalar $w _ { t }$ across both environments. The minimax lower bound follows directly from applying the triangle inequality to the interval endpoints. □

## C.4 BUDGET SCALING OF THE WEIGHT ERROR: HUMP OR FLOOR

Using the understanding we’ve built, we can understand how the error in our approximation scales. A key intuition we will need is that the surrogate’s weight error is generally not monotone in N. When N is minimal $( N = 1$ in our setting), we are just performing a single rollout. We aren’t really performing search, and both the surrogate and real weights understand this, so the error of the surrogate is 0. Conversely, as $N  \infty$ , our probability of success (with some caveats) becomes 1. Thus, both the surrogate and deployed weights tend to 0. The error only really occurs in the middle between these two extremes.

The minimal-budget endpoint is universal. At the smallest budget that can possibly succeed, any strategy must execute $\tau ^ { * }$ with no misses: both $J ^ { A }$ (trap channel) and $\widehat { J } _ { A }$ equal $\prod _ { t } p _ { t }$ , whose elasticity is 1 at every step. So $\widehat { w } _ { t } = w _ { t } ^ { \star } = 1 = w _ { t } ^ { \mathrm { C E } }$ and all three rules coincide (alternate proofs shorter than L are the only exception, entering through B in Equation (29)).

Proposition ${ \textbf { 5 } } ( \mathrm { P a s s } @ N ;$ : vanishing error). Fix the policy with $a \in ( 0 , 1 )$ . Then $\widehat { w } _ { t } ( 1 ) = 1$ and $w _ { t } ^ { \star } ( 1 ) = \eta _ { t }$ ; both $\widehat { w } _ { t } ( N )$ and $w _ { t } ^ { \star } ( \bar { N } )$ tend to 0 as $N \to \infty ;$ the bypass term ofEquation (25) is zero at $N = 1$ and vanishes as $N  \infty ( a$ hump), while the reallocation term decays monotonically from $1 - \eta _ { t }$ . In particular, the CAT error tends to 0 while the CE error $| 1 - w _ { t } ^ { \star } |$ tends to 1.

Proof. A $\mathrm { ~ \ a t } N = 1 , H _ { F } \equiv 1 . \mathrm { \ A s \ } N \to \infty , H _ { F } ( x ) \to 0$ for fixed $x > 0 ( \mathrm { I }$ Lemma 1); apply to $x = a$ and $x = P _ { \mathrm { a n y } }$ in Theorem 4 and Equation (25). □

For BFS, the outcome depends on the excursions. We isolate it to a specific case.

Assumption 3 (Absorbing-excursion model). In the trap-only censored process, each miss at step t, independently of everything else, launches a non-returning excursion $( E = \infty )$ with probability $q _ { t } \in [ 0 , 1 ]$ , and otherwise an excursion of bounded length $E \le \bar { \kappa } - 1$

Proposition 6 (Hump or floor). Under Assumption 3:

(a) (Recurrence ⇒ hump.) $I f q _ { t } = 0$ for all t and excursions are deterministic $( E \equiv \kappa - 1 )$ , the trap-channel error is $w _ { t } ^ { \star } ( N ) - \widehat { w } _ { t } ( N ) = \widehat { w } _ { t } ( N _ { \mathrm { e f f } } ) - \widehat { w } _ { t } ( N ) \geq 0 ,$ , which is 0 at $N = L ,$ tends to 0 as $N  \infty ( s i n c e \ N _ { \mathrm { e f f } }  \infty )$ , and is positive in between whenever $\kappa > 1 .$ : the error is humped. For bounded random excursions with no absorption, the error is likewise zero at $N = \bar { L }$ and vanishes as $N \to \infty ;$ itsfinite-budget behavior is governed by Equation (30).

(b) (Absorption ⇒floor.) $H q _ { t } > 0$ , then as $N  \infty ,$

$$
J ^ { \mathrm { t r a p } } ( N ) \longrightarrow \prod _ { t = 1 } ^ { L } \frac { p _ { t } } { p _ { t } + ( 1 - p _ { t } ) q _ { t } } , ~ w _ { t } ^ { \star } ( N ) \longrightarrow \frac { q _ { t } } { p _ { t } + ( 1 - p _ { t } ) q _ { t } } ~ > ~ 0 ,\tag{35}
$$

while $\widehat { w } _ { t } ( N ) \to 0 ;$ the CAT error converges to the strictly positive floor (35). As $q _ { t } \to 1$ thefloor tends to 1 and CE becomes asymptotically exact at step t, recovering the permanentabsorption limit; as $q _ { t } \to 0 ,$ , the limiting error at that step vanishes.

Proof. (a) is Proposition 3 plus Lemma 3. (b) With unlimited budget, finite excursions are free, so step t succeeds iff $y _ { t } ^ { * }$ is sampled before an absorbing excursion: per trial, advance $\mathbf { w . p . } p _ { t }$ , absorb w.p. $( 1 - p _ { t } ) q _ { t }$ , retry otherwise, giving per-step success $f ( p _ { t } ) = p _ { t } / ( p _ { t } + ( 1 - p _ { t } ) q _ { t } )$ , independent across steps. Direct computation gives ∂ log $f / \partial \log p = q / ( p + ( 1 - p ) q )$ The interchange of limit and derivative is justified within the model by monotone convergence of $\begin{array} { r } { J ^ { \mathrm { t r a p } } ( N ) \uparrow \prod _ { t } \bar { f } ( p _ { t } ) } \end{array}$ together with monotonicity of $w _ { t } ^ { \star } ( N )$ inherited from Lemma 3 applied at the extended-cost representation. □

Approximation is Incorrect Asymptotically, but Weights Converge. The pieces we don’t account for in our surrogate (the alternate proof mass $\beta ( N )$ and total excursion expansions) do grow with N. But their effect on the weights is self-limiting. Both objectives saturate near 1, and every elasticity is near 0, so no weighting rule can be far from the truth. This is a key advantage of test-time aligned training.

## D DEPTH FIRST SEARCH VARIANTS

Depth First Search (DFS) is not a standard algorithm used in policy guided search. We introduce it in this context because analyzing it and its variants serve as excellent examples about the mechanics of constructing search aligned loss functions.

## D.1 STANDARD DFS

Standard DFS utilizes its computational budget to sequentially construct a single path. When the algorithm expands a node $x _ { t - 1 }$ and the tactic is rejected $( \mathcal { T } ( x _ { t - 1 } , y _ { t } ) = x _ { e r r } )$ , the search does not abandon the entire trajectory. Instead, the decision rule dictates a retreat of exactly one step to the immediate parent state $x _ { t - 1 }$ , expending another unit of compute to sample an alternative tactic. This localized trial-and-error continues until the algorithm either successfully reaches $x _ { Q E D }$ or exhausts the global node expansion budget N.

Finally, we note that because standard DFS commits rigidly to a valid path, its chronologically local backtracking is a structural weakness. If the search ventures into a deep, valid, but useless branch, the localized backtracking cannot escape the broader strategic error. The LLM will simply exhaust its entire computational budget trapped in the depths of a spurious path.

Instantiation. Define $\widehat { c } = \langle x , b \rangle$ , where x is the active trace state and b is the remaining node expansion budget. The trace-supported frontier is always a singleton, ${ \widehat { \mathcal { F } } } = \{ x \}$

The positive update advances the path and consumes one expansion. For the negative update, an offtrace tactic immediately collapses the search back to the parent $x _ { t - 1 }$ at the cost of one expansion. Thus, the updates are:

$$
\widehat { U } _ { \mathrm { D F S } } ^ { + } ( x _ { t - 1 } , b ) = ( x _ { t } , b - 1 ) , \qquad \widehat { U } _ { \mathrm { D F S } } ^ { - } ( x _ { t - 1 } , b ) = ( x _ { t - 1 } , b - 1 )
$$

Because the frontier is always a single node, the selector is trivial: $S _ { \mathrm { D F S } } ( x \mid \widehat { c } ) = 1$ . Consequently, the trace-supported recurrence for DFS perfectly mirrors the recurrence for BFS (Equation (13)):

$$
\widehat { J } _ { \mathrm { D F S } } ( b , t ) = p _ { t } \widehat { J } _ { \mathrm { D F S } } ( b - 1 , t + 1 ) + ( 1 - p _ { t } ) \widehat { J } _ { \mathrm { D F S } } ( b - 1 , t ) .
$$

Thus, the closed form loss will collapse to that which we derived for BFS.

## D.2 RANDOMIZED DEPTH-FIRST SEARCH (RDFS)

To mitigate the vulnerability of DFS to spurious paths, we introduce a randomized variant. RDFS disrupts this failure mode by altering the backtracking target. When the policy $\pi _ { \theta }$ reaches a dead end, the algorithm does not simply return to $x _ { t - 1 }$ . Instead, it selects a state uniformly at random from the entire active path prefix $( x _ { 0 } , x _ { 1 } , \dots , x _ { t - 1 } )$ and resumes expansion from there. This allows the search to stochastically jump out of deep traps, efficiently reallocating compute to higher-level branching points while preserving the depth-seeking advantages of the standard algorithm. A visualization is given in Figure 6.

Instantiation. Let $\widehat { c } = \langle t , b \rangle$ , with the frontier tracking the full active path $\widehat { \mathcal { F } } = \{ 1 , 2 , \dots , t \}$ . To incorporate randomized backtracking, the randomness is injected directly into the selector. When a dead end is reached, rather than the update rule deterministically collapsing the path, the negative update triggers the selector to sample a state uniformly at random from the entire available frontier ${ \hat { \mathcal { F } } } .$

Trace-supported recurrence. Because of this randomized selector, the compute consumed at depth t cannot be modeled as a simple sequence of independent geometric random variables. Instead, the trace-supported objective forms a dynamic program.

Let $\widehat { J } _ { \mathrm { R D F S } } ( b , t )$ denote the exact probability of discovering the complete proof given that the search is currently evaluating the trace state at index t with a remaining budget b. The boundary conditions are:

$$
\widehat { J } _ { \mathrm { R D F S } } ( b , L + 1 ) = 1 \quad \mathrm { f o r ~ a l l } \ b \geq 0 \quad ( \mathrm { P r o o f ~ s u c c e s s f u l l y ~ c o m p l e t e d } )\tag{36}
$$

$$
\widehat { J } _ { \mathrm { R D F S } } ( 0 , t ) = 0 \quad \mathrm { f o r ~ a l l } \ t \leq L \quad ( \mathrm { C o m p u t e ~ b u d g e t ~ e x h a u s t e d } )\tag{37}
$$

For $b > 0$ and $t \leq L$ , the recurrence relation resolving Equation (5) branches into two outcomes. With probability $p _ { t } ,$ , the model generates the correct tactic and advances. With probability $( 1 - p _ { t } )$ the model fails, and the selector backtracks uniformly to any valid ancestor index $k \in \{ 1 , \ldots , t \}$ Thus, the recurrence becomes:

$$
\widehat { J } _ { \mathrm { R D F S } } ( b , t ) = p _ { t } \widehat { J } _ { \mathrm { R D F S } } ( b - 1 , t + 1 ) + ( 1 - p _ { t } ) \frac { 1 } { t } \sum _ { k = 1 } ^ { t } \widehat { J } _ { \mathrm { R D F S } } ( b - 1 , k )\tag{38}
$$

Unlike Pass@N or BFS, this objective does not reduce to a standard closed-form probability distribution.

## D.3 VALUE GUIDED DEPTH-FIRST SEARCH (VDFS)

To mitigate the vulnerability of DFS while leveraging the policy’s own internal confidence, we introduce Value Guided DFS (VDFS). Unlike RDFS, which backtracks uniformly, VDFS makes an informed retreat. When the search reaches a dead end, it backtracks to an ancestor state in the active path prefix with probability proportional to that state’s Levin score (from Equation (12)). By using the geometric mean of the policy’s probabilities as a confidence metric, the search preferentially returns to the most promising points on the current path. This concentrates the remaining computational budget on states the policy deems most reliable, while maintaining enough stochasticity to escape deep, spurious branches. A visualizaiton can be seen in Figure 7.

Instantiation. Let $\widehat { c } = \langle t , b \rangle$ , where t is the active trace index (corresponding to the state $x _ { t - 1 } )$ and b is the remaining node expansion budget. The trace-supported frontier tracks the full active path prefix of states: $\widehat { \mathcal { F } } = \{ x _ { 0 } , x _ { 1 } , \ldots , x _ { t - 1 } \}$

To compute the backtrack distribution, we evaluate the Levin score for each state on the frontier. Let $s _ { k }$ denote the Levin score of state $x _ { k }$ . For the root, $s _ { 0 } = 1$ . For $k \geq 1$ , the score is the geometric mean of the on-trace probabilities leading to that state:

$$
s _ { k } = \left( \prod _ { j = 1 } ^ { k } p _ { j } \right) ^ { \frac { 1 } { k } }\tag{39}
$$

The positive update advances the proof by one step: $\widehat { U } _ { \mathrm { V D F S } } ^ { + } ( t , b ) = ( t + 1 , b - 1 )$ . For the negative update, an off-trace tactic consumes a unit of budget and triggers a backtrack. The selector samples a backtrack target state $x _ { k } \in { \widehat { \mathcal { F } } }$ with probability proportional to its score $s _ { k }$ . Transitioning to $x _ { k }$ sets the new active trace index to $k + 1$ . Thus, the negative update maps stochastically:

$$
\widehat { U } _ { \mathrm { V D F S } } ^ { - } ( t , b ) = ( k + 1 , b - 1 ) , \quad \mathrm { w h e r e } \ k \sim P ( k ) = \frac { s _ { k } } { \sum _ { i = 0 } ^ { t - 1 } s _ { i } }\tag{40}
$$

Trace-supported recurrence. Let $\widehat { J } _ { \mathrm { V D F S } } ( b , t )$ denote the exact probability of discovering the complete proof from the trace state at index t with remaining budget b. The boundary conditions are identical to RDFS:

$$
\widehat { J } _ { \mathrm { V D F S } } ( b , L + 1 ) = 1 \quad \mathrm { f o r ~ a l l } \ b \geq 0 \quad ( \mathrm { P r o o f ~ s u c c e s s f u l l y ~ c o m p l e t e d } )\tag{41}
$$

$$
\widehat { J } _ { \mathrm { V D F S } } ( 0 , t ) = 0 \quad \mathrm { f o r ~ a l l } \ t \leq L \quad ( \mathrm { C o m p u t e ~ b u d g e t ~ e x h a u s t e d } )\tag{42}
$$

For $b > 0$ and $t \leq L ,$ the recurrence resolving Equation (5) weights the backtrack sum by the Levin scores. With probability $p _ { t } ,$ the model generates the correct tactic and advances. With probability $( 1 - p _ { t } )$ , the model fails, and the selector backtracks to state $x _ { k } \left( k \in \{ 0 , \ldots , t - 1 \} \right)$ ) according to the score distribution. The recurrence becomes:

$$
\widehat { J } _ { \mathrm { V D F S } } ( b , t ) = p _ { t } \widehat { J } _ { \mathrm { V D F S } } ( b - 1 , t + 1 ) + ( 1 - p _ { t } ) \sum _ { k = 0 } ^ { t - 1 } \frac { s _ { k } } { \sum _ { i = 0 } ^ { t - 1 } s _ { i } } \widehat { J } _ { \mathrm { V D F S } } ( b - 1 , k + 1 )\tag{43}
$$

Like RDFS, this objective doesn’t have a closed form. Notably, the gradients flow not only through the direct local transitions but also through the Levin scores $s _ { k } .$ , which are themselves functions of the policy probabilities. This couples the training objective, forcing the model to calibrate its on-path probabilities to optimize the resulting backtracking distribution.

## E MONTE CARLO TREE SEARCH (MCTS)

The strategies considered so far commit their budget by a rule fixed in advance: DFS always retries the deepest node, RDFS backtracks uniformly, and VDFS backtracks in proportion to a static pernode score. Monte Carlo Tree Search (Kocsis & Szepesvari, 2006; Coulom, 2006) instead allocates´ compute according to statistics it accumulates as the search runs. We include it because it is the only strategy in our taxonomy whose selector depends on the history of the search rather than on the current configuration alone, and it therefore tests our construction at its least convenient.

MCTS is a 4-phase process at deployment. At each node, Selection either commits to its most promising child or stops to gain more information about another tactic at the current node. The choice to widen is governed by a progressive-widening schedule: a node that has been visited n times is permitted $\lceil \sqrt { n + 1 } \rceil$ distinct tactic samples (Coulom, 2007). Expansion samples one tactic from $\pi _ { \theta }$ at the selected node and submits it to the kernel, consuming one node expansion. Evaluation assigns the resulting state a value; in the variant, we choose to use the policy’s own Levin score Equation (12) rather than a learned critic, so that MCTS requires no component the other strategies don’t utilize. Backup propagates that value to every ancestor on the selected path, updating the statistics that drive future selections. A visualization is given in Figure 8.

Instantiation. With abuse of notation, let a configuration be

$$
\widehat { c } = \big \langle ( n _ { k } , m _ { k } , W _ { k } ) _ { k = 0 } ^ { L - 1 } , \ : b \big \rangle ,
$$

where $n _ { k }$ is the visit count of the state $x _ { k } , m _ { k }$ the number of tactics already sampled there, $W _ { k }$ the accumulated backed-up value, and b the remaining expansion budget. The trace-supported frontier is the prefix $\{ x _ { 0 } , \dots , x _ { L - 1 } \}$ . The selector is not stochastic: the descent stops at the shallowest node that is either allowed a new tactic by the widening schedule or has no child yet,

$$
\begin{array} { r l } & { d ( \widehat { c } ) \ = \ \operatorname* { m i n } \Big \{ k \ : \ m _ { k } < \big \lceil \sqrt { n _ { k } + 1 } \big \rceil \ \mathrm { ~ o r ~ } \ x _ { k + 1 } \ \mathrm { n o t } \ y \mathrm { e t } \ \mathrm { r e a c h e d } \Big \} , } \\ & { \qquad \widehat { S } _ { \mathrm { M C T S } } ( x _ { k } \mid \widehat { c } ) = { \bf 1 } \{ k = d ( \widehat { c } ) \} . } \end{array}\tag{44}
$$

The updates apply the expansion and the backup together. Writing $d = d ( { \widehat { c } } )$ , a demonstrated tactic advances the trace and backs up the value $V ( \bar { x _ { d + 1 } } )$ of the state it reaches, while a miss backs up zero:

$$
\widehat { U } _ { \mathrm { M C T S } } ^ { + } ( \widehat { c } ) : \quad m _ { d } + = 1 , \quad n _ { k } + = 1 , W _ { k } + = V ( x _ { d + 1 } ) \ \forall k \leq d , \quad b - = 1 ,\tag{45}
$$

$$
\widehat { U } _ { \mathrm { M C T S } } ^ { - } ( \widehat { c } ) : \quad m _ { d } \gets 1 , \quad n _ { k } \gets = 1 , W _ { k } + = 0 \ \forall k \mathop { \leq } d , \quad b \gets 1 ,\tag{46}
$$

with solved configurations (for the trace supported version) those in which depth L has been reached and dead configurations those with $b = 0 .$ . Both updates decrement the budget, so the recurrence terminates.

For BFS, the naive selector is degenerate on a demonstrated trace because a chain never offers two frontier candidates at once. The same erasure happens here, in a slightly different place. The UCT rule ranks the children of a node by an exploration-adjusted value estimate, but under the trace transition Equation (4) every node has at most one child, so there is never a comparison for it to make. What survives is only the widening schedule in Equation (44): the rule that decides how many tactic samples a node receives before the search commits to descending. Thus in practice, thi strategy trains like a local Pass@N at each node.

## F STRATEGY MATCHING WITHIN THE SEARCH-AWARE FAMILY

The main comparison evaluates search-aware CAT and search-agnostic UA across deployed strategies. Here, we isolate a different question: within the search-aware family, does matching the training objective to the deployed strategy improve proof success? Using the fixed-budget setup of Section 5.1, we evaluate the Pass@N and BFS search-aware adapters under both deployment strategies, with CE as a reference.

Table 3: Strategy matching within the search-aware family. Proof-success rates (%) at N = 256 tactic expansions. Columns are deployed strategies; rows are training objectives. Bold marks the higher point estimate among the two search-aware objectives in each column.
<table><tr><td>Training objective</td><td>Pass@N</td><td>BFS</td></tr><tr><td>CE</td><td>21.2</td><td>23.8</td></tr><tr><td>Search-aware CAT (Pass @ N)</td><td>26.6</td><td>24.5</td></tr><tr><td>Search-aware CAT (BFS)</td><td>23.7</td><td>25.1</td></tr></table>

The matched objective has the higher observed success rate under each deployment strategy. This supports strategy matching within the tested search-aware pair, complementing the comparison with shared UA in Table 1.

## G EXPERIMENTAL DETAILS

## G.1 COMMON SETUP

Environment. All experiments run against Lean 4 through LeanDojo 4.20.0 (Yang et al., 2023). The policy is prompted with the pretty-printed proof state and emits a single tactic, which is submitted to the Lean kernel; the kernel verdict determines the transition. One node expansion denotes one policy sample together with its kernel verification, and is the unit in which all budgets are expressed. Kernel interaction is serial within a theorem, so wall-clock cost per unsolved theorem grows linearly in the budget.

Data. Training and evaluation both draw on cat-searcher/leandojo-benchmark-4- random, a random split of mathlib4 (The mathlib Community, 2020). The alignment stage consumes 40,660 traced proofs comprising 76,003 (state, tactic) training examples. The evaluation pool is the 458 held-out theorems whose reference proofs contain between 2 and 5 tactic steps; there is no overlap between training and evaluation theorems. Reference proofs are used only to compute training weights and to define the pool. At evaluation, a theorem counts as solved if the search returns any kernel-verified proof, whether or not it matches the reference.

Model and adapters. The policy is Qwen2.5-Math-7B-Instruct (Qwen Team, 2024) in bfloat16. All fine-tuning uses LoRA (Hu et al., 2021) with rank r = 16, α = 32, dropout 0.05, applied to the query, key, value, output, gate, up, and down projections. This trains 40,370,176 of 7,655,986,688 parameters (0.53%). Gradient checkpointing is enabled throughout. All runs use a single NVIDIA A100-40GB. Training runs used FlashAttention-2 (Dao, 2024).

## G.2 TRAINING PROCEDURE AND HYPERPARAMETERS

Stage 1: Warm-up. A single supervised checkpoint is the common ancestor of every model we report. The first set of experiments branch off of two warm up epochs of SFT; while the second set branch off of only one. This discrepancy is a result of human error; however, given that runs are not compared cross-experiment, we find it reasonable to keep.

Stage 2: Alignment. From the warm-up checkpoint the runs branch. The control receives one further epoch of standard cross-entropy SFT. Each aligned model receives one further epoch over the same examples in the same order under the same optimizer settings, with the only difference being the scalar weight $\widehat { w } _ { t } ^ { A }$ applied to each tactic’s cross-entropy gradient. An alignment epoch is 19,001 optimizer steps and takes approximately 2.4 h.

Table 4: Optimization hyperparameters, identical for the control and every aligned run.
<table><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Schedule</td><td>linear decay, warm-up over  $5 \%$  of steps</td></tr><tr><td>Micro-batch size</td><td>4</td></tr><tr><td>Gradient accumulation</td><td>16</td></tr><tr><td>Effective batch size</td><td>64</td></tr><tr><td>Gradient clipping</td><td>1.0 (global  $\ell _ { 2 }$  norm)</td></tr><tr><td>Maximum sequence length</td><td>512 tokens</td></tr><tr><td>Precision</td><td>bfloat16 with autocast</td></tr><tr><td>Alignment epochs</td><td>1 (on top of the shared warm-up)</td></tr><tr><td>Random seed</td><td>42</td></tr></table>

Non-finite losses are skipped without an optimizer step; this occurs rarely and is logged. Examples exceeding the maximum sequence length are dropped at dataset construction rather than truncated, so no partially-supervised tactic enters training. The closed-form weights are evaluated directly in log space for numerical stability. The weights defined by a dynamic program are obtained by autodiff.

## G.3 INFERENCE AND EVALUATION

Serving. Generation uses vLLM 0.6.3 with the LoRA adapter supplied per request, GPU memory fraction 0.85, and maximum model length 4096 tokens. Sampling temperature is 0.8. The evaluation seed (42) fixes both the shuffle of the theorem pool and the sampling stream, so every configuration sees the identical theorems in the identical order; this is what licenses the paired analysis below and what makes a truncated cell a well-defined prefix of the full pool.

Table 5: Search parameters. Budgets are node expansions.
<table><tr><td>Parameter</td><td>Experiment 1</td><td>Experiment 2</td></tr><tr><td>Budget N</td><td>256</td><td>16, 64, 256, 1024</td></tr><tr><td>Best-first beam width</td><td>2</td><td>2</td></tr><tr><td>Expansions per pop</td><td>4</td><td>4</td></tr><tr><td>Best-first score exponent α</td><td>0.5</td><td>0.5</td></tr><tr><td>Pass@N generation batch</td><td>16</td><td>64</td></tr><tr><td>Maximum proof depth</td><td>5</td><td>5</td></tr><tr><td>Per-strategy wall-clock cap</td><td>90 s</td><td>none</td></tr><tr><td>Per-theorem wall-clock cap</td><td>500 s</td><td>none</td></tr></table>

Scoring and truncation. A theorem is scored as solved only if a kernel-verified proof is returned within the budget; unsolved-within-budget counts as a failure, with no exclusion. The two experiments differ in one respect that we record explicitly. Experiment 1 imposes the wall-clock caps in Table 5, so a theorem may in principle be cut off before its expansion budget is exhausted; Experiment 2 disables them, guaranteeing that every theorem spends its full budget. The caps are the reason the two experiments’ absolute rates are not directly comparable, and they were removed fo Experiment 2 because the effect grows with the budget: a wall-clock cap binds far more often at N = 1024 than at $N = 6 4$ and would otherwise have depressed exactly the high-budget cells whose behavior the experiment is about.

Statistics. Because all configurations are evaluated on the same theorems in the same order, every comparison is paired. For two configurations we report b, the number of theorems solved by the second and not the first, and c, the number solved by the first and not the second, on the subset n that both evaluated. The reported difference is $( b - c ) / n ;$ its 95% interval uses the McNemar variance $\mathrm { V a r } [ ( b - c ) / n ] = ( b + \bar { c } ) / n ^ { 2 }$ ; and the reported p-value is the exact two-sided McNemar (sign) test on the b + c discordant pairs. We use the paired difference rather than the difference of marginal rates throughout, because the two diverge whenever the cells being compared cover different numbers of theorems, and only the paired quantity is consistent with the interval and the p-value beside it. Where a figure shows unpaired intervals on individual rates (Figure 1) this is stated in the caption; those intervals are wider than the paired ones and overlapping bars do not indicate a non-significant difference.

## G.4 EXPERIMENT 1: FIXED-BUDGET COMPARISON

Configurations. Six deployed strategies at $N = 2 5 6 \colon$ Pass@N and best-first search from Section 3, and DFS, RDFS, VDFS and MCTS from Section D. Each is paired with the CAT weight derived for it (same for BFS/DFS), and we additionally deploy the single shared UA adapter under every strategy.

Control. The control for this experiment is the epoch-matched model: warm-up plus one additional epoch of standard SFT. Control and aligned models therefore receive identical total optimiza tion, and the comparison isolates the loss weighting.

## G.5 EXPERIMENT 2: BUDGET SWEEP

Budget-matched adapters. We train a separate adapter per evaluation budget with the weight’s budget parameter set to the deployment value, and evaluate each at its own budget only.

Control. The control for this experiment is the shared warm-up checkpoint, with an additional SFT epoch.

Pools. The $N \in \{ 1 6 , 6 4 , 2 5 6 , 1 0 2 4 \}$ are evaluated on the full 458-theorem pool. However, we only allowed 80 GPU hours so not all evaluations were fully completed.

Reproduction. Given the trained adapters, each cell is one invocation of the evaluation harness with the strategy, budget, and adapter path. Cells write one JSON log per strategy keyed by theorem name, and all tables and figures in this paper are generated directly from those logs.

## H HIGH-BUDGET EVALUATION

To complement Section 5.2, we also evaluated budget-matched CAT adapters at $N = 1 0 2 4$ under Pass@N and BFS, using the same training and evaluation protocol. The evaluation-time limit restricted the paired comparisons to 57 theorems for Pass@N and 56 for BFS. Table 6 reports these results, and Figure 9 shows the complete budget sweep.

Table 6: Compute-limited evaluation at $N = 1 0 2 4$ node expansions. n is the paired subset size, b counts CAT-only successes, and c counts control-only successes. Gains and 95% intervals are in percentage points. Intervals and exact two-sided McNemar tests follow Table 2.
<table><tr><td>Strategy</td><td>Budget N</td><td>n</td><td> $b / c$ </td><td>Gain (pp), 95% interval</td><td>p</td></tr><tr><td>Pass@N</td><td>1024</td><td>57</td><td>6/1</td><td> $+ 8 . 8 \ [ - 0 . 3 , 1 7 . 9 ]$ </td><td>0.125</td></tr><tr><td>BFS</td><td>1024</td><td>56</td><td>5/2</td><td> $+ 5 . 4 \ [ - 3 . 9 , 1 4 . 6 ]$ </td><td>0.453</td></tr></table>

![](images/8be4a59134fa47015f406ac01f629885f42ddbb16c1016cfb070e7e874ce05d8.jpg)  
Figure 9: Complete budget sweep, including the compute-limited $N = 1 0 2 4$ evaluation. Error bars show 95% intervals; filled markers denote $p < 0 . 0 5$ and hollow markers denote $p \ge 0 . 0 5 .$ Dashed segments connect to the $N = 1 0 2 4$ estimates, annotated with their paired subset sizes. The completed subsets differ across budgets, so connecting lines are descriptive rather than a common cohort comparison.

Interpretation. Both high-budget point estimates favor CAT, but each comparison is based on only seven discordant theorem pairs, and both reported 95% intervals include zero. These results are therefore inconclusive about the magnitude of the alignment benefit at $N = 1 0 2 4$ and do not establish a trend from $N = 2 5 6 { \mathrm { ~ t o ~ } } N = 1 0 2 4$ . We report them for completeness rather than using them as evidence of increasing alignment gains at large budgets.

## I PRIORS ON OFF-TRACE BEHAVIOUR

In Section C, we characterized the gap between our trace-supported surrogate and the deployed search. However, we haven’t deeply explored if we can use this to meaningfully improve training. We implied that placing priors on off-trace seems like a reasonable way to utilize this observation. We use this section to take a brief step in that direction and see if a simple prior on loss can be beneficial.

## I.1 A TWO-PARAMETER PRIOR

Proposition 3 and Proposition 6 show that off-trace behaviour enters the objective through two scalars:

$\kappa \geq 1$ , the expected number of expansions a miss consumes, counting the off-trace excursion before the search returns to the trace;

$q \in [ 0 , 1 )$ , the probability that a miss launches an excursion that never returns.

Both enter mechanically and in a strategy-independent way. Proposition 3 states that a deployment in which a miss costs κ expansions realises the surrogate at the deflated budget

$$
N _ { \mathrm { e f f } } \ = \ L + { \frac { N - L } { \kappa } } \ \leq \ N ,\tag{47}
$$

so κ is applied by evaluating the strategy’s own objective at, $N _ { \mathrm { e f f } }$ in place of N. Absorption enters the trace-supported recurrence Equation (5) by splitting the miss branch: with probability $( 1 - p _ { t } ) q$ the search is absorbed and contributes nothing, and with probability $( 1 - p _ { t } ) ( 1 - q )$ it continues as the uncorrected recurrence prescribes. Every occurrence of $( 1 - p _ { t } )$ on a continuation branch is therefore replaced by $( 1 - p _ { t } ) ( 1 - q )$ , with the missing mass leaking to a terminal failure state.

Note that a better prior can make assumptions on κ and q as functions of p or prior probabilities, but we are merely going to test the simple case.

## I.2 APPLYING THE PRIOR TO EACH STRATEGY

The prior is applied to the objective the surrogate implies for each strategy.

Best-first search. The trace-supported objective for BFS is the retry recurrence Equation (13). The prior is applied directly to that recurrence,

$$
\widehat { J } ^ { \kappa , q } ( b , t ) = p _ { t } \widehat { J } ^ { \kappa , q } ( b - 1 , t + 1 ) + ( 1 - p _ { t } ) ( 1 - q ) \widehat { J } ^ { \kappa , q } ( b - 1 , t ) ,\tag{48}
$$

$$
\mathrm { w i t h ~ } \widehat { J } ^ { \kappa , q } ( b , L ) = 1 , \widehat { J } ^ { \kappa , q } ( 0 , t ) = 0 , \mathrm { r u n ~ t o ~ } b = N _ { \mathrm { e f f } } = L + ( N - L ) / \kappa .
$$

RDFS and VDFS. The same substitution applies to the dynamic programs of Section D: the budget axis runs to $N _ { \mathrm { e f f } }$ and the backtrack branch carries the factor $( 1 - q )$ , leaving the uniform and value-weighted backtrack kernels otherwise unchanged.

MCTS. Similarly, we just readjust the dynamic program in essentially the same fashion as the other search strategies.

## I.3 SETUP

The prior is set by hand. The primary setting is $\kappa = 5 , q = 0 . 0 2 ;$ under BFS we additionally train $\kappa = 3 , q = 0$ to probe sensitivity. Everything else matches Section 5.1: the same warm-up checkpoint, one alignment epoch over the same examples in the same order under the same optimiser settings, a budget of $N = 2 5 6$ node expansions, and the full 458-theorem pool. The only difference between a prior-corrected adapter and its uncorrected counterpart is the scalar weight applied to each tactic’s cross-entropy gradient. Each adapter is deployed under the strategy whose objective it corrects.

Table 7: The off-trace prior applied to each strategy’s own CAT objective (proofs found, %, N=256 node expansions, 458 theorems). $\ " { \mathrm { C A T } } \ '$ is the uncorrected weight of Sections 3 and D, i.e. the $( \kappa , q ) = \mathsf { ( 1 , 0 ) }$ member of the family.
<table><tr><td>Strategy</td><td>Prior</td><td>SFT</td><td>CAT</td><td>+prior</td></tr><tr><td>BFS</td><td> $\kappa { = } 5 , \ q { = } 0 . 0 2$   $\scriptstyle \kappa = 3 , q = 0$ </td><td>23.8 23.8</td><td>25.1 25.1</td><td>25.8 25.1</td></tr><tr><td>RDFS</td><td> $\kappa { = } 5 , \ q { = } 0 . 0 2$ </td><td>20.7</td><td>23.6</td><td>22.7</td></tr><tr><td>VDFS</td><td> $\kappa { = } 5 , \ q { = } 0 . 0 2$ </td><td>21.2</td><td>23.8</td><td>24.2</td></tr><tr><td>MCTS</td><td> $\kappa { = } 5 , \ q { = } 0 . 0 2$ </td><td>20.3</td><td>22.5</td><td>22.9</td></tr></table>

## I.4 RESULTS

Table 7 reports the strategies for which the prior was applied. Generally, the priors edged out the basic CAT weights but not by a significant amount in any case. The conclusion we take from this is that perhaps with optimal tuning of the prior, we can get increased gains, but the value is marginal.

## J SEARCH-AGNOSTIC UNIFORM-ALLOCATION OBJECTIVE: ADDITIONAL DETAILS

This appendix supplements the search-agnostic UA objective introduced in Section 2.5. UA accounts for the compute budget through a uniform local allocation without specifying the deployed search rule. We relate its weighting function to the shared-budget retry objective, providing an additional interpretation of how the two objectives account for compute.

From shared-budget to local criticality. Consider the retry objective in Equation (14), with independent $T _ { t } \sim \mathrm { G e o m } ( p _ { t } ) , p _ { t } \in ( 0 , 1 )$ , and integer $N \geq L$ . Write

$$
S _ { L } = \sum _ { s = 1 } ^ { L } T _ { s } , \qquad R _ { t } = \sum _ { s \ne t } T _ { s } , \qquad B _ { t } = N - R _ { t } .
$$

Here, $B _ { t }$ is the budget left for step t after accounting for the waiting times of all other steps. For integer $k \geq 1$ , define the success probability and gradient weight of a single-tactic retry problem by

$$
F ( p , k ) = 1 - ( 1 - p ) ^ { k } ,
$$

$$
h ( p , k ) = { \frac { \partial \log F ( p , k ) } { \partial \log p } } = { \frac { k p ( 1 - p ) ^ { k - 1 } } { 1 - ( 1 - p ) ^ { k } } } .\tag{49}
$$

The shared-budget retry weight then has the exact representation

$$
\widehat { w } _ { t } ^ { \mathrm { r e t r y } } ( N ) = \mathbb { E } [ h ( p _ { t } , B _ { t } ) | S _ { L } \leq N ] .\tag{50}
$$

To see this, extend $F ( p , k ) = 0$ for $k \leq 0$ and write $\widehat { \mathcal { T } } ^ { \mathrm { r e t r y } } ( N ) = \mathbb { E } [ F ( p _ { t } , B _ { t } ) ]$ . The distribution of $B _ { t }$ does not depend on $p _ { t } ,$ , so differentiating and dividing by the success probability weights each residual budget by $\mathbb { P } ( B _ { t } = k ) F ( p _ { t } , k ) / \hat { \mathcal { I } } ^ { \mathrm { r e t r y } } ( N ) = \mathbb { P } ( B _ { t } = k \mid S _ { L } \leq N )$ , giving Equation (50). Only $k \geq 1$ occurs under this conditional distribution.

Thus, the shared-budget weight combines two ingredients: the local criticality $h ( p _ { t } , k )$ and a distribution over the budget available to that step. The other tactic probabilities enter through this budget distribution.

Uniform allocation and objective. The UA objective instead assigns each demonstrated step a uniform local allocation,

$$
\bar { K } = \frac { N } { L } .\tag{51}
$$

This is a modeling choice, not an estimate of $\mathbb { E } [ B _ { t } \ | \ S _ { L } \leq N ]$ or an average over deployed search strategies. It yields the factorized objective

$$
\widehat { \mathcal { T } } ^ { \mathrm { U A } } ( \theta ) = \prod _ { t = 1 } ^ { L } \left[ 1 - ( 1 - p _ { t } ) ^ { \bar { K } } \right] .\tag{52}
$$

For integer $\bar { K } .$ , this is the probability that every demonstrated step succeeds within its own quota of $\bar { K }$ independent attempts, without transferring unused quota between steps. Noninteger $N \bar { / } L$ uses the continuous extension of the same formula. Applying the gradient identity in Equation (7) gives

$$
\widehat { w } _ { t } ^ { \mathrm { U A } } = \frac { \bar { K } p _ { t } ( 1 - p _ { t } ) ^ { \bar { K } - 1 } } { 1 - ( 1 - p _ { t } ) ^ { \bar { K } } } = h ( p _ { t } , \bar { K } ) .\tag{53}
$$

Comparing Equations (50) and (53), UA evaluates the local criticality at a fixed allocation, whereas the shared-budget retry objective averages it over the conditional distribution of the residual budget. The UA weight is the exact gradient weight of its own objective in Equation (52). This connection does not require UA to approximate the dynamics of a particular deployed search algorithm.

Properties of the uniform-allocation weights. UA is search-agnostic, but not compute-agnostic. $\mathbf { A } \mathbf { t } \ \bar { K } = 1$ , Equation (52) reduces to the demonstrated trace likelihood and all weights equal one, recovering CE. For fixed $\bar { K } \geq 1$ , the weight approaches one as $p _ { t } \to 0 ;$ ; for fixed $p _ { t } \in ( 0 , 1 )$ , it approaches zero as $\bar { K }  \infty$ . Thus, tactics that are unlikely to be found within the local budget retain high weight, whereas tactics that can already be recovered through repeated sampling receive less emphasis. Unlike the shared-budget retry weight, the UA weight does not depend on the other steps’ consumption of a shared budget. At fixed N and $L ,$ each weight depends only on $p _ { t }$

Interpretation. The shared-budget and UA objectives retain the same local retry criticality but differ in their treatment of compute allocation. As discussed in Section 4, demonstrations alone do not generally determine deployed gradient weights because they omit alternative proofs and off-trace recovery behavior. The empirical comparison in Table 1 evaluates the usefulness of search-aware CAT and search-agnostic UA for training; it does not establish which objective more accurately approximates the deployed gradient weights.

## K LLM DISCLOSURE

LLMs were used in 3 primary ways. The first is essentially all experiments were set up and instantiated with LLM coding. We had a separate LLM check for correctness of the code. The second is when the paper was drafted, we repeatedly asked LLMs to critique and we incorporated its feedback if we deemed it to be appropriate. The third was finding related papers. We read all of these before ever citing them to confirm the relation to our work.