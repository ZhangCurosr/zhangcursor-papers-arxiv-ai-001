# Graph-Guided Selective Unlearning for Language Models: Controlling Support Routes Beyond Forget Seeds

Waqas Khan, Tabinda Sarwar, Jingyue Cong, Xun Yi, and Estrid He

## Abstract

Enterprises fine-tune language models on proprietary data that may later require removal due to privacy, contractual, or compliance obligations. Selective unlearning removes requested knowledge while preserving model utility, offering a practical alternative to full retraining, but existing methods treat the explicitly identified forget examples as the complete deletion scope. This is insufficient when target knowledge remains recoverable through paraphrases, aliases, or neighboring training examples. We propose GRAPHSU, a graph-guided controller that expands the deletion scope beyond forget seeds by constructing a weighted support-route graph, propagating deletion pressure through it, and applying graded forgetting strengths to high-risk neighbors. On the Task of Fictitious Unlearning (TOFU), a synthetic author-profile question–answering benchmark, and PISTOL, a structural-unlearning benchmark built around interconnected factual samples, with GPT-2 Medium and Llama-3.2-3B-Instruct, GRAPHSU achieves the lowest utility-feasible soft leakage across all deletion settings, reducing leakage by up to 49.5 percentage points over a matched seed-only baseline, demonstrating that effective enterprise unlearning requires controlling support routes, not just forget seeds. Our code is available at https://anonymous.4open. science/r/graphSU-35B4.

## 1 Introduction

Large language models (LLMs) can retain private, copyrighted, stale, or user-rescinded information after training. Exact unlearning would retrain the model on the corpus with the deletion set removed, but this is typically infeasible for modern LMs. Approximate LLM unlearning therefore aims to update a deployed model so that target knowledge is no longer recoverable, while its behaviour on retained data remains close to the original or retain-

only model (Maini et al., 2024; Shi et al., 2024;   
Tian et al., 2024; Li et al., 2024).

As enterprises increasingly customize foundation models using proprietary organizational data, the ability to selectively remove previously learned information becomes critical. Data may later require removal due to privacy regulations, customer offboarding, intellectual-property concerns, policy revisions, or the discovery of sensitive content in training corpora. In such settings, machine unlearning provides a scalable mechanism for updating deployed models without incurring the computational and operational costs of full retraining.

Most current algorithms focus on the local objective: gradient ascent, KL-regularised ascent, preference optimisation, negative preference optimisation, gradient rectification, energy-based unlearning, or representation control (Jang et al., 2023; Zhang et al., 2024; Mekala et al., 2025; Wang et al., 2025b; Chaiyapattanaporn et al., 2026). This line of work is valuable, but it leaves a complementary question under-addressed: where should unlearning be applied? In selective unlearning, targeting only the canonical forget example is often insufficient, as the forgotten information may still be inferred from aliases, paraphrases, neighboring facts, or other training instances that describe the same entity from different perspectives. For example, forgetting “Acme Corp plans to acquire BetaAI for \$850 million” might be insufficient if another sample states, “Acme’s largest strategic investment this year is valued at approximately \$850 million”. Conversely, applying unlearning too broadly can remove information that should be retained, leading to unnecessary degradation of model utility. This creates a scope-control problem that is distinct from the choice of unlearning objective.

In this paper, we formulate selective unlearning as a leakage-route control problem, where the goal is to prevent forgotten information from being reconstructed through paraphrases, aliases, and semantically related training instances while preserving unrelated knowledge. We propose Graphguided Selective Unlearning (GRAPHSU), a lossagnostic controller that determines the fine-grained deletion scope before applying a local unlearning objective. GRAPHSU constructs a weighted support graph over training instances using semantic similarity, symbolic entity/relation/tail overlap, and answer-side gradient alignment. It then propagates deletion pressure from forget seeds, identifies influential support neighbors, assigns soft forgetting strengths, and incorporates graph-selected neighbors through a staged unlearning curriculum. On GPT-2 Medium, GRAPHSU achieves the lowest utility-feasible soft leakage across all deletion settings on both TOFU and PISTOL. Compared to a Seed-Only variant, it consistently reduces leakage while maintaining retain perplexity (PPL; the exponential of mean token-level negative log-likelihood, lower is better) below 10, demonstrating the importance of expanding beyond the initial forget set.

## 2 Methodology

Figure 1 illustrates proposed method.

## 2.1 Problem Statement

Let $f _ { \theta _ { 0 } }$ denote the original model trained on dataset $D = \{ z _ { i } \} _ { i = 1 } ^ { N }$ where $\boldsymbol z _ { i } = ( q _ { i } , a _ { i } ^ { \mathrm { t e x t } } , \mathcal F _ { i } )$ . Here, $q _ { i }$ is the prompt, $a _ { i } ^ { \mathrm { t e x t } }$ is the answer surface form, and $\mathcal { F } _ { i } = \{ ( h _ { i , k } , r _ { i , k } , t _ { i , k } ) \} _ { k = 1 } ^ { K _ { i } }$ is the set of factual units expressed in the answer. Each tuple $( h _ { i , k } , r _ { i , k } , t _ { i , k } )$ consists of head entity, relation, and tail entity or value. We obtain ${ \mathcal { F } } _ { i }$ by prompting GPT-4.1 (OpenAI, 2025) to extract factual predicate-argument tuples from the prompt–answer pair and normalizing the output into $( h , r , t )$ unit. The extracted units are used for target specification and support-graph construction, not as additional supervision for model training.

An unlearning request R identifies target fact units $\kappa _ { \mathcal { R } }$ to remove and a seed forget set $D _ { f } \subset D$ containing them; the initial retain pool is $D _ { r } =$ $D \backslash D _ { f }$ . The goal of selective unlearning is to obtain an updated model $f _ { \theta _ { u } }$ in which the knowledge specified by $\kappa _ { \mathcal { R } }$ is no longer recoverable, while the model’s utility on unrelated retain data is preserved.

## 2.2 Recovery-Aware Deletion Scope

As discussed in Section 1, the seed forget set $D _ { f }$ may miss related training instances that still support recovery of the target knowledge, leaving leakage routes outside $D _ { f }$ . We denote by $M _ { \mathcal { R } } ( f _ { \theta } )$ a scalar recoverability measure for $\kappa _ { \mathcal { R } }$ under requestspecific probes, capturing the extent to which the model can still reproduce, paraphrase, or semantically entail the target facts.

For any training example $z _ { j }$ , define its ideal support contribution as

$$
\Delta _ { j } ^ { \star } ( \mathcal { R } ) = \mathcal { M } _ { \mathcal { R } } ( f _ { \theta _ { 0 } } ) - \mathcal { M } _ { \mathcal { R } } ( f _ { \theta _ { 0 } } ^ { ( - j ) } ) ,\tag{1}
$$

where $f _ { \theta _ { 0 } } ^ { ( - j ) }$ denotes the model obtained by applying the same supervised fine-tuning procedure after removing $z _ { j }$ from the training set.

We define the ideal support closure of $\mathcal { R }$ as

$$
\begin{array} { r } { \mathcal { C } _ { \epsilon } ^ { \star } ( \mathcal { R } ) = D _ { f } \cup \{ z _ { j } \notin D _ { f } : \Delta _ { j } ^ { \star } ( \mathcal { R } ) \geq \epsilon \} , } \end{array}\tag{2}
$$

where $\epsilon > 0$ is a small operational threshold. This closure is the ideal forgetting scope: it includes both the seed forget examples and the non-seed examples whose removal would materially reduce recovery of $\kappa _ { \mathcal { R } }$ . Constructing ${ \mathcal { C } } _ { \epsilon } ^ { \star } ( { \mathcal { R } } )$ exactly is infeasible, because it would require leave-one-out retraining or exact causal attribution over the training corpus. GRAPHSU therefore approximates this closure using a request-conditioned support graph, introduced next.

## 2.3 Support Graph Construction.

The seed forget set $D _ { f }$ is not generally a closed evidence unit. Even after examples in $D _ { f }$ are suppressed, the target knowledge may remain recoverable through paraphrastic (Maini et al., 2024; Reimers and Gurevych, 2019; Reisizadeh et al., 2025), symbolic (Tian et al., 2024; Wang et al., 2025a; Wei et al., 2025; Meng et al., 2022; Zhong et al., 2023; Cohen et al., 2024), and modelinternal support routes (Koh and Liang, 2017; Pruthi et al., 2020; Park et al., 2023). We therefore construct a weighted support graph $G \ =$ $( V , E , W )$ , where each node corresponds to a training example and each edge weight estimates how strongly one example may support recovery of another. The graph is used as an affinity structure for support expansion, not as a calibrated estimate of causal influence.

Multi-view support affinity. For a candidate edge $( i , j )$ , GRAPHSU fuses five support views three symbolic (entity, relation, tail), one semantic, and one gradient-based:

$$
\begin{array} { r l } & { \tilde { w } _ { i j } = \alpha s _ { i j } ^ { \mathrm { e n t } } + \beta s _ { i j } ^ { \mathrm { r e l } } + \gamma s _ { i j } ^ { \mathrm { t a i l } } + \delta s _ { i j } ^ { \mathrm { s e m } } + \lambda s _ { i j } ^ { \mathrm { g r a d } } , } \\ & { w _ { i j } = \mathrm { m i n } \big ( 1 , \mathrm { m a x } ( 0 , \tilde { w } _ { i j } ) \big ) , } \end{array}\tag{3}
$$

![](images/90477da5136a329551675798d5e8c575eacd81d6769a9d0018c65431803df86d.jpg)  
Figure 1: Overview of GRAPHSU, which constructs a weighted multi-view support graph over QA samples, propagates deletion pressure from forget seeds through a one- and two-hop support closure, and applies graded unlearning objectives to the expanded edit set.

where each component score is normalized to [0, 1] and $( \alpha , \beta , \gamma , \delta , \lambda )$ are non-negative view weights (Table 1). The five views capture complementary recovery routes: the three symbolic overlaps capture structured factual support through shared entities, relations, and tail values; semantic similarity captures paraphrastic support; and gradient alignment captures model-local support under the frozen post-SFT model.

Semantic evidence. Let $e _ { i } = \operatorname { E n c } ( [ q _ { i } ; a _ { i } ] )$ be the ℓ<sub>2</sub>-normalized sentence embedding of the prompt– answer pair. The semantic score is

$$
s _ { i j } ^ { \mathrm { s e m } } = [ \cos ( e _ { i } , e _ { j } ) ] _ { + } , \qquad [ x ] _ { + } = \mathrm { m a x } ( 0 , x ) .\tag{4}
$$

Negative semantic cosine indicates dissimilar rather than supporting content. Because GRAPHSU models positive recovery support with a nonnegative diffusion process, negative semantic affinities are clipped to zero rather than allowed to subtract from supportive edges. Modelling inhibitory relationships would require a signed-graph formulation and is left for future work.

Symbolic evidence. Semantic similarity can miss structured dependencies such as aliases, inverse relations, and reused private values. We extract canonicalized entity, relation, and tail-value sets $\mathcal { E } _ { i }$ $\mathcal { R } _ { i } .$ , and $\tau _ { i }$ from $q _ { i } , a _ { i } .$ , and ${ \mathcal { F } } _ { i }$ respectively. Entity aliasing and relation canonicalization map equivalent mentions and inverse relations to shared identifiers. We then compute entity-, relation-, and taillevel symbolic similarities $s _ { i j }$ between instances $i$ and $j$ using Jaccard similarity over the corresponding sets.

Entity overlap $( s ^ { \mathrm { e n t } } )$ captures subject- and aliaslevel support, relation overlap $( s ^ { \mathrm { r e l } } )$ captures factual-route support, and tail-value overlap $( s ^ { \mathrm { t a i l } } )$ captures reused attributes or identifiers.

Gradient-alignment evidence. The semantic and symbolic views are corpus-level approximations. To capture model-local support, we compute answer-side gradient summaries under the frozen post-SFT model. Let $\mathbf { \mathcal { A } } _ { i }$ denote the answer-token positions of $z _ { i }$ . We define

$$
g _ { i } = \frac { 1 } { \left| \mathcal { A } _ { i } \right| } \sum _ { t \in \mathcal { A } _ { i } } \nabla _ { H _ { i t } ^ { ( L ) } } \mathcal { L } _ { \mathrm { C E } } ( z _ { i } ; \theta _ { 0 } ) ,\tag{5}
$$

where $H _ { i t } ^ { ( L ) }$ is the final-layer answer representation after LayerNorm and before the LM head. The gradient-alignment score is

$$
\begin{array} { r } { s _ { i j } ^ { \mathrm { g r a d } } = \left[ \cos ( g _ { i } , g _ { j } ) \right] _ { + } . } \end{array}\tag{6}
$$

Positive gradient cosine indicates compatible, mutually reinforcing local update directions, whereas negative cosine indicates conflicting directions rather than recovery support. Since GRAPHSU propagates non-negative support mass, negative gradient affinities are clipped to zero; representing inhibitory relations would instead require signedgraph propagation.

The graph captures recovery-aware support structure around the request. GRAPHSU propagates deletion pressure from $D _ { f }$ through G, assigning graded forgetting strengths to high-risk neighbors while leaving unrelated examples intact.

## 2.4 Graph-Guided Deletion Scope Approximation

Given an unlearning request $\mathcal { R }$ with the seed forget set $D _ { f }$ . We first estimate which non-seed examples are most likely to belong to the recovery-aware deletion scope of the request.

Let A denote the non-negative adjacency matrix of the sparsified support graph. We use the pairwise affinity $w _ { i j }$ from Eq. 3 directly, so

$$
A _ { i j } = \left\{ \begin{array} { l l } { w _ { i j } , } & { ( i , j ) \in E \mathrm { ~ a f t e r ~ s p a r s i f i c a t i o n } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{7}
$$

Because the component similarities in $\operatorname { E q . 3 }$ are pairwise symmetric, $w _ { i j } = w _ { j i }$ before sparsification; the retained graph is treated as undirected for diffusion. Let $d _ { i } = \textstyle \sum _ { j } A _ { i j }$ be the weighted degree of node i. We define the row-normalized transition matrix

$$
P _ { i j } = \left\{ \begin{array} { l l } { A _ { i j } / d _ { i } , } & { d _ { i } > 0 , } \\ { \mathbb { I } [ i = j ] , } & { d _ { i } = 0 , } \end{array} \right.\tag{8}
$$

where isolated nodes are assigned self-loops. The transition matrix specifies how support evidence moves across the graph: a node passes more evidence to neighbors connected by stronger semantic, symbolic, or gradient-aligned support edges. We initialize a request-specific seed distribution

$$
s _ { i } = \left\{ \begin{array} { l l } { 1 / | D _ { f } | , } & { z _ { i } \in D _ { f } , } \\ { 0 , } & { z _ { i } \notin D _ { f } . } \end{array} \right.\tag{9}
$$

The support score vector $r$ is then computed by personalized graph diffusion:

$$
\boldsymbol { r } ^ { ( k + 1 ) } = ( 1 - \rho ) \boldsymbol { P } ^ { \top } \boldsymbol { r } ^ { ( k ) } + \rho \boldsymbol { s } ,\tag{10}
$$

where $\rho \in \left( 0 , 1 \right)$ is the restart probability. The restart term anchors the diffusion to the deletion request, while the transition term propagates support evidence through the graph. At convergence, the solution satisfies

$$
\boldsymbol { r } = \rho \left( \boldsymbol { I } - ( 1 - \rho ) \boldsymbol { P } ^ { \intercal } \right) ^ { - 1 } \boldsymbol { s } ,\tag{11}
$$

or equivalently,

$$
r = \rho \sum _ { \ell = 0 } ^ { \infty } ( 1 - \rho ) ^ { \ell } ( P ^ { \top } ) ^ { \ell } s .\tag{12}
$$

This expansion shows that $r _ { i }$ aggregates support evidence over all paths from the seed forget examples to $z _ { i } ,$ with paths of length ℓ discounted by $( 1 - \rho ) ^ { \ell } .$ . Hence, $r _ { i }$ is high when $z _ { i }$ is connected to $D _ { f }$ through many strong short paths, or through multiple consistent longer paths. The score is not interpreted as a probability of membership in the deletion scope; it is a ranking signal for requestspecific support evidence.

The estimated support closure is obtained by selecting high-scoring non-seed examples together with the original seeds. For a threshold $\eta$ or an expansion budget $B ,$ we use

$$
\widehat { \mathcal { C } } _ { \eta } ( \mathcal { R } ) = D _ { f } \cup \{ z _ { i } \in D _ { r } : r _ { i } \geq \eta \} ,\tag{13}
$$

or equivalently the top-B non-seed examples ranked by $r _ { i }$ when a fixed expansion budget is required. This construction approximates the ideal support closure ${ \mathcal { C } } _ { \epsilon } ^ { \star } ( { \mathcal { R } } )$ defined in Section 2.2: examples with larger diffusion scores are treated as more likely to materially support recovery of the target knowledge.

## 2.5 Selective Unlearning

Let $D _ { s u p } ( \mathcal { R } )$ be the selected non-seed support set:

$$
D _ { \mathrm { s u p } } ( \mathcal { R } ) = \widehat { \mathcal { C } } _ { \eta } ( \mathcal { R } ) \setminus D _ { f } .\tag{14}
$$

Each example receives a forgetting weight

$$
\omega _ { i } = \left\{ \begin{array} { l l } { 1 , } & { z _ { i } \in D _ { f } , } \\ { \eta _ { \mathcal { R } } \left( \frac { r _ { i } } { \operatorname* { m a x } _ { z _ { j } \in D _ { \mathrm { s u p } } } r _ { j } } \right) ^ { \gamma _ { \omega } } , } & { z _ { i } \in D _ { \mathrm { s u p } } ( \mathcal { R } ) , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{15}
$$

Thus, seed examples receive full forgetting pressure, while non-seed support examples receive bounded pressure according to graph relevance. Let $m _ { i }$ be a binary answer-token mask, with $m _ { i , t } = 1$ for target tokens to be forgotten and $m _ { i , t } = 0$ for answer tokens that should be retained. The complement $1 - m _ { i }$ therefore selects the safe answer tokens. The unlearning objective is

$$
\begin{array} { l } { { \displaystyle { \mathcal { L } } _ { \mathrm { G r a p h S U } } = \sum _ { i = 1 } ^ { N } \Big [ \omega _ { i } { \mathcal { L } } _ { \mathrm { f o r g e t } } ( z _ { i } , m _ { i } ) } } \\ { { ~ + ~ \lambda _ { r } { \mathcal { L } } _ { \mathrm { r e t a i n } } ( z _ { i } , 1 - m _ { i } ) \Big ] } } \\ { { ~ + ~ \lambda _ { \mathrm { K L } } { \mathcal { L } } _ { \mathrm { K L } } . } } \end{array}\tag{16}
$$

The forgetting term is applied only to the target positions selected by $m _ { i }$ and combines three components:

$$
\mathcal { L } _ { \mathrm { f o r g e t } } = \lambda _ { \mathrm { e n t } } \mathcal { L } _ { \mathrm { e n t } } + \lambda _ { \mathrm { U L } } \mathcal { L } _ { \mathrm { U L } } + \lambda _ { \mathrm { r e p } } \mathcal { L } _ { \mathrm { r e p } } .\tag{17}
$$

Here, ${ \mathcal { L } } _ { \mathrm { e n t } }$ is the entropy-maximization term that flattens the predictive distribution at target tokens, ${ \mathcal { L } } _ { \mathrm { U L } }$ is the unlikelihood term that suppresses probability assigned to the target tokens, and $\mathcal { L } _ { \mathrm { r e p } }$ is the representation-repulsion term used to move target-token representations away from their frozen post-SFT representations. The coefficients $\lambda _ { \mathrm { e n t } }$ $\lambda _ { \mathrm { U L } } .$ , and $\lambda _ { \mathrm { { r e p } } }$ control the relative strengths of these terms. The retain loss $\mathcal { L } _ { \mathrm { { r e t a i n } } }$ is cross-entropy on the complementary safe-token mask $1 - m _ { i }$ , while ${ \mathcal { L } } _ { \mathrm { K L } }$ anchors non-target predictive behaviour to the frozen post-SFT model; $\lambda _ { r }$ and $\lambda _ { \mathrm { K I } }$ are their corresponding weights. Representation repulsion is enabled for complete and entity/instance deletion and disabled for partial deletion, where target and non-target facts may share the same answer context. In this way, GRAPHSU separates the scope controller from the local unlearning loss: the graph determines which examples and tokens are edited, while the loss defines how the selected targets are suppressed.

## 3 Experiments

## 3.1 Experimental Settings

Datasets and Models. We evaluate GRAPHSU on two unlearning benchmarks: TOFU (Maini et al., 2024), a synthetic author-profile QA benchmark with auditable forget/retain splits, and PISTOL (Qiu et al., 2024), a structuralunlearning benchmark built around interconnected factual samples. We consider complete, entity/instance, and partial deletion settings. We use GPT-2 Medium for the main controlled comparison and LLAMA-3.2-3B-INSTRUCT (meta-llama/Llama-3.2-3B-Instruct) to assess scalability to larger backbones. For each model, all methods are initialized from the same supervised fine-tuned checkpoint and evaluated using identical forget seeds and prompts.

Baselines. We compare GRAPHSU with existing methods including GA (Jang et al., 2023), GA+KL (Maini et al., 2024), DPO (Rafailov et al., 2023), NPO-TR (Zhang et al., 2024), AltPO (Mekala et al., 2025), GRU (Wang et al., 2025b), and SELU (Chaiyapattanaporn et al., 2026). NPO-TR denotes an NPO-style baseline with an explicit retain-term regularizer. We also compare GRAPHSU with Seed-Only, a controlled ablation that shares the same unlearning objective as GRAPHSU but removes graph-guided controller. Key implementation details are summarized in Table 1.

Table 1: Core implementation settings. Full hyperparameter sweeps are deferred to supplementary material.
<table><tr><td>Setting</td><td>Complete</td><td>Entity</td><td>Partial</td></tr><tr><td>Steps</td><td>1200</td><td>1200</td><td>720</td></tr><tr><td>Graph activatior  $\textbf { 1 } s _ { g }$ </td><td>120</td><td>120</td><td>220</td></tr><tr><td>Neighbour cap  $\boldsymbol { K _ { n } } ^ { \flat }$ </td><td>72</td><td>72</td><td>40</td></tr><tr><td>Neighbour  $\phi _ { m a x } ^ { ( n ) }$ </td><td>0.35</td><td>0.32</td><td>0.15</td></tr><tr><td>Entropy weight λent</td><td>1.35</td><td>1.40</td><td>1.15</td></tr><tr><td>Unlikelihood weight λUL</td><td>0.60</td><td>0.60</td><td>0.25</td></tr><tr><td>Retain CE weight λ</td><td>0.22</td><td>0.20</td><td>0.24</td></tr><tr><td>KL weight  $\lambda _ { \mathrm { K L } }$ </td><td>0.01</td><td>0.01</td><td>0.012</td></tr><tr><td>Repulsion  $\lambda _ { \mathrm { { r e p } } } \mathcal { L } _ { \mathrm { { r e p } } }$ </td><td>on</td><td>on</td><td>off</td></tr><tr><td>Route prompts per seed</td><td>4</td><td>5+aliases</td><td>6</td></tr><tr><td>Edge weights  $( \alpha , \beta , \gamma , \delta , \lambda )$ </td><td colspan="3"> $( 0 . 7 , 0 . 5 , 0 . 5 , 1 . 0 , 0 . 7 )$ </td></tr></table>

## 3.2 Metrics and Route Probes

The primary forget metric is soft leakage. For each forget seed $z _ { i } \in D _ { f }$ , let $\{ p _ { i , k } \} _ { k = 1 } ^ { K _ { i } }$ denote its validated recovery-route prompts, where k indexes an applicable route prompt and $K _ { i }$ is the number of valid prompts for that seed after route generation and filtering (Appendix E). Let $y _ { i , k } ^ { \prime }$ be the model continuation generated from $p _ { i , k } .$ , and let target denote the sensitive answer string or grounded target value whose recoverability is being tested for seed i. We define $s _ { i , k } =$ sim $( y _ { i , k } ^ { \prime } ;$ , target<sub>i</sub>) as the sentence-embedding cosine similarity between the generated continuation and that target. Soft leakage is

$$
\operatorname { S o F T L E A K } = \frac { 1 } { | D _ { f } | } \sum _ { z _ { i } \in D _ { f } } \frac { 1 } { K _ { i } } \sum _ { k = 1 } ^ { K _ { i } } \mathbb { 1 } [ s _ { i , k } \geq \tau _ { \mathrm { s o f t } } ] ,\tag{18}
$$

where $\tau _ { \mathrm { s o f t } } = 0 . 8 5$ is the similarity threshold used to count a route as leaking the target. For both datasets, we evaluate three deletion settings: complete deletion, where the target is the full answer; partial deletion, where the target is a set of sensitive spans; and entity deletion, where the target is the canonical entity profile together with its aliases. Retention is measured using retain perplexity (PPL), retain F1, ROUGE-L F1, and separate retain-neighbor, retain-far, and partial-retain-only panels. PPL is the exponential of the mean tokenlevel negative log-likelihood, so lower values indicate better retained language-modelling performance. Appendix I reports the additional utility panels jointly with soft leakage. Route families include direct, paraphrase, indirect, cloze, relatedfact, list-sum., relation-inversion, and entity-alias prompts.

## 3.3 Main Results

Utility-constrained unlearning. Table 2 reports the main GPT-2 Medium results on TOFU and PISTOL; results with LLAMA-3.2-3B-INSTRUCT are provided in Appendix A. Across both datasets and all deletion settings, GRAPHSU achieves the lowest soft leakage among methods that keep retain PPL below 10. This distinction is important because effective unlearning requires balancing forgetting and retention. DPO, AltPO, and SELU preserve low retain PPL but leave high leakage, while GA, GA+KL, and GRU can reduce leakage only with substantial retention loss. For example, on TOFU complete deletion, GRU obtains lower leakage than GRAPHSU (38.13% vs. 46.83%) but raises retain PPL to 79.11, making it unsuitable under the utility-feasible criterion.

Impact of graph-guided support expansion. Since Seed-Only uses the same unlearning objective but omits graph-neighbor expansion, the comparison isolates the effect of support-route control in GRAPHSU. On TOFU, GRAPHSU reduces aggregate soft leakage by 46.42 points for complete deletion, 41.65 points for entity/instance deletion, and 18.33 points for partial deletion. On PISTOL, the corresponding reductions are 49.50, 21.66, and 12.22 points. Figure 2 further shows that the improvement is consistent across recovery-route families: GRAPHSU improves over Seed-Only in all 21 task–route cells, with average route-level reductions of 45.9, 41.7, and 22.6 points for complete, entity/instance, and partial deletion, respectively. These results show that effective unlearning requires deleting support routes, not just seed examples. Appendix I complements PPL with retain F1, ROUGE-L F1, far-retain F1, neighbour-retain F1, and safe-span F1 for partial deletion. On TOFU, GRAPHSU raises retain F1 from 39.07 to 42.41 for complete deletion, from 37.16 to 59.11 for entity/instance deletion, and from 3.20 to 55.14 for partial deletion; under partial deletion, safe-span F1 rises from 1.26 to 61.31. These answer-level results show that the controlled Seed-Only gains reflect an improved forgetting–utility balance rather than leakage reduction in isolation.

Challenges of partial deletion. Complete and entity/instance deletion benefit most from graph expansion because target information can be supported through full answers, aliases, and related facts. Partial deletion is more constrained: the model must suppress sensitive spans while preserving the remaining answer content. Although GRAPHSU improves the partial-deletion frontier, achieving 81.67% leakage with retain PPL 1.91 on TOFU, substantial residual leakage remains. These results support graph-guided support expansion and route-level control, while stronger claims about certified erasure require further ablation.

![](images/849f8ea1764901a0194f7b02d7d1387e38397adf213ca66916779c5a053110ea.jpg)  
Figure 2: Route-level soft-leakage reduction of GRAPHSU relative to Seed-Only on GPT-2 Medium (TOFU). Values denote percentage-point reductions in soft leakage; higher is better.

## 3.4 Recovery-route robustness

Figure 2 reports the route-level soft-leakage reduction of GRAPHSU relative to Seed-Only across seven recovery-route families and three deletion settings. GRAPHSU improves over Seed-Only in all 21 task–route combinations, reducing leakage by an average of 36.8 percentage points. Gains are largest for complete deletion, with an average reduction of 45.9 points and a maximum of 55.0 points under relation-inversion prompts. Entity deletion shows a similar pattern, averaging 41.7 points with improvements of at least 35.2 points across all routes. Partial deletion yields smaller but consistent gains (22.6 points on average), reflecting the need to preserve non-sensitive answer content while suppressing sensitive spans. Since both methods use the same local objective and route prompts, these results provide controlled evidence that graphguided support expansion improves robustness to diverse recovery routes.

## 4 Related Work

Early LLM unlearning studies adapted gradient ascent or retain-regularised objectives to suppress forget examples (Jang et al., 2023; Maini et al., 2024). More recent methods address known failure modes of direct ascent: NPO slows catastrophic collapse through a preference-inspired loss (Zhang et al., 2024); AltPO combines negative and positive preference signals to avoid nonsensical forget responses (Mekala et al., 2025); GRU rectifies gradients to mitigate the unlearning– retention trade-off (Wang et al., 2025b); and SELU uses an energy-based LoRA formulation with straight-through estimators (Chaiyapattanaporn et al., 2026). These algorithms mainly improve the update rule. GRAPHSU is complementary: it determines the support routes, masks, and strengths to which an update rule is applied.

<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Method</td><td colspan="2">Complete</td><td colspan="2">Entity/Instance</td><td colspan="2">Partial</td></tr><tr><td>Leak↓</td><td>PPL↓</td><td>Leak↓</td><td>PPL↓</td><td>Leak↓</td><td>PPL↓</td></tr><tr><td rowspan="8">TOFU</td><td>GA</td><td>97.13</td><td>24.91</td><td>97.75</td><td>51.13</td><td>99.75</td><td>61.24</td></tr><tr><td>GA+KL</td><td>97.13</td><td>24.91</td><td>97.75</td><td>51.13</td><td>88.25</td><td>42.44</td></tr><tr><td>DPO</td><td>99.88</td><td>1.24</td><td>100.00</td><td>1.30</td><td>100.00</td><td>2.94</td></tr><tr><td>NPO-TR</td><td>91.50</td><td>4.79</td><td>88.63</td><td>9.12</td><td>95.13</td><td>10.75</td></tr><tr><td>AltPO</td><td>97.50</td><td>1.58</td><td>97.19</td><td>1.30</td><td>100.00</td><td>9.02</td></tr><tr><td>GRU</td><td>38.13†</td><td>79.11</td><td>65.31</td><td>59.09</td><td>86.25</td><td>24.42</td></tr><tr><td>SELU</td><td>92.88</td><td>1.29</td><td>97.88</td><td>1.32</td><td>98.75</td><td>1.27</td></tr><tr><td>Seed-Only</td><td>93.25</td><td>2.61</td><td>96.25</td><td>4.40</td><td>100.00</td><td>7.90</td></tr><tr><td></td><td>GRAPHSU (Proposed)</td><td>46.83</td><td>3.27</td><td>54.60</td><td>4.94</td><td>81.67</td><td>1.91</td></tr><tr><td rowspan="10">PISTOL</td><td>GA</td><td>100.00</td><td>1.02</td><td>100.00</td><td>1.05</td><td>100.00</td><td>1.20</td></tr><tr><td>GA+KL</td><td>100.00</td><td>1.02</td><td>99.39</td><td>1.06</td><td>99.34</td><td>1.20</td></tr><tr><td>DPO</td><td>99.83</td><td>1.02</td><td>100.00</td><td>1.02</td><td>100.00</td><td>1.28</td></tr><tr><td>NPO-TR</td><td>98.92</td><td>1.02</td><td>94.50</td><td>1.10</td><td>99.17</td><td>1.39</td></tr><tr><td>AltPO</td><td>95.33</td><td>1.15</td><td>99.50</td><td>1.03</td><td>63.67</td><td>3.13</td></tr><tr><td>GRU</td><td>99.87</td><td>1.04</td><td>98.70</td><td>1.06</td><td>98.76</td><td>1.37</td></tr><tr><td>SELU</td><td>96.66</td><td>1.02</td><td>100.00</td><td>1.03</td><td>97.78</td><td>1.01</td></tr><tr><td>Seed-Only</td><td>56.33</td><td>1.09</td><td>26.33</td><td>1.10</td><td>41.39</td><td>1.22</td></tr><tr><td>GRAPHSU (Proposed)</td><td>6.83</td><td>1.05</td><td>4.67</td><td>1.10</td><td>29.17</td><td>1.19</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr></table>

Table 2: Main forget–retain comparison on GPT-2 Medium across TOFU and PISTOL. Leak denotes soft leakage (%); PPL denotes retain perplexity; lower is better for both. Bold marks the best value in each column, regardless of method. Underlining marks the lowest-leakage result satisfying the operational retention constraint PPL ≤ 10. † marks the lower-leakage TOFU complete-deletion result that violates this constraint.

## 5 Conclusions

We presented GRAPHSU, a graph-guided controller that treats selective unlearning as a supportroute control problem: instead of editing only the explicitly identified forget seeds, it builds a weighted multi-view support graph, propagates request-conditioned relevance from the seeds, and applies graded forgetting strengths to the highrisk neighbours. GRAPHSU can be paired with existing unlearning objectives without modifying them. Across TOFU and PISTOL with GPT-2 Medium and LLAMA-3.2-3B-INSTRUCT, GRAPHSU achieves the lowest utility-feasible soft leakage in all deletion settings, reducing leakage by up to 49.5 percentage points over a matched Seed-Only ablation while maintaining low retain perplexity. These results suggest that the deletion scope, not only the unlearning loss, is a key determinant of reliable knowledge removal.

## Limitations

Unlike complete or entity deletion, partial deletion requires removing sensitive spans while preserving the surrounding answer content, making the task inherently more constrained. A well-posed partialdeletion instance requires an answer that contains at least two clearly separable facts, one sensitive and targeted for removal, and one safe and retained so that the sensitive span can be suppressed without disturbing the remainder. This is a property of the span-level deletion setting itself rather than of any particular method, and the residual leakage observed in this setting largely reflects this structural difficulty.

More broadly, soft leakage measures behavioural recoverability under a fixed family of recoveryroute probes rather than certified representational erasure, so our results are best interpreted as evidence of substantially reduced reachability of the target knowledge. GRAPHSU additionally performs offline fact extraction and support-graph construction; this cost is incurred once per corpus and amortised across deletion requests (Appendix H).

## Ethical Considerations

Selective unlearning can support privacy, copyright compliance, removal of stale information, and safer model maintenance. It can also create a false sense of compliance if evaluation is weak, or be misused to remove safety-relevant knowledge. We therefore present GRAPHSU as an approximate unlearning technique that should be paired with governance, audit trails, and independent verification.

## References

P. Chaiyapattanaporn, P. Stenetorp, and Y. Chen. 2026. Selu: Energy-based targeted unlearning in llms. OpenReview submission.

Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva. 2024. Evaluating the ripple effects of knowledge editing in language models. Transactions ofthe Associationfor Computational Linguistics, 12:283–298.

Joel Jang, Dongkeun Yoon, Sohee Yang, Sungmin Cha, Moontae Lee, Lajanugen Logeswaran, and Minjoon Seo. 2023. Knowledge unlearning for mitigating privacy risks in language models. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 14389–14408.

Pang Wei Koh and Percy Liang. 2017. Understanding black-box predictions via influence functions. In Proceedings of the 34th International Conference on Machine Learning, volume 70 of Proceedings of Machine Learning Research, pages 1885–1894. PMLR.

Nathaniel Li, Alexander Pan, Anjali Gopal, Summer Yue, Daniel Berrios, Alice Gatti, Justin D. Li, Ann-Kathrin Dombrowski, Shashwat Goel, Long Phan, Gabriel Mukobi, Nathan Helm-Burger, Rassin Lababidi, Lennart Justen, Andrew B. Liu, Michael Chen, Isabelle Barrass, Oliver Zhang, Xiaoyuan Zhu, and 38 others. 2024. The WMDP benchmark: Measuring and reducing malicious use with unlearning. In Proceedings of the 41st International Conference on Machine Learning.

Pratyush Maini, Zhili Feng, Avi Schwarzschild, Zachary C. Lipton, and J. Zico Kolter. 2024. TOFU: A task of fictitious unlearning for large language models. arXiv preprint arXiv:2401.06121.

Abhilash Mekala, Vahid Dorna, Shivani Dubey, Akshat Lalwani, Dawid Koleczek, Mukund Rungta, Sadid Hasan, and Elita Lobo. 2025. Alternate preference optimization for unlearning factual knowledge in large language models. In Proceedings of COL-ING.

Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov. 2022. Locating and editing factual associations in GPT. In Advances in Neural Information Processing Systems, volume 35, pages 17359–17372.

OpenAI. 2025. Introducing gpt-4.1 in the api. https: //openai.com/index/gpt-4-1/. Accessed: 2026- 06-16.

Sung Min Park, Kristian Georgiev, Andrew Ilyas, Guillaume Leclerc, and Aleksander Madry. 2023. TRAK:

Attributing model behavior at scale. In Proceedings of the 40th International Conference on Machine Learning, volume 202 of Proceedings of Machine Learning Research, pages 27074–27113. PMLR.

Garima Pruthi, Frederick Liu, Mukund Sundararajan, and Satyen Kale. 2020. Estimating training data influence by tracing gradient descent. In Advances in Neural Information Processing Systems, volume 33, pages 19920–19930.

Xinchi Qiu, William F. Shen, Yihong Chen, Nicola Cancedda, Pontus Stenetorp, and Nicholas D. Lane. 2024. How data inter-connectivity shapes llms unlearning: A structural unlearning perspective. arXiv preprint arXiv:2406.16810.

Rafael Rafailov, Archit Sharma, Eric Mitchell, Christopher D. Manning, Stefano Ermon, and Chelsea Finn. 2023. Direct preference optimization: Your language model is secretly a reward model. In Advances in Neural Information Processing Systems.

Nils Reimers and Iryna Gurevych. 2019. Sentence-BERT: Sentence embeddings using Siamese BERTnetworks. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing, pages 3982–3992, Hong Kong, China. Association for Computational Linguistics.

Hadi Reisizadeh, Jiajun Ruan, Yiwei Chen, Soumyadeep Pal, Sijia Liu, and Mingyi Hong. 2025. Leak@k: Unlearning does not make LLMs forget under probabilistic decoding. arXiv preprint arXiv:2511.04934.

Weijia Shi, J. Lee, Y. Huang, S. Malladi, J. Zhao, A. Holtzman, D. Liu, L. Zettlemoyer, N. A. Smith, and C. Zhang. 2024. Muse: Machine unlearning sixway evaluation for language models. arXiv preprint arXiv:2407.06460.

Bozhong Tian, Xiaozhuan Liang, Siyuan Cheng, Qingbin Liu, Mengru Wang, Dianbo Sui, Xi Chen, Huajun Chen, and Ningyu Zhang. 2024. To forget or not? towards practical knowledge unlearning for large language models. In Findings of the Associationfor Computational Linguistics: EMNLP 2024, pages 1524–1537, Miami, Florida, USA. Association for Computational Linguistics.

Wenyu Wang, Mengqi Zhang, Xiaotian Ye, Zhaochun Ren, Pengjie Ren, and Zhumin Chen. 2025a. UIPE: Enhancing LLM unlearning by removing knowledge related to forgetting targets. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2025, pages 25212–25227, Suzhou, China. Association for Computational Linguistics.

Y. Wang, Q. Wang, F. Liu, W. Huang, Y. Du, X. Du, and B. Han. 2025b. Gru: Mitigating the trade-off between unlearning and retention for llms. In Proceedings of the International Conference on Machine Learning.

Rongzhe Wei, Peizhi Niu, Hans Hao-Hsun Hsu, Ruihan Wu, Haoteng Yin, Mohsen Ghassemi, Yifan Li, Vamsi K. Potluru, Eli Chien, Kamalika Chaudhuri, Olgica Milenkovic, and Pan Li. 2025. Do LLMs really forget? evaluating unlearning with knowledge correlation and confidence awareness. In Advances in Neural Information Processing Systems. NeurIPS 2025 poster.

Ruiqi Zhang, L. Lin, Y. Bai, and S. Mei. 2024. Negative preference optimization: From catastrophic collapse to effective unlearning. arXiv preprint arXiv:2404.05868.

Zexuan Zhong, Zhengxuan Wu, Christopher Manning, Christopher Potts, and Danqi Chen. 2023. MQuAKE: Assessing knowledge editing in language models via multi-hop questions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 15686–15702, Singapore. Association for Computational Linguistics.

## A Experiment Results on LLAMA-3.2-3B-INSTRUCT

Larger-backbone evaluation. We evaluate GRAPHSU with LLAMA-3.2-3B-INSTRUCT on TOFU to examine whether graph-guided support expansion transfers beyond GPT-2 Medium. All methods start from the same supervised finetuned checkpoint and use the same deletion manifests, evaluation prompts, decoding settings, and checkpoint-selection criterion. Lower values are better for both soft leakage and retain perplexity. Complete deletion.

<table><tr><td>Method</td><td>Leak↓</td><td>PPL↓</td></tr><tr><td>Seed-Only</td><td>93.67</td><td>1.05</td></tr><tr><td>GA</td><td>99.70</td><td>1.08</td></tr><tr><td>GA+KL</td><td>94.33</td><td>1.09</td></tr><tr><td>DPO</td><td>94.33</td><td>1.30</td></tr><tr><td>NPO-TR</td><td>91.66</td><td>1.06</td></tr><tr><td>AltPO</td><td>90.00</td><td>1.08</td></tr><tr><td>GRU</td><td>93.33</td><td>1.08</td></tr><tr><td>SELU</td><td>93.66</td><td>1.05</td></tr><tr><td>GRAPHSU</td><td>13.33</td><td>1.96</td></tr></table>

## Entity/instance deletion.

<table><tr><td>Method</td><td>Leak↓</td><td>PPL↓</td></tr><tr><td>Seed-Only</td><td>25.11</td><td>1.04</td></tr><tr><td>GA</td><td>36.00</td><td>1.20</td></tr><tr><td>GA+KL</td><td>38.66</td><td>1.05</td></tr><tr><td>DPO</td><td>92.22</td><td>1.70</td></tr><tr><td>NPO-TR</td><td>33.33</td><td>1.09</td></tr><tr><td>AltPO</td><td>34.67</td><td>1.15</td></tr><tr><td>GRU</td><td>86.33</td><td>1.03</td></tr><tr><td>SELU</td><td>35.66</td><td>1.02</td></tr><tr><td>GRAPHSU</td><td>15.53</td><td>1.35</td></tr></table>

## Partial deletion.

<table><tr><td>Method</td><td>Leak↓</td><td>PPL↓</td></tr><tr><td>Seed-Only</td><td>94.33</td><td>1.56</td></tr><tr><td>GA</td><td>94.36</td><td>1.05</td></tr><tr><td>GA+KL</td><td>89.00</td><td>1.16</td></tr><tr><td>DPO</td><td>96.36</td><td>1.03</td></tr><tr><td>NPO-TR</td><td>89.66</td><td>1.07</td></tr><tr><td>AltPO</td><td>88.67</td><td>1.09</td></tr><tr><td>GRU</td><td>93.67</td><td>1.06</td></tr><tr><td>SELU</td><td>94.66</td><td>1.04</td></tr><tr><td>GRAPHSU</td><td>87.01</td><td>1.07</td></tr></table>

The larger-backbone results preserve the central pattern observed in the controlled GPT-2 Medium experiments: expanding the edit scope beyond the explicit forget seeds is most beneficial when the target remains reachable through related examples or entity-centred support routes. Partial deletion remains more difficult because the method must suppress only a sensitive span while preserving the remainder of the answer.

Perplexity consistency check. Retain PPL is computed by teacher-forced scoring on the same frozen retain set for a given comparison. Prompt and padding positions are masked, answer-token negative log-likelihood (NLL) is summed over the evaluation set and divided by the number of scored answer tokens, and the aggregate mean NLL is exponentiated once. Consequently, PPL must be at least one. The DPO partial-deletion value previously shown as 0.03 was mean NLL rather than PPL; the corrected value is exp(0.03) ≈ 1.03. This correction does not change the leakage ranking because DPO soft leakage is 96.36%.

Applicability beyond QA-formatted data and causal language models. GPT-2 Medium and LLAMA-3.2-3B-INSTRUCT are autoregressive causal language models (CLMs), not QA-specific architectures; the QA pairs used here are the benchmarks’ data representation. More generally, a GRAPHSU node can represent a document, passage, or conversation, with deletion masks identifying sensitive spans, while the same semantic, symbolic, and gradient support views define requestconditioned support routes for non-QA corpora.

## B Pareto Trade-Off Plots

The plots below show the forget–retain trade-off for the three evaluated model–dataset settings. Lowerleft is better; the shaded region marks the utilityfeasible region, and the dashed line denotes retain PPL = 10 where shown. To preserve legibility in the two-column EMNLP format, each model– dataset plot is rendered as a separate full-width vector figure.

The PPL ≤ 10 threshold is used only as a fixed operational guardrail to prevent severe retention collapse from being interpreted as successful unlearning; it is not presented as a universal cutoff. We therefore also inspect the unfiltered Pareto frontier, retaining every reported operating point regardless of PPL.

Table 3: Unfiltered Pareto-optimal points on GPT-2 Medium TOFU. Each pair is (soft leakage %, retain PPL); lower is better on both axes.
<table><tr><td>Setting</td><td>Pareto-optimal operating points</td></tr><tr><td>Complete</td><td>DPO (99.88, 1.24); SELU (92.88, 1.29); GRAPHSU (46.83, 3.27); GRU (38.13, 79.11)</td></tr><tr><td>Entity/instance</td><td>AltPO (97.19, 1.30); Seed-Only (96.25, 4.40); GRAPHSU (54.60, 4.94)</td></tr><tr><td>Partial</td><td>SELU (98.75, 1.27); GRAPHSU (81.67, 1.91)</td></tr></table>

GRAPHSU is Pareto-optimal in all three deletion settings. It has the lowest leakage among the reported methods under every PPL cap from 5 through 50. With no retention cap, GRU attains lower complete-deletion leakage (38.13%) but at retain PPL 79.11. NPO-TR is only slightly above the original guardrail for partial deletion (PPL 10.75), yet its soft leakage remains 95.13% compared with 81.67% for GRAPHSU. The guardrail should therefore be read as one practical operating region rather than as a filter that defines the underlying trade-off.

- "head": the entity the fact is about

## C Fact Extraction and GPT-4.1 Prompting

GRAPHSU uses GPT-4.1 only for offline fact grounding. For each QA pair, GPT-4.1 extracts schema-constrained head–relation–tail facts. The outputs are then validated, canonicalised, deduplicated, and type-checked using deterministic postprocessing. Extraction is performed once per corpus and cached; it is not repeated for each deletion request. The recovery-route prompts in Appendix E are generated locally and do not use an external LLM.

Table 4: Fact-extraction settings.
<table><tr><td>Setting</td><td>Value</td></tr><tr><td>Extraction model</td><td>GPT-4.1</td></tr><tr><td>Output format</td><td>JSON object (json_object mode)</td></tr><tr><td>Nominal workload</td><td>Approximately one initial request per QA sample</td></tr><tr><td>Maximum retry attempts</td><td>3</td></tr><tr><td>Request timeout</td><td>60 seconds</td></tr><tr><td>Maximum labelled few-shot examples</td><td>20</td></tr></table>

## System instruction.

You are an information extraction assistant that prepares data for a knowledge graph. Be CONSERVATIVE and ACCURATE: never invent books, awards, dates, or relationships that are not clearly stated in the question or answer. For each question/answer pair, extract ATOMIC factual triples suitable for graph edges. Each triple is an object with:

\- "head\_type": one of [PERSON, ORG, WORK, EVENT, LOCATION, AWARD, GENRE, CHARACTER, TOPIC, OTHER]

\- "relation": a short, descriptive edge label - "tail": the target node, value, or linked entity - "tail\_type": one of [PERSON, ORG, WORK, EVENT, LOCATION, DATE, NUMBER, OCCUPATION, AWARD, GENRE, CHARACTER, TOPIC, NATIONALITY, LANGUAGE, OTHER]

BOOKS/WORKS: create a separate triple per titled work; use PERSON--author\_of-->WORK; quoted strings in a works context are strong WORK candidates; add WORK--genre--> GENRE and WORK--publication\_year-->DATE when stated. LANGUAGE/NATIONALITY: tag tail\_type accordingly.

Use both question and answer; split comma/"and" lists into separate triples; do not invent facts. Return ONLY: {" triples": [...]}

## Per-sample user prompt and worked example.

Study the supplied canonical examples, then extract graph   
triples for:   
Question: Where does Avery Collins work?   
Answer: Avery Collins works at Northbridge University and   
writes in Spanish.   
-> {"triples": [   
{"head":"Avery Collins","head\_type":"PERSON",

```jsonl
"relation":"works_at","tail":"Northbridge University","
tail_type":"ORG"},
{"head":"Avery Collins","head_type":"PERSON",
"relation":"writes_in","tail":"Spanish","tail_type":"
LANGUAGE"}
]}
```

## D Local Fact Extraction and Error Propagation

Hosted fact extraction may be undesirable for proprietary corpora. We therefore rebuilt the symbolic graph with the local Qwen2.5-3B-Instruct model and repeated a matched TOFU complete-deletion experiment. Only extractor-derived facts and symbolic edges changed; GPT-2 Medium, all 184 forget targets, training code, seed 42, unlearning recipe, evaluation manifest, decoding, and detector thresholds were held fixed.

Qwen processed 4,005 QA examples and returned valid triples for 3,942 (98.43%) without external API calls. In a 200-example structured audit, exact-triple F1 was 15.57% for Qwen and 9.88% for GPT-4.1; GPT-4.1 retained higher exact-tail accuracy. Because semantically equivalent facts can be decomposed into different triples, this structured audit is diagnostic rather than a complete semantic gold standard.

The two rebuilt graphs remained close despite extractor differences: edge-weight Spearman correlation was $\rho \ : = \ : 0 . 9 2 5$ , degree correlation was $\rho = 0 . 9 7 9$ , diffusion-score correlation was ρ = 0.996, and the selected support sets had Jaccard similarity 0.895 (68 of 72 selected supports were identical). To test error propagation, we additionally corrupted graph-active heads, tails, and triples. At 40% random-active corruption, 69 of 72 support nodes were retained; under 40% graph-critical corruption, 66 of 72 were retained. Diffusion-score correlations remained at least 0.9974. These perturbations indicate that symbolic noise is attenuated by the semantic and gradient views, weighted fusion, and diffusion rather than being automatically propagated to the full support closure.

At this matched operating point, local extraction gives soft leakage 49.83%, approximately 3.0 percentage points above the re-verified GPT-4.1- based GRAPHSU result (46.83%), while avoiding external transmission of the corpus. Leakage decreases by 36–44 percentage points across every tested route family and hard leakage is 0–3%. The trade-off is not cost-free: lexical retain utility falls, and neighbour-retain PPL reaches 11.54 even though the three-panel aggregate retain PPL is 9.95. We therefore view local extraction as a privacy-preserving deployment option with a measurable forgetting–utility trade-off, not as an accuracy-equivalent replacement for GPT-4.1.

TOFU : Forget Retain Trade Off By Deletion Request  
![](images/c7d13b999436828067667fc02c4b41c0bd0bc6eefb12ac327f4efdb85d50aa0d.jpg)

![](images/4ef64e5f0853b84f87af64f1d8c2a3ee0e011a46e5ab758c2fca269aca103aa6.jpg)

![](images/bbd1278d7af29df7d8ccb1dd3a47249d21025441b2d76ee96deaa4d202f43cd5.jpg)

Figure 3: Forget–retain trade-off for GPT-2 Medium on TOFU. The complete, partial, and entity/instance deletion panels are shown at full width so axis labels, method labels, and the utility-feasible region remain readable.  
![](images/7f21b73c2765acf1859ce93cd47c40014f204aa0090dff735f83015f8d97c9bc.jpg)

PISTOL : Forget Retain Trade Off By Deletion Request  
![](images/bf950ec7f510f6e42aed2f8dd865c410bebaba7f88c6e03e128709a3548aedfa.jpg)

![](images/eb05c04437191a5ddb5d19d30ebea42e897c7af3f966986f3c3c5f8ef6412eaf.jpg)

![](images/0e18fa5351280b797c939708bfec196aac393e0848b96f68b98f9281cf1bc499.jpg)  
Figure 4: Forget–retain trade-off for GPT-2 Medium on PISTOL, shown as a full-width vector panel for improved readability.

![](images/7961a809a0a609f6850ee6c5ece4e69764d0c0bdbcd2330aaf52cc665f24cef1.jpg)

TOFU : Forget Retain Trade Off By Deletion Request  
![](images/02b20869cf5fc774a501f05ae3ad4c9edb989b0ff04409041f796df109f8b818.jpg)

![](images/6f0a09108bd93a669fce0b284341fa8dec9ab92fdb907dac7f2994879b2c2407.jpg)

![](images/5105a4390d49725f64766c15fde4ee270d8abf580f36b6b81e292197baefc1f4.jpg)  
Figure 5: Forget–retain trade-off for LLAMA-3.2-3B-INSTRUCT on TOFU, shown as a full-width vector panel for improved readability.

## E Adversarial Route-Prompt Generation

For each evaluated sample $z _ { i } = ( q _ { i } , a _ { i } ^ { \mathrm { t e x t } } , \mathcal { F } _ { i } )$ , we select a grounded target fact $k _ { i } = ( h _ { i } , r _ { i } , t _ { i } )$ , where $h _ { i }$ is the head entity, $r _ { i }$ is the relation, and $t _ { i }$ is the target value whose recoverability is assessed.

Table 5: Matched step-480 TOFU complete-deletion operating point using local Qwen2.5-3B-Instruct fact extraction. The Original-SFT column is the common pre-unlearning checkpoint.
<table><tr><td>Metric</td><td></td><td>Original-SFT Local-Qwen GRAPHSU</td></tr><tr><td>Soft leakage ↓</td><td>91.25%</td><td>49.83%</td></tr><tr><td>Hard leakage ↓</td><td>37.67%</td><td>1.33%</td></tr><tr><td>Forget-answer F1 ↓</td><td>84.37%</td><td>33.86%</td></tr><tr><td>Retain F1 ↑</td><td>72.55%</td><td>30.27%</td></tr><tr><td>Retain semantic cosine ↑</td><td>90.95%</td><td>79.53%</td></tr><tr><td>Retain PPL ↓</td><td>4.30</td><td>9.95</td></tr></table>

Recovery-route prompts are generated offline using deterministic, relation-conditioned templates; no external LLM is used in this stage.

The generator may additionally use a validated alias set $A ( h _ { i } )$ , an inverse-relation mapping $I ( r _ { i } )$ and related facts associated with the same entity or a graph-connected sample. Candidate prompts are normalised, deduplicated, checked for missing template slots, accidental target disclosure, and relation compatibility. The resulting manifests are frozen before evaluation and reused unchanged for GRAPHSU and all baselines. The task-specific monitoring panels use the prompt counts in Table 1; not every route is applicable to every target, and validated aliases may add entity-specific variants.

## E.1 Detailed Worked Example

Consider the illustrative QA sample:

Question: Where does Avery Collins work?

Answer: Avery Collins works at Northbridge University.

The fact-extraction stage produces the grounded target fact with head Avery Collins, relation works\_at, and tail Northbridge University. For this example, the generator is also provided with the validated aliases

$$
A ( h _ { i } ) = \{ \mathrm { A . C o l l i n s , A v e r y C . } \} ,\tag{19}
$$

and, when available, a related fact such as

$$
( \mathrm { A v e r y ~ C o l l i n s } , r e s e a r c h e s , \mathrm { m a c h i n e ~ e t h i c s } ) .\tag{20}
$$

All prompts below assess recovery of the same target value, Northbridge University; only the recovery route changes.

After template expansion, prompts pass through the following deterministic validation procedure:

1. fill route-specific slots using the grounded target fact, validated aliases, and available related facts;

2. remove prompts with missing or invalid slots;

3. remove prompts that reveal the target value, except when the route definition requires it, as in relation inversion;

4. remove exact and normalised duplicates;

5. retain only prompts compatible with the target relation;

6. order or sample valid variants using the fixed experimental seed; and

7. save the resulting manifest for reuse across all evaluated methods.

Thus, differences between GRAPHSU and the baselines cannot be attributed to different evaluation questions: each method uses the same frozen target facts, route prompts, decoding settings, and soft-leakage threshold.

## F Taxonomy-Blind and Held-Out Leakage Probes

The templated route families in Appendix E are useful for controlled diagnosis, but robustness measured only on that taxonomy can overstate generalisation. We therefore perform an additional taxonomy-blind evaluation on all 184 TOFU complete-deletion targets. A local Qwen2.5-3B-Instruct model generates free-form probes using only the original question and answer. It receives no graph structure, route labels, templates, hyperparameters, or outputs from the unlearned models. The resulting probes have zero exact matches with the original template bank. Checkpoints and detector thresholds are frozen before this evaluation.

We evaluate 736 free-form probes (four per target) and a stronger attack panel of 342 probes across 171 targets, where each probe is first verified to recover the target answer from Original-SFT. Seed-Only and GRAPHSU use identical prompts and decoding. Leakage is measured with (i) a fixedreference representation derived from Original-SFT, (ii) external MPNet semantic similarity, (iii) DeBERTa natural-language inference, (iv) direct target-phrase recovery, and (v) a conservative combined detector that fires when any constituent detector succeeds.

The direction is also consistent for direct phrase recovery, MPNet, and NLI considered separately. For continuity with the main evaluation, the original soft-leakage detector is retained as an auxiliary measure: on the full taxonomy-blind panel it favours GRAPHSU (93.48% versus 96.20% for Seed-Only; $p \ = \ 0 . 0 1 8 7 )$ , while it saturates on the smaller confirmed-attack panel. These results extend the claim from robustness within the predefined route family to improved robustness against unseen, independently generated recovery attempts; they do not constitute certified erasure.

Table 6: Recovery-route families and representative generation rules. The displayed prompts are illustrative; the implementation uses relation-conditioned template maps so that each prompt remains compatible with its grounded target relation.
<table><tr><td>Route family</td><td>Generation rule</td><td>Representative prompt</td></tr><tr><td>Direct</td><td>Preserve the original question, optionally adding a neutral instruction wrapper.</td><td>&quot;Please answer:  $\langle q _ { i } \rangle ^ { \prime \prime }$ </td></tr><tr><td>Paraphrase</td><td>Apply relation-specific lexical and syntactic transformations while preserving hi, ri, and target ti.</td><td>&quot;At which institution is 〈hi〉 employed?&quot;</td></tr><tr><td>Indirect</td><td>Ask for the same relation using contextual or descriptive wording rather than the original surface form.</td><td>&quot;Which organisation is associated with 〈hi〉&#x27;s employment?&quot;</td></tr><tr><td>Cloze</td><td>Convert the target fact into a declarative statement and replace the target value with a blank.</td><td>&quot;〈hi〉 works at ，</td></tr><tr><td>Related-fact</td><td>Introduce another validated fact associated with hi or a selected graph neighbour, and then query the target relation.</td><td>“The researcher who studies (related topic) is affiliated with which institu- tion?&quot;</td></tr><tr><td>Entity alias</td><td>Replace the canonical entity mention with a validated alias from A(hi).</td><td>&quot;Where does (alias(hi)〉 work?&quot;</td></tr><tr><td>List-summary</td><td>Request several profile attributes while ensuring that the target relation is included.</td><td>&quot;List the main profile details for (hi), including institutional affiliation.&quot;</td></tr><tr><td>Relation inversion</td><td>Reverse the grounded relation using I(ri) when the inverse is meaningful and sufficiently unambiguous.</td><td>&quot;Which author is affiliated with  $\langle t _ { i } \rangle ! ^ { \gamma }$ </td></tr></table>

Table 7: Detailed route generation for one grounded target fact.
<table><tr><td>Route</td><td>Generated prompt</td><td>Purpose</td></tr><tr><td>Direct</td><td>Where does Avery Collins work?</td><td>Tests recovery through the original question.</td></tr><tr><td>Paraphrase</td><td>At which institution is Avery Collins employed?</td><td>Changes lexical and syntactic form while preserving the target relation.</td></tr><tr><td>Indirect</td><td>Which university is associated with Avery Collins&#x27;s employment?</td><td>Avoids the original “where does ... work&quot; formulation.</td></tr><tr><td>Cloze</td><td>Avery Collins works at</td><td>Tests whether the target completes a factual statement. Uses a related fact to reach the target entity; emitted only when that</td></tr><tr><td>Related-fact</td><td>The researcher known for work on machine ethics is affiliated with which institution?</td><td>fact identifies the intended entity without ambiguity.</td></tr><tr><td>Entity alias</td><td>Where does A. Collins work?</td><td>Tests recovery after replacing the canonical entity name with a validated alias.</td></tr><tr><td>List-summary</td><td>List Avery Collins&#x27;s main profile details, including institutional affiliation.</td><td>Tests whether the target appears within a broader profile response.</td></tr><tr><td>Relation inversion</td><td>Which author is affiliated with Northbridge University?</td><td>Reverses the relation direction; emitted only when the inverse map- ping is defined and sufficiently unambiguous.</td></tr></table>

## G Qualitative Deletion Examples

The examples below use only the concepts defined in the main paper: the deletion setting, forget seed, grounded target fact, edit scope, support routes, and retention constraint. They illustrate how requests are represented and evaluated rather than introducing a separate response-transformation framework.

## G.1 Complete deletion

Forget seed. Question: Where does Avery Collins work? Answer: Avery Collins works at Northbridge University. The grounded target has head Avery Collins, relation works\_at, and tail Northbridge University.

Edit scope and interpretation. All answer tokens in the forget seed receive full forgetting pressure, while graph-selected support examples receive bounded pressure according to their requestconditioned support scores. A successful update prevents recovery of Northbridge University through applicable direct, paraphrase, cloze, related-fact, list-summary, and relation-inversion prompts while preserving utility on unrelated retain examples.

## G.2 Entity/instance deletion

Deletion request. Remove the profile associated with Hsiao Yun-Hwa. One seed asks: What is thefull name ofthe author born in Taipei, Taiwan on 05/11/1991 who writes in the leadership genre? The seed answer is: The author’sfull name is Hsiao Yun-Hwa.

Edit scope and interpretation. Canonical-name, alias, and entity-profile examples form the seed forget set, and additional high-risk support examples are selected through the request-conditioned graph closure. A successful update prevents recovery of the target profile through canonical-name, alias, indirect, related-fact, and list-summary prompts while preserving factual responses for unrelated entities. The target is suppressed without introducing a manually selected generic identity.

Table 8: Taxonomy-blind leakage on unseen free-form probes for TOFU complete deletion. Reduction is Seed-Only minus GRAPHSU in percentage points. Confidence intervals are paired target-level 95% bootstrap intervals; p is the two-sided McNemar test.
<table><tr><td>Panel</td><td>Detector</td><td>Seed-Only</td><td>GRAPHSU</td><td>Reduction</td><td>95% CI</td><td>McNemar p</td></tr><tr><td>All targets</td><td>Fixed-reference</td><td>74.18%</td><td>61.68%</td><td>12.50 pp</td><td>[8.02, 17.26]</td><td> $1 . 0 3 \times 1 0 ^ { - 8 }$ </td></tr><tr><td>All targets</td><td>Combined</td><td>75.68%</td><td>62.09%</td><td>13.59 pp</td><td>[8.97, 18.34]</td><td> $5 . 9 8 \times 1 0 ^ { - 1 0 }$ </td></tr><tr><td>Confirmed attacks</td><td>Fixed-reference</td><td>78.95%</td><td>65.20%</td><td>13.74 pp</td><td>[7.02, 20.76]</td><td> $4 . 9 2 \times 1 0 ^ { - 5 }$ </td></tr><tr><td>Confirmed attacks</td><td>Combined</td><td>79.82%</td><td>65.50%</td><td>14.33 pp</td><td>[7.89, 21.05]</td><td> $2 . 2 3 \times 1 0 ^ { - 5 }$ </td></tr></table>

## G.3 Partial deletion

Forget seed. Question: Where does Avery Collins work, and in which language do they write? Answer: Avery Collins works at Northbridge University and writes in Spanish. The sensitive target is the works\_at value Northbridge University; the safe fact is that Avery Collins writes in Spanish.

Edit scope and interpretation. Forgetting is localised to the target span, while the safe answer span remains supervised; representation repulsion is disabled for this setting. A successful update prevents recovery of the institutional affiliation across applicable routes while retaining the model’s ability to answer that Avery Collins writes in Spanish.

These examples clarify the three deletion granularities. Complete deletion suppresses the full answer associated with a seed; entity/instance deletion expands the request across an entity-centred profile and its aliases; and partial deletion suppresses only the grounded sensitive span while preserving the safe answer remainder.

## G.4 Diagnosing Residual Leakage in the Support Graph

The support graph localises potential support at the training-example level; it does not claim to identify a single neuron or parameter in which a fact is stored. When a target remains recoverable after unlearning, the request-conditioned graph provides a diagnostic view through each node’s diffusion score $r _ { i } ,$ hop distance from the forget seed, and the semantic, symbolic, and gradient views contributing to its incident edges. Residual leakage can arise when relevant evidence lies just below the selection threshold or top-B expansion budget, when support is distributed across several individually weak paths, or when partial deletion intentionally limits forgetting pressure because sensitive and safe facts share the same answer context. Thus, difficult-toremove information may correspond to redundant or diffuse support routes rather than a single missed example. This interpretation is diagnostic rather than a calibrated causal localisation claim.

## H Operational Cost, Memory, and Scalability

Unless otherwise stated, the measurements below use GPT-2 Medium on TOFU. Corpus-level facts and graph artefacts are cached and reused across deletion requests.

## H.1 Graph-construction scaling benchmark

We benchmark nested TOFU subsets using GPT-2 Medium. GPT-4.1 fact extractions were already cached and are therefore excluded from the buildtime measurements. Embedding and symbolic processing ran on the CPU, while gradient-alignment features used the GPU. The graph degree is capped at 30.

At 4,000 samples, graph construction takes 452.36 seconds (about 7.5 minutes), cached artefacts occupy 7.82 MB, and warm support selection takes 4.01 seconds. Because the stored degree is capped at K = 30, the number of stored edges is O(NK) and therefore linear in N for fixed K; graph storage and each sparse diffusion iteration scale with the stored edge set. The current exact cosine candidate-retrieval step is approximately quadratic in the number of samples and may become the dominant bottleneck at much larger corpus sizes. Approximate nearest-neighbour retrieval for the semantic view and symbolic inverted indices are natural scalable replacements that preserve the sparse downstream graph.

## H.2 Request-time and audit costs

The isolated warm/cold timings in Table 9 measure graph loading and support selection on cached subset artefacts. The full request-conditioned pipeline additionally includes request grounding, one- and two-hop retrieval, support scoring, and retain-safety filtering.

Table 9: Graph construction, memory, cache size, and request-conditioned selection cost as the number of stored QA samples increases. Warm selection reuses in-memory cached artefacts; cold selection includes loading cached graph artefacts before support selection.
<table><tr><td>Samples</td><td>Stored edges</td><td>s Avg. degree</td><td>Build time (s)</td><td>CPU RAM (GB)</td><td>GPU mem. (GB)</td><td>Cache (MB)</td><td>Warm sel. (s)</td><td>Cold sel. (s)</td></tr><tr><td>1K</td><td>11,279</td><td>22.56</td><td>106.01</td><td>1.41</td><td>2.36</td><td>2.87</td><td>0.68</td><td>1.27</td></tr><tr><td>2K</td><td>22,795</td><td>22.80</td><td>212.41</td><td>1.45</td><td>2.47</td><td>4.93</td><td>1.74</td><td>3.02</td></tr><tr><td>4K</td><td>45,524</td><td>22.76</td><td>452.36</td><td>1.40</td><td>2.36</td><td>7.82</td><td>4.01</td><td>6.53</td></tr></table>

Table 10: Stage-level operational costs from the full deletion workflow.
<table><tr><td>Stage</td><td>Observed cost or workload</td></tr><tr><td>Request-conditioned support selection</td><td>11.4–33.4 seconds per deletion request (mean 25.9 seconds), including request grounding, one- and two-hop support retrieval, support scoring, and</td></tr><tr><td>Model update and orchestration</td><td>retain-safety filtering. No external calls are used. Approximately 5.7–6.9 minutes per request, mea- sured as total wall time minus logged evaluation time. The residual includes setup and checkpoint-</td></tr><tr><td>Full recovery-route audit</td><td>ing and is not a pure training-time measurement. 63.0–83.1 minutes and 4,464–5,376 local model generations. This is an offline validation cost rather than request-time latency and uses no external</td></tr></table>

With cached corpus artefacts, the recurring graph-specific overhead is small relative to model updating and the offline recovery-route audit. Fact extraction and graph construction are one-time corpus-level preprocessing costs and are amortised across deletion requests.

## I Additional Utility Evaluation Beyond PPL

Perplexity is useful for detecting broad languagemodel degradation, but it does not fully capture whether retained factual behaviour is preserved. We therefore interpret soft leakage jointly with retain F1, ROUGE-L F1, far-retain F1, neighbourretain F1, and safe-span F1 for partial deletion. The PPL ≤ 10 criterion used in the main paper is an operational guardrail for these experiments rather than a universal unlearning threshold.

Table 11: Primary TOFU forgetting and utility metrics for the controlled Seed-Only comparison.
<table><tr><td>Setting</td><td>Method</td><td>Soft leak↓</td><td>Retain PPL↓</td><td>Retain F1↑</td><td>ROUGE-L F1↑</td></tr><tr><td>Complete</td><td>Seed-Only</td><td>93.25</td><td>2.61</td><td>39.07</td><td>35.03</td></tr><tr><td>Complete</td><td>GRAPHSU</td><td>46.83</td><td>3.27</td><td>42.41</td><td>47.36</td></tr><tr><td>Entity</td><td>Seed-Only</td><td>96.25</td><td>4.40</td><td>37.16</td><td>33.38</td></tr><tr><td>Entity</td><td>GRAPHSU</td><td>54.60</td><td>4.94</td><td>59.11</td><td>56.82</td></tr><tr><td>Partial</td><td>Seed-Only</td><td>100.00</td><td>7.90</td><td>3.20</td><td>3.01</td></tr><tr><td>Partial</td><td>GRAPHSU</td><td>81.67</td><td>1.91</td><td>55.14</td><td>51.07</td></tr></table>

Table 12: Locality-sensitive retain metrics corresponding to Table 11. Safe-span F1 applies only to partial deletion.
<table><tr><td>Setting</td><td>Method</td><td>Far-retain F1↑</td><td>Neighbour F1↑</td><td>Safe-span F1↑</td></tr><tr><td>Complete</td><td>Seed-Only</td><td>41.05</td><td>31.79</td><td>一</td></tr><tr><td>Complete</td><td>GRAPHSU</td><td>51.06</td><td>32.77</td><td></td></tr><tr><td>Entity</td><td>Seed-Only</td><td>43.42</td><td>34.69</td><td>一</td></tr><tr><td>Entity</td><td>GRAPHSU</td><td>57.13</td><td>46.17</td><td></td></tr><tr><td>Partial</td><td>Seed-Only</td><td>1.14</td><td>0.83</td><td>1.26</td></tr><tr><td>Partial</td><td>GRAPHSU</td><td>54.58</td><td>54.54</td><td>61.31</td></tr></table>

The controlled comparison shows a stronger forgetting–utility balance for GRAPHSU in all three settings. Under complete deletion, GRU reaches lower soft leakage (38.13%) than GRAPHSU but raises retain PPL to 79.11, illustrating why leakage should not be interpreted without retention. Under partial deletion, NPO-TR has a slightly higher safe-span F1 (64.97) than GRAPHSU (61.31), but its retain PPL is 10.75 and soft leakage remains 95.13%. Neighbour-retain F1 is preserved or improved by GRAPHSU relative to Seed-Only: 32.77 vs. 31.79 for complete deletion, 46.17 vs. 34.69 for entity/instance deletion, and 54.54 vs. 0.83 for partial deletion.

## J Ablation Coverage and Hyperparameter Scope

The controlled Seed-Only comparison is an ablation of the graph-guided scope controller as a whole: it uses the same local unlearning objective but omits graph-neighbour expansion. The local-extractor and corruption experiments in Appendix D further perturb the symbolic component while leaving the semantic and gradient views available. Together, these experiments establish that graph-guided support expansion matters and that the fused graph is robust to substantial symbolic noise.

The present experimental archive does not contain a complete leave-one-view-out sweep for all five affinity terms or independent performance sweeps over the fusion weights, restart probability $\rho ,$ hop limit, and expansion budget B. We therefore do not attribute the observed gains to any single graph view and do not claim that the reported hyperparameters are optimal. The evaluated configuration uses all five views, the fusion weights reported in Table 1, one- and two-hop support retrieval, and the task-specific neighbour caps in Table 1. A full factorial sensitivity analysis would require additional training runs and remains an important direction for establishing which views are necessary under different data regimes.

## K Evaluation Consistency and Statistical Reliability

All reported retain PPL values use the same teacherforced protocol within a comparison: prompt and padding positions are masked, answer-token NLL is accumulated over the frozen retain set, the total is divided by the number of scored answer tokens, and the resulting mean NLL is exponentiated once. This definition implies PPL ≥ 1 and motivated the correction in Appendix A. The local-extractor experiment additionally reports the Original-SFT retain PPL (4.30) at its matched operating point.

The unlearning runs reported in the paper use one training seed. We therefore do not describe evaluation resampling as across-seed variance. For the independently generated taxonomy-blind probes, Table 8 reports paired target-level bootstrap confidence intervals and two-sided McNemar tests because the same probes are evaluated under both checkpoints. The available aggregate benchmark outputs do not support a defensible reconstruction of across-seed variance for all baselines; accordingly, no such variance is fabricated. Training-seed sensitivity remains a limitation of the current empirical study. Future multi-seed comparisons should use paired target-level intervals for leakage, paired example-level intervals for retain F1/ROUGE-L, and example-resampled aggregate NLL before exponentiation for PPL, with multiplicity correction when several baselines are tested.