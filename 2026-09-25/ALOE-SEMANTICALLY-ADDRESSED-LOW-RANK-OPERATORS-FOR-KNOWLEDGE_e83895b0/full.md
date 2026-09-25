# ALOE: SEMANTICALLY ADDRESSED LOW-RANK OPERATORS FOR KNOWLEDGEEDITING

Zeyan Li<sup>1</sup>, Hu Xu<sup>1</sup>, Jianfeng Xu<sup>1∗</sup>

<sup>1</sup> Shanghai Jiao Tong University

## ABSTRACT

Knowledge editing changes what a model knows by modifying parameters so that a requested fact updates while unrelated behavior is preserved. This is usually treated as a write problem, but editing also involves an address problem: deciding which hidden states should receive the new residual. An update that activates too narrowly memorizes one prompt, while one that activates too broadly disrupts neighboring knowledge. Parametric editors encode this scope implicitly, whereas memory-based editors make the selection explicit but keep it outside the edited model. We propose ALOE (Addressed Low-rank Operator for Editing), which learns semantic addresses from paraphrases and hard same-subject negatives, aligns them with autoregressive hidden states through rollout refinement and gate calibration, and embeds the resulting gated low-rank operator within one MLP layer, so that the deployed model runs in a single forward pass with no external retriever or auxiliary router. Evaluated on CounterFact, ZSRE, and KnowEdit across three 7–8B model families, ALOE achieves efficacy between 0.955 and 0.999 and locality between 0.981 and 1.000; mechanistic analyses confirm that the learned geometry separates competing edits and that calibration suppresses out-of-scope activation. The remaining errors concentrate in paraphrase coverage and write fitting.

Index Terms— knowledge editing, low-rank operators, language models, model adaptation

## 1. INTRODUCTION

Language models store factual associations in their parameters, including feed-forward layers and individual neurons. Knowledge editing aims to revise such associations on request: the edited model should produce the new fact on the original request (efficacy), transfer the change to equivalent wordings of that request (generalization), and leave unrelated behavior untouched (locality). All three criteria depend on one choice that is rarely made explicit: where in activation space the update is allowed to act.

Editing therefore involves two distinct problems. The first is a write problem: specifying the residual that encodes the new fact. The second is an address problem: deciding which hidden states should receive that residual. A birthplace edit, for example, should cover rewordings of the same question without changing the model’s answer about the person’s employer, and at scale, different subjects that share a relation must receive distinct writes. An update whose address is too narrow memorizes the training prompt and misses its rewordings; one whose address is too broad activates on inputs it should leave alone (Fig. 1). These boundary failures are documented in current editors: an edit perturbs other facts about the same subject [1, 2], damages knowledge beyond its intended scope [3], and fails to propagate to facts implied by the edited one [4].

One write, different semantic scopes  
![](images/ff703a1c5cde42aba598d7d2cd1834ba0004b15700e2ee4b7c197679d546dd7a.jpg)  
Fig. 1. The address problem. The same write can miss paraphrases or affect neighboring facts. Filled cells denote an active write; empty cells denote no update (schematic).

Existing editors handle the address problem in one of two ways. ROME and MEMIT encode facts through low-rank MLP updates, while PMET refines the participating states [5, 6, 7]. In such an update $\Delta W = U V ^ { \top }$ , which acts on the layer input h(x) of a prompt x, the term $V ^ { ^ { \top } } h ( x )$ already serves as an address for each write column, but its factors are never trained to separate paraphrases from semantic neighbors, and causal localization need not identify the most editable layer [8]. SERAC, GRACE, and WISE instead make the selection explicit through a classifier, a codebook, or a side memory [9, 10, 11], but their deployed systems perform this selection outside the edited weights.

To obtain selection that is both explicitly learned and executed inside the edited model, we propose ALOE (Addressed Low-rank Operator for Editing). During construction, asymmetric query/key maps learn a scope representation for each edit from three kinds of examples: paraphrases of the request, locality inputs that should remain unchanged, and same-subject prompts whose relation differs from the edited one, which serve as hard negatives. The resulting codes initialize address vectors inside one MLP layer, and these addresses are refined and calibrated on autoregressive rollouts so that they respond to the states the model produces at generation time. A joint solve then fits the write directions under the calibrated gates. At inference, the edited model needs no external component, because the address is a trained part of the MLP itself. Our contributions are as follows. First, we formulate knowledge editing as a coupled address–write problem and show empirically that the two components fail independently, so each can be measured and diagnosed on its own. Second, we realize the address as an intrinsic gated low-rank MLP operator: it learns semantic edit boundaries at construction time and executes them inside the edited layer at inference. Third, on CounterFact, ZSRE, and KnowEdit across three 7– 8B model families, ALOE attains efficacy between 0.955 and 0.999 and locality between 0.981 and 1.000 on edit streams of 839 to 1,301 facts.

## 2. RELATED WORK

## 2.1. Parametric knowledge editing

Parametric editors differ mainly in how they construct and constrain weight updates. KnowledgeEditor and MEND learn to transform edit gradients into parameter changes, and MALMEN extends learned editing to large batches [12, 13, 14]. ROME instead writes a factual association directly into an MLP through a rank-one update, and MEMIT extends this construction to many edits at once [5, 6]; carefully configured fine-tuning remains a useful reference. More recent methods such as AlphaEdit constrain updates to preservation or orthogonal subspaces to reduce interference with existing knowledge [15, 16].

These methods decide how and where new knowledge is written into shared parameters. The scope of the resulting edit—which inputs should activate it—is induced implicitly by the interaction between the update and the model’s hidden states.

## 2.2. Selective access and edit scope

A second line of work makes access to edited knowledge explicit. SERAC retrieves a counterfactual model, GRACE stores edits in discrete key–value adaptors, and WISE routes inputs among memories [9, 10, 11]. In-context editing avoids parameter modification altogether by placing updated facts in demonstrations [17]. Locatethen-edit and associated-knowledge methods improve the placement or propagation of edits. In all of these systems, the selection mechanism lives outside the edited weights—a retriever, a codebook, a router, or the prompt—or targets where an edit should be placed rather than when it should fire. ALOE learns semantic addresses that gate low-rank writes directly inside an edited MLP, so selective access becomes part of the deployed model itself.

The scope of an edit is also central to how editing is evaluated. CounterFact+ strengthens tests of specificity [18], and RippleEdits and ReCoE measure whether an edit propagates to related knowledge [4, 19]. Sequential-edit studies show that poorly controlled updates cause forgetting and degrade general abilities. These evaluation results point to a shared requirement: an edit should activate for equivalent or relevant queries and remain inactive for nearby but out-of-scope knowledge. We formulate this boundary explicitly as the address of an edit and study it separately from the write itself.

## 3. METHOD

## 3.1. An addressed residual operator

Figure 2 illustrates the construction pipeline. Let $f _ { \theta }$ be a frozen transformer, and let $\mathcal { E } = \{ ( t _ { i } , y _ { i } ) \} _ { i = 1 } ^ { n ^ { - } }$ denote the requested edits, where $t _ { i }$ is the request text and y<sub>i</sub> is its target. We augment a single MLP down-projection $W \in \mathbb { R } ^ { d _ { \mathrm { o u t } } \times d }$ , whose input state is $h \in  { \mathbb { R } } ^ { d } .$ ALOE gives each edit two learned vectors: an address that decides which hidden states the edit applies to, and a write direction that carries the new fact. The address gates the write direction, so the edit only enters the output where its address activates:

$$
\begin{array} { r } { \bar { h } = h / \operatorname* { m a x } ( \| h \| _ { 2 } , \epsilon ) , } \end{array}
$$

$$
\begin{array} { r } { g ( h ) = \phi \Big ( \pmb { \alpha } \odot ( \pmb { V } ^ { \top } \bar { h } - \pmb { \tau } ) \Big ) , } \end{array}\tag{1}
$$

$$
F _ { \mathrm { A L O E } } ( h ) = W h + U g ( h ) .\tag{2}
$$

Here $V = [ v _ { 1 } , \ldots , v _ { n } ] \in \mathbb { R } ^ { d \times n }$ contains the semantic addresses, $U = [ u _ { 1 } , \dotsc , u _ { n } ] \in \mathbf { \widehat { \mathbb { R } } } ^ { d _ { \mathrm { o u t } } \times n }$ contains the write directions, and $\pmb { \tau } , \pmb { \alpha } \in \mathbb { R } ^ { n }$ are per-edit thresholds and temperatures. Normalizing the input state in h<sup>¯</sup> makes each gate depend on the direction of the hidden state rather than its magnitude. The dead-zone sigmoid $\phi ( z ) = \mathrm { m a x } ( \sigma ( z ) - \epsilon _ { g } , 0 ) / ( 1 - \epsilon _ { g } )$ with $\epsilon _ { g } = 1 0 ^ { - 3 }$ drives the gates of inactive addresses to exactly zero while retaining continuous activation above threshold, so an edit whose address does not match has no effect on the output. The operator is therefore state-dependent: the hidden state at each token determines which write directions enter the output, in contrast to the fixed linear update $W + U V ^ { \top }$ of standard low-rank editing.

## 3.2. Learning the semantic boundary

The first construction stage learns where each residual should act. For edit i, the full request serves as the canonical key, so that the address retains the identity of both the subject and the relation. Each training item contains a paraphrase $\boldsymbol { x } _ { i } ^ { + }$ of the request, a locality input $\boldsymbol { x } _ { i } ^ { - }$ that should not activate the edit, and same-subject prompts $r _ { i k } ^ { - }$ whose predicates differ from $t _ { i } ;$ because these prompts share the subject with the edit, they are harder negatives than random prompts.

We learn two distinct linear maps Q, $K \in \mathbb { R } ^ { p \times d }$ and a positive scale c. With $k _ { i } = { \mathrm { n o r m } } ( K h ( t _ { i } ) )$ as the key of edit $i ,$ the construction-time score between a query x and edit i is

$$
s ( x , t _ { i } ) = c \langle Q h ( x ) , k _ { i } \rangle .\tag{3}
$$

A high score means the query falls inside the edit’s scope. The training loss ranks $s ( x _ { i } ^ { + } , t _ { i } )$ above the scores of wrong-relation queries and keys, separates different requests within a minibatch, and enforces the locality margin $s ( x _ { i } ^ { - } , \bar { t } _ { i } ) + m < s ( x _ { i } ^ { + } , t _ { i } )$ . Orthogonality regularization discourages the keys of different edits from collapsing onto each other, and relation-balanced minibatches prevent frequent predicates from dominating the loss. Labels and contrast sets are used only during construction; once the addresses are built, inference relies on the resulting address parameters alone.

## 3.3. From construction to generation

The metric learned above separates construction prompts, but the deployed address must fire on hidden states encountered during generation, which need not coincide with the states seen at construction time: once the model begins producing its own tokens, its hidden states drift away from those measured on pre-written prompts. To bridge this shift, we initialize the address of edit i as

$$
\boldsymbol { v } _ { i } ^ { ( 0 ) } = c \boldsymbol { Q } ^ { \top } \operatorname { n o r m } ( K h ( t _ { i } ) ) , \qquad c > 0 ,\tag{4}
$$

where norm $\mathbf { \Lambda } _ { ( z ) } ~ = ~ z / \| z \| _ { 2 }$ Transposition preserves the ordering induced by the learned metric, and the normalization in Eq. (1) removes the magnitude of the hidden state. We then collect normalized anchor states $\mathbf { \bar { \nabla } } H ^ { + }$ and negative states $H ^ { - }$ from left-padded autoregressive rollouts, in which the model generates continuations of each request and we record the states it passes through, and refine $V ^ { ( 0 ) }$ on these generation-time states by

$$
\mathcal { L } _ { \mathrm { a d d r } } = \frac { 1 } { n } \sum _ { i } \bigg [ m _ { i } - \operatorname* { m i n } _ { h \in \mathcal { P } _ { i } } v _ { i } ^ { \top } h + \operatorname* { m a x } _ { h \in \mathcal { N } _ { i } } v _ { i } ^ { \top } h \bigg ] _ { + } ^ { 2 } ,\tag{5}
$$

for up to 3,000 AdamW steps at learning rate 0.01, preserving the norms of the addresses. This loss pushes each address toward its worst-scoring anchor state and away from its worst-scoring negative state, so that even the least typical valid phrasing still opens the gate. Edits with identical input–target pairs share one address support, while conflicting targets remain separate constraints. Calibration then converts the refined scores into gates. For a separable slot $i ,$ it sets $\tau _ { i }$ and $\alpha _ { i }$ so that the worst negative falls below the dead zone and the worst positive maps to 0.9; for a slot whose states cannot be separated, a finite-temperature midpoint initialized at 8 minimizes a balanced worst-case classification loss. After refinement, the metric-learning components are discarded: Q, K, the labels, and the prompts are no longer needed, and only V , τ , and α remain in the edited model.

![](images/f0a9547d18d338db86f192e20408aeaac9e5494263f6b585d8e2bcab02fa4ca3.jpg)  
Fig. 2. ALOE learns semantic addresses, refines and calibrates continuous edit gates, and inserts the resulting low-dimensional residual into one MLP projection. The edited LLM executes the operator in a standard forward pass.

## 3.4. Fitting writes jointly

Once the address support is fixed, the remaining task is to fit write directions that are compatible with the gates. Let Y contain target residuals obtained by independently optimizing each edit for 25 steps, compressing its down-projection change to rank 16, and evaluating the compressed change on its anchor; column i of Y is thus the output change that edit i should produce when its gate is fully open. With $G ^ { + } = g ( H ^ { + } )$ and $G ^ { - } \stackrel { - } { = } g ( H ^ { - } )$ collecting the gate activations of anchor and negative states, a preservation-regularized ridge solve initializes all writes jointly:

$$
U _ { 0 } = { Y } { G ^ { + } } ^ { \top } \Big ( { G ^ { + } } { G ^ { + } } ^ { \top } + \lambda { G ^ { - } } { G ^ { - } } ^ { \top } + \mu I \Big ) ^ { - 1 } .\tag{6}
$$

The gates of different edits can overlap, so one edit’s write direction can leak through another edit’s gate; solving for all writes in one system accounts for this cross-activation, and the $G ^ { - } G ^ { - \top }$ term keeps the writes small on states where no edit should fire. With V, τ, and α frozen, we then optimize U from $U _ { 0 }$ for up to 100 AdamW steps using the negative log-likelihood of the autoregressive targets. At deployment, the model computes all gates densely and adds $U g ( h )$ to Wh, which requires $O ( n \bar { ( } d + d _ { \mathrm { o u t } } ) )$ ) storage and compute linear in the number of edits.

## 4. EXPERIMENTS

## 4.1. Setup

We evaluate end-to-end editing on CounterFact [5], ZSRE [20], and KnowEdit [21], covering factual replacement, question-answer rephrasing, and diverse knowledge domains. The main comparison uses Llama-2-7B, Llama-3.1-8B-Instruct, and Qwen3-8B [22, 23, 24], and the address-transfer study adds Qwen2.5-7B-Instruct [25]. All models are run with seeds 0, 42, and 99, and every result reported in this section is the mean over the three runs. The comparison includes five established knowledge editors: ROME [5], MEMIT [6], MEND [13], GRACE [10], and AlphaEdit [15]. We report the standard dimensions separately: efficacy is exact match on the edited request, generalization is exact match on rephrases, and locality is normalized exact-generation stability on out-of-scope prompts relative to the unedited model. Baselines use their published configurations through EasyEdit where supported [26].

## 4.2. End-to-end editing

Table 1 reports the three metrics separately for each model– benchmark pair. The baselines split into two failure patterns. On Llama-2-7B and Llama-3.1-8B-Instruct, the locate-and-edit and meta-learning methods collapse under edit streams of this length: ROME, MEMIT, and MEND lose almost all efficacy on Counter-Fact, and GRACE retains efficacy only on Llama-2 while activating on almost no rephrase anywhere. On Qwen3-8B these methods survive, but the survivors pay for generalization with locality: ROME’s rephrase accuracy comes with locality at or below 0.20, and MEMIT and AlphaEdit show the same exchange at milder levels. AlphaEdit is the strongest baseline overall, yet its profile is uneven across model families—competitive on Llama-2, it keeps high generalization on Llama-3 only while locality falls to 0.33–0.58, and on Qwen3 both its efficacy and its locality drop well below ALOE’s. ALOE is the only method whose efficacy stays between 0.955 and 0.999 and whose locality stays between 0.981 and 1.000 in all nine model–benchmark cells, so its advantage is consistency: no failure cell, on any model family, under streams of up to 1,301 edits. The cost is equally visible. Generalization ranges from 0.217 to 0.472, below the best baseline cells on each model.

## 4.3. Mechanism analysis

We test the address mechanism on its own, using disjoint CounterFact train, calibration, and development partitions organized by predicate, with development subjects and exact surfaces excluded from training; the development set contains 632 examples from 34 predicates, each query paired with 16 fixed same-subject wrongrelation keys. We measure scope AUC, calibration-threshold recall, 16-way same-subject top-1, and transferred-threshold negative FPR against equal-dimensional raw activation controls, with all runs at seeds 0, 42, and 99. The learned address beats the raw controls by a wide margin: AUC 0.9976 against 0.6365, recall 0.9910 against 0.1804, same-subject top-1 0.8576 against 0.3117, and FPR 0.0418. Standard deviations across seeds stay below 0.006 on every metric, so the selectivity comes from the learned geometry rather than from the hidden states themselves. The choice of key matters just as much: an entity-neutral key collapses 632 edits onto 179 unique addresses, while the full request keeps all 632 distinct and raises syntheticwrite top-1 from 0.698 to 0.998. The same construction objective transfers without retuning to the other three model families, which reach AUC 0.997–0.998 and FPR 0.036–0.042; their slightly lower same-subject top-1 (0.830–0.848) points to architecture-dependent separation margins.

Table 1. Efficacy (Eff.), generalization (Gen.), and locality (Loc.) across base models and benchmarks. ALOE fits one operator to the complete benchmark stream (839 CounterFact, 1,266 KnowEdit, and 1,301 ZSRE edits); all results are means over seeds 0, 42, and 99. Best and second-best values are bolded and underlined, respectively.
<table><tr><td></td><td colspan="3">ALOE</td><td colspan="3">ROME (2022)</td><td colspan="3">MEMIT (2023)</td><td colspan="3">MEND (2022)</td><td colspan="3">GRACE (2023)</td><td colspan="3">AlphaEdit (2025)</td></tr><tr><td>Benchmark</td><td>Eff.</td><td>Gen.</td><td>Loc.</td><td>Eff.</td><td>Gen.</td><td>Loc.</td><td>Eff.</td><td>Gen.</td><td>Loc.</td><td>Eff.</td><td>Gen.</td><td>Loc.</td><td>Eff.</td><td>Gen.</td><td>Loc.</td><td>Eff.</td><td>Gen.</td><td>Loc.</td></tr><tr><td colspan="9">Llama-2-7B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CounterFact</td><td>0.956</td><td>0.289</td><td>0.982</td><td>0.093</td><td>0.083</td><td>0.014</td><td>0.056</td><td>0.188</td><td>0.030</td><td>0.011</td><td>0.012</td><td>0.005</td><td>0.949</td><td>0.003</td><td>0.972</td><td>0.821</td><td>0.129</td><td>0.838</td></tr><tr><td>KnowEdit</td><td>0.999</td><td>0.472</td><td>1.000</td><td>0.121</td><td>0.106</td><td>0.028</td><td>0.162</td><td>0.150</td><td>0.195</td><td>0.002</td><td>0.002</td><td>0.002</td><td>0.980</td><td>0.086</td><td>1.000</td><td>0.969</td><td>0.801</td><td>1.000</td></tr><tr><td>ZSRE</td><td>0.997</td><td>0.464</td><td>1.000</td><td>0.204</td><td>0.183</td><td>0.019</td><td>0.144</td><td>0.138</td><td>0.137</td><td>0.003</td><td>0.003</td><td>0.003</td><td>0.974</td><td>0.009</td><td>1.000</td><td>0.977</td><td>0.891</td><td>1.000</td></tr><tr><td colspan="9">Llama-3.1-8B-Instruct</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CounterFact 0.959</td><td></td><td>0.281</td><td>0.981</td><td>0.096</td><td>0.086</td><td>0.003</td><td>0.031</td><td>0.006</td><td>0.341</td><td>0.010</td><td>0.010</td><td>0.005</td><td>0.423</td><td></td><td>0.002</td><td>0.971</td><td>0.982 0.813</td><td>0.325</td></tr><tr><td>KnowEdit</td><td>0.999</td><td>0.459</td><td>1.000</td><td>0.141</td><td>0.108</td><td>0.010</td><td>0.079</td><td>0.073</td><td>0.028</td><td>0.026</td><td>0.026</td><td>0.008</td><td>0.350</td><td>0.018</td><td></td><td>1.000 0.996</td><td>0.798</td><td>0.578</td></tr><tr><td>ZSRE</td><td>0.995</td><td>0.407</td><td>1.000</td><td>0.162</td><td>0.138</td><td>0.020</td><td>0.101</td><td>0.097</td><td>0.151</td><td>0.007</td><td>0.007</td><td>0.006</td><td>0.343</td><td>0.019</td><td>1.000</td><td>0.994</td><td>0.901</td><td>0.527</td></tr><tr><td colspan="9">Qwen3-8B</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>CounterFact</td><td>0.955</td><td>0.217</td><td>0.981</td><td>0.979</td><td>0.571</td><td>0.091</td><td>0.859</td><td>0.542</td><td>0.276</td><td>0.012</td><td>0.013</td><td>0.007</td><td>0.414</td><td>0.001</td><td></td><td>0.971</td><td>0.8390.533</td><td>0.305</td></tr><tr><td>KnowEdit</td><td>0.998</td><td>0.359</td><td>1.000</td><td>0.986</td><td>0.625</td><td>0.194</td><td>0.946</td><td>0.636</td><td>0.450</td><td>0.005</td><td></td><td>0.004 0.002</td><td></td><td>0.349 0.008</td><td></td><td>1.000</td><td>0.8170.554</td><td>0.432</td></tr><tr><td>ZSRE</td><td></td><td>0.994 0.376 1.000</td><td></td><td>0.981</td><td></td><td>0.8980.202</td><td>0.845 </td><td>0.751</td><td>0.447</td><td></td><td></td><td>0.0060.005 0.005</td><td></td><td></td><td></td><td></td><td>0.343 0.011 1.0000.914 0.796</td><td>0.483</td></tr></table>

(a) Gates  
![](images/89bd5aac7a00d0d8f8082b92b94faf68a30408a231213c2d83e531ebb0478821.jpg)

(b) Response  
![](images/83d3d0127cd9406f541555465713a22834035c62c41bb94cb72b78ed5ea94810.jpg)

(c) States  
![](images/205658b8f35d587952dcbbf1503698760b73551ed03769fa1ac3cea286eb7e4b.jpg)  
Fig. 3. Runtime behavior after 839 CounterFact edits on Llama-3.1-8B-Instruct. (a) Matched-slot gates for edits and rephrases, and the maximum gate for locality queries. (b) Responses of the first 32 sampled queries to their associated slots. (c) Paired final-block locality states under shared PCA, with empirical marginals and a full-range inset. PC1/PC2 explain 45.1%/10.7%; mean relative state drift is 11.54% and next-token agreement is 96.5%.

The deployed checkpoint tells the same story from the runtime side and locates the remaining failures. On 256 fixed CounterFact cases drawn from the 839-edit model (Fig. 3), the matched slot dominates 98.4% of original requests, and locality states drift by 11.54% on average with a median of zero, consistent with the near-perfect locality scores; the visible weakness is on rephrases, whose mean gate activation is 0.057 against 0.860 for original requests, so most rephrasings never open the gate. Controlled interventions on 256 CounterFact edits, removing one construction stage at a time with the data, layer, and write budget fixed, assign this gap to specific stages: rollout refinement contributes 55.0 points of efficacy, confirming that the construction metric must be aligned with generation-time states; calibration contributes 24.8 points of efficacy and 96.5 points of locality while slightly reducing generalization, because thresholds suppress false activation but cannot create paraphrase support that the address lacks; the preservation term changes every measure by at most half a point. Paraphrase coverage is therefore an address problem, visible in the gate traces, while the residual gap between an open gate and a correct generation is a write-fitting problem, visible in the transfer study where address quality holds but end-to-end accuracy does not.

## 5. CONCLUSION

We formulated knowledge editing as a coupled address–write problem and proposed ALOE, which makes the address intrinsic to an edited MLP. An asymmetric metric learns the scope of each edit from paraphrases and same-subject hard negatives, rollout refinement and gate calibration align this scope with generation-time hidden states, and a joint solve fits the write directions under the calibrated gates. On CounterFact, ZSRE, and KnowEdit across three 7–8B model families, ALOE attains efficacy between 0.955 and 0.999 and locality between 0.981 and 1.000 on streams of up to 1,301 edits. Mechanism analyses show that the learned addresses separate in-scope states from competing ones well beyond what raw hidden states provide, and that the same geometry transfers to an untuned model family. Controlled ablations further show that rollout refinement drives most of the efficacy improvement, while gate calibration drives most of the locality improvement. The remaining errors follow the same decomposition: missed paraphrases are address failures, while an open gate followed by a wrong generation is a write failure. Generalization remains between 0.217 and 0.472 because calibrated gates rarely open for rephrasings far from the training paraphrases, which is the clearest limitation of the current system. Our evaluation is limited to three English factual benchmarks, 7–8B models, and a single edited layer, with storage growing linearly in the number of edits.

## 6. REFERENCES

[1] Jun-Yu Ma, Zhen-Hua Ling, Ningyu Zhang, and Jia-Chen Gu, “Neighboring perturbations of knowledge editing on large language models,” in Proceedings of the 41st International Conference on Machine Learning. 2024, vol. 235 of Proceedings of Machine Learning Research, pp. 33839–33854, PMLR.

[2] Zenghao Duan, Wenbin Duan, Zhiyi Yin, Yinghan Shen, Shaoling Jing, Jie Zhang, et al., “Related knowledge perturbation matters: Rethinking multiple pieces of knowledge editing in same-subject,” in Proceedings of NAACL-HLT: Short Papers. 2025, pp. 363–373, Association for Computational Linguistics.

[3] Jianchen Wang, Zhouhong Gu, Xiaoxuan Zhu, Lin Zhang, Haoning Ye, Zhuozhi Xiong, et al., “The missing piece in model editing: A deep dive into the hidden damage brought by model editing,” in 2025 IEEE International Conference on Acoustics, Speech, and Signal Processing (ICASSP). 2025, pp. 1–5, IEEE.

[4] Roi Cohen, Eden Biran, Ori Yoran, Amir Globerson, and Mor Geva, “Evaluating the ripple effects of knowledge editing in language models,” Transactions of the Association for Computational Linguistics, vol. 12, pp. 283–298, 2024.

[5] Kevin Meng, David Bau, Alex Andonian, and Yonatan Belinkov, “Locating and editing factual associations in GPT,” in Advances in Neural Information Processing Systems, 2022, vol. 35.

[6] Kevin Meng, Arnab Sen Sharma, Alex Andonian, Yonatan Belinkov, and David Bau, “Mass-editing memory in a transformer,” in International Conference on Learning Representations, 2023.

[7] Xiaopeng Li, Shasha Li, Shezheng Song, Jing Yang, Jun Ma, and Jie Yu, “PMET: Precise model editing in a transformer,” in Proceedings of the AAAI Conference on Artificial Intelligence, 2024, vol. 38, pp. 18564–18572.

[8] Peter Hase, Mohit Bansal, Been Kim, and Asma Ghandeharioun, “Does localization inform editing? surprising differences in causality-based localization vs. knowledge editing in language models,” in Advances in Neural Information Processing Systems, 2023, vol. 36.

[9] Eric Mitchell, Charles Lin, Antoine Bosselut, Christopher D. Manning, and Chelsea Finn, “Memory-based model editing at scale,” in Proceedings ofthe 39th International Conference on Machine Learning. 2022, vol. 162 of Proceedings of Machine Learning Research, pp. 15817–15831, PMLR.

[10] Thomas Hartvigsen, Swami Sankaranarayanan, Hamid Palangi, Yoon Kim, and Marzyeh Ghassemi, “Aging with GRACE: Lifelong model editing with discrete key-value adaptors,” in Advances in Neural Information Processing Systems, 2023, vol. 36.

[11] Peng Wang, Zexi Li, Ningyu Zhang, Ziwen Xu, Yunzhi Yao, Yong Jiang, et al., “WISE: Rethinking the knowledge memory for lifelong model editing of large language models,” in Advances in Neural Information Processing Systems, 2024, vol. 37.

[12] Nicola De Cao, Wilker Aziz, and Ivan Titov, “Editing factual knowledge in language models,” in Proceedings of EMNLP. 2021, pp. 6491–6506, Association for Computational Linguistics.

[13] Eric Mitchell, Charles Lin, Antoine Bosselut, Chelsea Finn, and Christopher D. Manning, “Fast model editing at scale,” in International Conference on Learning Representations, 2022.

[14] Chenmien Tan, Ge Zhang, and Jie Fu, “Massive editing for large language models via meta learning,” in International Conference on Learning Representations, 2024.

[15] Junfeng Fang, Houcheng Jiang, Kun Wang, Yunshan Ma, Jie Shi, Xiang Wang, et al., “AlphaEdit: Null-space constrained knowledge editing for language models,” in International Conference on Learning Representations, 2025.

[16] Haoyu Xu, Pengxiang Lan, Enneng Yang, Guibing Guo, Jianzhe Zhao, Linying Jiang, et al., “Knowledge decoupling via orthogonal projection for lifelong editing of large language models,” in Proceedings ofACL. 2025, pp. 13194–13213, Association for Computational Linguistics.

[17] Ce Zheng, Lei Li, Qingxiu Dong, Yuxuan Fan, Zhiyong Wu, Jingjing Xu, and Baobao Chang, “Can we edit factual knowledge by in-context learning?,” in Proceedings of EMNLP. 2023, pp. 4862–4876, Association for Computational Linguistics.

[18] Jason Hoelscher-Obermaier, Julia Persson, Esben Kran, Ioannis Konstas, and Fazl Barez, “Detecting edit failures in large language models: An improved specificity benchmark,” in Findings of ACL. 2023, pp. 11548–11559, Association for Computational Linguistics.

[19] Wenyue Hua, Jiang Guo, Mingwen Dong, Henghui Zhu, Patrick Ng, and Zhiguo Wang, “Propagation and pitfalls: Reasoning-based assessment of knowledge editing through counterfactual tasks,” in Findings of ACL. 2024, pp. 12503– 12525, Association for Computational Linguistics.

[20] Omer Levy, Minjoon Seo, Eunsol Choi, and Luke Zettlemoyer, “Zero-shot relation extraction via reading comprehension,” in Proceedings ofthe 21st Conference on Computational Natural Language Learning. 2017, pp. 333–342, Association for Computational Linguistics.

[21] Ningyu Zhang, Yunzhi Yao, Bozhong Tian, Peng Wang, Shumin Deng, Mengru Wang, et al., “A comprehensive study of knowledge editing for large language models,” arXiv preprint arXiv:2401.01286, 2024.

[22] Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, et al., “Llama 2: Open foundation and fine-tuned chat models,” arXiv preprint arXiv:2307.09288, 2023.

[23] Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, et al., “The Llama 3 herd of models,” arXiv preprint arXiv:2407.21783, 2024.

[24] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, et al., “Qwen3 technical report,” arXiv preprint arXiv:2505.09388, 2025.

[25] An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, et al., “Qwen2.5 technical report,” arXiv preprint arXiv:2412.15115, 2024.

[26] Peng Wang, Ningyu Zhang, Bozhong Tian, Zekun Xi, Yunzhi Yao, Ziwen Xu, et al., “EasyEdit: An easy-to-use knowledge editing framework for large language models,” in Proceedings ofACL: System Demonstrations. 2024, pp. 82–93, Association for Computational Linguistics.