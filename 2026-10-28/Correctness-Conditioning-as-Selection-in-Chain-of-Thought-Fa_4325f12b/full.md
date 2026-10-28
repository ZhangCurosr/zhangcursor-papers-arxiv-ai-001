# Correctness Conditioning as Selection in Chain-of-Thought Faithfulness Evaluation

Yueer Pan<sup>1</sup> Yuexi Pan<sup>2</sup>

<sup>1</sup>Zhejiang University

<sup>2</sup>University of Science and Technology of China panjoel@zju.edu.cn panyuexi@mail.ustc.edu.cn

Webpage

GitHub

![](images/ec0da3e3e0bd0763e593437381405621e93501e19671ca447f2ef9845390ae2d.jpg)

![](images/9428354a18f8c33f1af6bbd3fcc4855d7adbbd7b1c9dfd30cde11c41b5850bb0.jpg)

![](images/bf8fa4f91440a078762b2dea0ad58d3526e03b5584377a1710d54908c9a223d6.jpg)  
Figure 1: CoTsel overview. Conceptual task intervention and correctness selection (top), with measured faithfulness and selection gaps on arithmetic DAGs (bottom).

## ABSTRACT

Restricting chain-of-thought (CoT) faithfulness evaluation to correctly answered instances changes the population being measured. We study this selection step in a seven-variable structural causal model and derive the exact observed gap ∆ = Cov $( A , C ) / P ( \bar { C } { = } 1 )$ The model distinguishes difficulty confounding from shortcut-induced collider bias and states the conditional-independence assumption needed to express the gap through latent response propensities. A protocol review of 18 papers distinguishes answer-correctness restrictions from agreement-based selection, benchmark construction, and training-data filtering. We evaluate four sub-ceiling deployments on arithmetic tasks and observe positive selection gaps in all four. The primary GLM-5.3 run yields a +7.4 percentage-point gap $( n = 1 0 0 0 , p < 0 . 0 0 0 1 )$ , and pooled Cochran–Mantel–Haenszel analysis gives $\mathrm { O R } = 1 . 7 0 ( p < 1 0 ^ { - 6 } )$ . An assumption-free covariance decomposition assigns 42.8% of the GLM-5.3 gap to between-stratum variation in interventionable-operation count, leaving the majority unattributed. When counterfactual compliance is observable for every instance, evaluating all traces directly estimates population faithfulness without correctness filtering.

## 1 INTRODUCTION

Chain-of-thought (CoT) faithfulness evaluation asks whether a model’s reasoning trace reflects the computation producing its final answer [Lanham et al., 2023, Turpin et al., 2023]. One approach intervenes on a reasoning step and measures the response. Restricting the comparison to correctly answered instances changes the population being evaluated and, with it, the quantity being estimated. Correct-response analyses occur in published protocols, including Putnam shortcut detection [Arcuschin et al., 2025] and MMLU step-importance analysis [Bogdan et al., 2025]. These protocols measure different properties; our question is how correctness selection changes an intervention-based compliance estimate.

Figure 1 summarizes the evaluation protocol and the choice of reporting population. Checks and crosses indicate factual answer correctness; counterfactual compliance is measured separately. The task cards illustrate the protocol families, and the plots report measured results on arithmetic DAGs.

We formalize correctness conditioning in a seven-variable structural causal model (SCM, Figure 3). The identity $\Delta = \mathrm { C o v } ( A , C ) / P ( C { = } 1 )$ ) gives the observed selection gap. Under an explicit conditional-independence premise, it factors as $\Delta = \mathrm { C o v } ( \eta , \pi ) / P ( C { = } 1 )$ ), relating the gap to correctness and compliance propensities. The framework links protocol and estimand, guiding our audit, deployment tests, and adjustment for an observed difficulty proxy.

The SCM separates two pathways. Difficulty affects both correctness and compliance $( D \to C , D \to A )$ , while conditioning on correctness opens the collider $R \to C \gets S$ when latent shortcut competence is present. Our main experiments measure the association with operation count; the neutral-framing detector records no shortcut use. We derive a ranking-reversal condition and, under the perfect-faithful-reasoning special case, an accuracy-axis peak at $P ^ { * } = \sqrt { p _ { s } }$ . Separate exploratory probes elicit the required shortcut reliance (Appendix M), but do not test the peak or reversal thresholds.

Our main contributions are threefold:

• We develop a causal framework for correctness conditioning in CoT faithfulness evaluation. We derive an exact expression for the observed selection gap and specify the assumptions required to analyse difficulty confounding and shortcut-induced collider bias.

• We compare reporting populations in 18 papers, separating answer-correctness restrictions from agreementbased selection, benchmark construction, and training-data filtering. The review identifies which protocol choices correspond to the selection operator in our analysis.

• We evaluate four model deployments on arithmetic tasks and measure a 7.4 percentage-point selection gap on the primary GLM-5.3 run. We decompose the gap by operation count and test its sensitivity to reweighting and the handling of unparseable responses.

We analyse the explicit-reliance probes independently of the main experiments; they use separate endpoints (Appendix M). For evaluation, the framework gives a direct rule: use all traces whenever counterfactual compliance is observable for every instance (Section 3).

## 2 RELATED WORK

Intervention-based CoT faithfulness. Chain-of-thought prompting [Wei et al., 2022] provides an explicit reasoning trace for large language models. Trace evaluations use truncation, paraphrase, and mistake insertion [Lanham et al., 2023]. Injected cues can bias CoT while the reasoning remains plausible [Turpin et al., 2023]. An analysis of metric degeneracy examines how answer accuracy can confound unfaithfulness measures in one setting [Bentham et al., 2024]. More recent work studies faithfulness through information flow [Jia et al., 2026], counterfactual simulation [Hase and Potts, 2026], step-level importance [Bogdan et al., 2025], unlearning [Tutek et al., 2025], and benchmarking [Shen et al., 2026].

Two 2026 studies identify additional sources of measurement error. One measures alignment between latent answer commitment and the explicit trace (61.9% average alignment) [Li et al., 2026]. This analysis concerns trace timing, whereas our estimand depends on which traces enter the aggregate. The other shows that classifier choice alone reverses rankings (regex 74.4% vs. LLM judge 69.7% on identical data) [Young, 2026]. This changes measurement on a fixed sample; correctness filtering changes the sample itself. Accurate step-level assessment and a fixed classifier therefore do not resolve the selection problem. Conversely, including every trace does not resolve classifier sensitivity. We derive the bias introduced by correctness conditioning in closed form.

Causal approaches also examine the reasoning process and its inputs. Controlled perturbations serve as instrumental signals for assessing step-level faithfulness within a single chain (intra-chain dependence) [Sun et al., 2026], while causal models of CoT steps address correctness and interpretability [Fu et al., 2025]. Studies of vision-language models examine how text and image biases affect faithfulness articulation [Balasubramanian et al., 2025]. Confidence based trace filtering has been proposed as an alternative evaluation strategy [Pathak et al., 2026]. Our analysis addresses the evaluation sample: conditioning it on answer correctness changes the aggregate faithfulness measure.

Selection bias in evaluation. Studies of LLM evaluation address position effects [Shi et al., 2025] and metric disagreement. Collider bias has been identified in LLM training data curation [McMilin, 2022]; correctness filtering introduces a related selection step at evaluation time. Other studies assess the causal validity of faithfulness metrics [Zaman and Srivastava, 2025] and their accuracy against ground truth [Gur-Arieh et al., 2026]. We complement these analyses by characterizing selection in the sample on which the metric is computed.

Our analysis connects to sample-selection models [Heckman, 1979] and, when the shortcut pathway is active, collider bias [Elwert and Winship, 2014]. Correctness filtering also raises a recoverability question when compliance on excluded instances is unavailable [Rubin, 1976, Mohan and Pearl, 2021]. Related settings include selective classification with abstention [El-Yaniv and Wiener, 2010, Geifman and El-Yaniv, 2017], the selective-labels problem [Lakkaraju et al., 2017], and graphical recoverability criteria [Bareinboim and Pearl, 2012, Bareinboim et al., 2014]. These methods model selection explicitly. We use the same perspective to diagnose selection in a correctness-conditioned CoT evaluation. The distinction between faithfulness and self-consistency [Parcalabescu and Frank, 2024] and tests of explanation faithfulness [Atanasova et al., 2023] clarify metric semantics, complementing our sample-selection analysis.

Causal inference and shortcut learning. Our SCM follows Pearl’s structural approach [Pearl, 2009], and the intervention analysis builds on causal-mediation methods for neural models [Vig et al., 2020]. The collider $R $ $C \gets S$ is an instance of Berkson’s paradox [Berkson, 1946]. Conditioning on post-treatment variables can bias observational studies [Rosenbaum, 2002], and our variable S maps to the taxonomy of shortcut learning [Geirhos et al., 2020]. Evidence that LLMs use hidden cues [Chen et al., 2025] and that training against monitored reasoning can promote obfuscation [Baker et al., 2025] motivates including shortcut competence in the SCM.

## 3 A DIAGNOSTIC FRAMEWORK FOR SELECTION BIAS

We model correctness conditioning as selection in a structural causal model (SCM). We first define the evaluation variables and derive the selection gap, then analyse difficulty confounding and shortcut-induced collider bias under their respective assumptions.

Figure 2 illustrates why factual correctness alone does not establish counterfactual compliance. Both rules answer the original task correctly. After the edit, recomputing the dependency returns the new oracle answer, while copying the unchanged cue fails. The rules illustrate the evaluation protocol; they are not recorded model responses or latent types identified in the deployment data.

## 3.1 STRUCTURAL CAUSAL MODEL

Definition 1 (Faithfulness Evaluation SCM). Let $( \mathbf { U } , \mathbf { V } , { \mathcal { F } } )$ be a stochastic SCM with endogenous variables $\mathbf { V } = \{ D , S , R , T , C , I , A \}$ and mutually independent exogenous noise ${ \bf U } = \{ U _ { D } , U _ { S } , U _ { R } , U _ { T } , \overleftarrow { U } _ { C } , U _ { I } , U _ { A } \}$ . The Bernoulli kernels below are shorthandfor structuralfunctions ofthe corresponding independent noise terms:

$$
D = f _ { D } ( U _ { D } ) ,
$$

$$
S \sim \operatorname { B e r n o u l l i } ( p _ { S } ( D ) ) ,\tag{1}
$$

(2)

$$
R \sim \operatorname { B e r n o u l l i } ( p _ { R } ( D ) ) ,\tag{3}
$$

$$
T \sim \mathrm { B e r n o u l l i } ( p _ { T } ( R , D ) ) ,\tag{4}
$$

$$
C \sim \operatorname { B e r n o u l l i } ( \pi ( R , S , T , D ) ) ,\tag{5}
$$

$$
I \sim \mathrm { B e r n o u l l i } ( \rho ) ,\tag{6}
$$

$$
A \sim \operatorname { B e r n o u l l i } ( \eta ( R , S , T , D , I ) ) ,\tag{7}
$$

where D is task difficulty, S is shortcut competence (whether the model has a non-chain-of-thought route to the answer), R is chain-of-thought (CoT) reliance (whether the model uses the visible reasoning trace), T is trace correctness, C is answer correctness (the collider), I is randomized intervention assignment, and A is appropriate post-intervention answer change (the compliance outcome).

![](images/3f99453d6c0701e5b7a48948240cd41ddac82e32bfef9ed988e8b14a83f1c061.jpg)  
Figure 2: Correct factual answers can conceal different intervention responses. The two rules agree before the edit and diverge afterward.

The Bernoulli kernels represent binary outcomes. Filtering on $C = 1$ retains traces from correctly answered tasks; A records whether the answer changes appropriately after an intervention on the reasoning chain. Two structures can therefore produce a gap between the selected-trace and population estimates:

$$
\underbrace { R  C  S } _  \mathrm { c o l l i d e r ( n e e d s \} S > 0 ) } \qquad \mathrm { a n d } \qquad \underbrace { D  C , D  A } _ { \mathrm { d i f f i c u l t y c o n f o u n d i n g } } .\tag{8}
$$

Section 3.3 relates the confounding structure to the measured covariate. Section 3.4 derives the collider pathway under an active shortcut branch; the neutral-framing runs provide no detected shortcut use with which to test it (note $T \perp S \mid R , D )$

In Figure 3, orange edges identify the collider $R \to C \gets S$ , teal edges identify the difficulty paths $D  C$ and $\tilde { D }  A$ , and gray edges show the remaining structural dependencies. The double border marks conditioning on $C = 1 ;$ the dashed red line denotes an association that conditioning can open between reliance and shortcut competence within a fixed difficulty stratum. It is not a causal edge.

## 3.2 THE GENERAL BIAS FORMULA

Within a fixed difficulty stratum $D = d ,$ define the T-marginalized cell probabilities:

$$
\begin{array} { r l } & { \pi _ { r s } : = P ( C = 1 \mid R = r , S = s , D = d ) , } \\ & { \eta _ { r s } : = P ( A = 1 \mid R = r , S = s , I = 1 , D = d ) . } \end{array}\tag{9}
$$

(10)

Assume $R \perp S \mid D = d$ with $P ( R = 1 \mid D = d ) = p _ { r }$ and $P ( S = 1 \mid D = d ) = p _ { s }$ . In our SCM, $R = f _ { R } ( D , U _ { R } )$ and $S = f _ { S } ( D , \dot { U } _ { S } )$ where $U _ { R }$ and $U _ { S }$ are independent exogenous variables. The Markov property gives $R \perp S \mid D \colon$ reasoning ability and shortcut competence share only task difficulty as a common cause.

Definition 2 (Population and Selected-Trace Faithfulness). For causal interventions $( I = 1 )$ , within a fixed difficulty stratum $D = d .$

$$
F _ { \mathrm { p o p } } ^ { ( d ) } : = P ( A = 1 \mid I = 1 , D = d ) ,\tag{11}
$$

$$
F _ { \mathrm { s e l } } ^ { ( d ) } : = P ( A = 1 \mid I = 1 , C = 1 , D = d ) ,\tag{12}
$$

$$
\Delta ^ { ( d ) } : = F _ { \mathrm { s e l } } ^ { ( d ) } - F _ { \mathrm { p o p } } ^ { ( d ) } .\tag{13}
$$

The marginal analogues reported in Table 1 drop the stratification, with $F _ { \mathrm { p o p } } , F _ { \mathrm { s e l } } ,$ , and $\Delta : = F _ { \mathrm { s e l } } - F _ { \mathrm { p o p } }$ defined by the same expressions without conditioning on D.

![](images/5926a8a9195bc4d5e02539ac29003dbb23894df182fd953c6677b35de1310a07.jpg)  
Figure 3: Structural causal model before and after correctness conditioning. Both panels show all dependencies in Definition 1; the dashed line marks a selection-induced association.

Proposition 1 (General Bias Formula). Assume $A \perp C \mid \{ R , S , I , D \}$ (compliance is conditionally independent of correctness given the strategy type and intervention). Then the selection bias equals the covariance ofinterventionresponse propensity and correctness propensity, normalized by the selection probability:

$$
\Delta = \frac { \mathrm { C o v } ( \eta _ { R S } , \pi _ { R S } \mid D = d ) } { P ( C = 1 \mid D = d ) } ,\tag{14}
$$

where η<sub>RS</sub> and $\pi _ { R S }$ are random cell-level functions of (R, S) under the population distribution.

Proof. Expanding $F _ { \mathrm { s e l } }$ using Bayes’ rule:

$$
F _ { \mathrm { s e l } } = \sum _ { r , s } \eta _ { r s } \cdot \frac { \pi _ { r s } P ( R = r ) P ( S = s ) } { P ( C = 1 ) } = \frac { E [ \eta _ { R S } \pi _ { R S } ] } { P ( C = 1 ) } .\tag{15}
$$

Since $F _ { \mathrm { p o p } } = E [ \eta _ { R S } ]$ and $P ( C { = } 1 ) = E [ \pi _ { R S } ] ;$

$$
\Delta = \frac { E [ \eta _ { R S } \pi _ { R S } ] } { E [ \pi _ { R S } ] } - E [ \eta _ { R S } ] = \frac { E [ \eta _ { R S } \pi _ { R S } ] - E [ \eta _ { R S } ] E [ \pi _ { R S } ] } { E [ \pi _ { R S } ] } = \frac { \mathrm { C o v } ( \eta _ { R S } , \pi _ { R S } ) } { P ( C = 1 ) } .\tag{16}
$$

The observed selection gap has an assumption-free expression. With $F _ { \mathrm { s e l } } = E [ A C ] / E [ C ]$ and $F _ { \mathrm { p o p } } = E [ A ]$

$$
\Delta = { \frac { E [ A C ] } { E [ C ] } } - E [ A ] = { \frac { E [ A C ] - E [ A ] E [ C ] } { E [ C ] } } = { \frac { \mathrm { C o v } ( A , C ) } { P ( C = 1 ) } } ,\tag{17}
$$

using $P ( C { = } 1 ) = E [ C ]$ . Equation 17 is an algebraic identity on the observed pair $( A , C )$ , with no latent-variable or independence assumption. This is the gap estimated in our experiments. The factorization in equation 14 requires

the additional premise $A \perp C \mid \{ R , S , I , D \}$ : trace correctness $T$ enters both equation $^ { 5 }$ and equation 7, leaving $C \left. T \right. A$ open after conditioning on $\{ R , S , I , D \}$ . The premise therefore constrains the SCM rather than following from it. With this constraint, $\mathrm { \bar { C } o v } ( \mathrm { \bar { \eta } } _ { R S } , \pi _ { R S } ) = \mathrm { C o v } ( \mathrm { \bar { A } } , C ) :$ without it, the two covariances can differ.

Stratifying on an observed variable gives a separate, assumption-free decomposition. With $D$ the measured difficulty proxy, the law of total covariance gives

$$
\operatorname { C o v } ( A , C ) = \underbrace { \operatorname { C o v } ( E [ A \mid D ] , E [ C \mid D ] ) } _ { \mathrm { b e t w e e n - s t r a t u m } } + \underbrace { E [ \operatorname { C o v } ( A , C \mid D ) ] } _ { \mathrm { w i t h i n - s t r a t u m } } ,\tag{18}
$$

which holds for any joint distribution of $( A , C , D )$ . We use equation 18 in Section 6 to measure variation between strata of the observed proxy. The within-stratum term collects residual covariance without a mechanistic attribution.

Correctness filtering is unbiased when Cov $( A , C ) = 0$ . This includes the boundary $P ( C { = } 1 ) = 1$ and, under the conditional-independence premise, the case in which $\eta _ { R S }$ is constant across cells. Equation equation 14 expresses the gap through cell-level propensities under that premise, while equation 17 describes the observed gap directly. The theory identifies the stratum-specific $\Delta ^ { ( d ) }$ ; the marginal $\Delta$ of Table 1 additionally includes the between-stratum composition term (Appendix D).

## 3.3 DIFFICULTY CONFOUNDING

When task difficulty D reduces both correctness $( D \to C )$ and compliance $( D  A )$ , conditioning on $C = 1$ selects easier tasks with higher compliance. We measure the associated stratified variation using operation count as a difficulty proxy (Section 6). Between-stratum variation accounts for 21.9–42.8% of the observed gap. This is a descriptive share; identifying the causal contribution of latent difficulty requires varying difficulty while holding other causes of correctness fixed.

Proposition 2 (Difficulty-Confounding Bias). Let $\pi ( d ) = P ( C = 1 \mid D = d )$ and $\eta ( d ) = P ( A = 1 \mid I = 1 , D = d )$ Assume $A \perp C \mid \{ D , I \}$ . When both π and η decrease with difficulty (harder tasks have lower accuracy and lower compliance), conditioning on $C = 1$ selects easier instances and inflates thefaithfulness estimate:

$$
\Delta = \frac { \mathrm { C o v } ( \eta ( D ) , \pi ( D ) ) } { P ( C = 1 ) } ,\tag{19}
$$

the same covarianceform as Proposition 1 with $( R , S )$ replaced by the observed variable D. Whenever harder tasks produce both lower accuracy and lower compliance $( \mathrm { \tilde { C o v } } ( \eta ( D ) , \tilde { \pi } ( D ) ) > 0 )$ , the bias is strictly positive.

Proposition 2 holds under the additional assumption $A \perp C \mid \{ D , I \}$ , satisfied when compliance depends only on task difficulty and intervention type. This premise excludes residual dependence beyond $\bar { D }$ and must be imposed in addition to equation 5–equation 7. With a measured difficulty proxy, we estimate within-stratum correctness and compliance for the adjustment in Section 6. All-trace evaluation removes correctness selection for either pathway.

## 3.4 SHORTCUT COMPETENCE AND COLLIDER BIAS

With latent shortcut competence $S > 0 ,$ conditioning on correctness opens the collider $R \to C \gets S$ . Under the perfect-faithful-reasoning special case (Corollary 1), the gap decreases monotonically in shortcut prevalence (Corollary 2) and peaks at $p _ { r } ^ { * } = \sqrt { p _ { s } } / ( 1 + \sqrt { p _ { s } } )$ in CoT reliance (Corollary 3). Proposition 4 gives the conditions for reversing two models’ faithfulness rankings. These predictions concern an active shortcut branch. The neutralframing detector records 0% use (§6); separate explicit-reliance probes elicit shortcut use on several deployments (Appendix M), but do not test the peak or reversal predictions. Appendix F gives the derivations and proofs.

## 3.5 IMPLICATIONS FOR EVALUATION PRACTICE

Evaluators should report the population on which a faithfulness score is computed. Correctness selection leaves the score unchanged when compliance and correctness are uncorrelated, including perfect accuracy or, under the stated premise, constant compliance. Under the special case in Corollary 1, the gap peaks at intermediate CoT reliance (Corollary 3).

When compliance can be measured for every instance, including incorrect answers, evaluating all traces computes $F _ { \mathrm { p o p } }$ directly. Most natural-language settings lack a counterfactual oracle. Evaluators should then state the correctness condition and measurement limits, reporting $F _ { \mathrm { s e l } }$ and $F _ { \mathrm { p o p } }$ when both are estimable.

## 4 REPORTING POPULATIONS IN PRIOR EVALUATIONS

We inspect 18 papers on CoT faithfulness and related evaluation protocols. The set spans trace interventions, input perturbations, benchmark construction, and training procedures. We record the population used by each inspected analysis rather than assign a single filtering label to an entire paper. Table 4 gives version-specific source locations; Figure 4 summarizes the distinctions.

Review procedure. We inspect metric definitions, evaluation sections, and relevant appendices for the 18 papers in Table 4. For each protocol, we distinguish selection of training examples, construction of a benchmark, and restriction of the final evaluation denominator. This targeted review compares concrete protocol choices; a prevalence estimate would require a systematic sampling frame and a common unit of analysis.

Answer correctness. Arcuschin et al. restrict their Putnam shortcut analysis to correct responses, while their separate comparison-pair analysis uses all retained input pairs [Arcuschin et al., 2025]. Bogdan et al. use correct CoTs for MMLU step-importance analysis but outcome-balanced traces for their MATH analysis [Bogdan et al., 2025]. Each analysis has its own reporting population. A correct-only score estimates a conditional quantity; our analysis quantifies its difference from the population score.

Other selection rules. Tutek et al. restrict their faithfulness comparison to instances on which CoT and no-CoT predictions agree [Tutek et al., 2025]. Agreement does not require either answer to be correct. Chen et al. select pairs where adding a hint changes the answer to that hint [Chen et al., 2025]; correctness then depends on the hint’s validity. Bentham et al. average with/without-CoT agreement over the dataset [Bentham et al., 2024]. Their accuracy-confound analysis does not establish a correctness-filtered denominator.

![](images/4fcda9bf1297826d77e99f3193bda3c14fa743e08c148fa7ea032ba6e3cc400f.jpg)  
Figure 4: Selection rules in evaluation protocols. Answer correctness, behavioral agreement, and data construction define different populations.

Construction and training. C<sup>2</sup>-Faith perturbs chains whose steps all have positive human labels [Mittal and Arike, 2026]; this process-quality criterion is distinct from filtering an evaluated model’s final answers. Bona Fide constructs ground truth from correct outright answers or responses that follow a wrong hint [Gur-Arieh et al., 2026]. Correct-answer filtering also appears in the training pool of Knowledge-Driven CoT [Wang et al., 2023]. Comparing these protocols requires specifying when selection occurs and which population the score describes. Appendix H.1 details these distinctions.

## 5 EXPERIMENTAL DESIGN

We compare the correctness-filtered estimator $F _ { \mathrm { s e l } }$ with the all-trace estimator $F _ { \mathrm { p o p } }$ on procedural tasks with known causal structure. Deterministic counterfactual answers make compliance observable for every task, and configurable cues allow separate tests of shortcut reliance.

## 5.1 PROCEDURAL TASK DESIGN

Intervention-based evaluation requires a reference answer for the edited task. We construct procedural tasks in which a symbolic executor replays the dependency graph and computes this counterfactual answer exactly [Lyu et al., 2023, Hase and Potts, 2026]. The resulting oracle supports the same compliance measurement on correctly and incorrectly answered instances. Natural-language tasks generally lack a unique, verifiable answer after an edit; extending the results to that setting requires a separate evaluation.

We implement three task families with deterministic oracles and semantics-preserving placebo edits: arithmetic DAGs of integer operations (depth 2–4), Boolean circuits of logical gates (depth 2–8), and graph reachability on 5–12 nodes. Each family exposes an editable dependency structure. The pilot generators produce 100 candidate instances per family from deterministic seeds. We use smaller batches for the implementation check (Table 3) and separate run sizes for the deployment study (Table 1). A faithful baseline (n = 50) validates the evaluation procedure, giving $F _ { \mathrm { p o p } } = F _ { \mathrm { s e l } } = 1 . 0 0 0$ (Appendix G).

Figure 5 gives a worked example from each family. Arithmetic and Boolean edits override an intermediate value before replaying downstream operations. Graph edits change an input edge and recompute reachability. These examples illustrate the generators; the headline deployment results use arithmetic DAGs.

![](images/4b64936468c7291001d934a5160f7043d8b1766a6ca0baf4c4448b312eb82cd8.jpg)  
Figure 5: Procedural task families. Each example pairs the original task with an intervention and a deterministic counterfactual answer.

Planted shortcut channels. In the procedural tasks, each computed step depends on earlier values. Without a planted cue, the design targets $S \approx 0$ , suppressing the shortcut pathway; $\Delta = 0$ additionally requires the absence of other correctness–compliance dependence. To study shortcut use, we augment instances with a cue whose validity is controlled by q, the probability that it agrees with the oracle answer. The cue is not updated under intervention, so a cue-reliant model can pass factual correctness while failing counterfactual compliance. The generator control q is distinct from the theoretical $p _ { s } = P ( S = 1 \mid D )$ , which denotes the prevalence of shortcut-capable behavior in the SCM and is set directly in the implementation check.

## 5.2 ALL-TRACE INTERVENTION PROTOCOL

For each instance $i ,$ the language model generates a trace and answer, and the symbolic executor supplies the reference answer. We present three conditions independently. The factual condition uses the original task and records correctness $C _ { i } = { \mathbb { \breve { H } } } [ \hat { y } _ { i } ^ { \mathrm { f a c t } } = y _ { i } ^ { * } ]$ . The causal intervention condition changes one intermediate graph value; the oracle replays the downstream computation to obtain $y _ { i } ^ { \mathrm { c f } * }$ , and we record compliance $A _ { i } = \mathbb { H } [ \hat { y } _ { i } ^ { \mathrm { c f } } = y _ { i } ^ { \mathrm { c f } * } ]$ . The placebo condition applies a semantics-preserving edit and records $A _ { i } ^ { \mathrm { p l a c } } = \mathcal { k } [ \hat { y } _ { i } ^ { \mathrm { p l a c } } = y _ { i } ^ { * } ]$

Each condition is a standalone task. In the headline analysis, we score non-interventionable traces with missing or unparseable answers as non-compliant $( A _ { i } = 0 )$ . We also repeat the analysis with these responses treated as missing (Appendix P). The headline convention gives a larger estimated gap.

Figure 6 illustrates the three conditions and their oracle-based scores. The example is schematic; the placebo condition validates the instance set and is not a per-model result.

![](images/77cf3b5354c626db6742d2b1806fd523ecb822742dc4014bfb16f30c4bc995ac.jpg)

Figure 6: All-trace intervention protocol. On this illustrative arithmetic task, a value intervention changes the oracle answer and a semantics-preserving edit leaves it unchanged.

## 5.3 EVALUATION METRICS

The primary estimands are:

$$
{ \cal F } _ { \mathrm { p o p } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } { \cal A } _ { i } , \qquad { \cal F } _ { \mathrm { s e l } } = \frac { \sum _ { i = 1 } ^ { n } { \cal A } _ { i } \cdot { \cal C } _ { i } } { \sum _ { i = 1 } ^ { n } { \cal C } _ { i } } , \qquad \Delta = { \cal F } _ { \mathrm { s e l } } - { \cal F } _ { \mathrm { p o p } } .\tag{20}
$$

We estimate confidence intervals with a nonparametric bootstrap $( B = 1 0 , 0 0 0$ resamples of instances, the unit at which the model was queried). The pre-specified primary analysis is a single pooled Cochran–Mantel–Haenszel test. Individual model p-values are descriptive; we apply no multiplicity adjustment to the single primary hypothesis. All headline model runs use temperature = 0 (greedy decoding).

## 6 DEPLOYMENT RESULTS

We evaluate four sub-ceiling deployments from ZhipuAI and DeepSeek on arithmetic DAGs, alongside two 100%- accuracy boundary controls from Alibaba and Google (Table 1). We first measure the selection gap, then decompose it by operation count and test its sensitivity to adjustment and response handling.

Correctness conditioning raises the point estimates. All four sub-ceiling runs have positive selection gaps. Figure 7 compares the all-trace and selected estimates, with gap intervals at the precision reported in Table 1. Two intervals cross zero.

Table 1 reports results for the serving deployments that answered our requests (Appendix R). We retain missing responses in the run sizes and score them as non-compliant, following Section 5.2. Thus $F _ { \mathrm { p o p } }$ uses the full run, while $F _ { \mathrm { s e l } }$ uses the correctness-selected subset. Table 7 gives the joint counts, and asterisks mark two-sided $p < 0 . 0 5$ . The ceiling controls satisfy $P ( C = 1 ) = 1$ by construction, so their zero gap is an algebraic identity. The Gemini control includes only valid responses (Appendix O).

The primary GLM-5.3 run gives a +7.4pp gap $( p < 0 . 0 0 0 1 , n = 1 0 0 0 ; \mathrm { F i g u r e 7 } )$ , and deepseek-v4-flash gives +6.0pp $( p = 0 . 0 4 6 , n = 3 0 0 ) .$ <sup>1</sup> The remaining two gaps are positive and non-significant. Pooling the four deployments with the Cochran–Mantel–Haenszel test gives $\mathbf { O R } { \stackrel { - } { = } } 1 . 7 0 \ ( \mathrm { A p p e n d i x ~ L } )$ . The pooled association survives removal of any single deployment. Excluding GLM-5.3, the largest contributor, reduces the pooled z from 5.19 to 2.24 $( p = 0 . 0 2 5 )$ ). Both 100%-accuracy controls return the required boundary value: $P ( C { = } 1 ) \dot { = } 1$ makes the gap exactly zero.

Excluding the only batch requiring deduplication, which contributes 100 of the 1000 rows, leaves $n = 9 0 0$ rows with no deduplication. The gap remains $+ 7 . { \dot { 2 } } 6 \rho \mathrm { p } \left( \chi ^ { 2 } = 2 0 . 5 4 , p < 0 . 0 0 0 1 \right)$ , so the result does not depend on that batch.

![](images/1c7ede55683381a22628ab9ceaadaaf1d4314fa9e683d85bed77b8ab6e6768c5.jpg)  
At 100% correctness, ceiling controls have a zero selection gap.  
Figure 7: Faithfulness and selection gaps across deployments. Open markers denote all traces; filled markers denote selected traces. Intervals show uncertainty in the selection gap.

Table 1: Faithfulness across serving deployments. All-trace and correctness-conditioned estimates, with ceiling controls shown separately.
<table><tr><td>Model</td><td>n</td><td>Acc.</td><td> $F _ { \mathbf { p o p } }$ </td><td> $\pmb { F } _ { \mathrm { s e l } }$ </td><td>∆ (pp) 95% CI (pp)</td><td></td><td>p</td></tr><tr><td>glm-5.2</td><td></td><td>10079.0%</td><td>0.440</td><td>0.456</td><td>+1.6</td><td>[−3.4, +6.6]</td><td>0.540</td></tr><tr><td>deepseek-v4-pro</td><td></td><td>20066.5%</td><td>0.555</td><td>0.579</td><td>+2.4</td><td>[-2.6, +7.2]</td><td>0.337</td></tr><tr><td>glm-5.3</td><td></td><td>100051.5%</td><td>0.507</td><td>0.581</td><td>+7.4</td><td>[+4.3, +10.4] &lt;0.0001*</td><td></td></tr><tr><td>deepseek-v4-flash</td><td></td><td>30048.3%</td><td>0.527</td><td>0.586</td><td>+6.0</td><td>[+0.2, +11.8]</td><td>0.046*</td></tr><tr><td>Boundary controls (P(C=1) = 1, filter vacuous):</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>qwen3.8-max</td><td>20</td><td>100%</td><td>0.600</td><td>0.600</td><td>0.0</td><td>N/A</td><td>N/A</td></tr><tr><td>Gemini 3.5 Flash Lite</td><td>17</td><td>100%</td><td>0.529</td><td>0.529</td><td>0.0</td><td>N/A</td><td>N/A</td></tr></table>

Operation count accounts for part of the gap. Conditioning on C = 1 selects instances with fewer interventionable operations, where compliance is higher. The covariance decomposition equation 18 assigns 42.8% (GLM-5.3) and 21.9% (flash) of the gap to variation between operation-count strata (Table 8). The GLM-5.3 estimate is more precise (n = 1000); flash’s smaller strata (n = 300) yield a bootstrap interval spanning zero. The decomposition depends on the covariate: oracle-displacement magnitude accounts for 26.6% on the same GLM-5.3 records. The 42.8% share therefore measures an association with operation count and does not identify a causal effect of difficulty. Inverse-propensity weighting on that count removes 23–34% of the raw gap across two propensity specifications, leaving most of the gap after either adjustment.

The trace-matching detector records 0% shortcut use under neutral framing and 10% in GLM-5.3 under adversarial elicitation. After excluding 9 null rows, the complete-case cue-correctness contrast (n = 291) is −1.6pp (z = −0.24, p = 0.81), based on 112/208 correct-cue rows and 46/83 incorrect-cue rows. We detect no difference in this contrast, and the detector cannot identify unacknowledged use of a correct cue (Appendix Q). Appendix M reports targeted explicit-reliance probes on separate deployments. The headline runs do not test the collider predictions of Section 3.4; their scope is specified in Appendix R.

Figure 8 shows the GLM-5.3 decomposition computed from the joint counts in Table 6. Of the 7.358pp total gap, 3.146pp lies between operation-count strata and 4.212pp lies within them. The within-stratum component remains unattributed. The between-stratum share has a descriptive bootstrap interval of [26.8, 69.1]% (Table 8); the plot shows the point estimate.

The gap persists after adjustment and response filtering. Reweighting reduces the selected estimate but leaves it above the direct all-trace estimate (Figure 9). Dropping null responses also reduces every deployment’s gap, with all

Null = missing  
(a) Selection within strata  
![](images/c78a1452efa41c837df1817f10247212d8701623ed63ab6f25089d75f900d1f1.jpg)

(b) Gap decomposition  
![](images/51a40d0360a7251cb398958095ba6c68ab0548e03106912144a6ae32465f29b6.jpg)  
GLM-5.3 · n = 1,000  
Descriptive, not causal attribution

Figure 8: Operation-count decomposition. GLM-5.3 faithfulness across strata and the corresponding covariance split. The split is descriptive; the within-stratum component remains unattributed.

four point estimates remaining positive. Appendix J.2 specifies the propensity models, and Appendix P gives the null-handling counts.

## (a) Reweighting selected traces

(b) Null-response convention

GLM-5.3 · operation-count propensities

![](images/6438bb4f01b1f8c905969309074c032a31036923887c643200b887722a961eef.jpg)  
Four sub-ceiling deployments

![](images/c2ee888801de9cc0bd86825bcaeae962a436a52293755c299e6e9feb951537f1.jpg)  
Point estimates; weighting and null handling are separate sensitivity checks.

Figure 9: Sensitivity to adjustment and response handling. Weighted selected estimates are compared with all traces; deployment gaps are compared under two null-response conventions. Markers show point estimates.

Table 1 reports the marginal $\Delta = F _ { \mathrm { s e l } } - F _ { \mathrm { p o p } }$ , which includes selected-weighted within-stratum gaps and a betweenstratum composition term. The theory identifies the stratum-specific $\Delta ^ { ( d ) }$ (Appendix D). For GLM-5.3, the marginal gap is +7.4pp, while selected-weighted difficulty standardization, $\begin{array} { r } { \Delta _ { \mathrm { s t d } , C } : = \sum _ { d } P ( D = d \mid C = 1 ) ( F _ { \mathrm { s e l } } ^ { ( d ) } - F _ { \mathrm { p o p } } ^ { ( d ) } ) . } \end{array}$ gives +4.2pp; the naive average gives +4.81pp. Using the unrounded joint counts, the between-stratum term of equation 18 is approximately $3 . 1 5 { \mathsf { p p } } .$ , separate from the within-stratum effect.

The shortcut-mixture special case does not fit these runs. The strict-reliance mixture in §3.4 overpredicts the measured gaps. Under its active-shortcut assumption, solving each deployment’s $F _ { \mathrm { p o p } }$ and $P ( C { = } 1 )$ for an implied p<sub>s</sub> under Corollary 1 gives +0.625, +0.247, +0.016 and −0.092 for glm-5.2, v4-pro, GLM-5.3 and flash. Flash’s negative value falls outside the probability range. Substitution gives gaps of 11.7–56.3pp against 1.6–7.4pp observed, a factor of 6.4–11.7 (Table 2). These discrepancies support estimating the observed $( A , C )$ gap and its operation-count decomposition directly, without imposing the mixture assumptions.

Table 2: Observed and predicted selection gaps. Predictions use the shortcut-mixture special case; the flash prediction lies outside the valid probability regime.
<table><tr><td>Model</td><td> $F _ { \mathbf { p o p } }$  (= pr)</td><td>Acc.  $( = P ( C { = } 1 ) )$ </td><td>Implied  $\pmb { p } _ { s }$ </td><td>Predicted Observed Ratio ∆(pp)</td><td>∆(pp)</td><td></td></tr><tr><td>glm-5.2</td><td>0.440</td><td>79.0%</td><td>+0.625</td><td>11.7</td><td>1.6</td><td>7.3×</td></tr><tr><td>deepseek-v4-pro</td><td>0.555</td><td>66.5%</td><td>+0.247</td><td>28.0</td><td>2.4</td><td>11.7×</td></tr><tr><td>GLM-5.3</td><td>0.507</td><td>51.5%</td><td>+0.016</td><td>47.7</td><td>7.4</td><td>6.4×</td></tr><tr><td>deepseek-v4-flash</td><td>0.527</td><td>48.3%</td><td>-0.092</td><td>56.3</td><td>6.0</td><td>9.4×</td></tr></table>

Table 2 reports each deployment’s $F _ { \mathrm { p o p } }$ and $P ( C = 1 )$ , the implied p under Corollary 1, and the corresponding prediction. We retain the flash row to document the incompatibility: its negative implied shortcut prevalence makes that prediction invalid as a probability-model result.

## 7 CONCLUSION

We show that correctness conditioning changes the population used to estimate CoT faithfulness and quantify the resulting gap through $\Delta = \mathrm { C o v } ( \breve { A } , C ) / \breve { P } ( C { = } 1 )$ Our protocol review separates this restriction from other selection rules used in prior work. On arithmetic DAGs, the primary GLM-5.3 run has a +7.4pp gap $( n = 1 0 0 0$ $p < 0 . 0 0 0 1 ) $ ; operation-count variation accounts for 42.8%, and the remainder is unattributed. Two of the four sub-ceiling deployments have individually significant gaps. These results establish selection bias in the measured setting. No shortcut use was detected in the neutral-framing runs, so these runs do not support attributing the observed bias to the collider pathway. Faithfulness evaluations should state their reporting population and use all traces whenever counterfactual compliance is observable for every instance.

## REPRODUCIBILITY STATEMENT

The procedural task generators use deterministic seed streams, and the symbolic oracle computes exact outputs. The main text and appendices specify the simulation parameters, estimands, prompts, and scoring conventions. Headline model calls use temperature = 0 (greedy decoding). When a serving endpoint returned different responses at that setting, we retained the first completed response for the reported run. We conducted the headline experiments in August 2026 and the exploratory framing probe in September 2026. Each model–seed–instance is counted once.

## DATA AND CODE AVAILABILITY

The website source, project film, paper figures, PDF, and release checks are available at https://github.com/ CCS-CoTFE/CCS-CoTFE. The project website is https://ccs-cotfe.github. $\scriptstyle \mathrm { i } _ { \mathrm { } } \mathrm { o } / \mathrm { C C S - C o T F E } / .$ . The paper source bundle contains the LaTeX files and figure assets needed to compile the manuscript.

The release contains the aggregate results reported in the tables and figures. Raw API logs, model completions, provider request identifiers, intermediate exports, the working literature-audit ledger, and credential-bound generators are not included. Exact row-level reanalysis requires a separate anonymized data release.

## ETHICS STATEMENT

We study the statistical properties of evaluation methods without human subjects, private data, or model training. All language model API calls use publicly available commercial endpoints under standard terms of service. The procedural tasks are synthetic and contain no sensitive content.

## REFERENCES

Iván Arcuschin, Jett Janiak, Robert Krzyzanowski, Senthooran Rajamanoharan, Neel Nanda, and Arthur Conmy. Chain-of-thought reasoning in the wild is not always faithful. arXiv preprint arXiv:2503.08679, 2025.

Pepa Atanasova, Oana-Maria Camburu, Christina Lioma, Thomas Lukasiewicz, Jakob Grue Simonsen, and Isabelle Augenstein. Faithfulness tests for natural language explanations. In Proceedings ofthe 61st Annual Meeting ofthe Associationfor Computational Linguistics, pages 283–294, 2023.

Bowen Baker, Joost Huizinga, Leo Gao, Zehao Dou, Melody Y. Guan, Aleksander Madry, Wojciech Zaremba, Jakub Pachocki, and David Farhi. Monitoring reasoning models for misbehavior and the risks of promoting obfuscation. arXiv preprint arXiv:2503.11926, 2025.

Sriram Balasubramanian, Samyadeep Basu, and Soheil Feizi. A closer look at bias and chain-of-thought faithfulness of large (vision) language models. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 13406–13439, 2025.

Elias Bareinboim and Judea Pearl. Controlling selection bias in causal inference. In Proceedings of the Fifteenth International Conference on Artificial Intelligence and Statistics, volume 22 of Proceedings of Machine Learning Research, pages 100–108, 2012.

Elias Bareinboim, Jin Tian, and Judea Pearl. Recovering from selection bias in causal and statistical inference. In Proceedings ofthe Twenty-Eighth AAAI Conference on Artificial Intelligence, pages 2410–2416, 2014.

Oliver Bentham, Nathan Stringham, and Ana Marasovic. Chain-of-thought unfaithfulness as disguised accuracy.´ Transactions on Machine Learning Research, 2024.

Joseph Berkson. Limitations of the application of fourfold table analysis to hospital data. Biometrics Bulletin, 2(3): 47–53, 1946.

Paul C. Bogdan, Uzay Macar, Neel Nanda, and Arthur Conmy. Thought anchors: Which LLM reasoning steps matter? arXiv preprint arXiv:2506.19143, 2025.

Yanda Chen, Joe Benton, Ansh Radhakrishnan, Jonathan Uesato, Carson Denison, John Schulman, Arushi Somani, Peter Hase, Misha Wagner, Fabien Roger, Vlad Mikulik, Samuel R. Bowman, Jan Leike, Jared Kaplan, and Ethan Perez. Reasoning models don’t always say what they think. arXiv preprint arXiv:2505.05410, 2025.

Corinna Cortes, Mehryar Mohri, Michael Riley, and Afshin Rostamizadeh. Sample selection bias correction theory. In Algorithmic Learning Theory, 19th International Conference (ALT), volume 5254 of Lecture Notes in Computer Science, pages 38–53, 2008.

Ran El-Yaniv and Yair Wiener. On the foundations of noise-free selective classification. Journal ofMachine Learning Research, 11(53):1605–1641, 2010.

Felix Elwert and Christopher Winship. Endogenous selection bias: The problem of conditioning on a collider variable. Annual Review ofSociology, 40:31–53, 2014.

Jiarun Fu, Lizhong Ding, Hao Li, Pengqi Li, Qiuning Wei, and Xu Chen. Unveiling and causalizing CoT: A causal pespective. arXiv preprint arXiv:2502.18239, 2025.

Yonatan Geifman and Ran El-Yaniv. Selective classification for deep neural networks. In Advances in Neural Information Processing Systems, volume 30, pages 4878–4887, 2017.

Robert Geirhos, Jörn-Henrik Jacobsen, Claudio Michaelis, Richard Zemel, Wieland Brendel, Matthias Bethge, and Felix A. Wichmann. Shortcut learning in deep neural networks. Nature Machine Intelligence, 2(11):665–673, 2020.

Yoav Gur-Arieh, Ana Marasovic, and Mor Geva. Faithfulness metrics don’t measure faithfulness: A meta-evaluation´ with ground truth. arXiv preprint arXiv:2605.25052, 2026.

Peter Hase and Christopher Potts. Counterfactual simulation training for chain-of-thought faithfulness. In Conference on Language Modeling (COLM), 2026.

James J. Heckman. Sample selection bias as a specification error. Econometrica, 47(1):153–161, 1979.

D. G. Horvitz and D. J. Thompson. A generalization of sampling without replacement from a finite universe. Journal ofthe American Statistical Association, 47(260):663–685, 1952.

Jinghan Jia, Joe Benton, and Eric Easley. Faithfulness as information flow: Evaluating and training faithful chain-ofthought reasoning. arXiv preprint arXiv:2605.24286, 2026.

Himabindu Lakkaraju, Jon Kleinberg, Jure Leskovec, Jens Ludwig, and Sendhil Mullainathan. The selective labels problem: Evaluating algorithmic predictions in the presence of unobservables. In Proceedings ofthe 23rd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 275–284, 2017.

Tamera Lanham, Anna Chen, Ansh Radhakrishnan, Benoit Steiner, Carson Denison, Danny Hernandez, Dustin Li, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukoši˙ ut¯ e, Karina Nguyen, Newton Cheng, Nicholas˙ Joseph, Nicholas Schiefer, Oliver Rausch, Robin Larson, Sam McCandlish, Sandipan Kundu, Saurav Kadavath, Shannon Yang, Thomas Henighan, Timothy Maxwell, Timothy Telleen-Lawton, Tristan Hume, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Measuring faithfulness in chain-of-thought reasoning. arXiv preprint arXiv:2307.13702, 2023.

Wenkai Li, Fan Yang, Ananya Hazarika, Shaunak A. Mehta, and Koichi Onoue. When reasoning traces become performative: Step-level evidence that chain-of-thought is an imperfect oversight channel. arXiv preprint arXiv:2605.11746, 2026.

Qing Lyu, Shreya Havaldar, Adam Stein, Li Zhang, Delip Rao, Eric Wong, Marianna Apidianaki, and Chris Callison-Burch. Faithful chain-of-thought reasoning. In International Joint Conference on Natural Language Processing and Conference ofthe Asia-Pacific Chapter ofthe Associationfor Computational Linguistics (IJCNLP-AACL), pages 305–329, 2023.

Emily McMilin. Selection collider bias in large language models. In UAI 2022 Workshop on Causal Representation Learning (CRL), 2022. arXiv:2208.10063.

Avni Mittal and Rauno Arike. C<sup>2</sup>-Faith: Benchmarking LLM judges for causal and coverage faithfulness in chain-ofthought reasoning. In Proceedings of the Fifth Workshop on Generation, Evaluation and Metrics (GEM), pages 678–691, 2026.

Karthika Mohan and Judea Pearl. Graphical models for processing missing data. Journal ofthe American Statistical Association, 116(534):1023–1037, 2021.

Letitia Parcalabescu and Anette Frank. On measuring faithfulness or self-consistency of natural language explanations. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics, pages 6048–6089, 2024.

Manas Pathak, Xingyao Chen, Shuozhe Li, Amy Zhang, and Leqi Liu. Filtered reasoning score: Evaluating reasoning quality on a model’s most-confident traces. In Conference on Language Modeling (COLM), 2026.

Debjit Paul, Robert West, Antoine Bosselut, and Boi Faltings. Making reasoning matter: Measuring and improving faithfulness of chain-of-thought reasoning. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2024, pages 15012–15032, 2024.

Judea Pearl. Causality: Models, Reasoning, and Inference. Cambridge University Press, 2nd edition, 2009.

Ansh Radhakrishnan, Karina Nguyen, Anna Chen, Carol Chen, Carson Denison, Danny Hernandez, Esin Durmus, Evan Hubinger, Jackson Kernion, Kamile Lukoši˙ ut¯ e, Newton Cheng, Nicholas Joseph, Nicholas Schiefer, Oliver˙ Rausch, Sam McCandlish, Sheer El Showk, Tamera Lanham, Tim Maxwell, Venkatesa Chandrasekaran, Zac Hatfield-Dodds, Jared Kaplan, Jan Brauner, Samuel R. Bowman, and Ethan Perez. Question decomposition improves the faithfulness of model-generated reasoning. arXiv preprint arXiv:2307.11768, 2023.

Paul R. Rosenbaum. Observational Studies. Springer, 2nd edition, 2002.

Donald B. Rubin. Inference and missing data. Biometrika, 63(3):581–592, 1976.

Xu Shen, Song Wang, Zhen Tan, Laura Yao, Xinyu Zhao, Kaidi Xu, Xin Wang, and Tianlong Chen. FaithCoT-Bench: Benchmarking instance-level faithfulness of chain-of-thought reasoning. In International Conference on Learning Representations, 2026.

Lin Shi, Chiyu Ma, Wenhua Liang, Xingjian Diao, Weicheng Ma, and Soroush Vosoughi. Judging the judges: A systematic study of position bias in LLM-as-a-judge. In Proceedings ofthe 14th International Joint Conference on Natural Language Processing and the 4th Conference of the Asia-Pacific Chapter of the Association for Computational Linguistics, pages 292–314, 2025. doi: 10.18653/v1/2025.ijcnlp-long.18.

Yuxi Sun, Aoqi Zuo, Haotian Xie, Wei Gao, Mingming Gong, and Jing Ma. FACT-E: Causality-inspired evaluation for trustworthy chain-of-thought reasoning. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 20275–20293, 2026.

Sree Harsha Tanneru, Dan Ley, Chirag Agarwal, and Himabindu Lakkaraju. On the hardness of faithful chain-ofthought reasoning in large language models. arXiv preprint arXiv:2406.10625, 2024.

Miles Turpin, Julian Michael, Ethan Perez, and Samuel R. Bowman. Language models don’t always say what they think: Unfaithful explanations in chain-of-thought prompting. In Advances in Neural Information Processing Systems, volume 36, pages 74952–74965, 2023.

Martin Tutek, Fateme Hashemi Chaleshtori, Ana Marasovic, and Yonatan Belinkov. Measuring chain of thought´ faithfulness by unlearning reasoning steps. In Proceedings ofthe Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 9935–9960, 2025.

Jesse Vig, Sebastian Gehrmann, Yonatan Belinkov, Sharon Qian, Daniel Nevo, Yaron Singer, and Stuart Shieber. Investigating gender bias in language models using causal mediation analysis. In Advances in Neural Information Processing Systems, volume 33, pages 12388–12401, 2020.

Keheng Wang, Feiyu Duan, Sirui Wang, Peiguang Li, Yunsen Xian, Chuantao Yin, Wenge Rong, and Zhang Xiong. Knowledge-driven CoT: Exploring faithful reasoning in LLMs for knowledge-intensive question answering. arXiv preprint arXiv:2308.13259, 2023.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V. Le, and Denny Zhou. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837, 2022.

Richard J. Young. Measuring faithfulness depends on how you measure: Classifier sensitivity in LLM chain-of-thought evaluation. arXiv preprint arXiv:2603.20172, 2026.

Bianca Zadrozny. Learning and evaluating classifiers under sample selection bias. In Proceedings of the Twenty-First International Conference on Machine Learning (ICML), 2004. doi: 10.1145/1015330.1015425.

Kerem Zaman and Shashank Srivastava. A causal lens for evaluating faithfulness metrics. In Proceedings of the Conference on Empirical Methods in Natural Language Processing (EMNLP), pages 29425–29449, 2025.

## A EXPERIMENTAL CONFIGURATION

Task families and experimental phases use different sample sizes, which we report separately. The implementation check uses small deterministic batches across three synthetic task families. The real-model study uses arithmetic DAGs with a primary run of n = 1000 and additional sub-ceiling deployments of $n = 3 0 0 , n = 2 0 0$ , and n = 100, together with two 100%-accuracy boundary controls. Each model–seed–instance key is counted once before aggregation.

The arithmetic DAG generator uses a depth range of $[ 2 , 4 ]$ , a value range of [1, 20], and a planted-cue validity control of $q = 0 . 7$ . This control differs from the theoretical shortcut prevalence $p _ { s }$ used in the implementation check.

The prompt template for the real-model runs is:

Solve this step by step. Show your work.

{task prompt}

Your final answer (just the number):

The counterfactual condition replaces the intervened step line with the new value $( { \bf e . g . \mit S 3 \ = \ 1 2 ( \mit g i v e n ) } )$ and keeps the rest of the task unchanged. The empirical runs use no system-level objective or persona prompt. Personamodulated agents in simulation (Section B) are stochastic compliance models with a fixed random seed: for each instance, the agent answers correctly with probability $\pi _ { r s }$ and complies with intervention with probability $\eta _ { r s } ,$ drawn from the closed-form cell probabilities. These agents check the mathematics and implementation; they do not model the behavior of a real language model.

## B IMPLEMENTATION CONSISTENCY CHECK

We verify the numerical implementation by sampling agents from the derived cell probabilities. The simulator imposes the model assumptions, so this experiment checks the implementation and does not independently test the theory. Sweeping $( p _ { r } , p _ { s } ) \in [ 0 , 1 ] ^ { 2 }$ gives a mean absolute error of 0.006 against Corollary 1. The implementation reproduces the monotone decrease in $p _ { s }$ (Corollary 2) and the peak at $p _ { r } ^ { * } \approx 0 . 4 5 6$ (Figure 10, Appendix C).

The three task families differ in dependency structure and reliance level. Table 3 compares them at a common theoretical shortcut prevalence $p _ { s } = 0 . 7$ . The gaps agree to within 0.6pp (mean +8.2pp, range +7.9 to +8.5pp), while per-family reliance $p _ { r }$ ranges from 0.33 to 0.58. This checks the closed-form implementation across integer arithmetic, Boolean logic, and graph reachability. The headline real-model evidence uses arithmetic tasks.

Table 3: Implementation consistency. Three procedural task families evaluated under a common shortcut prevalence.
<table><tr><td>Family</td><td>n pr</td><td></td><td> $F _ { \mathbf { p o p } }$ </td><td> $\pmb { F } _ { \mathrm { s e l } }$ </td><td>Gap (pp)</td></tr><tr><td>Arithmetic DAG</td><td>30 0.33</td><td></td><td>0.333</td><td>0.417</td><td>+8.3</td></tr><tr><td>Boolean circuit</td><td>50 0.58</td><td></td><td>0.580</td><td>0.659</td><td>+7.9</td></tr><tr><td>Graph reachability</td><td>50 0.52</td><td></td><td>0.520</td><td>0.605</td><td>+8.5</td></tr><tr><td>Mean</td><td></td><td></td><td></td><td></td><td>+8.2</td></tr></table>

Under the correctness parameters specified in Appendix K, Model A $( p _ { r } = 0 . 4 5 , p _ { s } = 0 . 9 0 )$ has $F _ { \mathrm { p o p } } = 0 . 4 5 0$ but $F _ { \mathrm { s e l } } = 0 . 5 0 4$ , while Model $\mathcal { B } \left( p _ { r } \overset { - } { = } 0 . 7 0 , p _ { s } \overset { - } { = } \overset { - } { 0 } . 0 1 \right)$ has $F _ { \mathrm { p o p } } = 0 . 7 0 0$ but $F _ { \mathrm { s e l } } = 0 . 3 0 8 \colon$ conditioning reverses the ranking, with 40.6% of the 39,999 full-grid candidates reversing at $\pi _ { 1 0 } = 0 . 0 0 1$ . These conditions underlie the prospective analysis in Section 3.4. This example requires an active shortcut branch, which we did not detect in our real-model runs under neutral framing.

## C SIMULATION: CROSS-FAMILY DETAIL

Table 3 (Section B) reports the per-family simulation results. The theoretical shortcut prevalence is $p _ { s } = 0 . 7$ for every row; this is distinct from the planted-cue validity control q in Section 5.1. The per-family CoT-reliance level $p _ { r }$ differs and equals $F _ { \mathrm { p o p } }$ under strict reliance. Boolean circuit and graph tasks use $n = 5 0$ for richer structural variation; arithmetic DAGs use $n = 3 0$ due to simpler compositional structure. The comparison varies the dependency structure while retaining the intervention protocol and oracle design of Section 5.

## D ESTIMAND: WITHIN-STRATUM VS. MARGINAL

We distinguish stratum-specific and marginal estimands. The theory’s Propositions 1, 3 and 4 are stated at fixed difficulty $D = d$ and identify the stratum-specific $\Delta ^ { ( d ) }$ . Conditioning on d permits the use of $R \perp S \mid D$ and the T-marginalization. It also removes the difficulty-confounding pathway from $\Delta ^ { ( d ) }$ , since D no longer varies. Proposition 2 is marginal in D and applies to real models, where latent R and S cannot be conditioned on.

Table 1 reports the marginal $\Delta = F _ { \mathrm { s e l } } - F _ { \mathrm { p o p } }$ . Its within-stratum component averages the $\Delta ^ { ( d ) }$ under the difficulty distribution among selected instances; a separate composition term accounts for the change in that distribution caused by correctness conditioning. The marginal gap is therefore not identified by the within-stratum result alone. We interpret the real-model gaps as consistent with difficulty confounding, without applying the shortcut-mixture special case to them.

The within-stratum analysis (Appendix J) decomposes the marginal gap by operation count. Gaps persist within strata, beyond the between-stratum variation. This decomposition leaves the collider contribution unidentified.

Magnitude reconciliation. The shortcut-mixture special case is inconsistent with the observed deployments in two ways. First, setting $p _ { s } = 0$ to match the measured 0% shortcut-use rate reduces Corollary 1 to $\Delta \stackrel { \cdot } { = } 1 - P ( C { = } 1 )$ This expression is monotone, has no interior maximum, and predicts ∼48pp at GLM-5.3’s 52% accuracy, compared with +7.4pp observed. Second, the implementation-check parameter $p _ { s } = 0 . 7$ implies an accuracy floor of 0.70, above the measured 48–52% accuracy of these deployments.

Solving each model’s own $F _ { \mathrm { p o p } }$ and accuracy for an implied $p _ { s }$ gives +0.625, +0.247, +0.016, and −0.092 (glm-5.2, v4-pro, GLM-5.3, flash). The negative value is not a valid probability, and these inferred values do not establish the measured 0% shortcut use. Table 2 (Section 6) reports the inputs needed to reproduce the predicted/observed ratios; Section 6 discusses their disagreement.

We therefore use the mixture special cases for simulation and idealised design analysis, and analyse real-model gaps through the observed operation-count covariate.

The accuracy-axis peak occurs at $P ^ { * } = \sqrt { p _ { s } } ,$ , so with the implementation-check value $p _ { s } = 0 . 7$ it lies at 83.7%: the four sub-ceiling deployments $( 4 8 - 7 9 \% )$ lie below it, while the two ceiling controls $( 1 0 0 \% )$ lie above. No tested deployment is near the peak. The ceiling controls cannot localise it because $\bar { P ( C = 1 ) } = 1$ makes $\Delta = 0$ an algebraic identity. Probing a peak near 50% accuracy would require $p _ { s } \approx 0 . 2 5$ , which is a different theoretical regime from the one used in the implementation check. The accuracy band in Figure 14 is therefore descriptive, not a prediction.

![](images/227ff4e5a4fee19ac1dc844073899467c6708c6c52afbc597763d14fcc944f63.jpg)  
Figure 10: Selection bias surface. Closed-form predictions with simulated points overlaid on the marginal slices.

## E PROOF DETAILS

## E.1 COROLLARY 4: HUMP ON THE ACCURACY AXIS

With $r : = \sqrt { p _ { s } }$ and strict reliance, Corollary 3 locates the hump at $p _ { r } ^ { * } = r / ( 1 + r )$ in CoT-reliance space. Substituting into Corollary 1: $p _ { r } ^ { * } ( 1 - p _ { r } ^ { * } ) = r / ( 1 + r ) ^ { 2 }$ , and $p _ { r } ^ { * } + ( 1 - p _ { r } ^ { * } ) p _ { s } = r / ( 1 + r ) + r ^ { 2 } ( 1 + r ) ^ { - 1 } = r$ . Hence

$$
\Delta ( P ^ { * } ) = \frac { r ( 1 - p _ { s } ) } { ( 1 + r ) ^ { 2 } r } = \frac { 1 - r ^ { 2 } } { ( 1 + r ) ^ { 2 } } = \frac { 1 - r } { 1 + r } , \qquad P ( p _ { r } ^ { * } ) = p _ { r } ^ { * } + ( 1 - p _ { r } ^ { * } ) p _ { s } = r = \sqrt { p _ { s } } .\tag{21}
$$

Since P is affine and strictly increasing in $p _ { r }$ , the unique interior maximum in $p _ { r }$ maps to the unique interior maximum in P. Attainable accuracy spans $[ p _ { s } , 1 ]$ ; the peak $P ^ { * } = \sqrt { p _ { s } }$ lies in that interval for all $p _ { s } \in ( 0 , 1 )$ .

Proofs of Propositions 1 and 2 are given in the main text. The collider pathway derivations (Propositions 3 and 4, Corollaries 1–4) are in Appendix F. This appendix contains additional derivation details.

## F COLLIDER PATHWAY: FULL DERIVATIONS

This section derives the collider results summarized in Section 3.4 for a population with an active shortcut branch $( S > 0 )$ .

The strict-reliance special case sets $\eta _ { 0 0 } = \eta _ { 0 1 } = 0$ and $\eta _ { 1 0 } = \eta _ { 1 1 } = 1$ : the model’s answer changes under intervention if and only if it genuinely relies on the CoT. Then $F _ { \mathrm { p o p } } = p _ { r }$ . Define the stratum-specific correctness rates

$$
g _ { 1 } ( p _ { s } ) : = P ( C { = } 1 \mid R { = } 1 ) = ( 1 { - } p _ { s } ) \pi _ { 1 0 } + p _ { s } \pi _ { 1 1 } ,\tag{22}
$$

$$
g _ { 0 } ( p _ { s } ) : = P ( C { = } 1 \mid R { = } 0 ) = ( 1 { - } p _ { s } ) \pi _ { 0 0 } + p _ { s } \pi _ { 0 1 } .\tag{23}
$$

Proposition 3 (Strict-Reliance Gap). Under the strict-reliance assumption, $F _ { \mathrm { s e l } } = P ( R { = } 1 \mid C { = } 1 )$ ), and the gap is:

$$
\Delta = \frac { p _ { r } ( 1 - p _ { r } ) ( g _ { 1 } - g _ { 0 } ) } { p _ { r } g _ { 1 } + ( 1 - p _ { r } ) g _ { 0 } } .\tag{24}
$$

The sign of∆ is determined by $g _ { 1 } - g _ { 0 } .$ correctness conditioning overestimatesfaithfulness when CoT reliance makes correctness more $l i k e l y \left( g _ { 1 } > g _ { 0 } \right)$ , and underestimates it otherwise.

Proof. Under strict reliance, $F _ { \mathrm { s e l } } = P ( R { = } 1 \mid C { = } 1 ) = p _ { r } g _ { 1 } / [ p _ { r } g _ { 1 } + ( 1 { - } p _ { r } ) g _ { 0 } ]$ by Bayes’ rule. Subtracting $F _ { \mathrm { p o p } } = p _ { \ i }$ <sub>r</sub> and simplifying yields equation 24. 口

Corollary 1 (Simplified Gap Under Perfect Faithful Reasoning). When faithful reasoning is always correct $( \pi _ { 1 0 } =$ $\pi _ { 1 1 } = 1 ) _ { \mathrm { { } } }$ , shortcut-only correctness is certain $( \pi _ { 0 1 } = 1 )$ , and the neither-route cell is always wrong $( \pi _ { 0 0 } = 0 )$ , we have $g _ { 1 } = 1$ and $g _ { 0 } = p _ { s } , g i \nu i n g .$

$$
\Delta = \frac { p _ { r } ( 1 - p _ { r } ) ( 1 - p _ { s } ) } { p _ { r } + ( 1 - p _ { r } ) p _ { s } } .\tag{25}
$$

This simplified form is a theoretical mixture for an active shortcut branch. The planted-cue generator in Section 5.1 uses the separate validity control q and does not estimate $p _ { s }$ in the real-model runs. For multiple-choice tasks with k options, $\pi _ { 0 0 } = 1 / k > 0$ and the general formula equation 24 applies; the simplified form remains a close approximation when $1 / \dot { k } \ll p _ { s }$

Corollary 2 (Monotone Decrease in Shortcut Prevalence). Forfixed $p _ { r } \in ( 0 , 1 ) , \partial \Delta / \partial p _ { s } < 0 .$ the gap equation 25 is strictly decreasing in shortcut prevalence.

Proof. Write $\Delta ( p _ { s } ) = a b ( 1 - p _ { s } ) / ( a + b p _ { s } )$ with $a = p _ { r } > 0$ and $b = 1 { - } p _ { r } > 0$ . Differentiating: $\partial \Delta / \partial p _ { s } =$ $- a b ( a + b ) / ( a + b p _ { s } ) ^ { 2 } < 0$ □

Corollary 3 (Hump-Shaped Prevalence Response). Forfixed $p _ { s } \in ( 0 , 1 ) , \Delta ( p _ { r } )$ from equation 25 is zero at $p _ { r } = 0$ and $p _ { r } = 1$ , with a unique interior maximum a $p _ { r } ^ { * } = \sqrt { p _ { s } } / ( 1 + \sqrt { p _ { s } } )$

Proof. Setting $d \Delta / d p _ { r } = 0 ;$ the first-order condition yields $p _ { r } ^ { 2 } = p _ { s } ( 1 { - } p _ { r } ) ^ { 2 }$ , hence $p _ { r } / ( 1 - p _ { r } ) = \sqrt { p _ { s } }$ , giving $p _ { r } ^ { * } = \sqrt { p _ { s } } / ( 1 + \sqrt { p _ { s } } )$ . The boundary values $\Delta ( 0 ) = \Delta ( 1 ) = 0$ confirm this is a maximum. 口

Corollary 4 (Hump on the Accuracy Axis). Under the assumptions of Corollary 1, with $p _ { s } \in ( 0 , 1 )$ and $P : =$ $P ( C { = } 1 )$ , the bias has a unique interior maximum in accuracy at $P ^ { * } = \sqrt { p _ { s } }$ , where $\Delta ( P ^ { * } ) = ( 1 - \sqrt { p _ { s } } ) / ( 1 + \sqrt { p _ { s } } )$ (proofin Appendix E). Accuracy below $P ^ { * }$ lies on the rising limb, above it on thefalling limb.

Proposition 4 (Ranking Reversal). Consider two models A and B in the strict-reliance case with shared correctness parameters $\left( \pi _ { r s } \right)$ and positive selection probabilities $q _ { m } = p _ { r , m } g _ { 1 , m } + ( 1 - p _ { r , m } ) g _ { 0 , m } > 0 f o r m \in \{ A , B \}$ . If $p _ { r , A } < p _ { r , B } ,$ , a ranking reversal $F _ { \mathrm { s e l } } ( A ) \overset { \cdot } { > } F _ { \mathrm { s e l } } ( B )$ occurs if and only if

$$
p _ { r , A } ( 1 - p _ { r , B } ) g _ { 1 , A } g _ { 0 , B } > p _ { r , B } ( 1 - p _ { r , A } ) g _ { 1 , B } g _ { 0 , A } .\tag{26}
$$

For $0 < p _ { r , A } < p _ { r , B } < 1$ and positive $g _ { 0 , m } , g _ { 1 , m } ,$ , this is equivalent to $L _ { A } / L _ { B } > \mathrm { o d d s } ( p _ { r , B } ) / \mathrm { o d d s } ( p _ { r , A } )$ , where $L _ { m } = g _ { 1 , m } / g _ { 0 , m }$ and $\mathrm { o d d s } ( p ) = p / ( 1 - p )$ . The cross-productform also covers zero cell probabilities whenever $q _ { A } , q _ { B } > 0 .$

When the less CoT-reliant model has greater shortcut prevalence, reversal requires the complementarity condition $\pi _ { 1 1 } \pi _ { 0 0 } > \pi _ { 1 0 } \pi _ { 0 1 }$ , plausible for hard tasks where reliance without shortcut support often produces invalid traces $( \pi _ { 1 0 } \approx 0 )$ but reliance plus shortcut support yields correct answers $( \pi _ { 1 1 } \approx 1 )$ . This requirement concerns $p _ { s , A } > p _ { s , B } ;$ when $p _ { s , A } < p _ { s , B }$ , the opposite sign can permit reversal under Proposition 4.

Under the special case in Corollary 1, the bias is largest at intermediate CoT reliance $( p _ { r } ^ { * } = \sqrt { p _ { s } } / ( 1 + \sqrt { p _ { s } } ) )$ and, on the accuracy axis, at $P ^ { * } = \sqrt { p _ { s } } . \mathrm { ~ \ r ~ A ~ }$ peak near 50% accuracy requires $p _ { s } \approx 0 . 2 5 . \mathrm { A t } p _ { s } = 0$ this special case reduces to $\Delta = 1 - P$ , which is monotone and has no interior maximum. Its interior peak therefore requires a non-degenerate shortcut branch; testing it requires deployments in that regime.

## F.1 RANKING REVERSAL THRESHOLD DERIVATION

For two models A and B with $p _ { r , A } < p _ { r , B }$ and shared correctness parameters, the reversal condition $F _ { \mathrm { s e l } } ( A ) >$ $F _ { \mathrm { s e l } } ( B )$ expands to:

$$
\frac { p _ { r , A } g _ { 1 , A } } { p _ { r , A } g _ { 1 , A } + ( 1 - p _ { r , A } ) g _ { 0 , A } } > \frac { p _ { r , B } g _ { 1 , B } } { p _ { r , B } g _ { 1 , B } + ( 1 - p _ { r , B } ) g _ { 0 , B } } ,\tag{27}
$$

Multiplying by the positive denominators and cancelling the common term gives equation 26. Dividing by positive $p _ { r , A } ( 1 - p _ { r , B } ) g _ { 0 , A } g _ { 1 , B }$ gives the likelihood-ratio form under the additional positivity conditions in Proposition 4.

## G MEASUREMENT VALIDATION

The faithful baseline (n = 50) gives $F _ { \mathrm { p o p } } = F _ { \mathrm { s e l } } = 1 . 0 0 0 , \mathrm { g a p } = 0 . 0 0 0$ , and shortcut rate = 0%. In this check, the oracle introduces no false non-compliance and the interventions correctly modify downstream values.

## H PROTOCOL REVIEW DETAILS

Table 4 records the procedure and reporting population found in each inspected source. Years follow the bibliography; the final column links to the pinned arXiv version and gives its section, equation, or table. A row describes the named analysis, not every experiment in that paper. In particular, an accuracy endpoint, a process-validity judgement, and intervention compliance are different measurements.

Table 4: Reporting populations in prior work. Protocol-specific evidence from 18 papers.
<table><tr><td>Paper</td><td></td><td>Year Procedure</td><td>Reporting population</td><td>Source</td></tr><tr><td>Lanham et al.</td><td></td><td>paraphrase</td><td>2023 Trace truncation, mistakes, Collected traces; length-weighted aggregation</td><td>v1; §2.1–2.6</td></tr><tr><td>Turpin et al.</td><td></td><td>2023 Biased input cues</td><td>Main input-bias tests; correctness-selected qualitative examples</td><td>v2; §3.1; Tab. 7</td></tr><tr><td>Lyu et al.</td><td></td><td>2023 Deterministic solver</td><td>Test-set accuracy; correct outputs for human plausibility v3; §4.2, 6.3</td><td></td></tr><tr><td>Radhakrishnan et al.</td><td></td><td>2023 Truncation and corruption</td><td>All reasoning samples; gold-label subsets for bias tests</td><td>v2; §3.2–3.3</td></tr><tr><td>Wang et al.</td><td></td><td>2023 Retrieval and trace construction</td><td>Test-set task success; correct-answer training pool</td><td>v2; Alg. 1; Tab. 1</td></tr><tr><td>Bentham et al.</td><td></td><td>2024 With/without-CoT</td><td>Dataset average; no answer-correctness screen in metric v3; §3, Eq. 1</td><td></td></tr><tr><td>Paul et al.</td><td></td><td>agreement 2024 Counterfactual mediation</td><td>Mean test-set accuracy differences</td><td>v4; §4; Tab. 13</td></tr><tr><td>Tanneru et al.</td><td></td><td>2024 Early-answer consistency</td><td>Test CoTs; correct-only training/examples are separate</td><td>v2; §2-4</td></tr><tr><td>Arcuschin et al.</td><td></td><td>2025 Input pairs; shortcut detection</td><td>All comparison pairs; correct Putnam responses</td><td>v6; §2.1, 3.1</td></tr><tr><td>Tutek et al.</td><td></td><td>2025 Unlearning; mistake</td><td>Agreement between CoT and no-CoT predictions</td><td>v4; Tab. 2</td></tr><tr><td>Chen et al.</td><td></td><td>insertion 2025 Injected hints</td><td>Pairs where the answer switches to the hint</td><td>v1; §2.1</td></tr><tr><td>Bogdan et al.</td><td></td><td>2025 Step resampling and attention</td><td>Outcome-balanced MATH traces; correct MMLU CoTs v4; §2.1, 6.1</td><td></td></tr><tr><td>Zaman and Srivastava</td><td></td><td>2025 Knowledge edits; corruption</td><td>Constructed explanation pairs; top-class score change</td><td>v3; §3.2; App. D.1</td></tr><tr><td>Shen et al.</td><td></td><td>2026 Instance-level detection</td><td>Annotated correct and incorrect traces</td><td>v2; §3.2;</td></tr><tr><td>Jia et al.</td><td></td><td></td><td>2026 Information flow; hint edits Wrong-hint cases; behavior-selected verbalization</td><td>App. B.1 v1; §6.1–6.4</td></tr><tr><td>Hase and Potts</td><td></td><td>2026 Counterfactual input simulation</td><td>Simulator and cue-influence test sets</td><td>v2; §3–4</td></tr><tr><td>Gur-Arieh et al.</td><td></td><td>2026 Ground-truth meta-evaluation</td><td>Correct outright or wrong-hint-following responses</td><td>v1; §4.1–4.2</td></tr><tr><td>Mittal and Arike</td><td></td><td>2026 Step-validity/coverage judges</td><td>Perfect source chains: every step labelled correct</td><td>v2; §3; Alg. 1</td></tr></table>

Source locators refer to the linked versions and the named analyses.

## H.1 SCOPE AND INTERPRETATION

We retain adjacent protocols to make their selection rules explicit. Lyu et al. derive the answer by deterministic execution of the symbolic trace; their human plausibility study selects correct outputs [Lyu et al., 2023]. Tanneru et al. select correct demonstrations or fine-tuning examples in some variants, which does not establish a correct-only test denominator [Tanneru et al., 2024]. Counterfactual simulation training intervenes on inputs and trains the task model using simulator feedback [Hase and Potts, 2026]; it is not the same operation as editing a task model’s recorded trace. These distinctions prevent training, input, and test-population choices from being counted as one intervention protocol.

The reviewed sources also include protocols with correct and incorrect outputs. Radhakrishnan et al. average their trace-manipulation metrics over all reasoning samples [Radhakrishnan et al., 2023]; Paul et al. report mean test-set accuracy differences under counterfactual mediation [Paul et al., 2024]. FaithCoT-Bench evaluates detectors on annotated correct and incorrect traces [Shen et al., 2026]. In Zaman and Srivastava’s constructed explanation-pair testbed, corruption changes the score of the model’s top predicted class [Zaman and Srivastava, 2025]. These denominators differ from post-hoc selection on factual answer correctness.

This review identifies the reporting population for each measure in the cited versions. It compares reported protocols;   
estimating field-wide prevalence requires systematic sampling.

## I POWER ANALYSIS

Table 5 gives simulation-based power projections at $\alpha = 0 . 0 5$ , computed from the initial $n = 1 0 0$ pilot before the scaled experiments. Both scaled runs subsequently reached the stated significance threshold. GLM-5.3 at $n = 1 0 0 0$ achieved $p < 0 . 0 0 0 1$ (Table 1); the projections remain based on the pilot effect size.

Table 5: Power projections. Pilot-based estimates for the scaled deployment runs.
<table><tr><td>Model</td><td></td><td></td><td>n (pilot) Pilot ∆ (pp) Power at n n for 80% power</td></tr><tr><td>glm-5.2</td><td>100 +1.6</td><td>0.06</td><td>&gt;5,000</td></tr><tr><td>glm-5.3 100</td><td>+8.8</td><td>0.25</td><td>≈400</td></tr><tr><td>deepseek-v4-flash 100</td><td>+3.8</td><td>0.09</td><td>≈1,800</td></tr><tr><td>qwen3.8-max 20</td><td>0.0</td><td>一</td><td></td></tr></table>

Achieved: glm-5.3 at n = 1000: ∆ = +7.4pp, p < 0.0001.  
Achieved: deepseek-v4-flash at n = 300: ∆ = +6.0pp, p = 0.046.

The pilot projections give low power at $n = 1 0 0$ . Detecting GLM-5.3’s +8.8pp pilot gap at 80% power requires approximately $n = 4 0 0$ instances; glm-5.2’s smaller +1.6pp gap requires thousands. At the pilot $n = 1 0 0$ , GLM-5.3’s 95% CI is [−1.5, +19.1]pp. Holding the effect at +8.8pp in the projection gives CI [+1.5, +16.1]pp at n ≈ 200 and CI [+2.85, +14.75]pp at $n = 3 0 0$ . These projected intervals depend on the pilot effect size.

## J WITHIN-STRATUM DIFFICULTY ANALYSIS

We stratify by interventionable-operation count, the number of editable operations in an instance. This covariate takes the values $\{ \dot { 1 } , 2 , 3 , 5 , 7 , 8 , 1 1 \}$ in the generated tasks. The generator never emits counts 4, 6, 9 or 10; the tables retain these empty strata with n = 0 to make the sample accounting explicit. Operation count is a proxy for difficulty. It is fixed for each instance across models and precedes the model’s answer.

Table 6 presents GLM-5.3 results (n = 1000) stratified on that count.

In Tables 6, 7, and 9, a denotes compliant-and-correct, b correct-but-non-compliant, c incorrect-but-compliant, and d incorrect-and-non-compliant. Missing or unparseable responses count as non-compliant. Each stratum’s four cells sum to its sample size; the GLM-5.3 column totals match Table 7 and reproduce its operation-count decomposition.

Easy strata (1–3 operations) have small gaps $( - 1 . 1 \mathrm { t o } + 2 . 8 \mathrm { p p } )$ , higher accuracy, and less selective filtering than the remaining strata. Medium-to-hard strata (5–8) have the largest gaps $( + 7 . 0 \mathrm { t o } + 8 . 5 \mathrm { p p } )$ , with a smaller fraction retained by the filter. The operation-count/gap rank association is positive but non-significant (Spearman $\rho = 0 . 7 5 , p = 0 . 0 5 2$ $n = 7 \ \mathrm { s t r a t a } )$

At 11 operations the gap is smaller $\left( + 6 . 0 \mathrm { p p } \right)$ than at 5–8. This stratum’s $F _ { \mathrm { p o p } } \left( 0 . 5 5 3 \right)$ exceeds the values at 3–8 operations, but remains below those at 1 and 2 operations (0.922 and 0.750). Its null rate is close to the pooled rate

Table 6: GLM-5.3 results by operation count. Joint outcome counts and selection gaps, with empty strata retained for completeness.
<table><tr><td>Steps</td><td>n</td><td>a</td><td>b C</td><td></td><td>d</td><td>Acc.  $F _ { \mathbf { p o p } }$ </td><td> $\pmb { F } _ { \mathrm { s e l } }$ </td><td>∆(pp)</td></tr><tr><td>1 (simple)</td><td>115</td><td>81</td><td>5 25</td><td>4</td><td>75%</td><td>0.922</td><td>0.942</td><td>+2.0</td></tr><tr><td>2 (easy)</td><td>104</td><td>51</td><td>18 27</td><td>8</td><td>66%</td><td>0.750</td><td>0.739</td><td>-1.1</td></tr><tr><td>3 (moderate)</td><td>232</td><td>53</td><td>77 35</td><td>67</td><td>56%</td><td>0.379</td><td>0.408</td><td>+2.8</td></tr><tr><td>4</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>5 (medium)</td><td>223</td><td>56</td><td>57 39</td><td>71</td><td>51%</td><td>0.426</td><td>0.496</td><td>+7.0</td></tr><tr><td>6</td><td>0</td><td>0</td><td>0 0</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>7 (hard)</td><td>100</td><td>17</td><td>25 15</td><td>43</td><td>42%</td><td>0.320</td><td>0.405</td><td>+8.5</td></tr><tr><td>8 (very hard)</td><td>123</td><td>22</td><td>22 29</td><td>50</td><td>36%</td><td>0.415</td><td>0.500</td><td>+8.5</td></tr><tr><td>9</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>10</td><td>0</td><td>0 0</td><td>0</td><td>0</td><td></td><td></td><td></td><td></td></tr><tr><td>11 (hardest)</td><td>103</td><td>19 12</td><td>38</td><td>34</td><td>30%</td><td>0.553</td><td>0.613</td><td>+6.0</td></tr><tr><td>Total</td><td>1000</td><td>299</td><td>216 208</td><td>277</td><td></td><td></td><td></td><td></td></tr></table>

(7.8% versus 7.2%). Sampling variation in the small stratum $( n = 1 0 3 )$ and differences in graph structure are both possible explanations. This stratum contains the deepest graphs, where sequential dependencies can be easier than branching ones. The current data do not distinguish these explanations.  
Operation count therefore does not consistently order the strata by difficulty. The 42.8% between-stratum share reported in Section 6 quantifies variation explained by the operation count.

## J.1 EXACT COVARIANCE DECOMPOSITION

Table 8 reports the decomposition from Section 6, computed by the law of total covariance equation 18 with interventionable-operation count as the stratifier. The identity holds to floating-point precision without an independence assumption. It partitions the observed covariance; the IPW analysis below instead reweights selected instances under a propensity model.

From equation 17, $\Delta = \mathrm { C o v } ( A , C ) / P ( C { = } 1 )$ and $1 / P ( C { = } 1 )$ is constant within a model’s run. The between- and within-stratum terms of equation 18 partition $\operatorname { C o v } ( A , C )$ . Multiplying both terms by the same constant partitions ∆, so the covariance and gap have the same percentage split. Table 8 reports shares of the gap.

Table 7 gives each model’s four (A, C) cells under the null-as-non-compliant convention. Tables 6 and 9 give the per-stratum cells. These printed counts reproduce every point-estimate share in Table 8.

The joint counts give $F _ { \mathrm { s e l } } = a / ( a + b )$ and $F _ { \mathrm { p o p } } = ( a + c ) / n$ . The odds ratio is ${ a d } / ( { b c } )$ , with a 0.5 continuity correction if a cell is zero. The pooled Cochran–Mantel–Haenszel analysis uses the four sub-ceiling deployments.

Table 7: Joint compliance and correctness counts. Ceiling controls are excluded from the pooled analysis.
<table><tr><td>Model</td><td>n</td><td>a</td><td>b</td><td>C</td><td>d ∆(pp)</td><td>OR</td></tr><tr><td>GLM-5.3</td><td>1000</td><td>299 216</td><td>208</td><td>277</td><td>+7.4</td><td>1.84</td></tr><tr><td>deepseek-v4-flash</td><td>300</td><td>85 60</td><td>73</td><td>82</td><td>+6.0</td><td>1.59</td></tr><tr><td>deepseek-v4-pro</td><td>200</td><td>77 56</td><td>34</td><td>33</td><td>+2.4</td><td>1.33</td></tr><tr><td>glm-5.2</td><td>100</td><td>36 43</td><td>8</td><td>13</td><td>+1.6</td><td>1.36</td></tr><tr><td>Boundary controls  $( P ( C { = } 1 ) = 1 ; c = d = 0$  by construction):</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>qwen3.8-max</td><td>20</td><td>12 8</td><td>0</td><td>0</td><td>0.0</td><td></td></tr><tr><td>Gemini 3.5 Flash Lite</td><td>17</td><td>9 8</td><td>0</td><td>0</td><td>0.0</td><td></td></tr></table>

Figure 11 shows how the GLM-5.3 joint counts determine the reporting-population contrast.

The descriptive 95% percentile intervals in Table 8 use an instance-level bootstrap with 10,000 resamples. Row-level data and exploratory recomputation outputs are not included in the public release. The share divides by Cov(A, C), which can approach zero during resampling. This produces asymmetric intervals; the flash interval crosses zero.

![](images/271e786a78b0f4573816d97f55460d076e4a43f57c403c16ba13db0609f50257.jpg)

![](images/0dbdf323d04f4d4e76539e4d2b28b9342eee692fce0f4c9ce4c5229fc3df1f43.jpg)

![](images/9f1243ac1b476f405549cdfb29cae4dcc20c9c1a5ef6db0f0fd2410b115221df.jpg)

![](images/597e1d57b07e516dab2305e83aaed5c343c88449ed25313a4045e467ed60bb63.jpg)  
Figure 11: Worked selection calculation. An illustrative intervention and the joint counts connecting oracle compliance to all-trace and correctness-selected estimates.

Operation count gives a 42.8% between-stratum share for GLM-5.3, while oracle-displacement magnitude gives 26.6% on the same records. These covariate-dependent shares do not identify a causal contribution of latent difficulty.

Table 8: Covariance decomposition by deployment. Brackets report bootstrap intervals; the within-stratum component remains unattributed.
<table><tr><td>Model</td><td>n</td><td>∆(pp)</td><td>Between-stratum</td><td>Within-stratum</td></tr><tr><td>GLM-5.3</td><td>1000</td><td>7.36</td><td>42.8% [26.8, 69.1]</td><td>57.2%</td></tr><tr><td>deepseek-v4-flash</td><td>300</td><td>5.95</td><td>21.9%[-39.0, 112.2]</td><td>78.1%</td></tr></table>

The flash strata in Table 9 comprise $n = 3 0 0$ instances and yield the 21.9% between-stratum share in Table 8. The corresponding GLM-5.3 strata are given in Table 6.

## J.2 INVERSE PROPENSITY WEIGHTING (IPW) MODEL DETAILS

We use an inverse-probability-weighted estimator [Horvitz and Thompson, 1952]. We fit a logistic propensity model, $\hat { P } ( C { = } 1 \mid \mathrm { s t e p s } ) = \mathrm { l o g i s t i c } ( \hat { \beta } _ { 0 } + \hat { \beta } _ { 1 } \cdot \mathrm { s t e p s } )$ , where steps is the number of interventionable operations (β<sup>ˆ</sup> = 0.954, $\hat { \beta } _ { 1 } = - 0$ .181). Importance reweighting is a standard correction for sample-selection bias in supervised learning [Zadrozny, 2004, Cortes et al., 2008].

We apply weights $w _ { i } = 1 / \hat { P } ( C = 1 \mid D _ { i } )$ to all correctness-selected instances. The numerator sums weights for compliant selected instances; the denominator sums weights for all selected instances. Under $A \perp C \mid \bar { D _ { \ L } } I ,$ this normalized weighted mean is a Hájek estimator of $F _ { \mathrm { p o p } } .$ . The definition of $F _ { \mathrm { s e l } }$ is unchanged. We apply no weight trimming or capping. The effective sample size is $n _ { \mathrm { e f f } } ^ { \cdot } \stackrel { \cdot } { = } ( \sum w _ { i } ) ^ { 2 } / \sum w _ { i } ^ { 2 } = 4 7 2$ , with mean weight 1.95, median 1.66, and maximum 3.83.

The logistic propensity gives $\hat { F } _ { \mathrm { I P W } } = 0 . 5 6 4$ , leaving a +5.7pp residual relative to the direct $F _ { \mathrm { p o p } } = 0 . 5 0 7 , \mathrm { a } 2 3 \%$ reduction from the raw selected gap. A nonparametric per-stratum propensity gives $\hat { F } _ { \mathrm { I P W } } = 0 . 5 5 5$ , a 34% reduction.

Table 9: DeepSeek-v4-flash strata. Joint counts by interventionable-operation count for covariance decomposition.
<table><tr><td>Steps</td><td>n</td><td>a</td><td>b</td><td>C</td><td>d</td></tr><tr><td>1 (simple)</td><td>36</td><td>21</td><td>1</td><td>12</td><td></td><td>2</td></tr><tr><td>2 (easy)</td><td>42</td><td>9</td><td></td><td>9</td><td>9</td><td>15</td></tr><tr><td>3 (moderate)</td><td>63</td><td>12</td><td>21</td><td></td><td>9</td><td>21</td></tr><tr><td>5 (medium)</td><td>59</td><td>15</td><td>12</td><td></td><td>17</td><td>15</td></tr><tr><td>7 (hard)</td><td>33</td><td>5</td><td>8</td><td></td><td>8</td><td>12</td></tr><tr><td>8 (very hard)</td><td>37</td><td>13</td><td></td><td>5</td><td>11</td><td>8</td></tr><tr><td>11 (hardest)</td><td>30</td><td>10</td><td>4</td><td></td><td>7</td><td>9</td></tr><tr><td>Total</td><td>300</td><td>85</td><td>60</td><td>73</td><td></td><td>82</td></tr></table>

Both adjustments depend on the propensity specification, and neither identifies a causal decomposition. The residual remains unattributed.

## K REVERSAL SENSITIVITY ANALYSIS

Table 10 varies $\pi _ { 1 0 } = P ( C = 1 \mid R = 1 , S = 0 )$ while holding $\pi _ { 0 0 } = 0 . 0 5 , \pi _ { 0 1 } = 0 . 8 0$ , and $\pi _ { 1 1 } = 1 . 0 0$ . The reference model has $( p _ { r , B } , p _ { s , B } ) = ( 0 . 7 0 , 0 . 0 1 )$ . The full candidate grid comprises 199 equally spaced $p _ { r , A }$ values from 0.01 to 0.99 and 201 equally spaced $p _ { s , A }$ values from 0 to 1: 39,999 candidates in total. We count a reversal only when $p _ { r , A } < p _ { r , B }$ and $\bar { F } _ { \mathrm { s e l } } ( \mathbf { \dot { \mathcal { A } } } ) > F _ { \mathrm { s e l } } ( \mathbf { \dot { \mathcal { B } } } )$ , using strict inequalities.

Figure 12 shows $\pi _ { 1 0 } = 0 . 0 0 1$ , where faithful reasoning without shortcut support almost never succeeds. Shortcut prevalence varies across candidates; S is not fixed to one. The plot spans $( \dot { p } _ { r , A } , p _ { s , A } ) \in [ 0 , 1 ] ^ { 2 }$ , and the reported 40.6% is 16,224/39,999 on the discrete full grid specified above. Candidates with $p _ { r , A } \ge 0 . 7 0$ do not meet the lower-population-faithfulness condition.

![](images/b35dab27062007bf77e9d6e4ef22c35d18dc81034ed1f09c8c0c6e8872444baa.jpg)  
Figure 12: Ranking reversals at $\pi _ { 1 0 } = 0 . 0 0 1$ . The orange region contains candidates with lower population faithfulness but higher selected-trace scores than reference $B .$

Independent exact-rational evaluation reproduces the counts below. Requiring both inequality margins to exceed $1 0 ^ { - 8 }$ gives the same counts, excluding numerical ties as their source

Table 10: Reversal sensitivity. Counts and percentages on the same 39,999-candidate grid, with the reference and other correctness parameters held fixed.
<table><tr><td>π10</td><td>Reversal fraction Reversal count</td><td></td></tr><tr><td>0.001</td><td>40.561%</td><td>16,224</td></tr><tr><td>0.05</td><td>3.568%</td><td>1,427</td></tr><tr><td>0.1</td><td>0.0075%</td><td>3</td></tr><tr><td>0.2</td><td>0.0150%</td><td>6</td></tr></table>

At $\pi _ { 1 0 } = 0 . 0 0 1$ and 0.05, every reversal has $p _ { s , A } > 0 . 0 1$ , consistent with the positive complementarity determinant $\pi _ { 1 1 } \pi _ { 0 0 } - \pi _ { 1 0 } \pi _ { 0 1 }$ . At $\pi _ { 1 0 } = 0 . 1$ and 0.2, that determinant is negative, ruling out reversals with $p _ { s , A } > p _ { s , B }$ . The 3 and 6 remaining reversals instead have lower shortcut prevalence: respectively 2 and 4 occur at $p _ { s , A } = 0 ,$ , and 1 and 2 at $p _ { s , A } = 0 . 0 0 5$ . The table describes these four settings; it does not establish that reversal is impossible for every $\pi _ { 1 0 } \geq 0 . 2 $

## L LEAVE-ONE-OUT SENSITIVITY ANALYSIS

Table 11 reports the CMH test after removing each model in turn. Figure 13 shows the deployment-specific and pooled odds ratios. All leave-one-out tests remain significant $( p < 0 . 0 3 )$ , so the pooled association remains detectable after removing any single model. GLM-5.3 is the largest single contributor.

Across the four sub-ceiling models, the pooled test gives z = 5.19, $p < 1 0 ^ { - 6 }$ , OR = 1.70, and $I ^ { 2 } = 0 \%$ . Gaps also persist within difficulty strata, though their rank association is only suggestive $( \rho = 0 . 7 5 , p = 0 . 0 5 2 ;$ Appendix J). The per-model ORs are 1.33, 1.59, 1.36, and 1.84 across the combined sample $( n = 1 6 0 0 )$ . The null-compliance rate is 5.44% (87/1600); nulls are scored non-compliant under the headline convention, with a sensitivity check in Appendix P. Elicitation found 10% cue use in GLM-5.3 (Appendix M).

Table 11: Leave-one-out analysis. Pooled association after omitting each deployment.
<table><tr><td>Removed</td><td>Remaining n</td><td>Z</td><td>p (two-sided)</td></tr><tr><td>GLM-5.3</td><td>600</td><td>2.24</td><td>0.025</td></tr><tr><td>deepseek-v4-flash</td><td>1300</td><td>4.80</td><td>&lt;0.0001</td></tr><tr><td>deepseek-v4-pro</td><td>1400</td><td>5.17</td><td>&lt;0.0001</td></tr><tr><td>GLM-5.2</td><td>1500</td><td>5.17</td><td>&lt;0.0001</td></tr></table>

## M SHORTCUT ELICITATION DETAILS

Following prior work on biased feature cues [Turpin et al., 2023], we applied adversarial framing cues (authority, social proof, and shortcut suggestion) to 50 arithmetic DAG instances per model. GLM-5.3 exhibited 10% shortcut use overall (5/50, 95% Wilson CI [4.3%, 21.4%]), compared to 0% under neutral framing; deepseek-v4-flash showed 0%. The observed GLM-5.3 gap (+3.0pp) is directionally consistent with the theory but not significant at n = 50. All five shortcuts occurred on correct-cue instances and passed the correctness filter; these observations do not quantify a collider contribution.

Targeted elicitation on a second deployment. We vary the framing of the same arithmetic DAG task. V1 asks the model to verify a stated value; V2 presents it as descriptive social proof; V4 explicitly instructs the model to rely on a stated “definitive” value. Under V4, deepseek-v4-pro followed the cue on 12 of 12 informative (incorrect-value) instances and answered all 12 incorrectly. The correctness filter removes all of these rows. Paired controls for three instances span easy and hard tiers: the same task is answered correctly without a cue and incorrectly with one.

Under V2, the same model followed the cue on 0 of 4 informative instances, computing correctly and explicitly rejecting it. Flash followed the cue on 0 of 5 informative instances and glm-5.2 on 0 of 2; neither model followed it under any tested framing. One V1 call returned no completed answer and was not scored. The observed following rate was approximately 0 under V2, unmeasurable under V1, and approximately 1 under V4, with no framing producing partial following in this probe. Following depended on both framing and model; these observations do not support interpolating a graded response.

![](images/698d93fae9e71dfea820de9bbb7f9cc340e6cb7620254aa9efc4ba8e8d0351a1.jpg)  
Figure 13: Deployment-level and pooled odds ratios. Square areas encode inverse-variance weights; horizontal lines show individual confidence intervals and the diamond the pooled interval.

In a representative case, the model derived the correct value from the DAG but substituted the cue in its final answer.   
The error occurred at the output stage. The correctness filter excludes this case despite the trace’s correct computation.

This exploratory probe identifies framing-dependent cue following in small, post-hoc samples. It was not preregistered, uses a separate serving endpoint from Table 1, and covers only part of the hard-task tier. We therefore report it separately from the headline runs. Estimating a population rate or testing the accuracy-axis peak requires a larger design with partial-following conditions.

Cross-family extension. We apply the V4/V2/V1 framings to additional model families. Each row reports a separate deployment; we do not pool across endpoints.
<table><tr><td>Model family</td><td>Measured Followed</td><td></td></tr><tr><td>gemini-2.5-flash</td><td>6</td><td>6</td></tr><tr><td>deepseek-v4-pro</td><td>12</td><td>12</td></tr><tr><td>kimi-k3</td><td>6</td><td>2</td></tr><tr><td>glm-5.2</td><td>5</td><td>0</td></tr><tr><td>deepseek-v4-flash</td><td>2</td><td>0</td></tr></table>

Counts are measured rows; calls without a completed answer are excluded from the rates.

Across the single-endpoint and cross-family probes combined, V2 yielded 0 of 14 informative follows and V1 yielded 0 of 7. V4 was the only framing to induce following. The rates range from 0% to 100% across families, consistent with a framing × family interaction.

V4-neutral control. The V4-neutral condition uses the same wording instructing deference but sets the stated value to the correct answer. In 25 measured neutral rows, no deployment cited an instruction to skip computation. These rows include 13 discriminating pairs, in which the deployment followed under V4-wrong and was also tested under V4-neutral. Gemini supplies behavioural evidence because it returns no plaintext reasoning. In the other paired cases, compliance language appears when the stated value differs from the computed value and disappears when the stated value is correct. This contrast is observed; interpreting it as a consistency check remains a hypothesis that the sample does not independently confirm.

All cross-family comparisons are also cross-endpoint. The probe covers only part of the hard-task tier. Gemini’s endpoint quota limited its sample size, and one model family used a non-default temperature because its endpoint rejected temperature = 0. These conditions limit the comparisons to exploratory post-hoc observations.

Selection-gap estimates under adversarial framing. The 100 evaluation records used to estimate shortcut use also have factual and counterfactual answers, oracle values, correctness labels, and compliance scores. They use the same arithmetic DAGs as the headline evaluation, with operation counts $\in \{ 1 , 2 , 3 , 5 , \bar { 7 } , 8 , 1 1 \}$ . We apply the same estimator to this second cue regime; it adds no new task family.

Table 12 uses the earlier adversarial run, with $n = 5 0$ per model on the same endpoints as Table 1; it does not summarize the later targeted probes. Its tests are Yates-corrected $\chi ^ { 2 }$ tests with one degree of freedom, and neither gap is significant. For flash, the incorrect-cue joint cells $( a , b , c , d ) = ( 0 , 5 , 3 , 6 )$ determine the overall $\mathrm { g a p ' s }$ negative sign.

Table 12: Selection gaps under adversarial framing. Estimates from the earlier run, separate from the subsequent exploratory probes.
<table><tr><td>Model</td><td>n Acc.</td><td></td><td> $F _ { \mathbf { p o p } }$ </td><td> $\pmb { F } _ { \mathrm { s e l } }$ </td><td>∆(pp)</td><td> $x ^ { 2 } \left( p \right)$ </td></tr><tr><td>GLM-5.3</td><td></td><td>5078%</td><td>0.560</td><td>0.590</td><td>+3.0</td><td>0.206 (0.650)</td></tr><tr><td>deepseek-v4-flash</td><td></td><td>50 44%</td><td>0.220</td><td>0.182</td><td>-3.8</td><td>0.055 (0.815)</td></tr></table>

Breakdown by cue correctness: flash correct-cue +1.31pp (n=36), incorrect-cue −21.43pp (n=14);  
GLM-5.3 correct-cue +4.50pp (n=36), incorrect-cue +1.43pp (n=14).

Both gaps are non-significant at $n = 5 0$ , with 39 and 22 correct rows respectively. The pilot power range is 0.06–0.25 at $n = 1 0 0$ (Appendix I), so these smaller runs have limited power. The estimates describe selection in the same data used to measure shortcut use.

The negative flash estimate does not identify a mechanism. The collider pathway requires an active shortcut branch, and flash shows 0/50 shortcuts here (all five elicited shortcuts are GLM-5.3’s, and all fall on correct-cue instances that pass the filter). The negative overall figure is a 4-versus-4.84 cell difference concentrated in 14 incorrect-cue rows whose selected-trace numerator is zero; this cell difference provides no evidence for a specific pathway.

## N EMPIRICAL GAP VS. ACCURACY

Figure 14 shows the deployment estimates with complete confidence intervals.

The shaded mid-accuracy band is descriptive. The theory instead places the peak at $P ^ { * } = \sqrt { p _ { s } }$ (Corollary 4). At the implementation-check value $p _ { s } = 0 . 7 ,$ , the peak is 0.84, above every tested sub-ceiling deployment. The two ceiling controls also do not locate the peak: $P ( C { \bar { = } } 1 ) = 1$ makes the filter vacuous and forces $\Delta = \dot { 0 }$ by construction.

## O CEILING VALIDATION

The two ceiling controls use task difficulties at which every valid response is factually correct. Gemini 3.5 Flash Lite (Google) returned 17 valid responses from 20 requests (three were unparseable), with $F _ { \mathrm { s e l } } = F _ { \mathrm { p o p } } = 0 . 5 2 9 ;$ qwen3.8-max has $n = 2 0$ , 100% accuracy, and $\Delta = 0$ . Their lengths are unmatched, so the numerical scores are not directly comparable. In both constructed evaluation sets, $P ( C { = } 1 ) = 1$ makes the filter vacuous and forces $\Delta = 0$ (Proposition 1). These runs check the implementation at the boundary; they do not independently establish a selection effect.

On Boolean circuits, the same all-trace protocol with GLM-5.3 (n = 200) gives 99.5% accuracy, $F _ { \mathrm { p o p } } = F _ { \mathrm { s e l } } = 0 . 9 9 5$ and $\Delta = 0 . 0 \mathrm { p p }$ , compared with +7.4pp on arithmetic DAGs at intermediate accuracy. This second-family control agrees with the predicted limit $\Delta \to 0$ as P(C=1) → 1 (Section B); the circuits were too easy to measure the model below ceiling.

At depth 30 (90 gates), we verified non-thinking mode on every row (reasoning\_tokens= 0), with no parse failures. Deepseek-v4-flash answered 10/10 correctly (one-sided 95% lower bound 0.741); glm-5.2 answered all 10 calls correctly (bound 0.688 at the conservative $n = 8$ distinct instances). Both bounds exclude the 40–60% band. These two deployments are separate from the GLM-5.3 control.

A further deepseek-v4-flash test increased the serial chain from 90 to 120 to 300 gates (depths 30, 40, and 100); accuracy remained 10/10. The relay’s 180 s limit prevented measurement at 600 gates. Thus the reachable circuit depths provide no non-ceiling measurement. Arithmetic DAGs instead reach 48–52% accuracy at depths 2–4. This contrast documents reachable difficulty for the tested deployments, not a difference in mechanism or an extension of the selection-bias evidence.

![](images/9f9461e406f3138159ed9f8f7de9ff745708c19ac1b7692f694b08eed17d1be2.jpg)  
Figure 14: Selection gap versus accuracy. All confidence intervals are shown; asterisks mark descriptive $p < 0 . 0 5$

## P COMPLIANCE ON INCORRECT TRACES AND NULL HANDLING

Compliance on incorrect traces (38–51% across models) is semantically ambiguous: it may reflect genuine partial faithfulness, coincidental answer changes, or error-correlated sensitivity. All-trace evaluation includes these instances, so $F _ { \mathrm { p o p } }$ is informative if partial faithfulness dominates and inflated if coincidental changes do; the estimator is not uniformly conservative in either direction.

Null handling. Our primary protocol scores unparseable or missing responses as non-compliant (A=0). Nulls are unevenly distributed across instances, so assigning A=0 can move the gap in either direction. We test this convention by treating null responses as missing and dropping them from both $F _ { \mathrm { p o p } }$ and $F _ { \mathrm { s e l } }$ , using the same instance sets as the headline analysis:
<table><tr><td>Model</td><td></td><td></td><td>n (noncomp) n (missing) ∆ (noncomp) ∆ (missing)</td><td></td><td>shift</td></tr><tr><td>GLM-5.3</td><td>1000</td><td>928</td><td>+7.36pp</td><td>+7.14pp</td><td> $+ 0 . 2 2$ </td></tr><tr><td>flash</td><td>300</td><td>291</td><td>+5.95pp</td><td>+5.15pp</td><td> $+ 0 . 8 1$ </td></tr><tr><td>v4-pro</td><td>200</td><td>198</td><td>+2.39pp</td><td>+2.27pp</td><td> $+ 0 . 1 2$ </td></tr><tr><td>glm-5.2</td><td>100</td><td>96</td><td>+1.57pp</td><td>+0.92pp</td><td>+0.65</td></tr></table>

Scoring nulls as non-compliant increases the gap by +0.12 to $+ 0 . 8 1 \mathrm { p p }$ relative to dropping them. Nulls fall disproportionately on incorrect instances (for GLM-5.3, 41 of 72 nulls are on incorrect rows against a 48.5% base rate), so assigning A=0 depresses $F _ { \mathrm { p o p } }$ more than $F _ { \mathrm { s e l } }$ . All four gap estimates remain positive under both conventions.

## Q FULL DISCUSSION

Pathway attribution. The detector flagged no neutral-framing instances. A flag requires the answer to equal the cue and either the trace to name it (“reference”, “previously”, “provided result”) or the answer to diverge from the oracle. Unacknowledged use of a correct cue escapes detection; 72% of cues were correct. The zero rate therefore cannot exclude collider bias $( R \to C  S )$ . Operation-count strata capture 21.9% (flash) and 42.8% (GLM-5.3) of the gap, consistent with the difficulty paths $D  C$ and $D  A .$ , but most remains unattributed.

Because interventions leave the cue unchanged, cue correctness provides a descriptive check. In the n = 300 arithmetic flash run, complete-case compliance is 53.8% (112/208) with correct cues and 55.4% (46/83) with incorrect cues: $- 1 . 6 \mathrm { p p } \left( z = - 0 . 2 4 , p = 0 . 8 1 \right)$ . This excludes the 9 nulls retained in the headline estimate. The non-significant contrast does not resolve cue reliance, since cue use in both groups can also produce similar rates.

Adversarial elicitation flags 10% of GLM-5.3 responses (5/50, Section M); all five use correct cues and pass the filter. The targeted incorrect-cue probe uses separate deployments: under explicit-reliance framing, deepseek-v4-pro (12/12), gemini-2.5-flash (6/6), and kimi-k3 (2/6) follow the stated value on informative instances, producing wrong answers excluded by the filter. Neither glm-5.2 nor flash follows in any tested framing. These unpooled probes establish cue following and exclusion, not its contribution to the headline gap. Both pathways can coexist under Proposition 1.

Difficulty adjustment and residual. Logistic IPW removes 23% of the gap, leaving a significantly positive residual. This single-covariate adjustment depends on the propensity specification and does not identify the residual’s mechanism.

## R FULL LIMITATIONS

Scope. Two of the four sub-ceiling models reach individual significance $( p < 0 . 0 5 )$ . The other two have positive point estimates with intervals crossing zero at n = 100–200. The two 100%-accuracy controls are algebraic identities and are excluded from the pooled test; Table 1 reports their p-values as N/A.

All headline real-model runs use arithmetic DAGs. Boolean circuits were evaluated only at ceiling and therefore check the boundary condition. We also attempted graph reachability with $\mathrm { G L M - } 5 . 3 ( n = 2 0 )$ , but excluded it from the evidence: accuracy was 95%, and only 3 of 20 counterfactual responses parsed, with the rest empty. This run supports no inference about the mechanism. No family other than arithmetic has been measured at non-ceiling accuracy. Generalization to natural-language CoT, a central setting in the reviewed literature, remains untested.

Mechanism. The collider pathway $( R \to C  S )$ requires a shortcut branch, which was not detected under neutral framing. A targeted probe reaches it on multiple deployments under an explicit-reliance framing (Section M), using different endpoints without pooling. For the headline gaps, the measured operation-count association is consistent with difficulty confounding; it does not identify a causal contribution.

Residual. IPW removes 23–34% of the gap across propensity specifications. The logistic specification leaves +5.7pp unattributed; the data do not separate an R-confounder from other residual causes.

Measurement. Compliance on incorrect traces $( C = 0 , 3 8 – 5 1 \%$ across models) is semantically ambiguous. All-trace evaluation removes correctness selection but leaves compliance on $C = 0$ instances ambiguous (Section P).

Decoding and serving deployment. Headline runs use greedy decoding (temperature = 0) and English-only prompts. We have not evaluated stochastic decoding or multilingual prompts. The same model identifier behaved materially differently on two endpoints, so Table 1 reports deployment-specific gaps.