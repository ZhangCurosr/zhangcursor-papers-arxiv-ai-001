# FPSRG: Heterogeneous User Simulation First-Person Post-Training and Multidimensional Semantic Response Grids

Ruimin Pei <sup>1</sup> Haibo Liu <sup>2</sup> Fan Zhang <sup>2</sup> Xunhua Guo <sup>1</sup>

## Abstract

Simulating heterogeneous individuals with large language models is an active research direction, and heterogeneous user simulation requires models with diferent characteristics. We study profile construction: given dimensions, items, and scoring rules, build model configurations that can be retrieved by multidimensional target scores. We introduce First-Person Semantic Response Grids (FPSRG). Shared post-training first teaches the model to judge and answer as the participant; self-report and behavioral targets then refine dimension-specific control vectors. With the model frozen, we measure joint scores for vectorcoeficient combinations, filter inadmissible configurations, and store the results in a searchable catalog. Per-token configuration routing, continuous batching, and isolated caches support parallel measurement of diferent profiles. Non-thinking five- and ten-dimensional Qwen instances and a thinking-mode six-dimensiona instance following full-parameter post-training of Huihui demonstrate the complete workflow. With matched data, layers, and norms, refinement increases mean five-axis composite control span from 30.75 to 40.45. Within the five-dimensional 25–75 target region, the qualified five-level vector catalog covers 92.2% of fixed targets under per-axis error at most 20 and total error at most 60; combining it with the prompt catalog increases coverage to 98.7%. At a fixed coefficient range, three-, five-, and nine-level catalogs achieve 71.2%, 92.2%, and 94.6% coverage, demonstrating the gains from denser measurement. These results describe measured reachable regions under a fixed assessment protocol. Through shared adaptation, a small set of learned

directions, and reusable joint measurements, FP-SRG constructs heterogeneous individual configurations without retraining a model for every profile.

## 1 Introduction

Facing an unfinished task, some people worry about the deadline, others prioritize commitments, and others negotiate responsibilities. Customer-service training, interactive evaluation, and social research need to construct these diferences. Language models provide simulated participants (Ni et al., 2026; Argyle et al., 2023; Park et al., 2023) for comparing how diferent attribute combinations respond to the same information. We study heterogeneous model configurations that can be retrieved by target scores and repeatedly measured within a specified multidimensional evaluation system.

Prompts describe high or low tendencies, but “very,” “moderately,” and “slightly” still require score calibration. Training a LoRA for every profile introduces repeated optimization and storage as combinations increase. Personality Sliders, PERSONA, and ORBIT demonstrate strength adjustment and attribute composition (Hoppe et al., 2026; Feng et al., 2026; Ghasemi et al., 2026). Mapping coeficients to semantic coordinates additionally requires handling crossdimensional responses, saturation, and interaction: singleaxis efects do not determine combination outcomes.

First-Person Semantic Response Grids (FPSRG) first use shared post-training to organize expression around a participant’s facts, goals, and trade-ofs, then refine directions with self-report and behavioral targets. With the model frozen, we measure complete score vectors for coeficient combinations and store a searchable configuration–score catalog. This retains coupling and retrieves by target error. Prompt configurations can enter through the same calibration procedure to complement vector reachability. All vector profiles share one post-trained model rather than requiring separate adapters.

Our contributions are:

1. A task of constructing and retrieving heterogeneous configurations by multidimensional target scores. Reachability and coverage require one configuration to satisfy all dimension-wise error requirements.

2. Evidence on data, refinement, and injection scope in single-axis control. Matched-data, fixed-layer, norm-matched, and cognitive/behavioral supervision comparisons provide measured components for joint control.

3. Eficient semantic grids and quantified density/ range efects. Token routing, continuous batching, and safe cache reuse support parallel calibration; density comparisons measure precision gains, and complementary prompts expand reachable regions.

## 2 Related Work

## 2.1 Heterogeneous User Simulation and Participant Perspective

USP identifies role confusion caused by helpful-assistant training and generates user utterances through implicit profiles, conditional fine-tuning, and cycle-consistency reinforcement learning (Wang et al., 2025). This motivates learning shared participant-style expression before freezing the model and constructing diferentiated configurations.

Sources of heterogeneity include demographics (Argyle et al., 2023), Anthology’s backstories (Moon et al., 2024), Generative Agents’ memory and planning (Park et al., 2023), and individual records (Park et al., 2026). RoleLLM studies role capabilities (Wang et al., 2024); ThinkingUS models situated user reasoning (Wu et al., 2026); HER distinguishes first-person role reasoning from third-person model reasoning (Du et al., 2026); the Assistant Axis analyzes default assistant identity (Lu et al., 2026). FPSRG separates the shared participant standpoint from variable attributes, using directions and measured responses to create retrievable profiles.

## 2.2 Activation Interventions and Multi-Attribute Composition

PPLM modifies hidden states with attribute gradients; FUDGE adjusts probabilities using partial-sequence attribute predictions (Dathathri et al., 2019; Yang & Klein, 2021). ITI shifts attention-head activations (Li et al., 2023), CAA extracts contrastive residual directions (Rimsky et al., 2024), BiPO learns directions through bidirectional preference optimization (Cao et al., 2024), and ReFT learns parameterized representation interventions (Wu et al., 2024). We connect contrastive initialization to joint refinement on self-report and behavioral targets.

Composition approaches include dynamic attribute adjustment (Scalena et al., 2024), MAT-Steer’s sparse orthogonal interventions (Nguyen et al., 2025), K-Steering’s multilabel control (Oozeer et al., 2025), and PID/local-dynamics feedback (Nguyen et al., 2026; Skifstad et al., 2026). SAS sequentially fits probes under preceding interventions with layer, strength, and quality selection (Hoppe et al., 2026); PERSONA uses vector algebra and Persona-Flow (Feng et al., 2026); ORBIT combines subspace rotations and tokenwise gates (Ghasemi et al., 2026); Pre-Control tracks attribute scores through a value network and target-error gradients (Zhang et al., 2026).

LoRA adapts low-rank parameters (Hu et al., 2021), while LoraHub combines adapters using new-task examples (Huang et al., 2024). FPSRG shares a post-trained model, constructs profiles through activation coeficients, records complete joint responses ofline, and retrieves by scores online. Both vectors and prompts can serve as configurations with measured coordinates.

## 2.3 Semantic Responses and Behavioral Measurement

MAP-Elites bins behavior descriptors and retains elites through mutation and within-bin competition; Quality-Diversity combines regional diversity and quality (Mouret & Clune, 2015; Pugh et al., 2016). FPSRG uses a qualityconstrained catalog, currently measuring a fixed inputcoeficient grid, with emphasis on LLM direction construction, joint calibration, and computation reuse. Sparse grids reduce high-dimensional approximation cost under conditions such as mixed smoothness (Bungartz & Griebel, 2004), suggesting routes to fewer measurements.

Strength, dynamics, and output quality determine usable intervention ranges (Hoppe et al., 2026; Taimeskhanov et al., 2026; Xu et al., 2026). Persona Dosing inverts targets through isotonic regression and interpolation (Jin et al., 2026); FPSRG calibrates complete multidimensional configurations and compares reachable regions. Generalized additive models and Belief Dynamics provide linkfunction and log-odds perspectives (Hastie & Tibshirani, 1986; Bigelow et al., 2026); we use these to construct saturation-aware single-axis predictions and evaluate the information added by joint measurement.

Personality inventories (Serapio-García et al., 2025), Act-TraitBench’s behavioral paradigms (Yang et al., 2026), and Li et al.’s Chinese situational judgments (Li et al., 2026) provide complementary observations. We evaluate withinsystem coverage, complete-generation remeasurement, and external expression separately, grounding interpretability in the correspondence between dimension definitions, internal operations, and measured responses.

## 3 The FPSRG Method

FPSRG performs shared first-person post-training, direction learning, joint calibration, and target retrieval. Applications specify dimensions, candidate layers, coeficients, and measurement interfaces; Appendices A–B give instance settings.

## 3.1 Joint Configuration within an Evaluation System

Let $M _ { \theta }$ have L decoder blocks and width H, with d dimension definitions:

$$
D _ { k } = ( \sigma _ { k } , e _ { k } ^ { - } , e _ { k } ^ { + } , \mathcal { X } _ { k } , \phi _ { k } ) , \qquad k \in [ d ] .\tag{1}
$$

Here $\sigma _ { k } , e _ { k } ^ { - } / e _ { k } ^ { + } , \mathcal { X } _ { k }$ , and $\phi _ { k }$ denote semantics, low/high endpoints, situations, and scoring rules. Protocol π fixes the model, items, encodings, sampling, and injection. The joint response to a is

$$
F _ { \pi } ( a ) = \mathbb { E } _ { Z \sim P _ { \pi } ( \cdot \vert M _ { \theta _ { \mathrm { F P } } } , a ) } [ \phi _ { \pi } ( Z ) ] \in \mathbb { R } ^ { d } .\tag{2}
$$

The catalog records complete score vectors, including each direction’s cross-dimensional efects. Training, development, and calibration sets support learning, layer selection, and measurement of $\widehat { F } _ { \pi }$ ; configurations are fixed before remeasurement.

Coeficients a and scores $F _ { \pi } ( a )$ occupy diferent coordinate systems. Raising an extraversion coeficient may change both extraversion and agreeableness. Retrieval uses the complete measured response rather than treating a coeficient as its corresponding trait score.

## 3.2 Shared First-Person Adaptation and Initialization

Shared post-training teaches the model to judge and answer as the participant. Records $( x , \rho , y )$ contain a situation, participant-perspective reasoning, and final words or actions; mathematical and long-reasoning examples also concern personal decisions. Mean cross-entropy supervises only current-turn reasoning, answers, and termination tokens. We update LoRA (Hu et al., 2021) or all language parameters, then freeze $M _ { \theta _ { \mathrm { F P } } }$ . Profiles share this model, with vectors supplying attribute diferences.

High/low examples share facts and difer in endpoint descriptions or responses. Let $H _ { \ell } ( x )$ be the residual $\mathrm { s e - }$ quence, $\mathcal { P } / \mathcal { P } _ { 0 }$ the direction/reference pooling operators, R the reference-norm statistic, and $h _ { \ell } ^ { \mathcal { P } } = \mathcal { P } ( H _ { \ell } )$ :

$$
\begin{array} { r l r } {  { \delta _ { k , \ell } = \frac { 1 } { n _ { k } | \mathcal { M } _ { k } | } \sum _ { i = 1 } ^ { n _ { k } } \sum _ { m \in \mathcal { M } _ { k } } \big [ h _ { \ell } ^ { \mathcal { P } } ( x _ { k , i , m } ^ { + } ) - h _ { \ell } ^ { \mathcal { P } } ( x _ { k , i , m } ^ { - } ) \big ] , } } \\ & { } & \\ & { s _ { k , \ell } = \mathcal { R } \Big ( \Big \{ \| h _ { \ell } ^ { \mathcal { P } _ { 0 } } ( x ) \| _ { 2 } : x \in \mathcal { D } _ { k } ^ { \mathrm { r e f } } \Big \} \Big ) , } \\ & { } & \\ & { v _ { k , \ell } ^ { ( 0 ) } = \rho _ { s } s _ { k , \ell } \frac { \delta _ { k , \ell } } { \| \delta _ { k , \ell } \| _ { 2 } } , } \\ & { } & { v _ { k , \ell } = \rho _ { s } s _ { k , \ell } u _ { k , \ell } , \quad \| u _ { k , \ell } \| _ { 2 } \leq 1 . } \end{array}\tag{3}
$$

Encoding index $m ,$ reference samples $\mathcal { D } _ { k } ^ { \mathrm { r e f } }$ , layer scale $s _ { k , \ell } ,$ and budget $\rho _ { s }$ determine the constraint. We optimize $u ,$ project it onto the unit ball after updates, and exclude zero contrasts. Pooling excludes padding and is fixed beforehand. Main non-thinking and HEXACO role directions use last-position pooling and median reference norms; sixdimensional thinking uses mean reasoning contrasts and mean reference norms. Refinement retains learned lengths within budget; Appendix I.1 specifies other baselines.

Equation (3) averages $n _ { k }$ paired examples and encodings $\mathcal { M } _ { k }$ , determines the semantic direction, then scales it relative to the layer’s reference states. Separate direction/reference pooling accommodates diferent interfaces under the same norm constraint.

## 3.3 Cognitive–Behavioral Direction Refinement

For non-thinking measurements, the unsteered high-pole probability $p _ { 0 , m }$ defines the soft target at dose α:

$$
q _ { \alpha , m } = \sigma ( \mathrm { l o g i t } [ \mathrm { c l i p } ( p _ { 0 , m } , \varepsilon , 1 - \varepsilon ) ] + \kappa \alpha ) .\tag{4}
$$

$\sigma$ is sigmoid, ε clips endpoints, and κ specifies a logodds shift relative to each item’s baseline. $\mathcal { L } _ { \mathrm { A B } }$ is binary cross-entropy on high-pole probability; $\mathcal { L } _ { \mathrm { s e l f } }$ weights length-normalized high/low self-report losses by ${ q } _ { \alpha , m }$ and its complement. Opposite doses supervise complete actions in situation c:

$$
{ \mathcal { L } } _ { \mathrm { a c t i o n } } = { \frac { 1 } { 2 } } \left[ { \overline { { \mathbf { C E } } } } ( y _ { B } ^ { + } \mid c , + \beta v ) + { \overline { { \mathbf { C E } } } } ( y _ { B } ^ { - } \mid c , - \beta v ) \right] .\tag{5}
$$

Neutral preservation compares unsteered and steered fullvocabulary distributions at identical teacher-forced prefixes. The combined objective is

$$
\begin{array} { r l } & { \mathcal { L } _ { \mathrm { v e c } } ^ { ( n ) } = \lambda _ { \mathrm { A B } } \mathcal { L } _ { \mathrm { A B } } + \lambda _ { \mathrm { s e l f } } \mathcal { L } _ { \mathrm { s e l f } } + \lambda _ { \mathrm { a c t i o n } } \mathcal { L } _ { \mathrm { a c t i o n } } } \\ & { \quad \quad \quad + \lambda _ { N } \chi _ { n } \mathcal { L } _ { \mathrm { n e u t r a l } } + \lambda _ { u } \Vert u \Vert _ { 2 } ^ { 2 } . } \end{array}\tag{6}
$$

The λ terms weight losses, $\chi _ { n }$ schedules neutral preservation at step $n ,$ and $\beta > 0$ is behavioral dose. Thinking

interfaces directly supervise complete questionnaire/action demonstrations, with $q \in \{ Q , B \}$ and $\epsilon \in \{ - 1 , + 1 \}$ :

$$
\begin{array} { l } { \displaystyle \mathcal { L } _ { \mathrm { t r a c e } } = \frac { 1 } { 4 } \sum _ { q , \epsilon } \left[ \lambda _ { \rho } \overline { { \mathrm { C E } } } ( \rho _ { q } ^ { \epsilon } \mid x _ { q } , \epsilon v ) \right. } \\ { \displaystyle \left. + \lambda _ { y } \overline { { \mathrm { C E } } } ( y _ { q } ^ { \epsilon } \mid x _ { q } , \rho _ { q } ^ { \epsilon } , \epsilon v ) \right] . } \end{array}
$$

Each segment is normalized by valid length; neutral and norm terms are added. Endpoint instructions generate demonstrations but are absent from vector-training inputs. Both interfaces freeze the model and optimize only directions. Development selection uses bidirectional responses or complete-answer endpoint diferences and accounts for invalid outputs (Appendix B).

## 3.4 Full-Sequence Joint Injection

The controller $\mathcal { V } = \{ ( \ell _ { k } , v _ { k } ) \} _ { k = 1 } ^ { d }$ sums same-layer directions:

$$
\begin{array} { l } { \displaystyle \Delta _ { \ell } ( a ) = \sum _ { k : \ell _ { k } = \ell } a _ { k } v _ { k } , } \\ { \displaystyle h _ { \ell , j } ^ { \prime } = h _ { \ell , j } + m _ { j } \Delta _ { \ell } ( a ) . } \end{array}\tag{7}
$$

$m _ { j } = 1$ at nonpadding positions. Refinement, calibration, and deployment cover inputs and target/generated positions, including reasoning tokens. Subsequent joint measurement records the resulting coupling.

## 3.5 Joint Measurement, Filtering, and Retrieval

Finite coeficient sets $G _ { k }$ define configurations and their response catalog:

$$
\begin{array} { l } { \displaystyle { \mathcal { G } = \prod _ { k = 1 } ^ { d } G _ { k } , \qquad N _ { G } = \prod _ { k = 1 } ^ { d } | G _ { k } | , } } \\ { \displaystyle { \mathcal { C } _ { \pi } = \{ ( a , \widehat { F } _ { \pi } ( a ) , \eta ( a ) ) : a \in \mathcal { G } \} . } } \end{array}\tag{8}
$$

η retains quality and measurement denominators. Scoring weights are fixed before construction. Both modes support complete generation: parse final answers and aggregate repeated scores. Reasoning, multi-token labels, and free-response evaluation use this interface.

In non-thinking mode, when the first position is the answer and every legal label is one token, one forward pass obtains the conditional distribution on $\mathcal { V } _ { i , m }$

$$
p _ { i , m , u } ^ { L } ( a ) = \frac { \exp ( z _ { i , m , u } ( a ) / T ) } { \sum _ { v \in \mathcal { V } _ { i , m } } \exp ( z _ { i , m , v } ( a ) / T ) } .\tag{9}
$$

$z$ denotes logits, $T$ the scoring temperature, and i, m, u items, encodings, and labels. After restoring scoring keys, expectations are averaged over encodings and item groups, replacing repeated sampling with one pass. Both interfaces retain full-input injection. Conditional expectations and generation means form separate catalogs; cross-interface use requires remeasurement (Appendix B.3).

First-token scoring accelerates answer acquisition without changing item counts or scoring rules. Complete generation also depends on decoding and answer validity. Recording these conditions ties reported scores to the actual calling protocol.

Joint perturbations may disrupt output, while legal-set normalization can hide probability elsewhere. The protocol fixes quality conditions $\mathcal { I } _ { \pi }$ and admissible set $\mathcal { G } _ { \mathrm { a d m } } = \{ a \in$ $\mathcal { G } ~ : ~ \forall r ~ \in ~ \mathcal { J } _ { \pi } , ~ g _ { \pi , r } ( \eta ( a ) ) ~ = ~ 1 \}$ before matching targets. The non-thinking cognitive–behavioral interface requires ≥98% mean legal mass per dimension/interface and checks numeric mappings, behavioral order, and label bias. Questionnaire-only evaluation uses the first two conditions; generation checks termination and answer format (Appendices B, I).

Candidates for target t satisfy maximum coordinate error $\tau _ { \infty }$ and total error τ<sub>1</sub>:

$$
\mathcal { G } _ { t } = \left\{ a \in \mathcal { G } _ { \mathrm { a d m } } : \Vert \widehat { F } _ { \boldsymbol { \pi } } ( a ) - t \Vert _ { \infty } \leq \tau _ { \infty } , \right\} .\tag{10}
$$

If nonempty, return its nearest-L1 configuration; otherwise return the nearest admissible point and an unmet-target status. Coverage is

$$
\operatorname { C o v } ( \mathcal { T } ; \tau _ { \infty } , \tau _ { 1 } ) = \frac { 1 } { | \mathcal { T } | } \sum _ { t \in \mathcal { T } } \mathbf { 1 } [ \mathcal { G } _ { t } \neq \emptyset ] .\tag{11}
$$

The service returns configurations, calibration scores, errors, and answers. Changed models or measurement protocols require recalibration.

## 3.6 Configuration Routing and Cache-Isolated Parallel Inference

Each single-GPU replica retains one model and precomputes per-layer increments for node p. Let $r ( j )$ identify the request owning the jth scheduled valid token:

$$
\begin{array} { l } { { \displaystyle { \mathcal B } _ { \ell , p } = \sum _ { k : \ell _ { k } = \ell } a _ { p , k } v _ { k } , } } \\ { { \displaystyle h _ { \ell , j } ^ { \prime } = h _ { \ell , j } + m _ { j } \cot \left( { \mathcal B } _ { \ell , p _ { r ( j ) } } \right) . } } \end{array}\tag{12}
$$

Same-layer increments are summed in FP32 and cast to hidden-state precision. Routing follows actual token order through prefill, reasoning, and answers. Diferent personalities and prompts share continuous batches, with new requests immediately replacing completed ones. Paged KV, hybrid sequence-state management, and chunked prefill reduce memory cost and interleave long prompts with decoding.

Cache identity binds the model, directions, protocol, node, prefix, positions, and mask:

$$
\kappa _ { r , j } = { \mathrm { H a s h } } \big ( \mathrm { i d } ( M _ { \theta _ { \mathrm { F P } } } , \mathcal { V } , \pi ) , p _ { r } , \ x _ { r , \le j } , \ \mathrm { p o s } _ { \le j } , \ m _ { \le j } \big ) .\tag{13}
$$

Intervened states are reused only under identical computation conditions. Earlier first-token measurements additionally reuse boundary states grouped by upstream coeficients (Appendix H). Replicas claim disjoint tasks and preserve request records; recovery does not resample committed results. Current execution uses eager mode and an FP32 output head, without counting disabled CUDA Graphs.

Matching prefixes within a configuration may reuse state; diferent configurations retain separate post-intervention states even for identical text. Mixed batches thus accommodate diferent nodes, prompts, and lengths while preserving each request’s control conditions.

## 3.7 Complementary Prompt and Vector Catalogs

Under the same protocol, measure vector $\mathcal { C } _ { V }$ , prompt $\mathcal { C } _ { P } .$ and simultaneous-control $\mathcal { C } _ { V P }$ catalogs. Catalog complementarity returns one configuration from $\mathcal { C } _ { V } \cup \mathcal { C } _ { P } ;$ withingeneration complementarity measures the increment from adding vectors under fixed text. Both rely on the corresponding measured configurations.

## 4 Joint Construction and Component Evaluation

## 4.1 Non-Thinking Five- and Ten-Dimensional Construction

Single-axis curves evaluate directional response, joint coverage evaluates reachability, and nearest-match MAE evaluates retrieval error. Calibration and coverage use the same complete calibration items; separate training/development items learn directions and select layers. Section 5 evaluates reuse on external questions.

Oficial Qwen3.5-27B (Qwen Team, 2026) has 64 decoder blocks and width 5,120. Shared LoRA is frozen after one pass over 1,000 participant-perspective examples, using “You are a person; answer according to the facts of the scenario.” Each dimension has 30/6/24 training/development/calibration questionnaire items and 16/4/8 behavioral scenarios. Questionnaires follow IPIP and Johnson (International Personality Item Pool, 2026; Johnson, 2014). Two numeric mappings and A/B orders are equally weighted. Self-report $Q$ and behavior B are converted to 0–100, with $S = ( Q + B ) / 2$ . We use $S$ unless specified; ablations separate components. Sensory sensitivity is Se.

Coverage. Broad tolerance requires one node with coordinate errors $\leq 2 0$ and total error $\leq 1 2 d ;$ strict tolerance requires ${ < } 1 0$ and $< 6 d .$ Coverage counts targets with an admissible match; ordinary-L1 success checks the returned node. Each domain uses 1,000 fixed targets. Thinking complementarity uses coordinatewise tolerance; Appendix C.5 supplements bin occupancy.

Five dimensions. O/C/E/A/N select layers 40/32/32/16/32 from eight candidates (zero-based), with increasing singleaxis scores at five doses. Coeficients $\{ - 2 , - 1 , 0 , 1 , 2 \}$ give 3,125 configurations and 1,000,000 first-token probes, retaining 1,854 nodes after four-part filtering. Broad coverage in 25–75 is 95.3%/92.2% before/after filtering; ordinary-L1 success is $7 6 . 1 \% ,$ versus 92.2% with feasible-first retrieval. Under the same metric, zero vectors cover 10.3% and targetindependent random singletons average 5.82% (Appendix C.6).

O/C/E/A/N denote openness, conscientiousness, extraversion, agreeableness, and neuroticism. Single-axis measurements vary one coeficient with others zero; the joint catalog measures complete five-dimensional responses to simultaneous directions.

Ten dimensions. Adding risk taking, future consequences, prosocial allocation, material values, and sensory sensitivity (Blais & Weber, 2006; Strathman et al., 1994; Murphy et al., 2011; Richins & Dawson, 1992; Aron & Aron, 1997) with the same training proportions and {−1, 0, 1} yields 59,049 configurations, 37,791,360 probes, and 33,942 qualified nodes. All ten single-axis responses increase. Broad coverage before filtering is 41.5%/ 69.3%/ 95.4% in 20– 80/25–75/30–70, versus 34.7%/58.6%/88.2% after filtering. Five preselected nodes produce 3,200 complete nonthinking answers: 99.56% are scoreable, with MAE 3.33 against catalog predictions (Appendix E).

## 4.2 Thinking-Mode Six-Dimensional Individuals and Interaction

Huihui Qwen3.8-27B-abliterated uses a 64-layer, width-5,120 Qwen3.5 backbone. We train all language parameters for two epochs on 1,500 first-person examples, then freeze them and learn agreeableness, conscientiousness, neuroticism, risk taking, material values, and future consequences (A/C/N/R/M/T), without loading LoRA. Selected layers are 24/24/32/40/24/24. Three levels {−1, 0, 1} yield 729 configurations. We generate reasoning and answers but score only answers: 24 questionnaire and eight behavioral items per dimension, two repeats per encoding, give 559,872 outputs; 558,095 are format-valid (99.68%). Requiring 126/128 valid answers in all dimensions retains 663 nodes (Appendices A.4, B.5).

Personality responses. Eight new Chinese scenarios per dimension use low/high-decile nodes chosen to minimize other-dimension diferences, plus middle and zero-vector controls. Forty-eight scenarios × four configurations × no emotion/joy/anger × three samples produce 1,728 complete responses without personality or emotion text prompts. DeepSeek blindly rates final answers and reasoning on a 0– 4 scale without configuration labels; 1,702 records receive valid ratings. Same-scenario, same-seed pairs are averaged over three repeats, then equally over fully rated scenarios.

Without emotion, A/ C/ N/ R/ M/ T high-minus-low mean final-answer diferences are 0.11/0.17/0.25/0.95/0.96/0.62; reasoning diferences are 0.06/0.21/0.25/0.76/0.96/0.62, using 6/8/8/7/8/7 paired scenarios. All six paired means follow catalog directions, describing selected multidimensional configurations whose remaining coordinates are minimized in diference, not exactly matched.

Emotion overlay. With personality fixed, joy/anger vectors increase corresponding final-answer ratings by 0.23/0.19 and reasoning ratings by 0.27/0.16 over no emotion, using 33/37 fully paired scenarios. Both means follow the intended direction, supporting current-state adjustment over individual profiles.

Personality scores select participants; emotion ratings measure their current expression. These are separate measurements. Paired mean changes in this scenario set show configuration diferences in answers and generated reasoning without additional personality reminders.

## 4.3 Single-Axis Prediction versus Joint Calibration

We evaluate node-score prediction and target-retrieval errors when joint measurements are omitted.

## 4.3.1 CONSTRUCTING SINGLE-AXIS PREDICTIONS

A five-dimensional, five-level grid contains 3,125 configurations but only $1 + 5 \times 4 = 2 1$ anchors: zero and four nonzero doses per direction. Each anchor records all five scores, retaining cross-dimensional main efects. For unit coordinate $e _ { k }$

$$
\widehat F _ { \mathrm { a d d } } ( \boldsymbol a ) = F ( \boldsymbol 0 ) + \sum _ { \boldsymbol k = 1 } ^ { d } \left[ F ( a _ { \boldsymbol k } \boldsymbol e _ { \boldsymbol k } ) - F ( \boldsymbol 0 ) \right] .
$$

Measured single-axis values are added across directions. We compare direct addition, clipping to 0–100, and addition in logit space followed by sigmoid (Hastie & Tibshirani,

1986; Bigelow et al., 2026), without reading joint nodes. Prompts use all-middle descriptions as the origin and vary one dimension at a time.

## 4.3.2 INFORMATION ADDED BY JOINT MEASUREMENT

Prediction MAE evaluates node scores; target MAE evaluates retrieved scores against requests, averaging over nodes/ targets and dimensions (Appendix C.2). Nonthinking vectors and prompts use 1,836 and 3,104 qualified joint nodes. Direct/ clipped/ logit prediction MAEs are 6.325/6.246/5.244 and 5.304/5.068/3.973. On the same 1,000 targets in 20–80, ordinary-L1 target MAE changes from 9.617 to 7.076 for vector logit prediction versus measured lookup, and from 8.899 to 7.346 for prompts. Joint measurements provide more accurate retrieval coordinates.

A separate four-dimensional thinking catalog holds out each of three generation repeats, calibrating on two and remeasuring on the third (Appendix C.3).

## 4.3.3 IMPLICATIONS FOR CATALOG CONSTRUCTION

Joint calibration makes combination responses omitted by single-axis predictions available for retrieval. New evaluation systems reuse this workflow by specifying controls and scoring, without first fitting cross-dimensional relationships.

## 4.4 Complementary Reachable Regions of Prompts and Vectors

Catalog complementarity. The same non-thinking fivedimensional interface measures five-level vectors (−2 to 2), seven-level vectors (−3 to 3), and extremely low/low/middle/high/extremely high prompts. Table 1 uses the same 1,000 targets and four-part filtering; catalog union returns one configuration per target.

Five-level vectors plus prompts use 6,250 configurations yet exceed the broad coverage of 16,807 seven-level vectors. In 25–75, strict vector coverage is 44.0%, prompt coverage 7.4%, and union 48.9%. Four-dimensional thinking catalogs cover 9.16%/15.29%/22.03% with vectors/prompts/ union on 2,401 targets in 20–80 at coordinate tolerance ≤10 (Appendix C.4).

Within-generation complementarity. Adding signed vectors to fixed four-dimensional thinking endpoint prompts moves all eight endpoints further in the intended direction; all four conditions exceed 97% validity. Appendix C.7 preserves complete scores from 384 responses per condition. Catalog union extends reachability; simultaneous addition tests joint control under fixed text.

Table 1. Complementary coverage (%). Broad tolerance is coordinatewise 20 and total 60; strict tolerance is <10 and <30.
<table><tr><td colspan="6">Candidate</td></tr><tr><td>catalog</td><td>Configurations</td><td>20–80 broad</td><td>25–75 broad</td><td>20–80 strict</td><td>25–75 strict</td></tr><tr><td>Vector-5</td><td>3,125</td><td>82.7%</td><td>92.2%</td><td>32.1%</td><td>44.0%</td></tr><tr><td>Vector-7</td><td>16,807</td><td>89.7%</td><td>96.5%</td><td>41.0%</td><td>55.2%</td></tr><tr><td>Prompt-5</td><td>3,125</td><td>92.0%</td><td>88.7%</td><td>13.0%</td><td>7.4%</td></tr><tr><td>Vector-5 ∪ Prompt-5</td><td>6,250</td><td>97.0%</td><td>98.7%</td><td>42.3%</td><td>48.9%</td></tr><tr><td>Vector-7 U Prompt-5</td><td>19,932</td><td>98.7%</td><td>99.7%</td><td>49.9%</td><td>59.2%</td></tr></table>

## 4.5 Efects of Data Organization, Refinement, and Injection Scope

Data and refinement. The gradient-free comparison uses identical questionnaire/behavior targets, averages high–low content activations, combines groups equally, and matches refined norms. With layers, injection, and items fixed, mean −2-to+2 composite span increases from 30.75 to 40.45, improving all five dimensions; agreeableness questionnaire span changes from −10.82 to +16.79. Direct extraction has larger A/N behavioral spans (Appendix D.2).

Cognitive and behavioral supervision. Holding initialization, layers, 240 updates, and full-sequence injection fixed, we vary only supervision. Joint training achieves the highest composite coverage while preserving both interfaces’ reachability (Table 2).

Injection scope. With common data and layer-selection rules, mean questionnaire span is 7.88 for response-side injection and 27.16 for input-plus-response injection, larger on all five axes. Both are trained and layer-selected separately, comparing complete strategies (Appendix D.4).

## 4.6 Grid Density and Coverage Gains within a Fixed Range

Fixing the model, five directions, full-sequence injection, and 320-probe interface, we vary only density: three levels {−2, 0, 2}, five {−2, −1, 0, 1, 2}, and nine at 0.5 intervals from −2 to 2. The nested sets contain 243/3,125/59,049 nodes. Old measurements are reused, new nodes added, and identical 1,000 targets and filtering applied.

In 25–75, three to five levels adds 21.0 percentage points of broad coverage; five to nine adds 2.4. The latter raises strict coverage by 10.6 points and reduces MAE by 15.1%, using 18.90 times as many nodes. Density improves fine matching; range expansion and prompts add response regions, without retraining directions.

Nine levels reuse 17,308 nodes and add 41,741 with 16 replicas in about ten hours including aggregation. Paired benchmarks retaining identical batches, precision, and operator optimizations show warm cross-node-reuse speedups of 2.76/2.77 and fill-inclusive speedup 1.74, with identical probabilities and legal mass (Appendix H).

## 5 External Scenario Transfer and Control-Method Comparisons

We test external configuration reuse and compare controls on a common backbone and budget.

## 5.1 Scenario Transfer on Chinese PersonalityBench

We translate the 459 PersonalityBench scenarios used by PERSONA (Feng et al., 2026) into Chinese, preserving identities and meaning: 90/89/90/90/100 items for O/C/E/ A/N. Section 4’s model, LoRA, vectors, and full-sequence injection are unchanged. Among 1,854 jointly qualified nodes, each dimension selects minimum/maximum S and the node nearest their midpoint, without constraining other dimensions or reselecting after generation.

Thinking is disabled and no personality prompts are added, producing 1,377 responses. DeepSeek sees only questions and responses, blindly rating 1–5 trait intensity without configurations or calibration scores. This independent scenario scale is not ranked against the original leaderboard. Interpretable evidence within 512 tokens is scored; 32 records lack trait evidence and are excluded, with all others retained. Table 4 reports denominators (Appendix F.3).

High exceeds low on all five dimensions. O/ C/ E/ N increase across three levels; A’s middle slightly exceeds high. Same-item, scoreable high–low pairs average 2.14/2.52/2.85/2.16/1.57, showing same-direction expression of catalog diferences in external free responses.

## 5.2 Single-Axis Control Methods on a Shared Backbone

Public IPIP HEXACO covers H/E/X/A/C/O, with 240 items split by facet into 144 training, 48 development, and 48 held-out items. CAA (Rimsky et al., 2024), an independent probe (Hoppe et al., 2026), role initialization, adapted PER-SONA (Feng et al., 2026), BiPO (Cao et al., 2024), and refinement share Qwen3.5-27B, first-person LoRA, candidate layers, and full-sequence injection. Role initialization uses last-position contrasts and median reference norms. Direct vectors have length 0.1 times the reference scale; BiPO/refinement retain learned lengths within budget. Methods select layers independently on development items; held-out items do not enter optimization. Refinement here uses questionnaires only. The probe corresponds to SAS’s singleaxis component; other adaptations appear in Appendix I.1.

Table 2. Fixed-layer supervision coverage (%). The same 10,000 targets in $[ 2 5 , 7 5 ] ^ { 5 }$ use filtering and coordinatewise 20/total $\leq 6 0$ tolerances.
<table><tr><td>Refinement target</td><td>Q coverage</td><td>B coverage</td><td>S coverage</td></tr><tr><td>Q only</td><td>78.76</td><td>65.75</td><td>85.64</td></tr><tr><td>B only</td><td>58.87</td><td>84.77</td><td>82.32</td></tr><tr><td>Joint QB</td><td>77.32</td><td>80.81</td><td>91.85</td></tr></table>

Table 3. Density, coverage, and matching error. Five-dimensional tolerances follow Table 1; MAE is per-axis error of ordinary-L1 retrieval.
<table><tr><td></td><td>Levels/axis</td><td>Qualified/total nodes</td><td>Broad coverage</td><td>Strict coverage</td><td>Per-axis MAE</td></tr><tr><td>Target domain</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>25—75</td><td>3</td><td>120/243</td><td>71.2%</td><td>15.6%</td><td>8.01</td></tr><tr><td>25—75 25—75</td><td>5</td><td>1,854/3,125</td><td>92.2%</td><td>44.0%</td><td>5.78</td></tr><tr><td>20—80</td><td>9</td><td>38,932/59,049 120/243</td><td>94.6% 59.1%</td><td>54.6% 10.5%</td><td>4.91</td></tr><tr><td>2080</td><td>3 5</td><td>1,854/3,125</td><td>82.7%</td><td>32.1%</td><td>9.14 7.08</td></tr><tr><td>20—80</td><td>9</td><td>38,932/59,049</td><td>87.2%</td><td>40.1%</td><td>6.28</td></tr></table>

With thinking disabled, eight held-out items × two mappings × three samples give 48 complete answers per dimension/strength. Keyed 1–5 ratings are converted to 0–100 and averaged equally. PERSONA’s negative half-axis uses its independent low-pole vector.

Mean −2-to+2 endpoint diferences are 48.75 for refinement, 36.20 initialization, 27.26 BiPO, 23.13 PERSONA, 20.23 probe, and 19.10 CAA. Refinement leads on X/A/ C/O and initialization on H/E. Agreeableness spans 25.00– 90.63 after refinement versus 58.33–94.79 initially, primarily extending downward control.

Extending both directions until complete-answer validity falls below 90%, the observed valid-point ranges before first failure average 50.95/45.14/41.23/37.72/29.99/29.47 in that method order. Additional boundary checks leave refinement highest at 50.78 (Appendix I.2–I.3).

## 5.3 Joint Coverage at a Matched Measurement Budget

With directions/ layers fixed, each method measures 729 combinations at −1/0/+1. Forty-eight held-out items × two mappings give 96 first-token conditional-expectation probes per node. Each dimension requires ≥98% mean legal mass and ${ \leq } 2 5$ mapping-score diference. On the same 10,000 targets in $[ 2 5 , 7 5 ] ^ { 6 }$ , strict coverage requires per-axis error <10 and total <36; MAE is per-axis error of nearest-L1 retrieval.

Refinement improves strict coverage by 38.3% and reduces MAE by 6.6% relative to initialization. Unfiltered coverage also leads (13.48% versus 9.37%). It attains the highest strict coverage and lowest error in 20–80 and 30–70 (Appendix I.4), improving both single-axis span and joint matching under the common protocol.

## 5.4 Interactive Application with Heterogeneous Market Participants

We connect the thinking-mode four-dimensional catalog to a multi-turn bargaining environment, turning statically calibrated configurations into market participants that observe, deliberate, and act. Four vector-only profiles are selected by relatively low/high measured risk taking and material values, without personality prompts or separate model training for each individual. Frozen 27B participants negotiate price, delivery, and warranty with original or fine-tuned 9B sellers in the same ten trading worlds, producing 80 episodes. Configurations share product facts, budgets, and utility rules. Eight worlds have one buyer and one seller; two have two buyers and two sellers, with the other frozen participants also using the assigned configuration.

Heterogeneity appears in the trade-ofs made during negotiation. Participants can maintain an ofer, seek better terms, accept a certain return, or leave; generated reasoning and structured actions record these choices together. In the ofice-headset world with the fine-tuned seller, two highermaterial-value profiles face identical valid ofers: the seller requests CNY 100 with seven-day delivery and no warranty, while the buyer’s outstanding ofer is CNY 100 with threeday delivery and a twelve-month warranty. The lower-risk profile waits, explaining, “If s0 rejects or makes a new ofer, I will decide then” (translated from Chinese). The higherrisk profile retains its delivery and warranty requirements but lowers its ofer to CNY 90, probing for better terms. Both ultimately trade on the existing CNY 100 contract, but take diferent negotiation paths: one maintains its proposal while the other actively seeks a larger return.

Table 4. Chinese PersonalityBench transfer. Calibration triples denote low/middle/high; parentheses give valid rating counts.
<table><tr><td>Dimension</td><td>Calibration scores (0-100)</td><td>Low scenario rating</td><td>Middle scenario rating</td><td>High scenario rating</td></tr><tr><td>Openness O</td><td>11.33/46.92/82.58</td><td>1.63 (87)</td><td>2.72 (85)</td><td>3.81 (80)</td></tr><tr><td>Conscientiousness</td><td>11.65/47.78/83.96</td><td>1.71 (87)</td><td>3.33 (85)</td><td>4.22 (88)</td></tr><tr><td>C Extraversion E</td><td>8.22/46.26/84.30</td><td>1.21 (90)</td><td>2.60 (90)</td><td>4.05 (87)</td></tr><tr><td>Agreeableness A</td><td>32.04/57.55/83.08</td><td>1.96 (90)</td><td>4.28 (90)</td><td>4.12 (89)</td></tr><tr><td>Neuroticism N</td><td>16.32/46.78/77.18</td><td>3.38 (98)</td><td>4.23 (99)</td><td>4.95 (100)</td></tr></table>

![](images/8c3ebd52546973ab236afa979c86644a036627abeeb4b8e5d7fdb4bf01048639.jpg)  
Figure 1. Complete-answer curves for six methods over a fixed interval. Means use parseable answers; refined X/C/O have missing responses at some strong endpoints. Appendix I.2–I.3 reports extensions and repeats.

Configurations also express diferent demands for transaction terms. For the same product, another profile says at the opening that “a warranty is optional” (translated) and offers CNY 110 with three-day delivery and no warranty; the higher-material-value profile above retains twelve months of warranty in its subsequent deliberation. These records illustrate local diferences in service demands, waiting, and counterofers among callable individuals. They are interaction cases: service demands are not equated with material values, and no single action is treated as a fixed rule for high or low risk taking. Behavior emerges jointly from the configuration, economic conditions, and dialogue history.

This application connects score-based selection, participant invocation, negotiation, and settlement. Users can choose individuals with diferent calibrated characteristics from one model’s catalog, construct a heterogeneous population in a shared environment, and inspect their reasons, actions, and outcomes. Fine-tuning the 9B seller on interaction data increases mean contractual utility from CNY 13.835 to 20.725 (+49.8%) across forty evaluations in the same ten test worlds.

## 6 Conclusion and Future Work

FPSRG combines shared first-person post-training, direction learning, and joint measurement to construct heterogeneous profiles retrievable by multidimensional scores. Non-thinking five/ ten-dimensional and thinking six-dimensional instances implement the workflow. Data, refinement, and injection comparisons support single-axis improvements; the public six-dimensional experiment also shows joint-coverage and matching gains. Joint calibration records combination responses, density improves precision, prompts extend reachability, and token routing with isolated caches supports parallel use.

Table 5. Six-dimensional joint results at 729 nodes per method. Scores follow quality filtering.
<table><tr><td>Construction</td><td>Qualified nodes</td><td>Strict coverage (%)↑</td><td>Matching MAE↓</td></tr><tr><td>CAA</td><td>561</td><td>4.94</td><td>10.20</td></tr><tr><td>Independent probe</td><td>393</td><td>4.23</td><td>10.81</td></tr><tr><td>Role initialization</td><td>709</td><td>9.08</td><td>8.70</td></tr><tr><td>Adapted PERSONA</td><td>728</td><td>5.31</td><td>10.04</td></tr><tr><td>BiPO</td><td>617</td><td>3.95</td><td>10.39</td></tr><tr><td>Refined (ours)</td><td>570</td><td>12.56</td><td>8.13</td></tr></table>

Chinese PersonalityBench separates high/ low profiles in the intended direction on all five axes, with three-level increases on four. Applications can define assessments, build catalogs, and invoke configurations in interactive simulations. Future work addresses fewer measurements and cross-dimensional response mechanisms.

## 7 Ethics, Scope, and Reproducibility

We construct simulated participants; deployments should disclose model identity. Scores are not for real-person diagnosis or nonconsensual deception/manipulation. Catalog coverage, same-item remeasurement, and external scenarios evaluate diferent interfaces. Interpretability refers to traceable dimensions, operations, and responses; generated reasoning need not faithfully describe hidden computation (Turpin et al., 2023).

Thinking and non-thinking use separate catalogs. Changed models, questions, or scoring require recalibration or independent checks. Reproduction materials contain splits, training/layer selection, model/direction identities, scoring/ filtering, node outputs, and sources. Resources follow their licenses; formal submission will include public archives, contributions, funding, and applicable review information.

## References

Ivana Anusic, Ulrich Schimmack, Rebecca T. Pinkus, Penelope Lockwood. 2009. The Nature and Structure of Correlations Among Big Five Ratings: The Halo-Alpha-Beta Model. Journal of Personality and Social Psychology 97(6), 1142–1156. DOI: 10.1037/a0017159.

Lisa P. Argyle, Ethan C. Busby, Nancy Fulda, Joshua R. Gubler, Christopher Rytting, David Wingate. 2023. Out of One, Many: Using Language Models to Simulate Human Samples. Political Analysis. DOI: 10.1017/ pan.2023.2.

Elaine N. Aron, Arthur Aron. 1997. Sensory-processing sensitivity and its relation to introversion and emotionality. Journal ofPersonality and Social Psychology, 73(2), 345–368. DOI: 10.1037/0022-3514.73.2.345.

Eric Bigelow, Daniel Wurgaft, YingQiao Wang, Noah Goodman, Tomer Ullman, Hidenori Tanaka, Ekdeep Singh Lubana. 2026. Belief Dynamics Reveal the Dual Nature of In-Context Learning and Activation Steering. arXiv:2511.00617v2, revised 12 March 2026; first submitted November 2025. DOI: 10.48550/ arXiv.2511.00617.

Ann-Renée Blais, Elke U. Weber. 2006. A Domain-Specific Risk-Taking (DOSPERT) scale for adult populations. Judgment and Decision Making, 1(1), 33–47. DOI: 10.1017/S1930297500000334.

Hans-Joachim Bungartz, Michael Griebel. 2004. Sparse grids. Acta Numerica 13, 147–269. DOI: 10.1017/ S0962492904000182.

Yuanpu Cao, Tianrong Zhang, Bochuan Cao, Ziyi Yin, Lu Lin, Fenglong Ma, Jinghui Chen. 2024. Personalized Steering of Large Language Models: Versatile Steering Vectors Through Bi-directional Preference Optimization. Advances in Neural Information Processing Systems 37. DOI: 10.52202/079017-1567.

Sumanth Dathathri, Andrea Madotto, Janice Lan, Jane Hung, Eric Frank, Piero Molino, Jason Yosinski, Rosanne Liu. 2019. Plug and Play Language Models: A Simple Approach to Controlled Text Generation. arXiv:1912.02164.

Colin G. DeYoung. 2006. Higher-Order Factors of the Big Five in a Multi-Informant Sample. Journal of Personality and Social Psychology 91(6), 1138–1151. DOI: 10.1037/0022-3514.91.6.1138.

Chengyu Du, Xintao Wang, Aili Chen, Weiyuan Li, Rui Xu, Junteng Liu, Zishan Huang, Rong Tian, Zijun Sun, Yuhao Li, Liheng Feng, Deming Ding, Pengyu

Zhao, Yanghua Xiao. 2026. HER: Human-like Reasoning and Reinforcement Learning for LLM Role-playing. Findings of ACL 2026, 25725–25762. DOI: 10.18653/ v1/2026.findings-acl.1283.

Xiachong Feng, Liang Zhao, Weihong Zhong, Yichong Huang, Yuxuan Gu, Lingpeng Kong, Xiaocheng Feng, Bing Qin. 2026. PERSONA: Dynamic and Compositional Inference-Time Personality Control via Activation Vector Algebra. International Conference on Learning Representations 2026.

Hallvard Føllesdal, Christopher J. Soto. 2022. The Norwegian Adaptation of the Big Five Inventory-2. Frontiers in Psychology 13:858920. DOI: 10.3389/ fpsyg.2022.858920.

Narges Ghasemi, Amir Ziashahabi, Salman Avestimehr, Jonathan May. 2026. ORBIT: Training-Free Multi-Attribute Behavioral Steering via Orthogonal Subspace Rotation. arXiv:2606.22357v2.

Trevor Hastie, Robert Tibshirani. 1986. Generalized Additive Models. Statistical Science 1(3), 297–310. DOI: 10.1214/ss/1177013604.

Florian Hoppe, David Khachaturov, Robert D. Mullins, Mark Huasong Meng. 2026. Controllable and Explainable Personality Sliders for LLMs at Inference Time. Proceedings of the 43rd International Conference on Machine Learning, Proceedings of Machine Learning Research 306, 43871–43895.

Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, Weizhu Chen. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Chengsong Huang, Qian Liu, Bill Yuchen Lin, Tianyu Pang, Chao Du, Min Lin. 2024. LoraHub: Eficient Cross-Task Generalization via Dynamic LoRA Composition. arXiv:2307.13269v3; COLM 2024. DOI: 10.48550/arXiv.2307.13269.

International Personality Item Pool. 2026. IPIP items and scoring keys for the 30 facets of the NEO-PI-R; Johnson 120-item selection. Oficial public-domain item and scoring-key resource; accessed 30 September 2026.

Zehao Jin, Junran Wang, Ruixuan Deng, Jiahao Chen, Jingyuan Zhang, Yuxuan Zhang, Xinjie Shen. 2026. Persona Dosing: Calibrated Activation Steering for Graded Trait Control. arXiv:2609.36388v1, 28 September 2026. DOI: 10.48550/arXiv.2609.36388.

John A. Johnson. 2014. Measuring thirty facets of the Five Factor Model with a 120-item public domain inventory: Development of the IPIP-NEO-120. Journal

of Research in Personality 51, 78–89. DOI: 10.1016/ j.jrp.2014.05.003.

Chang-Jin Li, Jiyuan Zhang, Yun Tang, Jian Li. 2026. Automatic item generation for personality situational judgment tests with large language models. Computers in Human Behavior Reports, 21, Article 100964. DOI: 10.1016/j.chbr.2026.100964.

Kenneth Li, Oam Patel, Fernanda Viégas, Hanspeter Pfister, Martin Wattenberg. 2023. Inference-Time Intervention: Eliciting Truthful Answers from a Language Model. Advances in Neural Information Processing Systems 36. DOI: 10.52202/075280-1797.

Christina Lu, Jack Gallagher, Jonathan Michala, Kyle Fish, Jack Lindsey. 2026. The Assistant Axis: Situating and Stabilizing the Default Persona of Language Models. arXiv:2601.10387.

Suhong Moon, Marwa Abdulhai, Minwoo Kang, Joseph Suh, Widyadewi Soedarmadji, Eran Kohen Behar, David M. Chan. 2024. Virtual Personas for Language Models via an Anthology of Backstories. Proceedings of EMNLP 2024. DOI: 10.18653/v1/2024.emnlp-main.1110.

Jean-Baptiste Mouret, Jef Clune. 2015. Illuminating search spaces by mapping elites. arXiv:1504.04909.

Ryan O. Murphy, Kurt A. Ackermann, Michel J. J. Handgraaf. 2011. Measuring Social Value Orientation. Judgment and Decision Making, 6(8), 771–781. DOI: 10.1017/S1930297500004204.

Dung Viet Nguyen, Yen Pham, Hieu Vu, Lei Zhang, Tan Nguyen. 2026. Activation Steering with a Feedback Controller. International Conference on Learning Representations 2026.

Duy Nguyen, Archiki Prasad, Elias Stengel-Eskin, Mohit Bansal. 2025. Multi-Attribute Steering of Language Models via Targeted Intervention. Proceedings of ACL 2025, Volume 1: Long Papers, 20619–20634. DOI: 10.18653/v1/2025.acl-long.1007.

Bo Ni, Yu Wang, Leyao Wang, et al. 2026. A Survey on LLM-based Conversational User Simulation. Proceedings ofEACL 2026, Volume 1: Long Papers, 4266–4301. DOI: 10.18653/v1/2026.eacl-long.200.

Narmeen Fatimah Oozeer, Luke Marks, Fazl Barez, Amir Abdullah. 2025. Beyond Linear Steering: Unified Multi-Attribute Control for Language Models. Findings of EMNLP 2025, 23513–23557. DOI: 10.18653/ v1/2025.findings-emnlp.1278.

Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, Michael S. Bernstein. 2023.

Generative Agents: Interactive Simulacra of Human Behavior. arXiv:2304.03442.

Joon Sung Park, Carolyn Q. Zou, Jonne Kamphorst, Niles Egan, Aaron Shaw, Benjamin Mako Hill, Carrie Cai, Meredith Ringel Morris, Percy Liang, Robb Willer, Michael S. Bernstein. 2026. LLM Agents Grounded in Self-Reports Enable General-Purpose Simulation of Individuals. arXiv:2411.10109v3.

Justin K. Pugh, L. B. Soros, Kenneth O. Stanley. 2016. Quality Diversity: A New Frontier for Evolutionary Computation. Frontiers in Robotics and AI 3:40. DOI: 10.3389/frobt.2016.00040.

Qwen Team. 2026. Qwen3.5-27B: Oficial Model Card and Configuration. Hugging Face model repository; accessed 13 October 2026.

Marsha L. Richins, Scott Dawson. 1992. A Consumer Values Orientation for Materialism and Its Measurement: Scale Development and Validation. Journal ofConsumer Research, 19(3), 303–316. DOI: 10.1086/209304.

Nina Rimsky, Nick Gabrieli, Julian Schulz, Meg Tong, Evan Hubinger, Alexander Turner. 2024. Steering Llama 2 via Contrastive Activation Addition. Proceedings of ACL 2024, Volume 1: Long Papers, 15504–15522. DOI: 10.18653/v1/2024.acl-long.828.

Daniel Scalena, Gabriele Sarti, Malvina Nissim. 2024. Multi-property Steering of Large Language Models with Dynamic Activation Composition. Proceedings of the 7th BlackboxNLP Workshop, 577–603. DOI: 10.18653/ v1/2024.blackboxnlp-1.34.

Gregory Serapio-García, Mustafa Safdari, Clément Crepy, Luning Sun, Stephen Fitz, Peter Romero, Marwa Abdulhai, Aleksandra Faust, Maja Matarić. 2025. A psychometric framework for evaluating and shaping personality traits in large language models. Nature Machine Intelligence 7, 1954–1968. DOI: 10.1038/s42256-025-01115- 6.

Julian Skifstad, Xinyue Annie Yang, Glen Chou. 2026. Local Linearity of LLMs Enables Activation Steering via Model-Based Linear Optimal Control. Proceedings of ICML 2026, Proceedings ofMachine Learning Research 306, 113885–113919.

Alan Strathman, Faith Gleicher, David S. Boninger, C. S. Edwards. 1994. The consideration of future consequences: Weighing immediate and distant outcomes of behavior. Journal of Personality and Social Psychology, 66(4), 742–752. DOI: 10.1037/0022-3514.66.4.742.

Magamed Taimeskhanov, Samuel Vaiter, Damien Garreau. 2026. Towards Understanding Steering Strength. Proceedings ofICML 2026, Proceedings ofMachine Learning Research 306, 118084–118133.

Miles Turpin, Julian Michael, Ethan Perez, Samuel R. Bowman. 2023. Language Models Don’t Always Say What They Think: Unfaithful Explanations in Chain-of-Thought Prompting. arXiv:2305.04388.

Noah Wang, Z. Y. Peng, Haoran Que, et al. 2024. RoleLLM: Benchmarking, Eliciting, and Enhancing Role-Playing Abilities of Large Language Models. Findings of ACL 2024, 14743–14777. DOI: 10.18653/ v1/2024.findings-acl.878.

Kuang Wang, Xianfei Li, Shenghao Yang, Li Zhou, Feng Jiang, Haizhou Li. 2025. Know You First and Be You Better: Modeling Human-Like User Simulators via Implicit Profiles. Proceedings of ACL 2025, Volume 1: Long Papers, 21082–21107. DOI: 10.18653/ v1/2025.acl-long.1025.

Xiaoting Wu, Yi Huang, Chunyang Gao, Mengfei Guo, Jingyu Yao, Junlan Feng. 2026. Thinking Alignment of Scenario-Oriented User Simulation. Proceedings of ACL 2026, Volume 1: Long Papers, 42932–42945. DOI: 10.18653/v1/2026.acl-long.1989.

Zhengxuan Wu, Aryaman Arora, Zheng Wang, Atticus Geiger, Dan Jurafsky, Christopher D. Manning, Christopher Potts. 2024. ReFT: Representation Finetuning for Language Models. Advances in Neural Information Processing Systems 37. DOI: 10.52202/079017-2041.

Ziwen Xu, Chenyan Wu, Hengyu Sun, Haiwen Hong, Mengru Wang, Yunzhi Yao, Longtao Huang, Hui Xue, Shumin Deng, Zhixuan Chu, Huajun Chen, Ningyu Zhang. 2026. Why Steering Works: Toward a Unified View of Language Model Parameter Dynamics. Proceedings of ACL 2026, Volume 1: Long Papers, 31719– 31736. DOI: 10.18653/v1/2026.acl-long.1463.

Kevin Yang, Dan Klein. 2021. FUDGE: Controlled Text Generation With Future Discriminators. Proceedings of NAACL-HLT 2021, 3511–3535. DOI: 10.18653/ v1/2021.naacl-main.276.

Yutong Yang, Chenxi Miao, Weikang Li, Yunfang Wu. 2026. ActTraitBench: Quantifying the Knowledge-Decision Gap in Large Language Models via Human-Grounded Behavioral Validation. arXiv:2605.29791v2; accepted to Findings of EMNLP 2026. DOI: 10.48550/ arXiv.2605.29791.

Rongzhi Zhang, Liqin Ye, Yuzhao Heng, Xiang Chen, Tong Yu, Lingkai Kong, Sudheer Chava, Chao Zhang.

2026. Precise Attribute Intensity Control in Large Language Models via Targeted Representation Editing. arXiv:2510.12121v2, revised 18 February 2026; first submitted 14 October 2025. DOI: 10.48550/ arXiv.2510.12121.

## Appendices

The following appendices document data, optimization, scoring, coverage, remeasurement, external evaluation, and execution. Training, development, calibration, and independent repeated generation serve diferent roles throughout.

## Appendix A Data, Dimensions, and Shared Adaptation

## A.1 Data Roles and Dimension Definitions

Each dimension contains 60 questionnaire items, split into 30 for direction learning, six for development selection, and 24 for joint calibration. Twenty-eight behavioral scenarios are split 16/4/8, with event-family checks across splits. Ten dimensions therefore comprise 600 questionnaire items and 280 behavioral scenarios. Big Five items follow the IPIP framework; the additional constructs use the references in Section 4.1. Chinese items operationalize these constructs for this study.

The high poles are: O, exploring new experiences and ideas; C, fulfilling responsibilities and checking work; E, initiating social interaction and expression; A, cooperation and consideration with personal boundaries; N, worry and emotiona reactivity under stress; R, accepting uncertainty with specified probabilities and afordable losses; T, prioritizing delayed outcomes; V, voluntarily exchanging limited personal benefit for others’ or collective benefit; M, relating possessions to goals, happiness, and success; and sensory Se, sensitivity to sound, light, smell, touch, and concurrent stimulation. Se denotes the sensory trait; S denotes only the composite score.

Scores are model responses under the specified tasks, not population-norm diagnoses. Direction learning and calibration use diferent items. Exposure during shared post-training is recorded separately, including the six-dimensional consumer instance’s overlap in Appendix A.4. Catalog scores are distinguished from subsequent independent generations.

## A.2 First-Person Adaptation

The shared dataset contains 1,000 unique targets in 151 dialogue groups: 280 everyday-life and negotiation examples, 397 mathematics and extended-reasoning examples, and 323 participant decisions. Former tool interactions are rewritten as participant reasoning and conversational responses based on scenario facts. Supervision includes the current reasoning, reasoning terminator, final answer, and end-of-turn token, excluding history and context. There are 349,660 target tokens; the maximum sequence is 4,338 tokens under an 8,192-token training limit, with no target truncation.

## Table A1. Shared adaptation settings.

<table><tr><td>Parameter</td><td>Setting</td></tr><tr><td>Original model</td><td>Official Qwen/Qwen3.5-27B</td></tr><tr><td>Initialization</td><td>Original weights with a new LoRA; no preceding adapter</td></tr><tr><td>LoRA rank/scale/dropout</td><td>16/32/0</td></tr><tr><td>Optimizer</td><td>AdamW</td></tr><tr><td>Learning rate</td><td>10−5; 10 warm-up steps, then linear decay to 0.1 of initial rate</td></tr><tr><td>Weight decay/gradient clipping</td><td>0.01/1</td></tr><tr><td>Global batch</td><td>8; eight replicas, microbatch one each</td></tr><tr><td>Training passes</td><td>One full pass, 125 updates</td></tr><tr><td>Loss</td><td>Mean cross-entropy over current target tokens in the global batch</td></tr><tr><td>Reference KL/segment weights</td><td>Neither</td></tr><tr><td>Main checkpoint</td><td>Update 125, frozen thereafter</td></tr><tr><td>Direction/grid precision</td><td>BF16 backbone + unmerged FP32 LoRA</td></tr></table>

All current target tokens have equal weight.

## A.3 Symmetric Behavioral Targets

The approved openness example places the participant and a friend in a pottery workshop. Familiar brush painting and unfamiliar sponge stippling have both been demonstrated, cost the same, take ten minutes, and are equally safe and feasible. There is no competition or gift obligation. The shared instruction asks for first-person speech, action, or both. The high target chooses the new sponge technique to explore its texture; the low target chooses the familiar brush technique and completes the same plate. Scenario resources and task demands are unchanged.

The following Chinese targets are retained verbatim, rather than represented as original English generations:

High: 我把海绵移到碟子旁，先蘸一点颜料试出疏密，再沿碟边点出一圈。“今天我用这个新办法。刷子我已经很熟了，这次想亲手试试点色，看海绵落下去会留下怎样的纹理，再把整只碟子做完。”

Low: 我把刷子移到碟子旁，先蘸一点颜料试好厚薄，再沿碟边均匀刷开。“今天我还是用熟悉的平涂。海绵工具先放回去，这次就按我习惯的手法，把整只碟子的颜色铺得匀一些。”

## English translations of the Chinese targets (for readers, not English model outputs):

High. I move the sponge beside the plate, dip it lightly in paint to try diferent densities, and stipple a ring around the rim. “Today I will use this new technique. I already know the brush well; this time I want to try stippling myself, see what texture the sponge leaves, and then finish the whole plate.”

Low. I move the brush beside the plate, dip it lightly in paint to check the thickness, and spread it evenly around the rim. “Today I will stick to the familiar flat painting. I will put the sponge back and use my usual technique to cover the whole plate more evenly.”

Questionnaire extraction uses high/low role prefixes and both A/B orders, producing 120 role-conditioned and 60 reference inputs per dimension. Catalog measurement removes those prefixes, using two numeric mappings and both behavioral orders. Appendix H.4 indexes the exact templates and item identities.

## A.4 Full-Parameter Post-Training for the Six-Dimensional Consumer Instance

The Huihui instance starts from its original 27B weights and uses the original 1,000 participant examples, 250 nativereasoning examples with explicit participant identity, and 250 Chinese conflict-and-boundary examples. The identity examples are diferent outputs from 96 ACNR questionnaire items. The conflict-and-boundary set comprises 150 situations involving ofense, 50 boundary-setting examples, and 50 ordinary exchanges or de-escalation after an apology. These examples train fact-grounded participant responses, not aggression in every situation.

All language-model parameters are updated for two epochs: 188 steps and 3,000 actual example exposures. Unused vision parameters are not updated. The global batch size is 16; AdamW uses learning rate $5 \times 1 0 ^ { - 6 }$ , ten warm-up steps, weight decay 0.01, and gradient clipping at one. Training uses FSDP sharding, FP32 master parameters and optimizer states, BF16 forward computation, activation checkpointing, and chunked vocabulary loss. Supervision covers complete reasoning, fina answers, and termination tokens, with no target truncation or example packing. Training takes 54 min 28 s including loading and complete weight exports after both epochs. The second-epoch model is then frozen for new direction learning and catalog construction.

The 96 ACNR source items also appear in six-dimensional calibration; no exact source-item overlap is found for M/T. This instance measures static responses within a specified evaluation system, rather than claiming a test entirely held out from shared post-training. Appendix H’s six-dimensional evidence record provides model, data, and completion digests.

## Appendix B Refinement, Selection, and Measurement

## B.1 Complete Losses and Optimization

The choice and self-report losses are

$$
\mathcal { L } _ { \mathrm { A B } } = \mathbb { E } _ { m } [ - q _ { \alpha , m } \log p _ { \alpha , m } - ( 1 - q _ { \alpha , m } ) \log ( 1 - p _ { \alpha , m } ) ] ,
$$

$$
\mathcal { L } _ { \mathrm { s e l f } } = \mathbb { E } _ { m } \bigl [ q _ { \alpha , m } \overline { { \mathrm { C E } } } ( y _ { Q , m } ^ { + } \mid x _ { m } , \alpha v ) + ( 1 - q _ { \alpha , m } ) \overline { { \mathrm { C E } } } ( y _ { Q , m } ^ { - } \mid x _ { m } , \alpha v ) \bigr ] .
$$

Length normalization and neutral preservation are

$$
\overline { { \mathbf { C E } } } ( y \mid x , \alpha v ) = - \frac { 1 } { | y | } \sum _ { j } \log p _ { \theta _ { \mathrm { F P } } , \alpha v } ( y _ { j } \mid x , y _ { < j } ) ,
$$

$$
\mathcal { L } _ { \mathrm { n e u r a l } } = \frac { 1 } { | \mathcal { I } _ { N } | } \sum _ { j \in \mathcal { I } _ { N } } D _ { \mathrm { K L } } \big ( p _ { \theta _ { \mathrm { F P } } , 0 } ( \cdot \mid x _ { N } , y _ { N , < j } ) \mid \big | p _ { \theta _ { \mathrm { F P } } , \alpha _ { N } v } ( \cdot \mid x _ { N } , y _ { N , < j } ) \big ) .
$$

Each questionnaire encoding has its own baseline and equal weight. Complete targets include termination tokens and are normalized by their own valid lengths. Candidate zero-based layers are 8/16/24/32/40/48/56/60. The relative norm budget is 0.1. Adam uses learning rate 0.003 for 240 updates with gradient clipping at one. Questionnaire doses cycle through −1, −0.5, 0.5, 1; scoring temperature is 0.7 and log-odds displacement coeficient is two. Behavioral strength alternates between 0.5 and one after each 16-scenario cycle, with opposite signs for high/low targets.

Loss weights for choice, self-report, behavior, neutral preservation, and norm penalty are 0.5/0.1/0.5/0.1/0.01. Neutral preservation runs every four updates with alternating signed neutral doses. Each step includes both questionnaire orders and one behavioral pair. Projection is $u  u / \operatorname* { m a x } ( 1 , \| u \| _ { 2 } )$ . Fixed cycling periods are 30 questionnaire items and 16 behavioral scenarios. Questionnaire-only removes the action loss; behavior-only removes choice and self-report losses without rescaling remaining weights.

## B.2 Development Selection

Development doses are −1, −0.5, 0, 0.5, and 1. The questionnaire signal is the restored expected numeric score; the behavioral signal is the diference between length-normalized high/low target log likelihoods. For interface $m \in \{ Q , B \}$ , the weaker bidirectional efect is

$$
\begin{array} { r } { b _ { k , \ell } ^ { m } = \operatorname* { m i n } \{ s _ { k , \ell } ^ { m } ( a _ { \mathrm { d e v } } ) - s _ { k , \ell } ^ { m } ( 0 ) , \ s _ { k , \ell } ^ { m } ( 0 ) - s _ { k , \ell } ^ { m } ( - a _ { \mathrm { d e v } } ) \} . } \end{array}
$$

Convert efects to midrank percentiles across candidate layers. Selection maximizes, in order, the smaller interface percentile, mean percentile, behavioral adjacent monotonicity, and negative added questionnaire mapping discrepancy, then favors the shallower layer. With c candidates, midrank is $( \# \{ b ^ { \prime } < b \} + 0 . 5 ( \# \{ b ^ { \prime } = b \} - 1 ) ) / ( c - 1 )$ . Adjacent positive/ zero/negative changes receive 1/0.5/0. A single candidate is adopted directly.

## Table B1. Selected layers for the non-thinking main catalogs.

<table><tr><td>Dimension</td><td>0</td><td>C</td><td>E </td><td>A</td><td>N</td><td>R</td><td>T</td><td>V</td><td>M</td><td>Se</td></tr><tr><td>Main catalog layer</td><td>40</td><td>32</td><td>32</td><td>16</td><td>32</td><td>32</td><td>40</td><td>32</td><td>40</td><td>32</td></tr></table>

## B.3 Scores and Quality Filtering

Questionnaire expectations over legal labels 1–5 are restored to 0/25/50/75/100 using mapping and polarity keys. Behavioral scores are high-pole A/B probability multiplied by 100. Five- and ten-dimensional nodes use 320 and 640 probes a temperature 0.7. S averages Q and B equally. Full-vocabulary legal mass is

$$
\mu _ { i , m } ( a ) = \frac { \sum _ { u \in \mathscr { V } _ { i , m } } \exp ( z _ { i , m , u } ( a ) / T ) } { \sum _ { u \in \mathscr { V } } \exp ( z _ { i , m , u } ( a ) / T ) } .
$$

All four conditions must hold before retrieval:

1. Mean legal mass is at least 0.98 for every dimension and each Q/B interface.

2. Mean absolute questionnaire score discrepancy between restored mappings is at most 25.

3. Mean absolute behavioral high-pole probability discrepancy between orders is at most 0.25.

4. The raw A-label mean satisfies max $\left( \bar { p } _ { A } , 1 - \bar { p } _ { A } \right) \leq 0 . 9 0 .$

The first criterion is coordinate-specific; the other three aggregate the node. The 98% threshold budgets at most $2 \%$ mean non-answer probability. Same-layer inner products contribute to joint perturbation:

$$
\| \Delta _ { \ell } ( a ) \| _ { 2 } ^ { 2 } = \sum _ { k \in I _ { \ell } } a _ { k } ^ { 2 } \| v _ { k } \| _ { 2 } ^ { 2 } + 2 \sum _ { k < k ^ { \prime } ; ~ k , k ^ { \prime } \in I _ { \ell } } a _ { k } a _ { k ^ { \prime } } \langle v _ { k } , v _ { k ^ { \prime } } \rangle .
$$

Thus output quality is measured directly. Calibration-interface filtering and full-dialogue quality are evaluated separately;   
failed nodes remain in the raw catalog.

## B.4 Directions and Construction of the Original Four-Dimensional Thinking Catalog

The same adapter supplies high/low demonstrations generated by the frozen model. Only complete, target-consistent pairs are retained. Initialization averages hidden states over demonstration reasoning and scales the diference to 0.1 times the mean reference-state norm. Each step contains questionnaire and behavioral high/low examples at coeficients ±1. Reasoning/final-answer length-normalized cross-entropy weights are 0.7/0.3. Adam uses 0.003 for 240 updates, clipping one, and L2 weight 0.01. Signed neutral KL runs every four updates with weight 0.05 per sign.

Development selection uses mean directed Q/B separation minus invalidity, prioritizing 98% development validity. Eightdimensional single-axis development generates 9,216 responses. The O/R/M/A catalog uses layers 32/16/24/16, 81 nodes, 128 items, and three repeats, with fixed encoding, medium thinking, temperature 0.7, and 2,048 tokens. Vector and prompt catalogs have answer-validity rates 96.80% and 99.28%. Generation validity is distinct from non-thinking legal-probability gating.

Table B2. Joint responses by coeficient level in the original four-dimensional thinking catalog. Holding the focal coeficient fixed, values aggregate the other three dimensions’ 27 combinations and three repeats. This catalog supports Sections 4.3–4.4, not the six-dimensional full-parameter model.
<table><tr><td>Dimension</td><td>Coefficient -1</td><td>Coefficient 0</td><td>Coefficient +1</td></tr><tr><td>Openness O</td><td>54.25</td><td>62.91</td><td>69.20</td></tr><tr><td>Risk taking R</td><td>29.94</td><td>32.44</td><td>36.61</td></tr><tr><td>Material values M</td><td>16.83</td><td>27.15</td><td>43.34</td></tr><tr><td>Agreeableness A</td><td>59.92</td><td>65.43</td><td>72.82</td></tr></table>

## B.5 Refinement and Filtering in the Six-Dimensional Thinking Instance

The frozen, fully post-trained model generates 1,104 high/low training demonstrations and eight neutral anchors. Training demonstrations alone allow at most two logged format-repair attempts. A total of 1,549 attempts yields 548 complete high/ low pairs; invalid formal development or grid outputs are not resampled. For each axis, the first four complete questionnaire pairs and first four behavioral pairs initialize the direction. Mean reasoning activations are extracted under teacher-forced inputs without high/low role instructions. Their high–low diference is scaled to 0.1 times the mean norm of the corresponding reasoning-mean vectors.

Each of the six dimensions is refined at zero-based layers 8/16/24/32/40/48/56/60 for 240 steps with learning rate 0.003. Each step contains one high and one low example from both questionnaire and behavior, with coeficients ±1. Reasoning and final-answer losses are normalized separately by length and weighted 0.7/0.3. Bilateral neutral KL is applied every four steps with total weight 0.1, and the vector L2 weight is 0.01. Parameters are projected onto the unit ball after each update. Injection covers the full input and target sequence.

Development evaluation generates 2,880 answers at −1/0/+1. It first prioritizes candidates with validity at least 98%, then ranks the equal-weighted questionnaire/behavior endpoint diference minus 100(1−validity), breaking ties toward shallower layers. Invalid development outputs receive a temporary score of 50 and the separate penalty; formal scores do not impute invalid answers. All six selected directions have 100% development validity. This is layer-selection measurement, not an additional complete single-axis scan.

The formal catalog uses 24 questionnaire and eight behavioral items per dimension, with two repeats of each encoding, medium thinking, temperature 0.7, and a 4,096-output-token cap. Within each repeat, encodings and Q/B components receive equal weight; scores are then averaged over the two repeats. Invalid outputs remain in each dimension’s denominator of 128 answers. Requiring at least 126 valid answers in every dimension retains 663 nodes. Deployment separately requires a literal occurrence rate of the Chinese word “用户” (user) no greater than 10%—all nodes pass in this run—and removes six qualified nodes on an existing exclusion list, leaving 657 serving candidates. This deployment exclusion changes neither the original 729 scores nor the 663 format-qualified count. The lexical condition is not a semantic certification of first-person reasoning.

## Appendix C Coverage, Retrieval, and Joint Calibration

## C.1 Targets and Full Coverage

Primary coverage uses 1,000 ten-dimensional uniform unit draws with seed 20261004; five-dimensional analyses use the first five columns and rescale to each target domain. Feasibility requires per-axis error at most 20 and total at most 12d at the same node. Existence coverage and ordinary-L1 returned-node success are distinct. Refinement ablations use 10,000 targets per domain.

Table C1. Composite dimension-score coverage and returned-node success across target domains (%).
<table><tr><td>Catalog</td><td>Domain</td><td>Unfiltered coverage</td><td>Qualified coverage</td><td>Unfiltered L1 success</td><td>Qualified L1 success</td></tr><tr><td>5D</td><td>2080</td><td>87.6</td><td>82.7</td><td>67.4</td><td>61.9</td></tr><tr><td>5D</td><td>25—75</td><td>95.3</td><td>92.2</td><td>82.0</td><td>76.1</td></tr><tr><td>5D</td><td>30—70</td><td>99.6</td><td>97.7</td><td>93.3</td><td>86.6</td></tr><tr><td>10D</td><td>20—80</td><td>41.5</td><td>34.7</td><td>12.9</td><td>10.5</td></tr><tr><td>10D</td><td>25—75</td><td>69.3</td><td>58.6</td><td>26.5</td><td>22.0</td></tr><tr><td>10D</td><td>30—70</td><td>95.4</td><td>88.2</td><td>47.9</td><td>40.7</td></tr></table>

Catalog union returns one admissible vector or prompt configuration. Strict tolerance is per-axis <10 and total <30. In 25–75, five-level union S coverage is 98.7%, versus 88.0% ordinary-L1 returned success; feasible-first retrieval uses the full existence coverage.

## C.2 Single-Axis Additive Prediction and Retrieval

Let J contain evaluated joint nodes, excluding zero and single-axis configurations, and let $Y ( a )$ be their measured scores. Prediction MAE and target MAE are

$$
\mathrm { M A E } _ { \mathrm { p r e d } } = \frac { 1 } { | \mathcal I | d } \sum _ { a \in \mathcal I } \sum _ { j = 1 } ^ { d } | \widehat { F } _ { j } ( a ) - Y _ { j } ( a ) | ,
$$

$$
\mathrm { M A E } _ { \mathrm { t a r g e t } } = \frac { 1 } { | T | d } \sum _ { t \in \mathcal { T } } \sum _ { j = 1 } ^ { d } | Y _ { j } ( a _ { \widehat { F } } ( t ) ) - t _ { j } | .
$$

Here $a _ { \hat { F } } ( t )$ is the configuration retrieved using estimated scores. The first error concerns a node’s predicted response; the bsecond concerns the measured response of the node returned for a target. Candidate sets, target sets, and retrieval rules are held equal within each comparison.

Score-additive prediction is $\begin{array} { r } { \widehat { s } _ { j } ( a ) = s _ { j } ( 0 ) + \sum _ { k } [ s _ { j } ( a _ { k } e _ { k } ) - s _ { j } ( 0 ) ] } \end{array}$ . The clipped variant restricts the result to 0–100. With $z _ { j } ( a ) = \mathrm { l o g i t } ( \mathrm { c l i p } ( s _ { j } ( a ) / 1 0 0 , 0 . 0 1 , 0 . 9 9 ) )$ ), saturation-aware prediction is

$$
\hat { s } _ { j } ^ { \mathrm { l o g i t } } ( a ) = 1 0 0 \sigma \left( z _ { j } ( 0 ) + \sum _ { k } [ z _ { j } ( a _ { k } e _ { k } ) - z _ { j } ( 0 ) ] \right) .
$$

All cross-dimensional main efects are retained. Joint configurations are evaluation data only. Vector logit MAE is 5.244 over 1,836 admissible joint nodes, 4.968 over 1,225 nodes whose anchors are also admissible, and 2.701 over 190 nodes when coeficients are restricted to magnitude at most one. Corresponding prompt full/middle-three-level values are 3.973/3.169.

Table C2. Returned-node success on 10,000 targets with common qualified candidates and feasible-first retrieval. Additive retrieval is scored with archivedjoint responses. Intervals use 10,000 paired target-bootstrap resamples, conditional on the fixed catalogs.
<table><tr><td>Target domain</td><td>Single-axis additive (%)</td><td>Joint grid (%)</td><td>Joint – additive</td><td>Paired-target 95% CI</td></tr><tr><td>[25, 75]5</td><td>62.22</td><td>91.85</td><td>+29.63</td><td>[28.73, 30.53]</td></tr><tr><td>[20, 80]5</td><td>50.95</td><td>80.56</td><td>+29.61</td><td>[28.72, 30.55]</td></tr></table>

For composite targets in 25–75, mean actual L1 decreases from 44.16 to 29.98. With ordinary L1 for both methods, success is 60.49% versus 76.09%. Clipped addition preserves the primary success rate. Candidate and retrieval sensitivities are indexed in Appendix H.4.

## C.3 Thinking-Mode Held-Out Repeats

Two repeats estimate catalog and anchor responses, and the third evaluates them, rotating over three folds. Score prediction evaluates 72 non-single-axis nodes. Both retrieval methods use 81 candidates without filtering on held-out validity and the same 1,000 targets in 20–80.

## Table C3. All folds.

<table><tr><td>Control</td><td>Held-out repeat</td><td>Logit prediction MAE</td><td>Joint prediction MAE</td><td>Logit target MAE</td><td>Joint target MAE</td></tr><tr><td>Vector</td><td>1</td><td>9.123</td><td>5.684</td><td>15.490</td><td>16.158</td></tr><tr><td>Vector</td><td>2</td><td>7.874</td><td>5.695</td><td>18.514</td><td>15.578</td></tr><tr><td>Vector</td><td>3</td><td>7.797</td><td>5.403</td><td>16.411</td><td>15.676</td></tr><tr><td>Prompt</td><td>1</td><td>6.667</td><td>4.821</td><td>11.430</td><td>11.077</td></tr><tr><td>Prompt</td><td>2</td><td>7.206</td><td>4.843</td><td>12.001</td><td>11.231</td></tr><tr><td>Prompt</td><td>3</td><td>8.944</td><td>4.707</td><td>11.947</td><td>10.259</td></tr></table>

All six comparisons improve score prediction; five improve retrieval. Vector fold one has higher target error after joint calibration. Folds share items and overlapping training repeats, so their aggregate is a descriptive mean. The analysis indexed in Appendix H.4 also reports Q/B results.

## C.4 Thinking-Mode Prompt Complementarity

Targets are the 2,401 combinations of 20/30/40/50/60/70/80 on four dimensions. Vector and prompt catalogs contain 81 configurations each. Union covers 63/529/1,311 targets at per-axis tolerances 5/10/15; adding 30 simultaneously combined configurations yields 63/534/1,345. No total-error constraint is added in this analysis.

The fixed-prompt endpoint study has 71 unique conditions, 384 responses each, totaling 27,264 generations. All-high prompt-only and positive-vector-combined validity counts are 376/384 and 373/384; all-low prompt-only and combined counts are 380/384 and 383/384. A separately selected high catalog configuration has 292/384 valid responses, 88 incomplete and four malformed; it is distinct from the all-positive-coeficient condition in the main table. Prompt low/middle/high S means increase in all 108 measured slices fixing the other dimensions. Appendix H.4 indexes the complete measurement records.

## C.5 Disjoint-Bin Occupancy

As a supplement to target-tolerance coverage, partition score space into disjoint cells and count cells containing at least one qualified measured node. For a fixed partition P,

$$
N _ { \mathrm { b i n } } = \sum _ { P \in \mathcal { P } } \mathbf { 1 } \{ \exists a \in \mathcal { G } _ { \mathrm { a d m } } : \widehat { F } _ { \pi } ( a ) \in P \} , \qquad C _ { \mathrm { b i n } } = N _ { \mathrm { b i n } } / | \mathcal { P } | .
$$

All coordinates must fall into the same joint cell. Intervals are left-closed/right-open, except that the final upper boundary is included; out-of-domain scores are not clipped. An occupied cell does not imply that every target inside it is reachable.

Table C4. Qualified composite-score occupancy. OFF boundaries are 25,35,45,55,65,75 on each axis. ON low/medium/ high boundaries are 0,100/3,200/3,100, with independently calibrated catalogs.
<table><tr><td>Mode</td><td>Catalog</td><td>Occupied</td><td>Total</td><td>Proportion</td></tr><tr><td>OFF 5D</td><td>Three-level vector</td><td>41</td><td>3,125</td><td>1.31%</td></tr><tr><td>OFF 5D</td><td>Five-level vector</td><td>342</td><td>3,125</td><td>10.94%</td></tr><tr><td>OFF 5D</td><td>Nine-level vector</td><td>845</td><td>3,125</td><td>27.04%</td></tr><tr><td>ON 4D</td><td>Vector</td><td>10</td><td>81</td><td>12.35%</td></tr><tr><td>ON 4D</td><td>Prompt</td><td>43</td><td>81</td><td>53.09%</td></tr><tr><td>ON 4D</td><td>Union</td><td>46</td><td>81</td><td>56.79%</td></tr></table>

## C.6 Zero and Random-Configuration Baselines

We reuse the same 1,000 targets in [25, 75]<sup>5</sup>, composite scores, and four-part filtering. The zero node retains the shared first-person LoRA but applies no vectors; its O/C/E/A/N scores are 64.45/61.31/50.81/69.41/34.31. A random singleton is drawn uniformly from all 1,854 qualified five-level nodes without inspecting the target. We compute the exact expectation over these nodes.

Table C5. Single configurations versus catalog retrieval. A realized singleton occupies zero or one in-domain cell; its fractional occupancy below is an expectation.
<table><tr><td>Strategy</td><td>Available nodes</td><td>Broad coverage</td><td>Strict coverage</td><td>Occupied /3,125</td></tr><tr><td>Zero vector</td><td>1</td><td>10.30%</td><td>0.80%</td><td>1</td></tr><tr><td>Uniform random singleton</td><td>1</td><td>5.82%</td><td>0.31%</td><td>0.44 (expected)</td></tr><tr><td>Five-level catalog retrieval</td><td>1,854</td><td>92.20%</td><td>44.00%</td><td>342</td></tr></table>

We also draw 243 or 3,125 configurations without replacement from all 59,049 measured nine-level configurations, then apply the same quality filter and retrieve from survivors. Each budget has 30 draws, seed 2026102608, without targetdependent sampling. This is retrospective analysis of measured records, not new GPU generation.

Table C6. Equal-measurement-budget regular and random designs. Random entries are mean±standard deviation across 30 subsets.
<table><tr><td>Design</td><td>Measured</td><td>Qualified</td><td>Broad coverage</td><td>Strict coverage</td><td>Occupied bins</td></tr><tr><td>Regular</td><td>243</td><td>120</td><td>71.20%</td><td>15.60%</td><td>41</td></tr><tr><td>three-level</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random subset</td><td>243</td><td>162.43±9.32</td><td>78.44±2.81%</td><td>21.55±1.34%</td><td>62.43±6.63</td></tr><tr><td>Regular</td><td>3,125</td><td>1,854</td><td>92.20%</td><td>44.00%</td><td>342</td></tr><tr><td>five-level</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Random subset</td><td>3,125</td><td>2,066.53±25.01</td><td>91.52±0.64%</td><td>42.93±0.58%</td><td>371.80±11.45</td></tr></table>

Random sampling improves coverage at the smaller budget and occupies more bins at both budgets; the regular five-level design has slightly higher tolerance coverage. Thus joint measurement and retrieval are the core contribution, while Cartesian enumeration is one sampling implementation rather than a universally optimal design. Source identities and all subsets’ metrics are archived with the baseline analysis.

## C.7 Complete Values Summarized in the Main Text

Table C7. Vector increments under identical prompts. Each condition includes 384 responses, averaging three completegeneration repeats.

<table><tr><td colspan="6">Endpoint/</td></tr><tr><td>condition</td><td>0</td><td>R</td><td>M</td><td>A</td><td>Answer validity</td></tr><tr><td>High: prompt only</td><td>80.10</td><td>79.79</td><td>83.05</td><td>80.76</td><td>97.92%</td></tr><tr><td>High: vector + same prompt</td><td>89.05</td><td>86.74</td><td>91.72</td><td>88.72</td><td>97.14%</td></tr><tr><td>Low: prompt only</td><td>11.81</td><td>8.63</td><td>16.84</td><td>27.08</td><td>98.96%</td></tr><tr><td>Low: vector + same prompt</td><td>4.51</td><td>4.89</td><td>8.51</td><td>25.17</td><td>99.74%</td></tr></table>

Table C8. Prediction errors for non-thinking joint nodes. Nodes and dimensions are equally weighted (protocol in C.2).
<table><tr><td>Control</td><td>Joint nodes</td><td>Score-additive MAE</td><td>Clipped MAE</td><td>Logit additive MAE</td></tr><tr><td>5D vector</td><td>1,836</td><td>6.325</td><td>6.246</td><td>5.244</td></tr><tr><td>5D prompt</td><td>3,104</td><td>5.304</td><td>5.068</td><td>3.973</td></tr></table>

## Appendix D Construction Ablations

## D.1 Cognitive–Behavioral Supervision Across Construction Pipelines

Holding the shared first-person adapter fixed, we compare questionnaire-only directions (FP-Q) and questionnaire-plusbehavior directions (FP-QB). Each selects its own layers and builds its five-level catalog; this compares supervision across complete construction pipelines. Section D.3 separately fixes layers.

Table D1. Questionnaire and joint directions on the first-person model. The same 10,000 targets in $[ 2 5 , 7 5 ] ^ { 5 }$ and tolerances are used; qualified columns apply all four filters.
<table><tr><td>Interface</td><td>FP-Q raw</td><td>FP-QB raw</td><td>FP-Q qualified</td><td>FP-QB qualified</td></tr><tr><td>Questionnaire Q</td><td>78.52%</td><td>81.55%</td><td>76.76%</td><td>77.32%</td></tr><tr><td>Behavior B</td><td>68.32%</td><td>89.42%</td><td>63.73%</td><td>80.81%</td></tr><tr><td>Composite S</td><td>87.41%</td><td>96.10%</td><td>86.38%</td><td>91.85%</td></tr></table>

## D.2 Initialization, Matched-Data Extraction, and Refinement

Fixed layers are O40/C32/E32/A16/N32. Original initialization uses final-position contrasts between high/low role prompts;   
comparisons retain its native norm or match it to the refined norm as specified in archived variants.

Table D2. Original initialization and refined endpoint spans. Each entry is the +2 minus −2 score.
<table><tr><td>Dimension</td><td>Initial Q span</td><td>Refined Q span</td><td>Initial B span</td><td>Refined B span</td><td>Initial S span</td><td>Refined S span</td></tr><tr><td>Openness (Ó)</td><td>18.74</td><td>24.05</td><td>32.10</td><td>60.91</td><td>25.42</td><td>42.48</td></tr><tr><td>Conscien- tiousness</td><td>21.88</td><td>23.79</td><td>36.69</td><td>60.30</td><td>29.29</td><td>42.04</td></tr><tr><td>(C) Extraversion (E)</td><td>21.43</td><td>27.78</td><td>49.24</td><td>83.60</td><td>35.34</td><td>55.69</td></tr><tr><td>Agreeable- ness (A)</td><td>0.71</td><td>16.79</td><td>-3.74</td><td>30.03</td><td>-1.51</td><td>23.41</td></tr><tr><td>Neuroticism (N)</td><td>24.66</td><td>35.63</td><td>27.15</td><td>41.65</td><td>25.91</td><td>38.64</td></tr></table>

Matched-data extraction uses every refinement target: 30 questionnaire items with two orders plus 16 behavioral pairs per dimension, totaling 380 pairs and 760 teacher-forced sequences. Q-mean and B-mean average content-token activation

contrasts within each interface. QB-mean combines the group directions equally; QB-last uses the last target-content token.   
All four are norm-matched to the refined vector, with no parameter update.

Table D3. All matched-data variants. Layers, doses, and measurement interfaces are shared.
<table><tr><td>Dimension</td><td>Direction</td><td>Q span</td><td>B span</td><td>S span</td></tr><tr><td>0</td><td>Q-mean</td><td>17.43</td><td>43.96</td><td>30.69</td></tr><tr><td>0</td><td>B-mean</td><td>8.95</td><td>40.72</td><td>24.84</td></tr><tr><td>0</td><td>QB-mean</td><td>17.09</td><td>52.89</td><td>34.99</td></tr><tr><td>0</td><td>QB-last</td><td>11.58</td><td>43.06</td><td>27.32</td></tr><tr><td>0</td><td>Refined</td><td>24.05</td><td>60.91</td><td>42.48</td></tr><tr><td>C</td><td>Q-mean</td><td>15.84</td><td>15.98</td><td>15.91</td></tr><tr><td>C</td><td>B-mean</td><td>3.55</td><td>39.84</td><td>21.70</td></tr><tr><td>C</td><td>QB-mean</td><td>15.31</td><td>36.77</td><td>26.04</td></tr><tr><td>C</td><td>QB-last</td><td>7.22</td><td>37.39</td><td>22.30</td></tr><tr><td>C</td><td>Refined</td><td>23.79</td><td>60.30</td><td>42.04</td></tr><tr><td>E</td><td>Q-mean</td><td>18.24</td><td>56.14</td><td>37.19</td></tr><tr><td>E</td><td>B-mean</td><td>11.79</td><td>63.33</td><td>37.56</td></tr><tr><td>E</td><td>QB-mean</td><td>15.57</td><td>70.41</td><td>42.99</td></tr><tr><td>E</td><td>QB-last</td><td>13.04</td><td>53.95</td><td>33.50</td></tr><tr><td>E</td><td>Refined</td><td>27.78</td><td>83.60</td><td>55.69</td></tr><tr><td>A</td><td>Q-mean</td><td>10.30</td><td>-12.25</td><td>-0.98</td></tr><tr><td>A</td><td>B-mean</td><td>-12.32</td><td>35.35</td><td>11.52</td></tr><tr><td>A</td><td>QB-mean</td><td>-10.82</td><td>35.19</td><td>12.18</td></tr><tr><td>A</td><td>QB-last</td><td>-9.32</td><td>20.93</td><td>5.80</td></tr><tr><td>A</td><td>Refined</td><td>16.79</td><td>30.03</td><td>23.41</td></tr><tr><td>N</td><td>Q-mean</td><td>16.08</td><td>44.24</td><td>30.16</td></tr><tr><td>N</td><td>B-mean</td><td>12.45</td><td>48.70</td><td>30.58</td></tr><tr><td>N</td><td>QB-mean</td><td>18.81</td><td>56.25</td><td>37.53</td></tr><tr><td>N</td><td>QB-last</td><td>15.01</td><td>45.74</td><td>30.37</td></tr><tr><td>N</td><td>Refined</td><td>35.63</td><td>41.65</td><td>38.64</td></tr></table>

Refinement increases composite span relative to QB-mean on all axes and restores positive questionnaire direction for A. Direct extraction has larger B spans on A/N. Refined C at +2 has Q/B legal mass 97.69%/98.19%; the point is retained in curves and rejected by retrieval filtering. The corresponding measured own-interface masses for QB-mean exceed 99.6%. Appendix H.4 indexes the per-dose values and source identities.

Matched-data extraction versus refinement has mean Q/ B/ S spans of 11.19/50.30/30.75 versus 25.61/55.30/40.45.   
Dimension-wise values already appear in the QB-mean and refined rows of Table D3.

## D.3 Fixed-Layer Supervision

Q-only, B-only, and QB share model, initialization, layers, 240 updates, and full-sequence injection, each producing 3,125 nodes. Q-only removes action loss; B-only removes choice and self-report losses; QB retains all components.

Table D4. Full fixed-layer coverage. Each domain has 10,000 targets, maximum per-axis error 20, and total error 60. Entries are unfiltered/qualified percentages.
<table><tr><td>Target domain</td><td>Refinement target</td><td>Questionnaire Q</td><td>Behavior B</td><td>Composite S</td></tr><tr><td>[25, 75]5</td><td>Q</td><td>82.55/78.76</td><td>73.11/65.75</td><td>87.90/85.64</td></tr><tr><td>[25, 75]5</td><td>B</td><td>66.94/58.87</td><td>89.77/84.77</td><td>89.26/82.32</td></tr><tr><td>[25, 75]5</td><td>QB</td><td>81.55/77.32</td><td>89.42/80.81</td><td>96.10/91.85</td></tr><tr><td>[20, 80]5</td><td>Q</td><td>66.81/63.44</td><td>59.18/52.28</td><td>71.46/68.47</td></tr><tr><td>[20, 80]5</td><td>B</td><td>41.04/35.26</td><td>72.95/66.29</td><td>68.70/61.82</td></tr><tr><td>[20, 80]5</td><td>QB</td><td>65.73/61.55</td><td>79.97/70.16</td><td>87.02/80.56</td></tr></table>

This comparison evaluates how supervision changes reachability under both measurement interfaces. Independently selected pipelines in D.1 answer a diferent whole-system question.

## D.4 Injection-Scope Strategies

Both strategies use the same earlier approved behavioral bank, questionnaires, initialization, budget, candidate layers, and development rule. This bank precedes later O/E revisions. Response-side injection covers the final prompt position and subsequent target positions; full-sequence injection additionally covers every nonpadding input position. Training and measurement use their respective scopes.

Independent selection yields O48/C32/E40/A40/N40 versus O32/C32/E32/A16/N32. The experiment therefore compares complete strategies including layer selection. First-token measurement reads the same final output position after diferent input-side intervention histories.

Table D5. Endpoints and legal numeric mass under diferent input scopes. All endpoints are retained.
<table><tr><td>Dimension</td><td>Response-side Q (-2/0/+2)</td><td>Full-sequence Q (-2/0/+2)</td><td>Response-side legal mass (-2/+2)</td><td>Full-sequence legal mass (-2/+2)</td></tr><tr><td>0</td><td>52.24 / 52.95 / 53.45</td><td>37.60 / 52.95 / 61.60</td><td>99.79% / 99.54%</td><td>97.72% / 99.75%</td></tr><tr><td>C</td><td>57.17 / 57.89 / 58.31</td><td>44.37 / 57.89 / 68.25</td><td>99.76% / 99.90%</td><td>99.96% / 97.81%</td></tr><tr><td>E</td><td>47.61 / 48.53 / 51.62</td><td>29.19 / 48.53 / 64.82</td><td>99.90% / 99.90%</td><td>99.96% / 99.37%</td></tr><tr><td>A</td><td>60.10 / 70.44 / 71.35</td><td>61.35 / 70.44 / 77.99</td><td>99.90% / 99.91%</td><td>99.84% / 100.00%</td></tr><tr><td>N</td><td>50.59 / 51.08 / 72.37</td><td>39.56 / 51.08 / 75.21</td><td>99.88% / 99.66%</td><td>99.93% / 99.83%</td></tr></table>

Mean questionnaire span increases from 7.879 to 27.159. Quality filtering constrains deployment range. Appendix H.4 indexes the complete injection-scope measurements.

## D.5 Density Efects by Measurement Component

We retain the questionnaire and behavioral components separately to show how each contributes to the density results.   
Models, directions, score definitions, target sets, and filtering remain fixed.

Table D6. Component coverage (%) in 25–75.

<table><tr><td>Levels/axis</td><td>Q broad</td><td>Q strict</td><td>B broad</td><td>B strict</td></tr><tr><td>3</td><td>67.4</td><td>14.3</td><td>31.8</td><td>1.9</td></tr><tr><td>5</td><td>75.3</td><td>23.4</td><td>80.6</td><td>24.1</td></tr><tr><td>9</td><td>78.4</td><td>28.6</td><td>88.7</td><td>41.5</td></tr></table>

From five to nine levels, strict behavioral coverage increases by 17.4 percentage points and questionnaire coverage by 5.2.   
These component scores explain the composition of the unified measurement without changing the main retrieval rules.

## Appendix E Probability Calibration and Complete Generation

## E.1 Five-Node Non-Thinking Ten-Dimensional Remeasurement

From 33,942 qualified nodes, zero is fixed in advance. Remaining nodes are ordered by composite L1 distance from zero, divided into four equal-rank strata, and sampled once per stratum with seed 20261014. Each node completes all 640 calibration presentations: 480 questionnaire and 160 behavioral prompts, sampled once each, totaling 3,200 outputs. Model, vectors, injection, and temperature 0.7 match the catalog; top-p is one, top-k zero, and the cap 8,192 tokens.

Table E1. Configurations and remeasurement errors. Coeficient order is O/C/E/A/N/R/T/V/M/Se.

<table><tr><td>Node</td><td>10D coefficients</td><td>Dimension score MAE</td><td>Scoreable/640</td></tr><tr><td>29524</td><td>0,0,0,0,0,0,0,0,0,0</td><td>3.16</td><td>636</td></tr><tr><td>24359</td><td>0,-1,1,-1,0,-1,1,–1,0,1</td><td>3.94</td><td>638</td></tr><tr><td>42396</td><td>1,-1,0,0,-1,0,0,-1,1,-1</td><td>4.26</td><td>640</td></tr><tr><td>21182</td><td>0,-1,-1,1,-1,-1,0,0,0,1</td><td>3.52</td><td>635</td></tr><tr><td>1422</td><td>-1,-1,-1,0,1,1,0,1,-1,-1</td><td>1.76</td><td>637</td></tr></table>

Questionnaire first lines must contain a legal digit in the current mapping; behavior requires legal A/B. A valid digit without a required rationale is scoreable but not fully instruction-compliant. Invalid answers remain missing, without zero imputation or resampling.

Table E2. Scoreability and instruction compliance.
<table><tr><td>Generated</td><td>Scoreable answers</td><td>Scoreable rate</td><td>Fully compliant</td></tr><tr><td>3,200</td><td>3,186</td><td>99.56%</td><td>3,026</td></tr></table>

Table E3. Errors over 50 node–dimension cells. Valid responses are aggregated within each cell using the fixed weights in Appendix B.3.
<table><tr><td>Cells</td><td>MAE</td><td>Mean observed – predicted</td><td>RMSE</td><td>Max. absolute error</td><td>Five-node bootstrap MAE interval</td></tr><tr><td>50</td><td>3.33</td><td>-0.50</td><td>4.28</td><td>12.00</td><td>2.48—3.99</td></tr></table>

All responses terminate naturally without length truncation. Scoreability is 99.56%; full instruction compliance is 94.56%. Intervals use 2,000 resamples of the five complete nodes and describe between-node variation. This measures the deviation between probability expectations and a single round of actual answers for fixed configurations.

There are 14 missing answers. For planned count N, valid count n, and valid-score sum u, the full-denominator mean lies in $[ u / N , ( u + 1 0 0 ( N - n ) ) / N ]$ . Aggregation with fixed scoring weights gives dimension-score MAE bounds of 3.19– 3.51. The node-level scores and audit indexed in Appendix H.4 preserve all 50 prediction/response pairs, missing cells, and source identities.

## E.2 Quality-Threshold Sensitivity

On the same three-, five-, and nine-level non-thinking catalogs, we vary one threshold at a time while holding the other three fixed. No scores or model outputs are recomputed, so this directly evaluates how filtering changes the number of available candidates.

Table E4. Qualified nodes after one-at-a-time threshold changes. Full catalogs contain 243, 3,125, and 59,049 nodes. Original settings are legal mass ≥0.98, mapping-score diference ≤25, option-order probability diference ≤0.25, and labe bias ≤0.90.
<table><tr><td>Criterion</td><td>Threshold</td><td>Three levels</td><td>Five levels</td><td>Nine levels</td></tr><tr><td>Minimum legal mass</td><td>0.90</td><td>134</td><td>2,008</td><td>40,924</td></tr><tr><td>Minimum legal mass</td><td>0.95</td><td>130</td><td>1,955</td><td>40,332</td></tr><tr><td>Minimum legal mass</td><td>0.98</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Minimum legal mass</td><td>0.99</td><td>106</td><td>1,695</td><td>36,885</td></tr><tr><td>Maximum mapping-score gap</td><td>15</td><td>100</td><td>1,557</td><td>32,484</td></tr><tr><td>Maximum mapping-score gap</td><td>20</td><td>114</td><td>1,769</td><td>37,055</td></tr><tr><td>Maximum mapping-score gap</td><td>25</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Maximum mapping-score gap</td><td>30</td><td>124</td><td>1,890</td><td>39,668</td></tr><tr><td>Maximum order-probability gap</td><td>0.15</td><td>61</td><td>892</td><td>18,620</td></tr><tr><td>Maximum order-probability gap</td><td>0.20</td><td>99</td><td>1,545</td><td>33,331</td></tr><tr><td>Maximum order-probability gap</td><td>0.25</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Maximum order-probability gap</td><td>0.30</td><td>127</td><td>2,009</td><td>41,574</td></tr><tr><td>Maximum label bias</td><td>0.70</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Maximum label bias</td><td>0.80</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Maximum label bias</td><td>0.90</td><td>120</td><td>1,854</td><td>38,932</td></tr><tr><td>Maximum label bias</td><td>0.95</td><td>120</td><td>1,854</td><td>38,932</td></tr></table>

Within the examined ranges, order consistency has the largest efect on candidate count. Label-bias thresholds do not further change the admissible set when the other three constraints are fixed, indicating overlapping exclusions. Tightening lega

mass from 98% to 99% still retains 36,885 nine-level nodes; occupancy of 25–75 score bins decreases from 845 to 831.   
Applications can select thresholds according to acceptable output risk and required candidate count.

These statistics concern measured probabilities and consistency, not complete-dialogue failure rates. Independently sampling new nodes to compare complete-generation validity inside and outside the admissible set, and varying scoring temperature, require separate experiments.

## Appendix F External Scenarios and Interactive Applications

## F.1 Chinese Situational-Judgment Items

We use all 40 Study 3 items and original binary keys from Li et al. (Li et al., 2026). Five narrow facets have eight items each: openness to ideas, self-discipline, gregariousness, compliance/conflict handling, and self-consciousness/negativeevaluation sensitivity. Four cyclic option orders are used, and keys are absent from prompts. For each supervision arm, distinct nodes are selected at the 10th/50th/90th percentiles of calibrated Q; other-axis distance from 50 is only a secondary ranking criterion, not a hard constraint.

Forty-five nonzero configurations plus the shared zero baseline yield 7,360 unconstrained greedy, non-thinking responses with a 32-token cap. The 160 baseline outputs are reused after identity checks. Each designated facet has 1,440 valid outputs; 30 invalid outputs occur on background facets. Observed scores follow selected-option keys; conditional scores average first-token legal-option probabilities.

Table F1. Calibration coordinates and external responses. Triples denote low/middle/high.
<table><tr><td>Training arm</td><td>Facet</td><td>Calibration Q</td><td>Observed keyed score</td><td>Conditional score</td></tr><tr><td>Q</td><td>0</td><td>43.80/56.00/66.73</td><td>96.88/100.00/100.00</td><td>86.78/98.41/99.34</td></tr><tr><td>Q</td><td>C</td><td>36.08/55.50/75.96</td><td>93.75/100.00/100.00</td><td>91.21/96.23/97.05</td></tr><tr><td>Q</td><td>E</td><td>37.07/56.57/69.77</td><td>6.25/31.25/31.25</td><td>11.93/36.10/39.51</td></tr><tr><td>Q</td><td>A</td><td>43.39/61.31/78.53</td><td>93.75/93.75/90.63</td><td>86.27/91.53/90.48</td></tr><tr><td>Q</td><td>N</td><td>32.58/59.44/77.58</td><td>3.13/0.00/9.38</td><td>8.77/7.11/13.08</td></tr><tr><td>B</td><td>0</td><td>50.89/53.90/57.14</td><td>100.00/100.00/100.00</td><td>98.33/92.32/98.50</td></tr><tr><td>B</td><td>C</td><td>50.93/55.23/59.69</td><td>100.00/100.00/100.00</td><td>95.50/94.58/93.05</td></tr><tr><td>B</td><td>E</td><td>47.27/53.73/59.82</td><td>21.88/12.50/28.13</td><td>21.46/21.23/38.01</td></tr><tr><td>B</td><td>A</td><td>55.06/62.64/68.01</td><td>87.50/81.25/87.50</td><td>81.58/75.75/84.38</td></tr><tr><td>B</td><td>N</td><td>52.65/57.85/65.16</td><td>0.00/0.00/0.00</td><td>4.76/10.72/11.69</td></tr><tr><td>QB</td><td>0</td><td>41.45/57.08/64.61</td><td>93.75/100.00/100.00</td><td>84.56/99.49/99.20</td></tr><tr><td>QB</td><td>C</td><td>39.00/57.19/70.39</td><td>100.00/100.00/100.00</td><td>94.68/94.24/91.24</td></tr><tr><td>QB</td><td>E</td><td>34.35/57.69/69.98</td><td>6.25/25.00/46.88</td><td>10.28/28.58/50.42</td></tr><tr><td>QB</td><td>A</td><td>48.61/60.98/72.19</td><td>62.50/84.38/84.38</td><td>58.07/78.50/83.14</td></tr><tr><td>QB</td><td>N</td><td>40.71/55.82/75.55</td><td>0.00/0.00/3.13</td><td>4.74/11.87/11.76</td></tr></table>

Table F2. Adjacent-level directions. Increases/ties/decreases score 1/0.5/0 across ten adjacent facet-mean comparisons.
<table><tr><td rowspan="2">Training arm</td><td rowspan="2">Facet means: up/tie/ down</td><td rowspan="2">Direction score M  $D _ { r }$ </td><td colspan="3">Item-level: up/tie/</td></tr><tr><td>Nondecrease</td><td>down</td><td>Item nondecrease</td></tr><tr><td>Q</td><td>4/4/2</td><td>60%</td><td>8/10 (80%)</td><td>7/70/3</td><td>77/80 (96.25%)</td></tr><tr><td>B</td><td>2/6/2</td><td>50%</td><td>8/10 (80%)</td><td>5/71/4</td><td>76/80 (95.00%)</td></tr><tr><td>QB</td><td>5/5/0</td><td>75%</td><td>10/10 (100%)</td><td>14/65/1</td><td>79/80 (98.75%)</td></tr></table>

Joint-supervision observed choices show five increases and five ties, with clearer diferentiation on E/A. Conditional probabilities show six increases and four decreases, retaining variation hidden by discrete choices. Shared baseline O/C/E/A/ N scores are 100/100/21.88/87.50/3.13. Per-item records retain invalid denominators, background responses, and option orders.

## F.2 Early Open-Generation Diagnosis

Before quality filtering, original-model questionnaire, first-person questionnaire, and first-person joint pipelines generated 300 thinking-mode open answers. Naturally ended, complete outputs among 100 per arm were 62/92/53. Across 20 vector-

only pairs per arm, positive/tied/reversed/unjudgeable diferences were 1/9/2/8, 2/6/0/12, and 3/2/0/15. This diagnosis motivated separate quality checks for open outputs. Raw responses and archived labels remain available; primary external analyses use the frozen protocols in F.1.

## F.3 Chinese PersonalityBench Generation and Blind Judging

Section 5.1 uses Qwen3.5-27B with the same first-person LoRA, thinking disabled, greedy decoding, a 512-new-token cap, and repetition penalty 1.0. The system instruction remains “You are a person; answer according to the facts of the scenario.” Composite-score levels select nodes but are not written into personality prompts. Fifteen dimension–level configurations each answer all Chinese items for their focal dimension once. No node is reselected based on response quality.

Judging uses the oficial DeepSeek API, requested model deepseek-flash, thinking disabled, temperature zero, and JSON output with a 1,800-token cap. Inputs contain only the dimension, scenario question, and raw response. Ratings 1/2/3/4/5 mean clearly low, somewhat low, genuinely intermediate or mixed, somewhat high, and clearly high. Missing trait evidence is marked missing rather than assigned three. Ratings prioritize expressed choices, intended actions, preferences, and emotions; self-labels and politeness alone do not establish trait expression. Paired endpoint diferences use only same-item pairs with both responses scoreable.

The Chinese items are this study’s translation, not an oficial Chinese release, and retain all 459 source identifiers. The archive preserves translations, manual adjustments, selected profiles, the rubric, original responses, API-returned model identifiers, and individual scores. Table 4 is aggregated from these records. Source and protocol records are indexed in Appendix H.4.

## Appendix G Cross-Dimensional Structure and Population Reference

## G.1 Joint-Catalog Correlations

Human trait correlations have been studied through higher-order factors and common rating efects (DeYoung, 2006; Anusic et al., 2009). We use one population sample as a sign reference, not as a causal model of internal interventions.

All3,125 unfiltered five-dimensional configurations are equally weighted to compute Pearson correlation signs of the composite dimension scores. The population reference is the lower triangle of Table3, Study1, in Føllesdal and Soto’s Norwegian BFI-2 study, with pairwise sample sizes599–601 (Føllesdal & Soto, 2022).

Table G1. Signs for ten unordered dimension pairs.

<table><tr><td>Dimension pair</td><td>Population sign</td><td>Model-score sign</td></tr><tr><td>O—C</td><td>+ (near zero)</td><td>+</td></tr><tr><td>O—E</td><td>+</td><td>+</td></tr><tr><td>OA</td><td>+</td><td>+</td></tr><tr><td>ON</td><td>- (weak)</td><td>一</td></tr><tr><td>C—E</td><td>+</td><td>+</td></tr><tr><td>C—A</td><td>+</td><td>+</td></tr><tr><td>C—N</td><td>一</td><td>一</td></tr><tr><td>E—A</td><td>+</td><td>+</td></tr><tr><td>E—N</td><td>一</td><td>一</td></tr><tr><td>A—N</td><td>一</td><td>一</td></tr></table>

All ten composite dimension-score correlation signs agree with the population point-estimate signs. The catalog and population sample thus exhibit similar cross-dimensional association patterns, providing a descriptive reference for interpreting joint coordinates.

## G.2 Directed Single-Axis Responses before and after Refinement

Fix other coeficients at zero and compute $D _ { j k } ( h ) = [ F _ { j } ( h e _ { k } ) - F _ { j } ( - h e _ { k } ) ] / ( 2 h )$ . Twenty directed of-diagonal efects are compared with the corresponding population signs at $h = 1 , 2 ,$ , holding layers fixed between initialization and refinement.

Table G2. Directed-response sign agreement.

FPSRG: Heterogeneous User Simulation
<table><tr><td rowspan="2">Half-width h</td><td rowspan="2">Interface</td><td rowspan="2">Initial /20</td><td rowspan="2">Refined /20</td><td rowspan="2">Initial /16, weak excluded</td><td rowspan="2">Refined /16, weak excluded</td></tr><tr><td></td></tr><tr><td>1</td><td>Questionnaire score Q</td><td>16</td><td>18</td><td>14</td><td>15</td></tr><tr><td>1</td><td>Behavior score B</td><td>17</td><td>13</td><td>13</td><td>10</td></tr><tr><td>1</td><td>Composite score S</td><td>17</td><td>16</td><td>13</td><td>13</td></tr><tr><td>2</td><td>Questionnaire score Q</td><td>17</td><td>19</td><td>14</td><td>16</td></tr><tr><td>2</td><td>Behavior score B</td><td>17</td><td>13</td><td>13</td><td>10</td></tr><tr><td>2</td><td>Composite score S</td><td>17</td><td>15</td><td>13</td><td>12</td></tr></table>

Questionnaire-score agreement rises from16/20 to18/20 at h = 1, and17/20 to19/20 at h = 2. TableG2 explicitly distinguishes questionnaire scores Q, behavioral scores B, and composite scores S: behavioral agreement changes from17/20 to13/20; composite agreement changes from17/20 to16/20 or15/20. TableG1 reports only composite-score correlations. Directed intervention efects locate spillovers from individual directions; catalog correlations describe the overall configuration geometry. AppendixH.4 indexes the complete matrices and raw-response identities.

## Appendix H Execution, Timing, and Evidence

## H.1 Per-Token Routing and Scheduling for Complete Generation

The six-dimensional thinking instance uses independent single-GPU vLLM replicas. Each replica loads the same full mode and node-vector bank, with at most 32 active sequences and a 2,048-token prefill budget per scheduling step. Output and total-context limits are 4,096 and 32,768 tokens. The model uses BF16, with an FP32 output head and vector bank; no LoRA is loaded and CUDA Graph is disabled. Explicit mappings convert node identifiers to bank indices; node identifiers are not array ofsets.

## Algorithm H1. Continuous generation and calibration with mixed configurations.

Inputs: frozen model M\_FP, directions V, configurations G,   
items and repeats X, protocol pi, replica count P   
Output: response catalog C\_pi   
1 B <- FP32SumByLayer(V, G); p <- BuildPersonaIndex(G)   
2 W <- EnumerateRequests(G, X, pi) # node, prompt, seed, budget   
3 parallel for replica r = 1, ..., P:   
4 LoadOnce(M\_FP, B); CheckMixedAndReorderedRouting()   
5 while unfinished work remains:   
6 AdmitUncommittedRequests(W, atomic\_claim=True)   
7 U <- ScheduleChunkedPrefillAndDecode()   
8 route <- RebuildTokenOwnersAfterReorder(U, p)   
9 key <- BindCacheToModelVectorsProtocolAndPersona(U)   
10 Forward(U, key, layer\_injection=B[layer, route])   
11 SampleEachRequestUsingItsOwnParameters()   
12 for each completed request:   
13 AtomicCommit(raw\_thought, final, finish, seed, digest)   
14 ReleaseSlotAndAdmitNext()   
15 return ParseFinalsAndAggregate(all\_committed, pi)

The engine manages paged caches, whose identities additionally bind the model, vector bank, and profile. Diferent configurations do not share post-injection states. Startup checks include A/B/A profiles in one batch and reversed request order to detect routing errors after scheduler reordering. Raw reasoning, final answers, completion status, and invalid outputs are all saved before scoring. Recovery fills only unfinished work; uncertain commits are reconciled rather than blindly replayed.

With 16 GPUs, demonstration generation through final verification takes 4 h 11 min 18 s. Formal measurement of 559,872 complete outputs takes 3 h 5 min 4 s, averaging 50.42 answers/s overall and 3.15 per GPU. This stage includes startup overhead, excludes the preceding 54 min 28 s of full-parameter post-training, and is not a paired speedup against the older first-token engine. The chat service reuses token routing with a 16,384-token context and lower concurrency limits, without altering catalog scores.

## H.2 Boundary Reuse and Cost in Earlier First-Token Measurement

Earlier non-thinking experiments reuse states according to intervention dependencies. Computation before the first injection is coeficient-independent; later states depend only on upstream coeficients that have already acted, delaying branches for deeper coeficients. Boundary-cache keys bind model, adapter, item tokens, positions, masks, vectors, and upstream coeficients, and retain complete continuation states. This difers from the persona-isolated prefix caching in Appendix H.1.

Let zero-indexed injection layers be $b _ { 1 } < \dots < b _ { K }$ , with $g _ { s }$ combinations at layer group s and $\begin{array} { r } { P _ { s } = \prod _ { u \leq s } g _ { u } } \end{array}$ upstream combinations. Ideal boundary reuse has block-execution counts

$$
\begin{array} { l } { { \displaystyle C _ { \mathrm { i n d } } = N _ { G } L , } } \\ { ~ } \\ { { \displaystyle C _ { \mathrm { r e u s e } } = ( b _ { 1 } + 1 ) + \sum _ { s = 1 } ^ { K - 1 } P _ { s } ( b _ { s + 1 } - b _ { s } ) } } \\ { ~ } \\ { { \displaystyle ~ + N _ { G } ( L - 1 - b _ { K } ) . } } \end{array}
$$

This count explains frozen-forward savings from delayed branching, not complete autoregressive generation time. Original checks cover zero vectors, signs, masks, padding, causality, and cached decoding. Option log-probability and probability tolerances are $1 0 ^ { - 4 }$ and $1 0 ^ { - 5 }$ , with separate cross-replica anchor checks.

The ten-dimensional design uses 640 first-token probes per node, totaling 37,791,360 probes for 59,049 nodes. One shared adapter and dH direction parameters serve all profiles; catalog storage grows linearly with measured nodes. Full Cartesian enumeration remains multiplicative in dimension count. Acceleration removes repeated frozen computation rather than training a separate model for each configuration.

The original ten-dimensional catalog takes approximately 13.56 hours from admission to completion, using at most 16 replicas. Its five-dimensional reference reuses 1,800 nodes and measures 1,325 more in approximately 40.6 minutes. An early alternative vLLM implementation did not meet the then-current numerical and speed criteria, so those first-token results use the original pipeline. The new complete-generation implementation in Appendix H.1 builds a separate sixdimensional catalog and does not replace historical measurements.

## H.3 Density-Study Timing and Unfiltered Results

The nine-level catalog contains 59,049 nodes: 3,125 reused integer nodes, 14,183 previously completed half-step nodes, and 41,741 newly measured nodes. Sixteen replicas complete supplementation and aggregation in approximately ten hours after numerical checks and paired benchmarking. This is not a from-scratch runtime for all nine-level nodes.

Table H1. Paired same-device acceleration. Each host repeats three comparisons of three full nodes per path. Batch size 32, precision, and SDPA/FLA operators are unchanged. Reference execution disables cross-node boundary-state reuse, not all optimizations.
<table><tr><td>Host</td><td>Reference s/node</td><td>Fill-inclusive s/node</td><td>Warm s/node</td><td>Fill-inclusive speedup</td><td>Warm speedup</td></tr><tr><td>Host group 1</td><td>35.592</td><td>20.497</td><td>12.880</td><td>1.736</td><td>2.763</td></tr><tr><td>Host group 2</td><td>34.563</td><td>19.879</td><td>12.492</td><td>1.739</td><td>2.767</td></tr></table>

Maximum paired diferences in option probabilities, log probabilities, and legal mass are zero on both hosts. Observed formal throughput is approximately 24 first-token probes per GPU per second, depending on cache reuse and node grouping. Parallelism is distinct from the same-device algorithm speedup in Table H1.

Without filtering, three/five/nine-level composite broad coverage in 25–75 is 92.3%/95.3%/96.4%; strict coverage is 31.4%/ 55.5%/64.1%. Table 3 reports qualified coverage. Retaining both separates raw reachability from usable regions under quality constraints. Table 3 reports density outcomes; Table H1 reports paired timing.

## H.4 Evidence Organization

The evidence chain is protocol and inputs → raw commits → analysis → paper tables. Supporting files retain item identities, source digests, repeated-generation denominators, missing responses, background coordinates, and quality. The source index below identifies the main catalogs, component studies, and this revision. Scores remain response coordinates tied to the specified evaluation protocol.

Provenance record identifiers (retained by the authors; not bundled with the typesetting source):

<table><tr><td>Study</td><td>Supporting record</td></tr><tr><td>Templates and splits</td><td>evidence/current_prompt_examples.json</td></tr><tr><td>Main catalogs and complementarity</td><td>evidence/completed_catalog_coverage.json; evidence/v25_extension_verified.json</td></tr><tr><td>Six-dimensional full thinking workflow</td><td>evidence/huihui_six_pipeline_20261104.json</td></tr><tr><td>Fixed-layer comparisons and retrieval</td><td>evidence/fixed_layer_coverage_extract.json;</td></tr><tr><td>Matched-data refinement</td><td>evidence/joint_vs_additive_summary.json evidence/same data refinement 20261025 curves.csv;</td></tr><tr><td></td><td>evidence/same_data_refinement_20261025_provenance. json</td></tr><tr><td>Injection scope</td><td>evidence/injection_scope_20261008_summary.json</td></tr><tr><td>Joint prediction and held-out repeats</td><td>evidence/joint_additivity_20261025_results.json</td></tr><tr><td>Ten-dimensional non-thinking repeats</td><td>evidence/tenaxis_off5_node_scores.csv; evidence/tenaxis_off5_independent_audit.json</td></tr><tr><td>Cross-axis directions</td><td>evidence/directional_refinement_20261025.json;evid ence/directional_grid_correlation_20261025.json</td></tr><tr><td>Density and timing</td><td>evidence/density_complete.json;</td></tr><tr><td>Score bins and threshold sensitivity</td><td>evidence/density_benchmark_summary.json evidence/static_bins_20261026.json</td></tr><tr><td>Six-method HEXACO comparison</td><td>evidence/hexaco_six_methods_joint_coverage.json; evidence/hexaco_six_methods_joint_nodes.json;</td></tr><tr><td>Market interaction case</td><td>evidence/hexaco_six_methods_range_records.json</td></tr><tr><td>Revision audit</td><td>evidence/seller_market_case_20261027.json evidence/revision_v28.json</td></tr><tr><td>Chinese PersonalityBench</td><td>evidence/pb_scene_revision_20261027.json;</td></tr><tr><td></td><td>evidence/pb_reproduction_20261027.json</td></tr><tr><td>Main-aligned HEXACO ranges</td><td>evidence/hexaco_mainaligned_range_analysis.json</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td>Main-aligned HEXACO joint results</td><td>evidence/hexaco_mainaligned_joint_coverage.json;</td></tr><tr><td></td><td></td></tr><tr><td></td><td>evidence/hexaco_mainaligned_joint_nodes.json;</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr><tr><td></td><td>evidence/hexaco_mainaligned_verified.json</td></tr><tr><td></td><td></td></tr><tr><td></td><td></td></tr></table>

## Appendix I Public Six-Dimensional Inventory: Construction, Range, and Joint Measurement

## I.1 Direction Construction and Complete-Response Scoring

The IPIP HEXACO inventory is split by 24 facets, each contributing six training, two development, and two test items. Each dimension has eight test items and three complete-response samples under each of two numeric mappings. Thinking is disabled; temperature is 0.7, top-p is one, top-k is zero, the cap is 192 new tokens, and batch size is 16. Selected layers, a relative reference-norm budget of 0.1, and full-input/decode injection remain fixed during evaluation.

Figure 1 presents the fixed-range curves. Original English items and keys are from the IPIP public inventory, based on Ashton, Lee, and Goldberg (2007; bibliographic information). HEXACO E denotes emotionality and X denotes extraversion, distinct from the Big Five abbreviations.

CAA, the independent linear probe, role-endpoint initialization, and refinement use the same training items. CAA averages high-minus-low activation contrasts at answer positions; the independent linear probe fits a classification direction to those high/low activations. Role-endpoint initialization prefixes each questionnaire prompt with a high or low trait description, keeping facts, questions, and encoding unchanged. It averages the high-minus-low contrasts at the last valid prompt position. Reference scale is the median L2 norm at the last position of role-free training prompts. Refinement optimizes this initialization for 240 steps with choice loss, complete self-report loss, neutral preservation, and norm regularization, without behavioral training items. PERSONA uses twenty training items per dimension and five elicitation/suppression prompt

pairs to generate 2,400 construction responses. Oficial DeepSeek numeric ratings filter trait expression and coherence;   
content-mean activation contrasts then construct independent high and low poles.

Candidate outputs are decoder blocks 8, 16, 24, 32, 40, 48, 56, and 60, numbered from one. All methods except PERSONA select layers independently per dimension on development data; PERSONA selects one common layer across twelve poles. Directly constructed directions have fixed length equal to 0.1 times the reference scale. Refinement and BiPO project into the same norm ball after every update and retain their learned lengths without renormalization; this matches norm budgets, not exact final lengths. The independent probe represents a single-axis classification direction, not SAS’s sequential multi-axis augmentation. PERSONA uses inventory-response development selection, matched norms, and full-sequence injection. These comparisons adapt construction methods to shared data interfaces, norm budgets, and injection rules.

BiPO (Cao et al., 2024) initializes the vector at zero and applies a randomly sampled positive or negative coeficient on each update. Its bidirectional DPO objective uses summed log probabilities of complete high/low self-report responses, including the end token, relative to the frozen unsteered model. Both A/B mappings receive equal weight. AdamW uses weight decay 0.05, gradient clipping at one, and 240 updates, without our auxiliary choice, self-report cross-entropy, or neutral-preservation terms. We parameterize v = 0.1ru and project u into the unit L2 ball. Development selection considers $\beta \in \{ 0 . 1 , 0 . 5 \}$ and constant learning rates {0.001, 0.003} for dimensionless u at each of the same eight layers, giving 32 candidates per dimension. BiPO has four hyperparameter settings per layer versus one for refinement. This matched-data, norm-budget protocol retains BiPO’s zero initialization and bidirectional objective, but uses the shared 240-update budge rather than its original dataset and 100-epoch schedule.

PERSONA’s negative half-axis uses its independent low-pole direction with positive strength; its positive half-axis uses the high pole. Both share the unsteered origin. Complete-response means are computed within item–mapping cells and then weighted equally. Unparseable answers are not imputed as neutral scores.

## I.2 Adaptive Range and Response Validity

Six methods, six dimensions, and two signs yield 72 branches. Coarse absolute strengths are 0.5, 1, 1.5, 2, 3, 4, 6, 8, 10, 12, 16, 24, 32, 48, 64, and 100. Identical existing conditions are reused. At the first outward failure below 90% validity, the last passing/first failing interval is bisected to width at most 0.25. At least 44 of 48 answers must pass. All branches encountered failure before reaching 100.

Valid answers terminate normally, begin with a legal 1–5 digit, include a brief explanation, and contain no thinking tags. The check does not include human semantic-coherence judgments. Means are computed within item–mapping cells before equal-weight aggregation. Invalid texts are preserved, not assigned 50. Usable span is the maximum minus minimum score among measured conditions that pass before the first outward failure; connecting curves does not establish behavior at unmeasured coeficients.

Table I1. Refined observed ranges. Boundary columns give absolute-strength passing/failing brackets based on the first 48 answers, not confidence intervals.
<table><tr><td>Dimension</td><td>Minimum-maximum</td><td>Span</td><td>Negative bracket</td><td>Positive bracket</td></tr><tr><td>H</td><td>47.40—94.27</td><td>46.87</td><td>5.00—5.25</td><td>2.50—2.75</td></tr><tr><td>E</td><td>27.60—60.42</td><td>32.81</td><td>7.50—7.75</td><td>7.00—7.25</td></tr><tr><td>X</td><td>8.33—64.84</td><td>56.51</td><td>1.50—1.75</td><td>1.25—1.50</td></tr><tr><td>A</td><td>11.4696.88</td><td>85.42</td><td>6.75—7.00</td><td>6.25—6.50</td></tr><tr><td>C</td><td>31.77—72.40</td><td>40.62</td><td>1.25—1.50</td><td>1.25—1.50</td></tr><tr><td>0</td><td>29.95—73.44</td><td>43.49</td><td>1.75—2.00</td><td>1.00—1.25</td></tr></table>

## I.3 Boundary Remeasurement and Records

Each boundary endpoint receives another 48 answers, pooled with the original 48. Of 144 endpoints, 44 change their pass/ fail classification after pooling 96 responses. Thus 0.25 is search resolution, not boundary confidence precision. Excluding initially passing points that fail the pooled threshold gives mean usable spans of 50.78 for refinement, 45.14 for initialization, 41.23 for BiPO, 37.72 for adapted PERSONA, 29.47 for CAA, and 29.17 for the independent probe. Refinement retains the largest mean span.

The scan reuses matching strengths from 294 fixed-curve conditions and measures 636 further conditions, comprising

![](images/00ba2b2b4d6cdbda87386a7e30c04eaeaee9a7b280f54f58ee067f856c921f8b.jpg)  
Figure I1. Expanded complete-response score curves. Dots meet the initial 48-response validity threshold of at least 90%; crosses do not.

![](images/e7c1faa2572509470619b5dde75424ecb6f89d5fcd930e4d4dfa8ca44f1f22f3.jpg)  
Figure I2. Complete-response validity for the same conditions. Dashed lines mark the 90% threshold.

30,528 response attempts. All raw-record hashes were verified. Scanning does not retrain directions or reselect layers.   
Per-branch boundaries, norms, and pooled repeats are preserved in evidence/hexaco\_six\_methods\_range\_records.   
json.

## I.4 Six-Dimensional Joint Measurement

Each method uses −1, 0, and +1 per dimension, giving 729 combinations and 4,374 total nodes with 419,904 item measurements. All are complete. Models, LoRA, directions, layers, and full-input/decode injection match the single-axis experiments; PERSONA −1 still denotes the independent low pole. Each node evaluates 48 items under two numeric mappings at temperature 0.7. Conditional expectations over legal 1–5 labels are keyed to 0–100 and averaged equally over sixteen presentations per dimension.

Quality requires mean full-vocabulary legal numeric mass ≥0.98 per dimension and an absolute gap ≤25 between dimension means under the two mappings. With no behavioral A/B interface, only these two conditions apply, rather than the four-part cognitive–behavioral gate. A legal probability of 98% is not a complete-generation validity rate. Complete-response curves and first-token catalogs use distinct scoring interfaces.

Seed 2026102606 generates 10,000 six-dimensional unit-cube targets, rescaled to each domain. Strict coverage requires one candidate with all coordinate errors <10 and total error <36. MAE averages coordinatewise absolute errors of ordinary nearest-L1 retrieval. The statistics describe the measured catalog, not a new independent generation repeat. Test items are held out from training and selection, while catalog construction and static coverage share the same evaluation interface.

Table I2. Strict coverage and matching errors across target domains. All results use the same quality conditions. Coverage is in percent and MAE in score points.
<table><tr><td>Method</td><td>25–75 strict</td><td>25—75 MAE</td><td>20–80 strict</td><td>20—80 MAE</td><td>30–70 strict</td><td>30—70 MAE</td></tr><tr><td>CAA</td><td>4.94</td><td>10.20</td><td>2.61</td><td>12.14</td><td>10.20</td><td>8.37</td></tr><tr><td>Independent probe</td><td>4.23</td><td>10.81</td><td>2.22</td><td>12.72</td><td>7.47</td><td>9.02</td></tr><tr><td>Role initialization</td><td>9.08</td><td>8.70</td><td>4.60</td><td>10.62</td><td>20.80</td><td>6.87</td></tr><tr><td>Adapted PERSONA</td><td>5.31</td><td>10.04</td><td>2.97</td><td>11.98</td><td>10.08</td><td>8.23</td></tr><tr><td>BiPO</td><td>3.95</td><td>10.39</td><td>2.30</td><td>11.99</td><td>7.42</td><td>8.78</td></tr><tr><td>Refined</td><td>12.56</td><td>8.13</td><td>6.49</td><td>9.87</td><td>25.24</td><td>6.50</td></tr></table>

Table I3. Unfiltered results in 25–75. Other domains and bin statistics are retained in the supporting records.
<table><tr><td>Method</td><td>Strict coverage (%)</td><td>Matching MAE</td></tr><tr><td>CAA</td><td>5.52</td><td>9.98</td></tr><tr><td>Independent probe</td><td>6.29</td><td>9.87</td></tr><tr><td>Role initialization</td><td>9.37</td><td>8.66</td></tr><tr><td>Adapted PERSONA</td><td>5.31</td><td>10.04</td></tr><tr><td>BiPO</td><td>4.45</td><td>10.21</td></tr><tr><td>Refined</td><td>13.48</td><td>8.07</td></tr></table>

The comparison fixes each method’s number of combinations and item measurements, without assuming identical execution time across selected layers. Initial and refined methods each select layers on development data, so this experiment evaluates their final construction pipelines. Section 4.5 separately isolates refinement with layers fixed. Node-level results, pre/ post-filter summaries, and independent raw-probability checks are indexed in Appendix H.4.