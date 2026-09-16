# Afect-Prototype Guided Fusion for Open-Vocabulary Incomplete Multi-modal Emotion Recognition

Yichi Zhang<sup>∗1</sup>, Shenyue Wang<sup>∗2</sup>, Jing Luo<sup>1</sup>, Chunyang Yu<sup>†2</sup>, Xinyu Yang<sup>†1</sup>

<sup>1</sup>School of Computer Science and Technology, Xi’an Jiaotong University, Xi’an, China

<sup>2</sup>OPPO Research Institute, Shanghai, China

datasonezyc@stu.xjtu.edu.cn,wangshenyue@whu.edu.cn,chinglo@xjtu.edu.cn,chunyang.yu@ucalgary.ca,yxyphd@mail.xjtu.edu.cn

## Abstract

Open-vocabulary multimodal emotion recognition (OV-MER) aims to generate open natural-language emotion labels from multimodal afective cues. In real-world scenarios, however, complete and synchronized modal data are dificult to obtain due to limitations of acquisition devices and user privacy constraints. Existing OV-MER methods are largely designed for full-modal inputs, and fail to perform efective feature fusion under modal missing conditions. Meanwhile, current fusion approaches designed for incomplete modalities mainly focus on fixed-label recognition context, and cannot satisfy the demand for fuse emotional cues guided with arbitrary emotion semantics in OV-MER context. To tackle these challenges, this paper proposes an Afect-Prototype-Conditioned Fusion (APCF) framework for incomplete open-vocabulary emotion recognition. As a candidate-free generative framework, APCF extends modal contribution learning to scenarios guided by arbitrary emotional semantics. Specifically, we construct an afect-prototype library to explicitly model multimodal contribution characteristics corresponding to diverse emotions, which provides dynamic constraints for modal fusion under diferent emotional semantic perspectives. Conditional retrieval and feature aggregation are conducted based on available modal features. The refined fused afective representations are then fed into an LLM decoder to produce openvocabulary emotion labels. Experiments on the OV-MERD+ and MER-FG datasets demonstrate that APCF substantially outperforms state-of-the-art baselines.

## Introduction

Multimodal emotion recognition (MER) combines complementary audio, visual, and textual cues to infer a user’s affective state. Most traditional approaches formulate MER as classification over a fixed inventory of emotion words as labels (Zhang et al. 2022). Although this formulation supports straightforward supervision and evaluation, a small label inventory compresses nuanced states and cannot express concepts outside the predefined classes. Continuous emotion recognition instead predicts values in dimensional emotion labels to express arbitrary afect states (Praveen et al. 2022), however the dimension-based labels naturally present huge semantic gap between the human-understandable descriptions and dimensional descriptions, limiting dataset annotation and further usage on end-user facing systems. OV-MER proposed predicting emotion phrases in natural language for balancing semantic coverage with human readability (Lian et al. 2025c). Such phrases can describe intensity, mixtures, and fine distinctions, but they also enlarge the semantic space that must be connected to sensory evidence. Afect-GPT proved multi-modal large language models (MLLMs) is capable for generating fine-grained afect description based on multi-modal emotion evidence (Lian et al. 2025a). Recent work further advances this interface through perceptionoriented policy optimization (Han et al. 2026) and reinforcement learning (Lian et al. 2026b). Challenge-oriented generative evaluation (Lian et al. 2026a) and hybrid-evidential deduction (Liu et al. 2026) likewise support open-vocabulary emotion understanding.

Most current open-vocabulary MER pipelines nevertheless assume that all modalities are available. That assumption is fragile in practice: obtaining complete and synchronous modality data would be hard due to issues like sensor malfunction, data collection limitation and user privacy concerns. User may not look into the camera, or even reject giving permissions on capturing visual data. Transcriptions may not collect enough meaningful semantic information with heavily-accented or mainly-interjection speech. Incomplete modality data will hamper learning and reasoning abilities of the language modal on the relationship between afect phrase semantic and multi-modal emotional evidence. Lack of crucial trained emotion features, such as micro-expression, or the semantic information inside transcription, would impact emotion reasoning when fusing multi-modal emotional cues, and reduce the emotion recognition performance.

Incomplete-MER research has addressed missing inputs through reconstruction (Zhao, Li, and Jin 2021), learned prompts (Guo, Jin, and Zhao 2024), and flexible expert routing (Han et al. 2024). Other approaches use crossmodal queries (Miyoshi, Otani, and Okafuji 2026), balanced prompting (He, Zhu, and Zhang 2026), or uncertainty-aware difusion (Qiu et al. 2026). These methods improve the use of partial data for emotion understanding and reasoning. However, current algorithm design focused on semantic understanding under fixed-label context, not able to adapt its ability on learning emotional modality cues and fusion matrices to the arbitrary emotion space in open-vocabulary context. Suppose that a system is trained mainly with fearful but is evaluated on apprehensive. A language model may know that the phrases are related, yet the sensory pathway may not know that they should draw on a similar mixture of hesitant speech, tense facial behavior, and uncertain wording. As well as how to fully utilize the mixture of emotion cues to better instructing language models under incomplete multi-modal circumstances.

![](images/90290751b2e86b1e0926cc99d64690f476fbd72c85270031ae2784eb09604881.jpg)  
Figure 1: Overview of the proposed APCF framework.

We propose Afect-Prototype–Conditioned Fusion (APCF) to address this problem. APCF maintains a set of overlapping landmarks in emotion-language space called afect-prototypes, which are aligned between categorical emotion word semantic features, and label coordinates in the dimensional emotion space. A prototype is a reusable query that conditions shared computation, it learns a range of similar emotion evidence-wise, and a evidence fusion algorithm for emotions inside the prototype. For each observed audio, video, and text stream, APCF extracts prototype-specific evidence and combines the available modality streams with an observed-set fusion module. The resulting prototype memory is exposed to a pretrained language decoder. Its causal prefix determines which semantic regions and which samplespecific evidence slots are relevant, instructing the decoder to fully reason and generate open-vocabulary emotion descriptions with incomplete modality cues.

Our main contributions are listed as follows:

1. We extend open-vocabulary MER to incomplete multimodal scenario. This paper formulates incomplete openvocabulary MER as candidate-free generation from any nonempty subset ofmodalities, bridging open-vocabulary recognition and incomplete-modality fusion.

2. We construct semantic and modality-aware afectprototypes. Transferable prototypes are learned from paired categorical and dimensional source supervision, then adapted to target emotion phrases through bounded, regularized residuals. They organize both afect semantics and modality-specific evidence-use patterns.

3. We designed a novel modality fusion algorithm. We use the prototypes to condition timestamped evidence retrieval, observed-only set fusion, and causal routing into a language encoder for reasoning and generating openvocabulary emotional phrases. The algorithm achieves state-of-the-art (SOTA) on incomplete open-vocabulary MER task.

## Related Work

## Open-vocabulary Emotion Recognition

OV-MER replaces closed emotion classes with naturallanguage terms and evaluates predictions through semantic normalization and an emotion-wheel representation (Lian et al. 2025c). AfectGPT extends this formulation with emotion-oriented multi-modal instruction data and an audiovisual language-model scafold (Lian et al. 2025a). Emotion-LLaMA uses multi-modal instruction tuning to combine emotion recognition with explanation and reasoning (Cheng et al. 2024). Agent-MER applies hierarchical agent deliberation (Lai et al. 2025), Clue2Emo organizes multimodal clues before prediction (Zhang et al. 2026), Nano-EmoX develops a compact multitask afect model (Huang et al. 2026), and OPPO optimizes multimodal emotion reasoning with a perception-oriented policy objective (Han et al. 2026). AfectGPT-RL studies reinforcement learning for open-vocabulary recognition (Lian et al. 2026b), HyDRA performs structured clue-based deduction (Liu et al. 2026), AfectAgent coordinates retrieval-augmented agents (Wang et al. 2026), and AfectVerse predicts latent audiovisual futures for afective reasoning (Zhao et al. 2026). Together, these methods strengthen open-ended perception and reasoning. Some include missing-input mechanisms, but missingness is not uniformly studied across controlled modality subsets.

## Emotion Recognition with Incomplete Modalities

The Missing Modality Imagination Network (MMIN) reconstructs unavailable representations through cross-modal imagination and cycle consistency (Zhao, Li, and Jin 2021). The multi-modal prompt-learning method of Guo et al. associates learned prompts with missing-input patterns (Guo, Jin, and Zhao 2024), while the query-based method of Miyoshi et al. combines unimodal and cross-modal evidence (Miyoshi, Otani, and Okafuji 2026). BALM balances training under unequal missing rates (Nguyen et al. 2026), and SimMLM trains over nested modality subsets so that additional observations do not degrade predictions (Li, Chen, and Han 2025). FuseMoE and Flex-MoE support arbitrary modality combinations through mixture-of-experts routing (Han et al. 2024; Yun et al. 2024). ComP propagates cross-modal prompts and consensus information between modality branches to balance fixed-label predictions (He, Zhu, and Zhang 2026). HyperEF uses hypergraph-conditioned difusion to recover latent missing features and models source- and decision-level uncertainty in fixed-label conversational emotion recognition (Qiu et al. 2026). These approaches establish strong mechanisms for partial inputs, but their fusion modules are optimized primarily through fixed-class decisions. Applying them to a generator does not by itself provide an emotion-semantic query that transfers evidence-use patterns to previously unseen phrases.

## Semantic Prototypes and Afect Representation

Semantic information has long been used to organize representations across labels and modalities. UniBind uses language-enhanced prototypes to construct a shared multimodal space (Lyu et al. 2024). The semantic prompting method of Pipoli et al. conditions visual recognition on available-modality scenarios (Pipoli et al. 2025), and the label-semantic model of Gaonkar et al. uses label descriptions to guide attention and relations between emotional reactions (Gaonkar et al. 2020). The label-agnostic embeddings of Buechel et al. bridge heterogeneous categorical inventories (Buechel, Modersohn, and Hahn 2021). Park et al. map categorical emotions to dimensional coordinates (Park et al. 2021), while the afective-manifold method of Li et al. anchors discrete terms in continuous afective structure (Li et al. 2026).

## Methodology

## Problem Formulation

Let $\mathcal { M } = \{ A , V , T \}$ denote audio, video, and text. Sample i contains a nonempty observed subset $S _ { i } \subseteq { \mathcal { M } }$ and an availability vector $a _ { i }$ that records which streams are present. A frozen modality encoder $E _ { m }$ maps each observed stream to

$$
H _ { i } ^ { m } = \{ ( h _ { i , m , t } , \tau _ { i , m , t } ) \} _ { t = 1 } ^ { T _ { i } ^ { m } } , \qquad m \in S _ { i } ,\tag{1}
$$

where $h _ { i , m , t }$ is an encoded token and $\tau _ { i , m , t }$ is its normalized position in the clip. Every valid token follows this timestampaware representation. A naturally pooled stream is represented by a single valid token rather than expanded into an artificial temporal sequence. If a modality is unavailable, its encoder output is not created and no zero or learned missingmodality token is inserted into the evidence set.

The target $Y _ { i } = ( y _ { i , 1 } , \dots , y _ { i , N _ { i } } )$ is an ordinary tokenizer sequence containing one or more free-form emotion phrases. APCF is therefore not given an emotion-specific output vocabulary or a test-time candidate list. Given a task prompt $Q _ { i }$ and an evidence memory $Z _ { i } ^ { S _ { i } }$ built only from the observed streams, prediction is factorized causally as

$$
p ( Y _ { i } \mid X _ { i } ^ { S _ { i } } ) = \prod _ { n } p ( y _ { i , n } \mid y _ { i , < n } , Q _ { i } , Z _ { i } ^ { S _ { i } } ) .\tag{2}
$$

This formulation separates two requirements. The sensory pathway must reorganize whatever evidence remains under $S _ { i }$ , while the language pathway must express that evidence using unrestricted emotion language. The structure of APCF is shown in Figure 1. APCF connects the two through afect prototypes: overlapping semantic queries that organize evidence without becoming output classes. The model contains $K$ prototypes and L reference-time slots. Throughout this section, $W _ { * }$ denotes learned projections, and all unavailable streams are excluded from the corresponding operations.

## Afect-Prototype Bank

APCF represents afect using overlapping landmarks rather than mutually exclusive output classes. The source bank is constructed only from samples for which categorical and dimensional annotations are paired. Prototype slots are initialized from source emotion categories, but a sample may assign mass to several slots. Prototype $P _ { k } ^ { S }$ contains a language anchor $c _ { k , S } ^ { L } ,$ a dimensional center $\begin{array} { r } { c _ { k , S } ^ { D } , } \end{array}$ and a robust bandwidth $\sigma _ { k , \dot { S } } .$ A frozen semantic encoder embeds the category description. The dimensional center is estimated from the paired annotations, and the bandwidth summarizes withincategory dispersion.

For source category $y ,$ let $g _ { y }$ be its normalized semantic embedding and $\bar { v } _ { y }$ its robust dimensional center. Language and dimensional similarity induce two distributions:

$$
\alpha _ { y , k } ^ { L } = \mathrm { s o f t m a x } _ { k } \left( \frac { \cos ( g _ { y } , c _ { k , S } ^ { L } ) } { \tau _ { L } } \right) ,\tag{3}
$$

$$
\alpha _ { y , k } ^ { D } = \mathrm { s o f t m a x } _ { k } \left( - \frac { \| \bar { v } _ { y } - c _ { k , S } ^ { D } \| ^ { 2 } } { \tau _ { D } \sigma _ { k , S } ^ { 2 } } \right) .\tag{4}
$$

The temperatures $\tau _ { L }$ and $\tau _ { D }$ control how broadly a category overlaps neighboring prototypes. Their bridge target is

$$
\alpha _ { y } ^ { S } = \lambda _ { B } \alpha _ { y } ^ { L } + ( 1 - \lambda _ { B } ) \alpha _ { y } ^ { D } ,\tag{5}
$$

where $\lambda _ { B }$ controls the relative contribution of language and dimensional geometry. Because both annotations refer to the same source examples and share the same prototype slots, the bridge aligns categorical meaning with continuous afect structure instead of treating them as unrelated auxiliary tasks.

The source bank remains immutable during target training. This preserves a stable semantic coordinate system and prevents a small number ofunusual target phrases from rewriting the source geometry. Directly freezing that geometry, however, would assume that source categories and target openvocabulary descriptions organize afect identically. APCF therefore derives a target bank through bounded residual adaptation:

$$
c _ { k , T } ^ { L } = \mathrm { n o r m } \big ( c _ { k , S } ^ { L } + \rho _ { L } \operatorname { t a n h } { \Delta _ { k } ^ { L } } \big ) ,\tag{6}
$$

$$
c _ { k , T } ^ { D } = \mathrm { c l i p } \big ( c _ { k , S } ^ { D } + \rho _ { D } \operatorname { t a n h } { \Delta _ { k } ^ { D } } \big ) .\tag{7}
$$

The radii $\rho _ { L }$ and $\rho _ { D }$ bound how far a target prototype can move, while normalization and clipping keep the two components in their valid spaces. Source bandwidths remain fixed so that adaptation changes prototype location rather than silently changing what counts as a broad or narrow afect region.

Target emotion phrases provide soft supervision for the language residual through their similarity to the adapted anchors. No dimensional target is fabricated when a target corpus does not supply one. Instead, the dimensional residual receives gradients from downstream generation and is constrained by prototype-preservation terms. These terms penalize excessive residual norms, distortion of pairwise source relations, reversal of the source dimensional ordering, and collapse between target prototypes. The result is a targetaware bank that may shift toward domain-specific emotion language while remaining tethered to the source afect structure.

## Source-Domain Sensory Initialization

The prototype bank is also used to initialize the sensory pathway. For each paired source example, the bridge distribution $\alpha _ { y } ^ { S }$ supervises temporary prototype-assignment heads, while a temporary regression head predicts the dimensional annotation. These heads are attached to unimodal evidence and to the fused source representation. Their joint objective can be summarized as

$$
\mathcal { L } _ { \mathrm { s r c } } = \mathcal { L } _ { \mathrm { a s s i g n } } + \lambda _ { \mathrm { d i m } } \mathcal { L } _ { \mathrm { d i m } } + \lambda _ { \mathrm { c r o s s } } \mathcal { L } _ { \mathrm { c r o s s } } ,\tag{8}
$$

where $\mathcal { L } _ { \mathrm { c r o s s } }$ encourages compatible evidence organization across observed modalities. This stage teaches the shared evidence encoder that diferent afect regions may rely on different modality relations. It is therefore more than prototype construction: source supervision initializes the perceptual extraction and fusion parameters as well.

The temporary prediction heads are discarded after source training. The learned evidence parameters are transferred to the target stage, whereas the source anchors remain fixed and the target residuals begin from the source geometry. This separation prevents the source task from imposing a fixed output inventory on open-vocabulary generation.

## Prototype-Conditioned Evidence Extraction

Each adapted prototype is transformed into a route code $c _ { k }$ and a sensory-conditioning code $d _ { k }$

$$
c _ { k } = W _ { r } [ c _ { k , T } ^ { L } ; c _ { k , T } ^ { D } ] , \qquad d _ { k } = W _ { s } [ s _ { k } ; c _ { k } ] ,\tag{9}
$$

where $s _ { k }$ is a learned slot identifier without explicit semantic initialization. The route code carries afect meaning to the decoder, while the sensory code conditions shared perceptual computation. Keeping the slot identifier separate from the semantic code allows the architecture to preserve stable memory positions without equating a prototype with an output class.

For reference position $\bar { \tau } _ { \ell } ,$ the pair (k, ℓ) forms a query that retrieves evidence from each observed modality:

$$
\begin{array} { r } { q _ { k , \ell } = p _ { \ell } + W _ { q } d _ { k } , } \end{array}\tag{10}
$$

$$
e _ { i , k , m , \ell } = \mathrm { A t t n } ( q _ { k , \ell } , H _ { i } ^ { m } ; b _ { m } ( \tau _ { i , m } - \bar { \tau } _ { \ell } ) ) .\tag{11}
$$

Here $p \ell$ identifies a learned reference-time slot and $b _ { m }$ is a modality-specific relative-time bias. Timestamp embeddings are also added to the attention values. Consequently, the same sensory tokens can yield diferent evidence when queried from diferent afect regions or temporal positions. For example, a short vocal hesitation may be highly relevant to one prototype but weak evidence for another.

All prototypes and modalities share the retrieval parameters; only their conditioning codes and modality embeddings difer. This parameter sharing allows a target phrase near a source prototype to reuse computation rather than requiring a newly trained detector. Since retrieval is evaluated only for $m \in S _ { i }$ , the evidence tensor contains no representation derived from an unavailable channel.

## Observed-Only Set Fusion

Retrieved evidence must be combined without assuming a fixed number or order of observed streams. Let $\eta _ { m }$ be a modality identity embedding. APCF first applies prototypeconditioned FiLM (Perez et al. 2018), then uses a shared Set Transformer (Lee et al. 2019):

$$
x _ { i , k , m , \ell } = \mathrm { F i L M } ( \mathrm { L N } ( W _ { e } e + \eta _ { m } + W _ { a } a _ { i } ) ; d _ { k } ) ,
$$

$$
r _ { i , k , \ell } = \mathrm { P o o l } _ { d _ { k } , a _ { i } } \left( \mathrm { S e t A t t n } \{ x _ { i , k , m , \ell } : m \in S _ { i } \} \right)\tag{12}
$$

(13)

The availability encoding $a _ { i }$ tells the shared module which observation state produced the set, while the modality identities distinguish evidence sources. FiLM lets the same sensory content be emphasized diferently under diferent afect prototypes. Set attention then models relations among only the available streams, and prototype-conditioned pooling aggregates their contributions into $r _ { i , k , \ell }$

This operation is permutation invariant with respect to the presentation order of modalities and accepts every nonempty observed subset. Missingness is not represented by a synthetic feature, and the model is not required to reconstruct a plausible but unverifiable hidden stream. Instead, it learns how the contribution of an observed stream changes with both the afect query and the current observation state.

The fused slots form a prototype-by-time memory

$$
z _ { i , k , \ell } = W _ { g } r _ { i , k , \ell } + \eta _ { k } ^ { P } + \eta _ { \ell } ^ { R } ,\tag{14}
$$

where $\eta _ { k } ^ { P }$ and $\eta _ { \ell } ^ { R }$ are semantic-free prototype and referenceslot identifiers. Semantic-free identifiers preserve memory structure; afect meaning enters through the prototype codes that conditioned retrieval and fusion.

![](images/269e91f22f52961116c85b59fe9fda0abc84186499cc5c96b1f8ba9cbc2fa258.jpg)  
Figure 2: Structure of afect-prototype causal routing.

## Afect Prototype Causal Routing

The structure of this module is detailed in Figure 2. At a routed decoder layer, the causal state $h _ { i , n - 1 }$ contains only the prompt and previously generated tokens. Its projection $q _ { i , n }$ produces a semantic prior $\beta$ over prototypes and content scores s over the sample-specific memory:

$$
\beta _ { i , n , k } = \mathrm { s o f t m a x } _ { k } \left( \frac { \cos ( q _ { i , n } , W _ { P } c _ { k } ) } { \tau _ { G } } \right) ,\tag{15}
$$

$$
s _ { i , n , k , \ell } = \frac { q _ { i , n } ^ { \top } W _ { K } z _ { i , k , \ell } } { \sqrt { d _ { h } } } ,\tag{16}
$$

$$
a _ { i , n , k , \ell } = \mathrm { s o f t m a x } _ { k , \ell } \left( s _ { i , n , k , \ell } + \log ( \beta _ { i , n , k } + \epsilon ) \right) .\tag{17}
$$

Here $\tau _ { G }$ is the routing temperature, $d _ { h }$ is the query dimension, and ϵ is a numerical stabilizer. The semantic prior asks which afect regions are compatible with the current generation prefix; the content score asks which prototype-time slots contain relevant evidence for this particular sample. Their sum yields one joint distribution $a _ { i , n , k , \ell } ,$ , so prototype choice and evidence choice cannot drift into independent decisions.

The routed evidence is

$$
v _ { i , n } = \sum _ { k , \ell } a _ { i , n , k , \ell } W _ { V } z _ { i , k , \ell } ,\tag{18}
$$

$$
\widetilde { h } _ { i , n } = h _ { i , n } + g W _ { O } v _ { i , n } ,\tag{19}
$$

where $g$ is a zero-initialized residual gate. The value and output paths follow an identity-preserving factorization: learned transformations reconcile dimensions, while the prototype router controls selection rather than freely rewriting memory content. At initialization, the gate makes the augmented decoder equivalent to the frozen decoder. Training can then introduce evidence gradually without destabilizing its language ability.

Routing is strictly causal. A completed target phrase is never encoded and returned to the router during its own generation. At step n, both the semantic prior and the content route depend only on $Q _ { i } , y _ { i , < n }$ , and observed evidence. This prevents label leakage and permits ordinary autoregressive inference.

## Training Process

Training proceeds from source prototype construction and sensory initialization to target-domain adaptation and candidate-free generation. The modality encoders and pretrained language decoder remain frozen; optimization updates the evidence modules, observed-set fusion, target prototype residuals, and causal router. At inference, targetphrase assignments, temporary source heads, and completeview references are absent. APCF encodes only the observed streams, builds their prototype-conditioned memory, and generates emotion terms autoregressively.

## Experiments

We evaluate released open-vocabulary systems, controlled fusion methods, APCF component variants, and the learned prototype space, for proving our proposed framework can achieve efective emotion modelling and recognition within incomplete modality context.

## Datasets and Evaluation Metrics

Data and views. IEMOCAP (Busso et al. 2008) supplies paired categorical and dimensional source supervision. All 31,327 MER-Caption+ examples (Lian et al. 2025a) are used for target training, and evaluation uses all 532 OV-MERD+ examples (Lian et al. 2025a) and all 1,200 MER-FG examples (Lian et al. 2025b). A fixed non-emotional sentence replaces unavailable native transcripts for 22 and five examples, respectively; these rows are not counted as controlled missing-text cases. Controlled views are instead produced by deleting complete streams. Every sample is evaluated under {A, V, T, AV, AT, VT, AVT} modality combinations.

Metrics. For both datasets, normalized terms are mapped through the same five emotion wheels. $S _ { 1 }$ and $S _ { 2 }$ are the mean F-scores over their coarse and fine mappings, and $\mathrm { A v g } = ( S _ { 1 } + S _ { 2 } ) / 2$ is the primary metric. Avg is the native MER-FG ranking score and for OV-MERD+, its native score $( S _ { 1 }$ F-score) is used. All table entries are percentages; $S _ { 1 }$ and $S _ { 2 }$ remain available in the result artifacts. A frozen parser handles raw generations, and malformed or empty outputs remain failures.

## Experimental Setup

Audio, video, and text are encoded by frozen Chinese-HuBERT-large (Hsu et al. 2021), CLIP ViT-L/14 (Radford et al. 2021), and all-mpnet-base-v2 (Song et al. 2020), respectively. The decoder is Qwen2.5-7B-Instruct (Yang et al. 2024) with the released AfectGPT MER-Caption+ LoRA (Lian et al. 2025a; Hu et al. 2022) merged and frozen. APCF uses eight prototypes and eight temporal slots, constructed from IEMOCAP dataset, a 384-dimensional evidence space, and a router at decoder layer 13. Generation is deterministic with at most 64 new tokens. All experiments are conducted on RTX 3090 GPUs.

Table 1: Open-vocabulary comparison on OV-MERD+ and MER-FG. Cells report (%): level-1 F1 on OV-MERD+ and Avg on MER-FG.
<table><tr><td rowspan="2">Dataset Method</td><td rowspan="2"></td><td colspan="7">Modality view</td></tr><tr><td>A</td><td>V</td><td>T</td><td>AV</td><td>AT</td><td>VT</td><td>AVT</td></tr><tr><td rowspan="5">OV-MERD+</td><td>OV-MER (Lian et al. 2025c)</td><td>33.914</td><td>45.211</td><td>47.085</td><td>44.969</td><td>48.304</td><td>50.163</td><td>46.952</td></tr><tr><td>Emotion-LLaMA (Cheng et al. 2024)</td><td>29.104</td><td>29.464</td><td>38.215</td><td>48.257</td><td>42.108</td><td>44.961</td><td>54.439</td></tr><tr><td>AffectGPT (Lian et al. 2025a)</td><td>46.655</td><td>49.138</td><td>53.775</td><td>54.160</td><td>55.650</td><td>60.518</td><td>62.229</td></tr><tr><td>AffectAgent-R (Wang et al. 2026)</td><td>48.226</td><td>49.941</td><td>53.594</td><td>55.502</td><td>58.173</td><td>61.004</td><td>62.832</td></tr><tr><td>APCF (ours)</td><td>56.490</td><td>50.587</td><td>54.116</td><td>58.818</td><td>61.086</td><td>61.193</td><td>63.015</td></tr><tr><td rowspan="5">MER-FG</td><td>OV-MER (Lian et al. 2025c)</td><td>24.999</td><td>32.040</td><td>37.851</td><td>32.353</td><td>35.361</td><td>37.820</td><td>36.041</td></tr><tr><td>Emotion-LLaMA (Cheng et al. 2024)</td><td>17.533</td><td>20.976</td><td>26.024</td><td>31.410</td><td>28.699</td><td>29.471</td><td>36.889</td></tr><tr><td>AffectGPT (Lian et al. 2025a)</td><td>33.925</td><td>31.132</td><td>34.904</td><td>35.598</td><td>36.379</td><td>39.069</td><td>46.016</td></tr><tr><td>AffectAgent-R (Wang et al. 2026)</td><td>37.108</td><td>33.994</td><td>37.618</td><td>39.061</td><td>39.764</td><td>42.919</td><td>45.729</td></tr><tr><td>APCF (ours)</td><td>43.095</td><td>35.265</td><td>39.450</td><td>43.940</td><td>45.112</td><td>44.685</td><td>46.708</td></tr></table>

Table 2: Comparison with missing-modality methods adapted to open-vocabulary emotion generation. Cells report the datasetnative score (%): level-1 F1 on OV-MERD+ and Avg on MER-FG.
<table><tr><td rowspan="2">Dataset Method</td><td rowspan="2"></td><td colspan="7">Modality view</td></tr><tr><td>A</td><td>V</td><td>T</td><td>AV</td><td>AT</td><td>VT</td><td>AVT</td></tr><tr><td rowspan="6">OV-MERD+</td><td>MulT (Tsai et al. 2019)</td><td>9.491</td><td>10.361</td><td>47.778</td><td>18.591</td><td>47.786</td><td>47.702</td><td>47.791</td></tr><tr><td>MMIN (Zhao, Li, and Jin 2021)</td><td>17.740</td><td>21.856</td><td>48.359</td><td>22.249</td><td>48.230</td><td>48.351</td><td>48.399</td></tr><tr><td>MPLMM (Guo, Jin, and Zhao 2024)</td><td>49.852</td><td>49.981</td><td>48.989</td><td>55.724</td><td>53.447</td><td>54.660</td><td>56.148</td></tr><tr><td>ComP (He, Zhu, and Zhang 2026)</td><td>41.342</td><td>35.055</td><td>50.147</td><td>14.592</td><td>49.797</td><td>49.832</td><td>49.783</td></tr><tr><td>BALM-FCM (Nguyen et al. 2026)</td><td>45.144</td><td>40.286</td><td>46.771</td><td>55.580</td><td>53.719</td><td>48.441</td><td>54.402</td></tr><tr><td>APCF (ours)</td><td>56.490</td><td>50.587</td><td>54.116</td><td>58.818</td><td>61.086</td><td>61.193</td><td>63.015</td></tr><tr><td rowspan="6">MER-FG</td><td>MulT (Tsai et al. 2019)</td><td>7.371</td><td>9.227</td><td>34.280</td><td>13.948</td><td>34.824</td><td>34.887</td><td>34.884</td></tr><tr><td>MMIN (Zhao, Li, and Jin 2021)</td><td>15.561</td><td>20.368</td><td>34.433</td><td>21.295</td><td>35.424</td><td>35.539</td><td>35.457</td></tr><tr><td>MPLMM (Guo, Jin, and Zhao 2024)</td><td>40.306</td><td>34.955</td><td>38.500</td><td>41.482</td><td>42.936</td><td>41.944</td><td>44.173</td></tr><tr><td>ComP (He, Zhu, and Zhang 2026)</td><td>39.781</td><td>32.763</td><td>38.826</td><td>13.223</td><td>38.938</td><td>39.002</td><td>38.812</td></tr><tr><td>BALM-FCM (Nguyen et al. 2026)</td><td>38.816</td><td>34.734</td><td>39.135</td><td>39.921</td><td>43.150</td><td>40.714</td><td>43.828</td></tr><tr><td>APCF (ours)</td><td>43.095</td><td>35.265</td><td>39.450</td><td>43.940</td><td>45.112</td><td>44.685</td><td>46.708</td></tr></table>

Source initialization precedes eight epochs of target training. We employ a hybrid Muon–AdamW optimizer: Muon updates eligible two-dimensional hidden matrices with learning rate $1 0 ^ { - 3 } .$ , momentum 0.95, Nesterov momentum, and five Newton–Schulz iterations, while AdamW (Loshchilov and Hutter 2019) updates auxiliary parameters at $5 \times 1 0 ^ { - 5 }$ and prototype residuals at $1 0 ^ { - 5 }$ . All parameter groups use weight decay 0.01. Learning rates follow a cosine schedule with 500 warm-up steps and decay to 0.1 of their initial values; the global gradient norm is clipped at 1.0. Controlled methods share the prompt, decoder, parser, and scorer; released systems retain their native front ends and prompting.

## Comparison with Open-Vocabulary Systems

OV-MER (Lian et al. 2025c) uses its released acoustic and visual clue checkpoints with a view-aware merger. Emotion-LLaMA (Cheng et al. 2024) is evaluated zero-shot from its released checkpoint, and AfectGPT (Lian et al. 2025a) retains its released multimodal modules without additional adaptation. AfectAgent (Wang et al. 2026) proposed module for incomplete modal fusion and was evaluated on openvocabulary tasks, however, as it provides neither executable code nor a checkpoint, AfectAgent-R denotes our paperguided implementation of its reported agent, retrieval, fusion, and optimization components.

Table 1 shows that APCF ranks first under every modality view on both datasets. It gains 8.264 points for A, 3.316 for AV, and 2.913 for AT on OV-MERD+ dataset, and 5.987, 4.879, and 5.348 points for A, AV, and AT on MER-FG dataset, respectively. Text-rich views are already strong for language-decoder-based systems because the transcript is close to the generated output space, and APCF’s larger gains on A, AV , and AT provide more direct evidence that it can organize the surviving non-textual cues.

## Comparison with Incomplete-Modality Fusion Methods

Under the same target data, modality encoders, language decoder, prompt, evidence budget, and scorer, we adapt MulT (Tsai et al. 2019), MMIN (Zhao, Li, and Jin 2021), MPLMM (Guo, Jin, and Zhao 2024), ComP (He, Zhu, and Zhang 2026), and BALM-FCM (Nguyen et al. 2026) to openvocabulary generation.

Table 3: APCF component ablation. Filled and open circles denote enabled and disabled components. Values are the average over all seven modality combinations (All7, %), using level-1 F1 on OV-MERD+ and Avg on MER-FG. Best results are bold.
<table><tr><td colspan="4">Tested components</td><td colspan="2">A117 (%)</td></tr><tr><td colspan="4">MVL SPT PSF TPA SCR</td><td colspan="2">OV-MERD+ MER-FG</td></tr><tr><td>O</td><td>O</td><td>O</td><td>o</td><td>O</td><td>48.38 36.87</td></tr><tr><td>C</td><td>O</td><td>O</td><td>o</td><td>O 53.87</td><td>40.36</td></tr><tr><td></td><td>●</td><td>O</td><td>o</td><td>O 53.42</td><td>40.93</td></tr><tr><td>C</td><td>●</td><td>●</td><td>O</td><td>O 54.19</td><td>41.05</td></tr><tr><td>●</td><td>●</td><td>●</td><td>●</td><td>O 55.82</td><td>41.08</td></tr><tr><td>O</td><td>●</td><td>●</td><td>●</td><td>●</td><td>47.69 37.36</td></tr><tr><td>●</td><td>0</td><td>●</td><td>●</td><td>55.05 ●</td><td>42.18</td></tr><tr><td></td><td>●</td><td>O</td><td>●</td><td>53.94</td><td>41.96</td></tr><tr><td></td><td>●</td><td>●</td><td>o</td><td>53.89 C</td><td>41.84</td></tr><tr><td></td><td>●</td><td>●</td><td>●</td><td>57.90 ●</td><td>42.61</td></tr></table>

Open-vocabulary adaptation. For MulT, MMIN, MPLMM, and ComP, we remove the native fixed-label prediction head and pass the fused hidden sequenceinro the frozen language decoder. For BALM specifically, we insert its Feature Calibration Module (FCM), which can be transferred without altering the task. Gradient Rebalancing Module is not included because its design depends hevaily on fixed-label classification.

As shown in Table 2, APCF leads on open-vocabulary emotion phrases generation, attributing its advantage more directly to incomplete-evidence organization under a common generator. The shared language decoder makes textcontaining views a necessary but insuficient test of fusion. For MulT, MMIN, and ComP, their text-absent profiles are substantially weaker or unstable, particularly on V and AV. Adding audio or video therefore contributes little once text is present; the apparently competitive text-view results mainly reflect transcript processing by the common language backbone. This also explains why comparing only AVT would substantially overestimate the missing-modality capability of these methods.

## Ablation Studies

Table 3 evaluates missing-view learning (MVL), source prototype transfer (SPT), prototype-conditioned sensory fusion (PSF), target prototype adaptation (TPA), and semantic causal routing (SCR). The upper block adds these components to a common base in order, while the lower block removes one component from full APCF. All variants retain the same backbone, dimensions, data, and evaluation protocol.

MVL produces the largest cumulative improvement, adding 5.49 points on OV-MERD+ and 3.49 on MER-FG; removing it from full APCF reduces performance by 10.21 and 5.25 points, respectively. The prototype components are also beneficial in the full system. Removing SPT, PSF, or TPA costs 2.85/0.43, 3.96/0.65, and 4.01/0.77 points on OV-MERD+/MER-FG, while adding SCR to the otherwise complete model contributes 2.08/1.53 points.

![](images/2be373501d7bc6439c1a5292c850e49eb6ab2d0006b7fc2dd78941d903505bd4.jpg)

(a) Pairwise source and adapted anchor distances.  
![](images/e9b10790b50475d4e14fb07747b7d3da7d70a11984e889a0f6f731159cd17c94.jpg)  
(b) Axis-wise CCC on held-out emotion concepts.  
Figure 3: Afect-prototype adaptation diagnostics averaged over target-training seeds 64, 65, and 71.

## Afect-Prototype Adaptation Analysis

Figure 3 analyzes the two efects of target adaptation that are central to our prototype design. The bounded bank closely preserves the IEMOCAP pairwise anchor geometry (ρ = .952), whereas unconstrained adaptation largely destroys it (ρ = .095). On 1,024 held-out NRC-VAD concepts (Mohammad 2018), bounded adaptation also raises CCC (Lin 1989) from .254/.062/.086 to .305/.083/.111 for valence/arousal/dominance. These results show that bounded adaptation improves target-domain afect alignment while retaining the source structure.

## Conclusion

This paper presents APCF, a candidate-free framework for open-vocabulary emotion recognition when any subset of audio, video, and text may be unavailable. APCF transfers a categorical–dimensional afect geometry from existing datasets, adapts it to target emotion phrases, uses the resulting prototypes to query and fuse only observed evidence, and routes that evidence causally into a frozen language decoder. In this way, emotion semantics influence the sensory fusion process rather than serving only as output labels. Comparisons on OV-MERD+ and MER-FG datasets with released open-vocabulary systems and controlled incompletemodality pipelines show that APCF improves robustness across missing-input views while preserving competitive AVT performance.

## References

Buechel, S.; Modersohn, L.; and Hahn, U. 2021. Towards Label-Agnostic Emotion Embeddings. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, 9231–9249. Association for Computational Linguistics.

Busso, C.; Bulut, M.; Lee, C.-C.; Kazemzadeh, A.; Mower, E.; Kim, S.; Chang, J. N.; Lee, S.; and Narayanan, S. S. 2008. IEMOCAP: Interactive Emotional Dyadic Motion Capture Database. Language Resources and Evaluation, 42(4): 335– 359.

Cheng, Z.; Cheng, Z.-Q.; He, J.-Y.; Sun, J.; Wang, K.; Lin, Y.; Lian, Z.; Peng, X.; and Hauptmann, A. G. 2024. Emotion-LLaMA: Multimodal Emotion Recognition and Reasoning with Instruction Tuning. In Proceedings of the 38th International Conference on Neural Information Processing Systems, volume 37 of NIPS ’24, 110805–110853. Red Hook, NY, USA: Curran Associates Inc. ISBN 979-8-3313-1438-5.

Gaonkar, R.; Kwon, H.; Bastan, M.; Balasubramanian, N.; and Chambers, N. 2020. Modeling Label Semantics for Predicting Emotional Reactions. In Proceedings ofthe 58th Annual Meeting of the Association for Computational Linguistics, 4687–4692. Association for Computational Linguistics.

Guo, Z.; Jin, T.; and Zhao, Z. 2024. Multimodal Prompt Learning with Missing Modalities for Sentiment Analysis and Emotion Recognition. In Ku, L.-W.; Martins, A.; and Srikumar, V., eds., Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 1726–1736. Bangkok, Thailand: Association for Computational Linguistics.

Han, X.; Nguyen, H.; Harris, C.; Ho, N.; and Saria, S. 2024. FuseMoE: Mixture-of-Experts Transformers for Fleximodal Fusion. In Advances in Neural Information Processing Systems, volume 37, 67850–67900. Neural Information Processing Systems Foundation, Inc.

Han, Z.; Zhu, B.; Tong, W.; Shao, P.; Song, P.; Wang, X.; Chen, J.; Lu, L.; and Yang, X. 2026. Omni-Perception Policy Optimization for Multimodal Emotion Reasoning. In Proceedings of the 43rd International Conference on Machine Learning, volume 306 of Proceedings of Machine Learning Research. Seoul, South Korea: PMLR.

He, W.-J.; Zhu, X.; and Zhang, Z. 2026. Cross-Modal Prompting for Balanced Incomplete Multi-modal Emotion Recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 40(21): 17463–17471.

Hsu, W.-N.; Bolte, B.; Tsai, Y.-H. H.; Lakhotia, K.; Salakhutdinov, R.; and Mohamed, A. 2021. HuBERT: Self-Supervised Speech Representation Learning by Masked Prediction of Hidden Units. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 29: 3451–3460.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2022. LoRA: Low-rank Adaptation of Large Language Models. In International Conference on Learning Representations.

Huang, J.; Lin, F.; Yang, X.; Feng, C.; Zhu, K.; Yang, X.; and Chen, Z. 2026. Nano-EmoX: Unifying Multimodal Emotional Intelligence from Perception to Empathy. In Proceed-

ings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22986–22997.

Lai, Z.; Zhu, Z.; Hong, X.; and Wang, Y. 2025. Agent-MER: A Cognitive Agent with Hierarchical Deliberation for Open-Vocabulary Multimodal Emotion Recognition. In Proceedings of the 33rd ACM International Conference on Multimedia, MM ’25, 13864–13871. New York, NY, USA: Association for Computing Machinery. ISBN 979-8-4007- 2035-2.

Lee, J.; Lee, Y.; Kim, J.; Kosiorek, A.; Choi, S.; and Teh, Y. W. 2019. Set Transformer: A Framework for Attention-Based Permutation-Invariant Neural Networks. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings of Machine Learning Research, 3744–3753. PMLR.

Li, S.; Chen, C.; and Han, J. 2025. SimMLM: A Simple Framework for Multi-modal Learning with Missing Modality. In Proceedings of the IEEE/CVF International Conference on Computer Vision, 24068–24077.

Li, W.; Cheng, J.; Tang, X.; and Vong, C. M. 2026. Anchoring the Afective Manifold: Learning Canonical and Disentangled Representations via Generative Cross-Modal Alignment. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), 41605–41614. Association for Computational Linguistics.

Lian, Z.; Chen, H.; Chen, L.; Sun, H.; Sun, L.; Ren, Y.; Cheng, Z.; Liu, B.; Liu, R.; Peng, X.; Yi, J.; and Tao, J. 2025a. AfectGPT: A New Dataset, Model, and Benchmark for Emotion Understanding with Multimodal Large Language Models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, 36993–37014. PMLR.

Lian, Z.; Liu, R.; Xu, K.; Liu, B.; Liu, X.; Zhang, Y.; Liu, X.; Li, Y.; Cheng, Z.; Zuo, H.; Ma, Z.; Peng, X.; Chen, X.; Li, Y.; Cambria, E.; Zhao, G.; Schuller, B. W.; and Tao, J. 2025b. MER 2025: When Afective Computing Meets Large Language Models. In Proceedings of the 33rd ACM International Conference on Multimedia, 13837–13842.

Lian, Z.; Peng, X.; Xu, K.; Jia, Z.; Che, X.; Cheng, Z.; Ma, F.; Cui, L.; Zhang, Y.; Liu, X.; Yang, L.; Li, J.; Zhang, F.; Xue, L.; Cambria, E.; Zhao, G.; Schuller, B. W.; and Tao, J. 2026a. MER 2026: From Discriminative Emotion Recognition to Generative Emotion Understanding. arXiv:2604.19417.

Lian, Z.; Sun, H.; Sun, L.; Chen, H.; Chen, L.; Gu, H.; Wen, Z.; Chen, S.; Zhang, S.; Yao, H.; Liu, B.; Liu, R.; Liang, S.; Li, Y.; Yi, J.; and Tao, J. 2025c. OV-MER: Towards Open-Vocabulary Multimodal Emotion Recognition. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, 37015–37050. PMLR.

Lian, Z.; Zhang, F.; Chen, L.; Zhang, Y.; Liu, R.; Wu, J.; Chen, H.; Li, X.; Peng, X.; He, B.; and Tao, J. 2026b. AfectGPT-RL: Revealing Roles of Reinforcement Learning in Open-Vocabulary Emotion Recognition. arXiv:2605.06126.

Lin, L. I.-K. 1989. A Concordance Correlation Coeficient to Evaluate Reproducibility. Biometrics, 45(1): 255–268.

Liu, Y.; Zhang, L.; Li, H.; Shi, H.; Ding, Y.; Qu, L.; and Li, T. 2026. Follow the Clues, Frame the Truth: Hybrid-evidential Deductive Reasoning in Open-Vocabulary Multimodal Emotion Recognition. arXiv:2603.16463.

Loshchilov, I.; and Hutter, F. 2019. Decoupled Weight Decay Regularization. In International Conference on Learning Representations.

Lyu, Y.; Zheng, X.; Zhou, J.; and Wang, L. 2024. UniBind: LLM-Augmented Unified and Balanced Representation Space to Bind Them All. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 26742–26752. IEEE Computer Society.

Miyoshi, R.; Otani, M.; and Okafuji, Y. 2026. Robust Multimodal Emotion Recognition from Incomplete Modalities via Query-Based Unimodal and Cross-Modal Learning. In 2026 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), 4901–4911.

Mohammad, S. M. 2018. Obtaining Reliable Human Ratings of Valence, Arousal, and Dominance for 20,000 English Words. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), 174–184. Association for Computational Linguistics.

Nguyen, P.-A.; Pham, T. A.; Le, D.-T.; and Nguyen, C.-V. T. 2026. BALM: A Model-Agnostic Framework for Balanced Multimodal Learning under Imbalanced Missing Rates. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, 30246–30256.

Park, S.; Kim, J.; Ye, S.; Jeon, J.; Park, H. Y.; and Oh, A. 2021. Dimensional Emotion Detection from Categorical Emotion. In Proceedings ofthe 2021 Conference on Empirical Methods in Natural Language Processing, 4367–4380. Association for Computational Linguistics.

Perez, E.; Strub, F.; de Vries, H.; Dumoulin, V.; and Courville, A. 2018. FiLM: Visual Reasoning with a General Conditioning Layer. In Proceedings of the AAAI Conference on Artificial Intelligence, volume 32.

Pipoli, V.; Bolelli, F.; Sarto, S.; Cornia, M.; Baraldi, L.; Grana, C.; Cucchiara, R.; and Ficarra, E. 2025. Semantically Conditioned Prompts for Visual Recognition under Missing Modality Scenarios. In Proceedings of the Winter Conference on Applications ofComputer Vision, 4968–4977.

Praveen, R. G.; de Melo, W. C.; Ullah, N.; Aslam, H.; Zeeshan, O.; Denorme, T.; Pedersoli, M.; Koerich, A. L.; Bacon, S.; Cardinal, P.; and Granger, E. 2022. A Joint Cross-Attention Model for Audio-Visual Fusion in Dimensional Emotion Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition Workshops, 2485–2494. IEEE.

Qiu, X.; Fang, Y.; Zhou, Q.; Zhai, B.; Hong, J.; Zhang, W.; Lu, Y.; Zhang, Y.; and Li, C. 2026. Beyond Missing Modalities: Hypergraph Conditioned Difusion for Uncertainty-Aware Multimodal Emotion Recognition. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 22953–22963.

Radford, A.; Kim, J. W.; Hallacy, C.; Ramesh, A.; Goh, G.; Agarwal, S.; Sastry, G.; Askell, A.; Mishkin, P.; Clark, J.; Krueger, G.; and Sutskever, I. 2021. Learning Transferable Visual Models from Natural Language Supervision. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings of Machine Learning Research, 8748–8763. PMLR.

Song, K.; Tan, X.; Qin, T.; Lu, J.; and Liu, T.-Y. 2020. MP-Net: Masked and Permuted Pre-Training for Language Understanding. In Advances in Neural Information Processing Systems, volume 33, 16857–16867.

Tsai, Y.-H. H.; Bai, S.; Liang, P. P.; Kolter, J. Z.; Morency, L.-P.; and Salakhutdinov, R. 2019. Multimodal Transformer for Unaligned Multimodal Language Sequences. In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6558–6569.

Wang, Z.; Yu, Z.; Zhu, Y.; Zhao, B.; Liang, H.; Wang, T.; Xia, W.; Zhang, J.; Liu, Z.; Ma, H.; Ma, F.; and Tian, Q. 2026. AfectAgent: Collaborative Multi-Agent Reasoning for Retrieval-Augmented Multimodal Emotion Recognition. arXiv:2604.12735.

Yang, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Li, C.; Liu, D.; Huang, F.; Wei, H.; Lin, H.; Yang, J.; Tu, J.; Zhang, J.; Yang, J.; Yang, J.; Zhou, J.; Lin, J.; Dang, K.; Lu, K.; Bao, K.; Yang, K.; Yu, L.; Li, M.; Xue, M.; Zhang, P.; Zhu, Q.; Men, R.; Lin, R.; Li, T.; Tang, T.; Xia, T.; Ren, X.; Ren, X.; Fan, Y.; Su, Y.; Zhang, Y.; Wan, Y.; Liu, Y.; Cui, Z.; Zhang, Z.; and Qiu, Z. 2024. Qwen2.5 Technical Report. arXiv:2412.15115.

Yun, S.; Choi, I.; Peng, J.; Wu, Y.; Bao, J.; Zhang, Q.; Xin, J.; Long, Q.; and Chen, T. 2024. Flex-MoE: Modeling Arbitrary Modality Combination via the Flexible Mixture-of-Experts. In Advances in Neural Information Processing Systems, volume 37, 98782–98805. Neural Information Processing Systems Foundation, Inc.

Zhang, Y.; Chen, M.; Shen, J.; and Wang, C. 2022. Tailor Versatile Multi-Modal Learning for Multi-Label Emotion Recognition. Proceedings of the AAAI Conference on Artificial Intelligence, 36(8): 9100–9108.

Zhang, Z.; Chen, J.; Hu, Y.; Zhang, Z.; Yuan, X.; Yang, M.; Zhao, X.; Ngai, E. C. H.; Li, C.; and Hu, X. 2026. Clue2Emo: A Brain-Inspired Framework for Open-Vocabulary Multimodal Emotion Recognition. In ICASSP 2026 - 2026 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 12567–12571.

Zhao, B.; Ye, F.; Ji, Y.; Zhao, S.; Peng, X.; and Yu, Z. 2026. AfectVerse: Emotional World Models for Multimodal Affective Computing. arXiv:2605.19950.

Zhao, J.; Li, R.; and Jin, Q. 2021. Missing Modality Imagination Network for Emotion Recognition with Uncertain Missing Modalities. In Zong, C.; Xia, F.; Li, W.; and Navigli, R., eds., Proceedings ofthe 59th Annual Meeting ofthe Associationfor Computational Linguistics and the 11th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 2608–2618. Online: Association for Computational Linguistics.