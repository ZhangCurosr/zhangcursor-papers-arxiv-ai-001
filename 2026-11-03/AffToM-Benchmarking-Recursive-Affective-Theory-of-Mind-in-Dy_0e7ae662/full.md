# AffToM: Benchmarking Recursive Affective Theory of Mind in Dynamic Multi-Party Interactions

Jinpeng Hu<sup>1</sup>, Yiduo Jin<sup>1</sup>, Erqiang Wang<sup>1</sup>, Jia Li<sup>1</sup>, Jingjing Wu<sup>1</sup>, Peipei Song<sup>2</sup> <sup>1</sup>Hefei University of Technology <sup>2</sup>University of Science and Technology of China

## Abstract

Understanding emotion in realistic social interactions requires more than recognizing facial expressions or assigning affective labels. The same event can elicit different emotions depending on what each participant knows, what they believe others know or feel, and how these nested mental states evolve over time. Existing benchmarks largely study higher-order epistemic Theory of Mind (ToM) and multimodal emotion reasoning in isolation, leaving recursive affective reasoning under asymmetric and dynamically changing in formation underexplored. Therefore, we introduce AffToM, a benchmark for affective ToM with 1,922 test pairs and 19,759 training pairs. AffToM evaluates Individual Affective Reasoning (IAR), which infers emo tions from events and socio-cognitive con text, and Multi-party Affective Reasoning (MPAR), which recursively tracks participants nested beliefs and emotions as new evidence emerges. We develop an event-driven multi agent generation pipeline grounded in dy namic character interaction graphs that explic itly capture event sequences, interpersonal relations, and mental-state updates. Furthermore, we propose AffToM-R1, a two-stage posttraining framework combining supervised finetuning with perspective-aware rubric-guided reinforcement learning. Experiments show that MPAR remains challenging, with even the strongest evaluated MLLM achieving less than 60% accuracy, while AffToM-R1 improves MPAR accuracy by 13.38 percentage points over its base model. Our dataset is available at: https://github.com/MindIntLab-HFUT/AffToM.

## 1 Introduction

Recent Multimodal Large Language Models (MLLMs) have demonstrated strong capabilities in visually grounded affective perception, including facial expression recognition, emotion description, and action understanding (Lian et al., 2025; Cheng et al., 2024; Li et al., 2025a). However, recognizing an apparent emotion is fundamentally different from understanding it within the person’s broader cognitive and social context, as the same expression, such as a smile, may signal happiness, relief, embarrassment, concealment, or an attempt to reassure others. Genuine emotional understanding requires reasoning beyond perceptual appearance to determine what an individual knows, expects, and believes, how they interpret the situation, and how they understand the mental states of others.

This capability is commonly referred to as affective Theory ofMind (ToM), namely the ability to infer emotional states from the latent cognitive representations that shape them. In realistic social interactions, emotions are not direct and universal responses to observable events but emerge from how individuals appraise those events under their beliefs, goals, expectations, and social relationships (Premack and Woodruff, 1978; Lillard, 1993; Qu et al., 2015). Consequently, the same event may evoke different emotions in different participants, while perceptually similar reactions may reflect fundamentally different mental states. The challenge becomes greater in multi-party interactions, where an individual’s emotion may depend not only on their own beliefs but also on what they believe another participant knows, intends, or feels about someone else. For instance, a person may feel relieved because they believe a misunderstanding remains unnoticed, embarrassed because they infer that another participant has recognized their concealed intention, or guilty because they realize that one participant now understands that a third participant feels betrayed. In fact, to evaluate machine ToM, a growing body of benchmarks has been proposed (Chen et al., 2024; Wu et al., 2023; Xu et al., 2024; Jin et al., 2024; Shi et al., 2025).

Despite this progress, existing benchmarks leave two closely related challenges unresolved. First, existing higher-order ToM and affective reasoning capabilities are largely studied in isolation. Higherorder benchmarks focus on recursively nested epistemic states, whereas affective benchmarks typically predict each participant’s emotion independently, failing to capture cross-participant affective dependencies in which one person’s emotion is shaped by what they believe others know, believe, or feel. Second, current benchmarks provide only limited control over information asymmetry, recursion depth, and temporal belief updates, making it difficult to distinguish genuine higher-order affective reasoning from reliance on superficial perception. These limitations motivate the study of Multi-party Affective Reasoning (MPAR), which evaluates affective states whose correct interpretation depends on recursively nested mental representations distributed across multiple participants. Within an ongoing social interaction, an MPAR question asks how another person believes the target character feels before the character’s emotional response becomes observable. As illustrated in Fig ure 1, the husband’s affective inference changes as his information state evolves. After the wife observes a fabricated medical report, he infers that she mistakenly believes he is infertile and interprets her affective state as feeling wronged. When he later observes her discovering the authentic report, he updates his representation of her belief and instead infers disbelief and guilt. Although the participants and broader situation remain largely unchanged, the correct affective inference changes because the husband’s representation of what the wife knows and believes has evolved. Therefore, we introduce AffToM, a video-based benchmark for affective ToM that evaluates latent emotion inference from characters’ evolving cognitive representations. It contains 1,922 QA pairs from 258 videos for benchmark evaluation, complemented by 19,759 QA pairs from 2,291 videos for model training. AffToM evaluates two complementary capabilities: Individual Affective Reasoning (IAR), which requires models to infer an individual’s affective state by integrating observable events with situational context, and MPAR, which further requires models to recursively track how participants represent one another’s beliefs and emotions and use these nested representations to infer affective states. We develop an event-driven multi-agent questiongeneration pipeline grounded in dynamic character interaction graphs that explicitly represent event chronology, interpersonal relations, and mentalstate transitions. These structured graphs enable controlled generation of questions with different participant configurations, information asymmetries, temporal update depths, and levels of mentalstate recursion. Role-specialized agents generate IAR and MPAR questions from the graphs, after which the resulting samples are refined through an iterative dual-criteria evolution process that jointly improves logical validity and distractor discriminability, followed by human validation. These findings motivate us to examine whether targeted training can improve belief-conditioned emotion inference from others’ perspectives. We introduce AffToM-R1, a two-stage post-training framework that applies supervised fine-tuning (SFT) followed by perspective-aware rubric-guided reinforcement learning. Comprehensive evaluations of state-ofthe-art MLLMs under both multiple-choice and open-ended settings show that MPAR remains particularly challenging, with even the strongest evaluated model achieving less than 60% accuracy. Meanwhile, AffToM-R1 improves MPAR accuracy by 13 percentage points over its base model.

![](images/5f5c470e70f1a171465a49ed96bcbb83268ac733c082712cc096a8e403959aaf.jpg)  
Figure 1: An AffToM example showing multi-party affective reasoning and emotion shifts under information asymmetry.

Table 1: Comparison with representative social video and Theory-of-Mind benchmarks. Temporal Belief Update denotes explicit belief changes after new information, while Recursive Perspective denotes second-order or higher perspective-recursive QA.
<table><tr><td rowspan="2">Benchmark</td><td rowspan="2">Real-World</td><td colspan="4">Mental State</td><td rowspan="2">Temporal Belief Update</td><td rowspan="2">Recursive Perspective</td><td rowspan="2">Dataset Format</td></tr><tr><td>Emotion</td><td>Belief</td><td>Intention</td><td>Desire</td></tr><tr><td>Video-MME (Fu et al., 2025)</td><td></td><td>x</td><td>X</td><td>X</td><td>x</td><td>X</td><td>x</td><td>MCQ</td></tr><tr><td>Hi-ToM (Wu et al., 2023)</td><td></td><td>X</td><td></td><td></td><td></td><td>X</td><td></td><td>MCQ</td></tr><tr><td>Social-IQ (Zadeh et al., 2019)</td><td></td><td></td><td></td><td></td><td></td><td>+</td><td>x</td><td>MCQ</td></tr><tr><td>MMToM-QA (Jin et al., 2024)</td><td>X</td><td></td><td></td><td></td><td></td><td>+</td><td>X</td><td>MCQ</td></tr><tr><td>MOMENTS (Villa-Cueva et al., 2025)</td><td></td><td></td><td></td><td></td><td></td><td>X</td><td>x</td><td>MCQ</td></tr><tr><td>R³-Bench (Niu et al., 2026)</td><td></td><td></td><td></td><td></td><td></td><td>x</td><td>x</td><td>MCQ</td></tr><tr><td>AffToM</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>MCQ + Open-ended</td></tr></table>

## 2 Related Work

Theory of Mind. Theory of Mind (ToM) refers to the capacity to infer others’ mental states, including beliefs, intentions, and emotions (Premack and Woodruff, 1978; Decety and Jackson, 2004). Although ToM is frequently operationalized through belief and intention reasoning, psychological theories increasingly recognize emotion as an important component of mental-state representation. Since attributing emotion depends critically on tracking others’ mental states (Qu et al., 2015), emotion understanding should be regarded as a fundamental aspect of ToM rather than a peripheral addon. Early research on ToM has primarily been conducted in conversational or game-based settings (Liu et al., 2024). Text-based benchmarks evaluate belief tracking, false-belief reasoning, and recursive perspective-taking (Le et al., 2019; Kim et al., 2023; Wu et al., 2023; Xu et al., 2024; Chen et al., 2024), while recent multimodal benchmarks extend ToM to visually grounded and interactive settings. MMToM-QA and MuMA-ToM study beliefs and goals in multi-agent environments (Jin et al., 2024; Shi et al., 2025), whereas MOMENTS, Social-IQ 2.0, and IntentQA evaluate broader social reasoning abilities from real-world videos (Villa-Cueva et al., 2025; Wilf et al., 2023; Li et al., 2023). More recent work further explores mental-physical causal reasoning and multimodal emotion reasoning (Niu et al., 2026; Luo et al., 2026). Despite this progress, emotion inference through nested mental states and its evolution re-

mains underexplored.

Multimodal Question Answering. LLMs have achieved remarkable progress in question answering across a wide range of domains (Yang et al., 2025; Team et al., 2026; Hu et al., 2024; Dai et al., 2026a; Xu et al., 2025; Dou et al., 2026). With the integration of visual inputs, multimodal QA has further extended language-based reasoning to visually grounded scenarios. Early works in multimodal QA focus on image question answering. Subsequently, the field moves from recognition toward deeper understanding and commonsense modeling. Benchmarks such as PerceptionTest (Patr˘ aucean˘ et al., 2023) and HourVideo (Chandrasegaran et al., 2024) begin to emphasize temporal integration, long-video context, and limited forms of causal understanding, requiring models to go beyond perception and answer “what happened?” and “why did it happen?” On this basis, multimodal QA has increasingly expanded into the affective domain, where models must jointly interpret facial expressions, prosody, language, and broader social context. Early benchmarks mainly focus on sentiment or emotion recognition, emphasizing polarity, intensity, or categorical emotion predic tion, as exemplified by MOSI (Zadeh et al., 2016), CMU-MOSEI (Bagher Zadeh et al., 2018), MT-MEUR (Hu et al., 2025), CEDAR (Dai et al., 2026b), MELD (Poria et al., 2019), and MMH-Bench (Hu et al., 2026). More recent datasets have moved beyond closed-set labeling toward more explainable and reasoning-oriented affect understanding. EMER (Lian et al., 2025) and DEEMO (Li et al., 2025b) introduce explanatory rationales, OV-MER (Lian et al., 2024) relaxes fixed label spaces through open-vocabulary emotion prediction, and MME-Emotion (Zhang et al., 2025) further evalu ates both emotional understanding and causal rea soning in a unified and interpretable QA-style setting. Table 1 compares AffToM with representa tive social video and ToM benchmarks. AffToM brings together real-world video grounding, temporal belief updates, and recursive perspective-taking to evaluate how emotions depend on participants evolving beliefs about one another.

![](images/0ec08a7f53dc35d5791e0dca667a67f563d7a9946f994b7eb545820ffc3ab0ee.jpg)  
Figure 2: Overview of the dataset construction pipeline, including event-driven video collection, question generation, data evolution, and manual evaluation.

## 3 AffToM

In this section, we detail the systematic construction of the AffToM benchmark. We introduce a multi-stage data generation pipeline driven by a collaborative multi-agent framework, with the overall structure shown in Figure 2. This systematic approach sequentially consists of event-driven video curation, dual-paradigm question generation enhanced by a social interaction graph, dual-criteriaguided data evolution, and comprehensive postprocessing alongside expert validation.

## 3.1 Event-Driven Video Collection

We collect 1,504 raw video candidates from public video-sharing platforms, including Douyin, Tik-Tok, and YouTube. Collection details are provided in Appendix A.1. These videos span diverse social scenarios, thereby providing rich multimodal signals for modeling complex human behaviors. To systematically parse the semantic content of these candidate videos, we employ Seed-1.8 (Seed, 2026a) to extract fine-grained event metadata. For video $v ,$ the event extraction is defined as:

$$
\begin{array} { r } { \mathcal { E } = \mathrm { A g e n t } _ { \mathrm { e v e } } ( v ) , } \end{array}\tag{1}
$$

where $\mathcal { E } ~ = ~ \{ e _ { 1 } , e _ { 2 } , . . . , e _ { N } \}$ denotes the set of events extracted from the video. Each event $e _ { i }$ consists of its spatiotemporal grounding—comprising the timestamp, location, and involved individuals. To capture multi-party social dynamics for affective ToM reasoning, we retain videos with multiple characters connected by explicit directed interaction pairs. This filtering yields 308 videos, for which we extract character-level cognitive and affective attributes:

$$
\begin{array} { r } { \boldsymbol { S } = \mathrm { A g e n t } _ { \mathrm { s t a t e } } ( \boldsymbol { v } , \mathcal { E } ) . } \end{array}\tag{2}
$$

Here, $\boldsymbol { \cal S } = \{ s _ { 1 } , s _ { 2 } , \ldots , s _ { M } \}$ denotes the set of individual-level socio-cognitive states, encompassing identity traits, interpersonal relationships, and dynamically evolving mental states of the characters throughout the video, where $s _ { i }$ corresponds to the i-th character $c _ { i } .$ . Using events E and individuallevel information S, we construct a dynamic character interaction graph $G = \{ G _ { 1 } , G _ { 2 } , . . . , G _ { N } \}$ where nodes represent the characters and edges encode their evolving behavioral relationships.

## 3.2 Dual-Paradigm Question Generation

After constructing the dynamic character interaction graph G, we design two complementary question generation paths to construct affective ToM reasoning tasks.

Recursive Perspective Expansion. The first paradigm generates Multi-party Affective Reasoning (MPAR) questions through recursive perspective expansion. For each extracted event, we establish a base-level affective state or belief for a target individual, grounded in the event context:

$$
{ \mathcal Q } ^ { ( 1 ) } = \Phi _ { \mathrm { i n i t } } ( \mathcal E , v ) ,\tag{3}
$$

${ \mathcal Q } ^ { ( 1 ) } ~ = ~ \{ q _ { 1 } ^ { 1 } , q _ { 2 } ^ { 1 } , . . . , q _ { N } ^ { 1 } \}$ serves as the foundational belief questions from which more complex reasoning structures are derived. To construct MPAR questions, we progressively introduce additional individuals from the interaction graph G and recursively integrate their respective perspectives into the reasoning process. Second-order questions introduce a second individual and require reasoning about one person’s belief concerning another’s mental or emotional state. Third and higher orders add further parties and their perspectives. Rather than merely nesting beliefs, the questions capture how attitudes, expectations, and misunderstandings shape emotions within the shared interaction, evaluating dynamic, interaction-driven affective reasoning beyond isolated emotion recognition. We formally define this recursive generation of higherorder cognitive states as:

$$
( \mathcal { Q } ^ { ( i + 1 ) } , \mathcal { A } ^ { ( i + 1 ) } ) = \Phi _ { \mathrm { r e c u r } } ( \mathcal { Q } ^ { ( i ) } , G _ { i } , v ) ,\tag{4}
$$

where $\mathcal { Q } ^ { ( i ) }$ denotes the i-th order question and $\mathcal { A } ^ { ( i + 1 ) }$ denotes the corresponding answer. $\Phi _ { \mathrm { r e c u r } }$ is the recursive integration function that expands the current reasoning structure by incorporating the perspective of a newly introduced character $c _ { k } .$ using the corresponding event and character information encoded in $G _ { i }$ , together with the video v.

To quantify the behavioral effect of the interaction, we formulate some higher-order queries as temporal pairs spanning across the event timeline. Specifically, we generate coupled questions querying the nested affective states at both a preevent timestamp $t _ { \mathrm { p r e } }$ and a post-event timestamp $t _ { \mathrm { p o s t } }$ . This contrasting design forces models to explicitly trace the temporal evolution of emotional states and measure the causal impact of the specific event on resolving or exacerbating interpersonal misunderstandings. Upon deriving the definitive question-answer pairs $( \mathcal { Q } , \mathcal { A } )$ , we synthesize a set of semantically plausible distractors to construct the final multiple-choice questions (MCQs). Furthermore, to complement this standard discriminative evaluation, we introduce open-ended MPAR questions specifically designed to assess deeply nested higher-order affective reasoning. An eventdriven MPAR generation example is provided in Appendix E.2.

Individual Affective Reasoning Generation. Complementing multi-party relational reasoning, the second paradigm is designed to formulate individual affective reasoning (IAR) questions that probe the internal states and latent characteristics of individual characters within emotionally rich social contexts. Specifically, this track covers six dimensions of affective and behavioral understanding, including Preference Prediction, Integrated Mental-State Reasoning, Strategic Perspective-Taking, Hidden Intention, Personality Reasoning, and Counterfactual Reasoning. For this type, we generate the question through a question generation agent with video v and the graph G as inputs, guided by a curated set of few-shot exemplars derived from human-validated seed questions.

## 3.3 Dual-Criteria Iterative Refinement

To improve cognitive depth and linguistic quality, our iterative pipeline combines role-guided reflection with dual-criteria evaluation. The Role Reflection Agent $( \mathrm { A g e n t _ { R o R } } )$ examines each generated question from the event’s different roles, identifying improvements, ambiguities, and alternative viewpoints grounded in the interaction structure. Using this feedback, the Revision Agent $( \mathrm { { A g e n t } _ { r e v } ) }$ refines the initial drafts into a diverse candidate pool $\mathcal { Q } _ { \mathrm { c a n d } }$ . We then select high-quality questions from $\mathcal { Q } _ { \mathrm { c a n d } }$ using two criteria: Comparative Rationality and Distractor Unbiasedness.

Criterion 1: Comparative Rationality $( S _ { \mathbf { r a n k } } ) .$ Most existing data generation pipelines adopt scoring-based selection strategies to identify highquality candidates from a pool of generated samples (Li et al., 2025c; Agarwal et al., 2025). However, relying on standard absolute scoring or uncalibrated list-wise ranking often introduces severe vulnerabilities, such as verbosity, self-enhancement, and position biases (Zheng et al., 2023). To rigorously isolate the highest-quality questions and circumvent these biases, we employ a robust pairwise comparison mechanism benchmarked against human expertise. Specifically, we prompt an evaluation agent $( \mathrm { i . e . , A g e n t _ { e v a l } } )$ to conduct blind pairwise evaluations between a generated candidate $q _ { c } \in \mathcal { Q } _ { \mathrm { c a n d } }$ and a sampled human-authored reference question $q _ { h } \in \mathcal { Q } _ { \mathrm { h u m a n } }$ from the seed pool. $\mathrm { \sf A g e n t _ { e v a l } }$ judges the candidate superior, equivalent, or inferior to the human reference based on logical coherence, contextual alignment, and affective depth. The comparative rationality score is the expected win rate over the human references:

$$
\begin{array} { r l r } {  { S _ { \mathrm { r a n k } } ( q _ { c } ) = \frac { 1 } { | \mathcal { Q } _ { \mathrm { h u m a n } } | } \sum _ { q _ { h } \in \mathcal { Q } _ { \mathrm { h u m a n } } } } } \\ & { } & { \Big ( \mathbb { I } ( q _ { c } \succ q _ { h } ) + \frac { 1 } { 2 } \mathbb { I } ( q _ { c } \sim q _ { h } ) \Big ) . ~ } \end{array}\tag{5}
$$

where I(·) is the indicator function, $q _ { c } \succ q _ { h }$ denotes that the candidate outperforms the human reference, and $q _ { c } \sim \ q _ { h }$ denotes a tie in quality. This pairwise formulation, bounded between 0 and 1, mitigates list-wise position bias and ensures that a high score reflects a quality level authentically comparable to, or explicitly exceeding, expert human annotations.

Criterion 2: Distractor Unbiasedness $\left( S _ { \mathbf { b i a s } } \right)$ . In multimodal benchmarks, models often exploit linguistic priors to guess the correct answer without processing the visual input. To ensure our distractors (i.e., incorrect options) are plausible and effectively penalize blind guessing, we evaluate the candidate options $\mathcal { O } = \{ A , B , C , D \}$ under a videoblind setting. Given only the textual context, we estimate the prior probability distribution over the options using an LLM. The LLM produces output logits for each candidate option, which are converted into a probability distribution $P ( o \mid$ text-only) via a softmax function. An unbiased question should compel the text-only model into a state of maximum uncertainty, yielding a uniform distribution across all options. We formally quantify this uncertainty using Normalized Shannon Entropy:

$$
S _ { \mathrm { b i a s } } ( q _ { c } ) = - \frac { 1 } { \log | \mathcal { O } | } \sum _ { o \in \mathcal { O } } p _ { o } \log p _ { o } ,\tag{6}
$$

where $p _ { o } = P ( o \mid$ text-only) denotes the probability assigned to option o by the LLM given only the textual context, without access to the video. If the model is highly confident in a single option without seeing the video, $S _ { \mathrm { b i a s } }$ yields a low score, indicating that the distractors are ineffective or the question is textually biased.

Iterative Evolution and Termination. We combine the two criteria into a unified score for iterative candidate refinement:

$$
S _ { \mathrm { t o t a l } } ( q _ { c } ) = \lambda S _ { \mathrm { r a n k } } ( q _ { c } ) + ( 1 - \lambda ) S _ { \mathrm { b i a s } } ( q _ { c } ) ,\tag{7}
$$

where λ controls the trade-off between question quality and distractor robustness. If a generated candidate meets a predefined acceptance threshold τ , it is accepted as a high-quality question. Otherwise, the generation process enters the next evolution cycle. This iterative reflection–revision–ranking process continues until a candidate question satisfies the criterion. To prevent unbounded iterations, we impose a predefined maximum number of refinement rounds $T _ { \mathrm { m a x } }$ . Let $s _ { t } = S _ { \mathrm { t o t a l } } ( q _ { c } ^ { ( t ) } )$ denote the candidate’s score at iteration t. The selection and routing process is defined as:

$$
\begin{array} { r } { q ^ { * } = \left\{ \begin{array} { l l } { q _ { c } ^ { ( t ) } } & { s _ { t } \geq \tau , } \\ { \Phi _ { \mathrm { e v o l v e } } \big ( q _ { c } ^ { ( t ) } \big ) } & { s _ { t } < \tau , t < T _ { \operatorname* { m a x } } , } \\ { \emptyset } & { t \geq T _ { \operatorname* { m a x } } . } \end{array} \right. } \end{array}\tag{8}
$$

Here, $\Phi _ { \mathrm { e v o l v e } }$ performs role-guided reflection and revision. The reflection agent $\mathrm { \bf A g e n t _ { \mathrm { R o R } } }$ uses the interaction graph $G _ { i }$ to assess the candidate $q _ { c } ^ { ( t ) }$ The revision agent $\mathrm { \bf A g e n t } _ { \mathrm { r e v } }$ then revises the candidate using this feedback. The threshold τ controls acceptance, while $T _ { \mathrm { m a x } }$ limits the number of refinement cycles.

## 3.4 Post-Processing and Human Validation

We additionally apply the same generation pipeline to a separate pool of 2,291 training videos, yielding 19,759 QA samples. The test set undergoes two additional steps: linguistic post-processing and human validation. First, to reduce linguistic shortcuts, we apply linguistic neutralization to the questions and options retained after iterative refinement. We remove explicit emotional polarity and soften overly deterministic phrasing, for example by rewriting “react negatively” as “show a particular emotional reaction”. This encourages models to reason about multimodal context rather than rely on textual cues. We manually validate the questions against the source videos (see validation details in Appendix B.1). Through this multi-stage filtering and validation process, we obtain a benchmark containing 1922 high-quality questions grounded in 258 videos. Detailed statistics are provided in Appendix A.1.

## 4 AffToM-R1

In this section, we present AffToM-R1. As illustrated in Figure 3, SFT trains the model on structured reasoning trajectories, while Perspective-Aware Faithfulness Reasoning Optimization (PAFRO) further optimizes it with perspective-aware rubric-based rewards. We construct a 20k-sample training pool and split it into $\mathcal { D } _ { \mathrm { S F T } }$ for SFT and $\mathcal { D } _ { \mathrm { R I } }$ for PAFRO based on difficulty estimated from the pretrained backbone. Further training details are provided in Appendix C.1.

![](images/809e552f1955650a1595f3439e4d23892d2bbd2189d3f3d8629ef49b40fb268d.jpg)  
Figure 3: Overview of AffToM-R1.

## 4.1 Supervised Fine-Tuning

We first perform supervised fine-tuning (SFT) on perspective-structured reasoning trajectories that connect video evidence to the target answer. Each target response consists of a reasoning block enclosed by <think></think> and a final answer enclosed by <answer></answer> (see Appendix C.2 for more details). The reasoning block begins with a <perspective></perspective> field that explicitly identifies the viewpoint from which the question is resolved to guide later reasoning.

Perspective-structured reasoning. We adopt two reasoning structures depending on whether the question requires character-specific mental-state inference. For the character-centered structure, the trajectory follows Clues → Mental State → Inference, explicitly interpreting the observed evidence through the target character’s information and mental state before deriving the answer. For the context-centered structure, the trajectory follows Clues → Inference when explicit character-level mental-state modeling is unnecessary. All MPAR questions adopt the character-centered structure, while the remaining questions are assigned an appropriate structure with LLM assistance. We generate candidate trajectories accordingly and retain only those whose final answers match the reference answers, yielding $\mathcal { D } _ { \mathrm { S F T } }$ for SFT.

## 4.2 Perspective-Aware Faithfulness Reasoning Optimization

PAFRO further optimizes the SFT-trained model using rubric-based GRPO, as illustrated in Figure 3. Details of GRPO are provided in Appendix C.3.

Reward design. Answer correctness alone does not reveal whether the model adopts the appropriate viewpoint or supports its conclusion with video evidence. We therefore combine rule-based checks of answer correctness $( R _ { \mathrm { a c c } } )$ , response format $( R _ { \mathrm { f m t } } )$ , and perspective alignment $( R _ { \mathrm { p e r s p } } )$ with two reasoning rubrics. Mental-state faithfulness $( R _ { \mathrm { f a i t h } } )$ checks the response’s claims about beliefs, intentions, and emotions by comparing them with the dynamic character interaction graph G provided as reference, using the relevant event or whole-video scope. For character-centered responses, this rubric assesses claims from the declared character’s viewpoint. Context-centered responses are assessed without a single-character viewpoint. Evidence-answer consistency $( R _ { \mathrm { c o n s } } )$ checks whether sampled frames and subtitles support the reasoning and whether that reasoning supports the selected answer. It does not use the reference answer, allowing the reward to distinguish evidential support from answer correctness. The response-level reward is:

$$
\begin{array} { r l } & { r _ { i } = \lambda _ { \mathrm { a c c } } R _ { \mathrm { a c c } } + \lambda _ { \mathrm { f m t } } R _ { \mathrm { f m t } } + \lambda _ { \mathrm { p e r s p } } R _ { \mathrm { p e r s p } } } \\ & { ~ + \lambda _ { \mathrm { f a i t h } } R _ { \mathrm { f a i t h } } + \lambda _ { \mathrm { c o n s } } R _ { \mathrm { c o n s } } , } \end{array}\tag{9}
$$

where each component is evaluated for response $y _ { i }$ and lies in [0, 1]. The λ coefficients set their relative weights. Appendix C.4 provides the scoring rules for all five components.

## 5 Experiments

## 5.1 Experimental Settings

Evaluated Models. We evaluate open-source MLLMs across model families and scales, including the Qwen2.5-VL series (Bai et al., 2025b), the Qwen3-VL series (Bai et al., 2025a), the Qwen3.5 series (Team, 2026), the Gemma-3 series, the InternVL3.5 series (Wang et al., 2025), MiniCPM-V-2.6 (Yao et al., 2024), MiniCPM-V-4.5 (Yu et al., 2025), models from the Ministral-3 series, LLaVA-OneVision (Li et al., 2024), and LLaVA-OneVision-

Table 2: Performance comparison of different MLLMs. Human performance is evaluated on 250 questions randomly sampled from AffToM (see Appendix B.2 for more details).
<table><tr><td rowspan="2">Models</td><td rowspan="2">Params</td><td colspan="8">Multiple-Choice Question</td><td rowspan="2" colspan="3">Open-ended</td></tr><tr><td>PP</td><td>IMSR</td><td>SPT</td><td>HI</td><td>PR</td><td>CR</td><td>MPAR Avg.</td><td>EC</td><td>FC Avg.</td></tr><tr><td colspan="10">Open-Source MLLMs</td><td></td><td></td><td></td></tr><tr><td>Ministral-3-3B</td><td>3B</td><td>31.02</td><td>39.81</td><td>43.15</td><td>27.49</td><td>38.89</td><td>33.33</td><td>30.07</td><td>33.26</td><td>1.64</td><td>1.48</td><td>1.56</td></tr><tr><td>Gemma-3-4B</td><td>4B</td><td>40.11</td><td>52.43</td><td>43.15</td><td>39.77</td><td>42.78</td><td>37.85</td><td>29.67</td><td>37.43</td><td>1.62</td><td>1.26</td><td>1.44</td></tr><tr><td>LLaVA-OneVision-1.5</td><td>4B</td><td>47.59</td><td>63.11</td><td>60.96</td><td>47.95</td><td>45.00</td><td>48.59</td><td>39.34</td><td>46.87</td><td>1.82</td><td>1.51</td><td>1.67</td></tr><tr><td>Qwen3-VL-4B-Instruct</td><td>4B</td><td>50.00</td><td>63.41</td><td>51.37</td><td>53.80</td><td>44.44</td><td>46.89</td><td>36.42</td><td>45.50</td><td>1.88</td><td>1.57</td><td>1.73</td></tr><tr><td>Qwen3.5-4B</td><td>4B</td><td>52.13</td><td>61.46</td><td>66.44</td><td>55.56</td><td>54.44</td><td>58.19</td><td>37.22</td><td>49.29</td><td>1.77</td><td>1.50</td><td>1.64</td></tr><tr><td>Ministral-3-8B</td><td>8B</td><td>44.92</td><td>50.00</td><td>48.63</td><td>45.03</td><td>48.89</td><td>40.68</td><td>32.05</td><td>40.45</td><td>1.79</td><td>1.60</td><td>1.70</td></tr><tr><td>LLaVA-OneVision-Qwen2</td><td>7B</td><td>40.64</td><td>50.00</td><td>52.74</td><td>47.95</td><td>41.11</td><td>44.63</td><td>33.91</td><td>41.00</td><td>1.85</td><td>1.48</td><td>1.67</td></tr><tr><td>InternVL3.5-8B</td><td>8B</td><td>50.27</td><td>52.91</td><td>58.22</td><td>50.29</td><td>53.33</td><td>54.24</td><td>32.32</td><td>44.46</td><td>1.88</td><td>1.60</td><td>1.74</td></tr><tr><td>MiniCPM-V-2.6</td><td>7B</td><td>44.39</td><td>61.65</td><td>61.64</td><td>49.71</td><td>47.22</td><td>53.67</td><td>36.03</td><td>45.94</td><td>1.76</td><td>1.58</td><td>1.67</td></tr><tr><td>Qwen2.5-VL-7B-Instruct</td><td>7B</td><td>52.41</td><td>58.74</td><td>58.22</td><td>50.29</td><td>52.22</td><td>53.67</td><td>36.03</td><td>46.71</td><td>1.85</td><td>1.58</td><td>1.72</td></tr><tr><td>LLaVA-OneVision-1.5</td><td>8B</td><td>44.39</td><td>67.48</td><td>63.01</td><td>52.63</td><td>56.67</td><td>57.63</td><td>33.77</td><td>47.37</td><td>1.96</td><td>1.61</td><td>1.79</td></tr><tr><td>MiniCPM-V-4.5</td><td>8B</td><td>47.59</td><td>62.62</td><td>67.12</td><td>60.23</td><td>56.11</td><td>55.93</td><td>39.60</td><td>50.38</td><td>1.49</td><td>1.31</td><td>1.40</td></tr><tr><td>Qwen3-VL-8B-Instruct</td><td>8B</td><td>59.36</td><td>66.99</td><td>58.90</td><td>61.99</td><td>59.44</td><td>54.80</td><td>39.87</td><td>51.92</td><td>1.92</td><td>1.70</td><td>1.81</td></tr><tr><td>Qwen3.5-9B</td><td>9B</td><td>57.22</td><td>67.48</td><td>68.49</td><td>61.99</td><td>57.22</td><td>64.97</td><td>40.40</td><td>53.51</td><td>1.79</td><td>1.52</td><td>1.66</td></tr><tr><td>Gemma-3-27B</td><td>27B</td><td>50.27</td><td>58.74</td><td>45.21</td><td>55.56</td><td>53.89</td><td>45.20</td><td>30.07</td><td>42.81</td><td>1.87</td><td>1.45</td><td>1.66</td></tr><tr><td>InternVL3.5-14B</td><td>14B</td><td>49.20</td><td>54.37</td><td>63.70</td><td>49.71</td><td>48.89</td><td>51.41</td><td>34.04</td><td>44.90</td><td>1.82</td><td>1.62</td><td>1.72</td></tr><tr><td>InternVL3.5-38B</td><td>38B</td><td>52.94</td><td>58.25</td><td>64.38</td><td>46.78</td><td>53.33</td><td>59.32</td><td>35.36</td><td>47.26</td><td>1.99</td><td>1.76</td><td>1.88</td></tr><tr><td>Qwen2.5-VL-72B-Instruct</td><td>72B</td><td>54.55</td><td>72.33</td><td>65.75</td><td>61.40</td><td>61.67</td><td>66.10</td><td>44.24</td><td>55.65</td><td>2.22</td><td>1.95</td><td>2.09</td></tr><tr><td>Qwen3-VL-32B-Instruct</td><td>32B</td><td>61.50</td><td>77.67</td><td>66.44</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3.5-27B</td><td>27B</td><td>67.91</td><td>75.73</td><td></td><td>63.16</td><td>64.44</td><td>64.41</td><td>40.79</td><td>55.87</td><td>2.39</td><td>1.96</td><td>2.18</td></tr><tr><td></td><td></td><td></td><td></td><td>72.60</td><td>69.01</td><td>63.89</td><td>66.10</td><td>45.03</td><td>59.22</td><td>2.45</td><td>2.23</td><td>2.34</td></tr><tr><td colspan="10">Closed-Source MLLMs</td><td></td><td></td><td></td><td></td></tr><tr><td>Seed-2.0-lite GPT-5.5</td><td>一</td><td>65.78 71.28</td><td>72.82 76.59</td><td>71.23 68.49</td><td>70.76 71.93</td><td>62.78 66.67</td><td>66.10 74.01</td><td>44.24 52.05</td><td>58.29 63.56</td><td>2.02 3.06</td><td>1.88 2.99</td><td>1.95 3.03</td></tr><tr><td colspan="10"></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td>Ours</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AffToM-R1 (Qwen3.5-4B) Human Baseline</td><td>4B</td><td>54.79 91.30</td><td>73.17 92.59</td><td>72.60 90.48</td><td>63.16 95.00</td><td>63.33 92.00</td><td>66.67 91.67</td><td>50.60 84.55</td><td>59.33 88.80</td><td>2.17</td><td>1.89</td><td>2.03</td></tr></table>

1.5 (An et al., 2025). We also evaluate closedsource models, including GPT-5.5 and Seed-2.0- lite (Seed, 2026b). The evaluated configurations are listed in Table 2. For the training study, we compare Qwen3.5-4B, its SFT variant, and AffToM-R1 on AffToM and external benchmarks covering social reasoning and higher-order belief reasoning.

Evaluation Metrics. For MCQs, we utilize standard accuracy as the primary quantitative metric. For open-ended queries, we further employ GPT-5-mini to score the generated responses on a standard 1-to-5 scale across Emotion Correctness (EC) and Factual Consistency (FC), where 1 indicates poor quality and 5 indicates high quality. EC evaluates whether the predicted affective state accurately matches the ground-truth emotion experienced by the target character. FC measures the extent to which the generated response remains faithful to the ground-truth context. Agreement between GPT-5-mini and human scores is assessed

## in Appendix B.3.

Inference Settings. We set the decoding temperature to 0 during evaluation and use 64 uniformly sampled video frames per question unless otherwise specified. Open-source MLLMs are deployed and evaluated on NVIDIA RTX 5090 GPUs, while closed-source models are accessed through their vendors’ commercial APIs.

## 5.2 Main Results

We evaluate a diverse and representative set of MLLMs on the proposed benchmark. The overall performance metrics are summarized in Table 2, from which we derive several key and informative insights into current model capabilities. First, we can observe that MPAR is consistently the most challenging MCQ category. Even GPT-5.5, the strongest evaluated model, achieves only 52.05% accuracy on MPAR, substantially lower than its performance on the other question types. This gap highlights the difficulty of recursively tracking and integrating the affective states of multiple participants, exposing the limitations of current MLLMs in higher-order affective ToM reasoning. Second, larger models generally achieve stronger performance within the same family. This trend is observed in both MCQ and open-ended evaluations, suggesting that increased capacity is associated with better multimodal and socio-cognitive reasoning. The Qwen3.5 series is particularly competitive, with Qwen3.5-27B achieving the highest average scores among open-source MLLMs. Third, large-scale MLLMs yield further improvements across both evaluation formats. GPT-5.5 achieves the best overall performance on both MCQs and open-ended questions, demonstrating stronger capabilities in answer selection and free-form affective reasoning.

![](images/b348f3623b1ad53e92b7418db5138b3c4ae9a41958f52c718684ed1dfe1e13ff.jpg)  
Figure 4: Accuracy across frame counts.

![](images/98d542bbc7506503ab44ffb0c92f60aff11ca2a8e23d2a4089cd8b93feee9f2c.jpg)  
Figure 5: Accuracy across video durations.

## 5.3 Effect of Frame Density and Video Duration

Impact of Frame Density. We evaluate the impact of frame density by varying the number of uniformly sampled video frames from 0 to 64, where 0 frames corresponds to question-only evaluation. The results are shown in Figure 4. First, transitioning from 0 to 8 frames yields a sharp performance increase across all models, showing that even sparse visual cues are essential for reliably inferring narrative trajectories. This upward trend generally continues up to 64 frames, further confirming the importance and effectiveness of denser spatiotemporal information. Second, model capacity strictly dictates the ability to process these dense inputs. While larger models consistently improve with more frames, smaller models plateau and exhibit slight fluctuations after 16 frames. This saturation indicates a representational bottleneck in smaller networks, whereas larger models possess the capacity required to effectively integrate and reason over extended temporal sequences.

Table 3: Effect of subtitle assistance on AffToM.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Modality</td><td colspan="3">MCQ</td><td colspan="3">Open-ended</td></tr><tr><td>IAR</td><td>MPAR</td><td>Avg.</td><td>EC</td><td>FC</td><td>Avg.</td></tr><tr><td>Qwen3.5-4B</td><td>V V+T</td><td>57.83 63.07</td><td>37.22 40.40</td><td>49.29 53.68</td><td>1.77 1.78</td><td>1.50 1.57</td><td>1.64 1.68</td></tr><tr><td>AffToM-R1</td><td>V V+T</td><td>65.51 66.35</td><td>50.60 52.72</td><td>59.33 60.70</td><td>2.17 2.17</td><td>1.89 1.96</td><td>2.03 2.07</td></tr><tr><td>Qwen3.5-27B</td><td>V V+T</td><td>69.26 73.85</td><td>45.03 51.79</td><td>59.22 64.71</td><td>2.45 2.69</td><td>2.23 2.66</td><td>2.34 2.68</td></tr><tr><td>GPT-5.5</td><td>V V+T</td><td>71.70 76.10</td><td>52.05 56.95</td><td>63.56 68.17</td><td>3.06 3.51</td><td>2.99 3.46</td><td>3.03 3.49</td></tr></table>

Impact of Video Duration. We also investigate the effect of video duration, with results shown in Figure 5. Longer videos are generally more challenging, especially for GPT-5.5 and Qwen3-VL-32B, whose accuracy declines as the duration bucket increases. However, AffToM-R1 and Qwen3.5- 27B exhibit non-monotonic changes across duration buckets, with partial recovery in the longestvideo group. Among the evaluated models, GPT-5.5 achieves the best overall performance.

## 5.4 Effect of Subtitle Assistance

We augment video inputs with subtitles generated by Whisper large-v2 (Radford et al., 2022). Table 3 compares video-only inputs (V) with video-plussubtitle inputs (V+T) across the evaluated models. Subtitle assistance improves MCQ accuracy for every reported model, with the largest gains generally appearing on MPAR. Its effects on open-ended EC and FC scores are mixed across models, with improvements observed for some larger baselines. For the Qwen3.5-4B base model and AffToM-R1, adding subtitles improves MCQ accuracy by 4.39 and 1.37 percentage points, respectively. These results indicate that dialogue information provides complementary evidence for interpreting characters’ beliefs and emotions.

Table 4: Accuracy (%) across five benchmarks. Green values indicate gains over the base model in percentage points.
<table><tr><td>Dataset</td><td>Qwen3.5-4B</td><td>+SFT</td><td>AffToM-R1</td></tr><tr><td>AffToM</td><td>49.29</td><td>56.20 (+6.91)</td><td>59.33 (+10.04)</td></tr><tr><td>Social-IQ 2.0</td><td>65.13</td><td>67.37 (+2.24)</td><td>69.99 (+4.86)</td></tr><tr><td>Hi-ToM</td><td>43.75</td><td>49.33 (+5.58)</td><td>50.25 (+6.50)</td></tr><tr><td> $\mathrm { R ^ { 3 } - B e n c h - D X }$ </td><td>70.99</td><td>75.55 (+4.56)</td><td>75.74(+4.75)</td></tr><tr><td> $\mathtt { R } ^ { 3 }$  -Bench-Hard</td><td>32.98</td><td>37.94 (+4.96)</td><td> $3 9 . 7 2 \ : ( + 6 . 7 4 )$ </td></tr></table>

## 5.5 Analysis

Performance across Benchmarks. We evaluate the Qwen3.5-4B base model, its SFT variant, and AffToM-R1 on AffToM, Social-IQ 2.0 (Wilf et al., 2023), Hi-ToM (Wu et al., 2023), and the DX and Hard subsets of $\mathbb { R } ^ { 3 } .$ -Bench (Niu et al., 2026). These evaluations cover video-based social reasoning and text-based higher-order belief reasoning. Table 4 reports higher accuracy for both trained variants across all five evaluation sets, providing initial evidence that the benefits extend beyond AffToM. These gains suggest that learned reasoning transfers beyond the training distribution.

Ablation Study. We examine the contribution of these three reasoning rewards in PAFRO by enabling $R _ { \mathrm { p e r s p } } , R _ { \mathrm { f a i t h } }$ , and $R _ { \mathrm { c o n s } }$ while keeping the correctness and format rewards fixed. As shown in Table 5, $R _ { \mathrm { p e r s p } }$ provides the structural signal for reasoning from the required character viewpoint. Its effect is most visible on MPAR, where perspective control separates a character’s mental state from the video’s full information. Adding either $R _ { \mathrm { f a i t h } }$ or $R _ { \mathrm { c o n s } }$ further improves the two-reward configuration, with faithfulness more closely supporting recursive mental-state tracking and consistency providing a stronger signal for direct affective inference. Faithfulness grounds mental-state claims in the reference graph, while consistency links evidence, reasoning, and the answer. The full reward set achieves the best results across IAR, MPAR, and Overall, reaching 65.51%, 50.60%, and 59.33%, respectively. These results support jointly enforcing perspective control, mental-state grounding, and evidence-answer consistency in PAFRO.

Effect of Information Revelation. Information asymmetry plays a critical role in shaping emotional dynamics. We conduct an analysis comparing model performance on pre-event and postevent questions, with results shown in Figure 6. Pre-event questions require models to simulate a restricted cognitive state and reason under incomplete information, where characters hold divergent beliefs or unresolved misunderstandings. In contrast, post-event questions evaluate affective states after the interaction unfolds and the epistemic uncertainty is largely resolved. We observe that models consistently achieve better performance on postevent questions. The higher proportion of Pre-N/Post-Y than Pre-Y/Post-N cases suggests that models are more likely to reason from an omniscient perspective, using later revealed information rather than preserving the character’s pre-event knowledge state. These findings highlight a limitation of current models in handling information asymmetry, which is essential for robust affective ToM understanding.

Table 5: Ablation of PAFRO reward components on AffToM.
<table><tr><td colspan="3">Reward Components</td><td colspan="3">Accuracy (%)</td></tr><tr><td> $R _ { \mathrm { p e r s p } }$ </td><td> $R _ { \mathrm { f a i t h } }$ </td><td> $R _ { \mathrm { c o n s } }$ </td><td>IAR</td><td>MPAR</td><td>Overall</td></tr><tr><td>一</td><td>一</td><td>一</td><td>64.88</td><td>47.35</td><td>57.61</td></tr><tr><td>√</td><td></td><td>一</td><td>64.92</td><td>48.08</td><td>57.94</td></tr><tr><td>√</td><td>√</td><td>一</td><td>65.04</td><td>49.80</td><td>58.73</td></tr><tr><td>√</td><td></td><td>√</td><td>65.32</td><td>49.01</td><td>58.56</td></tr><tr><td>√</td><td>√</td><td>√</td><td>65.51</td><td>50.60</td><td>59.33</td></tr></table>

![](images/e88847a27352a6816b230afdbee37458c5d71cae5b3389ad1f3ab4c62247eafe.jpg)  
Figure 6: Joint correctness on paired MPAR questions. Rows show pre-event results and columns show postevent results. Y denotes a correct answer, and N an incorrect one.

Error Analysis. To further characterize model failures, we analyze the generated reasoning outputs of Qwen3.5-27B and Qwen3-VL-32B on their respective incorrectly answered MPAR MCQs and classify the resulting failures into four categories: Temporal Information Leakage, Recursive Perspective Conflation, Unwarranted Affective Deduction, and Others, as summarized in Figure 7. For Pre-Event questions, Temporal Information Leakage is the most prominent error, indicating that models tend to rely on information revealed at later event stages when reasoning under asymmetric information. For Post-Event questions, Unwarranted Affective Deduction becomes the dominant error type. This shift demonstrates that while models can observe the physical event, they struggle to accurately map these visual observations to updated psychological states, frequently hallucinating unsupported affective dependencies or over-interpreting sparse cues. For Non-pair questions, Recursive Perspective Conflation emerges as the most severe bottleneck. Absent explicit temporal anchoring, models persistently conflate objective multimodal facts with the target character’s subjective, nested beliefs. Notably, despite differences in overall model capacity, Qwen3.5-27B and Qwen3-VL-32B exhibit highly consistent error distributions across all three temporal conditions. Overall, these findings indicate that higher-order affective reasoning is mainly limited by temporal grounding under asymmetric information, faithful recursive perspective tracking, and psychologically plausible inference. Representative failure cases are discussed in Appendix D.

![](images/a2da03e6f12e83209593dc01732f8a4d3ac4037437ae7cef9c388b5c8d647310.jpg)  
Figure 7: MPAR MCQ error distribution (%). Non-Pair denotes questions outside the pre/post-event pairs.

## 6 Conclusion

In this paper, we propose AffToM, a pioneering multimodal benchmark specifically designed to evaluate affective ToM capabilities in dynamic multi-party social interactions. Moving beyond traditional epistemic reasoning and isolated emotion recognition, our dataset challenges models to decipher latent affective states within dynamic, multi-party social interactions. We construct the benchmark through an event-driven multi-agent generation pipeline grounded in dynamic character interaction graphs, followed by dual-criteria evolution and human validation to ensure data quality. To further investigate whether MPAR can be improved through targeted post-training, we introduce AffToM-R1, which combines SFT with perspective-aware rubric-guided optimization. Results across diverse MLLM families show that

MPAR remains challenging, while AffToM-R1 delivers a substantial 13.38-percentage-point MPAR gain over its Qwen3.5-4B base model and consistent gains on external ToM and social-reasoning benchmarks.

## References

Eshaan Agarwal, Raghav Magazine, Joykirat Singh, Vivek Dani, Tanuja Ganu, and Akshay Nambi. 2025. PromptWizard: Optimizing prompts via task-aware, feedback-driven self-evolution. In Findings ofthe Associationfor Computational Linguistics: ACL 2025, pages 19974–20003, Vienna, Austria. Association for Computational Linguistics.

Xiang An, Yin Xie, Kaicheng Yang, Wenkang Zhang, Xiuwei Zhao, Zheng Cheng, Yirui Wang, Songcen Xu, Changrui Chen, Didi Zhu, Chunsheng Wu, Huajie Tan, Chunyuan Li, Jing Yang, Jie Yu, Xiyao Wang, Bin Qin, Yumeng Wang, Zizhen Yan, and 4 others. 2025. Llava-onevision-1.5: Fully open framework for democratized multimodal training. Preprint, arXiv:2509.23661.

AmirAli Bagher Zadeh, Paul Pu Liang, Soujanya Poria, Erik Cambria, and Louis-Philippe Morency. 2018. Multimodal language analysis in the wild: CMU-MOSEI dataset and interpretable dynamic fusion graph. In Proceedings of the 56th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 2236–2246, Melbourne, Australia. Association for Computational Linguistics.

Shuai Bai, Yuxuan Cai, Ruizhe Chen, Keqin Chen, Xionghui Chen, Zesen Cheng, Lianghao Deng, Wei Ding, Chang Gao, Chunjiang Ge, Wenbin Ge, Zhifang Guo, Qidong Huang, Jie Huang, Fei Huang, Binyuan Hui, Shutong Jiang, Zhaohai Li, Mingsheng Li, and 45 others. 2025a. Qwen3-vl technical report. Preprint, arXiv:2511.21631.

Shuai Bai, Keqin Chen, Xuejing Liu, Jialin Wang, Wenbin Ge, Sibo Song, Kai Dang, Peng Wang, Shijie Wang, Jun Tang, Humen Zhong, Yuanzhi Zhu, Mingkun Yang, Zhaohai Li, Jianqiang Wan, Pengfei Wang, Wei Ding, Zheren Fu, Yiheng Xu, and 8 others. 2025b. Qwen2.5-vl technical report. Preprint, arXiv:2502.13923.

Keshigeyan Chandrasegaran, Agrim Gupta, Lea M. Hadzic, Taran Kota, Jimming He, Cristóbal Eyzaguirre, Zane Durante, Manling Li, Jiajun Wu, and Li Fei-Fei. 2024. Hourvideo: 1-hour video-language understanding. Preprint, arXiv:2411.04998.

Zhuang Chen, Jincenzi Wu, Jinfeng Zhou, Bosi Wen, Guanqun Bi, Gongyao Jiang, Yaru Cao, Mengting Hu, Yunghwei Lai, Zexuan Xiong, and Minlie Huang. 2024. ToMBench: Benchmarking theory of mind in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages

15959–15983, Bangkok, Thailand. Association for Computational Linguistics.

Zebang Cheng, Zhi-Qi Cheng, Jun-Yan He, Jingdong Sun, Kai Wang, Yuxiang Lin, Zheng Lian, Xiaojiang Peng, and Alexander G Hauptmann. 2024. Emotionllama: Multimodal emotion recognition and reasoning with instruction tuning. Advances in Neural Information Processing Systems, 37:110805–110853.

Chongyuan Dai, Jinpeng Hu, Hongchang Shi, Zhuo Li, Dan Guo, Xun Yang, and Meng Wang. 2026a. Psyche-r1: Towards reliable psychological llms through unified empathy, expertise, and reasoning. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 24889–24906.

Chongyuan Dai, Yaling Shen, Zihan Gao, Jia Li, Yishun Jiang, Yaxiong Wang, Liu Liu, Zongyuan Ge, and Jinpeng Hu. 2026b. Tears or cheers? benchmarking llms via culturally elicited distinct affective responses. In Proceedings of the 64th Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 38171–38196.

Jean Decety and Philip L Jackson. 2004. The functional architecture of human empathy. Behavioral and cognitive neuroscience reviews, 3(2):71–100.

Chengfeng Dou, Fan Yang, Fei Li, Jiyuan Jia, Qiang Ju, Shuai Wang, Tianpeng Li, Xiangrong Zeng, Yijie Zhou, Hongda Zhang, et al. 2026. Baichuanm3: Modeling clinical inquiry for reliable medical decision-making. arXiv preprint arXiv:2602.06570.

Chaoyou Fu, Yuhan Dai, Yongdong Luo, Lei Li, Shuhuai Ren, Renrui Zhang, Zihan Wang, Chenyu Zhou, Yunhang Shen, Mengdan Zhang, Peixian Chen, Yanwei Li, Shaohui Lin, Sirui Zhao, Ke Li, Tong Xu, Xiawu Zheng, Enhong Chen, Caifeng Shan, and 2 others. 2025. Video-mme: The first-ever comprehensive evaluation benchmark of multi-modal llms in video analysis. Preprint, arXiv:2405.21075.

Jinpeng Hu, Tengteng Dong, Luo Gang, Hui Ma, Peng Zou, Xiao Sun, Dan Guo, Xun Yang, and Meng Wang. 2024. Psycollm: Enhancing llm for psychological understanding and evaluation. IEEE Transactions on Computational Social Systems.

Jinpeng Hu, Hongchang Shi, Chongyuan Dai, Zhuo Li, Peipei Song, and Meng Wang. 2025. Beyond emotion recognition: A multi-turn multimodal emotion understanding and reasoning benchmark. In Proceedings of the 33rd ACM International Conference on Multimedia, pages 5814–5823.

Jinpeng Hu, Erqiang Wang, Shan Wang, Zhuo Li, Peipei Song, Xun Yang, and Meng Wang. 2026. Mmhbench: A multi-perspective benchmark for mental health understanding in long-form videos. arXiv preprint arXiv:2607.27895.

Chuanyang Jin, Yutong Wu, Jing Cao, Jiannan Xiang, Yen-Ling Kuo, Zhiting Hu, Tomer Ullman, Antonio

Torralba, Joshua Tenenbaum, and Tianmin Shu. 2024. Mmtom-qa: Multimodal theory of mind question answering. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16077–16102.

Hyunwoo Kim, Melanie Sclar, Xuhui Zhou, Ronan Bras, Gunhee Kim, Yejin Choi, and Maarten Sap. 2023. FANToM: A benchmark for stress-testing machine theory of mind in interactions. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 14397–14413, Singapore. Association for Computational Linguistics.

Matthew Le, Y-Lan Boureau, and Maximilian Nickel. 2019. Revisiting the evaluation of theory of mind through question answering. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Processing (EMNLP-IJCNLP), pages 5872–5877.

Ao Li, Longwei Xu, Chen Ling, Jinghui Zhang, and Pengwei Wang. 2025a. Emoverse: Enhancing multimodal large language models for affective computing via multitask learning. Neurocomputing, 650:130810.

Bo Li, Yuanhan Zhang, Dong Guo, Renrui Zhang, Feng Li, Hao Zhang, Kaichen Zhang, Peiyuan Zhang, Yanwei Li, Ziwei Liu, and Chunyuan Li. 2024. Llava-onevision: Easy visual task transfer. Preprint, arXiv:2408.03326.

Deng Li, Bohao Xing, Xin Liu, Baiqiang Xia, Bihan Wen, and Heikki Kälviäinen. 2025b. Deemo: Deidentity multimodal emotion recognition and reasoning. In Proceedings ofthe 33rd ACM International Conference on Multimedia, pages 5707–5716.

Jiapeng Li, Ping Wei, Wenjuan Han, and Lifeng Fan. 2023. Intentqa: Context-aware video intent reasoning. In 2023 IEEE/CVF International Conference on Computer Vision (ICCV), pages 11929–11940.

Zhuo Li, Yuhao Du, Xiaoqi Jiao, Steven Y Guo, Yuege Feng, Xiang Wan, Anningzhe Gao, and Jinpeng Hu. 2025c. Add-one-in: Incremental sample selection for large language models via a choice-based greedy paradigm. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 5321–5340.

Zheng Lian, Haoyu Chen, Lan Chen, Haiyang Sun, Licai Sun, Yong Ren, Zebang Cheng, Bin Liu, Rui Liu, Xiaojiang Peng, et al. 2025. Affectgpt: A new dataset, model, and benchmark for emotion understanding with multimodal large language models. In International Conference on Machine Learning, pages 36993–37014. PMLR.

Zheng Lian, Haiyang Sun, Licai Sun, Haoyu Chen, Lan Chen, Hao Gu, Zhuofan Wen, Shun Chen, Siyuan Zhang, Hailiang Yao, et al. 2024. Ov-mer: Towards open-vocabulary multimodal emotion recognition. arXiv preprint arXiv:2410.01495.

Angeline S Lillard. 1993. Pretend play skills and the child’s theory of mind. Child development, 64(2):348–371.

Ziyi Liu, Abhishek Anand, Pei Zhou, Jen-tse Huang, and Jieyu Zhao. 2024. InterIntent: Investigating social intelligence of LLMs via intention understanding in an interactive game context. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 6718–6746, Miami, Florida, USA. Association for Computational Linguistics.

Meng Luo, Bobo Li, Shanqing Xu, Shize Zhang, Qiuchan Chen, Menglu Han, Wenhao Chen, Yanxiang Huang, Hao Fei, Mong-Li Lee, and Wynne Hsu. 2026. Unveiling the cognitive compass: Theory-of-mind-guided multimodal emotion reasoning. Preprint, arXiv:2602.00971.

Lixing Niu, Jiapeng Li, Xingping Yu, Xinyi Dong, Shu Wang, Ruining Feng, Bo Wu, Ping Wei, Yisen Wang, and Lifeng Fan. 2026. Read the room: Video social reasoning with mental-physical causal chains. In The Fourteenth International Conference on Learning Representations.

Soujanya Poria, Devamanyu Hazarika, Navonil Majumder, Gautam Naik, Erik Cambria, and Rada Mihalcea. 2019. MELD: A multimodal multi-party dataset for emotion recognition in conversations. In Proceedings of the 57th Annual Meeting of the Associationfor Computational Linguistics, pages 527– 536, Florence, Italy. Association for Computational Linguistics.

David Premack and Guy Woodruff. 1978. Does the chimpanzee have a theory of mind? Behavioral and brain sciences, 1(4):515–526.

Viorica Patr˘ aucean, Lucas Smaira, Ankush Gupta,˘ Adrià Recasens Continente, Larisa Markeeva, Dylan Banarse, Skanda Koppula, Joseph Heyward, Mateusz Malinowski, Yi Yang, Carl Doersch, Tatiana Matejovicova, Yury Sulsky, Antoine Miech, Alex Frechette, Hanna Klimczak, Raphael Koster, Junlin Zhang, Stephanie Winkler, and 5 others. 2023. Perception test: A diagnostic benchmark for multimodal video models. Preprint, arXiv:2305.13786.

Li Qu, Pinxiu Shen, Yu Yan Chee, and Luxi Chen. 2015. Teachers’ theory-of-mind coaching and children’s executive function predict the training effect of sociodramatic play on children’s theory of mind. Social Development, 24(4):716–733.

Alec Radford, Jong Wook Kim, Tao Xu, Greg Brockman, Christine McLeavey, and Ilya Sutskever. 2022. Robust speech recognition via large-scale weak supervision. Preprint, arXiv:2212.04356.

ByteDance Seed. 2026a. Seed1.8 model card: Towards generalized real-world agency. Preprint, arXiv:2603.20633.

ByteDance Seed. 2026b. Seed2.0 model card: Towards intelligence frontier for real-world complexity. Preprint, arXiv:2607.00248.

Haojun Shi, Suyu Ye, Xinyu Fang, Chuanyang Jin, Leyla Isik, Yen-Ling Kuo, and Tianmin Shu. 2025. Muma-tom: Multi-modal multi-agent theory of mind. Preprint, arXiv:2408.12574.

Kimi Team, Tongtong Bai, Yifan Bai, Yiping Bao, Jianfeng Cai, Xinyuan Cai, Peizhou Cao, Yuxuan Cao, Ziwei Chai, Y Charles, et al. 2026. Kimi k3: Open frontier intelligence. arXiv preprint arXiv:2607.24653.

Qwen Team. 2026. Qwen3.5: Accelerating productivity with native multimodal agents.

Emilio Villa-Cueva, S M Masrur Ahmed, Rendi Chevi, Jan Christian Blaise Cruz, Kareem Elzeky, Fermin Cristobal, Alham Fikri Aji, Skyler Wang, Rada Mihalcea, and Thamar Solorio. 2025. MoMentS: A comprehensive multimodal benchmark for theory of mind. In Findings of the Association for Computational Linguistics: EMNLP 2025, pages 22591– 22611, Suzhou, China. Association for Computational Linguistics.

Weiyun Wang, Zhangwei Gao, Lixin Gu, Hengjun Pu, Long Cui, Xingguang Wei, Zhaoyang Liu, Linglin Jing, Shenglong Ye, Jie Shao, et al. 2025. Internvl3.5: Advancing open-source multimodal models in versatility, reasoning, and efficiency. Preprint, arXiv:2508.18265.

Alex Wilf, Leena Mathur, Sheryl Mathew, Claire Ko, Youssouf Kebe, Paul Pu Liang, and Louis-Philippe Morency. 2023. Social-iq 2.0 challenge: Benchmarking multimodal social understanding. https://github.com/abwilf/Social-IQ-2. 0-Challenge.

Yufan Wu, Yinghui He, Yilin Jia, Rada Mihalcea, Yulong Chen, and Naihao Deng. 2023. Hi-ToM: A benchmark for evaluating higher-order theory of mind reasoning in large language models. In Findings of the Association for Computational Linguistics: EMNLP 2023, pages 10691–10706, Singapore. Association for Computational Linguistics.

Hainiu Xu, Runcong Zhao, Lixing Zhu, Jinhua Du, and Yulan He. 2024. OpenToM: A comprehensive benchmark for evaluating theory-of-mind reasoning capabilities of large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 8593–8623, Bangkok, Thailand. Association for Computational Linguistics.

Yangyang Xu, Jinpeng Hu, Zhuoer Zhao, Zhangling Duan, Xiao Sun, and Xun Yang. 2025. Multiagentesc: A llm-based multi-agent collaboration framework for emotional support conversation. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 4665–4681.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Yuan Yao, Tianyu Yu, Ao Zhang, Chongyi Wang, Junbo Cui, Hongji Zhu, Tianchi Cai, Haoyu Li, Weilin Zhao, Zhihui He, et al. 2024. Minicpm-v: A gpt-4v level mllm on your phone. arXiv preprint arXiv:2408.01800.

Tianyu Yu, Zefan Wang, Chongyi Wang, Fuwei Huang, Wenshuo Ma, Zhihui He, Tianchi Cai, Weize Chen, Yuxiang Huang, Yuanqian Zhao, Bokai Xu, Junbo Cui, Yingjing Xu, Liqing Ruan, Luoyuan Zhang, Hanyu Liu, Jingkun Tang, Hongyuan Liu, Qining Guo, and 15 others. 2025. Minicpm-v 4.5: Cooking efficient mllms via architecture, data, and training recipe. Preprint, arXiv:2509.18154.

Amir Zadeh, Michael Chan, Paul Pu Liang, Edmund Tong, and Louis-Philippe Morency. 2019. Social-iq: A question answering benchmark for artificial social intelligence. In Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, pages 8807–8817.

Amir Zadeh, Rowan Zellers, Eli Pincus, and Louis-Philippe Morency. 2016. Mosi: Multimodal corpus of sentiment intensity and subjectivity analysis in online opinion videos. Preprint, arXiv:1606.06259.

Fan Zhang, Zebang Cheng, Chong Deng, Haoxuan Li, Zheng Lian, Qian Chen, Huadai Liu, Wen Wang, Yi-Fan Zhang, Renrui Zhang, et al. 2025. Mme-emotion: A holistic evaluation benchmark for emotional intelligence in multimodal large language models. arXiv preprint arXiv:2508.09210.

Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. 2023. Judging llm-as-a-judge with mt-bench and chatbot arena. Preprint, arXiv:2306.05685.

## A Dataset Details

## A.1 Dataset Statistics

Video Sources and Duration. AffToM contains 258 videos collected from public video-sharing platforms, including 186 videos from Douyin/Tik-Tok and 72 videos from YouTube. The dataset has a total duration of approximately 26.4 hours, with an average video length of 6.1 minutes. The collected videos primarily consist of social narratives, including short dramas and movie/TV drama clips, which naturally contain rich interpersonal interactions and evolving affective dynamics. The dataset includes videos in both Chinese and English, reflecting a range of social interaction contexts.

![](images/16790a00b406af08efa863df9f8fe24b3c13e63b31253876bfbee0b3dfd09f67.jpg)  
Figure 8: Scenario distribution of AffToM.

Scenario Diversity. As illustrated in Figure 8, AffToM covers a wide range of social scenarios rather than concentrating on a single interaction pattern. The collected videos span diverse interpersonal contexts, including family relationships, status/resource conflicts, hierarchical and political interactions, romantic relationships, peer friendships, and everyday social interactions. Such diversity provides rich social dynamics for evaluating affective Theory of Mind under realistic multi-party interactions.

Filtering Statistics. During dataset construction, the multi-agent pipeline initially produced 4,174 question-answer pairs from 308 filtered videos. After automatic filtering for text-only shortcuts and option-level biases, 2,340 candidates were retained for human validation. At this stage, all QA candidates associated with 50 videos were filtered out. Consequently, these videos were excluded, reducing the dataset from 308 to 258 videos. Human annotators subsequently spent approximately 150 hours reviewing and validating the retained candidates.

Question Taxonomy and Distribution. After filtering and human validation, AffToM contains 1,922 question-answer pairs, including 1,822 multiple-choice questions and 100 open-ended questions. The questions are organized into two broad categories: Individual Affective Reasoning (IAR) and Multi-party Affective Reasoning (MPAR). IAR questions are presented in a multiplechoice format and cover six task paradigms: Preference Prediction, Integrated Mental-State Reasoning, Strategic Perspective-Taking, Hidden Intention, Personality Reasoning, and Counterfactual Reasoning. MPAR includes both multiple-choice and open-ended questions that evaluate recursive perspective tracking and belief-conditioned affective inference. Figure 9 presents the overall distribution of question types, together with their distribution across different video-duration ranges.

Table 6: Inter-annotator agreement measured by ICC(3,k) across evaluation dimensions. ANSR, VDEP, DQLT, and QVAL denote answer reasonableness, video dependency, distractor quality, and question validity, respectively.
<table><tr><td>Metric</td><td>ANSR</td><td>VDEP</td><td>DQLT</td><td>QVAL</td><td>AVG.</td></tr><tr><td>ICC (3,k)</td><td>0.811</td><td>0.823</td><td>0.852</td><td>0.810</td><td>0.824</td></tr></table>

## A.2 Training Data Sources

The auxiliary training pool comprises 2,291 videos: 735 self-collected videos and 1,556 Condensed-Movies clips from the $R ^ { 3 } .$ -FDT video collection (Niu et al., 2026). Applying our multi-agent data construction pipeline to these videos yields 19,759 QA samples, including 6,490 from the self-collected videos and 13,269 from Condensed-Movies. We select subsets of this pool for SFT and PAFRO, with training settings detailed in Appendix C.1.

## B Human Validation and Evaluation

## B.1 Dataset Validation

Four students with backgrounds in computer science manually review the generated questions. Before large-scale annotation, they independently score a preliminary subset on four dimensions: answer reasonableness, video dependency, distractor quality, and question validity. Each dimension uses a 5-point Likert scale, ranging from 1 (poor quality) to 5 (superior quality). Substantial disagreements are discussed under the guidance of domain experts to establish a consistent evaluation standard. After calibration, all annotators independently score 100 randomly sampled instances to assess inter-annotator agreement. Table 6 reports ICC(3,k) scores above 0.80 across all four dimensions, indicating strong agreement. The annotators then verify the remaining question-answer pairs against the raw videos using the same four criteria, discarding any instance with an average score below 3.5.

## B.2 Human Baseline Evaluation

To evaluate human performance on AffToM, we recruited five graduate students with backgrounds in computer science and multimodal learning as human testers. Although affective states may admit semantically related interpretations, each evaluated multiple-choice question was manually validated to contain a single best-supported answer grounded in the video context. Before the formal evaluation, all testers independently answered the same 30 randomly sampled questions without time constraints. Across this shared calibration subset, no pair of testers differed on more than three questions, indicating broadly consistent task interpretation. For the formal evaluation, we randomly sampled 250 multiple-choice questions and partitioned them evenly into five non-overlapping subsets, with each tester answering one subset. Thus, each question received one human response. Aggregating the responses across all subsets yielded 222 correct answers out of 250 questions, corresponding to an overall accuracy of 88.80%.

## B.3 LLM-as-a-Judge Validation

To validate GPT-5-mini as the automatic evaluator for open-ended responses, a human annotator independently scores 100 responses generated by Qwen3.5-27B and another 100 responses generated by Qwen3-VL-32B using the same EC and FC rubrics. For Qwen3.5-27B, the Spearman correlations between human and GPT-5-mini scores are 0.766 for EC and 0.727 for FC. For Qwen3- VL-32B, they are 0.790 for EC and 0.769 for FC. These correlations support the use of GPT-5-mini for automatic scoring under the EC and FC rubrics.

## C Experimental Implementation Details

## C.1 Training Settings

We implement AffToM-R1 in ms-swift<sup>1</sup> using Qwen3.5-4B as the backbone. Both stages use AdamW, a constant learning rate schedule without warmup, bfloat16 precision, and a random seed of 42. Training inputs contain 16 video frames and the available subtitle text. Question difficulty is estimated from the pretrained backbone’s accuracy over eight rollouts per question.

SFT optimization. Using the retained SFT subset, we train for two epochs with a learning rate of $2 \times 1 0 ^ { - 5 }$ and a batch size of 16.

PAFRO. The PAFRO policy and its fixed reference are initialized from the SFT-trained model. We train on 5k QA pairs for one epoch with a learning rate of $2 \times 1 0 ^ { - 6 }$ and a KL coefficient of 0.04. We use a batch size of 32 and generate four rollouts per question at temperature 1.0. Qwen3.5-27B serves as the reward model for the faithfulness and consistency rubrics. The reward weights $( \lambda _ { \mathrm { a c c } } , \lambda _ { \mathrm { f m t } } , \lambda _ { \mathrm { p e r s p } } , \lambda _ { \mathrm { f a i t h } } , \lambda _ { \mathrm { c o n s } } )$ are set to (1.0, 0.1, 0.1, 0.4, 0.2).

![](images/202f5c2b031b7027918c1906d265352771704f922b2ec472cff573015a476c9a.jpg)

![](images/fe92df923e27ac508a7e4d8444e14a45e65e68df189c7a99feea2bff52db0044.jpg)  
Figure 9: Distribution of QA types in the dataset and the distribution of QA types across different video-duration ranges.

## C.2 SFT Trajectory Construction

We construct each SFT trajectory with two successive calls to Qwen3.5-27B. In the first call, a perspective-classification prompt provides the video, question, and answer options, and asks the model to determine whether solving the question requires focused reasoning about the target character’s information state and mental state. Questions that require character-specific beliefs or mentalstate attribution are assigned the character-centered label, whereas questions that can be answered from the broader event context are assigned the contextcentered label. In the second call, we provide the same video-question input together with the predicted perspective label and ask the model to generate a reasoning trajectory in the corresponding format. For character-centered questions, the trajectory follows Clues → Mental State → Inference. For context-centered questions, it follows Clues → Inference. The second prompt also constrains the response to include a <perspective> </perspective> field, a reasoning block enclosed by <think></think>, and a final answer enclosed by <answer></answer>. All MPAR questions use the character-centered structure, while the remaining questions receive the structure selected in the first step. We retain trajectories whose final answers match the reference answers, and split the resulting approximately 20k-sample pool into SFT and PAFRO subsets according to the difficulty estimated from the pretrained backbone.

## C.3 PAFRO Optimization with GRPO

For each video-question pair $( v , q ) \in \mathcal { D } _ { \mathrm { R L } }$ , the rollout policy $\pi _ { \mathrm { o l d } }$ samples a group of K responses $\{ y _ { i } \} _ { i = 1 } ^ { K }$ . Each response receives the weighted reward $r _ { i }$ in Equation 9, which combines rule-based checks with the two rubric scores. We compare responses within the same group rather than fitting a separate value model. Writing $\begin{array} { r } { \bar { r } = K ^ { - 1 } \sum _ { j = 1 } ^ { K } \bar { r } _ { j } } \end{array}$ the response-level advantage is

$$
\hat { A } _ { i } = \frac { r _ { i } - \bar { r } } { \mathrm { S t d } ( r _ { 1 } , . . . , r _ { K } ) + \delta } ,\tag{10}
$$

where $\delta > 0$ stabilizes groups with little reward variation. Thus, a response is favored when its reward exceeds that of other responses to the same input.

The trainable policy $\pi _ { \theta }$ and fixed reference policy $\pi _ { \mathrm { r e f } }$ are both initialized from the SFT-trained model, while $\pi _ { \mathrm { o l d } }$ denotes the policy used to generate the current rollout group. For token $y _ { i , t }$ , let $\rho _ { i , t } ~ = ~ \pi _ { \theta } ( y _ { i , t } ~ \vert ~ v , q , y _ { i , < t } ) / \pi _ { \mathrm { o l d } } ( y _ { i , t } ~ \vert ~ v , q , y _ { i , < t } )$ be the probability ratio between the updated and rollout policies. The clipped token objective is

$$
\ell _ { i , t } ( \theta ) = \operatorname* { m i n } \Bigl ( \rho _ { i , t } \hat { A } _ { i } , \ \mathrm { c l i p } ( \rho _ { i , t } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i } \Bigr ) ,\tag{11}
$$

where ϵ limits the size of policy changes. We maximize

$$
\mathcal { I } _ { \mathrm { G R P O } } ( \boldsymbol { \theta } ) = \mathbb { E } \left[ \frac { 1 } { K } \sum _ { i = 1 } ^ { K } \frac { 1 } { | \boldsymbol { y } _ { i } | } \sum _ { t = 1 } ^ { | \boldsymbol { y } _ { i } | } ( \boldsymbol { \ell } _ { i , t } ( \boldsymbol { \theta } ) - \beta \boldsymbol { d } _ { i , t } ) \right] ,\tag{12}
$$

where $d _ { i , t }$ is the sampled token-level KL penalty relative to $\pi _ { \mathrm { r e f } }$ , and $\beta$ sets its weight. The expectation covers training inputs and their sampled response groups. Responses receive equal weight, and tokens are averaged within each response. The same response-level advantage applies to all tokens of a rollout, linking the reward for its answer and reasoning to the policy update. Clipping and the fixed-reference penalty constrain the update while the group comparison favors higher-reward trajectories.

## C.4 Reward Definitions

This section details the five reward components combined in Equation 9.

Correctness reward $R _ { \mathrm { a c c } }$ . We compare the extracted answer $a ( y )$ with the reference $a ^ { * }$

$$
R _ { \mathrm { a c c } } ( y ) = \mathbb { I } ( a ( y ) = a ^ { * } ) ,\tag{13}
$$

where $\mathbb { I } ( \cdot )$ is the indicator function defined above.   
Invalid answer extractions receive zero reward.

Format reward $R _ { \mathrm { f m t } }$ . This reward preserves the response structure learned during SFT. Let $\mathcal { F }$ denote responses that match the prescribed structure over the entire output. The format is:

$$
\begin{array} { l } { { < \mathrm { t h i n k } > } } \\ { { < \mathrm { p e r s p e c t i v e } > l a b e l < / { \mathrm { p e r s p e c t i v e } } > } } \\ { { r e a s o n i n g } } \\ { { < / { \mathrm { t h i n k } } > } } \\ { { < \mathrm { a n s w e r } > a n s w e r l e t t e r < / { \mathrm { a n s w e r } } > } } \end{array}
$$

The <think> block must begin with the perspective field followed by reasoning, and the subsequent <answer> block must contain a single answer letter. Whitespace variations are permitted, but additional text outside these blocks is not allowed. This reward checks structural compliance. Perspective alignment and answer correctness are evaluated separately. We define

$$
R _ { \mathrm { f m t } } ( y ) = \mathbb { I } ( y \in { \mathcal { F } } ) .\tag{14}
$$

Perspective reward $R _ { \mathrm { p e r s p } } .$ . This reward compares the generated perspective label with the annotated route and target character. We extract the predicted label from the <perspective> field and normalize superficial differences in case and formatting. Context-centered questions require a contextcentered prediction, while character-centered questions require a character-specific label sharing at least one character-descriptive token with the normalized reference. This lexical matching accommodates variation in character names and descriptions without requiring exact string equality. The reward is:

$$
R _ { \mathrm { p e r s p } } ( y ) = { \left\{ \begin{array} { l l } { 1 , } & { { \mathrm { m a t c h e d } } , } \\ { 0 , } & { { \mathrm { o t h e r w i s e } } . } \end{array} \right. }\tag{15}
$$

Missing or ambiguous perspective extractions receive zero reward.

Faithfulness reward $R _ { \mathrm { f a i t h } }$ . The reward model assesses the accuracy and relevance of mentalstate claims by comparing them with the dynamic character interaction graph $G$ provided as reference, using the sample’s sub-event or whole-video scope. This reward applies to both routes, assessing context-centered reasoning without imposing a single-character viewpoint. For character-centered responses, it evaluates these claims from the viewpoint named in the generated label. We normalize the reward model’s assessment under the faithfulness rubric to obtain

$$
R _ { \mathrm { f a i t h } } ( y ) = \mathrm { N o r m } ( \mathcal { M } _ { \mathrm { f a i t h } } ( q _ { \mathrm { t e x t } } , p ( y ) , \tau _ { \mathrm { r } } ( y ) , \mathcal { G } ) )\tag{16}
$$

where $\mathrm { N o r m } ( z ) = z / 2$ maps the raw rubric score $z \in \{ 0 , 1 , 2 \}$ to [0, 1].

Consistency reward $R _ { \mathrm { c o n s } }$ . The multimodal reward model evaluates whether the reasoning in $y$ supports its final answer using evidence from video v and question $q .$ It receives sampled frames and associated subtitles as scoring evidence, without the reference answer. The reward is

$$
R _ { \mathrm { c o n s } } ( y ) = \mathrm { N o r m } ( \mathcal { M } _ { \mathrm { c o n s } } ( v , q , y ) ) ,\tag{17}
$$

using the same normalization, with $q$ including the answer options and $y$ containing both the reasoning and the final answer. The rubric favors clear evidential support and penalizes contradictions or unsupported conclusions. The functions $\mathcal { M } _ { \mathrm { f a i t h } }$ and ${ \mathcal { M } } _ { \mathrm { { c o n s } } }$ denote separate calls to the reward model, each using its own rubric prompt to produce a response-level score.

## D Case Study

Figure 10 presents representative failures that illustrate the error patterns above and the degradation on open-ended questions. In Figure 10 (a), the model’s reasoning trace reveals a clear temporal information leakage error. Although the question is anchored before any spray is used, the model justifies its answer by referring to the woman’s later reaction, namely that she sprays the man and calls him a bad person. This evidence is unavailable at the queried moment: before the spray, the man has not yet inferred that the woman mistakes him for a robber or feels threatened by him. By using this post-event reaction to infer the man’s preevent belief, the model answers from an omniscient timeline rather than from the man’s temporally constrained perspective.

![](images/e271beabae29ac1796dcd746f6e285af3abf9c0d02f2ef804f6632e74d1c4c9d.jpg)  
(a) Temporal Information Leakage  
Figure 10: Representative failure cases of MLLMs on AffToM, including three major error types in MPAR—Temporal Information Leakage, Recursive Perspective Conflation, and Unwarranted Affective Deduction—as well as a typical error in open-ended question answering involving missing recursive characters.

In Figure 10 (b), the model’s reasoning trace reveals a recursive perspective conflation error. The question targets the boyfriend’s sister’s belief about the girlfriend’s thoughts and feelings, but the model grounds its answer in the girlfriend’s own latent dissatisfaction. It cites Chu Chu’s remarks about needing a ceremonial wedding and her concerns about the house and betrothal gift, and then infers that the boyfriend’s sister’s tone and words show she perceives Chu Chu’s disappointment. In this way, the model treats the girlfriend’s internal disappointment as if it were already represented in the sister’s belief. The predicted answer therefore conflates the girlfriend’s true inner state with the sister’s second-order perspective, rather than reasoning only from what the sister can observe and infer.

In Figure 10 (c), the model’s reasoning trace exposes an unwarranted affective deduction rather than a temporal or perspective-tracking error. The model stays within the Emperor’s current viewpoint, but overinterprets his dialogue and tone. From his concern and the utterance “You don’t want to see me again?”, it infers that he believes the concubine is avoiding him out of guilt, sorrow, self-punishment, or a desire to spare him pain. This goes beyond the available evidence: the scene only supports that she claims illness and that the Emperor responds with concern for her condition. The failure is therefore an unsupported evidenceto-emotion mapping, where sparse cues are turned into a strong but unjustified emotional-causal chain.

In Figure 10(d), the model initially preserves the outer belief chain but partially collapses the recursive attribution in its subsequent explanation. Specifically, the final statement directly attributes the girlfriend’s emotional state to the woman in brown, rather than maintaining the woman’s belief about the boyfriend’s inference. The response therefore represents the recursive structure only incompletely.

## E Multi-Agent Data Construction

## E.1 System Configuration

Our multi-agent system is implemented using the LangGraph framework <sup>2</sup> and consists of six agents: the Event Agent, State Agent, Question Generation Agent, Role Reflection Agent, Revision Agent, and Evaluation Agent. The Event Agent and State Agent extract structured events and character-level mental states, which are integrated into G for subsequent question generation and refinement. Based on preliminary experiments, we use Seed-1.8 as the backbone of the Event Agent and State Agent due to its stronger semantic extraction capability, while the remaining reasoning and refinement agents use Qwen3-VL-32B as their backbone.

All agents use top $\begin{array} { r l r } { p } & { { } = } & { 0 . 9 5 } \end{array}$ The Event Agent, State Agent, and Evaluation Agent use temperature = 0 for deterministic extraction and evaluation, while the remaining agents use temperature = 0.7 for diverse generation, reflection, and revision. During iterative refinement, we randomly sample $| \mathcal { Q } _ { \mathrm { h u m a n } } | = 5$ human-authored reference questions from the seed pool for pairwise comparison. We set the acceptance threshold to $\tau = 0 . 8$ , the trade-off coefficient to $\lambda = 0 . 7 .$ and the maximum number of refinement rounds to $T _ { \mathrm { m a x } } = 3$ . For recursive perspective expansion, we set the maximum recursive depth to $d _ { \operatorname* { m a x } } = 3 ,$ balancing reasoning complexity and annotation reliability.

## E.2 MPAR Question Generation Example

## E.2.1 Event Extraction Output

Here we present an example of an event list extracted from a single video. It records the main characters, event timestamps, locations, event summaries, and involved individuals. The event list directly provides the event-level context for baselevel question generation and, together with the corresponding individual-level information, is used to construct the dynamic character interaction graph G for recursive perspective expansion.

Event List Example:

" character\_list ": [{   
" character\_id ": " Wang Zha",   
" description ": "A young man in a   
black jacket , trying to   
reconcile with his girlfriend   
Leng Meng and later   
proposing to her . His parents   
misinterpret his   
interactions with his friend   
11   
} ,   
{   
" character\_id ": " Leng Meng " ,   
" description ": " Wang Zha 's   
girlfriend , wearing a white   
coat , sitting upset on a park   
bench after a   
misunderstanding with Wang   
Zha ."

} ,   
{   
" character\_id ": " Leng Meng 's   
male friend ",   
" description ": "A man in a denim   
jacket with glasses ,   
protecting Leng Meng from   
Wang Zha initially , and later   
mistakenly targeted by Wang   
Zha 's parents ."   
} ,   
{   
" character\_id ": " Wang Zha 's   
father ",   
" description ": "An older man in   
a gray jacket , who   
misinterprets Wang Zha 's   
interactions with his male   
friend as a romantic   
relationship and becomes   
distraught ."   
} ,   
{   
" character\_id ": " Wang Zha 's   
mother ",   
" description ": "An older woman   
in a beige coat , who tries to   
calm her husband and also   
misinterprets Wang Zha 's   
interactions , later offering   
money to Leng Meng 's friend ."   
}] ,   
" events ": [   
{   
" event\_id ": " event\_1 ",   
" timestamp ": "00:00 - 00:37" ,   
" location ": " Outdoor park area ",   
" description ": " Wang Zha tries   
to approach the upset Leng   
Meng , but her male friend   
blocks him . Wang Zha argues   
with the friend , using casual   
terms like 'dear ' which are   
overheard by Wang Zha 's   
parents . The parents   
misinterpret the interaction   
as Wang Zha being in a   
romantic relationship with   
the male friend .",   
" interaction\_pairs ": [   
[" Wang Zha ", " Leng Meng 's male   
friend "],   
[" Leng Meng 's male friend ",   
Wang Zha "],   
[" Leng Meng 's male friend ",   
Leng Meng "],   
[" Wang Zha " , " Leng Meng "] ,   
[" Leng Meng ", " Wang Zha "],   
[" Wang Zha 's father ", " Wang   
Zha 's mother "],   
[" Wang Zha 's mother " , " Wang   
Zha 's father "],   
[" Wang Zha 's father ", " Wang   
Zha "] ,   
[" Wang Zha 's mother ", " Wang   
Zha "]   
]   
} ,

![](images/3c36e6df65b6ac622b7f5b0d301d66a083500baddc488e192ee44a9120bb8fd9.jpg)  
E.2.2 Graph Initialization and Recursive Question Generation

![](images/ceae96383ee74bc8e623fcea9c53ddf94e8bf0413013a47fb6abc657489af60c.jpg)  
Figure 11: An example of a character interaction graph. The nodes represent “L”: “Leng Meng”, “Fri”: “Leng Meng’s male friend”, “W”: “Wang Zha”, “M”: “Wang Zha’s mother”, and “F”: “Wang Zha’s father”. The left panel shows the initialized graph, while the right panel highlights the characters selected for recursive perspective construction during MPAR generation.

Based on the extracted events E and individuallevel information S, we construct the dynamic character interaction graph G, as illustrated in Figure 11. Each node represents an involved character, while each directed edge indicates the direction of an observed interaction. The left panel of Figure 11 shows the initialized graph, and the right panel illustrates how it supports Recursive Perspective Expansion for MPAR question generation.

We first establish a base-level affective state or belief for a target character grounded in the event context. In the running example, Wang Zha is selected as the target character. To expand the reasoning structure, we identify characters who directly interact with or observe the current character within the current event context encoded in $G _ { i }$ and treat them as candidate outer observers. When multiple candidates are available, the LLM jointly examines the graph $G _ { i } ,$ the event-localized video segment, and the current recursive belief chain. The selection considers whether the event supports information asymmetry between the candidate and the current subject, whether the candidate’s belief concerns the current subject, and whether the associated emotion, intention, or desire is relevant to the target relation. The selected character is then introduced as the new outer observer, extending the recursive perspective chain.

In Figure 11, Wang Zha’s father, represented by node F, is selected as the outer observer. This expands the reasoning structure from Wang Zha’s base-level mental state to a second-order perspective. The graph $G _ { i }$ , the corresponding video segment, and the current belief chain are subsequently provided to the Question Generation Agent, which produces an MPAR question-answer pair together with plausible distractors.

## E.3 Pipeline Robustness Audit

To further assess the robustness of the multi-agent construction pipeline, we conduct an additional audit of representative intermediate outputs. During dataset construction, intermediate supervision is incorporated into the pipeline: before large-scale generation, we inspect stage-level outputs and proceed only after the outputs become stable. To quantitatively examine whether upstream errors could systematically affect final QA quality, we further re-audit 20 randomly sampled generation traces across three stages that are most relevant to final QA validity: event extraction, role/mental-state extraction, and recursive QA construction.

We apply stage-specific criteria to inspect intermediate outputs at the level of individual checkable positions. Each occurrence that violates a criterion contributes one error count, and the error rate is computed as the total number of errors divided by the total number of corresponding positions inspected across the 20 generation traces. For event extraction, the inspected positions include character-interaction edges, event temporal boundaries, character identities, and required output format. Errors are counted for incorrectly constructed interaction edges, temporal intervals that substantially deviate from the event’s reasonable boundaries, misidentified characters, or violations of the prescribed output format. For role/mental-state extraction, we inspect character profiles, event-specific mental-state entries for each present character, and required output format. Errors include inaccurate profile information, missing mental-state entries for characters involved in an event, mental-state analyses that are clearly inconsistent with the video evidence, and formatting violations. For recursive QA construction, we inspect every belief-extraction step along the recursive perspective chain, together with the required output format. Each incorrect belief extraction or formatting violation contributes one error count.

As shown in Table 7, most inspected intermediate positions are consistent with the video, although residual errors remain across all three audited stages. Role/mental-state extraction shows an error rate of 5.22%, reflecting occasional inaccurate mental-state analysis in sub-events. Recursive QA construction has an error rate of 7.41%, mainly caused by unsupported relational or affective links. Given the limited sample size and the positionlevel nature of this audit, these results should be interpreted as a diagnostic analysis rather than a dataset-wide reliability estimate. The observed errors, particularly unsupported relational or affective links, are further screened during final human validation, where annotators verify, revise, or discard QA candidates based on the original video evidence and event-localization information.

Table 7: Pipeline robustness audit on 20 randomly sampled generation traces. Error rates are computed over all inspected intermediate positions.
<table><tr><td>Stage</td><td>Primary Observed Error</td><td>Error Rate</td></tr><tr><td>Event Extraction</td><td>Incorrect interaction edges</td><td>3.67%</td></tr><tr><td>Role/Mental-State Extraction</td><td>Incorrect mental-state analysis in events</td><td>5.22%</td></tr><tr><td>Recursive QA Construction</td><td>Unsupported relational/affective links</td><td>7.41%</td></tr></table>

## F Prompt Details

## F.1 Prompt for Multiple-Choice Questions

# Instruction:   
These are the frames of a video.   
(Optional) The video’s subtitles are pro  
vided below when subtitle assistance is en  
abled:   
{Subtitles}   
Select the best answer to the follow  
ing multiple-choice question. Re  
spond with the exact final sentence   
“<answer>X</answer>”, where X is one of   
A, B, C or D.   
# Question: {Question}   
# Options:   
A: {Option A}   
B: {Option B}   
C: {Option C}   
D: {Option D}   
# Answer:

## F.2 Prompt for Open-Ended Questions

## # Instruction:

These are the frames of a video.

(Optional) The video’s subtitles are pro-

vided below when subtitle assistance is en-

## abled:

{Subtitles}

Please answer the following question.

\# Question: {Question}

## # Task:

Your answer must explicitly describe a three-level belief structure. Specifically, you should express Agent A’s interpretation of Agent B’s belief about Agent C’s perspective on the event, as well as Agent A’s inference of Agent B’s belief about Agent C’s emotional attitude toward the event.

## F.3 Prompt for LLM-as-a-Judge in Open-Ended questions

## # Instruction

You are a helpful assistant in evaluating open-ended Q&As. Please act as an unbiased judge. The reference answer has been manually validated against the source video and serves as the factual anchor for evaluation. Evaluate the assistant’s response based only on the question and the humanvalidated reference answer. Assess whether the response preserves the factual content, character roles, and recursive viewpoint structure represented in the reference answer. Do not introduce or reward additional interpretations solely because they appear plausible.

## # Notes

\- Your evaluation should determine whether the assistant effectively follows the requirements of the question and correctly addresses it.

\- You must assess the response across the two dimensions and score each dimension independently.

\- Do not let the length of the response influence your judgment.

\- Do not favor any specific assistant identity or position. Remain as objective as possible.

\- Do not penalize wording differences when the response is semantically equivalent to the reference answer.

## # Criteria

Use scores to represent the quality of the

assistant’s response. Below are the detailed scoring rubrics.

## ## Emotion Correctness (EC)

This dimension evaluates whether the response correctly identifies the emotional state of the target character, consistent with the reference answer. Factual alignment of the emotion is the primary criterion.

\- 1 – Poor: The predicted emotion is entirely opposite to the reference answer. Even if the nested perspective is grammatically fluent, an opposite emotion must receive the lowest score.

\- 2 – Fair: The predicted emotion shows weak relevance and contains noticeable errors, such as identifying an incorrect or overly weak emotion.

\- 3 – Moderate: The general emotional direction is somewhat correct, but lacks clarity, specificity, or accurate recursive attribution.

\- 4 – Good: The emotional polarity is highly consistent with the reference answer, with only minor differences in wording or intensity.

\- 5 – Excellent: The predicted emotion fully matches the reference answer in polarity, intensity, and semantic precision, and is correctly attributed within the recursive belief structure.

## ## Factual Consistency (FC)

This dimension evaluates the accuracy of the embedded viewpoint content and whether the response correctly represents the recursive belief structure (i.e., Agent A’s perspective on Agent B’s belief about Agent C’s view).

\- 1 – Poor: The core viewpoint contradicts the reference answer or the recursive chain is missing or incorrect.

\- 2 – Fair: The viewpoint is largely inaccurate, focused on the wrong event, or severely misinterprets the characters’ intentions.

\- 3 – Moderate: The recursive structure is identifiable, but the viewpoint only partially aligns with the reference and contains omissions or minor role confusion.

\- 4 – Good: The recursive chain is clear and correctly ordered. The viewpoint closely matches the reference answer, with only minor semantic differences.

\- 5 – Excellent: The recursive structure is explicit, correctly ordered, and unambiguous. The embedded viewpoint perfectly matches the reference answer in meaning, focus, and interpretation.

## # Test Question

QUESTION: {question\_text}

ASSISTANT ANSWER: {pred\_answer}

GROUND TRUTH: {ground\_truth}

## # Output Format

You MUST output your evaluation strictly in the following JSON format:

{   
" Emotion\_Correctness ": < score   
from 1 to 5>,   
" Factual\_Consistency ": <score   
from 1 to 5 >   
}