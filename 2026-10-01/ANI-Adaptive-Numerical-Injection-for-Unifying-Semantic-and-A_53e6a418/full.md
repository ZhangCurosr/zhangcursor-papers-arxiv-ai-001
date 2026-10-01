# ANI: Adaptive Numerical Injection for Unifying Semantic and Arithmetic Representations in Numerical Reasoning

Jinsung Jeon KAIST InnoCORE LLM Seoul National University Korea University jinsungjeon@korea.ac.kr

Seung-won Hwang Seoul National University seungwonh@snu.ac.kr

## Abstract

Precise numerical reasoning with Large Language Models (LLMs) is essential for expanding their applicability to complex real-world tasks. However, text-based tokenization often fragments numbers, significantly hindering precise arithmetic reasoning. Meanwhile, numerical embeddings, despite arithmetic precision, rely on context-agnostic substitution that disregards the semantic role of numbers as identifiers. To combine the complementary strengths, we propose ANI (Adaptive Numerical Injection), a hybrid framework that governs the selective injection of numerical features based on the semantic context. By employing a context-aware gating mechanism, we selectively inject numerical embeddings (specifically FoNE) into the latent space, explicitly preserving nominal identifiers while enhancing quantitative operands. Through extensive evaluations across various LLMs, we demonstrate that ANI enhances MATH performance by 9.5 points over the official reference model, while maintaining robust performance on general linguistic benchmarks<sup>1</sup>.

## 1 Introduction

Large Language Models (LLMs) have demonstrated remarkable capabilities across a wide range of natural language processing tasks (Brown et al., 2020; Achiam et al., 2023; Grattafiori et al., 2024; Yang et al., 2025). Despite this success, their performance in precise numerical reasoning remains a significant challenge due to the inherent limitations of discrete tokenization. Traditional architectures typically treat numbers as fragmented sequences of text tokens (e.g., sub-words), relying on linguistic probability distributions to predict the next numerical digit (Wallace et al., 2019; Nogueira et al., 2021). Consequently, the model struggles to capture the numerical magnitude and mathematical properties, often generating results based on surface-level text patterns rather than rigorous logical computation (Dziri et al., 2023).

![](images/974244ea8b19ddb72e6dfbba4528e66f3751e823ab102b93e8ff8f65e6a3c07c.jpg)  
Figure 1: Overview of the proposed adaptive representation. Unlike existing methods that rely exclusively on uniform textual representations, our framework dynamically integrates a unified architecture by explicitly distinguishing between nominal representations and quantitative representations.

To enhance arithmetic precision, recent works have introduced continuous numerical embeddings (Golkar et al., 2023; Schwartz et al., 2024; Zhou et al., 2025). However, these methods often fall into the trap of Context-Agnostic Substitution, or indiscriminately replacing with a numerical embedding. While mathematically precise for pure arithmetic (e.g., addition, subtraction), this naive substitution leads to semantic erasure for tokens, working as nominal identifiers.

We argue that this deficiency presents a critical bottleneck in numerical reasoning, ranging from financial analysis to technical documentation. Figure 1 illustrates real-life numerical reasoning that require a unified representation of numerical data embedded within natural language contexts. For example, in the sentence "Room 505 has a capacity of 10", the number "505" serves as a nominal identifier (a room ID) that should be processed as a textual entity. In contrast, "10" is a quantitative value intended for arithmetic reasoning. Standard tokenizers would treat all digits uniformly, potentially misinterpreting a non-calculable identifier like "Room ID 505" as a summable quantity, or conversely, failing to capture numerical magnitudes by treating them as mere text tokens. Consequently, reasoning fails when models cannot distinguish between numeric labels and quantitative values.

To resolve this dilemma, we propose ANI (Adaptive Numerical Injection). Unlike static approaches, ANI does not blindly overwrite representations. Instead, it probes the backbone model’s latent states to interpret the role of a number, either as nominal IDs or quantitative operands. By deriving a differentiable binary decision via this internal probing, ANI adaptively injects mathematical structures only when the context explicitly demands arithmetic reasoning. This balanced approach allows the model to preserve its linguistic integrity while selectively enhancing numerical precision. Consequently, ANI excels not only in numerical reasoning where numbers and text coexist but also remains robust in text- or numeric-only scenarios. Contributions. Our contributions are as follows:

• We propose ANI, which dynamically adapts between nominal identifiers and quantitative representations by injecting mathematical topology only when necessary.

• ANI enables retaining pre-trained reasoning and selectively boosting arithmetic precision, which we identified as a key bottleneck in numerical reasoning.

• We validate the robustness of ANI through extensive evaluations across diverse model families and scales. Notably, ANI achieves a 9.5-point improvement on the MATH benchmark while incurring a marginal performance drop of only 1.1% on general linguistic tasks in Qwen3-8B.

## 2 Related work and Preliminaries

Arithmetic Reasoning in LLMs. While LLMs have achieved remarkable success in natural language understanding, they persistently struggle with arithmetic reasoning and precise calculation. Early hypotheses relied on scaling laws (Kaplan et al., 2020; Wei et al., 2022a), assuming that larger models would naturally acquire mathematical proficiency. However, empirical studies show that standard Transformers (Vaswani et al., 2017) often fail to perform consistent arithmetic operations due to the fragmented nature of subword tokenization mechanisms, such as Byte-Pair Encoding (Sennrich et al., 2016). Since these algorithms prioritize text compression, they often split numbers into inconsistent chunks (e.g., "1234" → "12", "34"), disrupting semantics necessary for calculation.

To address these architectural deficiencies, current state-of-the-art methods predominantly employ inference-time strategies such as Chain-of-Thought (CoT) prompting (Wei et al., 2022b; Tang et al., 2026), which decomposes complex problems into intermediate natural language steps. More recently, program-aided approaches have gained prominence, where models generate and execute programming code (e.g., Python) to offload precise calculations to external interpreters (Chen et al., 2022; Gao et al., 2023). While effective, these methods mitigate the limitations externally rather than the root cause: the suboptimal internal representation of numbers within the model itself.

Number Representations. Recent literature highlights that LLMs inherently possess highly accurate internal representations of numbers that remain stable throughout internal processing (Kadlcík et al. ˇ , 2025). Despite these accurate internal representations, structural fragmentation caused by subword tokenization severely hinders complex arithmetic. As empirically demonstrated in Section 4.3, this fragmentation leads to calculation failures on longer sequences.

To address this, early attempts to improve numeracy in LLMs often relied on digit-wise representations or naive additive fusion $( h _ { n u m } $ $h _ { n u m } + e _ { n u m } )$ . In these approaches, numerical features are simply summed with textual embeddings at the input layer. However, as analyzed in recent literature (Zhou et al., 2025), such indiscriminate addition leads to signal interference, where the quantitative and semantic signals become entangled and indistinguishable within the same vector space. This interference often leads to catastrophic forgetting, as the model’s pre-trained linguistic manifold is disrupted by raw numerical noise.

To mitigate such entanglement, state-of-the-art methods such as xVal (Golkar et al., 2023) and FoNE (Zhou et al., 2025) adopt a static substitution strategy. By replacing textual tokens with precise single-token embeddings $( h _ { n u m }  e _ { n u m } )$ , these methods successfully capture mathematical properties like magnitude and modular arithmetic without the noise of naive addition. Yet, this creates a new structural dilemma: semantic erasure. Since the substitution is static and context-agnostic, numbers serving as nominal identifiers (e.g., "Room 505") lose their linguistic identity, leading to performance degradation in tasks where numbers and text must coexist. ANI reconciles this conflict by transitioning from context-agnostic substitution back to an additive scheme, but one that is strictly governed by an ‘adaptive’ and ‘context-aware’ gating process $( h _ { n u m }  h _ { n u m } + \alpha \cdot e _ { n u m } )$

![](images/5364539db45d9d668d7d93a4eca0c1e912af712027a7b68570e0a4f2d2572cde.jpg)  
Figure 2: The overall architecture of ANI. Numerical embeddings are generated from raw scalar values and mapped to the model’s hidden dimension via a Linear Projection. Simultaneously, the Context-aware Gating Network computes a scalar coefficient (α) based on both the hidden state of each numerical token and its numerical embedding. The final representation is obtained by adding the projected numerical embedding, scaled by α, to the original hidden state. Note that non-numerical tokens bypass this module to preserve their linguistic features.

## 3 Proposed Method

In this section, we introduce ANI (Adaptive Numerical Injection), a hybrid architecture designed to adaptively inject numerical embeddings into the hidden states of pre-trained LLMs.

## 3.1 Problem Formulation

Existing numerical integration strategies suffer from a trade-off between semantic preservation and arithmetic precision. While additive fusion preserves semantics, it induces signal entanglement with the linguistic manifold, hindering precise calculation. Conversely, static substitution ensures precision but triggers semantic erasure, stripping numbers of their nominal identity. Both failure modes stem from context-blind intervention at the input level, which forces a functional interpretation before the model gains sufficient contextual depth to distinguish between quantitative operands and nominal identifiers.

To resolve this structural dilemma, we propose ANI, a framework designed to function as a contextaware semantic probe. Instead of premature intervention at the input level, ANI shifts the numerical injection to the optimal layer where the model has developed sufficient contextual maturity. By leveraging these contextually enriched representations, ANI enables the selective integration of numerical features, ensuring that mathematical structures are reinforced only when necessitated by the semantic context. This approach allows the model to enhance its numerical precision while fundamentally preserving its pre-trained linguistic integrity.

## 3.2 Overall Workflow

Based on the formulation above, we implement ANI as a plug-and-play module that intervenes in the intermediate layers of the backbone LLM. As illustrated in Figure 2, the framework operates through the following four phases:

1. Optimal Layer Selection: We select an intermediate layer l where the hidden states have developed sufficient contextual maturity to resolve numerical ambiguity.

2. Topology Encoding (Numerical Path): Raw scalars are transformed into numerical embeddings and aligned with the latent space through a Linear Projection.

3. Semantic Probing (Gating Path): A Gating Network computes a discrete gating coefficient α by interpreting the terminal token’s hidden state to distinguish between nominal and quantitative roles.

![](images/3e0a2912e601a63a688bdab59ad3e05e9d601d44088087ea7b99b574b51b3239.jpg)  
Figure 3: Linear probing analysis for optimal layer selection. We evaluate the degree of numerical disambiguation across layers using a diagnostic dataset. The peak in classification accuracy identifies the layer with maximum contextual maturity, which serves as the optimal point for adaptive numerical injection.

4. Adaptive Integration: The aligned embedding, scaled by α, is adaptively injected via gated residual connection, while nonnumerical tokens bypass the module to ensure zero interference.

Based upon this workflow, the following subsections detail the implementation of our framework.

## 3.3 Optimal Layer Selection and Topology Encoding

ANI is designed to integrate numerical information at a specific intermediate depth where the hidden representations have attained sufficient contextual maturity to resolve the functional ambiguity between nominal and quantitative roles.

Optimal Layer Selection. To address the limitations of context-agnostic input-level substitution, we identify an optimal injection layer l that exhibits sufficient contextual maturity via linear probing analysis (cf. Figure 3). Specifically, we train layer-wise linear probes on a balanced diagnostic dataset to classify the functional roles of numerical tokens (nominal vs. quantitative). The classification accuracy serves as a proxy for the degree of numerical disambiguation achieved at each depth. We designate the layer where this accuracy peaks as the optimal locus for numerical injection, allowing the gating mechanism to leverage contextually enriched representations $\mathbf { h } ^ { l } \in \mathbb { R } ^ { d _ { m o d e l } }$ , where $d _ { m o d e l }$ represents the hidden dimension of the backbone model. For our experiments with Qwen3-8B, we select l = 24 as the injection point based on this protocol; a detailed empirical validation of this selection is provided in Section 4.3.

Topology Encoding (Numerical Path) Once the injection layer is identified, we ensure that the numerical information is represented in a format that preserves its intrinsic mathematical prop-

![](images/eee360754a217a0e1046443d7ced6541269d642ed076e05d0e5dc64e14fc10b6.jpg)

Figure 4: Gating mechanism and last token injection. ANI selectively integrates numerical embeddings into the terminal sub-token’s hidden state via an adaptive gate α. This mechanism ensures that mathematical structures are reinforced only when semantically required, preserving the model’s linguistic integrity.

erties. To address the discreteness of subword tokenization, we adopt Fourier Number Embedding (Zhou et al., 2025) as our base encoder to represent each numerical value as a single-token entity. Given a scalar value x, Fourier Number Embedding (FoNE) maps it to a high-dimensional feature vector $\phi ( x ) \in \mathbb { R } ^ { d _ { f o u r i e r } }$ using periodic activation functions, effectively capturing both magnitude and fractional details.

Crucially, the raw Fourier features $\phi ( x )$ exist in a mathematical manifold distinct from the pretrained LLM’s semantic space. Direct usage of these features creates a distributional mismatch. To bridge this gap, we introduce a learnable Linear Projection $W _ { P } \in \mathbb { R } ^ { d _ { m o d e l } \times d _ { f o u r i e r } }$ . This projection serves a dual purpose: (i) dimensionality matching, and (ii) manifold alignment, transforming the raw mathematical features into a representation e<sub>num</sub> compatible with the backbone’s latent space:

$$
e _ { n u m } = W _ { P } \cdot \phi ( x ) .\tag{1}
$$

Unlike naive approaches that employ digit-wise injection, which often fail to capture the compositional structure of numbers, our approach generates a single unified numerical embedding for the entire number entity. This ensures that the global numerical value is preserved and injected strictly at the terminal token, enabling a compact and contextually grounded integration as detailed in the subsequent gating phase.

## 3.4 Semantic Probing and Adaptive Integration

As shown in Figure 4, ANI uses a context-aware gating mechanism to determine the selective activation of numerical features at the last sub-token.

Semantic Probing (Gating Path) To achieve a seamless integration that preserves linguistic context while enhancing arithmetic precision, we address two design challenges: (i) Target Locus (Where), the optimal aggregation point to minimize signal redundancy, and (ii) Functional Selection (How), the mechanism to discern the necessity of mathematical intervention.

First, we determine where to inject the numeri cal features by resolving the granularity mismatch inherent in subword tokenization. While a numeri cal entity possesses a unified mathematical identity, it is often fragmented into a sequence of sub-tokens $S = \{ t _ { 1 } , \ldots , t _ { L } \}$ in the textual space. To bridge this gap, we consider two integration strategies: (i) Digit-wise Injection: This approach broadcasts the modulated numerical embedding to every sub token $t _ { i } \in S$ . We argue that even with adaptive modulation, this strategy remains suboptimal as it induces L-fold signal interference. Such repetitive intervention leads to feature entanglement, where numerical noise cumulatively disrupts the estab lished representation space of the backbone. This dense redundancy effectively overwhelms the lo cal semantic context with redundant mathematical topology, negating the benefits of adaptive injection. (ii) Terminal Token Injection (Ours): Instead of broadcasting, we leverage the causal na ture of Decoder-only Transformers (Vaswani et al., 2017). Although attention weights can theoretically distribute across all preceding tokens, the last token $t _ { L }$ functionally serves as an information bottleneck that aggregates this context. Consequently, we strictly inject $e _ { n u m }$ into the hidden state $h _ { t _ { L } } ^ { ( l ) }$ . This sparse intervention ensures that the holistic numerical topology modulates the representation precisely once, effectively safeguarding the model’s linguistic integrity while minimizing the risk of degradation associated with dense feature injections. The empirical superiority of this sparse injection approach over the digit-wise alternative is further substantiated by our ablation results, as detailed in Table 3.

Next, we address how to derive a discrete gating decision through context-aware semantic probing. To ensure that numerical enhancement does not compromise pre-established fluency, our contextaware gating network operates as a continuous scaler during training, which is explicitly regularized to converge into a binary selector. By treating the contextual hidden state $\mathbf { h } _ { t _ { L } } ^ { ( l ) }$ as the subject that evaluates the mathematical necessity of the numerical object ${ \bf e } _ { n u m }$ , ANI selectively filters out potential interference. Our gating MLP computes a raw logit $z _ { n u m }$ mapped via a standard sigmoid function:

$$
z _ { n u m } = M L P _ { g a t e } ( [ \mathbf { h } _ { t _ { L } } ^ { ( l ) } ; \mathbf { e } _ { n u m } ] ) ,\tag{2}
$$

$$
\alpha _ { n u m } = \sigma ( z _ { n u m } ) ,\tag{3}
$$

where $\alpha _ { n u m } \in ( 0 , 1 )$ dictates the injection ratio. To induce discreteness and push the continuous scale toward a deterministic ’on/off’ switch without breaking differentiability, we introduce a binarization penalty to the standard Next-Token Prediction (NTP) objective:

$$
\mathcal { L } = \mathcal { L } _ { \mathrm { C E } } + \lambda \mathbb { E } [ \alpha _ { n u m } ( 1 - \alpha _ { n u m } ) ] ,\tag{4}
$$

where $\lambda ~ ( \mathrm { { e . g . } , ~ 0 . 1 } )$ controls the regularization strength. This penalty is maximized at $\alpha _ { n u m } = 0 . 5$ and minimized as $\alpha _ { n u m }  0$ or 1. During inference, to guarantee absolute zero-interference for nominal identifiers, we bypass the continuous scale and apply a hard threshold:

$$
\alpha _ { n u m } = \mathbb { I } ( \sigma ( z _ { n u m } ) > 0 . 5 ) .\tag{5}
$$

Adaptive Integration. The final integration of numerical features is realized through an additive augmentation, where the gate $\alpha _ { n u m }$ dictates the operational regime of the model:

$$
\tilde { \mathbf { h } } _ { t _ { L } } ^ { ( l ) } = \underbrace { \mathbf { h } _ { t _ { L } } ^ { ( l ) } } _ { \mathrm { B a s e S e m a n t i c } } + \underbrace { \boldsymbol { \alpha } _ { n u m } \cdot \mathbf { e } _ { n u m } } _ { \mathrm { C o n d i t i o n a l E n h a n c e m e n t } }\tag{6}
$$

As defined by the gating decision, this formulation enables two distinct functional modes: (i) Active Injection $( \alpha _ { n u m } = 1 )$ , where the numerical path is activated for quantitative operands to reinforce arithmetic reasoning, and (ii) Neutral Preservation $( \alpha _ { n u m } = 0 )$ , where the gate remains closed for nominal identifiers to ensure zero-interference.

In contrast to traditional mixing or substitution approaches—where enhancing one feature necessitates the degradation of another in a zero-sum trade-off—our additive scheme maintains the base semantic vector $\mathbf { h } _ { t _ { L } } ^ { ( l ) }$ as an immutable foundation. This architectural choice provides the flexibility to inject precise mathematical topology as a calibrative refinement while filtering out potential numerical noise. Crucially, all non-numerical tokens bypass this module entirely, ensuring that the pre-trained linguistic manifold remains uncompromised for the vast majority of the sequence. This selective assimilation allows ANI to maximize mathematical precision only when functionally necessary, effectively mitigating the risk of semantic erasure and catastrophic forgetting.

## 3.5 Training Strategy and Objective

To integrate the ANI module while preserving the pre-trained linguistic manifold, we utilize Quantized Low-Rank Adaptation (QLoRA) (Dettmers et al., 2023) to update only the lightweight adapters and the ANI module. Rather than relying on stochastic sampling estimators, the model is optimized using the standard Next-Token Prediction (NTP) objective augmented with our binarization penalty $\lambda \mathbb { E } [ \alpha _ { n u m } ( 1 - \alpha _ { n u m } ) ]$ . This explicit regularization ensures stable convergence from a continuous scaling state toward a deterministic hardselection regime. This unified protocol allows ANI to optimally utilize numerical topology while strictly safeguarding the model’s general-purpose capabilities. The detailed formulation and the full execution procedure are deferred to Appendix Algorithm 1.

## 4 Experiments

In this section, we empirically evaluate ANI’s capability to enhance mathematical reasoning while preserving general-purpose linguistic knowledge. Detailed setup is found in the Appendix C.

## 4.1 Experimental Setup

Datasets and Metrics. We evaluate ANI on two core dimensions: (1) Numerical Reasoning via MATH (Hendrycks et al., 2021b) and GSM8K (Cobbe et al., 2021), and (2) General Capabilities via MMLU (Hendrycks et al., 2021a) to monitor potential semantic erasure. For all tasks, we report Accuracy (Acc) via a Chain-of-Thought protocol to ensure rigorous logical reasoning. For full reproducibility, our complete prompt templates are provided in Appendix C.5.

Baselines. To evaluate the efficacy of our contextaware integration, we compare ANI against three categories of state-of-the-art methodologies: (1) Text-based models, including the official Reference and Standard SFT; (2) Input-level Integration, which modifies representations at the input stage via context-agnostic substitution—such as xVal (Golkar et al., 2023) and FoNE (Zhou et al., 2025)—or additive fusion (Digit-wise); and (3) Incontext methods such as NumeroLogic (Schwartz et al., 2024), which inject structural numerical metadata directly into the input sequence.

Implementation Details. We evaluate the Qwen3 (Yang et al., 2025) (4B, 8B, 32B) family as our primary backbones, with further validation on Llama-3.1 (Grattafiori et al., 2024) and Mistral. To train the gating mechanism’s ability to distinguish between nominal identifiers and quantitative operands, we construct a composite diagnostic dataset by pairing OpenMathInstruct-2 (Toshniwal et al., 2025) (mathematical context) with Magpie (Xu et al., 2025) (linguistic identifiers). All models are optimized via QLoRA.

<table><tr><td rowspan="2">MODEL</td><td colspan="2">MATHEMATICAL REASONING</td><td>GENERAL</td></tr><tr><td>MATH (ACC)</td><td>GSM8K (ACC)</td><td>MMLU (ACC)</td></tr><tr><td colspan="4">Reference Models (Official)</td></tr><tr><td>QWEN3-8B</td><td>49.28</td><td>89.84</td><td>73.75</td></tr><tr><td colspan="4">Standard Fine-tuning</td></tr><tr><td>QWEN3-8B</td><td>55.18</td><td>87.79</td><td>72.24</td></tr><tr><td colspan="4">Controlled Experiments (Methods)</td></tr><tr><td>DIGIT-WISE</td><td>56.44</td><td>88.17</td><td>72.56</td></tr><tr><td>XVAL</td><td>58.32</td><td>90.83</td><td>72.50</td></tr><tr><td>FoNE</td><td>55.08</td><td>89.61</td><td>72.30</td></tr><tr><td>NUMEROLOGIC</td><td>47.56</td><td>42.99</td><td>72.45</td></tr><tr><td>ANI (OURS)</td><td>58.76</td><td>90.14</td><td>72.67</td></tr></table>

Table 1: Main Results on Qwen3-8B. We compare our proposed ANI method with an official baseline and controlled baselines. Bold indicates the best performance among models trained on the same dataset.

## 4.2 Main Results

In this section, we empirically evaluate ANI by investigating its impact on specialized reasoning and general linguistic stability. Specifically, we address (RQ1) whether adaptive injection improves complex arithmetic performance and (RQ2) whether context-aware integration effectively mitigates the trade-off between specialized precision and general linguistic integrity.

RQ1: Impact on Mathematical Reasoning. To evaluate the arithmetic improvement enabled by ANI, we analyze the performance results summarized in Table 1. The results demonstrate that incorporating numerical embeddings significantly facilitates complex reasoning by providing the necessary mathematical topology.

On the challenging MATH benchmark, ANI achieves an accuracy of 58.76%, representing a substantial improvement of nearly 9.5 points over the official pre-trained reference model. Notably, ANI surpasses not only the Standard SFT baseline but also existing context-agnostic strategies such as xVal and FoNE. Furthermore, ANI’s advantage emerges on highly complex tasks. As detailed in our difficulty-wise analysis (Appendix D.1), it demonstrates robust gains on the hardest problems (MATH Levels 4 and 5) over input-level baselines.

![](images/c6436eff0d78d2334f9895bbb334a3a3154ce07e085fc5bc26b289e0619809ce.jpg)  
Figure 5: Layer Sensitivity Analysis on MATH. Performance is measured by relative accuracy (Rel. Acc.) on the MATH benchmark. The linear probe reaches a peak accuracy of 93% at Layer 24.

This confirms that our adaptive integration is highly effective for high-level arithmetic where indiscriminate substitution falters. While xVal shows a marginal lead on GSM8K, ANI remains highly competitive at 90.14%, effectively outperforming the Reference model and significantly exceeding incontext methods like NumeroLogic. These findings indicate that ANI provides precise numerical features that reinforce the model’s arithmetic reasoning without obstructing its fundamental reasoning path.

RQ2: Mitigation of General Language Understanding. To investigate whether ANI successfully preserves pre-trained linguistic knowledge, we monitor performance on the MMLU benchmark, with the comparative results reported in Table 1. The results highlight a critical advantage of our gating architecture: while Standard SFT suffers a notable degradation from 73.75% to 72.24%, ANI successfully recovers a significant portion of this loss, maintaining a robust 72.67% accuracy. In comparison, context-agnostic inputlevel baselines, such as FoNE and xVal, suffer from severe catastrophic forgetting. This performance delta confirms that our Gating Network effectively distinguishes between computational and non-computational contexts. By suppressing unnecessary numerical noise, ANI effectively mitigates the semantic drift typical of these baselines, thereby safeguarding the model’s original representation manifold.

## 4.3 Analysis of Adaptive Injection

In this section, we conduct a series of diagnostic experiments to interpret the internal mechanism of ANI, validating our hypothesis regarding optimal feature integration and tool-routing synergy.

<table><tr><td>Method</td><td>2-digit</td><td>4-digit</td><td>6-digit</td><td>8-digit</td><td>10-digit</td><td>12-digit</td></tr><tr><td>Pre-trained (Qwen3-8B)</td><td>75.0</td><td>48.0</td><td>6.0</td><td>0.0</td><td>0.0</td><td>0.0</td></tr><tr><td>Baseline (SFT)</td><td>100.0</td><td>95.0</td><td>88.0</td><td>73.0</td><td>44.0</td><td>50.0</td></tr><tr><td>ANI (Ours)</td><td>100.0</td><td>98.0</td><td>88.0</td><td>67.0</td><td>68.0</td><td>82.0</td></tr></table>

Table 2: N-Digit Arithmetic Scaling. Exact match accuracy (%) across varying lengths of numerical sequences. The results demonstrate the catastrophic calculation failures of standard subword tokenization on longer sequences (e.g., 6+ digits) and highlight ANI’s effectiveness as a structural stabilizer.

Robustness to Token Fragmentation. While LLMs possess a latent capacity to understand numerical semantics, numerical injection specifically addresses the critical functional bottleneck of information loss caused by token fragmentation in long numerical sequences. To empirically validate the necessity of explicit numerical injection, we conducted an N-Digit Arithmetic Scaling experiment using 100 evaluation problems for each digit length to measure exact match accuracy.

As shown in Table 2, Qwen3-8B completely collapses starting from just 6 digits (6.0%), proving that inherent LLM representations are highly fragile for longer inputs. Furthermore, even though standard SFT is trained on the same dataset, it falls significantly behind ANI as numbers extend to 10- 12 digits (e.g., SFT’s 50.0% vs. ANI’s 82.0% at 12 digits) due to cumulative information loss from token fragmentation. This confirms that by injecting a mathematically unified, single-token topology, ANI acts as a structural stabilizer that successfully bypasses this bottleneck.

Optimal Injection Depth and Contextual Maturity. To identify the most effective stage for numerical injection, we analyze the model’s internal capability to disambiguate numerical roles across its layers. Figure 5 shows that the ability to distinguish between nominal identifiers and quantitative operands exhibits a distinct progression, reaching its absolute peak of 93% at Layer 24. This peak indicates a state of maximum contextual maturity, where the hidden representations have fully integrated the surrounding linguistic context to resolve the functional ambiguity of numerical tokens.

The impact of this contextual maturity is directly reflected in downstream reasoning performance. Our layer sensitivity analysis on the MATH benchmark shows a strong correlation with the linear probe results, where injecting the ANI module at Layer 24 achieves the highest accuracy of 58.76%. Conversely, the early probing peak at Layer 3 fails to translate into MATH improvements, as it relies on surface-level lexical cues rather than deep contextual synthesis. This confirms our hypothesis that numerical injection is most effective at the depth where the model has attained sufficient contextual maturity to interpret the role of numbers.

![](images/1d02080bf71e28e46e9a87c48feee40560d72f1ce52abf8c74bc519db2f1d1ab.jpg)  
(a) Gating Selection  
(b) Alpha by Subject  
Figure 6: Interpretability Analysis of the Adaptive Gating Mechanism. (a) Selection ratio across benchmarks (y-axis: proportion). (b) Selection ratio across MMLU subjects (x-axis: proportion of α = 1).

Gating Behavior Analysis. To interpret the internal decision-making process of ANI, we analyze the gating network’s activation patterns across diverse reasoning tasks. As shown in Figure 6(a), the mechanism exhibits task-aware selectivity: for the calculation-heavy MATH benchmark, it activates numerical embeddings (α = 1) in 48% of instances, whereas this ratio drops to 24% for the predominantly linguistic MMLU. The subject-level granularity in Figure 6(b) reveals a distinct semantic divide. Within the MMLU benchmark, the gating network selects the numerical path (α = 1) significantly more often (over 35% of cases) for STEM disciplines like Econometrics and Chemistry, while the selection ratio remains near-zero for Humanities such as International Law. Notably, even within STEM, theoretical subjects like Conceptual Physics show a lower activation frequency compared to calculation-intensive ones. This finegrained discrimination confirms that ANI understands the functional necessity of numbers, ensuring that arithmetic reinforcement is triggered only when the semantic context demands it.

Impact of the Gating Mechanism. To dissect ANI’s architectural components, we evaluate the necessity of context-aware gating and our sparse injection strategy (Table 3) against two baselines: (i) Static Injection (no gating) and (ii) Digit-wise Injection (per-token gating). Without gating, Static Injection suffers a severe collapse in reasoning (8.60% MATH) and general capabilities (71.41%

<table><tr><td>Method</td><td>MATH</td><td>GSM8K</td><td>MMLU</td></tr><tr><td>Static Injection (No Gating)</td><td>8.60</td><td>62.70</td><td>71.41</td></tr><tr><td>Digit-wise Injection</td><td>58.70</td><td>90.37</td><td>72.29</td></tr><tr><td>ANI (Ours)</td><td>58.76</td><td>90.14</td><td>72.67</td></tr></table>

Table 3: Ablation Study. Impact of the gating mechanism and injection locus. Static Injection replaces tokens without gating, while Digit-wise Injection applies the gate to every sub-word digit token.

MMLU), confirming that forcing numerical topology without context acts as disruptive noise. Furthermore, while Digit-wise Injection achieves comparable arithmetic performance, it remains suboptimal in preserving linguistic integrity (72.29% MMLU). This indicates that broadcasting embeddings to every sub-token induces redundant signal interference that overwhelms the local semantic context. In contrast, ANI’s sparse intervention at the terminal token bottleneck successfully maximizes mathematical precision while effectively safeguarding the model’s linguistic manifold.

Gating Accuracy and Tool-Routing Synergy. To evaluate the utility of our gating mechanism, we analyze its behavior across both real-world linguistic contexts and complex agentic workflows. First, on the FiNER-139 dataset (Loukas et al., 2022), which provides explicit nominal ground truth (e.g., "Debt Instrument Maturity Date"), our gating network achieves a 75.8% precision in bypassing these nominal identifiers. This provides strong evidence that ANI effectively safeguards the model’s linguistic manifold from semantic erasure. Furthermore, we evaluate its tool-use applicability on a highly deceptive synthetic dataset (detailed in Appendix D.2) that mixes nominal traps with quantitative operands. While the baseline pre-trained model avoids passing nominal traps into tool arguments in only 69.6% of cases, ANI achieves a 92.9% success rate by precisely suppressing the nominal embeddings $( \alpha _ { n u m } = 0 )$ . Together, these results confirm that ANI functions not merely as an arithmetic booster, but as a stable foundational architecture for reliable tool utilization.

## 4.4 Generalization and Scalability

To verify whether ANI represents a universally applicable enhancement, we evaluate its performance across diverse architectures and parameter scales, as summarized in Table 4.

Architectural Robustness. ANI demonstrates robust generalization across the Llama-3.1 and Mistral families. Crucially, the input-level baseline xVal routinely regresses below the standard SFT baseline across these diverse architectures. This regression clearly highlights the instability of context-agnostic substitution. In contrast, ANI effectively avoids this cascading degradation. While ANI achieves a 2.0% gain on GSM8K for Mistral-7B, the slight regression in Mistral’s MATH suggests that late-stage injection at layer 30 (of 32) lacks sufficient residual depth for complex reasoning. Nevertheless, the overall trend confirms the framework’s cross-architecture portability. Optimal layers are detailed in Appendix C.6.

Scaling across Model Sizes. The effectiveness of numerical topology injection is further validated across the Qwen family, ranging from 4B to 32B parameters. Notably, on the compact Qwen3-4B, ANI achieves accuracy improvements of 1.3 points on MATH and 2.4 points on GSM8K over the SFT baseline, demonstrating that even capacityconstrained models benefit from explicitly injected numerical representations. Furthermore, while standard fine-tuning and xVal suffer significant performance degradation on the 4B’s GSM8K task compared to the pre-trained model, ANI effectively mitigates this loss, preserving robust reasoning capabilities. This consistent advantage extends to the larger Qwen3-32B model, where ANI continues to deliver the highest accuracy (63.1% on MATH and 93.5% on GSM8K), outperforming both the SFT baseline and xVal. This positive trajectory confirms that ANI provides reliable reasoning reinforcement as the backbone’s inherent capacity scales.

## 5 Conclusion

In this work, we addressed the fundamental tension between discrete subword tokenization and precise numerical reasoning in Large Language Models. We introduced ANI (Adaptive Numerical Injection), a hybrid framework that governs the selective activation of specialized numerical embeddings based on the linguistic context. By employing a context-aware gating mechanism, ANI effectively resolves the functional ambiguity of numbers, selectively enriching quantitative operands with mathematical topology while safeguarding nominal identifiers to prevent semantic erasure.

Our empirical results on the Qwen3-8B and various model families demonstrate that ANI significantly bolsters arithmetic reasoning on complex benchmarks while maintaining the model’s pretrained linguistic manifold. These findings confirm that the adaptive integration of numerical information is a key factor in unifying arithmetic precision with semantic fluency in LLMs.

<table><tr><td>Model Family / Size</td><td>Method</td><td>MATH</td><td>GSM8K</td></tr><tr><td rowspan="4">Llama-3.1-8B</td><td>Pre-trained</td><td>20.5</td><td>55.3</td></tr><tr><td>Baseline (SFT)</td><td>26.4</td><td>63.5</td></tr><tr><td>XVal</td><td>21.2</td><td>59.5</td></tr><tr><td>ANI (Ours)</td><td>27.2</td><td>63.5</td></tr><tr><td rowspan="4">Mistral-7B-v0.3</td><td>Pre-trained</td><td>11.4</td><td>37.5</td></tr><tr><td>Baseline (SFT)</td><td>16.6</td><td>52.4</td></tr><tr><td>XVal</td><td>16.2</td><td>51.0</td></tr><tr><td>ANI (Ours)</td><td>15.0</td><td>54.4</td></tr><tr><td rowspan="4">Qwen3-4B</td><td>Pre-trained</td><td>47.0</td><td>87.4</td></tr><tr><td>Baseline (SFT)</td><td>48.2</td><td>82.8</td></tr><tr><td>XVal</td><td>47.2</td><td>82.5</td></tr><tr><td>ANI (Ours)</td><td>49.5</td><td>85.2</td></tr><tr><td rowspan="4">Qwen3-32B</td><td>Pre-trained</td><td>60.0</td><td>91.4</td></tr><tr><td>Baseline (SFT)</td><td>62.5</td><td>92.0</td></tr><tr><td>XVal</td><td>61.6</td><td>89.2</td></tr><tr><td>ANI (Ours)</td><td>63.1</td><td>93.5</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

Table 4: Scalability and Generalization Results. Comparison of mathematical reasoning performance across various distinct architectures and model sizes.

## 6 Limitations

Despite strong empirical gains, several aspects of our framework remain open for further refinement.

First, while ANI is designed to be modelagnostic, its current implementation inherits the representational resolution of the underlying numerical encoder. Although our present configuration of 10 integer and 6 fractional digits is sufficient for most benchmarks, it may not generalize to specialized domains, such as science, requiring extreme high precision.

Second, our method injects numerical features at a layer identified via linear probing as the point of peak contextual maturity. While we observe this peak to be stable within model families, developing an automated, architecture-agnostic protocol for identifying this layer would further improve portability across emerging LLM architectures.

Finally, ANI introduces a lightweight gating and projection module during the forward pass. While negligible relative to the backbone model’s compute footprint, further optimization may be necessary for extreme cases of highly latency-sensitive deployments.

## Acknowledgments

This work was supported by Institute of Information & communications Technology Planning & Evaluation (IITP) grant funded by the Korea government (MSIT) (No. 2022-0-00077/RS-2022- II220077, AI Technology Development for Commonsense Extraction, Reasoning, and Inference from Heterogeneous Data).

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Wenhu Chen, Xueguang Ma, Xinyi Wang, and William W Cohen. 2022. Program of thoughts prompting: Disentangling computation from reasoning for numerical reasoning tasks. arXiv preprint arXiv:2211.12588.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, Mark Chen, Heewoo Jun, Lukasz Kaiser, Matthias Plappert, Jerry Tworek, Jacob Hilton, Reiichiro Nakano, Christopher Hesse, and John Schulman. 2021. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. 2023. Qlora: Efficient finetuning of quantized llms. Advances in neural information processing systems, 36:10088–10115.

Nouha Dziri, Ximing Lu, Melanie Sclar, Xiang Lorraine Li, Liwei Jiang, Bill Yuchen Lin, Sean Welleck, Peter West, Chandra Bhagavatula, Ronan Le Bras, and 1 others. 2023. Faith and fate: Limits of transformers on compositionality. Advances in neural information processing systems, 36:70293–70332.

Luyu Gao, Aman Madaan, Shuyan Zhou, Uri Alon, Pengfei Liu, Yiming Yang, Jamie Callan, and Graham Neubig. 2023. Pal: Program-aided language models. In International conference on machine learning, pages 10764–10799. PMLR.

Siavash Golkar, Mariel Pettee, Michael Eickenberg, Alberto Bietti, Miles Cranmer, Geraud Krawezik, Francois Lanusse, Michael McCabe, Ruben Ohana, Liam Parker, and 1 others. 2023. xval: A continuous number encoding for large language models. arXiv preprint arXiv:2310.02989.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2021a. Measuring massive multitask language understanding. Proceedings ofthe International Conference on Learning Representations (ICLR).

Dan Hendrycks, Collin Burns, Saurav Kadavath, Akul Arora, Steven Basart, Eric Tang, Dawn Song, and Jacob Steinhardt. 2021b. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Marek Kadlcík, Michal Štefánik, Timothee Mickus, ˇ Josef Kuchaˇr, and Michal Spiegel. 2025. Pre-trained language models learn remarkably accurate representations of numbers. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 26693–26702.

Jared Kaplan, Sam McCandlish, Tom Henighan, Tom B Brown, Benjamin Chess, Rewon Child, Scott Gray, Alec Radford, Jeffrey Wu, and Dario Amodei. 2020. Scaling laws for neural language models. arXiv preprint arXiv:2001.08361.

Lefteris Loukas, Manos Fergadiotis, Ilias Chalkidis, Eirini Spyropoulou, Prodromos Malakasiotis, Ion Androutsopoulos, and Georgios Paliouras. 2022. Finer: Financial numeric entity recognition for xbrl tagging. In Proceedings of the 60th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 4419–4431.

Rodrigo Nogueira, Zhiying Jiang, and Jimmy Lin. 2021. Investigating the limitations of transformers with simple arithmetic tasks. arXiv preprint arXiv:2102.13019.

Eli Schwartz, Leshem Choshen, Joseph Shtok, Sivan Doveh, Leonid Karlinsky, and Assaf Arbelle. 2024. Numerologic: Number encoding for enhanced llms numerical reasoning. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 206–212.

Rico Sennrich, Barry Haddow, and Alexandra Birch. 2016. Neural machine translation of rare words with subword units. In Proceedings of the 54th annual meeting ofthe associationfor computational linguistics (volume 1: long papers), pages 1715–1725.

Yao Tang, Li Dong, Yaru Hao, Qingxiu Dong, Furu Wei, and Jiatao Gu. 2026. Multiplex thinking: Reasoning via token-wise branch-and-merge. arXiv preprint arXiv:2601.08808.

Shubham Toshniwal, Wei Du, Ivan Moshkov, Branislav Kisacanin, Alexan Ayrapetyan, and Igor Gitman. 2025. Openmathinstruct-2: Accelerating ai for math

with massive open-source instruction data. In International Conference on Learning Representations, volume 2025, pages 19243–19275.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. 2017. Attention is all you need. Advances in neural information processing systems, 30.

Eric Wallace, Yizhong Wang, Sujian Li, Sameer Singh, and Matt Gardner. 2019. Do nlp models know numbers? probing numeracy in embeddings. In Proceedings ofthe 2019 Conference on Empirical Methods in Natural Language Processing and the 9th International Joint Conference on Natural Language Pro cessing (EMNLP-IJCNLP), pages 5307–5315.

Jason Wei, Yi Tay, Rishi Bommasani, Colin Raffel, Barret Zoph, Sebastian Borgeaud, Dani Yogatama, Maarten Bosma, Denny Zhou, Donald Metzler, and 1 others. 2022a. Emergent abilities of large language models. arXiv preprint arXiv:2206.07682.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed Chi, Quoc V Le, and Denny Zhou. 2022b. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems, volume 35, pages 24824–24837.

Zhangchen Xu, Fengqing Jiang, Luyao Niu, Yuntian Deng, Radha Poovendran, Yejin Choi, and Bill Yuchen Lin. 2025. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In International Conference on Learning Representations, volume 2025, pages 76346–76382.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Tianyi Zhou, Deqing Fu, Mahdi Soltanolkotabi, Robin Jia, and Vatsal Sharan. 2025. Fone: Precise singletoken number embeddings via fourier features. arXiv preprint arXiv:2502.09741.

## A Mathematical Derivation of the Binarization Penalty

The core innovation of ANI lies in its ability to perform discrete functional selection between nominal identifiers and quantitative operands while remaining end-to-end trainable. Since traditional discrete gating based on a Bernoulli distribution is non-differentiable, we utilize a continuous sigmoid relaxation coupled with an explicit binarization penalty.

## A.1 Continuous Relaxation and Binarization

Our gating MLP computes a continuous probability based on the target logit $z _ { n u m }$ :

$$
\alpha _ { n u m } = \sigma \big ( M L P _ { g a t e } ( [ \mathbf { h } _ { t _ { L } } ^ { ( l ) } ; \mathbf { e } _ { n u m } ] ) \big ) \in ( 0 , 1 )\tag{7}
$$

To induce discreteness and force the continuous scale $\alpha _ { n u m }$ toward a deterministic 0 or 1, we introduce a regularization term to the objective function:

$$
\mathcal { L } _ { g a t e } = \lambda \mathbb { E } [ \alpha _ { n u m } ( 1 - \alpha _ { n u m } ) ]\tag{8}
$$

This parabolic penalty reaches its maximum at $\alpha _ { n u m } = 0 . 5$ and minimizes at the bounds 0 and 1. By incorporating this penalty into the standard Next-Token Prediction loss, the network is effectively pushed away from ambiguous intermediate states during training, acting as a soft surrogate for discrete step functions.

## A.2 Deterministic Inference

During inference, we require absolute zerointerference for nominal identifiers to prevent semantic erasure. We achieve this by applying a hard threshold directly to the continuous activation without stochastic noise:

$$
\alpha _ { n u m } = \mathbb { I } ( \sigma ( z _ { n u m } ) > 0 . 5 )\tag{9}
$$

This formulation allows ANI to maintain a strict, deterministic ’on/off’ switching regime during inference while leveraging stable, continuous gradients to optimize the Gating Network during training, bypassing the instability commonly associated with stochastic estimators.

## B Procedural Overview of ANI

Algorithm 1 details the plug-and-play intervention of the ANI module within the intermediate layers of a backbone LLM. To minimize signal redundancy and potential feature entanglement, the injection is strictly targeted at the terminal sub-token index $( i d x _ { k } )$ of each numerical entity, which serves as an information bottleneck in decoder-only architectures.

Algorithm 1 Forward Pass of ANI   
Input: Token sequence $\overline { { S = \{ t _ { 1 } , \ldots , t _ { N } \} } }$ , Set of   
numerical entities $V = \{ ( v _ { k } , \mathrm { i d x } _ { k } ) \} _ { k = 1 } ^ { M }$ , Target   
Layer l, Mode is\_training   
Parameters: $\mathbf { W } _ { P }$ (Linear Projection), $\theta _ { \mathrm { g a t e } }$ (Gat  
ing MLP), FoNE periods P   
1: $\mathbf { H } ^ { ( l ) } \gets \mathrm { L L M } _ { 1 : l } ( S )$ // 1. Contextual   
Representation Extraction   
2: for each $( v _ { k } , \operatorname { i d x } _ { k } ) \in V$ do   
3: i ← idx<sub>k</sub> // Identify terminal sub-token   
index   
4: $\phi ( v _ { k } ) \gets \mathrm { F o N E } ( v _ { k } ; P )$ // 2. Topology   
Encoding   
5: $\mathbf { e } _ { n u m }  \mathbf { W } _ { P } \cdot \phi ( v _ { k } )$ // Manifold   
Alignment   
6: $z _ { n u m } \gets M L P _ { \mathrm { g a t e } } ( [ \mathbf { h } _ { i } ^ { ( l ) } ; \mathbf { e } _ { n u m } ] ; \theta _ { \mathrm { g a t e } } )$ //   
3. Semantic Probing   
7: if is\_training then   
8: $\alpha _ { n u m }  \sigma ( z _ { n u m } )$ // Continuous scale   
for binarization penalty   
9: else   
10: $\alpha _ { n u m } \gets \mathbb { I } ( \sigma ( z _ { n u m } ) > 0 . 5 )$ //   
Deterministic hard selection   
11: end if   
12: $\mathbf { h } _ { i } ^ { ( l ) }  \mathbf { h } _ { i } ^ { ( l ) } + \alpha _ { n u m } \cdot \mathbf { e } _ { n u m }$ // 4.   
Adaptive Integration   
13: end for   
14: Output $ \mathrm { L L M } _ { l + 1 : L } ( \mathbf { H } ^ { ( l ) } )$ // Residual   
Layers Processing   
15: Return Output

1. Topology Encoding (Lines 4-5): Raw scalars are mapped to a high-dimensional mathematical manifold via FoNE and subsequently aligned to the model’s latent space using a learnable Linear Projection $( W _ { P } )$

2. Context-Aware Gating (Lines 6-11): To determine arithmetic necessity, the MLP outputs a raw logit that is mapped via a standard sigmoid function. During training, the gate acts as a continuous scaler $( \alpha _ { n u m } \in ( 0 , 1 ) )$ , allowing end-to-end optimization guided by an explicit binarization penalty. During inference, it applies a strict deterministic threshold $( \alpha _ { n u m } \in \{ 0 , 1 \} )$ to ensure absolute zerointerference for nominal identifiers.

3. Efficiency: Tokens not identified as numerical bypass the module entirely, ensuring zero interference with the model’s pre-trained linguistic knowledge.

## C Implementation Details of ANI

## C.1 Experimental setup

We run our experiments on a machine equipped with Intel Xeon(R) Gold 6226R CPUs and Nvidia RTX 3090/A6000 GPUs. We implement ANI using Python 3.10, PyTorch 2.1, Hugging Face Transformers, and vLLM.

## C.2 Datasets and Evaluation Protocols

To evaluate the numerical reasoning and general linguistic stability of ANI, we employ specific few-shot Chain-of-Thought (CoT) protocols across three primary benchmarks:

• MATH (4-shot CoT): To assess complex multi-step mathematical reasoning, we use a 4-shot prompt where the model is instructed to provide step-by-step solutions and encapsulate the final answer within a \boxed{} command.

• GSM8K (8-shot CoT): For grade-school word problems, an 8-shot CoT protocol is utilized. The model is required to provide the final numerical value following the #### delimiter to ensure consistent answer extraction.

• MMLU (5-shot): To monitor generalpurpose linguistic knowledge and detect potential semantic erasure, we evaluate the model across 57 subjects using a standard 5- shot setting. Evaluation is performed by comparing the log-probabilities of the candidate options (A, B, C, D).

## C.3 Diagnostic Dataset Construction

The Gating Network is trained on a composite diagnostic dataset designed to help the model distinguish between the functional roles of numbers (Nominal vs. Quantitative).

• Quantitative Samples: 10,000 samples are extracted from OpenMathInstruct-2, focusing on contexts where numerical values serve as operands for arithmetic operations. Crucially, the OpenMathInstruct-2 dataset has already undergone a rigorous LLM-based decontamination pipeline to remove any potential paraphrases of the MATH and GSM8K evaluation sets (Toshniwal et al., 2025). This inherently precludes any risk of data leakage into our diagnostic training phase.

• Nominal Samples: 10,000 samples are curated from the Magpie (Pro-MT-300K) dataset, representing linguistic contexts where numbers act as identifiers.

• Filtering Strategy: To ensure the purity of nominal samples, we apply a strict keywordbased filter. Any Magpie entry containing math-related terms (e.g., math, calculation, algebra) or programming-related terms (e.g., python, algorithm, sql) is excluded to prevent the leakage of quantitative reasoning signals into the nominal set.

## C.4 Baselines

• Reference: The official pre-trained Qwen3-8B model without any additional fine-tuning, serving as the lower-bound performance metric.

• SFT Baseline: A standard supervised finetuning approach that utilizes the same composite dataset but relies exclusively on vanilla subword tokenization without specialized numerical representations.

• Digit-wise (Additive): Implements an additive fusion scheme where numerical features are summed with textual embeddings at the input layer $( h _ { n u m }  h _ { n u m } + e _ { n u m } )$ . This baseline evaluates the impact of simple feature augmentation without the benefit of contextaware gating.

• xVal (Golkar et al., 2023) (Substitution): This method addresses numerical fragmentation by mapping scalar values to a continuous, single-token representation via multiplicative magnitude encoding in a learnable direction. It is designed to capture quantitative magnitude and continuity directly within the embedding space. However, because it relies on context-agnostic substitution (replacing all numbers with a generic [NUM] token) at the input layer, it often fails to preserve the semantic role of numbers when they function as nominal identifiers.

• FoNE (Zhou et al., 2025) (Substitution): This approach utilizes Fourier features to provide precise single-token number embeddings by representing residues in various modular groups. By mapping raw scalars through periodic sine and cosine activation functions, it effectively captures both magnitude and fine-grained fractional details in a high-dimensional feature vector. Similar to xVal, its global substitution strategy $( h _ { n u m } $ $e _ { n u m } )$ ensures arithmetic precision but triggers semantic erasure, stripping numbers of their linguistic identity in multi-modal contexts.

• NumeroLogic (Schwartz et al., 2024) (Incontext): A representation reformatting approach that enhances reasoning by prepending the digit count to each number (e.g., "2:42"), providing explicit place-value information directly in the input sequence. This structural encoding acts as a simplified Chain-of-Thought (CoT), prompting the model to reason about magnitude before generation. Unlike continuous methods, it operates entirely within the discrete textual space via text pre-processing, focusing on improving the model’s inherent interpretation of decimal strings.

## C.5 Evaluation Prompts

## MATH Evaluation Template

[System]

You are a helpful math reasoning assistant. Please reason step by step.

[User]

Solve the following math problem step-by-step. Put your final answer in \boxed{}.

Below are some examples:

Question:

Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?

My solution:

First, calculate the total cost of the bagels. Olivia bought 5 bagels at \$3 each. Cost = 5 \* 3 = 15. She started with \$23 and spent \$15. Money left = 23 - 15 = 8. The final answer is: \boxed{8}

## Question:

A pen and its ink refill together cost \$1.10. The pen costs \$1 more than the

ink refill. What is the cost of the pen in dollars?

## My solution:

Let p be the cost of the pen and i be the cost of the ink refill. We are given: 1. p + $\mathrm { i } = 1 . 1 0 , 2 . \mathrm { ~ p = i } + 1$ . Substitute equation 2 into equation 1: (i + 1) + i = 1.10 -> 2i + 1 = 1.10 -> 2i = 0.10 -> i = 0.05. Now find p: p = i + 1 = 0.05 + 1 = 1.05. The final answer is: \boxed{1.05}

## Question:

What is the value of f(f(1)) if f(x) = 2x + 3?

My solution:

First, we calculate f(1): f(1) = 2(1) + 3 = 2 + 3 = 5. Now substitute this into f(x): f(f(1)) = f(5) = 2(5) + 3 = 10 + 3 = 13. The final answer is: \boxed{13}

## Question:

How many ways can 3 people (Alice, Bob, Charlie) line up in a single file line?

## My solution:

This is a permutation problem. For the first spot, there are 3 choices. For the second, 2 remaining. For the last, 1 choice left. Total ways = 3 \* 2 \* 1 = 6. The final answer is: \boxed{6} Now, solve this problem:

Question: {Target Problem}

## GSM8K Evaluation Template

[System]

You are a helpful math assistant. Please reason step by step.

[User]

Solve the following math problem stepby-step. At the end of your response, provide the final answer in the format ’#### [value]’.

Below are some examples of math problems:

Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?

A: There are 15 trees originally. Then there were 21 trees after the Grove workers planted some more. So there must have been 21 - 15 = 6. The answer is 6.

Q: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?

A: There are 3 cars originally. 2 more cars arrive. 3 + 2 = 5. The answer is 5.

Q: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?

A: Leah had 32 chocolates and her sister had 42. That means there were 32 + 42 = 74 chocolates. 35 were eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.

Q: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?

A: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he gave to Denny must have been 20 - 12 = 8. The answer is 8.

Q: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now? A: He has 5 toys. As he got 2 from mom and 2 from dad, he got 2 + 2 = 4 more. 5 + 4 = 9. The answer is 9.

Q: There were 9 computers in the server room. Five more computers were installed each day for 4 days. How many computers are now in the server room? A: There were 9 computers. Then 5 more were installed for 4 days, which means 5 \* 4 = 20 computers were added. 9 + 20 = 29. The answer is 29.

Q: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?

A: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he had 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.

Q: Olivia has \$23. She bought five bagels for \$3 each. How much money does she have left?

A: She bought 5 bagels for \$3 each. This

means she spent 5 \* 3 = \$15. She had \$23 so she has 23 - 15 = 8. The answer is 8.   
Now, solve this problem:   
Q: {Target Question}   
A:

## MMLU Evaluation Template

[System]   
The following are multiple choice questions (with answers) about {subject}. [User]   
Question: {Few-shot Example 1}   
(A) {Option 1} (B) {Option 2} (C) {Option 3} (D) {Option 4}   
Answer: A   
Question: {Few-shot Example 5}   
(A) {Option 1} (B) {Option 2} (C) {Op  
tion 3} (D) {Option 4}   
Answer: C   
Question: {Target Question}   
(A) {Option 1} (B) {Option 2} (C) {Op  
tion 3} (D) {Option 4}   
Answer:

## C.6 Hyperparameters

To ensure a fair and rigorous evaluation of ANI, we maintain a consistent set of hyperparameters across diverse model architectures and scales. All models are evaluated with a maximum sequence length of 4,096 tokens. To ensure high numerical resolution, the integer and fractional lengths for numerical representations are fixed at 10 and 6 digits, respectively.

The Gating Network consists of a 2-layer MLP with a dropout rate of 0.1 to prevent overfitting during the specialized training phase. The optimal injection layer index (l) is determined based on the contextual maturity of each model. For Qwen3- 8B, Layer 24 is selected based on our diagnostic linear probing analysis. For other architectures, including Qwen3-4B, Qwen3-32B, Llama-3.1-8B, and Mistral-7B, the injection layers are set to 21, 37, 14, and 30, respectively, determined through similar probing heuristics or empirical validation.

All models are optimized using the QLoRA framework with a rank (r) of 64 and an alpha (α) of 128. We employ the AdamW optimizer with a constant learning rate of 5 × 10<sup>−5</sup>, utilizing a perdevice batch size of 1 and a gradient accumulation of 32 steps to ensure stable convergence.

![](images/0f716646e7f551b17915e74fbcbfe361e952c2dfad14faba4c92d260c485d23b.jpg)  
Figure 7: Model performance comparison with 95% CI. Bars represent the overall accuracy on the MATH benchmark, with error bars denoting the 95% confidence intervals derived from paired bootstrap resampling.

## D Additional Experiment Results

## D.1 Confidence Intervals and Difficulty-wise Analysis

To evaluate the statistical reliability of our empirical results and ensure that the performance gains are not artifacts of random variance, we employ paired bootstrap resampling on the evaluation outputs. Specifically, we draw 1, 000 bootstrap samples from the test sets of the MATH benchmark $( N = 5 , 0 0 0$ total; Levels 1-3: N = 2, 462, Levels 4-5: $N = 2 , 5 3 8 )$ and the MMLU benchmark $( N = 1 4 , 0 4 2 )$ . We then compute the accuracy delta (∆) between ANI and the respective baselines for each sample to determine the $2 . 5 ^ { t h }$ and $9 7 . 5 ^ { t h }$ percentiles as the empirical bounds.

First, we report the statistical reliability compared against all input-level baselines. As illustrated in Figure 7, the 95% confidence interval (CI) analysis across all input-level variations reveals that ANI consistently achieves the highest lower and upper empirical bounds compared to all other baselines, thereby demonstrating the stability of our approach.

Furthermore, when diving deeper into a granular, difficulty-wise analysis against xVal (Table 5), this statistical advantage becomes even more pronounced. On simpler segments (Levels 1, 2, and 3), the performance difference is marginal with an accuracy delta of −0.0049 and a paired bootstrap 95% CI that includes zero $( [ - 0 . 0 1 5 0 , + 0 . 0 0 5 7 ] )$ resulting in a non-significant p-value $( p = 0 . 8 3 0 3 )$ Conversely, in the most challenging intervals (Levels 4 and 5), ANI exhibits a distinct performance improvement, with an accuracy delta of +0.0134. Crucially, this high-difficulty confidence interval strictly excludes zero $\left( \left[ + 0 . 0 0 0 4 , + 0 . 0 2 6 8 \right] \right)$ , yielding a statistically significant p-value of less than 0.05 $( p = 0 . 0 2 4 1 )$ . This fine-grained evaluation confirms that ANI’s performance is both meaningful and effective in complex arithmetic reasoning compared to input-level baselines.

<table><tr><td>GROUP</td><td>ACC. DIFF</td><td>PAIRED BS 95% CI</td><td> $\mathbf { \overline { { B S \ / { \cal P } } } } \left( \mathbf { O N E - S I D E D } \right)$  1</td></tr><tr><td>LEVEL 1, 2, 3</td><td>-0.0049</td><td>[-0.0150,+0.0057]</td><td>0.8303</td></tr><tr><td>LEVEL 4, 5</td><td>+0.0134</td><td> $\left| + 0 . 0 0 0 4 , + 0 . 0 2 6 8 \right|$ </td><td>0.0241*</td></tr></table>

Table 5: Fine-grained bootstrap statistical analysis. Comparison of ANI’s accuracy gain relative to xVal on the MATH benchmark.

Finally, our bootstrap analysis on MMLU confirms that ANI (72.67%) significantly mitigates semantic drift compared to the standard fine-tuning baseline (72.24%). The accuracy delta of +0.0043 yields a paired bootstrap 95% CI that strictly excludes zero $( [ + 0 . 0 0 1 7 , + 0 . 0 0 6 9 ] , p \ = \ 0 . 0 0 0 8 ) .$ proving its robustness in preserving general linguistic capabilities alongside mathematical improvements.

## D.2 Synthetic Evaluation on Gating Accuracy and Agentic Tool-Routing

To rigorously evaluate our internal gating mechanism against ground-truth answers where specific numerical values are explicitly irrelevant, we constructed a synthetic evaluation set comprising 50 highly deceptive queries. These queries intentionally deploy nominal identifiers (e.g., “Employee $2 0 4 8 ^ { \prime \prime } )$ as traps alongside actual quantitative operands $( \mathrm { e . g . }$ , “earns $\$ 5,000$ and “10% bonus”). This setup tests whether the model’s gating can correctly align with the ground truth and bypass nominal traps, precisely routing only the actual arithmetic operands to external tools (e.g., executing calculator $( [ " 5 8 8 0 " , ~ " 8 . 1 8 " ] )$ .

The experimental results demonstrate the robustness of ANI’s self-filtering capability. While the baseline pre-trained model avoids passing nominal traps into tool arguments in only 69.6% of cases, ANI achieves a 92.9% success rate. This substantial reduction in argument routing errors provides concrete evidence that ANI functions as a robust foundational architecture, ensuring representation integrity and reliable tool utilization within complex agent workflows.

## E Use of AI Assistants

In this paper, we utilized Gemini, a large-scale language model developed by Google, to refine English expressions and enhance linguistic clarity.