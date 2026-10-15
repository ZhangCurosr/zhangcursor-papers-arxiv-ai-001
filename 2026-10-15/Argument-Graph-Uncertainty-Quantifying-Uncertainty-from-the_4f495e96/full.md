# Argument Graph Uncertainty: Quantifying Uncertainty from the Logical Structure of Reasoning Chains

Simon Baur   
Fraunhofer Heinrich-Hertz-Institut 10587 Berlin, Germany   
simon.baur@hhi.fraunhofer.de

Wojciech Samek Fraunhofer Heinrich-Hertz-Institut 10587 Berlin, Germany wojciech.samek@hhi.fraunhofer.de

Sumire Honda Fraunhofer Heinrich-Hertz-Institut 10587 Berlin, Germany sumire.honda@hhi.fraunhofer.de

Jackie Ma   
Fraunhofer Heinrich-Hertz-Institut 10587 Berlin, Germany   
jackie.ma@hhi.fraunhofer.de

## Abstract

Reliable uncertainty estimation is essential for deploying large language models in high-stakes reasoning tasks, yet most existing approaches rely on expensive sampling strategies or shallow token-level signals that ignore the internal logical structure of model outputs. We introduce Argument Graph Uncertainty, a post-hoc framework for multiple choice question tasks, that estimates model confidence directly from the logical structure of a single chain-of-thought reasoning trace. Our method uses either a lightweight instruction-tuned model or a simple deterministic rule to segment the reasoning chain into propositional units, then applies a NLI model to infer relational edges and construct two complementary argument graphs over these segments. The resulting edge scores are aggregated into a Dirichlet posterior over answer options, whose concentration yields calibrated uncertainty estimates reflecting both local step-to-step coherence and global logical consistency across the reasoning trace. Evaluated on GPQA, MMLU-Pro and TruthfulQA, our method consistently surpasses token-based and consistency-based uncertainty baselines across multiple reasoning models. We further demonstrate through extensive combination analyses that argument graph uncertainty captures an orthogonal signal to existing approaches, making it a complementary component in uncertainty ensembles. Crucially, our framework requires no additional calls to the evaluated model and relies entirely on small, efficient auxiliary models, producing fully interpretable, per-node uncertainty attributions at low computational cost.

## 1 Introduction

The landscape of artificial intelligence has been fundamentally reshaped by the emergence of reasoning models [23; 16]. Unlike traditional LLMs that rely on immediate token prediction, these models engage in deliberate, multi-step computation before committing to an answer. As these systems move from scientific performance benchmarks to high-stakes deployments and autonomous agentic loops, the need for additional metrics, in particular, robust uncertainty quantification (UQ) becomes critical. A safe deployment in such domains (e.g. medical diagnosis or legal decisions) hinges not only on a model's accuracy, but on its ability to signal when its own internal reasoning process has failed to converge on a stable or logically consistent conclusion [1; 49]. Existing UQ methods for language models fall into two camps, and both are ill-suited to reasoning models. Consistency-based methods sample multiple outputs and measure their agreement [33; 28]. For reasoning models generating thousands of tokens per trace, this multiplies inference cost by an order of magnitude, making the approach impractical in many deployment settings. Moreover, by looking only at the final-answer agreement, these methods discard all information in the reasoning trace itself. Token-level entropy methods use token log-probabilities as a confidence proxy [12; 17]. These are computationally less expensive, but the signal degrades precisely where it is needed most [54], as the deeper context mechanically inflates token confidence regardless of the reasoning quality. Simpler heuristics like counting trace length [9], backtracking phrases [29], or peak-entropy tokens [50] seemed to offer an easy and inexpensive way to catch uncertainty signals from reasoning traces, however recent work uncovered major shortcomings of these approaches: Trace length is confounded by problem difficulty, RL-induced length bias [52; 38], overthinking [19], and the apparent signal in overthinking tokens has been traced to training artifacts rather than genuine uncertainty [6]. Token-entropy also cannot separate semantic uncertainty (genuine ambiguity about the answer) from irrelevant lexical variation. Figure 2 gives an intuition of noisy and incoherent behavior token uncertainty can exhibit when confronted with complex tasks. Finally, token- and consistency-based approaches are further compounded by entropy collapse [7; 34; 24]: RL training sharpens output distributions as models exploit high-reward trajectories, making token-level signals increasingly unstable and suppressing the diversity that consistency methods depend on. The core issue is that all of these approaches ignore what actually determines whether a reasoning chain should be trusted: its logical structure. What premises are introduced? Do intermediate steps support or contradict one another? Does the conclusion follow from well-grounded inferential chains? None of this is captured by surface statistics or consistency aggregations. Current research in UQ lacks a framework capable of assessing these internal dynamics. To fill this gap, we propose Argument Graph Uncertainty (AGU), a post-hoc UQ framework that estimates model confidence from the logical structure of a single chain-of-thought trace for Multiple-Choice-Question-Answering (MCQA). AGU segments the trace into propositional units via a small instruction-tuned model. It then employs a lightweight natural language inference (NLI) model to infer the support and attack relations between these units, constructing two complementary argument graphs that capture global consistency, local trajectory coherence, and node-to-option evidence. By mapping these structural signals into a Dirichlet posterior, AGU provides a calibrated uncertainty estimate that remains robust even when traditional probabilistic signals fail. Figure 1 provides a graphical overview of individual steps of our proposed method. We show that AGU consistently outperforms token-based and consistency-based baselines across multiple reasoning models for the GPQA, MMLU-PRO and TruthfulQA benchmarks [43; 46; 32]. We further empirically show that AGU captures an orthogonal signal, making it a natural complement to existing methods. Furthermore, it adds only small computational overhead compared to consistency based methods, since it relies entirely on small auxiliary models. Our contributions can be summarized as follows:

![](images/fd9a23fc1dd060afd09b9b0fed4e0526fbcc50e7f60e7b0eb7c0099dde34e0c2.jpg)  
Figure 1: Argument Graph Uncertainty. A reasoning trace is segmented into propositional nodes $\{ v _ { 1 } , \ldots , v _ { 4 } \}$ and passed through an NLI model to construct two complementary graphs: a Segment– Segment Graph capturing pairwise entailment, conflict, and asymmetry across all nodes, and a Segment–Option Commitment Graph mapping each node to answer options. Graph-derived evidence signals are aggregated into Dirichlet concentration parameters $\{ \bar { \alpha _ { o } } \}$ , from which a normalized predictive entropy $H _ { \mathrm { n o r m } }$ serves as the final uncertainty score.

![](images/c89e507e80b055ba513eaba9b8eab0f7d75ca5f2e6670609f06c03e852f96430.jpg)  
Figure 2: Token-level entropy across reasoning trace of a single GPQA example. Normalized per-token entropy (rolling mean, q10–q90 band) and backtracking cue tokens (×) across a single reasoning trace. The vast majority of tokens are generated with high confidence, with intermittent spikes, causing the rolling mean to settle roughly between ≈0.1 — ≈0.2. Backtracking cues exhibit strongly varying entropy, and do not align with spikes or plateaus. Reasoning traces of correct and incorrect predictions yield similar token entropy trajectories, making it unclear which tokens to select or how to aggregate them into a meaningful uncertainty signal.

• A Novel Reasoning Trace-Based Framework for UQ: We propose a new method for uncertainty quantification specifically designed for reasoning models. By exploiting the logical trajectories of reasoning traces, our approach captures structural uncertainty signals that are absent in established methods.

• Outperforming established Methods: We demonstrate that our method surpasses both token- and consistency-based uncertainty quantification techniques.

• Orthogonality of Graph-based UQ signals : Through detailed empirical combination analysis we show that graph-derived uncertainty serves as an orthogonal signal to consistency based approaches. This allows our method to be integrated with already existing UQ methods, and provide a stronger measure of uncertainty than either single method would on its own.

## 2 Related Work

Uncertainty Quantification for LLMs Xia et al. [48] categorized existing UQ approaches on LLMs into four: verbalizing methods, latent information methods, consistency-based methods, and semantic clustering methods. Verbalizing methods prompt a model to output uncertainty in natural language with the prediction [31]. Although the approach is straightforward and intuitive, instruction models' poor performance in expressing their uncertainty has been observed [51]. Latent information methods use probability distributions on each generated token, which we refer to as token-level entropy. They typically estimate uncertainty using specific measures, such as averaged or maximum of negative log-likelihood [36] or entropy of the response distribution in single or multiple responses [42]. Consistency-based methods measure consistency or variance over multiple responses based on similarity [5]. Several metrics are proposed to estimate similarity, from lexical-based metrics such as BLEU [40; 22] to more sophisticated approaches capturing the semantic level, such as SentenceBERT or NLI-based methods [15; 47]. Semantic clustering methods also consider multiple responses, however instead of measuring similarity scores, they cluster the response by their semantic equivalence and calculate the probability over the clusters to estimate uncertainty [28]. They employ

NLI classifier or general LLMs to determine the entailment relationships [28; 13]. Our approach relates to semantic clustering methods, however it differs in applying clustering methods in reasoning traces in terms of the logical relevance to the categorical answer options using argument graphs.

Uncertainty Quantification for Reasoning Models Harsha Tanneru et al. [18] proposed Chain of Thought Uncertainty, which measures the agreement between two Chain-of-Thought (CoT) explanations by employing an NLI classifier [20]. They found a strong correlation between confidence in CoT explanations and the correctness of the final responses, highlighting the importance of incorporating CoT into UQ. On the other hand, Agarwal et al. [2] observed surprisingly weak UQ performance especially on large reasoning models, leaving a critical gap in reliable uncertainty estimation for reasoning-intensive tasks. Zhang and Zhang [53] distilled critical information by extracting keywords and assessed their degree of importance to the final answer. Ballon et al. [3] investigated how accuracy evolves over the reasoning trajectory by partitioning CoT traces into deciles and truncating them at each cutoff point. On GPQA Diamond [43] and MMLU-PRO [45], which require multi-step reasoning, they observed a progressive score increase across successive deciles that aligned semantically with reasoning progress, rather than merely reflecting context length or structural form. However, for long-horizon reasoning chains, the issue of inflated uncertainty scores must be addressed [27]; to this end, Mao and Venkat [37] proposed a recurrent propagation mechanism that yields more accurate uncertainty estimates by integrating historical confidence information across steps. Da et al. [8] introduced a method for incorporating structural information from reasoning chains by applying a graph edit distance-based measure.

Abstract Argumentation Frameworks. Our approach is conceptually grounded in the work of Dung [11], which introduced the Abstract Argumentation Framework. Dung's framework treats arguments as atomic nodes in a directed graph where edges represent "attack" relations. The "acceptability" of a set of arguments is then determined by graph-theoretic properties such as conflictfreedom and admissibility. While Dung's work focused on formal logic programming and n-person games, it provides the mathematical foundation for identifying consistent subsets of reasoning steps. We adapt this intuition to reasoning chains of LLMs, moving from binary attack relations to probabilistic NLI-based scores that quantify the degree of conflict and support within a generated trace.

NLI Modelling Natural Language Inference is the task of determining whether a hypothesis sentence is entailed, contradicted, or neutral, given a premise [14; 4]. Given the relevance to the logical support and oppositions between propositions, NLI therefore provides a mechanism to infer relational edges, which we employ in our approach. We use DeBERTa [21] as pretrained backbone to infer entailment, neutrality or contradictions that serve as edges in the argument graphs we construct. DeBERTA further enables computationally efficient modeling of dependency between premisehypothesis pairs. DeBERTa model remains the strongest encoder-based model on NLI benchmarks despite the relatively low computational cost [30].

Evidential Deep Learning and Dirichlet Posteriors. To map our graph-based signals onto a formal uncertainty signal, we adopt a Evidential Deep Learning (EDL) framework. Following Sensoy et al. [44], we treat the parameters of a Dirichlet distribution as evidence for the predicted class probabilities. While EDL has been extensively utilized to quantify epistemic uncertainty and detect out-of-distribution (OOD) samples in computer vision [35] and subjective logic for NLP classification [25], its application to reasoning structures remains unexplored. AGU therefore represents a departure from classical EDL: rather than deriving evidence from raw output logits, we compute it from the structural integrity and logical coherence of the model's argumentation graph

Multiple-Choice Question Answering Multiple-choice question answering (MCQA) is a natural evaluation paradigm for calibrated uncertainty, as discrete selection over a fixed candidate set mirrors real-world decision problems including agentic tool selection [41], retrieval-augmented re-ranking, and structured clinical triage [26]. The closed-form answer space affords unambiguous accuracy measurement, providing a clean signal for evaluating calibration quality [39; 10; 55]. We adopt GPQA Diamond [43] as our general benchmark. Its difficulty ensures that memorization alone cannot explain performance, as evidenced by the limited accuracy achieved by state-of-the-art systems [43]. We further include MMLU-PRO and TruthfulQA to asses generalizability across broad domains [46; 32].

## 3 Method

AGU is a post-hoc uncertainty estimation framework that treats a model's chain-of-thought as an argument graph. By doing so, it exploits the internal logical structure of a single generated reasoning trace. It decomposes the reasoning chain into discrete propositional segments, constructs two complementary relational graphs over those segments using a NLI scorer, and aggregates the resulting signals into a Dirichlet posterior over answer options whose concentration is then used to infer entropy that serves as uncertainty estimate.

## 3.1 Reasoning Segmentation

Given a model-generated chain-of-thought R for a multiple-choice question with K answer options $\mathcal { A } = \{ a _ { 1 } , \ldots , a _ { K } \}$ , our goal is to arrive at an uncertainty score $U _ { \mathrm { A G U } } ( R )$ . To this end, we partition R into n textual segments $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { n } \}$ using a light-weight instruct tuned LLM, where each $v _ { i }$ corresponds to a semantically coherent reasoning step. Throughout, $i , j \in \{ 1 , \ldots , n \}$ index segments and $\bar { k \in \{ 1 , \ldots , K \} }$ indexes answer options.

## 3.2 NLI Edge Scoring

All relational edges are then scored by a pretrained NLI model $f _ { \mathrm { N L I } }$ . For a premise p and a hypothesis $h ,$ the model returns a probability simplex over three classes:

$$
f _ { \mathrm { N L I } } ( p , h ) = \bigl ( p _ { E } ( p , h ) , ~ p _ { N } ( p , h ) , ~ p _ { C } ( p , h ) \bigr ) , \qquad p _ { E } + p _ { N } + p _ { C } = 1 ,\tag{1}
$$

where $E , N , C$ denote entailment, neutral, and contradiction, respectively. The first argument is always the premise and the second the hypothesis. The NLI model is applied to two types of input pairs: segment-segment pairs $( v _ { i } , v _ { j } )$ , with segment $v _ { i }$ as premise and segment $v _ { j }$ as hypothesis (Section 3.4), and segment-option pairs $( v _ { i } , h _ { k } )$ , with segment $v _ { i }$ as premise and the hypothesis $h _ { k } =$ "The answer is Option X: [option text]" constructed from answer option $a _ { k }$ as hypothesis (Section 3.3), for which we write $p _ { E } ( v _ { i } , h _ { k } ) , p _ { N } ( v _ { i } , h _ { k } ) , p _ { C } ( v _ { i } , h _ { k } )$ analogously.

## 3.3 Node-Option Graph

For each segment $v _ { i } \in \mathcal V$ and each answer option $a _ { k } \in { \mathcal { A } }$ , we define the contradiction-adjusted support of node $v _ { i }$ for option $a _ { k }$ as:

$$
e _ { i , k } = \operatorname* { m a x } \ ( 0 , p _ { E } ( v _ { i } , h _ { k } ) - p _ { C } ( v _ { i } , h _ { k } ) ) .\tag{2}
$$

This quantity is strictly positive only when a node's entailment of an option outweighs its contradiction of that option, i.e. it counts as evidence for an option only if it supports that option more than it contradicts it. Rectifying at zero means a step arguing against an option contributes nothing, rather than cancelling evidence contributed by other steps.

## 3.4 All-to-All Segment Graph

For every ordered pair $( v _ { i } , v _ { j } )$ with $i \neq j$ , we score $v _ { i }$ as the premise and $v _ { j }$ as the hypothesis, yielding $p _ { E } ( i , j ) , p _ { N } ( i , j ) , p _ { C } ( i , j )$ . This segment-to-segment graph captures global logical consistency across the full reasoning chain. We extract two scalar properties from the segment-to-segment graph. We define global conflict as the mean pairwise contradiction across all directed edges:

$$
\mathrm { C o n f l i c t } = \frac { 1 } { n ( n - 1 ) } \sum _ { i \neq j } p _ { C } ( v _ { i } , v _ { j } ) .\tag{3}
$$

Global conflict measures how much the trace argues against itself, averaged over all scored segment pairs. High conflict indicates that reasoning steps are mutually inconsistent. We define directional asymmetry as the mean absolute difference in entailment scores between opposing directions:

$$
\mathrm { A s y m m e t r y } = { \frac { 2 } { n ( n - 1 ) } } \sum _ { i < j } \bigl | p _ { E } ( v _ { i } , v _ { j } ) - p _ { E } ( v _ { j } , v _ { i } ) \bigr | .\tag{4}
$$

Directional asymmetry measures whether the entailment between two steps survives swapping premise and hypothesis. High asymmetry indicates that the support between nodes depends on their order, i.e. that a step follows from another without the reverse holding. Scoring all $n ( n - 1 )$ ordered pairs is quadratic in the number of segments. Alternatively, we can restrict the segment-to-segment graph to pairs within a window of width w along the reasoning chain, $| i - j | \le w \bar { / 2 }$ , which reduces the number of scored pairs to $\approx n w$ . Each segment is thus compared only with its $w / 2$ preceding and $w / 2$ following segments, in both directions, so the graph captures local rather than global consistency. Conflict, asymmetry and the global coherence (see Section 3.5) are then averaged over the scored pairs instead of all pairs; the remaining steps are unchanged

## 3.5 Node Reliability Weights

Each node $v _ { i }$ receives a reliability weight combining both global and local graph signals. The raw coherence (global coherence) of node $v _ { i }$ over the all-to-all graph is its mean net entailment across all other nodes:

$$
\mathrm { c o h } _ { i } ^ { \mathrm { r a w } } = \frac { 1 } { n - 1 } \sum _ { j \neq i } \left( p _ { E } ( v _ { i } , v _ { j } ) - p _ { C } ( v _ { i } , v _ { j } ) \right) \in [ - 1 , 1 ] , \quad \mathrm { n o r m a l i z e d a s } \quad \mathrm { c o h } _ { i } = \frac { 1 } { 2 } ( \mathrm { c o h } _ { i } ^ { \mathrm { r a w } } + 1 ) .\tag{5}
$$

We define local trajectory coherence as follows: let $\tau ( i )$ denote the set of adjacent edges incident to $v _ { i } .$ i.e. $( v _ { i - 1 }  v _ { i } )$ and $( v _ { i }  v _ { i + 1 } )$ , each scored with the earlier segment as premise and the later one as hypothesis; the first and last segment have only one such edge. The raw local coherence is:

$$
\mathrm { t r a j } _ { i } ^ { \mathrm { r a w } } = \frac { 1 } { | \mathcal { T } ( i ) | } \sum _ { j \in \{ i - 1 , i \} } \left( p _ { E } ( v _ { j } , v _ { j + 1 } ) - p _ { C } ( v _ { j } , v _ { j + 1 } ) \right) \in [ - 1 , 1 ] ,\tag{6}
$$

normalized as $\begin{array} { r } { \operatorname { t r a j } _ { i } = \frac { 1 } { 2 } ( \operatorname { t r a j } _ { i } ^ { \operatorname { r a w } } + 1 ) } \end{array}$ . We then compute a final node weight as a weighted sum of global and local coherence:

$$
r _ { i } = \lambda \mathrm { c o h } _ { i } + ( 1 - \lambda ) \mathrm { t r a j } _ { i } ,\tag{7}
$$

where $\lambda \in [ 0 , 1 ]$ is a hyperparameter. Equations $( 5 ) { - } ( 7 )$ weight each step by how well it fits the trace as a whole (global coherence, Eq. (5)) and its immediate neighbours (local coherence, Eq. (6)): a step is trusted to the extent that it is consistent with both, with λ setting the balance $\left( \operatorname { E q . } \left( 7 \right) \right)$

## 3.6 Dirichlet Evidence Aggregation

We then aggregate node-level evidence into a per-option signal:

$$
E _ { k } = \sum _ { i = 1 } ^ { n } r _ { i } e _ { i , k } , \qquad k = 1 , \dots , K .\tag{8}
$$

To translate this evidence into a distribution over options, we adopt a Dirichlet model [44]. The effective evidence scale is penalized by the global graph irregularities identified above:

$$
\kappa _ { \mathrm { e f f } } = \kappa \exp ( - \gamma _ { c } \mathrm { C o n f i c t } - \gamma _ { a } \mathrm { A s y m m e t r y } ) ,\tag{9}
$$

where $\kappa , \gamma _ { c } , \gamma _ { a } > 0$ are hyperparameters. Chains with high internal conflict or directional instability thus receive a globally attenuated evidence scale, reflecting reduced trustworthiness of the reasoning trace as a whole. We can then go on to compute Dirichlet parameters as $\alpha _ { k } = \alpha _ { 0 } + \kappa _ { \mathrm { e f f } } E _ { k } , k =$ $1 , \ldots , K$ , where $\alpha _ { 0 } > 0$ is a symmetric prior. The predictive mean over options can be derived as:

$$
\hat { p } _ { k } = \frac { \alpha _ { k } } { S } , \qquad S = \sum _ { k = 1 } ^ { K } \alpha _ { k } .\tag{10}
$$

## 3.7 Uncertainty Quantification

We compute uncertainty estimates from the Dirichlet posterior as Shannon entropy over the predictive mean, normalized to [0, 1]:

$$
U _ { \mathrm { A G U } } ( R ) = H _ { \mathrm { n o r m } } = \frac { - \sum _ { k = 1 } ^ { K } \hat { p } _ { k } \log \hat { p } _ { k } } { \log K } .\tag{11}
$$

High entropy indicates that probability mass is spread broadly across options, while low entropy reflects a concentrated, confident prediction.

## 4 Experimental Setting and Results

Experimental Setting We evaluate our framework on the aforementioned benchmarks, using nine publicly available reasoning models: DeepSeek-R1-Llama-70B, DeepSeek-R1-Qwen-14B, DeepSeek-R1-Llama-8B, GPT-OSS-120B, GPT-OSS-20B, Nemotron-Nano-30B, Qwen3-30B-Thinking, Qwen3-4B-Thinking, and Phi-4-Mini-Reasoning. All models are run in FP16 precision with sampling temperature 0.7 and top-p 0.9. We report average metrics over $n { = } 5$ sampled answers and reasoning traces per question. We use Qwen-8B Instruct for deriving segments V from reasoning traces if not stated otherwise. We attach the prompt used for segmentation in Appendix D. For detailed description of how scores and metrics are defined see Appendix A. All experiments reported in Table 1 are conducted with hyperparameters α0 = 1.0, κ = 12.0, γconflict = 1.2, γasymm = 1.0 and $\lambda = 0 . 5$ . Appendix C.1 offers a thorough analysis on hyperparameter influences. While our chosen default set of parameters is sufficient to achieve good performance, some models could benefit from hyperparameter tuning, as shown in moderate increases of PRR and $\rho .$

Uncertainty Calibration Table 1 reports uncertainty calibration across all nine reasoning models for GPQA diamond. $U _ { \mathrm { A G U } }$ achieves the highest PRR in eight out of nine models, demonstrating consistent gains over both $U _ { \mathrm { C o n s } }$ and $U _ { \mathrm { T o k e n } }$ across model families and scales. The improvements are particularly pronounced for smaller models such as Phi-4 Mini Reasoning and Qwen3 4B Thinking, where token-level uncertainty proves unreliable and consistency-based methods underperform, suggesting that argument-graph structure captures uncertainty signals that are otherwise lost. The single exception is DeepSeek-R1-D.-Llama-70B, where $U _ { \mathrm { C o n s } }$ attains the highest PRR $( 0 . 4 6 \pm 0 . 0 9 )$ . Token uncertainty $U _ { \mathrm { T o k e n } }$ performs the least reliably overall, with negative PRR values for GPT-OSS 120B and DeepSeek-R1-D.-Llama-70B indicating that token probabilities are miscalibrated to the point of inverting the rejection ordering, in line with the critical stance we took in Section 1. Appendix B.1 provides calibration results for MMLU-Pro and TruthfulQA.

![](images/d0edd7bf9f116052252caa8f3e02fb00e236d66dc4c1b55ff5a835fd64e22516.jpg)  
Figure 3: PRR across all models for $U _ { \mathrm { A G U } } , U _ { \mathrm { C o n s } } ,$ and their equal-weight combination. The combined score consistently dominates both individual methods.

Table 1: Calibration performance across nine reasoning models on GPQA Diamond. We compare Argument Graph Uncertainty $U _ { \mathrm { A G U } } ( R )$ against consistency-based uncertainty $U _ { \mathrm { C o n s } }$ and token probability uncertainty $U _ { \mathrm { T o k e n } }$ using Prediction Rejection Rate (PRR ↑) and Spearman correlation $( \rho \downarrow )$ . Bold indicates the best method per model. $\bar { U _ { \mathrm { A G U } } } ( R )$ achieves the highest PRR in eight out of nine models.
<table><tr><td></td><td colspan="2">Argument Graph Uncertainty UAGU(R)</td><td colspan="2"> $U _ { \mathbf { C o n s } }$ </td><td colspan="2"> $U _ { \mathbf { T o k e n } }$ </td></tr><tr><td>Model</td><td>PRR↑</td><td> $\rho \downarrow$ </td><td>PRR↑</td><td> $\rho \downarrow$ </td><td>PRR↑</td><td> $\rho \downarrow$ </td></tr><tr><td>Nemotron-3 Nano 30B</td><td> $\mathbf { 0 . 6 5 \pm 0 . 0 7 }$ </td><td> ${ \bf - 0 . 3 7 \pm 0 . 0 6 }$ </td><td> $0 . 2 4 \pm 0 . 1 4$ </td><td> $- 0 . 2 4 \pm 0 . 0 8$ </td><td> $0 . 4 1 \pm 0 . 1 5$ </td><td> $- 0 . 3 2 \pm 0 . 0 8$ </td></tr><tr><td>GPT-OSS 120B</td><td> ${ \bf 0 . 5 6 \pm 0 . 0 9 }$ </td><td> $- 0 . 2 8 \pm 0 . 0 6$ </td><td> $0 . 3 6 \pm 0 . 1 5$ </td><td> ${ \bf - 0 . 3 7 \pm 0 . 0 8 }$ </td><td> $- 0 . 2 7 \pm 0 . 1 9$ </td><td> $0 . 0 2 \pm 0 . 0 8$ </td></tr><tr><td>GPT-OSS 20B</td><td> ${ \bf 0 . 5 5 \pm 0 . 1 0 }$ </td><td> ${ \bf - 0 . 3 7 \pm 0 . 0 7 }$ </td><td> $0 . 3 5 \pm 0 . 1 4$ </td><td> $- 0 . 3 6 \pm 0 . 0 9$ </td><td> $0 . 1 4 \pm 0 . 1 8$ </td><td> $- 0 . 2 0 \pm 0 . 0 9$ </td></tr><tr><td>Qwen3 4B Thinking</td><td> ${ \bf 0 . 5 1 \pm 0 . 0 7 }$ </td><td> ${ \bf - 0 . 3 9 \pm 0 . 0 6 }$ </td><td> $0 . 2 0 \pm 0 . 0 9$ </td><td> $- 0 . 2 1 \pm 0 . 0 7$ </td><td> $0 . 2 7 \pm 0 . 0 8$ </td><td> $- 0 . 2 1 \pm 0 . 0 7$ </td></tr><tr><td>Qwen3 30B Thinking</td><td> ${ \bf 0 . 4 3 \pm 0 . 0 9 }$ </td><td> $- 0 . 2 7 \pm 0 . 0 6$ </td><td> $0 . 3 1 \pm 0 . 1 1$ </td><td> ${ \bf - 0 . 3 2 \pm 0 . 0 7 }$ </td><td> $0 . 1 0 \pm 0 . 0 2$ </td><td> $- 0 . 1 0 \pm 0 . 0 2$ </td></tr><tr><td>Phi-4 Mini Reasoning</td><td> ${ \bf 0 . 4 3 \pm 0 . 0 7 }$ </td><td> ${ \bf - 0 . 3 5 \pm 0 . 0 6 }$ </td><td> $0 . 1 5 \pm 0 . 0 8$ </td><td> $- 0 . 1 3 \pm 0 . 0 7$ </td><td> $0 . 0 6 \pm 0 . 0 8$ </td><td> $- 0 . 0 5 \pm 0 . 0 7$ </td></tr><tr><td>DeepSeek-R1-D.-Qwen-14B</td><td> ${ \bf 0 . 4 1 \pm 0 . 0 8 }$ </td><td> ${ \bf - 0 . 3 2 \pm 0 . 0 6 }$ </td><td> $0 . 2 9 \pm 0 . 0 9$ </td><td> $- 0 . 2 8 \pm 0 . 0 7$ </td><td> $0 . 1 0 \pm 0 . 1 0$ </td><td> $- 0 . 1 4 \pm 0 . 0 7$ </td></tr><tr><td>DeepSeek-R1-D.-Llama-70B</td><td> $0 . 3 4 \pm 0 . 0 4$ </td><td> $- 0 . 2 4 \pm 0 . 0 3$ </td><td> ${ \bf 0 . 4 6 \pm 0 . 0 9 }$ </td><td> $\mathbf { - 0 . 4 5 \pm 0 . 0 6 }$ </td><td> $- 0 . 3 9 \pm 0 . 0 7$ </td><td> $0 . 2 8 \pm 0 . 0 6$ </td></tr><tr><td>DeepSeek-R1-D.-Llama-8B</td><td> ${ \bf 0 . 2 9 \pm 0 . 0 8 }$ </td><td> ${ \bf - 0 . 2 7 \pm 0 . 0 6 }$ </td><td> $0 . 2 6 \pm 0 . 0 8$ </td><td> $- 0 . 2 2 \pm 0 . 0 7$ </td><td> $- 0 . 1 3 \pm 0 . 0 8$ </td><td> $0 . 0 7 \pm 0 . 0 7$ </td></tr></table>

Figure 3 shows the effect of combining $U _ { \mathrm { A G U } }$ and $U _ { \mathrm { C o n s } }$ via equal-weight blending. The combined score achieves the highest PRR in every model, and the gains are substantial: for Nemotron-3 Nano 30B the combined score reaches 0.68, compared to 0.65 for $U _ { \mathrm { A G U } }$ alone and 0.24 for $U _ { \mathrm { C o n s } }$ alone, and for DeepSeek-R1-D.-Llama-70B, the one model where $U _ { \mathrm { C o n s } }$ outperforms $U _ { \mathrm { A G U } }$ individually, the combination still improves over both. This consistent pattern strongly suggests that the two signals are largely orthogonal: argument-graph structure captures uncertainty arising from internal reasoning conflicts, while consistency-based methods reflect variability across independent samples, and these failure modes do not overlap. Appendix B.6 provides an extensive analysis of alternative combination strategies, offering the insight that basic equal blending is enough to utilize the potential of combining the two uncertainty signals. Furthermore, through visual inspection of distributions of uncertainty score we can get a sense of how well and why they might behave well or ill-calibrated: figure 4 provides an overview over token distributions of different methods for different models. Token uncertainty collapses to near-zero across all models, rendering it uninformative. Consistency scores are strongly bimodal: predictions are either unanimous or maximally uncertain, thereby offering little resolution in between. Argument graph uncertainty, by contrast, spreads broadly across [0, 1], providing a diverse and well-calibrated uncertainty signal.

![](images/78c6eaba65a0f1a41468f8f6873b99c9d3de89a7e62dcf55a29a755e064b4196.jpg)  
Figure 4: Uncertainty score distribution examples. Token entropy (top) collapses to near-zero across all models, rendering it uninformative. $U _ { \mathrm { C o n s } }$ (bottom) is strongly bimodal (models either agree unanimously or disagree maximally) with little resolution in between. $U _ { \mathrm { A G U } }$ (middle) spreads broadly across [0, 1], capturing diverse uncertainty signals and filling precisely the range that consistency cannot resolve.

Segmentation ablation. We ablate the Qwen3-8B segmenter used in the main experiments against two alternatives: a deterministic split of the trace at paragraph boundaries, which requires no model call, and a larger segmenter, GPT-OSS-120B (Table 2). The deterministic split loses no calibration on average and gains it for several models, with mean PRR 0.47 against 0.46, and improves significantly on GPT-OSS 20B. The larger segmenter yields the highest mean PRR at 0.50 and, notably, cuts the standard deviation from 0.08 to 0.03 on average, so the segmentation stage is the main source of run-to-run variance in $U _ { \mathrm { A G U } }$ . Since the paragraph split removes the segmentation cost entirely (Eq. 34) at no loss in mean PRR, we consider it the default choice when compute is a concern.

Robustness to NLI errors. Since $U _ { \mathrm { A G U } }$ is computed from the judgments of a small NLI scorer, we test how errors at the edge level propagate to the uncertainty estimate. We perturb the scorer's outputs on all nine GPQA Diamond models and recompute $\bar { U } _ { \mathrm { A G U } }$ , either flipping the predicted class of a random fraction ε of edges or adding Gaussian noise of standard deviation σ to the class log-probabilities (Fig. 5). Degradation is graceful in both cases. With 10% of edges mislabelled, 78% of the clean PRR is retained, and half remains when 40% of all edges carry a wrong class. Noise on the probabilities costs almost nothing until it exceeds the typical margin between classes, with 97% retained at $\sigma { = } 0 . 5$ and 85% at $\sigma { = } 1$ . Because the Dirichlet evidence aggregates over many edges, no single misclassification moves the per-option estimate much, and $U _ { \mathrm { A G U } }$ depends more on the scorer's ordinal judgment rather than on per-edge accuracy or calibration. Appendix B.3 gives the full details. Appendix B.2 provides additional analysis on distribution of edge classification by the NLI-scorer.

Table 2: Segmentation ablation on GPQA Diamond. PRR of $U _ { \mathrm { A G U } }$ across different segmenters, mean ± std over seeds. \* marks cells that differ significantly from the Qwen3-8B row (Welch's t-test, $p < 0 . 0 5 )$
<table><tr><td>Segmenter</td><td>Nemotron 30B</td><td>GPT-OSS 120B</td><td>GPT-OSS 20B</td><td>Qwen3 4B-T</td><td>Qwen3 30B-T</td><td>Phi-4 mini-R</td><td>R1-Qwen 14B</td><td>R1-Llama 70B</td><td>R1-Llama 8B</td><td>Mean</td></tr><tr><td>Qwen3-8B</td><td>0.65 ± .07</td><td>0.56 ± .09</td><td>0.55 ± .10</td><td>0.51 ± .07</td><td>0.43 ± .09</td><td>0.43 ± .07</td><td>0.41 ± .08</td><td>0.34 ± .04</td><td>0.29 ± .08</td><td>0.46</td></tr><tr><td>GPT-OSS-120B</td><td>0.67 ± .07</td><td>0.68 ± .03*</td><td>0.76 ± .03*</td><td>0.39 ± .06*</td><td>0.47 ± .02</td><td>0.32 ± .03*</td><td>0.42 ± .01</td><td>0.46 ± .02*</td><td>0.31 ± .02</td><td>0.50</td></tr><tr><td>Paragraph split</td><td> $0 . 7 1 \pm . 0 5$ </td><td> $0 . 6 2 \pm . 0 3$ </td><td> $0 . 7 6 \pm . 0 1 ^ { * }$ </td><td>0.35 ± .01*</td><td>0.46 ± .01</td><td>0.29 ± .03*</td><td> $0 . 3 7 \pm . 0 4$ </td><td> $0 . 3 6 \pm . 0 4$ </td><td> $0 . 2 9 \pm . 0 3$ </td><td>0.47</td></tr></table>

![](images/bf0165c483e60967f6584cd640826961b939b9b60ee62e5565e94383e2a267bd.jpg)

![](images/25822702cc206e1d941c4b9748f3d20ae80476b59d3f138b12322c26a21337c9.jpg)  
Figure 5: Robustness of $U _ { \mathrm { A G U } }$ to NLI errors. PRR retained on GPQA Diamond when (a) a fraction ε of edge classes is flipped at random and (b) Gaussian noise σ is added to the class log-probabilities. Grey: individual models; bold: mean; band: ±1 std.

All-to-all vs. windowed segment-to-segment graph. As mentioned in 3.4 we can either use the full segment-to-segment (all-to-all) graph, or a windowed subset. We provide an ablation of the all-to-all vs. the windowed graph in Appendix B.4 showing that PRR matches to within 0.01 for a window of w = 10 while the cost of the NLI stage reduces from quadratic to linear in the number of segments. Therefore we suggest using the windowed approached as default

AGU vs. trace length. Since longer traces contain more segments, we verify that $U _ { \mathrm { A G U } }$ is not simply a proxy for trace length. On GPQA Diamond, the segment count used directly as an uncertainty score attains a mean PRR of 0.33, compared to 0.45 for $U _ { \mathrm { A G U } }$ , which is the stronger signal on seven of nine models. The two are moderately correlated (mean Spearman 0.43); after rank-regressing the segment count out of $U _ { \mathrm { A G U } } .$ , a PRR of 0.33 remains, indicating that the majority of the graph signal is independent of trace length. See AppendixB.5 for details.

Computational Cost AGU adds no calls to the evaluated model; its overhead runs on an optional segmenter with $P _ { \mathrm { s e g } }$ parameters and an NLI encoder with $P _ { \mathrm { N L I } }$ parameters. For a trace of L tokens split into segments of average length l, with K answer options and window w,

$$
C _ { \mathrm { A G U } } ~ = ~ \underbrace { \Bigg \{ P _ { \mathrm { s e g } } ~ L \in \mathcal { O } ( P _ { \mathrm { s e g } } ~ L ) \quad \mathrm { L L M ~ s e g m e n t e r } } _ { \mathrm { p a r a f r a p h ~ s p l i t } }  _ { \mathrm { s e g m e n t u a t i o n } } + \underbrace { \Bigg \{ 2 ( w + K ) ~ P _ { \mathrm { N L I } } ~ L \in \mathcal { O } ( P _ { \mathrm { N L I } } ~ L ) \quad \mathrm { ~ w i n d o w } ~ w i d t h } _ { \mathrm { N L I } ~ L } ,\tag{12}
$$

A more detailed explanation and derivation are given in Appendix C. Fig. 6 depicts these costs graphically alongside the consistency baseline and reports measured wall-clock times on GPQA Diamond.

![](images/f94bf5ecb9198dc5ea434abafad431b3cd442c098dd7f965d3ada92ffc09854c.jpg)

![](images/ba0798239e4e8c7be5c436916fb22de2e6c604641d3ec2b5eb41cc44efc12547.jpg)  
Figure 6: Compute cost comparison. (a) Analytical cost against trace length for the consistency and AGU configurations. (b) Measured wall-clock time per model on GPQA Diamond.

## 5 Discussion

We introduced Argument Graph Uncertainty (AGU), a post-hoc uncertainty quantification framework that estimates model uncertainty from the logical structure of a single chain-of-thought reasoning trace. AGU outperforms both token-level and consistency-based baselines across eight of nine reasoning models on GPQA Diamond, while remaining computationally cheap. Beyond its standalone performance, we find that $U _ { \mathrm { A G U } }$ captures an orthogonal uncertainty signal to $U _ { \mathrm { C o n s } } { \mathrm { : } }$ a simple equalweight blend uniformly improves over either method alone on GPQA Diamond, MMLU-Pro and TruthfulQA, even where $U _ { \mathrm { C o n s } }$ is the stronger single signal. This suggests that internal logical coherence and cross-sample output variance probe complementary failure modes. Our ablations show that the framework is robust to its auxiliary components: a deterministic paragraph split matches the LLM segmenter in mean PRR at zero segmentation cost, the windowed segment graph matches the all-to-all graph while reducing the NLI stage from quadratic to linear in the number of segments. We further show that $U _ { \mathrm { A G U } }$ degrades gracefully under NLI errors, suggesting a robust signal chain. Several directions remain open for future work. First, extending AGU beyond the MCQA setting, for instance to open-ended generation, requires rethinking the node-to-option commitment graph, which currently relies on a fixed answer set. Second, while AGU tolerates edge-level error, the quality of its estimates remains upstream-dependent on the NLI scorer; a scorer trained on reasoning text rather than general-domain NLI could yield further calibration gains.

## 6 Limitation

AGU depends on the quality of both the segmentation model and the NLI scorer. Although we show that a deterministic paragraph split suffices and that the aggregate tolerates substantial edge-level error, NLI models trained on general-domain corpora can misclassify relations in specialised scientific language, in particular between short numeric fragments, and our robustness analysis only covers synthetic perturbations of a single scorer. Our evaluation is limited to the MCQA setting on three benchmarks; whether AGU generalises to open-ended generation or other reasoning domains remains an open question. Finally, the windowed graph keeps the NLI stage linear in trace length, but very long traces still incur a segmentation cost that scales with the trace when an LLM segmenter is used.

Societal Impact. Reliable uncertainty quantification is a prerequisite for safe deployment of reasoning models in high-stakes domains. By providing interpretable, per-node uncertainty attributions at low computational cost, AGU could lower the barrier to auditing model reasoning in such settings. We note, however, that improved uncertainty estimates could also increase trust in model outputs in contexts where human oversight should be maintained. While we principally see the impact of research in UQ as societally beneficial, we want to emphasize that with the rising deployment of agentic systems, well-calibrated uncertainty estimates should complement rather than replace human oversight.

## Acknowledgements

This work was supported by the Senate of Berlin and the European Commision's Digital Europe Programme (DIGITAL) as grant TEF-Health (101100700).

## References

[1] M. Abdar, F. Pourpanah, S. Hussain, D. Rezazadegan, L. Liu, M. Ghavamzadeh, P. Fieguth, X. Cao, A. Khosravi, U. R. Acharya, et al. A review of uncertainty quantification in deep learning: Techniques, applications and challenges. Information fusion, 76:243–297, 2021.

[2] R. Agarwal, M. Wenzel, S. Baur, J. Zimmer, G. Harvey, and J. Ma. Made: A living benchmark for multi-label text classification with uncertainty quantification of medical device adverse events, 2026. URL https://arxiv.org/abs/2604.15203.

[3] M. Ballon, B. Verbeken, V. Ginis, and A. Algaba. Probing the trajectories of reasoning traces in large language models, 2026. URL https://arxiv.org/abs/2601.23163.

[4] S. R. Bowman, G. Angeli, C. Potts, and C. D. Manning. A large annotated corpus for learning natural language inference. In L. Màrquez, C. Callison-Burch, and J. Su, editors, Proceedings of the 2015 Conference on Empirical Methods in Natural Language Processing, pages 632–642, Lisbon, Portugal, Sept. 2015. Association for Computational Linguistics. doi: 10.18653/v1/D15-1075. URL https: //aclanthology.org/D15-1075/.

[5] J. Cole, M. Zhang, D. Gillick, J. Eisenschlos, B. Dhingra, and J. Eisenstein. Selectively answering ambiguous questions. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 530–543, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.35. URL https://aclanthology. org/2023.emnlp-main.35/.

[6] J. Cuesta-Ramirez, S. Beaussant, and M. Mounsif. Large reasoning models are not thinking straight: on the unreliability of thinking trajectories. arXiv preprint arXiv:2507.00711, 2025.

[7] G. Cui, Y. Zhang, J. Chen, L. Yuan, Z. Wang, Y. Zuo, H. Li, Y. Fan, H. Chen, W. Chen, Z. Liu, H. Peng, L. Bai, W. Ouyang, Y. Cheng, B. Zhou, and N. Ding. The entropy mechanism of reinforcement learning for reasoning language models, 2025. URL https://arxiv.org/abs/2505.22617.

[8] L. Da, X. Liu, J. Dai, L. Cheng, Y. Wang, and H. Wei. Understanding the uncertainty of llm explanations: A perspective based on reasoning topology, 2025. URL https://arxiv.org/abs/2502.17026.

[9] S. Devic, C. Peale, A. Bradley, S. Williamson, P. Nakkiran, and A. Gollakota. Trace length is a simple uncertainty signal in reasoning models. arXiv preprint arXiv:2510.10409, 2025.

[10] J. Duan, J. Diffenderfer, S. Madireddy, T. Chen, B. Kailkhura, and K. Xu. Uprop: Investigating the uncertainty propagation of llms in multi-step agentic decision-making. arXiv preprint arXiv:2506.17419, 2025.

[11] P. M. Dung. On the acceptability of arguments and its fundamental role in nonmonotonic reasoning, logic programming and n-person games. Artificial intelligence, 77(2):321–357, 1995.

[12] E. Fadeeva, A. Rubashevskii, A. Shelmanov, S. Petrakov, H. Li, H. Mubarak, E. Tsymbalov, G. Kuzmin, A. Panchenko, T. Baldwin, et al. Fact-checking the output of large language models via token-level uncertainty quantification. In Findings of the Association for Computational Linguistics: ACL 2024, pages 9367–9385, 2024.

[13] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, 2024.

[14] S. Farquhar, J. Kossen, L. Kuhn, and Y. Gal. Detecting hallucinations in large language models using semantic entropy. Nature, 630(8017):625–630, June 2024. ISSN 1476-4687. doi: 10.1038/s41586-024-07421-0. URL https://doi.org/10.1038/s41586-024-07421-0.

[15] X. Gao, J. Zhang, L. Mouatadid, and K. Das. SPUQ: Perturbation-based uncertainty quantification for large language models. In Y. Graham and M. Purver, editors, Proceedings of the 18th Conference of the European Chapter of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2336–2346, St. Julian's, Malta, Mar. 2024. Association for Computational Linguistics. doi: 10.18653/v1/ 2024.eacl-long.143. URL https://aclanthology.org/2024.eacl-long.143/.

[16] D. Guo, D. Yang, H. Zhang, J. Song, P. Wang, Q. Zhu, R. Xu, R. Zhang, S. Ma, X. Bi, et al. Deepseek-r1: Incentivizing reasoning capability in llms via reinforcement learning. arXiv preprint arXiv:2501.12948 2025.

[17] N. Gupta, H. Narasimhan, W. Jitkrittum, A. S. Rawat, A. K. Menon, and S. Kumar. Language model cascades: Token-level uncertainty and beyond. arXiv preprint arXiv:2404.10136, 2024.

[18] S. Harsha Tanneru, C. Agarwal, and H. Lakkaraju. Quantifying uncertainty in natural language explanations of large language models. In S. Dasgupta, S. Mandt, and Y. Li, editors, Proceedings of The 27th International Conference on Artificial Intelligence and Statistics, volume 238 of Proceedings of Machine Learning Research, pages 1072–1080. PMLR, 02–04 May 2024. URL https://proceedings .mlr. press/v238/harsha-tanneru24a.html.

[19] M. Hassid, G. Synnaeve, Y. Adi, and R. Schwartz. Don't overthink it. preferring shorter thinking chains for improved 1lm reasoning. arXiv preprint arXiv:2505.17813, 2025.

[20] P. He, X. Liu, J. Gao, and W. Chen. Deberta: Decoding-enhanced bert with disentangled attention. In International Conference on Learning Representations, 2021. URL https://openreview.net/forum? id=XPZIaotutsD.

[21] P. He, X. Liu, J. Gao, and W. Chen. Deberta: Decoding-enhanced bert with disentangled attention, 2021. URL https://arxiv.org/abs/2006.03654.

[22] Y. Huang, J. Song, Z. Wang, S. Zhao, H. Chen, F. Juefei-Xu, and L. Ma. <i>look before you leap:</i> an exploratory study of uncertainty analysis for large language models. IEEE Transactions on Software Engineering, 51(2):413–429, Feb. 2025. ISSN 2326-3881. doi: 10.1109/tse.2024.3519464. URL http://dx.doi.org/10.1109/TSE.2024.3519464.

[23] A. Jaech, A. Kalai, A. Lerer, A. Richardson, A. El-Kishky, A. Low, A. Helyar, A. Madry, A. Beutel, A. Carney, et al. Openai o1 system card. arXiv preprint arXiv:2412.16720, 2024.

[24] R. Jin, P. Gao, Y. Ren, Z. Han, T. Zhang, W. Huang, W. Liu, J. Luan, and D. Xiong. Revisiting entropy in reinforcement learning for large reasoning models. arXiv preprint arXiv:2511.05993, 2025.

[25] A. Jøsang. Subjective Logic: A Formalism for Reasoning Under Uncertainty. Springer, 2016.

[26] Z. Kanjee, B. Crowe, and A. Rodman. Accuracy of a generative artificial intelligence model in a complex diagnostic challenge. Jama, 330(1):78–80, 2023.

[27] T. Korbak, M. Balesni, E. Barnes, Y. Bengio, J. Benton, J. Bloom, M. Chen, A. Cooney, A. Dafoe, A. Dragan, S. Emmons, O. Evans, D. Farhi, R. Greenblatt, D. Hendrycks, M. Hobbhahn, E. Hubinger, G. Irving, E. Jenner, D. Kokotajlo, V. Krakovna, S. Legg, D. Lindner, D. Luan, A. Mądry, J. Michael, N. Nanda, D. Orr, J. Pachocki, E. Perez, M. Phuong, F. Roger, J. Saxe, B. Shlegeris, M. Soto, E. Steinberger, J. Wang, W. Zaremba, B. Baker, R. Shah, and V. Mikulik. Chain of thought monitorability: A new and fragile opportunity for ai safety, 2025. URL https://arxiv.org/abs/2507.11473.

[28] L. Kuhn, Y. Gal, and S. Farquhar. Semantic uncertainty: Linguistic invariances for uncertainty estimation in natural language generation. arXiv preprint arXiv:2302.09664, 2023.

[29] A. Kumar, V. Zhuang, R. Agarwal, Y. Su, J. D. Co-Reyes, A. Singh, K. Baumli, S. Iqbal, C. Bishop, R. Roelofs, et al. Training language models to self-correct via reinforcement learning. arXiv preprint arXiv:2409.12917, 2024.

[30] M. Laurer, W. van Atteveldt, A. Casas, and K. Welbers. Building efficient universal classifiers with natural language inference, 2024. URL https://arxiv.org/abs/2312.17543.

[31] S. Lin, J. Hilton, and O. Evans. Teaching models to express their uncertainty in words. arXiv preprint arXiv:2205.14334, 2022.

[32] S. Lin, J. Hilton, and O. Evans. Truthfulqa: Measuring how models mimic human falsehoods. In Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers), pages 3214–3252, 2022.

[33] Z. Lin, S. Trivedi, and J. Sun. Generating with confidence: Uncertainty quantification for black-box large language models. arXiv preprint arXiv:2305.19187, 2023.

[34] X. Liu, T. Chen, L. Da, C. Chen, Z. Lin, and H. Wei. Uncertainty quantification and confidence calibration in large language models: A survey. In Proceedings of the 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V. 2, pages 6107–6117, 2025.

[35] A. Malinin and M. Gales. Predictive uncertainty estimation via prior networks. Advances in neural information processing systems, 31, 2018.

[36] P. Manakul, A. Liusie, and M. Gales. SelfCheckGPT: Zero-resource black-box hallucination detection for generative large language models. In H. Bouamor, J. Pino, and K. Bali, editors, Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 9004–9017, Singapore, Dec. 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.emnlp-main.557. URL https://aclanthology.org/2023.emnlp-main.557/.

[37] Z. Mao and A. Venkat. Recurrent confidence chain: Temporal-aware uncertainty quantification in large language models, 2026. URL https://arxiv.org/abs/2601.13368.

[38] D. Nohara, T. Nakamura, and R. Yokota. On the optimal reasoning length for rl-trained language models. arXiv preprint arXiv:2602.09591, 2026.

[39] C. Oh, S. Park, T. E. Kim, J. Li, W. Li, S. Yeh, X. Du, H. Hassani, P. Bogdan, D. Song, et al. Uncertainty quantification in llm agents: Foundations, emerging challenges, and opportunities. In Agentic AI in the Wild: From Hallucinations to Reliable Autonomy.

[40] K. Papineni, S. Roukos, T. Ward, and W.-J. Zhu. Bleu: a method for automatic evaluation of machine translation. In P. Isabelle, E. Charniak, and D. Lin, editors, Proceedings of the 40th Annual Meeting of the Association for Computational Linguistics, pages 311–318, Philadelphia, Pennsylvania, USA, July 2002. Association for Computational Linguistics. doi: 10.3115/1073083.1073135. URL https: //aclanthology.org/P02-1040/.

[41] S. G. Patil, T. Zhang, X. Wang, and J. E. Gonzalez. Gorilla: Large language model connected with massive apis. Advances in Neural Information ProcessingSystems, 37:126544–126565, 2024.

[42] G. Portillo Wightman, A. Delucia, and M. Dredze. Strength in numbers: Estimating confidence of large language models by prompt agreement. In A. Ovalle, K.-W. Chang, N. Mehrabi, Y. Pruksachatkun, A. Galystan, J. Dhamala, A. Verma, T. Cao, A. Kumar, and R. Gupta, editors, Proceedings of the 3rd Workshop on Trustworthy Natural Language Processing (TrustNLP 2023), pages 326–362, Toronto, Canada, July 2023. Association for Computational Linguistics. doi: 10.18653/v1/2023.trustnlp-1.28. URL https://aclanthology.org/2023.trustnlp-1.28/.

[43] D. Rein, B. L. Hou, A. C. Stickland, J. Petty, R. Y. Pang, J. Dirani, J. Michael, and S. R. Bowman. Gpqa: A graduate-level google-proof q&a benchmark. arXiv preprint arXiv:2311.12022, 2023.

[44] M. Sensoy, L. Kaplan, and M. Kandemir. Evidential deep learning to quantify classification uncertainty. Advances in neural information processing systems, 31, 2018.

[45] Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, T. Li, M. Ku, K. Wang, A. Zhuang, R. Fan, X. Yue, and W. Chen. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark, 2024. URL https://arxiv.org/abs/2406.01574.

[46] Y. Wang, X. Ma, G. Zhang, Y. Ni, A. Chandra, S. Guo, W. Ren, A. Arulraj, X. He, Z. Jiang, et al. Mmlu-pro: A more robust and challenging multi-task language understanding benchmark. Advances in Neural Information Processing Systems, 37:95266–95290, 2024.

[47] Z. Wang and C. Holmes. On subjective uncertainty quantification and calibration in natural language generation, 2024. URL https://arxiv.org/abs/2406.05213.

[48] Z. Xia, J. Xu, Y. Zhang, and H. Liu. A survey of uncertainty estimation methods on large language models. In W. Che, J. Nabende, E. Shutova, and M. T. Pilehvar, editors, Findings of the Association for Computational Linguistics: ACL 2025, pages 21381–21396, Vienna, Austria, July 2025. Association for Computational Linguistics. ISBN 979-8-89176-256-5. doi: 10.18653/v1/2025.findings-ac1.1101. URL https://aclanthology.org/2025.findings-acl.1101/.

[49] Z. Xia, J. Xu, Y. Zhang, and H. Liu. A survey of uncertainty estimation methods on large language models. In Findings of the Association for Computational Linguistics: ACL 2025, pages 21381–21396, 2025.

[50] M. Xiong, Z. Hu, X. Lu, Y. Li, J. Fu, J. He, and B. Hooi. Can llms express their uncertainty? an empirical evaluation of confidence elicitation in 1lms. arXiv preprint arXiv:2306.13063, 2023.

[51] G. Yona, R. Aharoni, and M. Geva. Can large language models faithfully express their intrinsic uncertainty in words? In Y. Al-Onaizan, M. Bansal, and Y.-N. Chen, editors, Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 7752–7764, Miami, Florida, USA, Nov. 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.emnlp-main.443. URL https: //aclanthology.org/2024.emnlp-main.443/.

[52] D. Yuan, T. Xie, S. Huang, Z. Gong, H. Zhang, C. Luo, F. Wei, and D. Zhao. Efficient rl training for reasoning models via length-aware optimization. arXiv preprint arXiv:2505.12284, 2025.

[53] B. Zhang and R. Zhang. Cot-uq: Improving response-wise uncertainty quantification in llms with chain-ofthought, 2025. URL https://arxiv.org/abs/2502.17214

[54] X. Zhao. Entropy trajectory shape predicts llm reasoning reliability: A diagnostic study of uncertainty dynamics in chain-of-thought. arXiv preprint arXiv:2603.18940, 2026.

[55] H. Zhou and H. Y. Chan. Orch: many analyses, one merge-a deterministic multi-agent orchestrator for discrete-choice reasoning with ema-guided routing. arXiv preprint arXiv:2602.01797, 2026.

## A Definitions of Metrics and Uncertainty Scores

## A.1 Metrics

## A.1.1 Prediction Rejection Ratio (PRR).

PRR measures how well an uncertainty score u can be used to selectively abstain from answering. Given $N$ samples with correctness indicators $\textbf { c } = ~ \left( c _ { 1 } , \ldots , c _ { N } \right)$ and uncertainty scores for different methods $\mathbf { U } ( \mathbf { M e t h o d } ) = ( u _ { 1 } , \dots , u _ { N } )$ , we define a rejection curve as follows. For each rejection fraction $r \in [ 0 , 1 )$ the $\lvert r N \rvert$ samples with the highest uncertainty are discarded and accuracy is computed on the remaining $N - \lfloor r \bar { N } \rfloor$ samples:

$$
\operatorname { A c c } ( r ; \pi ) = \frac { 1 } { N - \lfloor r N \rfloor } \sum _ { i \in \mathcal { K } ( r , \pi ) } c _ { i } ,\tag{13}
$$

where π is an ordering of samples and $\kappa ( r , \pi )$ denotes the indices of retained samples after rejecting the first $\lfloor r N \rfloor$ elements of π. Three orderings are considered:

• Uncertainty $( \pi _ { u } ) \colon$ sort by $u _ { i }$ descending (most uncertain first).

• Oracle $( \pi _ { o } )$ sort by $c _ { i }$ ascending (incorrect answers first); this is the best achievable ordering

• Random $( \pi _ { \mathrm { r a n d } } )$ : uniformly random ordering, averaged over $T = 2 0 0$ independent trials.

The area under each rejection curve is

$$
\operatorname { A U C } ( \pi ) = \int _ { 0 } ^ { 1 } \operatorname { A c c } ( r ; \pi ) \mathrm { d } r ,\tag{14}
$$

approximated numerically with step size $\Delta r = 0 . 0 5$ . PRR normalises the uncertainty AUC relative to the oracle and random baselines:

$$
\mathrm { P R R } = \frac { \mathrm { A U C } ( \pi _ { u } ) - \mathrm { A U C } ( \pi _ { \mathrm { r a n d } } ) } { \mathrm { A U C } ( \pi _ { o } ) - \mathrm { A U C } ( \pi _ { \mathrm { r a n d } } ) } .\tag{15}
$$

PRR = 1 corresponds to perfect oracle rejection; $\mathrm { P R R } = 0$ corresponds to random rejection; negative values indicate that the uncertainty score is anti-correlated with error.

![](images/b63c276d551768a0083e4e6f805e551bed11e35ce018912e8aa13f5a223908c9.jpg)  
Figure 7: Example visualization of PRR.

## A.1.2 Spearman Rank Correlation.

To measure the monotonic relationship between an uncertainty score $u _ { i } \in \mathbb { R }$ and binary correctness $c _ { i } \in \{ 0 , 1 \}$ for a set of N questions, we compute the Spearman rank correlation

$$
\rho = \frac { \sum _ { i = 1 } ^ { N } ( r _ { i } - \bar { r } ) ( s _ { i } - \bar { s } ) } { \sqrt { \sum _ { i = 1 } ^ { N } ( r _ { i } - \bar { r } ) ^ { 2 } \ \sum _ { i = 1 } ^ { N } ( s _ { i } - \bar { s } ) ^ { 2 } } } ,\tag{16}
$$

where $r _ { i } = { \mathrm { r a n k } } ( u _ { i } )$ and $s _ { i } = { \mathrm { r a n k } } ( c _ { i } )$ are the average-tied ranks of the uncertainty scores and correctness labels, respectively, and ī,  denote their means. A negative ρ indicates that higher uncertainty is associated with lower correctness, which is the desired behaviour for a well-calibrated uncertainty estimate.

## A.2 Uncertainty Scores

## A.2.1 Argument Graph Uncertainty.

$U _ { \mathrm { A G U } } = H _ { \mathrm { n o r m } }$ as defined in Section 3.

## A.2.2 Consistency Uncertainty.

$U _ { \mathrm { c o n s } }$ measures disagreement across $S$ independent runs of the same model on the same question. Let $c _ { k }$ be the number of runs that predicted option k, and $\hat { p } _ { k } = c _ { k } / S$ the empirical vote fraction. The consistency uncertainty is the normalised Shannon entropy of the vote distribution:

$$
U _ { \mathrm { c o n s } } = \frac { - \sum _ { k = 1 } ^ { K } \hat { p } _ { k } \ln \hat { p } _ { k } } { \ln K } ,\tag{17}
$$

where the sum is taken only over options that received at least one vote. A value of 0 indicates unanimous agreement across runs; a value of 1 indicates maximally uniform disagreement.

## A.2.3 Token-Level Uncertainty.

$U _ { \mathrm { t o k e n } }$ is derived from the log-probabilities the model assigns to each answer-option letter at the final answer token position. Let $\ell _ { k } = \log p _ { \theta }$ (option k | context) be the log-probability of option k extracted from the model's top-κ logits. These are converted to a probability distribution via a softmax over the K options:

$$
p _ { k } = \frac { \exp ( \ell _ { k } ) } { \sum _ { j = 1 } ^ { K } \exp ( \ell _ { j } ) } .\tag{18}
$$

The token uncertainty is then the normalised Shannon entropy of this distribution:

$$
U _ { \mathrm { t o k e n } } = { \frac { - \sum _ { k = 1 } ^ { K } p _ { k } \ln p _ { k } } { \ln K } } .\tag{19}
$$

## B Additional Results

## B.1 Additional Benchmarks

Tables 3 and 4 report PRR on MMLU-Pro and TruthfulQA MC4 for the same signals as 1. Among the single signals, $U _ { \mathrm { A G U } }$ is the best on two of five MMLU-Pro models and two of nine TruthfulQA models, with $U _ { \mathrm { C o n s } }$ leading elsewhere and $U _ { \mathrm { T o k e n } }$ only once. The equal-weight combination $U _ { \mathrm { A G U } } + U _ { \mathrm { C o n s } }$ is the best signal for every model on both benchmarks, improving on the stronger of its two components by 0.10 PRR on average on MMLU-Pro and 0.09 on TruthfulQA. This further empirically supports our finding that UAGU is orthogonal to consistency-based uncertainty.

## B.2 NLI Edge Statistics

Fig. 8 provides summary statistics for GPQA diamond. Neutral dominates both edge types (65% of segment– segment, 78% of segment-option edges), as consecutive reasoning steps mostly add new content and most steps neither support nor rule out a given option. Segment-option edges are almost never entailment (1.6%) but contradiction in 20% of cases, so the per-option evidence rests on a few confident entailment edges from the concluding segments, while contradiction edges rule out the remaining options. The profile is generally stable across models; the segment-segment entailment share can varies notably (11% to 32%). The scorer is more decisive on segment–option edges (confidence 0.88 vs. 0.81, entropy 0.31 vs. 0.43); this is the channel that enters the Dirichlet evidence directly. The example in panels (c, d) shows a correctly answered trace with $U _ { \mathrm { A G U } } = 0 . 2 3$ : the two concluding segments entail the correct option at $p \geq 0 . 9 5$ and contradict the others at $p \geq 0 . 9 9$ , and the double-check segment $v _ { 4 }$ entails all preceding steps, giving it the highest coherence weight.

Table 3: PRR on MMLU-Pro. Mean ± std over seeds. Bold marks the best signal per model.
<table><tr><td>Model</td><td> $U _ { \mathrm { A G U } }$ </td><td> $U _ { \mathrm { T o k e n } }$ </td><td> $U _ { \mathrm { C o n s } }$ </td><td> $U _ { \mathrm { A G U } } + U _ { \mathrm { C o n s } }$ </td></tr><tr><td>GPT-OSS-120B</td><td> $0 . 5 4 \pm . 0 1$ </td><td> $0 . 0 5 \pm . 0 0$ </td><td> $0 . 4 8 \pm . 0 1$ </td><td> ${ \bf 0 . 6 6 \pm . 0 1 }$ </td></tr><tr><td>Nemotron-30B</td><td> $0 . 5 2 \pm . 0 1$ </td><td> $0 . 1 1 \pm . 0 1$ </td><td> $0 . 5 1 \pm . 0 0$ </td><td> ${ \bf 0 . 6 7 \pm . 0 0 }$ </td></tr><tr><td>Qwen3-30B-T</td><td> $0 . 4 3 \pm . 0 1$ </td><td> $0 . 0 5 \pm . 0 2$ </td><td> $0 . 4 9 \pm . 0 0$ </td><td> ${ \bf 0 . 6 3 \pm . 0 1 }$ </td></tr><tr><td>Phi-4-mini-R</td><td> $0 . 4 1 \pm . 0 1$ </td><td> $0 . 0 7 \pm . 0 0$ </td><td> $0 . 6 0 \pm . 0 1$ </td><td> ${ \bf 0 . 6 4 \pm . 0 0 }$ </td></tr><tr><td> $\mathrm { R 1 - L l a m a - } 7 0 \mathrm { B }$ </td><td> $0 . 4 0 \pm . 0 1$ </td><td> $0 . 0 4 \pm . 0 1$ </td><td> $0 . 5 1 \pm . 0 0$ </td><td> ${ \bf 0 . 6 3 \pm . 0 1 }$ </td></tr><tr><td>Mean</td><td>0.46</td><td>0.06</td><td>0.52</td><td>0.64</td></tr></table>

Table 4: PRR on TruthfulQA MC4. Mean ± std over seeds. Bold marks the best signal per model.
<table><tr><td>Model</td><td> $U _ { \mathrm { A G U } }$ </td><td> $U _ { \mathrm { T o k e n } }$ </td><td> $U _ { \mathrm { C o n s } }$ </td><td> $U _ { \mathrm { A G U } } + U _ { \mathrm { C o n s } }$ </td></tr><tr><td>Qwen3-30B-T</td><td> $0 . 4 8 \pm . 0 5$ </td><td> $0 . 1 3 \pm . 0 2$ </td><td> $0 . 3 7 \pm . 0 3$ </td><td> ${ \bf 0 . 6 0 \pm . 0 3 }$ </td></tr><tr><td>R1-Llama-70B</td><td> $0 . 4 2 \pm . 0 4$ </td><td> $0 . 3 3 \pm . 0 1$ </td><td> $0 . 4 4 \pm . 0 2$ </td><td> ${ \bf 0 . 5 9 \pm . 0 2 }$ </td></tr><tr><td>Qwen3-4B-T</td><td> $0 . 3 9 \pm . 0 4$ </td><td> $0 . 1 8 \pm . 0 4$ </td><td> $0 . 3 6 \pm . 0 3$ </td><td> ${ \bf 0 . 5 1 \pm . 0 4 }$ </td></tr><tr><td>Phi-4-mini-R</td><td> $0 . 3 8 \pm . 0 3$ </td><td> $0 . 1 1 \pm . 0 5$ </td><td> $0 . 4 6 \pm . 0 2$ </td><td> ${ \bf 0 . 5 5 \pm . 0 2 }$ </td></tr><tr><td>GPT-OSS-120B</td><td> $0 . 3 5 \pm . 0 3$ </td><td> $0 . 1 1 \pm . 0 2$ </td><td> $0 . 3 7 \pm . 0 3$ </td><td> ${ \bf 0 . 4 8 \pm . 0 3 }$ </td></tr><tr><td>Nemotron-30B</td><td> $0 . 3 4 \pm . 0 4$ </td><td> $0 . 1 7 \pm . 0 3$ </td><td> $0 . 4 6 \pm . 0 1$ </td><td> ${ \bf 0 . 5 1 \pm . 0 1 }$ </td></tr><tr><td>R1-Qwen-14B</td><td> $0 . 3 4 \pm . 0 3$ </td><td> $0 . 4 9 \pm . 0 3$ </td><td> $0 . 4 7 \pm . 0 1$ </td><td> ${ \bf 0 . 5 5 \pm . 0 2 }$ </td></tr><tr><td>GPT-OSS-20B</td><td> $0 . 2 9 \pm . 0 6$ </td><td> $0 . 1 1 \pm . 0 3$ </td><td> $0 . 4 5 \pm . 0 3$ </td><td> ${ \bf 0 . 5 3 \pm . 0 4 }$ </td></tr><tr><td>R1-Llama-8B</td><td> $0 . 2 4 \pm . 0 5$ </td><td> $0 . 2 5 \pm . 0 2$ </td><td> $0 . 3 3 \pm . 0 2$ </td><td> ${ \bf 0 . 4 1 \pm . 0 5 }$ </td></tr><tr><td>Mean</td><td>0.36</td><td>0.21</td><td>0.41</td><td>0.53</td></tr></table>

## B.3 Analysis of NLI Robustness

We measure how errors of the NLI scorer at the level of individual edges propagate to $U _ { \mathrm { A G U } }$ . For each of the nine GPQA Diamond models we take the scorer's class probabilities for every segment-segment and segment-option edge, perturb them, and recompute $U _ { \mathrm { A G U } }$ from the perturbed edges. Each perturbation is applied at several noise levels with three random seeds per level, and we report PRR as a fraction of the PRR obtained from the unperturbed edges, averaged over seeds and models.

Random label flips. With probability ε per edge, the scorer's predicted class is replaced by one of the two other classes chosen uniformly at random, for ε from 0.05 to 0.4. This simulates hard classification errors that are uncorrelated with the content of the edge. PRR decays smoothly and stays positive for every model: 90% of the clean PRR is retained at ε=0.05, 78% at 0.1, 62% at 0.2 and 52% when 40% of all edges carry a wrong label. The decline flattens beyond ε=0.15, since the remaining signal is carried by the majority of edges that are still correct.

Probability jitter. Gaussian noise $\eta \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ is added to the class log-probabilities of every edge, which are then renormalised, $\tilde { p } \propto \exp ( \log p + \eta )$ , for σ from 0.25 to 2. This simulates a miscalibrated scorer: the predicted class changes only where the margin between classes is smaller than the noise. PRR retention is 97% at σ=0.5, 85% at σ=1 and 65% at σ=2, where the noise exceeds the typical margin and a large share of predicted classes change. $U _ { \mathrm { A G U } }$ thus relies on the scorer's ordinal judgment rather than on calibrated probabilities.

Both quantities that enter the Dirichlet parameters are averages over many edges, so a single misclassified edge shifts the per-option evidence only marginally.

## B.4 Windowed vs. All-to-All Segment Graph

Scoring every ordered segment pair is quadratic in the number of segments and could become a bottleneck for long traces. The windowed variant of Section 3.4 scores only pairs with $| i - j | \leq w / 2$ so that with w = 10 each segment is compared against its five preceding and five following segments in both directions. All segment-option edges are still scored, and Conflict, Asym and the global coherence become averages over the windowed edge set; the local trajectory term is unaffected. We compare the two graphs on GPQA Diamond for all nine models (Table 5). Both columns are computed from the same NLI outputs of a single seed per model, so this is a paired comparison; the absolute PRR values therefore differ from the five-seed means of Table 1, but lie within the reported standard deviations.

The window uses 18 times fewer segment-segment pairs while PRR is essentially unchanged, 0.42 against 0.42 on average and at most 0.01 apart for any model. We attribute this to the locality of logical structure in a reasoning trace: a step supports or attacks the steps in its immediate neighbourhood, while distant pairs are predominantly neutral and contribute little signal. The NLI stage can therefore stay linear in the number of segments as traces grow, without affecting the quality of the uncertainty estimate.

![](images/6b88e084476a8c67d44b799b26308fe790620cd32deb401e86c7329e0f48970a.jpg)

![](images/92af3145955fd6ab699b6ac4b7487547fa053d40fe35619a12f4011da1b00bdf.jpg)

(c) Example: one reasoning trace (GPT-OSS-20B, GPQA Diamond)  
(d) All-to-all edge graph  
![](images/c07a3c922c0d1503041df9f4e3987126aac1ee62f211079a59bc6b9cfdc74dfb.jpg)  
Figure 8: NLI edge statistics. (a, b) Share of edges by predicted class per model for segment-segment edges within the window (w=10) and segment-option edges; conf. and entr. are the mean probability of the predicted class and the mean three-class entropy normalised by log 3. (c) A complete fivesegment trace of GPT-OSS-20B with the scorer's actual outputs; segment text is abbreviated. Adjacent segment-segment edges are shown along the chain. For visibility, only segment-option edges with $p \geq 0 . 7 5$ are drawn as arrows; $v _ { 1 }$ and $v _ { 2 }$ are neutral towards all options and $v _ { 5 }$ towards $o _ { A }$ (d) All ordered segment-segment edges of the same trace; row is premise, column is hypothesis, shading is the scorer's confidence.

Table 5: Windowed (w=10) vs. all-to-all segment graph on GPQA Diamond. PRR and number of scored segment-segment pairs, one seed per model.
<table><tr><td colspan="2"></td><td colspan="2">PRR</td><td colspan="2">Pairs</td></tr><tr><td>Model</td><td>Mean segs</td><td>Window</td><td>All-to-all</td><td>Window</td><td>All-to-all</td></tr><tr><td>GPT-OSS-20B</td><td>69.4</td><td>0.67</td><td>0.67</td><td>94.9k</td><td>1.64M</td></tr><tr><td>GPT-OSS-120B</td><td>61.8</td><td>0.59</td><td>0.60</td><td>102.3k</td><td>1.43M</td></tr><tr><td>Nemotron-30B</td><td>64.4</td><td>0.54</td><td>0.53</td><td>85.3k</td><td>1.02M</td></tr><tr><td>Qwen3-30B-T</td><td>99.2</td><td>0.48</td><td>0.47</td><td>190.4k</td><td>3.51M</td></tr><tr><td>R1-Qwen-14B</td><td>92.8</td><td>0.34</td><td>0.33</td><td>168.8k</td><td>2.86M</td></tr><tr><td>Qwen3-4B-T</td><td>59.7</td><td>0.33</td><td>0.33</td><td>111.2k</td><td>1.48M</td></tr><tr><td>R1-Llama-70B</td><td>123.7</td><td>0.32</td><td>0.32</td><td>239.1k</td><td>4.53M</td></tr><tr><td>R1-Llama-8B</td><td>146.9</td><td>0.28</td><td>0.27</td><td>274.9k</td><td>6.88M</td></tr><tr><td>Phi-4-mini-R</td><td>98.6</td><td>0.28</td><td>0.27</td><td>189.3k</td><td>3.45M</td></tr><tr><td>Mean / total</td><td>90.7</td><td>0.42</td><td>0.42</td><td>1.46M</td><td>26.81M</td></tr></table>

## B.5 Trace Length versus Graph Signal

We test whether $U _ { \mathrm { A G U } }$ carries information beyond the length of the reasoning trace. For each GPQA Diamond model we compute three uncertainty scores on the same items and evaluate each by PRR: (i) the number of segments n used directly as the score, (ii) $U _ { \mathrm { A G U } }$ , and (iii) $U _ { \mathrm { A G U } }$ with the length component removed. For (iii) we convert both $U _ { \mathrm { A G U } }$ and n to ranks, fit a linear regression of rank $\left( U _ { \mathrm { A G U } } \right)$ on $\operatorname { r a n k } ( n )$ , and use the residuals as the score; this removes any monotone dependence on length while keeping the remaining ordering of $U _ { \mathrm { A G U } }$ We also report the Spearman correlation ρ between $U _ { \mathrm { A G U } }$ and n.

Segment count alone attains a mean PRR of 0.33 against 0.45 for $U _ { \mathrm { A G U } }$ , and is the weaker signal on seven of nine models. The two scores are moderately correlated $( \rho = 0 . 4 3$ on average). Some correlation is expected. since harder questions elicit both longer traces and more internal conflict, so length and graph structure partly respond to the same underlying difficulty. Correlation does not imply redundancy, however: the residual score retains a PRR of 0.33, so the majority of the graph signal is independent of length. The residual is smallest for GPT-OSS-120B and R1-Llama-70B, the two models with the longest traces, where length accounts for the largest share of the signal.

Table 6: Trace length versus graph signal on GPQA Diamond. PRR of the segment count used as an uncertainty score, of $U _ { \mathrm { A G U } }$ , and of $U _ { \mathrm { A G U } }$ after rank-regressing out the segment count; $\rho$ is the Spearman correlation between $U _ { \mathrm { A G U } }$ and segment count. Mean ± std over seeds; bold marks the better of the two single scores.
<table><tr><td>Model</td><td>Segment count</td><td> $U _ { \mathrm { A G U } }$ </td><td> $U _ { \mathrm { A G U } } \mid \mathrm { l e n g t h }$ </td><td> $\rho$ </td></tr><tr><td>GPT-OSS-20B</td><td> $0 . 5 2 \pm . 0 6$ </td><td> ${ \bf 0 . 6 7 \pm . 0 6 }$ </td><td> $0 . 4 9 \pm . 0 8$ </td><td> $0 . 4 5 \pm . 0 7$ </td></tr><tr><td>Nemotron-30B</td><td> $0 . 4 6 \pm . 1 4$ </td><td> ${ \bf 0 . 6 0 \pm . 0 8 }$ </td><td> $0 . 4 5 \pm . 1 1$ </td><td> $0 . 5 2 \pm . 0 7$ </td></tr><tr><td>GPT-OSS-120B</td><td> ${ \bf 0 . 6 5 \pm . 0 3 }$ </td><td> $0 . 5 8 \pm . 0 2$ </td><td> $0 . 2 8 \pm . 0 2$ </td><td> $0 . 5 9 \pm . 0 1$ </td></tr><tr><td>Qwen3-30B-T</td><td> $0 . 4 1 \pm . 0 7$ </td><td> ${ \bf 0 . 5 0 \pm . 0 5 }$ </td><td> $0 . 3 8 \pm . 0 5$ </td><td> $0 . 5 4 \pm . 0 4$ </td></tr><tr><td>R1-Qwen-14B</td><td> $0 . 2 3 \pm . 0 9$ </td><td> ${ \bf 0 . 3 7 \pm . 0 4 }$ </td><td> $0 . 3 1 \pm . 0 6$ </td><td> $0 . 2 9 \pm . 0 6$ </td></tr><tr><td>Qwen3-4B-T</td><td> $0 . 1 5 \pm . 0 4$ </td><td> ${ \bf 0 . 3 5 \pm . 0 2 }$ </td><td> $0 . 2 9 \pm . 0 5$ </td><td> $0 . 3 6 \pm . 0 7$ </td></tr><tr><td>R1-Llama-70B</td><td> ${ \bf 0 . 3 4 \pm . 0 3 }$ </td><td> $0 . 3 3 \pm . 0 4$ </td><td> $0 . 2 2 \pm . 0 5$ </td><td> $0 . 4 6 \pm . 0 4$ </td></tr><tr><td>Phi-4-mini-R</td><td> $0 . 0 6 \pm . 0 7$ </td><td> ${ \bf 0 . } 3 2 \pm { \bf . 0 6 }$ </td><td> $0 . 3 2 \pm . 0 8$ </td><td> $0 . 2 9 \pm . 0 7$ </td></tr><tr><td>R1-Llama-8B</td><td> $0 . 1 3 \pm . 0 6$ </td><td> ${ \bf 0 . 2 8 \pm . 0 4 }$ </td><td> $0 . 2 2 \pm . 0 4$ </td><td> $0 . 3 1 \pm . 0 9$ </td></tr><tr><td>Mean</td><td>0.33</td><td>0.45</td><td>0.33</td><td>0.43</td></tr></table>

## B.6 Ablation on Uncertainty Score Combination

![](images/497c841aae945ba1abe55a32f88b729ee210660d9b414effd85be72ed1bbda11.jpg)  
Figure 9: Different uncertainty score combination techniques aggregated over models.

In this section we provide additional experimental results. $u _ { \mathrm { a g u } }$ denotes graph-based uncertainty and $u _ { \mathrm { c o n s } }$ denotes a consistency uncertainty signal (either normalized vote entropy or Laplacian-based consistency uncertainty). For each model, combinations are computed at the question level.

Normalization primitives. Given vectors of per-question uncertainties, we define:

$$
z _ { \mathrm { a g u } } = \frac { u _ { \mathrm { a g u } } - \mu _ { \mathrm { a g u } } } { \sigma _ { \mathrm { a g u } } } , \qquad z _ { \mathrm { c o n s } } = \frac { u _ { \mathrm { c o n s } } - \mu _ { \mathrm { c o n s } } } { \sigma _ { \mathrm { c o n s } } } ,\tag{20}
$$

$$
m _ { \mathrm { a g u } } = \frac { u _ { \mathrm { a g u } } - \operatorname* { m i n } ( u _ { \mathrm { a g u } } ) } { \operatorname* { m a x } ( u _ { \mathrm { a g u } } ) - \operatorname* { m i n } ( u _ { \mathrm { a g u } } ) } , m _ { \mathrm { c o n s } } = \frac { u _ { \mathrm { c o n s } } - \operatorname* { m i n } ( u _ { \mathrm { c o n s } } ) } { \operatorname* { m a x } ( u _ { \mathrm { c o n s } } ) - \operatorname* { m i n } ( u _ { \mathrm { c o n s } } ) } ,\tag{21}
$$

$$
r _ { \mathrm { a g u } } = \mathrm { r a n k p c t } ( u _ { \mathrm { a g u } } ) ,
$$

$$
r _ { \mathrm { c o n s } } = \mathrm { r a n k p c t } ( u _ { \mathrm { c o n s } } ) ,\tag{22}
$$

where rankpct(·) is the percentile rank in [0, 1]. If a denominator is zero (constant signal), the corresponding normalized values are set to 0.

Laplacian consistency uncertainty. Given m sampled predictions (votes) $\{ v _ { i } \} _ { i = 1 } ^ { m }$ , define the agreement matrix

$$
W _ { i j } = \mathbf { 1 } [ v _ { i } = v _ { j } ] .
$$

Let $\begin{array} { r } { d _ { i } = \sum _ { j = 1 } ^ { m } W _ { i j } } \end{array}$ and $D = \operatorname { d i a g } ( d _ { 1 } , \dots , d _ { m } )$ . The normalized graph Laplacian is

$$
L = I - D ^ { - 1 / 2 } W D ^ { - 1 / 2 } .
$$

If $\{ \lambda _ { k } \} _ { k = 1 } ^ { m }$ are eigenvalues of $L ,$ the Laplacian consistency uncertainty is

$$
u _ { \mathrm { c o n s } } ^ { \mathrm { l a p } } = \sum _ { k = 1 } ^ { m } \operatorname* { m a x } ( 0 , 1 - \lambda _ { k } ) .
$$

Blending methods. We evaluate the following fused uncertainty scores:

$$
\mathrm { E q u a l - w e i g h t z - b l e n d ( b l e n d \ 5 0 \_ 5 0 ) } ; \quad s _ { \mathrm { b l e n d 5 0 } } = \textstyle { \frac { 1 } { 2 } } z _ { \mathrm { a g u } } + \textstyle { \frac { 1 } { 2 } } z _ { \mathrm { c o n s } } ,\tag{23}
$$

$$
\mathrm { T u n c d - } \alpha \mathrm { z } \mathrm { b l e n d ~ ( b l e n d \_ t u n e d \_ { \alpha } a l p h a \_ { \alpha } } )  ; s _ { \mathrm { b l e n d \_ } } = \alpha z _ { \mathrm { a g u } } + ( 1 - \alpha ) z _ { \mathrm { c o n s } } , \alpha \in [ 0 , 1 ] ,\tag{24}
$$

$$
\mathrm { R a n k \ f u s i o n \ ( r a n k \_ f u s i o n ) } { : } \quad s _ { \mathrm { r a n k } } = { \textstyle { \frac { 1 } { 2 } } } r _ { \mathrm { a g u } } + { \textstyle { \frac { 1 } { 2 } } } r _ { \mathrm { c o n s } } ,\tag{25}
$$

$$
\mathrm { M a x - z \ f u s i o n \ ( m a x \_ z ) } ; \quad s _ { \mathrm { m a x \ } z } = \mathrm { m a x } ( z _ { \mathrm { a g u } } , z _ { \mathrm { c o n s } } ) ,\tag{26}
$$

$$
\mathrm { G e o m e t r i c ~ m i n - m a x ~ f u s i o n ~ ( g e o m \_ m i n m a x ) } ; \quad s _ { \mathrm { g e o m } } = \sqrt { m _ { \mathrm { a g u } } m _ { \mathrm { c o n s } } } .\tag{27}
$$

Tuning of $\alpha .$ For the tuned blend, we search α $\in \{ 0 . 0 , 0 . 1 , \ldots , 1 . 0 \}$ and select

$$
\alpha ^ { \star } = \arg \operatorname* { m a x } _ { \alpha } \frac { 1 } { | \mathcal { M } | } \sum _ { m \in \mathcal { M } } \mathrm { P R R } _ { m } ( \alpha ) ,\tag{28}
$$

where $\mathrm { P R R } _ { m } ( \alpha )$ is the PRR of model m using sblend,α·

Consistency variants. All formulas above are applied with:

$u _ { \mathrm { c o n s } } = u _ { \mathrm { c o n s } }$ (as defined in A), and

$$
\bullet u _ { \mathrm { c o n s } } = u _ { \mathrm { c o n s } } ^ { \mathrm { l a p } } ,
$$

yielding the corresponding “\_laplacian" method variants in the latter case in Fig.B.6.

## B.7 Uncertainty Score Distributions

Figure 10 offers uncertainty scores distributions for the models not shown in the main manuscript. Distributions are consistent across all models.

## C Computational Costs

Notation. Let P denote the parameter count of the evaluated reasoning model, $L _ { p }$ the prompt length and L the length of one reasoning trace T in tokens, and s the number of sampled traces used by the consistency baseline (s = 5 in our experiments). We measure compute in parameter-token products, which is proportional to FLOPs for a transformer forward pass (≈ 2P FLOPs per token).

![](images/e0d25ea937fa8396624a4b1c3c7d126bd7203f062f21bb2c79794e5f69a3ffcd.jpg)  
Figure 10: Histograms of uncertainty distributions per model and method.

Consistency-based uncertainty. Every sample requires a full forward pass of the evaluated model over the prompt and an autoregressive decode of a fresh trace $\mathbf { \hat { \boldsymbol { T } } } .$

$$
{ \cal C } _ { \mathrm { s a m p l e } } = P \left( L _ { p } + L \right) \in \mathcal { O } ( P L ) , \qquad L _ { p } \ll L .\tag{29}
$$

Since one trace is generated in any case to obtain the answer, the overhead of $U _ { \mathrm { C o n s } }$ is the cost of the $s - 1$ additional traces,

$$
C _ { \mathrm { C o n s } } \ = \ ( s - 1 ) P \left( L _ { p } + L \right) \in \mathcal { O } ( s P L ) .\tag{30}
$$

The overhead therefore scales linearly with the size of the evaluated model and with trace length, and is incurred on the evaluated model itself, i.e. at its full GPU footprint.

Argument Graph Uncertainty. AGU operates post hoc on the single trace T that is generated anyway and makes no further calls to the evaluated model. Its overhead is incurred entirely on two small auxiliary models: a segmenter with $p _ { \mathrm { s e g } }$ parameters and an NLI encoder with $p _ { \mathrm { N L I } }$ parameters. Let T be split into n segments of average length $\ell = L / n$ tokens, and let K denote the number of answer options.

Segmentation. With the instruction-tuned segmenter, $T$ is read once as a prompt and a short list of segment boundaries is emitted,

$$
C _ { \mathrm { s e g } } \ = \ p _ { \mathrm { s e g } } \left( L + \mathcal { O } ( n ) \right) \ \in \ \mathcal { O } ( p _ { \mathrm { s e g } } L ) .\tag{31}
$$

With the deterministic paragraph split this stage requires no model call and $C _ { \mathrm { s e g } } = 0 .$

NLI scoring. Every scored pair is one encoder pass over a premise and a hypothesis of ≈ 2l tokens in total, so the cost is the number of pairs times $2 \ell p _ { \mathrm { N L I } }$ . The segment-option graph always contributes nK pairs. The segment-segment graph contributes $n ( n - 1 )$ pairs for the all-to-all variant and ≈ n w pairs for a window of width w:

$$
C _ { \mathrm { N L I } } ^ { \mathrm { w i n d o w } } = \left( n w + n K \right) 2 \ell p _ { \mathrm { N L I } } = 2 \left( w + K \right) p _ { \mathrm { N L I } } L \in \mathcal { O } \big ( p _ { \mathrm { N L I } } \left( w + K \right) L \big ) ,\tag{32}
$$

$$
C _ { \mathrm { N L I } } ^ { \mathrm { a l l } } = \big ( n ( n - 1 ) + n K \big ) 2 \ell p _ { \mathrm { N L I } } \approx 2 p _ { \mathrm { N L I } } \frac { L ^ { 2 } } { \ell } \ \in \ \mathcal { O } \big ( p _ { \mathrm { N L I } } L ^ { 2 } / \ell \big ) .\tag{33}
$$

The windowed cost is linear in L and independent of n: a finer segmentation produces more pairs, but each pair is proportionally shorter. Only the all-to-all graph is quadratic in trace length.

Total overhead and comparison. Combining the two stages gives four configurations,

$$
\begin{array} { r } { C _ { \mathrm { A G U } } ~ = ~ \underbrace { \left[ p _ { \mathrm { s e g } } L \right] } _ { \mathrm { L L M s e g m e n t e r ~ o n l y } } ~ + ~ \left\{ { 2 \left( w + K \right) p _ { \mathrm { N L I } } { L } } \right. ~ \mathrm { w i n d o w } ~ w , } \end{array}\tag{34}
$$

For the windowed graph, the ratio to the consistency overhead of Eq. (30) is independent of trace length,

$$
\frac { C _ { \mathrm { A G U } } } { C _ { \mathrm { C o n s } } } \ = \ \frac { \left[ p _ { \mathrm { s e g } } \right] + 2 \left( w + K \right) p _ { \mathrm { N L I } } } { \left( s - 1 \right) P } \ \in \ \mathcal { O } \left( \frac { p _ { \mathrm { N L I } } } P \right) ,\tag{35}
$$

and decreases as the evaluated model grows.

## C.1 Hyper Parameter Ablation

We conduct a sweep over κ and α0 with $\gamma _ { c }$ and $\gamma _ { a }$ fixed at their default values of (1.2, 1.0), selecting the combination that maximises PRR per model. Table 7 reports the best-found configuration alongside, with bold entries indicating models that improve over the default $\bar { ( } \kappa = 1 2 . 0 , \alpha _ { 0 } = 1 . 0 )$ setting.

Five of nine models already achieve their best performance at the default hyperparameters. The remaining four models show moderate gains under tuned configurations, typically favouring a larger κ and smaller $\alpha _ { 0 } .$ which concentrate the Dirichlet evidence more aggressively and reduce the influence of the symmetric prior.

The sensitivity plots in Figure 11 shows PRR is largely stable across a broad range of κ and αo values, with performance degrading only at extreme settings. $\gamma _ { \mathrm { c o n f i c t } }$ and $\gamma _ { \mathrm { a s y m m } }$ exhibit similarly flat sensitivity curves, indicating that the penalty terms in Equation 9 are not critical to tune precisely. Taken together, these results support the use of a single fixed hyperparameter configuration across all models, as adopted in the main experiments, while careful choice of hyperparameters can lead to moderate improvments in some cases.

![](images/689564b0a11f41e49ddef14c28c3204ca5e44a91f4739553466550562e6d7214.jpg)  
Figure 11: Influence of hyperparameters evaluated for every model.

Table 7: Results of sweep over $\kappa , \alpha _ { 0 }$ and fixed $( \gamma _ { c } , \gamma _ { a } )$ . Improved models are bold.
<table><tr><td>Model</td><td>PRR↑</td><td> $\rho \downarrow$ </td><td> $\left( \kappa , \alpha _ { 0 } \right)$ </td><td> $( \gamma _ { c } , \gamma _ { a } )$ </td></tr><tr><td>GPT-OSS-20B</td><td>0.66</td><td>-0.39</td><td>(27.68, 0.10)</td><td>(1.20, 1.00)</td></tr><tr><td>Nemotron-30B</td><td>0.65</td><td>-0.37</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr><tr><td>GPT-OSS-120B</td><td>0.68</td><td>0.26</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr><tr><td>Qwen3-30B-T</td><td>0.51</td><td>-0.35</td><td>(37.95, 0.10)</td><td>(1.20, 1.00)</td></tr><tr><td>Qwen3-4B-T</td><td>0.51</td><td>-0.39</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr><tr><td>Phi-4-mini-R</td><td>0.43</td><td>-0.35</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr><tr><td>R1-Qwen-14B</td><td>0.41</td><td>-0.32</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr><tr><td>R1-Llama-70B</td><td>0.36</td><td>-0.25</td><td>(33.84, 0.87)</td><td>(1.20, 1.00)</td></tr><tr><td>R1-Llama-8B</td><td>0.29</td><td>-0.27</td><td>(12.00, 1.00)</td><td>(1.20, 1.00)</td></tr></table>

## D Segmentation Prompt and NLI Scorer Input

In this section we provide the prompt we used to segment reasoning chains into segments $\mathcal { V } = \{ v _ { 1 } , \ldots , v _ { n } \}$ and the design of NLI-Scorer inputs.

Prompt used for indexed segmentation   
You segment reasoning traces into logical blocks for argument-graph analysis. The reasoning trace has   
already been split into numbered units using a rule-based preprocessor. Return only compact segment   
ranges over those numbered units. Return exactly one valid JSON object with this schema and nothing else:   
{ "segment\_ranges": [[start\_unit\_index, end\_unit\_index], ...] }   
Rules:   
1. Do not rewrite, paraphrase, or quote the reasoning text.   
2. Each range is 1-based and inclusive, referring to the numbered units below.   
3. Ranges must be contiguous, non-overlapping, and cover every unit exactly once.   
4. Group consecutive units that serve the same immediate reasoning purpose into one segment.   
5. Start a new segment when the reasoning changes function (setup, sub-conclusion, computation, option   
evaluation, final answer, etc.).   
6. Keep closely related false starts or self-corrections together if they resolve the same local issue.   
7. If uncertainty resolves into a conclusion used in the next step, start a new segment at that boundary.   
8. Keep brief scene-setting with the reasoning move it introduces unless it is clearly its own step.   
9. A segment may span multiple units forming one coherent logical move.   
10. Use as many segments as the reasoning naturally requires.   
11. No markdown, no code fences, no extra keys, no prose outside the JSON.   
Reasoning trace to segment: [1] ... [2] ...

As mentioned earlier, segmentation by an instruct model can be substituted by a simple deterministic split without losing significant calibration quality. We can simply divide the trace at blank lines (the regular expression \n\s\*\n+, i.e. two or more consecutive line breaks), yielding one segment per paragraph.

We designed the input to our NLI-Scorer as follows:

Input format for NLI edge scoring   
Segment-segment edge (vi, vj):   
Premise text of segment vi   
Hypothesis text of segment vj   
Segment-option edge (vi, hk):   
Premise text of segment vi   
Hypothesis The answer is Option X: [option text]   
Example (segment v3 of Fig. 8c against option A):   
Premise The Skyrmion is a topological excitation, not a Goldstone boson.   
So the answer is Skyrmion, choice A.   
Hypothesis The answer is Option A: Skyrmion   
Output (pE, pN, pc) = (0.98, 0.02, 0.00) </parameter>

## E Computing Infrastructure

Experiments were run on 2 NVIDIA H200 GPUs.