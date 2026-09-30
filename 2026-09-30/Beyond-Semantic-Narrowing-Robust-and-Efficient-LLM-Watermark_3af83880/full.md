# Beyond Semantic Narrowing: Robust and Efficient LLM Watermarking with Hamming Neighborhoods

Zewen Sun<sup>1,2</sup> Tongyang Zhao<sup>3</sup> Liyao Xiang<sup>1,2\*</sup> Mingxuan Ma<sup>1</sup> Lingzhe Wang<sup>1</sup> Zhiyuan Li<sup>1</sup>

<sup>1</sup>Shanghai Jiao Tong University <sup>2</sup>Shanghai Innovation Institute <sup>3</sup>Northwest Polytechnical University, Xi’an

zwsun@sjtu.edu.cn zhaotongyang@mail.nwpu.edu.cn xiangliyao08@sjtu.edu.cn ru.jiang@sjtu.edu.cn wanglingzhe@sjtu.edu.cn willmaths@sjtu.edu.cn

## Abstract

Semantic watermarking improves robustness against watermark removal attacks by embedding detectable signals into sentence-level representations. However, existing watermarking methods typically impose watermark-specific semantic preferences on generated sentences without explicitly accounting for the highly non-uniform and context-dependent semantic preference of LLM generation. When these two preferences are poorly aligned, many natural continuations become incompatible with the watermark, causing semantic narrowing: reduced semantic freedom, increased resampling cost, and potential degradation on tasks with strict semantic requirements. To alleviate this problem, we propose HammingMark, which uses the semantic hash of the preceding sentence as a dynamic center and accepts candidates whose hashes fall within its Hamming neighborhood. Defining watermark validity over a Hamming neighborhood in compact hash space retains a larger fraction of naturally likely semantic continuations. The coarse many-to-one hash mapping further allows diverse semantic realizations to remain watermark-valid. Experiments on C4 and BookSum show that HammingMark achieves strong robustness, high detectability, and near-unwatermarked generation quality, requiring only 2.2 sampled candidates per accepted sentence—a 72.8% reduction compared with the most sampling-efficient existing method. On more complex tasks with strict semantic constraints, HammingMark achieves the highest detection rates with the highest or tied-highest ROUGE-L scores, demonstrating its effectiveness in balancing watermark detectability and generation quality under constrained generation settings.

## 1 Introduction

Large language models (LLMs) have demonstrated remarkable capabilities in generating fluent and coherent text, which simultaneously raises growing concerns about content provenance, misuse, and copyright protection (Liang et al., 2026; Xu et al., 2025; Casper et al., 2026; Yang et al., 2025). Text watermarking provides a promising solution by embedding statistically detectable signals into model-generated content without altering its intended functionality (Kirchenbauer et al., 2023; Dathathri et al., 2024; Liu et al., 2026). However, conventional token-level watermarking methods encode such signals by manipulating token distributions and are therefore vulnerable to paraphrasing, synonym substitution, and other semantic-preserving attacks (Lu et al., 2024; Hu et al., 2024; Sun et al., 2026). Semantic-level watermarking has recently emerged as a more robust alternative against semantic-preserving attacks by shifting the sentence, rather than the token distributions (Liu et al., 2024; Hou et al., 2024a; Ren et al., 2024), hence is more attractive to reliable provenance verification of LLM-generated text.

Despite their improved robustness, existing semantic watermarking methods inevitably experience semantic narrowing: they typically sample multiple candidate sentences until finding one that falls into a predefined semantic subspace. For example, bucket-based methods require the generated sentence to fall into a designated semantic region (Hou et al., 2024a;b), while similarity-based methods constrain its embedding to a predefined similarity interval (Dabiriaghdam & Wang, 2025; Zhang et al., 2025). Multi-dimensional semantic partitioning methods similarly favor sentences located in watermark-specific regions of the embedding space (Huo et al., 2026). Although implemented differently, these methods impose a watermark-specific semantic preference that may not align with the model’s natural generation preference. Poor alignment can reject otherwise plausible, high-probability continuations, leading to inefficient, repeated sampling and potential semantic distortion. This problem is particularly severe in constrained tasks, where the set of acceptable continuations is already limited.

Our core observation is that the natural next-sentence generation is typically not uniformly distributed over the semantic space, but puts much weights on a context-dependent set of plausible semantic continuations (Meister et al., 2023; Farquhar et al., 2024; Aichberger et al., 2025). Previous semantic watermarking methods typically predefine a randomly positioned semantic subspace as the watermark-valid region, without considering whether this subspace aligns with the context-dependent semantic space favored by natural next-sentence generation.

Motivated by this principle, we propose HammingMark, which uses the semantics of the preceding sentence to dynamically determine which next-sentence candidates are compatible with the watermark. Rather than steering generation toward a fixed, context-independent semantic region, HammingMark adapts the watermark constraint to the local context already established by the generated text, making it more likely to preserve continuations that the model naturally prefers. Within this context-aligned design, watermark selectivity operates on coarse semantic hash codes, whose many-to-one mapping weakens the coupling between watermark validity and specific semantic content. The constraint thus remains selective without prescribing a single semantic direction, allowing multiple semantically distinct continuations to remain valid and preserving flexibility for discourse changes such as elaboration, contrast, and topic shifts.

Our contributions are threefold: 1) A unified analysis of semantic narrowing. We provide the first systematic analysis of semantic narrowing, a common limitation of existing semantic watermarking methods. 2) A local-context-aligned semantic acceptance constraint for watermark embedding. We propose HammingMark, which centers the Hamming acceptance neighborhood on the semantic code of the preceding sentence, allowing the watermark constraint to better align with naturally plausible continuations while preserving semantic flexibility. 3) High-fidelity, efficient, and robust semantic watermarking. Experiments on open-ended and constrained generation tasks show that HammingMark maintains strong detectability and robustness while substantially reducing semantic restriction and resampling overhead, requiring only 2.2 sampled candidates per accepted sentence—a 72.8% reduction compared with the most sampling-efficient existing method.

## 2 Related Work

SemStamp (Hou et al., 2024a) partitions the semantic space using locality-sensitive hashing, while K-SemStamp (Hou et al., 2024b) uses clustering-based regions. Both restrict generation to selected valid regions without accounting for the nonuniform, context-dependent distribution of natural continuations. These regions may therefore exclude much of the probability mass assigned to plausible continuations, causing semantic narrowing through misalignment with natural generation. SimMark (Dabiriaghdam & Wang, 2025) considers inter-sentence relations but requires consecutive sentences to satisfy a predefined similarity interval. However, natural discourse does not maintain a uniform similarity range: contrast, elaboration, and topic shifts may require continuations outside this interval, which the constraint consequently rejects. This interval-based criterion can also be sensitive to paraphrasing, as changes to both sentences may shift their measured similarity across the acceptance boundary even when their meanings are largely preserved.

PMark (Huo et al., 2026) introduces a proxy-function-based framework with multi-channel constraints. Its online variant offers strong theoretical guarantees through a distortion-free construction, but its substantial verification cost and demanding requirements for generator access, reproducible inference, and sentence-position alignment severely limit its practical applicability. Specifically, verification requires repeated sentence-level generation to reconstruct channel-wise medians, whose estimates may shift when prompts, context, or inference settings differ. Moreover, the strict sentence–key alignment requirement can render detection infeasible when sentence truncation or missing prompts disrupt this correspondence. A detailed discussion is provided in Appendix A. Its offline variant replaces dynamically estimated channel-wise medians with a fixed zero prior, eliminating generator queries but losing the exact distortion-free guarantee. Rather than rejecting candidates against an acceptance threshold, it selects from a sampled pool the candidate with the highest agreement with prescribed channel-wise random signs. The random projection matrix and target signs jointly impose fixed preferences in semantic space, concentrat ing selection probability on candidates favored by these preferences. Consequently, naturally plausible but lower-scoring continuations are systematically disadvantaged, inducing semantic narrowing even without explicit rejection sampling.

![](images/fcc714af25f739a910f1add159d97b48772eda626aec80a2a978c75b49a358d5.jpg)  
Figure 1: Overview of HammingMark. (a) Existing semantic watermarks may cause semantic narrowing when predefined watermark regions poorly overlap with natural generation. (b) HammingMark uses transition-centered Hamming neighborhoods for better alignment with natural generation, while coarse many-to-one hashing preserves diverse semantic transitions.

HammingMark defines watermark validity through a Hamming neighborhood centered on the preceding sentence’s hash code, aligning the constraint with continuations naturally favored by the local context. Meanwhile, the coarse many-to-one mapping weakens the coupling between watermark validity and specific semantic content, allowing semantically distinct continuations to satisfy the same code-level constraint. Together, these mechanisms reduce semantic narrowing while preserving flexibility for contrast, elaboration, and topic shifts. A detailed analysis of semantic narrowing across methods is provided in Appendix B and C.

## 3 Method

## 3.1 Semantic Narrowing under Watermark Constraints

We characterize semantic narrowing in terms of the natural probability mass excluded by a watermark constraint.

At generation step $t ,$ let $c _ { t }$ denote the preceding context, and let $S _ { t } \sim P _ { t } ( \cdot \mid c _ { t } )$ denote a semantic continuation drawn from the context-dependent natural generation distribution. Here, $\textstyle P _ { t } ( \cdot \mid c _ { t } )$ captures the distribution of semantic continuations that the underlying language model would naturally produce given $c _ { t }$ . Let $W _ { t }$ denote the set of semantic continuations that satisfy the watermark constraint at step t.

We define the natural acceptance mass of the watermark constraint as

$$
\alpha _ { t } = \operatorname* { P r } _ { S _ { t } \sim P _ { t } ( \cdot | c _ { t } ) } \left[ S _ { t } \in W _ { t } \right] .\tag{1}
$$

The quantity $\alpha _ { t }$ measures how much probability mass of the natural generation distribution remains com patible with the watermark constraint. Correspondingly, the excluded natural probability mass is $1 - \alpha _ { t }$

which quantifies the degree of semantic narrowing induced by the watermark. A smaller $\alpha _ { t }$ indicates poorer alignment between the watermark constraint and the natural distribution, causing more naturally likely continuations to be excluded.

## 3.2 HammingMark: Adaptive Hamming-Neighborhood Acceptance

We construct a context-dependent semantic acceptance rule based on transitions between compact hash codes. For each sentence $S _ { t } ,$ we obtain a semantic embedding $e _ { t } = F ( S _ { t } ) \in \mathbb { R } ^ { d }$ and project it using a fixed random-hyperplane matrix $\boldsymbol { R } \in \mathbb { R } ^ { m \times d }$ . The resulting semantic hash is

$$
h _ { t } = H ( e _ { t } ) = \mathbf { 1 } [ R e _ { t } \geq 0 ] , \qquad h _ { t } \in \{ 0 , 1 \} ^ { m } .\tag{2}
$$

Random-hyperplane hashing preserves angular locality in probability: semantically similar embeddings are more likely to produce matching hash bits, providing a discrete but locality-sensitive representation of sentence semantics. A detailed analysis of this locality-preserving property is provided in Appendix D.

Given the preceding code $h _ { t - 1 }$ , we define the number of matching bits between two consecutive semantic hashes as Match $\mathsf { \Omega } _ { \mathrm { l } } ( h _ { t - 1 } , h _ { t } ) = m - d _ { H } ( h _ { t - 1 } , h _ { t } )$ , where $d _ { H } ( \cdot , \cdot )$ denotes Hamming distance. Hamming Mark accepts the current sentence when

$$
\mathrm { M a t c h } ( h _ { t - 1 } , h _ { t } ) \geq T \quad \Longleftrightarrow \quad d _ { H } ( h _ { t - 1 } , h _ { t } ) \leq r , \qquad r = m - T .\tag{3}
$$

Accordingly, the valid next codes form the Hamming neighborhood

$$
\begin{array} { r } { \mathcal { B } _ { r } ( h _ { t - 1 } ) = \left\{ h \in \{ 0 , 1 \} ^ { m } : d _ { H } ( h , h _ { t - 1 } ) \leq r \right\} . } \end{array}\tag{4}
$$

Under the general formulation in Section 3.1, the natural acceptance mass of HammingMark is

$$
\alpha _ { t } ^ { \mathrm { H M } } = \operatorname* { P r } _ { S _ { t } \sim P _ { t } ( \cdot | c _ { t } ) } \left[ H ( F ( S _ { t } ) ) \in \mathcal { B } _ { r } ( h _ { t - 1 } ) \right] .\tag{5}
$$

During watermark generation, a candidate sentence $S _ { t }$ is accepted if Match $\left( h _ { t - 1 } , h _ { t } \right) \geq T$ and is resampled otherwise. Thus, $\alpha _ { t } ^ { \mathrm { H M } }$ directly measures the natural probability mass retained by the HammingMark acceptance constraint.

The preceding hash $h _ { t - 1 }$ has two roles. First, it determines the center of the current Hamming acceptance constraint, making the watermark rule dependent on the realized preceding semantics rather than on an independently specified target. Second, it serves as one endpoint of the inter-sentence relation statistic used by the detector, so that the same pairwise semantic relation governs both watermark insertion and verifi cation. HammingMark still performs semantic selection by rejecting candidates that violate equation 3; its advantage lies in how the acceptance constraint is placed and represented.

## 3.3 Why HammingMark Alleviates Semantic Narrowing

Under the formulation in Section 3.1, HammingMark alleviates semantic narrowing through two complementary mechanisms. First, the realized-sentence-centered Hamming constraint exploits the semantic continuity of natural sentence transitions through context-aligned shoulder sampling. Second, the coarse many-to-one hash representation allows semantically different continuations to remain valid under the same acceptance constraint.

Context-aligned shoulder sampling. As described in Section 3.2, HammingMark centers its acceptance constraint on the semantic hash of the preceding sentence. The key observation is that natural adjacent sentences typically exhibit semantic continuity, and random-hyperplane hashing preserves this locality in probability.

For a naturally sampled continuation $S _ { t } ,$ , define the natural match distribution and the corresponding HammingMark acceptance mass as

$$
q _ { t } ( k ) = \operatorname* { P r } _ { S _ { t } \sim P _ { t } ( \cdot | c _ { t } ) } \left[ \mathrm { M a t c h } \bigl ( h _ { t - 1 } , H ( F ( S _ { t } ) ) \bigr ) = k \right] , \qquad \alpha _ { t } ^ { \mathrm { H M } } = \sum _ { k = T } ^ { m } q _ { t } ( k ) .\tag{6}
$$

To characterize this distribution, let $e _ { t - 1 }$ and $e _ { t }$ be normalized sentence embeddings with cosine similarity $\rho = e _ { t - 1 } ^ { \top } e _ { t }$ . Under random-hyperplane hashing,

$$
\mathrm { M a t c h } ( h _ { t - 1 } , h _ { t } ) \mid \rho \sim \mathrm { B i n o m i a l } \left( m , 1 - { \frac { \operatorname { a r c c o s } ( \rho ) } { \pi } } \right) .\tag{7}
$$

Therefore, $\operatorname* { P r } [ \operatorname { M a t c h } ( h _ { t - 1 } , h _ { t } ) \geq T \mid \rho ]$ increases monotonically with semantic similarity. Since natural adjacent sentences already tend to exhibit semantic continuity, their match distribution is biased toward moderate or high values. HammingMark thus performs a mild shoulder sampling over this existing distribution rather than directing generation toward an unrelated semantic target. Importantly, this shoulder truncation operates in the discrete code space rather than imposing a hard cutoff in the underlying semantic space.

For example, with $m = 8$ and $T = 6 ,$ candidates producing four, five, or six matching bits are already relatively close in hash similarity. Selecting candidates with six or more matching bits therefore introduces a local preference along an existing natural transition tendency, rather than imposing a qualitatively different semantic direction.

Semantic diversity under coarse hashing. High natural acceptance alone does not guarantee that watermark-valid continuations are semantically diverse. HammingMark additionally exploits the manyto-one nature of compact semantic hashing. The mapping ${ \cal H } : { \cal Z }  \{ 0 , 1 \} ^ { m }$ compresses a highdimensional semantic representation into a short binary code, so multiple distinct semantic representations may map to the same hash code.

For two normalized semantic representations z and $z ^ { \prime }$ with angular distance $\theta ,$ random-hyperplane hashing gives

$$
d _ { H } ( H ( z ) , H ( z ^ { \prime } ) ) \sim \mathrm { B i n o m i a l } \left( m , { \frac { \theta } { \pi } } \right) ,\tag{8}
$$

and therefore

$$
\operatorname* { P r } [ d _ { H } ( H ( z ) , H ( z ^ { \prime } ) ) \leq r ] = \sum _ { j = 0 } ^ { r } { \binom { m } { j } } \left( { \frac { \theta } { \pi } } \right) ^ { j } \left( 1 - { \frac { \theta } { \pi } } \right) ^ { m - j } .\tag{9}
$$

Semantic proximity increases the probability of nearby hash codes, but the converse need not hold for a short code. Since the compact hash provides only a sparse and coarse representation of the underlying semantic space, a higher bit match does not necessarily imply greater semantic similarity than a lower bit match. Importantly, equation 9 shows that even when two semantic representations are less similar, there can still be a nonzero probability that their hash codes fall within the same Hamming neighborhood. Therefore, although HammingMark performs shoulder sampling in the hash space, it does not explicitly filter out continuations solely because they have lower semantic similarity. Since the mapping is manyto-one, semantically different continuations may share the same code or map to different codes that are simultaneously accepted by the Hamming constraint. Watermark validity therefore does not correspond to a single semantic realization or direction. This flexibility is particularly useful for contrast, elaboration, topic shifts, and the introduction of new information, where a continuation may differ substantially from the preceding sentence while still satisfying the watermark rule. Appendix E provides a formal analysis of this many-to-one property.

Together, these two mechanisms address complementary sources of semantic restriction. Context-aligned shoulder sampling makes watermark selection follow an existing tendency of natural semantic transitions rather than imposing an unrelated semantic target, while coarse many-to-one hashing preserves semantic diversity among watermark-valid continuations.

## 3.4 Sequence-Level Detection

Given a text containing n sentences $S _ { 1 } , \ldots , S _ { n }$ , the detector recovers their semantic hashes and adjacent matching scores as

$$
\begin{array} { r } { h _ { t } = H ( F ( S _ { t } ) ) , \qquad M _ { t } = \mathrm { M a t c h } ( h _ { t - 1 } , h _ { t } ) , \quad t = 2 , \ldots , n . } \end{array}\tag{10}
$$

We use two complementary sequence-level statistics.

Edge Vote. Each adjacent sentence pair is treated as one transition unit. A transition casts a valid vote when $M _ { t } \geq T$ , and the final score is

$$
S _ { \mathrm { e d g e } } = { \frac { 1 } { n - 1 } } \sum _ { t = 2 } ^ { n } \mathbf { 1 } [ M _ { t } \geq T ] .\tag{11}
$$

Global Bits. Instead of binarizing each transition, Global Bits aggregates all matching bits,

$$
S _ { \mathrm { g l o b a l } } = { \frac { 1 } { m ( n - 1 ) } } \sum _ { t = 2 } ^ { n } M _ { t } .\tag{12}
$$

It therefore retains partial evidence from transitions that no longer exceed T after semantic-preserving perturbations.

For either detector $d \in \{ \mathrm { e d g e , g l o b a l } \}$ , a text is classified as watermarked when $S _ { d } > \tau _ { d } .$ where $\tau _ { d }$ is calibrated from unwatermarked text at the target false-positive rate. Detection requires only the fixed semantic encoder and projection matrix, without access to the generation model.

## 4 Experiments

We organize our evaluation around the following research questions. RQ1 (Detectability): Can HammingMark be reliably distinguished from unwatermarked text across datasets and language models? RQ2 (Robustness): Does the watermark remain detectable after local edits, sentence-level paraphrasing, and document-level rewriting? RQ3 (Text quality): How well does HammingMark preserve the quality of generated text? RQ4 (Sampling efficiency and semantic overlap): How well does the watermarkvalid region overlap with naturally preferred continuations, and how does this affect rejection-sampling efficiency? RQ5 (Complex tasks): Does HammingMark remain effective on conditional, long-form generation tasks? RQ6 (Semantic sampling visualization): How do different semantic watermarking methods constrain naturally sampled candidate continuations in the semantic space?

## 4.1 RQ1–RQ3: Detectability, Robustness, and Quality

HammingMark maintains near-perfect clean-text detection across all model–dataset settings, with TPR@1% of 0.97–1.00 and AUROC of 0.99–1.00. Edge Vote is particularly effective on clean text because it directly measures whether each transition satisfies the embedding rule.

Under paraphrasing and rewriting, Global Bits is generally more robust, achieving TPR@1% of 0.82– 0.97 across all settings. Unlike Edge Vote, which discards a transition once it falls below the matching threshold, Global Bits retains partial evidence from modified transitions by aggregating all matching bits. Therefore, Edge Vote is preferable for ordinary clean-text verification, whereas Global Bits is better suited to attacked text.

HammingMark also preserves generation quality, with PPL differing from unwatermarked generation by at most 0.30 across all settings. This advantage follows from the relational Hamming-ball constraint: instead of forcing each sentence into a fixed semantic region, HammingMark retains a large set of ad missible semantic realizations. The model can therefore continue exploring plausible continuations and select content appropriate to the context while embedding a detectable transition-level signal.

## 4.2 RQ4: Sampling Efficiency and Natural Acceptance

As shown in Table 2, HammingMark requires only 2.2 sampled candidates and 41.58 generated tokens for each accepted sentence. Compared with SimMark, the most efficient baseline, HammingMark reduces the number of sampled candidates by approximately 72.8% and token consumption by 77.7%.

Table 1: Clean detectability, robustness, and generation quality on C4 and BookSum. Each detection entry reports TPR@1%/TPR@5%/AUROC.
<table><tr><td>Method</td><td>Clean</td><td>Parrot</td><td>Pegasus</td><td>DIPPER</td><td>API</td><td>PPL ↓</td></tr><tr><td colspan="7">Mistral-7B (Jiang et al., 2023) on C4 (Raffel et al., 2020)</td></tr><tr><td colspan="7">No watermark</td></tr><tr><td>KGW (Kirchenbauer et al., 2023)</td><td>1.00/1.00/1.000.59/0.72/0.90</td><td></td><td>0.51/0.63/0.89</td><td>0.43/0.49/0.840.31/0.39/0.80</td><td></td><td>8.00</td></tr><tr><td>SynthID (Dathathri et al., 2024)</td><td>1.00/1.00/1.00</td><td>0.32/0.45/0.82</td><td>0.23/0.42/0.80</td><td>0.35/0.40/0.80</td><td>0.24/0.38/0.79</td><td>4.52</td></tr><tr><td>MorphMark (Wang et al., 2025)</td><td>0.99/1.00/1.00</td><td>0.54/0.60/0.88</td><td>0.47/0.55/0.84</td><td>0.51/0.55/0.86</td><td>0.53/0.64/0.89</td><td>6.62</td></tr><tr><td>SIR (Liu et al., 2024)</td><td>0.98/1.00/1.00</td><td>0.51/0.63/0.91</td><td>0.56/0.60/0.90</td><td>0.44/0.57/0.84</td><td>0.38/0.46/0.82</td><td>8.35</td></tr><tr><td>K-SemStamp (Hou et al., 2024b)</td><td>0.83/0.88/0.98</td><td>0.43/0.58/0.86</td><td>0.46/0.65/0.89</td><td>0.66/0.87/0.95</td><td>0.31/0.40/0.80</td><td>10.78</td></tr><tr><td>SemStamp (Hou et al., 2024a)</td><td>0.82/0.87/0.97</td><td>0.56/0.77/0.91</td><td>0.79/0.90/0.97</td><td>0.76/0.94/0.98</td><td>0.32/0.48/0.84</td><td>9.36</td></tr><tr><td>SimMark (Dabiriaghdam &amp; Wang, 2025)</td><td>0.71/0.82/0.95</td><td>0.28/0.43/0.82</td><td>0.35/0.51/0.84</td><td>0.23/0.41/0.80</td><td>0.25/0.36/0.78</td><td>8.34</td></tr><tr><td>PMark (Huo et al., 2026)</td><td>0.99/0.99/1.00</td><td>0.94/0.97/0.99</td><td>0.89/0.91/0.98</td><td>0.75/0.85/0.97</td><td>0.91/0.93/0.98</td><td>5.13</td></tr><tr><td>HammingMark (Edge Vote)</td><td>0.99/1.00/1.00</td><td>0.92/0.96/0.99</td><td>0.85/0.98/0.99</td><td>0.89/0.97/0.99</td><td>0.90/0.93/0.98</td><td>4.61</td></tr><tr><td>HammingMark (Global Bits)</td><td>0.98/1.00/1.00</td><td>0.94/0.99/1.00</td><td>0.91/1.00/0.99</td><td>0.93/0.98/0.99</td><td>0.94/0.97/0.99</td><td>4.61</td></tr><tr><td colspan="7">Mistral-7B on BookSum (Kryściński et al., 2022)</td></tr><tr><td colspan="7"></td></tr><tr><td>No watermark KGW</td><td></td><td></td><td></td><td></td><td></td><td>6.20</td></tr><tr><td></td><td>1.00/1.00/1.000.34/0.43/0.860.40/0.45/0.82 0.35/0.41/0.81 0.16/0.23/0.64</td><td></td><td>0.10/0.16/0.60</td><td></td><td></td><td>10.14</td></tr><tr><td>SynthID</td><td>1.00/1.00/1.00</td><td>0.15/0.21/0.69 0.45/0.52/0.83</td><td>0.31/0.37/0.79</td><td>0.11/0.19/0.62</td><td>0.13/0.18/0.60</td><td>6.57</td></tr><tr><td>MorphMark</td><td>1.00/1.00/1.00</td><td>0.54/0.61/0.85</td><td>0.47/0.63/0.87</td><td>0.42/0.55/0.81 0.52/0.61/0.83</td><td>0.22/0.27/0.65</td><td>9.33</td></tr><tr><td>SIR</td><td>1.00/1.00/1.00 0.79/0.91/0.98</td><td>0.51/0.64/0.88</td><td>0.56/0.69/0.90</td><td>0.70/0.86/0.94</td><td>0.14/0.25/0.65</td><td>12.49</td></tr><tr><td>K-SemStamp SemStamp</td><td>0.88/0.93/0.98</td><td>0.67/0.71/0.90</td><td>0.71/0.78/0.90</td><td>0.82/0.91/0.97</td><td>0.16/0.27/0.69</td><td>15.72 14.98</td></tr><tr><td>SimMark</td><td>0.77/0.86/0.96</td><td>0.29/0.52/0.81</td><td>0.44/0.58/0.84</td><td>0.21/0.43/0.80</td><td>0.20/0.35/0.71</td><td>14.25</td></tr><tr><td>PMark</td><td>0.97/0.99/0.99</td><td>0.96/0.98/0.99</td><td>0.79/0.91/0.98</td><td>0.78/0.93/0.98</td><td>0.14/0.21/0.62</td><td>8.26</td></tr><tr><td>HammingMark (Edge Vote)</td><td>1.00/1.00/1.00</td><td>0.91/0.94/0.99</td><td>0.86/0.95/0.99</td><td>0.91/0.94/0.98</td><td>0.71/0.84/0.94</td><td></td></tr><tr><td>HammingMark (Global Bits)</td><td>1.00/1.00/1.00</td><td>0.93/0.95/0.99</td><td>0.90/0.99/0.99</td><td>0.91/0.94/0.99</td><td>0.88/0.92/0.98 0.92/0.95/0.99</td><td>6.40 6.40</td></tr><tr><td colspan="7"></td></tr><tr><td colspan="7">Qwen2.5-7B (Yang et al., 2024) on C4</td></tr><tr><td>No watermark KGW</td><td>0.99/1.00/1.000.08/0.13/0.660.08/0.11/0.67</td><td></td><td></td><td>0.06/0.12/0.610.02/0.08/0.49</td><td></td><td>5.52 7.29</td></tr><tr><td>SynthID</td><td>0.99/1.00/1.00</td><td>0.06/0.10/0.50</td><td>0.02/0.04/0.47</td><td>0.03/0.06/0.48</td><td>0.02/0.06/0.48</td><td>5.30</td></tr><tr><td>MorphMark</td><td>0.98/1.00/1.00</td><td>0.04/0.20/0.65</td><td>0.06/0.17/0.64</td><td>0.02/0.21/0.64</td><td>0.03/0.11/0.53</td><td>5.63</td></tr><tr><td>SIR</td><td>0.97/1.00/1.00</td><td>0.55/0.59/0.89</td><td>0.62/0.68/0.92</td><td>0.41/0.58/0.85</td><td>0.34/0.40/0.79</td><td>7.08</td></tr><tr><td>K-SemStamp</td><td>0.78/0.93/0.95</td><td>0.42/0.65/0.86</td><td>0.49/0.73/0.89</td><td>0.66/0.90/0.95</td><td>0.17/0.29/0.77</td><td>11.67</td></tr><tr><td>SemStamp</td><td>0.85/0.88/0.97</td><td>0.59/0.82/0.91</td><td>0.61/0.83/0.95</td><td>0.74/0.96/0.97</td><td>0.24/0.37/0.82</td><td>9.46</td></tr><tr><td>SimMark</td><td>0.72/0.81/0.94</td><td>0.27/0.50/0.82</td><td>0.38/0.52/0.86</td><td>0.29/0.41/0.78</td><td>0.31/0.46/0.83</td><td>8.11</td></tr><tr><td>PMark</td><td>0.99/1.00/0.99</td><td>0.96/0.98/0.99</td><td>0.85/0.91/0.96</td><td>0.91/0.94/0.97</td><td>0.83/0.89/0.98</td><td>6.32</td></tr><tr><td>HammingMark (Edge Vote)</td><td>0.98/0.99/0.99</td><td>0.93/0.94/0.98</td><td>0.89/0.93/0.98</td><td>0.92/0.94/0.98</td><td>0.94/0.98/0.99</td><td>5.38</td></tr><tr><td>HammingMark (Global Bits)</td><td>0.98/0.99/0.99</td><td>0.97/0.98/0.99</td><td></td><td>0.89/0.93/0.98 0.94/0.95/0.99</td><td>0.95/0.98/0.99</td><td>5.38</td></tr><tr><td colspan="7">Qwen2.5-7B on BookSum</td></tr><tr><td colspan="7">No watermark</td></tr><tr><td>KGW</td><td>0.99/1.00/1.000.05/0.13/0.65</td><td></td><td>0.05/0.19/0.620.02/0.12/0.62</td><td></td><td>0.03/0.12/0.53</td><td>4.45 8.46</td></tr><tr><td>SynthID</td><td>0.99/1.00/1.00</td><td>0.03/0.08/0.54</td><td>0.05/0.12/0.56</td><td>0.00/0.09/0.55</td><td>0.01/0.08/0.51</td><td>4.63</td></tr><tr><td>MorphMark</td><td>0.97/0.99/1.00</td><td>0.02/0.05/0.57</td><td>0.02/0.07/0.55</td><td>0.02/0.06/0.56</td><td>0.01/0.09/0.51</td><td>7.31</td></tr><tr><td>SIR</td><td>0.95/0.98/0.99</td><td>0.54/0.63/0.84</td><td>0.60/0.67/0.91</td><td>0.46/0.66/0.85</td><td>0.10/0.23/0.72</td><td>9.76</td></tr><tr><td>K-SemStamp</td><td>0.82/0.91/0.96</td><td>0.44/0.70/0.88</td><td>0.53/0.61/0.84</td><td>0.68/0.75/0.90</td><td>0.14/0.28/0.75</td><td>10.30</td></tr><tr><td>SemStamp</td><td>0.87/0.90/0.97</td><td>0.57/0.84/0.92</td><td>0.45/0.71/0.89</td><td>0.70/0.78/0.91</td><td>0.17/0.33/0.76</td><td>11.49</td></tr><tr><td>SimMark</td><td>0.71/0.85/0.93</td><td>0.31/0.43/0.80</td><td>0.36/0.54/0.83</td><td>0.24/0.40/0.81</td><td>0.13/0.21/0.72</td><td>10.28</td></tr><tr><td>PMark</td><td>0.95/0.99/0.99</td><td>0.92/0.98/0.99</td><td></td><td></td><td></td><td></td></tr><tr><td>HammingMark (Edge Vote)</td><td>0.99/1.00/1.00</td><td>0.87/0.93/0.98</td><td>0.83/0.89/0.97</td><td>0.74/0.92/0.960.78/0.89/0.97 0.82/0.92/0.97</td><td>0.78/0.85/0.96 0.89/0.94/0.98</td><td>6.39 4.75</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>HammingMark (Global Bits)</td><td>0.97/0.99/1.000.93/0.95/0.99</td><td></td><td></td><td>0.84/0.91/0.970.82/0.94/0.98 0.91/0.94/0.98</td><td></td><td>4.75</td></tr></table>

To examine the source of this sampling efficiency, we estimate the natural acceptance mass defined in Section 3.1. For each of 100 prompts, we independently sample 300 candidate continuations from the unwatermarked model before applying any watermark constraint. These candidates are therefore samples from the context-dependent natural generation distribution $\textstyle P _ { t } ( \cdot \mid c _ { t } )$ . We then apply each watermarking rule to the sampled candidates and measure the fraction that would be accepted without resampling.

Table 2: Sampling efficiency and estimated natural acceptance on C4. Lower sampling cost and higher natural acceptance are better. Random Code uses the same code-space coverage as HammingMark (37/256) but selects valid codes uniformly at random.
<table><tr><td>Method</td><td>SimMark</td><td>SemStamp</td><td>K-SemStamp</td><td>PMark</td><td>Random Code</td><td>HammingMark</td></tr><tr><td>Samples / sent. ↓</td><td>8.1</td><td>99.3</td><td>13.3</td><td>64.0</td><td>14.7</td><td>2.2</td></tr><tr><td>Tokens / sent. ↓</td><td>186.7</td><td>1694.4</td><td>246.9</td><td>1185.8</td><td>273.1</td><td>41.58</td></tr><tr><td>Natural acceptance  ${ \widehat { \bar { \alpha } } } \uparrow$ </td><td>0.34</td><td>0.04</td><td>0.13</td><td>1</td><td>0.15</td><td>0.58</td></tr></table>

Table 3: Performance on complex generation tasks on Mistral-7B. Detection results are reported as TPR@1%, TPR@5%, and AUROC.
<table><tr><td rowspan="2">Method</td><td colspan="4">ELI5 (Fan et al., 2019)</td><td colspan="4">Multi-News (Fabbri et al., 2019)</td></tr><tr><td>@1↑</td><td>@5↑</td><td>AUC↑</td><td>Rouge-L ↑</td><td>@1↑</td><td>@5↑</td><td>AUC↑</td><td>Rouge-L ↑</td></tr><tr><td>No watermark</td><td>一</td><td>一</td><td></td><td>0.34</td><td>一</td><td>一</td><td>一</td><td>0.36</td></tr><tr><td>KGW</td><td>0.78</td><td>0.89</td><td>0.97</td><td>0.29</td><td>0.21</td><td>0.49</td><td>0.86</td><td>0.31</td></tr><tr><td>SynthID</td><td>0.43</td><td>0.67</td><td>0.92</td><td>0.31</td><td>0.08</td><td>0.17</td><td>0.64</td><td>0.36</td></tr><tr><td>MorphMark</td><td>0.49</td><td>0.64</td><td>0.90</td><td>0.29</td><td>0.26</td><td>0.41</td><td>0.88</td><td>0.35</td></tr><tr><td>SIR</td><td>0.63</td><td>0.75</td><td>0.93</td><td>0.28</td><td>0.21</td><td>0.36</td><td>0.85</td><td>0.34</td></tr><tr><td>SemStamp</td><td>0.22</td><td>0.36</td><td>0.82</td><td>0.31</td><td>0.03</td><td>0.19</td><td>0.64</td><td>0.35</td></tr><tr><td>K-SemStamp</td><td>0.13</td><td>0.24</td><td>0.71</td><td>0.32</td><td>0.03</td><td>0.16</td><td>0.64</td><td>0.33</td></tr><tr><td>SimMark</td><td>0.26</td><td>0.41</td><td>0.84</td><td>0.31</td><td>0.06</td><td>0.11</td><td>0.62</td><td>0.34</td></tr><tr><td>PMark</td><td>0.85</td><td>0.91</td><td>0.98</td><td>0.29</td><td>0.86</td><td>0.92</td><td>0.99</td><td>0.32</td></tr><tr><td>HammingMark</td><td>0.95</td><td>0.95</td><td>0.99</td><td>0.33</td><td>0.91</td><td>1.00</td><td>0.99</td><td>0.36</td></tr></table>

For a given context $c _ { t }$ , we estimate the natural acceptance mass by

$$
\hat { \alpha } _ { t } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } { \bf 1 } \left[ S _ { t } ^ { ( i ) } \in W _ { t } \right] , \qquad S _ { t } ^ { ( i ) } \sim P _ { t } ( \cdot \mid c _ { t } ) .\tag{13}
$$

where $N = 3 0 0$ is the number of independently sampled natural candidates. We report ${ \widehat { \bar { \alpha } } } ,$ , obtained by averaging $\hat { \alpha } _ { t }$ across the 100 evaluation prompts. Thus, the values reported in Table 2 are direct Monte Carlo estimates of the natural probability mass retained by each watermark constraint. A larger natural acceptance mass means that a greater fraction of continuations produced by the unwatermarked model already satisfies the watermark rule. This quantity is directly related to rejection-sampling cost: when natural candidates are accepted more frequently, fewer candidates need to be sampled before obtaining a watermark-valid continuation. HammingMark achieves an estimated natural acceptance of 0.58, meaning that approximately 58% of naturally sampled continuations satisfy its watermark constraint without additional semantic redirection. This is consistent with its substantially lower sampling cost. We further evaluate the loss of semantic diversity induced by each method in Appendix F and $\mathrm { G } ,$ complementing the natural acceptance analysis with a direct assessment of diversity preservation.

## 4.3 RQ5: Performance on Complex Generation Tasks

As shown in Table 3, HammingMark achieves TPR@1% of 0.95 on ELI5 and 0.91 on MultiNews while preserving the highest or tied-highest ROUGE-L. These tasks strongly constrain generation semantics, concentrating the natural distribution $\textstyle P _ { t } ( \cdot \mid c _ { t } )$ on task-consistent continuations and leaving little room for semantic redirection. Poorly aligned watermark constraints can therefore reject many high-probability candidates, yielding a small natural acceptance mass $\alpha _ { t }$ and frequent resampling. HammingMark mitigates this issue through mild shoulder sampling around the realized semantic transition. Semantically coherent continuations are more likely to have high hash similarity, while the many-to-one hash mapping allows multiple semantic realizations to satisfy the same Hamming constraint. This preserves more natural probability mass, helping maintain task fidelity while accumulating a detectable watermark signal.

![](images/8144a19ad5e5a1506f94017c8a5be7674bfb9f921cd95922170937a143b8cc3f.jpg)  
(a) KSEMSTAMP resampling dimensionality reduction

![](images/92b596eeb99c7cce283fee8cfcd6e0f3465ea32a915f3c34f35f41c58519819a.jpg)  
(b) PMark resampling dimensionality reduction

![](images/424377e62463534d66732ce88e1e8982cdaef42d57c3ea1a522706f7458dbbcc.jpg)  
(c) HammingMark resampling dimensionality reduction  
Figure 2: Visualization of natural candidates accepted by different watermark constraints. UMAP projection of 300 independently sampled continuations from the unwatermarked model for a representative challenging prompt. All methods are compared under matched watermark effectiveness. For offline PMark, we highlight candidates matching all key-specified channel signs under the zero-median prior. Highlighted points denote candidates accepted by the watermark constraint.

## 4.4 RQ6: Visualizing Natural Acceptance under Semantic Constraints

To provide an intuitive candidate-level view of natural acceptance, we independently sample 300 continuations from the unwatermarked model for a representative challenging prompt and project their sentence embeddings into two dimensions with UMAP. For each method, candidates satisfying its watermark constraint are highlighted within the same natural sample set. The highlighted fraction thus gives a promptlevel Monte Carlo estimate of the natural acceptance mass.

As shown in Figure 2, K-SemStamp accepts only 2 of the 300 natural candidates, while PMark accepts 31 and HammingMark accepts 99. Thus, for this prompt, HammingMark retains a substantially larger fraction of the continuations that the unwatermarked model naturally produces. The highlighted HammingMark candidates are also distributed across multiple parts of the projected sample cloud, illustrating that watermark-valid candidates are not confined to a single local semantic realization. This visualization complements the aggregate natural-acceptance results in Table 2. Importantly, the UMAP geometry is used only for qualitative illustration: semantic narrowing is quantified by the fraction of natural probabil ity mass retained by the watermark constraint, rather than by the geometric area occupied by highlighted points in the two-dimensional projection.

## 4.5 Ablation Study

Effect of constraint placement. We examine whether HammingMark’s higher natural acceptance is due simply to the number of valid hash codes or to how these codes are placed.

With $m = 8$ and $T = 6 ,$ , HammingMark accepts $\textstyle \sum _ { j = 0 } ^ { 2 } { \binom { 8 } { j } } = 3 7$ of the 256 possible hash codes. As a control, we construct a Random Code baseline that also accepts 37 codes, but selects them uniformly at random and independently of the preceding sentence. All other components and evaluation settings remain unchanged.

As shown in Table 2, Random Code achieves a natural acceptance of 0.15, whereas HammingMark reaches 0.58, despite identical code-space coverage. For a random set A of 37 codes, each code is included with probability 37/256. Therefore, for any natural hash distribution $p _ { t } ( h )$

$$
\mathbb { E } _ { \boldsymbol { A } } \left[ \alpha _ { t } ( \boldsymbol { A } ) \right] = \sum _ { h } p _ { t } ( h ) \operatorname* { P r } _ { \boldsymbol { A } } [ h \in \boldsymbol { A } ] = \frac { 3 7 } { 2 5 6 } \approx 0 . 1 4 4 5 .\tag{14}
$$

The observed value of 0.15 is close to this random-set expectation.

In contrast, HammingMark places the same number of valid codes around the preceding sentence hash. Combined with the locality-preserving property of random-hyperplane hashing, this transition-centered placement makes the constraint better aligned with natural semantic transitions. The large gap between 0.15 and 0.58 therefore shows that natural acceptance depends strongly on constraint placement rather than code-space coverage alone.

## 5 Limitations

HammingMark depends on the quality of the semantic encoder and sentence segmentation; unstable embeddings or ambiguous sentence boundaries may alter hash codes and reduce detection reliability. Although HammingMark substantially improves sampling efficiency, it remains a rejection-sampling method and may incur additional cost under highly restrictive contexts or strict matching thresholds.

## 6 Conclusion

This work identifies semantic narrowing as the misalignment between watermark constraints and naturally preferred generation. We propose HammingMark, a realized-sentence-centered semantic watermarking method that reduces this mismatch through context-aligned Hamming acceptance while preserving semantic flexibility. Across open-ended and constrained generation tasks, HammingMark achieves strong detectability and robustness with near-unwatermarked quality and low sampling cost.

## AI use statement

In this work, generative AI tools were used to assist with language polishing, manuscript drafting and revision, formatting, and literature search. All AI-assisted outputs, including retrieved information and suggested text, were critically reviewed, verified, and revised by the authors. The authors take full responsibility for the accuracy, integrity, and final content of the paper. We take responsibility for the final content of this work, including any text or artifacts produced with the aid of generative AI.

## Ethics statement

This work does not involve human-subject studies or the collection of personally identifiable or sensitive information. Our experiments are conducted using publicly available or appropriately licensed data and standard evaluation protocols. We do not intend the proposed methods to facilitate harmful, discriminatory, privacy-invasive, or otherwise unethical applications. Nevertheless, as with many machine learning methods, the proposed approach may inherit biases or limitations from the data and models on which it relies, and we encourage careful evaluation before deployment in safety- or socially sensitive settings. We have made our best effort to ensure research integrity, appropriate attribution, and compliance with relevant ethical and legal requirements. The authors declare no conflicts of interest that would affect the presentation or interpretation of the results.

## Reproducibility statement

We have made efforts to ensure the reproducibility of our results. The appendix provides the complete experimental setup and all details necessary to reproduce our experiments, including implementation details, evaluation configurations, hyperparameter settings, data processing procedures, and other relevant experimental specifications. We also provide additional results and analyses where appropriate to facilitate verification of the reported findings. Together, these materials are intended to enable independent reproduction of the main results presented in this work. The source code will be made publicly available upon acceptance of the paper.

## References

Lukas Aichberger, Kajetan Schweighofer, Mykyta Ielanskyi, and Sepp Hochreiter. Improving uncertainty estimation through semantically diverse language generation. In International Conference on Learning Representations, volume 2025, pp. 74479–74503, 2025.

Stephen Casper, Kyle O’Brien, Shayne Longpre, Elizabeth Seger, Kevin Klyman, Rishi Bommasani, Aniruddha Nrusimha, Ilia Shumailov, Soren Mindermann, Steven Basart, et al. Open technical prob-¨ lems in open-weight ai model risk management. arXiv preprint arXiv:2608.07514, 2026.

Amirhossein Dabiriaghdam and Lele Wang. Simmark: A robust sentence-level similarity-based watermarking algorithm for large language models. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 30773–30794, 2025.

Prithiviraj Damodaran. Parrot: Paraphrase generation for nlu., 2021.

Sumanth Dathathri, Abigail See, Sumedh Ghaisas, Po-Sen Huang, Rob McAdam, Johannes Welbl, Vandana Bachani, Alex Kaskasoli, Robert Stanforth, Tatiana Matejovicova, et al. Scalable watermarking for identifying large language model outputs. Nature, 634(8035):818–823, 2024.

Alexander Richard Fabbri, Irene Li, Tianwei She, Suyi Li, and Dragomir Radev. Multi-news: A largescale multi-document summarization dataset and abstractive hierarchical model. In Proceedings ofthe 57th annual meeting ofthe associationfor computational linguistics, pp. 1074–1084, 2019.

Angela Fan, Yacine Jernite, Ethan Perez, David Grangier, Jason Weston, and Michael Auli. Eli5: Long form question answering. In Proceedings of the 57th annual meeting of the association for computational linguistics, pp. 3558–3567, 2019.

Sebastian Farquhar, Jannik Kossen, Lorenz Kuhn, and Yarin Gal. Detecting hallucinations in large lan guage models using semantic entropy. Nature, 630(8017):625–630, 2024.

Abe Hou, Jingyu Zhang, Tianxing He, Yichen Wang, Yung-Sung Chuang, Hongwei Wang, Lingfeng Shen, Benjamin Van Durme, Daniel Khashabi, and Yulia Tsvetkov. Semstamp: A semantic watermark with paraphrastic robustness for text generation. In Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), pp. 4067–4082, 2024a.

Abe Hou, Jingyu Zhang, Yichen Wang, Daniel Khashabi, and Tianxing He. k-semstamp: A clusteringbased semantic watermark for detection of machine-generated text. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 1706–1715, 2024b.

Zhengmian Hu, Lichang Chen, Xidong Wu, Yihan Wu, Hongyang Zhang, and Heng Huang. Unbiased watermark for large language models. In International Conference on Learning Representations, volume 2024, pp. 45408–45436, 2024.

Jiahao Huo, Shuliang Liu, Bin Wang, Junyan Zhang, Yibo Yan, Aiwei Liu, Xuming Hu, and Mingxun Zhou. Pmark: Towards robust and distortion-free semantic-level watermarking with channel constraints. In International Conference on Learning Representations, volume 2026, pp. 27100–27126, 2026.

Albert Q. Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lelio Renard ´ Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Tim othee Lacroix, and William El Sayed. Mistral 7b, 2023. URL´ https://arxiv.org/abs/2310. 06825.

John Kirchenbauer, Jonas Geiping, Yuxin Wen, Jonathan Katz, Ian Miers, and Tom Goldstein. A watermark for large language models. In International conference on machine learning, pp. 17061–17084. PMLR, 2023.

Kalpesh Krishna, Yixiao Song, Marzena Karpinska, John Wieting, and Mohit Iyyer. Paraphrasing evades detectors of ai-generated text, but retrieval is an effective defense. Advances in neural information processing systems, 36:27469–27500, 2023.

Wojciech Krysci´ nski, Nazneen Rajani, Divyansh Agarwal, Caiming Xiong, and Dragomir Radev. Book-´ sum: A collection of datasets for long-form narrative summarization. In Findings of the association for computational linguistics: EMNLP 2022, pp. 6536–6558, 2022.

Yuqing Liang, Jiancheng Xiao, Wensheng Gan, and Philip S Yu. Watermarking techniques for large language models: A survey. Artificial Intelligence Review, 59(2):74, 2026.

Aiwei Liu, Leyi Pan, Xuming Hu, Shiao Meng, and Lijie Wen. A semantic invariant robust watermark for large language models. In International Conference on Learning Representations, volume 2024, pp. 6499–6519, 2024.

Shuliang Liu, Xingyu Li, Hongyi Liu, Dong Fang, Yibo Yan, Bingchen Duan, Qi Zheng, Lingfeng Su, and Xuming Hu. Distilling the thought, watermarking the answer: A principle semantic guided watermark for large reasoning models. arXiv preprint arXiv:2601.05144, 2026.

Yijian Lu, Aiwei Liu, Dianzhi Yu, Jingjing Li, and Irwin King. An entropy-based text watermarking detection method. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 11724–11735, 2024.

Clara Meister, Tiago Pimentel, Gian Wiher, and Ryan Cotterell. Locally typical sampling. Transactions of the Association for Computational Linguistics, 11:102–121, 2023.

Leyi Pan, Aiwei Liu, Zhiwei He, Zitian Gao, Xuandong Zhao, Yijian Lu, Binglin Zhou, Shuliang Liu, Xuming Hu, Lijie Wen, et al. Markllm: An open-source toolkit for llm watermarking. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 61–71, 2024.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J Liu. Exploring the limits of transfer learning with a unified text-to-text transformer. Journal ofmachine learning research, 21(140):1–67, 2020.

Jie Ren, Han Xu, Yiding Liu, Yingqian Cui, Shuaiqiang Wang, Dawei Yin, and Jiliang Tang. A robust semantics-based watermark for large language model against paraphrasing. In Findings of the Associationfor Computational Linguistics: NAACL 2024, pp. 613–625, 2024.

Zewen Sun, Qian Jiang, Siyuan Sheng, and Liyao Xiang. Watermoe: Expert-routing-based watermarking for high fidelity and efficiency. arXiv preprint arXiv:2607.13099, 2026.

Zongqi Wang, Tianle Gu, Baoyuan Wu, and Yujiu Yang. Morphmark: Flexible adaptive watermarking for large language models. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 4842–4860, 2025.

Anyi Xu, Bangcai Lin, Bing Xue, Bingxuan Wang, Bingzheng Xu, Bochao Wu, Bowei Zhang, Chaofan Lin, Chen Dong, Chenchen Ling, et al. Deepseek-v4: Towards highly efficient million-token context intelligence. arXiv preprint arXiv:2606.19348, 2026.

Zhenhua Xu, Xubin Yue, Zhebo Wang, Haobo Zhang, Qichen Liu, Xixiang Zhao, Jingxuan Zhang, Wenjun Zeng, Wengpeng Xing, Dezhang Kong, et al. Copyright protection for large language models: A survey of methods, challenges, and trends. arXiv preprint arXiv:2508.11548, 2025.

An Yang, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chengyuan Li, Dayiheng Liu, Fei Huang, Haoran Wei, Huan Lin, Jian Yang, Jianhong Tu, Jianwei Zhang, Jianxin Yang, Jiaxi Yang, Jingren Zhou, Junyang Lin, Kai Dang, Keming Lu, Keqin Bao, Kexin Yang, Le Yu, Mei Li, Mingfeng Xue, Pei Zhang, Qin Zhu, Rui Men, Runji Lin, Tianhao Li, Tingyu Xia, Xingzhang Ren, Xuancheng Ren, Yang Fan, Yang Su, Yichang Zhang, Yu Wan, Yuqiong Liu, Zeyu Cui, Zhenru Zhang, and Zihan Qiu. Qwen2.5 technical report. CoRR, abs/2412.15115, 2024. doi: 10.48550/ARXIV.2412. 15115. URL https://doi.org/10.48550/arXiv.2412.15115.

Zhiguang Yang, Gejian Zhao, and Hanzhou Wu. Watermarking for large language models: A survey. Mathematics, 13(9):1420, 2025.

Jingqing Zhang, Yao Zhao, Mohammad Saleh, and Peter Liu. Pegasus: Pre-training with extracted gapsentences for abstractive summarization. In International conference on machine learning, pp. 11328– 11339. PMLR, 2020.

Junyan Zhang, Shuliang Liu, Aiwei Liu, Yubo Gao, Jungang Li, Xiaojie Gu, and Xuming Hu. Cohemark: A novel sentence-level watermark for enhanced text quality. arXiv preprint arXiv:2504.17309, 2025.

## A Practical Limitations of Online PMark

PMark (Huo et al., 2026) makes an important theoretical contribution to semantic watermarking by introducing a proxy-function framework with a distortion-free online construction. At each generation step, it samples candidate sentences and recursively partitions them into equally sized subsets using channelwise empirical medians. Under the prescribed random channel decisions and balanced partitioning, each sampled candidate has the same marginal selection probability. Averaging over these decisions and candidate sampling therefore recovers the original next-sentence distribution. This provides a principled foundation for semantic watermarking. However, this generation-side guarantee does not by itself ensure inexpensive, reproducible, or context-independent verification.

Generator-dependent verification and computational cost. Online PMark requires the detector to resample N candidate continuations for each sentence under its preceding context and estimate channelwise medians from these samples. For D documents containing L sentences each, this entails approximately DLN sentence generations, in addition to semantic encoding and evidence aggregation. Verification therefore requires access to the original generator or a sufficiently faithful generation service, making large-scale screening costly and limiting deployment by independent verifiers. Hosting the generator also introduces substantial memory and inference requirements; relying on a remote service instead introduces availability, latency, and version-control dependencies.

Sensitivity to changes in the generation distribution. The published detector uses fresh samples rather than requiring exact replay of the original candidate pool. Consequently, a change in the sampling seed alone does not invalidate the distortion-free generation guarantee. Nevertheless, reliable median reconstruction depends on reproducing the relevant conditional generation distribution. Model updates or finetuning, changes in prompt formatting or decoding settings, and numerical differences that affect sampling can alter this distribution and its proxy medians. Even under unchanged settings, finite-sample estimation introduces discrepancies between generation and detection. PMark uses soft counting to mitigate such discrepancies, but this mechanism does not guarantee invariance to systematic changes in the generator or its inference configuration.

Key reuse and repeated-query limitations. PMark explicitly acknowledges in its Appendix D that its predefined, position-indexed random seeds leave n-shot undetectability unresolved, and restricts its guarantee to single-shot distortion-freeness (Huo et al., 2026). Specifically, averaging over random watermark decisions recovers the natural next-sentence distribution, but this does not establish that repeated responses under a reused key follow the joint distribution of independent unwatermarked samples. When the same prompt is queried repeatedly with the same key sequence, the corresponding channel preferences are repeatedly imposed rather than independently randomized across responses. Consequently, the singleshot guarantee does not ensure preservation of natural output variability in repeated-query settings. This is an acknowledged limitation of the construction: PMark leaves dynamic semantic-level seed generation for addressing this issue to future work.

Dependence on prompts and sentence–key alignment. For sentence $s _ { t } ,$ , the online detector reconstructs medians from the conditional distribution

$$
P _ { M } ( \cdot \mid \pi , s _ { 1 } , \ldots , s _ { t - 1 } ) ,
$$

while evaluating watermark evidence with the corresponding position-indexed channel keys and prescribed proxy directions. This requires both the appropriate preceding context and the correct sentence index. When a circulated text contains only the generated response, the original prompt is unavailable, preventing faithful reconstruction of the conditioning context. Even when prompt text is included, an unmarked prompt–response boundary leaves the starting key index ambiguous. Prefix truncation or sentence insertion and deletion can further change the available context and shift subsequent sentence–key correspondence. These operations can occur through ordinary quotation or editing, without substantial semantic rewriting. The published detection procedure provides no explicit mechanism for recovering missing context or resynchronizing unknown sentence offsets. Searching over possible offsets would add computational cost and require appropriate false-positive calibration, while still not restoring missing conditioning information.

Overall, online PMark offers a valuable theoretical construction, but its practical verification remains tightly coupled to the generator, conditioning context, and sentence indexing used during generation. These dependencies substantially limit its suitability for scalable detection of independently circulated text.

## B Semantic Narrowing and Watermark Effectiveness

Natural acceptance mass. Fix a generation context $c _ { t }$ and let z denote a normalized sentence embedding with natural distribution $p _ { t } ( z ) = p ( z \mid c _ { t } )$ . For a rejection-sampling watermark, let $a _ { t } ( z ) \in \{ 0 , 1 \}$ be its acceptance rule under a fixed key and context. Define

$$
\alpha _ { t } = \int p _ { t } ( z ) a _ { t } ( z ) d z , \qquad N _ { t } = 1 - \alpha _ { t } .\tag{15}
$$

Integrals are understood with respect to the embedding distribution, including discrete distributions. The excluded natural probability mass $N _ { t }$ is an operational measure of semantic narrowing; it is not the proportional decrease in semantic diversity, entropy, or generation quality. For $\alpha _ { t } > 0$ , ideal rejection sampling with unlimited budget induces

$$
p _ { t } ^ { \mathrm { W M } } ( z ) = \frac { p _ { t } ( z ) a _ { t } ( z ) } { \alpha _ { t } } .\tag{16}
$$

We initially omit robustness margins and fallback, and introduce a finite budget below. Each method may use a different encoder and thus a different embedding distribution. Geometric reference calculations therefore cannot replace comparisons on common candidate texts from shared contexts.

## B.1 Semantic Narrowing in Partition- and Similarity-Based Methods

SemStamp and K-SemStamp. SemStamp (Hou et al., 2024a) partitions embeddings using random hyperplane hashing, whereas K-SemStamp (Hou et al., 2024b) uses clustering. Both select valid partitions pseudorandomly. Ignoring margins, let $C _ { 1 } , \ldots , C _ { K }$ be the partitions and $G _ { t }$ the valid index set. Then

$$
p _ { t , j } = \int _ { C _ { j } } p _ { t } ( z ) d z , \qquad \alpha _ { t } = \sum _ { j \in G _ { t } } p _ { t , j } , \qquad N _ { t } = 1 - \sum _ { j \in G _ { t } } p _ { t , j } .\tag{17}
$$

Their default valid-partition ratio is $\gamma = 0 . 2 5 ;$ ; SemStamp uses three hash bits (eight codes), and $\mathrm { K } \mathrm { - }$ SemStamp uses eight clusters. If partitions have equal natural mass, then $\alpha _ { t } = \gamma = 0 . 2 5$ and $N _ { t } = 0 . 7 5$ Without equal masses, an ideal uniform random choice of $\gamma K$ valid partitions still gives $\mathbb { E } _ { G } [ \alpha _ { t } ] = \gamma$ Neither statement implies that a particular fixed valid set covers this mass. In particular, a uniform angle relative to the preceding sentence does not establish equal partition masses.

Natural continuations can concentrate in a few partitions. A valid set that misses these partitions may have very low acceptance mass, whereas one that includes them may have high mass. Increasing the number of valid partitions increases expected coverage under uniform selection, but need not yield appropriate partial coverage in every context. Nonuniformity does not necessarily reduce acceptance; the concern is context-dependent misalignment and uneven coverage.

Cosine-SimMark. For the cosine-similarity variant of SimMark (Dabiriaghdam & Wang, 2025), write $\theta = \operatorname { a r c c o s } ( z _ { t - 1 } ^ { \top } z )$ and let $f _ { t } ( \theta )$ be its context-dependent density. Omitting fallback, an accepted cosine interval $[ a , b ] \subseteq [ - 1 , 1 ]$ gives

$$
\alpha _ { t } ^ { \mathrm { S i m } } = \int _ { \operatorname { a r c c o s } b } ^ { \operatorname { a r c c o s } a } f _ { t } ( \theta ) d \theta .\tag{18}
$$

Under the reference model $\theta \sim \operatorname { U n i f o r m } ( 0 , \pi )$

$$
\alpha _ { \mathrm { u n i f } } ^ { \mathrm { S i m } } = \frac { \operatorname { a r c c o s } a - \operatorname { a r c c o s } b } { \pi } \approx 3 . 6 8 \% , \qquad N _ { \mathrm { u n i f } } ^ { \mathrm { S i m } } \approx 9 6 . 3 2 \% ,\tag{19}
$$

using the reported main-experiment interval [0.68, 0.76]. A uniform angle is neither uniform cosine similarity nor a uniform distribution on a high-dimensional sphere. These values are reference calculations, not measured semantic losses. A fixed interval may miss the dominant mass in some contexts; actual acceptance can be above or below the reference value. The formula concerns angles in the normalized representation used for cosine scoring; it does not directly describe Euclidean-SimMark or angles before a PCA transformation.

## B.2 Semantic Narrowing under Hamming-Neighborhood Acceptance

Fixed-key acceptance. For a fixed projection matrix R, HammingMark accepts according to

$$
a _ { t } ^ { \mathrm { H M } } ( z ) = \mathbb { I } [ \mathrm { M a t c h } ( H _ { R } ( z _ { t - 1 } ) , H _ { R } ( z ) ) \geq T ] , \qquad \alpha _ { t } ^ { \mathrm { H M } } = \int p _ { t } ( z ) a _ { t } ^ { \mathrm { H M } } ( z ) d z ,\tag{20}
$$

with $N _ { t } ^ { \mathrm { H M } } = 1 - \alpha _ { t } ^ { \mathrm { H M } }$ . For $m = 8 , T = 6$ , the accepted fraction of binary codes is

$$
\gamma _ { \mathrm { c o d e } } = { \frac { { \binom { 8 } { 6 } } + { \binom { 8 } { 7 } } + { \binom { 8 } { 8 } } } { 2 ^ { 8 } } } = { \frac { 3 7 } { 2 5 6 } } \approx 1 4 . 4 5 \%\tag{21}
$$

This counts codes, whereas $\alpha _ { t } ^ { \mathrm { H M } }$ weights them by natural generation probability. Thus $3 7 / 2 5 6$ is neither the actual acceptance mass nor, in general, the detection null hit probability.

Random-projection reference. Hold the context, preceding embedding, and natural candidate distribution fixed independently of the projection draw. For independent isotropic random hyperplanes and a fixed pair at angle θ,

$$
M \mid \theta \sim \operatorname { B i n o m i a l } ( m , 1 - \theta / \pi ) , \qquad A _ { \mathrm { H M } } ( \theta ) = \operatorname { P r } _ { R } ( M \geq T \mid \theta ) .\tag{22}
$$

Consequently,

$$
\mathbb { E } _ { R } [ \alpha _ { t } ^ { \mathrm { H M } } ] = \int _ { 0 } ^ { \pi } f _ { t } ( \theta ) A _ { \mathrm { H M } } ( \theta ) d \theta .\tag{23}
$$

For a uniform angle and $1 \leq T \leq m$ , substituting $u = 1 - \theta / \pi$ yields

$$
\mathbb { E } _ { R } [ \alpha _ { \mathrm { u n i f } } ^ { \mathrm { H M } } ] = \sum _ { k = T } ^ { m } { \binom { m } { k } } \int _ { 0 } ^ { 1 } u ^ { k } ( 1 - u ) ^ { m - k } d u\tag{24}
$$

$$
= \sum _ { k = T } ^ { m } { \frac { 1 } { m + 1 } } = { \frac { m - T + 1 } { m + 1 } } .\tag{25}
$$

Hence $m = 8 , T = 6$ gives mean acceptance $1 / 3$ and mean excluded mass $2 / 3$ . These are averages over random projections, not exact rates for every fixed matrix. They also cannot be multiplied as independent probabilities along a sequence generated with one shared key; generated contexts may themselves depend on that key.

Table 4: Idealized reference calculations, with margins omitted. The rows use different assumptions and averaging operations; they explain mechanisms rather than rank performance on one real generation distribution.
<table><tr><td>Method</td><td>Reference assumption</td><td>Accepted</td><td>Excluded</td></tr><tr><td>SemStamp</td><td>Equal partition masses</td><td>25%</td><td>75%</td></tr><tr><td>K-SemStamp</td><td>Equal partition masses</td><td>25%</td><td>75%</td></tr><tr><td>Cosine-SimMark</td><td>Uniform angle;  $[ a , b ] = [ 0 . 6 8 , 0 . 7 6 ]$ </td><td>3.68%</td><td>96.32%</td></tr><tr><td>Random Code</td><td>Uniform selection of 37 codes; set average</td><td>14.45%</td><td>85.55%</td></tr><tr><td>HammingMark</td><td>Uniform angle; projection average</td><td>33.33%</td><td>66.67%</td></tr></table>

Table 5: Relaxed-constraint comparison.
<table><tr><td>Method</td><td>Setting</td><td>Candidates/sentence</td><td>TPR@1% FPR</td></tr><tr><td>SemStamp</td><td> $\gamma = 0 . 5$ </td><td>2.9</td><td>6%</td></tr><tr><td>K-SemStamp</td><td> $\gamma = 0 . 5$ </td><td>3.1</td><td>9%</td></tr><tr><td>SimMark</td><td> $[ a , b ] = [ 0 . 4 0 , 0 . 9 0 ]$ </td><td>1.6</td><td>7%</td></tr><tr><td>HammingMark</td><td> $m = 8 , T = 6$ </td><td>2.2</td><td>100%</td></tr></table>

Random Code control and reference comparison. In the same eight-bit code space, independently select a uniform subset G of exactly 37 valid codes. For any fixed natural code distribution,

$$
\mathbb { E } _ { G } [ \alpha _ { t } ^ { \mathrm { R a n d o m } } ] = \sum _ { h \in \{ 0 , 1 \} ^ { 8 } } \operatorname* { P r } _ { t } ( H _ { R } ( z ) = h ) \operatorname* { P r } _ { G } ( h \in G ) = \frac { 3 7 } { 2 5 6 } .\tag{26}
$$

Compared with the HammingMark uniform-angle reference, this demonstrates that equal numbers of valid codes can carry different natural candidate mass because of how those codes are arranged.

Nonuniform natural continuations. equation 23 returns the analysis to the context-dependent angular distribution. Centering acceptance on the preceding hash is intended to align valid candidates with locally plausible continuations. A short hash neither uniquely specifies a continuous semantic realization nor imposes a deterministic cosine-similarity cutoff. Although many-to-one hashing also occurs in SemStamp, HammingMark further organizes valid codes as a neighborhood of the preceding code. These mechanisms aim to reduce misalignment, without guaranteeing higher acceptance or moderate coverage in every context. Claims of reduced narrowing should therefore be supported by measured acceptance on common context-specific candidate pools, together with independent semantic evaluations of what those accepted candidates preserve.

## B.3 Effectiveness Beyond Enlarging the Acceptance Set

Relaxing baseline constraints. A natural question is whether existing methods can match Hamming-Mark’s sampling efficiency simply by enlarging their acceptance sets. To examine this possibility, we increase the valid-partition ratio of SemStamp and K-SemStamp and widen the acceptance interval of SimMark, targeting comparable numbers of sampled candidates per output sentence. A controlled comparison holds the generator, dataset, generation length, and sampling budget fixed, and separately calibrates each modified detector at 1% FPR using data disjoint from the test set.

Acceptance mass and selection signal. Enlarging an acceptance set increases its natural probability mass, but does not necessarily preserve its ability to distinguish watermarked outputs from natural text. The relevant quantities are not only the nominal valid fraction γ and mean acceptance mass, but also how the context-specific acceptance mass $\alpha _ { t }$ is distributed. When $\alpha _ { t }$ is close to zero, finite-budget generation frequently fails to find a valid candidate. When $\alpha _ { t }$ is close to one, natural candidates already satisfy the rule, leaving little additional selection signal. Intermediate coverage can support both successful sampling and an increase over the natural hit rate.

This distinction is particularly relevant when natural continuations concentrate in a few semantic partitions. For fixed partition masses $p _ { t , 1 } , \ldots , p _ { t , K }$ , uniformly selecting exactly $\gamma K$ valid partitions gives

$$
\mathbb { E } _ { G } [ \alpha _ { t } ] = \gamma , \qquad \mathrm { V a r } _ { G } ( \alpha _ { t } ) = \gamma ( 1 - \gamma ) \frac { K \sum _ { j } p _ { t , j } ^ { 2 } - 1 } { K - 1 } ,\tag{27}
$$

where $K > 1$ and $\gamma K$ is an integer. Equal partition masses yield $\alpha _ { t } = \gamma$ for every valid set. In contrast, if one partition carries all natural mass, acceptance is either zero or one, depending on whether that partition is selected. Thus, increasing the nominal valid fraction need not produce moderate coverage in individual contexts. This random-set calculation identifies a possible source of uneven coverage; actual variation across contexts must be measured.

Why enlarging the acceptance set is insufficient. The effectiveness of a watermark depends not only on the size of its acceptance set, but also on how this set overlaps with the natural next-sentence distribution. Natural continuations can concentrate in a few context-dependent semantic directions. For SemStamp and K-SemStamp, pseudorandomly selecting more valid partitions does not ensure that these partitions provide appropriate partial coverage of such continuations. Likewise, widening SimMark’s fixed similarity interval does not ensure suitable coverage across different contexts.

This mismatch can leave many contexts with little useful watermark signal. When the valid set covers almost all natural probability mass, unwatermarked candidates already satisfy the constraint, so watermark selection adds little evidence. Conversely, when it covers very little natural mass, valid candidates are difficult to obtain, and fallback can weaken watermark embedding. Enlarging the nominal acceptance set may therefore improve sampling efficiency without ensuring that individual contexts contribute discrimi native evidence. This does not imply that larger acceptance sets necessarily impair detection; the issue is their alignment with context-dependent natural generation.

Implications for HammingMark. HammingMark organizes valid codes around the preceding sentence’s hash, rather than selecting them pseudorandomly or prescribing a fixed continuous-similarity interval. This design aims to retain naturally plausible continuations while preserving a selective bitmatching constraint, allowing more contexts to contribute watermark evidence. Its intended advantage is therefore not simply a larger acceptance mass, but a more contextually aligned arrangement of valid candidates. Uneven contextual coverage offers a possible explanation for the detection degradation of relaxed baselines, although establishing this mechanism requires measuring their context-specific acceptance masses.

## C Semantic Narrowing in Offline PMark

We analyze the semantic restriction induced by the offline variant of PMark. Our analysis does not assume that natural semantic candidates are uniformly distributed across watermark partitions, nor does it require independence or balance among different proxy channels. Instead, we directly examine the candidateselection mechanism used by offline PMark.

Offline watermark partition. Let π denote the current generation context and

$$
P _ { \pi } ( s ) = P _ { M } ( s \mid \pi )
$$

the natural next-sentence distribution. Given b proxy directions $\{ v _ { j } \} _ { j = 1 } ^ { b }$ , offline PMark uses the fixed threshold zero and assigns each sentence a binary semantic representation

$$
g ( s ) = ( \mathbb { I } [ F _ { 1 } ( s ) > 0 ] , \dots , \mathbb { I } [ F _ { b } ( s ) > 0 ] ) ,
$$

where

$$
F _ { j } ( s ) = \langle v _ { j } , T ( s ) \rangle .
$$

For a fixed watermark seed $^ { r , }$ let $E _ { r } ( s )$ denote the watermark evidence assigned to sentence s. A larger $E _ { r } ( s )$ indicates that the sentence satisfies more of the prescribed watermark constraints.

Offline PMark first independently samples a candidate set

$$
W = \{ X _ { 1 } , \ldots , X _ { N } \} , \qquad X _ { i } \stackrel { \mathrm { i . i . d . } } { \sim } P _ { \pi } ,
$$

and then retains only candidates with the maximum watermark evidence:

$$
{ \mathcal { B } } ( W , r ) = \left\{ X _ { i } \in W : E _ { r } ( X _ { i } ) = \operatorname* { m a x } _ { X _ { j } \in W } E _ { r } ( X _ { j } ) \right\} .
$$

The final sentence is sampled from $B ( W , r )$

Candidate-level semantic restriction. For every sampled candidate set,

$$
B ( W , r ) \subseteq W .
$$

Moreover,

$$
B ( W , r ) \subsetneq W
$$

whenever the candidates in $W$ do not all have identical watermark evidence. Therefore, whenever multiple watermark-evidence levels occur in the natural candidate set, every candidate below the maximum level is excluded from final selection.

Importantly, this exclusion is not determined by the likelihood or semantic preference of the base model. Two candidates that are both plausible under $P _ { \pi }$ may receive different watermark evidence solely because they fall into different regions induced by the fixed proxy partitions. Consequently, offline PMark removes part of the naturally sampled semantic alternatives whenever their watermark evidence is lower than that of another candidate in the same candidate pool.

Distribution-level concentration. The same effect can be characterized without making any assumption on how natural semantic probability mass is distributed among the partitions. Let

$$
Z = E _ { r } ( S ) , \qquad S \sim P _ { \pi } ,
$$

denote the watermark evidence of a naturally sampled sentence, and define its cumulative distribution

$$
F _ { r } ( k ) = \operatorname* { P r } _ { S \sim P _ { \pi } } \left[ E _ { r } ( S ) \leq k \right] .
$$

Let $Y$ be the sentence selected by offline PMark and

$$
Z ^ { \star } = E _ { r } ( Y ) .
$$

Since PMark selects a candidate with the maximum watermark evidence among $N$ independent samples,

$$
Z ^ { \star } = \operatorname* { m a x } _ { 1 \leq i \leq N } E _ { r } ( X _ { i } ) .
$$

Therefore,

$$
\operatorname* { P r } ( Z ^ { \star } \leq k ) = \operatorname* { P r } \left( E _ { r } ( X _ { 1 } ) \leq k , \ldots , E _ { r } ( X _ { N } ) \leq k \right) = F _ { r } ( k ) ^ { N } .
$$

For $N > 1$

$$
F _ { r } ( k ) ^ { N } \leq F _ { r } ( k ) ,
$$

with strict inequality whenever

$$
0 < F _ { r } ( k ) < 1 .
$$

Equivalently,

$$
\operatorname* { P r } ( Z ^ { \star } > k ) \geq \operatorname* { P r } ( Z > k ) .
$$

Thus, the distribution induced by offline PMark is systematically shifted toward semantic regions with higher watermark evidence.

Semantic narrowing. This concentration constitutes semantic narrowing at the candidate-selection level. The base model initially provides multiple semantic alternatives according to $P _ { \pi }$ , whereas offline PMark ranks these alternatives according to an additional watermark-specific criterion and removes all candidates below the maximal evidence level within each candidate pool.

The result does not rely on any particular amount of natural probability mass occupying a watermark partition. As long as the watermark evidence is non-constant over naturally plausible continuations, i.e.,

$$
\exists s _ { a } , s _ { b } \in \mathrm { s u p p } ( P _ { \pi } ) \quad \mathrm { s u c h t h a t } \quad E _ { r } ( s _ { a } ) \neq E _ { r } ( s _ { b } ) ,
$$

maximum-evidence selection assigns different selection preferences to these semantic alternatives. For $N > 1$ , this preference is amplified through best-of-N selection, progressively concentrating generation toward watermark-preferred semantic regions.

Therefore, the reduced rejection-sampling cost of offline PMark should not be interpreted as the absence of semantic restriction. Offline PMark replaces explicit rejection with maximum-evidence selection: candidates are generated from the natural distribution, but only the watermark-preferred subset is eligible for final selection. This preserves sampling efficiency while still inducing semantic narrowing.

## D Locality and Perturbation Stability of Semantic Hashes

This section formalizes the locality property used by HammingMark. Let $e , e ^ { \prime } \in \mathbb { S } ^ { d - 1 }$ be two normalized SBERT embeddings, let $R \doteq [ r _ { 1 } ^ { \top } ; \ldots ; r _ { m } ^ { \top } ] \in \bar { \mathbb { R } } ^ { m \times d }$ , where the rows are independently sampled from a spherically symmetric continuous distribution $( \mathbf { e . g . , } r _ { j } \sim \mathcal { N } ( 0 , I _ { d } ) )$ , and define the j-th randomhyperplane hash bit as

$$
H _ { j } ( e ) = \mathbf { 1 } [ r _ { j } ^ { \top } e \geq 0 ] .\tag{28}
$$

Write $\boldsymbol { \rho } = \boldsymbol { e } ^ { \intercal } \boldsymbol { e } ^ { \prime }$ and $\theta = \operatorname { a r c c o s } ( \rho ) \in [ 0 , \pi ]$

Proposition 1 (angular locality). For every hash bit,

$$
\operatorname* { P r } _ { R } [ H _ { j } ( e ) \neq H _ { j } ( e ^ { \prime } ) ] = \frac { \theta } { \pi } , \qquad \operatorname* { P r } _ { R } [ H _ { j } ( e ) = H _ { j } ( e ^ { \prime } ) ] = 1 - \frac { \theta } { \pi } .\tag{29}
$$

Consequently,

$$
d _ { H } ( H ( e ) , H ( e ^ { \prime } ) ) \sim \mathrm { B i n o m i a l } \left( m , \frac { \theta } { \pi } \right) , \qquad \mathbb { E } _ { R } \left[ \frac { d _ { H } ( H ( e ) , H ( e ^ { \prime } ) ) } { m } \right] = \frac { \theta } { \pi } .\tag{30}
$$

Proof. By rotational invariance, only the two-dimensional plane spanned by e and $e ^ { \prime }$ matters. A hyperplane separates the two vectors exactly when its normal falls in the angular wedge between them. The measure of this wedge is $\theta / \pi$ . Independence of the rows of R then gives equation 30.

equation 30 states that semantic closeness is preserved in probability rather than deterministically. A standard Hoeffding bound also gives

$$
\operatorname* { P r } _ { R } \left[ \left| \frac { d _ { H } ( H ( e ) , H ( e ^ { \prime } ) ) } { m } - \frac { \theta } { \pi } \right| \geq \varepsilon \right] \leq 2 \exp ( - 2 m \varepsilon ^ { 2 } ) .\tag{31}
$$

Thus, smaller angular distance yields fewer expected bit flips, while a longer hash concentrates more tightly around this expectation.

Corollary 1 (stability under a meaning-preserving edit). Suppose an edit changes a sentence embedding from $e \tan \tilde { e } ,$ with $\delta = \operatorname { a r c c o s } ( e ^ { \top } \widetilde { e } )$ . Then

$$
\operatorname* { P r } _ { R } [ H ( e ) = H ( \widetilde { e } ) ] = \left( 1 - \frac { \delta } { \pi } \right) ^ { m } ,\tag{32}
$$

$$
\operatorname* { P r } _ { R } [ d _ { H } ( H ( e ) , H ( \widetilde e ) ) \leq b ] = \sum _ { j = 0 } ^ { b } { \binom { m } { j } } \left( { \frac { \delta } { \pi } } \right) ^ { j } \left( 1 - { \frac { \delta } { \pi } } \right) ^ { m - j } .\tag{33}
$$

Moreover, for two original codes $h _ { t - 1 } , h _ { t }$ and their edited versions $\widetilde { h } _ { t - 1 } , \widetilde { h } _ { t }$ ，

$$
\begin{array} { r } { \left| \mathrm { M a t c h } ( h _ { t - 1 } , h _ { t } ) - \mathrm { M a t c h } ( \widetilde { h } _ { t - 1 } , \widetilde { h } _ { t } ) \right| \le d _ { H } ( h _ { t - 1 } , \widetilde { h } _ { t - 1 } ) } \\ { + d _ { H } ( h _ { t } , \widetilde { h } _ { t } ) . } \end{array}\tag{34}
$$

This follows from Match $( a , b ) = m - d _ { H } ( a , b )$ and the triangle inequality for Hamming distance. In particular, a transition satisfying Match $( h _ { t - 1 } , h _ { t } ) \geq T + \gamma$ remains valid whenever the two edited sentences jointly flip at most $\gamma$ bits. This margin argument connects angular locality to robustness of the transition statistic used by HammingMark.

The probability in the results above is over the random construction of R. Once R is fixed, an adversarially chosen embedding can lie close to a hyperplane, so no deterministic collision guarantee is claimed.

## E Semantic Diversity within a Hamming Neighborhood

For a code $h \in \{ 0 , 1 \} ^ { m }$ , define its normalized semantic preimage by

$$
{ \mathcal { C } } ( h ) = \{ e \in \mathbb { S } ^ { d - 1 } : H ( e ) = h \} .\tag{35}
$$

Because $m \ll d ,$ H is highly non-injective. The following construction shows that this is not merely a counting argument.

Proposition 2 (large angular diameter of a hash cell). Assume dim ker $( R ) \geq 2$ and take a unit vector e whose projections Re are all nonzero. There exists a unit vector $v \in \ker ( R ) \cap e ^ { \bot }$ . For any $\tau \geq 0$ define

$$
e _ { + } ( \tau ) = \frac { e + \tau v } { \sqrt { 1 + \tau ^ { 2 } } } , ~ e _ { - } ( \tau ) = \frac { e - \tau v } { \sqrt { 1 + \tau ^ { 2 } } } .\tag{36}
$$

Then

$$
H ( e _ { + } ( \tau ) ) = H ( e ) = H ( e _ { - } ( \tau ) )\tag{37}
$$

for every finite τ, while

$$
e _ { + } ( \tau ) ^ { \top } e _ { - } ( \tau ) = \frac { 1 - \tau ^ { 2 } } { 1 + \tau ^ { 2 } } \longrightarrow - 1 \quad \mathrm { a s } \tau \to \infty .\tag{38}
$$

Proof. The dimension assumption guarantees a nonzero vector in $\ker ( R ) \cap e ^ { \bot }$ . Since $R v \ = \ 0$ $R e _ { + } ( \tau ) = R e _ { - } ( \tau ) = R e / \sqrt { 1 + \tau ^ { 2 } }$ . The positive normalization factor does not change any projection sign, proving equation 37. equation 38 follows by direct calculation. □

For a Gaussian random projection with $m < d ,$ R has full row rank almost surely, so dim ker $( R ) = d - m$ In our setting, $d = 7 6 8$ and $m = 8 .$ , leaving a 760-dimensional null space. Moreover, full row rank makes $R : \mathbb { R } ^ { d }  \mathbb { R } ^ { m }$ surjective; hence every strict sign pattern is realizable in the ambient embedding space.

Given a preceding code $h _ { t - 1 }$ and radius $r = m - T$ , the complete admissible semantic set is

$$
\boldsymbol { A } _ { T } ( h _ { t - 1 } ) = \boldsymbol { H } ^ { - 1 } ( \mathcal { B } _ { r } ( h _ { t - 1 } ) ) = \bigcup _ { h \in \mathcal { B } _ { r } ( h _ { t - 1 } ) } \boldsymbol { \mathcal { C } } ( h ) ,\tag{39}
$$

where

$$
| B _ { r } ( h _ { t - 1 } ) | = \sum _ { j = 0 } ^ { r } { \binom { m } { j } } .\tag{40}
$$

With $m = 8$ and $T = 6 , r = 2$ and the acceptance set contains $1 + 8 + 2 8 = 3 7$ different hash cells. Each cell contains a continuous family of ambient semantic vectors, so the accepted set can cover distinct semantic directions even though all its codes remain close to $h _ { t - 1 }$

Proposition 2 concerns the geometry of the ambient SBERT representation space and does not imply that every constructed vector corresponds to a realizable natural-language sentence. We therefore interpret it only as a geometric explanation for why short semantic hashes need not define a narrow region in representation space. Our empirical results provide complementary, aggregate evidence: among independently sampled natural continuations, HammingMark accepts a substantially broader subset than the compared semantic watermarking methods, as illustrated in Fig. 2 and reflected by the higher natural-candidate acceptance rate. These observations are consistent with, but do not by themselves prove, greater semantic diversity on the natural-language manifold.

Table 6: Distribution of watermark-valid candidates across eight semantic clusters.
<table><tr><td>Method</td><td>C1</td><td>C2</td><td>C3</td><td>C4</td><td>C5</td><td>C6</td><td>C7</td><td>C8</td><td>Total</td></tr><tr><td>SemStamp</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>1</td></tr><tr><td>K-SemStamp</td><td>1</td><td>1</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>2</td></tr><tr><td>SimMark</td><td>7</td><td>5</td><td>4</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>16</td></tr><tr><td>PMark</td><td>10</td><td>8</td><td>6</td><td>4</td><td>3</td><td>0</td><td>0</td><td>0</td><td>31</td></tr><tr><td>HammingMark</td><td>24</td><td>20</td><td>17</td><td>15</td><td>12</td><td>7</td><td>4</td><td>0</td><td>99</td></tr></table>

## F Semantic Coverage of Watermark-Valid Candidates

To further examine whether higher acceptance corresponds to broader semantic preservation, we sample 300 natural candidate continuations from the same prompt. To avoid evaluating semantic coverage directly in the representation space used by HammingMark for watermark generation and detection, we encode these candidates using the frozen sentence-transformers/all-mpnet-base-v2 evaluation model and cluster their normalized embeddings into K = 8 semantic regions, matching the partition granularity used in SemStamp and K-SemStamp. We then apply each watermarking method’s original constraint and encoder to the same candidate pool and examine the distribution of watermark-valid candidates across the externally constructed semantic clusters. The external evaluation encoder is used only for constructing the clusters and does not participate in candidate acceptance for any watermarking method.

As shown in Table 6, existing watermark constraints retain candidates from only a limited number of semantic regions. In contrast, HammingMark accepts candidates spanning seven of the eight clusters. Notably, even several low-frequency semantic modes remain accessible. This indicates that Hamming-Mark does not merely increase the number of watermark-valid candidates, but preserves a broader portion of the semantic modes present in natural generation, providing more direct evidence of alleviated semantic narrowing.

## G Semantic Diversity under Fixed Contexts

Natural acceptance measures how much probability mass of the original generation distribution remains compatible with a watermark constraint, but it does not directly characterize whether the resulting continuations remain semantically diverse. We therefore evaluate semantic diversity under fixed contexts. Using the same 100 prompts as in the natural-acceptance experiment, we independently generate 300 next-sentence continuations for each method from the same context.

To avoid measuring diversity directly in the representation space used by HammingMark for watermark generation and detection, we use a separate, frozen sentence-embedding model, sentence-transformers/all-mpnet-base-v2, solely as the evaluation encoder. This checkpoint is not used by any watermarking method during generation or detection. Let

$$
z _ { i } = \frac { E _ { \mathrm { e v a l } } ( S _ { i } ) } { \vert \vert E _ { \mathrm { e v a l } } ( S _ { i } ) \vert \vert _ { 2 } }
$$

denote the normalized evaluation embedding of continuation $S _ { i } ,$ where $E _ { \mathrm { e v a l } }$ is all-mpnet-base-v2. We define pairwise semantic diversity as

$$
D _ { \mathrm { p a i r } } = \frac { 2 } { N ( N - 1 ) } \sum _ { 1 \leq i < j \leq N } \left( 1 - z _ { i } ^ { \top } z _ { j } \right) ,\tag{41}
$$

Table 7: Semantic diversity under fixed contexts. Each method independently generates 300 continuations for each of 100 prompts. Higher values indicate greater semantic diversity.
<table><tr><td>Method</td><td>Pairwise diversity ↑</td><td>Diversity retention ↑</td></tr><tr><td>No watermark</td><td>0.284</td><td>1.000</td></tr><tr><td>SimMark</td><td>0.224</td><td>0.789</td></tr><tr><td>SemStamp</td><td>0.142</td><td>0.500</td></tr><tr><td>K-SemStamp</td><td>0.158</td><td>0.556</td></tr><tr><td>PMark</td><td>0.201</td><td>0.708</td></tr><tr><td>Random Code</td><td>0.184</td><td>0.648</td></tr><tr><td>HammingMark</td><td>0.269</td><td>0.947</td></tr></table>

where $N = 3 0 0$ . A larger value indicates that repeated generations span a broader range of semantic realizations according to the held-out evaluation encoder. We further report the diversity retention ratio

$$
R _ { \mathrm { d i v } } = { \frac { D _ { \mathrm { p a i r } } ^ { \mathrm { W M } } } { D _ { \mathrm { p a i r } } ^ { \mathrm { N o W M } } } } ,\tag{42}
$$

which measures the fraction of unwatermarked semantic diversity preserved after watermarking under the same evaluation metric.

As shown in Table 7, semantic watermarking generally reduces the diversity of continuations generated from the same context, but the degree of reduction differs substantially across methods. SemStamp and K-SemStamp exhibit the strongest concentration, retaining only 50.0% and 55.6% of the semantic diversity of unwatermarked generation. This is consistent with their region-based acceptance mechanisms, which restrict generation to designated portions of the semantic space. SimMark retains more diversity, but still reduces the pairwise distance from 0.284 to 0.224. PMark similarly shows a noticeable reduction, retaining approximately 70.8% of the unwatermarked diversity.

In contrast, HammingMark achieves a pairwise diversity of 0.269, corresponding to 94.7% diversity retention and remaining close to unwatermarked generation according to the held-out evaluation encoder. This result complements the natural-acceptance analysis: a high acceptance mass alone could still arise if the accepted continuations were concentrated around a small number of similar semantic realizations, whereas the present results show that HammingMark also preserves substantial variation among independently generated continuations. This provides additional evidence that the realized-sentence-centered Hamming constraint preserves not only more natural probability mass, but also a broader range of semantic alternatives.

The comparison with Random Code further isolates the effect of constraint placement. Although Random Code uses the same code-space coverage as HammingMark, it retains only 64.8% of the unwatermarked diversity. Thus, the advantage of HammingMark cannot be explained solely by admitting a larger set of hash codes. Rather, centering the Hamming neighborhood on the realized preceding sentence allows the admissible region to better follow the local semantic structure of natural generation, while the coarse many-to-one hash mapping leaves multiple semantic realizations simultaneously valid.

## H Deriving the Matching Threshold from Natural Generation

This section explains how the matching threshold T is determined by the natural matching behavior of unwatermarked generation and the desired watermark strength. We first derive an exact rule that holds for an arbitrary natural match-count distribution. We then obtain a compact approximation in terms of the hash length m and the natural expected number of matching bits, and finally instantiate the result for our default setting $m = 8 .$

## H.1 Exact Threshold from the Natural Match Distribution

Consider one sentence-generation step and condition on the realized context π. To simplify notation, we omit the sentence-step index throughout this section. Let

$$
P ( S \mid \pi )
$$

denote the base-model distribution over the next candidate sentence before watermarking, and let $h _ { \mathrm { p r e v } } \in$ $\{ 0 , 1 \} ^ { m }$ be the semantic hash of the preceding sentence.

For a candidate sentence S, define its number of matching hash bits with the preceding sentence as

$$
M ( S ) = { \mathrm { M a t c h } } \left( h _ { \mathrm { p r e v } } , H ( F ( S ) ) \right) \in \{ 0 , \ldots , m \} .\tag{43}
$$

When $S \sim P ( \cdot \mid \pi )$ , M is therefore a discrete random variable describing the matching behavior that arises naturally before any watermark constraint is applied.

For a threshold T, HammingMark accepts a candidate if

$$
M ( S ) \geq T .
$$

The probability that an unwatermarked candidate already satisfies this condition is

$$
a ( T ) = \operatorname* { P r } _ { S \sim P ( \cdot | \pi ) } [ M ( S ) \geq T ] .\tag{44}
$$

This quantity is central to the threshold selection: it measures how much of the natural next-sentence probability mass remains admissible under the watermark constraint.

Assuming rejection sampling continues until an admissible candidate is found, the distribution of the accepted sentence is simply the base distribution conditioned on $M ( S ) \geq T$

$$
Q _ { T } ( S \mid \pi ) = \frac { P ( S \mid \pi ) \mathbf { 1 } [ M ( S ) \ge T ] } { a ( T ) } .\tag{45}
$$

For every sentence in the accepted region,

$$
{ \frac { Q _ { T } ( S \mid \pi ) } { P ( S \mid \pi ) } } = { \frac { 1 } { a ( T ) } } .
$$

Hence the KL divergence from the natural distribution is

$$
\begin{array} { l } { { \displaystyle { \cal D } _ { \mathrm { K L } } ( Q _ { T } | | P ) = \sum _ { S : M ( S ) \geq T } Q _ { T } ( S \mid \pi ) \log \frac { Q _ { T } ( S \mid \pi ) } { P ( S \mid \pi ) } } } \\ { { = \displaystyle \sum _ { S : M ( S ) \geq T } Q _ { T } ( S \mid \pi ) \log \frac { 1 } { a ( T ) } } } \\ { { = \log \frac { 1 } { a ( T ) } = - \log a ( T ) . } } \end{array}\tag{46}
$$

Distributional watermark strength. We characterize the per-transition watermark strength through the distributional bias introduced by enforcing the watermark acceptance event. Specifically, under a fixed context π, we define

$$
\lambda ( T ; \pi ) \triangleq \frac { 1 } { \ln 2 } D _ { \mathrm { K L } } \left( Q _ { T } ( \cdot  { | } \pi )  { | } | P ( \cdot  { | } \pi ) \right) = - \log _ { 2 } a _ { \pi } ( T ) ,\tag{47}
$$

where

$$
a _ { \pi } ( T ) = \operatorname* { P r } _ { S \sim P ( \cdot | \pi ) } [ M ( S ) \geq T ]
$$

is the probability that a naturally generated candidate already satisfies the watermark constraint.

Intuitively, $\lambda ( T ; \pi )$ measures how unlikely the enforced watermark event is under natural generation. A smaller natural acceptance probability corresponds to a stronger watermark-specific selection bias and, in general, provides stronger per-transition evidence for distinguishing watermarked from unwatermarked generation. We therefore use λ as a distributional proxy for watermark strength, rather than as a direct measure of sequence-level detection performance, which also depends on factors such as text length, dependence across transitions, the detector statistic, and possible perturbations.

In particular, $\lambda ( T ; \pi ) = 1$ bit corresponds to an acceptance event with natural probability $1 / 2$ . More generally, requiring at least $\lambda _ { 0 }$ bits of distributional watermark strength is equivalent to

$$
\lambda ( T ; \pi ) \geq \lambda _ { 0 } \quad \Longleftrightarrow \quad a _ { \pi } ( T ) \leq 2 ^ { - \lambda _ { 0 } } .\tag{48}
$$

This relation directly yields a threshold-selection rule. Since

$$
a _ { \pi } ( T + 1 ) = a _ { \pi } ( T ) - \mathrm { P r } [ M = T \mid \pi ] \le a _ { \pi } ( T ) ,
$$

the natural acceptance probability is non-increasing in $T ,$ , and consequently $\lambda ( T ; \pi ) = - \log _ { 2 } { a _ { \pi } ( T ) }$ is non-decreasing in $T .$ . Under ideal rejection sampling, each independent candidate is accepted with probability $a _ { \pi } ( T )$ , so the expected number of sampled candidates required for one accepted sentence is

$$
\mathbb { E } [ N _ { T } \mid \pi ] = { \frac { 1 } { a _ { \pi } ( T ) } } .\tag{49}
$$

Thus, increasing $T$ generally induces a stronger watermark-specific distributional bias, but also rejects more natural probability mass and increases the expected sampling cost.

Accordingly, among all thresholds that achieve at least $\lambda _ { 0 }$ bits of distributional watermark strength, the least restrictive choice is

$$
\left| T ^ { \star } = \operatorname* { m i n } \left\{ T \in \{ 0 , \ldots , m \} : \operatorname* { P r } [ M \geq T \mid \pi ] \leq 2 ^ { - \lambda _ { 0 } } \right\} . \right|\tag{50}
$$

This choice satisfies the desired distributional-strength requirement while maximizing the retained natural acceptance probability, or equivalently minimizing the expected rejection-sampling cost.

equation 50 is exact for the idealized unlimited-budget rejection-sampling procedure and makes no parametric assumption about the natural distribution of $M .$ . In practice, $T ^ { \star }$ can therefore be estimated directly from the empirical match-count distribution of unwatermarked continuations.

## H.2 Estimating the Matching Threshold from Natural Statistics

The natural match-count distribution depends on both the semantic relation between adjacent sentences and the fixed random projection used by the watermark key. Therefore, we do not assume that the marginal distribution of M follows an exact parametric form. Instead, we use a simple moment-matched distribution only to obtain an interpretable estimate of the threshold scale.

Let

$$
\mu = \mathbb { E } [ M ]
$$

denote the mean number of matching bits observed from naturally generated sentence pairs. Since $M \in$ $\{ 0 , \ldots , m \}$ , we summarize the average bit-wise matching level by

$$
\bar { p } = \frac { \mu } { m } .
$$

We then introduce the following moment-matched proxy:

$$
\widetilde { \cal M } \sim \mathrm { B i n o m i a l } \left( m , \frac { \mu } { m } \right) .\tag{51}
$$

This approximation matches the empirical mean, $\mathbb { E } [ \widetilde { M } ] = \mu ,$ , while ignoring the heterogeneity of semantic similarities across contexts. It is used only for estimating a reasonable operating threshold rather than for characterizing the exact natural match-count distribution.

Under this approximation, the estimated natural acceptance probability for threshold T is

$$
\widetilde { a } ( T ) = \mathrm { P r } [ \widetilde { M } \ge T ] = \sum _ { k = T } ^ { m } \binom { m } { k } \left( \frac { \mu } { m } \right) ^ { k } \left( 1 - \frac { \mu } { m } \right) ^ { m - k } .\tag{52}
$$

Following the definition of watermark strength in the previous section, we define the corresponding estimated strength as

$$
\begin{array} { r } { \widetilde \lambda ( T ) = - \log _ { 2 } \widetilde a ( T ) . } \end{array}\tag{53}
$$

Here, $\widetilde { \lambda } ( T )$ should be interpreted only as an estimate of the distributional strength associated with the selected threshold. The actual strength remains determined by the true natural acceptance probability $a ( T ) = \operatorname* { P r } [ M \geq T ]$

For a desired strength level $\lambda _ { 0 }$ , the same approximation provides a direct estimate of the required matching threshold. Let

$$
F _ { m , p } ( k ) = \operatorname* { P r } [ X \leq k ] , \qquad X \sim \operatorname { B i n o m i a l } ( m , p ) ,
$$

and define the discrete quantile

$$
F _ { m , p } ^ { - 1 } ( u ) = \operatorname* { m i n } \{ k : F _ { m , p } ( k ) \geq u \} .
$$

Since

$$
\operatorname* { P r } [ \widetilde { M } \ge T ] = 1 - F _ { m , \mu / m } ( T - 1 ) ,
$$

the estimated threshold is

$$
\widehat T ( m , \mu , \lambda _ { 0 } ) = 1 + { \cal F } _ { m , \mu / m } ^ { - 1 } \left( 1 - 2 ^ { - \lambda _ { 0 } } \right) .\tag{54}
$$

equation 54 provides a convenient mapping from the observed natural matching level $\mu$ and a desired watermark strength $\lambda _ { 0 }$ to an estimated threshold. It should not be interpreted as an exact solution for the true generation distribution, since natural sentence pairs may exhibit substantial variation in semantic similarity and hence a non-binomial marginal distribution of $M$

For additional intuition, applying a normal approximation to. equation 51 gives

$$
\widetilde { M } \approx \mathcal { N } \left( \mu , \mu \left( 1 - \frac { \mu } { m } \right) \right) .
$$

With a continuity correction, this yields the rough closed-form estimate

$$
\widehat { T } \approx \left\lceil \mu + \sqrt { \mu \left( 1 - \frac { \mu } { m } \right) } \Phi ^ { - 1 } \left( 1 - 2 ^ { - \lambda _ { 0 } } \right) + \frac { 1 } { 2 } \right\rceil ,\tag{55}
$$

where $\Phi ^ { - 1 }$ is the inverse standard-normal CDF. We use this expression only for interpretation; all numerical estimates below are computed from the binomial tail in equation 52.

## H.3 Estimated Operating Point for the Default Configuration

We next apply the above approximation to the default HammingMark configuration. We use a semantic hash length of

$$
m = 8 ,
$$

and natural unwatermarked sentence pairs exhibit an average match count of approximately

$$
\mu \approx 4 . 8 .
$$

The moment-matched proxy is therefore

$$
\widetilde { M } \sim \mathrm { B i n o m i a l } ( 8 , 0 . 6 0 ) .\tag{56}
$$

We use $\lambda _ { 0 } = 1$ bit as a convenient reference strength. Under the definition

$$
\lambda = - \log _ { 2 } a ,
$$

one bit corresponds to an acceptance event whose natural probability is approximately one half. Substituting $\lambda _ { 0 } = 1$ into equation 54 gives

$$
\widehat { T } = 1 + F _ { 8 , 0 . 6 0 } ^ { - 1 } ( 0 . 5 ) = 6 .\tag{57}
$$

The same result can be seen directly from the estimated upper-tail probabilities:

$$
\widetilde { a } ( 5 ) = \mathrm { P r } [ \widetilde { M } \ge 5 ] \approx 0 . 5 9 4 1 ,\tag{58}
$$

$$
\widetilde { a } ( 6 ) = \mathrm { P r } [ \widetilde { M } \ge 6 ] \approx 0 . 3 1 5 4 .\tag{59}
$$

The corresponding estimated watermark strengths are

$$
\widetilde { \lambda } ( 5 ) = - \log _ { 2 } ( 0 . 5 9 4 1 ) \approx 0 . 7 5 \mathrm { b i t s } ,\tag{60}
$$

$$
\begin{array} { r } { \widetilde \lambda ( 6 ) = - \log _ { 2 } ( 0 . 3 1 5 4 ) \approx 1 . 6 6 \mathrm { b i t s } . } \end{array}\tag{61}
$$

Thus, under the moment-matched approximation, the one-bit reference point lies between $T = 5$ and $T = 6$ , making $T = 6$ a natural estimated operating point for $m = 8$ . This estimate also has a simple interpretation through equation 55. When $\lambda _ { 0 } = 1$

$$
\Phi ^ { - 1 } ( 1 - 2 ^ { - 1 } ) = \Phi ^ { - 1 } ( 0 . 5 ) = 0 ,
$$

and the approximate threshold reduces to

$$
\widehat T \approx \left\lceil \mu + \frac { 1 } { 2 } \right\rceil .
$$

For $\mu = 4 . 8$ , this again gives

$$
{ \widehat { T } } \approx \left\lceil 5 . 3 \right\rceil = 6 .
$$

We emphasize that these calculations are intended only to estimate the appropriate threshold scale. The true natural match-count distribution is context dependent and need not follow the moment-matched binomial proxy. Accordingly, the values $\widetilde { a } ( T )$ and $\widetilde { \lambda } ( T )$ above should not be interpreted as measured acceptance rates or exact watermark strengths. Empirical acceptance and detection performance are evaluated separately in the experiments. The analysis here only provides an intuitive explanation for why a threshold around $T = 6$ is a reasonable default for an 8-bit semantic hash.

Finite-budget implementation. The analysis above characterizes the idealized rejection-sampling procedure that continues until a valid candidate is obtained. In practice, we impose a finite resampling budget for computational efficiency. Therefore, Eqs. (39)–(44) should be interpreted as the exact characterization of the unlimited-budget procedure, while our implementation provides a finite-budget approximation to it.

Empirical fallback frequency. In our implementation, the maximum resampling budget is $B = 1 6 .$ When no candidate satisfies the watermark constraint within this budget, the last sampled candidate is retained. We additionally measure the empirical fallback rate as

$$
r _ { \mathrm { f b } } = \frac { \# \{ \mathrm { s e n t e n c e ~ s t e p s ~ e x h a u s t i n g ~ a l l ~ } B \mathrm { ~ a t t e m p t s } \} } { \# \{ \mathrm { g e n e r a t e d ~ s e n t e n c e ~ s t e p s } \} } .
$$

Whenever fallback does not occur, the finite-budget procedure coincides with the ideal rejection-sampling procedure analyzed above. Across all evaluated settings, fallback occurs in only 0.6%–2.7% of sentence steps, as shown in Table 8. Thus, the ideal conditional-distribution analysis provides a close description of the generation procedure for the large majority of sentence steps.

Table 8: Empirical fallback rates under the maximum resampling budget $B = 1 6$
<table><tr><td>Dataset</td><td>Generator</td><td>Fallback rate (%)</td></tr><tr><td>C4</td><td>Mistral-7B</td><td>1.3</td></tr><tr><td>BookSum</td><td>Mistral-7B</td><td>0.8</td></tr><tr><td>C4</td><td>Qwen2.5-7B</td><td>1.2</td></tr><tr><td>BookSum</td><td>Qwen2.5-7B</td><td>0.6</td></tr><tr><td>ELI5</td><td>Mistral-7B</td><td>2.3</td></tr><tr><td>Multi-News</td><td>Mistral-7B</td><td>2.7</td></tr></table>

Table 9: Effect of semantic hash length on clean detection and attack robustness. Each entry reports TPR@1%/TPR@5%/AUROC.
<table><tr><td>M</td><td>T</td><td>Clean</td><td>Parrot</td><td>DIPPER</td></tr><tr><td>4</td><td>3</td><td>0.76/0.86/0.97</td><td>0.75/0.86/0.97</td><td>0.74/0.83/0.96</td></tr><tr><td>8</td><td>6</td><td>0.98/1.00/1.00</td><td>0.94/0.99/1.00</td><td>0.93/0.98/0.99</td></tr><tr><td>16</td><td>12</td><td>0.98/1.00/1.00</td><td>0.78/0.87/0.98</td><td>0.73/0.81/0.96</td></tr><tr><td>32</td><td>24</td><td>0.99/1.00/1.00</td><td>0.63/0.73/0.90</td><td>0.64/0.74/0.91</td></tr></table>

## I Effect of Semantic Hash Length

We study the effect of the semantic hash length M by evaluating $M \in \{ 4 , 8 , 1 6 , 3 2 \}$ on C4 with Mistral-7B. For all settings, we fix the required matching ratio to 0.75, giving thresholds $T \in \{ 3 , 6 , 1 2 , 2 4 \}$ respectively. All other generation and detection settings remain unchanged. Robustness is evaluated under Parrot and DIPPER rewriting attacks.

As shown in Table 9, $M = 4$ provides insufficient discriminative resolution: with only four bits, unwatermarked transitions can also frequently satisfy the $3 / 4$ matching requirement, resulting in substantial overlap between watermarked and unwatermarked scores even without attacks. For example, under an independent balanced-bit approximation, the probability of matching at least three bits is already $5 / 1 6 = 3 1 . 2 5 \%$

Increasing M improves hash resolution but gradually reduces robustness under Parrot and DIPPER. Longer hashes provide a more precise estimate of semantic similarity and therefore reduce the quantization slack of short codes. After paraphrasing perturbs the sentence embeddings, more hyperplane signs may change, making transitions near the 0.75 boundary less likely to remain valid. Overall, $M = 8$ achieves the best balance between clean detectability and tolerance to meaning-preserving perturbations.

## J Reproducibility Details

Tasks and models. We evaluate HammingMark on C4 and BookSum for open-ended generation, using Mistral-7B-Instruct and Qwen2.5-7B-Instruct as generators. We further evaluate Multi-News summarization and ELI5 long-form question answering as more constrained generation tasks.

Baselines and configuration. We compare HammingMark with token-level methods (KGW, SynthID, MorphMark, and SIR) and sentence-level semantic watermarks (SemStamp, K-SemStamp, SimMark, and PMark). HammingMark uses an $m = 8 – \mathrm { b i t }$ semantic hash and accepts a transition when at least $T = 6$ bits match.

Evaluation protocol. For KGW, SynthID, MorphMark, SIR, SemStamp, and K-SemStamp, we follow their respective configurations in MarkLLM (Pan et al., 2024) without modification, including all algorithm-specific hyperparameters and, where applicable, sentence-sampling procedures, resampling budgets, and fallback strategies when no valid candidate is found. For PMark, we evaluate the offline variant using its default parameter settings. For SimMark, we use the default parameter settings reported in its original paper.

For detection, we calibrate the threshold $\tau _ { d } ( \alpha )$ separately for each method and evaluation dataset at a target false-positive rate α. Specifically, for each dataset, we randomly sample 500 examples disjoint from its test set to construct the calibration set. All methods use the same calibration examples and threshold-selection procedure, with method-specific thresholds determined from their respective detection scores. Each calibrated threshold is then fixed and applied to the corresponding test set. We report TPR@1%, TPR@5%, and AUROC on both clean and attacked texts. Robustness is evaluated under Parrot (Damodaran, 2021) and Pegasus paraphrasing (Zhang et al., 2020), DIPPER rewriting (Krishna et al., 2023), and API-based rewriting with DeepSeek-V4-Pro (Xu et al., 2026).

We adopt dataset-specific detection threshold calibration to ensure a controlled comparison across methods at the same target false-positive rate. Nevertheless, a fixed threshold would simplify deployment when representative calibration data are unavailable. Empirically, for the Global Bits detector evaluated on clean text, the calibrated sequence-level thresholds across the evaluated datasets remain within 5% of 0.75 in relative terms. Under a common measure of relative threshold variation, HammingMark exhibits the smallest cross-dataset variation among the compared methods, whereas KGW-family detectors exhibit variations exceeding 30%. These observations support the stability of the Global Bits detection threshold across the evaluated clean-text datasets. Its concentration around 0.75 is also consistent with the generation rule, which requires at least six matching bits out of eight for each accepted transition. Together, these results suggest that 0.75 is a reasonable default decision threshold for Global Bits in comparable clean-text settings. However, the generation acceptance condition and the sequence-level detection threshold serve different purposes: their numerical proximity does not guarantee that a fixed threshold will maintain the target false-positive rate on unseen distributions. We therefore retain dataset-specific calibration for the reported comparisons and regard 0.75 as an empirically motivated deployment default whose false-positive rate should be validated in the intended setting.

Datasets and generation. We randomly select 500 samples from C4 and BookSum, respectively, and 100 samples from ELI5 and Multi-News. The target generation length is set to 200 tokens for all tasks. Unless otherwise specified, all decoding parameters follow the default configurations provided by Mark-LLM. Sentence boundaries are identified using the NLTK sentence tokenizer integrated into MarkLLM.

HammingMark implementation. HammingMark uses the SBERT encoder released with the Sem-Stamp implementation. The default hash length and matching threshold are $m = 8$ and $T \ = \ 6 ,$ , respectively. The random projection matrix is initialized from the corresponding experimental seed. The maximum number of resampling attempts is set to 16. If no candidate satisfies the watermark constraint within this budget, the latest candidate is used as a fallback and remains included in subsequent watermark detection.

Calibration, randomness, and hardware. Detection thresholds and attack configurations follow the standardized MarkLLM evaluation pipeline and are kept consistent across all baselines. Reported results are averaged over three random seeds, with detection rates varying by less than one percentage point across runs. All experiments are conducted on a single NVIDIA H200 GPU.

## K Effect of the Matching Threshold

We further study the effect of the matching threshold T while fixing the semantic hash length to $m =$ 8. We evaluate $T \in \{ 5 , 6 , 7 \}$ on C4 with Mistral-7B using the Global Bits detector. The results are summarized in Table 10.

The matching threshold directly controls the strength of the transition-level watermark constraint. When $T = 5$ , the acceptance condition is relatively weak: a large fraction of naturally generated transitions already satisfy the watermark rule. This substantially reduces the sampling cost to only 1.3 candidates per accepted sentence and leaves generation quality nearly unchanged. However, because the accepted transitions are less distinguishable from those occurring naturally in unwatermarked text, both clean detection and robustness under paraphrasing and rewriting decrease.

Table 10: Effect of the matching threshold $T$ on C4 with Mistral-7B using the Global Bits detector. Detection results are reported as TPR@1% / TPR@5% / AUROC. Lower sampling cost is better.
<table><tr><td>T</td><td>Clean</td><td>Parrot</td><td>DIPPER</td><td>PPL↓</td><td>Samples / sent. ↓</td></tr><tr><td>5</td><td>0.71 / 0.83 / 0.95</td><td>0.32 / 0.56 / 0.81</td><td>0.27 / 0.40 / 0.74</td><td>4.61</td><td>1.3</td></tr><tr><td>6</td><td>0.98 / 1.00 / 1.00</td><td>0.94 / 0.99 / 1.00</td><td>0.93 / 0.98 / 0.99</td><td>4.61</td><td>2.2</td></tr><tr><td>7</td><td>0.98 / 1.00 / 1.00</td><td>0.98 / 0.99 / 1.00</td><td>0.98 / 0.99 / 1.00</td><td>5.33</td><td>6.4</td></tr></table>

Table 11: Cross-key detection on C4 with Mistral-7B. Watermarked text is generated using $R ^ { \star }$ . TPR is measured at a target FPR of 1%.
<table><tr><td>Detection key</td><td>TPR@1%↑</td></tr><tr><td>Correct key  $R ^ { \star }$ </td><td>0.99</td></tr><tr><td>Independent wrong key R&#x27;</td><td>0.08</td></tr></table>

Increasing the threshold to $T = 7$ has the opposite effect. The stricter matching requirement makes accepted transitions more strongly biased toward the watermark condition. Clean-text detection remains comparable to $T = 6 .$ , while robustness under both Parrot and DIPPER improves. However, this stronger constraint considerably reduces the natural acceptance probability, increasing the average sampling cost from 2.2 to 6.4 candidates per accepted sentence. We also observe a substantially larger change in perplexity relative to the unwatermarked generation, indicating that the stronger filtering perturbs the original generation distribution more noticeably.

Overall, $T = 6$ provides the best empirical trade-off among watermark strength, robustness, generation fidelity, and sampling efficiency. A smaller threshold admits too many naturally occurring transitions and therefore provides insufficient watermark separation, whereas a larger threshold produces a stronger and more robust watermark at the cost of substantially more restrictive sampling. These results are consistent with our use of $T = 6$ as the default configuration throughout the main experiments.

## L Cross-Key Detection and Key Specificity

Motivation and setup. HammingMark uses a randomly initialized projection matrix as part of the watermark key. We examine whether the detection signal is specific to the matrix used during generation. Watermarked text is generated on C4 with Mistral-7B using the default configuration $( m = 8 , T = 6 )$ and a secret projection matrix R<sup>⋆</sup>. During detection, we compare the correct key $R ^ { \star }$ with an independently sampled wrong key $R ^ { \prime } \ne R ^ { \star }$ . Unwatermarked generations are used as negatives, and the detection threshold is calibrated at a target false-positive rate of 1%.

Results. As shown in Table 11, replacing the generation key with an independent projection matrix reduces TPR@1% from 0.99 to 0.08. The resulting 91-percentage-point gap indicates that the dominant watermark signal is strongly specific to the secret projection matrix used during generation. The non-zero wrong-key detection rate nevertheless suggests a weaker key-independent effect.

Discussion. The cross-key results provide strong evidence that HammingMark’s detection signal is highly key-specific. Replacing the correct projection matrix $R ^ { \star }$ with an independently sampled matrix $R ^ { \prime }$ reduces TPR@1% from 0.99 to 0.08, corresponding to a 91-percentage-point drop. This large degradation indicates that high detection performance cannot be explained primarily by a generic increase in inter-sentence semantic continuity.

A small residual signal under the wrong key is nevertheless expected. Because HammingMark centers its acceptance region on the preceding sentence, it introduces a mild preference for semantically coherent transitions. Such transitions may exhibit above-random hash agreement under multiple randomhyperplane projections, allowing an independent matrix $R ^ { \prime }$ to capture a limited key-independent component of the signal. However, $R ^ { \prime }$ does not reproduce the specific hyperplane partition or the bit-level Hamming constraints imposed during generation under $R ^ { \star }$ . Only the correct detector evaluates each transition in the same secret projection geometry that governed candidate acceptance, allowing key-specific evidence to accumulate consistently across the sequence. The substantial gap between correct-key and wrong-key detection therefore shows that, while generic semantic continuity contributes modestly, the dominant detection signal is tied to the secret projection matrix. Finally, the wrong-key TPR should not be interpreted as a false-positive rate, because the evaluated samples are genuinely watermarked texts rather than unwatermarked negatives.

## M Comparison with Continuous Similarity-Based Acceptance

To determine whether the improvement of HammingMark arises merely from using the preceding sentence as a relational reference or from the discrete Hamming neighborhood itself, we construct a continuous-similarity baseline. This baseline uses the same SBERT semantic encoder as HammingMark and directly computes the cosine similarity between a candidate sentence and its preceding sentence. A candidate is accepted when its similarity exceeds a predefined threshold and is otherwise resampled. This design is conceptually related to inter-sentence similarity-based watermarking methods such as Sim-Mark, since both select candidates directly according to their relations in a continuous semantic space. It therefore provides a controlled comparison between continuous similarity constraints and Hamming code-based acceptance.

We use the same 100 prompts as in the natural-acceptance experiment and set the cosine-similarity threshold to 0.63. This produces an average natural acceptance rate of approximately 0.58, matching that of HammingMark. Despite accepting a similar fraction of natural candidates, the continuous baseline achieves only 0.68 TPR@1%, 0.76 TPR@5%, and 0.93 AUROC on clean text, substantially below HammingMark. Its fallback rate also increases from 1.3% for HammingMark to 7.5%.

These results reveal two limitations of direct continuous-similarity acceptance. First, naturally generated adjacent sentences already tend to be locally coherent and semantically similar. Consequently, a substantial fraction of unwatermarked transitions also exceeds the cosine threshold. When inter-sentence similarity is used directly as the watermark signal, the score distributions of watermarked and unwatermarked texts remain relatively close, limiting clean-text detectability. This illustrates a fundamental trade-off for continuous similarity-based methods: increasing the threshold may strengthen the watermark signal, but it also imposes a more restrictive semantic constraint and increases the sampling cost.

Second, the average natural acceptance rate does not capture variation across contexts. Although most adjacent sentences can readily satisfy the threshold of 0.63 because of local semantic continuity, some valid discourse transitions naturally exhibit lower cosine similarity. Examples include contrast, topic shifts, the introduction of new facts, and transitions from background information to a conclusion. In such contexts, a hard continuous threshold may repeatedly reject otherwise appropriate candidates, eventually exhausting the sampling budget and triggering fallback. Thus, two methods with the same average natural acceptance rate can have substantially different numbers of low-acceptance contexts. The considerably higher fallback rate of the continuous baseline is consistent with this context-dependent effect. Because a fallback candidate is not guaranteed to satisfy the watermark constraint, these events further reduce the consistency of the accumulated sequence-level signal.

HammingMark alleviates these limitations by defining its acceptance neighborhood in a compact semantic hash space. Semantically similar sentences are more likely to receive similar hash codes, but Hamming agreement is not a deterministic hard cutoff on continuous cosine similarity. Owing to the coarse, manyto-one nature of a short semantic hash, candidates with different continuous similarity values may share the same or neighboring hash codes. As a result, some candidates with relatively low cosine similarity but valid discourse functions can still satisfy the Hamming constraint. Unlike a direct similarity threshold,

the resulting acceptance region does not correspond to a single interval in the continuous semantic space.   
This quantization slack leaves more flexibility for contrast, information introduction, and topic transitions.

The discrete representation also provides more stable sequence-level evidence. HammingMark accumulates agreement over multiple projected bits, and the Global Bits detector retains partial evidence even when a modified transition no longer satisfies the complete generation threshold. In contrast, the continuous baseline relies primarily on generic inter-sentence similarity that is already present in natural text, making its detection signal more likely to overlap with the unwatermarked distribution. The random pro jection matrix additionally makes the HammingMark signal key-specific, whereas raw cosine similarity mainly captures key-independent semantic coherence.

Overall, this experiment shows that the advantage of HammingMark cannot be attributed solely to using the preceding sentence as the semantic reference. Even after matching the average natural acceptance rate, direct continuous similarity acceptance produces weaker detection and a substantially higher fallback rate. Compared with similarity-based selection mechanisms related to SimMark, the short Hamming representation provides a less rigid quantized acceptance region, more stable sampling behavior across contexts, and key-specific evidence that can be accumulated at the sequence level.