![](images/975055b6d4c4415b06f56dc699890df5a6d85bf3f6fdc606ebedacb05c0444de.jpg)

# The Sirens’ Song: When Proximal Background Context Overshadows Distant Evidence

Xiaoyu Yang, Jie Lu, Wei Duan, En Yu Australian Artificial Intelligence Institute (AAII), Faulty of Engineering and Information Technology, University of Technology Sydney, Australia.

## Abstract

Long-context LLMs focus on retrieving distant evidence from extensive context, yet existing work has largely focused on overcoming distance alone. In this work, we identify the Proximity Trap, insufficient attention to distant evidence often arises less from distance itself than from cumulative competition with abundant, task-irrelevant proximal background. To address the Proximity Trap, we introduce LYRA (Long-context heavY-tailed Relevance Alignment), a t-distributed directional matching mechanism that reshapes the context retrieval distribution, directing more attention mass toward task-relevant evidence, while preserving the relative positional information encoded. Extensive experiments on LongBench-v2, RULER, and LongBench demonstrate consistent improvements across context lengths and task categories. We further introduce ProxBench, a multi-level fine-grained benchmark for evaluating distant evidence utilization under increasing proximal background interference.

Project page: https://xiaoyuyoung.github.io/LYRA/

## 1 Introduction

Recent work on long-context LLMs (Wang et al., 2026b; Zhang et al., 2026; Mudarisov et al., 2025) has focused on improving access to distant evidence. However, access alone does not ensure that such evidence will be used, while proximal, task-irrelevant context may also compete for attention. This raises a less examined question: when a model overlooks distant evidence, is distance alone responsible, or does proximal background also impede its use?

Prioritizing recent context is a reasonable inductive bias, as nearby tokens often provide useful information for the current prediction. Accordingly, modern LLMs commonly encode relative positions through RoPE (Su et al., 2024), making QK scores explicitly dependent on relative distance and potentially attenuating the scores of distant tokens. While this positional bias supports local context modeling, positional proximity does not necessarily indicate task relevance: essential evidence may lie far from the query, whereas nearby content may provide little information for the required prediction. This mismatch raises uncertainty about how attention is allocated when distant evidence competes with proximal background.

In this work, we present a counter-intuitive observation: reducing attention to proximal background can improve long-context understanding. As shown in Fig. 1, masking proximal background tokens consistently improves the overall accuracy on LongBench-v2 compared with the unmasked baseline, with a more pronounced improvement on Long In-context Learning. This result is striking because removing accessible context improves performance rather than degrading it. It suggests that the difficulty of using distant evidence cannot be attributed to distance alone; competition from proximal background also plays a critical role. We refer to this phenomenon as the Proximity Trap.

This counter-intuitive observation is closely related to how attention combines positional distance with contextual competition. Long-context inputs are inherently non-uniform: the evidence required for a prediction is often sparse and distant, whereas much of the nearby context serves only as background. Relative positional encoding makes QK matching dependent on distance and can weaken the directional agreement between the query and distant evidence. Under softmax normalization, however, this weakened evidence competes not with a single nearby token, but with all proximal background tokens simultaneously. Although each background token may have limited task relevance, their contributions accumulate in the softmax denominator and can collectively draw attention away from distant evidence. This interaction between positional attenuation and cumulative background competition underlies the Proximity Trap and explains why masking proximal background can improve evidence utilization.

![](images/3acf04f019ba14594d9d442cccb7fdd6140396d9e2d6e6cfae0fd218d3603800.jpg)  
Figure 1: Less attention to proximal background improves long-context understanding. On LongBench-v2 (Bai et al., 2025), we progressively mask the context tokens nearest to the query and report accuracy over the full benchmark and the Long Incontext Learning subset. The horizontal dashed line denotes the performance without masking, corresponding to zero masked tokens.

Consequently, we introduce LYRA, a simple t-distributed directional matching mechanism. Preserving distant evidence does not require explicitly favoring distant positions or manually suppressing proximal context. Instead, LYRA measures the directional agreement between the RoPE-transformed query and keys, and then applies a t-distributed transformation to reshape their score differences before

softmax normalization. This design reduces misleading advantages held by weakly related proximal background while making strongly matched evidence more distinguishable. Crucially, LYRA replaces only the conventional QK scoring function, leaving RoPE, causal masking, softmax normalization, and value aggregation unchanged. As a result, LYRA preserves distant evidence according to relevance rather than position, allowing it to resist cumulative background competition without indiscriminately promoting all distant tokens.

In summary, our main contributions are as follows:

• We characterize the Proximity Trap, a counter-intuitive phenomenon in which proximal, task-irrelevant background can collectively overshadow distant but relevant evidence.

• LYRA is introduced as a t-distributed directional matching mechanism that reshapes the context retrieval distribution to redistribute attention mass toward task-relevant evidence while preserving the relative positional information encoded.

• Extensive experiments are conducted on various benchmarks. The results demonstrate consistent improvements across different context lengths and task categories, supporting the effectiveness and generalization of LYRA.

• We introduce ProxBench, a multi-level controlled benchmark for evaluating distant evidence utilization under increasing proximal background interference.

## 2 The Proximity Trap

In this section, we first introduce the preliminaries in Section 2.1. Subsequently, we establish our central observation that proximal background context overshadows distant evidence in Section 2.2. Moreover, we provide an explanation of its underlying mechanism through cumulative attention competition in Section 2.3.

## 2.1 Preliminaries

Given a query at the current decoding position t, let $\mathbf { q } _ { t } \in \mathbb { R } ^ { d }$ denote the query representation and let $( \mathbf { k } _ { i } , \mathbf { v } _ { i } )$ denote the key–value pair at each historical position $i < t .$ . Standard attention first computes the QK score between the query and each key, followed by softmax normalization:

$$
s _ { t , i } = \frac { \mathbf { q } _ { t } ^ { \top } \mathbf { k } _ { i } } { \sqrt { d } } , \qquad a _ { t , i } = \frac { \exp ( s _ { t , i } ) } { \sum _ { j < t } \exp ( s _ { t , j } ) } .\tag{1}
$$

The resulting attention output is $\begin{array} { r } { \mathbf { o } _ { t } = \sum _ { i < t } a _ { t , i } \mathbf { v } _ { i } } \end{array}$ . Since $a _ { t , i } \geq 0$ and $\textstyle \sum _ { i < t } a _ { t , i } = 1$ , all historical positions compete for a finite amount of attention weight. Our analysis therefore focuses on how the QK scores assigned to different historical positions determine this competition when the model generates the token at position t.

In particular, the QK score defined in Eq. (1) becomes explicitly position-dependent when RoPE (Su et al., 2024) is applied to the query and key representations. Let $\bar { \mathbf { R } _ { m } } \bar { \in } \mathbb { R } ^ { d \times d }$ denote the rotation matrix associated with position $m$ . RoPE transforms the query and key as $\widetilde { \mathbf { q } } _ { t } = \mathbf { R } _ { t } \mathbf { q } _ { t }$ and $\widetilde { \mathbf { k } } _ { i } = \mathbf { R } _ { i } \mathbf { k } _ { i }$ <sub>i</sub>, respectively. The resulting QK score is:

$$
s _ { t , i } ^ { \mathrm { R o P E } } = \frac { \widetilde { \mathbf { q } } _ { t } ^ { \intercal } \widetilde { \mathbf { k } } _ { i } } { \sqrt { d } } = \frac { \mathbf { q } _ { t } ^ { \intercal } \mathbf { R } _ { t } ^ { \intercal } \mathbf { R } _ { i } \mathbf { k } _ { i } } { \sqrt { d } } = \frac { \mathbf { q } _ { t } ^ { \intercal } \mathbf { R } _ { - ( t - i ) } \mathbf { k } _ { i } } { \sqrt { d } } ,\tag{2}
$$

where $\mathbf { R } _ { t } ^ { \top } \mathbf { R } _ { i } = \mathbf { R } _ { - ( t - i ) }$ depends only on the relative offset $t - i ,$ , where $\Delta = t - i \ge 0$ denotes the relative distance. Consequently, RoPE makes the QK score jointly dependent on the content representations $\mathbf { q } _ { t }$ and $\mathbf { k } _ { i }$ and their relative position.

Building on the relative-position-dependent scoring formulation above, we denote task-relevant information far from the query as Distant Evidence E and nearby content not required for the prediction as Proximal Background B. Importantly, B does not need to be adversarial, corrupted, or linguistically unnatural. It may consist of ordinary contextual or grammatical content that is coherent within the input but uninformative for the required prediction.

Remark 2.1. Counterintuitively, distance does not determine task relevance: E can be distant but essential, whereas B can be nearby but uninformative. Nevertheless, tokens in both spans enter the same softmax normalization and therefore competefor attention weight

## 2.2 Proximal Background Context Overshadows Distant Evidence

Building on the preceding definitions, we conduct two complementary experiments to examine the competition between distant evidence $\mathcal { E }$ and proximal background B. The first determines whether pretrained LLMs fail to locate distant evidence or instead retrieve it but have its signal diluted by proximal background. The second compares moving the evidence closer with attenuating proximal background to identify which intervention more effectively restores evidence attention and answer confidence.

First, to distinguish whether the model fails to locate distant evidence or loses it under attention competition, we compare the layer-wise attention assigned to three regions: distant evidence, its neighboring tokens, and proximal background near the query. For each layer, we report the mean per-token attention density normalized by uniform attention, where a value of 1 denotes the uniform baseline. These statistics are computed using Qwen3-8B (Bai et al., 2025) on LongBench-v2 (Bai et al., 2025).

As illustrated in Fig. 2, the three regions are not consistently separated in the early and middle layers. From approximately layer 19 onward, however, attention to distant evidence increases sharply, whereas its neighborhood remains largely below the uniform baseline. This contrast shows that the model can precisely identify distant evidence among ordinary tokens at the same distant location. However, proximal background also receives substantial attention. Although its per-token attention is generally lower than that of distant evidence, its abundance produces sufficient cumulative attention mass to overshadow the evidence. The central issue is therefore not whether the model can locate distant evidence, but whether that evidence can withstand competition from abundant proximal background after being identified.

Subsequently, building on this observation, we examine whether evidence utilization in Qwen3-8B is constrained primarily by evidence distance or by competition from proximal background. Figure 3 compares two interventions, moving distant evidence closer and attenuating proximal background, and reveals their distinct effects on evidence attention and answer confidence. For each evidence-dependent question, we compare moving distant evidence closer with proximal background attenuation, where the latter suppresses QK scores of nearby task-irrelevant context without changing the input tokens or their positions. We report the resulting changes in Evidence Attention and correct-answer confidence relative to the original input.

Accordingly, as Fig. 3 illustrates, the blue points span all four quadrants, indicating that moving distant evidence closer does not consistently improve the performance. In contrast, the orange points are concentrated toward increased evidence attention, with many also showing improved confidence. This divergence suggests that shortening the evidence distance does

![](images/6fc7e46351c89f2304c26855d5202869cc07b80dbc9d589ee39b940f8dbcf3ad.jpg)  
Figure 2: Distant Evidence is selectively retrieved but still diluted by cumulative competition from proximal background. We report layer-wise attention density for distant evidence, its neighboring tokens, and proximal background near the query.

not remove competition from abundant proximal background. Attenuating proximal background instead directly weakens this competition, allowing more attention to be reassigned to relevant evidence. Therefore, compared with moving distant evidence closer, proximal background attenuation produces a more consistent improvement in evidence utilization.

Taken together, these two experiments reveal the following counter-intuitive finding:

Finding 2.2 (The Proximity Trap). Insufficient attention to distant evidence often arises lessfrom distance itself. Instead, it is more from cumulative competition with abundant taskirrelevant proximal background context. In this sense, proximal background acts as the Sirens’ Song by drawing attention toward nearby but uninformative content and away from the distant evidence that matters.

## 2.3 Mechanism: Proximal Background Overshadowing

To better understand why task-relevant distant evidence can be overshadowed by task-irrelevant proximal background, we examine how relative position alters QK scores and how softmax converts these changes into attention competition.

Decomposing Eq. (2) into $d / 2$ two-dimensional components, the QK score in the r-th component is

![](images/609e4c057f79671d294fb38ad1266bbceb65926775be4d830d23d95e799a0620.jpg)  
Figure 3: Proximal background limits evidence utilization in Qwen3-8B. Points show paired changes in evidence attention and answer confidence. Blue moves evidence closer, while orange attenuates proximal background scores with positions fixed.

$$
\begin{array} { r } { \mathbf q _ { r } ^ { \top } \mathbf R ( - \Delta \omega _ { r } ) \mathbf k _ { r } = a _ { r } \cos ( \Delta \omega _ { r } ) + b _ { r } \sin ( \Delta \omega _ { r } ) , } \end{array}\tag{3}
$$

where the content-dependent coefficients are defined as $a _ { r } = q _ { r , 1 } k _ { r , 1 } + q _ { r , 2 } k _ { r , 2 }$ and $b _ { r } = q _ { r , 1 } k _ { r , 2 } - q _ { r , 2 } k _ { r , 1 }$ and $\omega _ { r }$ denotes the rotation frequency. Summing Eq. (3) across all components gives

$$
s ( \Delta ) = \frac { 1 } { \sqrt d } \sum _ { r = 1 } ^ { d / 2 } \left[ a _ { r } \cos ( \Delta \omega _ { r } ) + b _ { r } \sin ( \Delta \omega _ { r } ) \right] .\tag{4}
$$

Because RoPE rotations are orthogonal, they preserve the query and key norms, namely $\lVert \widetilde { \mathbf { q } } _ { t } \rVert _ { 2 } = \lVert \mathbf { q } _ { t } \rVert _ { 2 }$ and $\lVert \widetilde { \mathbf { k } } _ { i } \rVert _ { 2 } = \lVert \mathbf { k } _ { i } \rVert _ { 2 }$ . Thus, distance changes the directional match between the query and key across frequencies rather than reducing their vector lengths.

To isolate the positional effect in its simplest form, we consider a maximally aligned query–key pair with $\mathbf { q } _ { r } = \mathbf { k } _ { r }$ . This serves only as an illustrative reference case and is not assumed to hold in pretrained LLMs. In this case, $b _ { r } = 0$ and $a _ { r } = \lVert \mathbf { q } r \rVert _ { 2 } ^ { 2 }$ , reducing Eq. (4) to

$$
s ( \Delta ) = \frac { 1 } { \sqrt { d } } \sum _ { r = 1 } ^ { d / 2 } \lVert \mathbf { q } _ { r } \rVert _ { 2 } ^ { 2 } \cos ( \Delta \omega _ { r } ) .\tag{5}
$$

For $\left| \Delta \omega _ { r } \right| \ll 1$ , a second-order expansion gives:

$$
s ( \Delta ) \approx s ( 0 ) - \frac { \Delta ^ { 2 } } { 2 \sqrt d } \sum _ { r = 1 } ^ { d / 2 } \lVert \mathbf { q } _ { r } \rVert _ { 2 } ^ { 2 } \omega _ { r } ^ { 2 } .\tag{6}
$$

Crucially, score attenuation alone does not determine the evidence attention. Consider one distant evidence token with QK score $\boldsymbol { s } _ { \mathcal { E } }$ and M proximal background tokens, each with QK score $s _ { B }$ . Their softmax competition gives

$$
\alpha _ { \mathcal { E } } = \frac { 1 } { 1 + M e ^ { s _ { \mathcal { B } } - s _ { \mathcal { E } } } } , \qquad \alpha _ { \mathcal { E } } < \frac { 1 } { 2 } \Longleftrightarrow s _ { \mathcal { E } } - s _ { \mathcal { B } } < \log M .\tag{7}
$$

Accordingly, Eq. (7) explains both experiments in Section 2.2. Fig. 2 shows that although the model attends to distant evidence, abundant proximal background can collectively overshadow it through M. Fig. 3 further shows that attenuating proximal background restores evidence attention and answer confidence more consistently than moving the evidence closer. Thus, the primary obstacle is cumulative background competition rather than distance alone.

Together, these results distinguish vulnerability from failure: distance can weaken distant evidence, but insufficient evidence attention arises when abundant proximal background accumulates in the softmax denominator and collectively overshadows the evidence. This mechanism motivates an attention formulation that limits cumulative background competition while preserving access to distant evidence, leading to the LYRA formulation introduced next.

## 3 LYRA for Distant Evidence

The findings in Section 2 reveal that distant evidence can be overshadowed by proximal background context, even when the latter contributes little task-relevant information. In this section, we introduce LYRA (Longcontext heavY-tailed Relevance Alignment), a t-distributed directional matching method designed to preserve distant yet relevant evidence in Section 3.1, and analyze how it mitigates background-induced suppression in Section 3.2.

## 3.1 LYRA: t-distributed QK Matching

As established in Section 2, Eq.(7) exhibits that even a moderate reduction in $\boldsymbol { s } \boldsymbol { \varepsilon }$ can be exponentially amplified by softmax normalization, while the cumulative suppression from proximal background grows with M. A suitable mechanism should therefore preserve distant evidence under moderate score attenuation, distinguish evidence from background according to directional alignment with the query, and avoid indiscriminately increasing the weights of all distant tokens.

Motivated by these requirements, we introduce LYRA, a heavy-tailed directional matching mechanism that preserves attenuated yet relevant evidence without indiscriminately promoting distant context. Given the RoPE-transformed query $\widetilde { \mathbf { q } } _ { t }$ and key $\widetilde { \mathbf { k } } _ { i } .$ , we first measure their directional agreement using the normalized similarity $\begin{array} { r } { c _ { t , i } = \frac { \widetilde { \mathbf { q } } _ { t } ^ { \intercal } \widetilde { \mathbf { k } } _ { i } } { \Vert \widetilde { \mathbf { q } } _ { t } \Vert _ { 2 } \Vert \widetilde { \mathbf { k } } _ { i } \Vert _ { 2 } } \in [ - 1 , 1 ] } \end{array}$ . We then reshape this similarity using the t-distributed transformation:

$$
\phi _ { \kappa } ( c _ { t , i } ) = \frac { 1 + c _ { t , i } } { 1 + \kappa ( 1 - c _ { t , i } ) } - 1 , \qquad \kappa \geq 0 ,\tag{8}
$$

where κ controls the angular concentration parameter and $\phi _ { 0 } ( c _ { t , i } ) = c _ { t , i }$ recovers the original cosine similarity. The resulting attention weight is defined as $\alpha ^ { L Y R A } = \mathfrak { s }$ softmax<sub>i</sub> $\left( \beta \phi _ { \kappa } ( c _ { t , i } ) \right)$ , where $\beta > 0$ is an inverse-temperature parameter. This formulation increases the contrast between highly aligned evidence and weakly aligned background without introducing an explicit position-dependent increase for distant tokens.

Operationally, LYRA modifies only the query-key scoring stage of attention. Instead of directly using the conventional QK score, it computes the directional similarity between the RoPE-transformed query and key and maps it to the score $\beta \phi _ { \kappa } ( c _ { t , i } )$ through Eq. (8). RoPE, causal masking, softmax normalization, and value aggregation remain unchanged. LYRA can therefore be integrated into an existing attention layer by replacing its QK scoring function without modifying the remaining architecture. The detailed implementation is provided in Appendix C.

## 3.2 Preserving Distant Evidence under Background Competition

To characterize how LYRA locally reshapes directional similarity, we consider the derivative of the transformation in Eq. (8) and the critical similarity at which this derivative equals one:

$$
\phi _ { \kappa } ^ { \prime } ( c ) = \frac { 1 + 2 \kappa } { [ 1 + \kappa ( 1 - c ) ] ^ { 2 } } > 0 , \qquad c _ { \kappa } ^ { * } = 1 - \frac { 2 } { \sqrt { 1 + 2 \kappa } + 1 } .\tag{9}
$$

The positive derivative shows that $\phi _ { \kappa }$ is strictly increasing and therefore preserves the original similarity ordering. For $\kappa > 0 , c _ { \kappa } ^ { * }$ divides the similarity domain into a compression regime with $c < c _ { \kappa } ^ { * }$ and $\phi _ { \kappa } ^ { \prime } ( c ) < 1$ and an amplification regime with $c > c _ { \kappa } ^ { * }$ and $\phi _ { \kappa } ^ { \prime } ( c ) > 1$ . The transformation therefore reduces similarity differences in the former regime and enlarges them in the latter.

To extend the local characterization above to the finite difference between distant evidence and proximal background, we define the similarity-gap scaling factor for $c { \boldsymbol { B } } \neq c { \boldsymbol { \varepsilon } }$ as

$$
G _ { \kappa } ( c _ { B } , c _ { \mathcal { E } } ) = \frac { \phi _ { \kappa } ( c _ { B } ) - \phi _ { \kappa } ( c _ { \mathcal { E } } ) } { c _ { B } - c _ { \mathcal { E } } } = \frac { 1 + 2 \kappa } { [ 1 + \kappa ( 1 - c _ { B } ) ] [ 1 + \kappa ( 1 - c _ { \mathcal { E } } ) ] } .\tag{10}
$$

Accordingly, the transformed gap satisfies $\phi _ { \kappa } ( c _ { \mathcal { B } } ) - \phi _ { \kappa } ( c _ { \mathcal { E } } ) = G _ { \kappa } ( c _ { \mathcal { B } } , c _ { \mathcal { E } } ) ( c _ { \mathcal { B } } - c _ { \mathcal { E } } )$ . As the secant slope of $\phi _ { \kappa }$ between the two similarities, $G _ { \kappa } ~ < 1$ indicates gap compression, whereas $G _ { \kappa } > 1$ indicates gap amplification. Compared with cosine-softmax under the same inverse temperature $\beta _ { - }$ , the relative change in the evidence-to-background attention ratio is

$$
{ \frac { \alpha _ { \mathcal { E } } ^ { L Y R A } / \alpha _ { \mathcal { B } } ^ { L Y R A } } { \alpha _ { \mathcal { E } } ^ { \mathrm { c o s } } / \alpha _ { \mathcal { B } } ^ { \mathrm { c o s } } } } = \exp \left[ \beta ( 1 - G _ { \kappa } ) ( c _ { \mathcal { B } } - c _ { \mathcal { E } } ) \right] .\tag{11}
$$

The relative weight of evidence is therefore improved when $( 1 - G _ { \kappa } ) ( c _ { \mathcal { B } } - c _ { \mathcal { E } } ) > 0$ . When proximal background has a higher similarity than distant evidence, $G _ { \kappa } < 1$ compresses the background advantage. When evidence retains a higher similarity, $G _ { \kappa } ~ > ~ 1$ enlarges the evidence advantage against cumulative background competition. Consequently, LYRA reduces the influence of weakly related background tokens while making strongly matched evidence more distinguishable, thereby directing attention toward taskrelevant information.

Beyond competition from a single proximal background token, we extend the analysis to M proximal background tokens, which does not change above improvement condition. The factor M increases the cumulative competing mass, while $G _ { \kappa }$ determines whether the score gap is adjusted in favor of distant evidence. Therefore, LYRA mitigates the Proximity Trap by narrowing a misleading background advantage or strengthening an existing evidence advantage before competition accumulates across background tokens.

## 3.3 ProxBench: Fine-Grained Proximal Perturbations for Long-Context

Existing long-context benchmarks primarily evaluate whether models can retrieve and integrate information as the input length increases. However, they provide limited control over the proximal background context that competes with distant evidence. As a result, it remains difficult to determine whether a model succeeds by robustly identifying task-relevant evidence or fails because semantically confusable information near the query captures excessive attention. To address this limitation, we introduce PROXBENCH, a controlled benchmark designed to evaluate the robustness of long-context models against proximal background interference. PROXBENCH places sparse evidence at a distant position and introduces task-irrelevant but increasingly confusable background context near the query, thereby directly measuring the Proximity Trap under controlled conditions.

PROXBENCH organizes proximal perturbations into four progressively challenging levels. Level 1 introduces style-matched background that follows a similar syntactic structure but differs in entity, topic, and answer relation. Level 2 creates crossed bindings by placing the target entity and an access-code cue in the same sentence while explicitly associating the candidate value with another entity. Level 3 further increases ambiguity by mixing the same relation for different entities with different attributes of the target entity. Level 4 introduces fine-grained perturbations involving subtypes, attributes, semantic roles, and value formats, producing background context that closely resembles the required evidence while remaining logically irrelevant to the answer. This hierarchy enables a fine-grained assessment of how model performance changes as proximal background becomes increasingly difficult to distinguish from distant evidence. Further details on data construction, quality control, and dataset statistics are provided in Appendix B.

## 4 Experiments

We organize our evaluation around three complementary aspects. First, in Section 4.1, we evaluate LYRA on LongBench-v2 (Bai et al., 2025) and RULER (Hsieh et al., 2024), examining its ability to retrieve and utilize distant evidence under increasingly long inputs. Second, in Section 4.2, we report results across the task categories of LongBench to assess whether the benefits of LYRA generalize beyond specific longcontext settings. Third, in Section 4.3, we evaluate LYRA on our proposed ProxBench, where progressively challenging perturbation levels are used to directly measure its robustness to different forms of proximal background interference. Additional ablation studies are provided in the Appendix D to examine the contribution and sensitivity of the key design in LYRA.

Experimental Settings. We adopt Qwen3-8B (Yang et al., 2025) as the base model and replace the standard attention mechanism only in its final Transformer block with LYRA, leaving the remaining architecture unchanged. We then fine-tune the model on LongAlign for one epoch, updating only the final

Transformer block while freezing all preceding layers, token embeddings, and the language-modeling head. Training is performed in bfloat16 using AdamW, with a learning rate of $2 \times 1 0 ^ { - 5 }$ , and a maximum sequence length of 16,384. We evaluate long-context retrieval and understanding on RULER and LongBench-v2, and assess robustness to proximity trap on ProxBench. The detailed derivation process is provided in Appendix C.

## 4.1 Main Results

First, we conduct experiments on LongBench-v2 (Bai et al., 2025), examining whether our LYRA improves long-context understanding across different input lengths. As exhibited in Table 1, LYRA achieves the best overall performance and consistently ranks among the strongest methods across all context-length splits. The leading performance on the Short split indicates that the improvement does not come at the cost of local context modeling, while the clear advantage on the Long split demonstrates a stronger ability to preserve and retrieve relevant evidence over extended contexts. This consistent behavior can be attributed to the combination of directional QK matching and the t-distributed transformation. Directional matching distinguishes semantically relevant evidence from weakly aligned background context, while the t-distributed transformation prevents relevant evidence from being excessively suppressed when its original QK score is weakened by dis-

Table 1: Evaluation results of different homogeneous methods on LongBench-v2 (Bai et al., 2025), following the evaluation setting of ProxyAttn (Wang et al., 2026b). We compare our method against the baseline, MInference (Jiang et al., 2024), FlexPrefill (Lai et al., 2025), XAttention (Xu et al., 2025), ProxyAttn (Wang et al., 2026b), S2O (Zhang et al., 2026) and PBS-Attn (Wang et al., 2026a). We report accuracy on the full evaluation set and across three context-length splits defined by word count: Short (<32K), Medium (32K–128K), and Long (>128K). The overall score is computed over the complete evaluation set. The best and second-best results in each column are highlighted in red and blue, respectively.
<table><tr><td>Methods</td><td>Venue</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>Baseline</td><td></td><td>36.67</td><td>29.30</td><td>30.56</td><td>32.21</td></tr><tr><td>Minference</td><td>NeurIPS&#x27;24</td><td>40.60</td><td>26.50</td><td>28.70</td><td>32.02</td></tr><tr><td>FlexPrefill</td><td>ICLR’25</td><td>41.70</td><td>30.20</td><td>27.80</td><td>33.80</td></tr><tr><td>Xattention</td><td>ICML&#x27;25</td><td>41.70</td><td>27.40</td><td>32.40</td><td>33.59</td></tr><tr><td>ProxyAttn</td><td>ICLR&#x27;26</td><td>43.90</td><td>28.40</td><td>27.80</td><td>33.82</td></tr><tr><td>S20</td><td>ACL&#x27;26</td><td></td><td></td><td></td><td>32.60</td></tr><tr><td>PBS-Attn</td><td>ICML&#x27;26</td><td>–</td><td>一</td><td></td><td>34.39</td></tr><tr><td>Ours</td><td></td><td>47.22</td><td>29.63</td><td>33.33</td><td>36.72</td></tr></table>

tance. As a result, LYRA reduces the competition from proximal background tokens without indiscriminately increasing the scores of all distant tokens. The results therefore demonstrate that LYRA provides a robust balance between local context modeling and distant evidence utilization, supporting its effectiveness in mitigating the Proximity Trap.

Moreover, we evaluate LYRA on RULER (Hsieh et al., 2024) to provide a more fine-grained analysis of robustness to context length. Unlike LongBench-v2, which groups examples into broad context-length splits, RULER evaluates the same set of controlled tasks at explicitly specified input lengths, allowing the performance variation of each method to be examined as the context expands.

As illustrated in Table 2, LYRA achieves the best performance at every evaluated context length and establishes the strongest average result. The consistent advantage from 8K to 128K indicates that the effectiveness of LYRA is not restricted to a particular context range. In particular, LYRA delivers substantial improvements at intermediate and long context lengths while retaining the leading performance at 128K, where relevant evidence faces increasingly strong competition from background tokens. These results suggest that directional QK matching and the t-distributed transformation remain effective as the amount of competing context increases. Together with LongBench-v2, the consistent performance on RULER demonstrates that LYRA improves both practical long-context understanding and fine-grained length robustness.

Table 2: Evaluation results of different homogeneous methods on RULER (Hsieh et al., 2024), following the evaluation setting of S2O (Zhang et al., 2026). We compare our method against FlexPrefill (Lai et al., 2025), XAttention (Xu et al., 2025), ProxyAttn (Wang et al., 2026b), S2O (Zhang et al., 2026), and PBS-Attn (Wang et al., 2026a). We report accuracy at five explicitly controlled context lengths, namely 8K, 16K, 32K, 64K, and 128K, providing a more fine-grained evaluation of length robustness. The average score is computed across these five context lengths. The best and second-best results in each column are highlighted in red and blue, respectively.
<table><tr><td></td><td>Venue</td><td>8K</td><td>16K</td><td>32K</td><td>64K</td><td>128K</td><td>Avg.</td></tr><tr><td>FlexPrefill</td><td>ICLR’25</td><td>71.65</td><td>73.89</td><td>75.38</td><td>72.65</td><td>68.51</td><td>72.42</td></tr><tr><td>Xattention</td><td>ICML&#x27;25</td><td>85.63</td><td>82.25</td><td>81.60</td><td>73.18</td><td>69.91</td><td>78.51</td></tr><tr><td>ProxyAttn</td><td>ICLR&#x27;26</td><td>92.07</td><td>89.75</td><td>86.68</td><td>83.57</td><td>77.09</td><td>85.83</td></tr><tr><td>S20</td><td>ACL&#x27;26</td><td>85.80</td><td>82.73</td><td>80.34</td><td>73.95</td><td>69.97</td><td>78.56</td></tr><tr><td>PBS-Attn</td><td>ICML&#x27;26</td><td>85.56</td><td>79.34</td><td>80.95</td><td>70.70</td><td>67.90</td><td>76.89</td></tr><tr><td>Ours</td><td></td><td>96.33</td><td>94.19</td><td>93.14</td><td>85.39</td><td>77.56</td><td>89.32</td></tr></table>

## 4.2 Generalization Across Diverse Tasks

Subsequently, we evaluate LYRA on LongBench (Bai et al., 2024b) to examine whether the improvements generalize across tasks with different evidence structures and reasoning requirements. In contrast to the preceding experiments, which focus on robustness across context lengths, this evaluation covers six task categories and therefore provides a broader assessment of task-level generalization.

Table 3: Evaluation results of different homogeneous methods on LongBench (Bai et al., 2024b). We compare our method against MInference (Jiang et al., 2024), FlexPrefill (Lai et al., 2025), XAttention (Xu et al., 2025), PBS-Attn (Wang et al., 2026a), Stem (Niu et al., 2026), and Kascade (Deshmukh et al., 2026). We report the official LongBench score across six task categories: single-document question answering (SQA), multi-document question answering (MQA), summarization, few-shot learning, code completion, and synthetic tasks. The best and second-best results in each column are highlighted in red and blue, respectively.
<table><tr><td>Method</td><td>Venue</td><td>SQA</td><td>MQA</td><td>Summ.</td><td>Few-shot</td><td>Code</td><td>Synthetic</td><td>Avg.</td></tr><tr><td>MInference</td><td>NeurIPS&#x27;24</td><td>46.92</td><td>37.11</td><td>16.53</td><td>34.15</td><td>2.98</td><td>66.17</td><td>33.98</td></tr><tr><td>FlexPrefill</td><td>ICLR’25</td><td>46.08</td><td>37.29</td><td>16.53</td><td>33.63</td><td>2.52</td><td>49.00</td><td>30.84</td></tr><tr><td>XAttention</td><td>ICML&#x27;25</td><td>45.66</td><td>37.77</td><td>16.66</td><td>36.20</td><td>2.02</td><td>64.67</td><td>33.83</td></tr><tr><td>PBS-Attn</td><td>ICML&#x27;26</td><td>47.04</td><td>37.17</td><td>16.66</td><td>33.88</td><td>2.80</td><td>66.33</td><td>33.98</td></tr><tr><td>STEM</td><td>ICML&#x27;26</td><td></td><td>14.97</td><td>20.21</td><td>61.84</td><td>19.43</td><td>62.19</td><td>35.73</td></tr><tr><td>Kascade</td><td>Arxiv&#x27;26</td><td>44.87</td><td>42.34</td><td>23.74</td><td>61.99</td><td>62.71</td><td>34.50</td><td>45.03</td></tr><tr><td>Ours</td><td></td><td>46.01</td><td>43.35</td><td>24.73</td><td>62.17</td><td>58.41</td><td>65.67</td><td>50.06</td></tr></table>

As presented in Table 3, LYRA achieves the best average performance, with leading results on multidocument question answering, summarization, and few-shot learning, as well as competitive performance on the remaining tasks. The improvements are particularly clear for tasks that require relevant information to be identified and integrated from multiple locations. In multi-document question answering and summarization, evidence is often distributed across long contexts and competes with substantial background information, making these tasks especially susceptible to the Proximity Trap. Similarly, few-shot learning requires the model to identify useful demonstrations despite interference from other examples, while code completion often depends on distant definitions and dependencies surrounded by locally similar code. The strong performance across these tasks suggests that directional QK matching effectively distinguishes relevant evidence from weakly aligned background context, while the t-distributed transformation prevents useful information from being excessively suppressed by positional distance. Meanwhile, the competitive results on single-document question answering and synthetic tasks indicate that this mechanism does not compromise tasks dominated by more localized or explicit evidence. Overall, these results demonstrate that LYRA mitigates evidence competition across diverse task structures rather than benefiting only a specific context length or retrieval pattern.

## 4.3 Alleviating the Proximity Trap on ProxBench

Beyond standard long-context and generalization benchmarks, we evaluate LYRA on our proposed ProxBench to directly examine robustness to the Proximity Trap. The benchmark fixes the query and distant evidence while progressively increasing the semantic similarity and relational ambiguity of proximal background context. We compare LYRA with representative LLMs under the same evaluation protocol across four difficulty levels.

As shown in Figure 4, LYRA achieves the best average performance and exhibits substantially greater robustness as the perturbation difficulty increases. Although Llama performs slightly better at the easiest level, LYRA consistently achieves the strongest results

![](images/fda7ca4a91dfca170542bb3e29b59256bc9277b0d922fe8d2aa58af6e0b5b651.jpg)  
Figure 4: Evaluation results on ProxBench across four progressively challenging proximal perturbation levels. We compare Qwen3-8B (Yang et al., 2025), Llama3.1-8B (Grattafiori et al., 2024), GLM-4-9B (Glm et al., 2024), and our LYRA model (8B) on Levels 1–4 and report accuracy (%) at each level. The average score is computed over all four difficulty levels.

at all subsequent levels, with the advantage becoming particularly clear under the most fine-grained perturbations. In contrast, the performance of the baseline models declines sharply when the proximal background shares entities, relations, attributes, or value formats with the target evidence. The smaller performance degradation of LYRA indicates that the model does not rely primarily on positional proximity or surface-level similarity when identifying relevant information. Instead, directional QK matching captures fine-grained relevance between the query and evidence, while the t-distributed transformation prevents distant evidence from being overwhelmed by accumulated competition from proximal background tokens. These results provide direct evidence that LYRA mitigates the Proximity Trap by preserving task-relevant evidence even when nearby context is highly similar but logically irrelevant.

## 5 Conclusion

In this work, we identify the Proximity Trap, showing that distant evidence can be underutilized not only because of positional distance, but also due to cumulative competition from task-irrelevant proximal background. To address this issue, we introduce LYRA, a t-distributed directional matching mechanism that reshapes the context retrieval distribution to better preserve task-relevant evidence under background competition. Extensive experiments on LongBench-v2, RULER, LongBench, and ProxBench demonstrate consistent improvements across context lengths, task categories, and increasingly challenging proximal perturbations. Overall, our findings suggest that effective long-context modeling should consider not only how far relevant evidence lies from the query, but also what it must compete with along the way.

## References

Yushi Bai, Xin Lv, Jiajie Zhang, Yuze He, Ji Qi, Lei Hou, Jie Tang, Yuxiao Dong, and Juanzi Li. Longalign: A recipe for long context alignment of large language models. In Findings of the Association for Computational Linguistics: EMNLP 2024, pages 1376–1395, 2024a.

Yushi Bai, Xin Lv, Jiajie Zhang, Hongchang Lyu, Jiankai Tang, Zhidian Huang, Zhengxiao Du, Xiao Liu, Aohan Zeng, Lei Hou, et al. Longbench: A bilingual, multitask benchmark for long context understanding. In Proceedings of the 62nd annual meeting of the association for computational linguistics (volume 1: Long papers), pages 3119–3137, 2024b.

Yushi Bai, Shangqing Tu, Jiajie Zhang, Hao Peng, Xiaozhi Wang, Xin Lv, Shulin Cao, Jiazheng Xu, Lei Hou, Yuxiao Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, 2025.

Dhruv Deshmukh, Saurabh Goyal, Nipun Kwatra, and Ramachandran Ramjee. Kascade: A practical sparse attention method for long-context llm inference, 2026. URL https://arxiv.org/abs/2512. 16391.

Yiran Ding, Li Lyna Zhang, Chengruidong Zhang, Yuanyuan Xu, Ning Shang, Jiahang Xu, Fan Yang, and Mao Yang. LongRoPE: Extending LLM context window beyond 2 million tokens. In Ruslan Salakhutdinov, Zico Kolter, Katherine Heller, Adrian Weller, Nuria Oliver, Jonathan Scarlett, and Felix Berkenkamp, editors, Proceedings of the 41st International Conference on Machine Learning, volume 235 of Proceedings ofMachine Learning Research, pages 11091–11104. PMLR, 21–27 Jul 2024. URL https://proceedings.mlr.press/v235/ding24i.html.

Team Glm, Aohan Zeng, Bin Xu, Bowen Wang, Chenhui Zhang, Da Yin, Dan Zhang, Diego Rojas, Guanyu Feng, Hanlin Zhao, et al. Chatglm: A family of large language models from glm-130b to glm-4 all tools. arXiv preprint arXiv:2406.12793, 2024.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Junqing He, Kunhao Pan, Xiaoqun Dong, Zhuoyang Song, Yibo Liu, Qianguo Sun, Yuxin Liang, Hao Wang, Enming Zhang, and Jiaxing Zhang. Never lost in the middle: Mastering long-context question answering with position-agnostic decompositional training. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 13628–13642, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.736. URL https://aclanthology.org/2024.acl-long.736/.

Cheng-Ping Hsieh, Simeng Sun, Samuel Kriman, Shantanu Acharya, Dima Rekesh, Fei Jia, Yang Zhang, and Boris Ginsburg. Ruler: What’s the real context size of your long-context language models? arXiv preprint arXiv:2404.06654, 2024.

Huiqiang Jiang, Yucheng Li, Chengruidong Zhang, Qianhui Wu, Xufang Luo, Surin Ahn, Zhenhua Han, Amir H Abdi, Dongsheng Li, Chin-Yew Lin, et al. Minference 1.0: Accelerating pre-filling for long-context llms via dynamic sparse attention. Advances in Neural Information Processing Systems, 37:52481–52515, 2024.

Xunhao Lai, Jianqiao Lu, Yao Luo, Yiyuan Ma, and Xun Zhou. Flexprefill: A context-aware sparse attention mechanism for efficient long-sequence inference. In Y. Yue, A. Garg, N. Peng, F. Sha, and R. Yu, editors, International Conference on Learning Representations, volume 2025, pages 963– 989, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/file/ 03645743ea35690f30d795d6bac149a5-Paper-Conference.pdf.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions of the association for computational linguistics, 12:157–173, 2024.

Timur Mudarisov, Mikhail Burtsev, Tatiana Petrova, and Radu State. Limitations of normalization in attention. In The Thirty-ninth Annual Conference on Neural Information Processing Systems, 2025. URL https://openreview.net/forum?id=16kX08MCav.

Lin Niu, Xin Luo, Linchuan Xie, Yifu Sun, Guanghua Yu, Jianchen Zhu, and S Kevin Zhou. Stem: Rethinking causal information flow in sparse attention. In Forty-third International Conference on Machine Learning, 2026. URL https://openreview.net/forum?id=8WrMPCd7Kr.

Bowen Peng, Jeffrey Quesnelle, Honglu Fan, and Enrico Shippole. Yarn: Efficient context window extension of large language models. In International Conference on Learning Representations, volume 2024, pages 31932–31951, 2024.

Konrad Staniszewski, Szymon Tworkowski, Sebastian Jaszczur, Yu Zhao, Henryk Michalewski, Łukasz Kucinski, and Piotr Miło´ s. Structured packing in llm training improves long context utilization. In´ Proceedings ofthe AAAI Conference on Artificial Intelligence, volume 39, pages 25201–25209, 2025.

Jianlin Su, Murtadha Ahmed, Yu Lu, Shengfeng Pan, Wen Bo, and Yunfeng Liu. Roformer: Enhanced transformer with rotary position embedding. Neurocomputing, 568:127063, 2024.

Xinghao Wang, Pengyu Wang, Dong Zhang, Chenkun Tan, Shaojun Zhou, Zhaoxiang Liu, Shiguo Lian, Fangxu Liu, Kai Song, and Xipeng Qiu. Sparser block-sparse attention via token permutation. In Fortythird International Conference on Machine Learning, 2026a. URL https://openreview.net/ forum?id=5u3Ra6Qf5C.

Yixuan Wang, Huang He, Siqi Bao, hua wu, Haifeng Wang, Qingfu Zhu, and Wanxiang Che. Proxyattn: Guided sparse attention via representative heads. In C. Vondrick, B. Hariharan, C. Raffel, L. Pinto, D. Yang, and A. Faust, editors, International Conference on Learning Representations, volume 2026, pages 18603–18617, 2026b. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ file/1fa81061d6d4d7fea88f803d89ae9d6e-Paper-Conference.pdf.

Wenhao Wu, Yizhong Wang, Guangxuan Xiao, Hao Peng, and Yao Fu. Retrieval head mechanistically explains long-context factuality. In International Conference on Learning Representations, volume 2025, pages 62143–62156, 2025a.

Xinyi Wu, Yifei Wang, Stefanie Jegelka, and Ali Jadbabaie. On the emergence of position bias in transformers. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning,

volume 267 of Proceedings ofMachine Learning Research, pages 67756–67781. PMLR, 13–19 Jul 2025b. URL https://proceedings.mlr.press/v267/wu25ad.html.

Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. In International Conference on Learning Representations, volume 2024, pages 21875–21895, 2024.

Ruyi Xu, Guangxuan Xiao, Haofeng Huang, Junxian Guo, and Song Han. XAttention: Block sparse attention with antidiagonal scoring. In Aarti Singh, Maryam Fazel, Daniel Hsu, Simon Lacoste-Julien, Felix Berkenkamp, Tegan Maharaj, Kiri Wagstaff, and Jerry Zhu, editors, Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 69819–69831. PMLR, 13–19 Jul 2025. URL https://proceedings.mlr.press/v267/ xu25ag.html.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Yu Zhang, Songwei Liu, Chenqian Yan, Beichen Ning, Fangmin Chen, Xing Wang, et al. S2o: Early stopping for sparse attention via online permutation. In Proceedings ofthe 64th Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers), pages 7737–7751, 2026.

Hongbo Zhao, Huibin Wang, Bin Tang, Xianming Hu, Yihong Huang, Yijun Shen, Nuoyi Chen, Ping Li, and Kai Zhang. Adaptive zooming via relevance-informed positional resource allocation for training-free llm context extension. In Findings ofthe Associationfor Computational Linguistics: ACL 2026, pages 21067–21083, 2026.

Chuanyang Zheng, Yihang Gao, Guoxuan Chen, Han Shi, Jing Xiong, Xiaozhe Ren, Chao Huang, Zhenguo Li, and Yu Li. Self-adjust softmax. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pages 7827–7847, 2025.

## Appendix

## A Related Works

Our work is primarily related to two lines of research. The first develops long-context language models through context-window extension, long-context adaptation, and efficient attention computation. The second investigates positional biases and attention normalization, examining how the location of relevant information and competition among contextual tokens affect its influence on model predictions. We review these directions below and distinguish them from our focus on the competition between distant evidence and task-irrelevant proximal background.

## A.1 Long-Context Modeling

Recent advances in long-context modeling have primarily focused on extending the context window, adapting models to long sequences, and improving the efficiency of long-context inference. One line of work extends the usable context range through positional adaptation. YaRN rescales rotary positional embeddings to support length extrapolation, while LongRoPE employs non-uniform positional interpolation to accommodate substantially longer sequences (Peng et al., 2024; Ding et al., 2024). More recently, RiPRA incorporates semantic relevance into positional resource allocation, assigning finer positional resolution to potentially useful content while compressing less relevant regions (Zhao et al., 2026). These approaches substantially expand the range over which contextual information remains representable, but an extended context window does not necessarily ensure that distant evidence is effectively incorporated into model predictions.

A complementary direction improves long-context capabilities through adaptation and alignment. LongAlign constructs long-form instruction data and develops corresponding training strategies to align language models with long-context tasks (Bai et al., 2024a). Such training-based approaches expose models to longer and more diverse dependencies, improving their ability to operate over extended inputs. Nevertheless, their primary objective is to establish general long-context competence, rather than to characterize how distant evidence is utilized when it competes with abundant, task-irrelevant context near the query.

Another substantial body of research seeks to reduce the computational and memory costs of processing long sequences. StreamingLLM maintains stable streaming inference by retaining attention sinks and a local context window (Xiao et al., 2024). Dynamic sparse-attention methods selectively allocate computation to important regions: MInference exploits recurring sparse attention patterns, FlexPrefill adapts sparse patterns and computational budgets to individual inputs and attention heads, and XAttention estimates block importance through antidiagonal scoring (Jiang et al., 2024; Lai et al., 2025; Xu et al., 2025). ProxyAttn further leverages cross-head similarity to obtain more fine-grained block estimates with reduced overhead (Wang et al., 2026b). Collectively, these methods make long-context processing increasingly scalable by preserving selected attention regions while avoiding exhaustive computation.

Despite this progress, existing studies largely concern whether distant information remains positionally representable, computationally accessible, or retained under efficient inference. Comparatively less attention has been paid to whether a model can effectively use distant evidence after it becomes accessible. In long inputs, distant evidence must compete with abundant proximal background under normalized attention; consequently, successful context extension or sparse selection alone does not guarantee that the evidence receives sufficient influence over the final prediction. Our work addresses this complementary problem by studying how proximal background systematically suppresses the utilization of distant evidence and by improving the allocation of attention under such contextual competition.

## A.2 Positional Bias and Attention Normalization

A growing body of research has shown that the ability of language models to use long contexts is strongly affected by the position of relevant information. Liu et al. (2024) reveal that models often perform well when relevant information appears near the beginning or end of the input, but struggle to use the same information when it is placed in intermediate positions. Subsequent studies improve positional robustness through long-context training. Position-agnostic decompositional training encourages models to identify and integrate information across different context positions (He et al., 2024), while structured packing organizes semantically related documents into training sequences to improve the utilization of distributed contextual information (Staniszewski et al., 2025). These studies demonstrate that extending the context window alone does not guarantee positionally robust information use and that targeted training can partially alleviate context underutilization.

Complementary work investigates the mechanisms underlying such positional preferences. Retrievalhead analyses identify a small subset of attention heads that plays a disproportionate role in incorporating contextual evidence into factual predictions (Wu et al., 2025a). From a broader architectural perspective, theoretical analyses attribute position bias to the joint effects of causal masking, positional encoding, and information propagation across layers (Wu et al., 2025b). Collectively, these findings suggest that the influence of contextual evidence is shaped not only by its semantic relevance, but also by where it appears and how attention propagates its information. However, most existing analyses characterize the effect of evidence position itself, leaving the role of the surrounding competing context comparatively underexplored.

Another line of research examines the limitations of attention normalization. Standard softmax couples all contextual tokens through a shared normalization denominator, such that increasing the number of competing tokens can weaken the selectivity of attention. Recent work improves its optimization properties through input-dependent adjustment (Zheng et al., 2025), while theoretical analyses show that the ability of normalized attention to separate informative from non-informative tokens deteriorates as the selected set or context grows (Mudarisov et al., 2025). These studies establish the general limitations of softmax-based selection, but do not explicitly consider the asymmetric competition between distant evidence and abundant proximal background.

Our work connects positional bias with attention normalization through the Proximity Trap. Rather than treating the underutilization of distant evidence solely as a consequence of its position, we show that it is jointly driven by distance-dependent matching and the cumulative competition induced by nearby, task-irrelevant context. Accordingly, LYRA strengthens attention to task-relevant distant evidence while suppressing interference from weakly relevant proximal background, without indiscriminately favoring all distant tokens.

## B ProxBench Details

## B.1 Task Formulation

PROXBENCH evaluates whether a model can retrieve a target fact when the context contains records that partially overlap with the entity, relation, attribute, semantic role, or value format required by the query. Each instance samples a target entity e from a predefined inventory of ten synthetic facility names and generates an access code v in the format AA-000. The relevant evidence and query follow the templates

$$
s ^ { + } = { } ^ { \ast \cdot } \mathrm { T h e ~ a c c e s s ~ c o d e ~ a s s i g n e d ~ t o ~ } e \mathrm { ~ i s ~ } v ! ^ { , } ,\tag{12}
$$

$$
q = { \mathrm { } } ^ { \mathrm { * } } \mathbf { W h a t i s t h e a c c e s s c o d e a s s i g n e d t o } e ? ^ { \mathfrak { P } } .\tag{13}
$$

The expected output is the code v without additional explanation.

Each instance contains four proximal background records. These records are constructed to share selected surface or semantic cues with $s ^ { + }$ and q, while remaining logically insufficient for answering the query. In particular, the correct value never appears in the background, and no background record establishes the complete binding between the target entity, the queried access-code attribute, and the candidate value.

## B.2 Perturbation Levels

The four levels control which components of the target fact are preserved in the background. Let the target fact be represented as the tuple

$$
( e , \ a _ { \mathrm { a c c e s s } } , \ r _ { \mathrm { a s s i g n e d } } , \ v ) ,
$$

where e is the target entity, $a _ { \mathrm { a c c e s s } }$ is the access-code attribute, $r _ { \mathrm { a s s i g n e d } }$ denotes the assignment relation and recipient role, and v is the correct value.

Level 1: style-matched background. L1 controls for superficial similarity. Its records use the same declarative style and assignment-oriented syntax as the evidence, but change the entity, topic, and answer relation. Example attributes include maintenance schedule, inspection timetable, freight allocation, and renovation plan. Because these records contain neither the target entity nor a code-shaped candidate, they can be rejected using relatively coarse lexical and semantic cues.

Level 2: crossed bindings. L2 creates a local binding conflict by placing the target entity, an access-code phrase, another entity, and a candidate code in the same sentence. A typical template is:

The [target entity] cross-reference lists [candidate code] as the access code assigned to [other entity], not to [target entity].

Although most query-relevant tokens appear together, the semantic recipient of the candidate value is the other entity. Correct prediction therefore requires resolving the argument structure rather than copying the closest code-shaped span.

Level 3: mixed entity–relation interference. L3 alternates between two complementary perturbation types. The first preserves the full access-code relation but replaces the entity:

The access code assigned to [other entity] is [candidate code].

The second preserves the target entity and assignment relation but changes the queried attribute and value format:

The emergency contact number assigned to [target entity] is [phone number].

With four background records, each subtype appears twice. Consequently, the background contains both relation-matching candidates and entity-matching candidates, requiring the model to jointly resolve entity and attribute bindings.

Level 4: fine-grained binding characterization. L4 provides a finer diagnostic representation of the mixed binding condition. Each background record is assigned a semantic subtype describing the precise source of overlap, such as preserving the relation while replacing the entity or preserving the entity while replacing the attribute. The annotations additionally record entity, relation, attribute, and semantic-role overlap, as well as whether a candidate shares the expected code or phone-number format. This representation supports subtype-level error analysis and separates failures caused by entity confusion, attribute confusion, role reversal, and value-format attraction.

## B.3 Answer-Neutrality and Quality Control

We apply the following automatic constraints during generation:

• No answer leakage. The gold access code is prohibited from appearing in any background record.

• Distinct candidate values. Distractor codes are different from the gold answer and from other distractor codes in the same instance.

• Entity separation. Whenever a template requires another entity, it is sampled from the entity inventory after excluding the target entity.

• Binding validity. No background record asserts that a distractor value is the access code assigned to the target entity.

• Exact background count. Every instance contains four perturbation records, and all four spans must be recoverable after tokenization.

• Length control. Contexts are constructed under the target tokenizer so that the tokenized input plus the reserved generation budget exactly matches the requested sequence length.

• Span verification. Evidence, query, and background spans are located after serialization and retokenization. Instances that fail round-trip alignment are regenerated using a deterministic retry seed.

• Reproducibility. Instance-level seeds are derived from the global seed, context length, perturbation level, and example index using SHA-256. Dataset shards are additionally recorded with byte counts and SHA-256 checksums.

Each example also includes diagnostic metadata such as entity\_overlap, relation\_overlap, attribute\_overlap, role\_overlap, candidate\_value\_overlap, joint\_cue\_overlap, distractor\_value\_format, and component\_strategies. These fields describe the intended interference mechanism and allow results to be decomposed by binding subtype.

## C Implementation and Computational Overhead

Implementation. For Qwen3-8B, we apply LYRA only to the final Transformer block (block 35 under zero-based indexing). We optimize the complete final block while freezing the preceding 35 blocks, token embeddings, final normalization layer, and language-modeling head. This results in approximately 193M trainable parameters, corresponding to 2.36% of the 8.19B-parameter model. The head-specific LYRA parameters themselves introduce only 2H = 64 additional scalars for H = 32 query heads. Training is performed for one epoch in bfloat16 using AdamW, with a maximum sequence length of 16,384. We use a per-device batch size of one on two devices, four gradient-accumulation steps, a learning rate of $2 \times 1 0 ^ { - 5 }$ , a 3% warmup ratio, zero weight decay, and random seed 42.

Computational Overhead. Consider dense attention with batch size B, sequence length n, H query heads, and head dimension $d _ { h }$ . Standard attention requires $\mathcal { O } ( B H n ^ { 2 } d _ { h } )$ arithmetic for query–key matching and value aggregation. LYRA replaces the query–key product with a product between normalized queries and keys and additionally performs $O ( B H n d _ { h } )$ normalization and $\mathcal { O } ( B H n ^ { 2 } )$ element-wise transformations. Its overall prefill complexity therefore remains $\mathcal { O } ( B H n ^ { 2 } d _ { h } )$ , identical to dense attention asymptotically. Autoregressive decoding likewise retains $O ( B H n d _ { h } )$ time per generated token, and LYRA does not alter the $\mathcal { O } ( B H _ { \mathrm { k v } } n d _ { h } )$ KV-cache size.

Table 4: Analytical computation of Qwen3-8B and the additional FLOPs introduced by LYRA. The relative overhead remains below 0.04% across all evaluated context lengths.
<table><tr><td>Context</td><td>Qwen3-8B (TFLOPs)</td><td>Additional LYRA (GFLOPs)</td><td>Relative Overhead</td></tr><tr><td>8K</td><td>153.382</td><td>17.382</td><td>0.0113%</td></tr><tr><td>16K</td><td>385.929</td><td>69.122</td><td>0.0179%</td></tr><tr><td>32K</td><td>1,088.517</td><td>275.683</td><td>0.0253%</td></tr><tr><td>64K</td><td>3,443.670</td><td>1101</td><td>0.0320%</td></tr><tr><td>128K</td><td>11,953.890</td><td>4401</td><td>0.0368%</td></tr></table>

Table 4 reports the analytical computation of Qwen3-8B and the additional FLOPs introduced by LYRA. Across context lengths from 8K to 128K, the relative overhead increases only from 0.0113% to 0.0368% and remains below 0.04% in all settings. This small overhead arises because LYRA modifies only the final Transformer block and adds primarily element-wise normalization and score-transformation operations, whose cost is minor relative to the matrix multiplications in dense attention. Consequently, LYRA preserves the asymptotic computational complexity of the base model while introducing negligible additional arithmetic.

## D Ablation Studies

Table 5: Ablation studies on different κ on LongBench-v2 (Bai et al., 2025). We vary κ, which controls the transformation strength of the t-distributed mapping, while keeping all other settings fixed. Accuracy is reported.
<table><tr><td>κ</td><td>Short</td><td>Medium</td><td>Long</td><td>Overall</td></tr><tr><td>2</td><td>41.11</td><td>27.91</td><td>31.48</td><td>33.40</td></tr><tr><td>4</td><td>47.22</td><td>29.63</td><td>33.33</td><td>36.72</td></tr><tr><td>8</td><td>44.44</td><td>28.84</td><td>30.63</td><td>34.81</td></tr><tr><td>16</td><td>38.33</td><td>24.19</td><td>33.33</td><td>31.21</td></tr></table>

We conduct an ablation study to examine the effect of κ, which controls the strength of the t-distributed transformation. We vary κ, and evaluate each configuration across the context-length splits of LongBench-v2. As shown in Table 5, the results exhibit a clear non-monotonic trend, with a moderate value of κ achieving the strongest and most consistent performance across all context lengths. A smaller value provides insufficient separation between relevant evidence and weakly aligned background context, whereas an excessively large value applies overly aggressive score transformation and may suppress moderately aligned keys that still contain useful information. The consistent preference for the same intermediate setting across all splits indicates that κ does not require length-specific tuning. These results confirm that effective mitigation of the Proximity Trap requires a balanced transformation that suppresses proximal background competition while preserving task-relevant evidence.