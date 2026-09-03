# Post-Training Ternarization of Qwen3-4B Capability, Efective Bit Budget, Storage Compression, and Deployment

CLOE V2.0 – OneBit AI

Qwen3-4B FP16 −→ KOTMS −→ E2M-ATQ −→ GPTQ compensation −→ W1.58A16

A capability, representation, compression, and deployment study of a 4B-parameter pretrained language model under aggressive post-training low-bit conversion.

Anirudh Malik | Poojith Devan | M Sparsh Mehra

OneBit AI | Technical Research Report | August 2026

Executive takeaway. This study asks a practical question: how far can an already-trained Qwen3-4B checkpoint be pushed toward an ultra-low-bit representation without full retraining, and what remains useful once representation, capability, storage, and execution are measured separately? Using the published TWLA pipeline components, with weight-only A16 execution, the resulting W1.58A16<sup>a</sup> retains substantial performance on several commonsense and context-use tasks while losing more on knowledge-heavy benchmarks. Across the ten scored comparisons in the final benchmark table, FP16 averages 64.5% accuracy and W1.58A16 54.7%, a mean cost of 9.8 percentage points. WikiText-2 perplexity increases from 13.639 to 18.748, while PTB and C4 expose a wider 1.30–1.46× degradation range. The representation itself is substantially smaller than FP16 once the quantizer’s lattice is persisted: the efective weight budget is 1.641 bits/weight and the later lossless packing record reports a 3.96 GiB artifact versus an 8.29 GiB pre-packed checkpoint. However, the current execution path does not establish a speed advantage. The central systems lesson is therefore explicit: compression is demonstrated; deployment acceleration remains an open engineering problem.

<sup>a</sup>Throughout this report, “ternary” is used as engineering shorthand rather than as a claim of a fully native single-plane ternary network. The evaluated model is a layer-wise mixed representation: most weights use one ternary plane, while a salience-selected subset uses a second ternary plane together with per-group continuous ofsets/scales and GPTQ compensation. The measured efective weight-information budget is approximately 1.641 bits/weight; embeddings and the language-model head remain BF16.

## Abstract

Ultra-low-bit language models are attractive because reducing weight precision can reduce persistent storage, memory bandwidth requirements, and the hardware resources required to move model parameters. Yet a nominal “1.58-bit” label does not by itself describe the representation actually stored, the amount of pretrained capability retained, or the runtime behavior of the resulting artifact.

We study an end-to-end post-training conversion of Qwen3-4B, an instruction-tuned 4B-parameter model, using the KOTMS rotation, E2M-ATQ ternarization, and GPTQ-style error compensation components of TWLA. The experiment is deliberately weight-only: activations remain at 16-bit precision and ILA-AMP is therefore skipped. The analysis emphasizes efective bit accounting, task-level capability retention, cross-corpus perplexity, calibration sensitivity, checkpoint composition, and deployment measurements.

The final evaluated conversion uses 1.641 efective bits/weight for the quantized linear weights, with 81.62% of model parameters targeted. On the ten scored capability comparisons, accuracy falls from 64.5% to 54.7%. The losses are not uniform: BoolQ retains 84.6% chance-corrected teacher performance, while ARC-Challenge retains 43.8%. The pattern is consistent with stronger degradation on knowledge-heavy tasks than on contextual plausibility tasks. Perplexity rises from 13.639 to 18.748 on WikiText-2, 24.700 to 31.992 on PTB, and 19.831 to 28.966 on C4.

The study also exposes an important deployment distinction. A later packing run persists the actual ternary planes and scales and reports 8.29 GiB to 3.96 GiB with essentially unchanged perplexity. A separate third-party packing attempt was lossy and is explicitly excluded from the primary artifact claim. The packed artifact has not been benchmarked end-to-end for task accuracy or generation throughput in this source set. A preliminary Triton GEMV microbenchmark is 4.6× slower than FP16 cuBLAS on one tested shape. We therefore do not claim that compression alone yields faster inference.

Keywords: post-training quantization, ternary quantization, 1.58-bit LLMs, Qwen3, GPTQ, model compression, deployment eficiency, low-bit inference

## 1 Introduction

The deployment cost of a language model is not determined only by the number of parameters. Transformer architectures themselves establish the dense-matrix foundation on which these compression methods operate.[1] The numerical representation of those parameters afects checkpoint size, memory trafic, loading time, accelerator utilization, and the feasibility of local or edge deployment. This has made low-bit model representations an active area of research, from conventional 8-bit and 4-bit PTQ through binary and ternary architectures.[2, 3, 4, 5, 6]

Ternary representations are particularly attractive because a single symbol drawn from {−1, 0, +1} has information content

$$
b _ { T } = \log _ { 2 } ( 3 ) \approx 1 . 5 8 5\tag{1}
$$

bits. This is the basis of the widely used “1.58-bit” terminology.[6] However, the information content of an ideal symbol is not the same thing as the storage cost of a complete model. Ofsets, scales, residual planes, salience metadata, non-quantized tensors, packing formats, and execution bufers all contribute to the deployed system.

The second distinction is capability. A pretrained model has already accumulated linguistic structure, factual associations, task heuristics, and representations. Aggressive post-training conversion therefore poses a diferent question from native low-bit training. Native ternary work asks whether a model can be trained under a constrained numerical regime; PTQ asks how much of an existing model survives conversion.[5, 6, 7, 8, 9]

The third distinction is execution. A smaller representation is not automatically a faster representation. Existing work such as AWQ and SmoothQuant demonstrates that hardware-aware transformations and kernels can turn lower precision into real deployment gains, but those gains depend on the target representation and execution stack.[3, 4] For a custom ultra-low-bit representation, unpacking, dequantization, rotation, auxiliary scales, and kernel overhead can erase the arithmetic advantage.

This paper therefore treats the conversion as a four-part empirical problem:

1. Capability: what downstream behavior survives?

2. Representation: what is the efective bit budget after adaptive planes and metadata are considered?

3. Storage: what size reduction is achieved by an actual artifact rather than a fake-quantized FP16 checkpoint?

4. Execution: does the compact representation improve inference on the tested hardware?

This framing is intentionally diferent from a leaderboard-only comparison. The objective is not to claim a new ternarization algorithm. KOTMS, E2M-ATQ, and GPTQ are prior methods. The contribution is an end-to-end empirical characterization of their application to Qwen3-4B, together with representation accounting, capability diagnostics, packing analysis, and deployment measurements.

Why this matters for an AI product. A compressed model is useful only if the representation can be stored, moved, loaded, and eventually executed at an acceptable cost. The present evidence supports a strong storage/compression result and a credible path toward hardware-aware execution, but it does not establish a universal inference-speed or serving-cost advantage. This distinction is central to the product interpretation of the work.

## 2 Research Positioning and Related Work

## 2.1 Native ternary training

BitNet established a Transformer architecture in which low-bit weights are part of the training formulation rather than a post-hoc conversion.[5] BitNet b1.58 subsequently popularized ternary weights {−1, 0, +1} and argued for the practical relevance of approximately 1.58-bit models.[6] These works provide the conceptual foundation for low-bit language modeling, but their training-time setting difers fundamentally from the present study.

## 2.2 General PTQ

GPTQ introduced one-shot post-training quantization using approximate second-order information to reduce the reconstruction error of large language models.[2] AWQ emphasized activation-aware protection of salient weights and hardware-friendly weight-only inference.[3] SmoothQuant instead transformed activation outliers into weight scaling to enable W8A8 execution.[4] OmniQuant explored optimization of quantization parameters in a training-free setting.[10]

These methods establish an important principle: calibration and representation design matter, but so does the execution format.

## 2.3 Ternary PTQ

Recent work has moved directly into the ternary regime. PTQTP represents weights using structured ternary planes and explicitly addresses the trade-of between expressiveness and execution complexity.[8] CAT-Q proposes a post-training ternary conversion with small calibration requirements.[7] ScaleQ-1.58 further studies calibration for reasoning models using self-generated reasoning traces.[11] TWLA combines KOTMS, E2M-ATQ, and ILA-AMP to target ternary weights and low-bit activations.[9]

The present work uses TWLA’s published components rather than claiming them as new. Importantly, our configuration is W1.58A16: the activation quantization component is deliberately disabled. The paper therefore characterizes the weight-only path and its practical storage/deployment implications rather than reproducing the full W1.58A4 system.

Table 1: Positioning of this study relative to representative low-bit directions.
<table><tr><td>Work</td><td>Regime</td><td>Core idea</td><td>Relevance to this study</td></tr><tr><td>BitNet</td><td>Native low-bit</td><td>Train directly with low-bit weights</td><td>Establishes native low-bit via- bility</td></tr><tr><td>BitNet b1.58</td><td>Native ternary</td><td>Ternary weights throughout training</td><td>Defines the 1.58-bit ternary tar- get</td></tr><tr><td>GPTQ</td><td>PTQ</td><td>Hessian-aware one-shot compensation</td><td>Provides the error- compensation paradigm</td></tr><tr><td>AWQ</td><td>PTQ</td><td>Activation-aware salient-weight protection</td><td>Demonstrates hardware-aware weight-only PTQ</td></tr><tr><td>PTQTP</td><td>Ternary PTQ</td><td>Structured trit planes</td><td>Closely related representation design</td></tr><tr><td>CAT-Q</td><td>Ternary PTQ</td><td>Calibration-efficient ternarization</td><td>Closely related pretrained con- version</td></tr><tr><td>ScaleQ-1.58</td><td>Ternary PTQ</td><td>Reasoning-trace calibration</td><td>Relevant to future calibration experiments</td></tr><tr><td>TWLA</td><td>Ternary PTQ</td><td>KOTMS + E2M-ATQ + ILA-AMP</td><td>Direct algorithmic basis of this experiment</td></tr><tr><td>This work</td><td></td><td>Empirical system study Qwen3-4B W1.58A16 characterization</td><td>Capability + bit accounting + storage + deployment</td></tr></table>

## 3 Model and Conversion Target

## 3.1 Base model

The base checkpoint is Qwen3-4B, an instruction-tuned 4B-parameter dense model with 36 transformer layers. The experiments were run from the public Qwen3 family checkpoint.[12]

The conversion targeted 252 linear tensors across all 36 layers. The architecture-level accounting in the project record identifies the MLP gate, up, and down projections together with attention �, �, �, and � projections as ternarized. Embeddings and the untied language-model head remain BF16. KOTMS rotation matrices are additional continuous-valued parameters.

Table 2: Parameter accounting for the converted Qwen3-4B checkpoint.
<table><tr><td>Component</td><td>Parameters</td><td>Share</td><td>Converted?</td></tr><tr><td>MLP gate projection</td><td>896.5 M</td><td>20.14%</td><td>Yes</td></tr><tr><td>MLP up projection</td><td>896.5 M</td><td>20.14%</td><td>Yes</td></tr><tr><td>MLP down projection</td><td>896.5 M</td><td>20.14%</td><td>Yes</td></tr><tr><td>Attention q projection</td><td>377.5 M</td><td>8.48%</td><td>Yes</td></tr><tr><td>Attention o projection</td><td>377.5 M</td><td>8.48%</td><td>Yes</td></tr><tr><td>Attention k projection</td><td>94.4 M</td><td>2.12%</td><td>Yes</td></tr><tr><td>Attention v projection</td><td>94.4 M</td><td>2.12%</td><td>Yes</td></tr><tr><td>Embeddings</td><td>389.0 M</td><td>8.74%</td><td>No, BF16</td></tr><tr><td>LM head</td><td>389.0 M</td><td>8.74%</td><td>No, BF16</td></tr><tr><td>KOTMS L, R</td><td>40.1 M</td><td>0.90%</td><td>No</td></tr><tr><td>RMSNorm</td><td>0.2 M</td><td>~0%</td><td>No</td></tr><tr><td>Ternarized linear weights</td><td>3.633 B</td><td>81.62%</td><td>Yes</td></tr></table>

The A16 designation is important: weights are aggressively quantized, but activations, attention operations, and the KV cache remain in FP16. This is not a fully low-bit execution path.

## 3.2 Terminology and representation boundary

The model is referred to as W1.58A16 throughout the report. When the shorter term “ternary” is used, it carries the qualification given in the footnote above: the representation is mixed and adaptive rather than a single ternary plane for every parameter.

The efective bit rate is therefore calculated from the observed use of one or two ternary planes, rather than by simply assigning every weight log 3 bits. The measured average is 1.641 bits/weight for the ternarized weight representation.

## 4 Method

## 4.1 End-to-end pipeline

The experimental pipeline is:

$$
\mathrm { F P l 6 ~ Q w e n 3 - 4 B }  \mathrm { K O T M S }  \mathrm { E 2 M - A T Q }  \mathrm { G P T Q ~ c o m p e n s a t i o n }  \mathrm { W 1 . 5 8 A l 6 } .\tag{2}
$$

KOTMS was reused from a pre-existing rotated checkpoint after its metadata was checked against the published recipe. ILA-AMP was skipped because the experiment used A16 activations. E2M-ATQ performed the actual weight conversion, followed by GPTQ-style error compensation.

![](images/351f49bbdf16fd1a1790a36df5c361f4642af32fc3d7f96248e18c12a5686927.jpg)  
Figure 1: Experimental conversion pipeline. The algorithmic components are prior work; the study contribution is their end-to-end characterization on Qwen3-4B.

## 4.2 KOTMS rotation

KOTMS uses structured orthogonal matrices to transform the weight space into a ternary-friendly basis. In simplified notation,

$$
W ^ { \prime } = W R , \qquad R ^ { \top } R = I .\tag{3}
$$

At inference, the equivalent rotation is applied to activations before the quantized linear operation. Because the transformation is orthogonal, it is norm-preserving in exact arithmetic. A prior project measurement reported rotation-only perplexity changing from 8.7998 to 8.8003, consistent with FP16 rounding noise.

For this experiment, a pre-existing rotated checkpoint was reused. Its metadata was verified against the TWLA configuration, including GMM iterations, learning rate, Cayley update, and balance parameters.

## 4.3 E2M-ATQ: order-2 asymmetric residual ternarization

For a masked weight block �, the conversion first computes a row-wise ofset:

$$
\mu = \operatorname { r o w m e a n } ( W ) , \qquad W _ { c } = W - \mu .\tag{4}
$$

A ternary support is then obtained using

$$
\Delta _ { 0 } = 0 . 7 5 \mathrm { r o w m e a n } ( | W _ { c } | )\tag{5}
$$

and

$$
T _ { 0 , i } = \left\{ \begin{array} { l l } { + 1 , } & { W _ { c , i } > \Delta _ { 0 } , } \\ { - 1 , } & { W _ { c , i } < - \Delta _ { 0 } , } \\ { 0 , } & { \mathrm { o t h e r w i s e } . } \end{array} \right.\tag{6}
$$

The first scale $\alpha _ { 0 }$ minimizes the squared reconstruction error:

$$
\alpha _ { 0 } = \arg \operatorname* { m i n } _ { \alpha } \| W _ { c } - \alpha T _ { 0 } \| _ { 2 } ^ { 2 } .\tag{7}
$$

The residual is then quantized again:

$$
R _ { 1 } = W _ { c } - \alpha _ { 0 } T _ { 0 } ,\tag{8}
$$

$$
\Delta _ { 1 } = 0 . 7 5 \mathrm { r o w m e a n } ( \left| R _ { 1 } \right| ) ,\tag{9}
$$

with a second ternary plane $T _ { 1 }$ and scale $\alpha _ { 1 }$ .

The reconstructed weight is therefore

$$
\widehat { W } = ( \mu + \alpha _ { 0 } T _ { 0 } + \alpha _ { 1 } T _ { 1 } ) \odot M ,\tag{10}
$$

where � denotes the applicable salience mask.

Fifteen alternating refinement iterations re-solve $\mu , \alpha _ { 0 } .$ , and �<sub>1</sub> against the current residual while holding the ternary supports fixed. This is the Euclidean-to-manifold refinement step.

The asymmetric ofset is not a cosmetic detail. Because $\mu \neq 0$ , the three values represented by one plane are efectively $\{ \mu - \alpha , \mu , \mu + \alpha \}$ rather than a globally centered $\{ - \alpha , 0 , + \alpha \}$ grid. Consequently, exact zeros are not expected to dominate the stored floating reconstruction, even though the underlying discrete supports are ternary.

## 4.4 Salience masks and efective bit budget

The GPTQ wrapper partitions each matrix into four disjoint Hessian-salience-ranked masks under the configuration

$$
\mathtt { n u m \_ m a s k } = 2 ( \mathtt { n u m \_ p } + 1 ) = 4 , \qquad \mathtt { o r d e r s } = [ 2 , 2 , 1 , 1 ] .\tag{11}
$$

Thus the two most salient masks receive two ternary planes and the remaining masks receive one. On the recorded layers.10.mlp.down\_proj.weight measurement, the mask coverages were 0.5%, 3.1%, 10.0%, and 86.5%, respectively. The first two masks therefore account for 3.6% of the measured weights, while 96.5% use a single plane.

The efective information budget is consequently

$$
B _ { \mathrm { e f f } } = \left( 1 + f _ { 2 } \right) \log _ { 2 } ( 3 ) ,\tag{12}
$$

where $f _ { 2 }$ is the fraction of weights assigned a second plane. The project-wide measured accounting gives

$$
B _ { \mathrm { e f f } } \approx 1 . 6 4 1 \ \mathrm { b i t s / w e i g h t . }\tag{13}
$$

This is the number that should be used when discussing the quantized weight representation. It is distinct from the ideal 1.585-bit information content of a single ternary symbol and from the total checkpoint size.

## 4.5 GPTQ error compensation

After ternary approximation, GPTQ-style compensation quantizes weights column-by-column while propagating quantization error into the remaining columns using the inverse Hessian estimated from calibration activations.[2]

The experimental configuration used WikiText-2 calibration, 64 samples, sequence length 2048, block size 128, group size 128, order2\_group=True, num\_p=1, and seed 0. The TWLA default damping parameter was percdamp=0.1. A later 0.01 experiment was attempted but did not complete and therefore provides no result.

## 5 Experimental Setup

## 5.1 Hardware and software

All primary measurements were obtained on a single NVIDIA RTX 5070 with 12,227 MiB VRAM under Windows. The recorded software environment was:

• PyTorch 2.11.0+cu128;

• CUDA 12.8;

• Transformers 4.53.2;

• Datasets 2.16.1;

• lm-evaluation-harness 0.4.12;

• NumPy 1.26.4;

• Python 3.10.11;

• TWLA commit 805cbb9.

The source record also identifies exact artifact hashes for the principal quantized checkpoint and rotated checkpoint. These should be retained in any release package.

## 5.2 Calibration

The model was calibrated on WikiText-2. The principal run used 64 calibration samples with sequence length 2048. The 64-sample choice was constrained by the widest projection’s calibration tensor and the practical VRAM ceiling after the execution environment was instrumented.

A critical experimental lesson is that memory reasoning based only on nominal GPU capacity was misleading. A broken no\_grad decorator caused a linear graph-retention leak from 1.99 GiB after the first calibration sample to 10.45 GiB by sample 64. Restoring the decorator reduced the per-layer requirement to below 3 GiB.

## 5.3 Perplexity evaluation

WikiText-2 was evaluated on the full test split using 146 sequences of 2048 tokens. PTB and C4 were subsequently evaluated to test the degree to which WikiText-2 calibration and evaluation might flatter the headline result.

The final source record gives:

• WikiText-2: 13.639 → 18.748;

• PTB: 24.700 → 31.992;

• C4: 19.831 → 28.966.

## 5.4 Capability evaluation

Capability was evaluated zero-shot with lm-evaluation-harness. The protocol used a limit of approximately 1500 examples per task, batch size 2, paired per-item records, and 10,000-resample paired bootstrap confidence intervals.

The project explicitly rejected raw accuracy ratios as a retention metric. For fixed-choice tasks, the corrected retention was defined as

$$
R = \frac { A _ { s } - B } { A _ { t } - B } ,\tag{14}
$$

where �<sub>�</sub> is student accuracy, �<sub>�</sub> is teacher accuracy, and � is the larger of the chance and majority-class floors.

This correction matters. A model exactly at chance can otherwise appear to retain non-zero “percentage” performance simply because the denominator is measured from zero. PIQA provides a concrete example: a raw ratio can be materially higher than the chance-corrected retention.

MMLU was additionally evaluated under both answer-letter and continuation scoring. This is necessary because the two protocols measure diferent output behaviors and because the earlier Cloe study demonstrated that answer-letter scoring can be distorted by degenerate output policies.

## 6 Main Results

## 6.1 Language modelling quality

![](images/f882a579faeba1025c7285031c35d7028be246e3ddb9c08a61de8520be503cb1.jpg)  
Figure 2: Perplexity on three corpora. WikiText-2 is the calibration distribution; PTB and C4 provide out-of-distribution checks.

Table 3: Perplexity across evaluation corpora.
<table><tr><td>Corpus</td><td>FP16</td><td>W1.58A16</td><td>Ratio</td></tr><tr><td>WikiText-2</td><td>13.639</td><td>18.748</td><td>1.375×</td></tr><tr><td>PTB</td><td>24.700</td><td>31.992</td><td>1.295×</td></tr><tr><td>C4</td><td>19.831</td><td>28.966</td><td>1.461×</td></tr></table>

The important observation is not that one number is “good” or “bad”, but that the degradation depends on the evaluation corpus. WikiText-2 gives the most favorable ratio, while C4 gives the least favorable ratio. This motivates reporting a range rather than presenting 1.375× as an invariant property of the model.

The C4 result is particularly useful scientifically because it demonstrates a calibration-distribution efect. It does not prove that C4 is universally harder, but it shows that the WikiText-2 headline is not suficient to characterize language-modelling degradation.

## 6.2 Task-level capability

![](images/66a7fbb12c31a426d432b5ca7900fdc78793bbabab175a8bb1bbf79e444c38fd.jpg)  
Figure 3: Task-level accuracy for FP16 and W1.58A16.

Table 4: Final task-level comparison. Confidence intervals are paired bootstrap intervals reported by the project.
<table><tr><td>Task</td><td>FP16</td><td>W1.58A16</td><td>∆ pts</td><td>95% CI</td><td>Retention</td></tr><tr><td>BoolQ</td><td>84.7</td><td>81.4</td><td>-3.3</td><td>[-5.5,-1.3]</td><td>84.6%</td></tr><tr><td>HellaSwag</td><td>59.5</td><td>52.9</td><td>-6.5</td><td>[-8.5,-4.5]</td><td>80.6%</td></tr><tr><td>PIQA</td><td>74.2</td><td>68.1</td><td>-6.1</td><td>[-8.1,-4.1]</td><td>73.9%</td></tr><tr><td>WinoGrande</td><td>66.0</td><td>61.6</td><td>-4.4</td><td>[-7.4,-1.3]</td><td>72.3%</td></tr><tr><td>ARC-Easy</td><td>78.9</td><td>62.4</td><td>-16.5</td><td>[-18.7,-14.3]</td><td>68.5%</td></tr><tr><td>MMLU (letter)</td><td>68.5</td><td>51.0</td><td>-17.5</td><td>point est.</td><td>59.7%</td></tr><tr><td>MMLU (continuation)</td><td>45.1</td><td>35.0</td><td>-10.1</td><td>point est.</td><td>50.0%</td></tr><tr><td>ARC-Challenge</td><td>54.0</td><td>38.6</td><td>-15.4</td><td>[-18.2,-12.5]</td><td>43.8%</td></tr><tr><td>LAMBADA-openai</td><td>59.9</td><td>52.1</td><td>-7.7</td><td>[-9.9,-5.5]</td><td>guarded</td></tr><tr><td>LAMBADA-standard</td><td>54.5</td><td>44.3</td><td>-10.2</td><td>[-12.5,-7.9]</td><td>guarded</td></tr><tr><td>Arithmetic mean</td><td>64.5</td><td>54.7</td><td>-9.8</td><td>一</td><td>一</td></tr></table>

No reported confidence interval crosses zero. The losses are therefore detectable under the tested paired bootstrap protocol, although the study uses one seed and a finite evaluation limit.

The arithmetic mean of 64.5% versus 54.7% is a descriptive average over the ten listed comparisons. It is not a claim of average performance over all language benchmarks, nor a chance-corrected aggregate retention score. The LAMBADA entries are explicitly guarded because an open-vocabulary task does not have the same simple chance floor as fixed-choice multiple-choice tasks.

## 6.3 Capability stratification

Chance-corrected retention where a meaningful floor is defined  
![](images/487c295beeaaa559ddad4e00680bb733cc250b0e74f354bceccbaadf5f928605.jpg)  
Figure 4: Chance-corrected retention for tasks with a meaningful baseline floor. LAMBADA is omitted because the source protocol guards against an invalid floor.

The strongest retained behavior is contextual reading and commonsense plausibility. BoolQ retains 84.6% of teacher performance. HellaSwag, PIQA, and WinoGrande fall in the 72–81% retention range. ARC-Easy is lower at 68.5%, while the harder knowledge-oriented tasks fall further: MMLU continuation retains 50.0%, and ARC-Challenge 43.8%.

The pattern is therefore stratified rather than uniform. The evidence is consistent with the hypothesis that aggressive post-training discretization disproportionately damages capabilities that depend on precise stored knowledge or fine-grained factual associations. It would be too strong, however, to claim a causal law from this small benchmark suite. The present result is a structured empirical observation requiring confirmation on broader task families.

Capability change after post-training ternarization  
![](images/f34f908b54d1a70d993b9a760958c998c11ca11136b4dcdf4e97db06f55e674b.jpg)  
Figure 5: Absolute accuracy change by task. The losses are largest on ARC-Easy, MMLU, and ARC-Challenge.

## 6.4 MMLU scoring protocol

MMLU is especially sensitive to evaluation protocol. The final record reports:

• letter scoring: 68.50% FP16 versus 50.96% W1.58A16;

• continuation scoring: 45.10% FP16 versus 35.04% W1.58A16.

The diference between protocols is substantial for both arms. The absolute FP16–student gap is 23.4 points under letter scoring and 10.1 points under continuation scoring. Consequently, neither MMLU number should be quoted without the scoring protocol.

This is not merely a reporting detail. In extreme quantization, output distributions can become biased toward particular answer formats. A protocol that converts answer formatting into benchmark score can therefore confound capability with decoding behavior.

## 7 Representation Analysis

## 7.1 From nominal 1.585 bits to 1.641 efective bits

The theoretical lower bound of 1.585 bits/weight corresponds to one ternary symbol per weight. The evaluated representation uses a second plane selectively. The measured project-level efective budget is 1.641 bits/weight.

This is an important improvement in accounting precision relative to simply calling the model “1.58-bit”. The latter is a useful shorthand for the ternary regime, but it does not fully describe the adaptive representation.

Table 5: Observed salience-mask structure on a recorded projection.
<table><tr><td>Mask</td><td>Planes</td><td>Coverage</td><td>Bits/weight</td></tr><tr><td>0</td><td>2</td><td>0.5%</td><td>3.17</td></tr><tr><td>1</td><td>2</td><td>3.1%</td><td>3.17</td></tr><tr><td>2</td><td>1</td><td>10.0%</td><td>1.58</td></tr><tr><td>3</td><td>1</td><td>86.5%</td><td>1.58</td></tr></table>

The 3.6% of weights receiving a second plane are not arbitrary. They are selected by Hessian-based salience. This is conceptually aligned with the broader observation in PTQ that parameter importance is highly non-uniform.[2, 3]

## 7.2 Why the stored floating tensor can look “non-ternary”

A naive inspection of a reconstructed FP16 tensor can report many distinct floating values per row. This does not imply that the discrete representation failed. The representation combines ofsets, scales, one or two ternary supports, salience masks, and GPTQ compensation.

In fact, the project observed a particularly useful verification signature: the number of distinct reconstructed values in a recorded row collapsed from 1,169 in the FP16 state to 235 after conversion. The decisive check was end-to-end perplexity: the converted model reproduced approximately 18.748 rather than the FP16 value of 13.639. This guards against accidentally evaluating a rotated or unconverted checkpoint

## 7.3 Representation versus complete checkpoint

The 1.641 bits/weight figure applies to the targeted weight representation. It does not mean that the complete checkpoint occupies 1.641 bits per parameter.

The complete model retains BF16 embeddings and the language-model head, plus KOTMS rotation matrices and other metadata. The source accounting estimates approximately 745 MiB for packed ternary weights, 1,483 MiB for the untouched embedding/head tensors, and about 78 MiB for the rotation matrices, for a projected total of approximately 2,271 MiB before the later verified packed artifact.

This is why storage measurements must be reported separately from theoretical bit rate.

## 8 Calibration and Quality Sweep

![](images/3868bc79e31e4302d6c05059fc457c7958e7929816808e741e62994afc16cc2a.jpg)  
Figure 6: Quality sweep on WikiText-2. The R1 increase from 64 to 96 calibration samples worsened perplexity by 1.8%.

The quality sweep changed one variable per run. Stage A used WikiText-2 calibration with 64 samples and percdamp=0.1, producing 18.749 perplexity. R1 increased the calibration sample count to 96 while holding the other configuration fixed, producing 19.087.

This is a negative result, not evidence that 64 samples are inherently better than 96. The source record correctly interprets the 1.8% diference as “no improvement from more calibration data” because the experiment is single-seed and the gap is small.

The remaining levers were not completed in the provided evidence:

• percdamp=0.01 was exposed but the R2 run paused early;

• a FineWeb-Edu calibration run was queued but not started;

• num\_p=2 was not evaluated;

• group\_size=64 was not evaluated.

The report must therefore not imply that the presented configuration is globally optimal. It is the best completed configuration in the provided sweep, not a proven optimum.

Calibration-memory scaling on the widest projection  
![](images/c7262e1974fd876e742efb2405cbc5e03a8772182d141e0052e8513264979a0d.jpg)  
Figure 7: Calibration memory scaling for the widest projection. The 128-sample request exceeds the measured allowed VRAM budget.

The widest projection requires a calibration tensor whose memory scales linearly with the number of samples. The recorded estimates are 6.39 GiB at 64 samples, 9.58 GiB at 96, 12.78 GiB at 128, and 25.56 GiB at 256. The practical allowed budget was approximately 10.98 GiB. This explains why 128 samples could not be used in the final run under the measured environment.

## 9 Storage and Packing

## 9.1 Fake-quantized versus actually packed

The first saved quantized checkpoint stored the converted values inside FP16 tensors. Consequently, it did not reduce disk footprint despite representing the weights at a much lower efective information rate. This is a critical distinction between quantization and serialization.

A later packing run persisted the actual ternary planes and their associated scales/ofsets using a lattice-aware format. The source record reports:

Table 6: Recorded packing result.
<table><tr><td>Quantity</td><td>Before</td><td>Packed</td></tr><tr><td>Checkpoint size</td><td>8.29 GiB</td><td>3.96 GiB</td></tr><tr><td>WikiText-2 perplexity</td><td>18.344322</td><td>18.341623</td></tr><tr><td>PTB perplexity</td><td>34.052238</td><td>34.042122</td></tr></table>

The packing report also records 0.037% relative RMSE and a 50-second packing time. Most importantly, the packed artifact reproduced the un-packed model’s perplexity to within the reported measurement precision.

![](images/570af2f210f39587a936b299f5ac8acf6fa2aad475460c863d3ab090a6f6480b.jpg)  
Figure 8: Projected composition of the packed checkpoint. The BF16 embedding/head tensors dominate the remaining footprint.

The diference between 1.641 bits/weight and 3.96 GiB is therefore not a contradiction. The former describes the adaptive quantized weight representation; the latter describes an entire serialized checkpoint with unquantized components and storage metadata.

## 9.2 The excluded lossy packing attempt

A separate third-party packing artifact used a single 2-bit ternary encoding with FP16 per-group scales. Its recorded whole-checkpoint size was 2.659 GiB, but it introduced a maximum absolute weight error of 0.305 because the second ternary plane and GPTQ residual were discarded.

The artifact had no perplexity or task benchmark. Its own analysis explicitly concluded that it should not be published as-is. We therefore do not use its size or speed as evidence for the primary model claim. This distinction prevents an attractive but unsupported compression number from contaminating the paper.

The underlying engineering lesson is valuable: post-hoc packing cannot reconstruct information that the quantizer discarded at save time. A correct format must persist the discrete supports and their scales during quantization.

## 10 Deployment and Execution

## 10.1 Reconstruction-based execution

The fake-quantized W1.58A16 model was benchmarked directly against FP16. The measured results were:

Table 7: Measured deployment behavior before native packed execution.
<table><tr><td>Metric</td><td>FP16</td><td>W1.58A16</td><td>Relative</td></tr><tr><td>Load time</td><td>6.57 s</td><td>12.39 s</td><td>1.89×</td></tr><tr><td>Weight VRAM</td><td>7.545 GiB</td><td>7.620 GiB</td><td>1.01×</td></tr><tr><td>Prefill</td><td>5,919 tok/s</td><td>5,029 tok/s</td><td>0.85×</td></tr><tr><td>Decode</td><td>27.41 tok/s</td><td>21.28 tok/s</td><td>0.78×</td></tr><tr><td>Latency</td><td>36.49 ms/tok</td><td>46.98 ms/tok</td><td>1.29×</td></tr></table>

Measured deployment ratio before true packed execution  
![](images/febb0e0b0c02a27a83576e0c39a66ce912eb04d99bb31765956bf8642f4532fd.jpg)  
Figure 9: Execution ratios relative to FP16. Values above 1 indicate higher cost; values below 1 indicate lower throughput.

These results are intentionally not presented as a failure of ternary representations in general. The measured path is a reconstruction-oriented implementation: weights are still stored in FP16, and the KOTMS rotation introduces a real activation-side matrix operation. The experiment therefore measures a specific software stack, not an optimized native ternary accelerator.

This is exactly why storage compression and inference eficiency must remain separate claims.

## 10.2 Preliminary packed-kernel microbenchmark

A separate Triton GEMV kernel was measured on a 4096 × 2560 projection on the RTX 5070:

Table 8: Preliminary kernel microbenchmark.
<table><tr><td>Path</td><td>Time (ms)</td><td>Relative to FP16 cuBLAS</td></tr><tr><td>FP16 cuBLAS</td><td>0.0451</td><td>1.0×</td></tr><tr><td>Packed Triton</td><td>0.2082</td><td>4.6× slower</td></tr><tr><td>Chunk-unpack + cuBLAS</td><td>0.8924</td><td>19.8× slower</td></tr></table>

![](images/110c145c0df77adfbd63b4fa2a4509aa8f0e1510a8645820d8189a9f61091ba8.jpg)  
Figure 10: Preliminary GEMV timing. The measurement demonstrates a kernel bottleneck, not an intrinsic lower bound on ternary execution.

The kernel’s own dense-matmul comparison reported approximately 0.00195 maximum absolute discrepancy, indicating that the kernel can reproduce its encoded dense operation to FP16-level numerical precision. The larger 0.305 error reported for the third-party packing artifact belongs to the encoding and re-quantization mismatch, not to the Triton kernel’s arithmetic.

The systems implication is clear. A compact representation becomes a deployment advantage only when the execution stack consumes that representation directly and eficiently. Existing optimized low-bit systems such as AWQ/TinyChat and BitNet’s inference stack demonstrate how much of the practical benefit comes from hardware-aware packing and kernels rather than numerical precision alone.[3, 6]

## 11 Failure Modes as Experimental Boundary Conditions

The project encountered multiple infrastructure and methodology failures. They are relevant because several could have produced invalid scientific conclusions.

## 11.1 Autograd-induced memory leak

A local checkpointing edit inserted a helper between @torch.no\_grad() and the target function, causing the calibration forward to retain computation graphs. Memory increased from 1.99 GiB after the first sample to 10.45 GiB after 64 samples. The bug was fixed and per-layer memory returned to below 3 GiB.

The scientific lesson is broader than this individual bug: VRAM constraints should be instrumented rather than inferred from model size alone.

## 11.2 Windows CUDA spill

A probe demonstrated that CUDA allocations beyond physical VRAM could spill into system memory under the Windows configuration, producing misleading memory readings and severe performance degradation. The project added a 0.92 per-process memory fraction limit so over-allocation fails cleanly.

## 11.3 False ternary artifact

A pre-existing 8.5 GiB checkpoint appeared to be a candidate ternary artifact but had not actually been ternarized. The verification layer rejected it. Conversely, the real asymmetric E2M-ATQ representation initially looked non-ternary because exact zeros were rare. The project resolved this by inspecting the quantizer’s asymmetric ofset and enforcing end-to-end perplexity verification.

This is an unusually important reproducibility point. A low-bit experiment should verify both the representation and the forward path.

## 11.4 Power loss and checkpoint recovery

The machine rebooted approximately 15 minutes before completion of a multi-hour run, losing roughly 2.5 hours of work.   
Per-layer checkpointing was added so that future interruptions cost approximately one layer rather than the entire run.

## 11.5 Benchmark protocol defects

An adversarial audit caught several issues before final benchmark acceptance:

1. raw accuracy ratios were rejected as a retention measure because chance floors difer by task;

2. default MMLU letter scoring was identified as potentially misleading;

3. per-item logs were made mandatory;

4. both model arms were evaluated per task to avoid an incomplete paired comparison.

These corrections are part of the experimental method, not merely editorial cleanup.

## 12 Comparison with the Previous OneBit AI Study

The previous Cloe study converted Qwen3.5-0.8B using 72.4M tokens of QAT and found a severe loss of specialist knowledge, while still observing measurable downstream learnability and deployment benefits on the tested packed artifact.[13]

The present study changes the central conversion strategy from QAT to PTQ. The new model is larger, the conversion requires no gradient training, and the quantizer uses Hessian-aware error compensation. This allows the paper to ask a diferent question: whether an existing 4B model can be aggressively discretized without spending a large token budget on recovery.

The comparison should not be interpreted as “PTQ universally beats QAT”. The experiments do not support that claim. The previous Cloe paper itself noted that its QAT budget was limited and that the observed capability gap could not be uniquely separated into irreversible quantization loss and incomplete recovery. The current project also does not provide a controlled QAT/PTQ head-to-head on the same model and compute budget.

What can be said is narrower and stronger: the PTQ route produces a working 4B W1.58A16 conversion without full retraining, with substantial retained capability on several tasks and a verified path to multi-gigabyte checkpoint compression.

The two papers also share an evaluation lesson. Capability should be stratified rather than reduced to a single average. In Cloe, knowledge-heavy behavior approached chance; in the present 4B conversion, knowledge-heavy tasks are degraded but remain substantially above chance on several benchmarks. This suggests that model scale and conversion procedure both matter, but the current evidence is insuficient to isolate their individual causal efects.

Table 9: High-level comparison with the previous OneBit AI Cloe study.
<table><tr><td>Dimension</td><td>Previous Cloe study</td><td>Present study</td></tr><tr><td>Base model</td><td>Qwen3.5-0.8B</td><td>Qwen3-4B</td></tr><tr><td>Conversion</td><td>QAT</td><td>PTQ</td></tr><tr><td>Training budget</td><td>72.4M tokens</td><td>No training; calibration only</td></tr><tr><td>Representation</td><td>Fully ternary targeted layers</td><td>Adaptive order-1/order-2 ternary planes</td></tr><tr><td>Primary emphasis</td><td>Capability stratification + learnability</td><td>Capability + bit accounting + storage + exe- cution</td></tr><tr><td>Main deployment evidence Packed artifact + throughput</td><td></td><td>Verified packed artifact; native packed speed remains open</td></tr><tr><td>Key limitation</td><td>Limited recovery budget</td><td>Single seed, incomplete tuning/reproduc- tion gate</td></tr></table>

## 13 Scientific Interpretation

## 13.1 Hypothesis assessment

The motivating hypothesis can be stated as follows:

An aggressively post-training quantized 4B pretrained language model can retain useful downstream capability while reducing its efective weight precision toward the ternary information limit, but practical deployment eficiency will depend on how the representation is serialized and executed.

The evidence supports this hypothesis in part.

First, the capability component is supported: the model remains substantially above chance on several tasks, with BoolQ at 81.4%, PIQA at 68.1%, and WinoGrande at 61.6%. At the same time, the hypothesis does not imply negligible degradation; ARC-Challenge and MMLU show substantial losses.

Second, the representation component is supported: the measured quantized weight budget is 1.641 bits/weight, close to the ternary information floor, while 96.5% of weights in the recorded salience example use a single plane.

Third, the storage component is supported by the later lossless packing record: 8.29 GiB becomes 3.96 GiB with essentially unchanged perplexity.

Fourth, the execution component is not yet supported as an advantage. The reconstruction path is slower than FP16, and the preliminary packed GEMV kernel is slower than cuBLAS. This is not a contradiction of the hypothesis; it identifies the

remaining engineering condition for turning representation eficiency into runtime eficiency.

## 13.2 What is genuinely new in this study

The paper does not claim novelty for:

• ternary weights;

• the 1.58-bit information limit;

• KOTMS;

• E2M-ATQ;

• GPTQ;

• generic post-training quantization;

• native ternary inference as a concept.

The defensible contributions are empirical and system-oriented:

1. an end-to-end Qwen3-4B application of the TWLA weight-only pipeline;

2. explicit efective-bit accounting for adaptive second-plane usage;

3. task-level characterization of capability degradation under aggressive PTQ;

4. cross-corpus perplexity analysis showing calibration-distribution sensitivity;

5. a verified lossless packing path with measured whole-checkpoint reduction;

6. a deployment study showing why the current execution path does not yet convert compression into speed;

7. engineering and evaluation safeguards that prevent several classes of false low-bit results.

This is a meaningful technical characterization even though the underlying quantizer is prior art.

## 14 Product and Deployment Implications

The strongest product narrative is compression-first, acceleration-next.

The current evidence supports a materially smaller model artifact and establishes that a 4B pretrained model can be pushed into a near-ternary weight regime without full retraining. That can matter for model distribution, checkpoint transfer, local storage, and future hardware-aware serving.

The evidence does not yet support the stronger statement that the current artifact is cheaper or faster to serve on arbitrary hardware. The measured reconstruction path is slower than FP16, and the preliminary Triton kernel is substantially slower on the tested GEMV. The commercial opportunity therefore lies in combining the demonstrated representation with a native execution stack rather than treating compression itself as the finished product.

The most immediate deployment opportunities are:

1. Model distribution: a 3.96 GiB artifact is materially easier to move and store than an 8.29 GiB checkpoint.

2. Memory-constrained deployment: further quantization of the BF16 embedding/head tensors could reduce the remaining footprint.

3. Native kernels: direct consumption of the stored ternary planes can remove reconstruction overhead.

4. Hardware co-design: the low-cardinality arithmetic structure is compatible with specialized low-bit kernels and accelerators, as demonstrated conceptually by prior BitNet systems.[6]

5. Specialized models: the capability profile suggests that broad general-purpose replacement is not yet justified, while targeted workloads may ofer a more favorable quality-eficiency trade-of.

## 14.1 What the current evidence does not establish

The following claims should not be made from this dataset:

• universal inference speedup;

• universal serving-cost reduction;

• equivalence to a native 1.58-bit model;

• state-of-the-art capability among ternary models;

• superiority over every alternative PTQ method;

• production robustness across hardware;

• general scaling laws across model sizes.

The study is strongest when it treats these as future engineering and research questions rather than as conclusions already settled.

## 15 Limitations and Required Next Experiments

The principal limitations are:

1. No independent reproduction gate on Qwen3-4B. The source record notes that Qwen3-4B is absent from the TWLA paper. A reproduction gate on the authors’ published reference model remains the highest-value validation.

2. Single seed. Capability results do not quantify run-to-run variation.

3. Incomplete tuning. The percdamp=0.01 run did not complete; alternative calibration corpora and representation parameters were not evaluated.

4. Incomplete packed-model capability evaluation. The later packing record demonstrates perplexity preservation but the provided source set does not contain an end-to-end packed-model task benchmark.

5. Preliminary kernel measurement. The Triton GEMV is a prototype and should not be treated as a lower bound on native ternary performance.

6. Partial parameter quantization. Embeddings and the LM head remain BF16 and account for a substantial fraction of the remaining artifact.

7. Limited benchmark breadth. The final task suite is informative but smaller than comprehensive LLM evaluations.

The highest-value next experiments are therefore:

1. complete the published-method reproduction gate;

2. complete percdamp=0.01;

3. rerun with a broader calibration corpus such as FineWeb-Edu;

4. benchmark the exact packed artifact end-to-end;

5. persist the discrete lattice directly during quantization and build a lossless native kernel;

6. evaluate embedding/head quantization;

7. repeat the capability suite across seeds;

8. evaluate the same conversion procedure on at least one additional model scale.

## 16 Conclusion

This study demonstrates that a pretrained Qwen3-4B model can be pushed into an aggressive post-training low-bit regime with a measured 1.641-bit/weight quantized representation and substantial retained capability on a subset of downstream tasks. The result is not uniform preservation: contextual and commonsense tasks are more robust than knowledge-heavy tasks, and the mean accuracy cost across the ten reported comparisons is 9.8 percentage points.

The storage story is stronger than the runtime story. A later lattice-aware packing run reduces the checkpoint from 8.29 GiB to 3.96 GiB while preserving the recorded perplexity to measurement precision. This establishes that the representation can become a materially smaller artifact rather than merely a fake-quantized FP16 checkpoint.

Execution remains the key unresolved systems problem. The reconstruction-based path is slower than FP16, and the preliminary packed Triton GEMV is 4.6× slower than FP16 cuBLAS on the tested shape. These measurements should not be generalized into a claim that ternary arithmetic is intrinsically slower; they show that the current software path is not yet hardware-eficient.

The strongest interpretation of the work is therefore neither “ternarization solves eficient inference” nor “ternarization destroys model capability.” The evidence supports a more useful engineering conclusion: near-ternary post-training conversion can create a substantially smaller 4B model artifact while retaining meaningful portions of pretrained capability, but turning that representation into a production eficiency advantage requires native packing, kernels, and broader validation.

## Reproducibility Note

All quantitative claims in this report are derived from the provided OneBit AI project records. The primary completed conversion used Qwen3-4B, KOTMS rotation, W1.58A16, 64 calibration samples, sequence length 2048, WikiText-2 calibration, percdamp=0.1, block size 128, group size 128, order2\_group=True, num\_p=1, and seed 0. The principal environment was Windows with an NVIDIA RTX 5070 (12,227 MiB), PyTorch 2.11.0+cu128, Transformers 4.53.2, Datasets 2.16.1, lm-eval 0.4.12, and Python 3.10.11.

The report deliberately excludes unsupported results from the third-party lossy packed artifact. It also treats the unfinished percdamp=0.01 experiment as unfinished rather than assigning it a result.

A Complete Experimental Ledger
<table><tr><td>Stage</td><td>Configuration</td><td>Outcome</td><td>Interpretation</td></tr><tr><td>Stage A</td><td>WikiText-2, 64 samples, 18.749 ppl pd=0.1</td><td></td><td>Best completed configuration</td></tr><tr><td>R1</td><td>WikiText-2, 96 samples, 19.087 ppl pd=0.1</td><td></td><td>Negative result; no improvement from more samples</td></tr><tr><td>R2</td><td>WikiText-2, 64 samples, Incomplete pd=0.01</td><td></td><td>No result; must not be cited as an experiment outcome</td></tr><tr><td>R3</td><td>Alternative calibration</td><td>Not started</td><td>Open experiment</td></tr><tr><td>Packing A</td><td>Lattice-aware serialization 8.29 GiB → 3.96 GiB; Primary packing result</td><td>PPL preserved</td><td></td></tr><tr><td>Packing B</td><td>Third-party 2-bit re- quantization</td><td>0.305</td><td>2.659 GiB; max error Excluded from primary artifact claim</td></tr><tr><td>Kernel</td><td>Packed Triton GEMV</td><td>0.2082 ms</td><td>4.6× slower than FP16 cuBLAS on one shape</td></tr></table>

## B Capability Table with Baselines

<table><tr><td>Task</td><td>Baseline</td><td>FP16</td><td>W1.58A16</td><td>Δ</td><td>CI</td><td>Retention</td></tr><tr><td>BoolQ</td><td>63.1</td><td>84.7</td><td>81.4</td><td>-3.3</td><td>[-5.5,-1.3]</td><td>84.6</td></tr><tr><td>HellaSwag</td><td>25.8</td><td>59.5</td><td>52.9</td><td>-6.5</td><td>[-8.5,-4.5]</td><td>80.6</td></tr><tr><td>PIQA</td><td>50.7</td><td>74.2</td><td>68.1</td><td>-6.1</td><td>[-8.1,-4.1]</td><td>73.9</td></tr><tr><td>WinoGrande</td><td>50.0</td><td>66.0</td><td>61.6</td><td>-4.4</td><td>[-7.4,-1.3]</td><td>72.3</td></tr><tr><td>ARC-Easy</td><td>26.4</td><td>78.9</td><td>62.4</td><td>-16.5</td><td>[-18.7,-14.3]</td><td>68.5</td></tr><tr><td>MMLU (letter)</td><td>25.0</td><td>68.5</td><td>51.0</td><td>-17.5</td><td>point</td><td>59.7</td></tr><tr><td>MMLU (cont.)</td><td>25.0</td><td>45.1</td><td>35.0</td><td>-10.1</td><td>point</td><td>50.0</td></tr><tr><td>ARC-Challenge</td><td>26.5</td><td>54.0</td><td>38.6</td><td>-15.4</td><td>[-18.2,-12.5]</td><td>43.8</td></tr><tr><td>LAMBADA-openai</td><td>一</td><td>59.9</td><td>52.1</td><td>-7.7</td><td>[-9.9,-5.5]</td><td>guarded</td></tr><tr><td>LAMBADA-standard</td><td>一</td><td>54.5</td><td>44.3</td><td>-10.2</td><td>[-12.5,-7.9]</td><td>guarded</td></tr></table>

## C Implementation and Reproducibility Checklist

• Base model revision: Qwen3-4B, instruction-tuned.

• Quantization method: TWLA E2M-ATQ with KOTMS rotation and GPTQ compensation.

• Activations: FP16/A16.

• Calibration: WikiText-2, 64 samples, 2048 tokens.

• Quantization damping: 0.1.

• Block/group size: 128.

• Order-2 salience grouping: enabled.

• Number of additional-plane groups: num\_p=1.

• Seed: 0.

• Evaluation: full WikiText-2 test split for PPL; zero-shot benchmark suite for capability.

• Statistical method: paired bootstrap, 10,000 resamples.

• Hardware: NVIDIA RTX 5070, 12,227 MiB, Windows.

• Software: PyTorch 2.11.0+cu128; Transformers 4.53.2; Datasets 2.16.1; lm-eval 0.4.12; NumPy 1.26.4; Python 3.10.11.

• Quantized checkpoint hash prefix: 13B6608425DF3245.

• Rotated checkpoint hash prefix: DE3769B658762A2F.

## D Interpretation Guardrails

The following statements are recommended for any future summary of this work:

1. “The model uses an adaptive ternary weight representation with a measured efective budget of 1.641 bits/weight.”

2. “The completed conversion retains substantial performance on several commonsense and contextual tasks, while knowledge-heavy tasks degrade more strongly.”

3. “A later lattice-aware packing run reduces the recorded checkpoint from 8.29 GiB to 3.96 GiB with essentially unchanged perplexity.”

4. “The current execution path is slower than FP16; native hardware-eficient execution remains future work.”

## Avoid:

1. “1.58-bit means the whole checkpoint is 1.58 bits/parameter.”

2. “The packed model is faster” without a full end-to-end benchmark of the packed artifact.

3. “The method is state of the art” based on these experiments.

4. “64 calibration samples are better than 96” from the single R1 comparison.

5. “PTQ is better than QAT” based only on comparison with the previous Cloe study.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. Attention is all you need. In Advances in Neural Information Processing Systems, volume 30, 2017. doi: 10.48550/arXiv.1706.03762. URL https://arxiv.org/abs/1706.03762.

[2] Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. Gptq: Accurate post-training quantization for generative pre-trained transformers. In International Conference on Learning Representations, 2023. doi: 10.48550/arXiv.2210.17323. URL https://arxiv.org/abs/2210.17323.

[3] Ji Lin, Jiaming Tang, Haotian Tang, Shang Yang, Wei-Ming Chen, Wei-Chen Wang, Guangxuan Xiao, Xingyu Dang, Chuang Gan, and Song Han. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. In Proceedings of Machine Learning and Systems, volume 6, 2024. URL https://proceedings.mlsys.org/ paper\_files/paper/2024/hash/42a452cbafa9dd64e9ba4aa95cc1ef21-Abstract-Conference.html.

[4] Guangxuan Xiao, Ji Lin, Mickael Seznec, Hao Wu, Julien Demouth, and Song Han. Smoothquant: Accurate and eficient post-training quantization for large language models. In Proceedings ofthe 40th International Conference on Machine Learning, volume 202, pages 38087–38099. PMLR, 2023. doi: 10.48550/arXiv.2211.10438. URL https://proceedings.mlr.press/v202/xiao23c.html.

[5] Hongyu Wang, Shuming Ma, Li Dong, Shaohan Huang, Huaijie Wang, Lingxiao Ma, Fan Yang, Ruiping Wang, Yi Wu, and Furu Wei. Bitnet: Scaling 1-bit transformers for large language models, 2023. URL https://arxiv. org/abs/2310.11453.

[6] Shuming Ma, Hongyu Wang, Lingxiao Ma, Lei Wang, Wenhui Wang, Shaohan Huang, Li Dong, Ruiping Wang, Jilong Xue, and Furu Wei. The era of 1-bit llms: All large language models are in 1.58 bits, 2024. URL https://arxiv.org/abs/2402.17764.

[7] Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, and Anbang Yao. Cat-q: Cost-eficient and accurate ternary quantization for llms, 2026. URL https://arxiv.org/abs/2606.26650.

[8] He Xiao, Runming Yang, Qingyao Yang, Wendong Xu, Zheng Li, Yupeng Su, Zhengwu Liu, Hongxia Yang, and Ngai Wong. Ptqtp: Post-training quantization to trit-planes for large language models, 2025. URL https: //arxiv.org/abs/2509.16989.

[9] Zhixiong Zhao, Zukang Xu, Zhixuan Chen, Xing Hu, Zhe Jiang, and Dawei Yang. Twla: Achieving ternary weights and low-bit activations for llms via post-training quantization, 2026. URL https://arxiv.org/abs/2606.13054.

[10] Wenqi Shao, Mengzhao Chen, Zhaoyang Zhang, Peng Xu, Lirui Zhao, Zhiqian Li, Kaipeng Zhang, Gao Peng, Yu Qiao, and Ping Luo. Omniquant: Omnidirectionally calibrated quantization for large language models. In International Conference on Learning Representations, 2024. doi: 10.48550/arXiv.2308.13137. URL https: //arxiv.org/abs/2308.13137.

[11] Shigeng Wang, Chao Li, Yangyuxuan Kang, Jiawei Fan, and Anbang Yao. Attend to your own thoughts: Breaking the barrier for post-training quantization of reasoning llms through the lens of 1.58-bit quantization, 2026. URL https://arxiv.org/abs/2608.01078.

[12] An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, Chujie Zheng, Dayiheng Liu, Fan Zhou, Fei Huang, Feng Hu, Hao Ge, Haoran Wei, Huan Lin, Jialong Tang, Jian Yang, Jing Zhou, Jingren Zhou, Junyang Lin, Kai Dang, Keqin Bao, Kexin Yang, Yu Le, Lianghao Deng, Mei Li, Mingfeng Li, Mingze Li, Rui Men, Shuang Luo, Tianyi Li, Wenbiao Yin, Xingzhang Ren, Xinyu Wang, Xinyu Zhang, Xuancheng Ren, Yang Fan, Yang Su, Yinger Wan, Yuqiong Wang, Zekun Zhang, Zeyu Cui, Zihan Zhou, Zihan Qiu, et al. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

[13] Anirudh Malik, M. Sparsh Mehra, and Poojith Devan. Capability-stratified degradation in ternary language models, 2026. URL https://arxiv.org/abs/2608.28809.