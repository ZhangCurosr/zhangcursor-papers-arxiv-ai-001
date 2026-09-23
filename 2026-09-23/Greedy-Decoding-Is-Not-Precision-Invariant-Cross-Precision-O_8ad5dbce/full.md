# Greedy Decoding Is Not Precision-Invariant: Cross-Precision Output Divergence in LLM Inference<sup>∗</sup>

Gaoyuan Du gaoyuan.du@utk.edu University of Tennessee, Knoxville

Anam Nawaz Khan akhan59@utk.edu University of Tennessee, Knoxville

Rex Zhou rexzhe1230@gmail.com University of Chicago

Xiaoyang Liu lxaoya@amazon.com Amazon

Deepayan Chakrabarti deepay@utexas.edu University of Texas at Austin

Fnu Suya fsuya@utk.edu University of Tennessee, Knoxville

Xueping Li Xueping.Li@utk.edu University of Tennessee, Knoxville

Reviewed on OpenReview: https: // openreview. net/ forum? id= QDOKyg7a5e

## Abstract

Greedy decoding from large language models is commonly treated as deterministic. We show it is not precision-invariant: the same model, prompt, and decoding algorithm produce diferent outputs in BF16 versus FP16 on identical hardware. Across our evaluations of six models (1.1B–7B parameters, four families; divergence additionally characterised at 12B) and three benchmarks, 49–100% of prompts diverge; a single token flip often cascades into trajectory-level divergence. We develop an empirical error-propagation analysis and find that 22 layers of accumulated body error do not distinguish flipping from non-flipping steps; the outcome depends primarily on the top-two logit margin at the lm\_head relative to the directional perturbation between the top-two candidates. The analysis makes five testable predictions about intervention outcomes, including that applying more FP32 compute (broader scope) makes agreement worse. The experiments match all five predictions. The best-performing low-overhead intervention we evaluate, selective FP32 lm\_head recomputation, triggered only when the margin falls below a threshold, delivers +22–36pp exact agreement on A10G (+12–21pp on L4 and A100) at $< 4 \%$ latency overhead in lowbatch (batch size ≤ 4) single-stream inference. We map the applicability boundary across six models and four batch sizes, and hypothesise that training-time precision stability is a determining factor. The method is a partial mitigation rather than a universal determinism guarantee: its benefit vanishes when body-originated error dominates, including at batch size $\geq 8$ and under end-to-end FP8 in our tests.

## 1 Introduction

Precision switching in LLM serving is a routine operational reality, not a hypothetical concern. Model hubs distribute a single checkpoint that serving frameworks cast to the deployment format; the same model may be served in BF16 on one replica and FP16 on another depending on library defaults, configuration flags, or operator choices. The same model, prompt, and greedy-decoding configuration can therefore produce diferent outputs depending on which numerical format is active. Under our controlled setup, standard reproducibility tooling (torch.use\_deterministic\_algorithms, fixed CUBLAS workspace) makes repeated runs bit-identical within each precision but does not reconcile outputs across precisions. A system can thus be repeatable within each format while failing cross-format replay. In principle, greedy decoding is deterministic: at each step t, the model selects $y _ { t } = \arg \operatorname* { m a x } _ { v } z _ { t } ( v )$

Why does cross-format agreement matter? First, reproducibility and auditing: when exact replay is required, an audit replica running a diferent numerical format may fail to reproduce the logged token sequence. Second, evaluation validity: benchmark scores and A/B tests can conflate model quality with format-induced output drift; two evaluations of the same checkpoint can disagree on individual predictions while reporting similar aggregate accuracy. Third, fidelity to a higher-precision reference: we show (Appendix XIII) that our intervention moves the BF16 arm closer to FP32 greedy inference (+19–21pp sequence agreement). FP16 alone remains closer to FP32 than BF16+C is, so where the serving format is a free choice, FP16 is the better single-format proxy; Intervention C addresses the case where the format is not a free choice. The relevant pair of formats will change as hardware evolves, but the mechanism—a low top-two margin meeting format-dependent rounding at the lm\_head—can be tested across formats. With FP8 confined to the head projection, a format-rescaled gate achieves exact agreement on TinyLlama and +56pp on Qwen2.5-3B. This controlled result is not representative of end-to-end FP8: when every linear layer is quantised, divergence becomes body-dominated and the head-scope repair recovers only +1pp (Appendix XV).

Running the same model with the same prompt under BF16 vs. FP16 produces diferent outputs on 59–70% of TinyLlama-1.1B prompts across public benchmarks (GSM8K, HumanEval, MBPP). The rate rises to 82% on Qwen2.5-3B-Instruct and reaches 100% on the tested BF16-saturated Qwen variants. The crossmodel intervention table covers six models from 1.1B to 7B across four families (Llama, Qwen, Mistral, and OLMoE), and a separate 12.2B experiment characterises divergence without running the full intervention comparison.<sup>1</sup>

The mechanism is more specific: hidden-state error accumulates uniformly through the transformer body (present in every prompt regardless of outcome), and a token flips when some competitor’s diferential perturbation exceeds its original logit gap; empirically, such flips concentrate at steps with a small top-two margin at the lm\_head (Section 7.5). This observation motivates the remedy: selectively recompute the lm\_head in FP32 at steps where the margin falls below a gating threshold τ .

This paper makes four contributions:

1. A systematic characterisation of cross-precision output divergence. We measure the phenomenon primarily at the output level across six models (1.1–7B, four families), four benchmarks, and three GPU microarchitectures: 49–100% of greedy generations difer between BF16 and FP16, with a mean length diference of 34 tokens, and on Qwen2.5-3B GSM8K 19% of prompts flip final-answer correctness. Numerical diferences across formats are well known; our measurements quantify their output-level extent and show how one token flip can cascade through an autoregressive trajectory.

2. A predictive mechanism for cross-precision greedy divergence. A token flips between BF16 and FP16 when some competitor’s diferential perturbation exceeds its original logit gap. Empirically, flips concentrate where the top-two margin is small relative to the per-logit perturbation scale $( \sigma _ { z } \approx 0 . 0 2 6 )$ ). Four independent experiments support this characterisation—divergence-conditioned tracing, layer grafting, KV-cache grafting, and logit-lens traceback—all converging on the lm\_head as the dominant amplification stage. The directional projection of body error onto the top-two decision direction supports lm\_head dominance (Section XII). The mechanism replicates across three tasks and two model families.

3. Five controlled interventions test the mechanism’s predictions. The mechanism predicts each intervention’s outcome before the experiment is run: integer quantization and temperature sharpening have zero efect; top-K FP32 recomputation works for all $K \geq 2 ;$ full FP32 lm\_head recomputation matches top-K; and extending FP32 scope beyond the lm\_head degrades agreement (63%→44%). All five predictions are confirmed.

4. An applicability map across models, kernels, and batch sizes. We evaluate the intervention on six models (1.1–7B, four families), additionally characterise divergence at 12B, test two attention kernels, and vary batch size (bs) over {1, 2, 4, 8} (with composition experiments extending to bs= 16). The lift ranges from $+ 3 6 \mathrm { p p }$ (Llama-3.2-3B) to 0pp (Qwen BF16-saturated); efectiveness appears to track training-time precision stability (hypothesis). Two boundaries emerge: the remedy survives FlashAttention-2 in the tested setting, but the lm\_head-only scope vanishes at ${ \mathrm { b s } } { \geq } 8 . ^ { 2 }$

(a) Divergence is a margin event  
![](images/d885d37c153371f6d9d7e92f87134cd4d214efb1a2ca79bb11faa3fcb09c869b.jpg)

(b) One token flip cascades  
![](images/450f928b4856acaa5e6904fc5a8caf2b642c94f0bfe49a35e2111bf35d99223f.jpg)  
Figure 1: Two empirical facts that shape this paper (TinyLlama-1.1B, GSM8K, BF16 vs. FP16 greedy decoding, $n = 1 0 0 )$ . (a) Distribution of the top-two logit margin at the measurement step $t ^ { * }$ , for prompts whose outputs diverge (red) and for matched agreed controls (green). The two groups are bimodal across more than five orders of magnitude. Diverged median ≈ $1 0 ^ { - 5 }$ , with 30/59 exactly at 0 (perfect BF16 ties); Agreed median ≈ 5.9. Divergence is a margin event at the lm\_head, not an error-magnitude event in the transformer body. (b) Sequence-length scatter of BF16 vs. FP16 outputs on the 59 Diverged prompts. A single token flip typically cascades into trajectory-level divergence: 24/59 prompts end up with diferent lengths, with a mean length diference of 34 tokens and maximum diference of 236 tokens.

## 2 Related work

As machine learning moves into audited deployments, reproducibility can be a deployment requirement in its own right, not merely a byproduct of accuracy. Floating-point non-determinism in training is classical (Goldberg, 1991; Higham, 2002; Zhuang et al., 2022). Framework-level determinism tools (PyTorch Team, 2020) are efective within a fixed format but have no efect on cross-precision divergence (Section 6.1).

BF16/FP16 serving is standard (Dettmers et al., 2022; Micikevicius et al., 2018; Kalamkar et al., 2019), but its output-level consequences and targeted mitigation remain incompletely characterised.

The closest prior work is LayerCast (Yuan et al., 2025), which stores weights in BF16 and casts weightrelevant computation to FP32 layer by layer to reduce system-level numerical variation. LayerCast also observes a connection between small top-token gaps and numerical output changes; we build on that observation by quantifying the margin’s role through grafting and gating experiments, localising the dominant correctable component to the lm\_head, and exploiting it for a selective head-scoped intervention. We do not implement LayerCast. Our separate global FP32-compute baseline upcasts the BF16/FP16-stored model before inference and adds +38% latency on A10G, whereas the selective head-scoped intervention adds $< 4 \%$ PrecisionDif (Wang et al., 2026) also studies BF16/FP16 output disagreements, systematically identifying where precision sensitivity appears across model components. PrecisionDif characterises the sensitivity surface, while we analyse the decision-level mechanism, quantify each stage’s contribution via grafting interventions, and derive a selective repair guided by an exact directional flip condition. Concurrent work (He et al., 2025; Atil et al., 2024; Qi et al., 2025; Gond et al., 2026; Fu et al., 2026) targets kernel- or batch-level sources orthogonal to format-level divergence, or characterises nondeterminism without localising its mechanism. Messina & Scotta (2026) formalise “background temperature” to characterise the efective stochasticity of floating-point perturbations at $T { = } 0 ;$ our work complements that framework with a mechanistic diagnosis and a targeted fix. To our knowledge, prior work has not combined output-level characterisation, stage-wise causal interventions, and selective lm\_head recomputation. Low-precision formats are known to produce numerical diferences; we study their extent at the output level (49–100% of greedy generations diverge in our evaluations), their mechanism (late low-margin events with an lm\_head-dominated correctable component), and their partial controllability (a margin-gated intervention recovering 22–36pp of exact agreement on A10G). Prior work on numerical (non)determinism also addresses within-format efects from batching and kernel scheduling (He et al., 2025; Atil et al., 2024), which are distinct from the cross-format comparison studied here.

## 3 Problem formulation

Consider an autoregressive LM with vocabulary V. At step $t ,$ the model computes logits $z _ { t } = f _ { \theta } ( x , y _ { < t } )$ and selects $y _ { t } = \arg \operatorname* { m a x } _ { v } z _ { t } ( v )$ (Vaswani et al., 2017). In floating-point arithmetic, $z _ { t }$ depends on the numerical precision (BF16, FP16, or FP32), tensor-parallelism degree, and GPU architecture.

We define the top-two logit gap $\Delta _ { t } ^ { ( c ) } = z _ { t } ^ { ( c ) } ( v ^ { ( 1 ) } ) - z _ { t } ^ { ( c ) } ( v ^ { ( 2 ) } )$ under configuration c. When this gap is small enough that floating-point rounding across configurations can flip the argmax, diferent configurations select diferent tokens. The autoregressive conditioning then diverges, producing a cascade of length $L _ { \mathrm { c a s c a d e } } =$ $T - t _ { \mathrm { d i v } }$

We measure four quantities. The Exact Agreement Rate (EAR) is the fraction of prompts producing identical token sequences across configurations. The First Divergence Step (t<sup>∗</sup>) is the position of the first token mismatch. The Trigger Rate is the fraction of steps where $\Delta _ { t } < \tau$ (used only when a selective intervention applies). The Mean Gap (∆<sup>¯</sup> ) averages $\Delta _ { t }$ over all decoding steps. For tasks with well-defined final answers, we additionally report Semantic Agreement Rate (SAR)—whether the extracted conclusion matches across configurations even if the full reasoning difers.

## 4 Inference-time interventions for cross-precision agreement

The contribution of this paper is a mechanistic diagnosis of cross-precision divergence—it is a top-two margin event localised at the lm\_head—together with a controlled ablation that rules out alternative explanations. We study three inference-time interventions within this ablation: each one modifies only the final decoding step (without retraining, modifying the transformer body, or adding persistent GPU memory) but tests a diferent hypothesis about the mechanism. Intervention A asks whether the issue is comparison-level (a tie-breaking artefact between BF16 and FP16 logit values that happen to be representable diferently). Intervention B asks whether a small-K FP32 correction is suficient, and is a direct empirical probe of the rank-1-vs-rank-2 prediction. Intervention C asks whether full FP32 lm\_head recomputation eliminates divergence, and is also the most natural deployable fix. The three interventions provide a sequence of mechanism tests: success of A would support a comparison-level explanation; if A fails but B with $K = 2$ succeeds, this supports a rank-1-vs-rank-2 correction mechanism in the tested gated setting; if B at $K = 2$ matches C at $K = | V |$ , recomputing beyond the top two provides no additional benefit in this experiment, supporting an lm\_head-localised correction with a 4000× smaller intervention. Figure 2 illustrates the shared selective-gating envelope; the three interventions below difer only in what they manipulate inside the gated step.

![](images/b5414443cdaee1f42b9e3ee7bfa878d8cfce86d81d51c789902383331a6635ff.jpg)  
Figure 2: Mechanism and method. (A) Cross-precision divergence is a margin event: BF16 and FP16 accumulate uniform background error through the transformer body, and the lm\_head—the dominant amplification stage—turns that error into a token flip only when the top-two logit margin is small. Prompts with a large margin agree across precisions; prompts with a small margin diverge. (B) Our method gates on this margin: the native (BF16/FP16) lm\_head logits $z _ { t }$ are computed first, and only when the margin is small does the gate trigger a selective FP32 lm\_head recomputation $( \mathrm { f p 3 2 } ( W _ { \mathrm { l m } } ) { \cdot } \mathrm { f p 3 2 } ( h _ { t } ) )$ ; large-margin (safe) steps take the standard argmax unchanged. Selective repair (∼1.4% of steps) leaves safe steps untouched, avoiding the butterfly efect of ungated recomputation.

Predictions from the analysis. An error-propagation analysis (Section 7, $\sigma _ { z } \approx 0 . 0 2 6 )$ predicts: A has zero efect, B and C produce comparable lifts (rank-1-vs-rank-2 only), and extending scope beyond lm\_head hurts. All predictions confirmed.

Selective triggering. All interventions fire only when $\Delta _ { t } ~ < ~ \tau ~ = ~ 1 0 ^ { - 3 } ~ ( 0 . 7 \mathrm { - } 1 . 4 \%$ of steps), keeping overhead <4%.

## 4.1 Intervention A: Integer logit quantization

The argmax over floating-point logits can difer across precisions because BF16 and FP16 represent the same real number with diferent roundings; integer comparison after quantization is bit-exact. Intervention A exploits this by quantizing logits into coarse bins of width τ before argmax, testing whether divergence is comparison-level:

$$
\tilde { z } _ { t } ( v ) = \mathrm { r o u n d } \left( \frac { z _ { t } ( v ) } { \tau } \right) , \quad y _ { t } = \arg \operatorname* { m a x } _ { v } \tilde { z } _ { t } ( v ) .\tag{1}
$$

Table 1: Inference-time interventions for cross-precision agreement. Cost is per-trigger overhead relative to a standard decoding step.
<table><tr><td>Intervention</td><td>Approach</td><td>Per-Trigger Cost</td><td>Mechanism</td></tr><tr><td>A. Integer quantization</td><td>Reduce precision</td><td> $O ( | V | )$  div+round</td><td>Merge small-gap logits</td></tr><tr><td>B. Top-K FP32 + int compare</td><td>Increase then reduce</td><td> $O ( K \cdot d )$  FP32 matmul</td><td>FP32 recompute + int merge</td></tr><tr><td>C. Full-vocab FP32 lm_head</td><td>Increase precision</td><td> $O ( d \cdot | V | )$  FP32 matmul</td><td>FP32 recompute of lm_head</td></tr></table>

Any two logit values within $\tau$ of each other map to the same integer. Ties in the integer domain are broken by selecting the smaller token index, ensuring a canonical, platform-independent decision. This is a lossy approach that deliberately discards sub-τ diferences. The cost is negligible: one division and one round per vocabulary element.

## 4.2 Intervention B: Top-K FP32 recomputation with integer comparison

Intervention A assumes that cross-precision logits are close enough to fall within the same quantization bin. When they are not—i.e., the logit values themselves difer across precisions—the quantization step alone cannot reconcile them. Intervention B addresses this by first correcting the logit values through FP32 recomputation of the top-K candidates, then applying integer quantization for a deterministic final comparison:

1. Extract the top-K candidate indices $\mathcal { K } = \mathrm { t o p } \mathrm { - } K ( z _ { t } )$ from the native-precision logits.

2. Recompute the lm\_head in FP32 for only those K rows: $\hat { z } _ { t } ( \mathcal { K } ) = \mathrm { f l o a t 3 2 } ( W _ { \mathrm { l m } } [ \mathcal { K } ] ) \cdot \mathrm { f l o a t 3 2 } ( h _ { t } ^ { ( L ) } )$ where $\bar { W } _ { \mathrm { l m } } [ \mathcal { K } ] \in \mathbb { R } ^ { K \times d }$ is the sub-matrix of lm\_head weights corresponding to the K candidates.

3. Quantize: $\tilde { z } _ { t } ( v ) = \mathrm { r o u n d } ( \hat { z } _ { t } ( v ) / \tau )$ for $v \in \kappa$ , then integer argmax with canonical tie-break (smallest token index wins).

This is a hybrid approach that combines value correction (partial FP32 recompute) with comparison stabilization (integer binning). The cost per trigger is one partial FP32 matrix multiply of size $K \times d \ ( K = 8$ $d = 2 0 4 8 \colon 1 6 \mathrm { K \ F L O P s } )$ , plus negligible quantization overhead—4000× cheaper than Intervention C’s full $| V | \times d$ recompute per trigger. We use $K = 8$ as a conservative default; Appendix IX shows that $K = 2$ sufices (all values $K \in \{ 2 , 4 , 8 , 1 6 , 3 2 , | V | \}$ produce identical EAR), showing that, for the tested gated cases, correcting the top-two candidates is suficient to match the full-vocabulary version.

## 4.3 Intervention C: Full-vocabulary FP32 lm\_head recomputation

The dominant source of cross-precision logit perturbation is the lm\_head: a single matrix multiplication $z _ { t } = W _ { \mathrm { l m } } \cdot h _ { t } ^ { ( L ) }$ projecting the d-dimensional hidden state onto the full vocabulary |V|. Because |V| is large (32K–152K), this operation amplifies small hidden-state diferences by a factor of $\Vert W _ { \mathrm { l m } } \Vert$ . Intervention C eliminates this amplification by recomputing the projection in FP32 over the full vocabulary:

$$
\hat { z } _ { t } = \mathrm { f l o a t 3 2 } ( W _ { \mathrm { l m } } ) \cdot \mathrm { f l o a t 3 2 } ( h _ { t } ^ { ( L ) } ) ,\tag{2}
$$

where both activations and weights are up-cast to FP32 on-the-fly, keeping GPU memory constant (no persistent FP32 weight copies). The recomputation scope can in principle extend to the last L transformer layers, trading additional compute for deeper error correction, though our scope ablation (Section 7.6) shows this degrades agreement in practice. This is the most expensive intervention per trigger but directly targets the dominant error source identified by our layer-level analysis (Section 7.5). The cost is one full FP32 matrix multiply over the entire vocabulary $( d \times | V | )$ .

## 4.4 Overhead and trade-of analysis

Table 1 summarizes the three interventions.

With empirical trigger rates of 0.7–1.4%, all interventions add ${ < } 4 \%$ latency under selective gating. The key methodological question is whether cross-precision divergence is caused by small logit diferences (within $\tau ,$

Algorithm 1 Inference-time intervention for cross-precision agreement   
Require: Model $f _ { \theta } ,$ prompt x, threshold τ , max tokens T, intervention ∈ {A, B, C}   
1: for $t = 1$ to T do   
2: $z _ { t } \gets f _ { \theta } ( x , y _ { < t } ) ; \quad v ^ { ( 1 ) } , v ^ { ( 2 ) } \gets \mathrm { t o p } { - 2 ( z _ { t } ) } ; \quad \Delta _ { t } \gets z _ { t } ( v ^ { ( 1 ) } ) - z _ { t } ( v ^ { ( 2 ) } )$   
3: if $\Delta _ { t } < \tau$ or $\tau = 0$ then   
4: $y _ { t } \gets \mathrm { A p p l y } ( f _ { \theta } , x , y _ { < t } , z _ { t } , \mathrm { A } / \mathrm { B } / \mathrm { C } )$   
5: else   
6: $y _ { t } \gets v ^ { ( 1 ) }$   
7: end if   
8: if $y _ { t } = < \mathsf { e o s } >$ then   
9: break   
10: end if   
11: end for

addressable by integer quantization) or large diferences (ranking reversals, requiring FP32 recomputation).   
Our experiments answer this definitively.

We additionally evaluate two supplementary baselines in Appendix IX: (D) Temperature sharpening (multiply logits by α>1 before argmax, zero cost) and (E) Consensus decoding (run BF16 and FP16 in parallel, FP32 tiebreak on disagreement, $\mathrm { \sim } 2 \times \mathrm { c o s t } )$ . Neither changes the cost–agreement frontier established by A–C: sharpening has zero efect (confirming the value-level nature of divergence), and consensus achieves perfect agreement only at double the compute budget.

## 5 Experimental setup

Public benchmarks. Our primary evaluation uses three public benchmarks: GSM8K (Cobbe et al., 2021) (math, 8-shot CoT), HumanEval (Chen et al., 2021) (code, 0-shot), and MBPP (Austin et al., 2021) (code, 3-shot).

Models. Our primary experiments (Sections 6.2–6.3) use TinyLlama-1.1B-Chat (Zhang et al., 2024) as the main testbed due to its fast iteration time and ability to run full ablations on a single GPU. For model sensitivity analysis (Section 6.5), we compare reasoning-tuned models (DeepSeek-R1-Distill-Qwen-1.5B/7B (DeepSeek-AI, 2025)) against instruction-tuned models (Qwen2.5-1.5B/7B-Instruct (Qwen Team, 2024)).

Configuration. We compare BF16 vs. FP16 greedy decoding on the same GPU (A10G 22GB). Max generation: 256 tokens. Fixed seeds throughout. To isolate numerical-format efects from PyTorch-level kernel non-determinism, we also run a sanity-check ablation (Section 6.1) that varies torch.use\_deterministic\_algorithms across {False, True} with CUBLAS\_WORKSPACE\_CONFIG=:4096:8 and cudnn.benchmark=False, confirming that under greedy decoding with batch size 1 and fixed seed, PyTorch kernels are already self-deterministic within a single precision and the flag has no efect on cross precision divergence.

Why BF16 vs. FP16. We focus on BF16 vs. FP16 because both are common 16-bit inference formats on modern GPUs (Dettmers et al., 2022; Kwon et al., 2023; NVIDIA, 2023), yet they allocate exponent and mantissa bits diferently (8/7 versus 5/10). This pair provides a controlled test of representation-dependent output divergence. We directly test transfer to head-isolated FP8/BF16 and to FP16/FP32 on T4; applying the methodology to other format pairs remains future work.

Baselines. Greedy BF16/FP16 (standard serving precision), Greedy FP32 (oracle upper bound— eliminates all precision-induced divergence), global FP32 compute (upcast the BF16/FP16-stored model before inference), and ungated FP32 recomputation (Intervention C with τ = 0, recomputing every step without selective gating).

Table 2: BF16 vs. FP16 divergence spectrum (TinyLlama-1.1B, n = 100). Exact agreement (EAR) ranges from 30% to 41%. 95% CI: Wilson score interval.
<table><tr><td>Dataset</td><td>Domain</td><td>EAR (%)</td><td>95% CI</td></tr><tr><td>MBPP</td><td>Code</td><td>30.0</td><td>[21.9, 39.6]</td></tr><tr><td>HumanEval</td><td>Code</td><td>36.0</td><td>[27.3, 45.8]</td></tr><tr><td>GSM8K</td><td>Math</td><td>41.0</td><td>[31.9, 50.8]</td></tr></table>

## 6 Results

Unless otherwise noted, all results in this section use TinyLlama-1.1B-Chat with BF16 vs. FP16 greedy decoding (n = 100 per dataset). Additional results with DeepSeek-R1 and Qwen2.5 models appear in Section 6.5.

## 6.1 Sanity check: kernel-level determinism is not the cause

Before attributing divergence to numerical format, we rule out two alternative explanations:

(1) PyTorch kernel non-determinism. Toggling torch.use\_deterministic\_algorithms, CUBLAS\_WORKSPACE\_CONFIG, and cudnn.benchmark between “loose” and “strict” settings in independent subprocesses produces bit-identical outputs within each precision: BF16 self-consistency is 100% and the 41% BF16-vs-FP16 EAR persists identically across both arms. The pattern generalises to MoE (OLMoE-1B-7B) and batch sizes {1, 4, 8, 16, 32} (Appendix VIII).

(2) GEMM kernel selection diference. cuBLAS may select diferent GEMM algorithms for BF16 vs. FP16 matmuls. Two observations rule this out. First, BF16 self-consistency is 100%, so kernel choice is deterministic within a dtype. Second, running the same BF16 weights under native BF16 vs. FP32 compute (diferent kernels, same weights) yields 42% EAR, indistinguishable from the 41% baseline where both dtype and kernel difer. The divergence is caused by BF16’s arithmetic rounding, not by kernel dispatch.

## 6.2 Divergence is substantial across tasks

Table 2 summarizes BF16-vs-FP16 divergence across all datasets.

On code generation tasks, 64–70% of prompts produce diferent outputs between BF16 and FP16. On math reasoning, 59% diverge.

Where does divergence happen? The first divergence step t<sup>∗</sup> has a median of 34–50 tokens on non-Qwen models (the divergence-step analysis), with 44% diverging in the first 30 tokens and a long tail past token 200. Qwen2.5-7B-Instruct is catastrophically early-biased (median t<sup>∗</sup> = 1, 100% first-token divergence), consistent with training-time BF16 saturation (Section 6.5). The left-skewed distribution motivates per-step gating: most divergences are localised to a small fraction of steps identifiable by the top-two margin.

## 6.3 Intervention comparison: a controlled ablation

Table 3 presents the central experiment: all three interventions and baselines on BF16 vs. FP16 $( \tau = 1 0 ^ { - 3 } )$ .

Five findings emerge, building from root cause to practical recommendation:

(1) The FP32 oracle confirms the root cause. Running both configurations in FP32 yields 100% agreement: all divergence originates from precision diferences, not from non-determinism in the decoding algorithm.

(2) Global FP32 compute achieves high agreement but at high cost. Global FP32 upcasting reaches 88–89% agreement—the highest among all methods short of the FP32 oracle—but doubles memory footprint, preventing deployment on the same hardware for larger models. Gated FP32 lm\_head recomputation (Interventions $\mathrm { B / C ) }$ achieves lower absolute agreement (55–63%) but with <4% overhead, making it practical for production serving.

Table 3: Intervention and baseline comparison: BF16 vs. FP16 exact agreement (%) (TinyLlama-1.1B, $n = 1 0 0 , \tau = 1 0 ^ { - 3 } )$ Gated FP32 lm\_head recomputation (B, C) achieves the best eficiency–agreement trade-of among the tested methods. All per-cell 95% Wilson CIs have half-width ±9–10pp at n = 100; because Baseline and Intervention C are evaluated on the same prompts, the relevant tests are the matchedpair McNemar $( p < 0 . 0 0 1 )$ and the paired bootstrap CI on the lift ([+11, +33]pp, Appendix X) rather than the per-cell intervals. Full CIs for each cell are reported in Appendix X. Appendix X replicates the Baseline / Intervention C rows at $n = 3 0 0$ (GSM8K), full-benchmark HumanEval $( n = 1 6 4 )$ , and full-split MBPP $( n = 2 5 7 )$ to confirm the lift survives at larger sample sizes (all n = 100 cells sit inside the scaled CIs). Appendix X additionally reports 10 000-sample bootstrap confidence intervals for every headline lift in the paper; the TinyLlama +22pp EAR gain has a 90% bootstrap CI of [+11, +33]pp.
<table><tr><td>Method</td><td>GSM8K</td><td>HumanEval</td><td>MBPP</td><td>Overhead</td></tr><tr><td>Baseline (no intervention)</td><td>41.0</td><td>36.0</td><td>30.0</td><td></td></tr><tr><td>Baselines</td><td></td><td></td><td></td><td></td></tr><tr><td>Greedy FP32 (oracle)</td><td>100.0</td><td>100.0</td><td>100.0</td><td>~2× mem</td></tr><tr><td>Global FP32 compute</td><td>88.0</td><td>88.0</td><td>89.0</td><td>~2× mem</td></tr><tr><td>Ungated FP32 recompute (τ=0)</td><td>60.0</td><td>56.0</td><td>54.0</td><td> ${ \sim } 2 . 5 \times$ </td></tr><tr><td>Gated interventions  $( \tau = 1 0 ^ { - 3 } )$ </td><td></td><td></td><td></td><td></td></tr><tr><td>A. Integer quantization</td><td>41.0</td><td>36.0</td><td>30.0</td><td>&lt;1%</td></tr><tr><td>B. Top-K FP32 + integer compare</td><td>63.0</td><td>61.0</td><td>55.0</td><td>&lt;2%</td></tr><tr><td>C. Full-vocab FP32 lm_head</td><td>63.0</td><td>61.0</td><td>55.0</td><td>&lt;4%</td></tr></table>

(3) Ungated recomputation (τ=0) underperforms gating. Recomputing at every step achieves only 54–60% agreement at ∼2.5× latency, worse than gated Interventions B/C (55–63% at <4% overhead). Global recomputation changes the logit landscape at safe steps, introducing new divergence points downstream (Section 7).

(4) Integer quantization has zero efect. Intervention A produces identical agreement to the baseline across all datasets, confirming that cross-precision divergence is caused by value-level logit diferences (BF16 and FP16 compute diferent numbers), not comparison-level ties (same numbers compared diferently by argmax).

(5) Gated FP32 recomputation is the best trade-of. Interventions B and C achieve identical improvements (+22pp on GSM8K, +25pp on HumanEval and MBPP) with <4% overhead, outperforming all baselines on the eficiency–agreement frontier. Gated FP32 lm\_head recomputation also raises SAR from 58% to 72% on GSM8K. The <4% overhead is backed by direct measurement on a single A10G (Appendix VIII). Gated C adds +1.4% latency and +11% peak memory versus the BF16 baseline; the global FP32-compute baseline adds +38% latency under the same measurement.

Where does the +22pp come from? Table 4 stratifies the 100 GSM8K prompts into four mutually exclusive categories based on whether gated C triggers and whether the flip reproduces on single-step re-run. Gated C fires on exactly 30 prompts (those with $\Delta _ { t } < \tau = 1 0 ^ { - 3 }$ at the lm\_head), and empirically fixes 22 of them (73% success rate on triggered prompts). The 29 prompts in the “margin $\geq \tau ^ { \ast }$ category are where the flip is driven by accumulated upstream error—gated C correctly declines to trigger there, because a single-step FP32 recomputation cannot reconcile two hidden states that have already drifted apart. This stratification predicts the exact headline lift $( 4 1 \%  6 3 \% )$ without tuning and explains both where the method applies and where it does not.

Robustness checks. We characterise divergence across six models spanning four families (Llama, Qwen, Mistral, and OLMoE-MoE; 1.1B–7B), with an additional divergence-only probe at 12B. Intervention lifts are +36pp on Llama-3.2-3B, +8–10pp on Mistral-7B and OLMoE, +3pp on Qwen2.5-3B, and 0pp on the BF16- saturated Qwen variants. Efectiveness appears to track training-time precision stability (hypothesis; not directly validated), rather than scale alone. The phenomenon and intervention persist with FlashAttention-2 and through 1024-token reasoning chains on MATH-500. We find no statistically significant pass@1 degradation in the matched-pair tests; a TOST analysis with margin $\delta = 0 . 0 5$ establishes non-inferiority on Qwen2.5-3B at $n = 6 0 0 \ ( p = 0 . 0 0 9 )$ , while the TinyLlama result is a low-accuracy consistency check rather than informative quality evidence. C’s lift degrades from +17pp at bs=1 to 0pp at bs=8; composition with global FP32 compute recovers +5pp at bs=8. Digit tokens are 8× over-represented among flipped tokens.

Table 4: Per-prompt stratification of when gated Intervention C applies (TinyLlama-1.1B, GSM8K, $n { = } 1 0 0$ $\tau = 1 0 ^ { - 3 } )$ Categories are mutually exclusive. The main-text +22pp EAR improvement (Table 3) arises because C fires on 30 prompts (B+C) and successfully converts 22 of them $( 2 2 / 3 0 \approx 7 3 \%$ of triggered prompts). Category D (29 prompts) is where the flip is driven by large-error accumulation and the lm\_head margin is already decisive at t<sup>∗</sup>; gated C correctly does not trigger there.
<table><tr><td rowspan="2">Prompt category</td><td rowspan="2">Count</td><td colspan="5">Distribution by t* (first divergence step)</td></tr><tr><td>t*=0</td><td>1-10</td><td>11-50</td><td>51-100</td><td>&gt;100</td></tr><tr><td>AGREED (no intervention needed)</td><td>41</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Gated, single-step flip reproducible (C most effective)</td><td>25</td><td>4</td><td>4</td><td>10</td><td>5</td><td>2</td></tr><tr><td>Gated, flip KV-cache-dependent (C may not fix)</td><td>5</td><td></td><td>2</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Margin  $\geq \tau$  (C does not trigger)</td><td>29</td><td></td><td>4</td><td>8</td><td>9</td><td>8</td></tr></table>

Table 5: TOST non-inferiority analysis (GSM8K, $\delta = 0 . 0 5$ , n per row; bare “Solve step by step” prompt format, hence absolute accuracies are lower than the chat-template numbers of Table 15 and are comparable only within this table). We test $H _ { 0 } \colon$ IntC accuracy ≤ Baseline accuracy − δ. Rejection $\left( p < 0 . 0 5 \right)$ establishes that IntC is no worse than 5pp below baseline. Interpretation note: TinyLlama’s baseline accuracy (2.3%) is itself below the $\delta = 0 . 0 5$ margin, so a 5pp shortfall is arithmetically impossible there and its rejection should be read as a consistency check rather than as evidence; Qwen2.5-3B (11.7%) is the informative row. Paired McNemar on the same Qwen run is non-significant $( \chi ^ { 2 } = 0 . 1 4$ $p = 0 . 7 0 8$ ; b = 34 discordant pairs favouring baseline, $c = 3 0$ favouring Intervention C). Qwen2.5-3B was re-run at n=600 to raise statistical power: non-inferiority is established $( p = 0 . 0 0 9 )$ , with the accuracy diference’s 95% CI [−4.3, +2.9]pp lying entirely inside the ±5pp margin.
<table><tr><td>Model</td><td>n</td><td>Acc (Baseline)</td><td>Acc (IntC)</td><td>Diff</td><td>TOST p</td><td>Non-inferior?</td></tr><tr><td>TinyLlama-1.1B</td><td>300</td><td>2.3%</td><td>2.0%</td><td>−0.3pp</td><td> $4 . 3 \times 1 0 ^ { - 5 }$ </td><td>Yes</td></tr><tr><td>Qwen2.5-3B</td><td>600</td><td>11.7%</td><td>11.0%</td><td>−0.7pp</td><td>0.009</td><td>Yes</td></tr></table>

Counter-intuitive predictions confirmed by the mechanism. Body-layer error does not discriminate between flipping and non-flipping prompts: the 22-layer TinyLlama and the 36-layer Qwen2.5-3B have high divergence rates (59% and 82%) despite diferent body-error budgets. In these measurements, the lm\_head margin distribution is more predictive of a flip than layer count or body-error magnitude.

## 6.4 Semantic agreement and gating sensitivity

Two findings extend the headline picture along orthogonal axes (full data in appendix).

Semantic exceeds exact agreement. On GSM8K, 58% of prompts produce the same final numerical answer despite diferent reasoning chains (SAR), compared to 41% exact token match (EAR); Intervention C lifts SAR to 72%. Token diferences also need not imply functional diferences on code: on Qwen2.5-3B HumanEval, execution outcomes agree on 97.6% of problems while full token sequences agree on only 26.2% (Appendix XVI). Exact agreement therefore measures a stricter property than task-outcome agreement.

Downstream evaluation impact. Cross-precision divergence can be hidden by aggregate benchmark accuracy. On Qwen2.5-3B-Instruct (GSM8K, n = 100), 19% of prompts flip correctness between BF16 and FP16: 9 prompts are correct only in BF16 and 10 only in FP16, while aggregate accuracy difers by only 1pp (27% vs. 28%). On Llama-3.2-3B-Instruct with the native chat template (66% accuracy), correctness flips drop to 2%. These two cases show that similar aggregate scores can mask diferent per-prompt outcomes; determining how this efect scales with model accuracy requires broader evaluation.

Selective triggering is essential. On DeepSeek-R1-1.5B (n = 200), gated $\mathrm { ~ C ~ } ( \tau \in [ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ] )$ achieves +6.5pp with ${ < } 4 \%$ overhead, while ungated recomputation $( \tau = 0 )$ achieves only +2.5pp with 2.5× latency— the “butterfly efect” makes blanket recomputation counterproductive (mechanism explained in Section 7). The threshold is insensitive across two orders of magnitude, consistent with the bimodal margin distribution.

## 6.5 Model sensitivity: reasoning vs. instruction tuning

At matched parameter scales, reasoning-tuned models show 10–15pp lower exact agreement and $1 . 7 \times$ higher trigger rates than instruction-tuned counterparts (DeepSeek-R1 vs. Qwen2.5; Appendix IV). Chain-ofthought reasoning produces more uncertain intermediate steps, increasing exposure to low-margin events. Longer CoT trajectories accumulate more close-to-tipping-point decoding decisions, each precision-sensitive. This matches the scale analysis accompanying Proposition 1: the divergence rate scales with the lower-tail mass of the margin distribution, and CoT concentrates more of this distribution at small values.

## 7 Mechanism: an empirical analysis of error propagation

We separate the empirical observation from the analysis. The observation is that divergence concentrates at steps with a near-zero top-two margin, while the body’s hidden-state error — which is present in every prompt and grows monotonically through the layers — does not discriminate between flipping and nonflipping steps once projected onto the decision direction. The analysis builds on that observation: an exact algebraic condition for when a flip occurs, and an estimate of when a head-scope repair can undo one. This section provides an empirical analysis rather than formal bounds on floating-point error propagation. We give one exact algebraic condition (which is elementary), then two quantitative estimates whose assumptions we state and which we evaluate against measurements. The model is organised around one exact result and one criterion: Proposition 1 states when a token $\mathrm { { \ f i p s } , }$ and the cross-arm hidden-state gap ∆h (Table 6) states when a head-scope repair can undo it. Between them they cover the four format pairs we measure — BF16/FP16, FP16/FP32, head-isolated FP8, and end-to-end FP8 — including the one in which the intervention fails. Two further failure modes, batched decode and cross-kernel variation, are governed by diferent mechanisms and we treat them separately. We state one exact result (Proposition 1, with proof), develop two quantitative heuristic estimates validated against measurement, and identify the irreducible weight-truncation floor that bounds any inference-time fix. Full derivations are in Appendix I.

## 7.1 Error propagation: from machine epsilon to logit perturbation

Consider a transformer with L layers and hidden dimension $d ,$ producing logits $z _ { t } = W _ { \mathrm { l m } } \cdot h _ { t } ^ { ( L ) }$ where $W _ { \mathrm { l m } } \in \mathbb { R } ^ { | V | \times d } .$ . Under precision $c \in \{ \mathrm { B F 1 6 , F P 1 6 } \}$ , each layer introduces a rounding error bounded by $\| \delta _ { \ell } \| \le \epsilon _ { c } \cdot C _ { \ell } \cdot \| h _ { \ell } \|$ , where $\epsilon _ { \mathrm { B F 1 6 } } \approx 3 . 9 \times 1 0 ^ { - 3 }$ and ϵ<sub>FP16</sub> ≈ $4 . 9 \times 1 0 ^ { - 4 }$ (an $8 \times$ gap due to the 7-bit vs. 10-bit mantissa). With residual connections keeping per-layer Jacobian norms $\| J _ { \ell } \| \approx 1$ , a worst-case (no-cancellation) accumulation gives $\| \hat { h } _ { L } - h _ { L } \| \le L \cdot \epsilon _ { c } \cdot \bar { C } \cdot \| h \|$

Empirically we measure a ${ \sim } 5 \times$ amplification from pre-lm\_head $\ell _ { 2 } \ ( 0 . 9 2 )$ to logit $\ell _ { 2 }$ (4.63) on TinyLlama-1.1B. The per-logit perturbation is $\lVert \hat { z } - z \rVert / \sqrt { | V | } \approx 4 . 6 3 / \sqrt { 3 2 0 0 0 } \approx 0 . 0 2 6$ . This yields the central divergence condition:

Proposition 1 (Exact directional flip condition) Fix a decoding step and let $z , \hat { z } \in \mathbb { R } ^ { | V | }$ be the logit vectors computed by two numerical configurations. Assume each vector has a unique maximiser, let $\boldsymbol { v } ^ { ( 1 ) } =$ $\arg \operatorname* { m a x } _ { v } z ( v )$ , and write $\Delta z ( v ) = \hat { z } ( v ) - z ( v )$ . Then arg max $_ { \cdot v } \hat { z } ( v ) \neq v ^ { ( 1 ) }$ if and only if there exists v ̸= $\boldsymbol { v } ^ { ( 1 ) }$

such that

$$
\Delta z ( v ) - \Delta z ( v ^ { ( 1 ) } ) > z ( v ^ { ( 1 ) } ) - z ( v ) .\tag{3}
$$

For the original runner-up $v ^ { ( 2 ) }$ , this inequality is exactly the condition for $v ^ { ( 2 ) }$ to overtake $\boldsymbol { v } ^ { ( 1 ) }$ . It is necessary for a flip whose new winner is $v ^ { ( 2 ) }$ ; together with $\hat { z } ( v ^ { ( 2 ) } ) > \mathrm { m a x } _ { u \not \in \{ v ^ { ( 1 ) } , v ^ { ( 2 ) } \} } \hat { z } ( u )$ , it is suficient. Exact ties are resolved by the implementation’s deterministic tie-breaking rule.

Proof. Under the uniqueness assumption, arg max<sub>v</sub> $\hat { z } ( v ) \neq v ^ { ( 1 ) }$ if some $v \neq v ^ { ( 1 ) }$ satisfies $\hat { z } ( v ) > \hat { z } ( v ^ { ( 1 ) } )$ i.e. $\dot { z } ( v ) + \Delta z ( v ) > \bar { z ( v ^ { ( 1 ) } ) } + \Delta z ( v ^ { ( 1 ) } )$ , which rearranges to the stated condition. Taking $v = v ^ { ( 2 ) }$ gives the pairwise crossing condition; checking $v ^ { ( 2 ) }$ against all remaining coordinates identifies it as the new winner. □

On the status of this proposition. The proposition is an algebraic rearrangement, stated separately to distinguish the proved condition from the estimates that follow. The two quantitative accounts below— the perturbation scale and the criterion for when a head-scope repair can undo a flip—are estimates with explicitly stated assumptions.

Scale analysis (heuristic, not part of the proposition). The proposition is exact but involves the unobservable per-coordinate perturbations $\Delta z ( v )$ To obtain a practical trigger statistic we approximate their typical magnitude by the vocabulary-wide RMS $\sigma _ { z } = \| \hat { z } - z \| / \sqrt { | V | } \approx 0 . 0 2 6$ (TinyLlama-1.1B), giving the directional diference a scale of $\approx \sqrt { 2 } \sigma _ { z } \approx 0 . 0 3 7$ This motivates the margin-gated trigger of Intervention $\mathrm { C } \mathrm { : }$ flips concentrate where $\Delta _ { t }$ is small relative to this scale. The approximation is deliberately heuristic—per-coordinate perturbations are neither Gaussian nor independent—and we assess its adequacy purely empirically: the directional criterion classifies flip-vs-no-flip correctly on 88–89% of steps across two model families (Section XII), and Diverged prompts show median margin 0.0 vs. 2.69 for Agreed (260× mean gap). An alternative suficient-condition formulation via an $\ell _ { \infty }$ bound is also available—no flip can occur if $2 \| \hat { z } - z \| _ { \infty } < \Delta _ { t }$ , since $| \Delta z ( v ^ { ( 2 ) } ) - \Delta z ( v ^ { ( 1 ) } ) | \leq 2 \| \hat { z } - z \| _ { \infty } -$ and Intervention C’s gate can be read as enforcing this safe condition at a per-format calibrated scale; we use the RMS form in the main text because $\| \hat { z } - z \| _ { \infty }$ is dominated by outlier coordinates that rarely coincide with the top-two candidates, so the $\ell _ { \infty }$ bound is safe but looser than the RMS-scale heuristic as a trigger statistic.

Empirical validation. Body-layer L2 is indistinguishable between Diverged and Agreed groups while the top-two margin separates them by orders of magnitude (Table 7), confirming that the margin—not the error magnitude—determines divergence.

## 7.2 Why FP32 recomputation works

During autoregressive generation with a KV cache, each decoding step processes only the new token through the transformer body; the prefix hidden states are retrieved from cache without recomputation. The fresh rounding error introduced at step t is therefore of order $\epsilon _ { c } \cdot C _ { \mathrm { s t e p } }$ dominated by the single lm\_head matmul, not the full L-layer body.

Why FP32 recomputation reconciles the paths. The starting point is an immediate observation, not a theorem: given the same hidden state $h _ { t } ^ { ( L ) }$ , the deterministic FP32 matmul floa $\mathrm { . 3 2 } ( W _ { \mathrm { l m } } )$ $\mathrm { { f l o a t 3 2 } } ( h _ { t } ^ { ( L ) } )$ produces bit-identical outputs regardless of whether the upstream path was BF16 or FP16. We keep this observation explicit precisely because it is trivial: it is what makes the intervention’s reach predictable. Since the FP32 recomputation is deterministic, the two arms agree at a gated step $i f$ and only if the hidden states they feed it agree. At a gated step, therefore, the intervention’s reach is governed by the cross-arm hidden-state gap $\Delta h$ rather than by any property of the head. We use this as a screening quantity, not as a law: it orders the four format pairs we test into regimes (Table 6), but it does not cover batching or kernel variation, and we have tested no configuration in the wide gap between the working and failing regimes. The substantive, empirically testable claim is quantitative: at gated steps the cross-path logit diference should drop from $\sigma _ { z } \approx 0 . 0 2 6$ to the body-originated residual only, which the error model of Section 7.3 heuristically

estimates as

$$
\sigma _ { z } ^ { \mathrm { ( a f t e r ~ C ) } } \approx \underbrace { \frac { L \cdot \alpha } { \sqrt { d } } \cdot \sigma _ { z } } _ { \mathrm { b o d y ~ r e s i d u a l ~ ( } \sim 2 . 4 \% \mathrm { ~ o f ~ } \sigma _ { z } ) } \approx 6 \times 1 0 ^ { - 4 } .\tag{4}
$$

For prompts whose divergence is entirely lm\_head-originated (body hidden states are efectively identical between precisions), this residual vanishes and C achieves exact reconciliation; the achievable EAR lift then equals the fraction of such prompts. This quantitative account is validated by the reconciliation data: the $+ 2 2 \mathrm { p p }$ improvement matches the fraction of prompts whose divergence originates from single-step margin events—30/100 prompts enter the gate (category B of Table 4), of which 22 are successfully reconciled (73% success rate within scope).

## 7.3 When lm\_head-only scope is suficient, and when it is not

The analysis above shows C works; we now show why the lm\_head is the dominant error source—making body-layer FP32 unnecessary.

Heuristic estimate: why the lm\_head dominates. Under three explicit assumptions—(i) per-layer Jacobian norms ≈ 1 maintained by residual connections, (ii) rounding errors accumulate without systematic cancellation across layers (conservative, worst-case linear in $L )$ , and (iii) the d per-element rounding errors of the lm\_head dot product accumulate as $\sqrt { d }$ (standard random-sign model)—the body-to-lm\_head perturbation ratio is approximately $L \alpha / { \sqrt { d } } ,$ where $\alpha = \| F _ { \ell } ( h ) \| / \| h \| \ll 1$ is the residual ratio:

$$
{ \frac { \sigma _ { z } ^ { ( \mathrm { b o d y } ) } } { \sigma _ { z } ^ { ( { \mathrm { l m } } \_ { \mathrm { h e a d } } ) } } } \approx { \frac { L \cdot \alpha } { \sqrt { d } } } .\tag{5}
$$

Argument. In KV-cached decoding at step $t > 0$ , only the new token passes through L body layers. The residual connection $h ^ { ( \ell ) } = h ^ { ( \ell - 1 ) } + \check { F } _ { \ell } ( h ^ { ( \ell - 1 ) } )$ confines rounding error to the $F _ { \ell }$ branch, whose output norm is $\alpha \cdot \| h \|$ with $\alpha \ll 1$ (a necessary condition for training stability). We bound the body contribution conservatively (worst-case linear accumulation): after L layers the body error is at most $L \cdot { \boldsymbol { \alpha } } \cdot \epsilon _ { c } \cdot \| h \|$ By contrast, the lm\_head computes a d-dimensional dot product on $h _ { t } ^ { ( L ) }$ at its full norm; the d rounding errors accumulate to a typical magnitude $\approx \sqrt { d } \cdot \epsilon _ { c } \cdot | | h | |$ (a random-sign estimate, not a bound; the worst case is $d \cdot \epsilon _ { c } \cdot \| h \|$ , attained only if every rounding error shares a sign). This ratio deliberately combines a worst-case numerator with a typical-case denominator. Under the random-sign assumption, this choice biases the diagnostic toward assigning greater relative importance to body error. Because the denominator remains an estimate rather than a bound, however, the ratio is not a formal bound on either contribution. Adversarial weight configurations in which the d products share a sign would make the head term $\sqrt { d }$ times larger and the ratio correspondingly smaller. We therefore report the ratio only as an order-of-magnitude estimate and assess it empirically: it matches the measured decomposition (∼97.6% head-originated in the storage-vs-computation ablation) and the ordering of intervention lifts across the four precision configurations in Table 6. The actual body accumulation may be sub-linear (statistical cancellation across layers would give $\sqrt { L } )$ , which would reduce the estimated body-to-head ratio further under the same assumptions. The resulting estimate is $L \cdot \alpha / { \sqrt { d } } .$

Numerical evaluation. For TinyLlama-1.1B $( L { = } 2 2 , d { = } 2 0 4 8 , \alpha \approx 0 . 0 5$ measured as the mean per-layer ratio $\| F _ { \ell } ( h ) \| / \| h \|$ across 22 layers and 100 GSM8K prompts via the unconditional layer trace in Appendix II):

$$
{ \frac { 2 2 \times 0 . 0 5 } { \sqrt { 2 0 4 8 } } } = { \frac { 1 . 1 } { 4 5 . 3 } } \approx 0 . 0 2 4 .\tag{6}
$$

Under this heuristic, the body contributes ∼2.4% of the per-logit perturbation and the lm\_head contributes ${ \sim } 9 7 . 6 \%$ . FP32 recomputation of the lm\_head therefore targets the estimated dominant source, consistent with the $+ 2 2 \mathrm { p p }$ lift. Extending the FP32 scope to body layers addresses only the estimated residual while introducing the butterfly efect (Section 7.6), which is why scope extension empirically hurts (Appendix IX: RMSNorm+lm\_head drops EAR from 63% to 44% on TinyLlama; Appendix IX replicates the result on DeepSeek-R1-7B with $\tau = 0 )$ . Full derivation and step-0 (prefill) edge case in Appendix I.

A conditional form that covers every precision pair we test. The ratio above is derived for two paths that run the same body in diferent low precisions, which is the BF16-vs-FP16 setting. Our FP8 experiments (Appendix XV) sit outside that assumption, so we state the estimate in the conditional form that governs all three configurations. The relevant quantity is not the head’s share of $\sigma _ { z }$ but the cross-arm hidden-state gap $\Delta h = \| h ^ { ( \bar { A } ) } - h ^ { ( B ) } \| / \| h \|$ , because Intervention C works by making the two arms compute identical logits: at a gated step both evaluate floa $\mathrm { t } 3 2 ( W _ { \mathrm { l m } } )$ · float32(h), which coincide only to the extent that h coincides. Writing $n _ { q }$ for the number of quantized linear operations per layer and $\epsilon _ { A } , \epsilon _ { B }$ for the two arms’ body precisions,

$$
\Delta h \approx \mathrm { m a x } ( n _ { q } , 1 ) L \alpha | \epsilon _ { A } - \epsilon _ { B } | ,\tag{7}
$$

and C can reconcile a gated step only when $\Delta h$ is small relative to the margin scale $\sigma _ { z }$ . This yields an ordering that matches every configuration we measure:

Table 6: The conditional form of the scope estimate, evaluated for all four format configurations we test (the batched-decode and cross-kernel boundaries are governed by a diferent mechanism; see text) (TinyLlama-1.1B: L=22, d=2048, α≈0.05, $n _ { q } = 7$ linear ops per layer). $\Delta h$ is the cross-arm hidden-state gap from Eq. 7. The ratio column uses a single reference scale, the BF16-vs-FP16 value $\sigma _ { z } \approx 0 . 0 2 6 ,$ , so that the four rows are comparable on one axis; the format’s own $\sigma _ { z }$ is larger for FP8 (Appendix XV) and smaller for FP16-vs-FP32, and using per-format values would compress the spread without changing which regime each row falls in.
<table><tr><td>Configuration</td><td>Body across arms</td><td> $\Delta h$ </td><td> $\Delta h / \sigma _ { z }$ </td><td>Predicted / observed</td></tr><tr><td>FP8 at head only</td><td>bit-identical</td><td>0</td><td>0</td><td>complete / +70pp, EAR 100%</td></tr><tr><td>FP16 vs. FP32 (T4)</td><td>same body, two precisions</td><td>0.0005</td><td>~0.02</td><td>head-dominated  $/ + 7 \mathrm { p p }$ </td></tr><tr><td>BF16 vs. FP16 (main)</td><td>same body, two precisions</td><td>0.004</td><td>~0.14</td><td>head-dominated  $/ + 2 2 \mathrm { p p }$ </td></tr><tr><td>End-to-end FP8 vs. BF16</td><td>body quantized in one arm</td><td>0.45</td><td> ${ \sim } 1 7$ </td><td>body-dominated  $/ + 1 \mathrm { p p }$ </td></tr></table>

The four configurations span nearly three orders of magnitude in $\Delta h / \sigma _ { z } .$ , and the regime tracks that ordering: complete reconciliation when the body is shared $( \Delta h = 0 )$ , a positive lift whenever the body gap stays well below the margin scale $( \Delta h / \sigma _ { z } \le 0 . 1 4 )$ , and a negligible one once it exceeds that scale (17). The raw lift is not monotone in $\Delta h$ and we do not claim it is: FP16-vs-FP32 yields +7pp at $\Delta h / \sigma _ { z } = 0 . 0 2$ against a baseline already at 88%, i.e. it closes 58% of the available headroom, whereas BF16-vs-FP16 yields +22pp from a 41% baseline, closing 37%. What $\Delta h$ orders is which regime a configuration falls in, not the size of the lift within a regime. The criterion also explains why the two pairs with the largest raw precision gaps behave oppositely: FP16-vs-FP32 has a large ϵ ratio but a tiny $\Delta h$ because FP32 carries the untruncated weights, whereas end-to-end FP8 has both a large ϵ and $n _ { q } = 7$ quantized operations per layer. The $L \alpha / \sqrt { d } \approx 0 . 0 2 4$ figure of the previous paragraph is the special case $n _ { q } { = } 1 , \epsilon _ { A } / \epsilon _ { B } = \mathrm { B F } 1 6 / \mathrm { F P } 1 6 ;$ it should not be read as a constant. This gives an operational test for a new format pair: measure $\Delta h$ between the two serving configurations on a handful of prompts and compare it to the margin scale. Measured on GSM8K (n=50, final-token hidden state), $\Delta h = 0 . 0 1 0 5$ for TinyLlama BF16-vs-FP16 and 0.0219 for Qwen2.5-3B under the same pair — about twice the body gap. We note two limits on reading that comparison causally. The predicted value from Eq. 7 for the TinyLlama pair is 0.004, so the estimate is a factor 2.6 below measurement. And Qwen’s lift under this same pair and the same $\Delta h$ is +3pp on A10G but +14pp on A100 and +20pp on L4 (Appendix XIV), so $\Delta h$ cannot be what sets the lift’s magnitude for that model; the kernel path matters at least as much. We are explicit about the criterion’s scope. It applies to format pairs evaluated at a fixed batch size, and it does not explain the batched-decode boundary: measuring bs=1 against bs=8 in the same precision gives $\Delta h = 0 . 0 1 2 7$ , statistically indistinguishable from the working BF16-vs-FP16 configuration, yet the lift there degrades to zero. Batching fails for a diferent reason, which we treat separately (Section 7.4): a changed reduction order produces a third trajectory rather than a larger gap between two, so gated recomputation introduces new divergence instead of removing it. The criterion likewise takes the kernel implementation as fixed — the +3/+14/+20pp spread for Qwen across A10G, A100 and L4 (Appendix XIV) shows the efective ϵ is itself implementation-dependent, a dimension the estimate does not cover.

Table 7: Divergence-conditioned trace (TinyLlama-1.1B, $n { = } 1 0 0 )$
<table><tr><td></td><td>DIVERGED</td><td>AGREED</td><td>Gap</td></tr><tr><td>Body L2 (layer 21)</td><td>1.022</td><td>1.135</td><td>-0.113</td></tr><tr><td>Logit L2</td><td>4.928</td><td>5.559</td><td>-0.631</td></tr><tr><td>Top-two margin</td><td>0.039</td><td>5.733</td><td>-5.694</td></tr><tr><td>Flip rate</td><td>58%</td><td>0%</td><td></td></tr></table>

## 7.4 Why ungated recomputation hurts

Ungated recomputation $( \tau = 0 )$ produces a third trajectory at safe steps, introducing new divergence downstream (“butterfly efect”). Result: +2.5pp vs. +6.5pp for gated on DeepSeek-R1-1.5B. Gating is mechanistically necessary.

## 7.5 Empirical validation: layer-level divergence tracing

We trace BF16-vs-FP16 hidden-state L2 through all 22 layers on $n = 1 0 0 \ \mathrm { G S M 8 K }$ prompts, comparing Diverged $\left( n { = } 5 9 \right)$ and Agreed (n=41) at their respective measurement steps. Key finding (Table 7): body-layer L2 is indistinguishable between groups, but the top-two margin difers by 150× (0.039 vs. 5.733). Four independent causal experiments (layer grafting, KV-cache grafting, logit-lens traceback, Qwen2.5-3B replication; Appendices II–II) confirm: the flip is produced at the lm\_head.

## 7.6 Summary

The analyses above converge on a three-factor causal picture. Cross-precision greedy divergence requires the simultaneous presence of three conditions—remove any one and the outputs agree:

1. Low arithmetic precision at the lm\_head. BF16’s 7-bit mantissa introduces per-element rounding of magnitude $\epsilon _ { \mathrm { B F 1 6 } } \approx 3 . 9 \times 1 0 ^ { - 3 }$ , 8× larger than $\mathrm { F P 1 6 ^ { \circ } s 4 . 9 \times 1 0 ^ { - 4 } }$ , so the two precisions compute materially diferent logits at every step (per-logit perturbation $\sigma _ { z } \approx 0 . 0 2 6 \mathrm { : }$ ; see Section 7.5).

2. The lm\_head amplifies hidden-state error. The $d \times \vert V \vert$ projection amplifies body-layer $\ell _ { 2 }$ error by $\sim 5 \times ( 0 . 9 2  4 . 6 3 )$ ; without this amplification, per-logit perturbations would sit at ∼0.005, below typical margins.

3. Greedy decoding has zero tolerance for rank flips. Unlike sampling, greedy argmax converts any logit-ranking reversal into a hard token diference, and a single flipped token cascades through autoregressive conditioning (mean length diference 34 tokens, Figure 1).

Intervention C breaks condition (1) at the dominant correctable stage. Recomputing the lm\_head in FP32 $( \epsilon _ { \mathrm { F P 3 2 } } \approx 6 \times 1 0 ^ { - 8 } )$ reduces the head-originated perturbation from $\sim 0 . 0 3 \mathrm { ~ t o ~ } \sim 1 0 ^ { - 5 }$ . Conditions (2) and (3) remain unchanged: the lm\_head still amplifies its input and greedy decoding still selects the argmax, but the corrected head contributes much less arithmetic error. The storage-vs-computation ablation supports this decomposition quantitatively: upgrading only the matmul compute path to FP32 (while keeping storage at BF16) drives EAR from 0.41 to 0.82, and the corresponding FP32-compute $/$ FP32-compute pair sits at 0.88 rather than 1.00, indicating that ∼87% of the reachable EAR gap is BF16-arithmetic-origin (addressable by C) while a residual ${ \sim } 1 2 \mathrm { p p }$ is associated with weight truncation. In the evaluated ablations, extending the FP32 scope beyond lm\_head (to RMSNorm or the last transformer layer) reduces EAR from 63% to 44–52% (Appendix IX); this shows that broader recomputation is not automatically better and can introduce new downstream trajectory diferences.

## 8 Discussion

## 8.1 Practical implications

Intervention C is intended for low-batch greedy decoding (bs≤4), a regime that can arise in code completion, structured-output generation, benchmark evaluation, and audit-oriented replay. In our A10G measurements it is a low-overhead intervention $( < 4 \% ;$ Section 6.3), but its benefit should be measured on the target model and platform. Two scope boundaries apply: (i) models with FP16-body NaNs or body-dominated divergence require a body-scope or training-time remedy, and (ii) at $\mathrm { b s } \ge 8$ the head-only intervention provides no lift by itself in our tests and must be composed with a body-scope method (Appendix VIII). The efect remains positive with both attention kernels tested (eager and FlashAttention-2) and on A10G, L4, and A100, although the lift varies substantially by GPU (Appendix XIV). Precision and kernel configuration should therefore be recorded explicitly rather than assuming platform-independent agreement.

## 8.2 What does partial agreement provide?

Exact whole-sequence agreement is a strict metric: one difering token anywhere in a 256-token generation counts as disagreement. In the tested BF16/FP16 setting, Intervention C raises this metric from 30–41% to 55–63% at < 4% A10G latency overhead, but it does not provide bit-exact reproduction. The practical value is application-dependent. On Qwen2.5-3B HumanEval, execution outcomes agree on 97.6% of problems while token sequences agree on 26.2% (4 of 164 execution outcomes flip; Appendix XVI); on GSM8K, however, Qwen2.5-3B shows a 19% cross-precision correctness-flip rate. These results rule out treating either token disagreement or task neutrality as universal proxies for the other. The TOST analysis establishes noninferiority within a 5pp margin on Qwen2.5-3B (n = 600, p = 0.009), while TinyLlama’s very low baseline accuracy makes its non-inferiority result only a consistency check (Table 5).

The head-isolated FP8 experiment is a controlled upper-bound diagnostic: sharing the BF16 body and weight storage removes body-originated divergence and the BF16/FP16 weight-truncation floor by construction, after which a format-rescaled gate reaches 100% exact agreement on TinyLlama and gains +56pp on Qwen2.5- 3B. It is not representative of end-to-end FP8 serving. When every linear layer is quantised to FP8, the same head-scope repair yields only +1pp (Appendix XV). Applications requiring bit-exact replay should pin the full numerical configuration or use an end-to-end higher-precision reference rather than relying on Intervention C alone.

## 8.3 Practitioner decision procedure

The deployment decision tree (Section 7.6) summarises the action tiers. The screening metric, the coeficient of variation of trigger density (CV(TD)), requires only a single BF16 run (∼5–10 minutes, no FP16 comparison needed). Retrospectively, it separates the models in our study into the observed action tiers; across the five models with a defined lift comparison, CV(TD) is associated with Intervention C’s EAR lift (Spearman $\rho = 0 . 9 0$ , Pearson $r = 0 . 8 4 \AA$ ). This is an in-sample screening heuristic, not a validated cross-model predictor, and should be recalibrated on the target hardware.

## 8.4 Limitations

We characterise divergence across six models from four families at scales from 1.1B to 7B and include a divergence-only probe at 12B; the full intervention comparison is not evaluated at either the 12B or 70B+ scale. Most experiments use an A10G, with selected headline replications on L4 and A100 and an FP16/FP32 probe on T4. The direction of the intervention efect is consistent in these replications, but its magnitude is hardware-dependent. C is evaluated for greedy decoding and is efective by itself only at bs≤ 4 in our batch-size sweep; at bs≥8 and under end-to-end FP8, divergence becomes body-dominated and a head-only repair is insuficient. We do not evaluate sampling-based decoding. C does not guarantee full agreement, and the ${ \sim } 1 2 \mathrm { p p }$ BF16/FP16 weight-storage gap measured in our ablation cannot be removed by changing only the lm\_head arithmetic.

## 9 Conclusion

LLM greedy decoding is not precision-invariant: 49–100% of prompts diverge between BF16 and FP16 across the models and benchmarks we test (59–82% on the four models that are neither BF16-saturated nor at the low end). Divergence concentrates at low top-two margins relative to the directional perturbation, with a 150× gap between the measurement-step margins of divergent and convergent prompts in the headline trace. Gated FP32 lm\_head recomputation raises exact agreement by +22–36pp on A10G at <4% latency overhead and is the best-performing low-overhead intervention among those evaluated. Selected L4 and A100 replications yield positive but hardware-dependent lifts of +12–21pp. The intervention is evaluated up to the 7B scale (with divergence additionally characterised at 12B), on dense and MoE models, and is limited to low-batch greedy serving. In the controlled head-isolated FP8 setting, a format-rescaled gate reaches exact agreement on TinyLlama and gains +56pp on Qwen2.5-3B; in end-to-end FP8, where bod error dominates, the gain is only +1pp. Overall, the results identify low-margin lm\_head events as an important and partially controllable source of cross-precision divergence, while also delineating regimes in which a head-scope intervention is insuficient.

## Broader impact

Cross-precision reproducibility can matter when exact replay, audit trails, or per-example evaluation records are required. Our intervention narrows the measured agreement gap at <4% A10G latency overhead in lowbatch (bs≤4) greedy inference and requires no model retraining, but it does not guarantee identical outputs. We see no direct misuse pathway introduced by the method: it changes numerical consistency rather than model capability. Reproducibility is not correctness; a repeatable but wrong model remains wrong, and this work is not a substitute for accuracy, safety, or domain-specific validation.

## References

Berk Atil, Sarp Aykent, Alexa Chittams, et al. Non-determinism of “deterministic” LLM settings. arXiv preprint arXiv:2408.04667, 2024.

Jacob Austin, Augustus Odena, Maxwell Nye, et al. Program synthesis with large language models. arXiv preprint arXiv:2108.07732, 2021.

Mark Chen, Jerry Tworek, Heewoo Jun, et al. Evaluating large language models trained on code. arXiv preprint arXiv:2107.03374, 2021.

Ranjith Chodavarapu and Lei Xu. The illusion of equivalence: Systematic FP16 divergence in KV-cached autoregressive inference. arXiv preprint arXiv:2604.15409, 2026.

Karl Cobbe, Vineet Kosaraju, Mohammad Bavarian, et al. Training verifiers to solve math word problems. arXiv preprint arXiv:2110.14168, 2021.

Tri Dao, Daniel Y. Fu, Stefano Ermon, Atri Rudra, and Christopher Ré. FlashAttention: Fast and memory-eficient exact attention with IO-awareness. In NeurIPS, 2022.

DeepSeek-AI. DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning. arXiv preprint arXiv:2501.12948, 2025.

Tim Dettmers, Mike Lewis, Younes Belkada, and Luke Zettlemoyer. LLM.int8(): 8-bit matrix multiplication for transformers at scale. In NeurIPS, 2022.

Tim Dettmers, Artidoro Pagnoni, Ari Holtzman, and Luke Zettlemoyer. QLoRA: Eficient finetuning of quantized language models. In NeurIPS, 2023.

Elias Frantar, Saleh Ashkboos, Torsten Hoefler, and Dan Alistarh. GPTQ: Accurate post-training quantization for generative pre-trained transformers. In ICLR, 2023.

Tairan Fu, Gonzalo Martínez, Javier Conde, Carlos Arriaga, Pedro Reviriego, Xiuyuan Qi, and Shanshan Liu. Beyond reproducibility: Token probabilities expose large language model nondeterminism. arXiv preprint arXiv:2601.06118, 2026.

Alireza Ghafari, Justin Yu, Mahsa Ghazvini Nejad, Masoud Asgharian, Boxing Chen, and Vahid Partovi Nia. Mitigating outlier activations in low-precision fine-tuning of language models. arXiv preprint arXiv:2312.09211, 2023.

David Goldberg. What every computer scientist should know about floating-point arithmetic. ACM Computing Surveys, 23(1):5–48, 1991.

Raja Gond, Aditya K. Kamath, Ramachandran Ramjee, and Ashish Panwar. LLM-42: Enabling determinism in LLM inference with verified speculation. arXiv preprint arXiv:2601.17768, 2026.

Horace He and Thinking Machines Lab. Defeating nondeterminism in LLM inference. Thinking Machines Lab technical blog, September 2025. https://thinkingmachines.ai/blog/ defeating-nondeterminism-in-llm-inference/.

Nicholas J. Higham. Accuracy and Stability of Numerical Algorithms. SIAM, 2nd edition, 2002.

Ari Holtzman, Jan Buys, Li Du, Maxwell Forbes, and Yejin Choi. The curious case of neural text degeneration. In ICLR, 2020.

Dhiraj Kalamkar, Dheevatsa Mudigere, Naveen Mellempudi, et al. A study of BFLOAT16 for deep learning training. arXiv preprint arXiv:1905.12322, 2019.

Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, et al. Eficient memory management for large language model serving with PagedAttention. In SOSP, 2023.

Yaniv Leviathan, Matan Kalman, and Yossi Matias. Fast inference from transformers via speculative decoding. In ICML, 2023.

Yingru Li, Jiawei Xu, Jiacai Liu, Yuxuan Tong, Ziniu Li, Tianle Cai, Ge Zhang, Qian Liu, and Baoxiang Wang. Dynamic vocabulary pruning: Stable LLM-RL by taming the tail. arXiv preprint arXiv:2512.23087, 2025.

Ji Lin, Jiaming Tang, Haotian Tang, et al. AWQ: Activation-aware weight quantization for LLM compression and acceleration. In MLSys, 2024.

Peter Mattson, Christine Cheng, Cody Coleman, et al. MLPerf training benchmark. In MLSys, 2020.

Alberto Messina and Stefano Scotta. Introducing background temperature to characterise hidden randomness in large language models. Transactions on Machine Learning Research, 2026.

Paulius Micikevicius, Sharan Narang, Jonah Alben, et al. Mixed precision training. In ICLR, 2018.

Niklas Muennighof, Luca Soldaini, Dirk Groeneveld, Kyle Lo, Jacob Morrison, Sewon Min, Weijia Shi, Pete Walsh, Oyvind Tafjord, Nathan Lambert, Yuling Gu, Shane Arora, Akshita Bhagia, Dustin Schwenk, David Wadden, Alexander Wettig, Binyuan Hui, Tim Dettmers, Douwe Kiela, Ali Farhadi, Noah A. Smith, Pang Wei Koh, Amanpreet Singh, and Hannaneh Hajishirzi. OLMoE: Open mixture-of-experts language models. arXiv preprint arXiv:2409.02060, 2024.

NVIDIA. TensorRT-LLM: A TensorRT toolbox for optimized large language model inference. GitHub Repository, 2023.

Xu Ouyang, Tao Ge, Thomas Hartvigsen, Zhisong Zhang, Haitao Mi, and Dong Yu. Low-bit quantization favors undertrained LLMs: Scaling laws for quantized LLMs with 100T training tokens. arXiv preprint arXiv:2411.17691, 2024.

Hung Viet Pham, Shangshu Qian, Jiannan Wang, et al. Problems and opportunities in training deep learning software systems: An analysis of variance. In ASE, 2020.

PyTorch Team. Reproducibility: Controlling sources of randomness. https://pytorch.org/docs/ stable/notes/randomness.html, accessed 2026. Also: torch.use\_deterministic\_algorithms and the CUBLAS\_WORKSPACE\_CONFIG environment variable.

Penghui Qi, Zichen Liu, Xiangxin Zhou, Tianyu Pang, Chao Du, Wee Sun Lee, and Min Lin. Defeating the training inference mismatch via FP16. arXiv preprint arXiv:2510.26788, 2025.

Qwen Team. Qwen2.5 technical report. arXiv preprint arXiv:2412.15115, 2024.

SGLang Team. Towards deterministic inference in SGLang and reproducible RL training. LMSYS technical blog, September 2025. https://www.lmsys.org/blog/2025-09-22-sglang-deterministic/.

Ashish Vaswani, Noam Shazeer, Niki Parmar, et al. Attention is all you need. In NeurIPS, 2017.

Yifei Wang, Tianlin Li, Xiaohan Zhang, Xiaoyu Zhang, Wei Ma, Mingfei Cheng, and Li Pan. Hidden reliability risks in large language models: Systematic identification of precision-induced output disagreements. arXiv preprin arXiv:2604.19790, 2026.

Jason Wei, Xuezhi Wang, Dale Schuurmans, et al. Chain-of-thought prompting elicits reasoning in large language models. In NeurIPS, 2022.

Guangxuan Xiao, Ji Lin, Mickael Seznec, et al. SmoothQuant: Accurate and eficient post-training quantization for large language models. In ICML, 2023.

Zhewei Yao, Reza Yazdani Aminabadi, Minjia Zhang, et al. ZeroQuant: Eficient and afordable post-training quantization for large-scale transformers. In NeurIPS, 2022.

Jiayi Yuan, Hao Li, Xinheng Ding, Wenya Xie, Yu-Jhe Li, Wentian Zhao, Kun Wan, Jing Shi, Xia Hu, and Zirui Liu Understanding and mitigating numerical sources of nondeterminism in LLM inference. In NeurIPS (Oral), 2025.

Peiyuan Zhang, Guangtao Zeng, Tianduo Wang, and Wei Lu. TinyLlama: An open-source small language model. arXiv preprint arXiv:2401.02385, 2024.

Ziyang Zhang, Xinheng Ding, Jiayi Yuan, Rixin Liu, Huizi Mao, Jiarong Xing, and Zirui Liu. Deterministic inference across tensor parallel sizes that eliminates training-inference mismatch. arXiv preprint arXiv:2511.17826, 2025.

Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, et al. SGLang: Eficient execution of structured language model programs. In NeurIPS, 2024.

Xunyu Zhu, Jian Li, Yong Liu, Can Ma, and Weiping Wang. A survey on model compression for large language models. Transactions of the Association for Computational Linguistics (TACL), 2024.

Juntang Zhuang, Tommy Tang, Yifan Ding, et al. Randomness in neural network training: Characterizing the impact of tooling. In MLSys, 2022.

## Appendix overview

The supplementary material is organised into six groups, each self-contained:

• Analysis details (App. I): the proof context for the exact flip condition and full derivations of the heuristic error-propagation estimates.

• Mechanism evidence (App. II–II): four causal experiments isolating divergence to the lm\_head.

• Cross-model robustness (App. III–VII): replication across six models, four families, math and code benchmarks.

• Production-stack robustness (App. VIII): FlashAttention, batch sizes, global-FP32 composition, overhead.

• Ablations (App. IX): scope, threshold, top-K, alternative interventions.

• Statistical validation and reproducibility (App. X–XI): CIs, scaled replication, code.

Cross-references between appendix sections are kept to a minimum.

## I Error-propagation analysis: derivations and assumptions

This appendix presents a heuristic error-propagation model. With the exception of Proposition 1 (exact), the estimates here rest on the stated simplifying assumptions and are validated empirically rather than proven; counterexamples to the asymptotic rates can be constructed under adversarial weight configurations. The model predicts the empirical findings in the main text and characterises the scope boundary of Intervention C.

## I.1 Setup and notation

Let $f _ { \theta }$ denote an L-layer autoregressive transformer with hidden dimension d and vocabulary size |V |. At decoding step t, the model computes:

$$
h _ { t } ^ { ( \ell ) } = \mathrm { L a y e r } _ { \ell } ( h _ { t } ^ { ( \ell - 1 ) } ) , \quad \ell = 1 , \ldots , L ,\tag{8}
$$

$$
z _ { t } = W _ { \mathrm { l m } } \cdot h _ { t } ^ { ( L ) } , \quad W _ { \mathrm { l m } } \in \mathbb { R } ^ { | V | \times d } ,\tag{9}
$$

$$
y _ { t } = \arg \operatorname* { m a x } _ { v } z _ { t } ( v ) .\tag{10}
$$

Under floating-point precision c with machine epsilon $\epsilon _ { c } ,$ each arithmetic operation has relative error bounded by $\epsilon _ { c }$ (Goldberg, 1991; Higham, 2002). The relevant values are $\epsilon _ { \mathrm { B F 1 6 } } = 2 ^ { - 8 } \approx 3 . 9 \times 1 0 ^ { - 3 }$ (7-bit mantissa) and $\epsilon _ { \mathrm { F P 1 6 } } = 2 ^ { - 1 1 } \approx 4 . 9 \times 1 0 ^ { - 4 }$ (10-bit mantissa), giving a precision ratio of $\epsilon _ { \mathrm { B F 1 6 } } / \epsilon _ { \mathrm { F P 1 6 } } = 8 $

## I.2 Error accumulation through the transformer body

Heuristic estimate: body-layer error accumulation. Let $\hat { h } _ { t } ^ { ( \ell ) }$ denote the hidden state computed under precision c (with the other precision as reference). Under three explicit assumptions—(i) standard transformer architecture with residual connections $h ^ { ( \ell ) } = \bar { h } ^ { ( \ell - 1 ) } + F _ { \ell } \bar { ( h ^ { ( \ell - 1 ) } ) }$ , (ii) per-layer relative rounding error bounded by $\epsilon _ { c } C _ { \ell }$ with condition number $C _ { \ell }$ of order unity, and (iii) no systematic error cancellation across layers (worst case)—the accumulated error satisfies:

$$
\| \hat { h } _ { t } ^ { ( L ) } - h _ { t } ^ { ( L ) } \| \leq \epsilon _ { c } \cdot \| h \| _ { \mathrm { r m s } } \cdot \sum _ { \ell = 1 } ^ { L } C _ { \ell } \prod _ { j = \ell + 1 } ^ { L } ( 1 + \epsilon _ { c } C _ { j } ) .\tag{11}
$$

When $\epsilon _ { c } C _ { \ell } \ll 1$ for all ℓ (satisfied for both BF16 and FP16 on standard transformers where $C _ { \ell }$ is of order unity), this simplifies to linear accumulation:

$$
\| \hat { h } _ { t } ^ { ( L ) } - h _ { t } ^ { ( L ) } \| \approx \epsilon _ { c } \cdot \bar { C } \cdot L \cdot \| h \| _ { \mathrm { r m s } } ,\tag{12}
$$

where $\bar { C } = ( 1 / L ) \sum _ { \ell } C _ { \ell }$ is the mean per-layer condition number. This is a growth-rate estimate under the stated assumptions, not a theorem about arbitrary weight configurations; its adequacy is assessed empirically below.

Empirical validation. On TinyLlama-1.1B $( L = 2 2 , \ d = 2 0 4 8 )$ , we measure BF16-vs-FP16 hidden-state L2 growing linearly from 0.002 at layer 0 to 0.92 at layer L (Appendix II), consistent with the linear-accumulation regime.

## I.3 Logit perturbation and the divergence condition

Derivation of the heuristic scale analysis. Proposition 1 gives the exact flip condition: the argmax changes if and only if there exists a competitor whose diferential perturbation exceeds its original logit gap to the top-1 token. For the original runner-up, this reduces to a comparison between its signed directional perturbation diference and the top-two margin $\Delta _ { t } = z _ { t } ( v ^ { ( 1 ) } ) - z _ { t } \bar { ( v ^ { ( 2 ) } ) }$ . Here we derive the heuristic scale estimate that turns this exact but unobservable condition into a practical trigger statistic. Let $\sigma _ { z } =$ $\| \hat { z } _ { t } - z _ { t } \| / \sqrt { | V | }$ denote the RMS per-logit perturbation. Under the simplifying assumption that perturbations spread approximately uniformly across the |V | logit dimensions, propagating the body-error estimate above through the lm\_head gives:

$$
\sigma _ { z } \approx \frac { \| \boldsymbol { W } _ { \mathrm { l m } } \| \cdot \epsilon _ { c } \cdot \bar { C } \cdot L \cdot \| h \| _ { \mathrm { r m s } } } { \sqrt { | V | } } ,\tag{13}
$$

and a flip becomes plausible only when $\Delta _ { t } \lesssim \sigma _ { z }$ , so that heuristically

$$
P ( { \mathrm { f f i p ~ a t } } t ) \lesssim P ( \Delta _ { t } < \sigma _ { z } ) ,\tag{14}
$$

which depends primarily on the margin distribution of the model at that step, not on the magnitude of the accumulated body error. This is an approximation, not a bound: per-coordinate perturbations are neither Gaussian nor independent, and its adequacy is assessed empirically.

Empirical validation. For TinyLlama-1.1B: $\sigma _ { z } \approx 4 . 6 3 / \sqrt { 3 2 0 0 0 } \approx 0 . 0 2 6$ . Measured margins: Diverged median = 0.039 (just above $\sigma _ { z } )$ , Agreed medi $\mathrm { u n } = 5 . 7 3 3 \ ( 2 2 0 \times \mathrm { a b o v e \ } \sigma _ { z } )$ . The scale analysis correctly predicts: (i) body-layer L2 does not distinguish the two groups (confirmed: Table 7); (ii) the margin is the dominant factor (confirmed: 150× gap); (iii) the K-ablation result (K = 2 sufices, Appendix IX), because a perturbation of magnitude $\sigma _ { z } \approx 0 . 0 2 6$ cannot bridge a rank-2-to-rank-3 gap that is typically ≫ 0.1.

## I.4 Intervention C: error reduction at the lm\_head

Quantitative account: why Intervention C reconciles the paths. FP32 lm\_head recomputation eliminates the cross-path logit diference at the lm\_head by making both precision paths compute identical FP32 logits from their respective hidden states—an immediate consequence of the determinism of the FP32 matmul, not a theorem. For prompts where the BF16 and FP16 hidden states $h _ { t } ^ { ( L ) }$ are efectively identical (the “lm\_head-originated” regime), both paths produce bit-identical FP32 logits and no flip can occur. Under the same simplifying assumptions as above, the lm\_head-specific rounding contribution is heuristically reduced by a factor of $\epsilon _ { \mathrm { F P 3 2 } } / \epsilon _ { \mathrm { B F 1 6 } } \approx 1 . 5 \times 1 0 ^ { - 5 }$

$$
\sigma _ { z } ^ { ( \mathrm { l m \_ h e a d , \ F P 3 2 } ) } = \frac { \epsilon _ { \mathrm { F P 3 2 } } } { \epsilon _ { \mathrm { B F l 6 } } } \cdot \sigma _ { z } ^ { ( \mathrm { l m \_ h e a d , \ B F l 6 } ) } \approx 0 . 0 2 6 \times 0 . 9 7 6 \times 1 . 5 \times 1 0 ^ { - 5 } \approx 4 \times 1 0 ^ { - 7 } .\tag{15}
$$

The residual cross-path diference after C is the body-originated component only (∼2.4% of the original $\sigma _ { z } .$ per the dominance estimate of Section 7.3). For lm\_head-originated divergences where body hidden states are shared, this residual vanishes and:

$$
P ( { \mathrm { f l i p ~ a t ~ } } t \mid { \mathrm { F P 3 2 ~ l m ~ \_ h e a d , ~ l m ~ \_ h e a d \mathrm { - } o r i g i n a t e d } } ) = 0 .\tag{16}
$$

The achievable EAR lift then equals the fraction of prompts whose divergence is entirely lm\_head-originated:

$$
\mathrm { E A R } _ { \mathrm { m a x } } ^ { ( C ) } = \mathrm { E A R } _ { \mathrm { b a s e l i n e } } + \frac { | \{ i : \mathrm { p r o m p t ~ } i \mathrm { ~ d i v e r g e s ~ o n l y ~ a t ~ l m \_ { \mathrm { l e a d } } \} | } } { n } .\tag{17}
$$

Empirical validation. On TinyLlama-1.1B, 30/100 prompts have divergence triggered by a margin event at the lm\_head (Table 4, category $\mathrm { B + C ) } ;$ of these, 22 are successfully reconciled by C (73% success rate within the scope). The predicted maximum lift is 30%; the achieved lift is $\mathrm { 2 2 p p } \left( = 2 2 / 1 0 0 \right)$ , consistent with a 73% reconciliation rate within the reachable set identified by the analysis.

## I.5 Extended derivation and edge cases for the lm\_head dominance estimate

The main text (Section 7.3) states and numerically evaluates the lm\_head dominance estimate. Here we provide the full derivation and discuss an important edge case.

Full derivation. Decompose the total per-logit perturbation into body and lm\_head contributions:

$$
\sigma _ { z } = \sigma _ { z } ^ { ( \mathrm { b o d y } ) } + \sigma _ { z } ^ { ( \mathrm { l m \_ h e a d } ) } .\tag{18}
$$

Under standard transformer architecture with KV cache, we bound each contribution directly in per-logit units. Consider a single logit position v with row vector $W _ { v } \in \mathbb R ^ { d } \ ( \| W _ { v } \| \approx 1 $ under Xavier scaling):

$$
\sigma _ { z } ^ { ( \mathrm { b o d y } ) } \lesssim \frac { L \cdot \alpha \cdot \epsilon _ { c } \cdot \| h \| } { \sqrt { d } } ,\tag{19}
$$

$$
\sigma _ { z } ^ { ( \mathrm { l m \_ h e a d } ) } \approx \epsilon _ { c } \cdot \| h \| .\tag{20}
$$

Body term. Each of L layers introduces error only on the non-residual branch $F _ { \ell }$ (norm $\alpha \cdot \left\| h \right\| )$ , so the accumulated body error is $\| \Delta h _ { \mathrm { b o d y } } \| \le L \cdot \alpha \cdot \epsilon _ { c } \cdot \| h \|$ . Projected onto the logit direction $W _ { v } .$ , the per-logit efect is $| W _ { v } \cdot \Delta h | \approx \| \Delta h \| / \sqrt { d }$ (standard projection of a d-dimensional vector onto a fixed unit direction).

lm\_head term. The dot product $\begin{array} { r } { z _ { v } = \sum _ { j } W _ { v j } h _ { j } } \end{array}$ accumulates d independent rounding errors. Each multiplyaccumulate step contributes an error of order $\epsilon _ { c } . | W _ { v j } h _ { j } | \approx \epsilon _ { c } . \| h \| / \sqrt { d }$ (using $| W _ { v j } | \sim 1 / \sqrt { d } , | h _ { j } | \sim \| h \| / \sqrt { d } )$ Under standard error-summation arguments with partial sign cancellation, $\sqrt { d }$ such terms accumulate to a typical magnitude $\approx \epsilon _ { c } \cdot | | h | |$ . Taking the ratio directly:

$$
\frac { \sigma _ { z } ^ { ( \mathrm { b o d y } ) } } { \sigma _ { z } ^ { ( \mathrm { l m \_ h e a d } ) } } \approx \frac { L \cdot \alpha } { \sqrt { d } } ,\tag{21}
$$

consistent with the main-text dominance estimate of Section 7.3.

Edge case: step t = 0 (prefill). At step 0, the model processes the entire prompt through all L layers without a KV cache. Unlike cached decoding, where each step introduces fresh rounding error from only one token’s pass through the body, the prefill pass accumulates rounding error across all $T _ { \mathrm { p r o m p t } }$ token positions in the attention reductions. Each layer’s attention output is a weighted sum over $T _ { \mathrm { p r o m p t } }$ value vectors, and the floating-point non-associativity of this reduction introduces ≈ $\sqrt { T _ { \mathrm { p r o m p t } } }$ additional error terms per layer under typical-case cancellation. The body-error contribution to the dominance ratio therefore scales roughly as $L \cdot \alpha \cdot \sqrt { T _ { \mathrm { p r o m p t } } } / \sqrt { d } .$ , which can approach or exceed 1 for long prompts $( T _ { \mathrm { p r o m p t } } \gtrsim d / ( L ^ { 2 } \alpha ^ { 2 } )$ ≈ 850 for TinyLlama). In this regime, the lm\_head is no longer the dominant error source, and Intervention C’s lm\_head-only scope becomes insuficient—consistent with our observation that first-token divergence on Qwen models is irreducible by any lm\_head-scope method (Section 6.5).

## I.6 Scope boundary: when Intervention C fails

The error model above also predicts the failure modes:

Body-dominated divergence (Tier 3). If a model’s training-time precision (typically BF16) produced weights whose FP16 cast yields unstable activations (NaN or extreme values at intermediate layers), the body-layer error $\| \hat { h } _ { t } ^ { ( L ) } - h _ { t } ^ { ( L ) } \|$ is no longer small relative to $\| h \| _ { \mathrm { r m s } }$ . In this regime, $\sigma _ { z }$ is dominated by the body contribution rather than the lm\_head matmul alone, and FP32 recomputation of the lm\_head addresses only a fraction of the total perturbation. This predicts 0pp lift on Qwen BF16-saturated models (confirmed: DS-R1-Qwen-7B, Table 12).

Batch-induced divergence. At batch ${ \mathrm { s i z e } } \geq 2 .$ , attention reductions over padded sequences introduce an additional source of floating-point variation whose magnitude grows with batch size. This is not simply a larger cross-arm gap: measured bs=1-vs-bs=8 hidden states difer by $\Delta h = 0 . 0 1 2 7$ , comparable to the working BF16-vs-FP16 configuration (Section 7.3). The reduction order instead makes the batched arm a third trajectory, so an FP32 head recompute reconciles neither pair, and C’s lift vanishes (confirmed: 0pp at bs= 8, Appendix VIII).

Weight-truncation floor. A storage-vs-computation ablation shows that even configurations that both compute in FP32 (store BF16 / compute FP32 vs. store FP16 / compute FP32) reach $\operatorname { E A R } = 0 . 8 8$ rather than 1.00, leaving a residual ∼12pp gap attributable to $W _ { \mathrm { B F 1 6 } } \neq W _ { \mathrm { F P 1 6 } }$ (the same FP32 weights truncated to diferent mantissa widths). This component of cross-precision divergence is irreducible under the two fixed low-precision storage formats—it enters through the weights themselves, not through the computation. Full agreement in this comparison requires a shared storage format, such as loading both arms from the same FP32 weights.

## I.7 Intervention comparison from the error model

The error model also explains the other interventions:

Intervention A (integer quantization). Quantization bins have width $\tau = 1 0 ^ { - 3 }$ . The cross-precision logit diference at divergent steps is $\sigma _ { z } \approx 0 . 0 2 6 { \ - - 2 6 } \times$ larger than the bin width. The two precisions therefore map to diferent bins with probability $\approx 1$ , and quantization faithfully preserves the disagreement. Prediction: zero efect. Confirmed.

Intervention D (temperature sharpening). Multiplying logits by $\alpha > 1$ is a monotone transformation: arg ma $\operatorname { x } _ { v } ( \alpha \cdot z _ { t } ( v ) ) = \arg \operatorname* { m a x } _ { v } z _ { t } ( v )$ for all $\alpha > 0$ . The margin becomes $\alpha \cdot \Delta _ { t }$ and the perturbation becomes $\alpha \cdot \sigma _ { z } ;$ their ratio $\Delta _ { t } / \sigma _ { z }$ is invariant. Prediction: zero efect at any α. Confirmed (Appendix IX).

Intervention E (consensus decoding). Running both precisions and using FP32 as tiebreaker eliminates all format-level divergence by construction. The cost is 2× compute. The per-step disagreement rate P(token\_BF16 $\neq$ token\_FP16) at any single step is bounded by $P ( \Delta _ { t } < \sigma _ { z } )$ ; we measure this at 44.5% of steps (Appendix IX), consistent with the heavy left tail of the margin distribution on GSM8K.

Top-K suficiency. The error model predicts that a token ranked k-th can only be promoted to rank 1 if the perturbation $\sigma _ { z }$ exceeds the gap between rank 1 and rank k. Since $\sigma _ { z } \approx 0 . 0 2 6$ and the typical rank 2-to-rank-3 gap is $\gg 0 . 1$ (median ≈ 0.5 on TinyLlama), only rank-2 tokens can realistically be promoted. Prediction: $K = 2$ sufices. Confirmed: all $K \in \{ 2 , 4 , 8 , 1 6 , 3 2 , | V | \}$ produce identical EAR (Appendix IX).

Group 2 backs the lm\_head-localisation claim of Section 7.5 with four mutually independent experiments that triangulate the same conclusion: (i) the unconditional per-layer L2 trace (App. II) characterises the typical hidden-state error budget; (ii) layer grafting (App. II) converts the correlational claim into a causal one; (iii) KV-cache grafting $( \operatorname { A p p . } \operatorname { I I } )$ isolates the autoregressive-cache contribution; (iv) per-layer logit-lens top-2 traceback (App. II) shows BF16 / FP16 agree on the top-1 token at 97–98% of body layers and disagree only at the lm\_head, with cross-task generalisation to HumanEval and MBPP. The four experiments converge on a single causal claim: divergence is produced at the last transformer layer, the final RMSNorm, and the lm\_head matmul, the modules Intervention C targets or sits immediately downstream of — C recomputes the lm\_head projection only, and works because that is the one stage where an FP32 correction lands on both paths’ logits without perturbing upstream state.

## II Mechanism evidence: four causal experiments

Summary. (i) Unconditional trace: L2 grows linearly from 0.002 (layer 0) to 0.92 (layer 21), amplified 5× by lm\_head to 4.63. (ii) Layer grafting: body-layer grafting fixes 36% of flips; Layer 21 + RMSNorm fixes 65–71%. (iii) KV-cache grafting: swaps convert 36–48% of flips; partial causal contributor. Consistently, 25/59 Diverged prompts agree under single-step re-run (no precision-specific cache) while diverging autoregressively—i.e., ∼42% of flips require the accumulated cache state, while the remaining ∼58% (34/59)

reproduce from current-step arithmetic alone. (iv) Logit-lens traceback: BF16/FP16 agree on top-1 at ≥93% of body layers; disagreement is one-shot at lm\_head.

The four protocols and their full result tables follow. Together they answer a specific diagnostic question— does the BF16/FP16 disagreement begin before or inside the lm\_head?—from four independent directions: an unconditional error budget, a causal graft of upstream activations, a causal graft of the autoregressive cache, and a per-layer readout of the decision itself.

## (i) Unconditional per-layer hidden-state trace

Protocol. We trace the BF16-vs-FP16 hidden-state $\ell _ { 2 }$ distance through all 22 TinyLlama-1.1B layers on the first generated token, averaged over n = 50 GSM8K prompts, and separately record the logits after the lm\_head. This measurement is unconditional: it does not condition on whether the prompt eventually diverges, so it characterises the typical error budget rather than the flip event.

Result. Error originates at layer 0 $( \ell _ { 2 } ~ = ~ 0 . 0 0 2 1$ , relative error 0.39%, cosine similarity 0.999993) and accumulates approximately linearly to $\ell _ { 2 } ~ = ~ 0 . 9 3 4 1$ at layer 21, with relative error staying below 1.3% at every layer and cosine similarity never dropping below 0.99992. The final RMSNorm $( 3 . 4 \times )$ and the lm\_head $( 5 \times )$ then provide two additional amplification stages, taking the logit-level $\ell _ { 2 }$ to 4.6275. On the matched $n = 1 0 0$ summary measurement the same picture holds: $\ell _ { 2 } = 0 . 9 1 5$ before the lm\_head and 4.641 after (amplification 5.1×), with relative error falling from 1.05% to 0.50% and cosine similarity rising from 0.99994 to 0.99999. Crucially, all 100 first-token measurements agree across BF16 and FP16: the first generated token never flips on this model, because the implied per-logit perturbation $( 4 . 6 3 / \sqrt { | V | } \approx 0 . 0 2 6 )$ sits well below the mean top-two margin at that step $( \bar { \Delta } _ { t } = 0 . 7 0 )$ . The error budget is therefore always present and is by itself insuficient to produce a flip—which is what motivates the three conditional and causal experiments below.

## (ii) Causal attribution via layer grafting

Setup. The divergence-conditioned trace of Section 7.5 is correlational: it shows body-layer L2 is indistinguishable between Diverged and Agreed groups, which suggests—but does not prove—that body-layer error is not the causal driver. We convert the claim into a causal one via layer grafting. For each valid Diverged prompt where the flip reproduces under single-step re-run $( n = 3 4 / 1 0 0 )$ , we run a BF16 forward pass at context = prompt $\parallel y _ { < t ^ { * } }$ but override the output of one module ℓ with the corresponding FP16 hidden state $h _ { \ell } ^ { \mathrm { F P 1 6 } }$ , then allow BF16 kernels to process the grafted activation through all subsequent layers plus the BF16 lm\_head. We classify the resulting argmax as “fix (FP16)” if it matches the FP16 target, “keep (BF16)” if it matches the BF16 baseline, or “other”.

## Findings. Two causal conclusions follow.

(1) Body-layer hidden states are not suficient to determine the flip. If upstream hidden-state error were the causal driver, replacing it with the FP16 counterpart should fix the flip near 100% of the time. Instead, body-layer grafts fix only ∼36% of flips, and no single body layer is distinguished from the others. Crucially, even the embedding graft—where the entire downstream BF16 computation operates on FP16 input tokens—fixes only 29.4% of flips, with 70.6% of prompts still producing the BF16 argmax. This shows that BF16 kernel arithmetic itself acts as an attractor: given any hidden-state input, a BF16 forward pass tends to produce the same argmax as the all-BF16 baseline. The causal driver is therefore the arithmetic path, not the input activations.

(2) Causal responsibility concentrates in the last transformer layer and the final RMSNorm. Only grafts at Layer 21 and Final RMSNorm fix flips substantially above the body baseline (65% and 71% vs. ∼36%), a ∼30pp jump that occurs in exactly the last two modules before the lm\_head. This localises causal responsibility to the last two computation stages, consistent with the unconditional per-layer error trace above: both stages are the ones whose BF16 arithmetic most directly shapes the lm\_head’s input. The residual 21% of “keep BF16” outcomes at the RMSNorm graft reflects flips attributable to the BF16 lm\_head matmul itself (the final arithmetic stage that no upstream graft can modify), consistent with the storage-vs-computation decomposition (Section 7.6) into kernel arithmetic and storage efects.

Table 8: Causal attribution via layer grafting (TinyLlama-1.1B, GSM8K, n=34 valid Diverged prompts). The FP16 hidden state at each layer is spliced into the BF16 forward pass; downstream BF16 kernels (and the BF16 lm\_head) produce the final argmax. Body-layer grafts (Layers 0–20) have a flat ∼36% fix rate, well below 100%, demonstrating that the BF16 kernel path acts as an “attractor” that pulls grafted activations back to the BF16 argmax. The decisive causal jump occurs at Layer 21 (the final transformer layer) and the final RMSNorm, which together fix 65%–71% of flips.
<table><tr><td rowspan=1 colspan=1>Graft site              Fix (→FP16) % Keep (→BF16) % Other %</td></tr><tr><td rowspan=1 colspan=1>Embedding                   29.4                70.6             0.0</td></tr><tr><td rowspan=1 colspan=1>Layer 0                       47.1                52.9             0.0</td></tr><tr><td rowspan=1 colspan=1>Layer 5                       29.4                70.6             0.0</td></tr><tr><td rowspan=1 colspan=1>Layer 10                      38.2                58.8             2.9</td></tr><tr><td rowspan=1 colspan=1>Layer 15                      47.1                52.9             0.0</td></tr><tr><td rowspan=1 colspan=1>Layer 20                      35.3                64.7             0.0</td></tr><tr><td rowspan=1 colspan=1>Layer 0–20 (mean)           ~36                ~62             ~2</td></tr><tr><td rowspan=1 colspan=1>Layer 21 (last body)         64.7                26.5             8.8</td></tr><tr><td rowspan=1 colspan=1>Final RMSNorm             70.6                20.6             8.8</td></tr></table>

Tables 7 and 8 provide complementary evidence: the divergence-conditioned trace shows body-layer error is statistically the same across Diverged and Agreed prompts, and the graft experiment shows bodylayer hidden states are causally not suficient to determine the flip. The causal weight sits in the last transformer layer, the final RMSNorm, and the lm\_head matmul—the three stages that Intervention C’s FP32 recomputation (Section 4) already targets or sits immediately downstream of.

## (iii) Causal role of the KV cache

Setup. The graft experiment above uses a single-step forward at context = prompt ∥ y ∗ and therefore eliminates a confound: in the original autoregressive runs, the BF16 and FP16 paths carry precision-specific KV caches at step t<sup>∗</sup>, whereas the single-step re-run recomputes everything from scratch. Indeed, 25/59 Diverged prompts show argmax agreement under single-step re-run while still disagreeing under autoregressive greedy—proof that some flips are KV-cache-dependent. We now ask: does the cache itself cause these flips, or does it merely amplify upstream error so that the current-step lm\_head margin becomes flippable?

To probe this, we re-run each Diverged prompt autoregressively to step t<sup>∗</sup> in each precision, keeping the final past\_key\_values. We then compare four configurations for the single step that emits the next token:

• Baseline BF16 : BF16 model with BF16 cache → arg max = y<sup>BF16</sup>

• Baseline FP16 : FP16 model with FP16 cache → arg max = y<sup>FP16</sup>

• Graft A: BF16 model with FP16 cache (cast to BF16 dtype)

• Graft B: FP16 model with BF16 cache (cast to FP16 dtype)

We restrict to prompts where Baselines 1 and 2 disagree (n = 59): the cache is then provably the only diference between the two precisions that is carried into the next forward pass.

Findings. Three observations.

(1) The KV cache is a partial, not exclusive, causal source. Graft A converts 47.5% of Diverged prompts from BF16 argmax to FP16 argmax, proving that for almost half the flips the precision-specific cache is itself suficient. For the remaining 52.5%, the BF16 kernel operating on the FP16 cache still produces the

Table 9: KV-cache graft ablation (TinyLlama-1.1B, GSM8K, n = 59 Diverged prompts). Each row swaps the cache between BF16 and FP16 paths for the single forward step that produces $y _ { t ^ { * } } . \stackrel { 6 6 } {  }$ donor” = the cache donor’s argmax won; “→ kernel” = the arithmetic kernel’s own argmax won despite the foreign cache.
<table><tr><td>Configuration</td><td>→ cache donor  $\%$ </td><td>→ kernel  $\%$ </td><td>Other %</td></tr><tr><td>Graft A: BF16 model, FP16 cache</td><td>47.5</td><td>52.5</td><td>0.0</td></tr><tr><td>Graft B: FP16 model, BF16 cache</td><td>35.6</td><td>64.4</td><td>0.0</td></tr></table>

BF16 argmax, indicating that the current-step BF16 arithmetic dominates and the cache is only a secondary amplifier.

(2) The cause is asymmetric between precisions. Graft B fixes only 35.6% of flips, about 12pp less than Graft A. The BF16 kernel is more susceptible to the FP16 cache than the FP16 kernel is to the BF16 cache— consistent with BF16’s wider rounding error tolerance (storage-vs-computation decomposition, Section 7.6): BF16 arithmetic has a larger “basin” in which the cache can push the argmax; FP16 arithmetic’s tighter precision pulls it back to its own argmax even when given a BF16 cache.

(3) Combining graft experiments localises the divergence budget. Putting Tables 8 and 9 together: body-layer hidden-state grafts fix ∼36% of flips (no single layer distinguished), Layer 21 + Final RMSNorm grafts fix 65–71%, KV-cache grafts fix 36–48%, and the BF16 lm\_head matmul alone is responsible for the residual ∼21% of “kept BF16” outcomes under RMSNorm graft. These percentages are not additive—graft experiments are not independent treatments—but they delineate the causal surface: no single upstream component (body layers, KV cache, or final-layer hidden states) is individually suficient to explain divergence, and the only stage where a minimum-scope FP32 correction can deterministically fix the flip is the lm\_head itself, which is why Intervention C targets exactly there.

## (iv) Per-layer top-2 traceback via logit lens

Motivation. The causal picture above localises the flip to the lm\_head, but it does not directly answer a natural question: at the layer before the lm\_head, are BF16 and FP16 already disagreeing about which token is most likely? If so, the lm\_head is merely the surface on which an upstream decision becomes visible. If not, the flip is genuinely produced at the lm\_head itself. To answer this we run a logit-lens analysis: at each layer ℓ we apply the model’s own lm\_head to the last-token hidden state $h _ { \ell }$ and record the resulting top-2 tokens, for both BF16 and FP16 paths.

Setup. Same n= 100 GSM8K prompts, same measurement step protocol as Section 7.5 (t<sup>∗</sup> for Diverged, matched random step for Agreed). For each layer we report

• top1\_agree: $\begin{array} { r } { \operatorname* { P r } [ \arg \operatorname* { m a x } _ { v } \ln \_ { - } \mathrm { h e a d } ( h _ { \ell } ^ { \mathrm { B F 1 6 } } ) _ { v } = \arg \operatorname* { m a x } _ { v } \ln \_ { \mathrm { h e a d } } ( h _ { \ell } ^ { \mathrm { F P 1 6 } } ) _ { v } ] , } \end{array}$

• top2\_jaccard: Jaccard similarity of the two top-2 sets,

• bf\_top1=final and fp\_top1=final: fraction of prompts where the layer-ℓ top-1 already equals the final lm\_head top-1 (i.e. the precision has “committed” to its final token by layer ℓ),

• bf\_margin, fp\_margin: layer-ℓ top-two gap under each precision.

Three findings sharpen the causal picture. (1) Top-2 token identity is shared along almost the entire forward pass. Under either precision’s own lm\_head, the top-1 token at every body layer $( \ell \leq 2 0 )$ agrees between BF16 and FP16 for 93–100% of prompts. The top-2 set agrees ≥93% at every layer, including the final lm\_head. In other words, upstream of the lm\_head the two precisions are not yet disagreeing about “what the likely next tokens are”; they disagree only about how to rank within an otherwise shared pair. This directly answers the diagnostic question: the divergence is not an accumulating token-identity drift; it is a one-shot ranking reversal at the lm\_head itself.

Table 10: Per-layer top-2 traceback via logit lens (TinyLlama-1.1B, GSM8K, n=100, selected layers). BF16 and FP16 agree on the top-1 token at every body layer (≥ 93%), and on the top-2 set at every layer (≥ 93%). The disagreement appears only at the final lm\_head projection (Layer 21), where Diverged-group top-1 agreement drops to 76% while the top-2 set stays 99% aligned. Agreed prompts commit to their final token much earlier (49%/76%/90% at layers 15/18/20) than Diverged prompts (7%/34%/39%).
<table><tr><td rowspan="2">Layer</td><td colspan="3">DIVERGED (n = 59)</td><td colspan="3">AGREED (n = 41)</td></tr><tr><td>top1 agree</td><td>top2 jac.</td><td>bf=final</td><td>top1 agree</td><td>top2 jac.</td><td>bf=final</td></tr><tr><td>Embedding</td><td>100.0</td><td>100.0</td><td>0.0</td><td>100.0</td><td>96.7</td><td>0.0</td></tr><tr><td>Layer 5</td><td>100.0</td><td>98.9</td><td>0.0</td><td>97.6</td><td>95.1</td><td>7.3</td></tr><tr><td>Layer 10</td><td>98.3</td><td>97.7</td><td>1.7</td><td>97.6</td><td>96.7</td><td>17.1</td></tr><tr><td>Layer 15</td><td>94.9</td><td>96.6</td><td>6.8</td><td>100.0</td><td>98.4</td><td>48.8</td></tr><tr><td>Layer 18</td><td>93.2</td><td>100.0</td><td>33.9</td><td>100.0</td><td>96.7</td><td>75.6</td></tr><tr><td>Layer 20</td><td>96.6</td><td>98.9</td><td>39.0</td><td>100.0</td><td>98.4</td><td>90.2</td></tr><tr><td>Layer 21 (lm_head)</td><td>76.3</td><td>98.9</td><td>100.0</td><td>100.0</td><td>100.0</td><td>100.0</td></tr><tr><td>Margins (top-two gap)</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Layer 15 margin</td><td>0.075</td><td></td><td></td><td>0.148</td><td></td><td></td></tr><tr><td>Layer 18 margin</td><td>0.288</td><td></td><td></td><td>0.921</td><td></td><td></td></tr><tr><td>Layer 20 margin</td><td>0.329</td><td></td><td></td><td>2.056</td><td></td><td></td></tr><tr><td>Layer 21 margin</td><td>0.039</td><td></td><td></td><td>5.733</td><td></td><td></td></tr></table>

(2) Divergence-prone prompts “decide late”. The Agreed group commits to its final output token much earlier than the Diverged group: bf=final reaches 75.6% at layer 18 and 90.2% at layer 20 for Agreed, compared with only 33.9% and 39.0% for Diverged. Because Diverged prompts are precisely the ones whose ranking has not stabilised by the final body layer, they enter the lm\_head with a low margin already baked in. The lm\_head projection then either amplifies a stable ranking (Agreed: margin grows from 2.06 to 5.73, ∼2.8× amplification) or fails to stabilise an ambiguous one (Diverged: margin shrinks from 0.33 to 0.04, ∼8× compression). The lm\_head is therefore the stage at which late-decided prompts become precision-sensitive, but the underlying cause is that the upstream trajectory did not arrive at the lm\_head with a decisive preference.

(3) This is consistent with the graft experiment (Table 8). Grafting FP16 hidden states into the BF16 forward at any body layer ℓ ≤ 20 changes top-1 only ∼36% of the time precisely because BF16 and FP16 body hidden states already rank the same top token ∼97% of the time (Table 10) and the downstream BF16 kernels on either hidden state preserve that ranking. The +30pp jump at Layer 21 and Final RMSNorm corresponds exactly to the single layer at which the ranking actually changes. The two analyses (graft, logitlens traceback) triangulate the same conclusion from opposite directions: the flip is produced at the lm\_head; upstream body layers provide the setup (an undecided, low-margin trajectory) rather than the decision.

Cross-task replication. The per-layer top-1 agreement, margin, and commit curves are computed on all three benchmarks (GSM8K, HumanEval, MBPP) with n = 100 prompts each. The same pattern appears across these tasks: (i) BF16/FP16 body-layer top-1 agreement stays at 97–98% for both groups and drops only at the lm\_head, and only for Diverged prompts (to ∼90%); (ii) the margin gap opens only at the final layer, with Diverged-to-Agreed ratios of 10× (GSM8K), 14× (HumanEval), and 9× (MBPP); and (iii) Diverged prompts “decide late” while Agreed prompts commit earlier. This supports the mechanism on the tested math and code tasks; it does not establish task independence beyond them.

Joint interpretation. The three measurements form a causal chain that unfolds across the final transformer layers. The figures quoted in this paragraph are averages over the three benchmarks (GSM8K, HumanEval, MBPP); Table 10 reports the GSM8K-only values, which are sharper at the final layer (margin 0.329 → 0.039, top-1 agreement 96.6% → 76.3%). At layers 15–19 there is no diferentiation: both groups show low commit rates (< 20%), moderate margins (0.6–1.4), and near-perfect cross-precision top-1 agreement (97–98%). At layers 19–21 divergent trajectories emerge: Agreed prompts begin to commit (margin rises steeply to 3–4) as their hidden state aligns with a single dominant direction in the lm\_head weight space, whereas Diverged prompts see their margin decrease (from ∼1.4 at layer 19 to ∼1.0 at layer 21) as two candidate tokens compete; cross-precision agreement nonetheless remains $> 9 5 \%$ because the margin, while shrinking, is still > 40× the per-logit perturbation $\sigma _ { z } \approx 0 . 0 2 6$ . At the lm\_head the flip occurs: the final RMSNorm plus lm\_head projection acts as a diferentiator, amplifying margins for prompts that already committed $( 3  4 )$ and compressing margins for prompts still undecided $( 1 . 0  0 . 4 )$ . Once the margin drops to ∼0.4—within ∼15× of σ<sub>z</sub>—the BF16/FP16 arithmetic diference becomes comparable to the decision boundary, top-1 agreement falls from ∼97% to ∼90%, and the ∼10% of prompts that flip at this final step cascade into trajectory-level divergence. This three-stage progression—undiferentiated body → emerging competition → lm\_head-amplified flip—is consistent across all three tasks and directly implies that any intervention targeting layers before the lm\_head is premature: the decisive margin compression occurs only at the final projection, which is exactly where Intervention C operates.

## III Divergence-conditioned trace on Qwen2.5-3B-Instruct

The lm\_head-margin mechanism replicates on Qwen2.5-3B-Instruct (36 layers, d=2048, |V |=151,936): toptwo margin difers by 90× between groups (0.088 vs. 7.925, 54% vs. 0% flip rate). Body L2 is again indistinguishable.

The two model families support the core mechanism while showing that the body-layer pattern varies by model.

## IV Cross-model intervention replication

The headline intervention result (Table 3) is on TinyLlama-1.1B. To test whether the lift transfers across model families and scales, we run the full intervention comparison on four additional models covering three scales and three families: Llama-3.2-3B-Instruct, Qwen2.5-3B-Instruct, DeepSeek-R1-Distill-Qwen-7B, and Mistral-7B-Instruct-v0.3 (GSM8K, n=100, same setup and thresholds as the main experiment). For Mistral-7B we run only Baseline and Intervention C, because Interventions A and B have been shown to be inefective or redundant with C on the smaller models and doubling the Mistral-7B GPU hours would not change the story; the two-row comparison still gives a matched-pair Pass@1 test (reported in Appendix VI).

The cross-model picture. Table 12 summarises Intervention C’s lift across all five tested models, grouped into three tiers by efect size.

Interpretation. (1) Intervention C transfers across scales within the Llama family (Tier 1). The lift is actually larger on Llama-3.2-3B (+36pp) than on TinyLlama-1.1B (+22pp), ruling out the hypothesis that the method is a 1B-specific artefact. Both models admit a substantial fraction of prompts in the low-margin, single-step-fixable regime (category B in the per-prompt stratification, Table 4).

(2) Scale alone does not explain the split at 7B (Tiers 2 vs. 3). The two 7B models in our study difer sharply: Mistral-7B-Instruct-v0.3 shows a +8pp EAR lift under Intervention C, while DS-R1-Distill-Qwen-7B shows 0pp. Because both have the same nominal scale, scale alone cannot explain this contrast. The diference is consistent with their observed precision behaviour: the tested Qwen-family models at 1.5B and 7B produce 100% first-token divergence and FP16-body NaNs on GSM8K, placing their divergence before the lm\_head and outside Intervention C’s reachable scope. Mistral-7B exhibits neither pathology and retains a measurable head-correctable component. This comparison rejects the narrower generalisation that C necessarily fails at 7B, while leaving the underlying model-family cause as a hypothesis.

(3) Tier 2 shares a common profile: Intervention C reconciles a minority of divergences. Three models span Tier 2: Qwen2.5-3B (+3pp), Mistral-7B (+8pp), and OLMoE-1B-7B-MoE (+10pp). The interpretation is consistent across all three and aligns with the mechanism in Section 7.5: a portion of divergence originates in body-layer computation (beyond lm\_head scope) and cannot be repaired by any logit-level intervention; only divergences that manifest as late, low-margin lm\_head events are reachable by C. The OLMoE data point is particularly informative: the router over MoE experts is a potential extra divergence source, since the softmax-over-expert-logits can flip the chosen expert under BF16-vs-FP16 and propagate through diferent compute paths entirely, yet Intervention C still recovers a +10pp lift— comparable to dense Tier 2 models. This suggests that router flips are either rare on GSM8K at the scale we test or are themselves concentrated in low-margin steps that C’s gate already covers. For all three Tier 2 models, a body-scope method would need to be evaluated to close the residual gap.

Table 11: Intervention comparison across five additional models (GSM8K, n = 100, $\tau = 1 0 ^ { - 3 } )$ . The same ranking holds on every model where the method applies: Integer quantization (A) has zero efect, and Interventions B and C are comparable. Llama-3.2-3B exhibits an even larger absolute lift than the headline TinyLlama numbers $\mathrm { ( + 3 6 p p ~ v s ~ + 2 2 p p ) }$ . Qwen2.5-3B shows a much smaller lift (+3pp), reflecting Qwen’s training-time BF16-saturation pattern (Section 6.5). Mistral-7B-Instruct-v0.3, the non-Qwen dense 7B representative, gives a small but clearly non-zero lift (+8pp)—confirming that Intervention C transfers out of the Llama family and to 7B scale when the model is not BF16-saturated, and that the DS-R1-Qwen-7B zero-lift result is a Qwen-family artefact rather than a scale limit. OLMoE-1B-7B, a Mixture-of-Experts architecture (Apache 2.0, Allen AI) with 6.9B total / ∼ 1B active parameters per token, gives a +10pp lift, confirming that the method extends beyond dense transformers despite the router-over-experts softmax being a potential extra cross-precision divergence source.
<table><tr><td>Model</td><td>Strategy</td><td>EAR (%)</td><td>95% CI</td><td>∆ vs. Baseline</td></tr><tr><td rowspan="3">Llama-3.2-3B-Instruct</td><td>Baseline</td><td>31.0</td><td>[22.8, 40.6]</td><td></td></tr><tr><td>A. Integer quantization</td><td>31.0</td><td>[22.8, 40.6]</td><td>+0</td></tr><tr><td>B. Top-K FP32 C. Full FP32 lm_head</td><td>70.0</td><td>[60.4, 78.1]</td><td>+39 +36</td></tr><tr><td rowspan="3">Qwen2.5-3B-Instruct</td><td></td><td>67.0</td><td>[57.3, 75.4]</td><td></td></tr><tr><td>Baseline A. Integer quantization</td><td>18.0 18.0</td><td>[11.7,26.7] [11.7, 26.7]</td><td>+0</td></tr><tr><td>B. Top-K FP32</td><td>23.0</td><td>[15.8, 32.1]</td><td>+5</td></tr><tr><td rowspan="2"></td><td>C. Full FP32 lm_head</td><td>21.0</td><td>[14.2, 30.0]</td><td>+3</td></tr><tr><td>Baseline</td><td>51.0</td><td>[41.4, 60.6]</td><td></td></tr><tr><td rowspan="2"></td><td>C. Full FP32 lm_head</td><td>59.0</td><td>[49.2, 68.1]</td><td>+8</td></tr><tr><td>Baseline</td><td>34.0</td><td>[25.5, 43.7]</td><td></td></tr><tr><td rowspan="2">OLMoE-1B-7B-0924-Instruct (MoE)</td><td>C. Full FP32 lm_head</td><td>44.0</td><td>[34.7, 53.8]</td><td>+10</td></tr><tr><td>Baseline, A, B, C</td><td>0.0</td><td>[0.0, 3.7]</td><td>+0 (Appendix V)</td></tr></table>

DS-R1-Distill-Qwen-7B

Table 12: Intervention C lift across model families, scales, and architectures (GSM8K, n = 100), grouped into three tiers. Large-lift models (Tier 1) benefit strongly from C; small-lift but non-zero models (Tier 2) still benefit measurably and span three diferent families (Qwen dense, Mistral dense, OLMoE MoE), showing that Intervention C transfers to the 7B non-Qwen regime (Mistral-7B, +8pp) and to sparse MoE architectures (OLMoE, +10pp); the zero-lift tier (Tier 3) comprises only Qwen-family BF16-pretrained dense models, which produce FP16 NaNs in body layers and 100% first-token divergence on GSM8K (Section 6.5). We hypothesise that Intervention C’s efectiveness tracks the model’s training-time precision stability, rather than parameter count or dense/sparse architecture. The 7B dense tier is split cleanly by family: Mistral-7B gains +8pp, while DS-R1-Qwen-7B gains 0pp.
<table><tr><td>Model</td><td>Family</td><td>Baseline</td><td>C EAR</td><td>C lift</td></tr><tr><td>Tier 1: large lift</td><td></td><td></td><td></td><td></td></tr><tr><td>TinyLlama-1.1B-Chat</td><td>Llama</td><td>41%</td><td>63%</td><td>+22pp</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>Llama</td><td>31%</td><td>67%</td><td>+36pp</td></tr><tr><td>Tier 2: small but non-zero</td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen2.5-3B-Instruct</td><td>Qwen</td><td>18%</td><td>21%</td><td>+3pp</td></tr><tr><td>Mistral-7B-Instruct-v0.3</td><td>Mistral</td><td>51%</td><td>59%</td><td>+8pp</td></tr><tr><td>OLMoE-1B-7B-0924-Instruct (MoE)</td><td>MoE</td><td>34%</td><td>44%</td><td>+10pp</td></tr><tr><td>Tier 3: zero lift (BF16-saturated)</td><td></td><td></td><td></td><td></td></tr><tr><td>DS-R1-Distill-Qwen-7B</td><td>Qwen</td><td>0%</td><td>0%</td><td>0pp</td></tr></table>

(4) The same intervention ranking holds on every model where A, B, C were all run. Integer quantization (A) has exactly zero efect, and Interventions B and C give comparable lifts. The ranking A < B ≈ C is a robust empirical finding, independent of model family or scale.

![](images/7e00a775e2d6a326cb0a83ad8a37353224b800500e7e1fe8759edade3bc1988e.jpg)

![](images/d08695bfaa8c2167bc337f0227352a3ddfeafe684deb313b73710cc4eed38ffb.jpg)  
Figure 3: Applicability map for Intervention C. (a) Cross-model lift at bs= 1 for six models (including the MoE architecture OLMoE), coloured by tier (Table 12): Tier 1 (Llama family, large lift), Tier 2 (non-Qwen 7B, Qwen-3B, and OLMoE-MoE; small but positive), Tier 3 (Qwen BF16-saturated). Error bars are 95% CIs on the paired lift. (b) Batch-size scaling on TinyLlama-1.1B (Appendix VIII): the lift holds at bs≤ 4 (green band, where C alone is suficient in this experiment) and vanishes at bs≥ 8 (grey band, where composition with global FP32 compute recovers a small positive lift). Together, the panels summarise the tested (model, bs) region in which the lm\_head-only intervention helps.

Implications for deployment. Practitioners choosing between Intervention C and body-scope alternatives (e.g., LayerCast) should first measure the target model’s divergence rate and margin distribution on the target platform. Our cross-model results suggest three empirical regimes: (i) Tier 1 (the tested Llamafamily models at 1–3B), where C delivers +20 to +36pp EAR; (ii) Tier 2 (Qwen2.5-3B, Mistral-7B, and OLMoE), where C delivers smaller +3 to +10pp gains; and (iii) Tier 3 (the tested BF16-saturated Qwen variants), where C is inefective by itself. These tiers describe the observed models rather than a guaranteed family-level taxonomy. Appendix VIII further reports +17pp at bs= 1, +7–8pp at bs∈ {2, 4}, and 0pp at bs= 8; composition with global FP32 compute recovers +5pp at bs= 8 in the tested setup.

## V 7B-scale scope boundary: DeepSeek-R1-Distill-Qwen-7B

Setup. To test whether the gated FP32 lm\_head recomputation (Intervention C) transfers to a 7Bparameter model, we rerun the full intervention table (baseline, A, B, C) on DeepSeek-R1-Distill-Qwen-7B, GSM8K, n = 100, 256-token budget, $\tau = 1 0 ^ { - 3 }$ , and the same hardware/software stack as the main experiments. Because the model’s BF16 memory footprint is ∼14 GB, we run BF16 and FP16 in sequentia subprocesses to fit the 22 GB GPU budget; each precision–strategy combination takes ∼9.2s per prompt, for a total ablation runtime of ∼2 h.

Interpretation: this is a scope boundary, not a method failure. The 0% EAR across all four arms is fully consistent with the paper’s analysis and screening data, and we read it as concrete evidence for where Intervention C’s scope ends.

(1) The first-divergence step occurs earlier than the intervention’s correctable regime. Table 14 breaks down the first-divergence step t<sup>∗</sup> on the baseline BF16 vs. FP16 outputs. 100/100 prompts diverge, with a median $t ^ { * } = 6$ and $8 6 / 1 0 0$ prompts diverging within the first 9 tokens. This matches the pattern of Qwen2.5-7B-Instruct on the same benchmark (t<sup>∗</sup> median = 1, Section 6.5) and sits in the regime Appendix I explicitly identifies as uncorrectable by lm\_head-only FP32 recomputation: when divergence happens during the early generation window, the L-layer error accumulated inside the transformer body dominates the single-step lm\_head error that Intervention C targets, and the KV cache drifts irrecoverably within a few autoregressive steps.

Table 13: 7B intervention replication (DeepSeek-R1-Distill-Qwen-7B, GSM8K, $n { = } 1 0 0 , \tau { = } 1 0 ^ { - 3 } )$ . All four arms collapse to 0% EAR with identical Wilson CIs $[ 0 . 0 , 3 . 7 ] \%$ : every prompt diverges, and the interventions cannot recover. Latency columns show per-prompt BF16 / FP16 time; the ∼5% C overhead confirms the intervention is triggering, not silently skipped.
<table><tr><td>Strategy</td><td>Match</td><td>EAR (%)</td><td>95% CI</td><td>BF16 time (s)</td><td>FP16 time (s)</td></tr><tr><td>Baseline</td><td>0/100</td><td>0.0</td><td>[0.0, 3.7]</td><td>9.20</td><td>9.22</td></tr><tr><td>A (int quant)</td><td>0/100</td><td>0.0</td><td>[0.0, 3.7]</td><td>9.40</td><td>9.17</td></tr><tr><td>B (top-K FP32)</td><td>0/100</td><td>0.0</td><td>[0.0, 3.7]</td><td>9.55</td><td>9.30</td></tr><tr><td>C (full FP32 lm_head)</td><td>0/100</td><td>0.0</td><td>[0.0, 3.7]</td><td>9.67</td><td>9.21</td></tr></table>

Table 14: First-divergence step distribution on DeepSeek-R1-Distill-Qwen-7B (GSM8K, $n = 1 0 0$ baseline BF16 vs. FP16). Median $t ^ { * } = 6 ;$ 86% of prompts diverge in the first 9 tokens, well within the $^ { 6 6 } \mathrm { s t e p - 0 } ^ { \ 3 }$ regime of Appendix I.
<table><tr><td>First divergence bucket</td><td>Prompts</td></tr><tr><td> $t ^ { * } = 0$ </td><td>10</td></tr><tr><td> $t ^ { * } \in [ 1 , 4 ]$ </td><td>21</td></tr><tr><td> $t ^ { * } \in [ 5 , 9 ]$ </td><td>55</td></tr><tr><td> $t ^ { * } \in [ 1 0 , 2 9 ]$ </td><td>14</td></tr><tr><td> $t ^ { * } \geq 3 0$ </td><td>0</td></tr><tr><td>AGREED  $( t ^ { * }$  does not exist)</td><td>0</td></tr></table>

(2) The intervention is firing. Latency for Intervention C on BF16 rises from the baseline’s 9.20s to 9.67s (+5%), confirming that FP32 lm\_head recomputation is being triggered (the per-step top-two margin does drop below $\tau { = } 1 0 ^ { - 3 }$ on this model, consistent with the small-margin phenomenology in Section 7.5). What fails is not the trigger, but the correctability of the resulting flip: by the time a low-margin step arrives, the autoregressive path has already been driven onto a precision-specific KV cache. Appendix II’s finding that KV-cache swaps fix only 36–48% of flips even on TinyLlama is amplified here, because on this 7B model all 100 prompts sit on the cache-locked branch.

(3) This fits the paper’s first-token screening narrative. The model-sensitivity analysis already flags Qwen2.5-7B-Instruct and the Qwen-1.5B family as “catastrophic first-token divergence” models, and our 7B replication extends that list to DeepSeek-R1-Distill-Qwen-7B. These models share an observed inference-time pattern in our experiments: FP16 casting produces unstable, occasionally NaN intermediate activations, and the corresponding argmax trajectories bifurcate within a handful of decoding steps. We hypothesise that training-time precision exposure contributes to this pattern, but we do not verify the models’ training histories or establish that causal mechanism. For this class of models, an lm\_head-scoped intervention is not suficient; the present experiment establishes only that Interventions A–C cannot recover agreement. Our TinyLlama-1.1B and Qwen2.5-3B-Instruct results (Section 7.5, Appendix III), which do admit a non-empty Agreed group and do respond to Intervention C, characterise where the method applies; the 7B result here characterises where it does not.

Practical takeaway. Practitioners deploying a 7B-scale BF16-saturated model should not expect Intervention C (or any other inference-time lm\_head-scoped fix) to reconcile BF16-vs-FP16 outputs. The first reproducibility measure for such models is to pin the serving precision. If cross-precision reproducibility remains required, a body-scope approach such as LayerCast or precision-aware retraining would need to be evaluated; we do not test those alternatives on this model. Our main-text method is designed for—and evaluated on—the substantial class of models where the divergence is a late, small-margin event at the lm\_head, which is exactly what makes it a single-step-correctable event.

Table 15: Per-precision task accuracy on GSM8K (n=100, final-answer extraction). McNemar exact paired tests compare Baseline vs. Intervention C within each (model, precision) cell. No test rejects the null at any conventional level. Despite the Llama-3.2-3B family showing the largest EAR lift in the paper $\mathrm { ( + 3 6 p p }$ Table 11), its task accuracy is unchanged under Intervention C; despite Mistral-7B being the only 7B non-Qwen model we tested, its Pass@1 point estimates drop by 3–5pp but with non-significant McNemar p values (0.125 and 0.250) and substantially overlapping Wilson CIs. The six within-model tests (two per model) span EAR lifts from +3pp to +36pp and provide no evidence that Intervention C systematically shifts either trajectory toward a wrong answer.
<table><tr><td>Model</td><td>Config</td><td>Correct</td><td>Acc (%)</td><td>95% CI</td><td>vs. Baseline</td></tr><tr><td rowspan="4">Qwen2.5-3B</td><td>BF16, Baseline</td><td>27/100</td><td>27.0</td><td>[19.3, 36.4]</td><td rowspan="2"> $\Delta \mathrm { A c c } = + 2 \mathrm { p p } ~ ( p = 0 . 6 8 8 )$ </td></tr><tr><td>BF16, Intervention C</td><td>29/100</td><td>29.0</td><td>[21.0, 38.5]</td></tr><tr><td>FP16, Baseline</td><td>28/100</td><td>28.0</td><td>[20.1, 37.5]</td><td></td></tr><tr><td>FP16, Intervention C</td><td>27/100</td><td>27.0</td><td>[19.3, 36.4]</td><td> $\Delta \mathrm { A c c } = - \mathrm { 1 p p ~ } ( p = 1 . 0 0 0 )$ </td></tr><tr><td rowspan="4">Llama-3.2-3B</td><td>BF16, Baseline</td><td>24/100</td><td>24.0</td><td>[16.7, 33.2]</td><td></td></tr><tr><td>BF16, Intervention C</td><td>23/100</td><td>23.0</td><td>[15.8, 32.1]</td><td> $\Delta \mathrm { A c c } = - \mathrm { 1 p p ~ } ( p = 1 . 0 0 0 )$ </td></tr><tr><td>FP16, Baseline</td><td>20/100</td><td>20.0</td><td>[13.3, 28.9]</td><td></td></tr><tr><td>FP16, Intervention C</td><td>21/100</td><td>21.0</td><td>[14.2, 30.0]</td><td> $\Delta \mathrm { A c c } = + 1 \mathrm { p p } ~ ( p = 1 . 0 0 0 )$ </td></tr><tr><td rowspan="4">Mistral-7B</td><td>BF16, Baseline</td><td>43/100</td><td>43.0</td><td>[33.7, 52.8]</td><td></td></tr><tr><td>BF16, Intervention C</td><td>38/100</td><td>38.0</td><td>[29.1, 47.8]</td><td> $\Delta \mathrm { A c c } = - 5 \mathrm { p p ~ } ( p = 0 . 1 2 5 , + 1 / - 6 )$ </td></tr><tr><td>FP16, Baseline</td><td>40/100</td><td>40.0</td><td>[30.9, 49.8]</td><td></td></tr><tr><td>FP16, Intervention C</td><td>37/100</td><td>37.0</td><td>[28.2, 46.8]</td><td> $\Delta \mathrm { A c c } = - 3 \mathrm { p p ~ } ( p = 0 . 2 5 0 , + 0 / - 3 )$ </td></tr></table>

## VI Quality preservation: does Intervention C change task accuracy?

Setup. Intervention C reconciles BF16-vs-FP16 outputs at the lm\_head, but a natural concern is whether this comes at the cost of task accuracy: perhaps the intervention pushes both precisions toward the same wrong answer. We test this directly on GSM8K (n = 100) by running four configurations—BF16 baseline, FP16 baseline, $\mathrm { B F 1 6 + C , F P 1 6 + C }$ and extracting the final numerical answer from each generation. We use three testbeds spanning the three family/scale tiers of Table 12: Qwen2.5-3B-Instruct (Tier 2, Qwenfamily), Llama-3.2-3B-Instruct (Tier 1, largest EAR lift in the paper), and Mistral-7B-Instruct-v0.3 (Tier 2, non-Qwen 7B). All three models have GSM8K zero-shot accuracy well above the ∼3% noise floor we observe on TinyLlama,<sup>3</sup> giving matched-pair tests enough statistical power to detect small accuracy shifts.

Findings. (1) No statistically significant accuracy change in either direction, on any model. Under matched-pair McNemar tests, no (model, precision) cell rejects the null at any conventional level. For Qwen2.5-3B, BF16 goes $+ 4 / - 2 \ ( p = 0 . 6 8 8 )$ and $\mathrm { F P 1 6 + 1 / - 2 } \ ( p \mathrm { = 1 . 0 0 0 ) }$ . For Llama-3.2-3B, BF16 goes $+ 8 / - 9 \ ( p { = } 1 . 0 0 0 )$ and $\mathrm { F P 1 6 + 2 / - 1 } \left( p \mathrm { = } 1 . 0 0 0 \right)$ . For Mistral-7B, BF16 goes $+ 1 / - 6 \ ( p { = } 0 . 1 2 5 )$ and FP16 $+ 0 / - 3 \ ( p { = } 0 . 2 5 0 )$ ; both p-values sit above the 0.05 threshold but below 0.3, and the point-estimate drops (−5pp on BF16, −3pp on FP16) are worth a dedicated discussion (see finding (3) below). The paper’s +22pp (TinyLlama), +36pp (Llama-3.2-3B), +3pp (Qwen2.5-3B), and +8pp (Mistral-7B) EAR gains are therefore not artefacts of Intervention C systematically pushing both precisions toward the same wrong answer.

Table 16: Prompt-format robustness: the same four-arm experiment on Llama-3.2-3B-Instruct using its native chat template with a short system prompt. Absolute Pass@1 more than doubles (24% → 66% on BF16 baseline), confirming that the lower numbers in the quality-preservation results reflect prompt format rather than model capability. Both McNemar paired tests remain non-significant, and FP16 shows zero Pass@1 flips under Intervention C $\left( + 0 / \textrm { -- } 0 \right)$ : the intervention reconciles outputs exactly. The qualitypreservation conclusion is therefore invariant to prompt format.
<table><tr><td>Config</td><td>Correct</td><td>Acc (%)</td><td>95% CI</td><td>vs. Baseline</td></tr><tr><td>BF16, Baseline</td><td>66/100</td><td>66.0</td><td>[56.3, 74.5]</td><td></td></tr><tr><td>BF16, Intervention C</td><td>64/100</td><td>64.0</td><td>[54.2, 72.7]</td><td> $\Delta \mathrm { A c c } = - 2 \mathrm { p p ~ } ( p { = } 0 . 6 2 5 , + 1 / - 3 )$ </td></tr><tr><td>FP16, Baseline</td><td>66/100</td><td>66.0</td><td>[56.3, 74.5]</td><td></td></tr><tr><td>FP16, Intervention C</td><td>66/100</td><td>66.0</td><td>[56.3, 74.5]</td><td> $\begin{array} { r l } { \Delta \mathrm { A c c } { = } } & { { } \mathrm { 0 p p } \ ( p { = } 1 . 0 0 0 , + 0 / - 0 ) } \end{array}$ </td></tr></table>

(2) BF16 and FP16 baselines have essentially the same task accuracy within each model. On Qwen2.5-3B, BF16 baseline Pass@1 is 27% and FP16 baseline is 28%; on Llama-3.2-3B, BF16 is 24% and FP16 is 20%; on Mistral-7B, BF16 is 43% and FP16 is 40%. The within-model gaps (≤ 4pp) are within paired-sample uncertainty. Aggregate task accuracy therefore does not identify a uniformly preferable lowprecision arm on disagreeing prompts. The separate FP32-fidelity analysis (Appendix XIII) shows that FP16 is closer to the FP32 trajectory overall; Intervention C is intended to improve cross-format replay when the serving format cannot simply be changed.

(3) Efect-size patterns across tiers: large EAR lift with balanced churn (Llama-3.2-3B) and small EAR lift with an uncertain accuracy shift (Mistral-7B). Llama-3.2-3B (Tier 1, +36pp EAR) exhibits large but balanced within-precision correctness churn under C (+8/ − 9 on BF16). Mistral-7B (Tier 2, +8pp EAR) exhibits asymmetric small churn (+1/ − 6 on BF16, +0/ − 3 on FP16) that does not reach significance but points in one direction. The data are compatible with either sampling noise or a small negative accuracy efect; at n=100 we cannot distinguish them. Consequently, a deployment should evaluate both agreement and task quality on its own workload rather than assuming that a modest agreement gain justifies an uncertain quality trade-of. A larger-n Mistral-7B follow-up remains useful.

(4) Small-model baseline for comparison. We also ran the same experiment on TinyLlama-1.1B (Pass@1 around 3–4% for all four arms, n = 100). Matched-pair McNemar tests are again non-significant $\left( p > 0 . 3 \right)$ . However TinyLlama’s accuracy sits at the noise floor for n=100 and the tests have limited power; the Qwen2.5-3B, Llama-3.2-3B, and Mistral-7B rows of the quality-preservation results are the primary quality-preservation evidence.

(5) Implication for the main claim. The quality results show that Intervention C improves BF16-vs-FP16 output agreement (+22pp EAR on TinyLlama, Table 3; +36pp on Llama-3.2-3B, Table 11; with scaled agreement replication in Appendix X) without a statistically significant accuracy change on the four tested models. This is evidence of quality preservation in the tested conditions, not proof of zero efect: the Mistral point estimates and the finite sample sizes motivate workload-specific validation.

Prompt-format robustness check (Llama-3.2-3B). The Llama-3.2-3B Pass@1 numbers in the quality preservation results (20–24%) sit well below the model’s advertised GSM8K accuracy because the bareprompt format used throughout our cross-family experiments does not invoke the chat template against which Llama-3.2-3B-Instruct was tuned. To confirm that the quality-preservation conclusion does not depend on this choice, we repeat the four-arm experiment with Llama-3.2-3B’s native chat template (using a short system prompt: “You are a careful math tutor. Solve the problem step by step. End your response with a line that contains the final numeric answer only.”), keeping every other setting identical (same 100 GSM8K test items, greedy decoding, $\tau = 1 0 ^ { - 3 }$ , max\_new\_tokens=512).

The chat-template results show two patterns. First, the FP16 arm produces identical per-prompt correctness under Baseline and Intervention $\mathrm { ~ C ~ } ( + 0 / - 0 )$ : on the 66 prompts where FP16 answered correctly under baseline it still does so under C, and symmetrically for the incorrect prompts. The intervention therefore preserves which FP16 prompts are answered correctly while reconciling their token trajectories with the

Table 17: Long-reasoning sanity check on MATH-500 $( n = 5 0 ,$ , max\_new\_tokens = 1024, bare-prompt format). Mean generated length is 430–610 tokens per prompt, 3–4× longer than the main-text GSM8K chains. Baseline EAR remains close to the GSM8K values, and Intervention C has a positive point-estimate lift on both models. The per-cell Wilson intervals are descriptive rather than paired intervals for the lifts.
<table><tr><td>Model</td><td>Variant</td><td>EAR (%)</td><td>95% CI</td><td> $\operatorname { A v g } .$  len</td><td> $\mathrm { T r i g / s t e p }$ </td></tr><tr><td rowspan="2">TinyLlama-1.1B-Chat</td><td>Baseline</td><td>44.0</td><td>[31.2,57.7]</td><td>606</td><td>0.00</td></tr><tr><td>Intervention C</td><td>64.0</td><td>[50.1, 75.9]</td><td>564</td><td>0.82</td></tr><tr><td rowspan="2">Llama-3.2-3B-Instruct</td><td>Baseline</td><td>30.0</td><td>[19.1, 43.8]</td><td>438</td><td>0.00</td></tr><tr><td>Intervention C</td><td>56.0</td><td>[42.3, 68.8]</td><td>430</td><td>1.89</td></tr></table>

BF16 counterpart. Second, the BF16 arm shows a 2pp drop $( - 3 / + 1$ , McNemar $p { = } 0 . 6 2 5 )$ , well within noise and with substantially overlapping Wilson CIs. Both arms are individually and jointly consistent with no accuracy change. Combined with the main quality-preservation table, we therefore have quality-preservation evidence on (Qwen2.5-3B, Llama-3.2-3B, and Mistral-7B with bare prompt, plus Llama-3.2-3B with chat template), across four model / prompt-format conditions, none of which detect a significant accuracy efect from Intervention C.

## VII Long-reasoning sanity check: does divergence grow with chain length?

Motivation. The main-text GSM8K experiments cap generation at 256 tokens, which matches the typical reasoning-chain length for GSM8K problems. Modern LLM workloads, however, routinely generate chains of 500–4000 tokens (competition math, agentic planning, long-context tool use). A natural question is whether the 30–64% cross-precision divergence we report is a short-chain artefact that would either amplify or vanish under longer generations. The a priori prediction was amplification: longer chains contain more low-margin steps, each one an opportunity for BF16 and FP16 to fork, and a single fork cascades. This appendix tests that prediction directly on MATH-500, a harder benchmark (Hendrycks-style competition math) at ma $\mathrm { { ? } \_ n e w \_ t o k e n s = 1 0 2 4 }$

Setup. We draw the first $n = 5 0$ MATH-500 test problems deterministically. Prompt format is bare ("Problem:\n{problem}\n\nSolution:\n"), consistent with every other cross-family experiment in the paper—no chat template, no system prompt. We run baseline and Intervention C on TinyLlama-1.1B-Chat and Llama-3.2-3B-Instruct (the two Tier 1 models from Table 12) in both BF16 and FP16, batch size 1, fixed seed, A10G 22 GB.

Findings. (1) Baseline divergence does not amplify with chain length. Compared to the GSM8K numbers in Table 11 (TinyLlama baseline 41%, Llama-3.2-3B baseline 31%), the MATH-500 baselines are 44% and 30% respectively—both within 3pp of their GSM8K counterparts despite the mean chain length tripling (170 tokens on GSM8K → 438–606 tokens on MATH-500) and the benchmark dificulty increasing substantially. This rules out the “long chain cascades more” hypothesis at the lengths we test.

(2) The saturation is consistent with the margin-based mechanism. Section 7.5 shows that divergence is associated with the density of low-margin lm\_head steps rather than chain length alone. MATH-500 has more gated steps per prompt than GSM8K (trigger rate 1.89 vs. ∼1 on Llama-3.2-3B GSM8K), yet prompt-level divergence remains similar. Once a token difers, the autoregressive trajectories usually remain diferent; before that first difering token, additional decoding steps create further opportunities for a low-margin fork. Over the tested length range, the observed rates suggest that this prompt-level probability has already approached a plateau.

(3) Intervention C’s point-estimate lift persists at longer chains. The lifts are $+ 2 0 \mathrm { p p }$ (TinyLlama) and $+ 2 6 \mathrm { p p }$ (Llama-3.2-3B) on MATH-500, compared with $+ 2 2 \mathrm { p p }$ (TinyLlama, Table 3) and $+ 3 6 \mathrm { p p }$ (Llama-3.2-3B, Table 11) on GSM8K. The MATH-500 point estimates are positive and of similar order, but the n=50 per-cell Wilson intervals are descriptive and do not constitute paired confidence intervals for the lifts.

The Llama-3.2-3B point estimate is lower on MATH-500; determining whether this reflects task dificulty, trigger density, or sampling variation would require a larger paired evaluation.

(4) An incidental latency observation. On MATH-500, Llama-3.2-3B’s FP16 baseline wall-clock time (10.93s/prompt) is about 24% faster than its BF16 counterpart (14.46s/prompt). This is the first latency observation of this magnitude in our experiments and is not relevant to the intervention claim, but we record it here for transparency. The most plausible explanation is that SM 86 (A10G) has a somewhat faster FP16 FMA path than its BF16 FMA path, plausibly amplified by the longer generation lengths on MATH-500; we did not pursue this further.

Summary. The phenomenon and a positive Intervention C point-estimate lift both persist on chains that are 3–4× longer. The Wilson intervals are wider at $n { = } 5 0$ than in the n=100 GSM8K comparison, so these results support qualitative transfer rather than equality of efect sizes. Over the range explored, divergence does not increase with reasoning length alone.

## VIII Production-stack robustness

Table 18: Production-stack robustness checks (TinyLlama-1.1B, GSM8K).
<table><tr><td>Axis</td><td>Result</td><td>Implication</td></tr><tr><td>MoE (OLMoE)</td><td>100% BF16 self-consistency</td><td>Kernel non-det. ruled out</td></tr><tr><td>FlashAttention-2</td><td> $\mathrm { E A R \ - 1 p p ; \mathrm { C \ l i f t + 1 8 p p \ ( = \mathrm { e a g e r } ) } }$ </td><td>Consistent across both tested kernels</td></tr><tr><td>Batch size</td><td>C lift: +17/+7/+7/0pp (bs  $1 / 2 / 4 / 8 )$ </td><td>Bounded to bs≤4</td></tr><tr><td>Global FP32 compute + C</td><td>+5pp at bs=8; noise at bs=16</td><td>Composition viable</td></tr><tr><td>Overhead</td><td>+1.4% latency, +11% memory</td><td>Negligible cost</td></tr></table>

## IX Hyperparameter and intervention ablations

Table 19: Ablation summary.
<table><tr><td>Ablation</td><td>Result</td><td>Confirms</td></tr><tr><td>Scope: RMSNorm+lm_head</td><td>EAR drops 63%→44% EAR insensitive</td><td>lm_head-only optimal</td></tr><tr><td>Threshold  $\tau \in [ 1 0 ^ { - 4 } , 1 0 ^ { - 2 } ]$  Top-K:  $K \in \{ 2 , . . . , | V | \}$ </td><td>All identical EAR</td><td>Bimodal margin Rank-1-vs-2 only</td></tr><tr><td>Temperature sharpening</td><td>Zero effect</td><td>Scale-invariant ratio</td></tr><tr><td>Consensus decoding</td><td>100% at 2× cost</td><td></td></tr><tr><td></td><td></td><td>Trivial upper bound</td></tr></table>

## X Statistical validation and reproducibility

Wilson 95% CIs have half-width ±9–10pp at n=100 and describe the individual agreement rates. Paired bootstrap intervals describe the lifts; for example, the TinyLlama +22pp lift has a 90% CI of [+11, +33]pp. Scaled replication at n=300 (GSM8K), n=164 (full HumanEval), and n=257 (full MBPP) confirms all n=100 point estimates fall inside the scaled CIs; the scaled lifts are +16.4pp (GSM8K), +28.1pp (HumanEval), and +27.2pp (MBPP), each with McNemar $p < 0 . 0 0 1$ , with per-cell CI half-widths reduced to ±5.5–7.4pp.

## XI Reproducibility statement

Environment and methodology. The primary experiments use an NVIDIA A10G (22 GB, Ampere SM 86) under $\mathrm { P y }$ Torch 1.13 / 2.1 with CUDA 11.7. Cross-hardware replications use an NVIDIA L4 (24 GB, Ada SM 89) and A100-SXM4 (40 GB, Ampere SM 80); the FP16/FP32 probe uses a T4 (16 GB, Turing SM 75). We enable deterministic flags (torch.use\_deterministic\_algorithms(True), fixed CUBLAS workspace) and use greedy decoding (T=0) with fixed prompt selection. Models are loaded from Hugging Face Hub without checkpoint modification, except for the explicitly described runtime precision transformations. EAR compares complete token-id sequences; task-level agreement uses final-answer extraction for GSM8K and execution outcomes for HumanEval. Confidence intervals are Wilson score intervals unless otherwise stated, and bootstrap CIs use 10,000 resamples. The prompt formats, sample sizes, hardware exceptions, and intervention hyperparameters $\scriptstyle ( \tau = 1 0 ^ { - 3 }$ for $\mathrm { B F 1 6 / F P 1 6 }$ and K=8 for Intervention B) are reported in the corresponding sections; the FP8 threshold is separately calibrated as described in Appendix XV.

## XII Directional body-error analysis

To test whether body-layer error contributes directionally to flips, we project the BF16-vs-FP16 logit-error vector onto the top-two decision direction $d = e _ { v ^ { ( 1 ) } } - e _ { v ^ { ( 2 ) } }$ . For the empirical top-two decision rule evaluated here, we compare the magnitude of this projection with the smaller top-two margin across the two arms; exceeding that margin predicts a flip. This rule approximates the exact all-competitor condition in Proposition 1. Note on populations: this analysis measures margins at the first-divergence step $t ^ { * }$ of each prompt (median 0.0 for Diverged), whereas Table 7 reports margins at each group’s fixed measurement step under the divergence-conditioned trace protocol (150× gap); the two tables therefore quote diferent statistics of diferent step populations, and both show the same qualitative separation of two-plus orders of magnitude.

Table 20: Directional body-error analysis (GSM8K, n=100 per model). The mean body projection onto the top-two direction is indistinguishable between Diverged and Agreed steps on both models, confirming that body error does not preferentially point toward the decision boundary for flipping steps. The margin, not the directional body projection, separates the two groups.
<table><tr><td>Model</td><td>Quantity</td><td>DIVERGED</td><td>AGREED</td><td>Interpretation</td></tr><tr><td rowspan="4">TinyLlama-1.1B</td><td>Hidden-state L2 ∥el|</td><td>0.999</td><td>1.052</td><td>Indistinguishable</td></tr><tr><td>Body projection (mean ± SD)</td><td>0.025 ± 0.023</td><td>0.026 ± 0.020</td><td>Equiv. (p=0.014)</td></tr><tr><td>Signed dir. perturb.  $\Delta z ^ { ( 1 ) } - \dot { \Delta z ^ { ( 2 ) } }$ </td><td>0.039</td><td>0.036</td><td>Indistinguishable</td></tr><tr><td>Top-two margin (mean) Top-two margin (median)</td><td>0.015 0.0</td><td>3.93 2.69</td><td>260× gap</td></tr><tr><td>Qwen2.5-3B</td><td>Body projection (mean ± SD) Top-two margin (mean)</td><td> $0 . 0 5 1 \pm 0 . 0 7 0$  0.035</td><td> $0 . 0 4 9 \pm 0 . 0 4 3$  6.05</td><td>Equiv. (p=0.006) 170× gap</td></tr></table>

The body-error null is formally established, not merely unrejected. Because a null result carries weight only if it is powered, we test equivalence rather than relying on a failure to reject. On TinyLlama-1.1B the diference in body projection between Diverged and Agreed is −0.0013 (95% CI [−0.0099, +0.0073]; Welch $t = - 0 . 3 0 , p = 0 . 7 7 )$ ), and the two one-sided tests (TOST) procedure, with an equivalence margin of half a pooled SD (0.011 for TinyLlama, 0.032 for Qwen), rejects non-equivalence at $p = 0 . 0 1 4$ . On Qwen2.5- 3B the diference is +0.0015 (95% CI [−0.0216, +0.0247]; t = +0.13, p = 0.90), with TOST $p = 0 . 0 0 6$ . We use a half-SD margin as a conventional medium efect size; the point estimates difer by $< 0 . 0 6 ~ \mathrm { S D }$ , far inside it. So the two groups are statistically equivalent on the decision-direction projection, while their top-two margins difer by two to three orders of magnitude (0.015 vs. 3.93; 0.035 vs. 6.05). The asymmetry between these two comparisons is the substantive content of this appendix.

A note on what these accuracies measure. Proposition 1 is exact, so it cannot have an error rate; the quantity it involves—the per-coordinate perturbation at the step in question—is not observable before the step is computed. What we evaluate here is therefore a decision rule derived from it: predict a flip when $| \Delta z ( v ^ { ( 1 ) } ) - \Delta z ( v ^ { ( 2 ) } ) |$ exceeds the smaller of the two arms’ margins, thresholding the magnitude $\bar { \vert \Delta z ( v ^ { ( 1 ) } ) - }$ $\Delta z ( v ^ { ( 2 ) } ) |$ against the margin. Under this rule the directional statistic classifies flip-vs-no-flip correctly on 88% of steps for TinyLlama-1.1B and 89% for Qwen2.5-3B, versus step-level base rates of 63%/55% and versus 86%/83% for the RMS statistic it replaces. (The magnitude, not the signed diference, is the right predictor here: the signed rule scores only 0.57/0.59, because a flip can be triggered by a perturbation of either sign relative to the current top-1.) The exact condition of Proposition 1 is proved; this decision rule is the deployable approximation to it, and the residual error is why we label the scale analysis heuristic. The intervention’s own gate does not use this statistic at all—it fires on the observable margin $\Delta _ { t } < \tau ;$ the decision rule here is only an analysis tool for attributing flips. This confirms that (i) the lm\_head margin is the dominant factor, (ii) body error does not preferentially align with the decision boundary, and (iii) the directional formulation of Proposition 1 is empirically supported.

## XIII FP32 fidelity of Intervention C

To verify that Intervention C pushes outputs toward the FP32-oracle trajectory (rather than toward an arbitrary third path), we measure agreement between Intervention C outputs and FP32 greedy outputs, comparing against the BF16-baseline-to-FP32 agreement rate.

Table 21: FP32 reference fidelity (GSM8K, n=100). Three metrics: (1) sequence-level agreement with the FP32 oracle, (2) repaired-token agreement with the FP32 argmax at triggered steps, and (3) downstreamcorrectness alignment: the fraction of FP32-correct prompts that are also answered correctly under each method. Intervention C raises sequence agreement with FP32 by +19–21pp. Brackets on the sequenceagreement rows are 95% CIs for the lift; on the repaired-token rows they are Wilson 95% CIs for the proportion.
<table><tr><td>Model</td><td>Metric</td><td>BF16→FP32</td><td>FP16→FP32</td><td>IntC→FP32</td></tr><tr><td rowspan="3">TinyLlama-1.1B</td><td>Sequence agreement</td><td>42%</td><td>90%</td><td>63% (+21pp) [+7, +35]</td></tr><tr><td>Repaired-token = FP32 argmax Correctness alignment w/ FP32</td><td>50%</td><td>100%</td><td>63% (97/154) [55, 70] 50%</td></tr><tr><td>Sequence agreement</td><td>23%</td><td>82%</td><td>42% (+19pp) [+6, +32]</td></tr><tr><td rowspan="2">Qwen2.5-3B</td><td>Repaired-token = FP32 argmax</td><td></td><td></td><td>51% (132/260) [45,57]</td></tr><tr><td>Correctness alignment w/ FP32</td><td>25%</td><td>92%</td><td>50%</td></tr></table>

On both models, Intervention C substantially increases FP32 fidelity: sequence agreement rises from 42% to 63% on TinyLlama (+21pp) and from 23% to 42% on Qwen2.5-3B (+19pp). At triggered steps, 63% (TinyLlama) and 51% (Qwen) of repaired tokens match the FP32 argmax. Downstream-correctness alignment with FP32 doubles on Qwen2.5-3B (25% → 50%); on TinyLlama the alignment is unchanged at 50%, though we caution that only 2 of 100 prompts are FP32-correct on this model (GSM8K accuracy ∼2%), so this cell carries little statistical weight. FP16 already shows high FP32 fidelity (90%/82%) because FP16’s 10-bit mantissa introduces smaller rounding errors; Intervention C moves BF16 outputs measurably toward this higher-precision reference rather than toward an arbitrary consensus. Two qualifications are important. First, FP16 alone attains higher FP32 fidelity than BF16+C: if maximal FP32 fidelity is the objective and the serving format is free, the appropriate action is to serve FP16. Our claim is narrower—within a fixed BF16 deployment, the intervention moves outputs toward the FP32 reference. Second, because a repaired step is a choice between exactly two candidates, the chance level for repaired-token agreement is 50%: TinyLlama’s 63% (CI [55, 70]) is above chance, whereas Qwen’s 51% (CI [45, 57]) is not distinguishable from it. Per-token fidelity is therefore established on TinyLlama only; on Qwen the sequence-level (+19pp) and correctness-alignment (25% → 50%, on 12 FP32-correct prompts) gains are the operative evidence.

## XIV Cross-architecture replication: L4 (Ada), A100 (Ampere), and T4 (Turing)

To verify that the mechanism and intervention are not A10G-specific, we replicate the headline experiment on an NVIDIA L4 (Ada Lovelace, SM 89, 24 GB) and an NVIDIA A100-SXM4-40GB (Ampere, SM 80), and additionally probe the Turing generation via an NVIDIA T4 (SM 75, 16 GB).

The intervention transfers to every microarchitecture we tested, but the magnitude of its benefit does not, and we state that plainly. Two readings follow from the three-GPU comparison.

First, the direction of the efect is robust; the size is not. All six model×GPU cells show a positive lift, but TinyLlama ranges from +22pp (A10G) to +12pp (A100) and Qwen2.5-3B from +3pp (A10G) to +20pp (L4). The headline +22–36pp figures reported elsewhere in this paper are therefore A10G measurements and should be read as one point in a hardware-dependent range, not as architecture-independent constants.

Table 22: Cross-architecture replication (GSM8K, $n { = } 1 0 0 , \tau { = } 1 0 ^ { - 3 } )$ . The intervention yields a positive lift on all three GPU microarchitectures tested; the magnitude is hardware-dependent, and A10G is the most favourable card for TinyLlama and the least favourable for Qwen2.5-3B.
<table><tr><td>Model</td><td>GPU (architecture)</td><td>Baseline EAR (%)</td><td>IntC EAR (%)</td><td>Lift</td></tr><tr><td rowspan="3">TinyLlama-1.1B</td><td>A10G (Ampere, SM 86)</td><td>41</td><td>63</td><td>+22pp</td></tr><tr><td>L4 (Ada, SM 89)</td><td>36</td><td>57</td><td>+21pp</td></tr><tr><td>A100 (Ampere, SM 80)</td><td>39</td><td>51</td><td>+12pp</td></tr><tr><td rowspan="3">Qwen2.5-3B-Instruct</td><td>A10G (Ampere, SM 86)</td><td>18</td><td>21</td><td>+3pp</td></tr><tr><td>L4 (Ada, SM 89)</td><td>30</td><td>50</td><td>+20pp</td></tr><tr><td>A100 (Ampere, SM 80)</td><td>20</td><td>34</td><td>+14pp</td></tr></table>

Second, the A10G result is the outlier in this three-GPU comparison. Qwen’s lift is +3pp on A10G, +14pp on A100, and +20pp on L4. Because A100 (SM 80) and A10G (SM 86) are both Ampere yet difer by 11pp, and because the two non-A10G cards agree more closely with each other, the variation cannot be attributed simply to the Ada architecture or the Qwen checkpoint. One possible mechanism is that A10G’s FP16 kernel path pushes more activations outside the low-margin, lm\_head-reachable regime; we have not instrumented the kernels to test this explanation.

The practical consequence is that the tier assignments and the CV(TD) screening metric elsewhere in this paper are calibrated on A10G and require per-platform recalibration; Qwen2.5-3B is Tier 2 on A10G but would be Tier 1 on L4.

A third generation: T4 (Turing). The T4 cannot run this paper’s primary comparison at all: Turing (SM 75) predates native BF16 support, which begins with Ampere (SM 80). Running BF16 there would measure software emulation rather than hardware format behaviour, and FP8 requires SM 89. The BF16/FP16 comparison therefore applies to Ampere-and-later serving stacks; on T4, FP16-vs-FP32 is the meaningfu cross-precision pair, which we ran (TinyLlama-1.1B, GSM8K, n=100, $\tau { = } 1 0 ^ { - 3 } )$ :

Table 23: T4 (Turing, SM 75) FP16-vs-FP32 replication. Because FP32 holds the original weights, this pair carries no weight-truncation floor, and baseline agreement is correspondingly far higher than in the BF16-vs-FP16 setting (88% vs. 41%). Gated FP32 lm\_head recomputation still yields a positive lift while triggering on only 20 steps.
<table><tr><td>Precision pair</td><td>Baseline EAR (%)</td><td>IntC EAR (%)</td><td>Lift</td></tr><tr><td>FP16 vs. FP32 (T4)</td><td>88</td><td>95</td><td>+7pp</td></tr></table>

Two observations. First, the much higher baseline agreement is what the storage-vs-computation decomposition predicts: FP32 is the untruncated weight, so the ∼12pp irreducible floor identified for BF16-vs-FP16 (Section 7.6) does not apply here, and the residual disagreement is arithmetic-only. Second, the intervention remains efective in this regime (+7pp from 20 triggered steps), so the low-margin mechanism is not an artefact of any single format pair or microarchitecture: across Turing, Ampere, and Ada, and across three precision pairs (FP16/FP32, BF16/FP16, BF16/FP8), gated FP32 lm\_head recomputation improves exact agreement. We report a single model here, as the T4’s 16 GB limits FP32-arm capacity.

## XV FP8 replication and threshold scaling

To test whether the mechanism and intervention extend beyond the BF16/FP16 pair to a lower-precision serving format, we replicate the headline experiment with FP8. We run on an NVIDIA L4 GPU (Ada Lovelace, compute capability 8.9, natively FP8-capable) and apply FP8-E4M3 with per-tensor dynamic scaling to the lm\_head projection, keeping the transformer body in BF16 in both arms so that the comparison isolates the head-projection format—consistent with the finding that the lm\_head is the decisive locus (Section 7). The setup otherwise matches the main experiments: GSM8K, n=100, greedy decoding, 256- token budget, on TinyLlama-1.1B-Chat and Qwen2.5-3B-Instruct.

Table 24: Baseline FP8 divergence (GSM8K, n=100; lm\_head in FP8-E4M3 vs. BF16, transformer body in BF16 in both arms). FP8 divergence is substantially more severe than the FP16 setting (70% vs. 59% of prompts on TinyLlama; 93% vs. 75% on Qwen), and flips still concentrate at low-margin steps. The FP16 reference column uses the same bare-prompt protocol as this experiment (see the cross-task appendix footnote for the protocol diference from the main-text chat-template numbers).
<table><tr><td>Model</td><td>EAR (BF16 vs FP8)</td><td>EAR (BF16 vs FP16) [reference]</td><td>First-div margin (median)</td></tr><tr><td>TinyLlama-1.1B</td><td>30%</td><td>41%</td><td>0.0625</td></tr><tr><td>Qwen2.5-3B</td><td>7%</td><td>25%</td><td>0.125</td></tr></table>

FP8’s 3-bit mantissa produces larger rounding perturbations than FP16’s 10 bits, so divergence rises on both models; yet the first-divergence margins remain small relative to the format’s perturbation scale, showing that the low-margin lm\_head pattern also appears in the tested head-isolated BF16/FP8 setting.

Threshold scaling. The FP16-calibrated trigger threshold $\tau = 1 0 ^ { - 3 }$ fails under FP8: the gate opens on only 3 of ∼25,600 decoding steps, and the lift (−1pp) is within noise. This failure is predicted by the scale analysis accompanying Proposition 1: FP8-E4M3’s per-logit perturbation scale is roughly 2–5× the FP16 scenario’s $\sigma _ { z } ~ \approx ~ 0 . 0 2 6$ , so a threshold calibrated to the FP16 perturbation scale sits far below the margins at which FP8 flips occur. The trigger threshold must scale with the format’s perturbation scale. Rescaling τ recovers and then completes the repair (TinyLlama, both-arms gated protocol identical to the main experiments):

Table 25: Threshold sweep for gated FP32 lm\_head recomputation under FP8 (GSM8K, n=100 per model, both-arms gated protocol). The FP16-calibrated $\tau = 1 0 ^ { - 3 }$ triggers almost never on either model; rescaling τ to the FP8 perturbation scale monotonically recovers agreement on both, reaching exact agreement on TinyLlama (100%) and 63% on Qwen at $\tau = 0 . 2 5$ , each at a ∼4% trigger rate. Bracketed values are Wilson 95% CIs; the τ = 0.25 lifts are +70pp (CI [+61, +79]) and +56pp (CI [+45, +67]).
<table><tr><td rowspan="2">T</td><td colspan="3">TinyLlama-1.1B</td><td colspan="3">Qwen2.5-3B</td></tr><tr><td>EAR (%)</td><td>Lift (pp)</td><td>Trig. steps</td><td>EAR (%)</td><td>Lift (pp)</td><td>Trig. steps</td></tr><tr><td> $1 0 ^ { - 3 }$ </td><td>29 [21, 39]</td><td>-1</td><td>3</td><td>10 [6, 17]</td><td>+3</td><td>4</td></tr><tr><td>0.02</td><td>37 [28, 47]</td><td>+7</td><td>73</td><td>13 [8, 21]</td><td>+6</td><td>94</td></tr><tr><td>0.05</td><td>48 [38, 58]</td><td>+18</td><td>201</td><td>16 [10, 24]</td><td>+9</td><td>197</td></tr><tr><td>0.1</td><td>77 [68,84]</td><td>+47</td><td>378</td><td>27 [19, 36]</td><td>+20</td><td>396</td></tr><tr><td>0.25</td><td>100 [96, 100]</td><td>+70</td><td>1010 (~4%)</td><td>63 [53, 72]</td><td>+56</td><td>994 (~4%)</td></tr></table>

Why the threshold must scale: the flip-margin distribution moves with the format. The scale analysis predicts that flips concentrate where the margin is small relative to the format’s perturbation scale— so a more aggressive format should produce flips at correspondingly larger margins. This is directly observable in the first-divergence margins:

The shift is quantitative, not merely directional: relative to the FP16-setting scale $\sigma _ { z } \approx 0 . 0 2 6$ , the FP8 flip-margin medians sit at 2.4× (TinyLlama) and 4.8× (Qwen) that value, consistent with the 2–5× larger per-logit perturbation of FP8-E4M3. The gate must widen by a comparable factor, which is what the sweep finds empirically (τ from $1 0 ^ { - 3 }$ to 0.25). We note this is a two-format comparison; whether the relationship is proportional across a wider range of formats is untested.

At $\tau = 0 . 2 5$ the intervention achieves exact agreement (EAR 100%) on TinyLlama at a ${ \sim } 4 \%$ trigger rate— comparable overhead to the FP16 setting. The rescaling behaviour replicates on Qwen2.5-3B: the same sweep recovers monotonically from +3pp at $\tau = 1 0 ^ { - 3 }$ to $+ 5 6 \mathrm { p p }$ at $\tau = 0 . 2 5$ (EAR 7% → 63%, at a comparable ∼4% trigger rate), though it does not reach exact agreement. Note that the explanation cannot be bodyoriginated divergence, which this design excludes: with identical BF16 bodies in both arms the shortfall must come from gate coverage. Qwen’s FP8 flip-margin distribution has a heavier tail (median 0.125 versus TinyLlama’s 0.0625, Table 26), so $\tau = 0 . 2 5$ leaves more of its flippable steps ungated; a larger τ would be needed, at proportionally higher trigger cost.

Table 26: Top-two margin at the first divergence step, by format pair (GSM8K, n=100 per cell). Under FP8 the flips occur at margins 2.4–4.8× larger than under FP16, matching the larger per-logit perturbation of the 3-bit-mantissa format and explaining why a threshold calibrated for FP16 $( \tau = 1 0 ^ { - 3 } )$ almost never fires under FP8.
<table><tr><td>Format pair</td><td>Model</td><td>Flip-margin median</td><td>Flip-margin mean</td></tr><tr><td>BF16 vs. FP16</td><td>TinyLlama-1.1B</td><td>0.000</td><td>0.015</td></tr><tr><td>BF16 vs. FP16</td><td>Qwen2.5-3B</td><td>0.000</td><td>0.035</td></tr><tr><td>BF16 vs. FP8</td><td>TinyLlama-1.1B</td><td>0.0625</td><td>0.052</td></tr><tr><td>BF16 vs. FP8</td><td>Qwen2.5-3B</td><td>0.1250</td><td>0.126</td></tr></table>

End-to-end FP8: where the method’s reach ends. The configuration above deliberately isolates the head projection. We also ran the deployment-shaped variant, quantising every nn.Linear in the model to FP8-E4M3 (155 modules; TinyLlama-1.1B, GSM8K, n=100). Divergence becomes far more severe — EAR 10%, i.e. 90% of prompts diverge — and the gated repair recovers only +1pp. Widening the gate does not help: $\tau = 0 . 1$ and $\tau = 0 . 2 5$ trigger on 321 and 738 steps respectively and both yield the same +1pp. This is the boundary the error model predicts rather than a contradiction of it: with all linear layers quantized, divergence is dominated by body-originated error accumulated across the forward pass, which lies outside the reach of any lm\_head-scope repair. It is the same limit we report for BF16-saturated Qwen variants (Appendix V) and for bs≥8 batched decode (Appendix VIII): where lm\_head-originated divergence is not the dominant source, composition with a body-scope method is required. The contrast between the two FP8 configurations (+70pp when the body is held fixed, +1pp when it is not) is consistent with the headdominance estimate of Section 7: when the body is quantized too, the body term the estimate treats as small (∼2.4% under BF16/FP16) grows until it dominates, and a head-scope repair can no longer reach it.

The two FP8 configurations establish diferent claims. Because the head-isolated comparison shares BF16 weights and a BF16 transformer body, the hidden state entering the head is identical across arms; this removes both the measured BF16/FP16 weight-storage gap and body-originated divergence by construction. It therefore tests transfer of (i) the low-margin lm\_head mechanism, (ii) the margin-gated repair, and (iii) format-dependent threshold calibration to FP8 head arithmetic. The end-to-end experiment quantises every linear layer and is the deployment-shaped stress test included here. Its +1pp result shows that the head-only method is largely inefective once FP8 error accumulates through the body. We do not evaluate FP8 KV-cache quantisation or claim an end-to-end FP8 reproducibility solution.

## XVI Execution-based correctness (HumanEval)

We run the OpenAI HumanEval execution sandbox on all four arms (BF16 baseline, FP16 baseline, BF16+C, FP16+C) using TinyLlama-1.1B (n=164, full benchmark).

The 68% token-level divergence produces zero execution-correctness flips: the divergent code completions are either both wrong or both right (in all observed cases, both wrong—TinyLlama’s code accuracy is low). Intervention C introduces zero pass@1 degradation in this run, but the near-zero baseline accuracy makes this a weak quality-preservation test; the Qwen experiment below is the informative execution-based result.

A capable model: Qwen2.5-3B-Instruct. TinyLlama’s near-zero code accuracy makes its “zero flips” result weak evidence: two arms that are both always wrong agree trivially. We therefore repeated the experiment on Qwen2.5-3B-Instruct, which solves half of HumanEval, so that agreement is measured over a population where correctness genuinely varies (Table 28).

Table 27: Execution-based pass@1 on HumanEval (TinyLlama-1.1B, n=164). Despite 68% token-level divergence between BF16 and FP16, execution correctness is identical across all configurations: zero correctness flips.
<table><tr><td>Config</td><td>pass@1 (%)</td><td>Correctness flips vs. BF16 baseline</td></tr><tr><td>BF16 baseline</td><td>0.6</td><td></td></tr><tr><td>FP16 baseline</td><td>0.6</td><td>0</td></tr><tr><td> $\mathrm { B F 1 6 + I n t C }$ </td><td>0.6</td><td>0</td></tr><tr><td> $\mathrm { F P 1 6 + I n t C }$ </td><td>0.6</td><td>0</td></tr><tr><td>Token-level divergence (BF16 vs FP16)</td><td colspan="2">68%</td></tr></table>

Table 28: Execution-based pass@1 on HumanEval (Qwen2.5-3B-Instruct, $n { = } 1 6 4$ , full benchmark). At a pass@1 of ∼50%, the two arms disagree on 73.8% of completions at the token level, yet agree on 97.6% of execution outcomes. Intervention C does not degrade pass@1.
<table><tr><td>Config</td><td>pass@1 (%)</td><td>Correctness flips vs. BF16 baseline</td></tr><tr><td>BF16 baseline</td><td>50.0</td><td></td></tr><tr><td>FP16 baseline</td><td>52.4</td><td>4 of 164 (2.4%)</td></tr><tr><td> $\mathrm { B F 1 6 + I n t C }$ </td><td>51.2</td><td>— (+1.2pp vs. BF16)</td></tr><tr><td>Token-level divergence (BF16 vs FP16)</td><td></td><td>73.8%</td></tr><tr><td>Execution-outcome agreement</td><td></td><td>97.6%</td></tr></table>

Three observations. First, the token-level/outcome-level gap is large and in the direction our discussion of practical adequacy predicts: 73.8% of completions difer somewhere in their token sequence, but 97.6% of them execute to the same verdict, so most cross-precision divergence on this task is not correctness-bearing. Second, unlike TinyLlama, the flips here are not zero: 4 of 164 problems flip, all in the same direction (FP16 correct, BF16 incorrect; exact binomial $\scriptstyle p = 0 . 1 2 5$ , not significant at $n { = } 1 6 4 )$ . We report the direction rather than suppress it, but with 4 discordant pairs the data cannot support a claim that either format is systematically better on code, and the aggregate 2.4pp pass@1 diference is well inside the ±5pp noninferiority margin used elsewhere in this paper. Third, Intervention C’s pass@1 (51.2%) is above the BF16 baseline it repairs toward (50.0%), so the gated recomputation does not trade correctness for reproducibility on this benchmark either.

Across the two models, execution-outcome agreement greatly exceeds token-level agreement: 100% vs. 32% on TinyLlama and 97.6% vs. 26.2% on Qwen2.5-3B.

## XVII Cross-task EAR on Qwen2.5-3B-Instruct

To verify that the phenomenon generalises beyond GSM8K, we report EAR on two additional code benchmarks for Qwen2.5-3B-Instruct.

Table 29: Cross-task EAR for Qwen2.5-3B-Instruct $( n { = } 1 0 0 , \tau { = } 1 0 ^ { - 3 } )$ . Divergence rates are consistent across math and code tasks.
<table><tr><td>Task</td><td>EAR (%)</td><td>Divergence (%)</td><td>Median first-div step</td></tr><tr><td>GSM8K</td><td>25</td><td>75</td><td></td></tr><tr><td>HumanEval</td><td>37</td><td>63</td><td>step 83</td></tr><tr><td>MBPP</td><td>40</td><td>60</td><td>step 43</td></tr></table>

Both code benchmarks show 60–63% divergence on Qwen2.5-3B-Instruct, comparable to the 75% GSM8K divergence measured under the same protocol. The slightly higher EAR on code tasks is consistent with their shorter average generations and fewer observed low-margin steps per prompt.<sup>4</sup>