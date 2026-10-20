# LIRQuant: An Efficient Locally Isolated Rotation Quantization Method for Large Language Models

Desheng Xu and Hanglin Li and Tianchen Song and Zhiyong Zhao and Kun Zhan Li Auto Inc.

{xudesheng,lihanglin,songtianchen,zhaozhiyong,zhankun}@lixiang.com

## Abstract

Post-Training Quantization (PTQ) enables efficient deployment of Large Language Models (LLMs), yet activation outliers remain a key challenge for low-bit quantization. Existing global transformation methods overlook heterogeneous channel sensitivity to quantization noise, leading to suboptimal accuracyefficiency trade-offs. To address this, we propose Locally Isolated Rotation Quantization (LIRQuant), a sensitivity-aware framework that selectively applies transformations. LIRQuant identifies sensitive channels using a noise-weighted sensitivity metric, applying expressive learnable rotations only to these channels and lightweight fixed rotations to the remaining ones. This decoupled design reduces optimization complexity without sacrificing outlier smoothing capability. Extensive experiments across the LLaMA-2/3, Mistral, Qwen3, and Mixtral families demonstrate that LIRQuant achieves state-of-the-art accuracy under the challenging W4A4KV4 setting. Meanwhile, LIRQuant calibrates LLaMA-3- 8B in only 25 minutes with 20.3 GB peak GPU memory, achieving the fastest calibration and the lowest memory footprint among recent learnable-rotation methods, and its fused inference kernels deliver up to 2.3× prefill speedup and 3.5× memory saving over FP16.

## 1 Introduction

Large Language Models (LLMs) have achieved remarkable capabilities across a wide range of tasks (Brown et al., 2020; Hendrycks et al., 2020). Models such as LLaMA (Touvron et al., 2023), GPT (Achiam et al., 2023), and Qwen (Yang et al., 2025) continue to drive progress toward universal language intelligence. However, their massive parameter counts incur substantial memory and computational costs, posing significant challenges for deployment on resource-constrained devices (Samsi et al., 2023; Fernandez et al., 2025; Luccioni et al., 2024). Post-Training Quantization (PTQ) addresses this issue by converting weights and activations into low-precision formats, thereby reducing memory footprint and inference cost with minimal performance degradation (Frantar et al., 2022; Shao et al., 2023; Ashkboos et al., 2024a). Consequently, PTQ has emerged as a critical technique for practical LLM deployment.

Activation outliers remain a major obstacle to low-bit quantization of LLMs (Dettmers et al., 2022; Xiao et al., 2023; Lin et al., 2024b). Recent studies mitigate activation outliers through various linear transformations, including orthogonal rotations and Hadamard transforms, which smooth activation distributions and improve quantization robustness (Ashkboos et al., 2024b; Liu et al., 2025). Existing rotation-based approaches implicitly adopt a homogeneous transformation strategy, where a single dense transformation is jointly optimized across all channels. Such fulldimensional rotations provide strong distribution reshaping capability, but also introduce substantial optimization complexity and parameter overhead.

Through extensive analysis, we observe that Transformer channels exhibit heterogeneous sensitivity to quantization noise. Specifically, only a small subset of channels is fragile under low-bit quantization, while most channels remain comparatively robust. This suggests that different channels require fundamentally different transformation capacities during quantization. However, existing methods such as SpinQuant (Liu et al., 2025) and OSTQuant (Hu et al., 2025) optimize dense learnable rotations over all channels indiscriminately. Jointly optimizing full-channel rotations forces sensitive and robust channels to share the same transformation subspace, causing optimization capacity to be unnecessarily allocated to robust channels while introducing cross-channel interference that destabilizes sensitive channels. Consequently, globally shared rotations result in suboptimal accuracyefficiency trade-offs.

Motivated by these observations, we propose Locally Isolated Rotation Quantization (LIRQuant), a sensitivity-aware rotation framework for LLM quantization. LIRQuant decouples sensitive and robust channels into isolated transformation subspaces according to their quantization sensitivity. Subsequently, it selectively applies highly expressive learnable rotations to high-sensitivity channels, while employing lightweight fixed Hadamard transforms for low-sensitivity ones, thereby substantially reducing optimization costs while preserving quantization accuracy. Extensive experiments across the LLaMA-2/3 families, Mistral-7B, Qwen3-8B, and the MoE model Mixtral-8×7B fully demonstrate the effectiveness of LIRQuant. Under the challenging W4A4KV4 setting, LIRQuant matches or outperforms existing rotation-based methods, such as SpinQuant (Liu et al., 2025) and FlatQuant (Sun et al., 2024), while requiring only 25% of the trainable parameters of a dense rotation. Our main contributions are summarized as follows:

• We reveal that quantization sensitivity in Transformer activations is highly heterogeneous across channels, and different channels require different transformation capacities during quantization.

• We propose LIRQuant, a sensitivity-aware locally isolated rotation framework. It selectively applies learnable rotations to sensitive channels and lightweight Hadamard transforms to insensitive ones, striking a superior accuracy-efficiency trade-off, and deeply integrates this design with the Transformer architecture.

• Extensive experiments across multiple LLMs and quantization settings demonstrate that LIRQuant achieves state-of-the-art quantization performance with the lowest calibration overhead among recent rotation-based methods, and further delivers practical inference speedups via fused block-diagonal kernels.

## 2 Motivation

## 2.1 Basics of Uniform Quantization

The deployment of Large Language Models (LLMs) faces severe memory and computational challenges. Post-Training Quantization (PTQ), as one of the most effective compression techniques, can significantly reduce memory footprint and accelerate inference speed by converting floatingpoint weights and activations into low-precision formats. For an arbitrary floating-point tensor X, uniform quantization partitions its continuous numerical range $[ x _ { \mathrm { m i n } } , x _ { \mathrm { m a x } } ]$ into $2 ^ { k } - 1$ intervals of equal length. The quantization process can be formulated as:

$$
Q ( X ) = \operatorname { r o u n d } \left( { \frac { X } { s } } \right) + z ,\tag{1}
$$

where $\begin{array} { r } { s = \frac { x _ { \mathrm { m a x } } - x _ { \mathrm { m i n } } } { 2 ^ { k } - 1 } } \end{array}$ is the quantization step size (scale), and z = −round $\left( { \frac { x _ { \mathrm { m i n } } } { s } } \right)$ is the zero point. The dequantization process is given by:

$$
\hat { \pmb X } = s \cdot ( { \cal Q } ( { \pmb X } ) - z ) .\tag{2}
$$

The Mean Squared Error (MSE) introduced by quantization is defined as:

$$
\operatorname { M S E } ( X , { \hat { X } } ) = \mathbb { E } \left[ ( X - { \hat { X } } ) ^ { 2 } \right] .\tag{3}
$$

Under the assumption of uniform quantization, the expectation of the quantization error satisfies:

$$
\mathbb { E } [ \mathrm { M S E } ] \approx \frac { s ^ { 2 } } { 1 2 } .\tag{4}
$$

Consequently, the quantization error is proportional to the square of the quantization step size. As indicated by the above equations, the quantization step size s is the core factor determining the quantization error. When outliers are present in the data, $x _ { \mathrm { m a x } } \ \mathrm { o r } \ x _ { \mathrm { m i n } }$ is significantly stretched by extreme values, causing the quantization step size s to increase substantially. Consequently, since the error is proportional to the square of the step size, this increase in s leads to a quadratic amplification of the overall quantization error.

## 2.2 Related Work

Post-Training Quantization (PTQ) for LLMs. PTQ is an efficient technique for compressing LLMs, primarily categorized into weight-only and weightactivation quantization. To alleviate memory bandwidth bottlenecks, methods such as GPTQ (Frantar et al., 2022), AWQ (Lin et al., 2024b), and QuIP (Chee et al., 2023) have made significant progress in weight-only quantization through techniques like error compensation, protecting salient weights, or introducing Hadamard matrices.

Unlike weight-only quantization, weightactivation quantization aims to comprehensively accelerate inference. However, the ubiquitous presence of activation outliers in LLMs dominates the dynamic range and is the primary cause of quantization accuracy degradation. To address this issue, early channel-wise scaling methods, such as SmoothQuant (Xiao et al., 2023), migrate the quantization difficulty from activations to weights, partially mitigating regular outliers. Subsequently, rotation-based methods such as QuaRot (Ashkboos et al., 2024b) and SpinQuant (Liu et al., 2025) leverage orthogonal transformations to redistribute activation energy across dimensions, effectively reducing peak magnitudes and improving quantization robustness. Recently, more complex transformation strategies have been proposed: DuQuant (Lin et al., 2024a) introduces a dual transformation combining block-wise rotation and zigzag permutation to tackle massive outliers; OSTQuant (Hu et al., 2025) combines orthogonal and scaling transformations to globally optimize the quantization space; and FlatQuant (Sun et al., 2024) combines Kronecker decomposition and affine transformations to smooth the distribution while minimizing runtime overhead. As illustrated in Figure 1, we compare the activation distributions of the MHSA input in the 15th layer of LLaMA-3- 8B under different transformation strategies. The original activations contain numerous extreme outliers. Although the Hadamard transform can partially smooth activations, its fixed weights may not be optimal for all activation distributions. In contrast, learning-based methods update the weights of the rotation matrices during training, thus achieving superior suppression of outliers.

![](images/ade050d1c8e63b08b63783528cd2a4a43fc0c57303fe600a00dcf7b61d8d49b7.jpg)  
Figure 1: Comparison of outlier suppression effects among different transformation methods. (a) Original activation distribution; (b) Distribution after Hadamard rotation; (c) Distribution after learnable rotation.

## 2.3 Heterogeneity of Quantization Sensitivity

Although rotation-based methods substantially improve quantization robustness, existing approaches allocate equal optimization effort and transformation capacity to all channels, uniformly applying dense full-dimensional rotations across the entire activation space. However, this approach fundamentally contradicts the intrinsic heterogeneity of quantization sensitivity in LLM activations, wasting computational resources on low-sensitivity channels that have a negligible impact on model performance.

![](images/65fbec988a65b23599c1bad469e468cb87270b3430e77d7d41ce5de754504506.jpg)  
Figure 2: Impact of selectively optimizing sensitive channels under W4A4KV4 quantization.

To validate this conclusion, we conduct a simple yet effective ablation study. As shown in Figure 2, on the LLaMA-3-8B model, applying a Hadamard transform to the subspace formed by the top 10% most sensitive channels of the MHSA input activations, while applying an identity mapping to the remaining channels prior to W4A4KV4 quantization, yields a perplexity (PPL) of 8.35. Conversely, applying an identity mapping to the top 10% sensitive channel subspace and a Hadamard transform to the remaining channels results in a PPL of 11.76, which is close to the RTN baseline of 11.85. These observations motivate a new insight: rather than transforming all channels indiscriminately, quantization transformations should allocate different transformation capacities based on channel sensitivity. This insight forms the foundation of our proposed LIRQuant.

## 3 Method

In this paper, we propose LIRQuant, the overall framework of which is illustrated in Figure 3. By accurately identifying critical subspaces through sensitivity analysis, this method reduces the number of learnable parameters and significantly improves training efficiency, while retaining the core advantages of rotation-based methods in smoothing outliers and facilitating low-bit quantization.

## 3.1 Locally Isolated Transformation

In LIRQuant, we apply an invertible transformation T to activations and weights to smooth outliers.

![](images/11b5353380864b601ac667f1ba2b0f95186b604c9729d2d5d459d75cb9e8d8f7.jpg)  
Figure 3: Architectural design of LIRQuant. (a) Global rotation flow across the Transformer blocks. (b) Integration of locally isolated rotations in the MHSA module. (c) Integration of local rotation in the FFN module. Dashed boxes indicate matrices that are fused together to reduce inference overhead.

Unlike traditional global dense rotation, LIRQuant introduces a permutation matrix $P \in \{ 0 , 1 \bar { \} } ^ { d \times d }$ sorted by channel sensitivity in descending order, parameterizing the transformation matrix into a block-diagonal form:

$$
T = P ^ { \top } \left[ \begin{array} { c c } { { H _ { h i } } } & { { 0 } } \\ { { 0 } } & { { H _ { l o } } } \end{array} \right] P ,\tag{5}
$$

where $M _ { h i }$ is a learnable matrix applied to the high-sensitivity subspace $C _ { h i }$ , and $H _ { l o }$ is a fixed Hadamard matrix applied to the low-sensitivity subspace $C _ { l o }$ This design aims to maximize the efficiency of the parameters. By allocating learnable parameters exclusively to the critical high-sensitivity channels and applying training-free fixed transformations to the low-sensitivity ones, LIRQuant significantly reduces training overhead while effectively preserving the accuracy of the quantized model.

## 3.2 Noise-Weighted Local Functional Sensitivity

To accurately partition $C _ { h i }$ and $C _ { l o } ,$ we evaluate the sensitivity of each channel based on local functional distortion. For an insertion point $p ,$ let the intermediate activation be $z _ { p }$ and the local target be $y _ { p } = f _ { p } ( z _ { p } )$ . The distortion introduced by the quantization perturbation $\hat { z } _ { p } = z _ { p } + \epsilon _ { p }$ is defined as the expected distance over the calibration set $\mathcal { C } \mathrm { : }$

$$
\mathcal { D } _ { p } = \mathbb { E } _ { \mathcal { C } } \left[ D _ { p } \big ( f _ { p } ( z _ { p } + \epsilon _ { p } ) , f _ { p } ( z _ { p } ) \big ) \right] .\tag{6}
$$

Through Taylor expansion and Hessian approximation (detailed derivation in Appendix A), this

distortion can be diagonalized and decomposed as $\begin{array} { r } { \mathcal { D } _ { p } \approx \sum _ { j = 1 } ^ { d } G _ { p , j j } \sigma _ { p , j } ^ { \tilde { 2 } } } \end{array}$ . Thus, we define the sensitivity of the j-th channel as:

$$
s _ { p , j } = G _ { p , j j } \sigma _ { p , j } ^ { 2 } .\tag{7}
$$

This metric decouples the functional sensitivity $G _ { p , j j }$ and the quantization noise intensity $\sigma _ { p , j } ^ { 2 }$ . If weight quantization is also considered, an additive approximation $s _ { p , j } = s _ { p , j } ^ { a c t } + s _ { p , j } ^ { w }$ is adopted.

## 3.3 Sensitivity Instantiation

Based on the above framework, we instantiate the sensitivity $s _ { p , j }$ for each insertion point as follows (where $\Delta$ denotes the quantization error):

• Attention and FFN Inputs: The local target is the output of the linear layers. The sensitivity of the j-th input channel is:

$$
s _ { j } ^ { i n } = \sum _ { r } \big ( \| \Delta x _ { j } \| _ { 2 } ^ { 2 } \| w _ { r , j } \| _ { 2 } ^ { 2 } + \| x _ { j } \| _ { 2 } ^ { 2 } \| \Delta w _ { r , j } \| _ { 2 } ^ { 2 } \big )\tag{8}
$$

• QK Path: The target is the attention matrix. Let $A = Q K ^ { \top } / \sqrt { d }$ and $P = \operatorname { s o f t m a x } ( A )$ Using logits-MSE to approximate the local KL distortion, the sensitivity of the j-th head channel is:

$$
s _ { j } ^ { q k } \approx \frac { 1 } { d } \mathbb { E } \big [ \| \Delta q _ { j } \| _ { 2 } ^ { 2 } \| k _ { j } \| _ { 2 } ^ { 2 } + \| q _ { j } \| _ { 2 } ^ { 2 } \| \Delta k _ { j } \| _ { 2 } ^ { 2 } \big ]\tag{9}
$$

• VO Path: The target is the value propagation under the fixed full-precision attention

weights $P _ { f p }$ . The sensitivity of the j-th value channel is:

$$
\begin{array} { r l } & { s _ { j } ^ { v o } = \mathbb { E } \big [ \| P _ { f p } \Delta v _ { j } \| _ { 2 } ^ { 2 } \| w _ { o , j } \| _ { 2 } ^ { 2 } } \\ & { \qquad + \| P _ { f p } v _ { j } \| _ { 2 } ^ { 2 } \| \Delta w _ { o , j } \| _ { 2 } ^ { 2 } \big ] } \end{array}\tag{10}
$$

• Down Projection Input: Let $z =$ $\phi ( x W _ { g a t e } ) \odot x W _ { u p }$ , and the target is the final output of the FFN, $z W _ { d o w n }$ . The sensitivity of the j-th channel is:

$$
\begin{array} { r l } & { s _ { j } ^ { d o w n } = \| \Delta z _ { j } \| _ { 2 } ^ { 2 } \| w _ { d o w n , j } \| _ { 2 } ^ { 2 } } \\ & { \quad \quad + \| z _ { j } \| _ { 2 } ^ { 2 } \| \Delta w _ { d o w n , j } \| _ { 2 } ^ { 2 } } \end{array}\tag{11}
$$

## 3.4 Overall Workflow and End-to-End Training

The overall pipeline of LIRQuant consists of two stages:

1. Sensitivity Analysis and Initialization: First, we collect activation statistics on a small calibration set and calculate the channel sensitivity $s _ { p , j }$ for each layer using the formulas in Section 3.3. Subsequently, based on the determined proportion of high-sensitivity channels, we determine the permutation matrix $P$ and the high-sensitivity subspace dimension $k _ { p }$ for each insertion point.

2. End-to-End Training: After determining the subspace partition, we freeze the original weights of the LLM and only set the transformation matrices $M _ { h i }$ of the high-sensitivity subspaces at each insertion point as learnable parameters. We perform end-to-end training on the calibration set using the standard Cross-Entropy (CE) loss:

$$
\underset { \{ M _ { h i } \} } { \operatorname* { m i n } } \mathbb { E } _ { ( X , Y ) \in \mathcal { C } } \left[ \mathcal { L } _ { \mathrm { C E } } \big ( \mathbf { L } \mathbf { L } \mathbf { M } _ { Q } ( X ; \{ M _ { h i } \} ) , Y \big ) \right]\tag{12}
$$

where C is the calibration set, $( X , Y )$ represents the input sequence and its corresponding target labels, L $\mathrm { L M } _ { Q }$ denotes the quantized large language model and locally isolated transformations, and $\left\{ { M } _ { h i } \right\}$ represents the set of transformation matrices to be optimized across all insertion points.

## 3.5 Integration with the Transformer Architecture

We now illustrate the integration of our proposed quantization framework with a Transformer block based on the LLaMA architecture. Following conventional practices, we apply low-bit quantization to the inputs and weights of linear projections, while keeping operations such as RMSNorm, RoPE embeddings, and softmax attention scores in FP16.

Global Residual Stream. Our architecture maintains the entire residual stream in a globally rotated space defined by $\mathbf { R } _ { G }$ . Specifically, we do not apply permutation to $\mathbf { R } _ { G }$ and instead utilize a full-size learnable matrix for it. As illustrated in Figure 3(a), the initial embedding is multiplied by $\mathbf { R } _ { G }$ , and all subsequent Transformer blocks operate within this transformed space. To ensure strict computational equivalence, the learnable scaling parameters of the RMSNorm $( \mathbf { W } _ { \mathrm { n o r m } } ^ { i } )$ are pre-absorbed into the subsequent projection weights offline. This allows the orthogonal matrix $\mathbf { R } _ { G }$ to commute with the normalization operation without introducing any mathematical discrepancy.

Self-Attention. The self-attention module incorporates the global rotation $\mathbf { R } _ { G }$ and local transformations $\{ \mathbf { P } _ { 1 } , \mathbf { R } _ { 1 } , \mathbf { P } _ { 2 } , \mathbf { R } _ { 2 } \}$ . Specifically, the input from the globally rotated residual stream is first projected back using the merged weights $\mathbf { R } _ { G } ^ { \top } \mathbf { W } _ { q } .$ $\mathbf { R } _ { G } ^ { \top } \mathbf { W } _ { k }$ , and $\mathbf { R } _ { G } ^ { \top } \mathbf { W } _ { v }$ . To preserve the positional information, the Rotary Position Embedding (RoPE) is applied before any channel-wise transformations. The permutation ${ \bf P } _ { 1 }$ and local rotation ${ \bf R } _ { 1 }$ are applied to the queries and keys to smooth the attention computation. Since the permutation matrix is inherently an orthogonal matrix, computational equivalence is strictly satisfied. Furthermore, we maintain the computation of the attention map in high precision and quantize the output of the K projection before storing it in the K cache. Similarly, $\mathbf { P } _ { 2 }$ and ${ \bf R } _ { 2 }$ are applied to the values. For the output projection, the inverse transformations ${ \bf R } _ { 2 } ^ { \top }$ and $\mathbf { P } _ { 2 } ^ { \top }$ , along with the global rotation $\mathbf { R } _ { G }$ , are analytically absorbed into the output weight $\mathbf { W } _ { o }$ forming the merged weight R<sup>⊤</sup> $\mathbf { \Delta P } _ { 2 } ^ { \top } \mathbf { W } _ { o } \mathbf { R } _ { G }$

Feed-forward Network. The feed-forward network (FFN) employs the global rotation $\mathbf { R } _ { G }$ alongside local transformations $\mathbf { P } _ { 3 }$ and ${ \bf R } _ { 3 }$ . The inputs to the up and gate projections are transformed by the merged weights $\mathbf { R } _ { G } ^ { \top } \mathbf { W } _ { u }$ and $\mathbf { R } _ { G } ^ { \top } \mathbf { W } _ { g } ,$ , respectively. The local permutation $\mathbf { P } _ { 3 }$ and rotation ${ \bf R } _ { 3 }$ are applied after the activation function; during inference, they will be fused and computed online to smooth the hidden states prior to quantization. Finally, similar to the attention module, the inverse transformations ${ \bf R } _ { 3 } ^ { \top }$ and $\mathbf { P } _ { 3 } ^ { \top }$ are fused into the down projection weight $\mathbf { W } _ { d } ,$ and the output is re-projected into the global rotated space via $\mathbf { R } _ { G }$ , resulting in the merged weight ${ \bf R } _ { 3 } ^ { \top } { \bf P } _ { 3 } ^ { \top } { \bf W } _ { d } { \bf R } _ { G }$ without introducing additional inference overhead.

## 4 Experiments

## 4.1 Experimental Setup

Evaluation Metrics and Baselines. We primarily validate the effectiveness of LIRQuant on the LLaMA-2 (Touvron et al., 2023) and LLaMA-3 (Grattafiori et al., 2024) model families. To further examine architectural generalization, we extend our evaluation to Mistral-7B-v0.3 (Jiang et al., 2023) , Qwen3-8B (Yang et al., 2025), and the Mixture-of-Experts model Mixtral-8×7B (Jiang et al., 2024). Following existing studies (Shao et al., 2023; Ashkboos et al., 2024b), we evaluate the language generation performance using Perplexity (PPL) on the WikiText-2 (Merity et al., 2016) and C4 (Raffel et al., 2023) datasets. For common sense reasoning, we select six zero-shot evaluation tasks: ARC-Challenge, ARC-Easy (Clark et al., 2018), HellaSwag (Zellers et al., 2019), LAMBADA (Paperno et al., 2016), PIQA (Bisk et al., 2020), and WinoGrande (Sakaguchi et al., 2021). We compare LIRQuant with main INT4 PTQ methods, including RTN, GPTQ, AWQ, QuIP, SmoothQuant, as well as recent state-of-the-art methods: QuaRot (Ashkboos et al., 2024b), SpinQuant (Liu et al., 2025), OSTQuant (Hu et al., 2025), FlatQuant (Sun et al., 2024), and DartQuant (Shao et al., 2026).

Implementation Details. Our implementation is based on HuggingFace (Wolf et al., 2019) and PyTorch (Paszke et al., 2019). We initialize the trainable matrices as diagonal matrices and optimize them using the RiemannAdam optimizer (Bé- cigneul and Ganea, 2018) for 200 iterations. The initial learning rate is set to $3 \times 1 0 ^ { - 2 }$ with a cosine annealing learning rate decay strategy. Following the standard experimental setup of GPTQ, our calibration set consists of 128 samples from the WikiText-2 dataset, each with a sequence length of 2048 tokens, and the batch size is set to 8. To ensure deployment-friendly inference, we adopt symmetric per-channel quantization for weights and symmetric per-token quantization for activations. For a fair comparison with other baseline methods, we employ Round-to-nearest (RTN) as the activation quantizer and GPTQ as the weight quantizer. Unless otherwise specified, the KV cache is quantized to 4 bits in all weight-activation experiments; that is, the W4A4 setting in this paper always refers to W4A4KV4. Calibration overhead is measured on a single-node 8×NVIDIA H20 (96 GB) platform, and inference efficiency is measured on a single NVIDIA A100-SXM4-80GB.

## 4.2 Main Results

Table 1 presents the perplexity on WikiText-2 and C4 datasets, alongside the average accuracy on zero-shot common sense reasoning tasks across the LLaMA-2 and LLaMA-3 model families. In W4A16, LIRQuant demonstrates highly competitive performance, matching or outperforming existing state-of-the-art methods like GPTQ and AWQ.

When transitioning to the more challenging W4A4KV4 ultra-low-bit quantization, the advantages of LIRQuant become even more pronounced. For instance, on the LLaMA-2 70B model, LIRQuant achieves a WikiText-2 perplexity of 3.50. This leaves a marginal gap of merely 0.18 compared to the 3.32 perplexity of the fullprecision baseline, thereby largely preserving the original generative capacity. Notably, on the newer LLaMA-3 8B model, LIRQuant establishes the optimal results under the W4A4KV4 setting with a perplexity of 6.81 on WikiText-2 and 9.24 on C4. This significantly outperforms recent advanced methods, surpassing SpinQuant by 0.62 and 1.80 points, and FlatQuant by 0.31 and 0.98 points on the two datasets, respectively.

Furthermore, in zero-shot common sense reasoning, LIRQuant consistently delivers the highest average accuracy across all evaluated models under both quantization settings. Detailed results for these downstream tasks are provided in Appendix D.

## 4.3 Generalization to Broader Architectures

To verify that the sensitivity-guided design is not specific to the LLaMA architecture, we further evaluate LIRQuant on Mistral-7B-v0.3, Qwen3- 8B, and the MoE model Mixtral-8×7B under W4A4KV4. As shown in Table 2, LIRQuant ranks first on both WikiText-2 perplexity and the average accuracy of six downstream tasks across all three architectures. On Mixtral-8×7B, LIRQuant achieves a perplexity of 4.14 and an average accuracy of 77.27%, which are respectively 0.13 lower and 0.48 percentage points higher than the runnerup FlatQuant, while OSTQuant fails to complete calibration due to out-of-memory errors. These results demonstrate that the heterogeneity of channel sensitivity is a general property across modern

<table><tr><td rowspan="2">Bits (W-A-KV)</td><td rowspan="2">Method</td><td colspan="3">LLaMA-2 7B</td><td colspan="3">LLaMA-2 13B</td><td colspan="3">LLaMA-2 70B</td><td colspan="3">LLaMA-3 8B</td></tr><tr><td>Wiki↓</td><td>C4↓</td><td> $\mathbf { A v g } \uparrow$ </td><td>Wiki↓</td><td>C4↓</td><td> $\mathbf { A v g } \uparrow$ </td><td>Wiki ↓</td><td>C4↓</td><td> $\mathbf { A v g } \uparrow$ </td><td>Wiki ↓</td><td>C4↓</td><td>Avg ↑</td></tr><tr><td>16-16-16</td><td>FP16</td><td>5.68</td><td>7.08</td><td>69.85</td><td>4.88</td><td>6.47</td><td>72.54</td><td>3.32</td><td>5.71</td><td>77.09</td><td>6.24</td><td>8.95</td><td>73.30</td></tr><tr><td rowspan="6">4-16-16</td><td>RTN GPTQ</td><td>5.96 5.65</td><td>7.38</td><td>65.41</td><td>4.98</td><td>6.59</td><td>64.56</td><td>3.87</td><td>8.91</td><td>75.13</td><td>6.82</td><td>9.72</td><td>68.92</td></tr><tr><td>AWQ</td><td></td><td>7.16</td><td>66.59</td><td>5.02</td><td>6.59</td><td>70.28</td><td>4.00</td><td>7.61</td><td>76.59</td><td>6.72</td><td>9.55</td><td>64.65</td></tr><tr><td></td><td>5.69</td><td>7.20</td><td>68.01</td><td>5.01</td><td>6.61</td><td>72.08</td><td>4.10</td><td>7.35</td><td>76.53</td><td>6.83</td><td>9.76</td><td>72.08</td></tr><tr><td>QuIP</td><td>5.87</td><td>7.34</td><td></td><td>5.05</td><td>6.69</td><td></td><td></td><td></td><td></td><td>6.82</td><td>9.92</td><td></td></tr><tr><td>LIRQuant (Ours)</td><td>5.69</td><td>7.11</td><td>69.25</td><td>4.98</td><td>6.55</td><td>72.25</td><td>3.65</td><td>6.36</td><td>77.38</td><td>6.45</td><td>9.55</td><td>73.85</td></tr><tr><td>RTN</td><td>5654.5</td><td>3181.9</td><td>30.43</td><td>5636.0</td><td>4330.9</td><td>29.93</td><td>7e4</td><td>5e4</td><td>31.42</td><td>4282.4</td><td>2936.5</td><td>31.58</td></tr><tr><td rowspan="6">4-4-4</td><td>SmoothQuant</td><td>9242.4</td><td>6961.6</td><td>30.07</td><td>4731.0</td><td>3820.3</td><td>32.63</td><td>1e3</td><td>34.61</td><td>34.01</td><td>25820.2</td><td>19069.3</td><td>30.72</td></tr><tr><td>SpinQuant</td><td>6.14</td><td>7.67</td><td>66.40</td><td>5.26</td><td>7.00</td><td>69.10</td><td>3.61</td><td>6.07</td><td>76.18</td><td>7.43</td><td>11.04</td><td>69.11</td></tr><tr><td>QuaRot</td><td>6.10</td><td>8.71</td><td>66.00</td><td>5.39</td><td>7.82</td><td>69.73</td><td>3.80</td><td>6.42</td><td>76.19</td><td>10.84</td><td>18.94</td><td>65.30</td></tr><tr><td>OSTQuant</td><td>5.96</td><td>8.01</td><td>67.43</td><td>5.40</td><td>6.81</td><td>70.14</td><td>3.59</td><td>5.95</td><td>76.20</td><td>6.83</td><td>9.55</td><td>70.09</td></tr><tr><td>FlatQuant</td><td>5.79</td><td>7.49</td><td>67.47</td><td>5.12</td><td>6.91</td><td>71.64</td><td>3.54</td><td>5.85</td><td>76.53</td><td>7.12</td><td>10.22</td><td>71.33</td></tr><tr><td>LIRQuant (Ours)</td><td>5.83</td><td>7.95</td><td>67.67</td><td>4.94</td><td>6.67</td><td>72.05</td><td>3.50</td><td>5.81</td><td>76.85</td><td>6.81</td><td>9.24</td><td>71.60</td></tr></table>

Table 1: Perplexity (PPL) and zero-shot common sense reasoning evaluation of various quantization methods on the LLaMA family under W4A16KV16 and W4A4KV4. For PPL (WikiText-2 and C4), lower is better. For Avg, higher is better. Best results are bolded.

<table><tr><td rowspan="2">Method</td><td colspan="2">Mistral-7B</td><td colspan="2">Qwen3-8B</td><td colspan="2">Mixtral-8×7B</td></tr><tr><td>Wiki↓</td><td>Avg ↑</td><td>Wiki↓</td><td>Avg ↑</td><td>Wiki ↓</td><td> $\mathbf { A v g } \uparrow$ </td></tr><tr><td>SpinQuant</td><td>5.82</td><td>70.80</td><td>7.99</td><td>69.23</td><td></td><td></td></tr><tr><td>OSTQuant</td><td>5.79</td><td>71.10</td><td>7.77</td><td>69.70</td><td>OOM</td><td>O0M</td></tr><tr><td>DartQuant</td><td>5.92</td><td>70.35</td><td>8.19</td><td>68.65</td><td>5.06</td><td>75.93</td></tr><tr><td>FlatQuant</td><td>5.76</td><td>71.38</td><td>7.44</td><td>70.57</td><td>4.27</td><td>76.79</td></tr><tr><td>LIRQuant (Ours)</td><td>5.70</td><td>71.77</td><td>7.32</td><td>71.00</td><td>4.14</td><td>77.27</td></tr></table>

Table 2: WikiText-2 perplexity and average zero-shot accuracy under W4A4KV4 on Mistral-7B-v0.3, Qwen3- 8B, and Mixtral-8×7B. OSTQuant results on Mixtral are unavailable due to out-of-memory errors during calibration. Per-task results are in Appendix D.

LLM architectures, including MoE models.

## 4.4 Efficiency Analysis

Calibration Overhead. Existing orthogonal rotation quantization methods typically require learning or applying dense rotation matrices in the full channel space $\mathbb { R } ^ { d \times d }$ , leading to a quadratic growth in learnable parameters and computational complexity $( \mathcal { O } ( d ^ { 2 } ) )$ . In contrast, LIRQuant only learns the transformation matrix $M _ { h i }$ within the highsensitivity subspace $C _ { h i }$ : under the default 50% split, the learnable parameters are reduced to one quarter of a dense rotation, which cuts the peak training VRAM from 81.25 GB to 20.31 GB (- 75%) and shortens the end-to-end training time by 2.44× on a single H20 GPU, compared with learning a full-dimensional rotation under the identical training recipe. Table 3 further compares the calibration overhead against four recent rotationbased methods under identical W4A4KV4 settings. LIRQuant retains the same Stiefel-manifold optimizer and end-to-end training paradigm as OS-TQuant, but the reduced parameter count directly lowers the per-step backpropagation cost and optimizer state size, and fewer learnable parameters also reduce the overfitting risk and the number of steps required for convergence. Consequently, LIRQuant completes calibration of LLaMA-3-8B in 25 minutes with 20.3 GB peak GPU memory and 16 GB host memory, achieving the best calibration time and host memory on every evaluated model and the lowest GPU memory on all dense models: it avoids SpinQuant’s 9-hour float64 rerotation bottleneck, OSTQuant’s teacher-induced GPU memory peaks, and DartQuant’s 200 GBlevel host memory for activation caching, and it scales to the MoE model Mixtral-8×7B where OS-TQuant runs out of memory. A detailed analysis of each baseline’s bottleneck is deferred to Appendix E.

Inference Efficiency. During inference, only two transformations remain online: ${ \bf R } _ { 1 }$ applied to Q/K after RoPE and ${ \bf R } _ { 3 }$ before the down projection, while $\mathbf { R } _ { G }$ and ${ \bf R } _ { 2 }$ are absorbed offline into the weights. Both online transformations preserve the block-diagonal structure blkdiag( $M _ { h i } , H _ { l o } )$ so the dense sub-block only operates on the highsensitivity dimensions and the remainder uses the fast Hadamard transform, reducing computation and memory traffic. We fuse the block-diagonal transform, per-token reduction, symmetric INT4 quantization, and packing into a single Triton kernel, and adopt CUDA Graph for fixed-shape decoding. As shown in Table 4, on LLaMA-3-8B LIRQuant achieves 2.18×–2.32× prefill speedup and up to 3.55× peak memory saving over the FP16 baseline; the decode speedup grows with sequence length and batch size, reaching 1.82× at batch size 64. Detailed kernel designs and full results are provided in Appendix F.

<table><tr><td rowspan="2">Method</td><td colspan="3">LLaMA-3-8B</td><td colspan="3">Mistral-7B-v0.3</td><td colspan="3">Qwen3-8B</td><td colspan="3">Mixtral-8×7B</td></tr><tr><td>Time</td><td>GPU</td><td>Host</td><td>Time</td><td>GPU</td><td>Host</td><td>Time</td><td>GPU</td><td>Host</td><td>Time</td><td>GPU</td><td>Host</td></tr><tr><td>SpinQuant</td><td>9h21m</td><td>26.1</td><td>16</td><td>9h19m</td><td>28.6</td><td>17</td><td>9h20m</td><td>29.3</td><td>17</td><td>一</td><td></td><td>一</td></tr><tr><td>OSTQuant</td><td>44m</td><td>46.5</td><td>421</td><td>42m</td><td>33.1</td><td>443</td><td>42m</td><td>60.4</td><td>419</td><td></td><td>OOM</td><td></td></tr><tr><td>FlatQuant</td><td>1h13m</td><td>32.4</td><td>27</td><td>1h11m</td><td>35.2</td><td>28</td><td>1h15m</td><td>38.1</td><td>30</td><td>3h12m</td><td>64.5</td><td>109</td></tr><tr><td>DartQuant</td><td>30m</td><td>22.7</td><td>201</td><td>30m</td><td>22.1</td><td>200</td><td>30m</td><td>22.3</td><td>220</td><td>1h32m</td><td>16.0</td><td>304</td></tr><tr><td>LIRQuant (Ours)</td><td>25m</td><td>20.3</td><td>16</td><td>25m</td><td>18.7</td><td>15</td><td>30m</td><td>21.0</td><td>16</td><td>1h12m</td><td>35.0</td><td>56</td></tr></table>

Table 3: Calibration overhead under W4A4KV4 on 8×NVIDIA H20. Time is the total wall-clock time for calibration (rotation learning + weight quantization); GPU and Host denote per-device peak GPU memory and peak host memory (GB). All baselines use their official implementations. Per-method analysis is provided in Appendix E.

<table><tr><td>Seq. Len.</td><td>Prefill Speedup</td><td>FP16 Prefill</td><td>Decode Speedup</td><td>Mem. Saving</td></tr><tr><td>128</td><td>2.18×</td><td>31.2 ms</td><td>0.95×</td><td>3.55×</td></tr><tr><td>256</td><td>2.20×</td><td>45.6 ms</td><td>1.02×</td><td>3.52×</td></tr><tr><td>1024</td><td>2.25×</td><td>156.9 ms</td><td>1.15×</td><td>3.45×</td></tr><tr><td>4096</td><td>2.29×</td><td>624.0 ms</td><td>1.28×</td><td>3.38×</td></tr><tr><td>8192</td><td>2.32×</td><td>1320.5 ms</td><td>1.40×</td><td>3.30×</td></tr></table>

Table 4: End-to-end inference efficiency of LIRQuant on LLaMA-3-8B (batch size 4, NVIDIA A100-80GB) relative to the FP16 baseline. Results on Mistral-7B and Qwen3-8B, and the batch-size scaling study, are in Appendix F.

<table><tr><td>Ratio (%)</td><td>0</td><td>10</td><td>20</td><td>30</td><td>40</td><td>50</td><td>60</td><td>70</td><td>80</td><td>90</td><td>100</td></tr><tr><td>PPL</td><td>11.59</td><td>13.28</td><td>8.92</td><td>7.55</td><td>7.32</td><td>6.78</td><td>6.82</td><td>6.88</td><td>6.85</td><td>6.92</td><td>8.01</td></tr></table>

Table 5: WikiText-2 PPL of LLaMA-3-8B (W4A4KV4) as a function of the high-sensitivity ratio r. r = 0% corresponds to the full-Hadamard configuration and $r =$ 100% to a fully learnable dense rotation.

High-Sensitivity Ratio. Table 5 sweeps the highsensitivity ratio r from 0% (full Hadamard) to 100% (fully learnable dense rotation). The sweep is conducted with an independently re-sampled calibration set, so the absolute PPL values may differ slightly from those in Table 1. The PPL exhibits a rise–fall–plateau pattern. $\mathbf { A } \mathbf { t } \ r = 1 0 \%$ , a diffusion loss peak emerges: the block-diagonal structure eliminates cross-block energy exchange, so outliers concentrated in a tiny learnable block can no longer be dispersed over the full channel space, temporarily degrading quantization precision. As r increases, the learnable rotation quickly compensates for this loss and the PPL drops steeply, reaching its best value at $r = 5 0 \%$ . Beyond this point, the PPL merely oscillates within a narrow band without consistent improvement, while the learnable parameters and the associated overfitting risk grow quadratically with r. We therefore adopt the 50/50 split as the principled default configuration of LIRQuant. The construction of $\mathbf { H } _ { \mathrm { l o } }$ for

## 4.5 Ablation Study

![](images/d51f793808704eb050c3c64a498d08578a5dbc12ef00a49610ad6c27bfb324e1.jpg)

![](images/7ff9718006677da908a24b44dd75c60987be516886aeda77e8ba66012b0645bc.jpg)

![](images/5a955986887dbd5239924a3ad07fb2fab10edc35fdd148b72d9be079cde5f11d.jpg)

![](images/189e6873f752db3a166ca51b4a5778f990df9b3580e31ec21e019e667157f6e9.jpg)  
Figure 4: Heatmap of pairwise Jaccard similarity between the sets of top-10% sensitive channel positions from different layers.

arbitrary dimensions is described in Appendix C, and the comparison against alternative channel selection metrics over the full sweep is provided in Appendix G.

Sensitivity Metric. We compare our noiseweighted functional sensitivity against six common channel importance metrics (maxabs, L2 norm, L1 norm as in AWQ, the 99.9th percentile, kurtosis, and random selection) across the full ratio sweep in Appendix G. Our metric yields the best PPL at every ratio in the practically relevant range (20%– 70%), including the optimal 50% split. Among the alternatives, only maxabs approaches our performance owing to its fidelity to extreme outliers, whereas magnitude-accumulation metrics (L1/L2) are diluted by moderate values, and scale-invariant kurtosis discards magnitude information entirely. The additional cost of computing our metric is on the order of minutes, which is negligible relative to the resulting accuracy gains.

![](images/6cbfd21051875e35d92ecdb1470a7437432fb2874eda6cd9778f6a6ab4b8d431.jpg)

![](images/12c61ce7226fe8fba3c67048d1f8dd3fccb5624df940a8176b68491793f260c2.jpg)  
Figure 5: (a) Lorenz curves of channel sensitivity. (b) Channel reduction factor of sensitivity-guided selection compared with random selection under a target coverage ratio τ .

Cross-Layer Independence Analysis. As shown in Figure 4, the four paths exhibit distinctly different inter-layer structures. The average offdiagonal Jaccard coefficient for the residual stream (labeled R1 in the figure) is 0.521, indicating a significant overlap of high-sensitivity channels across different layers. This aligns with the global semantic consistency of the residual stream and provides direct justification for sharing a single global rotation matrix $\mathbf { R } _ { G }$ across layers. In contrast, the inter-layer Jaccard coefficients for the QK path, VO path, and Down input are all below 0.07, and their heatmaps display an almost purely diagonal sparse structure, meaning the high-sensitivity channels across different layers have nearly zero overlap. This result demonstrates that the sensitive channels at local rotation positions possess strong layer specificity. Any attempt to share local rotation matrices across layers would lead to a significant loss in coverage, thereby providing empirical support for our Stage-2 design of independently training $\mathbf { R } _ { 1 } ^ { ( l ) } , \mathbf { R } _ { 2 } ^ { ( l ) }$ , and ${ \bf R } _ { 3 } ^ { ( l ) }$ for each layer.

Intra-Layer Concentration Analysis. As illustrated in Figure 5, the Lorenz curves reveal significant differences in the intra-layer distributions across various paths. The diagonal dashed line represents the theoretical baseline of a perfectly uniform distribution. The curve for the QK path deviates most significantly from the diagonal (Gini $= 0 . 6 3 )$ , whereas the Down input and the residual stream remain close to the diagonal with Gini coefficients of 0.24 and 0.10, respectively. This discrepancy is reflected more intuitively as an efficiency multiplier in Figure 5(b): at $\tau = 5 0 \%$ the sensitivity-guided selection efficiency for the QK path reaches up to 7.5×—meaning it requires only 1/7.5 of the channels compared to random selection to cover the same amount of sensitivity mass. The VO path achieves 2.8×, the Down input

1.5×, and the residual stream merely 1.1×, which is almost indistinguishable from the random baseline. Cross-layer consistency determines whether global or layer-wise rotations should be adopted, while intra-layer concentration dictates the gain efficiency brought by coverage-based channel selection. Considering both factors, we apply a dense global rotation to the residual stream, and employ sensitivity-based local rotations for the remaining paths.

## 5 Conclusion

In this paper, we propose LIRQuant, an efficient Locally Isolated Rotation Quantization framework. By strategically isolating high-sensitivity channels into a learnable subspace while applying fixed Hadamard transforms to the remaining lowsensitivity channels, our method successfully balances channel mixing quality and parameter efficiency. Under the challenging W4A4KV4 setting, LIRQuant achieves state-of-the-art accuracy across the LLaMA, Mistral, Qwen3, and Mixtral families, completes calibration of LLaMA-3-8B in 25 minutes with 20.3 GB peak GPU memory—the lowest overhead among recent rotation-based methods— and delivers up to 2.3× prefill speedup and 3.5× memory saving at inference, making the efficient optimization and deployment of massive language models on resource-constrained devices (e.g., a single GPU) highly feasible.

## Limitations

LIRQuant determines the channel partition statically from a small calibration set: the partition is shared across all inputs, and the 50% highsensitivity ratio is applied uniformly to every insertion point, whereas a per-layer or input-adaptive ratio may further improve the accuracy-efficiency trade-off. In addition, our study focuses on INT4 weight-activation-KV quantization; extending the sensitivity-guided isolation to more aggressive settings (e.g., W3A3 or low-bit floating-point formats) and to larger-scale MoE models is left for future work.

## References

Josh Achiam, Steven Adler, Sandhini Agarwal, Lama Ahmad, Ilge Akkaya, Florencia Leoni Aleman, Diogo Almeida, Janko Altenschmidt, Sam Altman, Shyamal Anadkat, and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Saleh Ashkboos, Ilia Markov, Elias Frantar, Tingxuan Zhong, Xincheng Wang, Jie Ren, Torsten Hoefler, and Dan Alistarh. 2024a. Quik: Towards end-to-end 4-bit inference on generative large language models. In Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing, pages 3355–3371.

Saleh Ashkboos, Amirkeivan Mohtashami, Maximilian L Croci, Bo Li, Pashmina Cameron, Martin Jaggi, Dan Alistarh, Torsten Hoefler, and James Hensman. 2024b. Quarot: Outlier-free 4-bit inference in rotated llms. Advances in Neural Information Processing Systems, 37:100213–100240.

Gary Bécigneul and Octavian-Eugen Ganea. 2018. Riemannian adaptive optimization methods. CoRR, abs/1810.00760.

Yonatan Bisk, Rowan Zellers, Ronan Le Bras, Jianfeng Gao, and Yejin Choi. 2020. Piqa: Reasoning about physical commonsense in natural language. In Thirty-Fourth AAAI Conference on Artificial Intelligence.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, and 1 others. 2020. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901.

Jerry Chee, Yaohui Cai, Volodymyr Kuleshov, and Christopher M De Sa. 2023. Quip: 2-bit quantization of large language models with guarantees. Advances in neural information processing systems, 36:4396– 4429.

Peter Clark, Isaac Cowhey, Oren Etzioni, Tushar Khot, Ashish Sabharwal, Carissa Schoenick, and Oyvind Tafjord. 2018. Think you have solved question answering? try arc, the ai2 reasoning challenge. arXiv:1803.05457v1.

T. Dettmers, M. Lewis, Y. Belkada, and 1 others. 2022. Gpt3.int8(): 8-bit matrix multiplication for transformers at scale. In Advances in Neural Information Processing Systems, volume 35, pages 30318–30332.

Jared Fernandez, Clara Na, Vashisth Tiwari, Yonatan Bisk, Sasha Luccioni, and Emma Strubell. 2025. Energy considerations of large language model inference and efficiency optimizations. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 32556–32569.

E. Frantar, S. Ashkboos, T. Hoefler, and 1 others. 2022. Gptq: Accurate post-training quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, and 1 others. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Dan Hendrycks, Collin Burns, Steven Basart, Andy Zou, Mantas Mazeika, Dawn Song, and Jacob Steinhardt. 2020. Measuring massive multitask language understanding. arXiv preprint arXiv:2009.03300.

Xing Hu, Yuan Cheng, Dawei Yang, Zukang Xu, Zhihang Yuan, Jiangyong Yu, Chen Xu, Zhe Jiang, and Sifan Zhou. 2025. Ostquant: Refining large language model quantization with orthogonal and scaling transformations for better distribution fitting. arXiv preprint arXiv:2501.13987.

Albert Q Jiang, Alexandre Sablayrolles, Antoine Roux, Arthur Mensch, Blanche Savary, Chris Bamford, Devendra Singh Chaplot, Diego de las Casas, Emma Bou Hanna, Florian Bressand, and 1 others. 2024. Mixtral of experts. arXiv preprint arXiv:2401.04088.

Albert Qiaochu Jiang, Alexandre Sablayrolles, Arthur Mensch, Chris Bamford, Devendra Singh Chaplot, Diego de Las Casas, Florian Bressand, Gianna Lengyel, Guillaume Lample, Lucile Saulnier, Lélio Renard Lavaud, Marie-Anne Lachaux, Pierre Stock, Teven Le Scao, Thibaut Lavril, Thomas Wang, Timothée Lacroix, and William El Sayed. 2023. Mistral 7b. ArXiv, abs/2310.06825.

Haokun Lin, Haobo Xu, Yichen Wu, Jingzhi Cui, Yingtao Zhang, Linzhan Mou, Linqi Song, Zhenan Sun, and Ying Wei. 2024a. Duquant: Distributing outliers via dual transformation makes stronger quantized llms. Advances in Neural Information Processing Systems, 37:87766–87800.

J. Lin, J. Tang, H. Tang, and 1 others. 2024b. Awq: Activation-aware weight quantization for on-device llm compression and acceleration. Proceedings of Machine Learning and Systems, 6:87–100.

Zechun Liu, Changsheng Zhao, Igor Fedorov, Bilge Soran, Dhruv Choudhary, Raghuraman Krishnamoorthi, Vikas Chandra, Yuandong Tian, and Tijmen Blankevoort. 2025. Spinquant: Llm quantization with learned rotations. In International Conference on Learning Representations, volume 2025, pages 92009–92032.

Sasha Luccioni, Yacine Jernite, and Emma Strubell. 2024. Power hungry processing: Watts driving the cost of ai deployment? In Proceedings ofthe 2024 ACM conference on fairness, accountability, and transparency, pages 85–99.

Stephen Merity, Caiming Xiong, James Bradbury, and Richard Socher. 2016. Pointer sentinel mixture models. Preprint, arXiv:1609.07843.

Denis Paperno, Germán Kruszewski, Angeliki Lazaridou, Ngoc Quan Pham, Raffaella Bernardi, Sandro Pezzelle, Marco Baroni, Gemma Boleda, and Raquel Fernandez. 2016. The LAMBADA dataset: Word prediction requiring a broad discourse context. In Proceedings of the 54th Annual Meeting of the Associationfor Computational Linguistics (Volume 1: Long Papers), pages 1525–1534, Berlin, Germany. Association for Computational Linguistics.

Adam Paszke, Sam Gross, Francisco Massa, Adam Lerer, James Bradbury, Gregory Chanan, Trevor Killeen, Zeming Lin, Natalia Gimelshein, Luca Antiga, Alban Desmaison, Andreas Köpf, Edward Z. Yang, Zach DeVito, Martin Raison, Alykhan Tejani, Sasank Chilamkurthy, Benoit Steiner, Lu Fang, and 2 others. 2019. Pytorch: An imperative style, high-performance deep learning library. CoRR, abs/1912.01703.

Colin Raffel, Noam Shazeer, Adam Roberts, Katherine Lee, Sharan Narang, Michael Matena, Yanqi Zhou, Wei Li, and Peter J. Liu. 2023. Exploring the limits of transfer learning with a unified text-to-text transformer. Preprint, arXiv:1910.10683.

Keisuke Sakaguchi, Ronan Le Bras, Chandra Bhagavatula, and Yejin Choi. 2021. Winogrande: An adversarial winograd schema challenge at scale. Communications ofthe ACM, 64(9):99–106.

Siddharth Samsi, Dan Zhao, Joseph McDonald, Baolin Li, Adam Michaleas, Michael Jones, William Bergeron, Jeremy Kepner, Devesh Tiwari, and Vijay Gadepally. 2023. From words to watts: Benchmarking the energy costs of large language model inference. In 2023 IEEE high performance extreme computing conference (HPEC), pages 1–9. IEEE.

W. Shao, M. Chen, Z. Zhang, and 1 others. 2023. Omniquant: Omnidirectionally calibrated quantization for large language models. arXiv preprint arXiv:2308.13137.

Yuantian Shao, Yuanteng Chen, Peisong Wang, Jianlin Yu, Jing Lin, Zhihui Wei, Jian Cheng, and 1 others. 2026. Dartquant: efficient rotational distribution calibration for llm quantization. Advances in Neural Information Processing Systems, 38:143936–143970.

Yuxuan Sun, Ruikang Liu, Haoli Bai, Han Bao, Kang Zhao, Yuening Li, Jiaxin Hu, Xianzhi Yu, Lu Hou, Chun Yuan, and 1 others. 2024. Flatquant: Flatness matters for llm quantization. arXiv preprint arXiv:2410.09426.

Hugo Touvron, Louis Martin, Kevin Stone, Peter Albert, Amjad Almahairi, Yasmine Babaei, Nikolay Bashlykov, Soumya Batra, Prajjwal Bhargava, Shruti Bhosale, and 1 others. 2023. Llama 2: Open foundation and fine-tuned chat models. arXiv preprint arXiv:2307.09288.

Thomas Wolf, Lysandre Debut, Victor Sanh, Julien Chaumond, Clement Delangue, Anthony Moi, Pierric Cistac, Tim Rault, Rémi Louf, Morgan Funtowicz, and Jamie Brew. 2019. Huggingface’s transformers: State-of-the-art natural language processing. CoRR, abs/1910.03771.

G. Xiao, J. Lin, M. Seznec, and 1 others. 2023. Smoothquant: Accurate and efficient post-training quantization for large language models. In International Conference on Machine Learning, pages 38087–38099.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, and 1 others. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Rowan Zellers, Ari Holtzman, Yonatan Bisk, Ali Farhadi, and Yejin Choi. 2019. Hellaswag: Can a machine really finish your sentence? In Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics.

## A Sensitivity Derivation

In this section, we provide a detailed derivation of the noise-weighted local functional sensitivity metric introduced in Section 3.2. Our objective is to quantify the expected distortion on the local target output $y _ { p } = f _ { p } ( z _ { p } )$ caused by introducing a quantization perturbation $\epsilon _ { p }$ to the activation $z _ { p }$ at insertion point p.

Given a calibration set C, the local functional distortion is defined as the expected distance between the outputs before and after the perturbation:

$$
\boldsymbol { \mathcal { D } } _ { p } = \mathbb { E } \boldsymbol { c } \big [ D _ { p } \big ( f _ { p } ( z _ { p } + \epsilon _ { p } ) , f _ { p } ( z _ { p } ) \big ) \big ] ,\tag{13}
$$

where $D _ { p }$ is a distance metric (e.g., Mean Squared Error). To analyze the impact of a small perturbation $\epsilon _ { p } ,$ we perform a second-order Taylor expansion of the distance function around the unperturbed activation $z _ { p }$

Let $L ( z _ { p } ) ~ = ~ D _ { p } \big ( f _ { p } ( z _ { p } ) , f _ { p } ( z _ { p } ) \big )$ Trivially, when there is no perturbation, the distance is zero, meaning the zeroth-order term is $L ( z _ { p } ) ~ = ~ 0 .$ Furthermore, since $z _ { p }$ is the global minimum of this distance function with respect to itself, the first-order derivative (gradient) is also zero, i.e., $\nabla L ( z _ { p } ) = \mathbf { 0 }$ . Therefore, the dominant term in the Taylor expansion is the second-order term:

$$
D _ { p } \big ( f _ { p } ( z _ { p } + \epsilon _ { p } ) , f _ { p } ( z _ { p } ) \big ) \approx \frac { 1 } { 2 } \epsilon _ { p } ^ { \top } H _ { p } \epsilon _ { p } ,\tag{14}
$$

where $H _ { p } = \nabla _ { z _ { p } } ^ { 2 } D _ { p } \big ( f _ { p } ( z _ { p } ) , f _ { p } ( z _ { p } ) \big )$ is the Hessian matrix of the distance metric with respect to the input activation $z _ { p }$

Next, we take the expectation over both the calibration data distribution and the quantization noise. By applying the cyclic permutation property of the trace operator (Tr), we can rewrite the expected distortion as:

$$
\begin{array} { r l } & { \mathcal { D } _ { p } \approx \mathbb { E } _ { \mathcal { C } , \epsilon _ { p } } \left[ \frac { 1 } { 2 } \mathrm { T r } \left( \epsilon _ { p } ^ { \top } H _ { p } \epsilon _ { p } \right) \right] } \\ & { \quad \quad = \mathbb { E } _ { \mathcal { C } } \left[ \frac { 1 } { 2 } \mathrm { T r } \left( H _ { p } \cdot \mathbb { E } _ { \epsilon _ { p } } [ \epsilon _ { p } \epsilon _ { p } ^ { \top } ] \right) \right] . } \end{array}\tag{15}
$$

In standard quantization theory, it is commonly assumed that the quantization noise across different channels is zero-mean and mutually independent. Consequently, the covariance matrix of the quantization noise $\bar { \mathbb { E } } _ { \epsilon _ { p } } [ \epsilon _ { p } \epsilon _ { p } ^ { \top } ]$ is a diagonal matrix, where the diagonal elements represent the noise variance $\sigma _ { p , j } ^ { 2 }$ of each channel:

$$
\mathbb { E } _ { \epsilon _ { p } } [ \epsilon _ { p } \epsilon _ { p } ^ { \top } ] = \mathrm { d i a g } ( \sigma _ { p , 1 } ^ { 2 } , \sigma _ { p , 2 } ^ { 2 } , \dots , \sigma _ { p , d } ^ { 2 } ) .\tag{16}
$$

We define the functional sensitivity matrix as half of the expected Hessian: $\begin{array} { r } { G _ { p } = \frac { 1 } { 2 } \mathbb E _ { \mathcal { C } } [ H _ { p } ] } \end{array}$ . Because the noise covariance matrix is diagonal, the trace operation eliminates all off-diagonal elements of $G _ { p }$ . The total distortion $\mathcal { D } _ { p }$ can thus be diagonalized and decomposed into a linear sum of independent contributions from each channel:

$$
\begin{array} { l } { { \displaystyle { \mathcal D } _ { p } \approx \mathrm { T r } \left( G _ { p } \cdot \mathrm { d i a g } ( \sigma _ { p , 1 } ^ { 2 } , \ldots , \sigma _ { p , d } ^ { 2 } ) \right) } } \\ { { \displaystyle \quad = \sum _ { j = 1 } ^ { d } G _ { p , j j } \sigma _ { p , j } ^ { 2 } } . } \end{array}\tag{17}
$$

From this, we naturally derive the sensitivity definition for the j-th channel:

$$
s _ { p , j } = G _ { p , j j } \sigma _ { p , j } ^ { 2 } .\tag{18}
$$

This formulation decouples the functional sensitivity $( G _ { p , j j }$ , the diagonal element of the expected Hessian, reflecting the model’s structural reliance on the channel) from the quantization noise intensity $( \sigma _ { p , j } ^ { 2 } .$ , reflecting the information loss incurred during quantization).

Extension to Weight Quantization. When the quantization of the weight matrix $W _ { p }$ is also considered, the local output becomes $y _ { p } = f _ { p } ( z _ { p } , W _ { p } )$ and the perturbation includes both activation noise $\epsilon _ { p }$ and weight noise $\delta _ { p }$ . Assuming that the activation quantization noise and weight quantization noise are independent (i.e., $\mathbb { E } [ \epsilon _ { p } \delta _ { p } ^ { \top } ] = \mathbf { 0 } )$ , the expectation of the cross-terms in the second-order Taylor expansion evaluates to zero. Therefore, the total distortion can be approximated as the linear superposition of the activation distortion and the weight distortion, yielding the additive approximation used in our framework:

$$
s _ { p , j } \approx s _ { p , j } ^ { a c t } + s _ { p , j } ^ { w } .\tag{19}
$$

## B Long-tail Effect in Local Paths

As shown in Figure 6(b) and (c), local paths such as the QK and VO paths exhibit a severe long-tail distribution, drastically deviating from the uniform diagonal. For example, the top 20% of channels in the QK path dominate approximately 70% of the total sensitivity. This extreme concentration indicates that most of the quantization distortion is caused by a tiny fraction of critical channels, providing direct empirical support for our design of allocating expressive learnable rotations exclusively to these high-sensitivity channels.

![](images/c91081f3cdfaa622286ec3eb2eabc4cfba4a839d0214f1c90f75c10b6f94e5ec.jpg)  
(a)

![](images/03ffae8160039cc5f35bd541f3e4ed8a2b8051b867f36a3472b785866b410f25.jpg)

![](images/e467882e3c05a29059b67f8d081cc10f7b2db02afdc99e9e35062a019209c9e8.jpg)  
(b)

(c)  
![](images/d1df82a7ff647de794f7baeddf6089a8ac08cc169a482d3788f8a14086707271.jpg)  
(d)  
Figure 6: Lorenz curves of intra-layer sensitivity distribution.

## C Hadamard Transforms for Non-Power-of-Two Dimensions

The fast Walsh–Hadamard transform (FWHT) is only defined for power-of-two dimensions, whereas the low-sensitivity dimension $d _ { \mathrm { l o } }$ is in general arbitrary. Following the implementation of QuaRot, we factorize the dimension as

$$
d _ { \mathrm { l o } } = K \cdot 2 ^ { m } ,\tag{20}
$$

where K is the size of a known Hadamard matrix construction, $K \in \mathsf { \Omega }$ {1, 12, 20, 28, 36, 40, 52, 60, 108, 140, 156, 172}, and build the transform as a Kronecker product:

$$
\mathbf { H } _ { \mathrm { l o } } = \mathbf { H } _ { K } \otimes \mathbf { H } _ { 2 ^ { m } } .\tag{21}
$$

In computation, the input is reshaped into K groups of size $2 ^ { m }$ , transformed by the FWHT along the $2 ^ { m }$ dimension, and then multiplied by the small dense matrix ${ \bf H } _ { K }$ , with a total cost of $\mathcal { O } ( d _ { \mathrm { l o } } \log 2 ^ { m } )$ plus one $K \times K$ matrix product. Since the Kronecker product of two Hadamard matrices is itself a Hadamard matrix, $\mathbf { H } _ { \mathrm { l o } }$ is a complete orthogonal transform over all $d _ { \mathrm { l o } }$ channels, i.e., the outlier energy is mixed across the entire low-sensitivity subspace rather than within isolated fragments.

When $d _ { \mathrm { l o } }$ is a power of two, $K = 1$ and the transform degenerates to a single FWHT without any additional matrix product, which is the most efficient case. Under our default 50% split on LLaMA-3-8B, the hidden-dimension paths use $d _ { \mathrm { l o } } = 2 0 4 8 ( K = 1 )$ , and the down-projection input uses $d _ { \mathrm { l o } } = 7 1 6 8 = 2 8 \times 2 5 6 ( K = 2 8 )$ , matching the fused kernels described in Appendix F. The set of valid $d _ { \mathrm { l o } }$ values is therefore $\{ K \cdot 2 ^ { m } \}$ over the supported K, which forms a dense grid over the sweep range in Table $5 ;$ ratios are rounded to the nearest valid dimension.

## D Detailed Results on Downstream Tasks

As shown in Table 6, LIRQuant consistently outperforms prior SOTA methods across all evaluated models and quantization configurations. Under the W4A16 setting, LIRQuant comprehensively surpasses strong baselines such as GPTQ and AWQ, minimizing the performance gap with full-precision (FP16) models in zero-shot reasoning. In the highly challenging W4A4KV4 ultralow-bit setting, traditional smoothing methods (e.g., SmoothQuant) suffer from severe performance collapse on specific tasks (e.g., LAMBADA), whereas our approach maintains a substantial lead. Particularly on the notoriously difficult-to-quantize LLaMA-3-8B model, LIRQuant achieves the highest average accuracy of 71.60%, directly surpassing the latest advanced methods including FlatQuant (71.33%) and SpinQuant (69.11%). Achieving this optimal accuracy firmly validates the effectiveness of our core design.

Table 7 reports the per-task zero-shot accuracy on Mistral-7B-v0.3, Qwen3-8B, and Mixtral-8×7B under W4A4KV4, complementing the summary in Table 2.

## E Detailed Analysis of Calibration Overhead

This section analyzes the calibration bottlenecks of each baseline in Table 3. All experiments are conducted on a single-node 8×NVIDIA H20 (96 GB per device) platform, with all competing methods using their official open-source implementations and quantization uniformly configured as W4A4KV4. Both GPU and host memory are reported per device to avoid incomparability arising from data-parallel replicas.

SpinQuant learns rotations via end-to-end crossentropy optimization with Cayley-SGD on the Stiefel manifold. However, its official implementation performs on-the-fly weight re-rotation in float64 for every linear layer with per-sample training (batch size 1), resulting in a calibration time as high as 9 hours, the slowest among all methods. Its memory advantage (26 GB) stems from frozen weights, gradient checkpointing, and DDP with full per-device replicas, requiring neither a teacher model nor activation caching.

<table><tr><td>Model</td><td>#Bits (W-A-KV)</td><td>Method</td><td>ARC-c</td><td>ARC-e</td><td>HellaS.</td><td>Lam.</td><td>PIQA</td><td>WinoG.</td><td>Avg.</td></tr><tr><td rowspan="8">LLaMA-2-7B</td><td rowspan="4">4-16-16</td><td>RTN</td><td>42.15</td><td>67.59</td><td>72.34</td><td>67.18</td><td>76.50</td><td>66.69</td><td>65.41</td></tr><tr><td>GPTQ</td><td>42.49</td><td>69.53</td><td>73.83</td><td>67.61</td><td>77.64</td><td>68.43</td><td>66.59</td></tr><tr><td>AWQ</td><td>44.11</td><td>70.75</td><td>74.98</td><td>70.68</td><td>78.13</td><td>69.38</td><td>68.01</td></tr><tr><td>LIRQuant (Ours)</td><td>44.80</td><td>73.60</td><td>75.30</td><td>73.90</td><td>79.10</td><td>68.80</td><td>69.25</td></tr><tr><td rowspan="6">RTN</td><td></td><td>27.22</td><td>27.06</td><td>27.34</td><td>0.93</td><td>49.51</td><td>50.51</td><td>30.43</td></tr><tr><td>SmoothQuant</td><td>26.37</td><td>25.63</td><td>27.05</td><td>1.11</td><td>51.90</td><td>34.49</td><td>30.07</td></tr><tr><td>QuaRot</td><td>41.43</td><td>69.32</td><td>72.50</td><td>70.66</td><td>77.42</td><td>64.64</td><td>66.00</td></tr><tr><td>SpinQuant</td><td>40.44</td><td>71.08</td><td>73.51</td><td>70.66</td><td>76.88</td><td>65.82</td><td>66.40</td></tr><tr><td>OSTQuant</td><td>42.92</td><td>72.56</td><td>73.14</td><td>71.76</td><td>77.42</td><td>66.77</td><td>67.43</td></tr><tr><td>FlatQuant</td><td>43.00</td><td>71.21</td><td>73.31</td><td>72.06</td><td>77.53</td><td>67.72</td><td>67.47</td></tr><tr><td rowspan="8">LLaMA-2-13B</td><td rowspan="4">4-16-16</td><td>LIRQuant (Ours)</td><td>43.20 42.92</td><td>71.50</td><td>73.50</td><td>72.20</td><td>77.70 76.93</td><td>67.90 65.35</td><td>67.67</td></tr><tr><td>RTN</td><td>49.63</td><td>66.54</td><td>66.62</td><td>68.99</td><td>78.51</td><td>70.64</td><td>64.56 70.28</td></tr><tr><td>GPTQ AWQ</td><td>48.63</td><td>73.95 78.16</td><td>73.77 78.48</td><td>73.20 75.20</td><td>79.54</td><td>72.45</td><td>72.08</td></tr><tr><td>LIRQuant (Ours)</td><td>49.00</td><td>76.50</td><td>78.50</td><td>76.80</td><td>80.50</td><td>72.20</td><td>72.25</td></tr><tr><td rowspan="6"></td><td>RTN</td><td>27.82</td><td>26.52</td><td>26.27</td><td>0.02</td><td>49.78</td><td>49.17</td><td>29.93</td></tr><tr><td>SmoothQuant</td><td>24.49</td><td>33.00</td><td>30.70</td><td>2.70</td><td>53.81</td><td>51.07</td><td>32.63</td></tr><tr><td>QuaRot</td><td>47.27</td><td>73.91</td><td>75.33</td><td>73.53</td><td>79.27</td><td>69.06</td><td>69.73</td></tr><tr><td>SpinQuant</td><td>46.67</td><td>74.49</td><td>75.22</td><td>72.19</td><td>78.29</td><td>67.72</td><td>69.10</td></tr><tr><td>OSTQuant</td><td>47.10</td><td>75.21</td><td>76.71</td><td>75.14</td><td>78.67</td><td>68.03</td><td>70.14</td></tr><tr><td>FlatQuant</td><td>48.38</td><td>76.94</td><td>77.88</td><td>76.40</td><td>79.65</td><td>70.56</td><td>71.64</td></tr><tr><td rowspan="7"></td><td rowspan="4">4-16-16</td><td>LIRQuant (Ours)</td><td>48.80</td><td>77.30</td><td>78.20</td><td>76.90</td><td>80.10</td><td>71.00</td><td>72.05</td></tr><tr><td>RTN</td><td>48.98</td><td>73.23</td><td>75.90</td><td>63.85</td><td>78.40</td><td>73.16</td><td>68.92</td></tr><tr><td>GPTQ</td><td>49.74</td><td>72.52</td><td>68.34</td><td>46.69</td><td>78.78</td><td>71.82</td><td>64.65</td></tr><tr><td>AWQ</td><td>52.22</td><td>76.68</td><td>77.51</td><td>74.81</td><td>79.60</td><td>71.67</td><td>72.08</td></tr><tr><td>LIRQuant (Ours) RTN</td><td>53.10 23.72</td><td>80.10</td><td>78.10</td><td>76.70</td><td>81.00</td><td>74.10</td><td>73.85</td></tr><tr><td rowspan="7">4-4-4</td><td>SmoothQuant</td><td>23.55</td><td>30.56 28.96</td><td>29.83 28.90</td><td>2.70 1.44</td><td>52.45</td><td>50.20</td><td>31.58</td></tr><tr><td>QuaRot</td><td>42.83</td><td></td><td></td><td></td><td>51.09</td><td>50.36</td><td>30.72</td></tr><tr><td></td><td></td><td>67.42</td><td>72.66</td><td>66.93</td><td>75.73</td><td>66.22</td><td>65.30</td></tr><tr><td>SpinQuant</td><td>46.33</td><td>73.57</td><td>75.43</td><td>71.40</td><td>79.16</td><td>68.75</td><td>69.11</td></tr><tr><td>OSTQuant</td><td>49.32</td><td>76.73</td><td>76.01</td><td>70.77</td><td>78.51</td><td>69.22</td><td>70.09</td></tr><tr><td>FlatQuant</td><td>50.51</td><td>75.88</td><td>76.49</td><td>73.20</td><td>79.00</td><td>72.93</td><td>71.33</td></tr><tr><td>LIRQuant (Ours)</td><td>50.80</td><td>76.10</td><td>76.70</td><td>73.50</td><td>79.30</td><td>73.20</td><td>71.60</td></tr><tr><td rowspan="8">LLaMA-2-70B</td><td rowspan="4">4-16-16</td><td>RTN</td><td>55.80</td><td>79.29</td><td>81.78</td><td>75.51</td><td>81.94</td><td>76.48</td><td>75.13</td></tr><tr><td>GPTQ</td><td>56.91</td><td>80.81</td><td>82.47</td><td>79.06</td><td>82.75</td><td>77.51</td><td>76.59</td></tr><tr><td>AWQ</td><td>56.67</td><td>80.54</td><td>82.54</td><td>78.83</td><td>82.97</td><td>77.62</td><td>76.53</td></tr><tr><td>LIRQuant (Ours)</td><td>57.60</td><td>81.60</td><td>84.10</td><td>80.00</td><td>82.90</td><td>78.10</td><td>77.38</td></tr><tr><td rowspan="6">4-4-4</td><td>RTN SmoothQuant</td><td>30.38 24.15</td><td>27.74 33.88</td><td>26.12</td><td>0.02</td><td>51.74</td><td>52.49</td><td>31.42</td></tr><tr><td>QuaRot</td><td>56.48</td><td>80.56</td><td>31.75 81.93</td><td>7.14 79.16</td><td>54.95</td><td>52.17</td><td>34.01 76.19</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td>82.21</td><td>76.80</td><td></td></tr><tr><td>SpinQuant</td><td>56.31</td><td>80.64</td><td>82.36</td><td>79.41</td><td>82.21</td><td>76.16</td><td>76.18</td></tr><tr><td>OSTQuant</td><td>56.58</td><td>80.17</td><td>82.49</td><td>78.72</td><td>82.76</td><td>76.49</td><td>76.20</td></tr><tr><td>FlatQuant</td><td>56.40</td><td>80.09</td><td>82.91</td><td>80.01</td><td>82.92</td><td>76.87</td><td>76.53</td></tr><tr><td></td><td>LIRQuant (Ours)</td><td>56.70</td><td>80.40</td><td>83.20</td><td>80.40</td><td>83.20</td><td>77.20</td><td>76.85</td></tr><tr><td>Model</td><td>Method</td><td>Wiki↓</td><td>ARC-c</td><td>ARC-e</td><td>HellaS.</td><td>Lam.</td><td>PIQA</td><td>WinoG.</td><td>Avg.</td></tr><tr><td rowspan="5">Mistral-7B-v0.3</td><td>SpinQuant</td><td>5.82</td><td>46.9</td><td>77.2</td><td>78.9</td><td>73.2</td><td>78.9</td><td>69.7</td><td>70.80</td></tr><tr><td>OSTQuant</td><td>5.79</td><td>47.2</td><td>77.4</td><td>79.1</td><td>73.6</td><td>79.2</td><td>70.1</td><td>71.10</td></tr><tr><td>DartQuant</td><td>5.92</td><td>46.4</td><td>76.7</td><td>78.5</td><td>72.8</td><td>78.5</td><td>69.2</td><td>70.35</td></tr><tr><td>FlatQuant</td><td>5.76</td><td>47.5</td><td>77.7</td><td>79.3</td><td>73.9</td><td>79.5</td><td>70.4</td><td>71.38</td></tr><tr><td>LIRQuant (Ours)</td><td>5.70</td><td>47.8</td><td>78.1</td><td>79.7</td><td>74.3</td><td>79.8</td><td>70.9</td><td>71.77</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>SpinQuant</td><td>7.99</td><td>52.8</td><td>79.5</td><td>74.1</td><td>64.2</td><td>76.6</td><td>68.2</td><td>69.23</td></tr><tr><td>OSTQuant</td><td>7.77</td><td>53.3</td><td>80.0</td><td>74.6</td><td>64.7</td><td>77.1</td><td>68.8</td><td>69.70</td></tr><tr><td>DartQuant</td><td>8.19</td><td>52.3</td><td>78.9</td><td>73.5</td><td>63.6</td><td>76.1</td><td>67.5</td><td>68.65</td></tr><tr><td>FlatQuant</td><td>7.44</td><td>54.1</td><td>80.8</td><td>75.4</td><td>65.6</td><td>77.8</td><td>69.7</td><td>70.57</td></tr><tr><td>LIRQuant (Ours)</td><td>7.32</td><td>54.4</td><td>81.2</td><td>75.8</td><td>66.1</td><td>78.2</td><td>70.3</td><td>71.00</td></tr><tr><td rowspan="3">Mixtral-8×7B</td><td>DartQuant</td><td>5.06</td><td>58.60</td><td>82.45</td><td>81.78</td><td>77.32</td><td>81.55</td><td>73.85</td><td>75.93</td></tr><tr><td>FlatQuant</td><td>4.27</td><td>59.30</td><td>83.21</td><td>82.92</td><td>78.13</td><td>82.26</td><td>74.90</td><td>76.79</td></tr><tr><td>LIRQuant (Ours)</td><td>4.14</td><td>59.54</td><td>83.65</td><td>83.48</td><td>78.55</td><td>82.84</td><td>75.62</td><td>77.27</td></tr></table>

Table 6: Detailed zero-shot accuracy on downstream tasks for LLaMA-2 and LLaMA-3 families under W4A16KV16 and W4A4KV4 quantization.

Table 7: Detailed zero-shot accuracy and WikiText-2 perplexity on Mistral-7B-v0.3, Qwen3-8B, and Mixtral-8×7B under W4A4KV4. SpinQuant and OSTQuant results on Mixtral are unavailable due to out-of-memory errors during calibration.

OSTQuant compresses the calibration time to about 44 minutes via top-K KL distillation and perhead orthogonal transforms. However, its end-toend training superimposes a per-step full-precision teacher forward pass and full-weight float32 rerotation, yielding the highest per-device peak GPU memory (33–60 GB). Its ∼420 GB host memory is a transient spike during initialization (float64 LayerNorm folding), with the steady state falling back to ∼16 GB. On Mixtral, the end-to-end framework must simultaneously hold the quantized model, the teacher, and the re-rotation buffers, exceeding the total GPU memory and causing OOM.

FlatQuant adopts layer-wise Kronecker affine reconstruction with strictly sequential training across layers, precluding multi-device parallelism and taking about 1h13m. On Mixtral, the 8-expert activation cache drives the host memory to 109 GB and the wall-clock time to 3h12m.

DartQuant achieves fast calibration by trading memory for computation: it decouples rotation learning from the model and trains on cached activations. The calibration time on dense models is only about 30 minutes, but it requires ∼200 GB of sustained host memory (as opposed to OSTQuant’s transient spike), and the global rotation uses only a single device. To train Mixtral on our cluster, we modified its caching mechanism, yet the host memory still reached 304 GB.

LIRQuant adopts the same Stiefel optimizer as OSTQuant and retains end-to-end training, but reduces the number of learnable parameters to one quarter of a dense rotation. This directly cuts the per-step backpropagation computation and optimizer state size, effectively controlling GPU memory overhead; fewer learnable parameters also naturally lower the risk of overfitting and reduce the number of steps required for convergence. In our implementation, we further apply FSDP weight sharding and use float32 rotations to circumvent the float64 DGEMM bottleneck and host memory spikes. As a result, LIRQuant achieves the best calibration time and host memory on every evaluated model; on Mixtral, DartQuant attains a lower GPU footprint only because it trains a single global rotation on cached activations, at the cost of 304 GB host memory.

## F Inference Efficiency Details

All inference experiments are conducted on a single NVIDIA A100-SXM4-80GB, with the FP16 unquantized model as the baseline. We evaluate end-to-end performance through two experiments: Experiment 1 fixes the batch size at 4 and sweeps the sequence length (128–8192) to examine prefill/decode speedup and memory savings as a function of sequence length; Experiment 2 fixes the prefill length at 2048 and sweeps the batch size (1–64) to examine decode speedup scalability with concurrency.

## F.1 Kernel Design

LIRQuant retains only two online rotations: ${ \bf R } _ { 1 }$ on Q/K after $\mathrm { R o P E }$ and ${ \bf R } _ { 3 }$ before the down projection; $\mathbf { R } _ { G }$ and $\mathbf { R } _ { 2 } ( V  W _ { O } )$ are absorbed offline into weights. We adopt the block-diagonal structure blkdiag $( M _ { h i } , H _ { l o } )$ —only high-sensitivity dimensions use the learnable dense sub-block $M _ { h i }$ while the rest use the fixed Hadamard transform $H _ { l o }$ —reducing computation and memory traffic while preserving expressiveness. The kernels are designed at three levels.

Operator Fusion. Both sub-blocks in ${ \bf R } _ { 3 }$ reduce to Kronecker structures $( y _ { d } = L ^ { \top } ( x _ { d } R _ { r } )$ $H _ { 7 1 6 8 } = H _ { 2 8 } \otimes H _ { 2 5 6 } )$ . We fuse the block-diagonal transform, per-token reduction, symmetric INT4 quantization, and dual-nibble packing into a single Triton program: the dense and Hadamard inputs are loaded from global memory, transformed on-chip with all intermediates in registers, jointly reduced, quantized and packed online, and written back in a single pass together with the per-token scales. The output is directly consumable by the downstream INT4 linear layers. ${ \bf R } _ { 1 }$ uses an isomorphic design with the dense and Hadamard blocks pre-assembled on the host into a single 128 × 128 block, requiring no kernel branching. Hadamard cores are cached on the GPU keyed by size, device, and dtype, eliminating CPU-side reconstruction and CPU→GPU synchronization while maintaining FP16 I/O.

GEMV Adaptation and Tiling. Under small-M decoding, we extend the one-program-pertoken-row mapping to one program per $B _ { M }$ token rows (auto-tuned), reusing the transform core and using 3D tiled tl.dot to improve Tensor Core occupancy. Per-layer quantization and dequantization are merged into the INT4 GEMM prologue/epilogue, eliminating fragmented kernel launches of the roughly seven linear layers per Transformer block.

Schedule Optimization. For fixed-shape decoding, we use CUDA Graph to staticize the batch, KV cache buckets, and per-layer shapes, reducing Python scheduling and kernel launch latency. Packed activation, scale, and transform buffers are pre-allocated and reused per batch/sequence bucket to reduce allocator fragmentation.

## F.2 Full Results

Tables 8 and 9 report the sequence-length sweep on Mistral-7B-v0.3 and Qwen3-8B, exhibiting the same trends as LLaMA-3-8B (Table 4). Table 10 reports the batch-size sweep with the prefill length fixed at 2048. The decode speedup grows steadily with the batch size, from below 1× at batch size 1 (where the workload is memory-latency bound and the online transformations are not amortized) to $1 . 7 9 \times - 1 . 8 5 \times$ at batch size 64. Overall, LIRQuant achieves a $2 . 2 { \times } { - } 2 . 4 { \times }$ prefill speedup and $3 . 3 \times -$ 3.5× peak memory savings; the configuration with batch size 64, sequence length 2048, and 256 decoding steps runs stably within 80 GB.

<table><tr><td>Seq. Len.</td><td>Prefill Speedup</td><td>FP16 Prefill</td><td>Decode Speedup</td><td>Mem. Saving</td></tr><tr><td>128</td><td>2.21×</td><td>31.5 ms</td><td>0.97×</td><td>3.55×</td></tr><tr><td>256</td><td>2.23×</td><td>45.1 ms</td><td>1.04×</td><td>3.52×</td></tr><tr><td>1024</td><td>2.28×</td><td> $1 5 5 . 4 \mathrm { m s }$ </td><td>1.17×</td><td>3.48×</td></tr><tr><td>4096</td><td>2.32×</td><td> $6 1 8 . 2 \mathrm { m s }$ </td><td>1.30×</td><td>3.42×</td></tr><tr><td>8192</td><td>2.35×</td><td> $1 3 0 1 . 7 \mathrm { m s }$ </td><td>1.42×</td><td>3.35×</td></tr></table>

Table 8: Inference efficiency on Mistral-7B-v0.3 (batch size 4, NVIDIA A100-80GB).

<table><tr><td>Seq. Len.</td><td>Prefill Speedup</td><td>FP16 Prefill</td><td>Decode Speedup</td><td>Mem. Saving</td></tr><tr><td>128</td><td>2.15×</td><td>34.2 ms</td><td>0.93×</td><td>3.45×</td></tr><tr><td>256</td><td>2.17×</td><td>47.0 ms</td><td>1.00×</td><td>3.42×</td></tr><tr><td>1024</td><td>2.22×</td><td>164.0 ms</td><td>1.13×</td><td>3.38×</td></tr><tr><td>4096</td><td>2.26×</td><td>652.1 ms</td><td>1.26×</td><td>3.33×</td></tr><tr><td>8192</td><td>2.29×</td><td>1388.6 ms</td><td>1.38×</td><td>3.30×</td></tr></table>

Table 9: Inference efficiency on Qwen3-8B (batch size 4, NVIDIA A100-80GB).

<table><tr><td>Batch</td><td> $\mathbf { L L a M A } { - } 3 { - } 8 \mathbf { B }$ </td><td>Mistral-7B</td><td>Qwen3-8B</td></tr><tr><td>1</td><td> $2 . 1 8 \times / 0 . 7 5 \times$ </td><td> $2 . 2 1 \times / 0 . 7 7 \times$ </td><td> $2 . 1 5 \times / 0 . 7 3 \times$ </td></tr><tr><td>16</td><td> $2 . 2 8 \times / 1 . 1 0 \times$ </td><td> $2 . 3 1 \times / 1 . 1 2 \times$ </td><td> $2 . 2 5 \times / 1 . 0 8 \times$ </td></tr><tr><td>32</td><td> $2 . 3 3 \times / 1 . 5 0 \times$ </td><td> $2 . 3 6 \times / 1 . 5 3 \times$ </td><td> $2 . 3 0 \times / 1 . 4 7 \times$ </td></tr><tr><td>64</td><td> $2 . 3 8 \times / 1 . 8 2 \times$ </td><td> $2 . 4 1 \times / 1 . 8 5 \times$ </td><td> $2 . 3 5 \times / 1 . 7 9 \times$ </td></tr></table>

Table 10: Prefill / decode speedup versus batch size (prefill length fixed at 2048, NVIDIA A100-80GB).

## G Full Ratio Sweep and Sensitivity Metric Comparison

We conduct a full sweep of the high-sensitivity channel ratio r on LLaMA-3-8B under W4A4KV4, comparing our noise-weighted functional sensitivity against six common channel importance metrics. The calibration set is independently re-sampled for this sweep (the same setting as Table 5), so the absolute PPL values may differ slightly from Table 1. The metrics are defined as follows, where ${ \boldsymbol { x } } _ { c , t }$ denotes the activation of channel c at token t:

• maxabs $\begin{array} { r } { ( L _ { \infty } ) \colon s _ { c } = \operatorname* { m a x } _ { t } | x _ { c , t } | } \end{array}$ , the maximum absolute value of channel c across all tokens;

• l2: $s _ { c } = \sqrt { \sum _ { t } x _ { c , t } ^ { 2 } }$ , the L2 norm of channel activations;

• awq: $s _ { c } = \textstyle \sum _ { t } | x _ { c , t } |$ , the L1 norm of channel activations, derived from $\mathrm { \bf A W Q ^ { \prime } s }$ channel importance measure;

• p999: $s _ { c } = Q _ { 0 . 9 9 9 } ( \{ | x _ { c , t } | \} )$ , the 99.9th percentile of channel activations, intended to provide a more robust estimate of extreme values than maxabs;

the high-sensitivity block. The training overhead difference among the metrics is on the order of minutes, which is negligible relative to the resulting accuracy gains.

• kurtosis: $s _ { c } ~ = ~ ( \sum _ { t } x _ { c , t } ^ { 4 } ) / ( \sum _ { t } x _ { c , t } ^ { 2 } ) ^ { 2 }$ , the scale-invariant kurtosis, measuring the heavytailedness of the distribution;

• random: a random permutation, independent of any data statistics.

<table><tr><td>Ratio</td><td>Ours</td><td>maxabs</td><td>12</td><td>awq</td><td>p999</td><td>kurtosis</td><td>random</td></tr><tr><td>0%</td><td>11.59</td><td>11.59</td><td>11.59</td><td>11.59</td><td>11.59</td><td>11.59</td><td>11.59</td></tr><tr><td>10%</td><td>13.28</td><td>13.42</td><td>13.87</td><td>14.26</td><td>14.92</td><td>11.33</td><td>11.52</td></tr><tr><td>20%</td><td>8.92</td><td>9.15</td><td>9.55</td><td>10.41</td><td>10.78</td><td>10.85</td><td>11.08</td></tr><tr><td>30%</td><td>7.55</td><td>7.68</td><td>8.10</td><td>8.78</td><td>9.05</td><td>10.52</td><td>10.72</td></tr><tr><td>40%</td><td>7.32</td><td>7.45</td><td>7.82</td><td>8.35</td><td>8.58</td><td>10.48</td><td>10.58</td></tr><tr><td>50%</td><td>6.78</td><td>6.90</td><td>7.25</td><td>7.88</td><td>8.12</td><td>10.28</td><td>10.38</td></tr><tr><td>60%</td><td>6.82</td><td>6.88</td><td>7.32</td><td>7.92</td><td>8.18</td><td>10.32</td><td>10.35</td></tr><tr><td>70%</td><td>6.88</td><td>6.95</td><td>7.40</td><td>7.98</td><td>8.22</td><td>10.45</td><td>10.42</td></tr><tr><td>80%</td><td>6.85</td><td>6.81</td><td>7.28</td><td>7.90</td><td>8.15</td><td>10.38</td><td>10.48</td></tr><tr><td>90%</td><td>6.92</td><td>7.00</td><td>7.45</td><td>8.02</td><td>8.28</td><td>10.51</td><td>10.52</td></tr><tr><td>100%</td><td>8.01</td><td>7.99</td><td>8.64</td><td>8.22</td><td>8.42</td><td>8.33</td><td>10.27</td></tr></table>

Table 11: WikiText-2 PPL of LLaMA-3-8B (W4A4KV4) as a function of the high-sensitivity ratio under different channel selection metrics. The 0% row corresponds to the full-Hadamard configuration.

As shown in Table 11, our proposed metric achieves the best PPL at every ratio in the practically relevant range (20%–70%), including the optimal 50% split, and remains within 0.04 PPL of the best metric at larger ratios (80%–100%). The reason maxabs $( L _ { \infty } )$ can approach our performance is its fidelity to extreme outliers: higher-order norms better highlight extreme values $( L _ { 1 } \leq L _ { 2 } \leq L _ { \infty } )$ L1 (awq) is diluted by numerous moderate values, incorrectly ranking uniformly-high but outlierfree channels ahead of genuinely spiky ones; L2, despite its quadratic amplification, remains confounded by the cumulative contribution of moderate values; p999, in pursuit of robustness, truncates the top 0.1% of extreme values—precisely those that the Hadamard transform most needs to disperse; kurtosis, by virtue of scale invariance, discards magnitude information entirely, misclassifying channels with tiny magnitude but pointy distributions as highly sensitive. Kurtosis and random exhibit no diffusion loss peak at small ratios—they fail to concentrate outliers into the high-sensitivity block, but the Hadamard transform on the low-sensitivity block still serves as a robust lower bound, keeping PPL within a reasonable range while wasting the learning opportunity of