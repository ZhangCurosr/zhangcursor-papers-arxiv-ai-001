# FP4Quant: Eficient 4-Bit Communication Compression for Model-Parallel LLM Training

<sub>Man</sub> <sub>Liu</sub>1[0009-0006-7390-0220]<sub>,</sub> <sub>Xingchen</sub> <sub>Liu</sub>2[0009-0006-9410-5365]<sub>,</sub> <sub>Hairui</sub>

<sub>Zhao</sub>2[0009-0008-7081-5172]<sub>,</sub> <sub>Guangming</sub> <sub>Tan</sub>2[0000-0002-6361-5948]<sub>,</sub> <sub>and</sub>

<sub>Dingwen</sub> <sub>Tao</sub>2[0000-0001-5422-4497]

<sup>1</sup> University of Chinese Academy of Sciences, Beijing, China 2 Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China liuman24@mails.ucas.ac.cn, liuxingchen232@mails.ucas.ac.cn, zhaohairui@ict.ac.cn, tgm@ict.ac.cn, taodingwen@ict.ac.cn

Abstract. The rapid scaling of large language models (LLMs) has made communication overhead a critical bottleneck in distributed model-parallel training. Although existing low-precision compression methods can efectively reduce communication costs, further reducing the communication footprint to ultra-low-bit formats such as FP4 introduces new challenges, including limited numerical range, sub-byte storage overhead, and increased computational cost from fine-grained quantization. In this paper, we present FP4Quant, a micro-grouped FP4 compression framework for eficient model-parallel LLM training. FP4Quant integrates a microgrouped dual-scale FP4 quantization mechanism to preserve local tensor distribution characteristics, a warp-cooperative FP4 packing scheme to enable eficient sub-byte data representation, and a hybrid two-level FWHT pipeline to mitigate quantization errors with GPU-aware execution optimization. Experiments on NVIDIA H100 GPUs with Qwen-7B demonstrate up to 1.56× end-to-end throughput improvement over BF16, revealing a trade-of between throughput and model quality.

Keywords: FP4 Quantization · Communication Compression · Distributed Training · Large Language Models.

## 1 Introduction

In recent years, large language models (LLMs) have advanced rapidly, with parameter scales growing exponentially to reach the trillion-parameter regime [5, 8, 12]. Training such models requires hundreds to thousands of GPUs, making model parallelism(MP) essential for eficiently distributing computational workloads across devices, typically through tensor parallelism (TP) and pipeline parallelism (PP) [20, 17]. As model sizes continue to increase, communication overhead, particularly AllReduce operations in TP, has increasingly become a critical performance bottleneck in distributed LLM training [22, 20, 23, 13].

To mitigate communication bottlenecks, a variety of optimization approaches have been proposed and actively explored in recent years. For DP, gradient compression techniques, including quantization [15, 3, 4, 27], sparsification [2, 26], and local accumulation [24], have been extensively studied to reduce communication costs. For TP, frameworks such as TACO [19] demonstrate that adaptive FP8-based compression can significantly reduce communication overhead while maintaining near-lossless training accuracy. More broadly, FP8-based methods (e.g., FP8-LM [21] and COAT [29]) improve training eficiency by extending low-precision representations to gradients, optimizer states, and activations.

However, the limited compression ratio of FP8 is often insuficient to fully alleviate communication bottlenecks at scale, motivating more aggressive low-bit approaches in distributed settings. While 4-bit quantization ofers the potential for substantially higher compression, it introduces fundamental challenges beyond simple precision scaling [9, 10]. Recent eforts have explored FP4 techniques for model computation [6, 7, 1] and low-bit communication in distributed training [15, 11]. Yet, these methods primarily target computation, DP or PP, leaving TP communication under extreme low-bit constraints largely unexplored.

Despite its potential to reduce communication overhead, directly applying coarse-grained 4-bit quantization to model-parallel communication faces three key challenges. First, FP4’s narrow dynamic range causes long-tail values to overflow and near-zero values to round to zero, distorting gradient directions and destabilizing early-stage training [28, 19]. Second, sub-byte storage requires packing two 4-bit values per byte, while naive packing incurs additional data movement and bit-manipulation overhead, limiting communication eficiency [16]. Third, reducing 4-bit quantization error requires fine-grained quantization, but finer granularity increases scaling-factor computation, potentially making quantization itself a system bottleneck [30]. To address these challenges, we propose FP4Quant, a micro-grouped FP4 compression framework that jointly optimizes representation fidelity and GPU execution eficiency for scalable LLM distributed training. The key contributions are summarized as follows:

(1) We propose a micro-grouped dual-scale FP4 quantization mechanism to preserve local tensor distributions under ultra-low-bit compression. Micro-grouping captures local variations, while dual-scale adaptive scaling and max-absolute scaling better match the FP4 dynamic range and reduce overflow.

(2) We introduce a hybrid two-level FWHT pipeline to support finegrained FP4 compression with limited computational overhead. Intra-warp butterfly operations are eficiently implemented using register-level shufles, while cross-warp transformations are executed using shared memory, reducing transformation overhead and enabling eficient execution on modern GPUs.

(3) We further develop a warp-cooperative FP4 packing scheme for eficient sub-byte FP4 storage. The kernel exploits warp-level primitives to exchange quantized values and performs nibble-level packing directly in registers, reducing intermediate memory operations and enabling aligned byte-wise storage.

(4) Experiments on NVIDIA H100 clusters show that FP4Quant improves end-to-end training throughput by up to 1.56× compared to BF16, while maintaining comparable convergence performance on Qwen models. FP4Quant provides an eficient and scalable solution for ultra-low-bit communication in modelparallel LLM training under practical TP and PP configurations.

![](images/92fb006daf2338d24ea5488e5d9d52ccdf1579cea99b5c5573ff9ba91c2d484f.jpg)  
Fig. 1. Example distributed training configuration with model parallelism, with 4 tensor-parallel groups, and 2 pipeline stages.

## 2 Background and Preliminaries

## 2.1 Model-Parallel LLM Training

Model parallelism, including TP and PP, is widely adopted in large-scale LLM training to distribute computation and model parameters across GPUs. As shown in Fig. 1, TP partitions individual layers across devices and relies on frequent collective communication operations (e.g., AllReduce) to exchange intermediate tensors [23], leading to substantial overhead as model size and the degree of parallelism increase [18]. In contrast, PP divides the model into stages and transfers activations and gradients between them, where communication eficiency directly afects pipeline utilization and overall throughput [14]. To alleviate these costs, recent studies have explored low-precision compression techniques to reduce communication volume while preserving training accuracy [21, 19]. However, extending these techniques to ultra-low-bit formats (e.g., FP4) remains challenging, as it is dificult to balance high compression ratios with stable training convergence.

## 2.2 Low-Precision Compression for LLM Training

Recent studies have explored 4-bit floating-point quantization to improve the eficiency of large-scale LLM training. METIS [6] reduces quantization error via spectral-domain FP4 design for accurate W4A4G4 training, while Quartet [7] enables native FP4 training through hardware-aware optimization. NVIDIA’s NVFP4 pre-training work [1] demonstrates scalable FP4 training with optimized scaling and mixed-precision scheduling. Subsequent eforts further enhance stability by addressing FP4’s limited dynamic range, including diferentiable quantization and outlier-aware scaling [25], as well as adaptive mixed-precision strategies across model components and training stages [31].

Beyond computation-oriented FP4 training, several studies have explored low-bit communication compression techniques for distributed LLM training. SDP4Bit [15] investigates 4-bit gradient communication in sharded data-parallel training and shows that substantial communication reduction can be achieved with limited accuracy degradation. Tahquant [11] explores low-bit activation compression for pipeline-parallel training to reduce inter-stage communication overhead. However, existing low-precision techniques primarily focus on computation acceleration, DP or PP communication, whereas the application of ultralow-bit quantization to TP communication remains relatively underexplored.

![](images/ac3602f8545e87792b8d59fc275bc2edc4bc580fd50632567ba31856f90d0320.jpg)  
Fig. 2. Overview of FP4Quant.

## 2.3 NVFP4 Representation and Scaling

NVFP4 is a 4-bit floating-point format designed for eficient AI workloads, retaining sign, exponent, and mantissa components. To compensate for its limited dynamic range, NVFP4 employs block-level scaling, in which multiple FP4 elements share a higher-precision scaling factoGiven an input tensor X, NVFP4 represents it as $X \approx s \cdot X _ { F P 4 }$ , where $X _ { F P 4 }$ denotes the quantized FP4 values and s is the corresponding scaling factor. NVFP4 defines two formats, E2M1 and E3M0 [1], both following the representation $x = ( - 1 ) ^ { s } 2 ^ { e - B } ( 1 + f )$ . E2M1 uses one sign bit, two exponent bits, and one mantissa bit, whereas E3M0 allocates one sign bit and three exponent bits without a mantissa. Despite this design, the extremely limited representation results in rapidly increasing quantization step sizes, leading to larger errors compared with higher-precision formats.

Although block-level scaling improves accuracy over naive FP4 quantization, fixed-size blocks remain insuficient for capturing fine-grained variations in LLM tensors. Large blocks are more sensitive to outliers, whereas smaller blocks incur higher scaling overhead. This trade-of motivates the development of adaptive scaling strategies and eficient GPU implementations to enable accurate and practical NVFP4-based communication.

![](images/8b8f7a96f963e7e32d589b863c00ca4241c91ae46a0cbe5f12e1b6cc3141ba3f.jpg)  
Fig. 3. Overview of the hybrid two-level FWHT pipeline.

## 3 FP4Quant: Ultra-Low-Precision Communication Framework

Motivated by the challenges of ultra-low-bit communication, we propose FP4Quant, a micro-grouped FP4 compression framework integrated into the collective communication path. As illustrated in Fig. 2, the compression workflow consists of three main components. Section 3.1 describes the micro-grouped dual-scale FP4 quantization mechanism (①); Section 3.2 presents the hybrid two-level FWHT pipeline (②); and Section 3.3 introduces warp-cooperative FP4 packing (③). At the receiver, the compressed payload is reconstructed through unpacking, dequantization, and the inverse FWHT. Together, these components address the numerical and execution challenges of FP4-based communication compression for model-parallel LLM training.

## 3.1 Micro-Grouped Dual-Scale FP4 Quantization

FP4Quant adopts a micro-grouped dual-scale scheme to address the limited dynamic range of NVFP4, combining fine-grained grouping with two-stage scaling for accurate distribution adaptation and eficient FP4 encoding.

Micro-grouping. Each communication tensor is partitioned into contiguous microgroups of G elements, where G ∈ 32, 64, 128, 256, 512. Each group is processed independently to capture localized statistics, enabling fine-grained and adaptive scaling while preserving high-eficiency GPU execution.

Dual-scale quantization. For each group k, FP4Quant first computes the second moment $\begin{array} { r } { \sigma _ { k } ^ { 2 } = \frac { 1 } { G } \sum _ { i } x _ { k , i } ^ { 2 } } \end{array}$ and derives an adaptive scale

$$
\alpha _ { k } = \mathrm { c l i p } \left( \frac { R } { \sqrt { \sigma _ { k } ^ { 2 } + \lambda } } , 1 0 ^ { - 3 } , 1 0 ^ { 3 } \right) , \qquad \tilde { x } _ { k , i } = \alpha _ { k } x _ { k , i } ,\tag{1}
$$

<table><tr><td>d=4</td><td>0</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td></tr><tr><td>Original(v)</td><td> $\veebar { \mathbf { \eta } } _ { 0 }$ </td><td> $\mathsf { v } _ { \uparrow }$ </td><td> $\mathsf { v } _ { 2 }$ </td><td> $\veebar { \mathbf { 3 } }$  一一</td><td> $\underline { { \mathsf { v } _ { 4 } } }$ </td><td> $\mathtt { v } _ { 5 }$ </td><td> $\veebar { \mathfrak { s } } _ { 6 }$ </td><td> $\mathsf { v } _ { 7 }$ </td></tr><tr><td>Partner</td><td> $\pmb { 4 }$ </td><td>5</td><td>6</td><td>7</td><td>0</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Received(n)</td><td> $\underline { { \mathsf { v } _ { 4 } } }$ </td><td> $\mathbf { \Delta } \mathbf { v } _ { 5 }$ </td><td> $\veebar { \mathfrak { e } }$ </td><td> $\mathsf { v } _ { 7 }$ </td><td> $\veebar { \mathbf { \eta } } _ { \mathfrak { o } }$ </td><td> $\mathsf { v } _ { \uparrow }$ </td><td> $\mathbf { { \boldsymbol { v } } } _ { 2 }$ </td><td> $\mathbf { { \underline { { v } } } _ { 3 } }$ </td></tr><tr><td>Operation</td><td colspan="4">(lane_  $\scriptstyle \mathbf { i d } \& \mathbf { d } ) = = 0 \longrightarrow \mathbf { v } = \mathbf { v } + \mathbf { n }$ </td><td colspan="4">-一 (lane_id&amp;d)!=0→v=n-v</td></tr><tr><td>New(v)</td><td> $\begin{array} { r } { \overline { { \mathsf { V } _ { 0 } + \mathsf { V } _ { 4 } } } \left\| \mathsf { V } _ { 1 } + \mathsf { V } _ { 5 } \right\| \nabla _ { 2 } \mathsf { + V } _ { 6 } \left\| \mathsf { V } _ { 3 } + \mathsf { V } _ { 7 } \right. } \end{array}$ </td><td></td><td></td><td></td><td> $\mathsf { v } _ { 0 } . \mathsf { v } _ { 4 }$ </td><td> $\mathsf { v } _ { 1 } \mathsf { - v } _ { 5 }$ </td><td> ${ \tt V } _ { 2 } { \tt - V } _ { 6 }$ </td><td> $\mathsf { v } _ { 3 } { \mathsf { - } } \mathsf { v } _ { 7 }$ </td></tr></table>

Fig. 4. Illustration of the butterfly operation in FWHT.

where $\lambda = 1 0 ^ { - 6 }$ avoids numerical instability. By normalizing the RMS magnitude of each group, α<sub>k</sub> maps diverse local distributions into the efective FP4 range. After the FWHT transformation in Section 3.2, the transformed values $h _ { k , i }$ are quantized using a group-wise max-absolute scale:

$$
s _ { k } = \mathrm { c l i p } \left( \frac { \operatorname* { m a x } _ { i } \vert h _ { k , i } \vert } { R } , 1 0 ^ { - 1 2 } , 1 0 ^ { 6 } \right) , \qquad q _ { k , i } = \mathrm { C v t T o F P 4 } \left( \mathrm { c l i p } ( \frac { h _ { k , i } } { s _ { k } } , - R , R ) \right) .\tag{2}
$$

Here, $\alpha _ { k }$ performs pre-rotation normalization, while $s _ { k }$ determines the final quantization range. Both scales are stored per group and used for reconstruction.

## 3.2 Hybrid Two-Level FWHT Pipeline

Motivation. FP4 quantization is sensitive to outliers, where a few extreme values can dominate the max-absolute scale and reduce efective precision. FP4Quant applies an orthonormal Walsh–Hadamard transform (FWHT) to redistribute energy across dimensions, producing a more balanced representation (see Fig. 3). The transform is self-inverse, allowing the same operations for decompression.

Two-level execution. A length-G FWHT consists of $\log _ { 2 } G$ butterfly stages. According to the GPU execution hierarchy, FP4Quant divides these operations into two levels. For intra-warp stages with butterfly distances $d < 3 2$ , paired elements are exchanged through register shufle operations without shared-memory level:

$$
n  \{ \neg { \bf s h f l \_ x o r \_ s y n c } ( 0 x f { \bf f \ / f \bf f \ / f \bf \cdot } \bf { \bf \Phi } ) , \qquad v  \{ \} { \displaystyle n \sim \cal U } \{ { \bf t i d \ } \& { \bf \ / d } \} \neq 0 ,\tag{3}
$$

which corresponds to the butterfly operation shown in Fig. 4. These stages are fully unrolled and executed in registers. For cross-warp stages with $d \ge 3 2$ intermediate results are exchanged through shared memory with synchronization barriers. Only $\log _ { 2 } ( G / 3 2 )$ such stages are required, while groups with $G \leq 3 2$ complete the entire transform using register-level operations. The final output is normalized by $1 / { \sqrt { G } }$ to obtain the orthonormal transform.. This design aligns FWHT stages with GPU communication primitives, minimizing synchronization overhead while efectively smoothing outliers for FP4 quantization.

![](images/739cc488471460f6bf6bdb0eb5d5a080b0c916e7b819bfe09f4a4c7005ed41af.jpg)  
Fig. 5. Validation loss of Qwen-7B with diferent micro-group sizes under TP and PP communication compression.

## 3.3 Warp-Cooperative FP4 Packing

FP4 codes occupy four bits, requiring two adjacent codes to be stored in one byte. A straightforward implementation may introduce intermediate memory accesses for byte assembly, where additional loads, shifts, and masks increase memory trafic and reduce the benefit of FP4 compression. FP4Quant therefore performs nibble packing directly in registers using warp-level shufle operations.

After quantization, thread i holds one 4-bit code $q _ { k , i }$ in its low nibble. Adjacent threads exchange their codes through a stride-1 butterfly shufle:

$$
\mathbf { \hat { h i } } _ { i } = \_ - { \mathbf { \hat { s h f } } } 1 _ { - } { \mathbf { x o r _ { - } s y n c } } \big ( 0 { \mathbf { x f f f f f } } , q _ { k , i } , 1 \big ) \& 0 { \mathbf { x } } 0 \mathbf { F } .\tag{4}
$$

For even lanes (i mod $2 = 0 )$ , the shufled value provides the high nibble from the adjacent thread, while the local code remains as the low nibble. Each even lane then assembles one byte as $b = \left( \mathtt { h i } _ { i } \ll 4 \right) | q _ { k , i }$ and performs a single coalesced store, whereas odd lanes remain inactive. This design avoids sharedmemory usage and additional memory round trips, producing naturally aligned byte-level writes across the warp. During decompression, the process is reversed: each thread loads the corresponding packed byte and extracts either the low or high nibble according to the parity of its index before dequantization. This warpcooperative design eliminates intermediate memory trafic and shared-memory usage, achieving near-zero packing overhead with eficient, aligned writes.

## 4 Evaluation

## 4.1 Experimental Setup

Platform. All experiments are conducted on a single node equipped with eight NVIDIA H100 GPUs (80,GB HBM3 each; driver 595.71.05; CUDA 13.2), interconnected via NVLink. FP4Quant is implemented as an NCCL compressor plugin and evaluated within a production-grade model-parallel training stack.

Table 1. Downstream accuracy on CMMLU and C-Eval (valid) with $G = 6 4$ under diferent model-parallel configurations.
<table><tr><td>Configuration Method</td><td></td><td>CMMLU</td><td>C-Eval (valid)</td></tr><tr><td>BF16 (Ring)</td><td></td><td> $0 . 5 3 6 5 \pm 0 . 0 0 4 5$ </td><td> $0 . 5 2 1 5 \pm 0 . 0 1 3 3$ </td></tr><tr><td rowspan="2">TP</td><td>FP4Quant</td><td> $0 . 5 2 8 7 \pm 0 . 0 0 4 5$ </td><td> $0 . 4 8 2 9 \pm 0 . 0 1 3 3$ </td></tr><tr><td>Tahquant</td><td> $0 . 2 5 3 0 \pm 0 . 0 0 4 0$ </td><td> $0 . 2 7 7 1 \pm 0 . 0 1 2 2$ </td></tr><tr><td rowspan="2">PP</td><td>FP4Quant</td><td> $0 . 5 2 8 8 \pm 0 . 0 0 4 5$ </td><td> $0 . 5 3 4 2 \pm 0 . 0 1 3 3$ </td></tr><tr><td>Tahquant</td><td> $0 . 5 1 7 9 \pm 0 . 0 0 4 5$ </td><td> $0 . 5 1 8 6 \pm 0 . 0 1 3 3$ </td></tr><tr><td rowspan="2">MP</td><td>FP4Quant</td><td> $0 . 5 3 0 1 \pm 0 . 0 0 4 5$ </td><td> $0 . 5 2 5 3 \pm 0 . 0 1 3 3$ </td></tr><tr><td>Tahquant</td><td> $0 . 2 5 3 4 \pm 0 . 0 0 4 0$ </td><td> $0 . 2 4 2 2 \pm 0 . 0 1 1 7$ </td></tr></table>

Model, data, and task. We train Qwen-7B on the OpenWebMath corpus. Model quality is evaluated using validation loss and the corresponding perplexity (PPL) $( { \mathrm { P P L } } = e ^ { \mathrm { l o s s } } )$ . System eficiency is measured by end-to-end training throughput $( \mathrm { T F L O P / s }$ per GPU) under diferent parallel settings. Downstream model capability is further evaluated using accuracy on CMMLU and C-Eval.

Parallel configurations. We evaluate FP4Quant under three parallel regimes: TP, PP, and their combined MP configuration in large-scale distributed training. For each regime, we investigate micro-group sizes $G \in 6 4 , 1 2 8 , 2 5 6 , 5 1 2$ and compare against two baselines: the uncompressed BF16 (Ring) and Tahquant, a representative method for activation communication compression in PP settings.

## 4.2 Convergence and Model Quality

To evaluate the convergence behavior and model quality of FP4Quant under ultra-low-bit communication compression, we measure the validation loss and perplexity across diferent micro-group sizes $( G \in \{ 6 4 , 1 2 8 , 2 5 6 , 5 1 2 \} )$ ). Table 2 summarizes the final validation metrics, while Figure 5 presents the convergence curves over 10, 000 iterations under TP and PP configurations.

The results demonstrate that FP4Quant consistently outperforms Tahquant in preserving model quality under FP4 communication compression. Under TP, Tahquant sufers from severe accuracy degradation (loss 3.4848, PPL = 32.6163), whereas FP4Quant with G = 64 achieves substantially lower loss (2.2306, PPL = 9.3050) and maintains a stable convergence trajectory close to the full-precision baseline, as illustrated in Fig. 5. Under PP, FP4Quant further demonstrates near-lossless performance (loss 2.1525, $\mathrm { P P L } = 8 . 6 1 0 5 )$ , matching BF16 accuracy and outperforming Tahquant.

In addition, FP4Quant exhibits clear sensitivity to group size, where increasing G leads to gradual degradation in TP and a sharp breakdown at large G (e.g., $G = 5 1 2 )$ in both TP and PP. This trend suggests that overly coarse grouping introduces scale mismatch and amplifies quantization error. Nevertheless, with appropriately chosen group sizes $( \mathrm { e . g . } , G \leq 2 5 6 )$ , FP4Quant achieves an excellent balance between accuracy and communication eficiency and consistently delivers superior performance compared to Tahquant.

Table 2. Comparison of training accuracy and throughput between FP4Quant and baselines (BF16 and Tahquant) under TP4 and PP on Qwen-7B.
<table><tr><td rowspan="2">Method</td><td colspan="4">TP</td><td colspan="4">PP</td></tr><tr><td>Loss↓</td><td>PPL↓</td><td>TFLOP/s↑ Speedup↑</td><td></td><td>Loss↓</td><td>PPL↓</td><td>TFLOP/s↑</td><td>Speedup↑</td></tr><tr><td>BF16 (Ring)</td><td>2.1505</td><td>8.5893</td><td>111.0</td><td>1.00×</td><td>2.1505</td><td>8.5893</td><td>189.1</td><td>1.00×</td></tr><tr><td>Tahquant</td><td>3.4848</td><td>32.6163</td><td>197.4</td><td>1.78×</td><td>2.1653</td><td>8.7174</td><td>189.6</td><td>1.00×</td></tr><tr><td>FP4Quant(G = 64)</td><td>2.2306</td><td>9.3050</td><td>155.3</td><td>1.40×</td><td>2.1525</td><td>8.6105</td><td>190.1</td><td>1.01×</td></tr><tr><td>FP4Quant(G = 128)</td><td>2.4223</td><td>11.2713</td><td>171.2</td><td>1.54×</td><td>2.1552</td><td>8.6298</td><td>191.1</td><td>1.01×</td></tr><tr><td>FP4Quant(G = 256)</td><td>3.0616</td><td>21.3613</td><td>173.3</td><td>1.56×</td><td>2.1626</td><td>8.6940</td><td>190.7</td><td>1.01×</td></tr><tr><td>FP4Quant(G = 512)</td><td>3.3512</td><td>28.5368</td><td>172.6</td><td>1.55×</td><td>3.4270</td><td>30.7830</td><td>190.3</td><td>1.01×</td></tr></table>

## 4.3 Downstream Task Accuracy

Training loss and downstream accuracy reflect diferent aspects of model performance: loss measures optimization convergence, whereas downstream tasks evaluate practical model capability. Although the two metrics are not strictly correlated, poor convergence or substantially higher loss than the baseline typically leads to degraded downstream performance. Table 1 reports evaluation results on CMMLU and C-Eval (valid) with G = 64. FP4Quant consistently preserves downstream performance across model-parallel settings, achieving scores of 0.5287/0.4829 (TP), 0.5288/0.5342 (PP), and 0.5301/0.5253 (MP), compared with the BF16 baseline (0.5365/0.5215). Notably, under PP and MP configurations, FP4Quant matches or slightly exceeds BF16 on C-Eval, demonstrating strong capability preservation under ultra-low-bit compression.

In contrast, TahQuant exhibits severe degradation under communicationintensive settings. Under TP and MP, its performance drops to near randomchance levels (0.2530/0.2771 and 0.2534/0.2422), while remaining stable only under PP. This suggests that coarse-grained 4-bit quantization is insuficient to maintain optimization quality, ultimately degrading downstream performance. Overall, FP4Quant achieves both consistently stable convergence and superior downstream fidelity compared with existing 4-bit communication methods.

## 4.4 System Eficiency

Table 2 shows that FP4Quant achieves a superior balance between training accuracy and throughput compared to BF16. Under TP, Tahquant attains the highest throughput (197.4 TFLOP/s/GPU, 1.78×), but sufers from severe accuracy degradation, indicating that its aggressive quantization compromises training fidelity. In contrast, FP4Quant substantially improves model quality while still providing significant speedup. At G = 64, FP4Quant reduces the loss to 2.2306 while achieving a 1.40× speedup over BF16, and continues to ofer a favorable trade-of at larger group sizes. For example, at G = 256, FP4Quant reaches 173.3 $\mathrm { T F L O P / s / G P U }$ while achieving significantly better accuracy than Tahquant. Although FP4Quant achieves lower throughput than Tahquant, it avoids severe accuracy degradation while delivering a better accuracy–eficiency trade-of.

Under PP, where computation dominates execution and communication overhead is less significant, all methods achieve comparable throughput (189–191 $\mathrm { T F L O P / s / G P U ) }$ . FP4Quant reaches 190.1–191.1 $\mathrm { T F L O P / s / G P U }$ , closely matching the BF16 baseline (189.1 $\mathrm { T F L O P / s / G P U ) }$ with up to 1.01× speedup, indicating that its quantization and scaling overhead is efectively hidden by computation. Unlike communication-bound TP training, PP ofers limited acceleration potential because performance is primarily determined by computation throughput. Overall, FP4Quant enables eficient low-bit communication with substantial acceleration under TP while incurring negligible runtime overhead under PP, demonstrating its efectiveness across diverse parallel configurations.

## 5 Conclusion and Future Work

In this paper, we present FP4Quant, an eficient FP4-based communication compression framework for model-parallel LLM training. It integrates micro-grouped dual-scale quantization, warp-cooperative packing, and a hybrid FWHT-based pipeline to improve quantization fidelity and system eficiency. Experiments show that FP4Quant achieves up to 1.56× end-to-end throughput improvement while preserving convergence. Our evaluation focuses on Qwen-7B on a single-node eight-GPU H100 system. Future work will investigate scalability across model sizes and multi-node environments, adaptive precision strategies, and applications to heterogeneous systems, multimodal models, and reinforcement learning.

## Acknowledgements

This work was supported by the National Natural Science Foundation of China through the Excellent Young Scientists Fund Program (Overseas) (Project No. E401470), the National Key Research and Development Program of China (Grant Nos. 2020YFA0907000 and 2025YFB30037002), the National Natural Science Foundation of China (Grant Nos. 62032023 and T2125013), and Ant Group. The AI-driven experiments, simulations, and model training were performed on the robotic AI-Scientist platform of the Chinese Academy of Sciences.

## References

1. Abecassis, F., Agrusa, A., Ahn, D., Alben, J., Alborghetti, S., Andersch, M., Arayandi, S., Bjorlin, A., Blakeman, A., Briones, E., et al.: Pretraining large language models with nvfp4. arXiv preprint arXiv:2509.25149 (2025)

2. Aji, A.F., Heafield, K.: Sparse communication for distributed gradient descent. In: Proceedings of the 2017 Conference on Empirical Methods in Natural Language Processing. p. 440–445. Association for Computational Linguistics (2017). https://doi.org/10.18653/v1/d17-1045, http://dx.doi.org/10. 18653/v1/D17-1045

3. Alistarh, D., Grubic, D., Li, J.Z., Tomioka, R., Vojnovic, M.: Qsgd: communicationeficient sgd via gradient quantization and encoding. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 1707–1718. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017)

4. Bernstein, J., Wang, Y.X., Azizzadenesheli, K., Anandkumar, A.: signsgd: Compressed optimisation for non-convex problems (2018), https://arxiv.org/abs/ 1802.04434

5. Brown, T., Mann, B., Ryder, N., Subbiah, M., Kaplan, J., Dhariwal, P., Neelakantan, A., Shyam, P., Sastry, G., Askell, A., et al.: Language models are few-shot learners (gpt-3). arXiv preprint arXiv:2005.14165 75 (2020)

6. Cao, H., Chen, M., Yang, Y., Huang, R., Dong, F., Zhou, J., Chen, A., Dong, M., Wang, Y., Hou, J., Cheng, Y., Wu, F., Yang, F., Lu, T., Gu, N., Shang, L.: Metis: Training llms with fp4 quantization (2025), https://arxiv.org/abs/2509.00404

7. Castro, R., Panferov, A., Tabesh, R., Sieberling, O., Chen, J., Nikdan, M., Ashkboos, S., Alistarh, D.: Quartet: Native fp4 training can be optimal for large language models. Advances in Neural Information Processing Systems 38, 43552– 43572 (2026)

8. Chowdhery, A., Narang, S., Devlin, J., Bosma, M., Mishra, G., Roberts, A., Barham, P., Chung, H.W., Sutton, C., Gehrmann, S., et al.: Palm: Scaling language modeling with pathways. arXiv preprint arXiv:2204.02311 (2022)

9. Dettmers, T., Lewis, M., Belkada, Y., Zettlemoyer, L.: Llm.int8(): 8-bit matrix multiplication for transformers at scale. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2022)

10. Frantar, E., Ashkboos, S., Hoefler, T., Alistarh, D.: Gptq: Accurate posttraining quantization for generative pre-trained transformers. arXiv preprint arXiv:2210.17323 (2022)

11. He, G., Cao, Y., He, Y., Bai, T., Chen, K., Yuan, K., Yuan, B.: Tah-quant: Efective activation quantization in pipeline parallelism over slow network (2026), https: //arxiv.org/abs/2506.01352

12. Hofmann, J., Borgeaud, S., Mensch, A., Buchatskaya, E., Cai, T., Rutherford, E., de Las Casas, D., Hendricks, L.A., Welbl, J., Clark, A., Hennigan, T., Noland, E., Millican, K., van den Driessche, G., Damoc, B., Guy, A., Osindero, S., Simonyan, K., Elsen, E., Vinyals, O., Rae, J.W., Sifre, L.: Training compute-optimal large language models. In: Proceedings of the 36th International Conference on Neural Information Processing Systems. NIPS ’22, Curran Associates Inc., Red Hook, NY, USA (2022)

13. Hosoki, R., Sato, K., Endo, T., Bigot, J., Audit, E.: Optimizing intralayer parallel communication for llm training on systems with fully-connected mesh gpu topology. In: Proceedings of the Supercomputing Asia and International Conference on High Performance Computing in Asia Pacific Region. p. 328–339. SCA/HPCAsia ’26, Association for Computing Machinery, New York, NY, USA (2026). https://doi.org/10.1145/3773656.3773675, https://doi.org/ 10.1145/3773656.3773675

14. Huang, Y., Cheng, Y., Bapna, A., Firat, O., Chen, M.X., Chen, D., Lee, H., Ngiam, J., Le, Q.V., Wu, Y., Chen, Z.: Gpipe: Eficient training of giant neural networks using pipeline parallelism (2019), https://arxiv.org/abs/1811.06965

15. Jia, J., Xie, C., Lu, H., Wang, D., Feng, H., Zhang, C., Sun, B., Lin, H., Zhang, Z., Liu, X., Tao, D.: Sdp4bit: toward 4-bit communication quantization in sharded data parallelism for llm training. In: Proceedings of the 38th International Conference on Neural Information Processing Systems. NIPS ’24, Curran Associates Inc., Red Hook, NY, USA (2024)

16. Kamath, A.K., Krishnamurthy, A., Canini, M., Peter, S.: Reducing the gpu memory bottleneck with lossless compression for ml. In: Proceedings of the 21st European Conference on Computer Systems. p. 899–918. EUROSYS ’26, Association for Computing Machinery, New York, NY, USA (2026). https://doi.org/10.1145/3767295.3803595, https://doi.org/10.1145/3767295. 3803595

17. Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., Catanzaro, B.: Reducing activation recomputation in large transformer models (2022), https://arxiv.org/abs/2205.05198

18. Korthikanti, V., Casper, J., Lym, S., McAfee, L., Andersch, M., Shoeybi, M., Catanzaro, B.: Reducing activation recomputation in large transformer models (2022), https://arxiv.org/abs/2205.05198

19. Liu, M., Liu, X., Tian, X., Lu, B., Lyu, S., Yin, S., Huang, W., Wei, Z., Zhao, H., Tan, G., Tao, D.: Taco: Eficient communication compression of intermediate tensors for scalable tensor-parallel llm training. In: Proceedings of the 35th International Symposium on High-Performance Parallel and Distributed Computing. p. 235–248. HPDC ’26, Association for Computing Machinery, New York, NY, USA (2026). https://doi.org/10.1145/3806645.3807584, https://doi.org/ 10.1145/3806645.3807584

20. Narayanan, D., Shoeybi, M., Casper, J., LeGresley, P., Patwary, M., Korthikanti, V., Vainbrand, D., Kashinkunti, P., Bernauer, J., Catanzaro, B., Phanishayee, A., Zaharia, M.: Eficient large-scale language model training on gpu clusters using megatron-lm. In: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. SC ’21, Association for Computing Machinery, New York, NY, USA (2021). https://doi.org/10.1145/3458817.3476209, https://doi.org/10.1145/3458817.3476209

21. Peng, H., Wu, K., Wei, Y., Zhao, G., Yang, Y., Liu, Z., Xiong, Y., Yang, Z., Ni, B., Hu, J., Li, R., Zhang, M., Li, C., Ning, J., Wang, R., Zhang, Z., Liu, S., Chau, J., Hu, H., Cheng, P.: Fp8-lm: Training fp8 large language models (2023), https://arxiv.org/abs/2310.18313

22. Rajbhandari, S., Rasley, J., Ruwase, O., He, Y.: Zero: memory optimizations toward training trillion parameter models. In: Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis. SC ’20, IEEE Press (2020)

23. Shoeybi, M., Patwary, M., Puri, R., LeGresley, P., Casper, J., Catanzaro, B.: Megatron-lm: Training multi-billion parameter language models using model parallelism (2020), https://arxiv.org/abs/1909.08053

24. Stich, S.U.: Local sgd converges fast and communicates little (2019), https:// arxiv.org/abs/1805.09767

25. Wang, R., Gong, Y., Liu, X., Zhao, G., Yang, Z., Guo, B., Zha, Z., Cheng, P.: Optimizing large language model training using fp4 quantization (2025), https: //arxiv.org/abs/2501.17116

26. Wangni, J., Wang, J., Liu, J., Zhang, T.: Gradient sparsification for communication-eficient distributed optimization. In: Proceedings of the 32nd International Conference on Neural Information Processing Systems. p. 1306–1316. NIPS’18, Curran Associates Inc., Red Hook, NY, USA (2018)

27. Wen, W., Xu, C., Yan, F., Wu, C., Wang, Y., Chen, Y., Li, H.: Terngrad: ternary gradients to reduce communication in distributed deep learning. In: Proceedings of the 31st International Conference on Neural Information Processing Systems. p. 1508–1518. NIPS’17, Curran Associates Inc., Red Hook, NY, USA (2017)

28. Wu, X., Li, C., Aminabadi, R.Y., Yao, Z., He, Y.: Understanding int4 quantization for language models: latency speedup, composability, and failure cases. In: Proceedings of the 40th International Conference on Machine Learning. ICML’23, JMLR.org (2023)

29. Xi, H., Cai, H., Zhu, L., Lu, Y., Keutzer, K., Chen, J., Han, S.: Coat: Compressing optimizer states and activations for memory-eficient fp8 training. In: International Conference on Learning Representations. vol. 2025, pp. 42989–43011 (2025)

30. Zhang, L., Fei, W., Wu, W., He, Y., Lou, Z., Zhou, H.: Dual grained quantization: Eficient fine-grained quantization for llm (2023), https://arxiv.org/abs/2310. 04836

31. Zhou, J., Tang, D., Fu, R., Hu, B., Xu, H., Wang, Y., Pei, Z., Su, Z., Liu, L., Zhang, X., Zhang, W.: Towards eficient pre-training: Exploring fp4 precision in large language models (2025), https://arxiv.org/abs/2502.11458