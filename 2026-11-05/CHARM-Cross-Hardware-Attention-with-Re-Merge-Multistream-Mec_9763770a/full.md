# CHARM+: Cross-Hardware Attention with Re-Merge Multistream Mechanism

Xinhang Zhang

Boning Zhang

Chengchun Liu

Chunpu Li

Lei Sun

Weifeng Zhang

Limin Xiao<sup>†</sup>

Lenovo Research

## Abstract

As sequence lengths continue to grow, distributed attention based on sequence parallelism (e.g., Ring-Attention) has become a mainstream solution for large language models (LLMs). Nevertheless, inherent communication dependencies limit computation–communication overlap, reducing GPU utilization. Moreover, in cross-hardware (NUMA or node) architectures, due to imbalanced communication bandwidth, frequent communication synchronization will limit overall performance. Based on these insights, we propose CHARM+, Cross-Hardware Attention with Re-Merge Multistream Mechanism. We reorganize sequential communications into several levels and achieve intra-level parallelism and computation overlap. Furthermore, we decouple cross-hardware communication from intra-hardware communication, eliminating synchronization overhead caused by bandwidth imbalance. Forward and backward algorithms were evaluated on multiple hardware architectures and consistently outperformed state-of-the-art methods. Compared to Ring-Attention, our forward and backward algorithms achieve average speedups of 1.3× and 1.7× on a dual-NUMA PCIe architecture, and 2.8× and 3.9× on a dual-node NVLink architecture, respectively.

## 1 Introduction

As large language models (LLMs) scale up [3], longer context lengths are required to fully exploit their performance [29, 24]. Notably, LLM training requires processing long documents to enhance model capabilities [5]. Meanwhile, the inference of LLMs involves cross-document understanding [1], long program comprehension [18], and multi-turn interactive dialogues [15]. To handle this, self-attention is the core mechanism for capturing long-range dependencies using queries (q), keys (k), and values (v) [9, 25]. However, its computation and memory costs scale with sequence length, making long-context training and inference expensive [13, 27]. Therefore, sequence parallelism (SP) and distributed attention have become mainstream approaches. They partition the sequence across accelerators to reduce per-device overhead, enabling longer context windows [23, 12].

Nevertheless, existing SP methods (e.g., Ring-Attention) exhibit two limitations. Firstly, due to the communication dependency, each communication relies on the completion of the preceding communication [28]. Consequently, it limits the communication-computation overlap and reduces GPU utilization. Secondly, in the cross-hardware (Non-Uniform Memory Access, NUMA or node), imbalanced bandwidth leads to extra synchronization overhead [21]. Therefore, communication dependencies cause GPUs to idle, degrading overall performance.

![](images/a33475420abcc25c0f67ce6f0a37aefd212a612ad98fbcfe3bc348e010ce8f5b.jpg)  
(a) Blocking comm due to link contention in PCIe

![](images/755323ace7f4e351d7ae95b3e9702448cf6209ccdd82c3eb29c0f6afa8f224c4.jpg)  
(b) Non-blocking comm on dedicated links in NVLink  
Figure 1: Communication under PCIe and NVLink. The left of (a) shows single communication at full bandwidth without contention, while the right shows multiple communications at limited bandwidth with contention. (b) shows multiple communications at full bandwidth without contention.

Based on these observations, we propose CHARM+, Cross-Hardware Attention with Re-Merge Multistream Mechanism. We design a merge-based communication mechanism to reorganize the sequential communications into several levels and achieve intra-level independence. Secondly, we leverage multiple communication streams to better overlap computation and communication. Furthermore, by decoupling intra- and inter-hardware communication, we avoid performance degradation due to synchronization overhead. As shown in Figure 2, GPU idle time has been significantly reduced, leading to substantial performance improvements. Our contributions are summarized as follows:

• We propose a novel communication and parallel mechanism. To optimize the ring pattern, we reorganize communications into logarithmic levels and design a multistream parallelism for kv and q to achieve intra-level parallelism.

• We develop a Cross-Hardware Query Parallelism to decouple intra- and inter-hardware communication. By isolating cross-hardware q and kv communications, the parallelism avoids synchronization induced by bandwidth imbalance and improves performance.

• We implement both forward and backward of our method and conduct extensive experiments across different hardware architectures. The results demonstrate that our forward and backward consistently outperform Ring-Attention, achieving maximum speedups of 1.4× and 1.9× on PCIe, and 3.2× and 4.3× on NVLink.

## 2 Related Work

## 2.1 Interconnection Architecture

Most large-scale training and inference are deployed on multi-GPU systems connected via either PCIe or NVLink [19, 6]. Figure 1 illustrates the interconnects of these two architectures. PCIe-based

![](images/dbce4edc64272311abc37e2f5d965739b0d54799330a5c6acc3be6321387fb31.jpg)  
Figure 2: GPU Idle Time. By setting various sequence lengths, we compare GPU idle time with Ring-Attention and TASP. Our method achieves higher GPU utilization.

systems often suffer from link contention due to concurrent communication requests, while NVLink provides high-bandwidth, dedicated links that enable contention-free communication [11]. Furthermore, both architectures have imbalanced communication bandwidth between cross-hardware (i.e., cross-NUMA or cross-node) and intrahardware [17]. Consequently, it incurs substantial synchronization overhead, severely degrading communication performance.

## 2.2 Sequence Parallelism

As long-context requirements in LLMs continue to grow, the sequence parallelism and the distributed attention mechanism have become critical techniques for scaling attention by distributing sequence computa-

tion across multiple devices [12, 23]. Existing methods (e.g., Ring-Attention) rely on distributed kv exchange, where kv blocks are transmitted across devices in a ring topology [14] and processed incrementally [16]. However, there exists a dependency in ring-based communication: each communication cannot start until the previous one completes. Moreover, each communication can only overlap with a single computation [22]. Consequently, as communication time grows, this dependency prolongs GPU idle time and reduces effective compute utilization.

![](images/d9e04fada4b1a53deed36988149333940c2d81bba5e6e19783c0dbc4bbd841f6.jpg)  
Figure 3: Overall Framework of CHARM+. (a) - (d) illustrate the complete process of the algorithm using an 8-GPU system with a dual-NUMA configuration. (a) displays parallel communication of q is first performed across NUMA. As a result, GPU 3 receives the q from GPU 7, eliminating the need for further cross-NUMA communication. (b) shows the parallel communication and computation of q and kv. In this step, GPU 3 communicates q with GPU 0 and kv with GPU 1. After completing the computation of $Q _ { 0 , 4 }$ , GPU3 communicates back to GPU 0 and receives the output. (c) performs the last step of RM+, where GPU 3 only communicates kv with GPU 2. (d) shows that once all kv results are computed, the output is communicated back to GPU 7. (e) illustrates the overall cross-NUMA communication and the full overlap of computation and communication of GPU3 happened in (a)-(d).

Several approaches have been proposed to improve the efficiency and scalability of Ring-Attention. TASP [26] aims to better utilize available communication links by adopting the all-to-all communication. Attention-2D [10] splits the self-attention computation into two dimensions along q and kv. Mesh-Attention [4] replaces one-dimensional Ring-Attention with a two-dimensional tiled assignment. Striped-Attention [2] reorders the input sequence into uniformly distributed blocks. However, these approaches do not optimize the ring-based pattern. Reducing the ring length or enabling multiple rings to communicate in parallel does not eliminate the communication dependencies, nor does it improve the communication–computation overlap. Therefore, efficient computation–communication overlap and high GPU utilization remain challenges in distributed attention.

## 3 Method

This section presents our method illustrated in Figure 3<sup>3</sup>. We first explain our motivation and introduce a merge-based communication mechanism. Secondly, we propose a multistream parallelism to improve the communication-computation pattern in the communication mechanism. Furthermore, we leverage a cross-hardware query parallelism to mitigate the imbalanced bandwidth between cross hardware (NUMA or node) communication and intra-hardware communication and use intra-pipeline to hide cross-hardware communication. All notations utilized in this section are described in Table 1.

Table 1: Notation Description.
<table><tr><td>Notation</td><td>Description</td><td>|Notation</td><td>Description</td></tr><tr><td> $Q , q$ </td><td>Query matrix</td><td> $K , k$ </td><td>Key matrix</td></tr><tr><td> $V , v$ </td><td>Value matrix</td><td>n</td><td>Degree of sequence parallelism</td></tr><tr><td> $O$ </td><td>Unhidden communication time</td><td> $\ell ( \cdot )$ </td><td>Function that clips value to zero</td></tr><tr><td> $T _ { \mathrm { c o m m } } ( \cdot )$ </td><td>Communication time of a tensor</td><td> $T _ { \mathrm { c o m p } } ( \cdot )$ </td><td>Computation time of Attention</td></tr></table>

## 3.1 Motivation

As introduced in Section 2, the current ring-based attention involves gradually passing kv through a ring path for attention computation and update. This mechanism enables partial overlap between communication and computation. However, dependencies between communications prevent a communication from starting before the previous one is complete. To maximize GPU utilization, we seek to maximize communication–computation overlap, defined as follows:

$$
\operatorname* { m i n } O = \sum _ { i = 1 } ^ { N } \ell ( T _ { \mathrm { c o m m } } ^ { i } ( k , v ) - T _ { \mathrm { c o m p } } ^ { i } ( q , k , v ) )\tag{1}
$$

Let $N$ denote the length of critical path that is non-overlapping, $T _ { \mathrm { c o m m } } ^ { i } ( \cdot )$ and $T _ { \mathrm { c o m p } } ^ { i } ( \cdot )$ are the $i _ { \mathrm { t h } }$ communication time, the $i _ { \mathrm { t h } }$ computation time of FlashAttention [8]. We define O as the unhidden communication time, where the GPU is idle. Therefore, minimizing O maximizes GPU utilization.

For Ring-Attention [16], due to the ring structure, this objective can be expressed as:

$$
O _ { \mathrm { r i n g } } = \sum _ { i = 1 } ^ { n - 1 } \ell ( T _ { \mathrm { c o m m } } ^ { i } ( k , v ) - T _ { \mathrm { c o m p } } ^ { i } ( q , k , v ) )\tag{2}
$$

This $O ( n )$ critical path dependency becomes the primary bottleneck limiting the ability to overlap communication and computation. In particular, the bandwidth imbalance between cross-hardware (NUMA or node) and intra-hardware communication causes extra communication synchronization to significantly increase the latency of each step along the critical path. Therefore, this effect further exacerbates the overall bottleneck, we further corroborate these observations with Nsight profiling results in Figure 9a of Section C.1. Based on these observations, our motivation becomes:

“How can we efficiently overlap computation and communication while simultaneously minimizing the overall critical path ofthe execution? "

As inspired by the traditional merge algorithm [20, 7], we present a communication mechanism. Compared to the ring-based kv exchange, the mechanism redesigns this process as kv aggregation. The aggregation is structured as a logarithmic-level merge (in Figure 4a, a sequence parallelism of 8 corresponds to a three-level merge). At each level i, kv are aggregated in parallel between device pairs with a stride of $2 ^ { i - 1 }$ . As communication dependencies persist between merge levels, the critical path of the merge mechanism becomes $O ( \log n )$ , which is significantly shorter than Ring-Attention.

For the merge communication mechanism, we model the communication time at each merge level as:

$$
C _ { i } ^ { ( 1 ) } = \sum _ { j = 1 } ^ { 2 ^ { i - 1 } } T _ { \mathrm { c o m m } } ^ { j } ( k , v )\tag{3}
$$

At the $i _ { \mathrm { t h } }$ merge level, the communication mechanism performs $2 ^ { i - 1 }$ communications. Meanwhile, the computation at level i processes the kv received at the communication of level $i - 1$ . Thus, KV communication at level i overlaps with computation at level i−1, whose cost is $2 ^ { \ell ( i - 2 ) }$ . Consequently, the unhidden communication time is given by:

$$
O _ { \mathrm { m e r g e } } = \sum _ { i = 1 } ^ { \log _ { 2 } ( n ) } \ell ( C _ { i } ^ { ( 1 ) } - \sum _ { j = 1 } ^ { 2 ^ { \ell ( i - 2 ) } } T _ { \mathrm { c o m p } } ^ { j } ( q , k , v ) )\tag{4}
$$

![](images/9930ae37d9458424a7c98e61ca56057dcecc1aca1f41e5a4275c52102b05a49b.jpg)  
(a) Merge Communication Mechanism.

![](images/93de08f5f928c4a5c8170f3867891698f429fb9e97a3857253caed0c1d193035.jpg)  
(b) Re-Merge Multistream Mechanism.  
Figure 4: The Computation and Communication Overlap on GPU 7. For 8 GPUs, (a) contains three levels. Each GPU communicates at groups of 1, 2, and 4 at different levels, aggregating the communicated kv while overlapping computation. (b) maintains consistency with (a) in the first levels, but further enables multistream communication and computation of q and kv at other each merge level. Each GPU concurrently communicates q and kv with different GPUs, computes the q of other GPU and communicates the output back. The red region indicates GPU idle time.

As can be observed, compared to the Ring-Attention, the merge communication mechanism restructures the linear communication path into a merge path with logarithmic level, significantly shortening the communication critical path. Each merge level accommodates more computation, increasing opportunities to overlap communication with computation and improving efficiency.

## 3.2 Re-Merge Multistream Mechanism

The merge communication mechanism reduces dependencies between ring communications, but communication at each merge level still follows the ring pattern. To further increase communication–computation overlap, we design the Re-Merge Multistream Mechanism, named RM+. It still follows the structure of the merge communication mechanism, but optimizes the communication at each merge level to enable more communication-computation overlap.

As described in Section 3.1, Ring-Attention involves communication only for kv, and each round of kv communication can start only after the previous round completes. Notably, the communication of $q$ is independent of the kv communication. By exploiting this independence, we parallelize q communication with kv communication, improving communication-computation overlap, as illustrated in Figure 4. Moreover, we aim to overlap the computation after the final merge level(e.g., $Q _ { 7 } K V _ { 0 } \cdot \cdot \cdot Q _ { 7 } \bar { K } V _ { 3 }$ in Figure 4a. During each merge level, kv is fetched sequentially from peer GPUs for computation. In addition, parallel communication of $q$ and kv further reduces the communication time of each merge level. Consequently, as illustrated in Figure 4b, we obtain:

$$
C _ { i } ^ { ( 2 ) } = \sum _ { j = 1 } ^ { 2 ^ { \ell ( i - 2 ) } } \operatorname* { m a x } \left( T _ { \mathrm { c o m m } } ^ { j } ( q ) , T _ { \mathrm { c o m m } } ^ { j } ( k , v ) \right)\tag{5}
$$

Note that, Eq. (5) follows Eq. (3). Due to the parallel communication between $q$ and $k v ,$ the communication time is bounded by the maximum. Meanwhile, advancing the computation after the final merge level increases computations at earlier merge levels, thereby reducing the unhidden communication time. Specifically, all communication except the final one at each level can be overlapped, with the last communication deferred to the next merge level. Therefore, for the $i _ { \mathrm { t h } }$ merge level, there are $2 ^ { i - 1 }$ computations. Consequently, the unhidden communication time is:

$$
O _ { \mathrm { R M + } } = \sum _ { i = 1 } ^ { \log n } \ell ( C _ { i } ^ { ( 2 ) } - \sum _ { j = 1 } ^ { 2 ^ { i - 1 } } T _ { \mathrm { c o m p } } ^ { j } ( q , k , v ) )\tag{6}
$$

Under the assumption that all parallel communications sustain identical bandwidth, Eq. (4) outperforms Eq. (2) due to its shorter critical path and greater communication overlap. However, parallel communication of multistream can lead to PCIe link contention, increasing latency. Consequently, on

the PCIe architecture, the single communication of RM+ can be longer than that in Ring-Attention. Therefore, performance on a GPU system using NVLink is significantly better than on a PCIe architecture, as further demonstrated in Section 4.4.

## 3.3 Cross-Hardware Query Parallelism

In multi-hardware architectures (NUMA or Node), each KV communication in Ring-Attention simultaneously involves both intra-hardware and cross-hardware communication. Since cross-hardware

![](images/8a564ccc21c19e018fcde7627a46f586b92b73ec53083ba7de1280cca2a21312.jpg)  
Figure 5: Cross-Hardware Query Parallelism. The complete process is illustrated with pipeline 0 as an example. Each intra-pipeline processes queries (Q) from KV Comm<sub>other</sub> <sub>pipelines</sub> <sub>and</sub> <sub>uses</sub> <sub>the</sub> <sub>same</sub> <sub>workflow</sub> <sub>with</sub> <sub>its</sub> local KV, returning the final out back. Effective overlap 1 2 3 4 5 6 7allows inter-pipeline communication to be hidden by intra-pipeline, further improving performance.

communication typically provides lower bandwidth, the faster intra-hardware communication is forced to be idle while waiting for the communication synchronization, leading to substantial performance inefficiency. As illustrated in Figure 5, we propose a novel parallelism where crosshardware communication of the q is interleaved with the intra-node pipeline. The pipelines process the current queries (Q) from different pipelines with local KV using the same computation and communication flow, as detailed in Section 3.2. Notably, since all kv and q communications occur in the pipeline come from the same NUMA or node, the bottleneck caused by synchronized communication is largely eliminated. With effective overlap illustrated in Figure 5, after all computations are completed, only a final out communi-

퐾 <sub>1</sub> 퐾 <sub>2</sub> 퐾 <sub>3</sub> 퐾 <sub>4</sub> 퐾 <sub>5</sub> 퐾 <sub>6</sub> 퐾 <sub>7cation is not hidden. Therefore, as the number of nodes increases, effective overlap ensures that</sub> only a single communication remains exposed, making the advantage of our method increasingly pronounced, which is further validated in the ablation study in Section 4.4 and Section 4.5. We provide Nsight profiling results of CHARM+ in Figure 9b of Section C.1, demonstrating that our overlap strategy significantly outperforms Ring-Attention.

## 3<sub>4 Experiments</sub>

## 4.1 Experimental Setup

� 퐾 Out퐾 ,   퐾 ,  We conduct experiments on GPU systems with different configurations and detailed specifications are provided in Table 2. We use official NCCL and RDMA test benchmarks to measure the intra-hardware

and cross-hardware unidirectional P2P bandwidth. On L20, we evaluate the performance in both cross-NUMA and intra-NUMA settings. Furthermore, we explore the impact of link contention in PCIe-based systems. On H20 and H100, we evaluate cross-node and intra-node performance and further compare results under NVLink architecture, where link contention is absent.

Table 2: The GPU Configuration and Bandwidth.
<table><tr><td rowspan="2">GPU</td><td rowspan="2">NVLink</td><td rowspan="2">Num</td><td rowspan="2">TFLOPS</td><td colspan="2">Bandwidth (GB/s)</td></tr><tr><td>Cross</td><td>Intra</td></tr><tr><td>L20</td><td>x</td><td>8</td><td>119.5</td><td>14.7</td><td>24.0</td></tr><tr><td>H20</td><td>√</td><td>16</td><td>148.0</td><td>42.6</td><td>297.2</td></tr><tr><td>H100</td><td>√</td><td>32</td><td>989.0</td><td>42.5</td><td>365.1</td></tr></table>

We evaluate our method under diverse configurations by varying batch size (B), sequence length (S), and hidden dimensions (H). Specifically, sequence lengths range from 1K to 100K (up to 200K in some scenarios), with four hidden-dimension configurations formed by combining dimensions of 12 or 20 with sizes of 128 or 192. We conduct a grid search over B, S, and H, ensuring that the product BS does not exceed 500K due to memory constraints. Each configuration is evaluated over 100 iterations following 20 warm-up iterations, with results averaged across the measured runs. To enable a unified characterization of different configurations, we use the computation-to-communication ratio (CCR) as a single metric. Following TASP [26], we define the CCR as:

![](images/e4f9bda7c2b613f765e1cf294620f656a4607af881652dbadc882ea16255fe90.jpg)

![](images/eac4b9c37bd7a533d239c2ac8e302ed6ac50a4a2f9cbb3f4ff83b5407e23b2a4.jpg)  
(a) 8×L20 GPU

![](images/b2aa42f69de650ec84b04a594a511b6662f28fa09719e5e3740edeb39c456c81.jpg)

![](images/3bf4a68573d82947cfabab4f4636e8ccfc157168f21d35c35d80ea10429a4a97.jpg)

![](images/299bf8fa4b0433128bea785d1be1ac15b832eeea35849af6d1faa28d6e6cd890.jpg)  
(b) 8×H20 GPU

![](images/35aceaf82f289550e23e12cce3908401274477d40cfd5beadee69b3bd430e4e8.jpg)

Figure 6: Single-node Experiments on 8×L20 (PCIe) and 8×H20 (NVLink). Speedup is normalized to Ring-Attention, comparing forward, backward, and end-to-end time. In different architectures, our method consistently outperforms the baseline and TASP.  
![](images/01c6c095d4892490d0203ab9fb5f8f1f40d4e0458ae5c6ecaa1a85846bfad10c.jpg)

![](images/1b344d56caf3c5e923e66b2305387bc06f19be46fa7515ad7420bbaf7d7628d4.jpg)

![](images/6b4e41e2a9d7eb08e1201709b28d2c39f48e1ce3a7a544271144135838138f76.jpg)

![](images/1f0352de9bd38a5ef356c95a4f8c76dd78cc66dd0b7ac714a7c4e4930a5d56af.jpg)

![](images/6400e5448be9e5be479668213c632f14b89f244a329c6d79e477fa9f915d44d4.jpg)

![](images/dc129293135d1c1080a3e9dc745e3930d694a0f44c5e53741830636ef9fe8b1f.jpg)  
Figure 7: Multi-node Experiments on 8×H20 and 16×H20 (NVLink). Speedup is normalized to Ring-Attention and Ring-Zigzag. In the forward (top) and the backward (bottom) comparison, our method achieves consistently superior performance. Particularly, the 2×4 and 2×8 experiments highlight that our method’s performance grows as the number of GPUs increases.

$$
C C R = \sum _ { i = 1 } ^ { N } T _ { \mathrm { c o m p } ( q , k , v ) } ^ { i } / \sum _ { i = 1 } ^ { N } T _ { \mathrm { c o m m } ( k , v ) } ^ { i }\tag{7}
$$

Here, q, k, and v are tensors that depend on the sequence length, the batch size, the hidden dimensions, and the number of attention heads. We accumulate the computation and communication time of Ring-Attention under sequential execution. The final metric is defined as the ratio of the computation time to the communication time (detailed analysis of CCR is provided in Section B.1). We compare our method against Ring-Attention [16], TASP, and their corresponding zigzag variants [2]<sup>4</sup>.

## 4.2 Single-node Evaluation

As illustrated in Figure 6, we perform single-node experiments on L20 with 8 GPUs under a dual-NUMA PCIe architecture and on H20 with 8 GPUs using NVLink. Our method outperforms TASP and Ring-Attention, even in the presence of link contention (Figure 6a). The poor performance of TASP on L20 is mainly attributed to its all-to-all collective communication, which fully utilizes all links and induces severe contention. Furthermore, in the H20 experiments, the observed speedup is slightly lower. This is due to the very high intra-node bandwidth (as reported in Table 2), which increases the CCR relative to other experiments. In other words, computation outweighs communication, limiting the benefits of the communication-computation overlap.

Table 3: Memory Overhead. By setting various parameter scales, we compare the memory overhead with Ring-Attention. Our method shows comparable overhead to Ring-Attention.
<table><tr><td>Batch Size</td><td>Seq Len</td><td>Heads</td><td>Ring (MB)</td><td>CHARM+ (MB)</td><td>Batch Size</td><td>Seq Len</td><td>Heads</td><td>Ring (MB)</td><td>CHARM+ (MB)</td></tr><tr><td>16</td><td>1024</td><td>32</td><td>2,179</td><td>2,215</td><td>2</td><td>53,760</td><td>12</td><td>13,123</td><td>13,467</td></tr><tr><td>16</td><td>1024</td><td>64</td><td>3,701</td><td>3,881</td><td>4</td><td>53,760</td><td>12</td><td>19,301</td><td>20,565</td></tr><tr><td>64</td><td>8192</td><td>32</td><td>45,079</td><td>45,423</td><td>2</td><td>26,880</td><td>12</td><td>10,745</td><td>10,803</td></tr><tr><td>128</td><td>8192</td><td>16</td><td>44,720</td><td>45,432</td><td>4</td><td>26,880</td><td>12</td><td>14,029</td><td>14,425</td></tr></table>

Memory Overhead Analysis. We evaluate memory overhead on L20 with 8 GPUs across varying sequence lengths for both forward and backward passes, comparing Ring-Attention with our method in Table 3. The results indicate that our method incurs only a modest increase in memory consumption. Furthermore, as the number of additional buffers introduced by our approach is constant (i.e., O(1)), memory overhead does not scale with the number of GPUs. See Section B.2 for detail analysis.

Numerical correctness verification. We evaluate the numerical correctness of CHARM+ by comparing its outputs and gradients against FlashAttention. The results show that CHARM+ exhibits numerical differences of the same order of magnitude as Ring-Attention across the output, log-sumexp (LSE), and Q/K/V gradients, indicating that CHARM+ introduces no substantial additional numerical error. Detailed experimental configurations and results are provided in Section C.3.

## 4.3 Multi-node Evaluation

We conducted multi-node experiments on the H20 system with 8 or 16 GPUs using NVLink. On the 16-GPU setup, we compare our method against Ring-Attention, TASP, and their load-balanced variants, Ring-Zigzag and TASP-Zigzag. As shown in Figure 7, our method shows a clear advantage

in cross-node configurations. This is because cross-node bandwidth is significantly lower than intra-node bandwidth, as listed in Table 2. Consequently, Ring-Attention and TASP incur GPU idle time due to frequent synchronized communication, resulting in a performance bottleneck that degrades efficiency.

Impact of Scaling on Performance. To evaluate scalability, we conducted experiments on the H20 system with 2 nodes using 8 GPUs and 2 nodes using 16 GPUs. As shown in Figure 7, our method achieves substantial speedups at larger scales by leveraging additional GPUs for better parallelism and effectively overlapping the communication with the computation.

## 4.4 Ablation Study

Table 4: Ablation Study. We compare the forward performance of Ring-Attention, RM+, and CHARM+ across various GPU configurations and sequence lengths. Notably, CHARM+ will be evaluated only in cross-hardware configurations.
<table><tr><td rowspan="2">GPUs</td><td rowspan="2">Cross</td><td rowspan="2">Seq</td><td colspan="3">Forward Time (ms)</td></tr><tr><td>Ring</td><td>RM+</td><td>Ours</td></tr><tr><td rowspan="2">4 × L20</td><td rowspan="2">x</td><td>1792</td><td>9.35</td><td>10.29</td><td rowspan="2">一 一</td></tr><tr><td>7168</td><td>36.86</td><td>39.66</td></tr><tr><td rowspan="2">8 × L20</td><td rowspan="2">√</td><td>1792</td><td>17.61</td><td>25.91</td><td>14.81</td></tr><tr><td>7168</td><td>69.85</td><td>98.49</td><td>52.40</td></tr><tr><td rowspan="2">4 × H20</td><td rowspan="2">x</td><td>1792</td><td>5.38</td><td>4.76</td><td>一</td></tr><tr><td>7168</td><td>38.62</td><td>33.20</td><td>一</td></tr><tr><td rowspan="2">8 × H20</td><td rowspan="2">x</td><td>1792</td><td>5.51</td><td>3.65</td><td>一</td></tr><tr><td>7168</td><td>26.79</td><td>13.84</td><td>一</td></tr><tr><td rowspan="2">16 × H20</td><td rowspan="2">√</td><td>1792</td><td>32.17</td><td>39.04</td><td>12.16</td></tr><tr><td>7168</td><td>84.58</td><td>98.98</td><td>28.91</td></tr></table>

The effectiveness of the Re-Merge Multistream Mechanism (RM+). We evaluate RM+

across sequence lengths on a single 8-GPU H20 node to isolate its effectiveness from link contention and cross-node communication. As shown in Table 4, RM+ consistently outperforms Ring-Attention. This improvement stems from the parallel communication enabled by multiple independent q and kv streams. As a result, RM+ achieves a maximum speedup of 1.9× over Ring-Attention.

The effectiveness of Cross-Hardware Query Parallelism. We compare CHARM+ against Ring-Attention and RM+ in both cross-NUMA and cross-node configurations. Specifically, for cross-

Table 5: Performance on H100. Note that TASP requires the sequence length to be divisible by n(n − 1). Therefore, the sequence lengths for the four-node and two-node settings differ slightly.
<table><tr><td rowspan="2">B</td><td rowspan="2">Seq</td><td colspan="4">Forward</td><td colspan="4">Backward</td></tr><tr><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td></tr><tr><td colspan="10"> $\mathbf { G P U s } = \mathbf { 2 } \times \mathbf { 8 }$ </td></tr><tr><td>16</td><td>26,880</td><td>149.97</td><td>98.01</td><td>63.93</td><td>2.35×</td><td>449.58</td><td></td><td>141.52</td><td>3.18×</td></tr><tr><td>16</td><td>13,440</td><td>72.65</td><td>49.27</td><td>31.84</td><td>2.28×</td><td>226.46</td><td></td><td>65.35</td><td>3.47×</td></tr><tr><td>64</td><td>6,720</td><td>141.75</td><td>97.65</td><td>61.80</td><td>2.29×</td><td>448.84</td><td>一</td><td>123.32</td><td>3.64×</td></tr><tr><td>128</td><td>3,360</td><td>140.55</td><td>97.15</td><td>61.54</td><td>2.28×</td><td>450.01</td><td></td><td>121.99</td><td>3.69×</td></tr><tr><td colspan="10"> $\mathbf { G P U s } = 4 \times 8$ </td></tr><tr><td>16</td><td>31,744</td><td>185.42</td><td>183.22</td><td>68.17</td><td>2.72×</td><td>559.34</td><td></td><td>146.01</td><td>3.83×</td></tr><tr><td>16</td><td>15,872</td><td>89.27</td><td>93.30</td><td>34.64</td><td>2.58×</td><td>279.85</td><td></td><td>70.62</td><td>3.96×</td></tr><tr><td>64</td><td>7,936</td><td>179.13</td><td>182.93</td><td>66.48</td><td>2.69×</td><td>556.25</td><td></td><td>134.48</td><td>4.14×</td></tr><tr><td>128</td><td>3,968</td><td>178.51</td><td>183.09</td><td>66.43</td><td>2.69×</td><td>560.63</td><td></td><td>133.05</td><td>4.21×</td></tr></table>

NUMA experiments, we use 8 GPUs on the L20 system, while for cross-node experiments, we use 16 GPUs on the H20 system. As shown in Table 4, RM+ suffers significant performance degradation in cross-hardware configurations, whereas CHARM+ achieves substantial improvements, with up to 1.9× and 3.5× speedup in cross-NUMA and cross-node configurations, respectively. The performance gain is smaller on L20 than on H20 due to link contention on L20, as discussed in the next subsection. The performance drop of RM+ is primarily due to a bandwidth imbalance between intra- and crosshardware communication, as well as increased cross-node communication resulting from multistream parallelism. In contrast, CHARM+ decouples inter-node and intra-node communication to reduce synchronization overhead and improve overall performance.

The impact of link contention in PCIe architectures. To avoid communication synchronization issues caused by cross-NUMA communication on the L20 system, we conducted single-NUMA experiments on an L20 node with 4 GPUs, and performed comparison experiments on an H20 node with 4 GPUs connected via NVLink, which eliminates link contention.

## 4.5 Evaluation on High-end GPU

As shown in Table 4, on L20, RM+ suffers significant performance degradation due to link contention, performing worse than Ring-Attention. In contrast, RM+ exhibits much better performance on H20.

![](images/f4d40180f50d174d337cd0a3727ee790e840dedc977e101a63f9b388b31c31ef.jpg)  
Figure 8: Performance of RM+ under PCIe and NVLink. We evaluate the forward time of RM+ on 4×H20 GPUs (dashed) and 4×L20 GPUs (solid) across different batch sizes (bs). Compared to Ring-Attention, performance on L20 is lower than on H20 due to the link contention. TASP, as shown in Figure 6a.

We further conduct a comprehensive evaluation with various batch sizes and sequence lengths, summarized in Figure 8. The results indicate that link contention degrades performance in almost all configurations. The exception occurs when both the batch size and sequence length are small.

In this case, the bandwidth is not fully utilized, so the link contention is negligible. Meanwhile, it is worth noting that in the 8× L20 experiments, link contention leads to smaller performance gains than on H20. Nevertheless, enabled by Cross-Hardware Query Parallelism, CHARM+ still significantly outperforms Ring-Attention and

To completely validate our method on the high-end GPUs, we rented 32 H100 GPUs equipped with high-bandwidth NVLink. We conducted full forward and backward experiments under both 2×8 and 4×8 configurations, and the results are reported in Table 5 (detailed results can be found in Section C.2). When TASP begins to leverage high-performance GPUs, our method still maintains a leading performance, achieving an average speedup of 2× in the forward and 3× in the backward compared to ring-attention. Additionally, as described in Section 3.3, our method consistently makes

Table 6: End-to-end Performance Comparison of Ring-Attention and CHARM+ with Model Scaling
<table><tr><td rowspan="2">Batch Size</td><td rowspan="2">Seq Len</td><td rowspan="2">Model</td><td colspan="2">Ring</td><td colspan="2">CHARM+</td><td rowspan="2">Speedup</td></tr><tr><td>Latency</td><td>Memory</td><td>Latency</td><td>Memory</td></tr><tr><td>16</td><td>2,048</td><td>GPT-350M</td><td>10082.6</td><td>11,372</td><td>4847.0</td><td>11,518</td><td>2.08×</td></tr><tr><td>8</td><td>4,096</td><td>GPT-350M</td><td>10032.6</td><td>11,348</td><td>4871.5</td><td>11,544</td><td>2.06×</td></tr><tr><td>4</td><td>8,192</td><td>GPT-350M</td><td>9914.5</td><td>11,436</td><td>5173.6</td><td>11,552</td><td>1.92×</td></tr><tr><td>2</td><td>16,384</td><td>GPT-350M</td><td>10089.5</td><td>12,112</td><td>5637.3</td><td>12,170</td><td>1.79×</td></tr><tr><td>16</td><td>2,048</td><td>GPT-1.3B</td><td>21505.5</td><td>30,816</td><td>11738.3</td><td>31,602</td><td>1.83×</td></tr><tr><td>8</td><td>4,096</td><td>GPT-1.3B</td><td>22073.7</td><td>31,190</td><td>11768.1</td><td>31,546</td><td>1.88×</td></tr><tr><td>4</td><td>8,192</td><td>GPT-1.3B</td><td>21857.8</td><td>31,414</td><td>12097.5</td><td>32,030</td><td>1.81×</td></tr><tr><td>2</td><td>16,384</td><td>GPT-1.3B</td><td>22064.5</td><td>32,270</td><td>12796.6</td><td>32,312</td><td>1.72×</td></tr></table>

intermediate cross-node communication effectively overlapped by the intra-node pipeline. In contrast, other approaches suffer from synchronization constraints, making them susceptible to the slower cross-node communication. Consequently, our method exhibits greater performance as the scale increases (speedups of $2 . 2 8 \times / 3 . 6 9 \bar { \times }$ on 16 H100 GPUs and $2 . 6 \bar { 9 } \times / 4 . 2 \bar { 1 } \times$ on 32 H100 GPUs at maximum performance), highlighting the advantage of our method.

## 4.6 End-to-end Training Performance

High-end GPU. We integrate our operator into Megatron-LM and compare it with Megatron’s ring implementation. The experimental results are shown in Table 7. We use context parallelism (CP) for training on 16 high-end H20/H100 GPUs and report the average per-iteration time in milliseconds.

Table 7: End-to-end Performance Comparison of Ring-Attention and CHARM+ with High-end GPU
<table><tr><td rowspan="2">Batch Size</td><td rowspan="2">Seq Len</td><td rowspan="2">Model</td><td colspan="2">Ring</td><td colspan="2">CHARM+</td><td colspan="2">Speedup</td></tr><tr><td>H20</td><td>H100</td><td>H20</td><td>H100</td><td>H20</td><td>H100</td></tr><tr><td>16</td><td>16,384</td><td>GPT3-350M</td><td>7304.6</td><td>3385.1</td><td>5601.3</td><td>1925.7</td><td>1.30×</td><td>1.76×</td></tr><tr><td>32</td><td>8,192</td><td>GPT3-350M</td><td>7193.7</td><td>3291.5</td><td>4669.7</td><td>1679.2</td><td>1.54×</td><td>1.96×</td></tr><tr><td>64</td><td>4,096</td><td>GPT3-350M</td><td>7169.6</td><td>3234.6</td><td>4452.5</td><td>1538.6</td><td>1.61×</td><td>2.10×</td></tr><tr><td>128</td><td>2,048</td><td>GPT3-350M</td><td>7143.6</td><td>3210.8</td><td>4242.5</td><td>1482.0</td><td>1.68×</td><td>2.17×</td></tr></table>

Notably, Megatron-LM includes extensive optimizations for ring attention that our method does not incorporate. Moreover, non-attention linear layers dominate end-to-end training, limiting the impact of distributed attention optimization. Despite this, our method still significantly outperforms ringattention on H20 and H100, achieving max 1.68× and 2.17× speedup, respectively. Furthermore, our method performs better at larger batch sizes, making it more suitable for real-world workloads.

Model scaling. To evaluate the scalability of CHARM+ across model sizes, we conduct additional end-to-end experiments on 16 L20 GPUs with model sizes of 350M and 1.3B, measuring iteration latency (ms) and GPU memory consumption (MB). Although model configurations affect the relative computational costs of attention and MLP layers, they do not fundamentally change the communication patterns targeted by CHARM+. As shown in Table 6, CHARM+ achieves consistent speedups of 1.72×–2.08× across both model sizes and all tested configurations, with nearly identical memory overhead. These results demonstrate the effectiveness of CHARM+ across different model scales.

## 5 Conclusion

In this work, we present CHARM+, a novel approach that reorganizes the communication process of Ring-Attention using a merge-based strategy. Furthermore, it leverages a multistream parallelism of q and kv to reduce communication dependencies and decouples cross-hardware and intra-hardware communication. We implement both the forward and backward algorithms of the mechanism and conduct extensive experiments and ablation studies across various settings. Experiment results across diverse hardware platforms and GPUs with varying computational throughput demonstrate that CHARM+ consistently achieves significant performance improvements over state-of-the-art methods.

## References

[1] Y. Bai, S. Tu, J. Zhang, H. Peng, X. Wang, X. Lv, S. Cao, J. Xu, L. Hou, Y. Dong, et al. Longbench v2: Towards deeper understanding and reasoning on realistic long-context multitasks. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 3639–3664, 2025.

[2] W. Brandon, A. Nrusimha, K. Qian, Z. Ankner, T. Jin, Z. Song, and J. Ragan-Kelley. Striped attention: Faster ring attention for causal transformers. arXiv preprint arXiv:2311.09431, 2023.

[3] T. Brown, B. Mann, N. Ryder, M. Subbiah, J. D. Kaplan, P. Dhariwal, A. Neelakantan, P. Shyam, G. Sastry, A. Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[4] S. Chen, J. Chen, S. Zhu, Z. Jiang, Y. Peng, and X. Qian. Mesh-attention: A new communicationefficient distributed attention with improved data locality. arXiv preprint arXiv:2512.20968, 2025.

[5] Y. Chen, S. Qian, H. Tang, X. Lai, Z. Liu, S. Han, and J. Jia. Longlora: Efficient fine-tuning of long-context large language models. arXiv preprint arXiv:2309.12307, 2023.

[6] J. Choquette, E. Lee, R. Krashinsky, V. Balan, and B. Khailany. 3.2 the a100 datacenter gpu and ampere architecture. In 2021 IEEE International Solid-State Circuits Conference (ISSCC), volume 64, pages 48–50. IEEE, 2021.

[7] B. Citko. On the nature of merge: External merge, internal merge, and parallel merge. Linguistic Inquiry, 36(4):475–496, 2005.

[8] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. Advances in neural information processing systems, 35:16344– 16359, 2022.

[9] P. Doukhan, G. Oppenheim, and M. Taqqu. Theory and applications oflong-range dependence. Springer Science & Business Media, 2002.

[10] V. Elango. Attention2d: Communication efficient distributed self-attention mechanism. arXiv preprint arXiv:2503.15758, 2025.

[11] A. Li, S. L. Song, J. Chen, J. Li, X. Liu, N. R. Tallent, and K. J. Barker. Evaluating modern gpu interconnect: Pcie, nvlink, nv-sli, nvswitch and gpudirect. IEEE Transactions on Parallel and Distributed Systems, 31(1):94–110, 2019.

[12] S. Li, F. Xue, C. Baranwal, Y. Li, and Y. You. Sequence parallelism: Long sequence training from system perspective. In Proceedings of the 61st Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 2391–2404, 2023.

[13] T. Li, G. Zhang, Q. D. Do, X. Yue, and W. Chen. Long-context llms struggle with long in-context learning. arXiv preprint arXiv:2404.02060, 2024.

[14] X. Li. Niching without niching parameters: particle swarm optimization using a ring topology. IEEE Transactions on Evolutionary Computation, 14(1):150–169, 2009.

[15] Y. Li, X. Shen, X. Yao, X. Ding, Y. Miao, R. Krishnan, and R. Padman. Beyond single-turn: A survey on multi-turn interactions with large language models. arXiv preprint arXiv:2504.04717, 2025.

[16] H. Liu, M. Zaharia, and P. Abbeel. Ring attention with blockwise transformers for near-infinite context. arXiv preprint arXiv:2310.01889, 2023.

[17] P.-J. Lu, M.-C. Lai, and J.-S. Chang. A survey of high-performance interconnection networks in high-performance computer systems. Electronics, 11(9):1369, 2022.

[18] S. Lu, D. Guo, S. Ren, J. Huang, A. Svyatkovskiy, A. Blanco, C. Clement, D. Drain, D. Jiang, D. Tang, et al. Codexglue: A machine learning benchmark dataset for code understanding and generation. arXiv preprint arXiv:2102.04664, 2021.

[19] R. Neugebauer, G. Antichi, J. F. Zazo, Y. Audzevich, S. López-Buedo, and A. W. Moore. Understanding pcie performance for end host networking. In Proceedings of the 2018 Conference ofthe ACM Special Interest Group on Data Communication, pages 327–341, 2018.

[20] P. O’Neil, E. Cheng, D. Gawlick, and E. O’Neil. The log-structured merge-tree (lsm-tree). Acta informatica, 33(4):351–385, 1996.

[21] D. A. Patterson. Latency lags bandwith. Communications of the ACM, 47(10):71–75, 2004.

[22] K. Punniyamurthy, K. Hamidouche, and B. M. Beckmann. Optimizing distributed ml communication with fused computation-collective operations. In SC24: International Conferencefor High Performance Computing, Networking, Storage and Analysis, pages 1–17. IEEE, 2024.

[23] A. Sun, W. Zhao, X. Han, C. Yang, Z. Liu, C. Shi, and M. Sun. Burstattention: An efficient distributed attention framework for extremely long sequences. arXiv preprint arXiv:2403.09347, 2024.

[24] A. J. Thirunavukarasu, D. S. J. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting. Large language models in medicine. Nature medicine, 29(8):1930–1940, 2023.

[25] A. Vaswani, N. Shazeer, N. Parmar, J. Uszkoreit, L. Jones, A. N. Gomez, Ł. Kaiser, and I. Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[26] Y. Wang, K. Hong, X. Li, Y. Xu, W. Wang, G. Dai, and Y. Wang. Tasp: Topology-aware sequence parallelism. arXiv preprint arXiv:2509.26541, 2025.

[27] G. Xiao, J. Tang, J. Zuo, J. Guo, S. Yang, H. Tang, Y. Fu, and S. Han. Duoattention: Efficient long-context llm inference with retrieval and streaming heads. arXiv preprint arXiv:2410.10819, 2024.

[28] L. Zhao, S. Pal, T. Chugh, W. Wang, J. Fantl, P. Basu, J. Khoury, and A. Krishnamurthy. Efficient {Direct-Connect} topologies for collective communications. In 22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25), pages 705–737, 2025.

[29] W. X. Zhao, K. Zhou, J. Li, T. Tang, X. Wang, Y. Hou, Y. Min, B. Zhang, J. Zhang, Z. Dong, et al. A survey of large language models. arXiv preprint arXiv:2303.18223, 1(2), 2023.

## Appendix

## A Implementation

We implement both the forward and backward of CHARM+. Algorithm 1 presents the logic for constructing the communication groups used in our method. Specifically, the communication peers are partitioned into two disjoint subgroups corresponding to kv communication and q communication at each merge level. To avoid imbalanced communication overhead where multiple GPUs simultaneously initiate communication to the same GPU, we determine the order of kv and q communication based on each GPU’s relative position within the global communication group. For example, when GPU 0 and 1 communicate with GPU 2 and 3, GPU 0 sends kv to GPU 2 and q to GPU 3, whereas GPU 1 sends q to GPU 2 and kv to GPU 3. This complementary scheduling ensures that both the kv and q processing streams on each GPU remain active, effectively preventing stream idling and improving overall communication efficiency.

Algorithm 1 GenerateCommList   
Input: processGroup pg   
Output: commL $\mathrm { i s t } _ { q } ,$ comm $\mathrm { i s t } _ { k v }$   
Initialize tmp ← [], n ← len(pg), x ← getRank $( p g )$   
$l \gets 1$   
while $l < n$ do   
$g \gets \{ x \oplus i \mid i \in [ l , 2 \cdot l ) \}$   
Append g to tmp   
$l \gets 2 \cdot l ^ { ^ { \vee } }$   
end while   
Initialize $g _ { 1 } , g _ { 2 }  [ ] .$ , append tmp[0] to $g _ { 1 }$   
$g _ { 3 } \gets \mathrm { S o r t } ( t { \bar { m } } p [ 0 ] \breve { \cup } \{ { \bar { x } } \bar { \} } )$   
for each g in tmp[1 :] do   
first, second ← split(g), index ← getPosition $( x , g _ { 3 } )$   
half $ \vert g _ { 3 } \vert / 2$   
if index $<$ half then   
Append first to $g _ { 1 } .$ , append second to $g _ { 2 }$   
else   
index ← index − half   
Append second to $g _ { 1 }$ , append first to $g _ { 2 }$   
end if   
$g _ { [ } 3 ] \gets \mathrm { S o r t } ( g _ { 3 } \cup g )$   
end for   
commL $\begin{array} { r } { \dot { \bf \nabla } _ { \bf { i s t } } _ { k v } , } \end{array}$ commLis $\mathfrak { t } _ { q } \gets$ concatenate $( g _ { 1 } , g _ { 2 } )$   
Return commL $\mathrm { i s t } _ { k v } ,$ commL $\mathrm { i s t } _ { q }$

Algorithm 2 provides the core logic chain of the forward algorithm <sup>5</sup>, which is consistent with the overall pipeline in Figure 3(f). Notably, for the q communication, the attention output needs to be communicated back to the original GPU after the attention computation finishes (i.e., the Out communication in Figure 3). Therefore, we prioritize the q communication stream during execution, enabling earlier output communications and effective overlap with the subsequent computation of the kv communication stream.

## B Analysis

We conduct a detailed analysis of the factors affecting CCR, as well as the memory usage of Ring-Attention and CHARM+.

Algorithm 2 CHARM+ Forward   
Input: tensor q, k, v, processGroup pg   
Output: out   
commList<sub>q</sub>, commList<sub>kv</sub> ← GenerateCommList(pg)   
q<sub>cross</sub> ← CrossComm(q)   
Initialize out, out<sub>cross</sub> ← None, n ← getsize(pg)   
Commwait()   
for i ← 1 to log(n) do   
k<sub>other</sub>, v<sub>other</sub> ← KVComm(k, v)   
q<sub>other</sub>, q<sub>otherC</sub> ← QComm(q, q<sub>cross</sub>)   
if i == 1 then   
out ← CompAttention(q, k, v, out)   
out<sub>cross</sub> ← CompAttention(q<sub>cross</sub>, k, v, out)   
else   
out<sub>cross</sub> ← CompAttentionComm(q<sub>otherC</sub>, k, v, out)   
out ← CompAttentionComm(q<sub>other</sub>, k, v, out)   
out<sub>cross</sub> ← CompAttention(q<sub>cross</sub>, k<sub>other</sub>, v<sub>other</sub>, out)   
out ← CompAttention(q, k<sub>other</sub>, v<sub>other</sub>, out)   
end if   
Commwait()   
end for   
out ← CommAndUpdate(out, out<sub>cross</sub>)

## B.1 Analysis of CCR

As mentioned in Eq. (7), CCR is defined by us as:

$$
C C R = \sum _ { i = 1 } ^ { N } T _ { \mathrm { c o m p } ( q , k , v ) } ^ { i } / \sum _ { i = 1 } ^ { N } T _ { \mathrm { c o m m } ( k , v ) } ^ { i }\tag{8}
$$

The communication time depends only on the amount of data being transferred and the available bandwidth. Different on-device and off-device connections have different bandwidths. Therefore, the relation between communication time and factors can be defined as:

$$
T _ { c o m m } \propto \frac { B S H } { B a n d w i d t h }\tag{9}
$$

And the computation time is essentially the time complexity of FlashAttention, which is

$$
T _ { c o m p } \propto \frac { B S ^ { 2 } H + B S H ^ { 2 } } { F L O P S }\tag{10}
$$

Note that, while the theoretical time remains the relation above, the actual runtime is dominated by memory IO in standard attention and both the actual FLOPS and bandwidth exhibit some variability. It can be seen that once the input size and computation pattern are fixed, the time per step is constant. Thus, the total time equals the sum of non-overlappable times along the critical path, which means reducing the number of steps (minimizing the overall critical path) directly reduces the total time. In addition, we provide some quantitative results under different hardware and input specifications, as shown in Section C.2. Due to fluctuations in actual time, we collected the time data in real-time through the interface provided by CUDA events.

## B.2 Analysis of Memory Overhead

Since we do not change the computation pattern but instead improve the overlap between computation and communication, we focus our analysis on the major memory-consuming buffers and ignore the memory overhead of the attention computation.

We assume that the input query, key, and value are all $[ b , s , h ] ( \mathrm { i . e } .$ ., batch size, sequence length and hidden dims). For Ring-Attention, the memory overhead of the buffers is:

$$
M _ { R i n g } \approx 3 b s h + 2 b s h + 1 . 5 b s h + 5 b s h + 3 b s h + 1 . 5 b s h + 2 b s h + 2 b s h \approx 2 0 b s h\tag{11}
$$

Specifically, under FP32 precision, each buffer occupies BSH memory. In Ring-Attention, one buffer stores the local query block, two buffers hold the key and value blocks being processed, and two more buffers receive the next KV blocks to overlap computation and communication. One buffer stores the accumulated output, and half a buffer holds intermediate FlashAttention results (usually in BF16/FP16). During backward, five buffers store q, k, v, the output, and its gradient; three buffers store total gradients of $q , k , v ; 1 . 5$ buffers store FlashAttention gradients; and two buffers handle the next KV block computation. Finally, because KVs propagate across GPUs in a ring, the accumulated gradients must be sent back to the original GPU, requiring two additional buffers.

We further analyze the memory overhead of CHARM+, which can be formulated as:

$$
\begin{array} { r l } & { M _ { \mathrm { { C H A R M S + } } } \approx 3 b s h + 2 b s h + 1 . 5 b s h + 5 b s h + 3 b s h + 1 . 5 b s h + 2 b s h + 6 b s h - 2 b s h } \\ & { \qquad \approx 2 4 b s h } \end{array}\tag{12}
$$

During forward and backward, each GPU must retain its own KV and receive Q of cross-hardware, requiring three extra buffers. Moreover, FlashAttention results can be updated immediately, so the buffers used in ring-attention for gradient sending can be omitted.

Overall, our improved method logically uses four more buffers than ring-attention. However, GPU memory is temporally reused across forward and backward and buffer lifetimes do not fully overlap, the peak buffer usage remains roughly the same, resulting in only a 3%\~5% increase in observed memory.

## C Experimental Details

## C.1 Nsight Result

![](images/ab5ecc1e3b2431425c5b3d6330663e123491afc182e49ef75db06bed38b75239.jpg)  
Figure 9: Nsight Analysis of Ring-Attention and CHARM+

To better understand the performance improvement, we profile the forward pass of Ring-Attention and CHARM+ on 8 L20 GPUs using Nsight Systems, as shown in Figure 9a and Figure 9b, respectively. The L20 platform has relatively limited communication bandwidth, making communication kernels substantially longer than computation kernels. This imbalance makes communication–computation overlap particularly important, as the overall execution time is largely determined by how effectively communication latency can be hidden behind computation.

In Figure 9a, Ring-Attention exhibits substantial GPU idle time due to its sequential communication dependencies. Each communication step depends on the completion of the preceding step, limiting scheduling flexibility and leaving computation unable to fully hide communication latency. In contrast, as shown in Figure 9b, CHARM+ reorganizes ring-based communication into a merge-based structure, shortening the dependency chain, and leverages multiple streams to execute independent communication and computation tasks concurrently. This design substantially reduces idle gaps and improves communication–computation overlap. Notably, both traces are plotted on the same time scale, clearly showing that CHARM+ completes execution in less time than Ring-Attention, demonstrating the effectiveness of CHARM+ in improving overall execution performance.

## C.2 Evaluation on High-end GPU

We validate our method on the high-performance H100 GPUs, with detailed configurations provided in Table 2. We conduct both forward and backward experiments under 2×8 and 4×8 setups (with a hidden dimensions of $1 2 \times 1 9 2 )$ . We additionally conducted experiments on ultra-long sequences, specifically those exceeding 100k tokens. Table 9 reports the forward results for the 2×8 configuration, Table 10 reports the corresponding backward results, Table 11 presents the forward results for the 4×8 configuration, and Table 12 presents the corresponding backward results. We report the CCR as well as the detailed computation time $( T _ { \mathrm { c o m p } } )$ and communication time $( T _ { \mathrm { c o m m } } )$

Our method shows better performance on 32 GPUs than on 16 GPUs. This is because the cross-node communication can be effectively overlapped with the intra-node pipeline. After decoupling via cross-hardware query parallelism, the only dependency is that cross-node query communication must finish before the intra-node pipeline begins. Accordingly, the query required for the next step can be communicated concurrently during the intra-node pipeline.

Based on this analysis and results, our method consistently confines cross-node communication to the start and end, with intermediate cross-node communication effectively overlapped by the intra-node pipeline. In contrast, other approaches suffer from synchronization constraints, making them susceptible to the slower cross-node communication. Consequently, our method exhibits greater performance as the scale increases (speedups of $2 . 2 8 \times / 3 . 6 9 \times \mathrm { o n }$ 16 H100 GPUs and 2.69×/4.21× on 32 H100 GPUs at maximum performance), demonstrating our advantage.

In the scenario of ultra-long sequence lengths (over 200k tokens) on 16 H100 GPUs in Table 9 and Table 10, the CCR exceeds 1, indicating that the computation time is greater than the communication time. At this point, computation becomes the main bottleneck, diminishing the benefits of overlap of communication and computation. Nevertheless, our method still significantly outperforms all other methods.

## C.3 Numerical correctness verification

In Ring-Attention, updates on each GPU occur in different orders, and the only precision loss comes from BF16 → FP32 → BF16 conversion. This type of error exists in all distributed attention methods, and our method does not introduce any additional precision errors.

We verified numerical correctness of our method. The results were obtained on an 8-GPU L20 setup. Specifically, we set the sequence length to 8K and generated 100 groups of randomly initialized Q, K, V , and output gradients, all sampled from a normal distribution. We executed FlashAttention, Ring-Attention, and CHARM+, respectively.

By performing element-wise subtraction, we computed the difference matrices between Ring-Attention and FlashAttention, as well as between CHARM+ and FlashAttention. We further evaluated these differences by calculating the mean and maximum, and finally reported the averages over the 100 runs, as summarized in Table 8.

Table 8: Comparison of numerical differences between CHARM+ and Ring-Attention.
<table><tr><td>Diff</td><td>CHARM+ (Max)</td><td>CHARM+ (Mean)</td><td>Ring (Max)</td><td>Ring (Mean)</td></tr><tr><td>out</td><td>8.34147e-04</td><td>4.51229e-05</td><td>8.07648e-04</td><td>4.24147e-05</td></tr><tr><td>lse</td><td>1.65642e-06</td><td>2.94155e-07</td><td>1.75893e-06</td><td>3.04284e-07</td></tr><tr><td>dq</td><td>1.26290e-03</td><td>3.29077e-05</td><td>1.25977e-03</td><td>3.27659e-05</td></tr><tr><td>dk</td><td>2.87713e-03</td><td>3.53419e-05</td><td>2.89413e-03</td><td>3.53193e-05</td></tr><tr><td>dv</td><td>4.47177e-03</td><td>3.42940e-05</td><td>4.48120e-03</td><td>3.42894e-05</td></tr></table>

These results demonstrate that the numerical differences between CHARM+ and FlashAttention remain at the same order of magnitude as those between Ring-Attention and FlashAttention, indicating that CHARM+ does not introduce additional numerical errors.

Table 9: Forward performance comparison on 16-H100.
<table><tr><td>B</td><td>Seq</td><td> $T _ { c o m m }$ </td><td> $T _ { c o m p }$ </td><td>CCR</td><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 2 \mathbf { 1 5 } \mathbf { , 0 4 0 }$ </td></tr><tr><td>1</td><td>215,040</td><td>81.49</td><td>86.01</td><td>1.06</td><td>116.76</td><td>86.93</td><td>66.74</td><td>1.75×</td></tr><tr><td>2</td><td>215,040</td><td>164.31</td><td>171.51</td><td>1.04</td><td>241.03</td><td>171.91</td><td>125.43</td><td>1.92×</td></tr><tr><td colspan="9">Seq = 107,520</td></tr><tr><td>1</td><td>107,520</td><td>38.34</td><td>23.16</td><td>0.60</td><td>44.08</td><td>26.36</td><td>21.43</td><td>2.06×</td></tr><tr><td>2</td><td>107,520</td><td>74.25</td><td>45.20</td><td>0.61</td><td>89.20</td><td>51.46</td><td>42.28</td><td>2.11×</td></tr><tr><td>4</td><td>107,520</td><td>146.17</td><td>88.94</td><td>0.61</td><td>182.15</td><td>101.74</td><td>81.88</td><td>2.22×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 5 3 , 7 6 0$ </td></tr><tr><td>2</td><td>53,760</td><td>38.27</td><td>14.39</td><td>0.36</td><td>39.18</td><td>25.81</td><td>18.25</td><td>2.15×</td></tr><tr><td>4 8</td><td>53,760</td><td>74.39</td><td>26.50 51.13</td><td>0.36 0.35</td><td>78.95</td><td>50.29</td><td>34.36</td><td>2.30×</td></tr><tr><td></td><td>53,760</td><td>146.24</td><td></td><td></td><td>156.57</td><td>99.39</td><td>66.91</td><td>2.34×</td></tr><tr><td colspan="9">Seq = 26,880</td></tr><tr><td>2</td><td>26,880</td><td>20.13</td><td>5.63</td><td>0.28</td><td>18.35</td><td>13.56</td><td>9.43</td><td>1.95×</td></tr><tr><td>4</td><td>26,880</td><td>38.24</td><td>9.87</td><td>0.26</td><td>37.90</td><td>25.64</td><td>17.03</td><td>2.22×</td></tr><tr><td>8</td><td>26,880</td><td>74.44</td><td>17.22</td><td>0.23</td><td>75.12</td><td>49.75</td><td>32.60</td><td>2.30×</td></tr><tr><td>16</td><td>26,880</td><td>146.53</td><td>32.05</td><td>0.22</td><td>149.97</td><td>98.01</td><td>63.93</td><td>2.35×</td></tr><tr><td colspan="9">Seq = 13,440</td></tr><tr><td>2 4</td><td>13,440</td><td>10.89</td><td>2.92</td><td>0.27</td><td>9.15</td><td>8.18</td><td>5.89</td><td>1.55× 1.97×</td></tr><tr><td>8</td><td>13,440 13,440</td><td>20.31</td><td>4.10 6.87</td><td>0.20 0.18</td><td>17.89 35.72</td><td>13.56 25.48</td><td>9.07 16.57</td><td>2.16×</td></tr><tr><td>16</td><td>13,440</td><td>38.31 74.55</td><td>11.83</td><td>0.16</td><td>72.65</td><td>49.27</td><td>31.84</td><td>2.28×</td></tr><tr><td>32</td><td>13,440</td><td>145.95</td><td>21.79</td><td>0.15</td><td>143.05</td><td>97.30</td><td>62.34</td><td>2.29×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 6 } \mathbf { , } 7 2 \mathbf { 0 }$ </td></tr><tr><td>4</td><td>6,720</td><td>10.92</td><td>2.53</td><td>0.23</td><td>9.12</td><td>8.19</td><td>5.63</td><td>1.62×</td></tr><tr><td>8</td><td>6,720</td><td>20.12</td><td>3.27</td><td>0.16</td><td>17.59</td><td>13.47</td><td>8.91</td><td>1.98× 2.19×</td></tr><tr><td>16</td><td>6,720</td><td>38.26</td><td>4.61</td><td>0.14</td><td>35.76</td><td>25.27</td><td>16.31</td><td></td></tr><tr><td>32</td><td>6,720</td><td>74.53</td><td>7.90</td><td>0.12</td><td>70.84</td><td>49.23</td><td>31.53</td><td>2.25×</td></tr><tr><td>64</td><td>6,720</td><td>146.14</td><td>17.05</td><td>0.11</td><td>141.75</td><td>97.65</td><td>61.80</td><td>2.29×</td></tr><tr><td colspan="9">Seq = 3,360</td></tr><tr><td>8</td><td>3,360</td><td>10.98</td><td>2.55</td><td>0.25</td><td>9.17</td><td>8.10</td><td>5.62</td><td>1.63×</td></tr><tr><td>16 32</td><td>3,360 3,360</td><td>20.17 38.23</td><td>3.05 4.61</td><td>0.15</td><td>17.49</td><td>13.37</td><td>8.82</td><td>1.98× 2.13×</td></tr><tr><td>64</td><td>3,360</td><td>74.53</td><td>7.90</td><td>0.12 0.11</td><td>34.50 70.54</td><td>25.13 49.06</td><td>16.23 31.21</td><td>2.26×</td></tr><tr><td>128</td><td>3,360</td><td>146.20</td><td>14.36</td><td>0.10</td><td>140.55</td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td>97.15</td><td>61.54</td><td>2.28×</td></tr></table>

Table 10: Backward performance comparison on 16-H100.
<table><tr><td>B</td><td>Seq</td><td> $T _ { c o m m }$ </td><td> $T _ { c o m p }$ </td><td>CCR</td><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 2 \mathbf { 1 5 } \mathbf { , 0 4 0 }$ </td></tr><tr><td>1</td><td>215,040</td><td>218.48</td><td>257.76</td><td>1.18</td><td>312.45</td><td>一</td><td>207.07</td><td>1.51×</td></tr><tr><td>2</td><td>215,040</td><td>437.82</td><td>503.06</td><td>1.15</td><td>628.21</td><td>一</td><td>405.90</td><td>1.55×</td></tr><tr><td colspan="9">Seq = 107,520</td></tr><tr><td>1</td><td>107,520</td><td>109.45</td><td>68.03</td><td>0.62</td><td>114.64</td><td></td><td>64.09</td><td>1.79×</td></tr><tr><td>2</td><td>107,520</td><td>218.37</td><td>135.44</td><td>0.62</td><td>228.70</td><td></td><td>125.01</td><td>1.83×</td></tr><tr><td>4</td><td>107,520</td><td>436.10</td><td>264.15</td><td>0.61</td><td>451.53</td><td>一</td><td>242.15</td><td>1.86×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 5 3 , 7 6 0$ </td></tr><tr><td>2</td><td>53,760</td><td>110.04</td><td>37.64</td><td>0.34</td><td>113.30</td><td></td><td>45.91</td><td>2.47×</td></tr><tr><td>4 8</td><td>53,760</td><td>219.95</td><td>74.53</td><td>0.34</td><td>226.22</td><td>一</td><td>88.55</td><td>2.55×</td></tr><tr><td></td><td>53,760</td><td>439.16</td><td>145.33</td><td>0.33</td><td>450.33</td><td>一</td><td>173.18</td><td>2.60×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 2 6 { , } 8 8 0$ </td></tr><tr><td>2</td><td>26,880</td><td>56.73</td><td>11.47</td><td>0.20</td><td>57.03</td><td></td><td>20.42</td><td>2.79×</td></tr><tr><td>4</td><td>26,880</td><td>112.12</td><td>22.35</td><td>0.20</td><td>112.24</td><td></td><td>37.51</td><td>2.99×</td></tr><tr><td>8 16</td><td>26,880</td><td>221.73</td><td>43.86</td><td>0.20</td><td>226.13</td><td>一</td><td>72.59</td><td>3.12×</td></tr><tr><td></td><td>26,880</td><td>442.86</td><td>86.72</td><td>0.20</td><td>449.58</td><td></td><td>141.52</td><td>3.18×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 1 } \mathbf { 3 } \mathbf { , 4 4 0 }$ </td></tr><tr><td>2 4</td><td>13,440</td><td>28.73</td><td>4.50</td><td>0.16</td><td>28.93</td><td></td><td>10.81</td><td>2.67×</td></tr><tr><td>8</td><td>13,440</td><td>56.86</td><td>8.42 15.44</td><td>0.15 0.14</td><td>57.08</td><td></td><td>18.48 33.84</td><td>3.09×</td></tr><tr><td>16</td><td>13,440 13,440</td><td>112.90 222.95</td><td>29.43</td><td>0.13</td><td>113.19 226.46</td><td></td><td>65.35</td><td>3.34× 3.47×</td></tr><tr><td>32</td><td>13,440</td><td>444.64</td><td>57.34</td><td>0.13</td><td>448.54</td><td>一</td><td>128.53</td><td>3.49×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 6 } \mathbf { , } 7 2 \mathbf { 0 }$ </td></tr><tr><td>4 8</td><td>6,720</td><td>28.88</td><td>3.23 5.94</td><td>0.11 0.10</td><td>29.11</td><td></td><td>10.22 17.34</td><td>2.85× 3.27×</td></tr><tr><td>16</td><td>6,720 6,720</td><td>56.98 112.98</td><td>10.90</td><td>0.10</td><td>56.67 112.72</td><td></td><td>32.38</td><td>3.48×</td></tr><tr><td>32</td><td>6,720</td><td>224.56</td><td>20.72</td><td>0.09</td><td>224.23</td><td></td><td>62.46</td><td>3.59×</td></tr><tr><td>64</td><td>6,720</td><td>445.33</td><td>40.32</td><td>0.09</td><td>448.84</td><td></td><td>123.32</td><td>3.64×</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="9"> $\mathbf { S e q } = 3 \mathbf { , } 3 6 \mathbf { 0 }$ </td></tr><tr><td>8</td><td>3,360</td><td>28.72</td><td>2.74</td><td>0.10</td><td>28.11</td><td></td><td>10.04</td><td>2.80×</td></tr><tr><td>16</td><td>3,360</td><td>56.94</td><td>4.98</td><td>0.09</td><td>56.74</td><td></td><td>17.06</td><td>3.33×</td></tr><tr><td>32 64</td><td>3,360 3,360</td><td>112.99 224.63</td><td>9.41 18.13</td><td>0.08 0.08</td><td>112.68 224.41</td><td></td><td>32.14 62.20</td><td>3.51× 3.61×</td></tr><tr><td>128</td><td>3,360</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td>445.33</td><td>35.34</td><td>0.08</td><td>450.01</td><td></td><td>121.99</td><td>3.69×</td></tr></table>

Table 11: Forward performance comparison on 32-H100.
<table><tr><td></td><td>Seq</td><td> $T _ { c o m m }$ </td><td> $T _ { c o m p }$ </td><td>CCR</td><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 1 } 2 6 , \mathbf { 9 7 6 }$ </td></tr><tr><td>2</td><td>126,976</td><td>95.28</td><td>36.30</td><td>0.38</td><td>102.34</td><td>94.17</td><td>39.95</td><td>2.56×</td></tr><tr><td>4</td><td>126,976</td><td>186.17</td><td>70.03</td><td>0.38</td><td>208.12</td><td>184.59</td><td>77.02</td><td>2.70×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 6 3 , 4 8 8$ </td></tr><tr><td>2</td><td>63,488</td><td>49.95</td><td>12.50</td><td>0.25</td><td>46.32</td><td>48.54</td><td>19.80</td><td>2.34×</td></tr><tr><td>4</td><td>63,488</td><td>96.13</td><td>22.28</td><td>0.23</td><td>95.79</td><td>93.60</td><td>36.62</td><td>2.62×</td></tr><tr><td>8</td><td>63,488</td><td>187.15</td><td>41.96</td><td>0.22</td><td>193.04</td><td>184.12</td><td>70.62</td><td>2.73×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 3 1 , 7 4 4$ </td></tr><tr><td>2</td><td>31,744</td><td>26.70</td><td>6.28</td><td>0.24</td><td>22.59</td><td>25.38</td><td>11.47</td><td>1.97×</td></tr><tr><td>4</td><td>31,744</td><td>50.19</td><td>8.99</td><td>0.18</td><td>45.25</td><td>48.38</td><td>18.91</td><td>2.39×</td></tr><tr><td>8</td><td>31,744</td><td>96.07</td><td>15.58</td><td>0.16</td><td>92.39</td><td>93.35</td><td>35.27</td><td>2.62×</td></tr><tr><td>16</td><td>31,744</td><td>187.54</td><td>28.46</td><td>0.15</td><td>185.42</td><td>183.22</td><td>68.17</td><td>2.72×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 1 5 } \mathbf { , 8 7 } 2$ </td></tr><tr><td>2</td><td>15,872</td><td>15.54</td><td>4.98</td><td>0.32</td><td>12.06</td><td>14.09</td><td>9.19</td><td>1.31×</td></tr><tr><td>4</td><td>15,872</td><td>26.73</td><td>5.09</td><td>0.19</td><td>22.29</td><td>25.24</td><td>10.93</td><td>2.04×</td></tr><tr><td>8 16</td><td>15,872</td><td>50.16</td><td>7.13</td><td>0.14</td><td>44.77</td><td>48.23</td><td>18.47</td><td>2.42×</td></tr><tr><td>32</td><td>15,872 15,872</td><td>96.20 186.74</td><td>11.93</td><td>0.12</td><td>89.27</td><td>93.30</td><td>34.64</td><td>2.58×</td></tr><tr><td></td><td></td><td></td><td>21.26</td><td>0.11</td><td>180.66</td><td>183.29</td><td>66.85</td><td>2.70×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 7 \mathbf { , } 9 3 6$ </td></tr><tr><td>4</td><td>7,936</td><td>15.63</td><td>4.96</td><td>0.32</td><td>12.02</td><td>14.09</td><td>8.97</td><td>1.34×</td></tr><tr><td>8 16</td><td>7,936 7,936</td><td>27.24 50.01</td><td>5.01 6.33</td><td>0.18</td><td>22.31</td><td>25.31 48.34</td><td>10.83 18.34</td><td>2.06× 2.40×</td></tr><tr><td>32</td><td></td><td></td><td></td><td>0.13</td><td>43.95</td><td></td><td></td><td>2.60×</td></tr><tr><td>64</td><td>7,936 7,936</td><td>96.10 186.69</td><td>10.17</td><td>0.11</td><td>89.66</td><td>93.28 182.93</td><td>34.44</td><td>2.69×</td></tr><tr><td></td><td></td><td></td><td>17.81</td><td>0.10</td><td>179.13</td><td></td><td>66.48</td><td></td></tr><tr><td colspan="9"> $\mathbf { S e q } = 3 \mathbf { , } 9 6 8$ </td></tr><tr><td>8</td><td>3,968</td><td>15.55</td><td>4.71</td><td>0.30</td><td>11.91</td><td>14.08</td><td>8.69</td><td>1.37×</td></tr><tr><td>16</td><td>3,968</td><td>27.24</td><td>5.12</td><td>0.19</td><td>22.29</td><td>25.23</td><td>10.90</td><td>2.04×</td></tr><tr><td>32</td><td>3,968</td><td>50.60</td><td>6.31</td><td>0.12</td><td>44.73</td><td>48.17</td><td>18.39</td><td>2.43×</td></tr><tr><td>64</td><td>3,968</td><td>96.26</td><td>9.56</td><td>0.10</td><td>89.71</td><td>93.11</td><td>34.44</td><td>2.60×</td></tr><tr><td>128</td><td>3,968</td><td>189.89</td><td>16.43</td><td>0.09</td><td>178.51</td><td>183.09</td><td>66.43</td><td>2.69×</td></tr></table>

Table 12: Backward performance comparison on 32-H100.
<table><tr><td></td><td>Seq</td><td> $T _ { c o m m }$ </td><td> $T _ { c o m p }$ </td><td>CCR</td><td>Ring</td><td>TASP</td><td>CHARM+</td><td>Speedup</td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 1 } 2 6 , \mathbf { 9 7 6 }$ </td></tr><tr><td>2</td><td>126,976</td><td>259.62</td><td>102.57</td><td>0.40</td><td>280.48</td><td>一</td><td>111.60</td><td>2.51×</td></tr><tr><td>4</td><td>126,976</td><td>521.26</td><td>196.56</td><td>0.38</td><td>559.74</td><td>一</td><td>213.85</td><td>2.62×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 6 3 , 4 8 8$ </td></tr><tr><td>2</td><td>63,488</td><td>138.65</td><td>29.95</td><td>0.22</td><td>141.18</td><td>一</td><td>46.30</td><td>3.05×</td></tr><tr><td>4</td><td>63,488</td><td>269.38</td><td>58.46</td><td>0.22</td><td>280.96</td><td>一</td><td>86.70</td><td>3.24×</td></tr><tr><td>8</td><td>63,488</td><td>529.05</td><td>111.96</td><td>0.21</td><td>560.30</td><td>一</td><td>166.44</td><td>3.37×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = 3 1 , 7 4 4$ </td></tr><tr><td>2</td><td>31,744</td><td>70.47</td><td>10.16</td><td>0.14</td><td>70.57</td><td>一</td><td>24.09</td><td>2.93×</td></tr><tr><td>4</td><td>31,744</td><td>140.16</td><td>19.39</td><td>0.14</td><td>141.71</td><td>一</td><td>40.55</td><td>3.49×</td></tr><tr><td>8</td><td>31,744</td><td>275.78</td><td>37.57</td><td>0.14</td><td>280.92</td><td>一</td><td>75.16</td><td>3.74×</td></tr><tr><td>16</td><td>31,744</td><td>536.42</td><td>73.45</td><td>0.14</td><td>559.34</td><td>一</td><td>146.01</td><td>3.83×</td></tr><tr><td colspan="9"> $\mathbf { S e q } = \mathbf { 1 5 } \mathbf { , 8 7 } 2$ </td></tr><tr><td>2</td><td>15,872</td><td>35.78</td><td>4.67</td><td>0.13</td><td>35.77</td><td>一</td><td>14.92</td><td>2.40×</td></tr><tr><td>4</td><td>15,872</td><td>70.70</td><td>7.28</td><td>0.10</td><td>71.35</td><td>一</td><td>22.11</td><td>3.23×</td></tr><tr><td>8</td><td>15,872</td><td>140.84</td><td>13.69</td><td>0.10</td><td>141.53</td><td>1</td><td>37.75</td><td>3.75×</td></tr><tr><td>16 32</td><td>15,872</td><td>277.48</td><td>26.20</td><td>0.09</td><td>279.85</td><td>一</td><td>70.62</td><td>3.96×</td></tr><tr><td></td><td>15,872</td><td>548.62</td><td>50.62</td><td>0.09</td><td>558.53</td><td>一</td><td>137.84</td><td>4.05×</td></tr><tr><td colspan="9">Seq = 7,936</td></tr><tr><td>4</td><td>7,936</td><td>35.82</td><td>3.71</td><td>0.10</td><td>35.18</td><td></td><td>14.51</td><td>2.42×</td></tr><tr><td>8 16</td><td>7,936 7,936</td><td>70.72</td><td>5.82 10.90</td><td>0.08</td><td>71.06</td><td>一</td><td>21.27</td><td>3.34× 3.86×</td></tr><tr><td>32</td><td></td><td>140.74</td><td>20.68</td><td>0.08</td><td>141.24</td><td>一</td><td>36.61</td><td>4.03×</td></tr><tr><td>64</td><td>7,936 7,936</td><td>278.53 550.51</td><td></td><td>0.07</td><td>278.38</td><td>一</td><td>69.02</td><td>4.14×</td></tr><tr><td></td><td></td><td></td><td>39.76</td><td>0.07</td><td>556.25</td><td>一</td><td>134.48</td><td></td></tr><tr><td colspan="9"> $\mathbf { S e q } = 3 \mathbf { , } 9 6 8$ </td></tr><tr><td>8</td><td>3,968</td><td>35.79</td><td>3.24</td><td>0.09</td><td>35.86</td><td>一</td><td>13.97</td><td>2.57×</td></tr><tr><td>16</td><td>3,968</td><td>70.73</td><td>5.16</td><td>0.07</td><td>71.33</td><td>一</td><td>20.95</td><td>3.41×</td></tr><tr><td>32</td><td>3,968</td><td>140.73</td><td>9.70</td><td>0.07</td><td>141.84</td><td>一</td><td>36.35</td><td>3.90×</td></tr><tr><td>64</td><td>3,968</td><td>278.31</td><td>18.40</td><td>0.07</td><td>279.62</td><td>一</td><td>68.40</td><td>4.09×</td></tr><tr><td>128</td><td>3,968</td><td>551.62</td><td>35.58</td><td>0.06</td><td>560.63</td><td>一</td><td>133.05</td><td>4.21×</td></tr></table>