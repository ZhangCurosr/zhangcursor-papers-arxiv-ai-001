# Asymptotic Optimality of ��-Rule for LLM Inference Scheduling under Memory Constraint

Zhenghao Li HKUST

Shengbo Wang University of Southern California

Nian Si HKUST

## Abstract

Large language model (LLM) inference, the process by which a pre-trained model receives an input prompt and generates output tokens sequentially, has become increasingly important with the widespread deployment of LLM-based applications. As inference workloads continue to scale, GPU memory has emerged as a critical bottleneck: the long input prompts and generation sequences common in agentic and reasoning workloads lead to substantial key–value (KV) cache usage. In this paper, we study the resulting online scheduling problem from a queueing perspective. Requests arrive stochastically with heterogeneous prompt and output lengths, undergo prefill and autoregressive decode, and compete for a finite KV-cache capacity. Unlike classical queueing models, the duration of a processing step depends on both prefill computation and KV-cache I/O, so service capacity is jointly determined by computation, memory access, and memory occupancy. We develop a fluid model that reduces the system to a multiclass single-server queue with efective service rates capturing computation, memory access, and memory capacity. This fluid analysis yields a �� priority rule, where the efective service rate � is jointly determined by computational speed, memory bandwidth, and memory capacity. We prove that this rule minimizes cumulative holding cost in the fluid model and, for a finite workload, also minimizes completion time and maximizes full-run output-token throughput. We then construct a feasible scheduling policy for the original finite system and prove that its scaled dynamics converge to the optimal fluid policy. Finally, experiments on a real LLM serving engine using an NVIDIA H800 show that practical implementations of the proposed priority rule substantially reduce average latency while maintaining high throughput relative to existing scheduling policies.

## 1 Introduction

The public debut of ChatGPT in late 2022 marked a turning point in the swift adoption of generative artificial intelligence, driven largely by advancements in large language models (LLMs) [7, 31, 34]. These systems have rapidly integrated into routine workflows across research, education, industry, and entertainment [17, 21, 30], facilitating billions of user interactions to date [16, 18]. However, alongside significant gains in human productivity, this explosive growth has introduced critical bottlenecks in serving large-scale LLM inference reliably and eficiently.

LLM inference refers to the process by which a pre-trained LLM receives an input prompt and generates output tokens sequentially. Compared with traditional machine learning models, LLM inference is memory and computation intensive, often involves long generation sequences, and can incur substantial latency and resource costs. The importance of eficient inference is further amplified by the growing adoption of reasoning models and test-time compute, which increase the amount of computation required for each request. Deloitte<sup>1</sup> estimates that inference will account for roughly two-thirds of total AI compute in 2026, up from about one-third in 2023. Consequently, eficient LLM inference is not merely a problem of computational optimization, but a system-level challenge that requires carefully balancing responsiveness, throughput, and resource costs.

Key-value (KV) caching [25, 32] has become a standard technique for accelerating LLM inference, largely due to the extensive use of attention mechanisms [39]. Roughly speaking, the KV cache stores the previously computed key and value representations of past tokens in GPU memory to avoid their repeated computation when generating each new token. Thus, KV caching trades increased memory usage for reduced computation and achieves substantially accelerated inference. However, this acceleration comes at the cost of substantial GPU memory consumption and intensive memory access, making memory capacity and bandwidth increasingly important. This becomes a severe issue for inference requests with long input contexts or long generated sequences, which are increasingly common in agentic and reasoning workloads, making memory a critical bottleneck in modern LLM inference [2].

In this paper, we study LLM inference with KV-cache memory constraints through an operations lens. We abstract and model the inference scheduling task for LLMs as a multiclass queue. In this queueing system model, inference requests arrive over time with heterogeneous workloads, characterized by their prompt and output lengths, and a GPU server processes them according to prescribed dynamic scheduling decisions. The workload each request brings to the system consists of two sequential phases that must be executed in order: a prefill phase followed by an autoregressive decode phase. The scheduler dynamically determines which requests are admitted and processed at each GPU time step. Our primary objective is to minimize the request latency, or, equivalently, the cumulative number of unfinished requests in the system.

A crucial feature of LLM serving systems, and hence our queueing model, is the KV-cache constraint on GPU memory capacity and bandwidth. As a request progresses through the prefill and decode stages, its KV cache in GPU memory continues to grow until the request completes, while each processing step must also access the KV states accumulated up to that point. These memory requirements can significantly afect the number of requests that can be processed concurrently and the time required for each processing step. In particular, GPU-step processing time depends on both the amount of prefill computation and the KV-cache I/O volume, while the set of concurrently processed requests must satisfy a hard memory-capacity constraint.

To develop an eficient dynamic scheduling policy for this LLM inference queue, we consider a fluid asymptotic regime in which the token processing rate and request arrival rate scale proportionally, while the total memory capacity remains fixed. This choice of regime directly captures the key memory bottleneck in LLM inference systems with KV caching, which shifts the primary computational burden toward memory constraints and I/O costs.

The fluid approximation yields a tractable yet expressive limiting system with a familiar scheduling structure. We show that, under the prescribed limiting regime, the memory-constrained LLM inference system can be related to a classical multiclass queueing model with homogeneous servers. However, the efective service rates in the fluid system must account for prefill computation, KV-cache I/O, and the efect of memory capacity constraints on service availability. This motivates us to adapt the �� priority rule [8, 28, 38], which minimizes cumulative holding cost while maximizing full-run output-token throughput in the fluid model. We establish that the ��-rule remains optimal for this fluid system under the two-phase workload structure and the memory constraint in the LLM serving system. Furthermore, to bridge the fluid control problem and the prelimit LLM inference system, we construct a dynamic scheduling policy inspired by the ��-rule. We show that the proposed policy converges to the ��-rule in the fluid limit and that the prelimit systems under this policy converge to the proposed fluid system controlled by the ��-rule.

Concretely, the paper makes the following main contributions.

• LLM system modeling. We develop a queueing model of LLM inference that explicitly accounts for KV-cache memory capacity constraints and the associated memory I/O costs, with processing times calibrated using real GPU measurements to capture their dependence on the composition of the scheduled workload.

• Identifying the asymptotic regime. We identify a fluid regime in which processing and arrival rates increase together while memory capacity remains fixed, preserving the system’s memory bottleneck. The resulting model reveals a classical single-server scheduling structure with LLM-specific behaviors such as two-phase workload and memory constraints.

• Asymptotic optimality of the �<sub>�</sub>-rule. We prove that the �<sub>�</sub>-rule minimizes cumulative holding cost and maximizes full-run throughput for finite workloads in the fluid limit. To bridge the fluid and finite-system models, we construct an implementable ��-inspired policy and show that the resulting dynamics converge to the optimal fluid limit.

• Numerical and real-system validation. We evaluate practical implementations of the proposed �<sub>�</sub>-rule using semi-synthetic workloads and real workload data on a real LLM serving engine. In both settings, our policy demonstrates substantial reductions in average latency relative to the default scheduler while maintaining high throughput.

Together, these results bridge classical queueing theory and priority-based scheduling policies with implementable, efective scheduling rules for modern LLM inference systems.

## 1.1 Related Work

Recent advances in LLM serving systems have enabled finer-grained control over batching and GPU memory management. Yu et al. [47] developed iteration-level scheduling, allowing the batch to be reconstructed after each model iteration. Kwon et al. [25] introduced PagedAttention, which stores KV caches in blocks to reduce GPU memory waste and enable multiple requests to share cached data. Agrawal et al. [1] split long prefill computations into smaller chunks and co-batched them with decode work. Together, these advances provide the infrastructure for fine-grained control over batching and GPU memory management, serving as both the foundation for our model and the practical testbed for our experiments.

To analyze LLM serving systems analytically, several pioneering studies have employed queueing models. Dai et al. [11] characterize stable throughput under a per-batch token-budget constraint and establish the throughput optimality of work-conserving policies, while Nie et al. [29] develop a fine-grained queueing model that tracks KV-cache growth and establish stability and overload conditions for continuous batching under the restrictive assumption of a constant batch processing time. Furthermore, Lin et al. [26] study prefill–decode contention in multi-GPU settings and develop a scheduling policy that achieves asymptotically optimal revenue as the number of GPUs grows, without explicitly modeling GPU memory-capacity constraints. Recently, Yao et al. [46] study KV-cache reuse in multi-turn LLM conversations under a least-recently-used (LRU) eviction policy and establish a mean-field limit for the cache hit ratio as the conversation arrival rate and memory capacity scale proportionally.

The most relevant literature is Ao et al. [4] and Ao et al. [3]. For requests with known output lengths, Ao et al. [4] propose WAIT to maximize throughput under a memory constraint. The policy waits for enough requests of the same class to accumulate, then processes them together. However, setting these thresholds requires arrival-rate information, and their extension to time-varying arrivals imposes additional load and class-mix conditions. Moreover, obtaining valid parameters for WAIT requires certain load and memory conditions to be satisfied, which may not hold in practical settings. We find that, although theoretically optimal, WAIT’s performance can deteriorate in real-GPU implementations due to unnecessary idling. This work models batch processing time only through KV-cache I/O, without considering the extra computation time required to process tokens during the prefill phase. Ao et al. [3] further show that admitting requests whenever memory is available can cause recurring evictions and throughput loss, and mitigate this via rate-limited admission and request mixing, especially in multi-GPU deployments.

Besides queueing-based modeling, other studies approach this problem from an online linear programming perspective: Jaillet et al. [23] develop a hindsight benchmark and the MC-SF algorithm, while Kong et al. [24] propose the Smallest Volume First policy based on cumulative KV-cache usage. For ofline scheduling, prior work develops algorithms and evaluates their total completion time against a clairvoyant benchmark. Specifically, Wang et al. [41] study the setting with known output lengths, Chen et al. [9] consider interval predictions of output lengths, and Feng et al. [12, 13] develop algorithms that do not rely on output-length predictions.

LLM inference under KV-cache memory constraints is closely related to stochastic bin packing, where randomly arriving requests must be served subject to resource capacity constraints. Early studies establish stability conditions for First-Fit and Best-Fit scheduling [10, 14]. In infinite-server models, greedy policies with safety stocks [36], greedy randomized placement [37], and randomized Best-Fit [15] asymptotically minimize the number of occupied servers. Stolyar [35] establishes asymptotic optimality of greedy allocation for a convex approximation to the server-count objective. For a fixed number of servers, Maguluri et al. [27] show that Best-Fit need not be throughput optimal and develop MaxWeight-based scheduling policies with throughput guarantees. Gupta and Radovanovic [20] develop Lagrangian-based packing algorithms for permanent and departing requests, including a model with configuration-dependent service slowdowns.

Despite similar constraints and objectives, LLM inference scheduling difers from classical stochastic bin-packing problems in several ways. First, while stochastic bin-packing models assume that each request has a fixed size upon arrival, the KV-cache usage of an LLM request grows as generation progresses. Second, LLM inference involves two distinct phases, prefill and decode, with diferent memory dynamics. Third, whereas service times in classical stochastic bin-packing models are typically independent of current memory occupancy, processing times in LLM inference depend on memory usage due to KV-cache I/O costs.

The objective of minimizing cumulative holding cost connects our problem to the literature on �� priority scheduling. Buyukkoc et al. [8] establish the optimality of the $c \mu$ rule for a preemptive single-server queue with linear holding costs, geometric service times, and a general arrival process. Van Mieghem [38] extends this framework to convex delay costs and establishes the asymptotic optimality of a generalized �� rule in heavy trafic. Mandelbaum and Stolyar [28] further extend the analysis to heterogeneous servers and prove heavy-trafic asymptotic optimality under complete resource pooling. Zychlinski et al. [48] study queues in which jobs may require multiple servers simultaneously and establish the asymptotic optimality of an idle-avoid $c \mu / m$ policy in a many-server regime. Their model is related to ours through resource constraints, but difers in three key aspects: in LLM inference, a job’s resource usage (KV cache) grows during service, service rates depend on the types and numbers of jobs processed together, and memory capacity remains $M = O ( 1 )$ , so each job’s resource requirement is non-negligible relative to total capacity.

The remainder of the paper is organized as follows. In Section 2, we present our queueing model for LLM inference. In Section 4, we establish the optimality of the $c \mu$ rule for the fluid model. In Section 5, we construct a feasible policy for the discrete system and prove that its scaled dynamics converge to those under the optimal fluid policy. Section 6 presents real GPU experiments that validate our theoretical results. Finally, we conclude the paper and discuss limitations and directions for future work in Section 7.

## 2 LLM Inference Systems

We consider an online LLM inference system in which user requests arrive stochastically over time and wait in a queue for service on a single GPU. Each arriving request contains an input prompt and asks the model to generate an output sequence. A scheduler determines which requests are processed by the GPU over time.

The processing of each request consists of a prefill phase followed by an autoregressive decode phase. During the prefill phase, the model processes the prompt tokens and constructs their attention key–value (KV) states. With chunked prefill, the input prompt is divided into several smaller chunks that are processed sequentially and the first output token is generated when the final chunk is processed. During the decode phase, the model generates the remaining output tokens one at a time. Each decode step takes the latest output token as input, appends its KV state, and generates the next output token. The request’s KV cache occupies GPU memory throughout execution and is released upon completion. Figure 1 summarizes this lifecycle.

![](images/265f8e00557f02156a11b652b11e0d85d0818e255c99bd8378a1904d6e695f2c.jpg)  
Figure 1: Lifecycle of a request with prompt length $s _ { j } = 4 ,$ output length $o _ { j } = 3 ,$ , and two-token prefill chunks. The first prefill step processes $P _ { 1 } , P _ { 2 }$ and writes their keys and values to the KV cache without emitting an output token. The final prefill step processes $P _ { 3 } , P _ { 4 }$ , writes their keys and values to the cache, and emits $O _ { 1 }$ . Each decode step takes the latest output token as input, appends its keys and values to the cache, and emits the next token. After the request completes, its KV cache is released.

Modern inference engines use a GPU step as the smallest unit of scheduling. At each GPU step, a batch is formed from requests that may be at diferent stages of execution. After the step completes, the scheduler updates request states and may re-form the next batch from eligible waiting and running requests; requests may therefore enter or leave the scheduled batch between GPU steps. This mechanism is commonly called continuous batching [47]. Modern engines may also use mixed prefill–decode batching, in which a GPU step processes prefill work from some requests and decode work from others [1].

Every processed token contributes KV states, so a request’s KV-cache occupancy grows during prefill and then by one token at each decode step. The running requests share a finite GPU KV-cache capacity, which we measure in tokens. Figure 2 illustrates how the scheduled batch changes at GPU-step boundaries.

Our queueing model represents request progress through its prefill and decode stages. Requests are partitioned into � classes. A class-� request has known prompt length $s _ { j }$ and output length $o _ { j }$ . The GPU has KV-cache capacity � tokens. Each prompt is partitioned into chunks of size $c _ { \mathrm { c h u n k } }$ . Accordingly, we define the number of prefill chunks of a class-� request by $\begin{array} { r } { L _ { j } = \left\lceil \frac { s _ { j } } { c _ { \mathrm { c h u n k } } } \right\rceil } \end{array}$ . Therefore, a class-� request has $L _ { j }$ sequential prefill stages followed by $o _ { j } - 1$ sequential decode stages, with the final prefill stage producing the first output token. If $o _ { j } = 1$ , the request leaves the system immediately after its final prefill stage.

Let $m ( t )$ denote the maximum KV-cache usage during step �. It consists of the KV states already occupying GPU memory and those newly written during the step. The memory constraint is

$$
m ( t ) \leq M , \qquad t = 0 , 1 , \ldots .
$$

For the batch executed in step �, let $n _ { \mathrm { p f } } ( t )$ be the total number of prompt tokens in the prefill stage to be processed in this batch, and let $n _ { \mathrm { k v } } ( t )$ be the batch’s total KV-cache I/O volume in token units. In our model, this volume counts the existing KV states accessed by the batch and the new KV states written during the step.

By empirically testing diferent mixed batches on an NVIDIA H800, we observe that the step time (batch processing time) varies approximately linearly with both KV-cache I/O volume and the number of prompt tokens processed. Figure 3 shows these linear relationships.

![](images/af79fea3b390401ffa7d74a7f576444ca54a559b4fe77f329a525a0d4c4b97fd.jpg)  
Figure 2: Continuous batching across GPU-step boundaries. During a step, the batch remains fixed and the scheduled requests advance by one step. At the next boundary, completed requests leave, and the scheduler may admit waiting requests or leave resident requests unscheduled.

![](images/b9eac63e6379ce88c772cfa37b375776ed9d2f87ddda7f0b28812154e213f39e.jpg)

![](images/33c66258c9266f143ab67a0ec4740eb94194b23dff6b4c6d6d609130fb255b2c.jpg)  
Figure 3: Conditional fits of the GPU step-time model, estimated on the training sample. Left: KV-cache I/O efect at fixed prefill levels. Right: prefill efect after adjusting held-out samples to fixed KV-cache I/O.

Motivated by these observations, we fit the step time of step � as

$$
\widehat { \Delta } ( t ) : = \widehat { a } _ { \mathrm { p f } } n _ { \mathrm { p f } } ( t ) + \widehat { a } _ { \mathrm { k v } } n _ { \mathrm { k v } } ( t ) + \widehat { b } ,\tag{1}
$$

where $\widehat { a } _ { \mathrm { p f } }$ is the fitted per-prompt-token processing-time coeficient, $\widehat { a } _ { \mathrm { k v } }$ is the fitted per-token KV-cache I/O coeficient, and $\widehat { b }$ is the estimated fixed per-step overhead. Table 1 reports the coeficient estimates and fit statistics.

## 3 System Modeling and Fluid Regime

In this section, we formulate a sequence of prelimit queueing models for the LLM inference system and derive the corresponding fluid control problem. We first specify the system state, queueing dynamics, control decisions, and memory constraints. We then apply fluid scaling and formulate the corresponding fluid control problem. Finally, we show that the fluid control problem also exhibits a state-space collapse phenomenon commonly observed in the queueing literature [6, 44].

Table 1: Batch processing time estimates in an NVIDIA H800. Out-of-sample statistics use held-out existing-KV strata.
<table><tr><td>Parameter</td><td>Estimate</td></tr><tr><td> $\widehat { a } _ { \mathrm { p f } }$  (s/token)</td><td> $2 . 2 6 4 \times 1 0 ^ { - 5 }$ </td></tr><tr><td> $\widehat { a } _ { \mathrm { k v } }$  (s/token)</td><td> $4 . 1 0 8 \times 1 0 ^ { - 8 }$ </td></tr><tr><td> $\widehat { b } \left( \mathrm { s } \right)$ </td><td> $7 . 7 9 4 \times 1 0 ^ { - 3 }$ </td></tr></table>

<table><tr><td>Metric</td><td>In sample</td><td>Out of sample</td></tr><tr><td>Observations</td><td>16,205</td><td>3,987</td></tr><tr><td> $R ^ { 2 }$ </td><td>0.999902</td><td>0.999905</td></tr><tr><td>RMSE (ms)</td><td>2.294</td><td>2.268</td></tr></table>

## 3.1 Queueing Model

Consider a sequence of queueing systems indexed by $r = 1 , 2 , \ldots$ , while the fluid model is obtained as $r \to \infty$ . The arrival rate of requests in each class and the GPU processing speed both scale with �, while the KV-cache capacity � remains fixed. This regime represents the memory bottleneck created by the KV caching techniques that trade memory capacity and I/O cost for computation speed in modern LLM inference systems. Concretely, for the step-time model in (1), we assume that the positive coeficients of the �th system satisfy

$$
r a _ { \mathrm { p f } } ^ { r } \to a _ { \mathrm { p f } } > 0 , \quad r a _ { \mathrm { k v } } ^ { r } \to a _ { \mathrm { k v } } > 0 , \quad r b ^ { r } \to b > 0 .\tag{2}
$$

Thus, the one-step processing time of fixed batch is of order $1 / r$ . Accordingly, define the step time

$$
\Delta ^ { r } ( t ) : = a _ { \mathrm { p f } } ^ { r } n _ { \mathrm { p f } } ^ { r } ( t ) + a _ { \mathrm { k v } } ^ { r } n _ { \mathrm { k v } } ^ { r } ( t ) + b ^ { r } ,
$$

where $n _ { \mathrm { p f } } ^ { r } ( t )$ and $n _ { \mathrm { k v } } ^ { r } ( t )$ are the numbers of prefill and decode tokens processed in step �, respectively, as formally defined in (3).

Throughout the paper, we index GPU processing steps by the integer $t = 0 , 1 , 2 , \ldots$ . and denote clock time by $\tau \geq 0 .$ , since processing steps may have diferent durations. Thus, the clock time at the beginning of step � is given by

$$
f ^ { r } ( t ) : = \sum _ { u = 0 } ^ { t - 1 } \Delta ^ { r } ( u ) .
$$

At the beginning of GPU step �, let $p _ { j , k } ^ { r } ( t )$ be the number of class-� requests with � prefill chunks waiting to be processed and let $d _ { j , k } ^ { r } ( t )$ be the number of class-� requests with � decode tokens remaining. The system state records the number of requests at each prefill and decode stage:

$$
z _ { t } ^ { r } : = \big ( \{ p _ { j , k } ^ { r } ( t ) \} _ { 1 \leq j \leq J , 1 \leq k \leq L _ { j } } , \{ d _ { j , k } ^ { r } ( t ) \} _ { 1 \leq j \leq J , 1 \leq k \leq o _ { j } - 1 } \big ) ,
$$

where $p _ { j , L _ { j } } ^ { r } ( 0 )$ represents the class-� requests already present in the system at time zero.

At each GPU step �, a policy $\pi ^ { r }$ determines how many requests to process at each prefill and decode stage. Specifically, the policy schedules one prefill chunk for $\Delta \mathcal { U } _ { j , k } ^ { r } ( t )$ class-� requests with � prefill chunks remaining, and one decode step for $\Delta \mathcal { V } _ { j , k } ^ { r } ( t )$ class-� requests with � output tokens remaining. Since in practice, each request usually consumes only a small fraction of the total memory capacity �, we allow fractional requests and treat queue lengths and per-step controls as nonnegative real-valued quantities. The number of requests processed at each step cannot exceed the corresponding queue length:

$$
0 \leq \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) \leq p _ { j , k } ^ { r } ( t ) \quad \mathrm { a n d } \quad 0 \leq \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) \leq d _ { j , k } ^ { r } ( t ) .
$$

To facilitate the fluid scaling and the analysis of the limiting dynamics in Section 3.2, we introduce the following cumulative stage-completion processes:

$$
\mathcal { U } _ { j , k } ^ { r } ( t ) : = \sum _ { s = 0 } ^ { t - 1 } \Delta \mathcal { U } _ { j , k } ^ { r } ( s ) \quad \mathrm { a n d } \quad \mathcal { V } _ { j , k } ^ { r } ( t ) : = \sum _ { s = 0 } ^ { t - 1 } \Delta \mathcal { V } _ { j , k } ^ { r } ( s ) ,
$$

with $\mathcal { U } _ { j , k } ^ { r } ( 0 ) = 0$ and $\mathcal { V } _ { j , k } ^ { r } ( 0 ) = 0$ . Since the increments are recovered by taking one-step diferences, the cumulative processes provide an equivalent representation of the controls. We therefore write $\pi ^ { r } = ( \mathcal { U } ^ { r } , \mathcal { V } ^ { r } )$ .

The selected batch must also satisfy the KV-cache capacity constraint. The maximum KV-cache usage during step � is

$$
\begin{array} { l } { m ^ { r } ( t ) = \displaystyle \sum _ { j = 1 } ^ { J } \Biggl ( \sum _ { k = 1 } ^ { L _ { j } } c _ { \mathrm { c h u n k } } ( L _ { j } - k ) p _ { j , k } ^ { r } ( t ) + \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k - 1 ) d _ { j , k } ^ { r } ( t ) } \\ { \displaystyle \qquad + \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) + \sum _ { k = 1 } ^ { o _ { j } - 1 } \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) \Biggr ) . } \end{array}
$$

The first two sums count the KV states already in the memory before step � begins, while the last two count the new KV states for which the selected batch reserves slots. Thus, the memory usage should satisfy the constraint $m ^ { r } ( t ) \leq M$ at all time steps.

For the selected batch, let $n _ { \mathrm { p f } } ^ { r } ( t )$ be the number of prefill tokens processed in step �, and let $n _ { \mathrm { k v } } ^ { r } ( t )$ be its KV-cache I/O volume in token units:

$$
\begin{array} { r l } & { n _ { \mathrm { p f } } ^ { r } ( t ) = \displaystyle \sum _ { j = 1 } ^ { J } \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) , } \\ & { n _ { \mathrm { k v } } ^ { r } ( t ) = \displaystyle \sum _ { j = 1 } ^ { J } \left( \sum _ { k = 1 } ^ { L _ { j } } \lbrack c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } \rbrack \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) + \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) \right) . } \end{array}\tag{3}
$$

Thus, each selected prefill stage contributes its existing KV states and the states written for its current chunk, while each selected decode stage contributes its existing KV states and one new write.

The arrival processes are modeled as nonhomogeneous Poisson processes, with arrival rates scaled by a factor of � to match the processing-time scaling. Specifically, let $\lambda _ { i } ^ { r } ( \tau ) = r \lambda _ { j } ( \tau )$ denote the class-� arrival rate at clock time $\tau$ of the �th system. For each $r \geq 1$ , let $N _ { i } ^ { r } , j = 1 , \cdot \cdot \cdot , J ,$ , be independent unit-rate Poisson processes. The number of class-� requests arriving during step � is represented as

$$
\eta _ { j } ^ { r } ( t ) : = N _ { j } ^ { r } \left( r \int _ { 0 } ^ { f ^ { r } ( t + 1 ) } \lambda _ { j } ( s ) d s \right) - N _ { j } ^ { r } \left( r \int _ { 0 } ^ { f ^ { r } ( t ) } \lambda _ { j } ( s ) d s \right)\tag{4}
$$

Therefore, including the class-� requests already present in the system at time zero, the cumulative class-� arrival process is

$$
A _ { j } ^ { r } ( t ) : = p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) = p _ { j , L _ { j } } ^ { r } ( 0 ) + N _ { j } ^ { r } \left( r \int _ { 0 } ^ { f ^ { r } ( t ) } \lambda _ { j } ( s ) d s \right) , \qquad t \in \mathbb { Z } _ { + } .\tag{5}
$$

These Poisson arrivals are the only sources of randomness in the prelimit systems. Let $\mathcal { F } _ { \tau } ^ { r }$ be the information contained in the arrival process before clock time �; i.e.

$$
\mathcal { F } _ { \tau } ^ { r } = \sigma \left( N _ { j } ^ { r } \left( r \int _ { 0 } ^ { u } \lambda _ { j } ( s ) d s \right) : j = 1 , \cdots , J ; u \leq \tau \right)
$$

Under the admissibility condition in Definition 1, we will establish that $\{ f ^ { r } ( t ) : t \geq 0 \}$ is a sequence of $\mathcal { F } _ { \tau } ^ { r }$ stopping times. Thus, we can define the history at each decision time as $\mathcal { G } _ { t } ^ { r } : = \mathcal { F } _ { f ^ { r } ( t ) } ^ { r } .$

Requests arriving during step � join the waiting queue at step $t + 1$ and can be scheduled from that step onward. The queueing dynamics are

$$
\begin{array} { r l } & { \quad p _ { j , L _ { j } } ^ { r } ( t + 1 ) = p _ { j , L _ { j } } ^ { r } ( t ) + \eta _ { j } ^ { r } ( t ) - \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) , } \\ & { \quad p _ { j , k } ^ { r } ( t + 1 ) = p _ { j , k } ^ { r } ( t ) + \Delta \mathcal { U } _ { j , k + 1 } ^ { r } ( t ) - \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) , \qquad k = 1 , \ldots , L _ { j } - 1 ; } \\ & { { \ d d } _ { j , o _ { j } - 1 } ^ { r } ( t + 1 ) = { d } _ { j , o _ { j } - 1 } ^ { r } ( t ) + \Delta \mathcal { U } _ { j , 1 } ^ { r } ( t ) - \Delta \mathcal { V } _ { j , o _ { j } - 1 } ^ { r } ( t ) , } \\ & { \quad { d } _ { j , k } ^ { r } ( t + 1 ) = { d } _ { j , k } ^ { r } ( t ) + \Delta \mathcal { V } _ { j , k + 1 } ^ { r } ( t ) - \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) , \qquad k = 1 , \ldots , o _ { j } - 2 , } \end{array}\tag{6}
$$

when $o _ { j } \ge 2$ . If $o _ { j } = 1$ , the request completes immediately after its final prefill stage. The first equation states that the class-� waiting-queue length at step $t + 1$ equals its value at step �, plus arrivals during step �, minus the admission increment $\Delta \mathcal { U } _ { j , L _ { i } } ^ { r } ( t )$ . The remaining three equations have the same inflow-minus-outflow form: completions from the preceding stage enter the state, and completions of the current stage leave it.

We now specify the admissibility requirements for prelimit policies.

Definition 1 (Admissible prelimit policy). Fix an admissible initial state $z _ { 0 } ^ { r } .$ A prelimit policy $\pi ^ { r } = ( \mathcal { U } ^ { r } , \mathcal { V } ^ { r } )$ is admissible if the following conditions hold at every GPU step �.

1. $\pi ^ { r }$ is non-anticipative; i.e. $\Delta \mathcal { U } _ { j , k } ^ { r } ( t )$ and $\Delta \mathcal { V } _ { j , k } ^ { r } ( t )$ are ${ \mathcal { G } } _ { t } ^ { r }$ -measureable for all indices $( j , k )$ ranging over prefill and decode stages.

2. The KV-cache capacity constraint holds: $m ^ { r } ( t ) \leq M$

3. For every class $j = 1 , \dots , J ,$ , the control increments satisfy

$$
0 \leq \Delta \mathcal { U } _ { j , k } ^ { r } ( t ) \leq p _ { j , k } ^ { r } ( t ) \quad \mathrm { a n d } \quad 0 \leq \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) \leq d _ { j , k } ^ { r } ( t ) .
$$

In Appendix A, we note that the non-anticipativity condition and ${ \mathcal { G } } _ { t } ^ { r }$ are well-defined by showing that $\{ f ^ { r } ( t ) : t \geq 0 \}$ is a sequence of $\mathcal { F } _ { \tau } ^ { r }$ stopping times.

Let $c _ { j } \geq 0$ be the holding-cost rate for each class-� request. The normalized expected cumulative holding cost under a prelimit scheduling policy $\pi ^ { r }$ is

$$
\mathcal { T } ^ { r } ( \pi ^ { r } ; z _ { 0 } ^ { r } ) : = \frac { 1 } { r } \mathbb { E } \left[ \sum _ { t = 0 } ^ { \infty } \Delta ^ { r } ( t ) \sum _ { j = 1 } ^ { J } c _ { j } \left( \sum _ { k = 1 } ^ { L _ { j } } p _ { j , k } ^ { r } ( t ) + \sum _ { k = 1 } ^ { o _ { j } - 1 } d _ { j , k } ^ { r } ( t ) \right) ~ \Bigg | ~ z _ { 0 } ^ { r } \right] .\tag{7}
$$

The prelimit control problem is to minimize $\mathcal { T } ^ { r } ( \pi ^ { r } ; z _ { 0 } ^ { r } )$ over all feasible scheduling policies.

To ensure that the objective remains finite, we assume that arrivals occur only up to a fixed horizon $H < \infty , \mathrm { i . e . , } \lambda _ { j } ( \tau ) = 0$ for all $\tau > H$ . We further assume that $\lambda _ { j }$ is nonnegative and locally bounded.

## 3.2 Fluid Scaling

Within an $O ( 1 )$ unit of clock time in the �th system, since the arrival rates scale as $O ( r )$ and each GPU step takes ${ \cal O } ( 1 / r )$ units of clock time, cumulative arrivals, cumulative stage completions, and waiting-queue lengths can all be of order $O ( r )$ . This suggests the following scaling, which we use to restore stability and obtain a sensible limiting system. For $\xi \in \mathbb { R } _ { + }$ , define

$$
\bar { p } _ { j , k } ^ { r } ( \boldsymbol { \xi } ) : = \frac { 1 } { r } p _ { j , k } ^ { r } ( \lfloor r \boldsymbol { \xi } \rfloor ) , \quad \bar { d } _ { j , k } ^ { r } ( \boldsymbol { \xi } ) : = \frac { 1 } { r } d _ { j , k } ^ { r } ( \lfloor r \boldsymbol { \xi } \rfloor ) ,
$$

$$
\bar { \mathcal { U } } _ { j , k } ^ { r } ( \boldsymbol { \xi } ) : = \frac { 1 } { r } \mathcal { U } _ { j , k } ^ { r } ( \lfloor r \boldsymbol { \xi } \rfloor ) , \bar { \mathcal { V } } _ { j , k } ^ { r } ( \boldsymbol { \xi } ) : = \frac { 1 } { r } \mathcal { V } _ { j , k } ^ { r } ( \lfloor r \boldsymbol { \xi } \rfloor ) ,
$$

$$
\bar { A } _ { j } ^ { r } ( \xi ) : = \frac 1 r A _ { j } ^ { r } ( \lfloor r \xi \rfloor ) , \qquad \bar { f } ^ { r } ( \xi ) : = f ^ { r } ( \lfloor r \xi \rfloor ) .
$$

Here $\xi = t / r$ is fluid-scaled step time. When limits of these scaled processes exist, we omit the bars and the index � and write the limiting processes as $p , d , \mathcal { U } , \mathcal { V }$ , and $f .$ . The formal convergence from the prelimit system to the fluid limit is established in Section 5.

A crucial phenomenon as a consequence of of this fluid regime and the memory constraint stems from the following observation. For $k = 1 , \ldots , L _ { j } - 1$ , each request counted by $p _ { j , k } ^ { r }$ occupies $c _ { \mathrm { c h u n k } } ( L _ { j } - k )$ KV-cache slots. Likewise, for $k = 1 , \ldots , o _ { j } - 1$ , each request counted by $d _ { j , k } ^ { r ^ { \ast } }$ occupies $s _ { j } + o _ { j } - k - 1$ KV-cache slots. Since the total capacity � is fixed independently of $r ,$ the memory constraint bounds each of these unscaled states by a constant independent of $r .$ Therefore,

$$
\bar { p } _ { j , k } ^ { r } ( \xi ) \leq \frac { M } { r c _ { \mathrm { c h u n k } } ( L _ { j } - k ) }  0 \quad \mathrm { a n d } \quad \bar { d } _ { j , k } ^ { r } ( \xi ) \leq \frac { M } { r ( s _ { j } + o _ { j } - k - 1 ) }  0 .\tag{8}
$$

Thus, every fluid-scaled intermediate-stage state vanishes as $r \to \infty$ . The fluid-scaled waiting-queue length $\bar { p } _ { j , L _ { j } } ^ { r }$ need not vanish because waiting requests do not occupy GPU memory. This observation naturally leads to the phenomenon of stage-level state-space collapse, yielding a more tractable limiting system for which we can characterize an optimal policy. We provide a detailed characterization of this phenomenon in Section 3.4.

As a result, on every bounded horizon, the normalized holding-cost contribution of requests in intermediate stages vanishes. Thus, although $\mathcal { T } ^ { r }$ counts all requests in the system, its fluid counterpart contains only the waiting-queue lengths $p _ { j , L _ { j } }$

The cumulative stage completions need not vanish with these states. A fixed fluid-time interval $[ \xi _ { 1 } , \xi _ { 2 } ]$ contains $O ( r )$ GPU steps. Since each step can complete $O ( 1 )$ requests at a given stage, the cumulative number of stage completions over the interval can be $O ( r )$ . After division by $r ,$ the corresponding cumulative flow can therefore remain �(1).

## 3.3 Fluid Control Problem

We now formulate the limiting fluid control problem driven by the arrival rates $\lambda _ { j }$ . Fluid evolution is indexed by step time $\xi \in \mathbb { R } _ { + }$ , the continuous analogue of the discrete GPU-step index, whereas $\tau \in \mathbb { R } _ { + }$ still represents clock time. Under a policy �, the real-time clock $f ^ { \pi }$ maps step time $\xi$ to clock time $\tau = f ^ { \pi } ( \xi )$

At step time $\xi , p _ { j , k } ( \xi ) \in \mathbb { R } .$ <sub>+</sub> is the fluid-scaled number of class-� requests with � prefill chunks remaining, and $d _ { j , k } ( \xi ) \in \mathbb { R } _ { + }$ is the fluid-scaled number of class-� requests with � decode tokens remaining. The system state is,

$$
z _ { \xi } = \big ( \{ p _ { j , k } ( \xi ) \} _ { 1 \leq j \leq J , 1 \leq k \leq L _ { j } } , \{ d _ { j , k } ( \xi ) \} _ { 1 \leq j \leq J , 1 \leq k \leq o _ { j } - 1 } \big ) .
$$

For the fluid system, (8) implies that the memory constraint restricts admissible initial states to

$$
\begin{array} { r l } & { p _ { j , L _ { j } } ( 0 ) \geq 0 , } \\ & { p _ { j , k } ( 0 ) = 0 , \quad k = 1 , \ldots , L _ { j } - 1 , } \\ & { d _ { j , k } ( 0 ) = 0 , \quad k = 1 , \ldots , o _ { j } - 1 , } \end{array}
$$

for every class $j .$

Prefill and decode schedules are represented by the respective cumulative control processes. For $k = 1 , \ldots , L _ { j } , \mathcal { U } _ { j , k } ( \xi )$ is the fluid-scaled cumulative number of class-� requests that have completed the transition from � to $k - 1$ remaining prefill chunks by step time $\xi .$ For $k = 1 , \ldots , o _ { j } - 1 , \mathcal { V } _ { j , k } ( \xi )$ is defined analogously for decode tokens.

For cumulative control processes $\pi = ( \mathcal { U } , \mathcal { V } )$ , the real-time clock is

$$
\begin{array} { r l } & { f ^ { \pi } ( \xi ) = a _ { \mathrm { p f } } \displaystyle \sum _ { j = 1 } ^ { J } \displaystyle \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } \mathcal { U } _ { j , k } ( \xi ) } \\ & { ~ + a _ { \mathrm { k v } } \displaystyle \sum _ { j = 1 } ^ { J } \Biggl ( \displaystyle \sum _ { k = 1 } ^ { L _ { j } } [ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } ] \mathcal { U } _ { j , k } ( \xi ) + \displaystyle \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) \mathcal { V } _ { j , k } ( \xi ) \Biggr ) + b \xi . } \end{array}
$$

The three clock terms accumulate the clock time due to prefill computation, KV-cache I/O, and the fixed per-step overhead, respectively.

Given this clock, the fluid state is determined by the cumulative form of the stagewise flow-balance equations, with arrivals evaluated up to clock time $f ^ { \pi } ( \xi )$ :

$$
\begin{array} { r l r } { \boldsymbol { p } _ { j , L _ { j } } ^ { \pi } ( \boldsymbol { \xi } ) = \boldsymbol { p } _ { j , L _ { j } } ( 0 ) + \displaystyle \int _ { 0 } ^ { f ^ { \pi } ( \boldsymbol { \xi } ) } \boldsymbol { \lambda } _ { j } ( s ) d s - \mathcal { U } _ { j , L _ { j } } ( \boldsymbol { \xi } ) , } & \\ { \boldsymbol { p } _ { j , k } ^ { \pi } ( \boldsymbol { \xi } ) = \boldsymbol { p } _ { j , k } ( 0 ) + \mathcal { U } _ { j , k + 1 } ( \boldsymbol { \xi } ) - \mathcal { U } _ { j , k } ( \boldsymbol { \xi } ) , } & { k = 1 , \ldots , L _ { j } - 1 ; } & \\ { d _ { j , o _ { j } - 1 } ^ { \pi } ( \boldsymbol { \xi } ) = d _ { j , o _ { j } - 1 } ( 0 ) + \mathcal { U } _ { j , 1 } ( \boldsymbol { \xi } ) - \mathcal { V } _ { j , o _ { j } - 1 } ( \boldsymbol { \xi } ) , } & \\ { d _ { j , k } ^ { \pi } ( \boldsymbol { \xi } ) = d _ { j , k } ( 0 ) + \mathcal { V } _ { j , k + 1 } ( \boldsymbol { \xi } ) - \mathcal { V } _ { j , k } ( \boldsymbol { \xi } ) , } & { k = 1 , \ldots , o _ { j } - 2 . } & \end{array}\tag{9}
$$

The equation for $d _ { j , o _ { j } - 1 } ^ { \pi }$ applies only when $o _ { j } \ge 2 ,$ . If $o _ { j } = 1$ , the request completes immediately after its final prefill stage, and there are no decode states.

The fluid control admissibility conditions below inherit the structure of the prelimit model.

Definition 2 (Admissible fluid policy). Fix an admissible initial state $z _ { 0 }$ . A policy $\pi = ( \mathcal { U } , \mathcal { V } )$ is admissible if the following conditions hold:

1. The cumulative control processes take values in $\mathbb { R } _ { + }$ , are continuous and nondecreasing, and start from zero; i.e. $\mathcal { U } _ { j , k } ( 0 ) = 0 \mathrm { a n d } \mathcal { V } _ { j , k } ( 0 ) = 0$

2. For every $\xi _ { 1 } > \xi _ { 2 } \ge 0$ , the cumulative controls satisfy the memory constraint

$$
\begin{array} { r l r } {  { \sum _ { j = 1 } ^ { J } \biggl ( \sum _ { k = 1 } ^ { L _ { j } } [ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } ] ( \mathcal { U } _ { j , k } ( \xi _ { 1 } ) - \mathcal { U } _ { j , k } ( \xi _ { 2 } ) ) } } \\ & { } & { + \displaystyle \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) ( \mathcal { V } _ { j , k } ( \xi _ { 1 } ) - \mathcal { V } _ { j , k } ( \xi _ { 2 } ) ) \biggr ) \le M ( \xi _ { 1 } - \xi _ { 2 } ) . } \end{array}\tag{10}
$$

3. For every $\xi \ge 0$ , the waiting-queue lengths are nonnegative and all intermediate-stage states are zero:

$$
\begin{array} { r l } { p _ { j , L _ { j } } ^ { \pi } ( \boldsymbol { \xi } ) \geq 0 , \quad } & { j = 1 , \ldots , J , } \\ { p _ { j , k } ^ { \pi } ( \boldsymbol { \xi } ) = 0 , \quad } & { j = 1 , \ldots , J , \quad k = 1 , \ldots , L _ { j } - 1 , } \\ { d _ { j , k } ^ { \pi } ( \boldsymbol { \xi } ) = 0 , \quad } & { j = 1 , \ldots , J , \quad k = 1 , \ldots , o _ { j } - 1 . } \end{array}\tag{11}
$$

Definition 2 impose three sets of constraints that reflect the requirements for prelimit policies in Definition 1. First, the (deterministic) fluid cumulative control processes U and $_ \textmd { ‰}$ must start from zero and be nondecreasing. Second, the per-step memory constraint $m ^ { r } ( t ) \leq M$ in the prelimit model translates into the Lipschitz constraint (10) in the fluid model. The left-hand side measures the cumulative KV-cache usage associated with work completed during $\left( \xi _ { 2 } , \xi _ { 1 } \right]$ , which cannot exceed $M ( \xi _ { 1 } - \xi _ { 2 } )$ , the memory capacity integrated over the interval. Third, the nonnegativity of the prelimit queues is preserved in the fluid limit, while the convergence in (8) implies that all intermediate-stage fluid states must vanish.

Remark 1. Since $b > 0$ and the cumulative controls of an admissible policy are continuous and nondecreasing, $f ^ { \pi }$ is a continuous strictly increasing bijection from $\mathbb { R } _ { + }$ onto itself and therefore admits a continuous inverse. Moreover, $f ^ { \pi }$ has bounded variation on every bounded interval and hence induces a Lebesgue–Stieltjes measure $d f ^ { \pi }$ . All integrals with respect to $d f ^ { \pi }$ below are understood in this sense.

Since admissibility implies that all intermediate-stage states are zero, the fluid control problem considers the analogue of the prelimit objective (7), seeking an admissible policy � that minimizes

$$
\mathcal { T } ( \pi ; z _ { 0 } ) = \int _ { 0 } ^ { \infty } \sum _ { j = 1 } ^ { J } c _ { j } p _ { j , L _ { j } } ^ { \pi } ( \boldsymbol { \xi } ) d f ^ { \pi } ( \boldsymbol { \xi } ) .\tag{12}
$$

The cumulative holding cost is measured in clock time, with $d f ^ { \pi }$ converting increments in step time into elapsed real-time increments.

## 3.4 Stage-Level Collapse

In this subsection, we focus on the stage-level state-space collapse phenomenon and use it to simplify the fluid control problem. For an admissible policy, substituting the zero intermediate-stage constraints (11) into the stagewise flow-balance equations (9) gives

$$
\begin{array} { r l } & { 0 = p _ { j , k } ^ { \pi } ( \xi ) = p _ { j , k } ( 0 ) + \mathcal { U } _ { j , k + 1 } ( \xi ) - \mathcal { U } _ { j , k } ( \xi ) = \mathcal { U } _ { j , k + 1 } ( \xi ) - \mathcal { U } _ { j , k } ( \xi ) , \quad k = 1 , \dots , L _ { j } - 1 , } \\ & { 0 = d _ { j , o _ { j } - 1 } ^ { \pi } ( \xi ) = d _ { j , o _ { j } - 1 } ( 0 ) + \mathcal { U } _ { j , 1 } ( \xi ) - \mathcal { V } _ { j , o _ { j } - 1 } ( \xi ) = \mathcal { U } _ { j , 1 } ( \xi ) - \mathcal { V } _ { j , o _ { j } - 1 } ( \xi ) , \quad o _ { j } \ge 2 , } \\ & { 0 = d _ { j , k } ^ { \pi } ( \xi ) = d _ { j , k } ( 0 ) + \mathcal { V } _ { j , k + 1 } ( \xi ) - \mathcal { V } _ { j , k } ( \xi ) = \mathcal { V } _ { j , k + 1 } ( \xi ) - \mathcal { V } _ { j , k } ( \xi ) , \quad k = 1 , \dots , o _ { j } - 2 . } \end{array}
$$

Thus, for every class $j ,$ , define its cumulative completion process by

$$
X _ { j } ^ { \pi } ( \xi ) : = \mathcal { U } _ { j , L _ { j } } ( \xi ) = \cdots = \mathcal { U } _ { j , 1 } ( \xi ) .\tag{13}
$$

For classes with $o _ { j } \ge 2$ , this process also equals the cumulative completions at every decode stage:

$$
X _ { j } ^ { \pi } ( \xi ) = { \mathcal { V } } _ { j , o _ { j } - 1 } ( \xi ) = \cdots = { \mathcal { V } } _ { j , 1 } ( \xi ) .
$$

These identities exhibit a stage-level state-space collapse, a phenomenon well studied in the heavy-trafic and fluid limits of queueing systems; see Xu [45, Chapter 7]. The fluid state is therefore fully described by the class-level waiting queues, and for each class $j ,$ , the cumulative completion process $X _ { j } ^ { \pi }$ represents all stage-level controls.

Based on the stage-level collapse, we here reformulate the constraints and objective of the fluid control problem in terms of the cumulative class-� request completions $X _ { j } ^ { \pi } ( \xi )$ . Let first define the cumulative KV-cache usage of a class-� request, summed over its prefill and decode stages, by

$$
\tilde { m } _ { j } : = \sum _ { k = 1 } ^ { L _ { j } } \left[ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } \right] + \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) .\tag{14}
$$

Throughout the remainder of the paper, assume $\tilde { m } _ { j } > 0$ for every class $j$

Substituting (13) into the stage-level model and using $\begin{array} { r } { \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } = s _ { j } } \end{array}$ and (14) yields

$$
p _ { j , L _ { j } } ^ { \pi } ( \boldsymbol { \xi } ) = p _ { j , L _ { j } } ( 0 ) + \int _ { 0 } ^ { f ^ { \pi } ( \boldsymbol { \xi } ) } \lambda _ { j } ( s ) d s - \chi _ { j } ^ { \pi } ( \boldsymbol { \xi } ) \geq 0 .\tag{15}
$$

$$
f ^ { \pi } ( \xi ) = \sum _ { j = 1 } ^ { J } \left( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } \right) \chi _ { j } ^ { \pi } ( \xi ) + b \xi .\tag{16}
$$

For $\xi _ { 1 } > \xi _ { 2 } \ge 0 .$

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \left( \boldsymbol { \chi } _ { j } ^ { \pi } ( \boldsymbol { \xi } _ { 1 } ) - \boldsymbol { \chi } _ { j } ^ { \pi } ( \boldsymbol { \xi } _ { 2 } ) \right) \leq M ( \boldsymbol { \xi } _ { 1 } - \boldsymbol { \xi } _ { 2 } ) .\tag{17}
$$

Conversely, any collection of nondecreasing processes $\{ X _ { j } ^ { \pi } \} _ { j = 1 } ^ { J }$ satisfying the initial conditions $X _ { i } ^ { \pi } ( 0 ) = 0$ the queue nonnegativity constraint (15), and the memory constraint (17), with the clock defined by (16), uniquely determines an admissible fluid policy $\pi = ( \mathcal { U } , \mathcal { V } )$ . Therefore, we henceforth study the fluid control problem in terms of $\{ X _ { j } ^ { \pi } \} _ { j = 1 } ^ { J }$

## 4 Optimal Control of the Fluid Model

In this section, we study the $c \mu$ priority policy for the fluid model. We first introduce the basic idea of the $c \mu$ rule and define the efective service rate $\mu _ { j }$ . We then describe the fluid $c \mu$ policy and establish that it minimizes the cumulative holding cost. Finally, we show that the policy also minimizes the completion time and, consequently, maximizes the full-run output-token throughput.

## 4.1 Efective Service Rates and the $c \mu$ Rule

The fluid control objective (12) can be viewed as a linear holding cost on the queue lengths of waiting requests. Each unit of waiting class $- j$ requests incurs a holding cost at rate $c _ { j }$ . Serving class- $- j$ requests at rate $\mu _ { j }$ therefore reduces the holding-cost rate by $c _ { j } \mu _ { j }$ per unit of time. This motivates the classical $c \mu$ rule, which prioritizes classes in decreasing order of $c _ { j } \mu _ { j }$ . To apply this rule to our fluid model and establish its optimality, we first identify the efective service rates determined by the memory constraint and the step-time model.

Consider an interval during which only class � is served. The memory constraint in (17) gives, in diferential form, that $\tilde { m } _ { j } d X _ { i } ^ { \pi } ( \xi ) \leq M d \xi$ . Hence the class-� service rate in step time is at most $M / \tilde { m } _ { j }$ , with equality when the KV capacity is saturated. At this saturated service rate, the amount of class-� service completed per unit of step time is $M / \tilde { m } _ { j }$ , while the corresponding elapsed clock time is

$$
\left( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } \right) \frac { M } { \tilde { m } _ { j } } + b .
$$

This motivates the following definition.

Definition 3 (Efective service rate). The efective service rate of class $j$ is

$$
\mu _ { j } : = \frac { M / \tilde { m } _ { j } } { ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) M / \tilde { m } _ { j } + b } = \left[ a _ { \mathrm { p f } } s _ { j } + \left( a _ { \mathrm { k v } } + \frac { b } { M } \right) \tilde { m } _ { j } \right] ^ { - 1 } .\tag{18}
$$

The same coeficients also characterize the allocation of real-time service among multiple classes. Substituting the memory constraint (17) into the equation for the real-time clock (16) gives

$$
\begin{array} { l } { { d f ^ { \pi } ( \xi ) = \displaystyle \sum _ { j = 1 } ^ { J } \left( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } \right) d X _ { j } ^ { \pi } ( \xi ) + b d \xi } } \\ { { \displaystyle \geq \sum _ { j = 1 } ^ { J } \left[ a _ { \mathrm { p f } } s _ { j } + \left( a _ { \mathrm { k v } } + \frac { b } { M } \right) \tilde { m } _ { j } \right] d X _ { j } ^ { \pi } ( \xi ) = \displaystyle \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } d X _ { j } ^ { \pi } ( \xi ) } , } \end{array}
$$

with equality whenever the KV capacity is saturated. Here and below, $d X _ { i } ^ { \pi } / d f ^ { \pi }$ denotes the Radon–Nikodym derivative of the Lebesgue–Stieltjes measure induced by $\chi _ { j } ^ { \pi }$ with respect to that induced by $f ^ { \pi }$ . Define

$$
u _ { j } ^ { \pi } ( \xi ) : = \mu _ { j } ^ { - 1 } \left( { \frac { d X _ { j } ^ { \pi } } { d f ^ { \pi } } } \right) ( \xi ) .
$$

Then

$$
\left( \frac { d X _ { j } ^ { \pi } } { d f ^ { \pi } } \right) ( \xi ) = \mu _ { j } u _ { j } ^ { \pi } ( \xi ) , \qquad \sum _ { j = 1 } ^ { J } u _ { j } ^ { \pi } ( \xi ) \le 1 ,
$$

where the inequality is an equality whenever the KV capacity is saturated. Here, $u _ { j } ^ { \pi }$ is the fraction of real-time service efort allocated to class $j ,$ and $\mu _ { j }$ is its efective service rate. Thus, the reduced GPU model has the same service-efort representation as the classical multiclass single-server model.

We apply the $c \mu$ rule using these efective service rates and index the classes so that

$$
c _ { 1 } \mu _ { 1 } \geq c _ { 2 } \mu _ { 2 } \geq \cdot \cdot \cdot \geq c _ { J } \mu _ { J } .
$$

The policy allocates service in this priority order, serving each class at the maximum rate permitted by its waiting requests, new arrivals, and the memory constraint (17), and passing any remaining capacity to lower-priority classes. For a more detailed characterization of the $c \mu$ rule and its generalizations, see for example Buyukkoc et al. [8].

## 4.2 Holding-Cost Optimality

We measure the GPU’s service budget in units of clock time: an interval of length $\Delta \tau$ provides $\Delta \tau$ units of budget. Since $\mu _ { j }$ is the efective service rate of class �, processing � class-� requests requires $x / \mu _ { j }$ units of efective work. At each time, the fluid $c \mu$ policy considers classes in decreasing order of $c _ { j } \mu _ { j }$ . For each class, it uses the available budget to serve waiting requests and new arrivals, then passes any remaining budget to the next class. To establish the holding-cost optimality of the $c \mu$ rule, we use cumulative completion processes to represent the policy and construct these processes recursively in priority order.

For this recursive construction, let $A _ { j } ( \tau )$ denote the cumulative class-� arrivals by clock time �:

$$
A _ { j } ( \tau ) : = p _ { j , L _ { j } } ( 0 ) + \int _ { 0 } ^ { \tau } \lambda _ { j } ( s ) d s .\tag{19}
$$

Definition 4 (Fluid $c \mu$ policy). Set $R _ { 0 } ^ { * } ( \tau ) : = \tau$ for $\tau \geq 0$ . Recursively, for $j = 1 , \dots , J ,$ define

$$
R _ { j } ^ { \ast } ( \tau ) : = \operatorname* { s u p } _ { 0 \leq s \leq \tau } \left[ R _ { j - 1 } ^ { \ast } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } .\tag{20}
$$

Define

$$
F ( \tau ) : = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \mu _ { j } \left( R _ { j - 1 } ^ { * } ( \tau ) - R _ { j } ^ { * } ( \tau ) \right) + \frac { 1 } { b } R _ { J } ^ { * } ( \tau ) ,\tag{21}
$$

and let the policy clock be its generalized inverse:

$$
f ^ { * } ( \xi ) : = \operatorname* { i n f } \left\{ \tau \geq 0 : F ( \tau ) \geq \xi \right\} .\tag{22}
$$

For each class $j = 1 , \dots , J ,$ , set

$$
\begin{array} { r } { X _ { j } ^ { \ast } ( \xi ) : = \mu _ { j } \left( R _ { j - 1 } ^ { \ast } ( f ^ { \ast } ( \xi ) ) - R _ { j } ^ { \ast } ( f ^ { \ast } ( \xi ) ) \right) , } \end{array}\tag{23}
$$

and define $\pi ^ { * } : = ( \mathcal { U } ^ { * } , \mathcal { V } ^ { * } )$ by

$$
\mathcal { U } _ { j , k } ^ { * } = X _ { j } ^ { * } , \quad k = 1 , \ldots , L _ { j } , \quad \mathrm { a n d } \quad \mathcal { V } _ { j , k } ^ { * } = X _ { j } ^ { * } , \quad k = 1 , \ldots , o _ { j } - 1 .
$$

Remark 2 (Policy interpretation). The basic idea is to use Skorokhod reflection to represent the remaining service budget passed to the next class and use the efective service rate $\mu _ { j }$ to convert the budget used by class $j$ into its cumulative completion process. Specifically, $R _ { 0 } ^ { * } ( \tau ) = \tau$ is the total service budget available by clock time �. After classes $1 , \ldots , j$ have been considered, $R _ { i } ^ { * } ( \tau )$ is the cumulative unused service budget passed to the lower-priority classes. Hence, $R _ { j - 1 } ^ { * } ( \tau ) - R _ { j } ^ { * } ( \tau )$ is the service budget used by class $j .$ . Since these budgets are indexed by clock time �, we evaluate this diference at $\tau = f ^ { * } ( \xi )$ . Multiplying it by $\mu _ { j }$ then gives $X _ { j } ^ { * } ( \xi )$ , the cumulative number of class-� completions by step time $\xi .$

Theorem 1 (Fluid holding-cost optimality). The fluid �� policy $\pi ^ { * }$ is admissible and minimizes the cumulative holding cost. Specifically,for every admissible policy $\pi ,$

$$
\mathcal { T } ( \pi ^ { \ast } ; z _ { 0 } ) \leq \mathcal { T } ( \pi ; z _ { 0 } ) .\tag{24}
$$

The proof is given in Appendix C.

To better illustrate the priority rule, we consider the case with no new arrivals. We show in Appendix E that the policy admits the following closed-form representation:

$$
\chi _ { j } ^ { * } ( \xi ) = \left[ { \frac { M } { \tilde { m } _ { j } } } \left( \xi - \sum _ { \ell = 1 } ^ { j - 1 } { \frac { \tilde { m } _ { \ell } p _ { \ell , L _ { \ell } } ( 0 ) } { M } } \right) \right] _ { 0 } ^ { p _ { j , L _ { j } } ( 0 ) } .\tag{25}
$$

Here $[ x ] _ { 0 } ^ { y } : = \operatorname* { m a x } \{ 0 , \operatorname* { m i n } \{ x , y \} \}$ . In (25), the slope $M / \tilde { m } _ { j }$ represents the service rate of class � when it occupies the full memory capacity. At this rate, completing its initial requests takes $\tilde { m } _ { j } p _ { j , L _ { j } } ( 0 ) / M$ units of step time. The shift $\textstyle \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } p _ { \ell , L _ { \ell } } ( 0 ) / M$ therefore equals the total service time required by the preceding classes. The truncation keeps $X _ { j } ^ { * } ( \xi )$ between 0 and $p _ { j , L _ { j } } ( 0 )$ . Thus, the formula serves the classes sequentially in index order, which is the decreasing order of $c _ { j } \mu _ { j }$ prescribed by the $c \mu$ rule. Figure 4 illustrates the $c \mu$ priority rule for three classes, with each class beginning service when the preceding class has been completed. For an illustration under general arrivals, see Appendix B.

![](images/0bb2688c7328a11127c66666f6a8374f32e6a9920622d2334519f669405de24a.jpg)  
Figure 4: Illustration of the fluid �� policy for $J = 3 ,$ , assuming classes are already indexed in priority order. Here $M = 1 , ( p _ { 1 , L _ { 1 } } ( 0 ) , p _ { 2 , L _ { 2 } } ( 0 ) , p _ { 3 , L _ { 3 } } ( 0 ) ) = ( 2 , 3 , 1 )$ , and $( \tilde { m } _ { 1 } , \tilde { m } _ { 2 } , \tilde { m } _ { 3 } ) = ( 1 , 2 , 4 )$

## 4.3 Completion Time and Throughput Optimality

For an admissible policy �, the clock $f ^ { \pi }$ admits a continuous inverse, as noted in Remark 1. Therefore we define its average output-token throughput over the real-time interval $[ 0 , \tau ]$ by

$$
\begin{array} { l } { { \displaystyle { \sf T P } ^ { \pi } ( \tau ) : = \frac { 1 } { \tau } \sum _ { j = 1 } ^ { J } \left[ \mathcal { U } _ { j , 1 } \big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau ) \big ) + \sum _ { k = 1 } ^ { o _ { j } - 1 } \mathcal { V } _ { j , k } \big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau ) \big ) \right] } } \\ { { \displaystyle ~ = \frac { 1 } { \tau } \sum _ { j = 1 } ^ { J } o _ { j } \chi _ { j } ^ { \pi } \big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau ) \big ) , \qquad \tau > 0 . } } \end{array}\tag{26}
$$

The second equality follows from (13).

Because $\lambda _ { j }$ vanishes after $H , A _ { j } ( H )$ is the total number of class- $- j$ requests. For an admissible policy � that eventually completes all requests, define its completion time as

$$
\tau _ { \mathrm { c } } ^ { \pi } : = \operatorname* { i n f } \Big \{ \tau \geq 0 : \chi _ { j } ^ { \pi } \Big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau ) \Big ) = A _ { j } ( H ) \Big \} .\tag{27}
$$

The equality must hold for all classes simultaneously. If $\textstyle \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H ) > 0$ , then $\tau _ { \mathrm { c } } ^ { \pi } > 0$ and

$$
\mathrm { T P } ^ { \pi } ( \tau _ { \mathrm { c } } ^ { \pi } ) = \frac { \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H ) } { \tau _ { \mathrm { c } } ^ { \pi } } .\tag{28}
$$

Theorem 2 (Fluid completion-time and throughput optimality). The �� policy $\pi ^ { * }$ in Definition 4 minimizes the time needed to complete all requests. Specifically, for every admissible policy � that eventually completes all requests,

$$
\tau _ { \mathrm { c } } ^ { \pi ^ { * } } \leq \tau _ { \mathrm { c } } ^ { \pi } .\tag{29}
$$

When $\textstyle \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H ) > 0$ , completion-time optimality also implies that $\pi ^ { * }$ maximizes thefull-run output-token throughput:

$$
\mathrm { T P } ^ { \pi } ( \tau _ { \mathrm { c } } ^ { \pi } ) \leq \mathrm { T P } ^ { \pi ^ { * } } ( \tau _ { \mathrm { c } } ^ { \pi ^ { * } } ) .\tag{30}
$$

The proof is given in Appendix D.

## 5 Fluid-Limit Convergence

In this section, we demonstrate that the optimal fluid $c \mu$ policy provides operational guidance for the prelimit �th system. We construct a smoothed ��-inspired dynamic policy that is admissible at every finite scale �. At a high level, it follows the $c \mu$ priority order when forming admission plans and spreads each plan over a short window to satisfy the KV-cache capacity constraint. In particular, this policy does not require knowledge of the arrival rates $\lambda _ { j } ( \cdot )$ . Then, we show that under this policy and fluid scaling, the prelimit system and control converge to the fluid system and control under the optimal $c \mu$ policy.

To track the memory used by a request that runs continuously after admission, define the reserved KV-cache usage of a class-� request at the start of the GPU step occurring ℎ steps after admission by

$$
\begin{array} { r } { k _ { j } ^ { \mathrm { k v } } ( h ) : = \left\{ \begin{array} { l l } { \operatorname* { m i n } \{ c _ { \mathrm { c h u n k } } ( h + 1 ) , s _ { j } \} , 0 \leq h \leq L _ { j } - 1 , } \\ { s _ { j } + h - L _ { j } + 1 , L _ { j } \leq h \leq L _ { j } + o _ { j } - 2 , } \\ { 0 , \mathrm { o t h e r w i s e } . } \end{array} \right. } \end{array}
$$

During prefill, this includes the KV slots reserved for the current chunk. During decode, it includes the prompt KV states, the KV states of previously generated tokens, and the slot reserved for the current input token. It becomes zero after the request completes. Thus, with

$$
H _ { \mathrm { k v } } : = \operatorname* { m a x } _ { 1 \leq j \leq J } ( L _ { j } + o _ { j } - 2 ) ,
$$

the cumulative KV-cache usage $\tilde { m } _ { j }$ satisfies

$$
\sum _ { h = 0 } ^ { H _ { \mathrm { k v } } } k _ { j } ^ { \mathrm { k v } } ( h ) = \sum _ { k = 1 } ^ { L _ { j } } \left[ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } \right] + \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) = \tilde { m } _ { j } .\tag{31}
$$

The KV-cache usage at step � is then

$$
m ^ { r } ( t ) = \sum _ { j = 1 } ^ { J } \sum _ { h = 0 } ^ { H _ { \mathrm { k v } } } k _ { j } ^ { \mathrm { k v } } ( h ) \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t - h ) .\tag{32}
$$

For class $j ,$ the requests admitted at step $t - h$ contribute $k _ { j } ^ { \mathrm { k v } } ( h ) \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t - h )$ to the KV-cache usage at step �. Summing these contributions over all classes and lags gives $m ^ { r } ( t )$ . Using the priority order $c _ { 1 } \mu _ { 1 } \geq \cdot \cdot \cdot \geq c _ { J } \mu _ { J }$ , we define the smoothed $c \mu$ -inspired policy as follows.

Definition 5 (Smoothed ��-inspired policy). Choose integer-valued $w _ { r }$ such that

$$
w _ { r } \longrightarrow \infty , \qquad \frac { w _ { r } } { r } \longrightarrow 0 , \qquad w _ { r } > H _ { \mathrm { k v } } ;
$$

set $y _ { j } ^ { r } ( t ) = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) = 0$ for $t < 0$ , and define the new class-� planned-admission amount recursively in priority order by

$$
y _ { j } ^ { r } ( t ) = \operatorname* { m i n } \Biggl \{ \Biggl [ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \sum _ { u < t } y _ { j } ^ { r } ( u ) \Biggr ] ^ { + } , \tilde { m } _ { j } ^ { - 1 } \left[ \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell < j } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \right] ^ { + } \Biggr \} .\tag{33}
$$

Construct the admission increment directly by the moving average

$$
\Delta { \mathcal U } _ { j , L _ { j } } ^ { r } ( t ) \ : = \ : \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } y _ { j } ^ { r } ( t - \nu ) .\tag{34}
$$

Recall that we allow fractional requests, so both the planned-admission amounts $y _ { j } ^ { r } ( t )$ and the admission increments $\Delta { \mathcal { U } } _ { j , L _ { j } } ^ { r } ( t )$ may take noninteger values. For $k = 1 , \ldots , L _ { j } - 1$ , we have

$$
\Delta \mathcal { U } _ { j , k } ^ { r } ( t ) = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } \big ( t - ( L _ { j } - k ) \big ) .\tag{35}
$$

For $k = 1 , \ldots , o _ { j } - 1$ , we have

$$
\Delta \mathcal { V } _ { j , k } ^ { r } ( t ) = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } \big ( t - ( L _ { j } + o _ { j } - 1 - k ) \big ) .\tag{36}
$$

Remark 3. The choice $w _ { r } = \operatorname* { m a x } \{ H _ { \mathrm { k v } } + 1 , \lceil \sqrt { r } \rceil \}$ satisfies the required conditions. The condition $w _ { r } \to \infty$ smooths the step-level admission plans, whereas $w _ { r } / r \to 0$ ensures that the smoothing window vanishes in fluid time. At step $t , y _ { j } ^ { r } ( t )$ specifies how many class-� requests are assigned an admission plan. Equation (34) schedules $y _ { j } ^ { r } ( t ) / w ,$ of this plan to enter the GPU in each of steps $t , \dots , t + w _ { r } - 1$ , so the plan is carried out over a �<sub>�</sub>-step window. Once admitted, each request runs continuously until completion; its subsequent stage-completion increments are therefore the time shifts in (35)–(36). In Equation (33), the first term is the number of class-� requests available at step � that have not yet received an admission plan, while the second is the remaining per-step planning budget expressed in class-� requests. The first term can equivalently be written using the waiting-queue length as

$$
\left[ p _ { j , L _ { j } } ^ { r } ( t ) - \frac { 1 } { w _ { r } } \sum _ { \nu = 1 } ^ { w _ { r } - 1 } ( w _ { r } - \nu ) y _ { j } ^ { r } ( t - \nu ) \right] ^ { + } .
$$

The subtracted sum is the portion of earlier admission plans that has not yet entered the GPU, so pending plans are not included when a new plan is formed.

The following proposition establishes the admissibility of the policy in the prelimit queueing model.

Proposition 1 (Admissibility). Suppose that, initially, some requests may be waiting, but no requests are being processed on the GPU. Then, the policy in Definition 5 is admissible for the prelimit queueing model.

Proposition 1 establishes policy admissibility at every finite scale. Its proof is deferred to Appendix F. We next show that the smoothed ��-inspired policy converges under fluid scaling to the fluid $c \mu$ policy.

Theorem 3 (Fluid-limit convergence). Suppose initially no requests are being processed on the GPU, and the initial waiting queue lengths satisfy

$$
\frac 1 r p _ { j , L _ { j } } ^ { r } ( 0 ) \longrightarrow p _ { j , L _ { j } } ( 0 ) , \qquad j = 1 , \ldots , J .
$$

Consider prelimit systems (6) controlled by the smoothed ��-inspired policy in Definition 5. Then,for every $T < \infty$ and every class $j ,$ the prelimit cumulative control processes and system clock converge to the limiting ��-policy and the corresponding clock:

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \frac { 1 } { r } \mathcal { U } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) | \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , L _ { j } ,\tag{37}
$$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \frac { 1 } { r } \mathcal { V } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - \chi _ { j } ^ { * } ( \xi ) | \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , o _ { j } - 1 ;\tag{38}
$$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \bar { f } ^ { r } ( \xi ) - f ^ { * } ( \xi )  \overset { \mathbb { P } } {  } 0 .\tag{39}
$$

Moreover, the state processes converge to that of the limiting system under the optimal ��-policy:

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \frac { 1 } { r } p _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - p _ { j , k } ^ { \pi ^ { * } } ( \xi ) | \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , L _ { j } ,\tag{40}
$$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \frac { 1 } { r } d _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - d _ { j , k } ^ { \pi ^ { * } } ( \xi ) | \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , o _ { j } - 1 .\tag{41}
$$

The proof of Theorem 3 is deferred to Appendix G. The limit $( f ^ { * } , X ^ { * } )$ is the fluid $c \mu$ policy in Definition 4. It is holding-cost optimal by Theorem 1 and completion-time and throughput optimal by Theorem 2. Thus, the smoothing used at finite scale disappears under fluid scaling, and the real-time clock, system states, and the fluid-scaled cumulative stage-completion processes converge uniformly on compacts in probability to the optimal fluid trajectory.

## 6 Numerical Experiments

In this section, we evaluate two practical implementations of the $c \mu$ ordering on a real LLM serving engine. We conduct semi-synthetic experiments using prompt lengths, decode lengths, and arrival rates derived from a real-world dataset [40].

## 6.1 Experimental Setup

We run Meta-Llama-3-8B-Instruct [19] using vLLM [25], with each run using a single NVIDIA H800 GPU. Within each experimental group, all policies are run sequentially on the same GPU. We calibrate the step-time model separately for each GPU to estimate $a _ { \mathrm { p f } } , a _ { \mathrm { k v } } .$ , and �. Substituting these estimates and request �’s prompt and output lengths into the efective-service-rate formula (18) gives its estimated service rate $\widehat { \mu _ { i } }$ for the $c \mu$ ordering.

We use the To-B, To-C, Thinking, and Coder traces from the Qwen-Bailian Anonymous Dataset [40]. To-B covers business-facing API calls and task automation, while To-C covers consumer-facing interactive chat. Thinking focuses on reasoning-intensive chat, and Coder covers code generation and interactive programming. These anonymized production traces each cover a two-hour period. Each record provides a relative arrival time and prompt and output lengths. We retain only requests whose combined prompt and output lengths do not exceed the 8K-token context window of Meta-Llama-3-8B-Instruct. We adjust the arrival rates to achieve a trafic intensity of approximately 0.95. For replay, we configure the LLM engine to generate exactly the specified number of output tokens for each request. Table 2 summarizes their lengths and eligible request counts.

Table 2: Eligible requests in the four production traces. Prompt and output lengths are measured in tokens.
<table><tr><td>Trace</td><td>Requests</td><td>Mean prompt length</td><td>Mean output length</td></tr><tr><td>To-B</td><td>172,015</td><td>863.0</td><td>88.4</td></tr><tr><td>To-C</td><td>39,937</td><td>1635.2</td><td>407.2</td></tr><tr><td>Thinking</td><td>7,057</td><td>2494.2</td><td>1993.9</td></tr><tr><td>Coder</td><td>27,099</td><td>2539.7</td><td>727.8</td></tr></table>

Figure 5 compares the joint prompt–output length distributions. On average, To-B has the shortest prompts and outputs, followed by To-C. Thinking and Coder have similarly long prompts, but Thinking has substantially longer outputs, consistent with its focus on long-form reasoning. We use To-B to construct the semi-synthetic workloads in Section 6.3, and replay all four traces after time compression in Section 6.4.

## 6.2 Policies and Performance Measures

Based on the $c \mu$ priority rule, we implement two practical scheduling policies: $c \mu$ and ��-backfill. We compare them with the default vLLM scheduler, MC-SF [23], SVF [24], and WAIT-frac, a fractional implementation of WAIT [4]. The $c \mu , c \mu \cdot$ -backfill, MC-SF, and SVF policies order waiting requests by priority. All policies except the default vLLM scheduler use nonpreemptive implementations and assume that each request’s output length is known.

These nonpreemptive policies check future KV-cache usage before admission, accounting for cache growth and memory released upon completion. A request is admitted only if the total KV-cache usage will remain within � at every future GPU step.

Using the future KV-cache feasibility check, the �� policy scans waiting requests in decreasing $c _ { i } \widehat { \mu } _ { i }$ order and stops at the first infeasible request. The $c \mu .$ -backfill policy uses the same ordering and feasibility check. When a request cannot be admitted immediately, the policy uses the future KV-cache feasibility check to find its earliest feasible future admission step and assumes that the request is admitted at that step in subsequent checks. It then continues scanning lower-priority requests, admitting a request immediately if the feasibility check is satisfied with all these planned admissions included. Figure 6 illustrates this procedure. Both policies are nonpreemptive and continuously serve admitted requests until completion.

![](images/14b59e0de689216b40619e1a4962f4fce77e6564e59581f1f63f9504eb23a328.jpg)  
(a) To-B

![](images/15f624ff00138c3074db18e6a9d380e38915e9c31a9f402541cb10b77303becc.jpg)  
(b) To-C

![](images/d1540f7b254d061640e0d49c76a85969e2170a9137f1fb63a4f52975db9f4455.jpg)  
(c) Thinking

![](images/981b3492cb0265437e43357c70cd77f3badf155d7e902fd36722883c00529142.jpg)  
(d) Coder  
Figure 5: Joint prompt–output length distributions of eligible requests in (a) To-B, (b) To-C, (c) Thinking, and (d) Coder. Both axes use logarithmic token scales.

![](images/9e72e128c58b620131a5e33ab8889a739b1084843422416a046cc43527cf46f3.jpg)  
Figure 6: Illustrative KV-cache profiles for reservations and backfilling. The blue area shows 10 resident or already selected requests. The orange, purple, and red areas add three reserved requests, each with 512 output tokens, starting at $t + 2 0 0 , t + 3 6 0$ , and $t + 5 6 0$ , respectively. The green area adds a smaller backfill request with 400 output tokens at �. KV-cache usage grows as tokens are generated and drops at completion. The combined usage remains below �, preserving all three reserved starts.

MC-SF orders waiting requests by increasing output length $o _ { i } .$ , while SVF orders them by increasing cumulative KV-cache volume $s _ { i } o _ { i } + o _ { i } ( o _ { i } + 1 ) / 2$ . Both use nonpreemptive implementations with the same future KV-cache feasibility check. When $c _ { i } = 1 , a _ { \mathrm { p f } } = 0$ , and prefill is not chunked, the $c \mu$ rule reduces to a volume-based priority rule, as in SVF.

The vanilla version of WAIT described in [4] uses positive integer service quotas for each class and stage, so the smallest positive quota is one request per GPU step. When the quota is one, a single waiting request is suficient to make its class eligible for service in the next GPU step, eliminating the intended waiting mechanism for request accumulation. In our experiments, however, matching class-wise arrival rates may require service quotas below one request per GPU step. We therefore introduce WAIT-frac, which maintains fractional credits separately for each class and stage. Serving one request at a stage consumes one unit of credit; fractional remainders carry over to subsequent steps, subject to an expiry rule. Runtime KV-cache checks enforce memory feasibility.

In WAIT-frac, the reference step time $T _ { \mathrm { W A I T } }$ converts class-wise arrival rates into the credit increments for each GPU step. Under our GPU configuration, the original parameter formula for WAIT does not yield a positive value of $T _ { \mathrm { W A I T } }$ , because Ao et al. [4] assumes a diferent batch service-time model. We therefore set $T _ { \mathrm { W A I T } } = 1 2 0$ ms, chosen to be comparable to the GPU-step time at peak load.

For a trace containing � requests, let $\tau _ { i } ^ { \mathrm { a r r } }$ and $\tau _ { i } ^ { \mathrm { { c o m p } } }$ be the arrival and completion times of request �. We define the total request latency as

$$
\mathrm { L a t e n c y } _ { \mathrm { t o t } } : = \sum _ { i = 1 } ^ { N } \left( \tau _ { i } ^ { \mathrm { c o m p } } - \tau _ { i } ^ { \mathrm { a r r } } \right) .
$$

We set $c _ { i } = 1$ for every request, so $c Q ( \tau )$ counts all arrived but unfinished requests, including those on the GPU. Our primary performance measure is the average request latency:

$$
\mathrm { L a t e n c y } _ { \mathrm { a v g } } : = { \frac { 1 } { N } } \mathrm { L a t e n c y } _ { \mathrm { t o t } } = { \frac { 1 } { N } } \int _ { 0 } ^ { \infty } c Q ( \tau ) d \tau .
$$

Remark 4. Request � contributes one to $c Q ( \tau )$ exactly when $\tau _ { i } ^ { \mathrm { a r r } } \leq \tau < \tau _ { i } ^ { \mathrm { c o m p } }$ . Summing these contributions

and integrating gives

$$
\int _ { 0 } ^ { \infty } c Q ( \tau ) d \tau = \sum _ { i = 1 } ^ { N } \int _ { \tau _ { i } ^ { \mathrm { a r } } } ^ { \tau _ { i } ^ { \mathrm { c o m p } } } 1 d \tau = \sum _ { i = 1 } ^ { N } \left( \tau _ { i } ^ { \mathrm { c o m p } } - \tau _ { i } ^ { \mathrm { a r r } } \right) = \mathrm { L a t e n c y } _ { \mathrm { t o t } } .
$$

We also report the mean time to first token (TTFT), measured from arrival to the first output, and the 99th percentile of request latency (P99 latency). Clearance time is max $\phantom { } _ { i } \tau _ { i } ^ { \mathrm { c o m p } } - \operatorname* { m i n } _ { i } \tau _ { i } ^ { \mathrm { a r r } }$ . Output-token throughput is the total number of generated tokens divided by the time from experiment start to the last completion.

## 6.3 Semi-Synthetic Workloads

We estimate a Gaussian mixture model to cluster eligible To-B requests into 16 classes. For each class, we generate Poisson arrivals with time-varying rates based on the source trace and rescale the arrival times to attain a trafic intensity of 0.95. We then assume that all requests within each class have identical prompt and decode lengths, given by their respective cluster centers. We repeat the experiment eight times, replaying the same generated trace for every policy within each repetition.

Table 3 summarizes all eight repetitions. The $c \mu { \mathrm { - } } { \mathrm { b a c k f i l l } }$ policy has the lowest mean latency, mean TTFT, and mean clearance time, and the highest mean output-token throughput. The $c \mu$ policy also achieves low mean latency, only slightly above ��-backfill and below all other baselines. WAIT-frac has the lowest mean P99 latency, showing that the ranking can difer between average and tail latency.

Table 3: Results on eight semi-synthetic traces. Average latency and clearance are mean ± sample standard deviation; the other columns report means across repetitions. P99 latency is computed within each repetition before averaging. Bold values are best among the reported policies.
<table><tr><td>Policy</td><td>Average latency (s)</td><td>Mean  $\mathrm { T T F T } \left( \mathrm { s } \right)$ </td><td>P99 latency (s)</td><td>Clearance (s)</td><td>Output tokens/s</td></tr><tr><td>cµ-backfill</td><td> $\mathbf { 1 0 . 3 6 6 \pm 0 . 4 2 4 }$ </td><td>3.593</td><td>68.678</td><td> $^ { 3 , 6 3 8 . 4 \pm 1 2 . 5 }$ </td><td>2,521.1</td></tr><tr><td>cμ</td><td> $1 0 . 7 5 9 \pm 0 . 4 4 7$ </td><td>3.901</td><td>71.923</td><td> $3 , 6 4 6 . 2 \pm 1 0 . 2$ </td><td>2,515.7</td></tr><tr><td>SVF</td><td> $1 1 . 1 7 7 \pm 0 . 4 8 4$ </td><td>4.340</td><td>140.005</td><td> $3 , 6 4 7 . 9 \pm 1 7 . 5$ </td><td>2,514.6</td></tr><tr><td>MC-SF</td><td> $1 8 . 2 4 3 \pm 1 . 4 8 0$ </td><td>11.746</td><td>348.470</td><td> $3 , 6 4 2 . 2 \pm 1 3 . 3$ </td><td>2,518.5</td></tr><tr><td>WAIT-frac (120 ms)</td><td> $1 4 . 6 7 1 \pm 0 . 7 0 3$ </td><td>8.604</td><td>67.163</td><td> $3 , 6 5 5 . 8 \pm 1 5 . 7$ </td><td>2,509.1</td></tr><tr><td>vLLM</td><td> $3 0 . 2 2 6 \pm 3 . 6 6 4$ </td><td>23.548</td><td>92.676</td><td> $3 , 6 5 0 . 6 \pm 1 8 . 3$ </td><td>2,512.8</td></tr></table>

## 6.4 Real Production Traces

We next evaluate the policies on To-B, To-C, Thinking, and Coder while retaining their original request sequences. For each eligible source trace, we multiply all relative arrival times by a common factor to attain a trafic intensity of 0.95. This changes the arrival rate while preserving the request order, relative interarrival gaps, and each prompt–output length pair. The length distributions are therefore those shown in Figure 5; Figure 7 shows the time-varying arrival rate of To-B after compression. We report one run per policy for each source trace, with all policies replaying the same compressed trace.

Table 4 summarizes the results on the four compressed traces. On To-B, both $c \mu$ policies achieve lower average latency and mean TTFT than the baselines, with $c \mu \cdot$ -backfill performing best. On To-C and Coder, ��-backfill, ${ } _ { c \mu , }$ , and SVF have similar average latency and mean TTFT. On Thinking, $c \mu$ achieves the lowest average latency and mean TTFT.

![](images/a97599e786fbb888429b28532fc9fdde1bc5c1b0eeb02700cb601937d75af9a1.jpg)  
Figure 7: Arrival rate of the time-compressed To-B production trace.

Table 4: Results on four compressed traces. Average latency $( \mathrm { A v g . } )$ , mean TTFT, and P99 latency are in seconds; throughput (TP) is in output tokens/s. Bold values are best within each trace among the compared policies.
<table><tr><td></td><td colspan="4">To-B</td><td colspan="4">To-C</td><td colspan="4">Thinking</td><td colspan="4">Coder</td></tr><tr><td>Policy</td><td>Avg.</td><td>TTFT</td><td>P99</td><td>TP</td><td>Avg.</td><td>TTFT</td><td>P99</td><td>TP</td><td>Avg.</td><td>TTFT</td><td>P99</td><td>TP</td><td>Avg.</td><td>TTFT</td><td>P99</td><td>TP</td></tr><tr><td>cµ-backfill</td><td>9.472</td><td>2.56</td><td>121</td><td>2851</td><td>28.265</td><td>14.20</td><td>666</td><td>3553</td><td>160.097</td><td>105.84</td><td>2134</td><td>3543</td><td>23.145</td><td>1.90</td><td>125</td><td>3139</td></tr><tr><td>cµ</td><td>10.398</td><td>3.20</td><td>128</td><td>2821</td><td>28.603</td><td>14.22</td><td>685</td><td>3521</td><td>128.826</td><td>74.79</td><td>1838</td><td>3526</td><td>23.151</td><td>1.95</td><td>127</td><td>3129</td></tr><tr><td>SVF</td><td>10.688</td><td>3.51</td><td>144</td><td>2821</td><td>28.137</td><td>14.05</td><td>675</td><td>3526</td><td>139.941</td><td>84.98</td><td>1918</td><td>3499</td><td>23.141</td><td>1.94</td><td>127</td><td>3130</td></tr><tr><td>MC-SF</td><td>11.911</td><td>5.15</td><td>182</td><td>2836</td><td>37.158</td><td>22.95</td><td>969</td><td>3498</td><td>152.133</td><td>97.21</td><td>1949</td><td>3471</td><td>25.902</td><td>3.69</td><td>152</td><td>3114</td></tr><tr><td>vLLM</td><td>35.492</td><td>28.77</td><td>130</td><td>2838</td><td>117.943</td><td>103.24</td><td>496 </td><td>3446</td><td>361.491</td><td>300.61</td><td>829</td><td>3282</td><td>69.700</td><td>46.89</td><td>285</td><td>3045</td></tr></table>

## 7 Conclusion and Future Work

In this paper, we propose a simple modification of the �� rule and establish its asymptotic optimality for LLM inference scheduling under memory constraints. Interesting directions for future research include extending the queueing model to multi-GPU settings and agentic workloads, as well as investigating the impact of output-length prediction errors on scheduling performance.

## References

[1] Amey Agrawal, Nitin Kedia, Ashish Panwar, Jayashree Mohan, Nipun Kwatra, Bhargav Gulavani, Alexey Tumanov, and Ramachandran Ramjee. Taming Throughput-Latency tradeof in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pages 117–134. USENIX Association, 2024. URL https://www.usenix.org/confere nce/osdi24/presentation/agrawal.

[2] Eduardo Alvarez, Vishal Mehta, and Farshad Ghodsian. Inside nvidia rubin gpu architecture: Powering the era of agentic ai. NVIDIA Technical Blog, July 2026. URL https://developer.nvidia.c om/blog/inside-nvidia-rubin-gpu-architecture-powering-the-era-of-agentic-ai/. Accessed: 2026-09-20.

[3] Ruicheng Ao, Jing Dong, Gan Luo, and David Simchi-Levi. Service-induced congestion in memoryconstrained LLM serving, 2026. URL https://arxiv.org/abs/2606.15555.

[4] Ruicheng Ao, Gan Luo, David Simchi-Levi, and Xinshang Wang. Optimizing LLM inference: Fluid-Guided online scheduling with memory constraints, 2026. URL https://arxiv.org/abs/2504.1 1320.

[5] Patrick Billingsley. Convergence of Probability Measures. Wiley, 2 edition, 1999.

[6] Maury Bramson. State space collapse with application to heavy trafic limits for multiclass queueing networks. Queueing Systems, 30(1–2):89–148, 1998. doi: 10.1023/A:1019160803783. URL https://doi.org/10.1023/A:1019160803783.

[7] Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020.

[8] Cagatay Buyukkoc, Pravin Varaiya, and Jean Walrand. The �� rule revisited. Advances in Applied Probability, 17(1):237–238, 1985. doi: 10.2307/1427064. URL https://doi.org/10.2307/1427 064.

[9] Zixi Chen, Yinyu Ye, and Zijie Zhou. Adaptively robust LLM inference optimization under prediction uncertainty, 2025. URL https://arxiv.org/abs/2508.14544.

[10] EG Cofman and Alexander L Stolyar. Bandwidth packing. Algorithmica, 29(1):70–88, 2001.

[11] J. G. Dai, Tianze Deng, Yueying Li, and Tianyi Peng. Throughput-optimal scheduling algorithms for LLM inference and AI agents, 2026. URL https://arxiv.org/abs/2504.07347.

[12] Yiding Feng, Siyu Liu, Zonghan Yang, and Yuhao Zhang. General non-clairvoyant KV-Cache scheduling via regime-aware routing, 2026. URL https://arxiv.org/abs/2607.09248.

[13] Yiding Feng, Zonghan Yang, and Yuhao Zhang. Competitive non-clairvoyant KV-Cache scheduling for LLM inference, 2026. URL https://arxiv.org/abs/2601.22996.

[14] David Gamarnik. Stochastic bandwidth packing process: stability conditions via lyapunov function technique. Queueing systems, 48(3):339–363, 2004.

[15] Javad Ghaderi, Yuan Zhong, and Rayadurgam Srikant. Asymptotic optimality of bestfit for stochastic bin packing. ACM SIGMETRICS Performance Evaluation Review, 42(2):64–66, 2014.

[16] GilPress. ChatGPT users statistics. WhatsTheBigData, 2024. URL https://whatsthebigdata.co m/chatgpt-users/#google\_vignette.

[17] GitHub. GitHub copilot. https://github.com/features/copilot, 2021.

[18] Cindy Gordon. ChatGPT and Generative AI innovations are creating sustainability havoc. Forbes, 2024. URL https://www.forbes.com/sites/cindygordon/2024/03/12/chatgpt-and-generativ e-ai-innovations-are-creating-sustainability-havoc/.

[19] Aaron Grattafiori et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024. doi: 10.48550/arXiv.2407.21783. URL https://arxiv.org/abs/2407.21783.

[20] Varun Gupta and Ana Radovanovic. Lagrangian-based online stochastic bin packing. ACM SIGMETRICS Performance Evaluation Review, 43(1):467–468, 2015.

[21] Chenyu Huang, Zhengyang Tang, Shixi Hu, Ruoqing Jiang, Xin Zheng, Dongdong Ge, Benyou Wang, and Zizhuo Wang. Orlm: A customizable framework in training large models for automated optimization modeling. Operations Research, 2025.

[22] Jean Jacod and Albert N. Shiryaev. Limit Theorems for Stochastic Processes. Springer, 2 edition, 2003. doi: 10.1007/978-3-662-05265-5.

[23] Patrick Jaillet, Jiashuo Jiang, Konstantina Mellou, Marco Molinaro, Chara Podimata, and Zijie Zhou. Online scheduling for LLM inference with KV cache constraints, 2025. URL https: //arxiv.org/abs/2502.07115.

[24] Li Kong, Qi Qi, Yinyu Ye, and Zijie Zhou. Geometry-aware online scheduling for LLM serving: From theoretical bound to system practice, 2026. URL https://arxiv.org/abs/2606.22327.

[25] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. Eficient memory management for large language model serving with PagedAttention. In Proceedings of the 29th Symposium on Operating Systems Principles, pages 611–626. Association for Computing Machinery, 2023. doi: 10.1145/3600006.3613165. URL https://doi.org/10.1145/3600006.3613165.

[26] Ruihan Lin, Zezhen Ding, Zean Han, and Jiheng Zhang. Large-scale LLM inference with heterogeneous workloads: Prefill–decode contention and asymptotically optimal control, 2026. URL https: //arxiv.org/abs/2602.02987.

[27] Siva Theja Maguluri, Rayadurgam Srikant, and Lei Ying. Stochastic models of load balancing and scheduling in cloud computing clusters. In 2012 Proceedings IEEE Infocom, pages 702–710. IEEE, 2012.

[28] Avishai Mandelbaum and Alexander L. Stolyar. Scheduling flexible servers with convex delay costs: Heavy-trafic optimality of the generalized ��-rule. Operations Research, 52(6):836–855, 2004. doi: 10.1287/opre.1040.0152. URL https://doi.org/10.1287/opre.1040.0152.

[29] Chengyi Nie, Nian Si, and Zijie Zhou. A queueing-theoretic framework for stability analysis of LLM inference with KV cache memory constraints. In Proceedings of the 43rd International Conference on Machine Learning, 2026. URL https://arxiv.org/abs/2605.04595.

[30] OpenAI. ChatGPT. https://chat.openai.com, 2019.

[31] OpenAI. GPT-4 technical report. arxiv 2303.08774. View in Article, 2(5), 2023.

[32] Reiner Pope, Sholto Douglas, Aakanksha Chowdhery, Jacob Devlin, James Bradbury, Jonathan Heek, Kefan Xiao, Shivani Agrawal, and Jef Dean. Eficiently scaling transformer inference. Proceedings of Machine Learning and Systems, 5:606–624, 2023.

[33] Walter Rudin. Principles ofMathematical Analysis. McGraw–Hill, 3 edition, 1976.

[34] Aaditya Singh, Adam Fry, Adam Perelman, Adam Tart, Adi Ganesh, Ahmed El-Kishky, Aidan McLaughlin, Aiden Low, AJ Ostrow, Akhila Ananthram, et al. Openai gpt-5 system card. arXiv preprint arXiv:2601.03267, 2025.

[35] Alexander L Stolyar. An infinite server system with general packing constraints. Operations Research, 61(5):1200–1217, 2013.

[36] Alexander L Stolyar and Yuan Zhong. A large-scale service system with packing constraints: Minimizing the number of occupied servers. ACM SIGMETRICS Performance Evaluation Review, 41(1):41–52, 2013.

[37] Alexander L Stolyar and Yuan Zhong. Asymptotic optimality of a greedy randomized algorithm in a large-scale service system with general packing constraints. Queueing Systems, 79(2):117–143, 2015.

[38] Jan A. Van Mieghem. Dynamic scheduling with convex delay costs: The generalized �� rule. The Annals of Applied Probability, 5(3):809–833, 1995. doi: 10.1214/aoap/1177004706. URL https://doi.org/10.1214/aoap/1177004706.

[39] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[40] Jiahao Wang, Jinbo Han, Xingda Wei, Sijie Shen, Dingyan Zhang, Chenguang Fang, Rong Chen, Wenyuan Yu, and Haibo Chen. KVCache cache in the wild: Characterizing and optimizing KVCache cache at a large cloud provider. In 2025 USENIX Annual Technical Conference (USENIX ATC 25). USENIX Association, 2025. URL https://www.usenix.org/conference/atc25/presentati on/wang-jiahao.

[41] Meixuan Wang, Yinyu Ye, and Zijie Zhou. LLM serving optimization with variable prefill and decode lengths, 2025. URL https://arxiv.org/abs/2508.06133.

[42] Ward Whitt. Stochastic-Process Limits: An Introduction to Stochastic-Process Limits and Their Application to Queues. Springer, 2002.

[43] Ward Whitt. Internet supplement to Stochastic-Process Limits: An Introduction to Stochastic-Process Limits and Their Application to Queues, 2002. URL https://www.columbia.edu/\~ww2040/supp lement.html.

[44] Ruth J. Williams. Difusion approximations for open multiclass queueing networks: Suficient conditions involving state space collapse. Queueing Systems, 30(1–2):27–88, 1998. doi: 10.1023/A:1019108819713. URL https://doi.org/10.1023/A:1019108819713.

[45] Kuang Xu. Drift method: from stochastic networks to machine learning. URL: https://web. stanford. edu/kuangxu/papers/driftmethod. pdf. Last visited on, 3(09), 2023.

[46] Heyuan Yao, Chutong Gao, Yuan Lyu, Izzy Grosof, and David Simchi-Levi. Multi-turn llm conversations under the least-recently-used policy: Mean-field asymptotics and hit ratio approximation. arXiv preprint arXiv:2609.02027, 2026.

[47] Gyeong-In Yu, Joo Seong Jeong, Geon-Woo Kim, Soojeong Kim, and Byung-Gon Chun. Orca: A distributed serving system for Transformer-Based generative models. In 16th USENIX Symposium on Operating Systems Design and Implementation (OSDI 22), pages 521–538. USENIX Association, 2022. URL https://www.usenix.org/conference/osdi22/presentation/yu.

[48] Noa Zychlinski, Carri W. Chan, and Jing Dong. Managing queues with diferent resource requirements. Operations Research, 71(4):1387–1413, 2023. doi: 10.1287/opre.2022.2284. URL https: //doi.org/10.1287/opre.2022.2284.

## Appendices

## A Non-Anticipativity of Prelimit Policies

Fix $r \geq 1$ and an initial state $z _ { 0 } ^ { r } .$ . The arrival processes are the only source of randomness in the system. Define their filtration by

$$
\mathcal { F } _ { \tau } ^ { r } : = \sigma \left( N _ { j } ^ { r } \left( r \int _ { 0 } ^ { s } \lambda _ { j } ( u ) d u \right) : j = 1 , \ldots , J , 0 \leq s \leq \tau \right) , \qquad \tau \geq 0 ,
$$

with the usual augmentation. A prelimit policy is non-anticipative if, at every GPU step �, all control increments $\Delta \mathcal { U } _ { j , k } ^ { r } ( t )$ and $\Delta \mathcal { V } _ { i , k } ^ { r } ( t )$ are measurable with respect to $\mathcal { G } _ { t } ^ { r } \mathrel { \mathop : } = \mathcal { F } _ { f ^ { r } ( t ) } ^ { r } .$

We show by induction that $f ^ { r } ( t )$ is a stopping time with respect to $\{ \mathcal { F } _ { \tau } ^ { r ^ { \prime } } : \check { \tau } \geq 0 \}$ for every $t \in \mathbb { Z } _ { + }$ . First, $f ^ { r } ( 0 ) = 0$ is a stopping time.

Suppose that $f ^ { r } ( n )$ is a stopping time for every $n = 0 , \ldots , t$ . By the non-anticipativity condition, the control increments $\Delta \mathcal { U } _ { i , k } ^ { r } ( t )$ and $\Delta \mathcal { V } _ { j , k } ^ { r } ( t )$ , over their respective prefill and decode stages, are ${ \mathcal { G } } _ { t } ^ { r }$ -measurable. The step duration $\Delta ^ { r } ( t ) \dot { { \mathbf { \psi } } }$ is a function of these increments, so $f ^ { r } ( t + 1 ) = f ^ { r } ( t ) + \Delta ^ { r } ( t )$ is also ${ \mathcal { G } } _ { t } ^ { r }$ -measurable.

For every $\tau \geq 0$ , the event $\{ \Delta ^ { r } ( t ) \leq \tau - f ^ { r } ( t ) \}$ belongs to $\mathcal { G } _ { t } ^ { r } = \mathcal { F } _ { f ^ { r } ( t ) } ^ { r }$ . By the definition of $\mathcal { F } _ { f ^ { r } ( t ) } ^ { r }$ intersecting this event with $\{ f ^ { r } ( t ) \leq \tau \}$ gives an event in $\mathcal { F } _ { \tau } ^ { r }$ . Since $\Delta ^ { r } ( t ) \geq 0$ , we obtain

$$
\{ f ^ { r } ( t + 1 ) \le \tau \} = \{ f ^ { r } ( t ) \le \tau , \ : \Delta ^ { r } ( t ) \le \tau - f ^ { r } ( t ) \} \in \mathcal { F } _ { \tau } ^ { r } .
$$

Thus, $f ^ { r } ( t + 1 )$ is a stopping time, completing the induction. Consequently, $\mathcal { G } _ { t } ^ { r } = \mathcal { F } _ { f ^ { r } ( t ) } ^ { r }$ is well-defined for every �.

## B Illustration of the Fluid �� Policy under General Arrivals

Figure 8 illustrates the policy for three classes with $c _ { 1 } \mu _ { 1 } > c _ { 2 } \mu _ { 2 } > c _ { 3 } \mu _ { 3 }$ . Here, $\chi _ { j , \mathrm { r t } } ^ { * } ( \tau )$ denotes the cumulative number of class-� completions by clock time �. During [0, 2], the class-1 backlog receives all available service, while lower-priority classes wait. During [6, 9], class-1 requests are served as they arrive, using only part of the service budget; the remainder is used to serve the class-2 backlog. During [9, 11], the policy keeps up with arrivals to both classes 1 and 2 and allocates the remaining budget to class 3.

## C Proof of Theorem 1

We first introduce the workload processes used to derive a lower bound on the cumulative holding cost.

Definition 6 (Workload processes). For an admissible policy � and each priority prefix $\{ 1 , \ldots , \ell \}$ , define the prefix workload at step time $\xi$ by

$$
W _ { \ell } ^ { \pi } ( \xi ) : = \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } p _ { j , L _ { j } } ^ { \pi } ( \xi ) , \qquad \ell = 1 , \ldots , J .\tag{42}
$$

In clock time $\tau ,$ define the net-input process $g _ { \ell }$ and the corresponding reflected workload process $\varphi _ { \ell }$ by

$$
g _ { \ell } ( \tau ) : = \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } A _ { j } ( \tau ) - \tau , \quad \quad \varphi _ { \ell } ( \tau ) : = g _ { \ell } ( \tau ) - \operatorname * { i n f } _ { 0 \leq s \leq \tau } ( g _ { \ell } ( s ) \wedge 0 ) .
$$

![](images/2bdae312718fe5af2384a5e0547ba3434a4fd95fe82eaf91472bef6aa39af0d8.jpg)  
Figure 8: Real-time representation of the optimal �� policy for three priority classes under general arrivals. The curves show the cumulative class-level completion processes, and the lower bar shows the corresponding service-efort allocation.

Remark 5 (Workload interpretation). The queue length $p _ { j , L _ { j } } ^ { \pi }$ records the number of waiting class-� requests. We use the efective service rate $\mu _ { j }$ to convert this quantity into real-time units, so $W _ { \ell } ^ { \pi }$ measures the total efective workload of waiting requests in the first ℓ classes. Applying the same conversion to cumulative arrivals, $g _ { \ell }$ describes the net workload input to these classes. Since unused service capacity cannot be saved for future arrivals, the Skorokhod reflection $\varphi _ { \ell }$ gives the remaining workload when the full service capacity is devoted to these classes whenever they have work waiting. This yields the smallest remaining workload permitted by the available capacity, and hence a lower bound on $W _ { \ell } ^ { \pi }$ at the corresponding clock time.

Proof. First, we establish a lower bound on $\mathcal { T } ( \pi ; z _ { 0 } )$ . To obtain this bound, we express the integrand in $\mathcal { T } ( \pi ; z _ { 0 } )$ as a nonnegative weighted sum of the prefix workloads. Setting $c _ { J + 1 } \mu _ { J + 1 } : = 0$ , we obtain

$$
\begin{array} { l } { { \displaystyle \sum _ { \ell = 1 } ^ { J } \left( c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } \right) W _ { \ell } ^ { \pi } ( \xi ) = \sum _ { \ell = 1 } ^ { J } \left( c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } \right) \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } p _ { j , L _ { j } } ^ { \pi } ( \xi ) } } \\ { { \displaystyle \qquad = \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } p _ { j , L _ { j } } ^ { \pi } ( \xi ) \sum _ { \ell = j } ^ { J } \left( c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } \right) } } \\ { { \displaystyle \qquad = \sum _ { j = 1 } ^ { J } c _ { j } p _ { j , L _ { j } } ^ { \pi } ( \xi ) . } } \end{array}\tag{43}
$$

It remains to show that $\varphi _ { \ell }$ provides a lower bound on the prefix workload $W _ { \ell } ^ { \pi }$ . Using the queue balance $p _ { j , L _ { i } } ^ { \pi } ( \xi ) = A _ { j } ( f ^ { \pi } ( \xi ) ) - \chi _ { j } ^ { \pi } ( \xi )$ , we first write

$$
W _ { \ell } ^ { \pi } ( \xi ) = \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } A _ { j } ( f ^ { \pi } ( \xi ) ) - \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } \chi _ { j } ^ { \pi } ( \xi ) .\tag{44}
$$

We next establish two inequalities for $\{ X _ { j } ^ { \pi } \} _ { j = 1 } ^ { J }$ . The memory constraint in (17) gives

$$
d \xi \geq \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } d X _ { j } ^ { \pi } ( \xi ) .
$$

Together with the definitions of $f ^ { \pi }$ and $\mu _ { j }$ , this yields

$$
\begin{array} { l } { { \displaystyle d f ^ { \pi } ( \xi ) = \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) d X _ { j } ^ { \pi } ( \xi ) + b d \xi } } \\ { ~ } \\ { { \displaystyle ~ \geq \sum _ { j = 1 } ^ { J } \left[ a _ { \mathrm { p f } } s _ { j } + \left( a _ { \mathrm { k v } } + \frac { b } { M } \right) \tilde { m } _ { j } \right] d X _ { j } ^ { \pi } ( \xi ) } } \\ { { \displaystyle ~ = \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } d X _ { j } ^ { \pi } ( \xi ) . } } \end{array}\tag{45}
$$

Since $X _ { i } ^ { \pi }$ is nondecreasing and $X _ { j } ^ { \pi } ( 0 ) = f ^ { \pi } ( 0 ) = 0$ , integrating over [0, �] and $[ \xi ^ { \prime } , \xi ]$ , respectively, gives, for $0 \leq \xi ^ { \prime } \leq \xi$

$$
\sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } X _ { j } ^ { \pi } ( \xi ) \leq f ^ { \pi } ( \xi ) , \quad \mathrm { a n d } \quad \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } \big [ X _ { j } ^ { \pi } ( \xi ) - X _ { j } ^ { \pi } ( \xi ^ { \prime } ) \big ] \leq f ^ { \pi } ( \xi ) - f ^ { \pi } ( \xi ^ { \prime } ) .\tag{46}
$$

Substituting the first inequality into (44), we obtain

$$
W _ { \ell } ^ { \pi } ( \xi ) \geq \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } A _ { j } ( f ^ { \pi } ( \xi ) ) - f ^ { \pi } ( \xi ) = g _ { \ell } ( f ^ { \pi } ( \xi ) ) .\tag{47}
$$

Substituting the second inequality and using $W _ { \ell } ^ { \pi } ( \xi ^ { \prime } ) \ge 0$ , we obtain

$$
\begin{array} { r l } & { W _ { \ell } ^ { \pi } ( \xi ) \geq \displaystyle \sum _ { j = 1 } ^ { \ell } \mu _ { j } ^ { - 1 } \left[ A _ { j } ( f ^ { \pi } ( \xi ) ) - X _ { j } ^ { \pi } ( \xi ^ { \prime } ) \right] - f ^ { \pi } ( \xi ) + f ^ { \pi } ( \xi ^ { \prime } ) } \\ & { \quad \quad \quad = W _ { \ell } ^ { \pi } ( \xi ^ { \prime } ) + g _ { \ell } ( f ^ { \pi } ( \xi ) ) - g _ { \ell } ( f ^ { \pi } ( \xi ^ { \prime } ) ) } \\ & { \quad \quad \geq g _ { \ell } ( f ^ { \pi } ( \xi ) ) - g _ { \ell } ( f ^ { \pi } ( \xi ^ { \prime } ) ) . } \end{array}\tag{48}
$$

Combining these two workload bounds and taking the supremum over $0 \leq \xi ^ { \prime } \leq \xi$ , we obtain

$$
\begin{array} { l } { { W _ { \ell } ^ { \pi } ( \xi ) \geq g _ { \ell } ( f ^ { \pi } ( \xi ) ) - \operatorname* { i n f } _ { 0 \leq \xi ^ { \prime } \leq \xi } \left( g _ { \ell } ( f ^ { \pi } ( \xi ^ { \prime } ) ) \wedge 0 \right) } } \\ { { \mathrm { ~ } = g _ { \ell } ( f ^ { \pi } ( \xi ) ) - \operatorname* { i n f } _ { 0 \leq s \leq f ^ { \pi } ( \xi ) } \left( g _ { \ell } ( s ) \wedge 0 \right) } } \\ { { \mathrm { ~ } = \varphi _ { \ell } ( f ^ { \pi } ( \xi ) ) , } } \end{array}\tag{49}
$$

where the change of variable $s = f ^ { \pi } ( \xi ^ { \prime } )$ uses the continuity and monotonicity of $f ^ { \pi }$ , together with $f ^ { \pi } ( 0 ) = 0$ The clock satisfies $f ^ { \pi } ( \xi ) \geq b \xi $ ∞ because $b > 0$ . Thus, changing variables to $\tau = f ^ { \pi } ( \xi )$ preserves the

integration limits 0 and ∞. Finally, using (43), (49), and $c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } \geq 0$ , we obtain

$$
\begin{array} { l l l } { \mathcal { I } ( \pi ; z _ { 0 } ) = \displaystyle \int _ { 0 } ^ { \infty } \displaystyle \int _ { - \infty } ^ { z } c _ { j } p _ { j , x _ { j } } ^ { \pi } ( \xi ) d f ^ { \pi } ( \xi ) d f ^ { \pi } ( \xi ) } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } \\ { \sum _ { \theta = \lambda } \displaystyle \int _ { 0 } ^ { \infty } \displaystyle \sum _ { \ell = 1 } ^ { \infty } ( c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } ) W _ { \ell } ^ { \pi } ( \xi ) d f ^ { \pi } ( \xi ) } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } { \sum _ { \ell = 1 } ^ { \infty } \displaystyle \int _ { t = 0 } ^ { \infty } c _ { \ell + 1 } \mu _ { \ell } \left( c _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } \right) \varphi _ { \ell } ( f ^ { \pi } ( \xi ) ) d f ^ { \pi } ( \xi ) } \\ { \displaystyle } \\ { \displaystyle } \\ { \displaystyle } { \displaystyle } \\ { \displaystyle } { \displaystyle } \\ { \displaystyle } { \int _ { t = 0 } ^ { \infty } \displaystyle \sum _ { \ell = 1 } ^ { \infty } ( c _ { \ell } \mu _ { \ell } - c _ { \ell + 1 } \mu _ { \ell + 1 } ) \varphi _ { \ell } ( \tau ) d \tau } \\ { \displaystyle } \\ { \displaystyle } { \displaystyle } \\ { \displaystyle } { \displaystyle } \\ { \displaystyle } { \displaystyle } \end{array}
$$

Next, we prove that $\pi ^ { * }$ is admissible by verifying the three conditions in Definition 2.

We first show that $F$ is a continuous strictly increasing bijection from $[ 0 , \infty )$ onto itself, so that $f ^ { * } = F ^ { - 1 }$ is well defined. Since $R _ { 0 } ^ { * }$ and $A _ { j }$ are continuous and the operations $[ \cdot ] ^ { + }$ and $\operatorname { s u p } _ { 0 \leq s \leq \tau }$ preserve continuity, each $R _ { j } ^ { * }$ defined recursively by (20) is continuous. Each $R _ { i } ^ { * }$ is also nondecreasing because it is defined using $\operatorname { s u p } _ { 0 \leq s \leq \tau }$ . The recursion also gives $R _ { j } ^ { * } ( 0 ) = 0$ . Therefore, by the definition of $F$ in (21), � is continuous with $F ( 0 ) = 0$

To establish strict monotonicity and unboundedness of $F ,$ , we use the following lemma.

Lemma 1. Let $0 \leq \tau _ { 1 } \leq \tau _ { 2 }$ . Then, for $j = 1 , \dots , J ,$

$$
R _ { j } ^ { \ast } ( \tau _ { 2 } ) - R _ { j } ^ { \ast } ( \tau _ { 1 } ) \leq R _ { j - 1 } ^ { \ast } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { \ast } ( \tau _ { 1 } ) .
$$

For $0 \leq \tau _ { 1 } < \tau _ { 2 } .$ , by the definition of � in (21), we have

$$
F ( \tau _ { 2 } ) - F ( \tau _ { 1 } ) = \sum _ { j = 1 } ^ { J } \frac { \tilde { m } _ { j } \mu _ { j } } { M } \big [ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) + R _ { j } ^ { * } ( \tau _ { 1 } ) \big ] + \frac { 1 } { b } \big [ R _ { J } ^ { * } ( \tau _ { 2 } ) - R _ { J } ^ { * } ( \tau _ { 1 } ) \big ] .
$$

By Lemma 1 and the monotonicity of $R _ { j } ^ { * }$ , all increments on the right-hand side are nonnegative. Since $b > 0$ $M > 0 , \tilde { m } _ { j } > 0$ , and $\mu _ { j } > 0$ , we can set

$$
\kappa : = \operatorname* { m i n } \left\{ \frac { 1 } { b } , \frac { \tilde { m } _ { 1 } \mu _ { 1 } } { M } , \dots , \frac { \tilde { m } _ { J } \mu _ { J } } { M } \right\} > 0 .
$$

We can therefore bound each coeficient below by �. By telescoping and using $R _ { 0 } ^ { * } ( \tau ) = \tau$ , we obtain

$$
\begin{array} { l } { { \displaystyle F ( \tau _ { 2 } ) - F ( \tau _ { 1 } ) \geq \kappa \left( \displaystyle \sum _ { j = 1 } ^ { J } \left[ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) + R _ { j } ^ { * } ( \tau _ { 1 } ) \right] + R _ { J } ^ { * } ( \tau _ { 2 } ) - R _ { J } ^ { * } ( \tau _ { 1 } ) \right) } } \\ { { \displaystyle \qquad = \kappa \left[ R _ { 0 } ^ { * } ( \tau _ { 2 } ) - R _ { 0 } ^ { * } ( \tau _ { 1 } ) \right] = \kappa ( \tau _ { 2 } - \tau _ { 1 } ) > 0 . } } \end{array}
$$

This proves that $F$ is strictly increasing. Taking $\tau _ { 1 } = 0$ and using $F ( 0 ) = 0 { \it \Omega }$ , we also obtain $F ( \tau ) \geq \kappa \tau  \infty$ Together with continuity, these properties show that � maps [0, ∞) bijectively onto itself. Consequently, $f ^ { * }$ in (22) is well defined and satisfies $F ( f ^ { \ast } ( \xi ) ) = \xi$

We next verify that each $\boldsymbol { \chi } _ { j } ^ { * }$ is continuous and nondecreasing, with $X _ { i } ^ { * } ( 0 ) = 0$ . Since $f ^ { * }$ and all $R _ { j } ^ { * }$ are continuous, $\chi _ { j } ^ { * }$ defined in $( \breve { 2 } 3 )$ is continuous. Since $f ^ { * } ( 0 ) = 0$ and $R _ { i } ^ { * } ( 0 ) = 0$ for all $j , ( 2 3 )$ also gives $\chi _ { j } ^ { * } ( 0 ) = 0$ . For $0 \stackrel { \cdot } { \leq } \xi _ { 1 } \leq \xi _ { 2 }$ , using $f ^ { * } ( \xi _ { 1 } ) \leq f ^ { * } ( \xi _ { 2 } )$ and Lemma 1, we obtain

$$
X _ { j } ^ { * } ( \xi _ { 2 } ) - X _ { j } ^ { * } ( \xi _ { 1 } ) = \mu _ { j } { \Big ( } R _ { j - 1 } ^ { * } { \big ( } f ^ { * } ( \xi _ { 2 } ) { \big ) } - R _ { j - 1 } ^ { * } { \big ( } f ^ { * } ( \xi _ { 1 } ) { \big ) } { \Big ) } - \mu _ { j } { \Big ( } R _ { j } ^ { * } { \big ( } f ^ { * } ( \xi _ { 2 } ) { \big ) } - R _ { j } ^ { * } { \big ( } f ^ { * } ( \xi _ { 1 } ) { \big ) } { \Big ) } \geq 0 .
$$

We next verify the state constraints. For the intermediate-stage states, substituting $\mathcal { U } _ { j , k } ^ { * } = \mathcal { V } _ { j , k } ^ { * } = \mathcal { X } _ { j } ^ { * }$ from Definition 4 into (9), we obtain

$$
p _ { j , k } ^ { \pi ^ { * } } ( \xi ) = p _ { j , k } ( 0 ) + \boldsymbol { X } _ { j } ^ { * } ( \xi ) - \boldsymbol { X } _ { j } ^ { * } ( \xi ) = 0 , \quad \mathrm { a n d } \quad d _ { j , k } ^ { \pi ^ { * } } ( \xi ) = d _ { j , k } ( 0 ) + \boldsymbol { X } _ { j } ^ { * } ( \xi ) - \boldsymbol { X } _ { j } ^ { * } ( \xi ) = 0 .
$$

It remains to show that the waiting-queue lengths satisfy $p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) \ge 0$

$$
p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) = A _ { j } ( f ^ { * } ( \xi ) ) - \chi _ { j } ^ { * } ( \xi ) = \mu _ { j } \left( \mu _ { j } ^ { - 1 } A _ { j } ( f ^ { * } ( \xi ) ) - R _ { j - 1 } ^ { * } ( f ^ { * } ( \xi ) ) + R _ { j } ^ { * } ( f ^ { * } ( \xi ) ) \right) .
$$

Since

$$
\begin{array} { r l } & { R _ { j } ^ { * } ( f ^ { * } ( \xi ) ) = \underset { 0 \leq s \leq f ^ { * } ( \xi ) } { \operatorname* { s u p } } \left[ R _ { j - 1 } ^ { * } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } } \\ & { \qquad \quad \geq \left[ R _ { j - 1 } ^ { * } ( f ^ { * } ( \xi ) ) - \mu _ { j } ^ { - 1 } A _ { j } ( f ^ { * } ( \xi ) ) \right] ^ { + } } \\ & { \qquad \quad \geq R _ { j - 1 } ^ { * } ( f ^ { * } ( \xi ) ) - \mu _ { j } ^ { - 1 } A _ { j } ( f ^ { * } ( \xi ) ) , } \end{array}
$$

we have $p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) \ge 0$ . We next verify the memory constraint (17). Using the identity $F ( f ^ { * } ( \xi ) ) = \xi$ , for each $\xi \ge 0$

$$
\begin{array} { l } { { \displaystyle \xi = F \big ( f ^ { * } ( \xi ) \big ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \mu _ { j } \left( R _ { j - 1 } ^ { * } \big ( f ^ { * } ( \xi ) \big ) - R _ { j } ^ { * } \big ( f ^ { * } ( \xi ) \big ) \right) + \frac { 1 } { b } R _ { J } ^ { * } \big ( f ^ { * } ( \xi ) \big ) } } \\ { { \displaystyle \qquad = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \chi _ { j } ^ { * } ( \xi ) + \frac { 1 } { b } R _ { J } ^ { * } \big ( f ^ { * } ( \xi ) \big ) . } } \end{array}\tag{50}
$$

Thus, for $0 \leq \xi _ { 1 } \leq \xi _ { 2 }$

$$
\xi _ { 2 } - \xi _ { 1 } = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \left( X _ { j } ^ { * } ( \xi _ { 2 } ) - X _ { j } ^ { * } ( \xi _ { 1 } ) \right) + \frac { 1 } { b } \left( R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 2 } ) ) - R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 1 } ) ) \right) .
$$

Equivalently,

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \left( \chi _ { j } ^ { * } ( \xi _ { 2 } ) - \chi _ { j } ^ { * } ( \xi _ { 1 } ) \right) = M ( \xi _ { 2 } - \xi _ { 1 } ) - \frac { M } { b } \left( R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 2 } ) ) - R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 1 } ) ) \right) .
$$

Since $f ^ { * }$ and $R _ { J } ^ { * }$ are nondecreasing, $R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 2 } ) ) - R _ { J } ^ { * } ( f ^ { * } ( \xi _ { 1 } ) ) \geq 0$ . Therefore, for $0 \leq \xi _ { 1 } \leq \xi _ { 2 }$

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \left( \boldsymbol { \chi } _ { j } ^ { * } ( \boldsymbol { \xi } _ { 2 } ) - \boldsymbol { \chi } _ { j } ^ { * } ( \boldsymbol { \xi } _ { 1 } ) \right) \leq M ( \xi _ { 2 } - \xi _ { 1 } ) ,
$$

which is exactly the memory constraint.

Finally, we verify that $f ^ { * }$ satisfies the clock equation (16). By telescoping and the definition of $\chi _ { j } ^ { * }$ in (23), we obtain

$$
\begin{array} { r l } { f ^ { * } ( \xi ) = \displaystyle \sum _ { j = 1 } ^ { J } \left( R _ { j - 1 } ^ { * } ( f ^ { * } ( \xi ) ) - R _ { j } ^ { * } ( f ^ { * } ( \xi ) ) \right) + R _ { J } ^ { * } ( f ^ { * } ( \xi ) ) } \\ & { = \displaystyle \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } X _ { j } ^ { * } ( \xi ) + R _ { J } ^ { * } ( f ^ { * } ( \xi ) ) } \\ & { \stackrel { ( \mathrm { a } ) } { = } \displaystyle \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } X _ { j } ^ { * } ( \xi ) + b \xi - \frac { b } { M } \displaystyle \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } X _ { j } ^ { * } ( \xi ) } \\ & { \stackrel { ( \mathrm { b } ) } { = } \displaystyle \sum _ { j = 1 } ^ { J } \left( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k w } } \tilde { m } _ { j } \right) X _ { j } ^ { * } ( \xi ) + b \xi , } \end{array}
$$

where (a) follows from (50) and (b) uses $\begin{array} { r } { \mu _ { j } ^ { - 1 } = a _ { \mathrm { p f } } s _ { j } + \left( a _ { \mathrm { k v } } + \frac { b } { M } \right) \tilde { m } _ { j } } \end{array}$ . Therefore, $\{ X _ { j } ^ { * } \} _ { j = 1 } ^ { J }$ satisfies all the required conditions. By Section 3.4, the corresponding policy $\pi ^ { * } = ( \mathcal { U } ^ { * } , \mathcal { V } ^ { * } )$ is admissible.

It remains to prove that $\pi ^ { * }$ attains the lower bound. The following lemma expresses $R _ { j } ^ { * }$ in terms of the net-input process $g _ { j }$

Lemma 2. For $j = 1 , \dots , J$ and $\tau \geq 0$

$$
R _ { j } ^ { * } ( \tau ) = - \operatorname * { i n f } _ { 0 \leq s \leq \tau } ( g _ { j } ( s ) \wedge 0 ) .
$$

We now prove $W _ { j } ^ { \pi ^ { * } } ( \xi ) = \varphi _ { j } ( f ^ { * } ( \xi ) )$ . Starting from (44) and using Lemma 2, we obtain

$$
\begin{array} { l } { { W _ { j } ^ { \pi ^ { * } } ( \xi ) = \displaystyle \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( f ^ { * } ( \xi ) ) - \displaystyle \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } \chi _ { r } ^ { * } ( \xi ) } } \\ { { \displaystyle \qquad = \sum _ { r = 1 } ^ { j } \left( \mu _ { r } ^ { - 1 } A _ { r } ( f ^ { * } ( \xi ) ) - R _ { r - 1 } ^ { * } ( f ^ { * } ( \xi ) ) + R _ { r } ^ { * } ( f ^ { * } ( \xi ) ) \right) } } \\ { { \displaystyle \qquad = \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( f ^ { * } ( \xi ) ) - f ^ { * } ( \xi ) + R _ { j } ^ { * } ( f ^ { * } ( \xi ) ) } } \\ { { \displaystyle \qquad = g _ { j } ( f ^ { * } ( \xi ) ) - \operatorname* { i n f } _ { 0 \le s \le f ^ { * } ( \xi ) } ( g _ { j } ( s ) \wedge 0 ) = \varphi _ { j } ( f ^ { * } ( \xi ) ) . } } \end{array}\tag{51}
$$

Finally, we compute the cost under this policy:

$$
\begin{array} { l } { { \displaystyle { \mathcal { J } } ( { \pmb \sigma } ^ { * } ; z _ { 0 } ) = \int _ { 0 } ^ { \infty } \displaystyle \sum _ { j = 1 } ^ { J } c _ { j } p _ { j , L _ { j } } ^ { { \pmb \sigma } ^ { * } } ( \xi ) d f ^ { * } ( \xi ) } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { \infty } \displaystyle \sum _ { j = 1 } ^ { J } \left( c _ { j } \mu _ { j } - c _ { j + 1 } \mu _ { j + 1 } \right) W _ { j } ^ { \pmb \sigma } ( \xi ) d f ^ { * } ( \xi ) } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { \infty } \displaystyle \sum _ { j = 1 } ^ { J } \left( c _ { j } \mu _ { j } - c _ { j + 1 } \mu _ { j + 1 } \right) \varphi _ { j } ( f ^ { * } ( \xi ) ) d f ^ { * } ( \xi ) } } \\ { { \displaystyle ~ = \int _ { 0 } ^ { \infty } \displaystyle \sum _ { j = 1 } ^ { J } \left( c _ { j } \mu _ { j } - c _ { j + 1 } \mu _ { j + 1 } \right) \varphi _ { j } ( \tau ) d \tau } . } \end{array}\tag{52}
$$

The policy therefore attains the lower bound derived above, proving Theorem 1.

## C.1 Proof of Lemma 1

Proof. By the definition of $R _ { j } ^ { * }$ and the monotonicity of $R _ { j - 1 } ^ { * }$ , for every $s \in [ 0 , \tau _ { 1 } ]$

$$
\begin{array} { r l r } {  { \Big [ R _ { j - 1 } ^ { * } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \Big ] ^ { + } \le \operatorname* { s u p } _ { 0 \le u \le \tau _ { 1 } } \Big [ R _ { j - 1 } ^ { * } ( u ) - \mu _ { j } ^ { - 1 } A _ { j } ( u ) \Big ] ^ { + } } } \\ & { } & { \quad = R _ { j } ^ { * } ( \tau _ { 1 } ) \le R _ { j } ^ { * } ( \tau _ { 1 } ) + \Big [ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) \Big ] . } \end{array}\tag{53}
$$

For $s \in ( \tau _ { 1 } , \tau _ { 2 } ]$ , since $R _ { j - 1 } ^ { * }$ and $A _ { j }$ are nondecreasing and $\mu _ { j } > 0$ , we have

$$
\begin{array} { r } { \Big [ R _ { j - 1 } ^ { * } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \Big ] ^ { + } \leq \Big [ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - \mu _ { j } ^ { - 1 } A _ { j } ( \tau _ { 1 } ) \Big ] ^ { + } . } \end{array}
$$

Using $[ x + y ] ^ { + } \leq x + [ y ] ^ { + }$ for $x \ge 0$ , we obtain

$$
\begin{array} { r l } & { \Big [ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - \mu _ { j } ^ { - 1 } A _ { j } ( \tau _ { 1 } ) \Big ] ^ { + } = \Big [ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) + R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) - \mu _ { j } ^ { - 1 } A _ { j } ( \tau _ { 1 } ) \Big ] ^ { + } } \\ & { \qquad \leq \big ( R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) \big ) + \Big [ R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) - \mu _ { j } ^ { - 1 } A _ { j } ( \tau _ { 1 } ) \Big ] ^ { + } } \\ & { \qquad \leq R _ { j } ^ { * } ( \tau _ { 1 } ) + \big ( R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) \big ) , } \end{array}
$$

where the last inequality follows from (53). Therefore, for $s \in [ 0 , \tau _ { 2 } ]$

$$
\begin{array} { r } { \left[ R _ { j - 1 } ^ { * } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } \leq R _ { j } ^ { * } ( \tau _ { 1 } ) + \left[ R _ { j - 1 } ^ { * } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { * } ( \tau _ { 1 } ) \right] . } \end{array}
$$

Taking the supremum of both sides over $s \in [ 0 , \tau _ { 2 } ]$ , we obtain

$$
R _ { j } ^ { \ast } ( \tau _ { 2 } ) = \operatorname* { s u p } _ { 0 \leq s \leq \tau _ { 2 } } \left[ R _ { j - 1 } ^ { \ast } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } \leq R _ { j } ^ { \ast } ( \tau _ { 1 } ) + \left[ R _ { j - 1 } ^ { \ast } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { \ast } ( \tau _ { 1 } ) \right] .
$$

Therefore, $R _ { j } ^ { \ast } ( \tau _ { 2 } ) - R _ { j } ^ { \ast } ( \tau _ { 1 } ) \leq R _ { j - 1 } ^ { \ast } ( \tau _ { 2 } ) - R _ { j - 1 } ^ { \ast } ( \tau _ { 1 } )$

## C.2 Proof of Lemma 2

Proof. We first prove, by induction on $j ,$ , that

$$
R _ { j } ^ { \ast } ( \tau ) = \operatorname* { s u p } _ { 0 \leq u \leq \tau } \left[ u - \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } ,
$$

where the sum is understood as 0 when $j = 0$ . For $j = 0 , R _ { 0 } ^ { * } ( \tau ) = \operatorname* { s u p } _ { 0 \leq u \leq \tau } [ u ] ^ { + }$ . Suppose the claim holds for $j - 1$ . Then

$$
R _ { j } ^ { * } ( \tau ) = \operatorname* { s u p } _ { 0 \leq s \leq \tau } \left[ R _ { j - 1 } ^ { * } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } = \operatorname* { s u p } _ { 0 \leq s \leq \tau } \left[ \operatorname* { s u p } _ { 0 \leq u \leq s } \left[ u - \sum _ { r = 1 } ^ { j - 1 } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } .
$$

For $0 \leq u \leq s \leq \tau$ , since $A _ { j }$ is nonnegative and nondecreasing, applying the identity $\left[ [ x ] ^ { + } - a \right] ^ { + } = [ x - a ] ^ { + }$ for $a \geq 0 \mathrm { g i v e s }$

$$
\left[ \left[ u - \sum _ { r = 1 } ^ { j - 1 } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } - \mu _ { j } ^ { - 1 } A _ { j } ( s ) \right] ^ { + } \leq \left[ u - \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } .
$$

Taking the supremum over $0 \leq u \leq s \leq \tau$ , we obtain

$$
R _ { j } ^ { \ast } ( \tau ) \leq \operatorname* { s u p } _ { 0 \leq u \leq \tau } \left[ u - \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } .
$$

Conversely, for any $0 \leq u \leq \tau$ , by the definition of $R _ { j } ^ { * }$

$$
R _ { j } ^ { * } ( \tau ) \geq \left[ R _ { j - 1 } ^ { * } ( u ) - \mu _ { j } ^ { - 1 } A _ { j } ( u ) \right] ^ { + } \geq \left[ \left[ u - \sum _ { r = 1 } ^ { j - 1 } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } - \mu _ { j } ^ { - 1 } A _ { j } ( u ) \right] ^ { + } = \left[ u - \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } .
$$

Taking the supremum over $u \in [ 0 , \tau ]$ gives the reverse inequality. Therefore,

$$
R _ { j } ^ { * } ( \tau ) = \operatorname* { s u p } _ { 0 \leq u \leq \tau } \left[ u - \sum _ { r = 1 } ^ { j } \mu _ { r } ^ { - 1 } A _ { r } ( u ) \right] ^ { + } = \operatorname* { s u p } _ { 0 \leq u \leq \tau } \left[ - g _ { j } ( u ) \right] ^ { + } = - \operatorname* { i n f } _ { 0 \leq u \leq \tau } \left( g _ { j } ( u ) \wedge 0 \right) .
$$

## D Proof of Theorem 2

Proof. We first show that the $c \mu$ policy $\pi ^ { * }$ completes all requests in finite time. By (42), the total workload is

$$
{ W _ { J } ^ { \pi ^ { * } } } ( \xi ) = \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) .
$$

Since $\mu _ { j } > 0$ and $p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) \geq 0 , W _ { J } ^ { \pi ^ { * } } ( \xi ) = 0$ if and only if $p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) = 0$ for all $j .$ . For any clock time $\tau \geq H$ all requests have arrived, so $W _ { \cal I } ^ { \pi ^ { * } } ( ( f ^ { * } ) ^ { - 1 } ( \tau ) ) = 0$ if and only if all requests have been completed by time $\tau .$ Let $\begin{array} { r } { \overline { { \tau } } : = H + \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } A _ { j } ( H ) } \end{array}$ . We will show that $W _ { J } ^ { \pi ^ { * } } \big ( ( f ^ { * } ) ^ { - 1 } ( \overline { { \tau } } ) \big ) = 0$ . Recall that, by (51),

$$
W _ { j } ^ { \pi ^ { * } } ( \xi ) = \varphi _ { j } ( f ^ { * } ( \xi ) ) , \qquad j = 1 , \dots , J , \quad \xi \geq 0 .
$$

By this identity, it sufices to prove that $\varphi _ { J } ( { \overline { { \tau } } } ) = 0$ . We do so by determining the minimum of $g _ { J }$ over $[ 0 , \overline { { \tau } } ]$ For $0 \leq s \leq H$

$$
g _ { J } ( s ) = \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } A _ { j } ( s ) - s \geq - s \geq - H .
$$

For $H \leq s \leq \overline { { \tau } }$ , the arrival truncation and the definition of $\overline { { \tau } }$ give

$$
g _ { J } ( s ) = \sum _ { j = 1 } ^ { J } \mu _ { j } ^ { - 1 } A _ { j } ( H ) - s = \overline { { { \tau } } } - H - s \geq - H ,
$$

with equality at $s = \overline { { \tau } }$ . Thus, $\mathrm { i n f } _ { 0 \le s \le \overline { { { \tau } } } } \left( g _ { J } ( s ) \wedge 0 \right) = - H$ . By the definition of $\varphi _ { J }$ , we obtain

$$
\varphi _ { J } ( \overline { { \tau } } ) = g _ { J } ( \overline { { \tau } } ) - \operatorname * { i n f } _ { 0 \leq s \leq \overline { { \tau } } } \left( g _ { J } ( s ) \wedge 0 \right) = ( - H ) - ( - H ) = 0 .
$$

Equation (51) now gives $W _ { J } ^ { \pi ^ { * } } ( ( f ^ { * } ) ^ { - 1 } ( \overline { { \tau } } ) ) = 0$ . Since $\overline { { \tau } } \geq H$ , all requests are completed by this time, so $\tau _ { \mathrm { c } } ^ { \pi ^ { * } } \leq \overline { { \tau } } < \infty$

Next, we prove that if an arbitrary admissible policy � completes all requests at $\tau _ { \mathrm { c } } ^ { \pi }$ , then $\pi ^ { * }$ also completes all requests by that time. For such a policy �, we have $W _ { J } ^ { \pi } { \big ( } ( f ^ { \pi } ) ^ { - 1 } ( \tau _ { \mathrm { c } } ^ { \pi } ) { \big ) } = 0$ . Recall that, by (49),

$$
\begin{array} { r } { W _ { J } ^ { \pi } \Big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau ) \Big ) \geq \varphi _ { J } ( \tau ) , \qquad \tau \geq 0 . } \end{array}
$$

Together with (51), this gives

$$
\begin{array} { r } { 0 = W _ { J } ^ { \pi } \Big ( ( f ^ { \pi } ) ^ { - 1 } ( \tau _ { \mathrm { c } } ^ { \pi } ) \Big ) \geq \varphi _ { J } ( \tau _ { \mathrm { c } } ^ { \pi } ) = W _ { J } ^ { \pi ^ { * } } \Big ( ( f ^ { * } ) ^ { - 1 } ( \tau _ { \mathrm { c } } ^ { \pi } ) \Big ) \geq 0 . } \end{array}
$$

Since all requests have arrived by $\tau _ { \mathrm { c } } ^ { \pi }$ , zero workload implies that $\pi ^ { * }$ has completed all requests by $\tau _ { \mathrm { c } } ^ { \pi }$ . Thus, $\tau _ { \mathrm { c } } ^ { \pi ^ { * } } \leq \tau _ { \mathrm { c } } ^ { \pi }$ , proving (29).

Finally, we prove the throughput claim. Every policy that completes all requests produces the same total number $\textstyle \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H )$ of output tokens. Therefore, when this number is positive, (28) and (29) give

$$
\mathrm { T P } ^ { \pi } ( \tau _ { c } ^ { \pi } ) = \frac { \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H ) } { \tau _ { c } ^ { \pi } } \leq \frac { \sum _ { j = 1 } ^ { J } o _ { j } A _ { j } ( H ) } { \tau _ { c } ^ { \pi ^ { * } } } = \mathrm { T P } ^ { \pi ^ { * } } ( \tau _ { c } ^ { \pi ^ { * } } ) ,
$$

which proves (30).

## E Derivation of the No-Arrival Closed Forms

Let $B _ { j } : = p _ { j , L _ { i } } ( 0 )$ for $j = 1 , \dots , J .$ Since $\lambda _ { j } ( \cdot ) \equiv 0$ , we have $A _ { j } ( \tau ) = B _ { j }$ . Substituting $A _ { j } ( s ) = B _ { j }$ into the definition of $R _ { j } ^ { * }$ in (20) gives

$$
R _ { j } ^ { * } ( \tau ) = \left[ \tau - \sum _ { \ell = 1 } ^ { j } \mu _ { \ell } ^ { - 1 } B _ { \ell } \right] ^ { + } , \qquad j = 0 , 1 , \ldots , J .\tag{54}
$$

For � $\epsilon [ \sum _ { \ell = 1 } ^ { j - 1 } \mu _ { \ell } ^ { - 1 } B _ { \ell } , \sum _ { \ell = 1 } ^ { j } \mu _ { \ell } ^ { - 1 } B _ { \ell } ]$ , we have $R _ { \ell } ^ { * } ( \tau ) = 0 \mathrm { f o r } \ell = j , . . . , J .$ . Substituting (54) into the definition of $F$ in (21) yields

$$
F ( \tau ) = \sum _ { \ell = 1 } ^ { j - 1 } \frac { \tilde { m } _ { \ell } B _ { \ell } } { M } + \frac { \tilde { m } _ { j } \mu _ { j } } { M } \left( \tau - \sum _ { \ell = 1 } ^ { j - 1 } \mu _ { \ell } ^ { - 1 } B _ { \ell } \right) .
$$

On this interval, substituting (54) into the definition of $\chi _ { j } ^ { * }$ in (23) gives

$$
X _ { j } ^ { * } ( F ( \tau ) ) = \mu _ { j } \left( R _ { j - 1 } ^ { * } ( \tau ) - R _ { j } ^ { * } ( \tau ) \right) = \mu _ { j } \left( \tau - \sum _ { \ell = 1 } ^ { j - 1 } \mu _ { \ell } ^ { - 1 } B _ { \ell } \right) = \frac { M } { \tilde { m } _ { j } } \left( F ( \tau ) - \sum _ { \ell = 1 } ^ { j - 1 } \frac { \tilde { m } _ { \ell } B _ { \ell } } { M } \right) ,
$$

where the last equality uses the expression for �( ) $F ( \tau )$ above. Since � is continuous and strictly increasing, $\xi = F ( \tau )$ ranges over $\begin{array} { r } { [ \sum _ { \ell = 1 } ^ { j - 1 } \frac { \tilde { m } _ { \ell } B _ { \ell } } { M } , \sum _ { \ell = 1 } ^ { j } \frac { \tilde { m } _ { \ell } B _ { \ell } } { M } ] } \end{array}$ on the above � interval. By (54), $X _ { j } ^ { * } ( \xi ) = 0$ before this step-time interval and $X _ { j } ^ { * } ( \xi ) = B _ { j }$ after it. Hence, for all $\xi \ge 0$

$$
\chi _ { j } ^ { * } ( \xi ) = \left[ \frac { M } { \tilde { m } _ { j } } \left( \xi - \sum _ { \ell = 1 } ^ { j - 1 } \frac { \tilde { m } _ { \ell } B _ { \ell } } { M } \right) \right] _ { 0 } ^ { B _ { j } } .
$$

Using $B _ { j } = p _ { j , L _ { j } } ( 0 )$ gives (25).

## F Proof of Proposition 1

Proof. According to Definition $5 , \textstyle \sum _ { u < t } \eta _ { j } ^ { r } ( u )$ and $\textstyle \sum _ { u < t } y _ { i } ^ { r } ( u )$ are determined by the history available at step �. The term $\begin{array} { r } { \sum _ { \ell < j } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) } \end{array}$ involves only higher-priority admission plans, which have already been computed using the same history. Thus, by recursion over the priority classes, $y _ { j } ^ { r } ( t )$ is ${ \mathcal { G } } _ { t } ^ { r }$ -measurable and hence non-anticipative. Since the moving average and time shifts in (34)–(36) use only current and past values, all control increments $\Delta \mathcal { U } _ { j , k } ^ { r } ( t )$ and $\Delta \mathcal { V } _ { j , k } ^ { r } ( t )$ are also ${ \mathcal { G } } _ { t } ^ { r }$ -measurable, so the policy is non-anticipative.

We next prove by induction on � that the cumulative planned admissions of class $j$ through step � do not exceed its initial queue length plus cumulative arrivals before step �:

$$
\sum _ { u = 0 } ^ { t } y _ { j } ^ { r } ( u ) \leq p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) , \qquad t \in \mathbb { Z } _ { + } .\tag{55}
$$

For $t = 0$ , the planning rule (33) gives $0 \leq y _ { j } ^ { r } ( 0 ) \leq p _ { j , L _ { i } } ^ { r } ( 0 )$ . Suppose that (55) holds at $t - 1$ . The induction hypothesis and the nonnegativity of arrivals give

$$
p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) - \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) = \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 2 } \eta _ { j } ^ { r } ( u ) - \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) \right] + \eta _ { j } ^ { r } ( t - 1 ) \ge \eta _ { j } ^ { r } ( t - 1 ) \ge 0 .
$$

Since this quantity is nonnegative, the first argument of the minimum in the planning rule (33) gives

$$
y _ { j } ^ { r } ( t ) \leq \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) - \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) \right] ^ { + } = p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) - \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) .
$$

Consequently,

$$
\begin{array} { r l } & { \displaystyle \sum _ { u = 0 } ^ { t } y _ { j } ^ { r } ( u ) = \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) + y _ { j } ^ { r } ( t ) } \\ & { \qquad \le \displaystyle \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) + \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \displaystyle \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) - \displaystyle \sum _ { u = 0 } ^ { t - 1 } y _ { j } ^ { r } ( u ) \right] = p _ { j , L _ { j } } ^ { r } ( 0 ) + \displaystyle \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) . } \end{array}
$$

This completes the induction.

Summing the admission increments in (34) and setting $u = s - \nu$ gives

$$
\begin{array} { r l } & { \mathcal { U } _ { j , L _ { j } } ^ { r } ( t + 1 ) = \displaystyle \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } \sum _ { s = 0 } ^ { t } y _ { j } ^ { r } ( s - \nu ) = \displaystyle \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } \sum _ { u = 0 } ^ { t - \nu } y _ { j } ^ { r } ( u ) } \\ & { \qquad \leq \left( \displaystyle \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } 1 \right) \displaystyle \sum _ { u = 0 } ^ { t } y _ { j } ^ { r } ( u ) = \displaystyle \sum _ { u = 0 } ^ { t } y _ { j } ^ { r } ( u ) \leq p _ { j , L _ { j } } ^ { r } ( 0 ) + \displaystyle \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) , } \end{array}
$$

where $y _ { j } ^ { r } ( u ) = 0$ for $u < 0$ and sums with an upper limit below zero are empty. Subtracting $\mathcal { U } _ { j , L _ { j } } ^ { r } ( t )$ from both sides of the preceding inequality and using the waiting-queue balance yields

$$
0 \leq \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) \leq p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { t - 1 } \eta _ { j } ^ { r } ( u ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) = p _ { j , L _ { j } } ^ { r } ( t ) .
$$

Next, we verify that the service increment at each intermediate stage does not exceed its queue length. For a prefill stage $k < L _ { j }$ , the queueing dynamics give

$$
p _ { j , k } ^ { r } ( u + 1 ) - p _ { j , k } ^ { r } ( u ) = \Delta \mathcal { U } _ { j , k + 1 } ^ { r } ( u ) - \Delta \mathcal { U } _ { j , k } ^ { r } ( u ) .
$$

Summing over $u = 0 , \ldots , t - 1$ and telescoping the left-hand side, we obtain

$$
p _ { j , k } ^ { r } ( t ) - p _ { j , k } ^ { r } ( 0 ) = \sum _ { u = 0 } ^ { t - 1 } \left[ p _ { j , k } ^ { r } ( u + 1 ) - p _ { j , k } ^ { r } ( u ) \right] = \sum _ { u = 0 } ^ { t - 1 } \left[ \Delta \mathcal { U } _ { j , k + 1 } ^ { r } ( u ) - \Delta \mathcal { U } _ { j , k } ^ { r } ( u ) \right] .
$$

Since the intermediate stages are initially empty, $p _ { j , k } ^ { r } ( 0 ) = 0$ . Substituting the stage shifts in (35) gives

$$
\begin{array} { r l } & { \boldsymbol { p } _ { j , \boldsymbol { k } } ^ { r } ( t ) = \displaystyle \sum _ { u = 0 } ^ { t - 1 } \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( u - ( L _ { j } - \boldsymbol { k } - 1 ) ) - \sum _ { u = 0 } ^ { t - 1 } \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( u - ( L _ { j } - \boldsymbol { k } ) ) } \\ & { \qquad = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t - ( L _ { j } - \boldsymbol { k } ) ) = \Delta \mathcal { U } _ { j , \boldsymbol { k } } ^ { r } ( t ) . } \end{array}
$$

The same calculation for the first decode stage and the remaining decode stages gives

$$
\begin{array} { l } { d _ { j , o _ { j } - 1 } ^ { r } ( t ) = \displaystyle \sum _ { u = 0 } ^ { t - 1 } \left[ \Delta \mathcal { U } _ { j , 1 } ^ { r } ( u ) - \Delta \mathcal { V } _ { j , o _ { j } - 1 } ^ { r } ( u ) \right] = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t - L _ { j } ) = \Delta \mathcal { V } _ { j , o _ { j } - 1 } ^ { r } ( t ) , \quad o _ { j } \geq 2 , } \\ { d _ { j , k } ^ { r } ( t ) = \displaystyle \sum _ { u = 0 } ^ { t - 1 } \left[ \Delta \mathcal { V } _ { j , k + 1 } ^ { r } ( u ) - \Delta \mathcal { V } _ { j , k } ^ { r } ( u ) \right] = \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } \big ( t - ( L _ { j } + o _ { j } - 1 - k ) \big ) = \Delta \mathcal { V } _ { j , k } ^ { r } ( t ) , \quad k = 1 , \dots , o _ { j } - 2 . } \end{array}
$$

Thus each intermediate stage serves exactly the requests available there. Together with the admission bound proved above, this verifies the queue-length bounds on all stage-completion increments and the nonnegativity of all queueing states.

We now prove that the memory constraint holds. Induction over the priority classes in the planning rule (33) gives, pathwise,

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } y _ { j } ^ { r } ( t ) \leq \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } M .\tag{56}
$$

Substituting the moving-average admission formula in (34) into the expression for $m ^ { r } ( t )$ in (32) gives

$$
m ^ { r } ( t ) = \sum _ { j = 1 } ^ { J } \sum _ { h = 0 } ^ { H _ { \mathrm { k v } } } k _ { j } ^ { \mathrm { k v } } ( h ) \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t - h ) = \frac { 1 } { w _ { r } } \sum _ { j = 1 } ^ { J } \sum _ { h = 0 } ^ { H _ { \mathrm { k v } } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } k _ { j } ^ { \mathrm { k v } } ( h ) y _ { j } ^ { r } ( t - h - \nu ) .
$$

Making the change of variable $u = h + \nu$ , we obtain

$$
m ^ { r } ( t ) = \frac { 1 } { w _ { r } } \sum _ { j = 1 } ^ { J } \sum _ { h = 0 } ^ { H _ { \mathrm { k v } } } \sum _ { u = h } ^ { h + w _ { r } - 1 } k _ { j } ^ { \mathrm { k v } } ( h ) y _ { j } ^ { r } ( t - u ) .
$$

To exchange the sums over ℎ and �, note that the full range of � is $0 \leq u \leq w _ { r } + H _ { \mathrm { k v } } - 1$ . For a fixed �, the admissible values of ℎ are exactly those satisfying $0 \leq h \leq H _ { \mathrm { k v } }$ and $0 \leq u - h \leq w _ { r } - 1$ . After exchanging the order of summation, we factor $y _ { j } ^ { r } ( t - u )$ out of the sum over $h ,$ , since it does not depend on ℎ. This gives

$$
\begin{array} { r l } & { m ^ { r } ( t ) = \displaystyle \frac { 1 } { w _ { r } } \sum _ { u = 0 } ^ { w _ { r } + H _ { \mathrm { k v } } - 1 } \sum _ { j = 1 } ^ { J } y _ { j } ^ { r } ( t - u ) \sum _ { 0 \le h \le H _ { \mathrm { k v } } } k _ { j } ^ { \mathrm { k v } } ( h ) } \\ & { \qquad \le \displaystyle \frac 1 { w _ { r } } \sum _ { u = 0 } ^ { w _ { r } + H _ { \mathrm { k v } } - 1 } \sum _ { j = 1 } ^ { J } \sum _ { m _ { j } \mathrm { y } _ { j } ^ { r } ( t - u ) } } \\ & { \qquad \le \displaystyle \frac { w _ { r } + H _ { \mathrm { k v } } } { w _ { r } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } M = M . } \end{array}
$$

This establishes the admissibility of the policy.

## G Proof of Theorem 3

We first show that the moving average and the fixed stage delays produce vanishing errors under fluid scaling, so it sufices to analyze the cumulative planned-admission processes and the clock approximation constructed from them (Lemma 3). We then establish joint �-tightness and show that every subsequential limit satisfies the fluid model’s queue nonnegativity constraints, memory constraint, and clock equation (Lemma 4). The key step is to recover the priority rule in the limit: on any interval where a class’s waiting-queue length is positive, its priority prefix uses the full capacity and the corresponding residual service budget remains constant. This yields the reflection representation that identifies every subsequential limit with the fluid $c \mu$ policy (Lemma 5).

We begin by defining the auxiliary processes used throughout the proof.

Definition 7 (Auxiliary processes). For $N \in \mathbb { Z } _ { + }$ , define the cumulative planned-admission process by

$$
P _ { j } ^ { r } ( N ) : = \sum _ { t = 0 } ^ { N - 1 } y _ { j } ^ { r } ( t ) , \qquad P _ { j } ^ { r } ( 0 ) = 0 .\tag{57}
$$

For $\xi \ge 0$ , define its fluid-scaled version by

$$
\bar { P } _ { j } ^ { r } ( \xi ) : = \frac 1 r P _ { j } ^ { r } ( \lfloor r \xi \rfloor ) .
$$

Let $n = \lfloor r \xi \rfloor$ . Define the linear interpolation $\widehat { P } _ { j } ^ { r }$ of $\bar { P } _ { j } ^ { r }$ and use it to construct a clock approximation ${ \widehat { f } } ^ { r } ( \xi )$ of $\bar { f } ^ { r } ( \xi )$ as follows:

$$
\widehat { P } _ { j } ^ { r } ( \xi ) : = \bar { P } _ { j } ^ { r } ( \xi ) + \frac { r \xi - n } { r } y _ { j } ^ { r } ( n ) ,\tag{58}
$$

$$
{ \widehat { f } } ^ { r } ( \xi ) : = \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } { \tilde { m } } _ { j } ) { \widehat { P } } _ { j } ^ { r } ( \xi ) + b \xi .\tag{59}
$$

The continuous approximation of the fluid-scaled cumulative arrival process $\bar { A } _ { j } ^ { r }$ is defined by

$$
\widehat { A } _ { j } ^ { r } ( \xi ) : = A _ { j } ( \widehat { f } ^ { r } ( \xi ) ) = p _ { j , L _ { j } } ( 0 ) + \int _ { 0 } ^ { \widehat { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s .\tag{60}
$$

Lemma 3. Under the initial condition of Theorem 3, consider the smoothed ��-inspired policy in Definition 5.   
Then thefollowing statements hold.

(i) For every scale �, class �, and integer time �,

$$
0 \leq y _ { j } ^ { r } ( t ) \leq \frac { M } { \tilde { m } _ { j } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } \leq \frac { M } { \tilde { m } _ { j } } ,\tag{61}
$$

$$
0 \leq \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) \leq \frac { M } { \tilde { m } _ { j } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } \leq \frac { M } { \tilde { m } _ { j } } .\tag{62}
$$

Moreover,

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } y _ { j } ^ { r } ( t ) \leq \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } .\tag{63}
$$

(ii) For every class � and every $T < \infty$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \bar { P } _ { j } ^ { r } ( \xi ) - \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor )  \leq \frac { M } { 2 \tilde { m } _ { j } } \frac { w _ { r } } { r }  0 .\tag{64}
$$

(iii) For every class � and every $T < \infty ,$ , thefollowing bounds holdfor $k = 1 , \ldots , L _ { j }$ and $k = 1 , \ldots , o _ { j } - 1$ respectively:

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { \mathcal { U } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) } { r } - \frac { \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) } { r } \right| \leq \frac { H _ { \mathrm { k v } } M } { r \tilde { m } _ { j } } \longrightarrow 0 ,\tag{65}
$$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { \mathcal { V } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) } { r } - \frac { \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) } { r } \right| \leq \frac { H _ { \mathrm { k v } } M } { r \tilde { m } _ { j } } \longrightarrow 0 .\tag{66}
$$

(iv) For every $T < \infty$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \bar { f } ^ { r } ( \xi ) - \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \bar { P } _ { j } ^ { r } ( \xi ) - b \xi \right| \longrightarrow 0 .\tag{67}
$$

(v) For every class �, every fixed $T < \infty _ { : }$ , and all $\xi _ { 1 } , \xi _ { 2 } \in [ 0 , T ]$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \widehat { P } _ { j } ^ { r } ( \xi ) - \bar { P } _ { j } ^ { r } ( \xi ) \right| \leq \frac { M } { r \tilde { m } _ { j } } \longrightarrow 0 , \qquad a n d \qquad \left| \widehat { P } _ { j } ^ { r } ( \xi _ { 2 } ) - \widehat { P } _ { j } ^ { r } ( \xi _ { 1 } ) \right| \leq \frac { M } { \tilde { m } _ { j } } | \xi _ { 2 } - \xi _ { 1 } | .\tag{68}
$$

Moreover, define

$$
L _ { f } : = b + M \sum _ { j = 1 } ^ { J } \left( \frac { a _ { \mathrm { p f } } s _ { j } } { \tilde { m } _ { j } } + a _ { \mathrm { k v } } \right) .\tag{69}
$$

Then ${ \widehat { f } } ^ { r }$ is $L _ { f ^ { - } } L$ ipschitz and

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left. \bar { f } ^ { r } ( \xi ) - \widehat { f } ^ { r } ( \xi ) \right. \longrightarrow 0 .\tag{70}
$$

(vi) For every class � and every fixed $T < \infty$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \bar { A } _ { j } ^ { r } ( \xi ) - \widehat { A } _ { j } ^ { r } ( \xi ) | \overset { \mathbb { P } } {  } 0 .\tag{71}
$$

Remark 6 (Interpretation of the prelimit estimates).

(i) Equations (61) and (62) give memory-based upper bounds on both the class-� planned-admission amount $y _ { j } ^ { r } ( t )$ and the admission increment $\Delta \mathcal { U } _ { j , L _ { i } } ^ { r } ( t )$ at each step �. Equation (63) states that the plans for all classes share one memory-based planning budget.

(ii) Equation (64) shows that the diference between the fluid-scaled cumulative planned admissions $\bar { P } _ { j } ^ { r } ( \xi )$ and the fluid-scaled cumulative admissions $\mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) / r$ vanishes uniformly on compact step-time intervals. Thus, spreading each plan over $w _ { r }$ steps has a negligible efect on cumulative admissions at the fluid scale.

(iii) The prefill-stage delay bound (65) and the decode-stage delay bound (66) show that the fixed delays between admission and later stages disappear under fluid scaling. The fluid-scaled cumulative stagecompletion processes $\mathcal { U } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) / r$ and $\mathcal { V } _ { i , k } ^ { r } ( \lfloor r \xi \rfloor ) / r$ are therefore asymptotically equivalent to the fluid-scaled cumulative admission process $\ddot { \mathcal { U } } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) / r$

(iv) The clock approximation in (67) uses the fluid-scaled cumulative planned admissions $\bar { P } _ { j } ^ { r } ( \xi )$ to approximate the real-time clock $\bar { f } ^ { r } ( \xi )$

(v) Equations (68) and (70) show that the linear interpolation $\widehat { P } _ { j } ^ { r } ( \xi )$ and the fluid-scaled cumulative planned-admission process $\bar { P } _ { j } ^ { r } ( \xi )$ , as well as ${ \widehat { f } } ^ { r } ( \xi )$ and $\bar { f } ^ { r } ( \xi )$ , are asymptotically equivalent uniformly on compact intervals. Moreover, $\widehat { P } _ { j } ^ { r }$ and ${ \widehat { f } } ^ { r }$ are Lipschitz with constants $M / \tilde { m } _ { j }$ and $L _ { f }$ , respectively, independent of $r .$

(vi) Equation (71) shows that the diference between the fluid-scaled cumulative arrival process $\bar { A } _ { j } ^ { r }$ and its continuous approximation $\widehat { A } _ { j } ^ { r }$ vanishes uniformly on compact intervals in probability.

Proof. (i) By the policy definition,

$$
0 \leq \tilde { m } _ { j } y _ { j } ^ { r } ( t ) \leq \left[ \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \right] ^ { + } \leq \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } .
$$

Equation (34) expresses the admission increments as the moving average of the planned-admission amounts, which are zero at negative times. Hence,

$$
0 \leq \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) = \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } y _ { j } ^ { r } ( t - \nu ) \leq \frac { M } { \tilde { m } _ { j } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } \leq \frac { M } { \tilde { m } _ { j } } .
$$

The planning-budget bound follows by induction over the priority classes: starting from zero, each class uses at most the budget remaining after the preceding classes. Specifically, if the budget holds through class $j - 1$ , its remaining budget is nonnegative, and

$$
\begin{array} { r l } & { \displaystyle \sum _ { \ell = 1 } ^ { j } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) = \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) + \tilde { m } _ { j } y _ { j } ^ { r } ( t ) } \\ & { \qquad \le \displaystyle \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) + \left[ \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \right] ^ { + } } \\ & { \qquad = \displaystyle \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } . } \end{array}
$$

Taking $j = J$ proves (63).

(ii) Fix $N \in \mathbb { Z } _ { + }$ . Changing variables $t = s - \nu$ in the inner sum, and using $y _ { j } ^ { r } ( t ) = 0$ for $t < 0$ , gives

$$
\mathcal { U } _ { j , L _ { j } } ^ { r } ( N ) = \sum _ { s = 0 } ^ { N - 1 } \frac { 1 } { w _ { r } } \sum _ { \nu = 0 } ^ { w _ { r } - 1 } y _ { j } ^ { r } ( s - \nu ) = \frac { 1 } { w _ { r } } \sum _ { s = 0 } ^ { N - 1 } \sum _ { \substack { t = s - w _ { r } + 1 } } ^ { s } y _ { j } ^ { r } ( t ) = \frac { 1 } { w _ { r } } \sum _ { s = 0 } ^ { N - 1 } \sum _ { \substack { t = \mathrm { m a x } \{ 0 , s - w _ { r } + 1 \} } } ^ { s } y _ { j } ^ { r } ( t ) .
$$

For each fixed $0 \leq t \leq N - 1$ , the admissible values of � satisfy $t \leq s \leq \operatorname* { m i n } \{ N - 1 , t + w _ { r } - 1 \}$ . Exchanging the order of summation therefore yields

$$
\mathcal { U } _ { j , L _ { j } } ^ { r } ( N ) = \frac { 1 } { w _ { r } } \sum _ { t = 0 } ^ { N - 1 } \sum _ { s = t } ^ { \operatorname* { m i n } \{ N - 1 , t + w _ { r } - 1 \} } y _ { j } ^ { r } ( t ) = \frac { 1 } { w _ { r } } \sum _ { t = 0 } ^ { N - 1 } \operatorname* { m i n } \{ w _ { r } , N - t \} y _ { j } ^ { r } ( t ) = \sum _ { t = 0 } ^ { N - 1 } \operatorname* { m i n } \left\{ 1 , \frac { N - t } { w _ { r } } \right\} y _ { j } ^ { r } ( t ) .
$$

Subtracting this expression from $P _ { j } ^ { r } ( N )$ , and setting $\nu = N - t$ in the last line, gives

$$
\begin{array} { l } { P _ { j } ^ { r } ( N ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( N ) = \displaystyle \sum _ { t = 0 } ^ { N - 1 } \left[ 1 - \operatorname* { m i n } \left\{ 1 , \frac { N - t } { w _ { r } } \right\} \right] y _ { j } ^ { r } ( t ) } \\ { = \displaystyle \sum _ { t = \operatorname* { m a x } \{ 0 , N - w _ { r } + 1 \} } ^ { N - 1 } \left( 1 - \frac { N - t } { w _ { r } } \right) y _ { j } ^ { r } ( t ) } \\ { = \displaystyle \frac { 1 } { w _ { r } } \operatorname* { m i n } \{ w _ { r } - 1 , N \} } \end{array}
$$

By (i),

$$
0 \leq P _ { j } ^ { r } ( N ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( N ) \leq \frac { 1 } { w _ { r } } \sum _ { \nu = 1 } ^ { w _ { r } - 1 } ( w _ { r } - \nu ) \frac { M } { \tilde { m } _ { j } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } = \frac { M } { \tilde { m } _ { j } } \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } \frac { w _ { r } - 1 } { 2 } .
$$

Set $N = \lfloor r \xi \rfloor$ , divide by $r ,$ and take the supremum over $0 \le \xi \le T$ . Since $w _ { r } / r \to 0$ , this gives (64). (iii) For any integers $N \geq 0$ and $0 \leq h \leq H _ { \mathrm { k v } } .$ , (i) gives

$$
0 \leq \mathcal { U } _ { j , L _ { j } } ^ { r } ( N ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( [ N - h ] ^ { + } ) = \sum _ { \iota = [ N - h ] ^ { + } } ^ { N - 1 } \Delta \mathcal { U } _ { j , L _ { j } } ^ { r } ( t ) \leq \left[ N - [ N - h ] ^ { + } \right] \frac { M } { \tilde { m } _ { j } } \leq h \frac { M } { \tilde { m } _ { j } } \leq H _ { \mathrm { k v } } \frac { M } { \tilde { m } _ { j } } .
$$

The stage-shift relations (35) and (36), together with zero initial cumulative controls and the convention of zero increments at negative times, give

$$
\begin{array} { r l } & { \mathcal { U } _ { j , k } ^ { r } ( N ) = \mathcal { U } _ { j , L _ { j } } ^ { r } \big ( [ N - ( L _ { j } - k ) ] ^ { + } \big ) , \qquad 1 \leq k \leq L _ { j } , } \\ & { \mathcal { V } _ { j , k } ^ { r } ( N ) = \mathcal { U } _ { j , L _ { j } } ^ { r } \big ( [ N - ( L _ { j } + o _ { j } - 1 - k ) ] ^ { + } \big ) , \qquad 1 \leq k \leq o _ { j } - 1 . } \end{array}
$$

Apply the preceding bound with $h = L _ { j } - k$ and $h = L _ { j } + o _ { j } - 1 - k$ , respectively; both delays lie between zero and $H _ { \mathrm { k v } }$ . Setting $N = \lfloor r \xi \rfloor$ , dividing by $r ,$ and taking the supremum over $0 \le \xi \le T$ proves the two stated bounds.

(iv) By the definitions of $\Delta ^ { r }$ and $f ^ { r }$ , for $N = \lfloor r \xi \rfloor$

$$
f ^ { r } ( N ) = \sum _ { t = 0 } ^ { N - 1 } \Delta ^ { r } ( t ) = a _ { \mathrm { p f } } ^ { r } \sum _ { t = 0 } ^ { N - 1 } n _ { \mathrm { p f } } ^ { r } ( t ) + a _ { \mathrm { k v } } ^ { r } \sum _ { t = 0 } ^ { N - 1 } n _ { \mathrm { k v } } ^ { r } ( t ) + b ^ { r } N .
$$

Substituting the expressions for $n _ { \mathrm { p f } } ^ { r } ( t )$ and $n _ { \mathrm { k v } } ^ { r } ( t )$ yields

$$
\begin{array} { l } { { \displaystyle f ^ { r } ( N ) = a _ { \mathrm { p f } } ^ { r } \sum _ { j = 1 } ^ { J } \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } \mathcal { U } _ { j , k } ^ { r } ( N ) + a _ { \mathrm { k v } } ^ { r } \sum _ { j = 1 } ^ { J } \sum _ { k = 1 } ^ { L _ { j } } \left[ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } \right] \mathcal { U } _ { j , k } ^ { r } ( N ) } } \\ { { + a _ { \mathrm { k v } } ^ { r } \sum _ { j = 1 } ^ { J } \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) \mathcal { V } _ { j , k } ^ { r } ( N ) + b ^ { r } N . } } \end{array}
$$

Combining (ii) and (iii) by the triangle inequality gives

$$
\left| \mathcal U _ { j , k } ^ { r } ( N ) - P _ { j } ^ { r } ( N ) \right| \leq \left| \mathcal U _ { j , k } ^ { r } ( N ) - \mathcal U _ { j , L _ { j } } ^ { r } ( N ) \right| + \left| \mathcal U _ { j , L _ { j } } ^ { r } ( N ) - P _ { j } ^ { r } ( N ) \right| \leq \frac { M } { \tilde { m } _ { j } } \left( H _ { \mathrm { k v } } + \frac { w _ { r } } { 2 } \right) , \quad 1 \leq k \leq L _ { j } ,
$$

Using $\begin{array} { r } { \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } = s _ { j } } \end{array}$ and the identity for $\tilde { m } _ { j }$ in (31), we can write the clock diference as

$$
\begin{array} { l } { \displaystyle \mathcal { f } ^ { r } ( N ) - \sum _ { j = 1 } ^ { J } ( a _ { \mathtt { p f } } ^ { r } s _ { j } + a _ { \mathtt { k v } } ^ { r } \tilde { m } _ { j } ) P _ { j } ^ { r } ( N ) - b ^ { r } N } \\ { = \sum _ { j = 1 } ^ { J } \Bigg ( a _ { \mathtt { p f } } ^ { r } \displaystyle \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } \big [ \mathcal { U } _ { j , k } ^ { r } ( N ) - P _ { j } ^ { r } ( N ) \big ] + a _ { \mathtt { k v } } ^ { r } \displaystyle \sum _ { k = 1 } ^ { L _ { j } } \big [ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } \big ] \big [ \mathcal { U } _ { j , k } ^ { r } ( N ) - P _ { j } ^ { r } ( N ) \big ] } \\ { \displaystyle \qquad + a _ { \mathtt { k v } } ^ { r } \displaystyle \sum _ { k = 1 } ^ { \infty } ( s _ { j } + o _ { j } - k ) \big [ \mathcal { V } _ { j , k } ^ { r } ( N ) - P _ { j } ^ { r } ( N ) \big ] \Bigg ) . } \end{array}
$$

Applying the two bounds above to these diferences yields

$$
\begin{array} { r l } & { \displaystyle \operatorname* { s u p } _ { 0 \leq \xi \leq T } | f ^ { r } ( N ) - \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } ^ { r } s _ { j } + a _ { \mathrm { k v } } ^ { r } \tilde { m } _ { j } ) P _ { j } ^ { r } ( N ) - b ^ { r } N | } \\ & { \displaystyle \leq M \Big ( H _ { \mathrm { k v } } + \frac { w _ { r } } { 2 } \Big ) \sum _ { j = 1 } ^ { J } \frac { 1 } { \tilde { m } _ { j } } [ a _ { \mathrm { p f } } ^ { r } \sum _ { k = 1 } ^ { L _ { j } } \ell _ { j , k } + a _ { \mathrm { k v } } ^ { r } \Biggl \langle \sum _ { k = 1 } ^ { L _ { j } } [ c _ { \mathrm { c h u n k } } ( L _ { j } - k ) + \ell _ { j , k } ] + \sum _ { k = 1 } ^ { o _ { j } - 1 } ( s _ { j } + o _ { j } - k ) ) ] } \\ & { \displaystyle = M \Big ( H _ { \mathrm { k v } } + \frac { w _ { r } } { 2 } \Big ) \sum _ { j = 1 } ^ { J } \Big ( \frac { a _ { \mathrm { p f } } ^ { r } s _ { j } } { \tilde { m } _ { j } } + a _ { \mathrm { k v } } ^ { r } \Big ) \longrightarrow 0 . } \end{array}
$$

The convergence follows because $r a _ { \mathrm { p f } } ^ { r }$ and $r a _ { \mathrm { k v } } ^ { r }$ are bounded by the scaling assumptions and $( H _ { \mathrm { k v } } + w _ { r } ) / r \to 0$ Moreover, (i) implies $0 \leq \bar { P } _ { j } ^ { r } ( \xi ) \leq \dot { M } T / \tilde { m } _ { j }$ for $0 \le \xi \le T$ . Since $r a _ { \mathrm { p f } } ^ { r }  a _ { \mathrm { p f } }$ , we have

$$
\operatorname* { s u p } _ { 0 \le \xi \le T } \left| ( r a _ { \mathrm { p f } } ^ { r } - a _ { \mathrm { p f } } ) \bar { P } _ { j } ^ { r } ( \xi ) \right| \le | r a _ { \mathrm { p f } } ^ { r } - a _ { \mathrm { p f } } | \frac { M T } { \tilde { m } _ { j } } \longrightarrow 0 .
$$

Similarly, $r a _ { \mathrm { k v } } ^ { r }  a _ { \mathrm { k } }$ <sub>v</sub> gives

$$
\operatorname* { s u p } _ { 0 \le \xi \le T } \left| ( r a _ { \mathrm { k v } } ^ { r } - a _ { \mathrm { k v } } ) \bar { P } _ { j } ^ { r } ( \xi ) \right| \le | r a _ { \mathrm { k v } } ^ { r } - a _ { \mathrm { k v } } | \frac { M T } { \tilde { m } _ { j } } \longrightarrow 0 .
$$

For the overhead term, $N / r \le T$ and $| N / r - \xi | \leq 1 / r$ imply

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | b ^ { r } N - b \xi | = \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| ( r b ^ { r } - b ) \frac { N } { r } + b \left( \frac { N } { r } - \xi \right) \right| \leq T | r b ^ { r } - b | + \frac { b } { r } \longrightarrow 0 ,
$$

where the convergence follows from $r b ^ { r } \to b$ . Finally, combining the preceding estimates by the triangle inequality gives

$$
\begin{array} { r l } & { \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left| \bar { f } ^ { r } ( \xi ) - \displaystyle \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \bar { P } _ { j } ^ { r } ( \xi ) - b \xi \right| } \\ & { \leq \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left| \bar { f } ^ { r } ( \xi ) - \displaystyle \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } ^ { r } s _ { j } + a _ { \mathrm { k v } } ^ { r } \tilde { m } _ { j } ) P _ { j } ^ { r } ( N ) - b ^ { r } N \right| + \displaystyle \sum _ { j = 1 } ^ { J } s _ { j } \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left| ( r a _ { \mathrm { p f } } ^ { r } - a _ { \mathrm { p f } } ) \bar { P } _ { j } ^ { r } ( \xi ) \right| } \\ & { \quad + \displaystyle \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left| ( r a _ { \mathrm { k v } } ^ { r } - a _ { \mathrm { k v } } ) \bar { P } _ { j } ^ { r } ( \xi ) \right| + \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } | b ^ { r } N - b \xi | \longrightarrow 0 . } \end{array}
$$

This proves (67).

(v) For $0 \le \xi \le T$ , the definition of $\widehat { P } _ { j } ^ { r }$ in (58) and the one-step bound on $y _ { j } ^ { r }$ in (i) give

$$
0 \leq \widehat { P } _ { j } ^ { r } ( \xi ) - \bar { P } _ { j } ^ { r } ( \xi ) = \frac { r \xi - \lfloor r \xi \rfloor } { r } y _ { j } ^ { r } ( \lfloor r \xi \rfloor ) \leq \frac { M } { r \tilde { m } _ { j } } .
$$

This proves the first estimate in (68). On each grid interval $[ n / r , ( n + 1 ) / r ]$ , the slope of $\widehat { P } _ { j } ^ { r }$ is $y _ { j } ^ { r } ( n )$ , and $\begin{array} { r } { 0 \leq y _ { j } ^ { r } ( n ) \leq \frac { M } { \tilde { m } _ { j } } } \end{array}$ . Consequently, for any $0 \leq \xi _ { 1 } \leq \xi _ { 2 } \leq T$

$$
0 \leq \widehat { P } _ { j } ^ { r } ( \xi _ { 2 } ) - \widehat { P } _ { j } ^ { r } ( \xi _ { 1 } ) \leq \frac { M } { \tilde { m } _ { j } } ( \xi _ { 2 } - \xi _ { 1 } ) ,
$$

which proves the second estimate.

The definition of ${ \widehat { f } } ^ { r }$ in (59) now gives, for $0 \leq \xi _ { 1 } \leq \xi _ { 2 } \leq T$

$$
\begin{array} { r } { \displaystyle \left| \widehat { f } ^ { r } ( \xi _ { 2 } ) - \widehat { f } ^ { r } ( \xi _ { 1 } ) \right| \leq \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \left| \widehat { P } _ { j } ^ { r } ( \xi _ { 2 } ) - \widehat { P } _ { j } ^ { r } ( \xi _ { 1 } ) \right| + b | \xi _ { 2 } - \xi _ { 1 } | } \\ { \leq \left[ b + M \sum _ { j = 1 } ^ { J } \left( \frac { a _ { \mathrm { p f } } s _ { j } } { \tilde { m } _ { j } } + a _ { \mathrm { k v } } \right) \right] | \xi _ { 2 } - \xi _ { 1 } | = L _ { f } | \xi _ { 2 } - \xi _ { 1 } | . } \end{array}
$$

Finally, the definition of ${ \widehat { f } } ^ { r }$ in (59), the clock approximation in (67), and the interpolation error bound (68) give

$$
\begin{array} { r l } { \displaystyle \operatorname* { s u p } _ { 0 \le \xi \le T } \Big | \bar { f } ^ { r } ( \xi ) - \widehat f ^ { r } ( \xi ) \Big | \le \displaystyle \operatorname* { s u p } _ { 0 \le \xi \le T } \bigg | \bar { f } ^ { r } ( \xi ) - \displaystyle \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \bar { P } _ { j } ^ { r } ( \xi ) - b \xi \bigg | } & { } \\ { + \displaystyle \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \displaystyle \operatorname* { s u p } _ { 0 \le \xi \le T } \Big | \bar { P } _ { j } ^ { r } ( \xi ) - \widehat { P } _ { j } ^ { r } ( \xi ) \Big | \to 0 . } \end{array}
$$

(vi) Since $0 \leq \widehat { f } ^ { r } ( \xi ) \leq L _ { f } T$ on [0, �], the clock approximation in (70) implies that, for all suficiently large �,

$$
0 \leq \bar { f } ^ { r } ( \xi ) \leq L _ { f } T + 1 , \qquad 0 \leq \xi \leq T .
$$

By the local boundedness of $\begin{array} { r } { \lambda _ { j } , \int _ { 0 } ^ { L _ { f } T + 1 } \lambda _ { j } ( s ) d s < \infty } \end{array}$ . Apply the functional law of large numbers for unit-rate Poisson processes [43, Corollary 3.2.1] on the cumulative-intensity interval $\begin{array} { r } { \left[ 0 , \int _ { 0 } ^ { L _ { f } T + 1 } \lambda _ { j } ( s ) d s \right] } \end{array}$ . We obtain

$$
\operatorname* { s u p } _ { 0 \leq u \leq \int _ { 0 } ^ { L _ { f } T + 1 } \lambda _ { j } ( s ) d s }  \frac { N _ { j } ^ { r } ( r u ) } { r } - u  \overset { \mathbb { P } } {  } 0 .
$$

Also, the local boundedness of the arrival rates gives, for all $x , y \in \left[ 0 , L _ { f } T + 1 \right]$

$$
\left| \int _ { 0 } ^ { x } \lambda _ { j } ( s ) d s - \int _ { 0 } ^ { y } \lambda _ { j } ( s ) d s \right| \leq \left\| \lambda _ { j } \right\| _ { L ^ { \infty } ( [ 0 , L _ { f } T + 1 ] ) } | x - y | .\tag{72}
$$

Both clocks take values in $[ 0 , L _ { f } T + 1 ]$ for all suficiently large �. Using the definitions of $\bar { A } _ { j } ^ { r }$ and $\widehat { A } _ { j } ^ { r }$ , for $0 \le \xi \le T$ we have

$$
\begin{array} { l } { \displaystyle \bar { A } _ { j } ^ { r } ( \xi ) - \widehat { A } _ { j } ^ { r } ( \xi ) = \frac { p _ { j , L _ { j } } ^ { r } ( 0 ) } { r } + \frac { 1 } { r } N _ { j } ^ { r } \left( r \int _ { 0 } ^ { \bar { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s \right) - p _ { j , L _ { j } } ( 0 ) - \int _ { 0 } ^ { \bar { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s } \\ { \displaystyle \qquad = \frac { 1 } { r } N _ { j } ^ { r } \left( r \int _ { 0 } ^ { \bar { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s \right) - \int _ { 0 } ^ { \bar { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s + \frac { p _ { j , L _ { j } } ^ { r } ( 0 ) } { r } - p _ { j , L _ { j } } ( 0 ) } \\ { \displaystyle \qquad + \int _ { 0 } ^ { \bar { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s - \int _ { 0 } ^ { \widehat { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s . } \end{array}
$$

Therefore, the Lipschitz bound for $A _ { j }$ in (72) gives

$$
\begin{array} { r l } & { \underset { 0 \leq \xi \leq \bar { Z } } { \operatorname* { s u p } } | \bar { A } _ { j } ^ { r } ( \xi ) - \widehat { A } _ { j } ^ { r } ( \xi ) | } \\ & { \overset { \mathrm { S } } { \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } } | \frac { 1 } { r } N _ { j } ^ { r } \Bigg ( r \int _ { 0 } ^ { \bar { j } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s \Bigg ) - \int _ { 0 } ^ { \bar { j } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s \Bigg | + | \frac { p _ { j , L _ { j } } ^ { r } ( 0 ) } { r } - p _ { j , L _ { j } } ( 0 ) |  } \\ & { \quad +  \underset { 0 \leq \xi \leq \bar { Z } } { \operatorname* { s u p } } | \int _ { 0 } ^ { \bar { j } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s - \int _ { 0 } ^ { \widehat { j } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s |  } \\ & { \overset { \mathrm { S } } { \underset { 0 \leq u \leq \bar { Z } } { \operatorname* { s u p } } } \underset { 0 \leq u } { \operatorname* { s u p } } | \frac { N _ { j } ^ { r } ( r u ) } { r } - u | + | \frac { p _ { j , L _ { j } } ^ { r } ( 0 ) } { r } - p _ { j , L _ { j } } ( 0 ) | + \| \lambda _ { j } \| _ { L ^ { \infty } ( [ 0 , L _ { j } T + 1 ] ) } \underset { 0 \leq \xi \leq \bar { T } } { \operatorname* { s u p } } | \bar { f } ^ { r } ( \xi ) - \widehat { f } ^ { r } ( \xi ) | \overset { \overset { \mathrm { P } } {  } } { \underset { 0 \leq \bar { Z } } { \operatorname* { s u p } } } 0 . } \end{array}
$$

Here the first term converges to zero by the Poisson estimate above, the second by the initial-condition convergence in Theorem 3, and the third by (v). This proves (71). □

Lemma 4. Under the initial condition of Theorem $^ { 3 , }$ consider the smoothed ��-inspired policy in Definition 5.   
Then thefollowing statements hold.

(i) Joint �-tightness. For every fixed $T < \infty$ , the sequence $\left\{ \left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } , \bar { A } _ { 1 } ^ { r } , \ldots , \bar { A } _ { J } ^ { r } \right) : r \geq 1 \right\}$ is �-tight in $D ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$ equipped with the Skorokhod $J _ { 1 }$ topology; $i . e . _ { \cdot }$ , their joint laws are tight in this space, and every subsequential weak limit is in $C ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$ with probability one. Taking $T = 1 , 2 , \dots$ . yields local joint �-tightness on [0, ∞).

(ii) Skorokhod representation of subsequential limits. From every scale subsequence, one can extract a further subsequence and construct copies having the same joint laws on a new probability space such that there exist continuous paths $X _ { 1 } , \ldots , X _ { J } , f$ for which, almost surely, uniformly on every compact interval,

$$
\left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } , \bar { A } _ { 1 } ^ { r } , \ldots , \bar { A } _ { J } ^ { r } \right) \longrightarrow ( X _ { 1 } , \ldots , X _ { J } , f , A _ { 1 } \circ f , \ldots , A _ { J } \circ f ) .\tag{73}
$$

(iii) Structural relations of subsequential limits. Every subsequential limit in (ii) satisfies $\chi _ { j } ( 0 ) = 0$ almost surely for $j = 1 , \dots , J .$ . The function $\chi _ { j }$ is nonnegative and nondecreasing on $[ 0 , \infty )$ . The processes $\chi _ { 1 } , \ldots , \chi _ { J }$ and the real-time clock � satisfy the clock equation

$$
f ( \xi ) = \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \chi _ { j } ( \xi ) + b \xi , \qquad \xi \geq 0 ,\tag{74}
$$

the queue nonnegativity constraints

$$
A _ { j } \big ( f ( \xi ) \big ) - \chi _ { j } ( \xi ) \ge 0 , \qquad j = 1 , \ldots , J , \quad \xi \ge 0 ,\tag{75}
$$

and, for $0 \leq \xi _ { 1 } \leq \xi _ { 2 }$ , the memory constraint

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \big ( \chi _ { j } ( \xi _ { 2 } ) - \chi _ { j } ( \xi _ { 1 } ) \big ) \leq M ( \xi _ { 2 } - \xi _ { 1 } ) .\tag{76}
$$

Proof.

(i) Joint �-tightness. Fix $T < \infty$ . To apply the Arzelà–Ascoli theorem, we first verify that $\widehat { P } _ { i } ^ { r } , \widehat { f } ^ { r }$ , and $\widehat { A } _ { j } ^ { r } .$ $j = 1 , \dots , J ,$ , are uniformly bounded and Lipschitz continuous on [0, �], with bounds and Lipschitz constants independent of $r .$

Lemma 3(v) gives, for all $0 \leq \xi , \xi _ { 1 } , \xi _ { 2 } \leq T$

$$
\begin{array} { c c c } { 0 \leq \widehat { P } _ { j } ^ { r } ( \xi ) \leq \displaystyle \frac { M T } { \tilde { m } _ { j } } , } & { \left| \widehat { P } _ { j } ^ { r } ( \xi _ { 2 } ) - \widehat { P } _ { j } ^ { r } ( \xi _ { 1 } ) \right| \leq \displaystyle \frac { M } { \tilde { m } _ { j } } | \xi _ { 2 } - \xi _ { 1 } | , } \\ { 0 \leq \widehat { f } ^ { r } ( \xi ) \leq L _ { f } T , } & { \left| \widehat { f } ^ { r } ( \xi _ { 2 } ) - \widehat { f } ^ { r } ( \xi _ { 1 } ) \right| \leq L _ { f } | \xi _ { 2 } - \xi _ { 1 } | . } \end{array}
$$

Combining these estimates with the Lipschitz bound for $A _ { j }$ in (72) and the definition of $\widehat { A } _ { j } ^ { r }$ in (60) gives

$$
\begin{array} { r l } { \widehat { A } _ { j } ^ { r } ( \xi _ { 2 } ) - \widehat { A } _ { j } ^ { r } ( \xi _ { 1 } ) \Bigg | = \Bigg | \int _ { 0 } ^ { \widehat { f } ^ { r } ( \xi _ { 2 } ) } \lambda _ { j } ( s ) d s - \int _ { 0 } ^ { \widehat { f } ^ { r } ( \xi _ { 1 } ) } \lambda _ { j } ( s ) d s \Bigg | } & { } \\ & { \leq \left\| \lambda _ { j } \right\| _ { L ^ { \infty } ( [ 0 , L _ { f } ] ^ { r + 1 } ] ) } \left| \widehat { \widehat { f } ^ { r } } ( \xi _ { 2 } ) - \widehat { f } ^ { r } ( \xi _ { 1 } ) \right| } \\ & { \leq \left\| \lambda _ { j } \right\| _ { L ^ { \infty } ( [ 0 , L _ { f } ] ^ { r + 1 } ] ) } L _ { f } | \xi _ { 2 } - \widehat { \xi } _ { 1 } | , } \\ { 0 \leq \widehat { A } _ { j } ^ { r } ( \xi ) = p _ { j , I _ { j } } ( 0 ) + \int _ { 0 } ^ { \widehat { f } ^ { r } ( \xi ) } \lambda _ { j } ( s ) d s } \\ & { \leq p _ { j , L _ { j } } ( 0 ) + \left\| \lambda _ { j } \right\| _ { L ^ { \infty } ( [ 0 , L _ { f } ] + 1 ) } \widehat { f } ^ { r } ( \xi ) } \\ & { \leq p _ { j , I _ { j } } ( 0 ) + \left\| \lambda _ { j } \right\| _ { L ^ { \infty } ( [ 0 , L _ { f } ] + 1 ) } L _ { f } T . } \end{array}
$$

Using the Euclidean norm on $\mathbb { R } ^ { 2 J + 1 }$ , define the constants

$$
\rho _ { T } : = \left[ \sum _ { j = 1 } ^ { J } \Bigl ( \frac { M T } { \tilde { m } _ { j } } \Bigr ) ^ { 2 } + ( L _ { f } T ) ^ { 2 } + \sum _ { j = 1 } ^ { J } \Bigl ( p _ { j , L _ { j } } ( 0 ) + \| \lambda _ { j } \| _ { L ^ { \infty } ( [ 0 , L _ { f } T + 1 ] ) } L _ { f } T \Bigr ) ^ { 2 } \right] ^ { 1 / 2 } ,
$$

$$
L _ { T } : = \left[ \sum _ { j = 1 } ^ { J } \Bigl ( \frac { M } { \tilde { m } _ { j } } \Bigr ) ^ { 2 } + L _ { f } ^ { 2 } + \sum _ { j = 1 } ^ { J } \Bigl ( \| \lambda _ { j } \| _ { L ^ { \infty } ( [ 0 , L _ { f } T + 1 ] ) } L _ { f } \Bigr ) ^ { 2 } \right] ^ { 1 / 2 } .
$$

Both constants are finite by the local boundedness of the arrival rates and $J < \infty$ . Therefore, all sample paths of the continuous approximation vector

$$
\widehat { \mathbf { Z } } ^ { r } : = \left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } , \widehat { A } _ { 1 } ^ { r } , \ldots , \widehat { A } _ { J } ^ { r } \right)
$$

belong to the deterministic set

$$
\mathcal { K } _ { T } : = \left\{ \mathbf { h } \in C ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } ) : \| \mathbf { h } \| _ { \infty , T } \leq \rho _ { T } , \| \mathbf { h } ( \xi _ { 2 } ) - \mathbf { h } ( \xi _ { 1 } ) \| \leq L _ { T } | \xi _ { 2 } - \xi _ { 1 } | , \forall \xi _ { 1 } , \xi _ { 2 } \in [ 0 , T ] \right\} .\tag{77}
$$

The two inequalities defining $\mathcal { K } _ { T }$ are preserved under uniform limits, so $\mathcal { K } _ { T }$ is closed under uniform convergence. It is also uniformly bounded and equicontinuous. By the Arzelà–Ascoli theorem [33, Theorem 7.25], it is relatively compact in the uniform topology and is therefore compact there. Moreover,

$$
\mathbb { P } \left( \widehat { \mathbf { Z } } ^ { r } \in \mathcal { K } _ { T } \right) = 1 , \qquad r \geq 1 .\tag{78}
$$

Thus the family of laws $\{ \mathcal { L } ( \widehat { \mathbf { Z } } ^ { r } ) : r \geq 1 \}$ is tight in $C ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$ under the uniform topology. Since the inclusion into $\overset { \cdot } { D } ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$ with the $J _ { 1 }$ topology is continuous, $\{ \widehat { \mathbf { Z } } ^ { r } \}$ is �-tight in $D ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$

Set $\mathbf { Z } ^ { r } : = \left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } , \bar { A } _ { 1 } ^ { r } , \ldots , \bar { A } _ { J } ^ { r } \right)$ . The vector $\widehat { \mathbf { Z } } ^ { r }$ approximates ${ \bf Z } ^ { r }$ by replacing only the cumulative arrival coordinates $\bar { A } _ { j } ^ { r }$ with their continuous approximations $\widehat { A } _ { j } ^ { r }$ . Hence,

$$
\begin{array} { r l } & { \| \mathbf { Z } ^ { r } - \widehat { \mathbf { Z } } ^ { r } \| _ { \infty , T } : = \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \| \mathbf { Z } ^ { r } ( \xi ) - \widehat { \mathbf { Z } } ^ { r } ( \xi ) \| } \\ & { \qquad = \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left( \underset { j = 1 } { \overset { J } { \sum } } \left| \bar { A } _ { j } ^ { r } ( \xi ) - \widehat { A } _ { j } ^ { r } ( \xi ) \right| ^ { 2 } \right) ^ { 1 / 2 } } \\ & { \qquad \leq \left( \underset { j = 1 } { \overset { J } { \sum } } \left[ \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } \left| \bar { A } _ { j } ^ { r } ( \xi ) - \widehat { A } _ { j } ^ { r } ( \xi ) \right| \right] ^ { 2 } \right) ^ { 1 / 2 } \overset { \mathbb { P } } { \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } } 0 , } \end{array}
$$

where the convergence follows from the approximation of $\bar { A } _ { j } ^ { r }$ by $\widehat { A } _ { j } ^ { r }$ in (71) and $J < \infty$ . Define $\Delta \mathbf { Z } ^ { r } : = \mathbf { Z } ^ { r } - \widehat { \mathbf { Z } } ^ { r }$ $\| \Delta \mathbf { Z } ^ { r } \| _ { \infty , T } \overset { \mathbb { P } } {  } 0$ implies that $\Delta \mathbf { Z } ^ { r }$ converges weakly to the zero process and hence is tight. Since the zero process is continuous, $\{ \Delta \mathbf { Z } ^ { r } \}$ is �-tight. Both $\{ \widehat { \mathbf { Z } } ^ { r } \}$ and $\{ \Delta \mathbf { Z } ^ { r } \}$ are therefore �-tight. Corollary VI.3.33(a) of Jacod and Shiryaev [22] gives that $\{ \mathbf { Z } ^ { r } = \widehat { \mathbf { Z } } ^ { r } + \Delta \mathbf { Z } ^ { r } \}$ is jointly �-tight in $D ( [ 0 , T ] , \mathbb { R } ^ { 2 J + 1 } )$ .

Finally, apply the preceding argument successively on [0, �] for $m = 1 , 2 , \ldots , { \mathrm { e x t r a c t } }$ nested subsequences, and take a diagonal subsequence. This yields local joint �-tightness on $[ 0 , \infty )$

(ii) Skorokhod representation of subsequential limits. We prove the almost-sure convergence on compact intervals in (73) by first identifying the joint weak limit and then applying the Skorokhod representation theorem. Since $\{ ( \overleftarrow { P } _ { 1 } ^ { r } , \ldots , \overbrace { P } _ { J } ^ { r } , \overleftarrow { \hat { f } ^ { r } } ) \}$ is jointly �-tight, every scale subsequence has a further subsequence, still indexed by �, along which

$$
\left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } \right) \Rightarrow \left( X _ { 1 } , \ldots , X _ { J } , f \right)
$$

in the local $J _ { 1 }$ topology, with continuous limit paths. The clock limit � is nonnegative because each ${ \widehat { f } } ^ { r }$ is nonnegative. Each $A _ { j }$ is locally Lipschitz, so the composition map $g \mapsto A _ { j } \circ g$ is continuous under uniform convergence on compact intervals. Since $f$ is continuous, local $J _ { 1 }$ convergence to � implies uniform convergence on compact intervals. The continuous mapping theorem and the definition of $\widehat { A } _ { j } ^ { r }$ in (60) therefore give

$$
\widehat { \mathbf { Z } } ^ { r } \Rightarrow ( X _ { 1 } , \ldots , X _ { J } , f , A _ { 1 } \circ f , \ldots , A _ { J } \circ f ) .
$$

For every $T < \infty$ , the error estimate in (i) gives $\| \Delta \mathbf { Z } ^ { r } \| _ { \infty , T } \overset { \mathbb { P } } {  } 0$ . Applying Lemma VI.3.31 of Jacod and Shiryaev [22] to the joint vector and the error $\Delta \mathbf { Z } ^ { r }$ yields

$$
{ \bf Z } ^ { r } = \widehat { { \bf Z } } ^ { r } + \Delta { \bf Z } ^ { r } \Rightarrow ( X _ { 1 } , \ldots , X _ { J } , f , A _ { 1 } \circ f , \ldots , A _ { J } \circ f ) .
$$

Since $D ( [ 0 , \infty ) , \mathbb { R } ^ { 2 J + 1 } )$ with the local $J _ { 1 }$ topology is Polish, the Skorokhod representation theorem [5, Theorem 6.7] provides copies on a common probability space, preserving the joint law of the coordinates for

each �, that converge almost surely in local $J _ { 1 }$ to a copy of the displayed limit. Since that limit has continuous paths, the convergence is uniform on compact intervals [5, p. 124]. Retaining the original notation for these copies gives (73).

(iii) Structural relations of subsequential limits. Work on a probability-one event where the uniform convergence in (73) and the prelimit path properties used below hold for the same-law copies.

We first verify the path properties and the clock equation. For every �, the path $\widehat { P } _ { j } ^ { r }$ is nonnegative and nondecreasing and satisfies $\widehat { P } _ { j } ^ { r } ( 0 ) = 0$ . Uniform convergence on compact intervals in (73) therefore gives $\ X _ { j } ( 0 ) = 0$

Recall the definition of ${ \widehat { f } } ^ { r }$ in (59):

$$
\widehat { f } ^ { r } ( \xi ) = \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \widehat { P } _ { j } ^ { r } ( \xi ) + b \xi .
$$

For every $T < \infty ,$ , the uniform convergence in (73) yields

$$
\begin{array} { r l } { \displaystyle } & { \displaystyle \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| f ( \xi ) - \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \chi _ { j } ( \xi ) - b \xi \right| } \\ { \displaystyle } & { \displaystyle \leq \operatorname* { s u p } _ { 0 \leq \xi \leq T } | f ( \xi ) - \widehat { f } ^ { r } ( \xi ) | + \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \displaystyle \operatorname* { s u p } _ { 0 \leq \xi \leq T } | \widehat { P } _ { j } ^ { r } ( \xi ) - \chi _ { j } ( \xi ) | \to 0 . } \end{array}
$$

Because � is arbitrary, (74) follows.

Queue nonnegativity constraints. For the original prelimit processes, the cumulative planned-admission bound (55) gives

$$
\widehat { P } _ { j } ^ { r } ( n / r ) = \frac { P _ { j } ^ { r } ( n ) } { r } = \frac { 1 } { r } \sum _ { u = 0 } ^ { n - 1 } y _ { j } ^ { r } ( u ) \leq \frac { 1 } { r } \sum _ { u = 0 } ^ { n } y _ { j } ^ { r } ( u ) \leq \frac { 1 } { r } \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u = 0 } ^ { n - 1 } \eta _ { j } ^ { r } ( u ) \right] = \bar { A } _ { j } ^ { r } ( n / r ) .
$$

Hence,

$$
0 \leq \bar { A } _ { j } ^ { r } ( n / r ) - \widehat { P } _ { j } ^ { r } ( n / r ) , \qquad n \in \mathbb { Z } _ { + } .\tag{79}
$$

This inequality is a joint path property of $( \bar { A } _ { j } ^ { r } , \widehat { P } _ { j } ^ { r } )$ and therefore also holds almost surely for their same-law copies.

Fix $T < \infty$ . Since $\lvert \lfloor r \xi \rfloor / r - \xi \rvert \leq 1 / r$ , the uniform convergence in (73) and the continuity of $A _ { j } \circ f$ and $\chi _ { j }$ imply, uniformly over $0 \le \xi \le T$

$$
\begin{array} { r } { \bar { A } _ { j } ^ { r } \big ( \lfloor r \xi \rfloor / r \big ) - \widehat { P } _ { j } ^ { r } \big ( \lfloor r \xi \rfloor / r \big ) \longrightarrow A _ { j } \big ( f ( \xi ) \big ) - \chi _ { j } ( \xi ) . } \end{array}
$$

The left-hand side is nonnegative by (79); its limit is therefore nonnegative. Since � is arbitrary, (75) follows.

We next verify the memory constraint (76). Summing the planning-budget bound (63) from $n _ { 1 }$ to $n _ { 2 } - 1$ gives, for integers $0 \leq n _ { 1 } \leq n _ { 2 }$ ,

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } ( P _ { j } ^ { r } ( n _ { 2 } ) - P _ { j } ^ { r } ( n _ { 1 } ) ) = \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \sum _ { t = n _ { 1 } } ^ { n _ { 2 } - 1 } y _ { j } ^ { r } ( t ) = \sum _ { t = n _ { 1 } } ^ { n _ { 2 } - 1 } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } y _ { j } ^ { r } ( t ) \leq \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } ( n _ { 2 } - n _ { 1 } ) .
$$

After division by � and use of $\widehat { P } _ { j } ^ { r } ( n / r ) = P _ { j } ^ { r } ( n ) / r$ , this becomes

$$
\sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \left[ \widehat { P } _ { j } ^ { r } \left( \frac { n _ { 2 } } { r } \right) - \widehat { P } _ { j } ^ { r } \left( \frac { n _ { 1 } } { r } \right) \right] \leq \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } \frac { n _ { 2 } - n _ { 1 } } { r } .
$$

This is again a path property retained by the same-law copies. Given $0 \leq \xi _ { 1 } \leq \xi _ { 2 }$ , take $n _ { i } = \left\lfloor r \xi _ { i } \right\rfloor$ . Then the uniform convergence in (73) and the continuity of $\chi _ { j }$ give

$$
\begin{array} { r l } & { \widehat { P } _ { j } ^ { r } ( n _ { i } / r ) \longrightarrow { X } _ { j } ( \xi _ { i } ) , \qquad i = 1 , 2 , } \\ { \mathrm { w h i l e } } & { \displaystyle \frac { n _ { 2 } - n _ { 1 } } { r } \longrightarrow \xi _ { 2 } - \xi _ { 1 } , \qquad \displaystyle \frac { w _ { r } } { w _ { r } + H _ { \mathrm { k v } } } \longrightarrow 1 . } \end{array}
$$

Passing to the limit in the preceding inequality proves (76).

Lemma 5 (Residual representation and uniqueness).

Under the initial condition ofTheorem 3, every subsequential limit obtained in Lemma 4 satisfies thefollowing properties almost surely.

(i) The clock � is a strictly increasing continuous bijection from $[ 0 , \infty )$ onto itself. Define

$$
R _ { 0 } ( s ) : = s , \qquad R _ { j } \bigl ( f ( \xi ) \bigr ) : = f ( \xi ) - \sum _ { \ell = 1 } ^ { j } \mu _ { \ell } ^ { - 1 } \chi _ { \ell } ( \xi ) , j = 1 , \ldots , J .\tag{80}
$$

Then each $R _ { j } , j = 0 , \dotsc , J ,$ , is continuous, nonnegative, and nondecreasing, with $R _ { j } ( 0 ) = 0$ . Moreover, $f o r j = 1 , \dotsc , J ,$

$$
R _ { j } ( s ) \geq R _ { j - 1 } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) , \qquad s \geq 0 .\tag{81}
$$

(ii) For every $j = 1 , \dots , J$ and $s \geq 0 ,$

$$
\int _ { 0 } ^ { s } \Big [ R _ { j } ( u ) - R _ { j - 1 } ( u ) + \mu _ { j } ^ { - 1 } A _ { j } ( u ) \Big ] \ d R _ { j } ( u ) = 0 .\tag{82}
$$

(iii) Thefollowing identities hold:

$$
R _ { j } = R _ { j } ^ { * } , \quad f = f ^ { * } , \quad X _ { j } = X _ { j } ^ { * } ,\tag{83}
$$

where $j = 0 , \ldots , J f o r R _ { j } a n d j = 1 , \ldots , J f o r \chi _ { j } .$ . Consequently, every subsequential limit is the same deterministic target path.

Proof. (i) Work on a probability-one event on which the conclusions of Lemma 4 and all prelimit path properties used below hold for the same-law copies. For $0 \leq \xi _ { 1 } < \xi _ { 2 }$ , the clock equation (74) and the monotonicity of the $\chi _ { j }$ give

$$
f ( \xi _ { 2 } ) - f ( \xi _ { 1 } ) = \sum _ { j = 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { j } + a _ { \mathrm { k v } } \tilde { m } _ { j } ) \big ( X _ { j } ( \xi _ { 2 } ) - X _ { j } ( \xi _ { 1 } ) \big ) + b ( \xi _ { 2 } - \xi _ { 1 } ) \geq b ( \xi _ { 2 } - \xi _ { 1 } ) > 0 ,
$$

where the last inequality uses the standing assumption $b > 0$ . Together with the continuity of $f , f ( 0 ) = 0$ and $f ( \xi ) \geq b \xi  \infty$ , this shows that $f$ is a continuous bijection onto $[ 0 , \infty )$ with a continuous inverse.

The clock equation (74) and the definition of $R _ { j }$ in (80) give

$$
\begin{array} { l } { { \displaystyle R _ { j } \big ( f ( \xi ) \big ) = \sum _ { \ell = j + 1 } ^ { J } ( a _ { \mathrm { p f } } s _ { \ell } + a _ { \mathrm { k v } } \tilde { m } _ { \ell } ) \chi _ { \ell } ( \xi ) + b \xi - \frac { b } { M } \sum _ { \ell = 1 } ^ { j } \tilde { m } _ { \ell } \chi _ { \ell } ( \xi ) } } \\ { { \displaystyle \quad = \sum _ { \ell = j + 1 } ^ { J } \bigg ( a _ { \mathrm { p f } } s _ { \ell } + a _ { \mathrm { k v } } \tilde { m } _ { \ell } + \frac { b } { M } \tilde { m } _ { \ell } \bigg ) \chi _ { \ell } ( \xi ) + b \xi - \frac { b } { M } \sum _ { \ell = j + 1 } ^ { J } \tilde { m } _ { \ell } \chi _ { \ell } ( \xi ) - \frac { b } { M } \sum _ { \ell = 1 } ^ { j } \tilde { m } _ { \ell } \chi _ { \ell } ( \xi ) } } \\ { { \displaystyle \quad = \sum _ { \ell = j + 1 } ^ { J } \mu _ { \ell } ^ { - 1 } \chi _ { \ell } ( \xi ) + \frac { b } { M } \left( M \xi - \sum _ { \ell = 1 } ^ { J } \tilde { m } _ { \ell } \chi _ { \ell } ( \xi ) \right) . } } \end{array}\tag{84}
$$

By Lemma 4, each $\chi _ { \ell }$ is continuous, nonnegative, and nondecreasing, with $X _ { \ell } ( 0 ) = 0$ . The function $\begin{array} { r } { \xi \longmapsto M \xi - \sum _ { \ell = 1 } ^ { J } \tilde { m } _ { \ell } \chi _ { \ell } ( \xi ) } \end{array}$ is also continuous and equals zero at $\xi = 0$ . The memory constraint (76) shows that it is nonnegative and nondecreasing. Thus the representation (84) gives these same properties for $R _ { j } \circ f$ Since $f ^ { - 1 }$ is continuous and increasing with $f ^ { - 1 } ( 0 ) = 0$ , they also hold for $R _ { j } = ( R _ { j } \circ f ) \circ f ^ { - 1 }$

Finally, the definition of $R _ { j }$ in (80) and the queue nonnegativity constraint (75) give

$$
R _ { j } \big ( f ( \xi ) \big ) - R _ { j - 1 } \big ( f ( \xi ) \big ) + \mu _ { j } ^ { - 1 } A _ { j } \big ( f ( \xi ) \big ) = \mu _ { j } ^ { - 1 } \big ( A _ { j } \big ( f ( \xi ) \big ) - X _ { j } ( \xi ) \big ) \geq 0 .
$$

Setting $\xi = f ^ { - 1 } ( s )$ proves (81).

(ii) We prove the complementarity relation (82) by showing that $R _ { j }$ is constant on intervals where the class-� waiting-queue length is positive. At every GPU step �, the cumulative planned-admission bound (55) gives

$$
p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \sum _ { u < t } y _ { j } ^ { r } ( u ) \ge y _ { j } ^ { r } ( t ) \ge 0 .
$$

The planning-budget bound (63) gives

$$
\frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \geq \sum _ { \ell = j } ^ { J } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \geq 0 .
$$

Thus the positive parts in the planning rule (33) can be removed:

$$
y _ { j } ^ { r } ( t ) = \operatorname* { m i n } \Biggl \{ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \sum _ { u < t } y _ { j } ^ { r } ( u ) , \tilde { m } _ { j } ^ { - 1 } \left[ \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell = 1 } ^ { j - 1 } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \right] \Biggr \} .
$$

Applying the identity $\left( a - \operatorname* { m i n } \{ a , b \} \right) \left( b - \operatorname* { m i n } \{ a , b \} \right) = 0$ (valid for $a , b \geq 0 )$ to the two arguments of the minimum and multiplying by $\tilde { m } _ { j }$ gives

$$
\left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \sum _ { u \le t } y _ { j } ^ { r } ( u ) \right] \left[ \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } - \sum _ { \ell = 1 } ^ { j } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) \right] = 0 .\tag{85}
$$

The first factor in (85) is the number of class-� requests that remain without an admission plan after planning at step �. After fluid scaling, it becomes

$$
\frac { 1 } { r } \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \sum _ { u \le t } y _ { j } ^ { r } ( u ) \right] = \bar { A } _ { j } ^ { r } ( t / r ) - \widehat { P } _ { j } ^ { r } ( ( t + 1 ) / r ) .
$$

For every $T < \infty$

$$
\begin{array} { r l } & { \underset { 0 \leq t \leq [ r ] } { \operatorname* { m a x } } \left| \frac { 1 } { r } \left[ p _ { j , L _ { j } } ^ { r } ( 0 ) + \displaystyle \sum _ { u < t } \eta _ { j } ^ { r } ( u ) - \displaystyle \sum _ { u \leq t } y _ { j } ^ { r } ( u ) \right] - \left[ A _ { j } \left( f \left( \frac { t } { r } \right) \right) - X _ { j } \left( \frac { t } { r } \right) \right] \right| } \\ & { = \underset { 0 \leq t \leq [ r , T ] } { \operatorname* { m a x } } \left| \bar { A } _ { j } ^ { r } \left( \frac { t } { r } \right) - A _ { j } \left( f \left( \frac { t } { r } \right) \right) - \left[ \widehat { P } _ { j } ^ { r } \left( \frac { t + 1 } { r } \right) - X _ { j } \left( \frac { t + 1 } { r } \right) \right] - \left[ X _ { j } \left( \frac { t + 1 } { r } \right) - X _ { j } \left( \frac { t } { r } \right) \right] \right| } \\ & { \overset { \leq } { 0 \leq t \leq [ r ] } \bar { A } _ { j } ^ { r } ( \xi ) - A _ { j } ( f ( \xi ) ) | + \underset { 0 \leq \xi \leq T + 1 } { \operatorname* { s u p } } | \widehat { P } _ { j } ^ { r } ( \xi ) - X _ { j } ( \xi ) | + \underset { 0 \leq x , y \leq T + 1 } { \operatorname* { s u p } } | X _ { j } ( x ) - X _ { j } ( y ) | \longrightarrow 0 . } \end{array}\tag{86}
$$

Here the first two terms vanish by the uniform convergence in (73), and the last by uniform continuity of $\chi _ { j }$ on $[ 0 , T + 1 ]$ . Let $[ u , \nu ]$ , with $u < \nu$ , be a compact interval contained in

$$
\left\{ \xi \geq 0 : A _ { j } { \big ( } f ( \xi ) { \big ) } - X _ { j } ( \xi ) > 0 \right\} .
$$

Since $A _ { j } \circ f - X _ { j }$ is continuous and strictly positive on the compact interval $[ u , \nu ]$ , there exists $\delta > 0$ such that

$$
A _ { j } \big ( f ( \xi ) \big ) - X _ { j } ( \xi ) \geq 2 \delta , \qquad \xi \in [ u , \nu ] .
$$

By (86) with $T = \nu _ { ; }$ , choose $r _ { 0 }$ so that the maximum error in (86) is less than $\delta$ and $\lceil r u \rceil < \lfloor r \nu \rfloor$ for every $r \geq r _ { 0 }$ . Then, for $t = \lceil r u \rceil , \ldots , \lfloor r \nu \rfloor - 1$

$$
\bar { A } _ { j } ^ { r } ( t / r ) - \widehat { P } _ { j } ^ { r } ( ( t + 1 ) / r ) > A _ { j } ( f ( t / r ) ) - X _ { j } ( t / r ) - \delta \geq \delta > 0 .
$$

Thus the first factor in (85) is strictly positive. Its second factor must be zero, giving

$$
\sum _ { \ell = 1 } ^ { j } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) = \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } .\tag{87}
$$

The planning-budget bound (63) now implies

$$
0 \leq \sum _ { \ell = j + 1 } ^ { J } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) = \sum _ { \ell = 1 } ^ { J } \tilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) - \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k v } } } \leq 0 .
$$

Since $\tilde { m } _ { \ell } > 0$ and $y _ { \ell } ^ { r } ( t ) \geq 0$ , every term in the first sum is zero, so $y _ { \ell } ^ { r } ( t ) = 0$ for $\ell > j$

For $\ell > j ,$ , summing over $t = \lceil r u \rceil , \ldots , \lfloor r \nu \rfloor - 1$ and using the definition of $P _ { \ell } ^ { r }$ gives

$$
0 = \frac { 1 } { r } \sum _ { t = \lceil r u \rceil } ^ { \lfloor r \nu \rfloor - 1 } y _ { \ell } ^ { r } ( t ) = \frac { P _ { \ell } ^ { r } ( \lfloor r \nu \rfloor ) - P _ { \ell } ^ { r } ( \lceil r u \rceil ) } { r } = \widehat { P } _ { \ell } ^ { r } ( \lfloor r \nu \rfloor / r ) - \widehat { P } _ { \ell } ^ { r } ( \lceil r u \rceil / r ) .
$$

Here we used $\widehat { P } _ { \ell } ^ { r } ( n / r ) = P _ { \ell } ^ { r } ( n ) / r$ at grid points. Since $y _ { \ell } ^ { r } ( t ) = 0$ for $\ell > j$ , summing both sides of the full-budget equality (87) over the same GPU steps and dividing by � gives

$$
\begin{array} { l } { \displaystyle \frac { w _ { r } M } { w _ { r } + H _ { \mathrm { k w } } } \displaystyle \frac { | r \nu | - [ r u ] } { r } = \displaystyle \frac { 1 } { r } \sum _ { \tau = \tau \varkappa } ^ { \lfloor r \nu \rfloor - 1 } \displaystyle \sum _ { \ell = 1 } ^ { J } \widetilde { m } _ { \ell } y _ { \ell } ^ { r } ( t ) } \\ { = \displaystyle \frac { 1 } { r } \sum _ { \ell = 1 } ^ { J } \widetilde { m } _ { \ell } \displaystyle \sum _ { \tau = \tau \varkappa } ^ { \lfloor r \nu \rfloor - 1 } y _ { \ell } ^ { r } ( t ) } \\ { = \displaystyle \sum _ { \ell = 1 } ^ { J } \widetilde { m } _ { \ell } \displaystyle \frac { P _ { \ell } ^ { r } ( [ r u ] ) } { r } - P _ { \ell } ^ { r } ( [ r u ] ) } \\ { = \displaystyle \sum _ { \ell = 1 } ^ { J } \widetilde { m } _ { \ell } \left[ \widehat { P } _ { \ell } ^ { r } ( [ r \nu ] / r ) - \widehat { P } _ { \ell } ^ { r } ( [ r u ] / r ) \right] . } \end{array}
$$

The convergence of the grid endpoints to � and �, the limit $w _ { r } / ( w _ { r } + H _ { \mathrm { k v } } ) \to 1$ , the uniform convergence of $\widehat { P } _ { \ell } ^ { r }$ to $\chi _ { \ell }$ in (73), and the continuity of $\chi _ { \ell }$ together give

$$
X _ { \ell } ( \nu ) - X _ { \ell } ( u ) = 0 \quad ( \ell > j ) , \qquad \mathrm { a n d } \qquad \sum _ { \ell = 1 } ^ { J } \tilde { m } _ { \ell } \bigl ( X _ { \ell } ( \nu ) - X _ { \ell } ( u ) \bigr ) = M ( \nu - u ) .
$$

Substituting these equalities into the representation of $R _ { j } \circ f$ in (84) gives

$$
R _ { j } \big ( f ( \nu ) \big ) - R _ { j } \big ( f ( u ) \big ) = \sum _ { \ell = j + 1 } ^ { J } \mu _ { \ell } ^ { - 1 } \big ( X _ { \ell } ( \nu ) - X _ { \ell } ( u ) \big ) + \frac { b } { M } \left[ M ( \nu - u ) - \sum _ { \ell = 1 } ^ { J } \tilde { m } _ { \ell } \big ( X _ { \ell } ( \nu ) - X _ { \ell } ( u ) \big ) \right] = 0 .
$$

Since $[ u , \nu ]$ is arbitrary and $f$ is a homeomorphism, the preceding equality implies that the Lebesgue– Stieltjes measure $d R _ { j }$ satisfies

$$
d R _ { j } \Big ( \big \{ s \geq 0 : A _ { j } ( s ) - \chi _ { j } \big ( f ^ { - 1 } ( s ) \big ) > 0 \big \} \Big ) = 0 .
$$

The definition of $R _ { j }$ in (80) gives

$$
R _ { j } ( s ) - R _ { j - 1 } ( s ) + \mu _ { j } ^ { - 1 } A _ { j } ( s ) = \mu _ { j } ^ { - 1 } \left[ A _ { j } ( s ) - \chi _ { j } \big ( f ^ { - 1 } ( s ) \big ) \right] \ge 0 .
$$

The integrand in (82) is therefore zero $d R _ { j }$ -almost everywhere, so for every $s \geq 0$

$$
\int _ { 0 } ^ { s } \Big [ R _ { j } ( u ) - R _ { j - 1 } ( u ) + \mu _ { j } ^ { - 1 } A _ { j } ( u ) \Big ] \ d R _ { j } ( u ) = 0 .
$$

(iii) For fixed $j ,$ the function $R _ { j - 1 } - \mu _ { j } ^ { - 1 } A _ { j }$ is continuous. By $( \mathrm { i } ) , R _ { j }$ is continuous and nondecreasing, with $R _ { j } ( 0 ) = 0$ , and

$$
R _ { j } ( s ) \geq R _ { j - 1 } ( s ) - \mu _ { j } ^ { - 1 } A _ { j } ( s ) , \qquad s \geq 0 .
$$

By (ii), for every $s \geq 0$

$$
\int _ { 0 } ^ { s } \Big [ R _ { j } ( u ) - R _ { j - 1 } ( u ) + \mu _ { j } ^ { - 1 } A _ { j } ( u ) \Big ] \ d R _ { j } ( u ) = 0 .
$$

The one-dimensional reflection characterization [42, Theorem 14.2.3], applied on each compact interval, therefore yields

$$
R _ { j } ( s ) = \operatorname* { s u p } _ { 0 \leq u \leq s } \left[ R _ { j - 1 } ( u ) - \mu _ { j } ^ { - 1 } A _ { j } ( u ) \right] ^ { + } .
$$

The base case is $R _ { 0 } = R _ { 0 } ^ { * }$ because both are the identity function. If $R _ { j - 1 } = R _ { j - 1 } ^ { * }$ , the preceding display is exactly the recursion defining $R _ { j } ^ { * }$ in (20). Induction gives

$$
R _ { j } ( s ) = R _ { j } ^ { * } ( s ) , \qquad j = 0 , \ldots , J , \quad s \geq 0 .\tag{88}
$$

Substituting the definitions of $R _ { j - 1 }$ and $R _ { j }$ from (80) gives

$$
R _ { j - 1 } \big ( f ( \xi ) \big ) - R _ { j } \big ( f ( \xi ) \big ) = \left[ f ( \xi ) - \sum _ { \ell = 1 } ^ { j - 1 } \mu _ { \ell } ^ { - 1 } \chi _ { \ell } ( \xi ) \right] - \left[ f ( \xi ) - \sum _ { \ell = 1 } ^ { j } \mu _ { \ell } ^ { - 1 } \chi _ { \ell } ( \xi ) \right] = \mu _ { j } ^ { - 1 } \chi _ { j } ( \xi ) .
$$

Multiplying by $\mu _ { j }$ yields

$$
\chi _ { j } ( \xi ) = \mu _ { j } \left[ R _ { j - 1 } \big ( f ( \xi ) \big ) - R _ { j } \big ( f ( \xi ) \big ) \right] .\tag{89}
$$

Taking $j = J$ in the representation of $R _ { j } \circ f$ in (84) gives

$$
\xi = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \chi _ { j } ( \xi ) + \frac { 1 } { b } R _ { J } \big ( f ( \xi ) \big ) .
$$

The definition of � in (21), the identification $R _ { j } ^ { * } = R _ { j }$ in (88), and the representation of $\chi _ { j }$ in (89) give

$$
\begin{array} { l } { { \displaystyle F \big ( f ( \xi ) \big ) = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \mu _ { j } \left[ R _ { j - 1 } ^ { * } \big ( f ( \xi ) \big ) - R _ { j } ^ { * } \big ( f ( \xi ) \big ) \right] + \frac { 1 } { b } R _ { j } ^ { * } \big ( f ( \xi ) \big ) } } \\ { { \displaystyle \qquad = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } \mu _ { j } \left[ R _ { j - 1 } \big ( f ( \xi ) \big ) - R _ { j } \big ( f ( \xi ) \big ) \right] + \frac { 1 } { b } R _ { J } \big ( f ( \xi ) \big ) } } \\ { { \displaystyle \qquad = \frac { 1 } { M } \sum _ { j = 1 } ^ { J } \tilde { m } _ { j } X _ { j } ( \xi ) + \frac { 1 } { b } R _ { J } \big ( f ( \xi ) \big ) } } \\ { { \displaystyle \qquad = \xi , \qquad \xi \ge 0 . } } \end{array}\tag{90}
$$

By (i), � is a homeomorphism of $[ 0 , \infty )$ onto itself. Hence (90) implies $F = f ^ { - 1 }$ . In particular, the generalized inverse in (22) is the ordinary inverse, and therefore $f ^ { * } = F ^ { - 1 } = f$ . Finally, comparing the representation of $\chi _ { j }$ in (89) with the definition of $\chi _ { j } ^ { * }$ in (23) gives $\chi _ { j } = \chi _ { j } ^ { * }$ for every $j$ □

ProofofTheorem 3. Fix $T < \infty$ . By Lemma $4 ,$ , every scale subsequence has a further weakly convergent subsequence. Lemmas 4 and 5 show that the $( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } )$ coordinates of every such limit equal the deterministic path $\left( X _ { 1 } ^ { * } , \ldots , X _ { J } ^ { * } , f ^ { * } \right)$ . Hence

$$
\begin{array} { r } { \left( \widehat { P } _ { 1 } ^ { r } , \ldots , \widehat { P } _ { J } ^ { r } , \widehat { f } ^ { r } \right) \xrightarrow [ ] { d } \left( X _ { 1 } ^ { * } , \ldots , X _ { J } ^ { * } , f ^ { * } \right) . } \end{array}
$$

Weak convergence to a deterministic path is convergence in probability, and at a continuous limit local $J _ { 1 }$ convergence strengthens to uniform convergence on compact intervals [5, p. 124]. Therefore

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \widehat { f } ^ { r } ( \xi ) - f ^ { * } ( \xi )  + \sum _ { j = 1 } ^ { J } \operatorname* { s u p } _ { 0 \leq \xi \leq T }  \widehat { P } _ { j } ^ { r } ( \xi ) - \chi _ { j } ^ { * } ( \xi )  \overset { \mathbb { P } } {  } 0 .\tag{91}
$$

By the clock approximation in (70) and the uniform convergence of ${ \widehat { f } } ^ { r }$ to $f ^ { * }$ in (91),

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } | \bar { f } ^ { r } ( \xi ) - f ^ { * } ( \xi ) | \leq \operatorname* { s u p } _ { 0 \leq \xi \leq T } | \bar { f } ^ { r } ( \xi ) - \widehat { f } ^ { r } ( \xi ) | + \operatorname* { s u p } _ { 0 \leq \xi \leq T } | \widehat { f } ^ { r } ( \xi ) - f ^ { * } ( \xi ) | \overset { \mathbb { P } } {  } 0 .
$$

This proves (39).

For each class $j ,$ , the bound on the diference between fluid-scaled cumulative planned admissions and fluid-scaled cumulative admissions in (64), the interpolation error bound (68), and the uniform convergence of $\widehat { P } _ { j } ^ { r }$ to $\chi _ { j } ^ { * }$ in (91) give

$$
\begin{array} { r l } & { \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } | \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) | } \\ & { \leq \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } | \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - \bar { P } _ { j } ^ { r } ( \xi ) | + \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } | \bar { P } _ { j } ^ { r } ( \xi ) - \widehat { P } _ { j } ^ { r } ( \xi ) | + \underset { 0 \leq \xi \leq T } { \operatorname* { s u p } } | \widehat { P } _ { j } ^ { r } ( \xi ) - X _ { j } ^ { * } ( \xi ) | \overset { \mathbb { P } } {  } 0 . } \end{array}
$$

This is the $k = L _ { j }$ case of (37). For $k = 1 , \ldots , L _ { j }$ , the prefill-stage delay bound (65) gives

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| - \mathcal { U } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) \right| \leq \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { \mathcal { U } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) } { r } \right| + \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) \right| \overset { \mathtt { P } } { \to } 0 .
$$

Similarly, for $k = 1 , \ldots , o _ { j } - 1$ , the decode-stage delay bound (66) gives

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| - \mathcal { V } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) \right| \leq \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { \mathcal { V } _ { j , k } ^ { r } ( \lfloor r \xi \rfloor ) - \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) } { r } \right| + \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - X _ { j } ^ { * } ( \xi ) \right| \overset { \mathbb { F } } { \to } 0 .
$$

This proves (37) and (38).

Finally, we prove convergence of the system states. The fluid states satisfy $p _ { j , k } ^ { \pi ^ { * } } ( \xi ) = 0 \mathrm { f o r } k = 1 , \ldots , L _ { j } - 1$ and $d _ { j , k } ^ { \pi ^ { * } } ( \xi ) = 0$ for $k = 1 , \ldots , o _ { j } - 1$ . By (8), we therefore obtain

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \bar { p } _ { j , k } ^ { r } ( \xi ) - p _ { j , k } ^ { \pi ^ { * } } ( \xi )  \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , L _ { j } - 1 ,
$$

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \bar { d } _ { j , k } ^ { r } ( \xi ) - d _ { j , k } ^ { \pi ^ { * } } ( \xi )  \overset { \mathbb { P } } {  } 0 , \quad k = 1 , \ldots , o _ { j } - 1 .
$$

It remains to prove (40) for $k = L _ { j }$ . By definition,

$$
\frac { 1 } { r } p _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) = \bar { A } _ { j } ^ { r } ( \xi ) - \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) \quad \mathrm { a n d } \quad p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) = A _ { j } \big ( f ^ { * } ( \xi ) \big ) - \chi _ { j } ^ { * } ( \xi ) .
$$

By (73) and Lemma $^ { 5 , }$ every subsequential limit of $\bar { A } _ { j } ^ { r }$ equals the deterministic path $A _ { j } \circ f ^ { * }$ . Hence,

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T }  \bar { A } _ { j } ^ { r } ( \xi ) - A _ { j } ( f ^ { * } ( \xi ) )  \overset { \mathbb { P } } {  } 0 .
$$

Combining this with the convergence of $\mathcal { U } _ { j , L _ { j } } ^ { r } / r \mathrm { t o } X _ { j } ^ { * }$ in (37), we obtain

$$
\operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { 1 } { r } p _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - p _ { j , L _ { j } } ^ { \pi ^ { * } } ( \xi ) \right| \leq \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \bar { A } _ { j } ^ { r } ( \xi ) - A _ { j } ( f ^ { * } ( \xi ) ) \right| + \operatorname* { s u p } _ { 0 \leq \xi \leq T } \left| \frac { 1 } { r } \mathcal { U } _ { j , L _ { j } } ^ { r } ( \lfloor r \xi \rfloor ) - \chi _ { j } ^ { * } ( \xi ) \right| \overset { \mathbb { P } } { \to } 0 .
$$

This proves the convergence of all prefill and decode states in (40) and (41).