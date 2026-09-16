# End-to-End Latency-Minimizing and Load-Balanced Request Scheduling for Edge LLM Inference in Agentic AI Services

Zhen Li, Jun Cai, Senior Member, IEEE, Haoran Gao, An Li, and Tan Li

Abstract—Large language model (LLM)-powered agentic AI services increasingly demand low-latency inference, motivating the deployment of LLMs across distributed edge servers. However, heterogeneous communication and computing capabilities, together with dynamically evolving inference states, make the edge server selection for each incoming request time-varying and tightly coupled across slots. In this paper, we investigate an online request scheduling framework for edge LLM inference that jointly minimizes long-term average end-to-end latency and regulates workload distribution across heterogeneous edge servers. Two main challenges arise in this context. First, conventional latency models cannot accurately capture the fine-grained dynamics of multi-stage LLM execution. Second, the latency consequence of a scheduling decision is observed only after request completion, making immediate decision evaluation difficult. To address these challenges, we develop a cross-slot inference model that captures transmission, prefill, iteration-level decoding, and key-value (KV) cache evolution for each diverse request, and characterize server workload through a KV cache memory-time consumption metric. We propose the LYREO approach that transforms the long-term load-balancing constraint via Lyapunov optimization and employs reward redistribution with sequencebased return prediction to convert delayed outcomes into timely learning signals for earlier decisions. Simulations under various configurations demonstrate that LYREO consistently achieves lower latency and more balanced load distribution than representative learning-based and heuristic baseline schemes.

Index Terms—Agentic AI, LLM inference, request scheduling, load balancing, reward redistribution.

## I. INTRODUCTION

GENTIC artificial intelligence (agentic AI) is emerging as a key paradigm for intelligent networks, where autonomous applications continuously perceive environments, reason, and act with limited human intervention [1]. As the core intelligence component of agentic AI systems, large language models (LLMs) provide the reasoning and planning capabilities required for goal-directed autonomy. Unlike conventional single-invocation applications, agentic AI repeatedly interacts with LLMs through observe-think-act cycles, so each invocation sits on the closed loop’s critical path. A delayed response directly postpones the next action, making inference latency a key determinant of system responsiveness [2]. Typically, LLM inference is hosted on remote cloud servers, but repeatedly forwarding inference requests to the cloud incurs substantial wide-area transmission delay, making agentic services vulnerable to network congestion and unstable connec-

tivity. Edge deployment of LLMs physically shortens transmission paths, mitigating cloud-side queuing and contention [3]. However, when multiple agentic AI applications share edgehosted LLMs, their requests inevitably overlap in time and vary widely in input/output lengths and latency expectations, competing for edge servers’ heterogeneous communication, computation, and memory resources. Since the end-to-end latency is shaped by fluctuating wireless conditions, disparate server capacities and states, and request-specific characteristics, edge inference scheduling must be request-dependent, state-aware, and time-varying, rather than static and one-shot.

Although recent studies advanced edge LLM inference through efficient batching, task offloading, and resource allocation [4]–[6], critical issues closely tied to practical edge LLM inference remain insufficiently explored. On one hand, unlike conventional edge-computing requests abstracted as atomic tasks completed within a single scheduling slot, LLM inference requests may remain active across multiple slots. During this period, subsequent requests continue to be scheduled and arrive, causing their service processes to overlap and interact through shared edge server resources. For example, an edge server supporting multiple agentic-AI-driven mobile robots may receive a new reasoning request from one robot while still generating a response for another. Admitting the new request changes the allocation of shared inference resources, thereby affecting the completion time of ongoing inference. Such interactions persist across scheduling slots, contrasting with conventional scheduling paradigms [7]–[9], where decision outcomes are immediately observed to guide the decision in the next time slot. Therefore, LLM scheduling exhibits crossrequest and cross-slot effects that most studies fail to capture: assigning a new request without accounting for unfinished ones prolongs latency, but this impact becomes observable only after the affected requests complete several slots later. This requires the scheduler to make sequential decisions before the outcomes of previous ones are revealed.

On the other hand, many inference scheduling studies prioritize latency while overlooking load imbalance among edge servers. A scheduler may continuously route requests to a high-capability server, gradually pushing it toward resource limits. For LLM serving, such saturation is particularly risky because each admission decision creates a stateful resource commitment on the edge server. Since a request’s resource demand continuously grows during response generation, redirecting new arrivals cannot immediately relieve an already saturated server. This concentration leaves servers vulnerable to lengthy requests, traffic bursts, or scheduling errors, and such localized pressure could be amplified into system-wide taillatency inflation and throughput loss, degrading the quality-ofservice (QoS) of heavily loaded servers while leaving lightly loaded ones under-utilized [10], [11]. Load balancing is thus not merely about resource utilization, but a vital long-term risk dispersion requirement missed by latency-only schedulers.

However, edge LLM inference scheduling introduces three key challenges. First, evaluating end-to-end latency requires modeling a request’s complete lifecycle across wireless transmission and the inference pipeline, including batching, prefill, and decoding stages. Emerging LLM serving techniques such as continuous batching [12], [13] further allow requests to dynamically join or leave active batches, introducing complex dependencies that make realistic LLM inference substantially harder to model and analyze than conventional edge tasks. Second, conventional indicators like queue length cannot fully reflect actual workloads, as requests with diverse input/output token lengths consume varying inference resources over different duration. It is necessary to map this resource consumption to a workload metric that also accounts for heterogeneous server capabilities [14]. Load balancing therefore relies on a metric capable of jointly capturing the intensity and duration of resource occupation across diverse requests and edge servers. Third, jointly addressing delayed latency feedback and longterm load balancing poses a further challenge. The end-to-end latency impact of a scheduling decision remains unobservable until the inference request completes, leaving the scheduler without a timely learning signal [15]. Meanwhile, load balancing depends on workloads accumulated across slots and cannot be enforced by penalizing isolated decisions. This mismatch substantially complicates scheduler design.

To address these challenges, in this paper, we model each request’s inference lifecycle and server-side resource occupation at a fine granularity, formulating an online scheduling problem that minimizes end-to-end latency subject to peak-memory feasibility and long-term load-balancing constraints. This problem is difficult to solve due to its longterm constraints, request-server heterogeneity, and cross-slot delayed feedback. We propose a Lyapunov-guided rewardredistribution online request scheduling (LYREO) approach that jointly addresses these difficulties through constraintaware long-term guidance and delayed-feedback-aware policy learning. The main contributions are summarized as follows.

• We develop a fine-grained model that captures edge LLM inference throughout transmission, batching, prefill, and iteration-level decoding. The model explicitly tracks the evolution of the key-value (KV) cache occupation on each edge server driven by unfinished requests and continuous batch updates, thereby quantifying how each assignment reshapes serving conditions and end-to-end latency.

• Building on this KV cache trajectory, we characterize each server’s workload using a normalized KV cache memory-time consumption metric that jointly reflects memory intensity and active duration. Based on this metric, we formulate the long-term end-to-end latencyminimization problem subject to peak memory feasibility and inter-server load-balancing constraints.

• We put forth a novel approach, named LYREO, to address this optimization problem. LYREO leverages Lyapunov virtual queues to adaptively satisfy long-term loadbalancing constraints. To effectively learn from delayed feedback, LYREO employs a long short-term memory (LSTM)-based reward redistribution mechanism, supported by sequence truncation and value bootstrapping, to provide timely learning signals for online policy learning.

• We evaluate the proposed LYREO approach across diverse system and algorithm configurations. The results show consistent improvements in end-to-end latency and load-balancing deviation over the baseline approaches.

The rest of this paper is organized as follows. Section II reviews related work and highlights the novelty of this paper. Section III presents the system model and problem formulation. Section IV presents LYREO and provides its theoretical analysis. Section V reports and discusses the evaluation results, and Section VI concludes the paper.

## II. RELATED WORK

Recently, numerous studies have improved LLM inference serving through batching, scheduling, and memory management. Orca [12] introduced iteration-level scheduling, which executes LLM inference at the granularity of individual tokengeneration iterations rather than complete requests. vLLM [13] introduced PagedAttention to dynamically manage the KV cache of requests with varying lengths. Together, these techniques establish the foundation of continuous batching, substantially improving GPU utilization and latency for agentic AI applications [16]. Nonetheless, they primarily optimize inference engines or tightly connected accelerator clusters, leaving wireless transmission, time-varying user connectivity, and request assignment across geographically distributed edge servers outside the scheduling process.

Some recent studies have extended LLM inference to mobile edge networks. The wireless edge LLM inference framework in [17] integrates request batching and model quantization, and jointly optimizes batch scheduling and resource allocation to maximize inference throughput under heterogeneous latency and accuracy requirements. In [6], a two-timescale framework was introduced to coordinate slow-timescale LLM deployment with fast-timescale batch scheduling, GPU resource allocation, and bandwidth allocation in edge-cloud networks to minimize energy cost and end-to-end latency. Additionally, [18] proposed a collaborative inference framework that jointly selects distributed devices and partitions the LLM into deployable shards to reduce inference latency and improve throughput. Nevertheless, these studies generally treat each inference request as a conventional atomic task with simplified latency models, failing to capture the multi-stage and cross-slot dynamics of practical LLM serving. Load balancing remains comparatively underexplored in edge LLM inference. A recent framework in [19] combined workload prediction with reinforcement learning in scheduling to reduce inference latency and balance GPU utilization across edge LLM instances. CoLLM [20] enabled collaborative LLM inference across resource-constrained devices to reduce inference latency while balancing energy consumption. However, these load-aware methods typically overlook time-integrated resource occupation, and long-term load balancing of heterogeneous requests and edge servers remains largely unexplored.

![](images/94b7262e16c025d86394f60b2bcfbd5e9602fdf441063682013d422def8061aa.jpg)  
Fig. 1: System model of the edge-assisted LLM inference framework.

To improve performance in edge LLM inference, recent studies have widely applied deep reinforcement learning (DRL) [21]–[23] and Lyapunov optimization [24], [25] to coordinate request scheduling, offloading, and resource allocation. Nonetheless, these studies rely on per-slot system observations and immediately available performance signals, without explicitly addressing the delayed nature of inference feedback. Consequently, how to attribute delayed outcomes to their causal scheduling decisions remains an open problem.

Unlike prior studies, this paper jointly models the cross-slot, multi-stage execution of edge LLM inference and characterizes the resulting edge server workload. Furthermore, we develop a Lyapunov-guided reward-redistribution online scheduling approach that resolves the delayed feedback while maintaining long-term load balance across heterogeneous edge servers. To our knowledge, these coupled issues have not been jointly addressed in previous work.

## III. SYSTEM MODEL & PROBLEM FORMULATION

In this section, we first present an overview of the considered edge-assisted LLM inference system in Section III-A. Sections III-B–III-D then detail the communication, LLM inference, and memory models, respectively. Finally, Section III-E formulates the corresponding optimization problem.

## A. System Overview

As illustrated in Fig. 1, the edge LLM-based agentic AI system comprises an LLM service scheduler, multiple mobile users, and a set of edge servers denoted by $\mathcal { S } = \{ 1 , 2 , \dots , S \}$ The system operates in equal discrete time slots $t \in \mathcal { T } =$ $\{ 1 , \ldots , T \}$ of length $\Delta$ seconds. Let $\mathcal { U } = \{ 1 , \dots , U \}$ denote the set of subscribed mobile users with time-varying locations. These mobile users represent devices running LLM-powered agentic applications, whose reasoning invocations are served by shared edge-hosted models. Operating at the inferenceservice layer, we represent these application-generated invocations using a slotted request model. Specifically, at the beginning of each time slot t, every user $u \in \mathcal { U }$ generates one inference request, indexed by $i = ( u , t )$ . The inference requests generated by users in time slot t are collected in set $\mathcal { Q } ( t ) = \{ ( u , t ) \mid u \in \mathcal { U } \}$ . Each request $i \in \mathcal { Q } ( t )$ is characterized by a tuple $\langle l _ { i } ^ { \mathrm { i n } } , l _ { i } ^ { \mathrm { o u t } } , o _ { i } \rangle$ , where $l _ { i } ^ { \mathrm { i n } } \in \mathbb { Z } ^ { + }$ and $\hat { l } _ { i } ^ { \mathrm { o u t } } \in \mathbb { Z } ^ { + }$ are the input and output token lengths, respectively. Here, $\hat { l } _ { i } ^ { \mathrm { { o u t } } }$ is treated as a known input, whether user-specified or estimated via existing methods [21]. $\mathbf { \sigma } _ { o _ { i } } \in \mathbb { R } ^ { 2 }$ denotes the geographical coordinates of the user when generating request i.

The scheduling decision is denoted by $x _ { i , s } ( t ) \in \{ 0 , 1 \}$ where $x _ { i , s } ( t ) = 1$ indicates that request $i \in \mathcal { Q } ( t )$ is assigned to edge server $s ,$ and $x _ { i , s } ( t ) = 0$ otherwise. The decisions are made by the service scheduler at the beginning of each time slot, based on the collected request metadata. To reflect practical LLM inference systems, we do not assume edge servers are idle at scheduling time, i.e., they may still be processing backlogged requests from previous time slots. Let $\mathcal { Q } _ { s } ( t ) = \{ i \in \mathcal { Q } ( t ) \ | \ x _ { i , s } ( t ) = 1 \}$ be the subset of requests scheduled to edge server s in time slot t. These requests are then transmitted to the edge servers and processed accordingly, as detailed in Sections III-B and III-C. Upon completion, the inference results are returned to the users.

## B. Communication Model

Following the scheduling decisions, users transmit their input request tokens to the assigned edge servers via wireless links. Let $B _ { s }$ denote the total bandwidth of edge server s, which is equally shared among all requests scheduled to it. For request $i \in \mathcal { Q } _ { s } ( t )$ , the transmission rate is given by

$$
R _ { i , s } ( t ) = \frac { B _ { s } } { | \mathcal { Q } _ { s } ( t ) | } \log _ { 2 } \left( 1 + \frac { p _ { i } h _ { i , s } ( t ) } { N _ { 0 , s } B _ { s } / | \mathcal { Q } _ { s } ( t ) | } \right) ,\tag{1}
$$

where $N _ { 0 , s }$ is the noise power spectral density (PSD) at edge server s, and $p _ { i }$ denotes the transmission power of the user generating request i. $h _ { i , s } ( t )$ denotes the channel gain between the user and edge server s, modeled as a function of the distance $\left\| o _ { i } - o _ { s } \right\|$ [26], where $\pmb { o } _ { s }$ is the location of edge server s. Then, the transmission delay of request i is given by

$$
D _ { i , s } ^ { \mathrm { t x } } ( t ) = \frac { \beta l _ { i } ^ { \mathrm { i n } } } { R _ { i , s } ( t ) } ,\tag{2}
$$

where $\beta$ (in bits) is the data size of a single token.

Since inference computation begins only after all scheduled prompts have been received [8], let $\begin{array} { r l } { D _ { s } ^ { \mathrm { s y } } ( t ) } & { { } = } \end{array}$ $\operatorname* { m a x } _ { i \in \mathcal { Q } _ { s } ( t ) } D _ { i , s } ^ { \operatorname { t x } } ( t )$ be the synchronization delay for the batch at edge server s. Correspondingly, the waiting delay for request i is determined by the difference between this synchronization delay and its transmission time, which is given by

$$
D _ { i , s } ^ { \mathrm { w a } } ( t ) = D _ { s } ^ { \mathrm { s y } } ( t ) - D _ { i , s } ^ { \mathrm { t x } } ( t ) .\tag{3}
$$

Given that edge servers are equipped with significantly higher transmit power and wider downlink bandwidth compared to mobile users, the feedback latency after inference completion is considered negligible in this work.

## C. LLM Inference Model

Modern LLMs rely on decoder-only transformer architectures, comprising prefill and decoding phases. In contrast to previous studies that simplify this process via static batching and single-slot completion, this work incorporates a highly realistic serving framework driven by continuous batching and iteration-level dynamics. The two phases are detailed below.

1) Prefill Phase: In the prefill phase, the input tokens are processed to compute the first output token and generate the corresponding KV cache. To fully exploit the parallel computing capability of GPUs, the tokens from multiple requests can be aggregated and processed simultaneously through largescale matrix multiplications [13]. Accordingly, all requests arriving at edge server s within time slot t, i.e., Q (t), are grouped into a single batch for one forward-pass computation, known as batched prefill [17]. Since prefill is inherently compute-bound with relatively stable throughput, we assume its computation is independent of decoding tasks as long as sufficient KV cache memory is provisioned. Let $v _ { s } ^ { \mathrm { p r e } }$ (in tokens/s) be the prefill rate of edge server s. The prefill latency of the batched requests scheduled to edge server s in time slot t is determined by the total number of input tokens and the edge server’s computational capability, expressed as [6], [27]

$$
\begin{array} { r } { D _ { s } ^ { \mathrm { p r e } } ( t ) = \frac { \sum _ { i \in \mathcal { Q } _ { s } ( t ) } l _ { i } ^ { \mathrm { i n } } } { v _ { s } ^ { \mathrm { p r e } } } . } \end{array}\tag{4}
$$

2) Decoding Phase: In the decoding phase, the LLM generates output tokens autoregressively, producing one token per iteration until the output length is reached. At each iteration, the inference engine takes the previously generated token as input and performs computation based on the KV cache of all preceding tokens, which is updated incrementally as decoding proceeds. Unlike prefill, decoding iterations cannot be parallelized due to strict data dependencies. Consequently, despite minimal computation, decoding remains highly memorybound because it requires repeatedly loading model weights and the accumulated KV cache. We consider an advanced decoding mechanism enabled by iteration-level scheduling termed dynamic entry-and-exit decoding (DEED) [12], featuring: 1) early-finished eviction, where completed requests are removed from the batch and their KV cache is released immediately from memory at the end of iteration; and 2) latejoining admission, where new requests are admitted to the active batch and decoded together with backlogged requests, provided sufficient KV cache capacity is available.

Based on the DEED mechanism, let $\mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t )$ be the set of backlogged requests remaining at the edge server s at the beginning of time slot t. Since the decoding phase typically dominates the end-to-end serving latency while the prefill latency is comparatively short [21], we assume, for a reasonable setting of $\Delta ,$ all backlogged requests in $\mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t )$ have already completed the prefill phase (including transmission)

and remain only in the decoding phase. Let $l _ { j } ^ { \mathrm { g e n } } ( t )$ be the cumulative number of generated tokens for request $j \in \mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t )$ at the beginning of slot t. While newly scheduled requests undergo transmission and prefill, the inference engine continues decoding these backlogged requests in $\mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t )$ . Specifically, during the time budget $\bar { D } _ { s } ^ { \mathrm { s y } } ( t ) + \bar { D } _ { s } ^ { \mathrm { p r e } } ( t )$ , the KV cache memory occupied on the GPU of edge server s increases linearly with the generated tokens [16]. The KV cache memory occupation at the k-th decoding iteration in time slot t on edge server s can be derived as

$$
m _ { s } ^ { \mathrm { I } } ( k , t ) = \alpha \sum _ { j \in \mathcal { G } _ { s } ( k , t ) } ( l _ { j } ^ { \mathrm { i n } } + l _ { j } ^ { \mathrm { g e n } } ( t ) + k ) ,\tag{5}
$$

where α is the per-token KV cache $\mathrm { s i z e } ^ { 1 }$ , and $\mathcal { G } _ { s } ( k , t ) = \{ j \in$ $\mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t ) \mid k \leq \hat { l } _ { j } ^ { \mathrm { o u t } } - l _ { i } ^ { \mathrm { g e n } } ( t ) \}$ denotes the set of active background requests at the k-th decoding iteration. The condition $k \ \leq$ $\hat { l } _ { i } ^ { \mathrm { o u i } } - l _ { i } ^ { \mathrm { g e n } } ( t )$ in $\mathcal { G } _ { s } ( k , t )$ reflects the early-finished eviction of DEED, where the request is removed from the summation as its KV cache is released once decoding is complete.

Denote the decoding throughput of edge server s (in steps/s) by $v _ { s } ^ { \mathrm { d e c } } ( \cdot )$ , which is a function of the current total KV cache size. The time per decoding iteration on edge server s when $m _ { s } ^ { \operatorname { I } } ( k , t ) > 0$ is given by

$$
\tau _ { s } ^ { \mathrm { I } } ( k , t ) = \frac { 1 } { ( 1 - w \mathbb { I } _ { s } ( k , t ) ) \cdot v _ { s } ^ { \mathrm { d e c } } ( m _ { s } ^ { \mathrm { I } } ( k , t ) ) } ,\tag{6}
$$

where $\mathbb { I } _ { s } ( k , t ) \ \in \ \{ 0 , 1 \}$ indicates the operational status of inference engine. Specifically, $\mathbb { I } _ { s } ( k , t ) = 0$ when the server is in a pure decoding state, while $\mathbb { I } _ { s } ( k , t ) = 1$ indicates the presence of concurrent prefill computation, which reduces the throughput by a fraction $w \in \mathsf { \Gamma } ( 0 , 1 )$ . Building on the periteration decoding time, the number of decoding iterations that can be completed on edge server s before the prefill phase finishes in time slot t can be determined by

$$
\begin{array} { r l r } & { } & { \tilde { k } _ { s } ( t ) = \operatorname* { m a x } \Bigl \{ k \Big | \displaystyle \sum _ { \kappa = 1 } ^ { k } \tau _ { s } ^ { \mathrm { I } } ( \kappa , t ) \le D _ { s } ^ { \mathrm { s y } } ( t ) + D _ { s } ^ { \mathrm { p r e } } ( t ) , } \\ & { } & { m _ { s } ^ { \mathrm { I } } ( k , t ) > 0 \Bigr \} , } \end{array}\tag{7}
$$

where the condition $m _ { s } ^ { \operatorname { I } } ( k , t ) > 0$ ensures that the iteration counts only when there are backlogged requests to be decoded.

Once the prefill phase finishes, the newly prefilled requests enter the decoding batch immediately through the late-joining admission mechanism of DEED<sup>2</sup>. Let $\mathcal { H } _ { s } ( k , t ) = \smash  \bigl \{ i \in \mathcal { Q } _ { s } ( t )$ $k - \tilde { k } _ { s } ( t ) \leq \hat { l } _ { i } ^ { \mathrm { o u t } } \}$ denote the set of active new requests at the k-th decoding iteration. Therefore, the total KV cache size on edge server s can be calculated as

$$
\begin{array} { r l } { m _ { s } ^ { \mathrm { I I } } ( k , t ) = \alpha \displaystyle \sum _ { i \in \mathcal { H } _ { s } ( k , t ) } ( l _ { i } ^ { \mathrm { i n } } + k - \tilde { k } _ { s } ( t ) ) } & { } \\ { + \alpha \displaystyle \sum _ { j \in \mathcal { G } _ { s } ( k , t ) } ( l _ { j } ^ { \mathrm { i n } } + l _ { j } ^ { \mathrm { g e n } } ( t ) + k ) , } & { } \end{array}\tag{8}
$$

where the first term accounts for the KV cache of active newly scheduled requests, and the second term accounts for that of active backlogged requests. Note that both the $m _ { s } ^ { \mathrm { I } } ( k , t )$ and $m _ { s } ^ { \mathrm { I I } } ( k , t )$ are not necessarily monotonic with respect to k, as the early-finished eviction in DEED continuously removes completed requests and releases their KV cache, while the latejoining admission of new requests and the token generation of active requests increase the KV cache. Given the total KV cache size, the per-iteration decoding time and the number of iterations completed between prefill and the end of time slot t on edge server s can be respectively expressed as

$$
\tau _ { s } ^ { \mathrm { I I } } ( k , t ) = \frac { 1 } { v _ { s } ^ { \mathrm { d e c } } ( m _ { s } ^ { \mathrm { I I } } ( k , t ) ) } ,\tag{9}
$$

$$
\begin{array} { r } { \bar { k } _ { s } ( t ) = \operatorname* { m a x } \Bigl \{ k \ \Big | \ \displaystyle \sum _ { \kappa = 1 } ^ { k } \tau _ { s } ^ { \mathrm { I I } } ( \tilde { k } _ { s } ( t ) + \kappa , t ) \leq \Delta - D _ { s } ^ { \mathrm { s y } } ( t ) } \\ { - D _ { s } ^ { \mathrm { p r e } } ( t ) , \ m _ { s } ^ { \mathrm { I I } } ( \tilde { k } _ { s } ( t ) + k , t ) > 0 \Bigr \} . } \end{array}\tag{10}
$$

Accordingly, for request i scheduled to edge server s in time slot t, the decoding latency is calculated by accumulating the per-iteration decoding latency over all output tokens, given by

$$
D _ { i , s } ^ { \mathrm { d e c } } ( t ) = \sum _ { k = 1 } ^ { \hat { l } _ { i } ^ { \mathrm { o u t } } } \tau _ { i , s } ^ { \prime } ( k ) ,\tag{11}
$$

where $\tau _ { i , s } ^ { \prime } ( k )$ denotes the decoding time when generating the k-th output token of request i. Note that $\tau _ { s } ^ { \mathrm { I } } ( k , t ) , \tau _ { s } ^ { \mathrm { I I } } ( k , t )$ , and $\tau _ { i , s } ^ { \prime } ( k )$ represent the same physical quantity viewed from two different perspectives: $\tau _ { i , s } ^ { \prime } ( k )$ tracks the per-iteration decoding time from the perspective of an individual request, whereas $\tau _ { s } ^ { \mathrm { I } } ( k , t )$ and $\tau _ { s } ^ { \mathrm { I I } } \bar { ( k , t ) }$ characterize it along the system timeline in time slot t. Since the per-iteration decoding time along the system timeline has been fully derived in (6) and $( 9 ) , \tau _ { i , s } ^ { \prime } ( k )$ can be directly obtained via the corresponding cross-slot index mapping, which is omitted here for brevity.

For edge server $s ,$ the total number of completed decoding iterations within time slot t is the sum of iterations from both phases, which is given by $K _ { s } ( t ) = \tilde { k } _ { s } ( t ) + \bar { k } _ { s } ( t )$ . Therefore, the set of backlogged requests at the beginning of time slot t + 1 is updated as

$$
\begin{array} { r l } & { \mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t + 1 ) = \{ i \in \mathcal { Q } _ { s } ( t ) \mid \hat { l } _ { i } ^ { \mathrm { o u t } } > \bar { k } _ { s } ( t ) \} } \\ & { \cup \{ j \in \mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t ) \mid \hat { l } _ { j } ^ { \mathrm { o u t } } > l _ { j } ^ { \mathrm { g e n } } ( t ) + K _ { s } ( t ) \} , } \end{array}\tag{12}
$$

where the first term represents the newly scheduled requests in ${ \mathcal { Q } } _ { s } ( t )$ that have not been completed by the end of time slot t, and the second term represents the previously backlogged requests in $\mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t )$ that remain unfinished. In contrast to the conventional “clear-then-schedule” assumption, here we do not require all requests to be completed within the current time slot, which more faithfully reflects the realistic operation of LLM inference systems.

## D. Memory Analysis

Unlike traditional computing tasks, LLM serving, especially the dominant autoregressive decoding, is inherently stateful and frequently bottlenecked by memory-resource availability. Since the KV cache footprint strongly affects the achievable decoding throughput and serving capacity, KV cache occupation is explicitly modeled as a key resource state [13], [28].

Based on the system workflow described above, the KV cache occupation at the k-th decoding iteration in time slot t on edge server s can be unified as

$$
\begin{array} { r } { m _ { s } ( k , t ) = \left\{ \begin{array} { l l } { m _ { s } ^ { \mathrm { I } } ( k , t ) , ~ 1 \leq k \leq \tilde { k } _ { s } ( t ) , } \\ { m _ { s } ^ { \mathrm { I I } } ( k , t ) , ~ \tilde { k } _ { s } ( t ) < k \leq K _ { s } ( t ) . } \end{array} \right. } \end{array}\tag{13}
$$

Let M<sub>s</sub> be the GPU memory capacity of edge server s reserved for KV cache during LLM inference. To ensure feasibility, the peak KV cache occupation must not exceed the available memory capacity throughout the inference process,

$$
M _ { s } ^ { \mathrm { p k } } ( t ) = \operatorname* { m a x } _ { k = 1 , \ldots , K _ { s } ( t ) } m _ { s } ( k , t ) \leq M _ { s } .\tag{14}
$$

Note that the peak memory constraint defined above naturally unifies two feasibility requirements. First, it ensures that the KV cache memory does not overflow during the autoregressive decoding phase as tokens are incrementally generated. Second, it acts as an admission feasibility condition for newly scheduled requests, ensuring that sufficient GPU memory is available for the KV cache generated during prefill. As a result, admitted requests can start the prefill phase immediately after batching, without incurring additional engine-level waiting.

In addition to acting as a hard feasibility constraint, the KV cache occupation on edge servers reflects the continuous inference workload, as it remains persistently allocated throughout the inference lifecycle and continuously consumes GPU memory resources. Accordingly, we define the workload of edge server s in time slot t as the normalized KV cache memory-time consumption, i.e.,

$$
\eta _ { s } ( t ) = \frac { \sum _ { k = 1 } ^ { K _ { s } ( t ) } m _ { s } ( k , t ) \tau _ { s } ( k , t ) } { M _ { s } \Delta } ,\tag{15}
$$

where the unified per-iteration decoding time at the k-th iteration on edge server s within time slot t is given by

$$
\tau _ { s } ( k , t ) = \left\{ \tau _ { s } ^ { \mathrm { I } } ( k , t ) , \quad 1 \leq k \leq \tilde { k } _ { s } ( t ) , \right.\tag{16}
$$

In (15), the numerator accumulates the memory occupied at each decoding iteration weighted by its duration, measuring the cumulative KV residency incurred by the inference engine, while the denominator represents the maximum available memory-time capacity of the edge server within a slot. By jointly integrating the spatial and temporal dimensions of KV cache occupation, $\eta _ { s } ( t )$ effectively captures the KV-centric workload pressure, particularly across heterogeneous edge servers. In the spatial dimension, normalizing the occupation by the total capacity $M _ { s }$ ensures that the same KV cache size imposes a much heavier load on edge servers with scarce memory. In the temporal dimension, under an identical KV cache state, edge servers with weaker processing capabilities exhibit longer per-iteration decoding times $\tau _ { s } ( k , t )$ , incurring a larger value of the memory-time consumption term in our metric. This dual sensitivity allows $\eta _ { s } ( t )$ to characterize the heterogeneous load distribution, providing a tractable and interpretable indicator of LLM inference workload.

## E. Problem Formulation

In summary, the end-to-end latency for request i scheduled to edge server s in time slot $t ,$ consisting of the transmission,

batch waiting, prefill, and decoding latency, is formulated as

$$
D _ { i , s } ^ { \mathrm { t o t } } ( t ) = D _ { i , s } ^ { \mathrm { t x } } ( t ) + D _ { i , s } ^ { \mathrm { w a } } ( t ) + D _ { s } ^ { \mathrm { p r e } } ( t ) + D _ { i , s } ^ { \mathrm { d e c } } ( t ) .\tag{17}
$$

Beyond latency, load balance across edge servers is also considered to maintain the workload deviation within acceptable bounds. To ensure long-term load balance, the variancebased workload deviation of each edge server $s \in S$ from the system average must satisfy the following constraint

$$
\operatorname* { l i m } _ { T  \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } ( \eta _ { s } ( t ) - \bar { \eta } ( t ) ) ^ { 2 } \leq \epsilon , \forall s \in \mathcal { S } ,\tag{18}
$$

where $\begin{array} { r } { \bar { \eta } ( t ) ~ = ~ \frac { 1 } { S } \sum _ { s \in \mathcal { S } } \eta _ { s } ( t ) } \end{array}$ denotes the average workload over all edge servers in time slot t and ϵ denotes the predefined long-term tolerance threshold.

With the objective of minimizing the long-term timeaverage end-to-end latency of all inference requests while ensuring load balance across edge servers, we formulate an online optimization problem over the scheduling decision $\mathbf { \boldsymbol { x } } ( t ) ~ = ~ \{ \boldsymbol { x } _ { i , s } ( t ) ~ | ~ \forall i ~ \in ~ \mathcal { Q } ( t ) , \forall s ~ \in ~ \mathcal { S } \}$ for each time slot $t \in \mathcal T$ . The problem is formally stated as

$$
\mathbf { P 1 } : \operatorname* { m i n } _ { \{ \pmb { x } ( t ) \} } \operatorname* { l i m } _ { T \to \infty } \frac { 1 } { T } \sum _ { t = 1 } ^ { T } \mathbb { E } \Big [ \sum _ { i \in \mathcal { Q } ( t ) } \sum _ { s \in \mathcal { S } } x _ { i , s } ( t ) D _ { i , s } ^ { \mathrm { t o t } } ( t ) \Big ] ,\tag{19a}
$$

s.t. (18),

$$
M _ { s } ^ { \mathrm { p k } } ( t ) \le M _ { s } , \forall s \in \mathcal { S } , \forall t \in T ,\tag{19b}
$$

$$
x _ { i , s } ( t ) \in \{ 0 , 1 \} , \forall i \in \mathcal { Q } ( t ) , \forall s \in \mathcal { S } , \forall t \in \mathcal { T } ,\tag{19c}
$$

$$
\sum _ { s \in { \cal S } } x _ { i , s } ( t ) = 1 , \forall i \in \mathcal { Q } ( t ) , \forall t \in \mathcal { T } ,\tag{19d}
$$

where constraint (18) specifies the long-term average loadbalancing constraint. Constraint (19b) ensures that the KV cache occupation at each edge server does not exceed its available memory capacity, guaranteeing the feasibility of LLM inference execution. Constraints (19c) and (19d) enforce the binary nature of the scheduling variables and the unique assignment of each request to an edge server, respectively.

Solving problem P1 directly is nontrivial due to the following complexities. Fundamentally, the discrete scheduling decisions $( \mathrm { i } . \mathrm { e } . , \ x ( t ) )$ make the problem a nonlinear integer programming problem, which is known to be NP-hard. In addition, the attributes of inference requests (i.e., input/output token lengths) and the locations of mobile users vary over time in an a priori unknown manner, resulting in fluctuating inference workloads and time-varying channel conditions. Meanwhile, the heterogeneous prefill and decoding throughputs across edge servers further complicate the overall service latency. Last but not least, the scheduling decisions are coupled across time slots. The end-to-end latency of a request (i.e., $D _ { i , s } ^ { \mathrm { t o t } } ( t ) )$ depends not only on its own scheduling decision in time slot $t ,$ but also on the system state resulting from past scheduling decisions and the future system evolution driven by subsequent scheduling decisions. This temporal coupling, compounded by the fact that $D _ { i , s } ^ { \mathrm { t o t } } ( t )$ only becomes observable after request completion, introduces strong interdependence between decisions across time, making the optimization problem particularly challenging.

## IV. LYREO APPROACH

In this section, we develop LYREO, a novel approach that treats P1 as a sequential decision-making problem and jointly addresses the challenges identified above. Specifically, we first reformulate problem P1 by leveraging Lyapunov optimization to handle the long-term load-balancing constraint. We then introduce a reward redistribution mechanism to attribute delayed latency feedback to its associated decisions before optimizing the scheduling policy. Finally, we provide the theoretical analysis of the proposed approach.

## A. Problem Reformulation via Lyapunov Optimization

Constraint (18) couples the scheduling decisions over the entire time horizon and therefore cannot be evaluated from the current time slot alone. To expose the accumulated load imbalance to each scheduling decision, we define a virtual queue for the workload deviation of each edge server, whose dynamics evolve as

$$
Z _ { s } ( t + 1 ) = \operatorname* { m a x } \{ Z _ { s } ( t ) + Y _ { s } ( t ) - \epsilon , 0 \} , s \in \mathcal { S } ,\tag{20}
$$

where $Y _ { s } ( t ) ~ = ~ ( \eta _ { s } ( t ) - \bar { \eta } ( t ) ) ^ { 2 }$ denotes the instantaneous workload deviation, and $Z _ { s } ( t )$ denotes the length of the virtual queue (with $Z _ { s } ( 0 ) = 0 ) \mathrm { { } }$ , tracking the cumulative amount by which $Y _ { s } ( t )$ exceeds the threshold ϵ. Queue stability implies that the long-term workload deviation does not exceed $\epsilon ,$ thereby satisfying constraint (18).

Collecting the virtual queues of all edge servers, we define $\pmb { Z } ( t ) = [ Z _ { 1 } ( t ) , Z _ { 2 } ( t ) , \ldots , Z _ { S } ( t ) ]$ as the queue backlog vector in time slot t. To characterize how well the scheduling adheres to long-term load balancing, we define the following quadratic Lyapunov function [29]

$$
\mathcal { L } ( Z ( t ) ) \triangleq \frac { 1 } { 2 } \sum _ { s \in \mathcal { S } } Z _ { s } ( t ) ^ { 2 } .\tag{21}
$$

A small value of $ { \mathcal { L } } ( Z ( t ) )$ indicates that all virtual queues are close to zero, i.e., the long-term constraint is well satisfied, and thus the system should aim to keep $\mathcal { L } ( Z ( t ) )$ small. Then, the conditional Lyapunov drift is given by

$$
\Delta ( Z ( t ) ) \triangleq \mathbb { E } \left[ \mathcal { L } ( Z ( t + 1 ) ) - \mathcal { L } ( Z ( t ) ) \ | \ Z ( t ) \right] ,\tag{22}
$$

which represents the expected change in the Lyapunov function over time slots, and a smaller drift indicates a more stable queue. However, according to (22), the Lyapunov drift still depends on the system state in the next time slot, making it intractable to compute directly. To avoid relying on future system information, we derive an upper bound on the Lyapunov drift, as provided in Lemma 1.

Lemma 1. The Lyapunov drift $\Delta ( Z ( t ) )$ is upper bounded by

$$
\Delta ( Z ( t ) ) \le \Gamma + \mathbb { E } \Big [ \sum _ { s \in { \cal S } } Z _ { s } ( t ) \left( Y _ { s } ( t ) - \epsilon \right) \mid Z ( t ) \Big ] ,\tag{23}
$$

where $\begin{array} { r } { \Gamma \geq \left( \sum _ { s \in { \mathcal { S } } } \mathbb { E } [ ( Y _ { s } ( t ) - \epsilon ) ^ { 2 } ] \right) / 2 } \end{array}$ is a positive constant that bounds the expected squared terms over all time slots.

Proof. Squaring both sides of the virtual queue update in (20) and using the fact that $\{ [ x ] ^ { + } \} ^ { 2 } \leq x ^ { 2 }$ for any x gives

$$
\begin{array} { r l r } {  { Z _ { s } ( t + 1 ) ^ { 2 } = \big \{ [ Z _ { s } ( t ) + Y _ { s } ( t ) - \epsilon ] ^ { + } \big \} ^ { 2 } } } \\ & { } & { \leq ( Z _ { s } ( t ) + Y _ { s } ( t ) - \epsilon ) ^ { 2 } , ~ s \in \mathcal { S } . } \end{array}\tag{24}
$$

Summing (24) over all $s \in S$ and dividing by 2 yields

$$
\begin{array} { l } { \displaystyle \frac 1 2 \sum _ { s \in { \mathcal S } } Z _ { s } ( t + 1 ) ^ { 2 } \leq \displaystyle \frac 1 2 \sum _ { s \in { \mathcal S } } Z _ { s } ( t ) ^ { 2 } + \displaystyle \frac 1 2 \sum _ { s \in { \mathcal S } } ( Y _ { s } ( t ) - \epsilon ) ^ { 2 } } \\ { + \sum _ { s \in { \mathcal S } } Z _ { s } ( t ) ( Y _ { s } ( t ) - \epsilon ) . } \end{array}\tag{25}
$$

Subtracting $\begin{array} { r } { \left( \sum _ { s \in \mathcal { S } } Z _ { s } ( t ) ^ { 2 } \right) / 2 } \end{array}$ from both sides and incorporating the conditional expectation with respect to $\mathbf { } Z ( t )$ , the upper bound for $\Delta ( Z ( t ) )$ can be derived as

$$
\begin{array} { r l r } {  { \Delta ( \pmb { Z } ( t ) ) = \mathbb { E } \Big [ \frac { 1 } { 2 } \sum _ { s \in \mathcal { S } } Z _ { s } ( t + 1 ) ^ { 2 } - \frac { 1 } { 2 } \sum _ { s \in \mathcal { S } } Z _ { s } ( t ) ^ { 2 } \mid \pmb { Z } ( t ) \Big ] } } \\ & { } & { \leq \mathbb { E } \Big [ \frac { 1 } { 2 } \sum _ { s \in \mathcal { S } } ( Y _ { s } ( t ) - \epsilon ) ^ { 2 } + \sum _ { s \in \mathcal { S } } Z _ { s } ( t ) ( Y _ { s } ( t ) - \epsilon ) \mid \pmb { Z } ( t ) \Big ] . } \end{array}\tag{26}
$$

Substituting the definition of Γ into the preceding inequality yields (23), which completes the proof. □

Lemma 1 provides an upper bound on the Lyapunov drift that no longer explicitly depends on the future virtual-queue state. Based on the Lyapunov optimization framework, the objective function of the original problem P1 can be rewritten as the minimization of the following drift-plus-penalty function,

$$
\begin{array} { r l } & { \Delta ( Z ( t ) ) + \nu \mathbb { E } \Big [ \displaystyle \sum _ { i \in \mathcal { Q } ( t ) } \sum _ { s \in \mathcal { S } } x _ { i , s } ( t ) D _ { i , s } ^ { \mathrm { t o t } } ( t ) \mid Z ( t ) \Big ] } \\ & { \qquad \le \Gamma + \mathbb { E } \Big [ \displaystyle \sum _ { s \in \mathcal { S } } Z _ { s } ( t ) \left( Y _ { s } ( t ) - \epsilon \right) \mid Z ( t ) \Big ] } \\ & { \qquad + \nu \mathbb { E } \Big [ \displaystyle \sum _ { i \in \mathcal { Q } ( t ) } \sum _ { s \in \mathcal { S } } x _ { i , s } ( t ) D _ { i , s } ^ { \mathrm { t o t } } ( t ) \mid Z ( t ) \Big ] , } \end{array}\tag{27}
$$

where $\nu > 0$ is a control parameter that balances the trade-off between minimizing end-to-end latency and maintaining longterm load balance. By dropping terms independent of ${ \pmb x } ( t )$ , we obtain the following per-slot reformulation of problem P1,

$$
\begin{array} { r l } { \mathbf { P 2 } : ~ \underset { \boldsymbol { x } ( t ) } { \mathrm { m i n } } ~ \mathbb { E } \Big [ \nu \displaystyle \sum _ { i \in { \mathcal { Q } } ( t ) } \displaystyle \sum _ { s \in { \mathcal { S } } } x _ { i , s } ( t ) D _ { i , s } ^ { \mathrm { t o t } } ( t ) } & { } \\ & { ~ + \displaystyle \sum _ { s \in { \mathcal { S } } } Z _ { s } ( t ) Y _ { s } ( t ) ~ | ~ Z ( t ) \Big ] , } \\ { ~ \mathrm { s . t . ~ } ( 1 9 \mathrm { b } ) , ~ ( 1 9 \mathrm { c } ) , ~ ( 1 9 \mathrm { d } ) . } \end{array}\tag{28}
$$

Note that, in problem P2, the long-term load-balancing constraint of P1 has been transformed into a per-slot corrective term. However, P2 remains intractable, mainly because the latency term $D _ { i , s } ^ { \mathrm { t o t } } ( t )$ in the objective depends on future system information. Specifically, the scheduling decision must be committed in the current time slot, whereas the resulting end-to-end latency can only be observed after inference completion, and is further influenced by the arrival and scheduling of subsequent requests assigned to the same edge server. To this end, in the next subsection, we propose a novel policy learning framework to efficiently address this challenge.

## B. Delayed-Feedback-Aware Policy Learning

1) Sequence-Markov Decision Process Formulation: Problem P2 constitutes a sequential decision-making process, as each scheduling action updates the system states encountered by subsequent decisions. Accordingly, we formulate it as a sequence-Markov decision process (SDP), represented by the tuple $\langle \mathbf { S } , \mathbf { A } , \mathbf { P } , \mathbf { R } , \gamma \rangle$ , where, unlike a standard Markov decision process, the rewards are not required to satisfy the Markov property. At each time step t, the scheduler observes the current state $\textbf { \textit { s } } \in \textbf { \textit { S } }$ and takes an action $\begin{array} { l l } { { \pmb { a } } _ { t } } & { \in { \begin{array} { l } { { \bf A } } \end{array} } } \end{array}$ which transitions the environment to a new state $\mathbf { } s _ { t + 1 }$ with probability $\operatorname* { P r } [ s _ { t + 1 } \ | \ s _ { t } , \mathbf { a } _ { t } ]$ and returns a reward $r _ { t + 1 } \in \mathbf { R }$ The discount factor $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ balances the immediate and future rewards. The detailed definitions of the state and action space, and reward function are provided as follows.

• State Space. The state represents the system information available to the scheduler in each time slot t, defined as

$$
\begin{array} { r } { \pmb { s } _ { t } = \{ \pmb { s } _ { t } ^ { \mathrm { r e q } } , \pmb { s } _ { t } ^ { \mathrm { s e r } } \} , } \end{array}\tag{29}
$$

where $\begin{array} { r l r } { { \bf { \tt { s } } } _ { t } ^ { \mathrm { { r e q } } } } & { { } = } & { \big [ l _ { i } ^ { \mathrm { { i n } } } , \hat { l } _ { i } ^ { \mathrm { { o u t } } } , { \bf { o } } _ { i } \big ] _ { i \in \mathcal { Q } ( t ) } } \end{array}$ denotes the attributes of the inference requests to be scheduled, and $\begin{array} { r l } { { \pmb { s } } _ { t } ^ { \mathrm { s e r } } } & { { } = } \end{array}$ $[ B _ { s } , v _ { s } ^ { \mathrm { p r e } } , v _ { s } ^ { \mathrm { d e c } } ( c ) , | \mathcal { Q } _ { s } ^ { \mathrm { b g } } ( t ) | , \eta _ { s } ( t \mathrm { ~ - ~ } 1 ) , Z _ { s } ( t ) ] _ { s \in S }$ denotes the server-side state information, including the attributes $( \mathrm { i . e . , }$ bandwidth, prefill and decoding throughput) and the dynamic status (i.e., the number of backlogged requests, the recent workload, and the virtual-queue length of load-balancing deviation) of each edge server. Here, $v _ { s } ^ { \mathrm { d e c } } ( c )$ denotes the decoding throughput evaluated at a constant reference occupation c, serving as an indicator of the server’s decoding capability.

• Action Space. In each time slot t, the scheduler determines the scheduling decision for all requests in $\mathcal { Q } ( t )$ . The action is defined as $\mathbf { \ } _ { \mathbf { \ } } a _ { t } = \mathbf { \ } x ( t ) = \{ x _ { i , s } ( t ) \mid \forall i \in \mathcal { Q } ( t ) , \forall s \in \mathcal { S } \}$

• Reward Function. By taking action $\mathbf { } \mathbf { a } _ { t }$ under state $\mathbf { } _ { s _ { t } } .$ , the scheduler receives a numerical reward defined according to the objective function in (28), given by

$$
r _ { t + 1 } = - \nu \sum _ { s \in S } \sum _ { i \in \mathcal { Q } _ { s } ^ { \prime } ( t ) } D _ { i , s } ^ { \mathrm { t o t } } ( t ) - \sum _ { s \in S } Z _ { s } ( t ) Y _ { s } ( t ) - \varrho \Upsilon ( t ) ,\tag{30}
$$

where $\mathcal { Q } _ { s } ^ { \prime } ( t )$ represents the set of requests that complete their inference on edge server s in time slot $t . \Upsilon ( t ) =$ $\begin{array} { r l } { \sum _ { s \in \mathcal { S } } [ M _ { s } ^ { \mathrm { p k } } ( t ) - M _ { s } ] ^ { + } } \end{array}$ is a penalty term enforcing the GPU memory constraint, and $\varrho > 0$ is the corresponding penalty coefficient. Overall, the reward jointly incorporates the total latency of requests, the load-balancing deviation captured by the virtual queue, and the memory feasibility penalty.

2) Reward Redistribution: As discussed above, the latency term in (30) introduces delayed rewards into the framework, so the scheduler cannot immediately evaluate which earlier assignment caused the eventual outcome. Inspired by [30], we introduce reward redistribution to recover a decision-level reward signal. Reward redistribution is a procedure for an SDP that redistributes the total return $\scriptstyle \sum _ { t = 0 } ^ { T } r _ { t + 1 }$ over the sequence of state-action pairs $( s _ { 0 } , { \pmb a } _ { 0 } , \dots , { \pmb s } _ { T } , { \pmb a } _ { T } )$ . Since reward redistribution preserves the cumulative return of the sequence, the expected return under any policy remains unchanged. Therefore, the original SDP and the redistributed SDP share the same optimal policy, indicating that reward redistribution does not alter the underlying optimization objective.

Following the optimal second-order Markov reward redistribution in [30]–[32], we consider a reward signal that satisfies

$$
\begin{array} { r } { \mathbb { E } [ r _ { t + 1 } ^ { \prime } \mid \chi _ { t - 1 } , \chi _ { t } ] = q ^ { \pi } ( \chi _ { t } ) - q ^ { \pi } ( \chi _ { t - 1 } ) , } \end{array}\tag{31}
$$

where ${ \boldsymbol { \chi } } _ { t } = \left( { \boldsymbol { s } } _ { t } , { \boldsymbol { a } } _ { t } \right)$ and $q ^ { \pi } ( \chi _ { t } )$ denote the state-action pair in time slot t and its corresponding Q-value under policy $\pi ,$ , respectively. Since the Q-value represents the expected cumulative return starting from a given state-action pair, Eq. (31) implies that the redistributed reward mathematically captures the increment in the expected return brought by the current state-action pair. By immediately assigning this increment to that pair, the expected future redistributed reward becomes zero, which eliminates reward delay in expectation. Consequently, this formulation provides a step-wise objective for policy learning. To illustrate this intuitively, if a state-action pair increases the cumulative return, i.e., $q ^ { \pi } ( \chi _ { t } ) > q ^ { \pi } ( \chi _ { t - 1 } )$ , the corresponding redistributed reward $r _ { t + 1 } ^ { \prime }$ becomes positive, even though this improvement may not be reflected immediately in the original delayed reward.

Although the second-order Markov reward redistribution is theoretically optimal for finite-horizon problems, it relies on predicting the cumulative return associated with each observed state-action sequence. In finite-horizon settings, the complete sequence return naturally serves as the supervision target for this prediction. However, under an infinite-horizon formulation, the complete sequence return is unavailable. To overcome this limitation, we develop a novel infinite-horizon return prediction framework by integrating sequence truncation with value bootstrapping. Specifically, instead of using the complete sequence return as the supervision target, we construct a return target that combines the actual rewards observed up to a truncation point H with a bootstrapped estimate for the remaining horizon. We define this return target as

$$
\hat { y } _ { H } \triangleq \sum _ { h = 0 } ^ { H - 1 } \gamma ^ { h } r _ { h + 1 } + \gamma ^ { H } V _ { \phi } ( \pmb { s } _ { H } ) ,\tag{32}
$$

where the first term is the cumulative discounted reward over the first H steps, i.e., the original infinite-horizon trajectory is truncated after H steps, while the second term is a bootstrap estimate of the remaining discounted return, provided by the value network introduced later.

Since this return prediction requires estimating the expected return conditioned on variable-length state-action sequences, we employ an LSTM network, parameterized by $\psi$ and denoted by $g _ { \psi } ( \cdot )$ , which is well-suited for modeling such sequential dependencies. Given the sequence observed up to time slot t, the network outputs a prediction

$$
g _ { t } \triangleq g _ { \psi } ( \chi _ { 0 : t } ) \approx { \mathbb E } [ \hat { y } _ { H } \ | \ \chi _ { 0 } , \dots , \chi _ { t } ] .\tag{33}
$$

The network $g _ { \psi } ( \cdot )$ is then trained in a supervised manner by minimizing the mean-squared error (MSE) between $g _ { t }$ and the truncated-bootstrapped target $\hat { y } _ { H }$ in (32). After training, the LSTM network serves as a return predictor that estimates the expected cumulative return from any observed state-action sequence. Given the sequence observed up to time slot t, the LSTM network outputs the corresponding prediction $g _ { t }$ . The redistributed reward is then computed as

$$
r _ { t + 1 } ^ { \prime } = g _ { t } - g _ { t - 1 } ,\tag{34}
$$

which measures the increment in the predicted return brought by the current state-action pair. Therefore, rewards are reassigned from delayed outcomes to the state-action pairs that contribute to the eventual return, serving as a practical approximation of the theoretical reward redistribution in (31).

3) Scheduling Policy Optimization: The redistributed reward provides an immediate signal for learning how each observed system state should be mapped to request assignments. Accordingly, we parameterize the scheduling policy $\pi _ { \pmb { \theta } } ( \pmb { a } \mid \pmb { s } )$ using a neural network with parameters θ, which outputs a categorical distribution over the edge servers for each incoming request. Also, we define a value network parameterized by ϕ to estimate the state value function $V _ { \phi } ( s )$ . Following proximal policy optimization (PPO) [33], we optimize these two networks as the actor and critic, respectively, eliminating the need for an explicit system transition model. To ensure stable policy updates, a surrogate objective function is utilized to constrain the optimization step by clipping the probability ratio between the new and old policies. Let $\rho _ { t } ( \theta ) = \pi _ { \theta } ( { \bf { a } } _ { t } | { \bf { s } } _ { t } ) / \pi _ { \theta _ { \mathrm { { o l d } } } } ( { \bf { a } } _ { t } | { \bf { s } } _ { t } )$ denote the probability ratio between the new and old policy. The surrogate objective function is defined as

$$
L ^ { \mathrm { C L I P } } ( \boldsymbol { \theta } ) = \mathbb { E } _ { t } \big [ \operatorname* { m i n } \big ( \rho _ { t } ( \boldsymbol { \theta } ) \hat { A } _ { t } , \mathrm { c l i p } ( \rho _ { t } ( \boldsymbol { \theta } ) , 1 - \varepsilon , 1 + \varepsilon ) \hat { A } _ { t } \big ) \big ] ,\tag{35}
$$

where $\varepsilon$ is a clipping hyperparameter. $\hat { A } _ { t }$ represents the advantage function, which quantifies the relative benefit of taking action $\mathbf { } \mathbf { a } _ { t }$ over the policy average. We apply generalized advantage estimation (GAE), given by $\begin{array} { r } { \hat { A } _ { t } = \sum _ { l = 0 } ^ { \infty } ( \gamma \lambda ) ^ { l } \delta _ { t + l } , } \end{array}$ where $\delta _ { t } = r _ { t + 1 } ^ { \prime } + \gamma V _ { \phi } ( \pmb { s } _ { t + 1 } ) - V _ { \phi } ( \pmb { s } _ { t } )$ denotes the temporaldifference (TD) error, and $\lambda \in [ 0 , 1 ]$ is a parameter of GAE. Accordingly, the actor network is updated by maximizing the surrogate objective function $L ^ { C L I P } ( \theta )$ using gradient ascent.

Subsequently, the value network is trained by minimizing the MSE between the estimated state value and target value,

$$
L ^ { \mathrm { V F } } ( \phi ) = \mathbb { E } _ { t } [ ( V _ { \phi } ( s _ { t } ) - \hat { V } _ { t } ) ^ { 2 } ] ,\tag{36}
$$

where $\hat { V } _ { t } = \hat { A } _ { t } + V _ { \phi _ { \mathrm { o l d } } } ( \pmb { s } _ { t } )$ is the target value. The critic parameters $\phi$ are then updated via gradient descent on $L ^ { V F } ( \phi )$

## C. Algorithm Summary and Theoretical Analysis

The primary steps of LYREO are summarized in Algorithm 1. We now establish the theoretical performance guarantee of the proposed LYREO approach.

Theorem 1. Let $\pi ^ { * }$ denote an optimal policy of the delayedreward SDP induced by problem P2. Let πˆ denote the optimal policy learned by LYREO under the discount factor $\gamma$ and truncation length H. Assume that the reward function is bounded by $| r _ { t } | \le r _ { \operatorname* { m a x } }$ , the value estimation error of the critic network is bounded by $| V ^ { \pi } ( s ) - V _ { \phi } ( s ) | \le \epsilon _ { \mathrm { v } }$ for any policy π, the conditional-mean regression error of the LSTMbased return predictor is bounded by $\left| g _ { t } - \mathbb { E } [ \hat { y } _ { H } \mid \chi _ { 0 : t } ] \right| \leq \epsilon _ { \mathrm { r } } ,$ and the policy optimization error of PPO is bounded by ϵ<sub>PPO</sub>. The performance gap of the learned policy is bounded by

$$
\begin{array} { r l } & { J ( \pi ^ { * } ) - J ( \hat { \pi } ) \leq \displaystyle \frac { 4 } { 1 - \gamma } \left( \gamma ^ { H } \epsilon _ { \mathrm { v } } + \frac { 2 \gamma ^ { H } } { 1 - \gamma } r _ { \mathrm { m a x } } + \epsilon _ { \mathrm { r } } \right) + \epsilon _ { \mathrm { P P O } } } \\ & { \quad \quad \quad = \mathcal { O } \left( \frac { \gamma ^ { H } \epsilon _ { \mathrm { v } } + \epsilon _ { \mathrm { r } } } { 1 - \gamma } + \frac { \gamma ^ { H } r _ { \mathrm { m a x } } } { ( 1 - \gamma ) ^ { 2 } } + \epsilon _ { \mathrm { P P O } } \right) . } \end{array}\tag{37}
$$

Proof. Let $r ^ { \prime * }$ denote the ideal untruncated redistributed reward derived from the true expected return under the optimal policy $\pi ^ { * }$ . Based on the optimality of the second-order Markov reward redistribution established in [30], we consider the SDP guided by $r ^ { \prime * }$ as the ideal objective that provides immediate step-wise reward signals. The performance gap introduced by LYREO arises from approximating these ideal signals via sequence truncation and value bootstrapping in policy learning. We establish the bound in the following four steps.

Algorithm 1: LYREO Approach   
Input: Truncation length $\overline { { H ; } }$ actor learning rate $\xi _ { A } ;$   
critic learning rate $\xi _ { C } ;$ clipping parameter $\varepsilon ;$   
discount factor $\gamma ;$ GAE parameter $\lambda ;$ PPO   
epochs $\Xi$   
Output: Learned scheduling policy $\pi _ { \theta }$   
1 Initialize actor network π, critic network $V ,$ and return   
predictor $g$ with random parameters θ, ϕ, ψ;   
2 while not converged do   
3 Collect a batch of trajectories   
$\mathfrak { B } = \{ ( s _ { t } , a _ { t } , r _ { t + 1 } , \overset { . } { s } _ { t + 1 } ) \} _ { t = 0 } ^ { H - 1 }$ using policy π<sub>θ</sub>;   
$/ \star$ Reward Redistribution Learning \*/   
4 foreach trajectory in B do   
5 Compute learning target $\hat { y } _ { H }$ via (32);   
6 Construct training pairs $\left\{ \left( \chi _ { 0 : t } , \hat { y } _ { H } \right) \right\} _ { t = 0 } ^ { H - 1 } ;$   
7 Update return predictor $g _ { \psi }$ by minimizing MSE   
between $g _ { \psi } \big ( \chi _ { 0 : t } \big )$ and $\hat { y } _ { H }$ over all training pairs;   
$/ \star$ Scheduling Policy Optimization /   
8 Freeze return predictor $g _ { \psi } ;$   
9 foreach trajectory in $\mathfrak { B }$ do   
10 Compute redistributed reward $r _ { t + 1 } ^ { \prime }$ via (34);   
11 Compute TD error $\delta _ { t }$ and advantage ${ \hat { A } } _ { t } ;$   
12 Set $\theta _ { \mathrm { o l d } }  \theta ;$   
13 for epoch = 1 to $\Xi$ do   
14 Update actor parameters θ by maximizing   
$\dot { L } ^ { \mathrm { C L I P } } ( \theta )$ via (35) with learning rate $\xi _ { A } ;$   
15 Update critic parameters $\phi$ by minimizing   
$\dot { L } ^ { \mathrm { V F } } ( \phi )$ via (36) with learning rate $\xi _ { C } ;$

Step I: Bounding the Return Prediction Error. Let $\begin{array} { r } { y _ { \infty } = \sum _ { h = 0 } ^ { \infty } \gamma ^ { h } r _ { h + 1 } } \end{array}$ denote the true infinite-horizon cumulative return, and yˆ denote the approximated return constructed via truncation and bootstrapping, given by (32). The estimation error can be derived as

$$
| y _ { \infty } - { \hat { y } } _ { H } | = \left| \sum _ { h = H } ^ { \infty } \gamma ^ { h } r _ { h + 1 } - \gamma ^ { H } V _ { \phi } ( \pmb { s } _ { H } ) \right| .\tag{38}
$$

By adding and subtracting the same term $\gamma ^ { H } V ^ { \pi } ( \pmb { s } _ { H } )$ and applying the triangle inequality, we obtain

$$
\begin{array} { r l } & { \displaystyle | y _ { \infty } - \hat { y } _ { H } | \leq \gamma ^ { H } | V ^ { \pi } ( \pmb { s } _ { H } ) - V _ { \phi } ( \pmb { s } _ { H } ) | } \\ & { \quad \quad + \left| \displaystyle \sum _ { h = H } ^ { \infty } \gamma ^ { h } r _ { h + 1 } - \gamma ^ { H } V ^ { \pi } ( \pmb { s } _ { H } ) \right| . } \end{array}\tag{39}
$$

By assumption, the first term is bounded by $\gamma ^ { H } \epsilon _ { \mathrm { v } }$ . For the second term, by the definition of the value function, multiplying both sides by $\gamma ^ { H }$ and re-indexing the summation, we have $\begin{array} { r } { \gamma ^ { H } V ^ { \pi } ( \pmb { \mathscr { s } } _ { H } ) = \mathbb { E } \left[ \sum _ { h = H } ^ { \infty } \gamma ^ { h } r _ { h + 1 } \overline { { \ | \ \pmb { \mathscr { s } } _ { H } } } \right] } \end{array}$ . Since the instantaneous reward is bounded by $| r _ { t } | \leq r _ { \operatorname* { m a x } }$ , the absolute deviation between any realization of the truncated return and its expectation is bounded by $2 \gamma ^ { H } r _ { \operatorname* { m a x } } / ( 1 - \gamma )$ . Combining the two bounds, the return prediction error satisfies

$$
\vert y _ { \infty } - \hat { y } _ { H } \vert \leq \gamma ^ { H } \epsilon _ { \mathrm { v } } + \frac { 2 \gamma ^ { H } } { 1 - \gamma } r _ { \mathrm { { m a x } } } \triangleq \Pi .\tag{40}
$$

Step II: Bounding the Redistributed Reward Error. The LSTM network predicts the expected return conditioned on the state-action sequence $\chi _ { 0 : t } .$ Let $g _ { t } ^ { * } ~ = ~ \mathbb { E } [ y _ { \infty } ~ | ~ \chi _ { 0 : t } ]$ and $\hat { g } _ { t } = \mathbb { E } [ \hat { y } _ { H } \mid \chi _ { 0 : t } ]$ denote the ideal prediction under the true return and the ideal conditional expectation of the truncatedbootstrapped target, respectively. Since $| y _ { \infty } - \hat { y } _ { H } | ~ \le ~ \Pi$ Jensen’s inequality implies

$$
\begin{array} { r } { | g _ { t } ^ { * } - \hat { g } _ { t } | \leq \vert \mathbb { E } \big [ y _ { \infty } - \hat { y } _ { H } \mid \chi _ { 0 : t } \big ] \vert \leq \mathbb { E } \big [ \vert y _ { \infty } - \hat { y } _ { H } \vert \vert \chi _ { 0 : t } \big ] \leq \Pi . } \end{array}\tag{41}
$$

By assumption, the regression error between $g _ { t }$ , trained via (33), and the conditional expectation $\hat { g } _ { t }$ is bounded by $\epsilon _ { \mathrm { r } } .$ Combining this with (41) via the triangle inequality,

$$
\begin{array} { r } { | g _ { t } ^ { * } - g _ { t } | \leq | g _ { t } ^ { * } - \hat { g } _ { t } | + | \hat { g } _ { t } - g _ { t } | \leq \Pi + \epsilon _ { \mathrm { r } } . } \end{array}\tag{42}
$$

Recalling the redistributed reward defined in (34), the error in the redistributed reward is bounded by

$$
| r _ { t + 1 } ^ { \prime \ast } - r _ { t + 1 } ^ { \prime } | \leq | g _ { t } ^ { \ast } - g _ { t } | + | g _ { t - 1 } ^ { \ast } - g _ { t - 1 } | \leq 2 ( \Pi + \epsilon _ { \mathrm { r } } ) ,\tag{43}
$$

where $r _ { t + 1 } ^ { \prime * }$ and $r _ { t + 1 } ^ { \prime }$ are the ideal redistributed reward and its approximation obtained via the proposed truncation and bootstrapping scheme, respectively.

Step III: Bounding the Value Function Error. For any policy π, the difference between its value functions evaluated under the ideal and approximated redistributed rewards can be accumulated over the infinite horizon as

$$
\begin{array} { r } { | V _ { r ^ { \prime * } } ^ { \pi } ( s ) - V _ { r ^ { \prime } } ^ { \pi } ( s ) | = \left| \mathbb { E } _ { \pi } \left[ \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } ( r _ { t + 1 } ^ { \prime * } - r _ { t + 1 } ^ { \prime } ) \right] \right| } \\ { \leq \displaystyle \sum _ { t = 0 } ^ { \infty } \gamma ^ { t } 2 ( \Pi + \epsilon _ { \mathrm { r } } ) = \frac { 2 ( \Pi + \epsilon _ { \mathrm { r } } ) } { 1 - \gamma } . } \end{array}\tag{44}
$$

Step IV: Deriving the Policy Gap. We now evaluate the suboptimality of policy $\hat { \boldsymbol { \pi } } _ { \boldsymbol { r ^ { \prime } } }$ . Note that we have $V _ { r ^ { \prime } } ^ { \pi ^ { * } } - V _ { r ^ { \prime } } ^ { \hat { \pi } _ { r ^ { \prime } } } \leq$ ϵ , as $\hat { \boldsymbol { \pi } } _ { \boldsymbol { r ^ { \prime } } }$ approximates the maximizer of the value function under $r ^ { \prime }$ with a bounded optimization error ϵ<sub>PPO</sub>. Leveraging the standard value difference decomposition then gives

$$
\begin{array} { r l } & { \quad J _ { r ^ { \prime * } } ( \pi ^ { * } ) - J _ { r ^ { \prime * } } ( \hat { \pi } _ { r ^ { \prime } } ) } \\ & { \leq | V _ { r ^ { \prime * } } ^ { \pi ^ { * } } - V _ { r ^ { \prime } } ^ { \pi ^ { * } } | + \epsilon _ { \mathrm { P P O } } + | V _ { r ^ { \prime } } ^ { \hat { \pi } _ { r ^ { \prime } } } - V _ { r ^ { \prime * } } ^ { \hat { \pi } _ { r ^ { \prime } } } | } \\ & { \leq \displaystyle \frac { 2 ( \Pi + \epsilon _ { \mathrm { r } } ) } { 1 - \gamma } + \epsilon _ { \mathrm { P P O } } + \frac { 2 ( \Pi + \epsilon _ { \mathrm { r } } ) } { 1 - \gamma } = \frac { 4 ( \Pi + \epsilon _ { \mathrm { r } } ) } { 1 - \gamma } + \epsilon _ { \mathrm { P P O } } . } \end{array}\tag{45}
$$

Substituting the definition of Π into the preceding inequality yields (37), which completes the proof. □

## V. PERFORMANCE EVALUATION

## A. Simulation Settings

We consider a system consisting of heterogeneous edge servers. Each edge server deploys the Llama-3.2-1B model and is equipped with one of three GPU platforms: NVIDIA GeForce RTX 3080, RTX 3090, and RTX 4090. Following the hardware specifications of these GPUs, the available GPU memory capacities are set to $M _ { s } \in \{ 1 0 , 2 0 , 2 4 \}$ GB, respectively. The prefill rate $v _ { s } ^ { \mathrm { p r e } } \in \{ 2 5 0 0 , 4 0 0 0 , 8 0 0 0 \}$ tokens/s and the decoding throughput $v _ { s } ^ { \mathrm { d e c } } ( \cdot )$ are obtained via offline profiling of the Llama-3.2 model using a vLLM-based serving engine on the corresponding GPU platforms. Specifically, $v _ { s } ^ { \mathrm { d e c } } ( \cdot )$ is modeled as a non-increasing function of the instantaneous KV cache occupation to reflect the memory-bound nature of the decoding phase. Inference requests are drawn from the LMSYS-Chat-1M dataset [34], with the input and output token lengths averaging 70 and 215 tokens, respectively. The KV cache memory occupation per token is set to $\alpha = 1 6 { } ~ \mathrm { K B }$ while the transmission size of a single token is $\beta = 1 6$ bits. We set $T = 8 0 0$ time slots, slot duration $\Delta = 1 \mathrm { ~ s } ,$ and loadbalancing tolerance $\epsilon = 0 . 0 1$ . User locations evolve randomly within a $2 0 0 \times 2 0 0 ~ \mathrm { m ^ { 2 } }$ service area in each time slot.

TABLE I: Simulation Parameters
<table><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Bandwidth  $B _ { s }$ </td><td>[6, 14] MHz</td><td>Clipping parameter ε 0.2</td><td></td></tr><tr><td>Transmission power pi [15, 25] dBm</td><td></td><td>Learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Noise PSD  $N _ { 0 , s }$ </td><td>-174 dBm/Hz</td><td>PPO training epochs 10</td><td></td></tr><tr><td>Discount factor γ</td><td>0.98</td><td>Random seeds</td><td>5</td></tr></table>

During training, none of the algorithms has access to the complete trajectory. Instead, policy optimization relies only on the currently observed trajectory, consistent with the infinitehorizon formulation. After the T-th time slot, all unfinished inference requests continue execution until completion, and the rewards are accumulated and assigned to the final step. The neural networks are implemented using PyTorch 1.8. Both the actor and critic networks in the PPO architecture consist of four fully connected layers. The return predictor for reward redistribution is implemented as a two-layer LSTM with a hidden dimension of 64. The remaining parameters are summarized in Table I, following widely adopted settings from the literature [17], [19].

## B. Baseline Approaches

To comprehensively evaluate the proposed LYREO approach, we consider three categories of baseline schemes.

First, two DRL-based baselines are implemented: the Lyapunov-assisted PPO algorithm (Ly-PPO) [22], which integrates Lyapunov optimization with PPO but excludes the proposed reward redistribution framework, and the conventional PPO algorithm (PPO) [23], which optimizes latency with the load-balancing constraint violation incorporated as a penalty term, without Lyapunov-based constraint transformation or reward redistribution framework. Second, to validate the effectiveness of the DEED architecture, we implement a static batching scheme (Static) [8], where inference requests are grouped into fixed batches for decoding: newly arrived requests are not admitted into an ongoing decoding batch, and completed requests remain until the entire batch finishes decoding. Finally, two heuristic scheduling strategies are included as classical non-learning baselines: Least-Loaded (LL), which schedules each request to the edge server with the lowest current workload, and Random, which schedules each request to a randomly selected edge server.

![](images/7a076ca36d7f3ca2beefd246d3b1b3f8270f41092e2fd5a68be30fe88b93f1eb.jpg)  
(a)

![](images/9017950d709814d0e613832775f24a3a12cbc3cef92ae17146cc46668c51ef53.jpg)  
(b)  
Fig. 2: Convergence performance of LYREO, Ly-PPO, and PPO approaches.

## C. Performance Comparison and Analysis

1) Convergence Performance: Fig. 2 presents the convergence behavior of the three learning-based approaches regarding average end-to-end latency and load-balancing queue length. Solid curves and shaded regions represent the mean and min-max range across seeds, respectively. As shown in Fig. 2(a), LYREO achieves the lowest end-to-end latency throughout the training process and converges to a stable value of approximately 2.4 s. In comparison, Ly-PPO stabilizes at around 3.9 s, while conventional PPO converges to a substantially higher latency of approximately 5.2 s. Fig. 2(b) shows a consistent performance advantage in load-balancing control. After convergence, LYREO reduces the average load-balancing queue length to approximately 0.02, compared with about 0.09 for $\mathrm { L y - P P O }$ and 0.22 for PPO (computed post hoc). Moreover, the two Lyapunov-guided approaches maintain relatively stable queue levels, whereas PPO remains at a higher and more variable plateau. The improvement of LYREO over Ly-PPO highlights the benefit of reward redistribution. By attributing delayed inference outcomes to the scheduling decisions responsible for them, LYREO provides more informative temporal feedback for policy learning, leading to the best overall convergence performance. This benefit extends beyond latency: since load balancing is incorporated into the drift-plus-penalty function, the same mechanism also helps the policy associate accumulated imbalance with the scheduling decisions, improving the load-balancing component. Meanwhile, the gap between PPO and the two Lyapunov-guided approaches demonstrates the advantage of incorporating the Lyapunov framework into policy learning. Although PPO also penalizes excessive workload deviation, its balancing pressure does not adapt to the accumulated deviation over time. In contrast, the Lyapunov virtual queue provides an adaptive corrective signal, resulting in lower and more stable load-balancing queue lengths.

2) Distribution Comparison: Fig. 3 examines the distributions of the two performance metrics over the evaluation runs. In Fig. 3(a), LYREO achieves the lowest median end-to-end latency and the most compact distribution, followed by Ly-PPO, whereas PPO, Static, LL, and Random exhibit higher medians and longer upper tails. Fig. 3(b) shows that LYREO and Ly-PPO maintain lower load-balancing deviations than the other baselines. Overall, the consistent advantages of LYREO and Ly-PPO demonstrate the efficacy of handling load-balancing constraints through adaptive Lyapunov guidance. On the other hand, PPO outperforms most non-learning baselines across the two metrics, indicating that the penalty term enables it to learn a meaningful trade-off between latency and load balancing. However, treating load balancing as a fixed penalty limits its ability to adapt to time-varying server imbalance. Also, the delayed feedback associated with the penalty causes PPO to react only after requests have accumulated on particular edge servers, leading to worse performance than LYREO and Ly-PPO. Among the non-learning baselines, Static achieves lower latency than LL, whereas LL provides better load balancing. Random performs worst on both metrics, exhibiting the highest medians and the widest distributions.

![](images/977a48068f884a7970715f6ce245a94e4fb21a3e1db1bcf1959f87477fc9ab2c.jpg)

![](images/16ae3dded2c1affd7792f76676fae409fdc4d309846a41a04b8cb3c145c72caf.jpg)  
Fig. 3: Boxplot comparison of different approaches.

## D. Load-Balancing Ablation

This subsection evaluates the necessity of long-term load balancing and validates the KV cache memory-time consumption workload through two variants of LYREO for the ablation study. Specifically, LYREO-Count replaces the proposed KV cache memory-time consumption with the number of active requests, while LYREO-NoLB removes the long-term loadbalancing constraint. Beyond the two primary optimization objectives, we introduce two additional evaluation metrics: the P99 end-to-end latency, defined as the 99th percentile of request-level latency, and the high-KV ratio, defined as the percentage of time slots across all edge servers where the peak KV cache utilization exceeds 90% of the reserved memory capacity limit. These two metrics, which are not directly optimized by the considered approaches, respectively characterize the worst-case, user-perceived latency that governs QoS satisfaction and the frequency of near-limit KV cache operation in edge servers, providing complementary evidence for the system-level performance of scheduling approaches.

As shown in Table II, LYREO-NoLB achieves the lowest average end-to-end latency, since it is free to concentrate inference requests on the servers offering the lowest latency. However, this latency advantage is accompanied by the cost of less balanced resource utilization. Without the load-balancing mechanism, request concentration drives the average loadbalancing deviation up by 125.0% relative to LYREO, while the high-KV ratio rises to 14.35%, a 123.5% relative increase. More importantly, this imbalance translates into substantially worse tail behavior, with the P99 end-to-end latency reaching 6.73 s, 79.9% higher than that of LYREO. These results show that latency-oriented scheduling alone cannot prevent persistent workload concentration. Explicit long-term load balancing instead preserves KV cache headroom across edge servers and mitigates the tail-latency degradation, at the expected cost of higher average latency. Meanwhile, LYREO-Count consistently underperforms LYREO across all four metrics, increasing the average latency to 2.79 s, the P99 latency to 4.56 s, the average load-balancing deviation to 0.011, and the high-KV slot ratio to 8.04%. This indicates that conventional count-based balancing does not balance the resource pressure imposed on heterogeneous edge servers, since inference requests differ in their KV cache occupation and duration. By jointly capturing occupation and residence time, the KV cache memory-time consumption provides a more reliable workload signal, validating its design for long-term load-balancing.

TABLE II: Load-Balancing and Workload Metrics Ablation
<table><tr><td rowspan="2">Approach</td><td colspan="4">Avg. End-to-End Avg. Load-Balancing P99 End-to-End</td></tr><tr><td>Latency (s)</td><td>Deviation  $( \times 1 0 ^ { - 2 } )$ </td><td>Latency (s)</td><td>High-KV Ratio (%)</td></tr><tr><td rowspan="2">LYREO</td><td>2.63</td><td>0.8</td><td>3.74</td><td>6.42</td></tr><tr><td>(Ref.)</td><td>(Ref.)</td><td>(Ref.)</td><td>(Ref.)</td></tr><tr><td rowspan="2">LYREO-Count</td><td>2.79</td><td>1.1 (+37.5%)</td><td>4.56 (+21.9%)</td><td>8.04</td></tr><tr><td>(+6.1%) 2.02</td><td>1.8</td><td>6.73</td><td>(+25.2%) 14.35</td></tr><tr><td>LYREO-NoLB</td><td>(-23.2%)</td><td>(+125.0%)</td><td>(+79.9%)</td><td>(+123.5%)</td></tr></table>

1) Impact of System Parameters: Fig. 4 presents the impact of the number of users on algorithm performance, with the number of edge servers fixed at S = 8. Since each user generates one request per time slot in our system, U directly determines the inference request intensity within each time slot. Fig. 4(a) illustrates that the average end-to-end latency of all methods increases as U grows from 20 to 60, since more users generate more requests within each time slot, which share the limited wireless bandwidth and result in larger prefill and decoding batches. The latency of LYREO increases from about $2 . 3 \mathrm { ~ s ~ t o ~ } 3 . 3 \mathrm { ~ s ~ }$ while the second-best $\mathrm { L y - P P O }$ rises from about $2 . 9 \mathrm { ~ s ~ t o ~ } 5 . 6 \mathrm { ~ s } .$ On average, LYREO therefore reduces latency by roughly 33% relative to Ly-PPO, with an even larger gap relative to PPO, Static, LL, and Random. Fig. 4(b) shows that heavier traffic also increases the server-averaged load-balancing deviation of all schemes. Nevertheless, LYREO keeps the deviation below approximately 0.009 for the entire tested range, satisfying the predefined tolerance on average. Overall, LYREO consistently achieves the best performance across both metrics, and its advantage widens significantly under high request intensity. This is because heavier request intensity prolongs request completion, leading to more delayed rewards, a setting in which reward redistribution is particularly effective and consequently amplifies the advantage of LYREO over Ly-PPO. Meanwhile, higher intensity pushes the system closer to its capacity limit, where minor allocation imbalances can rapidly accumulate into substantial workload disparities. The Lyapunov virtual queue resolves this by proactively regulating workload imbalance, allowing LYREO to yield its advantage over LL, PPO, and other baselines.

Fig. 5 compares system performance with respect to the number of edge servers, with the number of users fixed at $U ~ = ~ 4 0$ . Increasing S provides more transmission bandwidth, computation capacity, and KV cache capacity, so all algorithms achieve lower latency. From Fig. 5(a), LYREO consistently achieves the best latency performance across every server scale, followed by $\mathrm { L y - P P O }$ and PPO. Among the non-learning methods, LL improves markedly as the number of servers increases, since least-loaded routing can select from a large pool of lightly occupied servers. Static shows only modest improvement, as its fixed batching strategy limits the benefit of additional servers, while Random remains the worst throughout, owing to its state-agnostic assignment. The loadbalancing deviations in Fig. 5(b) show no clear monotonic trend as the number of edge servers increases. Specifically, the learning-based approaches generally decrease at first and then flatten or slightly rebound. With a fixed number of requests, more edge servers mitigate workload concentration but also result in sparser per-server demand and a larger action space, making effective scheduling increasingly difficult. Nevertheless, LYREO maintains the lowest deviation across all settings, showing its stronger ability to exploit additional server choices while preserving system-wide load balance.

![](images/a441c90542ab6445df86ef2592ab16d6b3522f3588a9be3409a1318ba7099be1.jpg)  
(a)

![](images/f30b86a34906a9ae677b82ef1218d952b746b0641befb25f379f53e71f2087f1.jpg)  
(b)

Fig. 4: Performance comparison under varying numbers of users.  
![](images/3b32c065976e723ba7f6b241212ecfea950c54691910a52b4eeee6416efe4c78.jpg)  
(a)

![](images/83931d928251ecff1a893a4e54c50014501223cf5d190bf92f8dd2b6f361b1eb.jpg)  
(b)  
Fig. 5: Performance comparison under varying numbers of edge servers.

Fig. 6 evaluates the impact of hardware heterogeneity under three distinct configurations: Homogeneous (Hom.) uses 8× RTX 3090 GPUs; Moderate (Mod.) uses 3× RTX 3080, 3× RTX 3090, and 2× RTX 4090 GPUs; and Heterogeneous (Het.) uses 4× RTX 3080, 1× RTX 3090, and $3 \times$ RTX 4090 GPUs. Average load-balancing deviation increases from Hom. to Mod. and Het. for every method, as growing heterogeneity in computational capacity across edge servers makes balanced workloads harder to maintain. Under the Hom. configuration, the three learning-based methods achieve the lowest deviations. As heterogeneity increases, LYREO and

Ly-PPO remain the two best-performing methods. LYREO consistently achieves the lowest deviation and remains slightly better than $\mathrm { L y - P P O } ,$ a small yet consistent advantage showing that reward redistribution improves not only latency but also load-balancing performance. LL performs best among the non-learning methods, a predictable outcome given that its routing is explicitly designed to correct current imbalances, thus providing a natural advantage in the deviation dimension. Static and Random perform worst under almost all configurations, as fixed batching and state-agnostic routing fail to adapt effectively to server heterogeneity.

## E. Hyperparameter Sensitivity

Fig. 7 illustrates the latency-balancing trade-off of LYREO controlled by the Lyapunov parameter ν. Increasing ν from 0.05 to 2 places greater weight on the latency term in the drift-plus-penalty objective. Accordingly, end-to-end latency falls from about $5 . 6 \mathrm { ~ s ~ t o ~ } 2 . 1 \mathrm { ~ s } ,$ with most of the gain obtained by $\nu = 0 . 5 ,$ , while the corresponding load-balancing deviation rises slowly at first and then increases from approximately 0.008 at $\nu \ = \ 0 . 5$ to 0.016 and 0.019 at $\nu ~ = ~ 1$ and 2, respectively. As expected, latency minimization comes at the cost of virtual-queue stabilization. Notably, $\nu = 0 . 5$ marks a turning point in the trade-off: latency improvements become markedly less responsive to further increases in ν, whereas the load-balancing deviation grows increasingly sensitive.

Finally, Fig. 8 shows the effect of the return predictor’s truncation length on LYREO’s average end-to-end latency. While all configurations converge, a longer H consistently yields a lower final latency. A larger H exposes the return predictor to more of the actual delayed reward, reducing its reliance on the bootstrapped estimation and yielding more accurate value predictions. This behavior matches our theoretical analysis of LYREO, where the policy performance gap shrinks as H increases. However, the improvement becomes marginal beyond $H ~ = ~ 2 5 0$ , suggesting that most decision-relevant dependencies in the system can already be captured within a bounded horizon. Meanwhile, a longer truncation length extends the LSTM-based predictor’s recurrent trajectories, increasing training complexity. Consistent with this, $H = 3 0 0$ converges to a latency level close to that of $H \ : = \ : 2 5 0$ but exhibits larger fluctuations, indicating diminishing benefits and increased training instability.

## VI. CONCLUSION

This paper studied LLM inference request scheduling for agentic AI services, aiming to minimize long-term average end-to-end latency while maintaining load balance across heterogeneous edge servers. To capture fine-grained LLM serving processes, we developed a system model that jointly characterizes wireless transmission, multi-stage inference, and KV cache evolution throughout each request’s lifecycle. Based on this model, we quantified edge server workload using the normalized KV cache memory-time consumption and formulated a long-term load-balancing constraint. We proposed the LYREO approach, which converts accumulated load imbalance into per-slot scheduling guidance and redistributes delayed outcomes to their responsible decisions, with sequence truncation and value bootstrapping enabling return estimation in the infinite-horizon setting. Extensive evaluations demonstrated that LYREO consistently reduces both end-to-end latency and load-balancing deviation, outperforming existing baselines.

![](images/57909ad5aec8cd8f1263f89e4ee88655077fe4d3a79a4d00dcac1eccbc689c02.jpg)  
Fig. 6: Performance comparison of load-balancing deviation under varying heterogeneity levels.

![](images/a20e5005cc519b1127b021c9b9dcd7868f20b8f9a68310256fc3b022997ea6ac.jpg)  
Fig. 7: Impact of Lyapunov parameter on system performance.

![](images/8ec2823a798ae374c48d30dbaeb530a965e857ba340d012273e7549f229f6a06.jpg)  
Fig. 8: Impact of truncation length on training performance.

## REFERENCES

[1] F. Jiang et al., “From large ai models to agentic ai: A tutorial on future intelligent communications,” IEEE J. Sel. Areas Commun., vol. 44, pp. 3507–3540, 2026.

[2] C. Zhao et al., “Edge general intelligence through world models, large language models, and agentic ai: Fundamentals, solutions, and challenges,” IEEE Trans. Cogn. Commun. Netw., vol. 12, pp. 5649– 5675, 2026.

[3] R. Zhang et al., “Toward edge general intelligence with agentic ai and agentification: Concepts, technologies, and future directions,” IEEE Commun. Surv. Tutor., vol. 28, pp. 4285–4318, 2026.

[4] T. Zheng et al., “Joint optimization of dynamic batching and adaptive partitioning for distributed llms inference in mobile edge computing,” IEEE Trans. Mobile Comput., vol. 25, no. 6, pp. 8747–8763, 2026.

[5] Y. He, J. Fang, F. R. Yu, and V. C. Leung, “Large language models (llms) inference offloading and resource allocation in cloud-edge computing: An active inference approach,” IEEE Trans. Mobile Comput., vol. 23, no. 12, pp. 11 253–11 264, 2024.

[6] H. Huang et al., “Dynamic model deployment, batch scheduling, and resource allocation in mllm-enabled edge–cloud networks: A multiagent two-timescale drl approach,” IEEE Internet Things J., vol. 12, no. 23, pp. 50 818–50 835, 2025.

[7] Y. Li, “Llm bandit: Cost-efficient llm generation via preferenceconditioned dynamic routing,” 2025. [Online]. Available: https: //arxiv.org/abs/2502.02743

[8] T. Li and Y. Gong, “Two-sided matching for batch-aware llm request scheduling in edge networks,” in Proc. IEEE 50th Conf. Local Comput. Netw. (LCN), 2025, pp. 1–7.

[9] A. Mekrache, A. Ksentini, and C. Verikoukis, “Drl-enabled slo-aware task scheduling for large language models in 6g networks,” in Proc. IEEE Int. Conf. Commun. (ICC), 2025, pp. 813–818.

[10] Z. Chen et al., “A universal load balancing principle and its application to large language model serving,” arXiv preprint arXiv:2601.17855, 2026.

[11] C. Yi et al., “Workload re-allocation for edge computing with server collaboration: A cooperative queueing game approach,” IEEE Trans. Mobile Comput., vol. 22, no. 5, pp. 3095–3111, 2023.

[12] G.-I. Yu, J. S. Jeong, G.-W. Kim, S. Kim, and B.-G. Chun, “Orca: A distributed serving system for transformer-based generative models,” in Proc. 16th USENIX Symp. Oper. Syst. Design Implementation (OSDI), 2022, pp. 521–538.

[13] W. Kwon et al., “Efficient memory management for large language model serving with pagedattention,” in Proc. 29th Symp. Operating Syst. Princ. (SOSP), 2023, p. 611–626.

[14] H. Jin, S. Li, and M. A. Gregory, “Gensched: Phase-aware generative scheduling for llm inference in heterogeneous edge networks,” in Proc. IEEE Conf. Comput. Commun. (IEEE INFOCOM), 2026, pp. 1–6.

[15] Y. Li et al., “Cloud-edge system for scheduling unpredictable llm requests with combinatorial bandit,” IEEE Trans. Serv. Comput., vol. 18, no. 6, pp. 3567–3580, 2025.

[16] B. Li, Y. Jiang, V. Gadepally, and D. Tiwari, “Llm inference serving: Survey of recent advances and opportunities,” in Proc. IEEE High Perform. Extreme Comput. Conf. (HPEC), 2024, pp. 1–8.

[17] X. Zhang et al., “Beyond the cloud: Edge inference for generative large language models in wireless networks,” IEEE Trans. Wireless Commun., vol. 24, no. 1, pp. 643–658, 2025.

[18] M. Zhang, X. Shen, J. Cao, Z. Cui, and S. Jiang, “Edgeshard: Efficient llm inference via collaborative edge computing,” IEEE Internet Things J., vol. 12, no. 10, pp. 13 119–13 131, 2025.

[19] F. Mou, Z. Tang, W. Jia, and W. Zhao, “Adaptive request scheduling and load balancing for edge deployed large language models,” IEEE Trans. Serv. Comput., vol. 19, no. 2, pp. 934–947, 2026.

[20] J. Li, B. Han, S. Li, X. Wang, and J. Li, “Collm: A collaborative llm inference framework for resource-constrained devices,” in Proc. IEEE/CIC Int. Conf. Commun. China (ICCC), 2024, pp. 185–190.

[21] B. Zhu, Z. Chen, L. Zhao, H. Shin, and A. Nallanathan, “Enabling efficient large language model inference over wireless networks with caching,” IEEE Trans. Wireless Commun., vol. 25, pp. 18 326–18 343, 2026.

[22] A. Younesi et al., “Splitwise: Collaborative edge–cloud inference for llms via lyapunov-assisted drl,” in Proc. IEEE/ACM Int. Conf. Utility Cloud Comput., 2025, pp. 1–11.

[23] N. Qiu et al., “Joint request batching and worker assignment in airan edge inference systems,” in Proc. Int. Conf. Future Commun. Netw. (FCN), 2025, pp. 1–6.

[24] X. Ma, H. Zhou, T. Wu, and X. Fan, “Preference-aware task routing for edge-cloud hierarchical large language model inference,” in IEEE Conf. Comput. Commun. (IEEE INFOCOM), 2026, pp. 2985–2986.

[25] Z. Tang, Y. Sun, W. Chen, J. Ding, and B. Ai, “Gelato: Generative entropy-and lyapunov-based adaptive token offloading for device-edge speculative llm inference,” arXiv preprint arXiv:2605.10124, 2026.

[26] Z. Li, C. Yang, X. Huang, W. Zeng, and S. Xie, “Coor: Collaborative task offloading and service caching replacement for vehicular edge computing networks,” IEEE Trans. Veh. Technol., vol. 72, no. 7, pp. 9676–9681, 2023.

[27] K. Cheng et al., “Slice-level scheduling for high throughput and load balanced llm serving,” arXiv preprint arXiv:2406.13511, 2024.

[28] B. Sun et al., “Llumnix: Dynamic scheduling for large language model serving,” in Proc. 18th USENIX Symp. Oper. Syst. Design Implementation (OSDI), 2024, pp. 173–191.

[29] M. Neely, Stochastic network optimization with application to communication and queueing systems. Morgan & Claypool Publishers, 2010.

[30] J. A. Arjona-Medina et al., “Rudder: Return decomposition for delayed rewards,” Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), vol. 32, 2019.

[31] M. Chen et al., “Ran information-assisted tcp congestion control using deep reinforcement learning with reward redistribution,” IEEE Trans. Commun., vol. 70, no. 1, pp. 215–230, 2021.

[32] J. Gui, Z. Li, J. Zhang, X. Deng, and G. Min, “Multi-heterogeneousagent drl for efficient congestion control with reward redistribution in space–air–ground integrated networks,” IEEE Trans. Netw. Sci. Eng., vol. 13, pp. 3035–3052, 2026.

[33] J. Schulman, F. Wolski, P. Dhariwal, A. Radford, and O. Klimov, “Proximal policy optimization algorithms,” arXiv preprint arXiv:1707.06347, 2017.

[34] L. Zheng et al., “Lmsys-chat-1m: A large-scale real-world llm conversation dataset,” in Proc. Int. Conf. Learn. Representations (ICLR), vol. 2024, 2024, pp. 22 225–22 257.