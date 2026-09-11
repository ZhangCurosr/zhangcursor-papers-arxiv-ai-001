# Characterizing Job Power Elasticity for Power-Flexible AI Training

Philip Colangelo, Charles Dawson, Shayan Sengupta, Ayse Coskun, Varun Sivaram Emerald AI

## Abstract

Large language model (LLM) training is among the fastest-growing sources of electricity demand in modern data centers, and power availability is a primary bottleneck to continued AI infrastructure growth. Making the power consumption of these workloads flexible could unlock additional power for AI growth, limit increases in electricity prices, and improve the utilization of existing grid infrastructure. However, to realize this flexibility, we must first understand how the performance of training workloads changes when GPU power is reduced.

This paper presents the first systematic characterization of job power elasticity (the sensitivity of throughput to power reductions) in LLM training. To quantify elasticity, we introduce the Power Flexibility Index (PFI), a normalized metric that quantifies the performance cost of power reductions and provides a control primitive for SLA-aware power flexibility.

We collect data from 131 LLM training runs on H200 (plus 24 H200 validation runs and 34 matched H100 runs), including both dense and mixture-of-experts models, pretraining and fine-tuning tasks, and up to 32 GPUs. We find that LLM training jobs exhibit substantial but variable power elasticity, and we identify telemetry signals that predict PFI at runtime. Finally, we demonstrate that PFIaware power allocation maximizes total tokens/second throughput under power constraints. Under a 30% power reduction, PFI-aware power allocation recovers 1.5k tokens/s per job, 63% of the performance gap between an equal-weight allocation and an oracle with perfect information. Our results establish power elasticity as a measurable property of training jobs and provide a foundation for power-aware, grid-responsive AI infrastructure.

## 1 Introduction

Large language model (LLM) training is a rapidly growing source of data center electricity demand. Power availability has become a bottleneck for the deployment of new AI compute capacity, with limited grid capacity causing multi-year interconnection delays for new data centers [12]. Recent analysis of the U.S. power system found that data centers operating as flexible loads, reducing power consumption in a few peak hours, could unlock nearly 100 GW of additional capacity [17]. The promise of such flexible capacity creates a strong incentive for AI training clusters to be powerflexible.

To realize these benefits, AI clusters must reduce power consumption when needed while maintaining acceptable training performance, as defined in customer service-level agreements (SLAs). Achieving SLA-aware power flexibility requires understanding the throughput cost of power reductions.

Prior studies of training efficiency primarily focus on energy consumption, Model FLOPs Utilization (MFU [2]), tokens-per-joule at nominal operating points [4], or energy-delay tradeoffs for individual jobs [32, 3]. However, these metrics provide limited insight into the quantity most relevant for infrastructure control: a job’s throughput sensitivity to sustained reductions in available power for up to several hours during periods of peak grid load.

![](images/ff1250293a0b4266c723aae26f87016320530bcf1b52d4a50aaf3afb37647291.jpg)  
Figure 1: LLM training jobs vary in how quickly performance degrades under power reductions. Relatively more power flexible (—) jobs maintain throughput despite power reductions, while relatively less flexible (- -) jobs quickly lose performance at lower power. We introduce the Power Flexibility Index (PFI), a new metric for measuring power elasticity, and conduct large-scale, systematic characterization of modern LLM training jobs. We develop a model for predicting PFI online from readily available GPU telemetry, and we demonstrate how PFI-aware power allocation improves total throughput under cluster-wide GPU power constraints and realistic job mixes.

Understanding job-level power elasticity is important because training jobs with similar baseline throughput and power draw can respond very differently to the same GPU power cap. As a result, heuristics for spreading power constraints across jobs can waste throughput that could otherwise be preserved through elasticity-aware power allocation. To enable these strategies, cluster power managers must be able to identify which jobs can absorb power reductions at the lowest performance cost.

To this end, we present the first large-scale empirical characterization of the power elasticity of modern LLM training jobs under controlled GPU power modulation. We conduct 131 training runs across multiple architectures, pretraining and fine-tuning tasks, and up to 32 NVIDIA H200 GPUs, sweeping GPU power caps across the full operating envelope. From these measurements, we compute the Power Flexibility Index (PFI), a normalized metric that quantifies the performance cost of power reductions and allows the cluster power manager to rank jobs by relative flexibility.

Building on this empirical characterization, we develop telemetry-driven estimators that infer a job’s PFI online from readily-available signals from NVIDIA Data Center GPU Manager (DCGM), allowing cluster power managers to estimate PFI without power-cap sweeps. We apply this estimator to simulated workloads to demonstrate how PFI-aware orchestration can protect relatively inflexible jobs while assigning deeper power reductions to relatively flexible jobs, maximizing tokens/sec under a global power budget. Our contributions are:

• We present the first systematic characterization of power elasticity across LLM training jobs, across a range of model architectures, training tasks, and distributed GPU scales.

• We introduce the Power Flexibility Index (PFI), a normalized job-level metric that quanti fies the throughput cost of power reduction.

• We identify runtime telemetry signals that predict PFI, showing that elasticity can be inferred online from GPU monitoring signals that correlate with different flexibility regimes.

• We demonstrate that PFI-aware power allocation improves aggregate cluster throughput under cluster-wide GPU power constraints, establishing a practical path toward power-aware, grid-responsive data center operation.

## 2 Power Elasticity and the Power Flexibility Index

A job’s power elasticity describes the sensitivity of its throughput to sustained reductions in available power. Highly elastic jobs experience rapid throughput degradation under GPU power capping, whereas inelastic jobs preserve throughput. Figure 1 illustrates representative elastic and inelastic power–throughput regimes.

Power elasticity is important because it enables throughput-aware power capping, allowing cluster power managers to reduce GPU power consumption while minimizing throughput loss. Despite its importance, two gaps remain in the current literature. First, there is no accepted quantitative definition of power elasticity that permits systematic comparison across jobs, model architectures, and hardware settings. Second, there is limited understanding of the job characteristics that determine elasticity or enable its prediction a priori.

To close these gaps, we begin by developing a formal definition of a power flexibility index $( \mathbf { P F I } ) ^ { 1 }$ Let $( T , P ) ,$ be tuples of the average throughput (in tokens/s) and aggregate GPU power (in W) of a job measured under a range of per-GPU power caps $\bar { P } _ { i } , i = 1 , \ldots , \bar { N } . ^ { 2 } P _ { i }$ is the sum of per-device GPU power readings across all GPUs participating in the job. The realized $P _ { i }$ generally falls below $G \cdot { \bar { P _ { i } } }$ (where $G$ is the GPU count), since the cap is an upper bound and the job need not saturate. We define $( T _ { \mathrm { m a x } } , P _ { \mathrm { m a x } } )$ as the throughput/power pair measured without a power cap (i.e. with $\bar { P } _ { N }$ set to thermal design power, TDP).

We define the power flexibility index as the ratio of power decrease to throughput decrease, averaged over measured power caps (excluding the uncapped measurement where $\bar { \Delta { T } } = 0 )$ :

$$
P F I = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { \Delta P _ { i } } { \Delta T _ { i } } = \frac { 1 } { N } \sum _ { \substack { i = 1 ; T _ { i } \neq T _ { \operatorname* { m a x } } } } ^ { N } \frac { 1 - P _ { i } / P _ { \operatorname* { m a x } } } { 1 - T _ { i } / T _ { \operatorname* { m a x } } }\tag{1}
$$

While elasticity describes throughput sensitivity to power loss, PFI inverts this relationship into a control-oriented measure of how efficiently a job can shed power. Consequently, jobs with high elasticity exhibit low PFI, whereas throughput-preserving inelastic jobs exhibit high PFI.

PFI is a hardware-dependent property of a job: the formula normalizes for per-platform reference points $( T _ { \mathrm { m a x } } , P _ { \mathrm { m a x } } )$ and so is comparable across models on the same accelerator, but values from different accelerator generations should not be compared directly because they reflect each platform’s specific power-cap mechanism (§5). The metric has a straightforward interpretation:

Inelastic $( P F I > 1 )$ : throughput declines more slowly than power $( \mathrm { e . g . , - 5 \% }$ power, $- 2 \%$ throughput).

Linear $( P F I = 1 ) .$ : throughput is proportional to power $( \mathrm { e . g . , - 5 \% }$ power, 5% throughput).

$E l a s t i c \left( P F I < 1 \right)$ : throughput declines faster than power $( \mathrm { e . g . , - 5 \% }$ <sup>power,</sup> −<sup>10%</sup> <sup>throughput).</sup>

Comparison to existing metrics. The closest existing metrics fall into two families. ML training efficiency metrics such as MFU [2] and tokens/joule characterize steady-state efficiency at a fixed operating point; they carry no information about how throughput responds to power reduction. Zeus [32] and Perseus [3] consider the energy–time Pareto frontier for individual jobs, but produce perjob optimization outputs rather than a scalar characterization metric comparable across architectures and training tasks. Power systemsflexibility metrics quantify load reduction capacity (MW), ramp rate, and response latency [14] but ignore quality-of-service degradation. PFI unifies both perspectives: it is computable from external power measurements without job internals, normalized for crossarchitecture comparison, and explicitly encodes the performance cost of power reduction.

Driving questions. Given this definition, three questions motivate the rest of this paper:

What factors influence power elasticity? Does PFI vary across architectures (e.g. dense models vs. mixture-of-experts) or training tasks (e.g. pretraining vs. fine-tuning)? How does training scale affect PFI? Are there underlying mechanisms (e.g. memory bottlenecks) that explain these variations?

Can we predict power elasticity in production workloads? The definition of PFI in (1) requires measurements across multiple power caps, which limits its ability to predict job elasticity in production. Predicting PFI from readily available telemetry $( \mathrm { e . g }$ . NVIDIA DCGM [18]) would allow ML engineers and data center operators to track power elasticity in real-time.

Can we use PFI for real-time control and power-aware workload orchestration? Prior works have demonstrated the ability of GPU clusters to respond to requests from grid operators [31, 5]. Data center operators could use real-time PFI estimates to respond efficiently, allocating power reductions to jobs that can best tolerate power reductions and minimizing disruption to their tenants.

Table 1: Experiment matrix. Each sweep group fixes one full configuration per job and varies only the per-GPU power cap; PFI is computed within a sweep group. 8-GPU groups sweep power caps in 100 W increments from 200–700 W, 16- and 32-GPU groups sweep 200/500/600/700 W only. EP = expert parallelism degree (pretraining only); PT = pretraining; LoRA = rank-32 LoRA SFT. GC = gradient checkpointing.
<table><tr><td>Model</td><td>Training Task</td><td>GPUs</td><td>SeqLen</td><td>Batch</td><td>EP</td><td>Compile</td><td>GC</td></tr><tr><td>gpt-oss-20b</td><td>PT, LoRA</td><td>8</td><td>2048,4096</td><td>2,8</td><td>1,4,8</td><td>Y, N</td><td>Y, N</td></tr><tr><td>Qwen3-30B-A3B</td><td>PT, LoRA</td><td>8,16</td><td>4096</td><td>2,16</td><td>1,8</td><td>Y</td><td>Y, N</td></tr><tr><td>Qwen3-32B</td><td>PT, LoRA</td><td>8,16,32</td><td>2048, 4096, 8192</td><td>4,8,16</td><td>—</td><td>Y, N</td><td>Y</td></tr><tr><td>Llama-3.1-70B</td><td>PT, LoRA</td><td>8,16</td><td>2048, 4096</td><td>2,16</td><td></td><td>Y,N</td><td>Y</td></tr></table>

Table 2: Models used in experiments, split into training and held-out validation sets. Active params denote the per-token activation count for sparse Mixture-of-Experts (MoE) models.
<table><tr><td></td><td>Model</td><td>Architecture</td><td>Total</td><td>Active</td><td>Routing</td></tr><tr><td></td><td>gpt-oss-20b</td><td>MoE decoder</td><td>21 B</td><td>≈3.6B</td><td>4 of 32 experts</td></tr><tr><td></td><td>Qwen3-30B-A3B</td><td>MoE decoder</td><td>30 B</td><td>≈3B</td><td>8 of 128 experts</td></tr><tr><td>Traning</td><td>Qwen3-32B</td><td>Dense decoder</td><td>32.8 B</td><td>32.8B</td><td></td></tr><tr><td></td><td>Llama-3.1-70B</td><td>Dense decoder</td><td>70.6B</td><td>70.6B</td><td></td></tr><tr><td></td><td>DeepSeek-V2-Lite</td><td>MoE decoder</td><td>15.7B</td><td>≈2.4B</td><td>6+2 of 64 experts</td></tr><tr><td>Hed lut</td><td>Mistral-Small-24B-Base-2501</td><td>Dense decoder</td><td>24 B</td><td>24 B</td><td></td></tr><tr><td></td><td>Gemma-4-26B-A4B</td><td>MoE decoder</td><td>26B</td><td>≈4B</td><td>8 of 128 experts</td></tr></table>

## 3 Experimental Setup

To characterize power elasticity for representative LLM training jobs, we sweep per-GPU power caps from 200 W to 700 W (TDP), using 100 W increments for 8-GPU jobs and a four-point sweep for 16- and 32-GPU jobs, while pretraining and fine-tuning four open-weight LLMs on clusters of 8/16/32 GPUs, yielding 25 headline sweep groups (131 individual training runs across power caps) plus 4 held-out validation sweeps (24 runs). Complete data tables showing the number of runs for each model are included in Appendix C. All runs execute on Google Kubernetes Engine (GKE) a3-ultragpu-8g nodes (8 NVIDIA H200, 141 GiB HBM3e, 700 W TDP), with multi-node jobs scheduled through Kueue. Clusters are provisioned from a single Terraform module and replicated across six GCP regions; full infrastructure details are given in Appendix B.

We profile four open-weights LLMs spanning dense and Mixture-of-Experts (MoE) architectures (gpt-oss-20b, Qwen3-30B-A3B, Qwen3-32B, and Llama-3.1-70B) under pretraining and rank-32 LoRA fine-tuning (Tables 1, 2). We chose sequence length and batch size to saturate GPU utilization at each cluster size, with gradient checkpointing applied where needed to fit activations in memory and torch.compile enabled where supported. We also profile three validation models and exclude their results from feature selection and predictor fitting: DeepSeek-V2-Lite for MoE pretrain-

![](images/267b2aaad1763ca2533f52de8b91eedeaa3a44b6f94834e8fed31c56f7199f6d.jpg)  
Figure 2: Aggregate GPU power and training throughput for a representative QWEN3-32B pretraining run on 32×H200 at 700 W/GPU.

ing, Mistral-Small-24B-Base-2501 for dense pretraining and dense LoRA SFT, and Gemma-4-26B-A4B for MoE LoRA SFT.

We enforce per-GPU power caps via nvidia-smi -pl before any CUDA context allocation and verify using DCGM power limit counters. We collect GPU utilization, power, and memory at 5 s intervals via a DCGM sidecar; training throughput and MFU are logged in each training step via a custom callback. We partition each trace into warmup, startup, and training phases; all reported metrics aggregate over the training phase only, using the mean of the top 10% of samples to exclude evals and checkpoints. A representative run is shown in Fig. 2.

## 4 Results

Three observations emerge from our study: PFI varies substantially across jobs, the variation depends on architecture and training task, and memory- and compute-related telemetry demonstrate strong correlations with PFI. Complete data tables are included in Appendix C.

PFI varies substantially across modern LLM training jobs. Fig. 3 shows the raw and normalized power–throughput curves for all runs in our dataset, and Fig. 4 summarizes PFI across cluster sizes, model architectures, and training tasks. Across the cohort, we observe substantial variation in PFI (1.10–2.67), indicating that modern LLM training jobs differ in how much throughput they preserve under GPU power capping. Some jobs exhibit nearly linear power–throughput degradation, while others maintain high throughput under aggressive power reduction. We did not observe any jobs with PFI < 1; however, as we discuss in §6, differences in relative PFI are still important for runtime power orchestration.

Architecture and training task shape the observed PFI distribution. As shown in Fig. 4, we observe that dense jobs have the lowest PFI (median 1.27), MoE fine-tuning jobs have the highest PFI (median 2.10), and MoE pretraining jobs fall in between (median 1.43). Applying Dunn’s test for statistical significance of group differences, we find support for MoE fine-tuning as a distinct cluster (corrected $p < 0 . 0 5 )$ but are unable to distinguish MoE pretraining as a third cluster distinct from dense or MoE fine-tuning. Appendix D.4 provides the results of our statistical group difference testing.

Memory and compute intensity-related telemetry show the strongest association with PFI. Fig. 5 shows the relationship between PFI and several metrics gathered through DCGM, including mean SM, DRAM, and tensor pipe activity, NVLink utilization, and selected composite metrics. We observe statistically significant positive correlation between PFI and memory-related signals (mean DRAM activity and DRAM copy product), as well as statistically significant negative correlation with tensor pipe activity. We tested several additional composite metrics, reported in Appendix D.1; however, due to the limited size of our dataset, we limit our analysis in §5 to features with a strong mechanistic link to power flexibility.

![](images/150ce92de3aba471c825cc12f63e77cc92af3a9bd474abac79002a5f8db5e7c8.jpg)

![](images/8068f6ea7591606884b30b0d2762a2fa97370d3639cceffe294fb2d2dd764a35.jpg)

![](images/496cf4f7c9826f88bd6cdd50ac6788f4bcba79be0905d9301c2bcb30f271039a.jpg)  
Figure 3: Power–performance curves for all sweep groups. Left: Aggregate GPU power vs. throughput. Center: Per-sweep normalized power vs. throughput. Right: Mean normalized power vs. throughput for each category (dense vs. MoE, pretraining vs LoRA SFT). Full data are provided in Appendix C.

![](images/5b56a6d0ec699c489c8185bce36fa6a56358e02aff2d9590f60ac866600a6002.jpg)

![](images/392f162b1377a0757bc59b94a67ff0730fc8392bc3617c34eeecc2b44fb5c70e.jpg)

![](images/1ccd4b4b6f3180f7009837d171ab1a73f3e7e97d0a493f34e3afc84ebeb2309b.jpg)  
Figure 4: Left: PFI distribution across all jobs. Right: PFI by architecture and training task; median shown as horizontal bar.

![](images/8a05e01821c63bb322332cb4d27ea9526a11c68108b717fedff7d59eeb7661c4.jpg)

![](images/70ee6be3c56b734743406092b612ec290416c27f44fd7f40c45823371c7c8843.jpg)

![](images/0f18461b6655b0e72a20f8be494dd967b1ae40acb9537a07ddd7827e46d8b41c.jpg)

![](images/9c082a531432c0812f7f90effd50f7d5547ef430da5a25328f391bb282e6fb9b.jpg)

![](images/f90509a68e03249b53aa1ef019b75535c076ee5841526ccce65cc920f9dde895.jpg)

![](images/7299ca6c6f51c99799027d4f6b7966c6ca6e4d071afb25bb723db9a7f8c27ea4.jpg)

![](images/73171b719758bc71a87513ce186167954eccc5709baa96bd927a821923bdac48.jpg)

![](images/134fc95aa93609ca29209e2f98d6c29966efb7a3d9baf2f0e75c1823ff543528.jpg)  
Figure 5: PFI vs. GPU telemetry metrics, showing the correlation (ordinary least-squares fit and Spearman ρ) between PFI and metrics measured for the uncapped job in each sweep group. DRAM copy product and tensor to SM ratio are derived metrics. $^ { * } q < 0 . 0 5 , ^ { * * } \hat { q } < \bar { 0 . } 0 1$ , where q is the Benjamini–Hochberg FDR-corrected p-value across all metrics tested in Fig. 14 in the appendix.

## 5 Discussion

To understand the mechanism behind the different PFI regimes in §4, as well as the correlation between memory activity and PFI, we can examine how the H200 implements power caps. Fig. 6 shows that the H200 caps power by preferentially reducing SM clock frequency while leaving memory clock frequency essentially unchanged: capped GPUs lose compute throughput, but retain memory bandwidth. Memory-bound jobs therefore absorb caps with a smaller throughput loss than computebound jobs. As shown in Fig. 5, MoE models exhibit higher values for memory-related telemetry, suggesting that these signals provide a measurable proxy for regime, separating compute-bound and memory-bound jobs. While extensively studying whether this mechanism generalizes across accelerator architectures is beyond the scope of this work, Appendix D.2 includes a limited validation on H100s, which also preferentially throttle compute speed over memory bandwidth [28], showing a similar pattern.

![](images/7a9eed70cd1685f80f9a47c2d1f6391193b8768eefd836d59b62b3a3b5c965d1.jpg)  
Figure 6: Steady-state SM clock and memory clock frequency as a function of power cap.

![](images/84bc550f18dcbd47c68cb2d05f8b928699a4aba9cba393427fbcc29fbb33ed5a.jpg)  
Figure 7: Predictions of the PFI model fit to DRAM copy product, including the validation set.

Table 3: Predictor ablation for PFI $( n = 2 5 )$ . Job metadata includes architecture type, training task, and log GPU count. p: # of features. LOO: leave-one-out. LOCO: leave-one-cluster-out. $\rho \colon$ Spearman rank correlation.
<table><tr><td></td><td></td><td></td><td colspan="3">L00</td><td colspan="3">LOCO</td></tr><tr><td>Predictors</td><td>p</td><td> $R _ { \mathrm { i n } } ^ { 2 }$ </td><td> $R ^ { 2 }$ </td><td>RMSE</td><td> $\rho$ </td><td> $R ^ { 2 }$ </td><td>RMSE</td><td> $\rho$ </td></tr><tr><td>(i) DRAM copy product</td><td>1</td><td>0.675</td><td>0.524</td><td>0.241</td><td>0.489</td><td>0.246</td><td>0.303</td><td>0.515</td></tr><tr><td>(ii) Arithmetic intensity</td><td>1</td><td>0.381</td><td>0.238</td><td>0.305</td><td>0.482</td><td>-1.652</td><td>0.569</td><td>0.311</td></tr><tr><td>(iii): (i) + (ii)</td><td>2</td><td>0.676</td><td>0.462</td><td>0.256</td><td>0.435</td><td>-0.588</td><td>0.440</td><td>0.357</td></tr><tr><td>(iv) Job metadata</td><td>3</td><td>0.519</td><td>0.310</td><td>0.290</td><td>0.377</td><td>-4.106</td><td>0.789</td><td>0.188</td></tr><tr><td>(v): (i) + (iv)</td><td>4</td><td>0.687</td><td>0.350</td><td>0.282</td><td>0.386</td><td>-2.391</td><td>0.643</td><td>0.208</td></tr><tr><td>(vi): All DCGM metrics</td><td>13</td><td>0.885</td><td>-0.187</td><td>0.381</td><td>0.323</td><td>-18.483</td><td>1.542</td><td>0.077</td></tr><tr><td>(vii): All DCGM metrics + (iv)</td><td>16</td><td>0.942</td><td>-0.010</td><td>0.351</td><td>0.412</td><td>-104</td><td>3.572</td><td>-0.508</td></tr></table>

As defined in Eq. 1, computing PFI requires power-throughput measurements at a range of power caps, making it difficult to calculate for production jobs. While we find that PFI varies by job type, the power management layer typically does not know which architecture or task is being run on any given hardware node. As a result, the ability to correlate GPU telemetry with task/architecture regime (and thus PFI) is a critical capability for production use.

To build this estimator, we use the feature selection and ablation results presented in Table 3. Specifically, we fit a least-squares linear model using two candidate features derived from DCGM data and motivated by the memory-bound mechanism of power flexibility discussed above: DRAM copy product (the product of DRAM activity and memory copy utilization) and arithmetic intensity (the ratio of tensor pipe activity to DRAM activity). We evaluate each model using both leave-oneout (LOO) and leave-one-cluster-out (LOCO, holding out one architecture-task cluster at a time), reporting in-sample and out-of-sample $R ^ { 2 }$ , RMSE, and Spearman $\rho .$ For completeness, we also show the results of fitting a model on all DCGM metrics as well as ground-truth job metadata (architecture, task, and GPU count).

We find that a simple linear model fit on DRAM copy product alone (shown in Fig. 7) provides the best performance across both LOO and LOCO, and we observe consistent $R ^ { 2 }$ values between the validation data and LOCO analysis. Consistent with our observation that PFI appears to cluster by job type (Dense vs. MoE, pretraining vs. fine-tuning), the ability of DCGM signals like DRAM copy product to predict PFI is likely because these signals provide both a measurable proxy for job type and a means of rank-ordering the expected relative PFI of different jobs. Appendix D.5 provides further sensitivity analysis for this model.

Given the limited number of observations (n = 25 samples of PFI derived from 131 runs), we cannot justify higher-order models (e.g., nonlinear relationships or using more DCGM metrics), but future work may expand to larger sample sizes and different functional forms.

## 6 Exploiting Power Flexibility Index

PFI is a control-oriented metric: our ultimate goal is to allow the infrastructure layer to maximize total tokens/sec throughput under a global power budget by protecting relatively inflexible jobs and assigning deeper power reductions to relatively flexible jobs. To demonstrate this capability, we construct simulated workloads by randomly sampling collections of jobs in our dataset and apply five different strategies to allocate power caps across these jobs:

• Oracle: optimal allocation using perfect knowledge of each job’s throughput-power curve.

• Equal weight: allocate power reductions proportionately to each job’s uncapped power.

• MoE FT-weighted: based on equal weight but doubles the power reduction allocated to MoE fine-tuning jobs (the highest flexibility cluster).

• PFI-aware: adjusts the proportional allocation using PFI, as described below.

• Uncalibrated DCGM (DRAM-copy-product): allocate power reductions proportionately to each job’s DRAM copy product, showing the contrast between the calibrated PFI model and raw telemetry.

![](images/206d3a3b3c238f20e109b1d6cbef86eca04ea0edf5fbc2718c59240de8ec6ee3.jpg)  
Figure 8: Difference in throughput for four allocation strategies compared against an oracle with perfect information of each job’s power-performance curve (less negative = closer to optimal) across a range of curtailment targets and workload mixes (random = jobs sampled at random from our dataset, production = jobs sampled according to the pretraining/fine-tuning mix from recent work [10]).

Denote the power reduction applied to job i as $P _ { \mathrm { c u r t a i l } } ^ { i }$ , the uncapped power and throughput of job i as $P _ { \mathrm { m a x } } ^ { i }$ and $T _ { \mathrm { m a x } } ^ { i }$ , and the PFI of job i as $P F I ^ { i }$ , then:

$$
\mathrm { E q u a l ~ w e i g h t : } ~ P _ { \mathrm { c u r t a i l } } ^ { i } \propto P _ { \mathrm { m a x } } ^ { i }
$$

$$
\mathrm { P F I - a w a r e : ~ } P _ { \mathrm { c u r t a i l } } ^ { i } \propto \frac { P _ { \mathrm { m a x } } ^ { i } } { T _ { \mathrm { m a x } } ^ { i } } { \cal P } { \cal F } I ^ { i }\tag{2}
$$

We simulate the performance of these strategies on 500 synthetic workloads in two scenarios, each with 100 jobs. In the “random” scenario, we sample jobs with replacement from our dataset. In the “production” scenario, we sample jobs according to the pretraining/fine-tuning ratio provided in recent work [10]. We estimate PFI using the model from §5, fitted to telemetry measured on the uncapped run of each job, mirroring the data that would be available from uncapped jobs running in production. We then allocate cluster-level GPU power reductions between 0% and 40% across jobs using each strategy and record the corresponding throughput reduction (simulated by interpolating the measured power-throughput curve for each job).

Fig. 8 shows that the PFI-aware strategy yields equal or higher throughput than the equal weight strategy across all power reduction levels and both workload mix scenarios. While the random job mix includes a roughly equal mix of pretraining and fine-tuning, the production mix [10] includes a large number of small fine-tuning jobs (which tend to have high PFI) and a small number of large pretraining jobs (which tend to be lower PFI). The PFI-aware orchestration strategy is better able to exploit the differences in power elasticity between these job types, leading to substantially improved performance with the production job mix. Under a 30% power reduction on the production mix, PFI-aware power allocation recovers 1.5k tokens/s per job, 63% of the performance gap between an equal-weight allocation and an oracle with perfect information. These results show how PFI can enable power-aware orchestration of LLM training workloads, providing the foundation for data center operators to offer flexible SLAs while preserving token throughput.

## 7 Related Work

Hardware-level power management. GPU power capping and DVFS have been studied as efficiency knobs for over a decade. Tang et al. [27] sweep core and memory frequency on Pascal and Volta GPUs for convolutional networks and identify workload-specific optimal frequencies. Krzywaniak et al. [13] apply NVML power caps during CNN training on a single V100 or A100 and report 22 to 32% energy savings at small slowdowns. Zhao et al. [34] demonstrate that capping V100 power across a mixed HPC and AI workload can be done with minimal overall performance loss, and Costa et al. [6] provide similar results on an exascale GPU system. For LLM workloads, Patel et al. [19] characterize A100 cluster power for both training and inference and show the ability to increase capacity via power-capping-based oversubscription. Ujeniya et al. [28] compare power capping on H100 and H200 and show that memory bandwidth differences affect energy and performance tradeoff under power caps. The HPC community has explored related ideas on CPU-based systems [21]. Prior work establishes power capping as a viable control knob but does not provide a job-level characterization of the power-performance tradeoff across architectures, cluster scales, and modern LLM training tasks.

ML systems optimization. A large body of systems work studies ML throughput or energy optimization under fixed power budgets. The dominant building blocks include tensor and pipeline parallelism (Megatron-LM [25]), sharded optimizer states (ZeRO [24] and FSDP [35]), IO-aware kernels such as FlashAttention [7], and MoE training stacks [15, 8, 11, 9]. Complementary works such as Zeus [32] and Perseus [3] optimize energy–time tradeoffs for training jobs, while ML.ENERGY [4] benchmarks inference energy use and recommends energy-efficient deployment configurations. While these works provide tools for optimizing energy use, our goal in this paper is to provide a characterization of job-level power flexibility that allows a training cluster to dynamically reduce power demand during periods of grid stress (rather than finding an efficient static operating point).

Energy-aware and carbon-aware AI orchestration. A related line of work seeks to reduce the energy or carbon footprint of large-scale computing through job scheduling and control. Carbonaware computing frameworks shift flexible workloads in time or location in response to real-time electricity carbon intensity and power availability [23, 1]. DynamoLLM dynamically adjusts instance counts, routing policies, and GPU frequencies to minimize energy and carbon cost while satisfying latency SLAs in LLM inference clusters [26]. By providing a systematic characterization of the power-throughput response of individual training jobs, our work complements these scheduler-level approaches to workload flexibility.

Demand response and power-flexible AI infrastructure. Data center demand response shifts computing demand in response to grid conditions [30, 16, 33] or oversubscribes infrastructure through dynamic power provisioning [22]. POLCA [20] oversubscribes inference clusters using statistical headroom, while Wang et al. [29] shape AI cluster power profiles for grid response by varying GPU frequency. At the operator and grid layer, field demonstrations [5, 31], together with EPRI’s DCFlex initiative [14], have shown that GPU clusters can function as dispatchable grid resources. As with the energy-aware orchestration methods discussed above, our work complements these approaches by characterizing the job-level power-throughput response, which enables grid-responsive operation with minimal throughput loss.

Positioning of this work. This paper fills an important gap by quantifying the throughput cost of power reductions for specific LLM training jobs. By introducing PFI and presenting the first (to our knowledge) systematic, large-scale characterization of power flexibility in LLM training, our work helps enable power-flexible data center operations by allowing schedulers to optimize power reductions to preserve cluster performance and maintain SLAs. In addition, by demonstrating that we can predict PFI from readily-available DCGM telemetry, we show that our approach is practically implementable and can measurably improve cluster performance on simulated workloads.

## 8 Limitations

Our characterization is scoped to NVIDIA H200 GPUs on a3-ultragpu-8g nodes, uses nvidia-smi -pl as the sole power-control mechanism, and covers pretraining and rank-32 LoRA SFT runs of 30 min each. The 32-GPU runs include only Qwen3-32B pretraining, since rank-32 LoRA SFT is rarely deployed at this scale and our 32-GPU capacity was constrained; broader scaling across architectures remains a natural extension. PFI is computed from aggregate GPU power only: host CPU/DRAM, NIC, PSU, fan, and rack-level cooling are not measured, and broadening the power signal in Eq. 1 as those streams become available is a direct extension.

Each PFI data point represents a single power sweep; a subset of 14 configurations was repeated (35 runs total) and within those repeats steady-state aggregate GPU power and tokens/s agreed with median coefficient of variation 0.4% and 0.5%. The n = 25 PFI sample size still limits crossarchitecture generalization, which we mitigate via held-out validation models. The curtailment simulation in §6 uses the same power-throughput curves used to measure PFI; validation on out-ofsample workloads is left to future work with a larger dataset. Closed-loop scheduler validation of §6 and extensions to other accelerators (including heterogeneous clusters), RLHF, and inference are left to future work.

## 9 Conclusion

While AI data centers can operate as power-aware, grid-responsive loads, doing so while maximizing training throughput requires a principled understanding of how LLM training workloads respond to sustained reductions in available GPU power. In this work, we show that training-job power elasticity can be quantified, estimated at runtime, and exploited for intelligent power-aware workload orchestration.

Broader impact Rapid growth in LLM training is placing significant new demands on electric power systems and is increasingly constrained by limited near-term power availability. By enabling data center operators to reduce power consumption when needed while maximizing throughput and minimizing disruption to customer workloads, this work supports a new class of power-flexible AI infrastructure that can improve utilization of constrained grid capacity, support faster AI deployment timelines, and reduce the infrastructure cost of large-scale AI expansion.

## Acknowledgments and Disclosure of Funding

We would like to acknowledge Fatih Acun and Brian Kulis at Emerald AI for their support and constructive feedback during the review phase.

## References

[1] T. Anderson, A. Belay, M. Chowdhury, A. Cidon, and I. Zhang. Treehouse: A case for carbon-aware datacenter software. SIGENERGY Energy Inform. Rev., 3(3):64–70, Oct. 2023.

[2] A. Chowdhery, S. Narang, J. Devlin, et al. PaLM: Scaling language modeling with pathways. Journal ofMachine Learning Research, 24(240):1–113, 2023. arXiv:2204.02311.

[3] J.-W. Chung, Y. Gu, I. Jang, L. Meng, N. Bansal, and M. Chowdhury. Reducing energy bloat in large model training. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP ’24, page 144–159, New York, NY, USA, 2024. Association for Computing Machinery.

[4] J.-W. Chung, J. J. Ma, R. Wu, J. Liu, O. J. Kweon, Y. Xia, Z. Wu, and M. Chowdhury. The ml.energy benchmark: Toward automated inference energy measurement and optimization. In Advances in Neural Information Processing Systems, Datasets and Benchmarks Track, volume 38. Curran Associates, Inc., 2025.

[5] P. Colangelo, A. K. Coskun, J. Megrue, C. Roberts, S. Sengupta, V. Sivaram, E. Tiao, A. Vijaykar, C. Williams, D. C. Wilson, B. Records, Z. MacFarland, D. Dreiling, N. Morey, A. Ratnayake, and B. Vairamohan. AI data centres as grid-interactive assets. Nature Energy, 11(2):254–261, Feb. 2026.

[6] M. T. Costa, A. Georgiadou, J. B. White, B. V. Alvarez, J. Polo, W. Shin, P. O. A. Navaux, B. Messer, and A. F. Lorenzon. Characterizing the impact of gpu power management on an exascale system. In Proceedings ofthe SC ’25 Workshops ofthe International Conferencefor High Performance Computing, Networking, Storage and Analysis, SC Workshops ’25, page 1524–1533, New York, NY, USA, 2025. Association for Computing Machinery.

[7] T. Dao, D. Fu, S. Ermon, A. Rudra, and C. Ré. Flashattention: Fast and memory-efficient exact attention with io-awareness. In Advances in Neural Information Processing Systems, volume 35, pages 16344–16359. Curran Associates, Inc., 2022.

[8] W. Fedus, B. Zoph, and N. Shazeer. Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity. Journal ofMachine Learning Research, 23(120):1–39, 2022.

[9] T. Gale, D. Narayanan, C. Young, and M. Zaharia. Megablocks: Efficient sparse training with mixture-of-experts. In D. Song, M. Carbin, and T. Chen, editors, Proceedings of Machine Learning and Systems, volume 5, pages 288–304. Curan, 2023.

[10] Q. Hu, Z. Ye, Z. Wang, G. Wang, M. Zhang, Q. Chen, P. Sun, D. Lin, X. Wang, Y. Luo, Y. Wen, and T. Zhang. Characterization of large language model development in the datacenter. In 21st USENIX Symposium on Networked Systems Design and Implementation (NSDI 24), pages 709–729, Santa Clara, CA, Apr. 2024. USENIX Association.

[11] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, H. Chau, P. Cheng, F. Yang, M. Yang, and Y. Xiong. Tutel: Adaptive mixture-of-experts at scale. In Proceedings of Machine Learning and Systems, volume 5, pages 269–287. Curan, 2023.

[12] International Energy Agency. Key Questions on Energy and AI. Technical report, IEA, Paris, Apr. 2026.

[13] A. Krzywaniak, P. Czarnul, and J. Proficz. Gpu power capping for energy-performance tradeoffs in training of deep convolutional neural networks for image recognition. In Computational Science – ICCS 2022, pages 667–681. Springer International Publishing, 2022.

[14] E. Lannoye. DCFlex flex MOSAIC framework. Electric Power Research Institute (EPRI), Mar. 2026.

[15] D. Lepikhin, H. Lee, Y. Xu, et al. GShard: Scaling giant models with conditional computation and automatic sharding. In ICLR, 2021. arXiv:2006.16668.

[16] Z. Liu, M. Lin, A. Wierman, S. Low, and L. L. H. Andrew. Greening geographical load balancing. IEEE/ACM Transactions on Networking, 23(2):657–671, 2015.

[17] T. H. Norris, T. Profeta, D. Patino-Echeverri, and A. Cowie-Haskell. Rethinking Load Growth: Assessing the Potential for Integration of Large Flexible Loads in US Power Systems. Technical Report NI R 25-01, Nicholas Institute for Energy, Environment & Sustainability, Duke University, Durham, NC, Feb. 2025.

[18] NVIDIA Corporation. Data center GPU manager (DCGM). https://developer.nvidia. com/dcgm. Accessed 2026.

[19] P. Patel, E. Choukse, C. Zhang, I. n. Goiri, B. Warrier, N. Mahalingam, and R. Bianchini. Characterizing power management opportunities for llms in the cloud. In Proceedings of the 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, (ASPLOS 24), page 207–222, New York, NY, USA, 2024. Association for Computing Machinery.

[20] P. Patel, C. Zhang, E. Choukse, et al. POLCA: Power oversubscription in LLM cloud providers. arXiv:2308.12908, 2024.

[21] T. Patki, D. K. Lowenthal, A. Sasidharan, M. Maiterth, B. L. Rountree, M. Schulz, and B. R. de Supinski. Practical resource management in power-constrained, high performance computing. In Proceedings ofthe 24th International Symposium on High-Performance Parallel and Distributed Computing, HPDC ’15, page 121–132, New York, NY, USA, 2015. Association for Computing Machinery.

[22] S. Pelley, D. Meisner, P. Zandevakili, T. F. Wenisch, and J. Underwood. Power routing: dynamic power provisioning in the data center. In Proceedings of the Fifteenth International Conference on Architectural Support for Programming Languages and Operating Systems, ASPLOS XV, page 231–242, New York, NY, USA, 2010. Association for Computing Machinery.

[23] A. Radovanovic, R. Koningstein, I. Schneider, B. Chen, A. Duarte, B. Roy, D. Xiao, M. Hari-´ dasan, P. Hung, N. Care, S. Talukdar, E. Mullen, K. Smith, M. Cottman, and W. Cirne. Carbonaware computing for datacenters. IEEE Transactions on Power Systems, 38(2):1270–1280, 2023.

[24] S. Rajbhandari, J. Rasley, O. Ruwase, and Y. He. Zero: Memory optimizations toward training trillion parameter models. In SC20: International Conference for High Performance Computing, Networking, Storage and Analysis, pages 1–16, 2020.

[25] M. Shoeybi, M. Patwary, R. Puri, P. LeGresley, J. Casper, and B. Catanzaro. Megatron-LM: Training multi-billion parameter language models using model parallelism. arXiv:1909.08053, 2019.

[26] J. Stojkovic, C. Zhang, Í. Goiri, J. Torrellas, and E. Choukse. Dynamollm: Designing llm inference clusters for performance and energy efficiency. In 2025 IEEE International Symposium on High Performance Computer Architecture (HPCA), pages 1348–1362, 2025.

[27] Z. Tang, Y. Wang, Q. Wang, and X. Chu. The impact of gpu dvfs on the energy and performance of deep learning: an empirical study. In Proceedings of the Tenth ACM International Conference on Future Energy Systems, e-Energy ’19, page 315–325, New York, NY, USA, 2019. Association for Computing Machinery.

[28] A. Ujeniya, J. Eitzinger, G. Hager, et al. Architectural trade-offs in the energy-efficient era: A comparative study of power-capping NVIDIA H100 and H200. arXiv:2604.11391, 2026.

[29] Y. Wang, Q. Guo, and M. Chen. Providing load flexibility by reshaping power profiles of large language model workloads. Advances in Applied Energy, 19:100232, 2025.

[30] A. Wierman, Z. Liu, I. Liu, and H. Mohsenian-Rad. Opportunities and challenges for data center demand response. In International Green Computing Conference, pages 1–10, 2014.

[31] C. Williams, P. Colangelo, A. Coskun, et al. Power-flexible AI factories: A UK-first demonstration of grid-responsive AI infrastructure. Whitepaper, Emerald AI, EPRI, National Grid, and Nebius, Mar. 2026.

[32] J. You, J.-W. Chung, and M. Chowdhury. Zeus: Understanding and optimizing GPU energy consumption of DNN training. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), pages 119–139, Boston, MA, Apr. 2023. USENIX Association.

[33] Y. Zhang, D. C. Wilson, I. C. Paschalidis, and A. K. Coskun. Hpc data center participation in demand response: An adaptive policy with qos assurance. IEEE Transactions on Sustainable Computing, 7(1):157–171, 2022.

[34] D. Zhao, S. Samsi, J. McDonald, B. Li, D. Bestor, M. Jones, D. Tiwari, and V. Gadepally. Sustainable supercomputing for ai: Gpu power capping at hpc scale. In Proceedings of the 2023 ACM Symposium on Cloud Computing, SoCC ’23, page 588–596. ACM, Oct. 2023.

[35] Y. Zhao, A. Gu, R. Varma, et al. PyTorch FSDP: Experiences on scaling fully sharded data parallel. Proc. VLDB, 2023. arXiv:2304.11277.

## A Alternative definitions of the power flexibility index (PFI)

We consider three candidate definitions of PFI and discuss their tradeoffs.

Ratio (adopted definition) Define PFI at each power cap,

$$
P F I _ { i } = \frac { 1 - P _ { i } / P _ { \operatorname* { m a x } } } { 1 - T _ { i } / T _ { \operatorname* { m a x } } } ,\tag{3}
$$

and take the average over measured caps: $\begin{array} { r } { P F I = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } P F I _ { i } } \end{array}$ (excluding the uncapped point). This definition is simple to compute and has a direct interpretation as the ratio of fractional power reduction to fractional throughput reduction. Its main limitation is sensitivity to the choice of sweep points: if different jobs are measured at different power caps, the averages are not directly comparable. However, Appendix D.3 shows that changing the sweep grid does not change the PFI results significantly (the relative ordering of PFI is preserved under different sweep grids).

Area between curves Define PFI as the area between the normalized power–throughput curve and the unit-slope diagonal,

$$
P F I = \int _ { 0 } ^ { 1 } \left( \frac { T } { T _ { \mathrm { m a x } } } - \frac { P } { P _ { \mathrm { m a x } } } \right) \frac { d P } { P _ { \mathrm { m a x } } } ,\tag{4}
$$

approximated with the trapezoid rule:

$$
P F I \approx \frac { 1 } { 2 T _ { \mathrm { m a x } } P _ { \mathrm { m a x } } } \sum _ { i = 1 } ^ { N } \left[ ( T _ { i } + T _ { i - 1 } ) ( P _ { i } - P _ { i - 1 } ) \right] - \frac { 1 } { 2 } ,\tag{5}
$$

with boundary condition $( P _ { 0 } , T _ { 0 } ) = ( 0 , 0 )$ . This definition is robust to the choice of sample points, but it is less intuitive than the ratio definition.

Constant-elasticity model Fit the log-linear model log $T = \alpha$ log $P +$ log k and define $P F I =$ 1 α, so that the implied power–throughput relationship is $T = k { \check { P } } ^ { 1 - P F I }$ . Under this parameterization, $P F I = 0$ corresponds to throughput proportional to power (fully inflexible), and $P F I = 1$ corresponds to throughput independent of power (fully flexible). The model has a single degree of freedom per job, making it well-suited to sweeps with few power caps; however, we found that this constant-elasticity model was a poor fit to our observed data, as shown in Fig. 9.

Empirically, we find that all three options are highly correlated (see Fig. 10), but we select the proposed ratio definition for two reasons. First, it has a direct interpretation as ratio of power reduction to throughput loss, and the physical interpretation of the other two metrics is less clear. Second, it provides a clear segmentation between power inelastic (PFI > 1) and elastic $( \mathrm { P F I } \leq 1 )$ with a dynamic range ( [1, 2.5]). The area definition also segments (at zero), but we found that it has a very narrow range ( [ 0.1, 0.1]).

![](images/50418952879dcd9edba73187e1f42b85bfb3e52fb87b22bf9b2df6e66f373f63.jpg)  
Figure 9: Comparison of constant-elasticity fit with the observed power-performance curve for gpt-oss-20b fine-tuning on 8×H200s.

![](images/28d471b0cfd580425f56a67fbe100dd8b4edb17c316f84eb2067a3e50268fb36.jpg)  
Figure 10: Comparison of the empirical distribution of the three alternative PFI definitions on the training set cohort.

## B Detailed experimental setup

## B.1 Infrastructure

We ran all experiments on Google Kubernetes Engine (GKE) using a3-ultragpu-8g nodes with 8 NVIDIA H200 GPUs (141 GiB HBM3e, 700 W TDP), fourth-generation NVLink, and a CX-7 RoCE fabric exposing eight GPUDirect-RDMA NICs per node over an MTU-8896 multi-VPC topology. Multi-node runs (16- and 32-GPU) are scheduled through Kueue with Dynamic Workload Scheduler (DWS) Flex-start or Spot capacity. We provisioned clusters in multiple GCP regions based on availability of Flex-start and Spot allocations.

## B.2 Software stack

Workers ran a Docker image (rayproject/ray:2.53.0-py312-cu128) with Ray 2.53.0, Py-Torch 2.10 (CUDA 12.8), Transformers 5.2, Accelerate 1.10, PEFT 0.17, FlashAttention 2.8.1 (cu12/torch2.10 prebuilt wheel), bitsandbytes 0.43.3, and torchtitan 0.2.2. Distributed training uses Ray Train’s TorchTrainer with NCCL backend; FSDP (full-shard, auto-wrap by transformer-decoder layer class) is used for dense jobs and for all LoRA SFT jobs. MoE pretraining is dispatched to a torchtitan subprocess that drives FSDP2 with native Expert Parallelism. We use a flat parallelism mesh: data\_parallel\_shard\_degree equals world\_size (FSDP shards across every rank), with expert parallelism nested inside the FSDP group at degree EP.

GPU telemetry is harvested by an nvcr.io/nvidia/k8s/dcgm-exporter:3.3.7-3.5.0-ubuntu 22.04 sidecar deployed alongside every worker pod and accessed from the Ray container. Prior to collecting data, we verified that logging telemetry using DCGM does not materially affect job performance.

## B.3 Models and datasets

Pretraining and SFT runs use FineWeb-Edu and UltraChat-200K, respectively, pre-tokenized into Ray Data shards in region-local object storage. gpt-oss-20b is dequantized from MXFP4-packed expert weights to BF16. We use the AdamW optimizer with default parameters.

## B.4 Power control mechanism

Per-GPU power caps are set using nvidia-smi -i <local\_rank> -pl <watts>, called at the start of every run (after torch.cuda.set\_device but before any CUDA context allocation). The cap is swept in 100 W increments from 200 W to 700 W on single-node (8-GPU) groups; the 16- and 32-GPU groups sweep 200 W, 500 W, 600 W, and 700 W only. We verify that each power cap was set correctly by reading DCGM\_FI\_DEV\_POWER\_MGMT\_LIMIT and DCGM\_FI\_DEV\_ENFORCED\_POWER\_LIMIT from DCGM during the steady-state phase of each run.

## B.5 Performance data collection

GPU metrics are scraped from the DCGM sidecar at 5 s intervals by a single local\_rank=0 worker per node. System metrics (CPU, RSS, disk and network bytes) are sampled in-process via psutil, also every 5 s. Training metrics are logged on every step by a custom TelemetryCallback and include global step, loss, learning rate, observed tokens-per-second, achieved TFLOPS, and Model FLOPs Utilization (MFU). RDMA counters from the kernel’s InfiniBand interface are appended on RoCE-equipped nodes when present.

## B.6 Measurement protocol

Each configuration is run for 30 minutes. On the HuggingFace Trainer path (dense and LoRA jobs), evaluation is triggered at 10 minutes and a single distributed-checkpoint write at 20 minutes via an all-reduce-synchronized timer callback. The torchtitan EP path triggers eval and checkpoint after fixed step counts chosen so that both events occur at similar times to the dense/LoRA jobs (10 and 20 minutes, respectively). To isolate steady state from transients, we partition traces into three phases: warmup (frame-buffer usage below 10 GiB), startup (weights loaded but gpu\_util = 0), and training (sustained gpu\_util  50%). All aggregates are reported over the training phase only. Per-step throughput, MFU, and per-GPU power are aggregated as the mean of the top 10% of samples within the training phase to exclude checkpoints and evals. Runs at different power caps or hyperparameter values are executed sequentially on a single cluster after a 60 second delay, after verifying that GPU memory has been released.

## B.7 Compute resources

The 131 headline runs and 24 held-out validation runs that produce the figures and tables in this paper consume approximately 1,000 H200 GPU-hours, broken down by cluster size in Table 4.

Table 4: Compute for paper-included runs. Wall-clock figures are the actual DCGM-bracketed durations from the master telemetry frame, not the 30 min training cap.
<table><tr><td>GPUs / cluster</td><td>Runs Mean wall-clock (min)</td><td>GPU-hours</td></tr><tr><td>8</td><td>120 34.9</td><td>558</td></tr><tr><td>16</td><td>19 36.8</td><td>186</td></tr><tr><td>32</td><td>16 38.1</td><td>325</td></tr><tr><td>Total</td><td>155</td><td>35.6 ~1,069</td></tr></table>

The full project consumed additional H200 compute due to preliminary sweeps, hyperparameter tuning, and reruns of jobs that failed mid-training. Across all six profiling regions, 410 distinct H200 submissions produced DCGM telemetry and together consumed approximately 1,725 GPU-hours (Table 5). The 155 reported runs are roughly one third of those submissions and account for 62% of H200 GPU-hours.

Table 5: Compute resources used for this project.
<table><tr><td>Cluster shape</td><td>DCGM-active runs</td><td>GPU-hours</td></tr><tr><td>H200, 8 GPUs</td><td>324</td><td>~969</td></tr><tr><td>H200, 16 GPUs</td><td>53</td><td>~272</td></tr><tr><td>H200, 32 GPUs</td><td>33</td><td>~485</td></tr><tr><td>Total H200 (DCGM-active)</td><td>410</td><td>~1,725</td></tr><tr><td>Reported in paper</td><td>155</td><td>~1,069</td></tr><tr><td>H100, 8 GPUs (reported in Appendix D.2)</td><td>34</td><td>~149</td></tr></table>

Table 6: Power Flexibility Index for each of the headline sweep groups on H200. Each group runs a single job (fixed model, task, cluster size, and training configuration) at multiple power caps; # runs is the number of power-caps measured. 95% CI is estimated from 10,000 Monte Carlo samples for each run, with power and throughput measurements resampled from the empirical noise distribution.
<table><tr><td>Job</td><td># runs</td><td>PFI</td><td>95% CI</td></tr><tr><td>Dense FT H200x16 (Llama-3.1-70B)</td><td>3</td><td>1.10</td><td>[1.096, 1.102]</td></tr><tr><td>Dense FT H200x8 (Llama-3.1-70B, no compile)</td><td>6</td><td>1.22</td><td>[1.218, 1.225]</td></tr><tr><td>Dense FT H200x8 (Llama-3.1-70B, compile)</td><td>6</td><td>1.27</td><td>[1.264, 1.269]</td></tr><tr><td>Dense FT H200x16 (Qwen3-32B)</td><td>4</td><td>1.52</td><td>[1.519, 1.527]</td></tr><tr><td>Dense PT H200x8 (Llama-3.1-70B, compile)</td><td>6</td><td>1.17</td><td>[1.171, 1.177]</td></tr><tr><td>Dense PT H200x32 (Qwen3-32B, seq4096/bs8)</td><td>4</td><td>1.23</td><td>[1.226, 1.230]</td></tr><tr><td>Dense PT H200x32 (Qwen3-32B, seq4096/bs4)</td><td>4</td><td>1.25</td><td>[1.248, 1.251]</td></tr><tr><td>Dense PT H200x8 (Llama-3.1-70B, no compile)</td><td>6</td><td>1.26</td><td>[1.254, 1.260]</td></tr><tr><td>Dense PT H200x32 (Qwen3-32B, seq8192/bs8)</td><td>4</td><td>1.27</td><td>[1.265, 1.270]</td></tr><tr><td>Dense PT H200x8 (Qwen3-32B, no compile)</td><td>6</td><td>1.30</td><td>[1.295, 1.301]</td></tr><tr><td>Dense PT H200x32 (Qwen3-32B, seq8192/bs4)</td><td>4</td><td>1.30</td><td>[1.300, 1.305]</td></tr><tr><td>Dense PT H200x16 (Llama-3.1-70B)</td><td>4</td><td>1.34</td><td>[1.335, 1.342]</td></tr><tr><td>Dense PT H200x16 (Qwen3-32B)</td><td>4</td><td>1.47</td><td>[1.470, 1.478]</td></tr><tr><td>Dense PT H200x8 (Qwen3-32B, compile)</td><td>6</td><td>1.51</td><td>[1.494, 1.518]</td></tr><tr><td>MoE FT H200x8 (gpt-oss-20b, seq2048/bs8)</td><td>6</td><td>1.79</td><td>[1.750, 1.833]</td></tr><tr><td>MoE FT H200x8 (Qwen3-30B-A3B)</td><td>6</td><td>2.00</td><td>[1.976, 2.028]</td></tr><tr><td>MoE FT H200x16 (Qwen3-30B-A3B)</td><td>4</td><td>2.20</td><td>[2.146, 2.244]</td></tr><tr><td>MoE FT H200x8 (gpt-oss-20b, seq4096/bs8)</td><td>6</td><td>2.67</td><td>[2.615, 2.729]</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP1, no compile)</td><td>6</td><td>1.24</td><td>[1.209, 1.268]</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP1, compile)</td><td>6</td><td>1.35</td><td>[1.335, 1.358]</td></tr><tr><td>MoE PT H200x8 (Qwen3-30B-A3B, EP1)</td><td>6</td><td>1.36</td><td>[1.349, 1.370]</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP8, compile)</td><td>6</td><td>1.43</td><td>[1.407, 1.454]</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP4, compile)</td><td>6</td><td>1.45</td><td>[1.435, 1.476]</td></tr><tr><td>MoE PT H200x8 (Qwen3-30B-A3B, EP8)</td><td>6</td><td>1.52</td><td>[1.456, 1.597]</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP8, no compile)</td><td>6</td><td>1.57</td><td>[1.557, 1.576]</td></tr></table>

## C Data tables

Tables 6 and 7 provide the raw PFI data for each sweep group in the headline and validation sets, respectively, including a 95% confidence interval derived using Monte Carlo resampling from the empirical noise distributions derived from intra-run time-series variation (median standard error of the mean 0.04% for power and 0.05% for throughput).

In addition, Figs. 11, 12, and 13 includes the raw and normalized power-performance curves for each job in Tables 6 and 7.

![](images/567583a37510aacef6df4efd039c3e121d3e1bf5cd8edb244afa84108216bb9d.jpg)  
Figure 11: Power-throughput curves (raw and normalized) for each headline and validation run on H200.

![](images/af413addf3e81205082603ff826fb06af217e8fc72848f1c01d7adf4077f61f0.jpg)  
Figure 12: Power-throughput curves (raw and normalized) for each headline and validation run on H200 (continued).

![](images/84d4a2577f8a5dfd07f7e57fedc8fa2d5cfa1be05bc0f24d6cdf7d3b2c0a1780.jpg)  
Figure 13: Power-throughput curves (raw and normalized) for each headline and validation run on H200 (continued).

Table 7: Power Flexibility Index for the 4 held-out validation sweep groups on H200.
<table><tr><td>Job</td><td># runs</td><td>PFI</td></tr><tr><td>Dense FT H200x8 (Mistral-Small-24B-Base-2501)</td><td>6</td><td>1.31</td></tr><tr><td>Dense PT H200x8 (Mistral-Small-24B-Base-2501)</td><td>6</td><td>1.18</td></tr><tr><td>MoE FT H200x8 (gemma-4-26B-A4B)</td><td>6</td><td>2.13</td></tr><tr><td>MoE PT H200x8 (DeepSeek-V2-Lite)</td><td>6</td><td>1.67</td></tr></table>

Table 8: PFI for six matched sweep groups on H100 and H200. All groups use 8 GPUs. Rank ordering is nearly preserved across generations (Spearman $\rho = 0 . 9 4 3 )$ .
<table><tr><td>Arch/Task</td><td>Model</td><td>EP</td><td>Runs (H100)</td><td>PFI (H200)</td><td>PFI (H100)</td></tr><tr><td>Dense PT</td><td>Mistral-Small-24B-Base-2501</td><td>1</td><td>6</td><td>1.18</td><td>1.26</td></tr><tr><td>Dense FT</td><td>Llama-3.1-70B</td><td>1</td><td>4*</td><td>1.22</td><td>1.19</td></tr><tr><td>Dense FT</td><td>Mistral-Small-24B-Base-2501</td><td>1</td><td>6</td><td>1.31</td><td>1.33</td></tr><tr><td>MoE PT</td><td>gpt-oss-20b</td><td>8</td><td>6</td><td>1.43</td><td>1.38</td></tr><tr><td>MoE PT</td><td>DeepSeek-V2-Lite</td><td>8</td><td>6</td><td>1.67</td><td>1.51</td></tr><tr><td>MoE FT</td><td>Qwen3-30B-A3B</td><td>1</td><td>6</td><td>2.00</td><td>1.93</td></tr></table>

<sup>∗</sup>This group contains four runs rather than six: the 500 W and 200 W runs failed with transient errors and could not be re-run within the available allocation.

## D Additional results

## D.1 All DCGM metrics

For completeness, Fig. 14 extends Figs. 5 to all measured DCGM metrics.

## D.2 H100 validation

Our main characterization is confined to H200 GPUs. To test whether the qualitative patterns we report depend on that specific platform, we ran a small set of matched sweeps on H100 GPUs: six sweep groups (34 runs total), with at least one group in each of the four architecture task cells (dense PT, dense FT, MoE PT, MoE FT). Table 8 reports these results along with those of the matched H200 sweeps.

The rank ordering of PFI is largely preserved between H100 and H200 (Spearman $\rho = 0 . 9 4 3 )$ , and we observe that the DCGM-to-PFI model fit on H200 data achieves $R ^ { 2 } = 0 . 5 6$ on the H100 data. The largest absolute difference in PFI between H100 and H200 is 0.16 and the median difference is 3.5%, which is small compared to the between-job PFI range of 1.10 to 2.67 on H200. We observe no systematic direction to the difference in PFI between platforms. The sample size presented here is not sufficient to claim generalizable results on H100 workloads.

## D.3 Sensitivity of PFI to the Power-Cap Sweep Grid

PFI averages the ratio of fractional power reduction to fractional throughput reduction over the sampled power caps, so its value depends on which caps are sampled. Because our 8-GPU groups sweep six caps (200–700 W) while the 16- and 32-GPU groups sweep only four (200, 500, 600, 700 W), comparing PFI between these groups confounds scale with the sweep grid. Table 9 quantifies the size of this effect by providing PFI for every 8-GPU group recomputed using only the four caps present in the coarse grid used for 16- and 32-GPU groups (excluding the 300 W and 400 W measurements). Restricting to the coarse grid increases PFI for all but one group (median increase 0.154, or 10.7%) but does not materially change the ordering of groups (Spearman ρ = 0.940).

We draw two conclusions. First, absolute PFI values are comparable only between groups measured on the same grid, and we accordingly avoid quantitative comparisons of PFI magnitude between the 8-GPU groups and the 16-/32-GPU groups. Second, the rank ordering that our allocation policy relies on is robust to this choice of grid, so PFI remains usable as an ordinal signal across groups measured at different resolutions. While the PFI predictor presented in §5 is trained on the full dataset, our end-to-end results on power allocation in §6 demonstrate that this predictor is still useful in practice (likely because PFI remains correlated across different grids).

Table 9: PFI for 8-GPU sweep groups computed on the full six-cap grid (200–700 W) and on the coarse four-cap grid (200, 500, 600, 700 W) used for the 16- and 32-GPU groups.
<table><tr><td>Sweep group</td><td>PFI (6 caps)</td><td>PFI (4 caps)</td><td>∆</td><td>∆(%)</td></tr><tr><td>Dense FT H200x8 (Llama-3.1-70B, no compile)</td><td>1.22</td><td>1.37</td><td>+0.145</td><td>+11.9</td></tr><tr><td>Dense FT H200x8 (Llama-3.1-70B, compile)</td><td>1.27</td><td>1.44</td><td>+0.177</td><td>+14.0</td></tr><tr><td>Dense PT H200x8 (Llama-3.1-70B, compile)</td><td>1.17</td><td>1.33</td><td>+0.155</td><td>+13.2</td></tr><tr><td>Dense PT H200x8 (Llama-3.1-70B, no compile)</td><td>1.26</td><td>1.43</td><td>+0.177</td><td>+14.1</td></tr><tr><td>Dense PT H200x8 (Qwen3-32B, no compile)</td><td>1.30</td><td>1.42</td><td>+0.121</td><td>+9.3</td></tr><tr><td>Dense PT H200x8 (Qwen3-32B, compile)</td><td>1.51</td><td>1.67</td><td>+0.163</td><td>+10.8</td></tr><tr><td>MoE FT H200x8 (gpt-oss-20b, seq2048/bs8)</td><td>1.79</td><td>1.98</td><td>+0.189</td><td>+10.5</td></tr><tr><td>MoE FT H200x8 (Qwen3-30B-A3B)</td><td>2.00</td><td>2.22</td><td>+0.221</td><td>+11.1</td></tr><tr><td>MoE FT H200x8 (gpt-oss-20b, seq4096/bs8)</td><td>2.67</td><td>3.32</td><td>+0.646</td><td>+24.2</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP1, no compile)</td><td>1.24</td><td>1.16</td><td>-0.082</td><td>-6.6</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP1, compile)</td><td>1.35</td><td>1.40</td><td>+0.055</td><td>+4.1</td></tr><tr><td>MoE PT H200x8 (Qwen3-30B-A3B, EP1)</td><td>1.36</td><td>1.40</td><td>+0.037</td><td>+2.7</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP8, compile)</td><td>1.43</td><td>1.57</td><td>+0.144</td><td>+10.1</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP4, compile)</td><td>1.45</td><td>1.65</td><td>+0.194</td><td>+13.3</td></tr><tr><td>MoE PT H200x8 (Qwen3-30B-A3B, EP8)</td><td>1.52</td><td>1.67</td><td>+0.153</td><td>+10.1</td></tr><tr><td>MoE PT H200x8 (gpt-oss-20b, EP8, no compile)</td><td>1.57</td><td>1.66</td><td>+0.092</td><td>+5.8</td></tr></table>

## D.4 Statistical tests for group difference

We test whether PFI differs across task/architecture regimes using the $n = 2 5$ headline sweep groups. Because dense pretraining and dense LoRA SFT are not separable in our data, we pool them and compare three groups: dense, MoE pretraining, and MoE fine-tuning. We use the Kruskal–Wallis H test followed by Dunn’s post-hoc test over all three pairs with Holm–Bonferroni correction $( m \ : = \ : 3 )$ The Kruskal–Wallis test rejects equality of the three groups $( H = 1 2 . 3 7 , \mathrm { d f = 2 } . $ $p = 0 . 0 0 2 1 )$ . For Dunn’s test, only the dense vs. MoE fine-tuning difference survives p-value correction. MoE pretraining is not separable from either dense or MoE fine-tuning at this sample size. These results indicate that our sample size can statistically distinguish a single high-PFI cluster (MoE fine-tuning) from a continuum of lower-PFI jobs. Tables 10 and 11 provide group medians and pairwise comparisons, respectively.

Table 10: PFI by task/architecture regime across the 25 headline sweep groups.
<table><tr><td>Group</td><td>n</td><td>Median PFI</td><td>Mean PFI</td></tr><tr><td>Dense (PT + LoRA SFT)</td><td>14</td><td>1.27</td><td>1.30</td></tr><tr><td>MoE pretraining</td><td>7</td><td>1.43</td><td>1.42</td></tr><tr><td>MoE fine-tuning</td><td>4</td><td>2.10</td><td>2.17</td></tr></table>

Table 11: Dunn’s post-hoc test, all three pairs, Holm–Bonferroni corrected $( m = 3 )$ . <sup>∗</sup> denotes $p _ { \mathrm { H o l m } } < 0 . 0 5$
<table><tr><td>Comparison</td><td>n</td><td>z</td><td> $p _ { \mathrm { r a w } }$ </td><td> $p _ { \mathrm { H o l m } }$ </td><td> ${ \mathrm { S i g . } }$ </td></tr><tr><td>Dense vs. MoE fine-tuning</td><td>(14, 4)</td><td>-3.44</td><td>0.0006</td><td>0.0017</td><td>*</td></tr><tr><td>MoE pretraining vs. MoE fine-tuning</td><td>(7, 4)</td><td>-1.90</td><td>0.0568</td><td>0.1137</td><td></td></tr><tr><td>Dense vs. MoE pretraining</td><td>(14, 7)</td><td>-1.64</td><td>0.1020</td><td>0.1137</td><td></td></tr></table>

Table 12: Spearman rank correlation between the DRAM copy product and PFI as the k highestleverage sweep groups are removed, over the 25 headline groups. Leverage is computed from the predictor only. Groups are removed in leverage order: Qwen3-30B-A3B (MoE, fine-tuning, 8 GPU), gpt-oss-20b (MoE, fine-tuning, 8 GPU), Qwen3-30B-A3B (MoE, fine-tuning, 16 GPU), Qwen3-32B (dense, pretraining, 8 GPU), Llama-3.1-70B (dense, pretraining, 16 GPU).
<table><tr><td>k dropped</td><td>n</td><td>ρ</td></tr><tr><td>0</td><td>25</td><td>0.535</td></tr><tr><td>1</td><td>24</td><td>0.477</td></tr><tr><td>2 3</td><td>23</td><td>0.406</td></tr><tr><td></td><td>22</td><td>0.321</td></tr><tr><td>4</td><td>21</td><td>0.438</td></tr><tr><td>5</td><td>20</td><td>0.463</td></tr></table>

Table 13: Spearman rank correlation between the DRAM copy product and PFI within architecture subgroups, over the 25 headline sweep groups.
<table><tr><td>Subgroup</td><td>ρ</td><td>n</td></tr><tr><td>Dense only</td><td>-0.178</td><td>14</td></tr><tr><td>MoE only</td><td>0.509</td><td>11</td></tr><tr><td>All except MoE fine-tuning</td><td>0.347</td><td>21</td></tr><tr><td>All groups</td><td>0.535</td><td>25</td></tr></table>

## D.5 Sensitivity analysis of fit PFI model

Table 12 provides a sensitivity analysis for the DRAM copy product-based predictor, dropping high-leverage sweep groups and providing the Spearman correlation between DRAM copy product and PFI at each step.

Table 13 splits the correlation by architecture. We find no correlation within the dense subset, a positive but not significant correlation within the MoE subset $( \rho = 0 . 5 0 9 , n = 1 1 , p = 0 . 1 1 )$ , and a significant correlation over all groups $( \rho = 0 . 5 3 5 , n = 2 5 , p = 0 . 0 0 6 )$ . As a result, we find that contrast between architecture-task clusters drives correlation between PFI and GPU telemetry, rather than variation within clusters.

![](images/63bf739fe96a5ac7034125ddf4013bbdfe260d44c284386ab7c9fcc0b71cd494.jpg)

![](images/5c8ae7b15a69ae12812a1997c105433067f26eebe73108f1c9f551cbe9509941.jpg)

![](images/a1190879b84a21aa7b427bf62a795c75873ca281bbea32c19bfa90a2625ff3bf.jpg)

![](images/c540c4bbd143c67d608b61353d422e435fd59ac8e56e0b5619e8c2cb6f6517da.jpg)

![](images/7aef786d326b1b18d44a6cef91e597431561865c8079e871f7166cb4f7c2aa77.jpg)

![](images/73505bf23baef1bfd55dcf0c09b1071cd8ad769a7b0f13c5d8a8b1649d18c620.jpg)

![](images/879de32a57689e7f6821f35d1880b91509de2907f771e39fb42b1458ed680da1.jpg)

![](images/3e2aa81a56e40dc34ab2b74b5b4b84ecb0a02c1f0c38cfb5dfc11078ec32eaca.jpg)

![](images/13c9442c28468cc3eb164d74dced4dab2fd4892e5b0bf02df2dba0a37dd5a503.jpg)

![](images/ea2c74e325e95dabf7c602914f8c16c93004ecbaeb05ee25e28d93cc5b0298f8.jpg)

![](images/36f3ae08d5d185795b876dc48d882097fece2f2a233977135c6694031cf0d258.jpg)

![](images/8164bbf4344b605fb032e53d1543f55b7d19a3dd382300ebbac6b2d65661246f.jpg)

![](images/42d164f7acf3a06e87db42b4c63747f20347add4d8827a11326ff5db03c7fe83.jpg)

![](images/fe7f2aa0ee854da48c46aa8417ff2ed257df81ed9b1c856a2683a2b0b1559384.jpg)

![](images/d578c9bd867e51a8e45573168f6a887d619b911f4d8befade05883a11dca2024.jpg)

![](images/5aedf32eb98c0ff8839970c71a8ec809430e5b86fbb968e1792213c167684c5d.jpg)

![](images/4c43e22ca5bfa5132c51d6b9ec2a429646bb4be38eb4984e96b8373a62907bf8.jpg)

![](images/7130476d8f358df753b0d8015930d4103f245035cc3f08f6ebf419196f2f1b1c.jpg)

![](images/89350ccaeabd6eed10820448afdd9cfa6ed7d68461e680b445f366c21eca35f9.jpg)

![](images/c128a2a17427332a8fabec7b6396ef7af6363572cd3305378c34c5284412d919.jpg)

![](images/91f72b13b89b961fdd20c822fd661157fb076c02d72140094b038cef91986f9d.jpg)

![](images/282c708e603c9ff2eae907da5519670c34f064e2e11c725581bcb01725381992.jpg)  
Figure 14: PFI vs. all DCGM GPU metrics and composite features. Dashed lines show ordinary least-squares fit; Spearman ρ and significance are annotated. $^ { * } q < 0 . 0 5 , ^ { * * } q < 0 . 0 1$ , where q is the Benjamini–Hochberg FDR-corrected p-value across all metrics in this figure.