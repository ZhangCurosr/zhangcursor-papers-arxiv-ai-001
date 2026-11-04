# CREATING POWER FLEXIBILITY IN DATACENTERS VIA QUERY ROUTING

Shivam Patel <sup>1</sup> <sup>†</sup> Bingqing Chen <sup>2</sup> Jared Fernandez <sup>3</sup> Guannan Qu <sup>1</sup> <sup>\*</sup> Gauri Joshi <sup>1</sup> <sup>\*</sup>

## ABSTRACT

The growing energy requirements of large language model inference highlight the need for datacenters to support safe grid operation by providing power demand flexibility, while maintaining service for interactive workloads. This work investigates query routing as a mechanism to enable such capability by dynamically allocating queries across different language model instances without deferring requests. We develop a general analytical framework that connects query arrival rates and sequence length distributions to token iteration time, request residence time, serving capacity, and power consumption of model instances. Leveraging this framework, we characterize the feasible query routing allocations and the corresponding power demand flexibilities, enabling the selection of routing strategies to achieve desired power consumption behavior. We validate our framework across four GPU platforms on language models ranging from 1B to 70B parameters, obtaining 2.8% mean absolute percentage error in estimating achievable power flexibility. We evaluate our approach on a real trace-driven simulation of a 10,000 H100 GPUs datacenter, exhibiting flexibility up to 16% of total power consumption, with 90% of the desired power reduction achieved within approximately 10 seconds. These results establish query routing as an effective mechanism for datacenter power demand flexibility and demonstrate the broader applicability of our proposed framework to datacenter level planning and orchestration.

## 1 INTRODUCTION

Large language models (LLMs) have experienced rapid development and widespread adoption across question answering, code generation, conversational assistants, and large data applications. The growing demand and evolving use cases for these models require increasing compute capacity, driving the construction of larger GPU datacenters. Recent assessment estimates show that datacenters consumed approximately 485 TWh globally in 2025, averaging a 55 GW power demand, with AI-focused datacenter capacity increasing more than three-fold over the last 18 months (International Energy Agency, 2026). Furthermore, this demand is continuously growing as individual AI campuses are being developed at the gigawatt scale (Crusoe, 2025; Meta, 2026).

The growth of datacenters is also changing the composition of overall electricity demand, with projections estimating that datacenters could account for 9.5–15.3% of U.S. electricity consumption by 2030 (Smith et al., 2026). As these facilities contribute to an increasing proportion of grid demand, their operating characteristics must be incorporated into power-system planning and control to ensure safe and reliable grid operations. In particular, large and abrupt changes in datacenter demand can create imbalance between the generation and load, affecting the grid voltage and distribution frequency. For example, a transmission fault in July 2024 in Northern Virginia caused approximately 1.5 GW of datacenter load to disconnect from the grid, resulting in sudden increase in voltages above normal operating levels, prompting corrective action (NERC, 2025).

![](images/a2789e88c4eb8b80c0cd6418ef770bebb72314c05987d019aa1223a9eb8a1d03.jpg)  
Figure 1. Power demand flexibility through query routing. Routing queries across large and small language models enables datacenters to adjust power demand for safe grid operation without dropping or deferring inference requests.

Integrating datacenters into grid planning and control thus requires characterization of their load behavior, and one of the most important questions that grid operators care about is the demand flexibility of datacenters: how much power can datacenter reduce when requested, during periods of high grid loads, supply shortages, and other contingencies (Norris et al., 2025)? Grid operators are progressively incorporating such power demand flexibility into agreements with datacenters, for example, Google reported contracting 1 GW of datacenter demand response capacity with U.S. utilities in March 2026 (Terrell, 2026). For datacenters serving inference workloads, adhering to these commitments requires reducing power consumption while continuing to serve inference request workloads.

Existing power reduction and curtailment strategies include deferring or rescheduling training and batch inference workloads (Radovanovic et al.´ , 2023; Xu et al., 2025), shifting computation across datacenters (Sukprasert et al., 2024), or adjusting GPU power consumption limits and clock frequencies (You et al., 2023; Chung et al., 2024; Stojkovic et al., 2025b; Hankendi et al., 2026). Demand response frameworks further coordinate power curtailment across query workloads (Xing et al., 2023), and alternative on-site power sources and storage provide additional flexibility (Norris et al., 2025). More discussion on prior work can be found in Appendix A. Deferring query workloads is suitable for delay-tolerant tasks, whereas interactive real-time inference must satisfy latency service level objectives (SLOs). This motivates examining additional sources of power demand flexibility within the inference computational workload. In this work, we aim to achieve power demandflexibilityfor datacenters through query routing.

Our key observation is that the choice of language model for generating response to a query provides a controllable lever for modulating power consumption. Language query routing literature has demonstrated that smaller and efficient models match the response quality of larger models on a broad distribution of queries (Chen et al., 2024; Ding et al., 2024; Ong et al., 2025). Such queries which are adequately served by varying language models present opportunities to dynamically control datacenter level power consumption. Figure 1 describes our approach where we utilize these flexible query–model assignments and existing serving headroom at model instances to predict and control the power demand flexibility by routing queries without dropping or deferring them.

Our contributions are outlined as follows:

• Modeling LLM inference and power consumption. We develop an analytical model (Section 2) that estimates average token iteration time, mean request residence time, serving capacity and power consumption based on query arrival rates and query and response length distributions. This analytical model is validated across different hardware platforms, and language models on real and synthetic workloads. On H100 GPUs, our predictions across configurations demonstrate consistent accuracy with approximately 6% error.

• Power flexibility through query routing. We calculate the feasible routing decisions under query workloads and model instance capacity constraints, and quantify the resulting power demand flexibility. This analysis establishes the effect of query workload and available serving headroom on power flexibility, and enables the selection of routing allocations to attain desired power consumption values. Evaluations substantiate our findings, with the predicted power demand flexibility within approximately 2.8% mean error to the observed values (Section 3).

• Power demand flexibility at datacenter scale. We simulate a deployment of 10,000 H100 GPUs using a week-long trace of inference queries on Azure platform (Stojkovic et al., 2025b). Our approach demonstrates a mean power demand flexibility of up to 1.19 MW (16%) for queries which are indifferent to model choice. We also evaluate a sustained curtailment event over four hours, with our transient response achieving 90% of the target power change in approximately 10 s. Furthermore, we study the impact of GPU resource unavailability and downtime on the available demand flexibility (Section 4).

We organize remainder of the paper as follows. Section 2 introduces the analytical model for inference state and power consumption, Section 3 derives the routing feasibility constraints and power control mechanism. Section 4 evaluates datacenter-scale power control and discusses broader applications, and Section 5 concludes our work. Related work and additional results are presented in Appendices A and B.

## 2 MODELING LLM INFERENCE AND POWER CONSUMPTION

To characterize the inference performance and power consumption during generation, we develop a general analytical model that translates the query workload into serving attributes. First, we model the relationship between the token generation time and the number of requests in each token iteration in Section 2.1, enabling us to estimate request residence time and request throughput. Next, we establish the power consumption profile of a single model instance and use it to estimate the aggregate power consumption across multiple model instances based on query workloads in Section 2.2. This analytical model of inference state and power consumption generalizes across domains and applications, and serves as the foundation for our subsequent analysis of demand flexibility through the job scheduling perspective. In addition to in-text description, Table B.1 summarizes our notation used in the subsequent discussion.

![](images/9193137c1b49fd3f4e17b1b98fe671970d39770175c09709c310f7a69210f0f1.jpg)  
Figure 2. Measured and predicted mean token iteration times (TBT) as a function of number of concurrent requests across models and query length distributions. The linear estimator in eq. (1) accurately captures the observed values $( R ^ { 2 } > 0 . 9 9 $ across all configurations). Larger models exhibit higher TBTs, and workloads with longer requests incur additional attention computation overhead.

## 2.1 Modeling LLM Inference Serving

Query arrival rates, query length distributions, and model instance profiling allows estimation of token iteration time, request residence time, throughput and request concurrency.

Workload and serving state. We consider a model instance with query arrival rate of λ reqs/sec, with ${ \bar { p } } , { \bar { d } }$ denoting the average prompt (prefill) and output (decode) lengths for queries respectively. Let b represent the request concurrency, which includes queries/requests actively undergoing prefill or decode computations, distinguishing it from the number of tokens processed in an iteration. In practice, the request concurrency limit $b _ { \mathrm { m a x } }$ is often controlled as a configurable limit based on memory or generation latency constraints. Let $\tau ( b )$ denote the average token iteration processing time, and ${ \widehat { T } } ( b )$ denote the estimated mean request residence time from beginning prefill computation to generation of last decode token.

Token Iteration Time. We estimate the average token processing time, also referred as time-between-tokens (TBT), for a given number of resident requests through the linear estimator

$$
\tau ( b ) = t _ { 0 } + t _ { 1 } b , \qquad t _ { 0 } > 0 , \quad t _ { 1 } \geq 0 ,\tag{1}
$$

where the intercept term $t _ { 0 }$ captures the fixed computational overhead per token iteration, and the slope $t _ { 1 }$ characterizes the additional computational load for each sequence for a given language model instance.

Figure 2 compares the measured and estimated token iteration processing times for different language models, each hosted on a single H100 GPU. Across all evaluated workloads and language model configurations, our proposed estimator in eq. (1) achieves an $R ^ { 2 }$ value of over 0.99, demonstrating a strong fit to the measured values. We observe that larger models with more parameters lead to longer batch processing times, and query distributions with longer requests incur higher TBT due to increasing per-token attention computation costs. The accurate TBT estimator provides a reliable basis for estimating mean request residence time and request throughput, which we subsequently use to quantify the aggregate power consumption across query workloads.

Request Residence Time. We estimate the mean request residence time, including prefill and decode tokens, by approximating the number of token iterations required for end-to-end generation as

$$
s = \bar { d } + \frac { \bar { p } } { \mathrm { c h u n k _ { p r e } } } , \qquad \widehat { T } ( b ) = s \tau ( b ) .\tag{2}
$$

Here $\bar { d }$ captures the average number of sequential decode iterations, and $\bar { p } / { \mathrm { c h u n k } _ { \mathrm { p r e } } }$ approximates the number of token batches for prefill computation, where $\mathrm { c h u n k _ { p r e } }$ represents the number of prompt tokens processed in parallel in each prefill iteration. Thus, s approximates the total number of token iterations required for generation, and ${ \widehat { T } } ( b )$ estimates the mean request residence time with b active requests. Figure 3 depicts the mean request residence times based on the request concurrency values (b), demonstrating a 5.5% mean absolute percentage error (MAPE) over the examined configurations.

Request Throughput and Concurrency. We use the Little’s law to estimate the mean request concurrency as request throughput times the mean request residence time. Thus, the request throughput at concurrency b is

$$
\widehat { \mu } ( b ) = \frac { b } { \widehat { T } ( b ) } = \frac { b } { s ( t _ { 0 } + t _ { 1 } b ) } , \quad \widehat { C } ( b _ { \mathrm { m a x } } ) = \widehat { \mu } ( b _ { \mathrm { m a x } } ) ,\tag{3}
$$

where $\widehat { C } ( \boldsymbol { b } _ { \mathrm { m a x } } )$ denotes the estimated serving capacity at the maximum concurrency limit, and $\widehat { \mu } ( 0 ) = 0$ . Increasing the concurrency b raises request throughput with diminishing returns, along with increasing the mean token iteration processing times. Figure 4 illustrates the predicted request throughput, with an overall 5.8% MAPE to the observed measurements.

Under stable operation without growing queues, the request throughput $\widehat { \mu } ( b )$ matches the arrival rate λ reqs/s. Rewriting the request throughput (eq. (3)) gives:

$$
\widehat { b } ( \lambda ) = \frac { \lambda s t _ { 0 } } { 1 - \lambda s t _ { 1 } } , \qquad 0 \leq \lambda \leq \widehat { C } ( b _ { \operatorname* { m a x } } ) .\tag{4}
$$

![](images/9662b933257947bec74c589ba9174fd100b7feb9ba5e27093d19ed98ab0c9e9c.jpg)  
Figure 3. Predicted versus measured mean request residence time across configurations. The analytical estimator (eq. (2)) captures the increase in request duration with concurrency, achieving an overall 5.5% MAPE, demonstrating reliable estimation of mean request residence time.

This expression connects the query arrival rate in req/s to the operating state at the model instance, permitting the inference state calculations for token iteration time (eq. (1)) and mean request residence time (eq. (2)). The predicted mean request concurrency depicts an average 9.2% MAPE from the observed values, as presented in Figure 5.

## 2.2 Modeling Power Consumption

Estimating power based on request concurrency connects query arrival rate to model instance power consumption, allowing joint analysis over multiple models.

Model Instance power profiles. We model the mean board power consumed by a model instance as a function of request concurrency by integrating the serving state developed above with power consumption. For a given model instance and query workload, we represent the power as

$$
P ( b ) = P _ { \mathrm { i d l e } } + P _ { \mathrm { d y n } } ( b ) , \qquad P _ { \mathrm { d y n } } ( 0 ) = 0 ,\tag{5}
$$

where $P _ { \mathrm { i d l e } }$ denotes the measured idle power consumption without any active requests (approximately 10 − 20% of rated maximum power consumption). The dynamic component $P _ { \mathrm { d y n } } ( b )$ captures the additional power consumed during active inference and its dependence on the concurrency $b .$ For each paired model and hardware configuration, we calibrate the dynamic component based on observed power consumption values.

![](images/b59e742b9335d09a0bb5b2b405452974d27f732b671e72696ca38b43debaf476.jpg)  
Figure 4. Predicted and measured request throughput as a function of request concurrency across configurations. Throughput increases with concurrency but exhibits diminishing returns as the token iteration times grow. The analytical estimator (eq. (3)) achieves an overall MAPE of 5.8%, accurately capturing the concurrencydependent request throughput of model instances.

Figure 6 illustrates the measured and estimated power consumption values from eq. (5). The total consumed power typically increases sharply from the idle power, followed by a more gradual rise with increasing number of requests served in parallel (b). Across all examined configurations in Figure 6, the analytical power model in eq. (5) achieves a median held-out RMSE of 5.2 W.

Power Consumption and Query Arrival Rate. Utilizing the occupancy estimate in eq. (4) with the model instance power profile (eq. (5)) results in

$$
\widehat { P } ( \lambda ) = P \widehat { ( b ( \lambda ) ) } , \qquad 0 \leq \lambda \leq \widehat { C } ( b _ { \operatorname* { m a x } } ) .\tag{6}
$$

This permits the estimation of power consumption directly from the arrival rate of queries, eliminating the need to measure average number of requests served in parallel at deployment.

Generalization to Multiple Instances. We extend the examination of a single model instance to multiple instances by aggregating power predictions across instances. Consider a pool of model instances $j$ with $n _ { j }$ homogeneous instances serving a total query arrival rate of $\lambda _ { j }$ req/s. The load distribution and scheduling policies determine how requests are distributed among model instances, resulting in predicted request concurrency values for which the aggregate request throughput equals the query arrival rate:

![](images/1fcdfa7dbee6cffe83d628c7e14b81ca281af93884a4a40c6b3f95d6a2731eb7.jpg)  
Figure 5. Estimating request concurrency (eq. (4)) from query arrival rate, demonstrating an overall MAPE of 9.2% over feasible operating points. The predicted request concurrency count connects the query arrival rate to the model instance’s inference and power consumption.

$$
\sum _ { i = 1 } ^ { n _ { j } } \widehat { \mu } _ { j } ( \widehat { b } _ { j , i } ) = \lambda _ { j } , \qquad 0 \leq \widehat { b } _ { j , i } \leq b _ { \operatorname* { m a x } , j } .\tag{7}
$$

The total power consumption is obtained by aggregating the predicted power of all model instances:

$$
{ \widehat { P } } _ { \mathrm { t o t a l } } = \sum _ { j } \sum _ { i = 1 } ^ { n _ { j } } P _ { j } ( { \widehat { b } } _ { j , i } ) .\tag{8}
$$

This formulation captures the varying choices of model instances and job scheduling policies, to predict the total power consumption without requiring additional calibration.

Request Scheduling across Identical Instances. Request scheduling and serving states across homogeneous model instances influence the aggregate power consumption. As observed in Figure 6, the model instance power increases sharply when transitioning from idle to active execution, followed by a more gradual increase with increasing number of concurrently served requests. This implies that consolidating requests onto fewer model instances, up to their concurrency limit $b _ { \mathrm { m a x } }$ , results in lower total power consumption compared to distributing requests evenly across all instances. Thus, request consolidation avoids the substantial idle-to-active power increase incurred for additional model instances, while benefiting from the diminishing incremental power consumption with higher request concurrency.

![](images/5361f198d53a82908ca2dafa69a50fd31b29065163a267b235baa64ac204caf8.jpg)  
Figure 6. Measured and predicted mean GPU board power as a function of request concurrency across configurations. Power consumption increases sharply when transitioning from idle to active inference, followed by a more gradual increase as concurrency grows. The diminishing incremental power at higher concurrency suggests opportunities for reducing aggregate power consumption through request consolidation across model instances.

Experimental setup. Our presented evaluation of the language generation and power consumption analytical model in this section (Figures 2 to 6) utilizes vLLM serving framework (Kwon et al., 2023) on H100 GPUs. We examine models from Llama (Grattafiori & Team, 2024) and Qwen (Yang & Team, 2025) opensource families, with queries sampled from the ShareGPT (Chiang et al., 2023), HumanEval (Chen et al., 2021), arXiv Summarization (Cohan et al., 2018) datasets along with fixed prompt/response length configurations of 256/256, 512/512, and 1024/1024 tokens. We show additional results on other hardware and models served in tensor parallel configuration in Appendix B.

## 3 POWER CONTROL VIA QUERY ROUTING

The analytical model developed in Section 2 connects the query workload to the inference state and power consumption by model instances. Building on this general analytical model, we examine how routing queries across model instances hosting different language models enables control over total power consumption. We first motivate this approach by discussing how query routing can exploit differences in model capabilities while preserving generative performance. We then formalize feasible routing allocations and capacity constraints, characterize the range of power control enabled by feasible routing decisions, and finally present a technique for selecting query-model instance assignments that meet specified power targets.

Query Routing for Power Control. Practical applications of language models span a variety of use cases, hence the query difficulty and model capability requirements vary greatly. A large fraction of queries/requests do not require extensive capabilities of language models, and can instead be served equally well by smaller and more efficient language models. Prior work in language query routing shows that selectively routing queries to smaller models can preserve the generative performance of large language models (Chen et al., 2024; Ding et al., 2024; Ong et al., 2025). This allows us to control total power consumption by scheduling inference queries across model pools, without affecting generative performance.

We primarily study query routing across small and large model pools as an example of this principle, which broadly applies to dynamic inference configurations such as testtime scaling through reasoning token budgets (Snell et al., 2025), multi-response aggregation (Ding et al., 2025) etc.

## 3.1 Formulation and Setup

What feasible routing allocations for queries do existing model instances and their serving capacities permit?

Consider a pool of large and a pool of small model instances. For our analysis, we consider queries classified into three routing classes, where each class denotes the model assignment choice for a given query. Large-only and small-only queries demand generation from respective model pools due to reasoning capability or latency requirements, whereas flexible queries can be served by either pool. Let their workload share in fraction of total queries be $\sigma _ { L } , \sigma _ { S }$ , and $\sigma _ { F }$ respectively, where $\sigma _ { L } + \sigma _ { S } + \sigma _ { F } = 1$ . We assume the aforementioned query assignment is derived from one of the existing query routing methods and focus our scope on characterizing the resulting capacity constraints and power demand flexibility.

For total query arrival rate Λ reqs/s, let $f \in [ 0 , 1 ]$ denote the fraction of flexible requests assigned to the pool of small language model instances. The total query arrival rate at large and small model instances is thus:

$$
\begin{array} { l } { { \lambda _ { L } ( f ) = \Lambda [ \sigma _ { L } + \sigma _ { F } ( 1 - f ) ] , } } \\ { { \lambda _ { S } ( f ) = \Lambda [ \sigma _ { S } + \sigma _ { F } f ] . } } \end{array}\tag{9}
$$

Varying the fraction f controls the distribution of flexible query workload across small and large language model pools along with existing large- and small-only query workloads.

Let $n _ { L } , n _ { S }$ denote the number of large and small model instances, and $C _ { L } , C _ { S }$ denote their per-instance request throughput (reqs/s). To ensure stable workload at each

language model pool, we constrain the routing fraction f to satisfy serving capacity constraints as

$$
\begin{array} { l } { { f _ { \mathrm { l o } } = \mathrm { m a x } \bigg ( 0 , \frac { \displaystyle \Lambda \big ( \sigma _ { L } + \sigma _ { F } \big ) - n _ { L } C _ { L } } { \displaystyle \Lambda \sigma _ { F } } \bigg ) , } } \\ { { f _ { \mathrm { h i } } = \mathrm { m i n } \bigg ( 1 , \frac { \displaystyle n _ { S } C _ { S } - \Lambda \sigma _ { S } } { \displaystyle \Lambda \sigma _ { F } } \bigg ) . } } \end{array}\tag{10}
$$

for $\Lambda > 0$ and $\sigma _ { F } > 0$ . The lower bound represents the minimum fraction of flexible queries that must be assigned to the small language model pool to prevent overloading large language model pool, and vice-versa for the upper bound on routing fraction f. Accordingly, the feasible routing set is $\mathcal { F } = [ f _ { \mathrm { l o } } , f _ { \mathrm { h i } } ]$ when $f _ { \mathrm { l o } } \leq f _ { \mathrm { h i } } ;$ ; otherwise the provisioned model instances lack sufficient capacity to serve the existing query arrival rate under any configuration.

As the total query arrival rate (Λ) increases, the extra headroom at existing model instances for serving flexible queries decreases, constraining the feasible routing decisions. We examine routing decisions on a four model instance pool with three Llama-3.1-8B and one Llama-3.2-1B instance, with 21.75 and 69 req/s serving capacity respectively for queries sampled from the ShareGPT dataset $( \sigma _ { F } = 1 )$ . As we increase the total query arrival rate from 66 to 119 reqs/s, we observe that the feasible routing range shrinks from [0.01, 1] to [0.45, 0.58], indicating greater congestion and reducing headroom at existing model instances.

![](images/92f36bb29f2581d7575dc7a2358198df5747387843a38cbc874812cd779af854.jpg)

![](images/6665d6fe88177a0c4e0fab4a9300569506c79876fd8b986ce2865f480fc712c3.jpg)  
Predicted feasible range Measured feasible range  
Figure 7. Capacity-feasible routing decisions (left, eq. (10)) and corresponding total-power consumption (right, eq. (11)) across query arrival rates and model–workload configurations. Shaded regions show analytical predictions, bars depict measured ranges.

The LLM inference model presented in (eqs. (1) to (3)) provides us with model instance capacities which we utilize in eq. (10) to analytically determine the feasible routing decisions, and compare them with empirically obtained values.

![](images/5a79a23f687d6a7df97ba7a8d5516b80fd07a0c07b996b18f6384f63320e4042.jpg)

![](images/234a6bce59073b97843a1c82986c9cee2d95dad20718d7a654a38156e10998fd.jpg)  
Figure 8. Total power consumption via query routing for the three Llama-8B and single Llama-1B model instances with queries sampled from ShareGPT. Left: measured and predicted total power (eq. (11)) versus the fraction of flexible requests (f) assigned to the smal model instance, with varying query arrival rates. Right: measured and predicted power flexibility range (eq. (12)) over feasible allocations.

Figure 7 demonstrates the accuracy of the estimated feasible routing fraction with respect to the empirically observed values, as the total query arrival rate at model instances increases. Next, we present the effect of feasible routing decisions on power demand flexibility.

## 3.2 Power Flexibility through Query Routing

What is the available power demand flexibility for a given query arrival rate and hardware configuration?

For a chosen pool of language model instances, let $\widehat { P } _ { j } ( \lambda _ { j } )$ denote the predicted aggregate power consumption from eqs. (7) and (8). Substituting the designated query arrival rate at small and large model pools from eq. (9) results in

$$
\widehat { P } _ { \mathrm { t o t a l } } ( \Lambda , f ) = \widehat { P } _ { L } \bigl ( \lambda _ { L } ( f ) \bigr ) + \widehat { P } _ { S } \bigl ( \lambda _ { S } ( f ) \bigr ) .\tag{11}
$$

Query routing affects the power consumed through both the request concurrency as well as the number of active model instances. Provisioned model instances without active requests continue to contribute their idle power. For a feasible range of routing decisions, we define the predicted power consumption, $[ \widehat { P } _ { \operatorname* { m i n } } ( \Lambda ) , \widehat { P } _ { \operatorname* { m a x } } ( \Lambda ) ]$ , and available flexibility, $\widehat { B } ( \Lambda )$ , as

$$
\begin{array} { r l } & { \widehat { P } _ { \operatorname* { m i n } } ( \Lambda ) = \underset { f \in \mathcal { F } } { \operatorname* { i n f } } \widehat { P } _ { \mathrm { t o t a l } } ( \Lambda , f ) , } \\ & { \widehat { P } _ { \operatorname* { m a x } } ( \Lambda ) = \underset { f \in \mathcal { F } } { \operatorname* { s u p } } \widehat { P } _ { \mathrm { t o t a l } } ( \Lambda , f ) , } \\ & { \quad \widehat { B } ( \Lambda ) = \widehat { P } _ { \operatorname* { m a x } } ( \Lambda ) - \widehat { P } _ { \operatorname* { m i n } } ( \Lambda ) . } \end{array}\tag{12}
$$

The power flexibility quantifies the available power control at a given query arrival rate and deployment configuration. Figure 7 illustrates the analytical predictions for power range with measured routing sweeps for four model instances, with three large model instances and one small model instance for Llama 3.2-1B/3.1-8B and Qwen3-4B/8B models. Across examined query workloads and model instance configurations, the feasible routing decision estimates are within 3.3% mean absolute error to the empirically observed values, and the predicted power range closely tracks the measured values with approximately 2.8% mean error.

Power Flexibility and Query Arrival Rate. At low query arrival rates, the presented computational workload limits the total power consumption across model instances. Alternatively, high query arrival rates saturate existing model instances, reducing the headroom for power flexibility by query routing. Therefore, intermediate arrival rates demonstrate the highest power demand flexibility for a pool of model instances.

Figure 8 depicts the total power consumed and the power demand flexibility across four model instances (three Llama3.1-8B, one Llama3.2-1B and $\sigma _ { F } = 1 )$ as the query arrival rate varies from 10% to 90% of the measured capacities. As the query arrival rate increases, the feasible set of routing decisions shrinks (also demonstrated in Figure 7), and the power consumed at a given routing fraction increases due to higher computational demand from the existing model instances. Additionally, the power demand flexibility is largest at intermediate query arrival rates, and decreases at higher or lower arrival rates.

## 3.3 Selecting Routing Allocations for Power Targets

How do we select the desired query routing allocation?

Given a target aggregate power consumption $P _ { \mathrm { t a r g e t } }$ , we leverage the analytical power model (eq. (11)) to identify a capacity-feasible routing allocation that minimizes the deviation from the desired power target:

$$
f ^ { * } = \arg \operatorname* { m i n } _ { f \in \mathcal { F } } \left| \widehat { P } _ { \mathrm { t o t a l } } ( \Lambda , f ) - P _ { \mathrm { t a r g e t } } \right| .\tag{13}
$$

We search for the required routing decision $f ^ { * }$ over the feasible routing interval ${ \mathcal F } ,$ resulting in an allocation which provides the closest achievable power level under the existing workload and model instance configurations. As query arrival rates, feasible routing decisions or available serving capacities change, the allocation can be recomputed using updated estimates from the analytical model.

Routing decisions are applied to new arriving queries, while the existing responses already being generated at model instances are unaffected. Consequently, the total power consumption does not change instantaneously following a change in the routing decisions. As discussed in Appendix B.2, the time required for these transitions depends on the completion time of outstanding requests, which is roughly in the order of seconds (Kwon et al., 2023).

Extension to Multiple Models and Inference Budgets. Our formulation extends to multiple language model pools by replacing the scalar routing fraction $f$ with an allocation vector which specifies query assignments across eligible model instance pools. Feasible allocations must satisfy respective capacity constraints, and subsequently the aggregate power and flexibility across model instances is estimated through eqs. (11) and (12). This principle can also be extended to inference configurations with varying test-time compute budgets, including but not limited to reasoningtoken budgets and multiple sampled responses. Since the benefits of additional computation depend on query difficulty (Snell et al., 2025), adaptive inference configurations can provide additional power flexibility while preserving generative quality (Ding et al., 2025).

In the next section, we discuss how query routing across model instances can support power demand flexibility and management at datacenter scale.

## 4 DATACENTER-SCALE POWER CONTROL

Increasingly large datacenter workloads and their contribution to grid power demand has resulted in exploration of datacenter power control for supporting grid operations (Terrell, 2026). In this section, we study query routing across heterogeneous language models as a source of demand flexibility for maintaining grid compliance and versatility. We build upon the mechanisms developed in Section 3 and examine them at scale to demonstrate the practical import of our approach. In Section 4.1, we describe the setup and design choices for our analysis, Section 4.2 presents empirical results on a real-world workload trace, and Section 4.3 comments on broader applications and scope.

## 4.1 Empirical Setup

We simulate a datacenter-scale deployment with 10, 000 Nvidia H100 GPUs, divided in a 3 : 1 ratio between large and small model instances respectively. We examine Llama-3.1-70B (Grattafiori & Team, 2024) served with tensor parallel configuration across four GPUs as the large model instance, and Mistral-24B (Mistral-AI, 2025) served on a single GPU as the small model instance (Appendix B.1). Our examined empirical results build on the inference and power consumption modeling for these language models as presented in Section 2.

To evaluate our approach under practical scenarios, we utilize a week-long Azure LLM chat conversation trace (Stojkovic et al., 2025b), with mean prompt and response lengths of 1630 and 105 tokens respectively. For this workload, the measured per-instance throughput is 7 reqs/s for the larger model instance (Llama-70B), and 6.25 reqs/s for the smaller model instance (Mistral-24B). We scale the query arrival rates such that the peak five-minute mean arrival rate over the week reaches 60% of the total aggregate serving capacity across all model instances.

For the baseline operation, the query proportions for large and small model pools follow their aggregate serving capacities, i.e., $( \sigma _ { L } + \sigma _ { F } ( 1 - f ) ) / ( \sigma _ { S } + \sigma _ { F } f ) = n _ { L } C _ { L } / n _ { S } C _ { S } ,$ following eq. (9). Under this allocation, both the large and small model pools operate at the same fraction of their respective capacities. We vary the fraction of flexible queries (σ<sub>F</sub>) in the total workload to evaluate the resulting power demand flexibility and available curtailment. For each $\sigma _ { F }$ value, the remaining workload among large-only $( \sigma _ { L } )$ and small-only $( \sigma _ { S } )$ queries is maintained in proportion to their corresponding capacities, such that $\sigma _ { L } / \sigma _ { S } = n _ { L } C _ { L } / n _ { S } C _ { S }$ During power curtailment events, flexible queries are progressively directed toward the smaller language model’s instances, subject to available headroom and feasibility constraints defined in eq. (10).

In datacenter establishments, components such as thermal regulation, networking equipment, storage systems etc consume additional power along with GPUs (Patel et al., 2024; Stojkovic et al., 2024; 2025a). Furthermore, power conversion and distribution losses introduce overheads which are captured through PUE (Power Usage Effectiveness) metrics. For our analysis, we model these workloads through an additional 300W per GPU, and an aggregate PUE value of 1.2, implying that the actual datacenter facility power is

$$
\widehat { P } _ { \mathrm { f a c i l i t y } } ( \Lambda , f ) = 1 . 2 \left[ \widehat { P } _ { \mathrm { t o t a l } } ( \Lambda , f ) + 3 . 0 \mathrm { M W } \right] .\tag{14}
$$

We examine three aspects of datacenter power demand flexibility through the aforementioned setup. First, we quantify the trends in available power demand flexibility over the week-long real workload trace. Second, we simulate a fourhour curtailment event to assess the achievable reduction and the observed response times. Third, we analyze how changes in GPU availability, including allocation to training jobs or temporary outages, affect the available flexibility.

![](images/3095c802a73d1a17d5e674e03beb14d4cc5c36146190cd586e4940d25a2f1218.jpg)  
Figure 9. Datacenter power demand flexibility over week-long Azure LLM workload trace. Top: Baseline facility power and minimum achievable power under varying fractions of flexible queries. Bottom: Distribution and hourly variation of available power curtailment, highlighting its dependence on query workload.

## 4.2 Power Flexibility under Real Workloads

Available power flexibility over time. Query arrival rate varies with time, changing both the datacenter facility power consumption and the demand flexibility achievable through routing. We therefore examine a real-world week-long trace to characterize the distribution of available power control throughout the week and hourly averages during the day.

Figure 9 (top) depicts the datacenter facility power consumption (eq. (14)) throughout the week, along with available power curtailment distribution values with varying fraction of flexible query workloads. The baseline power consumption varies from 6 to 9.1 MW, with a mean of 7.3 MW. With half of the query workload suited to both large and small model instances $( \sigma _ { F } = 0 . 5 )$ , query routing across language models results in a power curtailment of 0.6 MW, which increases to 1.19 MW for $\sigma _ { F } = 1$ , corresponding to 16% of the mean facility power.

Furthermore, the availability varies across the week based on distribution of the query arrival rate (bottom Figure 9). With half of all queries as flexible, atleast 0.32 MW power curtailment is available for 95% of the week. Additionally, high query arrival rates during evening hours result in lower power flexibility of 0.49 MW as compared to the weekly mean of 0.6 MW. Thus, the power demand flexibility benefits from larger headroom during periods of intermediate query arrival rates, and improves with more flexible queries.

![](images/38057e8d6b98bc1a6f7cb23393ee23c463102f02a5d36cd5d771aa8b679209c7.jpg)  
Figure 10. Datacenter power control during a four-hour curtailment event. Top: Sustained power reductions achieved through flexible query routing. Bottom: Event boundaries showing fast transitions between operating points in the order of seconds.

Practical power curtailment event. During periods of high electricity demand or grid disturbances, grid operators may request large loads to curtail their power usage for time periods up to hours. Responding to such curtailment events requires data centers to respond under short notice and sustain the reduction throughout the window, while continuing to serve incoming queries.

We simulate a power demand curtailment event for a fourhour duration in Figure 10 (top half), and inspect the adequacy of our approach for controlling the datacenter facility power. The mean facility power consumption for the baseline allocation throughout this window is 7.28 MW. At $\sigma _ { F } = 0 . 5$ , the average curtailed power consumption is 6.7 MW and 6.1 MW at $\sigma _ { F } = 1$ , yielding approximately 2.4 and 4.8 MWh energy reduction during the curtailment event for ensuring grid compliance.

Figure 10 (bottom half) also illustrates the power control response transients between distinct operating points in the routing decision space. For the query prompt and output lengths sampled from the Azure trace (Stojkovic et al., 2025b), we observe a transient power response time of approximately 10s to achieve 90% of the desired power reduction. Thus, query routing across heterogeneous language model instances enables response to grid events within seconds, and can sustain lower power consumption values as query workload and available GPU resources permit.

![](images/4cfb547d6f126759ee95c8a3ffe784c51dccbcedbb8f78fb7b781531462e611b.jpg)

![](images/40dde8727e4f2ea6eab9fe3dc96c836b2bc1aa9bf20a95aeaab2eba4deb744cd.jpg)

![](images/84a79fcb8e220de66c9f5e3dfa72579d91135c6c86e7a181181c0e350cb62172.jpg)  
Figure 11. Impact of GPU availability and resource sharing on power demand flexibility. Top: Achievable power curtailment under reduced GPU availability and the resulting utilization during scheduled sharing. Bottom: Daily power flexibility with GPU sharing, demonstrating resilience to moderate resource constraints.

GPU availability and resource sharing. The available resources for serving inference queries at a datacenter can vary as some GPUs may be allocated to training jobs and other workloads, or may be inaccessible due to maintenance or temporary failures. We therefore examine the power demand flexibility for a given query workload under varying resource availability.

Figure 11 (top left) demonstrates the average power demand flexibility against the available GPU resources. We observe that moderate resource unavailability preserves most of the demand flexibility, which gradually decays at higher GPU constraints. This robustness of the demand flexibility to GPU unavailability reflects the existing headroom available, and reducing the availability reduces the small model pool’s capacity to accommodate the flexible query workload, reducing the overall power curtailment capacity. Further unavailability eventually leads to violation of capacity and feasibility constraints (eq. (10)), leading to queue buildup and service denial for incoming queries.

We also simulate a scheduled resource sharing scenario, where 40% GPUs are unavailable for inference serving from 8pm to 8am. Figure 11 shows the resulting resource utilization and available power flexibility. With half the requests as flexible, we preserve up to 99% of the mean demand flexibility, but we lose a higher fraction of the demand flexibility for the full-flexible query case.

## 4.3 Broader Applications and Scope

The analytical model for autoregressive inference and power consumption developed in Section 2 supports designing serving systems at scale. Predictions of inference behavior can help guide appropriate choice of number of model instances of each type, deployment comparisons, and GPUsharing schedules. Along with measured power consumption profiles, these predictions bridge the gap between inference service requirements and feasible power constraints and grid compliance. Table 1 summarizes potential applications of our approach.

Extending the framework to training regimes or data storage scenarios can enable joint resource planning under common power objectives. For example, datacenters can orchestrate GPU sharing for training jobs, or alternate power sources and battery support when query routing alone cannot satisfy a curtailment target. Electricity costs and carbon-intensity goals also support comparing actions by cost and emissions.

Our broad formulation readily extends to multiple models, serving quantization levels, and reasoning budgets, where each choice can be characterized by the associated model instance profiles and query workloads and feasibility. For distributed datacenters, local grid constraints may lead to routing across regions while respecting network latencies and data-privacy constraints.

Table 1. Broader applications of our presented framework.
<table><tr><td>Domain</td><td>Application</td></tr><tr><td>Serving design</td><td>Model instance pool sizing, deployment comparisons, demand growth headroom.</td></tr><tr><td>Power Control</td><td>Feasible targets, sustained power curtail- ments, and transition planning.</td></tr><tr><td>Energy coordination</td><td>Joint inference/training budgets, storage support, and cost or carbon objectives.</td></tr><tr><td>Broader routing</td><td>Allocation across execution choices or lo- cations, subject to eligibility constraints.</td></tr></table>

## 5 CONCLUSION

In this work, we investigate the potential of query routing for datacenter power flexibility to support safe grid operation. We present a comprehensive analytical model for autoregressive inference and power consumption, allowing us to connect query workload to GPU power demand. Utilizing this, we examine the arising power demand flexibility in datacenter scale operations under practical scenarios. More broadly, this work represents an initial step toward the integration of inference systems and power-aware datacenter operations. We envision that insights and framework developed in this work will motivate a large body of research on grid-responsive AI infrastructure, enabling future advances in workload orchestration, power demand management, and reliable integration of large-scale computing into the grid.

## ACKNOWLEDGMENTS

This work was supported by a PITA Seed Grant, and Bosch RTC Pittsburgh. GJ is supported by an AI2C Seed grant and the US National Science Foundation under grants CCF 2045694, CNS-2112471, CPS-2111751, and ONR N00014- 23-1-2149. GQ is supported by NSF Grants 2339112, 2512805, AFOSR Grant FA9550261B281, Jane Street, and CMU Manufacturing Futures Institute.

## REFERENCES

Agrawal, A., Kedia, N., Mohan, J., Panwar, A., Kwatra, N., Gulavani, B. S., Ramjee, R., and Tumanov, A. Vidur: A large-scale simulation framework for llm inference. In Gibbons, P., Pekhimenko, G., and Sa, C. D. (eds.), Proceedings ofMachine Learning and Systems, volume 6, pp. 351–366, 2024a.

Agrawal, A., Kedia, N., Panwar, A., Mohan, J., Kwatra, N., Gulavani, B., Tumanov, A., and Ramjee, R. Taming Throughput-Latency tradeoff in LLM inference with Sarathi-Serve. In 18th USENIX Symposium on Operating Systems Design and Implementation (OSDI 24), pp. 117–134, Santa Clara, CA, July 2024b. USENIX Association. ISBN 978-1-939133-40-3. URL https: //www.usenix.org/conference/osdi24/p resentation/agrawal.

Aniti, L. Solar and wind power curtailments are increasing in California. Today in Energy, U.S. Energy Information Administration, May 2025. URL https://www.eia. gov/todayinenergy/detail.php?id=65364. Published May 28, 2025.

Chen, H., Liu, X., Liu, Y., Jiang, J., He, B., and Liu, X. The 1/w law: An analytical study of context-length routing topology and gpu generation gains for llm inference energy efficiency, 2026. URL https://arxiv.org/ abs/2603.17280.

Chen, L., Zaharia, M., and Zou, J. FrugalGPT: How to use large language models while reducing cost and improving performance. Transactions on Machine Learning Research, 2024. ISSN 2835-8856. Featured Certification.

Chen, M., Tworek, J., Jun, H., Yuan, Q., de Oliveira Pinto, H. P., Kaplan, J., Edwards, H., Burda, Y., Joseph, N., Brockman, G., Ray, A., Puri, R., Krueger, G., Petrov, M., Khlaaf, H., Sastry, G., Mishkin, P., Chan, B., Gray, S., Ryder, N., Pavlov, M., Power, A., Kaiser, L., Bavarian, M., Winter, C., Tillet, P., Such, F. P., Cummings, D., Plappert, M., Chantzis, F., Barnes, E., Herbert-Voss, A., Guss, W. H., Nichol, A., Paino, A., Tezak, N., Tang, J., Babuschkin, I., Balaji, S., Jain, S., Saunders, W., Hesse, C., Carr, A. N., Leike, J., Achiam, J., Misra,

V., Morikawa, E., Radford, A., Knight, M., Brundage, M., Murati, M., Mayer, K., Welinder, P., McGrew, B., Amodei, D., McCandlish, S., Sutskever, I., and Zaremba, W. Evaluating large language models trained on code, 2021. URL https://arxiv.org/abs/2107.0 3374.

Chiang, W.-L., Li, Z., Lin, Z., Sheng, Y., Wu, Z., Zhang, H., Zheng, L., Zhuang, S., Zhuang, Y., Gonzalez, J. E., Stoica, I., and Xing, E. P. Vicuna: An open-source chatbot impressing gpt-4 with 90%\* chatgpt quality, March 2023. URL https://lmsys.org/blog/2023-03-3 0-vicuna/.

Choukse, E., Warrier, B., Heath, S., Belmont, L., Zhao, A., Khan, H. A., Harry, B., Kappel, M., Hewett, R. J., Datta, K., Pei, Y., Lichtenberger, C., Siegler, J., Lukofsky, D., Kahn, Z., Sahota, G., Sullivan, A., Frederick, C., Thai, H., Naughton, R., Jurnove, D., Harp, J., Carper, R., Mahalingam, N., Varkala, S., Kumbhare, A. G., Desai, S., Ramamurthy, V., Gottumukkala, P., Bhatia, G., Wildstone, K., Olariu, L., Incorvaia, I., Wetmore, A., Ram, P., Raghuraman, M., Ayna, M., Kendrick, M., Bianchini, R., Hurst, A., Zamani, R., Li, X., Petrov, M., Oden, G., Carmichael, R., Li, T., Gupta, A., Patel, P., Dattani, N., Marwong, L., Nertney, R., Kobayashi, H., Liott, J., Enev, M., Ramakrishnan, D., Buck, I., and Alben, J. Power stabilization for AI training datacenters, August 2025. URL https://arxiv.org/abs/2508.14318.

Chung, J.-W., Gu, Y., Jang, I., Meng, L., Bansal, N., and Chowdhury, M. Reducing energy bloat in large model training. In Proceedings of the ACM SIGOPS 30th Symposium on Operating Systems Principles, SOSP ’24, pp. 144–159, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400712517. doi: 10.1145/3694715.3695970. URL https: //doi.org/10.1145/3694715.3695970.

Chung, J.-W., Ma, J. J., Wu, R., Liu, J., Kweon, O. J., Xia, Y., Wu, Z., and Chowdhury, M. The ML.ENERGY benchmark: Toward automated inference energy measurement and optimization. In The Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track, 2026. URL https: //openreview.net/forum?id=CXqq2v6qUv.

Cohan, A., Dernoncourt, F., Kim, D. S., Bui, T., Kim, S., Chang, W., and Goharian, N. A discourse-aware attention model for abstractive summarization of long documents. In Walker, M., Ji, H., and Stent, A. (eds.), Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), pp. 615–621, New Orleans, Louisiana, June 2018. Association for Computational Linguistics. doi:

10.18653/v1/N18-2097. URL https://aclantho logy.org/N18-2097/.

Colangelo, P., Coskun, A. K., Megrue, J., Roberts, C., Sengupta, S., Sivaram, V., Tiao, E., Vijaykar, A., Williams, C., Wilson, D. C., MacFarland, Z., Dreiling, D., Morey, N., Ratnayake, A., and Vairamohan, B. Turning AI data centers into grid-interactive assets: Results from a field demonstration in phoenix, arizona. arXiv:2507.00909, 2025. URL https://arxiv.org/abs/2507.0 0909.

Crusoe. Crusoe Expands AI Data Center Campus in Abilene to 1.2 Gigawatts, March 2025. URL https://www. crusoe.ai/resources/newsroom/crusoe-e xpands-ai-data-center-campus-in-abile ne-to-1-2-gigawatts. March 18, 2025.

de Vries, A. The growing energy footprint of artificial intelligence. Joule, 7(10):2191–2194, 2023. doi: 10.101 6/j.joule.2023.09.004. URL https://doi.org/10 .1016/j.joule.2023.09.004.

Ding, D., Mallick, A., Wang, C., Sim, R., Mukherjee, S., Ruhle, V., Lakshmanan, L. V. S., and Awadallah, A. H.¨ Hybrid LLM: Cost-efficient and quality-aware query routing. In The Twelfth International Conference on Learning Representations, 2024.

Ding, D., Mallick, A., Zhang, S., Wang, C., Madrigal, D., Garcia, M. D. C. H., Xia, M., Lakshmanan, L. V. S., Wu, Q., and Ruhle, V. BEST-route: Adaptive LLM¨ routing with test-time optimal compute. In Forty-second International Conference on Machine Learning, 2025.

ERCOT. ERCOT Nodal Operating Guides. Electric Reliability Council of Texas, February 2026a. URL https://www.ercot.com/files/docs/202 6/01/29/February-1-2026-Nodal-Operati ng-Guide.pdf. February 1, 2026 edition; Section 2, especially Sections 2.3.1.2 and 2.3.3.1.

ERCOT. ERCOT Nodal Protocols. Electric Reliability Council of Texas, 2026b. URL https://www.er cot.com/mktrules/nprotocols/current. Sections 3 and 8, August 1, 2026 versions. See Sections 3.14.3, 3.17, and 8.1.1.2.1. Accessed October 25, 2026.

Grattafiori, A. and Team, L. The llama 3 herd of models, 2024. URL https://arxiv.org/abs/2407.2 1783.

Hankendi, C., Shahout, R., Yu, M., and Coskun, A. K. Pals: Power-aware llm serving for mixture-of-experts models, 2026. URL https://arxiv.org/abs/2605.2 1427.

Hugging Face. AI Energy Score: Benchmarking the energy efficiency of ai models, 2025. URL https://huggin gface.github.io/AIEnergyScore/.

International Energy Agency. Key questions on energy and ai. Technical report, International Energy Agency (IEA), Paris, 2026. URL https://www.iea.org/re ports/key-questions-on-energy-and-ai. Licence: CC BY 4.0.

Kwon, W., Li, Z., Zhuang, S., Sheng, Y., Zheng, L., Yu, C. H., Gonzalez, J., Zhang, H., and Stoica, I. Efficient memory management for large language model serving with pagedattention. In Proceedings of the 29th Symposium on Operating Systems Principles, SOSP ’23, pp. 611–626, New York, NY, USA, 2023. Association for Computing Machinery. ISBN 9798400702297. doi: 10.1145/3600006.3613165. URL https://doi. org/10.1145/3600006.3613165.

Luccioni, A. S., Viguier, S., and Ligozat, A.-L. Estimating the carbon footprint of bloom, a 176b parameter language model, 2022. URL https://arxiv.org/abs/22 11.02001.

Meta. Inside prometheus: Meta’s journey to a gigawatt-scale supercluster. Engineering at Meta, October 2026. URL https://engineering.fb.com/2026/10/1 9/data-center-engineering/inside-pro metheus-meta-gigawatt-scale-superclus ter/. Accessed: 2026-10-28.

Mistral-AI. Magistral, 2025. URL https://arxiv.or g/abs/2506.10910.

National Grid Electricity Transmission plc. Demand Turn Up (DTU): Interactive Guidance Document and Invitation to Tender, February 2018. URL https://www. neso.energy/document/88466/download. Version 1.0, February 20, 2018.

NERC. Incident review: Considering simultaneous voltagesensitive load reductions. Technical report, North American Electric Reliability Corporation, January 2025. URL https://www.nerc.com/globalassets/ou r-work/reports/event-reports/inciden t\_review\_large\_load\_loss.pdf. Published January 8, 2025.

Norris, T., Profeta, T., Patino-Echeverri, D., and Cowie-Haskell, A. Rethinking load growth: Assessing the potential for integration of large flexible loads in us power systems. 2025.

Ong, I., Almahairi, A., Wu, V., Chiang, W.-L., Wu, T., Gonzalez, J. E., Kadous, M., and Stoica, I. Routellm: Learning to route llms from preference data. In Yue, Y.,

Garg, A., Peng, N., Sha, F., and Yu, R. (eds.), International Conference on Learning Representations, volume 2025, pp. 34433–34448, 2025.

Oviedo, F., Kazhamiaka, F., Choukse, E., Kim, A., Luers, A., Nakagawa, M., Bianchini, R., and Ferres, J. M. L. Energy use of ai inference, efficiency pathways, and testtime scaling. Joule, 2026.

Patel, P., Choukse, E., Zhang, C., Goiri, <sup>´</sup>I., Warrier, B., Mahalingam, N., and Bianchini, R. Polca: Power oversubscription in llm cloud providers, 2023. URL https://arxiv.org/abs/2308.12908.

Patel, P., Choukse, E., Zhang, C., Goiri, <sup>´</sup>I., Warrier, B., Mahalingam, N., and Bianchini, R. Characterizing power management opportunities for llms in the cloud. In Proceedings ofthe 29th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 3, ASPLOS ’24, pp. 207–222, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400703867. doi: 10.1145/3620666.3651329. URL https://doi. org/10.1145/3620666.3651329.

Patel, P., Choukse, E., Zhang, C., Shah, A., Goiri, <sup>´</sup>I., Maleki, S., and Bianchini, R. Splitwise: Efficient generative llm inference using phase splitting. In Proceedings of the 51st Annual International Symposium on Computer Architecture, ISCA ’24, pp. 118–132. IEEE Press, 2025. ISBN 9798350326581. doi: 10.1109/ISCA59077.2024 .00019. URL https://doi.org/10.1109/ISCA 59077.2024.00019.

PJM Interconnection. PJM Manual 11: Energy & Ancillary Services Market Operations. PJM Interconnection, revision 137 edition, July 2026a. URL https: //www.pjm.com/-/media/DotCom/documen ts/manuals/m11.pdf. Effective July 28, 2026; Sections 3 and 4, especially Section 4.5.

PJM Interconnection. PJM Manual 12: Balancing Operations. PJM Interconnection, revision 57 edition, April 2026b. URL https://www.pjm.com/-/media/ DotCom/documents/manuals/m12.pdf. Effective April 22, 2026; Sections 4.3 and 4.4.

PJM Interconnection. PJM Manual 13: Emergency Operations. PJM Interconnection, revision 98 edition, June 2026c. URL https://www.pjm.com/-/media/ DotCom/documents/manuals/m13.pdf. Effective June 24, 2026; Section 2.

PJM Interconnection. PJM Manual 18: PJM Capacity Market. PJM Interconnection, revision 63 edition, September 2026d. URL https://www.pjm.com/-/media/ DotCom/documents/manuals/m18.pdf. Effective September 24, 2026; Section 4.3.

Public Utility Commission of Texas. Order in PUC Docket No. 58881. Technical Report Docket 58881, Item 105, Public Utility Commission of Texas, April 2026. URL https://interchange.puc.texas.gov/Do cuments/58881\_105\_1628395.PDF. April 23, 2026; Ordering Paragraph 1, Condition 1, pp. 18–19.

Qiu, H., Mao, W., Patke, A., Cui, S., Jha, S., Wang, C., Franke, H., Kalbarczyk, Z., Bas¸ar, T., and Iyer, R. K. Power-aware deep learning model serving with mu-Serve. In 2024 USENIX Annual Technical Conference (USENIX ATC 24), pp. 75–93, Santa Clara, CA, July 2024. USENIX Association. ISBN 978-1-939133-41-0. URL https: //www.usenix.org/conference/atc24/pr esentation/qiu.

Radovanovic, A., Koningstein, R., Schneider, I., Chen, B.,´ Duarte, A., Roy, B., Xiao, D., Haridasan, M., Hung, P., Care, N., Talukdar, S., Mullen, E., Smith, K., Cottman, M., and Cirne, W. Carbon-aware computing for datacenters. IEEE Transactions on Power Systems, 38(2): 1270–1280, 2023. doi: 10.1109/TPWRS.2022.3173250.

Saad, H. At board meeting in Murray, TVA approves power agreement for Elon Musk’s supercomputer in Memphis, Tenn. WKMS, November 2024. URL https://ww w.wkms.org/energy/2024-11-07/at-boa rd-meeting-in-murray-tva-approves-pow er-agreement-for-elon-musks-superco mputer-in-memphis-tenn. November 7, 2024; contemporaneous reporting of the utility’s stated approval condition.

Samsi, S., Zhao, D., McDonald, J., Li, B., Michaleas, A., Jones, M., Bergeron, W., Kepner, J., Tiwari, D., and Gadepally, V. From words to watts: Benchmarking the energy costs of large language model inference, 2023. URL https://arxiv.org/abs/2310.03003.

Smith, S. J., Hubbard, A., Newkirk, A., Ganeshalingam, M., Holecek, B., Sartor, D., Mills, M., and Shehabi, A. United states data center energy usage report: 2025 update. Technical Report LBNL-2001758, Lawrence Berkeley National Laboratory, Berkeley, California, 2026. URL https://doi.org/10.71468/P1RP4F.

Snell, C. V., Lee, J., Xu, K., and Kumar, A. Scaling LLM test-time compute optimally can be more effective than scaling parameters for reasoning. In The Thirteenth International Conference on Learning Representations, 2025.

Stojkovic, J., Choukse, E., Zhang, C., Goiri, I., and Torrellas, J. Towards greener llms: Bringing energy-efficiency to the forefront of llm inference, 2024. URL https: //arxiv.org/abs/2403.20306.

Stojkovic, J., Zhang, C., Goiri, <sup>´</sup>I., Choukse, E., Qiu, H., Fonseca, R., Torrellas, J., and Bianchini, R. Tapas: Thermaland power-aware scheduling for llm inference in cloud platforms. In Proceedings of the 30th ACM International Conference on Architectural Support for Programming Languages and Operating Systems, Volume 2, ASPLOS ’25, pp. 1266–1281, New York, NY, USA, 2025a. Association for Computing Machinery. ISBN 9798400710797. doi: 10.1145/3676641.3716025. URL https: //doi.org/10.1145/3676641.3716025.

Stojkovic, J., Zhang, C., Goiri, <sup>´</sup>I., Torrellas, J., and Choukse, E. Dynamollm: Designing llm inference clusters for performance and energy efficiency. In HPCA, March 2025b.

Sukprasert, T., Souza, A., Bashir, N., Irwin, D., and Shenoy, P. On the limitations of carbon-aware temporal and spatial workload shifting in the cloud. In Proceedings ofthe Nineteenth European Conference on Computer Systems, EuroSys ’24, pp. 924–941, New York, NY, USA, 2024. Association for Computing Machinery. ISBN 9798400704376. doi: 10.1145/3627703.3650079. URL https: //doi.org/10.1145/3627703.3650079.

Terrell, M. A new milestone for smart, affordable electricity growth, March 2026. URL https://blog.googl e/innovation-and-ai/infrastructure-a nd-cloud/global-network/demand-respo nse-data-center-milestone/.

Williams, C., Colangelo, P., Coskun, A., Levine, E., Neale, A., Roberts, C., Sengupta, S., Shirolkar, N., Sivaram, V., Soares, S., Tiao, E., Underwood, S., Wilson, D., Sharp, F., Wainwright, L., Petty, H., Wallace, S., and Records, B. Power-flexible AI data centers: A new paradigm for grid-responsive compute. arXiv:2606.25098, 2026. URL https://arxiv.org/abs/2606.25098.

Xing, J., Acun, B., Sundarrajan, A., Brooks, D., Chakkaravarthy, M., Avila, N., Wu, C.-J., and Lee, B. C. Carbon responder: Coordinating demand response for the datacenter fleet, 2023. URL https://arxiv.org/ab s/2311.08589.

Xu, K., Sun, D., Tian, H., Zhang, J., and Chen, K. GREEN: Carbon-efficient resource scheduling for machine learning clusters. In 22nd USENIX Symposium on Networked Systems Design and Implementation (NSDI 25), pp. 999– 1014, Philadelphia, PA, April 2025. USENIX Association. ISBN 978-1-939133-46-5. URL https://www.usen ix.org/conference/nsdi25/presentatio n/xu-kaiqiang.

Yang, A. and Team, Q. Qwen3 technical report, 2025. URL https://arxiv.org/abs/2505.09388.

You, J. How much energy does chatgpt use?, 2025. URL https://epoch.ai/gradient-updates/how -much-energy-does-chatgpt-use. Accessed: 2026-10-29.

You, J., Chung, J.-W., and Chowdhury, M. Zeus: Understanding and optimizing GPU energy consumption of DNN training. In 20th USENIX Symposium on Networked Systems Design and Implementation (NSDI 23), pp. 119–139, Boston, MA, April 2023. USENIX Association. ISBN 978-1-939133-33-5. URL https: //www.usenix.org/conference/nsdi23/p resentation/you.

## A RELATED WORK

Datacenter demand response and workload flexibility. Studies on large flexible loads investigate how power curtailment can accommodate new demand using the existing grid capacity (Norris et al., 2025). Carbon aware scheduling and computing systems utilize temporal flexibility in batched machine learning workloads (Radovanovic et al.´ , 2023; Xu et al., 2025), and geographic workload transfer provides flexibility across distributed locations (Sukprasert et al., 2024). Zeus and Perseus manage the model training energy through GPU power limits and the clock frequency control (You et al., 2023; Chung et al., 2024). Carbon Responder coordinates performance aware power control across heterogeneous workloads (Xing et al., 2023). Field demonstrations further show that AI workload orchestration can provide sustained curtailment (Colangelo et al., 2025; Williams et al., 2026). We study power demand flexibility for online (real-time) inference workloads under varying query length distributions and heterogeneous model deployments.

Power-aware inference. Prior works on inference power control utilize hardware settings, serving configurations, and workload placement for achieving power objectives. POLCA (Patel et al., 2023) characterizes LLM power consumption and exploits remaining headroom for datacenter level oversubscription. DynamoLLM (Stojkovic et al., 2025b) adapts model instance counts, model parallelism and GPU clock frequencies, while µ-Serve jointly optimizes model multiplexing and frequency scaling (Qiu et al., 2024). PALS (Hankendi et al., 2026) combines power-performance models with feedback to coordinate power limits and batching policies, extending to dynamic power targets. TAPAS (Stojkovic et al., 2025a) uses model placements, request routing and serving reconfiguration to manage power and thermal constraints. Our examination of model routing complements these directions and can be composed along with their approaches and objectives.

Measurements of inference energy. Empirical studies characterize how inference energy consumption varies across language models, hardware choices, query workloads and deployment configurations. Samsi et al. (2023) benchmark LLM inference performance and energy across different GPU configurations. AI Energy Score provides standardized task-level energy comparisons (Hugging Face, 2025), and ML.ENERGY Benchmark (Chung et al., 2026) measures inference energy consumption under practical scenarios and supports optimization of serving configurations. Luccioni et al. (2022) assess the carbon footprint for a 176 B BLOOM model’s life-cycle and measure its deployment energy through a live inference endpoint. These studies benchmark energy consumption at the request level, while we utilize model-instance power consumption estimates to predict how query routing across heterogeneous models can provide power demand flexibility for safe and reliable grid operation.

Analytical models and energy estimates. Analytical measurements estimate AI energy consumption across different scales. Scenario-based assessments project electricity demand from assumptions about hardware resources and utilization (de Vries, 2023), whereas other approaches estimate per-query energy consumption from model computation and hardware performance (You, 2025). Oviedo et al. (2026) connect inference energy to the token throughput, node level power, deployment overheads, query lengths and their estimated datacenter level electricity demand. The 1/W law (Chen et al., 2026) examines how context length, KV-cache concurrency, routing topology and the GPU generation affect inference energy efficiency. Alternatively, our work connects the query arrival rate to inference state and power consumption, which we employ for quantifying power demand flexibility for grid compliance.

Serving systems and performance prediction. The performance of inference systems relies on the underlying serving frameworks and their scheduling and memory management policies. PagedAttention in vLLM (Kwon et al., 2023) improves KV-cache memory management through virtual memory and paging, and Sarathi-Serve (Agrawal et al., 2024b) uses chunked prefill computations for reducing prefill and decode interference and improving resource utilization. Splitwise (Patel et al., 2025) disaggregates prefill and decode computation across heterogeneous compute resources to better utilize compute and memory-capable GPUs. Vidur (Agrawal et al., 2024a) combines kernel operator profiling and simulation to predict serving performance and evaluate deployment configurations. Our analysis utilizes a one-time configuration specific calibration step which jointly accounts for scheduling policies, deployment configurations and serving framework policies.

Model routing and power control. FrugalGPT, Hybrid LLM, and RouteLLM study model selection and cascades that balance inference cost and response quality (Chen et al., 2024; Ding et al., 2024; Ong et al., 2025). Taking query– model assignments as a given design input, we study how query routing across model instances affects the total power consumption for a given query arrival rate and query and response length distribution. We choose desired allocations to achieve power objectives based on the feasible routing decisions, while respecting capacity constraints and available headroom at model instances. Our formulation connects quality-aware model routing to datacenter power management and control.

## B ADDITIONAL EXPERIMENTAL RESULTS

## B.1 Experimental Setup

Hardware and workloads. We extend our evaluation across hardware setups to NVIDIA RTX 5090 32 GB, Quadro RTX 8000 48 GB, and Tesla V100 32 GB GPUs. Models are served using bfloat16 on H100 and RTX-5090, and RTX 8000 and V100 utilize float16. Each GPU hosts one vLLM serving instance with chunked prefills, which is further generalized to tensor parallel configuration for Llama 70B on four H100 GPUs. The models used in our evaluations for single GPU serving configurations span 1B– 24B parameters.

The query and response length distribution for our analysis is sampled from fixed length control settings (128/128, 256/256, 512/512, 1024/1024, 2048/2048, and 6144/1024 tokens) and varying length practical distributions (ShareGPT, HumanEval, arXiv summarization). The GPU board power consumption is measured via the Nvidia Management Library.

Table B.1. Key quantities and their notation as presented in the LLM Inference and Power Consumption model, Section 2.
<table><tr><td>Notation</td><td>Description</td></tr><tr><td>λ</td><td>Query arrival rate (requests/s)</td></tr><tr><td> $^ { b }$ </td><td>request concurrency</td></tr><tr><td> $b _ { \mathrm { m a x } }$ </td><td>maximum request concurrency</td></tr><tr><td> ${ \bar { p } } , d$ </td><td>Mean prompt and output lengths (tokens)</td></tr><tr><td>chunk  $\zeta _ { \mathrm { p r e } }$ </td><td>number of prefill tokens in one chunk</td></tr><tr><td>S</td><td>approximate token iterations per request</td></tr><tr><td> $\tau ( b )$ </td><td>Mean time between output tokens (s)</td></tr><tr><td> $\widehat { T } ( b )$ </td><td>Estimated mean request residence time (s)</td></tr><tr><td> $\widehat { \mu } ( \boldsymbol { b } ) , \widehat { C } ( \boldsymbol { b } _ { \mathrm { m a x } } )$ </td><td>Estimated request throughput and serving ca-</td></tr><tr><td> $P ( b ) , P _ { \mathrm { i d l e } }$ </td><td>pacity (requests/s) Model Instance power profile, idle power (W)</td></tr></table>

## B.2 Evaluation and Discussion

Figure B.1 compares the predictions and the measured values across the four hardware platforms. Table B.2 reports the mean absolute percentage errors (MAPE) for the modeled quantities.

Prediction sensitivity near max throughput. Mean request concurrency estimation shows errors exceeding 10% MAPE in some cases. The large errors are majorly observed in long sequence configurations and concurrency when GPUs are near capacity. The fixed point estimate in eq. (4) becomes sensitive as the $\lambda s t _ { 1 } \to 1$ , resulting in magnification of small prediction errors in time-between-tokens estimates. However, the power consumption remains relatively unaffected by prediction errors in number of resident requests as power consumption profiles plateau quickly with increasing concurrency (Figure 6).

Tensor-parallel deployment. We additionally evaluate the Llama-3.1-70B language model served with tensor parallel configuration across four H100 GPUs, and model the inference and power consumption behavior for this model instance. Our evaluation covers nine query length distributions, and we observe a 2.8% error in power consumption estimates and 4.3% error in maximum throughput estimates across the query length distributions. Furthermore, the TBT measurements are accurate to 2.8% of the measured values. These results demonstrate that our analytical model for inference and power consumption generalizes well to large models served across multiple GPUs as a single model instance.

Memory-limited concurrency. KV cache capacity imposes a limit in addition to TBT and latency constraints that determine the operating concurrency cap $b _ { \mathrm { m a x } }$ . We estimate this memory-limited concurrency as

$$
b _ { \operatorname* { m a x } } \leq \left\lfloor \frac { K } { \bar { \ell } } \right\rfloor , \qquad \bar { \ell } \approx \frac { \mathbb { E } [ p d + d ( d + 1 ) / 2 ] } { \mathbb { E } [ d ] } ,
$$

where K denotes the KV-cache capacity in tokens, and $p$ and d denote individual prompt and output lengths of queries. The estimate <sup>¯</sup>ℓ represents the average context length weighted by the number of decode tokens, accounting for longer generations and context growth.

Time to Idle. When new requests are no longer admitted, a model instance continues processing resident requests until all sequence generations are complete. This transition typically takes order of seconds and depends primarily on the remaining output lengths and the number of active requests. Characterizing this interval helps capture the transient behavior of inference systems as they transition from active serving to idle operation.

Power Estimation. For each hardware configuration, language model and query length distributions, we model the power consumption as the sum of idle power and concurrency-dependent dynamic component (eq. (5)). We measure the idle model instance power directly from the model instance with no requests being served. To characterize the dynamic power, we measure average power across varying values of request concurrency (b), and calibrate candidate curves to the measured values and select the one with lowest prediction error on held out values. The resulting curves exhibit sharp increase in power consumption as number of active sequences grows, followed by a gradual plateau as GPU utilization approaches saturation. For H100 GPU evaluations, the measured curves achieve an average prediction error of approximately 1 %.

TBT (ms)  
Request Throughput (reqs/s)  
Power Consumption (W)  
![](images/394a72718aa10836b5a67bd9b611f0d104ae3dca764596a4b8c3ca2d2407e5fe.jpg)

![](images/b955197e8539c43f25aef3659d76e1a0683af49973af220b2650bc5fbd247f4b.jpg)

![](images/5d74fa8690e4e44450391b10cbcc491dc0701fbe37a623dbb61de14e47c02a98.jpg)

![](images/5662ea67bf491425c1ad5eb5fb846faa5a59ebbdb337a0759b4fe478ce2a64dc.jpg)

![](images/585957dc2b3da1d2457abdd330e72d80c88668acbef644a0d7832eeba19e24b5.jpg)  
O H100 80 GB 口 RTX 5090 32 GB RTX 8000 48 GB (fp16) ◇ V100 32 GB (fp16) measured = predicted  
Figure B.1. Single model instance predictions across hardware choices. Across query length distributions, language models and serving framework configurations, our predictions closely track the actual measured values, demonstrating accuracy throughout the examined scenarios.

## B.3 Additional Routing Results

Figures B.2 and B.3 extend the investigation of power demand flexibility through query routing (Figures 7 and 8) to other model and query length distribution choices. We also include analysis of an eight model instance pool, with three small and five large models (3S/5L). As observed in Section 3, the feasible routing decision range contracts as the query arrival rate increases, and power flexibility is generally largest at intermediate arrival rates. Across all the configurations and measurements, the total power consumption predictions achieve 3.3% error, consistent with the low prediction errors for single model instance estimates.

## B.4 Feedback Power Control

Our analytical model selects routing decisions for power control based on the average query arrival rate and query length distribution. To further augment decision making, we implement a PID (Proportional Integral Derivative) feedback controller which adjusts routing decisions based on real-time power consumption measurements.

Controller implementation. The PID controller updates the routing fraction (f) decision using NVML measurements at intervals of 1sec. The normalized control error used for feedback is

$$
e _ { k } = \frac { \widetilde { P } _ { k } - P _ { \mathrm { t a r g e t } , k } } { B _ { \mathrm { r e f } } } ,
$$

where $\scriptstyle \widetilde { P } _ { k }$ is the measured total power and $B _ { \mathrm { r e f } }$ is the measured power range. A positive error implies a higher than desired power consumption, prompting an increase in the fraction of queries routed to the small model instance pool. We use calibrated gains $k _ { p } ~ = ~ 0 . 1 5 4 , ~ k _ { i } ~ = ~ 0 . 1 4 3$ , and $k _ { d } = 0 . 0 7 7$ for the standard PID controller. We further apply a 40W deadband region to limit correction for small errors and fluctuations in power measurement.

Evaluation protocol. We evaluate both the analytical model and the PID controller on Llama-1B/8B (1S/3L) setup under varying query length distributions. We evaluate multiple target power reductions, and measure the stable behavior. We quantify the performance through mean-power error, which is the absolute difference between the target and the average measured power under steady operation. We also measure the standard deviation at steady state and the response time $( \boldsymbol { t } _ { 9 0 } )$ required to achieve 90% of the desired power reduction.

![](images/0024ed3109038e4e227a2e95f20a75894ded67d27a8759707487c34d3d60da2f.jpg)  
Figure B.2. Capacity-feasible routing decisions (left, eq. (10)) and corresponding total-power consumption (right, eq. (11)) across query arrival rates and model–workload configurations. Shaded regions show analytical predictions, bars depict measured feasible ranges. S/L denotes the number of small/large model instances. The routing interval contracts with increasing arrival rates across scenarios.  
Figure B.3. Total power consumption via query routing for the configurations in Figure B.2. Left: measured and predicted total power (eq. (11)) versus the fraction of flexible requests (f) assigned to the small model instance, with varying query arrival rate. Right: measured and predicted power consumption range (eq. (12)) over feasible routing allocations.

Results. Table B.3 describes that the PID feedback improves the mean power value achieved as compared to the analytical model (14 vs 34 W error) across the examined configurations. However, PID controller leads to higher power fluctuations due to continuous real-time correction and susceptibility to power measurement noise as compared to the analytical model which prescribes a fixed choice of operating routing decisions. Furthermore, PID controller also demonstrates a higher median $t _ { 9 0 }$ response time of 12 s as compared to 6.5 s for the analytical model.

Implications. Both the fixed routing decision through analytical model and real-time feedback control are complementary approaches. The fixed decision for power curtailment enables a rapid initial allocation, which can be composed with a slower feedback loop which subsequently corrects residual errors and also adapts to workload variations and evolving conditions under real deployment scenarios.

## B.5 Grid-Service Timescales and Datacenter Results

Grid programs and response requirements. We present popular grid operation requirements and power curtailment policies in Table B.4, and summarize the timescales of operation in Figure B.4. We consider formal grid service agreements, site-specific curtailment conditions and field demonstrations for this discussion.

Query Routing Response Timeframes. Across our evaluations on varying query workloads, the measured $t _ { 9 0 }$ span is approximately 3−18 s. These response times are considerably faster than most minute-hour scale deadlines described in Table B.4, whereas some fast-frequency response agreements require sub-second response latencies. Query routing alone cannot provide response in sub-second latencies, but coupled with other short-term quick response solutions like battery/UPS systems, our approach can sustain all of the presented grid operation policies across timescales.

Absorbing surplus power. Our main analysis and experiments focus on power curtailment applications, but our framework can also guide demand surge to absorb surplus renewable power generation in the grid (Aniti, 2025; National Grid Electricity Transmission plc, 2018). The serving and power consumption models can estimate the available headroom for additional power consumption by utilizing larger model instances subject to capacity and feasibility constraints. Prior work has demonstrated secondary GPU workloads for power smoothing (Choukse et al., 2025), along with deferred batch workload scheduling to attain higher power consumption values.

Table B.2. Single model instance prediction errors across all evaluated models and workloads on hardware setups. Mean Absolute Percentage Error (MAPE) presented for each case.
<table><tr><td>Quantity</td><td>H100</td><td>RTX 5090</td><td>RTX 8000</td><td>V100</td></tr><tr><td>Power consumption at concurrency b,  $P ( b )$ </td><td>0.9</td><td>2.5</td><td>0.7</td><td>2.5</td></tr><tr><td>Power consumption at query arrival rate  $\lambda , { \widehat { P } } ( \lambda )$ </td><td>5.3</td><td>6.9</td><td>13.3</td><td>8.8</td></tr><tr><td>Token iteration time,  $\tau ( b )$ </td><td>3.3</td><td>5.4</td><td>5.0</td><td>6.3</td></tr><tr><td>Request residence time,  ${ \widehat { T } } ( b )$ </td><td>5.5</td><td>6.2</td><td>11.1</td><td>10.5</td></tr><tr><td>Request Throughput,  ${ \widehat { \mu } } ( b )$ </td><td>5.6</td><td>6.7</td><td>13.5</td><td>9.0</td></tr><tr><td>Request Concurrency,  $\hat { b } ( \lambda )$ </td><td>10.6</td><td>9.8</td><td>30.1</td><td>31.0</td></tr><tr><td>Serving capacity</td><td>6.0</td><td>7.2</td><td>10.0</td><td>9.9</td></tr><tr><td>KV-cache concurrency ceiling</td><td>2.8</td><td>1.7</td><td>7.7</td><td>4.6</td></tr></table>

Table B.3. Comparison of open-loop analytical model based power control and PID feedback control across analyzed scenarios. Meanpower error is the absolute difference between average measured power and the target; power standard deviation is the standard deviation of measured power values; response time $t _ { 9 0 }$ denotes time to achieve 90% of the desired power reduction. PID controller demonstrates lower mean-power error, but at the cost of higher fluctuations and response times.
<table><tr><td></td><td colspan="2">Mean-power error (W)</td><td colspan="2">Power std. dev. (W)</td><td colspan="2">Response time t9o (s)</td></tr><tr><td>Controller</td><td>Median</td><td>90th pct.</td><td>Median</td><td>90th pct.</td><td>Median</td><td>90th pct.</td></tr><tr><td>Analytical model, open loop</td><td>34</td><td>131</td><td>27</td><td>136</td><td>6.5</td><td>16.5</td></tr><tr><td>PID feedback</td><td>14</td><td>85</td><td>108</td><td>194</td><td>12.0</td><td>18.6</td></tr></table>

![](images/7ab3402fe01c01fabba5f5ceba53377dd56d526e368bd033b4c6b1b79d31f69b.jpg)

Figure B.4. Response timescales under query routing for power control, with respect to grid-service and curtailment requirements. The initial response time (in seconds) for query routing is fast enough to accommodate majority of the demand flexibility requirements, which can be further coupled with approaches like battery/UPS systems for fast response and sustained power control.

Table B.4. Grid-service requirements and contextual examples accompanying Figure B.4. Holding capability, market commitments, cumulative deployment obligations, and demonstrated event durations are identified separately. Response speed alone does not establish service qualification or the ability to deliver the required magnitude.
<table><tr><td>Program or Guideline</td><td>Response Requirement</td><td>Duration and Objectives</td></tr><tr><td colspan="3">Grid services and demand response</td></tr><tr><td>ERCOT Fast Frequency Response (ERCOT, 2026a)</td><td>60 Hz).</td><td>Full response within 15 cycles (0.25 s at Sustain for 15 min or until recall, whichever occurs first.</td></tr><tr><td>ERCOT RRS, high-set under-frequency relay (ERCOT, 2026a;b)</td><td>Interrupt load within 30 cycles (0.5 s).</td><td>At least 1 h capability for this resource category following restoration instructions.</td></tr><tr><td>ERCOT Regulation (ERCOT, 2026b)</td><td>Respond to control signals within 5 s; deliver capacity within 5 min.</td><td>At least 30 min duration capability with continuous signal tracking.</td></tr><tr><td>PJM Regulation (PJM Interconnection, 2026b;a)</td><td>Respond immediately and reach bid capability within 5 min.</td><td>Half-hour commitments with continuous tracking and performance scoring.</td></tr><tr><td>ERCOT Emergency Response, ERS-10 (ERCOT, 2026b)</td><td>Meet deployment obligations within 10 min of dispatch.</td><td>Sustain until recall. Cumulative obligations are 24 h in December-March contract periods and 12 h in other terms.</td></tr><tr><td>ERCOT Contingency Reserve (ERCOT, 2026a)</td><td>Deploy within 10 min.</td><td>At least 1 h sustained capability under the cited operating guide.</td></tr><tr><td>PJM Synchronized Reserve (PJM Interconnection, 2026a)</td><td>Delivering assigned response within 10 min.</td><td>Maintain the achieved response until the event ends or 30 min from event start, whichever is earlier.</td></tr><tr><td>ERCOT Non-Spinning Reserve (ERCOT, 2026b)</td><td>Deploying reserves within 30 min.</td><td>At least 4 h qualification capability under described rules, load deployment is maintained until recall.</td></tr><tr><td>PJM Load Management (PJM Interconnection, 2026d;c)</td><td>exceptions allow 60 or 120 min.</td><td>Full reduction within 30 min by default, Dispatch-dependent duration until release.</td></tr><tr><td colspan="3">Site-specific curtailment conditions</td></tr><tr><td>PUCT Docket 58881 (Public Utility Commission of Texas, 2026)</td><td>Full curtailment within 30 min of instruction, 10 min if capability is</td><td>ERCOT provides 60 min notice before issuing curtailment instruction when applicable.</td></tr><tr><td>TVA-xAI, reported approval condition (Saad, 2024)</td><td>confirmed beforehand. Ability to curtail on a 30-minute TVA request.</td><td>Re-energization requires authorization. Site-specific condition.</td></tr><tr><td colspan="3">Field demonstrations</td></tr><tr><td>Phoenix, Emerald AI (Colangelo et al., Utility-specified 15-minute ramp. 2025)</td><td></td><td>Demonstrated 25% reduction held for 3 h.</td></tr><tr><td>London, Emerald AI (Williams et al., 2026)</td><td>Demonstrated approximately 30% reduction within 40 s.</td><td>Separate sustained-curtailment tests held 10%–40% reductions for 2–10 h.</td></tr></table>