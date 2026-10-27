# Context Is Not Free: Carbon-Aware LLM Routing for Multi-Turn LLM Web Services across Geo-Distributed Data Centers

Xiaoyang Zhang Nanyang Technological University

Qingang Zhang Nanyang Technological University

Yuejun Yan Alibaba Group

Yonggang Wen Nanyang Technological University

## ABSTRACT

The sustainability of the Web has drawn growing attention, as large language model (LLM)-based web services leave a substantial carbon footprint. These services typically run in geo-distributed data centers whose grid carbon intensity varies across regions and over time, so routing each query to a cleaner region or a smaller model can reduce emissions. A conversation, however, is a stateful web session: each turn reuses the key–value (KV) cache of its history, which resides only where earlier turns were served. Serving a turn on another deployment requires recomputing this history, whose carbon cost grows with the conversation and can outweigh the saving of the move. Existing carbon-aware LLM routers treat a conversation’s queries as independent and ignore this cost. We present CarbonRouter, an online framework that jointly selects the model, GPU type, and region for each query to minimize carbon emissions. CarbonRouter introduces a cache-aware carbon cost model, which charges history recomputation to every candidate deployment us ing cached-token counts from the serving backends, and uses a lightweight gate that keeps a session on its cached deployment when it would likely move back at the next turn and rebuild its history again. We evaluate CarbonRouter in trace-driven experiments on two real-world conversation datasets with five LLMs, three GPU types, and four regions, where it achieves the lowest carbon of all policies. CarbonRouter reduces CO<sub>2</sub> by 16.1% on average relative to FairCarbon, a state-of-the-art carbon-aware router with the same action space, while also improving predicted answer quality and estimated latency. Moreover, the saving rises further to 23.1% for sessions longer than 30 turns.

## KEYWORDS

Sustainability of Web, Carbon-aware Computing, LLM Routing

## 1 INTRODUCTION

Large language model (LLM)-based web services, such as chat assistants, AI search engines, and coding assistants, now answer users through web interfaces and APIs, and every turn of a conversation is a web request that runs inference in a data center [17]. Meanwhile, as platforms increasingly rely on LLMs as the front end of user interaction, the carbon footprint of model inference has drawn growing attention [16, 17]. Like other large Web services, an LLMbased web service typically runs in several geo-distributed data centers and routes each request among them [22]. This routing can cut operational carbon in two ways: serve a query with less energy (on a smaller model that is good enough or on a more eficient GPU), or in a region whose grid is currently cleaner, since grid carbon intensity difers across regions and changes within minutes.

Moving a query is not free. Most trafic of LLM-based web services consists of multi-turn conversations, and a conversation is a stateful web session. A turn is served with the key–value (KV) cache of the conversation history, which resides only on deployments that served earlier turns, until the serving backend evicts it [12, 18]. Switching to a cleaner region, a smaller model, or a more eficient GPU saves carbon, but the new deployment must recompute the history. Histories are often long: 40.3% of queries in the ShareGPT dataset [1] carry at least 4K history tokens, and for them a cache miss more than triples the emissions of a query (Section 2.2). The longer the conversation, the larger the carbon advantage a switch needs, a property we call context-induced carbon stickiness. With 4K history tokens, for example, the new region must be 68% cleaner than the current one (Section 2.3). This trade-of is challenging. The lowest-carbon combination of model, GPU, and region changes from turn to turn as queries need models of diferent sizes and the cleanest region shifts. Whether a switch pays of depends on an uncertain future: how many turns remain, what they need, and how each placement changes the cache state and regional capacity that later queries of all sessions face.

Recent work has begun to address the environmental sustainability of LLM routing, that is, to reduce the carbon footprint of serving each query. Carbon-aware LLM routers choose a model and a region for every query [20, 23, 29], but they treat the queries of a conversation as independent and therefore do not make this trade-of. This leaves a gap: no carbon-aware LLM router for interactive queries prices the carbon of the history recomputation that a deployment would incur. A deployment without the session’s cache thus looks as cheap as if the history were free, and no router can decide online for each incoming query whether leaving the cache pays of in carbon.

We present CarbonRouter<sup>1</sup>, an online routing framework that jointly selects the model, GPU type, and region for each query of a multi-turn conversation to minimize carbon emissions under quality and latency constraints. CarbonRouter combines the predicted quality of each model, the energy and latency profile of each model–GPU pair, the real-time carbon intensity of each region, and the KV-cache state of each deployment in a cache-aware carbon cost model, which charges every candidate deployment for recomputing the part of the history it does not cache. Its routing algorithm, Carbon- and Cache-Aware Routing (CCAR), selects the feasible deployment with the lowest estimated carbon under this cost, so a session leaves its cache only when a cleaner region, a smaller model, or a more eficient GPU outweighs the history it rebuilds. A lightweight learned gate additionally keeps a session on its cached deployment when the proposed switch is likely to be reversed at the next turn. CarbonRouter sits at the request-routing layer of an LLM-based web service, between the web gateway and the serving backends in each region, and replaces only the rout ing policy of this layer. It is therefore easy to deploy: its inputs come from carbon intensity APIs, ofline profiles, and the prefix caches that serving engines already keep, and the models, inference engines, and clients stay unchanged.

![](images/fb83e3e6cd3d2ce8b83ec424ff9dca35c5a3bfb2d883546a4214fd4da8126cd8.jpg)

![](images/84f28a076ffd88c1da910ac7aff7fc5862f3a3de9455e19652d5b81311f142cb.jpg)  
Figure 1: Carbon intensity. (a) 2024 annual mean of 21 countries. (b) Five-minute intensity of our four regions (strip: cleanest region).

We evaluate CarbonRouter in trace-driven simulation on real conversations from ShareChat [42] and ShareGPT [1], with five LLMs from 1B to 70B parameters, three GPU types, and four regions whose real-world five-minute carbon intensity comes from Electricity Maps [10]. The baselines cover each class of prior work: session afinity with and without carbon-aware placement, and two of the latest carbon-aware LLM routers, GAR [29] and FairCarbon [20]. CarbonRouter achieves the lowest carbon of all policies. Relative to FairCarbon, it reduces $\mathrm { C O _ { 2 } }$ by 16.1% on average over the two datasets while increasing mean predicted quality, because it breaks the cache less often: CarbonRouter and FairCarbon both serve about 57% of the ShareChat queries in the cleanest region, but CarbonRouter cuts the share of history tokens it recomputes from 27.7% to 9.4%. Consistent with carbon stickiness, the saving grows with the number of turns, to 23.1% for sessions longer than 30 turns.

Our contributions are as follows:

• We analyze grid carbon intensity and real conversations, and show that KV-cache locality changes carbon-aware routing for multi-turn LLM-based web services.

• We propose CarbonRouter, a carbon- and cache-aware routing framework that jointly selects the model, GPU type, and region for each query, with a routing algorithm, CCAR, that prices history recomputation and avoids switches likely to be reversed.

• We evaluate CarbonRouter with trace-driven experiments on real-world datasets against baselines that include the latest carbon-aware LLM routers, with analyses of conversation length, ablation, and sensitivity.

## 2 CARBON AND KV-CACHE ANALYSIS

This section motivates CarbonRouter with two factors that a carbonaware router for multi-turn LLM-based web services must weigh.

![](images/a5826a6961f788c6ff2fa66cf76ed763631f378510d33a9f3672782f18ade42f.jpg)

![](images/7f0ff397604920af5d89731d211177ee574f1196f4502447b66c52b273a5b7c1.jpg)

![](images/bacff1179d980ca9c4aee16e3c1564a6f5266bdf26fc52706290dec8d7345cfe.jpg)  
Figure 2: Carbon cost of a KV-cache miss on ShareGPT. (a) History length. (b) Extra carbon of a miss (bars: median, error bars: interquartile range) and extra prefill latency (line). (c) Break-even ratio $C I _ { B } / C I _ { A }$ for leaving cached region �, with 128 new and 256 output tokens. $H ^ { \star }$ : break-even history at a ratio of 0.5.

The first is the grid carbon intensity of the regions it can route to. The second is the cost of recomputing a conversation’s history when a query leaves its KV cache, and of how they interact. Our study seeks to answer three questions.

(1) How much does grid carbon intensity difer across regions and change over time, and is the diference large enough to warrant routing LLM queries across regions?

(2) How much does a KV-cache miss add to the carbon emissions of a query in real multi-turn conversations?

(3) Given this cost, when does moving a session to a cleaner deployment reduce emissions?

## 2.1 Carbon Intensity across Regions and Time

To quantify the opportunity, we use two carbon intensity datasets: the 2024 annual mean of 21 countries and areas from Ember [11], and the five-minute intensity of the four regions used in our evaluation, Austria (AT), Great Britain (GB), Belgium (BE), and California (US-CAL-CISO), in 2025 from Electricity Maps [10].

Figure 1(a) shows that annual grid carbon intensity spans a 20.4× range, from 705 gCO e/kWh in India to 35 gCO e/kWh in Switzerland. The intensity also changes within hours. Figure 1(b) shows a 2.7–5.9× range within each of our four regions and a median 2.25× gap between regions at the same time. No region is consistently the cleanest: each region is the lowest 13–32% of the time.

Key Takeaways. Grid carbon intensity difers by an order ofmagnitude across regions and changes within hours, so routing LLM queries to low-carbon regions can reduce emissions. Because the cleanest region changes over time, the router must choose the region online, per query, using real-time carbon intensity.

## 2.2 Carbon Cost of a KV-Cache Miss

A turn of a conversation is served with the KV cache of its history. This cache resides only on deployments that served earlier turns, and only until the backend evicts it; any other deployment must prefill the history again. To measure the carbon cost of this recomputation, we analyze random 2,000 ShareGPT sessions with the Mistral-7B–A100 profile and the default energy ratios of Section 4.1. Because a query’s emissions are its energy times the carbon intensity of its region, the relative increase from a miss on the same deployment does not depend on the carbon intensity.

Conversation histories are long. Figure 2(a) shows that 40.3% of queries carry at least 4K history tokens and 24.4% at least 8K. Rebuilding such a history dominates the carbon cost of a query: a cache miss raises the emissions of a query by a median of 126% for 2K–4K history tokens, 231% for 4K–8K, and 632% beyond 8K, and adds up to 2.5 s of prefill latency (Figure 2(b)).

Key Takeaways. In multi-turn conversations, recomputing the history after a cache miss can emit several times more carbon than the query itself. A carbon-aware router that ignores this cost misjudges the carbon ofevery deployment that does not hold the session’s cache, so it must charge recomputation on every cache miss.

## 2.3 When Does Switching Pay Of?

Consider a session whose cache resides in region � and whose current turn could move to a cleaner region � with the same model and GPU. Staying in � emits $C I _ { A } E ^ { \mathrm { h i t } } ( H )$ , where $E ^ { \mathrm { h i t } } ( H )$ is the energy of the current query, including its new input and output, when its history, the � tokens of earlier queries and responses, is cached. Moving to � emits ${ C I _ { B } E ^ { \mathrm { m i s s } } ( H ) }$ , which adds the recomputation of the history. Moving thus reduces emissions only when

$$
\frac { C I _ { B } } { C I _ { A } } < \tau ( H ) = \frac { E ^ { \mathrm { h i t } } ( H ) } { E ^ { \mathrm { m i s s } } ( H ) } .\tag{1}
$$

The recomputation in $E ^ { \mathrm { m i s s } } ( H )$ grows with the history, whereas a hit reuses the history at a small cost, so � (�) falls as the conversation grows: the longer the conversation, the cleaner � must be to justify a switch. We call this property context-induced carbon stickiness. Under our default energy ratios, � must be 68% cleaner than � with 4K history tokens (Figure 2(c)). Note that this single-turn analysis shows the trend rather than the exact threshold.

Key Takeaways. The cleanest region is not necessarily the best choice for a conversation. The router must weigh each deployment’s carbon intensity against the history recomputation it causes.

## 3 CARBONROUTER DESIGN

## 3.1 Overview

CarbonRouter is a carbon-aware and cache-aware online routing framework for multi-turn LLM-based web services. For each query, CarbonRouter jointly selects a model, a GPU type, and a region to minimize carbon emissions under quality and latency constraints. Figure 3 shows its architecture. When a query arrives, (i) the quality predictor estimates the response quality of each candidate model; (ii) the cost model estimates the energy, latency, and carbon emissions of every available deployment, using the energy and latency profile of each model–GPU pair and the current carbon intensity of each region from a carbon intensity API; (iii) the router selects the feasible deployment with the lowest estimated carbon, including the recomputation of uncached history, and a lightweight gate may keep the session on its cached deployment instead; and (iv) the session-state tracker records each turn’s deployment and completion time and, at dispatch, obtains from the backends the number of history tokens cached on every candidate deployment. The profiles and the carbon intensity API are external inputs: an operator can plug in measured profiles or another carbon intensity provider.

Deployability. CarbonRouter can be deployed without changing the existing serving stack. It is a routing policy at the requestrouting layer, between the web gateway and the serving backends in each region, so the models, inference engines, and clients stay as they are. Its inputs are already available: carbon intensity comes from public APIs [10], energy and latency profiles are measured or estimated ofline, and the cached history of each session is state that serving engines already keep for prefix reuse. The router need not query the serving engines for it on the critical path: the sessionstate tracker already records where and when each turn ran, so the router can track the history cached on each deployment locally from the TTL and from eviction notices that the engines send asynchronously, as cache-aware load balancers do [30, 39]. Because chat APIs commonly resend the full conversation with every request, the router also needs no new client state.

## 3.2 System Model

CarbonRouter serves a set S of multi-turn conversation sessions across geo-distributed data centers, with sets $M , g ,$ and R of models, GPU types, and regions. The �-th query ${ q } _ { s , k }$ of session � arrives at time $t _ { s , k }$ and carries the history of the earlier queries and responses. Turns of a session run sequentially, so the query is dispatched at $d _ { s , k } = \operatorname* { m a x } \{ t _ { s , k } , f _ { s , k - 1 } \}$ , where $f _ { s , k - 1 }$ is the completion time of the preceding turn $( d _ { s , 1 } = t _ { s , 1 } )$ . The router then selects a deployment

$$
\begin{array} { r } { a _ { s , k } = ( m _ { s , k } , h _ { s , k } , r _ { s , k } ) \in \mathcal { A } ( d _ { s , k } ) \subseteq \mathcal { M } \times \mathcal { G } \times \mathcal { R } , } \end{array}\tag{2}
$$

where ${ \mathcal { A } } ( t )$ is the set of deployments that the serving infrastructure makes available at time � after resource and admission control.

KV caches are managed by the existing serving backend of each data center. The cache produced at a turn stays on the deployment that served it until its time-to-live $T ^ { \mathrm { T T L } }$ expires, measured from the turn’s completion [21], or until the backend evicts it in leastrecently-used (LRU) order under memory pressure [18, 43]. The backend reports the cache state to the router.

## 3.3 Quality Prediction and Model Filtering

A quality predictor estimates the expected quality $Q _ { s , k } ( m )$ of each candidate model $m \in { \mathcal { M } }$ on a [0, 1] scale for the current query ${ q _ { s , k } }$ . Any learned router predictor can serve this role. The quality requirement of a query is relative to its best candidate: with $Q _ { s , k } ^ { \mathrm { t a r } } =$ ma $\nu _ { m \in \mathcal { M } } Q _ { s , k } ( m )$ and a required fraction $\beta _ { Q } \in ( 0 ,$ 1], the qualityfeasible model set is

$$
\mathcal { M } _ { s , k } ^ { \boldsymbol { Q } } ( \beta _ { \boldsymbol { Q } } ) = \left\{ m \in \mathcal { M } : \mathcal { Q } _ { s , k } ( m ) \geq \beta _ { \boldsymbol { Q } } \boldsymbol { Q } _ { s , k } ^ { \mathrm { t a r } } \right\} .\tag{3}
$$

The set is never empty, because the best candidate always qualifies, and it adapts to query dificulty: a small model is admitted only when it answers the query nearly as well as the best candidate.

## 3.4 Carbon Cost Model

For candidate model �, let $N _ { s , k } ^ { \mathrm { n e w } } ( m )$ and $N _ { s , k } ^ { \mathrm { h i s t } } ( m )$ be the numbers of new input tokens and history tokens of query $q _ { s , k }$ under �’s tokenizer, and $N _ { s , k } ^ { \mathrm { o u t } } ( m )$ its estimated output length. At dispatch, the backend returns for every candidate deployment $a = ( m , h , r )$ the number $L _ { s , k } ( a ) \leq N _ { s . k } ^ { \mathrm { h i s t } } ( m )$ of history tokens cached there. A cache belongs to the model, GPU, and region that produced it, so $L _ { s , k } ( a )$

![](images/7631538955bd92b695347dab3e732d72c34fce1487ba134a2be945669d2605c3.jpg)  
Figure 3: Overview of CarbonRouter: each query goes to the lowest-carbon model, GPU, and region, given its cached history.

equals the full history on the deployment that served the previous turn while its cache is intact, is smaller after a partial eviction, and is zero where the session has no cache.

The profile of each model–GPU pair (�, ℎ) gives its energy coefficients $e _ { m , h } ^ { \mathrm { i n } } , e _ { m , h } ^ { \mathrm { o u t } } .$ , and $e _ { m , h } ^ { \mathrm { c a c h e } }$ , in kWh per token, for uncached input, output, and cached history tokens. Omitting �, � and the model argument on the right, the estimated energy of candidate $a = ( m , h , r )$ is

$$
E _ { s , k } ( a ) = e _ { m , h } ^ { \mathrm { i n } } \big ( N ^ { \mathrm { n e w } } + N ^ { \mathrm { h i s t } } - L ( a ) \big ) + e _ { m , h } ^ { \mathrm { c a c h e } } L ( a ) + e _ { m , h } ^ { \mathrm { o u t } } N ^ { \mathrm { o u t } } ,\tag{4}
$$

which charges the cached part of the history at the cached-input rate and recomputes the rest. We write $E _ { s , k } ^ { \mathrm { b a s e } } ( a )$ for this energy with the full history cached, $L _ { s , k } ( a ) = N _ { s , k } ^ { \mathrm { h i s t } }$ <sup>t</sup>; the diference $E _ { s , k } ( a ) - E _ { s , k } ^ { \mathrm { b a s e } } ( a )$ is the recomputation penalty of a cache miss.

With the regional carbon intensity $\mathrm { C I } _ { r } ( t )$ in $\mathrm { g C O _ { 2 } e / k W }$ h reported by the carbon intensity API, the estimated emissions are

$$
\widehat { C } _ { s , k } ( \boldsymbol { a } ) = \mathrm { C I } _ { r } ( d _ { s , k } ) E _ { s , k } ( \boldsymbol { a } ) ,\tag{5}
$$

using the intensity at dispatch for the whole request. We write $C _ { s , k } ( a _ { s , k - 1 } , a _ { s , k } )$ for the realized emissions of a served query, computed with the realized output length. We write $\gamma = e _ { m , h } ^ { \mathrm { c a c h e } } / e _ { m , h } ^ { \mathrm { i n } }$ for the cache discount and $\kappa = e _ { m , h } ^ { \mathrm { o u t } } / e _ { m , h } ^ { \mathrm { i n } }$ for the output-to-input ratio.

## 3.5 Latency Model

CarbonRouter estimates the time to first token (TTFT) from the profile’s prefill throughput $\boldsymbol { v } _ { m , h } ^ { \mathrm { p r e } }$ in tokens per second. Prefill covers only the uncached part of the history, so with the notation of Equation (4), the TTFT of candidate $a = ( m , h , r )$ is

$$
\mathrm { T T F T } _ { s , k } ( a ) = \frac { N ^ { \mathrm { n e w } } + N ^ { \mathrm { h i s t } } - L ( a ) } { v _ { m , h } ^ { \mathrm { p r e } } } .\tag{6}
$$

It thus approximates TTFT by prefill execution time alone and excludes network transmission, queueing, scheduling, cache lookup, and first-token decoding, so it is a compute-side estimate rather than the end-to-end latency perceived by users. The model also neglects the variation of throughput with context length and concurrent load.

## 3.6 Constraints and Optimization Goal

Queries from diferent sessions arrive online and interleave. $\mathrm { A t } d _ { s , k }$ the router knows only the information $\mathcal { T } _ { s , k }$ available then, namely the observed contexts, previous actions and cache states, available deployments, and carbon intensities, but not future queries or how many turns remain. The online routing problem is

$$
\begin{array} { r l } { \displaystyle \operatorname* { m i n } _ { \pi } } & { \displaystyle \sum _ { s \in S } \displaystyle \sum _ { k = 1 } ^ { K _ { s } } C _ { s , k } ( a _ { s , k - 1 } , a _ { s , k } ) } \\ { \mathrm { s . t . } \quad a _ { s , k } = \pi ( \mathcal { T } _ { s , k } ) \in \mathcal { R } ( d _ { s , k } ) , \quad m _ { s , k } \in \mathcal { M } _ { s , k } ^ { Q } ( \beta _ { Q } ) , \quad \forall s , k , } \\ & { \textstyle \mathrm { T T F T } _ { s , k } ( a _ { s , k } ) \leq D ^ { \mathrm { T T F T } } , } \end{array}\tag{7}
$$

where $\pi$ is an online routing policy, $K _ { s }$ is the number of turns of session �, and $D ^ { \mathrm { T T F T } }$ bounds the TTFT. We call a deployment that satisfies these constraints feasible. If no deployment meets the TTFT bound but some meet every other constraint, typically because every deployment without the session’s cache would have to recompute a long history, the router serves the query best-efort on the deployments with the shortest estimated TTFT rather than rejecting the turn.

## 3.7 Routing Algorithm

Solving this online routing problem exactly is impractical for three reasons. First, the router must decide without knowing the future: whether keeping a session’s cache pays of depends on how many turns remain, when they arrive, how carbon intensity evolves, and whether the backend keeps the cache, and none of this is known at dispatch. Second, decisions are coupled. Within a session, each action determines where the next turn finds the cache; across sessions, routing one session to a deployment uses capacity and KV memory that other sessions may need and can evict their caches. Third, the router sits on the critical path of every query, so its decision adds directly to the TTFT and must take negligible time; routers for web LLM services keep this overhead below a millisecond [20].

Algorithm 1 CCAR: Carbon- and Cache-Aware Routing   
Input: Request $q _ { s , k } ;$ previous deployment $b _ { s , k - 1 }$ and completion time   
$f _ { s , k - 1 } ;$ current information ${ \cal T } _ { s , k } ,$ including the cache state reported   
by the backend; trained predictor $f _ { \boldsymbol { \theta } } ;$ threshold �   
Output: Routing action $\boldsymbol { a } _ { s , k } ,$ , if a feasible deployment exists   
1: Obtain $L _ { s , k } ( a )$ from the backend; find the feasible deployments (Equa  
tion $( 7 ) ,$ , relaxing the TTFT bound if needed.   
2: if no deployment is feasible then   
3: Reject the query; stop.   
4: end if   
5: for all feasible � do   
6: Compute $\widehat { C } _ { s , k } ( \boldsymbol { a } ) ,$ , including recomputation on a cache miss.   
7: end for   
8: $g _ { s , k } \gets \arg \operatorname* { m i n } _ { a } \widehat { C } _ { s , k } ( a ) ; \ a _ { s , k } \gets g _ { s , k } .$   
9: if $k > 1$ and $g _ { s , k } \neq b _ { s , k - 1 }$ and $b _ { s , k - 1 }$ feasible and $L _ { s , k } \big ( b _ { s , k - 1 } \big ) > 0$   
then   
10: Build features $z _ { s , k } ; \widehat { A } _ { s , k } \gets f _ { \theta } ( z _ { s , k } )$   
11: if $\widehat { A } _ { s , k } > \delta$ then   
12: $a _ { s , k } \gets b _ { s , k - 1 } .$   
13: end if   
14: end if   
15: Execute $\boldsymbol { a } _ { s , k } ;$ record its deployment and completion time.

CarbonRouter therefore routes each query with Carbon- and Cache-Aware Routing (CCAR), a fast per-query algorithm whose core is cache-aware selection: every candidate is charged the recomputation of the history it does not cache, so a session leaves its cache only when a cleaner region or a smaller model outweighs the rebuild. A lightweight learned gate covers the one efect this misses, the next turn.

As shown in Algorithm 1, for each query $q _ { s , k } ,$ , CCAR first obtains the cached history $L _ { s , k } ( a )$ of every available deployment from the backend and keeps the feasible deployments, relaxing the TTFT bound as in Section 3.6 when needed; if none remains, the query is rejected. It then computes the estimated emissions $\widehat { C } _ { s , k } ( a )$ of every feasible deployment, including recomputation, and proposes the one with the lowest estimated carbon, �<sub>�,�</sub> ∈ arg mi $\mathfrak { r } _ { a } \widehat { C } _ { s , k } ( a )$ Finally, the gate decides whether to keep the session on its previous deployment instead, and the query runs on the chosen deployment.

The gate acts only at a switch opportunity, where $g _ { s , k }$ would leave a feasible deployment $b _ { s , k - 1 }$ that still caches the session’s history: if the session left and returned after the backend evicted that cache, the history would be recomputed again. A predictor $f _ { \theta } ,$ an XGBoost regressor [7], estimates the carbon saved by staying over the current and next turn, $\widehat { A } _ { s , k } = f _ { \theta } ( z _ { s , k } )$ , from features available at dispatch, such as the estimated costs of both options and the cache state and memory pressure that the backend reports, and the router stays if $\widehat { A } _ { s , k } > \delta .$ . The predictor is trained on other sessions to predict the realized two-turn diference, and �, including +∞ (no gate), is selected by session-held-out validation.

CCAR runs in time linear in the number ofavailable deployments, and the gate adds at most one predictor call. Like other per-query routers for web services [20], it thus trades global optimality for responsiveness and adds negligible latency to each turn.

## 4 EVALUATION

We evaluate our proposed routing framework CarbonRouter, which combines carbon-aware and cache-aware routing, using real-world workloads and trace-driven simulation with an analytical energy and latency model.

## 4.1 Experimental Setup

Workloads and arrivals. We use two real-world LLM conversation datasets, ShareChat [42] and ShareGPT [1], to evaluate our Carbon-Router framework. The queries follow the multi-turn chat arrival model of UniCache [27]. A query that arrives before its predecessor completes waits for it. All policies share the same arrivals. Models and GPUs. Five instruction-tuned models (Gemma-3-1B, Llama-3.2-3B, Gemma-3-4B, Mistral-7B-v0.2, and Llama-3.1-70B) run on A10, A100-40GB, and H100-SXM in each of four regions, giving 60 deployments. Each model runs on one GPU except Llama-3.1-70B, which uses eight A10, eight A100-40GB, or four H100 GPUs. These counts divide its attention heads, as tensor parallelism requires, and leave 32–147 GB for the KV cache after its 141 GB of 16-bit weights (Appendix A). Quality predictions come from the released R2-Router predictors [34, 40], which score every model once per query from the current query and recent context.

Energy and latency profiles. The energy and latency profiles are a pluggable component of CarbonRouter (Figure 3), so an operator can load measured profiles of its own deployments. In our evaluation, they come from an analytical model built from published data rather than from measurements (Appendix A). Scaling all energy coeficients by a common factor does not change which candidate has the lowest estimated carbon, so routing depends only on the relative energy of deployments and on the ratios � and $\gamma .$ We set � = 6 and $\gamma = 0 .$ 05 from API price ratios [3, 14, 26], following Agent-Decarbonizer [41]. Measurements agree: a cached input token costs 0.045–0.084 of an uncached one [8], and output-to-input energy ratios are about 3–8 under batched serving [5, 38].

Carbon intensity signals. We use five-minute life-cycle carbon intensity from Electricity Maps [10] for AT, GB, BE, and US-CAL-CISO in 2025. Each query uses the latest value at or before its dispatch time.

Constraints and cache. We use $\beta _ { Q } = 0 . 8 5$ , so each query admits the models whose predicted quality is at least 85% of its best candidate’s, a TTFT bound of 5 s, enforced best-efort as in Section 3.6, and each model’s context limit. A KV cache stays on its deployment for at most 30 minutes after its turn completes, unless the emulated backend evicts it earlier (below).

Emulated backend. Without real geo-distributed data centers, we cannot observe what their backends would report, so we emulate availability and caching in the replay, as prior trace-driven studies of geographical load balancing emulate capacity [4, 19, 22, 31]. Each region holds at most $K = c P$ sessions, where � is the peak number of simultaneously active sessions, and a deployment is available only if its region already holds the session or has a free slot. The KV memory of a region holds ��<sup>¯</sup> tokens for the mean query context $\bar { C } ;$ when a new cache does not fit, a deployment evicts idle caches in LRU order, starting with those of sessions that have left [18, 43], and reports the remaining cached history $L _ { s , k } ( a )$ to the router. We use $c = 0 . 5$ , a total capacity of twice the peak demand, and vary � in

Table 1: Overall performance, normalized to Session-Afinity = 1. P95 TTFT is estimated from prefill time only (Equation (6)). $\mathbf { C O } _ { 2 }$ saving: CarbonRouter’s saving over each row, with 95% confidence intervals. <sup>†</sup>Adapted to our setting.
<table><tr><td></td><td colspan="4">ShareChat</td><td colspan="4">ShareGPT</td><td>Average</td></tr><tr><td>Method</td><td> $\mathrm { C O _ { 2 } \downarrow }$ </td><td>Quality↑</td><td>P95 TTFT (s)↓</td><td> $\mathrm { C O } _ { 2 }$  saving (%)</td><td> $\mathrm { C O } _ { 2 } \downarrow$ </td><td>Quality↑</td><td>P95 TTFT (s)↓</td><td> $\mathrm { C O _ { 2 } }$  saving (%)</td><td> $\mathrm { C O _ { 2 } }$  saving (%)</td></tr><tr><td>Session-Affinity</td><td>1.000</td><td>1.000</td><td>0.05</td><td>53.7 [52.3, 55.1]</td><td>1.000</td><td>1.000</td><td>0.04</td><td>54.9 [52.8, 56.9]</td><td>54.3 [53.1, 55.6]</td></tr><tr><td>Carbon-Affinity</td><td>0.586</td><td>0.999</td><td>0.09</td><td>21.0 [20.0, 22.0]</td><td>0.575</td><td>1.000</td><td>0.05</td><td>21.6 [19.7, 23.5]</td><td>21.3 [20.3, 22.4]</td></tr><tr><td>GAR†</td><td>0.630</td><td>0.973</td><td>1.72</td><td>26.5 [24.2, 28.8]</td><td>0.660</td><td>0.976</td><td>0.56</td><td>31.6 [28.5, 34.8]</td><td>29.1 [27.1, 30.9]</td></tr><tr><td>FairCarbon</td><td>0.545</td><td>0.973</td><td>1.67</td><td>15.1 [13.4, 16.9]</td><td>0.544</td><td>0.976</td><td>0.53</td><td>17.0 [14.3, 19.7]</td><td>16.1 [14.4, 17.6]</td></tr><tr><td>CarbonRouter (ours)</td><td>0.463</td><td>0.979</td><td>0.52</td><td></td><td>0.451</td><td>0.979</td><td>0.20</td><td></td><td></td></tr></table>

Section 4.5. Each region’s capacity is split equally across its three GPU types

Baselines. We compare CarbonRouter against four baselines. (i) Session-Afinity keeps each session on one deployment while it remains feasible; it is cache-aware but carbon-agnostic and serves as the reference for all ratios. (ii) Carbon-Afinity chooses the lowestcarbon feasible deployment when a session starts and keeps the session there while its cache remains valid and the deployment remains feasible, so it considers carbon only when it places a session. (iii) $G A R ^ { \dagger }$ adapts GAR [29], a recent carbon-aware LLM router, to choose the lowest-carbon model and GPU within one fixed region; it is carbon-aware but cache-oblivious. (iv) FairCarbon [20], a recent carbon-aware LLM router for web services, chooses the lowest carbon deployment across all regions. We omit its fairness term, which balances model capacity across user groups and requires group labels that our traces lack.

Metrics. We report (i) total operational $\mathrm { C O } _ { 2 } ,$ (ii) the mean predicted quality of the selected models, and (iii) the 95th-percentile TTFT estimated by the latency model of Section 3.5.

## 4.2 Overall Performance

Table 1 shows the overall performance of CarbonRouter compared to other baselines. CarbonRouter achieves the lowest $\mathrm { C O _ { 2 } }$ of all methods on both datasets. Overall, it reduces $\mathrm { C O _ { 2 } }$ by 16.1–54.3% on average relative to the four baselines. More specifically, Carbon-Router reduces $\mathrm { C O _ { 2 } }$ by 54.3% relative to Session-Afinity, by 21.3% relative to Carbon-Afinity, by 29.1% relative to GAR<sup>†</sup>, and by 16.1% relative to the SOTA baseline FairCarbon.

Where the saving comes from. GAR<sup>†</sup> and FairCarbon both ignore the cache, but FairCarbon emits 13.4% less $\mathrm { C O _ { 2 } }$ than GAR<sup>†</sup>, which isolates the benefit of routing across regions. Both move sessions between small and large models from turn to turn and recompute more than a quarter of the history tokens (28.6% and 27.7% on ShareChat). CarbonRouter avoids most of this recomputation. It serves the same share of queries in the cleanest region as FairCarbon (57% on ShareChat and 67% on ShareGPT), so its saving over FairCarbon comes from breaking the cache less often: it switches deployments 5.58 times per ShareChat session instead of 7.14, and the share of history tokens it recomputes falls from 27.7% to 9.4% (from 25.5% to 10.0% on ShareGPT). Carbon-Afinity fails in the opposite way. It serves as many queries in the cleanest region as FairCarbon (57.5% on ShareChat) and recomputes only 2.9% of the history tokens, but a session that moves to Llama-3.1-70B after a hard query stays there, so 87.5% of its ShareChat queries run on the 70B model and it emits more $\mathrm { C O _ { 2 } }$ than FairCarbon on both datasets. CarbonRouter avoids both failures because it leaves the cache only when a cleaner region or a smaller model outweighs the recomputation.

Quality. Session-Afinity has the highest quality, but as a side effect of over-provisioning: it re-binds almost only when the bound model fails the quality requirement, so a session moves to a larger model after a hard query and rarely returns, and 90.0% of ShareChat queries (82.0% on ShareGPT) run on Llama-3.1-70B. Carbon-Afinity reaches almost the same quality for the same reason. Compared with Session-Afinity, CarbonRouter lowers mean predicted quality by only 2.1% while reducing $\mathrm { C O _ { 2 } }$ by 54.3% on average. Carbon-Router’s mean predicted quality also exceeds FairCarbon’s on both datasets, because pricing the cache keeps a session on its cached large model instead of moving it to a small one for a single easy turn.

Latency. Because it recomputes less history, CarbonRouter also has a lower estimated TTFT than the cache-oblivious routers. Its estimated P95 TTFT is 0.52 s on ShareChat, against 1.72 s for $\mathrm { G A R } ^ { \dagger }$ and 1.67 s for FairCarbon (0.20 s against 0.56 s and 0.53 s on ShareGPT), and only 0.18% of its ShareChat queries are served best-efort above the 5 s bound, against 0.76% for FairCarbon. Session-Afinity and Carbon-Afinity have the lowest TTFT because they rarely leave their cache, at 2.16 and 1.27 times the $\mathrm { C O _ { 2 } }$ of CarbonRouter on ShareChat.

## 4.3 Efect of Session Turns

Figure 4 compares FairCarbon and CarbonRouter on ShareChat for sessions longer than 5, 10, 20, and 30 turns, using the same setting and sessions as Table 1. Each group includes all sessions above its threshold. ShareGPT shows the same trends with fewer sessions of many turns.

Savings grow with the number of turns. CarbonRouter’s $\mathrm { C O _ { 2 } }$ saving over FairCarbon rises from 15.3% for sessions longer than 5 turns to 16.6%, 19.8%, and 23.1% for sessions longer than 10, 20, and 30 turns (Figure 4(a)). This is context-induced carbon stickiness (Section 2.3) at work: the longer the history, the more a cachebreaking switch costs.

Mechanism. FairCarbon decides each query independently, so its cache reuse does not improve as sessions have more turns: its hit rate stays between 58.7% and 60.7% (Figure 4(b)). Its switches per session grow from 7.8 to 22.4 (Figure 4(c)), and because each switch rebuilds a longer history in sessions with more turns, its recomputation energy per query grows from 129 to 205 mWh (Figure 4(d)). CarbonRouter prices these rebuilds, and its gate blocks the switches whose next-turn cost outweighs their current saving. For sessions longer than 30 turns, it raises the hit rate from 60.7% to 73.7%, reduces switches per session by 33% (from 22.4 to 14.9), and cuts recomputation energy per query by 70% (from 205 to 61 mWh). CarbonRouter’s recomputation energy per query stays between 54 and 61 mWh across all groups, even though a miss costs more in sessions with more turns: the share of its turns that miss at least part of the cache falls from 32.1% to 26.3%, and 58.8% of these misses in sessions longer than 30 turns occur in the first half of the session, while the history is still short, against 50.0% for FairCarbon.

![](images/a718f2a809ac848470f867f0a7417fd31d0fbe528a4fa460cd187b3478539b81.jpg)  
(a) $\mathrm { C O } _ { 2 }$ saving vs. FairCarbon

![](images/7479e455b1b7d6e6a7dec401f8bb9665e20200257685406852800c24e98c3d0c.jpg)  
(b) KV-cache hit rate

![](images/87d0026c113fc13d6405c0e1305af54d812bd9ecd0b65e9a153f81e6ab6c18fa.jpg)  
(c) Deployment switches

![](images/0fdd0d53e94bc0091178b731a945ec03622c7d234c50b3da8890cb22a6fa9b2d.jpg)  
(d) Recomputation energy  
Figure 4: CarbonRouter versus FairCarbon on ShareChat by session length, with 95% confidence intervals in (a).

Table 2: Ablation on ShareChat, normalized to Session-Afinity = 1. TTFT: estimated P95, as in Table 1. Δ: increase over CarbonRouter, with 95% confidence intervals.
<table><tr><td>Variant</td><td></td><td>CO2↓ Energy↓</td><td>TTFT (s)↓</td><td>∆CO2 (%)</td><td>∆TTFT (%)</td></tr><tr><td>CarbonRouter</td><td>0.463</td><td>0.826</td><td>0.52</td><td>=</td><td></td></tr><tr><td>w/o Gate</td><td>0.479</td><td>0.854</td><td>0.89</td><td>3.5 [2.6, 4.6]</td><td>71.4 [60.3, 81.9]</td></tr><tr><td>w/o Pricing</td><td>0.504</td><td>0.907</td><td>0.74</td><td></td><td>8.9 [7.8, 10.2] 42.4 [35.7, 49.6]</td></tr></table>

CarbonRouter w/o Gate w/o Pricing

![](images/e720671ff73c2a6bcfb28a4edc78e6dc5499053218a817963573b0c156c1d544.jpg)  
(a) CO2 increase

![](images/69943779560b19cbf7b49d8f7879461623ae0c82a656091ed7ef36d480af99e5.jpg)  
(b) P95 TTFT

![](images/cf9ef0d1fc5b44bbd9545659186ad0bf5839bc553a1a79edbfca9e3d5905afa0.jpg)  
(c) Blocked switches  
Figure 5: Ablation on ShareChat by session length. (a) CO increase over the full model, with 95% confidence intervals. (b) 95th-percentile estimated TTFT. (c) Switches blocked by the gate per session.

## 4.4 Ablation Study

To quantify the contribution of cache pricing and the gate, we evaluate two variants of CarbonRouter in the setting of Table 1. w/o Gate disables the gate but keeps cache pricing. w/o Pricing keeps the gate but drops the recomputation term from the carbon estimate, so the proposal ignores the cache, and retrains the gate on this proposal. Table 2 reports the results.

Cache awareness. Both cache mechanisms reduce carbon and latency. Without pricing, $\mathrm { C O _ { 2 } }$ rises by 8.9%. The proposal no longer charges recomputation and proposes many switches whose recomputation exceeds their saving. The gate can only veto them, so it also blocks switches to smaller models that would save carbon, and more queries stay on the large model, which raises energy (Table 2). Removing the gate raises $\mathrm { C O _ { 2 } }$ by only 3.5% but the estimated P95 TTFT by 71.4%, much more than removing pricing does (Table 2), because a switch reversed at the next turn recomputes the history twice.

Session length. The gate becomes increasingly important as conversations grow longer. Removing the gate raises $\mathrm { C O _ { 2 } }$ by 3.5% over all sessions but by 7.3% for sessions longer than 30 turns, twice as much (Figure 5(a)), whereas the efect of removing pricing grows only from 8.9% to 11.7%. In a long session, a reversed switch recom putes a long history twice, so the gate intervenes more often and saves more: it blocks 4.1 switches per session longer than 30 turns, against 0.7 over all sessions (Figure 5(c)), and keeps the estimated P95 TTFT between 0.52 s and 0.61 s in every group, whereas without the gate it grows from 0.89 s to 1.47 s for sessions longer than 30 turns and exceeds that without pricing in every group (Figure 5(b)).

Overall, both components are necessary. Pricing contributes more to the carbon saving and the gate more to the latency: pricing prevents switches whose recomputation exceeds their saving, and the gate prevents switches that the next turn would reverse.

## 4.5 Sensitivity Analysis

Two energy ratios, the cache discount � and the output-to-input ratio �, set how much a cache miss costs (Section 4.1). The regional capacity �, which is emulated, and the quality requirement $\beta _ { Q }$ which is set by the operator, determine which deployments a query can use. We therefore vary these four parameters to test the robustness of our results. Overall, CarbonRouter saves 9.4–18.8% over FairCarbon in every setting we test (Tables 3 and 4), and the saving is largest when a cache miss costs much more than a hit and when the cleanest region has spare capacity.

Table 3: CO<sub>2</sub> saving (%) of CarbonRouter over FairCarbon on ShareChat by � and $\kappa ,$ with 95% confidence intervals.
<table><tr><td>γ</td><td>κ = 5</td><td> $\kappa = 6$ </td><td> $\kappa = 8$ </td></tr><tr><td>0.025</td><td>18.8 [16.7, 21.0]</td><td>16.7 [14.7, 18.8]</td><td> $1 3 . 6 \ [ 1 1 . 9 , 1 5 . 4 ]$ </td></tr><tr><td>0.05</td><td>16.8 [14.9, 18.7]</td><td>15.1 [13.4, 16.9]</td><td> $1 2 . 6 \ [ 1 1 . 1 , 1 4 . 1 ]$ </td></tr><tr><td>0.1</td><td>12.9 [11.2, 14.7] 12.7 [11.1, 14.1]</td><td></td><td> $9 . 7 \ [ 8 . 4 , 1 1 . 0 ]$ </td></tr></table>

Table 4: CO<sub>2</sub> saving (%) of CarbonRouter over FairCarbon on ShareChat by � and $\beta _ { Q } ,$ , with 95% confidence intervals.
<table><tr><td>C</td><td> $\beta _ { Q } = 0 . 8$ </td><td> $\beta _ { Q } = 0 . 8 5$ </td><td> $\beta _ { Q } = 0 . 9$ </td></tr><tr><td>0.5</td><td> $1 7 . 8 \ [ 1 6 . 1 , 1 9 . 4 ]$ </td><td>15.1 [13.4, 16.9]</td><td> $1 3 . 2 \ [ 1 1 . 9 , 1 4 . 6 ]$ </td></tr><tr><td>0.35</td><td>11.4 [9.7, 13.1]</td><td>11.2 [9.4, 12.8]</td><td> $1 1 . 4 \left[ 9 . 7 , 1 3 . 2 \right]$ </td></tr><tr><td>0.25</td><td>10.7 [9.0, 12.5]</td><td>9.4 [7.7, 11.0]</td><td> $1 0 . 5 \ [ 8 . 6 , 1 2 . 3 ]$ </td></tr></table>

Energy ratios. Current LLM APIs price a cached input token at $2 . 5 \mathrm { - }$ 10% of a regular input token and an output token at 5–8 times an input token [3, 14, 26]. We therefore vary $\gamma \in \lbrace 0 . 0 2 5 , 0 . 0 5 , 0 . 1 \rbrace$ and $\kappa \in \{ 5 , 6 , 8 \}$ to cover both ranges. CarbonRouter saves 9.7–18.8% over FairCarbon in all nine settings (Table 3). The saving is largest at $\gamma = 0 . 0 2 5 , \kappa = 5$ and smallest at $\gamma = 0 . 1 , \kappa = 8$ , following the break-even analysis of Section 2.3: a smaller � makes recomputation relatively more expensive, and a smaller � makes the history a larger share of a query’s energy, so both favor keeping a session on its cache.

Regional capacity and quality requirement. We vary $c \in \{ 0 . 5 ,$ 0.35, 0.25}, from a total capacity of twice the peak demand to one equal to it, with the KV memory of each region scaled accordingly, and $\beta _ { Q } \in \{ 0 . 8 , 0 . 8 5 , 0 . 9 \}$ . Tighter capacity reduces the saving (Table 4): at $\beta _ { Q } = 0 . 8 5$ , it falls from 15.1% at � = 0.5 to 9.4% at � = 0.25, because the cleanest region fills earlier and both policies serve more queries in dirtier regions. The quality requirement matters only when capacity is ample: at $c = 0 . 5 ,$ , the saving falls from 17.8% to 13.2% as $\beta _ { Q }$ rises from 0.8 to 0.9, because fewer models qualify and FairCarbon switches less, leaving less recomputation to avoid. At � = 0.35 and 0.25, the saving stays within 11.2–11.4% and 9.4–10.7%.

## 5 RELATED WORK

Carbon-aware computing cuts operational carbon by exploiting the temporal and spatial variation of grid carbon intensity. Temporal shifting delays flexible work to cleaner hours [28, 36] or scales it with the carbon signal [15], and spatial shifting moves load across data centers through geographical load balancing [19, 22], whose benefit simple policies largely capture [31]. These systems treat the workload as fixed and assume that it is deferrable or stateless, so moving it costs little.

Recent work brings these ideas to LLM inference but keeps the model fixed. Bernhard and Yardimci [4] send each query to the currently cleanest region. AgentDecarbonizer [41] migrates agent tasks between grids and charges a recomputation penalty for the lost cache, but it plans each task ofline from its estimated duration, whereas a chat turn must be served on arrival and the length of a conversation is unknown. GreenCache [32] sizes the KV cache of one deployment by carbon but does not move requests. Independently of carbon, cache-aware LLM serving systems schedule a query on the instance that holds its KV cache, within a cluster [30] or across regions [33, 39], to reduce latency.

Table 5: Comparison with existing carbon-aware methods.
<table><tr><td>Method</td><td>Model GPU Region</td><td></td><td></td><td>KV cache</td><td>Interactive workload</td></tr><tr><td>Carbon-aware computing</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Temporal shifting [15, 28, 36]</td><td>x</td><td>x</td><td>x</td><td>x</td><td>x</td></tr><tr><td>Geo. load balancing [19, 22]</td><td>x</td><td>x</td><td>√</td><td>x</td><td>√</td></tr><tr><td>Carbon-aware LLM inference</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Bernhard et al. [4] AgentDecarbonizer [41]</td><td>x x</td><td>x x</td><td>√ √</td><td>x</td><td>√</td></tr><tr><td>Carbon-aware LLM routing</td><td></td><td></td><td></td><td>√</td><td>x</td></tr><tr><td>GAR [29]</td><td>√</td><td>x</td><td>x</td><td>x</td><td>√</td></tr><tr><td>CEDAR [23]</td><td></td><td></td><td>√</td><td></td><td></td></tr><tr><td>FairCarbon [20]</td><td>√</td><td>x</td><td></td><td>x</td><td>x</td></tr><tr><td></td><td>√</td><td>√</td><td>√</td><td>x</td><td>√</td></tr><tr><td>CarbonRouter</td><td>√</td><td>√</td><td>√</td><td>√</td><td>√</td></tr></table>

LLM routing changes the computation itselfby choosing a model for each query, trading answer quality against cost [6, 9, 25, 40]. Carbon-aware LLM routers make this choice by carbon. GAR [29] picks the lowest-carbon model under accuracy and latency constraints, CEDAR [23] routes agentic requests across regions and models, and FairCarbon [20] jointly selects the model, GPU, and region while balancing carbon, quality, and regional fairness. These routers treat the queries ofa conversation as independent and do not price the recomputation caused by leaving a cache. CarbonRouter additionally prices the cache miss of every candidate deployment (Table 5). We use GAR and FairCarbon as baselines (Section 4.1) but not CEDAR, which in our setting, without deferral or monetary cost, makes the same cache-oblivious choice per request as FairCarbon over a smaller action space.

## 6 LIMITATIONS

Our evaluation is trace-driven simulation: energy and prefill latency come from an analytical model built from published data rather than from measurements on our own hardware, and answer quality is predicted rather than judged. CarbonRouter’s decisions depend only on the ratios between energy coeficients, and its saving persists across the ranges we test, but validating the cost model against measured hit and miss energy on production serving engines remains future work, as does modeling network latency, queueing, and cross-region KV-cache transfer.

## 7 CONCLUSION

In this paper, we studied carbon-aware routing of multi-turn conversations in LLM-based web services, where moving a turn to a cleaner region or a smaller model forces its history to be recomputed. We showed that a cache miss can more than triple the emissions of a query with a long history. We developed CarbonRouter, which prices history recomputation for every candidate deployment and uses a lightweight gate to avoid switches likely to be reversed. On two real-world datasets, it reduces CO<sub>2</sub> by 16.1% on average relative to FairCarbon and by 23.1% for sessions longer than 30 turns, while improving predicted quality and estimated latency.

## REFERENCES

[1] anon8231489123. 2023. ShareGPT\_Vicuna\_unfiltered: HTML-Cleaned Raw Conversations. Hugging Face Datasets. https://huggingface.co/datasets/ anon8231489123/ShareGPT\_Vicuna\_unfiltered Unsplit HTML-cleaned raw files, parts 1 and 2; revision 192ab2185289; accessed September 14, 2026.

[2] Oscar Antepara, Zhengji Zhao, Brian Austin, Nan Ding, Leonid Oliker, Nicholas J. Wright, and Samuel Williams. 2025. Benchmark-driven Models for Energy Anal ysis and Attribution of GPU-Accelerated Supercomputing. In Proceedings ofthe International Conference for High Performance Computing, Networking, Storage and Analysis (SC ’25). St Louis, MO, USA. doi:10.1145/3712285.3759815

[3] Anthropic. 2026. Claude API Pricing. https://platform.claude.com/docs/en/aboutclaude/pricing. Accessed: 2026-10-16.

[4] Aleks Bernhard and Arif Baran Yardimci. 2026. Routing LLM Inference to the Cleanest Grid in Real Time. arXiv:2608.06188 https://arxiv.org/abs/2608.06188

[5] Francisco Caravaca, Ángel Cuevas, and Rubén Cuevas. 2025. From Prompts to Power: Measuring the Energy Footprint of LLM Inference. arXiv:2511.05597 [cs.AI] doi:10.48550/arXiv.2511.05597

[6] Lingjiao Chen, Matei Zaharia, and James Zou. 2024. FrugalGPT: How to Use Large Language Models While Reducing Cost and Improving Performance. Transactions on Machine Learning Research (2024). https://openreview.net/forum?id= cSimKw5p6R

[7] Tianqi Chen and Carlos Guestrin. 2016. XGBoost: A Scalable Tree Boosting System. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining (KDD). 785–794. doi:10.1145/2939672. 2939785

[8] Seunghyuk Cho, Sunghyun Choi, Jaeseung Heo, Youngbin Choi, Saemi Moon, MoonJeong Park, and Dongwoo Kim. 2026. Long Live the Librarian! A Persis tent Search Sub-Agent for Energy-Eficient Multi-Agent Software Engineering Systems. arXiv:2605.27787 [cs.MA] doi:10.48550/arXiv.2605.27787 Accepted to EMNLP 2026 (Main Conference).

[9] Dujian Ding, Ankur Mallick, Chi Wang, Robert Sim, Subhabrata Mukherjee, Victor Rühle, Laks V. S. Lakshmanan, and Ahmed Hassan Awadallah. 2024. Hy brid LLM: Cost-Eficient and Quality-Aware Query Routing. In International Conference on Learning Representations (ICLR).

[10] Electricity Maps. 2026. Electricity Maps API: Carbon Intensity Data. Web API. https://www.electricitymaps.com Five-minute life-cycle carbon intensity for AT, GB, BE, and US-CAL-CISO; data retrieved February 10, 2026.

[11] Ember and Our World in Data. 2026. Lifecycle Carbon Intensity of Electricity Generation. Our World in Data. https://ourworldindata.org/grapher/carbonintensity-electricity Ember data processed by Our World in Data; data version 2026-06-30; indicator 1295504; downloaded September 14, 2026.

[12] Bin Gao, Zhuomin He, Puru Sharma, Qingxuan Kang, Djordje Jevdjic, Junbo Deng, Xingkun Yang, Zhou Yu, and Pengfei Zuo. 2024. Cost-Eficient Large Language Model Serving for Multi-turn Conversations with CachedAttention. In Proceedings of the 2024 USENIX Annual Technical Conference (USENIX ATC).

[13] Alicia Golden, Carole-Jean Wu, Gu-Yeon Wei, and David Brooks. 2026. The xPU-athalon: Quantifying the Competition of AI Acceleration. arXiv:2604.10852 [cs.AR] doi:10.48550/arXiv.2604.10852 Accepted to IS-PASS 2026.

[14] Google. 2026. Gemini API Pricing. https://ai.google.dev/gemini-api/docs/pricing. Accessed: 2026-10-16.

[15] Walid A. Hanafy, Qianlin Liang, Noman Bashir, David Irwin, and Prashant Shenoy. 2023. CarbonScaler: Leveraging Cloud Workload Elasticity for Optimizing Carbon-Eficiency. Proceedings ofthe ACM on Measurement and Analysis of Computing Systems 7, 3 (2023), 1–28. doi:10.1145/3626788

[16] International Energy Agency. 2025. Energy and AI. Technical Report. Interna tional Energy Agency, Paris. https://www.iea.org/reports/energy-and-ai

[17] Nidhal Jegham, Marwan Abdelatti, Chan Young Koh, Lassad Elmoubarki, and Abdeltawab Hendawi. 2025. How Hungry is AI? Benchmarking Energy, Water, and Carbon Footprint of LLM Inference. arXiv:2505.09598 [cs.CY] doi:10.48550 arXiv.2505.09598

[18] Woosuk Kwon, Zhuohan Li, Siyuan Zhuang, Ying Sheng, Lianmin Zheng, Cody Hao Yu, Joseph E. Gonzalez, Hao Zhang, and Ion Stoica. 2023. Eficient Memory Management for Large Language Model Serving with PagedAttention. In Proceedings ofthe 29th Symposium on Operating Systems Principles (SOSP).

[19] Pengfei Li, Jianyi Yang, Adam Wierman, and Shaolei Ren. 2024. Towards Environmentally Equitable AI via Geographical Load Balancing. In Proceedings of the 15th ACM International Conference on Future and Sustainable Energy Systems (e-Energy).

[20] Tingting Li, Ziming Zhao, Zhaoxuan Li, Xiaofei Yue, and Jiongchi Yu. 2026. Fair and Carbon-Aware LLM Routing for Web Services. In Proceedings ofthe ACM Web Conference 2026 (WWW ’26). Dubai, United Arab Emirates.

[21] Yuhan Liu, Yuyang Huang, Jiayi Yao, Shaoting Feng, Zhuohan Gu, Kuntai Du, Hanchen Li, Yihua Cheng, Junchen Jiang, Shan Lu, et al. 2026. DroidSpeak: KV Cache Sharing Across Fine-tuned Model Variants. In 23rd USENIX Symposium on Networked Systems Design and Implementation (NSDI 26). 319–338.

[22] Zhenhua Liu, Minghong Lin, Adam Wierman, Steven H. Low, and Lachlan L. H. Andrew. 2011. Greening Geographical Load Balancing. In Proceedings ofthe ACM SIGMETRICS International Conference on Measurement and Modeling ofComputer Systems. doi:10.1145/1993744.1993767

[23] Amit More, Tarique Anwar, and Poonam Yadav. 2026. CEDAR: Carbon Eficient Dynamic Allocation and Routing for Agentic LLM Inference. In Proceedings ofthe 2nd International Workshop on Systems and Methods for Sustainable Large-Scale AI (GreenSys ’26). ACM, 25–30. doi:10.1145/3802973.3804457

[24] Chenxu Niu, Wei Zhang, Jie Li, Yongjian Zhao, Tongyang Wang, Xi Wang, and Yong Chen. 2026. TokenPowerBench: Benchmarking the Power Consumption of LLM Inference. In Proceedings ofthe AAAI Conference on Artificial Intelligence (AAAI-26), Vol. 40. 32582–32590. doi:10.1609/aaai.v40i38.40535

[25] Isaac Ong, Amjad Almahairi, Vincent Wu, Wei-Lin Chiang, Tianhao Wu, Joseph E. Gonzalez, M. Waleed Kadous, and Ion Stoica. 2025. RouteLLM: Learning to Route LLMs from Preference Data. In International Conference on Learning Representations (ICLR).

[26] OpenAI. 2026. API Pricing. https://developers.openai.com/api/docs/pricing. Accessed: 2026-10-16

[27] Bei Ouyang, Yifan Qiao, and Jiarong Xing. 2026. UniCache: Unifying Prefix Cache Eviction for Heterogeneous LLM Serving Workloads. Proceedings ofthe ACM on Measurement and Analysis ofComputing Systems 10, 2, Article 54 (2026), 27 pages. doi:10.1145/3805652

[28] Ana Radovanović, Ross Koningstein, Ian Schneider, Bokan Chen, Alexandre Duarte, Binz Roy, Diyue Xiao, Maya Haridasan, Patrick Hung, Nick Care, Saurav Talukdar, Eric Mullen, Kendal Smith, MariEllen Cottman, and Walfredo Cirne. 2023. Carbon-Aware Computing for Datacenters. IEEE Transactions on Power Systems 38, 2 (2023), 1270–1280. arXiv:2106.11750 doi:10.1109/TPWRS.2022. 3173250

[29] Disha Sheshanarayana, Rajat Subhra Pal, Manjira Sinha, and Tirthankar Dasgupta. 2026. GAR: Carbon-Aware Routing for LLM Inference via Constrained Optimization. arXiv:2605.11603 https://arxiv.org/abs/2605.11603

[30] Vikranth Srivatsa, Zijian He, Reyna Abhyankar, Dongming Li, and Yiying Zhang. 2025. Preble: Eficient Distributed Prompt Scheduling for LLM Serving. In International Conference on Learning Representations (ICLR)

[31] Thanathorn Sukprasert, Abel Souza, Noman Bashir, David Irwin, and Prashant Shenoy. 2024. On the Limitations of Carbon-Aware Temporal and Spatial Workload Shifting in the Cloud. In Proceedings of the 19th European Conference on Computer Systems (EuroSys).

[32] Yuyang Tian, Desen Sun, Yi Ding, and Sihang Liu. 2026. Cache Your Prompt When It’s Green: Carbon-Aware Caching for Large Language Model Serving. Proceedings of the ACM on Measurement and Analysis of Computing Systems 10, 1 (2026). doi:10.1145/3788087

[33] Alessio Ricci Toniolo, Abinaya Dinesh, and Rome Thorstenson. 2026. GORGO: Maximizing KV-Cache Reuse While Minimizing Network Latency in Cross-Region LLM Load Balancing. arXiv:2602.11688v1 https://arxiv.org/abs/2602. 11688v1

[34] UCF-ML-Research. 2026. R2-Router: Released Predictors and Model Configu ration. GitHub repository. https://github.com/UCF-ML-Research/R2-Router README and r2\_router/config.json; accessed September 14, 2026.

[35] Haonan Wang, Xuxin Xiao, Mingyu Yan, Zhuoyuan Zhu, Dengke Han, Duo Wang, Wenming Li, Xiaochun Ye, Cunchen Hu, Hongyang Chen, and Guangyu Sun. 2025. A Systematic Characterization of LLM Inference on GPUs. arXiv:2512.01644 [cs.AR] doi:10.48550/arXiv.2512.01644

[36] Philipp Wiesner, Ilja Behnke, Dominik Scheinert, Kordian Gontarska, and Lauritz Thamsen. 2021. Let’s Wait Awhile: How Temporal Workload Shifting Can Reduce Carbon Emissions in the Cloud. In Proceedings ofthe 22nd International Middleware Conference (Middleware). doi:10.1145/3464298.3493399

[37] Grant Wilkins, Fiodar Kazhamiaka, and Ram Rajagopal. 2026. From Servers to Sites: Compositional Power Trace Generation of LLM Inference for Infrastructure Planning. arXiv:2603.18383 [cs.DC] doi:10.48550/arXiv.2603.18383

[38] Vince Wu, Jiaqi Han, Boris Ruf, Vincent Grari, and Marcin Detyniecki. 2026. Prefill, Decode, and Context: A Measured Energy Law for Agentic LLM Serving. In NeurIPS 2026 Workshop on Resource-Aware Agentic AI. https://openreview.net/ forum?id=IvZ8KlwQfb

[39] Tian Xia, Ziming Mao, Jamison Kerney, Ethan J. Jackson, Zhifei Li, Jiarong Xing, Scott Shenker, and Ion Stoica. 2026. SkyWalker: A Locality-Aware Cross-Region Load Balancer for LLM Inference. In Proceedings of the 21st European Conference on Computer Systems (EuroSys).

[40] Jiaqi Xue, Qian Lou, Jiarong Xing, and Heng Huang. 2026. R2-Router: A New Paradigm for LLM Routing with Reasoning. arXiv:2602.02823 [cs.CL] doi:10. 48550/arXiv.2602.02823 Version 2.

[41] Leyi Yan, Shuangning Li, and Sihang Liu. 2026. AgentDecarbonizer: Carbon-Aware Execution for AI Agents. arXiv:2608.20566 [cs.LG] doi:10.48550/arXiv. 2608.20566

[42] Yueru Yan, Tuc Nguyen, Bo Su, Melissa Liefers, and Thai Le. 2025. ShareChat: A Dataset of Chatbot Conversations in the Wild. arXiv:2512.17843 [cs.CL] doi:10. 48550/arXiv.2512.17843

[43] Lianmin Zheng, Liangsheng Yin, Zhiqiang Xie, Chuyue Sun, Jef Huang, Cody Hao Yu, Shiyi Cao, Christos Kozyrakis, Ion Stoica, Joseph E. Gonzalez, Clark Barrett, and Ying Sheng. 2024. SGLang: Eficient Execution of Structured Language Model Programs. In Advances in Neural Information Processing Systems (NeurIPS).

## A ENERGY AND LATENCY PROFILES

Energy and prefill. An uncached input token of a model with $P _ { m }$ parameters costs $2 P _ { m }$ FLOPs. With prefill utilization (MFU) $\eta _ { m , h } .$ per-GPU peak FP16 rate $F _ { h }$ , prefill power $f \mathrm { T D P } _ { h }$ per GPU, and $G _ { m , h }$ GPUs, its energy and the prefill throughput are

$$
\begin{array} { r l } & { e _ { m , h } ^ { \mathrm { i n } } = \frac { 2 P _ { m } f \mathrm { T D P } _ { h } } { \eta _ { m , h } F _ { h } } , \qquad v _ { m , h } ^ { \mathrm { p r e } } = \frac { \eta _ { m , h } G _ { m , h } F _ { h } } { 2 P _ { m } } , } \\ & { \eta _ { m , h } = \eta _ { \mathrm { r e f } } \left( \frac { P _ { m } } { 7 0 \mathrm { B } } \right) ^ { 1 - \alpha } \eta _ { h } . } \end{array}\tag{8}
$$

$F _ { h }$ is 125, 312, and 989.5 TFLOP/s and $\mathrm { T D P } _ { h }$ is 150, 400, and 700 W for A10, A100-40GB, and H100-SXM. We set $\mathnormal { f } = 0 . 8 5$ , within the measured 0.75–1.0 of TDP [13, 37], and $\eta _ { \mathrm { r e f } } = 0 . 5$ for Llama-3.1-70B on eight A100s. The exponent $\alpha = 0 . 8$ lies between the size scaling of energy per token measured on H100 (about 0.5) [24] and on A100 (about 0.9) [35]. The GPU factors $\eta _ { h }$ (0.653, 1, and 0.69) keep the relative energies of 1.433, 1, and 0.8 on A10, A100, and H100 of the 70B reference profile of Li et al. [20], and the H100 value is close to the measured H100-to-A100 energy per FLOP of 0.74 [2]. Output and cached input tokens cost $e ^ { \mathrm { o u t } } = \kappa e ^ { \mathrm { i n } }$ and $e ^ { \mathrm { c a c h e } } = \gamma e ^ { \mathrm { i n } }$

Reference energy. For Llama-3.1-70B on eight A100s, Equation (8) gives $e _ { \mathrm { r e f } } ^ { \mathrm { i n } } = 8 . 4 8 \times 1 0 ^ { - 8 } { \mathrm { k W h } }$ (0.305 J) per token, so $e _ { \mathrm { r e f } } ^ { \mathrm { o u t } } = 5 . 0 9 \times 1 0 ^ { - 7 }$ and $e _ { \mathrm { r e f } } ^ { \mathrm { c a c h e } } = 4 . 2 4 \times 1 0 ^ { - 9 }$ kWh per token. Table 6 lists the energy of every deployment relative to this reference, $\rho _ { m , h } = e _ { m , h } ^ { \mathrm { i n } } / e _ { \mathrm { r e f } } ^ { \mathrm { i n } }$ . Because � and $\eta _ { \mathrm { r e f } }$ cancel in $\rho _ { m , h ; \ l }$ , they do not afect which deployment has the lowest estimated carbon.

Decoding. A batch-one decode step is memory-bound and reads the 16-bit weights once. With per-GPU bandwidth $B _ { h }$ of 600, 1,555, and 3,350 GB/s, the decoding throughput is $v _ { m , h } ^ { \mathrm { d e c } } = G _ { m , h } B _ { h } / ( 2 P _ { m } )$ With $\kappa \ = \ 6 ,$ the energy of an output token corresponds to an efective decode batch of about 5–17 sequences. A memory-bound step changes little up to this batch size, so we use the batch-one decoding time.

Deployments. Each model runs on one GPU except Llama-3.1- 70B, which uses eight A10, eight A100-40GB, or four H100 GPUs. These counts divide its 64 query heads and 8 KV heads, as tensor parallelism requires. With 90% of GPU memory usable, they leave 32–147 GB for the KV cache after its 141 GB of 16-bit weights, whereas four A100-40GB or two H100 GPUs would leave about 3 GB.

Table 6: Analytical model–GPU profiles. Energy is relative to Llama-3.1-70B on A100, and throughputs are in tokens/s.
<table><tr><td rowspan="2">Model</td><td rowspan="2">GPU</td><td rowspan="2">Relative energy</td><td rowspan="2">Prefill</td><td rowspan="2">MFU</td><td rowspan="2">Decode</td></tr><tr><td></td></tr><tr><td rowspan="3">Gemma-3-1B</td><td>A10</td><td>0.048</td><td>8,725</td><td>0.14</td><td>300.0</td></tr><tr><td>A100</td><td>0.033</td><td>33,348</td><td>0.21</td><td>777.5</td></tr><tr><td>H100</td><td>0.027</td><td>72,950</td><td>0.15</td><td>1,675.0</td></tr><tr><td rowspan="3">Llama-3.2-3B</td><td>A10</td><td>0.115</td><td>3,623</td><td>0.17</td><td>100.0</td></tr><tr><td>A100</td><td>0.080</td><td>13,848</td><td>0.27</td><td>259.2</td></tr><tr><td>H100</td><td>0.064</td><td>30,292</td><td>0.18</td><td>558.3</td></tr><tr><td rowspan="3">Gemma-3-4B</td><td>A10</td><td>0.145</td><td>2,878</td><td>0.18</td><td>75.0</td></tr><tr><td>A100</td><td>0.101</td><td>11,001</td><td>0.28</td><td>194.4</td></tr><tr><td>H100</td><td>0.081</td><td>24,064</td><td>0.19</td><td>418.8</td></tr><tr><td rowspan="3">Mistral-7B-v0.2</td><td>A10</td><td>0.227</td><td>1,839</td><td>0.21</td><td>42.9</td></tr><tr><td>A100</td><td>0.158</td><td>7,031</td><td>0.32</td><td>111.1</td></tr><tr><td>H100</td><td>0.127</td><td>15,380</td><td>0.22</td><td>239.3</td></tr><tr><td rowspan="3">Llama-3.1-70B</td><td>8×A10</td><td>1.433</td><td>2,332</td><td>0.33</td><td>34.3</td></tr><tr><td>8×A100</td><td>1.000</td><td>8,914</td><td>0.50</td><td>88.9</td></tr><tr><td>4×H100</td><td>0.800</td><td>9,750</td><td>0.34</td><td>95.7</td></tr></table>