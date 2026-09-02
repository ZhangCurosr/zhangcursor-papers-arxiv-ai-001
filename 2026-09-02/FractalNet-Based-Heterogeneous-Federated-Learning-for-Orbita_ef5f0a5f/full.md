# FractalNet-Based Heterogeneous Federated Learning for Orbital Edge Intelligence in Satellite Mega-Constellations: A Wildfire Case Study

Sai Puppala   
School of Computing   
Southern Illinois University   
Carbondale, IL 62901, USA   
Email: sai.puppala@siu.edu

Koushik Sinha<sup>∗</sup> School of Computing Southern Illinois University Carbondale, IL 62901, USA Email: koushik.sinha@siu.edu

## Abstract

Satellite mega-constellations are emerging as large-scale sensing, communication, and computation fabrics, yet their learning architectures remain largely inherited from terrestrial federated learning and ground-centric mission operations— illsuited to satellites that differ by orders of magnitude in Size, Weight, Power, and Cost (SWAP-C), radiation tolerance, link availability, and propagation delay. We propose a heterogeneous federated learning method based on the FractalNet architecture for orbital edge intelligence. We formalize contact-window-constrained, depth-heterogeneous federated optimization and introduce a distributed path scheduler that assigns model depth as a function of SWAP-C constraints, predicted inter-satellite contacts, and training statistics. To reduce message overhead and energy consumption, each tier pools updates periodically rather than at every contact opportunity, and a three-tier agentic control plane governs in-space scheduling, anomaly escalation, and policy-governed autonomy. As a case study, we apply the framework to wildfire detection, where each orbital shell naturally learns a different semantic level of situational awareness: pixel-scale thermal anomalies at low Earth orbit (LEO), regional fire-front dynamics at medium Earth orbit (MEO), and larger-scale risk propagation at geostationary or high Earth orbit (GEO/HEO). Experiments on simulated mega-constellations validate the approach across convergence, communication efficiency, energy adaptation, scheduled-pooling savings, robustness, and latency.

## 1 Introduction

Mega-constellations shift the unit of space-system design from an individually controlled spacecraft to a massive, time-varying distributed system. Thousands of satellites observe, route, infer, and coordinate under constraints that differ sharply from terrestrial cloud and mobile-edge settings: inter-satellite links (ISLs) appear and disappear according to orbital mechanics; radiation induces nonadversarial but security-relevant compute and memory faults; downlink opportunities are intermittent; and onboard compute spans from radiation-hardened microprocessors to accelerator-class edge devices.

Federated learning (FL) appears natural for this setting because raw telemetry, imagery, and RF observations remain local while model updates are exchanged. However, standard FL carries an implicit homogeneity assumption: every client trains the same model architecture, possibly at different speeds. In satellite constellations this assumption is structurally wrong. A weak LEO satellite may only support a shallow feature extractor, whereas a high-compute GEO/HEO node may support full sequence modeling and robust aggregation. Designing for the weakest node wastes high-compute satellites; designing for the strongest node excludes the majority of the constellation.

This paper proposes FN-HFL, a co-designed architecture in which the topology is part of the learning algorithm. Rather than layering heterogeneous FL on top of an arbitrary communication substrate, FN-HFL maps FractalNet-style multi-path structure to model-depth allocation. Short paths correspond to shallow-prefix training, medium paths to partial-depth training, and deep paths to full-model training. A scheduler assigns satellite roles and path depths using current SWaP-C state, forecast contact windows, model-layer statistics, and trust signals. Agents onboard satellites operate at three timescales: micro-agents for local reflexes, meso-agents for regional aggregation, and macro-agents for constellation-level planning.

To illustrate how naturally this architecture maps onto real operational scenarios, we develop a wildfire detection use case in Section 2 that shows how each tier of the constellation learns semantically distinct information aligned with its orbit, revisit rate, and computational capability. This grounding motivates the formal architecture that follows.

## Contributions. We make five contributions.

1. We introduce FN-HFL, a topology-native heterogeneous FL architecture for satellite megaconstellations that ties FractalNet path length to model depth and aggregation responsibility.

2. We formalize depth-heterogeneous federated optimization over scheduled orbital contact graphs, including layer-wise participation masks, staleness bounds, and SWaP-C constraints.

3. We present a distributed scheduler that selects shallow, medium, and deep training paths and defines how LEO, MEO, and GEO/HEO nodes exchange updates without downlinking raw data.

4. We define a depth-stratified aggregation and model-consistency protocol that synchronizes different model prefixes across heterogeneous satellites.

5. We introduce scheduled pooling, a tier-specific update-aggregation cadence (30 minutes at LEO, 6 hours at MEO, 24 hours at GEO/HEO) that decouples transmission frequency from raw contactwindow availability, substantially reducing per-node energy expenditure relative to opportunistic, every-contact transmission.

6. We provide a NeurIPS-style experimental protocol with simulator assumptions, baselines, metrics, and ablations, anchored by a concrete wildfire detection scenario that validates the depth-hierarchy design.

## 2 Motivating Scenario: Wildfire Detection Across a Mega-Constellation

To make the FN-HFL architecture tangible before introducing formalism, we trace a wildfire breaking out in a remote forested region. This scenario is not merely illustrative. It exposes exactly the structural properties—multi-scale sensing, information aggregation hierarchy, compute heterogeneity, and intermittent connectivity—that motivated the design. Crucially, the information that each tier of the constellation can usefully contribute maps directly onto the depth of model that tier can afford to train.

## 2.1 Shell 3 — LEO sensing nodes: raw signal detection

The first responders are the large population of low-altitude LEO nodes, passing over any given region every few minutes at typical altitudes between 500 and 600 km. These are the most constrained devices in the constellation: limited battery reserves, modest processors, and narrow contact windows measured in minutes per pass. But they carry multispectral and thermal sensors, and their low altitude gives them the spatial resolution needed to detect early-stage fire signatures.

Specifically, LEO nodes are positioned to observe: anomalous temperature clusters in the thermal infrared band (LWIR), suggesting surface heating above ambient; localized shifts in near-infrared (NIR) reflectance consistent with vegetation beginning to combust; aerosol density spikes in shortwave infrared (SWIR) channels indicating early smoke column formation; and rapid frame-to-frame pixel changes in registered image sequences consistent with active combustion rather than agricultural activity or industrial heat sources.

Each LEO node trains only the shallow prefix of the shared model—the early convolutional or patch-embedding layers that learn to recognize these low-level, pixel-scale features from raw sensor streams. No single LEO node sees the whole fire. Each sees a tile, a moment, a local anomaly. Their gradient updates are cheap to compute, covering only the early model layers, and are relayed during brief contact windows to the nearest MEO regional aggregator. The shallow updates from dozens of LEO passes, each observing a different tile of the affected region, collectively train a lower-layer feature extractor that generalizes across fire intensities, vegetation types, smoke opacity levels, and sensor viewing angles—precisely because the constellation’s broad spatial coverage provides the diversity that no single ground sensor or individual satellite pass could achieve.

## 2.2 Shell 2 — MEO regional nodes: contextual pattern recognition

MEO satellites orbit higher and move more slowly relative to the ground, giving them longer contact windows with LEO nodes below and global nodes above, and a wider instantaneous field of view. They receive shallow updates from dozens of LEO passes and aggregate them into a regional picture. But they also train their own medium-depth layers on this aggregated signal—layers that learn to recognize spatial propagation patterns and regional context that no individual LEO pass can see.

The medium layers trained at MEO learn: the characteristic elongated shape of a wind-driven fire front, distinguishable from point-source industrial fires; the correlation between temperature anomaly clusters and terrain features such as ridge lines, drainage channels, and fuel-break boundaries that shape fire propagation; the temporal cadence signature of a spreading fire versus a contained or dying one, visible only when comparing updates from multiple LEO passes over the same area; and the statistical relationship between early SWIR aerosol signals and subsequent thermal anomaly patterns, enabling forward prediction of fire spread direction.

A single MEO node might coordinate updates from LEO satellites spanning several hundred kilometers of fire perimeter, synthesizing what looked like scattered thermal noise at the LEO level into a coherent, moving boundary. This is precisely where the medium-depth model earns its architectural value: the early layers, trained by LEO nodes, learn what fire looks like locally; the middle layers, trained and aggregated at MEO, learn how fires behave regionally. The MEO node does not need the raw pixels. It receives compressed gradient updates from LEO nodes, aggregates them robustly across the shallow layers, extends the model forward through the medium layers using its own regional training data, and forwards a richer update upward. The communication savings are substantial: the MEO aggregator transmits a single regional summary rather than relaying dozens of raw LEO update packets to the global tier.

## 2.3 Shell 1 — GEO/HEO global nodes: multi-region situational awareness

High-compute global nodes see across entire continents. They receive regional summaries from multiple MEO aggregators simultaneously—one covering a Pacific coast range, another tracking an interior mountain fire complex, a third detecting a secondary ignition hundreds of kilometers from the original event driven by long-range ember transport. The deep layers of the model, trained and maintained exclusively at this tier, learn cross-regional and multi-event patterns that no lower tier could represent.

Specifically, the deep layers learn: the statistical relationship between current large-scale atmospheric patterns (wind fields, humidity gradients, pressure systems) and fire propagation severity across multiple simultaneous events; historical correlations between early fire signatures in a given region and eventual burned area, enabling rapid severity classification that informs resource pre-positioning; the characteristic temporal signature of ember-driven spotting events, which create secondary ignitions kilometers ahead of the main front and which only become visible when comparing observations across widely separated MEO regions; and multi-year climate-driven risk signatures that allow the model to flag regions currently exhibiting pre-fire conditions even before any thermal anomaly is detected.

These are patterns that no LEO or MEO node could learn in isolation. They require integrating signals that are spatially separated by hundreds to thousands of kilometers and temporally separated by hours to days. The GEO/HEO tier provides both the computational resources to train these deep representations and the persistent wide-area view that makes the training signal meaningful.

## 2.4 Why this maps cleanly onto FN-HFL

The wildfire scenario is not a post-hoc rationalization of the architecture. It reveals the principled reason why depth should track orbital hierarchy rather than being assigned arbitrarily. The information content available at each tier is genuinely hierarchical: raw sensor features at LEO, spatial propagation context at MEO, multi-event systemic patterns at GEO/HEO. A model architecture that assigns the same depth to all tiers either wastes the compute of high-orbit nodes on shallow features they could learn trivially or asks constrained LEO nodes to train representations that require multi-region context they cannot observe.

FN-HFL makes the tier-depth correspondence explicit and schedulable. When a LEO satellite’s battery is degraded, the scheduler reduces its depth assignment; the shallow feature extractor still benefits from its local observations, and no MEO or GEO layer is starved of contributors. When a MEO node loses an ISL contact window, its regional update is marked stale and down-weighted in the global aggregation; the fire-front propagation layers still receive contributions from other MEO nodes covering adjacent regions. When a GEO node detects an anomalous gradient in a deep layer, its trust score is reduced and the aggregation rule falls back to a more conservative cross-node consensus, protecting the multi-event situational awareness layers that downstream emergency response systems depend on.

The wildfire scenario also exposes the novel security risk FN-HFL must address. A compromised deep node—one that has received a malicious firmware update or is operating under a sophisticated adversarial attack—can inject a gradient into the lower layers that all LEO nodes use for basic thermal detection, while simultaneously reporting plausible fire-front propagation signals at the medium layers. Shallow LEO nodes, which never inspect the deep layers, cannot detect this attack. The defense must therefore operate layer-wise, cross-referencing gradient statistics across nodes at each depth level, using the large population of LEO-trained shallow updates as a reference distribution against which to test contributions from deeper nodes to the shared lower layers.

## 3 Related Work

Fractal architectures and anytime computation. FractalNet introduced self-similar neural macroarchitectures with paths of different lengths and drop-path regularization, showing that shallow and deep subnetworks can be jointly trained and later extracted as useful fixed-depth subnetworks [1, 2]. FN-HFL reinterprets this property at system scale: instead of randomly dropping paths on one machine, orbital contact constraints and node resources determine which path depths participate in each round.

Heterogeneous federated learning. HeteroFL, FjORD, and DepthFL study FL with heterogeneous local model sizes or depths [3, 4, 5]. These methods motivate model heterogeneity but do not co-design client capability with a hierarchical orbital topology. FN-HFL differs by making communication path length the mechanism for assigning model depth and aggregation responsibility—a co-design that the wildfire scenario shows is semantically motivated, not merely an engineering compromise.

Satellite federated learning. Prior satellite FL systems study hierarchical aggregation, decentralized LEO training, offloading, and straggler mitigation under dynamic links [6, 7, 8, 9]. These works establish feasibility, but most retain homogeneous model assumptions or treat topology as a routing constraint rather than a learning primitive.

Robust aggregation and security. Byzantine-robust FL aggregation rules such as Krum, coordinatewise median, and trimmed mean protect against malicious client updates [10, 11]. In FN-HFL, adversarial and radiation-induced corruptions may be depth-targeted, as the wildfire security discussion illustrates. This motivates depth-stratified aggregation that cannot be reduced to existing layer-agnostic robust rules.

![](images/4cccdd6547402be5b84563f1d62a2742adce9f0d8c5a20fe5d262d5dc965c8c7.jpg)  
Figure 1: FN-HFL architecture for depth-heterogeneous satellite federated learning. Edge sensing satellites in Shell 3 train only shallow model blocks, producing local updates such as $w _ { a } , w _ { b } , w _ { c } , w _ { d }$ from onboard observations. These shallow updates are transmitted as gradients to regional Shell 2 satellites, where MEO nodes perform LEO model aggregation and train additional medium-depth blocks, e.g., $( w _ { a } , \ldots , w _ { h } )$ . Shell 1 high-compute GEO/HEO nodes receive regional summaries and aggregate full-depth updates, including deeper blocks $( w _ { i } , \ldots , w _ { l } )$ , to maintain the global model. The right side illustrates the FractalNet-native depth hierarchy: shorter paths correspond to shallow LEO training, intermediate paths to MEO regional learning, and longer paths to full-depth GEO/HEO aggregation. This topology-depth mapping enables resource-aware learning in which constrained satellites contribute useful shallow features while higher-orbit nodes learn regional and global wildfire semantics without requiring raw data downlink.

## 4 System Model

## 4.1 Orbital graph and hierarchy

Let $\mathcal { G } _ { t } = ( \nu , \mathcal { E } _ { t } )$ be the time-indexed contact graph at training round t, where vertices are satellites and edges are active ISLs. Each node $i \in \mathcal V$ has a shell label $s _ { i } \in \{ 1 , 2 , 3 \}$ , with Shell 3 denoting constrained low-altitude sensing nodes, Shell 2 denoting regional aggregation nodes, and Shell 1 denoting high-compute global aggregation nodes. The contact schedule $\overset { \vartriangle } { \boldsymbol { C } } = \{ \boldsymbol { \mathcal { E } } _ { t } \} _ { t = 1 } ^ { T }$ is known over a finite horizon from orbital propagation, but execution may deviate because of link outages, attitude maneuvers, or space-weather events.

Each node has a time-varying resource vector

$$
\mathbf { c } _ { i } ( t ) = \big ( E _ { i } ( t ) , P _ { i } ( t ) , M _ { i } ( t ) , B _ { i } ( t ) , R _ { i } ( t ) , Q _ { i } ( t ) \big ) ,\tag{1}
$$

where E is energy budget, P processor availability, M memory, B link bandwidth, R radiation/faultrisk state, and $Q$ queue pressure. We write $\mathcal { N } _ { i } ( t )$ for active neighbors of i.

## 4.2 Depth-heterogeneous model

The global model is decomposed into L ordered blocks,

$$
f ( x ; { \mathbf W } ) = f _ { L } \circ f _ { L - 1 } \circ \cdots \circ f _ { 1 } ( x ) , \qquad { \mathbf W } = \{ W _ { 1 } , \ldots , W _ { L } \} .\tag{2}
$$

Each satellite trains a prefix or segment determined by a depth assignment $d _ { i } ( t ) \in \{ 1 , \ldots , L \}$ . For the canonical three-tier system, shallow LEO nodes train $d _ { S }$ layers, medium MEO nodes train $d _ { M }$ layers, and deep GEO/HEO nodes train all L layers:

$$
d _ { i } ( t ) \in \{ d _ { S } , d _ { M } , L \} , \qquad 1 \leq d _ { S } < d _ { M } < L .\tag{3}
$$

In the wildfire scenario and throughout the experimental section, we use $L = 1 0 , d _ { S } = 3$ , and $d _ { M } = 6 \mathrm { : }$ LEO nodes contribute gradients for $\{ W _ { 1 } , \dot { W _ { 2 } } , W _ { 3 } \}$ (pixel-scale feature extraction), MEO nodes for $\{ W _ { 1 } , \ldots , W _ { 6 } \}$ (regional propagation modeling), and GEO/HEO nodes for the full $\{ W _ { 1 } , \ldots , W _ { 1 0 } \}$ (multi-event situational awareness).

![](images/7a65b88f8941191538bcd00bb6230469c29067fa2967abbdc308078284f5c014.jpg)  
Figure 2: Three-tier orbital hierarchy with depth-aware model partitioning. LEO (Shell 3) lowcompute nodes train shallow model prefixes, MEO (Shell 2) mid-compute nodes train partial models and perform regional aggregation, and HEO (Shell 1) high-compute nodes train the full model and conduct global aggregation. Model depth increases with orbital tier, matching each shell’s computational capacity.

Definition 1 (Layer-wise participation mask). A node i’s depth assignment induces a binary mask $m _ { i } ^ { \ell } ( t ) = \mathcal { H } [ \ell \leq \dot { d } _ { i } ( t ) ]$ . Its update at round t is

$$
\Delta _ { i } ( t ) = \{ m _ { i } ^ { \ell } ( t ) \Delta _ { i } ^ { \ell } ( t ) \} _ { \ell = 1 } ^ { L } ,\tag{4}
$$

where missing layers are represented by $m _ { i } ^ { \ell } ( t ) = 0$ rather than zero-valued gradients.

## 4.3 Agents and authority boundary

Agents are the control plane that schedules and validates FL execution, not the FL algorithm itself. Micro-agents live on every satellite and monitor local telemetry, link state, and shallow training. Meso-agents live on selected regional nodes and aggregate shallow updates. Macro-agents live on high-compute nodes and coordinate global aggregation, model distribution, role migration, and policy enforcement. Ground remains a policy oracle and auditor: it uploads validated policy rules and model releases during contact windows, receives compact logs, and issues emergency overrides, but is not in the real-time training or inference loop.

In the wildfire scenario, this separation is operationally significant. When a micro-agent on a LEO node detects that its thermal sensor has flagged a region of interest, it can immediately increase its local training frequency and signal its assigned MEO meso-agent without waiting for a ground command. When a meso-agent detects that multiple LEO nodes in its region are converging on similar thermal anomaly gradients, it can escalate to the GEO/HEO macro-agent with a compressed situational summary. Ground receives this summary during the next contact window and can, if warranted, issue policy guidance—but the detection and initial response have already propagated through the agentic hierarchy at orbital timescales.

## 5 FN-HFL Architecture

## 5.1 Training and aggregation path

Figure 2 summarizes one training round. LEO nodes train shallow prefixes on local data and send $\Delta \mathsf { \bar { W } } _ { 1 } , \Delta \mathsf { W } _ { 2 } , \Delta \mathsf { W } _ { 3 }$ to an assigned MEO aggregator. MEO nodes robustly aggregate shallow updates, optionally train medium-depth layers $\bar { W _ { 1 } } \bar { . . . . } W _ { 6 }$ , and forward regional updates upward. Shell-1 nodes aggregate layer-wise across all received contributors, update the global model, and distribute model slices back down. In the wildfire context, the upward flow carries progressively richer fire situational awareness—from individual thermal anomalies to regional fire fronts to continental risk patterns—while the downward flow distributes updated feature detectors that improve each tier’s recognition of new fire signatures informed by what other parts of the constellation have observed.

## 5.2 Aggregator assignment

For a set of shallow nodes $S _ { t } \subset \mathcal V$ , the assigned MEO aggregator is selected by the scheduler. Let $A _ { t }$ be candidate MEO aggregators with contact to at least one shallow node during the upload window. The scheduler solves

$$
a ^ { * } ( S _ { t } ) = \arg \operatorname* { m i n } _ { a \in A _ { t } } \ \lambda _ { 1 } \operatorname* { d e l a y } ( S _ { t } , a ) + \lambda _ { 2 } \operatorname { e n e r g y } ( S _ { t } , a ) + \lambda _ { 3 } \operatorname { l o a d } ( a ) - \lambda _ { 4 } \operatorname { t r u s t } ( a ) ,\tag{5}
$$

subject to memory, compute, link-window, and security constraints. In the wildfire scenario, the scheduler additionally uses fire-region priority weights: LEO nodes currently observing active thermal anomalies are assigned aggregators with higher bandwidth budgets and lower current load, ensuring that time-sensitive detection updates are not delayed by competing background telemetry traffic. This can be solved greedily, by distributed auction, or by receding-horizon optimization over predicted contact windows.

## 6 Depth-Heterogeneous Federated Optimization

## 6.1 Objective

Each node i has local data distribution $\mathcal { D } _ { i }$ and empirical loss $F _ { i } ( \mathbf { W } ) = \mathbb { E } _ { ( x , y ) \sim \mathcal { D } _ { i } } \ell ( f ( x ; \mathbf { W } ) , y )$ The ideal full-model objective is

$$
\operatorname* { m i n } _ { \mathbf { W } } F ( \mathbf { W } ) = \sum _ { i \in \mathcal { V } } p _ { i } F _ { i } ( \mathbf { W } ) , \qquad p _ { i } = \frac { n _ { i } } { \sum _ { j } n _ { j } } .\tag{6}
$$

However, if node i only trains prefix $d _ { i }$ , it observes a masked gradient. Layer ${ \boldsymbol { \ell } } { \boldsymbol { \mathrm { s } } }$ update is aggregated over the participating set

$$
\mathcal { P } _ { \ell } ( t ) = \{ i \in \mathcal { V } _ { t } : m _ { i } ^ { \ell } ( t ) = 1 , \ \Delta _ { i } ^ { \ell } ( t ) \mathrm { ~ r e c e i v e d ~ b e f o r e ~ d e a d l i n e } \} .\tag{7}
$$

The global update is

$$
W _ { \ell } ( t + 1 ) = W _ { \ell } ( t ) - \eta _ { t } \mathrm { ~ } \mathrm { ~ A g g } _ { \ell } \left( \{ \Delta _ { i } ^ { \ell } ( t ) : i \in \mathcal { P } _ { \ell } ( t ) \} \right) , \quad \ell = 1 , \ldots , L .\tag{8}
$$

In the wildfire scenario, this means that the shallow layers $W _ { 1 } , W _ { 2 } , W _ { 3 }$ receive updates from the entire constellation—hundreds or thousands of LEO, MEO, and GEO nodes that all train these layers—while the deep layers $W _ { 8 } , W _ { 9 } , W _ { 1 0 }$ receive updates only from GEO/HEO nodes. This participation asymmetry is by design: the shallow feature detectors for thermal anomalies benefit from the broadest possible diversity of observed fire types, vegetation conditions, and sensor angles, while the deep multi-event patterns require the wide-area persistent view that only the global tier can provide.

## 6.2 Layer-wise aggregation with missing depths

A minimal weighted aggregator is

$$
\mathrm { A g g } _ { \ell } ^ { \mathrm { m e a n } } = \frac { \sum _ { i \in \mathcal { P } _ { \ell } ( t ) } \alpha _ { i } ^ { \ell } ( t ) \Delta _ { i } ^ { \ell } ( t ) } { \sum _ { i \in \mathcal { P } _ { \ell } ( t ) } \alpha _ { i } ^ { \ell } ( t ) } ,\tag{9}
$$

where $\alpha _ { i } ^ { \ell } ( t )$ may depend on sample count, staleness, trust, and layer quality. A robust alternative trims or filters per layer:

$$
\operatorname { A g g } _ { \ell } ^ { \mathrm { r o b u s t } } = \operatorname { T r i m m e d M e a n } _ { \rho _ { \ell } } \left( \{ \alpha _ { i } ^ { \ell } ( t ) \Delta _ { i } ^ { \ell } ( t ) : i \in \mathcal { P } _ { \ell } ( t ) \} \right) ,\tag{10}
$$

with layer-specific trimming rate $\rho _ { \ell }$ . Lower layers have many contributors (every LEO node in the constellation trains them) and can support strong robust statistics with low false-rejection risk. Upper layers have fewer contributors and require trust-weighted or Krum-like filtering. In the wildfire context, this means that an attempted attack on the shallow thermal-detection layers faces a very large honest reference population—thousands of LEO nodes independently observing fire signatures— making it statistically difficult to bias the aggregate. An attack on the deep multi-event layers faces a smaller population of GEO/HEO nodes but is mitigated by the trust evolution protocol described in Section 9.

## 6.3 Model consistency

A satellite does not need to store the entire global model; it must store the latest validated prefix corresponding to its assigned depth:

LEO: consistent on $W _ { 1 : d _ { S } }$ , MEO: consistent on $W _ { 1 : d _ { M } }$ , Shell-1: consistent on $W _ { 1 : L }$ . (11)

Every update packet carries (model\_id, version, round, depth, layer\_hashes). Aggregators accept stale updates only within a bounded staleness window $\tau _ { \mathrm { m a x } }$ and down-weight them using $\gamma ( \tau ) = \exp ( - \beta \tau )$ or a piecewise rule. In an active wildfire event, the staleness budget for shallow layers is tightened because fire conditions can change significantly over the timescale of multiple LEO passes, and a stale thermal feature detector from an earlier orbit may reflect an already-contained ignition rather than the current active front.

## 6.4 Scheduled pooling

A naive policy would have every node transmit at every available contact opportunity. This is wasteful: LEO contact windows recur every few minutes, but consecutive shallow updates from the same node are highly correlated and contribute diminishing marginal gradient information, while each radio activation carries a near-fixed energy cost independent of payload size. FN-HFL instead assigns each shell a pooling interval $\Pi _ { s }$ that decouples transmission cadence from raw contact-window availability: a node accumulates local updates and only transmits its pooled update once per interval, rather than once per contact.

Definition 2 (Tier pooling interval). Each shell $s \in \{ 1 , 2 , 3 \}$ is assigned a base pooling interval $\Pi _ { s } .$

$$
\Pi _ { 3 } ^ { \mathrm { ( L E O ) } } = 3 0 m i n , \qquad \Pi _ { 2 } ^ { \mathrm { ( M E O ) } } = 6 h , \qquad \Pi _ { 1 } ^ { \mathrm { ( G E O / H E O ) } } = 2 4 h .\tag{12}
$$

A node i in shell $s _ { i }$ transmits its accumulated update $\Delta _ { i } ( t )$ only at pooling boundaries $t \in \{ k \Pi _ { s _ { i } }$ $k \in \mathbb { N } \}$ }, aggregating all locally computed gradients since the previous boundary into a single pooled update rather than emitting one packet per LEO pass, per MEO contact, or per GEO/HEO window.

The interval ordering $\Pi _ { 3 } < \Pi _ { 2 } < \Pi _ { 1 }$ mirrors the depth ordering $d _ { S } < d _ { M } < L \colon$ shallow layers benefit from frequent, low-latency updates because they track fast-changing local phenomena and because their large contributor population needs only a short window to accumulate a statistically useful aggregate, whereas deep layers represent slowly-varying multi-region patterns for which 24-hour pooling loses little signal while saving substantially on the energy cost of waking the higherpower GEO/HEO radio and compute subsystems. Pooling is layer-aware: a node still respects the staleness budget $\tau _ { \mathrm { m a x } }$ of Section 9 within its pooling interval, so pooling reduces transmission frequency, not gradient freshness beyond what staleness control already tolerates.

Event-adaptive override. The base intervals in Definition 2 apply during nominal operation. When a node’s micro-agent or meso-agent raises an event-priority flag $\pi _ { i } ( t )$ above threshold $( \mathrm { e . g . }$ , an active wildfire signature at LEO), the scheduler temporarily collapses $\Pi _ { s _ { i } }$ toward the next available contact window—in the limit, transmitting at every contact, as in the staleness-tightening behavior already described for $\tau _ { \mathrm { m a x } }$ . This preserves the latency guarantees of Sections 10.4 and 10 (time-todetect, autonomous escalation) while the energy savings of scheduled pooling apply to the dominant, non-event regime that constitutes the overwhelming majority of constellation operating time.

## 7 Scheduler Design

The scheduler solves a constrained online control problem. Its objective trades convergence progress against orbital communication cost, onboard resource depletion, and event-response latency. At a high level,

$$
\operatorname* { m i n } _ { \{ d _ { i } , a _ { i } , r _ { i } , \Pi _ { s _ { i } } \} } \quad \mathbb { E } \left[ F ( \mathbf { W } ^ { t + 1 } ) - F ( \mathbf { W } ^ { t } ) \right] + \lambda _ { E } C _ { E } + \lambda _ { B } C _ { B } + \lambda _ { R } C _ { R } + \lambda _ { S } C _ { S } + \lambda _ { T } C _ { T }\tag{13}
$$

$$
\begin{array} { r } { \mathrm { s . t . } \quad d _ { i } ( t ) \leq d _ { \operatorname* { m a x } } ( \mathbf { c } _ { i } ( t ) ) , \quad r _ { i } ( t ) \subseteq \mathcal { C } _ { t } , } \end{array}\tag{14}
$$

$$
\mathrm { m e m } ( d _ { i } ) \leq M _ { i } ( t ) , \quad \mathrm { e n e r g y } ( d _ { i } , r _ { i } ) \leq E _ { i } ( t ) ,\tag{15}
$$

$$
\operatorname* { P r } [ \mathrm { m i s s e d \ d e a d l i n e \ u n d e r } \ : r _ { i } ( t ) ] \leq \epsilon , \quad q _ { i } ( t ) \geq q _ { \operatorname* { m i n } } , \quad \Pi _ { s _ { i } } ( t ) \geq \Pi _ { s _ { i } } ^ { \operatorname* { m i n } } ,\tag{16}
$$

## Scheduled pooling: tier-specific transmission cadence

![](images/c9da53fde9f95c381ec454ac41c389bfef78472126079e3bef3b760ab3929903.jpg)  
Figure 3: Scheduled pooling cadence by tier. Each marker denotes one pooled uplink under the base intervals of Definition 2: LEO pools every 30 minutes, MEO every 6 hours, and GEO/HEO every 24 hours, rather than transmitting at every raw contact opportunity. The shaded band illustrates the event-adaptive override: when an active wildfire signature raises a node’s event-priority flag, pooling collapses toward per-contact transmission (orange markers) until the event subsides, after which the base cadence resumes.

where $C _ { T }$ is an event-response latency cost that is elevated during active wildfire events and drives the scheduler to prioritize short-path shallow updates from fire-observing LEO nodes over background telemetry tasks. The terms $\mathsf { \bar { C } } _ { E } , C _ { B } , C _ { R } ,$ , and $C _ { S }$ denote energy, bandwidth, radiation-risk, and staleness costs respectively. The pooling interval $\Pi _ { s _ { i } } ( t )$ is now a first-class scheduler decision variable alongside depth $d _ { i }$ , aggregator $a _ { i } .$ , and route $r _ { i } { : }$ it is initialized to the tier base value of Definition 2 and is the primary lever the scheduler uses to trade $C _ { E }$ against $C _ { T }$ , since lengthening $\Pi _ { s _ { i } }$ reduces the number of radio activations (and thus $C _ { E } )$ at the cost of coarser update freshness, while the event-adaptive override collapses $\Pi _ { s _ { i } }$ toward per-contact transmission whenever $C _ { T }$ dominates.

In practice, FN-HFL uses three horizons: a long-horizon global assignment over predicted orbital contacts, a medium-horizon update using fresh SWaP-C telemetry and event-priority signals, and a short-horizon failover policy for unexpected ISL disruptions. Scheduled pooling operates on top of this long-horizon assignment: the base intervals $\Pi _ { 3 } , \bar { \Pi } _ { 2 } , \Pi _ { 1 }$ are set at mission-planning time from predicted contact statistics and are only revised by the medium-horizon update when SWaP-C state or event-priority signals warrant a deviation. During an active fire event, the medium-horizon update frequency is increased to adapt to rapidly changing LEO observation priorities, and pooling intervals for affected nodes collapse toward per-contact transmission as described in Section 6.4.

## 8 Theoretical Framing

This section states the main theoretical targets. The point is to make precise which convergence and robustness properties the experimental system is designed to test.

Assumption 1 (Smoothness and bounded layer variance). For each layer ℓ, local stochastic gradients are unbiasedfor the masked local objective and have bounded variance $\sigma _ { \ell } ^ { 2 } .$ . Each local objective is $L _ { F }$ -smooth in the active parameter block.

Assumption 2 (Bounded participation gap). For every layer ℓ, there exists a window $H _ { \ell }$ such that at least $K _ { \ell }$ non-compromised updates for layer ℓ are received in every interval of length $H _ { \ell } .$ . Lower layers have $K _ { \ell }$ dominated by shallow, medium, and deep nodes; upper layers have $K _ { \ell }$ dominated by deep nodes.

Algorithm 1 One FN-HFL training round with scheduled pooling   
Require: Global model $\mathbf { W } ^ { t }$ , contact forecast $\mathcal { C } _ { t } ,$ node resources $\{ \mathbf { c } _ { i } ( t ) \}$ , trust scores $\{ q _ { i } ( t ) \}$ , event  
priority weights $\{ \pi _ { i } ( t ) \}$ , pooling intervals $\{ \Pi _ { s } \}$   
1: Scheduler assigns depth $d _ { i } ( t )$ , parent aggregator $a _ { i } ( t )$ , route $r _ { i } ( t )$ , staleness budget $\tau _ { i } ( t )$ , priority   
weight $\pi _ { i } ( t )$ , and (event-adjusted) pooling interval $\textstyle \operatorname { I I } _ { s _ { i } } ( t )$   
2: for each LEO shallow node i in parallel do   
3: Download or verify latest prefix ${ W } _ { 1 : d _ { S } } ^ { t }$   
4: Train locally on onboard sensor data; accumulate $\Delta _ { i } ^ { 1 : d _ { S } }$ across passes   
5: if current time is a pooling boundary for $\Pi _ { s _ { i } } ( t )$ , or ${ \dot { \pi } } _ { i } ( t )$ exceeds the event threshold then   
6: Send pooled packet $( \Delta _ { i } ^ { 1 : d _ { S } } , n _ { i } ,$ version, hash, $q _ { i } , \pi _ { i } )$ to assigned MEO $a _ { i } ( t )$   
7: end if   
8: end for   
9: for each MEO aggregator a in parallel do   
10: Verify packet signatures, hashes, version, and staleness   
11: Aggregate shallow updates per layer $1 : d _ { S }$ using priority-weighted robust aggregation   
12: Optionally train/aggregate medium-depth update $\stackrel { \bullet } { \Delta } _ { a } ^ { 1 : d _ { M } }$ on regional data   
13: Escalate fire-event signal to Shell-1 if priority threshold exceeded   
14: if current time is a pooling boundary for $\Pi _ { 2 } ,$ or an escalation is pending then   
15: Send regional packet $\overline { { ( \Delta } } _ { a } ^ { 1 : d _ { M } }$ , coverage, quality, trust, event\_flag) to Shell-1 aggregator   
16: end if   
17: end for   
18: Shell-1 aggregator computes depth-stratified update for each layer $\ell = 1 , \ldots , L$ at its own $\Pi _ { 1 }$   
pooling boundary   
19: Produce $\mathbf { W } ^ { t + 1 }$ , versioned layer hashes, and distribution manifest   
20: Distribute model slices downward: $W _ { 1 : L }$ to Shell-1 peers, $W _ { 1 : d _ { M } }$ to MEOs, $W _ { 1 : d _ { S } }$ to LEOs

Proposition 1 (Depth-induced gradient bias, informal). Let $g ^ { \ell } ( t )$ be thefull-population gradientfor layer ℓ, and let $\hat { g } ^ { \ell } ( t )$ be the layer-wise FN-HFL estimate. Then

$$
\begin{array} { r } { \mathbb { E } \| \hat { g } ^ { \ell } ( t ) - g ^ { \ell } ( t ) \| \le \underbrace { B _ { \ell } ^ { \mathrm { d e p t h } } ( \pi _ { t } ) } _ { d e p t h p a r t i c i p a t i o n } + \underbrace { B _ { \ell } ^ { \mathrm { n o n I I D } } } _ { d a t a h e t e r o g e n e i t y } + \underbrace { B _ { \ell } ^ { \mathrm { s t a l e } } ( \tau _ { \mathrm { m a x } } ) } _ { c o n t a c t \cdot w i n d o w s t a l e n e s s } , } \end{array}\tag{17}
$$

where $\pi _ { t }$ is the path-depth distribution selected by the scheduler. In the wildfire scenario, the data heterogeneity term $B _ { \ell } ^ { \mathrm { n o n I I D } }$ is high for shallow layers when different LEO nodes observe structurally differentfire types (grass fires versusforest crown fires), but this is precisely where the large LEO participation count provides natural diversity-induced averaging.

Theorem 1 (Convergence target, informal). Under smoothness, boundedparticipation gaps, bounded staleness, and bounded robust-aggregation error, FN-HFL should converge to a stationary neighborhood whose radius scales with the depth participation bias, layer-wise variance, non-IID data skew, and robustfiltering error. Establishing the tightform of this bound is a primary theoretical contribution targeted by the full paper.

## 9 Security and Fault Model

FN-HFL treats radiation faults and adversarial compromise under a unified update-integrity interface. A corrupted update may be caused by a single-event upset, stale memory, compromised firmware, poisoned local data, or a malicious gradient. The key new risk—motivated directly by the wildfire scenario analysis in Section 2—is depth-targeted poisoning: a deep node can send updates that appear plausible for full-model training while injecting a malicious signal into lower layers used by all satellites. Because shallow satellites do not inspect deeper representations, defense must operate layer-wise.

The defense has four components. First, layer-specific robust aggregation compares update norms, cosine similarity, and historical behavior among contributors at each depth level. For lower layers, the large shallow-node population (all LEO nodes training fire-detection features) provides a dense reference distribution that makes statistical outlier detection effective even against sophisticated attacks. Second, cross-depth consistency checks verify that updates to lower layers from deep nodes are consistent with the distribution of updates to those same layers from shallow nodes. A deep node sending updates that would move lower-layer weights far outside the distribution of LEO-trained updates on similar data is flagged for trust review. Third, version and hash verification ensures update packets are not replayed from earlier rounds when fire conditions were different. Fourth, trust evolution maintains per-node trust scores updated based on observed gradient behavior; a node whose updates are repeatedly flagged as inconsistent with the shallow-node reference distribution has its trust score reduced, causing its contributions to be down-weighted in the robust aggregation until a ground audit can validate its integrity.

## 10 Experiments

We evaluate FN-HFL across six axes: convergence quality, communication efficiency, resource adaptation, robustness under adversarial and fault conditions, operational latency, and energy savings from scheduled pooling. All experiments use the wildfire detection task as the primary benchmark, with intrusion detection and predictive health as secondary generalization tests. Unless stated otherwise, results are reported as mean ± one standard deviation over ten independent runs with different random seeds, constellation initializations, and contact-schedule perturbations.

## 10.1 Experimental setup

Simulator. We implement a discrete-event orbital simulator in Python, using SGP4 propagation [12] with two-line element sets from a synthetic Walker-star constellation to generate contact windows. Each simulation step corresponds to one orbital period (≈94 minutes for LEO at 550 km). ISL outages are injected as independent Bernoulli events with probability $p _ { \mathrm { o u t } } = 0 . 0 8$ per link per round, calibrated to observed outage rates in published LEO mesh studies. $\mathrm { S W a P – C }$ state evolves according to a physics-informed battery and thermal model: battery state-of-charge follows a charge-discharge cycle tied to eclipse fraction (≈37% for 550 km sun-synchronous), and processor availability degrades stochastically with radiation dose accumulated over the simulated mission duration.

Constellation configurations. We run three constellation sizes: Small (108 satellites: 90 LEO / 12 MEO / 6 GEO), Medium (540 satellites: 450 LEO / 60 MEO / 30 GEO), and Large (1080 satellites: 900 LEO / 120 MEO / 60 GEO). These ratios reflect realistic mega-constellation designs in which LEO sensing nodes vastly outnumber aggregation nodes. Compute budgets are assigned heterogeneously: LEO nodes draw from {ARM Cortex-M7-class, LEON4-class} profiles with 0.5– 4 GFLOPS, MEO nodes from edge-accelerator profiles with 10–40 GFLOPS, and GEO/HEO nodes from high-compute profiles with 80–200 GFLOPS.

Model architecture. The global model is a 10-block convolutional backbone with skip connections, approximately 12M parameters total. Shallow prefix $( d _ { S } = 3$ blocks, ≈1.1M parameters), medium prefix $( d _ { M } = 6$ blocks, ≈5.4M parameters), and full model (L = 10 blocks, ≈12M parameters). For LEO nodes this translates to a training memory footprint of ≈18 MB; MEO nodes require ≈87 MB; GEO/HEO nodes require the full ≈192 MB footprint including optimizer state.

Data distribution. For wildfire detection, we partition a synthetic multispectral tile dataset of 2.4M labeled samples across the constellation using four regimes: IID (uniform random), Orbital-plane non-IID (satellites in the same orbital plane share fire-type distribution bias), Geography-driven non-IID (LEO nodes over specific biomes observe structurally different fire signatures), and Faultskewed non-IID (degraded nodes receive corrupted or low-quality sensor readings). All main results use Geography-driven non-IID as the default, as it is the most operationally realistic.

Training protocol. Each local training step uses SGD with momentum 0.9 and weight decay $1 0 ^ { - 4 }$ Local epochs per round: 3 for LEO nodes, 5 for MEO nodes, 8 for GEO/HEO nodes, reflecting the longer contact windows available at higher orbits. Global learning rate is $\eta _ { 0 } = 0 . 0 5$ with cosine decay. We train for 300 global rounds (approximately 18.75 simulated days) unless early stopping on a held-out validation constellation triggers first. Staleness budget: $\tau _ { \operatorname* { m a x } } = 4$ rounds for shallow layers, $\tau _ { \operatorname* { m a x } } = 2$ rounds for deep layers, tightened to $\tau _ { \operatorname* { m a x } } = 1$ during active fire-event windows. Unless an active fire event is in progress, FN-HFL pools updates per Definition 2 at base intervals of

![](images/580a769b731e68408a55a31fd6c9821cc33c57107cb89064e80abb5c76667a8d.jpg)  
Figure 4: Convergence curves (AUROC vs. global round) for all methods on the wildfire detection task (Medium constellation, Geography-driven non-IID). Shaded bands show ±1 standard deviation over 10 seeds. Dashed horizontal line marks the 80% AUROC target. FN-HFL reaches target in 47 rounds—1.7× faster than FedAvg-full and 1.9× faster than Hierarchical FL—and converges to a higher final AUROC than all baselines.

30 minutes (LEO), 6 hours (MEO), and 24 hours (GEO/HEO); local gradients continue to accumulate between pooling boundaries and are not discarded.

Baselines. We compare against six baselines spanning the design space:

1. FedAvg-full: all participating nodes train the full model; constrained nodes drop out if infeasible.

2. FedAvg-small: all nodes train a shallow model sized for the weakest satellite.

3. Hierarchical FL: LEO-to-MEO-to-global aggregation with homogeneous model depth.

4. HeteroFL/FjORD-style: heterogeneous local model capacity without topology-aware depth scheduling.

5. DepthFL-style: depth-scaled local models assigned by compute budget but independent of orbital path structure.

6. Split-FL/offloading: constrained nodes offload intermediate activations to stronger nodes.

All six baselines follow the opportunistic convention from prior satellite-FL work: each node transmits at every available contact window rather than pooling across windows. Table 1 and Sections 10.2– 10.6 report results under this common opportunistic-transmission setting for all methods, including FN-HFL, so that convergence, communication, and robustness comparisons isolate the effect of depth heterogeneity alone. Section 10.10 then isolates the additional, separable effect of scheduled pooling by re-enabling it only for FN-HFL and reporting the resulting energy delta.

## 10.2 Research question 1: Convergence under extreme heterogeneity

Figure 4 shows AUROC versus global round for all methods on the Medium constellation under Geography-driven non-IID. Table 1 reports final-round performance and rounds-to-target (80% AUROC).

FN-HFL achieves a final AUROC of 0.891 ± 0.008, compared to 0.847 ± 0.012 for Hierarchical FL and 0.823 ± 0.019 for DepthFL-style. FedAvg-full achieves 0.863 ± 0.014 but only by excluding 61% of LEO nodes that cannot meet the full-model memory requirement, effectively wasting the majority of the sensing constellation. FedAvg-small reaches only 0.771 ± 0.021, confirming that designing for the weakest node sacrifices substantial representational capacity.

Table 1: Main experimental comparison across all methods (Medium constellation, Geography-driven non-IID, wildfire detection primary task). ↑: higher is better; ↓: lower is better. Bold indicates best result; † indicates statistically significant improvement over next-best $( p < 0 . 0 5 .$ , paired permutation test).
<table><tr><td>Method</td><td>AUROC↑</td><td></td><td>Rounds to 80% ↓ ISL bytes/round (MB) ↓ Energy/round (kJ) ↓ Missed windows (%) ↓ Attack success (%) ↓</td><td></td><td></td><td></td></tr><tr><td>FedAvg-full</td><td> $0 . 8 6 3 \pm 0 . 0 1 4$ </td><td> $7 8 \pm 9$ </td><td> $^ { 2 , 8 4 7 \pm 2 0 3 }$ </td><td> $1 8 . 4 \pm 1 . 9$ </td><td> $2 9 . 3 \pm 3 . 1$ </td><td> $4 1 . 2 \pm 4 . 8$ </td></tr><tr><td>FedAvg-small</td><td> $0 . 7 7 1 \pm 0 . 0 2 1$ </td><td> $> 3 0 0$ </td><td>312 ± 28</td><td> ${ \bf 4 . 1 \pm 0 . 6 }$ </td><td> ${ \bf 3 . 2 \pm 0 . 8 }$ </td><td> $3 8 . 7 \pm 5 . 2 $ </td></tr><tr><td>Hierarchical FL</td><td> $0 . 8 4 7 \pm 0 . 0 1 2$ </td><td> $8 9 \pm 1 1$ </td><td> $1 , 9 2 3 \pm 1 6 7$ </td><td> $1 4 . 2 \pm 1 . 4$ </td><td> $1 7 . 6 \pm 2 . 4$ </td><td> $4 3 . 5 \pm 4 . 1$ </td></tr><tr><td>HeteroFL/FjORD</td><td> $0 . 8 2 3 \pm 0 . 0 1 6$ </td><td> $1 1 2 \pm 1 7$ </td><td> $1 , 6 5 4 \pm 1 4 4$ </td><td> $1 1 . 8 \pm 1 . 3$ </td><td> $1 4 . 9 \pm 2 . 1$ </td><td> $3 9 . 1 \pm 5 . 0$ </td></tr><tr><td>DepthFL-style</td><td> $0 . 8 3 1 \pm 0 . 0 1 9$ </td><td> $9 8 \pm 1 4$ </td><td> $1 , 7 4 1 \pm 1 5 8$ </td><td> $1 2 . 6 \pm 1 . 5$ </td><td> $1 6 . 3 \pm 2 . 3$ </td><td> $3 7 . 4 \pm 4 . 6$ </td></tr><tr><td>Split-FL</td><td> $0 . 8 4 4 \pm 0 . 0 1 3$ </td><td> $8 4 \pm 1 0 $ </td><td> $^ { 3 , 2 1 4 \pm 2 8 9 }$ </td><td> $2 1 . 3 \pm 2 . 4$ </td><td> $2 2 . 8 \pm 3 . 4$ </td><td> $4 0 . 2 \pm 4 . 3$ </td></tr><tr><td>FN-HFL (ours)</td><td> $\mathbf { 0 . 8 9 1 \pm 0 . 0 0 8 ^ { \dagger } }$ </td><td> ${ \bf 4 7 \pm 4 ^ { \dagger } }$ </td><td> ${ \bf 8 9 4 \pm 6 1 } ^ { \dag }$ </td><td> ${ \bf 7 . 3 \pm 0 . 7 ^ { \dagger } }$ </td><td> ${ \bf 6 . 1 \pm 1 . 0 ^ { \dagger } }$ </td><td> ${ \bf 1 1 . 4 \pm 2 . 2 ^ { \dagger } }$ </td></tr></table>

Rounds-to-target reveals a sharper advantage. FN-HFL reaches 80% AUROC in ${ \bf 4 7 \pm 4 }$ rounds, versus $8 9 \pm 1 1$ for Hierarchical FL and $1 1 2 \pm 1 7$ for HeteroFL/FjORD-style. The acceleration arises from two compounding effects: shallow LEO nodes contribute useful gradient signal from round 1 rather than dropping out, and MEO regional aggregation prevents per-round gradient noise from overwhelming the global signal.

Across constellation sizes, FN-HFL scales consistently: Large constellation AUROC is $0 . 8 9 6 \pm$ 0.006, improving over Small $( 0 . 8 7 4 \pm 0 . 0 1 1 )$ . By contrast, FedAvg-full shows a declining effective participation rate at scale $( 3 9 \%  3 1 \%$ as more constrained LEO nodes are added), yielding a Large-constellation AUROC of only $0 . 8 5 1 \pm 0 . 0 1 7$ despite nominally having more data.

## 10.3 Research question 2: Communication efficiency

ISL bandwidth is the principal bottleneck in satellite FL: link capacities range from ≈200 Mbps for laser ISLs to ≈10 Mbps for $\mathrm { R F I S L s } ,$ and contact windows are measured in minutes. As shown in Table 1, FN-HFL transmits ${ \bf 8 9 4 \pm 6 1 }$ MB per round—a 53.5% reduction over Hierarchical FL (1,923 MB) and a 68.6% reduction over FedAvg-full (2,847 MB). Split-FL has the highest cost (3,214 MB) because intermediate activation tensors are large relative to gradient updates.

The reduction arises from two mechanisms. First, LEO-to-MEO transmission carries only the shallow update $( \Delta W _ { 1 : 3 } ,$ , ≈4.2 MB per node) rather than a full-model gradient. Second, MEO-to-GEO carries a single compressed regional aggregate rather than relaying dozens of individual LEO packets upward. FN-HFL achieves 80% AUROC at a cumulative transmission cost of ≈42 GB, compared to ≈222 GB for FedAvg-full at the same performance level.

The missed-window rate is ${ \bf 6 . 1 \pm 1 . 0 \% }$ for FN-HFL, versus $2 9 . 3 \pm 3 . 1 \%$ for FedAvg-full. Shallow gradient packets (≈4.2 MB) are dramatically more likely to complete within a 3–7 minute LEO contact window than full-model gradient packets (≈46 MB).

## 10.4 Research question 3: Resource adaptation under SWaP-C degradation

We simulate a 90-day fire season during which batteries degrade 15%, radiation-induced compute faults accumulate at 0.3 soft errors per node per simulated day, and 8% of LEO nodes undergo a permanent processor downgrade to 40% capacity at day 60. Figure 5 tracks AUROC, effective participation rate, and energy per round over the mission.

FN-HFL degrades gracefully: AUROC declines from 0.891 at day 0 to $0 . 8 7 8 \pm 0 . 0 1 1$ at day 90, a relative degradation of only 1.5%. The scheduler dynamically reduces depth assignments for degraded nodes (from $d _ { S } = 3$ to $d _ { S } = 2$ for the most constrained), maintains coverage of lower layers via remaining healthy LEO nodes, and migrates meso-agent responsibility to MEO nodes with lower accumulated radiation dose. Hierarchical FL degrades more sharply to $0 . 8 2 1 \pm 0 . 0 1 8$ at day 90 (3.1% relative decline) because it cannot reduce per-node training load without dropping nodes entirely, causing effective participation to fall from 78% to 58%. FedAvg-full’s participation collapses to 24% at day 90, yielding AUROC $0 . 8 0 9 \pm 0 . 0 2 4$

Notably, FN-HFL’s energy consumption decreases from $7 . 3 \pm 0 . 7$ kJ at day 0 to $6 . 8 \pm 0 . 9 \mathrm { k J }$ at day 90, as the scheduler trades marginal gradient quality for sustainable participation on degraded nodes. This is a design feature: the scheduler explicitly penalizes energy expenditure relative to expected gradient contribution.

![](images/2ab1c595e087fd6597d8153703aa7c9296653a4d14a84e2ae8e40e4f4a73ce2c.jpg)

![](images/bfa4c4e48d71de0022ae60a2ccac8711d28149cd9322ea7a6eeb2fb48381443e.jpg)

![](images/f2d659ae95f55f6be12eb44e0ba28831ddc84b40e1a0f172b5fc57b89433adc7.jpg)  
Figure 5: Resource adaptation over a 90-day simulated fire season. $L e f t { \mathrm { : } }$ AUROC degradation. Center: effective node participation rate. $R i g h t \colon$ energy per round. The vertical dotted line marks day 60 (onset of permanent LEO processor downgrade). FN-HFL maintains near-constant participation by dynamically reducing depth assignments on degraded nodes, and the scheduler’s energy adaptation yields a slight decrease in per-round energy by day 90 despite a growing degraded node population.

## 10.5 Research question 4: Robustness under adversarial and radiation faults

Depth-targeted poisoning. We inject attacks in which a fraction $f \in \{ 0 . 0 5 , 0 . 1 0 , 0 . 2 0 , 0 . 3 0 \}$ of GEO/HEO nodes send sign-flipped gradients to lower layers $W _ { 1 : 3 }$ while sending plausible gradients to upper layers $W _ { 7 : 1 0 }$ (to evade layer-agnostic detection). $\mathrm { A t } \ f = 0 . 1 0 , \mathrm { F N - H F L ^ { \circ } s }$ depth-stratified aggregation holds attack success to ${ \bf 1 1 . 4 \pm 2 . 2 \% }$ , compared to $4 3 . 5 \pm 4 . 1 \%$ for Hierarchical FL and $4 \bar { 1 } . 2 \pm 4 . 8 \%$ for FedAvg-full (Table 1). The defense is effective because ≈450 LEO-trained shallow updates provide a dense reference distribution: compromised GEO/HEO gradients for $W _ { 1 : 3 }$ are statistical outliers that the trimmed-mean aggregator rejects with high probability.

$\operatorname { A t } f = 0 . 2 0 , \operatorname { F N - H F I }$ degrades to $1 8 . 9 \pm 3 . 1 \%$ attack success, still well below the $> 5 0 \%$ success rate of all baselines. At $f = 0 . 3 0$ , all methods degrade significantly; FN-HFL reaches $3 1 . 4 \pm 4 . 2 \%$ as the GEO/HEO tier becomes too small to support strong robust statistics—a known boundary of trimmed-mean rules requiring a majority of honest contributors.

Radiation-induced update corruption. At $p _ { \mathrm { S E U } } = 1 0 ^ { - 3 }$ (conservative COTS processor estimate for low-inclination LEO), FN-HFL’s hash verification and gradient-norm filtering catch $9 4 . 3 \pm 1 . 8 \%$ of corrupted updates before aggregation. The remaining uncaught corruptions cause an AUROC degradation of ${ \mathrm { \bar { 0 } } } . 0 1 2 \pm 0 . 0 0 3 ,$ , comparable to FedAvg-full $( 0 . 0 1 1 \bar { \pm } 0 . 0 0 4 )$ but at substantially lower communication and energy cost.

## 10.6 Research question 5: Operational latency and wildfire-specific metrics

Figure 6 shows time-to-detect and autonomous escalation rate versus contact-window sparsity.

Time-to-detect. FN-HFL achieves mean time-to-detect of ${ \bf 2 . 3 \pm 0 . 6 }$ rounds for high-intensity fires and ${ \bf 4 . 1 \pm 1 . 1 }$ rounds for low-intensity fires (one round ≈ 94 simulated minutes). Hierarchical FL requires $4 . 8 \pm 1 .$ 4 and $9 . 3 \pm 2 . 7$ rounds respectively; FedAvg-full requires $6 . 2 \pm 2 . 1$ and $1 3 . 7 \pm 4 . 4$ rounds. The FN-HFL advantage is largest for low-intensity smoldering fires: the broadly-trained shallow LEO feature detectors are more sensitive to subtle early thermal signatures than shallow layers trained only on the subset of satellites able to afford full-model training.

As shown in Figure 6, the advantage compounds with sparsity. At 50% contact-window outage, FN-HFL’s time-to-detect for high-intensity fires rises to only $5 . 1 \pm 0 . 9$ rounds, whereas Hierarchical FL reaches $1 5 . 9 \pm 3 . 1$ rounds—a 3.1× gap that grows from 2.1× at zero sparsity. The architecture’s smaller per-packet footprint means updates are more likely to complete transmission within the available window even when outages are frequent.

![](images/414250953737cf61009dc86284c6cf5f05e9012dba3c64fde82ee6cbac858d5e.jpg)

![](images/37af32a932d8f184b2b95477684c329e1881660077968643e12f9fce300ba625.jpg)

Fraction of fire events escalated before next ground contact window (no ground-in-the-loop required)  
![](images/ac8e5f05dbdff45b701aca4862a9e042ae26352d64047cc26669cd1176975c63.jpg)  
Figure 6: Time-to-detect (rounds from ignition to first correct Shell-1 escalation at confidence $\geq 0 . 8 5 )$ versus contact-window outage rate, for high-intensity fires (top left) and low-intensity fires $( t o p r i g h t )$ Shaded bands show ±1 std dev over 200 fire events per sparsity level. Bottom: autonomous escalation rate (fraction of events escalated before the next ground contact window) at three sparsity levels. FN-HFL maintains high autonomous escalation even at 50% sparsity, where Hierarchical FL falls below 50%.

Time-to-contain. FN-HFL achieves 9.4 ± 2.1 rounds to consistent regional fire-perimeter classification, versus 18.2 ± 4.3 for Hierarchical FL. The difference reflects faster convergence of medium-depth regional layers: MEO nodes receive higher-quality shallow pre-features from the broadly-trained LEO prefix, requiring fewer additional training steps to accurately delineate fire boundaries.

Ground-loop elimination. As shown in Figure 6, FN-HFL autonomously escalates 91. $\mathbf { 2 \pm 2 . 8 \% }$ of high-intensity and ${ \bf 7 4 . 6 \pm 4 . 1 \% }$ of low-intensity fire events before the next ground contact window (mean inter-contact interval ≈6 hours). Even at 50% ISL sparsity, FN-HFL maintains 76.4% autonomous escalation for high-intensity events, compared to 41.2% for Hierarchical FL. The 25% failure rate for low-intensity events reflects genuine detection uncertainty that appropriately defers to ground review, consistent with the policy design.

Table 2: Ablation study. Each row removes one FN-HFL component. ∆AUROC is relative to the full FN-HFL system (0.891). Medium constellation, Geography-driven non-IID.
<table><tr><td>Ablation</td><td>AUROC ↑</td><td>ΔAUROC</td><td>Rounds to 80%↓</td></tr><tr><td>FN-HFL (full system)</td><td> $0 . 8 9 1 \pm 0 . 0 0 8$ </td><td></td><td> $4 7 \pm 4$ </td></tr><tr><td>(A1) No topology-aware aggregator selection</td><td> $0 . 8 6 9 \pm 0 . 0 1 3$ </td><td>-0.022</td><td> $6 8 \pm 9$ </td></tr><tr><td>(A2) No depth heterogeneity (uniform d = 6)</td><td> $0 . 8 5 1 \pm 0 . 0 1 5$ </td><td>-0.040</td><td> $9 1 \pm 1 2$ </td></tr><tr><td>(A3) No robust aggregation (layer-wise mean)</td><td> $0 . 8 4 4 \pm 0 . 0 1 8$ </td><td>-0.047</td><td> $8 3 \pm 1 1$ </td></tr><tr><td>(A4) No staleness control</td><td> $0 . 8 6 2 \pm 0 . 0 1 6$ </td><td>-0.029</td><td> $7 4 \pm 1 0$ </td></tr><tr><td>(A5) No agent role migration</td><td> $0 . 8 7 3 \pm 0 . 0 1 2$ </td><td>-0.018</td><td> $6 1 \pm 7$ </td></tr><tr><td>(A6) No event-priority weighting</td><td> $0 . 8 7 7 \pm 0 . 0 1 1$ </td><td>-0.014</td><td> $5 9 \pm 6$ </td></tr></table>

Table 3: Generalization to secondary tasks (Medium constellation, Geography-driven non-IID).
<table><tr><td>Method</td><td>Intrusion detection F1 ↑ Health prediction MAE (days) ↓</td><td></td></tr><tr><td>FedAvg-full</td><td> $0 . 8 1 9 \pm 0 . 0 1 8$ </td><td> $2 . 9 \pm 0 . 7$ </td></tr><tr><td>Hierarchical FL</td><td> $0 . 8 3 1 \pm 0 . 0 1 6$ </td><td> $2 . 7 \pm 0 . 6$ </td></tr><tr><td>DepthFL-style</td><td> $0 . 8 4 1 \pm 0 . 0 1 4$ </td><td> $2 . 4 \pm 0 . 5$ </td></tr><tr><td>FN-HFL (ours)</td><td> $\mathbf { 0 . 8 7 3 \pm 0 . 0 1 1 }$ </td><td> ${ \bf 1 . 8 \pm 0 . 4 }$ </td></tr></table>

## 10.7 Ablation study

Table 2 reports six ablations on the Medium constellation under Geography-driven non-IID.

The largest single-component effect is (A2) removing depth heterogeneity (−0.040 AUROC, rounds-to-target 47 → 91), confirming that the depth-topology co-design is the core architectural contribution. Forcing d = 6 uniformly causes 61% of LEO nodes to drop out (exceeding memory budget) while preventing GEO/HEO nodes from training the deep multi-event representations tha fire risk classification requires.

(A3) Removing robust aggregation produces the second-largest drop (−0.047), but this is dominated by the adversarial scenario: in no-attack conditions, removing robust aggregation causes only a −0.012 AUROC drop, suggesting that the depth-stratified robust rule earns most of its value under adversarial pressure. This is practically useful: practitioners deploying FN-HFL in trusted constellations can use the simpler layer-wise mean without significant performance penalty, reserving the robust aggregation for threat-elevated contexts.

(A4) Removing staleness control (−0.029) is concentrated in time-to-detect degradation for lowintensity fires (4.1±1.1 → 7.8±2.4 rounds), confirming that staleness control is especially important for time-sensitive detection tasks where fire conditions change rapidly across LEO passes.

(A5) and (A6) have smaller but significant effects, primarily visible in the 90-day resource adaptation experiment of Section 10.4 rather than in round-0 accuracy.

## 10.8 Generalization to secondary tasks

Table 3 summarizes FN-HFL performance on intrusion detection and predictive satellite health. The consistent gains across all three tasks confirm that the depth-topology co-design generalizes to any sensing task whose information hierarchy aligns naturally with the orbital tier structure.

## 10.9 Pareto analysis

FN-HFL dominates all baselines on the Pareto frontier between wildfire detection AUROC and ISL bytes per round: it achieves higher AUROC at lower communication cost than every comparison method. The closest competitor is Split-FL, which achieves comparable AUROC (0.844) but at 3.6× the communication cost—infeasible for ISL-constrained constellations during active fire events when link capacity is under pressure from telemetry and state-estimation traffic.

Table 4: Energy consumption with and without scheduled pooling, by orbital tier (Medium constellation, Geography-driven non-IID, nominal non-event operation). Baselines transmit at every contact window, matching their Table 1 configuration; only FN-HFL pools. Energy per node-day includes radio activation, compute, and idle-listening costs.
<table><tr><td>Method</td><td></td><td></td><td>LEO (J/node-day) ↓ MEO (J/node-day) ↓ GEO/HEO (J/node-day) ↓</td></tr><tr><td>FedAvg-full</td><td> $6 1 2 \pm 3 8$ </td><td> $1 , 8 4 0 \pm 9 5 $ </td><td> $^ { 3 , 1 2 0 \pm 1 6 0 }$ </td></tr><tr><td>Hierarchical FL</td><td> $4 8 7 \pm 2 9$ </td><td> $1 , 5 1 0 \pm 8 1$ </td><td> $2 , 6 4 0 \pm 1 4 2$ </td></tr><tr><td>DepthFL-style</td><td> $4 0 1 \pm 2 6$ </td><td> $1 , 2 6 0 \pm 7 4$ </td><td> $^ { 2 , 3 1 0 \pm 1 3 1 }$ </td></tr><tr><td>FN-HFL (opportunistic, no pooling)</td><td> $3 5 6 \pm 2 4$ </td><td> $1 , 0 7 0 \pm 6 8$ </td><td> $1 { , } 9 8 0 \pm 1 1 8$ </td></tr><tr><td>FN-HFL (scheduled pooling)</td><td> ${ \bf 2 1 4 \pm 1 7 ^ { \dagger } }$ </td><td> ${ \bf 6 4 5 \pm 4 4 } ^ { \dagger }$ </td><td> $\mathbf { 1 , 2 0 5 \pm 7 9 } ^ { \dagger }$ </td></tr></table>

Two qualifications apply to interpreting these results. First, our simulator uses idealized contactwindow prediction; real constellations will experience additional scheduling uncertainty from attitudecontrol maneuvers and unforecast space-weather events, likely widening the missed-window gap beyond what we report. Second, the 12M-parameter backbone is on the smaller end of what GEO/HEO nodes with 80–200 GFLOPS could support; a larger global model would increase the relative advantage of depth-heterogeneous training (since FedAvg-full’s participation rate would decrease further) but would require recalibrating the staleness budget for upper layers.

## 10.10 Research question 6: Energy savings from scheduled pooling

All results so far (Table 1, Sections 10.2–10.6) are reported under opportunistic transmission, where every method—including FN-HFL—transmits at every available contact window. We now isolate the additional, separable energy effect of scheduled pooling (Section 6.4, Definition 2) by re-running the Medium constellation under Geography-driven non-IID with pooling enabled only for FN-HFL, using base intervals $\Pi _ { 3 } = 3 0 \mathrm { m i n } , \Pi _ { 2 } = 6 \mathrm { h } , \Pi _ { 1 } = 2 4 \mathrm { h }$ , collapsed to per-contact transmission during active-event windows as in Section 10.6. We report energy per node-day rather than energy per round, since pooling changes how many transmissions occur per unit time, making per-round comparisons across differently-pooled methods misleading.

Table 4 shows that scheduled pooling reduces FN-HFL’s already-lowest energy footprint by a further 39.9% at LEO, 39.7% at MEO, and 39.1% at GEO/HEO relative to FN-HFL without pooling, and by 65.0%, 65.0%, and 61.4% respectively relative to FedAvg-full. The reduction is roughly tier-uniform in percentage terms despite the very different absolute energy budgets, because in all three tiers radio activation—not payload transmission—dominates per-event energy cost, and pooling reduces the number of activations by a similar relative factor at each tier given our base intervals: a LEO node that would otherwise transmit on every ≈3–7 minute contact instead transmits roughly once per 30-minute window, an MEO node transmits roughly once per 6-hour window instead of on every regional contact, and a GEO/HEO node transmits roughly once per 24-hour window instead of on every available crosslink opportunity.

Pooling does not measurably affect the convergence and accuracy results of Section 10.2: AUROC under pooled, nominal-operation training reaches $0 . 8 8 8 \pm 0 . 0 0 9$ at round 47 versus $0 . 8 9 1 \pm 0 . 0 0 8$ for opportunistic transmission—a difference within one standard deviation—because the pooling intervals were chosen to remain inside each tier’s existing staleness budget $\tau _ { \mathrm { m a x } }$ (Section 10) rather than to relax it. Rounds-to-target lengthens marginally, from $4 7 \pm 4 \tan 5 { \overline { { 1 } } } \pm 5 ,$ reflecting the slightly coarser update cadence at MEO and GEO/HEO during early training. Time-to-detect for active fire events (Section 10.6) is unaffected, since the event-adaptive override collapses pooling to per-contact transmission before the staleness or latency budgets are touched; the energy savings in Table 4 therefore accrue almost entirely during the non-event majority of mission time, which dominates the 90-day energy budget of Section 10.4. Combining scheduled pooling with the SWaP-C-aware depth reduction already reported in Section 10.4 yields a projected 90-day mission energy budget approximately 54% lower than FedAvg-full at the same constellation scale, though we have not yet re-run the full 90-day degradation simulation under pooling and report this as a projection pending full re-simulation.

![](images/77ceac00844c156c2a4439f95e6469bd7f23c5f067a6bcf0144102369abbb4c0.jpg)

![](images/2e7a47ccd2fb42c3a6c28c5974d04f2dcbde0e2b6bbaaa768e9699bd2a0303cb.jpg)

![](images/a3991adca134cad359d9fc5d84ab150c36af471d899176844a21c8a34406cd33.jpg)  
Figure 7: Per-tier energy savings from scheduled pooling. Energy per node-day (mean ± one standard deviation over ten seeds) for all baselines and for FN-HFL with and without scheduled pooling, broken out by orbital tier. All baselines and the unpooled FN-HFL variant transmit at every contact window. Enabling scheduled pooling cuts FN-HFL’s energy footprint by a further 39–40% at every tier (annotated arrows) without measurably affecting convergence or event-response latency (Section 10.10).

## 11 Discussion and Limitations

FN-HFL is not a flight-certified system. The proposed paper must still resolve simulator realism, convergence proof tightness, fault-injection fidelity, and safety validation for autonomous agent actions. The wildfire scenario, while motivationally compelling, also highlights limitations that deserve first-class experimental attention. First, fire behavior is highly non-stationary: a fire that behaves predictably for several hours can undergo a rapid transition driven by wind shift or fuel change, potentially invalidating the representations that the model has learned. The scheduler’s staleness budget tightening during active events partially addresses this, but the model’s ability to adapt to rapid fire-regime change requires explicit robustness evaluation. Second, the depth assignment assumed in the wildfire scenario (LEO learns pixels, MEO learns fronts, GEO learns patterns) reflects a domain assumption that may not hold for all fire types or sensor configurations; an ablation should test whether the architecture degrades gracefully when this assumption is violated. Third, the agentic escalation pathway introduces new operational risks: an incorrectly escalated fire signal reaching GEO/HEO could trigger resource pre-positioning decisions with real operational consequences. The false escalation rate metric directly addresses this, but the threshold calibration between sensitivity and specificity is a policy question that requires ground-level input. Fourth, the scheduled pooling intervals of Section 6.4 were tuned for the wildfire scenario’s event-arrival statistics; a domain with more frequent low-priority anomalies could erode the energy benefit if the event-adaptive override triggers often enough to keep nodes near per-contact transmission, and the present results do not characterize that crossover point.

## 12 Conclusion

We presented FN-HFL, a fractal-native heterogeneous federated learning architecture for autonomous space computing. The key design move is to identify path depth with compute depth and with information depth: shallow satellites train early model prefixes that detect local, pixel-scale phenomena; medium satellites aggregate regional context and train mid-level representations of spatial propagation; and high-compute satellites maintain deep representations of multi-event systemic patterns that span the constellation’s full field of view. The wildfire scenario developed in Section 2 demonstrates that this depth-topology correspondence is not an engineering convenience but a principled reflection of what each tier of the constellation can observe, compute, and contribute. Layering tier-specific scheduled pooling (30 minutes at LEO, 6 hours at MEO, 24 hours at GEO/HEO) on top of this depth hierarchy further cuts per-tier energy consumption by roughly 39–40% relative to opportunistic transmission without measurably affecting convergence or event-response latency, since the event-adaptive override falls back to per-contact transmission whenever timeliness is at stake.

By co-designing topology, learning, scheduled communication cadence, and agentic scheduling, FN-HFL offers a path toward scalable on-orbit intelligence without assuming homogeneous satellites, real-time ground control, or application scenarios with static sensing requirements.

## References

[1] G. Larsson, M. Maire, and G. Shakhnarovich, “Fractalnet: Ultra-deep neural networks without residuals,” arXiv preprint arXiv:1605.07648, 2016.

[2] ——, “Fractalnet: Ultra-deep neural networks without residuals,” in ICLR, 2017.

[3] E. Diao, J. Ding, and V. Tarokh, “Heterofl: Computation and communication efficient federated learning for heterogeneous clients,” in ICLR, 2021.

[4] S. Horvath, S. Laskaridis, M. Almeida et al., “Fjord: Fair and accurate federated learning under heterogeneous targets with ordered dropout,” in NeurIPS, 2021.

[5] M. Kim, S. Park, S. Park, S. Yun, and S.-Y. Lee, “Depthfl: Depthwise federated learning for heterogeneous clients,” in ICLR, 2023.

[6] N. Razmi, B. Matthiesen, A. Dekorsy, and P. Popovski, “Ground-assisted federated learning in leo satellite constellations,” in IEEE Wireless Communications Letters, 2022.

[7] Y. Zhai et al., “Fedleo: An offloading-assisted decentralized federated learning framework for low earth orbit satellite networks,” IEEE Transactions on Mobile Computing, 2024.

[8] Anonymous, “Dfedsat: Communication-efficient and robust decentralized federated learning for leo satellite constellations,” arXiv preprint, 2024.

[9] ——, “Hisatfl: Hierarchical satellite federated learning with heterogeneous models,” arXiv preprint, 2025.

[10] P. Blanchard, E. M. El Mhamdi, R. Guerraoui, and J. Stainer, “Machine learning with adversaries: Byzantine tolerant gradient descent,” in NeurIPS, 2017.

[11] D. Yin, Y. Chen, R. Kannan, and P. Bartlett, “Byzantine-robust distributed learning: Towards optimal statistical rates,” in ICML, 2018.

[12] D. A. Vallado, P. Crawford, R. Hujsak, and T. Kelso, Revisiting Spacetrack Report #3. AIAA, 2006.