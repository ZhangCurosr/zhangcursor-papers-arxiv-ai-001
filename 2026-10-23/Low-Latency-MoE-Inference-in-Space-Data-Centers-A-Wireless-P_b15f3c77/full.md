# Low-Latency MoE Inference in Space Data Centers: A Wireless-Power-Transfer Perspective

Haotian Zheng, Zhanwei Wang, and Kaibin Huang, Fellow, IEEE

Abstract—Space data centers (SpaceDCs) are emerging as platforms for large language models (LLMs), supported by continuous solar-energy harvesting and high-speed interconnections. The mixture-of-experts (MoE) architecture supports distributed LLM inference by deploying experts across satellites. However, skewed expert activation creates uneven computation power demands despite uniform solar-power supply at the dawn– dusk sun-synchronous orbit (SSO). Consequently, active experts remain power-constrained while harvested power at idle experts is underused. These reduce inference efficiency and increase latency. To address this supply–demand mismatch, we propose the SpaceWPT framework, which uses inter-satellite laser-based wireless power transfer (WPT) to redistribute harvested power. The key design challenge is to balance the computation demands of power-donating and power-receiving satellites under WPT losses: transferring more power accelerates computation at the receiver but leaves less power for the donor. After deriving design principles that simplify WPT control, we characterize the resulting latency tradeoff through a tractable surrogate of average inference latency. Minimizing the surrogate yields a novel twosided water-filling strategy that determines whether each active expert donates, retains, or receives power according to the water level and its WPT link efficiency. Simulations with Mixtral-8x7B and Phi-3.5-MoE show up to 29.25% lower inference latency than without WPT, with larger gains under more compact formations and higher transfer efficiencies.

Index Terms—Space data centers, Wireless Power Transfer, Mixture of Experts, Power Control.

## I. INTRODUCTION

The sixth-generation (6G) mobile networks are envisioned to extend ubiquitous intelligence beyond the terrestrial edge into space [1]. Aligned with this vision, space data centers (SpaceDCs), which are built on tight-formation satellite fleets with every satellite carrying a solar array and on-board GPUs, are emerging as a key platform for Space AI [2], [3]. Deployment in a dawn–dusk sun-synchronous orbit (SSO) enables near continuous and spatially uniform solar-energy harvesting across satellites, while inter-satellite distances of only hundreds of meters support high-bandwidth, low-latency connections among their GPUs [4], [5]. Together, these features make SpaceDCs a scalable, solar-powered computing platform for distributed large language models (LLMs) [6]. Mixture-of-Experts (MoE) is a widely adopted architecture for LLMs, comprising multiple expert networks that can be deployed across satellites [7]. Within this architecture, a gating mechanism activates experts according to the input token, and the activated experts perform computation in parallel [8]. However, token-dependent expert activation creates heterogeneous power demands that mismatch the uniform solar-power supply across the SpaceDC. Under this supply–demand mismatch, satellites with higher power demand remain constrained by their local solar-power supply. This limits their computation speed and hence increases MoE inference latency. To tackle this challenge, we propose SpaceWPT, a system that explores wireless power transfer (WPT) in SpaceDCs to reallocate power among satellites to accelerate MoE inference.

Existing methods mitigate the supply-demand mismatch by redistributing power demand across nodes through model placement at deployment or expert routing during inference. Model-placement strategies determine the power demand by assigning model components to computing nodes [9]. Several works optimize MoE expert placement according to token workloads and activation patterns to balance processing loads and reduce inter-node communication [10], [11]. However, inter-satellite link conditions in SpaceDCs vary continuously, causing the placement optimized at deployment to gradually become suboptimal. Adjusting the placement to the varying links requires transferring billion-parameter experts between satellites, incurring high communication overhead [12]. Alternatively, expert routing instead keeps model placement fixed and reshapes the token-level power demand by determining which experts process each token. Existing methods balance expert loads by routing tokens to less-loaded experts when gate scores are comparable [13], or reduce communication overhead by replacing or skipping experts based on representation similarity and channel conditions [14]. Nevertheless, such operations alter the MoE model output and hence degrade inference accuracy. The proposed WPT approach has no such limitations. Rather than balancing power demand through workload redistribution, it adjusts the supply side by reallocating harvested solar power among satellites. This leaves expert placement and routing unchanged, avoiding the overhead of transferring expert parameters and preserving inference accuracy.

There exists a rich literature on the application of microwavebased WPT to power different types of terrestrial wireless networks, ranging from communication and sensor networks to edge computing networks. Relevant studies use downlink WPT to maximize uplink throughput [15] and deploy dedicated power beacons to optimize network deployment under outage constraints [16]. More recently, WPT has been integrated with edge computing and federated learning, where transferred energy powers local processing and wireless transmission to accelerate computation and communication at energyconstrained devices [17], [18], [19]. In contrast, laser-based

WPT in space operates under different scales of inter-node distances and power demands. Inter-satellite distances span hundreds of meters to kilometers, beyond the operating range of microwave-based WPT, while LLM inference requires substantially more power than low-power communication or sensing. These requirements favor laser-based WPT, whose low-divergence laser beams maintain high power density over long distances and whose narrow spectrum enables photovoltaic receivers to be optimized for high conversion efficiency [20], [21]. Moreover, operating in space avoids the safety constraints that limit high-power laser transmission in terrestrial environments [22]. Existing laser-based WPT studies investigate transferring surplus energy from large satellites to energydeficient CubeSats, small satellites used for communication, and mitigating pointing errors to improve harvested power [23], [24]. These works establish the feasibility of laser-based WPT in space, but they mainly focus on point-to-point single-link design. However, WPT for distributed MoE in space is more sophisticated as it requires network-level coordinated designs across multiple satellites to accommodate their activation-based heterogeneous power demands.

Designing network-level WPT for distributed MoE inference poses two unique challenges in terms of controlling transfer directions and power levels. First, transfer directions need to accommodate the uneven power demands induced by skewed expert activation. As expert activation varies from token to token, the locations of power shortages and surpluses change, requiring corresponding adjustments to inter-satellite WPT. Second, even with the transfer direction determined, the power level should be carefully controlled to balance the receiver’s power demand against the transmitter’s own computation needs. Increasing the transfer power accelerates receiver-side computation but leaves less power for computation at the transmitter, correspondingly increasing its latency. Furthermore, inter-satellite WPT incurs losses from electrical–optical conversion and beam spreading over the air, which complicates power control. Therefore, effective WPT control requires optimizing transfer directions and power levels according to the computation demands at both ends and the associated link efficiencies.

To address these challenges, we present SpaceWPT, a WPT assisted inference framework for low-latency token generation in SpaceDCs. SpaceWPT builds on the existing SpaceMoE architecture, where the attention and gating modules are deployed on a gateway satellite, while the expert modules are distributed across the remaining satellites [7]. Each token is processed by the attention module at the gateway and then by the activated experts in parallel, whose outputs are aggregated at the gateway. To the best of our knowledge, this is the first attempt to deploy WPT for LLM inference in space. The key contributions and findings of this work are summarized as follows.

• Attention–Expert Latency Tradeoff: We theoretically characterize the effect of WPT power control on E2E MoE inference latency. To this end, two basic design principles are derived to guide the control of transfer directions and power levels. First, it is optimal for a satellite to transfer all its harvested power when its expert is inactive. Second, it never transmits and receives power simultaneously when its expert is active. Based on these principles, we construct a tractable surrogate for the inference latency under stochastic expert activation. The resulting surrogate reveals an attention–expert latency tradeoff: power transferred by an active expert to support attention computation reduces attention latency while increasing the expert latency. Balancing this tradeoff to reduce the total inference latency necessitates the optimization of power control in the sequel.

• Activation-Based WPT Control: To balance the tradeoff between attention and expert latencies, we design a nearoptimal strategy for WPT power controls by minimizing the latency surrogate. The resulting solution exhibits a twosided water-filling structure. Compared with conventional water filling, which allocates power from a single source to multiple links, this structure allows each satellite to donate, retain, or receive power. In this structure, the water level is determined by expert activation statistics, the MoE inference workload, and WPT link efficiencies. Then, the transfer direction and power level of each satellite can be obtained by comparing its link efficiency with the water level. The solution shows that satellites with more efficient WPT links have higher power levels, while those with inefficient links tend to retain their harvested power. This strategy reduces the original high-dimensional problem to a single scalar equation for the water level while providing a provable near-optimality guarantee.

• Experiments: Experiments with Mixtral-8x7B and Phi-3.5-MoE on HumanEval confirm the derived attention– expert tradeoff and the substantial latency reductions of SpaceWPT over No-WPT baselines. The latency gain increases as WPT efficiency improves and the SpaceDC formation becomes more compact, highlighting the benefit of efficient short-range WPT.

The remainder of this paper is organized as follows. Section II presents the system model and performance metrics. Section III analyzes the inference latency and derives a tractable surrogate. Section IV develops the WPT power-control strategy, and Section V presents the experimental results.

## II. SYSTEM MODEL AND METRICS

## A. MoE Inference Model

1) MoE Architecture and Inference: We consider a decoderbased large language model whose L decoder layers employ the MoE architecture, illustrated in Fig. 1. Each decoder layer comprises three modules: a multi-head attention module shared by every token, a gating module that selects the active experts, and an expert module containing K expert feed-forward networks (FFNs). First, each token’s input is processed by the attention module, which requires $C ^ { \mathsf { a t t } }$ FLOPs. Second, a gating matrix $W _ { \mathrm { g } }$ maps the output of the attention module to a score vector. Third, κ experts with the highest gating scores are activated, forming the active expert set $\mathcal { A } ( \mathbf { s } ) \triangleq \{ k : s _ { k } = 1 \}$ where $s _ { k }$ is the binary activation indicator for expert k. The activated experts process the token in parallel, each requiring $C ^ { \mathsf { f f n } }$ FLOPs. The gateway then aggregates their outputs using the normalized gating weights to obtain the output of the current MoE layer.

![](images/be313fd87ff71f5fdb6359dfb5fc6bb48b3a8946dbe7da21a03aad3f513d4eac.jpg)  
Fig. 1. MoE inference architecture and its deployment.

2) Expert Activation Model: To quantify the activation distribution, we adopt a classical probability-proportional-to size without replacement (PPSWOR) model for sampling $\kappa$ experts out of K candidates [25]. This theory associates each expert k with an importance weight $\omega _ { k } ~ > ~ 0$ . The resulting probability mass function (PMF) of top-κ experts is given by

$$
\operatorname* { P r } \{ \mathcal { A } ( \mathbf { s } ) = \mathcal { U } \} = \frac { \prod _ { k \in \mathcal { U } } \omega _ { k } } { e _ { \kappa } ( \omega _ { 1 } , \dots , \omega _ { K } ) } , \qquad | \mathcal { U } | = \kappa ,\tag{1}
$$

where ${ \mathcal { U } } \subseteq \{ 1 , \dotsc , K \}$ and $e _ { \kappa }$ is the κ-th elementary symmetric polynomial,

$$
e _ { \kappa } ( \omega _ { 1 } , \dots , \omega _ { K } ) \triangleq \sum _ { \begin{array} { l } { \mathcal { U } \subseteq \{ 1 , \dots , K \} } \\ { | \mathcal { U } | = \kappa } \end{array} } \prod _ { k \in \mathcal { U } } \omega _ { k } .
$$

Note that the per-expert activation probability of this model is $\alpha _ { k } \triangleq \operatorname* { P r } \{ k \in \mathcal { A } ( \mathbf { s } ) \} = \mathbb { E } [ s _ { k } ]$ , which is a monotone increasing function of $\omega _ { k }$ and can be computed by

$$
\alpha _ { k } = 1 - \frac { e _ { \kappa } ( \omega _ { 1 } , \ldots , \hat { \omega } _ { k } , \ldots , \omega _ { K } ) } { e _ { \kappa } ( \omega _ { 1 } , \ldots , \omega _ { K } ) } ,\tag{2}
$$

where $\hat { \omega } _ { k }$ denotes omission of $\omega _ { k }$ .

3) Computation Latency Model: We model the computation latency at each processor under dynamic voltage and frequency scaling (DVFS). The relation between the computation speed and the allocated compute power $P ^ { \mathsf { c m p } }$ is modeled as [26]

$$
f = \left( \frac { P ^ { \mathsf { c m p } } } { \gamma } \right) ^ { 1 / 3 } ,\tag{3}
$$

where $f$ is measured in FLOPs per second and $\gamma > 0$ is an architecture-dependent coefficient. The resulting latency of executing a workload of C FLOPs on such a processor is given by

$$
T ^ { \mathsf { c m p } } = C \left( \frac { \gamma } { P ^ { \mathsf { c m p } } } \right) ^ { 1 / 3 } .\tag{4}
$$

## B. Modelling Space Data Center

1) Expert-to-Satellite Mapping: We consider a SpaceDC comprising a formation of $K + 1$ satellites that jointly execute the MoE model. For each MoE layer, the attention and gating modules are deployed on one satellite, designated as the gateway satellite and indexed by 0. Each of the remaining

K satellites hosts one expert FFN and is referred to as an expert satellite. Under this deployment, each expert and its host satellite share the index $k \in { \mathcal { K } } \triangleq \{ 1 , \dots , K \}$

2) Geometry of Space Data Center: Following the orbital design recently proposed in [5], the SpaceDC illustrated in Fig. 2 operates as a formation-flying cluster in a dawn–dusk SSO. Since this orbit approximately follows Earth’s day–night terminator, it provides near-constant solar illumination [4]. We model each satellite’s harvested solar power as a constant budget P<sup>s</sup>.

To describe the cluster geometry, we adopt a local-verticallocal-horizontal (LVLH) coordinate system that co-rotates with the gateway satellite, as illustrated in Fig. 2: the xaxis points radially outward from the Earth, and the y-axis aligns with the gateway satellite’s instantaneous velocity. In this coordinate system, let $( x _ { k } ( t ) , y _ { k } ( t ) )$ denote the position of expert satellite k relative to the gateway satellite. The relative motion is governed by the Hill-Clohessy-Wiltshire (HCW) equations [27]

$$
\ddot { x } _ { k } - 2 \omega \dot { y } _ { k } - 3 \omega ^ { 2 } x _ { k } = 0 , \quad \ddot { y } _ { k } + 2 \omega \dot { x } _ { k } = 0 ,\tag{5}
$$

where $\omega$ denotes the orbital angular speed of the gateway satellite. When the initial radial offset and along-track speed satisfy $\dot { y } _ { k } ( 0 ) = - 2 \omega x _ { k } ( 0 )$ , expert satellite k stays on a closed relative orbit around the gateway instead of drifting away. The solutions of (5) admit the closed form, given by

$$
\begin{array} { r l } & { x _ { k } ( t ) = \rho _ { k } \cos ( \omega t + \phi _ { k } ) , } \\ & { y _ { k } ( t ) = - 2 \rho _ { k } \sin ( \omega t + \phi _ { k } ) + y _ { k } ^ { \mathrm { o f f } } , } \end{array}\tag{6}
$$

where $\rho _ { k }$ sets the size of the relative orbit, $\phi _ { k }$ sets the starting position on this orbit, and $y _ { k } ^ { \mathsf { o f f } }$ is the satellite’s average alongtrack offset from the gateway. Geometrically, (6) traces a closed 2:1 ellipse centered on (0, $y _ { k } ^ { \mathsf { o f f } } )$ with period $T _ { \mathrm { o r b } } = 2 \pi / \omega .$ Thus, expert satellite $k ' \mathrm { s }$ distance from the gateway satellite is

$$
d _ { k } ( t ) = { \sqrt { x _ { k } ^ { 2 } ( t ) + y _ { k } ^ { 2 } ( t ) } } ,\tag{7}
$$

which varies on the orbital timescale $T _ { \mathrm { o r b } } \sim 1 0 ^ { 3 } ~ \Omega$ s. In contrast, each token inference latency is on the order of milliseconds, six orders of magnitude shorter than $T _ { \mathrm { o r b } } \ [ 2 8 ]$ . Therefore, the cluster geometry is approximately fixed within any single inference pass, and we treat $\{ d _ { k } ( t ) \}$ as constants $\{ d _ { k } \}$ in the remainder of this paper. The resulting gateway-to-expert distances $\{ d _ { k } \}$ determine the link-specific communication and WPT characteristics developed in the following two subsections.

## C. Communication Model

Inter-satellite data exchange employs free-space optical (FSO) links operating in a diffraction-limited regime [29], [30]. For a link with transmit and receive aperture diameters $D _ { \mathrm { t x } } ^ { \mathrm { c } }$ and $D _ { \sf r \times } ^ { \sf c }$ at wavelength λ<sup>c</sup>, the channel power gain between two nodes separated by distance d is

$$
h ( d ) = \eta ^ { \mathsf { c } } \biggl ( \frac { \pi D _ { \mathsf { t x } } ^ { \mathsf { c } } D _ { \mathsf { r x } } ^ { \mathsf { c } } } { 4 \lambda ^ { \mathsf { c } } d } \biggr ) ^ { 2 } ,\tag{8}
$$

where $\eta ^ { \mathsf { c } } \in \mathsf { ( 0 , 1 ] }$ absorbs pointing errors and optical-path losses. Applying this model to gateway–expert link k at distance $d _ { k }$ gives

$$
h _ { k } \triangleq h ( d _ { k } ) .
$$

![](images/3372634ec3b72a8fba850ae6236352530cc9612313e37bc58fa4cda6aca7b08a.jpg)  
Fig. 2. SpaceDCs in a dawn–dusk orbit and the relative-orbit geometry of their satellite clusters.

Hence, different gateway-to-expert distances result in heterogeneous communication-channel gains across the SpaceDC. With transmit power $P ^ { \mathsf { c m m } }$ , noise spectral density $N _ { 0 }$ , and link bandwidth $B ,$ the achievable data rate over the FSO link is [31]

$$
R ( d ) = B \log _ { 2 } \left( 1 + \frac { P ^ { \mathsf { c m m } } h ( d ) } { N _ { 0 } B } \right) .\tag{9}
$$

Transmitting $Q$ bits over a link of distance d takes a one-way latency

$$
T ^ { \mathsf { c m m } } ( d , Q ) = { \frac { Q } { R ( d ) } } + { \frac { d } { c } } ,\tag{10}
$$

where $c$ is the speed of light and the second term is the propagation delay.

## D. Modelling Wireless-Power-Transfer

1) Laser-Based WPT Model: We adopt laser-based WPT [20], [23] for inter-satellite power transmission in the cluster. Each satellite carries a dedicated high-power laser as the WPT transmitter, together with an on-board photovoltaic (PV) receiver that converts incident laser irradiance into electrical power [21], [32]. Any satellite can therefore act as a WPT source for any other, over an optical channel separate from the data link.

Following the same diffraction-limited aperture-coupling law as in (8), the optical channel power gain at link distance d is

$$
g ( d ) = \eta ^ { \mathrm { w } } \bigg ( \frac { \pi D _ { \mathrm { t x } } ^ { \mathrm { w } } D _ { \mathrm { r x } } ^ { \mathrm { w } } } { 4 \lambda ^ { \mathrm { w } } d } \bigg ) ^ { 2 } ,\tag{11}
$$

where $\lambda ^ { \mathsf { w } } , D _ { \mathtt { t x } } ^ { \mathsf { w } }$ , and $D _ { \mathsf { r } \times } ^ { \mathsf { w } }$ are the laser wavelength and transmit/receive aperture diameters, and $\eta ^ { \mathsf { w } } \in ( 0 , 1 ]$ absorbs optical path losses. WPT energy traverses three cascaded conversion stages: an electrical-to-optical (E/O) stage at the transmit laser with wall-plug efficiency $\eta _ { \mathsf { e o } } ~ \in ~ ( 0 , 1 ]$ , the optical channel in (11), and an optical-to-electrical (O/E) stage at the PV receiver with conversion efficiency $\eta _ { \mathsf { o e } } \in ( 0 , 1 ]$ . The cascade yields the end-to-end electrical-to-electrical transfer efficiency

$$
\eta ( d ) \triangleq \eta _ { \mathsf { e o } } g ( d ) \eta _ { \mathsf { o e } } .\tag{12}
$$

For gateway–expert link k at distance $d _ { k }$ , the corresponding WPT efficiency is

$$
\eta _ { k } \triangleq \eta ( d _ { k } ) \in ( 0 , 1 ) .\tag{13}
$$

Hence, different gateway-to-expert distances result in heterogeneous WPT efficiencies across the SpaceDC. Let P denote the electrical input power to the transmit laser. Accordingly, the harvested electrical power at the receiver is then $\eta ( d ) P _ { \bf \Phi }$ When this link is directed from node i to node $j ,$ we write the transmit power as $P _ { i \to j }$ , the convention used throughout the remainder of the paper.

2) WPT Topology: We consider the star WPT topology illustrated in Fig. 3(a), in which each expert exchanges power only with the gateway. At every node, the available compute power equals the local solar-power budget plus the received WPT power minus the transmitted WPT power. The gateway aggregates contributions from all K experts, given by

$$
P _ { 0 } ^ { \mathsf { c m p } } = P ^ { \mathsf { s } } + \sum _ { k = 1 } ^ { K } \eta _ { k } P _ { k \to 0 } - \sum _ { k = 1 } ^ { K } P _ { 0 \to k } ,\tag{14}
$$

while each expert k couples only with the gateway, given by

$$
P _ { k } ^ { \mathsf { c m p } } = P ^ { \mathsf { s } } + \eta _ { k } P _ { 0 \to k } - P _ { k \to 0 } .\tag{15}
$$

Each WPT transmit power is non-negative and capped by the local solar-power budget, given by

$$
0 \leq P _ { k  0 } \leq P ^ { \mathrm { s } } , \qquad 0 \leq P _ { 0  k } \leq P ^ { \mathrm { s } } .\tag{16}
$$

## E. System Operations

We adopt expert parallelism for distributed MoE inference in the SpaceDC [33]. Each round begins with a gating decision at the gateway and ends before the subsequent gating decision. The operations within one round are illustrated in Fig. 3(b) and described as follows.

1) Gating: Given the attention output from the preceding round, the gateway performs the gating operation described in Section II-A, yielding the activation state s and the active expert set $\mathcal { A } ( \mathbf { s } )$ defined in Section II-A1.

2) WPT Power Control: Based on s, the gateway configures the WPT transmit powers using the power-control solution derived in Section IV, over the star topology described in Section II-D2.

3) Activation Transmission: The gateway transmits the input activation to the active experts over the FSO data links modeled in Section II-C.

4) Expert Computation: Each active expert processes the received activation using its onboard FFN, following the MoE inference model in Section II-A.

5) Token Aggregation: Each active expert returns its FFN output to the gateway over the FSO data link. The gateway aggregates the received outputs to obtain the layer output.

6) Attention Computation: The gateway computes the atten tion output from the aggregated expert output and passes it to the gating module, thereby initiating the next round.

During Steps 3–5, the active-expert paths operate in parallel, where each expert starts its FFN computation upon receiving the activation and returns its output upon completion. The WPT powers configured in Step 2 are held fixed throughout Steps 3– 6. Separate optical links allow WPT to operate concurrently with data communication.

![](images/06c0035f0c8fe34c4be0a494b029a0ad26fa87396ef77e2dae668250de137709.jpg)  
Fig. 3. SpaceWPT system and operations.

## F. Performance Metrics

1) Inference Accuracy: The inference accuracy of the MoE model depends on its gating matrix and expert FFNs, both of which remain fixed after deployment. The WPT power allocation changes only the compute power available at each node. Inference accuracy is therefore independent of the WPT power allocation.

2) Inference Latency: We therefore adopt the per-token inference latency as the performance metric. The gating and WPT-control signaling in Steps 1–2 incur negligible latency and are omitted.

The inference latency therefore comprises the parallel expert phase in Steps 3–5 and the gateway-side attention computation in Step 6. Let $Q ^ { \mathrm { d } | }$ and $Q ^ { \mathrm { u l } }$ denote the inputactivation and expert-output sizes, respectively. Following the FSO communication model in Section II-C, the communication latency along expert k’s path is

$$
T _ { k } ^ { \mathsf { c m m } } \triangleq T ^ { \mathsf { c m m } } ( d _ { k } , Q ^ { \mathsf { d l } } ) + T ^ { \mathsf { c m m } } ( d _ { k } , Q ^ { \mathsf { u l } } ) .\tag{17}
$$

Combining this communication latency with the computation model in Section II-A and the parallel execution of Steps 3–5 gives the expert-phase latency

$$
T ^ { \mathrm { e x p } } ( \{ P _ { k } ^ { \mathrm { c m p } } \} ; \mathbf { s } ) = \operatorname* { m a x } _ { 1 \leq k \leq K } s _ { k } \left[ T _ { k } ^ { \mathrm { c m m } } + C ^ { \mathsf { f f n } } \left( \frac { \gamma } { P _ { k } ^ { \mathrm { c m p } } } \right) ^ { \frac { 1 } { 3 } } \right] ,\tag{18}
$$

where $P _ { k } ^ { \mathsf { c m p } }$ is the compute power available at expert k, as given in (15). After token aggregation, the gateway performs the attention computation in Step 6. Following the computation model in Section II-A, its latency is

$$
T ^ { \mathsf { a t t } } ( P _ { 0 } ^ { \mathsf { c m p } } ) = C ^ { \mathsf { a t t } } \left( \frac { \gamma } { P _ { 0 } ^ { \mathsf { c m p } } } \right) ^ { 1 / 3 } ,\tag{19}
$$

where $P _ { 0 } ^ { \mathsf { c m p } }$ is the compute power available at the gateway, as given in (14). The gateway and expert compute powers are collected in the vector $\mathbf { P } ^ { \mathsf { c m p } } \triangleq ( P _ { 0 } ^ { \mathsf { c m p } } , P _ { 1 } ^ { \mathsf { c m p } } , \cdot \cdot \cdot , \hat { P } _ { K } ^ { \mathsf { c m p } } )$ . The per-token inference latency is given by

$$
T ( { \bf P } ^ { \mathsf { c m p } } ; { \bf s } ) = T ^ { \mathsf { e x p } } ( \{ P _ { k } ^ { \mathsf { c m p } } \} ; { \bf s } ) + T ^ { \mathsf { a t t } } ( P _ { 0 } ^ { \mathsf { c m p } } ) ,\tag{20}
$$

The optimization objective is the average per-token latency over the activation model in (1), given by

$$
\bar { T } ( \mathbf { P } ^ { \mathsf { c m p } } ) \triangleq \mathbb { E } _ { \mathbf { s } } [ T ( \mathbf { P } ^ { \mathsf { c m p } } ; \mathbf { s } ) ] .\tag{21}
$$

## III. INFERENCE LATENCY ANALYSIS

In this section, we formulate the inference latency through the lens of the WPT power control and then derive a tractable surrogate for its average. The surrogate reveals the power– latency tradeoff between the gateway and the experts, which is exploited in the next section to optimize the WPT control.

## A. Latency-Minimizing Problem

In this subsection, we analyze the per-token inference latency as a function of the WPT power and formulate the latency-minimizing problem. The compute powers in (20) are determined by the local solar-power budget and WPT transfers through (14) and (15). However, each expert power affects the latency only through the expert latency term (18), so increasing it reduces the latency only when expert k is active. Since the relation between the powers and the latency depends on the activation state, we allow each expert to set its WPT powers separately for its active and idle states, giving the statedependent functions $P _ { k \to 0 } ( s _ { k } )$ and $P _ { 0 \to k } ( s _ { k } )$ for $s _ { k } \in \{ 0 , 1 \}$ Thus, each expert carries four optimization variables, a donating and a receiving power in each of the two activation states, which are optimized to minimize the average latency.

We then formulate the latency-minimizing problem over the state-dependent WPT powers. Stacking them across all experts and activation states gives the optimization vector

$$
\mathbf { q } \triangleq \{ P _ { k  0 } ( s _ { k } ) , P _ { 0  k } \big ( s _ { k } \big ) \} _ { k = 1 , \ldots , K ; s _ { k } \in \{ 0 , 1 \} } ,\tag{22}
$$

which induces $\mathbf { P } ^ { \mathsf { c m p } } ( \mathbf { q } ; \mathbf { s } )$ through (14) and (15) for each activation state s. Accordingly, the per-token latency induced by q is

$$
\begin{array} { r } { T ( \mathbf { q } ; \mathbf { s } ) \triangleq T ( \mathbf { P } ^ { \mathsf { c m p } } ( \mathbf { q } ; \mathbf { s } ) ; \mathbf { s } ) . } \end{array}
$$

Substituting this induced latency into (21) gives

$$
\bar { T } ( \mathbf { q } ) \triangleq \mathbb { E } _ { \mathbf { s } } [ T ( \mathbf { q } ; \mathbf { s } ) ] .
$$

Thus, the latency-minimizing problem is

$$
\begin{array} { r l } { ( \mathcal { P } _ { 0 } ) : \quad \underset { \mathbf { q } } { \mathrm { m i n } } } & { \bar { T } ( \mathbf { q } ) } \\ & { \mathrm { s . t . } \quad 0 \leq P _ { k  0 } ( s _ { k } ) , P _ { 0  k } ( s _ { k } ) \leq P ^ { \mathrm { s } } , } \\ & { P _ { j } ^ { \mathrm { c m p } } > 0 , \quad j \in \{ 0 \} \cup \mathcal { A } ( \mathbf { s } ) , } \\ & { k = 1 , \dots , K , s _ { k } \in \{ 0 , 1 \} . } \end{array}
$$

Problem $( \mathcal { P } _ { 0 } )$ is intractable for two reasons. First, the optimization involves 4K variables in (22). Second, the expectation of the per-token latency over the random activation state s has no closed-form expression. We address the first challenge by exploiting the two design principles established in Section III-B to simplify the problem in Section III-C, and the second by deriving a tractable surrogate in Section III-D.

## B. Design Principles for Optimal WPT Control

In this subsection, we establish two design principles for optimal WPT control that reduce the number of optimization variables in problem $( \mathcal { P } _ { 0 } )$

First, consider an idle-state expert, where it runs no FFN, so its WPT power can serve only the gateway. The optimal solution is then to donate its full solar-power budget to the gateway, as established in Lemma 1.

Lemma 1 (Optimal idle-state powers). Every optimal power control satisfies

$$
P _ { k  0 } ^ { \star } ( s _ { k } { = } 0 ) = P ^ { \acute { s } } , \quad P _ { 0  k } ^ { \star } ( s _ { k } { = } 0 ) = 0 , \quad k = 1 , \ldots , K .\tag{23}
$$

Proof. An idle expert runs no FFN, so its powers affect only the gateway power $P _ { 0 } ^ { \mathsf { c m p } }$ and leave the expert-side latency unchanged. Since $P _ { 0 } ^ { \mathsf { c m p } }$ increases in $P _ { k \to 0 } ( s _ { k } { = } 0 )$ and decreases in $P _ { 0 \to k } ( s _ { k } { = } 0 )$ , while a larger $P _ { 0 } ^ { \mathsf { c m p } }$ always lowers the gateway latency, the optimum sets $P _ { k  0 } ( s _ { k } { = } 0 ) = P ^ { s }$ and $P _ { 0  k } ( s _ { k } { = } 0 ) = 0 .$ ■

Second, consider an active-state expert, where it may both send power to and receive power from the gateway. Transmitting in both directions at once wastes power on the round trip, so an optimal expert keeps only one direction and sets the other to zero, as established in Lemma 2.

Lemma 2 (No simultaneous counter-flow). Every optimal power control satisfies

$$
P _ { k \to 0 } ^ { \star } ( s _ { k } { = } 1 ) \cdot P _ { 0 {  } k } ^ { \star } ( s _ { k } { = } 1 ) = 0 , \qquad k = 1 , \ldots , K .\tag{24}
$$

Proof. Suppose an optimal control has $P _ { k \to 0 } ( s _ { k } { = } 1 ) ~ > ~ 0$ and $P _ { 0 \to k } ( s _ { k } { = } 1 ) ~ > ~ 0$ for some expert k. Reducing the two powers to $P _ { k \to 0 } ( s _ { k } { = } 1 ) ~ - ~ \eta _ { k } \varepsilon$ and $P _ { 0 \to k } ( s _ { k } { = } 1 ) \ : - \ : \varepsilon$ with $\varepsilon \ = \ \operatorname* { m i n } \{ P _ { 0 \to k } ( s _ { k } { = } 1 ) , \ : P _ { k { \to } 0 } ( s _ { k } { = } 1 ) / \eta _ { k } \}$ drives one direction to zero while leaving the expert compute power unchanged. This cuts the power the gateway sends out $\boldsymbol { \mathbf { b } } \mathbf { y } \ \varepsilon$ but the power it collects from the expert by only $\eta _ { k } ^ { 2 } \varepsilon$ , so $P _ { 0 } ^ { \mathsf { \bar { c } m p } }$ gains $\varepsilon ( 1 - \eta _ { k } ^ { 2 } ) > 0$ . Hence, the per-token latency strictly decreases, contradicting optimality. ■

The two design principles simplify WPT control by fixing idle-state donation and eliminating simultaneous transmission and reception in the active state. Since Lemma 2 precludes simultaneous counter-flow, the active-state power exchange can be represented without loss of optimality by a single signed scalar [34]

$$
P _ { k } \triangleq P _ { k \to 0 } ( s _ { k } { = } 1 ) - \eta _ { k } P _ { 0 { \to } k } ( s _ { k } { = } 1 ) , \quad k = 1 , { \dots } , K ,\tag{25}
$$

The cases $P _ { k } > 0 , P _ { k } = 0$ , and $P _ { k } < 0$ correspond to donation, neutrality, and reception, respectively. In the reception case,

$| P _ { k } | = \eta _ { k } P _ { 0  k } ( s _ { k } { = } 1 )$ is the power delivered to expert k after transfer loss.

## C. Inference Latency under Simplified WPT Control

With WPT control simplified due to the preceding design principles, we reformulate the inference latency and problem $( \mathcal { P } _ { 0 } )$ in terms of the signed-power vector $\mathbf { p } = ( P _ { 1 } , \dots , P _ { K } )$

Firstly, we explore the relation between $P _ { k }$ and the expert compute power, which affects the expert-side latency in (18). Since idle experts run no FFN, the expert-side latency depends only on active experts. For active states $s _ { k } = 1$ , the compute power of expert k is obtained from (15) as

$$
P _ { k } ^ { \mathrm { c m p } } ( s _ { k } { = } 1 ) = P ^ { \mathrm { s } } + \eta _ { k } P _ { 0 {  } k } ( s _ { k } { = } 1 ) - P _ { k {  } 0 } ( s _ { k } { = } 1 ) ,\tag{26}
$$

where $P _ { k \to 0 } ( s _ { k } { = } 1 )$ and $P _ { 0 \to k } ( s _ { k } { = } 1 )$ can be expressed in terms of $P _ { k }$ as

$$
P _ { k \to 0 } ( s _ { k } { = } 1 ) = ( P _ { k } ) ^ { + } , \quad P _ { 0 { \to } k } ( s _ { k } { = } 1 ) = \frac { ( - P _ { k } ) ^ { + } } { \eta _ { k } } ,\tag{27}
$$

with $( z ) ^ { + } \triangleq \operatorname* { m a x } \{ z , 0 \}$ . Substituting (27) into (26) expresses this compute power through $P _ { k }$ as

$$
P _ { k } ^ { \mathsf { c m p } } ( s _ { k } { = } 1 ) = P ^ { s } - P _ { k } , \qquad k = 1 , \ldots , K ,\tag{28}
$$

Thus, the expert-phase latency (18) can be expressed through p, given by

$$
T ^ { \mathrm { e x p } } ( { \bf p } ; { \bf s } ) = \operatorname* { m a x } _ { 1 \leq k \leq K } \left\{ s _ { k } \left[ T _ { k } ^ { \mathrm { c m m } } + \mu ^ { \mathrm { f f n } } ( P ^ { \mathrm { s } } - P _ { k } ) ^ { - \frac 1 3 } \right] \right\} ,\tag{29}
$$

where $\mu ^ { \mathsf { f f n } } \triangleq C ^ { \mathsf { f f n } } \gamma ^ { 1 / 3 }$ sets an expert’s FFN compute latency at a given compute power.

Secondly, we explore the relation between p and the gateway compute power, which affects the gateway-side latency in (19). Unlike the expert side, the gateway exchanges power with every expert in both activation states, so its compute power aggregates over all experts, expressed as

$$
P _ { 0 } ^ { \mathsf { c m p } } ( \mathbf { s } ) = P ^ { \mathsf { s } } + \sum _ { k = 1 } ^ { K } \bigl [ \eta _ { k } P _ { k \to 0 } ( s _ { k } ) - P _ { 0 \to k } ( s _ { k } ) \bigr ] ,\tag{30}
$$

where these transmit powers combine the active-state transfer (27) with the idle-state donation $P ^ { s }$ of Lemma 1, given by

$$
\begin{array} { l } { { P _ { k  0 } ( s _ { k } ) = s _ { k } ( P _ { k } ) ^ { + } + ( 1 - s _ { k } ) P ^ { \mathrm { s } } , } } \\ { { \displaystyle P _ { 0  k } ( s _ { k } ) = s _ { k } \frac { ( - P _ { k } ) ^ { + } } { \eta _ { k } } . } } \end{array}\tag{31}
$$

Substituting (31) into the gateway compute power (30), the contribution of an active expert depends on $P _ { k }$ and is given by

$$
\eta _ { k } P _ { k \to 0 } ( s _ { k } { = } 1 ) - P _ { 0 { \to } k } ( s _ { k } { = } 1 ) = \eta _ { k } ( P _ { k } ) ^ { + } - \frac { ( - P _ { k } ) ^ { + } } { \eta _ { k } } .
$$

Since $( P _ { k } ) ^ { + }$ and $( - P _ { k } ) ^ { + }$ cannot both be nonzero, the expression reduces according to the sign of $P _ { k }$ to the piecewise function $g _ { k } ( P _ { k } )$ , written as

$$
g _ { k } ( P _ { k } ) \triangleq \eta _ { k } ( P _ { k } ) ^ { + } - \frac { ( - P _ { k } ) ^ { + } } { \eta _ { k } } = \left\{ \eta _ { k } P _ { k } , \quad P _ { k } \ge 0 , \right.\tag{32}
$$

so a donating expert $( P _ { k } \ge 0 )$ contributes its harvested power $\eta _ { k } P _ { k }$ to the gateway, whereas a receiving one $( P _ { k } < 0 )$ draws $| P _ { k } | / \eta _ { k }$ from the gateway. Summing over donated and received power from all experts yields the gateway compute power through p, given by

$$
P _ { 0 } ^ { \mathsf { c m p } } ( \mathbf { p } ; \mathbf { s } ) = P ^ { s } + \sum _ { k = 1 } ^ { K } \bigl [ ( 1 - s _ { k } ) \eta _ { k } P ^ { s } + s _ { k } g _ { k } ( P _ { k } ) \bigr ] ,\tag{33}
$$

Substituting (33) into the attention latency (19) then expresses the gateway-side latency through p, given by

$$
T ^ { \mathrm { a t t } } ( { \bf p } ; { \bf s } ) = \mu ^ { \mathrm { a t t } } \left( P ^ { \mathrm { s } } + \sum _ { k = 1 } ^ { K } [ ( 1 - s _ { k } ) \eta _ { k } P ^ { \mathrm { s } } + s _ { k } g _ { k } ( P _ { k } ) ] \right) ^ { - \frac { 1 } { 3 } }\tag{34}
$$

where $\mu ^ { \mathrm { a t t } } \triangleq C ^ { \mathsf { a t t } } \gamma ^ { 1 / 3 }$ sets the attention compute latency at a given gateway compute power.

Combining the expert-side and gateway-side latencies, the per-token inference latency (20) becomes

$$
T ( \mathbf { p } ; \mathbf { s } ) = T ^ { \mathsf { a t t } } ( \mathbf { p } ; \mathbf { s } ) + T ^ { \mathsf { e x p } } ( \mathbf { p } ; \mathbf { s } ) ,\tag{35}
$$

Thus, the per-token inference latency (35) is fully determined by the reduced variable $\mathbf { p } ,$ which has only K dimensions, one per expert–gateway link. Since the activation pattern s is random, we focus on minimizing the average inference latency $\bar { T } ( \mathbf { p } ) \triangleq \mathbb { E } _ { \mathbf { s } } [ T ( \mathbf { p } ; \mathbf { s } ) ]$ . Under the sufficient condition in (49), the reduced problem is given by

$$
\begin{array} { r l } { ( \mathcal { P } _ { 1 } ) : \quad \underset { \mathbf { p } } { \mathrm { m i n } } } & { \mathbb { E } _ { \mathbf { s } } [ T ( \mathbf { p } ; \mathbf { s } ) ] } \\ & { \mathrm { s . t . } \quad - \eta _ { k } P ^ { \mathsf { s } } \leq P _ { k } < P ^ { \mathsf { s } } , \quad k = 1 , \ldots , K } \end{array}
$$

The constraint follows respectively from the transfer power caps in (16) and from keeping the expert compute power $P _ { k } ^ { \mathsf { \bar { c m p } } } = P ^ { \mathsf { s } } - P _ { k }$ positive.

## D. Tractable Surrogate of Inference Latency

After reducing the complexity of the optimization, we now address the second challenge within the intractable problem $( \mathcal { P } _ { 0 } )$ , resulting in a surrogate function for the average latency. Firstly, rather than averaging the gateway-side latency over random activations directly, we first average the gateway power and then compute the latency once from this average $\bf \bar { E } _ { s } [ \it { P } _ { 0 } ^ { c m p } ]$ . Secondly, since the expert-side latency is determined by the slowest active expert, we approximate it by the average latency over the active experts instead of this maximum. The resulting surrogate function of the average latency is given in Proposition 1.

Proposition 1 (Surrogate of inference latency). The average per-token latency ${ \bar { T } } ( \mathbf { p } )$ is approximated by the surrogate function, given by

$$
\begin{array} { l } { { \displaystyle \tilde { T } ( \mathbf { p } ) \triangleq \mu ^ { \mathrm { a t t } } \Big ( P ^ { \mathrm { s } } + \sum _ { k = 1 } ^ { K } \big [ ( 1 - \alpha _ { k } ) \eta _ { k } P ^ { \mathrm { s } } + \alpha _ { k } g _ { k } ( P _ { k } ) \big ] \Big ) ^ { - \frac { 1 } { 3 } } } } \\ { { \displaystyle ~ + \sum _ { k = 1 } ^ { K } \frac { \alpha _ { k } } { \kappa } \Big [ T _ { k } ^ { \mathrm { c m m } } + \mu ^ { \mathrm { f f n } } ( P ^ { \mathrm { s } } - P _ { k } ) ^ { - \frac { 1 } { 3 } } \Big ] . } } \end{array}\tag{36}
$$

Proof. See Appendix VII-A.

![](images/0e640861a5e0dda9931339d80ac82c8553b279a0c8389b8ffd86e670cfa9ead2.jpg)  
Fig. 4. Comparison of the per-token inference latency and its surrogate, illustrating the gateway–expert latency tradeoff.

The surrogate comprises a gateway-side and an expert-side term, each monotonic in $P _ { k }$ but in opposite directions. The first term in (36) represents the gateway attention latency and decreases with $P _ { k }$ . The second term represents the weighted average expert latency and increases with $P _ { k }$ through expert $k ' s$ contribution. The resulting tradeoff is illustrated in Fig. 4 and clarified in the following Remark 1.

Remark 1 (Gateway-expert tradeoff). The opposing monotonicities of the two terms in (36) reveal a gateway-expert tradeoff carried by each $P _ { k }$ . In particular, raising $P _ { k }$ enlarges the gateway compute power and shortens the gateway-side latency, but it simultaneously shrinks expert $k ' s$ compute power and lengthens its expert-side latency.

Therefore, the optimal power control should balance the tradeoff and can be obtained by solving the following problem.

$$
\begin{array} { r l } { ( \mathcal { P } _ { 2 } ) : \quad \underset { \mathbf { p } } { \mathrm { m i n } } \quad \tilde { T } ( \mathbf { p } ) } & { } \\ { \mathrm { s . t . } \quad } & { - \eta _ { k } P ^ { \mathrm { s } } \leq P _ { k } < P ^ { \mathrm { s } } , \quad k = 1 , \ldots , K , } \\ & { \quad \quad \quad P ^ { \mathrm { s } } + \sum _ { k = 1 } ^ { K } \bigl [ ( 1 - \alpha _ { k } ) \eta _ { k } P ^ { \mathrm { s } } + \alpha _ { k } g _ { k } ( P _ { k } ) \bigr ] > 0 , } \end{array}
$$

where the last constraint keeps the gateway compute power positive. Since the bounds $- \eta _ { k } P ^ { s } \le P _ { k } < P ^ { s }$ are intervals and the gateway-power constraint keeps a concave function positive (each $g _ { k }$ is concave by (32)), the feasible set of problem $\left( \mathcal { P } _ { 2 } \right)$ is convex [35]. Meanwhile, the following Lemma 3 establishes the convexity of the objective $\tilde { T } .$ , proving that problem $\left( \mathcal { P } _ { 2 } \right)$ is a convex problem.

Lemma 3 (Convexity of the surrogate). $\tilde { T }$ is convex on the feasible set of $\left( \mathcal { P } _ { 2 } \right)$

Proof. By $( 3 2 ) ,$ , each $g _ { k }$ is concave, so the gateway-side argument $\mathbb { E } _ { \mathbf { s } } [ P _ { 0 } ^ { \mathsf { c m p } } ]$ in $( 3 6 ) .$ , a nonnegative combination of the $g _ { k }$ , is concave, and composing it with the convex decreasing map $x \mapsto x ^ { - 1 / 3 }$ yields a convex gateway-side term. As each expert-side term $\overset { \cdot \alpha _ { k } } { \underset { \kappa } {  } } ( P ^ { \ s } - P _ { k } ) ^ { - 1 / \bar { 3 } }$ is convex on $P _ { k } < P ^ { s } , \tilde { T }$ is a sum of convex terms and hence convex. ■

Problem $\left( \mathcal { P } _ { 2 } \right)$ is thus a tractable convex surrogate of the original average-latency problem $\left( \mathcal { P } _ { 1 } \right)$ . The next section derives the optimal solutions to the surrogate problems under unidirectional and bidirectional WPT. Finally, we bound the average-latency loss incurred by optimizing the surrogate rather than the original objective in each WPT regime.

## IV. OPTIMAL WPT POWER CONTROL

In this section, we solve the surrogate problem $\left( \mathcal { P } _ { 2 } \right)$ to achieve the near-optimal WPT power control. We then establish its performance guarantee with respect to the original problem $\left( \mathcal { P } _ { 1 } \right)$ . Finally, we discuss unidirectional WPT as a practical restriction.

## A. Optimal WPT Power Control

In this subsection, we derive the optimal solution of problem $\left( \mathcal { P } _ { 2 } \right)$ and analyze its structure and performance. $\mathbf { A } s$ established in Lemma 3, the surrogate objective $\tilde { T }$ is convex. At the same time, the feasible set of problem $\left( \mathcal { P } _ { 2 } \right)$ is also convex. Hence, problem $\left( \mathcal { P } _ { 2 } \right)$ is a convex optimization problem. Its global optimum can therefore be obtained from the KKT conditions, as stated in the following theorem.

Theorem 1 (Optimal WPT Power Control). The unique optimal active-state power control for problem $\left( \mathcal { P } _ { 2 } \right)$ is given by

$$
P _ { k } ^ { \star } = \frac { P ^ { 5 } } { \nu ^ { \star } } \Big ( [ \nu ^ { \star } - \chi \eta _ { k } ^ { - \frac { 3 } { 4 } } ] ^ { + } - \operatorname* { m i n } \Big \{ [ \chi \eta _ { k } ^ { \frac { 3 } { 4 } } - \nu ^ { \star } ] ^ { + } , \eta _ { k } \nu ^ { \star } \Big \} \Big ) ,\tag{37}
$$

where $\begin{array} { r } { \chi \triangleq \left( \frac { \mu ^ { \mathrm { f f n } } } { \kappa \mu ^ { \mathrm { a t t } } } \right) ^ { 3 / 4 } } \end{array}$ . The water level $\nu ^ { \star } > 0$ is the unique solution to the following equation

$$
\begin{array} { c } { { \displaystyle \left[ 1 + \sum _ { k = 1 } ^ { K } ( 1 - \alpha _ { k } ) \eta _ { k } \right] \nu ^ { \star } + \sum _ { k = 1 } ^ { K } \alpha _ { k } \eta _ { k } \big [ \nu ^ { \star } - \chi \eta _ { k } ^ { - \frac { 3 } { 4 } } \big ] ^ { + } } } \\ { { - \displaystyle \sum _ { k = 1 } ^ { K } \frac { \alpha _ { k } } { \eta _ { k } } \operatorname* { m i n } \Bigl \{ \big [ \chi \eta _ { k } ^ { \frac { 3 } { 4 } } - \nu ^ { \star } \big ] ^ { + } , \eta _ { k } \nu ^ { \star } \Bigr \} = 1 . } } \end{array}\tag{38}
$$

Proof. See Appendix VII-B.

Theorem 1 divides the optimal power control into three modes. Under the signed-power definition in (25), positive, zero, and negative values of $P _ { k } ^ { \star }$ correspond to donation, neutrality, and reception, respectively. To make these three modes explicit, (37) can be rewritten as

$$
\begin{array} { r } { P _ { k } ^ { \star } = \left\{ \begin{array} { l l } { \displaystyle \frac { P ^ { 5 } } { \nu ^ { \star } } \big ( \nu ^ { \star } - \chi \eta _ { k } ^ { - \frac { 3 } { 4 } } \big ) , } & { \nu ^ { \star } > \chi \eta _ { k } ^ { - \frac { 3 } { 4 } } , } \\ { 0 , } & { \chi \eta _ { k } ^ { \frac { 3 } { 4 } } \leq \nu ^ { \star } \leq \chi \eta _ { k } ^ { - \frac { 3 } { 4 } } , } \\ { - \displaystyle \frac { P ^ { 5 } } { \nu ^ { \star } } \operatorname* { m i n } \Bigl \{ \chi \eta _ { k } ^ { \frac { 3 } { 4 } } - \nu ^ { \star } , \eta _ { k } \nu ^ { \star } \Bigr \} , } & { \nu ^ { \star } < \chi \eta _ { k } ^ { \frac { 3 } { 4 } } , } \end{array} \right. \quad \quad } \end{array}\tag{39}
$$

where the min{·} term enforces the receive cap $\eta _ { k } P ^ { s }$ imposed by the gateway transmit-power limit. The variation of $P _ { k } ^ { \star }$ with the water level is illustrated in Fig. 5.

Figure 5 shows the mode transition of an expert, driven by the water level $\smash { \nu ^ { \star } . \mathrm { ~ A s ~ } \nu ^ { \star } }$ increases, the expert moves from reception to neutrality and then to donation. The reception and donation thresholds lie on opposite sides of $\chi .$ Therefore, the water level cannot activate both transfer directions simultaneously, which is formalized in the following Corollary 1.

![](images/ddc7434ad5ac72a663fbe26dcfe0403c4f1474ee575cf81517b267070dc0f2ab.jpg)  
Fig. 5. Three transfer modes of the optimal power control.

Corollary 1 (Directionally Consistent Optimal Transfers). All nonzero optimal active-state transfers have the same direction, written as

$$
\begin{array} { r } { P _ { k _ { 1 } } ^ { \star } P _ { k _ { 2 } } ^ { \star } \geq 0 , \qquad \forall k _ { 1 } , k _ { 2 } \in \{ 1 , \ldots , K \} . } \end{array}
$$

Proof. By (39), donation by expert $k _ { 1 }$ requires $\nu ^ { \star } > \chi \eta _ { k _ { 1 } } ^ { - 3 / 4 } \ge$ $\chi ,$ whereas reception by expert $k _ { 2 }$ requires $\nu ^ { \star } < \chi \eta _ { k _ { 2 } } ^ { 3 / 4 } \leq \chi .$ Since $\nu ^ { \star }$ is shared across all experts, these conditions cannot hold simultaneously. Hence, all nonzero active-state transfers have the same direction. ■

Corollary 1 shows that power donation and reception cannot coexist under the optimal active-state control. We therefore separate the signed solution in (37) into its donation and reception components, written as

$$
\begin{array} { l } { { ( { \cal P } _ { k } ^ { \star } ) ^ { + } = \displaystyle \frac { { \cal P } ^ { s } } { \nu ^ { \star } } \big [ \nu ^ { \star } - \chi \eta _ { k } ^ { - 3 / 4 } \big ] ^ { + } , } } \\ { { ( - { \cal P } _ { k } ^ { \star } ) ^ { + } = \displaystyle \frac { { \cal P } ^ { s } } { \nu ^ { \star } } \operatorname* { m i n } \Bigl \{ \big [ \chi \eta _ { k } ^ { 3 / 4 } - \nu ^ { \star } \big ] ^ { + } , \eta _ { k } \nu ^ { \star } \Bigr \} . } } \end{array}
$$

The two directional components in (37) exhibit distinct powercontrol structures. For power donation, the optimal power is proportional to the gap between the water level $\nu ^ { \star }$ and the efficiency-dependent floor $\chi \eta _ { k } ^ { - 3 / 4 }$ . This level-minus-floor relation follows the conventional water-filling structure. For power reception, the optimal power is proportional to the capped gap between the ceiling $\chi \eta _ { k } ^ { 3 / 4 }$ and the water level. This ceiling-minus-level relation gives rise to a reverse waterfilling structure. Together, they form the two-sided water-filling structure illustrated in Fig. 6 and analyzed in the following Remark 2.

Remark 2 (Two-Sided Water-Filling Power Control). The optimal WPT control extends conventional water-filling to both sides of the water level $\nu ^ { \star }$ . Expert k donates power when $\nu ^ { \star }$ exceeds its floor and receives power when $\nu ^ { \star }$ falls below its ceiling. A higher WPT efficiency $\eta _ { k }$ lowers the floor and raises the ceiling, thereby increasing the transferred power between expert k and the gateway.

![](images/dd9f89e6b581a6defc2d714011236d07bf5b3685e6bd72cccb8e4114da3c7d7f.jpg)  
Fig. 6. Two-sided water-filling power control for SpaceWPT.

Remark 3 (Efficiency-Dependent Neutral Region). The reception ceiling and donation floor define a neutral region of width $\chi ( \eta _ { k } ^ { - 3 / 4 } - \eta _ { k } ^ { 3 / 4 } )$ for expert k. Since this region enlarges with η<sub>k</sub> decreases, experts with lower-efficiency WPT links are more likely to be neutral.

We now combine the active- and idle-state results to obtain the complete WPT control. Theorem 1 gives the optimal signed power for an active expert, while Lemma 1 gives the optimal powers for an idle expert. Using the mapping in (27), the optimal WPT powers are given by

$$
\begin{array} { r } { [ P _ { k \to 0 } ^ { \star } ( s _ { k } ) , P _ { 0 \to k } ^ { \star } ( s _ { k } ) ] = \left\{ \begin{array} { l l } { [ P ^ { \star } , 0 ] , } & { s _ { k } = 0 , } \\ { \big [ ( P _ { k } ^ { \star } ) ^ { + } , \frac { ( - P _ { k } ^ { \star } ) ^ { + } } { \eta _ { k } } \big ] , } & { s _ { k } = 1 . } \end{array} \right. } \end{array}\tag{40}
$$

Averaging (40) over $s _ { k }$ yields the long-term average transfer powers over inference rounds. The long-term transfer power from expert k to the gateway is given by

$$
\begin{array} { r } { \bar { P } _ { k  0 } ^ { \star } \triangleq \mathbb { E } _ { s _ { k } } [ P _ { k  0 } ^ { \star } ( s _ { k } ) ] = ( 1 - \alpha _ { k } ) P ^ { \ s } + \alpha _ { k } ( P _ { k } ^ { \star } ) ^ { + } , } \end{array}\tag{41}
$$

The long-term transfer power from the gateway to expert k is given by

$$
\bar { P } _ { 0 \to k } ^ { \star } \triangleq { \mathbb E } _ { s _ { k } } [ P _ { 0 \to k } ^ { \star } ( s _ { k } ) ] = \alpha _ { k } \frac { ( - P _ { k } ^ { \star } ) ^ { + } } { \eta _ { k } } .\tag{42}
$$

## B. Analysis of the Optimal Solution

This subsection evaluates the derived WPT control from three perspectives: solution optimality, complexity, and practical deployment. We first establish its near-optimality and then quantify its computational advantage over directly solving problem $( \mathcal { P } _ { 0 } )$ . Finally, we consider a practical deployment with unidirectional WPT.

We first evaluate $\mathbf { p } ^ { \star }$ against the optimal solution of the original problem $( \mathcal { P } _ { 0 } )$ . The design principles established in Lemmas 1 and 2 reduce problem $( \mathcal { P } _ { 0 } )$ to problem $\left( \mathcal { P } _ { 1 } \right)$ without compromising optimality. Problem $\left( \mathcal { P } _ { 2 } \right)$ then replaces the average latency in problem $\left( \mathcal { P } _ { 1 } \right)$ with the surrogate $\tilde { T } .$ The optimality gap between the original one and the transformed one is bounded by the following Corollary 2.

Corollary 2 (Near-optimality bound). The optimality gap of the solution $\mathbf { p } ^ { \star }$ satisfies

$$
0 \leq \bar { T } ( \mathbf { p } ^ { \star } ) - \operatorname* { m i n } _ { \mathbf { p } } \bar { T } ( \mathbf { p } ) \leq \varepsilon _ { 0 } ,\tag{43}
$$

where the constant $\varepsilon _ { \mathrm { 0 } }$ is given in (53) and depends only on the system parameters.

Proof. See Appendix VII-C.

We next compare the computational complexity of the proposed solution with that of directly solving problem $( \mathcal { P } _ { 0 } )$

Remark 4 (Computational Complexity). Directly solving problem $( \mathcal { P } _ { 0 } )$ involves 4K power variables and up to $\binom { K } { \kappa }$ activation patterns. Theorem 1 reduces the optimization to a single water-level equation in $\nu ^ { \star }$ , whose solution determines all K powers via (37). This water-level equation can be solved in O(K log K) operations.

Finally, we consider unidirectional WPT as a practical restriction of the general control. The gateway executes the attention block for every token, whereas only κ of the K experts are activated. This workload asymmetry motivates the expert-to-gateway restriction $P _ { k } \ge 0$ , which reduces onboard WPT hardware requirements and control overhead. The optimal control under this restriction is stated in the following corollary.

Corollary 3 (Power Control for Unidirectional-WPT Case). When $P _ { k } \ge 0 ,$ , the unique optimal active-state power of expert k is

$$
P _ { k } ^ { \mathsf { u n i } \star } = \frac { P ^ { \mathsf { s } } } { \nu _ { \mathsf { u n i } } ^ { \star } } \left[ \nu _ { \mathsf { u n i } } ^ { \star } - \chi \eta _ { k } ^ { - 3 / 4 } \right] ^ { + } ,\tag{44}
$$

where $\nu _ { \mathrm { u n i } } ^ { \star } > 0$ is uniquely determined by

$$
\begin{array} { r l r } {  { [ 1 + \sum _ { k = 1 } ^ { K } ( 1 - \alpha _ { k } ) \eta _ { k } ] \nu _ { \mathrm { u n i } } ^ { \star } } } \\ & { } & { + \sum _ { k = 1 } ^ { K } \alpha _ { k } \eta _ { k } [ \nu _ { \mathrm { u n i } } ^ { \star } - \chi \eta _ { k } ^ { - 3 / 4 } ] ^ { + } = 1 . } \end{array}\tag{45}
$$

The unidirectional control retains the expert-to-gateway water-filling branch of the general solution, illustrated in Fig. 6(a). We use it as a practical baseline in Section V to quantify the benefit of removing the directional restriction.

![](images/98e5be5de834e6fff1c085161b7282e1805e5e4716ed0b8476d04135f584e979.jpg)  
(a) Gateway-side power control

![](images/b997fd5243b11005562516ec5437d066d91ce80360121ac3f10d6350789fab12.jpg)  
(b) E2E per-token latency  
Fig. 7. Power flow and the resulting E2E latency reduction for Mixtral.

## V. EXPERIMENTAL RESULTS

## A. Experimental Setup

1) SpaceDC Configuration: The simulated SpaceDC comprises one gateway satellite and K co-orbiting expert satellites. Under the HCW relative-motion model in (6), expert k follows a zero-offset closed relative orbit with $\rho _ { k } = \rho , y _ { k } ^ { \mathsf { o f f } } = 0$ , and $\phi _ { k } = 2 \pi ( k - 1 ) / K$ . The resulting gateway–expert distance varies from $\rho$ to $2 \rho$ over one orbital period. Unless otherwise specified, we set $\rho = 5 0 \mathrm { m }$ and evaluate each scheme at four uniformly spaced orbital snapshots.

Each gateway–expert pair uses separate FSO and laserbased WPT links for data and power transfer, respectively. We set $R ( d _ { k } ) \ = \ 1 0 0 \mathrm { G b p s }$ and $Q ^ { \mathsf { d l } } \ = \ Q ^ { \mathsf { u l } } \ = \ 4 0 9 6 \times 1 6$ bits. Under the communication model in (10), this fixed rate leaves the propagation delay $d _ { k } / c$ as the only distancedependent communication term. The K dedicated WPT links form a star centered at the gateway. We use a saturated approximation of the end-to-end efficiency in (12), given by $\eta _ { k } = \eta _ { \mathrm { m a x } }$ min $\{ 1 , ( 1 0 0 \mathrm { m } / d _ { k } ) ^ { 2 } \}$ , with $\eta _ { \mathrm { m a x } } = 0 . 6 8 9$ by default.

2) GPU Configuration: The per-satellite solar-power budget is set to $P ^ { \mathrm { s } } = 6 9 . 0 9 \mathrm { W } ,$ and every node is assigned the same GPU profile. We calibrate the DVFS model in (3) using power– frequency measurements collected on an NVIDIA A100 GPU, with a reference clock frequency of 990 MHz at 69.09 W. The coefficient $\gamma$ incorporates the conversion from GPU clock frequency to effective computation speed. This calibration is shared by the gateway and all expert satellites. Unless otherwise specified, these settings remain fixed, and only the parameter under study is varied.

3) MoE Configuration: We evaluate Mixtral-8x7B-Instruct and Phi-3.5-MoE-Instruct on the SpaceDC [36], [37]. Both models contain $L = 3 2$ MoE layers and adopt native Top-2 routing. Mixtral has $K = 8$ experts per layer, whereas Phi has $K = 1 6$ . Both models are evaluated at a 221-token reference context, obtained by rounding the mean decoding context of 221.13 tokens measured over the full Mixtral–HumanEval workload.

For each model, we process all 164 HumanEval prompts using batch size one and decode 128 tokens per prompt [38]. We record the Top-2 active expert pair at every MoE layer and decoding step. Since WPT control changes only the compute power and latency, HumanEval supplies the activation workload rather than an accuracy metric.

![](images/7d87db0c00738d28111a203c2181232ba92826e52cb7d5a01696d0fc202095a3.jpg)  
(a) Mixtral-8x7B

![](images/1d60ee06217e084c1f000150c18c71612acc783e88f90e90621b20f0daa1402f.jpg)  
(b) Phi-3.5-MoE

Fig. 8. E2E latency versus the relative-orbit amplitude of the SpaceDC.  
![](images/fe9553983c97e2a0c97accd3b7b14e4dc0508c4b0514822767c83445bbc21112.jpg)

![](images/92f9aab4e68c380f40f2d5cfec4070bb4dabc2b6f0becf4696a75c49cb477902.jpg)  
(a) Mixtral-8x7B  
(b) Phi-3.5-MoE  
Fig. 9. E2E latency versus peak WPT efficiency.

For layer ℓ, we write $\alpha _ { \ell , k }$ for the empirical marginal activation probability of expert k, corresponding to $\alpha _ { k }$ in (2). At every layer and orbital snapshot, we compute the analytical Uni-WPT and SpaceWPT controls using $\{ \alpha _ { \ell , k } \}$ and the snapshotdependent WPT efficiencies $\{ \eta _ { k } \}$ . We then evaluate all schemes with the recorded joint expert-pair frequencies using the original average-latency model in (21), which preserves the active-expert maximum. The reported E2E per-token latency aggregates the modeled latency across layers and is averaged over expert activations and orbital positions. Thus, the marginal model is used only to design the controls, whereas the reported latency retains the observed Top-2 activation correlations.

4) Benchmarking Schemes: We compare the proposed analytical controls with a system without WPT and a fixed-ratio

heuristic.

• No-WPT: Each satellite uses only its local solar-power budget for computation.

• Fixed-WPT: Each idle expert transmits a fixed 50% of its solar-power budget to the gateway, whereas active experts perform no WPT.

• Uni-WPT: This unidirectional restriction of SpaceWPT allows power to flow only from the experts to the gateway. The active-state powers follow the water-filling control in (44), while idle experts follow Lemma 1.

• SpaceWPT: Power may flow in either direction between the gateway and each expert, following the two-sided water-filling control in Theorem 1.

All schemes use the same model deployment, routing decisions, orbital snapshots, and communication and computation settings; they differ only in the WPT directions and transmit powers.

## B. SpaceWPT Performance

At the default operating point, Fig. 7 shows that the analytical controls reduce the E2E latency of Mixtral in two stages. No-WPT incurs 5.7041 ms/token, whereas Uni-WPT reduces the latency to 5.2319 ms/token, representing an 8.28% improvement. SpaceWPT further reduces it to 4.6238 ms/token, which is 11.62% below Uni-WPT and 18.94% below No-WPT.

The Uni-WPT gain comes entirely from optimal idle-expert donation. As characterized in Lemma 1, each idle expert transmits its full solar-power budget, delivering 285.62 W on average to the gateway after WPT loss and raising its compute power from 69.09 to 354.71 W, as shown in Fig. 7(a). At the default operating point, all active experts remain neutral, with $P _ { k } ^ { \star } = 0 .$ , so their computation latency is unchanged from No-WPT. The resulting 8.28% reduction therefore stems solely from the shorter gateway attention latency.

SpaceWPT further reduces the latency by reallocating part of the gateway power to the active experts. Compared with Uni-WPT, the gateway allocates an average of 138.18 W to activeexpert WPT and retains 216.53 W for attention computation, as shown in Fig. 7(a). This reallocation increases the attentionlatency component by 0.1165 ms/token but reduces the criticalexpert component by 0.7246 ms/token, yielding the net E2E latency reduction of 11.62% in Fig. 7(b).

## C. Effects of System Parameters

This subsection compares the four schemes under variations in system parameters. The evaluation covers WPT-link conditions through the relative-orbit amplitude and peak WPT efficiency, and onboard computing resources through the persatellite solar-power budget and GPU reference frequency.

The gain of SpaceWPT is governed by the per-link WPT efficiencies, which depend on both the formation size and the peak transfer efficiency. As shown in Fig. 8, increasing $\rho$ from 40 to 90 m lengthens the gateway–expert links and reduces the gain over No-WPT from 18.95% to 7.35% for Mixtral and from 28.13% to 19.22% for Phi. Conversely, Fig. 9 shows that increasing $\eta _ { \mathrm { m a x } }$ from 0.35 to 0.75 raises the gain from 8.78% to 20.21% for Mixtral and from 18.80% to 29.25% for Phi.

![](images/162d84e584e9c72b246a9a28c43e212ae6413023036e60cf000a68d1fb01cbf5.jpg)  
(a) Mixtral-8x7B

![](images/e96340a74f9039de52d6a00ac3667cd8ec7bad05b6092ef80fac9a359e10eccd.jpg)  
(b) Phi-3.5-MoE

Fig. 10. E2E latency versus the per-satellite solar-power budget.  
![](images/148358a6707b8a33c319691f3395768c0247731cfa19a6ebccfb8ce3f07f136c.jpg)  
(a) Mixtral-8x7B

![](images/09e14f7a467729f93f683fc5937a0e2f87160b17f186a5a0d64df77efdb19448.jpg)  
(b) Phi-3.5-MoE  
Fig. 11. E2E latency versus the GPU reference frequency.

Both trends follow from the power delivered for computation: longer links increase transfer loss, whereas a higher peak efficiency reduces this loss in both WPT directions. These results indicate that compact formations and high-efficiency laser-based WPT hardware directly strengthen the latency advantage of SpaceWPT.

Increasing the solar-power budget or GPU frequency reduces computation latency across the SpaceDC. As shown in Figs. 10 and 11, this lowers the E2E latency for all four schemes, while SpaceWPT provides a further reduction throughout both sweeps—about 19% for Mixtral and 28% for Phi relative to No-WPT. The two mechanisms are complementary. Specifically, additional solar power or a higher GPU frequency accelerates computation at every satellite, whereas WPT redistributes the available power toward the gateway and active experts according to instantaneous demand.

## VI. CONCLUDING REMARKS

This work advocates inter-satellite WPT to support lowlatency distributed inference in SpaceDCs. The key feature lies in introducing a new degree of freedom for space computing, i.e., WPT-enabled energy migration, to cope with imbalanced computing loads. Such a power-sharing policy redistributes energy according to demand under spatially uniform sunlight exposure. The developed principles for migration direction and activation-based WPT control provide theoretical guidance for the development of emerging SpaceDCs in dawn–dusk SSOs.

This work opens a new frontier for integrating WPT into distributed computing in space. One direction is to extend the WPT design to more general topologies, where direct power exchange between expert satellites allows more flexible power redistribution according to computation demands and link efficiencies. Another direction is to unify the separate laser-based WPT and communication links into a single optical link that supports both energy delivery and data transmission. Such integration could reduce onboard hardware requirements while calling for joint designs that balance WPT efficiency and communication performance.

## VII. APPENDIX

## A. Proof of Proposition 1

The surrogate (36) follows from a certainty-equivalent approximation of the two random components of $\bar { T } ( \mathbf { p } )$ ).

Gateway-side. The gateway latency $\mu ^ { \mathsf { a t t } } ( P _ { 0 } ^ { \mathsf { c m p } } ) ^ { - 1 / 3 }$ is random through the activation s. Replacing the random power $P _ { 0 } ^ { \mathsf { c m p } }$ by its mean inside the nonlinearity yields the certaintyequivalent term $\mu ^ { \mathsf { a t t } } ( \mathbb { E } _ { \mathbf { s } } [ P _ { 0 } ^ { \mathsf { c m p } } ] ) ^ { - 1 / 3 }$ . Since expert k contributes $g _ { k } ( P _ { k } )$ to the gateway when active and $\eta _ { k } P ^ { s }$ when idle, with activation probability $\alpha _ { k }$

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { s } } [ P _ { 0 } ^ { \mathsf { c m p } } ] = P ^ { \mathsf { s } } + \sum _ { k = 1 } ^ { K } \big [ ( 1 - \alpha _ { k } ) \eta _ { k } P ^ { \mathsf { s } } + \alpha _ { k } g _ { k } ( P _ { k } ) \big ] , } \end{array}
$$

which is the gateway-side term of (36).

Expert-side. The expert latency is the maximum over the κ active experts, $\mathrm { n a x } _ { k \in \mathcal { A } ( \mathbf { s } ) } c _ { k } ( P _ { k } )$ with $c _ { k } ( P _ { k } ) = T _ { k } ^ { \mathsf { c m m } } \ .$ + $\mu ^ { \mathsf { f f n } } ( P ^ { \mathsf { s } } - P _ { k } ) ^ { - 1 / 3 }$ . Approximating this maximum by the average over the active experts and taking expectation,

$$
\begin{array} { r } { \mathbb { E } _ { \mathbf { s } } \Big [ \frac { 1 } { \kappa } \sum _ { k \in \mathcal { A } ( \mathbf { s } ) } c _ { k } ( P _ { k } ) \Big ] = \frac { 1 } { \kappa } \sum _ { k = 1 } ^ { K } \alpha _ { k } c _ { k } ( P _ { k } ) , } \end{array}
$$

where we used $\operatorname* { P r } \{ k \in \mathcal { A } ( \mathbf { s } ) \} = \alpha _ { k }$ . This is the expert-side term of (36), completing the construction.

## B. Proof of Theorem 1

The bidirectional surrogate ${ \tilde { T } } ( \mathbf { p } )$ is strictly convex on the feasible set of $\left( \mathcal { P } _ { 2 } \right)$ , so its KKT point is the unique minimizer. Define the mean gateway compute power induced by p as

$$
\bar { P } _ { 0 } ^ { \mathrm { c m p } } ( \mathbf { p } ) \triangleq P ^ { 5 } + \sum _ { j = 1 } ^ { K } \bigl [ ( 1 - \alpha _ { j } ) \eta _ { j } P ^ { 5 } + \alpha _ { j } g _ { j } ( P _ { j } ) \bigr ] .\tag{46}
$$

At an interior point, stationarity in coordinate $P _ { k }$ gives

$$
\mu ^ { \mathrm { a t t } } g _ { k } ^ { \prime } ( P _ { k } ) \bigl ( \bar { P } _ { 0 } ^ { \mathrm { c m p } } ( { \bf p } ) \bigr ) ^ { - \frac { 4 } { 3 } } = \frac { \mu ^ { \mathrm { f f n } } } { \kappa } ( P ^ { s } - P _ { k } ) ^ { - \frac { 4 } { 3 } } ,\tag{47}
$$

where the common factor $\alpha _ { k } > 0$ cancels from both sides.

Let $\bar { P } _ { 0 } ^ { \mathsf { c m p } \star } \triangleq \bar { P } _ { 0 } ^ { \mathsf { c m p } } ( \mathbf { p } ^ { \star } )$ and define $\nu ^ { \star } \triangleq P ^ { \mathsf { s } } / \bar { P } _ { 0 } ^ { \mathsf { c m p } \star }$ . For a donor, $g _ { k } ^ { \prime } = \eta _ { k }$ , and (47) gives

$$
P _ { k } ^ { \star } = \frac { P ^ { \mathsf { s } } } { \nu ^ { \star } } \big ( \nu ^ { \star } - \chi \eta _ { k } ^ { - 3 / 4 } \big ) .
$$

For an interior receiver, $g _ { k } ^ { \prime } = 1 / \eta _ { k }$ , which instead gives

$$
P _ { k } ^ { \star } = - \frac { P ^ { 5 } } { \nu ^ { \star } } \big ( \chi \eta _ { k } ^ { 3 / 4 } - \nu ^ { \star } \big ) .
$$

At $P _ { k } ^ { \star } = 0$ , the left and right derivatives straddle zero if and only if

$$
\begin{array} { r } { \chi \eta _ { k } ^ { 3 / 4 } \leq \nu ^ { \star } \leq \chi \eta _ { k } ^ { - 3 / 4 } . } \end{array}
$$

Finally, the lower bound $P _ { k } \ge - \eta _ { k } P ^ { s }$ clips the receiver relation at $- \eta _ { k } P ^ { s }$ . Combining the donor, neutral, interiorreceiver, and capped-receiver conditions yields (37).

Substituting (37) into (46) and dividing by $\bar { P } _ { 0 } ^ { \mathsf { c m p } \star } = P ^ { \mathsf { s } } / \nu ^ { \star }$ gives (38). Conversely, any positive solution of this balance equation together with (37) satisfies the KKT conditions, including the kink and receive-cap conditions. Existence follows from the optimum of the strictly convex problem, and strict convexity permits only one such solution. Hence both $\nu ^ { \star }$ and $\mathbf { p } ^ { \star }$ are unique. ■

## C. Proof of Corollary 2

For any feasible p, Jensen’s inequality shows that the certainty-equivalent gateway latency is no larger than the corresponding average latency. Moreover, the average latency of the κ active experts is no larger than their maximum latency. Therefore,

$$
{ \tilde { T } } ( \mathbf { p } ) \leq { \bar { T } } ( \mathbf { p } ) .\tag{48}
$$

Let $\eta _ { [ 1 ] } \ge \cdots \ge \eta _ { [ K ] }$ denote the ordered WPT efficiencies. The reduced-problem analysis and performance guarantee assume

$$
\delta \triangleq 1 - \kappa + \sum _ { k = 1 } ^ { K } \eta _ { k } - \sum _ { i = 1 } ^ { \kappa } \eta _ { [ i ] } > 0\tag{49}
$$

Under the simplified control, this ensures positive gateway power in every state. In fact, $- P ^ { s } \leq g _ { k } ( P _ { k } ) < \eta _ { k } P ^ { s }$ , and exactly κ experts are active. It follows that

$$
P _ { 0 } ^ { \mathsf { c m p } } ( \mathbf { p } ; \mathbf { s } ) \geq P ^ { s } \delta > 0 .\tag{50}
$$

Hence, the reduced problems need no additional per-state power constraints.

We next bound the expert compute powers at $\mathbf { p } ^ { \star }$ . Let $A \triangleq$ $\begin{array} { r } { 1 + \sum _ { i = 1 } ^ { K } ( 1 - \alpha _ { j } ) \eta _ { j } } \end{array}$ . If $P _ { k } ^ { \star } \ \leq \ 0$ , then $P ^ { s } - P _ { k } ^ { \star } \ge P ^ { s }$ . If $P _ { k } ^ { \star } > 0 ,$ , the directional consistency in Corollary 1 eliminates the reception term in (38), yielding $\nu ^ { \star } \leq A ^ { - 1 }$ . Combining this result with the threshold conditions in (39) and the receive cap gives

$$
P ^ { s } \operatorname* { m i n } \{ 1 , \chi A \eta _ { k } ^ { - 3 / 4 } \} \leq P ^ { s } - P _ { k } ^ { \star } \leq ( 1 + \eta _ { k } ) P ^ { s } .\tag{51}
$$

Let $\eta _ { \mathrm { m a x } } \triangleq \operatorname* { m a x } _ { k } \eta _ { k }$ . The gateway power varies by at most $\kappa ( 1 + \eta _ { \mathrm { m a x } } ) P ^ { s }$ over the activation states. Its variance is thus at most one quarter of the squared range. Applying Taylor’s theorem to $x ^ { - 1 / 3 }$ around the mean gateway power and using (50) bounds the gateway-side approximation gap.

For the expert-side gap, define $T _ { \mathrm { m a x } } ^ { \mathsf { c m m } } \triangleq \operatorname* { m a x } _ { k } T _ { k } ^ { \mathsf { c m m } }$ and $T _ { \mathrm { m i n } } ^ { \mathsf { c m m } } \triangleq \operatorname* { m i n } _ { k } T _ { k } ^ { \mathsf { c m m } }$ . The difference between the maximum and average latency of any κ active experts is at most $( \kappa - 1 ) / \kappa$ times the global expert-latency range. Applying (51) to this range and combining the two approximation gaps gives

$$
\bar { T } ( \mathbf { p } ^ { \star } ) - \tilde { T } ( \mathbf { p } ^ { \star } ) \leq \varepsilon _ { 0 } ,\tag{52}
$$

where

$$
\begin{array} { l } { \varepsilon _ { 0 } \triangleq \frac { \mu ^ { \mathrm { a t t } } \kappa ^ { 2 } ( 1 + \eta _ { \mathrm { m a x } } ) ^ { 2 } } { 1 8 P ^ { \mathrm { s } 1 / 3 } \delta ^ { 7 / 3 } } } \\ { + \frac { \kappa - 1 } { \kappa } \Bigg [ T _ { \mathrm { m a x } } ^ { \mathrm { c m m } } - T _ { \mathrm { m i n } } ^ { \mathrm { c m m } } } \\ { + \mu ^ { \mathrm { f i n } } \Big ( ( P _ { \mathrm { e , m i n } } ^ { \mathrm { c m p } } ) ^ { - 1 / 3 } - [ ( 1 + \eta _ { \mathrm { m a x } } ) P ^ { \mathrm { s } } ] ^ { - 1 / 3 } \Big ) \Bigg ] , } \end{array}\tag{53}
$$

where $P _ { \mathrm { e , m i n } } ^ { \mathsf { c m p } }$ is given by

$$
P _ { \mathrm { e , m i n } } ^ { \mathsf { c m p } } \triangleq P ^ { \mathsf { s } } \operatorname* { m i n } \{ 1 , \chi A \eta _ { \mathrm { m a x } } ^ { - 3 / 4 } \} .
$$

Finally, let $\widehat { \mathbf p }$ minimize the original average latency ${ \bar { T } } ( \mathbf { p } )$ Since $\mathbf { p } ^ { \star }$ minimizes the surrogate objective, (48) and (52) give

$$
\begin{array} { r l } & { \bar { T } ( \mathbf { p } ^ { \star } ) \leq \tilde { T } ( \mathbf { p } ^ { \star } ) + \varepsilon _ { 0 } \leq \tilde { T } ( \widehat { \mathbf { p } } ) + \varepsilon _ { 0 } } \\ & { \qquad \leq \bar { T } ( \widehat { \mathbf { p } } ) + \varepsilon _ { 0 } . } \end{array}
$$

Together with $\bar { T } ( \widehat { \mathbf { p } } ) \leq \bar { T } ( \mathbf { p } ^ { \star } )$ , this proves (43).

## REFERENCES

[1] W. Saad, M. Bennis, and M. Chen, “A vision of 6G wireless systems: Applications, trends, technologies, and open research problems,” IEEE Netw., vol. 34, no. 3, pp. 134–142, 2020.

[2] Y. Liu, Y. Han, H. Li, S. Gu, J. Qiu, and T. Li, “Computing over space: Status, challenges, and opportunities,” Engineering, vol. 54, pp. 20–25, 2025.

[3] Z. Wang, J. Huang, M. Sheng, and K. Huang, “Space AI infrastructure: Where the edge meets the orbital cloud,” arXiv:2610.16395, 2026.

[4] J. Virgili Llop, P. C. E. Roberts, K. Palmer, S. Hobbs, and J. Kingston, “Descending sun-synchronous orbits with aerodynamic inclination cor rection,” J. Guid. Control Dyn., vol. 38, no. 5, pp. 831–842, 2015.

[5] B. Agüera y Arcas, T. Beals, M. Biggs, J. V. Bloom, T. Fischbacher, K. Gromov, U. Köster, R. Pravahan et al., “Toward a future space-based, highly scalable AI infrastructure system design,” Joule, 2026, Art. no. 102678.

[6] A. de Vries, “The growing energy footprint of artificial intelligence,” Joule, vol. 7, no. 10, pp. 2191–2194, 2023.

[7] Z. Wang, H. Yang, M. Sheng, K. B. Letaief, and K. Huang, “SpaceMoE: Realizing distributed mixture-of-experts inference over space networks,” arXiv:2605.00515, 2026.

[8] N. Shazeer, A. Mirhoseini, K. Maziarz, A. Davis, Q. V. Le, G. E. Hinton, and J. Dean, “Outrageously large neural networks: The sparsely-gated mixture-of-experts layer,” in Proc. Int. Conf. Learn. Represent. (ICLR), Toulon, France, 2017.

[9] A. Mirhoseini, H. Pham, Q. V. Le, B. Steiner, R. Larsen, Y. Zhou, N. Kumar, M. Norouzi et al., “Device placement optimization with reinforcement learning,” in Proc. Int. Conf. Mach. Learn. (ICML), Sydney, Australia, 2017, pp. 2430–2439.

[10] S. Go and D. Mahajan, “MoETuner: Optimized mixture of expert serving with balanced expert placement and token routing,” arXiv:2502.06643, 2025.

[11] T. Wu, L. Wang, Z. Wen, X. Zhang, X. Chen, J. Duan, X. Zhang, and J. Zuo, “Accelerating edge inference for distributed MoE models with latency-optimized expert placement,” arXiv:2508.12851, 2025.

[12] Kimi Team, T. Bai, Y. Bai, Y. Bao, M. C., J. Cai, X. Cai, P. Cao et al., “Kimi K3: Open frontier intelligence,” arXiv:2607.24653, 2026.

[13] R. Shahout, C. Cai, Y. Du, M. Yu, and M. Mitzenmacher, “From score distributions to balance: Plug-and-play mixture-of-experts routing,” arXiv:2510.03293, 2025.

[14] Q. Chen, X. Chen, and K. Huang, “SiftMoE: Similarity-aware energyefficient expert selection for wireless distributed MoE inference,” IEEE Trans. Mobile Comput., 2026, to be published.

[15] H. Ju and R. Zhang, “Throughput maximization in wireless powered communication networks,” IEEE Trans. Wireless Commun., vol. 13, no. 1, pp. 418–428, 2014.

[16] K. Huang and V. K. N. Lau, “Enabling wireless power transfer in cellular networks: Architecture, modeling and deployment,” IEEE Trans. Wireless Commun., vol. 13, no. 2, pp. 902–912, 2014.

[17] C. You, K. Huang, and H. Chae, “Energy efficient mobile cloud computing powered by wireless energy transfer,” IEEE J. Sel. Areas Commun., vol. 34, no. 5, pp. 1757–1771, 2016.

[18] S. Bi and Y. J. Zhang, “Computation rate maximization for wireless powered mobile-edge computing with binary computation offloading,” IEEE Trans. Wireless Commun., vol. 17, no. 6, pp. 4177–4190, 2018.

[19] Q. Zeng, Y. Du, and K. Huang, “Wirelessly powered federated edge learning: Optimal tradeoffs between convergence and power transfer,” IEEE Trans. Wireless Commun., vol. 21, no. 1, pp. 680–695, 2022.

[20] S. A. H. Mohsan, H. Qian, and H. Amjad, “A comprehensive review of optical wireless power transfer technology,” Front. Inform. Technol. Electron. Eng., vol. 24, no. 6, pp. 767–800, 2023.

[21] S. Fafard and D. P. Masson, “Perspective on photovoltaic optical power converters,” J. Appl. Phys., vol. 130, no. 16, 2021, Art. no. 160901.

[22] M. Dehghani Soltani, E. Sarbazi, N. Bamiedakis, P. de Souza, H. Kazemi, J. M. H. Elmirghani, I. H. White, R. V. Penty et al., “Safety analysis for laser-based optical wireless communications: A tutorial,” Proc. IEEE, vol. 110, no. 8, pp. 1045–1072, 2022.

[23] M. Qaraqe, M. Usman, A. Serbes, I. S. Ansari, and M.-S. Alouini, “Power hotspots in space: Powering CubeSats via inter-satellite optical wireless power transfer,” IEEE Internet Things Mag., vol. 5, no. 3, pp. 180–185, 2022.

[24] B. Donmez, I. Azam, and G. K. Kurt, “Mitigation of misalignment errors over inter-satellite FSO energy harvesting,” in Proc. IEEE Int. Symp. Pers., Indoor Mobile Radio Commun. (PIMRC), Toronto, ON, Canada, 2023, pp. 1–5.

[25] B. Rosén, “On sampling with probability proportional to size,” J. Statist. Plann. Inference, vol. 62, no. 2, pp. 159–191, 1997.

[26] A. P. Chandrakasan, S. Sheng, and R. W. Brodersen, “Low-power CMOS digital design,” IEEE J. Solid-State Circuits, vol. 27, no. 4, pp. 473–484, 1992.

[27] W. H. Clohessy and R. S. Wiltshire, “Terminal guidance system for satellite rendezvous,” J. Aerosp. Sci., vol. 27, no. 9, pp. 653–658, 1960.

[28] R. Pope, S. Douglas, A. Chowdhery, J. Devlin, J. Bradbury, J. Heek, K. Xiao, S. Agrawal et al., “Efficiently scaling transformer inference,” in Proc. Mach. Learn. Syst. (MLSys), Miami Beach, FL, USA, 2023, pp. 606–624.

[29] A. K. Majumdar and J. C. Ricklin, Eds., Free-Space Laser Communications: Principles and Advances. New York, NY, USA: Springer, 2008.

[30] H. Kaushal and G. Kaddoum, “Optical communication in space: Challenges and mitigation techniques,” IEEE Commun. Surveys Tuts., vol. 19, no. 1, pp. 57–96, 2017.

[31] A. Goldsmith, Wireless Communications. Cambridge, U.K.: Cambridge Univ. Press, 2005.

[32] H. Helmers, E. Lopez, O. Höhn, D. Lackner, J. Schön, M. Schauerte, M. Schachtner, F. Dimroth et al., “68.9% efficient GaAs-based photonic power conversion enabled by photon recycling and optical resonance,” Phys. Status Solidi RRL, vol. 15, no. 7, 2021, Art. no. 2100113.

[33] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, “DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale,” in Proc. Int. Conf. Mach. Learn. (ICML), Baltimore, MD, USA, 2022, pp. 18 332– 18 346.

[34] R. K. Ahuja, T. L. Magnanti, and J. B. Orlin, Network Flows: Theory, Algorithms, and Applications. Englewood Cliffs, NJ, USA: Prentice Hall, 1993.

[35] S. Boyd and L. Vandenberghe, Convex Optimization. Cambridge, U.K.: Cambridge Univ. Press, 2004.

[36] A. Q. Jiang, A. Sablayrolles, A. Roux, A. Mensch, B. Savary, C. Bamford, D. S. Chaplot, D. d. l. Casas et al., “Mixtral of experts,” arXiv:2401.04088, 2024.

[37] M. Abdin, J. Aneja, H. Awadalla, A. Awadallah, A. A. Awan, N. Bach, A. Bahree, A. Bakhtiari et al., “Phi-3 technical report: A highly capable language model locally on your phone,” arXiv:2404.14219, 2024.

[38] M. Chen, J. Tworek, H. Jun, Q. Yuan, H. P. d. O. Pinto, J. Kaplan, H. Edwards, Y. Burda et al., “Evaluating large language models trained on code,” arXiv:2107.03374, 2021.