# From Reactive to Agentic Resilience: Multi-Timescale AI-RAN Orchestration Under Signaling Storms

Sotiris Chatzimiltis<sup>∗</sup>, Graduate Student Member, IEEE, Mahdi Boloursaz Mashhadi<sup>∗</sup>, Senior Member, IEEE, Mohammad Shojafar<sup>∗</sup>, Senior Member, IEEE, and Rahim Tafazolli<sup>∗</sup>, Fellow, IEEE <sup>∗</sup>5G/6GIC, Institute for Communication Systems (ICS), University of Surrey, Guildford, UK {sc02449, m.boloursazmashhadi, m.shojafar, r.tafazolli}@surrey.ac.uk

Abstract—Next-generation open radio access network (Open RAN) and AI-RAN deployments turn the RAN into a disaggregated, compute-aware platform whose service continuity depends on many control loops acting at different timescales. Resilience in this setting is best seen as a lifecycle spanning anticipation, absorption, adaptation, recovery, and evolution. Yet, existing mechanisms remain stage-specific and are rarely coordinated across the control hierarchy. This paper argues that agentic AI offers an orchestration layer that coordinates this lifecycle, turning operator intent into coordinated action while keeping slow language-model reasoning off the fast control path. Using mass-attachment signaling storms as a running study, we build a discrete-event Open RAN attach simulator that captures retry amplification, shared-compute contention, and non-instantaneous provisioning, and we propose a three-tier framework in which an orchestrator at the service management and orchestration layer sets a standing policy, a per-site non-realtime agent judges storms and anticipates scheduled crowds, and a deterministic near-real-time loop enforces every decision. Across experiments, the agentic controller approaches the resilience of a fully provisioned pool of servers at a quarter of its capacity by reading a scheduled event from a calendar and provisioning ahead of it. No reactive weight setting recovers this once the surge arrives. Free-text operator intents ground into control directives at 94% exact match on previously unseen intents, and a single instruction reshapes live behavior end to end. We also quantify the latency, cost, and contention limits that bound it. The simulator, the proposed agentic framework, and the agent prompts are publicly available on GitHub. The supplementary material provides analytical proof of the derived Theorem on anticipation horizon, as well as the subsequent Corollary results on reactive provisioning limits.

Index Terms—AI-RAN, network resilience, agentic AI, large language models, signaling storm, anticipation, orchestration.

## I. INTRODUCTION

Building on the foundations of open radio access networks (Open RAN) [1], AI-RAN advances the evolution toward AInative operation by enabling higher levels of automation and orchestration, and the co-execution of radio access network (RAN) and AI workloads on shared edge infrastructure. As a result, the RAN is evolving from a communication-centric platform into an intelligent, compute-aware, and dynamically orchestrated system [2], [3].

In this setting, resilience becomes a fundamental requirement that extends beyond traditional redundancy and fault tolerance [4], since in Open RAN and AI-RAN service continuity depends on coordinated behavior across disaggregated components, open interfaces, and control loops operating at different timescales [5]. Network resilience is thus best understood not as a single mechanism but as a lifecycle. The literature decomposes it into four interdependent stages: anticipation of emerging threats, absorption of their impact while acceptable service is preserved, adaptation under degraded or adversarial conditions, and recovery of stable operation [6]– [9]. We propose a fifth, evolution, capturing what a system carries forward from one disruption to the next.

Existing literature has made significant progress on individual resilience functions across the anticipation, absorption, adaptation, and recovery stages. Predictive analytics and digital twins support anticipation [10], [11]. Anomaly detection, fault isolation, and access control mechanisms support absorption [12]. Resource allocation, virtual network function provisioning, and network slicing support adaptation [13], and self-healing mechanisms support recovery. However, these capabilities are typically studied as stage-specific or mechanismspecific solutions, with limited attention to how they can be systematically related and orchestrated across the disaggregated control hierarchy of Open RAN and AI-RAN. This paper addresses that gap by treating resilience as one continuous lifecycle and examining agentic orchestration as a unifying layer for coordinating its stages.

Agentic AI is well suited to this role. In contrast to isolated predictive or reactive models, it emphasizes goaldirected behavior, contextual reasoning, planning, tool use, and interaction among specialized agents. These properties are well aligned with resilience management, where decisions must integrate observations across multiple layers, reconcile local and global objectives, and sequence actions across heterogeneous network functions and timescales. In the context of AI-RAN, agentic AI is therefore better viewed as an orchestration layer over the intelligence hierarchy than as another analytics capability [14]–[16]. These capabilities, however, do not come for free. Embedding large language model (LLM)-based agents in closed-loop RAN control adds signaling, computation, and coordination overhead, and introduces new failure modes such as hallucinated or unsafe actions and adversarial manipulation of agent reasoning, all of which must be contained during deployment.

To ground this discussion, we consider signaling storm attacks in next-generation RAN as a motivating study. This scenario is particularly relevant because signaling storms can arise from compromised devices, bot-driven activity, or mass attachment scenarios, and can simultaneously disrupt controlplane stability, radio resource availability, and service continuity [17], [18]. More importantly, the scenario naturally spans the full resilience lifecycle. As such, the study provides a concrete setting for illustrating how agentic orchestration can unify otherwise fragmented resilience mechanisms into a continuous control loop.

## Contributions.

• A multi-timescale agentic resilience orchestration algorithm. We propose a three-tier algorithm in which an intent orchestrator at the service management and orchestration (SMO) layer, a tool-augmented non-real-time (Non-RT) agent, and a deterministic near-real-time (Near-RT) control loop coordinate through a shared policy. The design combines contextual perception, reasoning, crossepisode memory, and hierarchical coordination while deliberately keeping probabilistic LLM inference outside the fast actuation path.

• Analytical guarantees on the value of anticipation. We derive and prove a sufficient minimum anticipation horizon under delayed capacity provisioning that guarantees the required capacity is online before a fast-onset surge and prevents backlog formation. We further prove that, when $\lambda _ { s } > c _ { \mathrm { e v } } \mu _ { \mathrm { e f f } }$ and $\tau _ { \mathrm { p r o v } } > 0 ,$ no purely reactive capacity policy can avoid a strictly positive backlog during the provisioning ramp, irrespective of its optimization objective or controller weights. This identifies operating conditions under which only anticipation, informed by external context, can prevent service loss.

• A signaling-storm resilience model and discrete-event AI-RAN simulator. We formulate resilient control under benign and malicious signaling surges and develop a discrete-event simulator<sup>1</sup> that jointly captures retrydriven amplification, shared-compute contention, and non-instantaneous capacity provisioning. Performance is scored through service quality, capacity efficiency, and the anticipation, absorption, adaptation, and recovery over time (A3RT) resilience metric.

• An end-to-end evaluation of agentic reasoning and orchestration. We evaluate several LLMs as the Non-RT agent, and compare the proposed approach with static, reactive Lyapunov, and rule-based controllers. We test reserve sizing on real scheduled events, show how provisioning delay and hidden compute contention limit every controller, and evaluate free-text intent grounding, compounded stressors, and multi-site orchestration. Throughout, we report resilience and benign attached rate alongside capacity efficiency, inference latency, and monetary cost, and we analyze the signaling overhead the agent adds.

Organization. The remainder of the paper develops the signaling-storm study and simulator (Section II), presents the three-tier agentic framework (Section III), derives the analytical limits of provisioning (Section IV), evaluates the framework (Section V), and concludes (Section VI).

## II. SIGNALING-STORM SYSTEM MODEL AND RESILIENCE FORMULATION

As a running study, we adopt a mass-attachment signaling storm in AI-RAN. It fits the purpose because it spans the full resilience lifecycle. Unusual signaling patterns enable anticipation, containing excess or malicious load supports absorption, adjusting processing capacity enables adaptation, draining the backlog constitutes recovery, and improving the response across repeated storms reflects evolution.

## A. Scenario, Threat Model, and Service Rate

We consider a mass-attachment scenario in which many user equipments (UEs) attempt to attach within a short interval, as in large-scale reconnection events or adversarial activity. We focus on UE initial attachment, since each attachment triggers a fixed exchange of Radio Resource Control (RRC) signaling and per-UE processing at the base station. The random-access preamble and response stay local to the radio unit (RU) and distributed unit (DU) below RRC, while the RRC Setup Request, Setup, and Setup Complete give $M \ : = \ : 3$ controlplane messages that must reach the central unit (CU) where RRC terminates [19].

We model attachment as an $M / M / c$ system of c parallel servers at service rate $\mu$ with arrivals at rate $\lambda ( t )$ , stable only while $\lambda ( t ) < c \mu$ . We define a signaling storm as an interval $[ t _ { 0 } , t _ { d } ] _ { : }$ , where $t _ { 0 }$ and $t _ { d }$ denote the storm onset and termination times, respectively, such that

$$
\lambda ( t ) > c \mu \quad \mathrm { s u s t a i n e d ~ o v e r } [ t _ { 0 } , t _ { d } ] ,\tag{1}
$$

so requests accumulate faster than they get served.

Threat model. The surge in attachment requests may be non-malicious, arising from mass reconnection after an outage or a large gathering, or adversarial. We accordingly split the arrival rate into a benign and a malicious component, $\lambda ( t ) = \lambda _ { \mathrm { b e n } } ( t ) + \lambda _ { \mathrm { m a l } } ( t )$ . In the adversarial case, an attacker drives compromised or bot-driven UEs that flood the control plane with legitimate-looking attach requests to deny service, without otherwise breaking the signaling protocol order. Benign surges must be absorbed and recovered from, whereas the malicious component can be identified and rate-limited at admission, motivating distinct responses. Flagging which arrivals are malicious is the task of an anomaly or intrusion detector, which we take as given and out of scope [12], so we study the response rather than the detection. A deployed detector is imperfect. We therefore assume that a fraction $\alpha _ { \mathrm { f p } }$ of legitimate attach attempts is wrongly flagged as malicious, so whenever the filter is active, these are dropped along with the botnet traffic.

Crowd-to-load mapping. For the scheduled-event surges used in the evaluation, we relate a real crowd to a controlplane load. An event of attendance N does not attach as one instantaneous burst. Only a fraction $\phi$ of attendees are active subscribers on the modeled cell, and their attempts spread over a short reconnection window $T _ { \mathrm { r w } }$ around the event boundary, giving

$$
\lambda _ { \mathrm { b e n } } ^ { \mathrm { r w } } = \frac { \phi N } { T _ { \mathrm { r w } } } .\tag{2}
$$

With $\phi \approx 0 . 2$ and $T _ { \mathrm { r w } } \approx 6 0 ~ \mathrm { s }$ this calibrates to $\lambda _ { \mathrm { b e n } } ^ { \mathrm { r w } } \approx N / 3 0 0$ so a crowd of 83,664 maps to about 279 UEs/s. This is roughly 60% of the full-pool capacity $c _ { \operatorname* { m a x } } \mu \approx 4 5 9$ UEs/s $( c _ { \mathrm { m a x } } =$ 16, with $\mu$ from (3)), but far beyond what the few servers online at rest can serve, so it constitutes a storm. Because the same mapping converts both the true attendance and every controller’s estimate into load, its constant does not favor any controller and only sets the load scale, which we fix so that the largest events approach $c _ { \operatorname* { m a x } } \mu .$

Service rate. The service rate $\mu$ follows from the open interfaces and per-node processing. Each of the M RRC messages crosses the disaggregated fronthaul and midhaul with a one-way delay $D _ { \mathrm { o n e w a y ~ } } = \ D _ { t } + D _ { p } + D _ { q } + D _ { r } ,$ the transmission, propagation, queuing, and processing delays along the O-RAN open fronthaul (O-FH) and the F1 controlplane (F1-C) midhaul. Combining this transport across all M messages with the internal CU/DU processing $\begin{array} { r } { T _ { p } = \sum _ { i = 1 } ^ { M } t _ { p , i } } \end{array}$ gives the mean attachment service time and per-server service rate

$$
\mu = \frac { 1 } { T _ { s } } , \qquad T _ { s } = T _ { p } + M D _ { \mathrm { o n e w a y } } .\tag{3}
$$

The O-FH, based on the O-RAN 7.2x split, falls in the ∼100 µs class [20], and the split-option-2 F1-C carries a 1.5–10 ms one-way budget [21] whose lower bound, added to the O-FH delay, gives $D _ { \mathrm { o n e w a y } } = 1 . 6$ ms, the delay that disaggregation adds on top of $T _ { p }$

## B. Discrete-Event Storm Dynamics

The $M / M / c$ model of Section II-A fixes arrivals externally and holds the service rate constant, so it misses three effects that dominate real storms and motivate a discreteevent model. The first is retry-driven amplification, where timed-out attempts re-enter as fresh load. The second is loaddependent service time, as the shared compute pool slows under contention. The third is non-instantaneous capacity provisioning, as new servers take time to come online. With all three disabled, the discrete-event model reduces to the analytical rate (3), so the two agree in the baseline case.

Retry-driven amplification. Each benign attach attempt is governed by an RRC connection-setup guard timer $T _ { 3 0 0 }$ (set at 1000 ms). An attempt not served within $T _ { 3 0 0 }$ is abandoned and retried after a random backoff (up to 500 ms), up to K attempts (set to 5 in the analysis), after which the UE fails to attach. A botnet UE is more aggressive, abandoning and reattaching on a much shorter impatient period $T _ { b } \ll T _ { 3 0 0 }$ , so its attempts accumulate far faster and dominate the amplification. Let $n ( t ) \in [ 1 , K ]$ be the average number of attempts a UE makes before it attaches or exhausts its retries. Since each UE generates n(t) attempts rather than one, the effective load is

$$
\lambda _ { \mathrm { e f f } } ( t ) = n ( t ) \lambda ( t ) .\tag{4}
$$

Under light load $n ( t ) \ \approx \ 1$ and $\lambda _ { \mathrm { e f f } } \approx \lambda$ , whereas under overload, attempts wait past $T _ { 3 0 0 }$ and retry, so $n ( t ) \approx K$ and $\lambda _ { \mathrm { e f f } } \approx K \lambda$ . The result is a self-reinforcing cycle in which longer queues cause more timeouts, and more retries lengthen the queues further.

Shared-compute contention. In AI-RAN, the virtualized CU (vCU) and virtualized DU (vDU) processing runs on a finite compute pool shared with other workloads, so its speed depends on the pool load. Let b be the number of busy attachworkers in a pool of size $\kappa ~ = ~ c _ { \mathrm { m a x } }$ , giving the compute occupancy $\omega = b / \kappa ,$ the fraction of the pool in use. A shared pool runs undisturbed while it has headroom and only contends once heavily loaded, so the processing time is the same up to a threshold $\omega _ { \mathrm { t h } }$ and then rises linearly,

$$
s ( \omega ) = \left\{ \begin{array} { l l } { 1 , } & { \omega \leq \omega _ { \mathrm { t h } } , } \\ { \displaystyle 1 + a \frac { \omega - \omega _ { \mathrm { t h } } } { 1 - \omega _ { \mathrm { t h } } } , } & { \omega > \omega _ { \mathrm { t h } } , } \end{array} \right.\tag{5}
$$

with severity $^ { a , }$ so the slowdown is bounded by $1 + a$ rather than diverging. We assume that only the processing part of the service time is affected, while the transport delay $M D _ { \mathrm { o n e w a y } }$ stays fixed,

$$
T _ { s } ^ { \mathrm { e f f } } = s ( \omega ) T _ { p } + M D _ { \mathrm { o n e w a y } } , \qquad \mu _ { \mathrm { e f f } } = \frac { 1 } { T _ { s } ^ { \mathrm { e f f } } } .\tag{6}
$$

Below the threshold $s = 1$ and $\mu _ { \mathrm { e f f } } = \mu ,$ recovering the uncontended rate, and above it each busy worker slows every attach a little more. Because the pool size is $\kappa = c _ { \mathrm { m a x } } .$ , contention begins only when most servers are busy $( \omega _ { \mathrm { t h } } = 0 . 6 5 )$ , which happens when the controller provisions heavily. A controller that plans with the nominal $\mu$ therefore assumes each server is faster than it is and under-provisions precisely when it is relying on that capacity most.

Capacity provisioning. A controller commands a target capacity $c ^ { \star } ( t )$ , but the online capacity $c ( t )$ tracks it with delay. Bringing additional attachment-processing capacity into service takes $\tau _ { \mathrm { p r o v } }$ seconds and is assumed to be added serially. Scaling down is immediate, as a server simply stops taking new work without interrupting any in-flight attach. Thus

$$
c ( t ) = \left\{ { \begin{array} { l l } { c ^ { \star } ( t ) , } & { c ^ { \star } ( t ) \leq c ( t - 1 ) , } \\ { c ( t - 1 ) + 1 { \mathrm { ~ p e r ~ } } \tau _ { \mathrm { p r o v } } , } & { c ^ { \star } ( t ) > c ( t - 1 ) . } \end{array} } \right.\tag{7}
$$

This delay is what gives anticipation its value. With $\tau _ { \mathrm { p r o v } } \ >$ 0, capacity added only after a surge begins arrives too late to prevent the backlog, so provisioning ahead of the surge materially changes the outcome.

Traffic and queue evolution. Arrivals follow a specific profile, λ(t) rises from a calm baseline during a storm, as a step or a ramp, and then returns to it. With $L _ { q } ( t )$ the number of attempts waiting for a server, the queue evolves as

$$
L _ { q } ( t + 1 ) = \operatorname* { m a x } \Bigl ( 0 , L _ { q } ( t ) + \lambda _ { \mathrm { e f f } } ( t ) - c ( t ) \mu _ { \mathrm { e f f } } ( t ) \Bigr ) ,\tag{8}
$$

which brings together the retry-amplified load $\lambda _ { \mathrm { e f f } } ( t )$ , the contention-degraded rate $\mu _ { \mathrm { e f f } } ( t )$ , and the delayed capacity c(t). Equation (8) is the fluid average, which the simulator realizes stochastically by modeling each UE and its retries.

## C. Resilience Quantification

To score system health over a storm we define a utility function $u ( t ) \in [ 0 , 1 ]$ , reported as a percentage, where higher values indicate a more stable, better-served system. It combines a capacity-utilization term and a queue-length term, each a logistic (sigmoid) function that stays near 100% under light conditions and falls toward zero as the system is stressed,

$$
\begin{array} { c c } { { u ( t ) = w _ { A } u _ { A } ( t ) + w _ { B } u _ { B } ( t ) , } } & { { w _ { A } + w _ { B } = 1 , } } \\ { { \displaystyle } } & { { \displaystyle } } \\ { { u _ { A } ( t ) = \frac { 1 } { 1 + e ^ { k _ { A } ( \rho ( t ) - m _ { \mathrm { f r a c } A } ) } } , } } & { { \rho ( t ) = \frac { \lambda ( t ) } { c ( t ) \mu } , } } \\ { { \displaystyle } } & { { \displaystyle } } \\ { { u _ { B } ( t ) = \frac { 1 } { 1 + e ^ { k _ { B } ( L _ { q } ( t ) - m _ { B } ) } } , } } & { { m _ { B } = L _ { q , \operatorname* { m a x } } m _ { \mathrm { f r a c } B } , } } \end{array}\tag{9}
$$

where $\rho ( t )$ is the utilization, the ratio of offered load $\lambda ( t )$ to online serving capacity $c ( t ) \mu ,$ so $\rho < 1$ indicates spare capacity and $\rho ~ \geq ~ 1$ saturation. The constants $k _ { A } , k _ { B }$ set the steepness of each transition and $m _ { \mathrm { f r a c } A } , m _ { B }$ their knees. $u _ { A }$ passes through 0.5 at the utilization knee $\rho = m _ { \mathrm { f r a c } A } ,$ retaining near-full utility while the cell runs comfortably below capacity and falling toward zero as it approaches and exceeds saturation, while $u _ { B }$ falls as the queue nears its overflow threshold $L _ { q , \mathrm { m a x } } .$ . Defining $u _ { A }$ through utilization rather than the raw arrival rate keeps it independent of pool size. The same parameters $( k _ { A } , m _ { \mathrm { f r a c } A } )$ therefore apply whether few or many servers are online, whereas a raw-rate form would need retuning as $c ( t )$ changes. Utility thus measures load relative to current capacity, staying near 100% under normal load and degrading as the cell nears saturation or the queue builds.

We quantify resilience $P \in [ 0 , 1 ]$ , reported as a percentage, over a disruption with the A3RT metric [9], which weights the utility retained during absorption and adaptation against a desired profile $u _ { \mathrm { d e s } }$ and adds a recovery-time term,

$$
P = w _ { 1 } \frac { \int _ { t _ { 0 } } ^ { t _ { d } } u d t } { \int _ { t _ { 0 } } ^ { t _ { d } } u _ { \mathrm { d e s } } d t } + w _ { 2 } \frac { \int _ { t _ { d } } ^ { t _ { r } } u d t } { \int _ { t _ { d } } ^ { t _ { r } } u _ { \mathrm { d e s } } d t } + w _ { 3 } t _ { \mathrm { r e c } } ,\tag{10}
$$

$$
t _ { \mathrm { r e c } } = \left\{ \begin{array} { l l } { 1 , } & { t _ { r } - t _ { 0 } \leq \Delta t _ { \mathrm { d e s } } , } \\ { \displaystyle \frac { \Delta t _ { \mathrm { d e s } } } { t _ { r } - t _ { 0 } } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right. \quad \quad \sum _ { i } w _ { i } = 1 ,\tag{11}
$$

where $[ t _ { 0 } , t _ { d } ]$ is the storm interval and $t _ { r }$ the recovery time, the first instant after $t _ { d }$ at which utility returns above a fraction $\eta _ { \mathrm { r e c } }$ of $u _ { \mathrm { d e s } }$ and holds there for at least $\Delta t _ { \mathrm { h o l d } } \left( \eta _ { \mathrm { r e c } } = 0 . 9 5 , \Delta t _ { \mathrm { h o l d } } = \right.$ 30 s). We set $u _ { \mathrm { d e s } }$ to the mean pre-storm utility measured while the cell operates at its steady resting capacity, excluding any pre-provisioning ramp, so that P measures resilience relative to the system’s own healthy baseline rather than a transiently over-provisioned state.

## D. Controllers

The dynamics of Section II-B expose two actuation points, the commanded capacity $c ^ { \star } ( t )$ and the malicious-drop probability $p _ { \mathrm { d r o p } }$ . We introduce three baseline controllers of increasing capability as reference points for the framework of Section III. The first two act on capacity alone, the third also actuates the admission filter through fixed rules.

Static. The server count is held constant, $c ^ { \star } ( t ) = c$ for a chosen $c \in \{ 1 , \ldots , c _ { \operatorname* { m a x } } \}$ . Using no feedback, it shows how a statically provisioned system copes with a storm.

Reactive. A feedback controller resizes the pool at every control step, balancing queue growth against the cost of idle servers through a Lyapunov drift-plus-penalty objective $J ( c )$

With $a _ { n } = \lambda ( t ) / \mu - c$ the unmet demand in servers and $q _ { n } =$ $L _ { q } ( t ) / L _ { q , \mathrm { { m a x } } }$ the normalized backlog, it selects

$$
\begin{array} { r l } & { c ^ { \star } ( t ) = \arg \underset { 1 \leq c \leq c _ { \mathrm { m a x } } } { \operatorname* { m i n } } J ( c ) , } \\ & { J ( c ) = q _ { n } a _ { n } + \frac { 1 } { 2 } a _ { n } ^ { 2 } - V \ * u _ { c } ( t ) + W \frac { c } { c _ { \mathrm { m a x } } } , } \end{array}\tag{12}
$$

where $u _ { c } ( t )$ is the utility of (9) evaluated at c. The drift terms $q _ { n } a _ { n }$ and ${ \textstyle { \frac { 1 } { 2 } } } a _ { n } ^ { 2 }$ push toward more servers when the backlog is long or arrivals outpace capacity, $- V u _ { c } ( t )$ rewards high utility, and $W c / c _ { \mathrm { m a x } }$ discourages over-provisioning. Both terms lie in [0, 1], so $V$ and $W$ are directly comparable and default to 1.

Rule-based. Our strongest baseline replaces the agent’s language-model policy with a fixed decision tree over the same telemetry, isolating the value of language-model reasoning. Like the agentic framework, it runs on two timescales. A slow rule periodically sets the policy, and the reactive update (12) applies it at every control step. The rule flags the cell as elevated when $\lambda ( t )$ rises clearly above its resting baseline and, lacking a calendar, treats every elevated load as malicious. It then raises V for headroom and engages the filter with a drop probability that grows with overload severity, hardening if utility keeps falling. It can also raise V early when the forecast extrapolates an already rising trend. This is close to the best that hand-tuned rules can achieve on this telemetry, but its limit is structural, since it responds only once arrivals begin to climb, never before.

Admission filtering is therefore imperfect. Dropping a fraction $p _ { \mathrm { d r o p } }$ of malicious attempts also removes benign traffic at rate $\alpha _ { \mathrm { f p } } p _ { \mathrm { d r o p } } ,$ so containment trades off against denying legitimate users, a cost the rule pays on every benign surge it cannot recognize. The framework of Section III addresses exactly these gaps, separating a scheduled surge from an attack and provisioning before the queue grows.

## E. Metrics

Per-window resilience. A run may contain several storm windows of different character, so we evaluate $P$ of (10) over each window $[ t _ { 0 } ^ { ( k ) } , t _ { r } ^ { ( k ) } ]$ separately and report the episode resilience as the mean of the per-window scores, $P =$ $\begin{array} { r } { \frac { 1 } { N _ { w } } \sum _ { k = 1 } ^ { N _ { w } } P ^ { ( k ) } } \end{array}$ , so a controller cannot hide a starved window behind a healthy one. The evaluation scenario pairs a malicious botnet ramp with a benign event surge, giving two scores $P _ { \mathrm { b o t } }$ and $P _ { \mathrm { s u r g e } }$ that reward opposite responses, filtering the attack and provisioning for the crowd.

Service quality. Resilience rewards utility held but not how many users were served. Each legitimate UE ends up attached, failed after exhausting its retries, or dropped at admission, with counts $A _ { b } , F _ { b } ,$ and $D _ { b } ,$ giving the benign attached and falsepositive rates

$$
S _ { b } = \frac { A _ { b } } { A _ { b } + F _ { b } + D _ { b } } , \qquad \phi _ { b } = \frac { D _ { b } } { A _ { b } + F _ { b } + D _ { b } } .\tag{13}
$$

When the surge is entirely benign, $\phi _ { b }$ measures pure overfiltering. For the attack we report the filtered fraction, the share

of botnet UEs deliberately dropped out of all botnet outcomes $A _ { m } , F _ { m }$ , and $D _ { m }$

$$
\Phi _ { m } = \frac { D _ { m } } { A _ { m } + F _ { m } + D _ { m } } ,\tag{14}
$$

which counts only admission drops, so a controller that never filters stays at zero however overloaded.

Capacity cost and efficiency. Resilience can always be bought with capacity, so we track its cost. With c¯ the mean number of servers online, the capacity efficiency

$$
E = \frac { P } { \bar { c } / c _ { \mathrm { m a x } } } ,\tag{15}
$$

is the resilience per unit of provisioned capacity, normalized so $E = 1$ marks a controller that holds all $c _ { \mathrm { m a x } }$ servers to reach $P = 1$ . Since it rewards economy, an under-provisioned low-P controller can also score high, so we read E only alongside P.

Agentic overhead and grounding. For the agentic controllers, we also record the per-decision latency and monetary cost of the language model, and we compare the agent’s estimated event attendance against ground truth to confirm that its provisioning is grounded rather than fortunate.

## III. THE PROPOSED AGENTIC RESILIENCE FRAMEWORK

Realizing resilience as a continuous lifecycle rather than a set of stage-specific mechanisms requires a framework that binds those stages into one coordinated cycle and carries what is learned from each disruption into the next. Agentic AI is a natural candidate, but placing LLM-based agents inside the real-time or Near-RT control loops adds signaling and decision latency exactly when the network is overloaded. We therefore propose an agentic resilience framework that coordinates the lifecycle across coupled loops running at different timescales while keeping slow, probabilistic LLM reasoning off the fast control path. Following the O-RAN control hierarchy, the framework spans three tiers, shown in Fig. 1. An SMO-level orchestrator turns operator intent into standing policy, a persite Non-RT agent judges storm conditions, and a deterministic Near-RT loop enforces those decisions at every control step.

## A. Near-RT Control Loop

The Near-RT loop is the framework’s deterministic actuator, detailed in Algorithm 1. It runs at a fixed control cadence and, at every step, reads the current attach arrival rate $\lambda ( t )$ , the queue length, and the number of servers online, together with an atomic snapshot of the current policy, which holds the storm verdict, the malicious drop probability, and the controller weights the agent may have set. From these, it solves the driftplus-penalty objective (12) for the commanded capacity $c ^ { \star } ( t )$ This command passes through two guardrails. It is clamped to the operator capacity floor and the physical maximum, and the loop refuses to scale down while the queue is still draining, which prevents it from oscillating during recovery. In parallel it applies the malicious drop probability $p _ { \mathrm { d r o p } } ,$ gated by the agent’s storm verdict, so admission filtering is active only when a storm is declared. Crucially, no language model sits on this path. The loop actuates on the last policy it received and never waits for the agent to respond, keeping decision latency off the fast control path even when the agent is slow or the network is overloaded. Each step is also computationally light, since every evaluation of $J ( c )$ in (12) is $\mathcal { O } ( 1 )$ , the capacity search costs $\mathcal { O } ( c _ { \mathrm { m a x } } )$ , and the optional signature lookup in the cross-episode memory $\mathcal { B }$ costs $\mathcal { O } ( \vert \mathcal { B } \vert d _ { \mathrm { s i g } } )$ for $d _ { \mathrm { s i g } }$ -dimensional signatures, giving $\mathcal { O } ( c _ { \mathrm { m a x } } + \vert \mathcal { B } \vert d _ { \mathrm { s i g } } )$ time per control period and $\mathcal { O } ( \vert \mathcal { B } \vert d _ { \mathrm { s i g } } )$ memory. With $c _ { \mathrm { m a x } }$ and |B| small in practice, the loop comfortably fits within its control period $T _ { \mathrm { c t l } }$ , which lies between 10 ms and 1 s. As shown in Fig. 1, an xApp applies admission filtering over E2, while the Near-RT loop, which alone observes the persecond queue and arrival telemetry, passes $c ^ { \star } ( t )$ to the SMO, which activates the servers on the O-RAN cloud (O-Cloud) over O2 [22]. The resulting instantiation delay is captured by $\tau _ { \mathrm { p r o v } } ,$ which we sweep in Section V-D.

![](images/889a9fed33f38c37e5721668d5165bb913b108e15e2810d7cb3916e3f1a357c7.jpg)

Fig. 1. The Proposed Agentic Resilience Framework Architecture for the AI-RAN.  
```latex
Algorithm 1 Near-RT Control Loop.
Input: λ(t), λ<sub>rest</sub>, $L _ { q } ( t ) ,$ , c(t), π, τ<sub>prov</sub>, $\beta _ { \mathrm { r e l } } .$ , optional cross-episode
memory B
Output: commanded capacity $c ^ { \star } ( t )$ , malicious-drop probability $p _ { \mathrm { d r o p } } ( t )$
1: while episode active do ▷ every $\dot { T } _ { \mathrm { c t l } }$
2: π˜ ← atomic snapshot of π
▷ Reserve timing: online BY the event, never earlier
3: T<sub>ramp</sub> ← (˜π.c<sub>res</sub> − 1) τ<sub>prov</sub> ▷ T <sup>min</sup> of (17) at $c _ { \mathrm { e v } } { = } 1$
4: due $\begin{array} { r } { \stackrel {  } {  } ( \tilde { \pi } . t _ { \mathrm { e v } } \leq 0 ) \mathrm { o r } ( t \geq \tilde { \pi } . t _ { \mathrm { e v } } - T _ { \mathrm { r a m p } } - \stackrel { \mathrm { a n } } { T } _ { \mathrm { c t l } } ) } \end{array}$
5: over $ ( \tilde { \pi } . t _ { \mathrm { e v } } > \tilde { 0 } )$ and $\left( t > \tilde { \pi } . t _ { \mathrm { e v } } \right)$ and $\left( \lambda ( t ) \leq \beta _ { \mathrm { r e l } } \lambda _ { \mathrm { r e s t } } \right)$
6: $c _ { \mathrm { r e s } } ^ { \mathrm { e f f } }  \tilde { \pi } . c _ { \mathrm { r e s } }$ if (due and not over) else 1
7: c ← max $\{ \tilde { \pi } . c _ { \mathrm { f l o o r } } , \ c _ { \mathrm { r e s } } ^ { \mathrm { e f f } } \}$ ▷ operator floor ∪ scheduled reserve
▷ Capacity command under the current posture
8: $c ^ { \star } \gets \arg \operatorname* { m i n } _ { 1 \leq c \leq c _ { \operatorname* { m a x } } } J ( c )$ under π.V,˜ π.W˜
9: c<sup>⋆</sup> ← min{max{c<sup>⋆</sup>, c}, c<sub>max</sub>}
10: if $L _ { q } ( t ) > \tilde { \pi } . L _ { \mathrm { h o l d } }$ then
11: $c ^ { \star } \gets$ max $\boldsymbol { \mathbf { \ell } } \cdot \{ \boldsymbol { c } ^ { \star } , \boldsymbol { c } ( t ) \}$ ▷ no scale-down while the queue drains
12: end if
▷ Admission filter, gated by the storm verdict
13: if π.˜ storm then
14: $p _ { \mathrm { d r o p } }  \tilde { \pi } . p _ { \mathrm { d r o p } }$
15: else if B enabled and the live signature matches an entry then
16: p<sub>drop</sub> $ \mathcal { B } . p _ { \mathrm { d r o p } }$ ▷ engage from memory before the agent rules
17: else
18: $p _ { \mathrm { d r o p } }  0$
19: end if
20: actuate $c ^ { \star } ; c ( t )$ tracks it with delay $\tau _ { \mathrm { p r o v } }$ under (7)
21: drop each flagged arrival with probability p<sub>drop</sub>
22: end while
23: on episode end do store the storm signature, its engagement level, the
containing drop strength, and the tuned weights in B
```

## B. Non-RT Agent

The Non-RT agent is the per-site decision maker, summarized in Algorithm 2. Its input is the telemetry that the Near-RT loop of Algorithm 1 produces at every control step, and its output is a shared policy object that the same loop reads back at its next step. The two algorithms therefore form a closed loop, with Algorithm 2 deciding and Algorithm 1 enforcing. It runs on its own schedule every few seconds, apart from the fast control path, and it never acts on the network directly. At each assessment, it receives a summary of the recent telemetry together with its own previous verdict, shown in Fig. 2. The summary reports how the arrival rate, the queue, and the retry rate have moved over the last window. Each signal is also tagged as leading or lagging. Arrivals and retries rise at storm onset, whereas the queue reacts only afterwards and is held down by the fast loop, so the agent should base its judgment on the leading signals. Feeding the previous verdict back keeps the decision steady and stops the agent from switching the storm state on and off on brief noise. To look beyond raw arrivals, the agent can also call a small set of tools over the Model Context Protocol (MCP) [23] that return the live episode logs, a short forecast of the arrival rate, and a calendar of planned events.

From this input, the agent returns one structured decision, shown in Fig. 3. In this example, arrivals are calm and no storm is present, but a call to the calendar reveals an upcoming event, so the agent estimates the crowd and the time of its surge, and the system turns these into a reserve provisioned just in time. When a storm is present instead, the same schema carries the storm verdict, a drop probability scaled to the overload severity, and controller weights that add capacity headroom.

Because the agent only decides while the deterministic loop enforces, its slow and uncertain reasoning shapes the control posture without ever reaching the Near-RT actuation path.

## C. Agent Tools

Left to telemetry alone, the agent sees only arrivals and cannot tell a malicious flood from a legitimate surge, nor act before load arrives. Three read-only tools close these gaps.

• Log tool (σ). It reports the live episode, returning cumulative service and security counters together with a running measure of how well utility is being held against its calm pre-storm baseline. This measure is computed only from the telemetry seen so far, with no knowledge of the future, and it tells the agent whether its current filtering is coping or needs to push harder.

• Forecast tool (λ<sup>ˆ</sup>). It returns a short prediction of where load is heading over the next few control steps, projecting the arrival rate along with the retry rate and the queue.

Algorithm 2 Non-RT Agent (site-local, period $T _ { \mathrm { a s m } } ,$ asyn  
chronous).   
Input: telemetry window H, previous policy $\pi ^ { - } .$ , π.instr   
Output: candidate policy $\pi ^ { + }$ , published into π   
1: while TRUE do ▷ every $T _ { \mathrm { a s m } }$   
2: extract arrival, retry, and queue trajectories from $\mathcal { H }$   
3: recall the verdict in $\pi ^ { - }$ to keep storm state steady   
4: σ ← LOGTOOL( ), λ<sup>ˆ</sup> ← FORECASTTOOL( ), E ←   
CALENDARTOOL( )   
▷ Anticipation: commit a reserve once for an upcoming event   
5: if E names an upcoming event not yet provisioned then   
6: N<sup>ˆ</sup> ← attendance inferred from the venue and context in $\mathcal { E } ^ { \mathcal { O } }$   
7: $\pi ^ { + } . c _ { \mathrm { r e s } } $ reserve holding N<sup>ˆ</sup> below the target utilization   
8: $\pi ^ { + } . t _ { \mathrm { e v } } \gets \mathcal { E } . t$ ▷ fast loop brings it online just in time   
9: end if   
▷ Classification: judge the current load, honoring any delegation   
10: if the latest arrival rate sits clearly above rest then   
11: if an event in E is due now $( t \overset { \cdot } { \geq } \mathcal { E } . t ) ,$ or π.instr marks this load   
benign then   
12: π<sup>+</sup>.storm ← FALSE, π<sup>+</sup>.p<sub>drop</sub> ← 0   
13: else   
14: π<sup>+</sup>.storm $ \mathrm { T R U E } , \pi ^ { + } . p _ { \mathrm { d r o p } }$ ← scaled to overload severity   
15: $\pi ^ { + } . V $ raised for capacity headroom   
16: end if   
17: else   
18: π<sup>+</sup>.storm ← FALSE; return V toward default once load is at rest   
19: end if   
20: write $: \pi ^ { + }$ into π atomically, operator overrides left in force   
21: end while

Non-RT Agent Input Assessment #12, t = 200 s   
• Current policy (updated 5 s ago):   
storm\_active = false   
malicious\_drop\_prob = 0.00   
reserve\_servers = 1   
queue\_hold\_threshold = 10   
lyapunov\_V = lyapunov\_W = 1.0   
• Telemetry (last 15 s, 16 samples):   
Arrival rate λ: 21 → 22 UEs/s (flat)   
peak λ: 25 at t=193 s (leading)   
λ baseline: ≈20 UEs/s   
λ bin means: 21, 20, 22, 21, 20, 22, 21, 22   
Retry rate: 3/s → 3/s (flat) (leading)   
Queue length: 0 → 0 (flat) (lagging)   
Task   
• “Judge storm-vs-noise from the window trends and   
return a PolicyUpdate.”  
Fig. 2. Non-RT agent input at a single assessment.

The prediction is a least-squares trend fitted to the recent telemetry window, with the fit quality reported as a confidence score so the agent can discount a poor fit. When the trend is rising, the agent can raise the headroom weight V a few cycles early, so the Near-RT loop begins adding servers before the queue builds. The forecast thus complements the calendar by flagging unscheduled ramps as they emerge, while reserves for scheduled crowds remain the calendar’s responsibility.

• Calendar tool (E ). It exposes scheduled load events that the SMO orchestrator has registered or a calendar subagent has scraped from the web, such as a stadium egress or a planned mass registration. Each event is named with its venue and whether it sold out, together with the time its surge is expected, but never the turnout. The agent estimates the crowd itself from the event and venue, and the system turns that estimate into a reserve provisioned ahead of the surge. Because the surge is on the calendar, the agent classifies it as benign and admits it without filtering, so no legitimate user is dropped during the event.

Algorithm 3 SMO Orchestrator (event-driven).   
Non-RT Agent Output Input: operator intent I (free text), site policy π   
• Storm state: Output: updated overrides in π, calendar entry for a scheduled event,   
storm\_active = false standing instruction π.instr   
malicious\_drop\_prob = 0.00 1: procedure ONINTENT(I) ▷ triggered on intent arrival   
• Control parameters: 2: d ← GROUND(I) ▷ map intent to directive over the five levers   
3: if d.posture ̸= ∅ then   
tighten = true   
4: (π.V, π.W) ← POSTUREWEIGHTS(d.posture)   
lyapunov\_V = lyapunov\_W = 1.0   
5: end if   
queue\_hold\_threshold = 10 6: if d.floor ̸= ∅ then   
• Expected load: 82,000 attendees at t = 240 s. 7: π.c<sub>floor</sub> ← d.floor   
• Rationale: “England v Brazil at Wembley, a 90k-seat 8: end if   
venue, not sold out; a marquee fixture drawing a large 9: if d.event ̸= ∅ ∧ d.event.t ̸= ∅ then   
but sub-capacity crowd.” 10: CALENDAR.REGISTER(d.event)   
11: end if   
Reasoning   
12: if d.instr ̸= ∅ then   
• “Load is at rest with no storm. A scheduled surge is 13: π.instr ← d.instr ▷ delegate to Non-RT agent   
due at t = 240 s, so pre-provision a reserve for the 14: end if   
estimated crowd.” 15: end procedure  
Fig. 3. Non-RT agent output for pre-event provisioning.

## D. Shared Policy

The two loops never call each other. They communicate through a single shared policy object that the agent writes and the Near-RT loop reads, which lets them run at different speeds. The agent updates the policy whenever it finishes an assessment, while the fast loop reads a complete copy of it at every control step and never sees a half-written update. The policy holds four kinds of fields. The first are the operational levers, the storm verdict and the malicious drop probability, which are refreshed on every assessment. The second are the slower controller settings, the utility and cost weights and the queue-hold threshold, which change only when the agent decides to retune, so the controller does not keep shifting on every cycle. The third is the scheduled reserve, a number of servers to hold online ahead of a planned surge together with the time the surge is due, which the agent commits once for an event and the fast loop brings online just before it starts. The fourth are operator overrides set by the SMO orchestrator, such as a fixed weight posture or a minimum capacity floor, which take priority over the agent’s own tuning. Because the loop reads only this object, it keeps working correctly no matter when, or even whether, the agent updates it. If the agent fails, the loop can fall back to the rule-based controller of Section II-D, which drives the same actuators.

## E. Agent Memory

The mechanisms above let the framework react and anticipate within a single storm. A further layer lets it improve across storms. The fast loop keeps a lightweight memory of each storm’s signature, its baseline rate, the level at which it engages, and the drop strength that contained it. Once a pattern is recognized, it can turn on the admission filter before the agent has ruled. This memory, along with the tuned controller settings, is carried across episodes, so a site that has already weathered a storm starts the next one configured for it rather than from defaults. Only this slow-moving state is kept. The live levers, which are decisions about the present, are always recomputed. Memory is disabled in all experiments of Section V to isolate within-storm behavior, and its evaluation is left to future work.

## F. Agentic AI Orchestrator

The orchestrator is the topmost tier and the interface between the human operator and the sites, shown in Algorithm 3. Its outputs feed both lower tiers. The posture and floor it writes are read by the Near-RT loop of Algorithm 1, while the calendar entry and standing instruction are consumed by the Non-RT agent of Algorithm 2. Sitting at the SMO above the per-site agents, and it is event-driven, staying idle while the network runs normally and activating only when an operator issues an intent, so it adds no steady-state signaling or realtime latency. Its role is to turn free-text operator intents into a single standing directive that the lower tiers apply without further review. The directive carries five levers, the posture direction and an optional graded strength that together set the utility and cost weights, a hard capacity floor, a scheduled load event with its venue, sold-out status, and time, and a standing instruction that tells the site agent how to interpret conditions rather than change capacity. The orchestrator writes the posture and the floor into the shared policy as operator overrides, which take priority over the site agent’s own tuning, registers the named event on the calendar the site agent reads, and leaves the standing instruction for it to apply on every assessment.

Sitting at the SMO above the sites, the orchestrator coordinates one or many of them from a single point. It decomposes one operator intent into heterogeneous per-site directives and dispatches them over the Agent2Agent (A2A) protocol [24], so a scheduled stadium egress raises the reserve only on the cells that serve it while a fleet-wide cost posture applies everywhere, reconciling one global objective with the local conditions of each site. Each site then runs its own Non-RT agent and Near-RT loop against the directive it receives, so coordination stays centralized at the SMO while enforcement stays local and independent.

TABLE I  
NON-RT AGENT LLM COMPARISON OVER FIVE SEEDS (MEAN ± 95% CI).
<table><tr><td>Model</td><td>Overall Resil. (%)  $P \uparrow$ </td><td>Bot Resil. (%)  $P _ { \mathrm { b o t } } \uparrow$ </td><td>Surge Resil. (%)  $P _ { \mathrm { s u r g e } } \uparrow$ </td><td>Benign Attached (%)  $S _ { b } \uparrow$ </td><td>Crowd Est.  $( \times 1 0 ^ { 3 } , \mathrm { t r u e } 8 3 . 7 )$ </td><td>Reserve Servers (srv, ideal 13)</td><td>Bot Filtered (%)  $\Phi _ { m } \uparrow$ </td><td>Mean Capacity ē (srv)</td><td>Latency/assess. (s) ↓</td><td>Cost/assess. ($) ↓</td></tr><tr><td>GPT-5.4-Mini (on)</td><td> $8 9 . 9 \pm 1 . 1$ </td><td>83</td><td>97</td><td> $9 9 . 8 \pm 0 . 1 $ </td><td> $8 2 . 2 \pm 3 . 3$ </td><td> $1 2 . 4 \pm 0 . 7$ </td><td>33 ± 29</td><td>4.1 ± 0.2</td><td> $2 3 . 2 \pm { 3 . 0 }$ </td><td>0.0123</td></tr><tr><td>QWEN3.7-Plus</td><td> $8 7 . 5 \pm 0 . 6$ </td><td>84</td><td>92</td><td> $9 9 . 7 \pm 0 . 0$ </td><td> $6 8 . 0 \pm 8 . 3$ </td><td> $1 0 . 2 \pm 1 . 0$ </td><td>73 ± 2</td><td> $3 . 7 \pm 0 . 1$ </td><td> $5 . 5 \pm 0 . 2$ </td><td>0.0026</td></tr><tr><td>GEMINI-3.1-Flash-Lite</td><td> $8 7 . 4 \pm 1 . 0$ </td><td>84</td><td>91</td><td> $9 9 . 7 \pm 0 . 1 $ </td><td> $6 9 . 0 \pm 1 0 . 2 $ </td><td> $1 0 . 2 \pm 1 . 4$ </td><td>84 ± 3</td><td> $3 . 6 \pm 0 . 2$ </td><td> $3 . 2 \pm 1 . 1$ </td><td>0.0018</td></tr><tr><td>GPT-5.4-Mini (off)</td><td> $8 7 . 1 \pm 2 . 0$ </td><td>84</td><td>91</td><td> $9 8 . 9 \pm 1 . 1$ </td><td> $8 1 . 4 \pm 1 2 . 7$ </td><td> $1 2 . 4 \pm 2 . 1$ </td><td>84 ± 4</td><td> $3 . 6 \pm 0 . 3$ </td><td> $6 . 3 \pm 0 . 7$ </td><td>0.0060</td></tr><tr><td>CLAUDE-HAIKU-4.5</td><td> $7 6 . 7 \pm 6 . 7$ </td><td>84</td><td>70</td><td> $5 0 . 4 \pm 3 4 . 2$ </td><td> $4 8 . 0 \pm 8 . 3$ </td><td> $7 . 4 \pm 1 . 1$ </td><td>77 ± 5</td><td>3.5 ± 0.0</td><td> $5 . 2 \pm 0 . 1$ </td><td>0.0103</td></tr><tr><td>GPT-4o-Mini</td><td> $7 1 . 6 \pm 2 . 6$ </td><td>84</td><td>60</td><td> $3 5 . 4 \pm 3 . 1$ </td><td> $1 4 . 0 \pm 2 7 . 2$ </td><td> $2 . 2 \pm 4 . 3$ </td><td>71 ± 5</td><td> $3 . 0 \pm 0 . 5$ </td><td> $4 . 5 \pm 0 . 7$ </td><td>0.0010</td></tr></table>

Note: (on)/(off) = reasoning enabled/disabled.

## IV. ANALYTICAL LIMITS OF PROVISIONING

The dynamics of Section II-B and the controllers of Section II-D together determine the lead time required for capacity provisioning to absorb a fast-onset surge without backlog. The result below gives this minimum anticipation horizon under the fluid-queue model of (8), which the discrete-event simulator realizes stochastically. Here, $c ( t )$ denotes the number of servers online at time $t , \ \mu _ { \mathrm { e f f } }$ is the effective per-server service rate under shared-compute contention, $c _ { \mathrm { m a x } }$ is the maximum pool size, $\tau _ { \mathrm { p r o v } }$ is the time to bring one new server online, and $L _ { q } ( t )$ is the queue length.

Theorem 1 (Minimum anticipation horizon). Consider a step surge of constant arrival rate $\lambda _ { s }$ beginning at time $t _ { e \nu } ,$ with an empty queue and $c _ { e \nu }$ servers online. Let

$$
c _ { r e q } = \left\lceil \frac { \lambda _ { s } } { \mu _ { e f f } } \right\rceil , \qquad c _ { r e q } \leq c _ { \mathrm { m a x } } ,\tag{16}
$$

and let capacity be added in unit increments every $\tau _ { p r o \nu }$ seconds as in (7). If the controller raises its capacity floor to $c _ { r e q }$ at least

$$
T _ { a n t } ^ { \operatorname* { m i n } } = ( c _ { r e q } - c _ { e \nu } ) ^ { + } \tau _ { p r o \nu } , ( x ) ^ { + } = \operatorname * { m a x } ( x , 0 ) ,\tag{17}
$$

ahead of $t _ { e \nu } ,$ then $c ( t _ { e \nu } ) \geq c _ { r e q }$ and $L _ { q } ( t ) = 0$ throughout the surge.

The proof is given in Supplementary Material, Section III, together with a Remark on Retry Amplification and a Corollary on Reactive Provisioning.

## V. FRAMEWORK ANALYSIS

This section evaluates whether the agentic controller improves resilience, at what cost, and through which mechanisms. All experiments use the simulator of Section II and report the mean over multiple seeds with 95% confidence intervals (CI).

## A. Selecting the Non-RT Agent LLM

Because one Non-RT agent runs per site every few seconds, model selection trades decision quality against latency and cost. We evaluate five efficient-tier LLMs, from the lightweight GPT-4o-Mini and CLAUDE-HAIKU-4.5, through GEMINI-3.1-Flash-Lite, to the stronger QWEN3.7-Plus and GPT-5.4-Mini, each running as the Non-RT agent with full tool use over five seeds. The scenario has two back-to-back spikes of opposite kinds, a malicious botnet ramp that should be filtered and a scheduled public event that should be provisioned for rather than filtered. The calendar returns only a free-text line such as “England v Brazil, Wembley, not sold out”, so the agent must infer the turnout itself (83,664, warranting a 13-server reserve). Temperature is fixed at 0.0, except for GPT-5.4-Mini with reasoning enabled, which uses the provider default.

![](images/b6feb0dede95fd7d2b80eae4102121ef23855a1561ec747e01a1d7c2a77ab80c.jpg)  
Fig. 4. Utility $u ( t )$ over one botnet-event episode for the two agentic controllers. Shaded regions mark the botnet and event windows, horizontal bars the storm-detection delay, and dotted lines the recovery time $t _ { r } .$

Table I shows that the botnet window separates the models weakly, with every model near $P _ { \mathrm { b o t } } = 8 3 \%$ , because the attack floods the cell however well it is filtered. The event window separates them, since the crowd must be inferred before it arrives and weaker models under-provision. GPT-5.4-Mini with reasoning is the strongest $( P = 8 9 . 9 \% , P _ { \mathrm { s u r g e } } = 9 7 \% )$ provisioning near the ideal reserve at the highest latency and cost. Enabling reasoning on the same model acts as a reasoning ablation, raising $P _ { \mathrm { s u r g e } }$ from 91% to 97% at nearly four times the latency. GEMINI-3.1-Flash-Lite is the deployable operating point $( P = 8 7 . 4 \%$ $P _ { \mathrm { s u r g e } } = 9 1 \% )$ at about a seventh of the latency and cost, and it avoids the tool-calling failures that cost QWEN3.7-Plus five assessments. We carry both forward for subsequent experiments.

![](images/655d4f170b2f91e582b6e0ac97e888fcf5638a67d729a5d0a905d601de348069.jpg)  
Fig. 5. Per-controller mean botnet-window $( P _ { \mathrm { b o t } } )$ and event-window $( P _ { \mathrm { s u r g e } } )$ resilience.

B. Resilience Against Static, Lyapunov, and Rule-based Baselines

We compare both agentic controllers against the baselines of Section II-D, with the static baseline at $c \in \{ 1 , 8 , 1 6 \}$ , in the scenario of Section V-A. Common random numbers across five seeds make the comparison paired. These are independent runs from those of Section V-A, so the agentic scores differ from Table I by up to two percentage points, reflecting the runto-run variability of language-model inference even at zero temperature. Fig. 4 traces utility over one episode, Fig. 5 reports $P _ { \mathrm { b o t } }$ and $P _ { \mathrm { s u r g e } } .$ , and Table II reports P and E.

Fig. 4 shows how the two agents differ over time. GEMINI-3.1-Flash-Lite declares the storm 8 s after onset and GPT-5.4-Mini after 29 s, because each verdict waits for an assessment cycle and reasoning lengthens that cycle. In the event window, the order reverses, since GPT-5.4-Mini commits a larger reserve and holds utility higher.

The botnet window discriminates weakly, with $P _ { \mathrm { b o t } }$ between 76% and 99% for all controllers, because the attack saturates the cell regardless of policy. The event window separates them. The surge arrives as a step, and servers come online one at a time, so a controller that reacts only once load is visible is too late. Static c=1, static $\scriptstyle c = 8 .$ , and Lyapunov serve at most 40% of benign users during the event and reach $P _ { \mathrm { s u r g e } } \leq 5 5 \%$ Static c=16 absorbs the surge but holds all servers throughout, so it is the least efficient (E=1.00) and, without filtering, the most exposed to the botnet. The rule-based controller filters 94% of malicious attempts, but, lacking a calendar, it treats the benign surge as an attack, denying 14% of legitimate arrivals and reaching $P _ { \mathrm { s u r g e } } = 5 8 \%$

Both agentic controllers read the event from the calendar, provision ahead of it, and withhold the filter, serving 99.7% of benign arrivals in the event window. GEMINI-3.1-Flash-Lite reaches $P _ { \mathrm { s u r g e } } ~ = ~ 8 9 \%$ and $P ~ = ~ 8 6 \%$ , and GPT-5.4-Mini reaches 98% and 90%. Both approach the fully provisioned pool with 3.5 to 4.2 servers on average, about a quarter of its capacity at three to four times its efficiency. Since the agentic and rule-based controllers share the same actuators, this margin isolates the value of anticipation.

TABLE II  
EPISODE RESILIENCE P AND CAPACITY EFFICIENCY E ON THE BOTNET-EVENT SCENARIO.
<table><tr><td>System</td><td>Overall Resil. (%)  $P \uparrow$ </td><td>Capacity Efficiency E↑(E=1: full pool)</td></tr><tr><td>Static (c=1)</td><td> $6 3 . 9 \pm 0 . 3$ </td><td> $1 0 . 2 2 \pm 0 . 0 5$ </td></tr><tr><td>Static (c=8)</td><td> $6 9 . 7 \pm 0 . 5$ </td><td> $1 . 3 9 \pm 0 . 0 1$ </td></tr><tr><td>Static (c=16)</td><td> $9 9 . 2 \pm 0 . 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Lyapunov</td><td> $6 8 . 4 \pm 0 . 2$ </td><td> $4 . 2 5 \pm 0 . 0 1$ </td></tr><tr><td>Rule-based</td><td> $7 1 . 0 \pm 0 . 5$ </td><td> $4 . 1 0 \pm 0 . 0 4$ </td></tr><tr><td>Agentic (GEMINI-3.1-Flash-Lite)</td><td> $8 6 . 4 \pm 0 . 3$ </td><td> $3 . 9 7 \pm 0 . 0 2$ </td></tr><tr><td>Agentic (GPT-5.4-Mini)</td><td> $9 0 . 3 \pm 0 . 9$ </td><td> $3 . 4 5 \pm 0 . 1 3$ </td></tr></table>

![](images/843fc307a07d2ed4fa9f7a13728e8af3af75b4682e1f60b61645a1bcdbaf74b8.jpg)  
Fig. 6. Estimated versus true attendance for the twelve scheduled events (log scale). The shaded band marks 20%.

## C. Estimating Scheduled Crowds for Reserve Sizing

The calendar helps only if the agent can infer how large an event will be. We test this on twelve real events, from stadium concerts to low-draw league games, where the agent sees the event name, venue, and sold-out status but never the turnout. The estimate is mapped to a reserve that holds the surge below a target utilization, floored at one server and capped at c<sub>max</sub>. A fixed rule cannot do this without a per-venue capacity table, and even then could not separate a near-full friendly from a nearly empty league game. Each event is estimated once per episode over ten seeds.

Fig. 6 shows both agents tracking attendance from about 600 to 90,000 people, with almost every large-event estimate within ±20%. Estimates drift above the band mainly for events that are not sold out, where the agents read the venue size and overestimate a sparse turnout. When wrong, the agents mostly over-provision rather than under-provision, costing a few idle servers rather than dropped users, and for small events this costs nothing, because the reserve is floored at one server, which absorbs crowds up to about 7,000. Even the 600-person fixture, overestimated several times over, receives the ideal single server (Table III). Across events that are not sold out, the mean absolute reserve error is 1.1 servers for GPT-5.4-Mini and 1.6 for GEMINI-3.1-Flash-Lite, never exceeding three, while sold-out events are sized within one server.

## D. Reactive Tuning Under Provisioning Delay

No Lyapunov weight setting recovers what anticipation provides, and Corollary S1 bounds why. We test this in a pure capacity study with the agent and filter removed, leaving only the weights $\dot { V }$ and $W$ and the provisioning delay. Two benign surges share peak and duration but differ in onset, a step and a 30 s ramp. We sweep $V \in \{ 1 , 1 0 , 2 0 \}$ at $W { = } 1$ and the provisioning delay over {0, 2, 5, 10} s, and Fig. 7 shows the resulting benign attached rate S .

TABLE III  
MEAN SERVERS PROVISIONED PER EVENT OVER TEN SEEDS AGAINST THE IDEAL RESERVE, WITH THE SIZING ERROR IN PARENTHESES.
<table><tr><td>Event</td><td>Ideal</td><td>GPT-5.4-Mini</td><td>GEMINI-3.1-Flash-Lite</td></tr><tr><td>Sold out</td><td></td><td></td><td></td></tr><tr><td>Taylor Swift</td><td>13</td><td>14.0 (+1.0)</td><td> $1 4 . 0 \ ( + 1 . 0 ) $ </td></tr><tr><td>Man U v Liverpool</td><td>11</td><td>11.0 (0.0)</td><td>11.0 (0.0)</td></tr><tr><td>Arsenal W v Man U W</td><td>9</td><td>9.0 (0.0)</td><td>9.0 (0.0)</td></tr><tr><td>Liverpool v Real Madrid</td><td>8</td><td>8.9 (+0.9)</td><td>8.7 (+0.7)</td></tr><tr><td>Adele</td><td>3</td><td>3.0 (0.0)</td><td>3.0 (0.0)</td></tr><tr><td>Not sold out</td><td></td><td></td><td></td></tr><tr><td>England v Brazil</td><td>13</td><td>12.6 (−0.4)</td><td>10.6 (−2.4)</td></tr><tr><td>Chelsea W v Man U W</td><td>12</td><td>11.2 (−0.8)</td><td>10.6 (−1.4)</td></tr><tr><td>West Ham v FK TSC</td><td>6</td><td>8.0 (+2.0)</td><td>7.8 (+1.8)</td></tr><tr><td>West Ham v Viborg</td><td>5</td><td>7.6 (+2.6)</td><td>7.8 (+2.8)</td></tr><tr><td>Aston Villa W v Man U W</td><td>2</td><td>3.2 (+1.2)</td><td>3.9 (+1.9)</td></tr><tr><td>Man U W v Brighton W</td><td>1</td><td>1.4 (+0.4)</td><td>2.0 (+1.0)</td></tr><tr><td>West Ham W v Everton W</td><td>1</td><td>1.0 (0.0)</td><td>1.0 (0.0)</td></tr></table>

![](images/2c93d129fa20c34f70d61306c2e0d3c2d738889fe43af776c60ebf5f82a93fea.jpg)  
Fig. 7. Benign users served versus server provisioning delay for a step and a ramp onset at low and high utility weight $V \ ( W { = } 1 )$

Provisioning delay matters far more than the weights. At zero delay, every configuration serves the full surge, and by 10 s every configuration collapses to a similarly low level. The onset shape then decides whether tuning helps at all. Under the step, from a 2 s delay onward, the controller serves only about 15% of benign users for any V, the regime of Corollary S1, where the backlog is fixed by $\tau _ { \mathrm { p r o v } }$ rather than by the objective. Under the ramp, the gradual rise is trackable, so $V { = } 2 0$ serves 88% at a 5 s delay against 19% at V=1.

## E. Compute Contention Mechanism

The preceding studies run the pool uncontended, so $\mu _ { \mathrm { e f f } } =$ $\mu .$ Here we enable the contention of Section II-B. A site cannot observe the occupancy of a shared pool, so the controller plans against the nominal $\mu$ and contention acts as an exogenous stressor.

Resilience envelope. We first remove controller dynamics by holding a pool of $c _ { \mathrm { m a x } }$ servers fixed and applying a benign step surge of growing intensity. We compare three settings, a dedicated pool with no contention and a shared pool at severities $a = 0 . 5$ and $a = 1 . 0$ . Since capacity and provisioning are fixed, only $\mu _ { \mathrm { e f f } }$ can change the outcome. In Fig. 8, the dedicated pool degrades gently, holding P above 85% until λ approaches $c _ { \mathrm { m a x } } \mu \approx 4 5 9$ UEs/s. Sharing moves this cliff to about 321 UEs/s at $a = 0 . 5$ and 247 UEs/s at $a = 1 . 0 .$ , so contention shifts a hard stability boundary rather than degrading service gracefully. Between these cliffs lies a range of surges, for example 247–459 UEs/s at $a = 1 . 0 { \mathrm { : } }$ that a dedicated pool absorbs but a shared pool does not, so contention alone turns a manageable surge into a collapse.

![](images/aaab16c16974d68a763af762cefd4092a7c9e70937e7cf9fbefef6671ebee0b5.jpg)  
Fig. 8. Resilience of a fixed fully provisioned pool $( c = c _ { \operatorname* { m a x } } )$ versus benign surge intensity, without contention and with a shared pool at severities $a \in \{ 0 . 5 , \bar { 1 . 0 } \}$ . Dashed lines mark each stability boundary $c _ { \mathrm { m a x } } \mu _ { \mathrm { e f f } } .$ , and bands are 95% CIs.

![](images/209a48bae341ae513d6ace1e5186788f7d9d72be841be228d3e677af34bb2fdd.jpg)  
Fig. 9. Resilience of a fully provisioned static pool $( c = c _ { \operatorname* { m a x } } )$ and the agentic controller (GPT-5.4-Mini) on one benign event inside the collapse region, without contention and with the shared pool at $a = 1 . 0 .$ . The agentic controller commits the same reserve at both settings. Bars are means with 95% CIs.

Controllers under contention. We place the scheduled event of Section V-A, about 279 UEs/s, inside this gap and compare the fully provisioned static pool with the agentic controller driven by GPT-5.4-Mini, without contention and at $a = 1 . 0$ (Fig. 9). The agent cannot see contention, so at both settings it estimates about 84,000 attendees and commits the same 13-server reserve. Its resilience nonetheless falls from 98% to 76%. The static pool loses more, 26 pp against 22 pp, because full occupancy maximizes the slowdown. Right-sizing is marginally less exposed, but no controller escapes a stressor outside its telemetry.

## F. From Operator Intents to Directives

The orchestrator maps a free-text intent to one OperatorDirective that downstream code applies without review, so we first test whether that mapping is faithful. We hold out 40 intents, worded and valued disjointly from the few-shot examples in the prompt. They cover the five levers, plain and oblique phrasing, lexical traps that name a concern only to waive it, multi-lever combinations, spelling noise, and vague inputs that must change nothing. Posture is scored by direction and graded strength by a lenient band, the floor must match exactly and stay null when absent, an event must fall within 60 s of its true time, and a delegation must appear exactly when requested. Setting an unrequested lever counts as an error, so hallucinated action is penalized as well as omission. We compare against a keyword matcher and a do-nothing baseline over three seeds.

TABLE IV  
OPERATOR-INTENT GROUNDING (%) ON 40 HELD-OUT INTENTS OVERTHREE SEEDS $\scriptstyle ( n = 1 2 0$ PER MODEL). GROUNDING IS THE SHARE OFAPPLIED LEVERS THAT ARE CORRECT, AND EXACT MATCH IS THE SHAREOF INTENTS CORRECT ON EVERY LEVER. BEST PER COLUMN IN BOLD.
<table><tr><td></td><td colspan="2">Overall  $( \% ) \uparrow$ </td><td colspan="5">Per-lever accuracy  $( \% ) \uparrow$ </td></tr><tr><td>Model</td><td>Grounding</td><td>Exact match</td><td>Posture</td><td>Strength</td><td>Floor</td><td>Event</td><td>Delegation</td></tr><tr><td>GPT-5.4-Mini</td><td>98.4</td><td>94.2</td><td>98</td><td>100</td><td>99</td><td>100</td><td>96</td></tr><tr><td>GEMINI-3.1-Flash-Lite</td><td>97.1</td><td>87.5</td><td>93</td><td>92</td><td>100</td><td>97</td><td>100</td></tr><tr><td>Keyword matcher</td><td>84.3</td><td>52.5</td><td>65</td><td>25</td><td>95</td><td>100</td><td>95</td></tr><tr><td>Do-nothing</td><td>66.3</td><td>5.0</td><td>35</td><td>0</td><td>82</td><td>85</td><td>82</td></tr></table>

Table IV shows that both agents ground almost every lever, GPT-5.4-Mini at 98.4% grounding and 94.2% exact match, with GEMINI-3.1-Flash-Lite close behind. Grounding overstates performance, because most intents leave most levers untouched, so even the do-nothing baseline scores 66.3% by never acting. Exact match, which requires every lever of an intent to be correct, is therefore the more informative measure, and on it both agents (94.2% and 87.5%) score far more than the keyword matcher (52.5%). The keyword matcher fails on the two semantic levers, posture (65%) and strength (25%), while staying competitive on the lexical ones. Every residual model error is over-action, a lever set that was not requested.

One directive, end to end. Grounding shows the mapping is faithful. We let a single directive act on the live system. The operator types a plain instruction, the orchestrator turns it into one directive, and the site agent and fast loop run untouched. The exchange runs as:

<table><tr><td>V</td><td>Operator Tonight is a major product launch. Protect service quality and spare no capacity.</td></tr><tr><td></td><td></td></tr><tr><td>Orchestrator</td><td></td></tr><tr><td>priority=qos, V=20, W=1</td><td></td></tr><tr><td></td><td>The launch has no usable time so nothing is scheduled, and</td></tr><tr><td></td><td>the service clauses set a full quality QoS posture.</td></tr><tr><td>V</td><td></td></tr><tr><td></td><td>Non-RT mid ramp. 200 UEs/s against 20 at rest with no calendar</td></tr><tr><td></td><td>event, so the agent stays cautious, yet capacity is held at</td></tr><tr><td>the directed  $v = \bar { 2 } 0$ </td><td></td></tr></table>

The posture adds headroom ahead of the ramp, so benign attached rate rises from 26% to 97% and resilience from 74% to 92% (Table V), so one plain instruction reshapes the standing posture and recovers service with no hand-tuned weights.

## G. Cascading and Simultaneous Stressors

The studies so far isolate one disturbance at a time. This experiment superimposes three in a single episode, so the controller must filter, anticipate, and absorb at once. We run the botnet-event scenario of Section ${ \mathrm { V } } { \mathrm { - } } { \mathrm { A } } ,$ a malicious botnet ramp followed by a benign scheduled-event step, with the shared-compute contention of Section V-E active throughout, sweeping severity $a \ \in \ \{ 0 , 0 . 5 , 1 . 0 \}$ (a=0 recovers the contention-free case, again as independent runs). Table VI shows episode resilience $P$ over five seeds for the fully provisioned static pool, the reactive Lyapunov and rule-based controllers, and the two agentic controllers.

TABLE V  
END-TO-END EFFECT OF THE QUALITY-OF-SERVICE (QOS) POSTURE INTENT ON THE BENIGN RAMP, MEAN OVER THREE SEEDS. BASELINE IS THE SAME RAMP WITH NO INTENT; $\Delta$ IS IN PERCENTAGE POINTS (PP) FOR RATES.
<table><tr><td>Metric</td><td>Baseline</td><td>+Intent</td><td> $\Delta$ </td></tr><tr><td>Benign attached rate,  $S _ { b } \ ( \% ) \ \uparrow$ </td><td>26.0</td><td>97.0</td><td>+71.0</td></tr><tr><td>Benign false positive rate, φb (%) ↓</td><td>16.0</td><td>4.0</td><td>-12.0</td></tr><tr><td>Resilience,  $P \left( \% \right) \uparrow$ </td><td>74.0</td><td>92.0</td><td>+18.0</td></tr><tr><td>Mean capacity, ē (srv)</td><td>3.05</td><td>3.57</td><td>+0.52</td></tr></table>

TABLE VI

EPISODE RESILIENCE P UNDER COMPOUNDED STRESSORS (BOTNET, EVENT, AND CONTENTION) AS CONTENTION SEVERITY a RISES, MEAN ± 95% CI OVER FIVE SEEDS. $\Delta P$ IS THE CHANGE FROM a=0 TO a=1.0 IN PERCENTAGE POINTS.
<table><tr><td rowspan="2">Controller</td><td colspan="3">Resilience P (%) ↑</td><td rowspan="2"> $\Delta P \ ( \mathrm { { p p } ) }$ </td></tr><tr><td> $\scriptstyle a = 0$ </td><td> $a { = } 0 . 5$ </td><td> $a { = } 1 . 0$ </td></tr><tr><td>Static (c=16)</td><td> $9 9 . 2 \pm 0 . 0$ </td><td> $9 9 . 0 \pm 0 . 2 $ </td><td> $8 5 . 8 \pm 0 . 4$ </td><td>-13.4</td></tr><tr><td>Lyapunov</td><td> $6 8 . 4 \pm 0 . 2$ </td><td> $6 8 . 4 \pm 0 . 2$ </td><td> $6 8 . 3 \pm 0 . 2$ </td><td>-0.1</td></tr><tr><td>Rule-based</td><td> $7 1 . 0 \pm 0 . 5$ </td><td> $7 1 . 0 \pm 0 . 4$ </td><td> $7 1 . 0 \pm 0 . 5$ </td><td>0.0</td></tr><tr><td>Agentic (GEMINI-3.1-Flash-Lite)</td><td> $8 7 . 4 \pm 1 . 2$ </td><td> $8 8 . 0 \pm 2 . 9$ </td><td> $8 2 . 3 \pm 8 . 9$ </td><td>-5.1</td></tr><tr><td>Agentic (GPT-5.4-Mini)</td><td> $9 0 . 5 \pm 0 . 7$ </td><td> $9 0 . 3 \pm 0 . 9$ </td><td> $8 1 . 0 \pm 6 . 0$ </td><td>-9.5</td></tr></table>

The stressors do not simply add up, they trigger one another. To serve the event, the controller provisions a reserve, which fills the pool, raises occupancy, and depresses $\mu _ { \mathrm { e f f } } ,$ so the reserve it committed delivers less capacity than planned and the surge it anticipated overruns it. The controller cannot correct for this, since it plans against the nominal $\mu$ and never observes the slowdown. The effect is therefore felt only by controllers that fill the pool. The reactive Lyapunov and rulebased controllers are unchanged across severity, because they sit near two servers, below the contention knee, but they already fail the event for lack of anticipation. Every provisioning controller loses resilience, the fully provisioned static pool most of all $( \Delta P = - 1 3 . 4 $ pp at a=1.0), because holding all $c _ { \mathrm { m a x } }$ servers hot maximizes occupancy and slowdown. The loss is also asymmetric across the episode. The botnet window is untouched at every severity, because the botnet is filtered and never fills the pool, so the collapse concentrates in the event window where legitimate load must be carried. This qualifies the conclusion of Section V-B. Over-provisioning, the safe choice without contention, becomes the most exposed under it, losing the most resilience of any controller, whereas the agentic controllers, which size capacity to demand rather than fill the pool, lose least among the controllers that provision for the event (∆P of −5.1 to −9.5 pp at a=1.0) and still serve more of the event than the static pool. No controller escapes a stressor outside its telemetry, but right-sizing capacity is less exposed to it than brute-force provisioning, a distinction only a compounded scenario reveals.

TABLE VII  
MULTI-SITE ORCHESTRATION, $\mathbf { M E A N } \pm 9 5 \%$ CI OVER FIVE SEEDS. ALL THREE INTENTS REACHED EXACTLY THE CORRECT SITE(S) IN 5/5 SEEDS.
<table><tr><td>Site</td><td>Intent applied</td><td> $P \left( \% \right) \uparrow$ </td><td> $S _ { b } \ ( \% ) \ \uparrow$ </td><td>ē (srv)</td></tr><tr><td>Stadium</td><td>QoS + legitimate (local override)</td><td> $8 4 . 1 \pm 1 0 . 0$ </td><td> $8 6 . 6 \pm 3 6 . 8$ </td><td>4.30</td></tr><tr><td>Residential</td><td>Fleet cost (not exempted)</td><td> $8 0 . 9 \pm 0 . 1$ </td><td> $7 4 . 7 \pm 2 . 0$ </td><td>1.00</td></tr><tr><td>Business</td><td>Cost + server floor</td><td> $9 3 . 7 \pm 0 . 6 $ </td><td> $7 7 . 1 \pm 1 . 0$ </td><td>2.68</td></tr></table>

## H. Multi-Site Orchestration

The intent grounding of Section V-F runs at one site. A network operator, however, issues one intent over many sites, and those intents may target a single site, the whole fleet, or clash with one another. This experiment runs three sites concurrently, each with its own simulator, compute pool, Non-RT agent, and fast loop, while a single SMO orchestrator decomposes each free-text intent into per-site directives over the A2A protocol (Section III-F). The fleet is a stadium cell running the compounded botnet-event scenario of Section V-G under its own contention, a residential macro cell carrying a benign demand ramp, and a business cell with a benign step. Three operator intents arrive in sequence. A fleet-wide cost posture, a stadium-only override that raises service and marks the crowd as legitimate, and a hard server floor at the business cell. Correct handling requires routing each to the right site, resolving the clash between the fleet cost posture and the stadium override in favor of the more specific one, and leaving untargeted sites alone.

Table VII reports the outcome over five seeds. The orchestrator routes and resolves every intent correctly in all five seeds. The fleet-wide posture reaches all three sites, the stadium override lands on the stadium alone and overrides the standing fleet cost posture there (scope precedence), and the floor reaches only the business cell, with no collateral directives to untargeted sites. The per-site outcomes follow the intents that reached them. The stadium, protected and told the surge is legitimate, serves its event while filtering the botnet, at the reduced level that the a=1.0 contention imposes (Section V-G). Its wide benign-service interval reflects the contention cliff of Section V-E, which some seeds cross, and others do not. The residential cell, left under the fleet cost posture, runs at its floor of one server and underserves its ramp. The business cell, granted the capacity floor, holds the most headroom of the benign sites and the highest resilience. Which intent reached each site therefore determined its outcome, which is the point of per-site decomposition.

## I. Computational and Signaling Overhead

Two tiers use a language model, but only one does so continuously. The SMO orchestrator invokes an LLM to ground an operator intent, an event-driven action that stays idle and adds no recurring cost (Section III-F). The Near-RT loop uses no model at all, solving the scalar objective (12) over $c \in [ 1 , c _ { \operatorname* { m a x } } ]$ at every step. The framework’s recurring cost is therefore the per-site Non-RT assessment reported in Table I. This cost is decoupled from data-plane scale, because each assessment consumes a fixed-size summary of the telemetry window and emits one policy struct, the same whether a cell serves ten or a thousand UEs, as arrivals, retries, and queues are aggregated into rates before the model sees them. The number of LLM invocations therefore grows with the number of sites and the assessment cadence, not with the UEs, cells, or DUs/CUs a site aggregates. The effective assessment cadence depends on the model, since each cycle is a 5 s waiting interval plus the decision latency. At the GEMINI-3.1-Flash-Lite operating point the roughly 3 s decision gives a cadence near 8 s, about 450 assessments per site per hour, and about \$0.8 per site per hour at \$0.0018 per decision. GPT-5.4-Mini with reasoning takes about 23 s per decision, which stretches the cadence to near 28 s and, at \$0.0123 per decision, costs about \$1.6 per site per hour. Cross-tier signaling is equally light, namely one small policy write per assessment from the Non-RT agent to the Near-RT loop and the rare operator intent, neither of which rides the data plane, so the framework adds no per-UE signaling. However, granting more autonomy to the orchestrator, for instance periodically probing the site agents to confirm they remain healthy and are acting as intended, would replace that idle steady state with a recurring healthcheck load that grows with the number of sites. As sites multiply, the single orchestrator decomposes one intent into per-site directives over A2A (Section V-H), while each site runs an independent Non-RT and Near-RT loop, so N sites are N parallel loops with no central bottleneck and the agent’s latency off the fast path never delays actuation.

## VI. CONCLUSION

This paper treated resilience in AI-RAN as a single lifecycle, spanning anticipation, absorption, adaptation, recovery, and evolution, and asked whether agentic AI can coordinate it without destabilizing the fast control loops it depends on. We studied this on mass-attachment signaling storms, using a discrete-event simulator that captures retry amplification, shared-compute contention, and delayed provisioning. Our three-tier framework keeps language-model reasoning off the real-time path. An SMO orchestrator turns operator intent into policy, a per-site Non-RT agent judges storms and anticipates scheduled crowds, and a deterministic Near-RT loop enforces every decision.

Anticipation proves decisive. We showed analytically that, under delayed provisioning, no reactive policy can prevent backlog during a fast-onset surge, whatever its tuning. Consistent with this, the agentic controller reaches resilience above 85% with about a quarter of the capacity of a fully provisioned pool, sizing reserves within about one server of the ideal for crowds whose turnout is never stated. Free-text intents ground into directives at 94% exact match, and a single orchestrator routed every intent correctly across three concurrent sites. The limits are equally clear. Shared-compute contention, which the agent cannot observe, shifts the stability cliff from about 459 to 247 UEs/s, and model reasoning adds latency and cost.

Several directions remain open. The framework already includes a cross-episode memory for the evolution stage, whose evaluation we leave to a dedicated study of learning across disruptions. Scaling multi-site orchestration to larger fleets with richer inter-agent messaging is a natural next step, as is reducing the signaling and compute footprint of the agent, especially where it shares the pool with the radio functions it protects. Placing language-model agents in closedloop control also raises safety and security concerns, from hallucinated actions to adversarial manipulation of agent reasoning, so strengthening the deterministic guardrails, rollback, and grounding that keep the probabilistic tier accountable is essential before deployment. Validating the framework on a real Open RAN testbed under live signaling load remains the ultimate test.

## REFERENCES

[1] M. Polese, L. Bonati, S. D’Oro, S. Basagni, and T. Melodia, “Understanding O-RAN: Architecture, Interfaces, Algorithms, Security, and Research Challenges,” IEEE Communications Surveys & Tutorials, vol. 25, no. 2, pp. 1376–1411, 2023.

[2] L. Kundu, X. Lin, R. Gadiyar, J.-F. Lacasse, and S. Chowdhury, “AI-RAN: Transforming RAN with AI-Driven Computing Infrastructure,” IEEE Communications Magazine, vol. 64, no. 1, pp. 168–174, 2026.

[3] AI-RAN Alliance, “AI-RAN Alliance Vision and Mission,” AI-RAN Alliance, White paper, Dec. 2024.

[4] P. Smith, D. Hutchison, J. P. Sterbenz, M. Schöller, A. Fessi, M. Karaliopoulos, C. Lac, and B. Plattner, “Network resilience: a systematic approach,” IEEE Communications Magazine, vol. 49, no. 7, pp. 88–97, 2011.

[5] S. Bi, X. Yuan, S. Hu, K. Li, W. Ni, E. Hossain, and X. Wang, “Resilience and Failure Analysis in Next-Generation Communication Networks: A Contemporary Survey,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 2793–2821, 2026.

[6] J. Farooq and Q. Zhu, “Cyber Resilience in Next-Generation Networks: Threat Landscape, Theoretical Foundations, and Design Paradigms,” 2025. [Online]. Available: https://arxiv.org/abs/2512.22721

[7] M. Bennis, S. Samarakoon, T. Alshammari, C. Weeraddana, Z. Tian, and C. B. Issaid, “Resilient-Native and Intelligent Next-Generation Wireless Systems: Key Enablers, Foundations, and Applications,” 2025. [Online]. Available: https://arxiv.org/abs/2506.22991

[8] N. H. Mahmood, S. Samarakoon, P. Porambage, M. Bennis, and M. Latva-Aho, “Resilient-by-Design: A Resilience Framework for Future Wireless Networks,” IEEE Communications Magazine, vol. 63, no. 11, pp. 158–164, 2025.

[9] R.-J. Reifert, Y. Karacora, C. Chaccour, A. Sezgin, and W. Saad, “Resilience and Criticality: Brothers in Arms for 6G,” 2024. [Online]. Available: https://arxiv.org/abs/2412.03661

[10] V. Kasuluru, L. Blanco, and E. Zeydan, “Enhancing Open RAN Operations: The Role of Probabilistic Forecasting in Network Analysis,” IEEE Transactions on Network and Service Management, vol. 22, no. 4, pp. 3676–3691, 2025.

[11] A. Masaracchia, V. Sharma, M. Fahim, O. A. Dobre, and T. Q. Duong, “Digital Twin for Open RAN: Toward Intelligent and Resilient 6G Radio Access Networks,” IEEE Communications Magazine, vol. 61, no. 11, pp. 112–118, 2023.

[12] S. Chatzimiltis, M. Shojafar, M. B. Mashhadi, and R. Tafazolli, “AI-on-RAN for Cyber Defense: An XAI-LLM Framework for Interpretable Anomaly Detection,” IEEE Transactions on Network Science and Engineering, vol. 13, pp. 3301–3319, 2026.

[13] R. M. Sohaib, S. Tariq Shah, and P. Yadav, “Towards Resilient 6G O-RAN: An Energy-Efficient URLLC Resource Allocation Framework,” IEEE Open Journal of the Communications Society, vol. 5, pp. 7701– 7714, 2024.

[14] M. Elkael, S. D’Oro, L. Bonati, M. Polese, Y. Lee, K. Furueda, and T. Melodia, “AgentRAN: An Agentic AI Architecture for Autonomous Control of Open 6G Networks,” 2026. [Online]. Available: https://arxiv.org/abs/2508.17778

[15] X. Wu, Y. Wang, J. Farooq, and J. Chen, “LLM-Driven Agentic AI Approach to Enhanced O-RAN Resilience in Next-Generation Networks,” in IEEE INFOCOM 2025 - IEEE Conference on Computer Communications Workshops (INFOCOM WKSHPS), 2025, pp. 1–6.

[16] H. Navidan, M. Cheraghinia, J. Fontaine, M. Seif, E. D. Poorter, H. V. Poor, I. Moerman, and A. Shahid, “Toward Autonomous O-RAN: A Multi-Scale Agentic AI Framework for Real-Time Network Control and Management,” 2026. [Online]. Available: https://arxiv.org/abs/2602.14117

[17] A. Tabiban, H. A. Alameddine, M. A. Salahuddin, and R. Boutaba, “Signaling Storm in O-RAN: Challenges and Research Opportunities,” IEEE Communications Magazine, vol. 62, no. 6, pp. 58–64, 2024.

[18] S. Mayhoub, S. Chatzimiltis, H. Alimohammadi, Z. He, S. A. Abdulkareem, M. Shojafar, A. Akbas, and C. H. Foh, “A New Sub-Use Case for Signaling Storm Attack in Open RAN and an ML-based Detection Approach,” in 2024 IEEE Conference on Standardsfor Communications and Networking (CSCN), 2024, pp. 308–313.

[19] 3rd Generation Partnership Project (3GPP), “5G NR Radio Resource Control (RRC) Protocol specification,” 3rd Generation Partnership Project (3GPP), Technical Specification TS 38.331, 2023, version 18.1.0.

[20] O-RAN Alliance, “O-RAN Open Xhaul Transport Working Group 9: Xhaul Transport Requirements,” O-RAN Alliance e.V., Alfter, Germany, Technical Specification O-RAN.WG9.XTRP-REQ.0-R005-v02.00, Nov. 2025.

[21] 3rd Generation Partnership Project (3GPP), “Study on New Radio Access Technology: Radio Access Architecture and Interfaces,” 3rd Generation Partnership Project (3GPP), Technical Report TR 38.801, Mar. 2017, version 14.0.0, Release 14.

[22] O-RAN Working Group 1, “O-RAN Architecture Description,” O-RAN Alliance e.V., O-RAN.WG1.OAD-R005-v16.00 Technical Specification, February 2026.

[23] Model Context Protocol Contributors, “Model context protocol specification,” https://modelcontextprotocol.io/specification/draft, 2026, accessed: 2026-05-18.

[24] A2A Project Contributors, “A2a: An open protocol enabling communication and interoperability between opaque agentic applications,” https://github.com/a2aproject/A2A, 2025, accessed: 2026-05-18.

Sotiris Chatzimiltis (Graduate Student Member) received a B.Sc. degree in computer science from the University of Cyprus, Cyprus, in 2021, and an M.Sc. degree in computer vision, machine learning and robotics from the University of Surrey, U.K., in 2022. He is a PhD student at the Institute for Communication Systems at the University of Surrey. His current research interests include machine learning, intrusion detection systems, distributed AI, and Open RAN security.

Mahdi Boloursaz Mashhadi (S’14-M’18, SM’23) is a Lecturer at the 5G/6G Innovation Centre (5G/6GIC) at the Institute for Communication Systems (ICS), University of Surrey (UoS), and a Surrey AI fellow. His research is focused at the intersection of AI/ML with wireless communication, learning and communication co-design, generative AI for telecommunications, and collaborative machine learning. He received B.S., M.S., and Ph.D. degrees in mobile telecommunications from the Sharif University of Technology (SUT), Tehran, Iran. He has more than 40 peer reviewed publications and patents in the areas of wireless communications, machine learning, and signal processing. He is a PI/Co-PI for various government and industry funded projects including the UKTIN/DSIT 12M£ national project TUDOR.

Mohammad Shojafar (M’17-SM’19) is an Associate Professor in network security and an Intel Innovator, professional ACM member and ACM distinguished speaker, a fellow of the Higher Education Academy, and a Marie Curie alumnus, working in the 6G Innovation Centre (6GIC), Institute for Communication Systems (ICS), at the University of Surrey, UK. Before joining 6GIC, he was a senior researcher and a Marie Curie fellow in the SPRITZ Security and Privacy Research group at the University of Padua, Italy. Dr Shojafar secured around £3M as PI in various EU/UK projects. https://www.surrey.ac.uk/people/mohammad-shojafar.

Rahim Tafazolli (Fellow, IEEE) is currently a Professor in mobile and personal communications and the Director of the Institute of Communication Systems (ICS) and the 5G and 6G Innovation Centre, University of Surrey. He has been active in research for over 30 years and published more than 1200 research papers. He has been a technical advisor to many mobile companies, and has lectured, chaired, and been invited as keynote speaker to a number of IEEE workshops and conferences. He served as the Chairperson for the EU Expert Group on Mobile Platform (e-mobility SRA), the Chairperson for the Post-IP working Group in e-mobility, and the past Chairperson of WG3 of WWRF. He is nationally and internationally known in the field of mobile communications. In May 2018, he was appointed as a Regius Professor in electronic engineering for recognition of his exceptional contributions to digital communications technologies over the past 30 years. He was elected as a fellow of the U.K. Royal Academy of Engineering, in 2020. He is a fellow of IET and the Wireless World Research Forum (WWRF).

# I. SYSTEM PROMPTS

## A. Non-RT Agent

## Prompt A

per-site Non-RT agent

You are the Non-RT-Agent for an AI-RAN site. You run every few seconds as a slow layer of judgment above a fast 1 Hz ,→ control loop you never block. That loop sizes capacity to the load it sees, but it only balances - it holds no slack.

Two decisions per cycle. When load is elevated, is it a malicious storm or a benign surge, since only a storm gets the

,→ malicious-UE filter. And does the loop need capacity - headroom for load that is here, or a reserve ahead of a crowd ,→ you can see coming. Write both to shared policy and stop.

## ## Input

6 Your previous verdict, the policy you last wrote. Use it for continuity.

A \~15 s telemetry window given as trends. The arrival rate is "lam".

\- LATEST lam - the rate right now. The number that matters most.

\- resting lam - the calm baseline over the whole episode, the reference you judge LATEST lam against. Valid even when this ,→ window holds no calm. Don't re-derive it.

11 - lam over the window - start → end, direction, peak and how long ago. [LEADING]

\- lam trajectory - binned means, so you see ramp, plateau or decay.

\- queue\_len - start → end, direction. [LAGGING]

\- retry-rate - first half vs second half, direction. [LEADING]

16 [LEADING] signals move at storm onset. The [LAGGING] one reacts afterwards and the controllers below you hold it down, so ,→ lead your call on arrivals.

{{tools}}

Call each tool once, then decide. Don't poll again because a window is ambiguous.

## 21 ## What is happening

Two questions. Is load elevated at all, which decides capacity. If so, is it malicious, which decides the filter.

Load is elevated when the LATEST lam sits clearly above rest and holds there. Judge the latest lam, not the peak - a high ,→ peak with the latest back near rest means the load has already passed.

26 Never read a calm queue or zero retries as no load. The capacity loop drains the queue and your filter kills retries, so ,→ during a real flood both look calm because the controllers below you are working. The flood is still arriving at the ,→ door. Arrivals are the only signal nothing downstream erases.

A benign surge pushes arrivals, queue and retries as high as a storm, because the overload itself drives all three. What ,→ separates them is the CALENDAR. get\_calendar gives you the current time and each event's scheduled time, both in ,→ seconds. Compare them.

\- An event's scheduled time has arrived (current time is at or just past it) and load is elevated - that load is the ,→ event. Benign. Leave the filter OFF however high the arrivals.

\- The event is still in the FUTURE and load is already elevated - the event cannot explain load that is here now, so that ,→ traffic came from elsewhere. Filter it as a storm.

31 - Nothing on the calendar and load is elevated - unexplained traffic. Malicious storm, filter it.

Arming and standing down are not symmetric. Arm on the first clearly elevated unexplained window, because arming late lets ,→ more of the botnet through. Stand down only once the latest lam has returned to rest and held there.

## ## Filter strength

36 Set malicious\_drop\_prob in (0, 1], no default. Scale it to how far the latest lam sits above rest. Dropping hard throws ,→ away good traffic, so a slight lift deserves a lighter touch than a flood.

Feedback is absorption from get\_episode\_stats, a trailing measure of how much calm-baseline utility you are holding, capped ,→ at 1. Near the cap you are holding, so hold or ease off. Well below it while lam stays high, push harder. It lags the ,→ change you just made, so don't escalate again before it has caught up.

## ## Capacity - two levers

41 lyapunov\_V, for load that is HERE. Raise it above its default of 1 (max 20) with tighten=true to give the loop more servers ,→ than bare balancing, so real users don't fail for lack of slack. Scale to severity, as with filter strength. Do this ,→ during elevated load, storm or surge alike.

expected\_attendance + event\_time, for a crowd that is COMING. When get\_calendar names an event, estimate how many people ,→ will attend it, and write that estimate together with the event's scheduled time (event\_time, the t=...s the calendar ,→ gives), ONCE, with tighten=true.

Do NOT provision the reserve early yourself and do NOT re-estimate each cycle. Write the plan once.

## ## Output (PolicyUpdate)

\- storm\_active - true only for a malicious storm.

\- malicious\_drop\_prob - your strength during a storm, 0.0 otherwise.

\- lyapunov\_V - default 1, max 20. Applied only when tighten=true.

51 - expected\_attendance - your crowd estimate for the scheduled event, 0 when none.

\- event\_time - the event's scheduled time in seconds, from get\_calendar.

\- queue\_hold\_threshold, lyapunov\_W - leave at defaults unless you have a reason.

\- tighten - true whenever a slow knob should change.

\- reasoning - one or two sentences.

## Prompt B

You are the Orchestrator of an AI RAN network at the SMO tier. An operator gives free text intents. You return one   
OperatorDirective.   
Below you sit a per-site Non-RT agent and a fast control loop. The Non-RT agent reads telemetry, forecasts, and the event   
,→ calendar to classify traffic, detect storms, set how aggressively to filter, and size upcoming events for   
,→ pre-provisioning. It also tunes the control posture on its own, which your directives override. The fast loop then,   
,→ each tick, sets the server count and applies that admission filter, dropping suspected malicious traffic while a   
,→ storm is flagged.   
Read the intent as a set of requests. Clauses joined by "and", "while" or a comma usually carry separate ones, one clause   
,→ can carry two, and several clauses can refine one lever. A clause that only explains or justifies requests nothing,   
,→ even when it names cost or service. If no clause requested a lever, leave it alone. priority stays "balanced" and   
,→ optional fields stay null. Silence is not permission to act.   
# Telling posture from delegation   
Posture changes HOW MANY servers run. Delegation changes HOW TRAFFIC IS CLASSIFIED. A clause that says what a condition   
,→ MEANS, or how to resolve uncertainty, is delegation ("this load is genuine, not an attack", "when in doubt lean this   
,→ way"). It asks nothing of capacity, so it sets nonrt\_instruction and no posture. Never write a nonrt\_instruction that   
,→ restates the posture you already set.   
# Direction   
10 Work out what the operator wants the network to end up doing, then read the direction off that. Do not decide from a single   
,→ cost or service word, because a concern is often named only to waive it. "Whatever", "regardless of", "no matter" and   
,→ "even if" mark a waived concern, so "whatever it costs" asks for service and not savings, and "even if performance   
,→ dips" asks for savings and not service.   
protect service, accept the cost → qos   
protect spend, accept the risk → cost   
neither named → balanced   
# Strength   
priority already resolves to the FULL posture (V or W at 20). Use the explicit lyapunov\_V/lyapunov\_W (0 to 20) only to   
,→ TEMPER a hedged ask below that full lean. Hold the other weight at 1, and never set V equal to W.   
hedged ("a slight lean", "trim a bit", "nothing drastic") an intermediate weight qos V=6 W=1 cost V=1 W=6 plain,   
,→ or maximal ("maximum protection", "cut to the bone") both null (priority is already full)   
20   
A hedge softens magnitude, it never flips or removes direction: "trim a bit" is still cost.   
# Remaining levers   
min\_servers is a hard SLA floor. Set it only when the operator states a QUANTITY with floor language ("at least N", "keep N   
,→ warm"). Never invent a number. Reliability with no count ("guarantee availability") is posture, not a floor, and   
,→ identifiers, dates, times and percentages are never floors.   
25   
schedule\_event\_name, \_t, \_venue and \_sold\_out register a known upcoming load event. Name and time go together, one without   
,→ the other is discarded, and the time must be simulated seconds, stated or convertible. For an event you are   
,→ scheduling (name and time present), also set \_venue if a place is named, and set \_sold\_out true or false only when   
,→ the intent states it, so the site can size the crowd, otherwise leave both null. An event with no usable time is not   
,→ scheduled, so carry it through posture or delegation instead.   
nonrt\_instruction is one or two sentences the site agent reads in every assessment. Write a self-contained imperative rule   
,→ that makes sense to someone who never saw the operator. No capacity numbers, weights or event times.   
30 If the operator reports nothing unusual, asks a question, or gives no actionable instruction, return balanced with   
,→ everything null. Acting on a vague intent is worse than not acting, because your directive overrides tuning that   
,→ already handles ordinary load.   
# Examples (shapes, not answers)   
"Ease off capacity a little, nothing dramatic."   
→ cost, V=1, W=6. a hedged lean, so temper below the full cost posture.   
35 "Spare no capacity for the ministerial visit, and hold twelve servers minimum."   
→ qos, min\_servers=12. priority alone is already the full posture, so no explicit weight.   
"A firmware push forces every handset to re-register at t=940, be ready."   
→ balanced, name='firmware re-registration', t=940. "Be ready" is what   
scheduling already does and justifies no posture change.   
40 "The sold-out keynote at the downtown conference centre starts at t=520."   
→ balanced, name='keynote', t=520, venue='downtown conference centre', sold\_out=true.   
"The meter fleet checks in on the hour in a big burst, that is normal for them."   
→ balanced, nonrt\_instruction="Synchronised bursts from the meter fleet are normal machine   
traffic. Do not treat their periodic spikes alone as a malicious storm." No time given,   
45 so nothing is scheduled.   
"All normal here, carry on."   
→ balanced, everything null.   
Return one OperatorDirective. In reasoning, name each clause and the lever you assigned it.

## II. NOMENCLATURE AND SIMULATION PARAMETERS

TABLE S1 NOTATION AND SIMULATION PARAMETERS
<table><tr><td>Symbol</td><td>Description</td><td>Value</td></tr><tr><td colspan="3">Section II-A Scenario, Threat Model, and Service Rate</td></tr><tr><td> $\alpha _ { \mathrm { f p } }$ </td><td>Benign false-positive fraction</td><td>0.05</td></tr><tr><td> $c _ { \mathrm { m a x } }$ </td><td>Maximum servers</td><td>16</td></tr><tr><td> $D _ { \mathrm { o n e w a y } }$ </td><td>One-way transport delay</td><td>1.6 ms</td></tr><tr><td> $\lambda ( t )$ </td><td>Attach arrival rate</td><td></td></tr><tr><td> $\lambda _ { \mathrm { b e n } } , \lambda _ { \mathrm { m a l } }$ </td><td>Benign / malicious arrivals</td><td></td></tr><tr><td> $\mu$ </td><td>Service rate</td><td></td></tr><tr><td> $M$ </td><td>CU-terminated RRC messages</td><td>3</td></tr><tr><td> $N$ </td><td>Event Attendance</td><td></td></tr><tr><td> $t _ { 0 } , \ t _ { d }$ </td><td>Storm start / end</td><td></td></tr><tr><td> $t _ { p , i }$ </td><td>Processing time of message i</td><td></td></tr><tr><td> $T _ { p }$ </td><td>Total internal CU/DU processing</td><td>30 ms</td></tr><tr><td> $T _ { \mathrm { r w } }$ </td><td>Reconnection window</td><td> $6 0 \mathrm { ~ s ~ }$ </td></tr><tr><td> $T _ { s }$ </td><td>Service time</td><td></td></tr><tr><td> $\phi$ </td><td>Active-subscriber fraction on the cell</td><td>0.2</td></tr><tr><td colspan="3">Section II-B Discrete-Event Storm Dynamics</td></tr><tr><td>a</td><td>Contention severity</td><td></td></tr><tr><td> $^ { b }$ </td><td>Busy attach-workers</td><td></td></tr><tr><td> $c ( t ) , \ c ^ { \star } ( t )$ </td><td>Online / commanded servers</td><td></td></tr><tr><td> $\kappa$ </td><td>Shared-compute pool size</td><td> $c _ { \mathrm { m a x } }$ </td></tr><tr><td> $K$ </td><td>Max attempts before failure</td><td>5</td></tr><tr><td> $\lambda _ { \mathrm { e f f } } ( t )$ </td><td>Retry-amplified load</td><td></td></tr><tr><td> $L _ { q } ( t )$ </td><td>Queue length</td><td></td></tr><tr><td> $\mu _ { \mathrm { e f f } }$ </td><td>Effective service rate</td><td></td></tr><tr><td> $n ( t )$ </td><td>Mean attempts per UE</td><td>[1, K]</td></tr><tr><td> $s ( \omega )$ </td><td>Slowdown</td><td></td></tr><tr><td> $\tau _ { \mathrm { p r o v } }$ </td><td>Per-increment provisioning delay</td><td>0, 2, 5, 10 s</td></tr><tr><td> $T _ { 3 0 0 }$ </td><td>RRC setup guard timer</td><td>1000 ms</td></tr><tr><td> $T _ { b }$ </td><td>Botnet impatient attach timeout</td><td>200 ms</td></tr><tr><td> $T _ { s } ^ { \mathrm { e f f } }$ </td><td>Effective service time</td><td></td></tr><tr><td> $\omega$ </td><td>Compute occupancy  $b / \kappa$ </td><td></td></tr><tr><td> $\omega _ { \mathrm { t h } }$ </td><td>Contention onset threshold</td><td>0.65</td></tr><tr><td colspan="3">Section II-C Resilience Quantification</td></tr><tr><td> $\Delta t _ { \mathrm { d e s } }$ </td><td>Desired recovery threshold</td><td>60 s</td></tr><tr><td> $\Delta t _ { \mathrm { h o l d } }$ </td><td>Recovery hold time</td><td>30 s</td></tr><tr><td> $\eta _ { \mathrm { r e c } }$ </td><td>Recovery threshold (fraction of  $u _ { \mathrm { d e s } } )$ </td><td>0.95</td></tr><tr><td> $k _ { A }$ </td><td>Utilization-term steepness (ρ-space)</td><td>10</td></tr><tr><td> $k _ { B }$ </td><td>Queue-term steepness</td><td>0.004</td></tr><tr><td> $\boldsymbol { L } _ { q , \mathrm { m a x } }$ </td><td>Queue overflow threshold</td><td>1500</td></tr><tr><td> $m _ { \mathrm { f r a c } A }$ </td><td>Utilization knee  $( \rho \mathrm { ~ a t ~ } u _ { A } { = } 0 . 5 )$ </td><td>0.90</td></tr><tr><td> $m _ { \mathrm { f r a c } B }$ </td><td>Queue midpoint fraction of  $\boldsymbol { L } _ { q , \mathrm { m a x } }$ </td><td>0.5</td></tr><tr><td> $\rho ( t )$ </td><td>Utilization  $\lambda / ( c \mu )$ </td><td></td></tr><tr><td> $P$ </td><td>Episode resilience</td><td>[0,1]</td></tr><tr><td> $t _ { r }$ </td><td>Recovery time</td><td></td></tr><tr><td> $u ( t )$ </td><td>Utility</td><td>[0,1]</td></tr><tr><td> $u _ { A } , \ u _ { B }$ </td><td>Utilization / queue utility terms</td><td>[0,1]</td></tr><tr><td> $u _ { \mathrm { d e s } }$ </td><td>Desired (baseline) utility</td><td></td></tr><tr><td> $w _ { 1 } , \ w _ { 2 } , \ w _ { 3 }$ </td><td>Absorp. / adaptat. / recov. weights</td><td>0.4, 0.4, 0.2</td></tr><tr><td> $w _ { A } ,$  WB</td><td>Utilization / queue utility weights</td><td>0.5, 0.5</td></tr></table>

TABLE S2  
NOTATION AND SIMULATION PARAMETERS (CONTINUED)

## III. ANALYTICAL LIMITS OF PROVISIONING

The dynamics of Section II-B and the controllers of Section II-D of the main paper together fix what any capacity policy can achieve against a fast-onset surge. We state two results, the first giving the lead time sufficient to absorb such a surge without backlog, the second showing that a purely reactive controller cannot obtain it. Both refer to the fluid queue of Eq. (8) of the main paper, which the discrete-event simulator realizes stochastically.

<table><tr><td>Symbol</td><td>Description</td><td>Value</td></tr><tr><td colspan="3">Section II-D Controllers</td></tr><tr><td> $a _ { n }$ </td><td>Normalized unmet demand</td><td></td></tr><tr><td> $J ( c )$ </td><td>Drift-plus-penalty objective</td><td></td></tr><tr><td> $p _ { \mathrm { d r o p } }$ </td><td>Malicious drop probability</td><td>[0, 1]</td></tr><tr><td> $q _ { n }$ </td><td>Normalized backlog</td><td></td></tr><tr><td> $V$ </td><td>Utility weight</td><td>1</td></tr><tr><td>W</td><td>Resource-cost weight</td><td>1</td></tr><tr><td colspan="3">Section II-E Metrics</td></tr><tr><td> $A _ { b } , \ A _ { m }$ </td><td>Benign / botnet attached</td><td></td></tr><tr><td>č</td><td>Mean online servers</td><td></td></tr><tr><td> $D _ { b } , \ D _ { m }$ </td><td>Benign / botnet dropped</td><td></td></tr><tr><td> $E$ </td><td>Capacity efficiency</td><td>≥0</td></tr><tr><td> $\phi _ { b }$ </td><td>Benign false-positive rate</td><td>[0, 1]</td></tr><tr><td> $\Phi _ { m }$ </td><td>Botnet dropped fraction</td><td>[0, 1]</td></tr><tr><td> $F _ { b } , \ F _ { m }$ </td><td>Benign / botnet failed</td><td></td></tr><tr><td> $N _ { w }$ </td><td>Number of storm windows</td><td></td></tr><tr><td> $P ^ { ( k ) }$ </td><td>Per-window resilience</td><td>[0, 1]</td></tr><tr><td> $P _ { \mathrm { b o t } } , \ P _ { \mathrm { s u r g e } }$ </td><td>Botnet / event window resil.</td><td>[0, 1]</td></tr><tr><td> $S _ { b }$ </td><td>Benign attached rate</td><td>[0, 1]</td></tr><tr><td colspan="3">Section III-A Near-RT Control Loop (Alg. 1)</td></tr><tr><td> $\beta _ { \mathrm { r e l } }$ </td><td>Reserve release threshold</td><td>3</td></tr><tr><td> $\mathcal { B }$ </td><td>Cross-episode memory</td><td></td></tr><tr><td> $c _ { \mathrm { e v } }$ </td><td>Servers online at surge onset</td><td></td></tr><tr><td> $c _ { \mathrm { r e s } } ^ { \mathrm { e f f } } , \ \underline { { c } }$ </td><td>Applied reserve / effective floor</td><td></td></tr><tr><td> $\lambda _ { \mathrm { { r e s t } } }$ </td><td>Resting arrival rate</td><td></td></tr><tr><td> $\tilde { \pi }$ </td><td>Atomic snapshot of shared policy</td><td></td></tr><tr><td> $\pi . L _ { \mathrm { h o l d } }$ </td><td>Queue-hold threshold</td><td></td></tr><tr><td> $T _ { \mathrm { c t l } }$ </td><td>Near-RT control period</td><td></td></tr><tr><td> $T _ { \mathrm { a n t } } ^ { \mathrm { m i n } }$ </td><td>Minimum anticipation horizon (S2)</td><td></td></tr><tr><td> $T _ { \mathrm { r a m p } }$ </td><td>Ramp time to bring res. servers online</td><td></td></tr><tr><td colspan="3">Section III-B Non-RT Agent (Alg. 2)</td></tr><tr><td> $\mathcal { H }$ </td><td>Non-RT telemetry window</td><td></td></tr><tr><td> $\hat { N }$ </td><td>Estimated event attendance</td><td></td></tr><tr><td> $\pi ^ { - } , \pi ^ { + }$ </td><td>Previous / candidate shared policy</td><td></td></tr><tr><td></td><td>Scheduled reserved servers / surge time</td><td></td></tr><tr><td> $\pi . c _ { \mathrm { r e s } } , ~ \pi . t _ { \mathrm { e v } }$   $T _ { \mathrm { a s m } }$ </td><td>Non-RT assessment period</td><td>5 s</td></tr><tr><td> $\sigma , ~ \hat { \lambda } , ~ \mathcal { E }$ </td><td>Log / forecast / calendar tool return</td><td></td></tr><tr><td colspan="3">Section III-F Agentic AI Orchestrator (Alg. 3)</td></tr><tr><td> $^ d$ </td><td>Grounded directive</td><td></td></tr><tr><td>d.event, d.floor, d.instr</td><td>Directive fields extracted from the intent</td><td></td></tr><tr><td>I</td><td>Operator intent</td><td></td></tr><tr><td> $\pi$ </td><td>Site shared policy</td><td></td></tr><tr><td> $\pi . c _ { \mathrm { f l o o r } }$ </td><td>Operator capacity floor</td><td></td></tr><tr><td>π.instr</td><td>Instruction to site agent from orchestrator</td><td></td></tr><tr><td>π.V, π.W</td><td>Site shared policy weights</td><td></td></tr><tr><td colspan="3">Section IV and Supp. Section III Analytical Limits of Provisioning</td></tr><tr><td> $c _ { \mathrm { r e q } }$ </td><td>Capacity required for λs (S1)</td><td></td></tr><tr><td> $\lambda _ { s }$ </td><td>Step-surge arrival rate</td><td></td></tr><tr><td> $c _ { \mathrm { f a s t } } ( t )$ </td><td>Fastest feasible capacity (S4)</td><td></td></tr></table>

Theorem S1 (Minimum anticipation horizon). Consider a step surge of constant arrival rate $\lambda _ { s }$ beginning at $t _ { e \nu }$ , with an empty queue and $c _ { e \nu }$ servers online. Let

$$
c _ { r e q } = \left\lceil \frac { \lambda _ { s } } { \mu _ { e f f } } \right\rceil , \qquad c _ { r e q } \leq c _ { \mathrm { m a x } } ,\tag{S1}
$$

and let capacity be added in unit increments every $\tau _ { p r o \nu }$ seconds as in Eq. (7) of the main paper. If the controller raises its capacity floor to $c _ { r e q }$ at least

$$
T _ { a n t } ^ { \operatorname* { m i n } } = ( c _ { r e q } - c _ { e \nu } ) ^ { + } \tau _ { p r o \nu } , ( x ) ^ { + } = \operatorname * { m a x } ( x , 0 ) ,\tag{S2}
$$

ahead of $t _ { e \nu } ,$ , then $c ( t _ { e \nu } ) \geq c _ { r e q }$ and $L _ { q } ( t ) = 0$ throughout the surge.

Proof: Under Eq. (7) of the main paper the online capacity rises by one increment per $\tau _ { \mathrm { p r o v } } .$ , so a floor raised a horizon $T _ { \mathrm { a n t } }$ ahead of $t _ { \mathrm { e v } }$ brings $c _ { \mathrm { e v } } +$ min $\{ \lfloor T _ { \mathrm { a n t } } / \tau _ { \mathrm { p r o v } } \rfloor , c _ { \mathrm { m a x } } - c _ { \mathrm { e v } } \}$ servers online by $t _ { \mathrm { e v } }$ . With $T _ { \mathrm { a n t } } ~ \ge ~ T _ { \mathrm { a n t } } ^ { \mathrm { m i n } }$ and $c _ { \mathrm { r e q } } ~ \leq ~ c _ { \mathrm { m a x } }$ this gives $c ( t _ { \mathrm { e v } } ) \geq c _ { \mathrm { r e q } } ,$ hence $c ( t _ { \mathrm { e v } } ) \mu _ { \mathrm { e f f } } \geq \lambda _ { s }$ by the ceiling in (S1). Substituting into Eq. (8) of the main paper with an empty queue gives max $\{ 0 , \lambda _ { s } - c ( t _ { \mathrm { e v } } ) \mu _ { \mathrm { e f f } } \} = 0$ , and the same step applies at every subsequent instant while the rate and the capacity hold. ■

Remark S1. The conclusion is self-consistent with the retry amplification of Eq. (4) of the main paper. Because the backlog stays at zero, no attempt waits past $T _ { 3 0 0 }$ , so $n ( t ) = 1$ and $\lambda _ { \mathrm { e f f } } = \lambda . \mathrm { ~ A ~ }$ surge absorbed with sufficient lead time therefore never amplifies, and $c _ { \mathrm { r e q } }$ is correctly computed against the raw arrival rate rather than a retry-inflated one.

Remark S2. Algorithm 1 instantiates (S2) at $c _ { \mathrm { e v } } ~ = ~ 1$ , the floor the reserve falls back to outside an event window, giving the ramp time $T _ { \mathrm { r a m p } } ~ = ~ ( c _ { \mathrm { r e s } } - 1 ) \tau _ { \mathrm { p r o v } }$ . It raises the floor one control interval $T _ { \mathrm { c t l } }$ earlier still, and the agent sizes $c _ { \mathrm { r e s } }$ to hold utilization below a target rather than at saturation, so $c _ { \mathrm { r e s } } ~ \geq ~ c _ { \mathrm { r e q } } .$ The implemented horizon therefore satisfies Theorem S1 with margin on both terms.

Corollary S1 (Reactive provisioning cannot absorb a fast onset). A purely reactive controller observes the surge only after it begins and so acts with $T _ { a n t } = 0 . \ I f \lambda _ { s } > c _ { e \nu } \mu _ { e f f }$ and $\tau _ { p r o \nu } > 0 ,$ , then for every $t \in [ t _ { e \nu } , t _ { e \nu } { + } T _ { a n t } ^ { \mathrm { m i n } } ]$ the backlog obeys

$$
L _ { q } ( t ) \ge \int _ { t _ { e v } } ^ { t } [ \lambda _ { s } - c _ { f a s t } ( t ^ { \prime } ) \mu _ { e f f } ] d t ^ { \prime } > 0 ,\tag{S3}
$$

where the fastest feasible capacity trajectory is

$$
c _ { f a s t } ( t ) = \operatorname* { m i n } \left\{ c _ { \operatorname* { m a x } } , c _ { e \nu } + \left\lfloor \frac { t - t _ { e \nu } } { \tau _ { p r o \nu } } \right\rfloor \right\} .\tag{S4}
$$

The bound holdsfor every reactive capacity policy irrespective of the objective it optimizes or the weights it carries.

Proof: No causal controller can command capacity before $t _ { \mathrm { e v } } .$ , so by Eq. (7) of the main paper its online capacity is upper bounded pointwise by $\mathcal { C } _ { \mathrm { f a s t } }$ . Substituting this bound into Eq. (8) of the main paper and integrating from $t _ { \mathrm { e v } }$ gives (S3). Strict positivity follows because $\lambda _ { s } > c _ { \mathrm { e v } } \mu _ { \mathrm { e f f } }$ makes the integrand positive at $t _ { \mathrm { e v } }$ , and $\it { C f a s t }$ stays below $c _ { \mathrm { r e q } }$ until $t _ { \mathrm { e v } } + T _ { \mathrm { a n t } } ^ { \mathrm { m i n } }$

Corollary S1 bounds the drift-plus-penalty controller of Eq. (12) of the main paper under every weight pair $( V , W )$ and the rule-based controller alike, since neither can act before load is visible. Lead time is the only remaining lever, and Theorem S1 quantifies how much of it suffices. Supplying that lead time from external context is the role of the calendar tool in the framework of Section III of the main paper.