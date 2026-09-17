# Autonomy in Check: Governor-Mediated Adaptive Security at the Edge

Ijaz Ahmad<sup>∗</sup> , Ijaz Ahmad<sup>†</sup> , Flavio Esposito<sup>‡</sup> , and Erkki Harjula<sup>∗</sup>

∗ Centre for Wireless Communications, University of Oulu, Oulu, Finland

Email: ahmad.ijaz@oulu.fi, erkki.harjula@oulu.fi

<sup>†</sup> VTT Technical Research Centre of Finland, Espoo, Finland

Email: ijaz.ahmad@vtt.fi

<sup>‡</sup> Department of Computer Science, Saint Louis University, St. Louis, Missouri, USA

Email: flavio.esposito@slu.edu

Abstract—Adaptive security at the network edge increasingly relies on automated planners, including rule-based controllers, learned policies, and LLM-assisted agents, that translate observations into enforcement actions. Once such a planner can influence live policy state, syntactic validity is not enough. A semantically wrong action, produced from incomplete or manipulated observations, can be faithfully executed by an enforcement substrate that cannot judge mission context. We address this problem by treating the boundary between planner output and kernel enforcement input as the primary security object. We propose a split-control architecture in which an untrusted planner emits typed security intents, a deterministic governor checks each intent against safety, resource, temporal-stability, and proportionality invariants, and only admitted actions are bound to signed receipts and compiled into pre-installed eBPF map updates. The paper formalizes this trust-boundary problem, defines three threat classes, develops the governor admission predicate, and reports an end-to-end prototype. Across rule-based and LLM-assisted planners on a Raspberry Pi 5 testbed connected to the university’s 5G Test Network, the governor admits, rejects, and bounds intents at microsecond cost without disrupting protected-flow regularity. The contribution is conceptual as much as empirical: adaptive security does not need to trust the author of an action. It needs a mediation boundary that decides whether the action is admissible.

Index Terms—Adaptive security, healthcare, agentic AI, edge computing, eBPF, policy governor, 5G, 6G, resilience, intentbased security, split-control architecture, reference-monitor, policy oscillation.

## I. INTRODUCTION

Modern edge systems increasingly rely on adaptive security mechanisms that adjust enforcement policies in response to changing network conditions, resource constraints, and observed threats. These mechanisms are often driven by automated planners, ranging from rule-based controllers to learning-based and LLM-assisted agents, that translate observations into actions such as monitoring, rate limiting, tier escalation, or isolation [1], [2]. This shift improves responsiveness, but it also changes the security problem. The adaptation process itself becomes part of the attack surface.

Current adaptive security systems often trust planner outputs once they are syntactically valid, even though semantically unsafe actions can emerge from manipulated observations or imperfect reasoning. A planner operating on incomplete, noisy, or adversarially shaped inputs may issue an action that is locally consistent with policy but unsafe for the mission context. The enforcement substrate then executes that action faithfully. The risk is not only that a threat is missed. It is also that a controller may throttle protected traffic, under-react to an active attack, or oscillate rapidly between enforcement states because its inputs have been shaped by an adversary [3]–[5].

This problem is not tied to a single application domain. In industrial control systems, security mechanisms must account for control-loop timing and survivability requirements [6]. In vehicular and emergency-response edges, adaptation must support delay-critical decisions under changing mobility and load [7]. Private 5G deployments face similar pressure because mixed-criticality traffic classes share the same access layer [8]. O-RAN xApp/rApp ecosystems add another instance of the same problem, since control-plane intelligence can influence operational state and may itself be compromised [9].

Healthcare ward networks provide a concrete example of the same pattern. Clinical alarms and routine telemetry may traverse the same access network [10], [11]. An adaptive controller that is too restrictive can delay or suppress urgent flows, while one that is too permissive leaves the same path open to a compromised endpoint. Across these domains, the key question is no longer only how to detect malicious behavior, but how to constrain what an adaptive controller is allowed to do when its inputs cannot be fully trusted.

Existing closed-loop and intent-based control frameworks automate policy refinement and actuation, while programmable substrates such as extended Berkeley Packet Filter (eBPF) provide efficient in-kernel observability and enforcement [12]–[16]. These foundations are important, but efficiency and well-typed actuation do not by themselves establish semantic safety. The kernel can enforce a rate limit or update a policy map, but it cannot decide whether the requested update is appropriate for the current mission context. Likewise, model-level defenses for LLM and agent security reduce planner-side risk, but they do not remove the need to mediate unsafe outputs that may still be produced [17], [18].

This paper treats the planner-enforcement boundary as the primary security object. We propose a split-control architecture grounded in the reference-monitor principle [19]. The planner may use rules, learned logic, or LLM-assisted reasoning, but it cannot directly modify enforcement state. Instead, it emits typed security intents. A deterministic governor mediates each intent against safety, resource, temporal-stability, and proportionality invariants before any action can be compiled into the eBPF enforcement substrate.

This paper makes the following contributions:

• Trust-boundary formulation. We frame adaptive edge security as a trust-boundary problem in which policy-valid actions can be semantically unsafe under manipulated observations or imperfect reasoning. We define three concrete threat classes at the planner-enforcement interface: context manipulation, policy oscillation, and residual-window exploitation (Section III).

• Governor-mediated architecture. We design a deterministic admission predicate over a restricted intent vocabulary. The predicate checks syntactic validity, protected-flow safety, resource headroom, temporal stability, and proportionality before enforcement. This separates intent mediation from planner-specific mechanisms such as action masking in safe reinforcement learning or per-call runtime verification (Section IV).

• Bounded enforcement with integrity. We bind every admitted intent to an HMAC-protected receipt and confine the compiler to a pre-installed set of enforcement state space, so neither the planner nor a compromised compiler can synthesize new policy logic. Every approved action lands in an append-only audit log (Section IV).

• Prototype evaluation under structured threats. We implement the design on a Raspberry Pi 5 attached to the University’s 5G Test Network with three ESP32 endpoints. We evaluate rule-based and LLM-assisted planners under the three threat classes, characterize governor overhead, and compare governed behavior against static and no-governor baselines (Section VI).

The results support a simple design principle: autonomy belongs in planning, but actuation should remain mediated, bounded, and auditable. Section II situates the work against closed-loop control, eBPF enforcement, AI-driven control ecosystems, LLM security, and runtime policy mediation. Section III develops the system and threat model. Section IV presents the governor-centered design. Section V describes the prototype and attack emulation methodology. Section VI reports the evaluation results and discusses deployment implications.

## II. RELATED WORK

Closed-loop and intent-based control. Closed-loop network control has developed through autonomic networking, zerotouch service management, intent-based networking, and edgecloud orchestration. ETSI ZSM models closed-loop automation as an observe, decide, and actuate process with minimal human intervention [12]. Intent-based networking similarly raises the interface from low-level configuration to desired outcomes, leaving the system to refine those outcomes into device-level actions [13]. Recent work on adaptive trust and edge-cloud orchestration applies related ideas to 6G and IoT settings, where service quality, resource headroom, and security posture must be adjusted at runtime [1], [20]. These systems are important foundations for adaptive security, but they typically treat the controller, planner, or intent source as trusted once its output is well formed. Our work focuses on the complementary problem. A planner output may be syntactically valid and still semantically unsafe when observations are manipulated or reasoning is imperfect.

Programmable enforcement with eBPF. eBPF has become a practical substrate for programmable observability, traffic control, and cloud-native security because it offers stable kernel hooks, verifier-checked programs, and low-overhead access to packet and system events [14], [16]. Recent systems also use eBPF to observe AI and agent workloads at the system level [21]. At the same time, work on BPF hardening and kernel-extension safety shows that verifier acceptance and efficient execution are not the same as end-to-end security correctness [22]–[24]. eBPF can enforce a rate limit, select an inspection path, or update a policy map efficiently. It does not decide whether the requested action is appropriate for the mission context.

Open and AI-driven control ecosystems. Open and programmable control ecosystems make the planner–enforcement boundary more important. O-RAN introduces xApps and rApps that can optimize radio and service behavior through near-real-time and non-real-time control loops, but this openness also creates risks from compromised or malicious control functions [9], [25]. Recent agentic-AI proposals extend this direction by using LLM-assisted reasoning for closed-loop RAN or network-security tasks [17], [18]. LLM-integrated systems also face prompt injection, indirect prompt injection, and adversarially shaped inputs that can cause an agent to infer or follow unintended context [3]. Guidance on deployed AI systems further emphasizes that monitoring remains difficult once AI components interact with tools, external data, and operational environments [4], [5]. These works reduce plannerside risk, but they do not remove the need to mediate unsafe outputs that may still be produced.

Runtime safety and policy mediation. The closest conceptual anchor for our work is runtime policy mediation. The reference-monitor principle requires security-sensitive operations to be mediated by a mechanism that is always invoked, tamper-resistant, and small enough to reason about [19]. Schneider’s work on enforceable security policies formalizes policies that can be enforced by monitoring execution and suppressing disallowed actions [26]. Runtime verification checks execution traces against formal specifications while a system runs [27]. In learning-enabled control, shielding prevents unsafe actions from a learner before they affect the environment [28]. Our governor follows the same safetyoriented spirit, but it mediates a different object. A shield typically masks individual learner actions against a verified safety model. Runtime verification observes program traces against a specification. Intent-based frameworks refine desired outcomes into actions while usually assuming that the intent source is trusted. Our governor instead mediates typed security intents before they can modify live enforcement state. The planner is explicitly outside the trusted computing base, and each proposed intent must clear the predicate Γ over deploymentspecific invariants such as $\mathcal { F } _ { \mathrm { c r i t } } , r _ { \mathrm { m i n } } , \Delta _ { \mathrm { m i n } } .$ , and ϕ.

Identified gap. Prior work has advanced adaptive control, programmable enforcement, open control ecosystems, AIassisted security, and runtime safety. These lines of work make critical edge systems more flexible, but they often assume that a well-formed controller output is safe to execute. We take a different position. In critical edge deployments, the planner’s output is part of the attack surface. A semantically wrong but syntactically valid action can delay protected traffic, weaken containment, or drive unstable policy changes. Our architecture therefore separates planning from actuation: the planner proposes, the governor mediates, and the eBPF substrate enforces only actions that pass explicit safety and stability checks.

## III. TRUST BOUNDARY AND THREAT MODEL

## A. Deployment Context and System Model

We consider a critical edge network carrying mixedcriticality traffic over a 5G access layer, with processing tiers at the local edge, MEC, and the cloud. Some flows are urgent and delay-sensitive, while others are routine and tolerant of inspection or queuing delays. A security controller observes the communication state and decides whether to continue passive monitoring, rate-limit a traffic class, escalate inspection to a higher-capacity tier, or isolate a flow subset. Each decision affects both security posture and the service quality of the flows under management.

The proposed defense mechanism is structured as three logical modules. A planner receives the current observation state and proposes a typed security intent. A governor checks whether that intent is admissible under the current mission and resource constraints. A compiler translates approved intents into bounded updates over a pre-installed eBPF enforcement substrate. The planner may use rule-based logic, a learned policy, or an agentic reasoning component. Its internal mechanism is outside the trust boundary argument. What matters is that it cannot modify enforcement state without governor approval.

## B. Threat Model

The governor and enforcement substrate are part of the trusted computing base, whereas the planner is not. The planner therefore cannot load arbitrary eBPF programs, hold file descriptors to enforcement maps, or directly invoke the compiler. This boundary is what we study, not something we abstract away. The adversary may inject malicious traffic [29], compromise a non-critical endpoint device, or influence the telemetry and context presented to the planner. From these capabilities, three threat classes emerge, summarized in Table I and illustrated in Fig. 1.

T1 – Context manipulation. The adversary manipulates observable inputs, such as inflated anomaly signals, masked malicious flows, or workloads shaped to mislead, to induce an action that is syntactically valid but semantically wrong. Suppressing the measurement of an ongoing attack, for instance, causes the planner to perceive low threat and propose passive monitoring when isolation is warranted. This class maps directly onto indirect injection attacks studied in LLMintegrated systems [3], where the attack targets the input environment rather than the model itself.

T2 – Policy oscillation. Rather than inducing a single large policy error, the adversary alternates observable behavior at a rate $f _ { \mathrm { a d v } }$ chosen to keep the planner near a decision boundary. If $f _ { \mathrm { a d v } } > 1 / { \Delta _ { \mathrm { m i n } } } ,$ where $\Delta _ { \mathrm { m i n } }$ is the governor’s minimum inter-action cooldown, each enforcement transition creates a brief window during which the active policy is mismatched with the actual traffic condition. Repeated over time, this degrades both protection coverage and service quality without requiring any single catastrophic decision.

T3 – Residual-window exploitation. Every enforcement state transition involves a brief window before the new policy takes full stable effect. An adversary who can trigger policy transitions, whether through context manipulation or oscillation, can time traffic delivery to coincide with this window. The primary mitigation is bounding the time from governor approval to stable kernel enforcement, addressed in Section IV-C.

TABLE I  
THREAT CLASSES AT THE PLANNER–GOVERNOR INTERFACE
<table><tr><td></td><td>Class Adversary action</td><td>Governor defense</td></tr><tr><td>T1</td><td>Shape observations to deceive the planner</td><td>Proportionality invariant (I5); uncertainty gating</td></tr><tr><td>T2</td><td>Alternate behavior to drive rapid enforcement switching</td><td>Temporal stability invari- ant (I4)</td></tr><tr><td>T3</td><td>Time delivery to coincide with transition gaps</td><td>Bounded compiler la- tency; receipt sequencing</td></tr></table>

![](images/003e5e7c7ff9e1692274136a2e304c11b1c471dd896b362ecc42c2003bb26f01.jpg)  
Fig. 1. Threat-annotated pipeline overview. The adversary targets three interfaces in the adaptation path.

## C. Problem Formulation: Governor-Mediated Actuation

To formalize how the governor constrains planner actions under the threats described above, we model each decision as an intent proposed from the current system state and independently admitted before enforcement. In particular, at decision epoch t, the security controller observes

$$
\begin{array} { r } { \mathbf { x } _ { t } = [ \mathbf { o } _ { t } , \mathbf { c } _ { t } , \mathbf { r } _ { t } , \mathbf { q } _ { t } , \mathbf { h } _ { t } , \mathbf { m } _ { t } , u _ { t } ] , } \end{array}\tag{1}
$$

where $\mathbf { o } _ { t }$ denotes traffic observations, $\mathbf { c } _ { t }$ flow criticality labels, $\mathbf { r } _ { t }$ resource headroom per enforcement tier, $\mathbf { q } _ { t }$ queue and path state, $\mathbf { h } _ { t }$ a bounded history of recent policy decisions, m mission-context indicators such as active alarm sessions and protected service classes, and $u _ { t } \in [ 0 , 1 ]$ an uncertainty signal derived from measurement freshness and cross-source consistency.

Let A denote the typed intent vocabulary and let ${ \mathcal { L } } =$ {local, MEC, cloud} denote the available execution tiers. A planner proposes an intent $a _ { t } \in \mathcal A$ and a tier $\ell _ { t } \in \mathcal { L }$ , which can be abstracted as the following planner-side problem:

$$
\begin{array} { r l } { \underset { a _ { t } \in \mathcal { A } } { \operatorname* { m i n } } } & { R _ { t } ( a _ { t } , \ell _ { t } ) + \lambda C _ { t } ( a _ { t } , \ell _ { t } ) + \mu O _ { t } ( a _ { t } , \mathbf { h } _ { t } ) } \\ { \ell _ { t } \in \mathcal { L } } \\ { \mathrm { s . t . ~ } } & { D _ { t } ^ { \mathrm { { p r o t } } } \leq D _ { \operatorname* { m a x } } , } \\ & { U _ { t } \leq U _ { \operatorname* { m a x } } , } \\ & { a _ { t } \in \mathcal { A } _ { \mathrm { s a f e } } ( \mathbf { x } _ { t } ) . } \end{array}\tag{2}
$$

Here, $R _ { t }$ denotes the planner’s estimate of residual security risk, $C _ { t }$ denotes resource and coordination cost, and $O _ { t }$ penalizes unstable policy changes such as reversals on the same scope. $D _ { t } ^ { \mathrm { p r o t } }$ is the protected-flow service bound and $U _ { t }$ is a resource-utilization bound. The state $\mathbf { x } _ { t }$ is supplied by the state builder at each epoch. The safe action set $\mathcal { A } _ { \mathrm { s a f e } } ( \mathbf { x } _ { t } )$ is not computed by the planner. It is the output of the governor admission predicate, evaluated independently of how the proposed intent was generated. A planner operating on manipulated observations may still produce an intent in A; the governor determines whether that intent belongs in $\mathcal { A } _ { \mathrm { s a f e } } ( \mathbf { x } _ { t } )$ under the actual system state. This separation between proposal and admissibility is the central structural property of the design.

Prototype instantiation. Equation (2) is a conceptual abstraction of the planner-side objective. The prototype does not implement a numerical optimizer for $R _ { t } , \ C _ { t } ,$ or $O _ { t } ,$ and the governor does not trust the planner’s estimates of these terms. Instead, the governor evaluates only the invariant predicate in Section IV-B. The state builder provides the a bounded severity signal from smoothed drop and queuepressure indicators, the uncertainty signal $u _ { t } ,$ , and the intent impact ordinal shown in Table II. The proportionality bound is $\phi ( s , u ) = \operatorname* { m a x } ( 1 , 4 s \cdot c ( u ) )$ , where $c ( u ) = 1$ for $u \leq 0 . 5$ and decreases linearly to 0 at $u = 1$ . The floor of 1 keeps ROLL-BACK admissible at any uncertainty. The cooldown $\Delta _ { \mathrm { m i n } }$ and reserve $r _ { \mathrm { m i n } }$ instantiate the temporal-stability and protectedheadroom checks used by I4 and I3.

## IV. GOVERNOR-CENTERED DESIGN

## A. Bounded Intent Vocabulary

Restricting the planner to a fixed vocabulary is the first step in bounding its actuation authority. Rather than accepting free-form policy expressions or arbitrary kernel operations, the governor sees only the typed intents in Table II. The vocabulary covers passive observation, bounded rate control, tier escalation, non-critical isolation, human-approval requests, and state rollback. High-impact actions are named explicitly so the governor can apply proportionality checks independently of planner intent.

TABLE II  
TYPED SECURITY INTENT VOCABULARY WITH IMPACT ORDINAL.
<table><tr><td>Intent (ord)</td><td>Parameters and effect</td></tr><tr><td>MONITOR (0)</td><td>scope, duration; passive observation, no traffic effect</td></tr><tr><td>RATELIMIT (2)</td><td>scope, rate tier, duration; bounded throttling with mandatory exemptions for flows in  $\mathcal { F } _ { \mathrm { c r i t } }$ </td></tr><tr><td>ESCALATE (3)</td><td>scope, target tier, timeout; requests higher-capacity inspection without modifying the local rate policy</td></tr><tr><td></td><td>IsOLATE-NC (4) scope, duration; isolates only non-critical flows or devices; critical flows are unaffected by construction</td></tr><tr><td>REQAPPROVAL (0)</td><td>reason, suggested action, urgency; suspends direct actuation and requests operator review</td></tr><tr><td>ROLLBACK (1)</td><td>receipt-id, target state; reverts to the last governor- approved stable enforcement state</td></tr></table>

## B. Admission Predicate

The governor admits $a _ { t }$ if and only if $\Gamma ( a _ { t } , { \bf x } _ { t } ) = 1$ , where $\Gamma = I _ { 1 } \wedge I _ { 2 } \wedge I _ { 3 } \wedge I _ { 4 } \wedge I _ { 5 }$ . The safe action set follows as $\mathcal { A } _ { \mathrm { s a f e } } ( \mathbf { x } _ { t } ) = \{ a \in \mathcal { A } | \Gamma ( a , \mathbf { x } _ { t } ) = 1 \}$ . Each invariant addresses a distinct failure mode.

I1 – Syntactic validity. $a _ { t }$ belongs to the declared vocabulary with valid parameter types and in-range values. Syntactic validity is necessary but not sufficient; the remaining invariants check semantic correctness.

I2 – Flow safety. A restrictive action may not affect flows in $\mathcal { F } _ { \mathrm { c r i t } }$ without an explicit policy exemption:

$$
\begin{array} { r l } & { \mathrm { s c o p e } ( a _ { t } ) \cap \mathcal { F } _ { \mathrm { c r i t } } = \emptyset } \\ & { \qquad \vee \mathrm { t y p e } ( a _ { t } ) \in \{ \mathrm { R E Q A P P R O V A L } , \mathrm { R o L L B A C K } \} . } \end{array}\tag{3}
$$

Regardless of what the planner proposes, this invariant prevents urgent flows from being throttled or isolated without explicit authorization.

I3 – Resource bound. Applying $a _ { t }$ must leave sufficient bandwidth headroom for protected services:

$$
\mathrm { h e a d r o o m } ( \mathbf { r } _ { t } , a _ { t } ) \geq r _ { \operatorname* { m i n } } .\tag{4}
$$

I4 – Temporal stability. The last approved action on scope(a ) must have occurred at least $\Delta _ { \mathrm { m i n } }$ epochs prior:

$$
t - t _ { \mathrm { l a s t } } ( \mathrm { s c o p e } ( a _ { t } ) ) \geq \Delta _ { \mathrm { m i n } } .\tag{5}
$$

An adversary driving oscillation at $f _ { \mathrm { a d v } } > 1 / { \Delta _ { \mathrm { m i n } } }$ cannot force enforcement changes at that rate. The governor rejects each intent until the cooldown elapses, bounding the achievable oscillation rate above by $1 / \Delta _ { \operatorname* { m i n } }$ regardless of how quickly the planner proposes new intents.

I5 – Proportionality. The impact of $a _ { t }$ must not exceed what observed threat severity and current uncertainty jointly support:

$$
\mathrm { i m p a c t } ( a _ { t } ) \leq \phi ( \mathrm { s e v e r i t y } ( \mathbf { o } _ { t } ) , u _ { t } ) ,\tag{6}
$$

where $\phi$ decreases monotonically in $u _ { t }$ . As uncertainty rises, ϕ contracts toward Low, directing the planner to MONITOR, REQAPPROVAL, or ROLLBACK rather than direct mitigation.

On rejection, the governor returns a typed reason identifying which invariant failed and, where applicable, a lower-impact alternative. This allows the planner to revise its proposal without a full state reset, preserving adaptability while maintaining the admission boundary. Each invariant removes a different unsafe path from the planner to enforcement. I1 prevents malformed or out-of-range intents from entering the compiler path. I2 protects flows in $F _ { \mathrm { c r i t } }$ from restrictive actions unless an explicit policy exemption exists. I3 prevents an admitted action from consuming the headroom reserved for protected services. I4 bounds the rate of policy transitions on a scope and therefore limits adversary-induced oscillation. I5 prevents high-impact actions when the observed severity and uncertainty do not justify them. Section VI shows how these invariants fire under the three attack classes and under nominal LLM-assisted planning.

## C. Integrity Path to Kernel Enforcement

An intent that passes the admission check is not written to enforcement maps directly by the planner or governor. Instead, the governor emits a signed approval record:

$$
\begin{array} { r } { \rho _ { t } = \big \langle t , a _ { t } , \mathcal { H } ( \mathbf { x } _ { t } ) , \mathrm { H M A C } _ { k } \big ( t \| a _ { t } \| \mathcal { H } ( \mathbf { x } _ { t } ) \big ) \big \rangle , } \end{array}\tag{7}
$$

where $\mathcal { H } ( \mathbf { x } _ { t } )$ is a hash of the decision-context snapshot and k is held exclusively by the governor process. A privileged compiler service, which is the only component holding eBPF map file descriptors and the relevant Linux capabilities, verifies $\rho _ { t }$ before performing any update. It then translates the approved intent into a bounded set of kernel-facing operations. These include selecting a pre-defined rate tier, enabling a predefined inspection path, or writing a schema-validated entry into a policy map.

## D. Proposed Split-Control Architecture

The proposed framework has six logical components arranged as a unidirectional enforcement pipeline, shown in Fig. 2. Autonomy is confined to the planning stage. All live actuation authority rests within the trusted base.

Observation plane. Kernel-resident eBPF programs attached at tc and XDP hooks export per-class packet and byte rates, drop counts, queue pressure, protected-flow activity indicators, and policy-map access timestamps through a ring buffer. These programs are pre-compiled and pre-verified at deployment. No user-space component can load or modify them.

State builder. A user-space daemon assembles $\mathbf { x } _ { t }$ from ringbuffer data, computes $u _ { t }$ as a confidence-weighted function of measurement freshness and cross-source consistency, and maintains $\mathbf { h } _ { t }$ as a bounded sliding window of past governor decisions and observed flow-quality outcomes.

Planner. The planner receives $\mathbf { x } _ { t }$ and returns a structured intent from the vocabulary in Table II. The implementation may be an LLM-based agent over a typed tool interface, a learned policy, or a rule-based controller. The architecture is independent of this choice.

Governor. The governor evaluates $\Gamma ( a _ { t } , \mathbf { x } _ { t } )$ per Section IV-B, emits $\rho _ { t }$ on admission, and returns a typed rejection with an optional lower-impact suggestion otherwise.

![](images/cae734d8e61bc0be4013e7d7b20c2f4d67107108e433b390dba26c75936fbbcd.jpg)

Fig. 2. Compact split-control architecture. The planner proposes typed security intents from the built state, but only the governor may admit them for enforcement. Approved actions become bounded eBPF map updates after compiler-side receipt verification. Rejected actions return typed feedback for replanning.  
![](images/125ca8aae1ccb4f1e7678b1e36a7ec97f849d2783d55de4a81fd1164b1da515b.jpg)  
Fig. 3. Testbed topology. ESP32 nodes connect to the RPi 5 enforcement node over WiFi (wlan0). The RPi 5 runs the state builder, governor, and planner, with eBPF hooks on wlan0 for enforcement. The wwan0 interface provides 5G connectivity to the 5GTN core.

Compiler. After verifying the HMAC in $\rho _ { t }$ , the compiler translates the approved intent into bounded map updates and appends the receipt to the audit log.

Enforcer. Approved actions become bounded eBPF map updates after compiler-side receipt verification.

## V. PROTOTYPE IMPLEMENTATION

## A. 5G Edge Testbed

The prototype runs on the University’s standalone 5G Test Network (5GTN) [30], an academic research and experimentation platform operating on the n78 band (3.5 GHz) with an on-premise 5G core. A Raspberry Pi 5 (4-core Cortex-A76, 8 GB RAM, kernel 6.12.79) serves as the enforcement node and gateway. Its wlan0 interface connects to the ESP32- S3 endpoint devices over a local WiFi access point, and its wwan0 interface provides 5G connectivity to the 5GTN core over the cellular uplink. eBPF programs compiled against a BTF-generated vmlinux.h attach at tc ingress on wlan0 and use three BPF maps: an LRU hash for per-source epoch rate counters, a hash for compiler-written rate limits, and a 128 kB ring buffer for telemetry export. All user-space components run on the RPi5, including a Go state builder that assembles x<sub>t</sub> every 100 ms epoch, a Go governor, a privileged Go compiler, and a planner. Fig. 3 shows the testbed topology.

Two traffic classes originate from the ESP32-S3 firmware. The protected alarm class generates 20-byte UDP datagrams at

10 Hz, corresponding to a nominal 100 ms inter-arrival period, while the routine class generates 1 kB telemetry bursts at 1 Hz. We do not measure the ESP32-to-RPi one-way latency as an evaluation metric because the long runs exposed cross-device clock drift and occasional invalid negative latency samples. Instead, protected-flow service is evaluated using receiverside cadence. Inter-arrival times are computed directly from RPi receive timestamps, grouped by source IP and sequence number. Baseline runs last approximately 30 minutes, and each attack run lasts approximately five minutes. The LLM planner uses llama3.1:8b in Q4\_K\_M quantized form, served through Ollama [31] on a co-located GPU node (RTX 4000 Ada, 20 GB VRAM). The planner wrapper submits only structured tool-call intents to the governor. Non-tool or no-op outputs from the LLM are logged separately and do not update enforcement state. The prototype uses $\Delta _ { \operatorname* { m i n } } ~ = ~ 5$ epochs, corresponding to 500 ms, and $r _ { \mathrm { m i n } } = 6 . .$ 4 kbps, a 2× reserve over the nominal protected-alarm rate.

## B. Adversarial Workloads

The three threat classes in Table I are executed independently with a 30 ms cooldown to avoid overlap. For T1, the context-manipulation script suppresses alarm-source reports from two of the three ESP32 nodes before they reach the state builder, causing the planner’s view of protected-flow activity to diverge from the receiver-side traffic record. For T2, a test process injects intents directly to the governor’s Unix socket at $f _ { \mathrm { a d v } } ~ = ~ 2 / { \Delta _ { \mathrm { m i n } } }$ . This models a co-located adversarial process and represents a conservative worst case, since a remote attacker would face additional transit latency that reduces effective injection frequency. For T3, hping3 on the RPi5 injects UDP bursts timed to epoch boundaries, immediately after a governor-approved enforcement transition, to probe the stabilisation window.

## VI. EVALUATION

This section evaluates the proposed split-control design in the 5G test network. The goal is not only to measure enforcement performance but also determine whether the governor preserves the security boundary between planner output and kernel actuation under realistic and adversarial conditions. We therefore examine four aspects: protected-flow cadence, blocking of unsafe actions, control-path overhead, and robustness to planner-side manipulation.

## A. Security and Performance Results

The evaluation uses four configuration labels, listed in Table III. We use the friendly labels in figures and prose, while the internal run names are kept in the artifact CSV files for reproducibility. Static is the no-adaptation control, Rule is the deterministic planner with governor, Rule, no gov. is the ablation that removes the admission boundary, and LLM is the proposed planner-governor configuration. REQAPPROVAL and ROLLBACK are part of the vocabulary but are not triggered by the attack workload in this evaluation. Their inclusion lets the proportionality bound contract toward operator-mediated states rather than direct mitigation when uncertainty is high.

TABLE III  
EXPERIMENTAL CONFIGURATIONS AND RECEIVER-SIDE PROTECTED-FLOW CADENCE.
<table><tr><td>Config.</td><td>Role</td><td>intervals</td><td>n p99 p99.9 excess (ms)</td><td>count</td><td>gaps &gt; 150</td><td>ms</td></tr><tr><td>Static</td><td>No adaptation control</td><td>54032</td><td>5.84</td><td>18.31</td><td>0</td><td>3</td></tr><tr><td>Rule</td><td>Rule planner with gover- nor</td><td>54092 7.27</td><td></td><td>40.40</td><td>0</td><td>18</td></tr><tr><td>Rule, no gov.</td><td>Governor ablation</td><td>54061 7.02</td><td></td><td>41.02</td><td>0</td><td>23</td></tr><tr><td>LLM</td><td>LLM planner with gov- ernor</td><td></td><td></td><td>54091 6.51 33.78</td><td>0</td><td>19</td></tr></table>

Note: Excess is measured over the nominal 100 ms alarm period using only RPi receive timestamps.

M1 – Receiver-side protected-flow cadence (Fig. 4, Table III). The earlier version of the experiment reported ESP32-to-RPi one-way latency. We do not use that metric here because the long runs revealed cross-device clock drift and invalid negative samples. M1 therefore asks a narrower and more reliable question: does the protected alarm stream remain regular at the receiver? Across all four 30-minute baseline runs, the answer is yes. The p99 excess over the nominal 100 ms alarm period remains below 7.3 ms, and no sequence gaps are observed in any configuration. The p99.9 excess remains below the 50 ms service slack for all four baselines. Rare intervals above 150 ms are reported explicitly in Table III rather than hidden inside a latency CDF.

![](images/1cec17ac62c7c765dbc15728895f8aaa3641ce3ce2ae9a9dce8a81b3fae39465.jpg)  
Fig. 4. Protected-flow delivery cadence computed from receiver-side timestamps only. The figure shows excess inter-arrival time over the nominal 100 ms alarm period. The 50 ms reference corresponds to the service slack used in the original alarm budget. Sequence-gap counts are reported in Table III.

M2 – Attack containment (Table IV). The attack experiments exercise the governor boundary in three different ways. T1 activates the context-manipulation path in the three-ESP32 testbed: suppressing two alarm sources raises $u _ { t }$ to 0.600 during the attack window. This is not reported as a traffic-latency result. It is evidence that the uncertainty channel responds when the planner’s context is manipulated. In T2, I4 rejects 49.7% of injected oscillation intents, which is the expected alternate-intent clipping pattern at $f _ { \mathrm { a d v } } = 2 / \Delta _ { \mathrm { m i n } }$ . T3 remains partially contained as the residual window is bounded but nonzero, with an average window of 2.616 ms and an average admitted burst of 1279.2 B. We report this as a bounded residual leak rather than claiming complete elimination of transition exposure. The proportionality contraction engages only when severity also rises; under T1 with nominal traffic, severity stays low and I5 fires only when the LLM proposes high-impact actions in this elevated-uncertainty state. The rule planner under B2 remained at low impact, so its I5 rate stays near zero — see Fig. 7.

TABLE IV ATTACK-CONTAINMENT SUMMARY.
<table><tr><td></td><td>Attack Measure</td><td>Observed Result</td><td></td></tr><tr><td>T1</td><td>max ut during attack</td><td>0.600</td><td>YES</td></tr><tr><td>T2</td><td>I4 rejection rate</td><td>49.67%</td><td>YES</td></tr><tr><td>T3</td><td>avg window / burst</td><td>2.616 ms 1279 B</td><td>PARTIAL</td></tr></table>

T1: context manipulation; T2: policy oscillation; T3: residual-window attack.

![](images/cd5e60db03d5dc6e8243d85a6d1afbcd0518a177f3422400c4cc623cbe550561.jpg)  
Fig. 5. Governor decision overhead across nominal and attack scenarios. For each scenario, the open circle is the mean and the filled square is the p99; the connecting line is the spread. All p99 values remain below 1 ms, so the governor is sub-millisecond even under active attack.

M3 – Governor overhead (Fig. 5, Table V). Governor interposition remains sub-millisecond in every scenario. Across nominal and attack runs, p99 decision overhead ranges from $7 6 . 2 4 \mu \mathrm { s }$ to $1 2 9 . 0 9 \mu \mathrm { s } .$ , well below the 1 ms reference used in Fig. 5. The result separates the safety boundary from the planner’s complexity. The governor check is fast even when the planner is stochastic or the input stream is adversarial. The original 5 ms engineering guardrail is still comfortably met, but a more impactful result for critical-edge systems is that admission control stays below one millisecond at p99.

M4 – Governance tradeoff across configurations (Fig. 6). Fig. 6 compares the proposed governed configurations with two natural alternatives: Static, which has no adaptation channel, and Rule, no gov., where the rule planner writes to the compiler without the admission predicate. Panel (a) reports the fraction of attack-induced structured intents that reached enforcement. Static is marked N/A because it emits no adaptive intents. Rule, no gov. admits all planner-emitted intents by construction, while the governed Rule and LLM configurations admit 86.9% and 48.0%, respectively, because rejected intents do not reach the compiler. Panel (b) shows the utility side: nominal p99 cadence excess remains within a 1.5 ms band across all four configurations. The governor therefore changes what can be actuated under attack without measurably degrading receiver-side protected-flow cadence.

TABLE V GOVERNOR DECISION OVERHEAD.
<table><tr><td>Scenario</td><td>n</td><td>mean (µs)</td><td>p99 (µs)</td></tr><tr><td>Rule+Gov, nominal</td><td>1801</td><td>54.34</td><td>76.24</td></tr><tr><td>Rule+Gov, T1</td><td>301</td><td>55.11</td><td>82.56</td></tr><tr><td>Rule+Gov, T2</td><td>601</td><td>41.33</td><td>78.52</td></tr><tr><td>Rule+Gov, T3</td><td>321</td><td>54.08</td><td>102.22</td></tr><tr><td>LLM+Gov, nominal</td><td>51</td><td>27.86</td><td>129.09</td></tr><tr><td>LLM+Gov, T1</td><td>10</td><td>27.86</td><td>100.78</td></tr><tr><td>LLM+Gov, T2</td><td>309</td><td>30.00</td><td>79.20</td></tr><tr><td>LLM+Gov, T3</td><td>25</td><td>32.03</td><td>126.26</td></tr></table>

![](images/1053ba2b0453ab7a651d68e87b458d4e21348816732ff02b1a1664e00195fcfe.jpg)

![](images/82d041e50172967bd9df441e9d8ef916228c603d06b2861360b211b997dddc6e.jpg)  
Fig. 6. Governance tradeoff across four configurations. (a) Attack-intent admit rate. Static has no adaptation channel and is marked N/A. Rule, no gov. admits all planner-emitted intents because no admission predicate is present. Rule and LLM with governor admit 86.9% and 48.0% of attack-induced structured intents, respectively. (b) Nominal p99 cadence excess over the 100 ms alarm period. All four configurations remain within a 1.5 ms band. Static is hatched to mark the absence of adaptation.

M5 – Planner abuse robustness (Fig. 7, Table VI). The governor’s value is clearest when the planner is imperfect. In the nominal LLM run, the wrapper logged 473 LLM decisions. Only 51 were structured intents submitted to the governor; the remaining 422 were non-actuating outputs, mostly prose or JSON-like text converted to do\_nothing. Among the 51 structured intents, 13 were admitted and 38 were rejected before enforcement: 19 by I1, five by I2, and 14 by I5. Table VI gives representative examples.

Under T2, both planner families show the same I4 signature, with 149 cooldown rejections. Under T3, both paths show ten I2 flow-safety rejections. The same admission boundary applies regardless of how the intent was generated. Invariant I3 reserves $r _ { \operatorname* { m i n } } = 6 .$ .4 kbps of bandwidth headroom for the protected class, a 2× margin over the nominal alarm-class

![](images/5ffad71c887dae0817a4488cfd2df9f8581ae57235912aeb19e7a8b6408b326d.jpg)  
Fig. 7. Intent disposition by governor invariant. T1 represents context manipulation, T2 is policy oscillation, and T3 is residual-window exploitation.

TABLE VI  
EXAMPLES OF STRUCTURED LLM INTENTS STOPPED BY THE GOVERNOR.
<table><tr><td>Proposed intent</td><td>Inv. Governor response</td></tr><tr><td>RATELIMIT with I1 zero rate</td><td>Invalid parameter; RATELIMIT re- quires a positive rate.</td></tr><tr><td>RATELIMIT I2 protected source</td><td>Restrictive action targets a source carrying alarm-class traffic.</td></tr><tr><td>ESCALATE global I5</td><td>Action impact exceeds the propor- tionality bound.</td></tr><tr><td>ISOLATE-NC high- I5 impact scope</td><td>Isolation impact is not justified by observed severity and uncertainty.</td></tr><tr><td colspan="2">load. In the measured workload, the alarm and routine traffic remain below this floor, so I3 does not activate. This is expected: I3 is a structural guardrail for saturation conditions, while I1, I2, I4, and I5 exercise the planner-governor boundary</td></tr></table>

## B. Deployment Implications

The results apply broadly to mixed-criticality scenarios in edge-could continuum. The protected flow set, the service budget for protected flows, and the cooldown $\Delta _ { \mathrm { m i n } }$ are the main deployment-specific parameters. In a healthcare ward, $\mathcal { F } _ { \mathrm { c r i t } }$ is the alarm class and a manipulated planner that throttles alarms threatens patient safety [10], [11]. In industrial control, $\mathcal { F } _ { \mathrm { c r i t } }$ becomes the control-plane messages and the service budget is the process cycle time. In vehicular edge, $\Delta _ { \mathrm { m i n } }$ shortens to match faster control decisions. The invariant logic, the HMAC receipt chain, and the eBPF substrate carry over unchanged.

The prototype uses $\Delta _ { \operatorname* { m i n } } = 5$ epochs (500 ms) and $r _ { \mathrm { m i n } } =$ 6.4 kbps, corresponding to a 2× reserve over the nominal protected-alarm rate. These parameters expose the expected safety–agility tradeoff. A larger $\Delta _ { \mathrm { m i n } }$ makes oscillation attacks harder by enforcing a longer cooldown between successive policy changes, but also slows adaptation to benign changes in operating conditions. A smaller $\Delta _ { \mathrm { m i n } }$ improves responsiveness, but increases susceptibility to oscillatory manipulation. Likewise, a larger $r _ { \mathrm { m i n } }$ preserves more headroom for protected traffic and tightens the governor’s admission boundary, while a smaller $r _ { \mathrm { m i n } }$ increases enforcement flexibility at the cost of a weaker safety margin. We leave a systematic parametersensitivity study to future work.

The T2 result reflects why temporal stability matters in clinical and industrial edge settings. An adversary does not need to defeat the enforcement mechanism outright. Repeatedly forcing policy transitions is enough to create short periods of mismatch between the active policy and the traffic condition. I4 clips this behaviour at the admission boundary. In our experiment, 149 of the 300 injected oscillation intents fail the cooldown check, producing the expected near-50% clipping pattern for $f _ { \mathrm { a d v } } ~ = ~ 2 / { \Delta _ { \mathrm { m i n } } }$ . This gives operators a simple deployment knob. $\Delta _ { \mathrm { m i n } }$ can be chosen with the protected-flow period and acceptable adaptation rate in mind.

The prototype evaluates the planner-governor boundary at the unit-cell scale. Three endpoints and a single enforcement node is the right granularity for the structural claims (I2 flow safety, I4 cooldown bounding, HMAC integrity). Larger-scale evidence possibly with synthetic flow replay across thousands of sources, multi-gateway coordination of cooldown state, and saturation testing for I3 is left for the extended version with a packet-level emulator.

## C. Scope and Implications

Our prototype evaluates a deliberately small deployment: one RPi5 enforcement node, three endpoints, and a 5G uplink. An open problem is the coordination of governor state across multiple enforcement points in larger deployments.

As importantly, the results suggest that adaptive security does not require trusting the component that makes the decision. The planner can be rule-based, learned, or LLM-assisted, while the same governor independently controls what is allowed to reach the enforcement layer. This separation is useful as planners become more capable but also harder to predict: new planning mechanisms can be introduced without granting them direct control over critical network state. The governor therefore provides a stable security boundary between evolving autonomous intelligence and the enforcement substrate. The experiments also expose a practical limit of this approach. Admission control can reject unsafe actions and bound how frequently policies change, but it cannot eliminate the short transition window after an action is approved. Applications with stricter timing requirements will therefore need faster enforcement activation in addition to governor-mediated admission.

## VII. CONCLUSION

Adaptive security in critical edge systems fails not only when a planner misses a threat, but also when enforcement faithfully executes an unsafe decision based on manipulated observations. This paper treats the planner–enforcement boundary as the primary security object and protects it through a governor with five admission invariants. A prototype on the University’s 5GTN testbed with in-kernel eBPF enforcement shows that protected alarm delivery remains regular, with p99 cadence excess below 7.3 ms and no sequence gaps across baseline runs. Governor decisions remain sub-millisecond, with p99 below 130 $\mu \mathrm { s }$ across nominal and attack scenarios. Under policy oscillation, the governor rejects 49.7% of injected intents, while under nominal LLM planning it rejects 38 of 51 structured intents before enforcement. These results show that adaptive planners can retain flexibility without being granted direct authority over critical enforcement state.

## ACKNOWLEDGMENTS

This research is supported by the Finnish Doctoral Program Network in Artificial Intelligence, AI-DOC (decision number VN/3137/2024-OKM-6), Business Finland funded projects TOMOHEAD (8095/31/2022) and SUNSET-6G (8682/31/2022), and by the Research Council of Finland funded projects 6G Flagship (369116) and Profi6 (336449). The work of Dr. Flavio Esposito is supported by USA NSF Awards CNS #2133407 and OAC #2530896.

## REFERENCES

[1] I. Ahmad, S. Gimhana, I. Ahmad, and E. Harjula, “Adaptive trust architecture for secure IoT communication in 6G,” IEEE Networking Letters, vol. 7, no. 2, pp. 113–116, 2025.

[2] A. Sangiorgi, A. Pinto, R. Tourani, and F. Esposito, “Mitigating deauthentication DoS attacks in 802.11 via eBPF and XDP,” in Proceedings of the IEEE Conference on Network Softwarization (NetSoft), Budapest, Hungary, June 2025.

[3] K. Greshake, S. Abdelnabi, S. Mishra, C. Endres, T. Holz, and M. Fritz, “Not what you’ve signed up for: Compromising real-world LLMintegrated applications with indirect prompt injection,” in Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security (AISec ’23), 2023.

[4] National Institute of Standards and Technology, “CAISI issues request for information about securing AI agent systems,” Jan. 2026, nIST news release, January 12, 2026. [Online]. Available: https://www.nist.gov/news-events/news/2026/01/ caisi-issues-request-information-about-securing-ai-agent-systems

[5] ——, “Challenges to the monitoring of deployed AI systems,” NIST AI 800-4, Mar. 2026. [Online]. Available: https://nvlpubs.nist.gov/nistpubs/ ai/NIST.AI.800-4.pdf

[6] A. A. Cárdenas, S. Amin, and S. Sastry, “Research challenges for the security of control systems.” HotSec, vol. 5, no. 15, p. 1158, 2008.

[7] L. Liu, C. Chen, Q. Pei, S. Maharjan, and Y. Zhang, “Vehicular edge computing and networking: A survey,” Mobile networks and applications, vol. 26, no. 3, pp. 1145–1168, 2021.

[8] J. Fue, J. A. Gutierrez, and Y. Donoso, “Understanding security vulnerabilities in private 5g networks: Insights from a literature review,” Future Internet, vol. 17, no. 11, p. 485, 2025.

[9] O-RAN ALLIANCE, “O-RAN security threat modeling and risk assessment,” O-RAN ALLIANCE, Technical Report O-RAN.WG11.Threat-Model, 2024. [Online]. Available: https://www.o-ran.org/specifications

[10] The Joint Commission, “Medical device alarm safety in hospitals,” Sentinel Event Alert, no. 50, pp. 1–3, Apr. 2013, pMID: 23767076. [Online]. Available: https://www.jointcommission.org/en-us/ knowledge-library/newsletters/sentinel-event-alert/issue-50

[11] S. Sendelbach and M. Funk, “Alarm fatigue: A patient safety concern,” AACN Advanced Critical Care, vol. 24, no. 4, pp. 378–386, 2013.

[12] ETSI, “Zero-touch network and service management (ZSM); closed-loop automation,” European Telecommunications Standards Institute, Tech. Rep. GS ZSM 009-1 v1.1.1, 2021. [Online]. Available: https://www.etsi.org/deliver/etsi\_gs/ZSM/001\_099/00901/01. 01.01\_60/gs\_zsm00901v010101p.pdf

[13] A. Clemm, L. Ciavaglia, L. Z. Granville, and J. Tantsura, “Intent-based networking—concepts and definitions,” RFC 9315, 2022.

[14] D. Soldani, P. Nahi, H. Bour, S. Jafarizadeh, M. F. Soliman, L. Di Giovanna, F. Monaco, G. Ognibene, and F. Risso, “ebpf: A new approach to cloud-native observability, networking and security for current (5g) and future mobile networks (6g and beyond),” IEEE Access, vol. 11, pp. 57 174–57 202, 2023.

[15] I. Ahmad, I. Ahmad, and E. Harjula, “Adaptive security at the edge for 6G-enabled healthcare IoT,” in 2026 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), 2026.

[16] Cilium Authors, “Introduction to cilium and hubble,” 2026, project documentation. [Online]. Available: https://docs.cilium.io/en/stable/ overview/intro/

[17] S. Chatzimiltis, M. B. Mashhadi, M. Shojafar, M. Debbah, and R. Tafazolli, “Agentic AI for 6G: A new paradigm for autonomous ran security compliance,” arXiv preprint arXiv:2512.12400, 2025.

[18] H. Wen, P. Sharma, V. Yegneswaran, A. Gehani, P. Porras, and Z. Lin, “Mobillm: An agentic ai framework for closed-loop threat mitigation in 6g open rans,” in MILCOM 2025 - 2025 IEEE Military Communications Conference (MILCOM), 2025, pp. 1–6.

[19] J. P. Anderson, “Computer security technology planning study,” Electronic Systems Division, Air Force Systems Command, Tech. Rep. ESD-TR-73-51, 1972, the foundational reference monitor concept.

[20] I. Ahmad, F. Shahid, I. Ahmad, J. Islam, K. N. Haque, and E. Harjula, “Adaptive lightweight security for performance efficiency in critical healthcare monitoring,” in 2024 18th International Symposium on Medical Information and Communication Technology (ISMICT), 2024, pp. 78–83.

[21] Y. Zheng, Y. Hu, T. Yu, and A. Quinn, “AgentSight: System-level observability for AI agents using eBPF,” in Proceedings of SOSP, 2025.

[22] D. Jin, A. J. Gaidis, and V. P. Kemerlis, “BeeBox: Hardening BPF against transient execution attacks,” in 33rd USENIX Security Symposium (USENIX Security ’24), 2024.

[23] H. Sun and Z. Su, “Approximation enforced execution of untrusted Linux kernel extensions,” in 34th USENIX Security Symposium (USENIX Security ’25), 2025.

[24] J. Jia, R. Qin, M. Craun, E. Lukiyanov, A. Bansal, M. Phan, and T. Xu, “Rex: Closing the language-verifier gap with safe and usable kernel extensions,” in 2025 USENIX Annual Technical Conference (USENIX ATC ’25), 2025.

[25] K. Alam, M. A. Habibi, M. Tammen, D. Krummacker, W. Saad, M. D. Renzo, T. Melodia, X. Costa-Pérez, M. Debbah, A. Dutta, and H. D. Schotten, “A comprehensive tutorial and survey of o-ran: Exploring slicing-aware architecture, deployment options, use cases, and challenges,” IEEE Communications Surveys & Tutorials, vol. 28, pp. 1637–1678, 2026.

[26] F. B. Schneider, “Enforceable security policies,” ACM Trans. Inf. Syst. Secur., vol. 3, no. 1, p. 30–50, Feb. 2000. [Online]. Available: https://doi.org/10.1145/353323.353382

[27] C. Sánchez, G. Schneider, W. Ahrendt, E. Bartocci, D. Bianculli, C. Colombo, Y. Falcone, A. Francalanza, S. Krstic, J. M. Lourenço´ et al., “A survey of challenges for runtime verification from advanced application domains (beyond software),” Formal Methods in System Design, vol. 54, no. 3, pp. 279–335, 2019.

[28] M. Alshiekh, R. Bloem, R. Ehlers, B. Könighofer, S. Niekum, and U. Topcu, “Safe reinforcement learning via shielding,” in Proceedings of the AAAI conference on artificial intelligence, vol. 32, no. 1, 2018.

[29] S. Gimhana, I. Ahmad, P. Porambage, and E. Harjula, “Mitigating DoS attacks in mMTC: An energy efficiency perspective,” in 2025 Joint European Conference on Networks and Communications & 6G Summit (EuCNC/6G Summit), 2025, pp. 199–204.

[30] University of Oulu, “5GTN: 5g test network,” University of Oulu research infrastructure, 2024. [Online]. Available: https://5gtn.fi

[31] Ollama, “Ollama,” 2026, local large language model serving framework. Accessed 2026-05-16. [Online]. Available: https://github.com/ollama/ ollama