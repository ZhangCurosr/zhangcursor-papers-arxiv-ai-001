# Model-Free Runtime Measurement of Task-Relevant Information

Dimitris Panagopoulos1\*, Adolfo Perrusquía1, Yang Xing1, Weisi Guo1

1\*Faculty of Engineering and Applied Science, Cranfield University, Cranfield, MK43 0AL, UK.

\*Corresponding author(s). E-mail(s): d.panagopoulos@cranfield.ac.uk; Contributing authors: adolfo.perrusquia-guzman@cranfield.ac.uk; yang.x@cranfield.ac.uk; weisi.guo@cranfield.ac.uk;

## Abstract

Autonomous systems increasingly monitor physical kinematics, environmental states and task completion outcomes. While existing work provides methods to plan information seeking actions and interpret acquired observations, current approaches do not monitor whether the live process of information gathering is functioning properly. Consequently, such systems can remain operationally active without resolving outstanding mission requirements. To address this limitation, we introduce Acquisition-State Telemetry (AST), a model free architecture that evaluates ongoing information acquisition from observable evidence relative to declared task requirements. Specifically, a domain interface maps raw observations into task resolution coordinates and normalized comparative representations, which are processed by a fixed causal monitor. This monitor emits two continuous real time signals that quantify (i) prospective progress across unresolved requirements and (ii) informational friction generated when recurrent evidence yields negligible informational gain. We evaluated a single, unadjusted monitor configuration across eight diverse domains including search and rescue dialogue, spatial grid exploration, hierarchical capability acquisition, continuous robotic manipulation, and probabilistic intent estimation. We show that across these domains the friction channel distinguished unproductive stagnation from purposeful exploration prior to task completion, while remaining muted during post completion activity. Furthermore, in closed loop reinforcement learning experiments under early termination constraints, we demonstrate that penalizing informational friction allowed agents to sustain high mission success, whereas unpenalized policies and baselines with matched operational costs failed to complete required tasks.

Keywords: Information acquisition, Runtime monitoring, Telemetry, Autonomous systems, Sequential decision-making, Human-machine interaction

## 1 Introduction

Autonomous cyberphysical systems mostly monitor physical kinematics, environmental surroundings, and binary task completion outcomes [1]. In particular, sensors track physical trajectories and actuators measure mechanical loads, whereas downstream success metrics record whether an objective was achieved. However, the ongoing process of acquiring the information required to achieve those objectives remains unmonitored at runtime. When operating in dynamic or incompletely modeled environments, an autonomous agent can sustain continuous operational activity while making no material progress toward its mission requirements [1]. For example, a search and rescue conversational assistant may repeatedly question witnesses without clarifying the coordinates of a missing person [2–4]. Similarly, a mobile robot may explore a partitioned environment while traversing repetitive loops [5, 6] and a probabilistic estimator may ingest continuous observations without resolving ambiguity among competing hypotheses [7-9]. Because conventional monitors track operational activity rather than information acquisition, systems displaying identical levels of activity can follow fundamentally different epistemic trajectories (Fig. 1).

Current decision-making and diagnostic paradigms do not solve this problem because each addresses a different operational level. Within the observe-orient-decide-act (OODA) framework [10], information acquisition contributes to orientation, where observations acquire meaning relative to operational goals. Methods in active sensing, Bayesian experimental design, and information seeking optimize action selection by predicting future information gain [11-18]. However, these policies assume their forward observation models are dependable and lack mechanisms to verify whether the actual runtime gathering process is yielding progress. Conversely, conversational breakdown detectors and interaction quality estimators assess surface level coherence and retrospective outcomes [8, 19–25], but they rely on domain specific heuristics that cannot generalize across heterogeneous physical or algorithmic systems. Similarly, intrinsic motivation and curiosity driven exploration incorporate novelty bonuses into reinforcement learning [2, 26–32]. Although they reward environmental novelty, they do not verify whether that novelty resolves declared mission objectives. This leaves a critical gap that directly motivates this study. Specifically, we lack a real-time, independent monitor that can continuously track whether incoming evidence is actually satisfying task requirements, completely separate from the policy gathering it. Developing such an independent observer presents a fundamental conceptual challenge. In classical control engineering, separating state estimation from feedback control is achieved through observers such as Kalman and Luenberger filters, which rely on known physical dynamics to infer internal states from sensor measurements [33, 34]. However, modern autonomous systems span vastly different domains and share no common physical laws.

To bridge this divide, we introduce Acquisition-State Telemetry (AST), a model-free monitoring framework that adopts declared task requirements, rather than physical dynamics, as the invariant reference frame. By decoupling domain semantics from monitoring logic, AST models interaction as a trajectory through a taskreferenced coordinate space. A domain-specific interface projects incoming evidence into cumulative resolution scores across declared requirements alongside normalized representations for temporal comparison. A fixed causal monitor then processes this sequence to emit two continuous runtime channels: (i) prospective progress across unresolved requirements and (ii) informational friction, which detects recurrent evidence when ongoing activity fails to advance open objectives.

We evaluated a single, unretuned monitor configuration across eight diverse experimental substrates spanning human dialogue, spatial grid exploration, capability unlocking, continuous robotic manipulation, and internal Bayesian intent estimation [5, 9, 35-39]. The experiments establish three primary empirical findings. First, informational friction reliably distinguishes unproductive stagnation from purposeful information-seeking across both linguistic and physical domains, while remaining muted during post-goal actiity. Second, the measurement remains robust when substituting expert human scoring with automated language model ensembles. Third, in closed loop reinforcement learning under early termination constraints, penalizing informational friction enables agents to complete mission requirements, whereas unpenalized policies and operational cost baselines fail due to persistent looping. Together, these results demonstrate that task-referenced acquisition telemetry provides a domain-agnostic foundation for runtime epistemic supervision in autonomous systems.

![](images/6312d26283ab1c1de3cbe69246462124812f9fe9a9c0ff65d91f3fc579d6c569.jpg)

![](images/6026f3936f16513b7c6ba696c8e1d85ce4b1aaba33ae12b65dd1f379b54c7b7b.jpg)  
Fig. 1 Distinguishing operational activity from information acquisition. a, A conversation, a robot arm, and an inference engine generate observations through distinct mechanisms that share no common model, yet all pursue a structured task. Projecting each piece of evidence against task requirements places diverse applications into a shared coordinate frame running from an empty state to complete task resolution. Each system traces a distinct trajectory through this frame, showing stepwise transitions in dialogue, continuous changes in manipulation, and non-linear paths in probabilistic inference, while the remaining unresolved task requirements supply a common frame for comparing their acquisition trajectories. b, Two systems exhibiting identical activity levels. Conventional monitors tracking operational liveness and binary task completion cannot distinguish between them prior to termination. Within the task frame, the divergence is immediate, as System A advances steadily toward full resolution with low friction, whereas System B repeatedly cycles through familiar representations while task requirements remain unfulfilled, triggering high friction.

## 2 Acquisition-State Telemetry

## 2.1 Task binding and acquisition state

Let $\mathcal { M } = \{ 1 , \dots , N \}$ denote a finite set of declared task requirements and $\mathcal { V }$ the space of observations produced by the monitored process. A task binding $\boldsymbol { B } = ( \mathcal { M } , \mathcal { V } , S , e )$ specifies a resolution map $S : \mathcal { y }  [ 0 , 1 ] ^ { N }$ and a comparative representation map $e : \mathcal { V } \to \mathbb { R } ^ { d }$ . Given evidence $y _ { t } \in \mathcal { V }$ at exchange t, the binding supplies

$$
\begin{array} { l } { \mathbf { s } ( t ) = S ( y _ { t } ) \in [ 0 , 1 ] ^ { N } , } \\ { \mathbf { r } ^ { ( t ) } = e ( y _ { t } ) , \qquad \quad \| \mathbf { r } ^ { ( t ) } \| _ { 2 } = 1 . } \end{array}\tag{1}
$$

The coordinate $s _ { i } ( t )$ measures the resolution of requirement i supported by the current observation and the unit-norm vector $\mathbf { r } ^ { ( t ) }$ provides a geometry for comparing evidence across exchanges. An attribution marker $i _ { t } \in \mathcal { M } \cup \{ \perp \}$ identifies the requirement targeted by the preceding action, with ⊥ denoting an unattributed exchange. Attribution records the target of an attempt and does not constrain which requirements its evidence can resolve.

Assuming that previously verified information remains valid within an episode, the attainedresolution state ${ \pmb v } ( t ) \in [ 0 , 1 ] ^ { \hat { N } }$ evolves as

$$
\begin{array} { r l } & { ~ v _ { i } ( t ) = \operatorname* { m a x } \{ v _ { i } ( t - 1 ) , s _ { i } ( t ) \} , } \\ & { \triangle v _ { i } ( t ) = v _ { i } ( t ) - v _ { i } ( t - 1 ) , \qquad v _ { i } ( 0 ) = 0 . } \end{array}\tag{2}
$$

Thus $\Delta v _ { i } ( t ) \geq 0$ is newly attained resolution and $1 - v _ { i } ( t )$ is the unresolved fraction of requirement i. The signed increment $\delta _ { i } ( t ) = s _ { i } ( t ) - v _ { i } ( t - 1 )$ additionally compares current evidence with the previously attained state and can be negative even though cumulative resolution is non-decreasing. Invalidation of previously verified information ends the episode and requires a reset of the state and evidence histories. The binding and episode conditions are specified in Appendix G.

## 2.2 Causal monitoring channels

Let $\tau _ { 1 : t }$ denote the ordered evidence and acquisition history available through exchange t. The causal monitor $\Phi _ { \Theta } .$ with fixed numerical configuration Θ, maps this history to two telemetry

![](images/001f4788ff0044a90241132e39be36e52f54d332fbeb4747d75175ec83e96136.jpg)  
Fig. 2 The Acquisition-State Telemetry architecture. $\mathbf { a } , \mathrm { A }$ task binding maps observations to task-resolution coordinates and unit-norm comparative representations. b, The frozen causal monitor accumulates resolution and reports prospective progress and gain-gated friction. Friction combines representational recurrence, realized gain and residual unresolved volume. $\mathbf { c } ,$ Downstream policies use telemetry for observation, reward shaping, intervention or termination without changing the monitoring rules.

channels:

$$
z _ { t } = \Phi _ { \Theta } ( \tau _ { 1 : t } ) = \big ( \mathbf { P } ( t ) , F ( t ) \big ) ,\tag{3}
$$

form

where $\mathbf { P } ( t ) \in \mathbb { R } _ { > 0 } ^ { N }$ estimates prospective acquisition value by requirement and $F ( t ) ~ \in ~ [ 0 , 1 ]$ measures acquisition friction.

The reference progress channel $\mathbf { P E } ^ { \mathrm { B } } ( t )$ updates a Beta-Bernoulli posterior from attributed attempts and combines estimated attempt yield with resolution-dependent uncertainty, declared prerequisite dependencies and a term based on the straightness of a gain-weighted evidence path. Its expected-information component vanishes as a requirement approaches full resolution. The separate trajectory component depends on directional consistency and can remain non-zero when that requirement is saturated. The full posterior and trajectory updates are given in Appendix H.

For each requirement i with earlier attributed evidence representations, recurrence $\mu _ { i } ( t ) \in [ 0 , 1 ]$ is the complement of the orthogonal novelty of $\mathbf { r } ^ { ( t ) }$ relative to the historical representation subspace. Let $\boldsymbol { A } _ { t }$ be the set of requirements with such a history. The friction channel has the structural

$$
\begin{array} { l } { { \displaystyle F ( t ) \simeq \sum _ { i \in { \mathcal { A } } _ { t } } \omega _ { i } ( t ) \mu _ { i } ( t ) } } \\ { { \displaystyle \qquad \times D \big ( \varrho ( t ) \big ) \big ( 1 - \upsilon _ { i } ( t ) \big ) , } } \\ { { \displaystyle \omega _ { i } ( t ) = \frac { \mu _ { i } ( t ) } { \sum _ { j \in { \mathcal { A } } _ { t } } \mu _ { j } ( t ) } . } } \end{array}\tag{4}
$$

Here $\omega _ { i } ( t )$ weights requirements by their relative recurrence, D gates recurrence according to realized gain $\varrho ( t )$ , and $1 - v _ { i } ( t )$ weights the remaining unresolved requirement. For a directly attributed exchange, $\varrho ( t ) = \delta _ { i _ { t } } ( t )$ Unattributed exchanges and bindings with inferred attribution use aggregate gain $\varrho ( t ) = \textstyle \sum _ { i } \Delta v _ { i } ( t )$ . The evaluated gate $D ( \varrho ) = \operatorname* { m a x } \{ D _ { \mathrm { m i n } } , 1 - \lambda \varrho \}$ attenuates recurrence under positive gain and amplifies it under negative signed gain. Completed requirements contribute zero through the unresolved-volume factor. The recurrence weights apply when total recurrence is positive and friction is zero without an active history or sufficient recurrence. Appendix H specifies the projection, numerical tolerances, regularization and clipping used in the implemented index.

## 2.3 Frozen reference implementation and evaluation

The evaluated implementation instantiates $\mathbf P ( t )$ as the Bayesian Progress Estimator $\mathbf { P E } ^ { \mathrm { B } } ( t )$ and $F ( t )$ as the orthogonal-subspace Stalling Index $\mathrm { S I } ^ { \perp } ( t )$ . The monitor configuration Θ and its update rules remain fixed across eight substrates.

Each substrate supplies its task-specific resolution and representation maps. The monitor updates causal statistics without parameter training. Downstream policies are trained separately in the closed-loop reinforcement-learning experiment.

## 3 Experiments & Results

## 3.1 Cross-domain evaluation protocol

We evaluated the frozen AST monitor across eight benchmarks spanning human-agent dialogue, discrete spatial exploration, capability unlocking, continuous robotic manipulation, and embedded probabilistic inference (Extended Data Table 1; binding specifications in Appendix A and Appendix Table A.1). The evaluation covers transfer of the frozen configuration Θ between substrate bindings, discrimination of pre-resolution loops and post-completion recurrence, acquisition dynamics under discrete and incremental task structures, and the effect of telemetry on downstream policy learning.

We applied the same diagnostic thresholds across all eight domains. Specifically, exchanges where $\mathrm { S I } ^ { \perp } ( t ) > \theta _ { \mathrm { S I } } = 0 . 2 0$ are classified as stalled, and coordinates reaching $v _ { i } ( t ) \geq v _ { \mathrm { c o m p l e t e } } =$ 0.80 are considered resolved. In environments with controlled loop injections, including Mini-Grid [5], Crafter [38], and MetaWorld [39], we quantified event-level stall discrimination using ROC-AUC, precision, recall, and temporal span coverage against ground-truth loop windows. In continuous session benchmarks, such as SAR, MultiWOZ [35], Map Task [36], AnnoMI [37], and BOIR [9], we evaluated system-level performance through cumulative knowledge acquisition, session stall burden representing the fraction of exchanges exceeding $\theta _ { \mathrm { S I } }$ , and multi-horizon $( k \in$ $\{ 1 , 3 , 5 \} )$ top-1 next-gain prediction accuracy for the progress channel $\bf { \bar { P } E ^ { \mathrm { B } } }$

## 3.2 Controlled inquiry regimes

A simulated SAR witness interview requiring eight missing-person categories provided three controlled acquisition regimes: systematic inquiry resolving the requirements (S1), repetitive agent questioning with uninformative witness replies (S2), and repetitive witness replies that leave requirements unresolved (S3). We also repeated the evaluation with automated evidence scoring.

The frozen monitor separated all three behavioral regimes (Fig. 3 and Appendix Table B.1). Efficient inquiry (S1) resolved all $8 . 0 0 ~ \pm ~ 0 . 0 0$ categories with zero stall burden and near-zero mean friction $( 0 . 0 2 ~ \pm ~ 0 . 0 0 4 )$ . Both recurrent regimes stopped near half-completion and produced elevated friction. Repetitive questioning (S2) incurred a $3 5 . 0 0 \pm 7 . 0 7 \%$ stall burden with a mean friction of $0 . 2 6 \pm \ : 0 . 0 8$ and $4 . 3 3 \ : \pm \ : 0 . 9 4$ categories resolved. Witness non-informativeness (S3) incurred a $3 6 . 6 7 \pm 1 2 . 4 7 \%$ stall burden with a mean friction of $0 . 2 8 \pm 0 . 0 9$ and $4 . 6 7 \pm 1 . 2 5$ categories resolved. The two stalling regimes produce comparable low-yield acquisition traces despite different sources of repetition, indicating that friction responds to evidence relative to requirements rather than the identity of the source.

We substituted the expert resolution map S with the mean of five local language-model scorers (pooled Krippendorff's $\alpha = 0 . 8 0 ;$ Appendix B). Despite yielding lower absolute completion counts, the ordering of friction across regimes remained similar under the frozen monitor. Specifically, friction values scaled from $0 . 0 4 \pm 0 . 0 1$ for S1 (3.33% stall burden), to $0 . 2 0 \pm 0 . 0 5$ for S2 (35.00%) and $0 . 2 4 \pm 0 . 1 1$ for S3 (33.33%; Appendix Table B.1). The scorer substitution preserved the relative friction ranking and changed the absolute telemetry values.

## 3.3 Human-interaction benchmarks

We evaluated three types of natural human interaction: atomic slot-filling (MultiWOZ [35]), incremental collaborative grounding (HCRC Map Task [36]), and reflective clinical counseling (AnnoMI [37]).

## 3.3.1 MultiWOZ: atomic slot filling

In transactional booking dialogues, requirements typically resolve in single-turn discrete steps upon mention. Consequently, 39 of the 40 evaluated MultiWOZ sessions (97.5%) contained no stalled exchanges. Across multi-slot transitions $( n = 5 6 )$ the progress channel $\mathbf { P E } ^ { \mathrm { B } }$ predicted the next resolved booking slot with 51.8% top-1 accuracy, nearly tripling a uniform guessing baseline (17.9%, $p = 9 . 1 \times 1 0 ^ { - 9 } )$ , whereas a recency baseline failed completely (0%) due to the lack of multi-turn incremental accumulation.

![](images/8b99c690c6dcc0bf0b0a7fc5eefba5c72286d6f2eb281810b330891bf200a75d.jpg)  
Fig. 3 AST telemetry across controlled SAR inquiry regimes (child corpus, expert scores). $\mathbf { a } , \mathbf { c } , \mathbf { e } ,$ Final resolution of each coordinate (filled) against full resolution (outer ring) and the completion threshold $v _ { \mathrm { c o m p l e t e } } = 0 . 8 0$ (dashed) for regimes S1, S2 and S3. $\mathbf { b , d , f , }$ Corresponding trajectories through mean resolution, friction $\mathrm { S I } ^ { \perp } ( t )$ and total progress $\begin{array} { r l } { \sum _ { i } \mathrm { P E } _ { i } ^ { \mathrm { B } } ( t ) ; } \end{array}$ each marker is an exchange, red markers exceed $\theta _ { \mathrm { S I } } = 0 . 2 0$ , the diamond marks the first exchange and grey points are projections onto the floor. S1 advances to full resolution with negligible friction, whereas S2 and S3 stall before completion with friction excursions close to 1.

## 3.3.2 HCRC Map Task: incremental grounding

Collaborative route navigation grounds spatial landmarks incrementally across multiple turns. Across 27 Map Task dialogues, $\mathbf { P E } ^ { \mathrm { B } }$ predicted the next grounded landmark with a top-1 accuracy 13.7 percentage points above chance at horizon $k = 5 ~ ( p = 1 . 4 \times 1 0 ^ { - 2 1 } )$ , outperforming the baseline in 21 of 27 sessions (sign test $p \ = \ 0 . 0 0 6 )$ Reflecting the exploratory and clarifying nature of spatial navigation, Map Task incurred a mean stall burden of 8.0%, contrasting with the near-zero baseline friction of atomic slot-filling.

## 3.3.3 AnnoMI: reflective clinical interaction

In 20 motivational interviewing sessions mapped across five client-state dimensions representing the presenting problem, motivation, barriers, resources, and action readiness, AST friction tracked informational throughput (Appendix C and Appendix Table C.1). Sessions with higher stall burden yielded significantly lower information gain per turn (Spearman $\rho = - 0 . 6 6 6 , p =$ $0 . 0 0 1 4 )$ and lower cumulative knowledge resolution $( \rho ~ = ~ - 0 . 5 7 0 , ~ p ~ = ~ 0 . 0 0 8 7 )$ . Notably, stall burdens were similar for sessions rated as highand low-quality counseling by external clinicians (35.20% and 35.77%, respectively; $p ~ = ~ 0 . 9 1 0 )$ This shows the scope of the AST reporting acquisition relative to the five declared client-state dimensions and not clinical quality, which remains a separate outcome as reflective pauses and empathetic pacing are common in counseling and have been discussed as positive friction in conversational artificial intelligence [40].

## 3.4 Robotic and inference benchmarks

We applied the same frozen monitor to four additional (non-linguistic) systems: 2D spatial exploration (MiniGrid [5]), hierarchical capability unlocking (Crafter [38]), continuous kinematic manipulation (MetaWorld [39]), and embedded Bayesian intent estimation (BOIR [9]).

## 3.4.1 MiniGrid: incremental spatial exploration

In a gridworld partitioned into 3×3 spatial sectors where resolution tracks the fraction of revealed walkable tiles, AST detected injected movement loops inside incomplete sectors under pre-resolution stalling, designated as Condition I1. For this condition, performance reached an ROC-AUC of 0.882, precision of 0.742, recall of 0.745, and span coverage of 0.800 (Extended Data Table 2), with mean friction rising from 0.103 during nominal exploration to 0.334 during stalls. Conversely, when identical movement loops occurred inside already explored sectors under post-goal waiting, designated as Condition I2, the unresolved volume gate $\left( 1 - v _ { i } ( t ) \right)$ attenuated the signal, reducing mean friction to 0.179 with an ROC-AUC of 0.715.

## 3.4.2 Crafter: discrete achievement unlocks

Across Crafter's 22-node achievement tree, 95.7% of positive updates occur as discrete milestones. When agents encountered injected no-op loops prior to unlocking prerequisites, mean friction rose from a nominal baseline of 0.133 to 0.509 (ROC-AUC 0.767, span coverage 0.952; Extended Data Table 2). Furthermore, because the resolution map scores each achievement by the fraction of its prerequisites satisfied, $\mathbf { P E } ^ { \mathrm { B } }$ predicted upcoming unlocks with 0.544 top-1 accuracy at horizon $k =$ 5, more than doubling both recency (0.228) and uniform (0.211) baselines (Appendix Table D.1).

## 3.4.3 MetaWorld: continuous robotic manipulation

In continuous pick-and-place manipulation tracked across four kinematic milestones (reach, grasp, lift, and place; Fig. 4), the gain gate differentiated the two controlled recurrence conditions. When the end-effector entered repetitive orbiting loops before securing the object (pre-resolution stall, Condition I1), friction increased from 0.051 to 0.999, with ROC-AUC 0.983, recall 1.000, span coverage 1.000 (Extended Data Table 2).

In contrast, when the arm executed repetitive holding motions at the target location after completing the placement milestone (post-goal station-keeping, Condition I2), saturation of the acquisition state v(t) silenced the friction channel (0.039 event mean versus 0.041 baseline), yielding chance-level discrimination (ROC-AUC 0.499). Across the two manipulation conditions, the gain gate strongly separated pre-grasp stalling from post-goal holding. Concurrently, $\scriptstyle { \dot { \mathbf { P } } } \mathbf { E } ^ { \mathrm { B } }$ achieved a top-1 next-milestone prediction accuracy of 0.985 at $k = 5$ (Appendix Table D.1).

## 3.4.4 BOIR: embedded intent inference

We also applied AST to an internal Bayesian intent estimator inferring human operator goals from continuous trajectories [9]. In scenario S3, the estimator began all 60 epochs with an incorrect maximum a posteriori hypothesis (posterior confidence above 0.70 in 42 epochs). The monitored true-hypothesis resolution rose from 0.142 to 0.966 (Appendix D and Appendix Table D.2). Friction remained low immediately following goal switches (0.014–0.075) and rose mid-epoch (0.051– 0.130) as competing evidence accumulated during hypothesis re-adjudication.

![](images/b0399b291a142634865dca5ccaa7b0eaf9954c67e18fb4e546e520d68585ceb0.jpg)  
Fig. 4 AST telemetry on MetaWorld pick-and-place (seed 0). a,b, A uniform-random policy resolves none of the reach, grasp, lift and place milestones, and friction saturates near 1 from the second exchange onwards. c,d, The scripted controller resolves the four milestones in sequence; friction exceeds $\theta _ { \mathrm { S I } }$ only at the second exchange, before the first milestone is reached, and once briefly (0.22) before placement. Panels and markers as in Fig. 3.

## 3.5 Closed-loop control with epistemic feedback

To evaluate downstream use of acquisition telemetry, we trained five Proximal Policy Optimization (PPO) variants on three SAR corpora (10 random seeds per variant and condition; Appendix E): an uninstrumented baseline (Baseline); an agent paying an operational-cost penalty for repeated actions and exchanges without realized gain, calibrated to match the mean magnitude of the friction penalty (Coupled-Blind); an agent receiving AST telemetry within its observation vector without reward shaping (AST-Obs); an agent observing progress while penalizing AST friction in the reward (AST-Pen); and an agent receiving full telemetry in both observations and reward shaping (Full-AST).

All policies were evaluated under two operational regimes, including a standard horizon where episodes run until an interaction budget expires, designated as Condition A, and a friction-sensitive early-stopping regime that terminates an episode immediately if $\mathrm { S I } ^ { \perp } ( t ) > \theta _ { \mathrm { S I } } = 0 . 2 0$ , designated as Condition B. Condition B represents an interaction setting with an externally enforced limit on low-yield exchanges. Under an unconstrained interaction budget within Condition A, all policies resolved between 7.67 and 8.00 of the eight mission categories (Table 1). However, under early stopping within Condition B, performance diverged (Fig. 5). The unpenalized Baseline frequently entered repetitive questioning loops that triggered early termination, resolving only 2.37 categories, which represents a 7% full-resolution rate and a 26.53% stall burden. The AST-Obs variant, which received telemetry only in observations, resolved 1.73 categories, with a 10% full-resolution rate and a 37.47% stall burden.

The matched operational cost penalty, designated as Coupled-Blind, improved performance over the baseline but resolved only 3.53 categories, which corresponds to a 20% full-resolution rate and a 12.90% stall burden. This penalty applies to repeated actions and exchanges without immediate gain, including productive follow-up queries. The friction-penalized policies recorded higher completion under early stopping. Specifically, AST-Pen resolved 7.07 of eight categories, representing an 87% full-resolution rate, and Full-AST resolved 6.93 categories, representing an 83% full-resolution rate. Both approaches recorded zero stall burden in this evaluation.

## 4 Discussion

Acquisition-State Telemetry demonstrates that the runtime health of an information gathering process can be tracked independently of the operational policy generating the evidence. By grounding the observer in declared mission requirements rather than inventing physical equations of motion, this architecture establishes a common measurement interface across heterogeneous computational and cyberphysical systems. This decoupling mirrors the foundational separation of state estimation and feedback control in classical systems engineering. Domain specific interfaces translate distinct forms of evidence into shared coordinates of cumulative resolution and comparative representation, allowing a single fixed causal monitor to evaluate epistemic progress across linguistic, robotic, and algorithmic environments.

The selective behavior of informational friction clarifies why traditional repetition penalties and loop detectors fall short in complex operational environments. Repetitive activity is not uniformly harmful. For example, an agent may linger to stabilize a physical grasp or rephrase a statement to facilitate mutual understanding. Our gain-gated friction telemtry signal accounts for this context by treating repetition as detrimental only when it coincides with stagnant task resolution. When incoming evidence yields new informational gains or when targeted objectives reach completion the friction signal attenuates. This mechanism explains why the monitor distinguishes unproductive pre goal looping from legitimate post goal activity across both spatial exploration and robotic manipulation benchmarks. In particular, the monitor isolates those exchanges that consume resources without advancing declared requirements instead of penalizing operational activity indiscriminately.

Moreover, these findings show the relationship between epistemic throughput and broader interaction dynamics. In transactional exchanges and discrete capability unlocking, information arrives in decisive milestones, resulting in minimal baseline friction. In contrast, collaborative grounding and reflective counseling naturally produce sustained recurrence as participants iteratively negotiate meaning or build interpersonal trust. The observed negative correlation between friction and information acquisition confirms that the telemetry accurately reflects operational stagnation relative to declared task coordinates. Crucially, however, identical friction levels can occur across interactions rated high and low in interpersonal quality. This distinction emphasizes that informational friction measures epistemic throughput rather than conversational rapport or therapeutic efficacy. Pauses, reflective summaries, and deliberate pacing often serve essential social or cognitive functions. Runtime telemetry provides an objective measurement of task progress, leaving the decision of how to balance throughput against social rapport to domain specific controllers.

In closed loop decision making, the results demonstrate that passive telemetry observation is insufficient to resolve persistent failure modes. Agents that observe progress signals or get penalized for operational repetition continue to cycle especially if they run out of time. Operational penalties inevitably punish necessary exploratory actions along with redundant ones. By contrast, incorporating gain-gated friction into reward shaping provides a targeted epistemic supervisory signal. Policies trained under this feedback learn to abandon unyielding inquiries and redirect operational effort toward alternative unresolved requirements. Under thgis controlled setting, this capability establishes that runtime acquisition monitoring could also be used as an actionable supervisory signal that can guide search, active sensing, and escalation policies in resourceconstrained deployments.

The scope and validity of this monitoring approach depend on specific structural boundaries. The architecture presumes a finite, declared schema whose requirements retain consistent definitions throughout an operational episode. Monotonic accumulation assumes that previously verified information remains valid until an explicit episode reset occurs. Consequently, operational settings involving transient observations, timevarying facts, or active misinformation require extending the state contract beyond monotonic updates. Furthermore, the absolute magnitude of emitted telemetry remains dependent on the resolution mapping and representation spaces provided by the host system. Extending this framework to accommodate dynamically evolving or learnable task schemas, asynchronous multimodal observation streams and automated belief revision represents a critical direction for future investigation. Within these operational boundaries, AST provides a reproducible, model free interface that establishes whether ongoing activity is successfully resolving mission objectives.

Table 1 Closed-loop PPO performance under budget termination (Condition A) and friction-sensitive early stopping (Condition B). Values report means and 95% bootstrap confidence intervals over 10 random seeds, each averaged over the three SAR corpora (one deterministic evaluation episode per policy). Bold indicates top performance under Condition B
<table><tr><td rowspan=1 colspan=11>Categories            Total            Full-res.            Mean              StallCond.Policy variantresolved          knowledge           rate          friction $( \mathrm { S I ^ { \perp } } )$        burden (%)</td></tr><tr><td rowspan=1 colspan=2>Baseline          7.67 [7.00, 8.00]</td><td rowspan=1 colspan=2>0.840 [0.766, 0.878]</td><td rowspan=1 colspan=1>0.93 [0</td><td rowspan=1 colspan=1>.80, 1.00]</td><td rowspan=1 colspan=1>0.009 [</td><td rowspan=1 colspan=1>0.002, 0.022]</td><td rowspan=1 colspan=2>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Coupled-Blind    8.00 [8</td><td rowspan=1 colspan=1>.00, 8.00]</td><td rowspan=1 colspan=1>0.874 [0</td><td rowspan=1 colspan=1>.872, 0.877]</td><td rowspan=1 colspan=1>1.00 [1</td><td rowspan=1 colspan=1>.00, 1.00]</td><td rowspan=1 colspan=1>0.004 [0</td><td rowspan=1 colspan=1>.003, 0.005]</td><td rowspan=1 colspan=1>0.00 [0</td><td rowspan=1 colspan=1>.00, 0.00]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>A   AST-Obs        7.80 [7</td><td rowspan=1 colspan=1>.40, 8.00]</td><td rowspan=1 colspan=1>0.854 [0</td><td rowspan=1 colspan=1>.808, 0.879]</td><td rowspan=1 colspan=1>0.97 [0</td><td rowspan=1 colspan=1>.90, 1.00]</td><td rowspan=1 colspan=1>0.007 [0</td><td rowspan=1 colspan=1>.003, 0.016]</td><td rowspan=1 colspan=1>0.00 [0</td><td rowspan=1 colspan=1>.00, 0.00]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full-AST         8.00 [8</td><td rowspan=1 colspan=1>.00, 8.00]</td><td rowspan=1 colspan=2>0.887 [0.885, 0.890]</td><td rowspan=1 colspan=1>1.00 [1</td><td rowspan=1 colspan=1>.00, 1.00]</td><td rowspan=1 colspan=1>0.002 [0</td><td rowspan=1 colspan=1>.002, 0.002]</td><td rowspan=1 colspan=1>0.00 [0</td><td rowspan=1 colspan=1>.00, 0.00]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=2>AST-Pen         8.00 [8.00, 8.00]</td><td rowspan=1 colspan=4>0.887 [0.884, 0.890]   1.00 [1.00, 1.00]</td><td rowspan=1 colspan=2>0.002 [0.002, 0.002]</td><td rowspan=1 colspan=2>0.00 [0.00, 0.00]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=6>Baseline          2.37 [1.67, 3.10]   0.348 [0.275, 0.426]   0.07 [0.00, 0.17]</td><td rowspan=1 colspan=2>0.119 [0.101, 0.132]</td><td rowspan=1 colspan=3>26.53 [21.34, 31.98]</td></tr><tr><td rowspan=1 colspan=2>Coupled-Blind    3.53 [2.67, 4.37]</td><td rowspan=1 colspan=1>0.515 [0</td><td rowspan=1 colspan=1>.431, 0.596]</td><td rowspan=1 colspan=1>0.20 [0</td><td rowspan=1 colspan=1>.10, 0.30]</td><td rowspan=1 colspan=1>0.065 [0</td><td rowspan=1 colspan=1>.048, 0.084]</td><td rowspan=1 colspan=2>12.90 [8.86, 17.43]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>B   AST-Obs        1.73 [0</td><td rowspan=1 colspan=1>.73, 3.03]</td><td rowspan=1 colspan=1>0.266 [0</td><td rowspan=1 colspan=1>.151, 0.398]</td><td rowspan=1 colspan=1>0.10 [0</td><td rowspan=1 colspan=1>.00, 0.27]</td><td rowspan=1 colspan=1>0.367 [0</td><td rowspan=1 colspan=1>.279, 0.442]</td><td rowspan=1 colspan=2>37.47 [27.58, 45.78]</td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1>Full-AST         6.93 [</td><td rowspan=1 colspan=1>6.23, 7.60]</td><td rowspan=1 colspan=1>0.772 [0</td><td rowspan=1 colspan=1>.698, 0.844]</td><td rowspan=1 colspan=1>0.83 [0</td><td rowspan=1 colspan=1>.73, 0.93]</td><td rowspan=1 colspan=1>0.005 [0</td><td rowspan=1 colspan=1>.002, 0.009]</td><td rowspan=1 colspan=2>0.00 [0.00, 0.00]</td><td rowspan=2 colspan=1></td></tr><tr><td rowspan=1 colspan=2>AST-Pen       7.07 [6.33, 7.77]</td><td rowspan=1 colspan=2>0.787 [0.709, 0.861]</td><td rowspan=1 colspan=2>0.87 [0.77, 0.97]</td><td rowspan=1 colspan=2>0.004 [0.002, 0.007]</td><td rowspan=1 colspan=2>0.00 [0.00, 0.00]</td></tr></table>

![](images/00dd4202201b9fb8c7502f4e0a63ebbbda857b24c19052be8dea237886c3c167.jpg)  
Fig. 5 Closed-loop PPO episodes under friction-triggered early stopping (Condition B; child corpus, seed 0). a,b, The Baseline policy ends the episode at the third exchange, when friction exceeds $\theta _ { \mathrm { S I } } ,$ with no category resolved. c,d, The Full-AST policy resolves all eight categories in eight exchanges without exceeding θsi. Panels and markers as in Fig. 3.

## 5 Conclusion

In this work, we introduced Acquisition-State Telemetry, a task-referenced framework for measuring information acquisition through a fixed causal monitor and task-specific evidence bindings. Across eight heterogeneous substrates, its progress and friction channels characterized acquisition dynamics, while friction-based reward shaping improved requirement completion in the evaluated closed-loop setting. Thus, AST provides a common runtime interface for tracking how ongoing activity advances declared information requirements.

## Code availability

The AST monitor, substrate bindings and experiment scripts, with a single command that reproduces every result, table and figure, are available at https://github.com/dimipan/ ast-telemetry/. MultiWOZ 2.4 [35], the HCRC Map Task corpus [36] and AnnoMI [37] are available from their original releases; the derived session files used here are included.

## References

[1] Ramesh, A., Stolkin, R. & Chiou, M. Robot vitals and robot health: Towards systematically quantifying runtime performance degradation in robots under adverse conditions. IEEE Robotics and Automation Letters 7, 10729–10736 (2022).

[2] Panagopoulos, D., Perrusquia, A. & Guo, W. Selective exploration and information gathering in search and rescue using hierarchical learning guided by natural language input. 2024 IEEE International Conference

on Systems, Man, and Cybernetics, 1175- 1180 (IEEE, 2024).

[3] Gruffeille, C., Perrusquía, A., Tsourdos, A. & Guo, W. Disaster area coverage optimisation using reinforcement learning. IEEE Access (2024).

[4] Panagopoulos, D. et al. Dialogue telemetry: Turn-level instrumentation for autonomous information gathering. arXiv preprint arXiv:2601.09570 (2026).

[5] Chevalier-Boisvert, M. et al. Minigrid & miniworld: Modular & customizable reinforcement learning environments for goaloriented tasks. Advances in Neural Information Processing Systems 36, 73383–73394 (2023).

[6] Chiou, M., Hawes, N. & Stolkin, R. Mixedinitiative variable autonomy for remotely operated mobile robots. ACM Transactions on Human-Robot Interaction 10, 1–34 (2021).

[7] Zhao, Y., Dastani, M., Long, J., Wang, Z. & Wang, S. Rescue conversations from deadends: Efficient exploration for task-oriented dialogue policy optimization. Transactions of the Association for Computational Linguistics 12, 1578–1596 (2024).

[8] Higashinaka, R., Funakoshi, K., Kobayashi, Y. & Inaba, M. The dialogue breakdown detection challenge: Task description, datasets, and evaluation metrics. Proceedings of the Tenth International Conference on Language Resources and Evaluation, 3146- 3150 (2016).

[9] Panagopoulos, D., Petousakis, G., Stolkin, R., Nikolaou, G. & Chiou, M. A bayesianbased approach to human operator intent recognition in remote mobile robot navigation. 2021 IEEE International Conference on Systems, Man, and Cybernetics, 125–131 (2021).

[10] Boyd, J. R. et al. A discourse on winning and losing Vol. 393 (Air University Press Maxwell Air Force Base, AL, 2018).

[11] Lindley, D. V. On a measure of the information provided by an experiment. The Annals of Mathematical Statistics 27, 986- 1005 (1956).

[12] Chaloner, K. & Verdinelli, I. Bayesian experimental design: A review. Statistical Science 273–304 (1995).

[13] Tellex, S., Thaker, P., Deits, R., Kollar, T. & Roy, N. Toward information theoretic human-robot dialog. Robotics: Science and Systems, Vol. 2, 3 (2012).

[14] Deits, R. et al. Clarifying commands with information-theoretic human-robot dialog. Journal of Human-Robot Interaction 2, 58–79 (2013).

[15] Rao, S. & Daumé III, H. Learning to ask good questions: Ranking clarification questions using neural expected value of perfect information. arXiv preprint arXiv:1805.04655 (2018).

[16] Aliannejadi, M., Zamani, H., Crestani, F. & Croft, W. B. Asking clarifying questions in open-domain information-seeking conversations 475–484 (2019).

[17] Lee, S.-W., Heo, Y.-J. & Zhang, B.-T. Answerer in questioner's mind: Information theoretic approach to goal-oriented visual dialog. Advances in Neural Information Processing Systems 31 (2018).

[18] Shukla, P. et al. What should I ask? using conversationally informative rewards for goal-oriented visual dialog. Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics, 6442- 6451 (2019).

[19] Deriu, J. et al. Survey on evaluation methods for dialogue systems. Artificial Intelligence Review 54, 755–810 (2021).

[20] Schmitt, A. & Ultes, S. Interaction quality: Assessing the quality of ongoing spoken dialog interaction by experts—and how it relates to user satisfaction. Speech Communication 74, 34–46 (2015).

[21] Ultes, S. et al. Domain-independent user satisfaction reward estimation for dialogue policy learning. Interspeech, 1721–1725 (2017).

[22] Su, P.-H., Gašić, M. & Young, S. Reward estimation for dialogue policy optimisation. Computer Speech & Language 47, 304–326 (2018).

[23] Ng, N., Ghassemi, M., Thangarajan, N., Pan, J. & Guo, Q. Improving dialogue breakdown detection with semi-supervised learning. arXiv preprint arXiv:2011.00136 (2020).

[24] Terragni, S. et al. Betold: A task-oriented dialog dataset for breakdown detection. Proceedings of the Second Workshop on When Creative AI Meets Conversational AI, 23–34 (2022).

[25] Chen, M., Wang, J., Liu, Z., Wang, Y. & Wang, Q. From failed trajectories to reliable llm agents: Diagnosing and repairing harness flaws. arXiv preprint arXiv:2606.06324 (2026).

[26] Oudeyer, P.-Y., Kaplan, F. & Hafner, V. V. Intrinsic motivation systems for autonomous mental development. IEEE Transactions on Evolutionary Computation 11, 265-286 (2007).

[27] Bellemare, M. et al. Unifying countbased exploration and intrinsic motivation. Advances in Neural Information Processing Systems, Vol. 29 (Curran Associates, Inc., 2016).

[28] Pathak, D., Agrawal, P., Efros, A. A. & Darrell, T. Curiosity-driven exploration by self-supervised prediction. International Conference on Machine Learning, 2778–2787 (PMLR, 2017).

[29] Houthooft, R. et al. Vime: Variational information maximizing exploration. Advances in Neural Information Processing Systems 29 (2016).

[30] Graves, A., Bellemare, M. G., Menick, J., Munos, R. & Kavukcuoglu, K. Automated curriculum learning for neural networks.

International Conference on Machine Learning, 1311–1320 (PMLR, 2017).

[31] Standley, T. et al. Which tasks should be learned together in multi-task learning? Proceedings of the 37th International Conference on Machine Learning, Vol. 119, 9120–9132 (PMLR, 2020).

[32] Gao, J. et al. Smartrag: Jointly learn ragrelated tasks from the environment feedback. International Conference on Learning Representations, 53117–53140 (2025).

[33] Kalman, R. E. A new approach to linear filtering and prediction problems. Journal of Basic Engineering 82, 35–45 (1960).

[34] Luenberger, D. G. Observing the state of a linear system. IEEE Transactions on Military Electronics 8, 74–80 (1964).

[35] Ye, F., Manotumruksa, J. & Yilmaz, E. Multiwoz 2.4: A multi-domain task-oriented dialogue dataset with essential annotation corrections to improve state tracking evaluation. Proceedings of the 23rd Annual Meeting of the Special Interest Group on Discourse and Dialogue, 351–360 (2022).

[36] Anderson, A. H. et al. The hcrc map task corpus. Language and Speech 34, 351–366 (1991).

[37] Wu, Z. et al. Anno-mi: A dataset of expertannotated counselling dialogues. ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and Signal Processing, 6177–6181 (IEEE, 2022).

[38] Hafner, D. Benchmarking the spectrum of agent capabilities. arXiv preprint arXiv:2109.06780 (2021).

[39] McLean, R. et al. Meta-world+: An improved, standardized, RL benchmark. Thirty-ninth Annual Conference on Neural Information Processing Systems Datasets and Benchmarks Track (2025).

[40] Inan, M. et al. Better slow than sorry: Introducing positive friction for reliable dialogue

systems. arXiv preprint arXiv:2501.17348 (2025).

Table 1 Evaluation map across eight experimental substrates. A single frozen monitor configuration Φ is evaluated across linguistic, symbolic, continuous robotic, and algorithmic bindings.
<table><tr><td>Study</td><td>Substrate</td><td>Acquisition structure</td><td>Primary evidence</td><td>Focus</td></tr><tr><td>1 — Controlled anchor</td><td>SAR</td><td>Scripted efficient and recurrent questioning regimes</td><td>Monitor sensitivity to controlled behavioral regimes; invariance to automated scorer substitution</td><td>Cross-domain reuse</td></tr><tr><td>2 — Human interaction</td><td>MultiWOZ [35]</td><td>Predominantly atomic slot filling</td><td>Next-slot progress prediction under minimal baseline friction</td><td>Reuse; task structure</td></tr><tr><td></td><td>Map Task [36]</td><td>Incremental collaborative grounding</td><td>Multi-horizon landmark pre- diction and sustained friction during iterative clarification</td><td>Reuse; task structure</td></tr><tr><td></td><td>AnnoMI [37]</td><td>Reflective motivational inter- viewing</td><td>Friction correlation with epistemic throughput and inde- pendence from clinical rapport ratings</td><td>Reuse; task structure</td></tr><tr><td>3 — Non- linguistic</td><td>MiniGrid [5]</td><td>Incremental 2D spatial explo- ration</td><td>Discrimination of trajectory loops in incomplete sectors ver-</td><td>Reuse; friction selectivity</td></tr><tr><td></td><td>Crafter [38]</td><td>Discrete achievement unlock tree</td><td>sus post-goal loitering Detection of injected no-op loops and prerequisite-guided unlock</td><td>Reuse; task structure</td></tr><tr><td></td><td>MetaWorld [39]</td><td>Continuous robotic manipulation</td><td>forecasting Discrimination of pre-grasp orbiting stalls versus post- completion goal holding</td><td>Reuse; friction selectivity</td></tr><tr><td></td><td>BOIR [9]</td><td>Embedded Bayesian intent esti- mation</td><td>Runtime tracking of internal probabilistic hypothesis adjudi- cation</td><td>Cross-domain reuse</td></tr><tr><td>4 — Control</td><td>SAR (PPO)</td><td>Budget-constrained and early- stopped inquiry</td><td>Ablation of passive telemetry observation against gain-gated friction and operational-cost reward penalties</td><td>Measurement— control separation</td></tr></table>

Table 2 Study 3: Stalling Index (SI–) event discrimination across non-linguistic bindings $( \theta _ { \mathrm { S I } } = 0 . 2 0 )$ Condition I1 evaluates pre-resolution stalling (repetitive loops while task volume remains unresolved). Condition I2 evaluates post-goal loitering (repetition after task completion). ${ \mathrm { S I } } _ { \mathrm { e v e n t } }$ and $\mathrm { S I } _ { \mathrm { c l e a n } }$ report mean friction inside injected loops and nominal execution windows, respectively
<table><tr><td>Condition</td><td>AUC</td><td>Precision @θ</td><td>Recall @θ</td><td>Span coverage</td><td> $\operatorname { S I } _ { \mathbf { e v e n t } }$ </td><td> $\mathrm { S I } _ { \mathbf { c l e a n } }$ </td></tr><tr><td>MiniGrid I1 (residual stall)</td><td>0.882</td><td>0.742</td><td>0.745</td><td>0.800</td><td>0.334</td><td>0.103</td></tr><tr><td>MiniGrid I2 (post-goal loiter)</td><td>0.715</td><td>0.455</td><td>0.467</td><td>0.500</td><td>0.179</td><td>0.106</td></tr><tr><td>Crafter (injected no-op loop)</td><td>0.767</td><td>0.853</td><td>0.698</td><td>0.952</td><td>0.509</td><td>0.133</td></tr><tr><td>MetaWorld I1 (pre-grasp orbit)</td><td>0.983</td><td>0.800</td><td>1.000</td><td>1.000</td><td>0.999</td><td>0.051</td></tr><tr><td>MetaWorld I2 (post-goal holding)</td><td>0.499</td><td>0.213</td><td>0.036</td><td>0.050</td><td>0.039</td><td>0.041</td></tr></table>

## A Substrate Binding Instantiations and Domain Schemas

Each evaluated substrate implements the task binding $\boldsymbol { B } = ( \mathcal { M } , \boldsymbol { y } , S , e )$ defined in Section 2.1. The set M specifies the declared task requirements and  specifies the observation space. The map $S ( y _ { t } )$ yields task-resolution coordinates and $e ( y _ { t } )$ yields a unit-norm representation for temporal comparison.

Appendix Table A.1 specifies the eight substrate bindings. The numerical configuration Θ and causal update rules defined in Section 2.3 remain fixed across substrates. All linguistic bindings use qwen3-embedding representations with 4,096 dimensions, served locally through Ollama. For BOIR, the estimator supplies the maximum a posteriori hypothesis for each five-step window as its attribution $i _ { t } .$ The BOIR binding therefore applies the aggregate gain gate specified in Appendix H.3.

## B Telemetry Robustness Under Automated Scorer Substitution

The controlled SAR inquiry experiment in Section 3.2 was repeated using automated evidence scores from five locally served language models. The ensemble comprised Gemma 3 12B, Ministral 3 14B, Gemma 2, Qwen 3 and Llama 3.1. Ollama served the models at temperature zero. Each model scored each response independently with the same task-resolution rubric. Krippendorff α for interval scores across turn-coordinate units was 0.827 for the hiker corpus, 0.716 for the elderly corpus and 0.853 for the child corpus. Pooled agreement was 0.801 across the five models and 0.757 after including the expert as a sixth annotator.

Automated scoring assigned lower absolute completion estimates in all three inquiry regimes (Appendix Table B.1). Efficient inquiry (S1) produced mean friction of $0 . 0 4 \pm 0 . 0 1$ and a stall burden of 3.33%. Repetitive questioning (S2) produced mean friction of $0 . 2 0 \pm 0 . 0 5$ and a stall burden of 35.00%. Repetitive witness replies (S3) produced mean friction of 0.24 ± 0.11 and a stall burden of 33.33%. The friction ordering across regimes matched that obtained with expert scores. The monitor configuration remained fixed during scorer substitution.

## C Additional Behavioral Analyses in Human Interaction

The AnnoMI evaluation in Section 3.3.3 maps client statements from 20 motivational interviewing sessions onto five declared client-state dimensions. These dimensions represent the presenting problem, motivation, barriers, resources and action readiness. Appendix Table C.1 reports Spearman correlations between friction and acquisition throughput. Greater friction was associated with lower information gain per turn and lower cumulative task resolution within these declared dimensions.

## D Extended Evaluation on Non-Linguistic and Estimator Substrates

## D.1 Progress prediction across forecast horizons

Appendix Table D.1 reports top-1 next-gain prediction accuracy for the Bayesian Progress Estimator $\mathbf { P E } ^ { \mathrm { B } }$ at horizons $k \in \{ 1 , 3 , 5 \}$ . The comparison methods are uniform guessing and recency-based prediction.

On clean Crafter and MetaWorld trajectories, the progress channel achieved top-1 accuracies of 0.544 and 0.985, respectively, at $k = 5$ . Injected loops reduced accuracy in both substrates, with horizonspecific changes in the strongest baseline. In MiniGrid, recency-based prediction exceeded $\mathbf { P E } ^ { \mathrm { B } }$ at all three horizons, consistent with locally coupled spatial transitions.

## D.2 Internal monitoring of the BOIR intent estimator

The BOIR evaluation in Section 3.4.4 binds AST to an internal Bayesian estimator of human goals. Appendix Table D.2 reports telemetry from 100 evaluation epochs, comprising 40 epochs from 20 twoepoch S1 trials and 60 epochs from 20 three-epoch S3 trials. Friction rose within epochs during hypothesis re-adjudication and remained low immediately after goal changes. Each goal change resets the acquisition state and evidence history.

## E Closed-Loop Reinforcement Learning Hyperparameters and Ablations

The closed-loop experiment in Section 3.5 trains five PPO policy variants with 10 random seeds for each of the three SAR corpora and two termination conditions. The corpus consists of child, elderly and hiker scenarios. This design yields 300 trained policies.

## E.1 Environment and observations

Each episode contains at most 50 interview exchanges. An action selects a questioning strategy and a target requirement or selects a general question. The environment returns the corpus response, expert resolution scores and evidence embedding. The frozen monitor then updates its causal state. Each exchange incurs a cost of 0.1. An episode ends with a bonus of 10 once every coordinate satisfies $v _ { i } \geq v _ { \mathrm { c o m p l e t e } } .$ All policy observations include ${ \pmb v } ( t )$ and the elapsed fraction of the interaction budget. AST-Pen additionally receives $\mathbf { P E } ^ { \mathrm { B } } ( t )$ . AST-Obs and Full-AST receive both $\mathbf { P E } ^ { \mathrm { B } } ( t )$ and $\mathrm { S I } ^ { \perp } ( t )$ . AST-Pen and Full-AST receive an exchange-level reward penalty equal to $\mathrm { S I } ^ { \perp } ( t )$ . Condition B terminates the episode at the first exchange satisfying $\mathrm { S I } ^ { \perp } ( t ) > \theta _ { \mathrm { S I } }$

## E.2 Operational-cost comparator

The Coupled-Blind comparator receives no telemetry and incurs an operational cost $\rho c ( t )$ . The cost $c ( t )$ adds two units for repeating an action already taken during the episode and two units when aggregate acquired resolution is below $1 0 ^ { - 3 }$ . A uniform random reference policy was evaluated over 200 Condition A episodes per corpus to calibrate $\rho$ before training. Mean friction was 0.314 and mean operational cost was 2.19 per exchange, yielding $\rho = 0 . 1 4 3$ . This calibration matches the mean penalty magnitude under the reference policy.

## E.3 Policy training and evaluation

All agents use a Stable-Baselines3 actor-critic multilayer perceptron with two hidden layers of 64 units and tanh activations. Policies are trained for 50,000 environment steps with learning rate $3 \times 1 0 ^ { - 4 }$ , rollout length 256, minibatch size 64, 10 epochs per update, discount factor $\gamma = 0 . 9 9$ , generalized advantage estimation parameter $\lambda _ { \mathrm { G A E } } = 0 . 9 5$ , clip ratio 0.2 and entropy coefficient 0.02. Training uses fixed seeds and one CPU thread. Greedy policy evaluation in the deterministic environment uses one episode per trained policy. Reported confidence intervals are bootstrapped over seed means using 10,000 resamples.

## F Architectural Notation and Contract Glossary

Appendix Table F.1 defines the symbols and data types used in the task binding, acquisition state and frozen monitor described in Section 2.

Table F.1: Core notation and interface definitions for Acquisition-State Telemetry.
<table><tr><td>Symbol</td><td>Type / Domain</td><td>Operational Definition</td></tr><tr><td> $\boldsymbol { B }$ </td><td>Interface Tuple</td><td>Task binding  $\boldsymbol { B } = ( \mathcal { M } , \boldsymbol { y } , S , e )$ </td></tr><tr><td> $\mathcal { M }$ </td><td>Finite Set</td><td>Schema coordinates  $\{ 1 , \ldots , N \}$  representing task dimensions.</td></tr><tr><td> $\mathcal { V }$ </td><td>Observation Space</td><td>Evidence space emitted by the monitored process.</td></tr><tr><td> $y _ { t }$ </td><td>Element ∈ </td><td>Evidence observation received at exchange t.</td></tr><tr><td> $i _ { t }$ </td><td> $M \cup \{ \bot \}$ </td><td>Requirement targeted by the preceding action. The marker ⊥ denotes an unattributed exchange.</td></tr><tr><td> $S$ </td><td>Function  $\mathcal { V } \to [ 0 , 1 ] ^ { N }$ </td><td>Task-resolution map projecting evidence onto schema resolution.</td></tr><tr><td> ${ \bf s } ( t )$ </td><td>Vector  $\in [ 0 , 1 ] ^ { N }$ </td><td>Instantaneous resolution observation  $\begin{array} { r l } { S ( y _ { t } ) } & { { } = } \end{array}$   $( s _ { 1 } ( t ) , \ldots , s _ { N } ( t ) )$ </td></tr><tr><td>e</td><td>Function  $\mathcal { V } \to \mathbb { R } ^ { d }$ </td><td>Evidence representation map with unit-norm output  $( \| e ( y ) \| _ { 2 } = 1 )$ </td></tr><tr><td> $\mathbf { r } ^ { ( t ) }$ </td><td>Vector  $\in \mathbb { R } ^ { d }$ </td><td>Unit-norm embedding of current evidence for subspace</td></tr><tr><td> $v _ { i } ( t )$ </td><td>Scalar ∈ [0, 1]</td><td>comparison. Cumulative attained resolution of coordinate i</td></tr><tr><td> $\Delta v _ { i } ( t )$ </td><td> $\mathrm { S c a l a r } \geq 0$ </td><td> $( \operatorname* { m a x } _ { \ell \leq t } s _ { i } ( \ell ) )$  Marginal resolution gain acquired at exchange</td></tr><tr><td> $\tau _ { 1 : t }$ </td><td>Ordered Sequence</td><td>Causal acquisition history available through exchange t.</td></tr><tr><td> $\Phi _ { \Theta }$ </td><td>Functional</td><td>Frozen causal monitor mapping history  $\tau _ { 1 : t }$  to telemetry  $z _ { t } .$ </td></tr><tr><td>Θ</td><td>Parameter Tuple</td><td>Frozen internal monitor configuration and numerical thresholds.</td></tr><tr><td> $\mathbf P ( t )$ </td><td> $\mathrm { V e c t o r } \in \mathbb { R } _ { \geq 0 } ^ { N }$ </td><td>Prospective acquisition value for each declared require- ment.</td></tr><tr><td> $F ( t )$ </td><td> $\mathrm { S c a l a r } \in [ 0 , 1 ]$ </td><td>Friction channel reporting gain-gated representational recurrence.</td></tr><tr><td> $z _ { t }$ </td><td>Tuple  $( \mathbf { P } ( t ) , F ( t ) )$ </td><td>Live telemetry tuple emitted at exchange t.</td></tr><tr><td> $\mathbf { P E } ^ { \mathrm { B } } ( t )$ </td><td>Estimator Vector</td><td>Reference Bayesian Progress Estimator instantiation.</td></tr><tr><td> $\mathrm { S I } ^ { \perp } ( t )$ </td><td>Index  $\in [ 0 , 1 ]$ </td><td>Reference gain-gated Stalling Index via orthogonal projection.</td></tr><tr><td>π</td><td>Policy Function</td><td>Downstream controller conditioning actions on</td></tr></table>

Table A.1 Substrate-specific AST bindings, evidence representations, and telemetry channel interpretations across all eight experimental domains. The task schema, evidence space, task-resolution map, and evidence representation vary across substrates, while the monitor configuration remains frozen.
<table><tr><td rowspan=1 colspan=14>Substrate  M              y               S(yt)           e(yt)           Progress      Frictioninterpreta-    interpreta-tion            tion</td></tr><tr><td rowspan=24 colspan=14>SAR        Task-critical    Witness/source Expert com-    Response       Residual        Repeated ormissing-person responses to    pleteness per   embedding      acquisitionfacts            queries          fact; five-model LLMensemble in                       factsthe robustnesscheckAnnoMI    Projected       Client state-    Manual rubric  Client           Residualclient-state      ments within   annotation of  utterancedimensions      MI sessions     client disclo-                                         evidence asso-dimensionsMultiWOZ Transactional   User utter-      Belief-statebooking slots  ances followingslot complete-system                                               value acrossprompts                                                                remain unre-Map Task Uniquesession-specificlandmarks                                           embeddinglower queriesremains unre-</td></tr><tr><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>degraded wit-</td></tr><tr><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>ness evidence</td></tr><tr><td rowspan=1 colspan=2>mission-critical</td><td rowspan=1 colspan=1>associated</td></tr><tr><td rowspan=1 colspan=1>with unre-</td></tr><tr><td rowspan=1 colspan=1>solved facts</td></tr><tr><td rowspan=1 colspan=1>Repeated low-</td></tr><tr><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>gain client</td></tr><tr><td rowspan=1 colspan=1>embedding</td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>evidence asso-</td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>projected</td><td rowspan=1 colspan=1>ciated with</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>client-state</td><td rowspan=1 colspan=1>unresolved</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>dimensions</td><td rowspan=1 colspan=1>dimensions</td></tr><tr><td rowspan=1 colspan=1>User-turn text</td><td rowspan=1 colspan=2>Residual</td><td rowspan=1 colspan=1>Repeated con-</td></tr><tr><td rowspan=1 colspan=1>embedding</td><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>straints while</td></tr><tr><td rowspan=1 colspan=2>ness changes</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>required slots</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>required book-</td><td rowspan=1 colspan=1>remain unre-</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>ing slots</td><td rowspan=1 colspan=1>solved</td></tr><tr><td rowspan=3 colspan=3></td><td rowspan=1 colspan=2>Manual rubric</td><td rowspan=1 colspan=1>Giver</td><td rowspan=1 colspan=2>Residual</td><td rowspan=1 colspan=1>Repeated</td></tr><tr><td rowspan=1 colspan=2>annotation</td><td rowspan=1 colspan=1>utterance</td><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>spatial descrip</td></tr><tr><td rowspan=1 colspan=2>of landmark</td><td rowspan=1 colspan=1>embedding</td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>tions while</td></tr><tr><td rowspan=1 colspan=2>grounding</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>landmark</td><td rowspan=1 colspan=1>landmark</td></tr><tr><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>coordinates</td><td rowspan=1 colspan=1>grounding</td></tr><tr><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>remains unre-</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>solved</td></tr><tr><td rowspan=3 colspan=3>MiniGrid</td><td rowspan=3 colspan=4>3 × 3</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>K-step spatial</td><td rowspan=1 colspan=2>Fraction of</td><td rowspan=1 colspan=1>Unit-norm vis- R</td><td rowspan=1 colspan=2>esidual</td><td rowspan=1 colspan=1>Recurrent</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=3></td><td rowspan=1 colspan=2>grid-sector</td><td rowspan=2 colspan=1>partition</td><td rowspan=2 colspan=2>visitation win-dows</td><td rowspan=1 colspan=1>walkable tiles</td><td rowspan=1 colspan=2>itation and</td><td rowspan=1 colspan=1>acquisition</td></tr><tr><td rowspan=1 colspan=3></td><td></td><td></td><td></td><td></td><td rowspan=1 colspan=2>revealed</td><td rowspan=1 colspan=1>action his-</td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>tion associated</td></tr><tr><td rowspan=4 colspan=3>Crafter</td><td rowspan=4 colspan=4>22-element</td><td rowspan=3 colspan=1></td><td rowspan=3 colspan=2></td><td rowspan=1 colspan=1>togram</td><td rowspan=1 colspan=2>spatial sectors</td><td rowspan=1 colspan=1>with incom-</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>plete spatial</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>coverage</td></tr><tr><td rowspan=1 colspan=1>Action, inven-</td><td rowspan=1 colspan=2>Fraction of</td><td rowspan=1 colspan=1>Unit-norm</td><td rowspan=1 colspan=2>Residual</td><td rowspan=1 colspan=1>Low nomi-</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=4>capability</td><td rowspan=1 colspan=1>tory, and</td><td rowspan=1 colspan=2>unlock prereq-</td><td rowspan=1 colspan=1>action-</td><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>nal friction</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=2 colspan=4>achievementtree</td><td rowspan=2 colspan=1>achievementlogs</td><td rowspan=1 colspan=2>uisites satisfied</td><td rowspan=1 colspan=1>inventory-</td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>under predom-</td></tr><tr><td rowspan=7 colspan=3></td><td rowspan=1 colspan=2>tree</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>milestone</td><td rowspan=1 colspan=2>achievement</td><td rowspan=1 colspan=1>inantly atomic</td></tr><tr><td rowspan=6 colspan=4></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>signature</td><td rowspan=1 colspan=2>prerequisites</td><td rowspan=1 colspan=1>updates;</td></tr><tr><td rowspan=5 colspan=1></td><td rowspan=2 colspan=2></td><td rowspan=2 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>sustained pre-</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>resolution</td></tr><tr><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>recurrence</td></tr><tr><td rowspan=2 colspan=2></td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>remains</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>detectable</td></tr><tr><td rowspan=3 colspan=3>MetaWorld</td><td rowspan=3 colspan=4>Manipulation</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>K-step contin-</td><td rowspan=1 colspan=2>Deterministic</td><td rowspan=1 colspan=1>Unit-norm R13</td><td rowspan=1 colspan=2>Residual</td><td rowspan=1 colspan=1>Recurrent</td></tr><tr><td rowspan=1 colspan=3></td><td rowspan=1 colspan=3>milestones</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>uous kinematic</td><td rowspan=1 colspan=2>completion</td><td rowspan=1 colspan=1>hand, object,</td><td rowspan=1 colspan=2>acquisition</td><td rowspan=1 colspan=1>kinematic</td></tr><tr><td rowspan=6 colspan=3>BOIR</td><td rowspan=1 colspan=3>comprising</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>traces</td><td rowspan=1 colspan=2>from simulator</td><td rowspan=1 colspan=1>gripper, and</td><td rowspan=1 colspan=2>value across</td><td rowspan=1 colspan=1>evidence asso-</td></tr><tr><td rowspan=1 colspan=4>reach, grasp,</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2>states</td><td rowspan=1 colspan=1>displacement</td><td rowspan=1 colspan=2>manipulation</td><td rowspan=1 colspan=1>ciated with</td></tr><tr><td rowspan=4 colspan=5>lift and placeBOIR goal     Short windows P</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>summary</td><td rowspan=1 colspan=2>milestones</td><td rowspan=1 colspan=1>unresolved</td></tr><tr><td rowspan=3 colspan=2>osterior pi</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>manipulation</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>coordinates</td></tr><tr><td rowspan=1 colspan=1>Unit-norm</td><td rowspan=1 colspan=2>Residual adju-</td><td rowspan=1 colspan=1>Recurrent</td></tr><tr><td rowspan=1 colspan=3>estimator</td><td rowspan=1 colspan=5>hypotheses      of posterior</td><td rowspan=1 colspan=2>of the true</td><td rowspan=1 colspan=1>posterior-</td><td rowspan=1 colspan=2>dication value</td><td rowspan=1 colspan=1>estimator</td></tr><tr><td rowspan=8 colspan=14>signature (pos-at the end of   and shift,each window   entropy level                      adjudicationstep, MAP18      ing change andreversals)</td></tr><tr><td rowspan=1 colspan=1>signature (pos-</td><td rowspan=1 colspan=2>hypotheses</td><td rowspan=1 colspan=1>ciated with</td></tr><tr><td rowspan=1 colspan=1>terior mean</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>unresolved</td></tr><tr><td rowspan=1 colspan=1>and shift,</td><td rowspan=1 colspan=2></td><td rowspan=1 colspan=1>hypothesis</td></tr><tr><td rowspan=1 colspan=1>entropy level</td><td rowspan=1 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=2 colspan=2></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>step, MAP</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>switches, head-</td><td rowspan=1 colspan=2></td></tr></table>

Table B.1 AST telemetry under expert annotations and automated language-model ensemble scoring in controlled SAR inquiry. S1 denotes efficient sequential acquisition. S2 denotes repetitive agent questioning. S3 denotes repetitive witness responses. Values are means and standard deviations across three mission scenarios.
<table><tr><td rowspan="2">Regime</td><td colspan="3">Human Expert Scorer</td><td colspan="3">Automated LLM Ensemble Scorer</td></tr><tr><td>Stall burden (%)</td><td>Mean SI+</td><td>Categories resolved</td><td>Stall burden (%)</td><td>Mean SI¹</td><td>Categories resolved</td></tr><tr><td>S1</td><td> $\mathbf { 0 . 0 0 \pm 0 . 0 0 }$ </td><td> $\mathbf { 0 . 0 2 \pm 0 . 0 0 4 }$ </td><td> $\mathbf { 8 . 0 0 \pm 0 . 0 0 }$ </td><td> ${ \bf 3 . 3 3 \pm 4 . 7 1 }$ </td><td> $\mathbf { 0 . 0 4 \pm 0 . 0 1 }$ </td><td> ${ \bf 4 . 6 7 \pm 1 . 2 5 }$ </td></tr><tr><td>S2</td><td> $3 5 . 0 0 \pm 7 . 0 7$ </td><td> $0 . 2 6 \pm 0 . 0 8$ </td><td> $4 . 3 3 \pm 0 . 9 4$ </td><td> $3 5 . 0 0 \pm 4 . 0 8$ </td><td> $0 . 2 0 \pm 0 . 0 5$ </td><td> $2 . 0 0 \pm 0 . 8 2$ </td></tr><tr><td>S3</td><td> $3 6 . 6 7 \pm 1 2 . 4 7$ </td><td> $0 . 2 8 \pm 0 . 0 9$ </td><td> $4 . 6 7 \pm 1 . 2 5$ </td><td> $3 3 . 3 3 \pm 1 4 . 3 4$ </td><td> $0 . 2 4 \pm 0 . 1 1$ </td><td> $3 . 0 0 \pm 0 . 8 2$ </td></tr></table>

Table C.1 Spearman rank correlations between AST friction measures and information acquisition in the 20 AnnoMI motivational interviewing sessions.
<table><tr><td>Diagnostic metric pair</td><td>Spearman ρ</td><td>p-value</td></tr><tr><td>Stall burden vs. information gain / turn</td><td>-0.666</td><td>0.0014</td></tr><tr><td>Mean friction  $( \mathrm { S I } ^ { \perp } )$  vs. information gain / turn</td><td>-0.666</td><td>0.0013</td></tr><tr><td>Stall burden vs. total knowledge resolved</td><td>-0.570</td><td>0.0087</td></tr><tr><td>Mean friction  $( \mathrm { S I } ^ { \perp } )$  vs. total knowledge resolved</td><td>-0.646</td><td>0.0021</td></tr></table>

Table D.1 Top-1 next-gain prediction accuracy of the Bayesian Progress Estimator $\mathrm { P E ^ { B } }$ and two baselines on nonlinguistic substrates at horizons k ∈ {1, 3, 5}. Bold values identify the highest accuracy within each substrate and horizon
<table><tr><td>Substrate</td><td colspan="3"> $\mathbf { P E } ^ { \mathrm { B } } \ \left( \mathbf { O u r s } \right)$ </td><td colspan="3">Recency Baseline</td><td colspan="3">Uniform Guessing</td></tr><tr><td></td><td>k=1</td><td>k=3</td><td>k=5</td><td> $k { = } 1$ </td><td> $k { = } 3$ </td><td> $k { = } 5$ </td><td>k=1</td><td>k=3</td><td> $k { = } 5$ </td></tr><tr><td>MiniGrid</td><td>0.253</td><td>0.326</td><td>0.373</td><td>0.603</td><td>0.624</td><td>0.651</td><td>0.114</td><td>0.155</td><td>0.235</td></tr><tr><td>Crafter (clean)</td><td>0.226</td><td>0.380</td><td>0.544</td><td>0.077</td><td>0.178</td><td>0.228</td><td>0.090</td><td>0.129</td><td>0.211</td></tr><tr><td>Crafter (with loops)</td><td>0.132</td><td>0.119</td><td>0.100</td><td>0.118</td><td>0.147</td><td>0.086</td><td>0.118</td><td>0.046</td><td>0.071</td></tr><tr><td>MetaWorld (clean)</td><td>0.906</td><td>0.977</td><td>0.985</td><td>0.531</td><td>0.360</td><td>0.333</td><td>0.333</td><td>0.320</td><td>0.294</td></tr><tr><td>MetaWorld (with loops)</td><td>0.643</td><td>0.819</td><td>0.893</td><td>0.685</td><td>0.573</td><td>0.593</td><td>0.274</td><td>0.313</td><td>0.390</td></tr></table>

Table D.2 AST telemetry in the BOIR Bayesian intent estimator. Confident incorrect starts are epochs with an initially incorrect maximum a posteriori hypothesis whose posterior probability exceeds 0.70. True-hypothesis resolution reports the mean posterior assigned to the correct human goal at the beginning and end of each epoch.
<table><tr><td>Scenario</td><td>Initial MAP errors</td><td>Confident wrong starts</td><td>True-hypothesis resolution</td><td> $\mathbf { S } \mathbf { I } ^ { \perp }$  post-change</td><td> $\mathbf { S I ^ { \perp } }$  mid-epoch</td></tr><tr><td>S1</td><td>20/40</td><td>4/40</td><td> $0 . 4 6 7  0 . 8 3 4$ </td><td>0.014</td><td>0.051</td></tr><tr><td>S3</td><td>60/60</td><td>42/60</td><td> $0 . 1 4 2  0 . 9 6 6$ </td><td>0.075</td><td>0.130</td></tr></table>

## G Formal Task-Referenced Acquisition Model

## G.1 Task binding and marked evidence

Let $\mathcal { M } = \{ 1 , \dots , N \}$ denote the finite set of task requirements and $\mathcal { V }$ the observation space. Let $S : \mathcal { V } $ $[ 0 , 1 ] ^ { N }$ map observations to task resolution and $e : \mathcal { V } \to \mathbb { R } ^ { d }$ map observations to unit-norm comparative representations. The task binding is $\boldsymbol { B } = ( \mathcal { M } , \boldsymbol { y } , S , e )$ . The marked observation ${ \boldsymbol { \chi } } _ { t } = \left( { y } _ { t } , { i } _ { t } \right)$ pairs the incoming evidence with the requirement targeted by the preceding action. The attribution marker satisfies $i _ { t } \in \mathcal { M } \cup \{ \perp \}$ , where ⊥ denotes an unattributed exchange. The two projections are

$$
\mathbf { s } ( t ) = S ( y _ { t } ) \in [ 0 , 1 ] ^ { N } , \qquad \mathbf { r } ^ { ( t ) } = e ( y _ { t } ) , \quad \| \mathbf { r } ^ { ( t ) } \| _ { 2 } = 1 .\tag{G.1}
$$

Attribution identifies the target of an action. Its evidence may resolve any declared requirement, including requirements outside the target and requirements acquired during unattributed exchanges.

## G.2 Acquisition state and admissibility

The attained-resolution state accumulates monotonically within a monitoring episode according to

$$
v _ { i } ( t ) = \operatorname* { m a x } \{ v _ { i } ( t - 1 ) , s _ { i } ( t ) \} , \qquad v _ { i } ( 0 ) = 0 , \quad \Delta v _ { i } ( t ) = v _ { i } ( t ) - v _ { i } ( t - 1 ) .\tag{G.2}
$$

The signed increment $\delta _ { i } ( t ) = s _ { i } ( t ) - v _ { i } ( t - 1 )$ records the difference between the current evidence score and the previously attained resolution. Each requirement retains a stable interpretation during the episode. Invalidation of verified evidence terminates that episode and resets the attained state and evidence histories. The causal monitor processes an ordered acquisition trace defined by

$$
\tau _ { 1 : t } = \big ( ( \chi _ { \ell } , \mathbf { s } ( \ell ) , \pmb { v } ( \ell ) , \Delta \pmb { v } ( \ell ) ) \big ) _ { \ell = 1 } ^ { t } , \qquad z _ { t } = \Phi _ { \Theta } ( \tau _ { 1 : t } ) = ( \mathbf { P } ( t ) , F ( t ) ) .\tag{G.3}
$$

The monitor emits prospective acquisition values $\mathbf { P } ( t ) \in \mathbb { R } _ { > 0 } ^ { N }$ and scalar friction $F ( t ) \in [ 0 , 1 ]$ . Appendix H specifies the frozen reference implementation used in the experiments.

## G.3 Gain-gated recurrence

The friction channel combines recurrence against historical attributed representations, a realized-gain gate and the remaining unresolved fraction of each requirement. The evaluated reference implementation follows the structural expression

$$
F ( t ) \simeq \sum _ { i \in \mathcal { A } _ { t } } \omega _ { i } ( t ) \underbrace { \mu _ { i } ( t ) } _ { \mathrm { r e c u r r e n c e ~ g a i n ~ g a t e } } \underbrace { \left( \varrho ( t ) \right) } _ { \mathrm { u n r e s o l v e d ~ v o l u m e } } , \qquad \omega _ { i } ( t ) = \frac { \mu _ { i } ( t ) } { \sum _ { j \in \mathcal { A } _ { t } } \mu _ { j } ( t ) } ,\tag{G.4}
$$

The implemented index includes numerical regularization, clipping and a zero-history convention, as specified in Eq. H.13. Positive realized gain attenuates recurrent evidence. Completed requirements contribute zero through their unresolved-volume factor. Recurrent evidence can raise friction when targeted gain is negligible or negative and the associated requirement remains incomplete.

## H Full Computational Specification of the Frozen Monitor

The frozen reference monitor implements the acquisition contract specified in Appendix G. Its causal update rules and numerical configuration are shared across all eight substrate bindings. The equations below define the evaluated estimator and friction index.

## H.1 Signed increments, counters and histories

At exchange t, for each coordinate $i \in { \mathcal { M } }$ the signed increment relative to the attained state and the realized gain are

$$
\delta _ { i } ( t ) = s _ { i } ( t ) - v _ { i } ( t - 1 ) , \qquad \Delta v _ { i } ( t ) = \operatorname* { m a x } \bigl ( 0 , \delta _ { i } ( t ) \bigr ) .\tag{H.1}
$$

The monitor updates attributed attempt counts $m _ { i } ( t )$ and material success counts $k _ { i } ( t )$ according to

$$
\begin{array} { r l r } & { } & { m _ { i } ( t ) = m _ { i } ( t - 1 ) + \mathbb { I } [ i _ { t } = i ] , \quad } \\ & { } & { k _ { i } ( t ) = k _ { i } ( t - 1 ) + \mathbb { I } [ i _ { t } = i ] \mathbb { I } [ \delta _ { i } ( t ) > \tau _ { \mathrm { g } } ] , } \end{array}\tag{H.2}
$$

where $\tau _ { \mathrm { g } }$ is the material-success threshold. Each coordinate keeps a gain-weighted evidence path

$$
\begin{array} { r } { { \bf c } _ { i } ( t ) = { \bf c } _ { i } ( t - 1 ) + \Delta v _ { i } ( t ) { \bf r } ^ { ( t ) } , \qquad { \bf c } _ { i } ( 0 ) = { \bf 0 } , } \end{array}\tag{H.3}
$$

and records its value after every positive-gain event. With $G _ { i } ( t )$ the number of such events through exchange $t ,$ the checkpoint sequence is $\mathcal { C } _ { i } ( t ) = ( \mathbf { c } _ { i } ^ { ( 0 ) } , \dots , \mathbf { c } _ { i } ^ { ( G _ { i } ( t ) ) } )$ , with ${ \bf c } _ { i } ^ { ( 0 ) } = { \bf 0 }$ . For friction estimation, each requirement retains a history of representations attributed to it, given by

$$
\mathcal { R } _ { i } ( t ) = \big \{ \mathbf { r } ^ { ( \ell ) } : 1 \leq \ell \leq t , i _ { \ell } = i \big \} .\tag{H.4}
$$

## H.2 Bayesian Progress Estimator

The yield $\theta _ { i }$ of an attributed attempt on coordinate i has a uniform Beta(1, 1) prior, so that

$$
\theta _ { i } ( t ) \mid m _ { i } ( t ) , k _ { i } ( t ) \sim \mathrm { B e t a } \big ( 1 + k _ { i } ( t ) , 1 + m _ { i } ( t ) - k _ { i } ( t ) \big ) ,\tag{H.5}
$$

with posterior mean and variance

$$
\begin{array} { l } { \displaystyle \mathbb { E } [ \theta _ { i } ( t ) ] = \frac { 1 + k _ { i } ( t ) } { 2 + m _ { i } ( t ) } , } \\ { \displaystyle \sigma _ { i } ^ { 2 } ( t ) = \frac { ( 1 + k _ { i } ( t ) ) ( 1 + m _ { i } ( t ) - k _ { i } ( t ) ) } { ( 2 + m _ { i } ( t ) ) ^ { 2 } ( 3 + m _ { i } ( t ) ) } . } \end{array}\tag{H.6}
$$

The uncertainty multiplier gives greater weight to coordinates with more uncertain estimated attempt yields and is defined by

$$
\epsilon _ { i } ( t ) = 1 + \beta \frac { \sigma _ { i } ^ { 2 } ( t ) } { \operatorname* { m a x } \big ( \varepsilon _ { \mathrm { P } } , \operatorname* { m a x } _ { j \in \mathcal { M } } \sigma _ { j } ^ { 2 } ( t ) \big ) } \in [ 1 , 1 + \beta ] .\tag{H.7}
$$

Let $n _ { i } ( t ) = \operatorname* { m i n } ( n , G _ { i } ( t ) + 1 )$ and $G = G _ { i } ( t )$ . The straightness of the most recent checkpoints is the ratio of net displacement to traversed path length, defined by

$$
\varsigma _ { i } ( t ) = \frac { \left\| \mathbf c _ { i } ^ { ( G ) } - \mathbf c _ { i } ^ { ( G - n _ { i } ( t ) + 1 ) } \right\| _ { 2 } } { \sum _ { j = G - n _ { i } ( t ) + 2 } ^ { G } \left\| \mathbf c _ { i } ^ { ( j ) } - \mathbf c _ { i } ^ { ( j - 1 ) } \right\| _ { 2 } + \varepsilon _ { \mathrm { G } } } \in [ 0 , 1 ] .\tag{H.8}
$$

The trajectory term is $\xi _ { i } ( t ) = 1 - \operatorname* { m i n } \bigl ( 1 , \upsilon _ { i } ( t ) / \upsilon ^ { \star } \bigr ) \varsigma _ { i } ( t )$ , with $\xi _ { i } ( t ) = 1$ before the first gain event $( G _ { i } ( t ) = 0 )$ . Declared prerequisites gate progress through $\begin{array} { r } { g _ { i } ( t ) = \prod _ { j \in \mathrm { d e p } ( i ) } \mathbb { E } [ \theta _ { j } ( t ) ] } \end{array}$ , with $g _ { i } ( t ) = 1$ when

$\mathrm { d e p } ( i ) = \emptyset$ . With binary entropy $\begin{array} { r } { H _ { b } ( v ) = - { v } \log _ { 2 } { v } - ( 1 - v ) \log _ { 2 } ( 1 - v ) } \end{array}$ , evaluated on ${ v } _ { i } ( t )$ clipped to $[ \varepsilon _ { \mathrm { P } } , 1 - \varepsilon _ { \mathrm { P } } ]$ , the progress channel is

$$
\begin{array} { r l } & { \mathrm { P E } _ { i } ^ { \mathrm { B } } ( t ) = w _ { i } g _ { i } ( t ) \Big [ \alpha \mathbb { E } [ \theta _ { i } ( t ) ] H _ { b } \big ( v _ { i } ( t ) \big ) \epsilon _ { i } ( t ) } \\ & { ~ + \left( 1 - \alpha \right) \xi _ { i } ( t ) \Big ] . } \end{array}\tag{H.9}
$$

The first term combines the posterior mean attempt yield, binary entropy of attained resolution and yield uncertainty. Its contribution approaches a numerical floor near full resolution because the entropy argument is clipped for stability. The second term decreases as the gain-weighted evidence path becomes straighter and attained resolution approaches its trajectory-normalization scale $\upsilon ^ { \star }$ . Circuitous paths and low attained resolution retain larger trajectory contributions.

## H.3 Gain-gated Stalling Index

The representations of earlier exchanges attributed to coordinate $i , \mathcal { R } _ { i } ( t - 1 )$ , form the columns of the history matrix $\mathbf { V } _ { i } ( t ^ { - } )$ . The projector onto the orthogonal complement of the historical subspace is

$$
\begin{array} { r } { \boldsymbol { \Pi } _ { i } ^ { \perp } ( t ) = \mathbf { I } - \mathbf { V } _ { i } ( t ^ { - } ) \mathbf { V } _ { i } ( t ^ { - } ) ^ { \dagger } , } \end{array}\tag{H.10}
$$

The operator $( \cdot ) ^ { \dagger }$ denotes the Moore-Penrose pseudoinverse. Orthogonal novelty and its complementary recurrence are defined by

$$
\eta _ { i } ( t ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \left( \left\| \mathbf { I } _ { i } ^ { \perp } ( t ) \mathbf { r } ^ { ( t ) } \right\| _ { 2 } \right) , \qquad \mu _ { i } ( t ) = 1 - \eta _ { i } ( t ) .\tag{H.11}
$$

Realized gain gates recurrence through

$$
\begin{array} { c } { D \big ( \varrho ( t ) \big ) = \operatorname* { m a x } \big ( D _ { \operatorname* { m i n } } , 1 - \lambda \varrho ( t ) \big ) , } \\ { \varrho ( t ) = \left\{ \begin{array} { l l } { \delta _ { i _ { t } } ( t ) , } & { i _ { t } \in \mathcal { M } , } \\ { \sum _ { i } \Delta v _ { i } ( t ) , } & { i _ { t } = \perp . } \end{array} \right. } \end{array}\tag{H.12}
$$

For directly attributed exchanges, negative signed gain increases the gating factor above one and amplifies recurrence. The factor D lies in $[ D _ { \mathrm { m i n } } , 1 + \lambda ]$ . Bindings with inferred attribution use aggregate realized gain $\varrho ( t ) = \textstyle \sum _ { i } \Delta v _ { i } ( t )$ at every exchange. BOIR uses this convention because its attribution $i _ { t }$ is the maximum a posteriori hypothesis. Let $\mathcal { A } _ { t } = \{ i \in \mathcal { M } \mid \mathcal { R } _ { i } ( t - 1 ) \neq \emptyset \}$ denote the active requirements. The implemented friction index is

$$
\mathrm { S I } ^ { \perp } ( t ) = \mathrm { c l i p } _ { [ 0 , 1 ] } \left[ \frac { \sum _ { i \in \mathcal { A } _ { t } } \mu _ { i } ( t ) ^ { 2 } D \big ( \varrho ( t ) \big ) \big ( 1 - v _ { i } ( t ) \big ) } { \sum _ { i \in \mathcal { A } _ { t } } \mu _ { i } ( t ) + \varepsilon _ { \mathrm { G } } } \right] ,\tag{H.13}
$$

The index is set to $\mathrm { S I } ^ { \perp } ( t ) = 0$ when $\begin{array} { r } { \mathcal { A } _ { t } = \emptyset \mathrm { ~ o r ~ } \sum _ { i \in \mathcal { A } _ { t } } \mu _ { i } ( t ) < \varepsilon _ { \mathrm { G } } } \end{array}$ . Each active requirement contributes recurrence weighted by its relative recurrence, gain gate and remaining unresolved fraction. Numerical regularization stabilizes the denominator and clipping bounds the resulting index in [0, 1]. Friction therefore increases with recurrent evidence that yields little resolution while task requirements remain open.

## I Example Telemetry Trajectories Across Substrates

Appendix Figures I.1 through I.10 show representative trajectories generated by experiments/figures.py from the released data files. For Map Task, MultiWOZ and AnnoMI, the selected sessions illustrate the lowest and highest friction cases in the evaluated samples. The left-hand panels show final attained resolution for each requirement, the full-resolution boundary and the completion threshold $v _ { \mathrm { c o m p l e t e } } = 0 . 8 0$ . The right-hand panels show trajectories through mean resolution, friction and total progress. Red markers indicate exchanges with $\mathrm { S I } ^ { \perp } > \theta _ { \mathrm { S I } } = 0 . 2 0$ . A diamond identifies the first exchange. These plots show the acquisition state, so consecutive exchanges with identical telemetry may overlap.

![](images/1414ba2a7343e20c14ca97af8e3b42f72515cb9ff217af14f0efde5748701dff.jpg)  
Fig. I.1 SAR hiker corpus with expert scoring. Efficient inquiry (S1) resolves all eight requirements without a stalled exchange. Repetitive questioning (S2) has a stall burden of 40% and resolves three requirements. Repetitive witness responses (S3) have a stall burden of 20% and resolve six requirements

![](images/344c79bed9983147481c0f85a14a1b311a2ae03c0dd8764b4f69bf2b1f4e3ce6.jpg)

![](images/43ab70ad124a88ffb94709d0d65f48e96edf1753d10846f1aad4088510568266.jpg)  
Fig. I.2 SAR elderly corpus with expert scoring. Efficient inquiry (S1) resolves all eight requirements without a stalled exchange. Repetitive questioning (S2) and repetitive witness responses (S3) resolve five requirements each and have stall burdens of 25% and 40%, respectively

Map Task q3nc1, highest mean friction  
Map Task q8ec7, lowest mean friction  
![](images/6cf1a47fefd31c92801d20c48b1c0633aa992ca380936aaf7ab8919321fa9c5f.jpg)

b  
![](images/c9aa28d8c9b21114e24fc720aa697b17df4a38365bace7d7047e91687073b766.jpg)

![](images/b06aac98e42c3ca1b57d255ba41b267c505e69260fc108a377d35d95ced9c9ba.jpg)

d  
![](images/4fc8e5c80f7e9fff02850c6171048738b0b446661f0bbe59d59167369de5f195.jpg)  
Fig. I.3 HCRC Map Task. Session q8ec7 has the lowest mean friction among $^ { 2 7 }$ evaluated dialogues, with 0.046 and no stalled exchange. Session q3nc1 has the highest mean friction, with 0.137 and a stall burden of 22%.

MultiWOZ SNG0403, highest mean friction  
MultiWOZ SNG0285, lowest mean friction  
![](images/2281211ac2116c196a57cc2e79e22c653aa3e88d281688222137d1c6c27848f7.jpg)

b  
![](images/e4ffa11403bbd1afd76d4d4eb55ef954e6945c431399616909f50d986af0f440.jpg)

![](images/deeb3c94ac6c600903b26f6bd1d035efe3cf7c0491d1cc130abffeee153c249c.jpg)

d  
![](images/6c03629643897b845f82cf4c7124e59f17a96ab37bddebe6d00fc86cc67fc495.jpg)  
Fig. I.4 MultiWOZ. Session SNG0285 contains no stalled exchange. Session SNG0403 is the only evaluated session with a stalled exchange, occurring once in four turns. Required slots resolve in single-turn updates.

![](images/7e9d8058111b6f252a0c63d7b5f64d64d218a38d91c85f942c3c6fb81c1a4e71.jpg)  
AnnoMI session 93, highest mean friction motivation\_change

b  
![](images/28529b69189a37c43ba1850a3b6aa9b4e7a32d4aa29ff5cf40a2e12b36821a55.jpg)

![](images/fe6911f9545c51e581e209ea67e450931af59819839cc2c96f6e10540b0516d9.jpg)

d  
![](images/d49c482ed30c7a1fb4d6ed8e5a8400b805af02be9f8fb56e42a0604d7895c3ef.jpg)  
Fig. I.5 AnnoMI. Session 16 has the lowest mean friction among 20 evaluated sessions, with 0.075 and no stalled exchange. Session 93 has the highest mean friction, with 0.442 and a stall burden of 58%. Both sessions received high clinical-quality ratings.

![](images/401232eaca302794af6dc96e89398a1698c759e1abd1aad3d978a5c3244d5db6.jpg)

b  
![](images/9da27780f9c7fd7af6e91e69af3e113eabf45648da2e9b1bdf48876d850c882e.jpg)

MiniGrid, loops in explored sectors (I2, seed 0)  
![](images/a8b761aed9f755a54a2845bf03955a3f130f616f6d8f2465c66cd79824b5f1e9.jpg)

![](images/86bab601b11a6e1301bdc2068e3942e3d9ffef334f9f491a30050982b4b96f1b.jpg)  
Fig. I.6 MiniGrid, seed 0. Panels a and b show Condition I1, with loops in incomplete sectors during exchanges 8 through 19 and 40 through 49. Panels c and d show Condition I2, with loops in completed sectors during exchanges 55 through 66.

![](images/b8db5d21933da40ef50c7f1d65a7138d3cd96a6b3f524470674f0eeabfed9271.jpg)

b  
![](images/8341afada1614443944fc83206c38601e87d10c32c8f28b42771bc3a40085718.jpg)

![](images/57e68c3e277b5996fb343fe421202069c2d064c39767f52fa458f8dd95f44df6.jpg)

d  
![](images/2acb47c88e43a1d01d6710eb8b99f3d50cb504d953846f618ebe3d889c960a1a.jpg)  
Fig. I.7 Crafter, seed 0. Panels a and b show clean execution. Panels c and d show no-op loops inserted during exchanges 3 through 5 and 8 through 10.

![](images/f58423ddced0373f4d4b07771801e2d0a0a2fc63ef37e1a4052021ab7a81a6d1.jpg)

![](images/eea6186376e26cc5b7b7b2d1e7a8a0cd31f7e3b55230436b0826751fe9f0ff0e.jpg)

![](images/3d0f10ed7a3ae6002bf7e9af22e90f77edb0e464c034b360d6846a5e4621c4b1.jpg)

![](images/1761b2f83b09971a60f783ab13d222c252522c1d3138817c833a3e76b3f1f76f.jpg)  
Fig. I.8 MetaWorld, seed 0. Panels a and b show Condition I1. Pre-grasp orbiting occurs during exchanges 2 through 15, with friction above the diagnostic threshold on all 14 exchanges at zero attained resolution. Panels c and d show Condition I2. Goal holding occurs during exchanges 22 through 35 after task completion at exchange 17. No holding exchange exceeds the friction threshold.

BOIR, scenario S3, trial 01  
![](images/19288b1cb2def95a7db3f5fdd9259818034c368ab3a0c1855c92c8bb0ae807e3.jpg)

b  
![](images/0cf0572ab26f12155e177a2462f321eec72016e11c1d37b8c13030c1409dbf6c.jpg)

![](images/b5086654c10d872de67c35108f33267c9ce454848ce016601e3ad00411590f03.jpg)

d  
![](images/c53d5862930d22bdc89362a7996b453acb8201762f3f51de6c2340c028fc3244.jpg)  
Fig. I.9 BOIR intent estimator, trial 01 in scenarios S1 and S3. The task coordinates correspond to goal hypotheses. The monitor resets after each goal change. The resolution diagram represents the final epoch, and the acquisition trajectory returns towards the origin after each reset.

![](images/ff646da8e6aba0161df57782de735b8c320638ce891f8e48d199f0a555010975.jpg)  
Fig. I.10 Closed-loop PPO episodes under Condition B, child corpus, seed 0. The figure shows Coupled-Blind, AST-Obs and AST-Pen policies. Figure 5 shows Baseline and Full-AST policies under the same condition.