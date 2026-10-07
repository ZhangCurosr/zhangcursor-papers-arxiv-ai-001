# Cooperating with Future Collaborators: Multi-Agent RL under Staggered Participation

Jianglin Qiao<sup>1</sup> Siyi Hu<sup>2</sup> Thien Hoang Nguyen<sup>1</sup> Zehong Cao<sup>3</sup> Salah Sukkarieh<sup>1</sup>

<sup>1</sup>ACFR, The University of Sydney, Sydney, Australia

<sup>2</sup>School of EECMS, Curtin University, Bentley, Australia

<sup>3</sup>School of CSIT, Adelaide University, Adelaide, Australia

{jianglin.qiao,thienhoang.nguyen,salah.sukkarieh}@sydney.edu.au

siyi.hu@curtin.edu.au jimmy.cao@adelaide.edu.au

In cooperative Multi-Agent Reinforcement Learning (MARL), agents are often trained under concurrent participation, while in many tasks some agents act earlier and leave task-relevant information that becomes useful to agents participating later. We study this setting as staggered participation (SP), which introduces a cross-time, cross-agent learning dependency because an early action may afect the return through the information it provides and the later policy that uses it. Learning under SP therefore requires both identifying what information is useful for future decisions and learning how later agents should use it. We propose Staggered Participation Learning (SPL), a training-time augmentation that addresses these two parts with prospective acquisition supervision for earlier agents and outcome-supervised receiver learning for later agents. We evaluate SPL across multiple policy-based MARL backbones, environments, and staggered-participation patterns. Across 60 MPE/RWARE backbone setting comparisons, SPL achieves higher observed mean task completion in every case, with an average diference of 14.1%. The gains also extend to eight-agent teams and a physics-based UAV–UGV environment in Isaac Lab, providing evidence across algorithmic, temporal, and embodied settings.

Keywords: Multi-Agent RL, Staggered Participation, Cooperative Tasks

## 1. Introduction

Cooperative agents do not always participate over the same time interval. For instance, in multi-robot systems, one agent may begin sensing or exploration earlier, while another becomes active later for navigation or task execution [1–3]. Their participation windows may partially overlap, allowing direct interaction for some time steps, or they may be fully separated. In these cases, earlier agents can acquire or leave task-relevant information that remains useful to agents participating later. We study this temporal structure as staggered participation (SP), where agents contribute to the same cooperative objective over temporally ofset participation windows. As illustrated in Fig. 1, these windows may partially overlap or be fully separated, while earlier participants can leave task-relevant information that supports later decisions.

Learning under SP is challenging in cooperative MARL because the efect of an early action may only become visible through the behavior of an agent acting later. An earlier agent may take an action that acquires or reveals information for downstream decisions, but the usefulness of that action depends on how the resulting information afects later behavior. The return therefore links decisions across both time and participants. When participation windows overlap, this dependency can coexist with direct interaction; when they are disjoint, it is carried through information and other consequences of earlier actions. This gives rise to two interdependent learning requirements: (i) Information Acquisition, where earlier agents learn which actions provide useful information for later decisions, and (ii) Downstream

![](images/ce45e357daf6fe47c28e49457078eebbb8aaa6e0234537f2694677c7a7ccdadd.jpg)  
Figure 1: Simultaneous and staggered participation. In simultaneous participation, agents contribute within the same participation period. Under staggered participation, earlier participants acquire or leave information that remains available to future collaborators and supports later decisions. Participation windows may partially overlap or be fully separated.

Utilization, where later agents learn how that information should influence action selection. Although a shared team return couples the two roles through a common objective, it does not provide separate supervision for acquiring useful evidence and using it efectively downstream.

SP is closely related to three existing directions in cooperative MARL. First, asynchronous MARL studies agents that act or communicate at diferent times [4–6]. Second, dynamic and open-team MARL considers agents that join, leave, or interact with changing teammates [7–9]. Third, communication MARL learns what information is useful for decision-making. Our focus is the learning dependency that appears when information acquisition and downstream utilization are distributed across diferent participation windows and potentially diferent agents. SP therefore places these two components within the same cooperative learning problem.

We introduce Staggered Participation Learning (SPL), a training-time augmentation that provides complementary supervision for the two learning requirements under SP. Prospective acquisition supervision guides earlier agents toward actions that provide useful information for future collaborators, while outcome-supervised receiver learning trains later agents to use inherited evidence based on downstream outcomes. SPL augments actor training without modifying environment rewards and introduces no additional communication or auxiliary inference at execution. We evaluate SPL across multiple policy-based MARL backbones, participation patterns, larger teams, and a physics-based UAV–UGV setting. Across the 60 primary Vanilla–SPL comparisons, SPL achieves higher observed mean task completion in every configuration, with an average gain of 14.1%. Our contributions are: (i) Staggered participation. We formulate cooperative MARL with temporally ofset participation and identify the dependency between evidence acquisition and downstream utilization; (ii) Staggered Participation Learning. We introduce complementary supervision for prospective acquisition and outcome-informed evidence use while preserving the underlying reward and decentralized execution; and (iii) Empirical evaluation. We evaluate SPL across policy-based MARL backbones, temporal participation patterns, larger teams, and physics-based UAV–UGV simulation.

## 2. Related Work

Simultaneous vs. Staggered Participation in Cooperative MARL. In simultaneous participation, agents contribute within the same interaction period. Cooperative MARL also studies temporal variation through asynchronous interaction, dynamic team composition, and staged decision making. Asynchronous methods consider agents that execute actions, communicate, or assign credit across asynchronous interactions [4–6, 10, 11], while dynamic and open-team approaches address agents that join, leave, or interact with changing teammates [7–9, 12]. Cooperative Consecutive Policies studies cooperation across consecutive decision stages [13]. These directions examine diferent aspects of when agents act and participate in a cooperative task. SP concerns temporally ofset participation windows, which may partially overlap or be fully separated. In the tasks studied here, earlier participants can acquire task-relevant evidence before future collaborators begin participating, and this evidence remains available to support downstream decisions.

Direct vs. Indirect Communication Protocols in Cooperative MARL. Communication methods support direct information exchange and learn what to send and with whom to communicate, including under intermittent connectivity [14–19]. Cooperation can also occur indirectly through persistent environmental traces or another agent’s subsequent actions. Stigmergic methods support coordination through such environmental traces [20], while MAE-MAPPO addresses indirect cooperation with invisible collaborators through marginal advantage estimation [21]. Related methods provide complementary mechanisms for retaining and interpreting information: sequence-based methods retain long-horizon context [22], while $S \mathbf { M P E } ^ { 2 }$ and $\mathtt { M A } ^ { 2 } \mathtt { E }$ model or reconstruct information unavailable from local observations [23, 24]. Policy distillation transfers training-time coordination knowledge into decentralized policies [25, 26]. Informative sensing and Dec-POMDP approaches further guide information acquisition according to its expected decision value [27–29]. The form of SP studied here combines temporally ofset participation with indirect cooperation through persistent task-relevant evidence. Within this setting, SPL focuses on evidence acquisition and downstream utilization, while related credit-assignment methods study how cooperative outcomes are attributed across agents or coalitions [21, 30]. SPL builds on value-aware information acquisition and policy distillation to supervise earlier participants and learn receiver preferences from attributable downstream outcomes. Persistent evidence connects the two roles without requiring additional execution-time communication.

## 3. Methodology

## 3.1. Staggered Participation (SP)

Cooperative decision process. We model the cooperative task as a decentralized partially observable Markov decision process (Dec-POMDP) [31], defined by $\mathcal { M } = \langle \mathcal { N } , \mathcal { S } , \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { N } } , \{ \mathcal { U } _ { i } \} _ { i \in \mathcal { N } } , P , R , \gamma \rangle$ where $\mathcal { N } = \{ 1 , \ldots , N \}$ is the set of agents and S is the global state space. At time $t ,$ the environment is in state $s _ { t } \in S$ and agent i receives a local observation $o _ { t } ^ { i } \in \mathcal { O } _ { i }$ . Each agent maintains an action– observation history $\tau _ { t } ^ { i }$ and selects an action $u _ { t } ^ { i } \in \mathcal { U } _ { i }$ according to $\pi _ { \theta _ { i } } \big ( u _ { t } ^ { i } \ \big | \ \tau _ { t } ^ { i } , M _ { t } ^ { i } \big )$ , where $M _ { t } ^ { i }$ denotes the legal-action mask. The joint action $\mathbf { u } _ { t } = ( u _ { t } ^ { 1 } , \dots , u _ { t } ^ { N } )$ produces the shared reward $r _ { t } = R ( s _ { t } , \mathbf { u } _ { t } )$ and the next state is sampled according to $s _ { t + 1 } \sim P ( \cdot \mid s _ { t } , \mathbf { u } _ { t } )$ . All agents optimize the common discounted team return $\begin{array} { r } { G = \sum _ { t = 0 } ^ { \hat { T } - 1 } \gamma ^ { t } r _ { t } } \end{array}$

Staggered participation. Agents may participate over diferent temporal intervals, so teammates need not be active throughout the same period. Let $\mathcal { N } _ { t } ^ { \mathsf { p a r t } } \subseteq \mathcal { N }$ denote the set of agents participating at time t; agents outside this set take a null action and make no policy decision. For each agent $i ,$ let $[ t _ { i } ^ { \mathrm { s t a r t } } , t _ { i } ^ { \mathrm { e n d } } )$ denote its participation interval, where $t _ { i } ^ { \mathrm { s t a r t } }$ is the first participating time and $t _ { i } ^ { \mathrm { e n d } }$ is the exclusive end time. We call agent j afuture collaborator of agent i if $t _ { i } ^ { \mathrm { s t a r t } } < t _ { j } ^ { \mathrm { s t a r t } }$ and both contribute to the same cooperative objective. Their participation intervals may be disjoint, with $t _ { i } ^ { \mathrm { e n d } } \leq t _ { j } ^ { \mathrm { s t a r t } }$ , or may partially overlap, with $t _ { j } ^ { \mathrm { s t a r t } } < t _ { i } ^ { \mathrm { e n d } }$ . In both cases, agent i can act before agent j begins participating.

![](images/3ef47f84cea02d4c74286860385db4d15b26445e15eb276198ed48cf9a2d159e.jpg)

![](images/47497ed3cbe4cb943728280edffce0843e87d7ea2200d8c38be5fa0398d4b1fe.jpg)  
Figure 2: Overview of SP and SPL. Earlier participants receive prospective acquisition supervision, while future collaborators learn evidence-conditioned preferences from downstream outcomes. SPL is used only during training.

Learning problem under SP Earlier information-gathering actions can change the evidence available to future collaborators and thereby influence downstream decisions and the shared return. This creates two interdependent learning requirements: what evidence earlier participants should acquire and how future collaborators should use it. Because the efect of an earlier action is realized through persistent evidence and subsequent decisions by another policy, its learning signal is delayed and indirect. We therefore focus on improving both evidence preparation and downstream evidence use. The next subsection introduces SPL to support these two requirements.

## 3.2. Staggered Participation Learning (SPL)

Under SP, earlier participants and future collaborators are coupled through persistent task-relevant evidence. SPL augments the native cooperative MARL objective with two complementary actor-side training signals. Prospective acquisition supervision guides earlier participants toward evidence-gathering actions expected to support future decisions, while outcome-supervised receiver learning teaches future collaborators how inherited evidence should influence downstream action selection. Both augment native actor training while leaving the environment reward and backbone critic objective unchanged, as illustrated in Fig. 2.

Structured training supervision. SPL uses structured task knowledge available during training to construct its auxiliary targets. Prospective acquisition supervision uses task-family validity priors, the fixed valid-task constraint, and the probe observation model to estimate the downstream value of candidate evidence-gathering actions. Receiver supervision uses fixed task-level utilities for successful and unsuccessful execution, while candidate-specific success probabilities are learned from attributable outcomes observed during training. These quantities describe task-generation, sensing, and outcome semantics; they do not reveal realized task validity, future outcomes, or a predefined correct action. In particular, the fixed acquisition model does not provide receiver success probabilities. The acquisition and receiver objectives are complementary but separately constructed: the acquisition target is fixed by the structured task model and is not updated from the learned receiver policy or outcome model. SPL therefore assumes that such task structure is available from the training environment or task specification. All SPL-specific models and targets remain training-only and are unavailable during execution. Implementation details are provided in Appendix B.4.

## 3.2.1. Prospective Acquisition Supervision

Prospective evidence value. Consider an eligible evidence-acquisition decision of an earlier participant i. Let $c _ { t } ^ { i }$ denote its actor-visible acquisition context and $M _ { t } ^ { i }$ the corresponding legal-action mask. For each legal task-directed action $u ,$ SPL uses a fixed training-time task model to estimate $\Delta _ { t } ^ { i } ( u )$ , the expected increase in downstream decision value associated with the evidence produced by that action. SPL combines this prospective value with the corresponding travel or acquisition distance $d _ { t } ^ { i } ( u )$

$$
s _ { t } ^ { i } ( u ) = \log \Bigl ( \operatorname* { m a x } \{ \Delta _ { t } ^ { i } ( u ) , \epsilon _ { \mathrm { a c q } } \} \Bigr ) - \frac { d _ { t } ^ { i } ( u ) } { \tau _ { d } } ,\tag{1}
$$

where $\tau _ { d }$ is a fixed distance temperature and $\epsilon _ { \mathrm { { a c q } } } > 0$ is a numerical stabilizer. A deterministic transform $\mathcal { T } _ { \mathrm { a c q } }$ masks illegal actions, normalizes the scores of positive-gain task actions, and reserves a small fixed probability mass for the scan action, which gathers local evidence without targeting a specific candidate. If no positive-gain task action is available, the scan action receives all target mass. The resulting prospective acquisition target is $q _ { \mathrm { a c q } } ^ { i } = \mathcal { T } _ { \mathrm { a c q } } ( s _ { t } ^ { i } , M _ { t } ^ { i } )$ .

Acquisition policy supervision. For eligible earlier-participant decisions $\mathcal { D } _ { \mathrm { a c q } } .$ , SPL distils the prospective acquisition target into the decentralized policy:

$$
\mathcal { L } _ { \mathrm { S P L } } ^ { \mathrm { a c q } } = \mathbb { E } _ { ( i , t ) \sim \mathcal { D } _ { \mathrm { a c q } } } \left[ \mathrm { C E } \left( q _ { \mathrm { a c q } } ^ { i } , \pi _ { \theta _ { i } } ( \cdot \mid \tau _ { t } ^ { i } , M _ { t } ^ { i } ) \right) \right] .\tag{2}
$$

This auxiliary loss updates only the earlier-participant actor.

## 3.2.2. Outcome-Supervised Receiver Learning

Outcome-labeled decisions. Consider a future collaborator j making a downstream decision at time $t \geq t _ { j } ^ { \mathrm { s t a r t } }$ . Let $\ v { x } _ { t } ^ { j }$ denote the actor-visible information available before the decision, including retained task-relevant evidence and the collaborator’s current local context. The collaborator selects $u _ { t } ^ { j }$ using its decentralized policy, and the selected task subsequently produces an attributable outcome $y _ { t } ^ { j } \in \{ 0 , 1 \}$ ， where $y _ { t } ^ { j } = 1$ denotes successful completion and $y _ { t } ^ { j } = 0$ an attributable unsuccessful attempt. Each eligible decision contributes a tuple $( x _ { t } ^ { j } , u _ { t } ^ { j } , j , y _ { t } ^ { j } )$ to the outcome replay set $\mathcal { D } _ { \mathrm { o u t } }$ . Subsequent observations are used only to determine the outcome label; we exclude interrupted, expired, or unresolved attempts when their outcomes cannot be reliably attributed.

Candidate outcome model. SPL learns how the evidence and local context available to a future collaborator relate to the outcome of each candidate action. Let $g _ { j } ( x , u )$ collect the actor-visible decision context, candidate-action features, and collaborator-role identity. A candidate-shared scorer $f _ { \psi }$ predicts

$$
p _ { \psi } ^ { j } ( u \mid x ) = \mathrm { s i g m o i d } \left( f _ { \psi } { \big ( } g _ { j } ( x , u ) { \big ) } \right) ,\tag{3}
$$

where $p _ { \psi } ^ { j } ( u \mid x )$ estimates the probability that candidate u succeeds under information x. The model is trained from attributable outcomes using binary cross-entropy,

$$
\mathcal { L } _ { \mathrm { o u t } } ( \psi ) = \mathbb { E } _ { ( x , u , j , y ) \sim \mathcal { D } _ { \mathrm { o u t } } } \left[ \ell _ { \mathrm { B C E } } \left( y , p _ { \psi } ^ { j } ( u \mid x ) \right) \right] .\tag{4}
$$

The outcome model is therefore learned from the observed consequences of the future collaborator’s own selected actions. It receives no labels identifying which candidate action should have been selected.

Evidence-conditioned action utility. Outcome probability alone does not determine the desirability of a candidate when successful and unsuccessful executions have diferent task-level consequences. Let $c _ { j } ^ { + } ( u )$ and $c _ { j } ^ { - } \left( u \right)$ denote fixed task-level utilities associated with successful and unsuccessful execution of candidate u, respectively. SPL combines these utilities with the evidence-conditioned success probability predicted by a slowly updated target outcome model with parameters ψ¯:

$$
U _ { \bar { \psi } } ^ { j } ( u \mid x ) = p _ { \bar { \psi } } ^ { j } ( u \mid x ) c _ { j } ^ { + } ( u ) + \left[ 1 - p _ { \bar { \psi } } ^ { j } ( u \mid x ) \right] c _ { j } ^ { - } ( u ) .\tag{5}
$$

Because $p _ { \bar { \psi } } ^ { j } ( u \mid x )$ is conditioned on decision-time information, the same candidate can receive diferent utility estimates under diferent evidence prepared by earlier participants.

Receiver target and policy supervision. Let $\mathcal { A } _ { t } ^ { j } = \{ u \in \mathcal { U } _ { j } : M _ { t } ^ { j } ( u ) = 1 \}$ denote the legal actions available to collaborator $j .$ . SPL standardizes the predicted utilities over this set and constructs a soft preference target:

$$
z _ { t } ^ { j } ( u ) = \frac { U _ { \bar { \psi } } ^ { j } ( u \mid x _ { t } ^ { j } ) - \mu _ { t } ^ { j } } { \sqrt { ( \sigma _ { t } ^ { j } ) ^ { 2 } + \epsilon _ { z } } } , \qquad q _ { \mathrm { S P L } } ^ { j } ( u ) = \frac { \exp ( \beta z _ { t } ^ { j } ( u ) ) } { \sum _ { v \in \mathcal { A } _ { t } ^ { j } } \exp ( \beta z _ { t } ^ { j } ( v ) ) } ,\tag{6}
$$

where $\mu _ { t } ^ { j }$ and $( \sigma _ { t } ^ { j } ) ^ { 2 }$ are the mean and variance of the legal-action utilities, $\epsilon _ { z } > 0$ is a variance stabilizer, and $\beta > 0$ controls the strength of the induced preference. Illegal actions receive zero probability. Since the target is constructed from learned evidence-conditioned outcome predictions, it expresses relative preferences among the currently legal actions and does not require predefined correct-action labels. For eligible future-collaborator decision points $\mathcal { D } _ { \mathrm { r e c } }$ , SPL distills this soft target into the decentralized policy:

$$
\mathcal { L } _ { \mathrm { S P L } } ^ { \mathrm { r e c } } = \mathbb { E } _ { ( j , t ) \sim \mathcal { D } _ { \mathrm { r e c } } } \left[ \mathrm { C E } \left( q _ { \mathrm { S P L } } ^ { j } , \pi _ { \theta _ { j } } ( \cdot \mid \tau _ { t } ^ { j } , M _ { t } ^ { j } ) \right) \right] .\tag{7}
$$

The target is treated as fixed during this update. Gradients therefore modify only the future-collaborator actor and do not propagate through the outcome model, recorded outcomes, or environment dynamics.

Training and Execution. SPL combines $\mathcal { L } _ { \mathrm { S P L } } ^ { \mathrm { a c q } }$ for earlier participants with $\mathcal { L } _ { \mathrm { S P I } } ^ { \mathrm { r e c } }$ for future collaborators, while the MARL backbone continues to optimize its native objective from the shared team return. After each rollout, attributable outcomes are added to $\mathcal { D } _ { \mathrm { o u t } }$ and used to update the outcome model once replay-readiness conditions are met; the target parameters are updated by $\bar { \psi }  ( 1 - \rho ) \bar { \psi } + \rho \psi$ SPL actor updates are interleaved with native actor optimization, while critic optimization follows the original backbone procedure. During execution, agents act solely through $u _ { t } ^ { i } \sim \pi _ { \theta _ { i } } ( { \cdot } \ | \ \tau _ { t } ^ { i } , M _ { t } ^ { i } )$ ; no acquisition model, auxiliary target, outcome replay, outcome model, or utility estimate is required. We provide detailed readiness conditions, update ordering, gradient boundaries, and the complete training algorithm in Appendices B.6 and B.7.

## 3.3. Theoretical Analysis

We provide a local analysis of the auxiliary objectives introduced by SPL. The results characterize the prospective acquisition target, calibration of the outcome-based receiver model, and local decision quality induced by the receiver target under the structured task assumptions of Sec. 3.2. The acquisition result instantiates a standard value-of-information argument for the top-k surrogate used by SPL, while the receiver results characterize calibration and distillation errors. Together, these results support the construction of the SPL training signals but do not form an end-to-end guarantee or establish global convergence of the cooperative return. Full proofs, assumptions, and guarantee boundaries are provided in Appendix A.

Prospective acquisition. Consider task families represented by the fixed binary-validity model used to construct acquisition supervision. Let $b = \left( b _ { 1 } , \ldots , b _ { n } \right)$ denote posterior validity marginals under this model, where $b _ { r } = \operatorname* { P r } ( W _ { r } = 1 \mid c _ { t } ^ { i } )$ . If a future collaborator can execute at most k tasks, define the top-k posterior completion value as $\begin{array} { r } { V _ { k } ( b ) = \operatorname* { m a x } _ { S : | S | \leq k } \sum _ { r \in S } b _ { r } } \end{array}$ . For a purely informational acquisition action $u ,$ let $Z _ { u }$ denote the resulting observation and $b ^ { Z _ { u } }$ the updated posterior. Its model-relative prospective gain is $\Delta _ { t } ^ { i } ( u ) = \mathbb { E } \big [ V _ { k } \big ( b ^ { Z _ { u } } \big ) \ \big | \ c _ { t } ^ { i } , u \big ] - V _ { k } ( b )$ . Let $\mathcal { P } _ { t } ^ { i } = \{ u : M _ { t } ^ { i } ( u ) = 1$ , u is task-directed, $\Delta _ { t } ^ { i } ( u ) > 0 \}$ denote the legal task-directed actions with positive prospective gain.

Proposition 1 (Prospective acquisition target). Assume that acquisition actions change only the available information and that posterior updating and predictive expectation are Bayesian-consistent under the acquisition model. Then $\Delta _ { t } ^ { i } ( u ) \geq 0$ for every legal acquisition action. Moreover, before adding the scan mass, $i f T _ { \mathrm { a c q } }$ applies Gibbs normalization over $\mathcal { P } _ { t } ^ { i } ,$ , the acquisition target satisfies $q _ { \mathrm { a c q } } ^ { i } ( u ) \propto \exp ( s _ { t } ^ { i } ( u ) )$ and uniquely maximizes the corresponding entropy-regularized score objective.

Proof sketch. The top-k value $V _ { k }$ is the maximum of linear functions of the posterior marginals and is therefore convex. Bayesian consistency gives $\mathbb { E } \big [ b ^ { Z _ { u } } \mid c _ { t } ^ { i } , u \big ] = b _ { \mathrm { z } }$ , so Jensen’s inequality implies $\mathbb { E } \big [ V _ { k } \big ( b ^ { Z _ { u } } \big ) \ \big | \ c _ { t } ^ { i } , u \big ] \geq V _ { k } ( b )$ . The Gibbs form follows from the standard entropy-regularized maximization problem. The full proof is provided in Appendix A.2.

Outcome prediction and local utility. For an attributable future-collaborator decision, let $Y \in \{ 0 , 1 \}$ denote its downstream outcome and let $C \in \{ 0 , 1 \}$ indicate whether that outcome is retained in $\mathcal { D } _ { \mathrm { o u t } }$ . We define $p _ { * } ^ { j } ( u \mid x ) = \operatorname* { P r } ( Y = 1 \mid X = x , U = u , J = j , C = 1 )$ , with corresponding local utility $U _ { * } ^ { j } ( u \mid x ) = p _ { * } ^ { j } ( u \mid x ) c _ { j } ^ { + } ( u ) + [ 1 - p _ { * } ^ { j } ( u \mid x ) ] c _ { j } ^ { - } ( u )$ . For any predictor $p ,$ , let $\mathcal { L } _ { \mathrm { o u t } } ( \boldsymbol { p } )$ denote its population binary-cross-entropy risk over retained outcomes and $U _ { p } ^ { j }$ the induced local utility.

Proposition 2 (Retained-outcome calibration). On the covered retained-outcome support, $\mathcal { L } _ { \mathrm { o u t } } ( p )$ is uniquely minimized pointwise by $\begin{array} { r } { p _ { * } ^ { j } . \ I f C _ { \Delta } = \operatorname* { s u p } _ { u , j } | c _ { j } ^ { + } ( u ) - c _ { j } ^ { - } ( u ) | < \infty . } \end{array}$ , then

$$
{  { \mathbb E } } \bigg [ \Big ( U _ { p } ^ { J } ( U \mid X ) - U _ { * } ^ { J } ( U \mid X ) \Big ) ^ { 2 } \mid C = 1 \bigg ] \le \frac { C _ { \Delta } ^ { 2 } } { 2 } \left[ {  { \mathcal L } } _ { \mathrm { o u t } } ( p ) - {  { \mathcal L } } _ { \mathrm { o u t } } ( p _ { * } ) \right] .\tag{8}
$$

Proof sketch. Excess binary-cross-entropy risk equals the Bernoulli KL divergence on the retainedoutcome distribution. Since $U _ { p } ^ { j } - U _ { * } ^ { j } = ( p ^ { j } - p _ { * } ^ { j } ) ( c _ { j } ^ { + } - c _ { j } ^ { - } )$ , Pinsker’s inequality gives Eq. 8. The result applies to retained, selected receiver actions and is conditional on $C = 1 ;$ ; an unconditional interpretation additionally requires $\ Y \perp C \mid X , U , J $ . The full proof is provided in Appendix A.3.

Receiver decision quality. Fix an eligible future-collaborator decision with legal action set $A$ and $K = | { \mathcal { A } } |$ . Let $z _ { * }$ denote the reference standardized score and $z _ { \bar { \psi } }$ the score produced by the target outcome model. Define $\begin{array} { r } { \delta = \| z _ { \bar { \psi } } - z _ { * } \| _ { \infty } , R _ { z } = \operatorname* { m a x } _ { u \in \mathcal { A } } z _ { * } ( u ) - \operatorname* { m i n } _ { u \in \mathcal { A } } z _ { * } ( u ) } \end{array}$ , and

$$
\eta = \mathrm { K L } ( q _ { \mathrm { S P L } } ^ { j } | | \pi _ { t } ^ { j } ) = \mathrm { C E } ( q _ { \mathrm { S P L } } ^ { j } , \pi _ { t } ^ { j } ) - H ( q _ { \mathrm { S P L } } ^ { j } ) ,
$$

![](images/174a5f9335f4bf4d9f2ade93dc8d2b6dfe5978f881e9c7a7d22188014f844782.jpg)  
Figure 3: Evaluation environments and temporal participation patterns. (a) MPE, (b) RWARE, (c) the physicsbased UAV–UGV environment in Isaac Lab, and (d) the five participation patterns used in MPE and RWARE.

where $q _ { \mathrm { S P L } } ^ { j } ( u ) \propto \exp ( \beta z _ { \bar { \psi } } ( u ) )$ . Thus, $\eta$ is directly measurable from the receiver target and actor distributions. The score error δ additionally depends on generalization across the full legal action set and score standardization; controlling it from outcome-model error requires stronger coverage conditions.

Theorem 1 (Local receiver score regret). For any legal receiver policy $\pi _ { t } ^ { j }$ satisfying KL $\begin{array} { r } { \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \mathopen { } \mathclose \bgroup \left( q _ { \mathrm { S P L } } ^ { j } \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \aftergroup \egroup \right) < \infty , } \end{array}$

$$
\mathrm { R e g } _ { z , t } ^ { j } ( \pi _ { t } ^ { j } ) \leq \frac { \log K } { \beta } + 2 \delta + R _ { z } \sqrt { \frac { \eta } { 2 } } ,\tag{9}
$$

where Reg<sup>j</sup><sub>z,t</sub>(π<sup>j</sup><sub>t</sub>) = max<sub>u∈A</sub> z<sub>∗</sub>(u) − E<sub>u∼πj</sub> [z<sub>∗</sub>(u)].

Proof sketch. The Boltzmann target contributes a softness gap of at most log $K / \beta .$ . Replacing $z _ { \bar { \psi } }$ with $z _ { * }$ contributes at most 2δ, and Pinsker’s inequality bounds the remaining distillation error by $R _ { z } \sqrt { \eta / 2 }$ The three terms therefore capture finite-temperature softness, score approximation, and incomplete receiver distillation. The full proof is provided in Appendix A.5.

Scope. The results characterize local components of SPL under the structured training assumptions of Sec. 3.2. Proposition 1 gives a model-relative information-value guarantee: internally consistent Bayesian updating preserves non-negative prospective gain, but misspecified priors can change how this gain relates to performance under the true environment distribution; this sensitivity is evaluated in Appendix D.6. Proposition 2 controls expected local-utility error on the retained selected-action distribution. Theorem 1 separates the measurable distillation error <sub>η</sub> from the score-approximation term δ and bounds standardized local-score regret, not raw utility or full team return. Relating δ directly to outcome-model training requires stronger coverage and generalization assumptions across legal actions.

## 4. Experimental Setup

Environments. We evaluate SPL in MPE, RWARE, and a physics-based UAV–UGV environment, where earlier participants gather task-relevant evidence before future collaborators perform downstream tasks. MPE-Base builds on the Multi-Agent Particle Environment [32], while MPE-Advanced further randomizes task assignments, spawn positions, and future-collaborator identities. RWARE-Base transfers the same evidence-mediated cooperation problem to the Multi-Robot Warehouse domain [33], and RWARE-Advanced additionally introduces task permutation, a shared candidate pool, and dynamic task claims. Isaac Lab [34] provides an embodied UAV–UGV setting with physical motion and obstacle constraints. Full environment specifications are given in Appendices B.1–B.3. For MPE and RWARE, we evaluate five patterns: Non-overlap, with a strict handof between roles; Overlap-25 and Overlap-50, with fixed temporal overlap; Dynamic, with stochastic pair-specific handof times; and Cascaded, with successive participation intervals. Isaac Lab uses corresponding UAV–UGV schedules. Exact participation windows are provided in Appendices B.2–B.3.

![](images/d78aa51fb33474275486be325d5f9d0075d575847e1c19a7956223a589d2c974.jpg)  
Figure 4: Training TC (%) for all methods across 4 environments and 5 participation patterns averaged over 5 training seeds.

Implementation We augment three PPO-based MARL backbones—IPPO, MAPPO, and HAPPO—to obtain SPL-IPPO, SPL-MAPPO, and SPL-HAPPO. MAPPO [35] is the primary backbone, while HAPPO [36] and IPPO [37] test compatibility with heterogeneous and independent learning, respectively. We additionally compare with MAE-MAPPO [21], Sable [22], and SMPE<sup>2</sup> [23] as complementary crossmethod references. The primary controlled comparisons remain between each Vanilla backbone and its SPL-augmented counterpart. Baseline adaptations and implementation details are provided in Appendix B.5. Our implementation builds on HARL [38]. SPL preserves the native actor–critic archi tecture and training objective of each backbone and adds only role-specific auxiliary actor updates during training; SPL-specific models are not required during execution. Optimization details, training procedures, and observed training costs are reported in Appendices B–C.

Evaluation. Task completion (TC) is the primary metric. The four-agent MPE/RWARE experiments use five independent training seeds per method and configuration, while the eight-agent and Isaac Lab evaluations use three; all final evaluations contain 200 episodes per seed. Results are reported as mean ± standard error across training seeds. Throughout the results, reported TC gains denote absolute diferences between percentage-valued TC scores and are written using %. The primary Vanilla–SPL comparisons are additionally examined using paired seed-level analyses where same-seed pairing is available; complete confidence intervals, Holm-adjusted tests, and pairing details are reported in Appendix D.7. Additional metrics, including full success, return, failed execution, and exploration diagnostics, are reported in Appendix D. All configurations are trained and evaluated independently, so the reported results characterize within-configuration performance rather than transfer across participation patterns or task structures.

Table 1: Final task completion (TC, %) after 2M environment steps. Values are mean ± standard error over five training seeds. SPL columns are shaded; bold indicates the highest observed mean in each setting and does not denote statistical significance.
<table><tr><td rowspan="2">Environment</td><td rowspan="2">Participation</td><td colspan="2">IPPO</td><td colspan="2">MAPPO</td><td colspan="2">HAPPO</td><td colspan="3">Other baselines</td></tr><tr><td>Van.</td><td>SPL</td><td>Van.</td><td>SPL</td><td>Van.</td><td>SPL</td><td>MAE-MAPPO</td><td>Sable</td><td>SMPE2</td></tr><tr><td rowspan="5">MPE-Base</td><td>Non-overlap</td><td>70.50 ± 0.00</td><td>78.38±0.38</td><td>70.50 ± 0.00</td><td>78.73±0.43</td><td>67.45 ± 3.05</td><td>79.18±0.23</td><td>70.50 ± 0.00</td><td>62.70 ± 0.25</td><td>42.73 ± 9.29</td></tr><tr><td>Overlap-25</td><td>68.25 ± 2.75</td><td>79.35±0.57</td><td>70.78 ± 0.14</td><td>80.25±0.21</td><td>70.70 ± 0.18</td><td>80.35±0.29</td><td>67.93 ± 2.95</td><td>63.50 ± 2.24</td><td>53.85 ± 5.43</td></tr><tr><td>Overlap-50</td><td>67.85 ± 3.15</td><td>78.90± 0.31</td><td>69.53 ± 1.27</td><td>78.90 ± 0.24</td><td>69.80 ± 0.92</td><td>78.50±0.47</td><td>70.63 ± 0.23</td><td>64.43 ± 1.85</td><td>63.95 ± 3.34</td></tr><tr><td>Dynamic</td><td>66.20 ± 3.01</td><td>79.65±0.18</td><td>70.23 ± 0.59</td><td>78.88±0.18</td><td>69.53 ± 0.99</td><td>79.63±0.39</td><td>64.85 ± 3.06</td><td>58.30 ± 1.11</td><td>49.70 ± 1.04</td></tr><tr><td>Cascaded</td><td>66.83±3.68</td><td>78.90 ± 0.70</td><td>70.63 ± 0.10</td><td>78.93±0.48</td><td>70.68 ± 0.19</td><td>78.78±0.22</td><td>70.45 ± 0.05</td><td>38.03 ± 0.93</td><td>27.93 ± 5.74</td></tr><tr><td rowspan="5">MPE-Advanced</td><td>Non-overlap</td><td>70.38 ± 0.00</td><td>77.95±0.19</td><td>64.08 ± 3.86</td><td>78.23 ± 0.16</td><td>67.28 ± 3.10</td><td>78.03 ± 0.16</td><td>63.28 ± 3.47</td><td>64.90 ± 1.36</td><td>49.78 ± 3.25</td></tr><tr><td>Overlap-25</td><td>65.65± 3.53</td><td>79.95±0.52</td><td>68.28 ± 1.50</td><td>79.38±0.27</td><td>70.93 ± 0.56</td><td>79.55±0.40</td><td>71.40 ± 0.23</td><td>64.30 ± 1.63 47.83 ± 9.02</td><td></td></tr><tr><td>Overlap-50</td><td>71.30 ± 0.22</td><td>78.63±0.23</td><td>68.58±3.05</td><td>78.33±0.40</td><td>71.40 ± 0.14</td><td>78.68± 0.25</td><td>62.63 ± 5.92</td><td>61.88 ± 1.485</td><td>52.08 ± 8.25</td></tr><tr><td>Dynamic</td><td>66.63 ± 3.66</td><td>78.23± 0.29</td><td>67.05 ± 3.20</td><td>79.08±0.54</td><td>61.73±5.55</td><td>78.80± 0.18</td><td>67.68 ± 1.76</td><td>57.60 ± 1.30</td><td>43.10 ± 6.61</td></tr><tr><td>Cascaded</td><td>67.68 ± 2.58</td><td>76.80± 0.13</td><td>67.28 ± 3.10</td><td>77.00 ± 0.33</td><td>69.28 ± 1.10</td><td>77.23±0.18</td><td>66.83 ± 3.02</td><td>62.73 ± 0.98</td><td>56.70 ± 5.50</td></tr><tr><td rowspan="5">RWARE-Base</td><td>Non-overlap</td><td>66.45± 5.86</td><td>86.83±0.40</td><td></td><td></td><td>73.18 ± 1.50</td><td>86.93±0.26</td><td>73.48 ±1.69</td><td>71.80 ± 0.98 6</td><td></td></tr><tr><td>Overlap-25</td><td>81.10 ± 3.41</td><td>89.15±0.16</td><td>71.38 ± 2.27 74.25 ± 3.41</td><td>86.70±0.49 89.08±0.35</td><td>70.15 ± 2.93</td><td>88.35±0.50</td><td>72.50 ± 2.86</td><td>76.70 ± 1.18 67.28 ± 2.87</td><td>66.05 ± 3.29</td></tr><tr><td>Overlap-50</td><td>82.00 ± 1.12</td><td>89.00 ± 0.19</td><td>74.73 ± 3.35</td><td>88.58±0.28</td><td>70.28 ± 2.41</td><td>88.80±0.23</td><td>77.53 ± 0.72</td><td>77.23 ± 1.97 61.95 ± 4.09</td><td></td></tr><tr><td>Dynamic</td><td>72.88 ±1.19</td><td>88.88±0.27</td><td>65.35±5.46</td><td>88.70 ± 0.30</td><td>68.80 ± 4.06</td><td>88.40± 0.42</td><td>72.28 ± 1.51</td><td>74.63 ± 1.22</td><td>62.03 ± 4.29</td></tr><tr><td>Cascaded</td><td>66.80 ± 1.98</td><td>86.25±0.38</td><td>67.53 ± 2.05</td><td>86.75±0.50</td><td>66.58 ± 2.24</td><td>85.95±0.79</td><td>71.63 ± 0.73</td><td>76.83 ± 0.93</td><td>64.18 ± 2.94</td></tr><tr><td rowspan="5"></td><td>Non-overlap</td><td>52.08 ± 1.37</td><td>79.93±0.59</td><td>58.63 ± 2.31</td><td>80.03± 0.23</td><td>57.58 ± 3.44</td><td>79.55±0.79</td><td>58.50 ± 2.20</td><td>50.70 ± 0.324</td><td>45.70 ± 0.25</td></tr><tr><td>Overlap-25</td><td>64.18 ± 1.62</td><td>83.23± 0.47</td><td>68.83 ± 1.40</td><td>83.65±0.32</td><td>63.28± 1.98</td><td>83.73±0.48</td><td>64.70 ± 1.81</td><td>54.75 ± 0.44 47.68 ± 0.66</td><td></td></tr><tr><td>RWARE-Advanced Overlap-50</td><td>70.35 ± 1.28</td><td>84.55±0.69</td><td>68.05 ± 4.14</td><td>84.45±0.33</td><td>67.50 ± 2.12</td><td>84.43± 0.56</td><td>72.88 ± 1.05</td><td>58.88 ± 0.77 47.05 ± 0.38</td><td></td></tr><tr><td>Dynamic</td><td>66.60 ± 1.95</td><td>81.30 ± 0.34</td><td>62.55 ± 2.83</td><td>82.50 ± 0.43</td><td>63.23 ± 0.83</td><td>82.20 ± 0.16</td><td>65.90 ± 2.33</td><td>56.30 ± 0.49 47.55 ± 0.40</td><td></td></tr><tr><td>Cascaded</td><td>58.95 ± 1.70</td><td>85.40± 0.42</td><td>65.55±1.76</td><td>85.98± 0.44</td><td>66.03 ± 2.06</td><td>84.90 ± 0.51</td><td>64.18 ± 3.98</td><td></td><td>54.90 ± 1.08 47.03 ± 0.55</td></tr></table>

## 5. Results

## 5.1. Main Results

Fig. 4 shows the training task-completion trajectories across the four environments and five participation patterns. In most settings, the SPL variants reach a higher task-completion regime than their corresponding Vanilla backbones and maintain this diference through 2M environment steps. Although the Vanilla IPPO, MAPPO, and HAPPO trajectories difer, their SPL counterparts often converge to similar final performance, suggesting that the observed improvement is not specific to a single PPO-based actor–critic backbone. Table 1 reports the final results for the 60 primary Vanilla–SPL comparisons. SPL achieves higher observed mean TC in every configuration, with an equal-weight descriptive mean diference of 14.09%. The corresponding mean diferences are 13.93%, 14.02%, and 14.33% for IPPO, MAPPO, and HAPPO, respectively. Among the 57 configurations with paired inference, 56 pointwise 95% confidence intervals have positive lower bounds, and 29 remain significant after Holm correction over the declared family of 60 comparisons at a familywise level of 0.05. Complete statistical results are reported in Appendix D.7, while full TC, FS, return, and FE results are provided in Appendices D.1–D.4.

Participation structure afects the magnitude of the observed improvement. Averaged across environments and backbones, SPL improves task completion by 15.1% under Non-overlap, 13.3% under Overlap-25, 11.7% under Overlap-50, 15.5% under Dynamic, and 14.9% under Cascaded participation. Across the fixed Non-overlap, Overlap-25, and Overlap-50 schedules, the average gain decreases as cross-role concurrency increases. Since the exact temporal constructions difer between MPE and RWARE, we treat this pattern as descriptive and do not interpret it as a controlled efect of overlap. The positive gains under both overlap settings nevertheless show that SPL does not require a strictly sequential handof. Improvements under Dynamic and Cascaded participation further show that the observed benefit extends to stochastic and successively staged participation schedules. We additionally evaluate SPL-MAPPO in eight-agent MPE-Advanced and RWARE-Advanced settings. As shown in Table 2, SPL-MAPPO achieves higher mean task completion than MAPPO in all six matched comparisons, with an average observed gain of 17.1%. The gains range from 3.3% in MPE-Advanced Cascaded to more than 30% in the Non-overlap and Dynamic RWARE-Advanced settings. Non-overlap and Dynamic use 2M environment steps, whereas Cascaded uses 4M to accommodate its longer participation horizon; diferences across participation patterns should therefore not be interpreted as controlled compar isons of training budget. SPL introduces no additional models during execution. Its training-time computational cost, however, varies across environments and team sizes: the overhead is small in several four-agent settings but becomes more pronounced in some larger-scale configurations. Detailed wall-clock measurements are reported in Appendix C.

Table 2: Task completion (%) for eight-agent scaling and physics-based UAV–UGV evaluations using MAPPO.
<table><tr><td rowspan="2">Participation</td><td colspan="2">MPE-Advanced (8)</td><td colspan="2">RWARE-Advanced (8)</td></tr><tr><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td></tr><tr><td>Non-overlap Dynamic</td><td> $6 4 . 2 1 \pm 0 . 6 1$   $6 8 . 6 0 \pm 2 . 6 4$ </td><td> $7 7 . 7 7 \pm 0 . 3 0$   $7 7 . 7 5 \pm 0 . 2 0$ </td><td> $4 1 . 1 0 \pm 2 . 6 6$   $4 4 . 8 1 \pm 1 . 5 7$ </td><td> ${ \pm } \mathbf { 1 . } 3 8 \pm \mathbf { 1 . } 2 4$   ${ \bf 7 6 . 1 5 \pm 0 . 8 8 }$ </td></tr><tr><td>Cascaded</td><td> $7 1 . 7 9 \pm 0 . 2 7$ </td><td> ${ \bf 7 5 . 1 3 \pm 0 . 1 4 }$ </td><td> $1 9 . 7 3 \pm 0 . 7 0$ </td><td> ${ \bf 3 1 . 7 5 \pm 1 . 0 6 }$ </td></tr><tr><td></td><td> $1 \mathrm { U A V } + 1 \mathrm { U G V }$ </td><td></td><td> $2 \mathrm { U A V s } + 2 \mathrm { U G V s }$ </td><td></td></tr><tr><td>Participation</td><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td></tr><tr><td>Non-overlap</td><td> $3 1 . 5 0 \pm 5 . 9 7$ </td><td> ${ \bf 6 1 . 5 8 \pm 0 . 6 7 }$ </td><td> $5 5 . 7 9 \pm 3 . 2 3$ </td><td> ${ \bf 7 6 . 3 8 \pm 1 . 5 2 }$ </td></tr><tr><td>Dynamic</td><td> $3 6 . 6 7 \pm 0 . 7 1$ </td><td> ${ \pm } 9 . 5 8 \pm { \bf 1 . 7 9 }$ </td><td> $5 8 . 4 6 \pm 3 . 4 0$ </td><td> ${ \bf 7 9 . 2 5 \pm 1 . 3 0 }$ </td></tr></table>

## 5.2. Physics-Based UAV–UGV Evaluation

We further evaluate SPL in a physics-based UAV–UGV setting in Isaac Lab, where earlier UAVs navigate to sensing locations, and later UGVs navigate to and execute downstream tasks. This introduces embodied motion and obstacle constraints on both evidence acquisition and downstream execution. As shown in Table 2, SPL-MAPPO achieves higher mean task completion than MAPPO in all four team-size– participation settings. The gains range from 20.6% to 30.1% across both team sizes and occur under both Non-overlap and Dynamic participation. Attempt-level diagnostics show more resolved executions and successful completions with SPL-MAPPO, together with a lower observed failure rate per resolved attempt, while map and published evidence coverage vary with team size. These results provide complementary evidence that the observed SPL improvement persists when evidence acquisition and downstream task execution are coupled to embodied motion. Execution, coverage, and safety diagnostics are reported in Appendix D.5.

Qualitative case study. Figure 5 compares MAPPO and SPL-MAPPO on the same deterministic fixed-handof rollout in the one-UAV–one-UGV Isaac setting. The hidden ground truth, shown only for interpretation, contains two decoy tasks (Tasks 1 and 3) and two serviceable tasks (Tasks 2 and 4). All candidate locations are visible to the agents, while task serviceability must be inferred from acquired evidence. The UAV acts during the first 200 steps of the 600-step episode, after which the UGV acts using the persistent shared board. In the trajectory panels, triangles and squares mark UAV and UGV starting positions, circles mark trajectory endpoints, plus/minus symbols denote published task cues, and check/cross symbols denote UGV service outcomes; trajectory color darkens with time. At handof, both methods have mapped 48 of 144 occupancy cells, but their task-specific evidence difers substantially. MAPPO publishes no task-quality cues, after which its UGV attempts decoy Task 3 at step 453 and completes serviceable Task 2 at step 577, leaving Task 4 unserved. SPL-MAPPO instead acquires cues for Tasks 2, 3, and 4, which are published to the shared board after the five-step delay: positive for Tasks 2 and 4 and negative for Task 3. Its UGV then completes Task 4 at step 438 and Task 2 at step 484 without attempting a decoy. This example makes the cross-stage information pathway explicit: similar spatial coverage can coexist with diferent task-specific evidence and, consequently, diferent downstream decisions.

![](images/e74987309ec3f3c924422243b49b82892f308e8d234770229d9f4ba6818c831d.jpg)

![](images/a88ef47839c650c9ae3344c2168cd1f757aed6c37bc0082efa46186772ebf8cd.jpg)

![](images/e12b94e91c8a4643dd7a7db7b4b6c1bd234442f9f79176aef4d486ac4415f49b.jpg)  
Figure 5: Isaac Lab case study comparing Vanilla and SPL based on MAPPO.

## 5.3. Ablation Study

We evaluate SPL-MAPPO under Non-overlap participation to examine the contribution of its two training components, the role of inherited evidence in receiver learning, and the dependence on structured training information. Acquisition-only and Receiver-only retain only the corresponding SPL component. Evidence-redacted removes inherited task evidence consistently from both outcome-model inputs and receiver-target construction, whereas SKD-MAPPO retains full evidence during outcome-model training but masks predecessor-generated evidence when constructing receiver targets. Outcome-agnostic fixes candidate success probabilities to 0.5, while Permuted assigns predicted utilities to incorrect legal actions while preserving the receiver-update budget. Table 3 shows that full SPL achieves the highest observed mean task completion in all four environments. Relative to full SPL, Acquisition-only and Receiver-only reduce mean task completion by 10.7% and 7.7% on average, respectively. Neither component alone reproduces the complete method, supporting the complementary roles of evidence preparation and downstream evidence use.

Table 3: SPL ablations under Non-overlap.
<table><tr><td>Variant</td><td>MPE-Base</td><td> $\mathrm { M P E - A d v . }$ </td><td></td><td>RWARE-Base RWARE-Adv.</td></tr><tr><td>Full SPL</td><td> ${ \bf 7 8 . 7 3 \pm 0 . 4 3 }$ </td><td> ${ \bf 7 8 . 2 3 \pm 0 . 1 6 }$ </td><td> ${ \bf 8 6 . 7 0 \pm 0 . 4 9 }$ </td><td> ${ \bf 8 0 . 0 3 \pm 0 . 2 3 }$ </td></tr><tr><td>SKD-MAPPO</td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $7 0 . 3 8 \pm 0 . 0 0$ </td><td> $7 0 . 7 5 \pm 0 . 0 0$ </td><td> $6 7 . 6 3 \pm 0 . 2 3$ </td></tr><tr><td>Acquisition-only</td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $6 9 . 8 8 \pm 0 . 5 0$ </td><td> $7 7 . 8 0 \pm 2 . 3 7$ </td><td> $6 2 . 8 3 \pm 1 . 9 8$ </td></tr><tr><td>Receiver-only</td><td> $7 3 . 4 5 \pm 1 . 2 9$ </td><td> $7 2 . 2 8 \pm 0 . 9 4$ </td><td> $7 4 . 4 0 \pm 1 . 9 4$ </td><td> $7 2 . 9 0 \pm 1 . 0 3$ </td></tr><tr><td>Evidence-redacted</td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $7 0 . 3 8 \pm 0 . 0 0$ </td><td> $7 0 . 7 5 \pm 0 . 0 0$ </td><td> $6 9 . 1 3 \pm 0 . 2 1$ </td></tr><tr><td>Outcome-agnostic</td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $7 0 . 3 8 \pm 0 . 0 0$ </td><td> $6 2 . 3 3 \pm 2 . 2 7$ </td><td> $4 8 . 4 8 \pm 0 . 8 0$ </td></tr><tr><td>Permuted</td><td> $3 7 . 1 3 \pm 0 . 3 7$ </td><td> $3 7 . 0 5 \pm 0 . 8 7$ </td><td> $2 6 . 8 0 \pm 2 . 7 6$ </td><td> $2 8 . 9 5 \pm 0 . 8 4$ </td></tr></table>

SKD-MAPPO also remains below full SPL in every environment, with an average gap of 11.1%. Thus, matching SPL’s training-side structured information and auxiliary machinery without evidence conditioned receiver supervision does not reproduce the full SPL performance. We further test sensitivity to misspecified validity priors while keeping the true environment distribution unchanged. Even with fully reversed priors $( \lambda = 1 )$ , SPL-MAPPO retains 76.73% and 77.45% TC in MPE-Advanced and RWARE-Advanced, compared with nominal values of 78.23% and 80.03%, respectively. Full success is more sensitive, particularly in RWARE-Advanced; complete results are reported in Appendix D.6. These results show that SPL retains substantial task-completion performance under validity-prior misspecification in the tested settings. Other forms of structured-model error remain outside the scope of this experiment. Receiver-target ablations further show that the information used to construct the target matters. Evidence-redacted, Outcome-agnostic, and Permuted reduce mean task completion by 10.7%, 18.0%, and 48.4% on average, respectively. The strong degradation under Permuted, despite preserving the receiver-update budget, shows that the correspondence between candidate actions and their predicted utilities is important. Preserving the auxiliary update budget alone is insuficient to recover the full SPL performance.

## 6. Conclusion

We studied cooperative MARL under staggered participation and introduced SPL, which combines prospective evidence acquisition with downstream evidence use. SPL achieves higher observed mean task completion across multiple policy-based backbones, participation patterns, larger teams, and physics-based UAV–UGV simulation. These results support explicitly addressing both information acquisition and downstream utilization when agents contribute over staggered participation windows.

## References

[1] Friedrich M Rockenbauer, Jaeyoung Lim, Marcus G Müller, Roland Siegwart, and Lukas Schmid. Traversing mars: Cooperative informative path planning to eficiently navigate unknown scenes. IEEE Robotics and Automation Letters, 2024.

[2] Takahiro Sasaki, Kyohei Otsu, Rohan Thakker, Sofie Haesaert, and Ali-akbar Agha-mohammadi. Where to map? iterative rover-copter path planning for mars exploration. IEEE Robotics and Automation Letters, 2020.

[3] Shiyong Zhang, Xuebo Zhang, Tianyi Li, Jing Yuan, and Yongchun Fang. Fast active aerial exploration for traversable path finding of ground robots in unknown environments. IEEE Transactions on Instrumentation and Measurement, 2022.

[4] Chao Yu, Xinyi Yang, Jiaxuan Gao, Jiayu Chen, Yunfei Li, Jijia Liu, Yunfei Xiang, Ruixin Huang, Huazhong Yang, Yi Wu, and Yu Wang. Asynchronous multi-agent reinforcement learning for eficient real-time multi-robot cooperative exploration. In Proceedings of the 2023 International Conference on Autonomous Agents and Multiagent Systems, 2023.

[5] Yifei Min, Jiafan He, Tianhao Wang, and Quanquan Gu. Cooperative multi-agent reinforcement learning: Asynchronous communication and linear function approximation. In International Conference on Machine Learning. PMLR, 2023.

[6] Sydney Dolan, Siddharth Nayak, Jasmine Jerry Aloor, and Hamsa Balakrishnan. Asynchronous cooperative multi-agent reinforcement learning with limited communication. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, 2025.

[7] Bo Liu, Qiang Liu, Peter Stone, Animesh Garg, Yuke Zhu, and Anima Anandkumar. Coach-player multi agent reinforcement learning for dynamic team composition. In International Conference on Machine Learning. PMLR, 2021.

[8] Arrasy Rahman, Ignacio Carlucho, Niklas Höpner, and Stefano V Albrecht. A general learning framework for open ad hoc teamwork using graph-based policy learning. Journal of Machine Learning Research, 2023.

[9] Caroline Wang. N-agent ad hoc teamwork. In Proceedings of the Thirty-Third International Joint Conference on Artificial Intelligence, 2024.

[10] Yongheng Liang, Hejun Wu, Haitao Wang, and Hao Cai. Asynchronous credit assignment for multi-agent reinforcement learning. In Proceedings of the Thirty-Fourth International Joint Conference on Artificial Intelligence, IJCAI ’25, 2025.

[11] Milad Farjadnasab and Shahin Sirouspour. Cooperative and asynchronous transformer-based mission planning for heterogeneous teams of mobile robots. Robotics and Autonomous Systems, page 105131, 2025.

[12] Siyi Hu, Mohamad A Hady, Jianglin Qiao, Jimmy Cao, Mahardhika Pratama, and Ryszard Kowalczyk. Adaptability in multi-agent reinforcement learning: A framework and unified review. arXiv preprint arXiv:2507.10142, 2025.

[13] Jordan Erskine and Christopher Lehnert. Developing cooperative policies for multi-stage reinforcement learning tasks. IEEE Robotics and Automation Letters, 2022.

[14] Changxi Zhu, Mehdi Dastani, and Shihan Wang. A survey of multi-agent deep reinforcement learning with communication. Autonomous Agents and Multi-Agent Systems, 38(1):4, 2024.

[15] Shengchao Hu, Li Shen, Ya Zhang, and Dacheng Tao. Learning multi-agent communication from graph modeling perspective. In International Conference on Learning Representations, 2024.

[16] Xinran Li and Jun Zhang. Context-aware communication for multi-agent reinforcement learning. Proceedings of the 2024 International Conference on Autonomous Agents and Multiagent Systems, 2024.

[17] Ming Yang, Kaiyan Zhao, Yiming Wang, Renzhi Dong, Yali Du, Furui Liu, Mingliang Zhou, and Leong Hou U. Team-wise efective communication in multi-agent reinforcement learning. Autonomous Agents and Multi-Agent Systems, 2024.

[18] Yi-Yu Lin, Jiahao Zhang, and Xiao-Jun Zeng. Why and whom to communicate? a dual-objective, cost-benefit framework for multi-agent communication. In Proc. of the 25th International Conference on Autonomous Agents and Multiagent Systems, 2026.

[19] Enguang Yao, Qidong Liu, Jialu Sun, Kaibo Huang, and Mingliang Xu. Direct: Decentralized intention and latent rule emergence for multi-agent cooperation under intermittent communication. In Proceedings of the Thirty-Fifth International Joint Conference on Artificial Intelligence, IJCAI-26, pages 437–445. International Joint Conferences on Artificial Intelligence Organization, 2026.

[20] Xing Xu, Rongpeng Li, Zhifeng Zhao, and Honggang Zhang. Stigmergic independent reinforcement learning for multiagent collaboration. IEEE transactions on neural networks and learning systems, 2021.

[21] Jianglin Qiao, Zehong Cao, Siyi Hu, Mingjun Fan, Mahardhika Pratama, and Ryszard Kowalczyk. Multi agent RL with invisible collaborators: Marginal advantage estimation for indirect cooperation. In Proceedings of the 42nd Conference on Uncertainty in Artificial Intelligence. PMLR, 2026.

[22] Omayma Mahjoub, Sasha Abramowitz, Ruan John de Kock, Wiem Khlifi, Simon Verster Du Toit, Jemma Daniel, Louay Ben Nessir, Louise Beyers, Juan Claude Formanek, Liam Clark, and Arnu Pretorius. Sable: a performant, eficient and scalable sequence model for MARL. In Forty-second International Conference on Machine Learning, 2025.

[23] Andreas Kontogiannis, Konstantinos Papathanasiou, Yi Shen, Giorgos Stamou, Michael M. Zavlanos, and George Vouros. Enhancing cooperative multi-agent reinforcement learning with state modelling and adversarial exploration. In Forty-second International Conference on Machine Learning, 2025.

[24] Sehyeok Kang, Yongsik Lee, Gahee Kim, Song Chong, and Se-Young Yun. MA<sup>2</sup>E: Addressing partial observability in multi-agent reinforcement learning with masked auto-encoder. In The Thirteenth International Conference on Learning Representations, 2025.

[25] Yuhang Pei, Tao Ren, Yuxiang Zhang, Zhipeng Sun, and Matys Champeyrol. Policy distillation for eficient decentralized execution in multi-agent reinforcement learning. Neurocomputing, 626:129617, 2025.

[26] Zimo Zhai, Manjie Xu, and Wei Liang. Distilling task-level coordination policies for generalizable multi-agent cooperation. In Forty-third International Conference on Machine Learning, 2026.

[27] Nannan Cao, Kian Hsiang Low, and John M. Dolan. Multi-robot informative path planning for active sensing of environmental phenomena: a tale of two algorithms. In Proceedings of the 2013 International Conference on Autonomous Agents and Multi-Agent Systems, 2013.

[28] Mikko Lauri, Joni Pajarinen, and Jan Peters. Information gathering in decentralized pomdps by policy graph improvement. In Proceedings of the 18th International Conference on Autonomous Agents and MultiAgent Systems, 2019.

[29] Wei-Chen Liao, Ti-Rong Wu, and I-Chen Wu. Dynamic sight range selection in multi-agent reinforcement learning. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, 2025.

[30] Yugu Li, Zehong Cao, Jianglin Qiao, and Siyi Hu. Nucleolus credit assignment for efective coalitions in multi-agent reinforcement learning. In Proceedings of the 24th International Conference on Autonomous Agents and Multiagent Systems, AAMAS ’25, page 1318–1326. International Foundation for Autonomous Agents and Multiagent Systems, 2025.

[31] Frans A Oliehoek and Christopher Amato. A concise introduction to decentralized POMDPs. Springer, 2016.

[32] Ryan Lowe, Yi Wu, Aviv Tamar, Jean Harb, Pieter Abbeel, and Igor Mordatch. Multi-agent actor-critic for mixed cooperative-competitive environments. Advances in neural information processing systems, 2017.

[33] Georgios Papoudakis, Filippos Christianos, Lukas Schäfer, and Stefano V. Albrecht. Benchmarking multiagent deep reinforcement learning algorithms in cooperative tasks. In Proceedings of the Neural Information Processing Systems Track on Datasets and Benchmarks (NeurIPS), 2021.

[34] Mayank Mittal, Pascal Roth, James Tigue, Antoine Richard, Octi Zhang, Peter Du, Antonio Serrano-Munoz,

Xinjie Yao, René Zurbrügg, Nikita Rudin, et al. Isaac lab: A gpu-accelerated simulation framework for multi-modal robot learning. arXiv preprint arXiv:2511.04831, 2025.

[35] Chao Yu, Akash Velu, Eugene Vinitsky, Jiaxuan Gao, Yu Wang, Alexandre Bayen, and Yi Wu. The surprising efectiveness of ppo in cooperative multi-agent games. Advances in neural information processing systems, 2022.

[36] Jakub Grudzien Kuba, Ruiqing Chen, Muning Wen, Ying Wen, Fanglei Sun, Jun Wang, and Yaodong Yang. Trust region policy optimisation in multi-agent reinforcement learning. In International Conference on Learning Representations, 2022.

[37] Christian Schroeder De Witt, Tarun Gupta, Denys Makoviichuk, Viktor Makoviychuk, Philip HS Torr, Mingfei Sun, and Shimon Whiteson. Is independent learning all you need in the starcraft multi-agent challenge? arXiv preprint arXiv:2011.09533, 2020.

[38] Yifan Zhong, Jakub Grudzien Kuba, Xidong Feng, Siyi Hu, Jiaming Ji, and Yaodong Yang. Heterogeneousagent reinforcement learning. Journal of Machine Learning Research, 2024.

## A. Proofs and Additional Analysis for SPL

This section provides full proofs for Proposition 1, Proposition 2, and Theorem 1 in the main paper, together with supporting lemmas and additional analysis of the SPL auxiliary objectives. The acquisition result applies a standard value-of-information argument to the top-k completion surrogate used by SPL. For the receiver analysis, we additionally relate the distillation term directly to the training objective and give a conditional connection between outcome-model error and standardized-score error. Section A.6 summarizes the scope of the analysis.

## A.1. Notation and Assumptions

Agent i denotes an earlier participant and agent j a future collaborator. The actor-visible acquisition context of i is denoted by $c _ { t : } ^ { i }$ , while $\ v { x } _ { t } ^ { j }$ denotes the actor-visible receiver context of j. The legal receiver action set is

$$
\mathcal { A } _ { t } ^ { j } = \{ u \in \mathcal { U } _ { j } : M _ { t } ^ { j } ( u ) = 1 \} .
$$

For an attributable receiver decision, let $Y \in \{ 0 , 1 \}$ denote its downstream outcome and let $C \in \{ 0 , 1 \}$ indicate whether that outcome is retained in $\mathcal { D } _ { \mathrm { o u t } }$ . The retained-outcome probability is

$$
p _ { * } ^ { j } ( u \mid x ) = \operatorname* { P r } ( Y = 1 \mid X = x , U = u , J = j , C = 1 ) .\tag{A.1}
$$

The corresponding local utility is

$$
U _ { * } ^ { j } ( u \mid x ) = p _ { * } ^ { j } ( u \mid x ) c _ { j } ^ { + } ( u ) + \left[ 1 - p _ { * } ^ { j } ( u \mid x ) \right] c _ { j } ^ { - } ( u ) ,\tag{A.2}
$$

where $c _ { j } ^ { + } ( u )$ and $c _ { j } ^ { - } \left( u \right)$ are the task-level utilities associated with successful and unsuccessful execution, respectively. Illegal actions are excluded from the distributions below.

Assumption 1 (Attribution and censoring). Whenever $C = 1$ , the recorded outcome is correctly attributed to the selected receiver action. Receiver-side probability and utility results are conditional on $C = 1$ . They admit an unconditional interpretation only when

$$
Y \perp C \mid X , U , J .
$$

Assumption 2 (Coverage). Every legal context–candidate pair on which a receiver guarantee is invoked is selected with positive probability under the retained rollout distribution.

## A.2. Proof of Proposition 1: Prospective Acquisition Target

Let $\mathcal { M } _ { \mathrm { a c q } }$ denote the fixed acquisition model used to construct SPL supervision. Probabilities and expectations in this subsection are taken with respect to this model. Consider binary task-validity variables $W _ { 1 } , \ldots , W _ { n }$ . Given acquisition context $c _ { t } ^ { i } .$ , let

$$
b _ { r } = \operatorname* { P r } _ { { \mathcal { M } } _ { \mathrm { a c q } } } ( W _ { r } = 1 \mid c _ { t } ^ { i } ) , \qquad b = ( b _ { 1 } , \ldots , b _ { n } ) .
$$

If a future collaborator can execute at most k tasks, define

$$
V _ { k } ( b ) = \operatorname* { m a x } _ { S \subseteq \{ 1 , \dots , n \} : | S | \leq k } \sum _ { r \in S } b _ { r } .\tag{A.3}
$$

For a legal acquisition action $u ,$ let $Z _ { u }$ denote its observation and $b ^ { Z _ { u } }$ the posterior marginals produced by $\mathcal { M } _ { \mathrm { a c q } } .$ Its model-relative prospective gain is

$$
\Delta _ { t } ^ { i } ( u ) = \mathbb { E } _ { \mathcal { M } _ { \mathrm { a c q } } } \left[ V _ { k } ( b ^ { Z _ { u } } ) \mid c _ { t } ^ { i } , u \right] - V _ { k } ( b ) .\tag{A.4}
$$

SPL combines this gain with acquisition distance through

$$
s _ { t } ^ { i } ( u ) = \log \Bigl ( \operatorname* { m a x } \{ \Delta _ { t } ^ { i } ( u ) , \epsilon _ { \mathrm { a c q } } \} \Bigr ) - \frac { d _ { t } ^ { i } ( u ) } { \tau _ { d } } .\tag{A.5}
$$

Let

$$
\mathcal { P } _ { t } ^ { i } = \Big \{ u : M _ { t } ^ { i } ( u ) = 1 , u \mathrm { ~ i s ~ t a s k { \mathrm { - } } d i r e c t e d } , \Delta _ { t } ^ { i } ( u ) > 0 \Big \}
$$

denote the legal positive-gain task-directed actions.

For completeness, the task-directed component of the SPL acquisition target is

$$
\widetilde { q } _ { \mathrm { a c q } } ^ { i } ( u ) = \frac { \exp ( s _ { t } ^ { i } ( u ) ) } { \sum _ { v \in \mathscr { P } _ { t } ^ { i } } \exp ( s _ { t } ^ { i } ( v ) ) } .\tag{A.6}
$$

Equivalently,

$$
\widetilde { q } _ { \mathrm { a c q } } ^ { i } = \arg \operatorname* { m a x } _ { \boldsymbol { q } \in \Delta ( \mathcal { P } _ { t } ^ { i } ) } \left\{ \sum _ { \boldsymbol { u } \in \mathcal { P } _ { t } ^ { i } } q ( \boldsymbol { u } ) s _ { t } ^ { i } ( \boldsymbol { u } ) + H ( \boldsymbol { q } ) \right\} ,\tag{A.7}
$$

where $\begin{array} { r } { H ( q ) = - \sum _ { u } q ( u ) } \end{array}$ $q ( u )$

Proof. For every feasible set S with $| S | \leq k ,$ , let

$$
L _ { S } ( b ) = \sum _ { r \in S } b _ { r } .
$$

Since

$$
V _ { k } ( b ) = \operatorname* { m a x } _ { S : | S | \leq k } L _ { S } ( b ) ,
$$

$V _ { k }$ is convex. Bayesian consistency under $\mathcal { M } _ { \mathrm { a c q } }$ gives

$$
\mathbb { E } _ { \mathcal { M } _ { \mathrm { a c q } } } \left[ b ^ { Z _ { u } } \mid c _ { t } ^ { i } , u \right] = b .
$$

Jensen’s inequality therefore gives

$$
\mathbb { E } _ { \mathcal { M } _ { \mathrm { a c q } } } \left[ V _ { k } ( b ^ { Z _ { u } } ) \mid c _ { t } ^ { i } , u \right] \ge V _ { k } \left( \mathbb { E } _ { \mathcal { M } _ { \mathrm { a c q } } } [ b ^ { Z _ { u } } \mid c _ { t } ^ { i } , u ] \right) = V _ { k } ( b ) ,
$$

which proves

$$
\Delta _ { t } ^ { i } ( u ) \geq 0 .
$$

This is the standard nonnegative value-of-information argument applied to the top-k completion surrogate.

For the Gibbs characterization, consider

$$
F ( q ) = \sum _ { u \in \mathcal { P } _ { t } ^ { i } } q ( u ) s _ { t } ^ { i } ( u ) + H ( q )
$$

over the probability simplex. Introducing a Lagrange multiplier for $\begin{array} { r } { \sum _ { u } q ( u ) = 1 } \end{array}$ gives the stationary condition

$$
s _ { t } ^ { i } ( u ) - \log q ( u ) - 1 + \lambda = 0 ,
$$

hence

$$
q ( u ) \propto \exp ( s _ { t } ^ { i } ( u ) ) .
$$

Normalization yields Eq. (A.6). Since entropy is strictly concave and the score term is linear, this solution is the unique maximizer of Eq. (A.7). □

The complete acquisition target assigns the prescribed fixed mass to the scan action and distributes the remaining mass according to $\widetilde { q } _ { \mathrm { a c q } } ^ { i }$ . If $\mathcal { P } _ { t } ^ { i }$ is empty, the scan action receives unit target mass. The implementation uses the small positive-gain threshold and numerical constants reported in Sec. B.4.

The nonnegativity result is model-relative. Bayesian consistency within $\mathcal { M } _ { \mathrm { a c q } }$ does not require its assumed priors to equal the true environment priors. Prior misspecification can therefore preserve nonnegative model-relative gains while changing how those gains relate to performance in the true environment; this sensitivity is evaluated in Sec. D.6.

## A.3. Proof of Proposition 2: Retained-Outcome Calibration

For any measurable predictor $p ^ { j } ( u \mid x ) \in [ 0 , 1 ]$ , define its population binary-cross-entropy risk over retained outcomes as

$$
\mathcal { L } _ { \mathrm { o u t } } ( p ) = \mathbb { E } \left[ \ell _ { \mathrm { B C E } } \left( Y , p ^ { J } ( U \mid X ) \right) \mid C = 1 \right] .\tag{A.8}
$$

The excess population risk satisfies

$$
\begin{array} { r } { \mathcal { L } _ { \mathrm { o u t } } ( p ) - \mathcal { L } _ { \mathrm { o u t } } ( p _ { * } ) = \mathbb { E } \left[ \mathrm { K L } \left( \mathrm { B e r } ( p _ { * } ^ { J } ( U \mid X ) ) \| \mathrm { B e r } ( p ^ { J } ( U \mid X ) ) \right) \mid C = 1 \right] . } \end{array}\tag{A.9}
$$

Let

$$
C _ { \Delta } = \operatorname* { s u p } _ { u , j } | c _ { j } ^ { + } ( u ) - c _ { j } ^ { - } ( u ) | < \infty .
$$

The induced local utility then satisfies

$$
\mathbb { E } \left[ \left( U _ { p } ^ { J } ( U \mid X ) - U _ { * } ^ { J } ( U \mid X ) \right) ^ { 2 } \mid C = 1 \right] \leq \frac { C _ { \Delta } ^ { 2 } } { 2 } \left[ \mathcal { L } _ { \mathrm { o u t } } ( p ) - \mathcal { L } _ { \mathrm { o u t } } ( p _ { * } ) \right] .\tag{A.10}
$$

Proof. Fix a covered triple $( x , u , j )$ and abbreviate

$$
p _ { * } = p _ { * } ^ { j } ( u \mid x ) , \qquad p = p ^ { j } ( u \mid x ) .
$$

The conditional binary-cross-entropy risk is

$$
r ( p ; p _ { * } ) = - p _ { * } \log p - ( 1 - p _ { * } ) \log ( 1 - p ) .
$$

Its excess risk is

$$
r ( p ; p _ { * } ) - r ( p _ { * } ; p _ { * } ) = \mathrm { K L } \left( \mathrm { B e r } ( p _ { * } ) \| \mathrm { B e r } ( p ) \right) ,
$$

which is non-negative and vanishes only when ${ p } = { p } _ { * }$ . Taking expectations over $( X , U , J ) \mid C = 1$ gives Eq. (A.9).

From Eq. (A.2),

$$
U _ { p } ^ { j } ( u \mid x ) - U _ { * } ^ { j } ( u \mid x ) = \left[ p ^ { j } ( u \mid x ) - p _ { * } ^ { j } ( u \mid x ) \right] \left[ c _ { j } ^ { + } ( u ) - c _ { j } ^ { - } ( u ) \right] .
$$

Hence

$$
\left| U _ { p } ^ { j } - U _ { * } ^ { j } \right| ^ { 2 } \leq C _ { \Delta } ^ { 2 } | p - p _ { * } | ^ { 2 } .
$$

Pinsker’s inequality gives

$$
| p - p _ { * } | ^ { 2 } \leq { \frac { 1 } { 2 } } \mathrm { K L } \left( \mathrm { B e r } ( p _ { * } ) \| \mathrm { B e r } ( p ) \right) .
$$

Combining the two inequalities and taking expectations proves Eq. (A.10).

The result is conditional on retained attributable outcomes. Under the additional condition

$$
Y \perp C \mid X , U , J ,
$$

the retained-outcome probability also equals the corresponding unconditional success probability.

## A.4. Receiver Target and Supporting Analysis

Fix an eligible receiver decision $( j , t )$ with legal action set

$$
\begin{array} { r } { \mathcal { A } = \mathcal { A } _ { t } ^ { j } , \qquad K = | \mathcal { A } | . } \end{array}
$$

For any local utility vector U over ${ \mathcal { A } } ,$ let $\mu _ { U }$ and $\sigma _ { U } ^ { 2 }$ denote its mean and variance over legal actions, and define the standardized score

$$
z _ { U } ( u ) = \frac { U ( u ) - \mu _ { U } } { \sqrt { \sigma _ { U } ^ { 2 } + \epsilon _ { z } } } , \qquad \epsilon _ { z } > 0 .\tag{A.11}
$$

For a score vector $z ,$ the SPL receiver target is

$$
q _ { \mathrm { S P L } } ^ { j } ( u ) = \frac { \exp ( \beta z ( u ) ) } { \sum _ { v \in \mathcal { A } } \exp ( \beta z ( v ) ) } , \qquad u \in \mathcal { A } .\tag{A.12}
$$

Lemma 1 (Soft receiver target). For $\beta > 0 ,$ the distribution in $E q .$ . (A.12) uniquely maximizes

$$
\sum _ { u \in \mathcal { A } } q ( u ) z ( u ) + \frac { 1 } { \beta } H ( q )\tag{A.13}
$$

over $q \in \Delta ( \mathcal { A } )$ , and satisfies

$$
\operatorname* { m a x } _ { u \in \mathcal { A } } z ( u ) - \mathbb { E } _ { u \sim q _ { \mathrm { S P L } } ^ { j } } [ z ( u ) ] \leq \frac { \log K } { \beta } .\tag{A.14}
$$

Proof. Introducing a Lagrange multiplier for the simplex constraint gives the stationary condition

$$
z ( u ) - \frac { 1 } { \beta } \left[ \log q ( u ) + 1 \right] + \lambda = 0 ,
$$

hence

$$
q ( u ) \propto \exp ( \beta z ( u ) ) .
$$

Since the entropy term is strictly concave, this solution is unique and equals Eq. (A.12).

$$
u ^ { * } \in \arg \operatorname* { m a x } _ { u \in \mathcal { A } } z ( u ) .
$$

Optimality of $q _ { \mathrm { S P L } } ^ { j }$ relative to the point mass on $u ^ { * }$ gives

$$
\mathbb { E } _ { q _ { \mathrm { { S P L } } } ^ { j } } [ z ] + \frac { 1 } { \beta } H ( q _ { \mathrm { { S P L } } } ^ { j } ) \geq z ( u ^ { * } ) .
$$

Using

$$
H ( q _ { \mathrm { S P L } } ^ { j } ) \leq \log K
$$

yields Eq. (A.14).

Receiver distillation. For a fixed SPL target $q _ { \mathrm { S P L } } ^ { j }$ and receiver policy $\boldsymbol { \pi } _ { t } ^ { j } ,$

$$
\mathrm { C E } ( q _ { \mathrm { S P L } } ^ { j } , \pi _ { t } ^ { j } ) = H ( q _ { \mathrm { S P L } } ^ { j } ) + \mathrm { K L } ( q _ { \mathrm { S P L } } ^ { j } | | \pi _ { t } ^ { j } ) .\tag{A.15}
$$

Thus,

$$
\eta = \mathrm { K L } ( q _ { \mathrm { S P L } } ^ { j } | | \pi _ { t } ^ { j } ) = \mathrm { C E } ( q _ { \mathrm { S P L } } ^ { j } , \pi _ { t } ^ { j } ) - H ( q _ { \mathrm { S P L } } ^ { j } ) ,\tag{A.16}
$$

so the local distillation term is directly computable from the receiver target and actor distributions.   
Averaging Eq. (A.16) over receiver decisions gives an empirical diagnostic of target distillation.

Connection between outcome error and score error. The calibration result in Eq. (A.10) controls utility error on selected actions, whereas receiver regret depends on scores over the full legal action set. A quantitative connection follows under stronger coverage. Let

$$
\rho ^ { j } ( u \mid x ) = \operatorname* { P r } ( U = u \mid X = x , J = j , C = 1 ) .
$$

Suppose that, on the retained context distribution considered below, there exists $\alpha > 0$ such that

$$
\rho ^ { j } ( u \mid x ) \geq \alpha
$$

for every legal candidate whose utility is supplied by the outcome model. Any remaining legal actions with fixed utilities are assumed identical in the predicted and reference utility vectors.

Lemma 2 (Mean score-error control). Let $p _ { \bar { \psi } }$ denote the target outcome predictor and let

$$
\delta ( X , J ) = \| z _ { \bar { \psi } } - z _ { * } \| _ { \infty } .
$$

Under the uniform coverage condition above,

$$
\mathbb { E } \left[ \delta ( X , J ) ^ { 2 } \mid C = 1 \right] \le \frac { C _ { \Delta } ^ { 2 } } { 2 \alpha \epsilon _ { z } } \left[ \mathcal { L } _ { \mathrm { o u t } } ( p _ { \bar { \psi } } ) - \mathcal { L } _ { \mathrm { o u t } } ( p _ { * } ) \right] \le \frac { C _ { \Delta } ^ { 2 } } { 2 \alpha \epsilon _ { z } } \mathcal { L } _ { \mathrm { o u t } } ( p _ { \bar { \psi } } ) .\tag{A.17}
$$

Proof. For a fixed legal action set, the stabilized standardization map in Eq. (A.11) is $1 / \sqrt { \epsilon _ { z } } \mathrm { - I }$ Lipschitz from the utility-vector $\ell _ { 2 }$ norm to the score-vector $\ell _ { \infty }$ norm. Hence

$$
\delta ( X , J ) ^ { 2 } \leq \frac { \| U _ { \bar { \psi } } - U _ { * } \| _ { 2 } ^ { 2 } } { \epsilon _ { z } } .
$$

The uniform coverage condition gives

$$
\sum _ { u \in \mathcal { A } } \left( U _ { \tilde { \psi } } ^ { j } ( u \mid x ) - U _ { * } ^ { j } ( u \mid x ) \right) ^ { 2 } \leq \frac { 1 } { \alpha } \mathbb { E } \left[ \left( U _ { \tilde { \psi } } ^ { J } ( U \mid X ) - U _ { * } ^ { J } ( U \mid X ) \right) ^ { 2 } \mid X = x , J = j , C = 1 \right] .
$$

Taking expectations over retained contexts and applying Eq. (A.10) proves the first inequality in Eq. (A.17). The second follows from

$$
\mathcal { L } _ { \mathrm { o u t } } ( p _ { * } ) \geq 0 .
$$

Equation (A.17) controls mean squared score error under the retained context distribution; it is not a uniform per-context bound. For fixed <sub>α</sub> and $\epsilon _ { z } ,$ vanishing population excess outcome risk implies vanishing mean squared score error. Held-out binary-cross-entropy provides an empirical estimate of population outcome risk, while finite-sample generalization remains separate from this population statement.

## A.5. Proof of Theorem 1: Local Receiver Score Regret

Let $z _ { * }$ denote the standardized score induced by the retained-outcome local utility $U _ { * } ^ { j }$ , and let $z _ { \bar { \psi } }$ denote the score induced by the target outcome model. Define

$$
\delta = \| z _ { \bar { \psi } } - z _ { * } \| _ { \infty } , \qquad \eta = \mathrm { K L } \left( q _ { \mathrm { S P L } } ^ { j } \| \pi _ { t } ^ { j } \right) ,
$$

where

$$
q _ { \mathrm { S P L } } ^ { j } ( u ) \propto \exp ( \beta z _ { \bar { \psi } } ( u ) ) ,
$$

and let

$$
R _ { z } = \operatorname* { m a x } _ { u \in \mathcal { A } } z _ { * } ( u ) - \operatorname* { m i n } _ { u \in \mathcal { A } } z _ { * } ( u ) .
$$

Theorem 1 in the main paper states that

$$
\mathrm { R e g } _ { z , t } ^ { j } ( \pi _ { t } ^ { j } ) \leq \frac { \log K } { \beta } + 2 \delta + R _ { z } \sqrt { \frac { \eta } { 2 } } ,\tag{A.18}
$$

where

$$
\mathrm { R e g } _ { z , t } ^ { j } ( \pi _ { t } ^ { j } ) = \operatorname* { m a x } _ { u \in \mathcal { A } } z _ { * } ( u ) - \mathbb { E } _ { u \sim \pi _ { t } ^ { j } } [ z _ { * } ( u ) ] .\tag{A.19}
$$

Proof. Add and subtract the expected reference score under $q _ { \mathrm { S P L } } ^ { j }$ :

$$
\mathrm { R e g } _ { z , t } ^ { j } ( \pi _ { t } ^ { j } ) = \left[ \operatorname* { m a x } _ { u } { z _ { * } ( u ) } - \mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { * } ] \right] + \left[ \mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { * } ] - \mathbb { E } _ { \pi _ { t } ^ { j } } [ z _ { * } ] \right] .
$$

Since

$$
\| z _ { \bar { \psi } } - z _ { * } \| _ { \infty } \leq \delta ,
$$

we have

$$
\operatorname* { m a x } _ { u } z _ { * } ( u ) - \mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { * } ] \leq \operatorname* { m a x } _ { u } z _ { \bar { \psi } } ( u ) - \mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { \bar { \psi } } ] + 2 \delta .
$$

Applying Eq. (A.14) gives

$$
\operatorname* { m a x } _ { u } z _ { * } ( u ) - \mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { * } ] \leq \frac { \log K } { \beta } + 2 \delta .
$$

For the second term,

$$
\mathbb { E } _ { q _ { \mathrm { S P L } } ^ { j } } [ z _ { * } ] - \mathbb { E } _ { \pi _ { t } ^ { j } } [ z _ { * } ] \leq R _ { z } \mathrm { T V } \left( q _ { \mathrm { S P L } } ^ { j } , \pi _ { t } ^ { j } \right) .
$$

Pinsker’s inequality gives

$$
\mathrm { T V } \left( q _ { \mathrm { S P L } } ^ { j } , \pi _ { t } ^ { j } \right) \le \sqrt { \frac { \mathrm { K L } ( q _ { \mathrm { S P L } } ^ { j } | | \pi _ { t } ^ { j } ) } { 2 } } = \sqrt { \frac { \eta } { 2 } } .
$$

Combining the two bounds proves Eq. (A.18).

The bound separates three local sources of receiver error: finite-temperature softness, error in the evidence-conditioned score estimate, and incomplete distillation of the SPL target into the decentralized policy. Equation (A.16) makes the distillation term directly observable during training, while Lemma 2 gives a population-level connection between score approximation and outcome-model error under uniform coverage.

## A.6. Scope of the Local Analysis

The results above characterize the local auxiliary objectives used by SPL. They support the construction and interpretation of the training signals but do not establish global convergence or monotonic improvement of the cooperative return.

Acquisition model and prior misspecification. Proposition 1 in the main paper is a model-relative value-of-information result. Bayesian consistency within the acquisition model guarantees non-negative prospective gain for the top-k surrogate, including when the model uses internally consistent but misspecified priors. Such misspecification can change the relation between the acquisition score and performance under the true environment distribution. Section D.6 evaluates this sensitivity while keeping the remaining structured model components fixed.

Receiver approximation scope. Proposition 2 in the main paper controls expected local-utility error on the retained selected-action distribution. The receiver distillation quantity <sub>η</sub> is directly computable from the target and actor distributions through Eq. (A.16). Lemma 2 further connects mean squared standardized-score error to population outcome-model risk under a uniform retained-action coverage condition. It does not provide a uniform per-context bound, and empirical validation loss introduces the usual finite-sample generalization error. Theorem 1 in the main paper therefore remains a local standardized-score guarantee, not a bound on raw utility or full team return.

Local utility and retained outcomes. The retained-outcome utility in Eq. (A.2) describes the tasklevel consequence of an attributable receiver action and need not equal the full Dec-POMDP action value. Receiver-side results are conditional on attributable outcomes retained in $\mathcal { D } _ { \mathrm { o u t } }$ and admit an unconditional interpretation only under Assumption 1. Under partially overlapping participation, the receiver context may combine evidence acquired before and after participation begins; the analysis conditions on the complete actor-visible context and does not isolate the causal contribution of pre participation evidence.

Backbone optimization. SPL auxiliary actor updates are applied in addition to the native MAPPO, IPPO, or HAPPO optimization. The analysis does not provide a convergence guarantee for the resulting joint optimization; the reported task-completion and return results are empirical.

## B. Implementation and Reproducibility Details

This section provides the environment configurations, participation schedules, SPL target construction, optimization settings, and training procedure used in the reported experiments. All methods are implemented within the HARL framework. We denote the SPL extensions of MAPPO, IPPO, and HAPPO as SPL-MAPPO, SPL-IPPO, and SPL-HAPPO, respectively. Within each matched comparison, the corresponding unmodified backbone is referred to as MAPPO, IPPO, or HAPPO.

## B.1. Environment Configuration

The primary four-agent MPE and RWARE evaluations contain two earlier participants, two future collaborators, and eight candidate tasks. Each domain includes Base and Advanced variants. We additionally evaluate SPL in physics-based UAV–UGV tasks in Isaac Lab with either one UAV and one UGV or two UAVs and two UGVs. Persistent task-relevant evidence produced by earlier participants remains available to future collaborators.

The Base variants use fixed task assignments and earlier-participant–future-collaborator correspondences. MPE-Advanced randomizes balanced task shards, spawn assignments, and future-collaborator identities, while RWARE-Advanced permutes task identifiers and uses a shared candidate pool with dynamic task claims. The Isaac Lab settings introduce physical motion, obstacles, and planner-mediated downstream task execution. The eight-agent scaling configurations are described separately below.

Table B.1: Four-agent MPE/RWARE and Isaac Lab configurations.
<table><tr><td>Environment</td><td>Earlier</td><td>Future</td><td>Tasks</td><td>Horizon</td></tr><tr><td>MPE-Base</td><td>2</td><td>2</td><td>8</td><td>125 (250)</td></tr><tr><td>MPE-Advanced</td><td>2</td><td>2</td><td>8</td><td>125 (250)</td></tr><tr><td>RWARE-Base</td><td>2</td><td>2</td><td>8</td><td>500 (1000)</td></tr><tr><td>RWARE-Advanced</td><td>2</td><td>2</td><td>8</td><td>500 (1000)</td></tr><tr><td>Isaac Lab (2-agent)</td><td>1 UAV</td><td>1 UGV</td><td>4</td><td>600</td></tr><tr><td>Isaac Lab (4-agent)</td><td>2 UAVs</td><td>2 UGVs</td><td>6</td><td>840</td></tr></table>

For the primary four-agent MPE and RWARE settings, the environments use the same eight task-validity priors and a fixed-valid-task sampling rule. Probe outcomes are noisy but persistent, and each future collaborator can execute at most two candidate tasks. Table B.2 summarizes the main task and evidence parameters.

## B.2. Temporal Participation Patterns

Participation intervals follow the half-open convention $\left[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } \right)$ . Inactive agents take a null environment action and make no efective policy decision. The Base and Advanced variants within each domain use the same participation schedules.

We evaluate five participation patterns. Non-overlap uses a strict team-wide handof from earlier participants to future collaborators. Overlap-25 and Overlap-50 introduce fixed levels of cross-role overlap. Dynamic samples pair-specific handof times each episode while preserving aggregate role

Table B.2: Task and evidence settings for MPE and RWARE.
<table><tr><td>Setting</td><td>MPE</td><td>RWARE</td></tr><tr><td>Candidate tasks</td><td>8</td><td>8</td></tr><tr><td>Validity priors</td><td>(0.80,0.80,0.65,0.65,0.35,0.35,0.20,0.20)</td><td></td></tr><tr><td>Probe budget / earlier participant</td><td>5</td><td>6</td></tr><tr><td>Probe limit / task</td><td>2</td><td>2</td></tr><tr><td>Probe accuracy</td><td>0.85</td><td>0.85</td></tr><tr><td>Publication delay</td><td>5</td><td>5</td></tr><tr><td>Execution budget / future collaborator</td><td>2</td><td>2</td></tr><tr><td>Valid-task value</td><td>{12, 12, 10, 10, 8, 8, 6, 6}</td><td>1</td></tr><tr><td>Probe cost</td><td>0.2</td><td>0.01</td></tr><tr><td>Invalid-execution cost</td><td>8.0</td><td>0.25</td></tr></table>

exposure. Cascaded distributes the two earlier-participant–future-collaborator pairs across successive participation intervals, with an extended horizon to preserve total role exposure.

Table B.3: Four-agent participation schedules.
<table><tr><td></td><td>Domain Participation Horizon</td><td></td><td> $P _ { 0 }$ </td><td> $P _ { 1 }$ </td><td> $R _ { 0 }$ </td><td> $R _ { 1 }$ </td></tr><tr><td>MPE</td><td>Non-overlap Overlap-25</td><td rowspan="4">125 125 125</td><td rowspan="4">[0, 45) [0, 45) [0, 45)</td><td rowspan="4">[0,45) [0, 45) [0, 45)  $[ 0 , h _ { 1 } )$ </td><td rowspan="4">[45,125) [34, 125) [22, 125) [h0, 125)</td><td rowspan="4">[45,125) [34, 125) [22, 125) 125)</td></tr><tr><td>MPE</td></tr><tr><td>MPE</td></tr><tr><td>MPE</td><td>Dynamic 125</td></tr><tr><td>MPE</td><td>Cascaded</td><td>250</td><td>[0, 45)</td><td>[125,170)</td><td>[45, 125)</td><td>[170,250)</td></tr><tr><td>RWARE</td><td>Non-overlap</td><td>500</td><td>[0,200)</td><td>[0, 200)</td><td>[200,500)</td><td>[200,500)</td></tr><tr><td>RWARE</td><td>Overlap-25</td><td>500</td><td>[0,175)</td><td>[0,225)</td><td>[175,500)</td><td>[225,500)</td></tr><tr><td>RWARE</td><td>Overlap-50</td><td>500 500</td><td>[0, 150) [0, h_0)</td><td>[0,250) [0, h1)</td><td>[150,500)</td><td>[250,500)</td></tr><tr><td>RWARE</td><td>Dynamic</td><td></td><td></td><td></td><td>[h0, 500)</td><td>[h1, 500)</td></tr><tr><td>RWARE</td><td>Cascaded</td><td>1000</td><td>[0,200)</td><td>[500,700)</td><td>[200,500)</td><td>[700, 1000)</td></tr></table>

Here, $P _ { 0 } , P _ { 1 }$ denote earlier participants and $R _ { 0 } , R _ { 1 }$ denote future collaborators. In MPE, Overlap-25 and Overlap-50 provide 11 and 23 steps of cross-role overlap, respectively; the corresponding RWARE overlaps are 50 and 100 steps. For Dynamic participation, $h _ { 0 } = \bar { h } + \delta$ and $h _ { 1 } = \bar { h } - \delta$ , where δ is sampled once per episode from a symmetric discrete uniform distribution. MPE uses $\bar { h } = 4 5$ and $\delta \in \{ - 1 1 , \ldots , 1 1 \}$ , while RWARE uses $\dot { \bar { h } } = 2 0 0$ and $\delta \in \{ - 5 0 , \ldots , 5 0 \}$ . Consequently, Dynamic changes the pair-specific handof times while preserving aggregate earlier-participant and future-collaborator exposure. Cascaded preserves the same role exposure while changing its temporal ordering.

## B.3. Isaac Lab Configuration

The two-agent Isaac Lab task contains four candidate sites, of which two are serviceable in each episode, on a $1 2 \times 1 2$ occupancy grid with five barriers. The four-agent setting contains six candidate sites, of which four are serviceable. Candidate positions are public from reset, while serviceability remains hidden. UAV sensing has accuracy 0.85 and a five-step publication delay. UGV motion is planned from the published occupancy map, and selecting a task commits the UGV to its physical execution. Neither the actor nor the critic receives hidden serviceability.

Dynamic handof times are sampled per episode. The two-agent setting exposes the normalized handof step to the actors, while the four-agent setting uses a shared UGV participation window with a sampled start time. MAPPO and SPL-MAPPO use matched participation schedules and evaluation instances. Each setting uses three independent training seeds and 200 final evaluation episodes per seed. Final policies are evaluated deterministically, and checkpoints are not selected by performance.

Table B.4: Isaac Lab participation and training configurations.
<table><tr><td>Agents</td><td></td><td></td><td>Participation Training steps Participation windows</td></tr><tr><td> $1 \mathrm { U A V } + 1 \mathrm { U G V }$ </td><td>Non-overlap 1,996,800</td><td></td><td>UAV [0, 200); UGV [200,600)</td></tr><tr><td> $1 \mathrm { U A V } + 1 \mathrm { U G V }$ </td><td>Dynamic</td><td>1,996,800</td><td> $\operatorname { U A V } \left[ 0 , h \right) ; \operatorname { U G V } \left[ h , 6 0 0 \right) _ { ! }$   $h \sim \mathrm { U n i f } \{ 1 5 0 , \dots , 2 5 0 \}$ </td></tr><tr><td> $2 \mathrm { U A V s } + 2 \mathrm { U G V s }$ </td><td>Non-overlap 3,010,560</td><td></td><td>UAVs [0, 300); UGVs [300,840)</td></tr><tr><td> $2 \mathrm { U A V s } + 2 \mathrm { U G V s }$ </td><td>Dynamic</td><td>3,010,560</td><td> $\mathrm { U A V s } ~ [ 0 , 3 0 0 ) ; \mathrm { U G V s }$   $[ h , h + 5 4 0 ) ;$   $h \in \{ 1 5 0 , 1 8 0 , \dots , 3 0 0 \}$ </td></tr></table>

## B.4. SPL Target Construction

Prospective acquisition target. The frozen acquisition model constructs the task-validity posterior from the configured validity priors, the fixed number of valid tasks, and public probe outcomes available in the actor-visible context. Repeated probe observations are aggregated through their positive and negative counts using the configured probe accuracy. Hidden task validity and future outcomes are never provided to either the actor or the target-construction procedure.

Posterior marginals $b _ { r } = \operatorname* { P r } ( W _ { r } = 1 \ | \ c _ { t } ^ { i } )$ are used to compute the prospective gain $\Delta _ { t } ^ { i } ( u )$ defined in Eq. (A.4). In the implementation, task-directed actions with $\Delta _ { t } ^ { i } ( u ) \leq 1 0 ^ { - 8 }$ receive no acquisition-target mass. For the remaining legal task-directed actions,

$$
w _ { t } ^ { i } ( u ) = \operatorname* { m a x } \{ \Delta _ { t } ^ { i } ( u ) , \epsilon _ { \mathrm { a c q } } \} \exp \left( - \frac { d _ { t } ^ { i } ( u ) } { \tau _ { d } } \right) , \qquad \epsilon _ { \mathrm { a c q } } = 1 0 ^ { - 1 2 } , \quad \tau _ { d } = 0 . 5 0 .\tag{B.1}
$$

The weights are normalized over eligible task-directed actions. The target assigns 0.95 total probability mass to this distribution and 0.05 to the scan action. If no positive-gain task-directed action is legal, the scan action receives unit target mass. The acquisition model and target-construction rule remain fixed throughout training.

Outcome-supervised receiver target. For each future collaborator $j ,$ SPL uses a candidate-shared outcome model

$$
p _ { \psi } ^ { j } ( u \mid x ) = \mathrm { s i g m o i d } \left( f _ { \psi } { \big ( } g _ { j } ( x , u ) { \big ) } \right) ,\tag{B.2}
$$

where $g _ { j } ( x , u )$ contains the actor-visible pre-decision context, candidate-action features, and collaborator-role identity. The same scorer $f _ { \psi }$ is applied across candidate actions. Training labels are available only for selected actions whose outcomes can be reliably attributed; unselected actions receive no counterfactual labels.

A slowly updated target model $p _ { \bar { \psi } }$ is used to construct the receiver target. For legal candidate action u,

$$
U _ { \bar { \psi } } ^ { j } ( u \mid x ) = p _ { \bar { \psi } } ^ { j } ( u \mid x ) c _ { j } ^ { + } ( u ) + \left[ 1 - p _ { \bar { \psi } } ^ { j } ( u \mid x ) \right] c _ { j } ^ { - } ( u ) .\tag{B.3}
$$

The task-level utilities are fixed during training. In MPE, $c _ { j } ^ { + } ( u ) = v ( u ) / 1 2$ with $v ( u ) \in \{ 1 2 , 1 0 , 8 , 6 \}$ and $c _ { i } ^ { - } ( u ) ~ = ~ - 8 / 1 2$ . In RWARE, $c _ { j } ^ { + } ( u ) = 1$ and $c _ { j } ^ { - } ( u ) \stackrel { \cdot } { = } - 0 . 2 5$ . In Isaac Lab, $c _ { j } ^ { + } ( u ) = 1$ and $c _ { j } ^ { - } ( u ) = 0$ . The no-op utility is zero. These quantities define the local utilities used for SPL supervision and are not interpreted as the complete environment return.

Utilities are standardized over the current legal action set using $\epsilon _ { z } = 1 0 ^ { - 6 }$ as the variance stabilizer, following Eq. (A.11). The receiver target is then the Boltzmann distribution in Eq. (A.12). All reported experiments use $\beta = 1$ . Receiver supervision uses this soft utility target directly, without mixing it with the current actor policy or converting it to a one-hot target.

Table B.5: Training hyperparameters for the PPO backbones and SPL.
<table><tr><td colspan="2">PPO backbone</td><td colspan="2">SPL-specific</td></tr><tr><td>Parameter</td><td>Value</td><td>Parameter</td><td>Value</td></tr><tr><td>Actor hidden layers</td><td>[128,128]</td><td>Outcome hidden width</td><td>128</td></tr><tr><td>Activation</td><td>ReLU</td><td>Outcome learning rate</td><td> $3 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>Actor learning rate</td><td> $1 \times 1 0 ^ { - 4 }$ </td><td>Replay capacity</td><td>20,000</td></tr><tr><td>Critic learning rate</td><td> $5 \times 1 0 ^ { - 4 }$ </td><td>Outcome minibatch size</td><td>128</td></tr><tr><td>PPO epochs</td><td>3</td><td>Outcome gradient steps</td><td>4</td></tr><tr><td>Critic epochs</td><td>3</td><td>Target EMA rate  $\rho$ </td><td>0.05</td></tr><tr><td>PPO clip</td><td>0.2</td><td>Outcome-model gate</td><td>128</td></tr><tr><td>Entropy coefficient</td><td>0.002</td><td>Per-class model gate</td><td>16</td></tr><tr><td>Discount  $\gamma$ </td><td>0.99</td><td>Receiver-policy gate</td><td>1,024</td></tr><tr><td>GAE λ</td><td>0.95</td><td>Per-class receiver gate</td><td>128</td></tr><tr><td>Value normalization</td><td>enabled</td><td>Acquisition epochs</td><td>1</td></tr><tr><td>Recurrent policy</td><td>disabled</td><td>Receiver epochs</td><td>20</td></tr><tr><td>Parameter sharing</td><td>disabled</td><td>Acquisition task mass</td><td>0.95</td></tr><tr><td>Gradient norm cap</td><td>10</td><td>Distance temperature  $\tau _ { d }$ </td><td>0.50</td></tr><tr><td></td><td></td><td>Receiver target strength  $\beta$ </td><td>1.0</td></tr></table>

## B.5. Optimization and Training Settings

MAPPO, IPPO, and HAPPO use the same feed-forward actor architecture and PPO hyperparameters listed in Table B.5. Policies are not parameter-shared within a team. MAPPO and HAPPO retain centralized critics, whereas IPPO uses independent critics based on local observations. SPL does not use centralized critic state and does not modify the native reward, return estimator, critic objective, or critic target. The corresponding augmented methods are denoted SPL-MAPPO, SPL-IPPO, and SPL-HAPPO. Vanilla–SPL comparisons use the same backbone configuration, environment, participation schedule, training budget, and evaluation protocol.

All primary four-agent MPE and RWARE method–environment–participation combinations are trained for two million environment steps without early stopping. Reported results use five independent training seeds per method and configuration, with 200 final evaluation episodes per seed. Training seeds are the independent replication units. Final policies are evaluated deterministically using a fixed 200-episode reset-seed schedule reused across training seeds within each environment protocol. The scaling and Isaac Lab settings use the training procedures described in their respective configuration sections.

Training-seed pairing. A retrospective audit recovered all 600 expected method–training-seed records for the 60 primary Vanilla–SPL comparisons. Same-training-seed pairing was verified for 285 pairs across 57 configurations, providing five paired training seeds in each of those configurations. The Non-overlap MAPPO comparisons in MPE-Advanced, RWARE-Base, and RWARE-Advanced use diferent Vanilla and SPL training-seed sets. Their observed method means remain included in the descriptive results, but paired confidence intervals and paired p-values are reported as unavailable. Consequently, paired inference covers 57 of the 60 primary configurations. Full statistical details are provided in Appendix D.7.

Baseline implementation and adaptation. MAE-MAPPO uses the same per-agent observations, legal task-option masks, team rewards, participation schedules, and evaluation protocol as MAPPO. For Sable and SMPE<sup>2</sup>, we retain the original learning architectures and optimization objectives and adapt only the environment interface to the SP tasks. Both methods receive the same environment-specific local observations, legal task-option masks, shared rewards, participation schedules, and episode boundaries as the PPO-based methods. A common task-option controller maps each legal high-level choice to the corresponding native environment action, while inactive agents receive only the environmentdefined no-op action. Where global or state-level inputs are required, they are constructed solely from non-privileged observable information and do not expose hidden task validity. Sable retains its sequence-model rollout procedure, while SMPE<sup>2</sup> retains its recurrent policy, critic, state-VAE, and state-filtering components. All primary four-agent baseline comparisons use the same two-million-step training budget, five independent training seeds per method and configuration, and 200 final evaluation episodes per seed.

Knowledge-matched control. SKD-MAPPO uses the same structured task information, prospective acquisition supervision, task utilities, outcome-model training procedure, target-model update, readiness conditions, and auxiliary update budget as SPL-MAPPO. Its online outcome model is trained from the same complete actor-visible pre-decision context and attributable outcome labels as in SPL. When the learned target outcome model is subsequently used to construct receiver supervision, however, predecessor-generated task evidence is masked from its input. The decentralized actor observation remains unchanged. SKD-MAPPO therefore retains evidence during outcome-model learning but withholds that evidence when constructing the receiver target. This control tests the efect of removing inherited evidence specifically from the target-construction pathway while leaving the remaining SPL training machinery unchanged.

Evidence-redacted control. Evidence-redacted uses the same MAPPO backbone, acquisition supervision, task utilities, attributable outcome labels, readiness conditions, receiver utility construction, and auxiliary update budget as SPL-MAPPO. Unlike SKD-MAPPO, inherited predecessor-generated task evidence is removed consistently from the feature vectors supplied to both the online outcome model during outcome-model training and the target outcome model during receiver-target construc tion. The replay bufer retains the original actor-visible pre-decision context, and redaction is applied when outcome-model features are constructed. The decentralized actor observation is not modified. Evidence-redacted therefore evaluates receiver learning when inherited task evidence is unavailable to both outcome prediction and receiver-target construction, while preserving the remaining training and execution pipeline.

The distinction between the two controls is thus the stage at which inherited evidence is removed. SKD-MAPPO allows the outcome model to learn from the complete evidence-conditioned context but removes predecessor evidence when that model is queried to construct receiver supervision. Evidence-redacted removes the same class of inherited evidence from both outcome-model learning and receiver-target construction. Neither control removes the evidence from the decentralized actor observation, and neither exposes hidden task validity or future outcomes to the actor or critic.

## B.6. Training Procedure and Decentralized Execution

Outcome replay and readiness. When a future collaborator selects a task-directed action, SPL stores its actor-visible pre-decision context, selected action, collaborator-role identity, and subsequently attributable outcome. Successful completion is labeled 1, while an attributable unsuccessful attempt is labeled 0. Interrupted, expired, unresolved, or otherwise ambiguous attempts are excluded and receive no synthetic labels. Information observed after the decision is used only to determine the outcome label and is not included in the stored pre-decision context.

Outcome-model optimization begins once $\mathcal { D } _ { \mathrm { o u t } }$ contains at least 128 retained decisions, including at least 16 examples from each outcome class. Each eligible training iteration then applies four outcomemodel gradient steps with minibatches of at most 128 replay samples. Receiver-policy supervision begins once the replay contains at least 1,024 retained decisions with at least 128 examples from each class. Before this condition is met, future collaborators receive only their native MARL actor update. Prospective acquisition supervision does not depend on the receiver-side readiness conditions.

Update order and gradient boundaries. After each rollout, attributable receiver outcomes are appended to $\mathcal { D } _ { \mathrm { o u t } }$ . Once the outcome-model readiness condition is satisfied, the online outcome model is updated by binary cross-entropy, followed by the exponential-moving-average update

$$
\begin{array} { r } { \bar { \psi }  ( 1 - \rho ) \bar { \psi } + \rho \psi . } \end{array}
$$

When receiver supervision is ready, receiver targets are constructed from the updated target outcome model before backbone optimization.

SPL actor updates are interleaved with the native actor updates. For each eligible actor, the native backbone actor update is performed first, followed by the corresponding SPL update. MAPPO processes actors in fixed agent order and updates the centralized critic after all actor updates. IPPO applies the native actor update, SPL update, and independent critic update for each agent before proceeding to the next agent.

HAPPO retains its sequential heterogeneous actor-update order. For each actor, the current cumulative factor is stored before the native actor update, followed by the applicable SPL update. Action probabilities are then re-evaluated to update the cumulative factor used by subsequent actors. Thus, later actors can receive factors reflecting the post-SPL policies of earlier actors. We do not claim that SPL-HAPPO preserves HAPPO’s original monotonic-improvement guarantee.

The outcome loss updates only the online outcome model ψ, the acquisition loss updates only the eligible earlier-participant actor, and the receiver loss updates only the eligible future-collaborator actor. Gradients do not propagate through the frozen acquisition model, target outcome model, utility construction, replayed outcomes, future trajectories, or environment dynamics. SPL does not modify the environment reward, transition function, observation space, action space, native return estimator, or critic objective.

Decentralized execution. At execution time, each active agent acts solely through its decentralized policy,

$$
u _ { t } ^ { i } \sim \pi _ { \theta _ { i } } \left( \cdot \mid \tau _ { t } ^ { i } , M _ { t } ^ { i } \right) .\tag{B.4}
$$

The acquisition model, outcome replay, online and target outcome models, utility calculations, and SPL target distributions are used only during training. SPL-MAPPO, SPL-IPPO, and SPL-HAPPO use the original decentralized policy architecture at execution time, without privileged state, auxiliary inference, or additional communication.

## B.7. Training Algorithm

Algorithm 1 summarizes one SPL training iteration. SPL preserves the native optimization procedure of each MARL backbone and adds the role-appropriate auxiliary actor update. Earlier participants receive prospective acquisition supervision, while future collaborators receive receiver supervision once the corresponding replay-readiness condition is satisfied.

Algorithm 1 One training iteration with Staggered Participation Learning   
Require: Backbone B; policies $\{ \pi _ { \boldsymbol { \theta } _ { i } } \} _ { i \in \mathcal { N } } ;$ ; frozen acquisition model; online outcome model $p _ { \psi } ;$ target outcome model $p _ { \bar { \psi } } ;$   
outcome replay $\mathcal { D } _ { \mathrm { o u t } }$   
1: Collect one cooperative rollout   
2: Compute native return and advantage inputs   
3: Extract attributable future-collaborator outcomes   
4: Add retained samples $( x _ { t } ^ { j } , u _ { t } ^ { j } , j , y _ { t } ^ { j } )$ to $\mathcal { D } _ { \mathrm { o u t } }$   
5: if outcome-model readiness condition is satisfied then   
6: Update ψ using $\mathcal { L } _ { \mathrm { o u t } }$   
7: $\bar { \psi }  ( 1 - \rho ) \bar { \psi } + \rho \psi$   
8: end if   
9: Construct acquisition targets for eligible earlier participants   
10: if receiver-supervision readiness condition is satisfied then   
11: Construct receiver targets using $p _ { \bar { \psi } }$   
12: end if   
13: if B is MAPPO then   
14: for each actor i do   
15: Apply the native MAPPO actor update to i   
16: Apply the eligible SPL actor update to i   
17: end for   
18: Update the centralized critic   
19: else if B is IPPO then   
20: for each actor i do   
21: Apply the native IPPO actor update to i   
22: Apply the eligible SPL actor update to i   
23: Update the local critic of i   
24: end for   
25: else if B is HAPPO then   
26: Initialize cumulative factor $F \gets 1$   
27: Select the HAPPO actor-update order   
28: for each actor i in the selected order do   
29: Store the current factor F for actor i   
30: Evaluate the pre-update action probabilities   
31: Apply the native HAPPO actor update to i   
32: Apply the eligible SPL actor update to i   
33: Evaluate the post-update action probabilities   
34: Update the cumulative factor F   
35: end for   
36: Update the centralized critic   
37: end if   
38: return updated policies, critics, outcome models, and replay

## B.8. Eight-Agent Scaling Protocol

Experimental scope. The eight-agent scaling study evaluates MPE-Advanced and RWARE-Advanced with four earlier participants $P _ { 0 } , \ldots , P _ { 3 }$ , four future collaborators $R _ { 0 } , \ldots , R _ { 3 }$ , and sixteen candidate tasks. We evaluate Non-overlap, Dynamic, and Cascaded participation using MAPPO and SPL-MAPPO. Nonoverlap and Dynamic are trained for two million environment steps, while Cascaded uses four million steps to accommodate its extended participation horizon; all configurations use three independent training runs.

Environment configuration. The sixteen candidate tasks form four groups of four, with two valid tasks sampled in each group. In MPE-Advanced, balanced task shards are randomly assigned at reset and the earlier-participant–future-collaborator correspondence is randomly assigned at handof. This correspondence is not available to earlier participants beforehand. Earlier participants and future collaborators spawn on opposite sides of the environment at $x = - 1 . 8 2$ and $x = 1 . 8 2$ , respectively, with

Table B.6: Task and evidence configuration for the eight-agent scaling experiments. The notation $x ^ { \times 4 }$ denotes four repetitions of value x.
<table><tr><td>Setting</td><td>MPE-Advanced</td><td>RWARE-Advanced</td></tr><tr><td>Earlier participants</td><td>4</td><td>4</td></tr><tr><td>Future collaborators</td><td>4</td><td>4</td></tr><tr><td>Candidate tasks</td><td>16</td><td>16</td></tr><tr><td>Fixed valid tasks</td><td>8</td><td>8</td></tr><tr><td>Valid tasks / four-task group</td><td>2</td><td>2</td></tr><tr><td>Validity priors</td><td> $( 0 . 8 0 ^ { \times 4 } , 0 . 6 5 ^ { \times 4 } , 0 . 3 5 ^ { \times 4 } , 0 . 2 0 ^ { \times 4 } )$ </td><td></td></tr><tr><td>Probe budget / earlier participant</td><td>5</td><td>6</td></tr><tr><td>Probe limit / task</td><td>2</td><td>2</td></tr><tr><td>Probe accuracy</td><td>0.85</td><td>0.85</td></tr><tr><td>Publication delay</td><td>5</td><td>5</td></tr><tr><td>Execution budget / future collaborator</td><td>2</td><td>2</td></tr><tr><td>Valid-task value</td><td> $( 1 2 ^ { \times 4 } , 1 0 ^ { \times 4 } , 8 ^ { \times 4 } , 6 ^ { \times 4 } )$ </td><td>1</td></tr><tr><td>Probe cost</td><td>0.20</td><td>0.01</td></tr><tr><td>Invalid-execution cost</td><td>8.00</td><td>0.25</td></tr></table>

$$
y \in \{ 0 . 7 6 , 0 . 2 5 3 3 , - 0 . 2 5 3 3 , - 0 . 7 6 \} .
$$

The agent speed, view radius, interaction radius, and placement jitter are 1.0, 4.0, 0.18, and 0.02. Each actor observation has dimension 339, and the MAPPO critic input concatenates the eight actor observations, giving dimension 2712. Hidden task validity is unavailable to both representations.

RWARE-Advanced uses one shelf row, three shelf columns of height eight, and sensor range one. Candidate task identifiers are permuted each episode, future collaborators select from a shared candidate pool, and competing claims are resolved by first arrival. Hidden task validity is excluded from both actor observations and the MAPPO critic input.

Temporal participation schedules. All participation intervals follow the half-open convention $\left[ t _ { \mathrm { s t a r t } } , t _ { \mathrm { e n d } } \right)$ . Let H denote the ordinary episode horizon, a the nominal handof time, d the maximum Dynamic deviation, and $H _ { \mathrm { c a s } }$ the Cascaded horizon. Table B.7 summarizes the corresponding values.  
Table B.7: Temporal parameters and aggregate role exposure for the eight-agent experiments. Exposure is measured in agent-steps per episode.
<table><tr><td>Domain</td><td>H</td><td>a</td><td> $d$ </td><td> $H _ { \mathrm { c a s } }$ </td><td>Earlier</td><td>Future</td></tr><tr><td>MPE-Advanced</td><td>125</td><td>45</td><td>11</td><td>500</td><td>180</td><td>320</td></tr><tr><td>RWARE-Advanced</td><td>500</td><td>200</td><td>50</td><td>2000</td><td>800</td><td>1200</td></tr></table>

For Non-overlap,

$$
\mathcal { T } _ { \mathrm { n o n } } ^ { P _ { i } } = [ 0 , a ) , \qquad \mathcal { T } _ { \mathrm { n o n } } ^ { R _ { i } } = [ a , H ) , \qquad i \in \{ 0 , 1 , 2 , 3 \} .\tag{B.5}
$$

For Dynamic participation, two independent deviations are sampled per episode,

$$
\delta _ { k } \sim \operatorname { U n i f } \{ - d , \ldots , d \} , \qquad k \in \{ 0 , 1 \} ,\tag{B.6}
$$

with paired handofs

$$
h _ { 2 k } = a + \delta _ { k } , \qquad h _ { 2 k + 1 } = a - \delta _ { k } .\tag{B.7}
$$

The corresponding intervals are

$$
\begin{array} { r } { \mathcal { T } _ { \mathrm { d y n } } ^ { P _ { i } } = [ 0 , h _ { i } ) , \qquad \mathcal { T } _ { \mathrm { d y n } } ^ { R _ { i } } = [ h _ { i } , H ) . } \end{array}\tag{B.8}
$$

The handof vector is sampled each episode and included in the actor-visible schedule information.   
Since $\textstyle \sum _ { i = 0 } ^ { 3 } h _ { i } = 4 a$ , Dynamic preserves the same aggregate role exposure as Non-overlap.

For Cascaded participation,

$$
\begin{array} { r } { \mathcal { Z } _ { \mathrm { c a s } } ^ { P _ { i } } = [ i \boldsymbol { H } , i \boldsymbol { H } + a ) , \qquad \mathcal { Z } _ { \mathrm { c a s } } ^ { R _ { i } } = [ i \boldsymbol { H } + a , ( i + 1 ) \boldsymbol { H } ) , \qquad i \in \{ 0 , 1 , 2 , 3 \} . } \end{array}\tag{B.9}
$$

The resulting horizon is $H _ { \mathrm { c a s } } = 4 H$ , preserving aggregate role exposure while distributing the four earlier-participant–future-collaborator pairs across successive intervals.

Training and evaluation. The eight-agent MAPPO and SPL-MAPPO runs use the feed-forward architecture, PPO settings, and SPL-specific hyperparameters reported in Table B.5. Policies are not parameter-shared and recurrent policies are disabled. Non-overlap and Dynamic are trained for two million environment steps, whereas Cascaded is trained for four million steps; each configuration uses three independent runs and is evaluated over 200 final episodes per run. MPE-Advanced uses eight parallel rollout environments, whereas RWARE-Advanced uses one. Reported task-completion results use the final checkpoint at the corresponding training budget, without performance-based checkpoint selection.

## C. Observed Training Cost

We report observed wall-clock training time as a practical indication of computational cost. Measurements correspond to completed full-budget runs; short qualification and smoke-scale runs are excluded. Because experiments ran under varying system load and host hardware was not recorded consistently, these values are descriptive runtime measurements. They are not controlled hardware-throughput benchmarks or estimates of algorithmic complexity.

Table C.1 summarizes the observed training time. In the primary four-agent experiments, all methods use two million environment steps, and Table C.1(a) reports mean minutes per completed run averaged over the ten settings formed by two environment variants and five participation patterns. Table C.1(b) reports the eight-agent scaling and Isaac Lab experiments. The eight-agent Non-overlap and Dynamic settings use 2M environment steps, whereas Cascaded uses 4M. The two- and four-agent Isaac Lab settings use approximately 2.0M and 3.0M steps, respectively.

Observation. The practical training-time efect of SPL varies across environments and team configurations. In the primary four-agent experiments, SPL incurs moderate additional wall-clock cost for most PPO backbones, while MAPPO and SPL-MAPPO are nearly identical in RWARE. In the eight-agent experiments, the relative increase is modest in MPE and larger in RWARE; absolute times across participation patterns should be interpreted together with their diferent training budgets, particularly the 4M-step

Table C.1: Observed wall-clock training time in minutes. Primary four-agent values are averages across the ten environment–participation settings in each domain; scaling and Isaac Lab values are mean ± standard error over three independent runs. Light blue shading identifies SPL results.  
(a) Primary four-agent experiments.  
(b) Scaling and Isaac Lab experiments.
<table><tr><td>Method</td><td>MPE</td><td>RWARE</td></tr><tr><td>IPPO</td><td>154.1</td><td>166.4</td></tr><tr><td>SPL-IPPO</td><td>175.3</td><td>177.9</td></tr><tr><td>MAPPO</td><td>114.0</td><td>128.3</td></tr><tr><td>SPL-MAPPO</td><td>131.1</td><td>129.6</td></tr><tr><td>HAPPO</td><td>134.6</td><td>128.5</td></tr><tr><td>SPL-HAPPO</td><td>152.4</td><td>145.2</td></tr><tr><td>MAE-MAPPO</td><td>157.2</td><td>128.6</td></tr><tr><td>Sable</td><td>205.1</td><td>210.8</td></tr><tr><td> $S \mathbf { M P E } ^ { 2 }$ </td><td>119.2</td><td>218.0</td></tr></table>

<table><tr><td>Setting</td><td>Participation</td><td>Steps</td><td>MAPPO</td><td>SPL-MAPPO</td></tr><tr><td>MPE-Adv. (8)</td><td>Non-overlap</td><td>2.00M</td><td> $7 6 . 0 \pm 0 . 2$ </td><td> $8 7 . 0 \pm 0 . 1$ </td></tr><tr><td rowspan="4">RWARE-Adv. (8)</td><td>Dynamic</td><td>2.00M</td><td> $8 9 . 5 \pm 0 . 6 $ </td><td> $1 0 3 . 0 \pm 0 . 5$ </td></tr><tr><td>Cascaded</td><td>4.00M</td><td> $1 7 5 . 5 \pm 0 . 2$ </td><td> $1 8 5 . 5 \pm 2 . 7$ </td></tr><tr><td>Non-overlap</td><td>2.00M</td><td> $2 4 2 . 9 \pm 6 . 1$ </td><td> $3 9 6 . 9 \pm 1 5 . 3$ </td></tr><tr><td>Dynamic</td><td>2.00M</td><td> $2 1 6 . 6 \pm 3 . 6$ </td><td> $3 7 7 . 5 \pm 1 1 . 0$ </td></tr><tr><td rowspan="2">Isaac Lab (2)</td><td>Cascaded</td><td>4.00M</td><td> $4 5 3 . 6 \pm 1 6 . 6$ </td><td> $5 3 8 . 3 \pm 1 5 . 4$ </td></tr><tr><td>Non-overlap</td><td>2.00M</td><td> $5 4 8 . 8 \pm 1 7 . 5$ </td><td> $5 6 2 . 9 \pm 1 8 . 8$ </td></tr><tr><td rowspan="2">Isaac Lab (4)</td><td>Dynamic</td><td>2.00M</td><td> $4 3 8 . 1 \pm 9 . 9$ </td><td> $4 5 0 . 0 \pm 7 . 0$ </td></tr><tr><td>Non-overlap Dynamic</td><td>3.01M 3.01M</td><td> $1 0 9 0 . 2 \pm 1 7 . 8$   $1 1 2 0 . 1 \pm 1 5 . 1$ </td><td> $1 2 4 4 . 7 \pm 4 3 . 2$   $1 2 9 5 . 0 \pm 7 . 3$ </td></tr></table>

Cascaded settings. The two-agent Isaac Lab runs show only a small diference, whereas the four-agent settings require approximately 14–16% more observed training time with SPL-MAPPO. Overall, SPL introduces additional training-time computation whose magnitude depends on the environment and configuration, while decentralized execution remains unchanged.

## D. Detailed Experiment Results

The following tables provide the complete evaluation metrics underlying the main-paper results. Task completion (TC) measures the percentage of required downstream task objectives successfully completed within an evaluation episode, while full success (FS) is the percentage of episodes in which all required downstream tasks are completed. We also report episode return and failed execution (FE). Higher values are better for TC, FS, and return, whereas lower FE is better. MPE and RWARE values are reported as mean ± standard error over five independent training seeds and 200 final evaluation episodes per seed. A reported standard error of 0.00 indicates that the five seed-level evaluation means are exactly identical at full precision; further discussion is provided in Appendix D.7.

## D.1. MPE-Base

Table D.1 reports the complete MPE-Base results across the five participation patterns, including task completion, full success, return, and failed execution.

Table D.1: Complete MPE-Base four-agent results at 2M environment steps. Values are mean ± standard error over five independent training seeds and 200 final evaluation episodes per seed. Bold marks the best reported mean in each participation–metric row. For legacy MPE-Base Non-overlap HARL runs, FS is recovered from the retained per-episode completed-task counts using the same all-four-tasks definition.
<table><tr><td rowspan="2">Participation Metric</td><td rowspan="2"></td><td colspan="2">IPPO</td><td colspan="2">MAPPO</td><td colspan="2">HAPPO</td><td colspan="4">Other baselines</td></tr><tr><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>MAE-MAPPO</td><td></td><td>Sable</td><td>SMPE2</td></tr><tr><td rowspan="5">Non-overlap</td><td>TC (%) ↑</td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $7 8 . 3 8 \pm 0 . 3 8$ </td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td></td><td> $7 8 . 7 3 \pm 0 . 4 3$ </td><td> $6 7 . 4 5 \pm 3 . 0 5$ </td><td> ${ \bf 7 9 . 1 8 \pm 0 . 2 3 }$ </td><td> $7 0 . 5 0 \pm 0 . 0 0$ </td><td> $6 2 . 7 0 \pm 0 . 2 5$ </td><td> $4 2 . 7 3 \pm 9 . 2 9$ </td></tr><tr><td>FS (%) ↑</td><td> $1 7 . 5 0 \pm 0 . 0 0$ </td><td> $3 1 . 8 0 \pm 0 . 8 7$ </td><td> $1 7 . 5 0 \pm 0 . 0 0$ </td><td></td><td> $3 2 . 8 0 \pm 1 . 0 6$ </td><td> $1 4 . 0 0 \pm 3 . 5 0$ </td><td> ${ \bf 3 4 . 2 0 \pm 0 . 4 9 }$ </td><td> $1 7 . 5 0 \pm 0 . 0 0$ </td><td> $9 . 0 0 \pm 0 . 7 1$ </td><td> $4 . 2 0 \pm 2 . 5 7$ </td></tr><tr><td>Return ↑</td><td> $2 1 . 5 6 9 \pm 0 . 0 2 9$ </td><td> $2 7 . 3 3 7 \pm 0 . 0 8 3$ </td><td></td><td> $2 1 . 6 4 4 \pm 0 . 0 3 6$ </td><td> $2 7 . 2 3 1 \pm 0 . 0 9 1$ </td><td> $2 1 . 0 1 0 \pm 0 . 6 7 5$ </td><td> ${ \pm 7 . 4 0 9 \pm 0 . 0 4 8 }$ </td><td> $2 1 . 7 4 7 \pm 0 . 0 7 6$ </td><td>15.100 ± 0.199</td><td> $1 1 . 0 1 6 \pm 2 . 4 0 9$ </td></tr><tr><td>FE↓</td><td> $1 . 1 2 5 \pm 0 . 0 0 0$ </td><td> $\mathbf { 0 . 5 8 2 \pm 0 . 0 0 8 }$ </td><td> $1 . 1 2 7 \pm 0 . 0 0 2$ </td><td></td><td> $0 . 6 1 0 \pm 0 . 0 1 2$ </td><td> $1 . 0 5 6 \pm 0 . 0 6 9$ </td><td> $0 . 6 0 8 \pm 0 . 0 0 4$ </td><td> $1 . 1 2 5 \pm 0 . 0 0 0$ </td><td> $1 . 4 5 0 \pm 0 . 0 1 1$ </td><td> $0 . 8 7 3 \pm 0 . 2 3 2$ </td></tr><tr><td>TC (%) ↑</td><td> $6 8 . 2 5 \pm 2 . 7 5$ </td><td> $7 9 . 3 5 \pm 0 . 5 7$ </td><td></td><td> $7 0 . 7 8 \pm 0 . 1 4$ </td><td> $8 0 . 2 5 \pm 0 . 2 1$ </td><td> $7 0 . 7 0 \pm 0 . 1 8$ </td><td> ${ \bf 8 0 . 3 5 \pm 0 . 2 9 }$ </td><td> $6 7 . 9 3 \pm 2 . 9 5$ </td><td> $6 3 . 5 0 \pm 2 . 2 4$ </td><td> $5 3 . 8 5 \pm 5 . 4 3$ </td></tr><tr><td rowspan="4">Overlap-25</td><td>FS (%) ↑</td><td> $1 4 . 9 0 \pm 2 . 6 0$ </td><td> $3 4 . 1 0 \pm 1 . 4 5$ </td><td> $1 7 . 5 0 \pm 0 . 0 0$ </td><td> ${ \bf 3 6 . 8 0 \pm 0 . 4 6 }$ </td><td> $1 7 . 5 0 \pm 0 . 1 6$ </td><td>36.40±0.81</td><td> $1 4 . 0 0 \pm 3 . 3 8$ </td><td> $9 . 3 0 \pm 2 . 2 2$ </td><td></td><td> $4 . 3 0 \pm 2 . 5 3$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 3 0 8 \pm 1 . 0 0 9$ </td><td> $2 7 . 0 6 8 \pm 0 . 2 1 2$ </td><td> $2 1 . 5 2 1 \pm 0 . 0 8 2$ </td><td> $2 7 . 2 3 2 \pm 0 . 1 3 9$ </td><td> $2 1 . 4 3 9 \pm 0 . 1 7 4$ </td><td>27.358±0.122</td><td></td><td> $2 0 . 9 5 0 \pm 0 . 5 5 4$ </td><td> $1 5 . 2 9 5 \pm 1 . 8 6 8$ </td><td> $1 2 . 7 7 8 \pm 2 . 1 2 4$ </td></tr><tr><td>FE↓</td><td> $1 . 1 2 4 \pm 0 . 0 3 6$ </td><td> $\pm 0 . 6 5 4 \pm 0 . 0 0 9$ </td><td> $1 . 1 6 1 \pm 0 . 0 0 7$ </td><td>0.670 ± 0.014</td><td></td><td> $1 . 1 5 9 \pm 0 . 0 1 0$ </td><td>0.658± 0.017</td><td> $1 . 0 7 9 \pm 0 . 0 7 9$ </td><td> $1 . 4 6 0 \pm 0 . 0 9 0$ </td><td> $1 . 2 4 6 \pm 0 . 2 3 6$ </td></tr><tr><td>TC (%) ↑</td><td> $6 7 . 8 5 \pm 3 . 1 5$ </td><td> ${ \bf 7 8 . 9 0 \pm 0 . 3 1 }$ </td><td></td><td></td><td> ${ \bf 7 8 . 9 0 \pm 0 . 2 4 }$ </td><td> $6 9 . 8 0 \pm 0 . 9 2$ </td><td>78.50±0.47</td><td> $7 0 . 6 3 \pm 0 . 2 3$ </td><td> $6 4 . 4 3 \pm 1 . 8 5$ </td><td> $6 3 . 9 5 \pm 3 . 3 4$ </td></tr><tr><td rowspan="4">Overlap-50</td><td>FS (%) ↑</td><td> $1 4 . 0 0 \pm 3 . 5 0$ </td><td> ${ \bf 3 2 . 2 0 \pm 0 . 6 0 }$ </td><td> $\begin{array} { c } { 6 9 . 5 3 \pm 1 . 2 7 } \\ { 1 5 . 7 0 \pm 1 . 3 6 } \end{array}$ </td><td>31.90 ± 0.71</td><td>16.20 ± 0.97</td><td>30.70±1.11</td><td> $1 7 . 5 0 \pm 0 . 1 6$ </td><td></td><td> $1 0 . 3 0 \pm 2 . 1 2$ </td><td> $1 0 . 6 0 \pm 3 . 2 0$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 7 4 4 \pm 0 . 6 8 0$ </td><td>26.429±0.112</td><td> $2 1 . 2 1 3 \pm 0 . 2 8 7$ </td><td>26.569± 0.098</td><td> $2 1 . 2 2 8 \pm 0 . 3 4 0$ </td><td>26.125±0.134</td><td></td><td> $2 1 . 3 5 3 \pm 0 . 1 6 6$ </td><td> $1 5 . 8 8 8 \pm 1 . 5 0 3$ </td><td> $1 5 . 5 2 9 \pm 2 . 9 6 6$ </td></tr><tr><td>FE↓</td><td> $1 . 0 8 6 \pm 0 . 0 7 4$ </td><td>0.731 ± 0.003</td><td>1.136 ± 0.033</td><td>0.715±0.008</td><td>1.148 ± 0.009</td><td>0.748 ± 0.003</td><td></td><td>1.161 ± 0.010</td><td> $1 . 4 2 2 \pm 0 . 0 7 4$ </td><td> $1 . 4 4 2 \pm 0 . 1 3 3$ </td></tr><tr><td>TC (%) ↑</td><td> $6 6 . 2 0 \pm 3 . 0 1$ </td><td></td><td>70.23 ± 0.59</td><td></td><td>78.88±0.18</td><td>69.53± 0.99</td><td>79.63±0.39</td><td>64.85±3.06</td><td> $5 8 . 3 0 \pm 1 . 1 1 $ </td><td> $4 9 . 7 0 \pm 1 . 0 4$ </td></tr><tr><td rowspan="4">Dynamic</td><td>FS (%) ↑</td><td> $1 2 . 6 0 \pm 3 . 4 3$ </td><td> $\begin{array} { c } { 7 9 . 6 5 \pm { \bf 0 . 1 8 } } \\ { 3 5 . 3 0 \pm 0 . 5 1 } \end{array}$ </td><td>16.70 ± 0.80</td><td>33.60±0.43</td><td>16.30 ± 1.20</td><td>35.60±1.14</td><td>11.20 ± 3.39</td><td></td><td>5.30 ± 0.58</td><td> $0 . 6 0 \pm 0 . 6 0$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 6 9 4 \pm 0 . 6 0 0$ </td><td>27.643±0.090</td><td>21.364 ± 0.232</td><td>27.306±0.124</td><td>21.153 ± 0.367</td><td>27.499±0.146</td><td></td><td>20.613 ± 0.613</td><td> $1 3 . 9 7 0 \pm 0 . 8 3 9$ </td><td> $1 2 . 5 7 9 \pm 1 . 5 9 1$ </td></tr><tr><td>FE ↓</td><td> $1 . 0 0 3 \pm 0 . 0 7 9$ </td><td>0.599 ± 0.009</td><td> $1 . 1 2 5 \pm 0 . 0 1 0$ </td><td>0.610 ± 0.012</td><td> $1 . 1 1 9 \pm 0 . 0 1 5$ </td><td>0.610 ± 0.014</td><td></td><td> $0 . 9 6 3 \pm 0 . 0 7 7$ </td><td> $1 . 3 1 8 \pm 0 . 0 3 8$ </td><td> $1 . 0 2 2 \pm 0 . 1 3 9$ </td></tr><tr><td>TC (%) ↑</td><td> $6 6 . 8 3 \pm 3 . 6 8$ </td><td>78.90 ± 0.70</td><td>70.63 ± 0.10</td><td>78.93±0.48</td><td></td><td>70.68 ± 0.19</td><td>78.78±0.22</td><td> $7 0 . 4 5 \pm 0 . 0 5$ </td><td></td><td></td></tr><tr><td rowspan="4">Cascaded</td><td>FS (%) ↑</td><td>14.00 ± 3.50</td><td>33.70±1.79</td><td>17.50 ± 0.00</td><td>33.50±0.85</td><td></td><td>17.40 ± 0.10</td><td>33.00 ± 0.57</td><td>17.50 ± 0.00</td><td> $3 8 . 0 3 \pm 0 . 9 3$   $0 . 5 0 \pm 0 . 3 2$ </td><td> $2 7 . 9 3 \pm 5 . 7 4$   $0 . 2 0 \pm 0 . 2 0$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 8 7 2 \pm 0 . 7 5 3$ </td><td>27.563±0.259</td><td>21.529 ± 0.060</td><td>27.447±0.207</td><td></td><td>21.502 ± 0.048</td><td>27.581±0.059</td><td> $2 1 . 6 4 2 \pm 0 . 0 4 7$ </td><td> $9 . 7 4 5 \pm 0 . 8 6 1$ </td><td></td></tr><tr><td>FE↓</td><td>1.040 ± 0.085</td><td>0.577 ± 0.015</td><td>1.138 ± 0.004</td><td>0.590 ± 0.018</td><td>1.144 ± 0.008</td><td></td><td>0.578 ± 0.014</td><td> $1 . 1 2 7 \pm 0 . 0 0 2$ </td><td> $0 . 8 5 0 \pm 0 . 0 5 8$ </td><td> $6 . 8 7 5 \pm 2 . 6 6 8$ </td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td> $0 . 6 5 0 \pm 0 . 0 5 9$ </td></tr></table>

Observation. Across all five participation patterns, SPL improves TC, FS, and return while reducing FE relative to the matched Vanilla backbone. The improvement is consistent across IPPO, MAPPO, and HAPPO, indicating that the additional gains are not specific to a single PPO-based backbone.

## D.2. MPE-Advanced

Table D.2 reports the corresponding results in MPE-Advanced, where task assignments, spawn configurations, and future-collaborator identities vary across episodes.

Table D.2: Complete MPE-Advanced four-agent results at 2M environment steps. Values are mean ± standard error over five independent training seeds and 200 final evaluation episodes per seed. Bold marks the best reported mean in each participation–metric row.
<table><tr><td rowspan="2" colspan="2">Participation Metric</td><td colspan="2">IPPO</td><td colspan="2">MAPPO</td><td colspan="2">HAPPO</td><td colspan="3">Other baselines</td></tr><tr><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>MAE-MAPPO</td><td>Sable</td><td>SMPE²</td></tr><tr><td rowspan="4">Non-overlap</td><td>TC (%) ↑</td><td> $7 0 . 3 8 \pm 0 . 0 0$ </td><td> $7 7 . 9 5 \pm 0 . 1 9$ </td><td> $6 4 . 0 8 \pm 3 . 8 6$ </td><td> ${ \bf 7 8 . 2 3 \pm 0 . 1 6 }$ </td><td> $6 7 . 2 8 \pm 3 . 1 0$ </td><td> $7 8 . 0 3 \pm 0 . 1 6$ </td><td> $6 3 . 2 8 \pm 3 . 4 7$ </td><td> $6 4 . 9 0 \pm 1 . 3 6$ </td><td> $4 9 . 7 8 \pm 3 . 2 5$ </td></tr><tr><td>FS (%) ↑</td><td> $1 9 . 0 0 \pm 0 . 0 0$ </td><td> $3 0 . 5 0 \pm 0 . 6 5$ </td><td> $1 1 . 4 0 \pm 4 . 6 5$ </td><td> ${ \bf 3 1 . 9 0 \pm 0 . 6 2 }$ </td><td> $1 5 . 2 0 \pm 3 . 8 0$ </td><td> $3 1 . 1 0 \pm 0 . 5 3$ </td><td> $1 0 . 6 0 \pm 4 . 3 5$ </td><td> $1 0 . 8 0 \pm 1 . 4 1$ </td><td> $1 . 6 0 \pm 1 . 0 9$ </td></tr><tr><td>Return ↑</td><td> $2 1 . 4 8 5 \pm 0 . 1 3 5$ </td><td> $2 7 . 2 2 9 \pm 0 . 0 9 2$ </td><td> $1 9 . 9 8 6 \pm 0 . 8 4 7$ </td><td> ${ \bf 2 7 . 3 9 0 \pm 0 . 1 0 6 }$ </td><td> $2 0 . 7 5 4 \pm 0 . 6 4 5$ </td><td> $2 7 . 3 2 8 \pm 0 . 0 7 6$ </td><td> $1 9 . 9 3 9 \pm 0 . 7 9 8$ </td><td> $1 6 . 6 8 3 \pm 1 . 0 8 5$ </td><td> $1 0 . 7 3 0 \pm 2 . 2 6 9$ </td></tr><tr><td>FE↓</td><td> $1 . 1 2 5 \pm 0 . 0 0 0$ </td><td> $0 . 5 8 9 \pm 0 . 0 0 8$ </td><td> $0 . 9 8 9 \pm 0 . 0 8 3$ </td><td> $\mathbf { 0 . 5 7 9 \pm 0 . 0 0 7 }$ </td><td> $1 . 0 5 6 \pm 0 . 0 6 9$ </td><td> $0 . 5 8 0 \pm 0 . 0 0 6$ </td><td> $0 . 9 7 1 \pm 0 . 0 7 8$ </td><td> $1 . 3 6 9 \pm 0 . 0 5 7$ </td><td> $1 . 2 3 8 \pm 0 . 2 5 6$ </td></tr><tr><td rowspan="4">Overlap-25</td><td>TC (%) ↑</td><td> $6 5 . 6 5 \pm 3 . 5 3$ </td><td> ${ \bf 7 9 . 9 5 \pm 0 . 5 2 }$ </td><td> $6 8 . 2 8 \pm 1 . 5 0$ </td><td> $7 9 . 3 8 \pm 0 . 2 7$ </td><td> $7 0 . 9 3 \pm 0 . 5 6$ </td><td> $7 9 . 5 5 \pm 0 . 4 0$ </td><td> $7 1 . 4 0 \pm 0 . 2 3$ </td><td> $6 4 . 3 0 \pm 1 . 6 3$ </td><td> $4 7 . 8 3 \pm 9 . 0 2$ </td></tr><tr><td>FS (%) ↑</td><td> $1 2 . 6 0 \pm 4 . 2 8$ </td><td>35.50 ±0.57</td><td> $1 6 . 6 0 \pm 1 . 2 7$ </td><td> $3 3 . 8 0 \pm 0 . 7 3$ </td><td> $1 8 . 7 0 \pm 0 . 8 0$ </td><td> $3 4 . 5 0 \pm 1 . 2 1$ </td><td> $1 9 . 3 0 \pm 0 . 2 0$ </td><td> $1 0 . 5 0 \pm 2 . 2 0$ </td><td> $5 . 2 0 \pm 4 . 1 7$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 3 4 3 \pm 0 . 7 3 4$ </td><td> ${ \pm 7 . 5 3 6 \pm 0 . 2 3 5 }$ </td><td> $2 0 . 6 1 1 \pm 0 . 4 9 4$ </td><td> $2 7 . 2 4 4 \pm 0 . 1 3 9$ </td><td> $2 1 . 6 3 4 \pm 0 . 2 0 5$ </td><td>27.502±0.185</td><td> $2 1 . 9 6 7 \pm 0 . 1 2 0$ </td><td> $1 5 . 5 8 9 \pm 1 . 2 3 8$ </td><td> $1 3 . 1 5 2 \pm 2 . 7 1 8$ </td></tr><tr><td>FE↓</td><td> $1 . 0 1 1 \pm 0 . 0 7 7$ </td><td> $0 . 6 2 2 \pm 0 . 0 1 2$ </td><td> $1 . 1 0 1 \pm 0 . 0 2 7$ </td><td> $0 . 6 4 1 \pm 0 . 0 0 7$ </td><td> $1 . 1 2 6 \pm 0 . 0 1 6$ </td><td> $\mathbf { 0 . 6 1 1 \pm 0 . 0 1 2 }$ </td><td> $1 . 1 3 1 \pm 0 . 0 0 4$ </td><td> $1 . 4 2 7 \pm 0 . 0 6 6$ </td><td>0.901 ± 0.255</td></tr><tr><td rowspan="4">Overlap-50</td><td>TC (%) ↑</td><td> $7 1 . 3 0 \pm 0 . 2 2$ </td><td>78.63±0.23</td><td>68.58±3.05</td><td>78.33±0.40</td><td></td><td> $7 1 . 4 0 \pm 0 . 1 4$  78.68±0.25</td><td> $6 2 . 6 3 \pm 5 . 9 2$ </td><td> $6 1 . 8 8 \pm 1 . 4 8 $ </td><td> $5 2 . 0 8 \pm 8 . 2 5$ </td></tr><tr><td>FS (%) ↑</td><td>19.20 ± 0.30</td><td>31.90 ± 0.60</td><td>15.80 ± 3.70</td><td>31.10±1.11</td><td>19.40 ± 0.10</td><td>32.10±0.76</td><td>12.00 ± 4.60</td><td> $9 . 1 0 \pm 1 . 2 0$ </td><td>6.10 ± 4.23</td></tr><tr><td>Return ↑</td><td> $2 1 . 6 6 9 \pm 0 . 0 8 3$ </td><td>26.481±0.084</td><td>21.005 ± 0.701</td><td>26.358±0.171</td><td> $2 1 . 5 6 7 \pm 0 . 0 9 0$ </td><td>26.446±0.097</td><td> $1 9 . 7 4 3 \pm 1 . 4 4 3$ </td><td>13.716 ± 1.266</td><td> $1 3 . 0 1 8 \pm 3 . 4 1 2$ </td></tr><tr><td>FE↓</td><td> $1 . 1 2 4 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 7 1 3 \pm 0 . 0 0 7 }$ </td><td> $1 . 0 6 9 \pm 0 . 0 6 6$ </td><td>0.715 ± 0.010</td><td> $1 . 1 4 3 \pm 0 . 0 0 6$ </td><td>0.721 ± 0.005</td><td> $0 . 9 5 8 \pm 0 . 1 1 5$ </td><td> $1 . 5 2 5 \pm 0 . 0 5 9$ </td><td> $1 . 1 3 9 \pm 0 . 0 6 7$ </td></tr><tr><td rowspan="4">Dynamic</td><td>TC (%) ↑</td><td> $6 6 . 6 3 \pm 3 . 6 6$ </td><td> $7 8 . 2 3 \pm 0 . 2 9$ </td><td> $6 7 . 0 5 \pm 3 . 2 0$ </td><td></td><td>79.08±0.54</td><td>78.80±0.18</td><td>67.68 ± 1.76</td><td>57.60 ± 1.30</td><td> $4 3 . 1 0 \pm 6 . 6 1$ </td></tr><tr><td>FS (%) ↑</td><td> $1 4 . 7 0 \pm 3 . 6 8$ </td><td>32.30 ± 0.64</td><td> $1 4 . 7 0 \pm 3 . 6 8$ </td><td>34.10 ± 1.38</td><td> $\begin{array} { c } { 6 1 . 7 3 \pm 5 . 5 5 } \\ { 1 1 . 1 0 \pm 4 . 5 3 } \end{array}$ </td><td>34.20±0.64</td><td> $1 5 . 6 0 \pm 2 . 4 2$ </td><td> $6 . 5 0 \pm 1 . 1 0$ </td><td> $0 . 0 0 \pm 0 . 0 0$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 5 0 9 \pm 0 . 8 9 3$ </td><td> $2 7 . 0 0 6 \pm 0 . 1 3 1$ </td><td> $2 0 . 7 7 0 \pm 0 . 6 8 1$ </td><td></td><td>27.312±0.258  $1 9 . 8 6 7 \pm 1 . 2 1 5$ </td><td>27.330± 0.081</td><td> $2 0 . 9 3 6 \pm 0 . 4 2 0$ </td><td> $1 4 . 0 2 7 \pm 1 . 0 1 8$ </td><td> $1 3 . 6 7 8 \pm 2 . 5 7 7$ </td></tr><tr><td>FE↓</td><td> $1 . 0 3 7 \pm 0 . 0 7 2$ </td><td>0.608± 0.005</td><td> $1 . 0 4 7 \pm 0 . 0 6 4$ </td><td>0.608±0.013</td><td> $0 . 9 1 5 \pm 0 . 1 2 3$ </td><td>0.596 ± 0.012</td><td> $1 . 0 4 7 \pm 0 . 0 3 8$ </td><td> $1 . 2 7 2 \pm 0 . 0 5 2$ </td><td> $0 . 6 0 8 \pm 0 . 0 8 3$ </td></tr><tr><td rowspan="4">Cascaded</td><td>TC (%) ↑</td><td> $6 7 . 6 8 \pm 2 . 5 8$ </td><td> $7 6 . 8 0 \pm 0 . 1 3$ </td><td></td><td> $6 7 . 2 8 \pm 3 . 1 0$ </td><td>77.00 ± 0.33</td><td>69.28± 1.10</td><td>77.23± 0.18</td><td> $6 6 . 8 3 \pm 3 . 0 2$ </td><td> $6 2 . 7 3 \pm 0 . 9 8$ </td><td> $5 6 . 7 0 \pm 5 . 5 0$ </td></tr><tr><td>FS (%) ↑</td><td> $1 5 . 9 0 \pm 3 . 1 0$ </td><td>29.00 ± 0.45</td><td></td><td>15.20 ± 3.80</td><td>29.20 ± 0.68</td><td>17.80 ± 1.20</td><td>30.70± 0.72</td><td>14.80 ± 3.72</td><td> $8 . 3 0 \pm 1 . 6 6$ </td><td> $7 . 1 0 \pm 3 . 7 2$ </td></tr><tr><td>Return ↑</td><td> $2 0 . 8 9 5 \pm 0 . 6 3 8$ </td><td> $2 5 . 9 7 6 \pm 0 . 0 7 2$ </td><td></td><td> $2 0 . 6 2 3 \pm 0 . 6 6 1$ </td><td>25.988±0.143</td><td>21.038 ± 0.238</td><td>26.033±0.100</td><td>20.707 ± 0.662</td><td> $1 4 . 7 0 6 \pm 0 . 8 1 1$ </td><td> $1 8 . 6 5 0 \pm 1 . 4 2 3$ </td></tr><tr><td>FE↓</td><td> $1 . 0 6 5 \pm 0 . 0 5 5$ </td><td>0.734±0.006</td><td></td><td>1.056 ± 0.069</td><td>0.742 ± 0.007</td><td>1.101 ± 0.024</td><td>0.741 ± 0.007</td><td>1.043 ± 0.067</td><td>1.457 ± 0.035</td><td> $0 . 7 9 5 \pm 0 . 1 0 8$ </td></tr></table>

Observation. The same pattern persists under the additional structural variation of MPE-Advanced. SPL improves all four reported metrics for every matched backbone–participation comparison, while the three SPL backbones remain close in final performance.

## D.3. RWARE-Base

Table D.3 reports the complete RWARE-Base results across the five participation patterns.

Table D.3: Complete RWARE-Base four-agent results at 2M environment steps. Values are mean ± standard error over five independent training seeds and 200 final evaluation episodes per seed. Bold marks the best reported mean in each participation–metric row.
<table><tr><td rowspan="2" colspan="2">Participation Metric</td><td colspan="2">IPPO</td><td colspan="2">MAPPO</td><td colspan="2">HAPPO</td><td colspan="3">Other baselines</td></tr><tr><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>MAE-MAPPO</td><td>Sable</td><td>SMPE2</td></tr><tr><td rowspan="4">Non-overlap</td><td>TC (%)↑</td><td>66.45 ± 5.86</td><td> $8 6 . 8 3 \pm 0 . 4 0$ </td><td> $7 1 . 3 8 \pm 2 . 2 7$ </td><td>86.70±0.49</td><td> $7 3 . 1 8 \pm 1 . 5 0$ </td><td>86.93±0.26</td><td> $7 3 . 4 8 \pm 1 . 6 9$ </td><td>71.80 ± 0.98</td><td> $6 6 . 0 5 \pm 3 . 2 9$ </td></tr><tr><td>FS (%) ↑</td><td> $1 4 . 5 0 \pm 6 . 2 7$ </td><td> $5 3 . 8 0 \pm 1 . 6 8$ </td><td> $1 9 . 5 0 \pm 4 . 1 4$ </td><td> $5 3 . 7 0 \pm 1 . 4 6$ </td><td>19.20 ± 3.09</td><td> ${ \bf 5 4 . 0 0 \pm 0 . 5 7 }$ </td><td> $2 1 . 3 0 \pm 2 . 2 8$ </td><td> $2 0 . 0 0 \pm 1 . 3 0$ </td><td> $1 4 . 1 0 \pm 3 . 6 1$ </td></tr><tr><td>Return ↑</td><td> $2 . 3 4 2 \pm 0 . 2 2 6$ </td><td> $3 . 2 9 2 \pm 0 . 0 1 8$ </td><td> $2 . 5 4 8 \pm 0 . 1 0 1$ </td><td> $3 . 2 8 6 \pm 0 . 0 2 3$ </td><td> $2 . 6 1 8 \pm 0 . 0 7 0$ </td><td> $\mathbf { 3 . 2 9 7 \pm 0 . 0 1 0 }$ </td><td> $2 . 6 4 1 \pm 0 . 0 7 2$ </td><td> $2 . 4 7 1 \pm 0 . 0 4 9$ </td><td> $2 . 3 0 1 \pm 0 . 1 6 4$ </td></tr><tr><td>FE↓</td><td> $0 . 8 4 8 \pm 0 . 0 8 8$ </td><td> $0 . 4 9 4 \pm 0 . 0 1 3$ </td><td> $1 . 0 7 5 \pm 0 . 1 1 0$ </td><td> $0 . 5 0 3 \pm 0 . 0 1 9$ </td><td>1.064 ± 0.057</td><td>0.483 ± 0.011</td><td> $1 . 0 0 3 \pm 0 . 0 6 8$ </td><td> $1 . 1 2 8 \pm 0 . 0 3 9$ </td><td> $1 . 3 5 8 \pm 0 . 1 3 1$ </td></tr><tr><td rowspan="4">Overlap-25</td><td>TC (%) ↑</td><td> $8 1 . 1 0 \pm 3 . 4 1$ </td><td> ${ \bf 8 9 . 1 5 \pm 0 . 1 6 }$ </td><td> $7 4 . 2 5 \pm 3 . 4 1$ </td><td> $8 9 . 0 8 \pm 0 . 3 5$ </td><td> $7 0 . 1 5 \pm 2 . 9 3$ </td><td>88.35±0.50</td><td>72.50 ± 2.86</td><td> $7 6 . 7 0 \pm 1 . 1 8$ </td><td> $6 7 . 2 8 \pm 2 . 8 7$ </td></tr><tr><td>FS (%) ↑</td><td>38.60 ± 7.30</td><td>61.00 ± 0.32</td><td>25.90 ± 7.18</td><td> $6 0 . 6 0 \pm 0 . 9 4$ </td><td>16.60 ± 4.20</td><td> $5 8 . 8 0 \pm 1 . 9 5$ </td><td>19.20 ± 5.35</td><td>28.10 ± 3.03</td><td>16.00 ± 2.83</td></tr><tr><td>Return ↑</td><td>2.960 ± 0.157</td><td>3.404 ± 0.008</td><td>2.667 ± 0.157</td><td> $3 . 4 0 1 \pm 0 . 0 1 8$ </td><td>2.500 ± 0.131</td><td> $3 . 3 6 9 \pm 0 . 0 2 3$ </td><td> $2 . 6 3 3 \pm 0 . 1 0 5$ </td><td>2.720 ± 0.056</td><td> $2 . 3 5 8 \pm 0 . 1 4 5$ </td></tr><tr><td>FE↓</td><td>0.728 ± 0.142</td><td> $0 . 4 1 1 \pm 0 . 0 1 1$ </td><td>0.935 ± 0.165</td><td> $\mathbf { 0 . 4 1 0 \pm 0 . 0 1 8 }$ </td><td>1.075 ± 0.092</td><td>0.424 ± 0.014</td><td> $0 . 8 7 7 \pm 0 . 1 1 4$ </td><td> $0 . 9 3 2 \pm 0 . 0 4 7$ </td><td> $1 . 3 0 9 \pm 0 . 1 1 5$ </td></tr><tr><td rowspan="4">Overlap-50</td><td>TC (%) ↑</td><td> $8 2 . 0 0 \pm 1 . 1 2$ </td><td>89.00 ± 0.19</td><td>74.73 ± 3.35</td><td> $8 8 . 5 8 \pm 0 . 2 8$ </td><td>70.28 ± 2.41</td><td>88.80±0.23</td><td> $7 7 . 5 3 \pm 0 . 7 2$ </td><td> $7 7 . 2 3 \pm 1 . 9 7$ </td><td> $6 1 . 9 5 \pm 4 . 0 9$ </td></tr><tr><td>FS (%) ↑</td><td> $4 1 . 8 0 \pm 1 . 7 4$ </td><td>59.40±0.68</td><td>24.60 ± 6.55</td><td> $5 9 . 3 0 \pm 0 . 9 4$ </td><td>15.20 ± 3.03</td><td> ${ \bf 5 9 . 7 0 \pm 0 . 8 0 }$ </td><td> $3 0 . 2 0 \pm 1 . 3 2$ </td><td> $3 1 . 2 0 \pm 4 . 4 7$ </td><td>10.10 ± 4.15</td></tr><tr><td>Return ↑</td><td> $3 . 0 0 8 \pm 0 . 0 5 2$ </td><td>3.397 ± 0.010</td><td>2.706 ± 0.153</td><td> $3 . 3 7 5 \pm 0 . 0 1 3$ </td><td>2.487 ± 0.112</td><td>3.388± 0.011</td><td> $2 . 8 3 4 \pm 0 . 0 3 3$ </td><td> $2 . 7 4 1 \pm 0 . 0 9 9$ </td><td>2.143 ± 0.175</td></tr><tr><td>FE↓</td><td> $0 . 6 8 6 \pm 0 . 0 3 7$ </td><td>0.414 ± 0.010</td><td>0.979 ± 0.144</td><td> $0 . 4 3 2 \pm 0 . 0 0 9$ </td><td> $1 . 1 8 4 \pm 0 . 0 9 9$ </td><td>0.415 ± 0.009</td><td> $0 . 8 5 8 \pm 0 . 0 3 9$ </td><td> $0 . 9 1 1 \pm 0 . 0 7 9$ </td><td> $1 . 3 2 2 \pm 0 . 1 5 0$ </td></tr><tr><td rowspan="4">Dynamic</td><td>TC (%) ↑</td><td> $7 2 . 8 8 \pm 1 . 1 9$ </td><td> ${ \pm 0 . 8 8 \pm 0 . 2 7 }$ </td><td> $6 5 . 3 5 \pm 5 . 4 6$ </td><td> $8 8 . 7 0 \pm 0 . 3 0$ </td><td>68.80 ± 4.06</td><td>88.40 ±0.42</td><td>72.28 ± 1.51</td><td> $7 4 . 6 3 \pm 1 . 2 2$ </td><td> $6 2 . 0 3 \pm 4 . 2 9$ </td></tr><tr><td>FS (%) ↑</td><td> $1 9 . 3 0 \pm 1 . 5 6$ </td><td> ${ \bf 6 0 . 1 0 \pm 0 . 8 1 }$ </td><td> $1 4 . 5 0 \pm 5 . 1 4$ </td><td> $5 8 . 7 0 \pm 1 . 2 0 $ </td><td> $1 6 . 3 0 \pm 4 . 8 9$ </td><td>58.40±1.37</td><td>18.90 ± 2.62</td><td> $2 4 . 7 0 \pm 2 . 7 2$ </td><td> $1 0 . 2 0 \pm 4 . 0 7$ </td></tr><tr><td>Return ↑</td><td> $2 . 5 7 9 \pm 0 . 0 4 1$ </td><td> $\mathbf { 3 . 3 8 9 \pm 0 . 0 1 3 }$ </td><td> $2 . 3 0 4 \pm 0 . 2 1 1$ </td><td> $3 . 3 8 5 \pm 0 . 0 1 4$ </td><td>2.425 ± 0.199</td><td> $3 . 3 6 8 \pm 0 . 0 2 0$ </td><td>2.572 ± 0.072</td><td> $2 . 6 1 6 \pm 0 . 0 5 8$ </td><td> $2 . 0 9 5 \pm 0 . 2 1 5$ </td></tr><tr><td>E↓</td><td>0.910 ± 0.118</td><td> $0 . 4 2 3 \pm 0 . 0 0 7$ </td><td>1.119 ± 0.129</td><td> $\mathbf { 0 . 4 1 5 \pm 0 . 0 0 6 }$ </td><td>1.166 ± 0.203</td><td>0.438 ± 0.014</td><td>1.106 ± 0.061 1.015 ± 0.049</td><td></td><td> $1 . 5 1 9 \pm 0 . 1 7 2$ </td></tr><tr><td rowspan="4">Cascaded</td><td>TC (%) ↑</td><td> $6 6 . 8 0 \pm 1 . 9 8$ </td><td> $8 6 . 2 5 \pm 0 . 3 8$ </td><td> $6 7 . 5 3 \pm 2 . 0 5$ </td><td> ${ \bf 8 6 . 7 5 \pm 0 . 5 0 }$ </td><td> $6 6 . 5 8 \pm 2 . 2 4$ </td><td> $8 5 . 9 5 \pm 0 . 7 9$ </td><td> $7 1 . 6 3 \pm 0 . 7 3$ </td><td> $7 6 . 8 3 \pm 0 . 9 3$ </td><td> $6 4 . 1 8 \pm 2 . 9 4$ </td></tr><tr><td>FS (%) ↑</td><td>13.20 ± 2.83</td><td>52.40±1.07</td><td> $1 1 . 8 0 \pm 3 . 6 4$ </td><td> ${ \pm } 3 . 6 0 \pm 1 . 0 2$ </td><td>11.90 ± 2.64</td><td>51.50±1.67</td><td>18.70 ± 0.97 28.80 ± 1.97</td><td></td><td>11.00 ± 3.67</td></tr><tr><td>Return ↑</td><td>2.346 ± 0.097</td><td>3.281 ± 0.014</td><td>2.417 ± 0.0743.299 ± 0.023</td><td></td><td>2.322 ± 0.110</td><td>3.269±0.032</td><td>2.573 ± 0.031 2.721 ± 0.047</td><td></td><td> $2 . 2 0 7 \pm 0 . 1 4 6$ </td></tr><tr><td>FE↓</td><td>1.298 ± 0.079</td><td>0.456 ± 0.012</td><td>1.095 ± 0.141</td><td> $0 . 4 6 6 \pm 0 . 0 1 1$ </td><td>1.333 ± 0.088</td><td>0.448 ± 0.008</td><td>1.135 ± 0.029 0.927 ± 0.037</td><td></td><td> $1 . 4 3 3 \pm 0 . 1 1 8$ </td></tr></table>

Observation. SPL produces larger gains in RWARE than in MPE, particularly in FS and FE. The increase in full success indicates that higher task completion is accompanied by more episodes in which the cooperative objective is completed in full.

## D.4. RWARE-Advanced

Table D.4 reports the results in RWARE-Advanced, which additionally includes task permutation, a shared candidate pool, and dynamic task claims.

Table D.4: Complete RWARE-Advanced four-agent results at 2M environment steps. Values are mean ± standard error over five independent training seeds and 200 final evaluation episodes per seed. Bold marks the best reported mean in each participation–metric row.
<table><tr><td rowspan="2" colspan="2">Participation Metric</td><td colspan="2">IPPO</td><td colspan="2">MAPPO</td><td colspan="2">HAPPO</td><td colspan="3">Other baselines</td></tr><tr><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>Vanilla</td><td>SPL</td><td>MAE-MAPPO</td><td>Sable</td><td>SMPE2</td></tr><tr><td rowspan="4">Non-overlap</td><td>TC (%) ↑</td><td>52.08 ± 1.37</td><td>79.93±0.59</td><td>58.63 ± 2.31</td><td>80.03± 0.23</td><td>57.58 ± 3.44</td><td>79.55±0.79</td><td>58.50 ± 2.20</td><td>50.70 ± 0.32</td><td>45.70 ± 0.25</td></tr><tr><td>FS (%) ↑</td><td>2.60 ± 0.58</td><td>40.60 ± 0.87</td><td>4.70 ± 0.93</td><td>42.20±1.44</td><td>5.70 ± 1.79</td><td>38.70±1.71</td><td>4.30 ± 1.19</td><td>1.60 ± 0.19</td><td>1.20 ± 0.62</td></tr><tr><td>Return ↑</td><td>1.654 ± 0.059</td><td>2.950 ± 0.026</td><td>1.972 ± 0.096</td><td>2.951 ± 0.010</td><td>1.891 ± 0.144</td><td>2.928± 0.030</td><td>1.951 ± 0.099</td><td>1.510 ± 0.017</td><td>1.346 ± 0.022</td></tr><tr><td>FE↓</td><td>1.573 ± 0.043</td><td>0.577 ± 0.013</td><td>1.174 ± 0.098</td><td>0.583 ± 0.011</td><td>1.386 ± 0.115</td><td>0.599 ± 0.020</td><td>1.330 ± 0.119</td><td>1.660 ± 0.019</td><td>1.882 ± 0.025</td></tr><tr><td rowspan="4">Overlap-25</td><td>TC (%) ↑</td><td>64.18 ± 1.62</td><td>83.23 ± 0.47</td><td>68.83 ± 1.40</td><td>83.65±0.32</td><td>63.28± 1.98</td><td>83.73± 0.48</td><td>64.70 ± 1.81</td><td>54.75 ± 0.44</td><td>47.68 ± 0.66</td></tr><tr><td>FS (%) ↑</td><td>7.00 ± 0.84</td><td>42.30±0.94</td><td>11.60 ± 1.27</td><td>44.50±1.01</td><td>7.60 ± 1.64</td><td>43.50±2.11</td><td>8.40 ± 1.71</td><td>1.80 ± 0.34</td><td>1.00 ± 0.42</td></tr><tr><td>Return ↑</td><td>2.175 ± 0.062</td><td>3.069±0.023</td><td>2.360 ± 0.066</td><td>3.092 ± 0.016</td><td>2.122 ± 0.086</td><td>3.095±0.020</td><td>2.193 ± 0.087</td><td></td><td>1.636 ± 0.0211.331 ± 0.045</td></tr><tr><td>FE↓</td><td>1.316 ± 0.054</td><td>0.590 ± 0.017</td><td>1.159 ± 0.061</td><td>0.567 ± 0.018</td><td>1.368 ± 0.073</td><td>0.566 ± 0.011</td><td>1.256 ± 0.100</td><td>1.770 ± 0.013</td><td>2.007 ± 0.039</td></tr><tr><td rowspan="4">Overlap-50</td><td>TC (%) ↑</td><td>70.35± 1.28</td><td>84.55±0.69</td><td>68.05 ± 4.14</td><td>84.45±0.33</td><td>67.50 ± 2.12</td><td>84.43±0.56</td><td>72.88 ±1.05</td><td>58.88 ± 0.77</td><td>47.05 ± 0.38</td></tr><tr><td>FS (%) ↑</td><td>13.40 ± 1.51</td><td>45.90±1.92</td><td>12.40 ± 3.67</td><td>45.20 ± 0.77</td><td>11.70 ± 3.14</td><td>45.10±1.78</td><td>17.50 ± 1.31</td><td>4.90 ± 0.33</td><td>0.80 ± 0.49</td></tr><tr><td>Return ↑</td><td>2.437 ± 0.056</td><td>3.124 ± 0.034</td><td>2.344 ± 0.184</td><td>3.122 ± 0.017</td><td>2.327 ± 0.104</td><td>3.120 ± 0.024</td><td>2.566 ± 0.051</td><td></td><td>1.836 ± 0.039 1.340 ± 0.015</td></tr><tr><td>FE↓</td><td>1.118 ± 0.056</td><td>0.587 ± 0.026</td><td>1.209 ± 0.156</td><td>0.574 ± 0.015</td><td>1.141 ± 0.124</td><td>0.574 ± 0.011</td><td>1.002 ± 0.037</td><td>1.629 ± 0.035</td><td>2.054 ± 0.020</td></tr><tr><td rowspan="4">Dynamic</td><td>TC (%) ↑</td><td>66.60 ± 1.95</td><td>81.30 ± 0.34</td><td>62.55 ± 2.83</td><td>82.50 ± 0.43</td><td>63.23 ± 0.83</td><td>82.20 ± 0.16</td><td>65.90 ± 2.33</td><td>56.30 ± 0.49</td><td>47.55 ± 0.40</td></tr><tr><td>FS (%) ↑</td><td>11.60 ± 2.43</td><td>39.30±0.64</td><td>8.50 ± 2.32</td><td>42.80 ± 0.72</td><td>7.90 ± 0.93</td><td>41.20 ± 0.78</td><td>11.10 ± 1.87</td><td>3.80 ± 0.64</td><td>0.60 ± 0.40</td></tr><tr><td>Return ↑</td><td>2.282 ± 0.084</td><td>2.995±0.014</td><td>2.090 ± 0.124</td><td>3.048 ± 0.018</td><td>2.162 ± 0.041</td><td>3.031 ± 0.006</td><td>2.269 ± 0.098</td><td></td><td>1.713 ± 0.024 1.353 ± 0.035</td></tr><tr><td>FE↓</td><td>1.122 ± 0.077</td><td>0.590 ± 0.004</td><td>1.351 ± 0.114</td><td>0.564 ± 0.014</td><td>1.207 ± 0.046</td><td>0.586 ± 0.015</td><td>1.147 ± 0.086</td><td>1.705 ± 0.025</td><td>2.054 ± 0.031</td></tr><tr><td rowspan="4">Cascaded</td><td>TC (%) ↑</td><td>58.95±1.70</td><td>85.40 ± 0.42</td><td>65.55±1.76</td><td>85.98± 0.44</td><td>66.03 ± 2.06</td><td>84.90 ± 0.51</td><td>64.18 ± 3.98</td><td>54.90 ± 1.08</td><td>47.03 ± 0.55</td></tr><tr><td>FS (%) ↑</td><td>4.10 ± 0.87</td><td>46.00 ±1.17</td><td>8.60 ± 1.76</td><td>48.30±1.54</td><td>9.50 ± 2.13</td><td>44.80±1.85</td><td>9.40 ± 2.50</td><td>2.70 ± 0.37</td><td>0.60 ± 0.19</td></tr><tr><td>Return ↑</td><td>1.936 ± 0.076</td><td>3.153 ± 0.021</td><td>2.194 ± 0.078</td><td>3.182 ± 0.021</td><td>2.226 ± 0.093</td><td>3.126± 0.026</td><td>2.150 ± 0.167</td><td></td><td>1.625 ± 0.054 1.339 ± 0.035</td></tr><tr><td>FE↓</td><td>1.539 ± 0.115</td><td>0.573 ± 0.017</td><td>1.368 ± 0.070</td><td>0.547 ± 0.014</td><td>1.317 ± 0.077</td><td>0.599 ± 0.021</td><td>1.331 ± 0.130</td><td></td><td>1.804 ± 0.043 2.119 ± 0.022</td></tr></table>

Observation. The greatest improvements appear in RWARE-Advanced, where downstream decisions depend more strongly on the available task-relevant evidence. SPL increases TC, FS, and return and reduces FE across every matched backbone and participation pattern.

## D.5. Additional Isaac Lab Metrics

Table D.5 reports the complete Isaac Lab evaluation, including task completion (TC), full success (FS), episode return, failed execution (FE), map coverage, task discovery, and wall-penetration steps. These additional diagnostics complement the TC results reported in the main paper and provide a broader view of the behaviors induced by SPL-MAPPO under embodied UAV–UGV execution.

Task performance. SPL-MAPPO achieves higher TC, FS, and return in all four team-size–participation settings. For the two-agent system, TC increases from 31.50% to 61.58% under Non-overlap and from 36.67% to 59.58% under Dynamic participation. The corresponding FS gains are also substantial, increasing from 5.17% to 35.17% and from 9.00% to 34.50%. The same pattern persists with four agents: TC increases from 55.79% to 76.38% under Non-overlap and from 58.46% to 79.25% under Dynamic participation, while FS increases from 11.83% to 39.83% and from 12.17% to 44.83%. Episode return follows the same direction in all four settings.

Table D.5: Isaac Lab evaluation metrics. Values are mean ± standard error over three independent training seeds and 200 final evaluation episodes per seed.
<table><tr><td rowspan="2">Agents Metric</td><td rowspan="2"></td><td colspan="2">Non-overlap</td><td colspan="2">Dynamic</td></tr><tr><td>MAPPO</td><td>SPL-MAPPO</td><td>MAPPO</td><td>SPL-MAPPO</td></tr><tr><td rowspan="7">2</td><td> $\mathrm { T C } \left( \% \right) \uparrow$ </td><td> $3 1 . 5 0 \pm 5 . 9 7$ </td><td> ${ \bf 6 1 . 5 8 \pm 0 . 6 7 }$ </td><td> $3 6 . 6 7 \pm 0 . 7 1$ </td><td> ${ \pm \bf { 5 9 . 5 8 \pm 1 . 7 9 } }$ </td></tr><tr><td> $\mathrm { F S } \ ( \% ) \ \uparrow$ </td><td> $5 . 1 7 \pm 3 . 0 3$ </td><td> ${ \bf 3 5 . 1 7 \pm 0 . 7 3 }$ </td><td> $9 . 0 0 \pm 0 . 7 6$ </td><td> ${ \pm } 4 . 5 0 \pm 2 . 2 9$ </td></tr><tr><td> $\mathrm { R e t u r n } \uparrow$ </td><td> $0 . 6 4 8 \pm 0 . 1 4 7$ </td><td> $\mathbf { 1 . 5 4 0 \pm 0 . 0 1 6 }$ </td><td> $0 . 7 8 5 \pm 0 . 0 2 2$ </td><td> $\mathbf { 1 . 4 9 4 \pm 0 . 0 5 9 }$ </td></tr><tr><td> $\mathrm { F E \downarrow }$ </td><td> $\mathbf { 0 . 6 3 2 \pm 0 . 0 5 7 }$ </td><td> $0 . 8 2 2 \pm 0 . 0 0 9$ </td><td> $0 . 7 9 8 \pm 0 . 0 0 3$ </td><td> $\mathbf { 0 . 7 6 0 \pm 0 . 0 3 8 }$ </td></tr><tr><td>Map coverage  $( \% ) \uparrow$ </td><td> $3 6 . 2 1 \pm 1 . 6 9$ </td><td> ${ \bf 4 4 . 0 3 \pm 0 . 1 7 }$ </td><td> $3 3 . 2 1 \pm 1 . 6 0$ </td><td> ${ \pm } 4 . 0 3 \pm 1 . 0 7$ </td></tr><tr><td>Tasks discovered ↑</td><td> $0 . 2 3 8 \pm 0 . 0 0 9$ </td><td> $\mathbf { 0 . 2 8 7 \pm 0 . 0 0 3 }$ </td><td> $0 . 1 5 0 \pm 0 . 0 2 0$ </td><td> $\mathbf { 0 . 2 6 1 } \pm \mathbf { 0 . 0 1 6 }$ </td></tr><tr><td>Wall-penetration steps ↓</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr><tr><td rowspan="7">4</td><td>TC (%) ↑</td><td> $5 5 . 7 9 \pm 3 . 2 3$ </td><td> ${ \bf 7 6 . 3 8 \pm 1 . 5 2 }$ </td><td> $5 8 . 4 6 \pm 3 . 4 0$ </td><td> $\pm \mathbf { 9 . 2 5 \pm 1 . 3 0 }$ </td></tr><tr><td> $\mathrm { F S } \ ( \% ) \ \uparrow$ </td><td> $1 1 . 8 3 \pm 4 . 6 9$ </td><td> ${ \bf 3 9 . 8 3 \pm 3 . 2 2 }$ </td><td> $1 2 . 1 7 \pm 4 . 2 1$ </td><td> ${ \pm } 4 . 8 3 \pm 1 . 1 7$ </td></tr><tr><td> $\mathrm { R e t u r n } \uparrow$ </td><td> $2 . 2 3 5 \pm 0 . 1 7 2$ </td><td> $\mathbf { 3 . 3 3 0 \pm 0 . 0 9 2 }$ </td><td> $2 . 3 4 3 \pm 0 . 1 7 4$ </td><td> ${ \bf 3 . 4 9 9 \pm 0 . 0 6 4 }$ </td></tr><tr><td>FE↓</td><td> $\mathbf { 1 . 1 0 0 \pm 0 . 0 1 6 }$ </td><td> $1 . 1 6 7 \pm 0 . 0 2 7$ </td><td> ${ \bf 1 . 1 6 0 \pm 0 . 0 2 3 }$ </td><td> $1 . 1 9 3 \pm 0 . 0 2 0$ </td></tr><tr><td>Map coverage (%)↑</td><td> ${ \bf 6 7 . 9 6 \pm 1 . 4 5 }$ </td><td> $5 4 . 9 0 \pm 0 . 3 6$ </td><td> ${ \bf 6 9 . 2 5 \pm 1 . 5 9 }$ </td><td> $5 3 . 8 4 \pm 0 . 7 6$ </td></tr><tr><td>Tasks discovered ↑</td><td> $\mathbf { 0 . 4 7 0 \pm 0 . 0 2 2 }$ </td><td> $0 . 2 0 0 \pm 0 . 0 0 7$ </td><td> $\mathbf { 0 . 5 1 2 \pm 0 . 0 4 1 }$ </td><td> $0 . 1 9 0 \pm 0 . 0 0 9$ </td></tr><tr><td>Wall-penetration steps ↓</td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td><td> $0 . 0 0 0 \pm 0 . 0 0 0$ </td></tr></table>

Execution and exploration diagnostics. FE does not change uniformly with SPL. In the two-agent setting it is higher under Non-overlap and slightly lower under Dynamic participation, while in the four-agent setting it is slightly higher under both schedules. These absolute FE values occur under substantially diferent exploration and task-discovery behavior and should therefore be interpreted together with those diagnostics.

The exploration pattern also difers with team size. With two agents, SPL-MAPPO attains greater map coverage and discovers more tasks under both participation schedules. For example, under Dynamic participation, map coverage increases from 33.21% to 44.03%, while task discovery increases from 0.150 to 0.261. This pattern is consistent with broader evidence acquisition. With four agents, SPL-MAPPO exhibits the opposite exploration pattern: map coverage decreases from 67.96% to 54.90% under Nonoverlap and from 69.25% to 53.84% under Dynamic participation, while task discovery also decreases. Despite this reduced exploration, SPL-MAPPO achieves higher TC, FS, and return in both four-agent settings. This is consistent with more selective acquisition and/or more efective use of the available task-relevant evidence, although the reported diagnostics do not separate these efects causally.

No wall-penetration steps are observed for either method in any of the four settings, indicating that the performance diferences reported above are not accompanied by violations of this physical-safety diagnostic.

## D.6. Prior-Misspecification Robustness

SPL uses task-family validity priors as part of the structured model for prospective acquisition supervision. We therefore evaluate how sensitive SPL is to misspecification of these priors. The true environment distribution is kept unchanged, and only the validity priors assumed by the acquisition model are perturbed. Let

$$
\mathbf { p } = ( 0 . 8 0 , 0 . 8 0 , 0 . 6 5 , 0 . 6 5 , 0 . 3 5 , 0 . 3 5 , 0 . 2 0 , 0 . 2 0 )
$$

denote the nominal prior vector. For misspecification level $\lambda \in [ 0 , 1 ]$ , the acquisition model instead uses

$$
{ \widehat { \bf p } } _ { \lambda } = ( 1 - \lambda ) { \bf p } + \lambda ( 1 - { \bf p } ) .\tag{D.1}
$$

Thus, $\lambda = 0$ recovers the nominal priors, $\lambda = 0 . 5$ produces an uninformative prior of 0.5 for every task, and $\lambda = 1$ fully reverses the nominal prior ordering. All other structured information, including the fixed-valid-task constraint, probe observation model, task utilities, environment dynamics, and receiver learning procedure, remains unchanged.

We evaluate SPL-MAPPO under Non-overlap participation in MPE-Advanced and RWARE-Advanced. Each misspecified condition uses five independent training seeds, two million environment steps, and 200 final evaluation episodes per seed, matching the primary four-agent protocol. The $\lambda = 0$ values correspond to the nominal SPL-MAPPO results reported in the main experiments; the $\lambda \in \{ 0 . 2 5 , 0 . 5 0 , 1 . 0 0 \}$ conditions comprise 30 additional training runs. In all misspecified runs, the environment continues to sample tasks from the nominal prior distribution in p.

Table D.6: Sensitivity of SPL-MAPPO to misspecified validity priors under Non-overlap participation. Values are mean ± standard error over five training seeds and 200 final evaluation episodes per seed. The environment always uses the nominal prior distribution; only the acquisition model uses $\widehat { \mathbf { p } } _ { \lambda }$
<table><tr><td>Environment</td><td>λ</td><td>TC (%) ↑</td><td>FS (%) ↑</td><td>Return ↑</td><td>FE↓</td></tr><tr><td rowspan="4">MPE-Advanced</td><td>0.00</td><td> $7 8 . 2 3 \pm 0 . 1 6$ </td><td> $3 1 . 9 0 \pm 0 . 6 2$ </td><td> $2 7 . 3 9 0 \pm 0 . 1 0 6$ </td><td> $0 . 5 7 9 \pm 0 . 0 0 7$ </td></tr><tr><td>0.25</td><td> $7 7 . 7 0 \pm 0 . 3 9$ </td><td> $3 0 . 7 0 \pm 0 . 8 0$ </td><td> $2 7 . 0 2 \pm 0 . 1 5$ </td><td> $0 . 5 6 0 \pm 0 . 0 0 6$ </td></tr><tr><td>0.50</td><td> $7 6 . 4 3 \pm 0 . 1 1$ </td><td> $2 9 . 9 0 \pm 0 . 3 3$ </td><td> $2 6 . 7 0 \pm 0 . 1 0$ </td><td> $0 . 5 5 7 \pm 0 . 0 0 9$ </td></tr><tr><td>1.00</td><td> $7 6 . 7 3 \pm 0 . 2 1$ </td><td> $2 9 . 8 0 \pm 0 . 7 2$ </td><td> $2 6 . 5 9 \pm 0 . 0 5$ </td><td> $0 . 6 1 2 \pm 0 . 0 0 7$ </td></tr><tr><td rowspan="4">RWARE-Advanced</td><td>0.00</td><td> $8 0 . 0 3 \pm 0 . 2 3$ </td><td> $4 2 . 2 0 \pm 1 . 4 4$ </td><td> $2 . 9 5 1 \pm 0 . 0 1 0$ </td><td> $0 . 5 8 3 \pm 0 . 0 1 1$ </td></tr><tr><td>0.25</td><td> $7 9 . 9 3 \pm 0 . 6 8$ </td><td> $3 9 . 7 0 \pm 1 . 5 9$ </td><td> $2 . 9 4 7 \pm 0 . 0 2 8$ </td><td> $0 . 5 8 6 \pm 0 . 0 1 1$ </td></tr><tr><td>0.50</td><td> $7 8 . 3 5 \pm 0 . 9 4$ </td><td> $3 3 . 9 0 \pm 1 . 4 3$ </td><td> $2 . 8 8 8 \pm 0 . 0 4 0$ </td><td> $0 . 6 6 6 \pm 0 . 0 2 3$ </td></tr><tr><td>1.00</td><td> $7 7 . 4 5 \pm 0 . 5 6$ </td><td> $3 5 . 5 0 \pm 1 . 1 5$ </td><td> $2 . 8 2 6 \pm 0 . 0 2 3$ </td><td> $0 . 6 5 2 \pm 0 . 0 0 7$ </td></tr></table>

Observation. Prior misspecification reduces performance relative to the nominal SPL configuration, but does not cause a collapse in task completion in either tested environment. With fully reversed priors $( \lambda = 1 )$ , SPL-MAPPO achieves 76.73% TC in MPE-Advanced and 77.45% in RWARE-Advanced, compared with nominal values of 78.23% and 80.03%, respectively. These values also remain above the corresponding Vanilla MAPPO means of 64.08% and 58.63% reported in the main experiments.

The efect is not strictly monotonic in MPE-Advanced, where TC at $\lambda = 1$ is slightly higher than at $\lambda = 0 . 5$ . RWARE-Advanced shows a clearer reduction in TC as the assumed prior becomes less accurate, while full success is more sensitive than TC: FS decreases from 42.20% in the nominal setting to 33.90% at $\lambda = 0 . 5$ and 35.50% at $\lambda = 1$ . These results indicate that SPL retains substantial task-completion performance under validity-prior misspecification in the tested settings, while stricter episode-level success can be more sensitive.

This experiment perturbs only the validity priors used by the acquisition model. The fixed-valid-task constraint, probe accuracy, task utilities, and other elements of the structured task model remain correctly specified. Accordingly, the results provide evidence of tolerance to validity-prior misspecification in the tested settings. Robustness to arbitrary errors in the full structured task model is not established.

## D.7. Statistical Analysis of Primary Comparisons

We conduct a retrospective statistical analysis of the 60 primary Vanilla–SPL comparisons reported in the main paper. Each trained policy is summarized by its mean task completion (TC) over 200 final evaluation episodes, and independent training seeds are treated as the statistical replication units. Throughout this subsection, ∆TC denotes the absolute diference between percentage-valued SPL and Vanilla TC means, $\mathrm { T C } _ { \mathrm { S P L } } - \mathrm { T C } _ { \mathrm { V a n i l l a } } ,$ and is reported using %.

Same-training-seed pairing is available for 57 of the 60 configurations, providing five paired seed-level diferences in each testable configuration. For these comparisons, we report the mean paired diference, a two-sided paired Student t test with four degrees of freedom, and a pointwise 95% t confidence interval. The Non-overlap MAPPO comparisons in MPE-Advanced, RWARE-Base, and RWARE-Advanced use diferent Vanilla and SPL training-seed sets. Their observed five-seed method means remain available for descriptive comparison, but no paired confidence interval or paired p-value is reported.

To account for multiplicity, we apply Holm correction over the declared family of all 60 primary comparisons at familywise level 0.05. The three configurations without verified seed pairing are assigned $p = 1$ internally for the correction and are displayed as unavailable below. This preserves the original 60-comparison family and yields a conservative adjustment. Pointwise confidence intervals are not multiplicity-adjusted.

Table D.7: Retrospective statistical analysis for MPE-Base. Each training seed is a replication unit. ∆TC is the absolute SPL-minus-Vanilla diference in percentage-valued TC scores. Confidence intervals are pointwise 95% intervals from five paired training seeds. Holm p values are adjusted over the declared family of 60 primary comparisons.
<table><tr><td>Participation</td><td>Backbone</td><td>∆TC (%)</td><td>95% CI (%)</td><td>Raw p</td><td>Holm p</td></tr><tr><td>Non-overlap</td><td>IPPO</td><td>7.88</td><td>[6.82, 8.93]</td><td> $3 . 1 7 \times 1 0 ^ { - 5 }$ </td><td>0.0017</td></tr><tr><td rowspan="4">Overlap-25</td><td>MAPPO</td><td>8.23</td><td>[7.03, 9.42]</td><td> $4 . 4 8 \times 1 0 ^ { - 5 }$ </td><td>0.0023</td></tr><tr><td>HAPPO</td><td>11.73</td><td>[3.01, 20.44]</td><td>0.0202</td><td>0.3028</td></tr><tr><td>IPPO</td><td>11.10</td><td>[2.11, 20.09]</td><td>0.0266</td><td>0.3073</td></tr><tr><td>MAPPO</td><td>9.48</td><td>[8.74, 10.21]</td><td> $3 . 5 6 \times 1 0 ^ { - 6 }$ </td><td> $2 . 1 0 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="3">Overlap-50</td><td>HAPPO</td><td>9.65</td><td>[8.86, 10.44]</td><td> $4 . 6 1 \times 1 0 ^ { - 6 }$ </td><td> $2 . 6 7 \times 1 0 ^ { - 4 }$ </td></tr><tr><td>IPPO</td><td>11.05</td><td>[2.81, 19.29]</td><td>0.0204</td><td>0.3028</td></tr><tr><td>MAPPO</td><td>9.38</td><td>[5.41, 13.34]</td><td>0.0028</td><td>0.0752</td></tr><tr><td rowspan="3">Dynamic</td><td>HAPPO</td><td>8.70</td><td>[6.28, 11.12]</td><td> $5 . 6 5 \times 1 0 ^ { - 4 }$ </td><td>0.0237</td></tr><tr><td>IPPO</td><td>13.45</td><td>[5.36, 21.54]</td><td>0.0099</td><td>0.2081</td></tr><tr><td>MAPPO</td><td>8.65</td><td>[7.41, 9.89]</td><td> $4 . 2 4 \times 1 0 ^ { - 5 }$ </td><td>0.0022</td></tr><tr><td rowspan="4">Cascaded</td><td>HAPPO</td><td>10.10</td><td>[6.97, 13.23]</td><td> $8 . 5 9 \times 1 0 ^ { - 4 }$ </td><td>0.0326</td></tr><tr><td>IPPO</td><td>12.08</td><td>[2.41, 21.74]</td><td>0.0256</td><td>0.3073</td></tr><tr><td>MAPPO</td><td>8.30</td><td>[7.06, 9.54]</td><td> $5 . 0 1 \times 1 0 ^ { - 5 }$ </td><td>0.0025</td></tr><tr><td>HAPPO</td><td>8.10</td><td>[7.25, 8.95]</td><td> $1 . 2 3 \times 1 0 ^ { - 5 }$ </td><td> $7 . 0 1 \times 1 0 ^ { - 4 }$ </td></tr></table>

Table D.8: Retrospective statistical analysis for MPE-Advanced. Notation and statistical procedure follow Table D.7. The Non-overlap MAPPO comparison uses diferent Vanilla and SPL training-seed sets; its ∆TC is therefore descriptive and paired inference is unavailable.
<table><tr><td>Participation</td><td>Backbone</td><td>∆TC (%)</td><td>95% CI (%)</td><td>Raw p</td><td>Holm p</td></tr><tr><td>Non-overlap</td><td>IPPO</td><td>7.58</td><td>[7.05, 8.10]</td><td> $2 . 2 6 \times 1 0 ^ { - 6 }$ </td><td> $1 . 3 6 \times 1 0 ^ { - 4 }$ </td></tr><tr><td rowspan="4">Overlap-25</td><td>MAPPO</td><td>14.15</td><td></td><td>一</td><td></td></tr><tr><td>HAPPO</td><td>10.75</td><td>[2.06, 19.44]</td><td>0.0264</td><td>0.3073</td></tr><tr><td>IPPO</td><td>14.30</td><td>[5.25, 23.35]</td><td>0.0118</td><td>0.2358</td></tr><tr><td>MAPPO</td><td>11.10</td><td>[7.48, 14.72]</td><td>0.0010</td><td>0.0378</td></tr><tr><td rowspan="3">Overlap-50</td><td>HAPPO</td><td>8.63</td><td>[6.13, 11.12]</td><td> $6 . 5 9 \times 1 0 ^ { - 4 }$ </td><td>0.0264</td></tr><tr><td>IPPO</td><td>7.33</td><td>[6.43, 8.22]</td><td> $2 . 2 3 \times 1 0 ^ { - 5 }$ </td><td>0.0012</td></tr><tr><td>MAPPO</td><td>9.75</td><td>[1.26, 18.24]</td><td> $0 . 0 3 3 2$ </td><td>0.3073</td></tr><tr><td rowspan="3">Dynamic</td><td>HAPPO</td><td>7.28</td><td>[6.31, 8.24]</td><td> $3 . 0 9 \times 1 0 ^ { - 5 }$ </td><td>0.0017</td></tr><tr><td>IPPO</td><td>11.60</td><td>[0.73, 22.47]</td><td>0.0414</td><td>0.3073</td></tr><tr><td>MAPPO</td><td>12.03</td><td>[3.78, 20.27]</td><td>0.0155</td><td>0.2784</td></tr><tr><td rowspan="4">Cascaded</td><td>HAPPO</td><td>17.08</td><td>[1.68, 32.47]</td><td>0.0370</td><td>0.3073</td></tr><tr><td>IPPO</td><td>9.13</td><td>[1.73, 16.52]</td><td>0.0266</td><td>0.3073</td></tr><tr><td>MAPPO</td><td>9.73</td><td>[0.98, 18.47]</td><td>0.0366</td><td>0.3073</td></tr><tr><td>HAPPO</td><td>7.95</td><td>[5.10, 10.80]</td><td>0.0015</td><td>0.0492</td></tr></table>

– indicates that paired inference is unavailable because Vanilla and SPL use diferent training-seed sets. The corresponding untestable comparison is assigned $p = 1$ internally for the 60-comparison Holm correction.

Table D.9: Retrospective statistical analysis for RWARE-Base. Notation and statistical procedure follow Table D.7. The Non-overlap MAPPO comparison uses diferent Vanilla and SPL training-seed sets; its ∆TC is descriptive and paired inference is unavailable.
<table><tr><td>Participation</td><td>Backbone</td><td>∆TC (%)</td><td>95% CI (%)</td><td>Raw p</td><td>Holm p</td></tr><tr><td>Non-overlap</td><td>IPPO</td><td>20.38</td><td>[4.50, 36.25]</td><td>0.0235</td><td>0.3056</td></tr><tr><td rowspan="4">Overlap-25</td><td>MAPPO</td><td>15.33</td><td>一</td><td></td><td>一</td></tr><tr><td>HAPPO</td><td>13.75</td><td>[9.92, 17.58]</td><td> $5 . 6 7 \times 1 0 ^ { - 4 }$ </td><td>0.0237</td></tr><tr><td>IPPO</td><td>8.05</td><td>[-1.30, 17.40]</td><td>0.0751</td><td>0.3073</td></tr><tr><td>MAPPO</td><td>14.83</td><td>[6.33, 23.32]</td><td>0.0084</td><td>0.1842</td></tr><tr><td rowspan="3">Overlap-50</td><td>HAPPO</td><td>18.20</td><td>[9.60, 26.80]</td><td>0.0042</td><td>0.1049</td></tr><tr><td>IPPO</td><td>7.00</td><td>[3.73, 10.27]</td><td>0.0040</td><td>0.1043</td></tr><tr><td>MAPPO</td><td>13.85</td><td>[4.26, 23.44]</td><td>0.0160</td><td>0.2784</td></tr><tr><td rowspan="3">Dynamic</td><td>HAPPO</td><td>18.53</td><td>[12.18, 24.87]</td><td>0.0013</td><td>0.0440</td></tr><tr><td>IPPO</td><td>16.00</td><td>[12.30, 19.70]</td><td> $2 . 7 6 \times 1 0 ^ { - 4 }$ </td><td>0.0124</td></tr><tr><td>MAPPO</td><td>23.35</td><td>[7.88, 38.82]</td><td>0.0138</td><td>0.2624</td></tr><tr><td rowspan="4">Cascaded</td><td>HAPPO</td><td>19.60</td><td>[8.63, 30.57]</td><td>0.0077</td><td>0.1774</td></tr><tr><td>IPPO</td><td>19.45</td><td>[13.18, 25.72]</td><td>0.0010</td><td>0.0371</td></tr><tr><td>MAPPO</td><td>19.23</td><td>[12.63, 25.82]</td><td>0.0013</td><td>0.0440</td></tr><tr><td>HAPPO</td><td>19.38</td><td>[11.51, 27.24]</td><td>0.0024</td><td>0.0693</td></tr></table>

Table D.10: Retrospective statistical analysis for RWARE-Advanced. Notation and statistical procedure follow Table D.7. The Non-overlap MAPPO comparison uses diferent Vanilla and SPL training-seed sets; its ∆TC is descriptive and paired inference is unavailable.
<table><tr><td>Participation</td><td>Backbone</td><td>∆TC (%)</td><td>95% CI (%)</td><td>Raw p</td><td>Holm p</td></tr><tr><td>Non-overlap</td><td>IPPO</td><td>27.85</td><td>[23.14, 32.56]</td><td> $8 . 0 6 \times 1 0 ^ { - 5 }$ </td><td>0.0040</td></tr><tr><td rowspan="4">Overlap-25</td><td>MAPPO</td><td>21.40</td><td>一</td><td></td><td></td></tr><tr><td>HAPPO</td><td>21.98</td><td>[11.18, 32.77]</td><td>0.0048</td><td>0.1160</td></tr><tr><td>IPPO</td><td>19.05</td><td>[14.95, 23.15]</td><td> $2 . 0 8 \times 1 0 ^ { - 4 }$ </td><td>0.0096</td></tr><tr><td>MAPPO</td><td>14.83</td><td>[11.13, 18.52]</td><td> $3 . 7 1 \times 1 0 ^ { - 4 }$ </td><td>0.0163</td></tr><tr><td rowspan="3">Overlap-50</td><td>HAPPO</td><td>20.45</td><td>[16.11, 24.79]</td><td> $1 . 9 8 \times 1 0 ^ { - 4 }$ </td><td>0.0093</td></tr><tr><td>IPPO</td><td>14.20</td><td>[9.08, 19.32]</td><td>0.0015</td><td>0.0492</td></tr><tr><td>MAPPO</td><td>16.40</td><td>[4.79, 28.01]</td><td>0.0172</td><td>0.2784</td></tr><tr><td rowspan="3">Dynamic</td><td>HAPPO</td><td>16.93</td><td>[10.26, 23.59]</td><td>0.0021</td><td>0.0639</td></tr><tr><td>IPPO</td><td>14.70</td><td>[8.73, 20.67]</td><td>0.0024</td><td>0.0693</td></tr><tr><td>MAPPO</td><td>19.95</td><td>[12.23, 27.67]</td><td>0.0020</td><td>0.0620</td></tr><tr><td rowspan="3">Cascaded</td><td>HAPPO</td><td>18.98</td><td>[16.46, 21.49]</td><td> $3 . 0 5 \times 1 0 ^ { - 5 }$ </td><td>0.0017</td></tr><tr><td>IPPO</td><td>26.45</td><td>[21.10, 31.80]</td><td> $1 . 6 3 \times 1 0 ^ { - 4 }$ </td><td>0.0078</td></tr><tr><td>MAPPO</td><td>20.43</td><td>[15.09, 25.76]</td><td> $4 . 4 3 \times 1 0 ^ { - 4 }$ </td><td>0.0190</td></tr><tr><td></td><td>HAPPO</td><td>18.88</td><td>[13.19, 24.56]</td><td> $7 . 6 9 \times 1 0 ^ { - 4 }$ </td><td>0.0300</td></tr></table>

Summary. SPL has a higher observed mean TC than its corresponding Vanilla backbone in all 60 primary configurations. The equal-weight descriptive mean diference across these configurations is 14.09%; the corresponding backbone-level mean diferences are 13.93%, 14.02%, and 14.33% for IPPO, MAPPO, and HAPPO, respectively. Among the 57 configurations with paired inference, 56 pointwise 95% confidence intervals have a positive lower bound. The single exception is RWARE-Base Overlap-25 with IPPO, whose pointwise interval is [−1.30, 17.40]%. After Holm correction over the declared family of 60 comparisons, 29 configurations remain significant at familywise level 0.05, while 28 testable comparisons do not. The three remaining configurations have no paired test because same-training-seed pairing is unavailable.

These results distinguish descriptive consistency from multiplicity-adjusted statistical evidence. Higher observed SPL means occur in all 60 configurations, but this does not imply that all 60 comparisons are individually statistically significant. Similarly, failure to reject the null hypothesis after Holm correction should not be interpreted as evidence of equivalence between SPL and the corresponding Vanilla method.

Evaluation design and zero standard errors. Final evaluation uses deterministic policies and a fixed set of 200 reset seeds within each environment protocol, reused across training seeds and methods. This controls variation due to the sampled evaluation instances while retaining independent policy training across training seeds.

Four TC cells in the primary results table have an exact seed-level standard error of zero. These are Vanilla IPPO and Vanilla MAPPO under MPE-Base Non-overlap, MAE-MAPPO under MPE-Base Non-overlap, and Vanilla IPPO under MPE-Advanced Non-overlap. In each case, the five independently trained policies produce exactly identical 200-episode mean TC values at full precision. The reported zero therefore concerns variability among the five seed-level evaluation means; it does not imply that individual evaluation episodes have zero variability. The available records establish this numerical plateau under the fixed deterministic evaluation schedule but do not establish a particular behavioral mechanism responsible for it.

Limitations of the statistical analysis. This analysis is retrospective and was conducted on the existing completed runs without additional training or evaluation. The paired tests treat the five training seeds as independent replications and assume that the seed-level diferences are approximately normally distributed. With only five pairs per configuration, this normality assumption cannot be assessed convincingly. Furthermore, because the same fixed evaluation-instance schedule is reused across training seeds and methods, the reported inference is conditional on this evaluation design and does not quantify uncertainty over an unrestricted distribution of deployment tasks.

The three Non-overlap MAPPO configurations without verified same-seed pairing prevent a complete 60-comparison paired analysis. Assigning these unavailable tests p = 1 internally makes the reported Holm correction conservative while preserving the declared family size, but does not replace the missing paired evidence. Pointwise confidence intervals should also be interpreted separately from the Holm-adjusted hypothesis tests.