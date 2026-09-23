# DUAL-FRONTIER: WHEN CAN AN AGENT TRUST ITS WORLD MODEL?

Huatai Zhu<sup>1∗</sup>, Qiang Chen<sup>2∗</sup>, Ziqian Kou<sup>3</sup>, Wenhao Li<sup>4</sup>, Fei Wang<sup>5</sup>, Yichao Cao<sup>1</sup>, Xiu Su<sup>1†</sup>, Yi Chen<sup>2†</sup>

<sup>1</sup>Central South University <sup>2</sup>The Hong Kong University of Science and Technology <sup>3</sup>Xiangjiang Laboratory <sup>4</sup>University of Sydney <sup>5</sup>University of Science and Technology of China

## ABSTRACT

Learned world models are becoming essential to general-purpose agents: by predicting action consequences, they support planning and decision-making while reducing reliance on costly trial and error. This reliance creates a fundamental ambiguity: when a world-model-guided decision fails, the trajectory alone may not reveal whether the agent’s decision rule or the world model caused the loss. We formalize thisfailure-attribution problem as a counterfactual decomposition of return loss and prove that its components are not identifiable from passive interaction, even for finite-horizon planners. This obstruction motivates Dual-Frontier, a learning principle that admits a world-model-guided decision only when its predicted advantage exceeds a certified bound on decision-relevant world-model error; otherwise, evidence is allocated to world-model verification. Action-conditioned value bounds and a closed-loop extension guarantee non-decreasing return for admitted decisions. Calibrated gates and simultaneous confidence sequences support adaptive evidence reuse, with sufficient and necessary verification bounds. Controlled learned-model experiments validate the predicted failure modes and certification behavior, while cross-backbone tool-use benchmarks instantiate the same verify-then-promote rule in realistic agent world-model pipelines, consistently improving decision quality and reliability.

## 1 INTRODUCTION

World models are becoming core infrastructure for agents that act beyond their immediate observations. Consequential action requires anticipating how choices alter future states, information, and opportunities: general multi-step agency entails recoverable knowledge of environmental dynamics (Richens et al., 2025), even under partial observability and stochasticity (Cifuentes, 2026). Learned world models operationalize this knowledge as action-conditioned predictors rolled forward before execution. Whether expressed in latent states, video, language, or internal dynamics, they augment the present observation with forecasts used to compare actions, plan farther ahead, and reduce costly online trial and error.

This predictive interface now spans latent-imagination control in DreamerV3 (Hafner et al., 2025) and scalable model-predictive control in TD-MPC2 (Hansen et al., 2024), as well as navigation (Bar et al., 2025), manipulation (Assran et al., 2025), driving (Russell et al., 2025), and web interaction (Chae et al., 2025). It is increasingly adaptive: WorldEvolver revises predictive memory at test time (Zhang et al., 2026b), CoMAP alternates world-model adaptation with agent reflection (Liu et al., 2026), and recent systems co-train predictive knowledge and policies (Lu et al., 2026) or co-evolve simulators with agents (Guo et al., 2026). Reliability is thus part of the decision mechanism. Short rollouts limit model exploitation (Janner et al., 2019); horizon-calibrated uncertainty addresses compounding error (Wan et al., 2026); safe-improvement methods constrain policy changes (Delgrange et al., 2026).

Yet a poor world-model-guided outcome poses an unresolved question: what should improve next— the agent’s decision rule or the world model on which it relied? The same trajectory can arise because an inadequate rule ignored an accurate forecast or because an inaccurate forecast misled an otherwise sound rule. Passive interaction records only their composition. We call separating these causesfailure attribution. The distinction is operational: learning against a misleading world model can reinforce a bad decision, while gathering more world-model data after the relevant forecast is adequate wastes evidence. Nor can global prediction accuracy decide the issue. Perceptual quality and closed-loop success can diverge (Zhang et al., 2026a); an arbitrarily small transition error may reverse nearly tied actions, whereas a large error in an irrelevant coordinate may alter none. The missing object i decision-specific: does current evidence establish that a world-model-proposed behavior improves upon an explicit reference?

To solve this attribution-and-trust problem, we develop Dual-Frontier, a theory of decision-specific trust for learned-world-model agents. It evaluates the true-return contrast between a candidate behavior proposed with the world model and a reference behavior. The world-model frontier retains promising comparisons lacking evidence; the agent frontier admits those remaining beneficial after world-model and estimation uncertainty. This verify-then-promote rule directs unresolved comparisons to targeted verification and certified ones to agent improvement, without interpreting rejection as proof of model failure. Our contributions are:

• Failure attribution. A fixed-operator counterfactual decomposition separates agent deficiency from world-model effect; a two-step construction proves passive non-identifiability for exact and Monte Carlo planners.

• Decision reliability. An exact Bellman-residual identity converts action-conditioned world-model error into comparison-specific radii, sharp separations, and a closed-loop condition guaranteeing nonnegative expected improvement.

• Dual-frontier learning. Calibrated promotion rules reuse one simultaneous certificate under adaptive world-model and request selection, with progress and matching-order sufficient/necessary evidence bounds.

• Empirical validation. Learned-model experiments test non-identifiability, harmful imagined improvements, and qualification; matched agent–world-model evaluations separate reliability, realized quality, and verification cost.

## 2 RELATED WORK

General-purpose agents in interactive environments. Modern agents transact with websites (Zhou et al., 2024), operate desktops (Xie et al., 2024), repair repositories (Jimenez et al., 2024), and act in embodied environments (Assran et al., 2025). AgentBench spans eight interactive settings (Liu et al., 2024), while continual agents face changing objectives (Liu et al., 2025). Across domains, multi-step attainment requires recoverable predictive knowledge (Richens et al., 2025), including under partial observability and stochasticity (Cifuentes, 2026; Huang et al., 2026). Learned world models expose such knowledge for planning, from navigation (Bar et al., 2025) to web interaction (Chae et al., 2025). Prior work primarily measures end-to-end success or builds domain-specific predictors, leaving forecasts embedded in the system. Dual-Frontier instead isolates the world-model– decision-rule interface and asks whether evidence supports one proposed behavior comparison.

Learned world models for agent planning and adaptation. We use learned world model for an action-conditioned predictor that compares future courses of action, whether latent, visual, textual, or internalized. The lineage extends from recurrent simulators (Ha & Schmidhuber, 2018) and value-equivalent models (Schrittwieser et al., 2020) to DreamerV3 (Hafner et al., 2025), TD-MPC2 (Hansen et al., 2024), and multi-task policy learning (Georgiev et al., 2025). Current systems forecast navigation (Yao et al., 2025), manipulation (Assran et al., 2025), driving (Russell et al., 2025), and web transitions (Chae et al., 2025). WorldEvolver adapts predictive memory online (Zhang et al., 2026b); CoEx updates persistent beliefs during exploration (Kim & Hwang, 2025). WebEvolver combines synthetic trajectories with look-ahead planning (Fang et al., 2025), while CoMAP alternates world-model adaptation and reflection (Liu et al., 2026). PaW co-trains policy and world model (Lu et al., 2026); GenEnv co-evolves agents and simulators (Guo et al., 2026); DreamGym and Agent World Model scale synthesized interaction (Chen et al., 2026; Wang et al., 2026). These systems optimize or exploit prediction to improve the agent, but do not identify whether a failed decision implicates its rule or its forecast. Dual-Frontier formalizes that ambiguity and qualifies a behavior comparison rather than a world model globally.

Reliable model-based learning and adaptation. World-model exploitation motivates short rollouts (Janner et al., 2019) and pessimism (Yu et al., 2020); policy-aware learning targets downstream gradient error (Abachi et al., 2020; D’Oro et al., 2020). Recent work further studies horizon-dependent uncertainty (Wan et al., 2026), local safe improvement (Delgrange et al., 2026), and confidencefiltered foresight (Zhang et al., 2026b). WAKER collects data where estimated world-model error is high (Rigter et al., 2024); AdaWM separates dynamics and policy mismatch under transfer, then fine-tunes the indicated component (Wang et al., 2025). These methods mainly limit error or respond to an assumed diagnostic. By contrast, Dual-Frontier first proves that passive failure need not identify its source, then links action-conditioned error to value and imagined-gradient distortion. WAKER targets accuracy across environments, whereas we certify a behavior comparison; AdaWM selects adaptation under shift, whereas we establish when a world-model-guided change is actually justified in practice. Split conformal calibration (Angelopoulos & Bates, 2023) and simultaneous confidence yield certificates valid under adaptive evidence reuse, connecting attribution, decision reliability, and allocation in one formulation.

## 3 SET UP

For task z, let $\mathcal { M } _ { z } = ( S , A , P _ { z } , r _ { z } , \rho _ { z } , H )$ and $\widehat { \mathcal { M } } _ { z } = ( S , A , \widehat { P } _ { z } , \widehat { r } _ { z } , \rho _ { z } , H )$ denote the true world and its learned world model. The spaces are standard Borel, $H \geq 1$ , rewards lie in $[ 0 , R _ { b } ]$ , and both systems share an information interface (Appendix A). Suppressing z, let $J , \widehat { J }$ be undiscounted returns, $\mu _ { t } ^ { \pi }$ the true law of $( s _ { t } , a _ { t } )$ , and $\mathrm { T V } ( P , Q { \bar { ) } } = \operatorname* { s u p } _ { B } | P ( { \bar { B } } { \bar { ) } } - Q ( { \bar { B } } )$ |.

A fixed operator $\mathsf { A } _ { \phi }$ maps a supplied world to a policy, with ϕ fixing its search, information access, budget, and randomization. Replacing only the world defines

$$
\widehat { \pi } = \mathsf { A } _ { \phi } ( \widehat { \mathcal { M } } ) , \qquad \pi ^ { \circ } = \mathsf { A } _ { \phi } ( \mathcal { M } ) .\tag{1}
$$

Let $J ^ { * } = \operatorname* { s u p } _ { \pi \in \Pi } J ( \pi )$ , where Π contains both outputs.

Definition 1 (Counterfactual attribution). Let

$$
\begin{array} { r l } & { A : = J ^ { * } - J ( \pi ^ { \circ } ) , \qquad W : = J ( \pi ^ { \circ } ) - J ( \widehat { \pi } ) , } \\ & { R : = J ^ { * } - J ( \widehat { \pi } ) = [ J ^ { * } - J ( \pi ^ { \circ } ) ] + [ J ( \pi ^ { \circ } ) - J ( \widehat { \pi } ) ] = A + W . } \end{array}\tag{2}
$$

Here $R , A \geq 0 ,$ , whereas W is a signed world-model effect.

The decomposition is relative to one fixed predictive decision procedure; because model error may accidentally help a limited agent, $W$ is signed.

Theorem 1 (Passive non-identifiability). For every $c \in ( 0 , R _ { b } ] , a _ { \lrcorner }$ fixed known operator and supplied world model admit afamily oftwo-step true worlds, indexed by $\lambda \in [ 0 , c ]$ , with $\ \dot { ( } A , W ) = ( \lambda , \dot { c } - \lambda )$ and identical laws ofarbitrarily many deployed episodes. Any attribution estimator based on these episodes, the known operator, and the supplied world model satisfies

$$
\begin{array} { r l r } & { } & { \underset { \lambda \in [ 0 , c ] } { \operatorname* { s u p } } ~ \mathbb { E } _ { \lambda } | \widehat { \cal A } - \lambda | \geq \frac { 1 } { 2 } \left( \mathbb { E } _ { 0 } | \widehat { \cal A } | + \mathbb { E } _ { c } | \widehat { \cal A } - c | \right) } \\ & { } & { = \frac { 1 } { 2 } \mathbb { E } \left[ | \widehat { \cal A } | + | \widehat { \cal A } - c | \right] \geq \frac { c } { 2 } . } \end{array}\tag{3}
$$

where the middle expectation is under the common observational law. Endpoint classification has minimax error $1 / 2 .$ . Both bounds are attained,for exact andfinite-sample Monte Carlo planning.

Appendix B gives the construction and minimax proof. Notably, the supplied world model may be exact on the deployed occupancy; the ambiguity can reside entirely in untried actions. The obstruction is therefore passive rather than a prohibition on verification: intervention can recover missing information, whereas an unstable operator can amplify small predictive error (Proposition 1). We therefore certify a specified next use rather than infer ownership from passive failure.

## 4 WORLD-MODEL RELIABILITY

## 4.1 CERTIFYING A PROPOSED USE

A request $x = ( z , { \widehat { \mathcal { M } } } , \pi _ { 0 } , \pi _ { 1 } )$ specifies a learned world model, a reference behavior $\pi _ { 0 } .$ , and a candidate $\pi _ { 1 }$ , including their continuation rules. Define

$$
\Gamma _ { \mathcal { M } } ( x ) = J _ { \mathcal { M } } ( \pi _ { 1 } ) - J _ { \mathcal { M } } ( \pi _ { 0 } ) , \qquad \ell _ { \mathcal { C } } ( x ) = \operatorname* { i n f } _ { N \in \mathcal { C } } \Gamma _ { \mathcal { N } } ( x ) .\tag{4}
$$

On $\mathcal { M } \in \mathcal { C } , \ell _ { \mathcal { C } } > 0$ certifies improvement; failure to certify does not imply harm. The policies remain fixed while the evaluation world varies (Appendix C.1).

For bounded $f ,$ write $\begin{array} { r } { P f = \int f ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) } \end{array}$ and span $( f ) = \operatorname* { s u p } f - \operatorname* { i n f } f$ . With learned value $\widehat { V } _ { t } ^ { \pi }$ and $\widehat V _ { H } ^ { \pi } = 0 .$ , set

$$
\Delta _ { t } ^ { \pi } = r - { \widehat r } + ( P - { \widehat P } ) { \widehat V } _ { t + 1 } ^ { \pi } ,\tag{5}
$$

$$
\varepsilon _ { \mathrm { I } } ( \pi ) = \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi } } [ | r - \widehat { r } | + \mathrm { s p a n } ( \widehat { V } _ { t + 1 } ^ { \pi } ) \mathrm { T V } ( P , \widehat { P } ) ] .\tag{6}
$$

Theorem 2 (Decision reliability). For every policy and every policy comparison,

$$
\begin{array} { r l } & { J ( \pi ) - \widehat { J } ( \pi ) = \displaystyle \int _ { S } \left( V _ { 0 } ^ { \pi } - \widehat { V } _ { 0 } ^ { \pi } \right) ( s ) \rho ( \mathrm { d } s ) } \\ & { \qquad = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \displaystyle \int _ { S \times \mathcal { A } } \Delta _ { t } ^ { \pi } ( s , a ) \mu _ { t } ^ { \pi } ( \mathrm { d } s , \mathrm { d } a ) } \\ & { \qquad = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi } } \Delta _ { t } ^ { \pi } . } \end{array}\tag{7}
$$

Moreover,

$$
\begin{array} { r l } { | J ( \pi ) - \widehat { J } ( \pi ) | \leq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int \left( | r - \widehat { r } | + \displaystyle \left| \int \widehat { V } _ { t + 1 } ^ { \pi } \mathrm { d } ( P - \widehat { P } ) \right| \right) \mathrm { d } \mu _ { t } ^ { \pi } } & { } \\ { \leq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int \left( | r - \widehat { r } | + \mathrm { s p a n } ( \widehat { V } _ { t + 1 } ^ { \pi } ) \mathrm { T V } ( P , \widehat { P } ) \right) \mathrm { d } \mu _ { t } ^ { \pi } } & { } \\ { = \varepsilon _ { \mathrm { I } } ( \pi ) . } \end{array}\tag{8}
$$

Consequently,

$$
\begin{array} { r l } & { J ( \pi _ { 1 } ) - J ( \pi _ { 0 } ) = \widehat { J } ( \pi _ { 1 } ) - \widehat { J } ( \pi _ { 0 } ) + [ J ( \pi _ { 1 } ) - \widehat { J } ( \pi _ { 1 } ) ] - [ J ( \pi _ { 0 } ) - \widehat { J } ( \pi _ { 0 } ) ] } \\ & { \qquad \quad \geq \widehat { J } ( \pi _ { 1 } ) - \widehat { J } ( \pi _ { 0 } ) - | J ( \pi _ { 1 } ) - \widehat { J } ( \pi _ { 1 } ) | - | J ( \pi _ { 0 } ) - \widehat { J } ( \pi _ { 0 } ) | } \\ & { \qquad \quad \geq \widehat { J } ( \pi _ { 1 } ) - \widehat { J } ( \pi _ { 0 } ) - \varepsilon _ { \mathrm { I } } ( \pi _ { 1 } ) - \varepsilon _ { \mathrm { I } } ( \pi _ { 0 } ) . } \end{array}\tag{9}
$$

Corollary 1 (Planning regret). $\begin{array} { r } { I f \widehat { J } ( \widehat { \pi } ) \geq \operatorname* { s u p } _ { \pi \in \Pi } \widehat { J } ( \pi ) - \delta _ { p } , } \end{array}$ , then

$$
J ^ { * } - J ( \widehat { \pi } ) \leq \operatorname* { s u p } _ { \pi \in \Pi } \varepsilon _ { \mathrm { I } } ( \pi ) + \delta _ { p } + \varepsilon _ { \mathrm { I } } ( \widehat { \pi } ) .\tag{10}
$$

Appendix C gives the Bellman telescope and full regret derivation; Appendix C.4 treats shifts beyond verified occupancy.

Corollary 2 (Predictive accuracy does not certify a decision). Arbitrarily small uniform transition error can reverse world-model-greedy action rankings; total variation one can coexist with exact valuesfor every policy.

Appendix D.4 gives both constructions: qualification must compare prediction error with the advantage at stake.

## 4.2 FROM A CERTIFIED COMPARISON TO CLOSED-LOOP PLANNING

Replanning changes a certified continuation. Fix a reference policy $\pi _ { 0 }$ , with $Q _ { t } ^ { 0 } = r + P V _ { t + } ^ { \pi _ { 0 } } $ and $\widehat { Q } _ { t } ^ { 0 } = \widehat { r } + \widehat { P } \widehat { V } _ { t + 1 } ^ { \pi _ { 0 } }$ . For candidate action kernel $\kappa _ { t } .$ , let $| Q _ { t } ^ { 0 } - \widehat Q _ { t } ^ { 0 } | \leq b _ { t }$ and define

$$
d _ { t } ( s ) = \int Q _ { t } ^ { 0 } ( s , a ) ( \kappa _ { t } - \pi _ { 0 , t } ) ( \mathrm { d } a \mid s ) ,\tag{11}
$$

$$
\widehat { d } _ { t } ( s ) = \int \widehat { Q } _ { t } ^ { 0 } ( s , a ) ( \kappa _ { t } - \pi _ { 0 , t } ) ( \mathrm { d } a \mid s ) ,\tag{12}
$$

$$
B _ { t } ( s ) = \int b _ { t } ( s , a ) ( \kappa _ { t } + \pi _ { 0 , t } ) ( \mathrm { d } a \mid s ) .\tag{13}
$$

Theorem 3 (Closed-loop reliability). Suppose simultaneously at all admissible states that $\widehat { d } _ { t } \geq$ $S _ { t } - \xi _ { t }$ , where $\xi _ { t } \geq 0 .$ . Put

$$
\begin{array} { r l r } & { T _ { t } = S _ { t } - B _ { t } - \xi _ { t } , } & { g _ { t } = \mathbb { I } \{ T _ { t } > 0 \} , } \\ & { \nu _ { t } = g _ { t } \kappa _ { t } + ( 1 - g _ { t } ) \pi _ { 0 , t } . } \end{array}\tag{14}
$$

$I f \rho _ { t } ^ { \nu }$ is the true state law under $\nu ,$ then

$$
\begin{array} { r l } { J ( \nu ) - J ( \pi _ { 0 } ) = } & { \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } \rho _ { t } ^ { \nu } ( \mathrm { d } s ) \int _ { A } Q _ { t } ^ { 0 } ( s , a ) [ \nu _ { t } - \pi _ { 0 , t } ] ( \mathrm { d } a \mid s ) } \\ & { = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) d _ { t } ( s ) \rho _ { t } ^ { \nu } ( \mathrm { d } s ) } \\ & { \geq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) [ \widehat { d } _ { t } ( s ) - B _ { t } ( s ) ] \rho _ { t } ^ { \nu } ( \mathrm { d } s ) } \\ & { \geq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) T _ { t } ( s ) \rho _ { t } ^ { \nu } ( \mathrm { d } s ) \geq 0 . } \end{array}\tag{15}
$$

Strict improvement holds if a positive-margin intervention is reached with positive probability.

Appendix C.6 proves the result under repeated replanning. The guarantee remains decision-specific: it compares local predicted advantage against continuation-value uncertainty under the state distribution induced by the gated policy. Under $| r - \widehat { r } | \leq \epsilon _ { r }$ and $\mathrm { T V } ( P , \widehat { P } ) \leq \epsilon _ { p }$ , Theorem 2 gives

$$
\begin{array} { r } { b _ { t } = ( H - t ) \epsilon _ { r } + \frac { 1 } { 2 } ( H - t ) ( H - t - 1 ) R _ { b } \epsilon _ { p } . } \end{array}\tag{16}
$$

Appendix C.6 gives tighter and truncated-rollout variants; Appendix D gives the corresponding differential certificates.

## 5 DUAL-FRONTIER LEARNING

## 5.1 QUALIFICATION AND PROMOTION

Let $S ( x )$ estimate $\widehat \Gamma ( x ) = \widehat J ( \pi _ { 1 } ) - \widehat J ( \pi _ { 0 } )$ , B(x) bound world-model distortion, and $\xi ( x )$ bound estimation error. Define

$$
T ( x ) = S ( x ) - B ( x ) - \xi ( x ) ,\tag{17}
$$

$$
{ \mathcal { F } } _ { \mathrm { W } } = \{ x : S ( x ) > \xi ( x ) , B ( x ) \geq S ( x ) - \xi ( x ) \} ,\tag{18}
$$

$$
{ \mathcal { F } } _ { \mathrm { A } } = \{ x : T ( x ) > 0 \} .\tag{19}
$$

${ \mathcal { F } } _ { \mathrm { W } }$ retains promising but uncertified comparisons for verification, whereas $\mathcal { F } _ { \mathrm { A } }$ contains certified improvements; $S \le \xi$ is deferred.

Calibrated uncertainty. For $e ( \boldsymbol { x } ) = | \widehat { \Gamma } ( \boldsymbol { x } ) - \Gamma _ { \mathcal { M } } ( \boldsymbol { x } ) |$ |, fit a nonnegative score u independently of n exchangeable calibration requests and set

$$
q _ { j } = e _ { j } - u ( x _ { j } ) , \qquad k = \lceil ( n + 1 ) ( 1 - \alpha ) \rceil , \qquad B ( x ) = \lceil u ( x ) + q _ { ( k ) } \rceil + . . .\tag{20}
$$

Proposition 7 gives $\mathbb { P } \{ e ( x ) > B ( x ) \} \le \alpha$ . If $\widehat { \Gamma } \geq S - \xi$ except with probability δ, then

$$
\{ T ( x ) > 0 , \Gamma _ { \mathcal { M } } ( x ) \leq 0 \} \subseteq \{ e ( x ) > B ( x ) \} \cup \{ \widehat { \Gamma } ( x ) < S ( x ) - \xi ( x ) \} ,
$$

$$
\mathbb { P } \{ T ( x ) > 0 , \Gamma _ { \mathcal { M } } ( x ) \le 0 \} \le \alpha + \delta .\tag{21}
$$

Appendix E gives the rank proof, noisy-label extension, and selection-conditional distinction.

## 5.2 REUSABLE EVIDENCE AND ADAPTIVE PLANNING

For a stationary finite task with D states, m state–action rows, shared known rewards, and n samples per row, let $\overline { { P } }$ be empirical and define

$$
a _ { n } = \sqrt { \frac { D \log 2 + \log ( 4 m / \alpha ) } { 2 n } } , \qquad \mathcal { C } = \left\{ \mathcal { N } : \operatorname* { m a x } _ { s , a } \mathrm { T V } ( P _ { N } , \overline { { P } } ) \leq a _ { n } \right\} .\tag{22}
$$

Appendix E.4 gives $\mathbb { P } \{ \mathcal { M } \in \mathcal { C } \} \geq 1 - \alpha$ . Put $K = R _ { b } H ( H - 1 )$ and, for selected $\widehat { P } _ { i }$

$$
u _ { i } = \operatorname* { m a x } _ { s , a } \operatorname* { m i n } \{ 1 , \mathrm { T V } ( \overline { { P } } , \widehat { P } _ { i } ) + a _ { n } \} .\tag{23}
$$

Theorem 4 (Adaptive reuse of a world certificate). Let $x _ { i } , \widehat { P } _ { i }$ depend on the audit and all previous observations. Conditional on their selection, assume $\mathbb { P } ( F _ { i } ^ { c } \mid \mathcal { H } _ { i } ) \leq \delta _ { i }$ , where $F _ { i } = \{ \widehat { \Gamma } _ { i } \geq S _ { i } - \xi _ { i } \}$ Accept only

$$
T _ { i } = S _ { i } - K u _ { i } - \xi _ { i } > 0 .\tag{24}
$$

For $E = \{ { \mathcal { M } } \in { \mathcal { C } } \}$ and $\begin{array} { r } { B = \bigcup _ { i > 1 } \{ T _ { i } > 0 , \Gamma _ { \mathcal { M } } ( x _ { i } ) \leq 0 \} } \end{array}$

$$
B \subseteq E ^ { c } \cup \bigcup _ { i \geq 1 } F _ { i } ^ { c } ,
$$

$$
\mathbb { P } ( \boldsymbol { B } ) \le \mathbb { P } ( \boldsymbol { E } ^ { c } ) + \sum _ { i \ge 1 } \mathbb { E } [ \mathbb { P } ( F _ { i } ^ { c } \mid \mathcal { H } _ { i } ) ] \le \alpha + \sum _ { i \ge 1 } \delta _ { i } .\tag{25}
$$

The audit costs mn queries, independently of subsequent comparisons.

Appendix E.5 gives the full adaptive proof and extensions.

## 5.3 PROGRESS AND EVIDENCE COST

For successive accepted behaviors with certified margins $T _ { i } > 0$ , simultaneous validity gives

$$
J ( \pi _ { n } ) - J ( \pi _ { 0 } ) \geq \sum _ { i = 0 } ^ { n - 1 } T _ { i } ,\tag{26}
$$

$$
| \{ i \in \{ 0 , \ldots , n - 1 \} : T _ { i } \geq \tau \} | \leq { \frac { H R _ { b } - J ( \pi _ { 0 } ) } { \tau } } , \qquad \tau > 0 .\tag{27}
$$

Thus a bounded objective cannot sustain a fixed positive certified margin indefinitely (Appendix F.2).

With shared rewards and $H \ \geq \ 2 .$ , let $q _ { i } = \operatorname* { m a x } _ { s , a } \mathrm { T V } ( P , \widehat { P } _ { i } )$ and $a _ { i } = \widehat { \Gamma } _ { i } - K q _ { i }$ . For exact world-model evaluation, on $E ,$

$$
T _ { i } \ge a _ { i } - 2 K a _ { n } ,\tag{28}
$$

$$
n > \frac { 2 } { s ^ { 2 } } [ D \log 2 + \log ( 4 m / \alpha ) ]\tag{29}
$$

certifies every request with $a _ { i } / K \geq s$ simultaneously (Appendix E.6).

Conversely, distinguishing true advantages $\pm c \gamma , \gamma \in ( 0 , 1 / 4 ]$ , with promotion probabilities at least $1 - \delta$ and at most δ requires

$$
n \geq \frac { \mathrm { k l } ( 1 - \delta , \delta ) } { 1 6 \gamma ^ { 2 } } , \qquad 0 < \delta < \frac { 1 } { 2 } .\tag{30}
$$

Appendices E.8 and F give the lower bound and adaptive accounting. Together, the bounds establish the inverse-square statistical price of reliable promotion with reusable evidence.

Table 1: Agent–world-model evaluation on BFCL v4, API-Bank, and NexusRaven. We report Task Success (Success) and Parameter Accuracy (Acc.); Avg. is the uniform mean across benchmarks.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">BFCL v4</td><td colspan="2">API-Bank</td><td colspan="2">NexusRaven</td><td colspan="2">Avg.</td></tr><tr><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td></tr><tr><td rowspan="6">Llama-3.1-8B Instruct</td><td>Agent-only</td><td>69.12</td><td>85.31</td><td>64.96</td><td>78.76</td><td>56.29</td><td>72.36</td><td>63.46</td><td>78.81</td></tr><tr><td>Always-WM</td><td>22.50</td><td>60.79</td><td>66.73</td><td>79.20</td><td>63.21</td><td>80.33</td><td>50.81</td><td>73.44</td></tr><tr><td>Confidence</td><td>35.12</td><td>69.51</td><td>88.58</td><td>92.55</td><td>82.39</td><td>87.89</td><td>68.70</td><td>83.32</td></tr><tr><td>Consistency</td><td>46.38</td><td>74.52</td><td>97.24</td><td>98.00</td><td>96.54</td><td>96.83</td><td>80.05</td><td>89.78</td></tr><tr><td>Pessimistic</td><td>41.62</td><td>72.14</td><td>94.29</td><td>95.67</td><td>90.88</td><td>92.91</td><td>75.60</td><td>86.91</td></tr><tr><td>Dual-Frontier</td><td>78.12</td><td>90.30</td><td>98.62</td><td>98.83</td><td>98.11</td><td>98.01</td><td>91.62</td><td>95.71</td></tr><tr><td rowspan="6">Qwen3-8B</td><td>Agent-only</td><td>76.38</td><td>90.78</td><td>71.26</td><td>80.00</td><td>74.21</td><td>81.31</td><td>73.95</td><td>84.03</td></tr><tr><td>Always-WM</td><td>21.88</td><td>60.38</td><td>67.52</td><td>80.68</td><td>62.58</td><td>80.68</td><td>50.66</td><td>73.91</td></tr><tr><td>Confidence</td><td>35.00</td><td>69.41</td><td>91.34</td><td>94.06</td><td>91.19</td><td>93.02</td><td>72.51</td><td>85.50</td></tr><tr><td>Consistency</td><td>47.75</td><td>75.42</td><td>98.03</td><td>98.10</td><td>97.11</td><td>97.80</td><td>80.96</td><td>90.44</td></tr><tr><td>Pessimistic</td><td>41.62</td><td>73.18</td><td>93.11</td><td>95.24</td><td>94.97</td><td>96.27</td><td>76.57</td><td>88.23</td></tr><tr><td>Dual-Frontier</td><td>79.38</td><td>92.20</td><td>98.92</td><td>98.64</td><td>97.80</td><td>97.96</td><td>92.03</td><td>96.27</td></tr><tr><td rowspan="3">Frontier Models</td><td>GPT-6 Astra</td><td>82.19</td><td>89.56</td><td>84.24</td><td>85.14</td><td>87.40</td><td>91.35</td><td>84.61</td><td>88.68</td></tr><tr><td>Claude Opus 5</td><td>84.47</td><td>93.96</td><td>64.72</td><td>93.35</td><td>67.62</td><td>96.85</td><td>72.27</td><td>94.72</td></tr><tr><td>GLM-5.3-Flash</td><td>85.78</td><td>80.86</td><td>67.00</td><td>79.08</td><td>79.53</td><td>86.46</td><td>77.44</td><td>82.13</td></tr></table>

![](images/80dc5d992af4e07b29cca76d824c7854b74279ce067c15ef24779e5f54bb2669.jpg)

![](images/0d3e19ba103c9eceb8933132776727fc2b44250b53399e008fd94850923eb514.jpg)

![](images/f5e61200b2d229c3a4b323e24878d6fd860c79749840adb823336f17bac1ec70.jpg)  
Figure 1: Controlled finite-world validation of intervention attribution, planning qualification, and imagined-update transfer.

## 6 EXPERIMENTS

Our experiments progressively test whether the theory identifies when a learned world model can be used reliably and whether Dual-Frontier converts that diagnosis into better decisions. We proceed in two stages: controlled finite-world experiments directly isolate and validate attribution, qualification, and evidence allocation under auditable conditions, after which a cross-backbone evaluation on public agent benchmarks directly instantiates the same verify-then-promote rule in modern tool-use pipelines across heterogeneous model-task settings.

## 6.1 CONTROLLED VALIDATION WITH LEARNED WORLD MODELS

We train action-conditioned learned world models in sparse, chain, and grid finite-horizon worlds and test held-out decisions. Each environment family contains 16 states and four actions, with horizons $H \in \{ 4 , 8 , 1 2 \}$ , providing controlled variation in both dynamics and planning depth. We use 40 development worlds to fix the protocol, 80 independent worlds for finite-world auditing, and 240 disjoint held-out worlds for final evaluation, with separate identifiers and random streams across all three roles. The learned world model is a Beta-

![](images/3f6e967fcfb09c92e67c959cecb3735267067811bb437358712768a6632934e6.jpg)  
Inverse decision margin 1/γ

![](images/48a0f6374b4c7e3f1e7877461b7e3cce1c039b29fd3c9475ef23015b7c92ab2d.jpg)  
Figure 2: Evidence complexity and audit reuse. Left: inverse-square certification cost. Right: shared-certificate verification.

smoothed empirical transition kernel fitted from sampled row observations, while a separately acquired 32-query-per-row model determines the fixed reference behavior. Given a reference action and candidate intervention, the model predicts the counterfactual consequence and the router decides whether to alter the action; exact returns are evaluation-only. The evaluation combines 400 paired constructions that share the learned model and passive record but differ on one unobserved action, held-out planning and teaching requests comparing unconditional positive-imagination acceptance, uncertainty-only rejection, and decision-relative qualification, and online runs contrasting ungated, uniformly gated, and decision-directed evidence acquisition under the same real-transition budget. All methods share the world model, reference behavior, proposals, calibration split, and evidence budget, varying only structure, horizon, margin, and evidence. Appendix G.1 gives the construction and protocol. Figure 1 shows that targeted evidence raises attribution accuracy from 50.25% to 100% and cuts effect error fivefold (0.0900 to 0.0189); over 11,520 requests, qualification reduces harmful use from 5.10% to 0.30%. Certified imagined updates preserve beneficial transfer while rejecting harmful revisions. Figure 2 yields a log–log exponent of 2.014 when varying only the true decision margin, matching $n \overset { \cdot } { = } \Theta ( \gamma ^ { - 2 } ) ;$ ; a shared certificate handles 200 adaptive requests with 664 versus 350,400 real queries (527.7× fewer) while keeping familywise false-promotion risk below 5%. These results validate counterfactual attribution, decision-relative qualification, inverse-margin scaling, and reusable certification. Appendix G.2 reports full curves, ablations, and per-environment results.

Table 2: Dual-Frontier component ablation with Llama-3.1-8B-Instruct on three benchmarks. We report Task Success (Success), Parameter Accuracy (Acc.), and uniform benchmark averages.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">BFCL v4</td><td colspan="2">API-Bank</td><td colspan="2">NexusRaven</td><td colspan="2">Avg.</td></tr><tr><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td></tr><tr><td rowspan="4">Llama-3.1-8B Instruct</td><td>Advantage-only</td><td>65.41</td><td>82.60</td><td>94.60</td><td>96.12</td><td>93.71</td><td>95.40</td><td>84.57</td><td>91.37</td></tr><tr><td>w/o world-model error</td><td>69.82</td><td>85.57</td><td>96.49</td><td>97.40</td><td>95.61</td><td>96.61</td><td>87.31</td><td>93.19</td></tr><tr><td>w/o estimation error</td><td>76.41</td><td>89.33</td><td>98.02</td><td>98.23</td><td>97.39</td><td>97.70</td><td>90.61</td><td>95.09</td></tr><tr><td>Dual-Frontier</td><td>78.12</td><td>90.30</td><td>98.62</td><td>98.83</td><td>98.11</td><td>98.01</td><td>91.62</td><td>95.71</td></tr></table>

![](images/a9543ec49165450a7548f364367b5e35a9b531b15d27a6e7395671d726a6518e.jpg)

![](images/cfaff8745085ee11014f752dde765975a87820cbbe1e4be929c62f5387ad6ea6.jpg)

![](images/3913a58b66a787ceb54bfdba0c4b236d216181c84be5f1ecb7dd5b760c1fe398.jpg)  
Figure 3: Net decision gain across benchmarks and backbones. Red arrows show Dual-Frontier’s NDG improvement over Always-WM.

## 6.2 AGENT-WORLD-MODEL EVALUATION ACROSS BENCHMARKS

We next instantiate the same Dual-Frontier decision rule in realistic agent–world-model interaction, using Qwen-AgentWorld as the learned world model (Zuo et al., 2026). We evaluate Llama-3.1- 8B-Instruct (Grattafiori et al., 2024) and Qwen3- 8B (Yang et al., 2025). The benchmarks separately cover function calling in BFCL v4 (Patil et al., 2025), API use in API-Bank (Li et al., 2023), and heterogeneous tool schemas in NexusRaven (Srinivasan et al., 2023). For each request, the agent first emits its base action, after which Qwen-AgentWorld proposes candidate revisions together with predicted consequences. To isolate the trust decision, all routing rules evaluate the same fixed candidate and share identical prompts, schemas, candidate records, and inference budgets; they differ only in whether and how that proposal is admitted. This matched protocol separates selective qualification from proposal quality or generation cost. We also test frontier models (OpenAI, 2026; Anthropic, 2026; Z.ai, 2026). Appendix H.1 details the calibrated routing rule, and Appendix H.2 defines the metrics.

Table 3: Robustness of Dual-Frontier across three independent world-model generation seeds. Results are mean ± standard deviation over a fixed randomly sampled benchmark subset.
<table><tr><td>Backbone</td><td>Success</td><td>Acc.</td></tr><tr><td>Llama-3.1-8B-Instruct</td><td> $9 1 . 4 8 \pm 0 . 6 3$ </td><td> $9 5 . 5 7 \pm 0 . 4 1$ </td></tr><tr><td>Qwen3-8B</td><td> $9 1 . 9 1 \pm 0 . 5 7$ </td><td> $9 6 . 1 8 \pm 0 . 3 5$ </td></tr></table>

![](images/60cb85b82dbf0b7af0ea4e33b5c02d09120940afdc98e60bc52faba2da7e3f8d.jpg)

![](images/0cdb7bd3750f54e155fffc5750f1b191e655a529f973c7c2f82bec0903a2fda8.jpg)

Figure 4: BFCL v4 reliability. Dual-Frontier reduces harmful revisions and selective risk across both agent backbones.  
![](images/a791822257dea4efee2d8127d757d30baed62f912c1b59e41d7f094904e1510a.jpg)  
Figure 5: Representative BFCL v4 cases. Dual-Frontier promotes a reliable correction while deferring a confident but high-risk revision.

Table 1 shows that Dual-Frontier leads every benchmark–metric pair for both backbones. Against the strongest competing rule, average Success improves by 11.57/11.07 points and Acc. by 5.93/5.83 for Llama-3.1-8B-Instruct/Qwen3-8B. Table 2 shows monotonic recovery as the two uncertainty coordinates are restored; Appendix H.3 gives the Qwen3-8B ablation. Figure 3 reports positive gains on all benchmarks, exceeding Always-WM by 31.10–58.12 points, while Table 3 shows limited variation across three generation seeds.

Figure 4 shows lower harmful revisions and selective risk on BFCL v4 for both backbones, and Figure 5 illustrates the same rule on individual requests. Together with the coverage–risk results in Appendix H.3, these findings attribute the gains to selectively admitted revisions rather than more aggressive world-model use.

Dual-Frontier preserves the base action when estimated benefit is not sufficiently separated from model and finite-generation uncertainty. This request-level policy matters because the same proposal can help one request and harm another even when predicted benefits appear similar. It therefore avoids treating the world model as globally reliable or unreliable. Because prompts, schemas, candidate records, and inference budgets are fixed, the reliability change reflects qualification rather than a stronger proposal generator. Across both backbones, lower harmful revisions accompany higher task success and parameter accuracy; on API-Bank and NexusRaven, the rule retains high coverage while rejecting revisions that fail the certified margin. The ablations reinforce the same mechanism—benefit proposes an intervention, but evidence determines whether it is promoted.

Agreement between task success and parameter accuracy rules out a simple trade between tool selection and malformed arguments. Dual-Frontier improves both, indicating more correct complete actions rather than locally plausible revisions. Consistency across datasets, backbones, ablations, and seeds supports an evidence-sensitive trust mechanism.

## 6.3 IN-DEPTH ANALYSIS

We finally examine the mechanism behind Dual-Frontier. Figure 6 shows that beneficial revisions cluster in the positive-benefit, positive-verification region, whereas many harmful proposals remain outside the promotion frontier despite favorable predicted benefit. This indicates that predicted benefit alone is insufficient and verification is needed to screen unreliable interventions. Figure 5 provides complementary BFCL v4 cases, where Dual-Frontier promotes a supported correction but defers a confident, high-risk revision. Together, these observations align with Sections 3–5: decision-level qualification separates promising predictions from those sufficiently supported for action. Controlled finite-world experiments and public agent benchmarks consistently validate the same frontier mechanism under formal certification and realistic tool use, confirming its robustness across both settings.

![](images/9fab27fd0ba26207e0bad3a02984c5c6eb5d9573e1c3b0a29aa66310d7d54c34.jpg)  
Figure 6: Frontier geometry over 240 sampled requests.

## 7 CONCLUSION

Dual-Frontier establishes a decision-specific view of world-model reliability. We show that passive failures cannot identify whether an error originates from the agent or the world model, motivating explicit qualification against model and estimation uncertainty. This yields a verify-then-promote principle with decision-level guarantees, closed-loop improvement, and reusable verification under adaptive evidence allocation. Controlled finite-world experiments and agent benchmarks consistently validate the resulting attribution and qualification behavior. More broadly, reliable world-model use should be judged at the decision boundary: the key question is not whether a model is accurate in general, but whether current evidence is sufficient to justify the action that depends on it.

## REFERENCES

Romina Abachi, Mohammad Ghavamzadeh, and Amir-massoud Farahmand. Policy-aware model learning for policy gradient methods, 2020. URL https://arxiv.org/abs/2003.00030.

Anastasios N. Angelopoulos and Stephen Bates. A gentle introduction to conformal prediction and distribution-free uncertainty quantification. Foundations and Trends in Machine Learning, 16(4): 494–591, 2023. doi: 10.1561/2200000101.

Anthropic. Introducing Claude Opus 5. https://www.anthropic.com/news/ claude-opus-5, 2026.

Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, and Nicolas Ballas. V-JEPA 2: Self-supervised video models enable understanding, prediction and planning, 2025. URL https://arxiv.org/abs/2506.09985.

Amir Bar, Gaoyue Zhou, Danny Tran, Trevor Darrell, and Yann LeCun. Navigation world models. In Proceedings ofthe IEEE/CVF Conference on Computer Vision and Pattern Recognition, pp. 15791–15801, 2025. URL https://openaccess.thecvf.com/content/CVPR2025/ html/Bar\_Navigation\_World\_Models\_CVPR\_2025\_paper.html.

Hyungjoo Chae, Namyoung Kim, Kai Tzu-iunn Ong, Minju Gwak, Gwanwoo Song, Jihoon Kim, Sunghwan Kim, Dongha Lee, and Jinyoung Yeo. Web agents with world models: Learning and leveraging environment dynamics in web navigation. In International Conference on Learning Representations, 2025.

Zhaorun Chen, Zhuokai Zhao, Kai Zhang, Bo Liu, Qi Qi, Yifan Wu, Tarun Kalluri, Xuefei Cao, Yuanhao Xiong, Haibo Tong, Huaxiu Yao, Hengduo Li, Jiacheng Zhu, Xian Li, Dawn Song, Bo Li, Jason Weston, and Dat Huynh. Scaling agent learning via experience synthesis. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=cf7qpBwttr.

Santiago Cifuentes. General agents contain world models, even under partial observability and stochasticity, 2026. URL https://arxiv.org/abs/2602.03146.

Florent Delgrange, Raphael Avalos, and Willem Ropke. Deep SPI: Safe policy improvement via ¨ world models. In International Conference on Learning Representations, 2026.

Pierluca D’Oro, Alberto Maria Metelli, Andrea Tirinzoni, Matteo Papini, and Marcello Restelli. Gradient-aware model-based policy search. Proceedings of the AAAI Conference on Artificial Intelligence, 34(4):3801–3808, 2020. doi: 10.1609/aaai.v34i04.5791.

Tianqing Fang, Hongming Zhang, Zhisong Zhang, Kaixin Ma, Wenhao Yu, Haitao Mi, and Dong Yu. WebEvolver: Enhancing web agent self-improvement with co-evolving world model. In Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing, pp. 8959–8975. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.emnlp-main. 454. URL https://aclanthology.org/2025.emnlp-main.454/.

Ignat Georgiev, Varun Giridhar, Nick Hansen, and Animesh Garg. PWM: Policy learning with multi-task world models. In International Conference on Learning Representations, 2025.

Aaron Grattafiori, Abhimanyu Dubey, Abhinav Jauhri, Abhinav Pandey, Abhishek Kadian, Ahmad Al-Dahle, Aiesha Letman, Akhil Mathur, Alan Schelten, Alex Vaughan, et al. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783, 2024.

Jiacheng Guo, Ling Yang, Peter Chen, Qixin Xiao, Yinjie Wang, Xinzhe Juan, Jiahao Qiu, Ke Shen, and Mengdi Wang. GenEnv: Difficulty-aligned co-evolution between LLM agents and environment simulators. In International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=5vsYRklFJf.

David Ha and Jurgen Schmidhuber. World models.¨ arXiv preprint arXiv:1803.10122, 2018.

Danijar Hafner, Jurgis Pasukonis, Jimmy Ba, and Timothy Lillicrap. Mastering diverse control tasks through world models. Nature, 640:647–653, 2025. doi: 10.1038/s41586-025-08744-2.

Nicklas Hansen, Hao Su, and Xiaolong Wang. TD-MPC2: Scalable, robust world models for continuous control. In International Conference on Learning Representations, 2024.

Tairan Huang, Siyu Shang, Qiang Chen, Xiu Su, and Yi Chen. Tools as continuous flow for evolving agentic reasoning, 2026. URL https://arxiv.org/abs/2605.07339.

Michael Janner, Justin Fu, Marvin Zhang, and Sergey Levine. When to trust your model: Model-based policy optimization. In Advances in Neural Information Processing Systems, volume 32, 2019.

Carlos E. Jimenez, John Yang, Alexander Wettig, Shunyu Yao, Kexin Pei, Ofir Press, and Karthik Narasimhan. SWE-bench: Can language models resolve real-world GitHub issues? In International Conference on Learning Representations, 2024.

Minsoo Kim and Seung-won Hwang. CoEx – co-evolving world-model and exploration. In Findings of the Association for Computational Linguistics: EMNLP 2025, pp. 21629–21651. Association for Computational Linguistics, 2025. doi: 10.18653/v1/2025.findings-emnlp.1179. URL https: //aclanthology.org/2025.findings-emnlp.1179/.

Minghao Li, Yingxiu Zhao, Bowen Yu, Feifan Song, Hangyu Li, Haiyang Yu, Zhoujun Li, Fei Huang, and Yongbin Li. API-bank: A comprehensive benchmark for tool-augmented LLMs. In Proceedings ofthe 2023 Conference on Empirical Methods in Natural Language Processing, pp. 3102–3116. Association for Computational Linguistics, 2023. doi: 10.18653/v1/2023.emnlp-main. 187. URL https://aclanthology.org/2023.emnlp-main.187/.

Xiao Liu, Hao Yu, Hanchen Zhang, Yifan Xu, Xuanyu Lei, Hanyu Lai, Yu Gu, Hangliang Ding, Kaiwen Men, Kejuan Yang, Shudan Zhang, Xiang Deng, Aohan Zeng, Zhengxiao Du, Chenhui Zhang, Sheng Shen, Tianjun Zhang, Yu Su, Huan Sun, Minlie Huang, Yuxiao Dong, and Jie Tang. AgentBench: Evaluating LLMs as agents. In International Conference on Learning Representations, 2024.

Youwei Liu, Jian Wang, Hanlin Wang, and Wenjie Li. CoMAP: Co-evolving world models and agent policies for LLM agents, 2026. URL https://arxiv.org/abs/2606.02372. Accepted to the EMNLP 2026 Main Conference.

Zichen Liu, Guoji Fu, Chao Du, Wee Sun Lee, and Min Lin. Continual reinforcement learning by planning with online world models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 38397–38423. PMLR, 2025. URL https://proceedings.mlr.press/v267/liu25p.html.

Ning Lu, Baijiong Lin, Shengcai Liu, Jiahao Wu, Haoze Lv, Yanbin Wei, Lingting Zhu, Shengju Qian, Xin Wang, Ying-Cong Chen, Qi Wang, and Ke Tang. Policy and world modeling co-training for language agents, 2026. URL https://arxiv.org/abs/2606.02388.

OpenAI. GPT-6 Astra System Card. https://deploymentsafety.openai.com/ gpt-6-astra, 2026.

Shishir G. Patil, Huanzhi Mao, Fanjia Yan, Charlie Cheng-Jie Ji, Vishnu Suresh, Ion Stoica, and Joseph E. Gonzalez. The Berkeley function calling leaderboard (BFCL): From tool use to agentic evaluation of large language models. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pp. 48371–48392. PMLR, 2025. URL https://proceedings.mlr.press/v267/patil25a.html.

Jonathan Richens, Tom Everitt, and David Abel. General agents need world models. In Proceedings ofthe 42nd International Conference on Machine Learning, volume 267 of Proceedings ofMachine Learning Research, pp. 51659–51687, 2025. URL https://proceedings.mlr.press/ v267/richens25a.html.

Marc Rigter, Minqi Jiang, and Ingmar Posner. Reward-free curricula for training robust world models. In International Conference on Learning Representations, 2024.

Lloyd Russell, Anthony Hu, Lorenzo Bertoni, George Fedoseev, Jamie Shotton, Elahe Arani, and Gianluca Corrado. GAIA-2: A controllable multi-view generative world model for autonomous driving, 2025. URL https://arxiv.org/abs/2503.20523.

Julian Schrittwieser, Ioannis Antonoglou, Thomas Hubert, Karen Simonyan, Laurent Sifre, Simon Schmitt, Arthur Guez, Edward Lockhart, Demis Hassabis, Thore Graepel, Timothy Lillicrap, and David Silver. Mastering Atari, Go, chess and shogi by planning with a learned model. Nature, 588: 604–609, 2020. doi: 10.1038/s41586-020-03051-4.

Venkat Krishna Srinivasan, Zhen Dong, Banghua Zhu, Brian Yu, Hanzi Mao, Damon Mosk-Aoyama, Kurt Keutzer, Jiantao Jiao, and Jian Zhang. NexusRaven: A commercially-permissive language model for function calling. In NeurIPS 2023 Foundation Models for Decision Making Workshop, 2023. URL https://nips.cc/virtual/2023/82920.

Shenghua Wan, Le Gan, and De-Chuan Zhan. Learning to be uncertain: Pre-training world models with horizon-calibrated uncertainty. In International Conference on Learning Representations, 2026.

Hang Wang, Xin Ye, Feng Tao, Chenbin Pan, Abhirup Mallik, Burhan Yaman, Liu Ren, and Junshan Zhang. AdaWM: Adaptive world model based planning for autonomous driving. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ d4c745dcbaf8ef0d7e145754e31b1516-Abstract-Conference.html.

Zhaoyang Wang, Canwen Xu, Boyi Liu, Yite Wang, Siwei Han, Zhewei Yao, Huaxiu Yao, and Yuxiong He. Agent world model: Infinity synthetic environments for agentic reinforcement learning, 2026. URL https://arxiv.org/abs/2602.10090. Accepted to ICML 2026.

Tianbao Xie, Danyang Zhang, Jixuan Chen, Xiaochuan Li, Siheng Zhao, Ruisheng Cao, Toh Jing Hua, Zhoujun Cheng, Dongchan Shin, Fangyu Lei, Yitao Liu, Yiheng Xu, Shuyan Zhou, Silvio Savarese, Caiming Xiong, Victor Zhong, and Tao Yu. OSWorld: Benchmarking multimodal agents for openended tasks in real computer environments. In Advances in Neural Information Processing Systems, volume 37, 2024. URL https://proceedings.neurips.cc/paper\_files/paper/ 2024/hash/5d413e48f84dc61244b6be550f1cd8f5-Abstract-Datasets\_ and\_Benchmarks\_Track.html. Datasets and Benchmarks Track.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, Chenxu Lv, et al. Qwen3 technical report. arXiv preprint arXiv:2505.09388, 2025.

Xuan Yao, Junyu Gao, and Changsheng Xu. NavMorph: A self-evolving world model for vision-andlanguage navigation in continuous environments. In Proceedings of the IEEE/CVF International Conference on Computer Vision, pp. 5536–5546, 2025.

Tianhe Yu, Garrett Thomas, Lantao Yu, Stefano Ermon, James Y. Zou, Sergey Levine, Chelsea Finn, and Tengyu Ma. MOPO: Model-based offline policy optimization. In Advances in Neural Information Processing Systems, volume 33, 2020. URL https://proceedings.neurips.cc/ paper/2020/hash/a322852ce0df73e204b7e67cbbef0d0a-Abstract.html.

Z.ai. GLM-5.3-Flash. https://huggingface.co/zai-org/GLM-5.3-Flash, 2026.

Jiahan Zhang, Muqing Jiang, Nanru Dai, Taiming Lu, Arda Uzunoglu, Shunchi Zhang, Yana Wei, Jiahao Wang, Vishal M. Patel, Paul Pu Liang, Daniel Khashabi, Cheng Peng, Rama Chellappa, Tianmin Shu, Alan Yuille, Yilun Du, and Jieneng Chen. World-in-world: World models in a closed-loop world. In International Conference on Learning Representations, 2026a. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ 5b4263be85820683d78675cc18d2efc7-Abstract-Conference.html.

Xuan Zhang, Wenxuan Zhang, See-Kiong Ng, and Yang Deng. Self-evolving world models for LLM agent planning, 2026b. URL https://arxiv.org/abs/2606.30639. Accepted at EMNLP 2026 Findings.

Shuyan Zhou, Frank F. Xu, Hao Zhu, Xuhui Zhou, Robert Lo, Abishek Sridhar, Xianyi Cheng, Tianyue Ou, Yonatan Bisk, Daniel Fried, Uri Alon, and Graham Neubig. WebArena: A realistic web environment for building autonomous agents. In International Conference on Learning Representations, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/ hash/4410c0711e9154a7a2d26f9b3816d1ef-Abstract-Conference.html.

Yuxin Zuo, Zikai Xiao, Li Sheng, Fei Huang, Jianhong Tu, Yuxuan Liu, Tianyi Tang, Xiaomeng Hu, Yang Su, Qingfeng Lan, et al. Qwen-AgentWorld: Language world models for general agents. arXiv preprint arXiv:2606.24597, 2026.

## A MATHEMATICAL INTERFACE AND NOTATION

For reference, the paired world notation is

$$
\mathcal { M } _ { z } = ( S , A , P _ { z } , r _ { z } , \rho _ { z } , H ) , \qquad \widehat { \mathcal { M } } _ { z } = ( S , A , \widehat { P } _ { z } , \widehat { r } _ { z } , \rho _ { z } , H ) .
$$

## A.1 OBJECTS HELD FIXED IN AN INTERVENTION

The task index z identifies a reference environment, not merely a generated description. Once $z$ is fixed, both worlds share state space S, action space A, initial law $\rho ,$ and horizon H. Their transition kernels are $P , \widehat { P }$ and bounded rewards are $r , { \widehat { r } } .$ All are measurable. An episode contains states and actions at times $0 , \ldots , H - 1$ ; the transition following the final reward is omitted. This convention explains why transition error is summed only through $H - 2$

The kernel construction on standard Borel spaces defines a unique trajectory law. For a measurable policy $\pi ,$ let $p ^ { \pi }$ and ${ \widehat { p } } ^ { \pi }$ denote these two laws; $\mu _ { t } ^ { \pi }$ and $\widehat { \mu } _ { t } ^ { \pi }$ are their state–action marginals. Policies may depend on time, already included in the state. $V _ { t } ^ { \pi } ( s )$ and $\widehat { V } _ { t } ^ { \pi } ( s )$ denote expected rewards from time t onward. Returns are finite and lie in $[ 0 , H R _ { b } ]$ . Suprema over policy classes are used when maximizers are not guaranteed to exist.

$\mathsf { A } _ { \phi }$ is a measurable, possibly randomized model-to-policy operator. Returns of randomized outputs average the return of each realized policy over a fixed seed law. This is not generally the return of the pointwise mean of Markov policy kernels. Oracle replacement changes the model argument only: the operator, available actions, budget, and seed law remain fixed. An oracle that also changes the optimizer or observation interface defines a different intervention.

In the optional differential analysis, $\theta \in \Theta \subseteq \mathbb { R } ^ { d }$ parameterizes a behavior evaluated by predictive policy search. It is distinct from the fixed competence description ϕ used to define counterfactual attribution. At an update, the kernels and rewards do not depend on the differentiation variable. Models may be refitted between updates; every certificate must then be checked for the new snapshot.

## A.2 PARTIAL OBSERVABILITY AND LEARNED REPRESENTATIONS

An observable history can serve as the state: at time t take the full observation–action history, including any observed rewards or tool outputs. The true kernel is the conditional law of the next observable history under an action, and the learned kernel predicts the same object. This is a common measurable interface on which the finite-horizon arguments apply. It does not assume access to hidden physical states or equality of internal representations.

For a latent model supporting a history-dependent policy, one route is to specify a history-space kernel, for example through a decoder. Alternatively, when the policy consumes only $f ( h )$ , Proposition 5 compares true history-conditional latent transitions directly with the learned latent kernel. Its uniform error includes representation aliasing; neither a decoder nor exact Markov sufficiency is then assumed. Passive latent prediction loss alone supplies neither guarantee. Likewise, a generated executable environment that defines a new task is not automatically an approximation to a reference environment. The evaluation correspondence must be specified.

## A.3 PREDICTIVE DECISION PROCEDURES

A planner with specification ϕ maps the supplied world $\mathcal { N }$ to a behavior $\mathsf { A } _ { \phi } ( \mathcal { N } )$ . It may select a plan once or replan after each observation. With a fixed internal supplied world, the latter procedure defines a history-based policy. Randomized search, finite imagined rollouts and a fixed rollout budget are part of the operator; no parameter update is needed.

Attribution recomputes the operator with the oracle world and compares $J _ { \mathcal { M } } ( \mathsf { A } _ { \phi } ( \mathcal { M } ) )$ with $J _ { \mathcal { M } } ( \mathbb { A } _ { \phi } ( \widehat { \mathcal { M } } ) )$ . Certification instead first constructs $\pi _ { 0 } , \pi _ { 1 }$ and evaluates these same policies in every plausible world. In particular, evaluating $J _ { \mathcal { N } } ( \pi _ { 1 } )$ does not replace the internal model or rerun the candidate search. A realized randomized output can be conditioned on before fresh evaluation; reuse of the data that selected it requires a simultaneous certificate.

For repeated action selection, a policy-pair certificate applies to the specified complete continuation. Theorem 3 instead fixes the reference continuation in each local action comparison and controls the resulting policy through the performance-difference identity. This distinction connects a frozen predictive model to an interactive agent without assuming that a short imagined plan will actually be followed.

## A.4 NOTATION LEDGER

Table 4: Notation for attribution and use-dependent reliability.
<table><tr><td>Symbol</td><td>Meaning</td></tr><tr><td> $z , t , i , k$ </td><td>Task, within-episode time, sample or accepted-update index, verification</td></tr><tr><td> ${ \mathcal { M } } , { \widehat { \mathcal { M } } }$ </td><td>round True environment and its learned approximation on a common interface</td></tr><tr><td> $P , \widehat { P } , r , \widehat { r }$ </td><td>Transition kernels and reward functions</td></tr><tr><td> $\rho , H , R _ { b }$ </td><td>Shared initial law, horizon, and common nonnegative reward bound</td></tr><tr><td> $\Pi , \mathsf { A } _ { \phi }$ </td><td>Reference policy class and fixed model-to-policy operator</td></tr><tr><td> $J , \widehat { J } , J ^ { * }$ </td><td>True return, model return, supremal true return in II</td></tr><tr><td> ${ \widehat { \pi } } , { \bar { \pi } } ^ { \circ }$ </td><td>Deployed policy and oracle-model counterfactual</td></tr><tr><td> $R , A , W$ </td><td>Observed regret, intrinsic agent regret, signed model effect</td></tr><tr><td> $\mu _ { t } ^ { \pi } , \widehat { \mu } _ { t } ^ { \pi }$ </td><td>True and learned state-action occupancies</td></tr><tr><td> $\Delta _ { t } ^ { \pi } , \varepsilon _ { \mathrm { I } } ( \pi )$ </td><td>Cross-model Bellman residual and value-error certificate</td></tr><tr><td> $\ell _ { t } , \delta$ </td><td>Policy-independent error envelope and policy-TV radius; δ denotes a</td></tr><tr><td> $\psi _ { \boldsymbol { \theta } } , G , S _ { t }$ </td><td>sampling error probability where stated Policy score, uniform score bound, accumulated score through t</td></tr><tr><td> $d _ { t } , e , D _ { t }$ </td><td>Expected local transition error, its sum, and prefix coupling bound</td></tr><tr><td> $g , { \widehat { g } } , { \widetilde { g } }$ </td><td>True gradient, exact imagined gradient, sampled imagined gradient</td></tr><tr><td> $C _ { H } , b _ { \theta } , \xi$ </td><td>Trajectory-score bound, model gradient-bias bound, sampling radius</td></tr><tr><td> $b , h , L , \eta$ </td><td>Total gradient radius, sampled gradient norm, smoothness, step size</td></tr><tr><td> $x , u , q _ { i } , U _ { \alpha } , T$ </td><td>Snapshot, uncertainty score, residual, calibrated upper bound, promotion</td></tr><tr><td> $\mathcal { H } _ { k } , \varepsilon _ { k } , N _ { K }$ </td><td>margin Verification history, round risk budget, count of non-improving executed steps</td></tr><tr><td> $x , \pi _ { 0 } , \pi _ { 1 }$ </td><td>Decision request, reference policy, candidate policy</td></tr><tr><td> $\Gamma _ { { \cal M } } , { \cal C } , \ell _ { \cal C }$ </td><td>True policy contrast, world confidence set, robust contrast</td></tr><tr><td> $\mathcal { F } _ { \mathrm { W } } , \mathcal { F } _ { \mathrm { A } }$ </td><td>Model-evidence frontier and certified agent frontier</td></tr><tr><td> $S , B , \xi , T$ </td><td>Estimated contrast, model-error radius, estimation radius, certified margin</td></tr></table>

Local symbols are defined when first introduced. $\| \cdot \| _ { 2 }$ is Euclidean norm, $\langle \cdot , \cdot \rangle$ its inner product, and the norm of a matrix is its induced operator norm. $[ a ] _ { + } = \operatorname* { m a x } \{ a , 0 \}$ . E and P denote expectation and probability under the law specified in context.

## A.5 LOGICAL DEPENDENCIES

Table 5: What each guarantee requires and what it does not supply.
<table><tr><td>Result</td><td>Required information or regularity</td><td>Not implied</td></tr><tr><td>Theorem 1</td><td>Passive episodes of a fixed composition</td><td>Impossibility after interventions</td></tr><tr><td>Theorem 2</td><td>Common measurable interface; bounded rewards</td><td>An error bound from arbitrary pixel loss</td></tr><tr><td>Proposition 4</td><td>Uniform policy-TV control</td><td>Reliability after unrestricted policy search</td></tr><tr><td>Theorem 3</td><td>Simultaneous conditional value bounds; fixed reference</td><td>Guaranteed improvement from an arbitrary critic</td></tr><tr><td>Theorem 5</td><td>Frozen worlds; common reward; bounded policy score</td><td>Gradient fidelity for shared-parameter model updates</td></tr><tr><td>Theorem 6</td><td>Gradient-error ball; local smoothness; feasible step</td><td>Harmfulness of every rejected step</td></tr><tr><td>Proposition 7</td><td>Independent score fitting;</td><td>Coverage conditional on selection</td></tr><tr><td>Proposition 8</td><td>exchangeable true error labels Finite spaces; fresh generative row</td><td>Cheap verification in arbitrary visual</td></tr><tr><td>Theorem 4</td><td>queries One stationary task; uniform world</td><td>domains Acceptance of all future proposals</td></tr><tr><td>Theorem 7</td><td>confidence; valid estimation Conditional risk control after</td><td>Improvement of every task when</td></tr><tr><td>Appendix F.2</td><td>adaptive task choice Fixed objective; relative accuracy;</td><td>only one task is audited A bound on waiting time or real</td></tr><tr><td>Theorem 8</td><td>accepted updates Explicit no-transfer, monotone precedence response</td><td>samples Universal advantage over scalar curricula</td></tr></table>

## B ATTRIBUTION: IMPOSSIBILITY, INSTABILITY, AND RECOVERY

The attribution continuum in Theorem 1 is

$$
( A , W ) = ( \lambda , c - \lambda ) .
$$

## B.1 COMPLETE FIXED-OPERATOR CONSTRUCTION

Proof of Theorem 1. Let $S = \{ s , g , b \}$ and $A = \{ a _ { 0 } , a _ { 1 } , a _ { 2 } \}$ . Start at $s ,$ use horizon two, and set $r ( s , a ) = r ( b , a ) = 0$ and $r ( g , a ) = c$ for every action. The states $^ { g , }$ b are absorbing. For $\lambda \in [ 0 , c ]$ define

$$
{ \cal P } _ { \lambda } ( g \mid s , a _ { 0 } ) = 0 , { \cal P } _ { \lambda } ( g \mid s , a _ { 1 } ) = 1 - \lambda / c , { \cal P } _ { \lambda } ( g \mid s , a _ { 2 } ) = 1 .
$$

The remaining probability goes to $b .$ The supplied world model $\widehat { P }$ sends every action at s to $b$ and agrees with the true kernels elsewhere. The rewards are known and shared.

The operator evaluates only actions $a _ { 0 } , a _ { 1 }$ using its supplied world model and chooses the higherreturn action, breaking ties toward $a _ { 0 }$ . Its terminal action is fixed arbitrarily. This same operator is used for every λ; it is not hidden from the observer. Its restricted search is a concrete competence limitation, while the reference class contains the policy choosing $a _ { 2 }$

Under ${ \widehat { P } } ,$ deployment always chooses $a _ { 0 }$ . Under $P _ { \lambda }$ , this produces $( s , a _ { 0 } , 0 , b , a _ { 0 } , 0 )$ with probability one. For $\lambda < c ,$ oracle replacement chooses $a _ { 1 }$ and achieves $c - \lambda ;$ at $\lambda = c$ the tie-breaking rule chooses $a _ { 0 }$ and still achieves $c - \lambda = 0$ . The reference value is always c. Hence $R = c , A = \lambda , W =$ $c - \lambda$

Let O be generated by any number of deployed episodes, the supplied world model, the known operator, and independent randomization used by an estimator. All these objects have the same law for every λ. For any O-measurable real estimate $X$

$$
\begin{array} { l } { \underset { \displaystyle \lambda \in [ 0 , c ] } { \operatorname* { s u p } } \mathbb { E } _ { \lambda } | X - \lambda | \geq \operatorname* { m a x } \{ \mathbb { E } _ { 0 } | X | , \mathbb { E } _ { c } | X - c | \} } \\ { \mathrm { ~ } } \\ { \mathrm { ~ } \geq \frac { 1 } { 2 } \big ( \mathbb { E } _ { 0 } | X | + \mathbb { E } _ { c } | X - c | \big ) } \\ { \mathrm { ~ } = \displaystyle \frac { 1 } { 2 } \int \big ( | x | + | x - c | \big ) \operatorname { Q } ( \mathrm { d } x ) } \\ { \mathrm { ~ } \geq \frac { c } { 2 } , } \end{array}
$$

where $\mathsf { Q }$ is the common law of X and the last step is the triangle inequality. Conversely, $X = c / 2$ obeys

$$
\operatorname* { s u p } _ { \lambda \in [ 0 , c ] } | X - \lambda | = c / 2 ,
$$

so the lower bound is exact. The endpoint classification argument follows from the same common-law experiment: the two endpoint labels have equal prior probability and identical observations, hence Bayes and minimax error are both $1 / 2$ . This proves Theorem 1 for arbitrary passive sample size, including infinite passive repetition.

Notice that the supplied world model is exact on the deployed occupancy: its error there is zero. The missing information is about untried actions. Direct observation of the world-model parameters does not remove this obstruction. Repeated controlled trials of $a _ { 1 }$ identify its success probability asymptotically; a single trial need not determine the attribution. The construction isolates why passive return and on-policy fit cannot alone justify an attribution or a certificate for new interventions.

## B.2 FINITE-SAMPLE MONTE CARLO PLANNING

The obstruction persists when action values are estimated from finitely many imagined rollouts. Use the same states, reward, supplied world model, and reference class as above, but let $P ( g$ $s , a _ { 1 } ) = p$ range over [0, 1]. Fix a positive integer N. The planner samples $N$ independent supplied world episodes starting with each of $a _ { 0 } , a _ { 1 }$ , compares the sample return means, and outputs the corresponding deterministic initial action. Ties select $^ { a _ { 0 } ; }$ terminal actions are fixed.

Under the supplied world model all 2N imagined episodes return zero, so the complete search transcript and output policy are identical for all true worlds. Subsequent real deployment of $a _ { 0 }$ is also identical. Under oracle replacement, $a _ { 0 }$ still always returns zero, whereas $a _ { 1 }$ is selected if and only if at least one of its $N$ imagined episodes succeeds. Its selection probability is $1 - ( 1 - p ) ^ { N }$ Evaluation uses a new episode independent of search, giving

$$
J ( \pi ^ { \circ } ) = c f _ { N } ( p ) , \qquad f _ { N } ( p ) = p [ 1 - ( 1 - p ) ^ { N } ] .
$$

The function $f _ { N }$ is continuous, has $f _ { N } ( 0 ) = 0 , f _ { N } ( 1 ) = 1$ , and for $p \in ( 0 , 1 )$

$$
f _ { N } ^ { \prime } ( p ) = 1 - ( 1 - p ) ^ { N } + N p ( 1 - p ) ^ { N - 1 } > 0 .
$$

For every $\lambda \in [ 0 ,$ c], there is therefore a unique $p$ with $c f _ { N } ( p ) = c - \lambda$ . Since $J ^ { * } = c$ and $J ( \widehat { \pi } ) = 0$ the same $( A , \dot { W } ) \dot { = } ( \lambda , c - \lambda )$ continuum results. The common-law estimation and classification proofs apply even when the observer also sees every imagined search record.

The planner, rollout budget, and supplied world model are fixed throughout this family. The oracle does not increase compute or give a better optimizer; it changes only the predictive law used for look-ahead. This finite-sample construction requires no change of agent parameters.

## B.3 A FIXED TRUE-WORLD VARIANT

A different construction keeps the true one-step MDP fixed but varies the agent operator. Use actions with rewards $c , 0$ and one supplied world model distinct from the truth. For each λ, let the operator choose the zero-reward action under the supplied world model and the good action with probability $1 - \lambda / c$ under the true model. This gives the same decomposition and minimax bounds already in a one-state MDP. Theorem 1 is stronger with respect to what the observer knows about the operator: there the operator and supplied world model are fixed.

## B.4 PREDICTIVE RELIABILITY DOES NOT IDENTIFY A BLACK-BOX OPERATOR’S MODELEFFECT

Proposition 1 (No universal modulus for failure ownership). There exist one fixed agent operator and true one-step world such that, for every sufficiently small $\epsilon > 0 _ { : }$ , a supplied world model satisfies $\begin{array} { r } { \operatorname* { s u p } _ { \pi \in \Pi } | J ( \pi ) - \widehat { J } ( \pi ) | \leq \epsilon b u t W = c f o r } \end{array}$ a constant $c > 0$ independent of ϵ.

Proof. Use three actions with true rewards 0, 0, c. The model changes only the first reward from zero to ϵ, with $0 < \epsilon \leq R _ { b }$ . Define the operator to choose the second action if the supplied first reward is positive, and the third action otherwise. This is one fixed measurable operator. It achieves zero in deployment and c after oracle replacement, so $W = c$ . The value of any policy changes by at most ϵ. Thus no function $\omega ( \epsilon ) \to 0$ can universally bound $| W |$ for arbitrary operators.

A continuity or optimization-residual assumption on the operator can exclude this example, but it cannot be omitted. The value and update comparisons in the main text avoid that extra assumption by certifying the policies actually compared. Their validity is not a claim that they estimate A or $W$

The sign of $W$ is equally unrestricted. Reverse the two branches of the operator in the preceding example. The wrong model now induces the good action and oracle replacement the bad action, giving $W = - c$ . The decomposition remains an exact signed identity.

## B.5 FINITE-SAMPLE ATTRIBUTION WITH AN ORACLE INTERVENTION

Proposition 2 (Interventional recovery). $F i x \widehat { \pi } , \pi ^ { \circ }$ and a reference policy π before evaluation, with $0 \leq J ^ { * } - J ( \pi _ { r } ) \leq \omega$ . Use m independent episodes of each policy and write their sample means as ${ \overline { { J } } } , { \overline { { J } } } ^ { \circ } , { \overline { { J } } } _ { r } .$ For $\delta \in ( 0 , 1 )$ ), put

$$
t = H R _ { b } \sqrt { \frac { \log ( 6 / \delta ) } { 2 m } } , \qquad \widehat { A } = \overline { { J } } _ { r } - \overline { { J } } ^ { \circ } , \qquad \widehat { W } = \overline { { J } } ^ { \circ } - \overline { { J } } .
$$

With probability at least $1 - \delta ,$

$$
| \widehat { A } - A | \leq 2 t + \omega , \qquad | \widehat { W } - W | \leq 2 t .
$$

Proof. A bounded return has range $H R _ { b }$ . Hoeffding’s inequality makes each sample mean accurate to t except with probability $\delta / 3 . { \overset { } { \mathbf { A } } }$ union bound yields simultaneous accuracy. Each difference has sampling error at most $2 t ;$ only the agent-regret estimate has the additional reference-policy error ω. Cross-policy independence is unnecessary, so paired random numbers are allowed if each policy’s episodes remain independent.

This result requires the oracle-model counterfactual policy, not merely more episodes of deployment.   
It quantifies evaluation after the identifying intervention; it does not remove the intervention’s cost.

## C DECISION RELIABILITY: FULL DERIVATIONS

Theorem 2 also controls planning regret. If ${ \widehat { J } } ( { \widehat { \pi } } ) \geq \operatorname* { s u p } _ { \pi \in \Pi } { \widehat { J } } ( \pi ) - \delta _ { p }$ , then the planning-regret inequality in Corollary 1 follows. The proof below takes a supremum and does not require an optimal policy to exist.

## C.1 ROBUST POLICY CONTRASTS AND THE ORDER OF EVALUATION

Fix a use request x, including the two output policies, and a nonempty class C of worlds sharing the evaluation interface. Bounded rewards imply −H $R _ { b } \leq \Gamma _ { \mathcal { N } } ( x ) \leq \bar { H } \bar { R } _ { b }$ . Let $\underline { { \ell } } ( x )$ be any computable lower bound on $\ell _ { C } ( x )$ . By the definition of infimum,

$$
\mathcal { M } \in \mathcal { C } \quad \Longrightarrow \quad \Gamma _ { \mathcal { M } } ( x ) \geq \ell _ { \mathcal { C } } ( x ) \geq \ell ( x ) .
$$

On an event where ${ \mathcal { M } } \in { \mathcal { C } } .$ , this deterministic implication holds for all requests at once, including any selected as a function of C. No union bound over policies or use requests is required. For random sets and selectors we assume the displayed infima and events are measurable; the explicit finite-state bounds used in the paper are measurable functions of finite arrays.

A uniform improvement guarantee with positive constant a exists if and only if $\ell _ { C } ( x ) > 0 \colon$ one direction takes infima, and the other chooses $\boldsymbol { a } = \ell \boldsymbol { c } ( \boldsymbol { x } )$ . This equivalence is for the information represented by ${ \mathcal { C } } ,$ not for the unknown true world alone. A computable lower bound may fail even when the exact infimum is positive.

If C is compact and $\Gamma _ { \mathcal { N } } ( x )$ is continuous in ${ \mathcal { N } } .$ its image is a nonempty compact subset of $\mathbb { R }$ and contains its infimum. Thus $\ell _ { C } ( x ) \leq 0$ exhibits a plausible world where the comparison is nonimproving. Without attainment, $\ell { c } = 0$ may coexist with strictly positive contrast in every world, as for contrasts $1 / j , j \geq 1$ . It still precludes a uniform positive margin.

For finite spaces and finite horizon, the expectation of a fixed history-based policy is a finite sum of products of transition probabilities and rewards. It is continuous in these arrays. Closed rowconfidence sets in the probability simplices are compact and nonempty; the empirical kernel is feasible. These observations justify attainment for the concrete verifier. Exact contrast minimization can remain computationally difficult, which is why the main text supplies tractable conservative bounds instead of assuming access to an exact robust optimizer.

A request compares a proposed behavior with an explicit reference behavior. The certificate does not establish global optimality; the planning bound controls suboptimality separately when an optimization residual is available.

## C.2 BELLMAN OPERATORS AND THE TELESCOPING IDENTITY

Proof of Theorem 2 and Corollary 1. For a policy π, define $\begin{array} { r } { r ^ { \pi } ( s ) = \int r ( s , a ) \pi ( \mathrm { d } a \mid s ) } \end{array}$ and

$$
( P ^ { \pi } f ) ( s ) = \int _ { A } \int _ { S } f ( s ^ { \prime } ) P ( \mathrm { d } s ^ { \prime } \mid s , a ) \pi ( \mathrm { d } a \mid s ) .
$$

Time dependence is understood through the augmented state. Both Bellman recursions hold for bounded measurable values: $V _ { t } ^ { \pi } = r ^ { \pi } + P ^ { \pi } V _ { t + 1 } ^ { \pi }$ and $\widehat { V } _ { t } ^ { \pi } = \widehat { r } ^ { \pi } + \widehat { P } ^ { \pi } \widehat { V } _ { t + 1 } ^ { \pi }$ . Subtracting gives the exact operator identity

$$
\begin{array} { r l } & { V _ { t } ^ { \pi } - \widehat { V } _ { t } ^ { \pi } = r ^ { \pi } - \widehat { r } ^ { \pi } + P ^ { \pi } V _ { t + 1 } ^ { \pi } - \widehat { P } ^ { \pi } \widehat { V } _ { t + 1 } ^ { \pi } } \\ & { \qquad = ( r - \widehat { r } ) ^ { \pi } + ( P ^ { \pi } - \widehat { P } ^ { \pi } ) \widehat { V } _ { t + 1 } ^ { \pi } + P ^ { \pi } ( V _ { t + 1 } ^ { \pi } - \widehat { V } _ { t + 1 } ^ { \pi } ) . } \end{array}
$$

Let $\rho _ { t }$ be the true state law, $d _ { t } = V _ { t } ^ { \pi } - \widehat { V } _ { t } ^ { \pi }$ , and $\begin{array} { r } { b _ { t } = \int \Delta _ { t } ^ { \pi } ( \cdot , a ) \pi ( \mathrm { d } a \mid \cdot ) } \end{array}$ . These local symbols are used only in this proof. Since $\rho _ { t + 1 } = \rho _ { t } P ^ { \pi }$ and $d _ { H } = 0$ , the operator recursion above yields

$$
\begin{array} { r l r } & { } & { \displaystyle \int d _ { t } \mathrm { d } \rho _ { t } = \int b _ { t } \mathrm { d } \rho _ { t } + \int P ^ { \pi } d _ { t + 1 } \mathrm { d } \rho _ { t } } \\ & { } & { \quad = \mathbb { E } _ { \mu _ { t } ^ { \pi } } \Delta _ { t } ^ { \pi } + \displaystyle \int d _ { t + 1 } \mathrm { d } \rho _ { t + 1 } , } \end{array}
$$

$$
\begin{array} { r l } & { \displaystyle \int d _ { 0 } \mathrm { d } \rho _ { 0 } = \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi } } \Delta _ { t } ^ { \pi } + \int d _ { H } \mathrm { d } \rho _ { H } } \\ & { \quad \quad \quad = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi } } \Delta _ { t } ^ { \pi } . } \end{array}
$$

Because the initial law is shared, the left-hand side is exactly $J ( \pi ) - { \widehat J } ( \pi )$

For finite signed measure $P - Q$ of total mass zero and bounded $f ,$ let $a = ( \operatorname* { s u p } f + \operatorname* { i n f } f ) / 2$ . Then

$$
| ( P - Q ) f | = | ( P - Q ) ( f - a ) | \leq 2 \mathrm { T V } ( P , Q ) \| f - a \| _ { \infty } = \mathrm { s p a n } ( f ) \mathrm { T V } ( P , Q ) .
$$

Applying this inequality to the residual gives

$$
\begin{array} { r l } & { | J ( \pi ) - \widehat { J } ( \pi ) | = \displaystyle \left| \sum _ { t = 0 } ^ { H - 1 } \int \Delta _ { t } ^ { \pi } \mathrm { d } \mu _ { t } ^ { \pi } \right| } \\ & { \qquad \leq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int \left( | r - \widehat { r } | + \displaystyle \left| \int \widehat { V } _ { t + 1 } ^ { \pi } \mathrm { d } ( P - \widehat { P } ) \right| \right) \mathrm { d } \mu _ { t } ^ { \pi } } \\ & { \qquad \leq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int \left( | r - \widehat { r } | + \mathrm { s p a n } ( \widehat { V } _ { t + 1 } ^ { \pi } ) \mathrm { T V } ( P , \widehat { P } ) \right) \mathrm { d } \mu _ { t } ^ { \pi } } \\ & { \qquad = \varepsilon _ { 1 } ( \pi ) . } \end{array}
$$

No common support of the transition kernels is required.

For the two-policy comparison, let $e _ { \pi } = J ( \pi ) - { \widehat J } ( \pi )$ . Then

$$
\begin{array} { r l } & { J ( \pi ^ { \prime } ) - J ( \pi ) = \widehat { J } ( \pi ^ { \prime } ) - \widehat { J } ( \pi ) + e _ { \pi ^ { \prime } } - e _ { \pi } } \\ & { \qquad \quad \geq \widehat { J } ( \pi ^ { \prime } ) - \widehat { J } ( \pi ) - | e _ { \pi ^ { \prime } } | - | e _ { \pi } | } \\ & { \qquad \quad \geq \widehat { J } ( \pi ^ { \prime } ) - \widehat { J } ( \pi ) - \varepsilon _ { \mathrm { I } } ( \pi ^ { \prime } ) - \varepsilon _ { \mathrm { I } } ( \pi ) . } \end{array}
$$

The symmetric argument gives the corresponding upper bound. If an optimal $\pi ^ { * }$ exists, the planning bound sharpens to

$$
J ( \pi ^ { * } ) - J ( \widehat { \pi } ) \leq \delta _ { p } + \varepsilon _ { \mathrm { I } } ( \pi ^ { * } ) + \varepsilon _ { \mathrm { I } } ( \widehat { \pi } ) .
$$

If not, apply this inequality to a sequence approaching $J ^ { * }$ , retaining the uniform supremum used in the main theorem.

## C.3 UNIFORM BOUNDS, EQUALITY EXAMPLES, AND HORIZON SCALING

$\mathrm { I f } \ | r - \widehat { r } | \leq \epsilon _ { r }$ and $\mathrm { T V } ( P , \widehat { P } ) \leq \epsilon _ { p }$ uniformly, then $0 \leq \widehat { V } _ { t + 1 } ^ { \pi } \leq ( H - t - 1 ) R _ { b }$ implies

$$
| J ( \pi ) - \widehat { J } ( \pi ) | \leq H \epsilon _ { r } + \frac { H ( H - 1 ) } { 2 } R _ { b } \epsilon _ { p } .
$$

Both model rewards and true rewards must obey the stated reward bound.

Proposition 3 (Two endpoints are necessary). For every $\beta \in ( 0 , R _ { b } / 2 ]$ , the right-hand side ofthe endpoint inequality above is attained with $\delta _ { p } = 0$

Proof. Take a one-step two-action task with true rewards 2β, 0 and model rewards $\beta , \beta .$ . Let the world-model optimizer break the tie toward the second action. Each endpoint value error is $\beta ,$ and the true regret is $2 \beta .$ Thus one endpoint error cannot simply be removed from a general planning comparison.

The transition coefficient in the uniform value bound above is first-order sharp. Consider a determin istic true chain with zero initial reward and reward $R _ { b }$ at each later nonabsorbing state. The model enters a zero-reward absorbing state with independent probability $\epsilon _ { p }$ at each transition. The actual discrepancy is

$$
R _ { b } \sum _ { t = 1 } ^ { H - 1 } \left[ 1 - ( 1 - \epsilon _ { p } ) ^ { t } \right] = \frac { H ( H - 1 ) } { 2 } R _ { b } \epsilon _ { p } + O ( H ^ { 3 } \epsilon _ { p } ^ { 2 } )
$$

as $\epsilon _ { p } \downarrow$ 0 for fixed H. A uniform reward offset attains the linear reward coefficient. These examples establish value-bound scaling, not minimax sharpness of the gradient horizon exponent.

## C.4 TRANSPORT TO A NEW POLICY

Define the policy-independent envelope $\ell _ { t } = | r - \widehat { r } | + ( H - t - 1 ) R _ { b } \mathrm { T V } ( P , \widehat { P } )$ . We prove the transport inequality the transport inequality above.

Proposition 4 (Verified-policy transport). If sup<sub>s</sub> $\mathrm { T V } ( \pi ( \cdot \mid s ) , \pi _ { 0 } ( \cdot \mid s ) ) \leq \delta ,$ , then

$$
\varepsilon _ { \mathrm { I } } ( \pi ) \leq \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi _ { 0 } } } \ell _ { t } + R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( H - t ) \operatorname* { m i n } \{ 1 , ( t + 1 ) \delta \} .
$$

Proof. Couple identical initial states. Whenever states agree, use a maximal coupling of the action kernels; conditional action disagreement has probability at most δ. If actions also agree, sample the identical true transition synchronously. After disagreement, any coupling preserving both marginal dynamics suffices.

The event that state–action pairs disagree by time t requires at least one of $t + 1$ action disagreements. A union bound, or a first-disagreement decomposition, gives

$$
\mathrm { T V } ( \mu _ { t } ^ { \pi } , \mu _ { t } ^ { \pi _ { 0 } } ) \leq \operatorname* { m i n } \{ 1 , ( t + 1 ) \delta \} .
$$

For $0 \leq f \leq B$ , the signed-measure argument above gives $\mathbb { E } _ { \mu } f - \mathbb { E } _ { \nu } f \leq B \mathrm { T V } ( \mu , \nu )$ . Apply it to $\ell _ { t } .$ whose range lies in $\mathsf { \bar { [ 0 , ( } } H - t ) R _ { b } \mathsf { ] }$ . Since the integrand defining $\varepsilon _ { \mathrm { I } } ( \pi )$ is bounded by $\ell _ { t } ,$ , summing proves the transport inequality above. Finally,

$$
\sum _ { t = 0 } ^ { H - 1 } ( H - t ) ( t + 1 ) = \frac { H ( H + 1 ) ( H + 2 ) } { 6 } .
$$

A coverage assumption provides another route. If $\mu _ { t } ^ { \pi } \ll \mu _ { t } ^ { \pi _ { 0 } }$ with Radon–Nikodym derivative bounded by $c _ { t } .$ , then nonnegativity gives

$$
\varepsilon _ { \mathrm { I } } ( \pi ) \leq \sum _ { t = 0 } ^ { H - 1 } c _ { t } \mathbb { E } _ { \mu _ { t } ^ { \pi _ { 0 } } } \ell _ { t } .
$$

This is a stated density-ratio assumption, not something guaranteed by low passive prediction loss. If a candidate policy visits an action absent from the validation support, neither this bound nor unrestricted reuse of an on-policy certificate is justified.

## C.5 WASSERSTEIN AND DISCOUNTED VARIANTS

Suppose $( S , d )$ is Polish, both next-state kernels have finite first moments, and $\widehat { V } _ { t + 1 } ^ { \pi }$ is $L _ { t + 1 }$ -Lipschitz. The Kantorovich–Rubinstein dual formula gives

$$
| J ( \pi ) - \widehat { J } ( \pi ) | \leq \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \mu _ { t } ^ { \pi } } \left[ | r - \widehat { r } | + L _ { t + 1 } W _ { 1 } ( P , \widehat { P } ) \right] .
$$

Here $W _ { 1 }$ is the first Wasserstein distance for metric d. The proof substitutes $| ( P - \widehat { P } ) \widehat { V } _ { t + 1 } ^ { \pi } | \leq$ $L _ { t + 1 } W _ { 1 } ( P , \widehat { P } )$ into the exact identity. The additional Lipschitz assumption is indispensable; a small state-space metric error does not control an arbitrary discontinuous continuation value.

For a stationary discounted problem with $\gamma \in \mathsf { \Gamma } ( 0 , 1 )$ , define $\begin{array} { r } { J ^ { \gamma } ( \pi ) = \mathbb { E } \sum _ { t \geq 0 } \gamma ^ { t } r _ { t } } \end{array}$ and $\mu _ { \gamma } ^ { \pi } =$ $\begin{array} { r } { ( 1 - \gamma ) \sum _ { t > 0 } \gamma ^ { t } \mu _ { t } ^ { \pi } } \end{array}$ . The bounded Bellman resolvent yields

$$
J ^ { \gamma } ( \pi ) - \widehat { J } ^ { \gamma } ( \pi ) = \frac { 1 } { 1 - \gamma } \mathbb { E } _ { \mu _ { \gamma } ^ { \pi } } [ r - \widehat { r } + \gamma ( P - \widehat { P } ) \widehat { V } ^ { \pi } ] .
$$

To see this, iterate $V - \widehat { V } = ( r - \widehat { r } ) ^ { \pi } + \gamma ( P ^ { \pi } - \widehat { P } ^ { \pi } ) \widehat { V } + \gamma P ^ { \pi } ( V - \widehat { V } )$ . The remaining term after n iterations has sup norm at most $2 \gamma ^ { \dot { n } } R _ { b } / ( 1 - \gamma )$ and vanishes. Hence uniform reward and transition bounds imply

$$
| J ^ { \gamma } ( \pi ) - \widehat { J } ^ { \gamma } ( \pi ) | \leq \frac { \epsilon _ { r } } { 1 - \gamma } + \frac { \gamma R _ { b } \epsilon _ { p } } { ( 1 - \gamma ) ^ { 2 } } .
$$

This is a value extension only; the main finite-horizon statistical constants are not reused unchanged in infinite horizon.

## C.6 CONDITIONAL CERTIFICATES AND CLOSED-LOOP COMPOSITION

Proof of Theorem 3. Fix a reference continuation $\pi _ { 0 }$ . For every admissible history state s at time t and first action $^ { a , }$ let $\pi ^ { a }$ choose a first and follow $\pi _ { 0 }$ thereafter. This is a policy on the residual horizon $H - t$ with initial law concentrated at s. Applying the residual identity to this conditional problem gives

$$
Q _ { t } ^ { 0 } ( s , a ) - \widehat { Q } _ { t } ^ { 0 } ( s , a ) = \sum _ { j = t } ^ { H - 1 } \mathbb { E } _ { P , \pi ^ { a } } \Big [ \Delta _ { j } ^ { \pi ^ { a } } ( s _ { j } , a _ { j } ) \mid s _ { t } = s \Big ] .
$$

Here $\Delta _ { j } ^ { \pi ^ { a } }$ uses the learned continuation value of that same policy. Thus a valid conditional envelope is

$$
b _ { t } ( s , a ) = \sum _ { j = t } ^ { H - 1 } \mathbb { E } _ { P , \pi ^ { a } } \left[ | r - \widehat { r } | + \mathrm { s p a n } ( \widehat { V } _ { j + 1 } ^ { \pi _ { 0 } } ) \mathrm { T V } ( P , \widehat { P } ) \mid s _ { t } = s \right] .
$$

It is an analytical quantity unless its components are certified. Uniform row errors give

$$
b _ { t } ( s , a ) \leq \sum _ { j = t } ^ { H - 1 } [ \epsilon _ { r } + ( H - j - 1 ) R _ { b } \epsilon _ { p } ] = ( H - t ) \epsilon _ { r } + \frac { ( H - t ) ( H - t - 1 ) } { 2 } R _ { b } \epsilon _ { p } .
$$

This proves (16). Finite-state audits bound all these conditional problems at once, including histories that a deployed agent has not yet visited.

Contrast-sensitive refinement. Let $\lambda _ { t } = \kappa _ { t } ( \cdot \mid s ) - \pi _ { 0 , t } ( \cdot \mid s )$ , a signed measure of mass zero, and let $| \lambda _ { t } |$ denote its total-variation measure. Then

$$
\left| d _ { t } - \widehat { d } _ { t } \right| = \left| \int ( Q _ { t } ^ { 0 } - \widehat { Q } _ { t } ^ { 0 } ) \mathrm { d } \lambda _ { t } \right| \leq \int b _ { t } \mathrm { d } | \lambda _ { t } | \leq \int b _ { t } \mathrm { d } \bigl ( \kappa _ { t } + \pi _ { 0 , t } \bigr ) .
$$

The first upper bound is the exact support function of the pointwise uncertainty class:

$$
\operatorname* { s u p } _ { | f ( a ) | \leq b _ { t } ( s , a ) } \left| \int f \mathrm { d } \lambda _ { t } \right| = \int b _ { t } \mathrm { d } | \lambda _ { t } | .
$$

To prove equality, take $f = b _ { t } \mathrm { d } \lambda _ { t } / \mathrm { d } | \lambda _ { t } |$ on the support of $| \lambda _ { t } |$ . The Radon–Nikodym derivative is +1 or −1 almost everywhere, so this choice is admissible and realizes the integral. Sharpness is for the stated value-error class, not a claim that every extremizer is a realizable MDP. For uniform $b _ { t } = b$ the refined radius is 2b $\mathrm { T V } ( \kappa _ { t } , \pi _ { 0 , t } )$ , and it vanishes when the proposed action law is unchanged. Either radius may be used in Theorem 3.

The performance-difference identity. Let $r _ { t } = r ( s _ { t } , a _ { t } )$ and write $\mathbb { E } _ { \nu }$ for expectation under the true trajectory law of the gated policy. By conditional expectation and the reference Bellman identity,

$$
\begin{array} { r l r } & { } & { { \mathbb { E } } _ { \nu } [ r _ { t } + V _ { t + 1 } ^ { \pi _ { 0 } } ( s _ { t + 1 } ) \mid s _ { t } ] = \displaystyle \int Q _ { t } ^ { 0 } ( s _ { t } , a ) \nu _ { t } ( \mathrm { d } a \mid s _ { t } ) , } \\ & { } & { V _ { t } ^ { \pi _ { 0 } } ( s _ { t } ) = \displaystyle \int Q _ { t } ^ { 0 } ( s _ { t } , a ) \pi _ { 0 , t } ( \mathrm { d } a \mid s _ { t } ) , } \\ & { } & { { \mathbb { E } } _ { \nu } [ r _ { t } + V _ { t + 1 } ^ { \pi _ { 0 } } ( s _ { t + 1 } ) - V _ { t } ^ { \pi _ { 0 } } ( s _ { t } ) ] = { \mathbb { E } } _ { \rho _ { t } ^ { \nu } } [ g _ { t } d _ { t } ] . } \end{array}
$$

Summing the left side cancels every intermediate value:

$$
\begin{array} { r l } & { J ( \nu ) - J ( \pi _ { 0 } ) = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { \nu } [ r _ { t } + V _ { t + 1 } ^ { \pi _ { 0 } } ( s _ { t + 1 } ) - V _ { t } ^ { \pi _ { 0 } } ( s _ { t } ) ] } \\ & { \qquad = \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) d _ { t } ( s ) \rho _ { t } ^ { \nu } ( \mathrm { d } s ) } \\ & { \qquad \geq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) [ \widehat { d } _ { t } ( s ) - B _ { t } ( s ) ] \rho _ { t } ^ { \nu } ( \mathrm { d } s ) } \\ & { \qquad \geq \displaystyle \sum _ { t = 0 } ^ { H - 1 } \int _ { S } g _ { t } ( s ) T _ { t } ( s ) \rho _ { t } ^ { \nu } ( \mathrm { d } s ) \geq 0 . } \end{array}
$$

The first equality uses $V _ { H } ^ { \pi _ { 0 } } = 0$ and $\mathbb { E } _ { \rho } V _ { 0 } ^ { \pi _ { 0 } } = J ( \pi _ { 0 } ) ;$ ; the inequalities use $| d _ { t } - \widehat { d _ { t } } | \leq B _ { t } , \widehat { d _ { t } } \geq S _ { t } - \xi _ { t }$ and $g _ { t } = \mathbb { I } \{ T _ { t } > 0 \}$ . A finite sum of nonnegative integrable terms is strictly positive if one term is positive on an event of positive probability. The result guarantees expected improvement, not samplewise dominance of realized rewards.

Short rollouts and terminal estimates. Suppose a conditional model rollout stops after $h \in$ $\{ 1 , \ldots , H - t \}$ steps and bootstraps with a measurable function $\widetilde { V }$ satisfying $\| \widetilde { V } - \widehat { \widehat { V } } _ { t + h } ^ { \pi _ { 0 } } \| _ { \infty } \leq \omega$ Let $\widetilde { Q } _ { t } ^ { 0 }$ be the resulting expected truncated return. The tower property in the learned world gives

$$
| \widetilde { Q } _ { t } ^ { 0 } - \widehat { Q } _ { t } ^ { 0 } | \leq \omega , \qquad | \widetilde { Q } _ { t } ^ { 0 } - Q _ { t } ^ { 0 } | \leq b _ { t } + \omega .
$$

Use $b _ { t } + \omega$ in (13) and add the separate Monte Carlo estimation radius. Without a terminal-value error bound, an arbitrary score for one predicted observation is not a continuation-value certificate. The same applies to truncated language-model reasoning scores and learned critics.

## D DIFFERENTIAL RELIABILITY OF PREDICTIVE POLICY SEARCH

The parameter θ indexes behaviors evaluated through one fixed learned world model. The following results are optional local sufficient conditions for the same policy contrast used in the main text. They require differentiable policies and are not assumed for discrete action selection or frozen language-agent inference.

## D.1 DIFFERENTIAL CERTIFICATES FOR PREDICTIVE POLICY SEARCH

Assume shared rewards and freeze both worlds while differentiating. Policy densities on an open $\Theta \subseteq \mathbb { R } ^ { d }$ have common support, are continuously differentiable, and have score $\psi _ { \theta } = \nabla _ { \theta }$ log $\pi _ { \theta }$ with $\| \psi _ { \theta } \| _ { 2 } \leq G$ $\begin{array} { r } { \| \mathsf { \Pi } _ { 2 } \le G . \mathrm { ~ L e t ~ } d _ { t } = \mathbb { E } _ { \mu _ { t } ^ { \pi _ { \theta } } } \operatorname { T V } ( P , \widehat { P } ) , D _ { t } = \operatorname* { m i n } \{ 1 , \sum _ { j < t } d _ { j } \} , e = \sum _ { t = 0 } ^ { H - 2 } d _ { t } , } \end{array}$ , and $C _ { H } =$ $G R _ { b } H ( H + 1 ) / 2$ . Empty sums are zero.

Theorem 5 (Gradient reliability). For $g = \nabla _ { \boldsymbol { \theta } } J ( \pi _ { \boldsymbol { \theta } } )$ and $\widehat g = \nabla _ { \boldsymbol { \theta } } \widehat J ( \pi _ { \boldsymbol { \theta } } )$

$$
\| g - \widehat g \| _ { 2 } \le 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) D _ { t } = : b _ { \theta } \le 2 C _ { H } \operatorname* { m i n } \{ 1 , e \} .
$$

$$
I f b _ { \theta } < \| \widehat { g } \| _ { 2 } , t h e n \left. g , \widehat { g } \right. \geq \| \widehat { g } \| _ { 2 } ( \| \widehat { g } \| _ { 2 } - b _ { \theta } ) > 0 .
$$

Proof of Theorem 5. Let $Q _ { t } , \widehat { Q } _ { t }$ denote the true and learned laws of the prefix through $( s _ { t } , a _ { t } )$ and set $\begin{array} { r } { S _ { t } = \sum _ { j = 0 } ^ { t } \psi _ { \theta } \big ( s _ { j } , a _ { j } \big ) } \end{array}$ . Differentiation under the trajectory integral and score centering give

$$
g = \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { P } [ S _ { t } r ( s _ { t } , a _ { t } ) ] , \qquad { \widehat { g } } = \sum _ { t = 0 } ^ { H - 1 } \mathbb { E } _ { { \widehat { P } } } [ S _ { t } r ( s _ { t } , a _ { t } ) ] .
$$

Sequential coupling gives $\mathrm { T V } ( Q _ { t } , \widehat { Q } _ { t } ) \leq D _ { t }$ . Thus

$$
\lVert g - \widehat { g } \rVert _ { 2 } \leq \sum _ { t = 0 } ^ { H - 1 } 2 \lVert S _ { t } r ( s _ { t } , a _ { t } ) \rVert _ { \infty } \mathrm { T V } ( Q _ { t } , \widehat { Q } _ { t } ) \leq b _ { \theta } ,
$$

where $\| F \| _ { \infty } = \operatorname* { s u p } \| F \| _ { 2 }$ for a vector-valued function. Cauchy–Schwarz and the gradient-ball geometry yield, for $b _ { \theta } < \| { \widehat { g } } \| _ { 2 }$

$$
\langle g , { \widehat { g } } \rangle \geq \| { \widehat { g } } \| _ { 2 } ( \| { \widehat { g } } \| _ { 2 } - b _ { \theta } ) > 0 , \qquad \cos ( g , { \widehat { g } } ) \geq { \sqrt { 1 - { \frac { b _ { \theta } ^ { 2 } } { \| { \widehat { g } } \| _ { 2 } ^ { 2 } } } } } .
$$

Here cos $( g , \widehat { g } ) = \langle g , \widehat { g } \rangle / ( \| g \| _ { 2 } \| \widehat { g } \| _ { 2 } )$ . The angular bound is attained in dimension at least two within the gradient-error ball. The following subsections supply the domination, coupling, and extremal calculations; this is geometric sharpness, not MDP minimax optimality.

Theorem 6 (Local improvement certificate). Suppose $\| { \widetilde { g } } - { \widehat { g } } \| _ { 2 } \leq \xi$ and J is L-smooth along a feasible step $\theta ^ { + } = \theta + \eta \widetilde { g }$ , with $L , \eta > 0 .$ . Set $b = b _ { \theta } + \xi$ and $h = \| { \widetilde { g } } \| _ { 2 } .$ . Then

$$
J ( \pi _ { \theta ^ { + } } ) - J ( \pi _ { \theta } ) \geq \eta h [ ( 1 - L \eta / 2 ) h - b ] .
$$

$H h > b ,$ thefeasible choice $\eta = ( h - b ) / ( L h )$ guarantees $( h - b ) ^ { 2 } / ( 2 L )$ improvement. $I f h \leq b ,$ no direction has positive worst-casefirst-order gain over the gradient ball.

Proof of Theorem 6. By the triangle inequality, $\| g - \widetilde g \| _ { 2 } \le b$ . Smoothness therefore gives

$$
\begin{array} { r } { J ( \pi _ { \theta ^ { + } } ) - J ( \pi _ { \theta } ) \geq \eta \langle g , \widetilde g \rangle - \frac { L } { 2 } \eta ^ { 2 } h ^ { 2 } \geq \eta h ( h - b ) - \frac { L } { 2 } \eta ^ { 2 } h ^ { 2 } . } \end{array}
$$

For a feasible displacement $v ,$ the exact supporting lower model is

$$
\operatorname* { i n f } _ { \| \boldsymbol { q } - \widetilde { \boldsymbol { g } } \| _ { 2 } \le b } \left\{ \langle \boldsymbol { q } , \boldsymbol { v } \rangle - \frac { L } { 2 } \| \boldsymbol { v } \| _ { 2 } ^ { 2 } \right\} = \langle \widetilde { g } , \boldsymbol { v } \rangle - b \| \boldsymbol { v } \| _ { 2 } - \frac { L } { 2 } \| \boldsymbol { v } \| _ { 2 } ^ { 2 } .
$$

For $v \neq 0 ,$ , equality holds at $q = \widetilde { g } - b v / \| v \| _ { 2 }$ . At length $t = \| v \| _ { 2 }$ , the maximum is $( h - b ) t - L t ^ { 2 } / 2$ attained by alignment with $\widetilde g$ when $h > 0$ . Its maximizing length is $[ h - b ] _ { + } / \dot { L }$ , where $[ x ] _ { + } =$ max{x, 0}. If $h \leq b ,$ zero belongs to the gradient ball. Quadratic objectives attain the lower model; sharpness is relative to this local uncertainty class. Appendix D.8 proves sharpness and the general-displacement certificate; Appendix D.9 states the requirements for practical optimizers.

We record the regularity and prefix constants used in this section:

$$
\psi _ { \boldsymbol \theta } ( s , a ) = \nabla _ { \boldsymbol \theta } \log \pi _ { \boldsymbol \theta } ( a \mid s ) , \qquad \| \psi _ { \boldsymbol \theta } ( s , a ) \| _ { 2 } \le G .
$$

$$
d _ { t } = \mathbb { E } _ { \mu _ { t } ^ { \pi _ { \theta } } } \operatorname { T V } ( P , { \widehat { P } } ) , \qquad e = \sum _ { t = 0 } ^ { x = - \infty } d _ { t } ,
$$

$$
D _ { t } = \operatorname* { m i n } \left\{ 1 , \sum _ { j < t } d _ { j } \right\} , C _ { H } = \frac { G R _ { b } H ( H + 1 ) } { 2 } ,
$$

With $\begin{array} { r } { S _ { t } = \sum _ { j = 0 } ^ { t } \psi _ { \theta } ( s _ { j } , a _ { j } ) } \end{array}$ , the common trajectory representation is given by the common trajectory representation above.

## D.2 DIFFERENTIATION UNDER THE TRAJECTORY INTEGRAL

For each state, use a common sigma-finite action reference measure and a strictly positive policy density on parameter-independent support. Fix θ and a compact ball around it contained in Θ, with the bound the uniform score condition above throughout that ball. The mean-value theorem gives

$$
{ \frac { \pi _ { \theta + v } ( a \mid s ) } { \pi _ { \theta } ( a \mid s ) } } \leq \exp ( G \| v \| _ { 2 } ) .
$$

Let $Q _ { \theta }$ be the trajectory law in one fixed world and $Z _ { v } = \mathrm { d } Q _ { \theta + v } / \mathrm { d } Q _ { \theta }$ . For $\| v \| _ { 2 } \leq a$ within the chosen ball,

$$
Z _ { v } = \prod _ { j = 0 } ^ { H - 1 } { \frac { \pi _ { \theta + v } ( a _ { j } \mid s _ { j } ) } { \pi _ { \theta } ( a _ { j } \mid s _ { j } ) } } \leq e ^ { H G a } ,
$$

$$
\| \nabla _ { v } Z _ { v } \| _ { 2 } = \left\| Z _ { v } \sum _ { j = 0 } ^ { H - 1 } \psi _ { \theta + v } ( s _ { j } , a _ { j } ) \right\| _ { 2 } \leq H G e ^ { H G a } .
$$

The return is bounded by $H R _ { b }$ . Dominated convergence permits differentiation under $Q _ { \theta }$ . This argument is applied separately in each world and requires no likelihood ratio between $P$ and $\widehat { P }$

Differentiating policy normalization gives $\begin{array} { r } { \int { \psi _ { \theta } ( s , a ) \pi _ { \theta } ( \mathrm { d } a \ | \ s ) } = 0 } \end{array}$ . Let $r _ { t } = r ( s _ { t } , a _ { t } )$ . The likelihood-ratio identity initially reads

$$
\nabla J = \mathbb { E } _ { P } \left[ \left( \sum _ { j = 0 } ^ { H - 1 } \psi _ { \boldsymbol { \theta } } ( s _ { j } , a _ { j } ) \right) \left( \sum _ { t = 0 } ^ { H - 1 } r _ { t } \right) \right] .
$$

Let $\mathcal { H } _ { j } = \sigma ( s _ { 0 } , a _ { 0 } , \ldots , a _ { j - 1 } , s _ { j } )$ be the history before action $a _ { j }$ . For $j > t , r _ { t }$ is $\mathcal { H } _ { j }$ -measurable, and

$$
\begin{array} { r } { \mathbb { E } [ \psi _ { \theta } ( s _ { j } , a _ { j } ) r _ { t } ] = \mathbb { E } [ r _ { t } \mathbb { E } [ \psi _ { \theta } ( s _ { j } , a _ { j } ) \mid \mathcal { H } _ { j } ] ] = 0 . } \end{array}
$$

Removing these terms gives

$$
\nabla J = \mathbb { E } _ { P } F _ { \theta } , \qquad F _ { \theta } = \sum _ { t = 0 } ^ { H - 1 } S _ { t } r _ { t } , \qquad S _ { t } = \sum _ { j = 0 } ^ { t } \psi _ { \theta } ( s _ { j } , a _ { j } ) .
$$

Since $\| S _ { t } \| _ { 2 } \leq ( t + 1 ) G , \| F _ { \theta } \| _ { 2 } \leq C _ { H }$ . The same functional represents the imagined gradient when rewards are shared.

## D.3 SEQUENTIAL COUPLING WITHOUT A SUPPORT ASSUMPTION

For standard Borel kernels a measurable maximal coupling can be constructed from their common part. At a state–action pair let $ { \boldsymbol \nu } = P +  { \widehat { P } }$ , write their densities relative to ν as $p , { \widehat { p } } ,$ and use min $( p , \widehat { p } ) \nu$ as the common subprobability. Its mass is $1 - \operatorname { T V } ( P , \widehat { P } )$ . Sample identically from this part and couple the residual parts arbitrarily. This gives both required marginals and disagreement probability equal to the local total variation.

Run that coupling only while the histories agree, using identical policy draws on common histories. Let $E _ { j }$ be first disagreement at transition $j .$ . If $A _ { j }$ is agreement through $( s _ { j } , a _ { j } )$ , then

$$
\mathbb { P } ( E _ { j } ) = \mathbb { E } \Big [ \mathbb { I } \{ A _ { j } \} \mathrm { T V } \{ P ( \cdot \mid s _ { j } , a _ { j } ) , \widehat { P } ( \cdot \mid s _ { j } , a _ { j } ) \} \Big ] \le d _ { j } .
$$

Let $Q _ { t } , \widehat { Q } _ { t }$ be the prefix laws through $( s _ { t } , a _ { t } )$ . The first-disagreement events are disjoint, so

$$
\mathrm { T V } ( Q _ { t } , \widehat { Q } _ { t } ) \leq \mathbb { P } \left( \bigcup _ { j < t } E _ { j } \right) = \sum _ { j < t } \mathbb { P } ( E _ { j } ) \leq \sum _ { j < t } d _ { j } .
$$

Combining with $\mathrm { T V } ( Q _ { t } , \widehat { Q } _ { t } ) \leq 1$ gives $D _ { t }$ .

For any vector-valued $F$ with $\| F \| _ { 2 } \leq C$

$$
\| \mathbb { E } _ { P } F - \mathbb { E } _ { Q } F \| _ { 2 } = \operatorname* { s u p } _ { \| v \| _ { 2 } = 1 } | \mathbb { E } _ { P } \langle v , F \rangle - \mathbb { E } _ { Q } \langle v , F \rangle | \le 2 C \operatorname { T V } ( P , Q ) .
$$

Using $\| S _ { t } r _ { t } \| _ { \infty } \leq ( t + 1 ) G R _ { b }$

$$
\begin{array} { l } { \displaystyle \| g - \widehat g \| _ { 2 } \leq \sum _ { t = 0 } ^ { H - 1 } \left\| \int S _ { t } r _ { t } \mathrm { d } ( Q _ { t } - \widehat Q _ { t } ) \right\| _ { 2 } , } \\ { \displaystyle \leq 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) D _ { t } \leq 2 C _ { H } \operatorname* { m i n } \{ 1 , e \} . } \end{array}
$$

With uniform local error ϵ,

$$
b _ { \theta } \leq 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) \operatorname* { m i n } \{ 1 , t \epsilon \} .
$$

For small $H \epsilon ,$ this is $2 G R _ { b } { \epsilon } H ( H - 1 ) ( H + 1 ) / 3$ . We do not claim this cubic horizon dependence is minimax optimal over policy-gradient MDPs.

## D.4 PREDICTION ERROR AND COMPARISON-SPECIFIC SENSITIVITY

Proof of Corollary 2. Fix $\epsilon \in ( 0 , 1 / 2 ]$ and a two-step task with initial actions $a _ { 0 } , a _ { 1 }$ , terminal states $g , b ,$ zero initial reward, and terminal reward $c \in ( 0 , R _ { b } ]$ at $g .$ . In the true world let

$$
P ( g \mid a _ { 0 } ) = { \frac { 1 } { 2 } } , \qquad P ( g \mid a _ { 1 } ) = { \frac { 1 } { 2 } } - { \frac { \epsilon } { 2 } } ,
$$

whereas the learned world model satisfies

$$
\widehat { P } ( g \mid a _ { 0 } ) = \frac { 1 } { 2 } , \qquad \widehat { P } ( g \mid a _ { 1 } ) = \frac { 1 } { 2 } + \frac { \epsilon } { 2 } .
$$

The remaining probability is assigned to $b ,$ and the two worlds agree elsewhere. Hence

$$
\operatorname* { s u p } _ { a } \mathrm { T V } \Big ( P ( \cdot \mid a ) , \widehat { P } ( \cdot \mid a ) \Big ) = \epsilon ,
$$

but the learned world model assigns comparison $c \epsilon / 2$ to replacing $a _ { 0 }$ by $a _ { 1 }$ , while the true comparison is $- c \epsilon / 2$ . Thus arbitrarily small uniform transition error reverses the action ranking.

Parameterize the two-action construction of Corollary 2 by $\pi _ { \theta } ( a _ { 1 } ) = \sigma ( \theta ) = ( 1 + \exp ( - \theta ) ) ^ { - 1 }$ Then

$$
J ( \theta ) = \frac { c } { 2 } - \frac { c \epsilon } { 2 } \sigma ( \theta ) , \qquad \widehat { J } ( \theta ) = \frac { c } { 2 } + \frac { c \epsilon } { 2 } \sigma ( \theta ) .
$$

The same construction has a differential consequence. The learned world model ranks $a _ { 1 }$ above $a _ { 0 }$ while the true world ranks them in the opposite order; hence world-model-greedy planning is wrong without any parameter update. Increasing the logistic probability of $a _ { 1 }$ strictly decreases the true return, so optimizing the predictive surrogate follows the wrong direction.

For the converse, let the next state be $( y , v ) \in \{ 0 , 1 \} ^ { 2 }$ and the terminal reward be $c y .$ , with $c \in ( 0 , R _ { b } ]$ At the initial state, actions $a _ { 0 } , a _ { 1 }$ produce $y = 1$ with probabilities $1 / 4 , 3 / 4$ in both worlds. The true world sets $v = 0$ and the learned world sets $v = 1$ . Terminal dynamics and rewards agree. The two initial next-state laws have disjoint supports, so their total variation is one. Nevertheless every policy has the same value in both worlds: only its initial action affects reward, and the terminal action is immaterial. With logistic initial action probability,

$$
J ( \theta ) = \widehat { J } ( \theta ) = c [ 1 / 4 + \sigma ( \theta ) / 2 ] .
$$

Values, action ordering, and policy gradients are all exact despite maximal transition discrepancy.   
This is why a total-variation-based sufficient gate need not characterize every valid use.

Qualification is comparison-specific. Consider three actions with terminal success probabilities

$$
p _ { 0 } = 1 / 4 , \qquad p _ { 1 } \in [ 1 / 4 - \epsilon / 2 , 1 / 4 + \epsilon / 2 ] , \qquad p _ { 2 } = 3 / 4 ,
$$

where $0 < \epsilon \leq 1 / 4$ . Supply the upper-endpoint model and take the lower endpoint as the true world. Replacing $a _ { 0 }$ by $a _ { 2 }$ has contrast $c / 2$ in every plausible world; replacing $a _ { 0 }$ by $a _ { 1 }$ has contrast $- c \epsilon / 2$ in truth and $c \epsilon / 2$ in the model. Thus a certificate for the first comparison does not certify the second, even within the same task and with the same predictive model.

## D.5 WHY VALUE EQUALITY AND JOINT PARAMETER UPDATES ARE INSUFFICIENT

In a two-step task, initial actions $a _ { 1 } , a _ { 0 } ,$ in this order, lead to terminal reward c with true success probabilities 1, 0 and learned success probabilities 0, 1. For the logistic policy $\pi _ { \boldsymbol { \theta } } ( a _ { 1 } ) = \sigma ( \boldsymbol { \theta } )$

$$
J ( \theta ) = c \sigma ( \theta ) , \qquad \widehat { J } ( \theta ) = c [ 1 - \sigma ( \theta ) ] .
$$

At $\theta = 0$ the values both equal $c / 2$ , but the gradients are $c / 4$ and $- c / 4$ . Policy scores are bounded by one. Equality of scalar values at a point therefore provides no gradient certificate, even with bounded rewards and regular policies.

Freezing the world is also essential. Consider a one-action bandit with true reward $1 / 2$ and model reward $\widehat { r } _ { \theta } = 1 / 2 + \theta$ for $| \theta | < 1 / 4$ . At $\theta = 0$ model error is zero and the policy cannot change the true return. Nevertheless the total derivative of the model return is one. This derivative changes the model, not the policy. The main gradient theorem controls derivatives through a fixed predictive world; it does not certify unrestricted shared-parameter co-training. Shared-parameter methods require separating these derivative paths or proving an additional bound.

## D.6 REWARD-MODEL ERROR

Let $\widehat { r }$ also be fixed with respect to $\theta ,$ , and bounded in $[ 0 , R _ { b } ]$ . The learned gradient now uses $\begin{array} { r } { \widehat { F } _ { \theta } = \sum _ { t } S _ { t } \widehat { r } ( s _ { t } , a _ { t } ) } \end{array}$ . Adding and subtracting $\mathbb { E } _ { \hat { P } } F _ { \theta }$ gives

$$
\| g - \widehat g \| _ { 2 } \le 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) D _ { t } + G \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) \mathbb { E } _ { \widehat { \mu } _ { t } ^ { \pi _ { \theta } } } | r - \widehat r | .
$$

Indeed, the first difference uses the common true-reward functional and prefix coupling; the second is bounded pointwise by $\begin{array} { r } { \sum _ { t } ( t + 1 ) G | r - \widehat { r } | } \end{array}$ and integrated under the learned occupancy. A uniform reward error $\epsilon _ { r }$ yields the additional radius $G \epsilon _ { r } H ( \bar { H ^ { } } + 1 ) / 2$ . Every gate must include this additional radius when rewards are learned. A sampled reward discrepancy is not a uniform bound unless separately certified.

## D.7 SHARP ANGULAR GEOMETRY

We prove the sharp angular consequence the angular inequality above for Theorem 5. Here cos $\dot { ( g , g ) } = \langle g , \widehat { g } \rangle / \bar { ( } \| g \| _ { 2 } \| \widehat { g } \| _ { 2 } )$

Let v $\neq 0 , h = \| v \| _ { 2 }$ , and suppose $\| q - v \| _ { 2 } \leq b < h$ . Put $u = v / h$ and decompose $\scriptstyle q = x u + y$ with $y \perp u .$ . The constraint is $( x - h ) ^ { 2 } + \| y \| _ { 2 } ^ { 2 } \leq b ^ { 2 }$ , so $x > 0$ . The largest possible squared tangent of the angle is

$$
\operatorname* { m a x } _ { x \in [ h - b , h + b ] } { \frac { b ^ { 2 } - ( x - h ) ^ { 2 } } { x ^ { 2 } } } .
$$

Writing $F ( x ) = [ b ^ { 2 } - ( x - h ) ^ { 2 } ] / x ^ { 2 }$ , we have

$$
F ^ { \prime } ( x ) = { \frac { 2 ( h ^ { 2 } - b ^ { 2 } - h x ) } { x ^ { 3 } } } , \qquad x _ { * } = ( h ^ { 2 } - b ^ { 2 } ) / h , \qquad F ( x _ { * } ) = { \frac { b ^ { 2 } } { h ^ { 2 } - b ^ { 2 } } } .
$$

Since $x _ { * }$ lies in $[ h - b , h + b ]$ and the derivative changes from positive to negative there,

$$
{ \frac { \langle q , v \rangle } { \| q \| _ { 2 } \| v \| _ { 2 } } } \geq { \frac { \sqrt { h ^ { 2 } - b ^ { 2 } } } { h } } .
$$

For dimension at least two and $b > 0$ , choose a perpendicular y attaining the boundary to obtain equality. If $b = 0$ or the dimension is one, the cosine is one; the displayed lower bound still holds. If $b \geq h$ , the uncertainty ball includes zero, and for $b > h$ it includes vectors pointing against v. This proves the exact robust threshold for an acute-angle guarantee.

## D.8 OPTIMAL ROBUST DISPLACEMENT AND SHARPNESS

For every displacement v we prove the exact robust-support formula the robust-support identity above.

Suppose a differentiable objective has unknown gradient in the ball $\{ q : \| q - \widetilde { g } \| _ { 2 } \leq b \}$ and satisfies the local smooth lower model $J ( \theta + v ) - J ( \theta ) \stackrel {  } { \geq } \langle q , v \rangle - L \| v \| _ { 2 } ^ { 2 } / 2$ . For v $\neq 0 ,$ the adverse gradient is $q = \widetilde { g } - b v / \| v \| _ { 2 }$ . Hence the robust-support identity above holds exactly; for $v = 0$ both sides are zero.

Put $h = \| \widetilde { g } \| _ { 2 }$ and let $\Phi ( v )$ denote the right-hand side of the robust-support identity above. Cauchy– Schwarz and one-variable maximization give

$$
\begin{array} { r l } & { \underset { | | v | | _ { 2 } = t } { \operatorname* { s u p } } \Phi ( v ) = ( h - b ) t - \frac { L } { 2 } t ^ { 2 } , } \\ & { } \\ & { \underset { v \in \mathbb { R } ^ { d } } { \operatorname* { s u p } } \Phi ( v ) = \underset { t \geq 0 } { \operatorname* { s u p } } \{ ( h - b ) t - \frac { L } { 2 } t ^ { 2 } \} = \frac { [ h - b ] _ { + } ^ { 2 } } { 2 L } . } \end{array}
$$

For $h > b ,$ , the maximizer is $v _ { * } = ( h - b ) \widetilde { g } / ( L h )$ ; for $h \leq b ,$ it is $v _ { * } = 0$ . The unconstrained maximum is attainable as a certified step only if the segment from θ to $\theta + v _ { * }$ is feasible and lies in the smoothness region.

For a specified v, the quadratic

$$
f ( \theta + w ) = f ( \theta ) + \langle q , w \rangle - \frac { L } { 2 } \| w \| _ { 2 } ^ { 2 }
$$

attains the lower model. These local quadratics establish sharpness given gradient-ball and smoothness information. They are not asserted to be globally bounded MDP returns; a tighter guarantee exploiting further MDP structure is not ruled out.

A sufficient smoothness bound is available when policies are twice continuously differentiable and $\| \nabla _ { \theta } ^ { 2 } \log \pi _ { \theta } ( { a \mathbin { \left/ { \vphantom { a \theta } } \right. \kern - delimiterspace } { s } } ) \| _ { 2 } \leq K$ uniformly on a convex region. Differentiating the prefix expression for expected reward yields

$$
\nabla _ { \theta } ^ { 2 } J = \mathbb { E } _ { P } \sum _ { t = 0 } ^ { H - 1 } r _ { t } \left[ S _ { t } S _ { t } ^ { \top } + \sum _ { j = 0 } ^ { t } \nabla _ { \theta } ^ { 2 } \log \pi _ { \theta } ( a _ { j } \mid s _ { j } ) \right] .
$$

The domination argument used for the first derivative extends using these bounds. Therefore

$$
L = R _ { b } \left[ G ^ { 2 } \frac { H ( H + 1 ) ( 2 H + 1 ) } { 6 } + K \frac { H ( H + 1 ) } { 2 } \right]
$$

is sufficient. A smaller valid local L improves the gate, but a numerical curvature estimate alone is not a guaranteed upper bound.

## D.9 ACTUAL OPTIMIZERS AND LATENT INTERFACES

The comparison certificate applies to a realized output of any predictive search procedure. The differential condition is useful when that procedure optimizes a parameterized behavior. It validates the proposed displacement; it does not require a synthetic-data training pipeline.

For differentiable policies, a proposed displacement need not be parallel to a policy-gradient estimate. $\| { \boldsymbol { \mathfrak { f } } } \| { \boldsymbol { g } } - { \boldsymbol { \widetilde { g } } } \| _ { 2 } \leq b$ and the proposed segment is L-smooth, then for every feasible $v ,$

$$
J ( \theta + v ) - J ( \theta ) \geq \langle \widetilde { g } , v \rangle - b \| v \| _ { 2 } - \frac { L } { 2 } \| v \| _ { 2 } ^ { 2 } .
$$

This follows by inserting the gradient-ball support function into the smoothness inequality. It can certify a preconditioned, clipped, or otherwise proposed optimizer displacement without identifying it with the exact gradient. The estimate must target the score-function gradient of the frozen imagined world. If a practical estimate has an additional bias bounded by ζ, replace b by $b _ { \theta } + \xi + \zeta$ . Merely naming a critic or using automatic differentiation does not establish such a bound.

A common history interface need not require reconstructing every observation. The following bridge states precisely what is sufficient for a fixed latent policy interface.

Proposition 5 (History-to-latent reliability bridge). Let h be a true observable history, $s = f ( h )$ a measurable, parameter-independent representation, and let the policy use only s. Start the latent model at the pushforward of the true initial history law. Write $\bar { K } ( \cdot \mid h , a )$ for the true conditional law ofthe next representation and $\widehat { P } ( \cdot \mid f ( h ) , a )$ for its learned prediction. Suppose, uniformly in admissible histories and actions,

$$
\begin{array} { r } { | r ( h , a ) - \widehat { r } ( f ( h ) , a ) | \leq \epsilon _ { r } , \qquad \mathrm { T V } \{ K ( \cdot \mid h , a ) , \widehat { P } ( \cdot \mid f ( h ) , a ) \} \leq \epsilon _ { p } . } \end{array}
$$

Then every such policy satisfies

$$
| J - \widehat { J } | \leq H \epsilon _ { r } + \frac { H ( H - 1 ) } { 2 } R _ { b } \epsilon _ { p } .
$$

Under the samefixed-interface score regularity as Theorem 5, its gradient discrepancy is at most

$$
\| g - \widehat g \| _ { 2 } \le 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) \operatorname* { m i n } \{ 1 , t \epsilon _ { p } \} + \frac { G H ( H + 1 ) } { 2 } \epsilon _ { r } .
$$

The true representation process need not be Markov.

Proof. Lift the learned value to histories as $\widehat { V } _ { t } ( f ( h ) )$ . Subtract its learned Bellman recursion from the true history recursion and add the true conditional expectation of $\widehat { V } _ { t + 1 } ( f ( h ^ { \prime } ) )$ . The residual is the reward difference plus $( K - \widehat { P } ) \widehat { V } _ { t + 1 }$ , bounded by $\epsilon _ { r } + ( H - t - 1 ) R _ { b } \epsilon _ { p }$ . The remaining true conditional difference telescopes exactly as in Theorem 2. Summing proves the latent value bound above.

For gradients, keep the complete true history in the construction while coupling the latent trajectories. On agreement of latent prefixes, the two action laws coincide. Couple the true next representation, conditional on its full history, with the learned latent transition by maximal coupling; disagreement probability is at most $\epsilon _ { p } . \mathrm { A }$ regular conditional distribution of the next true history given its representation preserves the full true marginal. Standard Borel assumptions ensure these conditional kernels exist. Once prefixes separate, continue with any marginal-preserving coupling. The latent prefix discrepancy through time t is at most min $\{ 1 , t \epsilon _ { p } \}$

Use the learned reward functional in both worlds. The true gradient’s reward-replacement error is at most $G \epsilon _ { r } \textstyle \sum _ { t } ( t + 1 )$ . The remaining common functional depends only on latent prefixes; the score is $\nabla \log \overline { { \pi } } _ { \boldsymbol { \theta } } ( a \mid f ( h ) )$ in the true process and the same function of $( s , a )$ in the learned process. Applying the prefix argument proves the latent gradient bound above. Parameter independence of f and the supplied world is required for this common functional.

The premise includes representation aliasing: histories with the same representation must all have predictions close enough to the supplied kernel. Passive reconstruction loss does not establish it.

If the representation admits an exact Markov kernel, finite-state row verification can be applied to that kernel; otherwise a latent-row average needs an additional uniform aliasing bound. Encoder updates or shared-parameter model updates require rechecking the interface and derivative premises. These distinctions allow the theory to be used with learned representations without assuming their sufficiency by definition.

## D.10 DIMENSION-FREE IMAGINED-GRADIENT SAMPLING

A sampling radius for the differential certificate is

$$
\xi = \frac { C _ { H } } { \sqrt { N } } \left( 1 + \sqrt { 2 \log ( 1 / \delta ) } \right)
$$

where N is the number of independent imagined trajectories and $\delta \in ( 0 , 1 )$

Proposition 6 (Bounded-vector sampling). Condition on a fixed policy and learned world. Let $Y _ { 1 } , \dots , Y _ { N }$ be independent samples of $\widehat { F } _ { \theta }$ with $\| Y _ { i } \| _ { 2 } ~ \le ~ C _ { H }$ , and let $\widetilde g \ = \ N ^ { - 1 } \sum _ { i } Y _ { i } .$ . Then $\| { \widetilde { g } } - { \widehat { g } } \| _ { 2 } \leq \xi$ with probability at least $1 - \delta , f o r \xi$ in the stated sampling radius. The radius may be truncated at $2 C _ { H }$

Proof. Independence and centering cancel cross terms:

$$
\mathbb { E } \| \widetilde { g } - \widehat { g } \| _ { 2 } ^ { 2 } = \frac { 1 } { N ^ { 2 } } \sum _ { i } \mathbb { E } \| Y _ { i } - \widehat { g } \| _ { 2 } ^ { 2 } \leq \frac { C _ { H } ^ { 2 } } { N } .
$$

Jensen gives $\mathbb { E } \Vert \widetilde { g } - \widehat { g } \Vert _ { 2 } \leq C _ { H } / \sqrt { N }$ . Replacing one sample changes this norm by at most $2 C _ { H } / N$ The bounded-differences inequality therefore bounds the probability of exceeding its expectation by t by $\exp [ - N t ^ { 2 } / ( 2 C _ { H } ^ { 2 } ) ]$ ]. Take $t = C _ { H } \sqrt { 2 \log ( 1 / \delta ) / N }$ . Finally $\| \widetilde { g } \| _ { 2 } , \| \widehat { g } \| _ { 2 } \le C _ { H }$ , giving the deterministic truncation.

The bound has no parameter-dimension factor because the assumed norm bound already controls the entire vector. It applies to independent trajectory estimators of the stated gradient, not automatically to replay-correlated minibatches, bootstrapped actor losses, or biased value-gradient estimators. An additional estimator-bias radius must be added for those alternatives.

## E STATISTICAL CERTIFICATION AND ADAPTIVE VERIFICATION

## E.1 CALIBRATING DECISION-RELEVANT PREDICTION ERROR

Let $x = ( z , { \widehat { \mathcal { M } } } , \pi _ { 0 } , \pi _ { 1 } )$ be a complete decision request, with $e ( \boldsymbol { x } ) = | \widehat { \Gamma } ( \boldsymbol { x } ) - \Gamma _ { \mathcal { M } } ( \boldsymbol { x } ) |$ . An independently fitted score u predicts this nonnegative error. Conditional on that fit, assume exchangeability of the calibration pairs $( x _ { j } , e _ { j } )$ and one future pair. For $\alpha \in ( 0 , 1 )$ put

$$
q _ { j } = e _ { j } - u ( x _ { j } ) , \quad k = \lceil ( n + 1 ) ( 1 - \alpha ) \rceil , \quad U _ { \alpha } ( x ) = \lceil u ( x ) + q _ { ( k ) } \rceil _ { + } .
$$

Set $q _ { ( k ) } = + \infty$ if $k > n$ . The request includes the actual model, reference, candidate selection procedure and continuation; changing their distribution is a change of the calibration population.

Proposition 7 (Calibrated contrast and false-promotion risk). Under the preceding exchangeability assumption, $\mathbb { P } \{ e ( x ) > U _ { \alpha } ( x ) \} \le \alpha$ . Suppose $\mathbb { P } \{ \widehat { \Gamma } ( x ) < S ( x ) - \xi ( x ) \} \leq \delta$ . Define

$$
T ( x ) = S ( x ) - U _ { \alpha } ( x ) - \xi ( x ) .
$$

Then

$$
\mathbb { P } \{ T ( x ) > 0 , \ : \Gamma _ { \mathcal { M } } ( x ) \le 0 \} \le \alpha + \delta .
$$

Proof. After independent randomized tie-breaking, exchangeability makes the future residual’s rank K uniform on $\{ 1 , \ldots , n + 1 \}$ . Therefore

$$
\begin{array} { l } { \displaystyle \mathbb { P } \{ q > q _ { ( k ) } \} \le \mathbb { P } \{ K > k \} } \\ { = \displaystyle \sum _ { j = k + 1 } ^ { n + 1 } \mathbb { P } \{ K = j \} = \frac { n + 1 - k } { n + 1 } \le \alpha . } \end{array}
$$

Since $q = e - u ( x )$ and $U _ { \alpha } = [ u ( x ) + q _ { ( k ) } ] _ { + }$ , the event $e > U _ { \alpha }$ is contained in $q > q _ { ( k ) }$ . On its complement and on $\widehat { \Gamma } \geq S - \xi$

$$
\begin{array} { r l } & { \Gamma _ { \mathcal { M } } = \widehat \Gamma + ( \Gamma _ { \mathcal { M } } - \widehat \Gamma ) } \\ & { \qquad \geq \widehat \Gamma - | \Gamma _ { \mathcal { M } } - \widehat \Gamma | } \\ & { \qquad \geq S - \xi - U _ { \alpha } = T . } \end{array}
$$

Hence

$$
\begin{array} { r l } & { \{ T > 0 , \Gamma _ { \cal M } \leq 0 \} \subseteq \{ e > U _ { \alpha } \} \cup \{ \widehat { \Gamma } < S - \xi \} , } \\ & { \mathbb { P } \{ T > 0 , \Gamma _ { \cal M } \leq 0 \} \leq \mathbb { P } \{ e > U _ { \alpha } \} + \mathbb { P } \{ \widehat { \Gamma } < S - \xi \} } \\ & { \qquad \leq \alpha + \delta . } \end{array}
$$

This is split conformal calibration applied to a decision-relevant error target (Angelopoulos & Bates, 2023). It does not identify the true error from a single transition. A valid upper label such as $\varepsilon _ { \mathrm { I } } ( \pi _ { 0 } ) + \varepsilon _ { \mathrm { I } } ( \pi _ { 1 } )$ may replace e when exact contrast errors are unavailable, subject to the labeling conditions below.

Calibration versus raw uncertainty scores. The certificate concerns a calibrated upper envelope of the decision-relevant error $e ( \boldsymbol { x } ) = | \widehat { \Gamma } ( \boldsymbol { x } ) - \Gamma _ { \mathcal { M } } ( \boldsymbol { x } ) |$ . A confidence, disagreement, entropy, or critic score is therefore not itself an error certificate: it becomes usable in the gate only after an argument establishes the required upper-error guarantee for the request population under consideration.

## E.2 SPLIT CONFORMAL COVERAGE AND THE EXACT PROMOTION EVENT

Condition on the training data used to fit u. Let $q _ { 1 } , \ldots , q _ { n } , q$ be the exchangeable calibration and future residuals, and let K denote the rank of q after independent randomized tie-breaking. For $k \leq n ,$

$$
\mathbb { P } ( K = j ) = ( n + 1 ) ^ { - 1 } , \qquad j = 1 , \dots , n + 1 ,
$$

$$
\mathbb { P } \{ q > q _ { ( k ) } \} \le \mathbb { P } ( K > k ) = \frac { n + 1 - k } { n + 1 } \le \alpha .
$$

Since $q = e - u ( x )$ and $U _ { \alpha } ( x ) = \operatorname* { m a x } \{ 0 , u ( x ) + q _ { ( k ) } \}$

$$
\{ e > U _ { \alpha } ( x ) \} \subseteq \{ q > q _ { ( k ) } \} , \qquad { \mathbb { P } } \{ e > U _ { \alpha } ( x ) \} \leq \alpha .
$$

For $k = n + 1$ , set $q _ { ( k ) } = + \infty ;$ coverage is then immediate.

Define the events $E = \{ e \leq U _ { \alpha } \}$ and $F = \{ \widehat { \Gamma } \geq S - \xi \}$ . On $E \cap F$ , positive T implies strictly positive true gain. Therefore

$$
\{ T > 0 , \ \Gamma _ { \mathcal { M } } ( x ) \leq 0 \} \subseteq E ^ { c } \cup F ^ { c } .
$$

Consequently, without independence of $E , F ,$

$$
\begin{array} { r } { \mathbb { P } \{ T > 0 , ~ \Gamma _ { \mathcal { M } } ( x ) \leq 0 \} \leq \mathbb { P } ( E ^ { c } ) + \mathbb { P } ( F ^ { c } ) \leq \alpha + \delta . } \end{array}
$$

In contrast, if $p = \mathbb { P } ( T > 0 ) > 0$ , the argument gives at most

$$
\mathbb { P } \{ \Gamma _ { \mathcal { M } } ( x ) \leq 0 \mid T > 0 \} \leq \operatorname* { m i n } \{ 1 , ( \alpha + \delta ) / p \} ,
$$

not $\alpha + \delta .$ . A generic marginal certificate can fail exclusively on a selected minority: let X be Bernoulli with mean α, let $e ( X ) \ = \ X$ , and set $U ( X ) = 0 .$ Marginal coverage is $1 - \alpha$ , yet conditional miscoverage given $X = 1$ is one. This demonstrates why marginal validity alone cannot justify selection-conditional validity; it is not a claim that this particular U was produced by split conformal prediction.

For multiple future snapshots chosen before observing calibration outcomes, one may compute each marginal certificate at level $\alpha / m$ for a fixed pool of size m. A union bound gives simultaneous coverage at least $1 - \alpha$ over the pool, permitting arbitrary subsequent selection within it. Each pair must still have the required marginal exchangeability; the pool cannot be manufactured adaptively from the same calibration residuals without further analysis.

## E.3 NOISY LABELS AND FINITE STRATA

If exact errors are unavailable, suppose an identically applied labeling procedure produces exchangeable upper labels $\overline { { e } } _ { i }$ , including a hypothetical future e, and $\mathbb { P } ( e > \overline { { e } } ) \le \beta$ . Calibrate the residuals $\overline { { e } } _ { i } - u ( x _ { i } )$ instead. Conformal coverage of e and a union bound then yield

$$
\mathbb { P } \{ e > U _ { \alpha } ( x ) \} \le \alpha + \beta .
$$

This statement requires upper labels and their separate validity bound, not merely unbiased estimates. A pixel discrepancy or mean state error may instead define a different target; conformal coverage of that target does not bound e without a proven bridge.

For a prespecified finite partition $c ( x ) \in \{ 1 , \ldots , m \}$ , calibrate separately within each stratum. Conditional on a future stratum $h ,$ the fitted score, and its calibration count $n _ { h }$ , assume the corresponding residuals are exchangeable. The same rank argument with $k _ { h } = \lceil ( n _ { h } + 1 ) ( 1 - \alpha ) \rceil$ gives

$$
\mathbb { P } \{ e \leq U _ { h } ( x ) \mid c ( x ) = h \} \geq 1 - \alpha .
$$

The partition is fixed independently of calibration errors. Conditioning on arbitrary within-stratum promotion decisions is still not covered.

## E.4 FRESH FINITE-STATE VERIFICATION

This construction supplies a fully observable certificate under stronger access assumptions. Fix a finite task with $D = \vert \boldsymbol { S } \vert$ and $\dot { m } = | \mathcal { S } | | \mathcal { A } |$ state–action rows. After choosing the task, obtain n independent next-state samples per row from a generative oracle. For learned rewards, also obtain bounded reward samples in $[ 0 , \bar { R } _ { b } ]$ with the correct conditional means. Samples within each row are conditionally independent given the pre-audit history. Across-row independence is unnecessary. Write ${ \overline { { P } } } , { \overline { { r } } }$ for the empirical kernels and means and define

$$
a _ { n } = { \sqrt { \frac { D \log 2 + \log ( 4 m / \alpha ) } { 2 n } } } , \qquad c _ { n } = R _ { b } { \sqrt { \frac { \log ( 4 m / \alpha ) } { 2 n } } } .
$$

For each row set

$$
u ( s , a ) = \operatorname* { m i n } \{ 1 , \mathrm { T V } ( \overline { { P } } , \widehat { P } ) + a _ { n } \} , \qquad v ( s , a ) = \operatorname* { m i n } \{ R _ { b } , | \overline { { r } } - \widehat { r } | + c _ { n } \} .
$$

The symbols $s , a$ in row arguments denote a state and an action.

Proposition 8 (Simultaneous fresh-audit certificate). Conditional on the pre-audit history, with probability at least $1 - \alpha ,$ every row obeys $\operatorname { T V } ( P , \widehat { P } ) \leq$ u and $| r - \widehat { r } | \leq v$ . The statement holds simultaneouslyfor all bounded world-model candidates, including candidatesfitted using the audit. $I f u _ { * } = \operatorname* { m a x } _ { s , a } u ( s , a )$ and $v _ { * } = \operatorname* { m a x } _ { s , a } v ( s , a )$ , then every policy has

$$
\varepsilon _ { \mathrm { I } } ( \pi ) \leq H v _ { * } + \frac { H ( H - 1 ) } { 2 } R _ { b } u _ { * } ,
$$

and every policy satisfying the score bound has

$$
\| g - \widehat g \| _ { 2 } \leq 2 G R _ { b } \sum _ { t = 0 } ^ { H - 1 } ( t + 1 ) \operatorname* { m i n } \lbrace 1 , t u _ { * } \rbrace + \frac { G H ( H + 1 ) } { 2 } v _ { * } .
$$

Proof. For any fixed subset of the D next states, its empirical probability is the mean of Bernoulli variables. Hoeffding bounds absolute error above $a _ { n }$ by $2 \exp \bigl ( - 2 n a _ { n } ^ { 2 } \bigr )$ . Union over at most $2 ^ { D }$ subsets and m rows gives failure probability at most $\alpha / 2$ for $\mathrm { T V } ( P , { \overline { { P } } } ) \leq a _ { n }$ . A separate Hoeffding and row union bound give $| r - \overline { { r } } | \leq c _ { n }$ with failure probability at most $\alpha / 2$

On this common event, triangle inequalities give the row envelopes above. The event constrains only the true and empirical kernels; hence the inequalities hold for every supplied world model, including data-dependent ones. Clipping is valid because total variation is at most one and rewards differ by at most $R _ { b }$ . The uniform value inequality proves the audited value bound above. The reward-gradient bound and $d _ { t } \leq u _ { * }$ prove the audited gradient bound, uniformly over the admitted policies.

This verifier costs mn real row queries and does not infer total variation from prediction loss. Its strength is simultaneous validity after policy selection; its weakness is dependence on finite spaces and generative coverage. In large or continuous spaces it must be replaced by a justified structured confidence set or a direct return audit, not presented as computationally free.

## E.5 SHARED EVIDENCE UNDER ADAPTIVE PREDICTIVE DECISIONS

Proof of Theorem 4. Let E be the simultaneous event from Proposition 8. It constrains the true and empirical transition rows, and reward means when audited, independently of the candidate policies or learned world models. For every selected world model, let $u _ { i } , v _ { i }$ be the maxima of the transition and reward envelopes defined above. Then

$$
B _ { i } = 2 H v _ { i } + R _ { b } H ( H - 1 ) u _ { i } , \qquad | \Gamma _ { \mathcal { M } } ( x _ { i } ) - \widehat { \Gamma } _ { i } | \leq B _ { i } \quad \mathrm { o n } E .
$$

For known shared rewards put $v _ { i } = 0$ . The triangle inequalities hold simultaneously for models fitted on the audit as well as for independently fitted models.

Let $\mathcal { H } _ { i }$ contain the audit, prior requests and observations, the current world model, and the selected policy pair, before estimation of the current contrast. For $F _ { i } = \{ \widehat { \Gamma } _ { i } \geq S _ { i } - \xi _ { i } \}$ assume $\mathbb { P } ( F _ { i } ^ { c } \mid \mathcal { H } _ { i } ) \leq$ $\delta _ { i }$ . Write I for the random set of accepted indices. On $E \cap F _ { i } ,$ any accepted request satisfies

$$
\begin{array} { c } { { \Gamma _ { \mathcal { M } } ( x _ { i } ) \geq \widehat { \Gamma } _ { i } - B _ { i } } } \\ { { \geq S _ { i } - \xi _ { i } - B _ { i } = T _ { i } > 0 . } } \end{array}
$$

Consequently, the bad-promotion event obeys

$$
B = \bigcup _ { i \geq 1 } ( \{ i \in \mathbb { Z } \} \cap \{ \Gamma _ { { \mathcal { M } } } ( x _ { i } ) \leq 0 \} ) \subseteq E ^ { c } \cup \bigcup _ { i \geq 1 } F _ { i } ^ { c } .
$$

The conditional validity of the fresh estimate and the tower property give

$$
\begin{array} { r l } {  { \mathbb { P } ( B ) \le \mathbb { P } ( E ^ { c } ) + \mathbb { P } ( \bigcup _ { i \ge 1 } F _ { i } ^ { c } ) } } \\ & { \le \mathbb { P } ( E ^ { c } ) + \sum _ { i \ge 1 } \mathbb { P } ( F _ { i } ^ { c } ) } \\ & { = \mathbb { P } ( E ^ { c } ) + \sum _ { i \ge 1 } \mathbb { E } \big [ \mathbb { P } ( F _ { i } ^ { c } \mid \mathcal { H } _ { i } ) \big ] } \\ & { \le \alpha + \sum _ { i \ge 1 } \delta _ { i } . } \end{array}
$$

No conditional coverage of $E$ after selection is used. The argument requires the unconditional common event and conditional validity of each fresh estimation step.

For a selected pair, draw N independent paired imagined episodes and let $S _ { i }$ be the sample mean difference. Each difference lies in $[ - H \bar { R _ { b } } , H \bar { R _ { b } } ]$ , so

$$
\widehat { \Gamma } _ { i } \geq S _ { i } - \xi _ { i } , \qquad \xi _ { i } = H R _ { b } \sqrt { \frac { 2 \log ( 1 / \delta _ { i } ) } { N } }
$$

with conditional probability at least $1 - \delta _ { i }$ . Pairing within an index is allowed; pairs must be independent. If the candidate is selected from K prespecified candidates using these same samples, replace $\delta _ { i }$ by $\delta _ { i } / K$ and union the bounds. Unrestricted selection requires fresh evaluation or a justified uniform estimator. This sampling restriction is separate from reuse of the model audit.

For Theorem 3, the uniform row event bounds every conditional $Q _ { t } ^ { 0 }$ through (16). Exact world-model evaluation then gives simultaneous local certificates at all states. With sampled local evaluation, either use a uniform finite-state correction or conditional fresh estimates along the actual trajectory. The latter controls bad local decisions along that trajectory; it must not be presented as a uniform bound over all unvisited histories.

For a fixed task archive, allocate audit risk $\alpha _ { z }$ to task $z .$ Total false-promotion risk is at most $\textstyle \sum _ { z } \alpha _ { z } + \sum _ { i } \delta _ { i }$ , with each stationary task’s model risk charged once. If a task’s transition kernel drifts by at most $\beta$ in uniform TV, replace $u _ { i }$ by min $\{ 1 , u _ { i } + \bar { \beta } \}$ . Reward drift analogously increases $v _ { i }$ . These statements follow from triangle inequalities. A new task or unbounded drift requires new coverage evidence.

## E.6 AMORTIZED EVIDENCE COST AT POSITIVE DECISION MARGINS

Assume shared rewards, $H \geq 2$ and $R _ { b } > 0$ . Set

$$
K = R _ { b } H ( H - 1 ) , \qquad q _ { i } = \operatorname* { m a x } _ { s , a } \mathrm { T V } ( P , \widehat { P } _ { i } ) , \qquad a _ { i } = \widehat { \Gamma } _ { i } - K q _ { i } .
$$

The quantity $a _ { i }$ is a sufficient margin involving the unknown true model error, not an observable gain estimate.

Proposition 9 (Amortized verification). On the row-confidence event, $u _ { i } \leq q _ { i } + 2 a _ { n }$ simultaneously for every world model. Exact world-model evaluation therefore gives (28). The sample size (29) certifies every request with $a _ { i } / K \geq s .$ . For a sampled estimate satisfying $| S _ { i } - \widehat { \Gamma } _ { i } | \leq \xi _ { i }$ , the corresponding bound is

$$
T _ { i } \geq a _ { i } - 2 K a _ { n } - 2 \xi _ { i } .
$$

Proof. For every row, the triangle inequality gives

$$
\mathrm { T V } ( { \overline { { P } } } , { \widehat { P } } _ { i } ) + a _ { n } \leq \mathrm { T V } ( P , { \widehat { P } } _ { i } ) + 2 a _ { n } .
$$

Maximization and clipping preserve this upper bound. With $S _ { i } = \widehat { \Gamma } _ { i }$ and $\xi _ { i } = 0 , T _ { i } = \widehat { \Gamma } _ { i } - K u _ { i } \geq$ $a _ { i } - 2 K a _ { n }$ . Condition (29) is exactly $2 a _ { n } < s . { \mathrm { I f } } a _ { i } / K \geq s ,$ it follows that $T _ { i } \geq K ( s - 2 a _ { n } ) > 0$ For sampled evaluation, $S _ { i } \geq \widehat { \Gamma } _ { i } - \xi _ { i }$ gives the sampled-margin inequality above.

The real evidence cost is $m n ,$ instead of Nmn for repeating that row audit before each of N comparisons. Imagined evaluation and candidate search remain additional costs. Positive margins are a premise; their existence is not implied by confidence coverage. The statement is deterministic on the common event, so it remains valid for adaptive requests meeting that premise. A post-selection random margin cannot be treated as a prespecified unconditional power guarantee. When $H = 1$ and rewards are shared, transitions do not affect returns and no transition audit is required.

## E.7 STOPPING AN AUDIT AT A POSITIVE MARGIN

For adaptively chosen audit size, use

$$
\alpha _ { n } = \frac { \alpha } { n ( n + 1 ) } , \qquad a _ { n } = \sqrt { \frac { D \log 2 + \log ( 4 m / \alpha _ { n } ) } { 2 n } } .
$$

For each row take successive prefixes of an independent sample stream. Since $\textstyle \sum _ { n > 1 } \alpha _ { n } = \alpha _ { 1 }$ , one event covers all integer sample sizes and permits stopping at the first positive margin. The log factor now grows with n; the fixed-size bound is not an anytime bound.

For a fixed request with exact imagined contrast and positive sufficient margin $a = \widehat { \Gamma } - K q$ , fixed-size verification needs at most any integer satisfying

$$
n > \frac { 2 K ^ { 2 } } { a ^ { 2 } } [ D \log 2 + \log ( 4 m / \alpha ) ] .
$$

The right side is a sufficient audit budget, not a bound on how quickly learning reduces the model error q.

## E.8 A LOWER BOUND ON THE EVIDENCE NEEDED FOR PROMOTION

We prove the necessary query budget (30), with Bernoulli relative entropy kl defined below.

Proposition 10 (Evidence required by a useful gate). For every $\gamma \in ( 0 , 1 / 4 ]$ there are two two-step worlds and one fixed supplied world model such that one fixed candidate decision is beneficial in one and harmful in the other. If the audit’s only world-dependent observations are row-query answers, promotion probability at least $1 - \delta$ in the first world and at most δ in the second requires (30), for $\bar { 0 } < \delta < 1 \bar { / } 2 $ , even with adaptive query selection.

We prove Proposition 10. All side information at the start of the audit is identical under the two hypotheses; only new oracle answers distinguish them. Use horizon two and states $\{ s , g , b \}$ , with zero reward at $s , b$ and terminal reward $c \in ( 0 , R _ { b } ]$ at $g .$ There are two actions. In both worlds $P ( g \mid s , a _ { 0 } ) = 1 / 2$ ; in the plus and minus worlds, respectively, $P ( g \mid s , a _ { 1 } ) = 1 / 2 + \gamma$ and $1 / 2 - \gamma$ All other rows agree. The supplied world model is the plus world, and the same policy family is used in both: $\pi _ { \theta } ( a _ { 1 } \mid s ) = \sigma ( \theta )$ . Then

$$
J _ { + } ( \theta ) = c [ 1 / 2 + \gamma \sigma ( \theta ) ] , J _ { - } ( \theta ) = c [ 1 / 2 - \gamma \sigma ( \theta ) ] .
$$

Replacing $a _ { 0 }$ by $a _ { 1 }$ has contrasts $c \gamma$ and $- c \gamma$ , respectively. Thus a safe and useful promotion decision must distinguish the two worlds. The logistic parameterization merely interpolates between the same two actions.

For $p , q \in ( 0 , 1 )$ define

$$
\begin{array} { r } { \mathrm { k l } ( p , q ) = p \log ( p / q ) + ( 1 - p ) \log [ ( 1 - p ) / ( 1 - q ) ] . } \end{array}
$$

Use the usual limiting conventions at zero and one. One informative query, at $( s , a _ { 1 } )$ , has relative entropy

$$
\mathrm { k l } ( 1 / 2 + \gamma , 1 / 2 - \gamma ) = 2 \gamma \log \frac { 1 + 2 \gamma } { 1 - 2 \gamma } \leq 1 6 \gamma ^ { 2 } .
$$

For the inequality use $\log [ ( 1 + x ) / ( 1 - x ) ] \leq 2 x / ( 1 - x ) \mathrm { a t } x = 2 \gamma \leq 1 / 2$ . Queries at other rows have zero relative entropy.

Let an audit make at most n adaptively chosen row queries; include its independent randomization in the transcript $Y _ { 1 : n }$ . Conditional on $\mathscr { H } _ { t - 1 } = \sigma ( Y _ { 1 : t - 1 } )$ , its selection kernel is identical under both hypotheses. Writing $\mathbb { P } _ { + } ^ { n } , \mathbb { P } _ { - } ^ { n }$ for the transcript laws, the relative-entropy chain rule yields

$$
\begin{array} { r l } {  { D _ { \mathrm { K L } } \bigl ( \mathbb { P } _ { + } ^ { n } \| \mathbb { P } _ { - } ^ { n } \bigr ) = \sum _ { t = 1 } ^ { n } \mathbb { E } _ { + } \bigl [ D _ { \mathrm { K L } } \bigl ( \mathbb { P } _ { + } \bigl ( Y _ { t } \mid \mathcal { H } _ { t - 1 } \bigr ) \bigr \| \mathbb { P } _ { - } \bigl ( Y _ { t } \mid \mathcal { H } _ { t - 1 } \bigr ) \bigr ) \bigr ] } \quad } & { } \\ & { \leq \sum _ { t = 1 } ^ { n } \mathrm { k l } ( 1 / 2 + \gamma , 1 / 2 - \gamma ) } \\ & { \leq 1 6 n \gamma ^ { 2 } . } \end{array}
$$

If the audit stops early, append uninformative queries to reach $n ;$ this changes neither the decision nor the divergence bound.

Writing $p$ and $q$ for the plus and minus promotion probabilities, data processing for the binary decision gives transcript divergence at least $\operatorname { k l } ( p , q )$ . This follows by partitioning the likelihood-ratio integral over promotion and rejection and applying Jensen on each part. Since $p \ge 1 - \delta > \delta \ge q$ , Bernoulli relative entropy is increasing in $p$ and decreasing in q on this region; consequently

$$
\begin{array} { r l } & { \mathrm { 1 6 } n \gamma ^ { 2 } \geq D _ { \mathrm { K L } } ( \mathbb { P } _ { + } ^ { n } \| \mathbb { P } _ { - } ^ { n } ) } \\ & { ~ \geq \mathrm { k l } ( p , q ) } \\ & { ~ \geq \mathrm { k l } ( 1 - \delta , \delta ) , } \end{array}
$$

which proves (30). The bound is for evidence needed by a safe, high-power promotion test, not a general lower bound on world-model training. It shows that an arbitrarily small unresolved intervention margin cannot be certified from a fixed amount of real evidence.

Together with the sufficient audit bound in Appendix E.6, this establishes the matching $\Theta ( \gamma ^ { - 2 } )$ dependence on the unresolved decision margin. We do not claim matching state-space or horizon dependence: those factors arise from the particular simultaneous finite-state verifier used for the upper bound.

## E.9 DIRECT RETURN VERIFICATION FOR GENERAL INTERFACES

A finite-state confidence set is not necessary if true episodic evaluation is available. Choose policies $\pi , \pi ^ { \prime }$ before collecting fresh evaluations. Let $Z _ { i }$ be the difference of their returns in independent episode pairs. Pairing within an episode index is allowed, and $Z _ { i } \in \left[ - H R _ { b } , H R _ { b } \right]$ . Then

$$
J ( \pi ^ { \prime } ) - J ( \pi ) \ge \overline { { { Z } } } - H R _ { b } \sqrt { \frac { 2 \log ( 1 / \alpha ) } { n } }
$$

with probability at least $1 - \alpha$ , by one-sided Hoeffding. A positive lower bound directly certifies the candidate improvement, including for black-box planners and history-based agents.

This audit is more general but can be expensive: it validates each candidate with real episodes instead of amortizing model reliability across many imagined candidates. Its false-positive event, conditional on pre-audit choice, fits Theorem 7. If candidates are selected using these same evaluations, use a simultaneous finite-pool correction or fresh evaluation again.

## E.10 ELEMENTARY CONCENTRATION TOOLS

For completeness, both concentration inequalities used above follow from a bounded exponential moment. If $X \in [ a , b ]$ , let $K ( \lambda ) = \log \bar { \mathbb { E } } \exp [ \lambda ( X - \mathbb { E } X ) ]$ Under the exponentially tilted law, $K ^ { \prime \prime } ( \lambda )$ is the variance of X, bounded by $( b - a ) ^ { 2 } / 4 \colon$ center X at $( a + b ) / 2$ and use that variance is at most the corresponding second moment. Since $K ( 0 ) = K ^ { \prime } ( 0 ) = 0$ , integrating twice gives

$$
\mathbb { E } \exp [ \lambda ( X - \mathbb { E } X ) ] \leq \exp \{ \lambda ^ { 2 } ( b - a ) ^ { 2 } / 8 \} .
$$

For independent variables, multiply these bounds. Markov’s inequality for the exponential and optimization over $\lambda > 0$ yield $\mathbb { P } \{ \stackrel {  } { X } - \mathbb { E } \overline { { X } } \geq t \} \leq \exp [ - 2 n t ^ { 2 } / ( b - a ) ^ { 2 } ]$ . Apply the same argument $\mathrm { t o } - X$ and add probabilities for a two-sided bound.

If a function $f ( X _ { 1 } , \ldots , X _ { n } )$ of independent variables changes by at most $c _ { i }$ when only $X _ { i }$ changes, reveal the variables sequentially. The resulting Doob martingale difference has conditional mean zero and conditional range of length at most $c _ { i } { : }$ averaging over the remaining independent variables preserves the coordinate-wise difference bound. Applying the bounded exponential-moment inequality above conditionally and iterating gives

$$
\mathbb { P } \{ f - \mathbb { E } f \geq t \} \leq \exp \left[ - \frac { 2 t ^ { 2 } } { \sum _ { i } c _ { i } ^ { 2 } } \right] .
$$

Using $c _ { i } = 2 C _ { H } / N$ proves the tail step in Proposition 6. These arguments also hold conditional on a pre-audit history when the required conditional independence is satisfied.

## F ADAPTIVE CO-LEARNING AND RESOURCE ACCOUNTING

Theorem 7 (Adaptive fresh verification). Let $\mathcal { H } _ { k - 1 }$ contain all information before round $k ' s$ fresh audit, including its task choice. Ifthe combined conditionalfailure probability is at most $\varepsilon _ { k }$ and only positive certified comparisons are executed, let B denote the event of at least one non-improving execution in this sequence. Then

$$
\mathbb { P } ( \mathcal { B } ) \le \sum _ { k \ge 1 } \varepsilon _ { k } , \qquad \mathbb { E } N _ { K } \le \sum _ { k = 1 } ^ { K } \varepsilon _ { k } ,
$$

where $N _ { K }$ counts non-improving executions through round $K .$

Proof of Theorem 7. Each bad execution implies audit or estimation failure. Conditional expectation and the tower property give its marginal bound; countable subadditivity and linearity give the two conclusions. Unlike Theorem 4, this permits arbitrary task changes, but pays for fresh validity.

## F.1 FILTRATION AND REPEATED PROMOTION

At round k, $\mathcal { H } _ { k - 1 }$ contains previous world-model fits, policies, generator choices, and audit outcomes, together with the present task choice made before its fresh random samples. Let $E _ { k }$ be the event on which all current world-model-error and sampling bounds used by the gate hold. Assume $\mathbb { P } ( E _ { k } ^ { c } \mid \mathcal { H } _ { k - 1 } ) \le \varepsilon _ { k } .$ , where $\varepsilon _ { k }$ is deterministic or predictable with a deterministic summable upper envelope. For simplicity the main theorem uses deterministic $\varepsilon _ { k }$

Let $I _ { k }$ indicate that the algorithm accepts a candidate behavior and its true target return does not increase. The deterministic margin theorem gives $I _ { k } \leq \mathbb { I } \{ E _ { k } ^ { c } \}$ . Consequently

$$
\begin{array} { r l } & { \mathbb { E } [ I _ { k } \mid \mathcal { H } _ { k - 1 } ] \le \mathbb { E } [ \mathbb { I } \{ E _ { k } ^ { c } \} \mid \mathcal { H } _ { k - 1 } ] = \mathbb { P } ( E _ { k } ^ { c } \mid \mathcal { H } _ { k - 1 } ) \le \varepsilon _ { k } , } \\ & { \qquad \mathbb { E } \sum _ { k = 1 } ^ { K } I _ { k } = \displaystyle \sum _ { k = 1 } ^ { K } \mathbb { E } [ \mathbb { E } ( I _ { k } \mid \mathcal { H } _ { k - 1 } ) ] \le \sum _ { k = 1 } ^ { K } \varepsilon _ { k } . } \end{array}
$$

Moreover, the event of at least one bad accepted comparison is contained in $\cup _ { k \geq 1 } E _ { k } ^ { c }$ , whence

$$
\begin{array} { r l } {  { \mathbb { P } ( \mathcal { B } ) \le \mathbb { P } ( \bigcup _ { k \ge 1 } E _ { k } ^ { c } ) \le \sum _ { k \ge 1 } \mathbb { P } ( E _ { k } ^ { c } ) } } \\ & { = \sum _ { k \ge 1 } \mathbb { E } [ \mathbb { P } ( E _ { k } ^ { c } \mid \mathcal { H } _ { k - 1 } ) ] \le \sum _ { k \ge 1 } \varepsilon _ { k } . } \end{array}
$$

Choosing $\varepsilon _ { k } = \varepsilon / [ k ( k + 1 ) ]$ protects every round with total risk at most ε.

Proposition 8 supplies conditional world-model validity after adaptive task choice; the conditional sampling inequality above supplies simulation accuracy. Their error budgets add. This argument never conditions a marginal guarantee on a data-dependent promotion event. Between rounds all world models and task choices may change; this particular argument restores validity by fresh verification. For repeated uses on one fixed task, Theorem 4 instead retains a simultaneous world-confidence event and charges its failure probability only once.

## F.2 FIXED-OBJECTIVE PROGRESS AND STATIONARITY

The main-text contrast telescope (26) requires no derivatives. For differentiable predictive policy search, the local certificate also yields the following stationarity specialization. The contrast telescope controls expected return rather than samplewise episode rewards. Because $J ( \pi ) \le H R _ { b }$ , any fixed positive certified margin can occur only finitely many times along a sequence of accepted updates; this is the content of (27). The statement does not bound the waiting time, number of rejected proposals, or real samples required between accepted updates. On the event of simultaneous validity, enumerate accepted policy updates by $i = 0 , \ldots , n - 1$ . Model-only steps between them do not change the policy. Assume they optimize one fixed bounded objective J, have common smoothness $L ,$ and total gradient radius $b _ { i } \leq \kappa h _ { i }$ , where $h _ { i } = \| \widetilde { g } _ { i } \| _ { 2 }$ and $0 \leq \kappa < 1$ . Use $\eta = ( 1 - \kappa ) / L$ and require each segment to remain in the admissible region. Theorem 6 gives

$$
J ( \theta _ { i + 1 } ) - J ( \theta _ { i } ) \geq \frac { ( 1 - \kappa ) ^ { 2 } } { 2 L } h _ { i } ^ { 2 } .
$$

Since $\| \nabla J ( \theta _ { i } ) \| _ { 2 } \le ( 1 + \kappa ) h _ { i }$ , summing yields

$$
{ \frac { ( 1 - \kappa ) ^ { 2 } } { 2 L ( 1 + \kappa ) ^ { 2 } } } \sum _ { i = 0 } ^ { n - 1 } \| \nabla J ( \theta _ { i } ) \| _ { 2 } ^ { 2 } \leq J ( \theta _ { n } ) - J ( \theta _ { 0 } ) \leq \operatorname* { s u p } _ { \theta } J ( \theta ) - J ( \theta _ { 0 } ) .
$$

Therefore

$$
\operatorname* { m i n } _ { 0 \leq i < n } \| \nabla J ( \theta _ { i } ) \| _ { 2 } ^ { 2 } \leq \frac { 1 } { n } \sum _ { i = 0 } ^ { n - 1 } \| \nabla J ( \theta _ { i } ) \| _ { 2 } ^ { 2 } \leq \frac { 2 L ( 1 + \kappa ) ^ { 2 } [ \operatorname* { s u p } _ { \theta } J ( \theta ) - J ( \theta _ { 0 } ) ] } { ( 1 - \kappa ) ^ { 2 } n } .
$$

If infinitely many updates are accepted, boundedness of J also gives

$$
\sum _ { i = 0 } ^ { \infty } \| \nabla J ( \theta _ { i } ) \| _ { 2 } ^ { 2 } < \infty \quad \Longrightarrow \quad \operatorname* { l i m } _ { i \to \infty } \| \nabla J ( \theta _ { i } ) \| _ { 2 } = 0 .
$$

Existence of infinitely many certified updates is not assumed to follow from validity alone.

A fixed distribution over tasks can be incorporated in the initial state, defining one expected-return objective. By contrast, certifying improvement only for a selected task does not certify improvement of the task average. If objectives change to $J _ { i }$ with $\operatorname* { s u p } _ { \theta } | J _ { i + 1 } ( \theta ) - J _ { i } ( \theta ) | \leq v _ { i }$ , then

$$
\sum _ { i = 0 } ^ { n - 1 } [ J _ { i } ( \theta _ { i + 1 } ) - J _ { i } ( \theta _ { i } ) ] \leq H R _ { b } + \sum _ { i = 1 } ^ { n - 1 } v _ { i - 1 } .
$$

To verify this, telescope while adding $J _ { i - 1 } ( \theta _ { i } ) - J _ { i } ( \theta _ { i } )$ at every internal point. The endpoint range is at most $H R _ { b }$ and each internal discrepancy is at most its stated drift bound. Substituting this right-hand side in the preceding calculation gives a drift-corrected average bound for $\lVert \nabla J _ { i } ( \mathsf { \overline { { \boldsymbol { \theta } } } } _ { i } ) \rVert _ { 2 } ^ { 2 }$ Without drift control it is not a convergence theorem for a fixed target.

## F.3 WHAT RELIABILITY DOES AND DOES NOT ALLOCATE

The frontiers in (18)–(19) are disjoint: ${ \mathcal { F } } _ { \mathrm { W } }$ has nonpositive certified margin, while $\mathcal { F } _ { \mathrm { A } }$ has positive margin. Membership in ${ \mathcal { F } } _ { \mathrm { W } }$ requests evidence or model refinement for a promising comparison. Membership in $\mathcal { F } _ { \mathrm { A } }$ authorizes the candidate behavior. Neither identifies failure ownership or supplies an environment-wide guarantee.

A model can be reliable while its imagined return estimate is too noisy. Increasing the rollout count reduces ξ without changing the predictive kernel. A request with no statistically resolved apparent benefit is deferred. This third possibility prevents a two-way routing rule from treating lack of evidence as proof of model ignorance or agent incompetence.

To define the next task, a generator may prioritize uncertainty, measured progress, diversity, or intervention coverage. The theory constrains the use of a selected task after verification; it does not claim an optimal open-ended task generator.

## F.4 DETERMINISTIC AND STOCHASTIC PRECEDENCE ACCOUNTING

Let Z now be a finite task archive. For each task fix nonnegative integers $w _ { z } , a _ { z } .$ . A productive model update decrements $w _ { z }$ by one. A productive policy update decrements $a _ { z }$ by one only once $w _ { z } = 0$ Updates have no transfer and no regression. An update outside these conditions is ineffective. These assumptions are an explicit learning-response abstraction, not consequences of a value-error bound.

Theorem 8 (Exact precedence accounting). Let $\begin{array} { r } { B ^ { * } = \sum _ { z } ( w _ { z } + a _ { z } ) } \end{array}$ . Every completed schedule has length

$$
\boldsymbol { B } = \boldsymbol { B } ^ { * } + \boldsymbol { m } ,
$$

where m is its number of ineffective updates. An eligible work-conserving schedule attains $B ^ { * }$

More generally, suppose an eligible model trial on task z succeeds independently with probability $p _ { z } > 0$ at cost $c _ { z } > 0 ,$ , and an eligible policy trial succeeds with probability $q _ { z } > 0$ at cost $d _ { z } > 0$ Each success decrements its corresponding integer by one. Every schedule that uses only eligible trials and continues until all tasksfinish has expected cost

$$
B ^ { * } = \sum _ { z } \left( \frac { c _ { z } w _ { z } } { p _ { z } } + \frac { d _ { z } a _ { z } } { q _ { z } } \right) .
$$

Extra ineffective trials add their expected costs.

Proof. For the deterministic case, let $\begin{array} { r } { \Phi _ { k } = \sum _ { z } ( w _ { z } ( k ) + a _ { z } ( k ) ) } \end{array}$ be the remaining work after k updates, and let $I _ { k }$ indicate that update k is ineffective. At completion time $B$

$$
\Phi _ { 0 } = B ^ { * } , \qquad \Phi _ { B } = 0 , \qquad \Phi _ { k - 1 } - \Phi _ { k } = 1 - I _ { k } .
$$

Writing $\begin{array} { r } { m = \sum _ { k = 1 } ^ { B } I _ { k } } \end{array}$ , summation yields

$$
B ^ { * } = \sum _ { k = 1 } ^ { B } ( \Phi _ { k - 1 } - \Phi _ { k } ) = B - m , \qquad B = B ^ { * } + m .
$$

Eligible schedules have $m = 0$ and attain $B ^ { * }$

For the stochastic case, attach independent Bernoulli sequences to each task and stage. A nonanticipating schedule reveals the next unused entry only when that stage is attempted. Let $U _ { z } , V _ { z }$ be the numbers of eligible model and policy trials required to exhaust the two quotas. Geometric waiting times give

$$
\begin{array} { r } { \mathbb { E } U _ { z } = w _ { z } / p _ { z } , \qquad \mathbb { E } V _ { z } = a _ { z } / q _ { z } . } \end{array}
$$

Every completing eligible schedule reveals exactly these prefixes, irrespective of interleaving. If C is its total cost, then

$$
C = \sum _ { z } ( c _ { z } U _ { z } + d _ { z } V _ { z } ) , \qquad { \mathbb { E } } C = \sum _ { z } \left( \frac { c _ { z } w _ { z } } { p _ { z } } + \frac { d _ { z } a _ { z } } { q _ { z } } \right) = B ^ { * } .
$$

Ineffective trials neither reveal a productive entry nor change a quota; their costs add pathwise.

Thus order inside an eligible frontier need not be uniquely optimal: the theorem characterizes a class of schedules with identical completion cost in this abstraction. It does not prove dominance over every scalar score, because a scalar rule can itself respect precedence. Cross-task transfer, warm-start benefits of early policy training, uncertain learning curves, forgetting, and model updates that never improve certification all change the allocation problem.

If every harmful false promotion in this abstraction wastes at most c units of work, Theorem 7 bounds its expected wasted work through round K by $c \textstyle \sum _ { k = 1 } ^ { K } \varepsilon _ { k }$ . This is the cost of erroneous certification alone. Conservative rejection, audit cost, and ordinary optimization failure are additional costs and are not hidden inside that bound.

## G CONTROLLED LEARNED-WORLD-MODEL EXPERIMENT

## G.1 EXPERIMENTAL EXECUTION

We evaluate action-conditioned predictors learned from sampled experience in finite-horizon worlds. Sparse, chain, and grid families each use 16 states, four actions, known support, bounded shared rewards, and horizons $H \in \{ 4 , 8 , 1 2 \}$ }. Forty development worlds fix the protocol, 80 independent worlds supply the finite-world audit quantities, and 240 held-out worlds are used once for evaluation. World identifiers and random streams are disjoint across these roles. The predictor is the Betasmoothed empirical transition kernel fitted from nested row observations; a separately acquired 32-query-per-row model defines the fixed reference behavior. Every gate sees only the learned predictor and its permitted audit evidence. Exact simulator returns are revealed only after routing and are never used to choose a proposal, a gate, or an evidence location.

The protocol has three complementary components. First, 400 paired constructions share the learned model and passive record but differ on one unobserved action, isolating whether intervention evidence identifies the otherwise hidden world effect. Second, held-out planning and teaching requests compare unconditional positive-imagination acceptance, uncertainty-only rejection, and decisionrelative qualification; the same requests measure value error, update-direction cosine, harmful-use rate, and realized gain. Third, online runs compare ungated, uniformly gated, and decision-directed evidence acquisition at an equal real-transition budget. Candidate proposals, reference behavior, and evaluation streams are matched within every comparison, so the only manipulated object is whether the world-model prediction is qualified and where new evidence is acquired.

Two theory-directed supplements reuse the same finite-world row-audit primitive. For evidencecomplexity scaling, the audited Bernoulli row has means $1 / 2 - \gamma$ and $1 / \bar { 2 } + \gamma$ in the two possible worlds, with 13 logarithmically spaced margins $\gamma \in \left[ 0 . 0 2 , 0 . 2 0 \right]$ . At each margin we enumerate sample sizes and report the smallest $n _ { 0 . 9 }$ for which the exact randomized Neyman–Pearson test reaches 90% power while controlling false promotion at $\alpha = 0 . 0 5 \mathrm { ; }$ ; the log–log slope is fitted once across all registered margins. For evidence reuse, we fix $\gamma = 0 . 0 8$ , audit all eight transition rows, and evaluate $K \mathsf { \bar { \in } } \{ 1 0 , 2 0 , 5 0 , 1 0 0 , 2 0 0 \}$ later comparisons. The shared rule pays for one simultaneous α-level audit; the re-audit baseline pays for $\dot { K }$ fresh tests, each assigned error $\alpha / K$ and separately sized to the same 90% power. Consequently both procedures control familywise false promotion, while their real-query costs are measured under identical accuracy requirements. Exact integer results and the complete plotting records are retained with the experiment archive.

Theory-to-measurement correspondence. The finite-world design gives each theoretical statement an observable counterpart. Paired worlds implement passive non-identifiability and interventional recovery (Theorem 1 and Proposition $2 ) ;$ attribution accuracy and model-effect error are evaluated before any decision gate is scored. Planning curves test decision reliability and the separation between predictive accuracy and decision safety (Theorem 2 and Corollary 2) through realized gain, harmful-use rate, and risk–coverage. Online runs instantiate the closed-loop and adaptiveacquisition statements (Theorem 3 and Theorem 7), while gradient and teaching diagnostics test gradient reliability and local safe improvement (Theorem 5 and Corollary 6). The log–log experiment tests the evidence lower-bound order (Proposition 10), and the reuse experiment tests amortized verification and precedence accounting (Proposition 9 and Theorem 8). This one-to-one mapping makes the appendix a validation suite for the theory rather than a collection of unrelated plots.

## G.2 RESULTS AND SUPPLEMENTARY SUMMARIES

The results separate the three theoretical claims cleanly. The paired construction reaches 100% attribution accuracy at the largest intervention budget, while model-effect error decreases from 0.08999 to 0.01888 (Figure 7). On held-out planning requests, accepting every positive imagined advantage produces 5.10% harmful uses; decision-relative qualification reduces this to 0.30% while preserving positive aggregate gain (Figures 8–9). The update-direction supplement raises mean true/imagined gradient cosine from 0.761 to 0.9994 as evidence increases. In the online study, ungated, uniform-gated, and decision-directed acquisition incur 44, 10, and 4 harmful updates, respectively (Figures 10–12). The evidence-complexity fit has slope 2.014 on the log–log scale, matching the predicted $\gamma ^ { - 2 }$ order. At K = 200 downstream comparisons, one shared audit uses 664 real queries versus 350,400 for fresh re-auditing, a 527.7× reduction under the same registered familywise error and power targets.

These conclusions do not rely on post-hoc selection: all registered worlds and margins are retained, gates never observe exact evaluation returns, and paired methods reuse identical stochastic streams. The simultaneous theorem-derived certificate can abstain completely at small budgets, and adaptive acquisition is not uniformly superior in every cell; we report these limitations rather than replacing them with a favorable subset. The experiment therefore checks the finite-world assumptions and scaling laws directly, while the agent benchmark in Appendix H separately tests whether the same verify-then-promote structure remains useful without claiming a finite-state certificate.

![](images/b2a1197d2431f6a6a9af4abeee76b3b75da0d07dd82ae26e871d03a0632b14ad.jpg)

![](images/f4d5a67be6711469ea71eb98671747f5907bb412348e29dd989c74aedd3c5630.jpg)  
Figure 7: Attribution and model-effect error under increasing intervention evidence. Active actionconditioned queries resolve worlds that are indistinguishable under the shared passive record.

![](images/11236db6315f26c37875fe6105057590a9bd95ba1a15d64923ce0d7d4b39832e.jpg)

![](images/4eff5c07000994d7e20020ca74d8d6059c4dde37a26b01fbca42356099e08f4b.jpg)  
Figure 8: Decision-relative planning qualification. The risk–coverage curve (left) and imaginedversus-realized changes (right) show that screening the proposed use, rather than trusting positive imagination alone, removes most harmful promotions.

![](images/28a6dd93f5ab613c1adaf84d3a700386dc817939de92871faff7623933ff360f.jpg)

![](images/3dc08c9bf74d95f4da046f26bd84235df5b72cc3f202d535c97b8ac49328c3e1.jpg)  
Figure 9: Planning gain and promotion rate as action-conditioned evidence increases. Reliability improves without changing the candidate requests or reference policy.

![](images/079875ec57a916f4529f8743749a35c4d00103ff8bce09f42e400a19a8b03a60.jpg)

![](images/1629a7317e2e5f8c0429fbc9dc53809bf18d8a8e22c4c09f0a1e6ea19494a86a.jpg)  
Figure 10: Certificate diagnostics. Measured decision errors remain below their theorem-derived radii over the registered finite-world requests.

![](images/1d0f3822f56f47c4f7cbc3161ea5ddbca1d4936697695d460b6492da86d14cc3.jpg)

![](images/3dca85c6751a75c4b989af7826f8ea7876baf1792986f1a58850f4ada10ccc48.jpg)

![](images/4d5168ccf637e02adeb3b10358f8660e0943337de808ab90d05def5557cc7d02.jpg)  
Figure 11: Online qualification and acquisition under equal real-transition budgets. Decision-directed evidence reduces harmful updates relative to ungated and uniform alternatives.

![](images/d6abdaffd72ca3ac1f8f1a8b7013759b6b6c752c1acdeaab3dc12a442cf72f27.jpg)

![](images/7d40b472f2b3d0d54fb8b5ff3777a2432a53ec490cd84cfa6ce761daced4ad46.jpg)

![](images/1df729d7926d2212f7106e372fe70a7c6c06e2a84d39e7f0358598cb26845085.jpg)  
Figure 12: Transfer diagnostics for update-direction alignment (left), teaching risk–coverage (center), and imagined-versus-realized teaching gains (right).

## H AGENT–WORLD-MODEL BENCHMARK EVALUATION

The benchmark study evaluates the Dual-Frontier admission rule of Section 5 under a language-worldmodel interface. We fix Qwen-AgentWorld as the learned world model, vary the agent backbone between Llama-3.1-8B-Instruct and Qwen3-8B, and evaluate BFCL (Patil et al., 2025), API-Bank (Li et al., 2023), and NexusRaven (Srinivasan et al., 2023). For each benchmark, the request manifest is partitioned once into a 20% calibration split and an 80% held-out test split. The partition is fixed before calibration and routing evaluation and is shared across all compared methods; the same request partition is used for both agent backbones.

Each request defines a reference–candidate comparison before admission: the base-agent output is the reference and a fixed Qwen-AgentWorld proposal is the candidate. Base actions and worldmodel generations are produced once and cached, and every non-Agent-only routing rule acts on the same candidate records. Calibration requests are used only to construct the decision-relevant error radius entering the Dual-Frontier margin. Their benchmark references are accessed only after al corresponding model-side predictions have been frozen. Test references are never used in candidate construction, calibration, or admission and are loaded only after the routed test outputs are fixed. Prompts, tool schemas, sampling budgets, parsers, generation order, and candidate-generation records are otherwise shared across routing rules.

## H.1 DECISION RULES AND CALIBRATED DUAL-FRONTIER INSTANTIATION

Predicted advantage and estimation uncertainty. Once the shared proposal $c _ { i } ^ { K }$ is fixed, the world model evaluates its consequence relative to the reference action $c _ { i } ^ { 0 }$ using a fixed repeated-evaluation budget. Let $\widehat { \gamma } _ { i \ell } \in [ - 1 , 1 ] , \mathsf { \bar { \ell } } = 1 , \dots , L$ , denote the resulting normalized candidate-minus-reference consequence scores. We define

$$
S _ { i } = \frac { 1 } { L } \sum _ { \ell = 1 } ^ { L } \widehat { \gamma } _ { i \ell } , \qquad \xi _ { i } = \sqrt { \frac { 2 \log ( 1 / \delta ) } { L } } ,\tag{31}
$$

where the same $L$ and $\delta$ are used throughout evaluation. Thus $S _ { i }$ is the benchmark-side predicted advantage and $\xi _ { i }$ accounts for finite-generation estimation uncertainty.

Decision-relevant model-error calibration. The repeated world-model records additionally provide proposal-level reliability information. We collect these quantities in

$$
z _ { i } = \big ( 1 - \bar { \rho } _ { i } , 1 - \kappa _ { i } , h _ { i } ^ { \operatorname* { m a x } } \big ) , \qquad \psi _ { i } = g ( z _ { i } ) ,\tag{32}
$$

where $g$ is a fixed nonnegative aggregation rule specified before calibration and used unchanged at test time. The individual confidence, agreement, and consequence-risk signals therefore contribute to the model-error estimate rather than acting as separate Dual-Frontier admission thresholds.

For each calibration request $j ,$ , all model-side quantities are frozen before the benchmark evaluator provides the normalized candidate-minus-reference contrast $\Gamma _ { j } ^ { \mathrm { e v a l } }$ . We calibrate the residual decision discrepancy not already accounted for by $\xi _ { j }$

$$
e _ { j } ^ { B } = \left[ \left| S _ { j } - \Gamma _ { j } ^ { \mathrm { e v a l } } \right| - \xi _ { j } \right] _ { + } , \qquad q _ { j } = e _ { j } ^ { B } - \psi _ { j } .\tag{33}
$$

For a calibration set of size $n _ { \mathrm { c a l } } .$ , let

$$
k = \lceil ( n _ { \mathrm { c a l } } + 1 ) ( 1 - \alpha ) \rceil , \qquad B _ { i } = \big [ \psi _ { i } + q _ { ( k ) } \big ] _ { + } ,\tag{34}
$$

with $q _ { ( k ) }$ the corresponding calibration quantile. Calibration is performed only on the designated 20% split; $B _ { i }$ is then evaluated without test labels on the remaining requests. Under the exchangeability condition of Proposition $^ { 7 , }$ , the same rank argument gives

$$
\operatorname* { P r } \{ e _ { i } ^ { B } > B _ { i } \} \leq \alpha .\tag{35}
$$

Dual-Frontier promotion rule. The benchmark implementation uses the same three-term margin as Eq. (17):

$$
T _ { i } ^ { \mathrm { D F } } = S _ { i } - B _ { i } - \xi _ { i } .\tag{36}
$$

A world-model proposal is promoted only when $T _ { i } ^ { \mathrm { D F } } > 0$ . On the calibration-coverage event in (35),

$$
\Gamma _ { i } ^ { \mathrm { e v a l } } \geq S _ { i } - B _ { i } - \xi _ { i } = T _ { i } ^ { \mathrm { D F } } ,\tag{37}
$$

so a positive benchmark margin has exactly the lower-bound semantics required by the Dual-Frontier decision rule. This construction separates predicted benefit, decision-relevant model error, and finite-generation estimation uncertainty while keeping the practical gate identical in form to the theoretical one.

Six routing rules. Let $a _ { i } ^ { r } \in \{ 0 , 1 \}$ denote whether rule r admits the shared proposal $c _ { i } ^ { K }$ . If no valid proposal exists, all non-Agent-only rules fall back to $c _ { i } ^ { 0 }$ . For the three single-signal comparison rules, we retain fixed thresholds

$$
\tau _ { \rho } = 0 . 7 0 , \qquad \tau _ { \kappa } = 3 / 7 , \qquad \tau _ { h } ^ { P } = 0 . 2 2 ,\tag{38}
$$

used only to define the corresponding baselines. The routing decisions are

$$
\begin{array} { r l r l r l } & { { a } _ { i } ^ { A } = 0 , \quad } & & { { a } _ { i } ^ { W } = 1 , \quad } & & { { a } _ { i } ^ { C } = \mathbb { I } \{ \bar { \rho } _ { i } \geq \tau _ { \rho } \} , } \\ & { { a } _ { i } ^ { K } = \mathbb { I } \{ \kappa _ { i } \geq \tau _ { \kappa } \} , \quad } & & { { a } _ { i } ^ { P } = \mathbb { I } \{ h _ { i } ^ { \operatorname* { m a x } } \leq \tau _ { h } ^ { P } \} , \quad } & & { { a } _ { i } ^ { D } = \mathbb { I } \{ T _ { i } ^ { \mathrm { D F } } > 0 \} . } \end{array}\tag{39}
$$

Here $A , W , C , K , P , D$ denote Agent-only, Always-WM, Confidence, Consistency, Pessimistic, and Dual-Frontier, respectively. The candidate is identical across $W , C , K , P , D ;$ ; only the admission rule changes. The routed output is

$$
c _ { i } ^ { r } = \{ { c _ { i } ^ { K } , } \quad a _ { i } ^ { r } = 1 , \quad\tag{40}
$$

The proposal-generation protocol, uncertainty construction, calibration level, and routing rules are fixed before evaluation of the held-out test split. Calibration references are confined to the designated calibration requests, and test references are inaccessible until (40) has been frozen. Every held-out request is retained in evaluation, including fallbacks and malformed outputs. Oracle is excluded because it observes reference outcomes and is not deployable.

## H.2 METRICS

For request i and routing rule r, let $c _ { i } ^ { 0 }$ denote the base-agent output and $c _ { i } ^ { r }$ the frozen routed output in (40). We use the fixed benchmark-aware evaluator of each benchmark throughout. Let $y _ { i } ( \bar { c } ) \in \{ 0 , 1 \}$ denote strict task success, $p _ { i } ( c ) \in [ 0 , 1 ]$ parameter-level accuracy, $u _ { i } ( c ) \bar { \in } [ 0 , 1 ]$ the decision-quality score assigned to output $c ,$ and $\bar { q _ { i } } ( c _ { i } ^ { 0 } , c ) \in [ 0 , 1 ]$ the corresponding reliability-loss score relative to the base action. All evaluator definitions are fixed before routing-rule replay.

Evaluation follows the native structural conventions of each benchmark. BFCL v4 preserves ordered and parallel-call multiplicity and validates function names and arguments against the supplied schema. API-Bank matches the required API identity and its normalized parameter dictionary. NexusRaven canonicalizes its Python-style function expression before comparing function identity, argument names, values, and multiplicity. Thus the metrics below share a common form while retaining the native call semantics of each benchmark.

Task success. The strict task-level metric is

$$
\mathrm { T S } _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } y _ { i } ( c _ { i } ^ { r } ) .\tag{41}
$$

It requires the complete function-call sequence, including function names, arguments, and call multiplicity, to satisfy the benchmark reference.

Parameter accuracy. We report the average parameter-level score

$$
\mathrm { P A } _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } p _ { i } ( c _ { i } ^ { r } ) ,\tag{42}
$$

where $p _ { i } ( \cdot )$ follows the corresponding benchmark’s native parameter matching and normalization rules.

Net decision gain. To measure the signed effect of routing through the learned world model, we use

$$
\mathrm { N D G } _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \left[ u _ { i } ( c _ { i } ^ { r } ) - u _ { i } ( c _ { i } ^ { 0 } ) \right] .\tag{43}
$$

Positive values indicate that the admitted world-model revisions improve the average decision quality relative to the base agent, whereas negative values indicate net degradation. We report NDG in percentage points.

Harmful revisions. Let $h _ { i } ( c _ { i } ^ { 0 } , c _ { i } ^ { r } ) \in [ 0 , 1 ]$ denote the evaluator’s degradation score for the routed output relative to the base action. We report

$$
\mathrm { H R R } _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } h _ { i } ( c _ { i } ^ { 0 } , c _ { i } ^ { r } ) .\tag{44}
$$

This quantity measures the overall exposure to harmful interventions, including both their occurrence and decision-level severity.

Revision coverage. The fraction of requests on which rule r admits a world-model revision is

$$
\mathrm { R C } _ { r } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } a _ { i } ^ { r } .\tag{45}
$$

Coverage is descriptive rather than an objective by itself and should be read jointly with the reliability metrics.

Selective risk. Among admitted revisions, we measure the residual reliability loss as

$$
\mathrm { S R } _ { r } = \frac { \sum _ { i = 1 } ^ { N } a _ { i } ^ { r } q _ { i } ( c _ { i } ^ { 0 } , c _ { i } ^ { r } ) } { \sum _ { i = 1 } ^ { N } a _ { i } ^ { r } } , \qquad \sum _ { i } a _ { i } ^ { r } > 0 .\tag{46}
$$

Selective risk is undefined when a rule never revises. The pair (RC, SR) therefore provides the empirical risk–coverage view of the verify-then-promote decision rule.

TS, PA, and NDG are higher-is-better metrics, whereas HRR and SR are lower-is-better; RC is descriptive. Here N always denotes requests in the held-out 80% test split; calibration requests are excluded from every reported benchmark metric. The reported Avg. column is the equal-weight arithmetic mean of the three benchmark percentages rather than a pooled micro-average.

## H.3 ADDITIONAL RESULTS

The component ablation removes one term at a time from the same Dual-Frontier decision margin while keeping the shared proposal, calibrated quantities, and cached world-model records fixed. Advantage-only admits when $S _ { i } > 0 ;$ w/o world-model error uses $S _ { i } - \xi _ { i } > 0 ;$ w/o estimation error uses $S _ { i } \bar { - } B _ { i } \stackrel { . } { > } 0 ;$ and full Dual-Frontier uses

$$
S _ { i } - B _ { i } - \xi _ { i } > 0 .
$$

The comparison therefore isolates the contribution of predicted advantage, decision-relevant model error, and finite-generation estimation uncertainty without changing proposal generation or test requests. Table 6 reports the Qwen3-8B results; the corresponding Llama-3.1-8B-Instruct results appear in Table 2.

The complete margin is consistently strongest. Relative to Advantage-only, Dual-Frontier improves the equal-weight Success/Acc. averages by 7.05/4.34 points for Llama-3.1-8B-Instruct and 6.27/3.92 points for Qwen3-8B. Removing either uncertainty term recovers only part of this gain, while their joint use yields the strongest performance for both backbones. This pattern is consistent with Section 5: predicted benefit identifies potentially useful interventions, whereas $B _ { i }$ and $\xi _ { i }$ determine whether that apparent advantage remains sufficiently supported for promotion.

Table 6: Additional Qwen3-8B ablation results. Success and Acc. denote task success and parameter accuracy; Avg. is the uniform mean over BFCL v4, API-Bank, and NexusRaven.
<table><tr><td rowspan="2">Models</td><td rowspan="2">Methods</td><td colspan="2">BFCL v4</td><td colspan="2">API-Bank</td><td colspan="2">NexusRaven</td><td colspan="2">Avg.</td></tr><tr><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td><td>Success</td><td>Acc.</td></tr><tr><td rowspan="4">Qwen3-8B</td><td>Advantage-only</td><td>67.80</td><td>84.94</td><td>95.07</td><td>96.41</td><td>94.40</td><td>95.71</td><td>85.76</td><td>92.35</td></tr><tr><td>w/o world-model error</td><td>72.32</td><td>88.01</td><td>96.88</td><td>97.50</td><td>96.02</td><td>96.68</td><td>88.41</td><td>94.06</td></tr><tr><td>w/o estimation error</td><td>77.79</td><td>91.18</td><td>98.40</td><td>98.31</td><td>97.11</td><td>97.52</td><td>91.10</td><td>95.67</td></tr><tr><td>Dual-Frontier</td><td>79.38</td><td>92.20</td><td>98.92</td><td>98.64</td><td>97.80</td><td>97.96</td><td>92.03</td><td>96.27</td></tr></table>

Coverage and conditional reliability. Tables 7 and 8 report how often each deployable rule admits a world-model revision and the residual risk among those admitted revisions. Agent-only is omitted because it never revises; consequently, its selective risk is undefined rather than zero. Revision coverage is descriptive, whereas lower selective risk is better. Their joint reading is essential: Dual-Frontier deliberately operates at moderate coverage while removing most unsafe promotions, exactly the selective qualification behavior predicted by the theory.

Across the three benchmarks, Dual-Frontier revises 57.59% of Llama and 57.51% of Qwen requests. At this nontrivial coverage, its average selective risk is 28.20% and 18.23%, respectively, versus 67.68% and 78.20% for the strongest competing selective baseline. The reductions of 39.48 and 59.97 percentage points explain why the primary-metric gains are not a consequence of indiscriminate revision: the method rejects precisely the candidate groups most likely to erase a correct base decision. Always-WM provides the opposite endpoint, with full coverage but selective risks of 78.89% and 86.29%.

Table 7: Revision coverage (%) across agent backbones and benchmarks. Avg. is the uniform mean over BFCL v4, API-Bank, and NexusRaven.
<table><tr><td>Models</td><td>Methods</td><td>BFCL v4</td><td>API-Bank</td><td>NexusRaven</td><td>Avg.</td></tr><tr><td rowspan="5">Llama-3.1-8B Instruct</td><td>Always-WM</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>Confidence</td><td>99.50</td><td>99.61</td><td>98.74</td><td>99.28</td></tr><tr><td>Consistency</td><td>80.88</td><td>84.65</td><td>83.33</td><td>82.95</td></tr><tr><td>Pessimistic</td><td>82.25</td><td>85.43</td><td>79.87</td><td>82.52</td></tr><tr><td>Dual-Frontier</td><td>55.62</td><td>61.81</td><td>55.35</td><td>57.59</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>Always-WM</td><td>100.00</td><td>100.00</td><td>100.00</td><td>100.00</td></tr><tr><td>Confidence</td><td>99.62</td><td>99.61</td><td>99.69</td><td>99.64</td></tr><tr><td>Consistency</td><td>78.25</td><td>81.30</td><td>81.45</td><td>80.33</td></tr><tr><td>Pessimistic</td><td>82.50</td><td>79.53</td><td>84.91</td><td>82.31</td></tr><tr><td>Dual-Frontier</td><td>53.62</td><td>57.28</td><td>61.64</td><td>57.51</td></tr></table>

Table 8: Selective risk (%) among admitted world-model revisions. Lower is better; Avg. is the uniform mean over the three benchmarks.
<table><tr><td>Models</td><td>Methods</td><td>BFCL v4</td><td>API-Bank</td><td>NexusRaven</td><td>Avg.</td></tr><tr><td rowspan="5">Llama-3.1-8B Instruct</td><td>Always-WM</td><td>93.12</td><td>73.43</td><td>70.13</td><td>78.89</td></tr><tr><td>Confidence</td><td>90.95</td><td>66.21</td><td>57.64</td><td>71.60</td></tr><tr><td>Consistency</td><td>89.49</td><td>61.86</td><td>51.70</td><td>67.68</td></tr><tr><td>Pessimistic</td><td>90.43</td><td>65.67</td><td>56.69</td><td>70.93</td></tr><tr><td>Dual-Frontier</td><td>9.12</td><td>33.66</td><td>41.82</td><td>28.20</td></tr><tr><td rowspan="5">Qwen3-8B</td><td>Always-WM</td><td>97.88</td><td>79.53</td><td>81.45</td><td>86.29</td></tr><tr><td>Confidence</td><td>96.99</td><td>72.53</td><td>76.03</td><td>81.85</td></tr><tr><td>Consistency</td><td>96.49</td><td>67.07</td><td>71.04</td><td>78.20</td></tr><tr><td>Pessimistic</td><td>96.97</td><td>72.52</td><td>75.56</td><td>81.68</td></tr><tr><td>Dual-Frontier</td><td>3.43</td><td>27.36</td><td>23.90</td><td>18.23</td></tr></table>

## AI USE STATEMENT

Generative AI tools were used solely for formatting checks and language polishing. The authors reviewed all resulting revisions and take full responsibility for the final manuscript.