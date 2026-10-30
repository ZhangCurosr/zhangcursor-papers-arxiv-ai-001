# Certified Access: Test-Time Compute on Conservative Physics-Structured AI Memory

Pratik Jawahar <sup>1,2∗</sup> Riccardo Maggioni <sup>3</sup> Jonas Petersen <sup>3</sup> Matteo Cozzi <sup>3</sup>

Gian-Alessandro Lombardi <sup>2</sup> Marek Gajewski <sup>2</sup> Camilla Mazzoleni <sup>3</sup>

Federico Martelli <sup>2</sup>

<sup>1</sup> CERN <sup>2</sup> Forgis AG <sup>3</sup> ETH Zurich

## Abstract

Test-time compute mechanisms, such as retries, confidence-based escalation and non-local routing, are conventionally deployed as black-box additions to trained models. They typically add forward passes or halting heads, but give no structural account of what the extra compute buys or what it costs against the model’s stability guarantees. We therefore propose a theoretical and empirical framework for test-time compute on a Latent Dynamics Model used as the memory. When an encoder’s latent state (q, p) is advanced by a damped symplectic (velocity-Verlet) step of a learned Hamiltonian, every mechanism that expands access has an exact phase-volume Jacobian, a bounded energy change and an exact latch-transport law. These certificates are conditional: the volume certificate holds for damping frozen within a step, not for the state-dependent governor, and the energy statements hold on the stated testbed potentials. We split reachability failures into escape failures, where a target inside a kinematic causal box C is trapped behind a latent barrier, and reach failures, where the target lies outside C . A Lorentz squeeze resolves escape failures by injecting bounded, re-absorbable energy. A gated canonical translation (a one-step channel across C ) resolves reach failures at det J = 1 exactly. We verify this certificate stack on a designed analytic testbed. Reaching just beyond C<sub>T</sub> with the squeeze requires energy that grows quadratically in the excess distance. The gated channel’s energy cost equals the potential difference between its endpoints, which is zero on the matched channel measured here, and so does not grow with distance in this testbed. An untrained state-replacing map annihilates phase volume and erases stored charge, while the gated channel transports it. On a learned physics-structured memory, we show calibrated compute rationing and evaluate the scaling limits of non-local routing against a physics-free baseline. The main contribution is the formalization and verification of the certificate stack, not benchmark performance.

## 1 Introduction

Deep learning models increasingly allocate inference compute adaptively. Learned halting (ACT, Graves, 2016; PonderNet, Banino et al., 2021), confidence-gated early exits (CALM, Schuster et al., 2022), learned routing (Mixture-of-Depths, Raposo et al., 2024; Mixture-of-Experts, Shazeer et al., 2017) and energy-based verification (Gladstone et al., 2025) have established the value of input-conditional escalation. However, these mechanisms generally attach a learned scalar gate to a feedforward stack. Escalation then repeats the standard forward operation and gives no structural guarantee of how the extra computation changes the model’s latent state.

We therefore define test-time compute for Latent Dynamics Models (LDMs), the class of models that structure the dynamics of an abstract latent space produced by an encoder. Suppose the queried memory is a conservative latent state $( q , p )$ , produced by the encoder and advanced by a structurepreserving symplectic integrator of a learned separable Hamiltonian $H ( q , p ) = T ( p ) + V _ { \theta } ( q )$ . This Hamiltonian describes the latent space of any encoder and is not limited to the input system. Test-time compute then becomes access to a defined phase space at inference. As in Jawahar and Pierini (2026), the kinetic term is a causal relativistic governor. In this regime the properties of the phase space define every access mechanism. We call this certified access: each mechanism carries a physical certificate.

The certificate has three parts: the Jacobian determinant det $J ,$ which certifies phase-volume conser vation; a bounded energy change, which quantifies injected energy; and a latch-transport law, which states how content stored in protected continuous-symmetry directions transforms. These certificates are theorems that follow from the symplectic structure under stated assumptions (damping fixed within a step, and the testbed potentials). They are not empirical curves fit to data, and they are not guarantees for the full learned system. We first state each certificate and verify it on a designed testbed (§3): the exactness claims (det J, the energy change and the latch-transport law) hold to machine precision, while the re-absorption and channel-hop thresholds hold to leading order. We then place the mechanisms on learned memories (§4).

## This work makes six contributions:

1. Formalization of failure modes. We partition reachability failures into two kinds. In kinematic reach failure, the target lies outside a causal box $C _ { T }$ bounded by $L _ { i } = T \varepsilon c / \sqrt { M _ { i } }$ In energetic escape failure, the target is inside $C _ { T }$ but shielded by a potential barrier. Distinct mechanisms, with different certificates, resolve the two (§2).

2. Verification of the certificate stack. We verify our theoretical predictions on an analytic testbed with dimensions 2 and 4, 5 seeds, and oracle placement. The Lorentz squeeze cures escape with bounded injection and extends reach at an energy exponential in the rapidity. The gated channel cures reach continuously with det J = 1, exact latch transport, and an energy change equal to the endpoint potential difference, zero on the matched channel measured here (§3).

3. Empirical measurement of latch transport. We show that an untrained state-replacing map has det J = 0 exactly and erases the stored Goldstone charge. The certified gated channel instead transports the charge with preserved spread. Phase volume alone does not suffice for this transport; a matched channel is required. A bounded energy change is also necessary but not sufficient for Bounded-Input Bounded-Output (BIBO) stability without coercive-component screening (§3.2.1, §3.1).

4. Calibrated compute rationing. We implement a self-calibrating compute-rationing gate on an escalatable learned memory. The gate reduces relaxation steps by 4.81 ± 0.44× at capacity kv16 with a 400-epoch budget. Its value lies in escalatability and rationing, not in a better baseline energy signal (§4.1).

5. Scaling limits of non-local routing. We verify that a direct one-hop non-local edge keeps a flat FLOP count as distance grows, unlike multi-hop diffusion. We also report that a parameter-matched, physics-free router outperforms the energy-gated version of this mechanism in both floating-point operations (FLOPs) and accuracy (§4.3).

6. Capacity and epoch-budget map of the compute dial. We map the rationing gate across capacity and training budget. The measured step reductions (9.9×, 9.5×, 6.2× across kv32, kv64, kv96) are intra-CLU rationing against a full-budget CLU baseline. Beyond kv64, the epoch budget, not a hard capacity limit, constrains performance. Gate accuracy degrades sharply under evaluation cue noise, while storage fidelity stays at 1.0 (§4.2).

## 2 Setup: A Conservative Memory and its Reachable Set

Our reference memory is the Causal Learning Unit (CLU), introduced as CHLU in Jawahar and Pierini (2026). A CLU is a latent dynamics model whose state is a phase-space point $( q , p )$ : an encoder writes the state, and a learned Hamiltonian evolves it with a damped symplectic integrator, so retrieval is relaxation under learned dynamics. Our empirical results use the $\dot { \mathrm { C L U } ^ { \flat } } \mathrm { s }$ reference training objective but measure general properties of the learned Hamiltonian.

One CLU step, the governed map, advances $( q , p )$ by a dissipative velocity-Verlet step (Martys and Mountain, 1999) with step size ε and per-step damping $\gamma \in [ 0 , 1 )$ . Velocity-Verlet is a kick– drift–kick scheme: a half-step momentum update (kick), a full position update (drift), then a second half-step kick. Here it integrates the separable Hamiltonian $\bar { H } = T ( p \bar { ) } + V _ { \theta } ( q )$ and is followed by the momentum scaling $p \mapsto ( 1 - \gamma ) p .$ . For fixed $\gamma$ this update is conformally symplectic, $J ^ { \top } \Omega J = ( 1 - \gamma ) \Omega$ , so det $J = ( 1 - \gamma ) ^ { d }$ in d dimensions. In relativistic mode the reference architecture’s kinetic energy is

$$
T ( p ) = \sqrt { c ^ { 2 } p ^ { \top } M ^ { - 1 } p + m _ { 0 } ^ { 2 } c ^ { 4 } } ,
$$

with $M = \mathrm { d i a g } ( M _ { 1 } , \dots , M _ { d } ) \succ 0$ the learned inertial mass, $m _ { 0 } > 0$ the rest mass (values in the flag table, $\operatorname { A p p . } \operatorname { A } )$ and c a hyperparameter. This kinetic mode caps velocity at $| \dot { q } _ { i } | < c / \sqrt { M _ { i } }$ . To bleed excess energy toward a target $E ^ { \star }$ , a state-dependent governor sets $\gamma _ { n } = s \operatorname { t a n h } ( \operatorname* { m a x } ( 0 , H - E ^ { \star } ) )$ ). The determinant certificate holds for the damping value frozen within a step. Because $\gamma _ { n }$ depends on the state, its derivative adds terms to the Jacobian, and the certificate does not cover the governed map.

The latent reachable set of the governed map $\Phi _ { \varepsilon , \gamma }$ over $T$ steps from $z _ { \mathrm { 0 } }$ is $R _ { T } ( z _ { 0 } ) = \{ \Phi ^ { n } ( z _ { 0 } ) : 1 \leq$ $n \leq T \}$ , with position shadow $Q _ { T } ( z _ { 0 } )$ . A mechanism gives access to a latent state only if it enlarges $Q _ { T }$ . Throughout, $\boldsymbol { z } = ( q , p ) \in \mathbb { R } ^ { d } \times \mathbb { R } ^ { d }$

The causal box follows from a bound on the drift: one drift sub-step advances each coordinate by strictly less than $\varepsilon c / \sqrt { M _ { i } }$ , for every momentum p and every potential $V _ { \theta } .$ . The drift sub-step of the kick–drift–kick map moves only the position, $q \mapsto q + \varepsilon \nabla _ { p } T ( p )$ , at whatever momentum is current at that sub-step. Differentiating $T ,$

$$
\frac { \partial T } { \partial p _ { i } } = \frac { c ^ { 2 } p _ { i } } { M _ { i } T ( p ) } , \qquad T ( p ) = \left( c ^ { 2 } \sum _ { j } p _ { j } ^ { 2 } / M _ { j } + m _ { 0 } ^ { 2 } c ^ { 4 } \right) ^ { 1 / 2 } > \frac { c | p _ { i } | } { \sqrt { M _ { i } } } ,
$$

the last inequality strict for every $p$ because $m _ { 0 } > 0$ and the remaining kinetic terms are non-negative. Hence

$$
| { \dot { q } } _ { i } | = \left| { \frac { \partial T } { \partial p _ { i } } } \right| = { \frac { c ^ { 2 } | p _ { i } | } { M _ { i } T ( p ) } } < { \frac { c ^ { 2 } | p _ { i } | } { M _ { i } } } \cdot { \frac { \sqrt { M _ { i } } } { c | p _ { i } | } } = { \frac { c } { \sqrt { M _ { i } } } } ,
$$

so a single drift obeys $| q _ { i } ^ { \prime } - q _ { i } | < \varepsilon c / \sqrt { M _ { i } }$ . The two kick sub-steps and the damping $p \mapsto ( 1 - \gamma ) p$ act on momentum only and move no position. The same bound therefore holds for the full step, uniformly in $\theta , \gamma$ , and the state. Summing over $n \leq T$ steps gives $\lvert q _ { n , i } - q _ { 0 , i } \rvert < n \varepsilon c / \sqrt { M _ { i } } \leq L _ { i }$ so the reachable set is bounded:

$$
\begin{array} { r } { Q _ { T } ( z _ { 0 } ) \ \subseteq \ C _ { T } ( q _ { 0 } ) : = \{ q : \ | q _ { i } - q _ { 0 , i } | \leq L _ { i } , \ L _ { i } = T \varepsilon c / \sqrt { M _ { i } } \} . } \end{array}
$$

The bound is energy-blind: it involves only $( \varepsilon , c , M _ { i } )$ , and neither $V _ { \theta }$ nor the magnitude of $\dot { p }$ enters. Injected momentum drives $| { \dot { q } } _ { i } |$ | toward the cap $c / \sqrt { M _ { i } }$ from below but never past it. The relative deficit is $1 - | \dot { q } _ { i } | \sqrt { M _ { i } } / c \approx m _ { 0 } ^ { 2 } c ^ { 2 } M _ { i } / ( 2 p _ { i } ^ { 2 } )$ as the momentum $| p _ { i } |$ along coordinate i grows, so the velocity saturates at the cap. In Newtonian mode $\nabla _ { p } T = M ^ { - 1 } \bar { p }$ is unbounded, no such box exists, and reach is energy-limited. This is why the Newtonian control in §3.2 crosses the box.

This setup has two failure modes. A kinematic reach failure occurs when the target $q ^ { \star } \notin C _ { T } ( q _ { 0 } ) .$ only a non-local jump solves it. An energetic escape failure occurs when $q ^ { \star } \in C _ { T } ( q _ { 0 } )$ but is shielded by a potential barrier $\Delta V _ { b }$ that the latent energy cannot cross. Any bounded energy injection that acts faster than the governor dissipates it resolves this failure.

## 3 Mechanism Certificates and Limitations

We first certify the access mechanisms and the latch, then state where the certificate stops (BIBO stability), and finally compare the mechanisms on a testbed.

To resolve escape failures, we apply a mass-weighted Lorentz squeeze $S _ { \zeta } ^ { ( M ) }$ , which injects energy to cross barriers in the spirit of basin-hopping (Wales and Doye, 1997). On the targeted coordinate pair $( q _ { i } , p _ { i } )$ , in mass-weighted coordinates $u = \sqrt { M _ { i } } q _ { i } , v = { p _ { i } } / { \sqrt { M _ { i } } }$ , the squeeze acts as the hyperbolic rotation $u ^ { \prime } = u$ cosh ζ + v sinh $\zeta , v ^ { \prime } = v$ cosh $\zeta + u$ sinh ζ, and as the identity on all other coordinates. The map is linear with determinant cos $\mathrm { i } ^ { 2 } \dot { \zeta } - \mathrm { s i n h } ^ { 2 } \dot { \zeta } = 1$ , hence exactly symplectic $( \mathrm { d e t } = 1 )$ . It amplifies velocity and displacement toward the saddle, $\delta q ^ { \prime } = \delta q \cosh \bar { \zeta } + \bar { p _ { 0 } } \sinh \zeta$ where the rapidity ζ sets the squeeze strength. For a Hamiltonian that is quadratic and isotropic in the same mass-weighted coordinates, $\begin{array} { r } { H _ { \mathrm { q } } = \frac { 1 } { 2 } \kappa \left( u ^ { 2 } + v ^ { 2 } \right) } \end{array}$ , the key step is

$$
u ^ { \prime 2 } + v ^ { \prime 2 } = ( u ^ { 2 } + v ^ { 2 } ) \cosh 2 \zeta + 2 u v \sinh 2 \zeta \leq ( u ^ { 2 } + v ^ { 2 } ) e ^ { 2 | \zeta | } ,
$$

using $2 | u v | \leq u ^ { 2 } + v ^ { 2 }$ . The injected energy therefore obeys $H ( S _ { \zeta } z ) \leq e ^ { 2 \left| \zeta \right| } H ( z )$ (full derivation in $\mathrm { A p p . } \mathrm { F } )$ , and the governor dissipates the excess in about $2 \zeta / \gamma _ { c }$ steps. This bound is proved for a Hamiltonian quadratic in position and momentum, whereas the causal box uses the relativistic kinetic energy, so the two results rest on different kinetic energies and are not a single joint statement.

To resolve reach failures, we introduce an intra-unit gated channel: a learned pair of loci $( a , b )$ with a hard gate frozen at capture. The channel applies the constant canonical translation $q \mapsto q + \Delta$ $p \mapsto p ,$ , with $\Delta = b - a$ . This translation is the flow of the linear Hamiltonian $W = \Delta \cdot p / \tau ,$ so the Jacobian is $I _ { 2 d }$ and det $J = 1$ exactly, independent of $\Delta$ and of the gate. The energy cost is the discrete jump $\Delta H _ { \mathrm { w h } } = V _ { \boldsymbol { \theta } } ( \boldsymbol { q } + \Delta ) - \dot { V _ { \boldsymbol { \theta } } } ( \boldsymbol { q } )$ , the potential difference between the endpoints, which must be tracked; matched loci give zero-energy transport.

We next derive the Jacobian when the gate may vary during the jump, $q \mapsto q + g ( q ) \Delta , p \mapsto p .$ This Jacobian is block-triangular with momentum block $I _ { d }$ and position block the rank-one update ${ \partial q ^ { \prime } } / { \partial q } = I _ { d } + \Delta \nabla g ( q ) ^ { \top }$ . By the matrix determinant lemma,

$$
\operatorname* { d e t } J = \operatorname* { d e t } \big ( I _ { d } + \Delta \nabla g ( q ) ^ { \top } \big ) = 1 + \nabla g ( q ) ^ { \top } \Delta \neq 1 .
$$

The defect is first order in the gate’s variation along the jump, so no smallness of $\Delta$ removes it (unit-test value 2.05). Freezing the gate at capture (g constant during the jump) restores det $J = 1$ exactly, so the gate must stay frozen. This is a per-branch statement (gate fixed); it does not show that the full routing policy is invertible.

The latch transport law concerns a protected direction with the invariant $Q = p ^ { \top } X q$ , where $X$ is the generator of the broken symmetry. We call Q the Goldstone charge, borrowing from field theory along the lines of Iqbal et al. (2026). The channel shifts it by exactly $\mathbf { \bar {  { Q } } } ^ { \prime } - { \cal { Q } } = p ^ { \top } { \boldsymbol { X } } \Delta$ , so the charge is preserved if and only if $\Delta \perp X ^ { \top } p$ . Because the channel maps a single phase point, it transports the latch rather than copying or erasing it. This law describes the jump only: the damping substep then scales $p ,$ and with it $Q ,$ so the charge is not preserved over a full damped retrieval.

Table 1 collects the certificate per mechanism. Both jump mechanisms must be paired with a coercive potential $V _ { \theta }$ . Deep convolutional potentials are non-coercive out-of-unit by architecture, so exits must lie inside a coercive sub-level component (§3.1).

Table 1: Certificate per mechanism. The $( 1 - \gamma ) ^ { d }$ entry holds for the damping value frozen within a step; the state-dependent governor is not covered by it.
<table><tr><td>Mechanism</td><td>Energy Injected</td><td>Volume det J</td><td>Governor Re-absorption</td><td>Latch Impact</td></tr><tr><td>Gated channel (transla- tion)</td><td> $\scriptstyle \Delta V = V ( b ) - V ( a )$  discrete jump</td><td>1 exactly</td><td>≈  $( 1 / \gamma _ { c } ) \ln ( 1 + \Delta V / E ^ { \star } )$ </td><td>Transport via shift  $p ^ { \top } X \bar { \Delta }$ </td></tr><tr><td>Gated channel (throat)</td><td>standing wave, no</td><td> $( 1 - \gamma ) ^ { d }$  (fixed</td><td>continuous</td><td>adds  $\mu ^ { 2 } { \propto } A / \ell ^ { 2 }$ </td></tr><tr><td>Squeeze  $S ^ { ( M ) }$ </td><td>jump  $\leq ( \bar { e } ^ { 2 | \zeta | } - 1 ) H$ </td><td>γ) 1 exactly</td><td>≈  $2 \zeta / \gamma _ { c }$ </td><td>preserved</td></tr></table>

## 3.1 BIBO Stability Limits

The certificate det $J = 1$ guarantees volume preservation, not bounded trajectories: an exit outside the coercive connected component of $V _ { \theta }$ can escape to infinity even when the jump is symplectic and costs zero energy. We test boundedness on a controlled non-coercive well, $\begin{array} { r } { \dot { V { } } = \frac { 1 } { 2 } k q _ { 0 } ^ { 2 } { } ^ { - } \epsilon q _ { 0 } ^ { 4 } } \end{array}$ , whose connected component spans $| q _ { 0 } | < x _ { b } = 3 . 5 3 6$ . We measure the maximum excursion radius $r ^ { * }$ at horizons $T$ and 2T (Table 2, Fig. 1).

Table 2: The BIBO battery: maximum excursion radius $r ^ { * }$ at 2T against the exit locus $b ,$ escape rates, and the admit/reject decision of each screen.
<table><tr><td>Algorithm  $( r ^ { * } \mathrm { a t } 2 T )$ </td><td> $b { = } 1 . 0$ </td><td>2.0</td><td>3.0</td><td>3.6</td><td>4.0</td><td>5.0</td></tr><tr><td>Gated channel with certificate</td><td>1.01</td><td>2.01</td><td>3.01</td><td>0.09</td><td>0.09</td><td>0.09</td></tr><tr><td>Gated channel, ablation</td><td>1.01</td><td>2.01</td><td>3.01</td><td>102.13</td><td>103.43</td><td>104.83</td></tr><tr><td>State-replacing map</td><td>1.01</td><td>2.01</td><td>3.01</td><td>102.13</td><td>103.43</td><td>104.83</td></tr><tr><td>Escape rate, certified</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Escape rate, blind</td><td>0</td><td>0</td><td>0</td><td>1.0</td><td>1.0</td><td>1.0</td></tr><tr><td>Gated channel certificate logic Energy sub-level test</td><td>ADMIT admit</td><td>ADMIT admit</td><td>ADMIT admit</td><td>REJECT reject</td><td>REJECT admit</td><td>REJECT admit</td></tr></table>

![](images/a4749db3e2c0c983ebf1884dc120a17b690bf4f873dd581db2285692e74a4bfb.jpg)  
Figure 1: The BIBO battery. Maximum excursion radius $r ^ { * } = \operatorname* { m a x } _ { t } \| q _ { t } \|$ at horizon $2 T$ , against the requested exit locus b. Exits inside the coercive component stay bounded for every arm. Beyond the coercive edge $x _ { b } = 3 . 5 4$ the unscreened arms grow as $r ^ { * } \propto T .$ , while the screened gated channel holds at $r ^ { * } = 0 . 0 9$ because it refuses the jump rather than making it safe. The state-replacing map coincides exactly with the screen-ignored ablation, so the screen, not the jump, provides boundedness. Note $b = 5 . 0 \colon$ the energy change is exactly zero and the exit still escapes.

An admissible exit at $b = 3 . 0$ costs $\Delta H = 2 . 8 8$ but stays bounded. The escaping arms grow linearly, with $r ^ { * } ( 2 T ) / r ^ { * } ( T ) \approx 1 . 9$ . These trajectories are unbounded and limited only by the terminal velocity cap. The key case is exit $b = 5 . 0 .$ . There the energy change is exactly zero $( \dot { \Delta } H = 0 . 0 )$ , so an energy-only sub-level test wrongly admits the jump, and the state escapes at once. This shows that det $J = 1$ with a zero energy change is not sufficient for BIBO stability; the required property is coercive-component membership. The blind gated channel and the state-replacing map also coincide exactly on all failures. This shows that the certificate screen, not the jump itself, provides the bound.

## 3.2 Differentiating Access Mechanisms

To compare the mechanisms, we build a controlled reach battery on a small analytic testbed that does not depend on the architecture. The testbed has a K-basin (symmetric) double well along the reach coordinate, whose barriers span the causal box $L = 2 . 5 \ : ( T = 1 0 0 , \varepsilon = 0 . 0 5 , c = \bar { 1 }$ heavy reach coordinate $M _ { 0 } = 4 . 0 )$ , and an $S O ( 2 )$ sector for the latch. All results in this section use latent dimensions 2 and 4, 5 random seeds, designed channel placement, single-CPU execution, a sharp-boundary $\gamma = 0$ rollout, and a mass band of [4.0, 0.25], which gives a 16× inertial contrast. This controlled setting isolates the exactness of the theory; learning where to place entrances is outside the current scope. We set the governor so that plain relaxation cannot escape (kinetic energy

$0 . 7 2 < \Delta V _ { b } = 1 )$ . We measure landing rates at distances $d < L , d \in \{ 0 . 8 , 1 . 6 , 2 . 4 \}$ , and $d > L$ $d \in \{ 3 . 2 , 4 . 0 , 5 . 0 \}$ (Table 3; per-distance grid in $\mathrm { A p p . ~ B ) }$  
Table 3: Reach battery summary: landing rates inside $( d { < } L )$ and beyond $( d { > } L )$ the causal box, measured det $J ,$ and diagnostic. The $( 1 - \bar { \gamma } ) ^ { d }$ entries hold for fixed γ.
<table><tr><td>Algorithm</td><td> $d { < } L$ </td><td> $d { > } L$ </td><td>Measured det J</td><td>Diagnostic</td></tr><tr><td>Plain relaxation</td><td>0</td><td>0</td><td> $( 1 - \gamma ) ^ { d }$ </td><td>Escape-blocked</td></tr><tr><td>Squeeze  $S ^ { ( M ) }$ </td><td>1</td><td>→ 0</td><td> $1 . 0 0 0 \pm 4 \mathrm { e } { - 6 }$ </td><td>Reach beyond  $C _ { T }$  needs energy  $\leq e ^ { 2 | \zeta | } H$ </td></tr><tr><td>Gated channel</td><td>1</td><td>1</td><td> $1 . 0 \mathrm { e x a c t }$ </td><td>Flat access scaling; energy change 0.0</td></tr><tr><td>Newtonian control</td><td>1</td><td>1</td><td>1.0</td><td>Confirms relativistic cap bounds flow</td></tr><tr><td>State-replacing map</td><td>1</td><td>1</td><td>0.0 exact</td><td>State overwritten; volume annihilated</td></tr><tr><td>Dense/V-throat</td><td>→ 0</td><td>0</td><td> $( 1 - \gamma ) ^ { d }$ </td><td>Modulates barrier; fails global reach</td></tr></table>

Squeeze reach follows from the launch state of the reach battery $( q = 0 ,$ , momentum $p _ { 0 }$ along the heavy coordinate), where the squeeze produces the instantaneous displacement $q _ { 0 } ^ { \prime } = ( p _ { 0 } / M _ { 0 } )$ sinh ζ, and the subsequent governed flow advances by at most the box half-width L. The squeeze therefore extends the reach bound to $[ L , L + p _ { 0 }$ sinh $\zeta / { M _ { 0 } } ]$ (shaded in Fig. 2(a)). Landing at distance d within the tolerance 0.4 requires $\zeta \geq$ arcsinh $\left( \left( d { - } 0 . 4 { - } L \right) M _ { 0 } / p _ { 0 } \right)$  (App. F). $\mathrm { A t } p _ { 0 } = 1 . 2$ this gives $\zeta \geq 2 . 0 1 0 5$ for $d = 4 . 0$ and $\zeta \geq 2$ .64 for $d = 5 . 0$ . Both exceed the swept budget $\zeta \leq 2 . 0$ , so the squeeze row drops to zero at $d \geq 4 . 0$ while $d \leq 3 . 2$ remains covered. The squeeze misses the outer distances because they need more than the swept rapidity budget, not because it cannot reach them with enough energy. Its energy grows as $e ^ { 2 | \zeta | }$ <sup>|</sup>H, exponentially in rapidity ζ and quadratically in the excess distance; the energy is not exponential in the distance itself (App. F). We validate the squeeze at several rapidities, (ζ, ratio, bound) = (0.25, 1.13, 1.65), (0.5, 1.55, 2.72), (1.0, 3.79, 7.39), (2.0, 27.5, 54.6), with det $S ^ { ( M ) } = 1 . 0 0 0 ( \pm 4 \times 1 0 ^ { - 6 } )$ throughout. The $e ^ { 2 | \zeta | }$ bound is a matched-quadratic certificate; on a quartic well the raw ratio can exceed it.

We test the reach of the gated channel on the symmetric double well, with channel $- a  + a .$ . The energy change is exactly 0.0, and the channel translates the state to $q = + 1 . 0 0 0$ with det $J = 1 . 0 0 0$ The gated channel’s energy cost is the endpoint potential difference, which is zero on this symmetric testbed, so here it does not grow with distance (Fig. 2(a)). The untrained state-replacing map $( q , p ) \mapsto ( b , p )$ lands as often as the gated channel. However, its Jacobian is block-diagonal with zero blocks, so det $J = 0$ exactly (forward-mode automatic differentiation): the map annihilates volume, is not invertible, and erases the state. It is an untrained analytic constant map, distinct from the learned decision head of §4.3. The Newtonian control reaches past the box, which confirms that the relativistic cap limits the flow. Plain relaxation is escape-blocked at all distances.

(a)  
![](images/e506a7557d18e094975f904c31ec5970dc643673b1c863d903b336b15505d797.jpg)

(b)  
![](images/e1bd209a913015539e16b73748fc8b3a813886d962db0888536adb488c5cc5a1.jpg)  
Figure 2: Verification of the certificate stack on the designed testbed (Sec. 3.2). (a) Reach: landing rate against basin distance d on a symmetric double well (dim 2, 5 seeds, $L = 2 . 5 )$ . (b) The certificate in force: outgoing against incoming Goldstone charge for 16 states drawn from the capture ball. Dashed line in (a): causal box L; grey band: crossover bracket. In (b), coset-tangent and across-coset are gated-channel jumps; the dashed line is $Q _ { \mathrm { o u t } } = Q _ { \mathrm { i n } }$

## 3.2.1 The Certificate’s Consequence: State Erasure vs. Transport

To measure the consequence of the det J certificates, we track 16 states drawn uniformly from the capture ball $( \mathrm { s t d } ( Q _ { \mathrm { i n } } ) = 0 . 0 8 0 3 )$ and compare the channel with an untrained state-replacing map that overwrites the state (Table 4).

The channel’s transport guarantee holds within a branch (gate fixed), where its canonical translation is injective. It shifts every incoming charge by the constant $p ^ { \top } X \Delta$ (max error $\leq 1 . 2 \times 1 0 ^ { - 7 } )$ and preserves the spread $\left( \mathrm { s t d } ( Q _ { \mathrm { o u t } } \right) ^ { - } = 0 . \bar { 0 } 8 0 3 , \mathrm { F i g . ~ } 2 ( \mathrm { b } ) )$ . The pre-jump state is recoverable as $q _ { \mathrm { i n } } = q _ { \mathrm { o u t } } - \Delta$ (max error $2 . 2 \times 1 0 ^ { - 8 } )$ . We do not show that the full routing policy, which chooses between branches, is invertible. We observe a zero shift $\Delta Q = 0 . 0$ (predicted precision $7 . 5 \times 1 0 ^ { - 1 0 } )$ for a coset-tangent channel and an across-coset shift of $\dot { \Delta } Q = \dot { 0 . 2 } 5 0 0 = \dot { p } ^ { \top } X \Delta$ . The matched channel, a coset-tangent $\Delta$ that also meets the energy constraint, guarantees stable transit.

Table 4: Latch transport on 16 states from the capture ball.
<table><tr><td>Mechanism</td><td>det J</td><td>∆Q mean</td><td> $\Delta Q \mathrm { s t d }$ </td><td>Predicted  $p ^ { \top } X \Delta$ </td><td>Max Error</td><td> $\mathrm { s t d } ( Q _ { \mathrm { o u t } } )$ </td></tr><tr><td>Gated channel (coset-tangent)</td><td>1.0000</td><td>-0.0000</td><td>0.0000</td><td>0.0000</td><td> $1 . 2 \mathrm { e } - 7$ </td><td>0.0803</td></tr><tr><td>Gated channel (across-coset)</td><td>1.0000</td><td>0.2500</td><td>0.0000</td><td>0.2500</td><td>0.0</td><td>0.0803</td></tr><tr><td>Random shift</td><td>1.0000</td><td>0.0972</td><td>0.2465</td><td>N/A</td><td></td><td>0.2793</td></tr><tr><td>State-replacing map</td><td>0.0000</td><td>0.0379</td><td>0.0803</td><td>N/A</td><td>一</td><td>0.0</td></tr></table>

On this designed testbed, the det $J = 0$ state-replacing map instead sends all 16 states to one bit-identical charge $( \mathrm { s t d } ( Q _ { \mathrm { o u t } } ) = 0 )$ and erases the coset content irreversibly. Volume preservation alone is not the latch certificate: a random shift also has det $J = 1$ but scrambles the spread and erases the latch, whereas the matched channel preserves it.

## 4 Translating Mechanisms to Learned Memories

The testbed potentials are designed, but a learned memory’s potential is not. We therefore test the mechanisms on trained associative memories and report where baselines outperform them. We first calibrate a compute-rationing gate, then map it across capacity and training budget, and finally compare non-local routing with baselines.

## 4.1 Calibrated Compute-Rationing

We implement a self-calibrating gate on a trained CLU associative memory for MQAR key-value recall (vocab 256, capacity $\mathrm { k v } { \in } \left. 1 6 , 2 4 , 3 2 \right.$ , 5 seeds). At write time the memory probes itself with jittered copies of its stored keys and fits a per-model head that maps the relaxation residual to an error probability. A Learn-then-Test (LTT; Angelopoulos et al., 2025) wrapper sets exit thresholds on a relaxation ladder (Geifman and El-Yaniv, 2017). The wrapper held on all 30 cells, with coverage $0 . 6 4 7 { \scriptstyle \pm 0 . 0 6 3 }$ at target risk $\varepsilon = 0 . 0 5$ . This rests on an exchangeability assumption between write-time probes and deployment queries, which our jittered protocol only approximates. The residual is a valid gate feature but adds at most +0.024 AUROC over a standard readout margin, and is therefore not the right exit threshold.

Rationing helps most at low capacity: at an under-converged 400-epoch budget, the gate reduces relaxation steps by $4 . 8 1 \pm 0 . 4 4 \times$ at kv16 relative to an always-full budget, with higher accuracy. The gain shrinks as capacity grows: $1 . 5 7 \pm 0 . 0 7 \times$ at kv24 and $1 . \dot { 1 } 4 \pm 0 . 0 6 \dot { \times }$ at kv32. At full convergence (2000 epochs) the gate matches the full-budget baseline rather than exceeding it. The gate’s value is therefore compute rationing on clean retrieval, not general accuracy gains.

## 4.2 Capacity and Training-Budget Map

We map the gate across capacity and training budget on a 198-job grid (Fig. 3, Table 5). Storage fidelity converges to near-perfect levels with training, so the initial underperformance was an undertraining artifact. Beyond kv64 the epoch budget, not a hard capacity limit, constrains performance; at 4000 epochs smaller cells over-train, and accuracy falls from 1.00 to 0.89. The required number of epochs grows with capacity: kv32 saturates by 1000 epochs, while kv96 needs 4000 epochs to reach an accuracy of 0.975. The gate’s step reductions (9.9×, 9.5×, 6.2× at kv32, kv64, kv96) are intra-CLU rationing against a full-budget CLU baseline. Under evaluation cue noise $( \sigma \in \{ 0 . 3 , 0 . 6 \} )$ the gate degrades sharply: at $\sigma = 0 . 6$ on kv32 its accuracy is 0.36, while CLU storage fidelity stays at 1.0. The governed relaxation over-commits to the corrupted cue, and we record this noise wall as an open problem.

![](images/7a1e6574e5772bcd864459d45c7e3705eaddd980c03f9325dfc60ba928001bc2.jpg)

(b)  
![](images/3012cc0c0837f420cf66009504cce9662dc3f0c2e722db6530ae9facc27e3277.jpg)  
Figure 3: Epoch-scaling frontier: CLU-EBM storage fidelity and gated accuracy against training budget for kv∈ {32, 64, 96, 128}. Required epochs scale with capacity rather than saturating at a fixed ceiling, and kv32 over-trains between 2000 and 4000 epochs. Mean over 3 seeds (42–44).

Table 5: Gate across capacity and training budget.
<table><tr><td>Cell (N,kv)</td><td>Epochs</td><td> $\mathrm { C L U F i d e l i t y }$ </td><td> $\mathbf { G a t e _ { A c c } }$ </td><td>Intra-CLU Step-Reduction</td></tr><tr><td>N128/kv32</td><td>500</td><td> $0 . 7 6 \pm 0 . 0 9$ </td><td> $0 . 3 1 \pm 0 . 0 4$ </td><td>1.2×</td></tr><tr><td>N128/kv32</td><td>2000</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td>9.9×</td></tr><tr><td>N256/kv64</td><td>500</td><td> $0 . 4 3 \pm 0 . 0 5$ </td><td> $0 . 0 6 \pm 0 . 0 2$ </td><td>1.0×</td></tr><tr><td>N256/kv64</td><td>2000</td><td> $1 . 0 0 \pm 0 . 0 0$ </td><td> $0 . 9 9 \pm 0 . 0 0$ </td><td>9.5×</td></tr><tr><td>N384/kv96</td><td>500</td><td> $0 . 4 0 \pm 0 . 0 3$ </td><td> $0 . 0 2 \pm 0 . 0 1$ </td><td>1.0×</td></tr><tr><td>N384/kv96</td><td>2000</td><td> $0 . 9 7 \pm 0 . 0 1$ </td><td> $0 . 9 1 \pm 0 . 0 2$ </td><td>6.2×</td></tr></table>

## 4.3 Non-Local Routing

On a lattice routing task (Table 6), a one-hop non-local edge keeps a flat FLOP count as distance grows, unlike multi-hop diffusion. However, a parameter-matched, physics-free router outperforms the energy-gated edge in both FLOPs and accuracy. A learned decision router may select whether to use the edge, but the data still move through the certified det J = 1 channel.

Table 6: Routing: local and distant accuracy (L/D) and floating-point operations per query over 5 seeds.
<table><tr><td>Algorithm</td><td>N=4 Acc (L/D)</td><td> $N { = } 4 \ \mathrm { F L O P }$ </td><td>N=8 Acc (L/D)</td><td> $N { = } 8 \mathrm { F L O P }$ </td></tr><tr><td>Local only</td><td>0.500 (1.00/0.00)</td><td>5.88e7</td><td>0.500 (1.00/0.00)</td><td>5.88e7</td></tr><tr><td>Gated energy</td><td> $0 . 8 8 7 \pm 0 . 1 3 9 ( 0 . 9 0 / 0 . 8 8 )$ </td><td>1.18e8</td><td> $0 . 7 1 5 \pm 0 . 1 7 2 ( 0 . 8 2 / 0 . 6 1 )$ </td><td>1.18e8</td></tr><tr><td>Dense network</td><td> $0 . 5 0 0 \left( 0 . 0 0 / 1 . 0 0 \right)$ </td><td>1.18e8</td><td> $0 . 4 4 8 \pm 0 . 0 7 0 ( 0 . 0 0 / 0 . 9 0 )$ </td><td>1.18e8</td></tr><tr><td>Diffusion chain</td><td> $0 . 6 5 2 \pm 0 . 1 5 9 ( 0 . 9 0 / 0 . 4 1 )$ </td><td>1.76e8</td><td> $0 . 5 4 8 \pm 0 . 1 3 8 ( 0 . 8 2 / 0 . 2 8 )$ </td><td>2.94e8</td></tr><tr><td>Physics-free router</td><td> $1 . 0 0 0 \pm 0 . 0 0 0 ( 1 . 0 0 / 1 . 0 0 )$ </td><td>8.81e7</td><td> $0 . 9 4 8 \pm 0 . 0 7 0 ( 1 . 0 0 / 0 . 9 0 )$ </td><td>8.81e7</td></tr></table>

## 5 Discussion

For a conservative memory built as an LDM, test-time compute is certified access: under the stated assumptions, the dynamics fix the volume, energy and latch consequences of every access mechanism. A squeeze resolves escape with bounded injection, at a cost exponential in rapidity and quadratic in the excess distance. A gated channel reaches globally at det $J = 1$ exactly, at an energy cost equal to the endpoint potential difference (zero on the matched channel here, so independent of distance in this testbed), with exact latch transport at the jump. These guarantees are conditional: the volume certificate assumes damping frozen within a step and does not cover the state-dependent governor, the latch law holds at the jump only, and the channel is injective within a branch. They also do not bound trajectories: BIBO stability requires coercive-component screening (§3.1). On learned memories, the mechanisms ration compute on clean retrieval, while a physics-free router is stronger for non-local routing. The main contribution is the theoretical formalization of the certificate stack and the empirical mapping of its limits; we do not claim a general leaderboard advantage. App. C records the main negative results.

With a quadratic payload channel and a fixed read budget, this implementation does not achieve exact value fidelity, amplitude-independent address hold and amplitude-independent read latency at once; we observe this as a design trade-off of the implementation, not a proved incompatibility. The implementation gives up one of these, so a faded memory requires more integration steps to read. Future work includes learned entrance-steering for the gated channels, retry acceptance as a certified kernel (App. E), and tests on learned latent potentials at scale.

## References

Anastasios N. Angelopoulos, Stephen Bates, Emmanuel J. Candès, Michael I. Jordan, and Lihua Lei. Learn then test: Calibrating predictive algorithms to achieve risk control. The Annals of Applied Statistics, 19(2), June 2025. ISSN 1932-6157. doi: 10.1214/24-AOAS1998. URL https://projecteuclid. org/journals/annals-of-applied-statistics/volume-19/issue-2/ Learn-then-test--Calibrating-predictive-algorithms-to-achieve-risk/10. 1214/24-AOAS1998.full.

Andrea Banino, Jan Balaguer, and Charles Blundell. PonderNet: Learning to Ponder, September 2021. URL http://arxiv.org/abs/2107.05407. arXiv:2107.05407 [cs.LG].

Simon Duane, A.D. Kennedy, Brian J. Pendleton, and Duncan Roweth. Hybrid Monte Carlo. Physics Letters B, 195(2):216–222, September 1987. ISSN 03702693. doi: 10.1016/0370-2693(87) 91197-X. URL https://linkinghub.elsevier.com/retrieve/pii/037026938791197X.

Yonatan Geifman and Ran El-Yaniv. Selective Classification for Deep Neural Networks, June 2017. URL http://arxiv.org/abs/1705.08500. arXiv:1705.08500 [cs.LG].

Alexi Gladstone, Ganesh Nanduru, Md Mofijul Islam, Peixuan Han, Hyeonjeong Ha, Aman Chadha, Yilun Du, Heng Ji, Jundong Li, and Tariq Iqbal. Energy-Based Transformers are Scalable Learners and Thinkers, July 2025. URL http://arxiv.org/abs/2507.02092. arXiv:2507.02092 [cs.LG].

Alex Graves. Adaptive Computation Time for Recurrent Neural Networks, March 2016. URL http://arxiv.org/abs/1603.08983. arXiv:1603.08983 [cs.NE].

Nabil Iqbal, T. Anderson Keller, Yue Song, Takeru Miyato, and Max Welling. Spontaneous symmetry breaking and Goldstone modes for deep information propagation, 2026. URL http://arxiv. org/abs/2605.14685. arXiv:2605.14685.

Pratik Jawahar and Maurizio Pierini. CHLU: The Causal Hamiltonian Learning Unit as a Symplectic Primitive for Deep Learning, March 2026. URL http://arxiv.org/abs/2603.01768. arXiv:2603.01768 [cs.LG].

Nicos S Martys and Raymond D Mountain. Velocity verlet algorithm for dissipative-particledynamics-based models of suspensions. Physical Review E, 59(3):3733, 1999.

Radford M. Neal. MCMC Using Hamiltonian Dynamics. In Handbook of Markov Chain Monte Carlo, pages 113–162. Chapman and Hall/CRC, New York, 1 edition, May 2011. ISBN 978- 0-429-13850-8. doi: 10.1201/b10905-6. URL https://www.taylorfrancis.com/books/ 9780429138508/chapters/10.1201/b10905-6.

David Raposo, Sam Ritter, Blake Richards, Timothy Lillicrap, Peter Conway Humphreys, and Adam Santoro. Mixture-of-Depths: Dynamically allocating compute in transformer-based language models, April 2024. URL http://arxiv.org/abs/2404.02258. arXiv:2404.02258 [cs.LG].

Gareth O. Roberts and Richard L. Tweedie. Exponential Convergence of Langevin Distributions and Their Discrete Approximations. Bernoulli, 2(4):341, December 1996. ISSN 13507265. doi: 10.2307/3318418. URL https://www.jstor.org/stable/3318418?origin=crossref.

Tal Schuster, Adam Fisch, Jai Gupta, Mostafa Dehghani, Dara Bahri, Vinh Tran, Yi Tay, and Donald Metzler. Confident Adaptive Language Modeling. In Advances in Neural Information Processing Systems 35, pages 17456–17472, New Orleans, Louisiana, USA, 2022. Neural Information Processing Systems Foundation, Inc. (NeurIPS). ISBN 978-1-7138-7108-8. doi: 10.52202/068431-1269. URL http://www.proceedings.com/068431-1269.html.

Noam Shazeer, Azalia Mirhoseini, Krzysztof Maziarz, Andy Davis, Quoc Le, Geoffrey Hinton, and Jeff Dean. Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer, January 2017. URL http://arxiv.org/abs/1701.06538. arXiv:1701.06538 [cs.LG].

David J. Wales and Jonathan P. K. Doye. Global Optimization by Basin-Hopping and the Lowest Energy Structures of Lennard-Jones Clusters Containing up to 110 Atoms. The Journal ofPhysical Chemistry A, 101(28):5111–5116, July 1997. ISSN 1089-5639, 1520-5215. doi: 10.1021/ jp970984n. URL https://pubs.acs.org/doi/10.1021/jp970984n.

## A Flag-Provenance Tables

All computational results use the non-default configurations listed below. We ran all computational experiments on standard laptop CPUs.

## A.1 Certificate Stack Configuration

Flag Value   
JAX 0.9.0   
Kinetic modes relativistic (reach, gated channel, throat); newtonian\_learned (Newtonian control)   
Mass band [4.0, 0.25] yielding $M _ { \mathrm { e f f } , 0 } = 4 . 0 $ , 16× contrast   
c / rest mass / m<sub>0</sub> 1.0 / 1.0 / 1.0   
Time step dt / Horizon T 0.05 / 100 yielding box $L = T \varepsilon c / \sqrt { M _ { 0 } } = 2 . 5 , v _ { \mathrm { m a x , 0 } } = 0 . 5$   
Damping γ reach rollout γ = 0 (sharp box)   
Rapidity ζ grid $[ 0 , 0 . 1 , 0 . 2 , 0 . 3 , 0 . 4 , 0 . 6 , 0 . 8 , 1 . 0 , 1 . 5 , 2 . 0 ]$   
Basin geometry double well along coord 0, barrier ∆V<sub>b</sub> = 1.0, d ∈ {0.8, 1.6, 2.4, 3.2, 4.0, 5.0}   
Initialization q<sub>0</sub> = 0, p<sub>0</sub> = (1.2, 0, . . . ) + 0.02 N (kinetic energy below barrier)   
Landing criterion min<sub>traj</sub> |q<sub>0</sub> − d| < 0.4   
Seeds reach {0, 1, 2, 3, 4}; latch default\_rng(0)   
Dimensions 2 and 4

## A.2 Calibrated Gate Configuration

<table><tr><td>Flag</td><td>Value</td></tr><tr><td>Task</td><td>MQAR CLU-EBM, vocab 256, capacity kv∈ {16, 24, 32}</td></tr><tr><td>Seeds</td><td>5 (base 42); 90 models</td></tr><tr><td>Write configuration</td><td>kinetic relativistic, potential mlp, hidden 128, epochs 400, lr 1e—3, batch 16, k_steps 50, buffer 128, friction 0.3, temperature 0.3, input noise σ 0.05</td></tr><tr><td>Gate configuration</td><td>calib features r_margin, learned τ at write time,  $p _ { \mathrm { e x i t } } ~ = ~ 0 . 5 ;$  cost checkpoints 300/1200/2100/3000</td></tr><tr><td>Learn-then-Test</td><td>fixed-sequence, exact binomial p-values, targets ε ∈ {0.05, 0.1}</td></tr><tr><td>Hopfield baseline</td><td>Platt-calibrated logit margin, identically probed</td></tr></table>

## A.3 Routing Configuration

<table><tr><td>Flag</td><td>Value</td></tr><tr><td>Lattice and task</td><td>distance N ∈ {4, 8}, embed_dim 12, vocab 128</td></tr><tr><td>Write configuration</td><td>kinetic relativistic, potential mlp (coercive), hidden 128, epochs 400, lr 1e—3</td></tr><tr><td>Retrieval routing</td><td>dt 0.05, relax_steps 250, governor_sensitivity 0.95</td></tr><tr><td>Router MLP</td><td>hidden 32 (449 params), epochs 300, lr 3e—3</td></tr><tr><td>Workload mixes</td><td>[[0.5, 0.5], [0.8, 0.2], [0.95, 0.05]]</td></tr></table>

## A.4 Regime Map Configuration

<table><tr><td>Flag</td><td>Value</td></tr><tr><td>Training epochs</td><td>swept across {500, 1000, 2000, 4000}</td></tr><tr><td>Kinetic and potential</td><td>relativistic / mlp (coercive)</td></tr><tr><td>Architecture</td><td>embed_dim 16 (CLU dim 32) / hidden 128 / vocab 256</td></tr><tr><td>Retrieval</td><td>relax_steps 300, governor_sensitivity 0.95, calib r_margin</td></tr><tr><td>Seeds</td><td>{42, 43, 44, 45, 46}</td></tr></table>

## A.5 Latch Transport and BIBO Battery Configuration

<table><tr><td>Flag</td><td>Value</td></tr><tr><td>Training</td><td>none; analytic potentials, oracle channel placement</td></tr><tr><td>Kinetic mode</td><td>relativistic (reach, gated channel, throat, BIBO)</td></tr><tr><td>Damping γ</td><td>reach 0 (sharp box); BIBO 0.02 (bounded arms must settle)</td></tr><tr><td>BIBO potential</td><td>k = 1.0, € = 0.02 yielding  $\begin{array} { r } { x _ { b } = 3 . 5 3 5 5 , } \end{array}$  barrier  $V _ { b } = 3 . 1 2 5$ </td></tr><tr><td>BIBO parameters</td><td>exits  $b \in \{ 1 . 0 , 2 . 0 , 3 . 0 , 3 . 6 , 4 . 0 , 5 . 0 \}$  ; horizon  $T = 2 0 0 0$ </td></tr><tr><td>Latch transport</td><td>16 incoming states, ball radius  $0 . 3 < \rho = 0 . 3 5$ </td></tr><tr><td>Seeds</td><td>reach/BIBO {0, 1, 2, 3, 4}; latch cloud default_rng(0)</td></tr></table>

Two configuration axes most affect comparative performance: damping and training epochs. The calibrated-gate evaluation uses $\gamma = 0 . 3$ and 400 epochs, an under-converged regime. The regime map instead sweeps up to 4000 epochs to isolate converged behavior.

## B Reach Battery Full Grid

We use 5 seeds and causal box $L = 2 . 5$

<table><tr><td>Algorithm</td><td>d=0.8</td><td>1.6</td><td>2.4</td><td>3.2</td><td>4.0</td><td>5.0</td></tr><tr><td>Plain relaxation</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0</td></tr><tr><td>Squeeze  $S ^ { ( M ) }$ </td><td>1</td><td>1</td><td>1</td><td>1</td><td>0</td><td>0</td></tr><tr><td>Gated channel</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Newtonian control</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>State-replacing map</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td><td>1</td></tr><tr><td>Dense/V-throat</td><td>1</td><td>0.8</td><td>0</td><td>0</td><td>0</td><td>0</td></tr></table>

## C Documented Negative Results

We record the main negative findings to mark the limits of our mechanisms. First, squeeze retries recover no performance at single-unit scale when selecting among already reachable attractors. The squeeze therefore addresses kinematic reach failure, not local selection. Second, the compute dial does not give robustness to corrupted cues. Gate accuracy degrades sharply under evaluation cue noise while storage fidelity stays at 1.0, so the governed relaxation over-commits to the corrupted cue rather than failing to store the pattern. Noise-aware thresholds, longer relaxation budgets and denoising initialization are open work. Finally, the raw energy signal adds negligible predictive value over a standard readout margin. Our framework therefore relies on the calibrated rationing mechanism itself, not on a superior physics signal.

## D Analytic Verification Protocols

We ran all verifications in Python at standard numerical precision on CPU. The causal cap of the reach box matches its prediction to a relative error of $2 . 0 \times 1 0 ^ { \frac { \cdot } { - 1 2 } }$ . The Jacobian determinant matches 1.000, and its state-dependent deviations match the analytical prediction to $2 . 2 \times 1 0 ^ { - 1 6 }$ . The Goldstone latch transit error stays below $1 . 8 \times 1 0 ^ { - 1 5 }$ . The governor re-absorption matches the predicted exponential injections at all swept rapidities.

## E Certified Markov Kernels: Derivations and Design Rules

To obtain a stationarity certificate for test-time retries, we derive four operating constraints, which are also the design rules of the MCMC implementation. First, the symmetric Lorentz squeeze is exactly volume-preserving (det $J = 1 )$ , so it needs no Jacobian correction in the acceptance ratio. A retry that proposes a sign-symmetrized squeeze and accepts with min $( 1 , e ^ { - \Delta H / T } )$ forms a detailed-balance kernel for $e ^ { - \bar { H } / T }$ (Duane et al., 1987). The state-dependent governor, however, is non-volume-preserving dissipation. The deployed sequence is therefore Metropolis-within-annealing: it converges to a colder maximum-a-posteriori state and does not act as a Gibbs sampler, and we do not claim stationarity for the governed composite.

Second, the orbit of the squeeze family is a single hyperbola, so the squeeze alone is reducible and not ergodic. Ergodicity requires momentum refreshment: the squeeze must be layered with a Metropolis-Adjusted Langevin Algorithm (MALA) step (Roberts and Tweedie, 1996) to mix across the state space (Neal, 2011).

Third, the Langevin noise scale must be calibrated with the discrete fluctuation-dissipation relation, $\sigma _ { i } ^ { \star } = \sqrt { M _ { \mathrm { e f f } , i } T \gamma ( 2 - \gamma ) }$ , to prevent finite-budget shadow bias. This calibration reduces the finitebudget shadow-bias error from 0.0995 to 0.0065. In relativistic kinetic modes the target distribution follows Maxwell-Jüttner statistics. Fixed-covariance Gaussian refreshes are therefore invalid, and only the Metropolis-Hastings corrections maintain stationarity.

Finally, the Gibbs measure is flat along Goldstone cosets, so any coset-tangent proposal has zero energy difference and is always accepted. These proposals generate an unbiased random walk that diffuses and erodes the stored register at a rate proportional to the proposal-scale variance. Projecting proposals orthogonal to the coset tangent suppresses this drift, and is required to prevent this topological degradation. This shows that without projection, even a certified MCMC retry progressively erases stored memory contents.

## F Derivation of the Bounded-Injection Certificate

This appendix gives the full derivation of the squeeze certificate $H ( S _ { \zeta } z ) \leq e ^ { 2 \left| \zeta \right| } H ( z )$ and of the reach bracket used in $\ S 3 . 2 ;$ the causal box and the hard-gate Jacobian are derived in $\ S 2$ and $\ S 3$ The causal box uses the relativistic kinetic energy, while the squeeze energy bound is proved for a Hamiltonian quadratic in position and momentum (the matched quadratic). The two results therefore rest on different kinetic energies and are not a single joint statement.

To define the squeeze map on the targeted coordinate pair $( q _ { i } , p _ { i } )$ , we pass to mass-weighted coordinates $u = \sqrt { M _ { i } } q _ { i } , v = p _ { i } / \sqrt { M _ { i } }$ and let the mass-weighted squeeze $S _ { \zeta } ^ { ( { \cal M } ) }$ act as the hyperbolic rotation

$$
u ^ { \prime } = u \cosh \zeta + v \sinh \zeta , \qquad v ^ { \prime } = v \cosh \zeta + u \sinh \zeta ,
$$

and as the identity on all other coordinates. In the original variables, $q _ { i } ^ { \prime } = q _ { i }$ cosh $\zeta + ( p _ { i } / M _ { i } )$ sinh $\zeta$ and $p _ { i } ^ { \prime } = p _ { i } \cosh \zeta + M _ { i } q _ { i }$ sinh ζ. The map is linear with determinant cos $\begin{array} { r } { \mathrm { { 1 } } ^ { 2 } \zeta - \sinh ^ { 2 } \zeta = 1 } \end{array}$ on the active pair and 1 elsewhere, hence exactly symplectic. This is the det $S ^ { ( M ) } = 1$ certificate.

To derive the displacement law and the reach bracket, we apply the squeeze at the launch state of the reach battery $( q = 0 ;$ , momentum $p _ { 0 }$ along the heavy coordinate). It produces the instantaneous displacement $q _ { 0 } ^ { \prime } = \left( p _ { 0 } / M _ { 0 } \right)$ sinh ζ, and the subsequent governed flow advances by at most the box half-width $\overline { { L \left( \ S 2 \right) } }$ . The squeeze-extended reach is therefore contained in the bracket $[ L , L +$ $p _ { 0 }$ sinh $\zeta / M _ { 0 } $ ] of Figure 2. Landing at distance d within the tolerance 0.4 of the flag table requires $( p _ { 0 } / M _ { 0 } ) \sinh \zeta \geq d - 0 . 4 - L , \mathrm { i . e }$

$$
\zeta \geq \mathrm { a r c s i n h } \big ( ( d - 0 . 4 - L ) M _ { 0 } / p _ { 0 } \big ) ,
$$

which at $L = 2 . 5 , p _ { 0 } = 1 . 2 , M _ { 0 } = 4 . 0$ evaluates to $\zeta \geq 2 . 0 1 0 5$ for $d = 4 . 0$ and $\zeta \geq 2 . 6 4$ for $d = 5 . 0$ . Both exceed the swept budget $\zeta \leq 2 . 0$ . This is why the squeeze row of the reach battery drops to zero at d $\mathrm { : \geq 4 . 0 }$ while $d \leq 3 . 2$ remains covered.

The energy-bound certificate is a matched-quadratic statement: on the active pair the quadratic Hamiltonian is isotropic in the same mass-weighted coordinates, $H _ { \mathrm { q } } = { \textstyle \frac { 1 } { 2 } } \kappa ( u ^ { 2 } + v ^ { 2 } )$ with $\kappa > 0$

Under the squeeze,

$$
u ^ { \prime 2 } + v ^ { \prime 2 } = ( u ^ { 2 } + v ^ { 2 } )
$$

$$
2 \zeta +
$$

$$
\leq \ ( u ^ { 2 } + v ^ { 2 } ) \big ( \cosh 2 \zeta + | \sinh 2 \zeta | \big ) = \big ( u ^ { 2 } + v ^ { 2 } \big ) e ^ { 2 | \zeta | }
$$

using $2 | u v | \leq u ^ { 2 } + v ^ { 2 }$ . Hence $H ( S _ { \zeta } z ) \leq e ^ { 2 | \zeta | } H ( z )$ , with equality exactly on the diagonal $u = \pm v$ (sign matched to that of $\zeta )$ . The injected energy obeys $\Delta H \le ( e ^ { 2 | \zeta | } - 1 ) H$ , the form quoted in the certificate table. Spectator coordinates only tighten the bound: they contribute the same non-negative energy to both sides and pull the ratio toward one. The text uses two consequences. First, at the launch state $u = 0$ the ratio equals cosh $2 \zeta$ exactly, so the measured ratios must lie between cosh $2 \zeta$ and the bound $e ^ { 2 | \zeta | }$ , as they do. Second, the bound is attained only on the diagonal, so generic states sit strictly below it.

We next show why the energy is exponential in rapidity and quadratic in the excess distance. Writing $\delta = ( p _ { 0 } / M _ { 0 } )$ sinh $\zeta$ for the displacement beyond $L ,$ , the launch-state ratio is

$$
\cosh 2 \zeta = 1 + 2 \sinh ^ { 2 } \zeta = 1 + 2 ( M _ { 0 } \delta / p _ { 0 } ) ^ { 2 } ,
$$

exactly quadratic in $\delta .$ The certificate $e ^ { 2 | \zeta | } = ( \cosh \zeta + | \sinh \zeta | ) ^ { 2 }$ is exponential in the rapidity and also quadratic in $\delta$ at large ζ. This is the scaling stated in the abstract; the energy is not exponential in the distance itself.

The scope of the inequality is the matched quadratic H, for which it is proved. On a quartic well the potential term transforms with higher powers of cosh $\zeta ,$ so the raw ratio can exceed $e ^ { 2 \bar { | \zeta | } }$ , as reported.