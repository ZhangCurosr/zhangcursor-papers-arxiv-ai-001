# Softness is Something You also Need! Why Compliance Helps Control: Information Metrics of a Passive Walker with Compliant Ankle

Fabio Bonsignorio<sup>1</sup>, Elisabetta Barletta, Sorin Dragomir, Francesco Esposito<sup>2</sup> and Enrica Zereik<sup>3</sup>

Abstract— A great gap in adaptivity, flexibility, and dexterity still exists between intelligent robots, whether they are made of rigid or (partially) compliant components, and their naturally evolved animal and vegetal counterparts. One reason could be of ‘paradigm’ nature. On the one hand, most robots (both rigid and soft) are still designed by following a ‘Cartesian’ approach, with top-down control and a clear mind-body separation. On the other hand, the principles of organization of animals and plants rely on morphological computation (which can be regarded as a partial or total outsourcing of information processing to the agent’s dynamics) and emergence of fit behaviors from loosely coupled - sometimes huge and complex - networks of embodied agents. Designing artificial agents on the basis of the principles that we observe in biology is not an easy task. Modeling and control require radically new approaches. In this paper we briefly recall and explain some seminal mathematical results linking information metrics and dynamics of the agents, already published years ago; we integrate them with some recent methods developed by us and by the community, we highlight some brief remarks about their relevance and we hint to our current and near future work towards finding concrete and efficient ways for the modeling and control of completely or partially compliant artificial agents and for a better understanding of physical intelligence in nature. We present a technical note summarizing phase–space volume calculations for inverted-pendulum and passive-dynamic walking models, following our proposed framework. Starting from the McGeer compass gait, we introduce knee–ankle compliance, foot–rocker effects, and a Ruina-style compass gait with distributed leg inertia. Standard heel–strike maps are derived and used in the estimation of viable phase–space volumes $W ( \gamma ) ,$ based on a state–dependent Monte–Carlo and Reproducing Kernel Hilbert Spaces methods. We show how the methods based on RKHS allow deeper and more subtle analysis.

Index Terms— New methods, Physical AI, Morphological Computation, Information theory, Dynamical Systems

## I. BACKGROUND

This work builds on [1], utilizing the findings from [2]. In [1] it is argued that there are good reasons to believe that ’intelligence’ (and ’meaning’) are ’emerging’ traits arising from the evolution of loosely coupled networks of intelligent ’embodied’ and ’situated’ agents. The author, in line with the approach proposed by Rolf Pfeifer in “How the body shapes the way we think” [3], elaborates a preliminary quantitative comprehensive theory that aims to enable us to predict the interaction between the agents’ ‘morphology’, or mechanical structure and dynamics, and the emergence of ‘intelligence and ‘meaning’ in physical systems.

Specifically, a number of relations are derived with a objective to clarify:

1) The relationship between an embodied agent’s information/computing capability and its dynamics (morphological computation)

2) The behavior of computing and information capacities in a multi-body agent system

3) How networks of embodied agents give rise to ’intelli gence’ and ’meaning

This work has a narrower scope. Our objective is to elucidate the significance of softness, or compliance, in robotics and natural embodied agents and provide quantitative guidance for hybrid hard and soft bodied robots.

## II. MIND–BODY NEXUS EQUATIONS

The main modeling issue is, in that line of thought, to link the information processing metrics of the ‘mind’ of the agent to those of the body’s system dynamics.

In [1] and [4] the following expressions were derived:

$$
K ( X ) { \stackrel { + } { \leq } } \ln { \frac { \Delta W _ { c l o s e d } } { \Delta W _ { \mathrm { o p e n } } ^ { \mathrm { m a x } } } }\tag{1}
$$

$$
\Delta H _ { N } + \sum _ { i = 1 } ^ { n } \Delta H _ { i } - \Delta I \leq I ( X ; C )\tag{2}
$$

$$
K ( X ) = \Delta H _ { N } + \sum _ { i = 1 } ^ { n } \Delta H _ { i } - \Delta I\tag{3}
$$

$$
\Delta H _ { N } = \ln \frac { \Delta \Omega _ { c l o s e d } } { \Delta \Omega _ { \mathrm { o p e n } } ^ { \mathrm { m a x } } } + \Delta I\tag{4}
$$

Equation (1) follows in particular from:

$$
H _ { S h a n n o n } = - \frac { 1 } { k _ { b } } H _ { B o l t z m a n } = \frac { 1 } { k _ { b } } \ln W\tag{5}
$$

For details see [1]. The underlying assumptions are discussed in the next section.

## III. DISCUSSION OF EQUATION (5): MEANING OF W

The concept of entropy comes from statistical mechanics, but in particular if we follow an axiomatic probabilistic instead of a frequentistic approach, it can be generalized to a large class of systems including the biological ones, see [5], [6]. Entropy allows to connect a large number of ‘micro’ variables to a much smaller number of observed variables. In this case we can deduct the expected length of the control program of a dynamical system. Equation (5) holds when all Shannon-Kolgomorov (S-K) four axioms (1)- (4) are valid and the underlying dynamics is then estensive. This is true if we look at the possible paths in the phase– space of our systems. The S-K 4th axyom is satisfied as the trajectories of the system in the phase–space only depend on its dynamical equations. If we had for example hysteresis interactions among the paths (in a quantum) or any kind of history dependent behaviors the form of the relation S(W), S for thermodinamic entropy, and thus of H(W), H for Shannon Entropy, would change, see [5]. A more detailed discussion can be found in Supplementary Information in Supplementary Section I.

## IV. PASSIVE WALKERS

Passive dynamic walkers are a powerful tool to illustrate morphological computation: body geometry and gravity restrict the set of admissible motions so strongly that little or no control is required. Figures 1 and 2 show a representation of the phase-space and of the main variables for a simple passive walker, respectively. Following the suggested framework able to link phase–space volume W to information and control complexity, we compute W for progressively richer walking morphologies.

## A. Ruina / Garcia Simplest Walker

Model assumptions: hip point mass, massless legs, point feet, plastic heel–strike. The standard heel–strike law (exact result) is expressed as

$$
\omega ^ { + } = \cos ( 2 \alpha ) \omega ^ { - }\tag{6}
$$

and the admissible domain is represented by:

$$
\begin{array} { r } { D = \big \{ ( \theta , \omega ) : | \theta | \leq \alpha , ~ \frac { 1 } { 2 } I \omega ^ { 2 } + m g \ell ( 1 - \cos \theta ) \leq E ^ { + } \big \} } \end{array}\tag{7}
$$

Hence, the phase–space volume can be easily computed:

$$
W = \int _ { - \alpha } ^ { \alpha } 2 \sqrt { \frac { 2 } { I } \left( E ^ { + } - m g \ell ( 1 - \cos \theta ) \right) } d \theta .\tag{8}
$$

## B. Knee–Ankle Compliance and Foot–Rocker Effects

Adding compliance to the structure (extra coordinate ϕ) we obtain

$$
\begin{array} { r } { H = \frac { 1 } { 2 } I \omega ^ { 2 } + m g \ell ( 1 - \cos \theta ) + \frac { 1 } { 2 } J \dot { \phi } ^ { 2 } + \frac { 1 } { 2 } k \phi ^ { 2 } } \end{array}\tag{9}
$$

with main inverted pendulum angle $\theta ,$ tip mass m, internal compliant joint $\phi ,$ torsional spring k, and secondary inertia $J .$ The admissible domain becomes four-dimensional and its geometry can be estimated by Monte–Carlo sampling.

![](images/c0a1aa92e831f155a60a6ed20390407797b9a5d3c59af99689d1f4acd6e3bc8c.jpg)  
Fig. 1. Phase–space admissible region (passive leg).

![](images/1b561597cfc73243b9d6ddeb97560f67e9de20ae9dfe72ced5ca85d716f8d7b0.jpg)  
Fig. 2. Passive Walkers in the phase-space. Left: γ is the angle of the slope for a pure passive walker. Center: Illustrative Picture. Right: Our Simplified Model

Compliance can modify both the geometry of the viable state space and the probability measure induced over it. In particular, internal elastic energy storage may redistribute occupancy over the task variables without necessarily contracting their projected support. Reduced collision losses, on the other hand, may modify the viable domain of the full hybrid dynamics. Rocker feet can similarly affect viability by modifying the impact mechanics and the corresponding post-impact continuation conditions.

Equation 9 is obtained by augmenting the standard invertedpendulum energy of the passive walker with a single compliant internal coordinate ϕ. The compliant subsystem is modeled as a linear torsional spring of stiffness k and inertia $J ,$ yielding the quadratic energy contribution

$$
H _ { \eta } = \frac { 1 } { 2 } k \phi ^ { 2 } + \frac { 1 } { 2 } J \dot { \phi } ^ { 2 } .
$$

Such a representation corresponds to the small-deflection approximation commonly adopted in compliant multibody systems and passive-dynamic locomotion models.

## V. COMPASS GAIT WITH UNIFORM-ROD LEGS

For a walker with uniform-rod legs, let us define coordinates $\boldsymbol { q } = [ q _ { 1 } , q _ { 2 } ] ^ { T }$ , each corresponding to a leg. Then, the swing-foot position can be expressed as

$$
p _ { \mathrm { s w } } ( q ) = \left[ \ell ( \sin q _ { 1 } - \sin q _ { 2 } ) , \ell ( \cos q _ { 1 } - \cos q _ { 2 } ) \right] ^ { T }\tag{10}
$$

and the Jacobian matrix can be computed:

$$
\begin{array} { r } { J = \left[ \begin{array} { c c } { \ell \cos q _ { 1 } } & { - \ell \cos q _ { 2 } } \\ { - \ell \sin q _ { 1 } } & { \ell \sin q _ { 2 } } \end{array} \right] . } \end{array}\tag{11}
$$

The heel–strike velocity reset is formulated as:

$$
\dot { q } ^ { + } = \left( I - M ^ { - 1 } J ^ { T } ( J M ^ { - 1 } J ^ { T } ) ^ { - 1 } J \right) \dot { q } ^ { - } ,\tag{12}
$$

![](images/23b490a3ad57705fed85171d7c764295f973219be97e768129c4274f3a78ef39.jpg)  
Fig. 3. W(γ) with full heel–strike acceptance.

<table><tr><td>γ (deg)</td><td>W(γ)</td></tr><tr><td>0.5</td><td>0.056</td></tr><tr><td>1.0</td><td>0.079</td></tr><tr><td>2.0</td><td>0.112</td></tr><tr><td>3.0</td><td>0.137</td></tr><tr><td>4.0</td><td>0.159</td></tr></table>

TABLE I REPRESENTATIVE VIABLE PHASE–SPACE VOLUME VALUES.

followed by leg relabeling. Numeric example at $\alpha = 0 . 3 \colon$

$$
\begin{array} { r } { A \approx \left[ \begin{array} { l l } { 0 . 8 1 9 8 } & { - 0 . 0 1 5 4 } \\ { 0 . 5 1 4 9 } & { - 0 . 0 1 9 1 } \end{array} \right] . } \end{array}\tag{13}
$$

## A. Full State–Dependent Monte–Carlo Estimation

The post–impact viable set $D ^ { + }$ assumes:

$$
q ^ { + } \approx [ - \alpha , + \alpha ] \pm d q , \qquad T ^ { + } \leq E ^ { + } ( \gamma ) .\tag{14}
$$

Monte–Carlo acceptance with exact heel–strike reset yields the representative values in Table I, also plotted in Figure 3.

More details can be found in Supplementary Information Section IV.

B. Compliance, Phase–Space Geometry, and Informational Complexity

As mentioned, the main modeling issue is to link the information processing metrics of the ‘mind’ of the agent to those of the body’s system dynamics.

Following [1] and [4], the complexity bound can be written as in equations (1–4). Equation (1) follows in particular from the relation between Shannon and Boltzmann entropy, given in (5):

$$
H _ { \mathrm { S h a n n o n } } = - { \frac { 1 } { k _ { b } } } H _ { \mathrm { B o l t z m a n n } } = { \frac { 1 } { k _ { b } } } \ln W\tag{5}
$$

where W denotes the number (or phase–space measure) of admissible microstates of the system.

## C. Meaning of W in Passive Dynamic Walking

In passive dynamic walkers, the admissible configuration set is defined by an energy constraint in the phase–space.

For a simple inverted pendulum walker

$$
D ( E ) = \{ ( \theta , \omega ) : H _ { x } ( \theta , \omega ) \leq E \}\tag{15}
$$

the Hamiltonian

$$
\begin{array} { r } { H _ { x } = \frac { 1 } { 2 } I \omega ^ { 2 } + m g \ell ( 1 - \cos { \theta } ) } \end{array}\tag{16}
$$

corresponds to the standard energy function of a simple pendulum [7], [8].

The viable phase–space volume is therefore:

$$
W ( E ) = \int _ { D ( E ) } d \theta d \omega\tag{17}
$$

Under a uniform occupancy model, $W ( E )$ determines a Boltzmann-type support entropy associated with the admissible set. In the more general nonuniform case considered below, however, scalar support volume alone does not determine the distributional complexity; the probability measure induced over the viable set must also be specified.

## D. Effect of Compliance: Support Geometry and Induced Measure

Consider the addition of compliant degrees of freedom,

$$
z = ( x , \eta ) , \qquad x = ( \theta , \omega ) , \qquad \eta = ( \phi , \dot { \phi } ) .\tag{18}
$$

For analytical tractability, assume the separable Hamiltonian

$$
H ( z ) = H _ { x } ( x ) + H _ { \eta } ( \eta ) , \qquad H _ { \eta } ( \eta ) \geq 0 ,\tag{19}
$$

where

$$
H _ { \eta } ( \eta ) = \frac { 1 } { 2 } k \phi ^ { 2 } + \frac { 1 } { 2 } J \dot { \phi } ^ { 2 } .\tag{20}
$$

The rigid admissible set is

$$
D _ { r } ( E ) = \{ x : H _ { x } ( x ) \leq E \} ,\tag{21}
$$

whereas the compliant admissible set is

$$
D _ { c } ( E ) = \{ ( x , \eta ) : H _ { x } ( x ) + H _ { \eta } ( \eta ) \leq E \} .\tag{22}
$$

Let

$$
\Pi _ { x } D _ { c } ( E ) = \{ x : \exists \eta \mathrm { s u c h ~ t h a t } H _ { x } ( x ) + H _ { \eta } ( \eta ) \leq E \}\tag{23}
$$

denote the projection onto the task variables.

Since $H _ { \eta } ( \eta ) \geq 0 .$

$$
\Pi _ { x } D _ { c } ( E ) \subseteq D _ { r } ( E ) .\tag{24}
$$

However, the compliant subsystem possesses the admissible zero-energy equilibrium

$$
\eta _ { 0 } = ( 0 , 0 ) , \qquad H _ { \eta } ( \eta _ { 0 } ) = 0 .\tag{25}
$$

Therefore, for every $x \in D _ { r } ( E )$ , choosing $\eta = \eta _ { 0 }$ gives

$$
H _ { x } ( x ) + H _ { \eta } ( \eta _ { 0 } ) = H _ { x } ( x ) \leq E ,\tag{26}
$$

and hence

$$
\boxed { \Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) . }\tag{27}
$$

Thus, in the separable model considered here, compliance does not by itself contract the projected geometric support of the task variables. Instead, the additional compliant degrees of freedom modify how the available energy is distributed over that support.

## E. Consequences for Informational Complexity

The equality of projected supports does not imply equality of the probability measures induced on those supports. Denoting the rigid and compliant measures by $P _ { r }$ and $P _ { c }$ respectively, one may have

$$
\operatorname { s u p p } ( P _ { r } ) = \operatorname { s u p p } ( P _ { c } ) = D _ { r } ( E ) ,\tag{28}
$$

while

$$
\begin{array} { r } { P _ { r } \neq P _ { c } . } \end{array}\tag{29}
$$

Morphology can therefore influence locomotion complexity through two conceptually distinct mechanisms: it may alter the geometry of the viable set, or it may reshape the probability distribution induced over that set. These effects need not occur simultaneously.

Accordingly, the scalar viable phase–space volume W should be interpreted as a zeroth-order descriptor of support size rather than as a complete characterization of informational complexity. Section VII introduces an RKHS description that incorporates the dynamically induced probability measure and therefore distinguishes systems having identical viable support but different statistical organization.

## VI. MONTECARLO ANALYSIS

This section presents quantitative results based on Montecarlo simulation, demonstrating how the morphological properties of passive dynamic walkers affect the viable phase–space volume W and the associated informational complexity measure $K ( X )$ . More details can be found in the Supplementary Information section IV.

## A. Simulation Setup

We consider three progressively richer models of passive dynamic walking:

1) A rigid compass gait walker with two degrees of freedom $( \theta , \omega )$

2) A compliant walker with an additional joint elasticity coordinate $( \phi , \dot { \phi } )$

3) A compliant walker with distributed leg inertia and heel–strike dynamics.

For each model, viable phase–space volumes were estimated using the Monte–Carlo method described in detail in Supplementary Information section IV. Samples were drawn uniformly in a bounding region and accepted if they satisfied both energy constraints and post–impact feasibility.

Kernel complexity was computed using a Gaussian kernel Gram matrix:

$$
K ( X ) = \log \operatorname* { d e t } ( G + \lambda I ) .\tag{30}
$$

All simulations used $1 0 ^ { 5 }$ samples drawn uniformly from the admissible energy-bounded region and a kernel band width chosen according to Silverman’s rule. The reported averages are therefore geometric estimates of the viable phase–space volume rather than dynamical time averages.

## B. Effect of Compliance on the Task-State Distribution

The Monte–Carlo analysis illustrates the distinction between geometric support and probability distribution discussed above. Consider the local quadratic model

$$
H _ { x } = { \frac { 1 } { 2 } } a \theta ^ { 2 } + { \frac { 1 } { 2 } } b \omega ^ { 2 } , \qquad H _ { \eta } = { \frac { 1 } { 2 } } k \phi ^ { 2 } + { \frac { 1 } { 2 } } J \dot { \phi } ^ { 2 } .\tag{31}
$$

For uniform sampling inside the two-dimensional rigid energy ellipse,

$$
\mathbb { E } _ { r } [ H _ { x } ] = \frac { E } { 2 } .\tag{32}
$$

For the compliant system, uniform sampling inside the full four-dimensional energy ellipsoid instead gives

$$
\mathbb { E } _ { c } [ H _ { x } ] = \frac { E } { 3 } .\tag{33}
$$

Hence

$$
\boxed { \mathbb { E } _ { c } [ H _ { x } ] < \mathbb { E } _ { r } [ H _ { x } ] }\tag{34}
$$

despite

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) .\tag{35}
$$

Compliance therefore redistributes probability mass toward lower-energy task states without necessarily reducing the projected task-space support. The corresponding derivation and numerical illustration are reported in Section VII-F and in the Supplementary Information.

## C. Impact of Heel–Strike Dynamics

Including realistic heel–strike dynamics further reduced viable phase–space volume. The impact map introduces an instantaneous projection of the velocity state onto a lower– energy manifold.

For the Ruina–Garcia simplest walker, the heel–strike reset is

$$
\omega ^ { + } = \cos ( 2 \alpha ) \omega ^ { - } .\tag{36}
$$

For the value $\alpha = 0 . 3$ rad used in the numerical example, the velocity-retention coefficient is

$$
\rho \equiv \cos ( 2 \alpha ) = \cos ( 0 . 6 ) \simeq 0 . 8 2 5 3 .\tag{37}
$$

Consequently, the kinetic energy associated with the angular velocity is reduced at impact according to

$$
\frac { T ^ { + } } { T ^ { - } } = \left( \frac { \omega ^ { + } } { \omega ^ { - } } \right) ^ { 2 } = \cos ^ { 2 } ( 2 \alpha ) \simeq 0 . 6 8 1 .\tag{38}
$$

Thus, approximately 32% of the pre-impact kinetic energy associated with this degree of freedom is lost at heel strike.

It is important to distinguish this energetic contraction from the contraction of phase–space volume. In the reduced $( \theta , \omega )$ representation, the reset map alone has absolute Jacobian

$$
| \mathrm { d e t } D \Delta | = | \cos ( 2 \alpha ) | \simeq 0 . 8 2 5 .\tag{39}
$$

Hence, the collision map alone contracts coordinate phase– space area by approximately 17.5%. This geometric contraction must be distinguished from the reduction in kinetic energy, $T ^ { + } / T ^ { - } \simeq 0 . 6 8 1$ , and from any change in the viable phase–space volume $W ( D _ { v } )$ . The latter additionally depends on which post-impact states can successfully complete the subsequent step and is therefore a property of the full hybrid walking dynamics rather than of the reset map alone.

This shows that collision dynamics plays a significant role in shaping the accessible phase–space of passive locomotion systems.

## VII. RKHS GEOMETRY OF THE VIABLE STATE DISTRIBUTION

The previous sections characterized the viable behavior of passive walking systems through Hamiltonian energy constraints and projected phase–space geometry. However, geometry alone does not completely determine the computational burden associated with locomotion.

The crucial observation is that the walking dynamics does not merely define a viable set of admissible states. It also induces a probability measure over this set. While different morphologies may possess identical geometric support, they generally generate different probability distributions over the corresponding viable trajectories.

Consequently, the complexity experienced by the controller should not be regarded solely as a function of the size of the viable region, but rather as a function of the probability measure supported on that region.

This viewpoint naturally leads to a geometric-statistical description based on reproducing kernel Hilbert spaces (RKHS), where both the geometry of the viable manifold and the probability measure induced by the dynamics can be represented within a common mathematical framework.

## A. Geometry of the Viable State Space

Let X denote the state space of the walker. For a given total energy budget E the Hamiltonian

$$
H : \mathcal { X }  \mathbb { R }\tag{40}
$$

defines the viable domain

$$
{ \cal D } ( E ) = \left\{ x \in \mathcal { X } : H ( x ) \leq E \right\} .\tag{41}
$$

When heel–strike events are present, only states whose reset map produces an admissible post-impact configuration are retained. Accordingly, the effective viable domain becomes

$$
D _ { v } ( E ) \subseteq D ( E ) .\tag{42}
$$

The set $D _ { v } ( E )$ is therefore determined jointly by

• energetic feasibility;

• collision constraints;

• impact consistency;

• passive stability.

It represents the subset of the state space from which passive walking can continue without external intervention.

Notice that this construction is purely geometric. It specifies which states are admissible, but it does not specify how frequently these states are visited during locomotion.

Consequently, $D _ { v } ( E )$ should be interpreted as the support of the dynamics rather than their statistical description.

For later developments we denote by

$$
W ( D _ { v } ) = \int _ { D _ { v } } d V\tag{43}
$$

the geometric volume of the viable region with respect to the chosen reference measure.

Although this quantity provides an intrinsic measure of the size of the viable domain, it does not capture how trajectories are distributed inside it.

## B. Dynamically Induced Probability Measure

The walking dynamics generates trajectories $x ( t ) \in D _ { v }$ which induce a probability measure P over the viable domain.

Instead of regarding every viable state as equally relevant, the probability measure weighs each region according to the frequency with which it is visited by the dynamics.

For any measurable subset $A \subseteq D _ { i }$ the induced measure is defined as

$$
P ( A ) = \operatorname* { l i m } _ { T \to \infty } { \frac { 1 } { T } } \int _ { 0 } ^ { T } \mathbf { 1 } _ { A } ( x ( t ) ) d t ,\tag{44}
$$

whenever the limit exists, and where $\mathbf { 1 } _ { A } ( x ( t ) )$ denotes the indicator function of the set A.

Equivalently, if the dynamics admits an invariant density $\rho ( x )$ then

$$
d P ( x ) = \rho ( x ) d V ( x ) ,\tag{45}
$$

with

$$
\int _ { D _ { v } } \rho ( x ) d V ( x ) = 1 .\tag{46}
$$

The geometric volume $W ( D _ { v } )$ and the probability measure P are fundamentally different objects.

The former characterizes the support of the viable dynamics, while the latter characterizes how the dynamics populate that support.

This distinction is particularly important when comparing rigid and compliant walkers.

For the separable Hamiltonian

$$
H ( x , \eta ) = H _ { x } ( x ) + H _ { \eta } ( \eta ) ,\tag{47}
$$

with

$$
H _ { \eta } ( \eta ) \geq 0 ,\tag{48}
$$

the projection of the compliant viable set onto the task variables may coincide with the rigid support,

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) ,\tag{49}
$$

even though the corresponding induced measures differ.

Indeed, numerical simulations presented later show that uniform sampling inside the full compliant energy ellipsoid produces a projected distribution that is substantially more

concentrated around low-energy task states than the rigid distribution (refer to Figure 4).

Consequently, $P _ { \mathrm { c o m p l i a n t } } \ne P _ { \mathrm { r i g i d } }$ , despite having essentially identical geometric support.

As already suggested, this observation strongly motivates the exploitation of the RKHS formulation.

Rather than measuring only the size of the viable region, we seek to quantify the complexity of the probability measure supported on that region. In this perspective,

Geometry determines where locomotion is possible, Dynamics determine how locomotion is distributed, RKHS quantify the complexity of that distribution.

## C. RKHS Embedding of the Viable State Distribution

The previous subsection established that the relevant object is not only the viable state space itself, but also the probability measure induced by the walking dynamics.

This naturally leads to the use of reproducing kernel Hilbert spaces, where probability measures can be represented as geometric objects through their kernel mean embeddings.

Let $k : \mathcal { X } \times \mathcal { X } \to \mathbb { R }$ be a continuous positive-definite kernel and let $\mathcal { H } _ { k }$ denote the associated reproducing kernel Hilbert space.

The canonical feature map is

$$
\Phi ( x ) = k ( \cdot , x ) .\tag{50}
$$

Given the dynamically induced probability measure P introduced in the previous subsection, its RKHS representation is defined through the kernel mean embedding

$$
\mu _ { P } = \int _ { \mathcal X } k ( \cdot , x ) d P ( x )\tag{51}
$$

whenever the integral exists in the Bochner sense. This representation can be interpreted as the RKHS-valued expectation of the feature map $k ( \cdot , x )$

The kernel mean embedding provides a complete representation of the probability measure whenever the kernel is characteristic. Consequently,

$$
P _ { 1 } = P _ { 2 } \Longleftrightarrow \mu _ { P _ { 1 } } = \mu _ { P _ { 2 } } ,\tag{52}
$$

allowing different walker morphologies to be compared directly in feature space.

Unlike geometric volume, which only characterizes the support of the viable region, the kernel mean embedding captures both the support and the probability mass distributed over it.

Therefore two walkers possessing identical viable domains may still produce different RKHS representations whenever the induced probability measures differ.

This situation naturally arises in compliant passive walkers. Although the projected viable domains may coincide,

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) ,\tag{53}
$$

their induced measures satisfy

$$
\begin{array} { r } { P _ { \mathrm { c } } \neq P _ { \mathrm { r } } , } \end{array}\tag{54}
$$

leading to distinct kernel mean embeddings

$$
\mu _ { c } \neq \mu _ { r } .\tag{55}
$$

The computational complexity experienced by the controller is therefore associated with the geometry of these probability embeddings rather than with the size of the viable domain alone.

## D. Pullback Geometry

The RKHS embedding induces a natural differential ${ \mathrm { g e - } }$ ometry on the viable state distribution.

Following the coherent-state construction introduced in Section IV, every state $x \in D _ { v }$ is mapped onto the projective Hilbert space through

$$
\mathcal { K } : D _ { v }  \mathbb { P } ( \mathcal { H } _ { k } ) , \qquad x \mapsto [ k ( \cdot , x ) ] .\tag{56}
$$

The projective Hilbert space is endowed with the Fubini– Study metric $g _ { F S }$ , [9].

Pulling this metric back through K induces the Riemannian metric

$$
g _ { k } = \mathcal { K } ^ { * } g _ { F S }\tag{57}
$$

on the viable state manifold.

The corresponding infinitesimal distance is

$$
\begin{array} { r } { d s ^ { 2 } = g _ { i j } ( x ) d x ^ { i } d x ^ { j } . } \end{array}\tag{58}
$$

Unlike the Euclidean metric of the original coordinates, this metric measures distinguishability in feature space.

Two states that are close in Euclidean coordinates may become well separated whenever they induce substantially different kernel responses.

Conversely, dynamically similar states become naturally clustered, independently of the coordinate system used to describe the walker.

The pullback metric therefore defines an intrinsic geometry of the viable dynamics rather than of the state coordinates.

The corresponding Riemannian volume form is

$$
d V _ { k } = { \sqrt { \operatorname* { d e t } g _ { k } ( x ) } } d x .\tag{59}
$$

Integrating this volume form over the viable domain yields the kernel-induced geometric volume

$$
W _ { k } = \int _ { D _ { v } } d V _ { k } .\tag{60}
$$

Notice, however, that $W _ { k }$ should not generally be interpreted as the physical phase-space volume.

Instead, it measures the geometric size of the viable region under the similarity notion induced by the kernel.

The equality $W _ { k } ~ = ~ W _ { \mathrm { p h y s } }$ holds only if the pullback metric coincides with the physical metric, $g _ { k } = g _ { \mathrm { p h y s } }$

More generally, $W _ { k }$ defines an intrinsic geometric measure associated with the RKHS representation of viable locomotion. This distinction is fundamental.

The geometric volume $W _ { k }$ characterizes the manifold supporting the dynamics, whereas the probability measure $P$ characterizes how trajectories populate that manifold.

The next subsection combines these two ingredients by introducing a covariance operator over the RKHS, from which an intrinsic measure of distributional complexity is obtained.

## E. RKHS Distributional Complexity

The previous sections introduced the geometric volume $W ( D _ { v } )$ in Equation ( 43) as a quantitative measure of the size of the viable state space.

This quantity provides a zeroth-order geometric descriptor of the size of the viable state space and, under the assumptions underlying Equation (5) , can be related to a supportbased informational description of the walking task.

However, the geometric volume alone does not completely determine the effective complexity experienced by the controller.

Indeed, two walkers may possess viable regions of identical geometric size while exhibiting substantially different dynamical behavior owing to different probability distributions over the same viable manifold.

The volume therefore characterizes the geometry of the support, whereas the induced probability measure characterizes how this support is actually explored by the dynamics.

The natural extension of the geometric volume is thus obtained by considering the pair $( D _ { v } , P )$ , where $P$ denotes the dynamically induced probability measure introduced in the previous subsection.

Within the RKHS framework this pair is represented through its kernel mean embedding and covariance operator.

From Equations (50) and (51) recall that $\Phi ( x ) = k ( \cdot , x )$ denotes the feature map associated with the reproducing kernel, and that the kernel mean embedding of the viable dynamics is $\begin{array} { r } { \mu _ { P } = \int \Phi ( x ) d P ( x ) } \end{array}$ . The centered covariance operator is then expressed as:

$$
C _ { P } = \int ( \Phi ( x ) - \mu _ { P } ) \otimes \left( \Phi ( x ) - \mu _ { P } \right) d P ( x ) .\tag{61}
$$

This covariance operator describes how the probability measure is distributed over the geometric directions defined by the RKHS.

If the dynamics occupies only a few independent directions, the spectrum of $C _ { P }$ becomes concentrated. Conversely, broadly distributed dynamics generate covariance operators with larger effective rank.

This observation motivates the following definition.

Definition VII.1 (RKHS Distributional Complexity). Let $C _ { P }$ be the covariance operator associated with the probability measure $P .$

The RKHS distributional complexity is defined as

$$
\begin{array} { r } { \boxed { \mathcal { C } ( P ) = \log \operatorname* { d e t } _ { F } ( I + \gamma C _ { P } ) , } } \end{array}\tag{62}
$$

where $\gamma > 0$ is a scaling parameter and det denotes the Fredholm determinant.

Equivalently,

$$
\mathcal C ( P ) = \sum _ { i } \log ( 1 + \gamma \lambda _ { i } ) ,\tag{63}
$$

where $\{ \lambda _ { i } \}$ are the eigenvalues of $C _ { P }$

The complexity introduced above naturally generalizes the phase-space volume employed throughout the previous sections.

Indeed, if the viable dynamics populate the viable manifold uniformly,

$$
d P = { \frac { 1 } { W ( D _ { v } ) } } d V ,\tag{64}
$$

implying that the probability distribution is completely determined by the geometry of the support.

In this limiting situation, the probability measure is fixed once the viable manifold and its reference volume measure are specified. The covariance operator therefore depends only on the geometry of the viable support as represented through the chosen kernel. Importantly, this does not imply that the complexity is a function of the scalar volume $W ( D _ { v } )$ alone: domains having identical volume may possess different shapes, metric structures, and kernel covariance spectra.

Accordingly, $W ( D _ { v } )$ should be interpreted as a zeroth– order descriptor of viable-support size, whereas $\mathcal { C } ( P )$ provides a richer characterization of the metric–measure structure of the viable dynamics.

Theorem 7.2 (Generalized Complexity Principle). Let $P$ be a probability measure supported on the viable manifold $D _ { v }$ , and let

$$
{ \mathcal { C } } ( P ) = \log \operatorname* { d e t } _ { F } ( I + \gamma C _ { P } )\tag{65}
$$

denote its RKHS distributional complexity.

Then the following statements hold.

1) If the viable dynamics populate $D _ { v }$ uniformly,

$$
d P ( x ) = \frac { 1 } { W ( D _ { v } ) } d V ( x ) ,\tag{66}
$$

then

$$
\mathcal { C } ( P ) = \mathcal { F } _ { k } ( D _ { v } , d V ) ,\tag{67}
$$

where $\mathcal { F } _ { k }$ is a functional of the geometry of the viable manifold and of the chosen kernel k.

Thus, under uniform occupancy, the distributional complexity is completely determined by the geometry of the viable support, but not, in general, by its scalar volume $W ( D _ { v } )$ alone.

2) For an arbitrary probability measure $P ,$

$$
\mathcal { C } ( P ) = \mathcal { F } _ { k } ( D _ { v } , P ) ,\tag{68}
$$

and therefore depends jointly on the geometry of the viable support and on the probability distribution induced over that support by the dynamics.

Proof. For the uniform measure,

$$
d P ( x ) = \frac { 1 } { W ( D _ { v } ) } d V ( x ) ,\tag{69}
$$

the kernel mean and covariance operator are

$$
\mu _ { P } = { \frac { 1 } { W ( D _ { v } ) } } \int _ { D _ { v } } \Phi ( x ) d V ( x )\tag{70}
$$

and

$$
C _ { P } = { \frac { 1 } { W ( D _ { v } ) } } \int _ { D _ { v } } \bigl ( \Phi ( x ) - \mu _ { P } \bigr ) \otimes \bigl ( \Phi ( x ) - \mu _ { P } \bigr ) d V ( x ) .\tag{71}
$$

Hence the spectrum of $C _ { P }$ , and consequently $\mathcal { C } ( P )$ , is determined by the full geometry of $D _ { v }$ as represented through the kernel k.

In general, two viable domains may have equal scala volume,

$$
W ( D _ { v } ^ { ( 1 ) } ) = W ( D _ { v } ^ { ( 2 ) } ) ,\tag{72}
$$

while possessing different shapes or metric structures. Their kernel covariance spectra may therefore differ, implying g

$$
{ \mathcal { C } } ( P _ { 1 } ) \neq { \mathcal { C } } ( P _ { 2 } ) .
$$

Consequently, scalar phase–space volume alone is not suffi cient to determine RKHS distributional complexity.

1) Support Geometry: Since the internal equilibrium state $\eta = 0$ is admissible, every task state satisfying

$$
H _ { x } ( x ) \leq E\tag{81}
$$

admits at least one compliant realization. Therefore,

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) .\tag{82}
$$

Hence, for this separable quadratic model, compliance does not reduce the support of the projected viable set. In particular, the rigid and compliant projected task-space volumes are equal.

The exact Euclidean area of the rigid ellipse is

$$
W _ { \mathrm { E u c } } = { \frac { 2 \pi E } { \sqrt { a b } } } = 2 \pi \approx 6 . 2 8 3 2 .\tag{83}
$$

r A rejection-sampling Monte–Carlo estimate using $N =$ $3 \times 1 0 ^ { 5 }$ samples gave

$$
\widehat { W } _ { \mathrm { E u c } } = 6 . 2 8 9 1 ,\tag{84}
$$

which is consistent with the analytical value.

2) Dynamically Induced Task-State Distribution: Al-<sub>(73)</sub>though the projected supports coincide, the probability measures induced on task space are different.

<sup>For</sup> <sup>a</sup> <sup>nonuniform</sup> <sup>measure</sup> <sup>P,</sup> <sup>the</sup> <sup>covariance</sup> <sup>operator</sup>ant walker, samples were drawn uniformly from the four- rant walker, samples were drawn uniformly from the fouradditionally depends on how probability mass is distributed<sub>dimensional energy ellipsoid and subsequently projected</sub> ddimensional energy ellipsoid and subsequently projected within $D _ { v }$ . Therefore C(P) is a functional of the metric–<sub>onto the task variables.</sub> $\mathcal { C } ( P )$ onto the task variables.

$$
( D _ { v } , P )
$$

## F. Numerical Illustration

We now illustrate the distinction between geometric sup port, kernel-induced volume, and RKHS distributional com plexity by means of a quadratic passive-walker model.

Let the task variables be

$$
x = ( \theta , \omega ) ,
$$

$$
\eta = ( \phi , \dot { \phi } ) .
$$

and let the compliant internal variables be

$$
H _ { x } ( \theta , \omega ) = \frac { 1 } { 2 } a \theta ^ { 2 } + \frac { 1 } { 2 } b \omega ^ { 2 } ,\tag{75}
$$

We consider the quadratic energies

(76)

and

$$
H _ { \eta } ( \phi , \dot { \phi } ) = \frac { 1 } { 2 } k \phi ^ { 2 } + \frac { 1 } { 2 } J \dot { \phi } ^ { 2 } .
$$

The numerical parameters are

$$
a = b = J = E = 1 , \qquad k = 4 .\tag{78}
$$

The rigid viable set is

For the rigid walker, samples were drawn uniformly from the two-dimensional energy ellipse. For the compli-

$$
D _ { r } ( E ) = \{ x : H _ { x } ( x ) \leq E \} ,\tag{79}
$$

and and whereas the compliant viable set in the augmented state space

$$
D _ { c } ( E ) = \left\{ ( x , \eta ) : H _ { x } ( x ) + H _ { \eta } ( \eta ) \leq E \right\} .
$$

The resulting rigid distribution is uniform over the taskspace ellipse, whereas the projected compliant distribution is more concentrated around low-energy task states.

<sup>-</sup> For uniform sampling inside a d-dimensional quadratic <sup>-</sup>energy ball, the expected total normalized energy is

$$
\mathbb { E } [ H ] = { \frac { d } { d + 2 } } E .\tag{85}
$$

<sup>(74)</sup> For the rigid two-dimensional model, all quadratic coordinates belong to the task subsystem, and therefore

$$
\mathbb { E } [ H _ { x } ] _ { \mathrm { r i g i d } } = \frac { 2 } { 2 + 2 } E = \frac { E } { 2 } .\tag{86}
$$

For the compliant four-dimensional model,

$$
\mathbb { E } [ H _ { x } ] = \frac { 4 } { 4 + 2 } E = \frac { 2 E } { 3 } .\tag{87}
$$

Since two of the four quadratic coordinates belong to the task subsystem, the energy fraction received by such subsystem is ${ \frac { 2 } { 4 } } .$ , implying:

$$
\mathbb { E } [ H _ { x } ] _ { \mathrm { c o m p l i a n t } } = { \frac { 1 } { 2 } } E [ H _ { x } ] = { \frac { E } { 3 } } .\tag{88}
$$

The simulations gave

$$
\begin{array} { r } { \widehat { \mathbb E } [ H _ { x } ] _ { \mathrm { r i g i d } } = 0 . 5 0 0 4 , } \end{array}\tag{89}
$$

$$
\widehat { \mathbb E } [ H _ { x } ] _ { \mathrm { c o m p l i a n t } } = 0 . 3 3 2 8 ,\tag{90}
$$

(80)in close agreement with the theoretical values $\frac { 1 } { 2 }$ and ${ \frac { 1 } { 3 } } .$

![](images/3b286c2b97086261e98f3d1501cb4c78d3940c7b8574bb2771ff19df55c86c97.jpg)

Fig. 4. Rigid and projected compliant task-state samples. The two distributions have the same support, but the projected compliant distribution is more concentrated around low-energy states.  
![](images/9daeac5337bd7444189bd6e148644df2782576d2db057948fa7e333561816c3e.jpg)  
Fig. 5. Task-energy distributions for rigid and projected compliant samples. Uniform sampling in the full compliant energy ellipsoid shifts probability mass toward lower values of $H _ { x }$

Figure 4 shows that the two distributions have the same support, while the compliant projection is more densely concentrated near the origin. Figure 5 shows the corresponding shift toward lower task energy.

3) Kernel-Induced Pullback Volume: To illustrate the pullback construction, we used the inhomogeneous polynomial kernel

$$
K ( x , y ) = ( c + x ^ { \top } y ) ^ { p } ,\tag{91}
$$

with

$$
c = 1 , \qquad p = 2 .
$$

For this kernel, the Fubini–Study pullback metric on the two-dimensional task space has volume density

$$
{ \sqrt { \operatorname* { d e t } g _ { K } ( x ) } } = { \frac { p { \sqrt { c } } } { \left( c + \| x \| ^ { 2 } \right) ^ { 3 / 2 } } } .\tag{92}
$$

The corresponding kernel-induced volume was estimated as

$$
\widehat { W } _ { K } = \frac { \mathrm { V o l } ( B ) } { N } \sum _ { i = 1 } ^ { N } \mathbf { 1 } _ { D _ { r } } ( x _ { i } ) \sqrt { \operatorname* { d e t } g _ { K } ( x _ { i } ) } .\tag{93}
$$

The numerical result was

$$
\widehat { W } _ { K } = 5 . 3 1 7 1 .\tag{94}
$$

Since the rigid and projected compliant supports coincide and the same kernel-induced metric is used in both cases, their pullback volumes also coincide.

This confirms that pullback volume remains a geometric quantity: it distinguishes geometries, but it does not by itself distinguish different probability measures supported on the same viable set.

4) Empirical RKHS Distributional Complexity: To quantify the difference between the induced probability measures, we computed the empirical RKHS distributional complexity.

Given $N _ { s } = 3 0 0$ task-state samples, the Gaussian Gram matrix was constructed as

$$
G _ { i j } = \exp \left( - \frac { \| x _ { i } - x _ { j } \| ^ { 2 } } { 2 \sigma ^ { 2 } } \right) .\tag{95}
$$

The bandwidth σ was selected once from the pooled rigid and compliant samples and then kept fixed for both models.

Let ${ \bf 1 } _ { N } \in \mathbb { R } ^ { N }$ denote the N-dimensional vector of ones and define

$$
H _ { N } = I _ { N } - \frac { 1 } { N } \mathbf { 1 } _ { N } \mathbf { 1 } _ { N } ^ { \top }\tag{96}
$$

as the centering matrix. The empirical complexity was evaluated as

$$
\widehat { \mathcal { C } } _ { \lambda } = \log \operatorname* { d e t } \left( I _ { N } + \frac { 1 } { N \lambda } H _ { N } G H _ { N } \right) ,\tag{97}
$$

with

$$
\lambda = 1 0 ^ { - 2 } .
$$

The experiment was repeated over 20 independent realizations. The resulting mean values were

$$
\widehat { \mathcal { C } } _ { \lambda } ^ { \mathrm { r i g i d } } = 1 1 . 9 3 5 ,
$$

and

$$
\widehat { \mathcal { C } } _ { \lambda } ^ { \mathrm { c o m p l i a n t } } = 9 . 8 6 5 .
$$

Thus,

$$
\widehat { \mathcal { C } } _ { \lambda } ^ { \mathrm { c o m p l i a n t } } < \widehat { \mathcal { C } } _ { \lambda } ^ { \mathrm { r i g i d } } ,\tag{98}
$$

even though

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) .\tag{99}
$$

The reduction is therefore distributional rather than support-based. The compliant morphology redistributes probability mass toward a smaller number of effective featurespace directions, thereby reducing the spectral spread of the centered kernel covariance operator.

Figure 6 reports the empirical complexity values over the independent realizations.

![](images/d0b5058f50870579b83652580d3f6bc5c71c90999cb44256c3963568fcb3af83.jpg)

Fig. 6. Empirical RKHS distributional complexity over independent realizations. The projected compliant distribution exhibits lower featurespace spectral complexity than the rigid distribution.
<table><tr><td>Quantity</td><td>Rigid</td><td>Compliant</td></tr><tr><td>Projected Euclidean support volume</td><td>6.2891</td><td>6.2891</td></tr><tr><td>Exact projected support volume</td><td>6.2832</td><td>6.2832</td></tr><tr><td>Kernel pullback volume</td><td>5.3171</td><td>5.3171</td></tr><tr><td>Mean task energy</td><td>0.5004</td><td>0.3328</td></tr><tr><td>RKHS distributional complexity</td><td>11.935</td><td>9.865</td></tr></table>

TABLE II  
NUMERICAL COMPARISON BETWEEN RIGID AND COMPLIANT MODELS.

5) Interpretation: The numerical results separate three conceptually distinct quantities:

$W _ { \mathrm { E u c } }$ measures the coordinate volume of the viable support.

$W _ { K }$ measures the geometric volume induced by the RKHS pullback metric.

• C(P) measures the spectral complexity of the probability distribution supported on that geometry.

In the present separable example, the first two quantities are the same for the rigid and compliant projected systems because their supports coincide. The third quantity differs because compliance changes how the viable region is populated.

The example therefore supports the generalized interpretation developed in the preceding subsection:

$$
{ \mathrm { g e o m e t r y } }  W , ~ { \mathrm { d y n a m i c s } }  P , ~ ( W , P )  { \mathcal { C } } ( P )
$$

Accordingly, viable phase-space volume remains the zeroth-order geometric descriptor, while RKHS distributional complexity captures the additional organization introduced by morphology and dynamics.

The results provide quantitative support for a refined interpretation of the morphological computation hypothesis. Morphology affects locomotion not only by shaping the set of physically viable states, but also by organizing the probability distribution generated by the dynamics over that set.

This distinction is particularly clear in the compliant quadratic example. The rigid and compliant systems possess the same projected task-space support, yet compliance redistributes probability mass toward lower-energy task states. The corresponding RKHS covariance spectra and distributional complexities therefore differ despite the equality of their projected geometric supports.

Impact mechanics provides a complementary mechanism. Heel–strike changes both energy and coordinate phase–space geometry, but neither the kinetic-energy retention factor nor the Jacobian of the reset map can be identified directly with the viable-volume ratio. Viability depends on the subsequent hybrid dynamics.

These results suggest that the contribution of morphology to control should be described through two complementary objects: the geometry of the viable state space and the probability measure induced over that geometry. Scalar phase– space volume $W ( D _ { v } )$ remains a useful zeroth-order descriptor of support size, whereas the RKHS formulation captures the additional statistical organization of viable behavior.

Within the Mind–Body Nexus framework, morphological computation can therefore be interpreted more generally as the capacity of physical embodiment to reshape the metric– measure structure of viable behavior, thereby altering the informational burden presented to the controller.

From a more general point of view, Equation (1) relies on very general assumptions and has a wide applicability. It provides a useful method to link the system dynamics of a robot to the computational effort needed to control it. It can explain, model, and measure the benefits - and shortcomings - of ‘softness’ in a given robot (or parts of it) for a given application, by following the approach proposed in [10]. This can be achieved by looking at how the morphology shapes the phase–space and consequently the information processing and the intelligent behaviors of a physical agent.

Furthermore, the results presented in Section VII-F support a refined interpretation of the morphological computation hypothesis.

Rather than acting solely by reducing the size of the viable phase-space region, morphology modifies both the geometry of the viable state space and the probability distribution induced by the system dynamics over that space. These two effects represent complementary mechanisms through which the physical embodiment contributes to the overall computation required for locomotion. The geometric aspect is captured by the viable manifold $D _ { v }$ and by its associated phase-space volume W, which quantifies the set of dynamically admissible states compatible with successful walking. The statistical aspect is captured by the dynamically induced probability measure P, describing how trajectories actually populate the viable manifold. The present work shows that these two quantities should not be confused. Indeed, different morphologies may generate identical viable supports while inducing substantially different probability measures over the same state space. Consequently, the computational burden experienced by the controller cannot, in general, be inferred from geometric volume alone. The RKHS formulation de veloped in Section VII provides a natural framework for integrating these two complementary aspects. The viable phase-space volume introduced in the first part of the paper represents a purely geometric description of locomotion com plexity. Kernel mean embeddings and covariance operators extend this description by incorporating the statistical orga nization of the dynamics. The resulting RKHS distributional complexity therefore constitutes a genuine generalization of the original phase-space volume measure. In the special case in which trajectories populate the viable manifold uniformly, the RKHS complexity becomes a functional of the geometry of the viable domain and of the chosen kernel. Scalar phase–space volume remains an important zeroth order descriptor, but domains of equal volume need not possess identical RKHS covariance spectra. More generally, distributional complexity depends jointly on the geometry of the support and on the probability measure induced by the dynamics. This interpretation is consistent with the Mind– Body Nexus framework. Within this perspective, the body does not merely constrain the space of possible behaviors. Instead, morphology actively shapes the probability struc ture of the behaviors generated by the interaction between body, environment, and controller. The informational burden assigned to the controller is therefore determined not only by the size of the viable state space but also by the statistical organization of the trajectories evolving within it. Morphological computation can thus be interpreted as an information-geometric phenomenon emerging from the coupled effects of viable-state geometry and induced dynamics. This viewpoint considerably broadens the applicability of the proposed framework. The theory is not restricted to passive dynamic walking or to compliant mechanisms. Whenever the dynamics of an embodied agent induces a probability distribution over a viable manifold, the same geometric and RKHS construction can be employed to quantify computational complexity. Potential application domains include compliant robotic manipulation, legged locomotion, soft robotics, adaptive morphologies, embodied reinforcement learning, and biological sensorimotor systems. In all these cases, the proposed framework provides a principled connection between morphology, dynamics, information processing, and controller complexity. Several research directions naturally follow from the present work. We emphasize that C(P) is an information-geometric descriptor of the organization of viable behavior, rather than a direct measurement of computational time, controller memory, or control energy. The hypothesis underlying the present framework is that reductions in the effective informational structure presented by the embodied dynamics can reduce the complexity required of a controller. Establishing quantitative relations between C(P) and specific controller resources constitutes an important direction for future work. First, extending the theory to stochastic and high-dimensional systems requires efficient numerical approximations of the induced probability measure and of the associated RKHS covariance operators. Second, the pullback geometry introduced in Section VII suggests a novel class of kernel-induced metrics for comparing embodied systems that may share the same Euclidean phase-space geometry while exhibiting fundamentally different dynamical organizations. Finally, integrating these information-geometric quantities into physics-aware learning algorithms may provide a systematic route toward morphology-aware control architectures in which learning, physical embodiment, and information processing are optimized jointly.

## ACKNOWLEDGMENT

This work has been partially supported by the PRIN project BEASTIE (2022FZJ7MR) and by “RKHS and Applications” (CUP C39J21042630006), both funded by the Italian Ministry of University and Research and by the European Union – NextGenerationEU, and supported by the EU project Horizon 2020 AIFORS 952275. E. Barletta, S. Dragomir and F. Esposito were partially supported by G.N.S.A.G.A - INdAM, Italy. F. Esposito is the holder of a Research Contract at the Universita degli Studi della Basilicata, funded\` by the European Union (NEXTGENERATIONEU - CUP C43C25000280007).

# Softness is Something You also Need! Why Compliance Helps Control: Information Metrics of a Passive Walker with Compliant Ankle SUPPLEMENTAL INFORMATION

Fabio Bonsignorio<sup>1</sup>, Elisabetta Barletta, Sorin Dragomir, Francesco Esposito<sup>2</sup> and Enrica Zereik<sup>3</sup>

Abstract— This document provides additional detailed information, in particular mathematical definitions and proofs, related to the paper. They are referenced in the paper text where they were considered suitable.

## I. CALCULATION OF W QUANTITIES

A. How to calculate the state space volume of the solutions of a Stochastic Partial Differential Equation (SPDE)

Understand the SPDE: analyze the variables, the stochastic components (like noise terms), boundary and initial conditions, and any other relevant parameters.

Numerical Simulation: analytical solutions are often impossible to obtain. You can use numerical methods like finite difference, finite elements, or spectral methods to discretize the SPDEs and simulate them over a range of parameters.

Monte Carlo Methods: Monte Carlo simulations can be particularly useful. By running a large number of simulations with different realizations of the stochastic components, you can obtain a distribution of the solution space.

Estimate State Space Volume: once you have a distribution of solutions from the Monte Carlo simulations, you can start estimating the volume of the state space by statistical techniques (e.g principal component analysis).

High–Dimensional Data Analysis: since the state space is likely high-dimensional, techniques from highdimensional data analysis might be necessary. Tools like manifold learning can help in understanding the structure of the solution space.

The interested reader can refer to [11] for more details.

## B. Usage of Reproducing Kernel Hilbert Spaces

Reproducing Kernel Hilbert Spaces (RKHS) are an impor tant mathematical concept and are widely used in machine learning, functional analysis, and quantum mechanics. They play a critical role in kernel methods, where they provide a way to map data into a higher-dimensional space where the data is more easily separated or manipulated. Differential equations typically define a flow in phase–space. Evaluating the volume of a phase–space usually involves integrating over this space.

The RKHS approach provides a weighted Bergman kernel function $K _ { \gamma }$ that maps from $\Omega \times \Omega$ (where Ω is the phase–space) to the set of complex numbers C, in correspondence to an admissible weight $\gamma \in A W ( \Omega )$ . The association of $K _ { \gamma }$ is an alternative measure to the Liouville measure, which allows the calculations of volumes based on complex analysis and complex geometry techniques. Renouncing real valued functions in favor of complex valued functions (of one or several complex variables) is coherent with the philosophy in [12] and [13]. For more details see [14] and [15].

## C. Practical Example

The passive walkers are not soft; however they exploit morphological computation, and they have power needs comparable to humans, as underlined in [16] and [17].

Why are their control programs so simple? Equation (1) in the main document allows to answer and to develop a working modeling framework for a new breed of humanoids,

![](images/edcb38155e195521c9e8dbc8b156684ad5241931cc2a2c10d9b810c9cd01554e.jpg)  
Fig. 1. The ‘Ecce’ robot.

like the venerable Ecce in Figure 1, developed in the EU project ECCEROBOT, [18].

Calculating the phase–space volume of an inverted pendulum involves considering the variables that describe its state and the constraints they are subject to.

State Variables. For an inverted pendulum, typical state variables include:

• θ: angle of the pendulum from the vertical (in radians);

• <sup>˙</sup>θ: pendulum angular velocity (in radians/s).

Phase–Space. The phase space is spanned by the above listed variables. ${ \mathrm { S o } } ,$ the phase–space point $( \theta , { \dot { \theta } } )$ describes the pendulum state.

Constraints. The inverted pendulum is subject to physical constraints such as conservation of energy (which determines the total energy of the pendulum) and the dynamics described by its motion equations (which relate θ and $\dot { \theta } )$

Phase–Space Volume. To calculate the phase–space volume, one needs to integrate over the accessible region in the $( \theta , { \dot { \theta } } )$ space that satisfies all the relevant constraints. The volume element in phase–space is $d \operatorname { v o l } ( g _ { B } )$ where $g _ { B } = \mathcal { K } ^ { * } g _ { \mathrm { F u b i n i - S t u d y } }$ and $\boldsymbol { \mathcal { K } } : \boldsymbol { \Omega } $ CP $\left[ L ^ { 2 } \dot { O } ( \Omega , \gamma ) \right]$ is a coherent state map (an antiholomorphic embedding).

Energy Constraint. The total energy E of the pendulum is typically given by $\begin{array} { r } { E = \frac 1 2 I \bar { \dot { \theta } } ^ { 2 } + m g h \cos ( \theta ) } \end{array}$ , where I is the moment of inertia, m is the mass, g is the acceleration due to gravity, and h is the height of the pendulum’s center of mass.

The accessible region $\begin{array} { r } { D = \{ ( \theta , \dot { \theta } ) \in [ 0 , \frac { \pi } { 2 } ] \times \mathbb { R } \ : } \end{array}$ $0 \leq \dot { \theta } \leq \sqrt { \textstyle { \frac { 2 g } { h } } } \big ( 1 - \cos \theta \big ) ^ { \dot { 1 } / 2 } \big \}$ in phase space is determined by the energy constraint $E = \mathrm { c o n s t a n t }$

Calculating the Volume. The phase–space volume V for a given energy E is then obtained by integrating over the allowed region in (θ, <sup>˙</sup>θ) space:

$$
V ( E ) = \int _ { D } d \mathrm { v o l } ( g _ { B } ) .
$$

In summary, calculating the phase–space volume of an inverted pendulum involves setting up the energy constraint, determining the allowed region in $( \theta , { \dot { \theta } } )$ space, and then integrating over that region to find the volume.

D. Classical States versus Coherent States for the Inverted Pendulum

Equation (1) in the main document introduces W as a geometric descriptor of the admissible phase–space region. For simple systems such as the harmonic oscillator, this quantity can be computed directly from the geometry of the corresponding energy ellipse. For realistic systems, however, the viable phase space may be curved, bounded by nonlinear constraints, and affected by stochastic or noisy dynamics.

A more general mathematical framework is therefore useful for defining and computing phase–space geometry. In this section we introduce such a framework based on complex geometry, reproducing kernel Hilbert spaces (RKHS), and coherent states.

The purpose of this construction is to replace the intuitive notion of “counting classical phase–space states” by a continuous, geometry-aware notion of volume. Each classical state is mapped to a coherent state in a Hilbert space using weighted Bergman kernels. This provides a rigorous geometric representation of the admissible phase– space support.

Importantly, the geometric volume of this support should not be identified with the full informational complexity of the dynamics. As developed in Section II, morphology may modify the geometry of the viable set, the probability measure induced over that set, or both. The RKHS framework allows these effects to be represented within a common mathematical setting.

1) Classical State: A classical state is the usual phase– space point

$$
z = ( \theta , { \dot { \theta } } ) ,\tag{1}
$$

which, for the inverted pendulum, corresponds physically to the pendulum angle and angular velocity. The admissible states form a domain

$$
D \subset \mathbb { R } ^ { 2 } .\tag{2}
$$

2) Coherent State: Instead of treating a state only as a point in the classical phase space, we associate each $z \in$ D with a coherent state in a Hilbert space. Let $K _ { \gamma }$ denote the weighted Bergman kernel associated with an admissible weight $\gamma .$ . The corresponding coherent-state map is

$$
{ \ K } : D \to \mathbb { C P } \big ( L _ { H } ^ { 2 } ( D , \gamma ) \big ) ,\tag{3}
$$

with

$$
\begin{array} { r } { \mathcal { K } ( z ) = [ K _ { \gamma } ( \cdot , z ) ] . } \end{array}\tag{4}
$$

Thus, each classical state is represented by a ray in a projective Hilbert space.

3) Why Use Coherent States?: Direct computation of volumes in nonlinear or curved phase–space domains can be difficult. The coherent-state construction equips the classical domain with a geometry induced by the RKHS representa tion.

In this representation, geometric quantities can be obtained from kernel functions and their associated metric and curvature structures. The basic strategy is therefore

classical state −→ coherent state −→ kernel-induced geometry.

(5)

This construction provides a geometry-aware representation of the viable state space and complements direct phase– space integration.

4) Overview of the Construction: The construction proceeds through the following steps.

First, define a weighted Hilbert space on the domain D. Let

$$
\gamma : D  [ 0 , + \infty )\tag{6}
$$

be an admissible weight and let $L _ { H } ^ { 2 } ( D , \gamma )$ denote the corresponding Hilbert space of square-integrable holomorphic functions. Its reproducing kernel is denoted by $K _ { \gamma } ( \zeta , z )$

Second, construct the coherent-state map

$$
\begin{array} { r } { K : D \to \mathbb { C P } \big ( L _ { H } ^ { 2 } ( D , \gamma ) \big ) . } \end{array}\tag{7}
$$

Third, the overlap between two coherent states provides the transition amplitude

$$
a ^ { 0 0 } ( \zeta , z ) = \left. \mathcal { K } ( z ) , \mathcal { K } ( \zeta ) \right. .\tag{8}
$$

Finally, the Liouville volume of the admissible domain can be related to the diagonal of an appropriately weighted Bergman kernel. For the family of weights introduced below,

$$
\mathrm { v o l } _ { L } ( D ) = \int _ { D } d \mu _ { L } = { \frac { C ( s ) } { 2 } } \int _ { D } K _ { \gamma _ { s } } ( \zeta , \zeta ) \gamma _ { s } ( \zeta ) d \zeta \wedge d { \bar { \zeta } } .\tag{9}
$$

Thus, the geometrical quantity $W ( D )$ can be evaluated through the kernel representation of the domain rather than only through direct coordinate-space integration.

5) Relation to the Mind–Body Nexus: The geometric construction above provides one component of the Mind– Body Nexus framework. Morphology determines, or contributes to determining, the viable geometry accessible to the physical system. The hybrid dynamics additionally induce a probability measure over that geometry.

The resulting hierarchy is therefore

$$
\boxed { \mathrm { M o r p h o l o g y } \longrightarrow \left( D _ { v } , W ( D _ { v } ) \right) \longrightarrow P \longrightarrow \left( \mu _ { P } , C _ { P } \right) \longrightarrow \mathcal { C } ( P ) }\tag{10}
$$

where $D _ { v }$ denotes the viable support and $W ( D _ { v } )$ is a zerothorder descriptor of its geometric size. The probability measure P describes how the dynamics populate that support, while $\mu _ { P }$ and $C _ { P }$ denote, respectively, its RKHS mean embedding and centered covariance operator.

The corresponding distributional complexity is

$$
\begin{array} { r } { \mathcal { C } ( P ) = \log \operatorname* { d e t } _ { F } \left( I + \gamma C _ { P } \right) . } \end{array}\tag{11}
$$

Consequently, morphological computation should not be identified exclusively with phase–space contraction. Morphology may alter the geometry of the viable set, modify the probability distribution induced over an unchanged support, or produce both effects. The relevant object is therefore more generally the metric–measure structure $( D _ { v } , P )$ rather than the scalar volume $W ( D _ { v } )$ alone.

This distinction provides the formal bridge between body mechanics and information-processing complexity used in the main paper and developed further in Section II of this Supplemental Information.

6) Formal Development: We now provide the mathematical details of the construction. A weight function is a measurable map

$$
\gamma : D  [ 0 , + \infty ) .\tag{12}
$$

Let $L ^ { 2 } ( D , \gamma )$ be the Hilbert space of square integrable functions $f : D \to \mathbb { C }$ satisfying

$$
\int _ { D } | f ( z ) | ^ { 2 } \gamma ( z ) d \mu ( z ) < \infty ,\tag{13}
$$

with inner product

$$
( f , g ) _ { \gamma } = \int _ { D } f ( z ) { \overline { { g ( z ) } } } \gamma ( z ) d \mu ( z ) ,\tag{14}
$$

where µ denotes Lebesgue measure on $\mathbb { R } ^ { 2 }$

Let $L _ { H } ^ { 2 } ( D , \gamma )$ denote the subspace of holomorphic functions in $L ^ { 2 } ( D , \gamma )$ . A weight γ is admissible if the evaluation functional

$$
\delta _ { z } : L _ { H } ^ { 2 } ( D , \gamma ) \to \mathbb { C } , \qquad \delta _ { z } ( f ) = f ( z ) ,\tag{15}
$$

is continuous and $L _ { H } ^ { 2 } ( D , \gamma )$ is a closed subspace of $L ^ { 2 } ( D , \gamma )$

Under these assumptions the evaluation functional is represented by the weighted Bergman kernel $K _ { \gamma } ( \zeta , z )$ , and $L _ { H } ^ { 2 } ( D , \gamma )$ is an RKHS.

The coherent-state map is then

$$
\mathcal { K } : D \to \mathbb { C P } \left( L _ { H } ^ { 2 } ( D , \gamma ) \right) , \qquad z \mapsto [ K _ { \gamma } ( \cdot , z ) ] .\tag{16}
$$

For classical states $z , \zeta \in D _ { : }$ , the associated transition probability amplitude is

$$
a ^ { 0 0 } ( \zeta , z ) = \left. \mathcal { K } ( z ) , \mathcal { K } ( \zeta ) \right. .\tag{17}
$$

Let

$$
\mathbb { B } = \left\{ z \in \mathbb { C } : | z | < 1 \right\}\tag{18}
$$

be the unit disk and let

$$
R : \mathbb { B }  D\tag{19}
$$

be a Riemann map. For $s > - 1$ , define

$$
\phi _ { s } ( \zeta ) = \left( 1 - | \zeta | ^ { 2 } \right) ^ { s } ,\tag{20}
$$

and obtain an admissible weight on D by pullback,

$$
\gamma _ { s } = \phi _ { s } \circ R ^ { - 1 } .\tag{21}
$$

The corresponding weighted Bergman kernel transforms under the conformal map according to the appropriate kernel transformation law. With the normalization adopted here, the Liouville volume can then be written in the form of Eq. (9).

The central point for the present paper is that this construction provides a rigorous geometric representation of the viable domain $D _ { v }$ and its associated volume $W ( D _ { v } )$ . The probability measure P induced by the physical dynamics is a distinct object. Section II makes this distinction explicit and introduces the corresponding distributional RKHS description.

## E. Application Example

The approach described in the preceding subsection is quite general. We illustrate it here using the inverted pendulum, for which less abstract approaches are also available. We detail the computational procedure step by step:

1) Map the domain D to the unit disk: since D is simply connected (no holes), there exists a conformal (Riemann) map $R : \mathbb { B }  D$ from the unit disk $\mathbb { B } = \{ z : | z | < 1 \}$ to the domain $D .$

2) Choose a weight on the disk and pull it back to D: we set a family of weights on the disk, $\phi _ { s } ( \zeta ) = ( 1 - | \zeta | ^ { 2 } ) ^ { s }$ then we define a weight on D by pulling back through R: $\gamma _ { s } = \phi _ { s } \circ R ^ { - 1 }$

3) Compute the weighted Bergman kernel on D: we start from the known kernel on the disk (closed form), then transform it via the mapping R. The transformation rule is $K _ { \gamma _ { s } } ( \zeta , z ) ~ = ~ R ^ { \prime } ( \zeta ) \overline { { { R ^ { \prime } ( z ) } } } ~ K _ { \phi _ { s } } ( R ( \zeta ) , R ( z ) )$ . Conceptually, the derivative terms correct for geometric distortion under the map.

4) Get volume from the kernel diagonal: the key computational shortcut is based on the expression vol(D) ∝ $\begin{array} { r } { \int _ { D } K _ { \gamma _ { s } } ( z , z ) \gamma _ { s } ( z ) d z } \end{array}$ , which holds up to constants like $C ( s )$ above and the chosen measure conventions.

5) Practical approximation (what you actually implement): since R is rarely analytically known, you will need to:

• approximate D by polygons $P _ { N }$

• compute polygon-to-disk maps via Schwarz–Christoffel methods (through one of the mentioned existing fast algorithms)

• get approximations $R _ { N }  R$

• build $\gamma _ { s , N }$ and $K _ { \gamma _ { s } }$ ,N

• integrate numerically to retrieve the volume.

Historical Note. Bergman kernels (unweighted, so that $\gamma \equiv 1 )$ are due to Steven Bergman [19], and an early account of their wide applicability is furnished in the monograph [20]. In 1952 Gaetano Fichera started a personal crusade (refer to [21]) against Bergman kernels, on the ground of their being “inessential” vis-a-vis to other methods in Mathematical Analysis. Bergman kernels fit into the general theory of RKHS [22], which was already available at the time Fichera’s paper [21] was written. The further growth of the use of kernel methods in the theory of PDEs (discussed e.g. in [23] and [24]), complex geometry (relative to Bergman metrics, see pages 367-371 of [25]) and approximation theory (in, e.g., [26] and [27]) suggest that the great Italian scientist was, at least once, abandoned by his unquestionable mathematical intuition.

## II. PHYSICAL INTERPRETATION: SUPPORT GEOMETRY AND DISTRIBUTIONAL COMPLEXITY

Compliance introduces internal energy-storage degrees of freedom that modify how the available mechanical energy is distributed between task variables and internal morphological variables. It is important, however, to distinguish the geometric support of the admissible task states from the probability measure induced on that support.

Let

$$
z = ( x , \eta ) ,\tag{22}
$$

where x denotes the task variables and η the compliant internal degrees of freedom. For analytical tractability, assume the separable Hamiltonian

$$
H ( z ) = H _ { x } ( x ) + H _ { \eta } ( \eta ) , \qquad H _ { \eta } ( \eta ) \geq 0 ,\tag{23}
$$

and define

$$
D _ { r } ( E ) = \{ x : H _ { x } ( x ) \leq E \} ,\tag{24}
$$

$$
D _ { c } ( E ) = \{ ( x , \eta ) : H _ { x } ( x ) + H _ { \eta } ( \eta ) \leq E \} .\tag{25}
$$

## A. Projection of the compliant energy domain

Proposition II.1 (Projected-support equivalence). Assume that the compliant subsystem possesses an admissible zero-energy state η<sub>0</sub> such that

$$
H _ { \eta } ( \eta _ { 0 } ) = 0 .\tag{26}
$$

Then

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) .\tag{27}
$$

$$
H _ { x } ( x ) + H _ { \eta } ( \eta ) \leq E .
$$

Proof. If $x \in \Pi _ { x } D _ { c } ( E )$ , there exists an η such that

Since $H _ { \eta } ( \eta ) \geq 0$ , it follows that $H _ { x } ( x ) \leq E$ , and therefore

$$
\Pi _ { x } D _ { c } ( E ) \subseteq D _ { r } ( E ) .
$$

Conversely, let $x \in D _ { r } ( E )$ . Choosing $\eta = \eta _ { 0 }$ gives

$$
H _ { x } ( x ) + H _ { \eta } ( \eta _ { 0 } ) = H _ { x } ( x ) \leq E ,
$$

and hence $( x , \eta _ { 0 } ) \in D _ { c } ( E )$ . Therefore

$$
D _ { r } ( E ) \subseteq \Pi _ { x } D _ { c } ( E ) .
$$

The two inclusions prove Eq. (27).

Thus, for the separable model considered here, compliance does not by itself contract the projected geometric support of the task variables. Its effect appears instead in the measure induced on that support, because the additional compliant coordinates redistribute the available energy.

## B. Energy redistribution under uniform sampling

Consider the local quadratic model

$$
H _ { x } = { \frac { 1 } { 2 } } a \theta ^ { 2 } + { \frac { 1 } { 2 } } b \omega ^ { 2 } , \qquad H _ { \eta } = { \frac { 1 } { 2 } } k \phi ^ { 2 } + { \frac { 1 } { 2 } } J \dot { \phi } ^ { 2 } .\tag{28}
$$

After rescaling the four coordinates, the compliant energy domain becomes a four-dimensional Euclidean ball,

$$
u _ { 1 } ^ { 2 } + u _ { 2 } ^ { 2 } + u _ { 3 } ^ { 2 } + u _ { 4 } ^ { 2 } \leq 2 E .\tag{29}
$$

For a point sampled uniformly in a d-dimensional ball of radius R,

$$
\mathbb { E } [ r ^ { 2 } ] = \frac { d } { d + 2 } R ^ { 2 } .\tag{30}
$$

For $d = 4$ and $R ^ { 2 } = 2 E$ , this gives

$$
\mathbb { E } [ r ^ { 2 } ] = \frac { 4 } { 6 } ( 2 E ) = \frac { 4 E } { 3 } .\tag{31}
$$

By rotational symmetry, each quadratic coordinate contributes one fourth of this expectation. Since $H _ { x }$ contains two of the four quadratic coordinates,

$$
\boxed { \mathbb { E } [ H _ { x } ] _ { \mathrm { c o m p l i a n t } } = \frac { E } { 3 } . }\tag{32}
$$

For comparison, uniform sampling inside the twodimensional rigid energy ellipse gives

$$
\boxed { \mathbb { E } [ H _ { x } ] _ { \mathrm { r i g i d } } = \frac { E } { 2 } . }\tag{33}
$$

Therefore,

$$
\mathbb { E } [ H _ { x } ] _ { \mathrm { c o m p l i a n t } } < \mathbb { E } [ H _ { x } ] _ { \mathrm { r i g i d } } ,\tag{34}
$$

even though

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) .\tag{35}
$$

The effect of compliance in this separable example is therefore not a contraction of the projected support, but a redistribution of probability mass toward lower-energy task states. This distinction motivates the distributional RKHS complexity measure introduced in the main paper.

## C. Distributional complexity

Let $P _ { r }$ and $P _ { c }$ denote the probability measures induced on the task space by the rigid and compliant systems, respectively. Although their supports may coincide,

$$
\operatorname { s u p p } ( P _ { r } ) = \operatorname { s u p p } ( P _ { c } ) = D _ { r } ( E ) ,\tag{36}
$$

the measures need not coincide:

$$
\begin{array} { r } { P _ { r } \neq P _ { c } . } \end{array}\tag{37}
$$

Let $C _ { P }$ denote the centered covariance operator associated with the RKHS embedding of $P ,$ and define

$$
\mathcal C ( P ) = \log \operatorname* { d e t } _ { F } ( I + \gamma C _ { P } ) .\tag{38}
$$

If

$$
0 \preceq C _ { P _ { c } } \preceq C _ { P _ { r } } ,\tag{39}
$$

then, by monotonicity of the eigenvalues of positive traceclass operators,

$$
\begin{array} { r } { \mathcal { C } ( P _ { c } ) \leq \mathcal { C } ( P _ { r } ) . } \end{array}\tag{40}
$$

This result makes explicit that the reduction of informational complexity is a property of the induced probability distribution, rather than a necessary consequence of a reduction in scalar phase–space volume.

Figure 2 summarizes the resulting hierarchy. In particular, morphological computation should not be identified exclusively with contraction of the viable support. Morphology may modify either the geometry of the viable set, the probability measure induced over that set, or both. The RKHS formulation provides a common representation of these effects.

## III. A SIMPLE NUMERICAL ESTIMATION OF PHASE–SPACE VOLUME AND COMPLEXITY

The viable phase–space volume W is estimated using a Monte–Carlo sampling procedure.

## A. Sampling Procedure

States are sampled uniformly inside a bounding box B in the full state space:

$$
z _ { i } \sim \mathcal { U } ( B ) .\tag{41}
$$

Each sample is accepted if it satisfies

$$
H ( z _ { i } ) \leq E ,\tag{42}
$$

and, for walking models, if the heel–strike reset produces a valid post–impact state.

The volume is estimated as:

$$
W \approx \mathrm { V o l } ( B ) { \frac { N _ { \mathrm { a c c } } } { N } } .\tag{43}
$$

## B. Kernel Complexity Estimation

Let $k ( \cdot , \cdot )$ denote a positive-definite kernel. In the numerical examples we employ the Gaussian kernel

$$
k ( x _ { i } , x _ { j } ) = \exp \biggl ( { - \frac { \| x _ { i } - x _ { j } \| ^ { 2 } } { 2 \sigma ^ { 2 } } } \biggr ) .
$$

The corresponding Gram matrix is

$$
G _ { i j } = k ( x _ { i } , x _ { j } ) .
$$

To avoid numerical singularities, a small regularization parameter $\lambda > 0$ is introduced and the complexity proxy is defined as:

$$
K ( X ) = \log \operatorname* { d e t } ( G + \lambda I ) ,\tag{44}
$$

which we use here as a finite-sample kernel complexity proxy. It should be distinguished from the operator-level distributional complexity C(P) = log det $_ { \ / F } ( I + \gamma C _ { P } )$ introduced in Section II and in the main paper. The latter is defined from the centered RKHS covariance operator of the induced probability measure, whereas $K ( X )$ is a numerical finite-sample descriptor computed directly from the Gram matrix.

## C. Effect of Compliance

Compliance introduces additional energy-storage degrees of freedom and can therefore modify the probability distribution induced over the task variables. Importantly, this need not imply a contraction of the projected task-space support. For the separable quadratic model of Section II, the projected rigid and compliant supports coincide,

$$
\Pi _ { x } D _ { c } ( E ) = D _ { r } ( E ) ,\tag{45}
$$

while the corresponding probability measures differ.

Consequently, changes in RKHS distributional complexity should be attributed to changes in the induced measure and its covariance spectrum, rather than, in general, to a reduction of the scalar projected phase–space volume.

## IV. FULL STATE–DEPENDENT MONTE–CARLO ESTIMATION

## A. Goal

Estimate the measure/volume of the post–impact viable set $D ^ { + } ( \gamma )$ (a subset of state space right after heel-strike) such that the walker can successfully complete the next step on slope $\gamma .$

## B. Post-impact viable set $D ^ { + }$

Right after heel-strike (the reset), we define the postimpact state:

$x ^ { + } = \left[ { q \atop \dot { q } ^ { + } } \right]$ (or higher-dimensional if you include more joints). 46)

(46)

We assume:

1) Post-impact configuration uncertainty / tolerance

$$
q ^ { + } \in [ - \alpha , + \alpha ] \pm d q\tag{47}
$$

![](images/c35843c85098ee4ac3e768d06226f957869825e14353218529c0427c1dd16b4c.jpg)  
Fig. 2. Hierarchy connecting morphology to information-processing complexity. Morphology determines the geometry of the viable state space $D _ { v }$ and can simultaneously modify the probability measure P induced by the dynamics over that space. The scalar volume $\dot { W } ( D _ { v } )$ provides a zeroth-order descriptor of support size, whereas the RKHS mean embedding µ<sub>P</sub> and covariance operator $C _ { P }$ retain information about the statistical organization of the viable dynamics. The resulting distributional complexity $\begin{array} { r l } { { \cal { C } } ( P ) = } & { { } } \end{array}$ log det $_ { F } ( I + \gamma \bar { C } _ { P } )$ therefore depends on the metric–measure structure of the viable dynamics rather than on scalar phase–space volume alone.

meaning: the post-impact configuration lies in a narrow band around the nominal reset configuration (for a rimless/compass-like step, nominal is typically $q ^ { + }$ ≈ $- \alpha .$ , we further assume a symmetric band).

2) Post-impact energy bound

$$
T ^ { + } \leq E ^ { + } ( \gamma )\tag{48}
$$

where $T ^ { + }$ is post-impact kinetic energy (e.g. $\begin{array} { r l } { T ^ { + } } & { { } = } \end{array}$ $\textstyle { \frac { 1 } { 2 } } I ( { \dot { q } } ^ { + } ) ^ { 2 }$ in 1-DOF), and $E ^ { + } ( \gamma )$ is an available energy budget that can depend on slope (or actuation limits, etc.). In practice this gives an admissible range for velocities:

$$
| \dot { q } ^ { + } | \leq \dot { q } _ { \mathrm { m a x } } ( \gamma ) \quad \mathrm { w i t h } \quad \dot { q } _ { \mathrm { m a x } } ( \gamma ) = \sqrt { \frac { 2 E ^ { + } ( \gamma ) } { I } } .\tag{49}
$$

## C. Monte-Carlo acceptance

We want the fraction of post-impact states in that admissible box that are actually viable.

Algorithm (state-dependent, because acceptance depends on dynamics and reset):

1) Sample N post-impact states $x _ { k } ^ { + }$ uniformly (or quasiuniformly) from:

$q ^ { + } \in [ q _ { 0 } - d q , q _ { 0 } + d q ]$ (or your symmetric interval) $\dot { q } ^ { + } \ \in \ [ - \dot { q } _ { \operatorname* { m a x } } ( \gamma ) , + \dot { q } _ { \operatorname* { m a x } } ( \gamma ) ]$ or only $\dot { q } ^ { + } ~ > ~ 0$ if physically required.

2) Propagate each sampled state forward through the continuous dynamics until the next heel-strike event.

3) Apply acceptance test:

• “Success” if the trajectory reaches the next heelstrike event without failure (no reversal, no fall, no forbidden constraint violation, etc.).

• Optionally include additional constraints (foot clearance, impact velocity bounds, etc.).

4) Estimate viable phase-space volume. If your sampling region has known volume Vol(B), then:

$$
\widehat W ( \gamma ) = \mathrm { V o l } ( B ) \ \frac { N _ { \mathrm { a c c } } } { N }\tag{50}
$$

The variance comes from a binomial estimator; you can attach confidence intervals if you want:

$$
\mathrm { S E } \approx \mathrm { V o l } ( \boldsymbol { B } ) \sqrt { \frac { \hat { p } ( 1 - \hat { p } ) } { N } } \quad \mathrm { w i t h } \ \hat { p } = N _ { \mathrm { a c c } } / N\tag{51}
$$

## D. “Exact heel-strike reset”

This means that when the next impact event happens, you use the exact discrete reset map:

$$
x _ { k + 1 } ^ { + } = \Delta \left( x _ { k + 1 } ^ { - } \right)\tag{52}
$$

rather than an approximate one.

## E. Exact heel–strike contraction in the reduced model

For the simplest passive walker considered in the main paper, the plastic heel–strike law is

$$
\omega ^ { + } = \cos ( 2 \alpha ) \omega ^ { - } .\tag{53}
$$

For α = 0.3 rad,

$$
\rho = \cos ( 2 \alpha ) = \cos ( 0 . 6 ) \simeq 0 . 8 2 5 3 3 6 .\tag{54}
$$

The kinetic energy associated with the angular velocity therefore changes according to

$$
\frac { T ^ { + } } { T ^ { - } } = \frac { \frac { 1 } { 2 } I ( \omega ^ { + } ) ^ { 2 } } { \frac { 1 } { 2 } I ( \omega ^ { - } ) ^ { 2 } } = \cos ^ { 2 } ( 2 \alpha ) \simeq 0 . 6 8 1 1 7 9 .\tag{55}
$$

Hence approximately 31.9% of the corresponding preimpact kinetic energy is dissipated by the plastic collision.

This energetic contraction must be distinguished from the geometric contraction of phase–space area. In the reduced state $x = ( \theta , \omega )$ , leg relabeling has unit absolute Jacobian, while the velocity component is multiplied by cos(2α). Consequently,

$$
\left| \operatorname* { d e t } D \Delta \right| = \left| \cos ( 2 \alpha ) \right| \simeq 0 . 8 2 5 3 3 6 .\tag{56}
$$

Thus the reset map alone contracts coordinate phase–space area by approximately 17.5%, whereas the kinetic-energy retention factor is approximately 0.681. Neither quantity should be identified directly with the ratio of viable phase– space volumes.

The latter additionally depends on the post-impact viability conditions. If $D ^ { - }$ denotes a pre-impact domain, the postimpact viable set may be written as

$$
D _ { v } ^ { + } = \left\{ \Delta ( x ^ { - } ) : x ^ { - } \in D ^ { - } , \quad \Delta ( x ^ { - } ) \ \right\} .\tag{57}
$$

Its volume must therefore be determined from the full hybrid dynamics, for example through the state-dependent Monte– Carlo procedure described above.

[1] F. Bonsignorio, “Preliminary considerations for a quantitative theory of networked embodied intelligence,” in Fifty years ofAI, M. Lungarella, F. Iida, J. Bongard, and R. Pfeifer, Eds. Berlin–Heidelberg, Germany: Springer, 2007.

[2] H. Touchette and S. Lloyd, “Information-theoretic approach to the study of control systems,” Physica A, vol. 331, pp. 140–172, 2003.

[3] R. Pfeifer and J. Bongard, How the body shapes the way we think: A new view of intelligence. Bradford Books, 2006.

[4] F. Bonsignorio, “Steps to a cyber-physical model of networked embodied anticipatory behavior,” in ABiALS 2008, LNAI, e. a. G. Pezzulo, Ed., vol. 5499. Berlin–Heidelberg, Germany: Springer, 2007, pp. 77–94.

[5] S. Balogh, G. Palla, P. Pollner, and D. Czegel, “Generalized entropies,´ density of states, and non-extensivity,” Sci. Rep., no. 15516, 2020.

[6] J. Pachter, Y. Yang, and K. Dill, “Entropy, irreversibility and inference at the foundations of statistical physics,” Nat Rev Phys, no. 6, pp. 382—-393, 2024.

[7] H. Goldstein, C. P. Poole, and J. L. Safko, Classical Mechanics, 3rd ed. San Francisco, CA: Addison-Wesley, 2002.

[8] V. I. Arnold, Mathematical Methods of Classical Mechanics, 2nd ed., ser. Graduate Texts in Mathematics. New York: Springer, 1989, vol. 60.

[9] A. Tozzi, “Fubini-study geometry of representation drift in highdimensional data,” arXiv preprint arXiv:2602.02596, 2026. [Online]. Available: https://arxiv.org/abs/2602.02596

[10] F. Bonsignorio, “On some information metrics of intelligent material systems,” in Proceedings of the ASME 2008 9th Biennial Conference on Engineering Systems Design and Analysis. Volume 2: Automotive Systems; Bioengineering and Biomedical Technology; Computational Mechanics; Controls; Dynamical Systems, vol. 2. ASME, 2008, pp. 711–718.

[11] B. Kruglikov and V. Lychagin, “Dimension of the solutions space of pdes,” arXiv preprint math/0610789, 2006.

[12] J. Ville, “Theory and applications of the notion of complex signal,” U.S. Air Force, Project RAND, 1958, translated from French by I. Selin, T-92, August 1, 1958.

[13] P. P. Bouboulis and S. Theodoridis, “Extension of wirtinger’s calculus to reproducing kernel hilbert spaces and the complex kernel

lms,” IEEE Transactions on Signal Processing, vol. 3, no. 59, 2011, arXiv:1006.3033v3 [cs.LG] 27 Nov 2010.

[14] E. Yildirim, A. Akgul, and M. Inc, “Reproducing kernel method for the¨ solutions of non-linear partial differential equations.” Applied Sciences, vol. 28, no. 1, pp. 80––86, 2021.

[15] E. Barletta, S. Dragomir, and F. Esposito, “Weighted bergman kernels and mathematical physics,” Axioms, vol. 9, no. 48, 2020, Special Issue Geometric Analysis and Mathematical Physics, Ed. by Sorin Dragomir].

[16] M. Gomes and A. Ruina, “A walking model with no energy cost,” Physical Review E, vol. 83, no. 3, pp. 032 901–1 – 032 901–4, 2011.

[17] P. Bhounsule, J. Cortell, A. Grewal, B. Hendriksen, J. Karssen, C. Paul, and A. Ruina, “Low-bandwidth reflex-based control for lower power walking: 65 km on a single battery charge,” The International Journal of Robotics Research, vol. 33, no. 10, pp. 1305–1321, 2014.

[18] ECCEROBOT (Embodied Cognition in a Compliantly Engineered ROBOT) Project website. Accessed November 11, 2026. [Online]. Available: http://eccerobot.org/

[19] S. Bergman and M. Schiffer, “Kernel functions and conformal mapping,” Compositio Mathematica, vol. 8, pp. 205–249, 1951.

[20] ——, Kernel functions and elliptic differential equations in mathematical physics. Courier Corporation, 2005.

[21] G. Fichera, “Sulla“kernel function”.” Bollettino dell’Unione Matematica Italiana, vol. 7, no. 1, pp. 4–15, 1952.

[22] N. Aronszajn, “Theory of reproducing kernels,” Transactions of the American mathematical society, vol. 68, no. 3, pp. 337–404, 1950.

[23] C. Yeung Lo, “Bergman’s kernel function on a class of elliptic partial differential equations in four variables,” Annali di Matematica Pura ed Applicata, vol. 79, pp. 93–105, 1968.

[24] G. Weill, “On bergman’s kernel function for some uniformly elliptic partial differential equations,” Proceedings of the American Mathematical Society, vol. 16, no. 6, pp. 1299–1304, 1965.

[25] S. Helgason, Differential geometry, Lie groups, and symmetric spaces. Academic press, 1979, vol. 80.

[26] E. Barletta and S. Dragomir, “Uniform approximation of holomorphic forms,” Complex Variables and Elliptic Equations, vol. 35, no. 4, pp. 359–366, 1998.

[27] E. Barletta, F. Bonsignorio, S. Dragomir, F. Esposito, and E. Zereik, “Holomorphic l<sup>2</sup> signals of several complex variables,” Complex Anal. Oper. Theory, vol. 20, no. 50, 2026.