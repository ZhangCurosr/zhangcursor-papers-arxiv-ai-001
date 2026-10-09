# AtomWorld-Mirror: Macro-Step World Modeling of Critical Evolution Backbones for Materials Dynamics

Ziming Pan<sup>1,7,∗</sup>, Ruge Zhang<sup>2,3,7,∗</sup>, Haozhi Han<sup>4,7,∗</sup>, Junkai Zhou<sup>5</sup>, Xingyuan Chen<sup>6</sup>, Yifeng Chen<sup>4</sup>, Yunquan Zhang<sup>2</sup>, Ting Cao<sup>7</sup>, Yunxin Liu<sup>7</sup>, and Kun Li<sup>7,†</sup>

<sup>1</sup>Yonsei University, Seoul, Republic of Korea   
<sup>2</sup>Institute of Computing Technology, Chinese Academy of Sciences, Beijing, China   
<sup>3</sup>University of Chinese Academy of Sciences, Beijing, China   
<sup>4</sup>School of Computer Science, Peking University, Beijing, China   
<sup>5</sup>Economics & Technology Research Institute, China National Petroleum Corporation, Beijing, China <sup>6</sup>Shenzhen Research Institute of Big Data, Shenzhen, China   
<sup>7</sup>Institute for AI Industry Research (AIR), Tsinghua University, Beijing, China   
<sup>∗</sup>Equal contribution. <sup>†</sup>Corresponding author.   
This work was supported by Tecorigin.

## Email: likun@air.tsinghua.edu.cn

Atomistic simulation is a fundamental tool for studying long-term materials evolution, from diffusion and defect dynamics to interfacial reactions and fracture. Yet conventional simulators typically advance at microscopic resolution, spending substantial computation on low-impact local updates before reaching structurally consequential states—an evolutionary-resolution bottleneck that limits long-horizon simulation. We propose AtomWorld-Mirror, a time-aware macro-step world model for the critical evolution backbone of atomic systems. For Step-Wise atomistic simulation, AtomWorld-Mirror distills short micro-event segments into physically reachable transitions between key states, jointly predicting sparse structural edits and accumulated physical time through latent macro-step dynamics. Local reachability, inventory conservation, and continuous-time consistency constrain each transition. By amortizing local atomic physics into a reusable latent macro model and replacing explicit micro-event replay with macro-step inference, this formulation provides a path toward substantially faster prediction of long-term materials evolution while preserving structural validity and time semantics. Across five atomic systems, spanning Cu-rich RPV steel irradiation aging, Cu–Zr metallic glass, and Li<sub>3</sub>N-based anti-perovskite solid electrolyte, macro-step inference delivers a speed up of 10<sup>3</sup> to 10<sup>4</sup> times over event-by-event simulation.

Project Page: https://atomworld-mirror.github.io Code: https://github.com/RSIScience/AtomWorld-Mirror

## 1 Introduction

Conventional atomistic simulation methods usually describe materials evolution through explicit time stepping or micro-event updates, e.g., kinetic Monte Carlo (KMC) and molecular dynamics (MD) (Alder & Wainwright, 1959; Bortz et al., 1975). This Step-Wise formulation treats different state changes uniformly, even though they contribute differently to long-horizon evolution. Some changes are transient fluctuations, whereas others accumulate into persistent structural progress. Figure 1 illustrates this mismatch in Cu-rich reactor pressure vessel (RPV) steel aging trajectories, where Cu-vacancy exchange events associated with key structural evolution occur sparsely and at seed-dependent times within the same fixed replay budget. As a result, standard simulators may spend substantial computation on low-impact intermediate states before reaching structurally consequential configurations (Voter et al., 2002; Soisson et al., 1996; Soisson & Fu, 2007). We characterize this mismatch as an evolutionary-resolution bottleneck. Overcoming it requires compressing low-impact intermediate states while retaining the sparse transitions that carry persistent structural progress. We refer to this retained sequence as the critical evolution backbone.

A rollout over this backbone must determine how far the system should advance while preserving both structural progress and the physical time accumulated in between. The underlying atomistic simulator resolves valid local events and their time costs, but it does not directly indicate which later state should be retained next. That state may lie many microscopic events ahead, and both its structural change and duration depend on the intervening path. Rather than resolving this path event by event, long-horizon modeling can compress it into a sequence of macro steps that predicts the next physically reachable backbone transition together with the time required to reach it.

To provide this capability, we propose AtomWorld-Mirror, a time-aware macro-step world model for atomistic evolution AtomWorld denotes the underlying atomistic simulator, which serves during training as a teacher that provides reference trajectories, local transition rules, and physical-time labels. Mirror distills these microscopic dynamics into a latent macro model that predicts reachable sparse edits and their accumulated physical duration directly from the current configuration At rollout, it evaluates candidate macro horizons, selects a legal transition, and advances both structure and clock without explicitly replaying the intervening microscopic events (Kingma & Welling, 2014; Hafner et al., 2019; 2020; 2025; Fichthorn & Weinberg, 1991; Chatterjee & Vlachos, 2007). This yields a compressed but physically constrained state-time rollout for long-horizon structural prediction. To provide this capability, we develop AtomWorld-Mirror, a distillation framework that distills conventional atomistic simulation methods into time-aware macro-step world models.

itiosnsionssit nionst ons(a) Event raster: key­event positions  
![](images/2fcbff3f56f0b51919e9026bef606c18b54f59e06d2339335edbdea698ad3498.jpg)

umu tiCumCuCCm l(b) Cumulative timing  
![](images/a7c4d80d8de7431592c082f95e7e4120d761f9a3406b2fff566f07926867c8f6.jpg)  
Figure 1: Random timing of key structural evolution within a fixed kinetic simulator replay budget. Each row is one kinetic simulator trajectory at a specified Cu density and random seed; ticks mark Cu-vacancy exchange events within a 1000-micro event budget, and the cumulative curves show how these events accumulate across the budget.

Such compression is useful only if each predicted macro transition remains consistent with the underlying atomistic dynamics, which leads to three physical constraints. (i) Local Reachability. AtomWorld-Mirror restricts structural prediction to candidate supports constructed from the local event rules, ensuring that edits remain within sites accessible to the underlying dynamics (ii) Inventory Conservation. The projection-closed sparse-edit formulation preserves material counts when predicted edits are converted into lattice states. (iii) Continuous-Time Consistency. Physical duration is modeled as a path-conditioned quantity because related macro endpoints can accumulate different waiting times along different microscopic paths (Gillespie, 1976; 1977; Fichthorn & Weinberg, 1991). Accordingly, teacher path summaries supervise duration modeling during training, while prior path latents provide the corresponding information during rollout.

Together, these components allow Mirror to replace microscopic replay with physically constrained macro evolution without discarding structural or temporal information. Across controlled atomistic simulations, AtomWorld-Mirror captures backbonelevel sparse state changes through physically valid edits, aligns path-conditioned time and energy predictions along teacher trajectories, and delivers a speed up of 10<sup>3</sup> to 10<sup>4</sup> times in a timed end-to-end diagnostic against event-by-event simulation.

Our contributions are threefold:

• We introduce a simulator-to-world-model distillation method that converts step-wise atomistic teacher simulators into time-aware macro-step models over critical evolution backbones.

• We propose AtomWorld-Mirror, a time-aware macro-step world model distilled from these teacher simulators to predict critical state-time transitions beyond step-wise microscopic replay.

• Across five atomic systems with KMC and MD teachers, AtomWorld-Mirror delivers physically valid rollouts and $1 0 ^ { 3 } – 1 0 ^ { 4 } \times$ speedup while preserving structural and temporal accuracy.

## 2 Related work

Fine-Grained Atomistic Dynamics. Atomistic simulators traverse material evolution through different microscopic representations. Molecular dynamics (MD) integrates atomic coordinates through short time steps (Verlet, 1967; Plimpton, 1995), whereas kinetic Monte Carlo (KMC) samples state-dependent events and advances a residence-time clock (Bortz et al., 1975; Fichthorn & Weinberg, 1991). High-performance implementations such as OpenKMC extend event-driven KMC to very large systems (Li et al., 2019). In RPV steel aging, neutron irradiation drives the formation of Cu-rich precipitates (Soisson et al., 1996; Soisson & Fu, 2007); vacancy-mediated KMC represents this process through explicit local events, transition rates, and physical-time labels.

Accelerated Atomistic Dynamics. Several lines of work reduce the cost of traversing these trajectories. Superbasin KMC aggregates recurrent low-barrier events within local basins and advances their effective dynamics (Fichthorn & Lin, 2013). Rate Scaling KMC reduces the rates of fast events to limit repetitive sampling and accelerate access to slower evolution (Lin et al., 2019). Adaptive KMC discovers transition mechanisms and constructs event tables during simulation, extending the accessible dynamics beyond a predefined event catalogue (Henkelman & Jónsson, 2001; Chatterjee & Vlachos, 2007). Learned potentials, graph and equivariant simulators, and physics-informed models amortize local force and interaction calculations across atoms (Behler & Parrinello, 2007; Bartók et al., 2010; Schütt et al., 2017; Batatia et al., 2022; Zhang et al., 2018; Raissi et al., 2019; Pfaff et al., 2021).

## 3 Problem Setup

We formalize this state-time compression problem starting from the microscopic trajectory of an atomistic simulator.

## 3.1 Micro-event resolution and key-state evolution

RPV steel aging makes the resolution mismatch explicit. Let $X _ { t }$ denote the atomic configuration after the t-th microscopic update and $\bar { T } _ { t }$ its physical clock. A conventional simulator resolves a fine-grained trajectory $\{ X _ { t } \} _ { t = 0 } ^ { N }$ , where each transition $X _ { t }  X _ { t + 1 }$ is a local integration step (Verlet, 1967; Plimpton, 1995; Thompson et al., 2022) or an atom-vacancy event (Bortz et al., 1975; Gillespie, 1977). Each update follows the simulator’s local physics, while long-term behavior concentrates in a much smaller set of structurally decisive states. The learning problem is to identify those states and retain the structural progress they carry.

We denote a sparse key-state subsequence by

$$
\mathcal { B } = ( X _ { n _ { 0 } } , X _ { n _ { 1 } } , . . . , X _ { n _ { M } } ) , \quad 0 = n _ { 0 } < n _ { 1 } < \dots < n _ { M } \leq N , \quad M \ll N ,\tag{1}
$$

and call it the critical evolution backbone. The retention criterion is system-dependent: RPV steel retains successive Cu-cluster growth and connectivity milestones, and MD cage-swap systems retain topology-change event signatures. The realized macro horizon is $k _ { m } = n _ { m + 1 } - n _ { m } \in \{ 1 , \dots , 1 0 2 4 \}$ , so each transition ends at the next important state and adapts to the current configuration.

## 3.2 Macro-step state-time objective

We train this predictive selection over $K = \{ 1 , \ldots , 1 0 2 4 \}$ . For each state, the next important state determines the realized macro horizon k and the number of teacher micro events compressed into the transition. The corresponding physical duration $\tau _ { t : t + k }$ is predicted jointly with the endpoint. A direct full-state objective is

$$
p _ { \theta } ( X _ { t + k } , \tau _ { t : t + k } \mid X _ { t } , k ) ,\tag{2}
$$

where $\tau _ { t : t + k } = T _ { t + k } - T _ { t }$ for a realized path. Most lattice sites remain unchanged over a macro step, making full-state prediction inefficient. We express the structural target as sparse edit labels over a physically reachable candidate set $\mathcal C _ { k } ( X _ { t } )$

$$
\Delta X _ { t : t + k } = \{ ( i , a _ { i } ) \} _ { i \in \mathcal { C } _ { k } ( X _ { t } ) } ,\tag{3}
$$

where $a _ { i }$ is the final edit label or occupancy state at candidate site i. Unchanged candidates retain their current occupancy, so the effective edit remains sparse. The horizon k determines how far the teacher dynamics can propagate and consequently changes the candidate set. The resulting objective is

$$
p _ { \theta } ( \Delta X _ { t : t + k } , \tau _ { t : t + k } \mid X _ { t } , \mathcal { C } _ { k } ( X _ { t } ) , k ) .\tag{4}
$$

This objective couples structural support, endpoint edits, and path duration. A macro transition is defined by their joint prediction.

## 3.3 KMC supervision of continuous time

In KMC, the current state $X _ { t }$ has possible events $\mathcal { E } ( X _ { t } ) = \{ e _ { 1 } , . . . , e _ { n } \}$ with rates $r _ { i } ( X _ { t } )$ (Bortz et al., 1975; Gillespie, 1976; Fichthorn & Weinberg, 1991). The total escape rate is

$$
R ( X _ { t } ) = \sum _ { i } r _ { i } ( X _ { t } ) ,\tag{5}
$$

and event selection and waiting time are

$$
p ( e _ { i } \mid X _ { t } ) = { \frac { r _ { i } ( X _ { t } ) } { R ( X _ { t } ) } } , \qquad \Delta t \sim \operatorname { E x p } ( R ( X _ { t } ) ) .\tag{6}
$$

KMC realizes a continuous-time Markov chain (CTMC). The normalized rates determine which local event occurs, and the total escape rate determines how long the system remains in its current state. Each micro event consequently advances both structure and physical time. Compressing several such events into one macro transition requires an endpoint edit and the time accumulated across the intervening path.

The expected waiting time at $X _ { t } \operatorname { i s } 1 / R ( X _ { t } )$ , and the sampled clock update is $T _ { t + 1 } = T _ { t } + \Delta t _ { t }$ . For a teacher segment of k events, define

$$
\tau _ { \mathrm { e x p } } ( t , k ) = \sum _ { j = 0 } ^ { k - 1 } \frac { 1 } { R ( X _ { t + j } ) } , \qquad \tau _ { \mathrm { r e a l } } ( t , k ) = T _ { t + k } - T _ { t } = \sum _ { j = 0 } ^ { k - 1 } \Delta t _ { t + j } .\tag{7}
$$

Mirror uses $\tau _ { \mathrm { e x p } }$ as its primary macro-time target, preserving the expected CTMC clock along the teacher path. The realized duration $\tau _ { \mathrm { r e a l } }$ records one stochastic draw and serves as an uncertainty diagnostic. Paths that reach similar endpoints through different rate environments can accumulate different durations, making macro time path-conditioned. KMC supplies all parts of this supervision directly: local event semantics, reachable trajectories, and continuous-time labels (Chatterjee & Vlachos, 2007).

## 4 AtomWorld-Mirror

AtomWorld-Mirror implements the macro-transition objective as a teacher-student world model for backbone-aware atomistic evolution. AtomWorld denotes the teacher-exposed evolution world, including configurations, local event support, transition rules, and physical clock. Mirror is the student-side macro-step world model over backbone state-time transitions. The teacher supplies short, locally valid state-time segments, and Mirror predicts reachable sparse edits, accumulated duration, and the next macro latent state for selected horizons. The primary instantiation uses atomistic KMC; other Step-Wise atomistic simulators that provide sparse structural states and physical-time supervision can expose the same teacher interface and be distilled into Mirror.

Mirror learns macro state-time transitions between consecutive states of the critical evolution backbone B of Eq. 1, distilled from KMC-generated segments. For each candidate macro horizon, it takes the current lattice configuration $X _ { t } ,$ , its reachable candidate set $\mathcal C _ { k } ( X _ { t } )$ , and state-derived context as input. It predicts a sparse lattice edit, the next latent state, the accumulated physical duration, and reward/energy-related quantities. The planner scores the resulting transitions, retains a legal candidate, and projects its edit before execution. The projected state becomes the input to the next macro step. One macro step maps

$$
\bigl ( X _ { t } , \mathcal { C } _ { k } ( X _ { t } ) , k \bigr ) \longmapsto \bigl ( \Delta \hat { X } _ { t : t + k } , \hat { z } _ { t + k } , \hat { \tau } _ { t : t + k } \bigr ) ,\tag{8}
$$

and rollout applies $\hat { X } _ { t + k } = X _ { t } \oplus \Delta \hat { X } _ { t : t + k }$ and repeats. Each training example contains an endpoint edit, a microscopic path summary, and two duration targets derived from a teacher segment.

The teacher KMC simulator supplies configurations, local reachable supports, microscopic path summaries, and continuous-time labels. Mirror maps the current lattice state and active candidate patch into latent representations, predicts horizon-conditioned macro dynamics, decodes a sparse edit within the candidate support, and predicts accumulated duration. Projection precedes the latent, reward, and duration losses, giving training and rollout the same projection-closed forward path. Figure 5 shows the pipeline, Table 2 in Appendix B consolidates component-level dimensions, inputs, and outputs, and Appendix A lists the data splits and filtering rules.

## 4.1 Teacher supervision

Given an initial configuration $X _ { t }$ , the teacher advances the underlying microscopic dynamics over a short segment of horizon $k \in \mathcal { K }$ , producing the local trajectory $\{ X _ { t + i } \} _ { i = 0 } ^ { k }$ (Fichthorn & Weinberg, 1991; Chatterjee & Vlachos, 2007). Each supervised macro sample contains

$$
( X _ { t } , k , X _ { t + k } , \Delta X _ { t : t + k } , \tau _ { \mathrm { e x p } } , \tau _ { \mathrm { r e a l } } , s _ { \mathrm { p a t h } } , \mathcal { C } _ { k } ) .\tag{9}
$$

Here K is the controlled Multi-K horizon set, $\Delta X _ { t : t + k }$ is the start-to-end sparse edit, $s _ { \mathrm { p a t h } }$ summarizes the microscopic path, and $\mathcal { C } _ { k }$ is the reachable candidate set. Rollout receives the current state $X _ { t } ,$ , candidate support $\mathcal C _ { k } ( X _ { t } )$ , horizon k, and

state-derived context. The endpoint $X _ { t + k }$ , sparse-edit target, path summary, and duration labels supervise training. The realized duration is

$$
\tau _ { \mathrm { r e a l } } = \sum _ { j = 0 } ^ { k - 1 } \Delta t _ { t + j } ,\tag{10}
$$

while the primary time target is the path-conditioned accumulated expected time

$$
\tau _ { \mathrm { e x p } } = \sum _ { j = 0 } ^ { k - 1 } \frac { 1 } { R ( X _ { t + j } ) } .\tag{11}
$$

We use $\tau _ { \mathrm { e x p } }$ as the main duration target because it reflects the expected CTMC clock. $\tau _ { \mathrm { r e a l } }$ remains an auxiliary stochastic target and time-uncertainty diagnostic.

The three physical commitments stated in Section 1 turn these supervised segments into Mirror rollouts: local reachability restricts edits to the teacher-derived candidate support, inventory conservation constrains the projection that converts latent predictions into lattice edits, and continuous-time consistency makes duration path-conditioned (Gillespie, 1976; 1977; Fichthorn & Weinberg, 1991). These commitments enter Mirror through latent dynamics, sparse-edit projection, and duration losses.

## 4.2 Mirror macro-step world model

Mirror is a macro-step world model developed from the DreamerV4 latent dynamics architecture (Hafner et al., 2025) and adapted to teacher-supervised atomistic state-time transitions. Its encoder combines a latent-variable representation with graph message passing and equivariant site features (Kingma & Welling, 2014; Gilmer et al., 2017; Satorras et al., 2021),

$$
z _ { t } = \operatorname { E n c } _ { \theta } ( X _ { t } , g _ { t } ) ,\tag{12}
$$

where $g _ { t }$ may contain local composition, defect statistics, rate summaries, vacancy information, or other global context. The active-patch encoder also produces site embeddings for $\mathcal C _ { k } ( X _ { t } )$ , coupling global latent dynamics with the local edit space. The same ${ \bar { X } } _ { t }$ can lead to different reachable endpoints, and each k defines a distinct admissible path family. AtomWorld-Mirro represents this variation with a horizon-conditioned path latent $h _ { t }$

During training, the posterior path latent conditions on the endpoint and teacher path summary,

$$
q _ { \phi } ( h _ { t } \mid z _ { t } , z _ { t + k } , s _ { \mathrm { p a t h } } , k ) ,\tag{13}
$$

which identifies the supervised transition. During inference, the model samples or selects from a prior

$$
p _ { \theta } ( h _ { t } \mid z _ { t } , g _ { t } , k ) .\tag{14}
$$

The macro dynamics then predicts a horizon-conditioned next latent,

$$
\hat { z } _ { t + k } = z _ { t } + F _ { \theta } ( z _ { t } , h _ { t } , k ) .\tag{15}
$$

A KL term transfers posterior information into the rollout prior. At rollout, every input is computed from the current state, state-derived context, and macro horizon.

## 4.3 Reachability-constrained sparse edits

The edit decoder operates on $\mathcal { C } _ { k } ( X _ { t } )$ , the reachable candidate support generated by the teacher dynamics. Candidate-restricted event supports are standard in spatial and adaptive KMC (Chatterjee & Vlachos, 2007; Henkelman & Jónsson, 2001), with related barrier and path-sampling context in NEB and time-reversal path-sampling work (Henkelman et al., 2000; Liu, 2023). For candidate site i, the decoder takes site embedding $u _ { i } ,$ patch context, predicted macro latent $\hat { z } _ { t + k }$ , path latent $h _ { t } .$ , and horizon $k ,$ and predicts change logit $\ell _ { i }$ and final type distribution $\pi _ { i }$

$$
( \ell _ { i } , \pi _ { i } ) = D _ { \theta } ( u _ { i } , c _ { k , i } , \hat { z } _ { t + k } , h _ { t } , k ) ,\tag{16}
$$

Let $m _ { i } = \mathbf { 1 } \{ x _ { t + k , i } \neq x _ { t , i } \}$ be the teacher change mask and $a _ { i }$ be the target final type. A compact sparse-edit loss is

$$
\mathcal { L } _ { \mathrm { e d i t } } = - \sum _ { i \in \mathcal { C } _ { k } ( X _ { t } ) } \left[ m _ { i } \log \sigma ( \ell _ { i } ) + ( 1 - m _ { i } ) \log ( 1 - \sigma ( \ell _ { i } ) ) + \beta m _ { i } \log \pi _ { i } ( a _ { i } ) \right] ,\tag{17}
$$

where $c _ { k , i }$ is the active-patch context, the first two terms learn sparse support, the last learns changed-site type, and $\beta = 1$ in the reported experiments. This candidate-restricted output space focuses prediction on sites that the local dynamics can affect and supplies the support for physically reachable vacancy-mediated edits (Soisson et al., 1996; Soisson & Fu, 2007).

Inventory conservation further requires the candidate patch to preserve material counts:

$$
\sum _ { i \in \mathcal { C } _ { k } ( X _ { t } ) } \mathbf { 1 } \{ \hat { x } _ { t + k , i } = c \} = \sum _ { i \in \mathcal { C } _ { k } ( X _ { t } ) } \mathbf { 1 } \{ x _ { t , i } = c \} , \qquad c \in \{ \mathrm { F e , C u , V a c } \} ,\tag{18}
$$

Outside the patch, sites are copied from the current configuration. Projection is implemented as paired vacancy–atom transport, selecting vacancy-to-atom and atom-to-vacancy pairs under the budget

$$
d _ { \mathrm { v a c } } \leq k , \qquad d _ { \mathrm { C u } } \leq k , \qquad d _ { \mathrm { v a c } } + d _ { \mathrm { C u } } \leq 2 k , \qquad | \Delta \hat { X } _ { t : t + k } | \leq 2 k ,\tag{19}
$$

where distances use the periodic BCC hop metric. Projection is applied before latent, reward, and duration losses are evaluated. The projected state is re-encoded and passed to the next macro step, aligning optimization with the state consumed during rollout.

## 4.4 Continuous-time duration modeling and loss

Duration is a core macro-transition output because stochastic simulation couples event selection and waiting time through the same transition rates (Gillespie, 1976; 1977; Fichthorn & Weinberg, 1991; Lin et al., 2019). We separate path-conditioned expected CTMC duration from realized sampled duration: $\tau _ { \mathrm { e x p } }$ is the main macro-time target, while $\tau _ { \mathrm { r e a l } }$ is auxiliary stochastic supervision.

The duration branch predicts expected macro time in log space,

$$
b _ { k } ( X _ { t } ) = \log k - \log R ( X _ { t } ) , \qquad \mu _ { \mathrm { e x p } } = b _ { k } ( X _ { t } ) + r _ { \theta } ^ { \tau } ( z _ { t } , \hat { z } _ { t + k } , h _ { t } , g _ { t } , \rho _ { t } , k ) ,\tag{20}
$$

where $b _ { k } ( X _ { t } )$ is the log start-state CTMC estimate $k / R ( X _ { t } ) , \rho _ { t }$ is edit/projection context, and $r _ { \theta } ^ { \tau }$ is a residual. The residual corrects the start-state estimate using rate changes accumulated along the teacher path. The expected-time head captures both the initial escape-rate scale and path-dependent dynamics. It also predicts log $\sigma _ { \mathrm { e x p } } .$ , and the main duration loss is the implementation’s log-space Gaussian negative log-likelihood:

$$
\mathcal { L } _ { \tau } = \log \sigma _ { \mathrm { e x p } } + \frac { 1 } { 2 } \left( \frac { \log ( \tau _ { \mathrm { e x p } } + \epsilon ) - \mu _ { \mathrm { e x p } } } { \sigma _ { \mathrm { e x p } } } \right) ^ { 2 } .\tag{21}
$$

Realized duration remains stochastic even under identical rates, so it is modeled as a conditional lognormal auxiliary target:

$$
\mu _ { \mathrm { r e a l } } = b _ { k } ( X _ { t } ) + r _ { \theta } ^ { \mathrm { r e a l } } ( z _ { t } , \hat { z } _ { t + k } , h _ { t } , g _ { t } , \rho _ { t } , k ) , \qquad \tau _ { \mathrm { r e a l } } \sim \mathrm { L o g N o r m a l } ( \mu _ { \mathrm { r e a l } } , \sigma _ { \mathrm { r e a l } } ) .\tag{22}
$$

Default training uses prior-side duration losses, matching the quantities available during rollout. We use smooth- $. L _ { 1 }$ regression for $\mathcal { L } _ { \mathrm { r e w a r d } }$ and $\mathcal { L } _ { \mathrm { l a t e n t } }$ . The projection consistency loss $\mathcal { L } _ { \mathrm { p r o j } }$ aligns the re-encoded projected state with the teacher endpoint latent, and $\mathcal { L } _ { \mathrm { r e a l } }$ is the lognormal negative log-likelihood of $\tau _ { \mathrm { r e a l } }$ . The implementation objective is

$$
\begin{array} { r } { \mathcal { L } = \lambda _ { \mathrm { e d i t } } \mathcal { L } _ { \mathrm { e d i t } } + \lambda _ { \mathrm { r e w a r d } } \mathcal { L } _ { \mathrm { r e w a r d } } + \lambda _ { \mathrm { l a t e n t } } \mathcal { L } _ { \mathrm { l a t e n t } } + \lambda _ { \mathrm { p r o j } } \mathcal { L } _ { \mathrm { p r o j } } } \\ { + \lambda _ { \mathrm { K L } } D _ { \mathrm { K L } } ( q _ { \phi } \parallel p _ { \theta } ) + \lambda _ { \tau } \mathcal { L } _ { \tau } + \lambda _ { \mathrm { r e a l } } \mathcal { L } _ { \mathrm { r e a l } } . } \end{array}\tag{23}
$$

The macro horizon is selected independently at each step from $\mathcal { K } = \{ 1 , \dots , 1 0 2 4 \}$ . The model encodes $X _ { t } ,$ , draws $h _ { t }$ from the prior, and decodes a projected sparse edit for each candidate k. Candidates violating reachability or inventory are discarded, and the surviving legal candidates are ranked by predicted structural progress per unit physical time:

$$
k ^ { \star } = \arg \operatorname* { m a x } _ { k \in \mathcal { K } _ { \mathrm { l e g a l } } } \ \frac { \widehat { r } _ { \theta } ( z _ { t } , \widehat { z } _ { t + k } , k ) } { \widehat { \tau } _ { t : t + k } + \epsilon } ,\tag{24}
$$

The selected edit is applied as $\hat { X } _ { t + k } = X _ { t } \oplus \Delta \hat { X } _ { t : t + k }$ , the duration $\scriptstyle { \hat { \tau } } _ { t : t + k }$ advances the clock, and the step repeats to form a backbone trajectory.

The backbone $\boldsymbol { B }$ of Eq. 1 remains stable across realized macro horizons and trajectory seeds. We measure sensitivity over $k \in \{ 1 , 2 , 4 , . . . , 1 0 2 4 \}$ micro steps on 16 held-out trajectories and compute the persistence ratio, defined as the fraction of backbone events surviving across horizons. Persistence stays above 0.98 throughout. Across trajectory seeds, the largest-Cucluster identity reaches 0.9066, showing that the persistent structural hierarchy remains stable under microscopic stochasticity.

## 5 Experimental design

## 5.1 Experimental scope

Across five atomic systems, Mirror maintains zero reachability violations and zero inventory violations, and delivers an end-to-end speed up of $1 0 ^ { 3 }$ to $1 0 ^ { 4 }$ times. A reachability violation is an edit outside the candidate set $\mathcal C _ { k } ( X _ { t } )$ , and an inventory violation is a projected patch that changes any species count; both are reported as the fraction of macro steps affected. The evaluation begins with binary RPV steel aging and extends to 3-element and 10-element RPV steels under the same KMC teacher. Cu–Zr metallic glass and $\mathrm { L i _ { 3 } N } .$ -based anti-perovskite solid electrolyte use MD as the teacher, with cage-swap topology events defining their macro-step boundaries. The primary experiments instantiate this interface with KMC, and Appendix D reports all per-system results, including the MD-teacher systems that reuse the same Mirror architecture and training loop with tuple construction adapted to topology-change events.

The RPV steel systems center the study on alloy aging, where Cu-rich precipitate formation under irradiation controls the degradation of nuclear-reactor pressure vessels (Soisson et al., 1996; Soisson & Fu, 2007; Li et al., 2019). Cu–Zr metallic glass extends the evaluation to an amorphous alloy with cage-swap dynamics and structural relaxation (Cheng & Ma, 2011) The $\mathrm { L i _ { 3 } N }$ -based anti-perovskite solid electrolyte extends it to ionic migration in a solid-electrolyte composition for lithium batteries (Cordier et al., 1989; Dembitskiy et al., 2025; Zhao & Daemen, 2012). Together, these systems test state-time macro modeling across composition, structural order, and simulator type, while the measured speed up supports longer aging, diffusion, and parameter-sweep studies.

AtomWorld-Mirror uses a state-time macro transition as its prediction unit. It identifies structurally decisive waypoints along the critical evolution backbone and distills segments from a traditional atomistic simulator teacher into sparse reachable edits, next latent states, and accumulated physical duration. Each transition retains the local support and inventory conditions required for a valid rollout, while macro-step inference bypasses explicit replay of intervening micro-events. We apply this teacher interface to event-driven KMC, MD-generated topology-change segments in Cu–Zr metallic glass (Cheng & Ma, 2011), and $\mathrm { L i _ { 3 } N } .$ -based anti-perovskite solid electrolyte (Cordier et al., 1989; Dembitskiy et al., 2025; Zhao & Daemen, 2012), as well as to other Step-Wise atomistic simulators.

(a) Valid Sparse Edits  
![](images/2a204aa76ed4b155446bb63a0cf72ed49b59253a524f45c917b2f079868b3e10.jpg)

(b) Segment Time Alignment  
![](images/388399538dfa5002e994dfa6adf983894af602cf0ef78388a97dda67c74575ae.jpg)

(c) Cumulative Time  
![](images/b305cfc8d0d1603d6ee3d7f4a27027fea29c58c412c9fb9117e4a37b0e134716.jpg)

(d) Structural Fidelity  
![](images/4abef073ee3123797a863569a20433666ae0f6f667c740459de0fea51ba668ea.jpg)  
Figure 2: Controlled macro-step validation against the teacher simulator. The panels show (a) cumulative structural edits, (b) single-segment expected-time alignment across temperatures, (c) long-trajectory cumulative expected time, and (d) cumulative correctly typed edits for Teacher and Mirror.

## 5.2 Event-driven instantiation

![](images/c7285c241c975c0f942d5d46a0bbdff2630dae99a3077f7a24df0deb1e7c7100.jpg)  
Figure 3: Cu-density and temperature ablation matrices for Multi-K teacher-probe rollout. Darker cells denote larger values of the corresponding failure metric.

We instantiate AtomWorld as an atom-vacancy KMC process to evaluate projected edit validity, expected-duration alignment, closed-loop rollout consistency, and timed macro-inference efficiency.

Unless otherwise stated, the controlled physical setup uses a $4 0 ^ { 3 }$ BCC lattice, Cu density 0.0134, vacancy density 0.0002, a 2NN event neighborhood, 16 defect-graph shells, and reward scale 10.0. The quantitative matrix uses adaptive $k \in$ $\{ 1 , \ldots , 1 0 2 4 \}$ and horizon-selected macro diagnostics. Figure 2 summarizes the controlled macro-step validation against the KMC teacher/reference.

Given these paired macro segments, the first evaluation separates raw prediction quality from projection-closed physical validity. Structural quality is summarized by the edit F1 score, the harmonic mean of precision and recall between predicted and teacher change masks over $\mathcal C _ { k } ( X _ { t } )$ , and by type accuracy, the fraction of teacher-changed sites whose predicted species matches ${ { a } _ { i } } .$ . The expected-time head is evaluated against the path-conditioned CTMC target $\tau _ { \mathrm { e x p } }$ using log-MAE, log-RMSE, log-correlation, and scale ratio:

$$
\log \mathrm { M A E } _ { \tau } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \left| \log \hat { \tau } _ { \mathrm { e x p } } ^ { ( n ) } - \log \tau _ { \mathrm { e x p } } ^ { ( n ) } \right| , \qquad \mathrm { s c a l e } _ { \tau } = \frac { 1 } { N } \sum _ { n = 1 } ^ { N } \frac { \hat { \tau } _ { \mathrm { e x p } } ^ { ( n ) } } { \tau _ { \mathrm { e x p } } ^ { ( n ) } } .\tag{25}
$$

Scale ratio tests systematic time bias, log-correlation tests relative variation across teacher segments, and log-RMSE weights the largest per-segment errors. The stochastic $\tau _ { \mathrm { r e a l } }$ head is evaluated separately as an auxiliary lognormal using NLL, probabilityintegral-transform diagnostics, and 68%/95% interval coverage, the fraction of segments whose realized duration falls inside the predicted interval. Appendix A reports training parameters, compute resources, and runtime settings.

## 5.3 Closed-loop validity and ablations

The model maintains an autonomous state $\hat { X } _ { m }$ by applying the previous projected sparse edit. The evaluated model trajectory is the closed-loop sequence $\hat { X } _ { 0 } \stackrel { k _ { 0 } } { \longrightarrow } \hat { X } _ { 1 } \stackrel { k _ { 1 } } { \longrightarrow } \cdots \stackrel { k _ { M - 1 } } { \longrightarrow } \hat { X } _ { M }$ , while each transition is scored against a KMC probe started from the current model state. This protocol evaluates whether the self-generated macro trajectory remains locally reachable, inventory-preserving, temporally calibrated, and physically plausible under local probes.

Figure 3 reports Multi-K teacher-probe ablation matrices across Cu concentrations 0.005 and 0.0134 and temperatures 263, 293, 333, and 373K. Shared ablation rows measure candidate-support validity, inventory preservation, path-conditioned duration, and the components that maintain editable states during rollout. These matrices use two non-learned references. The KMC teacher supplies the rate-governed state-time target, and the copy-state baseline keeps the lattice fixed while using the CTMC start-state duration from $\mathsf { \bar { R } } ( X _ { t } )$ . Removing local reachability measures candidate-support dependence, removing inventory conservation measures material-count preservation, and replacing learned duration with a start-state CTMC baseline measures the contribution of path-conditioned information.

![](images/beb9651edf4406440b98e87616afade89303001e1e0f9be39b0271e672f0e296.jpg)  
Figure 4: End-to-end timing diagnostic across lattice sizes and temperatures.

## 5.4 Long-horizon diagnostics and efficiency

Replacing explicit micro-event replay with macro-step inference delivers a speed up of $1 0 ^ { 3 }$ to $1 0 ^ { 4 }$ times across lattice size, temperature, and material system (Fig. 4, Appendix D, Appendix I). The timing includes candidate construction, projection, benchmark orchestration, and batched neural inference for one adaptive macro transition. The diagnostic measures replay reduction under reachable, inventory-preserving, and temporally calibrated edits, with the reference cost set by classical and KMC simulators (Plimpton, 1995; Thompson et al., 2022; Li et al., 2019).

Macro-step cost tracks the candidate patch defined by the event neighborhood. Inference cost stays near-constant in system size, and a model trained on $4 0 ^ { 3 }$ runs on larger configurations without retraining. Across Multi-K steps with k ranging from 1 to 1024, one macro step replaces 480 teacher micro events on average and advances the clock by 12.98 seconds of expected time; Appendix H reports this timing.

Capturing Cu-rich precipitate evolution in RPV steel requires following vacancy-mediated kinetics across large spatial domains and long physical timescales. We evaluate a 50-year autonomous closed-loop rollout of Mirror on the full $5 . \dot { 4 } \times 1 0 ^ { 1 0 }$ -atom RPV steel aging system across four temperatures in Appendix I.

The teacher-forced diagnostic measures state-conditioned time and reward alignment along a contiguous KMC trajectory $\{ X _ { t _ { m } } \} _ { m = 0 } ^ { M }$ , where $X _ { t _ { m } } \ { \xrightarrow { k _ { m } } } \ X _ { t _ { m + 1 } }$ . The readout accumulates predicted and reference expected times and the final

clock-scale ratio:

$$
\hat { T } _ { M } ^ { \mathrm { e x p } } = \sum _ { m = 0 } ^ { M - 1 } \hat { \tau } _ { \mathrm { e x p } , m } , \qquad T _ { M } ^ { \mathrm { e x p } } = \sum _ { m = 0 } ^ { M - 1 } \tau _ { \mathrm { e x p } , m } , \qquad \rho _ { T } = \frac { \hat { T } _ { M } ^ { \mathrm { e x p } } } { T _ { M } ^ { \mathrm { e x p } } + \epsilon } .\tag{26}
$$

## 6 Conclusion

We introduced a simulator-to-world-model distillation method that converts step-wise atomistic teachers into time-aware macro-step models over critical evolution backbones. AtomWorld-Mirror instantiates this framework by predicting reachable sparse edits and path-conditioned physical time while preserving material inventory. Across five atomic systems with KMC and MD teachers, it achieves physically valid rollouts and $1 0 ^ { 3 } – 1 0 ^ { 4 } \times$ speedups over event-by-event simulation.

## References

B. J. Alder and T. E. Wainwright. Studies in Molecular Dynamics. I. General Method. The Journal ofChemical Physics, 31(2): 459–466, 1959. doi: 10.1063/1.1730376.

Albert P. Bartók, Mike C. Payne, Risi Kondor, and Gábor Csányi. Gaussian Approximation Potentials: The Accuracy of Quantum Mechanics, Without the Electrons. Physical Review Letters, 104:136403, 2010. doi: 10.1103/PhysRevLett.104.136403.

Ilyes Batatia, Dávid Péter Kovács, Gregor N. C. Simm, Christoph Ortner, and Gábor Csányi. MACE: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields. In Advances in Neural Information Processing Systems, volume 35, pp. 11423–11436, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/ hash/4a36c3c51af11ed9f34615b81edb5bbc-Abstract-Conference.html.

Peter Battaglia, Razvan Pascanu, Matthew Lai, Danilo Jimenez Rezende, and Koray Kavukcuoglu. Interaction Networks for Learning About Objects, Relations and Physics. In Advances in Neural Information Processing Systems, volume 29, pp. 4502–4510, 2016. URL https://proceedings.neurips.cc/paper/2016/hash/ 3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html.

Simon Batzner, Albert Musaelian, Lixin Sun, Mario Geiger, Jonathan P. Mailoa, Mordechai Kornbluth, Nicola Molinari, Tess E Smidt, and Boris Kozinsky. E(3)-Equivariant Graph Neural Networks for Data-Efficient and Accurate Interatomic Potentials. Nature Communications, 13:2453, 2022. doi: 10.1038/s41467-022-29939-5.

Jörg Behler and Michele Parrinello. Generalized Neural-Network Representation of High-Dimensional Potential-Energy Surfaces. Physical Review Letters, 98:146401, 2007. doi: 10.1103/PhysRevLett.98.146401.

Troels Arnfred Bojesen. Policy-Guided Monte Carlo: Reinforcement-Learning Markov Chain Dynamics. Physical Review E, 98:063303, 2018. doi: 10.1103/PhysRevE.98.063303.

A. B. Bortz, M. H. Kalos, and J. L. Lebowitz. A New Algorithm for Monte Carlo Simulation of Ising Spin Systems. Journal of Computational Physics, 17(1):10–18, 1975. doi: 10.1016/0021-9991(75)90060-1.

Abhijit Chatterjee and Dionisios G. Vlachos. An Overview of Spatial Microscopic and Accelerated Kinetic Monte Carlo Methods. Journal ofComputer-Aided Materials Design, 14:253–308, 2007. doi: 10.1007/s10820-006-9042-9.

Yongqiang Cheng and Evan Ma. Atomic-level structure and structure–property relationship in metallic glasses. Progress in Materials Science, 56(4):379–473, 2011. doi: 10.1016/j.pmatsci.2010.12.002.

John D. Chodera and Frank Noé. Markov State Models of Biomolecular Conformational Dynamics. Current Opinion in Structural Biology, 25:135–144, 2014. doi: 10.1016/j.sbi.2014.04.002.

Kamal Choudhary, Kevin F. Garrity, Andrew C. E. Reid, Brian DeCost, Adam J. Biacchi, Angela R. Hight Walker, Zachary Trautt, Jason Hattrick-Simpers, A. Gilad Kusne, Andrea Centrone, et al. The Joint Automated Repository for Various Integrated Simulations (JARVIS) for Data-Driven Materials Design. npj Computational Materials, 6:173, 2020. doi: 10.1038/s41524-020-00440-1.

Gerhard Cordier, Axel Gudat, Rüdiger Kniep, and Albrecht Rabenau. LiCaN and ${ \mathrm { L i } } _ { 4 } { \mathrm { S r N } } _ { 2 } .$ Derivatives of the Fluorite and Lithium Nitride Structures. Angewandte Chemie International Edition in English, 28(12):1702–1703, 1989. doi: 10.1002/anie.198917021.

Stefano Curtarolo, Wahyu Setyawan, Gus L. W. Hart, Michal Jahnatek, Roman V. Chepulskii, Richard H. Taylor, Shidong Wang, Junkai Xue, Kesong Yang, Ohad Levy, et al. AFLOW: An Automatic Framework for High-Throughput Materials Discovery. Computational Materials Science, 58:218–226, 2012. doi: 10.1016/j.commatsci.2012.02.005.

Artem D. Dembitskiy, Innokentiy S. Humonen, Roman A. Eremin, Dmitry A. Aksyonov, Stanislav S. Fedotov, and Semen A. Budennyy. Benchmarking machine learning models for predicting lithium ion migration. npj Computational Materials, 11 (1):131, 2025. doi: 10.1038/s41524-025-01571-z.

Kristen A. Fichthorn and Yangzheng Lin. A Local Superbasin Kinetic Monte Carlo Method. The Journal ofChemical Physics, 138(16):164104, 2013. doi: 10.1063/1.4801869.

Kristen A. Fichthorn and William H. Weinberg. Theoretical Foundations of Dynamical Monte Carlo Simulations. The Journal ofChemical Physics, 95(2):1090–1096, 1991. doi: 10.1063/1.461138.

Thomas Garnier and Maylise Nastar. Coarse-Grained Kinetic Monte Carlo Simulation of Diffusion in Alloys. Physical Review B, 88:134207, 2013. doi: 10.1103/PhysRevB.88.134207.

Johannes Gasteiger, Janek Groß, and Stephan Günnemann. Directional Message Passing for Molecular Graphs. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum?id=B1eWbxStPH.

Daniel T. Gillespie. A General Method for Numerically Simulating the Stochastic Time Evolution of Coupled Chemical Reactions. Journal ofComputational Physics, 22(4):403–434, 1976. doi: 10.1016/0021-9991(76)90041-3.

Daniel T. Gillespie. Exact Stochastic Simulation of Coupled Chemical Reactions. The Journal of Physical Chemistry, 81(25): 2340–2361, 1977. doi: 10.1021/j100540a008.

Justin Gilmer, Samuel S. Schoenholz, Patrick F. Riley, Oriol Vinyals, and George E. Dahl. Neural Message Passing for Quantum Chemistry. In Proceedings ofthe 34th International Conference on Machine Learning, volume 70 of Proceedings ofMachine Learning Research, pp. 1263–1272. PMLR, 2017. URL https://proceedings.mlr.press/v70/gilmer17a.html.

Danijar Hafner, Timothy Lillicrap, Ian Fischer, Ruben Villegas, David Ha, Honglak Lee, and James Davidson. Learning Latent Dynamics for Planning from Pixels. In Proceedings ofthe 36th International Conference on Machine Learning, volume 97 of Proceedings ofMachine Learning Research, pp. 2555–2565. PMLR, 2019. URL https://proceedings.mlr.press/ v97/hafner19a.html.

Danijar Hafner, Timothy Lillicrap, Jimmy Ba, and Mohammad Norouzi. Dream to Control: Learning Behaviors by Latent Imagination. In International Conference on Learning Representations, 2020. URL https://openreview.net/forum? id=S1lOTC4tDS.

Danijar Hafner, Wilson Yan, and Timothy Lillicrap. Training Agents Inside of Scalable World Models. arXiv preprint arXiv:2509.24527, 2025. doi: 10.48550/arXiv.2509.24527.

Graeme Henkelman and Hannes Jónsson. Long Time Scale Kinetic Monte Carlo Simulations Without Lattice Approximation and Predefined Event Table. The Journal ofChemical Physics, 115(21):9657–9666, 2001. doi: 10.1063/1.1415500.

Graeme Henkelman, Blas P. Uberuaga, and Hannes Jónsson. A Climbing Image Nudged Elastic Band Method for Finding Saddle Points and Minimum Energy Paths. The Journal ofChemical Physics, 113(22):9901–9904, 2000. doi: 10.1063/1.1329672.

Anubhav Jain, Shyue Ping Ong, Geoffroy Hautier, Wei Chen, William Davidson Richards, Stephen Dacek, Shreyas Cholia, Dan Gunter, David Skinner, Gerbrand Ceder, et al. Commentary: The Materials Project: A Materials Genome Approach to Accelerating Materials Innovation. APL Materials, 1:011002, 2013. doi: 10.1063/1.4812323.

George Em Karniadakis, Ioannis G. Kevrekidis, Lu Lu, Paris Perdikaris, Sifan Wang, and Liu Yang. Physics-Informed Machine Learning. Nature Reviews Physics, 3:422–440, 2021. doi: 10.1038/s42254-021-00314-5.

Diederik P. Kingma and Max Welling. Auto-Encoding Variational Bayes. In International Conference on Learning Representations, 2014. doi: 10.48550/arXiv.1312.6114.

Thomas N. Kipf, Ethan Fetaya, Kuan-Chieh Wang, Max Welling, and Richard S. Zemel. Neural Relational Inference for Interacting Systems. In Proceedings of the 35th International Conference on Machine Learning, volume 80 of Proceedings of Machine Learning Research, pp. 2688–2697. PMLR, 2018. URL https://proceedings.mlr.press/v80/kipf18a. html.

Kun Li, Honghui Shang, Yunquan Zhang, Shigang Li, Baodong Wu, Dong Wang, Libo Zhang, Fang Li, Dexun Chen, and Zhiqiang Wei. OpenKMC: A KMC Design for Hundred-Billion-Atom Simulation Using Millions of Cores on Sunway TaihuLight. In Proceedings ofthe International Conferencefor High Performance Computing, Networking, Storage and Analysis, pp. 1–16. ACM, 2019. doi: 10.1145/3295500.3356165. Article 68, 16 pages.

Yen Ting Lin, Song Feng, and William S. Hlavacek. Scaling Methods for Accelerating Kinetic Monte Carlo Simulations of Chemical Reaction Networks. The Journal ofChemical Physics, 150(24):244101, 2019. doi: 10.1063/1.5096774.

Zhirong Liu. Accelerating Kinetics with Time-Reversal Path Sampling. Molecules, 28(24):8147, 2023. doi: 10.3390/ molecules28248147.

Siewert J. Marrink, H. Jelger Risselada, Serge Yefimov, D. Peter Tieleman, and Alex H. de Vries. The MARTINI Force Field: Coarse Grained Model for Biomolecular Simulations. The Journal ofPhysical Chemistry B, 111(27):7812–7824, 2007. doi: 10.1021/jp071097f.

Amil Merchant, Simon Batzner, Samuel S. Schoenholz, Muratahan Aykol, Gowoon Cheon, and Ekin D. Cubuk. Scaling Deep Learning for Materials Discovery. Nature, 624:80–85, 2023. doi: 10.1038/s41586-023-06735-9.

Albert Musaelian, Simon Batzner, Anders Johansson, Lixin Sun, Cameron J. Owen, Mordechai Kornbluth, and Boris Kozinsky. Learning Local Equivariant Representations for Large-Scale Atomistic Dynamics. Nature Communications, 14:579, 2023. doi: 10.1038/s41467-023-36329-y.

W. G. Noid, Jhih-Wei Chu, Gary S. Ayton, Vinod Krishna, Sergei Izvekov, Gregory A. Voth, Avisek Das, and Hans C. Andersen. The Multiscale Coarse-Graining Method. I. A Rigorous Bridge Between Atomistic and Coarse-Grained Models. The Journal ofChemical Physics, 128:244114, 2008. doi: 10.1063/1.2938860.

Shyue Ping Ong, William Davidson Richards, Anubhav Jain, Geoffroy Hautier, Michael Kocher, Shreyas Cholia, Dan Gunter, Vincent L. Chevrier, Kristin A. Persson, and Gerbrand Ceder. Python Materials Genomics (pymatgen): A Robust, Open-Source Python Library for Materials Analysis. Computational Materials Science, 68:314–319, 2013. doi: 10.1016/j. commatsci.2012.10.028.

Tobias Pfaff, Meire Fortunato, Alvaro Sanchez-Gonzalez, and Peter W. Battaglia. Learning Mesh-Based Simulation with Graph Networks. In International Conference on Learning Representations (ICLR 2021), 2021. URL https://openreview. net/forum?id=roNqYL0\_XP.

Steve Plimpton. Fast Parallel Algorithms for Short-Range Molecular Dynamics. Journal ofComputational Physics, 117(1): 1–19, 1995. doi: 10.1006/jcph.1995.1039.

Aneesur Rahman. Correlations in the Motion of Atoms in Liquid Argon. Physical Review, 136:A405–A411, 1964. doi: 10.1103/PhysRev.136.A405.

M. Raissi, P. Perdikaris, and G. E. Karniadakis. Physics-Informed Neural Networks: A Deep Learning Framework for Solving Forward and Inverse Problems Involving Nonlinear Partial Differential Equations. Journal ofComputational Physics, 378: 686–707, 2019. doi: 10.1016/j.jcp.2018.10.045.

Víctor Garcia Satorras, Emiel Hoogeboom, and Max Welling. E(n) Equivariant Graph Neural Networks. In Proceedings of the 38th International Conference on Machine Learning, volume 139 of Proceedings ofMachine Learning Research, pp. 9323–9332. PMLR, 2021. URL https://proceedings.mlr.press/v139/satorras21a.html.

John Schulman, Filip Wolski, Prafulla Dhariwal, Alec Radford, and Oleg Klimov. Proximal Policy Optimization Algorithms. arXiv preprint arXiv:1707.06347, 2017. doi: 10.48550/arXiv.1707.06347.

Kristof T. Schütt, Pieter-Jan Kindermans, Huziel Enoc Sauceda Felix, Stefan Chmiela, Alexandre Tkatchenko, and Klaus-Robert Müller. SchNet: A Continuous-Filter Convolutional Neural Network for Modeling Quantum Interactions. In Advances in Neural Information Processing Systems, volume 30, pp. 991–1001, 2017. URL https://proceedings.neurips.cc/ paper/2017/hash/303ed4c69846ab36c2904d3ba8573050-Abstract.html.

F. Soisson, A. Barbu, and G. Martin. Monte Carlo Simulations of Copper Precipitation in Dilute Iron-Copper Alloys During Thermal Ageing and Under Electron Irradiation. Acta Materialia, 44(9):3789–3800, 1996. doi: 10.1016/1359-6454(95) 00447-5.

Frédéric Soisson and Chu-Chun Fu. Cu-Precipitation Kinetics in α-Fe from Atomistic Simulations: Vacancy-Trapping Effects and Cu-Cluster Mobility. Physical Review B, 76:214102, 2007. doi: 10.1103/PhysRevB.76.214102.

Mads R. Sørensen and Arthur F. Voter. Temperature-Accelerated Dynamics for Simulation of Infrequent Events. The Journal of Chemical Physics, 112(21):9599–9606, 2000. doi: 10.1063/1.481576.

Hao Tang, Boning Li, Yixuan Song, Mengren Liu, Haowei Xu, Guoqing Wang, Heejung Chung, and Ju Li. Reinforcement Learning-Guided Long-Timescale Simulation of Hydrogen Transport in Metals. Advanced Science, 11(5):2304122, 2024. doi: 10.1002/advs.202304122.

Aidan P. Thompson, H. Metin Aktulga, Richard Berger, Dan S. Bolintineanu, W. Michael Brown, Paul S. Crozier, Pieter J. in ’t Veld, Axel Kohlmeyer, Stan G. Moore, Trung Dac Nguyen, et al. LAMMPS – A Flexible Simulation Tool for Particle-Based Materials Modeling at the Atomic, Meso, and Continuum Scales. Computer Physics Communications, 271:108171, 2022. doi: 10.1016/j.cpc.2021.108171.

Loup Verlet. Computer “Experiments” on Classical Fluids. I. Thermodynamical Properties of Lennard-Jones Molecules. Physical Review, 159:98–103, 1967. doi: 10.1103/PhysRev.159.98.

Arthur F. Voter. Parallel Replica Method for Dynamics of Infrequent Events. Physical Review B, 57:R13985–R13988, 1998. doi: 10.1103/PhysRevB.57.R13985.

Arthur F. Voter, Francesco Montalenti, and Timothy C. Germann. Extending the Time Scale in Atomistic Simulation of Materials. Annual Review ofMaterials Research, 32:321–346, 2002. doi: 10.1146/annurev.matsci.32.112601.141541.

Han Wang, Linfeng Zhang, Jiequn Han, and Weinan E. DeePMD-kit: A Deep Learning Package for Many-Body Potential Energy Representation and Molecular Dynamics. Computer Physics Communications, 228:178–184, 2018. doi: 10.1016/j. cpc.2018.03.016.

Logan Ward, Alexander Dunn, Alireza Faghaninia, Nils E. R. Zimmermann, Saurabh Bajaj, Qi Wang, Joseph Montoya, Jiming Chen, Kyle Bystrom, Maxwell Dylla, et al. Matminer: An Open Source Toolkit for Materials Data Mining. Computational Materials Science, 152:60–69, 2018. doi: 10.1016/j.commatsci.2018.05.018.

Tian Xie and Jeffrey C. Grossman. Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties. Physical Review Letters, 120:145301, 2018. doi: 10.1103/PhysRevLett.120.145301.

Richard J. Zamora, Arthur F. Voter, Danny Perez, Nandakishore Santhi, Susan M. Mniszewski, Sunil Thulasidasan, and Stephan J. Eidenbenz. Discrete Event Performance Prediction of Speculatively Parallel Temperature-Accelerated Dynamics. SIMULATION: Transactions of The Society for Modeling and Simulation International, 92(12):1065–1086, 2016. doi: 10.1177/0037549716674806.

Linfeng Zhang, Jiequn Han, Han Wang, Roberto Car, and Weinan E. Deep Potential Molecular Dynamics: A Scalable Model with the Accuracy of Quantum Mechanics. Physical Review Letters, 120:143001, 2018. doi: 10.1103/PhysRevLett.120. 143001.

Yusheng Zhao and Luke L. Daemen. Superionic conductivity in lithium-rich anti-perovskites. Journal ofthe American Chemical Society, 134(36):15042–15047, 2012. doi: 10.1021/ja305709z.

## A Training details and parameters

<table><tr><td>Data / Teacher</td><td>Value</td><td>Student / Optim.</td><td>Value</td></tr><tr><td>Teacher</td><td>atom-vacancy KMC</td><td>Student</td><td>Mirror macro-step world model</td></tr><tr><td>Lattice</td><td> $4 0 ^ { 3 }$  BCC</td><td>Candidates</td><td> $\mathcal { C } _ { k } ( X _ { t } ) ; 8 \mathrm { ~ V ~ }$  seeds</td></tr><tr><td>Cu / V density</td><td>0.0134 / 0.0002</td><td>Path input</td><td>teacher path summary</td></tr><tr><td>Temperature</td><td>300 K</td><td>Latents</td><td>global 32× 16; patch/path 64/32</td></tr><tr><td>Event support</td><td>2NN hops</td><td>Epochs / batch</td><td>120 / 32</td></tr><tr><td>Train / val</td><td>2000 / 400 segments per k</td><td>Optimizer</td><td>Adam, lr 10−4</td></tr><tr><td>Split policy</td><td>independent envs</td><td>Weight decay</td><td> $1 0 ^ { - 5 }$ </td></tr><tr><td>Filtering</td><td>no-op skipped</td><td>Expected / realized time</td><td>1.0 / 0.25</td></tr><tr><td>Time targets</td><td>Texp main;  $\tau _ { \mathrm { r e a l } }$  aux</td><td>Edit / reward / latent / proj.</td><td>1.0 / 0.5 / 0.5 / 0.5</td></tr><tr><td>Aux. losses</td><td>proj. / reward / priors</td><td>KL / prior edit / prior latent</td><td>0.05 / 0.25 / 0.25</td></tr><tr><td>Compute</td><td>NVIDIA A100 GPUs</td><td>Runtime</td><td>35.5 min train; 23.3 / 47.1 s eval</td></tr></table>

Table 1: Training and evaluation configuration.

The reported training and evaluation runs use NVIDIA A100 GPUs. Timing values are taken from persisted run logs for the reported configuration.

## B Implementation details by component

![](images/46e17accc47702b55f8a0fc9ba47c36a0ef8ae593e00e10cfd13cc022417faed.jpg)  
Figure 5: AtomWorld-Mirror architecture. The teacher supplies segments with reachable candidate supports, path summaries, and continuous-time labels. Mirror encodes the configuration and active patch, draws a horizon-conditioned path latent from the posterior during training and the prior during rollout, predicts the macro latent, and decodes a sparse edit, accumulated duration, and reward/energy quantity. Inventory projection precedes the downstream losses, so the projected state consumed during rollout is also the state used for optimization.

<table><tr><td>Component</td><td>Input</td><td>Output</td><td>Rollout-time</td></tr><tr><td>Graph encoder</td><td>Xt, global context  $g _ { t }$ </td><td>global latent  ${ z } _ { t } ~ ( 3 2 \times 1 6 )$  , site embeddings  $u _ { i }$ </td><td>yes</td></tr><tr><td>Candidate builder</td><td>Xt, 2NN event support,  $k$ </td><td> $\mathcal { C } _ { k } ( X _ { t } )$  from 8 vacancy seeds</td><td>yes</td></tr><tr><td>Path posterior  $q _ { \phi }$ </td><td> $z _ { t } , z _ { t + k } , s _ { \mathrm { p a t h } } , k$ </td><td>path latent  $h _ { t }$  (dim 32)</td><td>no (training only)</td></tr><tr><td>Path prior pθ</td><td> $z _ { t } , g _ { t } , k$ </td><td>path latent  $h _ { t }$  (dim 32)</td><td>yes</td></tr><tr><td>Macro dynamics  $F _ { \theta }$ </td><td> $z _ { t } , h _ { t } , k$ </td><td> $\mathrm { r e s i d u a l } { \mathrm { t o } } \hat { z } _ { t + k }$ </td><td>yes</td></tr><tr><td>Edit decoder  $D _ { \theta }$ </td><td> $u _ { i } .$  patch context (dim 64),  $\hat { z } _ { t + k } , h _ { t } .$ </td><td>change logit  $\ell _ { i } ,$  type distribution  $\pi _ { i }$ </td><td>yes</td></tr><tr><td>Inventory projection</td><td> $k$  raw edit,  ${ \mathcal { C } } _ { k } ,$  budget 2k</td><td> $\mathrm { o v e r } \ \mathrm { \{ F e , C u , V a c \} }$   $\mathrm { p r o j e c t i o n – c l o s e d } \ \Delta \hat { X } _ { t : t + k }$ </td><td>yes</td></tr><tr><td>Duration head</td><td> $z _ { t } , \hat { z } _ { t + k } , h _ { t } , g _ { t }$  , edit context  $\rho _ { t } , k$ </td><td> $\mu _ { \mathrm { e x p } } , \log \sigma _ { \mathrm { e x p } } \mathrm { o n } \mathrm { t o p } \mathrm { o f } b _ { k } ( X _ { t } )$ </td><td>yes</td></tr><tr><td>Realized-time head</td><td>same as duration head</td><td> $\mu _ { \mathrm { r e a l } } , \sigma _ { \mathrm { r e a l } } ~ ( \mathrm { l o g n o r m a l } )$ </td><td>yes</td></tr><tr><td>Reward head</td><td> $z _ { t } , \hat { z } _ { t + k } , k$ </td><td> $\hat { r } _ { \theta } ~ ( \mathrm { r e w a r d s c a l e 1 0 . 0 } )$ </td><td>yes</td></tr><tr><td>Horizon selector</td><td> $\{ \hat { r } _ { \theta } , \hat { \tau } \} _ { k \in \mathcal { K } _ { \mathrm { l e g a l } } }$ </td><td> $k ^ { \star } \log \mathrm { E q } . 2 4$ </td><td>yes</td></tr></table>

Table 2: Component-level implementation details. Rollout-time availability marks whether a component’s inputs are computable from $X _ { t }$ and k alone.

## C Macro-step pseudocode

Algorithm 1 restates the rollout forward path of Section 4 as executable pseudocode. Colored comments mark the three stages the implementation separates: latent encoding of the configuration and active patches, per-horizon prediction followed by inventory projection, and horizon selection with clock advance. Every quantity consumed inside the loop is computable from the current state and the horizon alone, the contract recorded in the rollout-time column of Table 2.

Algorithm 1 Mirror macro-step rollout over the critical evolution backbone. Colored comments mark encoding, projection  
closed prediction, and horizon selection with clock advance.   
Require: configuration $X _ { 0 } ,$ clock $T _ { 0 } ,$ trained Mirror $\theta ,$ horizon set $\mathcal { K } = \{ 1 , \ldots , 1 0 2 4 \}$ , candidate builder $\mathcal { C } _ { k } ( \cdot )$   
Ensure: backbone trajectory B   
1: $t \gets 0 ; m \gets 0 ; \check { X _ { t } } \gets \check { X } _ { 0 } ; T \gets T _ { 0 } ; \mathcal { B } \gets [ ( X _ { 0 } , T _ { 0 } ) ]$   
2: while the rollout budget remains do   
3: // encode $z _ { t } \gets \mathrm { E n c } _ { \theta } ( X _ { t } , g _ { t } ) ;$ site embeddings $u _ { i }$ per site   
4: $\kappa _ { \mathrm { l e g a l } }  \emptyset$   
5: for $\breve { k } \in$ K do   
6: $\mathcal { C } _ { k } \gets ($ CandidateSupport $( X _ { t } , k ) ;$ construct active-patch context $c _ { k , i }$   
7: $h _ { k } \sim p _ { \theta } ( h \mid z _ { t } , g _ { t } , k ) ; \hat { z } _ { t + k }  z _ { t } + F _ { \theta } ( z _ { t } , h _ { k } , k )$   
8: $( \ell _ { i } , \pi _ { i } ) \gets D _ { \theta } ( u _ { i } , c _ { k , i } , \hat { z } _ { t + k } , h _ { k } , k )$ over $i \in \mathcal { C } _ { k }$   
9: $\Delta \hat { X } _ { k } \gets \mathrm { D e c o d e } ( \ell , \pi ) ; ~ \#$ inventory projection   
10: $\Delta \hat { X } _ { k } \gets \mathrm { P r o j e c t } ( \Delta \hat { X } _ { k } , \mathcal { C } _ { k } , 2 k )$ ▷ paired vacancy–atom transport   
11: $\hat { r } _ { k } \gets \mathrm { R e w a r d } _ { \theta } ( z _ { t } , \hat { z } _ { t + k } , k ) ; \hat { \tau } _ { k } \gets \mathrm { D u r a t i o n } _ { \theta } ( z _ { t } , \hat { z } _ { t + k } , h _ { k } , g _ { t } , k )$   
12: if $\Delta \hat { X } _ { k }$ satisfies reachability and inventory then add k to $\kappa _ { \mathrm { l e g a l } }$ else discard k end if   
13: end for   
14: $\mathbf { i f } \ K _ { \mathrm { l e g a l } } = \varnothing$ then record the stop reason and break   
15: // horizon selection $k ^ { \star } \gets \arg \operatorname* { i n a x } _ { k \in { \mathcal { K } _ { \mathrm { l e g a l } } } } \hat { r } _ { k } / ( \hat { \tau } _ { k } + \epsilon )$   
16: $X _ { t } \gets X _ { t } \oplus \Delta \hat { X } _ { k ^ { \star } } ; T \gets T + \hat { \tau } _ { k ^ { \star } } ; t \gets t + k ^ { \star } ; m \gets m + 1 ;$ append $( X _ { t } , T )$ to B   
17: end while   
18: return B

## D Generality across material systems and teacher types

The four additional systems extend the three physical commitments across chemistry, bonding character, and teacher type. RPV steel keeps the atom-vacancy KMC teacher of Section 4.1 and raises the element count to 3 (Fe–Cu–Ni) and 10 (adding Mn, Cr, Al, Si, P, Co, Mo). Cu–Zr metallic glass and the $\mathrm { L i _ { 3 } N } .$ -based anti-perovskite solid electrolyte use molecular dynamics, with a macro step ending at a cage-swap topology event and its duration given by the elapsed MD time between consecutive events. The MD systems carry no vacancy species. A snap-to-lattice bijection places one atom at each reference site, conserving inventory by construction; species-level edits appear as paired swaps, the direct analog of KMC Cu–vacancy transport.

Each table reports the teacher as a reference alongside AtomWorld-Mirror. The decoder restricts edits to $\mathcal { C } _ { k }$ , and the projection conserves species counts. The time columns report log-MAE and the per-segment scale ratio of Eq. 25.

## D.1 RPV steel, multi-component (KMC teacher)

<table><tr><td>System</td><td>Model</td><td>violation% ·F1</td><td>log-MAE · scale</td></tr><tr><td>RPV 3-element</td><td>KMC</td><td>0%· 1.00</td><td>0.00 · 1.00</td></tr><tr><td></td><td>Mirror</td><td>0%· 0.77</td><td>0.21·1.05</td></tr><tr><td>RPV 10-element</td><td>KMC</td><td>0% · 1.00</td><td>0.00 · 1.00</td></tr><tr><td></td><td>Mirror</td><td>0%· 0.78</td><td>0.32·1.06</td></tr></table>

Table 3: RPV steel: physical validity and time calibration at 3 and 10 elements.

![](images/9a3e35c84ad2fb0471ac13a2f0c404954650d271a7d125cf370a03e4b8b688fc.jpg)

![](images/a5d78bd0dda25a92a5658cbb9df23fe35d007678f9d7d6bb11e4374af124f2e2.jpg)  
Figure 6: End-to-end speed up over the KMC teacher for the RPV steel systems at 263, 293, 333, and 373 K. Bars show runtime and the line shows speed up.

Increasing the RPV composition from 3 to 10 elements roughly doubles the speed up. Added chemistry raises the teacher’s per-event cost, while macro-step cost remains tied to the candidate patch.

## D.2 Cu–Zr metallic glass (MD teacher)

<table><tr><td>Model</td><td>violation%·F1</td><td>log-MAE · scale</td></tr><tr><td>MD teacher</td><td>0%· 1.00</td><td>0.00 · 1.00</td></tr><tr><td>Mirror</td><td>0%· 0.84</td><td>0.26 · 1.10</td></tr></table>

Table 4: Cu–Zr metallic glass: physical validity and time calibration.

![](images/6e54dd0a0fc31e6d013369e34f4ea87a348cdfa709fdf8c5127ea91fd7d6db8c.jpg)  
Figure 7: End-to-end speed up over the MD teacher for the Cu–Zr metallic glass across temperature and system size.

Cu–Zr provides the largest speed up in this study, with edit F1 0.84 and duration scale 1.10. The MD teacher resolves femtosecond integration steps between rare cage swaps, allowing each macro step to replace a long stretch of integration. Higher temperature increases cage-swap frequency and shortens the interval covered by one macro step.

## D.3 Li N-based anti-perovskite solid electrolyte (MD teacher)

<table><tr><td>Model</td><td>violation%·F1</td><td>log-MAE·scale</td></tr><tr><td>MD teacher</td><td>0%· 1.00</td><td>0.00 · 1.00</td></tr><tr><td>Mirror</td><td>0%·0.71</td><td>0.27 · 1.07</td></tr></table>

Table 5: $\mathrm { L i _ { 3 } N } .$ based anti-perovskite solid electrolyte: physical validity and time calibration.

![](images/482feb7590c188faa4a8098faca34e4a1945b50387992e3aa985fd8dba41feb6.jpg)  
Figure 8: End-to-end speed up over the MD teacher for the $\mathrm { L i _ { 3 } N } .$ -based anti-perovskite solid electrolyte across temperature and system size.

Mirror reaches edit F1 0.71 on the electrolyte, with duration scale 1.07. Ionic migration produces well-separated topology events, keeping the duration target sharp.

## E Separating the macro-step formulation from learned dynamics

We isolate the macro-transition formulation and its three physical constraints through three matched comparisons.

The first comparison holds the latent backbone fixed while removing the macro-step structure. A Dreamer-style variant omits reachability, inventory, and continuous-time constraints. A direct endpoint regressor predicts $X _ { t + k }$ over the full lattice without candidate restriction, path latent, or duration head, reusing the main model’s graph encoder for representation parity.

<table><tr><td>Model</td><td>Physically valid sparse edits</td><td>log-MAE · scale</td><td> $\rho _ { T }$ </td></tr><tr><td>Teacher KMC</td><td>violation 0% · F1 1.00</td><td>0.00 · 1.00</td><td>1.00</td></tr><tr><td>Dreamer-style (no constraints)</td><td>reach 45.2% · inv 59.2%</td><td>4.08· 0.35</td><td>0.35</td></tr><tr><td>Endpoint regression</td><td>change-F1 0.42 · type-acc 0.45</td><td>N/A</td><td>N/A</td></tr><tr><td>AtomWorld-Mirror</td><td>violation 0% · F1 0.91</td><td>0.13 · 1.00</td><td>1.01</td></tr></table>

Table 6: Constraint and formulation baselines against AtomWorld-Mirror on the shared RPV steel aging held-out split (Cu 0.0134, 300 K, $k \in \{ 1 , \ldots , 1 0 2 4 \} ,$ ). The table quantifies the contributions of reachability, inventory conservation, and duration modeling.

At matched computational cost, the constrained macro-step formulation delivers reachable edits, inventory preservation, and calibrated physical time. AtomWorld-Mirror matches the Dreamer-style variant within 0.5% in wall-clock time, raises reachable edits from 45.2% to 100%, and raises the cumulative clock ratio from 0.35 to 1.01. AtomWorld-Mirror reaches change-F1 0.91 with path-conditioned duration modeling, compared with 0.42 for the endpoint regressor, which has no duration output.

## F Teacher-forced long-trajectory protocol

At segment m, from teacher state $X _ { t _ { m } }$ , the model predicts a projected macro candidate. When several horizons are legal, the planner selects one using Eq. 24. The teacher advances the matching microscopic dynamics to provide the reference next state, reward, expected time, and realized time. The model contributes $( \Delta \hat { X } _ { m } , \hat { \tau } _ { \mathrm { e x p } , m } , \hat { r } _ { m } )$ , and the evaluator reports cumulative expected-time curves, selected-horizon histograms, stop reasons, and cumulative energy-error summaries. The teacher-forced diagnostic uses the teacher state as input at every segment, isolating duration-scale error, horizon-selection bias, and reward misalignment from autonomous rollout drift. The per-segment scale ratio of Eq. 25 is the time-calibration summary reported in the per-system tables of Appendix D. The cumulative clock ratio $\rho _ { T }$ of Eq. 26 is reported in Table 6 and Table 8.

## G Adaptive horizon selection

At each macro step, the selector ranks legal horizons after projection. The resulting histogram records local configuration difficulty through selected horizons and projection stop events. When every candidate is rejected, the segment is recorded with its stop reason.

## H Macro-step inference over $k \in \{ 1 , \ldots , 1 0 2 4 \}$

Macro-step inference cost is set by the candidate patch defined by the event neighborhood. Cost per macro step stays near-constant in the total number of atoms, and a checkpoint trained at $4 0 ^ { 3 }$ applies to larger configurations without retraining.

We measure the prior forward pass plus inventory projection on held-out active patches drawn from KMC trajectories over the operating range $\mathbf { \bar { \boldsymbol { k } } } \in \{ 1 , \dots , 1 \mathbf { \bar { 0 } } 2 4 \}$ . One macro step covers 480 teacher micro events on average and advances the physical clock by 12.98 seconds of accumulated expected time. Table 7 reports representative measurements within this range.

<table><tr><td>Horizon k</td><td>128</td><td>256</td><td>512</td><td>1024</td></tr><tr><td>Coverage</td><td>0.889</td><td>0.906</td><td>0.780</td><td>0.649</td></tr></table>

<table><tr><td>Aggregate measurement</td><td>Value</td></tr><tr><td>Samples / batch size</td><td>96/32</td></tr><tr><td>Wall time per pass</td><td>134.7 s</td></tr><tr><td>Throughput</td><td>0.713 segments/s</td></tr><tr><td>Mean micro events per macro step Mean expected duration per macro step</td><td>480.0 12.98 s</td></tr></table>

Table 7: Measured macro-step inference within the operating range $\overline { { k \in \{ 1 , \dots , 1 0 2 4 \} } }$ on NVIDIA A100 GPUs. The table reports representative horizons; coverage is the fraction of attempted segments that yield a usable macro sample after terminal and no-op filtering.

Longer horizons replace more micro-event replay per macro step. Coverage changes from 0.889 at k = 128 to 0.649 at k = 1024, making the efficiency-coverage relationship explicit.

## I 54-billion-atom RPV steel aging benchmark

We evaluate a 50-year autonomous closed-loop rollout of Mirror on the full $5 . 4 \times 1 0 ^ { 1 0 }$ -atom RPV steel aging system at 1.34 at.% Cu concentration across four temperatures. OpenKMC runs on 5.2 million cores of the Sunway TaihuLight supercomputer, whose many-core processors are based on SW26010, while Mirror runs on NVIDIA A100 GPUs. The benchmark records physical validity, cumulative time alignment, and end-to-end speed up against OpenKMC.

<table><tr><td>Model</td><td>Physically valid edits</td><td>Single-segment time</td><td>Cumulative clock</td></tr><tr><td>OpenKMC</td><td>violation 0% · F1 1.00</td><td>log-MAE 0.00 ·scale 1.00</td><td> $\rho _ { T }$  1.00</td></tr><tr><td>Mirror</td><td>violation 0% · F1 0.91</td><td>log-MAE 0.27 · scale 1.17</td><td> $\rho _ { T } \mathbf { \nabla } \mathbf { 0 } . 8 9$ </td></tr></table>

Table 8: Physical validity and time alignment at the ${ \overline { { 5 . 4 \times 1 0 ^ { 1 0 } } } }$ -atom scale in the RPV steel aging benchmark.

At the $5 . 4 \times 1 0 ^ { 1 0 }$ -atom scale, Mirror retains physically valid edits with F1 0.91 and reaches a cumulative clock ratio $\rho _ { T } = 0 . 8 9$

![](images/529b083055a2a6b58ef8d1ca506e388b7e011a6938c76528b8ce16286de1d793.jpg)  
Figure 9: End-to-end speed up of Mirror over OpenKMC for the $5 . 4 \times 1 0 ^ { 1 0 }$ -atom RPV steel aging benchmark over a 50-year physical timescale. Bars show runtime and the line shows speed up.

## J RPV steel aging visualization

Figure 10 shows Cu-cluster aging in a vacancy-mediated KMC trajectory on a $4 0 ^ { 3 }$ BCC lattice. The full boxes show the simulated lattice, and the partial boxes enlarge a representative sub-volume to resolve local cluster growth. Colors encode Cu-cluster sizes computed on the full periodic BCC lattice from the teacher configuration.

## K Limitations and future work

AtomWorld-Mirror targets long-horizon materials evolution in regimes where explicit microscopic replay consumes most of the computational budget before structurally consequential states are reached. The study evaluates this setting with controlled simulator teachers and establishes extensions to additional teacher types and materials systems. Future work will extend the framework to broader simulator-teacher combinations and optimize distributed deployment.

The five-system evaluation spans binary and multi-component RPV steel aging, Cu–Zr metallic glass, and $\mathrm { L i _ { 3 } N } .$ -based antiperovskite solid electrolyte across KMC and MD teachers. The controlled RPV steel aging setting establishes the primary edit benchmark, and the MD systems demonstrate transfer of the state-time interface across bonding character and teacher type. The current operating range reaches $k = 1 0 2 4$ . Extending candidate-support construction and conservation projection to systems with larger species inventories, together with topology criteria for continuously evolving structures, defines the next methodological step.

## L Expanded related-work discussion

Materials simulation spans several representations of physical evolution. Coarse-grained molecular dynamics, coarse-grained KMC, Monte Carlo, and Markov state models reduce atomic, event, or state resolution through coarse variables, aggregated events, or metastable-state representations (Marrink et al., 2007; Noid et al., 2008; Garnier & Nastar, 2013; Chodera & Noé, 2014). Classical molecular dynamics advances atomic configurations through short integration steps (Alder & Wainwright, 1959; Rahman, 1964; Verlet, 1967; Plimpton, 1995; Thompson et al., 2022). High-throughput materials databases (Jain et al., 2013; Curtarolo et al., 2012) and structure/property analysis toolchains (Ong et al., 2013; Ward et al., 2018; Choudhary et al., 2020) standardize structures, computed properties, and analysis workflows. These abstractions reduce computational cost. AtomWorld-Mirror adds a predictive backbone representation that selects future structural waypoints and assigns the accumulated physical duration of each sparse transition.

![](images/59e6183a9ab58da2240786f75adb212e253a2af1a56d49bd2bb858ca6edd8c6a.jpg)  
Figure 10: Cu-cluster visual evolution from initial to final states in an RPV steel aging simulation. Each point denotes a Cu atom in the teacher-simulated lattice. Colors use a shared red–pink-purple–blue scale for the actual 1NN/2NN-connected Cu cluster size $C _ { i } ,$ with red denoting single-Cu clusters and blue denoting the maximum cluster size in this trajectory.

Learning-based materials simulation develops reusable models of energies, forces, local structures, and interacting-system dynamics (Merchant et al., 2023). Neural-network and Gaussian approximation potentials (Behler & Parrinello, 2007; Bartók et al., 2010), graph neural networks for molecules and materials (Schütt et al., 2017; Xie & Grossman, 2018; Batatia et al., 2022; Gilmer et al., 2017; Gasteiger et al., 2020), and equivariant force fields (Batzner et al., 2022; Musaelian et al., 2023) amortize local physical computations across atoms and molecules. Deep potential molecular dynamics and related packages turn these learned local models into scalable dynamical simulators (Zhang et al., 2018; Wang et al., 2018). Physics-informed learning (Raissi et al., 2019; Karniadakis et al., 2021) and graph-network simulators (Battaglia et al., 2016; Kipf et al., 2018; Pfaff et al., 2021) extend learned dynamics toward differential-equation, mesh-based, and interacting-system settings. AtomWorld-Mirror complements these approaches with a sparse macro state-time prediction target that explicitly represents backbone transitions and accumulated physical duration.

Step-Wise atomistic simulation provides the continuous-time teacher used in our experiments. KMC samples local events from physical rates and advances a residence-time clock (Bortz et al., 1975; Gillespie, 1977; Fichthorn & Weinberg, 1991), supplying local event semantics, reachable trajectories, and continuous-time labels (Chatterjee & Vlachos, 2007). High-performance KMC systems such as OpenKMC use engineered parallel execution to reach very large systems (Li et al., 2019). Established KMC accelerators include superbasin KMC and adaptive or self-learning KMC, which consolidate repeated events or construct transition mechanisms during simulation (Fichthorn & Lin, 2013; Henkelman & Jónsson, 2001). Parallel-replica, temperatureaccelerated, rate-scaling, and policy-guided methods further alter event traversal or event selection (Voter, 1998; Sørensen & Voter, 2000; Zamora et al., 2016; Lin et al., 2019; Schulman et al., 2017; Tang et al., 2024; Bojesen, 2018). These methods operate on event-level traversal, event aggregation, or trajectory selection. AtomWorld-Mirror operates at the macro-transition level: the simulator teacher supplies the state-time reference, and Mirror learns sparse backbone transitions with accumulated physical duration. speed up follows from this predictive representation of atomistic dynamics.

Rate Scaling KMC reduces the sampling frequency of fast event channels by scaling their rates, allowing a fixed event budget to cover more of the slower evolution (Lin et al., 2019). Event selection and waiting times are then computed from the scaled rates, so the resulting clock follows the modified process; agreement with the original kinetics depends on the scaling regime and the observable being evaluated. This approach retains explicit event-level updates and targets the cost of repeatedly sampling fast dynamics. AtomWorld-Mirror changes the prediction unit to a reachable macro edit with a jointly predicted duration, amortizing the intervening event sequence into a learned state-time transition.

Table 9 organizes the comparison by the quantity changed and the physical clock preserved. Trajectory accelerators modify the accepted event sequence. Analytical accelerators retain event-level execution and aggregate fast events inside basins. Ratescaling methods retain event-level execution while modifying fast-channel rates and their associated waiting-time distribution. AtomWorld-Mirror predicts a sparse edit between backbone states, assigns its accumulated duration as a path-conditioned quantity, and enforces reachability and inventory at every macro step. Structural and temporal quantities remain explicit at the macro level.

<table><tr><td>Family</td><td>What it changes</td><td>Physical clock</td><td>Examples</td></tr><tr><td>Trajectory accelerators</td><td>the accepted event sequence</td><td>not directly traceable once event order is altered</td><td>RL/PPO policies for KMC, neural surrogate event samplers (Schulman et al., 2017; Tang et al., 2024; Bojesen, 2018)</td></tr><tr><td>Analytical accelerators</td><td>rates or grouping of fast events; event-level execution retained</td><td>preserved, every effective event still resolved</td><td>superbasin KMC, adaptive KMC (Fichthorn &amp; Lin, 2013; Henkelman &amp; Jónsson, 2001)</td></tr><tr><td>Rate-scaling accelerators</td><td>fast-event rates; event-level execution retained</td><td>sampled from scaled rates; fidelity to original kinetics depends on scaling regime</td><td>Rate Scaling KMC (Lin et al., 2019)</td></tr><tr><td>AtomWorld-Mirror</td><td>the prediction unit: sparse macro edit between backbone states</td><td>predicted explicitly as path-conditioned accumulated duration</td><td>this work</td></tr></table>

Table 9: Where AtomWorld-Mirror sits among KMC acceleration families.