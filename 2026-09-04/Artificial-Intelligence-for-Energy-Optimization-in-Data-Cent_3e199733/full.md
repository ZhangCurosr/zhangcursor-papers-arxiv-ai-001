# Artificial Intelligence for Energy Optimization in Data Centers: Closing the Optimizer–Load Loop

Mohammed Basharath Ullah Hyderabad, India mohdbasharath8@gmail.com

Summaiya Unnisa Begum Hyderabad, India summaiyaunisa@gmail.com

Mohammed Nadeem Ullah Hyderabad, India mdnadeemullah@gmail.com

## ABSTRACT

Data centers are increasingly optimized by artificial intelligence and, at the same time, increasingly loaded by it. The literature treats these as two unrelated problems: control studies model workload as an exogenous arrival process, while sustainability studies model infrastructure as a fixed multiplier. We screen roughly 194 papers retrieved through a documented protocol, code 63 of them, and report what the coding shows. Of 28 primary control-oriented studies, 18 are validated in simulation alone and 5 reach physical hardware or a production facility; none account for water withdrawal, and none account for embodied carbon. Reported savings intervals across four technique families overlap almost completely, which means the field cannot presently rank its own methods. Ten recurring gaps are scored for consequence and tractability, and we set out CLEAR-DC, a framework coupling a control-policy branch to a workload-demand branch through an explicit elasticity term, reads out net rather than direct benefit, and emits a schemaconformant record covering energy, carbon, water, embodied share and validation venue. The framework is an architectural and methodological proposal, not a trained system; the contribution we defend empirically is the corpus analysis and the reporting schema derived from it. Coding sheet, derived statistics and all result artifacts: https://github.com/Kimalice/AI-for-Energy-Optimization-in-Data-Centers-Closing-the-Optimizer-Load-Loop Keywords: Data Center Energy Efficiency, Deep Reinforcement Learning, Cooling Optimization, Workload Scheduling, Carbon-Aware Computing, Sustainable AI, Rebound Effects, Systematic Review, Reporting Standards

## I. INTRODUCTION: WHY EFFICIENCY CLAIMS NEED A CLOSED LOOP

Digital infrastructure has become an energy sector in its own right. Early accounting placed data center electricity at roughly one percent of world consumption once cooling and power distribution were included [20], and subsequent bottomup modelling showed that demand for services grew far faster than electricity use because efficiency improvements absorbed most of the difference [50], [31]. That reassurance is now under pressure. Forecasts diverge sharply depending on assumptions about hardware trends and service growth [21], with recent scenario work spanning roughly 1,800 to 5,000 TWh by midcentury [10], while a competing long-run analysis argues that computing’s share of world electricity peaked over a decade ago and has since stabilised [40]. The disagreement is not merely academic: it determines whether efficiency research is a marginal optimisation or a decarbonisation priority.

Against that backdrop, artificial intelligence entered the field first as a modelling tool and then as a controller. Neural network models learned facility behaviour directly from operational sensor data and predicted power usage effectiveness with error small enough to guide set-point selection [13]. Deep reinforcement learning then closed the loop, replacing handtuned control logic with learned policies [26], and the control surface widened to span information technology and cooling subsystems jointly [42], [43], to enforce thermal safety during learning [55], [4], and to follow carbon intensity across time and geography [41], [46]. Reported savings are substantial and consistent enough that several reviews now treat the approach as established [18], [25], [39].

## A. A Commissioning Scenario

Consider an operator commissioning a learned controller in a facility that also hosts large-scale model training. The controller is expected to reduce facility energy while the workload it manages is growing at a rate no cooling optimisation can offset, drawing water in a basin that may be stressed, and imposing power transients the facility’s own models do not represent. Three obstacles frustrate any attempt to evaluate whether that deployment is a net improvement:

1) Open-Loop Framing: Control studies treat workload as an exogenous stochastic process and report the energy saved at fixed demand. Sustainability studies treat infrastructure efficiency as a fixed coefficient applied to a demand trajectory determined elsewhere. Neither models the path by which efficiency reduces the effective unit cost of computation and thereby influences how much computation is purchased. A systematic review spanning the artificial intelligence lifecycle arrives at a compatible position from a different direction, reporting that gains realised at one stage are repeatedly offset by responses elsewhere in the system, so that stage-level improvements do not aggregate into lifecycle ones [38]. That offset is named there but not formalised, and it is the path it leaves open that this paper tries to close.

![](images/ca3545a21ce33b84e21e4ac5782922b213217c026015549b14e55253e6c85f39.jpg)  
Fig. 1: Drivers of the data center footprint. The optimisation layer controls the facility, but the demand it makes cheaper feeds back into workload — a path left unmodelled in the studies surveyed here.

2) Validation Deficit: Learned controllers are overwhelmingly evaluated in simulation. A review of reinforcement learning for building climate control found that under a quarter of studies reached a real building [2], and a systematic review focused on data centers names the absence of real-time validation as a principal gap [18]. A dedicated evaluation across algorithms, tasks, system dynamics and transfer settings reports that leading methods remain sensitive to hyperparameters, reward specification and operating scenario, with constraint violations and sample efficiency still unsatisfactory [62].

3) Measurement Incomparability: Savings are reported against heterogeneous baselines and boundaries. One review records energy reductions ranging from roughly 5 to 90 percent for heuristics, 8 to 97 percent for metaheuristics and 2 to 89 percent for machine learning [39], intervals so wide and so overlapping that no ranking survives them. The underlying evidence base is itself uneven: an audit of 258 published estimates found under a third of their sources peer-reviewed [35], and global figures for a single year have been reported across a sixfold range [57].

Research Question: Can data center energy optimisation be evaluated in a way that accounts simultaneously for the direct savings a controller achieves, the induced demand its efficiency may create, the resources beyond electricity that it consumes, and the conditions under which its reported result can be compared with anyone else’s? This paper addresses that question through a coded corpus analysis, a prioritised gap synthesis, and a framework designed against the gaps identified.

Figure 1 summarises the drivers that jointly determine a facility’s footprint and marks the feedback path that current formulations omit.

## B. What Is Missing, and What We Add

Existing reviews of this field catalogue architectures and compare reported accuracy, but rarely code their corpus, and none we located quantify how the evidence base is distributed across validation venues or resource scopes. The contributions are listed below, each with the limits on what it claims:

• Seven-Paradigm Synthesis with Gap Identification: A comparative reading of measurement, prediction, cooling control, scheduling, carbon-aware computing, sustainable AI and enabling technologies (Section II), each closed by naming what it leaves undone.

• Coded Corpus Analysis: Sixty-three papers coded by validation venue, scope, and resource breadth, with the resulting distribution reported in Section VI. Scope: coding is performed at abstract level, not full text, and is single-coded rather than double-coded (see Section VII).

• Ranked Gap Set: Ten deficiencies derived from the coding and ranked on two axes, how much each matters and how readily it could be closed (Section III). Scope: both scores reflect our own reading of the corpus; no panel was convened to set them.

• CLEAR-DC: A framework coupling a control-policy branch to a workload-demand branch through an explicit elasticity term, traced component by component to the gaps it targets (Section III). Scope: CLEAR-DC is specified here, not implemented, trained or measured.

• Minimum Reporting Schema: A set of fields that, reported together, would let future results be compared with one another; Section IV sets each beside its present status.

The paper continues as follows. Section II reviews the seven paradigms and their deficiencies. Section III states the review protocol, formalises the problem and presents CLEAR-DC. Section IV gives the evaluation design and an India-focused case. Section V describes reproducibility. Section VI reports the corpus analysis. Section VII discusses limitations and impact, and Section VIII concludes.

## II. SEVEN PARADIGMS AND WHAT EACH LEAVES UNDONE

The corpus divides into seven paradigms. Each subsection below ends by naming the deficiency that the paradigm leaves behind, and it is those deficiencies that Section III scores.

## A. Measurement and Macro-Estimation

The oldest strand asks how much electricity data centers actually consume. Bottom-up accounting established the early baseline [20] and was later refined to show that United States demand had flattened even as workloads grew [50], a finding generalised globally to argue that efficiency trends had offset service growth [31]. Forecasting work extends the horizon under explicit socio-economic assumptions [21], [10], while critical audits question the provenance of the numbers everyone cites [35], [57] and tutorial surveys have begun to gather infrastructure and sustainability concerns into one account [8].

A parallel line quantifies footprints beyond electricity, including water withdrawal and its spatial concentration in stressed watersheds [34], [51], [23], [19].

Gap Identification: This strand supplies the denominator but not the controller. Its resource breadth — water, spatial stress, embodied impact — has not propagated into the optimisation literature that cites it for motivation.

## B. Predictive Modelling

Data-driven models of facility and workload behaviour preceded learned control. Neural models trained on operational data predicted power usage effectiveness accurately enough to guide configuration [13], and comparable methods were later deployed at hyperscale to tune cooling set points and detect unreasonable operating conditions [60]. Workload forecasting matured in parallel, with systematic comparison of classical and deep predictors on common cluster traces [48].

Gap Identification: Prediction accuracy is rarely connected to realised energy outcomes. A model that forecasts utilisation well is reported as a forecasting result, leaving the downstream saving it enables unquantified.

## C. Deep Reinforcement Learning for Cooling

Learned control began with off-policy actor-critic methods applied end to end to cooling plant, reporting cooling cost reductions in the low double digits against manually configured baselines [26]. Joint optimisation of scheduling and airflow followed, using a parameterised action space to handle the mixed discrete-continuous decision and a two-timescale mechanism to reconcile subsystems that respond at different rates [42], [43], later extended across geographically distributed sites [44]. Multi-agent decomposition addressed the dimensionality of the joint action space [6]. Safety then became a firstclass constraint: imitation pretraining combined with post-hoc action rectification reduced thermal violations by a wide margin relative to reward shaping while still saving substantial facility power [55], and a Lyapunov-based formulation replaced the uniform-temperature assumption with a rack-level compliance metric [4]. Surrogate modelling made real-time optimisation tractable by displacing computational fluid dynamics from the control loop [63], [29].

Gap Identification: Deployment evidence has not kept pace with algorithmic sophistication. Sensitivity to reward design, hyperparameters and scenario shift is documented but unresolved [62], degradation under non-stationary conditions has been observed in the field [56], and simplified occupancy or load assumptions are known to inflate simulated performance [33].

## D. Workload Scheduling and Consolidation

On the compute side, consolidation reduces server power by packing load onto fewer active machines. Prediction-aware policies that anticipate future utilisation rather than reacting to current utilisation reduce needless migrations and servicelevel violations [11], [15], and deep reinforcement learning has largely displaced heuristics for placement [61], [14]. The central tension is thermal: aggressive packing concentrates heat and creates hotspots that erode the saving [16], a concern extended to heat recirculation in recent scheduling work [24]. Serverlevel control combining frequency scaling with fan management is one of the few strands validated on physical hardware [27].

Gap Identification: Reported savings in this paradigm span nearly the entire feasible range across studies [39], and almost all evaluation is trace-driven simulation on a small number of public cluster traces, which makes the spread impossible to attribute to method rather than setting.

## E. Carbon-Aware and Grid-Interactive Computing

A production system at hyperscale demonstrated that temporally flexible workloads can be delayed toward lower-carbon hours using day-ahead intensity forecasts and hourly capacity limits [41]. Subsequent work extends flexibility across space as well as time [46], [59], the latter explicitly pricing the carbon cost of migration rather than counting only its benefit. Data centers have since been reframed as flexibility assets for the grid [52], with empirical trace analysis quantifying how much load is genuinely deferrable [5], and robust or uncertainty-aware formulations addressing renewable variability [1].

Gap Identification: A review of this paradigm finds that spatial and temporal shifting are usually studied in isolation, with only a small minority combining them in one optimisation [3]. More troubling, flexibility is not unambiguously beneficial: under low renewable penetration it can reduce system cost while increasing emissions [49].

## F. Sustainable AI: When AI Becomes the Load

A distinct literature measures the footprint of the models themselves. Lifecycle accounting for a large language model separated operational training emissions from the substantially larger total once manufacturing and surrounding processes were included [28], a distinction formalised into a projection tool covering dense and sparse architectures [9]. A systematic review mapped the field and found it concentrated on the training phase [53] — a focus since overtaken by evidence that inference efficiency depends heavily on workload geometry, software stack and accelerator, and that theoretical operation counts materially understate real consumption [12]. Holistic evaluation showed that undisclosed model development contributed a large share of total impact alongside significant water use [32], and architectural analysis argued for optimising across the whole infrastructure lifecycle rather than efficiency alone [58]. Direct measurement of accelerator nodes during training provides the hardware-level ground truth [22], while power stabilisation work shows that synchronised training produces transients whose frequency content can threaten grid equipment [7]. A dissenting analysis argues that per unit of output, model inference can be far less impactful than the human labour it substitutes [45].

Gap Identification: This paradigm and the control paradigms above are effectively disjoint. Neither cites the other’s mechanisms, and no reviewed study lets an efficiency result in one feed back into a demand assumption in the other.

TABLE I: Coverage of representative prior work on five dimensions.
<table><tr><td>Paradigm</td><td>Joint Safe Res Loop Real</td><td></td><td></td><td></td></tr><tr><td>Measurement [31], [35]</td><td></td><td>O</td><td></td><td>√</td></tr><tr><td>Predictive ML [13], [48]</td><td></td><td></td><td></td><td>√</td></tr><tr><td>DRL cooling [26], [43]</td><td>√</td><td></td><td></td><td></td></tr><tr><td>Safe RL [55], [4] Consolidation [11],</td><td>√</td><td>√</td><td></td><td></td></tr><tr><td>[16]</td><td>0</td><td></td><td></td><td>√</td></tr><tr><td>Carbon-aware [41], [46]</td><td>o</td><td></td><td>√</td><td>√</td></tr><tr><td>Sustainable AI [28], [58] Benchmarks [36],</td><td></td><td></td><td>O</td><td></td></tr><tr><td>[37]</td><td>√</td><td>o</td><td></td><td></td></tr><tr><td>Ours (CLEAR-DC)</td><td>√</td><td>√</td><td>√ √</td><td>o</td></tr></table>

✓ addressed; ◦ partially addressed. The Real column for CLEAR-DC is marked partial because the framework specifies a validation-venue field rather than itself reporting a deployment.

## G. Enablers: Digital Twins, Benchmarks and Explainability

Three enabling technologies determine whether the preceding work becomes deployable. Automated calibration of facility models against sensor measurements removed a manual bottleneck and achieved sub-degree agreement on production halls [54]. Shared benchmark environments appeared only recently, first covering scheduling, cooling and battery management jointly [36], then extending to liquid cooling with interpretable and language-model-based control baselines [37] and to hierarchical control across facility clusters [47]. Explainability, by contrast, is mature in adjacent power-system research [30] but has barely transferred, even though reviews of data center optimisation list transparent decision-making among their unresolved challenges [25].

Gap Identification: Because shared benchmarks arrived in 2024 and 2025, effectively the entire preceding decade was evaluated on private, non-reproducible configurations, and no retrospective re-evaluation of headline results on common environments has been published.

## H. Where This Work Sits

Table I sets the proposed direction beside representative prior work on five axes: joint optimisation of compute and cooling (Joint), explicit safety guarantees during learning (Safe), resource breadth beyond electricity and carbon (Res), closure of the optimiser–load feedback loop (Loop), and validation beyond simulation (Real).

## III. METHOD: PROTOCOL, GAP SCORING AND FRAMEWORK DESIGN

## A. Review Protocol

Retrieval was performed over an index aggregating Semantic Scholar, PubMed, Scopus and arXiv. Twenty queries were executed, structured as one exploratory query, five sub-area queries, three secondary-study queries restricted to review and systematic-review types, two era-gated queries bounding publication year, and nine targeted queries covering water, grid interaction, accelerator power, workload forecasting, explainability and benchmark environments. Each query returned at most ten records, giving 199 records and approximately 194 unique papers after deduplication by title.

TABLE II: Notation for Sections III–VI.
<table><tr><td>Symbol Description</td><td></td></tr><tr><td> $\pi , \pi _ { 0 }$ </td><td>Learned control policy and incumbent baseline controller</td></tr><tr><td> $\mu _ { 0 }$ </td><td>Workload demand process under the baseline</td></tr><tr><td> $\mu _ { \varepsilon }$ </td><td>Workload demand after efficiency-induced ad- justment</td></tr><tr><td> $\varepsilon$ </td><td>Elasticity of compute demand to effective unit cost</td></tr><tr><td> $\delta$ </td><td>Relative reduction in effective unit cost of com- pute</td></tr><tr><td> $E ( \pi , \mu )$   $C , W$ </td><td>Facility energy under policy π and demand  $\mu$  Operational carbon and water withdrawal, same arguments</td></tr><tr><td> $\Xi$ </td><td>Embodied carbon attributed to the evaluation window</td></tr><tr><td> $\Phi$ </td><td>Thermal-safety constraint set (e.g. rack cooling index)</td></tr><tr><td> $\Delta E _ { \mathrm { n e t } }$ </td><td>Net energy benefit after induced demand</td></tr><tr><td> $\sigma$ </td><td>Reporting record emitted for one evaluation</td></tr><tr><td> $\nu$ </td><td>Validation venue label attached to σ</td></tr></table>

Screening retained records that addressed data center or cloud energy consumption, its optimisation, or the footprint of the workloads themselves, and that reported either a method, a measurement or a synthesis. Sixty-three papers were retained and coded. Each was assigned one validation venue (simulation, analytical model, measurement study, physical testbed or field deployment, production facility, or secondary study), one primary scope (cooling, scheduling, carbon-aware, AIas-load, enabler, or macro-estimation), and two binary flags recording whether water withdrawal and embodied carbon were accounted for. Where a paper spanned scopes, the scope corresponding to its principal experimental contribution was recorded.

## B. Notation and Symbols

Symbols recur throughout Sections III to VI; Table II defines each one once.

## C. Scoring the Gaps

Taken together, the seven paradigms leave ten deficiencies that recur across more than one of them. Table III lists these; Figure 2 places each on axes of consequence and tractability. G3 scores highest on both axes: nothing else can be compared until reporting is standardised, and standardising it demands no new technology. Gaps G1 and G9 carry the highest impact but lower feasibility, since closing them requires either economic modelling the field has not attempted or re-running published work on shared environments.

TABLE III: Ten recurring gaps with consequence and tractability scores.
<table><tr><td>ID</td><td>Gap</td><td>Imp.</td><td>Feas.</td></tr><tr><td>G1</td><td>Optimizer-load loop left open</td><td>5</td><td>3</td></tr><tr><td>G2</td><td>Validation confined to simulation</td><td>4</td><td>5</td></tr><tr><td>G3</td><td>No standardised reporting or met- rics</td><td>5</td><td>5</td></tr><tr><td>G4</td><td>Water excluded from control objec- tives</td><td>4</td><td>4</td></tr><tr><td>G5</td><td>Embodied carbon excluded</td><td>3</td><td>4</td></tr><tr><td>G6</td><td>Space and time flexibility studied apart</td><td>4</td><td>4</td></tr><tr><td>G7</td><td>AI load dynamics violate load as- sumptions</td><td>3</td><td>4</td></tr><tr><td>G8</td><td>Safe RL under restrictive thermal models</td><td>3</td><td>3</td></tr><tr><td>G9</td><td>No retrospective re-evaluation on benchmarks</td><td>5</td><td>3</td></tr><tr><td>G10</td><td>Explainability not transferred from grids</td><td>4</td><td>3</td></tr></table>

![](images/7b969c245d91c3dc5e926532b6b0a2eb52a05f47b3d21d577f3291a02f397e75.jpg)  
Fig. 2: Each gap of Table III plotted by how much it matters against how readily it could be closed. Where two share a cell, one is offset so neither is hidden. Shading marks the quadrant worth attacking first.

## D. Problem Formulation

Definition 1 (Net-Benefit Data Center Optimisation). Given a facility operating under an incumbent controller $\pi _ { 0 }$ and a workload demand process $\mu _ { 0 } ,$ and a candidate learned policy π subject to a thermal-safety constraint set Φ, the problem is to report, for an evaluation window: (i) the direct saving achieved at fixed demand; (ii) the induced demand attributable to the efficiency gain, parameterised by an elasticity ε; (iii) the resulting net effect across energy, operational carbon, water withdrawal and the embodied carbon Ξ attributable to the window; and (iv) a record σ carrying the validation venue ν and measurement boundary under which the result was obtained.

Writing δ for the relative reduction in the effective unit cost of compute that the efficiency gain produces, and $\mu _ { \varepsilon }$ for the adjusted demand, the net energy benefit is

$$
\Delta E _ { \mathrm { n e t } } = \underbrace { \big [ E ( \pi _ { 0 } , \mu _ { 0 } ) - E ( \pi , \mu _ { 0 } ) \big ] } _ { \mathrm { r e p o r t e d ~ t o d a y } } - \underbrace { \big [ E ( \pi , \mu _ { \varepsilon } ) - E ( \pi , \mu _ { 0 } ) \big ] } _ { \mathrm { r e b o u n d ~ t e r m } } ,\tag{1}
$$

$$
\mu _ { \varepsilon } = \mu _ { 0 } ( 1 + \varepsilon \delta ) .\tag{2}
$$

The first bracket is what the literature reports. The second, which the demand adjustment of (2) drives, is absent from every control study coded in Section VI. When $\varepsilon = 0$ the definition collapses to the conventional formulation, which makes the standard result a special case rather than a rival. The same decomposition applies to carbon, water and, with Ξ amortised over the window, to embodied impact; the multi-resource objective is then to report the vector $( \Delta E _ { \mathrm { n e t } } , \Delta C _ { \mathrm { n e t } } , \Delta W _ { \mathrm { n e t } } , \Xi )$ subject to Φ rather than to optimise any single component.

Scope Clarification: This is a specification for how results should be reported and evaluated, not a solved optimisation with convergence guarantees. In particular, ε is not identified by any dataset in this corpus; the definition makes its absence visible rather than supplying its value.

## E. CLEAR-DC Architecture

Figure 3 lays out CLEAR-DC (Closed-Loop Energy Accounting and Reporting for Data Centers) in three stages. Stage 1 instruments the facility across six scopes — information technology power, facility overhead, grid signals, water withdrawal, embodied hardware impact and workload composition — and aligns them onto a declared common measurement boundary, so that a reported figure carries the scope it was measured over. Stage 2 encodes the control problem and the demand problem in separate branches: an optimiser branch holding the policy π, realisable by any sequence model that exposes attention weights or a comparable attribution signal, and a load branch representing the demand process $\mu .$ The two are joined by an elasticity coupling that carries the efficiency gain from the optimiser branch into the demand branch, which is the component that closes the loop of Definition 1.

Stage 3 reads out four heads from the coupled representation. The net-benefit head evaluates (1) rather than the first bracket alone. The safety head enforces Φ by projection, following the rectification approach that has proved effective in constrained cooling control [55], [4]. The resource head reports water and embodied components alongside energy. The explanation head produces attributions with a stability check across resampled backgrounds, since an unstable explanation is not auditable. All four feed a reporting layer that emits the record σ in the schema of Section IV.

Algorithm 1 specifies the proposed evaluation for one control interval.

## F. From Gaps to Components

Every component of Figure 3 exists because of a specific entry in Table III. Table IV records which, so a reader can check the design against the diagnosis instead of taking it on trust.

![](images/5cf9d24a3933ce6785098ad602815d041c0267f4479b4bf571ce989826914750.jpg)  
net-benefit feedback: realised δ updates the demand branch  
Fig. 3: CLEAR-DC: six instrumentation scopes are aligned onto a declared boundary, encoded by an optimiser branch and a load branch coupled through an explicit elasticity term, and read out by four heads into a schema-conformant record.

Algorithm 1 CLEAR-DC net-benefit evaluation, one control   
interval.   
Notation: $\Omega _ { t } \mathbf { : }$ instrumented window; B: resamples for attribution   
stability.   
Require: Window $\Omega _ { t } ,$ incumbent $\pi _ { 0 } ,$ policy π, constraints Φ, elas  
ticity prior ε   
Ensure: Net-benefit vector, safe action aˆ, record σ   
$1 \colon \bar { \Omega } \gets \mathrm { A l i g n B o u n d a r y } ( \Omega _ { t } ) $ ▷ Stage 1   
2: $z _ { \cdot } ^ { P } \gets \bar { \mathrm { O p t i m i s e r B r a n c h } } ( \bar { \Omega } ; \theta _ { P } )$ ▷ Stage 2   
3: $z ^ { L } \gets \mathrm { L } \dot { \operatorname { o a d } } \mathrm { B r a n c h } ( \bar { \Omega } ; \dot { \theta _ { L } } )$ ▷ Stage 2   
4: $\delta \gets \mathrm { U n i t C o s t D e l t a } ( z ^ { P } , \bar { \pi _ { 0 } } )$   
5: $\boldsymbol { z } _ { t } \gets \mathrm { C o u p l e } ( \boldsymbol { z } ^ { P } , \boldsymbol { z } ^ { L } , \boldsymbol { \varepsilon } , \delta )$ ▷ closes the loop   
6: $a \gets \mathrm { P o l i c y H e a d } ( z _ { t } )$   
7: $\hat { a } \gets \mathrm { P r o j e c t } ( a , \Phi )$ ▷ safety head   
8: $( \Delta E , \Delta \dot { C } , \Delta \dot { W } ) \gets \mathrm { N e t B e n e f i t } ( z _ { t } , \varepsilon , \delta )$ ▷ Eq. (1)   
9: $\dot { \Xi }  \iota$ AmortiseEmbodied(Ω)<sup>¯</sup> ▷ resource head   
10: for $b = 1 , \dots , B$ do   
11: $A _ { b }  \mathbf { A } 1$ ttribute $( z _ { t } ,$ , resample )   
12: end for   
13: A ← StabilityCheck $( \{ A _ { b } \} _ { b = 1 } ^ { B } )$ ▷ explanation head   
14: σ ← Record $\begin{array} { r } { ( \Delta E , \Delta \tilde { C } , \Delta \tilde { W } , \tilde { \Xi } , A , \nu , } \end{array}$ boundary)   
15: return $( \hat { a } , \sigma )$ to the reporting layer

## IV. HOW THE FRAMEWORK SHOULD BE EVALUATED

## A. A Design Case: India’s Data Center Build-Out

India is among the fastest-growing data center markets, with concentrated capacity in Mumbai, Chennai and Hyderabad and a grid whose carbon intensity varies substantially by region and hour. Several conditions make it a useful design case rather than a benchmark. Water stress is spatially uneven and, in several hosting regions, severe, which activates the tradeoff documented between electrical efficiency and water withdrawal in hot-arid climates [19], [23]. Grid carbon intensity varies enough to make temporal shifting meaningful, while the evidence that flexibility can raise emissions under low renewable penetration [49] means the direction of benefit cannot be assumed. Table V lists candidate data categories for an instantiation. We stress that this is a design case: no results are reported for it, and public operational data at the granularity CLEAR-DC requires is not currently available.

TABLE IV: Each component and the gaps it responds to.
<table><tr><td>Component</td><td>Gaps</td><td>Rationale</td></tr><tr><td>Boundary align- ment</td><td>G3</td><td>Forces the measurement scope to be declared before any figure</td></tr><tr><td>Water, embodied</td><td>G4, G5</td><td>is reported Instruments resources the coded corpus never carries into</td></tr><tr><td>scopes Optimiser</td><td>G8,</td><td>control Exposes an attribution signal; safety set applied to its output</td></tr><tr><td>branch Load branch</td><td>G10 G1, G7</td><td>Represents demand as a mod-</td></tr><tr><td></td><td></td><td>elled process, including AI load dynamics The rebound term of Eq. (1),</td></tr><tr><td>Elasticity coupling</td><td>G1</td><td>absent from prior formulations</td></tr><tr><td>Net-benefit head</td><td>G1, G3</td><td>Reports the decomposition, not the direct bracket alone</td></tr><tr><td>Safety head</td><td>G8</td><td>Projection onto Φ rather than reward shaping</td></tr><tr><td>Resource head</td><td>G4, G5, G6</td><td>Vector read-out enables joint space-time-resource objectives</td></tr><tr><td>Explanation head</td><td>G10</td><td>Attribution with a stability check across resamples</td></tr><tr><td>Reporting layer</td><td>G2, G3, G9</td><td>Emits venue and boundary, en- abling retrospective compari-</td></tr></table>

## B. A Minimum Set of Reported Fields

Table VI sets out the fields we argue should be reported together, beside what is reported at present. The schema is deliberately modest: every field is either already measured by some subset of the corpus or derivable from information the authors necessarily possess. Its value is not novelty but joint reporting, since it is the absence of the venue and boundary fields that makes the savings intervals of Section VI uninterpretable.

TABLE V: Data categories an India deployment would require.
<table><tr><td>Category</td><td>Candidate sources</td></tr><tr><td>Facility</td><td>Operator PUE and WUE disclosures; colocation reports</td></tr><tr><td>Grid</td><td>Central Electricity Authority generation mix and CO2 baseline database; regional</td></tr><tr><td>Climate</td><td>load dispatch data India Meteorological Department station records; wet-bulb series for economiser</td></tr><tr><td>Water</td><td>feasibility Central Ground Water Board stress classi- fication; municipal supply tariffs</td></tr><tr><td>Workload</td><td>Public cluster traces as proxy; accelerator power profiles from measurement studies</td></tr><tr><td>Embodied</td><td>[22] Hardware lifecycle inventories; refresh cy- cle assumptions</td></tr><tr><td>Policy</td><td>MeitY data center policy instruments; state incentive frameworks</td></tr></table>

TABLE VI: Reporting fields proposed here, with their present status.
<table><tr><td>Field</td><td>Content</td><td>Current status</td></tr><tr><td>Baseline</td><td>Identity of  $\pi _ { 0 }$  (rule- based, PID, MPC,</td><td>Reported, but often unspecified</td></tr><tr><td>Boundary</td><td>prior policy) IT-only, cooling subsystem, facility, or lifecycle</td><td>Rarely stated explic- itly</td></tr><tr><td>Venue ν</td><td>Simulation, testbed, field, production</td><td>Inferable at best; never a field</td></tr><tr><td>Direct saving</td><td> $\Delta E$  at fixed de- mand</td><td>Universally reported</td></tr><tr><td>Net saving</td><td> $\Delta E _ { \mathrm { n e t } }$  after induced demand</td><td>Not reported anywhere in this corpus</td></tr><tr><td>Water</td><td>∆W and basin stress class</td><td>Only in macro studies</td></tr><tr><td>Embodied</td><td>三 amortised over window</td><td>Only in AI-load stud- ies</td></tr><tr><td>Safety</td><td>Violations of Φ dur- ing learning and op- eration</td><td>Reported by safe-RL work only</td></tr><tr><td>Explanation</td><td>Attribution stability across resamples</td><td>Essentially untested</td></tr></table>

## C. Scoring the Paradigms

Figure 4 scores the paradigms of Table I on the six axes implied by Table VI, with CLEAR-DC’s intended position shown alongside. These scores are our reading of the literature and rest on no experiment; the bars for the proposed framework state what it is meant to achieve, and quoting them as achieved performance would misrepresent this paper.

![](images/3e9274da9edd076c102337de984923e5433ca049af6f248ffd18fa7385fb617d.jpg)  
Fig. 4: Six axes, five paradigms. Bars for the proposed framework record intentions, not measurements.

## V. REPRODUCIBILITY AND ARTEFACT AVAILABILITY

The empirical claims in this paper are claims about a corpus, so the corpus is the artefact. The coding sheet, the derived statistics and the code that computes every count in Section VI are released at:

https://github.com/Kimalice/AI-for-Energy-Optimization-in-Data-Centers-Closing-the-Optimizer-Load-Loop

We report the following:

• Search record: All twenty queries are listed with their filters, returned counts and status. Retrieval was performed on a single date; the access tier used caps results at ten per query, which bounds coverage at roughly 200 records before deduplication.

• Coding sheet: Each of the 63 retained papers carries a venue label, a scope label and two binary resource flags. The full sheet, including the coding rules and the ambiguous cases, accompanies this paper.

• Derived statistics: Every count in Section VI is computed from the coding sheet by a short script rather than tallied by hand, so the numbers can be regenerated and disagreements localised to individual coding decisions.

• Figure provenance: Figures 5 and 6 are generated directly from the coding sheet and from the intervals reported in [39] respectively; no value in either is estimated.

## VI. CORPUS ANALYSIS: FIRST QUANTITATIVE FINDINGS

The 63 coded papers comprise 13 secondary studies and 50 primary studies. Of the primary studies, 28 are control-oriented, meaning their principal contribution is a method that acts on a facility through cooling control, scheduling, or carbon-aware placement. Table VII collects the counts; the findings below concern that subset unless stated otherwise, and are reported as coded.

Validation venue. Eighteen of the 28 control-oriented studies, or 64 percent, are validated in simulation or trace-driven replay alone. Two use a physical testbed or field deployment and three report results from a production facility, giving five studies, or 18 percent, with evidence from real infrastructure. The remaining five are analytical or measurement studies without a controller in the loop. Figure 5 shows the distribution by scope and makes the asymmetry visible: production evidence exists almost entirely in the cooling paradigm, and is concentrated in work originating from operators rather than universities [13], [60], [41]. This is consistent with, and quantifies for data centers specifically, the deployment shortfall reported for building control more broadly [2]. Twenty-eight is a small denominator, and the released analysis prices that rather than leaving it to the reader: resampling the coded corpus places a 95 percent interval of 46 to 82 percent around the simulationonly share, and reassigning two venue codes would carry the point estimate below 60 percent. The direction of the result is secure at this sample size; the second significant figure is not.

![](images/afce4a073a4cbdc77255c149342233ef9319152d7728bec6a0ffb6d05cc8fcb5.jpg)  
Fig. 5: Validation venue by scope across the 63 coded papers. Production evidence is confined almost entirely to cooling, and the AI-as-load strand is dominated by measurement rather than control.

Resource breadth. Nine of the 63 papers account for water withdrawal and six account for embodied carbon. None of the 28 control-oriented studies does either. Water appears exclusively in macro-estimation and measurement work [34], [51], [23], [19], [17], and embodied carbon exclusively in the AI-as-load strand [28], [9], [58], [32]. The separation is clean enough to be structural rather than incidental: the studies that measure these resources do not build controllers, and the studies that build controllers do not measure these resources. A policy minimising facility energy therefore optimises against an objective that provably excludes two of the four impact channels its own motivating literature identifies.

The savings-range problem. Figure 6 plots the intervals reported by a systematic review across four technique families [39] against the range spanned by the deep reinforcement learning cooling studies in this corpus. Three observations follow. The four review intervals overlap across almost their entire extent, so no ordering of technique families is supported. The intervals are also far wider than any plausible physical range for a single facility, which indicates that they aggregate across incommensurable baselines and boundaries rather than measuring method quality. And the corpus range for cooling control, roughly 7 to 27 percent, is both narrower and lower than the upper reaches of the review intervals — not because those methods are better, but because the studies behind the wider intervals report against weaker baselines over narrower boundaries.

![](images/5f33b81df20acfdffba0d67593512176659f544bfcb48cde36ef183b35c1641a.jpg)  
Fig. 6: Reported energy savings intervals. The four upper bars are the ranges recorded across technique families in a systematic review; the lower bar is the span of DRL cooling results in this corpus. Overlap is near-total.

TABLE VII: Coded corpus, summary counts.
<table><tr><td>Dimension</td><td>Count</td><td>Share</td></tr><tr><td>Papers coded</td><td>63</td><td></td></tr><tr><td>Secondary studies</td><td>13</td><td>21% of 63</td></tr><tr><td>Control-oriented primary studies</td><td>28</td><td>44% of 63</td></tr><tr><td>simulation or trace only</td><td>18</td><td>64% of 28</td></tr><tr><td>testbed or field deployment</td><td>2</td><td>7% of 28</td></tr><tr><td>production facility</td><td>3</td><td>11% of 28</td></tr><tr><td>accounting for water</td><td>0</td><td>0% of 28</td></tr><tr><td>accounting for embodied carbon</td><td>0</td><td>0% of 28</td></tr><tr><td>reporting net of induced demand</td><td>0</td><td>0% of 28</td></tr><tr><td>AI-as-load studies</td><td>10</td><td>16% of 63</td></tr><tr><td>published 2023 or later</td><td>9</td><td>90% of 10</td></tr><tr><td>Papers accounting for water</td><td>9</td><td>14% of 63</td></tr><tr><td>Papers accounting for embodied carbon</td><td>6</td><td>10% of 63</td></tr></table>

Temporal structure. Ten papers form the AI-as-load strand, and nine of the ten were published in 2023 or later. That strand is therefore almost entirely younger than the control literature it needs to inform, which offers a benign explanation for the disjunction reported above — the two bodies of work have barely had time to meet. It also implies that the disjunction is tractable, since no entrenched methodological commitment separates them.

Reading these results. These counts do not show that learned control does not work; several of the coded studies report credible savings under credible constraints. They show that the field’s evidence base has a shape, and that the shape is narrow in three specific ways — venue, resource scope, and demand framing — each of which corresponds to a gap in Table III. The purpose of the schema in Table VI is to make that shape visible in future work at the cost of a few additional reported fields.

## VII. DISCUSSION

## A. What This Paper Establishes

No method in Table I holds all five properties at once: joint optimisation, safety guarantees, resource breadth, loop closure, and evidence from outside a simulator. CLEAR-DC is drawn to fill that empty cell. We do not claim it succeeds in doing so; settling that requires building it. What this paper does establish empirically is narrower and, we would argue, more useful: the distribution of the evidence base reported in Section VI, and the observation that the two literatures most relevant to the question are structurally disjoint.

## B. Before Any Implementation

Several design choices are worth surfacing before any implementation. We treat the safety head as mandatory rather than optional, following evidence that reward shaping learns constraint compliance only after experiencing violations [55]. The explanation head is included because operator trust, not algorithmic performance, appears to be the binding constraint on adoption, and because the explanation techniques most likely to be reached for are known to be locally unstable unless checked. The elasticity coupling introduces a parameter that cannot currently be estimated from any dataset in this corpus; in an implementation it would be supplied as a scenario range rather than a point value, and results reported across that range.

## C. Limits of the Evidence

1) What the evidence supports: the quantitative component counts publications; it does not measure facilities. It characterises what has been published, not what is physically true of data centers. A framework evaluated as unpublished or proprietary work would be invisible to it, and industrial results are systematically under-published.

2) Retrieval bound: twenty queries at ten results each bound coverage at roughly 200 records. Sub-areas addressed by a single query are correspondingly thin, and one query intended to cover edge deployment returned largely mobile federated-learning work, so edge scenarios are underrepresented.

3) Coding method: coding was performed at abstract level by a single coder. A paper that accounts for water only in its full text would be coded as not accounting for it. Double coding with an inter-rater statistic would strengthen the resourcebreadth claims in particular.

4) Prioritization method: the impact and feasibility scores in Table III reflect the authors’ synthesis rather than formal elicitation; a structured expert panel would place them on firmer ground.

5) Unidentified elasticity: Definition 1 introduces ε without supplying it. The definition makes the omission explicit and bounded rather than resolving it, and a net-benefit figure computed with an assumed ε should be read as a sensitivity, not a measurement.

6) Framework not implemented: CLEAR-DC has not been built, trained or benchmarked. The bars in Figure 4 are design targets, and no performance claim is made for it.

## D. Broader Impact and Ethical Considerations

Three considerations warrant emphasis. Resource justice: a controller optimising electricity alone may increase water withdrawal in a stressed basin, and the populations affected by that withdrawal are not parties to the optimisation; the resource head exists for this reason. Measurement honesty: an efficiency claim that omits induced demand can support a decision to expand capacity that the underlying analysis does not justify, and the risk grows as such claims enter policy discussion. Regional asymmetry: regulators and smaller operators in emerging markets are less equipped to audit a learned controller independently than are the hyperscale operators that produce most of the production evidence in this corpus, which is part of why explanation stability and declared measurement boundaries are treated here as requirements rather than refinements.

## VIII. CONCLUSION AND FUTURE WORK

Across predictive modelling, reinforcement learning, scheduling and carbon-aware placement, the reported gains rest on a premise this paper has tried to make visible: the field evaluates its controllers against a demand process it holds fixed, over a resource scope that excludes water and embodied impact, in settings that are overwhelmingly simulated. Coding 63 papers makes the shape of that evidence base measurable rather than anecdotal.

## A. Summary of Contributions

• A reading of seven paradigms, each closed by naming what it leaves undone (Section II).

• A coded corpus of 63 papers showing that 18 of 28 control studies are simulation-only and none account for water or embodied carbon (Section VI).

• Ten recurring gaps, scored on consequence and tractability and ranked (Section III).

• CLEAR-DC, a framework traceable component by component to those gaps (Section III).

• A minimum reporting schema whose joint adoption would make future results comparable (Section IV).

## B. Where to Go Next

• Elasticity estimation: identify ε empirically from operator data linking efficiency improvements to subsequent capacity provisioning, which would convert Definition 1 from a specification into a measurement.

• Retrospective benchmarking: re-evaluate headline results from the coded corpus on the shared environments that appeared in 2024 and 2025 [36], [37], which is the cheapest available test of whether reported savings survive a common setting.

• Water-aware control: extend a safe cooling controller with a water objective and evaluate it across climate zones where the electricity–water tradeoff reverses [19].

• Load-dynamics-aware control: relax the smooth-load assumption using measured accelerator power profiles [22], [7] and test whether existing controllers remain stable under realistic training transients.

• Double-coded replication: repeat the corpus analysis with two independent coders and full-text access to establish interrater reliability for the resource-breadth findings.

## C. Adopting Parts of the Framework

CLEAR-DC is specified as an accounting and reporting structure rather than a fixed pipeline. The optimiser branch accepts any policy class exposing an attribution signal, so an existing safe controller can be dropped in unchanged. The load branch accepts any demand model, including a constant one, in which case the framework degrades gracefully to the conventional open-loop formulation. The reporting layer is independent of both and could be adopted on its own by studies that decline the rest of the architecture — which, given the findings of Section VI, would be the single most valuable outcome of this paper.

## REFERENCES

[1] A. Ajagekar and F. You, “Variational quantum circuit learning-enabled robust optimization for AI data center energy control and decarbonization,” Advances in Applied Energy, 2024.

[2] K. Al Sayed, A. Boodi, R. Sadeghian Broujeny, and K. Beddiar, “Reinforcement learning for HVAC control in intelligent buildings: A technical and conceptual review,” Journal of Building Engineering, 2024.

[3] N. Asadov et al., “Carbon-aware spatio-temporal workload shifting in edge-cloud environments: A review and novel algorithm,” Sustainability, 2025.

[4] Z.-Y. Cao et al., “Toward model-assisted safe reinforcement learning for data center cooling control: A Lyapunovbased approach,” Proc. 14th ACM International Conference on Future Energy Systems, 2023.

[5] A. Caprara et al., “Data center workload flexibility for power system demand response: Evidence from Alibaba traces,” International Journal of Electrical Power & Energy Systems, 2026.

[6] C. Chi et al., “Cooperatively improving data center energy efficiency based on multi-agent deep reinforcement learning,” Energies, 2021.

[7] E. Choukse et al., “Power stabilization for AI training datacenters,” arXiv, 2025.

[8] S. Cruzes, “Data centers in the age of AI: A tutorial survey on infrastructure, sustainability, and emerging challenges,” Journal of Network and Computer Applications, 2026.

[9] A. Faiz et al., “LLMCarbon: Modeling the end-to-end carbon footprint of large language models,” arXiv, 2023.

[10] Y. V. Fan et al., “Data centre energy demand projections within Shared Socioeconomic Pathways,” Energy and Climate Change, 2026.

[11] F. Farahnakian et al., “Energy-aware VM consolidation in cloud data centers using utilization prediction model,” IEEE Transactions on Cloud Computing, 2019.

[12] J. Fernandez et al., “Energy considerations of large language model inference and efficiency optimizations,” 2025.

[13] J. Gao, “Machine learning applications for data center optimization,” Google White Paper, 2014.

[14] H. Hou et al., “Energy efficient task scheduling based on deep reinforcement learning in cloud environment: A specialized review,” Future Generation Computer Systems, 2023.

[15] S.-Y. Hsieh et al., “Utilization-prediction-aware virtual machine consolidation approach for energy-efficient cloud data centers,” Journal of Parallel and Distributed Computing, 2020.

[16] S. Ilager, K. Ramamohanarao, and R. Buyya, “ETAS: Energy and thermal-aware dynamic virtual machine consolidation in cloud data center with proactive hotspot mitigation,” Concurrency and Computation: Practice and Experience, 2019.

[17] Y. Jiang et al., “ThirstyFLOPS: Water footprint modeling and analysis toward sustainable HPC systems,” Proc. International Conference for High Performance Computing (SC), 2025.

[18] H. Kahil et al., “Reinforcement learning for data center energy efficiency optimization: A systematic literature review and research roadmap,” Applied Energy, 2025.

[19] L. Karimi et al., “Water-energy tradeoffs in data centers: A case study in hot-arid climates,” Resources, Conservation and Recycling, 2022.

[20] J. Koomey, “Worldwide electricity used in data centers,” Environmental Research Letters, 2008.

[21] M. Koot and F. Wijnhoven, “Usage impact on data center electricity needs: A system dynamic forecasting model,” Applied Energy, 2021.

[22] I. Latif et al., “Single-node power demand during AI training: Measurements on an 8-GPU NVIDIA H100 system,” IEEE Access, 2024.

[23] N. Lei and E. Masanet, “Climate- and technology-specific PUE and WUE estimations for U.S. data centers using a hybrid statistical and thermodynamics-based approach,” Resources, Conservation and Recycling, 2022.

[24] J. Li et al., “An energy-aware virtual machine scheduling approach for cloud data centers,” IEEE Transactions on Sustainable Computing, 2025.

[25] X. Li et al., “A review on AI-driven optimization of data center energy efficiency and thermal management,” International Journal of Applied Science, 2025.

[26] Y. Li et al., “Transforming cooling optimization for green data center via deep reinforcement learning,” IEEE Transactions on Cybernetics, 2017.

[27] W. Lin et al., “A multi-agent reinforcement learningbased method for server energy efficiency optimization combining DVFS and dynamic fan control,” Sustainable Computing: Informatics and Systems, 2024.

[28] A. S. Luccioni, S. Viguier, and A.-L. Ligozat, “Estimating the carbon footprint of BLOOM, a 176B parameter language model,” Journal of Machine Learning Research, 2022.

[29] S. Ma et al., “Artificial intelligence-enabled predictive energy saving planning of liquid cooling system for data centers,” Advanced Engineering Informatics, 2025.

[30] R. Machlev et al., “Explainable artificial intelligence (XAI) techniques for energy and power systems: Review, challenges and opportunities,” Energy and AI, 2022.

[31] E. Masanet, A. Shehabi, N. Lei, S. Smith, and J. Koomey, “Recalibrating global data center energy-use estimates,” Science, 2020.

[32] J. D. Morrison et al., “Holistically evaluating the environmental impact of creating language models,” arXiv, 2025.

[33] O. B. Mulayim et al., “On the impact of simulated occupancy behavior assumptions on reinforcement learning for HVAC controls,” Proc. 16th ACM International Conference on Future and Sustainable Energy Systems, 2025.

[34] D. Mytton, “Data centre water consumption,” npj Clean Water, 2021.

[35] D. Mytton and M. Ashtine, “Sources of data center energy estimates: A comprehensive review,” Joule, 2022.

[36] A. Naug et al., “SustainDC: Benchmarking for sustainable data center control,” Advances in Neural Information Processing Systems 37, 2024.

[37] A. Naug et al., “LC-Opt: Benchmarking reinforcement learning and agentic AI for end-to-end liquid cooling optimization in data centers,” arXiv, 2025.

[38] A. P. Oliveira et al., “Beyond efficiency: A systematic review of energy consumption and carbon footprint across the AI lifecycle,” Sustainability, 2026.

[39] S. Panwar et al., “A systematic review on effective energy utilization management strategies in cloud data centers,” Journal of Cloud Computing, 2022.

[40] R. Pinto et al., “Long-run electricity consumption in computing: Exponential growth followed by stabilization due to efficiency gains,” iScience, 2026.

[41] A. Radovanovic et al., “Carbon-aware computing for datacenters,” IEEE Transactions on Power Systems, vol. 38, no. 2, 2023.

[42] Y. Ran et al., “DeepEE: Joint optimization of job scheduling and cooling control for data center energy efficiency using deep reinforcement learning,” Proc. IEEE 39th International Conference on Distributed Computing Systems, 2019.

[43] Y. Ran et al., “Optimizing energy efficiency for data center via parameterized deep reinforcement learning,” IEEE Transactions on Services Computing, 2023.

[44] Y. Ran et al., “D3T: Dual-timescale optimization of task scheduling and thermal management for energy efficient geo-distributed data centers,” IEEE Transactions on Parallel and Distributed Systems, 2026.

[45] S. Ren et al., “Reconciling the contrasting narratives on the environmental impact of large language models,” Scientific Reports, 2024.

[46] I. Riepin et al., “Spatio-temporal load shifting for truly clean computing,” arXiv, 2024.

[47] S. Sarkar et al., “Hierarchical multi-agent framework for carbon-efficient liquid-cooled data center clusters (Green-DCC),” 2025.

[48] D. Saxena et al., “Performance analysis of machine learning centered workload prediction models for cloud,” IEEE Transactions on Parallel and Distributed Systems, 2023.

[49] J. R. L. Senga et al., “Flexible data centers reduce power system costs but can increase emissions,” iScience, 2026.

[50] A. Shehabi, S. J. Smith, E. Masanet, and J. Koomey, “Data center growth in the United States: Decoupling the demand for services from electricity use,” Environmental Research Letters, 2018.

[51] M. A. B. Siddik, A. Shehabi, and L. Marston, “The environmental footprint of data centers in the United States,” Environmental Research Letters, 2021.

[52] M. Takcı et al., “Data centres as a source of flexibility for power systems,” Energy Reports, 2025.

[53] R. Verdecchia, J. Sallou, and L. Cruz, “A systematic review of Green AI,” WIREs Data Mining and Knowledge Discovery, 2023.

[54] R. Wang et al., “Kalibre: Knowledge-based neural surrogate model calibration for data center digital twins,” Proc. 7th ACM International Conference on Systems for Energy-Efficient Buildings, Cities, and Transportation, 2020.

[55] R. Wang et al., “Green data center cooling control via physics-guided safe reinforcement learning,” ACM Transactions on Cyber-Physical Systems, 2023.

[56] X. Wang et al., “From simulation to reality: A study of reinforcement learning control in operational building environments,” ASME Journal of Engineering for Sustainable Buildings and Cities, 2025.

[57] Y. Wang et al., “Building accurate energy-use statistics for data centers,” Engineering, 2025.

[58] C.-J. Wu et al., “Beyond efficiency: Scaling AI sustainably,” IEEE Micro, 2024.

[59] T. Yang et al., “Carbon management of multi-datacenter based on spatio-temporal task migration,” IEEE Transactions on Cloud Computing, 2023.

[60] Z. Yang et al., “Increasing the energy efficiency of a data center based on machine learning,” Journal of Industrial Ecology, 2021.

[61] J. Zeng et al., “Adaptive DRL-based virtual machine consolidation in energy-efficient cloud data center,” IEEE Transactions on Parallel and Distributed Systems, 2022.

[62] Q. Zhang et al., “Deep reinforcement learning towards real-world dynamic thermal management of data centers,” Applied Energy, 2023.

[63] H. Zhou et al., “A transformer-based thermal surrogate model for cooling control in data centers,” IEEE Robotics and Automation Letters, 2025.