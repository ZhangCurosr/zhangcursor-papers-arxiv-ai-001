## Highlights

## OperAItor: An Agentic Framework for Fault-Tolerant Operation of Chemical Processes Javal Vyas, Mehmet Mercangöz

• Hypothetical reactor–distillation benchmark with process control, alarms, and 11 independent SIFs.

• Five validation configurations are compared for LLM-generated supervisory actions.

• Twenty single and multiple-fault scenarios are assessed using plant stops and illustrative economics.

• Deterministic admissibility checks improve outcomes for the 3B-parameter language model.

• Ofline admissibility analysis identifies four condenser-loss cases with no feasible single-action recovery.

# OperAItor: An Agentic Framework for Fault-Tolerant Operation of Chemical Processes

Javal Vyas<sup>a,∗</sup>, Mehmet Mercangöz<sup>a</sup>

<sup>a</sup>Autonomous Industrial Systems Laboratory, Department ofChemical Engineering, Imperial College London, Imperial College Road, South Kensington, London, SW7 2AZ, United Kingdom

## A R T I C L E I N F O

Keywords:   
Fault-tolerant control   
Large language models   
Agentic AI   
Process monitoring   
Safety instrumented systems   
Digital twin   
Distillation

## A BS T RA C T

This paper presents OperAItor, an open-source framework for developing and evaluating large language model (LLM) agents for supervisory fault management in chemical processes. The framework is demonstrated on a dynamic reactor–distillation process representing isopropanol dehydrogenation, with physically based thermodynamics and modified reaction kinetics to permit liquid-phase operation. The process is represented by a 113-variable diferential–algebraic model and operates at a nominal feed rate of 100 kmol/h, producing approximately 4 t/h of acetone. The simulated plant includes eight PI control loops, 22 process alarms, five duty–standby pump trains, and 11 independent safety instrumented functions. The agent architecture integrates multivariate statistical process monitoring, fault diagnosis, knowledge-graph retrieval, action planning, and supervisory action generation. Five action-validation configurations are investigated, comprising operation without validation, an LLM-based judge, digital-twin simulation, and deterministic admissibility checks with and without consequence projection. The framework is evaluated using four LLMs across 20 scenarios involving single and multiple simultaneous faults. Without agent intervention, 18 scenarios result in plant stops, corresponding to an economic loss of \$399,000 relative to fault-free operation under the assumed benchmark economics. For GPT-5 nano, Qwen3 8B, and Mistral 7B, validated agent configurations reduce the number of stops to 5–8 and recover \$209,000–265,000 of the loss. For Llama 3.2 3B, deterministic admissibility checks reduce the number of stops from 13 without validation to 7, whereas LLM-based validation results in 15 stops. An ofline admissibility analysis finds no feasible recovery among the evaluated single supervisory actions for four condenser-cooling-loss scenarios. The study provides a reproducible benchmark for comparing agent architectures, language models, and actionvalidation methods for supervisory fault management in chemical processes.

## 1. Introduction

Abnormal operating conditions in chemical processes often require interventions that extend beyond the capabilities of regulatory control. A failed duty pump may require the activation of standby equipment, a deteriorating heat exchanger may require a reduction in process throughput, and an actuator fault may require an alternative operating strategy. Although the basic process control system (BPCS) regulates normal operation and an independent safety instrumented system (SIS) provides protection against specified hazardous conditions, neither necessarily determines the most appropriate corrective intervention following a fault (ANSI/ISA, 2016; IEC, 2016). This responsibility typically rests with plant operators, who must interpret alarms, identify the likely cause of the abnormal condition, assess available interventions, and act before the process reaches a protective shutdown. The challenge is to detect abnormal operation and select corrective actions that sustain production while respecting process constraints.

Fault detection and diagnosis (FDD) and fault-tolerant control (FTC) have addressed diferent aspects of this problem. Statistical process monitoring, including principal component analysis (PCA), provides established methods for detecting deviations from nominal operation (MacGregor and Kourti, 1995; Qin, 2003; Venkatasubramanian et al., 2003a; Chiang et al., 2001). Model-based diagnosis and active FTC extend this capability towards fault isolation and control reconfiguration (Venkatasubramanian et al., 2003b; Blanke et al., 2006; Zhang and Jiang, 2008; Mhaskar et al., 2013). However, the availability of a fault diagnosis does not automatically determine an appropriate operator intervention. Recovery may require discrete equipment operations, changes to supervisory setpoints, or coordinated adjustments across interacting process units. The feasibility of an intervention depends on the process state, actuator availability, operating procedures, intervention authority, and the consequences of the proposed action. These considerations motivate supervisory decision-making frameworks that connect process monitoring and diagnosis with executable, plantspecific recovery actions.

Large language models (LLMs) ofer a potential mechanism for such supervisory decision-making. Through structured interfaces and access to plant-specific information, LLM-based agents can interpret alarms and operating trends, retrieve equipment relationships and procedures, formulate recovery plans, and translate these plans into operator-level commands (Brown et al., 2020; Yao et al., 2023; Decardi-Nelson et al., 2024; Schweidtmann, 2024). Previous studies have demonstrated the feasibility of using LLM agents for industrial process control and fault recovery, including architectures that combine knowledge retrieval, planning, and simulation-based action validation (Vyas and Mercangöz, 2024; Vyas et al., 2026). Related work has investigated deterministic verification of LLM-generated industrial actions using engineering information such as piping and instrumentation diagrams (P&IDs) and operating constraints (Schall, 2026). These developments establish that LLMs can generate meaningful supervisory interventions when their access to the process is appropriately constrained. They do not, however, eliminate the possibility of incorrect fault attribution, unsuitable control actions, or interventions that are individually executable but undesirable when applied to the current process state.

This distinction motivates the central question investigated in this work: given an LLM-generated operator action, how should its suitability for execution be evaluated? The question is important because action generation and action validation address diferent problems. An agent may correctly identify a fault yet recommend an intervention that accelerates a protective trip. Conversely, a validation mechanism may reject a beneficial intervention because its acceptance conditions are overly restrictive. A useful validation layer must therefore balance the exclusion of undesirable actions against the preservation of opportunities for fault recovery. Its efectiveness cannot be established solely by measuring the validity of generated text or the proportion of proposals accepted. It must also be assessed through the resulting closed-loop process behaviour.

Several validation approaches can be considered. A model-based validator evaluates a proposed action by simulating its predicted consequences using a process model or digital twin, allowing candidate interventions to be compared before execution (Kritzinger et al., 2018; Wabersich and Zeilinger, 2021). An admissibility-based validator evaluates whether the proposed intervention is supported by appropriate evidence, permitted within the agent’s authority, and consistent with operating procedures and predefined constraints. The ADMIT framework provides a structured approach to such action-level evaluation through a sequence of deterministic checks, optionally supplemented by consequence projection (Misra et al., 2026). Alternatively, an LLM-based judge can examine the proposed intervention and associated process information and produce an acceptance decision without requiring an explicit process simulator (Zheng et al., 2023). These approaches difer in their information requirements, computational expense, rejection criteria, and ability to evaluate the consequences of an intervention. Whether these diferences translate into improved fault recovery and economic performance remains an important empirical question.

A controlled comparison requires a process benchmark in which candidate interventions have observable consequences for both process operation and protective systems. Established benchmarks such as the Tennessee Eastman process have supported extensive research on fault detection, process monitoring, and control (Downs and Vogel, 1993; Reinartz et al., 2021). However, conventional fault-detection benchmarks do not necessarily provide the complete supervisory environment needed to evaluate operator-level actions, including equipment redundancy, explicit action permissions, alarm handling, and independent protective interlocks. Furthermore, detection accuracy alone does not capture whether an intervention prevents a plant stop or preserves production. An action- validation benchmark must therefore represent process dynamics, fault mechanisms, the operational interface for executing decisions, and the criteria used to assess their consequences.

To address these requirements, this paper presents OperAItor, an open-source framework and benchmark for evaluating LLM-generated supervisory interventions during abnormal chemical process operation. The case study uses a hypothetical reactor–distillation process with physically based thermodynamics and modified liquid-phase reaction kinetics, represented by a 113-variable diferential–algebraic equation (DAE) model. It incorporates eight regulatory PI control loops, 22 process alarms, five duty/standby pump trains, and eleven simulated safety instrumented functions. The agentic architecture connects PCA-based monitoring, fault diagnosis, knowledge-graph retrieval, planning, and LLM-generated operator actions through a modular validation interface. The agents operate above the BPCS and are not permitted to bypass the independent protective logic.

The framework is used to compare five action-validation configurations: no validation beyond command executability, an LLM judge, a digital-twin validator, deterministic ADMIT admissibility gates, and ADMIT gates supplemented by consequence projection. These configurations are evaluated using four LLMs across twenty scenarios containing individual, coupled, and triple faults. All configurations use the same simulated plant, supervisory action interface, fault scenarios, and benchmark scoring procedure. Performance is evaluated using plant-stop frequency, operational economic performance, action rejection behaviour, and computational cost. An additional ofline admissibility analysis examines the recovery limitations imposed by the available supervisory actions. The digital-twin experiments use a matched-model setting, allowing the comparison to isolate validation behaviour without introducing plant–model mismatch.

The principal contributions of this work are:

1. An integrated benchmark for agentic fault-tolerant process operation. We develop an open dynamic reactor–distillation benchmark that combines process dynamics, regulatory control, alarm management, equipment redundancy, and simulated safety interlocks. A common operator action interface enables alternative supervisory decision-making methods to be evaluated under identical process conditions (Sections 2–3).

2. A modular framework for generating and validating supervisory interventions. OperAItor integrates statistical monitoring, fault diagnosis, process knowledge retrieval, planning, and LLM-based action generation with interchangeable validation mechanisms.

The framework separates action proposal, validation, and execution, allowing their individual roles to be examined without changing the underlying plant model (Section 4).

3. A comparative evaluation ofaction-validation strategies. Five validation configurations are assessed using four LLMs and twenty fault scenarios, including interacting faults. The evaluation quantifies how alternative validation mechanisms afect fault recovery, plant stops, operational economic performance, rejected interventions, and computational requirements (Sections 6–7).

4. An analysis of validation and recovery limitations. By examining rejection outcomes and ofline admissibility results, we distinguish limitations associated with the action-generation and validation mechanisms from faults that cannot be resolved through the available individual supervisory interventions. This analysis provides a basis for interpreting benchmark performance beyond action acceptance rates alone (Section 7).

LLM-based agents ofer opportunities to automate supervisory decisions in chemical process operations, particularly in responding to abnormal operating conditions and process faults. However, actions proposed by these agents may be inappropriate for the current plant state, violate operating constraints, or lead to undesirable process behaviour. Methods for validating proposed actions before their implementation are therefore an important component of agentic supervisory systems. Evaluating these methods requires a dynamic process environment in which the consequences of supervisory interventions can be assessed under repeatable fault conditions. This study introduces OperAItor, an open-source framework that combines a dynamic process simulator, fault scenarios, LLM-based supervisory agents, and interchangeable action-validation mechanisms. The framework enables systematic comparisons of agent and validation configurations in terms of operational performance, fault recovery, and plant stops. The simulator, agent framework, fault scenarios, and evaluation records are made publicly available to support reproducible research (AISL, Imperial College London, 2026a,b).

## 2. Case study: An integrated reactor–distillation benchmark

Evaluating an LLM-generated supervisory intervention requires a process in which the intervention has observable consequences for material inventories, operating constraints, and protective systems. We therefore consider the integrated reactor–distillation process in Fig. 1. The benchmark combines continuously evolving process states with discrete equipment failures, redundant pumps, regulatory control, and protective interventions. It is designed to assess fault recovery through operator-level actions rather than to reproduce the detailed design of a particular industrial facility.

The process represents the generic transformation � → $B { + } C$ , with the gaseous product � removed from the reactor and the liquid �–� mixture separated downstream. The species used in the simulation are isopropanol (IPA, �), acetone (�), and hydrogen (�), following the stoichiometry of IPA dehydrogenation (Turton et al., 2012; Luyben, 2011),

$$
\begin{array} { r } { ( \mathrm { C H } _ { 3 } ) _ { 2 } \mathrm { C H O H } \longrightarrow ( \mathrm { C H } _ { 3 } ) _ { 2 } \mathrm { C O } + \mathrm { H } _ { 2 } , \qquad \Delta H _ { r } = + 6 8 \mathrm { k J m o l } ^ { - 1 } . } \\ { ( 1 ) \qquad ( 1 ) \qquad } \end{array}
$$

Fresh IPA enters the jacketed continuous stirred-tank reactor R-101 through the duty/standby feed pumps P-101A/B. At the nominal operating point, the fresh feed is $1 0 0 \mathrm { k m o l h ^ { - 1 } }$ the reactor temperature is 335.74 K, the pressure is 3.0 bar, and the conversion is 70%. The heating jacket supplies the endothermic reaction duty, while a separate emergencycooling circuit can be actuated by the protective system. Hydrogen is discharged through the pressure-control valve PCV-101; an independently actuated blowdown valve and a pressure-relief-valve surrogate provide additional discharge paths.

The reactor efluent is transferred by P-102A/B to tray 6 of the ten-tray distillation column T-201. Overhead vapour is condensed in E-201 and collected in reflux drum V-201. The column has separate duty/standby trains for reflux, distillate withdrawal, and bottoms withdrawal (P-201A/B, P-202A/B, and P-203A/B, respectively). Cooling-water flow to the condenser and steam flow to the kettle reboiler E-202 provide the principal thermal manipulations. The nominal column pressure is 1.0 bar, with a reflux rate of $8 0 . 0 \mathrm { k m o l h ^ { - 1 } }$ , a distillate rate of $7 3 . 2 \mathrm { k m o l h ^ { - 1 } }$ , and a bottoms rate of $2 6 . 8 \mathrm { k m o l h ^ { - 1 } }$ Their acetone mole fractions are 0.942 and 0.040, respectively. The nominal fresh-feed rate corresponds to approximately 6.01 t/h of IPA, while 70% reactor conversion gives approximately 4.07 t/h of acetone formation. At the stated distillate rate and composition, the acetone product rate is approximately 4.0 t/h. The nominal stream calculations report an acetone-balance closure within $1 0 ^ { - 2 } \mathrm { k m o l h ^ { - 1 } }$ when evaluated using the unrounded simulated stream values.

Physical interpretation. The process uses thermophysical properties of the specified chemical species, while the dehydrogenation rate is modified to permit reaction in the liquid phase at approximately 336 K. Industrial IPA dehydrogenation normally takes place in the gas phase at substantially higher temperatures (Turton et al., 2012; Luyben, 2011). The reaction kinetics therefore serve as a dynamic surrogate for a hypothetical endothermic $A  B + C$ process. The model is used to evaluate causal process responses, operator-level interventions, and protective functions rather than to predict the operation of a commercial acetone plant.

## 2.1. Dynamic process model

The plant is represented by coupled component, totalholdup, and energy balances, with algebraic constraints for thermodynamics, hydraulics, and equipment behaviour.

![](images/92a788e06030671aa8de3aedfec7cfbe45cf1399375392b9b833340d01dc8584.jpg)

<table><tr><td rowspan=1 colspan=1>Stream</td><td rowspan=1 colspan=1>Description</td><td rowspan=1 colspan=1>Flow [kmol/h]</td><td rowspan=1 colspan=1>X acetone [mol/mol]</td><td rowspan=1 colspan=1>T [K]</td><td rowspan=1 colspan=1>P [bar]</td></tr><tr><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>Fresh feed</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.000</td><td rowspan=1 colspan=1>298.1</td><td rowspan=1 colspan=1>一</td></tr><tr><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>Reactor effluent / column feed</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>0.700</td><td rowspan=1 colspan=1>335.7</td><td rowspan=1 colspan=1>3.000</td></tr><tr><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1> ${ \sf H } _ { 2 }$ vent (PCV-101)</td><td rowspan=1 colspan=1>70.0</td><td rowspan=1 colspan=1>H2</td><td rowspan=1 colspan=1>335.7</td><td rowspan=1 colspan=1>3.000</td></tr><tr><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Overhead vapour</td><td rowspan=1 colspan=1>153.2</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>331.7</td><td rowspan=1 colspan=1>1.005</td></tr><tr><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Reflux</td><td rowspan=1 colspan=1>80.0</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>327.5</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>6</td><td rowspan=1 colspan=1>Distillate (product)</td><td rowspan=1 colspan=1>73.2</td><td rowspan=1 colspan=1>0.942</td><td rowspan=1 colspan=1>327.5</td><td rowspan=1 colspan=1>1.000</td></tr><tr><td rowspan=1 colspan=1>7</td><td rowspan=1 colspan=1>Bottoms</td><td rowspan=1 colspan=1>26.8</td><td rowspan=1 colspan=1>0.040</td><td rowspan=1 colspan=1>354.4</td><td rowspan=1 colspan=1>1.044</td></tr></table>

Figure 1: Process flow diagram and nominal stream data for the reactor–distillation benchmark. Five liquid services use duty/standby pump trains. Detailed instrumentation and protective interlocks are presented in Section 3.

Acetone is the light component and IPA the heavy component in the separation model. Vapour pressures are represented by a two-parameter Antoine correlation, and nonideal liquid-phase behaviour by an asymmetric Margules activity-coeficient model (Poling et al., 2001). The vapour– liquid equilibrium closure is

$$
P _ { \mathrm { b u b } } ( T , x ) = \sum _ { j = 1 } ^ { 2 } x _ { j } \gamma _ { j } ( x ) P _ { j } ^ { \mathrm { s a t } } ( T ) , \qquad y _ { 1 } = \frac { x _ { 1 } \gamma _ { 1 } ( x ) P _ { 1 } ^ { \mathrm { s a t } } ( T ) } { P } ,\tag{2}
$$

where $x _ { j }$ and $y _ { j }$ are liquid and vapour mole fractions. Liquid and vapour enthalpies use constant heat capacities and component heats of vaporisation. The complete property correlations and numerical parameter values are retained in the released implementation.

Reactor dynamics. Reactor R-101 has four diferential states: total liquid holdup �, acetone holdup $n _ { B } ,$ , internal energy $U _ { R } ,$ , and gas-phase hydrogen holdup $n _ { H _ { 7 } }$ . Reaction rate � depends on the unconverted IPA inventory and an Arrhenius rate coeficient,

$$
r = a k _ { \mathrm { r e f } } \exp \left[ { - \frac { E _ { a } } { R _ { g } } \left( \frac { 1 } { T _ { R } } - \frac { 1 } { T _ { \mathrm { r e f } , k } } \right) } \right] \left. M - n _ { B } \right. ,\tag{3}
$$

where $a = 1$ at nominal catalyst activity and $\begin{array} { r } { \langle z \rangle = \frac { 1 } { 2 } ( z + } \end{array}$ $\sqrt { z ^ { 2 } + \varepsilon ^ { 2 } } )$ is a smooth positive-part approximation with $\varepsilon =$ $1 0 ^ { - 1 0 }$ . The principal balances are

$$
\dot { M } = F - ( F _ { \mathrm { o u t } } + F _ { \mathrm { l e a k } } ) ,\tag{4}
$$

$$
\dot { n } _ { B } = r - ( F _ { \mathrm { o u t } } + F _ { \mathrm { l e a k } } ) x _ { R } ,\tag{5}
$$

$$
\dot { n } _ { H _ { 2 } } = r - F _ { \mathrm { v e n t } } ,\tag{6}
$$

$$
\begin{array} { r l } & { \dot { U } _ { R } = F h ^ { L } ( T _ { F } , 0 ) - ( F _ { \mathrm { o u t } } + F _ { \mathrm { l e a k } } ) h ^ { L } ( T _ { R } , x _ { R } ) } \\ & { \qquad - F _ { \mathrm { v e n t } } h _ { H _ { 2 } } ( T _ { R } ) + Q _ { j } - Q _ { \mathrm { l o s s } } - Q _ { \mathrm { e m } } - r \Delta H _ { r } . } \end{array}\tag{7}
$$

Here � is the feed rate, $F _ { \mathrm { o u t } }$ is the liquid efluent, and $F _ { \mathrm { l e a k } }$ is a fault-induced leak. Thejacket, ambient heat loss, and emergency cooling contribute $Q _ { j } , Q _ { \mathrm { l o s s } }$ , and $Q _ { \mathrm { e m } } .$ The algebraic relations include $x _ { R } ~ = ~ n _ { B } ^ { \mathrm { { ' } } } / M$ and $P _ { R } \ = \ n _ { H _ { 2 } } R _ { g } T _ { R } / V _ { g } ,$ with an efective gas capacitance represented through $V _ { g } .$ Hydrogen discharge depends on the actual openings of the pressure-control and blowdown valves and on a simplified relief-valve characteristic. These relations allow reduced vent capacity, pressure-sensor bias, and a heater stuck at high output to produce distinct dynamic responses.

Distillation and inventory dynamics. Column T-201 is a ten-tray binary equilibrium-stage model with a kettle reboiler, a reflux drum, and a lumped condenser-pressure state, following conventional dynamic column formulations (Luyben, 1990; Skogestad, 1997). Each tray � contains liquid holdup $M _ { i }$ , acetone holdup $n _ { i } .$ , and internal energy $U _ { i }$ . With liquid flowing downward and vapour upward, its material balances are

$$
\begin{array} { r } { \dot { M } _ { i } = L _ { i - 1 } + V _ { i + 1 } + F _ { i } - L _ { i } - V _ { i } , } \end{array}\tag{8}
$$

$$
\dot { n } _ { i } = L _ { i - 1 } x _ { i - 1 } + V _ { i + 1 } y _ { i + 1 } + F _ { i } z - L _ { i } x _ { i } - V _ { i } y _ { i } ,\tag{9}
$$

where $F _ { i }$ is non-zero only on feed tray 6. Tray energy balances account for the corresponding liquid, vapour, and feed enthalpy flows. Tray temperatures and phase compositions are determined algebraically from the vapour–liquid equilibrium relations, while liquid and vapour flows follow holdup- and pressure-dependent hydraulic correlations. The reboiler includes steam heating and bottoms withdrawal.

The condenser model includes finite cooling capacity. With cooling-water valve position $u _ { c }$ , efectiveness factor $\eta ,$ and top-tray temperature $T _ { 1 }$ , its available duty, condensation rate, and pressure dynamics are

$$
Q _ { \mathrm { a v a i l } } = U A _ { c } u _ { c } \eta \left. T _ { 1 } - T _ { c w } \right. ,\tag{10}
$$

$$
F _ { \mathrm { c o n d } } = \operatorname* { m i n } \left( \frac { Q _ { \mathrm { a v a i l } } } { \lambda _ { c } } , \langle V _ { 1 } + k _ { P } ( P _ { c } - 1 . 0 ) \rangle \right) ,\tag{11}
$$

$$
\dot { P } _ { c } = \frac { V _ { 1 } - F _ { \mathrm { c o n d } } } { C _ { P } } .\tag{12}
$$

Here $P _ { c }$ is column-overhead pressure, $\lambda _ { c }$ is condensation enthalpy, and $C _ { P }$ is an efective vapour capacitance. $\mathbf { A }$ reduction in $\eta$ limits condensation, raises $P _ { c }$ , and can initiate a protective high-pressure trip. The reflux-drum liquid inventory follows

$$
\dot { M } _ { d } = F _ { \mathrm { c o n d } } - ( R + D + F _ { \mathrm { l e a k } } ^ { d } ) ,\tag{13}
$$

where � and � denote reflux and distillate withdrawal. This balance makes pump trips and drum leakage directly relevant to the feasibility of supervisory recovery actions.

Coupling and numerical solution. The reactor liquid outflow supplies the column feed, giving a single semiexplicit, index-1 diferential–algebraic system,

$$
\mathbf { F } \big ( t , \mathbf { y } , \dot { \mathbf { y } } _ { D } , \mathbf { u } \big ) = \mathbf { 0 } , \qquad \mathbf { y } \in \mathbb { R } ^ { 1 1 3 } ,\tag{14}
$$

with 41 diferential and 72 algebraic variables. The differential states comprise four reactor states, three states on each of ten trays, three states each in the reboiler and reflux drum, and one condenser-pressure state. The system is integrated by implicit backward Euler at a nominal process step of 0.1 min. Each step is solved using Powell’s hybrid method with scaled residuals and adaptive step subdivision when convergence is dificult (Powell, 1970; Virtanen et al., 2020). A scaled residual norm below $1 0 ^ { - 5 }$ is required; if repeated subdivision and the final least-squares fallback fail, the episode is terminated as a numerical failure. Such terminations are counted as plant stops in the benchmark and are therefore distinguished from physical protective trips in the recorded outcomes. Regulatory controllers update every 1 min, while process observations used by the monitoring and evaluation layers are sampled every 0.5 min. Numerical details and the full parameter set are available with the simulator.

## 2.2. Fault mechanisms and disturbances

A central feature of the benchmark is the separation between a requested intervention and the action actually delivered to the process. For each manipulated variable, the simulator distinguishes the BPCS request, the supervisory request, the safety-arbitrated command, and the final plant input. Actuator and equipment faults afect this last stage. As a result, a safety interlock may demand a prescribed valve movement without being able to restore the failed final element. The arbitration rules and independent protective logic are described in Section 3.

The fault library contains 18 named fault mechanisms covering actuator, sensor, equipment, and process faults. Ten are used in the twenty-scenario evaluation campaign (Table 1). Equipment trips remove flow from the duty pump until the standby pump is started. Actuator faults impose capacity restrictions or fixed positions; sensor faults bias the measurements available to regulatory control; and process faults modify reaction activity, heat-transfer efectiveness, or material inventories. These interventions enter the differential and algebraic equations directly rather than being represented only as alarm labels.

To distinguish injected faults from normal operating variability, the benchmark also permits finite-duration disturbances: 15-min steps of ±3 K in cooling-water and ambient temperature, ±5 K in fresh-feed and heating-medium temperature, and ±10% in steam-header pressure, together with Gaussian measurement noise. The twenty benchmark scenarios define the combinations, onset times, and fault durations used for comparisons; these are specified in Section 6. The following section describes the regulatory and protective layers through which these physical faults become alarms, control responses, and potential plant stops.

## 3. Control and protection architecture

The reactor–distillation benchmark incorporates three operational layers with distinct responsibilities: a basic process control system (BPCS) for regulatory control, an alarm system for detecting and communicating abnormal conditions, and a simulated safety instrumented system (SIS) for protective intervention. The LLM-based operator framework introduced in Section 4 operates above these layers. It may modify permitted supervisory targets and initiate equipment operations, but it does not replace regulatory controllers or override the protective logic. This separation is essential for evaluating whether an agent-generated intervention is executable, operationally beneficial, and consistent with the constraints imposed by the plant.

Table 1  
Faults selected for the benchmark campaign. Machine-readable fault identifiers are provided with the released scenarios.
<table><tr><td>Fault</td><td>Class</td><td>Element</td><td>Injected mechanism</td></tr><tr><td>Feed pump trip</td><td>Equipment</td><td>P-101A</td><td>Duty pump stops; P-101B remains available.</td></tr><tr><td>Reflux pump trip</td><td>Equipment</td><td>P-201A</td><td>Duty pump stops; P-201B remains available.</td></tr><tr><td>Distillate pump trip</td><td>Equipment</td><td>P-202A</td><td>Duty pump stops; P-202B remains available.</td></tr><tr><td>Bottoms pump trip</td><td>Equipment</td><td>P-203A</td><td>Duty pump stops; P-203B remains available.</td></tr><tr><td>Reactor outlet capacity loss</td><td>Actuator</td><td>FV-102</td><td>Maximum achievable flow reduced to 20% of design.</td></tr><tr><td>Hydrogen vent degradation</td><td>Actuator</td><td>PCV-101</td><td>Achievable opening falls to 20% over 10 min.</td></tr><tr><td>Reactor heater stuck high</td><td>Actuator</td><td>TV-101</td><td>Heating factor fixed at 1.0 (nominal 0.61).</td></tr><tr><td>Reactor pressure bias</td><td>Sensor</td><td>PT-101</td><td>BPCS pressure measurement biased by -1.2 bar.</td></tr><tr><td>Condenser cooling loss</td><td>Process</td><td>E-201</td><td>Cooling effectiveness falls to  $\eta = 0 . 2 .$ </td></tr><tr><td>Reflux drum leak</td><td>Process</td><td>V-201</td><td>Liquid inventory loss of 100 kmol h⁻¹.</td></tr></table>

## 3.1. Regulatory control

The BPCS comprises eight single-loop proportional– integral (PI) controllers updated every $\Delta t _ { c } ~ = ~ 1$ min. The control structure follows conventional reactor and distillation control principles (Skogestad, 1997; Luyben, 2011). Reactor liquid inventory is regulated through its outlet flow, reactor temperature through jacket heating, and reactor pressure through hydrogen venting. For the distillation column, top-tray temperature is controlled through reflux, bottom temperature through reboiler steam, and column pressure through condenser cooling water. Reflux-drum and reboiler inventories are regulated by distillate and bottoms withdrawal, respectively. Fresh-feed throughput is specified separately through FIC-101 as a supervisory production target.

Each controller implements a positional PI law with a measurement deadband. Denoting the setpoint and measured process variable at control interval � by $S P _ { k }$ and $P V _ { k }$ , the control error and output are

$$
e _ { k } = \left\{ \begin{array} { l l } { 0 , } & { | S P _ { k } - P V _ { k } | \leq \delta , } \\ { S P _ { k } - P V _ { k } , } & { \mathrm { o t h e r w i s e } , } \end{array} \right.\tag{15}
$$

$$
\begin{array} { r } { u _ { k } ^ { \mathrm { B P C S } } = \mathrm { c l i p } _ { [ u _ { \mathrm { m i n } } , u _ { \mathrm { m a x } } ] } \left( b + K _ { p } e _ { k } + I _ { k } \right) , } \end{array}\tag{16}
$$

$$
I _ { k + 1 } = I _ { k } + \alpha _ { k } \frac { K _ { p } } { T _ { i } } e _ { k } \Delta t _ { c } ,\tag{17}
$$

where $K _ { p }$ is the proportional gain, $T _ { i }$ is the integral time, � is the nominal controller-output bias, and � is the deadband. The conditional integration indicator $\alpha _ { k } \in \{ 0 , 1 \}$ prevents further integral accumulation when the output is saturated and the proposed integral update would increase saturation. Bumpless transfer from manual to automatic operation is implemented by initializing the integral state as $I = u _ { \mathrm { m a n u a l } } -$ $b - K _ { p } e$ (Åström and Hägglund, 2006).

The controller pairings, tuning parameters, permissible supervisory setpoint ranges, and closed-loop response characteristics are summarised in Table 2. The setpoint limits define the range within which an operator or agent may request a change; they do not guarantee that the resulting process trajectory will remain within safe operating limits.

The regulatory loops were evaluated from the nominal steady state using setpoint changes at 5 min followed by a return to the original setpoint at 125 min, with measurement noise included. Figure 2 shows the resulting responses. All eight loops initially move in the intended control direction, settle within the specified tolerance without activating a protective interlock or priority-1 alarm, and return to the nominal operating point. Reported overshoots range from 0 to 20%, with settling times between 8 and 33 min. The reactor-pressure loop is the slowest, reflecting the efective hydrogen-gas capacitance represented in the model.

Disturbance-rejection tests were also performed for 40- min perturbations of the utility and feed conditions. A +3 K cooling-water temperature disturbance produced a maximum column-pressure deviation of 0.011 bar, while the cooling-water valve moved from approximately 60 to 69% opening. Fresh-feed temperature and steam-header pressure disturbances produced maximum controlled-temperature deviations of 0.16 and 0.35 K, respectively. In each case, the controlled variable subsequently returned to its setpoint. These tests establish the nominal regulatory response against which the fault-recovery interventions are evaluated.

## 3.2. Alarm management

The benchmark includes 22 configured alarms with priorities, activation limits, hysteresis, and acknowledgement states, following the basic alarm-management concepts of ISA-18.2 (ANSI/ISA, 2016). The alarm catalogue covers abnormal reactor pressure, temperature and liquid inventory; column pressure; reflux-drum, reboiler and tray inventories; product-quality deviations; and pump-train failures. Process alarms are assigned priorities 1–2, product-quality alarms priority 3, and pump-trip alarms priority 1.

Each alarm is activated when its corresponding measured variable crosses the configured threshold. Hysteresis prevents repeated activation and clearing near the limit. For example, a high alarm with threshold � and hysteresis ℎ becomes active at $P V \ \geq \ L$ and clears only after $P V ~ <$ $L - h .$ . The implemented alarm-state sequence distinguishes unacknowledged active alarms, acknowledged active alarms, and return-to-normal conditions that have not yet been acknowledged.

Regulatory control configuration and closed-loop step-test performance. OS denotes percentage overshoot and $t _ { s }$ the settling time within 20% of the imposed step, or two deadbands.
<table><tr><td>Loop</td><td>Controlled variable → manipulated variable</td><td> $K _ { p }$ </td><td> $T _ { i }$  [h]</td><td>Deadband</td><td>Setpoint range</td><td>OS [%]</td><td> $t _ { s }$  [min]</td></tr><tr><td>R_LIC</td><td>Reactor holdup → outlet flow</td><td>-6.0</td><td>2.00</td><td>0.50 kmol</td><td>40-60 kmol</td><td>4</td><td>9.5</td></tr><tr><td>R_TIC</td><td>Reactor temperature → heating</td><td>+0.030</td><td>0.12</td><td>0.30 K</td><td>325-340 K</td><td>6</td><td>10.5</td></tr><tr><td>R_PIC</td><td>Reactor pressure →  ${ \sf H } _ { 2 }$  vent</td><td>-0.10</td><td>0.30</td><td>0.015 bar</td><td>2.5–3.4 bar</td><td>0</td><td>33.0</td></tr><tr><td>TOP_TIC</td><td>Top-tray temperature → reflux</td><td>-7.0</td><td>0.15</td><td>0.30 K</td><td>325-340 K</td><td>8</td><td>10.5</td></tr><tr><td>COL_PIC</td><td>Column pressure → cooling water</td><td>-4.5</td><td>0.20</td><td>0.002 bar</td><td>0.90–1.12 bar</td><td>20</td><td>8.5</td></tr><tr><td>DR_LIC</td><td>Drum holdup → distillate</td><td>-12.0</td><td>2.00</td><td>0.30 kmol</td><td>16-40 kmol</td><td>8</td><td>8.0</td></tr><tr><td>BOT_TIC</td><td>Bottom temperature → steam</td><td>+0.0325</td><td>0.15</td><td>0.30 K</td><td>345-365 K</td><td>0</td><td>15.5</td></tr><tr><td>RB_LIC</td><td>Reboiler holdup → bottoms</td><td>-6.0</td><td>2.00</td><td>0.35 kmol</td><td>18–46 kmol</td><td>7</td><td>14.0</td></tr></table>

![](images/12215a328243cf951507e275331c13ef8a652e640c33d5060fadef8919dfc546.jpg)  
Figure 2: Closed-loop setpoint responses of the eight regulatory PI controllers. Setpoint changes are introduced at 5 min and reversed at 125 min. The traces include noisy BPCS measurements.

Alarm thresholds are selected to provide a warning region before corresponding protective limits are reached. Consequently, an abnormal condition may first generate an alarm, subsequently activate an interlock, and finally cause a plant stop if the process continues to deteriorate. The alarm system provides information to the operator or LLM agents but does not, by itself, determine a corrective action. An alarm acknowledgement also does not alter the underlying process state or clear a protective interlock.

This distinction matters for supervisory fault recovery. An agent may observe several simultaneous alarms caused by a single equipment fault, while the regulatory controllers may already be saturated in attempting to compensate. The appropriate intervention must therefore account for the underlying process condition rather than treating each alarm as an independent control objective.

## 3.3. Protective interlocks and plant-stop logic

The simulated SIS comprises eleven independently evaluated protective interlocks, denoted I1–I11. Each interlock monitors a specified process variable and, when its activation threshold is reached, imposes predefined overrides on the afected process commands. The interlocks are evaluated using the simulated physical process state independently of the potentially biased or noisy measurements supplied to the BPCS. This separation allows sensor faults to compromise regulatory control without simultaneously corrupting the protective measurements.

The interlocks are latched and require an explicit reset request after the monitored variable has returned inside its reset threshold. The reset hysteresis prevents an interlock from repeatedly activating and clearing close to its limit. Six of the eleven interlocks additionally initiate a latched plantstop event, which terminates the corresponding benchmark episode. Table 3 summarises the activation conditions, principal protective actions, and plant-stop consequences. The complete actuator-level cause-and-efect matrix is retained in the supplementary material and the software release.

Table 3  
Summary of the eleven protective interlocks. A plant-stop latch terminates the benchmark episode. The full cause-and-efect matrix specifies the actuator-level overrides applied by each interlock.
<table><tr><td>ID</td><td></td><td>Interlock Activation</td><td>Reset condition</td><td>Principal protective action</td><td>Plant stop</td></tr><tr><td>11</td><td>R_T_HH</td><td> $T _ { R } \geq 3 4 8 ~ \mathsf { K }$ </td><td> $T _ { R } < 3 4 5 ~ \mathsf { K }$ </td><td>Stop reactor heating; activate emergency cooling</td><td>No</td></tr><tr><td>12</td><td> ${ \mathsf { R } } _ { - } { \mathsf { P } } _ { - } { \mathsf { H H } }$ </td><td> $P _ { R } \geq 4 . 0 0 ~ \mathsf { b a r }$ </td><td> $P _ { R } < 3 . 8 0 ~ \mathsf { b a r }$ </td><td>Stop heating; fully open hydrogen vent and blowdown</td><td>No</td></tr><tr><td>13</td><td>R_P_HHH</td><td> $P _ { R } \geq 4 . 3 5 { \mathrm { ~ b a r } }$ </td><td> $P _ { R } < 4 . 1 3 ~ { \mathsf { b a r } }$ </td><td>Limit feed to 40%; stop heating; open vent and blowdown</td><td>Yes</td></tr><tr><td>14</td><td>R_L_HH</td><td> $M _ { R } \geq 7 2 ~ \mathrm { k m o l }$ </td><td> $M _ { R } < 6 8 . 4 ~ \mathrm { k m o l }$ </td><td>Stop fresh feed and reactor heating</td><td>No</td></tr><tr><td>15</td><td> $R \_ L \_ L \ L$ </td><td> $M _ { R } \leq 2 0 ~ \mathrm { k m o l }$ </td><td> $M _ { R } > 2 1 ~ { \ k m o l }$ </td><td>Stop reactor heating and reactor liquid outlet</td><td>No</td></tr><tr><td>16</td><td> ${ \mathsf { R B } } _ { - } { \mathsf { L } } _ { - } { \mathsf { H H } }$ </td><td> $M _ { r b } \geq 5 1 . 7 6 ~ \mathsf { k m o l }$ </td><td> $M _ { r b } < 4 9 . 2 ~ \mathsf { k m o l }$ </td><td>Stop feed, reactor heating, and reactor outlet</td><td>Yes</td></tr><tr><td>17</td><td> $R 8 \_ L \_ L \lfloor$ </td><td> $M _ { r b } \leq 1 2 . 0 8$  kmol</td><td> $M _ { r b } > 1 2 . 7 ~ \mathsf { k m o l }$ </td><td>Stop bottoms withdrawal and reboiler steam</td><td>No</td></tr><tr><td>18</td><td> ${ \mathsf { D R \_ L _ { - } H H } }$ </td><td> $M _ { d } \ge 4 5 . 3 7$  kmol</td><td> $M _ { d } < 4 3 . 1 ~ \mathsf { k m o l }$ </td><td>Stop feed, reactor heating, reactor outlet, and steam</td><td>Yes</td></tr><tr><td>19</td><td> $D R \_ L \_ L L$ </td><td> $M _ { d } \leq 1 0 . 5 9$  kmol</td><td> $M _ { d } > 1 1 . 1 ~ { \mathsf { k m o l } }$ </td><td>Stop feed, reactor heating, reactor outlet, reflux, distillate, and steam</td><td>Yes</td></tr><tr><td>110</td><td> ${ \mathsf { C O L } } _ { - } { \mathsf { P } } _ { - } { \mathsf { H H } }$ </td><td> $P _ { c } \ge 1 . 3 5 \ \mathsf { b a r }$ </td><td> $P _ { c } < 1 . 2 8 ~ \mathsf { b a r }$ </td><td>Stop feed, reactor heating, reactor outlet, and steam; fully open cooling water</td><td>Yes</td></tr><tr><td>I11</td><td> $\mathtt { C O L \_ L _ { - } L } \mathtt { L L }$ </td><td> $M _ { 1 0 } \leq 0 . 3 0$  kmol</td><td> $M _ { 1 0 } > 0 . 3 1 5$ </td><td>kmol Stop feed, reactor heating, reactor outlet, and steam</td><td>Yes</td></tr></table>

The protective responses are defined for the benchmark rather than obtained from a formal hazard and operability study or safety integrity level determination. In particular, the reactor-pressure protection uses two stages. At the first high-high pressure threshold, heating is removed and gas discharge capacity is maximised while fresh feed is maintained. This configuration preserves the feed-associated thermal load of the endothermic reactor. At the escalated high-high-high threshold, feed is additionally restricted and a plant-stop event is latched. The thresholds and override actions are therefore properties of the simulated benchmark, not recommended settings for an industrial installation.

The protective logic is implemented independently of the LLM decision workflow. The agents cannot modify trip thresholds, suppress the protective calculations, or bypass their commanded overrides. Nevertheless, an imposed protective command does not guarantee the corresponding physical response. For example, an interlock may demand that PCV-101 open fully, while a vent-capacity fault prevents the actual discharge capacity from increasing to the requested level. The benchmark thus distinguishes correct protective decision logic from successful execution by the afected equipment.

## 3.4. Supervisory authority and command arbitration

The interface between supervisory decisions and the physical process is explicitly defined. For each manipulated input, the simulator distinguishes four successive command stages:

![](images/d08f2ccb28a8b1c5a08f8328e5b0942fa03d2695297a6f5d8d28a2fbab71863e.jpg)

(18)

The BPCS determines regulatory commands from process measurements and controller setpoints. Supervisory requests can change permitted setpoints, production throughput, standby-equipment states, and auxiliary equipment operation. The protective layer then applies any active interlock overrides before the commanded value reaches the plant. Finally, actuator or equipment faults determine whether the final element can deliver the requested action.

Conceptually, this ordering can be represented as

$$
\begin{array} { r } { u ^ { \mathrm { s u p } } = \mathcal { A } _ { \mathrm { s u p } } \left( u ^ { \mathrm { B P C S } } , a \right) , } \end{array}
$$

$$
\boldsymbol { u } ^ { \mathrm { S I S } } = \boldsymbol { A } _ { \mathrm { S I S } } \left( \boldsymbol { u } ^ { \mathrm { s u p } } , \boldsymbol { q } \right) ,\tag{19}
$$

$$
\boldsymbol { u } ^ { \mathrm { a c t } } = \boldsymbol { F } \left( \boldsymbol { u } ^ { \mathrm { S I S } } , \boldsymbol { f } \right) ,\tag{20}
$$

(21)

where � denotes the permitted supervisory request, � the active interlock states, and � the equipment and actuator fault conditions. The maps $\mathcal { A } _ { \mathrm { s u p } } , \mathcal { A } _ { \mathrm { S I S } }$ , and  represent supervisory command processing, protective arbitration, and physical fault efects, respectively. Equation (21) describes the order of authority and is not an additional dynamic model.

This structure imposes two important constraints on autonomous operation. First, supervisory interventions are restricted to the operator action interface; an agent cannot directly assign arbitrary values to physical plant inputs or alter fault variables. Second, even an accepted and executable action remains subordinate to active protective commands and physical equipment limitations.

An illustrative example is a reduction in reactor outlet capacity. The reactor-level controller may increase its requested outlet opening, while the capacity fault in the valve restricts the actual flow. A supervisory setpoint adjustment may then reduce the rate of inventory accumulation, but it cannot make the physical fault in the plant go away. Similarly, starting a healthy standby pump can recover flow after a duty-pump trip, whereas commanding a failed valve to open further may have no efect. These distinctions allow the benchmark to evaluate the logical validity of proposed operator actions and their consequences in the simulated plant.

![](images/4f48261e274ec569c960da0c9b1a09d1964acec506902031c2693c363e0d70a3.jpg)  
Figure 3: Piping and instrumentation diagram of the integrated reactor–distillation benchmark. Dashed grey lines represent BPCS signals, dash-dotted red lines represent simulated protective signals, and numbered diamonds identify interlocks I1–I11.

The detailed supervisory action grammar, including permitted command types and action validation, is introduced in Section 4.

## 3.5. Piping and instrumentation representation

Figure 3 presents the benchmark piping and instrumentation diagram (P&ID), using instrumentation conventions based on ISA-5.1 (ANSI/ISA, 2009). The diagram identifies the process equipment, measurement points, eight regulatory control loops, final elements, protective switches, and eleven interlocks. Regulatory control signals and protective command paths are represented separately to distinguish the BPCS and SIS responsibilities.

The P&ID and cause-and-efect documentation are generated from the simulator’s equipment and protective-logic definitions. Automated consistency checks compare the documented interlock behaviour with the implemented logic, reducing the possibility of discrepancies between the benchmark description and the executable simulator. The software also provides an editable diagram and an interactive P&ID view through the operator console.

Together, the regulatory, alarm, and protective layers establish the operational environment in which the agentic framework is evaluated. The following section describes how OperAItor uses process observations and plant knowledge to generate supervisory actions, and how alternative validation mechanisms determine which proposals are dispatched to this interface.

## 4. OperAItor: an agentic framework for supervisory fault recovery

## 4.1. Architecture and execution workflow

OperAItor is an agentic supervisory framework that converts process observations into operator-level interventions during abnormal operation. It operates above the BPCS and simulated SIS introduced in Section 3: regulatory controllers continue to regulate the process, protective interlocks retain priority over commanded actions, and the agent is restricted to the supervisory interface. The objective is not to replace these layers, but to connect fault evidence with an executable response when regulatory control alone cannot restore acceptable operation.

Figure 4 illustrates the workflow, implemented as a LangGraph state machine (LangChain, Inc., 2024). At a decision point, the monitoring module supplies abnormality statistics and process trends; the diagnosis module forms fault hypotheses; knowledge-graph retrieval identifies associated equipment, constraints and candidate interventions;

![](images/c1b7d1035b2d8f923380533db87b02eee1d836c211b5adb8c36882716bfd5c96.jpg)  
Figure 4: OperAItor supervisory workflow. Plant observations support monitoring, diagnosis, knowledge-graph retrieval, planning and LLM action generation. Proposed actions are screened for executability and passed to the selected validation layer before dispatch through the command-arbitration hierarchy of Section 3.4. Rejected proposals can be returned for revision.

the planning module constructs an operational brief; and an LLM action agent proposes a structured plan. The plan then passes through a common executability check and a selected action-validation mechanism before dispatch. When a proposal is rejected, the validation outcome is returned to the action agent through a reprompting edge. Section 5 describes the validators and their retry and fallback rules; the present section describes the common proposal-generation architecture. Configuration-specific choices of deterministic or LLM planning are reported with the evaluation protocol.

The plant simulator and the agent workflow execute as separate asynchronous tasks. For the comparative benchmark, simulated process time is held fixed during each agent decision. Consequently, LLM inference and validation latency contribute to measured wall-clock decision cost but do not allow the simulated plant to evolve before an action is applied. The software also permits asynchronous execution with a continuously advancing plant clock, but that operating mode is not the basis of the results reported here. This distinction avoids interpreting benchmark outcomes as demonstrations of real-time decision-making under communication and inference delays.

## 4.2. Process monitoring and decision triggers

The monitoring module uses principal component analysis (PCA) to identify departures from normal operating behaviour (MacGregor and Kourti, 1995; Qin, 2003). Its input vector contains 42 features: 34 process measurements covering reactor and column temperatures, pressures, inventories and compositions, including the ten tray temperatures and pressures, together with the outputs of the eight regulatory controllers. Including controller outputs allows the monitoring system to observe compensatory control efort as well as deviations in the measured process variables.

The PCA model is fitted to 3 040 samples collected at 0.5-min intervals during normal-operation simulations. These include nominal operation, three runs with randomised disturbance steps, steady operation at 70% and 85% throughput, and positive and negative variations of the configured disturbances. For a standardised observation � = (� − �̄) ⊘ �, retained loading matrix � and diagonal eigenvalue matrix �, the monitoring statistics are

$$
\mathbf { t } = \mathbf { z } \mathbf { P } ,
$$

$$
T ^ { 2 } = \mathbf { t } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } \mathbf { } ^ { - 1 } \mathbf { } \mathbf { } \mathbf { t } ^ { \top } ,\tag{22}
$$

(23)

$$
Q = \left\| \mathbf { z } - \mathbf { t P } ^ { \top } \right\| _ { 2 } ^ { 2 } ,\tag{24}
$$

where $T ^ { 2 }$ measures variation within the retained PCA subspace and �, also termed the squared prediction error (SPE), measures the residual outside that subspace. The model retains 23 components, explaining 95.6% of the trainingset variance. For each statistic, the detection threshold is selected as the larger of its theoretical 99th-percentile control limit and the empirical 99th percentile of the training statistic. The resulting thresholds are $T _ { \mathrm { l i m } } ^ { 2 } \ = \ 6 7 . 9$ and $Q _ { \mathrm { l i m } } = 5 . 0 7$ ; the corresponding theoretical limits are 42.1 and 4.78, respectively (Hotelling, 1947; Jackson and Mudholkar, 1979; Nomikos and MacGregor, 1995). The empirical comparison accommodates departures from Gaussian assumptions in the disturbance-containing training data.

To support fault attribution, the monitor ranks variables using their individual contributions to the two statistics:

$$
c _ { i } ^ { T ^ { 2 } } = z _ { i } \big ( \mathbf { P } \mathbf { A } ^ { - 1 } \mathbf { P } ^ { \top } \mathbf { z } ^ { \top } \big ) _ { i } ,\tag{25}
$$

$$
c _ { i } ^ { Q } = \big ( z _ { i } - ( \mathbf { t P } ^ { \top } ) _ { i } \big ) ^ { 2 } ,\tag{26}
$$

$$
s _ { i } = \frac { \operatorname* { m a x } ( c _ { i } ^ { T ^ { 2 } } , 0 ) } { T _ { \mathrm { l i m } } ^ { 2 } } + \frac { c _ { i } ^ { Q } } { Q _ { \mathrm { l i m } } } .\tag{27}
$$

A fault is confirmed when at least one monitoring statistic exceeds its threshold for three consecutive samples. In the benchmark’s initial decision protocol, PCA confirmation is the decision trigger; an alarm, pump-trip indication or saturated controller does not independently trigger an agent decision. These signals nevertheless enter the subsequent diagnosis as evidence. The monitor also supplies local trends and a linear extrapolation of time to an alarm or protective limit, $\tau = | L - x | / | \dot { x } |$ for a non-zero estimated rate directed towards the limit. This quantity is a short-horizon diagnostic indicator, not a validated prediction of trip time. Decision triggering and the episode timing protocol are specified in Section 6.

## 4.3. Fault diagnosis and evidence assembly

The diagnosis stage translates statistical deviations and operational signals into ranked fault hypotheses. OperAItor provides two alternatives. A deterministic heuristic compares the highest PCA contributions, active alarms, failed pump indications and saturated loops against the symptom signatures in the fault catalogue. Alternatively, an LLM diagnosis agent receives the same operating evidence together with the fault catalogue and returns a structured ranking of candidate faults.

The LLM diagnoser is given the current $T ^ { 2 }$ and � values relative to their limits, the variables contributing most strongly to each statistic, active alarms, controller saturation, pump states, measured trends and estimated times to relevant limits. It is constrained to hypotheses named in the catalogue and is asked to identify the fault class (sensor, actuator, equipment or process), evidence supporting and contradicting each hypothesis, and checks an operator could perform to discriminate among them. Hypotheses are therefore accompanied by an explicit evidence trail rather than being passed downstream as unsupported fault labels.

A diagnosis remains an inference, not a verified statement about the injected fault. In particular, an incorrect or incomplete diagnosis can afect which interventions are considered by the planner. The action-validation mechanisms are evaluated downstream of this imperfect information pipeline. In the matched-model digital-twin configuration, however, the rollout uses a clone of the true simulated plant state; this oracle information is a specific benchmark assumption discussed in Section 5.

## 4.4. Process knowledge graph and recovery planning

OperAItor represents plant-specific knowledge in a process knowledge graph (KG) (Hogan et al., 2021). The graph comprises 680 nodes and 1 157 relations, serialised as JSON-LD/TriG and organised using concepts from the VDI 3682 formalised process description and the SOSA sensor/actuator ontology (VDI/VDE, 2015; Janowicz et al., 2019). It encodes equipment connectivity, 40 observable and 11 actuatable process properties, eight controllers, five pump trains, 22 alarms and eleven protective interlocks with 37 override rules. The graph additionally includes 16 represented fault modes linked to 20 symptoms and operator actions linked to the equipment or variables they afect.

The KG preserves the distinction between requested, supervisory, safety-arbitrated and actual actuator signals introduced in Section 3.4. This is important when a fault changes the physical response to an otherwise valid command. For example, a controller and a protective interlock may both demand additional hydrogen venting without being able to overcome a vent-capacity restriction. A translation layer links graph identifiers to the corresponding simulator entities and legal supervisory commands, making retrieved relationships actionable rather than merely descriptive.

At each decision, retrieval begins with the three highestranked diagnostic hypotheses, active alarms, saturated loops, failed pump trains and the most influential PCA variables. A typed graph traversal expands one hop in both directions over relevant equipment, property, fault and protection relationships. Actions connected to retrieved entities through an actsOn relation are then added to the candidate context. This provides a bounded, plant-specific form of knowledgegrounded retrieval rather than passing the entire graph to the LLM (Lewis et al., 2020; Pan et al., 2024).

A planning module converts the diagnosis and retrieved context into an operational brief comprising the suspected fault, supporting symptoms, recovery objective, feasible supervisory levers and constraints on their use. Either a deterministic KG-based planner or an LLM planner may generate this brief from the same evidence. For example, a diagnosed reflux duty-pump trip can retrieve the associated standby train and the START\_SPARE command; a saturated columnpressure controller instead suggests that further setpoint or valve demands may not deliver the intended process response. The planner supplies candidate actions and their rationale, but does not issue commands to the plant. The benchmark configuration of planner and diagnoser is documented with the experimental protocol in Section 6.

## 4.5. Action generation and supervisory interface

The action agent receives the current plant snapshot, the ranked diagnostic hypotheses, the planning brief, the retrieved KG evidence, the controller directions and operating limits, and a machine-readable action grammar. It returns a JSON plan with at most three commands and a justification. A plan can combine interventions, for example starting a standby pump while adjusting a supervisory target. Every command belongs to the permitted interface in Table 4. All commands in the workflow come from the LLM while the validation and dispatch stages decide whether they are accepted and executed.

Supervisory action interface exposed to the OperAItor action agent. Reset requests and alarm acknowledgements are additional interface operations, rather than direct plant-input assignments.
<table><tr><td>Command</td><td>Permitted effect</td></tr><tr><td>SET_SETPOINT</td><td>Change the target of one of the eight PI loops within the range in Table 2.</td></tr><tr><td>SET_THROUGHPUT</td><td>Change the fresh-feed pro- duction target through FIC- 101.</td></tr><tr><td>START_SPARE</td><td>Start the available standby pump of a specified duty/s- tandby train.</td></tr><tr><td>OPERATE_AUXILIARY</td><td>Request a permitted auxiliary operation, such as hydrogen blowdown or emergency cool-</td></tr><tr><td>NO_ACTION</td><td>ing. Make no supervisory change at the current decision point.</td></tr></table>

The interface also supports alarm acknowledgement and requests to reset eligible latched non-trip interlocks once their reset conditions are satisfied. It does not permit the agent to change PI tuning parameters, write physical plant inputs, manipulate fault or disturbance variables, change protection thresholds or bypass the SIS. Thus, a proposed intervention may be operationally meaningful without granting the agent unrestricted access to the simulation state.

Before reaching the selected validator, a plan undergoes an executability check against the action schema and the current interface constraints. Unknown action types or targets, out-of-range setpoints and malformed requests are rejected and returned for correction. Passing this check means only that the request can be represented and dispatched; it does not establish that the action will recover the process or avoid a protective trip. That distinction motivates the validation methods examined in Section 5.

For the benchmark, inference settings and model choices are held fixed within each experimental configuration. GPT-5-family reasoning models are called with low reasoning efort, while locally served open models use Ollama at temperature zero (Ollama, 2024). These implementation settings afect reproducibility and decision cost but do not alter the supervisory authority described above.

In summary, OperAItor establishes a common path from evidence of abnormal operation to candidate operator actions. Monitoring and diagnosis determine what the agent believes has happened; KG retrieval and planning determine which interventions appear relevant; and the action agent converts those interventions into executable requests.

The following section compares mechanisms for deciding whether these requests should be applied at the same supervisory interface; configuration-specific choices of planner are made explicit in the benchmark protocol.

## 5. Action validation methods

The principal experimental variable in this study is the mechanism used to evaluate an LLM-generated operator action before it is dispatched to the simulated process. OperAItor separates action generation from validation: the common agent workflow described in Section 4 produces an operator-level proposal, and a selected validation configuration determines whether the proposal may proceed through the supervisory command interface. The regulatory controllers and protective interlocks remain active irrespective of the selected validator. Thus, admission by a validator does not bypass the final command arbitration described in Section 3.4.

The validation problem is distinct from syntactic executability. A command may refer to a known controller, respect its setpoint range, and be accepted by the interface while still worsening a process inventory or accelerating a protective trip. Conversely, rejecting every executable command would prevent agent-induced changes but also eliminate opportunities for recovery.

We therefore examine five configurations that impose diferent requirements on a proposed intervention: executability only, review by a second LLM, prediction with a digital twin, deterministic ADMIT admissibility checks, and ADMIT checks supplemented by consequence projection. Table 5 summarises their information requirements, acceptance criteria, and fallback behaviour.

## 5.1. Common proposal and validation interface

At decision point �, the action agent receives the operating evidence assembled by the preceding modules and submits a proposal $a _ { k }$ containing at most three commands from Table 4. Each proposal includes a justification; AD-MIT configurations additionally require structured action records as described below. Before reaching any of the five validators, a common executability check verifies the command types and targets, parameter formats, permitted setpoint ranges, and current interface constraints. Proposals that fail this check are rejected without being dispatched. Passing executability establishes only that the request can be represented at the operator interface, not that it is appropriate for the current fault.

The selected validator receives the proposal and the information available to that configuration. This information may include the current plant snapshot, alarm and interlock states, process trends, diagnostic hypotheses, the planning brief, and retrieved knowledge-graph (KG) context. Importantly, the configurations do not have identical information requirements: the LLM judge reviews the supplied evidence, ADMIT evaluates an action record against a trusted ledger and safety case, and the digital twin evaluates predicted process trajectories. A rejection is returned to the action agent with feedback that may support a revised proposal. At most four proposals are permitted within one decision, comprising the initial proposal and up to three subsequent attempts. The configuration-specific behaviour when these attempts are exhausted is described below.

Action-validation configurations. All proposals pass the common executability check before the listed validation mechanism. A “no action” fallback means that no new supervisory command is dispatched at that decision point; existing BPCS and protective functions continue to operate.
<table><tr><td>Configuration</td><td>Additional requirement</td><td>Decision criterion</td><td>After unsuccessful attempts</td></tr><tr><td>No validation</td><td>None</td><td>Dispatch the first executable proposal without evaluating its operational consequences.</td><td>No validation-specific fallback.</td></tr><tr><td>LLM judge</td><td>Second LLM call</td><td>Accept when the judge approves the proposed intervention against written fault-relevance, direction, proportionality, and protective-limit criteria.</td><td>No new action is dispatched.</td></tr><tr><td>Digital twin</td><td>30-min process rollout</td><td>Prefer a proposal that restores the active recovery targets without violating the protective envelope or creating new high-priority alarms; otherwise compare predicted mitigations against no action.</td><td>Dispatch the best eligible mitigation only if it improves on the no-action projection.</td></tr><tr><td>ADMIT T0-T3</td><td>Trusted evidence ledger and action safety case</td><td>Check record integrity (T0), admissible evidence (T1), hazard representation (T2), and authority, operating mode, procedures, confidence, and reversibility (T3)</td><td>No new action is dispatched.</td></tr><tr><td>ADMIT T0-T4</td><td>ADMIT T0-T3 plus a 30-min rollout</td><td>Apply T0-T3 and reject a projected protective-limit violation or a sufficient loss of safety margin relative to no action (T4).</td><td>No new action is dispatched.</td></tr></table>

## 5.2. Executability-only and LLM-judge baselines

The executability-only configuration is the unvalidated baseline. It applies the first proposal that passes the common interface checks, without asking whether the plan addresses the diagnosed fault or whether its predicted efects are favourable. This establishes the outcome of the actiongeneration pipeline without an additional validation step. The absence of an explicit validator does not imply unrestricted control: the BPCS, supervisory action grammar, and protective arbitration remain in force.

In the LLM-judge configuration, a second call to the same underlying model reviews the proposal before dispatch. The judge receives the candidate commands, current plant snapshot, diagnosis, planning brief, KG evidence, and command limits. Its written criteria ask whether the intervention addresses the suspected fault, respects control directions, avoids moving the process towards a protective limit, and is proportionate to the observed condition. The judge returns a structured decision containing accept, confidence, reason, risks, and suggestion. Only an accepted proposal is dispatched; a rejected proposal or failed judge call produces rejection feedback to the action agent. If no proposal is accepted within the permitted attempts, no new action is applied.

This baseline provides an example of model-based linguistic review, rather than a physical consequence check: the judge reasons over the supplied description and observations but does not simulate the plant or independently verify an engineering safety case. Using the same model family for proposal and review may also leave the two calls vulnerable to correlated errors. Its measured cost includes the additional LLM inference and any proposals generated after rejection.

## 5.3. Digital-twin consequence validation

The digital-twin validator tests candidate plans using a cloned state of the dynamic plant. Each candidate, including NO\_ACTION, is evaluated over a fixed horizon of � = 30 min. The rollout begins from the simulator state at the decision point, preserves faults active at that instant, and excludes subsequently scheduled events, including future fault removal. The twin therefore estimates the consequences of continuing under the current fault conditions rather than relying on knowledge of when those faults will disappear. In the reported benchmark, the twin is an oracle matchedmodel clone: it uses the same physical model and underlying state as the simulated plant, not a separately identified model with state-estimation or parameter uncertainty.

A candidate is classified as restoring operation when the recovery targets identified at the decision point return to their acceptable conditions and remain there for at least 2 min during the rollout, without violating the simulated protective envelope or producing a new priority-1 or priority-2 alarm. The targets include active process-alarm conditions, pumprelated operating states, and, when PCA has confirmed the fault, the requirement that both $T ^ { 2 }$ and squared prediction error � return below their respective limits. The fullrestoration criterion is stricter than avoiding an immediate trip: a proposal can keep the plant operating for 30 min without eliminating every fault signature.

When full restoration is not predicted, the validator compares candidate mitigations with the NO\_ACTION rollout. Its lexicographic evaluation orders candidates using

$$
\begin{array} { r l } & { \ell ( a ) = \big ( \mathbb { 1 } [ \mathrm { t r i p } ] , - t _ { \mathrm { t r i p } } ( a ) , \ N _ { \mathrm { u n m e t } } ( a ) , } \\ & { \qquad f _ { \mathrm { u n s a f e } } ( a ) , \ J _ { \mathrm { P C A } } ( a ) \big ) , } \end{array}\tag{28}
$$

$$
J _ { \mathrm { P C A } } ( a ) = \frac { T ^ { 2 } ( a ) } { T _ { \mathrm { l i m } } ^ { 2 } } + \frac { Q ( a ) } { Q _ { \mathrm { l i m } } } .\tag{29}
$$

Here $t _ { \mathrm { t r i p } }$ is the projected trip time when a trip occurs, $N _ { \mathrm { u n m e t } }$ counts unresolved recovery targets, $f _ { \mathrm { u n s a f e } }$ is the fraction of the projected trajectory outside the specified acceptable operating region, and the final term summarises the remaining normalised monitoring deviation. Non-tripping trajectories are preferred to tripping trajectories; among the latter, later trips are preferred. The tuple is compared lexicographically rather than combined into a weighted score. Terms involving trip time are relevant only when a trip is predicted.

After the available proposals have been considered, the best eligible mitigation may be dispatched if it improves on the no-action projection; otherwise no new action is applied. When a mitigation rather than full restoration is selected, the framework schedules a further decision after 10 min. This fallback is an important distinction between the digital twin and the other validators: a rejected full-recovery proposal does not necessarily imply that the plant receives no intervention. The ranking and mitigation rule are specific to the implemented benchmark and do not constitute a formal guarantee of constraint satisfaction beyond the modelled horizon.

## 5.4. ADMIT admissibility and consequence gates

The two ADMIT configurations evaluate the action record accompanying each proposed intervention rather than relying solely on the action-agent’s explanation. They implement the Admissible Decisions for Machine Interventions (ADMIT) framework (Misra et al., 2026; Gutta et al., 2026). The platform assembles an auditable, hash-chained ledger from the current measurements, PCA statistics, alarm and interlock states, pump conditions, and process trends. KG facts are also available as trusted context. LLM-generated diagnostic statements are treated as untrusted hypotheses and cannot, by themselves, satisfy a required evidence citation.

Each proposed command is associated with an action class, required authority, permitted operating modes, supporting evidence, applicable standard operating procedure (SOP) steps, confidence requirement, reversibility category, and any required recovery plan. The agent supplies a record referencing the evidence it relies on, the checks performed, the hazard addressed, and the rationale for the proposed intervention. For example, a supervisory setpoint change requires relevant process evidence and procedural checks that the loop is in automatic operation and its measurement is verified. Opening the reactor blowdown valve is treated as a higher-consequence intervention requiring pressurerelated evidence and a recovery plan. Bypassing a protective interlock is not within the permissible operator action set.

The deterministic checks are organised as a sequence of gates:

T0: Record integrity. Verify the completeness and structural validity of the submitted action record and its references.

T1: Evidence. Verify that cited evidence is present in the trusted ledger and satisfies the requirements associated with the proposed command.

T2: Hazard. Verify that the record identifies the relevant hazard and provides the required hazard-related information.

T3: Operational admissibility. Check command authority, current operating mode, required SOP steps, confidence and reversibility conditions, and recovery-plan requirements where applicable.

T4: Consequence projection. When enabled, compare a 30-min rollout of the proposed action with the corresponding no-action projection and check projected protective-limit and safety-margin conditions.

The operating mode is classified as normal, upset (a PCA-confirmed fault or latched interlock), or emergency (following a plant trip). Mode-specific permissions restrict which actions are admissible; for example, supervisory setpoint changes are not permitted in emergency mode. Gates T0–T3 evaluate the action’s documented admissibility without predicting its physical efect. Consequently, an action can pass T0–T3 yet still have an undesirable process consequence. The T0–T3 configuration intentionally retains this limitation so that the contribution of consequence projection can be examined separately.

The T0–T4 configuration adds a 30-min model-based projection. Let $s _ { v }$ be a monitored process variable with nominal value $s _ { v } ^ { \mathrm { n o m } }$ and relevant upper or lower protective limit $L _ { v } . \mathrm { A }$ normalised margin is defined by

$$
m _ { v } ( s ) = \left\{ \begin{array} { l l } { \displaystyle \frac { L _ { v } - s _ { v } } { L _ { v } - s _ { v } ^ { \mathrm { n o m } } } , } & { \mathrm { u p p e r ~ l i m i t } , } \\ { \displaystyle \frac { s _ { v } - L _ { v } } { s _ { v } ^ { \mathrm { n o m } } - L _ { v } } , } & { \mathrm { l o w e r ~ l i m i t } , } \end{array} \right. \qquad m ( s ) = \displaystyle \operatorname* { m i n } _ { v } m _ { v } ( s ) .\tag{30}
$$

The scaling gives $m _ { v } = 1$ at the nominal point and $m _ { v } = 0$ at the corresponding limit, with negative values indicating a limit violation. T4 rejects a proposal if its projected trajectory crosses a protective limit or if it degrades the minimum margin by more than 0.2 relative to no action:

$$
\operatorname* { m i n } _ { 0 \le t \le H } m \bigl ( s _ { \mathrm { p l a n } } ( t ) \bigr ) < \operatorname* { m i n } _ { 0 \le t \le H } m \bigl ( s _ { \mathrm { n u l l } } ( t ) \bigr ) - 0 . 2 , \qquad H = 3 0 \operatorname* { m i n } .\tag{31}
$$

The relative-margin comparison and the projected-limit check are separate rejection conditions. Both use the same matched simulator assumption as the digital-twin benchmark, but ADMIT T4 is a constraint-oriented admissibility test, whereas the digital-twin validator explicitly searches for full recovery or a better mitigation. Their decision rules are therefore not interchangeable even when both use a 30-min model rollout.

A failed gate returns violation codes and suggested remedies for another proposal or a corrected action record. If the record is not admitted after the permitted attempts, no new command is dispatched. In protocol $\mathbf { v } 2 ,$ a record that is corrected without changing its commands can be checked again without treating the unchanged commands as a diferent intervention. Both ADMIT configurations use an LLM planning stage in the reported campaign. The plantspecific ADMIT cartridge includes the system and safetycase graphs, hazards linked to the eleven protective functions, five SOPs, and the twenty benchmark cases; it is generated from the plant definitions to maintain consistency with the benchmark implementation.

## 5.5. Rejection handling, fallback, and interpretation

The five configurations share the same operator-level dispatch interface but do not impose the same rejection or fallback policy. Executability-only dispatch applies the first executable proposal. The LLM judge and both ADMIT configurations withhold new supervisory commands after unsuccessful attempts. The digital twin can instead dispatch an eligible counterfactual mitigation that improves on doing nothing, even if none of the candidates meets its fullrestoration criterion. These distinctions are retained as part of the complete validation configurations being compared rather than obscured by a single acceptance-rate metric.

Rejection feedback is also protocol dependent. In protocol v1, validator failures are returned through the common reprompt path. Protocol v2 additionally provides explicit plain-language feedback identifying the current attempt and previously rejected plans, and records complete LLM exchanges for replay. The experiments therefore compare closed-loop outcomes of agent–validator pipelines with these defined interfaces; they do not isolate every individual contribution of feedback wording, planning, gate logic, and fallback strategy. Section 6.3 specifies the model and protocol assignments, and Section 7 reports plant-stop outcomes, economic performance, rejection mechanisms, token use, and wall-clock decision time.

It should be noted that none of the five configurations certifies industrial safety explicitly. The rule-based gates establish compliance with an encoded action policy, not correctness of the policy itself. The projected validators rely on a matched 30-min simulator rollout and do not test robustness to model error or unavailable future information. The LLM judge ofers neither deterministic admissibility checks nor physical-state prediction. These limitations are especially pertinent when comparing a method that rejects many proposals with one that permits a larger number of recoveries: the relevant outcome is essentially the resulting plant behaviour and not validation acceptance in isolation.

## 6. Benchmark design and evaluation protocol

The benchmark evaluates the closed-loop consequences of supervisory actions proposed by OperAItor, rather than the linguistic correctness of individual proposals. Each experiment uses the integrated reactor–distillation simulator, regulatory controllers, protective interlocks, and restricted supervisory interface described in Sections 2–4. The factors varied across agent experiments are the decision protocol, action-generation model, and complete validation configuration; any planner diferences associated with a configuration are treated as part of that configuration. Within a scenario, fault identities, onset and removal times, and the scoring horizon are fixed across configurations. This design allows diferences in plant-stop frequency and economic outcome to be interpreted against the same simulated fault conditions, subject to the protocol distinctions discussed below.

## 6.1. Fault scenarios

Twenty scenarios are constructed from the ten benchmark faults in Table 1. They comprise ten single faults (S01– S10), four two-fault cases involving diferent units or control functions (D01–D04), three coupled cases with overlapping efects (C01–C03), and three triple-fault cases (T01–T03). The triple-fault cases span nominally separate efects (T01), two interacting faults accompanied by a third fault (T02), and three interacting faults (T03). These groupings describe how the cases were designed; the resulting severity is established by the simulated process response rather than assumed from the number of faults alone.

The first fault is injected at $t _ { 0 } ~ = ~ 1 0$ min. Additional faults, when present, are introduced at the ofsets shown in Table $\begin{array} { r } { 6 ; } \end{array}$ a missing ofset indicates simultaneous onset with the first fault. To determine a common intervention window for each scenario, a probe simulation is first conducted with the faults left active and without agent intervention. The probe identifies the first time $t _ { \mathrm { t r i p } }$ at which a plant-stop interlock would activate, if one occurs. The fault-removal times are subsequently fixed by

$$
t _ { j } ^ { \mathrm { l i f t } } = \left\{ \begin{array} { l l } { \mathrm { m a x } \Big ( t _ { \mathrm { t r i p } } + 1 0 , t _ { j } ^ { \mathrm { o n } } + 1 0 \Big ) , } & { \mathrm { i f ~ t h e ~ p r o b e ~ t r i p s , } } \\ { t _ { j } ^ { \mathrm { o n } } + 3 0 , } & { \mathrm { o t h e r w i s e , } } \end{array} \right.\tag{32}
$$

where all times are in minutes. The scenario evaluation horizon is

$$
W = \mathrm { m a x } \biggl ( 1 2 0 , \mathrm { ~ m a x } t _ { j } ^ { \mathrm { l i f t } } + 6 0 \biggr ) \quad \mathrm { m i n } .\tag{33}
$$

The removal schedule is therefore determined once from the uncontrolled probe and is not recalculated for each agent or validator. For tripping probes, faults persist beyond the uncontrolled trip time, ensuring that a successful agent must influence the process before that event. An additional period after fault removal permits evaluation of subsequent recovery and inventory restoration. Every 5 min after the final fault is removed, a simulated shift operator requests the reset of any remaining latched interlocks, subject to their normal reset conditions. This common reset mechanism is distinct from the agent’s recovery decisions.

Table 6  
Twenty benchmark fault scenarios. Ofsets in parentheses are minutes after the first fault at � = 10 min. Trip times and interlocks refer to persistent-fault probe simulations without agent intervention, not to the finite-duration benchmark baseline. Parentheses around an interlock indicate a non-trip latch; a dash indicates that no plant-stop interlock was reached in the probe.
<table><tr><td>ID</td><td>Fault combination (onset offset)</td><td>Probe trip [min]</td><td>First trip interlock</td></tr><tr><td>S01</td><td>Reflux pump trip</td><td>16.9</td><td>COL_L_LL</td></tr><tr><td>S02</td><td>Bottoms pump trip</td><td>48.7</td><td>RB_L_HH</td></tr><tr><td>S03</td><td>Distillate pump trip</td><td>22.4</td><td>DR_L_HH</td></tr><tr><td>S04</td><td>Feed pump trip</td><td>81.1</td><td>DR_L_LL</td></tr><tr><td>S05</td><td>Reactor outlet capacity loss</td><td></td><td>(R_L_HH)</td></tr><tr><td>S06</td><td> ${ \sf H } _ { 2 }$  vent-capacity degradation</td><td>177.8</td><td>RB_L_HH</td></tr><tr><td>S07</td><td>Reactor heater stuck high</td><td></td><td>(R_T_HH)</td></tr><tr><td>S08</td><td>Condenser cooling loss</td><td>23.2</td><td>DR_L_LL</td></tr><tr><td>S09</td><td>Reflux-drum leak</td><td>46.2</td><td>DR_L_LL</td></tr><tr><td>S10</td><td>Reactor pressure-sensor low bias</td><td>197.5</td><td>RB_L_HH</td></tr><tr><td>D01</td><td>Pressure-sensor bias + bottoms pump trip (+3)</td><td>49.9</td><td>RB_L_HH</td></tr><tr><td>D02</td><td>Heater stuck high + distillate pump trip</td><td>21.9</td><td>DR_L_HH</td></tr><tr><td>D03</td><td>Vent degradation + reflux pump trip (+5)</td><td>21.9</td><td>COL_L_LL</td></tr><tr><td>D04</td><td>Pressure-sensor bias + drum leak (+2)</td><td>46.7</td><td>DR_L_LL</td></tr><tr><td>C01</td><td>Condenser cooling loss + reflux pump trip (+3)</td><td>20.4</td><td>COL_L_LL</td></tr><tr><td>C02</td><td>Vent degradation + pressure-sensor bias (+2)</td><td>176.5</td><td>RB_L_HH</td></tr><tr><td>C03</td><td>Drum leak + distillate pump trip (+4)</td><td>46.8</td><td>DR_L_LL</td></tr><tr><td>T01</td><td>Vent degradation + distillate pump trip (+2) + bottoms pump trip (+4)</td><td>24.9</td><td>DR_L_HH</td></tr><tr><td>T02</td><td>Condenser loss + pressure-sensor bias (+1) + reflux pump trip (+3)</td><td>20.4</td><td>COL_L_LL</td></tr><tr><td>T03</td><td>Condenser loss + reflux pump trip (+2) + drum leak (+4)</td><td>19.3</td><td>COL_L_LL</td></tr></table>

Eighteen probe scenarios reach a plant-stop interlock. The remaining two, S05 and S07, activate non-trip interlocks but do not produce a plant stop under the persistent-fault probe conditions. In the finite-duration benchmark without agents, 18 of the 20 scenarios also end in a plant stop. The probe and the no-agent baseline serve diferent purposes: the former establishes fault duration and evaluation horizons, whereas the latter supplies the operational and economic comparator under those fixed durations.

## 6.2. Operational and economic evaluation

The principal operational outcome is whether the plant remains available throughout the scenario horizon. A plant stop is recorded if a trip-latching protective interlock activates or an episode terminates because of damage, a modellimit violation, or numerical failure. These termination causes are retained separately in the recorded data, because a numerical solver failure is not equivalent to an actual protective intervention, even though both prevent completion of the simulated production episode. Non-trip interlock activation and successful subsequent recovery are not automatically counted as plant stops.

An economic score complements plant-stop counts by measuring the consequences of continued operation, production quality, utility consumption, and inventory changes. The benchmark uses assumed commodity prices, utility charges, and stop penalties for the hypothetical process; these are not measured industrial costs or estimates of commercial profitability. Table 7 summarises the parameters. All economic calculations use the simulated true plant state, rather than potentially biased BPCS measurements. At each 0.5- min observation, the instantaneous operating margin Π<sup>̇</sup> in $\$  h ^ { - 1 }$ is

$$
\begin{array} { r l } & { \dot { \Pi } = \mathbb { 1 } [ x _ { D } \ge 0 . 9 0 ] D x _ { D } \frac { M _ { \mathrm { a c } } } { 1 0 0 0 } p _ { \mathrm { a c } } } \\ & { \qquad + 0 . 9 5 B ( 1 - x _ { B } ) \frac { M _ { \mathrm { I P A } } } { 1 0 0 0 } p _ { \mathrm { I P A } } } \\ & { \qquad + F _ { \mathrm { v e n t } } M _ { H _ { 2 } } p _ { H _ { 2 } } - F \frac { M _ { \mathrm { I P A } } } { 1 0 0 0 } p _ { \mathrm { I P A } } } \\ & { \qquad - 3 . 6 p _ { \mathrm { s t } } \left( Q _ { \mathrm { r e b } } + Q _ { j } \right) , } \end{array}\tag{34}
$$

where $D , B , F ,$ and $F _ { \mathrm { v e n t } }$ are molar flow rates in kmol $\mathrm { h } ^ { - 1 }$ $x _ { D }$ and $x _ { B }$ are acetone mole fractions; and $M _ { \mathrm { a c } } , M _ { \mathrm { I P A } }$ , and $M _ { H _ { 2 } }$ are molecular weights in kg kmol<sup>−1</sup>. The prices $p _ { \mathrm { a c } } ,$ $p _ { \mathrm { I P A } } , p _ { H _ { \gamma } }$ , and $p _ { \mathrm { s t } }$ are defined in Table 7. Both $Q _ { \mathrm { r e b } }$ and $Q _ { j }$ are expressed in MW; the factor 3.6 converts MW to $\mathrm { G J } \mathrm { h } ^ { - 1 }$ The factor 0.95 values recovered IPA relative to fresh IPA, while distillate below the specified acetone mole-fraction threshold earns no acetone revenue.

Let $t _ { \mathrm { e n d } }$ denote the end of the simulated episode, equal to � if operation continues or to the earlier termination time if the plant stops. The scenario-level score is

Nominal operating scale and illustrative economic assumptions for the hypothetical benchmark. Monetary coeficients are assumed values in USD
<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Nominal IPA feed rate</td><td>100 kmol/h (6.01  $\mathbf { t } / \mathsf { h } )$ </td></tr><tr><td>Nominal acetone product rate</td><td>Approx. 4.0 t/h</td></tr><tr><td>Acetone selling price,  $p _ { \mathrm { a c } }$  Fresh IPA price,</td><td>$1,300/t</td></tr><tr><td> $p _ { \mathrm { I P A } }$  Recovered IPA credit</td><td>$1,000/t 95% of fresh IPA price</td></tr><tr><td>Hydrogen fuel credit,</td><td></td></tr><tr><td> $p _ { H _ { 2 } }$  Heating utility,</td><td>$1/kg</td></tr><tr><td> $p _ { \mathrm { s t } }$ </td><td>$8/GJ</td></tr><tr><td>On-spec acetone mole fraction</td><td> $x _ { D } \ge 0 . 9 0$ </td></tr><tr><td>Shutdown charge per plant stop</td><td>$5,000</td></tr><tr><td>Restart charge per plant stop</td><td>$15,000</td></tr></table>

$$
\begin{array} { r } { \Pi _ { W } = \displaystyle \sum _ { k : t _ { k } < t _ { \mathrm { e n d } } } \dot { \Pi } ( t _ { k } ) \Delta t _ { \mathrm { o b s } } ^ { \mathrm { h } } + V _ { \mathrm { i n v } } ( t _ { \mathrm { e n d } } ) - V _ { \mathrm { i n v } } ( t _ { 1 } ) } \\ { - \mathbb { 1 } [ \mathrm { p l a n t \ s t o p \ b y } W ] C _ { \mathrm { s t o p } } , \qquad ( \mathrm { } } \end{array}\tag{35}
$$

where $\Delta t _ { \mathrm { o b s } } ^ { \mathrm { h } } = 0 . 5 / 6 0$ h is the observation interval and $V _ { \mathrm { i n v } }$ is the value of liquid holdup in the reactor, reboiler, and reflux drum under the benchmark’s IPA-equivalent inventory convention. The stop penalty is $C _ { \mathrm { s t o p } } = \$ 2000$ , comprising \$5 000 for shutdown and \$15 000 for restart. After a stop, no further operating margin is credited within the fixed scenario horizon. The inventory term adjusts the score for the value of material retained in or withdrawn from the modelled process holdups.

For a configuration �, the aggregate saving relative to the no-agent baseline is

$$
S _ { c } = \sum _ { s = 1 } ^ { 2 0 } \left( \Pi _ { W , s } ^ { ( c ) } - \Pi _ { W , s } ^ { ( \mathrm { n o ~ a g e n t s } ) } \right) .\tag{36}
$$

The horizon � is scenario-specific but identical across the configurations compared within that scenario. Aggregated over the twenty scenarios, the nominal fault-free reference has a score of \$34 202, whereas the no-agent baseline has a score of −\$364 878. Their diference, \$399 080, is the nominal-reference performance gap; it is a comparison point rather than a mathematical upper bound on what a supervisory strategy could earn. The 18 no-agent stops contribute \$360 000 in fixed shutdown and restart charges, approximately 90% of this gap. The remaining diference is attributable to the operating-margin and inventory terms. These totals aggregate 20 separate simulated scenarios with horizons defined by Eq. (33); they are not annual production losses or projected industrial savings. Monetary comparisons should therefore be interpreted alongside the plant-stop outcomes and the assumed trip charges.

## 6.3. Decision protocols and agent configurations

Two decision protocols are included. Protocol v1 initiates a decision after a PCA-confirmed fault and imposes a 5-min cooldown between decisions. As long as the fault remains detected, additional proposals can be generated at subsequent eligible decision points. Protocol v1 uses GPT-5 nano for the action agent and an LLM-based diagnosis stage.

Protocol $\nu 2$ uses an event-driven decision schedule. A decision is opened once for a fault episode when an alarm becomes active or the squared prediction error remains above its control limit; a further decision is initiated only by a new priority-1 alarm or a newly failed pump. Within a decision, the action agent receives explicit rejection feedback identifying the attempt number (out of four) and the proposals already rejected. If an ADMIT action record is corrected without changing the commands, the corrected record is rechecked rather than treated as a new intervention. Protocol $\mathbf { v } 2$ also retains complete LLM request–response records for replay. Unlike v1, v2 uses the deterministic heuristic diagnoser for every model. This common diagnoser reduces one source of variation when comparing the actiongeneration models within v2, but the two protocols should not be interpreted as difering only in decision frequency.

Both protocols are evaluated with the five validation configurations defined in Section 5 and Table 5: executabilityonly dispatch, an LLM judge, digital-twin validation, AD-MIT T0–T3, and ADMIT T0–T4. Protocol v1 is evaluated with GPT-5 nano (OpenAI, 2025). Protocol v2 is evaluated with GPT-5 nano and three open-weight models served locally using Ollama: Qwen3 8B, Mistral 7B, and Llama 3.2 3B (Yang et al., 2025; Jiang et al., 2023; Grattafiori et al., 2024; Ollama, 2024). The action-generation settings described in Section 4.5 are applied consistently within the respective configurations. Validation rejection, retry, and fallback behaviour are specified in Section 5, not redefined here. The ADMIT configurations use LLM planning; if other arms use a diferent planner, the resulting comparison is between complete agent–validator configurations rather than an isolated validator ablation.

The resulting experiment comprises $2 0 \times 5 ~ = ~ 1 0 0$ agent runs for protocol v1 and $2 0 \times 5 \times 4 ~ = ~ 4 0 0$ for protocol $\mathbf { v } 2 ,$ giving 500 agent runs in total, in addition to the persistent-fault probes, no-agent baseline, and nominal reference simulations. Thus GPT-5 nano appears in both protocol campaigns, whereas the three open-weight models appear only in $\mathbf { v } 2 .$ Every model–validator–scenario combination is executed once. The principal paired comparisons are between validation configurations within the same model andprotocol; comparisons between models are made within protocol $\mathbf { v } 2 .$ Cross-protocol diferences are descriptive because triggering, diagnosis, and feedback behaviour change together. With only one execution per configuration and scenario, the reported results describe this fixed fault campaign and do not by themselves establish statistical reliability over diferent random seeds, LLM samples, or disturbance realisations.

In all reported benchmark runs, the simulated process clock is paused while the agent and validator operate, as explained in Section 4.1. Inference and validation times are therefore recorded as computational costs rather than inserted as delays into the simulated plant dynamics. The benchmark records plant stops, economic scores, issued and rejected actions, rejection reasons, inference tokens, and decision latency, so that validation efectiveness can be assessed jointly in terms of fault recovery, operational performance, and computational burden.

## 7. Results

The results are organised around three questions: whether the simulated plant and monitoring system provide a credible setting for supervisory fault recovery; how alternative agent– validation configurations afect closed-loop operation; and what operational and computational trade-ofs are associated with validation. As established in Section 6.3, comparisons are made primarily within a fixed model and decision protocol. Protocols v1 and v2 use diferent triggering, diagnosis, and rejection-feedback procedures, so their outcomes are not treated as a controlled ablation of any one of these changes.

## 7.1. Plant-model verification

The process simulator is accompanied by automated physical-consistency and software tests, incorporated into a 178-test suite executed in continuous integration. The audit checks total material accounting during transients involving feed changes, liquid leaks, and reflux interruptions; component and hydrogen balances at the nominal operating point; and consistency of the nominal condenser and reboiler duties. A step-size refinement test using one quarter of the nominal integration step changes the tested temperatures by approximately 0.001 K. These checks support the numerical consistency of the benchmark under the examined operating conditions, rather than establishing fidelity to an industrial plant.

The implemented faults also produce the intended causal responses. For example, a reflux-pump trip reduces reflux, increases top-tray temperature, decreases distillate purity, and can ultimately trigger a low-holdup interlock. Reduced catalyst activity lowers conversion, while condenser cooling loss limits condensation capacity and can deteriorate the column pressure and reflux-drum inventories. Two modelling limitations are relevant to interpretation. First, the reactor uses the irreversible surrogate chemistry described in Section 2. Second, the lumped condenser-pressure model does not include a separate vapour-composition inventory; the resulting transient acetone-accounting discrepancy is reported as $C _ { P } P _ { c } \Delta y _ { 1 }$ , below 1% of acetone holdup in the audited cases, and reverses as the top composition returns. Accordingly, subsequent safety and recovery findings refer to this simulator and its defined fault mechanisms.

## 7.2. Fault detection and diagnosis

Figure 5 illustrates the monitoring response in four diagnostic tests, each with a fault injected at 15 min; these tests are separate from the benchmark scenarios whose first fault occurs at 10 min. Neither PCA statistic crossed its detection threshold before fault onset. The reflux-pump trip, condenser cooling loss, and reactor-pressure sensor bias were confirmed within 2.0, 1.5, and 1.0 min, respectively. The progressively degrading hydrogen pressure-control valve (PCV) was confirmed after 8.5 min, once the reduced vent capacity had caused a suficiently large pressure deviation.

The leading variable contributions were consistent with the afected equipment. Reflux-pump failure increased the contributions of drum holdup and the top-temperature controller output; condenser cooling loss afected the column and overhead pressure measurements; PCV degradation was reflected in reactor pressure and vent flow; and pressuresensor bias was dominated by the associated pressure measurement. For the sensor fault, the initial signature was principally an increase in squared prediction error (�), indicating a disrupted relationship among measurements, whereas the larger ${ \dot { T } } ^ { 2 }$ response developed later as the biased pressure reading afected regulatory operation, at approximately 47 min. These examples show that the monitor supplies useful fault evidence, but they do not establish perfect detection or isolation over the entire fault library.

At the first PCA confirmation in each of the 20 benchmark scenarios, the heuristic diagnoser ranked at least one true injected fault first in 14 scenarios. The corresponding numbers for GPT-5 nano and GPT-5 mini diagnostic evaluations were 15 and 18. All three approaches included at least one true fault among their top three hypotheses in all 20 scenarios. Most top-ranked errors arose from faults with similar early symptoms, including vent restrictions and pressuremeasurement bias, or from upstream disturbances whose initial column-side efects resembled condenser problems. GPT-5 mini is included here as a diagnosis-only comparison; it is not one of the four action-generation models in the principal control campaign. The downstream experiments consequently test action generation and validation under plausible but imperfect diagnostic evidence.

## 7.3. Closed-loop fault-recovery performance

The principal campaign comprises 500 episodes with agents: 100 with GPT-5 nano under protocol v1, and 400 under protocol v2 across four action-generation models. Each model is evaluated with five validation configurations and 20 scenarios. The no-agent baseline stops in 18 scenarios. Table 8 reports the number of plant stops, the aggregate saving relative to the no-agent baseline, and the mean decision wall time. Figure 6 provides the corresponding graphical comparison. The nominal-reference gap of \$399,080 (Section 6.2) is used as a reference level in the savings plot; it is not an upper bound on achievable performance.

Repeated supervisory decisions (protocol v1). Under protocol v1, GPT-5 nano could make a new decision every 5 min while a fault remained detected. The executabilityonly agent stopped the plant in nine scenarios, whereas the LLM-judge configuration stopped it in eleven. The digital twin and ADMIT T0–T3 configurations each stopped the plant in five scenarios; ADMIT T0–T4 stopped it in six. Relative to no agents, the corresponding savings were \$179k (no validation), \$135k (judge), \$260k (digital twin), \$259k (T0–T3), and \$237k (T0–T4). Thus, in this fixed repeated-decision campaign, simulation-based validation and admissibility checking improved the operational outcome over unvalidated dispatch. The LLM judge did not: it permitted some proposals associated with subsequent trips and rejected other proposals that could have supported recovery. These observations describe complete pipeline outcomes, not independently measured false-acceptance or false-rejection probabilities.

![](images/722736cdc901ee1d571a1c3e2b1a1a8837fe9b1ad7b43b32a58aefb38ca8802e.jpg)

![](images/2794adbf46b5a7b3d43ce6938d50b6228a2594986dcfb5eef60d52aba712f68d.jpg)

H<sub>2</sub> PCV capacity ramp to 20 %  
![](images/e88be9db975779aca69f05207d37face0a51935ea9ade2859a7b8e7e98a14922.jpg)

![](images/67b7f67e78e7246cd20997f41f5033d91013697cedc1c755e25799d958e4fc7c.jpg)  
Figure 5: PCA monitoring during four representative diagnostic tests with faults introduced at 15 min (vertical grey line). Hotelling’s �<sup>2</sup> and squared prediction error (�) are normalised by their respective 99th-percentile thresholds (dashed lines). Dotted red markers indicate confirmation after three consecutive threshold violations. Traces terminate when the simulated plant stops.

![](images/babc2b2c591e0f984072014f231776ed36ffcb1aa288443904d399fb109905b2.jpg)

![](images/11f84ade232887a938b0f070ba88463af71e9435064a4b3663c1f6da241c6895.jpg)  
Figure 6: Campaign outcomes across action-generation models and validation configurations: (a) plant stops out of 20 scenarios and (b) aggregate savings relative to the no-agent baseline. GPT-5 nano is shown separately for protocols v1 and v2 because the protocols difer in diagnosis, triggering, and feedback. The savings reference line represents the \$399,080 nominal-reference performance gap, not a theoretical maximum.

Event-driven decisions with larger models (protocol v2). GPT-5 nano, Qwen3 8B, and Mistral 7B produced relatively similar outcomes across most validation configurations, with five to eight plant stops and \$209–265k in savings. The best observed v2 result was GPT-5 nano with ADMIT T0–T3, which stopped the plant in five scenarios and saved \$265k. For Qwen3 8B, however, T0–T3 resulted in eight stops and \$209k compared with seven stops and \$235k without validation; Mistral 7B retained seven stops in every configuration. Consequently, the results do not establish that one validator dominates across capable models. Diferences in the intervention selected and the time consumed by validation can matter even when aggregate stop counts are similar.

Results over 20 scenarios for each model, protocol, and validation configuration. Savings are relative to the no-agent baseline (k\$), and times are mean wall-clock seconds per decision. Each row represents 20 episodes, with one execution per scenario.
<table><tr><td>Model and protocol Validation Plant stops Saving [k$]</td></tr><tr><td>Time [s/decision] Protocol v1: repeated decisions; LLM diagnosis 21</td></tr><tr><td>GPT-5 nano None 9 179 LLM judge 11 135</td></tr><tr><td>167</td></tr><tr><td>Digital twin 5 260 ADMIT T0-T4 6 237</td></tr><tr><td>143 ADMIT T0-T3 5 259 42</td></tr><tr><td></td></tr><tr><td>Protocol v2: event-driven decisions; heuristic diagnosis</td></tr><tr><td>GPT-5 nano None 7 233 8 LLM judge 6 245 20</td></tr><tr><td>Digital twin 7 232 118</td></tr><tr><td>ADMIT T0-T4 7 237 99 ADMIT T0-T3 5 265 28</td></tr><tr><td></td></tr><tr><td></td></tr><tr><td>Qwen3 8B None 7 235 LLM judge 7 235</td></tr><tr><td>96 Digital twin 7 234 336</td></tr><tr><td></td></tr><tr><td>ADMIT T0-T4 7 238 398 ADMIT T0-T3 8 209 219</td></tr><tr><td></td></tr><tr><td>Mistral 7B None 7 235 76</td></tr><tr><td>LLM judge 7 235 66</td></tr><tr><td>Digital twin 7 234 405</td></tr><tr><td>ADMIT T0-T4 7 238 361</td></tr><tr><td>ADMIT T0-T3 7 235 196</td></tr><tr><td>Llama 3.2 3B None 13 109 138</td></tr><tr><td></td></tr><tr><td>LLM judge 15 70 164</td></tr><tr><td>Digital twin 12 131 109</td></tr><tr><td></td></tr><tr><td>ADMIT T0-T4 9 197 191 90</td></tr><tr><td>ADMIT T0-T3 7 235</td></tr></table>

Sensitivity to the action-generation model. The clearest variation among the v2 models occurred with Llama 3.2 3B. It stopped the plant in 13 scenarios without validation, 15 with the LLM judge, and 12 with the digital twin. ADMIT T0–T4 reduced the number to nine, while T0–T3 reduced it to seven and increased savings to \$235k, comparable to the observed totals for the 8B and 7B models. In the four single-pump-trip cases (S01–S04), the T0–T3 configuration started a standby pump within 2.5 min of the fault, whereas the unvalidated 3B configuration issued no intervention in those cases. One plausible mechanism is that the structured ADMIT action record and associated rejection feedback guide the smaller model towards an actionable command. However, the experiments do not isolate this prompting efect from the acceptance gates, planner, or reprompting procedure. The result should therefore be interpreted as an improvement of the complete ADMIT-enabled configuration, not proof that deterministic checking alone improves small-model reasoning.

## 7.4. Scenario-level recovery and failure modes

The aggregate results conceal marked diferences in scenario dificulty. Figure 7 shows the outcome of each model–validator combination alongside the no-agent baseline. Individual duty-pump trips (S01–S04) are generally recoverable when an agent identifies the failed train and uses the KG-linked START\_SPARE command. The same mechanism can mitigate pump trips in multi-fault cases such as D01– D03 and T01, although simultaneous faults create additional constraints. Slowly developing reactor faults, including hydrogen PCV degradation (S06), pressure-sensor bias (S10), and their combination (C02), require an intervention that maintains acceptable operating conditions across the coupled reactor and column after the initial disturbance.

Figure 8 provides illustrative protocol-v1 trajectories with GPT-5 nano. In S06, repeated throughput reductions and pressure-setpoint changes by the unvalidated and LLMjudge agents destabilised the coupled operating balance. Their episodes ended in reflux-drum low-low trips at approximately 237 and 246 min, respectively, after the injected fault had been removed. By contrast, a digital-twinselected mitigation maintained operation over the evaluated horizon. This case illustrates that avoiding the immediate fault signature is insuficient if the intervention depletes an inventory elsewhere in the process. In S09, the unvalidated agent lowered the reflux-drum level setpoint while the drum was leaking, thereby increasing net withdrawal and accelerating the trip to approximately 23 min, compared with approximately 46 min without an agent. The digitaltwin configuration instead selected a throughput reduction and higher top-temperature setpoint, reducing reflux demand and maintaining drum inventory until the scheduled leak removal. These examples show why an intervention’s efect must be assessed across connected units rather than solely against the measurement that triggered the decision.

![](images/2e65ff6ac9b636315ed9c5bd7c7f78eee087124933de3c27c66fff77b1f6f7a1.jpg)  
Figure 7: Plant-stop outcome for each scenario and agent–validation configuration. The no-agent reference is shown separately. A cross denotes an episode ending in a plant stop; an unmarked cell denotes operation without a recorded stop over the benchmark horizon.

Seven scenarios were particularly dificult across the campaign: the four involving condenser cooling loss (S08, C01, T02, and T03) and three strongly afected by refluxdrum leakage (S09, D04, and C03). Most configurations stopped in these cases, although GPT-5 nano with ADMIT T0–T3 under protocol v2 avoided a stop in S09. These observations identify a demanding subset of the benchmark; they do not, on their own, establish that every possible supervisory strategy is incapable of recovering all seven scenarios. The more restricted single-action feasibility question is examined next.

## 7.5. Ofline admissibility and action-authority limits

The ofline ADMIT evaluation uses one case per benchmark scenario at the time of PCA confirmation, together with an associated hazard definition and a catalogue of candidate supervisory interventions. For cases whose persistent-fault probe trips the plant, the observed trip interlock provides the principal hazard. An oracle policy selects the admissible candidate with the greatest projected safety margin from standby-pump starts, throughput changes, blowdown, emergency cooling, and top- or bottom-temperature setpoint changes. If no candidate satisfies the safety envelope, it selects escalation rather than an unsupported plant intervention.

For the four condenser-cooling-loss cases (S08, C01, T02, and T03), no evaluated single permitted action maintained the projected trajectory within the prescribed envelope. The oracle therefore escalated. This outcome is consistent with the observed loss of condenser duty and deterioration of pressure or drum-inventory conditions, however it does not rule out multi-action recovery policies, additional equipment, or a diferent protection design. It also distinguishes a desirable controlled shutdown or escalation from successful continued production.

In the ofline harness, the oracle records were admitted in all 20 cases, with a reported mean aggregate score of 0.998; an escalation can be admissible even when it does not recover production. A test agent generating predefined inadmissible proposals was blocked in 12 of 20 cases: four at T0 for attempting to bypass a protective function, six at T1 for requesting blowdown or emergency cooling without the required evidence, and two at T3 for missing prescribed procedural steps on setpoint changes. These checks illustrate the gate mechanism’s ability to reject specified authority, evidence, and procedure violations in the ofline test. They are not a statistical estimate of its safety performance on arbitrary proposals.

![](images/54dc5435a46a3083138c5197b3941c1606f992c33a87f53014f4b1c438e76385.jpg)

![](images/a03ef4b38166dcd1a57534acd6fcc637488613df1bf878538d7deed5780b67c4.jpg)

![](images/83612bb1538162bec1d811a634c395934b1a5f53d638bab555244c24bb0585df.jpg)  
Figure 8: Representative closed-loop trajectories for GPT-5 nano under protocol v1: (a,b) reactor pressure and reflux-drum holdup during S06 (hydrogen-vent degradation), and (c) reflux-drum holdup during S09 (drum leak). Shaded regions indicate active faults, dotted red lines denote protective limits, and crosses indicate plant stops. The S06 persistent-fault probe stops at approximately 178 min.

## 7.6. Rejection behaviour and computational cost

Figure 9 groups proposal rejections by cause under protocol v2. The digital-twin validator commonly rejected proposed full-recovery actions when its 30-min projection failed to restore the active targets or predicted a plant stop. ADMIT rejections were more often associated with incomplete or inconsistent action records, including missing or incorrectly ordered procedure steps (T3) and citations to evidence absent from the trusted ledger (T1). In protocol v1 with GPT-5 nano, only 3% of digital-twin decisions accepted the first proposal under the default full-recovery criterion, indicating that revisions and the fallback mitigation mechanism were important parts of that configuration. The diferent rejection distributions reflect diferent acceptance questions: predicted process consequence for the twin, and documented admissibility conditions for the gates. Rejection counts alone therefore cannot rank validation efectiveness.

The performance benefits of validation must be considered alongside its computational requirements. Figure 10 relates savings to token use and wall-clock decision latency for protocol v2. With GPT-5 nano, the 20-scenario totals were approximately 0.9 million tokens for no validation, 2.7 million for the LLM judge, 2.8 million for the digital twin, 8.0 million for ADMIT T0–T4, and 4.8 million for ADMIT T0–T3. Mean decision times were approximately 7.7, 20, 118, 99, and 28 s, respectively. The reported API charge for the GPT-5 nano no-validation configuration was approximately \$0.08, and the full 100-run GPT-5 nano v2 campaign cost \$1.58 in metered API usage at the time of the experiments. These charges exclude local simulation, hardware, electricity, and development costs.

For the GPT-5 nano digital-twin configuration, forward simulation accounted for approximately 117 min of the campaign’s 147 min total decision time. Local Qwen3, Mistral, and Llama inference incurred no metered model-API charge, but individual LLM calls took approximately 22–115 s on the tested GPU, excluding the unreported economic cost of local compute. ADMIT T0–T3 achieved the highest v2 saving for GPT-5 nano and Llama 3.2 3B at lower mean latency than their digital-twin or T0–T4 configurations. That operating point was not universal: for Qwen3 8B, T0–T3 reduced savings and increased the stop count, while the Mistral configurations had nearly identical economic outcomes despite substantially diferent latencies. Thus, the preferred validation configuration depends on the action-generation model and the operational objective; the available data do not identify a uniformly dominant validator.

## 7.7. Additional prompt-context ablation and limitations

A supplementary prompt-context ablation examined six selected protocol-v1 failure scenarios for GPT-5 nano, using repeated trials and also testing GPT-5 mini. Adding loop information, an operating manual, or further processdynamics descriptions did not consistently improve the outcomes in this selected subset. The reported success counts ranged from zero to four non-stopping runs out of twelve in the KG-supported conditions, compared with zero out of eight in the conditions without the KG; these denominators difer and should not be interpreted as a matched statistical comparison. In several trials, both models continued to propose increased reflux while the drum was draining, despite receiving relevant loop and inventory information. This result suggests that more context or a larger model does not necessarily resolve incorrect intervention selection in these examples, motivating the explicit evaluation of action consequences.

OperAItor  
![](images/4159659069b4ad53acf99d1668181997cd1d29911f054629fb2741ae4928c6ac.jpg)  
Figure 9: Rejection causes under protocol v2, grouped by model and validation configuration. Each configuration is evaluated on 20 scenarios; the total number of proposals can difer because rejected actions may be revised and resubmitted.

![](images/25754f969d9c816478adb84a12fd8ae8cef0c9331a8a6d1dca75f498ae4bef34.jpg)  
LLM tokens per 20-scenario campaign [millions]

![](images/fdcb453e658c124b6f1a168e42c9f74e31da5ff4b4c2024cf4006079dfd986c1.jpg)  
mean wall time per decision [s]  
Figure 10: Protocol-v2 operating-point comparison: aggregate saving versus (a) total LLM tokens per 20-scenario configuration and (b) mean wall-clock time per decision. Markers distinguish action-generation models and validation configurations. The points are descriptive campaign outcomes rather than estimates of expected performance across random seeds.

Several limitations constrain the conclusions. First, only one run was conducted for each model–validation–scenario combination in the main campaign, so the reported diferences are descriptive and do not establish their variability across repeated LLM samples, initialisations, or fault realisations. Second, validator comparisons can include changes in planning and action-record requirements; where the planner difers, they compare complete agent–validator configurations rather than isolating the validator alone. Third, protocol v1 and v2 difer simultaneously in diagnosis, triggering, and rejection feedback, preventing a causal attribution of their aggregate diferences to any one change. Fourth, the digital twin is a matched clone of the simulated plant and is therefore an oracle relative to realistic model mismatch;

all decisions are also made with the simulated process clock paused. Finally, the process chemistry, protective thresholds, and economic penalties are benchmark assumptions rather than validated industrial design values.

Within these boundaries, the results support three conclusions. Validation can materially change the outcome of repeated supervisory interventions, particularly when an agent proposes actions that worsen inventories elsewhere in the process. Structured admissibility requirements can improve the performance of a smaller action-generation model in the tested configurations, but admission is not a guarantee for beneficial process consequences. Most importantly, the number of actions accepted or rejected is not itself an operational objective: a useful validation system must be evaluated in terms of avoided stops, preserved recovery opportunities, economic performance, and decision cost. These results motivate further tests under uncertain digital twins, asynchronous execution, and repeated stochastic trials rather than claims of safety certification from the present campaign.

## 8. Discussion

Validation outcomes. The efect of validation depends on the action-generation model and the complete decision pipeline. Under protocol v1, repeated intervention by GPT-5 nano led to fewer plant stops with the digital-twin and AD-MIT configurations than with executability-only dispatch. Under protocol v2, the results were less consistent: several configurations gave similar outcomes for GPT-5 nano, Qwen3 8B, and Mistral 7B, while ADMIT T0–T3 increased the stop count for Qwen3 8B. The most pronounced improvement was observed for Llama 3.2 3B with ADMIT T0– T3. These comparisons also reflect configuration-specific planning, action records, rejection feedback, and fallback behaviour; they do not isolate the efect of a validation algorithm. The LLM judge worsened outcomes for GPT-5 nano under protocol v1 and Llama 3.2 3B under protocol v2, but its efect was small or absent in several other configurations.

Digital twin and admissibility gates. The digital twin projects candidate actions over 30 min using a matched copy of the simulated plant. Its results depend on having an accurate current state and dynamics; plant–model mismatch and state-estimation errors were not evaluated. In contrast, ADMIT T0–T3 checks authority, evidence, hazards, and prescribed operating procedures without requiring a dynamic model. ADMIT T4 supplements these checks with a consequence projection. The methods therefore test diferent properties of an intervention: compliance with encoded operating requirements and predicted process behaviour. Neither establishes safety beyond the assumptions of its validation mechanism.

Operational and recovery limits. Seven scenarios involving condenser cooling loss or reflux-drum leakage were particularly dificult across the campaign, although one drum-leak scenario was recovered by a tested configuration.

The ofline analysis found that none of the evaluated single supervisory actions could keep the four condenser-coolingloss scenarios within the projected safety envelope. This does not establish infeasibility for sequences or combinations of actions, additional equipment, or alternative plant designs. The recovery limits should therefore be interpreted relative to the tested action catalogue and its projected horizon.

Study limitations. The reactor–distillation process is a hypothetical operating benchmark with physically based mixture properties and modified liquid-phase reaction kinetics, rather than a validated acetone manufacturing plant. The stage-equilibrium and hydraulic models also omit several industrial phenomena, including tray flooding, weeping, and detailed pump curves. The economic assumptions, notably the fixed \$20,000 plant-stop charge, are illustrative and afect the magnitude of the monetary results. Each main-campaign configuration was executed once per scenario; therefore, the reported diferences do not provide statistical estimates of run-to-run variability. The process clock was paused during agent and validator execution. Finally, the reported results use simulator version 1.0; version 1.1 changes the reboiler steam-valve range and certain setpoint limits while retaining the nominal duty, so its benchmark scores may difer.

Benchmark extensibility. The released framework allows alternative plants, faults, knowledge graphs, prompts, validators, and decision pipelines to be evaluated using the same measurement and scoring interface. Human-operator and agent results can also be compared on a common scenario set, subject to matching information access and timing assumptions.

## 9. Conclusions

OperAItor provides an open benchmark and framework for evaluating LLM-generated supervisory actions in a dynamic reactor–distillation process equipped with regulatory control, alarms, and simulated independent protective functions. Five action-validation configurations were compared using four language models and 20 single- and multiple-fault scenarios. The no-agent baseline ended in 18 plant stops. Across GPT-5 nano, Qwen3 8B, and Mistral 7B, the evaluated validated configurations resulted in 5–8 stops and recovered \$209,000–265,000 of the \$399,080 nominal-reference performance gap under illustrative economic assumptions. The largest efect of admissibility checking was observed for Llama 3.2 3B, for which ADMIT T0–T3 reduced stops from 13 to 7. No validator dominated all model and protocol combinations, and LLM-based review worsened outcomes in some configurations. Four condenser-cooling-loss scenarios had no feasible recovery among the evaluated single supervisory actions in the ofline admissibility analysis. Future work will evaluate model mismatch and state estimation for consequence projection, repeated stochastic runs, agent-decision latency during plant evolution, and additional plants and supervisory action policies.

## Code and data availability

The simulator, agents, knowledge graph, ADMIT cartridge, benchmark harness and all campaign records (every run, LLM request and ADMIT trace) are available under the Apache-2.0 licence at https://github.com/AISL-at-Imper ial- College- London/operAItor (AISL, Imperial College London, 2026a). The project website (AISL, Imperial College London, 2026b), https://aisl-at-imperial-college -london.github.io/operAItor/, hosts the live console, the interactive campaign dashboard and installers for Windows, macOS and Linux.

## CRediT authorship contribution statement

Javal Vyas: Conceptualization, Methodology, Software, Investigation, Formal analysis, Visualization, Writing – original draft. Mehmet Mercangöz: Supervision, Funding acquisition, Writing – review and editing.

## Declaration of competing interest

The authors declare that they have no known competing financial interests or personal relationships that could have appeared to influence the work reported in this paper.

## Declaration of generative AI and AI-assisted technologies in the manuscript preparation process

During the preparation of this work, the authors used OpenAI’s ChatGPT to assist with language refinement, manuscript organization, and drafting support. After using this tool, the authors reviewed and edited the content as needed and take full responsibility for the content of the published article.

## Acknowledgements

The authors acknowledge support from Imperial College London.

## References

AISL, Imperial College London, 2026a. OperAItor: an open-source framework for AI operator agents in process plants, version 1.1.0. https: //github.com/AISL-at-Imperial-College-London/operAItor. Apache-2.0 licence. Accessed 20 October 2026.

AISL, Imperial College London, 2026b. OperAItor project website, live console and benchmark dashboard. https://aisl-at-imperial-college -london.github.io/operAItor/. Accessed 20 October 2026.

ANSI/ISA, 2009. ANSI/ISA-5.1-2009: Instrumentation symbols and identification. International Society of Automation, Research Triangle Park, NC.

ANSI/ISA, 2016. ANSI/ISA-18.2-2016: Management of alarm systems for the process industries. International Society of Automation, Research Triangle Park, NC.

Åström, K.J., Hägglund, T., 2006. Advanced PID Control. ISA – The Instrumentation, Systems, and Automation Society, Research Triangle Park, NC.

Blanke, M., Kinnaert, M., Lunze, J., Staroswiecki, M., 2006. Diagnosis and Fault-Tolerant Control. 2nd ed., Springer, Berlin.

Brown, T.B., Mann, B., Ryder, N., et al., 2020. Language models are fewshot learners, in: Advances in Neural Information Processing Systems, pp. 1877–1901.

Chiang, L.H., Russell, E.L., Braatz, R.D., 2001. Fault Detection and Diagnosis in Industrial Systems. Springer, London.

Decardi-Nelson, B., Alshehri, A.S., Ajagekar, A., You, F., 2024. Generative AI and process systems engineering: The next frontier. Computers & Chemical Engineering 187, 108723. doi:10.1016/j.compchemeng.2024.1 08723.

Downs, J.J., Vogel, E.F., 1993. A plant-wide industrial process control problem. Computers & Chemical Engineering 17, 245–255. doi:10.101 6/0098-1354(93)80018-I.

Grattafiori, A., Dubey, A., Jauhri, A., et al., 2024. The Llama 3 herd of models. arXiv preprint arXiv:2407.21783 .

Gutta, S., Naicker, V., Misra, Y., Mercangöz, M., Vyas, J., 2026. ADMIT Bench: An admissibility standard for AI agents that act on physical plants, version 0.1.0. https://github.com/Refiant-Inc/admit-bench.

Hogan, A., Blomqvist, E., Cochez, M., et al., 2021. Knowledge graphs. ACM Computing Surveys 54, 1–37. doi:10.1145/3447772.

Hotelling, H., 1947. Multivariate quality control illustrated by the air testing of sample bombsights, in: Eisenhart, C., Hastay, M.W., Wallis, W.A. (Eds.), Techniques of Statistical Analysis. McGraw-Hill, New York, pp. 111–184.

IEC, 2016. IEC 61511-1:2016: Functional safety – safety instrumented systems for the process industry sector – part 1: Framework, definitions, system, hardware and application programming requirements. International Electrotechnical Commission, Geneva.

Jackson, J.E., Mudholkar, G.S., 1979. Control procedures for residuals associated with principal component analysis. Technometrics 21, 341– 349. doi:10.1080/00401706.1979.10489779.

Janowicz, K., Haller, A., Cox, S.J.D., Le Phuoc, D., Lefrançois, M., 2019. SOSA: A lightweight ontology for sensors, observations, samples, and actuators. Journal of Web Semantics 56, 1–10. doi:10.1016/j.websem.2 018.06.003.

Jiang, A.Q., Sablayrolles, A., Mensch, A., et al., 2023. Mistral 7B. arXiv preprint arXiv:2310.06825 .

Kritzinger, W., Karner, M., Traar, G., Henjes, J., Sihn, W., 2018. Digital twin in manufacturing: A categorical literature review and classification. IFAC-PapersOnLine 51, 1016–1022. doi:10.1016/j.ifacol.2018.08.474.

LangChain, Inc., 2024. LangGraph: building stateful, multi-actor applications with LLMs. https://github.com/langchain-ai/langgraph.

Lewis, P., Perez, E., Piktus, A., et al., 2020. Retrieval-augmented generation for knowledge-intensive NLP tasks, in: Advances in Neural Information Processing Systems, pp. 9459–9474.

Luyben, W.L., 1990. Process Modeling, Simulation and Control for Chemical Engineers. 2nd ed., McGraw-Hill, New York.

Luyben, W.L., 2011. Design and control of the acetone process via dehydrogenation of 2-propanol. Industrial & Engineering Chemistry Research 50, 1206–1218. doi:10.1021/ie101591p.

MacGregor, J.F., Kourti, T., 1995. Statistical process control of multivariate processes. Control Engineering Practice 3, 403–414. doi:10.1016/0967 -0661(95)00014-L.

Mhaskar, P., Liu, J., Christofides, P.D., 2013. Fault-Tolerant Process Control: Methods and Applications. Springer, London.

Misra, Y., Vyas, J., Gutta, S., Mercangöz, M., 2026. ADMITBench: A safety-governed reference framework for evaluating the admissibility of industrial LLM advisories. arXiv preprint arXiv:2608.03866 doi:10.485 50/arXiv.2608.03866.

Nomikos, P., MacGregor, J.F., 1995. Multivariate SPC charts for monitoring batch processes. Technometrics 37, 41–59. doi:10.1080/00401706.1 995.10485888.

Ollama, 2024. Ollama: run large language models locally. https://ollama .com.

OpenAI, 2025. GPT-5 system card. https://openai.com/index/gpt-5-sys tem-card/.

Pan, S., Luo, L., Wang, Y., Chen, C., Wang, J., Wu, X., 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Transactions on Knowledge and Data Engineering 36, 3580–3599. doi:10.1109/TKDE

.2024.3352100.

Poling, B.E., Prausnitz, J.M., O’Connell, J.P., 2001. The Properties of Gases and Liquids. 5th ed., McGraw-Hill, New York.

Powell, M.J.D., 1970. A hybrid method for nonlinear equations, in: Rabinowitz, P. (Ed.), Numerical Methods for Nonlinear Algebraic Equations. Gordon and Breach, London, pp. 87–114.

Qin, S.J., 2003. Statistical process monitoring: basics and beyond. Journal of Chemometrics 17, 480–502. doi:10.1002/cem.800.

Reinartz, C.C., Kulahci, M., Ravn, O., 2021. An extended tennessee eastman simulation dataset for fault-detection and decision support systems. Computers & Chemical Engineering 149, 107281. doi:10.1 016/j.compchemeng.2021.107281.

Schall, D., 2026. Safe integration of large language models into industrial process control: a multi-agent architecture with p&id-grounded validation. Autonomous Intelligent Systems 6, 14. doi:10.1007/s43684-026-0 0136-1.

Schweidtmann, A.M., 2024. Generative artificial intelligence in chemical engineering. Nature Chemical Engineering 1, 193. doi:10.1038/s44286 -024-00041-5.

Skogestad, S., 1997. Dynamics and control of distillation columns: A tutorial introduction. Chemical Engineering Research and Design 75, 539–562. doi:10.1205/026387697524092.

Turton, R., Bailie, R.C., Whiting, W.B., Shaeiwitz, J.A., Bhattacharyya, D., 2012. Analysis, Synthesis, and Design of Chemical Processes. 4th ed., Prentice Hall, Upper Saddle River, NJ.

VDI/VDE, 2015. VDI/VDE 3682: Formalised process descriptions. Beuth Verlag, Berlin.

Venkatasubramanian, V., Rengaswamy, R., Kavuri, S.N., Yin, K., 2003a. A review of process fault detection and diagnosis: Part III: Process history based methods. Computers & Chemical Engineering 27, 327– 346. doi:10.1016/S0098-1354(02)00162-X.

Venkatasubramanian, V., Rengaswamy, R., Yin, K., Kavuri, S.N., 2003b. A review of process fault detection and diagnosis: Part I: Quantitative model-based methods. Computers & Chemical Engineering 27, 293– 311. doi:10.1016/S0098-1354(02)00160-6.

Virtanen, P., Gommers, R., Oliphant, T.E., et al., 2020. SciPy 1.0: fundamental algorithms for scientific computing in Python. Nature Methods 17, 261–272. doi:10.1038/s41592-019-0686-2.

Vyas, J., Gill, M.S., Markaj, A., Gehlhof, F., Mercangöz, M., 2026. From detection to action: Using llm agents for fault-tolerant control. URL: https://arxiv.org/abs/2606.28011, arXiv:2606.28011.

Vyas, J., Mercangöz, M., 2024. Autonomous industrial control using an agentic framework with large language models. arXiv preprint arXiv:2411.05904 .

Wabersich, K.P., Zeilinger, M.N., 2021. A predictive safety filter for learning-based control of constrained nonlinear dynamical systems. Automatica 129, 109597. doi:10.1016/j.automatica.2021.109597.

Yang, A., et al., 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388 .

Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y., 2023. ReAct: Synergizing reasoning and acting in language models, in: International Conference on Learning Representations (ICLR).

Zhang, Y., Jiang, J., 2008. Bibliographical review on reconfigurable faulttolerant control systems. Annual Reviews in Control 32, 229–252. doi:10.1016/j.arcontrol.2008.03.008.

Zheng, L., Chiang, W.L., Sheng, Y., et al., 2023. Judging LLM-as-a-judge with MT-Bench and Chatbot Arena, in: Advances in Neural Information Processing Systems (Datasets and Benchmarks Track).