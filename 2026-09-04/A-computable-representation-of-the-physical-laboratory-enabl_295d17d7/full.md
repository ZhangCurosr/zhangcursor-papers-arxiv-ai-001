# A computable representation of the physical laboratory enables verifiable workflows

Xiaobo Li<sup>1,2,#,\*</sup>, Luyao Ge<sup>1,#</sup>, Xiaohui Li<sup>2</sup>, Lulu Guo<sup>1</sup>, Ming Mao<sup>3</sup>, Jiwang Zheng<sup>3</sup>, Wenting Guan<sup>3</sup>, Xin Yang<sup>3</sup>, Yi Luo<sup>1,2</sup>, Jun Jiang<sup>1,2,\*</sup>, Linjiang Chen<sup>1,2\*</sup>

<sup>1</sup> State Key Laboratory of Precision and Intelligent Chemistry, Hefei National Research Center for Physical Sciences at the Microscale, School of Chemistry and Materials Science, University of Science and Technology of China, Hefei, China

<sup>2</sup> Center for Scientific Intelligence Innovation, Hefei, China

<sup>3</sup> Huawei Technologies Co., Ltd.

\# These authors contributed equally.

\*Correspondence: xiaoboli@ustc.edu.cn; jiangj1@ustc.edu.cn; linjiangchen@ustc.edu.cn.

## Abstract

Making science computable requires representations of both scientific knowledge and the physical world in which scientific claims are tested. A computable representation of the physical laboratory is established through typed research objects, capability-bound operations and a compositional workflow algebra. It provides the physical-world counterpart to machine-readable knowledge, expressing workflows as programs over evolving laboratory states with explicit dependencies, decisions, iteration and concurrency. The representation was implemented in a modular agentic robotic laboratory by binding formal operations to executable Function Skills. For diverse scientific intents, capability-relative workflows were generated, while stateful simulation propagated object transformations and verified operation preconditions and laboratory constraints before dispatch. The proposed representation and its engineering framework jointly establish a general computational interface between agent reasoning and capability-bound physical transformations, providing a foundation for end-to-end autonomous scientific discovery.

## A computable laboratory state–action space

Agentic self-driving laboratories combine artificial-intelligence reasoning with modular systems for synthesis, processing, reaction, characterization and analysis.<sup>1–3</sup> Their central computational problem is not merely to select instruments. Scientific intent is expressed semantically and can vary without bound, whereas execution must conserve samples, respect container and device limits, maintain location, order dependencies and condition later actions on measurements. A plausible plan can therefore be chemically meaningful yet physically inadmissible. Chemical programming languages provide essential operation vocabularies and control structures,<sup>4–6</sup> and laboratory orchestrators coordinate instruments, resources and data.<sup>7–9</sup> Agentic operation therefore requires a computable representation of the physical laboratory, in which research objects, laboratory capabilities and control logic are encoded within a common statetransition semantics.

A chemical procedure links physical operations, evolving samples and measurement-dependent decisions (Fig. 1a). The representation abstracts this procedure as a state–action system for the actionable laboratory—not an idealized chemical universe (Fig. 1b). At workflow step t, St denotes the state of a typed research object, comprising its persistent identity and structured sample, container, evidence and provenance fields. An operation is a typed, workstation-bound function fi(θi) whose contract declares accepted states, parameters, constraints, effects, returns and failures. When its mandatory predicates are satisfied, the operation maps St to $s _ { t + 1 }$ and records evidence; otherwise, it returns a diagnostic without changing the affected simulated state. The computational unit is thus a physically bounded state transition, rather than a free-text instruction or an isolated device call (Supplementary Notes 1–2, Supplementary Tables S1–S3, Supplementary Fig. S1 and Supplementary Data 1).

## a Typical chemical workflow

Operations are bounded by their physical capabilities  
![](images/1947667b0f61c3d06650960f7821a11f3afee31b3bf162dafc3307d1912c84d5.jpg)

![](images/ecba197712cfc878419bea099a834b06935ec12d15d5db6e003b18a85c7cb2fe.jpg)  
Sample objects are bounded with physical context and defined by their states.

![](images/fbeb12e29200264376f73983402bc3775e3104a9a52eec1e67b79c6478f7ff07.jpg)

c Control-flow composition operators  
![](images/771eeebde9a73f0de7ec64f82039e8d11d0906035cece25e528d53c74f3bcc7d.jpg)

![](images/d8e477c1b76e92ae0364d5a95bccf2d940a397ce0b79669ae1f98f678792f7af.jpg)  
Fig. 1 | A computable representation of the physical laboratory. a, A chemical procedure comprises transformations of samples and containers, with measurements and control decisions determining subsequent actions. b, The same procedure is represented as a state–action system: St denotes a typed research-object state, and each capability-bound operation fi(θi) declares accepted inputs, constraints, effects, returns and failures before mapping ${ \sf S } _ { \mathrm { t } } \mathrm { t } 0 { \sf S } _ { \mathrm { t } + 1 . } \dot { \sf c } ,$ Five operators encode sequence, bounded iteration, evidence-dependent branching, parallel fork–join and protected execution. d, Recursive nesting of operations and operators forms a workflow algebra over evolving laboratory state. The panels therefore progress from physical procedure to computable state, explicit control logic and compositional program. XRD, X-ray diffraction; UV–vis, ultraviolet–visible spectroscopy.

## Workflows as compositional programs

Primitive transitions become workflows through five operators (Fig. 1c). Sequence propagates state between operations. A bounded loop makes iteration and non-convergence explicit. A conditional selects a branch from current evidence. Parallel fork–join projects objects into independent branches and reconciles compatible updates. A protected interval keeps a resource-sensitive block contiguous. The workflow algebra nests these operators (Fig. 1d), allowing a scientific procedure to be expressed as a program over evolving laboratory state. Type compatibility is checked at sequential boundaries; loops require finite bounds; conditional guards must resolve to a permitted branch; parallel branches must not introduce conflicting writes; and protected blocks preserve declared ordering. Workflow abstraction therefore converts experimental logic into structures that can be composed and computationally inspected (Supplementary Notes 3–4, Supplementary Table S4, Supplementary Fig. S2 and Supplementary Data 2).

This construction separates scientific plausibility from operational admissibility. The formal layer tests the latter against an explicit capability inventory and cannot invent unavailable operations. Its generative space is the closure of registered primitives under the five operators. Generality is capability-relative: transfer requires each laboratory to register local operations, constraints and adapters.

## Engineering executable semantics

To test whether the representation can be operationalized, it was implemented in a modular agentic self-driving laboratory (Fig. 2).<sup>10</sup> Forty-nine workstation specifications—45 physical and 4 virtual controlflow modules—were converted from structured device descriptions into agent-readable Function Skills. The frozen snapshot exposes 63 functions covering liquid and solid handling, mixing, heating, separation, reaction, electrochemical testing, chromatography, spectroscopy, diffraction, imaging and control flow. Each Function Skill preserves the corresponding operation contract and uses workstation-qualified names where repeated actions could otherwise be ambiguous. This compilation turns the installed laboratory into a machine-readable capability space, while keeping the formal representation independent of any single device command set.

Scientific intent is translated by an agent into a Python workflow containing only registered calls. Before dispatch, the abstract syntax tree is admitted against allow-listed imports, functions and entry points, then normalized conservatively. The resulting program enters a shared stateful simulator. Each call evaluates all encoded mandatory and advisory conditions before applying effects. A mandatory failure blocks the update; a successful call commits the new object state and appends a step record. Diagnostics localize the function, step, observed condition, expected condition and suggested remediation. This validate-thencommit cycle allows a workflow fragment to be revised without treating the entire protocol as unstructured text. It verifies operational consistency with encoded contracts; it does not prove chemical truth, experimental safety beyond those rules or device reliability (Supplementary Notes 5–7, Supplementary Tables S5–S8, Supplementary Figs. S3–S6, Supplementary Data 1 and 4, and Additional Files 1–2).

![](images/b5060e428235996386efe85b93aa9580213e87622140be883b3e78e9e36a8764.jpg)  
Fig. 2 | Engineering and checking computable laboratory workflows. a, Structured specifications register the physical and virtual capabilities of the modular agentic self-driving laboratory. b, These records are compiled into agent-readable Function Skills, shared runtime types and simulator functions while preserving operation contracts. c, Scientific intent is compiled into a Python workflow containing registered calls, admitted through abstract-syntax-tree analysis and evaluated by stateful simulation before review and downstream serialization. Successful calls commit object-state changes; failed MUST predicates leave SimState unchanged and return a localized diagnostic set Δ<sub>k</sub> for regeneration. Here, S<sub>t</sub> denotes a research-object state, and SimState<sub>t</sub> denotes the bounded executable projection used by the simulator. O0–O4 denote optimizer levels and are unrelated to these state symbols.

## Capability-relative generation from scientific intent

Compositional scope was examined with four prompts that specified scientific objectives and control logic but not complete workstation sequences (Extended Data Fig. 1). The first compared five Feintroduction routes for NiFe-based catalysts under matched alkaline oxygen-evolution testing and produced a 27-node sequence across 11 workstations. The second coupled layered double hydroxide synthesis and electrode preparation in a protected interval to HMF oxidation, then forked solid-electrode and liquid-product objects into three analytical branches; it contained 22 physical nodes, 31 simulated steps and 11 workstations. The third combined furfural hydrogenation with a bounded gas-chromatography method-optimization loop $( N _ { m a x } = 6 )$ , yielding 20 nodes and 23 simulated steps. The fourth coupled ZnO preparation to parallel infrared and ultraviolet–visible characterization, with a conditional dilution loop (Nmax = 3), yielding 27 nodes, 30 simulated steps and 15 workstations.

All four accepted workflows used only registered capabilities and passed structural analysis and state simulation with no mandatory errors or advisory warnings. The cases span sequence, conditional branching, bounded iteration, fork–join parallelism and protected execution. They are not four hard-coded protocol templates; each is a different composition of shared operations selected through required object transitions and evidence dependencies (Supplementary Notes 8–11, Supplementary Tables S9–S11, Supplementary Figs. S7–S10, Supplementary Data 3 and Additional File 3).

## A representation layer for agentic laboratories

Recent efforts have added a foundational laboratory schema, literature-to-protocol agents and simulation-guided refinement to the established programming and orchestration layers.<sup>11–13</sup> Broader developments encompass modular self-driving-laboratory architectures, autonomous materials synthesis and distributed closed-loop discovery,<sup>14–17</sup> while knowledge graphs, natural-language robotic planning, orchestration middleware and sample-centred automata address complementary parts of laboratory autonomy.<sup>18–21</sup> The contribution here is the unification of three levels that are commonly separated: a typed representation of physical research objects, an algebra for composing capability-bound transitions, and an executable realization that checks those transitions against evolving state. Unlike protocol-centric representations, the object state being transformed remains an explicit operand throughout generation, composition and checking. Agent reasoning is thereby connected to physical action through an explicit computational object rather than through prose alone.

The present checks are intentionally bounded. Their validity depends on the completeness of workstation contracts and the fidelity of simulated state. Scheduler-level resource isolation, live telemetry, device faults, empirical repeatability and chemical outcomes require execution evidence or additional models. These are extension interfaces rather than hidden assumptions: constraints, resources, observations and adapters can be added without changing the intent layer. Local failures can consequently be assigned to an object, operation, guard or join rather than to the workflow as a whole. By making laboratory state, permitted transformations and control structure jointly computable, the framework establishes a basis for generating increasingly complex experiments while keeping their physical commitments verifiable before execution.

## Methods

## Research-object and operation representation

At workflow step t, St denotes the state of a typed research object. Each object contains a persistent identifier and typed subrecords for sample identity and composition, phase, amount, container type and state, location, evidence and provenance. Evidence may contain scalar observations or references to generated data. A workstation contract declares the projection of these fields required for an operation. The current simulator implements the subset needed for workstation checking: container identifier and type, closure state, sample state, volume, mass, location, source-bottle use, step logs, errors and warnings.

Each physical operation is represented by a Function Skill containing the workstation binding, operation name, input-state requirements, cautions, typed parameters, constraints, state changes, return schema and diagnostic categories. For operation fi with parameters $\theta _ { I } ,$ a transition from St to $\boldsymbol { S } _ { t + 1 }$ is admitted only when all mandatory predicates $P _ { i } ( S _ { t } ,$ θi) hold. For operations involving multiple research objects, the function acts on the corresponding tuple of object states. Advisory predicates record warnings but do not block the transition. Mandatory failure returns an unsuccessful result and leaves the affected simulated state unchanged.

## Workflow composition

Sequence was defined by functional composition, with each intermediate output required to lie in the next operation’s input domain. A loop repeated its body until the continuation guard became false or a declared $N _ { m a x }$ was reached; an active guard at the bound returned a non-convergence diagnostic. Conditional composition required a unique permitted branch. Parallel fork–join projected the objects required by each branch and admitted the join only when branch write sets did not conflict and shared updates agreed. A protected interval was constrained to occupy consecutive scheduler positions. The five operators could be nested recursively.

## Capability compilation

Workstation capabilities were supplied as structured records or curated Skill documents. Required fields included workstation identity, available operations, accepted input and output states, parameters, bounds, container requirements, state effects and returns. A development-stage quality gate assessed documentation completeness; the archived configuration admitted records at the required-field score of 60. Accepted records were converted into workstation-qualified interfaces, shared data classes and simulator implementations. The frozen inventory contains 28 synthesis, 5 reaction-and-testing, 12 characterization and 4 virtual specifications, declaring 63 functions.

## Structural analysis and stateful simulation

Generated workflows were Python modules with one generate\_recipe() entry point. Each module was parsed into an abstract syntax tree. Admission permitted only registered runtime imports and allow-listed function calls, and rejected wildcard imports, dynamic calls, unsupported top-level statements, forbidden names and additional function definitions. Literal arithmetic and Boolean expressions were folded, inactive constant branches were removed and redundant pass statements were eliminated. Conditional-output pruning was available as a domain-aware transformation but was disabled in the frozen configuration.

For simulation, the shared state was reset, artificial delays were disabled and registered workstation functions were injected into the workflow namespace. Each operation collected all applicable checks before applying any effect. Checks covered container existence and count, container and sample state, cumulative and per-call volume, mass, parameter ranges, file constraints and declared workstation dependencies. A diagnostic stored the function, step, code, category, expected and observed values, parameter snapshot, severity and suggested remediation. Mandatory diagnostics blocked passage. Successful operations committed state changes and appended a structured step record.

## Case construction and evaluation

The four prompts requested Fe coordination-environment modulation and matched alkaline oxygenevolution testing; layered double hydroxide synthesis, electrochemical HMF oxidation and parallel analysis; furfural hydrogenation with iterative gas-chromatography optimization; and ZnO synthesis with parallel spectroscopy and adaptive dilution. Complete workstation sequences were not supplied. The agent received the controller Skill and registered Function Skills. The workflow-generation agent was powered by GPT-5.6 Sol (OpenAI). Model invocations used in this study were performed between 15 and 25 August 2026. Accepted workflows were serialized into canonical JSON, and node counts, workstation counts, simulated steps and diagnostic totals were calculated from the frozen workflow and validation records. No workflow was physically executed for this study.

## Statistical analysis

The study evaluates a representation, its software realization and four workflow constructions. No wetlaboratory outcome comparison or inferential statistical analysis was performed.

## Data availability

The typed capability registry, formal definitions, four prompts, canonical workflow representations, validation summaries and evidence manifest are provided as Supplementary Data 1–4. Complete workflow and validation records for the four cases shown in Extended Data Fig. 1 are supplied as Additional File 3. No new wet-laboratory measurement data were generated for the workflows reported here.

## Code availability

Source code for Function-Skill conversion, workflow analysis, simulation and serialization is supplied as Additional File 1, and the frozen Function-Skill library as Additional File 2. Additional File 4 contains the internal release-page record supplied for provenance only; it is not presented as a public repository, persistent public access route or DOI.

## References

1. Burger, B. et al. Nature 583, 237–241 (2020). https://doi.org/10.1038/s41586-020-2442-2

2. Canty, R. B. & Abolhasani, M. Nat. Rev. Chem. 10, 523–537 (2026). https://doi.org/10.1038/s41570-026-00847-2

3. Boiko, D. A., MacKnight, R., Kline, B. & Gomes, G. Nature 624, 570–578 (2023). https://doi.org/10.1038/s41586-023-06792-0

4. Steiner, S. et al. Science 363, eaav2211 (2019). https://doi.org/10.1126/science.aav2211

5. Rauschen, R., Guy, M., Hein, J. E. & Cronin, L. Nat. Synth. 3, 488–496 (2024). https://doi.org/10.1038/s44160-023-00473-6

6. Šiaučiulis, M. et al. Nat. Commun. 15, 10261 (2024). https://doi.org/10.1038/s41467-024-54238-6

7. Roch, L. M. et al. PLoS ONE 15, e0229862 (2020). https://doi.org/10.1371/journal.pone.0229862

8. Sim, M. et al. Matter 7, 2959–2977 (2024). https://doi.org/10.1016/j.matt.2024.04.022

9. Fei, Y. et al. Digit. Discov. 3, 2275–2288 (2024). https://doi.org/10.1039/D4DD00129J

10. Li, X. et al. ACS Nano (2026). https://doi.org/10.1021/acsnano.6c15286

11. Gottstein, W., Blanc, A., Feng, S., Sutherland, B. R. & Pablo-García, S. Device 101202 (2026). https://doi.org/10.1016/j.device.2026.101202

12. Pagel, S., Jirasek, M. & Cronin, L. Commun. Chem. 9, 191 (2026). https://doi.org/10.1038/s42004-026-01993-w

13. Hsu, B. et al. Digit. Discov. 5, 2613–2628 (2026). https://doi.org/10.1039/D6DD00004E

14. Abolhasani, M. & Kumacheva, E. Nat. Synth. 2, 483–492 (2023). https://doi.org/10.1038/s44160-022-00231-0

15. Tom, G. et al. Chem. Rev. 124, 9633–9732 (2024). https://doi.org/10.1021/acs.chemrev.4c00055

16. Szymanski, N. J. et al. Nature 624, 86–91 (2023). https://doi.org/10.1038/s41586-023-06734-w

17. Strieth-Kalthoff, F. et al. Science 384, eadk9227 (2024). https://doi.org/10.1126/science.adk9227

18. Bai, J. et al. Nat. Commun. 15, 462 (2024). https://doi.org/10.1038/s41467-023-44599-9

19. Darvish, K. et al. Matter 8, 101897 (2025). https://doi.org/10.1016/j.matt.2024.10.015

20. Angelopoulos, A. et al. In 2025 IEEE International Conference on Robotics and Automation 15900–15906 (IEEE, 2025). https://doi.org/10.1109/ICRA55743.2025.11128578

21. Tahara-Arai, Y. et al. Digit. Discov. 5, 1411–1424 (2026). https://doi.org/10.1039/D5DD00409H

## Acknowledgements

The workflow generation and stateful simulations were performed on the Robotic AI-Scientist Platform of the Chinese Academy of Sciences (CAS).

## Author contributions

X.L. and L.G. contributed equally to this work. X.L., L.C. and J.J. conceived and supervised the project. X.L., L.G., X.H.L., L.G., M.M., J.Z., W.G. and X.Y. developed the formal representation and engineering framework, constructed and analysed the workflows, and curated the supporting materials. All authors discussed the results, revised the manuscript and approved the final version.

## Competing interests

The authors declare no competing interests.

## Additional information

Supplementary information accompanies this paper. Reproducibility settings and the software and evidence manifest are provided in Supplementary Note 12, Supplementary Table S12 and Supplementary Data 4; the corresponding source-code archive, Function-Skill library, workflow archive and internal release-page record are supplied as Additional Files 1–4.

## Prompt 1:

Design an experiment to investigate how the existing form and coordination environment of Fe modulate the alkaline oxygen evolution reaction (OER) performance of NiFe-based catalysts. NiFe-based catalytic materials with different Fe introduction methods and distinct Fe coordination environments are required to be fabricated.

![](images/ee70485ce3ee03979506f164a33874361221613b4147240cb6ac55f463120444.jpg)

## Prompt 3:

Design an automated catalyst synthesis and closed-loop analytical optimization experiment for the electrocatalytic hydrogenation of furfural to furfuryl alcohol. After catalyst preparation, a closed-loop algorithm is required to iteratively optimize the gas-chromatographic separation of furfural, furfuryl alcohol and major liquid byproducts until predefined peak-resolution criteria are satisfied. The optimized method will then be applied to quantify the reaction mixture and evaluate furfural conversion, furfuryl alcohol yield and product selectivity. Need to use a loop node.  
![](images/ce7726c2c7ad5a974a5c5388e2d27a6ed9a92c626d33dc4614fab844864fdfac.jpg)

## Prompt 2:

Design an integrated synthesis, electrochemical testing and multimodal characterization experiment for LDH catalysts toward alkaline 5-hydroxymethylfurfural (HMF) oxidation. The catalyst preparation and electrode-fabrication steps must follow a time-locked sequential workflow. After electrochemical testing, the solid electrode and liquid reaction products are required to undergo parallel contact-angle measurement, XRD analysis and liquid-phase product analysis, followed by integrated evaluation of the results.

![](images/614a509e06b976cc5bc974442f562e4b988249415b43b80c17f6e91508176e11.jpg)

## Prompt 4:

Design an automated synthesis and adaptive spectroscopic characterization experiment for ZnO nanoparticles. Following synthesis, centrifugal purification, drying and redispersion, the samples are required to undergo parallel FTIR and UV-vis spectroscopic characterization. In the UV-vis branch, a conditional dilution-andremeasurement loop must be implemented until the measured absorbance falls within the predefined quantitative range.  
![](images/04d5715a4c7eb314603b61106f84c7be58bcc584ff0f19477173814355bc7df6.jpg)

Extended Data Fig. 1 | Capability-relative workflows generated from four scientific intents. Each panel pairs a prompt with its generated workflow: a, five routes to modulate the Fe coordination environment of NiFe-based catalysts followed by matched alkaline oxygen-evolution testing; b, layered double hydroxide synthesis and electrode preparation within a protected interval, electrochemical HMF oxidation, and fork–join analysis of solid-electrode and liquid-product objects; c, furfural hydrogenation coupled to a bounded gas-chromatography method-optimization loop; and d, ZnO synthesis followed by parallel infrared and ultraviolet–visible spectroscopy and a bounded evidence-triggered dilution loop. The graphs expose ordering, branching, iteration, synchronization and protected execution composed from registered capabilities. They report pre-dispatch workflow generation and verification by structural analysis and stateful simulation, not wet-laboratory outcomes. HMF, 5-hydroxymethylfurfural.

# A computable representation of the physical laboratory enables verifiable workflows

Xiaobo Li<sup>1,2,#,\*</sup>, Luyao Ge<sup>1,#</sup>, Xiaohui Li<sup>2</sup>, Lulu Guo<sup>1</sup>, Ming Mao<sup>3</sup>, Jiwang Zheng<sup>3</sup>, Wenting Guan<sup>3</sup>, Xin Yang<sup>3</sup>, Yi Luo<sup>1,2</sup>, Jun Jiang<sup>1,2,\*</sup>, Linjiang Chen<sup>1,2\*</sup>

<sup>1</sup> State Key Laboratory of Precision and Intelligent Chemistry, Hefei National Research Center for Physical Sciences at the Microscale, School of Chemistry and Materials Science, University of Science and Technology of China, Hefei, China

<sup>2</sup> Center for Scientific Intelligence Innovation, Hefei, China

<sup>3</sup> Huawei Technologies Co., Ltd.

\# These authors contributed equally.

\*Correspondence: xiaoboli@ustc.edu.cn; jiangj1@ustc.edu.cn; linjiangchen@ustc.edu.cn.

This Supporting Information contains Supplementary Notes 1–12, Supplementary Tables S1–S12 and Supplementary Figs. S1–S10. Supplementary Data files and authorsupplied Additional Files are described in the SI Guide at the end of this document. References cited here follow the numbering of the main Article.

## Supplementary Notes

Supplementary Note 1. Typed research-object representation

The workflow representation begins with a laboratory state, denoted by $S _ { t }$ at workflow step t, comprising one or more typed research objects. Fig. 1a,b defines the object schema as

S={id,{sample},{container},{evidence},…}.

The representation binds scientific material to the physical context required for manipulation. The sample component records identity, composition, phase, mass and volume. The container component records type, geometry, capacity, lid or holder state and other handling-relevant properties. The evidence component records observations and data products, including temperature, pH, colour and characterization outputs. The ellipsis permits laboratory-specific fields without altering the required object–container– evidence structure.

Fig. 1a,b illustrates the representation through three research-object states. $\boldsymbol { \mathsf { S } } _ { 1 }$ contains a suspension of Chemicals A and B in a lidded 50 ml tube with no evidence recorded. $\$ 5$

contains a liquid–solid object in an open 50 ml tube with pH evidence. $\$ 14$ contains a dry solid derived from $S _ { 1 3 , }$ , placed in a characterization holder with XRD and UV–vis evidence. These examples show that provenance is retained while the material, container and evidence states evolve.

The executable simulator implements a bounded projection of this conceptual object. The supplied SimState<sub>t</sub> records containers, errors, warnings, step logs, source-bottle usage and a step counter. Each Container records container\_id, container\_type, container\_status, sample\_status, sample\_volume\_ml, sample\_mass\_g and location. This implementation does not replace the richer definition of St. It provides the subset required to test the currently encoded workstation contracts. The conceptual and implemented fields are aligned in Supplementary Table S2.

Research objects are identified independently from workflow operations. An operation can therefore accept one or more objects, test their current states, update approved fields and append evidence without discarding object identity. This separation is required for provenance, branching and convergence. It also permits two branches to operate on distinct projections of the same parent object and later join their evidence, provided that the parallel consistency conditions in Supplementary Note 3 are satisfied.

Cross-references: Fig. 1a,b; Supplementary Tables S1 and S2; Supplementary Fig. S1; Supplementary Data 1.

## Supplementary Note 2. Primitive operation contracts

A primitive experimental operation is represented by the state transition

$$
f _ { i } ( \pmb { \theta } _ { i } ) \colon S _ { t } \to S _ { t + 1 } , P _ { i } ( S _ { t } , \pmb { \theta } _ { i } ) = 1\tag{1}
$$

where $f _ { j }$ is operation $i , \theta _ { i }$ contains its arguments and $P _ { j }$ is its precondition predicate. The operation is admissible only when the predicate evaluates to one. Its output must remain in the domain accepted by the next operation.

The Function-Skill interface realizes this primitive contract as a workstation-bound function declaration. The supplied documents define the interface through workstation identity, preconditions, cautions, typed arguments, constraints, state changes, structured returns and declared failure categories. Each physical function uses a namespace call, for example Liquid\_Handling\_Station\_1ml\_V2.open\_caps(...). The namespace encodes workstation ownership and removes ambiguity when different devices expose similarly named operations.

Function signatures use explicit types and unit-bearing parameter names. Enumerations are constrained through Literal aliases, nested inputs are represented by named data classes, and shared physical limits are centralized in \_common.py. A Function-Skill document contains only the callable interface and contract. The corresponding implementation in simulator.py performs the actual parameter, precondition and statetransition checks. The interface and implementation are required to remain synchronized.

Constraints are separated into MUST and SHOULD rules. A failed MUST rule is added to SimState.errors and blocks acceptance. A failed SHOULD rule is added to SimState.warnings and does not block acceptance. State changes are committed only when every MUST rule for the operation passes. Data-producing operations return typed result objects containing file or measurement references in addition to the common StepResult fields.

The current common runtime enumerates 11 container types: sample vial, 50-ml heatresistant bottle, screw vial, chromatography vial, carbon-paper holder, 10-ml pressure reaction tube, test holder, 96-well plastic plate (96-well quartz plate) and 54-position GC plate (54-position LC plate). Laboratory-specific limits remain attached to the corresponding Function-Skill and shared constant, rather than being generalized beyond the documented capability.

Cross-references: Fig. 1b; Supplementary Table S3; Supplementary Data 1.

## Supplementary Note 3. Mathematical semantics of workflow composition

The five control-flow operators in Fig. 1c organize primitive transitions without changing their individual contracts. Equations (2)–(14) below reproduce the supplied mathematical-definition document. Symbols are defined in Supplementary Table S1.

## Sequential composition

$$
F _ { s e q } = f _ { n } ( \pmb { \theta } _ { n } ) \circ \cdots \circ f _ { 2 } ( \pmb { \theta } _ { 2 } ) \circ f _ { 1 } ( \pmb { \theta } _ { 1 } )\tag{2}
$$

Execution proceeds from right to left. Every intermediate state must satisfy the next precondition and fall within the next domain:

$$
\begin{array} { r l } { \forall j \in \{ 2 , \dots , n \} \colon } & { { } f _ { j - 1 } ( \pmb { \theta } _ { j - 1 } ) ( D _ { j - 1 } ) \subseteq D _ { j } \cap P _ { j } ^ { - 1 } ( \{ \pmb { 1 } \} ) } \end{array}\tag{3}
$$

Here, n is the number of operations, and j indexes operation $f _ { j }$ . The set $P _ { j } ^ { - 1 } ( \{ 1 \} ) = \{ S \in$ $D _ { j } | P _ { j } ( S , \pmb { \theta } _ { j } ) = \mathbf { 1 } \}$ contains the states satisfying the precondition of $f _ { j }$ . Complete notation definitions are provided in Supplementary Table S1.

## Bounded loop

For a loop body spanning operations $f _ { a }$ through $\scriptstyle f _ { b }$ , define

$$
B = f _ { b } ( \theta _ { b } ) \circ \cdots \circ f _ { a } ( \theta _ { a } ) , \qquad F _ { l o o p } = B ^ { \tau }\tag{4}
$$

Here, $\pmb { B } ^ { \pmb { \tau } }$ denotes the A-fold composition of the complete loop body @, with $B ^ { 0 } = I ,$ where B is the the identity mapping. Let ${ \pmb x } ^ { ( 0 ) } = { \pmb x } _ { i n }$ denote the loop-entry state and $\pmb { x } ^ { ( j + 1 ) } =$ $B ( { \pmb x } ^ { ( j ) } )$ . Thus, $\pmb { x } ^ { ( j ) }$ is the state after 3 completed executions of the loop body. Define ${ \pmb g } _ { j } : =$ $g ( x ^ { ( j ) } )$ , where $\pmb { g } _ { j } = \pmb { 1 }$ indicates that another iteration is required and $\mathbf { \nabla } _ { \mathbf { \eta } } \mathbf { g } _ { j } = \mathbf { 0 }$ indicates normal exit. The iteration count is

$$
\pmb { \tau } = \pmb { m } i \pmb { n } ( \{ j \in \{ 0 , \dots , N _ { m a x } \} \mid g _ { j } = 0 \} \cup \{ N _ { m a x } \} )\tag{5}
$$

Reaching the upper bound while the continuation predicate remains true produces a nonempty loop diagnostic:

$$
( \pmb { \tau } = N _ { m a x } ) \wedge ( \pmb { g } _ { \tau } = \pmb { 1 } ) \Longrightarrow d _ { l o o p } \neq \emptyset\tag{6}
$$

## Conditional branch

For m branches with mutually exclusive and collectively exhaustive guards, let $\pmb { q } _ { i } ( \pmb { x } )$ denote the binary guard associated with branch J.

$$
q _ { i } \colon D _ { c o n d }  \{ { \bf 0 } , { \bf 1 } \} ( i = 1 , \ldots , m ) , \qquad \forall x \in D _ { c o n d } \colon \sum _ { i = 1 } ^ { m } q _ { i } ( x ) = { \bf 1 }\tag{7}
$$

For each input $x ,$ let $i ^ { * }$ denote the unique active branch index satisfying $\pmb q _ { i ^ { * } } ( { \pmb x } ) = { \pmb 1 }$ The corresponding conditional operator is defined as:

$$
{ \pmb F } _ { c o n d } ( { \pmb x } ) = { \pmb f } _ { i ^ { * } } ( { \pmb \theta } _ { i ^ { * } } ) ( { \pmb x } )\tag{8}
$$

## Parallel fork–join

For a global state C and $p { \geq } 2$ parallel branches, let $\pmb { \pi } _ { D i }$ project C onto the branch-local objects and resources required by branch J ,and let W merge the resulting branch outputs at the synchronization point:

$$
F _ { p a r } ( x ) = J \left( f _ { 1 } ( \theta _ { 1 } ) { \big ( } \pi _ { D 1 } ( x ) { \big ) } , \ldots , f _ { p } { \big ( } \theta _ { p } { \big ) } { \Big ( } \pi _ { D p } ( x ) { \Big ) } \right) , p \geq 2\tag{9}
$$

Let $W _ { i }$ and $\mathbf { { { L } } } _ { i }$ denote the write and read sets of branch J, respectively. Cross-branch write–read and write–write conflicts are excluded by

$$
\forall i , j \in \{ 1 , \dots , p \} , i \neq j \colon W _ { i } \cap \left( L _ { j } \cup W _ { j } \right) = \emptyset .\tag{10}
$$

If two branch-local projections overlap, their outputs must agree on the shared state components before they are joined:

$$
\begin{array} { r l } & { \forall i , j \in \{ 1 , \dots , p \} , i \neq j , D _ { i } \cap D _ { j } \neq \emptyset : } \\ & { } \\ & { \pi _ { D _ { i } \cap D _ { j } } ( f _ { i } ( \theta _ { i } ) \big ( \pi _ { D i } ( x ) \big ) ) = \pi _ { D _ { i } \cap D _ { j } } ( f _ { j } \big ( \theta _ { j } \big ) \Big ( \pi _ { D j } ( x ) \Big ) ) } \end{array}\tag{11}
$$

Equation (10) permits shared read-only state while excluding conflicting cross-branch updates, whereas Equation (11) ensures deterministic merging of overlapping branchlocal outputs. This formulation supports parallel operations on shared read-only state or distinct subsamples, followed by synchronization at the join point.

## Protected atomic interval

For a protected interval A spanning operations $f _ { a }$ through $\scriptstyle f _ { b }$ , where $1 \leq a \leq b \leq n$ , define

$$
A = \{ f _ { a } , \dots , f _ { b } \} , \qquad F _ { A } = f _ { b } ( \theta _ { b } ) \circ \cdots \circ f _ { a } ( \theta _ { a } )\tag{12}
$$

Let $\pmb { \sigma } ( f _ { i } )$ denote the unique position of operation instance $f _ { i }$ in the global execution schedule. The operations in c occupy consecutive schedule positions:

$$
\pmb { \sigma } ( f _ { a + k } ) = \pmb { \sigma } ( f _ { a } ) + k , \qquad k = 0 , \ldots , b - a\tag{13}
$$

No operation outside c may be scheduled within the interval bounded by its endpoints:

$$
\forall h \notin A \colon \pmb { \sigma } ( \mathbf { h } ) \notin [ \pmb { \sigma } ( \pmb { f } _ { a } ) , \pmb { \sigma } ( \pmb { f } _ { b } ) ]\tag{14}
$$

The figure label ${ { F } _ { l o c k } }$ and the mathematical symbol $\pmb { F } _ { A }$ refer to the same protected, noninterleavable execution construct.

Cross-references: Fig. 1c; Supplementary Tables S1 and S4; Supplementary Fig.   
S2; Supplementary Data 2.

# Supplementary Note 4. Nested workflow algebra and serialization

The expanded workflow in Fig. 1d composes the five operator classes within a single graph. Primitive functions form locally ordered sequences. A bounded outer loop controls repeated execution. The loop contains a fork–join region, within which one branch contains an inner loop, one contains a conditional branch and one contains a protected interval. The join synchronizes branch outputs before evaluation of the outer stopping predicate.

The executable representation preserves this hierarchy as an abstract syntax tree. Physical operations are namespace function calls. Parallel blocks are represented by parallel\_execution containing named branch contexts. Protected intervals use atomic\_lock. Conditions use Python if/elif/else, and bounded repetition uses while or for. The converter recursively serializes these constructs into JSON without flattening their nesting.

The current converter produces a canonical structural string using SEQ, PAR, ATOMIC, IF, WHILE and FOR. Individual operations are indexed as f<sub>i</sub>. The canonical string records topology rather than chemical threshold values. The JSON additionally records plan name, variables, ordered steps, workstation names, operation names, parameters, graph paths and switch mappings. Namespace calls retain workstation\_eng\_name, enabling deterministic routing back to a workstation-specific interface.

Cross-references: Fig. 1d; Supplementary Fig. S2; Supplementary Tables S1 and S4; Supplementary Data 2.

## Supplementary Note 5. Capability acquisition and Function-Skill admission

The engineering pipeline converts laboratory capability descriptions into agent-readable and simulator-readable contracts. The supplied release contains 49 workstation specifications: 28 synthesis workstations, 5 reaction and testing workstations, 12 characterization workstations and 4 virtual control-flow workstations. The 45 physical workstations provide laboratory operations, whereas the 4 virtual modules provide parallel, conditional, loop and protected-interval constructs. Across these files, the current snapshot contains 63 declared Function-Skill functions. Shared operation names remain distinguishable through workstation namespaces.

Capability acquisition begins from a structured workstation-information file or manual authoring. The information schema includes workstation identity, operation descriptions, parameters, input and output constraints, state transitions, returns and file parameters. Natural-language Skill documents are admitted through a configurable 100-point assessment. The six required checks contribute 60 points, and a score of at least 60 is required for admission. Nine optional checks contribute the remaining 40 points. Virtual modules and workstations without material state receive documented relaxations for inapplicable checks.

Admitted Skills are converted into Function-Skill interface documents, \_common.py and simulator.py. Each interface function contains a typed signature and an eight-part docstring: Summary, Preconditions, Cautions, Args, Constraints, State Changes, Returns and Example. The converter centralizes shared constants, data classes, result classes, exception categories and namespace classes. The simulator implements the declared checks and state changes.

Fig. 2a–c separates development from use. Stages I–III construct, assess and convert the capability layer. Stage IV loads the control Skill, generates a recipe, analyzes and simulates the recipe, reviews it and returns diagnostics for regeneration when necessary. A generated workflow becomes a candidate for downstream dispatch only after the configured checks and workflow review pass.

The supplied release materials include Additional File 1 (source-code archive) and Additional File 2 (Function-Skill library), with 49 Skill documents, the shared runtime, simulator and conversion/assessment sources. No version tag or commit identifier is provided. Local archive checksums are AF-01: 59C0F68A82C3E7A9EF168F305658D62B5D0D70D2F7B12BB1DF97AAA1EFD430E7 and AF-02:

A3D71B6EB741F95900E3E5505E232B6BF9B1C80C5B5796D9C5F6ABB11DC855E E.

Cross-references: Fig. 2a,b; Supplementary Tables S5 and S6; Supplementary Figs. S3 and S4; Supplementary Data 1.

The recipe optimizer parses a generated Python recipe into an abstract syntax tree before simulation. Its security gate rejects syntax and constructs outside the bounded recipe language. The supplied implementation restricts imports to \_common, blocks dynamic execution and system, network, reflection and file-access functions, and validates function calls against names obtained from the Function-Skill library and common runtime.

The optimizer implements five cumulative levels. O0 performs the security gate. O1 adds recipe admission, normalization and flow analysis. O2 applies conservative AST simplification while preserving workstation-call order and experimental semantics. O3 can disable unused conditional outputs using laboratory-specific rules. O4 can merge compatible consecutive calls to the same workstation operation. Each transformed tree is checked again after modification.

The configuration supplied with the source uses O0 as the default. O0, O1 and O2 are enabled, whereas O3 and O4 are disabled. All four archived case-validation reports record O2 structural analysis as passed; no O3 or O4 transformation was applied.

The converter subsequently serializes the accepted AST into canonical JSON. Static acceptance establishes conformance to the encoded recipe language and workstation call inventory. It does not replace stateful checks, which are performed by the simulator, or scientific review, which evaluates whether the workflow addresses the stated intent.

Cross-references: Fig. 2c; Supplementary Table S7; Supplementary Fig. S5;   
Supplementary Data 4; Additional File 1.

Supplementary Note 7. Stateful simulation and diagnostic-guided regeneration The simulator evaluates each operation against its executable state. For operation k, Fig. 2c defines the validation indicator

$$
V _ { k } ( S i m S t a t e _ { t } , \theta _ { k } ) = A _ { j } \in M _ { k } c _ { k j } ( S i m S t a t e _ { t } , \theta _ { k } ) ,
$$

where $M _ { k }$ is the set of MUST constraints and $c _ { k j }$ is constraint j. When all MUST constraints pass,

$$
V _ { k } = 1 \colon ( S i m S t a t e _ { t + 1 } , \Delta _ { k } ) = ( T _ { k } ( S i m S t a t e _ { t } , \theta _ { k } ) , \emptyset ) ,
$$

where $T _ { k }$ is the encoded transition. When at least one MUST constraint fails,

$$
V _ { k } = 0 \colon ( S i m S t a t e _ { t + 1 } , \Delta _ { k } ) = ( S i m S t a t e _ { t } , \left\{ e _ { k j } \ \middle | \ c _ { k j } = 0 \right\} ) .
$$

This validate-then-commit rule prevents a failed call from partially modifying the simulation state. The simulator continues checking compatible downstream information and returns a batch of diagnostics. SHOULD failures are retained separately as warnings.

Each diagnostic is an ErrorEntry containing function name, step number, unique code, category, message, expected value, actual value, remediation, parameter snapshot and MUST/SHOULD status. The supplied categories cover volume, container, status, parameter, temperature, time, speed, pressure, file, material, sample-state, workflow and general simulation failures. Supplementary Table S8 lists the reporting schema and categories.

The runner dynamically injects simulator implementations into the workstation namespace classes declared in \_common.py. It resets SimState, executes generate\_recipe(), reports errors and warnings separately and returns acceptance only when the MUST-error list is empty. The virtual control-flow contexts record entry and exit events for protected and parallel blocks. The JSON converter preserves the nested topology for downstream adapters.

Fig. 2c is a schematic of the validate–diagnose–regenerate pattern rather than a casespecific archived regeneration trace.

Cross-references: Fig. 2c; Supplementary Table S8; Supplementary Figs. S5 and S6; Supplementary Data 4; Additional File 1.

Supplementary Note 8. Fe coordination environment and alkaline OER workflow Extended Data Fig. 1a specifies a workflow to compare Fe-free Ni(OH)<sub>2</sub>, co-precipitated Fe, post-impregnated Fe, citrate-complexed Fe and EDTA-complexed Fe while holding nominal Fe loading and electrochemical conditions constant. The recipe covers matched precursor addition, room-temperature mixing, staged drying and redispersion, XRD/infrared characterization and alkaline OER testing.

The attached case archive contains a 27-step linear workflow using 11 unique workstations. Its canonical representation is $\mathsf { S E Q } ( \mathsf { f } _ { 1 } ( \Theta _ { 1 } ) , \hdots , \mathsf { f } _ { 2 7 } ( \Theta _ { 2 7 } ) )$ ; no parallel branch or bounded loop is encoded because the same vial class is reused across the formulation routes. O2 structural analysis passed, simulation passed with 27 steps, zero MUST errors and zero SHOULD warnings, and workflow generation returned template 2701712185720832.

The case manifest specifies three independent batches, matched synthesis/OER aliquots and external confirmation of local Fe coordination by Fe K-edge XANES/EXAFS, XPS, ICP-OES and optional $^ { 5 7 } \mathsf { F e }$ Mossbauer spectroscopy. The automated workflow therefore establishes capability-relative structure and simulation admissibility; it does not by itself establish the coordination assignment or measured OER performance.

The case archive supplies the experiment-design manifest, validated JSON plan, executable recipe, validation report and workflow-service response. The reported evidence is limited to automated semantic review, structural checks and stateful simulation; physical execution data are not included.

Cross-references: Extended Data Fig. 1a; Supplementary Tables S9–S11;   
Supplementary Fig. S7; Supplementary Data 3; Additional File 3.

Supplementary Note 9. LDH–HMF synthesis, electrochemistry and multimodal analysis workflow

Extended Data Fig. 1b specifies integrated synthesis and electrochemical testing of layered double hydroxide catalysts for alkaline 5-hydroxymethylfurfural oxidation. Catalyst preparation and electrode fabrication form a protected sequential interval. Electrochemical testing then produces solid-electrode and liquid-product objects that are routed to parallel contact-angle, XRD and liquid-phase product analyses. Their outputs are joined for integrated assessment.

The archived topology is $F _ { h y b r i d } = F _ { p a r } \circ F _ { l o c k } \circ F _ { s e q }$ . It contains a nine-operation protected preparation interval, sequential electrochemical evaluation, three objectspecific analytical branches and an explicit post-branch join.

The attached case archive supplies the 22-step LDH-HMF JSON plan and Python recipe, with a nine-operation protected preparation interval and three parallel post-test branches. O2 analysis passed; simulation executed 31 steps with zero MUST errors and zero SHOULD warnings; workflow generation returned template 2695905526579200. The case-specific run sheet fixes 1.0 M KOH plus 10 mM HMF, while the validation report records adapter and sample-handoff assumptions.

The archive records NiFe-LDH preparation, electrode fabrication, CV/EIS/constantpotential testing, contact-angle, XRD and liquid-chromatography branches, followed by integrated evaluation. The reported evidence is limited to automated semantic review and stateful simulation; physical outcome data are not included. The contact-angle holder adapter, electrode scraping/transfer and electrolyte composition remain explicit execution assumptions.

Cross-references: Extended Data Fig. 1b; Supplementary Tables S9–S11;   
Supplementary Fig. S8; Supplementary Data 3; Additional File 3.

Supplementary Note 10. Furfural hydrogenation and closed-loop GC optimization workflow

Extended Data Fig. 1c specifies catalyst synthesis followed by electrocatalytic hydrogenation of furfural to furfuryl alcohol and closed-loop optimization of gaschromatographic separation. The analytical loop adjusts method variables until the predefined peak-resolution criterion is met. The selected method is then applied to quantify furfural, furfuryl alcohol and major liquid by-products and to calculate conversion, yield and selectivity.

The archived topology is $F _ { h y b r i d } = F _ { s e q } \circ F _ { l o o p }$ . The full workflow must distinguish the chemistry branch from the method-optimization loop and define the join that makes the optimized method available to the quantification step. The loop requires an explicit objective, candidate parameter space, stopping predicate and maximum iteration count. Reaching the bound without satisfying the resolution predicate must return ${ d _ { I o o p } }$ , not a successful method.

The attached case archive supplies the exact furfural workflow specification, Python recipe, canonical JSON, GC method and batch files, and validation report. The workflow contains 20 physical nodes, an O2-passed bounded WHILE loop with $\Nu _ { \mathrm { m a x } } = 6 _ { \mathrm {  m } }$ pass/loop-back/fail-at-cap routes and a final quantification stage. Simulation passed with 23 executed steps, zero MUST errors and zero SHOULD warnings; workflow generation returned template 2696024588517376.

The closed-loop decision requires $\mathsf { R s } \geq 1 . 5$ for both peak pairs, tailing $\leq 2 . 0$ and retentiontime $\mathsf { R S D } \leq 1 . 0 \%$ . The validation report identifies the GC controller, measured metrics and method persistence as assumptions. No chromatograms or physical conversion/yield/selectivity outcomes are supplied, so the case supports workflow construction rather than measured chemical performance.

Cross-references: Extended Data Fig. 1c; Supplementary Tables S9–S11;   
Supplementary Fig. S9; Supplementary Data 3; Additional File 3.

Supplementary Note 11. ZnO synthesis and adaptive spectroscopic characterization workflow

Extended Data Fig. 1d specifies automated ZnO nanoparticle synthesis, centrifugal purification, drying, redispersion and parallel FTIR and UV–vis characterization. The UV– vis branch includes dilution and remeasurement until absorbance lies within a predefined quantitative interval. The FTIR and accepted UV–vis outputs are then joined.

The attached case archive supplies the ZnO Python recipe, canonical JSON, submitted workflow request, solid-weighing workbook and validation report. The workflow contains 27 physical nodes across 15 workstations, two parallel characterization branches and a bounded UV–vis dilution/remeasurement loop with $\mathsf { N } _ { \sf m a x } = 3$ and a 0.20–1.00 AU acceptance interval. O2 analysis passed; simulation executed 30 steps with zero MUST errors and zero SHOULD warnings; workflow generation returned template 2690180845338624.

The FTIR and UV–vis branches use disjoint container classes. The UV–vis guard applies the measured absorbance interval conceptually, while the supplied simulation uses a deterministic starting absorbance and a 0.5 dilution response.

Cross-references: Extended Data Fig. 1d; Supplementary Tables S9–S11;   
Supplementary Fig. S10; Supplementary Data 3; Additional File 3.

# Supplementary Note 12. Reproducibility and release package

Reproduction requires a frozen software release, a frozen Function-Skill library and the exact artifacts for every reported workflow. The release should include the recipe optimizer, simulator runner, simulator implementation, common runtime, recipe converter, Function-Skill converter, quality assessor, configuration files, dependency specification and documentation. It should also include the four prompts, archived recipes, AST/JSON outputs, automated semantic-review reports and simulation logs.

The supplied package now includes Additional File 1 (source-code archive), Additional File 2 (49-workstation Function-Skill library), Additional File 3 (the four-case workflow archive) and Additional File 4 (release-page text). The local source archive contains the optimizer, simulator, converter, runtime and configuration files. A commit identifier, dependency lockfile and final non-sensitive model record remain unspecified.

The local release artifacts have been assigned archive-level SHA-256 values in Supplementary Data 4. Before public or reviewer release, the source archive must exclude credential-bearing configuration, internal object-storage URLs, temporary files and local paths. The AF-04 address is not treated as a persistent public link.

The case dialogues record the scientific prompts and generation dates in the four AF-03 folders. Exact provider/model, decoding parameters, random seeds and a deterministic reproduction statement are not present in the supplied package; those fields remain release metadata rather than inferred values.

Additional Files 1–3 are present locally. Additional File 4 contains an author-supplied internal release-page record, not a verified persistent public or reviewer-access repository. The three local archives are catalogued in Supplementary Table S12 and their checksums recorded in Supplementary Data 4. The supplied package does not include a dependency lockfile or final model metadata, and the source archive requires credential and internal-path sanitization before release.

Cross-references: Supplementary Tables S10–S12; Supplementary Data 4; Additional Files 1–4.

## Supplementary Tables

Supplementary Table S1. Authoritative notation used in Figs. 1 and 2, Extended Data Fig. 1 and the mathematical-definition document.
<table><tr><td colspan="1" rowspan="1">Symbol</td><td colspan="1" rowspan="1">Definition</td><td colspan="1" rowspan="1">Scope or condition</td></tr><tr><td colspan="1" rowspan="1"> $S _ { t }$ </td><td colspan="1" rowspan="1">Typed research-object state atworkflow step t</td><td colspan="1" rowspan="1">Containsidentity, sample, container,evidenceand extensible object-levelfields</td></tr><tr><td colspan="1" rowspan="1">X</td><td colspan="1" rowspan="1">Generic state argument</td><td colspan="1" rowspan="1">Generic state argument supplied toloop, conditional and parallel operators</td></tr><tr><td colspan="1" rowspan="1">t</td><td colspan="1" rowspan="1">Workflow-step index</td><td colspan="1" rowspan="1">Used in St and SimStatet</td></tr><tr><td colspan="1" rowspan="1"> $f _ { j }$ </td><td colspan="1" rowspan="1">Primitive operation i</td><td colspan="1" rowspan="1">Workstation-bound state transition</td></tr><tr><td colspan="1" rowspan="1"> $\mathrm { i , j , k }$ </td><td colspan="1" rowspan="1">Locally    quantified    integerindices</td><td colspan="1" rowspan="1">Index  operations,  branches,   loopchecks, constraints or schedule offsetsas specified in each equation</td></tr><tr><td colspan="1" rowspan="1"> $\theta _ { i }$ </td><td colspan="1" rowspan="1">Parameters of operation i</td><td colspan="1" rowspan="1">Typeddand  constrained   by   thecorresponding Function Skill</td></tr><tr><td colspan="1" rowspan="1"> $\scriptstyle { \theta _ { k } }$ </td><td colspan="1" rowspan="1">Non-negative       error       orconvergence metric evaluatedafter iteration k of a loop.</td><td colspan="1" rowspan="1">Used in $F _ { I o o p } ;$ convergence is reachedwhen $\theta \kappa \leq \varepsilon .$ </td></tr><tr><td colspan="1" rowspan="1">ε</td><td colspan="1" rowspan="1">Prespecified        non-negativeconvergence tolerance; the loopterminates when $\theta \kappa \leq \varepsilon .$ </td><td colspan="1" rowspan="1">Convergence threshold for  $F _ { I o o p } ;$  theloop continues while $\theta _ { k } > \varepsilon .$ </td></tr><tr><td colspan="1" rowspan="1">Z</td><td colspan="1" rowspan="1">Observed          measurement,diagnostic quantity, or state-derived value supplied to aconditional decision function.</td><td colspan="1" rowspan="1">Input quantity supplied to   $Q ( z )$  forevaluation of an $F _ { c o n d }$ branch.</td></tr><tr><td colspan="1" rowspan="1"> $Q ( z )$ </td><td colspan="1" rowspan="1">Scalar      decision      metricevaluated from z for conditionalbranch selection.</td><td colspan="1" rowspan="1">Used in $F _ { c o n d }$ ; the positive branch isselected when Q(z)≥Q thre.</td></tr><tr><td colspan="1" rowspan="1"> $\textsf { Q } _ { t h r e }$ </td><td colspan="1" rowspan="1">Prespecified decision threshold;the positive branch is selectedwhen</td><td colspan="1" rowspan="1">Decision threshold for ; separates thepositive and negative branches.</td></tr><tr><td colspan="1" rowspan="1">r</td><td colspan="1" rowspan="1">Current repetition count of theenclosing workflow.</td><td colspan="1" rowspan="1">Repetition counter for the enclosingworkflow; compared with  $R _ { m a x }$  aftereach repetition.</td></tr><tr><td colspan="1" rowspan="1"> $R _ { m a x }$ </td><td colspan="1" rowspan="1">Maximum permitted number ofrepetitions of theenclosingworkflow     before     forcedtermination.</td><td colspan="1" rowspan="1">Hard repetition   bound;executionterminates when $r \geq R _ { m a x }$ , otherwise theenclosing workflow repeats.</td></tr><tr><td colspan="1" rowspan="1"> $P _ { j }$ </td><td colspan="1" rowspan="1">Precondition   predicate   foroperation i</td><td colspan="1" rowspan="1">Operation is admissible when $P _ { i } ( S _ { t } , \theta _ { i } ) { = } 1$ </td></tr><tr><td colspan="1" rowspan="1"> $P _ { j } ^ { - 1 } ( \{ 1 \} )$ </td><td colspan="1" rowspan="1">Set  of  states   satisfyingprecondition Pj</td><td colspan="1" rowspan="1"> $P _ { j } ^ { - 1 } ( \{ \mathbf { 1 } \} ) = \{ S \in D _ { j } | P _ { j } \big ( S , \pmb \theta _ { j } \big ) = \pmb { 1 } \}$ </td></tr><tr><td colspan="1" rowspan="1"> $D _ { j }$ </td><td colspan="1" rowspan="1">Domain accepted by operationor branch i</td><td colspan="1" rowspan="1">Used as an accepted input domain inEq. (3) and as a branch-local projectiontarget in Eqs. (9) and (11)</td></tr><tr><td colspan="1" rowspan="1"> $F _ { s e q }$ </td><td colspan="1" rowspan="1">Sequential composition</td><td colspan="1" rowspan="1">Defined in Eq. (2); adjacent-domaincompatibility is specified by Eq. (3)</td></tr><tr><td colspan="1" rowspan="1">n</td><td colspan="1" rowspan="1">Number of operations in theconsidered ordered sequence</td><td colspan="1" rowspan="1">Used in Eqs. (2) and (3) and in 1≤a≤b≤n</td></tr><tr><td colspan="1" rowspan="1">B</td><td colspan="1" rowspan="1">Transformation correspondingto one complete loop-bodyexecution</td><td colspan="1" rowspan="1">Eq. (4)</td></tr><tr><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">Identity state transformation</td><td colspan="1" rowspan="1"> $1 ( x ) = x$ </td></tr><tr><td colspan="1" rowspan="1"> $\pmb { \chi } _ { i n }$ </td><td colspan="1" rowspan="1">Loop-entry state</td><td colspan="1" rowspan="1"> ${ \ v { x } } ^ { ( O ) } = \ v { x } _ { i n }$ </td></tr><tr><td colspan="1" rowspan="1"> $x ^ { ( j ) }$ </td><td colspan="1" rowspan="1">State after j completed loop-body executions</td><td colspan="1" rowspan="1"> $\pmb { x } ^ { ( j + 1 ) } = \pmb { B } ( \pmb { x } ^ { ( j ) } )$ </td></tr><tr><td colspan="1" rowspan="1"> $F _ { I o o p }$ </td><td colspan="1" rowspan="1">Bounded loop composition</td><td colspan="1" rowspan="1">Eqs. (4)–(6)</td></tr><tr><td colspan="1" rowspan="1">gj</td><td colspan="1" rowspan="1">Loop continuationpredicateafter check j</td><td colspan="1" rowspan="1">One continues; zero exits</td></tr><tr><td colspan="1" rowspan="1">T</td><td colspan="1" rowspan="1">Realized loop count</td><td colspan="1" rowspan="1">Bounded by $N _ { m a x }$ </td></tr><tr><td colspan="1" rowspan="1"> $N _ { m a x }$ </td><td colspan="1" rowspan="1">Maximum permitted loop count</td><td colspan="1" rowspan="1">Produces $d _ { I o o p }$  if continuation remainstrue</td></tr><tr><td colspan="1" rowspan="1"> $d _ { I o o p }$ </td><td colspan="1" rowspan="1">Non-convergence diagnostic</td><td colspan="1" rowspan="1">Non-empty under Eq. (6)</td></tr><tr><td colspan="1" rowspan="1"> ${ \sf q } _ { \sf \dot { \sf \Lambda } } ( { \sf x } )$ </td><td colspan="1" rowspan="1">Binary guard associated withconditional branch i</td><td colspan="1" rowspan="1">exactly one guard is active for eachadmissible input</td></tr><tr><td colspan="1" rowspan="1">*</td><td colspan="1" rowspan="1">Active conditional-branch indexfor the current input x</td><td colspan="1" rowspan="1">Unique index satisfying $\pmb q _ { i ^ { * } } ( { \pmb x } ) = { \pmb 1 }$ </td></tr><tr><td colspan="1" rowspan="1"> $F _ { c o n d }$ </td><td colspan="1" rowspan="1">Conditional composition</td><td colspan="1" rowspan="1">Unique active branch under Eqs. (7) and(8)</td></tr><tr><td colspan="1" rowspan="1"> $m$ </td><td colspan="1" rowspan="1">Number of conditional branches</td><td colspan="1" rowspan="1">Exactly one qi(x) equals one</td></tr><tr><td colspan="1" rowspan="1"> $F _ { p a r }$ </td><td colspan="1" rowspan="1">Parallel fork-join composition</td><td colspan="1" rowspan="1">Eqs. (9)–(11)</td></tr><tr><td colspan="1" rowspan="1"> $p$ </td><td colspan="1" rowspan="1">Number of parallel branches</td><td colspan="1" rowspan="1"> $p { \geq } 2$ </td></tr><tr><td colspan="1" rowspan="1"> $\pi _ { D i }$ </td><td colspan="1" rowspan="1">Projection onto branch   $\boldsymbol { i } \mathsf { s }$ required state and resources</td><td colspan="1" rowspan="1">Applied before branch execution</td></tr><tr><td colspan="1" rowspan="1"> $J$ </td><td colspan="1" rowspan="1">Join operator</td><td colspan="1" rowspan="1">Merges consistent branch outputs</td></tr><tr><td colspan="1" rowspan="1"> $W _ { i }$ </td><td colspan="1" rowspan="1">Write set of branch i</td><td colspan="1" rowspan="1">Used in Eq. (10)</td></tr><tr><td colspan="1" rowspan="1"> $L _ { j }$ </td><td colspan="1" rowspan="1">Read set of branch i</td><td colspan="1" rowspan="1">Used in Eq. (10); retain this symbol fromthe supplied definition</td></tr><tr><td colspan="1" rowspan="1"> $A$ </td><td colspan="1" rowspan="1">Protected  atomic  interval $\{ f _ { a } , \ldots , f _ { b } \}$ </td><td colspan="1" rowspan="1">Eq. (12)</td></tr><tr><td colspan="1" rowspan="1"> $F _ { A }$ </td><td colspan="1" rowspan="1">Composition inside protectedinterval A</td><td colspan="1" rowspan="1">Same control-flow class as figure label $F _ { I O C k }$ </td></tr><tr><td colspan="1" rowspan="1"> $F _ { I o c k }$ </td><td colspan="1" rowspan="1">Figure label for protected non-interleavable composition</td><td colspan="1" rowspan="1">Use consistently with $F _ { A }$ </td></tr><tr><td colspan="1" rowspan="1"> $\sigma ( \boldsymbol { f } )$ </td><td colspan="1" rowspan="1">Position of operation f in theschedule</td><td colspan="1" rowspan="1">Eqs. (13) and (14)</td></tr><tr><td colspan="1" rowspan="1"> $h$ </td><td colspan="1" rowspan="1">Scheduled operation instanceoutside protected interval A</td><td colspan="1" rowspan="1">Used in Eq. (14)</td></tr><tr><td colspan="1" rowspan="1"> $F _ { h y b r i d }$ </td><td colspan="1" rowspan="1">Workflow composed from two ormore operator classes</td><td colspan="1" rowspan="1">Used in Fig. 1d and Extended Data Fig.1</td></tr><tr><td colspan="1" rowspan="1"> $k$ </td><td colspan="1" rowspan="1">Wash-loop iteration index</td><td colspan="1" rowspan="1"> $\ k { = } 1 , \ldots , \top ;$   used in   ${ \pmb S } _ { i } ^ { ( k ) }$   for statesgenerated during the k-th execution ofthe wash-loop body</td></tr><tr><td colspan="1" rowspan="1"> $S i ^ { ( k ) }$ </td><td colspan="1" rowspan="1">Research-object    state   atworkflow node i during wash-loop iteration k</td><td colspan="1" rowspan="1">Applied to $i \in \{ 4 , 5 , 6 , 7 , 8 , 9 \}$ </td></tr><tr><td colspan="1" rowspan="1">SimStatet</td><td colspan="1" rowspan="1">Executable simulator state atstep t</td><td colspan="1" rowspan="1">Bounded executable projection of theconceptual research-object collection $S _ { t }$ </td></tr><tr><td colspan="1" rowspan="1"> $V _ { k }$ </td><td colspan="1" rowspan="1">MUST-validation indicator foroperation k</td><td colspan="1" rowspan="1">Conjunction of $c _ { k j }$ </td></tr><tr><td colspan="1" rowspan="1"> $M _ { k }$ </td><td colspan="1" rowspan="1">Set of MUST constraints foroperation k</td><td colspan="1" rowspan="1">Used by $V _ { k }$ </td></tr><tr><td colspan="1" rowspan="1"> $c _ { k j }$ </td><td colspan="1" rowspan="1">Constraint j for operation k</td><td colspan="1" rowspan="1">Boolean predicate</td></tr><tr><td colspan="1" rowspan="1"> $T _ { k }$ </td><td colspan="1" rowspan="1">Encoded state transition foroperation k</td><td colspan="1" rowspan="1">Committed only when $V _ { k } { = } 1$ </td></tr><tr><td colspan="1" rowspan="1"> $\Delta _ { \mathsf { k } }$ </td><td colspan="1" rowspan="1">Structured    diagnostic    setreturned for operation k</td><td colspan="1" rowspan="1">Empty on success; contains failedconstraints on failure</td></tr><tr><td colspan="1" rowspan="1"> $\Theta _ { \boldsymbol { k } j }$ </td><td colspan="1" rowspan="1">Diagnostic  associated  withfailed $c _ { k j }$ </td><td colspan="1" rowspan="1">Included when $\scriptstyle c _ { k j } = 0$ </td></tr></table>

Supplementary Table S2. Conceptual research-object fields and their implemented simulation projection.
<table><tr><td rowspan=1 colspan=1>Component</td><td rowspan=1 colspan=1>Conceptual fieldsin Fig. 1b</td><td rowspan=1 colspan=1>Currentexecutable fields</td><td rowspan=1 colspan=1>Evidence or limitation</td></tr><tr><td rowspan=1 colspan=1>Identity</td><td rowspan=1 colspan=1>Object identifierand provenancelinks</td><td rowspan=1 colspan=1>Container keycombines type andidentifier; step logretains operationorder</td><td rowspan=1 colspan=1>Object-level provenancebeyond container identityrequires case JSON</td></tr><tr><td rowspan=1 colspan=1>Sample</td><td rowspan=1 colspan=1>Identity,composition,phase, mass,volume andextensibleproperties</td><td rowspan=1 colspan=1>sample_status,sample_volume_ml,sample_mass_g</td><td rowspan=1 colspan=1>Composition isrepresented in recipeparameters or caserecords, not as a generalSimState field</td></tr><tr><td rowspan=1 colspan=1>Container</td><td rowspan=1 colspan=1>Type, shape,capacity, lid orholder state andlocation</td><td rowspan=1 colspan=1>container_id,container_type,container_status,location</td><td rowspan=1 colspan=1>Geometry and capacitylimits are encoded in Skillconstraints and sharedconstants</td></tr><tr><td rowspan=1 colspan=1>Evidence</td><td rowspan=1 colspan=1>Temperature, pH,colour, XRD, UV–vis and otheroutputs</td><td rowspan=1 colspan=1>Typed resultobjects,test_result_file, logsand diagnostics</td><td rowspan=1 colspan=1>Operation-specificevidence schemas arerecorded in typed resultclasses and case JSONrecords</td></tr><tr><td rowspan=1 colspan=1>Capabilitybinding</td><td rowspan=1 colspan=1>Acceptedcontainer, phaseor volume,parameter limitsand expectedeffects</td><td rowspan=1 colspan=1>Functionnamespace, typedsignature,preconditions,MUST/SHOULDconstraints andstate changes</td><td rowspan=1 colspan=1>Defined in Function-Skilldocuments</td></tr><tr><td rowspan=1 colspan=1>State history</td><td rowspan=1 colspan=1>St → St+1 withretainedprovenance</td><td rowspan=1 colspan=1>SimState.logs, stepcounter andcommittedcontainer state</td><td rowspan=1 colspan=1>Stepwise history isretained in SimState.logsand the case JSONrecords</td></tr></table>

Supplementary Table S3. Function-Skill contract schema.
<table><tr><td rowspan=1 colspan=1>Contractelement</td><td rowspan=1 colspan=1>Required content</td><td rowspan=1 colspan=1>Executablerealization</td><td rowspan=1 colspan=1>Acceptancerequirement</td></tr><tr><td rowspan=1 colspan=1>Workstation</td><td rowspan=1 colspan=1>Unique Englishnamespace,Chinese name,module andidentifier</td><td rowspan=1 colspan=1>Namespace classin _common.py</td><td rowspan=1 colspan=1>Namespace must resolveto a registered simulatorfunction</td></tr><tr><td rowspan=1 colspan=1>Summary</td><td rowspan=1 colspan=1>One-sentenceoperation purpose</td><td rowspan=1 colspan=1>Docstring</td><td rowspan=1 colspan=1>Must agree with sourcecapability description</td></tr><tr><td rowspan=1 colspan=1>Preconditions</td><td rowspan=1 colspan=1>Accepted object,container andsample states</td><td rowspan=1 colspan=1>require_* checksand operation-specific checks</td><td rowspan=1 colspan=1>Every declared MUSTprecondition must beimplemented</td></tr><tr><td rowspan=1 colspan=1>Cautions</td><td rowspan=1 colspan=1>Cross-step effectsand device-specificrestrictions</td><td rowspan=1 colspan=1>Documentationand automatedchecks</td><td rowspan=1 colspan=1>Must not be convertedinto hidden simulatorbehavior</td></tr><tr><td rowspan=1 colspan=1>Args</td><td rowspan=1 colspan=1>Typed values withunit-bearing namesand enumerations</td><td rowspan=1 colspan=1>Function signatureand data classes</td><td rowspan=1 colspan=1>Values must parse andfall within documentedtypes</td></tr><tr><td rowspan=1 colspan=1>Constraints</td><td rowspan=1 colspan=1>MUST/SHOULDBoolean conditionswith failurecategories</td><td rowspan=1 colspan=1>record_error andrecord_warning</td><td rowspan=1 colspan=1>All feasible checks arecollected in one pass</td></tr><tr><td rowspan=1 colspan=1>StateChanges</td><td rowspan=1 colspan=1>Explicit target fieldsand effects</td><td rowspan=1 colspan=1>Validate-then-commit update</td><td rowspan=1 colspan=1>No state update isallowed after a failedMUST rule</td></tr><tr><td rowspan=1 colspan=1>Returns</td><td rowspan=1 colspan=1>StepResult oroperation-specificsubclass</td><td rowspan=1 colspan=1>Typed resultobject</td><td rowspan=1 colspan=1>Data-producingoperations includeevidence references</td></tr><tr><td rowspan=1 colspan=1>Example</td><td rowspan=1 colspan=1>Valid, internallyconsistent call</td><td rowspan=1 colspan=1>Skill document</td><td rowspan=1 colspan=1>Must use a registerednamespace and legalvalues</td></tr><tr><td rowspan=1 colspan=1>Raises ordiagnostics</td><td rowspan=1 colspan=1>Category, message,expected, actualand remediation</td><td rowspan=1 colspan=1>ErrorEntry</td><td rowspan=1 colspan=1>MUST and SHOULDrecords remain separate</td></tr></table>

Supplementary Table S4. Composition operators, invariants and code representation.
<table><tr><td rowspan=1 colspan=1>Operator</td><td rowspan=1 colspan=1>Formaldefinition</td><td rowspan=1 colspan=1>Required invariant</td><td rowspan=1 colspan=1>Pythonrepresentation</td><td rowspan=1 colspan=1>CanonicalJSONtoken</td></tr><tr><td rowspan=1 colspan=1>Sequence</td><td rowspan=1 colspan=1>Eqs. (2), (3)</td><td rowspan=1 colspan=1>Each output lies in thenext domain and eachprecondition holds</td><td rowspan=1 colspan=1>Orderedstatements</td><td rowspan=1 colspan=1>SEQ(...)</td></tr><tr><td rowspan=1 colspan=1>Boundedloop</td><td rowspan=1 colspan=1>Eqs. (4)–(6)</td><td rowspan=1 colspan=1>Explicit continuationpredicate, finite Nmax,diagnostic at non-convergence</td><td rowspan=1 colspan=1>while or for</td><td rowspan=1 colspan=1>WHILE(...)or FOR(...)</td></tr><tr><td rowspan=1 colspan=1>Conditional</td><td rowspan=1 colspan=1>Eqs. (7), (8)</td><td rowspan=1 colspan=1>Exactly one branch isselected</td><td rowspan=1 colspan=1>if/elif/else</td><td rowspan=1 colspan=1>IF(...</td></tr><tr><td rowspan=1 colspan=1>Parallelfork-join</td><td rowspan=1 colspan=1>Eqs. (9)–(11)</td><td rowspan=1 colspan=1>No write conflict andagreement on sharedstate</td><td rowspan=1 colspan=1>parallel_executionwith branchcontexts</td><td rowspan=1 colspan=1>PAR(...)</td></tr><tr><td rowspan=1 colspan=1>Protectedinterval</td><td rowspan=1 colspan=1>Eqs. (12)–(14)</td><td rowspan=1 colspan=1>Contiguousscheduled positionswith no externalinterleaving</td><td rowspan=1 colspan=1>atomic_lockcontext</td><td rowspan=1 colspan=1>ATOMIC(...)</td></tr></table>

Supplementary Table S5. Capability inventory in the supplied Function-Skill snapshot.
<table><tr><td rowspan=1 colspan=1>Module</td><td rowspan=1 colspan=1>Workstations, grouped by declaredcapability</td><td rowspan=1 colspan=1>Workstationcount</td><td rowspan=1 colspan=1>Declaredfunctions</td></tr><tr><td rowspan=1 colspan=1>Synthesis</td><td rowspan=1 colspan=1>Container storage; general material;liquid handling at 1 ml, 4 channelsand 5 ml in multiple versions; liquidpouring; drying; single- and multi-channel solid dosing; solid transfer;centrifugation; cleaning anddispensing; cooling; heat-resistantcontainer supply; heated and room-temperature stirring; muffle furnace;plate storage; spectroscopy holderpreparation; ultrasonic dispersion andliquid handling; intelligentphotocatalysis container transfer;purification</td><td rowspan=1 colspan=1>28</td><td rowspan=1 colspan=1>41</td></tr><tr><td rowspan=1 colspan=1>Reaction andtesting</td><td rowspan=1 colspan=1>Dual-station electrochemistry; high-temperature/high-pressuremicroreaction; photocatalysis V1 andV2; post-reaction processing</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>6</td></tr><tr><td rowspan=1 colspan=1>Characterization</td><td rowspan=1 colspan=1>Dark-box imaging; fluorescence; gaschromatography; high-speed imaging;infrared; contact angle; illuminationand membrane clamping; liquidchromatography; microplate reader;spectroscopy container transfer; UV—visible spectroscopy; XRD</td><td rowspan=1 colspan=1>12</td><td rowspan=1 colspan=1>13</td></tr><tr><td rowspan=1 colspan=1>Virtual controlflow</td><td rowspan=1 colspan=1>Atomic lock; parallel branch;conditional jump; loop</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>3 explicitcontextfunctionsplus nativePythoncontrol</td></tr><tr><td rowspan=1 colspan=1>Total</td><td rowspan=1 colspan=1>45 physical and 4 virtual workstationspecifications</td><td rowspan=1 colspan=1>49</td><td rowspan=1 colspan=1>63</td></tr></table>

Supplementary Table S6. Natural-language Skill quality-assessment rubric.
<table><tr><td rowspan=1 colspan=1>Check</td><td rowspan=1 colspan=1>Maximumscore</td><td rowspan=1 colspan=1>Requirement</td></tr><tr><td rowspan=1 colspan=1>R1 Workstation name/title</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Name, code and primary title</td></tr><tr><td rowspan=1 colspan=1>R2 Operation description</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Explicit operation section andpurpose</td></tr><tr><td rowspan=1 colspan=1>R3 Parameter information</td><td rowspan=1 colspan=1>15</td><td rowspan=1 colspan=1>Parameter names, types, levels,units and constraints</td></tr><tr><td rowspan=1 colspan=1>R4 Initial state</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Container type, container state andsample state where applicable</td></tr><tr><td rowspan=1 colspan=1>R5 State update</td><td rowspan=1 colspan=1>10</td><td rowspan=1 colspan=1>Explicit effect or state-transitionmapping</td></tr><tr><td rowspan=1 colspan=1>R6 Constraint rules</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Business and parameter constraints</td></tr><tr><td rowspan=1 colspan=1>Required subtotal</td><td rowspan=1 colspan=1>60</td><td rowspan=1 colspan=1>Admission threshold is at least 60</td></tr><tr><td rowspan=1 colspan=1>O1 Frontmatter completeness</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Name, description, module andcode</td></tr><tr><td rowspan=1 colspan=1>O2 Similar-workstation distinction</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Version or capability distinction</td></tr><tr><td rowspan=1 colspan=1>O3 Overall workflow constraints</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Required predecessor or processdependency</td></tr><tr><td rowspan=1 colspan=1>O4 Parameter-constraint format</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Explicit ranges, enumeration orconditional dependency</td></tr><tr><td rowspan=1 colspan=1>O5 Nested-parameter expansion</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Structured nested inputs</td></tr><tr><td rowspan=1 colspan=1>O6 File parameters</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Template, format and transferrequirements</td></tr><tr><td rowspan=1 colspan=1>O7 Return values</td><td rowspan=1 colspan=1>5</td><td rowspan=1 colspan=1>Typed output schema</td></tr><tr><td rowspan=1 colspan=1>O8 State-transition completeness</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>Coverage of declared initial-statefields</td></tr><tr><td rowspan=1 colspan=1>O9 Example completeness</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>Valid example values and call</td></tr><tr><td rowspan=1 colspan=1>Optional subtotal</td><td rowspan=1 colspan=1>40</td><td rowspan=1 colspan=1>Used to assess conversioncompleteness</td></tr></table>

Supplementary Table S7. Recipe optimizer levels and supplied configuration.
<table><tr><td rowspan=1 colspan=1>Level</td><td rowspan=1 colspan=1>Implemented behavior</td><td rowspan=1 colspan=1>Suppliedconfiguration</td><td rowspan=1 colspan=1>Final case-use evidence</td></tr><tr><td rowspan=1 colspan=1>00</td><td rowspan=1 colspan=1>Syntax and attack-surfacegate; import, node, nameand function-callrestrictions</td><td rowspan=1 colspan=1>Enabled;default</td><td rowspan=1 colspan=1>Included in the O2 checksrecorded for all four cases; noseparate command linesupplied</td></tr><tr><td rowspan=1 colspan=1>01</td><td rowspan=1 colspan=1>O0 plus generate_recipe()admission, normalization,flow warnings and mergesuggestions</td><td rowspan=1 colspan=1>Enabled</td><td rowspan=1 colspan=1>Included in the O2 admissionand flow checks</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>O1 plus conservative ASTsimplification; noworkstation-call reordering</td><td rowspan=1 colspan=1>Enabled</td><td rowspan=1 colspan=1>Passed for Fecoordination/OER, LDH-HMF,furfural and ZnO</td></tr><tr><td rowspan=1 colspan=1>03</td><td rowspan=1 colspan=1>O2 plus disabling ofunused conditional outputsunder configured rules</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>Disabled; no case evidence ofuse</td></tr><tr><td rowspan=1 colspan=1>04</td><td rowspan=1 colspan=1>O3 plus compatibleconsecutive-call merging</td><td rowspan=1 colspan=1>Disabled</td><td rowspan=1 colspan=1>Disabled; no case evidence ofuse; advisory merges were notapplied</td></tr></table>

Supplementary Table S8. Structured simulation diagnostics.
<table><tr><td rowspan=1 colspan=1>Field or category</td><td rowspan=1 colspan=1>Meaning</td><td rowspan=1 colspan=1>Required reporting</td></tr><tr><td rowspan=1 colspan=1>function_name</td><td rowspan=1 colspan=1>Operation that produced thediagnostic</td><td rowspan=1 colspan=1>Registered namespaceoperation</td></tr><tr><td rowspan=1 colspan=1>step_number</td><td rowspan=1 colspan=1>Position in simulatedworkflow</td><td rowspan=1 colspan=1>Integer step identifier</td></tr><tr><td rowspan=1 colspan=1>code</td><td rowspan=1 colspan=1>Unique machine-readableerror code</td><td rowspan=1 colspan=1>Stable across reruns of thesame release</td></tr><tr><td rowspan=1 colspan=1>category</td><td rowspan=1 colspan=1>Diagnostic class</td><td rowspan=1 colspan=1>One of the release categoriesbelow</td></tr><tr><td rowspan=1 colspan=1>message</td><td rowspan=1 colspan=1>Observed failure</td><td rowspan=1 colspan=1>Current value and expectedcondition</td></tr><tr><td rowspan=1 colspan=1>expected, actual</td><td rowspan=1 colspan=1>Normalized comparison</td><td rowspan=1 colspan=1>Explicit where applicable</td></tr><tr><td rowspan=1 colspan=1>remediation</td><td rowspan=1 colspan=1>Proposed repair</td><td rowspan=1 colspan=1>Parameter adjustment ormissing/reordered operation</td></tr><tr><td rowspan=1 colspan=1>params_snapshot</td><td rowspan=1 colspan=1>Relevant call arguments</td><td rowspan=1 colspan=1>Sufficient to reproduce thefailure</td></tr><tr><td rowspan=1 colspan=1>is_must</td><td rowspan=1 colspan=1>Blocking status</td><td rowspan=1 colspan=1>MUST enters errors; SHOULDenters warnings</td></tr><tr><td rowspan=1 colspan=1>Volume and volume-constraint categories</td><td rowspan=1 colspan=1>Insufficient or excessivesample or transfer volume</td><td rowspan=1 colspan=1>VolumeError,VolumeConstraintError</td></tr><tr><td rowspan=1 colspan=1>Container andmaterial categories</td><td rowspan=1 colspan=1>Missing, incompatible orexcessive container/material</td><td rowspan=1 colspan=1>ContainerConstraintError,MaterialConstraintError</td></tr><tr><td rowspan=1 colspan=1>State categories</td><td rowspan=1 colspan=1>Invalid container or samplestate</td><td rowspan=1 colspan=1>StatusError,SampleStatusError</td></tr><tr><td rowspan=1 colspan=1>Parametercategories</td><td rowspan=1 colspan=1>Missing, out-of-range,dependent or otherwiseinvalid parameter</td><td rowspan=1 colspan=1>ParameterError,ParameterRangeError,ParameterConstraintError,ParameterDependencyError,ParameterMissingError</td></tr><tr><td rowspan=1 colspan=1>Physical-limitcategories</td><td rowspan=1 colspan=1>Temperature, time, speed orpressure failure</td><td rowspan=1 colspan=1>TemperatureError, TimeError,SpeedError, PressureError</td></tr><tr><td rowspan=1 colspan=1>File categories</td><td rowspan=1 colspan=1>Invalid file or template input</td><td rowspan=1 colspan=1>FileFormatError,FileConstraintError</td></tr><tr><td rowspan=1 colspan=1>Workflow andsimulation categories</td><td rowspan=1 colspan=1>Missing, unordered orotherwise invalid workflowelement</td><td rowspan=1 colspan=1>WorkflowError, SimulationError</td></tr></table>

Supplementary Table S9. Extended Data Fig. 1 scientific intents and displayed operator topologies.
<table><tr><td rowspan=1 colspan=1>Panel</td><td rowspan=1 colspan=1>Scientific intent</td><td rowspan=1 colspan=1>Displayed algebra</td><td rowspan=1 colspan=1>Required evidence</td><td rowspan=1 colspan=1>Reconciliation status</td></tr><tr><td rowspan=1 colspan=1>ExtendedData Fig.1a</td><td rowspan=1 colspan=1>Fe coordinationenvironmentmodulation andalkaline OER</td><td rowspan=1 colspan=1> $F _ { h y b r i d } = F _ { s e q }$ </td><td rowspan=1 colspan=1>Five Fe-introduction routes,matched OER aliquots,XRD/IR outputs andexternal coordinationconfirmation</td><td rowspan=1 colspan=1>27-step linear archive;O2/simulation/workflow generationpassed; coordination assignment andphysical OER data remain external</td></tr><tr><td rowspan=1 colspan=1>ExtendedData Fig.1b</td><td rowspan=1 colspan=1>LDH synthesis, HMFoxidation andmultimodal analysis</td><td rowspan=1 colspan=1> $F _ { h y b r i d } = F _ { p a r } \circ F _ { l o c k } \circ F _ { s e q }$ </td><td rowspan=1 colspan=1>Protected endpoints, objectprojections, parallelbranches and join</td><td rowspan=1 colspan=1>Exact archive supplied; O2 andsimulation passed with zero errors;adapter assumptions and physicaloutcomes remain explicit</td></tr><tr><td rowspan=1 colspan=1>ExtendedData Fig.1c</td><td rowspan=1 colspan=1>Furfuralhydrogenation andclosed-loop GCoptimization</td><td rowspan=1 colspan=1> $F _ { h y b r i d } = F _ { s e q } \circ F _ { l o o p }$ </td><td rowspan=1 colspan=1>Objective, parameterspace, stopping predicate,bound and method trace</td><td rowspan=1 colspan=1>Exact archive supplied; bounded loopand O2/simulation passed;chromatographic measurementsremain assumptions</td></tr><tr><td rowspan=1 colspan=1>ExtendedData Fig.1d</td><td rowspan=1 colspan=1>ZnO synthesis andadaptive FTIR/UV—visible analysis</td><td rowspan=1 colspan=1> $F _ { h y b r i d }$  $= F _ { p a r } \circ F _ { l o o p } \circ F _ { c o n d } \circ F _ { s e q }$ </td><td rowspan=1 colspan=1>UV–vis condition, dilutionaction, repeat bound andjoin</td><td rowspan=1 colspan=1>Exact archive supplied; boundedultraviolet-visible loop and nestedconditional are explicit in canonicalJSON</td></tr></table>

Supplementary Table S10. Required per-case artifact and graph-metric record.
<table><tr><td colspan="1" rowspan="1">Record</td><td colspan="1" rowspan="1">Fe coordination/OER</td><td colspan="1" rowspan="1">LDH-HMF</td><td colspan="1" rowspan="1">Furfural</td><td colspan="1" rowspan="1">ZnO</td></tr><tr><td colspan="1" rowspan="1">Exact prompt</td><td colspan="1" rowspan="1">Dialogue history.md; exact Fecoordination/OER prompt</td><td colspan="1" rowspan="1">Dialogue history.md;exact LDH-HMF prompt</td><td colspan="1" rowspan="1">Dialogue history.md;exact furfural/GC prompt</td><td colspan="1" rowspan="1">Dialogue history.md; exact ZnOprompt</td></tr><tr><td colspan="1" rowspan="1">First-passrecipe</td><td colspan="1" rowspan="1">One archived recipe; noseparate v1/final pair</td><td colspan="1" rowspan="1">One archived recipe; noseparate v1/final pair</td><td colspan="1" rowspan="1">One archived recipe; noseparate v1/final pair</td><td colspan="1" rowspan="1">One archived recipe; noseparate v1/final pair</td></tr><tr><td colspan="1" rowspan="1">Final recipe</td><td colspan="1" rowspan="1">Same archived recipe; noseparate final pair</td><td colspan="1" rowspan="1">Same archived recipe;no separate final pair</td><td colspan="1" rowspan="1">Same archived recipe; noseparate final pair</td><td colspan="1" rowspan="1">Same archived recipe; noseparate final pair</td></tr><tr><td colspan="1" rowspan="1">CanonicalAST/JSON</td><td colspan="1" rowspan="1">experiment_plan.json;SEQ(f1-f27)</td><td colspan="1" rowspan="1">recipe_ldh_hmf_oxidation.json; ATOMIC + PAR</td><td colspan="1" rowspan="1">recipe_furfural_furfuryl_alcohol_gc_loop.json;WHILE</td><td colspan="1" rowspan="1">recipe_ZnO_nanoparticle_adaptive_spectroscopy.json; PAR +WHILE</td></tr><tr><td colspan="1" rowspan="1">Primitive-operation nodes</td><td colspan="1" rowspan="1">27</td><td colspan="1" rowspan="1">22</td><td colspan="1" rowspan="1">20</td><td colspan="1" rowspan="1">27</td></tr><tr><td colspan="1" rowspan="1">Unique physicalworkstations</td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">11</td><td colspan="1" rowspan="1">15</td></tr><tr><td colspan="1" rowspan="1">Sequenceblocks</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">5</td><td colspan="1" rowspan="1">2</td><td colspan="1" rowspan="1">4</td></tr><tr><td colspan="1" rowspan="1">Loop blocksand bounds</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">1 (Nmax = 6)</td><td colspan="1" rowspan="1">1 (Nmax = 3)</td></tr><tr><td colspan="1" rowspan="1">Conditionalblocks</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">1</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">Parallelbranches</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">3</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">2</td></tr><tr><td colspan="1" rowspan="1">Protectedintervals</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">1 (f1-f9)</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td></tr><tr><td colspan="1" rowspan="1">MUST errorsbeforeregeneration</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td></tr><tr><td colspan="1" rowspan="1">MUST errorsafterregeneration</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td></tr><tr><td colspan="1" rowspan="1">SHOULDwarnings afterregeneration</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td><td colspan="1" rowspan="1">0</td></tr></table>

Supplementary Table S11. Automated semantic review and simulation outcomes.
<table><tr><td rowspan=1 colspan=1>Case</td><td rowspan=1 colspan=1>Automated semantic review</td><td rowspan=1 colspan=1>Stateful simulation</td></tr><tr><td rowspan=1 colspan=1>Fecoordination/OER</td><td rowspan=1 colspan=1>Passed with warnings; externalcoordination confirmation remainsoutside the automated workflow.</td><td rowspan=1 colspan=1>Passed; 27 steps, 0MUST errors, 0 SHOULDwarnings.</td></tr><tr><td rowspan=1 colspan=1>LDH-HMF</td><td rowspan=1 colspan=1>Passed with warnings; adapter andsample-handoff assumptions arerecorded.</td><td rowspan=1 colspan=1>Passed; 31 steps, 0MUST errors, 0 SHOULDwarnings.</td></tr><tr><td rowspan=1 colspan=1>Furfural-GC</td><td rowspan=1 colspan=1>Passed with assumptions; GCmetric publication and methodpersistence are external controllerassumptions.</td><td rowspan=1 colspan=1>Passed; 23 steps, 0MUST errors, 0 SHOULDwarnings.</td></tr><tr><td rowspan=1 colspan=1>ZnOspectroscopy</td><td rowspan=1 colspan=1>Passed with information;absorbance and dilution responseare simulation controls.</td><td rowspan=1 colspan=1>Passed; 30 steps, 0MUST errors, 0 SHOULDwarnings.</td></tr></table>

Supplementary Table S12. Software, model and release manifest.
<table><tr><td rowspan=1 colspan=1>Artifact</td><td rowspan=1 colspan=1>Role</td><td rowspan=1 colspan=1>Supplied status</td></tr><tr><td rowspan=1 colspan=1>recipe_optimizer.py</td><td rowspan=1 colspan=1>Security, admission andAST optimization</td><td rowspan=1 colspan=1>AF-01 source archive; localarchive hash recordedbelow</td></tr><tr><td rowspan=1 colspan=1>run_simulation.py</td><td rowspan=1 colspan=1>Runtime registrationand simulation entrypoint</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>references/simulator.py</td><td rowspan=1 colspan=1>Constraint checks andstate transitions</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>references/_common.py</td><td rowspan=1 colspan=1>Shared constants,types, results andnamespaces</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>recipe_converter.py</td><td rowspan=1 colspan=1>Python/JSONconversion andcanonical form</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>skills_to_functions.py</td><td rowspan=1 colspan=1>Natural-language toFunction-Skillconversion</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>skill_quality_assessor.py</td><td rowspan=1 colspan=1>Skill admission</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>optimizer_config.json</td><td rowspan=1 colspan=1>Enabled optimizerlevels</td><td rowspan=1 colspan=1>AF-01 source archive; O0—O2 enabled, O3–O4disabled</td></tr><tr><td rowspan=1 colspan=1>o3_conditional_rules.yaml</td><td rowspan=1 colspan=1>Conditional-outputoptimization rules</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>domain_validation_rules.yaml</td><td rowspan=1 colspan=1>Laboratory-specificvalidation rules</td><td rowspan=1 colspan=1>AF-01 source archive</td></tr><tr><td rowspan=1 colspan=1>quality_scoring_config.json</td><td rowspan=1 colspan=1>Skill admission rubric</td><td rowspan=1 colspan=1>AF-01 source archive;threshold 60</td></tr><tr><td rowspan=1 colspan=1>Python and dependencies</td><td rowspan=1 colspan=1>Execution environment</td><td rowspan=1 colspan=1>No lockfile in suppliedpackage</td></tr><tr><td rowspan=1 colspan=1>Generative model</td><td rowspan=1 colspan=1>Recipe or Skillgeneration</td><td rowspan=1 colspan=1>AF-03 dialogues andvalidation reports</td></tr><tr><td rowspan=1 colspan=1>Source-code release</td><td rowspan=1 colspan=1>Reproduction package</td><td rowspan=1 colspan=1>AF-01 Source-codearchive.zip</td></tr><tr><td rowspan=1 colspan=1>Function-Skill release</td><td rowspan=1 colspan=1>Capability package</td><td rowspan=1 colspan=1>AF-02 Function-Skilllibrary.zip</td></tr><tr><td rowspan=1 colspan=1>Extended Data Fig. 1 casearchive</td><td rowspan=1 colspan=1>Exact prompts, recipesand logs</td><td rowspan=1 colspan=1>AF-03 Extended Data Fig.1 workflow archive.zip</td></tr><tr><td rowspan=1 colspan=1>Release-page record</td><td rowspan=1 colspan=1>Internal provenancerecord; not a verifiedpersistent public/reviewerroute</td><td rowspan=1 colspan=1>AF-04 Release page.txt</td></tr></table>

## Supplementary Figures

![](images/505382257b79b7ee35b99a9afb0a0f3c1b03076a87d03b750da315c470a4a858.jpg)

Field-level state changes throughout the representative workflow
<table><tr><td>State</td><td>Triggering operation</td><td>Sample state</td><td>Mass/volume state</td><td>Container and lid state</td><td>Evidence appended / principal change</td></tr><tr><td> ${ \mathsf { S } } _ { 1 }$ </td><td> ${ \mathfrak { f } } _ { 1 } \colon$  mixing</td><td>Suspension of A and B</td><td>Total mass/volume</td><td>50mL tube; lid closed</td><td>Mixing completed; sample identity and provenance initialized</td></tr><tr><td> $\mathsf { S } _ { 2 }$ </td><td>stirring  $\mathfrak { f } _ { 2 } \colon$ </td><td>Homogenized suspension</td><td>Unchanged</td><td>50mL tube; lid closed</td><td>Stirring time and speed appended</td></tr><tr><td> $\mathsf { S } _ { 3 }$ </td><td> $\mathsf { f } _ { 3 } \mathsf { : }$  heating</td><td>Heated reaction mixture</td><td>Unchanged</td><td>50mL tube; lid closed</td><td>Heating temperature and duration appended</td></tr><tr><td> $\mathsf { S } _ { 4 }$ </td><td> $\mathsf { f } _ { 4 } \colon$  centrifuge</td><td>Liquid-solid separated</td><td>Unchanged</td><td>50mL tube; lid closed</td><td>Centrifugation speed and duration appended</td></tr><tr><td> ${ \mathsf S } _ { 5 }$ </td><td> ${ \sf f } _ { 5 } \colon$  open lid f6: pH</td><td>Liquid-solid mixture</td><td>Unchanged</td><td>50mL tube; lid open</td><td>Lid-state transition appended Measured pH appended as</td></tr><tr><td> ${ \mathsf S } _ { 6 }$ </td><td>measurement decant after</td><td>Liquid-solid mixture</td><td>Unchanged</td><td>50mL tube; lid open</td><td>evidence</td></tr><tr><td> $\mathsf { S } _ { 7 }$ </td><td> $\mathsf { f } _ { 7 } \colon$  pH≤7</td><td>Retained solid</td><td>Supernatant removed</td><td>50mL tube; lid open</td><td>Conditional result and decanted volume appended</td></tr><tr><td> $\mathsf { S } _ { 8 }$ </td><td> $\mathsf { f } _ { 8 } \colon$  wash</td><td>Washed suspension Washed suspension</td><td>Wash liquid added</td><td>50mL tube; lid open</td><td>Washing reagent, volume and cycle number appended</td></tr><tr><td> ${ \sf S } _ { 9 }$ </td><td> $\mathsf { f } _ { 9 } \colon$  close lid</td><td>prepared for repeated separation Final washed solid</td><td>Unchanged</td><td>50mL tube; lid closed</td><td>Lid closure and loop-back event appended</td></tr><tr><td> ${ \mathsf { S } } _ { 1 0 }$ </td><td> $\mathsf { f } _ { 1 0 } \colon$  decant after pH&gt;7</td><td>retained; supernatant removed</td><td>Supernatant removed</td><td>50mL tube; lid open</td><td>Passing decision and final decant record appended</td></tr><tr><td> $\mathsf { S } _ { 1 1 }$ </td><td> $\mathsf { f } _ { 1 1 } \colon \mathsf { d r y }$ </td><td>Dry solid</td><td>Solvent removed</td><td>50ml tube; lid open</td><td>Drying temperature and duration appended</td></tr><tr><td> $\mathsf { S } _ { 1 2 }$ </td><td> $\mathsf { f } _ { 1 2 } \colon$  weigh  $\mathsf { f } _ { 1 3 } \colon$  parallel</td><td>Dry solid aliquot</td><td>Measured mass</td><td>50ml tube; lid open</td><td>Mass measurement appended as evidence</td></tr><tr><td> $\mathsf { S } _ { 1 3 }$ </td><td>sample preparation/for  $\boldsymbol { \mathsf { k } }$ </td><td>Characterized solid</td><td>Unchanged</td><td>XRD holder and quartz cuvette</td><td>Branch identities, aliquot masses and parent-child provenance appended</td></tr><tr><td> $\mathsf { S } _ { 1 4 }$ </td><td> $\mathsf { f } _ { 1 4 } \mathrm { : }$  parallel characterizatio n and join</td><td>Characterized Chemical C; material state nominally unchanged</td><td>Unchanged except for any consumed characterization aliquot</td><td>Characterization holders; branches completed and joined</td><td>XRD and UV-Vis results appended and combined into the final evidence record</td></tr></table>

Supplementary Fig. S1 | Stateful synthesis–characterization workflow and field-level state changes. The workflow includes a protected sequence, a pH-dependent wash loop, and parallel UV-vis/XRD analysis. States S1-S14 are research-object states summarized in the table below. Superscript k denotes the wash-loop iteration index $( \mathsf { k } { = } 1 , { \ldots } , \mathsf { T } ) ; S _ { i } { } ^ { ( k ) }$ represents the research-object state at workflow node i during iteration k. States ${ \sf S } _ { 4 } { \left( k \right) } _ { - }$ $S _ { 9 } { } ^ { ( k ) }$ belong to the repeated loop body, whereas $\$ 10$ is reached after the exit condition $p H ( S _ { 6 } ( k ) { > } 7 )$ is satisfied.

Violations of mandatory operator invariants return a localized diagnostic and prevent the affected state transition from being committed; the validate-then-commit mechanism is detailed in Figure S6  
![](images/cf5474cc4e54a1b71cee24408cfc61ff61d1468b9e7f706c84e05b325737745b.jpg)

Supplementary Fig. S2 | Admissibility conditions and representative violations of the five workflow operators. Valid and rejected execution patterns are illustrated for sequence, loop, conditional, parallel, and protected-interval operators. Each operator is defined by a mandatory structural invariant; violations trigger a localized diagnostic and prevent the corresponding state transition from being committed.

b Capability families and representative operations  
![](images/b3ba88e834b53d74116c0876045dd61d39225fa09196865092a633bcfd1bad44.jpg)

<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Material / containermanagement</td><td rowspan=1 colspan=1>Store / retrieve container, transfer container</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Liquid handling</td><td rowspan=1 colspan=1>Add liquid, transfer liquid, open lid, close lid</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Solid handling</td><td rowspan=1 colspan=1>Weigh solid, transfer solid, dispense solid</td></tr><tr><td rowspan=1 colspan=1>来米</td><td rowspan=1 colspan=1>Thermal / mechanicalprocessing</td><td rowspan=1 colspan=1>Stir / mix, heat, cool, maintain temperature,ultrasonic disperser</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Separation &amp;purification</td><td rowspan=1 colspan=1>Centrifuge, decant, filter, wash, dry,evaporate, purify</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Reaction &amp; testing</td><td rowspan=1 colspan=1>Run electrochemical test, perform heterogeneouscatalytic reactions, photocatalysis, illuminate,clamp membrane, monitor reaction conditions</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Spectroscopy &amp;diffraction</td><td rowspan=1 colspan=1>Measure UV-Vis, measure FTIR, measurefluorescence, measure XRD</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Chromatography</td><td rowspan=1 colspan=1>Run GC, run LC, collect chromatogram,quantify peaks</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Imaging &amp; interfacecharacterization</td><td rowspan=1 colspan=1>Image sample, contact angle, interfacialwettability, mass-transfer characterization,high-speed imaging</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Virtual workflowcontrol</td><td rowspan=1 colspan=1>Sequence, conditional branch, loop, parallelcomposition, lock</td></tr></table>

Supplementary Fig. S3 | Hierarchical inventory of workstation specifications and capability families. (a) The 49 workstation specifications are organized into synthesis, reaction and testing, characterization, and virtual control-flow categories, with representative workstations or operators shown. (b) The corresponding capability taxonomy summarizes representative operations spanning material and container management, sample handling, processing, separation, reaction, characterization, and workflow control.

## a Assessment logic of one natural-language workstation SKILL.md document

b Rubric items defined in Supplementary Table S6  
![](images/1ee60c012d8cd44e201d38197f0bd41eba0bf832ac48b319c88767b2fe6b30b5.jpg)

<table><tr><td rowspan=1 colspan=3>Required checks (60 points)</td></tr><tr><td rowspan=1 colspan=1>R1</td><td rowspan=1 colspan=1>Workstation name/title</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>R2</td><td rowspan=1 colspan=1>Operation description</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>R3</td><td rowspan=1 colspan=1>Parameter information</td><td rowspan=1 colspan=1>15</td></tr><tr><td rowspan=1 colspan=1>R4</td><td rowspan=1 colspan=1>Initial state</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>R5</td><td rowspan=1 colspan=1>State update</td><td rowspan=1 colspan=1>10</td></tr><tr><td rowspan=1 colspan=1>R6</td><td rowspan=1 colspan=1>Constraint rules</td><td rowspan=1 colspan=1>5</td></tr></table>

<table><tr><td rowspan=1 colspan=3>Optional checks (40 points)</td></tr><tr><td rowspan=1 colspan=1>01</td><td rowspan=1 colspan=1>Frontmatter completeness</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>02</td><td rowspan=1 colspan=1>Similar-workstationdistinction</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>03</td><td rowspan=1 colspan=1>Overall workflow constraints</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>04</td><td rowspan=1 colspan=1>Parameter-constraint format</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>05</td><td rowspan=1 colspan=1>Nested-parameter expansion</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>06</td><td rowspan=1 colspan=1>File parameters</td><td rowspan=1 colspan=1>3</td></tr><tr><td rowspan=1 colspan=1>07</td><td rowspan=1 colspan=1>Return values</td><td rowspan=1 colspan=1>5</td></tr><tr><td rowspan=1 colspan=1>08</td><td rowspan=1 colspan=1>State-transitioncompleteness</td><td rowspan=1 colspan=1>4</td></tr><tr><td rowspan=1 colspan=1>09</td><td rowspan=1 colspan=1>Example completeness</td><td rowspan=1 colspan=1>3</td></tr></table>

The quality score measures document completeness /conversion readiness; it is not an experimental performance metric.

Supplementary Fig. S4 | Natural-language Skill quality assessment and admission logic. (a) Workflow for structural checks, LLM-based semantic assessment, special-case handling, score aggregation, threshold-based admission, and assessment recording. (b) Required and optional rubric items used for Skill quality scoring, with a total score of 100 points.

Release evidence: all four validation reports supporting Extended Data Fig. 1 record O2 structural analysis as passed  
![](images/269875f73f56d963db578e1e3e64bbfafb673120b3414def1187e80919185b2f.jpg)  
O3 and O4 were disabled; advisory merge suggestions were not applied

Supplementary Fig. S5 | Static analysis and conservative optimization of generated Python recipes. The workflow covers security admission, normalization, conservative AST simplification, and optional domain-aware optimization, with rejected recipes returned for regeneration and accepted recipes serialized for simulation and review. O0– O2 are enabled, whereas O3–O4 are disabled in the archived configuration. The workflow and configuration are aligned with Supplementary Note 6 and Supplementary Table S7.

![](images/e29d8bd0bf1c83304547161632859f479e7e89d4ca65eb6beefd01c93b383353.jpg)  
Workflow-level acceptance: no MUST errors remain; SHOULD warnings are reported separately  
Supplementary Fig. S6 | Validate-then-commit stateful simulation and structured diagnostic reporting. Following static recipe admission, each operation is dynamically evaluated against all applicable MUST and SHOULD predicates before any state mutation. Operations satisfying all MUST constraints are committed, whereas failed MUST constraints trigger a no-commit path that preserves the current state and records structured diagnostics in ErrorEntry; SHOULD violations are retained as non-blocking warnings. Here, SimState<sub>t</sub> is the bounded executable projection of the global laboratory state $\mathcal { L } _ { t } .$ The schematic illustrates the operation-level validation logic and contains no case-specific values.

## Fe coordination environment and alkaline OER workflow

## a. Scientific intent and comparison design

Objective: Design an experiment to investigate how the existing form and coordination environment of Fe modulate the alkaline oxygen evolution reaction (OER) performance of NiFe-based catalysts. NiFe-based catalytic materials with different Fe introduction methods and distinct Fe coordination environments are required to be fabricated.

<table><tr><td>Fe-free control</td><td>Co-precipitated Fe</td><td>Post-impregnated Fe</td><td>Citrate-mediated Fe</td><td>EDTA-mediated Fe</td></tr><tr><td>Fe-free reference</td><td>Lattice-associated Fe-O-Ni</td><td>Surface-enriched</td><td>Ligand-modulated</td><td>Stronger chelation-</td></tr><tr><td></td><td>hypothesis</td><td>FeOx/FeOOH-like hypothesis</td><td>hypothesis and incorporation</td><td>modulated Fe availability</td></tr></table>

External coordination confirmation: Fe K-edge XANES/EXAFS • XRD • ICP-OES • optional 57Fe Mossbauer spectroscopy.

## b. Workstation architecture

![](images/0e403e6f75e3283b7620bbcdb0f469967afb7adbe715c925a687943eb7f95190.jpg)

Five matched formulation routes are executed as a linear sequence because the same sample-vial container class is reused. Working coordination motifs remain hypotheses until confirmed by the external measurements listed above

c. Validation and archived record
<table><tr><td>Optimizer</td><td>02</td><td>Optimization</td><td>Passed</td><td>Simulation</td><td>Passed</td><td>Simulated steps</td><td>27</td><td>Physical steps</td><td>27</td></tr><tr><td>MUST errors 0</td><td></td><td>SHOULD warnings 0</td><td></td><td>Independent batches 3</td><td></td><td>Minimum OER measurement 15</td><td></td><td></td><td></td></tr></table>

Archived artifacts: fe\_coordination\_environment\_alkaline\_oer.py • experiment\_plan.json • experiment\_design\_manifest.json • validation\_report.json • workflow\_generation\_response.json

Supplementary Fig. S7 | Full Fe coordination-environment comparison and alkaline OER workflow. The figure summarizes the experimental design used to investigate how the existing form and coordination environment of Fe influence the alkaline OER performance of NiFe-based catalysts. Five Fe-introduction routes are compared under matched synthesis and OER conditions, followed by a sequential workflow for preparation, characterization, and electrochemical testing. The archived validation results confirm successful workflow generation and simulation, while route-specific Fe coordination motifs are treated as working hypotheses requiring independent experimental verification.

Join and integrated evaluation: Activity • selectivity • wettability • phase stability

## Full LDH-HMF synthesis, electrochemistry and multimodal analysis workflow a. Scientific intent

Objective: Design an integrated synthesis, electrochemical testing and multimodal characterization experiment for LDH catalysts toward alkaline 5-hydroxymethylfurfural (HMF) oxidation. The catalyst preparation and electrode-fabrication steps must follow a timelocked sequential workflow. After electrochemical testing, the solid electrode and liquid reaction products are required to undergo parallel contact-angle measurement, XRD analysis and liquid-phase product analysis, followed by integrated evaluation of the results.

![](images/4309d3d36f509ae5c083d8e67c8328138babed666d74b3907b9e4a3e1b3ede3f.jpg)  
Archived artifacts: recipe\_ldh\_hmf\_oxidation.py · recipe\_ldh\_hmf\_oxidation.json ·integrated\_evaluation.json • validation\_report.json • workflow\_generation\_response.json

Supplementary Fig. S8 | Integrated synthesis, electrochemical testing, and multimodal characterization workflow for LDH catalysts toward alkaline HMF oxidation. The workflow combines a time-locked catalyst preparation sequence with sequential electrochemical evaluation, followed by parallel contact-angle, XRD, and liquid-phase product analyses. Results from all branches are subsequently joined for integrated evaluation, with archived validation confirming successful workflow generation and simulation.

## Furfural hydrogenation and bounded GC optimization workflow

## a. Scientific intent and analytical requirements

Objective: Design an automated catalyst synthesis and closed-loop analytical optimization experiment for the electrocatalytic hydrogenation of furfural to furfuryl alcohol. After catalyst preparation, a closed-loop algorithm is reguired to iteratively optimize the gas-chromatographic separation of furfural, furfuryl alcohol and major liguid by-products until predefined peak-resolution criteria are satisfied. The optimized method will then be applied to quantify the reaction mixture and evaluate furfural conversion, furfuryl alcohol yield and product selectivity. Need to use a loop node.

Monitored analytes: furfural • furfuryl alcohol • tetrahydrofurfuryl alcohol • cyclopentanone • cyclopentanol

All pass criteria: Rs (target pair) ≥ 1.50 • Rs (nearest major by product) ≥ 1.50 • max tailing ≤ 2.00 • retention time RSD ≤ 1.00%.

![](images/3d35cd3a2233cb81ba46c06bdc97a4629dc19ff62423ad9907ef33fa5c40dad0.jpg)

<table><tr><td colspan="10">c. Validation and archived record</td></tr><tr><td>Optimizer</td><td>02</td><td>Admission Passed</td><td>Simulation</td><td>Passed</td><td>Simulated steps</td><td>23</td><td>Physical steps</td><td>20</td></tr><tr><td>MUST errors 0</td><td></td><td>SHOULD warnings 0</td><td>Loop verified Yes</td><td></td><td>Loop bound Nmax</td><td>6</td><td>Workflow generation</td><td>Passed</td></tr></table>

Archived artifacts: recipe\_furfural\_furfuryl\_alcohol\_gc\_loop.py • recipe\_furfural\_furfuryl\_alcohol\_gc\_loop.json • experiment\_design\_spec.json • validation\_report.json • workflow\_generation\_response.json

Supplementary Fig. S9 | Automated furfural hydrogenation and closed-loop GC optimization workflow. The workflow integrates automated Cu-catalyst preparation, duplicate constant-potential electrocatalytic hydrogenation, and iterative GC method optimization for furfural, furfuryl alcohol, and major liquid by-products. GC conditions are iteratively refined until predefined separation criteria are satisfied or the maximum iteration limit is reached, after which the accepted method is used for quantitative analysis of conversion, yield, selectivity, and carbon balance. Archived validation results confirm successful workflow generation and simulation.

<table><tr><td>FT-IR branch</td><td>UV-vis branch</td><td>Absorbance gate</td><td>Loop bound</td></tr><tr><td>Independent characterization</td><td>Adaptive dilution loop</td><td>0.20-1.00 AU</td><td>Maximum 3 iterations</td></tr></table>

Archived artifacts: recipe\_ZnO\_nanoparticle\_adaptive\_spectroscopy.py • recipe\_ZnO\_nanoparticle\_adaptive\_spectroscopy.json • workflow\_request\_ZnO\_nanoparticle\_adaptive\_spectroscopy.json·validation\_report.json · workflow\_response.json

## ZnO synthesis and adaptive parallel spectroscopy workflow

a. Scientific intent and adaptive measurement requirement

Objective: Design an automated synthesis and adaptive spectroscopic characterization experiment for ZnO nanoparticles. Following synthesis, centrifugal purification, drying and redispersion, the samples are required to undergo parallel FTIR and UV–vis spectroscopic characterization. In the UV–vis branch, a conditional dilution-and-remeasurement loop must be implemented until the measured absorbance falls within the predefined quantitative range.

## b. Workstation architecture

![](images/e10abf377b69f74bebadbfd555f935a927a887800e1ef68fd56d69934f54a985.jpg)

## c. Validation and archived record

<table><tr><td rowspan="2">Optimizer</td><td rowspan="2">02</td><td rowspan="2">Admission</td><td rowspan="2">Passed</td><td rowspan="2">Simulation</td><td rowspan="2">Passed</td><td rowspan="2">Executed steps</td><td rowspan="2">30</td><td rowspan="2">Physical steps</td><td rowspan="2">27</td></tr><tr><td></td></tr><tr><td>MUST errors 0</td><td></td><td>SHOULD warnings 0</td><td></td><td>Workstation</td><td>15</td><td>Parallel branches</td><td>2</td><td>Conditional switches 2</td><td></td></tr></table>

Supplementary Fig. S10 | Automated ZnO synthesis and adaptive parallel spectroscopy workflow. The workflow integrates ZnO nanoparticle synthesis, centrifugation, drying, and redispersion with parallel FTIR and UV–Vis characterization. The UV–Vis branch incorporates a bounded dilution-and-remeasurement loop until the predefined quantitative absorbance range is reached or the iteration limit is met. Archived validation confirms successful workflow generation and simulation.

## SI Guide entries

Supplementary Information

Supplementary Information. Supplementary Notes 1–12, Supplementary Tables S1–S12 and Supplementary Figs. S1–S10 supporting the formal representation, Function-Skill implementation, stateful simulation and four workflow cases.

## Supplementary Data

Supplementary Data 1. Capability and operation registry. Machine-readable inventory of the final workstation specifications, Function-Skill operations, accepted states, arguments, constraints, state changes, returns, source provenance, versions and checksums. File: Supplementary\_Data/Supplementary\_Data\_1.xlsx

Supplementary Data 2. Formal grammar and canonical workflows. JSON definitions of the workflow grammar and canonical AST/JSON records for Fig. 1d and the four workflows in Extended Data Fig. 1a–d, with node-to-operation maps. File: Supplementary\_Data/Supplementary\_Data\_2.json

Supplementary Data 3. Four-case evidence record. Exact prompts, recipe versions, workflow metrics, simulation results, structured diagnostics for Fe-coordination/OER, LDH–HMF, furfural and ZnO workflows. File: Supplementary\_Data/Supplementary\_Data\_3.xlsx

Supplementary Data 4. Reproducibility manifest. Software, model and configuration versions, commands, environment information, filenames, SHA-256 checksums and links to author-supplied release artifacts. File: Supplementary\_Data/Supplementary\_Data\_4.xlsx

## Author-supplied Additional Files

Additional File 1. Source-code archive. Final optimizer, simulator, converter, configuration, dependency and reproduction files.

Additional File 2. Function-Skill library. Frozen 49-workstation capability library.

Additional File 3. Extended Data Fig. 1 workflow archive. Exact prompts, archived recipes, canonical JSON plans and validation reports for the four cases; physicalexecution logs are not included. File: AF-03 Extended Data Fig. 1 workflow archive.zip (legacy filename retained).

Additional File 4. Release-page record. Author-supplied internal release-address record. File: AF-04 Release page.txt. T=33he recorded route is not a verified persistent public or reviewer-access repository.