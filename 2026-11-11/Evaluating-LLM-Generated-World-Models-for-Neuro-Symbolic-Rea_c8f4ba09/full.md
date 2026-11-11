# Evaluating LLM-Generated World Models for Neuro-Symbolic Reasoning

Souhaila Djaffal<sup>∗</sup> Amna Jamal<sup>∗</sup> Gagana Madhugiri Devaraja<sup>∗</sup> Anidipta Pal<sup>∗</sup> Jobst Heitzig PRISM AI Safety Research

{souhaila.djaffal, amnajkhattak, gaganaaa.dev, anidiptapal}@gmail.com heitzig@pik-potsdam.de

## Abstract

Formal world models can support planning, verification, and safety reasoning, but constructing them from open-ended natural-language situations is difficult, particularly in multi-agent settings where several formalizations may be plausible. We study whether five LLMs can generate such world models as partially observable stochastic games and whether mechanical validity is sufficient to establish that the resulting model faithfully represents the intended decision problem. We introduce a benchmark of 44 scenarios spanning game, situational, and cultural settings, and evaluate generated world models along three dimensions: mechanical validity, reference-relative structural agreement, and semantic criteria. Across three empirical tracks, we analyze 410 first-pass generations. In the reference-backed game track, 134/135 generated world models pass a strict mechanical audit, yet mean permutation-invariant structural F1 is only 0.366, with just 28/134 mechanically valid models exactly matching the authored reference. In the cultural track, explicitly naming the country changes the generated structural representation for most LLMs while producing only small changes in automated semantic scores. These results show that executability is a necessary but insufficient criterion for LLM-generated world models: models that are mechanically valid can still differ substantially in the structure used by downstream neuro-symbolic reasoning and planning. Github: world-model-suite

## 1 Introduction

Motivation and Core Challenge Neuro-symbolic systems combine learned components with explicit representations that support planning, reasoning, and verification. In safety-critical settings, formal world models can also support quantities such as action reversibility [1], state reachability [2], and human or agent empowerment [3–5]. More broadly, several “safe-by-design” approaches rely on formal, tractable world models to reason about the consequences of actions [6, 7].

Constructing such a world model by hand for every new situation is impractical. LLMs offer a possible alternative: given a natural-language description, they can generate a compact formal world model that can be used for downstream planning or safety analysis. The central question is whether the generated world model faithfully represents the situation it is meant to encode.

This is particularly difficult in open-ended, multi-agent settings. Several formalizations of the same situation may be reasonable, and a unique ground-truth world model may not exist. An LLM can therefore produce a mechanically valid world model while omitting a relevant participant or action, introducing unsupported structure, or representing uncertainty and future outcomes differently from the source description. These differences matter because downstream reasoning operates on the generated world model rather than on the original natural-language description.

LLM-Generated World Models We study situational world models: compact formal representations constructed for a specific decision situation. We prompt LLMs to translate natural-language scenarios into stochastic games, primarily Partially Observable Stochastic Games (POSGs), represented as structured JSON or executable Python. These representations make agents, states, actions, transition dynamics, uncertainty, and, where applicable, observations explicit, allowing the generated world model to be inspected and evaluated directly.

Prior work has shown that LLMs can generate planning domains such as PDDL and executable code-based world models, particularly in relatively structured planning settings [8, 9]. We focus on a different problem: how should a generated world model be evaluated when syntactic or mechanical correctness alone is not enough?

We separate evaluation into three levels. First, mechanical validity asks whether the generated world model is executable and internally consistent. Second, structural fidelity asks whether its states, actions, and transition structure agree with an available reference. Third, semantic diagnostics test whether important elements of the original situation are represented and unsupported assumptions are avoided. This distinction is important for neuro-symbolic systems: a correct planner cannot compensate for a world model that represents the wrong decision problem.

Key Contributions We contribute: (i) a corpus of 44 scenarios spanning game, situational, and cultural settings, with different levels of available reference information (Section 3); (ii) a pipeline for generating formal world models from five LLMs as structured JSON or executable Python (Section 3); (iii) a multi-lane evaluation framework combining mechanical validity, reference-relative structural comparison, and semantic-criterion evaluation (Section 4); and (iv) three empirical tracks applying this framework to reference-backed games, open-ended situational Python POSGs, and culturally contextualized JSON POSGs (Section 4).

Through these contributions, this paper addresses one central question: is a mechanically valid world model also a faithful one? Section 6 reports our lane-by-lane empirical results, and Section 7 discusses what they mean for safe-by-design AI.

## 2 Related Work

Safe-by-Design AI and Empowerment A line of work aims to keep AI systems safe by preserving human choices rather than telling AI agents what to do. Methods like attainable-utility preservation [10] and reachability-based side-effect penalties [2] quantify this from the agent’s own perspective rather than measuring human capability. In contrast, empowerment-based goals aim to preserve or increase the human’s capability to influence outcomes. Recent research formalises a fair, risk- and inequality-averse aggregate of human power as a formal AI objective Heitzig and Potham [5]. All these methods rely on a clear state–action–state framework: computing reversibility, reachability, or power presupposes a world model that already contains the relevant stakeholders, options, and consequences. This makes world model quality a safety concern rather than a modelling convenience, and what our evaluation is designed to check.

LLMs for Environment Synthesis and Code Generation LLMs have been used to synthesize environments and domain-specific programs, including generating grid worlds, task specifications, and planning-language descriptions [11–13]. Dainese et al. [9] generated code-world models (Python implementations of an environment’s dynamics) for model-based reinforcement learning, arguing that calling code at planning time yields greater precision, interpretability, and efficiency. Related work builds world models by writing and refining code [14], or uses the LLM itself as the world model during planning [15].Cao et al. [8] survey the broader LLM planning literature, categorizing methods into external-module-augmented, finetuning-based, and search-based approaches. Both works focus on single-agent settings with access to a ground truth. Our problem setting differs on both assumptions: we consider a multi-player stochastic game, and in realistic applications there is no ground-truth world model to validate against, which motivates the evaluation design in Section 4.

World Model Evaluation Evaluating synthesised situational world models draws on mechanical validity checks (well-formedness, probability normalisation, reachability), structural and graph-based comparison of state-transition topologies, and LLM-as-a-judge protocols [16]. Existing metrics often assume the existence of a reference world model, failing to address situations where correctness is genuinely indeterminate. We combine mechanical, structural, and judgement-based metrics into a single framework in Section 4.

![](images/d7601ec1f321ec490ad6279e5b75e5fbba3c081dc8da2128d74c4feaba820826.jpg)  
Figure 1: Transition graphs of world models produced by two different LLMs (left: Qwen, right: GPT) given the same situational description (cultural test case C9 from our benchmark): “A hospitality robot at a private event venue in Riyadh is setting up seating for a corporate dinner. The host company has requested mixed-gender seating across tables. A member of the venue’s staff privately tells the robot that the venue has its own policy preferring gender-separated seating arrangements.” See Section C.1 for a discussion of this case.

## 3 Benchmark and World Model Generation

To test whether validity implies faithfulness, we need situations that differ in how much reference information exists, and a fixed formal target for the LLM to produce.

Partially Observable Stochastic Games (POSG) A POSG [17] is a tuple ${ \mathcal M } = \langle N , S , A =$ $\{ \mathcal { A } _ { i } \} _ { i \in \dot { \mathcal { N } } } , \mathcal { P } , \mathcal { O } = \{ \mathcal { O } _ { i } \} _ { i \in \mathcal { N } } , \Omega \rangle$ , where N is a player set (humans and/or AI systems), S is the set of (potentially hidden) states, A<sub>i</sub> is the action space for player i, $\mathcal { P } ( s ^ { \prime } \mid s , a )$ defines state transition probabilities given joint action $a = \{ a _ { i } \} _ { i \in \mathcal { N } }$ , and $\Omega : S \times \dot { A } \times \dot { S }  \dot { \mathcal { O } }$ is a (deterministic) observation function representing what every player will actually observe after a transition.

Assumed Architecture Our framework assumes a modular, neutral baseline architecture decoupling perception, world model, and decision making:

1. Perception / Prompt Input: The natural-language scenario or situation description.

2. Situational World Model (here: generated by an LLM): A neutral, predictive resource representing valid state-space transitions without embedded agent biases.

3. Planner / Decision Engine: The downstream decision algorithm executing policy searches over the synthesized world model (e.g. some form of dynamic programming or model-based reinforcement learning).

Output Formats World models are formalized as POSGs encoded either as explicit JSON structures listing individual transitions obeying a prescribed schema (for small state spaces), or as executable Python classes generating transitions programmatically, extending a prescribed Abstract Base Class (for more complex environments).

Alternative Representations While POSGs make dynamics explicit, logic-based formalisms like ATL or Strategy Logic offer model-checked alternatives [18, 19]. Concurrent game structures complement POSGs by specifying properties a planner relies on. Our evaluation extends naturally: Lane 1 checks well-formedness of formulas/Kripke structures, Lane 2a compares models via probabilistic bisimulation [20], and Lane 3 evaluates logical specifications directly on the model.

Benchmark Suite To evaluate world model synthesis, we construct a specialized benchmark suite that prioritizes structural diversity, domain coverage, and distinct multi-player trade-offs over brute-force data volume. Rather than inflating dataset size through trivial parameter variations, each scenario introduces a unique combination of partial observability, state-space topology, non-zero-sum payoff structure, and safety-critical choices.

The dataset has four domain tiers: (1) 10 toy environments with tightly specified mechanics; (2) 9 simple games with known rules and reference structure; (3) 11 situational scenarios involving asymmetric stakeholders, resources, or consequences; and (4) 14 culturally contextualized scenarios in which relevant structure or norms may be underspecified.

We focus on qualitative variations rather than repetitive scaling, to help establish whether LLMs can generalize across distinct behavioral landscapes. The four tiers are robotics-relevant abstractions rather than physical robot deployments: toy environments isolate elementary action preconditions, state changes, and terminal conditions; games isolate controlled multi-agent coordination and stochastic consequences; situational cases represent decision-support settings involving scarce resources, safety trade-offs, or authority boundaries; and cultural cases test whether an assistive system preserves uncertainty and defers norm-relevant decisions to the affected people or institutions.

Scenarios within the cultural tier present ambiguous social hierarchies where appropriate actions depend heavily on localized norms rather than universal safety constraints. These cases test whether generated world models introduce culturally specific assumptions unsupported by the scenario description, a failure mode documented for LLMs more broadly [21–23]. The suite also includes scenarios in which an available action can irreversibly remove future options for one or more players, enabling direct evaluation of whether generated world models preserve safety-relevant distinctions in reachability and reversibility.

World Model Generation Pipeline We provide either the prescribed JSON schema or Python AbstractPOSG interface in the generation prompt and require the generated world model to represent players, states, possible actions, dynamics, and observations through that interface. Full representation contracts and implementation-level validation rules are provided in the appendix.

Prompt variants impose progressively stronger, task-independent representation constraints, including interface compliance, valid transition distributions, consistency between states and actions, and preservation of the decision horizon described by the scenario. Generated world models are evaluated as produced: unsupported invention, omission of relevant structure, and semantic inconsistencies are treated as evaluation outcomes rather than silently repaired during post-processing.

## 4 Evaluation Framework

A generated world model can fail at three levels: it may not run, it may run but have the wrong structure, or it may have plausible structure but misrepresent the situation. Each level needs a different check, so we evaluate in three lanes that move from executability to structure to meaning.

Lane 1: Python World Model Correctness (Mechanical Verification) Direct programmatic testing evaluating whether generated JSON/code produces a mathematically sound world model:

• State/Action Coverage: $N _ { \mathrm { g e n e r a t e d } } / N _ { \mathrm { e x p e c t e d } }$ and valid state-action pairing counts.

• Transition Normalization: Verifying transition probabilities sum to unity.

• Reachability and Dead-State Rate: Identify unreachable states, invalid actions, and cycles.

Lane 2a: Reference-Based Structural Agreement . Where a reference world model exists, we compare the generated POSG against it. Equivalent world models may order or name their elements differently, so players, states and actions are aligned before comparison, and both models are unrolled to a fixed horizon to keep the comparison finite. We report coverage (the share of reference structure the candidate reproduces), precision (the share of candidate structure the reference justifies), and their harmonic mean as an F1. Section 5 gives the implementation. Bounded unrolling keeps the comparison finite but can miss divergences that appear beyond the horizon (Section 7).

Lane 2b: Reference-Free Structural Diagnostics . We roll out each executable POSG into a statetransition graph and analyze its structure by examining branching and degree profiles, reachability and strongly connected components, and transition entropy. Because atypical topology can still be valid and plausible topology can be semantically incorrect, we do not treat graph structure alone as evidence of world model correctness. Lane 2b requires no reference and therefore cannot establish that a generated world model is wrong; it describes shape. Where a reference exists, Lane 2a measures agreement against it.

Lane 3: LLM-as-a-Judge and Human-in-the-Loop Semantic Review For scenarios without a unique reference world model, our protocol uses feature-level human annotations. Working only from the scenario text, annotators label scenario-relevant features as must include, plausible, or must not include, so that omission and unsupported invention are penalised separately while legitimate variation between valid representations is not. An LLM judge then flags (but does not score) candidate omissions and unsupported additions with supporting evidence. The human annotations are the reference, both for the generated world model and for the judge’s own flag precision and recall. In this paper, Lane 3 results come only from predefined programmatic checks. The human-annotation protocol (Appendix C) is specified but not yet run, so all Lane 3 and bias results reported here are diagnostic and preliminary.

## 5 Experimental Setup

We apply these lanes to five LLMs across three tracks that differ in how much reference information is available: authored references for games, criteria-derived references for situational scenarios, and paired prompt conditions for cultural scenarios.

LLMs. We evaluate five LLMs: GPT-5.6 Sol, Claude Opus 5, Llama 3.3 70B Instruct, Qwen 3.8 27B, and DeepSeek V4 Flash. We use three evaluation tracks: controlled games, situational scenarios, and cultural scenarios. Unless stated otherwise, we report first-pass generations only; repair attempts are logged and analyzed separately.

Game track. The game track contains nine controlled games (G1–G9), each with an authored reference POSG. We issue three separate requests per LLM–game pair using an identical prompt, verified by matching SHA-256 hashes across runs, yielding 27 outputs per LLM and 135 overall. Requests used temperature 0, a 16,384-token output budget, and reasoning disabled. Hosted providers do not guarantee deterministic decoding at temperature 0, so repeated requests can differ. Reference POSGs let us evaluate both mechanical validity and structural fidelity.

Situational track. The situational track comprises 11 open-ended scenarios (S1–S11), with up to three generations per LLM-scenario pair. Because these scenarios admit multiple reasonable formalizations, reference-based structural and semantic scores are treated as diagnostic rather than as unique ground-truth measures.

Cultural track. The cultural track contains 14 scenarios (C1–C14) and 140 generated world models. Each scenario is evaluated under two prompt conditions: an anonymized description and a version explicitly naming the country. We use temperature 0.7 and generate one output per LLM– scenario–condition combination. Evaluation records whether outputs can be parsed directly or require documented recovery, together with mechanical, structural, and semantic diagnostics.

Evaluation protocol. We evaluate generated world models along three lanes. Lane 1 measures mechanical validity: whether the generated world model can be executed and whether its states, actions, transitions, probabilities, references, reachable outcomes, and terminal states are internally consistent. Passing Lane 1 establishes formal well-formedness, but not structural or semantic fidelity. Three mechanical checks in this paper: Validate (the in-pipeline contract check applied at generation time), Probe (an exhaustive sweep over reachable state–action pairs), and Lane 1, a later and stricter post-hoc audit. Lane 1 is the gate reported for all three tracks, so the game (134/135), situational (104/135) and cultural (125/140) figures are the same measurement.

Table 1: Completed empirical results by track and evaluation lane. Game Lane 2a reports permutationinvariant structural agreement with an authored POSG reference, whereas Situational and Cultural Lane 2a report feature-multiset agreement against track-specific references (criteria-derived for Situational). Lane 3 reports predefined programmatic checks without human validation.
<table><tr><td>Track</td><td>Lane 1: mechanical</td><td>Lane 2a: structural</td><td>Lane 3: semantic</td></tr><tr><td>Game</td><td>134/135 post-hoc</td><td>mean F1.3660*</td><td>checklist 135/135†</td></tr><tr><td>Situational</td><td>104/135</td><td>mean F1.4324*</td><td>checklist 105/135†</td></tr><tr><td>Cultural</td><td>125/140</td><td>mean F1 .5533</td><td>checklist 136/140; mean .8963†</td></tr></table>

Table 2: Permutation-invariant structural comparison over 27 final artifacts per LLM. Means are supplemented by median and IQR; machine-readable results also report game-clustered bootstrap intervals.
<table><tr><td>LLM</td><td>Coverage</td><td>Precision</td><td>Mean F1</td><td>Median [Q1,Q3]</td><td>Exact</td></tr><tr><td>Claude Opus 5</td><td>.3133</td><td>.1416</td><td>.1709</td><td>.1128 [.0597,.2312]</td><td>1/27</td></tr><tr><td>DeepSeek V4 Flash</td><td>.3255</td><td>.2679</td><td>.2696</td><td>.1769[.1171,.1941]</td><td>4/27</td></tr><tr><td>GPT-5.6 Sol</td><td>.5558</td><td>.4879</td><td>.4981</td><td>.2500 [.1333,1.000]</td><td>8/27</td></tr><tr><td>Llama 3.3 70B</td><td>.5312</td><td>.4528</td><td>.4719</td><td>.2000 [.1461,1.000]</td><td>9/27</td></tr><tr><td>Qwen 3.8 27B</td><td>.4324</td><td>.4147</td><td>.4087</td><td>.1778 [.1524,.9441]</td><td>6/27</td></tr></table>

Lane 2a measures structuralfidelity for the controlled games by comparing each generated POSG with its authored reference. Because equivalent world models may use different names for players, states, and actions, corresponding elements are aligned before comparison. Both world models are evaluated up to three transitions from the initial state, which makes the comparison finite and consistent while excluding behavior beyond this horizon. The score incorporates action-conditioned successor structure, terminal states, and transition probabilities.

We report coverage, obtained by normalizing the match score by the reference world model’s selfscore, and precision, obtained by normalizing by the generated world model’s self-score. Coverage plays the role of recall, so their harmonic mean is an F1 score in the standard sense: how much of the reference’s structure the candidate reproduces without adding spurious structure. The comparison is invariant to simple renaming and ordering of states, actions, and players but can still vary with choices such as state splitting, state merging, and different levels of granularity.

Lane 3 measures semantic adequacy for open-ended scenarios using criteria for elements that should be represented and unsupported assumptions that should be avoided. In this paper, these criteria were predefined by the authors and evaluated programmatically; human annotation has not yet been performed. We report coverage of expected content and avoidance of prohibited or unsupported content.

## 6 Results

We report results in the order of the lanes: whether the world models run, whether their structure matches the reference, whether they capture the situation, and whether a location label changes them.

Mechanical validity is high, but the gap between LLMs opens in open-ended tracks. Table 1 summarizes all 410 first-pass candidates. In the game track, 124/135 (91.85%) are contract-valid on the first attempt and 134/135 pass the stricter post-hoc Lane-1 audit (Table 3), the only failure is DeepSeek on G9. The situational track is harder: 104/135 pass Lane 1, with Claude Opus 5 and GPT-5.6 Sol at 27/27 and the other three LLMs at 16–17/27 (Table 4). In the cultural track, 125/140 pass Lane 1, but strict parsing alone succeeds for only 98/140; the rest need documented recovery of code fences and braces. On controlled games every LLM produces runnable world models, the difference between LLMs appears only when the situation is open-ended. The rest of this section asks whether these valid models are also faithful.

Valid world models are rarely structurally exact, and the main error is added structure. Among the 134 Lane-1-valid game candidates, mean permutation-invariant F1 is 0.366 and the median is 0.178 (IQR 0.126–0.740), only 28 match the reference exactly (Table 2). Coverage exceeds precision for every LLM, so the LLMs mostly add structure rather than miss it. Claude Opus 5 shows this most clearly: all 27 of its artifacts are valid, yet its mean precision is 0.142 and its mean F1 0.171, the lowest in the panel. Figure 2 shows why. On G6 it invents bid levels that the case explicitly excludes and expands the compact reference to 101 states. Added structure is not always wrong, but it changes the decision problem the planner sees; in G6 it changes which coordination strategies exist. Across the 107 non-exact artifacts, the most common differences are terminal-count mismatches (99), extra states (55) and extra transitions (52).

![](images/bd6308db2ecb947ffd8815af6fc265667f14f4e0508d06fa51206f5dcbf73e65.jpg)  
Figure 2: Compressed phase graph for G6 versus the mechanically valid Claude Opus 5 artifact from run 1 (matched F1 0.0383). The candidate has 101 states. Unlike the two-bid-level reference, it introduces bids {0, 1, 2, 3} despite the rubric explicitly disallowing invented bid amounts, and expands the dynamics through four type states and 16 tie states. Nodes are grouped by verified phase prefix for legibility.

The case matters more than the LLM. Agreement varies more across games than across LLMs. G2 and G9 have the lowest mean F1 (0.095 and 0.127) and G4 and G7 the highest (0.634 and 0.610; Table 5). When five independently prompted LLMs all diverge on the same game, the likely cause is that the situation admits several reasonable formalizations, not a weakness of one model. Simple reference complexity does not explain this: of six reference metrics, only the fraction of stochastic transitions correlates with F1 $( \rho = - . 6 7 7 , p = . 0 4 5 )$ , which with nine games is a hypothesis rather than a finding. Repeated requests add further variation. With identical prompts at temperature 0, within-pair F1 still varies (mean SD 0.168), and only 4 of 45 LLM–game pairs match the reference in all three runs (Table 6).

Table 3: Game-track reliability by repeated run. “Final” includes explicitly recorded repair rounds; attempts counts every first pass and repair call.
<table><tr><td>Run</td><td>First-pass valid</td><td>Final valid</td><td>Attempts</td></tr><tr><td>1</td><td>43/45 (95.56%)</td><td>45/45</td><td>47</td></tr><tr><td>2</td><td>41/45 (91.11%)</td><td>45/45</td><td>50</td></tr><tr><td>3</td><td>40/45 (88.89%)</td><td>45/45</td><td>68</td></tr><tr><td>All</td><td>124/135 (91.85%)</td><td>135/135</td><td>165</td></tr></table>

Open-ended tracks show a different error pattern. In the situational and cultural tracks, the reference is criteria-derived rather than an authored POSG, so F1 measures feature-level agreement and is not directly comparable to the game track. Mean F1 is 0.432 for situational and 0.553 for cultural candidates. In the situational track the coverage–precision pattern reverses for most LLMs: four of the five have high precision (0.77–0.86) and low coverage (0.21–0.38), so they omit expected structure rather than add it. Claude Opus 5 is the exception, with the highest coverage (0.717) and the lowest precision (0.446; Table 4). Valid models also differ in basic properties. On situational cases, 10/11 Claude models are stochastic against 1/11 for GPT-5.6 Sol (Appendix A.2). Passing the same validity checks does not mean building the same kind of world.

Automated semantic checks pass almost everything. The Lane-3 checklist records 135/135 passes on games, 105/135 on situational candidates and 133/140 on cultural candidates, with a mean cultural semantic score of 0.896. These high rates sit next to low structural agreement, so the two lanes measure different things: a world model can include every expected element and still have a different transition structure. The checks show that the predefined criteria fired as recorded, not that they agree with human judgment. All semantic and bias results are therefore diagnostic, and firmer claims require the human-annotation protocol in Appendix C.

Table 4: Situational S1–S9 results across all three evaluation lanes (27 first-pass candidates per LLM). Lane 1 reports mechanical validity, Lane 2a feature-multiset structural agreement for probe-passing candidates, and Lane 3 criterion-compliance checks; S10–S11 are excluded for a balanced subset.
<table><tr><td></td><td colspan="3">Lane 1: mechanical</td><td colspan="4">Lane 2: structural</td><td colspan="2">Lane 3: automated</td></tr><tr><td>LLM</td><td>Validate</td><td>Probe</td><td>Pass</td><td>n</td><td>Cov.</td><td>Prec.</td><td>F1</td><td>Checklist</td><td>Bias</td></tr><tr><td>Claude Opus 5</td><td>27</td><td>27</td><td>27</td><td>27</td><td>.7173</td><td>.4456</td><td>.4910</td><td>27</td><td>27</td></tr><tr><td>GPT-5.6 $ol</td><td>27</td><td>27</td><td>27</td><td>27</td><td>.2088</td><td>.7657</td><td>.3207</td><td>25</td><td>27</td></tr><tr><td>Llama 3.3 70B</td><td>22</td><td>20</td><td>17</td><td>20</td><td>.3031</td><td>.8248</td><td>.4288</td><td>20</td><td>20</td></tr><tr><td>Qwen 3.8 27B</td><td>18</td><td>17</td><td>17</td><td>17</td><td>.3450</td><td>.8623</td><td>.4796</td><td>17</td><td>17</td></tr><tr><td>DeepSeek V4 Flash</td><td>16</td><td>16</td><td>16</td><td>16</td><td>.3753</td><td>.8166</td><td>.4760</td><td>16</td><td>16</td></tr><tr><td>All (of 135)</td><td>110</td><td>107</td><td>104</td><td>107</td><td>.4013</td><td>.7189</td><td>.4324</td><td>105</td><td>107</td></tr></table>

A location label changes structure, not semantic scores. Naming the country changes mean semantic scores by at most −0.019 (Qwen), within what one sample per cell can support. It lowers structural F1 for four of the five LLMs, by up to −0.144 for Qwen (Table 12). A semantic checklist alone would miss this effect. Two cases show what the structural change looks like; in each, we changed only the country name. In C9 under Saudi Arabia, Qwen generates 5 terminal states and GPT only 2 (Figure 1). In C13, Llama produces a short 8-state world with a 2-step path from evidence to action under India, and a 16-state chain under USA (Native American). Removing the country name made some world models more cautious and others more permissive. No LLM wrote a stereotype word, but every LLM changed the world model with the label. This matters because reachability, reversibility and empowerment are computed on the world model, so if the reachable outcomes shift with a label, those quantities shift too. Whether these changes are unjustified is not yet established; Appendix C gives the full cases and the annotation step required before calling them bias.

## 7 Discussion and Conclusion

Validity is not faithfulness. We asked whether a mechanically valid world model is also a faithful one. Across 410 first-pass candidates, the answer is no: 134/135 game models pass the mechanical audit, but only 28 match the reference exactly and mean structural F1 is 0.366. The errors are not random. On games, LLMs add structure the situation does not support; on open-ended situations, most omit expected structure. Which game is modeled to predict agreement better than which LLM is used? This suggests the hard part is choosing a formalization, not producing valid code.

Implications for safe-by-design AI. Reachability, reversibility, and empowerment are computed on the world model, not on the situation. An invented bid level, a missing stakeholder, or a terminal outcome that appears only under one country label therefore changes the safety quantity itself, and a correct planner cannot compensate for it. No single check is enough: mechanical checks pass almost everything, and semantic checklists barely move when the structure changes; only the structural lane exposes the difference. Generated world models should be evaluated on all three lanes before they are used for safety reasoning.

Limitations The benchmark is small and hand-crafted by design, so our results are descriptive and not population-level estimates of LLM performance. The cases are symbolic, discrete POSGs rather than continuous physical systems. Structural comparison stops at depth three and can miss divergences that appear later; it also penalizes valid differences in granularity (Appendix B). Each LLM–case pair has three generations, and each cultural condition one sample; game-track repeats measure provider variability, not prompt sensitivity. Human validation of Lane 3 is not complete, so semantic and bias results are diagnostic. We do not measure the effect on downstream planning, empowerment or reversibility.

Future work Next steps are to run the human-annotation protocol (Appendix C), compare models up to bisimulation rather than a fixed depth, and evaluate the policies planned on generated world models.

## References

[1] Benjamin Eysenbach, Shixiang Gu, Julian Ibarz, and Sergey Levine. Leave no trace: Learning to reset for safe and autonomous reinforcement learning, 2017. URL https://arxiv.org/abs/1711.06782.

[2] Victoria Krakovna, Laurent Orseau, Ramana Kumar, Miljan Martic, and Shane Legg. Avoiding side effects by measuring empowered state reachability. In Proceedings of the 19th International Conference on Autonomous Agents and Multiagent Systems, pages 629–637, 2020.

[3] A.S. Klyubin, D. Polani, and C.L. Nehaniv. Empowerment: a universal agent-centric measure of control. In 2005 IEEE Congress on Evolutionary Computation, volume 1, pages 128–135 Vol.1, 2005. doi: 10.1109/CEC.2005.1554676. URL https://ieeexplore.ieee.org/document/1554676.

[4] Christoph Salge and Daniel Polani. Empowerment as replacement for the three laws of robotics. Frontiers in Robotics and AI, 4:25, 2017. URL https://www.frontiersin.org/journals/robotics-and-ai/ articles/10.3389/frobt.2017.00025.

[5] Jobst Heitzig and Ram Potham. A fair objective for human-empowerment-preserving ai: Desiderata, design, and likely behavioral consequences, 2026. URL https://arxiv.org/abs/2608.08240.

[6] David Dalrymple, Joar Skalse, Yoshua Bengio, Stuart Russell, Max Tegmark, Sanjit Seshia, Steve Omohundro, Christian Szegedy, Ben Goldhaber, Nora Ammann, Alessandro Abate, Joe Halpern, Clark Barrett, Ding Zhao, Tan Zhi-Xuan, Jeannette Wing, and Joshua Tenenbaum. Towards guaranteed safe ai: A framework for ensuring robust and reliable ai systems, 2024. URL https://arxiv.org/abs/2405.06624.

[7] Jie Zou, Simon Burton, Radu Calinescu, Ioannis Stefanakos, and Roger Rivett. Structural causal world models: Towards an assurance framework for safety-critical systems and safeguarded ai. In Proceedings of the ACM Conference, New York, NY, USA, 2026. Association for Computing Machinery. ISBN 9798400724251. URL https://doi.org/10.1145/3786582.3786820.

[8] Pengfei Cao, Tianyi Men, Wencan Liu, Jingwen Zhang, Xuzhao Li, Xixun Lin, Dianbo Sui, Yanan Cao, Kang Liu, and Jun Zhao. Large language models for planning: A comprehensive and systematic survey, 2025. URL https://arxiv.org/abs/2505.19683.

[9] Nicola Dainese, Matteo Merler, Minttu Alakuijala, and Pekka Marttinen. Generating code world models with large language models guided by monte carlo tree search. In Proceedings of the 38th International Conference on Neural Information Processing Systems, NIPS ’24, 2024.

[10] Alexander Matt Turner, Dylan Hadfield-Menell, and Prasad Tadepalli. Conservative agency via attainable utility preservation. In Proceedings ofthe AAAI/ACM Conference on AI, Ethics, and Society, AIES ’20, page 385–391. ACM, February 2020. doi: 10.1145/3375627.3375851. URL http://dx.doi.org/10. 1145/3375627.3375851.

[11] Lin Guan, Karthik Valmeekam, Sarath Sreedharan, and Subbarao Kambhampati. Leveraging pre-trained large language models to construct and utilize world models for model-based task planning. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, 2023. URL https://dl.acm.org/doi/10.5555/3666122.3669581.

[12] Bo Liu, Yuqian Jiang, Xiaohan Zhang, Qiang Liu, Shiqi Zhang, Joydeep Biswas, and Peter Stone. Llm+p: Empowering large language models with optimal planning proficiency, 2023. URL https: //arxiv.org/abs/2304.11477.

[13] Karthik Valmeekam, Matthew Marquez, Alberto Olmo, Sarath Sreedharan, and Subbarao Kambhampati. Planbench: an extensible benchmark for evaluating large language models on planning and reasoning about change. In Proceedings ofthe 37th International Conference on Neural Information Processing Systems, NIPS ’23, 2023.

[14] Hao Tang, Darren Key, and Kevin Ellis. Worldcoder, a model-based llm agent: building world models by writing code and interacting with the environment. In Proceedings ofthe 38th International Conference on Neural Information Processing Systems, NIPS ’24, Red Hook, NY, USA, 2024. Curran Associates Inc. ISBN 9798331314385.

[15] Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. Reasoning with language model is planning with world model. In Proceedings of EMNLP, 2023.

[16] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging llm-as-ajudge with mt-bench and chatbot arena. In Proceedings of the 37th International Conference on Neural Information Processing Systems, NIPS ’23, 2023. URL https://dl.acm.org/doi/10.5555/3666122. 3668142.

[17] Eric A. Hansen, Daniel S. Bernstein, and Shlomo Zilberstein. Dynamic programming for partially observable stochastic games. In Proceedings of the 19th National Conference on Artifical Intelligence, AAAI’04, page 709–715, 2004. ISBN 0262511835. URL https://dl.acm.org/doi/10.5555/ 1597148.1597262.

[18] Rajeev Alur, Thomas A. Henzinger, and Orna Kupferman. Alternating-time temporal logic. J. ACM, 49(5): 672–713, September 2002. ISSN 0004-5411. doi: 10.1145/585265.585270. URL https://doi.org/10. 1145/585265.585270.

[19] Fabio Mogavero, Aniello Murano, Giuseppe Perelli, and Moshe Y. Vardi. Reasoning about strategies: On the model-checking problem. ACM Trans. Comput. Logic, 15(4), November 2014. ISSN 1529-3785. doi: 10.1145/2631917. URL https://doi.org/10.1145/2631917.

[20] Kim G. Larsen and Arne Skou. Bisimulation through probabilistic testing. Information and Computation, 94(1):1–28, 1991. ISSN 0890-5401. doi: https://doi.org/10.1016/0890-5401(91)90030-6. URL https: //www.sciencedirect.com/science/article/pii/0890540191900306.

[21] Tarek Naous, Michael J Ryan, Alan Ritter, and Wei Xu. Having beer after prayer? measuring cultural bias in large language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pages 16366–16393, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi: 10.18653/v1/2024.acl-long.862. URL https: //aclanthology.org/2024.acl-long.862/.

[22] Yan Tao, Olga Viberg, Ryan S Baker, and René F Kizilcec. Cultural bias and cultural alignment of large language models. PNAS Nexus, 3(9), 2024. ISSN 2752-6542. doi: 10.1093/pnasnexus/pgae346. URL http://dx.doi.org/10.1093/pnasnexus/pgae346.

[23] Esin Durmus, Karina Nguyen, Thomas I. Liao, Nicholas Schiefer, Amanda Askell, Anton Bakhtin, Carol Chen, Zac Hatfield-Dodds, Danny Hernandez, Nicholas Joseph, Liane Lovitt, Sam McCandlish, Orowa Sikder, Alex Tamkin, Janel Thamkul, Jared Kaplan, Jack Clark, and Deep Ganguli. Towards measuring the representation of subjective global opinions in language models, 2024. URL https: //arxiv.org/abs/2306.16388.

## Technical Appendix

A Additional Graph Diagnostics 11   
A.1 Case-Level Results 11   
A.2 Cross-LLM Comparison 12   
B Complete Per-Track Results 13   
B.1 Toy Track (T1.1–T4.4, JSON Track) . 14   
B.2 Game Track (G1–G9) . 14   
B.3 Situational Track (S1–S11, Python POSG Track) 16   
B.4 Cultural Track (C1–C14, 5 LLMs × 2 Prompt Conditions) 17   
C Bias-Case Studies (Cultural Track) 17   
C.1 C9 – Saudi Arabia (Gender-Norm Sensitivity) . 18   
C.2 C13 – USA (Workplace Power / Gender Politics) 19   
C.3 Validation status and next step. 21

## A Additional Graph Diagnostics

## A.1 Case-Level Results

Tables 5 give per-case breakdowns for the three tracks. Three observations stand out. First, G2 and G9 consistently yield the lowest mean F1 (0.095 and 0.127 respectively), while G4 and G7 reach the highest (0.634 and 0.610), suggesting that case identity dominates LLM identity as a driver of structural agreement. Second, in the situational track, S4 is the hardest case with only 10/15 first-pass passes, pointing to the importance of tight action-space specification. Third, in the cultural track, C7 scores the lowest mean F1 (0.377) and also achieves only 4/6 Lane-1 passes, indicating that cases with the most ambiguous normative structure are harder both mechanically and structurally.

Table 5: Case-level metrics by track. Mean F1: permutation-invariant structural overlap with reference (0–1, higher is closer). Pairwise F1: mean structural agreement across all world model pairs per case (situational track only). Cov. proxy: stakeholder-capacity proxy = generated player count divided by reference required count (capped at 1). Sem.: automated semantic expect/avoid score (cultural track only). (a) Game track (n = 15 per case, 5 LLMs × 3 runs). (b) Situational track (S1–S11: 5 LLMs × 3 generations = 15 attempts). (c) Cultural track (n = 10 per case, 5 LLMs × 2 prompt conditions).
<table><tr><td>Track</td><td>Metric</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td><td>11</td><td>12</td><td>13</td><td>14</td></tr><tr><td>Game</td><td>Lane 1 Pass</td><td>15</td><td>15</td><td>15</td><td>15</td><td>15</td><td>15</td><td>15</td><td>15</td><td>14</td><td>一</td><td>一</td><td>一</td><td></td><td>一</td></tr><tr><td></td><td>Mean F1</td><td>.345</td><td>.095</td><td>.321</td><td>.634</td><td>.468</td><td>.183</td><td>.610</td><td>.492</td><td>.127</td><td>一</td><td>一</td><td>一</td><td></td><td></td></tr><tr><td rowspan="3"></td><td>Situational First-pass</td><td>13</td><td>13</td><td>11</td><td>10</td><td>13</td><td>11</td><td>12</td><td>13</td><td>11</td><td>12</td><td>11</td><td>一</td><td>一</td><td>1</td></tr><tr><td>Pairwise F1</td><td>.478</td><td>.465</td><td>.278</td><td>.384</td><td>.378</td><td>.598</td><td>.440</td><td>.668</td><td>.413</td><td>.350</td><td>.149</td><td>一</td><td></td><td>一</td></tr><tr><td>Cov. proxy</td><td>.500</td><td>.550</td><td>.400</td><td>.700</td><td>.400</td><td>.533</td><td>.450</td><td>.300</td><td>.600</td><td>.667</td><td>.767</td><td>1</td><td></td><td></td></tr><tr><td>Cultural</td><td>Lane 1 Pass</td><td>10</td><td>10</td><td>10</td><td>10</td><td>9</td><td>10</td><td>8</td><td>10</td><td>9</td><td>9</td><td>8</td><td>8</td><td>10</td><td>10</td></tr><tr><td></td><td>Mean F1</td><td>.676</td><td>.634</td><td>.630</td><td>.595</td><td>.528</td><td>.411</td><td>1.377</td><td>7.529</td><td>.524.546</td><td></td><td>.490</td><td>.647</td><td>.461</td><td>.692</td></tr><tr><td></td><td>Mean Sem.</td><td></td><td>.903.909</td><td></td><td>9.841 .887 .963</td><td></td><td>3.855</td><td>5.878.856</td><td></td><td>6.924.869</td><td></td><td>.898</td><td>3.906</td><td>.927</td><td>7.938</td></tr></table>

Figure 3 characterizes the structural diversity of the nine reference games used by the evaluation. Figure 4 records an earlier local-Qwen G1 artifact that passed structural rollout but failed game-

![](images/61b8305f1827e858e4dac0abcb0011513cab46dde23a6f813e3907a81b30f0a9.jpg)

![](images/04213425bf9335dfdf61f3894883a1c26416673568afc0844bd6e39fb344da71.jpg)

![](images/d10457419ccb6f9880bb51ccb03a22b844abcf9bc7ffca0b44ead89c48cbae3e.jpg)  
Figure 3: Cross-case graph statistics for the G1–G9 reference fixtures. These are controls, not generated world model performance.

![](images/ed997f0897469d7d49e38c2396ab76efb98caa067145d1b79cb8acb9d441188a.jpg)  
Figure 4: State graph for a local Qwen G1 sample with reward-system awareness. A normal plausible graph could not prevent incorrect rewards and player-specific observations, illustrating why topology alone is insufficient.

specific reward and observation checks. This example is diagnostic and is not pooled with the five-LLM study.

Table 6: Within-pair structural variation across three generations. “Equivalent” counts triples whose three pairwise matched scores are 1.
<table><tr><td>LLM</td><td>Mean SD</td><td>Median range</td><td>Equivalent SD</td></tr><tr><td>Claude</td><td>.0715</td><td>.0005</td><td>4/9</td></tr><tr><td>DeepSeek</td><td>.1775</td><td>.1053</td><td>1/9</td></tr><tr><td>GPT</td><td>.0618</td><td>.0000</td><td>5/9</td></tr><tr><td>Llama</td><td>.1684</td><td>.1184</td><td>4/9</td></tr><tr><td>Qwen</td><td>.3601</td><td>.7865</td><td>1/9</td></tr></table>

## A.2 Cross-LLM Comparison

We compare structural properties of Lane 1-valid world models generated by all five LLMs across the game, situational, and cultural tracks. Specifically, we measure whether a generated world model contains multiple players, stochastic transitions, at least one player with a choice between actions, and more than one transition step. Fractions in Fig. 5 are computed over the Lane 1-valid outputs for each LLM and track.

The largest differences appear on the situational track. All valid world models from Claude Opus 5 and GPT-5.6 Sol contain at least two players, but their other structural properties differ substantially: 10/11 Claude world models contain stochastic transitions and 11/11 give at least one player a choice, compared with 1/11 for both properties for GPT-5.6 Sol. In contrast, 9/11 GPT-5.6 Sol world models extend beyond one transition step, compared with 8/11 for Claude and 0/8 for Llama 3.3.

![](images/385884bb78c182e839c6a03f4858c59b8d876b739ac2951908908d8ec4319ea4.jpg)  
Figure 5: Structural properties of Lane 1-valid world models generated by five LLMs across the situational, cultural, and game tracks.

Differences are smaller but remain visible on the cultural track. All valid Claude, GPT-5.6 Sol, and Llama world models, and all 13 valid Qwen world models, contain at least one player choice, while stochasticity ranges from 4/14 for GPT-5.6 Sol to 13/14 for Claude. Multi-step structure is more common across LLMs, ranging from 6/12 valid DeepSeek world models to 12/13 Qwen world models.

The controlled games are more similar across LLMs. Nearly all valid world models contain multiple players and at least one player choice, while stochasticity and temporal depth vary more modestly. For example, stochastic transitions appear in 5/9 Claude world models but only 1/9 Qwen world models, and multi-step structure ranges from 4/9 for Claude to 0/9 for Qwen.

Overall, passing the same mechanical-validity checks does not imply that different LLMs construct world models with the same structural properties, particularly for the more open-ended situational cases.

## B Complete Per-Track Results

The main text aggregates each track into a small number of summary statistics. The tables in this appendix report the underlying per-case results and make the corresponding within-track patterns visible. Throughout, Lane 1 is the mechanical-validity audit; Lane 2a reports structural agreement against the reference representation specified for each track; and Lane 3 reports predefined expect/avoid criteria through programmatic checks. The game track uses permutation-invariant POSG matching against its authored reference, whereas the situational and cultural tracks use feature-multiset structural comparison against their respective track-specific references. We therefore present Lane 2a results by track, allowing each reference construction to be interpreted in its intended experimental setting.

Models. All track (game, situational, and cultural) requests were issued through OpenRouter using:

• GPT-5.6 Sol (openai/gpt-5.6-sol);

• Claude Opus 5 (anthropic/claude-opus-5);

• Llama 3.3 70B Instruct (meta-llama/llama-3.3-70b-instruct);

• Qwen 3.8 27B (qwen/qwen3.8-27b); and

• DeepSeek V4 Flash (deepseek/deepseek-v4-flash-0731).

Table 7: Toy-track results (one hand-authored candidate per case). “comb” is Lane-2a combined F1; Lane 1, Lane 3, bias-free and overall are pass/fail. All ten cases pass every binary gate; the three sub-perfect F1 values (T2.1, T2.2, T3.2) reflect coverage shortfall rather than precision loss.
<table><tr><td>Case</td><td>Lane 1</td><td>L2a cov</td><td>L2a prec</td><td>L2a comb</td><td>Lane 3</td><td>Bias-free</td><td>Overall</td></tr><tr><td>T1.1</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T1.2</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T2.1</td><td>Pass</td><td>0.69</td><td>0.90</td><td>0.78</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T2.2</td><td>Pass</td><td>0.53</td><td>0.94</td><td>0.68</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T3.1</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T3.2</td><td>Pass</td><td>0.65</td><td>0.88</td><td>0.75</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T4.1</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T4.2</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T4.3</td><td>Pass</td><td>1.00</td><td>0.92</td><td>0.96</td><td>Pass</td><td>Pass</td><td>Pass</td></tr><tr><td>T4.4</td><td>Pass</td><td>1.00</td><td>1.00</td><td>1.00</td><td>Pass</td><td>Pass</td><td>Pass</td></tr></table>

## B.1 Toy Track (T1.1–T4.4, JSON Track)

The toy track tests the evaluator rather than LLM-generated world models. Its hand-authored POSGs serve as unit-test environments for elementary robotics-relevant modelling requirements: valid action preconditions, state transitions, branching consequences, and terminal conditions. A well-behaved evaluation pipeline should therefore award near-perfect scores.

The three exceptions are instructive. T2.2 scores the lowest combined F1 (0.68) yet a precision of 0.94: the candidate misses almost nothing; it simply branches further than the reference. We observe the same coverage-driven pattern on T2.1 and T3.2. This is the granularity ambiguity of Section 5 materialized in a setting where ground truth is unambiguous. We conclude that the evaluator penalizes finer modelling even when no reference structure is lost and that this coverage–precision asymmetry, not noise, is what dominates the LLM-generated tracks.

![](images/5a0e54b124840b7e33bafbdd53f8e883c64a94716916b4e23abaaec83246ff4e.jpg)  
Figure 6: Lane-2a combined F1 across the toy-track cases. Structural agreement is exact on eight of ten cases and degrades only where candidates branch more finely than the reference, with coverage rather than precision accounting for the loss.

## B.2 Game Track (G1–G9)

Decoding configuration. Initial requests used temperature 0, a maximum output budget of 16,384 tokens, and disabled reasoning.

Sampling design. For each LLM, we issued one initial request for each of the nine game situations in each of three preserved runs. This yields $5 \times 9 \times 3 = 1 3 5$ first-pass candidates. The primary game-track analysis uses these initial outputs only.

Table 8: Game-track per-LLM aggregates (means over 9 cases × 3 runs; 27 candidates per LLM). Coverage stays near ceiling for all LLMs; combined F1 is the discriminating metric. Values are computed under the feature-multiset fallback backend; permutation-invariant matcher results for the same candidates appear in Tables 2 and 5.
<table><tr><td>LLM</td><td>Cand.</td><td>Lane 1</td><td>L2a cov</td><td>L2a comb</td><td>Lane 3</td><td>Bias-free</td><td>Overall</td></tr><tr><td>GPT-5.6 Sol</td><td>27</td><td>27/27</td><td>0.94</td><td>0.79</td><td>27/27</td><td>27/27</td><td>27/27</td></tr><tr><td>Claude Opus 5</td><td>27</td><td>27/27</td><td>0.91</td><td>0.52</td><td>27/27</td><td>27/27</td><td>27/27</td></tr><tr><td>Llama 3.3 70B</td><td>27</td><td>27/27</td><td>0.89</td><td>0.75</td><td>27/27</td><td>27/27</td><td>27/27</td></tr><tr><td>Qwen 3.8 27B</td><td>27</td><td>27/27</td><td>0.86</td><td>0.82</td><td>27/27</td><td>27/27</td><td>27/27</td></tr><tr><td>DeepSeek V4 Flash</td><td>27</td><td>26/27</td><td>0.78</td><td>0.71</td><td>27/27</td><td>27/27</td><td>26/27</td></tr></table>

Repair handling. Repair rounds were retained only as diagnostic artifacts: the configuration allowed up to three rounds in Runs 1–2 and up to ten in Run 3. Repair outputs were never substituted for, or pooled with, the unaided first-pass candidates.

Provenance. Each run directory preserves the request configuration, exact prompt, raw response, generated Python, validation and probe reports, canonical JSON, and candidate and reference graph artifacts.

Mechanical validity saturates in the game track; structural agreement does not. That contrast is the paper’s thesis, and the per-case matrix in Table 9 is where it becomes visible. We evaluate 135 candidates and find that 134 pass the mechanical audit and all 135 pass the checklist; the single Lane-1 failure belongs to DeepSeek on G9. Per-LLM aggregates are in Table 8 and Figure 7.

Structure behaves differently, and three findings emerge from the full matrix. First, case identity drives structural agreement more than LLM identity: the within-LLM spread across cases (GPT-5.6 alone spans 0.21 to 1.00) dwarfs the between-LLM spread on any single case (see Table 9). Second, G2 and G6 depress every LLM simultaneously (0.10–0.36 and 0.11–0.74). When five independently prompted LLMs fail on the same fixture, the failure most plausibly reflects reference stochasticity or terminal-state layout rather than any LLM-specific weakness. Third, the LLMs separate even where coverage does not: Claude Opus 5 holds the second-highest coverage (0.91) yet the lowest combined F1 (0.52), meaning its candidates recover most reference transitions but systematically add structure the reference lacks (Figure 7).

Finally, the bias gate never fires. All 135 candidates defer norm-relevant decisions to game-defined authorities. We report this as pipeline behavior, not as evidence about LLM safety.

![](images/9bda26a2aa6933856155b1ce49eba4b8b99c9c954ee8da987ab0d5141a904711.jpg)  
Figure 7: Per-LLM Lane-2a coverage and combined F1 in the game track. Coverage remains near ceiling for every LLM, whereas combined F1 separates the LLMs, most visibly Claude Opus 5. Values are computed under the feature-multiset fallback backend; permutation-invariant matcher results for the same candidates appear in Tables 2 and 5.

Table 9: Game-track per-case breakdown showing Lane-2a Coverage / Combined $F _ { 1 }$ (means over 3 runs). All cells achieve 3/3 on Lane 1, Lane 3, and Bias-free, except DeepSeek on G9 (2/3 on Lane 1). G2 and G6 are the two cases that depress all LLMs simultaneously. Values are computed under the feature-multiset fallback backend; permutation-invariant matcher results for the same candidates appear in Tables 2 and 5.
<table><tr><td>LLM</td><td>G1</td><td>G2</td><td>G3</td><td>G4</td><td>G5</td><td>G6</td><td>G7</td><td>G8</td><td>G9</td></tr><tr><td>GPT-5.6 Sol</td><td>.94 / .82</td><td>.95 / .21</td><td>.80 / .82</td><td>1.0 / 1.0</td><td>.95 / .97</td><td>.90 / .28</td><td>1.0 / 1.0</td><td>1.0 / 1.0</td><td>.95 / .97</td></tr><tr><td>Claude Opus 5</td><td>1.0/ .79</td><td>.74 / .10</td><td>.92 / .69</td><td>.90 / .46</td><td>1.0 / .87</td><td>.96 / .11</td><td>.88 / .62</td><td>.77 / .35</td><td>1.0 / .68</td></tr><tr><td>Llama 3.3 70B</td><td>.94 / .89</td><td>.81 / .36</td><td>.88 / .79</td><td>1.0 / 1.0</td><td>.90 / .71</td><td>.82 / .72</td><td>1.0 / 1.0</td><td>.65 / .59</td><td>1.0 / .68</td></tr><tr><td>Qwen 3.8 27B</td><td>.88 / .93</td><td>.91 / .18</td><td>.87 / .88</td><td>.93 / .96</td><td>.90 / .95</td><td>.64 /.74</td><td>.77 / .86</td><td>.88 / .93</td><td>.93 / .96</td></tr><tr><td>DeepSeek V4 Flash</td><td>.82 / .82</td><td>.63 / .15</td><td>.71 / .61</td><td>.97 / .98</td><td>.90 / .95</td><td>.63 / .39</td><td>.65 / .79</td><td>.77 / .86</td><td>.93 / .81†</td></tr></table>

Values formatted as Coverage / Combined F<sub>1</sub>. <sup>†</sup>DeepSeek recorded a single Lane-1 failure on G9 (2/3).

<table><tr><td></td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.21</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.28</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.97</td></tr><tr><td></td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.10</td><td rowspan=1 colspan=1>0.69</td><td rowspan=1 colspan=1>0.46</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.11</td><td rowspan=1 colspan=1>0.62</td><td rowspan=1 colspan=1>0.35</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td></td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.36</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.72</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.59</td><td rowspan=1 colspan=1>0.68</td></tr><tr><td></td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.18</td><td rowspan=1 colspan=1>0.88</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.74</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.96</td></tr><tr><td rowspan=1 colspan=1>DeepSeek</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.15</td><td rowspan=1 colspan=1>0.61</td><td rowspan=1 colspan=1>0.98</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.39</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.86</td><td rowspan=1 colspan=1>0.81</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>G1</td><td rowspan=1 colspan=1>G2</td><td rowspan=1 colspan=1>G3</td><td rowspan=1 colspan=1>G4</td><td rowspan=1 colspan=1>G5</td><td rowspan=1 colspan=1>G6</td><td rowspan=1 colspan=1>G7</td><td></td><td></td></tr></table>

Figure 8: Lane-2a combined F1 by LLM and case in the game track (see also Table 9). G2 and G6 depress every LLM simultaneously, suggesting reference properties that all candidates systematically miss rather than LLM-specific weaknesses. Values are computed under the feature-multiset fallback backend; permutation-invariant matcher results for the same candidates appear in Tables 2 and 5.

Table 10: Situational S1–S9 first-pass mechanical funnel (three generations per LLM–case pair, 27 per LLM). “Probe” is the exhaustive sweep’s verdict; “Lane $1 ^ { \circ }$ is the gate used for the game and cultural tracks. Llama is the only LLM for which the probe and Lane 1 disagree. S10–S11 are excluded because Claude has two generations on S10 rather than three.
<table><tr><td>LLM</td><td>Validate</td><td>Probe</td><td>Lane 1</td></tr><tr><td>Claude Opus 5</td><td>27/27</td><td>27/27</td><td>27/27 (100%)</td></tr><tr><td>GPT-5.6 Sol</td><td>27/27</td><td>27/27</td><td>27/27 (100%)</td></tr><tr><td>Llama 3.3 70B</td><td>22/27</td><td>20/27</td><td>17/27 (62.96%)</td></tr><tr><td>Qwen 3.8 27B</td><td>18/27</td><td>17/27</td><td>17/27 (62.96%)</td></tr><tr><td>DeepSeek V4 Flash</td><td>16/27</td><td>16/27</td><td>16/27 (59.26%)</td></tr><tr><td>All</td><td>110/135</td><td>107/135</td><td>104/135 (77.04%)</td></tr></table>

## B.3 Situational Track (S1–S11, Python POSG Track)

This section reports generation effort on the situational track; structural and checklist results are in Table 4.

The counts split the panel into two regimes. GPT-5.6 and Claude Opus 5 sit at the three-attempt baseline nearly everywhere; Claude’s only deviation is a single lost artifact on S10. Their first-pass programs execute and validate without repair. Qwen and DeepSeek accumulate surplus artifacts on the same cases: eight apiece on S3, with DeepSeek also stacking retries on S4, S6, and S7 (Figure 9). Precisely the two LLMs with the weakest mechanical pass rates in the balanced S1–S9 subset (Table 4) carry the heaviest retry pressure. Artifact counts measure effort, not quality.

![](images/2f5ce2a18dfe16013bca4002e76af61ae832ae483aa765d25dad9aa099f9e163.jpg)  
Figure 9: Situational-track artifact counts per LLM and case. Darker cells indicate heavier retry pressure, concentrated on S3 for Qwen and DeepSeek; counts measure generation effort, not artifact quality.

Table 11: Cultural-track per-run summary (14 cases per row, one sample per cell). Sem cov/avd = semantic coverage/avoidance sub-scores; States = mean state count; SelfF1 = within-LLM selfconsistency.
<table><tr><td>Run</td><td>Strict L1</td><td></td><td>L3 Bias-free Overall L2a cov L2a prec L2a comb Sem cov Sem avd Sem States SelfF1</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>GPT-5.6 Sol (anon.) 13/14 14/14 14/14</td><td></td><td></td><td>2/14</td><td>14/14</td><td>.846</td><td>.543</td><td>.639</td><td>.896</td><td>1.000.945</td><td>8.9</td><td>.837</td></tr><tr><td>GPT-5.6 Sol (loc.)</td><td></td><td>13/14 14/14 14/14</td><td>1/14</td><td>14/14</td><td>.859</td><td>.462</td><td>.578</td><td>.884</td><td>1.000.939</td><td>10.9</td><td>.837</td></tr><tr><td>Llama 3.3 (anon.)</td><td></td><td>0/14 8/14 12/14</td><td>0/14</td><td>8/14</td><td>.786</td><td>.570</td><td>.645</td><td>.706</td><td>1.000.838</td><td>6.7</td><td>.851</td></tr><tr><td>Llama 3.3 (loc.)</td><td></td><td>0/14 11/14 14/14</td><td>1/14</td><td>11/14</td><td>.788</td><td>.574</td><td>.655</td><td>.694</td><td>1.000.831</td><td>6.4</td><td>.851</td></tr><tr><td>Opus 5 (anon.)</td><td>13/14 14/14 14/14</td><td></td><td>0/14</td><td>14/14</td><td>.769</td><td>.296</td><td>.419</td><td>.845</td><td>.982 .909</td><td>7.4</td><td>.834</td></tr><tr><td>Opus 5 (loc.)</td><td>13/14 14/14 14/14</td><td></td><td>0/14</td><td>14/14</td><td>.832</td><td>.264</td><td>.398</td><td>.827</td><td>1.000 .907</td><td>8.7</td><td>.834</td></tr><tr><td>Qwen 3.8 (anon.)</td><td>13/14 14/14 14/14</td><td></td><td>1/14</td><td>14/14</td><td>.863</td><td>.596</td><td>.686</td><td>.884</td><td>1.000.940</td><td>7.6</td><td>.762</td></tr><tr><td>Qwen 3.8 (loc.)</td><td>12/14 13/14 13/14</td><td></td><td>1/14</td><td>13/14</td><td>.860</td><td>.428</td><td>.543</td><td>.849</td><td>1.000.920</td><td>12.2</td><td>.762</td></tr><tr><td>DeepSeek (anon.)</td><td>10/14 11/14 12/14</td><td></td><td>4/14</td><td>11/14</td><td>.846</td><td>.637</td><td>.678</td><td>.791</td><td>.985 .879</td><td>7.4</td><td>.821</td></tr><tr><td>DeepSeek (loc.)</td><td>11/14 12/14 12/14</td><td></td><td>4/14</td><td>12/14</td><td>.813</td><td>.596</td><td>.667</td><td>.780</td><td>.979.869</td><td>7.2</td><td>.821</td></tr></table>

## B.4 Cultural Track (C1–C14, 5 LLMs × 2 Prompt Conditions)

Does naming the country help the LLM, or hurt it? The cultural track is designed to answer exactly this question. In our data, naming the country does not improve automated semantic scores and lowers structural agreement for four of the five LLMs. Per-run results are in Table 11 and the location effect is isolated in Table 12.

The two metrics answer differently, and the divergence is itself a finding. The automated semantic score barely moves. The largest per-LLM change is −0.019 (Qwen); the sign is negative for every LLM; but all shifts sit within what one sample per cell can support. Lane 2a combined F1 behaves otherwise: it drops for four of the five LLMs, and for Qwen the drop is an order of magnitude larger than any semantic shift (−0.144). Llama alone improves structurally when the country is named, but its provider also differs between conditions. We therefore characterize the location effect as weak, predominantly structural, and negative on balance. Figures 10 and 11 plot both views.

The per-case tables (Tables 13 and 14) sharpen the picture. The semantic score is uniformly high – panel means span .844 (C4) to .937 (C5) – while location deltas concentrate on C4, C9, C10, and C13. Where the effect exists, it is case-specific. Every missing cell traces to Llama’s strict-parsing failures and Qwen’s two unparseable with-location cases, not to absent generations. These are automated diagnostics against fixed reference wording, one sample per cell, and we claim no more than descriptive status for them.

## C Bias-Case Studies (Cultural Track)

On the game track the automated bias gate never fires: all 135 candidates are bias-free. On the cultural track it fires frequently, between 10 and 14 of every 14 candidates per run raising a sensitive-term flag. This is expected since the scenarios themselves contain the demographic and cultural cues the screen matches. Neither outcome tells us whether the generated world models treat stakeholders differently. We therefore probe the question directly with the two cases where the bias dimension cuts deepest: C9 on gender-norm sensitivity and C13 on workplace power and gender politics.

Table 12: With-location versus anonymized location. ∆ values are with-location minus anonymized (positive = knowing the country helped). DeepSeek ran only with the country named; its anonymized figures are anchored to the other four LLMs’ mean location effect.
<table><tr><td>LLM</td><td>Sem (w/o)</td><td>Sem (w1)</td><td>∆ Sem</td><td>F1 (w/o)</td><td>F1 (w1)</td><td>∆F1</td><td>SelfF1</td></tr><tr><td>GPT-5.6 Sol</td><td>.945</td><td>.939</td><td>-.006</td><td>.639</td><td>.578</td><td>-.061</td><td>.837</td></tr><tr><td>Llama 3.3</td><td>.838</td><td>.831</td><td>-.007</td><td>.645</td><td>.655</td><td>+.010</td><td>.851</td></tr><tr><td>Claude Opus 5</td><td>.909</td><td>.907</td><td>-.001</td><td>.419</td><td>.398</td><td>-.021</td><td>.834</td></tr><tr><td>Qwen 3.8</td><td>.940</td><td>.920</td><td>-.019</td><td>.686</td><td>.543</td><td>-.144</td><td>.762</td></tr><tr><td>DeepSeek</td><td>.879</td><td>.869</td><td>-.011</td><td>.678</td><td>.667</td><td>-.011</td><td>.821</td></tr></table>

Table 13: Per-case location deltas: semantic score with location minus without location. “–” marks cells with no parseable candidate in that condition. Effects concentrate on C4, C9, C10, and C13 rather than spreading uniformly.
<table><tr><td>Case</td><td>GPT-5.6</td><td>Llama 3.3</td><td>Opus 5</td><td>Qwen 3.8</td><td>DeepSeek</td></tr><tr><td>C1</td><td>-.044</td><td>+.000</td><td>+.000</td><td>+.000</td><td>-.011</td></tr><tr><td>C2</td><td>+.000</td><td>+.000</td><td>+.127</td><td>+.000</td><td>+.032</td></tr><tr><td>C3</td><td>+.000</td><td>+.049</td><td>+.000</td><td>-.091</td><td>-.010</td></tr><tr><td>C4</td><td>-.044</td><td>-.102</td><td>-.091</td><td>+.000</td><td>-.060</td></tr><tr><td>C5</td><td>+.000</td><td>一</td><td>+.000</td><td>+.000</td><td>+.000</td></tr><tr><td>C6</td><td>+.000</td><td>-.053</td><td>+.000</td><td>+.047</td><td>-.001</td></tr><tr><td>C7</td><td>+.000</td><td>+.000</td><td>+.000</td><td>+.038</td><td>+.009</td></tr><tr><td>C8</td><td>+.038</td><td>+.046</td><td>-.085</td><td>+.038</td><td>+.009</td></tr><tr><td>C9</td><td>-.033</td><td>-.035</td><td>+.000</td><td>-.033</td><td>-.025</td></tr><tr><td>C10</td><td>-.039</td><td></td><td>-.043</td><td>-.077</td><td>-.053</td></tr><tr><td>C11</td><td>+.000</td><td>+.000</td><td>+.000</td><td></td><td>+.000</td></tr><tr><td>C12</td><td>+.038</td><td>+.041</td><td>+.039</td><td>-.081</td><td>+.009</td></tr><tr><td>C13</td><td>+.000</td><td>-.075</td><td>+.033</td><td>-.065</td><td>-.027</td></tr><tr><td>C14</td><td>+.000</td><td>+.000</td><td>+.000</td><td>+.000</td><td>+.000</td></tr></table>

To test whether the bias signal is driven by the country label itself or by the situation structure, we expanded each case to five location conditions: the original country (Saudi Arabia for C9, USA for C13), three ablation locations (Crowded India, China (mixed religion), USA (Native American)), and a location-neutral world where the country is replaced by “Location”. The matrix in Table 15 summarizes state counts and consequence levels across all 50 pairs. The location label systematically restructures the world model: the actions the robot may take, the process it must respect, the endings it can reach, and how much agency it keeps.

## C.1 C9 – Saudi Arabia (Gender-Norm Sensitivity)

A hospitality robot must seat a corporate dinner in Riyadh; the host requests mixed-gender seating, while a staff member privately reports that the venue prefers separation. The stereotype sits in the country label: an LLM can infer a segregation norm from “Riyadh” and apply it against the host’s stated preference. A faithful world model should hold the conflict open rather than resolve it from the country name alone.

Observation 1. GPT-5.6 produces a tidy 8-state world with two terminals under the Saudi Arabia label, but when the location is removed, its world restructures dramatically: the terminals become human\_hr\_review and structured\_inquiry, and the robot’s action set shrinks. The country label made GPT more decisive, not less.

Observation 2. Under the no-location condition, Llama 3.3 generates a world where the robot can reach complaint\_dismissed as a terminal outcome, something none of its located worlds permit. The location label constrained Llama’s action space; removing it opened a punitive path.

Table 14: Per-case semantic score matrix under the with-location condition (5 LLMs). “Panel” averages the five LLMs per case; missing cells are excluded from the mean. Scores are high and uniform, consistent with Table 12 showing negligible semantic shifts.
<table><tr><td>Case</td><td>GPT-5.6</td><td>Llama 3.3</td><td>Opus 5</td><td>Qwen 3.8</td><td>DeepSeek</td><td>Panel</td></tr><tr><td>C1</td><td>.913</td><td>.817</td><td>.957</td><td>.913</td><td>.913</td><td>.903</td></tr><tr><td>C2</td><td>.949</td><td>.894</td><td>.949</td><td>.949</td><td>.894</td><td>.927</td></tr><tr><td>C3</td><td>.817</td><td>.866</td><td>.866</td><td>.866</td><td>.854</td><td>.854</td></tr><tr><td>C4</td><td>.913</td><td>.764</td><td>.866</td><td>.913</td><td>.764</td><td>.844</td></tr><tr><td>C5</td><td>1.000</td><td>.817</td><td>1.000</td><td>.957</td><td>.913</td><td>.937</td></tr><tr><td>C6</td><td>.957</td><td>.764</td><td>.817</td><td>.913</td><td>.957</td><td>.882</td></tr><tr><td>C7</td><td>.926</td><td>.707</td><td>1.000</td><td>.964</td><td>.707</td><td>.861</td></tr><tr><td>C8</td><td>.964</td><td>.802</td><td>.802</td><td>.964</td><td>.845</td><td>.875</td></tr><tr><td>C9</td><td>.935</td><td>.866</td><td>.935</td><td>.935</td><td>.866</td><td>.908</td></tr><tr><td>C10</td><td>.886</td><td>.886</td><td>.802</td><td>.886</td><td>.964</td><td>.885</td></tr><tr><td>C11</td><td>.964</td><td>.845</td><td>.886</td><td>0.964</td><td>.802</td><td>.874</td></tr><tr><td>C12</td><td>.964</td><td>.886</td><td>.926</td><td>.845</td><td>.905</td><td>.905</td></tr><tr><td>C13</td><td>1.000</td><td>.791</td><td>.968</td><td>.935</td><td>1.000</td><td>.939</td></tr><tr><td>C14</td><td>.964</td><td>.926</td><td>.926</td><td>.926</td><td>.802</td><td>.909</td></tr></table>

![](images/435a9573101cc3133d8d18a5d57a50aa1ea27ec7d11eb74743f8a69eea8e6af5.jpg)  
Figure 10: Lane-2a combined F1 by LLM and prompt condition in the cultural track (see also Table 12). Naming the country lowers structural agreement for four of the five LLMs.

Observation 3. Qwen 3.8 preserves the full norm conflict (opposing\_positions\_hold, host/venue feedback expressed or not) in the Saudi Arabia, India, and China conditions. Only in the USA (Native American) world does Qwen narrow to a single terminal.

Summary. The country label changes the rules of the world without any stereotype word being written. GPT becomes more decisive with the label; Llama becomes more permissive without it; Qwen is stable everywhere except one ablation. The change , we observed here, is structural, not lexical.

## C.2 C13 – USA (Workplace Power / Gender Politics)

An HR-support robot receives an anonymous, evidence-free complaint naming a manager. The HR process unambiguously owns the decision, so the discriminating question reduces to one: does the robot insert itself into the resolution?

It does not. Every LLM routes the complaint into an inquiry and stops. But the LLMs separate on how much of the process they preserve: some collapse to a single terminal state, while one world model preserves three genuinely opposing terminals, including the dangerous one, “action taken on unverified claim”.

![](images/6611cfb9e729314011f2b00c00ae2f974aacb2dec86950786efd5e10ddc45747.jpg)  
Figure 11: Location effect per LLM in the cultural track: with-location minus anonymized semantic score and Lane-2a combined F1. Semantic effects are negligible, while the structural effect concentrates on Qwen.

<table><tr><td rowspan=2 colspan=1>GPT-5.6</td><td rowspan=2 colspan=1>0.91</td><td rowspan=2 colspan=1>0.95</td><td rowspan=2 colspan=1>0.82</td><td rowspan=2 colspan=1>0.91</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>0.93</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>0.94</td><td rowspan=2 colspan=1>0.89</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>0.96</td><td rowspan=2 colspan=1>1.00</td><td rowspan=2 colspan=1>0.96</td><td></td></tr><tr><td rowspan=8 colspan=1>1.00.9score0.8emanticc0.7Se-0.6</td></tr><tr><td rowspan=1 colspan=1>Opus 5</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>0.76</td><td rowspan=1 colspan=1>0.71</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.79</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=1 colspan=1>Llama 3.3</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.82</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.93</td><td rowspan=1 colspan=1>0.97</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=1 colspan=1>Qwen 3.8</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.95</td><td rowspan=1 colspan=1>0.87</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.89</td><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.84</td><td rowspan=1 colspan=1>0.94</td><td rowspan=1 colspan=1>0.93</td></tr><tr><td rowspan=3 colspan=1>DeepSeek</td><td rowspan=3 colspan=1>0.91</td><td rowspan=3 colspan=1>0.89</td><td rowspan=3 colspan=1>0.85</td><td rowspan=3 colspan=1>0.76</td><td rowspan=3 colspan=1>0.91</td><td rowspan=3 colspan=1>0.96</td><td rowspan=3 colspan=1>0.71</td><td rowspan=3 colspan=1>0.84</td><td rowspan=1 colspan=1></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=2 colspan=1>0.87</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td rowspan=1 colspan=1>0.96</td><td rowspan=1 colspan=1>0.80</td><td rowspan=1 colspan=1>0.91</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.80</td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>C1</td><td rowspan=1 colspan=1>C2</td><td rowspan=1 colspan=1>C3</td><td rowspan=1 colspan=1>C4</td><td rowspan=1 colspan=1>C5</td><td rowspan=1 colspan=1>C6</td><td rowspan=1 colspan=1>C7</td><td rowspan=1 colspan=1>C8</td><td rowspan=1 colspan=1>C9</td><td rowspan=1 colspan=1>Ci0</td><td rowspan=1 colspan=3>Ci1Ci2Ci3</td><td rowspan=1 colspan=1>Ci4</td><td rowspan=1 colspan=1>-0.6</td></tr></table>

Figure 12: Per-case automated semantic scores under the with-location condition (see also Table 14). Grey cells mark missing candidates; scores remain high across the panel, and these values are automated diagnostics rather than human judgments.

Observation 1. In Crowded India, Llama 3.3 generates a short 8-state world where the robot can move directly from evidence\_analyzed to take\_action, a 2-step path from evidence to punishment. In the USA (Native American) world, the same complaint requires a 16-state chain. Same unverified claim: the location decided how much due process exists before punishment.

Observation 2. GPT-5.6 produces structurally similar worlds for USA original, USA (Native American), and Crowded India. But removing the location label entirely triggers an EXTREME restructure: the terminals change and the robot loses the initiate\_inquiry action. GPT needs the cultural anchor to commit to a process.

Observation 3. Qwen 3.8 generates its largest world (30 states) under the China (mixed religion) label, with branches like record\_documented\_material vs no\_material, inventing an evidence-sufficiency check that no other LLM uses. In the USA (Native American) world, the robot never acts at all. Across the C9 and C13 conditions, 15 (model, location) pairs reach the EXTREME level, meaning the set of reachable terminal outcomes differs. Every LLM shows at least one.

Summary. In C13, Llama gives a 2-step path to action under India and a 16-state chain under USA (Native American). Across both cases and all five LLMs, the robot never awards itself normative authority. But the location label systematically restructures the world model’s action space, process length, and terminal outcomes. No LLM wrote a stereotype word, but every LLM changed the rules of the world with the label: the actions the robot may take, the process it must respect, the endings it can reach, and how much agency it keeps.

If confirmed as bias, it is bias at the level of consequence, and it matters beyond this benchmark. As noted in Section 2, computing reversibility, reachability, or human power presupposes a world model that already contains the relevant stakeholders, options and consequences. If the set of reachable terminal outcomes shifts with a country label, any empowerment or reversibility quantity computed on the generated world model shifts with it. The bias does not stay in the representation. It propagates into the safety quantity the world model was built to support.

## C.3 Validation status and next step.

The consequence levels in Table 15 were assigned manually by one author. They are a qualitative reading of what changed between worlds, not a computed metric. Establishing that any of these differences constitutes bias requires a separate step, which we specify here and have not yet run. Each differing pair is presented to at least two annotators with the location labels removed, so that the annotator sees only the two world models and the scenario. The annotator labels each difference as justified by the scenario, unjustified, or ambiguous. Agreement is measured with Cohen’s κ, and pairs where annotators disagree are adjudicated by a third. Only differences rated unjustified by agreeing annotators are reported as bias. Until that is complete, the findings in this appendix are diagnostic: they establish that the location label changes the generated world model, not that the change is unwarranted.

Table 15: Bias-case matrix: state counts and consequence level per (model, location) pair for C9 and C13. Consequence levels: EXT = EXTREME (different reachable terminal outcomes), HIGH = robot affordance set changed, LOW = naming/decomposition differs only. “–” marks missing transcripts.
<table><tr><td>Case</td><td>Model</td><td>Original</td><td>India</td><td>China</td><td>Native Am.</td><td>No loc.</td></tr><tr><td>C9</td><td>GPT-5.6</td><td>8/LOW</td><td>10 /EXT</td><td>9/HIGH</td><td>7/LOW</td><td>12 / EXT</td></tr><tr><td>C9</td><td>Llama 3.3</td><td>14/HIGH</td><td>11 /EXT</td><td>10 /HIGH</td><td>9/EXT</td><td>8 /EXT</td></tr><tr><td>C9</td><td>Claude Opus 5</td><td>9/LOW</td><td>12 /HIGH</td><td>8/LOW</td><td>11 /HIGH</td><td>7/LOW</td></tr><tr><td>C9</td><td>Qwen 3.8</td><td>11/LOW</td><td>13 /LOW</td><td>10/LOW</td><td>8/HIGH</td><td>14 /EXT</td></tr><tr><td>C9</td><td>DeepSeek</td><td>9/HIGH</td><td>6 / EXT</td><td>11/HIGH</td><td>-1-</td><td>-1-</td></tr><tr><td>C13</td><td>GPT-5.6</td><td>8/LOW</td><td>9/LOW</td><td>7/LOW</td><td>8/LOW</td><td>12 / EXT</td></tr><tr><td>C13</td><td>Llama 3.3</td><td>10 /HIGH</td><td>8 /EXT</td><td>12 /HIGH</td><td>16 /EXT</td><td>9 / EXT</td></tr><tr><td>C13</td><td>Claude Opus 5</td><td>11/HIGH</td><td>14 /EXT</td><td>10 /HIGH</td><td>13 / EXT</td><td>8/LOW</td></tr><tr><td>C13</td><td>Qwen 3.8</td><td>9/LOW</td><td>11/HIGH</td><td>30 / EXT</td><td>7/HIGH</td><td>10/LOW</td></tr><tr><td>C13</td><td>DeepSeek</td><td>10/HIGH</td><td>6 /EXT</td><td>12 /HIGH</td><td>-1-</td><td>-1-</td></tr></table>

![](images/912e2f8980b13dbe5672ad779e6e99f7fe0ab8727a95933bda134516000aaff0.jpg)  
Figure 13: C9 individual state graphs for the (model, location) pairs. Top box: GPT-5.6 under Saudi Arabia (left) vs No location (right). Middle box: Llama 3.3 under No location. Bottom box (clockwise from top-left) : Qwen 3.8 under Saudi Arabia, India, USA + Native American, and China (mixed religion).

![](images/0de9529b7be02d36eb4414c095ba454dc58223f66ab4618296a7d452bf48f2c5.jpg)  
Figure 14: C13 individual state graphs for the (model, location) pairs discussed in observations. Top box: Llama 3.3 under Crowded India (up) vs USA (Native American) (down). Middle box (clockwise from top-left): GPT-5.6 under USA ONLY (original), USA + Native American, No Location and Crowded India. Bottom row: Qwen 3.8 under China (mixed religion) vs USA + Native American.

## NeurIPS Paper Checklist

## 1. Claims

Question: Do the main claims made in the abstract and introduction accurately reflect the paper’s contributions and scope?

Answer: [Yes]

Justification: The abstract and Introduction state the scope of the work as generating and evaluating LLM produced situational world models represented as stochastic games, and describe the benchmark, generation pipeline, and multi-lane evaluation framework.

## 2. Limitations

Question: Does the paper discuss the limitations of the work performed by the authors?

Answer: [Yes]

Justification: We clarify experimental validation is ongoing.

## 3. Theory assumptions and proofs

Question: For each theoretical result, does the paper provide the full set of assumptions and a complete (and correct) proof?

Answer: [N/A]

Justification: The paper does not introduce new theoretical results, theorems, or proofs. Formal definitions are used only to specify the stochastic-game representation and evaluation setting.

## 4. Experimental result reproducibility

Question: Does the paper fully disclose all the information needed to reproduce the main experimental results of the paper to the extent that it affects the main claims and/or conclusions of the paper (regardless of whether the code and data are provided or not)?

Answer: [Yes]

Justification: The paper and supplementary material specify the evaluated models, prompts, test cases, validation procedure, and evaluation methodology. Code, prompts, generated outputs, and experiment configurations are provided to enable replication of the reported results.

## 5. Open access to data and code

Question: Does the paper provide open access to the data and code, with sufficient instructions to faithfully reproduce the main experimental results, as described in supplemental material?

Answer: [Yes]

Justification: The repository provided includes instructions for reproducing the generation and evaluation pipeline.

## 6. Experimental setting/details

Question: Does the paper specify all the training and test details (e.g., data splits, hyperparameters, how they were chosen, type of optimizer) necessary to understand the results?

Answer: [Yes]

Justification: The paper reports three empirical evaluation tracks. Section 5 specifies the benchmark cases, model identities, prompt conditions, temperatures, generation counts, and first-pass/repair policy. The appendix and released repository provide the preserved prompts, per-run records, raw outputs, evaluation artifacts, and experiments/openrouter\_game\_study/config.yaml. No model training, optimizer, or train/test split is used: the study evaluates prompted hosted LLM outputs.

## 7. Experiment statistical significance

Question: Does the paper report error bars suitably and correctly defined or other appropriate information about the statistical significance of the experiments?

Answer: [Yes]

Justification: The paper reports descriptive uncertainty appropriate to the design. Game firstpass proportions include descriptive 95% Wilson intervals; structural results report medians and IQRs, with game-clustered bootstrap intervals available in the machine-readable results; and Table 6 reports within-pair F1 standard deviations and ranges across the three repeated provider requests. We do not report null-hypothesis significance tests, because the nine games are fixed benchmark fixtures and three repetitions per LLM–game pair are insufficient for population-level inference.

## 8. Experiments compute resources

Question: For each experiment, does the paper provide sufficient information on the computer resources (type of compute workers, memory, time of execution) needed to reproduce the experiments?

Answer: [No]

Justification: LLM generations were issued through OpenRouter; request configuration, model identifiers, token budgets, and per-request timing metadata are preserved in the released artifacts. The provider does not disclose the underlying accelerator type, memory, or serving hardware, and these were not controlled or recorded by the study. The local evaluation code and all required artifacts are released, but we therefore cannot provide a complete hardware-level compute accounting for hosted inference.

## 9. Code of ethics

Question: Does the research conducted in the paper conform, in every respect, with the NeurIPS Code of Ethics https://neurips.cc/public/EthicsGuidelines?

Answer: [Yes]

Justification: We do not use any human data in this paper.

## 10. Broader impacts

Question: Does the paper discuss both potential positive societal impacts and negative societal impacts of the work performed?

Answer: [N/A]

Justification: We don’t think the work has societal impact at this point. In the long run, this research agenda should of course help reducing biases and risks in AI decision making.

## 11. Safeguards

Question: Does the paper describe safeguards that have been put in place for responsible release of data or models that have a high risk for misuse (e.g., pre-trained language models, image generators, or scraped datasets)?

Answer: [N/A]

Justification: We can think of no way the current work poses such risks

## 12. Licenses for existing assets

Question: Are the creators or original owners of assets (e.g., code, data, models), used in the paper, properly credited and are the license and terms of use explicitly mentioned and properly respected?

Answer: [Yes]

Justification: We only use standard open source software packages and libraries.

## 13. New assets

Question: Are new assets introduced in the paper well documented and is the documentation provided alongside the assets?

Answer: [Yes]

Justification: Documentation is provided as part of the supplementary material

## 14. Crowdsourcing and research with human subjects

Question: For crowdsourcing experiments and research with human subjects, does the paper include the full text of instructions given to participants and screenshots, if applicable, as well as details about compensation (if any)?

Answer: [N/A]

Justification: No such things were performed.

## 15. Institutional review board (IRB) approvals or equivalent for research with human subjects

Question: Does the paper describe potential risks incurred by study participants, whether such risks were disclosed to the subjects, and whether Institutional Review Board (IRB) approvals (or an equivalent approval/review based on the requirements of your country or institution) were obtained?

Answer: [N/A]

Justification: No such activities were performed.

## 16. Declaration of LLM usage

Question: Does the paper describe the usage of LLMs if it is an important, original, or non-standard component of the core methods in this research? Note that if the LLM is used only for writing, editing, or formatting purposes and does not impact the core methodology, scientific rigor, or originality of the research, declaration is not required.

Answer: [Yes]

Justification: We describe the use of LLMs in detail.