# Procedural Memory Under Change: Reuse and Interference in Controlled Web Tasks

Yanze Cao

Independent Researcher

Xi’an, China

caoyanze426@gmail.com

## Abstract

Procedural memory lets language agents reuse successful routines, but reuse presumes that a stored routine remains applicable. We study what happens when that presumption is deliberately violated. The study combines a retrospective, human-assisted interface-adaptation case from BrowserGym TimeWarp with controlled frozen-memory comparisons on synthetic shopping decisions. During the documented WebShop V1–V6 development path, interface-specific code was adapted while the separately stored high-level procedure was not reported to change; this phase does not constitute an autonomous memoryagent evaluation. In the controlled phase, an early pilot produced one task on which two memory conditions selected a more expensive item while the no-memory condition selected the reference minimum. Follow-up probes did not establish a recurring row-order or identity-binding pattern. We then tested four forms of mismatch—changed quantities, a diferent evidence representation, a conflict between local and global optimization, and distributed promotion evidence—across 32 formal cells. Each cell used one temperature-0 generation with the same local qwen3:8b configuration and no adaptive retry. Across these pairs, none of the predefined diagnostic interference signatures appeared on the tasks for which they were defined when current-task evidence was explicit and suficient. The result identifies a tested region of non-interference: a procedural memory can be mismatched without becoming behaviorally disruptive. It does not establish general safety or a mechanism. The remaining question is which additional conditions turn applicability mismatch into observable, memory-caused error.

## 1 Introduction

A web agent learns that the cheapest acceptable purchase can be found by searching each requested category, rejecting semantic false positives, and summing the category minima. The procedure is useful until the task introduces a bundle discount. At that point the old routine remains coherent, executable, and previously successful, yet its local optimization rule no longer solves the current problem. This is the dificult case for long-term agent memory: previously useful memory whose applicability has changed.

Recent language agents store reflections, experiences, workflows, or executable skills and retrieve them to improve later decisions [5, 6, 8–10]. Work focused specifically on procedural memory makes these routines increasingly explicit and reusable [1, 4, 11]. Positive transfer is only half of the problem. A long-lived agent also needs to operate after interfaces, quantities, constraints, and optimization objectives change. Retrieval relevance does not guarantee procedural applicability, and procedural mismatch alone does not establish harm.

We separate three observations that are often collapsed:

$$
\mathrm { m e m o r y - t a s k ~ m i s m a t c h ~ \neq ~ \ o b s e r v a b l e ~ e r r o r ~ \neq ~ \ m e m o r y - c a u s e d ~ e r r o r . }\tag{1}
$$

A stored instruction can be incomplete or locally inappropriate while the final response remains correct. Conversely, an incorrect response under a memory condition is not by itself causal evidence against the memory. The same task may be dificult without memory, or the error may reflect identity binding, parsing, arithmetic, or another process. Figure 1 shows the evidential steps from mismatch to behavior consistent with interference.

![](images/91cc10dfd0ba121181a5e696a75e40d8725f362f27b7601bcf2d6b550c19e90f.jpg)  
Figure 1: From memory–task mismatch to behavioral interference. Mismatch concerns whether a procedure is applicable; interference concerns observed behavior. An error matching a stale procedure is evidence consistent with interference, not by itself a complete causal identification.

This paper asks: What additional conditions are required for mismatched procedural memory to produce observable interference? We approach the question through two linked phases. Phase 1 is a retrospective interface-adaptation case study based on TimeWarp Task 57 across WebShop V1–V6 [2, 3]. The retained Python baseline searches for almonds and rice crackers and extracts product titles and prices across interface variants. It does not load the stored memory, perform the final semantic selection and aggregation autonomously, or submit the answer. Phase 1 therefore provides provenance about a human-assisted development path, not an end-to-end memory-agent success claim.

Phase 2 uses frozen prompts and saved first responses to compare procedural-memory conditions. A nine-cell pilot exposes a concrete anomaly: on one task, the original and table-adapted memories yield \$36.80 while the no-memory condition yields the \$20.00 reference minimum. Eighteen follow-up cells test row-order and identity-binding explanations without establishing a recurring pattern. Batch 03 then sharpens the mismatch. Four paired task designs ask whether a memory omits changed quantities, misuses a diferent representation, follows independent local minima despite a bundle discount, or fails when discount evidence must be composed across fields. The four memory conditions include the original memory, no memory, a table-adapted instruction, and a semantic paraphrase of the original.

The main empirical result is deliberately narrow. Across all four Batch 03 pairs, none of the predefined interference signatures appears on the diagnostic tasks for which it was defined under the tested short-horizon conditions with explicit and suficient current-task evidence. Pair 01 and Pair 02 each yield eight of eight reference-compatible selections and totals under their historical evaluation boundary. Pair 03 yields correct global optimization in eight of eight cells; its six null strict-correctness fields remain unknown rather than being imputed. Pair 04 yields global optimization and strict task correctness in eight of eight cells, including explicit promotion-rule composition in all four distributed-evidence cells.

![](images/0a0c180d92c8ebf0b48ae0a91c806aa794a5e10eac24245f14dd8cdef32f5ff3.jpg)  
Figure 2: Experimental progression. Counts for the pilot and Batch 02 are descriptive reference-compatible outcomes. Pair 01/02 counts are not retrospective strict-correctness scores. Arrows denote the order of scientific motivation, not independent replications.

The study separates interface adaptation from high-level procedural revision, and memory–task mismatch from observable interference. The experimental sequence moves from an initial anomaly to targeted diagnostic conflicts involving quantity, representation, local versus global optimization, and distributed composition. Under explicit and suficient current-task evidence, these tested mismatches did not produce the predefined interference behaviors. The study does not determine whether the model ignored the memory, weakly weighted it, or corrected an intermediate bias. A direct next test would reduce the accessibility of current evidence while keeping the task objectively solvable.

## 2 Related Work

## 2.1 Reusing experience in language agents

Language-agent memory is often introduced as a route from isolated trials to cumulative improvement. Reflexion stores verbal feedback from prior attempts and reuses it in later trials [6]. ExpeL extracts naturallanguage insights and recalls both insights and experiences at inference time [10]. ReasoningBank distills strategies from successful and failed experience and couples their retrieval with test-time scaling [5]. These systems difer in how memories are produced and retrieved, but share an interest in carrying information across episodes.

Our focus is downstream of successful storage. Once a memory has been learned and retrieved, when is it still suitable for the present task? This question is distinct from whether memory improves average benchmark performance. A memory can be relevant to the task family while encoding an optimization assumption that is no longer valid.

## 2.2 Procedural memory and skill transfer

Procedural representations package experience as reusable action structure. Voyager accumulates an executable skill library for open-ended embodied exploration [8]. Agent Workflow Memory induces recurring workflows from trajectories and supplies selected workflows to later web-navigation decisions [9]. ProcMEM formalizes reusable skills with activation, execution, and termination structure [4]; AFTER studies control, adaptation, and transfer of procedures across enterprise tasks and model backbones [1]. Neural Procedural Memory instead represents procedures through activation steering and studies its complementarity with explicit workflows [11].

These approaches make applicability a first-class concern, whether through retrieval, activation conditions, verification, or transfer evaluation. The present work isolates a simpler behavioral question. We keep the supplied procedural text fixed, alter task requirements, and predefine response-level signatures of stale procedure use. The experiment does not evaluate a learned retrieval policy or a memory-update algorithm.

## 2.3 Web agents under change

Web agents face variation in layout, controls, observations, and content. BrowserGym provides a common environment and evaluation ecosystem for web-agent research [3]. TimeWarp turns web change into an explicit benchmark variable by supplying historical interface versions [2]. WebArXiv pursues time-invariant evaluation through fixed web snapshots and also reports a failure mode involving overly rigid reuse of interaction histories [7].

Our two phases occupy diferent points on this spectrum. The retrospective phase concerns adaptation of search, parsing, and action code under UI variation. The controlled phase removes live browsing and presents compact decision evidence directly, allowing procedural conflict and diagnostic errors to be specified exactly. It does not reproduce the observation uncertainty or long trajectories of active web interaction.

## 3 Problem Setup

Let M be a retrieved procedural memory and T the current task. A procedure contains a sequence of recommendations together with assumptions about observations, constraints, or objectives. We call M mismatched when at least one recommendation or assumption is not suficient for, or is locally inappropriate $\mathrm { t o } , T .$ This is an applicability relation between text and task; it does not depend on whether the agent follows the procedure.

For a task T, let $y ^ { * } ( T )$ denote the frozen reference outcome and let $s ( T )$ denote a predefined diagnostic signature. A signature is a response pattern expected if a particular stale instruction controls behavior. Examples include adding unit minima while omitting requested quantities, or selecting independent category minima while ignoring an eligible bundle discount. An observed response y is reference-compatible when its reported selection and total match $y ^ { * } ( T )$ under the evaluation fields historically available for that evidence block. It exhibits the diagnostic when $s ( T , y ) = 1$

We reserve observable interference for a task-level error that is consistent with the mismatched procedure. Even this label is behavioral: establishing that the memory caused the error requires a comparison such as a no-memory condition and adequate controls for alternative explanations. Our design therefore includes condition B, which supplies no procedural memory. The study reports memory-associated anomalies when an error occurs in memory conditions but not B, while avoiding a mechanistic conclusion from a single contrast.

The distinction matters because evaluation fields are not uniform across the historical sequence. Reference compatibility, global-optimization correctness, strict task correctness, and diagnostic signatures answer diferent questions. A null strict-correctness value means the saved answer lacks evidence required by that evaluator; it is not converted into either success or failure. Likewise, a pipeline that finds and parses products is not an end-to-end task solver.

The controlled tasks use two product categories, almonds and rice crackers. Candidate tables specify valid categories, item identifiers, prices, and—where relevant—quantities or promotions. The compact setting makes the stale-procedure signatures transparent. It also bounds the claims: these are short synthetic decisions with explicit evidence, not estimates over a web-task population.

## 4 Methods

## 4.1 Study design

The study has two phases with diferent evidential roles. Phase 1 is a retrospective, human-assisted case study of interface adaptation. Phase 2 is a sequence of frozen-prompt comparisons. Within Phase 2, the first pilot and Batch 02 precede Batch 03 and retain their own scoring histories. We do not pool the phases or batches into a common success rate.

The formal Phase 2 structure is:

First Frozen Batch Pilot 01 (T1–T3); Batch 02, comprising 02A Row-order (T4–T5), 02B Identity Binding (T6–T7), and 02C Replication-style probe (T8–T9); and Batch 03 Memory Content Interference, comprising Pair 01 (T10–T11), Pair 02 (T12–T13), Pair 03 (T14–T15), and Pair 04 (T16–T17).

“Replication-style” describes the related design of T8–T9. It is not stochastic replication, a multi-seed experiment, or a repeated random trial.

## 4.2 Phase 1 baseline

Phase 1 concerns BrowserGym TimeWarp Task 57 and WebShop versions V1–V6. The retained artifact is baseline\_v05.py. Direct code inspection shows that it searches for almonds and rice crackers, extracts product titles and prices, checks expected product counts, and handles several UI, parser, and locator variants. It does not load or execute the separately stored memory M1. It also does not autonomously perform final semantic product selection, minimum-cost aggregation, or answer submission.

The historical workflow combined user-run code, terminal or screenshot evidence, and joint user–assistant interpretation. Its record reports a V3 reader patch, a V4 locator and result-boundary patch, a V5 actionadapter patch, and no further V6 patch. The stored high-level procedure was not reported to change during this path. Per-version cryptographic snapshots were not retained; the unchanged-M1 statement is therefore retrospective provenance, not checksum proof across all six historical points.

## 4.3 Procedural-memory conditions

Condition A is the original frozen M1: a browser-oriented procedure that searches each category, rejects semantic false positives, records a valid minimum, and sums category minima. Its SHA256 is fb5722d2a66f9c2a2eaa2dbf43428ffb605d1f1e3da3cf559615ef1846ee34af. Condition <sup>B</sup> provides no procedural memory. Condition C supplies a table-adapted instruction that reads a complete candidate table and explicitly applies requested quantities. Condition D is a semantic paraphrase of M1, with SHA256 a3c2a576eb1a951ee0277adbee2566887399ad450ab47a2860ed5ddad425732e.

The contrasts are intentionally descriptive. C difers from A in quantity explicitness, interface assumptions, wording, and length, so A versus C is not a clean single-factor intervention. D was designed as a semantic paraphrase, but semantic intent does not imply identical model processing.

## 4.4 Pilot and follow-up probes

Pilot 01 crosses T1–T3 with A/B/C for nine cells. The tasks had been examined previously and are not held out. T2 requests four almond units and three rice-cracker units from a complete table; its reference total is \$20.00. The pilot’s purpose is to preserve the first responses and identify concrete failure patterns worth testing.

Batch 02 contains 18 cells. Row-order tasks T4–T5 vary the placement of preferred candidates. Identity Binding tasks T6–T7 separate item identity and price-binding hypotheses. The replication-style tasks T8–T9 instantiate a related probe without random repeats. These follow-ups are diagnostic variants, not independent samples from a task distribution.

## 4.5 Batch 03

Batch 03 crosses two tasks per pair with four conditions A–D, giving eight cells per pair and 32 formal cells in total. Table 1 states the manipulation and reference outcome.

All information required to solve each task is explicitly present. T17 is less direct than T16 because the solver must match A03 and R03 to Promotion group G1 and then apply the same-group discount rule. It does not require external search or recovery of missing evidence.

## 4.6 Generation setup

All 32 formal Batch 03 cells used the same local qwen3:8b configuration: temperature 0, inference seed 0, num\_predict=512, num\_ctx=16384, think=false, and stream=false. Each cell was executed once, with

Table 1: Batch 03 paired-task design. Diagnostic signatures apply only where the task creates the corresponding conflict.
<table><tr><td>Pair</td><td>Tasks</td><td>Manipulation</td><td>Reference outcome</td><td>Diagnostic failure</td></tr><tr><td>01</td><td>T10/T11</td><td>Quantities change from 1 + 1 to 4 + 3</td><td>T10: $5; T11: 4(2) + 3(3) = $17</td><td>T11 unit-minimum sum $5</td></tr><tr><td>02</td><td>T12/T13</td><td>Search-result blocks versus complete table; candidates</td><td>A03 + R02 = $5.10</td><td>Interface confusion, unnecessary search, or unsupported incompleteness</td></tr><tr><td>03</td><td>T14/T15</td><td>fixed No discount versus an A02+R02 $5 discount</td><td>T14: A01+R01=$8; T15: A02+R02=$7</td><td>T15 A01+R01=$8, or ignored discount</td></tr><tr><td>04</td><td>T16/T17</td><td>Direct eligible pair versus distributed Promotion-group evidence</td><td>A03+R03: 6 + 6 − 5 = $7</td><td>Local pair $8, ignored discount, or T17 composition error</td></tr></table>

one generation request, one model-call attempt, and no adaptive retry. The inference seed is a recorded reproducibility setting. Execution-order seeds, where present, only organize cell order.

Temperature-0 decoding and a recorded seed do not make the experiment a stochastic multi-seed replication. Robustness to repeated execution, alternative decoding policies, and sampling variation was not evaluated.

## 4.7 Evaluation

Evaluation separates response receipt, reference-compatible product selection and total, global optimization, arithmetic conditional on the reported choices, strict task correctness, and predefined diagnostic signatures. Pilot and Batch 02 descriptive counts were audited against saved first answers and references. Pair 01/02 retain historical task\_correctness=null fields and are reported through selection, total, and pair-specific diagnostics.

Pair 03 and Pair 04 use frozen evaluator specifications and saved evaluations. Under Pair 03’s extraction discipline, a quantity is not filled in from the task when the answer does not state it; this produces six null strict-correctness values. Pair 04 evaluation inputs were created by deterministic text-pattern extraction from saved answers, with raw text, field evidence, and source hashes retained. This is auditable extraction, not an independent blinded human annotation set. No evaluator was rerun for manuscript preparation.

## 5 Results

## 5.1 Phase 1

The retrospective record attributes failures to the interface layer: title/price parsing in V3, submission controls and result-region boundaries in V4, and the absence of a compatible search-button path in V5. Corresponding code changes add heading and price-node reading, icon-button and recommendation-boundary handling, and Enter-key submission. The V6 record reports no further patch. During this documented path, the separately stored high-level procedure was not reported revised.

This finding is about a human-assisted development history. The retained baseline searches and extracts candidates but neither reads M1 nor independently completes the final product decision. It therefore cannot support an autonomous six-version task-success claim. The supported observation is that local interface adaptation proceeded without a reported revision to the stored procedure.

## 5.2 Pilot and Batch 02

Pilot 01 yields reference-compatible optimal outcomes of 2/3 for A, 3/3 for B, and 2/3 for C. T1 and T3 are correct in all three conditions. On T2, A and C choose the more expensive almond candidate and report \$36.80, whereas B selects the reference minimum and reports \$20.00. The tasks had prior exposure, so this is not a held-out efect estimate.

Across Batch 02, the corresponding descriptive counts are $5 / 6$ for A and $6 / 6$ for each of B and C. T5\_A selects A01 but uses A02’s price. This is an ID–price or attribute-binding error, not an arithmetic error. The 02B identity-binding and 02C replication-style probes are correct in all 12 cells. Thus, the follow-ups do not establish recurring first-row following, A01 attraction, or identity–price binding. They also do not show that the earlier error was random; its cause remains unresolved.

## 5.3 Pair 01

All eight saved Pair 01 answers have reference-compatible selections and totals. T10 yields \$5. On T11, every condition explicitly computes $4 \times \ S 2 + 3 \times \ S 3 = \ S 1 7$ . The predefined quantity-omission signature, \$5 from adding unit minima without quantities, is absent. Historical task-correctness fields are null, so no strict score is assigned.

## 5.4 Pair 02

All eight saved Pair 02 answers choose A03 and R02 and report the \$5.10 reference total. Explicit interface confusion occurs in $0 / 8$ cells. The answers do not request unnecessary search and do not claim that the supplied information is incomplete. As in Pair 01, these are reference-compatible outcomes under the historical scoring boundary, not retrospectively assigned strict scores.

## 5.5 Pair 03

Global optimization is correct in $8 / 8$ cells. On control task T14, $\mathrm { A 0 1 + R 0 1 } { = } \mathfrak { S } 8$ is both the independent local minimum and the global optimum. On conflict task T15, all four conditions select A02+R02 and apply the \$5 discount, yielding the unique global optimum of \$7. The stale local-minimum signature $_ { \mathrm { A 0 1 + R 0 1 = 5 8 } }$ occurs in $0 / 4$ T15 cells, and the discount-ignored field is false in all four.

Strict task correctness remains separate: two cells are true and six are null. The null cells omit suficient explicit per-category quantity evidence under the frozen extraction discipline. They are neither six observed failures nor six successes and are not imputed.

## 5.6 Pair 04

Pair 04 records global-optimization correctness, strict task correctness, and arithmetic correctness given the chosen products in $8 / 8$ cells. Every answer selects A03+R03 and reports ${ \mathfrak { H } } 6 + { \mathfrak { H } } 6 - { \mathfrak { H } } 5 = { \mathfrak { H } } 7$ . The local-minimum signature and discount-ignored fields are false in all eight cells.

T17 distributes the promotion evidence across product attributes and a general rule. All four conditions explicitly identify the shared G1 Promotion group and explicitly apply the \$5 discount. Explicit composition errors occur in $0 / 4$ cells. No deterioration is visible relative to T16 in this pair. The no-memory condition matches the three memory conditions on the principal task-level metrics.

## 5.7 Batch 03 summary

Table 2 keeps the distinct evaluation boundaries visible. It intentionally omits a pooled task-success percentage.

Across the four Batch 03 pairs, none of the predefined interference signatures appeared on the diagnostic tasks for which they were defined under the tested short-horizon conditions with explicit and suficient currenttask evidence. This is a cross-pair behavioral observation, not proof that the model used any particular internal strategy.

## 6 Discussion

Across Batch 03, several forms of mismatch did not produce the predefined behavioral failures under the tested conditions. M1 does not state how to multiply arbitrary quantities, assumes a browser search process, and recommends independent category minima. Nevertheless, the specified quantity, representation, localoptimization, and distributed-composition errors do not occur in the controlled batch. A procedural memory can therefore be mismatched without becoming behaviorally disruptive in this tested setting.

Table 2: Results by evidence block. Counts are not interchangeable across historical scoring protocols.
<table><tr><td>Evidence block</td><td>Cells</td><td>Supported outcome</td><td>Boundary or diagnostic</td></tr><tr><td>Pilot 01</td><td>9</td><td>A 2/3; B 3/3; C 2/3</td><td>Previously examined tasks; T2 anomaly retained</td></tr><tr><td>Batch 02</td><td>18</td><td> $\textrm { A } 5 / 6 ; \textrm { B } 6 / 6 ; \textrm { C } 6 / 6$ </td><td>T5_A binding error; follow-ups are related probes</td></tr><tr><td>Pair 01</td><td>8</td><td>8/8 reference-compatible selections and totals</td><td>T11 quantity omission absent; no retrospective strict score</td></tr><tr><td>Pair 02</td><td>8</td><td>8/8 reference-compatible selections and totals</td><td>Explicit interface confusion 0/8; no retrospective strict score</td></tr><tr><td>Pair 03</td><td>8</td><td>Global optimization 8/8</td><td>Strict correctness 2 true, 6 null; T15 local signature 0/4</td></tr><tr><td>Pair 04</td><td>8</td><td>Global optimization and strict task correctness 8/8</td><td>T17 explicit composition error 0/4</td></tr></table>

The earlier anomalies remain part of the evidence. T2 produces a higher-cost choice in two memory conditions but not in the no-memory condition. T5\_A produces a preserved identity–price binding error. Later probes do not reproduce simple first-row, A01-attraction, or identity-binding patterns, but successful follow-ups do not retroactively erase those observations. The evidence supports an unresolved anomaly followed by increasingly direct tests that did not establish the proposed recurring signatures.

Current-task evidence may be a boundary variable. Even the distributed-evidence task keeps the decisive product attributes and discount rule explicit, complete, and nearby in a short prompt. Such evidence may be suficient for correct behavior despite a mismatched procedure. Other explanations fit equally well: the memory may have been weakly influential, its framing may not have been salient enough, or a partial bias may not have survived to the final response.

The present experiments cannot determine whether the model ignored the memory, weakly weighted it, or corrected an intermediate bias before producing its answer. Explanations in the saved responses are behavioral evidence, not access to a hidden process. In particular, explicit mention of Promotion group G1 supports the composition-evaluation field, but it does not establish an internal compensation mechanism.

The results also argue for keeping evaluation levels separate. Response receipt, a correct product pair, explicit quantities, arithmetic, and strict task correctness can diverge. Pair 03’s null fields expose this distinction: the response can select the global optimum while omitting evidence that a strict evaluator requires. Treating null as failure would invent an optimization error; treating it as success would overstate answer completeness.

A natural next study would reduce the accessibility or strength of current evidence while keeping the task objectively solvable and retaining a no-memory control. This tests the role of current evidence more directly than another nominal task variation. Longer trajectories, stronger memory salience, multiple competing memories, and alternative models are also relevant, but each changes a diferent dimension and deserves a separately frozen design. The next question is which additional conditions make a mismatched procedure behaviorally disruptive.

## 7 Limitations

Phase 1 is retrospective and human-assisted. Answer exposure, incomplete contemporaneous logs, and the retained baseline’s limited scope prevent an autonomous memory-benefit claim. The stored procedure’s reported continuity is a provenance statement; per-version cryptographic snapshots were not retained.

Batch 03 uses one local model, temperature 0, inference seed 0, and one formal execution per cell. It does not test repeated-execution robustness, stochastic sampling variation, alternative decoding policies, or cross-model transfer. The observations do not estimate a population-level failure rate or support statistical

equivalence.

The controlled tasks are short synthetic shopping decisions with explicit evidence. They omit long browser trajectories, noisy observations, active evidence acquisition, and accumulation of small biases over repeated actions. Even T17 requires only local composition of complete supplied evidence.

The memory conditions cover a narrow lineage. Condition C changes more than interface wording, and condition D cannot guarantee equivalent internal processing. Behavioral outputs do not identify mechanism, and the predefined signatures cannot detect every subtler influence of memory.

## 8 Conclusion

We examined procedural memory under change through a retrospective interface-adaptation case and controlled frozen-memory comparisons. The retrospective phase documents interface-specific code adaptation without supporting an autonomous M1 success claim. In the controlled phase, early anomalies motivated four increasingly direct mismatch pairs. Across 32 one-shot Batch 03 cells, none of the predefined diagnostic interference signatures appeared under the tested short-horizon conditions with explicit and suficient currenttask evidence.

The finding marks a tested region of non-interference, not a universal absence of harm. Applicability mismatch did not by itself produce the expected behavior, and the mechanism remains unknown. The next question is which additional conditions make a mismatched procedure behaviorally disruptive.

## References

[1] Julia Belikova, Rauf Parchiev, Evgeny Egorov, Grigorii Davydenko, Gleb Gusev, Andrey Savchenko, and Maksim Makarenko. Managing procedural memory in LLM agents: Control, adaptation, and evaluation. arXiv preprint arXiv:2606.23127, 2026. URL https://arxiv.org/abs/2606.23127.

[2] Md Farhan Ishmam and Kenneth Marino. TimeWarp: Evaluating web agents by revisiting the past. arXiv preprint arXiv:2603.04949, 2026. URL https://arxiv.org/abs/2603.04949.

[3] Thibault Le Sellier de Chezelles, Maxime Gasse, Alexandre Lacoste, Massimo Caccia, Alexandre Drouin, Léo Boisvert, Megh Thakkar, Tom Marty, Rim Assouel, Sahar Omidi Shayegan, Lawrence Keunho Jang, Xing Han Lù, Ori Yoran, Dehan Kong, Frank F. Xu, Siva Reddy, Graham Neubig, Quentin Cappart, Russ Salakhutdinov, and Nicolas Chapados. The BrowserGym ecosystem for web agent research. Transactions on Machine Learning Research, 2025. URL https://openreview.net/forum?id=5298fKGmv3.

[4] Qirui Mi, Zhijian Ma, Mengyue Yang, Haoxuan Li, Yisen Wang, Haifeng Zhang, and Jun Wang. ProcMEM: Learning reusable procedural memory from experience via non-parametric PPO for LLM agents. arXiv preprint arXiv:2602.01869, 2026. URL https://arxiv.org/abs/2602.01869.

[5] Siru Ouyang, Jun Yan, I-Hung Hsu, Yanfei Chen, Ke Jiang, Zifeng Wang, Rujun Han, Long T. Le, Samira Daruki, Xiangru Tang, Vishy Tirumalashetty, George Lee, Mahsan Rofouei, Hangfei Lin, Jiawei Han, Chen-Yu Lee, and Tomas Pfister. ReasoningBank: Scaling agent self-evolving with reasoning memory. In The Fourteenth International Conference on Learning Representations, 2026. URL https: //openreview.net/forum?id=jL7fwchScm.

[6] Noah Shinn, Federico Cassano, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems, volume 36, pages 8634–8652, 2023. URL https://proceedings.neurips.cc/paper\_files/ paper/2023/hash/1b44b878bb782e6954cd888628510e90-Abstract-Conference.html.

[7] Zihao Sun and Ling Chen. WebArXiv: Evaluating multimodal agents on time-invariant arXiv tasks. arXiv preprint arXiv:2507.00938, 2025. URL https://arxiv.org/abs/2507.00938.

[8] Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research, 2024. URL https://openreview.net/forum?id=ehfRiF0R3a.

[9] Zora Zhiruo Wang, Jiayuan Mao, Daniel Fried, and Graham Neubig. Agent Workflow Memory. In Proceedings of the 42nd International Conference on Machine Learning, volume 267 of Proceedings of Machine Learning Research, pages 63897–63911, 2025. URL https://proceedings.mlr.press/v267/ wang25bx.html.

[10] Andrew Zhao, Daniel Huang, Quentin Xu, Matthieu Lin, Yong-Jin Liu, and Gao Huang. ExpeL: LLM agents are experiential learners. Proceedings of the AAAI Conference on Artificial Intelligence, 38(17): 19632–19642, 2024. doi: 10.1609/aaai.v38i17.29936.

[11] Chengfeng Zhao, Yuqiao Tan, Shizhu He, Yequan Wang, Jun Zhao, and Kang Liu. Neural Procedural Memory: Empowering LLM agents with implicit activation steering. arXiv preprint arXiv:2606.29824, 2026. URL https://arxiv.org/abs/2606.29824.

## A Frozen Procedural Memories

This appendix reproduces the three supplied memory texts used in the paper. Condition B has no proceduralmemory text.

## A.1 Original M1

Memory ID: M1. Source Environment: WebShop V1 / webshop2000. Source Task: TimeWarp Task 57. Memory Type: Procedural.

Goal Pattern: Find the minimum total cost for multiple requested product types.

Procedure: (1) Identify each requested product type separately. (2) Locate the site product search function. (3) Search for one target product type. (4) Inspect all returned candidates. (5) Reject candidates that only match keywords but are semantically the wrong product. (6) Among valid candidates, record the lowest price. (7) Repeat for every requested product type. (8) Sum the selected minimum prices. (9) Return the total.

Constraints: Do not rely on browser element IDs, screen coordinates, or button positions. Do not store old prices or the previous answer. If product relevance is ambiguous, inspect product title/details before accepting it.

SHA256: fb5722d2a66f9c2a2eaa2dbf43428ffb605d1f1e3da3cf559615ef1846ee34af.

## A.2 Table-adapted condition C

Memory ID: TABLE\_PROCEDURE\_C\_v1. Derived from: frozen M1; fixed candidate-table interface condition. Memory Type: Procedural.

Goal Pattern: Find the minimum total cost for multiple requested product types.

Procedure: (1) Identify each requested product type separately. (2) Read the complete candidate table already provided in the task. (3) Locate the candidates for one target product type in that table. (4) Inspect all provided candidates. (5) Reject candidates that only match keywords but are semantically the wrong product. (6) Among valid candidates, record the lowest price. (7) Repeat for every requested product type. (8) Calculate each selected product type’s cost using the quantity requested in the task, then sum those costs. (9) Return the total.

Constraints: Do not store old prices or the previous answer. If product relevance is ambiguous, inspect the supplied title/details before accepting it; if the supplied information is insuficient, state that limitation.

Condition C changes interface assumptions and explicitly introduces quantity handling. It is not an interface-wording-only edit of A.

## A.3 Semantic paraphrase D

Memory ID: M1\_PARAPHRASE\_D\_v1. Source Environment: WebShop V1 / webshop2000.   
Source Task: TimeWarp Task 57. Memory Type: Procedural.

Goal Pattern: Determine the minimum combined cost for several requested product categories.

Procedure: (1) Separate the request into its individual target product categories. (2) Find the productsearch capability provided by the site. (3) Query the site for one target category at a time. (4) Review every candidate returned for that category. (5) Exclude items that match the search words but do not actually belong to the intended product category. (6) From the remaining valid candidates, keep the one with the lowest price. (7) Perform the same process for each requested product category. (8) Add together the minimum prices selected for all requested categories. (9) Report the resulting total cost.

Constraints: Do not depend on browser element identifiers, fixed screen coordinates, or button placement. Do not retain old prices or the previous answer. When it is unclear whether a candidate is relevant, inspect its title or available details before deciding.

SHA256: a3c2a576eb1a951ee0277adbee2566887399ad450ab47a2860ed5ddad425732e. The file was designed as a semantic paraphrase of M1; the study does not assume identical internal processing.

## A.4 Hashes and provenance

The frozen memory files are stored at the following repository-relative paths:

• M1: research\_memory/M1\_v1.txt

• C: experiments/INTERFACE\_FIT\_FIRST\_FROZEN\_PILOT\_01/C\_procedural\_memory.txt

• D: experiments/MEMORY\_CONTENT\_INTERFERENCE\_BATCH\_03/D\_MEMORY\_v1.txt

The repository checksum manifests and the writing-time evidence ledger map these bytes to their archived uses. The Phase 1 statement that M1 was unchanged is retrospective because no per-version cryptographic snapshots survive.

## B Task Construction

## B.1 Summary of T1–T17

Table 3: Task inventory. “RC” denotes the frozen reference total.
<table><tr><td>Task</td><td>Evidence block</td><td>Role</td><td>RC</td></tr><tr><td>T1</td><td>Pilot</td><td>Previously examined control</td><td>$21.40</td></tr><tr><td>T2</td><td>Pilot</td><td>Four almonds, three rice crackers; anomaly task</td><td>$20.00</td></tr><tr><td>T3</td><td>Pilot</td><td>Previously examined control</td><td>$22.10</td></tr><tr><td>T4</td><td>Batch 02A</td><td>Row-order control/variant</td><td>archived reference</td></tr><tr><td>T5</td><td>Batch 02A</td><td>Row-order variant; T5_A binding anomaly</td><td>archived reference</td></tr><tr><td>T6</td><td>Batch 02B</td><td>Identity-binding probe</td><td>archived reference</td></tr><tr><td>T7</td><td>Batch 02B</td><td>Identity-binding probe</td><td>archived</td></tr><tr><td>T8</td><td>Batch 02C</td><td>Replication-style related design</td><td>reference archived</td></tr><tr><td>T9</td><td>Batch 02C</td><td>Replication-style related design</td><td>reference archived</td></tr><tr><td>T10</td><td>Pair 01</td><td>One almond and one rice-cracker unit</td><td>reference $5.00</td></tr><tr><td>T11</td><td>Pair 01</td><td>Four almond and three rice-cracker units</td><td>$17.00</td></tr><tr><td>T12</td><td>Pair 02</td><td>Search-result-style category blocks</td><td>$5.10</td></tr><tr><td>T13</td><td>Pair 02</td><td>Complete combined candidate table</td><td>$5.10</td></tr><tr><td>T14</td><td>Pair 03</td><td>No-discount control; local=global optimum</td><td>$8.00</td></tr><tr><td>T15</td><td>Pair 03</td><td>Local/global conflict via A02+R02 discount</td><td>$7.00</td></tr><tr><td>T16</td><td>Pair 04</td><td>Direct statement of eligible promotion pair</td><td>$7.00</td></tr><tr><td>T17</td><td>Pair 04</td><td>Distributed Promotion-group composition</td><td>$7.00</td></tr></table>

## B.2 Representative prompts

The frozen prompts are in Chinese. For readability, the following are faithful English renderings of their decision-critical content, including the complete candidate data and rules used to derive the reference. The archived original-language task.txt files remain authoritative.

T2 (pilot anomaly). Buy four units of almonds and three units of rice crackers from a complete synthetic table. Each unit is an indivisible sales unit; confirmed category labels determine validity; repeated purchase of one item is allowed; stock is suficient; prices are fixed; internal package count and weight do not make units equivalent; shipping, tax, discounts, and gifts are ignored. Candidates are A01 almonds at \$7.40, A02 other/cracker at \$1.10, A03 almonds at \$3.20, R01 rice crackers at \$8.10, R02 rice crackers at \$2.40, and X01 other/cereal bar at \$0.80. Report item IDs, quantities, rationale, calculation, and total. The reference is 4(3.20) + 3(2.40) = \$20.00.

T11 (quantity diagnostic). Buy four almond units and three rice-cracker units. Candidate almonds are A03 \$2, A02 \$4, A01 \$5, A04 \$6; rice crackers are R01 \$5, R02 \$3, R03 \$6, R04 \$7. The same indivisible-unit, fixed-price, suficient-stock, category-label, no-discount, and no-weight-normalization rules apply. Report IDs, quantities, rationale, calculation, and total. The reference is A03 and R02 with 4(2) + 3(3) = \$17.

T15 (local/global conflict). Buy exactly one almond unit and one rice-cracker unit. Candidate almonds are A01 \$4, A02 \$6, A03 \$7.50, A04 \$9; rice crackers are R01 \$4, R02 \$6, R03 \$7, R04 \$8.50. Exactly the pair A02+R02 receives a \$5 discount; no other discount applies. The table is complete and additional search is disallowed. Report IDs, quantities, rationale, calculation, and final total. Independent category minima give A01+R01=\$8, while the unique global optimum is A02+R02 with 6 + 6 − 5 = \$7.

T17 (distributed composition). Buy exactly one almond unit and one rice-cracker unit. Candidates and (price, Promotion group) are A01 (\$4, GA1), A02 (\$5, GA2), A03 (\$6, G1), A04 (\$7, GA4), R01 (\$4, GR1), R02 (\$5, GR2), R03 (\$6, G1), and R04 (\$7, GR4). If the selected products’ Promotion groups match exactly, subtract \$5; otherwise no discount applies. The complete table supplies all evidence and no browsing is needed. Report both IDs, both quantities, rationale, calculation, and total. The unique global optimum is A03+R03 with 6 + 6 − 5 = \$7.

## C Predefined Interference Signatures

Table 4: Diagnostic signatures and their interpretation. “Predefined” is used only for signatures supported by frozen design or pre-freeze records.
<table><tr><td>Pair</td><td>Task</td><td>Signature</td><td>Interpretation</td></tr><tr><td>01</td><td>T11</td><td>Final total $5</td><td>Adds unit minima $2+$3 while omitting quantities 4 and 3</td></tr><tr><td>02</td><td>T12/T13</td><td>Unnecessary search request, unsupported incompleteness, or explicit interface confusion</td><td>Applies a browser/search assumption despite complete supplied evidence</td></tr><tr><td>03</td><td>T15</td><td>A01+R01=$8; discount ignored</td><td>Follows independent category minima rather than the unique discounted global optimum</td></tr><tr><td>04</td><td>T16/T17</td><td>A01+R01=$8 or discount ignored</td><td>Falls back to independent local minima despite the promotion</td></tr><tr><td>04</td><td>T17</td><td>Incorrect Promotion-group match or failure to apply the same-group rule</td><td>Fails to compose A03→G1, R03→G1, and the general $5 rule</td></tr></table>

The A01+R01 selection is correct on T14 and is therefore not interference in that control. A raw signature field must be interpreted relative to the task that creates the procedural conflict. More generally, a signature is evidence about a specified behavior; absence of the signature does not rule out all memory efects.

## D Complete Cell-Level Results

The tables below preserve each block’s historical metrics. “Reference-compatible” means the saved selection and total match the archived reference; it is not a substitute for a later strict evaluator.

Table 5: All nine pilot cells.
<table><tr><td>Task</td><td>Condition</td><td>Reference</td><td>Saved total</td><td>Ref.-compatible</td></tr><tr><td>T1</td><td>A</td><td>$21.40</td><td>$21.40</td><td>yes</td></tr><tr><td>T1</td><td>B</td><td>$21.40</td><td>$21.40</td><td>yes</td></tr><tr><td>T1</td><td>C</td><td>$21.40</td><td>$21.40</td><td>yes</td></tr><tr><td>T2</td><td>A</td><td>$20.00</td><td>$36.80</td><td>no</td></tr><tr><td>T2</td><td>B</td><td>$20.00</td><td>$20.00</td><td>yes</td></tr><tr><td>T2</td><td>C</td><td>$20.00</td><td>$36.80</td><td>no</td></tr><tr><td>T3</td><td>A</td><td>$22.10</td><td>$22.10</td><td>yes</td></tr><tr><td>T3</td><td>B</td><td>$22.10</td><td>$22.10</td><td>yes</td></tr><tr><td>T3</td><td>C</td><td>$22.10</td><td>$22.10</td><td>yes</td></tr></table>

On T2, A and C select the more expensive almond candidate. B selects the reference minimum. The pilot tasks had previously been examined.

## D.2 Batch 02

Table 6: All 18 Batch 02 cells. The archived task-specific references are used rather than reproduced as one artificial common metric.
<table><tr><td>Block</td><td>Task</td><td>Condition</td><td>Reference-compatible</td></tr><tr><td>02A</td><td>T4</td><td>A</td><td>yes</td></tr><tr><td>02A</td><td>T4</td><td>B</td><td>yes</td></tr><tr><td>02A</td><td>T4</td><td>C</td><td>yes</td></tr><tr><td>02A</td><td>T5</td><td>A</td><td>no: ID-price binding</td></tr><tr><td>02A</td><td>T5</td><td>B</td><td>yes</td></tr><tr><td>02A</td><td>T5</td><td>C</td><td>yes</td></tr><tr><td>02B</td><td>T6</td><td>A</td><td>yes</td></tr><tr><td>02B</td><td>T6</td><td>B</td><td>yes</td></tr><tr><td>02B</td><td>T6</td><td>C</td><td>yes</td></tr><tr><td>02B</td><td>T7</td><td>A</td><td>yes</td></tr><tr><td>02B</td><td>T7</td><td>B</td><td>yes</td></tr><tr><td>02B</td><td>T7</td><td>C</td><td>yes</td></tr><tr><td>02C</td><td>T8</td><td>A</td><td>yes</td></tr><tr><td>02C</td><td>T8</td><td>B</td><td>yes</td></tr><tr><td>02C</td><td>T8</td><td>C</td><td>yes</td></tr><tr><td>02C</td><td>T9</td><td>A</td><td>yes</td></tr><tr><td>02C</td><td>T9</td><td>B</td><td>yes</td></tr><tr><td>02C</td><td>T9</td><td>C</td><td>yes</td></tr></table>

T5\_A selects A01 but uses A02’s price. It is not classified as an arithmetic error. Batch 02C is a replication-style probe only; the cells are not repeated random trials.

## D.3 Batch 03 Pair 01

All four T11 answers explicitly state $4 \times 2 + 3 \times 3 = 1 7 .$

## D.4 Batch 03 Pair 02

No cell makes an unnecessary search request or an unsupported claim that the supplied information is incomplete.

Table 7: Pair 01 cell-level results. Historical strict task correctness remains null.
<table><tr><td>Task</td><td>Cond.</td><td>Selection</td><td>Total</td><td>Ref.-compatible</td><td>Quantity omission</td></tr><tr><td>T10</td><td>A</td><td> $\mathrm { A 0 3 { + } R 0 2 }$ </td><td>$5</td><td>yes</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T10</td><td>B</td><td>A03+R02</td><td>$5</td><td>yes</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T10</td><td>C</td><td>A03+R02</td><td>$5</td><td>yes</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T10</td><td>D</td><td>A03+R02</td><td>$5</td><td>yes</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T11</td><td>A</td><td>A03+R02</td><td>$17</td><td>yes</td><td>no</td></tr><tr><td>T11</td><td>B</td><td>A03+R02</td><td>$17</td><td>yes</td><td>no</td></tr><tr><td>T11</td><td>C</td><td>A03+R02</td><td>$17</td><td>yes</td><td>no</td></tr><tr><td>T11</td><td>D</td><td>A03+R02</td><td>$17</td><td>yes</td><td>no</td></tr></table>

Table 8: Pair 02 cell-level results. Historical strict task correctness remains null.
<table><tr><td>Task</td><td>Cond.</td><td>Selection</td><td>Total</td><td>Ref.-compatible</td><td>Interface confusion</td></tr><tr><td>T12</td><td>A</td><td> $\mathrm { A 0 3 + R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T12</td><td>B</td><td> $\mathrm { A 0 3 + R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T12</td><td>C</td><td> $\mathrm { A 0 3 + R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T12</td><td>D</td><td> $\mathrm { A 0 3 { + } R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T13</td><td>A</td><td>A03+R02</td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T13</td><td>B</td><td> $\mathrm { A 0 3 + R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T13</td><td>C</td><td> $\mathrm { A 0 3 + R 0 2 }$ </td><td>$5.10</td><td>yes</td><td>no</td></tr><tr><td>T13</td><td>D</td><td>A03+R02</td><td>$5.10</td><td>yes</td><td>no</td></tr></table>

## D.5 Batch 03 Pair 03

Table 9: Pair 03 cell-level frozen-evaluator results. GO is global-optimization correctness; LM is the raw local-minimum signature.
<table><tr><td>Task</td><td>Cond.</td><td>Selection</td><td>Total</td><td>GO</td><td>Strict</td><td>LM</td><td>Discount ignored</td></tr><tr><td>T14</td><td>A</td><td> $\mathrm { A 0 1 + R 0 1 }$ </td><td>$8</td><td>true</td><td>null</td><td>true</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T14</td><td>B</td><td> $\mathrm { A 0 1 + R 0 1 }$ </td><td>$8</td><td>true</td><td>null</td><td>true</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T14</td><td>C</td><td> $\mathrm { A 0 1 + R 0 1 }$ </td><td>$8</td><td>true</td><td>true</td><td>true</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T14</td><td>D</td><td> $\mathrm { A 0 1 + R 0 1 }$ </td><td>$8</td><td>true</td><td>null</td><td>true</td><td> $\mathrm { n / a }$ </td></tr><tr><td>T15</td><td>A</td><td> $\mathrm { A 0 2 + R 0 2 }$ </td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td></tr><tr><td>T15</td><td>B</td><td>A02+R02</td><td>$7</td><td>true</td><td>null</td><td>false</td><td>false</td></tr><tr><td>T15</td><td>C</td><td> $\mathrm { A 0 2 + R 0 2 }$ </td><td>$7</td><td>true</td><td>null</td><td>false</td><td>false</td></tr><tr><td>T15</td><td>D</td><td>A02+R02</td><td>$7</td><td>true</td><td>null</td><td>false</td><td>false</td></tr></table>

The T14 local-minimum signature is expected and correct because the local and global optima coincide. The six strict nulls reflect missing explicit quantity evidence, not a wrong choice or total.

## D.6 Batch 03 Pair 04

Arithmetic correctness given the chosen products is true in all eight cells. In all four T17 cells, both Promotion-group matching and application of the group rule are explicit.

## E Execution and Reproducibility

## E.1 Formal Batch 03 configuration

The writing audit checked the actual request and result records indexed in paper/CELL\_INDEX.csv. That index records all formal cells, output directories, request counts, attempts, settings, and source hashes. It is a manuscript-audit artifact rather than a prospective experiment manifest.

Table 10: Pair 04 cell-level frozen-evaluator results. GO is global-optimization correctness; TC is strict task correctness; LM is the local-minimum signature; CE is explicit composition error.
<table><tr><td>Task</td><td>Cond.</td><td>Selection</td><td>Total</td><td>GO</td><td>TC</td><td>LM</td><td>Discount ignored</td><td>CE</td></tr><tr><td>T16</td><td>A</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>n/a</td></tr><tr><td>T16</td><td>B</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>n/a</td></tr><tr><td>T16</td><td>C</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>n/a</td></tr><tr><td>T16</td><td>D</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>n/a</td></tr><tr><td>T17</td><td>A</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>false</td></tr><tr><td>T17</td><td>B</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>false</td></tr><tr><td>T17</td><td>C</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>false</td></tr><tr><td>T17</td><td>D</td><td>A03+R03</td><td>$7</td><td>true</td><td>true</td><td>false</td><td>false</td><td>false</td></tr></table>

Table 11: Configuration shared by all 32 Batch 03 formal cells.
<table><tr><td>Field</td><td>Value</td></tr><tr><td>Model</td><td>qwen3:8b</td></tr><tr><td>Temperature</td><td>0</td></tr><tr><td>Inference seed</td><td>0</td></tr><tr><td>num_predict</td><td>512</td></tr><tr><td>num_ctx</td><td>16384</td></tr><tr><td>think</td><td>false</td></tr><tr><td>stream</td><td>false</td></tr><tr><td>Generation requests per cell Model-call attempts per cell</td><td>1</td></tr><tr><td></td><td>1</td></tr><tr><td>Adaptive retries</td><td>0</td></tr></table>

## E.2 Pair 04 runtime plumbing

The Pair 04 on-disk manifests did not contain the inference-seed field expected by the execution path. After frozen validation succeeded, the runner made a shallow in-memory copy and inserted the already frozen inference seed of 0 from the Pair 04 run-order configuration. The adaptation did not write to the manifest, change a task or prompt, relax hash validation, or alter the seed. Archived execution\_manifest.json and turn\_001\_request.json files record seed 0 in every cell. This is execution plumbing, not an experimenta intervention.

## E.3 Scope of reproducibility

Each formal cell was evaluated once. The recorded seed supports reconstruction of the intended call configuration, but no claim is made about bitwise reproducibility across runtimes. Repeated execution, alternative decoding, stochastic sampling, and cross-model validation were outside the frozen study.

## F Provenance and Integrity

The repository preserves frozen task files, memory files and hashes, reference answers, evaluator specifications, run orders, raw responses, evaluated records, and post-run checksum reports. The principal frozen Batch 03 synthesis is experiments/MEMORY\_CONTENT\_INTERFERENCE\_BATCH\_03/BATCH\_03\_FINAL\_ANALYSIS.md, SHA256 0c93f9142d16567d77f9c3a42db8669f7dba14bd7fd9ca88981f8c254e8f4b4a. It was read but not modified during manuscript preparation.

For Pair 04, PAIR\_04\_POSTRUN\_INTEGRITY.json reports 3,405 existing files before formal execution and confirms that all 3,405 remained unchanged afterward: changed 0, missing 0. Each of the eight per-cell frozen\_check.json records remains true. The accompanying PAIR\_04\_POSTRUN\_SHA256SUMS.txt preserves file-level fingerprints.

paper/EVIDENCE\_LEDGER.md and paper/CELL\_INDEX.csv were created during manuscript auditing to map claims to archived artifacts; they are not treated as prospectively frozen experimental materials. The intended chain is:

paper claim −→ evidence-ledger entry −→ actual frozen source artifact.

The ledger groups evidence as E1 Phase 1 provenance, E2 memory definitions, E3 pilot, E4 Batch 02, E5–E8 Batch 03 Pairs 01–04, and E9 frozen final analysis. A later writing-session integrity snapshot covers a broader set of pre-existing files; it is an audit of manuscript preparation, not a reconstruction of missing historical evidence.

No frozen experiment artifact, baseline, memory, prompt, raw answer, evaluator result, or checksum manifest was changed to produce this paper. No model was run, no evaluator was rerun, and no new experimental condition was created.

## F.1 Data and code availability

The local repository contains the audit trail described above. No public repository upload or archival deposition is claimed by this draft.