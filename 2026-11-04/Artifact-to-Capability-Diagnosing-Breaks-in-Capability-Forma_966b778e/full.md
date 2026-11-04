# Artifact-to-Capability: Diagnosing Breaks in Capability Formation of Self-Evolving Agents in a Controlled Environment

Ziqian Ban<sup>1</sup>, Haoxiang Yang<sup>1</sup>, Yanfei Zhang<sup>2</sup> Shixin Liang<sup>1</sup>, Haoran Yu<sup>1</sup>, Xu Lin<sup>3</sup> Shaobin Zhao<sup>1</sup>, Xuran Li<sup>1</sup>, Zhiyi Cheng<sup>1</sup>   
<sup>1</sup>Tianjin University <sup>2</sup>Independent Researcher   
<sup>3</sup>International Digital Economy Academy (IDEA)

## Abstract

Self-evolving agents write interaction experience into artifacts and reuse them in later tasks. Most work supports such methods with three proxy signals: artifacts accumulate, behavior changes once they are loaded, and metrics rise. We call this signal-based evaluation. What these signals are meant to indicate, however, has not been defined. To fill this gap, we define capability formation from artifacts as five sequentially dependent properties, two of which form a checkpoint for capability formation, and introduce three-scale diagnosis, which judges these properties at the state, game, and lineage levels. In freestyle Gomoku, Reflexion, Voyager, and DGM evolve with the same base model and all show the three signals, yet each exhibits a distinct capability-formation break. Reflexion shows a legalization gain: its pass rate rises only because more of its moves are legal. Voyager shows a compositional override: components that are sound on their own combine to miss all 20 available one-move wins. DGM shows loss-only drift: after passing the checkpoint, later versions only lose passing states. These counterexamples show that signalbased evaluation cannot determine whether a capability has formed.

## 1 Introduction

Self-evolving agents generate artifacts through interaction with an environment—textual reflections, executable skills, or code—and inject them into a harness so that experience can be reused across tasks (Xi et al., 2023). Work in this area typically argues that a method is effective using three signals: artifacts accumulate, behavior changes once they are loaded, and accuracy, pass rate, or evolutionary score rises. We call this signal-based evaluation. All three signals are proxies for capability. None directly tests whether a changed behavior advances the task, whether an established effect is overturned by another part of the system, or whether later evolution builds on what has been gained. More fundamentally, what “effective” should refer to has not been defined. We therefore ask: what verifiable task capability do the artifacts a system generates and reuses actually leave behind?

Prior work has argued that endpoint performance on a single episode cannot characterize self-evolution, and has turned to long-horizon task streams, capability trajectories across frozen versions, or separate analyses of reflections, tool libraries, and transfer (Section 2). These analyses still take episodes, versions, or tools as their objects. Answering our question instead requires taking the transition from artifact to capability itself as the object: observing how a single action changes, whether that change survives in actual decision making, and whether it carries over across versions. Such observations are hard to make in open-ended tasks, where states cannot be replayed exactly, the correctness of an action lacks an independent criterion, and environmental noise and unstructured states confound attribution. We therefore study the question in a controlled environment.

Such an environment must make states replayable, task effects rule-judged, and games and versions fully recorded (Section 4.1). Inspired by (Li et al., 2023), who study world-model representations with controlled board-game sequences, we use freestyle Gomoku (Gomoku without forbidden-move restrictions) as a “microscope” for self-evolving mechanisms. Gomoku meets these requirements. Simple rules do not make the task simple. The agent reads the board only as a sequence of coordinates and must reconstruct spatial relations along rows, columns, and diagonals on its own (Figure 1); without memory, about half of the model’s moves land on occupied points, which leaves real room for evolution to improve.

In this environment, we define capability formation as five properties that an artifact passes through in sequence, forming the Artifact-to-Capability diagnostic framework (Section 3.1). An artifact must first form and enter decision making, and then change actions. A change is neutral in itself: it must also produce a positive effect without undermining effects already established, and only when both hold has a capability formed. Later evolution should then extend it. The five properties are judged at three scales—states, games, and lineages—which we call three-scale diagnosis (Section 4).

![](images/c40a2e59dcabda9faa9d043257aef459b5946684dcc3b17d92c915805c3dab28.jpg)  
Figure 1: Different reasoning requirements from the same board state.

We let Reflexion (Shinn et al., 2023), Voyager (Wang et al., 2023), and DGM (Zhang et al., 2026b) evolve autonomously with the same base model, environment, and opponents, then freeze and diagnose them. To show that signal-based evaluation cannot establish capability formation, a single counterexample suffices: a system that shows all three signals but has not formed a capability. We present three, each breaking at a different property— three capability-formation breaks. Reflexion shows a legalization gain: its memories make the model play fewer illegal moves but do not make it choose better among legal ones. Voyager shows a compositional override: each component is sound on its own, but in combination the calling script blocks the opponent and returns early at conflict states, overriding an established win; across 20 games, it completes five at none of its immediate-win states. DGM shows loss-only drift: OB-3 passes the checkpoint, but later versions only lose passing states, and the last barely changes behavior. That the three breaks fall on different properties also shows that no property can stand in for another.

Our contributions are:

1. Problem and definition. We pose a question that signal-based evaluation leaves open— what verifiable task capability do the artifacts a system generates and reuses actually leave behind?—and define capability formation as five sequentially dependent properties, with task-relation realization and boundary-safe reuse forming the checkpoint for capability formation.

2. Method. We introduce three-scale diagnosis, which judges these properties at the state, game, and lineage levels in controlled freestyle Gomoku.

3. Findings. We present three counterexamples in Reflexion, Voyager, and DGM: all three signals are present, yet the diagnosis detects a legalization gain, a compositional override, and loss-only drift, breaking at task-relation realization, boundary-safe reuse, and capability extension, respectively. Signal-based evaluation therefore cannot determine whether a capability has formed.

## 2 Related Work

Prior work has examined self-evolving agents and the roles of their artifacts from several angles. Process-oriented evaluations track performance changes along checkpoints or task streams. EvoPathBench freezes successive artifact checkpoints and uses paired evaluations with artifacts enabled and disabled on held-out episodes to test whether performance depends on them (Lin et al., 2026a). Agent Plasticity measures selfimprovement at each checkpoint in environments including board games; using LLM-generated observational annotations, it distinguishes two bottlenecks—failure to reuse a relevant artifact and failure even after reuse—and notes that controlled interventions such as removal or replacement could further separate these mechanisms (Singh et al., 2026). Lin et al. (2026b) trace the limited gains of weaker models to artifacts that are either not loaded or not followed after loading.

Other work documents negative effects of artifacts: when skills with overlapping functions coexist, conflicts arise mainly when the system selects which skill to read (Yan et al., 2026); as a skill pool keeps growing, its pass rate first rises and then falls (Shang et al., 2026); and reflective memory can lower success relative to a memory-free retry in some settings (Bai et al., 2026). These studies characterize change mainly through aggregate performance over tasks, episodes, or checkpoints; we refer to this as outcome analysis. Outcome analysis can reveal gains, regressions, and retention, and combined with interventions it can also test whether performance depends on artifacts.

Closer to the state level are studies that make paired comparisons on the same input. Misevolution compares action safety with and without memory for the same query (Shao et al., 2026); SkillApt compares answer correctness and cost with and without a skill in the same task state (Guo et al., 2026); Counterfactual Memory Audit crosses two histories while holding a verified-identical input fixed, distinguishing whether memory changes an action from whether it leads to the correct task branch (Zhang et al., 2026c); and Huang et al. (2024) classify transitions before and after self-correction on the same problem, including incorrect-to-correct and correct-to-incorrect changes.

Our object of diagnosis is capability formation itself: whether a frozen system can turn behavioral change into task capability. We decompose this transition into sequentially dependent properties (Section 3.1) and pair each with independently assessable evidence: move differences and rulebased judgments in state-level diagnosis, call traces in game-level diagnosis, and newly passing and lost states in lineage-level diagnosis. Within this framework, the two bottlenecks identified by Agent

Plasticity correspond to behavioral activation and task-relation realization, whereas the breaks in Reflexion, Voyager, and DGM fall at task-relation realization, boundary-safe reuse, and capability extension. Additional related work appears in Appendix G.

## 3 Artifact-to-Capability Diagnostic Framework

## 3.1 Five Diagnostic Properties

In the self-evolution loop (Figure 2), the system compresses part of its interaction trajectories into persistent artifacts and invokes them in later decisions. Compression makes experience reusable across tasks, but it also discards the state conditions, failure causes, and alternative actions of the original trajectory. We must therefore test whether the retained experience still produces task-relevant effects in new decisions. We define five properties along the stages an artifact passes through to become a capability:

1. Artifact formation: a persistent structure has formed and is connected to subsequent decision making;

2. Behavioral activation: under the same state and interface conditions, actions with and without the artifact differ by more than sampling noise;

3. Task-relation realization: the artifact produces a positive effect—the resulting actions better satisfy the relevant task relation overall, rather than merely differing from the original ones;

4. Boundary-safe reuse: the artifact does not undermine an established effect—in actual decision making, an established positive relation is not overturned by another part of the system;

5. Capability extension: a later version adds new positive effects while preserving existing ones.

The dependency structure is 1 → 2 → (3 and 4) → 5; we call this sequence the capability chain. Properties 3 and 4 form a single checkpoint: the former asks whether the artifact produces a positive effect, the latter whether it undermines an established one. When both hold, we say that capabilityformation has occurred. Property 4 concerns only established positive relations that are overturned in actual decisions; gains and losses on fixed states already count toward Property 3. Property 5 is meaningful only for versions that pass the checkpoint. Diagnosis proceeds in this order: the first property that fails is the break, and properties that depend on it are not judged.

![](images/b35b35a790bb34326aee15ecc90078ac7f02fab049623486d796e33a817568f7.jpg)  
Figure 2: The self-evolution loop: experience, evaluation, reflection, artifact updates, and feedback.

The five properties apply at three scales. Properties 2 and 3 concern single moves and are judged on fixed states; Property 4 concerns established effects in actual decision making and is judged in complete games; Property 5 concerns evolution and is judged along the version lineage; and Property 1 is confirmed directly from call records. We call this three-scale diagnosis, comprising statelevel, game-level, and lineage-level diagnosis (Section 4). Game-level diagnosis does not judge wins and losses as such, but whether an established effect is overturned.

A task relation is defined by the task objective: it specifies which actions advance the task in a given state, such as taking one’s own immediate winning point, blocking the opponent’s, or forming an open four. Environmental rules, such as not playing on an occupied point, are not task relations: every legal action satisfies them, yet they advance nothing. Property 3 therefore compares only legal actions; reducing rule violations does not by itself count as a positive effect.

## 3.2 System-Level Unit of Analysis

Our object of diagnosis is thefrozen system: all artifacts that actually participate in decision making in a frozen version, taken as a whole. Decisions are produced jointly by several components: skills propose candidates, a calling script decides which skill to query first and which answer to accept, and code branches override one another. A component may be correct on its own yet have its effect overturned by another once they are combined. With single components as the unit, a failure could only be blamed on “a poorly written skill,” leaving open whether the system as a whole has formed a capability. We therefore judge Properties 1–4 on the frozen system; which component a break lies in serves only as explanation and does not change the judgment. Formal definitions appear in Appendix A.

## 4 Experimental Design

The three mechanisms evolve autonomously with the same base model (DeepSeek-V4-Flash), environment, and opponents. After evolution, they are frozen and undergo three-scale diagnosis (Table 1). Judgments rely only on experimental data and rules.

## 4.1 Environment: Freestyle Gomoku

Three properties of the environment make threescale diagnosis possible (Figure 3). Replayable states: any board can be restored exactly, so statelevel diagnosis compares moves with and without an artifact on the same state, and set A can be expanded by rotation and reflection. Rule-judged task effects: a rule program enumerates the acceptable moves of each state independently of the system under diagnosis and of any other model, so “changed for the better” is separated from “merely changed” for Properties 3 and 4. Fully recorded games and versions: every move, call path, and version is archived, so game-level diagnosis can locate conflict states and lineage-level diagnosis can count newly passing and lost states.

The board is 15 × 15. Players alternate placing stones on empty intersections; playing on an occupied intersection or off the board is illegal. Forming a horizontal, vertical, or diagonal line of five stones (five in a row) wins, and game length is counted in moves.

![](images/38498c79df685330a581d2d0c7b2c1a2dcac3db024c312985085afe2f9d41ca8.jpg)  
Figure 3: The three properties of the rule-based environment and the diagnostic scale each one supports: replayable states (state level), rule-judged task effects (state and game level), and fully recorded games and versions (game and lineage level).

Table 1: Three-scale diagnosis and the properties it judges.
<table><tr><td>Scale</td><td>Properties and criteria</td><td>Why this works</td></tr><tr><td>State-level</td><td>Compare moves with and without the tested artifact on the same states. Property 2: move changes exceed sampling noise. Property 3: the pass rate among legal moves increases.</td><td>The states are identical and the tested artifact is the only difference, so any change in moves can only come from it. A rule program independently specifies the acceptable moves, distinguishing &quot;changed for the better&quot; from “merely changed.&quot;</td></tr><tr><td>Game-level</td><td>Play complete games between the frozen system and a programmatic opponent. Property 4: complete five at every immediate-win state.</td><td>At an immediate-win state, “win in one move&quot;is already established and the rules specify the unique correct move. Failing to complete five shows that another part of the system overturned this effect, so no control condition is needed.</td></tr><tr><td>Lineage-level</td><td>Compare adjacent versions state by state on the same states, along with their wins. Property 5: later versions retain prior effects while adding new ones; losing without adding, or barely changing behavior, fails.</td><td>Paired state-level comparison shows exactly what each version adds or loses, and whether its modifications still reach behavior.</td></tr></table>

A point at which one move completes five is an immediate winning point. A state in which it is the agent’s turn and the agent has an immediate winning point is an immediate-win state; if the opponent also has one, it is a conflict state. Four in a row with both ends open is an open four, and three in a row with both ends open is an open three. In each state, the legal moves that satisfy the primary task relation form the acceptable set G(s); a move in G(s) passes, and the fraction of states passed in a set is its pass rate. All three opponents are programmatic (rule-based, threat-aware, and search\_2); the strongest, search\_2, searches two plies ahead. Appendix B gives the full rules and criteria.

## 4.2 Three Scales

State-level diagnosis. We use 1,876 fixed states: set A contains 120 states obtained by rotating and reflecting manually constructed tactical states, and set B contains 1,756 states drawn from actual games during DGM evolution (construction in Appendix B). The control removes the artifact under test: empty memory for Reflexion; round1 for Voyager, which keeps only the first-round tactical skills and leaves the calling script unchanged; and the agent’s initial strategy P0 for DGM. For Voyager and DGM, we thus test the effect of what evolution added. Because Reflexion samples its moves, we use the difference between two empty-memory runs as the noise baseline.

Table 2: Signals and three-scale diagnosis results for the three mechanisms. Behavior change corresponds to Property 2; Property 1 is confirmed from call records for all three mechanisms (Section 5); Properties 3 and 4 must both hold.
<table><tr><td rowspan="2">Mechanism</td><td>Signals</td><td colspan="3">Three-scale diagnosis</td><td rowspan="2">Break</td></tr><tr><td>Accumulation / behavior change / metric gain</td><td>State-level: Property 3</td><td>Game-level: Property 4</td><td>Lineage-level: Property 5</td></tr><tr><td>Reflexion</td><td>Six cells, seven memories each; moves change on 59.8%–78.2% of B states (noise: 46.7%); overall pass rate rises from 8.0% to 9.2%–10.3% on B and from 21.7% to 22.5%–27.5% on A.</td><td>Legal pass rate improves in none of 12 combinations; fails.</td><td>Not judged.</td><td>Not judged.</td><td>Property 3</td></tr><tr><td>Voyager</td><td>Seven skills and a calling script; ablating components changes moves on 89–808 B states; open fours rise from 0/20 to 20/20.</td><td>round1 → full: 459 newly passing and 73 lost on B; holds.</td><td>Completes five at 0/20 immediate-win states; fails.</td><td>Not judged.</td><td>Property 4</td></tr><tr><td>DGM</td><td>Six accepted descendants; OB-3 changes moves on at least 661 B states relative to PO; B pass rate rises from 48.7% (P0) to 63.0% (OB-7); evolutionary score rises from 150.4 (OB-1) to 312.6 (OB-7).</td><td>OB-3 vs. PO: 522 newly passing and 139 lost; holds.</td><td>Completes five at 19/19; holds.</td><td>OB-5 and OB-6 only lose; OB-7 barely changes behavior; fails.</td><td>Property 5</td></tr></table>

Game-level diagnosis. The frozen system plays 20 games against search\_2, each with a different random seed. We rebuild the board after every move, identify all immediate-win states, and record whether the system completes five; for games in which it does not, we keep the per-move call trace for explanation.

Lineage-level diagnosis. This applies only to mechanisms with a version lineage, here DGM. We judge Properties 1–4 on OB-3, the evolved version with the most wins against search\_2, and use lineage-level diagnosis to test whether later versions retain and extend its effects. Along the main lineage, we compare adjacent versions state by state on set B, count newly passing and lost states, and compare wins. We also build a differential pool of 7,234 states from archived evolution games and compare the moves of the last two versions, OB-6 and OB-7, on every state (Appendix D).

## 4.3 Three Mechanisms

The three mechanisms cover three artifact forms; implementation details are in Appendix F. Reflexion writes game feedback into textual memories, split by opponent and feedback type (outcome gives only the result; trace adds the full move sequence) into six cells of seven entries each, all of which we test; the model chooses moves directly. Voyager writes experience as executable skills that only detect board features or propose candidates;

for example, detect\_immediate\_wins finds immediate winning points, and create\_open\_three proposes attacking moves that form an open three. The final move is chosen by auto\_player, a calling script that the agent writes itself; it queries skills in a fixed order and returns as soon as one fires. DGM modifies strategy code directly, producing the lineage P0 → OB-1 → OB-2 → OB-3 → OB-5 → OB-6 → OB-7 (OB-4 is a side branch).

## 5 Diagnostic Results

Property 1 holds for all three mechanisms and is confirmed directly from call records: Reflexion’s memories are included in every move prompt, Voyager’s skill library and auto\_player are invoked at every move, and every DGM version supplies the select\_move function that the evaluator calls. We do not restate it below. Table 2 summarizes the results: all three mechanisms show the three signals, yet none completes the capability chain, and each exhibits a different capability-formation break.

## 5.1 Reflexion: Legalization Gain, Not Selection Gain

After memories are loaded, moves change on 59.8%–78.2% of B states, above the noise baseline of 46.7% (Property 2 holds), and the overall pass rate P(pass) rises on both sets (Table C.1). Taken alone, this metric suggests that reflection helps the model play better.

![](images/086021083c09ce22cd1dc1ece009503139acba75e8f4ff85f3e15b9155fa096d.jpg)  
Figure 4: Voyager against search\_2. Twenty independent games with different random seeds converge to four move sequences (G1–G4); numbers give move order. In every sequence, Voyager forms an open four whose two immediate winning points (blue stars) stay empty. At move 19, a conflict state, auto\_player returns early on the defense branch and blocks one of search\_2’s immediate winning points (orange ring); search\_2 completes five at the other on move 20 (orange line). Immediate winning points are recomputed from the rules; Appendix E.1 lists them and the auto\_player call order.

![](images/76120d1d5077b1e91bb82f07c2f1dbd1d379fd64d85010b57af447ec441270ea.jpg)  
Figure 5: Reflexion on set B: change in each rate relative to empty memory, one dot per frozen memory (six in total; each memory keeps the same horizontal position in all three columns). All six raise the overall pass rate and the legality rate, yet all six lower the legal pass rate. The empty-memory rerun changes the three rates by +2.5%, +2.4%, and +0.6%. Full results for both sets are in Table C.1.

By definition, however, $P ( { \mathrm { p a s s } } ) = P ( { \mathrm { l e g a l } } ) \times$ P(pass | legal), and the rise in overall pass rate comes entirely from a higher legality rate—a legalization gain (Figure 5). The legal pass rate, which measures move selection, improves in none of the 12 combinations and, on the larger, lower-noise set B falls by 1.6–4.4 percentage points. State-level pairing agrees: most newly passing states come from illegal moves becoming legal, and on states where both conditions yield legal moves, the net change is negative (Appendix C). Memory suppresses rule violations but produces no selection gain. Task-relation realization (Property 3) fails, and the break is at Property 3 (alternative explanations in Appendix C.1).

## 5.2 Voyager: Compositional Override

Ablating any component changes moves on 89 to 808 B states (Property 2 holds), and from round1 to full, 459 B states become passing and 73 are lost (Property 3 holds; ablation data in Appendix E). Open fours in games also rise from 0/20 to 20/20.

In game-level diagnosis, however, Voyager reaches an immediate-win state in all 20 games yet completes five at none of them (0/20), blocking the opponent’s immediate winning point instead and losing every game (Figure 4). All 20 are conflict states. Call traces locate the break

Table 3: Wins against search\_2, B pass rate, and paired changes relative to the preceding version along the main DGM lineage.
<table><tr><td>Version</td><td>Wins</td><td>B pass rate</td><td>Newly passing / lost</td></tr><tr><td>PO</td><td>20/20</td><td>48.7%</td><td></td></tr><tr><td>OB-3</td><td>19/20</td><td>70.5%</td><td>+522 / −139 (vs. P0)</td></tr><tr><td>OB-5</td><td>9/20</td><td>64.6%</td><td>0/-103</td></tr><tr><td>OB-6</td><td>0/20</td><td>63.0%</td><td>0/-29</td></tr><tr><td>OB-7</td><td>0/20</td><td>63.0%</td><td>1/0</td></tr></table>

## 6 Discussion

in how the components combine. The “win if possible” logic in evaluate\_threats is correct on its own; the defense-first rule in auto\_player is reasonable when only the opponent has an immediate winning point; and create\_open\_three drives attacks that form open fours. These attacks, however, make conflict states frequent, and at such states auto\_player returns at its first branch, which blocks the opponent (defense-first early return), so neither its own immediate-win check nor the “win if possible” logic in evaluate\_threats is ever reached (Appendix E.1). Each component is sound on its own, yet in combination they overturn an established effect—a compositional override. Boundary-safe reuse (Property 4) fails, and the break is at Property 4 (alternative explanations in Appendix E.2).

## 5.3 DGM: Loss-Only Drift After Capability Formation

DGM shows all three signals (Table 2). OB-3, the evolved version with the most wins, passes the checkpoint. Relative to P0, 522 B states become passing and 139 are lost, a net gain of 383 (Properties 2 and 3 hold). It completes five at all 19 immediate-win states in its 20 games (19/19; Property 4 holds) and wins 19 games, one fewer than P0. OB-3 has formed a capability.

Lineage-level diagnosis, however, shows no extension afterward (Table 3). OB-5 and OB-6 add no newly passing states and only lose, while wins fall to 0—loss-only drift. OB-7 differs from OB-6 on only one of the 7,234 states in the differential pool, so its modification barely reaches actual decisions (Appendix D). Meanwhile, the evolutionary score still rises from OB-3 to OB-5 (Appendix D.1). Capability extension (Property 5) fails, and the break is at Property 5 (alternative explanations in Appendix D.1).

Although derived in Gomoku, the properties concern task relations rather than the game; we conjecture, without having tested it, that the same breaks arise wherever artifacts are reused through a fixed interface. Consider a GUI agent with two skills. Skill A moves the cursor to a target element and clicks; it can reach the target (a positive effect), land on a non-interactive region (an ineffective change), or open an unrelated page (a harmful change), and a version that only repairs malformed coordinates would yield a legalization gain. Skill B, written for pages behind a pop-up, first clicks to dismiss the pop-up and then moves to its own target. Each skill reaches its target alone, but when B runs after A, its first click hits whatever lies under the cursor on the page A just reached and navigates away: a compositional override of A’s established effect. Testing this requires replayable screens and a rule-based checker of application state (Section 4.1).

## 7 Conclusion

Most work supports self-evolving methods with artifact accumulation, behavioral change, and metric gains—signal-based evaluation—without defining what counts as capability formation. We define the transition from artifact to capability as five properties and judge them with three-scale diagnosis. Under the same base model, environment, and opponents, the diagnosis detects three capabilityformation breaks: a legalization gain in Reflexion, a compositional override in Voyager, and loss-only drift in DGM. These counterexamples show that the three signals are useful but insufficient to determine whether a capability has formed. Evaluating the benefits of self-evolution requires directly testing positive effects, the preservation of established effects, and extension across versions, at the state, game, and lineage levels.

## Limitations

How the three break types are distributed across mechanisms and tasks requires diagnosis at a larger scale. We cover three representative self-evolving mechanisms, whose artifacts are textual memories, executable skill libraries, and code-version lineages, but not every form of self-evolution: multiagent co-evolution or direct parameter updates may break at other stages. Which properties can be judged also depends on a mechanism’s structure; for example, capability extension can be judged only for mechanisms with a version lineage. During diagnosis, Reflexion chooses moves with the model directly, whereas Voyager and DGM use evolved code. Whether different base models, longer evolution, or repeated runs would move the breaks elsewhere remains to be tested.

Finally, our empirical scope is limited to freestyle Gomoku. We chose it because rules can independently judge whether a move is correct and states can be replayed exactly; this also means that its task relations are local and enumerable. Task relations in open-ended tasks are often hard to determine from rules, and how to operationalize the framework there remains open.

## References

Yu Bai, Yukai Miao, Dawei Wang, Li Chen, Yanyu Ren, Yuqian Shi, Dan Li, Ying Xiong, Chengqiu Tan, Run Zhou, and Li Li. 2026. VRL-Bench: Benchmarking agents on computer control tasks under finite trial budgets. arXiv preprint arXiv:2609.12404.

Seonghyeon Cho and Chanjun Park. 2026. Skill Following: Evaluating actual skill use in retrieval-enabled LLM agents. In Findings ofthe Associationfor Computational Linguistics: EMNLP 2026.

Prakhar Dixit, Sadia Kamal, and Tim Oates. 2026. Honest lying: Understanding memory confabulation in reflexive agents. In ICML 2026 Workshop on Failure Modes in Agentic AI.

Gen Dong, Yanjie Gao, Liqun Li, Tianyin Xu, Yu Hua, and Fan Yang. 2026. Agent skills can be harmful: An empirical study of skill-induced failures in LLM agents. arXiv preprint arXiv:2608.11888.

Xingze Gao, Chuanrui Hu, Hongda Chen, Pengfei Yao, Zhao Wang, Yi Bai, Zhengwei Wu, Yunyun Han, Xiaofeng Cong, Jie Gui, Yafeng Deng, and Teng Li. 2026. EvoAgentBench: Benchmarking agent self-evolution via ability transfer. arXiv preprint arXiv:2607.05202.

Shuang Guo, Xiaobo Sui, Fei Jiang, Ruixin Li, Manxiang Li, Rong Li, and Ziyang Ou. 2026. Towards

strategy-level RSI for skill-augmented agents: Learning when to reuse skills from execution feedback. arXiv preprint arXiv:2609.26863.

Jinwei Hu, Yi Qi, Xinmiao Huang, Youcheng Sun, Yi Dong, and Xiaowei Huang. 2026. Skill use or skill theater? Evaluating the reasoning backroom in skill-augmented language agents. arXiv preprint arXiv:2607.27484.

Jiahao Huang, Fei Cheng, Junfeng Jiang, Zefan Yu, and Akiko Aizawa. 2026. BenchTrace: A benchmark for testing reflection ability and controlled evolution in LLM agents. arXiv preprint arXiv:2605.29225.

Jie Huang, Xinyun Chen, Swaroop Mishra, Huaixiu Steven Zheng, Adams Wei Yu, Xinying Song, and Denny Zhou. 2024. Large language models cannot self-correct reasoning yet. International Conference on Learning Representations.

Sihang Jiang, Lipeng Ma, Zhonghua Hong, Keyi Wang, Zhiyu Lu, Tengfei Wang, Shisong Chen, Jinghao Zhang, Tianjun Pan, Weijia Li, Jiaqing Liang, and Yanghua Xiao. 2026. SEA-Eval: A benchmark for evaluating self-evolving agents beyond episodic assessment. arXiv preprint arXiv:2604.08988.

Sungwoo Jung and Seonil Son. 2026. How much heavy lifting can an agent harness do?: Measuring the LLM’s residual role in a planning agent. arXiv preprint arXiv:2604.07236.

Kenneth Li, Emily Hopkins, David Bau, Fernanda Viegas, Hanspeter Pfister, and Martin Wattenberg. 2023. Emergent world representations: Exploring a sequence model trained on a synthetic task. In International Conference on Learning Representations.

Hongqiang Lin, Chao Liu, Xiaofan Bai, Xuan Jin, Yuhong Li, Nenggan Zheng, and Xipeng Cao. 2026a. Beyond endpoint performance: Process-level evaluation of self-evolving agents. arXiv preprint arXiv:2609.24663.

Minhua Lin, Juncheng Wu, Zijun Wang, Zhan Shi, Yisi Sang, Bing He, Zewen Liu, Tianxin Wei, Zongyu Wu, Zhiwei Zhang, Dakuo Wang, Xiang Zhang, Benoit Dumoulin, Cihang Xie, Yuyin Zhou, Suhang Wang, and Hanqing Lu. 2026b. Harness updating is not harness benefit: Disentangling evolution capabilities in self-evolving LLM agents. arXiv preprint arXiv:2605.30621.

Wenye Lin, Jonathan Roberts, Yunhan Yang, Samuel Albanie, Zongqing Lu, and Kai Han. 2025. GAME-BoT: Transparent assessment of LLM reasoning in games. In Proceedings of the 63rd Annual Meeting ofthe Associationfor Computational Linguistics.

Nico Pelleriti, Sree Harsha Nelaturu, Zhanke Zhou, Zongze Li, Max Zimmer, Bo Han, and Sebastian Pokutta. 2026. What do evolutionary coding agents evolve? arXiv preprint arXiv:2605.20086.

Chen Qian, Peng Wang, Dongrui Liu, Junyao Yang, Dadi Guo, Ling Tang, Jilin Mei, Qihan Ren, Shuai Shao, Yong Liu, Jie Fu, Jing Shao, and Xia Hu. 2026. The why behind the action: Unveiling internal drivers via agentic attribution. arXiv preprint arXiv:2601.15075.

Linfang Shang, Ming Xu, Yiding Sun, Tianle Xia, Lingxiang Hu, Lan Xu, and Ning Zheng. 2026. When self-evolution backfires: Pre-commit gating against skill contamination in LLM agents. arXiv preprint arXiv:2608.05810.

Shuai Shao, Qihan Ren, Chen Qian, Boyi Wei, Dadi Guo, Jingyi Yang, Xinhao Song, Linfeng Zhang, Weinan Zhang, Dongrui Liu, and Jing Shao. 2026. Your agent may misevolve: Emergent risks in selfevolving LLM agents. In International Conference on Learning Representations.

Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. 2023. Reflexion: Language agents with verbal reinforcement learning. In Advances in Neural Information Processing Systems.

Harman Singh, Anton Bakhtin, Rulin Shao, Gabriel Synnaeve, Ilia Kulikov, Rob Fergus, Sanjeev Arora, Kurt Keutzer, Jason Weston, Anuj Mahajan, and Anirudh Goyal. 2026. Agent Plasticity: Measuring self-improvement through experience. arXiv preprint arXiv:2610.08902.

Guanzhi Wang, Yuqi Xie, Yunfan Jiang, Ajay Mandlekar, Chaowei Xiao, Yuke Zhu, Linxi Fan, and Anima Anandkumar. 2023. Voyager: An open-ended embodied agent with large language models. Transactions on Machine Learning Research.

Wenyi Wang, Piotr Piekos, Nanbo Li, Firas Laakom, Yimeng Chen, Mateusz Ostaszewski, Mingchen Zhuge, and Jurgen Schmidhuber. 2026a. Huxley-Gödel Machine: Human-level coding agent development by an approximation of the optimal self-improving machine. International Conference on Learning Representations.

Yihao Wang, Linhan Xia, Rui Liu, Zhaofeng Zhang, Hongyu Wu, Yang Yang, Jinglu He, Yu Guo, and Kai Lei. 2026b. SAGE: A statistical acceptance gate for self-evolving agents. arXiv preprint arXiv:2609.36043.

Yixuan Wang, Yiyang Zhou, Yiming Liang, Congyu Zhang, Fuxiao Liu, Jiawei Zhou, and Huaxiu Yao. 2026c. Not all skills help: Measuring and repairing agent knowledge. arXiv preprint arXiv:2606.15390.

Zhiheng Xi, Wenxiang Chen, Xiaoran Guo, Wei He, Yeyun Ding, et al. 2023. The rise and potential of large language model based agents: A survey. arXiv preprint arXiv:2309.07864.

Zidi Xiong, Yuping Lin, Wenya Xie, Pengfei He, Zirui Liu, Jiliang Tang, Himabindu Lakkaraju,

and Zhen Xiang. 2025. How memory management impacts LLM agents: An empirical study of experience-following behavior. arXiv preprint arXiv:2505.16067.

Chaoliang Yan, Zihao Xu, Yuekang Li, Shangzhi Xu, Yi Liu, Gelei Deng, and Siqi Ma. 2026. One skill too many: How co-installed skills conflict in coding agents. arXiv preprint arXiv:2610.11647.

Dylan Zhang, Yanshan Lin, Zhengkun Wu, Yihang Sun, Bingxuan Li, Dianqi Li, and Hao Peng. 2026a. Useful memories become faulty when continuously updated by LLMs. arXiv preprint arXiv:2605.12978.

Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jeff Clune. 2026b. Darwin Gödel Machine: Open-ended evolution of self-improving agents. In International Conference on Learning Representations.

Jiajie Zhang, Yankai Xiang, and Changhao Chen. 2026c. Memory that changes action is not memory that guides it: Counterfactual auditing of history-conditioned robot policies. arXiv preprint arXiv:2609.27247.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. 2025. Which agent causes task failures and when? On automated failure attribution of LLM multi-agent systems. In Proceedings of the 42nd International Conference on Machine Learning.

Xing Zhang, Yanwei Cui, Guanghui Wang, Ziyuan Li, Wei Qiu, Bing Zhu, and Peiyang He. 2026d. Library Drift: Diagnosing and fixing a silent failure mode in self-evolving LLM skill libraries. arXiv preprint arXiv:2605.19576.

Weixiang Zhao, Yingshuo Wang, Yichen Zhang, Yanyan Zhao, Yu Zhang, Yang Wu, Dandan Tu, Bing Qin, and Ting Liu. 2026. Rethinking experience utilization in self-evolving language model agents. arXiv preprint arXiv:2605.07164.

Xiaolin Zhou, Jinbo Liu, Li Li, Ryan A. Rossi, and Xiyang Hu. 2026. Counterfactual trace auditing of LLM agent skills. arXiv preprint arXiv:2605.11946.

## A Diagnostic Definitions

Let s be a fixed state, let $\pi _ { A } ( s )$ denote the move selected by the system with the artifact under comparison loaded, and let $\pi _ { \mathbb { O } } ( s )$ denote the move under the control condition. Let $G ( s )$ be the set of legal moves in s that satisfy the relevant task relation; a move is counted as passing if and only if it belongs to $G ( s )$

Behavioral activation. We count the states for which $\pi _ { A } ( s ) \neq \pi _ { \emptyset } ( s )$ . For mechanisms in which the model samples moves, we compare this count with the same quantity computed between two independent control runs (the noise baseline); a change counts as activation only if it exceeds that baseline. Voyager and DGM select moves deterministically in code, so any changed output counts as activation.

Task-relation realization. At the call level, the pass rate decomposes as $P ( { \mathrm { p a s s } } ) = P ( { \mathrm { l e g a l } } ) \times$ P(pass | legal); we judge improvement only by whether P(pass | legal) increases. At the state level, newly passing states are classified by source as “illegal → pass” or “legal non-pass → pass.” The former indicates only that a move became legal and does not count as a positive task effect.

Boundary-safe reuse. We reconstruct each game move by move and identify every immediate-win state. If there are n such states and the system completes five in k of them, we report $k / n ;$ any $k < n$ indicates that an already-established winning relation was overturned.

Capability extension. Let $C _ { v }$ be the set of states passed by version v on set B. Between adjacent versions, newly passing and lost states are $N =$ $C _ { v ^ { \prime } } \setminus C _ { v }$ and $L = C _ { v } \setminus C _ { v ^ { \prime } }$ , respectively; we also compare wins against search\_2. This criterion applies only to iterations after the version that passes the checkpoint. Such an iteration fails to preserve and extend capability when $| N | = 0$ and $| L | > 0$ or when the number of wins declines. We classify an iteration as stagnant when $\left| N \right| + \left| L \right|$ is close to zero and the selected moves are almost unchanged.

## B Environment and Fixed States

An independent rule-based referee checks the legality and outcome of every move. When the agent has an immediate winning point and it is its turn, the uniquely correct move under the rules is to complete five, even if the opponent also has an immediate winning point: completing five ends the game immediately, whereas blocking only allows the game to continue.

We construct G(s) as follows. Set A is formed from 22 manually constructed tactical states through rotations and reflections; we retain 120 states with unambiguous task relations. Set B consists of 1,756 deduplicated states sampled from games during DGM evolution and balanced across task-relation categories. For both sets, a rule program enumerates all legal moves satisfying the primary task relation in each state; for example, the immediate-win relation includes every move that completes five at once.

In game-level diagnosis, each system or version plays 20 complete games against search\_2, with a limit of 200 moves per game. Games exceeding the limit are recorded as undecided.

## C Reflexion Diagnostic Data

We evaluate seven diagnostic conditions: an emptymemory condition and six frozen-memory conditions. The six memories correspond to the rule-based, threat-aware, and search\_2 opponent cells under the outcome and trace feedback settings. Each memory contains seven entries. For each state and condition, we query the model once with temperature 0.2 and max\_tokens=512. We independently run the empty-memory condition a second time to estimate sampling noise.

## C.1 Alternative Explanations

Move changes are merely sampling noise. Between the two empty-memory runs, the selected move differs on 46.7% of states in set B. Passes change in both directions at similar rates (41 versus 37), suggesting little net change. With the six memories loaded, moves change on 59.8%–78.2% of states, and every condition yields a positive net change (Table C.2), indicating that the memoryinduced changes exceed the observed sampling variation.

More legal moves dilute the legal pass rate. Newly legal moves may occur on more difficult states, thereby lowering the legal pass rate. We address this possibility by considering only states for which both the empty-memory and loaded-memory conditions produce legal moves. On set B, the net change within this legal subset is −12 and −19 for the outcome and trace conditions, respectively (Table C.3). Thus, although the number of states passing the diagnostic increases, most of the increase comes from illegal actions becoming legal rather than from improvements among actions that were already legal.

The memories may not match the diagnostic states. The memories are organized into three opponent-specific groups, whereas the diagnostic states do not encode opponent identity. Any memory may therefore be irrelevant to some of the states.

Table C.1: Reflexion call outcomes on sets A and B. Each state is queried once under each condition. The opponent cell associated with each memory condition is shown in parentheses.
<table><tr><td>Condition</td><td>Set</td><td>Legality rate</td><td># Pass</td><td>P(pass)</td><td>P(pass | 1egal)</td></tr><tr><td>Empty memory</td><td>A</td><td>58.3%</td><td>26</td><td>21.7%</td><td>37.1%</td></tr><tr><td>Empty memory</td><td>B</td><td>49.8%</td><td>140</td><td>8.0%</td><td>16.0%</td></tr><tr><td>Empty memory (second run)</td><td>A</td><td>65.8%</td><td>26</td><td>21.7%</td><td>32.9%</td></tr><tr><td>Empty memory (second run)</td><td>B</td><td>51.0%</td><td>144</td><td>8.2%</td><td>16.1%</td></tr><tr><td>outcome (threat-aware)</td><td>A</td><td>75.8%</td><td>33</td><td>27.5%</td><td>36.3%</td></tr><tr><td>outcome (threat-aware)</td><td>B</td><td>67.3%</td><td>164</td><td>9.3%</td><td>13.9%</td></tr><tr><td>trace (threat-aware)</td><td>A</td><td>96.7%</td><td>29</td><td>24.2%</td><td>25.0%</td></tr><tr><td>trace (threat-aware)</td><td>B</td><td>83.5%</td><td>180</td><td>10.3%</td><td>12.3%</td></tr><tr><td>outcome (rule-based)</td><td>A</td><td>76.7%</td><td>31</td><td>25.8%</td><td>33.7%</td></tr><tr><td>outcome (rule-based)</td><td>B</td><td>76.5%</td><td>175</td><td>10.0%</td><td>13.0%</td></tr><tr><td>trace</td><td>A</td><td>93.3%</td><td>27</td><td>22.5%</td><td>24.1%</td></tr><tr><td>(rule-based) trace (rule-based)</td><td>B</td><td>81.2%</td><td>166</td><td>9.5%</td><td>11.6%</td></tr><tr><td>outcome</td><td>A</td><td>83.3%</td><td>32</td><td>26.7%</td><td>32.0%</td></tr><tr><td>(search_2) outcome (search_2)</td><td>B</td><td>73.3%</td><td>178</td><td>10.1%</td><td>13.8%</td></tr><tr><td>trace (search_2)</td><td>A</td><td></td><td></td><td></td><td></td></tr><tr><td>trace (search_2)</td><td>B</td><td>75.8% 63.8%</td><td>30 162</td><td>25.0% 9.2%</td><td>33.0% 14.5%</td></tr></table>

Table C.2: Results for the six frozen memories on set B (n = 1,756). Net changes and p-values are based on state-level paired outcomes using the exact McNemar test.
<table><tr><td>Condition</td><td>Move change</td><td>Net change</td><td>p</td><td>P(pass | legal)</td></tr><tr><td>Empty memory (second</td><td>46.7%</td><td>+4</td><td>0.73</td><td>16.0% → 16.1%</td></tr><tr><td>run; noise baseline) threat-aware, outcome</td><td>61.3%</td><td>+24</td><td>0.0433</td><td>13.9%</td></tr><tr><td>threat-aware, trace</td><td>78.2%</td><td>+40</td><td>0.0037</td><td>12.3%</td></tr><tr><td>rule-based, outcome</td><td>62.9%</td><td>+35</td><td>0.0033</td><td>13.0%</td></tr><tr><td>rule-based, trace</td><td>72.8%</td><td>+26</td><td>0.0478</td><td>11.6%</td></tr><tr><td>search_2, outcome</td><td>65.5%</td><td>+38</td><td>0.0031</td><td>13.8%</td></tr><tr><td>search_2, trace</td><td>59.8%</td><td>+22</td><td>0.0548</td><td>14.5%</td></tr></table>

Table C.3: State-level paired outcomes for the threat-aware memories and the sources of newly passing states. “Net change among legal moves” is computed only over states on which the model makes a legal move both with empty memory and with loaded memory; it equals the number of “legal non-pass → pass” transitions minus the number of “pass → legal non-pass” transitions.
<table><tr><td>Feedback</td><td>Set</td><td>Newly passing</td><td>Lost</td><td>p</td><td>Illegal → pass</td><td>Legal non-pass → pass</td><td>Net change among legal moves</td></tr><tr><td>outcome</td><td>A</td><td>8</td><td>1</td><td>0.039</td><td>6</td><td>2</td><td>+1</td></tr><tr><td>outcome</td><td>B</td><td>77</td><td>53</td><td>0.043</td><td>53</td><td>24</td><td>-12</td></tr><tr><td>trace</td><td>A</td><td>7</td><td>4</td><td>0.55</td><td>4</td><td>3</td><td>-1</td></tr><tr><td>trace</td><td>B</td><td>111</td><td>71</td><td>0.0037</td><td>72</td><td>39</td><td>-19</td></tr></table>

We consequently evaluate all six memories separately. Across the 12 combinations of memory condition and state set (A or B), the legal pass rate improves in none of them (Table C.1), so the conclusion does not depend on which memory is used. Because set A contains relatively few states, its noise-induced fluctuation is −4.2 percentage points; the main evidence therefore comes from set B, where the corresponding fluctuation is only +0.1 percentage points.

## D DGM Diagnostic Data

During evolution, the agent autonomously decides whether to retain each candidate and which version should serve as the parent for the next generation based on the evaluation results. The main lineage is P0 → OB-1 → OB-2 → OB-3 → OB-5 → OB-6 → OB-7; OB-4 is a branch derived from OB-3.

OB-6/OB-7 differential pool. The differential pool was constructed from 472 archived games collected during DGM evolution. After exact deduplication by board state and player to move, it contained 7,212 states. We then added the 22 tactical states before expansion into set A, yielding 7,234 states in total. We queried the move-selection function of OB-6 and OB-7 separately on every state. Their selected moves differed on only one state, consistent with the fact that OB-7 adds only one newly passing state on set B.

## D.1 Alternative Explanations

Using OB-3 as the reference is cherry-picking. After OB-3, both OB-5 and OB-6 add zero newly passing states on set B. The issue is therefore not simply that later versions failed to return to the peak; they added no new passing states at all, regardless of the reference version. Using OB-1, which achieves the highest pass rate, as the reference leads to the same conclusion: the B pass rate only decreases thereafter (Table D.1).

Local misalignment between the score and the diagnostic results. The composite score and the diagnostic results show broadly similar trends across versions: from OB-1 to OB-3, both the score and the number of wins against search\_2 increase. However, a local mismatch appears from OB-3 to OB-5: the score increases from 347.0 to 357.8, whereas the number of wins decreases from 19 to 9. This occurs because the game-length term in the evaluation function compensates for the decline in the win term. Thus, the composite score broadly reflects changes in version performance, but this local mismatch shows that a higher score alone does not guarantee improved task capability. Whether later versions preserve or extend existing capabilities must therefore be assessed using both fixed-state diagnostics and complete-game outcomes.

In addition, the score remains unchanged from OB-6 to OB-7, while only one move differs across the 7,234 replayed states. At the same time, OB-6’s B pass rate (63.0%) still exceeds P0’s (48.7%). The simultaneous occurrence of these independent state-level results and score stagnation indicates that the later stagnation cannot be attributed solely to the evaluation function; rather, the code changes barely entered the actual decision path.

Set B is derived from DGM’s own games and may favor particular versions. The direction of change on set A is consistent with that on set B: after OB-3, the A pass rate decreases from 80.0% to 66.7% (Table D.1).

Table D.1: Terminal outcomes against search\_2 and pass rates on sets A and B for the main DGM lineage. The denominators for A and B are 120 and 1,756, respectively.
<table><tr><td>Version</td><td>search_2 wins/undecided/losses</td><td>A</td><td>B</td></tr><tr><td>PO</td><td>20/0/0</td><td>66.7%</td><td>48.7%</td></tr><tr><td>OB-1</td><td>0/0/20</td><td>80.0%</td><td>78.6%</td></tr><tr><td>OB-2</td><td>15/0/5</td><td>74.2%</td><td>70.7%</td></tr><tr><td>OB-3</td><td>19/0/1</td><td>80.0%</td><td>70.5%</td></tr><tr><td>OB-5</td><td>9/11/0</td><td>66.7%</td><td>64.6%</td></tr><tr><td>OB-6</td><td>0/11/9</td><td>66.7%</td><td>63.0%</td></tr><tr><td>OB-7</td><td>0/11/9</td><td>66.7%</td><td>63.0%</td></tr></table>

## E Voyager Diagnostic Data

For each ablation, we copied the frozen configuration and removed the specified component while keeping the remaining components and auto\_player unchanged. round1 was implemented by removing evaluate\_board and the components that depend on it.

## E.1 Conflict States and Call Order

We replay all 20 games and recompute immediate winning points from the rules. Each game has exactly one immediate-win state for Voyager, at move 19, and every one is a conflict state: Voyager holds an open four with two empty immediate winning points, and search\_2 also has two. In all 20 games Voyager plays on one of search\_2’s points, and search\_2 completes five at the other on move 20. Table E.3 lists the four groups; coordinates are one-based (row, column), as in Figure 4.

The order of checks in the frozen auto\_player explains this outcome. Its first branch checks the opponent’s immediate winning points and returns as soon as one exists; the check for the agent’s own immediate win comes second and is therefore never reached in a conflict state:

```python
# 0. Check opponent immediate wins FIRST
opp_wins = detect_immediate_wins(board, opponent)
if opp_wins:
row, col = opp_wins[0]
write_move(...); return
# 1. Check my immediate win
my_wins = detect_immediate_wins(board, agent_color)
if my_wins:
row, col = my_wins[0]
write_move(...); return
```

Each branch is reasonable on its own: blocking is correct when only the opponent can win, and completing five is correct when only the agent can. The failure arises only when create\_open\_three drives the game into states where both apply.

## E.2 Alternative Explanations

A single ineffective component cannot explain the failure. Holding all other skills and the calling script fixed, removing create\_open\_three still allows the system to form an open four in all 20 games and win all 20 games, whereas the full system wins 0/20 games. This intervention changes the subsequent game trajectories, but that trajectory change is part of the total effect of removing create\_open\_three from the full system. The results show that, with auto\_player and the remaining skills retained, the system can still form an open four and win. The call traces further show that prioritizing the own-immediate-win branch in conflict states would also prevent the same failure path. Thus, the problem is not that create\_open\_three or auto\_player fails to provide a positive contribution in isolation. Rather, their combination uses a call order that does not correctly handle conflict states.

Table E.1: Game outcomes against search\_2 for the Voyager configurations (20 games per configuration).
<table><tr><td>Configuration</td><td>Own open four formed</td><td>Wins</td><td>Mean length (moves)</td></tr><tr><td>full</td><td>20/20</td><td>0/20</td><td>20.0</td></tr><tr><td>-open3 (without create_open_three)</td><td>20/20</td><td>20/20</td><td>129.0</td></tr><tr><td>-minimax (without minimax_search)</td><td>0/20</td><td>0/20</td><td>20.0</td></tr><tr><td>round1 (first-round tactical skills only)</td><td>0/20</td><td>0/20</td><td>18.0</td></tr></table>

Table E.2: Pass counts for the Voyager configurations on sets A and B, with paired changes relative to the full system (moves changed / newly passing / lost).
<table><tr><td>Configuration</td><td>A (/120)</td><td>B (/1,756)</td></tr><tr><td>full</td><td>88</td><td>1,076</td></tr><tr><td>-open3 (without</td><td>96; 8/8/0</td><td>1,345; 330/274/5</td></tr><tr><td>create_open_three)</td><td></td><td></td></tr><tr><td>-minimax (without minimax_search)</td><td>88; 0/0/0</td><td>1,084; 89/18/10</td></tr><tr><td>round1 (first-round tactical skills only)</td><td>80; 16/0/8</td><td>690; 808/73/459</td></tr></table>

Table E.3: The conflict state at move 19 in each Voyager trajectory group against search\_2. Immediate winning points are recomputed from the rules.
<table><tr><td>Group</td><td>Records</td><td>Voyager&#x27;s immediate winning points</td><td>search_2&#x27;s immediate winning points</td><td>Move played</td></tr><tr><td>G1</td><td>4</td><td>(4,10), (9,5)</td><td>(4,5), (9,10)</td><td>(4,5)</td></tr><tr><td>G2</td><td>5</td><td>(4,10), (9,5)</td><td>(5,4), (10,9)</td><td>(5,4)</td></tr><tr><td>G3</td><td>5</td><td>(6,4), (11,9)</td><td>(6,9), (11,4)</td><td>(6,9)</td></tr><tr><td>G4</td><td>6</td><td>(4,6), (9,11)</td><td>(4,11), (9,6)</td><td>(4,11)</td></tr></table>

## F Experimental Setup

All three mechanisms use DeepSeek-V4-Flash as the base model and evolve through games against the same three programmatic opponents: rule-based, threat-aware, and search\_2. We do not intervene in the artifacts during evolution, and we freeze them during diagnosis. The descriptions below specify the implementations used in our experiments; they preserve the core update paradigm of each mechanism.

Reflexion. Each opponent has a separate memory cell. After each game, the system generates one textual lesson from the game feedback and appends it to the corresponding cell. Each cell retains at most seven memory entries. For diagnosis, we use the frozen state after the seventh entry has been written. The outcome condition provides only the win/loss result, whereas the trace condition also provides the complete move sequence. At decision time, all entries in the current cell are included in the move prompt, and the model directly outputs an action. Our implementation does not retrieve, filter, or weight memories based on the current state.

Voyager. The agent writes its experience as executable Python skill files. The frozen configuration contains seven agent-written files: seven skills and the calling script auto\_player. Our diagnostic target is the complete decision system comprising the skill files and auto\_player. Skills detect board features or propose candidate moves, but do not directly output the final action. auto\_player invokes the skills in a fixed order and selects the move according to the branch that fires. During evolution, the agent modifies existing skills or adds new ones based on the previous round’s evaluation, then enters the next round with the updated skill library.

DGM. At each round, the agent modifies the strategy source code to produce a candidate version. Every version exposes a select\_move(board, player) interface, which the evaluator calls directly to obtain a move. Each candidate plays 5, 10, and 5 games against the rule-based, threat-aware, and search\_2 opponents, respectively. A composite evaluation score determines whether the candidate is retained and which version is used as the parent in the next round; candidates that are not selected remain in the version archive.

This process produces a code lineage from the initial strategy P0 to later versions.

Lineage-level diagnosis uses OB-3 as the reference because it records the largest number of wins against search\_2 during evolution. The evaluation score is computed programmatically and serves as a reference signal for the agent’s retention decisions. For each candidate, the score is computed from 5, 10, and 5 games against rule-based, threat-aware, and search\_2, respectively:

$$
S = 5 0 \mathrm { W R } _ { \mathrm { R B } } + 1 0 0 \mathrm { W R } _ { \mathrm { T A } } + 1 5 0 \mathrm { W R } _ { \mathrm { S D 2 } }\tag{1}
$$

$$
+ 2 0 \operatorname* { m a x } _ { o } B _ { o } + \overline { { L } } _ { \mathrm { S D 2 } } + 2 \sum _ { o } F _ { o } ,\tag{2}
$$

where WR is the win rate against the corresponding opponent, $B _ { o }$ is the block rate against opponent o: among the agent’s moves made when o has an immediate winning point, the fraction that occupy one of those points, $\overline { { L } } _ { \mathrm { S D 2 } }$ is the average game length (in moves) against search\_2, and $F _ { o }$ is the number of open fours formed against opponent o. The scores of the main-line versions are 150.4 for OB-1, 275.2 for OB-2, 347.0 for OB-3, 357.8 for OB-5, 312.6 for OB-6, and 312.6 for OB-7. OB-5 scores higher than OB-3 despite winning fewer games against search\_2: the game-length term increases as games are prolonged and offsets the decline in the win-rate term.

## G Extended Related Work

This appendix supplements the discussion with 20 related works; Table G.1 compares their design elements with ours.

## G.1 Process Evaluation and Version Comparison

• BenchTrace (Huang et al., 2026) truncates a test snapshot before the target failure occurs, lets each framework and the ReAct baseline continue from the same prefix, and uses a rulebased detector to determine whether the failure is reproduced. It thereby measures failureavoidance rate and also evaluates whether reflection can localize and diagnose failures. On Qwen3-32B and GPT-4.1, the end-to-end pass rate of reflection under this funnel-style, stagewise evaluation is below 30%.

• EvoAgentBench (Gao et al., 2026) uses the accuracy gain over Vanilla for each “task × evolutionary state” cell and infers bottlenecks in experience encoding, routing, and adoption from the resulting gain patterns.

• SEA-Eval (Jiang et al., 2026) measures token consumption and success rate at each position in task sequences of length 5, and distinguishes genuine evolution from pseudoevolution according to whether the cost converges.

• Huxley-Gödel Machine (Wang et al., 2026a) estimates the meta-productivity of nodes in a self-modifying tree using an aggregate of descendant performance (CMP). On SWE-Verified-60, its estimator has a weighted correlation of 0.778 with empirical CMP, compared with 0.285 for DGM.

• SAGE (Wang et al., 2026b) compares the incumbent skill with a candidate edit on the same validation set. It retains only items on which the two versions disagree and applies a one-sided exact conditional binomial test (equivalent to a one-sided exact McNemar test when λ = 1) as a statistical admission criterion for the edit.

Our version-by-version comparison in the DGM experiment is similar to SAGE in that both compare individual items in a fixed set. The comparison target here is the rule-defined state-level pass rate.

## G.2 Memory and Reflection Artifacts

• Zhang et al. (2026a) compares different memory-integration strategies and finds that the utility of text memories continuously integrated by an LLM first increases and then decreases. On an ARC-AGI slice containing 19 problems that GPT-5.4 solved completely without memory, accuracy still falls to 52.6% by the tenth round of streaming integration, even when the standard solution is provided at every step.

• Dixit et al. (2026) analyzes publicly available Reflexion logs. In 16 ALFWorld environments with a reflection near-duplicate rate of at least 0.5, none of the 121 reflections mentions the correct object. Their no-memory control was run separately, with a different trial budget from the original logs.

• ExpWeaver (Zhao et al., 2026) compares different timings for incorporating experience into decision making: not using it, using it only at initialization, using it at every step, and triggering it on demand. Its causal ablation either blanks out the retrieved content at the point where the agent triggers retrieval or randomly injects experience at the average trigger frequency. The outcome is measured by success over the entire trajectory. On Qwen3- 32B, G-Memory, and ALFWorld, ExpWeaver achieves 83.33, compared with 75.37 when the retrieved content is blanked.

• Xiong et al. (2025) finds that the more similar retrieved memories are to the current input, the more closely the agent’s output resembles the output recorded in memory. They also construct an “error-free” variant that keeps the retrieved examples unchanged but replaces the recorded execution with a correct execution, allowing them to compare the effect of memory-content quality.

These works measure memory effects at the task, trial, or trajectory level. We instead determine the direction of memory-induced single-step action changes on fixed states.

## G.3 Attributing the Effects of Skill and Code Artifacts

• ASSAY (Wang et al., 2026c) randomly masks skills on software-development tasks and estimates each skill’s effect as the mean difference in binary task outcomes for each “skill × task” pair. With GPT-5.1, 103 skills, and 15 development tasks, more than 90% of skills have a cross-task effect range greater than 0.40.

• Hu et al. (2026) creates variants of a skill’s meaning, wording, name, and content in single-turn question answering and pairs them with a no-skill condition. It measures answer flips and the consistency of self-reported attribution. On mathematical tasks, attribution consistency does not exceed 0.43 across models and conditions.

• Dong et al. (2026) compares full runs of the same task with and without a skill and manually annotates failure causes. Of 125 functional failures, 68.8% are attributed to task implementation errors.

• Counterfactual Trace Auditing (Zhou et al., 2026) uses dynamic time warping (DTW) to align stages in two independent trajectories, with and without a skill, and compares patterns of behavioral influence. The authors describe this comparison as descriptive rather than a causal estimand.

• Skill Following (Cho and Park, 2026) computes the outcome difference between executions with and without a skill, restricted to tasks in which the agent actually retrieved the skill. On Math500, Llama-3.3-70B has an aggregate improvement of +14.2 percentage points, whereas this metric is −39.4 percentage points.

• Library Drift (Zhang et al., 2026d) enables or disables different skill-library governance mechanisms at the full-run level and tallies each skill’s successes and failures in injection trials.

• Pelleriti et al. (2026) analyzes the types of code edits along an evolutionary lineage and replays saved generation prompts. About 30% of added lines exactly match lines removed earlier; after replaying 36 breakthrough events, the median score ratio is 0.76.

These works use tasks, single-turn answers, full runs, aligned stages, or code edits as their paired or analytical units. Our unit of analysis is the frozen system. We pair comparisons on fixed states within multi-step games and trace how multiple components jointly affect the system’s control flow.

## G.4 Failure Attribution

• Who&When (Zhang et al., 2025) defines a decisive error step counterfactually: replacing the step with a correct action changes the outcome from failure to success. In practice, however, both human annotations and three LLM-based judgments are made by inspecting logs. On manually constructed systems, the Hybrid method achieves 12.28% step-level accuracy.

• Agentic Attribution (Qian et al., 2026) fixes a target action, progressively reveals historical prefixes at the component level, and measures changes in the action’s likelihood. It also perturbs sentence-level context by deletion and retention to identify the components and sentences driving the action across eight trajectories.

We instead judge the task effect of each action with independent rules, and our intervention target is the frozen artifact set.

## G.5 Games and Rule-Based Environments

• GAMEBoT (Lin et al., 2025) decomposes each turn in eight games into two or three subproblems and checks intermediate reasoning against ground truth generated by rulebased algorithms. In its experiments, GPT-4o achieves the highest average intermediate score, 0.52.

• Jung and Son (2026) incrementally adds manually designed harness layers in noisy Battleship and measures win rate. Adding the second layer raises the win rate from 50.0% to 74.1%; this layer does not call an LLM.

We use rule-based verifiability in Gomoku to diagnose artifacts produced through natural evolution.

## G.6 Design-Element Comparison

In Table G.1, the “same-state action pairing” column indicates whether actions or outputs with and without an artifact are compared on the same decision state with a verified-identical input.

Table G.1: Design elements compared with the closest related work.
<table><tr><td>Work</td><td>Artifact source</td><td>Pairing or intervention unit</td><td>Same-state action pairing</td><td>Primary readout</td><td>Cross-version</td></tr><tr><td>EvoPathBench (Lin et al., 2026a)</td><td>Evolutionary artifacts</td><td>Artifact on/off on held-out episodes</td><td>No (episode-level; trajectories diverge)</td><td>Episode risk-adjusted score difference</td><td>Yes (successive checkpoints)</td></tr><tr><td>Agent Plasticity (Singh et al., 2026)</td><td>Evolutionary artifacts</td><td>Checkpoint-level games; LLM-observed annotations</td><td>Single case (appendix only)</td><td>Game score; usage annotations</td><td>Yes (successive checkpoints)</td></tr><tr><td>Lin et al. (2026b)</td><td>Evolutionary artifacts</td><td>Agent × evolver cross-product</td><td>No</td><td>Pass-rate gain; loading and following rates</td><td>No</td></tr><tr><td>Misevolution (Shao et al., 2026)</td><td>Hand- constructed (memory experiment)</td><td>Same query with and without memory</td><td>Yes</td><td>Safety label</td><td>Yes (before and after evolution)</td></tr><tr><td>SkillApt (Guo et al., Externally 2026)</td><td>provided</td><td>and without a skill Two histories</td><td>Same task state with Yes (single answer)</td><td>Answer correctness; No token cost</td><td></td></tr><tr><td>CMA (Zhang et al., 2026c)</td><td>Policy-internal history</td><td>crossed on a verified-identical input</td><td>Yes</td><td>Task branch selected No by the action; pairwise reliability</td><td></td></tr><tr><td>Huang et al. (2024)</td><td>No persistent artifact</td><td>Same problem before and after correction</td><td>Yes (single problem)</td><td>Incorrect-to-correct and correct-to-incorrect transitions</td><td>No</td></tr><tr><td>One Skill Too Many (Yan et al., 2026)</td><td>Externally provided</td><td>Skill configurations for the same request</td><td>No</td><td>Skill use; core-function satisfaction rate</td><td>No</td></tr><tr><td>Shang et al. (2026)</td><td>Evolutionary artifacts</td><td>Single-skill A/B replay; lineage deletion</td><td>No</td><td>pass@1</td><td>Yes (round by round)</td></tr><tr><td>VRL-Bench (Bai et al., 2026)</td><td>Evolutionary artifacts</td><td>Cases sharing the first round</td><td>No (diverging from round 2)</td><td>Case-level success rate</td><td>No</td></tr><tr><td>SAGE (Wang et al., 2026b)</td><td>Evolutionary artifacts</td><td>Incumbent and candidate on the same validation item</td><td>No (binary outcome comparison)</td><td>Per-item win/loss</td><td>Yes (adjacent versions)</td></tr><tr><td>ASSAY (Wang et al., 2026c)</td><td>Distilled external pipeline</td><td>Random masking</td><td>No</td><td>Mean difference in task success</td><td>No</td></tr></table>