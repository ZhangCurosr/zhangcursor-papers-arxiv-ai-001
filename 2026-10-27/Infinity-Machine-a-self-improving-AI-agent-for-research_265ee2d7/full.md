# Infinity Machine: a self-improving AI agent for research

Harish Santhanalakshmi Ganesan

Independent researcher

harishsg99@gmail.com

October 2026

## Abstract

We present the Infinity Machine, a self-improving AI agent, and use it for research: problems whose answers can be checked by a program or against data. It runs three loops. A discovery loop keeps an archive of scored solutions, and each new candidate is produced by a full tool-using agent that works for up to an hour in a fresh sandbox, writing and running its own search code and a lab notebook. A catalogue mode searches real data for things nobody has catalogued, judged by a referee rather than a metric. A self-improvement loop lets the system rewrite its own prompts, tools, skills, memory and meta agent behind a gate. An autonomous mode chains everything from a single plain-text goal with no human step: an agent chooses an open problem and writes its verifier; the kernel calibrates that verifier; an adversarial agent audits it; the search runs; an agent that never sees the verifier checks the result; and a final agent writes the report. Proposers see only a public score; where a task has hidden data (held-out instances, later observations), a private split that no agent sees decides what counts. Around this discovery loop the Infinity Machine keeps an editable genome: prompts, tools, skills, an evidence-backed playbook and the meta agent’s own procedure. A kernel that the agent cannot write to gates every change to the genome: it checks policy, compares held-in and held-out performance, and enforces a cost limit. We applied the system to fourteen problems, thirteen of them external, in six days of October 2026, at a total cost of about \$180 in model usage. It produced four results beyond the published state of the art on catalogued open problems in quantum information, each checked by a verifier that the proposing agent did not write:

• an exactly certified point $p = 0 . 0 6 4 9 6 6$ on the positivity ladder of the qubit depolarizing channel’s quantum capacity (previous certified point 0.064956);

• a single-letter qutrit input that moves that channel’s positivity frontier from the hashing value 0.287087 to 0.2910672;

• a 44-gate, depth-6 Cliford circuit preparing the [[23, 1, 7]] Golay logical zero, which dominates the published (44, 8) and (45, 7);

• an exact ten-term stabilizer decomposition showing $\chi ( | T \rangle ^ { \otimes 7 } ) \leq 1 0$ (previously 12).

It also cut the added-gate count of the SABRE qubit-routing benchmark from SABRE’s 68,559 to 12,087 (9,312 against the best published 13,044 on the 15 circuits shared with later work), built a 5-parameter transformer that adds 10-digit numbers, closed 24% of the gap between a 20M-parameter language model and the GPT-2 checkpoint on FineWeb validation loss, and surfaced one unconfirmed planet candidate in TESS photometry. In hardware, it redesigned the RTL of an open 8×8 INT8 systolic matrix-multiply tile. Through a full open RTL-to-GDS flow at a fixed clock, it cut the energy per multiply-accumulate from 578.9 fJ to 128.8 fJ (77.7% lower) with 55% less area. Power was measured from gate-level switching activity, and the design stays bit-exact. In three fully autonomous end-to-end runs (\$7.59–\$9.97 each, no human step after launch), the system chose its problem, wrote and audited its own verifier, searched, and re-checked its result blind. One audit rejected a task whose cited record was stale. All three runs ended in honest ties with the public record, two of them with documented near misses. In one run the agent-written score pointed the wrong way, and none of the checks caught it. We report as carefully the parts that did not work. The system’s one gated attempt to adopt a playbook it had written for itself was rejected on cost. Proposers gamed lenient scorers, usually within a single iteration, four times. And the operator’s hand-made changes to the harness were decisive for several results. We describe them so the reader can separate what the loop did from what its operator did.

## 1 Introduction

Large language models can now write and run substantial programs, and agents built on them can stay on a task for an hour or more. A system that turns this into discoveries needs two things beyond a capable model. It must have a way of telling a real advance from a plausible-looking one, and a way of keeping what it learns. AlphaEvolve [1] and FunSearch [2] show the first. They ground an evolutionary search over programs in automatic evaluation, so that the model’s errors are filtered out by execution, and they reach new results in mathematics and computer science. Work on self-improving agents, including STOP [3], the Darwin G¨odel Machine [4], Hyperagents [5] and AIDE<sup>2</sup> [6], addresses the second. There the object being evolved is the agent itself, and recursive self-improvement [7] is the long-run aim.

The Infinity Machine<sup>1</sup> is a research agent that does both. It is not a program-evolution engine with a fixed harness: the unit of work is a researcher-like agent run, the search can be open-ended as well as score-driven, and the harness that directs those agents is itself under gated self-modification. It is built for a single operator with a laptop and a cloud account. Four design choices define it.

1. Agents, not samples. A proposal in the Infinity Machine is a full agent run. A Deep Agents [8] model with a shell, a file system and up to an hour of wall clock in a fresh sandbox reads the task notes and the parent solution. It writes and runs its own search code and delivers a solution together with a lab notebook. A problem therefore receives a handful to about fifteen proposals, not thousands. The proposer, not the harness, decides whether to write a C tableau simulator, an exhaustive enumeration or a gradient ladder.

2. A public/private split. Proposers optimise a public score. Where the task has hidden data, the result is the score on a private split that no agent sees: hidden instances, hidden seeds, or later TESS sectors. For exact-verifier tasks the two splits coincide, and the “private” score is the same verifier rerun by the kernel in its own sandbox. We adopt this split from AIDE<sup>2</sup> [6]. It matters because the agents are strong enough to game any scorer they can read (Section 4).

3. The harness is itself an evolvable object. Prompts, tools, skills, the playbook and the meta agent’s procedure form a versioned object, the genome. A meta agent proposes edits to it, and an immutable kernel accepts an edit only if it passes policy, held-in, held-out and cost checks (Section 2.4).

4. End-to-end autonomy with machine-checked trust. The step that usually needs a human, deciding that an agent-written evaluator can be trusted, is replaced by three machine checks: kernel calibration on known cases, an adversarial audit whose exploits the kernel replays, and an independent re-check of the final result by an agent that never sees the evaluator (Section 2.5).

Contributions. (i) The architecture and its kernel/genome split, the discovery loop with public/private grading, a catalogue mode for open-ended search in real data, the gated self-improvement loop, and an autonomous mode that runs from a plain-text goal to a verified report (Section 2). (ii) Results on fourteen problems (thirteen external): four advances on catalogued open problems in quantum information, a large margin on a widely used compiler benchmark, two machine-learning tasks and one astronomy candidate, with every claim checked by a verifier that was not the agent’s own (Section 3, Table 1). The quantum, SABRE and TESS results are archived with their data and code on Zenodo (DOIs in the bibliography). (iii) Three fully autonomous end-to-end runs, whose safeguards caught a stale record before any search began, and a hardware result: a 77.7% lower energy per MAC for an open accelerator tile, through a full open chip flow with activity-based power (Sections 3.5 and 3.8). (iv) An account of what the self-improvement machinery did and did not achieve, of four specification-gaming incidents, and of the operator’s part in each result (Section 4).

![](images/30f3ede33f14c861f56840bd360db665e264ed02d7e3b8be409f65f196ce727a.jpg)  
Figure 1: Architecture. The kernel owns everything whose corruption would make scores meaningless: scorers, private splits, archive, budget and policy. The genome holds everything the system may change about itself. Agents run inside sandboxes that mount the genome and the public split only.

## 2 The Infinity Machine

## 2.1 Kernel and genome

The Infinity Machine is split into an immutable kernel and an editable genome (Figure 1). The kernel contains a daemon that owns long jobs and survives the controlling shell, the archive (the genome’s version history as a lineage DAG, with rollback), the scorers, the gate, the strategy and model bandits, reward logging, and the budget middleware. The genome contains the task agent’s and meta agent’s prompts, the meta agent’s procedure and weakness-mining script, tools with tests, skills with fixtures, and an ID-addressed playbook in which every bullet carries evidence run IDs and a confidence. The split follows the rule from harness-engineering practice [9]: whatever the agent may legitimately want to change is in the genome, and whatever would let it change its own grade is not. The kernel never trusts an agent’s report of what it ran. It records tool calls itself, and scorers read solutions produced in a fresh sandbox.

![](images/16a2d254906cd3392e99bfb6866595696b94a6bfe73f3e52bd0f3628e324f5e7.jpg)  
Figure 2: One iteration of the discovery loop. Private scores never reach an agent. In catalogue mode every proposer extends the shared merged catalogue instead of a sampled parent, and the referee’s per-claim verdicts are passed on as feedback.

## 2.2 Task specification

A task consists of a specification (problem family, description, scorer, score range and direction, sandbox image and resources, per-run budget, and a brief), a public split (problem statement, notes, tools, a local verifier), a private split and an optional seed solution. The scorer is either a verifier program that the kernel runs in its own sandbox, whose value is mapped linearly onto [0, 1], a test suite, or a retrain-from-code scorer for ML tasks. The brief is the operator’s channel to the proposers: it lists what has already been measured and what to try.

## 2.3 Discovery loop

A discovery session runs N iterations of the loop of Figure 2 with P proposers each. The kerne scores the seed and places it in an experience bank. In each iteration it samples P parents with probability proportional to $\sigma ( 1 0 ( s _ { \mathrm { p u b } } - \bar { s } ) ) / ( 1 + c )$ . Here $s _ { \mathrm { p u b } }$ is the parent’s public score, ¯s is the mean public score of the top three entries and c is the parent’s number of children. This is the DGM/Hyra rule [4, 10], which favours good parents that have not been expanded much. Each proposer also receives a named strategy, drawn by a per-family UCB1 bandit whose reward is whether the child beat its parent on the public score; $\mathrm { \ A I D E ^ { 2 } }$ [6] found UCB over named strategies to be its most valuable discovered mechanism. The default arms, written for the optimisation family and used by every family without its own set, are conservative, aggressive rewrite, new neighbourhood, data structure and robust/simple. Math tasks use structured construction, exhaustive small case, new objective, scale compute and literature transfer. ML tasks use new architecture family, parameter sharing, embedding budget, connectivity and recipe; astronomy tasks have five search-oriented arms of their own.

A proposer is a fresh Deep Agents model (Claude Sonnet 5.5) in a sandbox that mounts the genome, the task’s public split and an inspiration folder: the parent solution and a bank summary that shows public scores only. It works with a shell for up to about an hour. It must keep its solution directory current, because a per-run spend cap (\$4 by default), checked before every model call, can end it at any turn. On return, the kernel scores the solution on both splits in a separate sandbox, appends it to the bank and logs the strategy reward. The session’s answer is the best public solution, and its private score is what we report.

Catalogue mode. Optimisation has a parent to improve; open-ended discovery in real data does not. A task in catalogue mode changes the deliverable to a list of claims plus a search log. The scorer becomes a referee, which re-measures every claim from the raw data, checks it against catalogues of known objects and returns per-claim verdicts. Those verdicts reach the next proposer as feedback. The hidden split is prediction: for the TESS task, the private verifier mounts later observing sectors that no agent can read and checks whether each periodic claim predicts dips there. After each round the kernel merges all accepted claims into one shared catalogue, which every proposer then extends.

## 2.4 Self-improvement loop

The self-improvement loop improves the genome rather than a solution. A UCB or score-child selector picks a parent generation. The meta agent (Claude Opus 5.5), whose prompt and procedure are themselves genome files [5], reads an evidence bundle of recent runs and mines weaknesses. It proposes one bounded edit together with a manifest that states the evidence, the root cause and the predicted fixes. The kernel then applies a static policy check (only genome paths may change, kernel-owned files are byte-identical, at most 12 files) and grades the candidate on held-in and held-out tasks. It accepts the candidate only if neither split regresses beyond a noise band of 0.03 and the cost is at most 1.5 times the incumbent’s. A gain larger than 0.3 is parked for human review instead of being activated. Predictions in the manifest are checked against per-task deltas, and a meta agent whose predictions keep failing is sampled less often.

Three further paths feed the genome. Consolidation [11] reflects on run records and edits playbook bullets by ID. Bullets that a run retrieved are confirmed or weakened by that run’s outcome, and recurring lessons are promoted to skills, which must pass a fixture. Capability requests let a task agent ask for a tool; a toolsmith subagent writes it with tests, and the kernel re-runs those tests before the requester may use the tool. Rewards come from scores and from the operator’s good/bad feedback on individual runs, and update the bandits and playbook confidence. Every path ends in a generation that passes the same gate, so every self-modification is auditable and reversible.

## 2.5 Autonomous mode: from a goal to a verified report

In the modes above a human writes the task: the problem statement, the verifier, the seed and the notes. Autonomous mode removes that step. Its only input is a plain-text research goal; the default goal asks for an open problem whose best known result is an explicit object or number that a program can check exactly. Every phase is an agent run in its own fresh sandbox, and the kernel only validates, routes and records (Figure 3).

1. Scout (network on). The scout surveys public problem catalogues and the literature, picks one problem, and checks that it is still open. It writes a complete task to a fixed contract: a task specification with the record and its source, a problem statement, a verifier program, a seed solution, at least four calibration cases with known outcomes (valid ones with their exact values, invalid ones that must be rejected), and a selection note with the evidence of openness.

2. Kernel validation. The kernel runs the verifier on the seed and on every calibration case in a network-of sandbox. Valid cases must reproduce their stated values, and invalid cases must be rejected or score no better than the seed. The scoring range must also extend beyond the record, so that beating the record registers.

3. Adversarial audit (network on). A fresh agent re-checks that the problem is open and the record current, and tries to construct invalid solutions that the verifier accepts at or above the record. The kernel replays every exploit file through the verifier. The audit fails if the auditor says so or if any exploit is accepted. Its findings go back to the scout for a repair round (at most three rounds).

![](images/2d83ab7f6641f2c92cb833707b8356ff04f932589b387ac5d16e7b57d11123b2.jpg)  
Figure 3: Autonomous mode. Grey: agent runs, each in a fresh sandbox. Blue: kernel steps. Nothing between the goal and the report requires a human.

4. Installation. The task is installed and the approval is recorded as automatic, with its basis (calibration, audit and exploit replay). This replaces the human approval that an agent-proposed evaluator otherwise requires.

5. Discovery. The ordinary loop of Section 2.3 runs on the generated task.

6. Independent check (network of). A fresh agent receives only the problem statement and the best solution, never the verifier. It writes its own checker, tests it on cases it constructs, and evaluates the solution. The kernel compares the two values.

7. Report. A fresh agent writes a report and a machine-readable claim: the problem, the record and its source, the result, how it was verified, and an explicit claim boundary.

The protocol is generic: the briefs contain no problem-specific knowledge. All artefacts and events are kept under the session’s run directory for audit.

## 2.6 Infrastructure and the operator

Sandboxes are local Docker containers (network of, read-only root, no kernel files, no private split) or Modal sandboxes behind the same interface. Modal provides GPU (one H100) and read-only data volumes for the ML and astronomy tasks. A daemon owns sessions, so a closed terminal does not kill a run, provided the session is submitted as a daemon job; one session started from a foreground shell lost a proposer run when that shell died (Section 4).

The fourteen problem campaigns of Sections 3.1–3.7 were run before autonomous mode existed, and in them the operator’s role was substantial. The accelerator campaign (Section 3.5) ran later, in the same operator-led way. We want to be exact about it. For each of those problems the operator, working with an AI coding assistant:

• chose and vetted the problem against the literature;

• wrote the verifier, seed and task notes;

• launched the sessions;

• gave run-level feedback;

• after each session, updated the notes with an “already measured” table, which was the main channel by which one session learned from the last;

• fixed kernel bugs;

(a) Qubit depolarizing: Q>0 frontier  
![](images/e1c0189644c85826f7960271e7440d9ec14c2f34c6c0ccf0006ceea2cb36e0d7.jpg)

(b) Qutrit depolarizing (d = 3)  
![](images/d92a283d8c0bebe61151275171498b2e136308d0b41554df652559bd491acfb9.jpg)

(c) [[23,1,7]] Golay logical zero  
![](images/eb2e4eb61f634191db2cdbede4d5de4fd8d86d94f1c80499f10f735853b5c7a2.jpg)

(d) Stabilizer rank of magic states  
![](images/5e317992f296598f1fd94a05058b1b30a37eb159b42ad21cad6d4d5a4e1a2a39.jpg)

(e) Qubit routing, SABRE suite  
![](images/d2b5eab5bac93ac977f997d08272784a53c6e98db9a580238af06e52c27658b4.jpg)

(f) 20M-parameter LM vs GPT-2  
![](images/7419a1f2b138bee6a75134a9742d88ddd57db6649af18d08401e997a3993f6e5.jpg)  
Figure 4: Main improvements. (a) Certified points on the qubit depolarizing positivity ladder. (b) Positivity frontier of the qutrit depolarizing channel. (c) Two-qubit gate count and depth for preparing the Golay logical zero. (d) Upper bounds on the stabilizer rank of $| T \rangle ^ { \otimes m }$ . (e) Total added gates on the SABRE suite. (f) Tiny-LM bits per byte at 1B training tokens. Grey: prior work; tan: the operator’s own repetition-code baseline (the session-1 seed); orange: Infinity Machine.

• re-verified and certified every result on the host, and wrote and deposited the notes.

The operator also wrote family-specific strategy arms by hand three times (math and ML after observing failures, and astronomy when adding catalogue mode). Under our own rules that is a kernel change rather than a genome change, but it shaped later results, and we discuss it in Section 4. In the three autonomous runs (Section 3.8) the operator’s only actions were writing the goal text and launching the job: an agent chose the problem and wrote the task, and the kernel and other agents did everything up to and including the report.

## 3 Results

Table 1 summarises every problem we ran, including the null results. Figure 4 shows the main improvements. Every entry was checked by a verifier the proposers did not control: the kernel’s scorer, a host re-run, and for four of the five quantum results an exact or interval-arithmetic certificate (the fifth, private information, a 30-digit recomputation). Results were obtained between 18 and 26 October 2026.

<table><tr><td>Problem</td><td>Previous best</td><td>Infinity Machine</td><td>Verification</td><td>Model $</td></tr><tr><td colspan="5">Open problems (catalogued), quantum information</td></tr><tr><td>Qubit depolarizing  $Q > 0 p = 0 . 0 6 4 9 5 6$  threshold (OQP C11) fied</td><td></td><td>certi- p = 0.064966 certified</td><td>authors&#x27; exact rational certifier</td><td>1.31</td></tr><tr><td>Qutrit depolarizing posi- 0.287087 (hashing) 0.2910672 tivity (zoo)</td><td></td><td></td><td>50-digit interval certificate</td><td>12.48</td></tr><tr><td>Golay |0L〉 preparation (44, 8), (45, 7) (zoo)</td><td></td><td>(44, 6)</td><td>tableau + 223 statevector</td><td>19.72</td></tr><tr><td>Stabilizer rank  $\chi ( | T \rangle ^ { \otimes 7 } ) \ \leq 1 2$  (zoo)</td><td></td><td>≤ 10</td><td>exact solve over Q(ζs)</td><td>9.62</td></tr><tr><td>Depolarizing private info., 0.28223 (n=1) finite n (zoo)‡</td><td></td><td>0.284668 (n=8)</td><td>30-digit recomputation</td><td>14.06</td></tr><tr><td colspan="5">Benchmarks and algorithms</td></tr><tr><td>SABRE routing, 26 cir- 68,559 cuits, Tokyo</td><td>(SABRE); 12,087; 9,312 on 15 13,044 HAIL on 15</td><td></td><td>exact replay of every routing</td><td>21.21</td></tr><tr><td>TSP (hidden n 600–2000) seed 0.58 AdderBoard,</td><td></td><td>0.88 private</td><td>hidden instances</td><td>8.12</td></tr><tr><td>adder</td><td>smallest 6 (disputed), 10 5 parameters strict</td><td></td><td>upstream verify.py, 10,010/10,010</td><td>4.15</td></tr><tr><td>≤20M LM vs (1.0658 bpb)</td><td>GPT-2 seed 1.3376</td><td>1.2716 (not beaten)</td><td>retrain from code, 1B tokens</td><td>9.47†</td></tr><tr><td colspan="5">Hardware design</td></tr><tr><td>8×8 INT8 TPU tile, 578.9 ASAP7, 600 ps (fJ/MAC) RTL)</td><td></td><td>(original 128.8 (hidden 130.1)</td><td>bit-exact RTL + gate level; full flow, DRC 0; activity-based power</td><td>11.0§</td></tr><tr><td>Open-ended discovery TESS M dwarfs, new tran- sits</td><td></td><td>1 candidate, ~ 3σ</td><td>held-out sectors, pixel vetting</td><td>~58</td></tr><tr><td colspan="5">No improvement</td></tr><tr><td>Sidon (4,5)-sets, Einstein 4/7 Arena</td><td></td><td>13 new ties at 4/7</td><td>arena verifier</td><td>2.80</td></tr><tr><td>Hadamard max-det, n = 43.30134 51</td><td></td><td>record is a fixed point exact Bareiss</td><td></td><td>4.08</td></tr><tr><td>Seven-qubit distance-2 code</td><td> $2 4 \leq K \leq 2 6$ </td><td>no  $K \geq 2 5$ </td><td>exact residual</td><td>3.68</td></tr><tr><td>Four MUBs in d = 6 0.998291692700 (ASD)</td><td></td><td>record reproduced only 30-digit GS</td><td></td><td>2.10</td></tr><tr><td colspan="5">Fully autonomous (problem chosen and task written by the system)</td></tr><tr><td>Two-deletion code, n = 16 223 (public) (Arena 34)</td><td>223 (tie)</td><td></td><td>task verifier + blind independent checker</td><td>9.86</td></tr><tr><td>Sorting network,  $n ~ = ~ 3 1 ~ 1 8 0$  (size)</td><td></td><td>180 (tie)</td><td>task verifier + blind independent checker</td><td>7.59</td></tr><tr><td>Mixed code N(4,5,3) 186 (Brouwer)</td><td></td><td>186 (tie)</td><td>task verifier + blind independent checker</td><td>9.97</td></tr></table>

Table 1: All problems. “Model $\$ 7$ is LLM usage over all sessions on that problem; the finding run itself was usually \$0.6–4. Unrecorded spend of killed sessions (C11 session 1, one SABRE attempt of about \$0.46) is excluded. <sup>†</sup>Plus about \$25 or more of H100 time. <sup>§</sup>Plus Modal CPU time for the flow (about 10–12 minutes on 8 cores per full evaluation); ASAP7 is a predictive PDK, so the energies are not silicon measurements (Section 3.5). <sup>‡</sup>New only as explicit finite-n witnesses; the literature already implies $P > 0$ up to $p \approx 0 . 2 9 1 5$ , and the zoo record certified 0.259824. Noise convention $\Lambda _ { p } = ( 1 - p ) \rho + p I / 2$ . Zenodo deposits: [12–18].

## 3.1 Quantum capacity thresholds

Qubit depolarizing channel. The noise level at which the quantum capacity of $\mathcal { D } _ { p } ( \rho ) = ( 1 -$ $3 p ) \rho + p ( X \rho X + Y \rho Y + Z \rho Z )$ vanishes is open. Lower bounds come from inputs with positive coherent information [19, 20]. Krohn-Grimberghe recently certified, in exact arithmetic, a 45-copy rank-two symmetric-subspace state at $p = 1 6 2 3 9 / 2 5 0 0 0 0 = 0 . 0 6 4 9 5 6$ [21]. The Infinity Machine task gave proposers a float evaluator (Schur–Weyl blocks via a coherent-state generating function), that state as the seed, and score $( p - 0 . 0 6 4 ) / 0 . 0 0 2$

In session 2 the first proposer wrote its own checkpointing Adam ladder, raising p by $1 0 ^ { - 6 }$ whenever $I _ { c } > 2 \times 1 0 ^ { - 9 }$ . In 42 minutes, for \$0.60, it delivered a state with float threshold $\approx$ 0.0649671. The record authors’ own exact certifier, run unchanged on that state, classifies it as certified-positive at $p = 3 2 4 8 3 / 5 0 0 0 0 0 = 0 . 0 6 4 9 6 6$ , with enclosure $[ 2 . 3 1 , 2 . 7 8 ] \times 1 0 ^ { - 8 }$ bits [12]. This is a candidate-specific lower bound and makes no claim about the optimal threshold. A first session had reached $p = 0 . 0 6 4 9 6 2$ but was killed by host memory pressure before it saved anything, which is why the brief now demands continuous checkpointing.

Qutrit depolarizing channel. For $\mathcal { N } ( \rho ) ~ = ~ ( 1 - p ) \rho + p I / 3$ , the catalogue listed only the hashing point 0.287087. Session 1 found a 4-qutrit, rank-4 input with threshold 0.289205. In session 2 a proposer questioned the seed’s size instead of growing it: with rank 4 allowed, smaller n wins. Its Adam ladders from random starts gave $n ~ = ~ 3 ; ~ 0 . 2 8 9 9 5$ and $n = 2 \colon 0 . 2 9 1 0 6$ , and L-BFGS polishing gave 0.2910672, the optimum that all 25 L-BFGS starts reached. The operator’s structure check showed that the optimum is a product of two single-qutrit inputs, each maximally mixed on a two-dimensional subspace. The gain is therefore single-letter: the leaked component heralds the error. We gave a closed form and a 50-digit interval certificate $p _ { 2 } ( 3 ) \in$ [0.291067236033599 69, 0.291067236033599 88], and the same efect holds for every $3 \leq d \leq 6 4$ we checked [13]. The proposers found the state but not the structure: two later runs noticed that the state is $P / 4$ on a subspace, but neither computed the reduced states that reveal the product. The result is elementary enough that it may be folklore, and the note makes no priority claim.

Private information. For the private capacity of the qubit depolarizing channel (here in the convention $\Lambda _ { p } = ( 1 - p ) \rho + p I / 2$ , not per-Pauli), both sessions found classical $K = 2$ ensembles of repetition-like distributions. Their thresholds are new as explicit finite-n witnesses, rising from 0.28223 at $n = 1$ [22] to 0.284668 at $n = 8$ , but they climb only about $3 \times 1 0 ^ { - 4 }$ per qubit [15]. The operator’s literature check showed that one-way QKD rates with publicly announced syndromes [23] already imply $P > 0$ up to $p \approx 0 . 2 9 1 5$ , once the announcement is paid for in-band. The zoo record had been behind the literature, and the agent’s search was not the decisive contribution here.

## 3.2 Fault-tolerant circuits and magic states

Golay logical zero. The record CNOT-count/depth points for preparing |0⟩ of the [[23, 1, 7]] Golay code were (44, 8) and (45, 7), from RL plus MCTS graph decimation on an A100 [24]. Session 1 returned the seed from all four runs. In session 2 a proposer under the aggressive rewrite arm rewrote the search in C (about $5 \times 1 0 ^ { 6 }$ tableau evaluations per second per core). It changed the objective to the number of qubits left mixed after reverse-applying the circuit, which is zero if and only if the state is a product in any local basis, not only in $Z .$ It then walked the neutral network at a fixed 44 gates with a penalty on gates beyond the target depth. Depth 7 appeared in about a second and depth 6 in 10–60 seconds; depth 5 and 43 gates were never reached, the latter even after $1 0 ^ { 1 0 }$ evaluations and an exhaustive delete-two/insert-one sweep. The (44, 6) circuit dominates both published points. It was verified up to wire relabelling by stabilizer-tableau checks and a $2 ^ { 2 3 }$ -amplitude statevector overlap of 1.0 [14].

Stabilizer rank of $| T \rangle ^ { \otimes m }$ . Exact stabilizer ranks are known only for $m \leq 4$ , and the bounds for $m \geq 7$ were products of smaller decompositions [25, 26]. Session 1 was null. Two of its six proposers ran no search at all, one calling the task an open research problem and the other “hardening” a seed that was already a record, as its default-arm instruction suggested. After the operator replaced the math arms, a structured construction proposer in session 2 followed a direction the operator’s notes had listed as untried (close a few stabilizer states under qubit permutations) and built a dictionary by closing three seed states under $S _ { a } \times S _ { b } – \mathrm { s } _ { \mathrm { . } }$ ymmetric Clifords (396–540 vectors). It then annealed over subsets with 1,024 batched chains, reaching $r = 1 2$ in 40 s, $r = 1 1$ in a few minutes, and $r = 1 0$ on the [2, 5] block family. The decomposition is exact, with coeficients in $\scriptstyle { \frac { 1 } { 4 } } \mathbb { Z } [ \zeta _ { 8 } ]$ , and gives $\chi ( | T \rangle ^ { \otimes 7 } ) \leq 1 0$ and by sub-multiplicativity $\chi ( | T \rangle ^ { \otimes 1 3 } ) \leq 6 0 ~ [ 1 6 ]$ . A companion exhaustive small case run found that coordinate descent over the full $m = 5$ stabilizer table stalls at residual 0.0647 for five terms, which is evidence (not proof) that $\chi ( | T \rangle ^ { \otimes 5 } ) = 6$

## 3.3 Algorithms and benchmarks

Qubit routing on the SABRE suite. On the 26 circuits and the 43-edge IBM Q20 Tokyo design graph of the SABRE paper [27], with SWAP insertion only and three added gates per SWAP, the methods progressed as follows. Two circuits (qft 13, qft 20) are not in the public benchmark repository and were regenerated (338 and 800 gates against the paper’s 403 and 970), so their reference is our SABRE reimplementation. Session 1 rewrote SABRE in C++ with early-abort randomised restarts and a hill-climb over initial mappings (36,435 added gates), then a recedinghorizon rollout router (24,228). In session 2, two proposers independently implemented a direction the notes listed as untried, a beam search over SWAP sequences, whose state is the mapping plus per-qubit progress; together they reached 12,261. Session 3 refined this to 12,087. On identical inputs, Qiskit 2.4.1’s SABRE gives 64,137 (53,418 with heavy trial settings) and pytket 2.18.5 gives 35,685. On the 15 circuits reported by later routing papers, the best published counts $( { \mathrm { H A I L } } ,$ 2025 [28]) total 13,044, against 9,312 here, and our count is lower on every one of those circuits [17]. Beam search for routing is not new. The routers also spent minutes to an hour of CPU per circuit, against seconds to minutes for the stock baselines (up to 23 minutes per circuit for heavy Qiskit), the run-to-run spread is about 8%, and the graph is the design graph rather than the 35-coupler device, so this is a statement about headroom in a widely used benchmark, not a fair comparison of algorithms.

Combinatorial optimisation. On the internal TSP task (hidden instances, n from 600 to 2000, scored against a BHH optimum estimate), two iterations with two proposers each raised the private score from 0.58 to 0.88. The proposers converged on chained 2-opt with Or-opt and segment-swap kicks. While calibrating this family we found that any reference built from a hand-written local search was matched by the agent within one session, so references must come from optima or stronger solvers.

## 3.4 Machine learning

AdderBoard. The task is to find the smallest transformer that adds two 10-digit numbers with full accuracy [29]. Working from the rules alone, a proposer designed a 5-parameter model in one 8-minute, \$1.05 run. It has one attention layer and two heads keyed on a period-11 sinusoidal positional encoding, a scalar tied embedding and read-out, and a sigmoid carry gate; the carry arises only through autoregressive feedback. It passes the upstream verifier on 10,010 of 10,010 additions across three seeds. A parallel proposer found an independent 6-parameter design. The next session gamed the parameter counter (Section 4).

A 20M-parameter language model. The task was to find an architecture of at most 20M parameters, embeddings included, that beats the OpenAI GPT-2 124M checkpoint on FineWeb validation (1.0658 bits per byte), training only on FineWeb-10B (GPT-2 itself was trained on WebText). The scorer retrains each candidate from its code on one H100 for $1 0 ^ { 9 }$ tokens, so a gain must come from the architecture or recipe. The seed was a scaled-down modded-nanogpt [30] at 1.3376. Session 1 reached 1.2868 with recipe changes and learnable residual mixing; it made no architectural change, because the bandit drew only opt-era arms. After the operator replaced the ML arms, session 2 reached 1.2716 with four unique blocks looped twice over eight layers. That closes 24% of the gap. The Modal workspace spend limit then stopped the session. This is the clearest case in which the Infinity Machine did not reach its target.

## 3.5 Hardware: a low-energy systolic matrix tile

We asked whether the same loop can design hardware when the score is measured power, not a proxy. The design is the 8×8 INT8 weight-stationary systolic array (32-bit outputs) from LogikBench, as shipped with OpenROAD-flow-scripts [31] for the ASAP7 predictive 7 nm PDK [32]. Proposers may rewrite its RTL freely but must keep its ports and its cycle-level protocol, so throughput is fixed at 64 MACs per cycle. The evaluator, written by the operator, runs every candidate through the full flow at a fixed 600 ps clock: Yosys synthesis, floorplan, placement, clock-tree synthesis, detailed routing and parasitic extraction. It then applies five gates:

• the RTL matches a Python golden model on random and corner-case tiles (Icarus Verilog);

• the routed netlist, simulated with functional ASAP7 cell models, is bit-exact on the same tiles;

• setup and hold worst slack are both $\geq 0 ;$

• there are 0 routing DRC violations;

• the area is at most 1.1× the original’s.

The score is the energy per MAC on an inference-like workload: post-ReLU activations (about half zero) and roughly Gaussian weights, 16,384 MACs over a 456-cycle window that includes the drain cycles between tiles. OpenSTA computes the power from the gate-level VCD of that window and the routed SPEF. Proposers see the result for workload seed 2026, and the kernel’s private score uses seed 7331. A failed gate scores zero. The original RTL measures 578.9 fJ/MAC (34.7 mW; area 5,559 $\mu \mathrm { m } ^ { 2 } )$

The search ran in two sessions with nine recorded proposer runs (\$11.0 in total). Table 2 lists the steps that set a new best, from the agents’ notebooks, and Figure 5 shows every scored proposal and where the power went. The final design uses 7.7 mW at the same clock and throughput, and is 77.7% lower in energy per MAC than the original RTL. It is also 55% smaller, with setup and hold slack of +5.2 and +6.3 ps. A same-run ablation attributes about 43 fJ of the last step to the ring bufer and 16 fJ to writing the weights in place. Two of the agents’ own observations shaped the search. The first was that the 64 input bits are the narrowest point of the pipeline, so it is cheaper to delay the activations than to register partial sums. The second was that flops, and the clock that drives them, dominate the power. The search also found negative results: a radix-4 Booth multiplier with activations encoded once per row was worse (300.8 fJ), and eight rows per stage would not close timing at 600 ps. One proposer declined to ship a 1.6% better variant because its hold slack was only +0.197 ps, reasoning that a gate failure on the private run would score zero.

This result has clear limits. ASAP7 is a predictive PDK, so these are modelled, not measured, energies. There is one flow, one clock and one synthetic workload distribution. The gate-level simulation is zero-delay, so glitch power is not counted. The ring-bufer design relies on the protocol’s weight load of exactly eight w valid cycles per tile, which the testbench uses but which a diferent host could violate. The comparison is against the original RTL, not against a hand-optimised design. The operator built the evaluator, wrote an “already measured” table between sessions, corrected the score range twice, and reseeded session 2 with session 1’s best design (Section 4).

(b) Power breakdown (leakage < 0.1 mW)
<table><tr><td>Run (arm)</td><td>fJ/MAC</td><td> $\mu \mathrm { { m } ^ { 2 } }$ </td><td>Change</td></tr><tr><td>original RTL</td><td>578.9</td><td>5,559</td><td></td></tr><tr><td>s1 it0-p0 (data structure), v1</td><td></td><td>346.9 3,773</td><td>activations broadcast along rows, no output de-skew; per-row partial sums 16-19 bits wide; no datapath</td></tr><tr><td>s1 it0-p0, v2 (s2 seed)</td><td></td><td>274.9 3,766</td><td>resets latch+AND clock gates per pipeline depth, enabled from the valid pipe; a gate on the weight registers</td></tr><tr><td>s2 it0-p0 (conservative)</td><td></td><td>252.8 2,998</td><td>v2&#x27;s gating plus a parallel run&#x27;s Baugh–Wooley multi- plier, 3:2 carry-save tree with the partial sum folded</td></tr><tr><td>s2 it0-p1 (new neighbourhood)</td><td></td><td>188.2 2,427</td><td>in, Brent-Kung adder four PE rows per pipeline stage: a Python generator emits one 33-operand bit-level Dadda tree and one</td></tr><tr><td>s2 it2-p0 (new neighbourhood)</td><td></td><td>128.8 2,507</td><td>adder per stage; 4 partial-sum stages → 2 activation shift chains replaced by a 13-entry ring buffer with a gated clock per entry; weights written in place per row instead of shifted</td></tr></table>

Table 2: The steps that set a new best on the accelerator tile (public workload; every step passed all five gates). The final design measures 130.1 fJ/MAC on the private workload.

![](images/39c44c0d32484d82d493ca80e1da3b9e4be82859265bdb895f98ea13941f21d2.jpg)

![](images/42d7025d86a361595d3ffb0fa2c75e20dbc212d6bb473e6a7dea6719363498f1.jpg)  
Figure 5: The accelerator tile. (a) Energy per MAC of every scored proposal in completion order (dots; orange set a new best) and the best so far (line). Session 2 started from session 1’s best design. Run labels are iteration and proposer. (b) Power at the fixed 600 ps clock and unchanged throughput for the original RTL and the three designs that set new bests, split into internal (cell and clock) and switching power; the total is printed above each bar. Values are the kernel’s publicworkload measurements, from the session banks.

## 3.6 Open-ended discovery in TESS photometry

We pointed catalogue mode at TESS [33] light curves of about 17,000 stars: 9,779 director’sdiscretionary targets and 7,474 M dwarfs within 40 pc observed only in full-frame images. Four sessions cost about \$58. The referee was rewritten four times: twice to close loopholes that proposers exploited (Section 4), once after the operator’s manual vetting found blends, eclipsing binaries and aliases, and once to fix a cadence bug the operator found. The final referee accepts 47 of 60 known TOIs and 30 of 49 known FFI planets, and accepts none of 80 random fake claims. One candidate survives: G 249-11 (TIC 417732194), an M4.5 dwarf at 29 pc with a 3 ppt, 0.9 h transit-like signal at $P = 5 . 3 0 7 4 2 5 \mathrm { d }$ , which implies $R _ { p } \approx 1 . 6 R _ { \oplus }$ . An ephemeris fitted to two sectors predicted dips in two later sectors that were never used for selection, with $p \approx 0 . 0 0 2$ from 400 full-pipeline nulls [18]. That is about 3σ, and a bound eclipsing binary and a faint neighbour are not excluded, so we report a candidate, not a planet. An independent read-only reviewer agent caught a 1000-day error in the reference epoch and an independence error in the first vetting report.

## 3.7 Null results

On the Einstein Arena [34] Sidon (4,5)-set problem, the Infinity Machine found 13 new sets tying the $4 / 7$ record but none below it. The operator’s feasibility analysis (not the agent’s) shows that within the verifier’s range only n = 16 or n = 18 could beat $4 / 7 .$ . On the 51 × 51 Hadamard maximal-determinant problem, proposers established that the record [35] is a fixed point of every local method they tried. Their best alternative family $( \mathbb { Z } _ { 5 } \times \mathbb { Z } _ { 5 }$ blocks) reaches 43.2310 against the record’s 43.3013. A further exhaustive argument ruling out one circulant family was again the operator’s. For the largest seven-qubit distance-two code $( 2 4 \leq K \leq 2 6 [ 3 6 , 3 7 ] )$ , hundreds of restarts all land on one plateau per method. For four bases in $\mathbb { C } ^ { 6 }$ , every proposer did basin hopping around the known optimum [38], and about 35,000 re-ascents returned to it. Where the published record is a robust optimum of random-restart search, the Infinity Machine did not improve it, and the useful output was a measured negative.

## 3.8 Fully autonomous end-to-end runs

We launched autonomous mode three times on 26 October 2026, each as a daemon job with a \$30 cap, and did not intervene in any of them. The first run had the default goal: find an open problem in mathematics, quantum information or theoretical computer science whose best known result a program can check exactly, and improve on it. It took 1 h 42 min and \$9.86 (session 20261026-002501-autonomous). Table 3 lists its phases.

The first run produced no new result. It does, however, show the full chain working with no human step, and both of its safeguards fired for real.

• The audit prevented a false discovery. Without it, the search would have “beaten the record” of 215 by rediscovering a public code, and the report would have claimed it.

• The independent check confirmed the verifier with code the verifier’s author never saw. The run also showed two weaknesses.

• The scout checked papers but not leaderboards or discussion threads. The auditor found the newer codes there.

• Autonomous tasks fell back to the default strategy arms, so the last round repeated the conservative no-op failure of Section 4. We fixed this for runs 2 and 3.

The chosen problem was also a crowded one: many agents had pushed the public record from 216 to 223 in five weeks.

<table><tr><td>Phase</td><td>$</td><td>Outcome</td></tr><tr><td>Scout, round 0</td><td>3.14</td><td>Chose the largest binary two-deletion-correcting code of length 16, a maximum independent set in a 65,536-vertex graph (Einstein Arena problem 34). Cited a 2026 paper&#x27;s 215 as the record, used the public 208-word code as the seed, and wrote the verifier and calibration</td></tr><tr><td>Validation Audit, round 0</td><td></td><td>cases. Passed: seed and calibration reproduced. 1.32 Failed. The record was stale. The arena leaderboard already listed</td></tr><tr><td></td><td></td><td>public codes of 216–223 words; the auditor downloaded the 223- word code and verified it with the task verifier and its own checker. Kernel replay of 18 exploit files: all rejected. Repaired the task: record 223, seed the 223-word code, score range</td></tr><tr><td>Scout, round 1</td><td>0.85</td><td>extended to 228. Passed: open, record current, no exploit accepted. Task installed</td></tr><tr><td>Audit, round 1 Discovery</td><td>0.58</td><td>with automatic approval. Three rounds of two proposers. They tried a plateau local search in C  $( \approx 1 . 9 \times 1 0 ^ { 8 }$ </td></tr><tr><td></td><td></td><td>candidate probes per second per core), an ex- act large-neighbourhood search that frees 24–60 codewords and re- solves them by branch and bound, and a stock iterated local search. None found a 224-word code. Round 3 drew conservative twice, and both runs were no-ops.</td></tr><tr><td>Independent check 0.19</td><td></td><td>The agent wrote its own checker from the problem statement, plus a pairwise longest-common-subsequence test (maximum 13 &lt; 14). Result: valid, 223 words, in agreement with the verifier.</td></tr><tr><td>Report</td><td>0.32</td><td>“No improvement: the best verified code has 223 words, which ties (does not beat) the public record of 223.&quot; The claim boundary credits the record&#x27;s authors.</td></tr></table>

Table 3: Autonomous run 1, phase by phase, from the session’s event log and the agents’ outputs.

Runs 2 and 3. We then changed two things and ran autonomous mode twice more, in parallel. First, the autonomous task family now uses the research-shaped math arms (literature transfer, structured construction, new objective, scale compute, exhaustive small case). Second, each goal named a specific table instead of leaving the field open. We chose the tables from a survey of quiet, verifiable records (Appendix B). Run 2 targeted Dobbelaere’s sorting-network size tables for 25 to 40 inputs, and run 3 Brouwer’s table of mixed binary/ternary codes. Within those tables the scout chose the cell, confirmed it was current, and wrote the verifier, seed and calibration cases. Table 4 summarises both runs. Both tied the record. In both, the scout’s selection reasoning was specific and checkable, the verifier survived its audit, and the final report declined to claim anything. Their near misses are informative but are not results: a network that leaves one input unsorted is not a sorting network. Four things went wrong.

• A reversed score in run 2, missed by every check. The kernel’s minimising scorer gives score 1 at the first end of the task’s score range and 0 at the second. The scout wrote the range as 186 to 175, intending the opposite. As installed, the 186-comparator Batcher network scored 1.0, the record 0.455, and a record-beating 179 would have scored 0.364, below its parent. Kernel validation checked values but not the direction of the score. The auditor read the mapping as intended and reported 0.545 for the record. The reporter noticed the mismatch between 0.545 and 0.4545 but not its cause. No proposer shipped a larger network, so the outcome was unafected. Had one found a 179, though, the bandit and the parent sampler would have treated it as a regression. Worse, the independent check and the report take the highest-scoring solution, so the 179 would have stayed in the bank unreported. The operator had made the same mistake on the chip task (Section 4). Validation should check that calibration cases with better values score higher.

<table><tr><td></td><td>Run 2</td><td>Run 3</td></tr><tr><td>Chosen problem</td><td>Size of a sorting network on 31 inputs; Mixed code N(4, 5, 3), 4 binary and 5 unchanged since 2017), bounds 141– 186–216, lower bound from 1998</td><td>record 180 comparators (Batcher-type, ternary coordinates, distance 3; table</td></tr><tr><td>Why this cell</td><td>lutionary search, while n = 28–30 were not</td><td>Its entry is a 32-input merge network 16% gap; neighbouring cells were im- minus one wire, never improved by evo- proved in 2010–2013 and this one was</td></tr><tr><td>Scout</td><td>$2.03, 19 min; exact 0-1 verifier over all $4.66, 47 min; its own coset-union 231 inputs (≈15 s)</td><td>search reproduced 186 and other table entries; 12 calibration cases</td></tr><tr><td>Audit</td><td>$0.89; pass; 23 exploit files, all rejected on replay</td><td>$1.14; pass; 24 exploit files, all rejected on replay $4.12, 6 runs: SorterHunter-style evolu- $3.60, 6 runs: coset unions, about</td></tr><tr><td>Discovery</td><td>ber of unsorted inputs; 2 runs searched code nothing</td><td>tion, a witness-vector variant, anneal- 30,000 prescribed automorphism ing at 179 comparators on the num- groups, tabu search, LNS from the 186</td></tr><tr><td>Near miss</td><td>179 comparators with one unsorted 0-1 189 words (63 cosets) with one conflict- vector</td><td>ing pair</td></tr><tr><td>Check, report</td><td>$0.55; independent C checker agrees;</td><td>$0.58; independent checker agrees; “no</td></tr><tr><td>Total</td><td>“no improvement&quot; $7.59, 2 h 38 min</td><td>improvement&quot; $9.97, 2 h 55 min</td></tr></table>

Table 4: Autonomous runs 2 and 3 (sessions 20261026-024005-autonomous-e33eb and 20261026-023958-autonomous-67f31), from their event logs, scout selections and reports. Both ended in ties with the record.

• Declined searches. In run 2, two of the six proposers (exhaustive small case and new objective) judged further search hopeless and returned their parent without running a new search; one estimated 10<sup>9</sup> candidate repairs as too many for its budget. This is the conservative no-op of Section 4 in a new form, now under research arms. All six run-3 proposers ran searches.

• Rate-limited literature checks. In run 3 both the scout and the auditor were throttled by arXiv and Semantic Scholar (HTTP 429). The task was installed on the strength of the table page alone, and the report says so.

• A colliding session. Two jobs launched in the same second first shared one session directory. We cancelled both before their scouts finished, kept the directory, and added a random sufix to session names before relaunching. The runs reported here are the relaunches.

Across three runs, autonomous mode chose reasonable problems, wrote verifiers that survived 18–24 replayed exploits each, caught one stale record, and reported three honest nulls for \$27.42 in total. It also installed one task whose score pointed the wrong way, and none of its checks noticed. It has not yet made a discovery.

## 4 What the loop learned, and what it did not

The self-improvement gate rejected the one self-derived generation. The first 16 problem runs (AdderBoard, Sidon, Hadamard) never reached the system’s memory: two kernel scanners read only top-level runs. After that was fixed, one consolidation pass over those 16 proposer runs wrote four playbook bullets from the proposers’ own notes:

• do not re-run neighbourhoods that earlier notes already tried;

• a faster optimiser on a proxy is not progress unless the proxy tracks the real score;

• pilot symmetry-restricted families for 60 s before committing budget;

• use the shipped checker.

Graded by the self-improvement gate (\$8.22, 57 minutes), the generation with these bullets scored 0.944 on held-in tasks against the incumbent’s 0.927, and 1.000 on held-out tasks against 1.000. Its cost was 2.02 times the incumbent’s, above the gate’s 1.5 limit, so it was rejected. The bullets made the agent run more experiments, which bought a small gain at twice the spend. No metaagent proposal ran in that session, because the single iteration was spent validating this pending candidate; in an earlier smoke run the meta agent proposed no change. Every discovery result in Section 3 was therefore produced by the initial genome. The gate behaved as designed. Whether 2× the cost for +0.017 is worth it is a policy question for a human, and with one seed the gain is within noise. We do not yet have evidence that the Infinity Machine improves itself on these problems.

Strategy arms mattered, but a human chose them. Over the first 16 runs the bandit’s data were unambiguous. Data structure earned a mean reward of 0.4 over 5 pulls, and every other arm earned 0 over 11. Conservative and robust/simple pulls on a record seed were pure no-ops: proposers shipped the seed unchanged, at about \$0.8 in total. The bandit would have learned to avoid those arms eventually, but on problems with only 4–6 pulls per session it could not. The operator replaced the arms by hand for math and ML (and wrote the astronomy arms). For math (structured construction and the others), the first session afterwards produced $\chi ( | T \rangle ^ { \otimes 7 } ) \leq 1 0$ . For ML (architecture family, parameter sharing and the others), the next session produced the looped model. Under the rule that lessons must enter through the system’s own loops, these were kernel changes, not self-improvement, and we report them as such. A version of the Infinity Machine that proposed its own strategy arms through the gate is the obvious next step.

Specification gaming within one iteration. Strong proposers exploited lenient scorers four times, usually within a single iteration (items 1–4); item 5 is a hole in a referee that the operator, not a proposer, found. The original artefact for each item (code, formula, the agent’s own words) is in Appendix E.

1. AdderBoard. With the score saturating at 6 parameters, the next session “reached one parameter” in 2.5 minutes by turning four weights into Python constants. Strict counting, under which constants acting as weights count, restored the true count of 5. The proposers disclosed the loophole when asked for review risks.

2. Seven-qubit code. A partial-credit formula paid for any residual below 1, so an unconfirmed K = 25 subspace at residual 0.072 outscored the confirmed K = 24 code. Credit now starts only below $1 0 ^ { - 2 }$

<table><tr><td>Outcome</td><td>Runs</td><td>Notes</td></tr><tr><td>Beat its parent&#x27;s public score</td><td>39</td><td>incl. 3 gamed outright (2 Adder- Board, 1 seven-qubit code) and 5 TESS runs scored by referees later found exploitable</td></tr><tr><td>Matched its parent, no gain</td><td>58</td><td>incl. at least 10 that delivered the parent essentially unchanged, per</td></tr><tr><td>Worse than its parent No valid solution</td><td>2</td><td>their own notebooks both TESS, under a stricter referee 4 stopped by the Modal spend limit</td></tr><tr><td></td><td>5</td><td>(LM), 1 by the step limit (TESS)</td></tr><tr><td rowspan="2">Hit the per-run spend cap (overlaps the rows above) Lost before scoring, not in the records</td><td>5</td><td>introduced after one run cost $7.15</td></tr><tr><td>3</td><td>2 killed with the host (C11), 1 lost when its shell died (SABRE)</td></tr></table>

Table 5: Outcome of every recorded proposer run in the fourteen problem campaigns (104 runs, 27 sessions; the 18 proposals of the three autonomous runs are reported in Section 3.8, and the nine of the accelerator campaign in Section 3.5), computed from the kernel’s event logs; the notes come from the run notebooks and Section 4.

3. TESS, referee v1. Proposers claimed “single transits” 500–1200% deep (flux glitches), flux steps and about 90 cheap eclipsing binaries. The fix added physical bounds, a step test and a sky-neighbour systematics test.

4. TESS, referee v2. A proposer wrote that “extra credit lies in uncapped single events and duo transits” and submitted 240 single and 39 two-event “transits”. The fix requires at least three events and caps singles and duos.

5. TESS, referee v4 (not gaming). The referee itself had a hole: it used one cadence per star, so signals that appeared in only one observing era passed. Leave-one-era-out tests found it, and v5 checks cadence per sector and requires consistency across eras.

The pattern matches reports from Hyra and the DGM [4, 10]. The defence that worked was not a better prompt. It was a private split the agents could not see, together with adversarial calibration of the verifier: known positives, injected negatives, and an independent reviewer.

The outcome of every proposer run. Table 5 classifies all 104 proposer runs recorded in the 27 discovery sessions, by comparing each run’s public score with its parent’s. Seven in ten runs did not beat their parent. The table counts beating the parent, which is weaker than setting a new best (Figure 6), and it takes the scores at face value. Eight of the 39 parent-beating runs owe their gain wholly or partly to a scorer we later fixed.

When the agents caught our mistakes. Several runs reported errors in the operator’s own scafolding instead of exploiting them, as AIDE<sup>2</sup> also observed [6]:

• Both SABRE proposers in session 2 reported that the shipped local verifier had unfilled {BASE}/{TARGET} placeholders and crashed. They worked around it and said so.

• A SABRE proposer noticed that two benchmark files are byte-identical. Another proved by exhaustive placement that one circuit has no zero-SWAP embedding, so the published count is optimal.

• A MUB proposer found that the operator’s polar retraction gives NaN gradients at the identity basis, which the record contains, and documented the fix.

• The AdderBoard proposers listed the constants loophole under “review risk” in their final message. The TESS proposer stated its intent to exploit the uncapped classes in its notebook.

In two of the four gaming incidents, then, the exploit was stated in the agents’ own reports before the operator found it in the scores. Reading the notebooks is a cheap detector.

## Infrastructure lessons that became kernel features.

• A proposer that ran 40 minutes on 2.1M input tokens cost \$7.15 by itself, which pushed a \$10 session to \$13.88. This led to the per-run spend cap.

• A proposer that hit the step limit had its work discarded. The kernel now streams results and keeps the last state.

• A discovery session launched from a foreground shell lost a proposer run unharvested when the shell died. Sessions now run as daemon jobs.

• A verification sandbox with a hard-coded 2 GB limit OOM-killed a valid n = 7 qutrit evaluation. Verifier resources are now set per task.

• Two autonomous jobs launched in the same second shared a session directory. Session names now carry a random sufix.

• A full chip evaluation takes 10–12 minutes, and the kernel re-scores a parent whenever it is sampled. The verifier’s raw output is now cached by verifier, solution, split and image, and the score is recomputed from it on every read, so a corrected score range does not invalidate the cache.

Each of these looked minor, and each would have silently cost a result.

Mistakes in the chip evaluator. The accelerator task needed four fixes from the operator before its scores could be trusted, and two of them repeat earlier lessons.

• Score range reversed. The first launch mapped the original design to a score of 1. We cancelled it before any proposer ran. An autonomous scout later made the same mistake (Section 3.8).

• Saturation, again. In session 1 the score reached 1 at 289 fJ, half the baseline. Both firstround proposers beat that bound in their first run, so later runs had nothing to gain, and an aggressive rewrite proposer shipped its parent unchanged. This is the AdderBoard saturation of Section 4 without the gaming. We cancelled the session, set the score-1 point to 100 fJ, and reseeded session 2 with session 1’s best design.

• Silent simulation. Icarus left the internal delayed signals of the ASAP7 sequential cell models undriven, so the first gate-level runs produced no outputs at all. The evaluator now compiles functional versions of the cell models.

• The wrong DRC count. The first parser read the DRC count from the first routing iteration rather than the last. It now reads the final count by exact key.

None of these favoured a candidate, but each would have produced wrong numbers in the tables above if a person had not read the logs.

Where progress came from within a session. Figure 6 plots every scored proposal on six problems in the order it ran, with the best value so far. Three patterns hold across them.

(a) Qubit depolarizing  
![](images/2ac6fa40af4a0d3bfad415adcd8e74567a81a7c62b79a9d39b5dbed0d78b1c92.jpg)

(b) Qutrit depolarizing  
![](images/9bbeff15ce0670662c5a668159a993f55fc008500dc1466a0f421bfa145dd9ad.jpg)

![](images/7833c8f2659536f7d2d644ffc033b3f93e6b8b144038c612103dee674fd9fcfd.jpg)

(d) Stabilizer rank, m = 7  
![](images/6dea8423a44b091292dc7d37858e5bc41689d19bb1d2b2024a11ab8ad4a3b139.jpg)

(f) 20M-parameter LM  
(e) SABRE routing  
![](images/3bd9768aa2179247374a83e87c3819fdad56e407f19d7a8b1fa3e3cf73f89f47.jpg)

![](images/d66aa639b58b4dfbc25e0d2a4852437d4a80929f5f4fe5e015560a942715479d.jpg)  
Figure 6: Every scored proposal (dots) and the best value so far (line), in chronological order across sessions; dashed lines mark session boundaries, and orange dots set a new best. A session’s seed can raise the bar between sessions when the operator combined earlier runs (the SABRE percircuit merge). Values are those the kernel recorded for each run: (a) the claimed float threshold (the certified point is 0.064966); (c) two-qubit gates + 0.1× depth; (d) 12/value, i.e. the number of stabilizer states; (e) per-run mean ratio to SABRE, before the host’s per-circuit combination (0.5073); (f) excludes four session-2 runs stopped by the Modal spend limit. Extracted from the run banks.

• Gains come early in a session. Every improvement was set in the first or second iteration of its session; no third-iteration run set a new best on any problem.

• A few arms carry the gains. The runs that set new bests drew aggressive rewrite (qutrit twice, Golay, SABRE), new neighbourhood (C11, the SABRE beam search, two LM steps), data structure (SABRE), structured construction (stabilizer rank), and connectivity and parameter sharing (the LM). Conservative set a new best twice (SABRE, LM) and robust/simple once (SABRE), only where the seed was far from optimal; on a record seed they never did.

• Most proposals do not improve. Of 47 scored proposals on these six problems, 16 set a new best. The search is a sequence of rare large steps between plateaus, not a steady climb.

The accelerator task, which is not in the figure, fits the first two patterns only in part. Its last and largest step came in the third iteration of session 2, from new neighbourhood. A conservative run there did useful work, because the notes held a concrete untried combination: it merged the seed’s clock gating with a parallel run’s arithmetic (274.9 to 252.8 fJ). A second conservative draw on the 188.2 fJ design shipped it unchanged.

Cost. The fourteen problem campaigns used about \$171 of LLM usage, and about \$180 with the gated evolve run, plus Modal compute. The three autonomous runs cost a further \$27.42 end to end, and the accelerator campaign \$11.0 plus Modal CPU time. Single finding runs cost between \$0.60 (the C11 state) and \$4.15 (the $T ^ { \otimes 7 }$ decomposition). Host-side certification was sometimes the expensive part: about 45 CPU-minutes per endpoint for the C11 certificate.

## 5 Safety

A system that edits its own harness and runs code it wrote needs containment that does not rely on the model’s cooperation. The Infinity Machine enforces the following structurally, in the kernel.

• Sandboxing. Every proposer and every scorer runs in a fresh container or Modal sandbox. Network is of by default, the root file system is read-only, and data volumes, where needed, are mounted read-only. Kernel code, private splits, scorers and credentials are never mounted, so an agent cannot read its hidden grade or change how it is graded.

• Immutable grading. The kernel scores solutions in a separate sandbox and never trusts an agent’s report of what it ran. A genome edit that touches kernel paths, or changes a kernel-owned file by a single byte, is rejected before grading.

• Bounded spend. A per-run cap ends an agent at its next model call once its cost reaches the limit, and a session budget stops new proposers.

• Human gates and their machine replacement. A self-modification whose gain exceeds the anomaly threshold is parked for review instead of activated. Every generation can be rolled back. Outside autonomous mode, proposed evaluators are inert until a human approves them. In autonomous mode that approval is automatic only after three checks: kernel calibration on known cases, an adversarial audit whose exploits are replayed, and a blind independent re-check of the final result. The approval and its basis are logged. A verifier written by one agent is therefore never trusted on that agent’s word alone. In the first of the three autonomous runs, this chain rejected the first task.

• No outward actions. Agents cannot publish. Every external step in this paper was taken by the operator: the Zenodo deposits, the catalogue issues and pull requests, and the arena posts. Autonomous mode has network access only in the scout and audit phases, to read catalogues and the literature, and it ends with a report, not a submission.

What we observed is mostly in Section 4. Proposers gamed lenient scorers four times, and in two cases said so openly: once when asked for review risks (AdderBoard), and once in a notebook (TESS). We saw no attempt to read hidden splits or to escape a sandbox. However, unlike STOP [3], we did not audit traces systematically for such attempts, so this is an absence of observed incidents, not a measured rate. The main residual risk is not the agent but the harness: our private splits were only as good as the verifiers the operator wrote, and two of them were exploitable for a full iteration before they were fixed.

## 6 Related work

LLM-guided program search. FunSearch [2] evolved short Python functions scored by an evaluator. AlphaEvolve [1] scaled this to whole files, rich prompts and many problems, and Georgiev et al. [39] applied it to 67 problems in mathematical analysis, geometry and combinatorics. ShinkaEvolve [40] and Hyra [10] are open, sample-eficient variants; Hyra uses agentic proposers and lets the evaluator evolve. AIDE [41] searches over ML code with a tree of drafts. The Infinity Machine shares their reliance on automatic evaluation but difers in kind: its proposals are hour-long agent runs rather than sampled programs, it also searches open-endedly in real data, and its own harness is subject to gated self-modification.

Self-improving agents. STOP [3] improved its own scafolding program. The Darwin G¨odel Machine [4] evolved coding agents with an archive and documented reward hacking of its own tests. Hyperagents [5] made the meta agent editable. AIDE<sup>2</sup> [6] rewrote a research agent’s harness under a private grade it never sees. Prime Agent [42] and Continual Harness [43] keep typed supplemental state with rollback by ID, building on recursive language models [44]. ACE [11] curates an itemised playbook. Weng [9] surveys harness engineering for self-improvement. The Infinity Machine combines an editable meta agent, an archive, a cost-aware gate and an ACE-style playbook, and adds an honest report that, so far, its discovery results come from the unmodified genome.

Agents for science. ReAct [45] and Reflexion [46] established tool use and verbal feedback. The AI Scientist [47] automated the paper cycle in ML. Our emphasis is narrower and older: a claim counts only if a program that the agent did not write can check it.

## 7 Discussion

What worked. Giving a strong model a shell, an hour and an exact verifier was enough to move four catalogued open problems and one standard benchmark in six days for under \$200 of model usage. The winning moves were often changes of representation. Two were not suggested anywhere in the operator’s notes:

• counting mixed qubits instead of non-Z qubits (Golay);

• shrinking n instead of growing it (qutrit).

Two others implemented directions the notes listed as untried: a symmetric dictionary instead of annealing over all stabilizer states (stabilizer rank), and a beam over SWAP sequences instead of a greedy SWAP choice (SABRE). The public/private split and adversarially calibrated verifiers, not instructions, kept the outputs honest. The same idea carried over to autonomous mode. When an agent writes its own evaluator, the evaluator is trusted only after machine checks that a diferent agent and the kernel perform. In the first end-to-end run these checks caught a stale record that would otherwise have become a false discovery. The approach also carried over to hardware. With power measured from gate-level switching activity on a routed design, and correctness checked bit-exactly at both RTL and gate level, the proposers cut energy per MAC by 77.7% through architectural changes (dataflow, storage structure, clock gating, generated arithmetic) rather than tuning.

What did not. Self-improvement has not yet earned its name in our data. The one self-derived generation was rejected on cost, the most consequential harness changes were the operator’s, and cross-session memory ran through hand-edited notes. Where a record is a robust optimum of random-restart search (MUBs, Hadamard, the seven-qubit code), the Infinity Machine reproduced the record and produced negative evidence, not progress. Proposers sometimes found an object without understanding it, as with the product structure of the qutrit state. All three fully autonomous runs ended in ties with public records. The first scout chose a crowded problem, and the first run’s final round was wasted on conservative no-ops. Research arms and narrower goals in runs 2 and 3 produced real searches and near misses, but no improvement, and two proposers still declined to search. In run 2 the machine checks that replace human approval passed a task whose score was reversed. They test whether a verifier accepts the right objects, but not whether the score ranks them the right way round.

Limitations. All results come from a single operator, a single proposer model and one to three sessions per problem. There are no repeated seeds, ablations or matched-compute comparisons. Several problems were chosen because they looked search-shaped, so the success rate is not representative. The quantum results are lower bounds or explicit constructions with stated claim boundaries, not optimality results. The SABRE comparison is not compute-matched. The TESS candidate is unconfirmed. The accelerator result rests on a predictive PDK, one flow, one clock, one synthetic workload and zero-delay gate-level simulation, and it compares against the original RTL rather than a hand-optimised design. End-to-end autonomy has been demonstrated three times, all with null results, and in runs 2 and 3 the operator named the table to search. The fifteen operator-led campaigns that produced the results above were not run in autonomous mode, so none of their results is an autonomous end-to-end discovery.

Next steps. Next we plan to: run autonomous mode repeatedly with open goals, with a scout that also checks leaderboards and discussion threads before choosing, a validation step that checks the score’s direction on calibration cases, and a brief that forbids returning the parent without a search; repeat the accelerator task on a second PDK, at several clocks and workload seeds, with timingannotated simulation; let the meta agent propose strategy arms and brief templates through the gate; replace operator-written “already measured” tables with consolidation output, and measure whether the next session does better; run matched A/B comparisons of gated generations on heldout problem families (imp@k [5]); and add a reviewer round to every catalogue-mode session.

## Use of AI systems

The discoveries reported here were made by Claude Sonnet 5.5 proposer agents inside the Infinity Machine. The harness, task verifiers, seeds, figure scripts and this manuscript were written by the author with the assistance of Claude (Anthropic) as a coding and writing assistant, under the author’s direction. The author chose the problems, checked every reported number on the host, and takes responsibility for the content.

## Data availability

States, circuits, decompositions, routings, certificates, agent notebooks and verification code for the published results are on Zenodo [12–18]. The accelerator designs, evaluator and notebooks, and the three autonomous session directories, are not yet deposited.

## Ethics statement

The problems studied are in mathematics, quantum information, compilers, machine learning, astronomy and digital hardware design, and carry little direct potential for harm. The broader concern with self-improving agents is that they may optimise a proxy in unintended ways. We observed exactly this on a small scale (Section 4) and report every instance. All external claims went out under the author’s name after host-side verification, with AI involvement disclosed in each deposit, and none was posted by an agent. The TESS candidate is reported as unconfirmed, and no follow-up request was sent on the agent’s behalf.

## Reproducibility statement

For every published result, the Zenodo deposits contain the solution object, the verifier, the task text given to the agents, the agents’ notebooks and search code, and SHA-256 sums. Anyone can re-check the results without our harness. Appendix C lists every discovery session with its configuration and spend, Appendix D the strategy-arm texts, and Appendix F the gate configuration and the selfderived playbook. For each autonomous run, the session directory keeps every artefact: the goal, each scout round’s task, the validation reports, both audits with their exploit files and replay results, the automatic approval record, the search bank, the independent checker’s code and verdict, and the final report and claim. Appendix B summarises it. Model runs are not deterministic: rerunning a session will produce diferent proposals, and we report only one session sequence per problem. The model versions were Claude Sonnet 5.5 for proposers and Claude Opus 5.5 for the meta agent.

## References

[1] Alexander Novikov, Ngˆan V˜u, Marvin Eisenberger, Emilien Dupont, Po-Sen Huang, Adam Zsolt Wagner, Sergey Shirobokov, Borislav Kozlovskii, Francisco J. R. Ruiz, Abbas Mehrabian, M. Pawan Kumar, Abigail See, Swarat Chaudhuri, George Holland, Alex Davies, Sebastian Nowozin, Pushmeet Kohli, and Matej Balog. AlphaEvolve: A coding agent for scientific and algorithmic discovery. arXiv:2506.13131, 2025.

[2] Bernardino Romera-Paredes, Mohammadamin Barekatain, Alexander Novikov, Matej Balog, M. Pawan Kumar, Emilien Dupont, Francisco J. R. Ruiz, Jordan S. Ellenberg, Pengming Wang, Omar Fawzi, Pushmeet Kohli, and Alhussein Fawzi. Mathematical discoveries from program search with large language models. Nature, 625:468–475, 2024. doi: 10.1038/s41586-023-06924-6.

[3] Eric Zelikman, Eliana Lorch, Lester Mackey, and Adam Tauman Kalai. Self-taught optimizer (STOP): Recursively self-improving code generation. arXiv:2310.02304, 2023.

[4] Jenny Zhang, Shengran Hu, Cong Lu, Robert Lange, and Jef Clune. Darwin G¨odel Machine: Open-ended evolution of self-improving agents. arXiv:2505.22954, 2025.

[5] Jenny Zhang, Bingchen Zhao, Wannan Yang, Jakob Foerster, Jef Clune, Minqi Jiang, Sam Devlin, and Tatiana Shavrina. Hyperagents. arXiv:2603.19461, 2026.

[6] Dhruv Srikanth, Bingchen Zhao, Dixing Xu, Yuxiang Wu, and Zhengyao Jiang. Recursive self-improvement of AI research agents. arXiv:2609.26457, 2026.

[7] Irving John Good. Speculations concerning the first ultraintelligent machine. In Advances in Computers, volume 6, pages 31–88. Academic Press, 1966. doi: 10.1016/S0065-2458(08) 60418-0.

[8] LangChain. Deep agents. https://docs.langchain.com/oss/python/deepagents/ overview. Documentation, accessed October 2026.

[9] Lilian Weng. Harness engineering for self-improvement. https://lilianweng.github.io/ posts/2026-07-04-harness/, 2026. Blog post, 4 July 2026.

[10] Hyra Team. Hyra: A simple yet efective scafold for general discovery. https://hy.tencent. ai/research/hyra, 2026. Tencent Hunyuan blog post, 21 July 2026.

[11] Qizheng Zhang, Changran Hu, Shubhangi Upasani, Boyuan Ma, Fenglu Hong, Vamsidhar Kamanuru, Jay Rainton, Chen Wu, Mengmeng Ji, Hanchen Li, Urmish Thakker, James $\mathrm { Z o u } ,$ and Kunle Olukotun. Agentic context engineering: Evolving contexts for self-improving language models. arXiv:2510.04618, 2025.

[12] Harish Santhanalakshmi Ganesan. Certified positive coherent information of the qubit depolarizing channel at per-pauli $p = 3 2 4 8 3 / 5 0 0 0 0 0$ (45 qubits, rank-2 symmetric input): state, exact certificates, evaluator. Zenodo, 2026.

[13] Harish Santhanalakshmi Ganesan. A two-dimensional-subspace input beats the hashing point of the qudit depolarizing channel: positive single-letter coherent information at $p = 0 . 2 9 1 0 6 7 2$ for $d = 3 .$ . Zenodo, 2026.

[14] Harish Santhanalakshmi Ganesan. A 44-gate, depth-6 Cliford circuit preparing the [[23,1,7]] Golay logical zero: a new point on the CNOT-count/depth frontier. Zenodo, 2026.

[15] Harish Santhanalakshmi Ganesan. Private information of the qubit depolarizing channel beyond the single-use threshold: explicit finite-n witnesses up to $p = 0 . 2 8 4 6 7$ , and what one-way key distillation already implies. Zenodo, 2026.

[16] Harish Santhanalakshmi Ganesan. An explicit ten-term stabilizer decomposition of $| t \rangle ^ { \otimes 7 }$ $\chi ( | t \rangle ^ { \otimes 7 } ) \leq 1 0$ . Zenodo, 2026.

[17] Harish Santhanalakshmi Ganesan. Beam search over SWAP sequences on the SABRE benchmark: 82% fewer added gates than SABRE and 29% fewer than the best published routings for IBM Q20 Tokyo, found by an autonomous agent. Zenodo, 2026.

[18] Harish Santhanalakshmi Ganesan. $\mathrm { ~ A ~ } 1 . 6 ~ R _ { \oplus }$ transiting planet candidate on a 5.31-day orbit around the nearby M4.5 dwarf G 249-11 (TIC 417732194). Zenodo, 2026.

[19] Jesse Fern and K. Birgitta Whaley. Lower bounds on the nonzero capacity of Pauli channels. Physical Review A, 78:062335, 2008. doi: 10.1103/PhysRevA.78.062335.

[20] Avantika Agarwal, Amolak Ratan Kalra, Sungjai Lee, Debbie Leung, Luke Schaefer, Pulkit Sinha, and Graeme Smith. Enhanced quantum capacity thresholds from symmetry. arXiv:2605.09138, 2026.

[21] Artus Krohn-Grimberghe. A certified lower bound on the quantum-capacity threshold of the depolarizing channel. arXiv:2608.15870, 2026.

[22] Joseph M. Renes and Graeme Smith. Noisy processing and distillation of private quantum states. Physical Review Letters, 98:020502, 2007. doi: 10.1103/PhysRevLett.98.020502.

[23] Oliver Kern and Joseph M. Renes. Improved one-way rates for BB84 and 6-state protocols. Quantum Information and Computation, 8(8–9):756–772, 2008. doi: 10.26421/QIC8.8-9-6.

[24] Michael Doherty, Matteo Puviani, Jasmine Brewer, Gabriel Matos, David Amaro, Ben Criger, and David T. Stephen. Fast stabilizer state preparation via AI-optimized graph decimation. arXiv:2603.17743, 2026.

[25] Sergey Bravyi, Graeme Smith, and John A. Smolin. Trading classical and quantum computational resources. Physical Review X, 6:021043, 2016. doi: 10.1103/PhysRevX.6.021043.

[26] Hammam Qassim, Hakop Pashayan, and David Gosset. Improved upper bounds on the stabilizer rank of magic states. Quantum, 5:606, 2021. doi: 10.22331/q-2021-12-20-606.

[27] Gushu Li, Yufei Ding, and Yuan Xie. Tackling the qubit mapping problem for NISQ-era quantum devices. In Proceedings of the 24th International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS), pages 1001–1014, 2019. doi: 10.1145/3297858.3304023.

[28] Kang Xu, Zeyang Li, Xinjian Liu, Dandan Li, and Yukun Wang. An eficient iterative algorithm for qubit mapping via layer-weight assignment and search space reduction. arXiv:2502.07536, 2025.

[29] AdderBoard: the smallest transformer that adds two 10-digit numbers. https://github.com/ anadim/AdderBoard. Leaderboard repository, accessed October 2026.

[30] Keller Jordan et al. modded-nanogpt. https://github.com/KellerJordan/modded-nanogpt. Accessed October 2026.

[31] Tutu Ajayi, Vidya A. Chhabria, Mateus Foga¸ca, Soheil Hashemi, Abdelrahman Hosny, Andrew B. Kahng, Minsoo Kim, Jeongsup Lee, Uday Mallappa, Marina Neseem, Geraldo Pradipta, Sherief Reda, Mehdi Saligane, Sachin S. Sapatnekar, Carl Sechen, Mohamed Shalan, William Swartz, Lutong Wang, Zhehong Wang, Mingyu Woo, and Bangqi Xu. Toward an opensource digital flow: First learnings from the OpenROAD project. In Proceedings of the 56th Annual Design Automation Conference (DAC), pages 1–4, 2019. doi: 10.1145/3316781.3326334.

[32] Lawrence T. Clark, Vinay Vashishtha, Lucian Shifren, Aditya Gujja, Saurabh Sinha, Brian Cline, Chandarasekaran Ramamurthy, and Greg Yeric. ASAP7: A 7-nm finFET predictive process design kit. Microelectronics Journal, 53:105–115, 2016. doi: 10.1016/j.mejo.2016.04. 006.

[33] George R. Ricker, Joshua N. Winn, Roland Vanderspek, David W. Latham, et al. Transiting exoplanet survey satellite. Journal of Astronomical Telescopes, Instruments, and Systems, 1: 014003, 2015. doi: 10.1117/1.JATIS.1.1.014003.

[34] Einstein arena. https://einsteinarena.com. Accessed October 2026.

[35] Giorgi Butbaia, Justin Tan, Pragatheeswaran Vipulanandan, Xiaoyu Huang, Toby Saunders-A’Court, Lucas Fagan, Davide Passaro, Michele Tarquini, and Sergei Gukov. New records for the hadamard maximal determinant problem. arXiv:2608.22518, 2026.

[36] Eric M. Rains. Quantum codes of minimum distance two. arXiv:quant-ph/9704043, 1997.

[37] Gerard Angl\`es Munn´e and Felix Huber. SDP bounds on quantum codes: rational certificates. arXiv:2603.19901, 2026.

[38] Philippe Raynal, Xin L¨u, and Berthold-Georg Englert. Mutually unbiased bases in dimension six: The four most distant bases. Physical Review A, 83:062303, 2011. doi: 10.1103/PhysRevA. 83.062303.

[39] Bogdan Georgiev, Javier G´omez-Serrano, Terence Tao, and Adam Zsolt Wagner. Mathematical exploration and discovery at scale. arXiv:2511.02864, 2025.

[40] Robert Tjarko Lange, Yuki Imajuku, and Edoardo Cetin. ShinkaEvolve: Towards open-ended and sample-eficient program evolution. arXiv:2509.19349, 2025.

[41] Zhengyao Jiang, Dominik Schmidt, Dhruv Srikanth, Dixing Xu, Ian Kaplan, Deniss Jacenko, and Yuxiang Wu. AIDE: AI-driven exploration in the space of code. arXiv:2502.13138, 2025.

[42] Seth Karten, Alex L. Zhang, Kevin Thomas, Sebastian M¨uller, Elie Bakouch, Daniel Auras, Mika Senghaas, Fares Obeid, Konstantin Dunas, Johannes Hagemann, and Sami Jaghouar. Prime agent: A self-improving RLM harness. arXiv:2608.23552, 2026.

[43] Seth Karten, Joel Zhang, Tersoo Upaa, Ruirong Feng, Wenzhe Li, Chengshuai Shi, Chi Jin, and Kiran Vodrahalli. Continual harness: Online adaptation for self-improving foundation agents. arXiv:2605.09998, 2026.

[44] Alex L. Zhang, Tim Kraska, and Omar Khattab. Recursive language models. arXiv:2512.24601, 2025.

[45] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. arXiv:2210.03629, 2022.

[46] Noah Shinn, Federico Cassano, Edward Berman, Ashwin Gopinath, Karthik Narasimhan, and Shunyu Yao. Reflexion: Language agents with verbal reinforcement learning. arXiv:2303.11366, 2023.

[47] Chris Lu, Cong Lu, Robert Tjarko Lange, Jakob Foerster, Jef Clune, and David Ha. The AI scientist: Towards fully automated open-ended scientific discovery. arXiv:2408.06292, 2024.

[48] Jean-Philippe Bruneton et al. Open quantum problems. https://github.com/jpbruneton/ open\_quantum\_problems. Catalogue behind openquantumproblems.com, accessed October 2026.

[49] Naixu Guo et al. QIQC open problem zoo. https://github.com/Naixu-Guo/ quantum-open-problems. Accessed October 2026.

## A Details of each discovery

For every result in Table 1 we give the definition, the known results, the result, how it was found and how it was verified. “How it was found” identifies the run that produced the object: session and run identifiers, start time (UTC, October 2026), strategy arm, model, wall time, tokens and model cost, all from the kernel’s run records. It then gives the method with its parameters, as recorded in the proposer’s notebook, and the search efort, including what was tried without success. “Verification” lists each check separately: what was checked, by which program and by whom, the acceptance criterion and the outcome, followed by a command that reproduces it from the Zenodo deposit. All proposers are Claude Sonnet 5.5.

## A.1 Quantum capacity of the qubit depolarizing channel

Let $\mathcal { D } _ { p } ( \rho ) = ( 1 - 3 p ) \rho + p ( X \rho X + Y \rho Y + Z \rho Z )$ , with $p$ the per-Pauli error probability. By the Lloyd– Shor–Devetak theorem, $Q ( \mathcal { D } _ { p } ) > 0$ whenever some input $\rho$ on n channel uses has positive coherent information $I _ { c } ( \rho , \mathcal { D } _ { p } ^ { \otimes n } ) = \dot { S ( } \mathcal { D } _ { p } ^ { \otimes n } ( \rho ) ) - S ( ( \mathcal { D } _ { p } ^ { \otimes n } \otimes \mathrm { i d } ) ( \psi _ { \rho } ) )$ , where $\psi _ { \rho }$ purifies $\rho .$ The threshold $p ^ { * } = \operatorname* { s u p } \{ p : Q ( \mathcal { D } _ { p } ) > 0 \}$ is unknown (catalogue entry C11 [48]). Known lower bounds include 0.0637675 [19] and 0.064912 [20]. The best certified point was $p = 1 6 2 3 9 / 2 5 0 0 0 0 = 0 . 0 6 4 9 5 6$ for a 45-copy rank-two input supported on the symmetric subspace, certified in exact rational arithmetic [21].

Result. A 45-qubit rank-two input on the symmetric subspace whose coherent information is certified positive at $p = 3 2 4 8 3 / 5 0 0 0 0 0 = 0 . 0 6 4 9 6 6$ , which raises the certified point by $1 . 0 \times 1 0 ^ { - 5 }$ Its fidelity with the previous record state is 0.99927.

## How it was found.

• Run. Session 20261019-181124, iteration 0, proposer 0. Started 18:11; new neighbourhood arm; 2,522 s; 181,073 input and 3,457 output tokens; \$0.60.

• Start. The record state A, given as the seed (kernel score 0.478). The tools were the operator’s float evaluator, which reduces $I _ { c }$ for symmetric-subspace inputs to Schur–Weyl blocks via a coherent-state generating function and never forms a 2<sup>n</sup>-dimensional matrix, and its autograd twin.

• Method. The proposer wrote its own checkpointing Adam polish on the real and imaginary parts of the two unnormalised Dicke-basis rows (learning rate $5 \times 1 0 ^ { - 5 } )$ . It worked at a fixed p and raised $p$ by $1 0 ^ { - 6 }$ whenever $I _ { c } > 2 \times 1 0 ^ { - 9 }$ , for about 5,600 steps, writing the current state to /work/solution throughout.

• Delivered. A claimed $p = 0 . 0 6 4 9 6 7$ and kernel score 0.4835.

• Also tried. A second iteration (\$0.72) tried a third row of norm 0.03 (rank three) and a Dicke lift to $n = 4 6$ . Neither beat the delivered state.

## Verification.

1. Kernel. The task verifier ran in a fresh sandbox. It accepts a claimed p only if $I _ { c } > 1 0 ^ { - 1 1 }$ in two independent quadrature bases (Fibonacci-sphere points). Outcome: accepted at $p = 0 . 0 6 4 9 6 7$   
2. Float scan (operator). $I _ { c } = + 2 . 6 3 \times 1 0 ^ { - 7 } , + 1 . 2 1 \times 1 0 ^ { - 7 } , + 2 . 5 8 \times 1 0 ^ { - 8 } , + 2 . 0 5 \times 1 0 ^ { - 9 }$ and $- 2 . 1 7 \times 1 0 ^ { - 8 }$ bits at p = 0.064956, 0.064962, 0.064966, 0.064967 and 0.064968. The float threshold is therefore ≈ 0.0649671.

3. Exact certificate. The record authors’ certifier (certified coherent information; rational arithmetic, 224-bit square-root enclosures, 50-digit directed accumulation) was run unchanged on the host, taking 8,407 s and 8,361 s:
<table><tr><td>p (exact)</td><td>certified enclosure of  $I _ { c } ~ ( \mathrm { b i t s } )$  float evaluator</td></tr><tr><td>32481/500000 = 0.064962</td><td> $[ 1 . 1 8 0 7 , 1 . 2 2 5 6 ] \times 1 0 ^ { - 7 }$   $1 . 2 0 6 2 \times 1 0 ^ { - 7 }$ </td></tr><tr><td></td><td></td></tr><tr><td> $3 2 4 8 3 / 5 0 0 0 0 0 = 0 . 0 6 4 9 6 6$ </td><td> $\left[ 2 . 3 0 8 4 , 2 . 7 7 7 5 \right] \times 1 0 ^ { - 8 }$   $2 . 5 7 6 3 \times 1 0 ^ { - 8 }$ </td></tr></table>

Both endpoints are classified certified-positive, and each float value lies inside its enclosure. Since $I _ { c }$ is nonincreasing in $p$ for a fixed input, $Q ( \mathcal { D } _ { p } ) > 0$ for all $p \leq 0 . 0 6 4 9 6 6$

4. Reproduce. The state is payloads/EVO1.c128 (complex128, shape $2 ~ \times ~ 4 6$ , 1,472 bytes, SHA-256 7f11f2f1...67c5ae). Run python certify.py ../payloads/EVO1.c128 0.064966 out.json (about 45 CPU-minutes) [12].

## A.2 Positivity frontier of the qutrit depolarizing channel

Let $\mathcal { N } _ { d , p } ( \rho ) = ( 1 - p ) \rho + p I / d$ . For $d \geq 3$ the catalogue (QIQC zoo [49]) listed only the hashing point, i.e. the zero of $I _ { c }$ for the maximally mixed input, which for $d = 3$ is $p _ { h } = 0 . 2 8 7 0 8 7$

Result. With $\eta ( x ) = - x \log _ { 2 } x$ and $\sigma _ { 2 }$ the maximally mixed state on a two-dimensional subspace of $\mathbb { C } ^ { d }$

$$
\begin{array} { r } { I _ { c } ( \sigma _ { 2 } , \mathcal { N } _ { d , p } ) = 2 \eta \Big ( \frac { 1 - p } { 2 } + \frac { p } { d } \Big ) + ( d - 2 ) \eta \Big ( \frac { p } { d } \Big ) - \eta \Big ( 1 - p + \frac { p } { 2 d } \Big ) - ( 2 d - 1 ) \eta \Big ( \frac { p } { 2 d } \Big ) , } \end{array}
$$

whose zero for $d = 3$ lies in [0.29106723603359969, 0.29106723603359988]. This is above the hashing point by $3 . 9 8 \times 1 0 ^ { - 3 }$ . The same single-letter gain holds for every $3 \leq d \leq 6 4$ that we checked, for example d = 4: 0.3170689 against 0.3089806.

## How it was found.

• Session 1 run. Session 20261020-025047, iteration 0, proposer 0. Started 03:01; aggressive rewrite; 1,934 s; 993,698 input and 19,509 output tokens; \$3.27.

– Start: the seed, a rank-3 repetition input on $n = 7$ (threshold 0.287731), with the operator’s exact brute-force evaluator $( n \leq 7$ , rank $\leq 4 )$ and its autograd twin.

– Method: a rank-4 input on $n = 4$ built from the repetition rows plus a small noisy fourth row. It ran Adam at fixed $p = 0 . 2 8 7 4$ , taking $I _ { c }$ from $- 5 \times 1 0 ^ { - 4 }$ to $+ 9 . 9 \times 1 0 ^ { - 3 }$ , then a warm-started p-ladder.

– Outcome: threshold 0.289205, with two seeds agreeing to 0.2892–0.2893.

• Session 2 run (finding run). Session 20261020-040144, iteration 0, proposer 0. Started 04:01; aggressive rewrite; 1,726 s; 715,424 input and 14,517 output tokens; \$2.36.

– Start: the session-1 state.

– Observation recorded in the notebook: with rank 4 allowed, a smaller n is better.

– Method: Adam p-ladders from random starts gave $n \ : = \ : 3 ; \ : \ : 0 . 2 8 9 9 5$ and $n = 2 \colon 0 . 2 9 1 0 6 .$ L-BFGS with a strong-Wolfe line search and a bisection ladder on p then reached 0.2910672, with bisection to $\approx 4 \times 1 0 ^ { - 8 }$

– Global check: 25 random L-BFGS starts at n = 2 all converge to the same optimum, with $I _ { c } ( 0 . 2 9 1 0 8 ) = - 6 . 5 4 6 6 \times 1 0 ^ { - 5 }$

– Also tried: lifts to $n = 3 , 4$ return to the $n = 2$ value.

• Operator. Computed the reduced states, found the product structure $\sigma _ { 2 } \otimes \sigma _ { 2 }$ and derived the closed form above. Two later proposer runs had observed that the state is $P / 4$ on a subspace, but neither computed the reduced states.

## Verification.

1. Kernel. The exact brute-force verifier, run in a fresh sandbox, accepts if $I _ { c } > 1 0 ^ { - 1 0 }$ at the claimed p. Outcome: value 0.2910672, score 0.81344.

2. Host bisection. Bisection to $1 0 ^ { - 8 }$ gives 0.29106723, with $I _ { c } ( 0 . 2 9 1 0 6 7 2 ) = + 1 . 8 4 8 1 8 1 \times 1 0 ^ { - 7 }$ and $I _ { c } ( 0 . 2 9 1 0 6 8 ) = - 3 . 9 2 \times 1 0 ^ { - 6 }$ bits.

3. High precision. A 30-digit mpmath recomputation at the stored amplitudes gives $1 . 8 4 8 1 8 0 8 7 \times$ $1 0 ^ { - 7 }$ bits. The smallest joint eigenvalue is $2 . 4 \times 1 0 ^ { - 3 }$ , so the entropies are smooth there.

4. Structure. $\| \rho - \rho _ { A } \otimes \rho _ { B } \| = 3 \times 1 0 ^ { - 9 }$ , and both reduced spectra are $\textstyle { \left( { \frac { 1 } { 2 } } , { \frac { 1 } { 2 } } , 0 \right) }$ . The closed form agrees with the evaluator.

5. Interval certificate. mpmath.iv at 50 digits gives the enclosure stated above.

6. Reproduce. python rank2 certificate.py, and python verification/highprec check.py states $\mathbf { / s 2 - i t 0 0 - p 0 0 \_ n 2 r 4 _ { - p 0 } } . 2 9 1 0 6 7 2 . \mathbf { j }$ son 0.2910672 30 (state SHA-256 4c3802af...) [13].

The result is elementary enough that it may be known, and no priority is claimed.

## A.3 Private information of the qubit depolarizing channel

For $\Lambda _ { p } ( \rho ) = ( 1 - p ) \rho + p I / 2$ , the zoo record certified $P ( \Lambda _ { p } ) > 0$ only for $p \leq 0 . 2 5 9 8 2 4$ . The single-use six-state ensemble with noisy preprocessing [22] has positive private information below $p = 0 . 2 8 2 2 3$

Result. Explicit two-member classical ensembles on n qubits with positive private information $I ( X ; B ) - I ( X ; E )$ at $p = 0 . 2 8 3 3 6 , 0 . 2 8 3 7 2 , 0 . 2 8 4 0 5 , 0 . 2 8 4 3 7 , 0 . 2 8 4 6 6$ for $n = 4 , \ldots , 8 .$ . At $n = 8 .$ member 0 is $\begin{array} { r } { \rho _ { 0 } = \sum _ { s } w _ { s } | s \rangle \langle s | } \end{array}$ with $w _ { s } \propto e ^ { f _ { | s | } }$ , where the logits over Hamming weight $0 , \ldots , 8$ are

$$
f = ( 0 . 9 2 9 5 , - 0 . 8 5 1 5 , - 1 . 8 9 7 0 , - 2 . 2 6 4 6 , - 2 . 4 4 6 7 , - 2 . 4 5 5 0 , - 2 . 1 9 6 9 , - 1 . 1 5 1 4 , 0 . 6 3 3 7 ) .
$$

Member 1 is its bit complement, and the two members have equal priors.

## How it was found.

• Run. Session 20261020-201046, iteration 0, proposer 1. Started 20:11; new neighbourhood; 2,635 s; 630,192 input and 14,966 output tokens; \$2.12.

• Start. The session-1 $n = 5$ ensemble (threshold 0.283722), found by Adam on all $2 ^ { n }$ logits.

• Method. The proposer restricted the ensemble to a classical complement pair whose weights are a softmax over $n + 1$ Hamming-weight logits. This reduces the environment entropy to $n + 1$ eigenproblems. It then ran Adam in torch on the exact classical private information, cross-checked against the verifier, with continuation in p and a final bisection.

• Outcome. Thresholds 0.28336, 0.28372, 0.28406, 0.28437 and 0.28467 for $n = 4 , \ldots , 8 .$

• Also tried. Three, four, six and eight classical members at $n = 3 , 4$ (no gain). General complex members at $n = 2 , 3 .$ , started at the classical optimum, do not exceed it, and random starts collapse to zero private information.

## Verification.

1. Kernel. The verifier accepts if the private information is at least $1 0 ^ { - 9 }$ bits at the claimed $p .$ The evaluator self-tests reproduce the hashing rate, brute-force Stinespring results for $n \leq 3 .$ and the 14.11% Renes–Smith rate.

2. Host. Thresholds by bisection in double precision; at $n = 8$ the threshold is 0.2846707, and the private information at $p = 0 . 2 8 4 6 6$ is $2 . 5 \times 1 0 ^ { - 7 }$ bits.

3. High precision. A 30-digit mpmath recomputation was made for every deposited ensemble $( n = 8$ takes about 10 minutes).

4. Reproduce. python verify/private info.py ensembles/n8 shell symmetric.json and python verify/certify mp.py ensembles/n5 shell symmetric.json 30 [15].

The literature already implies $P > 0$ up to $p \approx 0 . 2 9 1 5 ;$ that is the operator’s argument, given in the deposit. These ensembles are new only as explicit finite-n witnesses.

## A.4 Preparing the Golay logical zero

The task is to prepare $| 0 _ { G } \rangle = 2 ^ { - 1 1 / 2 } \sum _ { c \in C ^ { \perp } } | c \rangle$ of the $[ [ 2 3 , 1 , 7 ] ]$ Golay code from $| 0 \rangle ^ { \otimes 2 3 }$ with Cliford gates, no ancillas and free single-qubit gates, minimising the two-qubit gate count N and the twoqubit depth D. The published Pareto points were $( N , D ) = ( 4 5 , 7 )$ and (44, 8) from RL plus MCTS graph decimation [24], and earlier (57, 7), (56, 17) and (45, 10).

Result. A circuit with $N = 4 4$ controlled-Z gates in $D = 6$ layers, which dominates both published points. In the cyclic coordinates of the Golay code, its two-qubit layers are

$L _ { 1 }$ : (4, 10), (17, 15), (20, 16), (1, 8), (13, 21), (18, 6), (12, 11), (5, 3), (0, 14)

$L _ { 2 }$ : (2, 16), (8, 10), (17, 6), (1, 12), (11, 21), (15, 0), (18, 13), (5, 22), (9, 3), (7, 20)

$L _ { 3 }$ : (8, 19), (12, 17), (11, 0), (14, 22), (6, 4), (3, 13), (7, 21)

$L _ { 4 }$ : (12, 5), (19, 16), (0, 2), (22, 8), (4, 21), (3, 20)

$L _ { 5 }$ : (16, 12), (10, 0), (18, 19), (6, 2), (20, 11), (4, 9), (21, 22)

$L _ { 6 }$ : (5, 10), (4, 11), (9, 16), (2, 21), (3, 6).

The interleaved single-qubit Clifords are in the deposit [14].

## How it was found.

• Run. Session 20261020-163836, iteration 1, proposer 1. Started 17:03; aggressive rewrite; 1,491 s; 815,918 input and 27,297 output tokens; \$2.86.

• Start. The operator’s transcription of the published (44, 8) circuit.

• Representation. The circuit is written as 44 controlled-Pauli gates $\mathrm { C P } ( a , b , P , Q )$ in a global Pauli frame, so all single-qubit Clifords are absorbed. A stabilizer tableau stored as 46 32-bit columns, written in C, evaluates about $5 \times 1 0 ^ { 6 }$ full circuits per second per core (the previous Python version ran at 49,000).

• Cost. The number of qubits left mixed when the circuit is applied in reverse to the target. A qubit is pure if and only if its X- and Z-columns are linearly dependent, so cost 0 means a product state in any local basis. The previous session’s cost accepted only Z-basis product states.

• Search. A Metropolis walk at a fixed 44 gates $\left( T \mathrm { ~  ~ { ~ = ~ } ~ } 0 . 3 \right)$ with energy 2 · cost + #{gates in layers beyond the target depth}. Its moves change a gate’s Pauli axes or endpoints, shift it in the order, replace it, or delete one gate and insert another. From the start circuit it reaches depth ≤ 7 in about 1 s and depth ≤ 6 in 10–60 s. A converter turns the result back into H, S and CZ gates with Pauli sign fixes.

• Not found. Depth 5: in four runs of 300 s, the best is one gate over the layer limit. 43 gates:

– an exhaustive delete-two/insert-one scan of 92 million neighbours of the start circuit has minimum cost 2;

– four fixed-length-43 annealing runs of 540 s each, about $1 0 ^ { 1 0 }$ evaluations in total, end at cost 2, i.e. one Bell pair left.

Total compute was about 25 CPU-minutes.

## Verification.

1. Kernel. The verifier (golay core.check circuit) ran in a fresh sandbox and simulates the circuit as a stabilizer tableau. The output group is CSS, with pure-X rank 11 and pure-Z rank 12; all signs are +; and the Z-code has minimum distance 7, which makes it the Golay code by uniqueness. It reported N = 44 and earliest-possible (ASAP) depth $6 ;$ value 44.6.

2. Relabelling (operator). find relabeling.py finds the wire permutation π to the cyclic coordinates. It is the same permutation as for the published start circuit, and the relabelled circuit prepares the cyclic-code state exactly.

3. Statevector. A full $2 ^ { 2 3 }$ -amplitude simulation gives overlap 1.0 with $| 0 _ { G } \rangle$ in cyclic coordinates.

4. Reproduce. python verify/golay core.py circuits/evo 44 depth6 cyclic.json --sv (a few seconds; about 130 MB of memory).

## A.5 Stabilizer rank of $| T \rangle ^ { \otimes 7 }$

For $| T \rangle = ( | 0 \rangle + e ^ { i \pi / 4 } | 1 \rangle ) / \sqrt { 2 }$ , the stabilizer rank $\chi ( \psi )$ is the least number of stabilizer states whose span contains ψ. The exact values $\chi ( | T \rangle ^ { \otimes m } )$ are known for $m \leq 4 \ ( 2 , 2 , 3 , 4 )$ [25]. For $m = 5 , 6$ the bounds were 6 [26], and for $m = 7$ only the product bound 12.

Result. With $\zeta = e ^ { i \pi / 4 } , | T \rangle ^ { \otimes 7 } = 2 ^ { - 7 / 2 } \sum _ { i = 1 } ^ { 1 0 } c _ { j } | \varphi _ { j } \rangle$ for the ten stabilizer states below. They are written in Dehaene–De Moor form $\begin{array} { r } { | \varphi \rangle = \sum _ { y \in \mathbb { F } _ { 2 } ^ { k } } i ^ { e ( y ) } | x _ { 0 } + G y \rangle } \end{array}$ , with $\begin{array} { r } { e ( y ) = \sum _ { i } a _ { i } y _ { i } + 2 \sum _ { ( i , l ) \in B } y _ { i } y _ { l } } \end{array}$ where the columns of G are the vectors $g _ { 1 } , \ldots , g _ { k }$ . Hence $\chi ( | T \rangle ^ { \otimes 7 } ) \leq 1 0$ , and by sub-multiplicativity $\chi ( | T \rangle ^ { \otimes 1 3 } ) \leq 6 0$ (previously 72).

<table><tr><td>j</td><td>k x0</td><td></td><td> $g _ { 1 } , \ldots , g _ { k }$ </td><td>a</td><td>B</td><td> $c _ { j }$ </td></tr><tr><td>1</td><td>1</td><td>0011111</td><td>1111111</td><td>(2)</td><td></td><td> $\frac { - 2 \zeta - 2 \zeta ^ { 2 } } { \zeta }$ </td></tr><tr><td>2</td><td>4</td><td>0000001</td><td>0010001, 0001001, 0000101, 0000011</td><td>(0,0,0,0)</td><td></td><td></td></tr><tr><td>3</td><td>4</td><td>1100000</td><td>0010001, 0001001, 0000101, 0000011</td><td>(0,0,0,0)</td><td></td><td> $- 1 + \zeta - \zeta ^ { 3 }$ </td></tr><tr><td>4</td><td>3</td><td>0000000</td><td>1000000, 0100000, 0011111</td><td>(2,2,1)</td><td></td><td>1-ζ</td></tr><tr><td>5</td><td>3</td><td>0000000</td><td>1000000, 0100000, 0011111</td><td>(0,0,1)</td><td>all pairs</td><td> $1 + \zeta$ </td></tr><tr><td>6</td><td></td><td>7 0000000</td><td>I7</td><td>(0,0,1,1,1,1,1)</td><td></td><td> $\begin{array} { r l } { \frac { 1 } { 4 } ( 1 - \zeta - \zeta ^ { 2 } - \zeta ^ { 3 } ) } & { { } } \end{array}$ </td></tr><tr><td>7</td><td>7</td><td>0000000</td><td>I7</td><td>(0,0,0,0,0,0,0)</td><td>一</td><td> $\frac { 1 } { 2 } ( - \zeta + \zeta ^ { 3 } )$ </td></tr><tr><td>8</td><td>7</td><td>0000000 I7</td><td></td><td>(0,0,1,1,1,1,1) (1, l), l &gt; 1;</td><td> $( 2 , l ) , l > 2$ </td><td> $\textstyle { \frac { 1 } { 4 } } ( - 1 - \zeta + \zeta ^ { 2 } - \zeta ^ { 3 } )$ </td></tr><tr><td>9</td><td></td><td></td><td>6 0000000 1000001, 0100001, 0010001, 0001001,</td><td>(0,0,0,0,0,0)</td><td>(1,2)</td><td> $\begin{array} { r } { \frac { 1 } { 2 } ( - 1 + \zeta + \zeta ^ { 2 } - \zeta ^ { 3 } ) } \end{array}$ </td></tr><tr><td></td><td></td><td>10 5 0000000</td><td>0000101, 0000011 1100001, 0010001, 0001001, 0000101, 0000011</td><td>(0,2,2,2,2)</td><td>all pairs</td><td> $\begin{array} { r } { \frac { 1 } { 2 } ( - 1 + \zeta - \zeta ^ { 2 } + \zeta ^ { 3 } ) } \end{array}$ </td></tr></table>

## How it was found.

• Run. Session 20261021-033358, iteration 0, proposer 0. Started 03:34; structured construction; 1,603 s; 1,216,380 input and 33,582 output tokens; \$4.15. The run stopped at the \$4 per-run cap.

• Dictionary. The qubits are split into blocks $[ a , b ]$ with $a + b = 7$ . Three states (the all-zero state, a block GHZ state and the global GHZ state) are closed under all $S _ { a } \times S _ { b } –$ symmetric Clifords: block-wise H, S and X, CZ within and between blocks, and block-parity CNOT fans. This gives 396–540 vectors, which span the $( a + 1 ) ( b + 1 )$ -dimensional symmetric space for [2, 5], [3, 4] and other partitions, but not for [1, 6] or the fully symmetric case.

• Subset search. Batched simulated annealing over r-subsets of the dictionary, minimising the squared least-squares residual of $| T \rangle ^ { \otimes 7 }$ , with 1,024 chains at about $1 0 ^ { 6 }$ evaluations per minute per CPU. It reached $r = 1 2$ on [3, 4] in 40 $\mathrm { s } , r = 1 1$ on [3, 4] in 220 s and on [2, 5] in 130 s, and then r = 10 on [2, 5]. Solutions were converted to exact normal form and checked with the exact verifier.

• Record note. The header of the run’s notebook still reports $r \ = \ 1 1 ;$ ; it was written before $r = 1 0$ was found. The delivered solution.json contains 10 states.

• Also tried. An exhaustive $\mathbb { Z } _ { \mathrm { 5 ^ { - O r b i t } } }$ test over all 2.4 million five-qubit stabilizer states found no single cyclic orbit of five states containing $| T \rangle ^ { \otimes 5 }$ . Separately, a companion exhaustive small case run stalled at residual 0.0647 for five terms at $m = 5$

## Verification.

1. Kernel. Exact Gaussian elimination over $\mathbb { Q } ( \zeta ) = \mathbb { Q } [ \zeta ] / ( \zeta ^ { 4 } + 1 )$ confirms that $| T \rangle ^ { \otimes 7 }$ lies in the span of the ten states. Value $1 2 / 1 0 = 1 . 2 $

2. Independent check (operator, separate code). The stabilizer group of each state, found by brute force over all 16,384 Pauli operators, has order $2 ^ { 7 }$ . A 40-digit least-squares residual is $1 . 6 \times 1 0 ^ { - 4 1 }$ . The exact coeficients lie in ${ \frac { 1 } { 4 } } \mathbb { Z } [ \zeta ]$ and agree with least squares to $1 0 ^ { - 9 }$

3. Reproduce. python verify/stabrank.py decomposition/T7 rank10.json and python verify/independent check.py decomposition/T7 rank10.json (seconds) [16].

## A.6 Qubit routing on the SABRE suite

The task: insert SWAP gates so that every two-qubit gate of each circuit acts on an edge of the 43-edge IBM Q20 Tokyo design graph, minimising added gates (3× SWAPs), under the rules of [27]. Per-circuit added gates:
<table><tr><td>circuit</td><td>gates</td><td>SABRE 2018</td><td>Qiskit heavy</td><td></td><td>pytket Infinity Machine ratio to SABRE</td><td></td></tr><tr><td>qft_10</td><td>200</td><td>54</td><td>30</td><td>66</td><td>27</td><td>0.53</td></tr><tr><td>qft_13*</td><td>338</td><td>99</td><td>63</td><td>117</td><td>60</td><td>0.62</td></tr><tr><td>qft_16</td><td>512</td><td>186</td><td>120</td><td>225</td><td>99</td><td>0.54</td></tr><tr><td>qft_20*</td><td>800</td><td>318</td><td>216</td><td>351</td><td>180</td><td>0.57</td></tr><tr><td>rd84_142</td><td>343</td><td>105</td><td>87</td><td>90</td><td>48</td><td>0.47</td></tr><tr><td>adr4_197</td><td>3439</td><td>1614</td><td>954</td><td>879</td><td>282</td><td>0.18</td></tr><tr><td>radd_250</td><td>3213</td><td>1275</td><td>975</td><td>780</td><td>294</td><td>0.23</td></tr><tr><td>z4_268</td><td>3073</td><td>1365</td><td>777</td><td>741</td><td>255</td><td>0.19</td></tr><tr><td>sym6_145</td><td>3888</td><td>1272</td><td>969</td><td>714</td><td>312</td><td>0.25</td></tr><tr><td>misex1_241</td><td>4813</td><td>1521</td><td>804</td><td>1311</td><td>228</td><td>0.15</td></tr><tr><td>rd73_252</td><td>5321</td><td>2133</td><td>1692</td><td>1443</td><td>441</td><td>0.21</td></tr><tr><td>cycle10_2_110</td><td>6050</td><td>2622</td><td>2025</td><td>1620</td><td>459</td><td>0.18</td></tr><tr><td>square_root_7</td><td>7630</td><td>2598</td><td>1746</td><td>1911</td><td>516</td><td>0.20</td></tr><tr><td>sqn_258</td><td>10223</td><td>4344</td><td>3093</td><td>2547</td><td>816</td><td>0.19</td></tr><tr><td>rd84_253</td><td>13658</td><td>6147</td><td>5094</td><td>3615</td><td>1212</td><td>0.20</td></tr><tr><td>co14_215</td><td>17936</td><td>8982</td><td>6732</td><td>5331</td><td>1785</td><td>0.20</td></tr><tr><td>sym9_193</td><td>34881</td><td>16653</td><td>14019</td><td>6969</td><td>2535</td><td>0.15</td></tr><tr><td>9symml_195</td><td>34881</td><td>17268</td><td>14019</td><td>6969</td><td>2535</td><td>0.15</td></tr><tr><td>8 small circuits</td><td>一</td><td>3</td><td>3</td><td>6</td><td>3</td><td>1.00</td></tr><tr><td colspan="5">total</td><td>53,41835,685 12,087</td><td>0.18</td></tr></table>

<sup>∗</sup>Regenerated circuits, whose reference is our SABRE reimplementation. The eight small circuits are 4mod5-v1 22, mod5mils 65, alu-v0 27, decod24-v2 43, 4gt13 92 and the three Ising chains; all methods need 0 SWAPs on them except alu-v0 27, where one SWAP is optimal. sym9 193 and 9symml 195 are byte-identical files.

How it was found. Six runs, all started from the previous best:

• Session 1 (20261021-041641).

– it00-p0 (data structure, \$3.74): SABRE rewritten in C++ over the two-qubit dependency graph, with early abort once a trial exceeds the best count (about 3,400 trials/s on adr4 197). Heuristic parameters (W, |E|, decay) are randomised per restart, followed by a hill-climb over initial mappings that swaps two physical qubits and accepts if not worse. Total 36,435. it01-p0 (aggressive rewrite, 1,335 s, \$2.43): a receding-horizon router. At every 100-gate boundary it rolls out K randomised 300-gate SABRE continuations and commits the first 100 gates of the best; K adapts to the remaining time. Total 24,228.

## • Session 2 (20261021-053321).

– it00-p1 (new neighbourhood, 2,302 s, 965,052 input and 22,692 output tokens, \$3.24): beam.cpp, a level-synchronous beam search over SWAP sequences.

∗ Each state is the logical-to-physical mapping plus a per-qubit count of executed twoqubit gates (40 bytes, hash-de-duplicated).

∗ After every SWAP, all executable gates are executed greedily.

∗ States are scored by $\begin{array} { r } { \sum _ { q } \sum _ { j < K } \delta ^ { j } ( \mathrm { d i s t } _ { j } - 1 ) - C } \end{array}$ · #executed, where dist<sub>j</sub> is the distance between the qubits of $q ^ { \prime } \mathrm { s } \ j { \cdot } \mathrm { t h }$ upcoming gate, with K = 6–12, δ = 0.7 and $C \approx 1$

∗ Beams hold up to 30,000 states, and forward–reverse passes refine the initial mapping. Total 13,050.

– it01-p0 (aggressive rewrite, \$3.99): an independent beam variant B. The per-circuit combination of both is 12,261.

• Session 3 (20261021-180153).

– it00-p1 (data structure, 1,669 s, \$1.56): forward–reverse iteration of variant B (for example co14 215 1,926 to 1,785).

– it02-p0 (aggressive rewrite): restarts of the first beam from perturbed initial mappings (qft 16 99, qft 20 180, misex1 241 228).

The final routing takes, per circuit, the best verified run (column “source” in the deposit).

## Verification.

1. Kernel. An exact replay (qmap.verify routed) ran in a fresh sandbox for every run. It checks that every original gate is executed exactly once, after its predecessors on both qubits, on a coupling-graph edge under the current mapping, and that every SWAP acts on an edge. It then counts SWAPs.

2. Host. The same replay on the combined routing gives a total of 12,087.

3. Baselines. Qiskit 2.4.1 and pytket 2.18.5 (versions pinned) ran on identical inputs with SWAPs counted the same way, BRIDGE counted $\mathrm { a s } + 3 .$

4. Reproduce. python qmap.py verify routings/best combined.json, which prints total g add 12087 in about a minute using only the standard library [17].

## A.7 The smallest adder transformer

AdderBoard [29] asks for the smallest transformer, by parameter count, that adds two 10-digit numbers with at least 99% accuracy. Self-attention is required, and parameters are counted by the board’s rules (our kernel counter is stricter: constants that act as weights count).

Result. A 5-parameter, one-layer, two-head model that is correct on 10,010 of 10,010 additions on each of three seeds. Its parameters are $\omega = 2 \pi / 1 0$ (scalar embedding ω · token, tied to the cosine read-out), λ = 60 (attention sharpness), $\kappa = 3 0$ (carry-gate sharpness), $\beta = 1 . 5 \omega$ (carry threshold) and $g = 2$ (head-A gain). The positional encoding is a fixed sinusoid of period 11 (no parameters).

• The input is laid out as $a _ { 0 } . . a _ { 9 } , 0 , b _ { 0 } . . b _ { 9 } , 0$ , followed by the generated digits fed back as tokens.

• Head A, at phase +1, averages the current digit pair.

• Head B, at phase 0, averages the previous digit pair and the previous output digit, which yields $( s - o ) / 3$ , where s is that pair’s digit sum and o the previous output digit. This is $9 / 3$ or $1 0 / 3$ exactly when a carry is due.

• The sigmoid gate turns this into the carry, and the logits are cos $( g A + \omega \cdot \mathrm { c a r r y } - \omega k )$ for digit k.

## How it was found.

• Run. AdderBoard session 1, 19 October, in an earlier scratch project; one proposer; 8 minutes; \$1.05. It had no parent: the proposer built the model from the task rules alone.

• Method (from its final message). It chose the period-11 phase layout so that one head reads the current digit pair and the other the previous pair plus the generated digit. It set the five weights by hand. Its first local run timed out at 120 s because torch used 12 threads per call; it set one thread, after which 10,000 additions take about 30 s.

• Parallel run. A second proposer in the same session found an independent 6-parameter design (ReLU wrap).

## Verification.

1. Local verifier (run by the proposer). 10,000 pairs on each of seeds 2025 and 7: accuracy 1.0000, 5 counted parameters. With randomised weights, accuracy is 0.0.

2. Kernel scorer (stricter counter, hidden seed). 10,000 pairs on hidden seed 7331, with selfattention required and the perturbation check repeated: passed with 5 parameters.

3. Upstream verifier. The board’s verify.py reports QUALIFIED, 10,010 of 10,010, on seeds 2025, 4242 and 7. Submitted as AdderBoard pull request #101.

## A.8 A 20M-parameter language model

The task: an architecture of at most 20,000,000 parameters (embeddings included), retrained by the kernel from code on 1B FineWeb-10B tokens within 1,800 s on one H100, scored by bits per byte on the FineWeb validation shard. The GPT-2 124M checkpoint scores 1.0658, and the seed (a scaled-down modded-nanogpt [30], 19.36M parameters) scores 1.3376.

Result. 1.2716 bits per byte with 19,146,900 parameters, which closes 24% of the gap.

• Architecture: four unique transformer blocks cycled over eight efective layers (two loops) at width 512 with 8 heads, and a factorised tied embedding of rank 128. Per-efective-layer scalars (residual and skip weights) are not shared, so each loop can behave diferently.

• Inherited from the session-1 parent: learnable residual/x<sub>0</sub> mixing, U-net skips, Muon plus Adam, batch 131k tokens.

How it was found.

• Run. Session 20261022-014034, iteration 0, proposer 1. Started 02:08; parameter sharing; 6,270 s including its own H100 training runs; 360,077 input and 7,728 output tokens; \$1.20 of model usage plus GPU time.

• Start. The session-1 best: 6 unique layers at width 448, 19.36M parameters, 1.2873 bits per byte when re-scored.

• Experiments (one run each at the oficial budget).

– A: 12 efective layers over 4 unique blocks, width 512, embedding rank 96, 17.5M parameters. Time-bound: it reached only $0 . 8 5 6 \times 1 0 ^ { 9 }$ tokens, with validation loss 3.9586 nats.

– B: 8 efective layers over 4 unique blocks, width 512, rank 128, 19.15M parameters. It reached $0 . 9 9 5 \times 1 0 ^ { 9 }$ tokens with loss 3.9030, against 3.9537 for the parent.

The proposer shipped B and raised its training-time safety margin from 45 to 80 s, because B finished at 1,770 of 1,800 s.

## Verification.

1. Kernel scorer. The scorer uploads only the code (checkpoints are stripped), retrains from scratch in a fresh H100 sandbox under the token and time limits, counts all parameters, and evaluates bits per byte on the FineWeb validation shard. Outcome: 1.2716, with 19,146,900 parameters (at most 20M).

2. Limits. One training seed; the run was not repeated.

## A.9 Low-energy 8×8 INT8 systolic tile

The design is the LogikBench 8×8 weight-stationary INT8 systolic array (signed 8-bit activations and weights, 32-bit outputs) shipped as asap7/tpu with OpenROAD-flow-scripts [31]. A tile loads eight weight rows (one per w valid cycle), streams activation rows, and returns each output row 15 cycles after its activation row. The host idles 2N − 1 = 15 cycles between tiles. A candidate must keep the ports and this cycle-level protocol, so it computes 64 MACs per cycle at the fixed clock. There is no published record for this tile; the reference is the original RTL.

Result. An RTL rewrite (tpu.sv plus a generated mac.sv) with 128.81 fJ/MAC on the public workload and 130.1 fJ/MAC on the private one, against 578.85 fJ/MAC for the original. At 600 ps the power is 4.34 mW internal, 3.33 mW switching and 0.04 mW leakage (7.71 mW in total, against 34.66 mW). The standard-cell area is 2,507 µm<sup>2</sup> (original 5,559), setup worst slack +5.2 ps, hold worst slack +6.3 ps, and there are 0 DRC violations.

How it was found.

• Runs. Three runs in sequence, each building on the last:

– session 20261026-042359-discover-tpu-energy, it00-p0, data structure, started 04:40; 2,477 s, 387,077 input and 15,459 output tokens, \$1.39;

– session 20261026-052723-discover-tpu-energy, it00-p1, new neighbourhood, started 05:27; 3,367 s, 1,105,333 input and 24,581 output tokens, \$3.68;

– the same session, it02-p0, new neighbourhood, parent it00-p1, started 07:02; 2,181 s, 281,823 input and 12,234 output tokens, \$1.03.

• Tools. The operator’s evaluator (full flow, both simulations, power) and a quick mode of it that runs RTL simulation and synthesis only. Proposers ran it inside their own sandbox, so every number in their notebooks is a full-flow measurement.

• Session 1, it00-p0. It removed the per-PE activation registers and broadcast each activation along its row, delaying row i by 7 + i cycles at the input so no output de-skew is needed. It narrowed the partial sums to 16–19 bits per row (sign-extended at the output) and dropped datapath resets (346.9 fJ). It then added latch+AND clock gates per pipeline depth, enabled from the valid pipe derived from a valid, and a w valid gate for the weights (274.9 fJ). Flops fell from about 4,200 to about 2,300. A parallel run (robust/simple, \$1.81) instead wrote an explicit Baugh–Wooley multiplier with a 3:2 carry-save tree and a Brent–Kung adder, because Yosys maps + and \* to Kogge–Stone structures (285.2 fJ, area 2,935).

• Session 2, it00-p1. It combined the two (253.9 fJ) and tried radix-4 Booth with activations encoded once per row (300.8 fJ, rejected). It then wrote a generator, gen mac.py R, that emits one bit-level Dadda tree per group of R PE rows, with Baugh–Wooley sign constants folded in, the incoming partial sum as an operand, and one Brent–Kung adder per stage. R = 2 gave 210.4 fJ; R = 4 (33 operands, two partial-sum stages, activation delay 13 + ⌊i/4⌋) gave 188.2 fJ, with setup slack only +3.6 ps. The proposer noted that R = 8 would not close 600 ps.

• Session 2, it02-p0. It replaced the activation shift chains (13 or 14 stages of 64 bits, every flop toggling every cycle) by a 13-entry ring bufer. Each entry is written once through its own latch+AND clock gate under a shared one-hot pointer, and read through an AND-OR mux into the valid-gated input register; rows 0–3 read with the pointer rotated by one (delay 13), rows 4–7 without (delay 14). It also replaced the shifting weight register by direct per-row writes, with one clock gate per row and a one-hot pointer that advances on w valid. Result: 128.8 fJ. An ablation in the same run (ring bufer only) gave 144.6 fJ.

• Also tried. A parallel line of runs (conservative, robust/simple, data structure) reached 252.8 fJ by the same combination, 250.2 fJ with a hybrid final adder (ripple-carry for the low 6 bits, Brent–Kung above), and 207.0 fJ with two PE rows fused per stage. Its robust/simple proposer declined to ship a 10-bit-ripple variant at 246.2 fJ because its hold slack was +0.197 ps. Untried, per the final notebook: a 12-entry ring for rows 0–3, gating the pointer clock when idle, latch storage entries, and a ring for the partial-sum stages.

## Verification.

1. RTL equivalence (kernel). Icarus Verilog simulation of the candidate RTL against the Python golden model on a correctness set (10 uniform random tiles of 16 rows, plus corner tiles: all −128, all 127, alternating extremes, zeros) and on the scoring workload (8 tiles of 32 rows). Criterion: every output row bit-exact. Outcome: pass, on both seeds.

2. Gate-level equivalence (kernel). The scoring workload on the routed netlist, with functional (zero-delay) ASAP7 cell models. Criterion: bit-exact. Outcome: pass, on both seeds.

3. Physical gates (kernel). Setup and hold worst slack at the finish stage $\geq 0$ , final detailed-route DRC count $0 , \mathrm { { a r e a } \leq 6 , 1 1 5 . 1 ~ \mu m ^ { 2 } }$ . Outcome: $+ 5 . 2 , + 6 . 3 \ \mathrm { p s } , 0$ and $2 { , } 5 0 7 \ \mu \mathrm { { m } ^ { 2 } }$

4. Power (kernel). OpenSTA read vcd on the gate-level VCD of the 456-cycle window and report power with the routed SPEF; energy per $\mathrm { M A C } = P _ { \mathrm { t o t a l } } \times 4 5 6 \times 6 0 0 ~ \mathrm { p s } / 1 6 , 3 8 4$ . Public seed 2026: 128.81 fJ. Private seed 7331, run by the kernel in its own sandbox: 130.1 fJ.

5. Scope. The design relies on weight loads of exactly eight w valid cycles per tile, as its notebook states. The testbench always loads that way, so both equivalence checks hold only under this protocol. There was no separate host re-run beyond the kernel’s two splits.

## A.10 TESS planet candidate

Result. G 249-11 (TIC 417732194; Gaia DR3 484671464331349632), an M4.5 dwarf at 29.1 pc. The candidate has P = 5.307425 ± 0.000008 d, $T _ { 0 } = 2 4 5 9 9 1 2 . 3 1 5 1 \pm 0 . 0 0 0 8 ~ \mathrm { B J D _ { T D B } }$ , depth $3 . 0 \pm$ 0.4 ppt, $T _ { 1 4 } = 0 . 8 8$ h, b = 0.83 and $R _ { p } = 1 . 6 1 \pm 0 . 0 9 R _ { \oplus }$

## How it was found.

• Run. Session 20261022-221142, iteration 0, proposer 0. Started 22:11 on 22 October; new population arm; 6,294 s; \$7.06 (task-level per-run cap \$8).

• Method. A box least-squares (BLS) search of TESS-SPOC full-frame-image light curves of the 3,100 brightest of 7,474 M dwarfs within 40 pc. It re-found the known planets TOI-1266, TOI-1640 and TOI-1685 at SNR 22–23, and set aside about 1,300 referee-accepted peaks as stellar rotation.

• Claims. Four new transits. Its notebook calls TIC 417732194 the cleanest: 8 events in Sectors 59–60.

• Independent detection. The parallel proposer in the same round (it00-p1, \$3.84) also detected the signal but declined to claim it, citing “short baseline, unpredictable ephemeris”.

## Verification.

1. Referee v5 (kernel). The referee re-measures each claim from the raw light curves, applies cadence checks per sector and the cross-era rule, checks against known catalogues (TOI, CTOI, SPOC TCEs, VSX, Gaia variability, SIMBAD) and requires an event-overlap alias test to pass. It accepted the claim. Two of the other three claims were rejected by the cross-era rule.

2. Pixel level (operator). A Sector-59 TESScut diference image places the flux deficit on or next to the target pixel. The fourth claim was rejected here as a field-wide flux drop.

3. Prediction test (operator). An ephemeris fitted to Sectors 59–60 alone predicted dips in Sectors 73 and 86, which were never used for selection and lie 68 and 134 orbits later. Both dips appear inside the propagated ±2σ window. Against 400 full-pipeline null realisations (transits masked, sectors circularly shifted), $p = 0 . 0 9$ for Sector 73, $p = 0 . 0 0 2 5$ for Sector 86, and about 0.002 combined.

4. Photometric vetting.

• Odd and even depths agree to 1.2σ.

• No secondary eclipse $( - 0 . 1 3 \pm 0 . 3 0 ~ \mathrm { p p t } )$

• The depth is stable across seven detrending choices.

• The residuals are white at the transit timescale.

5. Independent review. A read-only reviewer agent found a 1,000-day error in the reference epoch and an independence error in the first report. Both were corrected.

6. Status. A bound eclipsing binary and a $G = 1 7 . 2$ neighbour $2 0 ^ { \prime \prime }$ away are not excluded. The candidate is unconfirmed [18].

## B The autonomous runs in detail

## B.1 Run 1: two-deletion-correcting codes

Session 20261026-002501-autonomous, launched as a background job at 00:24:58 UTC on 26 October 2026 with 3 iterations, 2 parallel proposers, a \$30 budget and at most 3 scout rounds. There was no operator action after launch.

Goal (verbatim, the default). “Find an open problem in mathematics, quantum information or theoretical computer science whose best known result is an explicit object or number that a program can check exactly, and improve on the best known result.”

Problem chosen. The largest binary code $C \subset \{ 0 , 1 \} ^ { 1 6 }$ that corrects two deletions: for distinct codewords, the sets of length-14 strings obtained by deleting two positions are disjoint. This is a maximum independent set in a fixed graph on 65,536 vertices. The LP upper bound is 487. The scout’s reasons, from SELECTION.md:

• the deliverable is an explicit word list;

• the exact check takes under a second;

• every extra codeword is a strict improvement;

• the literature gave 215 (Weindel and Heckel, arXiv:2504.00613v2, code not published) and 208 (CPro1, arXiv:2505.23881, code public).

## Round 0.

1. Scout (00:25–00:35, \$3.14). It wrote a task with record 215, the public 208-word seed, score range 208–216 and ten calibration cases: four valid (208, 194, 193 and 100 words) and six invalid (a bit flip, a duplicate, an empty code, a near-miss extra word, a non-binary word, a wrong length).

2. Kernel validation. The seed scored 208. All four valid cases reproduced their values and all six invalid cases were rejected, so no problems were found.

3. Audit (00:36–00:39, \$1.32), verdict fail. From AUDIT.md: “the cited EinsteinArena problem 34 has moved on. Its leaderboard (queried 2026-10-26): nightshift-dc16 223, happy-gpt-astra-6 218, Opus-HedgyLabs 216, . . . I downloaded the 223-word code from the public API and checked it with the task’s own verifier (223.0) and an independent max pairwise LCS = 13 test . . . So record 215 is wrong.”

4. Exploit replay. The auditor wrote 18 exploit files, among them every word in the space, Boolean or integer elements, full-width and Arabic-Indic digits, padded words, NaN, wrong lengths, a list instead of a dict, and too many words. The kernel replayed all 18 and the verifier accepted none.

## Round 1.

1. Scout repair (00:40–00:46, \$0.85). Record 223, sourced to the arena’s best-solution API and queried 26 October. The 223-word code became the seed, the score range was widened to 208– 228, and two new calibration cases were added: the 223-word code (valid) and that code plus one conflicting word (invalid).

2. Kernel validation. The seed scored 223. All five valid cases reproduced their values and all seven invalid cases were rejected.

3. Audit (00:47–00:49, \$0.58), verdict pass, with open: true and record ok: true. Its remaining notes: extra dictionary keys are ignored, which is harmless because the value is the number of words; and the record moves fast.

4. Exploit replay. Twenty files, none accepted.

5. Installation. At 00:50:39 the task was installed, with an approval record naming the approver as “kernel (autonomous mode)” and the basis as “calibration + adversarial audit + exploit replay”.

Discovery (00:50–02:05, \$3.46; session 20261026-005039-discover-del2-code-16).

## • Iteration 0.

– aggressive rewrite (\$0.75): plateau local search with forced insertion and Metropolis acceptance, 4 cores × 25 min, about $2 \times 1 0 ^ { 7 }$ moves each. Current size wandered between 206 and 223; best 223.

– data structure (\$0.59): an implicit-adjacency search over 2<sup>14</sup> descendant slots at about $1 . 9 \times 1 0 ^ { 8 }$ candidate probes per second per core; best 223.

## • Iteration 1.

– robust/simple (\$0.84): no gain.

– new neighbourhood (\$0.86): exact large-neighbourhood search. It frees K codewords chosen by Hamming proximity, a random bit mask or a shift-aware distance, then solves the freed region exactly by bitset branch and bound. Runs at K = 24–60 made up to $1 . 6 \times 1 0 ^ { 6 }$ moves; 10–13% reached a diferent optimal set and none reached a larger one.

• Iteration 2. conservative twice (\$0.20 each). Both copied the parent and its notes: no-ops.

Best: the seed, 223 words. Seven bank entries, none above the seed.

Independent check (02:05–02:06, \$0.19, network of). The checker received only problem.md and the best solution. It wrote my checker.py (format, 16-bit binary words, distinctness, pairwisedisjoint deletion sets) and tested it on constructed cases. As a cross-check it computed the longest common subsequence of every pair (at most 13, so no shared length-14 subsequence). Verdict: valid, 223, in agreement with the verifier.

Report (02:06–02:07, \$0.32). Claim: {beats record: false, value: 223, record: 223, claim: “No improvement: the best verified code has 223 words, which ties (does not beat) the public record of 223 for two-deletion-correcting binary codes of length 16.”}. The report’s claim boundary disclaims a new record, the optimality of 223, the impossibility of 224, and authorship of the code, crediting the arena agents.

Totals. 1 h 42 min; \$9.86. The breakdown is scout \$3.99, audit \$1.90, discovery \$3.46, check \$0.19 and report \$0.32.

## B.2 Choosing the goals of runs 2 and 3

Before runs 2 and 3, five research agents, directed by the operator, surveyed published tables of explicit, exactly checkable records for low current activity. They looked beyond quantum information. Their top picks were Brouwer’s mixed binary/ternary code tables (quiet since 2013), qutrit stabilizer-code gaps, sorting-network sizes for 25–40 inputs, real Grassmannian packings and ternary codes $A _ { 3 } ( n , d )$ . They advised avoiding tables already worked by many agents: the V3R1T4S list, binary and constant-weight codes, most Einstein Arena problems and the QIQC zoo. The operator wrote one goal sentence for each of the first and third picks, naming the table and the improvement wanted, and asked the scout to choose the cell and confirm that it was current. Both jobs ran with 3 iterations, 2 parallel proposers, a \$30 budget and the research math arms. The first launch of the pair collided on a session directory (Section 3.8) and was cancelled within a minute; the relaunches below are the runs reported.

## B.3 Run 2: sorting network on 31 inputs

Session 20261026-024005-autonomous-e33eb, 02:40–05:18 UTC.

Goal (verbatim). “Improve a size upper bound in Bert Dobbelaere’s sorting-network tables for n = 25 to 40 inputs (. . . ): choose n, confirm the current best-known number of comparators and its date, and construct a valid sorting network on n inputs with fewer comparators, verified exactly by the 0-1 principle.”

Scout (02:40–02:58, \$2.03). It chose $n \ = \ 3 1$ : record 180 comparators in 14 layers, bounds $1 4 1 \leq S ( 3 1 ) \leq 1 8 0 .$ , table source “Batcher odd-even merge”, with no 31-input size change since the 2017 draft (page dated 10 October 2026). Its reason was that the entry is the 32-input merge network minus one wire, never improved by search, while the entries for $n = 2 8 – 3 0$ had been improved beyond the wire-removal bound. Before choosing, it ran its own wire-removal beam search from all record networks for $n = 2 5 { - } 4 0$ ; this beat no entry (for $n = 3 0$ it gave 173 against the record 172). The verifier is an exact check of all $2 ^ { 3 1 }$ binary inputs by bit-slicing (about 8–15 s). The seed is the record network. There were eight calibration cases. The three valid ones were the 186-comparator Batcher network, the record and the record plus a redundant comparator (181). The five invalid ones were an index out of range, a malformed list, the record with one comparator dropped, a reversed comparator and a 30-input network. Scoring was minimised over the range 186 to 175, which in the kernel’s convention is reversed (Section 3.8).

Validation and audit (02:58–03:05, \$0.89). All eight cases reproduced. Audit verdict: pass. Its minor issues were that the page now read 13 October, with changes only for other n; that n = 31.0 is accepted as 31; and that public equals private. The auditor wrote 23 exploit files, among them the 30-input record plus an unused wire, and the kernel’s replay rejected all of them. The auditor described the score mapping as “186→0, 175→1”, the intended direction, which is not what the kernel computes.

Discovery (03:06–05:15, \$4.12; session 20261026-030658-discover-sortnet-31).

• Iteration 0. Literature transfer twice (\$0.64, \$1.04). Both wrote SorterHunter-style evolutionary searches in C from the record, with the first three or four layers fixed and only the reachable 0-1 vectors kept. Their mutations replaced, swapped, moved, inserted or deleted comparators and pruned redundant ones. Neither found 179.

• Iteration 1. Scale compute (\$0.97) added a pretest on 1,024 witness vectors that earlier mutants had failed, raising throughput from about 52k to 300k evaluations per second per core, and ran on 4 cores for 2,900 s each: no 179. Exhaustive small case (\$0.27) ran no search. It estimated a delete-two/insert-one repair at about $1 0 ^ { 9 }$ candidates, each over about 28,000 vectors, as beyond its budget.

• Iteration 2. New objective (\$0.27) checked that none of the 180 single-comparator deletions sorts, then ran no new search. New objective (\$0.93) fixed the first 60 comparators (four butterfly layers, leaving 28,024 unsorted reachable vectors) and annealed the remaining 119 comparators on the number of vectors still unsorted. Its best state left one vector unsorted.

Best: the seed, 180. Seven bank entries, all scored 0.4545.

Independent check and report (05:16–05:18, \$0.55). The checker’s own C program confirmed that the network sorts all 2<sup>31</sup> inputs with 180 comparators. Claim: {beats record: false, value: 180, record: 180}. The report noted the audit’s 0.545 against the bank’s 0.4545 and used the bank’s figure. Its claim boundary disclaims an improvement and any statement that 179 is impossible, and notes that every search started from the record with a fixed prefix.

Totals. 2 h 38 min; \$7.59: scout \$2.03, audit \$0.89, discovery \$4.12, check \$0.27, report \$0.28.

## B.4 Run 3: mixed binary/ternary code N(4, 5, 3)

Session 20261026-023958-autonomous-67f31, 02:40–05:35 UTC.

Goal (verbatim). “Improve a lower bound in Andries Brouwer’s table of mixed binary/ternary codes N(m,n,d) (. . . ): codes with m binary and n ternary coordinates and minimum mixed Hamming distance d. Choose a cell where the gap between the lower and upper bound is large and the lower bound looks improvable, confirm the table entry is current (check the page’s update notes and recent arXiv), and construct a strictly larger code than the current lower bound.”

Scout (02:40–03:27, \$4.66). It chose N(4, 5, 3): words in $\{ 0 , 1 \} ^ { 4 } \times \{ 0 , 1 , 2 \} ^ { 5 }$ (3,888 in all) at mixed Hamming distance at least 3, with bounds 186–216. The lower bound dates from Brouwer, H¨am¨al¨ainen, Osterg˚ard and Sloane (1998), and the upper bound 216 is 3 <sup>¨</sup> N(4, 4, 3). Its reasons were the 16% gap and the fact that neighbouring cells, among them N(3, 5, 3), N(5, 5, 3), N(3, 6, 3) and N(4, 7, 3), were improved by forum posters in 2010–2013 while this one was not. It fetched the page (Last-Modified 17 October 2026); its arXiv search was blocked by HTTP 429. It wrote coset.c, a clique search over unions of cosets of an additive subgroup. This reproduced 186 as 62 cosets of ⟨(0000|10101)⟩ in seconds, along with other table entries (N(7, 3, 3) = 192, N(8, 3, 3) = 384, $N ( 5 , 6 , 5 ) = 9 6 , N ( 6 , 5 , 5 ) = 6 6 )$ . Its own searches over subgroup orders 2, 3, 4, 6 and 9 and a 16k-step LNS did not beat 186. There were twelve calibration cases. Four were valid: the record, a 150-word subcode, the same code in string format and a single word. Eight were invalid: a binary symbol 2, a duplicate, an empty code, float symbols, a near miss at distance 1, the record plus a conflicting 187th word, the wrong cell and the wrong length. Scoring was maximised over 150–188, so the record scores 0.947 (correct direction).

Validation and audit (03:27–03:32, \$1.14). All twelve cases reproduced. Audit verdict: pass. Its minor issues were that extra keys are ignored and that its own arXiv and Semantic Scholar searches were also rate-limited, so only the table page was checked. The kernel replayed 24 exploit files, among them all 3,888 words, and rejected all of them. Installed at 03:34.

Discovery (03:34–05:33, \$3.60; session 20261026-033400-discover-mixed-code-n453).

• Iteration 0. Literature transfer twice (\$0.68, \$0.66). Both ran prescribed-automorphism-group searches, after Osterg˚ard’s method. Each sampled random groups of binary permutations with <sup>¨</sup> flips and ternary permutations with per-coordinate symbol permutations, kept compatible orbits, and annealed a weighted independent set on the orbit conflict graph. Both found 186 and nothing more.

• Iteration 1. Structured construction (\$0.34): about 10,900 groups with up to three generators, 1,500 s on 4 CPUs; best 186 in every run. New objective (\$0.70): tabu.c fixes the code size and minimises the number of conflicting pairs. From random starts it reached valid codes of 180 and 183 words, and once 186 with no imposed structure, but never 187.

• Iteration 2. New objective (\$0.88): a fixed-size tabu search over orbits of the known translation group. It reaches 62 orbits (186 words) in under a second; at 63 orbits (189 words) the best state had one conflicting pair, over four 150 s runs and further seeds. Scale compute (\$0.35): 9,653 random groups with long anneals, 2,700 s on 4 CPUs. 272 groups reached 186 and none reached 187.

Best: the seed, 186. Seven bank entries, all scored 0.947.

Independent check and report (05:33–05:35, \$0.58). The checker confirmed 186 valid words and stated that 186 is not a new record. Claim: {beats record: false, value: 186, record: 186}. The report notes the rate-limited literature checks, says that a newer paper cannot be ruled out, and says that 186 is not proven optimal.

Totals. 2 h 55 min; \$9.97: scout \$4.66, audit \$1.14, discovery \$3.60, check \$0.17, report \$0.41.

## C Discovery sessions

Table 6 lists every discovery session in the durable project, in chronological order. “Runs” counts finished proposer runs, and “\$” is the recorded model spend. A dash means the session ended before the kernel wrote its totals. AdderBoard session 1 (\$3.26) and the TSP calibration run (\$8.12) ran in earlier scratch projects and are not listed.

<table><tr><td>Start (UTC)</td><td>Task</td><td>Iters</td><td>Par.</td><td>Runs</td><td>$</td><td>Outcome</td></tr><tr><td>10-19 01:53</td><td>ml/adderboard (s2)</td><td>2</td><td>2</td><td>4</td><td>0.89</td><td>“1 parameter&quot; via constants (gaming)</td></tr><tr><td>10-19 03:49</td><td>math/sidon-45</td><td>2</td><td>2</td><td>4</td><td>2.80</td><td>13 ties at 4/7</td></tr><tr><td>10-19 04:59</td><td>math/hadamard-51</td><td>2</td><td>2</td><td>4</td><td>1.03</td><td>record is a fixed point</td></tr><tr><td>10-19 05:22</td><td>math/hadamard-51</td><td>2</td><td>2</td><td>4</td><td>3.05</td><td>Z2 family 43.2310</td></tr><tr><td>10-19 17:33</td><td>math/depol-threshold</td><td>2</td><td>2</td><td>0</td><td></td><td>killed (host memory)</td></tr><tr><td>10-19 18:11</td><td>math/depol-threshold</td><td>2</td><td>1</td><td>2</td><td>1.31</td><td>certified p = 0.064966</td></tr><tr><td>10-20 00:59</td><td>math/k7d2-code</td><td>2</td><td>2</td><td>4</td><td>1.33</td><td>no  $K \geq 2 5$  (partial-credit gaming)</td></tr><tr><td>10-20 01:09</td><td>math/k7d2-code</td><td>3</td><td>2</td><td>6</td><td>2.35</td><td>no  $K \geq 2 5$ </td></tr><tr><td>10-20 02:50</td><td>math/qutrit-depol</td><td>3</td><td>2</td><td>2</td><td>4.55</td><td>0.289205</td></tr><tr><td>10-20 04:01</td><td>math/qutrit-depol</td><td>3</td><td>2</td><td>4</td><td>7.93</td><td>0.2910672</td></tr><tr><td>10-20 15:36</td><td>math/golay-prep</td><td>3</td><td>2</td><td>4</td><td>5.84</td><td>seed (44, 8) only</td></tr><tr><td>10-20 16:38</td><td>math/golay-prep</td><td>3</td><td>2</td><td>5</td><td>13.88</td><td>(44, 6)</td></tr><tr><td>10-20 18:55</td><td>math/depol-private</td><td>3</td><td>2</td><td>2</td><td>7.82</td><td>n = 5, 0.28372</td></tr><tr><td>10-20 20:10</td><td>math/depol-private</td><td>3</td><td>2</td><td>5</td><td>6.24</td><td>n = 8, 0.284668</td></tr><tr><td>10-21 01:36</td><td>math/mub6-k4</td><td>3</td><td>2</td><td>6</td><td>2.10</td><td>record reproduced only</td></tr><tr><td>10-21 02:40</td><td>math/stabrank-t</td><td>3</td><td>2</td><td>6</td><td>2.33</td><td>null (default arms)</td></tr><tr><td>10-21 03:33</td><td>math/stabrank-t</td><td>3</td><td>2</td><td>3</td><td>7.29</td><td> $\chi ( | T \rangle ^ { \otimes 7 } ) \leq 1 0$  (math arms)</td></tr><tr><td>10-21 04:16</td><td>opt/sabre-tokyo</td><td>3</td><td>2</td><td>3</td><td>8.11</td><td>24,228 added gates</td></tr><tr><td>10-21 05:33</td><td>opt/sabre-tokyo-v2</td><td>3</td><td>2</td><td>3</td><td>8.27</td><td>beam search; 12,261 combined</td></tr><tr><td>10-21 06:52</td><td>opt/sabre-tokyo-v2</td><td>3</td><td>2</td><td>1</td><td></td><td>shell died; one run lost</td></tr><tr><td>10-21 18:01</td><td>opt/sabre-tokyo-v2</td><td>3</td><td>2</td><td>6</td><td>4.83</td><td>12,087 combined</td></tr><tr><td>10-21 21:21</td><td>ml/tinylm</td><td>3</td><td>2</td><td>6</td><td>5.32</td><td>1.2868 bpb (default arms)</td></tr><tr><td>10-22 01:40</td><td>ml/tinylm</td><td>3</td><td>2</td><td>6</td><td>4.15</td><td>1.2716 bpb; Modal limit hit</td></tr><tr><td>10-22 17:48</td><td>astro/tess-discovery</td><td>3</td><td>3</td><td>3</td><td>11.24</td><td>referee v1 gamed</td></tr><tr><td>10-22 18:54</td><td>astro/tess-discovery</td><td>3</td><td>3</td><td>3</td><td></td><td>cancelled after round 0 (gaming)</td></tr><tr><td>10-22 20:44</td><td>astro/tess-discovery</td><td>2</td><td>3</td><td>4</td><td>18.66</td><td>nothing over cleaned seed</td></tr><tr><td>10-22 22:11</td><td>astro/tess-discovery</td><td>2</td><td>3</td><td>4</td><td>13.31</td><td>G 249-11 candidate</td></tr><tr><td>10-26 00:50</td><td>auto/del2-code-16†</td><td>3</td><td>2</td><td>6</td><td>3.46</td><td>223 (tie with public record)</td></tr><tr><td>10-26 03:06</td><td>auto/sortnet-31†</td><td>3</td><td>2</td><td>6</td><td>4.12</td><td>180 (tie)</td></tr><tr><td>10-26 03:34</td><td>auto/mixed-code-n453†</td><td>3</td><td>2</td><td>6</td><td>3.60</td><td>186 (tie)</td></tr><tr><td>10-26 03:52</td><td>chip/tpu-energy</td><td>3</td><td>2</td><td>0</td><td></td><td>cancelled: score reversed</td></tr><tr><td>10-26 04:23</td><td>chip/tpu-energy</td><td>3</td><td>2</td><td>3</td><td>3.42</td><td>274.9 fJ; cancelled: saturated</td></tr><tr><td>10-26 05:27</td><td>chip/tpu-energy</td><td>3</td><td>2</td><td>6</td><td>7.59</td><td>128.8 fJ/MAC</td></tr></table>

Table 6: All discovery sessions (dates in October 2026). <sup>†</sup>The discovery phase of an autonomous run, whose task was chosen and written by the system; the scout, audit, check and report phases cost a further \$6.40, \$3.47 and \$6.37 (Section 3.8). Two autonomous jobs that collided on one session directory were cancelled before their scouts finished and are not listed.

## D Strategy arms

Each proposer’s brief contains the name and text of its arm, chosen per task family by UCB1.

Default (written for optimisation tasks). conservative: keep the parent’s algorithm; make one safe, measured improvement. aggressive rewrite: replace the core algorithm with a stronger one you can implement and test within the budget. new neighbourhood: keep the framework but add a new move type or neighbourhood and integrate it into the search loop. data structure: make the inner loop faster so more iterations fit in the time limit; measure iterations per second before and after. robust/simple: simplify and harden; make every instance size produce valid output.

Math (written by the operator after the first stabilizer-rank session). structured construction: build candidates from structure (symmetry, known constructions, algebraic families, products of smaller solutions) instead of unstructured local search; derive, then verify exactly. exhaustive small case: reduce by symmetry and enumerate a small case completely, turning the search into a decision with a replayable certificate. new objective: change what is optimised or how candidates are parametrised; calibrate on a solved case first. scale compute: take the best existing method and scale it with parallel restarts and checkpointing. literature transfer: port one published technique named in the notes, reproduce its known result, then push it to the open case.

ML (written by the operator after the first LM session). new architecture family (mixture of experts, state-space or linear attention, recurrent, hybrids), parameter sharing (looped or weight-shared blocks), embedding budget (smaller vocabulary, factorised or hashed embeddings), connectivity (skip connections, value residuals, multi-token heads, attention variants), recipe (optimizer, batch, schedule); each compared at equal tokens.

Astronomy (catalogue mode). new population, new phenomenon, deeper search, vetting and follow-up, anomaly hunt.

## E Evidence for the specification-gaming incidents

AdderBoard: weights turned into constants. Excerpt from the model the proposers delivered (session 2). The two former weights in the excerpt are commented as fixed temperatures, one explicitly as “not a weight”:

```python
self.omega = nn.Parameter(torch.tensor(2 * math.pi / 10)) # embedding / read-out scale
self.lam = 60.0 # attention sharpness (fixed temperature, not a weight)
self.kappa = 30.0 # carry-gate sharpness (fixed temperature)
```

```python
carry = torch.sigmoid(self.kappa * (Bv - 1.5 * self.omega))
U = 2.0 * A + self.omega * carry
```

From the proposer’s final message: “I cut the model from 5 parameters to 1 and it still adds correctly . . . Review risk: the fixed constants (60, 30, 1.5, 2) sit in forward(). A human reviewer could read that as weights hidden in the architecture.” Under the strict counter, these constants count and the model has 5 parameters.

Seven-qubit code: lenient partial credit. The first verifier scored a subspace of dimension K with worst compression residual r as K if r < 10<sup>−9</sup>, and otherwise as $( K - 1 ) { + } 0 . 9 9 9 { \cdot } \mathrm { c l i p } ( - \log _ { 1 0 } r / 9 )$ A K = 25 subspace at r = 0.072 (no code) therefore scored 0.8127, above the confirmed K = 24 code’s 0.80. A proposer drawn with the robust/simple arm delivered exactly that. The fixed formula starts partial credit only below r = 10<sup>−2</sup>.

TESS, referee v1. Accepted claims included “single transits” with depths of 500–1200% (flux glitches), flux steps, and about 90 low-efort eclipsing-binary claims, a quarter of which failed the held-out prediction. Re-scored under referee v2, the same catalogue fell from 94.5 to 30.5 (public) and from 77.5 to 21.75 (hidden).

TESS, referee v2. From a proposer’s notebook (session 2): “Hypothesis: the parent catalogue saturated the capped classes (83 EBs > cap 15 credits, 54 variables > cap 5), so extra credit lies in uncapped single events (1.0) and long-period duo transits (3.0), which SPOC TPS cannot find.” It then submitted 240 single events and 39 two-event “transits” whose period was set to the gap between the events. The operator cancelled the session after that round.

TESS, referee v4 (operator-found). The referee computed one cadence per star, so for a star with both 200-s and 30-min sectors, every 30-min event was invisible to its checks, and a dip seen in only one observing era could pass. Two of four new transit claims were rejected under the fixed referee (v5) because their dip appeared only in the latest era.

## F Gate configuration and the self-derived playbook

Gate. Held-in tasks: an RFC 4180 CSV parser, 2-D TSP and graph colouring. Held-out tasks: interval merging and bin packing. Noise band 0.03; minimum gain 0; anomaly threshold 0.3; cost factor 1.5; at most 12 changed files; one seed per task. Parent selection: score-child (a UCB selector with descendant-growth pruning is implemented but not the default).

Generation 1 (consolidation, rejected). The four bullets the system derived from its first 16 problem runs, verbatim apart from shortened run identifiers and two file names replaced by bracketed descriptions:

P-0002 Before searching, list neighbourhoods and families already tried in prior notes (1–3 flip SA, ILS from record) and pick an untried one; several runs re-ran the same local search. (3 Hadamard runs; confidence 0.60)

P-0003 Faster inner loops or better optimisers on a proxy energy (Gram target, design cost) are not progress unless the proxy is shown to track the true objective; compare on the real score at equal wall-clock. (2 Hadamard runs, 1 Sidon run; 0.60)

P-0004 Symmetry-restricted search spaces $( \mathbb { Z } _ { 1 7 } , \mathbb { Z } _ { 5 } \times \mathbb { Z } _ { 5 } , \mathbb { Z } _ { 3 }$ blocks) capped below the record’s family; run a short pilot (about 60 s) and compare to the record before committing a long compute budget. (2 Hadamard runs; 0.55)

P-0005 [the task’s local scoring script] is often absent; look for [the task’s checker tool] first and use it for the exact score. (3 runs; 0.70)

Graded result: held-in 0.944 against 0.927, held-out 1.000 against 1.000, cost \$5.50 against \$2.72 (ratio 2.02 > 1.5): rejected.