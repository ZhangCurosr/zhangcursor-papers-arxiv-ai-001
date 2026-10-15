# Instruction-Conditioned Motion Control for Robotic Arms: Language-to-Code Decomposition and Reinforcement Learning

Fan Huang<sup>1,∗</sup>, Dezhong Tong<sup>2</sup>, Changyu Chen<sup>3</sup>, Haewoon Kwak<sup>1</sup>, and Jisun An<sup>1</sup>

<sup>1</sup>Indiana University Bloomington <sup>2</sup>University of Michigan <sup>3</sup>Singapore Management University

Abstract— Instruction-Conditioned Motion Control (ICMC) separates parameter specification, instruction generation and code synthesis for language-driven robot control, making execution outcomes traceable to instructions and programs. We compare one-shot generation with ICMC under o1 and o4-mini on cube pick-and-place and letter-reshaping tasks for a UR5e arm in MuJoCo. We then train an open-weight instruction policy by GRPO through a frozen code generator on cube placement onto two- and five-cube letter layouts, averaging the rewards of five programs per instruction. The logged positive-reward rate rose from 31.6% to 35.3% (253/800 to 282/800) between the first and last five training steps. A log audit identifies positive binary labels associated with geometric-failure messages: at least 19 in that run, 18 in the dense-reward run and 5 in the run with one program per instruction. In one case, all five rewards were 1.0 while the log reports a 0.9515 m miss against a 0.05 m tolerance; the verdict came from the program’s own output stream. We distinguish logged reward rates from independently verified task success and report descriptive results for each configuration’s single training trajectory. We introduce an independent evaluation protocol that checks task completion against fixed specifications, separating robot control from success assessment to support reproducible evaluation.

## I. INTRODUCTION

Language models can generate programs that invoke robot primitives, so that a manipulation request can be expressed through a familiar code interface [1]–[3]. Prior work grounds actions in robot capabilities and current state [2], [4] and uses feedback to recover from failures [3], [5], while rearrangement evaluation compares the achieved scene with a specified goal [6]. These considerations go beyond syntactic correctness and motivate inspecting how a task becomes an executable program.

Two issues motivate this study. First, our one-shot baseline produces a complete program without a separately generated instruction artifact. Prior work combines code with executiongrounded reasoning [7]; we investigate a trainable instruction stage for tracing downstream failures. Second, one stochastic code sample can confound instruction quality with sampling variation, motivating our reward-averaging analysis. Separately, imperfect reward signals can favor unintended behavior or diverge from intended performance [8], [9]. Instruction-Conditioned Motion Control (ICMC) separates parameter specification, instruction generation and code synthesis over a fixed robot API for a UR5e arm in MuJoCo [10]. The intermediate action is the entire generated instruction text, which in practice contains both prose and a referencecode block (median code-line fraction 79.3% across the 32 instructions of the final step of the main binary-reward training run), so it is a language-to-code interface rather than a short symbolic plan.

We ask three questions. RQ1: how do one-shot code generation and ICMC compare under o1 and o4-mini on cube pick-and-place and letter reshaping, and how do samebackbone Chain-of-Code and iterative-correction implementations compare on reshaping? RQ2: what happens when an open-weight instruction policy is trained by GRPO [11] through a frozen code generator on cube placement onto lettershaped layouts, with rewards parsed from program execution? RQ3: what do the archived comparisons of reward sampling (M=5 versus M=1), learning rate and update epochs establish, and which effects remain confounded?

This paper makes four contributions. First, a language-tocode decomposition over a fixed UR5e API that trains an inspectable instruction policy through a frozen code generator; retained records link instructions to syntax errors, undefined names and discrepancies between reward labels and logged geometry. Second, an audit of reward consistency: the main run’s logged positive-reward rate rose from 31.6% to 35.3% between the first and last five steps, while inspection identifies at least 19 positive-reward logs reporting geometric failure in that run, 18 in the dense-reward run and 5 with one program per instruction (M=1). The audit traces success labels to the generated program’s output stream. Third, a characterization of downstream reward variance: an estimator argument for averaging M program rewards per instruction, supported by the observation that 302 of the 640 instructions in the main run received mixed rewards across their five programs. Fourth, an independent evaluation protocol for checking task completion against fixed specifications, with simulator state and scoring controlled by a separate software supervisor. Its CPU implementation passed ten negative controls; a hand authored reference controller completed all 40 predeclared scenes under this protocol.

The study characterizes simulated execution through one archived training trajectory per configuration. Evaluation of model-generated programs under the independent protocol, replication across training seeds, and matched tests of learning-rate and epoch effects remain future work.

## II. RELATED WORK AND SCOPE

Code-as-Policies [1] uses hierarchical code generation, and ProgPrompt [2] generates program-like task plans with execution-time assertions; Chain-of-Code [7] interleaves reasoning with execution, and ChatGPT for Robotics [3] studies robot-specific prompting and feedback. ICMC inserts an instruction stage between the task and the program and trains that stage with execution reward while the coder stays fixed, using group-based policy optimization [11] with LoRA [12]. Vision-language-action policies and embodied language models, including RT-1, RT-2, PaLM-E and OpenVLA [13]–[16], and affordance-grounded or closed-loop language systems [4], [5], use observation and action interfaces distinct from our text-only instruction policy; we discuss them as related approaches.

![](images/f92ad137a4cf60f0a5a1d5564d4325b33d586fe707a4dc3b24d24de3a7baf56f.jpg)  
Fig. 1. Instruction-conditioned motion control. Top: archived I (2 cubes) and M (5 cubes) targets in world XY rotated 180<sup>◦</sup>, each in a 0.2 m square; circle radius 0.05 m. Dotted connections are visual guides. The stages show template and step 20, sample 001 excerpts. (a) Five programs yield logged labels 0, 1, 1, 0, 0 and mean 0.4; only the instruction policy is updated. (b) The appended checker shares the generated program’s process and stdout. Archived training used the shared-process checker; independent task verification is specified for future model evaluations. TikZ drafted with Anthropic Claude and revised with OpenAI Codex; initial prompt excerpt: “Build a stronger Figure 1 as a standalone TikZ document.”

Transporter Networks [17] learn pick/place poses from RGB-D demonstrations in Ravens. CLIPort [18] adds language-conditioned semantic features to spatial imitation learning, using task-dependent partial-credit scores. Both use manipulation-policy interfaces distinct from Python-program generation. Rearrangement frameworks [6] distinguish goal specification, sensing, actuation and evaluation. Our text-only RL inputs, generated-code action interface, letter tasks and intended all-object criterion differ on each of those axes.

## III. METHOD

## A. Instruction-conditioned control

Let s = (τ, o) contain a task description and available scene information. An instruction policy generates $l \sim \pi _ { \theta } ( \cdot \mid s )$ ; a frozen code policy generates $c \sim \pi _ { \mathrm { c o d e } } ( \cdot \mid s , l )$ . A program invokes a fixed robot interface inside the simulator. The conceptual objective is

$$
J ( \theta ) = \mathbb { E } _ { s , l , c } [ R ( c , s ) ] ,\tag{1}
$$

where a task-specific evaluator determines R. In the archived runs R was parsed from program output; Section IV-B explains how these labels are interpreted.

Stage 1 specifies task parameters, robot configuration, object identifiers, target coordinates and operational context. Stage 2 generates instructions for perception, fetching and placing, including state transitions. Stage 3 receives those instructions with the API and reference code and generates a complete program. The whole stage-2 token sequence is the trainable action; stage 3 stays frozen. Exposing the intermediate artifact helps identify whether a downstream program ignores an instruction, changes a function contract, or terminates before reaching the requested scene. Fig. 1 summarizes the three stages.

The simulator interface includes cube-pose lookup, gripper opening/closing and trajectory execution. A trajectory must follow the implementation’s (N, 8) action convention. Reference state variables include steps, middle\_step, cube\_id and break\_flag. These implementation details constrain the form of generated programs.

## B. Reward sampling and optimization

For a fixed task s and binary evaluator, let q(l) be the probability that a fresh downstream sample given instruction l receives label 1. With conditionally independent samples from that same distribution,

$$
\widehat { q } _ { M } ( l ) = \frac { 1 } { M } \sum _ { m = 1 } ^ { M } R ( c _ { m } , s ) ,\tag{2}
$$

$$
\mathbb { E } [ \widehat { q } _ { M } \mid l ] = q ( l ) , \qquad \operatorname { V a r } ( \widehat { q } _ { M } \mid l ) = \frac { q ( l ) ( 1 - q ( l ) ) } { M } .\tag{3}
$$

This estimates the chosen evaluator’s label probability, which represents task success only if the evaluator is valid. The variance reduction is conditional on the instruction and costs M times the nominal downstream sampling budget. The archive shows that the variance it targets is present: of the 640 instructions sampled in the main binary run, 269 received five zero rewards, 69 five positive rewards and 302 mixed rewards, so for 302 instructions (47.2%) a single program would have assigned a reward different from the five-program mean. These counts characterize logged-label variation; attributing training-curve changes to specific noise sources requires controlled evaluation.

For K instruction samples, a centered reward coefficient is $\begin{array} { r } { A _ { k } = \widehat { r } _ { k } - K ^ { - 1 } \sum _ { j } \widehat { r } _ { j } } \end{array}$ . The score-function component has direction

$$
g ( \theta ) = \mathbb { E } \left[ \sum _ { k } A _ { k } \nabla _ { \theta } \log \pi _ { \theta } ( l _ { k } \mid s ) \right] .\tag{4}
$$

This is the update direction before normalization, clipping and regularization; a group whose K rewards are equal contributes no centered signal. The archive does not identify the source version governing clipping, the KL coefficient or advantage normalization, so we report the update at this level.

The archived runs combine M=5 averaging with learning rate $1 0 ^ { - 6 }$ and E=3 update passes on each sampled batch. The learning rate and the number of passes are empirical choices. Binary and dense variants were trained separately. The dense reward is reported as

$$
r = R - 0 . 3 V _ { \mathrm { b u g } } + 0 . 2 5 r _ { \mathrm { g o a l } } + 0 . 2 5 r _ { \mathrm { c o m p } } ,\tag{5}
$$

where $V _ { \mathrm { b u g } }$ averages signature, scope and literal-index heuristics, $r _ { \mathrm { g o a l } }$ reflects distance progress, and $r _ { \mathrm { { c o m p } } }$ grades completion. The detector is heuristic: precision and recall remain to be measured, and false positives can penalize valid programs. Reward-hacking risks in the shaped terms motivate independent evaluation [8], [9].

## IV. TASKS, CONFIGURATIONS AND EVALUATION

The following terms are used throughout. An archived run is one of the runs whose records this paper analyzes; its scorer parsed program output, and a logged rate is the fraction of observed binary labels equal to 1, including the binary component of the dense reward. Validated success is reserved for geometric checks outside the generated program; the archived runs provide labels parsed from program output. The denoised reward is the mean of M=5 program rewards for one instruction. The prompt-fix baseline is the instruction policy without LoRA training, given the same API addendum, on the M task, with an instruction-condensing call before code generation that the training runs do not have. I and M name the archived cube layouts: training uses the twoand five-cube targets in Fig. 1; the benchmark suite uses distinct nine- and thirteen-cube versions. The single-epoch runner is a later re-implementation used only in the learningrate and epoch comparison. The preliminary sweep is an earlier set of seven trained settings and two no-RL controls whose configurations vary in group size, horizon, reward and instruction format.

## A. Environment, tasks and configurations

The study uses MuJoCo [10] through a Gymnasium interface [19], with a UR5e arm and parallel gripper (Fig. 2). RQ1 includes image-conditioned prompting; RQ2/RQ3 use textual task/scene descriptions.

![](images/729ad85a702a934934facbbe41d10f58157bfd6f197718d22efef83a98faf520.jpg)  
Fig. 2. MuJoCo Simulation Environment. A UR5e arm with parallel gripper manipulates red cubes on the rear table and transfers them to the disposal area on the front table.

Pick-and-place moves selected cubes to target positions or a disposal area. Shaping generates letter coordinates; its workspace-bounds metric is distinct from correct letter geometry and from robot execution. Reshaping introduces removals, additions and assignments between source and target coordinates. A defined suite contains 12 targets and therefore 66 unordered pairs, or 132 directed transformations.

A task-distance score $D ( A , B ) = \left| A \triangle B \right| + 6 \mathrm { \mathbf { 1 } } _ { \mathrm { e n c l } } + 3 \mathrm { \mathbf { 1 } } _ { \mathrm { s y m } }$ bands the 66 pairs by coordinate-set difference and changes in archived enclosure and symmetry labels; on the 16 planning pairs its Spearman correlation with an assignment-derived operation proxy is 0.919, and the coordinate-set term alone gives 0.975. The score is symmetric and is not a measure of execution difficulty.

RQ1 reports o1 and o4-mini prompting configurations; the o1 execution cells contain five trials and o4-mini twenty. RQ2/RQ3 report a Qwen3.5-35B-A3B instruction policy with rank-32 LoRA (α = 64) and a frozen Qwen3-Coder-Next service through OpenRouter. Central settings are K=32, M=5, E=3, 20 steps, instruction/code temperatures 0.7/0.1, a 4096- token generation limit and 120-second execution timeout.

Offline recovery establishes 60 saved o4-mini coordinate outputs (12 shapes, five each), all within strict workspace bounds. A separate archive has 80 plans over 16 directed pairs; 76 satisfy an explicit coverage/uniqueness rubric, and 38 also satisfy maximal reuse. The second condition is a stricter preference layered on the first, and both count plans rather than executions. In the small five-to-two-node example, all five o1 and five o4-mini plans are coverage-feasible; adding maximal reuse gives 4/5 and 5/5. In this example the Hungarian solver and the listed model assignments have the same padded objective (0.653); the smaller real-to-real subtotal quoted for the model plans measures a different quantity.

## B. Evaluation protocol and interpretation of logged rewards

A geometric criterion requires an explicit correspondence between final objects and immutable targets, a coordinate frame, tolerance boundary, object-count rule, and removal/addition protocol. The archived descriptions state a 0.1 m tolerance for RQ1 and a 0.05 m XY tolerance for RQ2/RQ3. The tolerances bound placement error; comparison with cube size requires version-specific cube dimensions that are not established by the archive.

In the archived runs, success and geometric-error messages were parsed from the program’s standard output, and process exit and timeout status marked execution failures. In the retained records the placement checker is appended by the runner to the generated program and runs inside the program’s process, so its verdict shares an output stream with whatever the program prints; the archive does not establish which scoring-source version was used for each run. The retainedrecord scan finds at least 19 positive rewards whose retained log contains an explicit geometric-failure message in the main binary run, 18 in the dense run and 5 in the M=1 run (Section VII). Success-line records (352, 343, 163) print at most 0.009 m, failure-line records (87, 78, 99) at least 0.0874 m, and the 19, 18 and 5 positively labeled failure records 0.0874 to 0.9515 m (Fig. 3). These are lower bounds: the driver kept only the first reward-maximizing program per instruction, so the 19 occur among the 371 instructions of the main run with at least one positive logged reward and the 5 among the 168 such instructions of the M=1 run, and the other programs behind each averaged reward were not saved. The single-epoch runner used in the learning-rate and epoch comparison, like the current source of the main runner, falls back to accepting any output that contains SUCCESS and no FAIL; whether the archived runs executed that source is not established. Removing these visible contradictions from the counts leaves the unsaved programs unresolved and does not change the policy trained on the original rewards.

The proposed independent protocol assigns simulator state, immutable task data and result reporting to a separate software supervisor. Generated programs issue control actions; the supervisor computes and reports task success. The protocol adopted for new evaluations inventories all ten cubes and predeclares placement, addition, disposal and reserve roles; primary success requires task placement, required disposal, reserve preservation and clean execution, with diagnostics reported separately, and reserves cannot substitute for task cubes. Every method receives the same specification.

The CPU implementation of this protocol passed ten negative controls and five predeclared 120 s no-action holds under a 0.1 mm drift rule (with a 1 mm XYZ reserve tolerance for the pinned setup), and a hand-authored reference controller passed 40 of 40 predeclared scenes (20 I, 20 M) under it at 1 cm placement tolerance, with no case replacement or controller adaptation. These results characterize the evaluation infrastructure. Evaluation of model-generated programs, target-host validation and re-scoring retained programs are the next steps. Such re-scoring covers one program per instruction; reconstructing averaged rewards would require the unsaved samples.

![](images/da1953ece748beb1dc8197b2f8784126888aa2eeaaa59af131318c2666654a71.jpg)

![](images/b3aeed488a24fa1e77aed95400107511591750048cc90bcd8c3a2d092d64a7a3.jpg)  
Fig. 3. Evidence behind a positive logged label. (a) The 1,055 positive labels of the binary M=5 run: 684 with no retained record, 352 with the checker success line, 19 with a geometric-failure message. (b) Largest printed matched distance per record against the 0.05 m tolerance; failure logs may stop at the first violation. The open diamond (dense step 13, sample 026) has a compacted component list and is excluded from the index-aligned 18. Right: records without a geometry line. One program is retained per instruction (one of five for M=5; one for M=1), so retention differs across M. Success messages require independent verification. Each configuration has one archived run.

TABLE I  
REPORTED EXECUTION COUNTS (SUCCESSES/TRIALS) FOR ONE-SHOT CODE GENERATION AND ICMC. THE ONE-SHOT CELLS WERE RECOUNTED FROM THE SAVED NOTEBOOK STATUS LABELS; THE ICMC CELLS ARE THE ARCHIVED REPORT VALUES, OF WHICH ONLY THE O4-MINI RESHAPING CELL HAS A NOTEBOOK RECOUNT (SEE THE TEXT AT TABLE II).
<table><tr><td>Task</td><td>Model</td><td>One-shot</td><td>ICMC</td></tr><tr><td>Pick/place</td><td>01</td><td>5/5</td><td>5/5</td></tr><tr><td>Pick/place</td><td>o4-mini</td><td>7/20</td><td>12/20</td></tr><tr><td>Reshaping</td><td>01</td><td>1/5</td><td>3/5</td></tr><tr><td>Reshaping</td><td>04-mini</td><td>5/20</td><td>3/20</td></tr></table>

## V. PROMPTING COMPARISONS

Table I compares one-shot code generation with ICMC on pick-and-place and reshaping, with five trials per o1 cell and twenty per o4-mini cell. The o1 pick-and-place cells tie at 5/5. With o4-mini, ICMC records 12/20 against 7/20 for one-shot on pick-and-place (Wilson 95% intervals [38.7, 78.1]% and [18.1, 56.7]%; Fisher exact $\mathtt { p } = 0 . 2 0 5 )$ , and one-shot records 5/20 against 3/20 on reshaping. These results characterize their respective task protocols and model configurations; isolating reasoning effects across models requires matched conditions.

TABLE II  
ARCHIVED O4-MINI RESHAPING OUTCOMES (N=20 PER ROW). THE FIRST TWO ROWS ARE CHAIN-OF-CODE AND ITERATIVE-CORRECTION IMPLEMENTATIONS ON THE SAME BACKBONE; THE ICMC ROWS DIFFER IN FUNCTION SET AND PROMPT. BASELINE AND ICMC ROWS USE DIFFERENT TASK SETS AND GENERATION BUDGETS; THE COMPARISON IS DESCRIPTIVE.
<table><tr><td>Configuration</td><td>Code-valid</td><td>Task-success</td></tr><tr><td>Chain-of-Code implementation</td><td>55%</td><td>20%</td></tr><tr><td>Iterative correction</td><td>15%</td><td>5%</td></tr><tr><td>Base ICMC</td><td>75%</td><td>15%</td></tr><tr><td>ICMC + amended functions</td><td>100%</td><td>20%</td></tr><tr><td>ICMC + constrained prompt</td><td>70%</td><td>35%</td></tr><tr><td>ICMC + both</td><td>90%</td><td>50%</td></tr></table>

![](images/94dcde195b5d5648613504b60360bd7e8a43f1ec230e794caa0e43fffc37f822.jpg)  
Fig. 4. Reported code validity and task success from Table II for the six o4-mini reshaping configurations (n=20 each). Configurations use different task sets.

Table II separates base ICMC from two engineering changes; the final row reports ICMC with both modifications. The iterative implementation allows an initial generation plus three retries and is scored on the final attempt; its any-attempt success is also 1/20. The 40 baseline runs all request the M-to-I transformation, whereas the ICMC notebooks mix four pairs from the Easy band of the task-distance score and include manual reruns, so the baseline rows and the ICMC rows do not share a task set (Fig. 4). Under a perfile recount of the notebooks, the notebook matching the base ICMC configuration yields 14/20 code-valid and 2/20 explicit success messages, and the final row yields 4/20 or 13/20 depending on whether the first or last saved manual execution is taken. The table keeps the archived report values; code validity in some rows counts timeouts as valid.

Wilson 95% intervals for n=20 cells are wide: base ICMC at 3/20 has [5.2, 36.0]%, the combined ICMC row at 10/20 has [29.9, 70.1]% and Chain-of-Code at 4/20 has [8.1, 41.6]%, so these three task-success intervals overlap pairwise. These intervals summarize individual proportions; testing differences between rows remains future work.

![](images/8a5dc9ec530415596ffa8c5e2bde79f3e20a7425579c6e84c0beaa794db3f1e9.jpg)  
Fig. 5. Logged positive-label rates for binary M=5 (a), dense M=5 (b) and binary M=1 (c), one trajectory each. Ribbons: descriptive Wilson 95% intervals over 160, 153 to 160 and 32 labels per step; the strip under (b) prints denominators below 160. Black segments: quarter means of step rates. Gray band: prompt-fix M-task replications, 32/128 and 35/128. Red ticks: retained positive-label records reporting geometric failure (19, 18 and 5; counts above one printed). Crosses: steps with missing retained files. Letters: recovered I/M task schedule. The rates summarize correlated logged labels within runs; independent task verification and seed replication remain future work (Section IV-B).

## VI. TRAINING THROUGH A FROZEN CODE GENERATOR

Fig. 5 plots the logged positive-reward rate per training step for the three archived runs, binary M=5, dense M=5 and binary M=1, each a single trajectory of 20 steps with K=32 instructions per step. The task follows a recovered schedule shared by all three runs (two-cube I target at steps 1, 3, 4, 12, 13, 14, 16, 17 and 20; five-cube M target otherwise), so the first and last five-step windows have the same task composition, and the pooled binary rate is 537/1440 on I steps and 518/1760 on M steps. The binary run has 160 reward entries per step and rises from 45/160 at step 1 to 63/160 at step 20. The dense run’s binary component is present for 3,170 of its 3,200 scheduled slots (153 to 160 per step); its first, last and peak step rates are 46/155, 60/158 and 63/159, and its 989 positive labels are 31.2% of the observed slots and 30.9% of the scheduled slots if every missing slot is counted as a failure. The gray band spans the prompt-fix baseline on the M task, 32/128 and 35/128 in its two replications (mean 26.2%), run with an instructioncondensing call that the training runs do not have and at code-generation temperature 0.3 where the training runs used 0.1.

Averaged over the first and last five steps, the logged rate is 31.6% and 35.3% for the binary run (253/800 and 282/800), 30.0% and 33.7% for the dense run (pooled counts 238/794 and 268/795) and 25.0% and 30.6% for the M=1 run (40/160 and 49/160). These percentages are means of perstep rates; the dense pooled fractions differ slightly because its denominators vary. These increases describe logged training behavior within one trajectory per configuration, with shared instructions and tasks; independent held-out checkpoint evaluation remains future work. Wilson ribbons assume independent Bernoulli labels; estimating uncertainty across training seeds requires replicated runs.

The preliminary sweep’s configurations differ from the central scheme in more than one factor at a time (its reference configuration uses K=128 and 10 steps where the central scheme uses K=32 and 20), and the single-epoch and E=3 runs also differ in execution fixes and provenance, so neither comparison isolates learning-rate or epoch effects.

A future matched design would fix M and the pipeline across three arms (low-LR/E=3, low-LR/E=1 and high-LR/E=3), plus high-LR/E=1 for the interaction. The archive lacks these matched arms. The M=5 and M=1 runs compare reward estimators at different downstream sampling costs; equal-compute comparisons remain future work.

A five-step M=1 record in the RQ2 archive is a byteidentical copy of the first five steps of the M=1 run and is counted once.

## VII. TRACEABLE INSTRUCTIONS AND FAILURE ANALYSIS

Four instruction samples from step 20 of the main binary run show what the intermediate makes visible. The pick-andplace request in every pick-and-place run is: “Fetch the red cubics from their original spots (on the table at the back) and then place them in the same disposal area (the central point of the table on the right).” The stage-2 request asks for specific logical control instructions for the parameters and three core motion-planning functions. In the RL runs, stage 3 receives the base API and task code, the full sampled instruction, the completion request and the API addendum reproduced in Appendix A. Appendix A reproduces the task requests, the stage requests, the API addendum and excerpts of one traced step-20 record byte-for-byte from the archived files.

Sample 000 of that step includes the following instruction excerpt:

B. fetching\_cube (State Machine Steps 2, 3, 4)

\- State 2 (Approach):

\- Define target\_pos slightly above the cube (z + 0.02).

\- Generate trajectory to move arm to this position.

\- On completion (terminated), move to State 3.

\- State 3 (Grasp):

\- Check closeGripper.

\- If successful, move to State 4.

The wording is retained with Markdown formatting and indentation normalized. Its five logged rewards are zero; retained code has a bracket-mismatch SyntaxError. This record separates the readable instruction from the syntax error in its generated program.

Three other samples from the same step expose distinct evidence types. Sample 001 has rewards (0, 1, 1, 0, 0), with a retained geometric-success message and maximum matched distance 0.0005 m; independent task verification is pending. Sample 002 has five zeros and a retained NameError for undefined pattern\_B. Sample 026 has five ones but its retained log reports distance 0.9515 m against 0.05 m tolerance and failed placement. Because only the first reward-maximizing program was retained (Section IV-B), the contradiction is established for that program; the other four programs behind the five rewards were not saved.

The 32 instruction outputs at this step contain 152–345 lines. With fence markers in the denominator but excluded from code interiors, median code-line fraction is 79.3%; 26 first lines are distinct. All 32 code interiors and all 32 outsidecode texts have different hashes. The 32 instructions therefore differ textually in both prose and code; behavioral diversity would need an execution-level comparison.

## VIII. DISCUSSION, LIMITATIONS AND REPRODUCIBILITY

The retained records support two diagnostic contributions. The intermediate instruction makes syntax errors, undefined names and discrepancies between rewards and execution logs traceable to specific instructions and programs. The reward audit identifies 19 positive-label records with reported geometric failure in the binary run and 18 in the dense run. These findings motivate independent task verification and distinguish changes in logged reward from improvements in validated task performance.

The study focuses on simulated manipulation. The RQ1 images serve as prompts rather than as a perception benchmark, and the RL runs are text-only. Hardware transfer would require camera and robot calibration, perception and stateestimation testing, contact and actuation robustness, latency measurement, and a separate safety review. Evaluation covers a subset of the 132 defined transformations; tests of the remaining tasks and generalization to held-out prompts, scenes and hardware remain future work.

The reproducibility scripts hash the evidence files they consume and regenerate the recounted values, traced examples and Fig. 5 without model calls, and contain the CPU tests for geometric matching, tolerance boundaries, counts and incomplete statistical windows, together with the protocol checks of Section IV-B. Further work includes integrating model-generated programs with the independent protocol, validating the target host, and establishing archived model and checkpoint provenance.

The pipeline’s cost has five components: local instructionpolicy training, hosted code generation, inference retries, simulator attempts and auxiliary parsing. Recovered codegeneration ledgers total \$6.88 for binary M=5, \$6.95 for dense M=5 and \$1.40 for M=1 (prefix and resumed segment counted once); these are partial API charges and exclude local training compute, and the 160 A100-hour and 200 USD aggregates are estimates not reconciled with the recovered ledgers.

## IX. CONCLUSION

ICMC exposes the instruction between task parameters and frozen code synthesis, so execution outcomes can be traced to specific instructions and programs. The main training trajectory shows a rise in logged positive-reward rate from 31.6% to 35.3%. Auditing program-written verdicts also identifies at least 19, 18 and 5 positive rewards with retained logs reporting geometric failure across the three runs, establishing the need to distinguish logged rates from validated success. The independent protocol supports reproducible evaluation: its CPU implementation passed ten negative controls, and a hand-authored reference controller completed all 40 predeclared scenes. Next steps include evaluating model-generated programs under the independent protocol, replicating training across seeds, matching learning-rate/epoch comparisons and expanding o1 samples.

in a while loop. Each call advances middle\_step by one   
<sub>,→</sub> phase when terminated==True.

```erlang
cube_id = move_to_pattern(cube_id,<sub>,→</sub>
```

for each parameters in [Motion Control Logistic Paramters]<sub>,→</sub>

```twig
for the ## three core functions for motion planning.<sub>,→</sub>
```

Please generate the specific [Logical Control Instructions]

terminated = env.step(robotic\_arm)[2]

Do NOT call these functions just once — loop until cube\_id   
<sub>,→</sub> >= pattern\_B.shape[0].

## ACKNOWLEDGMENT AND AI ASSISTANCE

An AI assistant was used to debug and refactor experiment code for efficiency. The models studied (o1, o4-mini, Qwen3.5-35B-A3B and Qwen3-Coder-Next) are evaluated or trained as part of the experimental design; their generated instructions and programs are experimental outputs. The authors take responsibility for all content, evidence and citations.

## APPENDIX A

## RECOVERED PROMPT COMPONENTS AND A TRACEDRECORD

Listings reproduce the cited archived line ranges without edits; original spelling is retained and long lines wrap.

## A. Task request strings

Pick-and-place (prompts/rq1-pick-place-user.txt, line 3, also line 3 of both RQ2 I/M instruction templates): [Task]: Fetch the red cubics from their original spots (on the ,→ table at the back) and then place them in the same ,→ disposal area (the central point of the table on the ,→ right).

Reshaping (prompts/rq1-reshaping-user.txt, line 3, also line 3 of both RQ2 reshaping templates):

## B. Stage requests

The 158-line RL stage-2 template prompts/rq2-I-instr uction-user.txt refers to an image and arm-control task (line 1), gives the task string (3), and introduces the 140-line reference code (5, 7–146). The text-only runs supplied no image. Lines 148–158 contain the parameters and instruction

```python
[Motion Control Logistic Paramters]
env = gymnasium.make('manipulator_mujoco/TestEnv-v0',
<sub>,→</sub> render_mode='rgb_array')
observation, info = env.reset(seed=42)
robotic_arm = np.array([0.3, 0, 0.8, 0, 0, 0, 1]).reshape(1,
<sub>,→</sub> -1)
cube_id = 0
steps = 1
middle_step = 1
break_flag = 0
terminated = env.step(robotic_arm)[2]
```

regarding the whole [Basic operational codes], especially<sub>,→</sub>

The RL stage-3 builder (prompts/rq2-code-builder-sou rce.txt, lines 23–30) uses the stage-2 template without line 158 as trimmed\_prompt and the entire sampled instruction as logical\_suggestions. The denoised driver appends the API addendum below and sends one user message.

```hcl
code_prompt = f"""{trimmed_prompt}
```

[Logical Control Instructions]:   
{logical\_suggestions}

Based on the above logical control instructions, please ,→ complete the Python code.

Return the complete code with all functions implemented. Please do not change the functions in [Basic operational codes] unless you must do that."""

RQ1 uses a different request/separator (prompts/rq1-cod e-builder-source.txt, lines 11–13) and an image whose original bytes are absent from the retained archive:

sub\_task='''\nBased on the [Basic operational codes] and ,→ the [Logical Control Instructions], return the

completed version of the [Basic operational codes].'''<sub>,→</sub>

new\_text = new\_task\_description + '\n\n[Logical Control

Instructions]\n\n' + generated\_logical\_suggestion +<sub>,→</sub>

sub\_task<sub>,→</sub>

## C. API-documentation addendum

prompts/rq2-api-addendum.txt stores the denoised driver’s exact appended Python string; lines 1–2 are blank, and 3–27 follow:

[CRITICAL API NOTES — Read carefully before writing code]:

1. env.step() expects a 2D trajectory array of shape (N, 8)   
<sub>,→</sub> where column 0 is time   
and columns 1-7 are the pose [x, y, z, qx, qy, qz, qw].   
Do NOT iterate over trajectory rows. Pass the full   
<sub>,→</sub> trajectory at once:   
terminated = env.step(robotic\_arm)[2]   
NOT: for row in robotic\_arm: env.step(row[1:]) # WRONG —   
,→ causes IndexError

while cube\_id < pattern\_B.shape[0]:

if break\_flag: break

elif steps == 2:

steps)<sub>,→</sub>

elif steps == 3:

```python
robot_pos, steps, middle_step, robotic_arm,
```

robot\_pos, env, middle\_step, robotic\_arm,<sub>,→</sub>

3. Use \`pattern\_B\` consistently (the variable loaded from the <sub>,→</sub> .txt file).

```rust
The get_target_pos() helper uses `pattern` — replace it
,→ with `pattern_B` or
```

define \`pattern = pattern\_B\` at the top.

## D. One traced record: step 20, instruction sample 001

examples/step020-rollout001 exactly copies the denoised-binary archive’s step\_020/rollout\_001. Its reward.json records [0,1,1,0,0] over M = 5 programs (mean 0.4); only sample 1, the first reward maximum, and its logs were retained. Lines 1–14 of the 229-line instruction follow:

Here are the specific \*\*Logical Control Instructions\*\* to complete the task of fetching red cubes from the back<sub>,→</sub> table and placing them in the central disposal area on the<sub>,→</sub> ,→ right.

These instructions fill in the missing logic within the \`### starting the replacement\` and \`### ending the,→

replacement\` blocks and refine the control flow to ensure,→

the \`fetching\_cube\` and \`move\_to\_pattern\` functions are,→

called sequentially for every cube.<sub>,→</sub>

## ### Logical Control Instructions

\#### 1. Initialization & Pre-flight Check

Before entering the main loop, ensure the environment is stable and the initial state variables are correctly set<sub>,→</sub> ,→ to trigger the first perception phase.

\* \*\*Action\*\*: Reset the simulation time and verify the ,→ initial \`robotic\_arm\` state.

\* \*\*Logic\*\*:

\* Wait until \`sim\_time > 2.0\` (as per your existing ,→ code) to ensure the arm has settled.

\* Reset \`steps\` to \`1\` (indicating we are ready to start <sub>,→</sub> the perception phase).

In the 188-line code.py, lines 122–154 delimit the modelwritten main loop; 122–131 follow. The runner appends a placement checker (158–188) after env.close(). Absent from the raw model response, the checker shares the program’s process.

\### starting the replacement

# Main Loop Logic   
while cube\_id < pattern\_B.shape[0]:   
# 1. Perception Phase: Identify cube location   
if steps == 1:   
pos, robot\_pos, steps, cube\_id, break\_flag =   
starting\_perception(cube\_id, pattern\_B, env,<sub>,→</sub>   
steps)<sub>,→</sub>   
# Safety check if cube\_id exceeded   
if cube\_id >= pattern\_B.shape[0]:   
break

exec\_stderr.txt is empty; the checker’s six-line exec\_s tdout.txt follows: tdout.txt follows:

planned\_positions: [[ 0.4 -0.1]   
[ 0.6 -0.1]]   
cube\_positions: [[ 0.40042537 -0.09990171]   
[ 0.60047451 -0.09992712]]   
Success: All cubes correctly placed! max\_match\_dist=0.0005   
Pick-and-place routine completed successfully!

The success line and 0.0005 m distance are this sample’s only retained placement evidence; the other four programs/logs are unsaved. Independent checking would determine whether this logged positive meets task-success criteria (Section IV-B). Section VII contrasts step-20 samples 000, 002 and 026.

## E. Provenance boundary

These recovered templates, builder source and outputs are user-text components, not complete API transcripts. The original trainer, chat serializer, any system message, tokenizer

version and RQ1 image bytes are not available in the retained archive. Complete templates, four full step-20 records and a SHA-256 evidence manifest are retained for release.

## REFERENCES

[1] J. Liang, et al., “Code as policies: Language model programs for embodied control,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA). IEEE, 2023, pp. 9493–9500.

[2] I. Singh, et al., “ProgPrompt: Generating situated robot task plans using large language models,” in Proc. IEEE Int. Conf. Robot. Autom. (ICRA). IEEE, 2023, pp. 11 523–11 530.

[3] S. H. Vemprala, R. Bonatti, A. Bucker, and A. Kapoor, “ChatGPT for robotics: Design principles and model abilities,” IEEE Access, vol. 12, pp. 55 682–55 696, 2024.

[4] M. Ahn, et al., “Do as I can, not as I say: Grounding language in robotic affordances,” in Proc. 6th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205. PMLR, 2023, pp. 287–318.

[5] W. Huang, et al., “Inner monologue: Embodied reasoning through planning with language models,” in Proc. 6th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 205. PMLR, 2023, pp. 1769–1782.

[6] D. Batra, et al., “Rearrangement: A challenge for embodied AI,” arXiv preprint arXiv:2011.01975, 2020. [Online]. Available: https://arxiv.org/abs/2011.01975

[7] C. Li, et al., “Chain of code: Reasoning with a language modelaugmented code emulator,” in Proc. 41st Int. Conf. Mach. Learn. (ICML), ser. Proc. Mach. Learn. Res., vol. 235. PMLR, 2024, pp. 28 259–28 277.

[8] D. Amodei, C. Olah, J. Steinhardt, P. Christiano, J. Schulman, and D. Mané, “Concrete problems in AI safety,” arXiv preprint arXiv:1606.06565, 2016. [Online]. Available: https://arxiv.org/abs/16 06.06565

[9] L. Gao, J. Schulman, and J. Hilton, “Scaling laws for reward model overoptimization,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), ser. Proc. Mach. Learn. Res., vol. 202. PMLR, 2023, pp. 10 835–10 866.

[10] E. Todorov, T. Erez, and Y. Tassa, “MuJoCo: A physics engine for model-based control,” in Proc. IEEE/RSJ Int. Conf. Intell. Robots Syst. (IROS). IEEE, 2012, pp. 5026–5033.

[11] Z. Shao, et al., “DeepSeekMath: Pushing the limits of mathematical reasoning in open language models,” arXiv preprint arXiv:2402.03300, 2024. [Online]. Available: https://arxiv.org/abs/2402.03300

[12] E. J. Hu, et al., “LoRA: Low-rank adaptation of large language models,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2022. [Online]. Available: https://openreview.net/forum?id=nZeVKeeFYf9

[13] A. Brohan, et al., “RT-1: Robotics transformer for real-world control at scale,” in Proc. Robotics: Science and Systems (RSS), 2023.

[14] A. Brohan, et al., “RT-2: Vision-language-action models transfer web knowledge to robotic control,” in Proceedings of The 7th Conference on Robot Learning, ser. Proceedings of Machine Learning Research, J. Tan, M. Toussaint, and K. Darvish, Eds., vol. 229. PMLR, 06–09 Nov 2023, pp. 2165–2183. [Online]. Available: https://proceedings.mlr.press/v229/zitkovich23a.html

[15] D. Driess, et al., “PaLM-E: An embodied multimodal language model,” in Proc. 40th Int. Conf. Mach. Learn. (ICML), ser. Proc. Mach. Learn. Res., vol. 202. PMLR, 2023, pp. 8469–8488.

[16] M. J. Kim, et al., “OpenVLA: An open-source vision-language-action model,” in Proc. 8th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 270. PMLR, 2025, pp. 2679–2713.

[17] A. Zeng, et al., “Transporter networks: Rearranging the visual world for robotic manipulation,” in Proc. 2020 Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 155. PMLR, 2021, pp. 726–747.

[18] M. Shridhar, L. Manuelli, and D. Fox, “CLIPort: What and where pathways for robotic manipulation,” in Proc. 5th Conf. Robot Learn. (CoRL), ser. Proc. Mach. Learn. Res., vol. 164. PMLR, 2022, pp. 894–906.

[19] M. Towers, et al., “Gymnasium: A standardized interface for reinforcement learning environments,” in Adv. Neural Inf. Process. Syst., vol. 38. Curran Associates, Inc., 2025, pp. 163 114–163 129.