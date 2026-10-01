# Can Agents Trust Their Skills? Uncovering Unsafe Chains of Trust in Skill-Based LLM Agents

Yan Wang<sup>1,3†</sup> Zhihao Zhang<sup>2†</sup> Ke Chen<sup>1,3</sup> Kai Chen<sup>1✉</sup> Yaqin Zhang<sup>1✉</sup> Duohe Ma<sup>1</sup> Jun Dai<sup>2</sup> Xiaoyan Sun<sup>2✉</sup>

<sup>1</sup>Institute of Information Engineering, Chinese Academy of Sciences

<sup>2</sup>Worcester Polytechnic Institute

<sup>3</sup>School of Cyberspace Security, University of Chinese Academy of Sciences

† These authors contributed equally to this work.

✉ Corresponding authors.

## Abstract

LLM agents increasingly rely on installable skills, which are packages of instructions, code, and resources that equip them with task-specific capabilities and, once installed, can be automatically invoked across subsequent user tasks. This creates a chain of trust in which users delegate authority to agents, while agent frameworks admit skill-provided content into the agents’ context with insufficient validation, allowing malicious skills to influence agent behavior under that delegated authority. Yet, little is known about whether this trust model adequately constrains untrusted skill content before it reaches security-sensitive operations, or how frequently such trust violations arise in real-world agents. We present TrustProbe, a framework for uncovering unsafe chains of trust in skill-based LLM agents. First, TrustProbe analyzes agent source code to identify source-to-sink call paths from skill-controlled inputs to security-sensitive operations. Second, it generates semantically realistic SKILL.md seeds with injected canaries and evolves them through feedback-guided scheduling and mutation. Finally, it validates vulnerabilities using an oracle that confirms attacker-controlled flows and verifies observable harm. Across 11 opensource agents, eight with more than 10,000 GitHub stars, TrustProbe identifies 104 taint-style vulnerabilities. Validation on a large corpus of real-world skills collected from public hubs such as ClawHub further shows that 25.1% of skill-agent trials exercise the identified vulnerable paths, with payload injection successfully weaponizing 15 of the vulnerabilities. These results reveal a systematic trust failure in skill-based LLM agents: untrusted skill content can reach securitysensitive operations and exercise authority delegated by users to their agents.

## 1 Introduction

LLM agents act on behalf of users through tools that access files, execute commands, and communicate with external services (Yao et al., 2023; Wang et al., 2024). Installable skills extend these capabilities through reusable packages of instructions, code, and resources (Destefanis et al., 2026; Ouyang et al., 2026). The framework discovers skills from the metadata in their SKILL.md and loads relevant instructions on demand using progressive disclosure (Anthropic, 2025). Marketplaces such as ClawHub and SkillsMP distribute these artifacts across agent ecosystems (ClawHub, 2026; SkillsMP, 2026). This creates a chain of trust: users delegate authority to agents, while agents rely on third-party skill providers to determine how that authority is exercised.

This chain of trust creates a direct attack surface. A malicious author can embed instructions or parameters in a skill and publish it through ordinary distribution channels. Once installed, the artifact persists and can induce harmful actions when subsequently invoked, without further attacker interaction. This extends the supply-chain risks of software packages to agent instructions (Ohm et al., 2020; Zimmermann et al., 2019; Duan et al., 2021). A study identifies 157 malicious skills in circulation (Liu et al., 2026), while OWASP ranks Software Supply Chain Failures third in its 2025 Top 10 (OWASP Foundation, 2025). Recent studies examine attacks across agent workflows (Zhang et al., 2025; Debenedetti et al., 2024; Chen et al., 2024; Zou et al., 2025; Shi et al., 2025; Zhan et al., 2025) and through skill files (Schmotz et al., 2026; Jia et al., 2026). We study the framework mechanisms underlying these failures: how installed skill content reaches security-sensitive operations, how skill-delivery architectures and approval controls shape the resulting behavior, and how broadly these failures occur across real-world agents and skills.

Systematically studying these questions requires tracing agent-skill interactions to concrete execution evidence, which raises three challenges. First, heterogeneous discovery, loading, and execution paths across agents obscure how skill content propagates to security-sensitive operations. Second, a skill must satisfy the framework’s loading conditions and be invoked through a plausible task before its content can affect execution, while each attempt incurs a costly and nondeterministic agent run. Third, exploitation is often non-crashing, so establishing attacker influence and confirming harmful effects require distinct forms of evidence (Ruan et al., 2024; Evtimov et al., 2025).

To address these challenges, we present TrustProbe, a framework for uncovering unsafe chains of trust in skill-based LLM agents. TrustProbe uses source-to-sink analysis to characterize paths through which skill content can exercise the agent’s authority. It then applies directed greybox fuzzing (Bohme et al., 2017; Huang et al., 2022), using runtime feedback to select and revise plausi-¨ ble SKILL.md inputs. Injected canary markers reveal how attacker-controlled content propagates. The oracle verifies whether the resulting execution produces an observable harmful effect. Finally, paired skill-versus-prompt and approval-control experiments isolate how skill delivery and authorization mechanisms shape these failures.

Across 11 open-source agents, eight with more than 10,000 GitHub stars, TrustProbe identifies 104 verified taint-style vulnerabilities spanning command injection, arbitrary file disclosure and tampering, attacker-directed network requests, and code execution. When the same skill bodies are delivered as direct prompts, only 33 of the 104 (31.7%) vulnerabilities remain exploitable. Source inspection connects these results to recurring skill-delivery patterns and gaps in approval enforce ment and policy coverage. In our real-world skill evaluation, 743 of 2,963 (25.1%) tests across 633 real-world skills trigger the source-to-sink call paths behind the identified vulnerabilities. Payload injection with minimal edits turns 15 triggering skills into complete attacks (§5.4). Together, these results demonstrate that third-party skill content can systematically reach and exercise delegated authority across heterogeneous agent frameworks. Our contributions are threefold:

• Characterization of skill-mediated trust failures. We identify failures in how agents rely on installed skills under delegated authority, and characterize how skill-delivery mechanisms and approval controls shape these failures.

• TrustProbe. We develop TrustProbe, a framework for systematically exposing paths from attackercontrolled skill content to security-sensitive operations. TrustProbe combines source-to-sink analysis, directed greybox fuzzing, semantic seed generation and mutation, and a runtime oracle that establishes attacker influence and verifies observable harm.

• Empirical evidence across agents and real-world skills. Our study identifies 104 verified taintstyle vulnerabilities across 11 agents. Paired delivery and approval-control experiments characterize the role of framework mechanisms, while validation on 633 real-world skills establishes the practical exposure of the identified paths.

## 2 Related Work

Agent attacks and untrusted content. Research on agent security examines how untrusted inputs redirect LLM behavior through user prompts (Toyer et al., 2024; Schulhoff et al., 2023) and external content (Greshake et al., 2023; Zhan et al., 2024; Yi et al., 2025; Chen et al., 2025a,b; Debenedetti et al., 2025; Beurer-Kellner et al., 2025). Agent Security Bench (ASB) evaluates attacks and defenses across prompt, tool, and memory stages (Zhang et al., 2025), while contextualintegrity benchmarks examine inappropriate disclosure across information sources (Mireshghallah et al., 2024; Shao et al., 2024; Fu et al., 2026; Mireshghallah et al., 2026). These risks motivate methods that identify suspicious content and determine whether it leads to harmful execution.

Static analysis and skill auditing. Static analysis inspects skill artifacts and agent code to identify potential risks. ClawHub Security Signals compares outputs from skill scanners (Koc et al., 2026). SkillProbe combines admission checks, comparisons of documented and implemented capabilities, and simulation of risks from skill composition (Guo et al., 2026b). LLMSmith combines static callpath analysis with prompt-based exploitation to investigate code-execution vulnerabilities (Liu et al., 2024). However, suspicious content or a candidate call path alone does not establish that a harmful operation will occur. Actual execution also depends on agent decisions and framework execution controls.

Dynamic testing and fuzzing. Dynamic testing complements static analysis with evidence from actual execution. Dynamic taint analysis tracks input propagation at runtime (Newsome & Song, 2005; Schwartz et al., 2010). Skill audits combine static inspection with behavioral verification to identify malicious skills (Liu et al., 2026), while Skill-Inject evaluates skill-file attacks by measuring harmful instruction following alongside legitimate task completion (Schmotz et al., 2026). AgentFuzz applies directed greybox fuzzing to detect taint-style vulnerabilities (Liu et al., 2025). Our study complements this literature by examining how skill-delivery architectures and approval controls shape harmful operations across agent frameworks.

## 3 Problem Statement

## 3.1 Trust Model and Failure Characterization

Skill-based agents establish a chain of trust among users, agent frameworks, and skill authors. Users delegate authority to an agent to carry out a task. The framework discovers and loads an installed skill, incorporating its content into the agent’s execution context. A malicious skill author can ex ploit this relationship to influence how the agent exercises delegated authority over the file system, shell, and network. We study trust failures in which this influence produces observable harm under the agent’s delegated authority. We evaluate such flows against a single operational invariant. Skilloriginated content may reach an operation that exercises the agent’s authority only when the content is validated for that operation, the user has consented to this class of use, or its provenance remains available at the enforcement point. A flow meeting none of these conditions is a trust failure. This invariant is grounded in the security mechanisms already present in the evaluated frameworks rather than imposed as an external policy. Eight of the eleven evaluated agent frameworks expose configurable approval layers that encode this boundary, and Appendix B shows that the SKILL.md path carries neither validation nor attribution. Where agent frameworks consult skill provenance at all, the signal is used to grant rather than deny. OpenClaw, for example, ships a setting that automatically trusts programs bundled inside an installed skill, so they run without asking the user.

We operationalize trust through the framework’s treatment of installed skill content and its role in execution. It is observable through discovery metadata, loading instructions, activation events, and prompt binding, rather than inferred from a model’s internal beliefs. These mechanisms determine how skill content enters the agent’s workflow and reaches its tools. We characterize unsafe chains of trust through taint-style vulnerabilities, where attacker-controlled input flows without adequate validation into a security-sensitive operation (Yamaguchi et al., 2014; Liu et al., 2025). The source is skill-controlled content, and the sink is an operation that exercises the agent’s capabilities. Propagation may occur through the LLM’s construction of tool-call arguments (Schick et al., 2023; Qin et al., 2024) or through the framework’s deterministic skill-installation code. Unlike a transient prompt, the skill is a persistent artifact that can be loaded in subsequent tasks. Our audit examines validation along these paths and finds inadequate handling of skill-originated content in the evaluated agents (Appendix B). Source-to-sink analysis identifies candidate propagation paths, while the oracle establishes attacker control and verifies observable harm (§4.5).

## 3.2 Motivating Example

We illustrate a confirmed case of arbitrary command execution in OpenClaw (OpenClaw, 2026): the agent executes an attacker-chosen command. Figure 1 traces the flow along exec → . . . → spawn. The user asks only “please use this skill and follow its instructions.” The framework’s system prompt lists installed skills in a mandatory section and instructs the agent to read the matching SKILL.md and follow it, so the agent retrieves the body of the installed skill. Its Inputs section declares an argv field whose value the agent adopts as the command, as its recorded reasoning shows. In the verification run this value is a benign command that writes a marker, and the recorded sink event shows spawn("/bin/zsh", ["-c", "echo ... > side-hit"]). The field value, supplied as a bare string with no wrapping, becomes the shell command itself, establishing direct attacker control over the command line, and the side-hit file independently confirms the resulting execution effect. This example illustrates an unsafe chain of trust: the user delegates a task to the agent, the framework binds third-party skill instructions into the agent’s execution context, and the installed skill supplies the command that the agent executes under that authority without an intervening validation boundary.

![](images/ab4008970da299807c0fdc0e45c6248190e36f3062d30de5ec2552281a0f4bab.jpg)  
Figure 1: How a SKILL.md field reaches the terminal.  
Figure 2: Overview of TrustProbe: source-to-sink analysis, seed generation, rule-guided mutation, and the bug oracle.

## 3.3 Threat Model

We assume non-malicious agent developers and an uncompromised runtime environment. The attacker is a malicious skill author who controls the content of a published skill. Once the skill is installed, its payload persists until invocation and requires no further interaction from the author. Distribution may occur through a marketplace or community sharing.

The attacker exploits the agent’s reliance on installed skill content to induce harmful operations under authority delegated by the user. The user expects the agent to complete the requested task, while the malicious skill supplies attacker-chosen instructions or parameters that guide execution toward the attacker’s objective. This creates a confused-deputy failure (Hardy, 1988; MITRE, 2025): the agent possesses legitimate authority, while third-party skill content influences how that authority is exercised. The threat resembles malicious software dependencies (Ohm et al., 2020; Zimmermann et al., 2019; Guo et al., 2026a), with natural-language instructions adding another way to direct execution. No privilege escalation is required, because the agent’s existing authority suffices.

## 4 TrustProbe Design

TrustProbe studies the trust failures defined in §3.1: attacker-controlled skill content induces harmful operations under the agent’s delegated authority. Source-to-sink analysis identifies candidate propagation paths from skill-controlled content to security-sensitive operations. Directed greybox fuzzing generates and revises skills to exercise these paths through the agent framework’s native loading and execution mechanisms. Because skill execution is expected, TrustProbe uses a bug oracle to establish attacker control and separately verify observable harm.

Figure 2 and Algorithm 1 summarize the campaign. Source-to-sink analysis first identifies candidate paths and the framework components that expose them. For each path, LLM-assisted generation produces a candidate SKILL.md test input (seed) with a plausible task and a canary in attacker-controlled fields. Feedback scheduling selects a seed, which is installed and executed in a fresh agent session. Runtime instrumentation records security-sensitive calls, their arguments, and the corresponding stack frames. Taint-confirmed paths exit the mutation loop for verification of ob servable harm. Unconfirmed seeds undergo rule-guided mutation based on the oracle verdict, trace stage, and feedback scores, then return to the pool. Each fuzzing iteration consumes a full agent session, and the resulting trace, sink events, and feedback scores guide subsequent scheduling.

These components address the challenges of §1. Source-to-sink analysis and seed generation map heterogeneous execution paths to plausible skill tasks. Feedback-guided scheduling and mutation allocate costly, nondeterministic agent runs, while the oracle separates evidence of attacker control from evidence of harmful effect. The resulting traces and verified vulnerabilities support the study of prevalence, framework mechanisms, and real-world exposure in §5.

## 4.1 Source-to-Sink Analysis

Source-to-sink analysis identifies candidate propagation paths from skill-controlled content to operations that exercise agent capabilities. We collect function calls and arguments using Python abstract syntax tree (AST) traversal or the TypeScript Compiler API. Calls are matched against the language-specific sink inventory (Appendix F). From each matched sink call, we trace callers backward through the call graph using breadth-first search, retaining project-specific wrapper functions. Across 11 agents this yields 1,566 source-to-sink call paths, grouped into 202 sink-function pools. These paths define the targets for dynamic testing. Runtime taint confirmation and behavioral verification then determine which paths constitute trust failures.

## 4.2 LLM-assisted Seed Generation

To study how skill content directs execution, a seed must pass the agent framework’s loading machinery and describe a task relevant to the target component. TrustProbe uses an LLM to construct a valid SKILL.md seed for each candidate source-to-sink path, including loading metadata, task instructions, and attacker-controlled fields targeting the sink. Seed generation is guided by a semantic profile containing the agent, language, sink, source field, and relevant program identifiers, together with a worked example that helps infer a plausible task for the target component.

Figure 3 in Appendix C shows the prompt. The resulting skill satisfies the framework’s native loading requirements, while the generated task directs execution toward the target component. Trust-Probe then inserts a deterministic canary into the attacker-controlled field, derived deterministically from the target and variant identifiers,

$$
\mathrm { c a n a r y } = H ( \mathrm { i d } _ { t } \parallel \mathrm { i d } _ { v } ) ,\tag{1}
$$

where H is a hash function, ${ \mathrm { i d } } _ { t }$ and $\mathrm { i d } _ { v }$ are the target and variant identifiers, and ∥ denotes string concatenation. A fixed target-variant pair yields the same canary. One initial seed is generated per source-to-sink path, and the identifiers link each confirmed vulnerability to its source-to-sink target. The recorded evidence supports reassessment and controlled replay with automatic approval disabled in §5.2.

## 4.3 Feedback-driven Seed Scheduling

Each fuzzing iteration requires a costly and nondeterministic agent session. The scheduler therefore prioritizes seeds whose executions both align with the target component and make progress toward

its security-sensitive sink. Seeds targeting the same sink function form a pool $P _ { j } ,$ with an independent mutation budget. Within each pool, the scheduler selects the next seed using directed greybox feedback (Chen et al., 2018),

$$
s ^ { * } = \operatorname * { a r g m a x } _ { s \in P _ { j } } F ( s ) .\tag{2}
$$

$$
F = \alpha \sigma + \beta \delta - \pi ,\tag{3}
$$

$$
\pi = \gamma c _ { \mathrm { s } } + \eta c _ { \mathrm { c } } , \qquad \delta = 1 0 ( 1 + d ) ^ { - k } .\tag{4}
$$

Here $\sigma \in [ 0 , 1 0 ]$ is a semantic score assigned by an LLM using the rubric in Figure 4. Given the target path, seed, and execution trace tr, the scorer estimates how well the generated task targets and exercises the intended framework component. The execution trace grounds this judgment in observed behavior rather than seed text alone. The distance score $\delta$ measures proximity to the sink on the audited call graph, with

$$
d = \operatorname* { m i n } _ { \substack { n \in \mathrm { R e a c h e d ( t r ) } } } \mathrm { h o p s } ( n , \mathrm { s i n k } ) ,\tag{5}
$$

where d is the smallest hop count from any node reached in the trace to the sink callsite; $\delta$ saturates at 10 when the sink is reached. The penalty π discourages repeated selection, with $c _ { \mathrm { s } }$ and $c _ { \mathrm { c } }$ counting selections of the seed and its path. The weights α and $\beta$ balance semantic and distance scores, γ and η scale the selection counts, and k is the distance decay exponent, all defaulting to 1.

## 4.4 Rule-guided Seed Mutation

Mutation revises the skill components that determine loading and target-path execution. When a selected seed does not confirm its target path, rules choose a mutation direction and an LLM produces the revised seed before returning it to the same pool. Direction selection uses the oracle verdict, feedback scores, and the stage at which execution ceases to progress toward the sink. Rules are evaluated top-down. A semantic-mismatch rule applies when σ falls below its threshold. A distance-stall rule applies when δ remains below threshold and the trace terminates at either the metadata or body stage. A sink-argument rule applies when execution reaches the sink but no argument contains a provable canary. Semantic alignment takes priority because the task must exercise the intended component. Once the sink is reached, δ reaches its maximum value, so no distance rule applies.

The four mutators target distinct stages of the skill-to-sink path: task semantics, metadata processing, instruction rendering, and sink arguments. The Semantic Mutator rewrites the task to align with the target component. The Metadata Mutator rewrites frontmatter used by the agent framework for loading and routing. The Body Mutator rewrites rendered instructions with sink-kind-specific structure. The Sink-Argument Mutator uses observed sink arguments to place the canary in the skill field most likely to control the target argument. All four preserve the current seed’s Inputs section to retain the source being tested. Appendix C gives the full rules and prompts.

## 4.5 Bug Oracle

TrustProbe’s bug oracle checks for taint-style vulnerabilities by confirming attacker-controlled flows and verifying observable harm. It treats attacker-controlled SKILL.md content as the source and security-sensitive operations as sinks. For a target path c and its seed $s ,$ the core taint predicate is

$$
\mathrm { T a i n t } ( c , s ) \longleftrightarrow \mathrm { c a n a r y } \in \mathrm { a r g s } ( \mathrm { s i n k } _ { s } ) ,\tag{6}
$$

where the canary appears only in attacker-authored skill content. The oracle checks whether the target security-sensitive operation is invoked (reached) and one of its arguments contains the canary (tainted). It also checks whether recorded stack frames traverse the statically recovered path (framed). The verdict chain confirmed is true only when

$$
\mathrm { c o n f i r m e d } ( c ) \iff \operatorname { r e a c h e d } ( c ) \wedge \operatorname { t a i n t e d } ( c ) \wedge \operatorname { f r a m e d } ( c ) .\tag{7}
$$

This establishes attacker control over a security-sensitive operation along the audited path.

For each taint-confirmed path, TrustProbe separately verifies that execution realizes an attackerspecified effect with an observable consequence, following Skill-Inject’s Attacker Task evaluation (Schmotz et al., 2026). Verification relies on case-specific evidence from session records and artifacts rather than a generic classifier (Huang et al., 2026). Evidence includes file contents disclosed in the conversation, a write to a test file, benign command output, or a request reaching a controlled endpoint. Synthetic sensitive content and controlled effects enable safe verification.

A target path is reported as an exploitable vulnerability only if attacker-controlled flow is confirmed and the corresponding harmful effect is independently verified. These two observations jointly establish the operational trust failure defined in §3.1. The attack-success rate (ASR) is the fraction of taint-confirmed paths with verified consequences (§5.2).

## 5 Evaluation

We evaluate the prevalence, mechanisms, and real-world exposure of agent-skill trust failures, as well as the effectiveness of TrustProbe through five research questions:

• RQ1. Prevalence of Trust Failures: How widely do trust failures arise across agents, and to what extent do approval controls prevent exploitation?

• RQ2. Delivery Mechanisms: How do skill delivery mechanisms affect exploitation, and which mechanisms in agent frameworks explain the observed differences?

• RQ3. Real-World Exposure: To what extent do real-world skills exercise vulnerable paths, and can payload injection convert these interactions into attacks?

• RQ4. Baseline Comparison: How do static and dynamic baselines detect these vulnerabilities?

• RQ5. Ablation Study: How do semantic seed generation, semantic feedback, and seed scheduling contribute to vulnerability discovery?

## 5.1 Experimental Setup

We evaluate 11 open-source TypeScript and Python agents that natively consume SKILL.md files (OpenClaw, 2026; OpenCode, 2026; Hermes Agent, 2026; Pi Coding Agent, 2026; Cline, 2026; Qwen Code, 2026; Kimi Code CLI, 2026; Pochi, 2026; Agent Zero, 2026; DB-GPT, 2026; Mistral Vibe, 2026). Table 1 lists the agents and their GitHub stars, recorded on September 5, 2026. Skills enter through each agent framework’s native loading path. Each test run uses a fresh session and isolated workspace. Each sink-function pool receives at most eight mutation rounds or 15 minutes, whichever comes first. We use deepseek-v4-flash throughout for both the target agents and TrustProbe’s generation, scoring, and mutation, holding the model fixed across frameworks. Appendices D–F provide revisions, configurations, and sink models.

## 5.2 RQ1: Prevalence of Trust Failures

The main campaign uses permissive, non-interactive execution configurations. We count a vulnerability once per audited source-to-sink path only when its attacker-controlled flow is confirmed by equation 7 and its observable harm is verified. ASR is the fraction of taint-confirmed paths with verified consequences. In Table 1, Time Cost sums campaign execution time, and Time to Exposure (TTE) measures the time to the first verified vulnerability; the aggregate reports the median across agents.

Across all evaluated agents, attacker-controlled skill content reaches security-sensitive operations and produces observable harm under the agent’s delegated authority. The campaign reaches 1,150 of 1,566 audited paths (73.4%) and 98 of 202 sink functions (48.5%), while 6.6% of audited paths satisfy the taint-confirmation predicate. All taint-confirmed paths also produce the intended observable consequence, yielding 100% ASR. These measurements distinguish operational reachability from confirmed attacker control and verified harm. The vulnerabilities span command injection (49), file disclosure (27), file modification (22), network requests (five), and code injection (one), covering multiple forms of authority delegated to the agents. TrustProbe’s seed generation, scoring, and mutation consume 11.08 million tokens at a cost of \$2.19; target-agent inference is excluded.

Table 1: Verified vulnerabilities, execution time, and time to first exposure across agents.
<table><tr><td></td><td>OpenClaw</td><td>OpenCode</td><td>Hermes Agent</td><td>Pochi</td><td>Kimi Code CLI</td><td>Qwen Code</td><td>Cline</td><td>Pi Coding Agent</td><td>Mistral Vibe</td><td>DB-GPT</td><td>Agent Zero</td><td>Total</td></tr><tr><td>Stars (K)</td><td>388.9</td><td>204.5</td><td>241.7</td><td>0.1</td><td>7.3</td><td>27.7</td><td>67.5</td><td>102.0</td><td>4.9</td><td>19.9</td><td>19.1</td><td>1.08M</td></tr><tr><td>Verified Vulns.</td><td>24</td><td>17</td><td>16</td><td>13</td><td>12</td><td>8</td><td>7</td><td>3</td><td>2</td><td>1</td><td>1</td><td>104</td></tr><tr><td>Time Cost (h)</td><td>3.79</td><td>2.64</td><td>13.17</td><td>1.15</td><td>4.04</td><td>5.38</td><td>4.97</td><td>0.93</td><td>0.88</td><td>2.67</td><td>5.57</td><td>45.20</td></tr><tr><td>TTE (min)</td><td>10.32</td><td>42.78</td><td>12.70</td><td>11.07</td><td>70.62</td><td>23.65</td><td>92.27</td><td>23.73</td><td>26.70</td><td>2.00</td><td>255.25</td><td>23.73</td></tr></table>

We also replay the applicable vulnerabilities under the strictest usable non-interactive approval configurations (Appendices H and I). Eight agents expose configurable approval layers covering 89 of the 104 vulnerabilities, and 31 of these 89 (34.8%) remain exploitable. The surviving cases reveal two mechanisms: configured policies are bypassed by execution paths that never consult them, or policies omit the affected operations. Some other cases show explicit blocking, establishing successful enforcement in those runs. Approval therefore limits exploitation when the relevant operation is both covered and checked, while the remaining vulnerabilities expose gaps in enforcement and policy scope.

## 5.3 RQ2: Delivery Mechanisms

We test whether the 104 vulnerabilities verified during discovery reproduce when their skill bodies are delivered as direct prompts. The installed-skill counts come from the successful discovery runs. For replay, we remove the YAML frontmatter and place the remaining body verbatim in a neutral user message, with no SKILL.md installed. Other settings and the oracle criteria remain unchanged. For agent a, Table 2 reports the discovery-confirmed count $V _ { a } ^ { \mathrm { s k i l l } }$ and the prompt-reproduced count $V _ { a } ^ { \mathrm { p r o m p t } }$ . Their ratio, $\bar { \rho } _ { a } = V _ { a } ^ { \mathrm { p r o m p t } } / V _ { a } ^ { \mathrm { s k i l l } }$ , measures the fraction of this reference set reproduced in the direct-prompt replays. Direct-prompt delivery fails to reproduce 68.3% of the verified vulnerabilities. Paired traces link these losses to changed execution routes or sink arguments rather than explicit refusals. Source inspection helps explain the low surviving fractions in OpenClaw and Kimi Code CLI (Appendix G).

Table 2: Direct-prompt replay of vulnerabilities verified through installed skills during discovery.
<table><tr><td></td><td></td><td>OpenClaw OpenCode</td><td> $\begin{array} { c } { { \overline { { \mathrm { H e r m e s } } } } } \\ { { \overline { { \mathrm { A g e n t } } } } } \end{array}$ </td><td> $\mathrm { P o c h i }$ </td><td>Kimi Code CLI</td><td> $\begin{array} { c } { { \overline { { \mathrm { Q w e n } } } } } \\ { { \mathrm { C o d e } } } \end{array}$ </td><td> $\mathrm { c l i n e }$ </td><td>Pi Coding Agent</td><td> $\begin{array} { c } { \overline { { \mathrm { { M i s t r a l } } } } } \\ { \overline { { \mathrm { { V i b e } } } } } \end{array}$ </td><td>DB-GPT</td><td> $\begin{array} { c } { { \overline { { { \mathrm { A g e n t } } } } } } \\ { { \overline { { { \mathrm { Z e r o } } } } } } \end{array}$ </td><td>Total</td></tr><tr><td>Skill discovery</td><td>24</td><td>17</td><td>16</td><td>13</td><td>12</td><td>8</td><td>7</td><td>3</td><td>2</td><td>1</td><td>1</td><td>104</td></tr><tr><td>Prompt replay</td><td>1</td><td>7</td><td>6</td><td>9</td><td>1</td><td>4</td><td>2</td><td>2</td><td>0</td><td>0</td><td>1</td><td>33</td></tr><tr><td>Reproduction rate  $( \rho _ { a } )$ </td><td>4.2%</td><td>41.2%</td><td>37.5%</td><td>69.2%</td><td>8.3%</td><td>50.0% 28.6%</td><td></td><td>66.7%</td><td>0.0%</td><td>0.0%</td><td>100.0% 31.7%</td><td></td></tr></table>

OpenClaw illustrates framework-directed selection. It lists installed skills and instructs the agent to read and follow a matching entry. The framework thereby presents third-party content as guidance for the task. Direct delivery supplies the same body without this selection and retrieval process. Kimi Code CLI illustrates explicit activation. Invoking a registered skill causes the framework to load its instructions and direct the agent to follow them. The skill becomes the procedure designated for the current task; ordinary prompt delivery carries no corresponding activation signal. DB-GPT additionally incorporates the loaded skill into the agent’s configured instructions. Direct delivery leaves this configuration unchanged, showing another route through which skills guide execution.

In Pochi, ordinary prompts can still reach some of the same operations, consistent with its larger prompt-reproduction rate. These cases show how framework-directed selection, activation, and configuration position third-party skill content as guidance for actions under user-delegated authority. Overall, framework-mediated skill delivery reproduces substantially more verified vulnerabilities than direct-prompt delivery, showing that the delivery path itself materially shapes execution.

## 5.4 RQ3: Real-World Exposure

We evaluate 633 real SKILL.md files from SkillsMP, ClawHub, and official agent repositories. Each is installed on every compatible agent in a fresh session driven by a generic instruction. These skills carry no canary, so we derive markers from their paths, URLs, commands, file names, and environment tokens, filtering them by corpus frequency and an idle-agent baseline. A trigger requires the reached, tainted, and framed checks of equation 7, using these markers (Appendix J).

Of 2,963 skill-agent test runs, 743 (25.1%) trigger an audited vulnerable path, demonstrating that real-world skills exercise paths exposed by the synthetic campaign. This rate measures exposure through skill execution; it does not classify the original skills as malicious. Minimal payload edit convert 15 evaluated triggering skills into complete attacks. Observed outcomes include remote code execution, credential exfiltration (sending credentials to an attacker), OAuth phishing, and connections to attacker-controlled Model Context Protocol (MCP) servers. Together, path triggering and successful payload validation demonstrate that existing skill-mediated interactions can carry attacker-controlled content into operations exercised under the agent’s authority.

## 5.5 RQ4: Baseline Comparison

We compare TrustProbe with the static agent-code analyzer LLMSmith (Liu et al., 2024) and directed greybox fuzzer AgentFuzz (Liu et al., 2025). AgentFuzz’s released dataset specifies appli cation versions from before January 2025. Anthropic publicly introduced SKILL.md-based Agent Skills on October 16, 2025, and the format subsequently gained broader adoption across agent products. None of the versions in AgentFuzz’s original dataset natively supports loading and executing SKILL.md skills. Applying TrustProbe would therefore require adding a skill-loading mechanism, changing the evaluated systems and violating our threat model. We instead compare the methods on the skill-supporting versions used in our main evaluation. Both baselines support Python only, so the comparison covers all four Python agents in our evaluation: DB-GPT, Agent Zero, Hermes Agent, and Mistral Vibe.

Precision and recall in Table 3 are evaluated using the 20 dynamically verified vulnerabilities from these agents as the reference set. TP, FP, and FN denote true positives, false positives, and false negatives, respectively. Recall therefore measures recovery of this reference set rather than coverage of all vulnerabilities.

Table 3: Baseline comparison using the 20 verified vulnerabilities as the reference set.
<table><tr><td>Method</td><td>TP</td><td>FP</td><td>FN</td><td>Prec(%)</td><td>Recall(%)</td></tr><tr><td>LLMSmith</td><td>5</td><td>328</td><td>15</td><td>1.50</td><td>25.0</td></tr><tr><td>AgentFuzz</td><td>0</td><td>0</td><td>20</td><td>N/A</td><td>0.0</td></tr><tr><td>TrustProbe</td><td>20</td><td>0</td><td>0</td><td>100</td><td>100</td></tr></table>

LLMSmith’s missed vulnerabilities arise from sink-inventory mismatches: its modeled sinks are limited to eval, exec, and subprocess.run. Its recall therefore reflects sink coverage as well as analysis capability. We adjudicated the non-reference reports by manually inspecting every finding LLMSmith would itself report, together with a stratified sample of the remaining findings, and no inspected finding showed skill content controlling a sink argument. False positives constitute 98.5% of its reports, showing that static call-path reachability alone does not establish runtime control over a sink argument or observable harm.

Under its published settings, AgentFuzz exercises 674 deduplicated sink pools through approximately 2,425 attempts over 57 hours. We adapt AgentFuzz to the evaluated agents while retaining its published configuration, but obtain no verified findings in the completed valid runs. RQ2 shows that installed-skill delivery reproduces substantially more vulnerabilities than direct-prompt delivery. This gap exposes a limitation for prompt-based discovery in our setting: generated prompts must first induce the framework and model to realize the same execution path that installed skills reach natively. This difficulty is consistent with AgentFuzz’s lack of verified findings in our setting. These results illustrate the need to connect candidate paths to skill-driven execution and verified consequences.

## 5.6 RQ5: Ablation Study

We ablate three components while holding the remaining campaign settings and evaluation budgets fixed. GENERICSEED replaces semantic seed generation with a template omitting sink names, callpath semantics, and sink-directed inputs; canary injection, mutation, and scheduling remain. NOSIGMA removes semantic feedback from seed ranking and mutator selection. RANDOMSCHED replaces feedback-based selection with uniform sampling. Table 4 normalizes GENERICSEED by the full campaign’s 104 verified vulnerabilities, and the other variants by the 29 vulnerabilities found by mutation.

Table 4: Relative yield of ablations within their evaluated stages.
<table><tr><td>Configuration</td><td>Relative yield</td></tr><tr><td>GENERICSEED</td><td>44.2%</td></tr><tr><td>NOSIGMA</td><td>37.9%</td></tr><tr><td>RANDOMSCHED</td><td>31.0%</td></tr></table>

The full campaign’s initial seeds expose 75 vulnerabilities, while mutation adds 29. The reduced yield of GENERICSEED shows that translating an audited path into a plausible skill task improves discovery beyond template construction; subsequent mutation does not recover the full yield within the same budget. The reductions under NOSIGMA and RANDOMSCHED show that semantic feedback and adaptive seed selection each increase vulnerability yield during mutation. Because the variants evaluate different stages, their relative yields quantify contributions within each stage rather than ranking component importance.

## 6 Conclusion

Our study uncovers unsafe chains of trust in skill-based LLM agents, where reliance on attacker controlled skill content leads to harmful operations under user-delegated authority. TrustProbe combines source-to-sink analysis and directed greybox fuzzing with an oracle that confirms attackercontrolled flows and verifies observable harm. Across 11 agents, we identify 104 verified taint-style vulnerabilities spanning command execution, file disclosure and modification, network requests, and code injection. Direct-prompt replay reproduces only 31.7% of these vulnerabilities. The paired experiments and source inspection connect these failures to skill-delivery mechanisms and gaps in approval coverage and enforcement. Real-world skills exercise the identified vulnerable paths, and minimal payload edits produce complete attacks in 15 triggering skills selected for validation. These findings motivate preserving skill provenance through tool execution, declaring capabilities at installation, and enforcing argument- and path-aware controls at shared execution points. Securing skill-based agents therefore requires treating third-party skill content as part of the agent’s execution security boundary, not merely as auxiliary instructions.

## AI use statement

Generative AI was used in this work in two capacities. As a research instrument, the deepseekv4-flash model performs seed generation, semantic scoring, and mutation inside TrustProbe, as described in Sections 4 and 5.1. As writing aids, generative AI tools assisted with drafting and polishing the manuscript under author guidance and revision. All experimental numbers and vulnerability findings in this paper come from real campaign executions and are not AI-generated content. We have reviewed all AI-assisted text and take responsibility for the final content of this work, including text, claims, and artifacts produced with the aid of generative AI.

## Ethics statement

TrustProbe aims to improve the security of skill-based LLM agents by uncovering unsafe chains of trust. Its findings can help agent developers and framework maintainers identify how third-party skill content induces harmful operations under user-delegated authority, and inform the design of provenance tracking, capability controls, and approval enforcement. The method also presents dualuse risks. Automated discovery of these execution paths could reduce the effort required to construct malicious skills, potentially enabling unauthorized command execution, file manipulation, or data disclosure.

To limit these risks during evaluation, all agent executions took place in isolated local test workspaces. Synthetic campaigns used injected canaries, while real-skill validation used contentderived markers and locally modified skill copies. We verified effects using benign commands, scratch files, synthetic sensitive content, and controlled endpoints, including a decoy home directory and a local listener. No real user data or credentials were used as attack targets, and weaponized skills were never published to public skill marketplaces.

## Reproducibility statement

We document the revisions and execution configurations of all 11 evaluated agents in Appendix D. Appendix A provides the campaign pseudocode, Appendix E specifies the campaign parameters, Appendix F lists the sink models, and Appendix C provides the prompts for seed generation, scoring, and mutation. The target agents and these TrustProbe components use deepseek-v4-flash as the backend model. The oracle and its evidence requirements are described in Section 4.5. Appendices H and J document the restrictive approval configurations and the real-skill collection, adjudication, and weaponization procedures, respectively.

## References

Agent Zero. Agent Zero, 2026. URL https://github.com/agent0ai/agent-zero. Accessed September 26, 2026.

Anthropic. Agent Skills, 2025. URL https://platform.claude.com/docs/en/ agents-and-tools/agent-skills/overview. Accessed September 26, 2026.

Luca Beurer-Kellner, Beat Buesser, Ana-Maria Cret¸u, Edoardo Debenedetti, Daniel Dobos, Daniel Fabian, Marc Fischer, David Froelicher, Kathrin Grosse, Daniel Naeff, Ezinwanne Ozoani, Andrew Paverd, Florian Tramer, and V\` aclav Volhejn. Design patterns for securing LLM´ agents against prompt injections, 2025. URL https://arxiv.org/abs/2506.08837. arXiv:2506.08837.

Marcel Bohme, Van-Thuan Pham, Manh-Dung Nguyen, and Abhik Roychoudhury. Directed grey- ¨ box fuzzing. In Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security, pp. 2329–2344, 2017. doi: 10.1145/3133956.3134020.

Hongxu Chen, Yinxing Xue, Yuekang Li, Bihuan Chen, Xiaofei Xie, Xiuheng Wu, and Yang Liu. Hawkeye: Towards a desired directed grey-box fuzzer. In Proceedings of the 2018 ACM SIGSAC Conference on Computer and Communications Security, pp. 2095–2108, 2018. doi: 10.1145/3243734.3243849. URL https://chenbihuan.github.io/paper/ ccs18-chen-hawkeye.pdf.

Sizhe Chen, Julien Piet, Chawin Sitawarin, and David Wagner. StruQ: Defending against prompt injection with structured queries. In 34th USENIX Security Symposium (USENIX Security 25), pp. 2383–2400, 2025a. URL https://www.usenix.org/conference/ usenixsecurity25/presentation/chen-sizhe.

Sizhe Chen, Arman Zharmagambetov, Saeed Mahloujifar, Kamalika Chaudhuri, David Wagner, and Chuan Guo. SecAlign: Defending against prompt injection with preference optimization. In Proceedings of the 2025 ACM SIGSAC Conference on Computer and Communications Security, pp. 2833–2847, 2025b. doi: 10.1145/3719027.3744836.

Zhaorun Chen, Zhen Xiang, Chaowei Xiao, Dawn Song, and Bo Li. AgentPoison: Red-teaming LLM agents via poisoning memory or knowledge bases. In Advances in Neural Information Processing Systems, volume 37, pp. 130185–130213, 2024. doi: 10.52202/079017-4136. URL https://proceedings.neurips.cc/paper\_files/paper/2024/hash/ eb113910e9c3f6242541c1652e30dfd6-Abstract-Conference.html.

ClawHub. ClawHub, 2026. URL https://clawhub.ai/. Accessed September 26, 2026.

Cline. Cline, 2026. URL https://github.com/cline/cline. Accessed September 26, 2026.

DB-GPT. DB-GPT, 2026. URL https://github.com/eosphoros-ai/DB-GPT. Accessed September 26, 2026.

Edoardo Debenedetti, Jie Zhang, Mislav Balunovic, Luca Beurer-Kellner, Marc Fischer,´ and Florian Tramer. AgentDojo: A dynamic environment to evaluate prompt injec- \` tion attacks and defenses for LLM agents. In Advances in Neural Information Processing Systems, volume 37, pp. 82895–82920, 2024. doi: 10.52202/079017-2636. URL https://proceedings.nips.cc/paper\_files/paper/2024/ hash/97091a5177d8dc64b1da8bf3e1f6fb54-Abstract-Datasets\_and\_ Benchmarks\_Track.html. Datasets and Benchmarks Track.

Edoardo Debenedetti, Ilia Shumailov, Tianqi Fan, Jamie Hayes, Nicholas Carlini, Daniel Fabian, Christoph Kern, Chongyang Shi, Andreas Terzis, and Florian Tramer. Defeating prompt injections\` by design, 2025. URL https://arxiv.org/abs/2503.18813. arXiv:2503.18813.

Giuseppe Destefanis, Daniel Graziotin, Matteo Vaccargiu, and Marco Ortu. GitSkills: A dataset of agent skills on GitHub, 2026. URL https://arxiv.org/abs/2608.10906. arXiv:2608.10906.

Ruian Duan, Omar Alrawi, Ranjita Pai Kasturi, Ryan Elder, Brendan Saltaformaggio, and Wenke Lee. Towards measuring supply chain attacks on package managers for interpreted languages. In Network and Distributed System Security Symposium, 2021. doi: 10.14722/ndss. 2021.23055. URL https://www.ndss-symposium.org/wp-content/uploads/ 2021-055a-paper.pdf.

Ivan Evtimov, Arman Zharmagambetov, Aaron Grattafiori, Chuan Guo, and Kamalika Chaudhuri. WASP: Benchmarking web agent security against prompt injection attacks. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-0666. URL https://proceedings.nips.cc/paper\_files/paper/2025/ hash/1c9818387f5dd0a0bc151214660f059d-Abstract-Datasets\_and\_ Benchmarks\_Track.html. Datasets and Benchmarks Track.

Wenjie Fu, Xiaoting Qin, Jue Zhang, Qingwei Lin, Lukas Wutschitz, Robert Sim, Saravan Rajmohan, and Dongmei Zhang. CI-Work: Benchmarking contextual integrity in enterprise LLM agents. In Proceedings ofthe 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 6: Industry Track), pp. 1483–1508, 2026. doi: 10.18653/v1/2026.acl-industry.103. URL https://aclanthology.org/2026.acl-industry.103/.

Kai Greshake, Sahar Abdelnabi, Shailesh Mishra, Christoph Endres, Thorsten Holz, and Mario Fritz. Not what you’ve signed up for: Compromising real-world LLM-integrated applications with indirect prompt injection. In Proceedings of the 16th ACM Workshop on Artificial Intelligence and Security, pp. 79–90, 2023. doi: 10.1145/3605764.3623985.

Wenbo Guo, Chengwei Liu, Ming Kang, Yiran Zhang, Jiahui Wu, Zhengzi Xu, Vinay Sachidananda, and Yang Liu. Cutting the Gordian knot: Detecting malicious PyPI packages via a knowledgemining framework. In 35th USENIX Security Symposium (USENIX Security 26), pp. 1827– 1845, 2026a. URL https://www.usenix.org/conference/usenixsecurity26/ presentation/guo-wenbo.

Zihan Guo, Zhiyu Chen, Xiaohang Nie, Jianghao Lin, Yuanjian Zhou, and Weinan Zhang. Skill-Probe: Security auditing for emerging agent skill marketplaces via multi-agent collaboration, 2026b. URL https://arxiv.org/abs/2603.21019. arXiv:2603.21019.

Norm Hardy. The confused deputy (or why capabilities might have been invented). ACM SIGOPS Operating Systems Review, 22(4):36–38, 1988. doi: 10.1145/54289.871709.

Hermes Agent. Hermes Agent, 2026. URL https://github.com/NousResearch/ hermes-agent. Accessed September 26, 2026.

Heqing Huang, Yiyuan Guo, Qingkai Shi, Peisen Yao, Rongxin Wu, and Charles Zhang. BEACON: Directed grey-box fuzzing with provable path pruning. In IEEE Symposium on Security and Privacy, pp. 36–50, 2022. doi: 10.1109/SP46214.2022.9833751.

Ruixuan Huang, Xunguang Wang, Zongjie Li, Daoyuan Wu, and Shuai Wang. GuidedBench: Measuring and mitigating the evaluation discrepancies of in-the-wild LLM jailbreak methods. In International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=ZVg8y3ibyM.

Xiaojun Jia, Jie Liao, Simeng Qin, Jindong Gu, Wenqi Ren, Xiaochun Cao, Yang Liu, and Philip Torr. SkillJect: Effectively automating skill-based prompt injection for skill-enabled agents, 2026. URL https://arxiv.org/abs/2602.14211. arXiv:2602.14211.

Kimi Code CLI. Kimi Code CLI, 2026. URL https://github.com/MoonshotAI/ kimi-code. Accessed September 26, 2026.

Vincent Koc, Patrick Erichsen, Jacob Tomlinson, Agustin Rivera, Michael Appel, and Nir Paz. ClawHub security signals: When VirusTotal, static analysis, and SkillSpector disagree, 2026. URL https://arxiv.org/abs/2606.01494. arXiv:2606.01494.

Fengyu Liu, Yuan Zhang, Jiaqi Luo, Jiarun Dai, Tian Chen, Letian Yuan, Zhengmin Yu, Youkun Shi, Ke Li, Chengyuan Zhou, Hao Chen, and Min Yang. Make agent defeat agent: Automatic detection of taint-style vulnerabilities in LLM-based agents. In 34th USENIX Security Symposium (USENIX Security 25), pp. 3767–3786, 2025. URL https://www.usenix.org/ conference/usenixsecurity25/presentation/liu-fengyu.

Tong Liu, Zizhuang Deng, Guozhu Meng, Yuekang Li, and Kai Chen. Demystifying RCE vulnerabilities in LLM-integrated apps. In Proceedings of the 2024 ACM SIGSAC Conference on Computer and Communications Security, pp. 1716–1730, 2024. doi: 10.1145/3658644.3690338.

Yi Liu, Zhihao Chen, Yanjun Zhang, Gelei Deng, Yuekang Li, Jianting Ning, and Leo Yu Zhang. “Do not mention this to the user”: Detecting and understanding malicious agent skills in the wild. In 35th USENIX Security Symposium (USENIX Security 26), pp. 1727–1746. USENIX Association, 2026. URL https://www.usenix.org/conference/usenixsecurity26/ presentation/liu-yi.

Niloofar Mireshghallah, Hyunwoo Kim, Xuhui Zhou, Yulia Tsvetkov, Maarten Sap, Reza Shokri, and Yejin Choi. Can LLMs keep a secret? Testing privacy implications of language models via contextual integrity theory. In International Conference on Learning Representations, pp. 1892– 1915, 2024. URL https://openreview.net/forum?id=gmg7t8b4s0.

Niloofar Mireshghallah, Neal Mangaokar, Narine Kokhlikyan, Arman Zharmagambetov, Manzil Zaheer, Saeed Mahloujifar, and Kamalika Chaudhuri. CIMemories: A compositional benchmark for contextual integrity in LLMs. In International Conference on Learning Representations, pp. 95513–95532, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/ 2026/hash/9a2bcfaf383638e166162a25b6dff125-Abstract-Conference. html.

Mistral Vibe. Mistral Vibe, 2026. URL https://github.com/mistralai/ mistral-vibe. Accessed September 26, 2026.

MITRE. CWE-441: Unintended proxy or intermediary (‘confused deputy’), 2025. URL https: //cwe.mitre.org/data/definitions/441.html. Accessed September 26, 2026.

James Newsome and Dawn Song. Dynamic taint analysis for automatic detection, analysis, and signature generation of exploits on commodity software. In Network and Distributed System Security Symposium, 2005. URL https://bitblaze.cs.berkeley.edu/papers/ taintcheck.pdf.

Marc Ohm, Henrik Plate, Arnold Sykosch, and Michael Meier. Backstabber’s knife collection: A review of open source software supply chain attacks. In International Conference on Detection of Intrusions and Malware, and Vulnerability Assessment (DIMVA), pp. 23–43, 2020. doi: 10.1007/ 978-3-030-52683-2 2.

OpenClaw. OpenClaw, 2026. URL https://github.com/openclaw/openclaw. Accessed September 26, 2026.

OpenCode. OpenCode, 2026. URL https://github.com/anomalyco/opencode. Accessed September 26, 2026.

Yipeng Ouyang, Yi Xiao, Yuhao Gu, and Xianwei Zhang. SkCC: Portable and secure skill compilation for cross-framework LLM agents, 2026. URL https://arxiv.org/abs/2605. 03353. arXiv:2605.03353.

OWASP Foundation. OWASP Top 10:2025, 2025. URL https://top10.owasp.org/ 2025/. Accessed September 26, 2026.

Pi Coding Agent. Pi Coding Agent, 2026. URL https://github.com/earendil-works/ pi. Accessed September 26, 2026.

Pochi. Pochi, 2026. URL https://github.com/TabbyML/pochi. Accessed September 26, 2026.

Yujia Qin, Shihao Liang, Yining Ye, Kunlun Zhu, Lan Yan, Yaxi Lu, Yankai Lin, Xin Cong, Xiangru Tang, Bill Qian, Sihan Zhao, Lauren Hong, Runchu Tian, Ruobing Xie, Jie Zhou, Mark Gerstein, Dahai Li, Zhiyuan Liu, and Maosong Sun. ToolLLM: Facilitating large language models to master 16000+ real-world APIs. In International Conference on Learning Representations, pp. 9695– 9717, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/ hash/28e50ee5b72e90b50e7196fde8ea260e-Abstract-Conference.html.

Qwen Code. Qwen Code, 2026. URL https://github.com/QwenLM/qwen-code. Accessed September 26, 2026.

Yangjun Ruan, Honghua Dong, Andrew Wang, Silviu Pitis, Yongchao Zhou, Jimmy Ba, Yann Dubois, Chris J. Maddison, and Tatsunori Hashimoto. Identifying the risks of LM agents with an LM-emulated sandbox. In International Conference on Learning Representations, pp. 27031– 27098, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/ hash/7274ed909a312d4d869cc328ad1c5f04-Abstract-Conference.html.

Timo Schick, Jane Dwivedi-Yu, Roberto Dess\`ı, Roberta Raileanu, Maria Lomeli, Eric Hambro, Luke Zettlemoyer, Nicola Cancedda, and Thomas Scialom. Toolformer: Language models can teach themselves to use tools. In Advances in Neural Information Processing Systems, volume 36, pp. 68539–68551, 2023. doi: 10.52202/ 075280-2997. URL https://proceedings.neurips.cc/paper/2023/hash/ d842425e4bf79ba039352da0f658a906-Abstract-Conference.html.

David Schmotz, Luca Beurer-Kellner, Sahar Abdelnabi, and Maksym Andriushchenko. Skill-Inject: Measuring agent vulnerability to skill file attacks, 2026. URL https://arxiv.org/abs/ 2602.20156. arXiv:2602.20156.

Sander Schulhoff, Jeremy Pinto, Anaum Khan, Louis-Franc¸ois Bouchard, Chenglei Si, Svetlina Anati, Valen Tagliabue, Anson Kost, Christopher Carnahan, and Jordan Boyd-Graber. Ignore this title and HackAPrompt: Exposing systemic vulnerabilities of LLMs through a global prompt hacking competition. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 4945–4977, 2023. doi: 10.18653/v1/2023.emnlp-main.302. URL https://aclanthology.org/2023.emnlp-main.302/.

Edward J. Schwartz, Thanassis Avgerinos, and David Brumley. All you ever wanted to know about dynamic taint analysis and forward symbolic execution (but might have been afraid to ask). In IEEE Symposium on Security and Privacy, pp. 317–331, 2010. doi: 10.1109/SP.2010.26. URL https://edmcman.github.io/bib/schwartz\_2010\_ dynamic-abstract.html.

Yijia Shao, Tianshi Li, Weiyan Shi, Yanchen Liu, and Diyi Yang. PrivacyLens: Evaluating privacy norm awareness of language models in action. In Advances in Neural Information Processing Systems, volume 37, pp. 89373–89407, 2024. doi: 10.52202/079017-2837. URL https://papers.nips.cc/paper\_files/paper/ 2024/hash/a2a7e58309d5190082390ff10ff3b2b8-Abstract-Datasets\_ and\_Benchmarks\_Track.html. Datasets and Benchmarks Track.

Jiawen Shi, Zenghui Yuan, Guiyao Tie, Pan Zhou, Neil Zhenqiang Gong, and Lichao Sun. Prompt injection attack to tool selection in LLM agents, 2025. URL https://arxiv.org/abs/ 2504.19793. arXiv:2504.19793.

SkillsMP. SkillsMP, 2026. URL https://skillsmp.com/. Accessed September 26, 2026.

Sam Toyer, Olivia Watkins, Ethan Mendes, Justin Svegliato, Luke Bailey, Tiffany Wang, Isaac Ong, Karim Elmaaroufi, Pieter Abbeel, Trevor Darrell, Alan Ritter, and Stuart Russell. Tensor Trust: Interpretable prompt injection attacks from an online game. In International Conference on Learning Representations, pp. 18714–18746, 2024. URL https://proceedings.iclr.cc/paper\_files/paper/2024/hash/ 519c51529c3544b3430bd8b17d400365-Abstract-Conference.html.

Xingyao Wang, Yangyi Chen, Lifan Yuan, Yizhe Zhang, Yunzhu Li, Hao Peng, and Heng Ji. Executable code actions elicit better LLM agents. In Proceedings ofthe 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pp. 50208– 50232, 2024. URL https://proceedings.mlr.press/v235/wang24h.html.

Fabian Yamaguchi, Nico Golde, Daniel Arp, and Konrad Rieck. Modeling and discovering vulnerabilities with code property graphs. In IEEE Symposium on Security and Privacy, pp. 590–604, 2014. doi: 10.1109/SP.2014.44.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. ReAct: Synergizing reasoning and acting in language models. In International Conference on Learning Representations, 2023. URL https://arxiv.org/abs/2210.03629.

Jingwei Yi, Yueqi Xie, Bin Zhu, Emre Kiciman, Guangzhong Sun, Xing Xie, and Fangzhao Wu. Benchmarking and defending against indirect prompt injection attacks on large language models. In Proceedings ofthe 31st ACM SIGKDD Conference on Knowledge Discovery and Data Mining V.1, pp. 1809–1820, 2025. doi: 10.1145/3690624.3709179.

Qiusi Zhan, Zhixiang Liang, Zifan Ying, and Daniel Kang. InjecAgent: Benchmarking indirect prompt injections in tool-integrated large language model agents. In Findings of the Association for Computational Linguistics: ACL 2024, pp. 10471–10506, 2024. doi: 10.18653/v1/2024. findings-acl.624. URL https://aclanthology.org/2024.findings-acl.624/.

Qiusi Zhan, Richard Fang, Henil Shalin Panchal, and Daniel Kang. Adaptive attacks break defenses against indirect prompt injection attacks on LLM agents. In Findings of the Association for Computational Linguistics: NAACL 2025, pp. 7116–7132, 2025. doi: 10.18653/v1/2025. findings-naacl.395. URL https://aclanthology.org/2025.findings-naacl. 395/.

Hanrong Zhang, Jingyuan Huang, Kai Mei, Yifei Yao, Zhenting Wang, Chenlu Zhan, Hongwei Wang, and Yongfeng Zhang. Agent Security Bench (ASB): Formalizing and benchmarking attacks and defenses in LLM-based agents. In International Conference on Learning Representations, pp. 35331–35366, 2025. URL https://openreview.net/forum?id= xPwqg0B659.

Markus Zimmermann, Cristian-Alexandru Staicu, Cam Tenny, and Michael Pradel. Small world with high risks: A study of security threats in the npm ecosystem. In 28th USENIX Security Symposium (USENIX Security 19), pp. 995–1010, 2019. URL https://www.usenix.org/ conference/usenixsecurity19/presentation/zimmerman.

Andy Zou, Maxwell Lin, Eliot Jones, Micha Nowak, Mateusz Dziemian, Nick Winter, Valent Nathanael, Ayla Croft, Xander Davies, Jai Patel, Robert Kirk, Yarin Gal, Dan Hendrycks, Zico Kolter, and Matt Fredrikson. Security challenges in AI agent deployment: Insights from a large scale public competition. In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-2676. URL https://proceedings.nips.cc/paper\_files/paper/2025/ hash/73368bc7644c054b5bcc6490a8f2fb1c-Abstract-Datasets\_and\_ Benchmarks\_Track.html. Datasets and Benchmarks Track.

## A Campaign Algorithm

Algorithm 1 gives the full pseudocode of the campaign summarized in §4. The per-pool budget $B = \left( R _ { \mathrm { m a x } } , \tau _ { \mathrm { m a x } } \right)$ limits the number of rounds and elapsed time; $r _ { j }$ and $\tau _ { j }$ track these quantities for pool $P _ { j }$

Algorithm 1 The TrustProbe campaign   
Input: Agent code A, sink list K, per-pool budget $B = ( R _ { \mathrm { m a x } } , \tau _ { \mathrm { m a x } } )$   
Output: Taint flows T, exploitable vulnerabilities V   
1: $T \gets \emptyset ; \quad V \gets \emptyset ;$ paths ← AuditPaths(A, K)   
2: $P _ { j }  \emptyset$ for each sink $j \in \{ \sinh ( c ) : c \in \mathrm { p a t h s } \}$   
3: for each path c in paths do   
4: canary $ H ( \operatorname { i d } _ { t } \parallel \operatorname { i d } _ { v } ) ;$ s<sub>0</sub> ← GenSeed(c)   
5: $P _ { \mathrm { s i n k } ( c ) }  P _ { \mathrm { s i n k } ( c ) } \cup \{ s _ { 0 } \}$   
6: end for   
7: for each pool $P _ { j }$ with budget B do   
8: $r _ { j } \gets 0 ; ~ \tau _ { j } \gets 0 ; ~ t _ { j } ^ { 0 } \gets \mathrm { C l o c k } ( )$   
9: while $P _ { j } \ne \emptyset \land r _ { j } < \overline { { R } } _ { \operatorname* { m a x } } \land \tau _ { j } < \tau _ { \operatorname* { m a x } }$ do   
10: $s ^ { * } \gets \mathrm { a r g m a x } _ { s \in P _ { i } } F ( s ) ; c \gets \mathrm { p a t h } ( s ^ { * } )$   
11: tr ← RunSession(s<sup>∗</sup>)   
12: update $\sigma , \delta , c _ { \mathrm { s } } , c _ { \mathrm { c } }$ from tr   
13: if reached(c) ∧ tainted $. ( c ) \wedge$ framed(c) then   
14: $T \gets T \cup \{ c \} ; V \gets V \cup \{ ( c , s ^ { * } ) \}$ if Harm(tr)   
15: $P _ { j }  P _ { j } \setminus \{ s : \mathrm { p a t h } ( s ) = c \}$   
16: else   
17: s<sup>′</sup> ← Mutate(RuleSelect(σ, δ, stage(tr)), s<sup>∗</sup>, tr); $P _ { j }  P _ { j } \cup \{ s ^ { \prime } \}$   
18: end if   
19: r<sub>j</sub> ← r<sub>j</sub> + 1; τ<sub>j</sub> ← Clock() − t<sup>0</sup><sub>j</sub>   
20: end while   
21: end for   
22: return T, V

## B Sanitizer Audit Across All Agents

To examine validation along the paths characterized in §3.1, we audited all 11 agents for contentlevel filtering, sanitization, and origin tracking on the SKILL.md code path. The audit searched each codebase for sanitization, validation, escaping, allowlisting, permission, provenance, and origin-tracking logic along the entire route from skill file loading through context injection to downstream tool-call execution.

Finding. Once loaded, the SKILL.md body text propagates along execution paths to securitysensitive operations without adequate filtering and provenance marking, and no runtime mechanism distinguishes skill-originated tool-call arguments from user-originated ones. Four agents escape metadata fields (name and description) in system-prompt listings to prevent XML tag breakout, which is a structural protection for the prompt format rather than a content filter. Two agents apply install-time scanners, but OpenClaw’s scanner ignores SKILL.md entirely and Hermes’ scanner applies only lexical pattern rules to the body, and both are bypassable or disabled by default for agent-created skills. Three agents apply pattern-based checks on terminal command strings, yet these are origin-blind in that they evaluate the command text without knowing whether it was suggested by skill content or by the user.

![](images/14197edbdff3e3f2756aaa30c86504067ae95a0840066bb1ba6b70fb6eceba28.jpg)  
Figure 3: The seed-generation prompt.

![](images/efbefffd498dcbd762d80fbe6bcbf0e2cf466091603d6c6e4cb338189b80fd4e.jpg)  
Figure 4: The semantic-scoring (σ) prompt.

![](images/ab08f072380a61192fac75dd937a24e829be316e4d11089f26d9bb6b634f9d38.jpg)  
Figure 5: The Semantic Mutator prompt.

![](images/7ff7abec871e291c4cf038cec6593140f0f76fd73d35f6379e48d5a10f105beb.jpg)  
Figure 6: The Metadata Mutator prompt.

![](images/5f381551f99b697696650c1a8d1b4fee98bda2722858773417d8786b162c5a39.jpg)  
Figure 7: The Body Mutator prompt.

![](images/861e985dad91ae882855cb0c85501da46c65221ce237635e89c59770aa86622d.jpg)  
Figure 8: The Sink-Argument Mutator prompt.

## D Evaluated Agents and Revisions

Table 5 identifies each subject by its full product name, GitHub repository, released or packaged version, and the exact source revision used in our audit and experiments. Revision values are 12- character prefixes of the corresponding Git commit identifiers. Where a repository contained several packages, the version is that of the tested agent or CLI rather than an unrelated workspace package.

Hermes Agent, DB-GPT, Agent Zero, and Mistral Vibe are implemented in Python. OpenClaw, OpenCode, Pi Coding Agent, Cline, Qwen Code, Kimi Code CLI, and Pochi are implemented in TypeScript.

Table 5: GitHub repositories, versions, and tested revisions of the 11 evaluated agents.
<table><tr><td>Agent</td><td>GitHub repository</td><td>Version</td><td>Commit</td></tr><tr><td>OpenClaw</td><td>https://github.com/openclaw/openclaw</td><td>2026.5.6</td><td>b70a2451f8c9</td></tr><tr><td>Hermes Agent</td><td>https://github.com/NousResearch/</td><td>0.19.0</td><td>8fc278207b0f</td></tr><tr><td></td><td>hermes-agent</td><td></td><td>8a03fc265b6d</td></tr><tr><td>OpenCode Pi Coding Agent</td><td>https://github.com/sst/opencode</td><td>1.17.18 0.82.1</td><td>b4f293684bba</td></tr><tr><td>Cline</td><td>https://github.com/earendil-works/pi</td><td>3.0.47</td><td>7d63376d9824</td></tr><tr><td>Qwen Code</td><td>https://github.com/cline/cline</td><td></td><td></td></tr><tr><td>DB-GPT</td><td>https://github.com/QwenLM/qwen-code</td><td>0.21.0</td><td>58fa6cf85b99</td></tr><tr><td>Agent Zero</td><td>https://github.com/eosphoros-ai/DB-GPT</td><td>0.8.1</td><td>7996544a4375</td></tr><tr><td>Kimi Code CLI</td><td>https://github.com/agent0ai/agent-zero</td><td>2.4 0.29.2</td><td>fddcc3deea3d</td></tr><tr><td>Mistral Vibe</td><td>https://github.com/MoonshotAI/kimi-code https://github.com/mistralai/</td><td>2.23.2</td><td>8a45f10eddbb</td></tr><tr><td></td><td>mistral-vibe</td><td></td><td>99a6efa9ca1f</td></tr><tr><td>Pochi</td><td>https://github.com/TabbyML/pochi</td><td>0.6.0-dev</td><td>160e41ce429e</td></tr></table>

Table 6 records the native skill-loading mechanisms and execution configurations used for RQ1. We used each agent’s native loading path rather than injecting the skill body directly into the user prompt. Permission-capable agents used their most permissive non-interactive mode; agents without an approval layer used their normal headless entry point. None of the evaluated agents enables a sandbox by default, and this unsandboxed state is also the one most commonly used in practice. All executions were therefore unsandboxed except DB-GPT’s native WebAssembly (Wasm) isolation for code execution.

Table 6: Native skill-loading mechanisms and execution configurations in the main campaign.  
Agent Skill-loading mechanism Execution configuration   
OpenClaw Workspace skills/<name>/SKILL.md Fresh configuration; no approval override   
Hermes Agent Isolated HERMES HOME/skills Automatic approval; headless one-shot   
OpenCode Per-run skills.paths directory --dangerously-skip  
permissions   
Pi Coding Agent Explicit --skill loading Print/headless mode; no approval gate   
Cline Workspace .cline/skills --auto-approve true   
Qwen Code Workspace .qwen/skills; explicit invoca- --yolo   
tion   
DB-GPT Isolated FileBasedSkill binding Direct framework invocation; native Wasm   
retained   
Agent Zero Repository skills/<name>/SKILL.md Direct headless invocation; no approval gate   
Kimi Code CLI Isolated --skills-dir directory Headless automatic approval   
Mistral Vibe Isolated skill paths directory --auto-approve --trust   
Pochi Workspace .pochi/skills No built-in approval gate

## E Campaign Parameters

Table 7 records the parameters shared by all subjects in RQ1. The selection penalty is intentionally unclipped, so with the reported weights it equals the nonnegative sum of the seed- and path-selection counts, and repeatedly selected candidates gradually yield priority to alternatives. No per-agent parameter tuning was performed.

## F Sink List

Following the presentation of AgentFuzz’s sink list (Liu et al., 2025), Tables 8 and 9 group the sink models used by our source-to-sink analysis. The Python profile contains the 47 callees from AgentFuzz after expanding its os.exec<sub>\*</sub> and os.spawn<sub>\*</sub> families, plus 22 models needed for asynchronous execution, modern file APIs, archive extraction, HTTP clients, and deserialization. The TypeScript profile contains 115 qualified names and accepted aliases covering Node.js, IDEagent, LLM-runtime, and tool-dispatch APIs. All agents implemented in the same language share the same inventory, and project-specific wrapper functions are retained only as intermediate nodes in source-to-sink call paths, never as sinks. For brevity, each row merges qualified and unqualified aliases, and an asterisk denotes the exact variants enumerated in the manifest. Resolving these models in each codebase and grouping the retained source-to-sink call paths produces the 202 sinkfunction pools reported in Section 5.2.

Table 7: Campaign parameters shared by all subjects in the main campaign.
<table><tr><td>Parameter</td><td>Main-experiment setting</td></tr><tr><td>Framework and agent backend</td><td>deepseek-v4-flash</td></tr><tr><td>Initialization</td><td>One generated seed per audited source-to-sink call path</td></tr><tr><td>Pooling and budget</td><td>One pool per sink function with eight mutation rounds or a 15-minute cap, whichever comes first</td></tr><tr><td>Scheduler weights</td><td> $\alpha = \beta = \gamma = \eta = 1$  and distance exponent  $k = 1$ </td></tr><tr><td>Score ranges</td><td>Semantic score  $\sigma \in [ 0 , 1 0 ]$  and distance score  $\delta \in [ 0 , 1 0 ]$ </td></tr><tr><td>Mutation thresholds</td><td> $\sigma < 6$  triggers semantic mutation and  $\delta < 8$  triggers stage-directed mutation after the semantic rule</td></tr><tr><td>Selection penalty</td><td> $\pi = c _ { \mathrm { s } } + c _ { \mathrm { c } } ,$  nonnegative and not clipped</td></tr><tr><td>Execution isolation</td><td>Fresh agent session and isolated scratch workspace for every trial</td></tr></table>

CMDi and CODEi denote command and code injection, PATHi covers attacker-directed filesystem paths, SSRF (server-side request forgery) denotes attacker-directed network requests, Deser. denotes unsafe deserialization, SSTI denotes server-side template injection, and Proto. denotes prototype pollution. The remaining labels identify archive extraction, IDE-mediated actions, downstream prompt construction, and generic tool dispatch. These tables define candidates for source-tosink analysis rather than vulnerabilities by themselves, and TrustProbe reports a finding only after a retained source-to-sink call path passes both taint confirmation and exploitability verification.

Table 8: Python sink inventory used by the source-to-sink analysis, grouped by package.
<table><tr><td>Package</td><td>Class</td><td>Methods</td><td>Type</td></tr><tr><td>subprocess, os</td><td></td><td>run, Popen, exec*, spawn*</td><td>CMDi</td></tr><tr><td>asyncio</td><td></td><td>create_subprocess_shell,</td><td>CMDi</td></tr><tr><td>builtins</td><td></td><td>create_subprocess_exec eval, exec, compile</td><td>CODEi</td></tr><tr><td>shutil,pathlib</td><td>Path</td><td>open, copy, move</td><td>PATHi</td></tr><tr><td>requests, httpx, aiohttp, -, Session</td><td></td><td>get, post, request, urlopen</td><td>SSRF</td></tr><tr><td>urllib yaml,pickle</td><td></td><td></td><td></td></tr><tr><td>jinja2</td><td>Environment</td><td>load, loads from_string</td><td>Deser. SSTI</td></tr><tr><td>sqlite3, sqlalchemy</td><td>Cursor,</td><td>execute</td><td>SQLi</td></tr><tr><td>tarfile,zipfile</td><td>Session</td><td>extract,extractall</td><td>Archive</td></tr></table>

Table 9: TypeScript sink inventory used by the source-to-sink analysis, grouped by package or runtime.
<table><tr><td>Package or runtime</td><td>Class</td><td>Methods</td><td>Type</td></tr><tr><td>Node.js</td><td>child-process</td><td>exec, spawn, fork</td><td>CMDi</td></tr><tr><td>JavaScript</td><td>builtins, vm</td><td>eval,Function, runInThisContext</td><td>CODEi</td></tr><tr><td>Node.js</td><td>fs</td><td>readFile,writeFile, copyFile</td><td>PATHi</td></tr><tr><td>Web and Node.js</td><td>fetch,axios</td><td>get,post, request</td><td>SSRF</td></tr><tr><td>js-yaml,v8</td><td></td><td>load, deserialize</td><td>Deser.</td></tr><tr><td>Object, lodash</td><td></td><td>assign, merge</td><td>Proto.</td></tr><tr><td>tar,extract-zip</td><td></td><td>extract,Extract</td><td>Archive</td></tr><tr><td>VS Code, Electron</td><td>terminal, webContents</td><td>createTerminal,executeJavaScript</td><td>IDE</td></tr><tr><td>LLM SDKs</td><td>OpenAI, Anthropic</td><td>chat.completions.create,</td><td>LLM</td></tr><tr><td>Agent runtimes</td><td>MCP dispatchers</td><td>messages.create callTool</td><td>Tool dis-</td></tr></table>

## G Skill and Direct-Prompt Execution Paths

We inspected the exact revisions in Table 5 to determine what changes when identical attackerauthored content is delivered through an installed skill rather than a direct user message. Here, trust is an operational property of the framework path, not a claim about a model’s internal state, and a skill receives framework-generated discovery metadata, invocation instructions, a typed activation event or tool result, a bound prompt, or some combination of these signals. In the direct-prompt replays of RQ2, the body enters only through the ordinary user-message interface and none of those skill-specific transformations occurs. The source paths fall into three recurring designs, namely catalog-and-retrieval paths (§G.1), explicit activation paths (§G.2), and framework-bound prompt paths (§G.3).

## G.1 Catalog-and-Retrieval Paths

OpenClaw makes the provenance difference especially explicit. Its workspace scanner discovers eligible skill directories and constructs a model-facing available skills catalog containing each name, description, and location. The system prompt places this catalog under “Skills (mandatory),” requires the model to scan it before replying, and instructs the model to read the exact file and follow it when a task matches. An installed skill therefore travels through discovery, system-level routing, and file retrieval before its body influences tool selection. The direct prompt contains the same body but creates neither a catalog entry nor the mandatory read-and-follow route. This strong difference is consistent with only 1 of 24 vulnerabilities being reproduced in direct-prompt replay.

Hermes Agent uses a related two-stage design. build skills system prompt scans the configured skill roots and emits a mandatory index, and when an entry is relevant the system prompt requires skill view(name) to load it. That tool returns a structured object containing the body, path, related files, and readiness information, and records the loaded artifact for subsequent use. A direct prompt is absent from both the index and the load result, although it can still drive ordinary tools directly. The framework tells the model to load and follow installed skills, while direct prompts can still reach some of the same tools. This combination is consistent with 6 of 16 vulnerabilities being reproduced in direct-prompt replay.

OpenCode similarly separates discovery from full-body retrieval. Skill.fmt contributes an installed-skill index to the system context, the system instructions tell the model to use the dedicated skill tool when a task matches, and that tool returns the selected body in a tagged skill content block together with its base directory and bundled files. The direct-prompt replay bypasses this catalog, tool, and content sequence. Nevertheless, because both routes ultimately expose many of the same file, shell, and network tools, 7 of 17 vulnerabilities were reproduced through direct prompts.

Pi Coding Agent provides a lighter-weight variant. The --skill option registers an explicitly supplied skill, and its system prompt lists the registered name, description, and location while directing the model to use the read tool to load the body. Direct delivery omits registration and retrieval but leaves the general-purpose tool surface unchanged. Two of its three vulnerabilities were reproduced, suggesting that its skill path adds a useful routing cue without forming a strong execution boundary.

Agent Zero combines retrieval with conversational persistence, and skills tool first lists or searches installed skills and then returns the selected body from its load action as a tool result carrying skill instructions metadata. The framework also stores the loaded skill name in chat-wide state and reattaches a missing body after context compaction. Plain prompt content creates neither this metadata nor the reattachment ledger. The single Agent Zero vulnerability was reproduced through direct prompts because the relevant execution tool remains directly reachable, and with one case this result establishes route equivalence for that case rather than a general absence of skill-specific trust.

## G.2 Explicit Activation Paths

Kimi Code CLI turns a skill invocation into a typed event rather than forwarding the slash text as an ordinary message. The configured --skills-dir populates a registry. The /skill:<name> command resolves through the skill command map and calls Session.activateSkill. Then SkillManager.activate renders the registered body inside a kimi-skill-loaded block and records a skill activation origin. It starts the turn with an instruction to follow the loaded skill. A direct prompt instead calls the ordinary prompt method, without registry resolution, the activation origin, or the loaded-skill wrapper. Only 1 of 12 vulnerabilities was reproduced in direct-prompt replay.

Mistral Vibe joins an automatic catalog with an explicit invocation path. Skills discovered from skill paths appear in the system prompt with instructions to use the skill tool when a description matches. An exact slash invocation is parsed before the normal turn, and the loop inserts a synthetic skill-tool call and a tool result containing the full body in a tagged skill content block. The plain prompt receives neither the synthetic tool exchange nor the active-instruction signal. Neither of the two Mistral Vibe vulnerabilities was reproduced in direct-prompt replay.

Pochi performs an analogous transformation in the user-message pipeline. replaceSlash CommandReferences recognizes an installed skill name and rewrites the slash reference into a structured skill tag that explicitly tells the model to call useSkill. The tool then returns the stored instructions and, when declared, an allowed-tool restriction. A direct prompt bypasses both the structured reference and the retrieval tool. Even so, 9 of 13 vulnerabilities were reproduced through direct prompts because Pochi’s ordinary message path can still lead to the same generalpurpose operations, so the activation protocol strengthens routing without being necessary for most of the observed cases.

Qwen Code registers discovered user and project skills as model-invocable commands. Explicit invocation resolves the registered command, loads its full body as a submit prompt payload, and can carry the skill’s declared tool configuration, while the alternative model-invoked path advertises available entries and instructs the model to call the skill tool first when one is relevant. The RQ2 direct-prompt replay is neither a registered command nor a skill-tool result. Four of eight vulnerabilities were reproduced, placing Qwen Code between designs whose control flow is largely dependent on skill-specific loading and activation and those whose ordinary prompt path reaches the same sinks directly.

Cline discovers skill metadata from project and global skill directories and exposes that catalog to the model while deferring full instructions until activation. In our execution path, the slash command identifies the installed entry and the skill executor injects its full instructions into the turn, whereas the direct-prompt replay is processed only as a user message and carries no discovered-skill identity. Two of seven vulnerabilities were reproduced, indicating that the activation route materially affects which tool path and sink arguments the agent constructs even though the underlying tools are still available.

## G.3 Framework-Bound Prompt Paths

DB-GPT does not merely advertise or retrieve the test skill. FileBasedSkill parses the frontmatter and body into a framework object. Then ConversableAgent.bind converts that object into the common skill representation and adopts its prompt template as the agent’s bound prompt. The installed instructions thus become part of the agent’s configuration during construction. The direct-prompt replay constructs no skill object and leaves the agent without that bound prompt. Its one verified vulnerability was not reproduced under direct-prompt delivery. Although the sample is too small for a broad quantitative claim about DB-GPT, the code establishes a qualitative boundary in which identical text occupies the framework-bound prompt path only when it is packaged and loaded as a skill.

Across all three designs, the skill-delivery mechanism changes more than formatting. Catalogs determine which artifact the model considers relevant, activation protocols create explicit followthis-skill events or tool results, and binding can place the body in an agent-level prompt before the user message is processed. These mechanisms affect both willingness, by presenting thirdparty instructions as a framework-selected capability, and ability, by routing fields and resources through code that a plain message never traverses. These mechanisms give installed skills discovery, activation, and prompt-binding context that is absent from direct user messages.

## H Approval-Control Configurations

Table 10 gives the exact restrictive configuration paired with RQ1. In non-interactive execution, ask-mode requests are rejected or left pending rather than approved. Hermes Agent and Kimi Code CLI do not expose a usable ask-per-action mode on the tested headless path, so we use their userconfigurable deny rules. Pi Coding Agent’s print mode has no built-in approval gate, and following its documented extension mechanism we adapt the official permission-gate example to block Bash, Write, and Edit calls. The final column reports vulnerabilities with confirmed attacker-controlled flows and verified observable harm, not merely sink reachability.

Table 10: Per-agent restrictive approval configurations used in the RQ1 approval replay.
<table><tr><td>Agent</td><td>Restrictive configuration</td><td>Verified outcomes</td></tr><tr><td>OpenClaw</td><td>tools.exec with security=allowlist and ask=always, empty allowlist</td><td>22/24</td></tr><tr><td>Hermes Agent</td><td>Isolated configuration with approvals . deny= [ &quot; * &quot; ]</td><td>1/16</td></tr><tr><td>OpenCode</td><td>Remove permission-skipping flag and set Bash and Edit to ask</td><td>0/17</td></tr><tr><td>Pi Coding Agent</td><td>Official-style extension blocks Bash, Write, and Edit in headless mode</td><td>1/3</td></tr><tr><td>Cline</td><td>--auto-approve false</td><td>0/7</td></tr><tr><td>Qwen Code</td><td>--approval-mode default</td><td>1/8</td></tr><tr><td>Kimi Code CLI</td><td>Isolated deny rules for Bash, Write, and Edit</td><td>6/12</td></tr><tr><td>Mistral Vibe</td><td>Remove --auto-approve and retain workspace trust</td><td>0/2</td></tr></table>

We verified that each restrictive mode was active using its native evidence channel, including approval-request or denial events, rejected tool results, the initialized tool list, or pending-approval records. Runs with no such evidence and no confirmed target were classified only as not reproduced, not as successful enforcement. This distinction prevents stochastic route changes from being credited to the approval mechanism.

Causes of the surviving cases. The nine enforcement failures trace to two defects in approval enforcement. Kimi Code CLI (six cases) has a configuration schema that accepts user deny rules for Bash, Write, and Edit, yet source inspection finds no startup caller that adds those rules to the permission-rule service, so the policy evaluates an empty rule set. OpenClaw (three cases) enforces security and ask in the node and gateway execution branches, but the tested packaged local-CLI path falls through to runExecProcess without consulting them when no host is explicitly selected.

The 22 policy-scope gaps break down as follows. OpenClaw (19 cases) defines per-call approval fields only on ExecToolConfig, so the file tools carry no approval checkpoint and workspaceOnly defaults to false, allowing all 13 disclosure and six write cases to proceed. Qwen Code (one) blocks its search tool at the approval boundary, yet an internally spawned subprocess reaches the process sink outside that boundary. Pi Coding Agent (one) has an official-style gate extension that blocks Bash, Write, and Edit but not Read. Hermes Agent (one) has a global deny rule that covers the terminal tool, but search files launches an internal shell subprocess outside that approval boundary. In all enforcement-failure cases, the logs contain no approval event before the protected operation executes.

## I Approval-Control Results

In autonomous, multi-step workflows, users typically grant agents automatic approval because an approval round trip for every tool call defeats unattended execution. RQ1 therefore measures the exposed surface under the permissive configuration commonly used for autonomous operation. For the approval replay in RQ1, we remove automatic approval for each eligible agent and run it in the strictest usable non-interactive approval configuration. The replay examines whether approval controls constrain the agent’s use of user-delegated authority when acting on installed skill content. The skill, confirmed variant, target, model, and execution environment are held fixed. Appendix H records the exact per-agent configurations.

Three subjects, namely Pochi, Agent Zero, and DB-GPT, do not expose a configurable approval layer in the tested execution mode, so their 15 vulnerabilities cannot be reevaluated under a corresponding restrictive approval setting. We pair the remaining 89 vulnerabilities across eight agents and count a vulnerability only if the oracle again confirms its attacker-controlled flow and verifies observable harm. Table 11 shows that 31 of 89 vulnerabilities (34.8%) remain exploitable after automatic approval is removed. Residual Exploitability is the percentage of vulnerabilities verified under the permissive configuration that remain verified under the restrictive configuration; the Total column uses the aggregate counts. Of the other 58, 40 carry explicit evidence that the configured control blocked the operation, while 18 do not reproduce the target attack in that run. All 31 surviving cases again produce observable command output, file disclosure, or file-system modification.

Table 11: Verified counts per agent under the permissive and restrictive approval configurations.
<table><tr><td></td><td>OpenClaw</td><td>OpenCode</td><td>Hermes Agent</td><td>Kimi Code CLI</td><td>Qwen Code</td><td>Cline</td><td>Pi Coding Agent</td><td>Mistral Vibe</td><td>Total</td></tr><tr><td>Permissive</td><td>24</td><td>17</td><td>16</td><td>12</td><td>8</td><td>7</td><td>3</td><td>2</td><td>89</td></tr><tr><td>Restrictive</td><td>22</td><td>0</td><td>1</td><td>6</td><td>1</td><td>0</td><td>1</td><td>0</td><td>31</td></tr><tr><td>Residual Exploitability</td><td>91.7%</td><td>0.0%</td><td>6.3%</td><td>50.0%</td><td>12.5%</td><td>0.0%</td><td>33.3%</td><td>0.0%</td><td>34.8%</td></tr></table>

The 31 surviving vulnerabilities separate into two causes. In nine cases, the configured policy covers the attempted action, but the execution path never checks it. We classify these cases as enforcementfailures. Kimi Code CLI contributes six, because its configuration schema accepts user deny rules for Bash, Write, and Edit, but nothing at startup loads those rules, so the permission policy evaluates an empty rule set. OpenClaw contributes three, because its approval settings are checked on two of its execution paths while the default local command path skips the check entirely. In both agents, the logs contain no approval event before the protected operation executes, which means the gate was never consulted rather than consulted and overruled. Appendix H traces each case to the responsible source path.

The second cause is a policy-scope gap (22 cases), where the control works as designed but its threat model does not cover the operation. Two blind spots recur. First, policies classify tools by whether they mutate state, so read-only operations are allowed outright even when they disclose attacker-chosen files. OpenClaw’s approval schema governs command execution but not its file-read and file-write tools (19 cases). Pi Coding Agent’s gate extension covers Bash, Write, and Edit but not Read. Second, the approval boundary is drawn at the tool layer rather than at the operating-system capability it wraps. Hermes Agent’s global deny rule blocks its terminal tool, yet a file-search tool that internally spawns a shell reaches the same process sink outside that boundary. Qwen Code blocks its search tool, yet an internal execution path spawns a subprocess that reaches the process sink outside that boundary. Consequently, file disclosure remains the weakest-protected family, with 14 of 20 cases (70.0%) exploitable. By contrast, OpenCode, Cline, and Mistral Vibe retain no verified vulnerability under their restrictive approval configurations. Approval controls therefore substantially reduce exploitation, but cannot protect operations they do not govern, and a configured control provides no security when its enforcement hook is absent from the active execution path. The surviving trust failures reflect gaps in policy coverage or enforcement along the active execution paths.

## J Real-Skill Validation Details

Corpus. The 694 real SKILL.md files come from four sources, namely 300 selected by stratified sampling from the SkillsMP long tail, 245 from the ClawHub site, 62 from the ClawHub registry, and 87 from official agent repositories and built-in packs. After excluding 61 files that yield no derivable marker, 633 remain adjudicable.

Adjudication protocol. Real skills carry no canary, so the taint source is a content-derived marker. We extract paths, URLs, commands, file names, and environment tokens from the skill itself. We discard tokens appearing in more than three corpus documents. We also exclude markers observed in an agent idle baseline run without the skill installed. A trigger requires the same three checks as the taint-confirmation predicate in equation 7, namely that the target sink was invoked, that a derived marker appears in a sink argument, and that the recorded stack frames cover the audited path.

Weaponization cases. Table 12 lists the 15 weaponized twins using anonymized skill identifiers.   
These are researcher-modified copies; the results do not imply that the original skills are malicious.

Table 12: The 15 weaponized twins, reported using anonymized skill identifiers, with their agents, techniques, and observed outcomes.
<table><tr><td>Skill ID</td><td>Agent</td><td>Technique</td><td>Outcome</td></tr><tr><td>S01</td><td>OpenClaw</td><td>Command</td><td>Downloads and executes an attacker script, remote code execution</td></tr><tr><td>S02</td><td>OpenClaw</td><td>Inline domain</td><td>The full credential pair reaches the attacker</td></tr><tr><td>S03</td><td>Cline</td><td>Inline URL</td><td>Agent initiates an OAuth flow using an attacker-controlled endpoint</td></tr><tr><td>S04</td><td>OpenClaw</td><td>Inline domain</td><td>Agent creates an API key and mailbox through the attacker</td></tr><tr><td>S05</td><td>OpenClaw</td><td>Inline domain</td><td>All bot authentication traffic, 20 requests, passes through the attacker</td></tr><tr><td>S06</td><td>Cline</td><td>Inline URL</td><td>Agent connects to the attacker over MCP and calls its tools</td></tr><tr><td>S07</td><td>OpenClaw</td><td>Inline endpoint</td><td>The full document reaches the attacker</td></tr><tr><td>S08</td><td>Pochi</td><td>Inline domain</td><td>Agent initiates an MCP handshake with the attacker</td></tr><tr><td>S09</td><td>Cline</td><td>Inline URL</td><td>Agent downloads an application installer from the attacker</td></tr><tr><td>S10</td><td>Cline</td><td>Inline URL</td><td>Agent downloads a CLI installer from the attacker</td></tr><tr><td>S11</td><td>Cline</td><td>Inline URL</td><td>Agent downloads a CLI installer from the attacker</td></tr><tr><td>S12</td><td>OpenClaw</td><td>Inline domain</td><td>DNS reconnaissance data reaches the attacker</td></tr><tr><td>S13</td><td>Pochi</td><td>Inline domain</td><td>Search keywords reach the attacker</td></tr><tr><td>S14</td><td>Pochi</td><td>Inline domain</td><td>All API queries pass through the attacker</td></tr><tr><td>S15</td><td>OpenClaw</td><td>Inline domain</td><td>Query data is POSTed to the attacker</td></tr></table>

Modification types. We selected these 15 from the triggering skills whose bodies contain service addresses suitable for substitution, and every completed twin is included. The modifications replace existing service-domain strings, installer addresses, reference endpoints, or a bootstrap command. All substituted addresses point to a listener under our control, so no external host receives any data. No twin adds instructions or rewrites task text; ten of the 15 change at most two lines, and the largest edit is a single global substitution of one service domain.