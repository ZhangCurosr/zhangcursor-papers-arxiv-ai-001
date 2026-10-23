# Automating Feature Flag Cleanup at Scale with Multi-Agent LLM Systems: An Industrial Experience at DoorDash

Atharv Chandratre

DoorDash, Inc.

San Francisco, CA, USA

atharv.chandratre@doordash.com

Jai Datar

DoorDash, Inc.

San Francisco, CA, USA

jai.datar@doordash.com

Abstract—Feature flags are runtime configuration mechanisms used for gradual feature rollouts, experimentation, and kill switches. Once a feature is fully rolled out, the flag and its gated code become technical debt and can introduce operational risk through inadvertently enabling dormant code paths. At DoorDash, over 56,000 feature flags span 623 repositories, with 1,659 currently stale as of May 2026, with internal estimates of one to two hours per manual cleanup. Existing rule-based cleanup tools proved insufficient in DoorDash’s codebases. Our contribution is an end-to-end automated LLM-based feature flag cleanup system that integrates querying live feature flag rollout data using MCP, Jira ticket processing, and end-to-end pull request creation. Our system successfully produced engineerapproved flag removal PRs for 45 (90%) of the 50 stale flags it was evaluated on, proving the system’s operational viability at scale with each cleanup averaging only \$4.79 and 13.8 minutes. We evaluated the system’s quality and reliability boundary based on success rates, cost per cleanup and time metrics. Furthermore, we discuss our lessons learned when architecting this system, to provide practical guidance for developing similar multi-agent systems.

## I. INTRODUCTION

Feature flags (also known as feature toggles) are a fundamental practice in modern software delivery, enabling safe gradual rollouts, A/B testing, and kill switches [1]–[3]. However, once a feature flag has been fully rolled out, the flag becomes stale. Its conditional logic no longer serves a purpose and constitutes technical debt. Stale flags also pose a direct operational risk: they preserve dormant code paths that can be inadvertently reactivated. In one well-documented incident, a trading firm repurposed a stale feature flag during deployment, triggering \$460M in unintended trades within 45 minutes [4]. Despite these risks, prior work has found that approximately 25% of feature flags persist for over eight years in opensource projects [2], and practitioners consistently report that flag cleanup is burdensome and frequently deferred [3], [5].

At DoorDash, the problem manifests at a significantly large scale. Our experimentation platform manages over 56,000 Dynamic Values (DVs), the company’s feature flag abstraction, across approximately 623 repositories, with 1,659 currently stale. Approximately 2,300 new DVs are created every month.

With internal estimates of one to two hours per manual cleanup, the backlog grows faster than teams can address it.

Throughout this paper, we will be using the abbreviation DV to refer to Dynamic Values, the term for feature flags at DoorDash.

The most prominent automated approach, Piranha [6] and its polyglot successor [7], simplifies conditional expressions by matching AST patterns. As we detail in Section II-A and quantify in Section II-B, DoorDash’s dependency-injected wrapper pattern, non-boolean DVs, and cross-file cleanup requirements make these tools structurally inapplicable in our setting.

To address these challenges, we designed and deployed a multi-agent LLM system built on Google ADK [8] that automates the full lifecycle of feature flag cleanup at Door-Dash. An orchestrator agent analyzes each stale flag, queries the experimentation platform for its live production rollout state via MCP (Model Context Protocol) [9], and formulates a cleanup strategy. It then spawns specialized removal agents that execute code changes in parallel, each operating in an isolated git worktree [10] to prevent interference between concurrent cleanups. The pipeline autonomously handles Jira ticket processing, code modification, test validation, and pull request creation, producing merge-ready PRs requiring zero manual intervention between the initial engineer confirming the inferred target value and final PR creation.

Our contributions are:

• The design and deployment of a multi-agent architecture at scale for feature flag cleanup, where an orchestrator coordinates parallel removal agents operating in isolated git worktrees.

• An end-to-end automated pipeline performed by each cleanup agent that autonomously bridges issue tracking, querying the experimentation platform using MCP, code modification, test validation, and pull request creation for every stale feature flag.

• A detailed industrial evaluation of our LLM-based feature flag cleanup system, reporting success rates, cost, time, complexity breakdown, and failure analysis, along with an analysis of why rule-based tools such as Piranha do not apply to DoorDash’s DV access pattern (Section II-A). This evidence is absent from literature we reviewed [11]– [16].

```kotlin
Listing 1. Example three-layer DV access pattern (simplified).
// Layer 1: RuntimeKeys.kt - string constant
object RuntimeKeys {
const val ENABLE_X = "enable_feature_x"
}
// Layer 2: FeatureFlags.kt - wrapper function
@Singleton
class FeatureFlags @Inject constructor(
private val dvClient: DynamicValuesClient
) {
fun shouldEnableX(userId: String): Boolean =
dvClient.getBoolean(ENABLE_X,
withContext("user_id" to userId), false)
}
// Layer 3: Business logic - usage via DI
class MyService @Inject constructor(
private val featureFlags: FeatureFlags
) {
fun process(userId: String) {
if (featureFlags.shouldEnableX(userId)) {
handleNewPath()
} else { handleOldPath() }
}
}
```

## II. BACKGROUND & MOTIVATION

## A. Feature Flags at DoorDash

DVs follow a lifecycle: creation → experimentation → rollout → stale → removal. A DV is classified as stale when it: (1) has not been modified for 90 days (2) is still referenced in code (3) is not in an end-of-lifecycle status (archived or retired) (4) is not on an exclusion list. Jira tickets are automatically created for stale DVs identified by a daily cron job and assigned to the DV’s creator.

DVs implementation follows a three-layer access pattern in Doordash services, as illustrated in Listing 1. Layer 1 defines the DV name as a string constant in a Runtime keys file. Layer 2 provides a dependency-injected wrapper function in a feature flag file that calls the DynamicValuesClient API (getBoolean, getString, or getObject) with context dimensions such as user ID or country code. Layer 3 is business logic that injects the FeatureFlags singleton via dependency injection and calls the wrapper.

This pattern, while providing clean separation of concerns, means that the DV name, the API call, and the conditional logic exist in different files connected only through method calls and dependency injection. These are semantic relationships that AST-based tools cannot follow.

## B. Why Automation Is Needed

With 1,659 stale DVs and internal estimates of one to two hours per manual cleanup, clearing the backlog would require thousands of engineering hours. Approximately 2,300 new DVs are created monthly, ensuring the problem compounds faster than teams can address it.

A critical challenge is target value determination. When a DV has been fully rolled out, the code default may differ from the production value. A tool operating on source code alone could silently hardcode the wrong value. Our agent queries the experimentation platform via MCP to retrieve the live rollout state, ensuring the correct value is used when performing the cleanup.

Cleaning up a DV requires removing the constant definition, the wrapper function, all call sites (replacing with the target value or simplifying conditional logic), updating or removing affected test cases, and ensuring the CI/CD pipeline passes. Even for a simple boolean DV, this involves coordinated changes across 3-20 files. These changes require understanding code semantics, not just syntax.

## III. SYSTEM DESIGN

## A. Two-Phase Architecture

Figure 1 shows the system architecture. The system operates in two phases with a human-in-the-loop checkpoint.

Phase 1 (Analysis & Report). The orchestrator agent, powered by Claude Sonnet 4.6 [17], processes stale DV cleanup requests. For each DV, it: (1) fetches the STALEDV Jira tickets via CLI, (2) queries the experimentation platform via MCP for DV metadata (UUID, rollout percentage, target value), (3) discovers the local repository and searches for all code references, and (4) generates a structured confirmation report listing the proposed target value and affected files for each DV.

The experimentation MCP integration is noteworthy: by calling GetDynamicValue and GetDynamicValue-Details endpoints, the agent retrieves the current production rollout state, including rollout percentage and winning variant. This enables correct target value determination. For example, a DV introduced for a feature that was ultimately abandoned will reflect a 0% rollout state in the platform, ensuring the agent removes it using the baseline value rather than assuming a rolled out flag.

Human Checkpoint. The engineer verifies each DV’s proposed target value before cleanup proceeds. This acts as a safety check and ensures DVs with ambiguous rollout states, such as partial rollouts or abandoned experiments are correctly handled.

Phase 2 (Parallel Cleanup). For each confirmed DV, the orchestrator spawns a dedicated, isolated removal agent powered by Claude Opus 4.6 [17] in a git worktree [10]. Up to four agents run concurrently per repository (configurable value). Each agent executes the iterative cleanup workflow described in Section III-B, resulting in a pull request assigned to the DV’s owner.

## B. Removal Agent Design

Each removal agent follows an iterative workflow: (1) search for all DV references in the codebase, including the constant definition, wrapper function, and all call sites; (2) determine a cleanup strategy based on DV type (boolean, integer, string) and usage pattern; (3) apply edits, replacing wrapper calls with the target value and simplifying conditional logic; (4) remove the wrapper function, definition and constant; (5) update or remove all affected test cases (e.g., deleting “DV disabled” test branches, removing @DynamicValueOverrides annotations); (6) ensure build passes (7) verify tests pass (8) verify patch coverage ≥95%; (9) run the Detekt linter; and (10) if validation fails, diagnose and fix the issue before retrying.

![](images/5707c8d121c54ec0f2f3ed51038a1a49d55b698c4fa6150aefb7f5b8ba463310.jpg)  
Fig. 1. System architecture. The orchestrator agent (dashed rectangle) manages the cleanup lifecycle in two phases. In Phase 1, it retrieves stale DVs from Jira, queries the experimentation platform for rollout state, and searches the codebase for all references. The resulting DV report is reviewed by an engineer who confirms target values. In Phase 2, the orchestrator spawns parallel cleanup agents in isolated git worktrees, each editing code, running tests, verifying coverage, and linting before creating a pull request.

Safety mechanisms. Four mechanisms prevent incorrect changes: (a) git worktree isolation: each agent operates in a dedicated worktree, preventing interference between concurrent agents; (b) a one-hour hard timeout per agent, preventing runaway sessions; (c) JaCoCo patch coverage validation ensuring ≥95% coverage on changed lines; and (d) Detekt static analysis enforcing code style and detecting common issues. Gradle is executed with --no-daemon to eliminate crossworktree state pollution during parallel operations.

## IV. EVALUATION

## A. Research Questions and Setup

We evaluate our system along three dimensions:

• RQ1 (Deployment Outcome): How often does the deployed system produce PRs that engineers can merge directly or after minor revision?

• RQ2 (Reliability Boundary): What feature flag characteristics predict whether the system produces a mergeready cleanup in a single shot, requires minor revision, or needs engineer intervention?

• RQ3 (Operational Viability): What are the time and cost characteristics of running the system in practice? Can it run at scale?

Dataset. We analyzed the 50 most recent stale DV removals performed by our system (as of 4/14/26) across multiple Kotlin-based repos at DoorDash. We classified these 50 DVs into three complexity levels: simple (single boolean check, 1- 2 files; n=6), medium (multiple usages, 3-5 files, some test changes; n=18), and complex (nested conditionals, cross-file dependencies, non-boolean types, or significant test updates;

n=26). This classification was done post hoc using these criteria. DV types include boolean (41), string (6), and integer/long (3).

Methodology. For each DV, we ran the agent and evaluated the generated pull request on three criteria: CI passes (including 95% patch coverage), code correctness, and developer review. We classify outcomes as: Accepted (PR accepted in the first shot without changes); Accepted with revision (accepted after minor fixes, typically one additional agent iteration); or Failed (engineer had to step in for significant corrections).

## B. Results

RQ1 (Deployment Outcome). Table I summarizes the evaluation results. Of the 50 DVs, 31 (62%) were cleaned up successfully in the first shot, 14 (28%) required minor revisions, and 5 (10%) failed. The overall cleanup rate, defined as DVs where the agent produced a usable PR, was 90%.

Of the 14 revisions, the most common cause was insufficient patch coverage (6 cases), where the agent needed one additional iteration to add or update tests. The remaining 8 involved incomplete dead code removal, typically missing a dead variable or reference in a deep call chain, which the agent fixed when reprompted.

CI passed on the first attempt for 44 of 50 DVs (88%). The 6 initial failures were all coverage-related; none were due to incorrect code changes. This aligns with our system design: the agent is constrained from creating a PR unless unit tests and static analysis (Detekt) pass locally. While the agent’s system prompt explicitly mandates 95% patch coverage locally, we observed occasional miscalculations in its LLM-driven selfassessment. DoorDash’s CI pipeline caught these discrepancies as it fails if patch coverage is <95%.

Figure 2 breaks down results by complexity. All simple DVs were cleaned up in a single pass. Medium-complexity DVs achieved a 94% cleanup rate (17/18). Even for complex DVs involving cascading parameter removal through 5+ method layers, nested conditionals, and significant test refactoring, the overall cleanup rate was 85% (22/26).

TABLE I  
EVALUATION RESULTS ON 50 STALE DVS.
<table><tr><td>Metric</td><td>Value</td></tr><tr><td>DV cleanups analyzed</td><td>50</td></tr><tr><td>Accepted (first-shot)</td><td>31 (62%)</td></tr><tr><td>Accepted with revision Failed</td><td>14 (28%)</td></tr><tr><td>Overall cleanup rate</td><td>5 (10%) 45/50 (90%)</td></tr><tr><td>CI first-pass rate</td><td>44/50 (88%)</td></tr><tr><td>Avg. time per DV</td><td>13.8 min</td></tr><tr><td>Median time per DV</td><td>12.3 min</td></tr><tr><td>Avg. LLM API cost per DV</td><td>$4.79</td></tr><tr><td>Median LLM API cost per DV</td><td>$3.49</td></tr></table>

![](images/851452d3cffe7bf08ef1afe29ca0296e9731c4b4f1abe9ad105656bee295e3af.jpg)  
Fig. 2. Cleanup outcomes by complexity level. Simple DVs achieve 100% first-shot acceptance; complex DVs show higher revision and failure rates but still achieve 85% overall cleanup.

RQ2 (Reliability Boundary). Figure 2 breaks down outcomes by complexity. All simple DVs (n=6) produced mergeready PRs in a single pass. Medium-complexity DVs achieved a 94% cleanup rate: 14 of 18 in a single pass, 3 after one revision (2 for incorrect test modifications, 1 for insufficient patch coverage), and 1 requiring engineer intervention due to incomplete removal. Complex DVs present the reliability boundary at an 85% cleanup rate: 11 of 26 succeeded in a single pass, 11 required one revision (4 for incorrect test modifications, 2 for residual unused variables, 5 for insufficient patch coverage), and 4 required engineer intervention due to incomplete removal. This is because removing Complex DVs is characterized by cascading parameter removal through 5+ method layers, nested conditionals, and significant test refactoring. The distinguishing factor across all 5 intervention cases is call-chain depth with cross-interface parameter threading, not flag type or file count. In each case, the agent correctly identified and removed the majority of dead code but missed one or two links in a multi-file chain; the remaining work was completable by an engineer. Critically, across all 50 DV removal PRs generated, we did not observe any bugs or regressions introduced by the changed code, indicating that the system’s reliability boundary is one of completeness, not correctness.

RQ3 (Operational Viability). Figure 3 plots wall-clock time against API cost for all 50 DVs, colored by complexity. Simple and medium DVs cluster in the low-cost, lowtime region, while complex DVs exhibit wider variance with outliers reaching 40 minutes and \$19. The 5 failed DVs tend toward longer execution times. Table II breaks down averages by complexity.

TABLE II  
TIME AND COST BY DV COMPLEXITY.
<table><tr><td>Complexity</td><td>n</td><td>Avg. time</td><td>Med. time</td><td>Avg. cost</td><td>Med. cost</td></tr><tr><td>Simple</td><td>6</td><td>7.5 min</td><td>8.2 min</td><td>$2.69</td><td>$2.50</td></tr><tr><td>Medium</td><td>18</td><td>10.4 min</td><td>9.9 min</td><td>$3.46</td><td>$3.42</td></tr><tr><td>Complex</td><td>26</td><td>17.7 min</td><td>14.4 min</td><td>$6.20</td><td>$4.50</td></tr></table>

![](images/afffd13fc5f2d850850c18e25799efbfafde8590338b4607fb7c00562813d888.jpg)  
Fig. 3. Agent cost vs. wall-clock time for each DV, colored by complexity. Failed DVs are marked with ×. Simple and medium DVs cluster in the lowcost region; complex DVs show higher variance.

The agent processed DVs in an average of 13.8 minutes (median: 12.3 min) at \$4.79 (median: \$3.49) per cleanup. Internal engineering estimates put manual cleanup at one to two hours of focused effort per DV. Even at the conservative end of this range, the agent’s runtime represents a substantial reduction in engineering effort. This demonstrates that the system has the time and cost characteristics to run at scale and saves a substantial amount of time and money when compared to an engineer performing feature flag cleanup manually.

## V. DISCUSSION & LESSONS LEARNED

Model role specialization yields cost-effective results. We use Claude Sonnet 4.6 [17] for orchestration (requiring broad context management) and Claude Opus 4.6 [17] for code modification (requiring deep reasoning about code semantics). This split balances cost and capability: orchestration involves lightweight metadata gathering and planning, while code modification requires deep reasoning about semantics, call chains, and test dependencies. By reserving the more expensive model for the cognitively demanding editing phase, we reduce overall API cost without sacrificing cleanup quality.

Git worktree isolation is essential for parallel agents. Each removal agent operates in an isolated worktree, preventing interference between concurrent cleanups. A failed cleanup is safely discarded by removing the worktree. Running Gradle with --no-daemon prevents daemon conflicts across worktrees. Early experiments without worktree isolation led to race conditions where concurrent agents modified the same files, producing corrupted diffs.

Experimentation platform integration prevents silent regressions. Without access to live rollout state, any automated cleanup approach, whether rule-based or LLM-based, risks hardcoding incorrect values and preserving the wrong behavior after flag removal. In our setting, this integration added negligible latency while eliminating a recurring source of target-value errors, making it one of the highest-leverage design choices in the system.

Developer feedback. We collected informal feedback from engineers who reviewed agent-generated PRs. One engineer noted: “The agent got every dead code line and dead unit test removed. Given the size of this PR, that’s quite impressive.” Another observed a code quality limitation: “The removal was correct, but variable names should have been updated, since the flag’s removal changed their semantics.” A third described a common pattern across revised PRs: “Everything worked in the first commit apart from patch coverage. After one reprompt the agent added the missing tests and it was good to merge.”

For teams considering similar automation, our results suggest three practical prerequisites: a reliable source of deployed flag state, worktree isolation for parallel edits, and deterministic validation gates before PR creation.

Threats to validity. Internal validity: Each DV was processed once; LLM non-determinism means re-running could yield different outcomes. Complexity labels were assigned post hoc by the authors based on the classification described in Section IV-A. The one to two hour manual cleanup estimate per feature flag is based on internal engineering experience rather than precise wall-clock measurement. External validity: Our evaluation covers repositories at a single company with a specific DV access pattern (wrapper-function access via dependency injection). Organizations using direct flag API calls where tools like Piranha are more directly applicable may see different results. The analysis was also performed on feature flags removed in Kotlin-based repos at DoorDash. Generalizability to other languages, flag frameworks, or codebases requires further study.

## VI. RELATED WORK

Feature flag management. Prior work shows that feature flags accumulate as technical debt and that cleanup practices are inconsistently followed in practice [1]–[3], [5], [18]. These studies establish both the persistence of stale flags and the diversity of flag-usage patterns in large codebases.

Automated code cleanup. Piranha [6] is the most directly comparable prior system, automatically generating cleanup diffs for 1,381 stale flags at Uber. Its polyglot successor [7] extends support to multiple languages via a unified rule engine. These tools target settings where explicit AST rewrite rules can simplify flag-gated code; our setting differs due to dependency-injected wrapper pattern, non-boolean DVs, and cross-file refactoring including test maintenance. Other automated cleanup tools address adjacent but distinct problems: dead code removal (SCARF [19]), large-scale AST refactoring (ClangMR [20], OpenRewrite [21]), LLM-based dead code elimination at snippet level (DCE-LLM [22]), and LLMvalidated technical debt remediation (ACE [23]). None target end-to-end feature flag cleanup.

LLM-assisted flag cleanup in industry. Several organizations have recently deployed LLM-based systems for feature flag removal, including Google (Gemini) [11], Duolingo (Codex CLI) [12], Relevance AI (Claude) [13], Reflag (GPT-4o) [14], Glovo (Copilot) [15], and Atlassian (Rovo Dev) [16]. These reports describe system architectures and qualitative outcomes but do not publish quantitative evaluation with success rates, cost, time, complexity breakdown, failure analysis, or comparison against rule-based baselines. The closest architectural analog is bye-bye-flag [13], which shares our use of parallel agents in git worktrees and rollout-based target value derivation; our system differs in model role specialization, deterministic patch coverage enforcement, and MCP-based experimentation platform integration. Our work also provides a rigorous empirical evaluation of this class of systems.

## VII. CONCLUSION

We presented an LLM-based system for automated feature flag cleanup at DoorDash that combines orchestration, parallel removal agents, experimentation-platform lookup, and workflow-integrated validation. On the 50 most recent stale Dynamic Values processed by the system, it produced usable pull requests for 45 flags at an average cost of \$4.79 and 13.8 minutes per flag. Our evaluation covers success rates, cost, time, complexity breakdown, failure modes, and a comparison against rule-based baselines, providing empirical evidence absent from prior industry reports of LLM-based flag cleanup. Our lessons learned highlight that model role specialization, repository isolation, live platform integration, and deterministic validation gates are practical prerequisites for deploying LLM-based code cleanup at scale. Future work includes several directions: (1) introducing a confidence score in Phase 1 to auto-approve low-risk cleanups - e.g., boolean DVs at 100% rollout where the code default matches the production value. This reserves the human checkpoint only for ambiguous cases; (2) adding a post-cleanup code quality review step to catch semantic issues such as variable names that no longer reflect their purpose after flag removal; (3) extending support to additional languages; and (4) integrating with CI pipelines to trigger cleanup automatically at staleness thresholds.

## REFERENCES

[1] M. T. Rahman, L.-P. Querel, P. C. Rigby, and B. Adams, “Feature toggles: practitioner practices and a case study,” in Proceedings of the 13th international conference on mining software repositories, 2016, pp. 201–211.

[2] J. Meinicke, J. Hoyos, B. Vasilescu, and C. Kastner, “Capture the feature¨ flag: Detecting feature flags in open-source,” in Proceedings of the 17th International Conference on Mining Software Repositories, 2020, pp. 169–173.

[3] J. Hoyos, R. Abdalkareem, S. Mujahid, E. Shihab, and A. E. Bedoya, “On the removal of feature toggles: A study of python projects and practitioners motivations,” Empirical Software Engineering, vol. 26, no. 2, p. 15, 2021.

[4] U. Securities, E. Commission et al., “Sec charges knight capital with violations of market access rule,” SEC Press Release No. 2013-222, 2013.

[5] R. Mahdavi-Hezaveh, J. Dremann, and L. Williams, “Software development with feature toggles: practices used by practitioners,” Empirical Software Engineering, vol. 26, no. 1, p. 1, 2021.

[6] M. K. Ramanathan, L. Clapp, R. Barik, and M. Sridharan, “Piranha: Reducing feature flag debt at uber,” in Proceedings of the ACM/IEEE 42nd International Conference on Software Engineering: Software Engineering in Practice, 2020, pp. 221–230.

[7] A. Ketkar, D. Ramos, L. Clapp, R. Barik, and M. K. Ramanathan, “A lightweight polyglot code transformation language,” Proceedings of the ACM on Programming Languages, vol. 8, no. PLDI, pp. 1288–1312, 2024.

[8] Google, “Agent development kit (ADK): An open-source framework for building AI agents,” https://google.github.io/adk-docs/, 2025.

[9] Anthropic, “Model context protocol (MCP): An open standard for connecting AI systems to data sources,” https://modelcontextprotocol.io/, 2024.

[10] Git Project, “git-worktree – manage multiple working trees,” https:// git-scm.com/docs/git-worktree, 2024.

[11] S. Nikolov, D. Codecasa, A. Sjovall, M. Tabachnyk, S. Chandra,¨ S. Taneja, and C. Ziftci, “How is google using ai for internal code migrations?” in 2025 IEEE/ACM 47th International Conference on Software Engineering: Software Engineering in Practice (ICSE-SEIP). IEEE, 2025, pp. 481–492.

[12] J. Welch, “Building an AI Agent to Remove Feature Flags,” https://blog. duolingo.com/buildingaiagents/, Dec. 2025, Duolingo Blog, accessed Apr. 13, 2026.

[13] I. Hagoel, “How We Used AI to Clean Up 200+ Feature Flags (And Open Sourced It),” https://relevanceai.com/blog/ how-we-used-ai-to-clean-up-feature-flags-and-open-sourced-it, 2026, relevance AI Blog, Accessed: Apr. 13, 2026.

[14] Gareth, “Building AI Flag Cleanup,” https://reflag.com/blog/ building-ai-flag-cleanup, 2025, reflag Blog, Accessed: Apr. 13, 2026.

[15] G. Celada, “How we use AI to keep feature flags under control,” https://tech-blog.glovoapp.com/blog/ how-we-use-ai-to-keep-feature-flags-under-control/, 2025, glovo Engineering Blog, Accessed April 2026.

[16] M. Gajowski, “How We Turned Feature Flag Cleanup Into a Mostly-Hands-Off AI Workflow Problem,” https://www.atlassian.com/ blog/developer/turning-ff-cleanup-into-handsoff-ai-workflow, 2026, atlassian Engineering Blog, Accessed April 2026.

[17] Anthropic, “Claude: Model overview and documentation,” https://docs. anthropic.com/en/docs/about-claude/models, 2025.

[18] T. Rahman, “Feature toggle usage patterns: A case study on google chromium,” in 2023 IEEE/ACM 20th International Conference on Mining Software Repositories (MSR). IEEE, 2023, pp. 142–147.

[19] W. Shackleton, K. Cohn-Gordon, P. C. Rigby, R. Abreu, J. Gill, N. Nagappan, K. Nakad, I. Papagiannis, L. Petre, G. Megreli et al., “Dead code removal at meta: Automatically deleting millions of lines of code and petabytes of deprecated data,” in Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering, 2023, pp. 1705–1715.

[20] H. K. Wright, D. Jasper, M. Klimek, C. Carruth, and Z. Wan, “Largescale automated refactoring using clangmr,” in 2013 IEEE International Conference on Software Maintenance. IEEE, 2013, pp. 548–551.

[21] J. Gehring, “Deterministic automatic refactoring at scale,” in Proceedings of the 39th IEEE International Conference on Software Maintenance and Evolution (ICSME). IEEE, 2023.

[22] M. Chen, G. Li, L.-I. Wu, and R. Liu, “Dce-llm: Dead code elimination with large language models,” in Proceedings of the 2025 Conference of the Nations of the Americas Chapter of the Association for Computational Linguistics: Human Language Technologies (Volume 1: Long Papers), 2025, pp. 9942–9955.

[23] A. Tornhill, M. Borg, N. Hagatulah, and E. Soderberg, “Ace: automated¨ technical debt remediation with validated large language model refactorings,” in Proceedings of the 33rd ACM International Conference on the Foundations of Software Engineering, 2025, pp. 1318–1324.