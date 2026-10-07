# BEYOND CORRECTED MEMORY: EXECUTION CONSISTENCY IN MULTI-AGENT SYSTEMS

Zhe Yu<sup>1,\*</sup> Zixuan Wang<sup>1,\*</sup> Peidong Wang<sup>1,\*</sup> Hehai Lin<sup>1</sup> Ruochen Zhao<sup>2</sup> Chengwei Qin<sup>1,†</sup>

<sup>1</sup>The Hong Kong University of Science and Technology (Guangzhou)

<sup>2</sup>Singapore University of Technology and Design

<sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding author.

## ABSTRACT

Shared memory coordinates agents’ actions, but correct records do not establish that those actions satisfy task requirements. Memory governance and failure diagnosis regulate or inspect recorded information; they do not by themselves establish whether it is sufficient to judge task duties. We define execution consistency through duties governing state use, information handoffs, and final-state agreement, with explicit evidence conditions for judging fulfillment. Our core claim is that identical retained records can correspond to compliant and violating executions under the same task rule. Controlled removal of evidence such as receipt, action dependence, or response validity leaves 82.4% of opposite-label pairs indistinguishable; restoration separates 97.9% of the merged pairs. Natural-log annotations identify the defined violations in actual executions. However, existing logs do not always explicitly represent the execution relationships needed for these judgments. To assess the definition’s practical value, we use CAVERT, a framework for consistency diagnosis and recovery, to extract supported relationships from logs and apply these criteria. It consistently outperforms contract-prompted LLM and rule-based baselines in diagnosis across all 12 benchmark–executor settings. Under the same gate and executor limits, it also outperforms rule-guided recovery in all four evaluated environments. These findings identify execution evidence that agent-memory and execution interfaces should preserve for reliable judgment.

## 1 INTRODUCTION

Shared memory lets one agent’s observations guide another’s actions (Yu et al., 2026b; Margalit et al., 2026). A procurement agent may complete an order after receiving notice that its approval has been revoked. The order succeeds, but the agent has not acted consistently with the updated approval. Judging this behavior requires checking how shared information was used, not just whether the task succeeded or the stored records agree. We ask what makes execution over shared memory consistent, and what evidence establishes that consistency.

Memory governance and transactions constrain state, dependencies, and effects (Margalit et al., 2026; Li et al., 2026). Diagnosis and process auditing evaluate available traces (Barke et al., 2026; Cao et al., 2026), while commitment protocols and runtime verification address duties under partial observation (Baldoni et al., 2015; Mahe et al., 2022). The missing link is a criterion for when an execution record contains enough evidence to distinguish duty fulfillment from violation. This requires linking task duties to message receipt, state use, tool effects, and observation coverage.

Our key insight is that record correctness does not guarantee evidence sufficiency. A recorded correction need not establish receipt or downstream use; a cancellation request need not establish an effect lasting through the deadline (Figure 1). Thus, identical retained records can correspond to compliant and violating executions under the same task rule. Better reasoning over those records cannot recover the missing distinction. Execution consistency therefore treats evidence sufficiency—not record agreement alone—as part of the judgment problem.

We define execution consistency as judging task-duty fulfillment from available execution evidence. A judgment must establish the applicable duty, sufficient observation of its execution window, and whether an admissible response fulfilled it; unresolved evidence yields an undetermined outcome rather than presumed compliance or violation. We instantiate this principle across state use, information handoffs, and final-state agreement, covering twelve mechanisms.

![](images/e0d205f3c501f03924224c382024b154ca9634f43a4d13e8ced01538c4aa919e.jpg)  
Figure 1: One cancellation duty, three continuations after revocation (a): fulfillment (b, N), nonfulfillment with complete relevant records (c, V), or insufficient evidence (d, U).

To test whether accurate shared-memory contents suffice for judgment, we remove and restore selected execution evidence across 450 controlled pairs, holding each pair’s contract, query, and fullexecution labels fixed. Removal can merge compliant and violating executions into identical retained inputs; restoration separates nearly all merged pairs. Independent annotations of 360 natural executions confirm that the defined violations occur in practice.

However, these relationships are not always explicit in existing logs. CAVERT, a framework for consistency diagnosis and pre-commit intervention, extracts supported relationships, checks source references, and applies the criteria through deterministic rules, retaining unresolved evidence as unknown. Its findings guide recovery. It consistently outperforms contract-prompted LLM and rulebased diagnostic baselines across all 12 benchmark–executor settings, and rule-guided recovery across all four evaluated environments under the same gate and executor limits.

## Our contributions are:

1. We define task-level execution consistency through duties and evidence of fulfillment, with explicit rules for violation, no violation, and undetermined judgments.

2. We establish when retained records cannot distinguish compliant from violating runs, test the required relationships through controlled removal and restoration, and identify the defined violations in natural logs.

3. We apply these criteria through CAVERT to diagnose existing logs and guide pre-commit intervention, demonstrating their practical value through diagnosis and recovery experiments.

## 2 RELATED WORK

State governance. Which state should agents retain and be allowed to use? Distributed consis tency constrains operation order, update visibility, and replica convergence (Herlihy & Wing, 1990; Lloyd et al., 2011; Vogels, 2009; Shapiro et al., 2011). Agent-memory research studies retention, access, conflicting records, and visibility (Hu et al., 2026; Margalit et al., 2026; Ren et al., 2026; Volkov et al., 2026; Yu et al., 2026b). TOKI formalizes write-time conflict resolution (Wang, 2026); Khan (2026) verifies a concurrency hierarchy under deterministic generation. MemTX adds belief commit, action gating, and cascading repair (Li et al., 2026). Two runs can share the same approvalregister history yet differ in whether the order is canceled. We define the task duties and execution evidence that distinguish such runs, linking shared state to actions, handoffs, and outcomes (Appendix A.5). RQ3 compares CAVERT with MemTX.

Diagnosis and process auditing. Which behavior explains a failure or violates a task requirement? Failure analysis classifies errors and locates responsible agents and steps (Cemri et al., 2025; Zhang et al., 2025; 2026b). AgentRx checks constraints for attribution (Barke et al., 2026); DoVer tests failure hypotheses through interventions (Ma et al., 2026). PAE audits procedural integrity, including agreement between action claims and tool execution (Cao et al., 2026). We focus on shared-memory duties and test whether the retained evidence distinguishes their fulfillment from violation, complementing diagnosis and process audit.

Obligation and behavior verification. Does observed behavior satisfy an explicit rule? Commitment protocols and runtime verification formalize duties, deadlines, and three-valued judgments (Singh, 1999; Mallya & Singh, 2007; Fornara, 2011; Bauer et al., 2011). They also analyze how partial observation affects compliance checking (Baldoni et al., 2015; Mahe et al., 2022). Applicationlevel checks enforce or repair constraints (Balegas et al., 2015; Chang & Geng, 2025); other work checks reasoning, plans, policy compliance, and task actions (Feng et al., 2026a; Wu et al., 2026; Levy et al., 2026; Froger et al., 2026). Building on these semantics, we specify how generated messages, state use, and tool effects establish task-duty fulfillment. Controlled pairs test which relationships must be retained for judgment (Section 3; Appendix A.6).

These lines of work answer different questions. Memory consistency governs what state is retained, exposed, or eligible for use; diagnosis and runtime verification reason about failures or duties from available traces. Execution consistency asks whether the retained trace contains the relationships needed to distinguish fulfillment from violation under a task duty. Our removal–restoration study tests this distinction directly: removing those relationships can make oppositely labeled executions observationally identical to the judge.

## 3 EXECUTION CONTRACTS AND EVIDENCE

Execution consistency is judged relative to a task duty and the evidence available to assess it. It does not require global agreement among agents or memory states, only enough evidence to determine whether the relevant duty was fulfilled, violated, or remains unresolved. The unit of judgment is a task duty within an execution; trace-level consistency aggregates the resulting findings.

The public contract P encodes task instructions and declared task or framework policies, fixed before evaluation. For procurement, receipt of an approval revocation activates the duty to cancel the affected order by the final checkpoint. Thus P specifies responsibility, activation, scope, permitted response, and deadline; execution evidence establishes fulfillment.

Three contract families. We group duties by where consistency must hold: at state use, across information handoffs, or among designated final states. Strict-temporal contracts govern state use and contradiction resolution at an action or checkpoint. Conflicting writes commit “approved” despite a visible cancellation. Stale dependency use places an order using a replaced approval. Unresolved contradictions leave both states active at a required checkpoint. Contract-permitted exceptions, such as version pinning, are excluded.

Causal-semantic contracts require downstream use to preserve responsibility, evidential support, and scope. A missed handoff ignores a delivered supplier change. An unsupported assumption promises shipment after a failed stock lookup. Unsupported reconstruction presents an unreadable invoice’s inferred amount as observed. Source scope loss extends one supplier’s evidence to all suppliers. Source-view scope loss treats one page as all orders. Evidence status escalation turns “unverified” into “verified.”

Eventual-semantic contracts require agreement among designated final states by a declared endpoint. Unresolved final divergence leaves an order both active and canceled. Competing final answers submit conflicting totals without selecting one. An unresolved final evidence state leaves required checks both complete and pending.

Figure 2 groups twelve candidate mechanisms within these families. Families organize duties; mechanisms identify suspected failures; RQ1’s five factors test their evidence requirements (Section 5.1). An execution can involve multiple families. Appendix A.2 gives detailed conditions and exceptions.

Judging these duties requires more than the updated record. Writing a revocation does not establish receipt; receipt does not establish use. A cancellation request does not establish a successful effect lasting through the deadline. No logged cancellation may mean either that none occurred or that its channel was not recorded.

![](images/68461abedf7006ce37e5affe8ee66cb3526aef630ed7eebd2089f8497bb6b261.jpg)  
Figure 2: Twelve candidate mechanisms grouped by state use, information handoffs, and final-state agreement. Conditions and exclusions: Table A2.

The execution record T preserves these distinctions by linking messages, memory operations, tool calls and results, handoffs, and final actions. It records event order, who could see a state, which actions used it, and which updates replaced it. Coverage records identify captured channels and times. The recording endpoint says when logging stopped, not whether all relevant events were captured. Appendix A.1 gives the formal schema.

## 3.1 EVIDENCE-BASED CONSISTENCY JUDGMENTS

Return to the procurement agent that orders after receiving the revocation. It must cancel by the checkpoint. An authorized cancellation that succeeds on time and remains effective fulfills this duty: no violation (N). Complete relevant records with no qualifying cancellation establish a violation (V). If the cancellation channel was not fully recorded and fulfillment remains unresolved, neither conclusion is justified: the verdict is undetermined (U). U is an evidential outcome, not a confidence score: the record justifies neither fulfillment nor violation. These judgments concern the cancellation duty, not whether the earlier stale use occurred (Figure 1).

A finding F is a suspected problem to check, not a confirmed violation. Given the contract P and execution record T, four questions determine its verdict. The first two identify the problem and duty; the last two assess the evidence and response:

1. What might be wrong (C)? Using an old approval is not necessarily a problem. Compare records about the same entity, attribute, and scope (their semantic key). Using that approval after a verified replacement became visible to the agent is stale use. Historical references and contract-permitted pinned versions are excluded.

2. Who must do what, and by when (O)? A suspected problem does not establish that a response is already due. The contract and activation evidence establish the actor, response, and deadline. An activated duty still binding at its deadline is active-due; before then it is pending. Missing, ambiguous, or conflicting policy yields uncertainty, not absence. Established absence of an applicable duty yields N.

3. Is the relevant process fully observed (W)? A cancellation may occur without appearing in the log. Check that relevant channels and causal dependencies are recorded, event order is reliable, and state visibility is established. The deadline must be known and recording must extend through it. W is true (a certified window) if all six hold, false if any fails, and unknown otherwise.

4. Was the duty fulfilled, and did the response remain valid (D)? A cancellation request alone does not establish fulfillment. Check for a response from the responsible or authorized actor, linked to the duty and matching its key and scope. It must occur after activation and by the deadline. Its effect must succeed and remain valid at the deadline or recording endpoint. These conditions define admissible fulfillment. Once it is established, the duty is marked discharged before judgment (Table A4).

![](images/82dcdcd05d3f0d02e1fe6f6f2b285727cbd2c89c9eaf2a9218d55c23031161a7.jpg)

Figure 3: Finding-level decisions. N: no violation; V: violation; U: undetermined. The discharged branch permits N without a window certificate.  
![](images/9bf23155e825036d701d505dc5e0b62b8c83ce6cc2a367c989784627473a9764.jpg)  
Figure 4: An evaluated pair becomes indistinguishable after removing rollback and validity evidence, then separates after restoration. N/V label the full executions.

A valid response can establish fulfillment. Establishing non-fulfillment also requires observing where and when a response could occur. Figure 3 therefore checks established fulfillment before requiring a full observation window. The complete rule table is in Appendix A.1.

In words, V requires an established problem, a binding duty that is due, complete relevant observation, and no response that qualifies as fulfillment:

$$
v _ { F } = \mathsf { V } \quad \Longleftrightarrow \quad \Longleftrightarrow \quad C = \mathsf { p r e s e n t } \wedge O = \mathsf { a c t i v e \_ d u e } \wedge W = \mathsf { T } \wedge D \in \{ \mathsf { a b s e n t } , \mathsf { i n a d m i s s i b l e } \} \cdot \mathrm { ( 1 ) }
$$

At trace level, judgments are aggregated under $P \colon$ any confirmed violation yields V. With no confirmed violation, any unresolved applicable finding yields U; otherwise the trace receives N. Timely cancellation fulfills the duty without erasing the earlier stale use. Late cancellation establishes recovery, not timely fulfillment.

## 3.2 INFORMATION NEEDED FOR A JUDGMENT

The cancellation example requires a response whose effect lasts. AgentWebBench pair pair awb 005 isolates the same requirement for repair: restore TLS 1.3 and maintain it through the deadline (Figure 4). Both executions record the same patch and remediation token. A rollback invalidates A’s repair (V); B’s remains effective (N). Keeping only the contract and initial repair leaves identical inputs. Restoring active at deadline, the evidence-derived field recording lasting validity, separates them again.

These executions need different answers but look identical to the judge: an input collision. Formally, sufficiently observed executions have opposite N/V labels but identical retained inputs under the same contract, finding, and deadline. A deterministic judge restricted to these inputs must answer identically, making at least one error. Better reasoning cannot recover the missing distinction.

![](images/2455312ed47ad72feffc1431a9823ca2b3a16478bb0764246e129950c420883e.jpg)  
Figure 5: CAVERT recovers execution evidence, applies consistency rules, and optionally gates uncommitted writes. The gate refreshes state before a new action proposal (Section 4.3).

RQ1 tests which evidence preserves it, independently of CAVERT (Section 5.1; formal criterion in Appendix A.1).

## 4 APPLYING THE DEFINITION WITH CAVERT

The definition specifies the evidence required for judgment, but existing logs do not always expose these relationships. CAVERT recovers supported relationships through semantic extraction and source checks, then applies the same deterministic rules. Its findings guide diagnosis, recovery prompts, and pre-commit intervention (Figure 5).

## 4.1 FROM LOGS TO CHECK STATES

Given a log and contract P, two LLM stages recover the evidence needed for judgment. Scope filtering selects relevant messages, memory operations, and tool records. Causal-state reasoning links states to recipients, dependent actions, and response effects.

For procurement, records link the revocation’s receipt to the later order’s use of the old approval (C). The contract supplies the cancellation duty and deadline (O). Capture metadata and adapter checks establish coverage (W); the cancellation’s outcome and lasting effect establish fulfillment (D). The adapter retains source-event links and updates the duty’s status once fulfillment is established. Unresolved relationships or coverage remain unknown.

LLM interpretation supplies semantic relationships; source checks verify record references (Appendix A.3.1). Structured-trace checks read explicit relationship fields (Appendix A.4.1). Both use the same deterministic verdict rules.

## 4.2 APPLYING THE VERDICT RULES

The validator follows Figure 3 without LLM calls and aggregates verdicts as in Section 3.1. Model and call budgets appear in Appendix B.2.

Each finding reports its verdict, responsible agent, unresolved evidence, and source links (Appendix A.3). To avoid double-counting a duty, CAVERT groups findings only when they share an obligation, compatible keys and scopes, and a verified connecting path. The most specific upstream cause is primary; downstream consequences are secondary.

## 4.3 FROM DIAGNOSIS TO RECOVERY

The Core gate checks proposed writes before commitment. In procurement, it blocks an order using superseded approval and identifies the state to refresh through a public read before reproposal. Intervention uses the dependency finding before the deadline, without awaiting a trace-level violation.

Table 1: Information removal merges oppositely labeled executions.
<table><tr><td>Evidence factor</td><td></td><td>Removal: merged Restoration: separated</td></tr><tr><td>Consumer receipt</td><td>69/80</td><td>69/69</td></tr><tr><td>Action dependency</td><td>65/79</td><td>63/65</td></tr><tr><td>Actor authorization</td><td>68/81</td><td>65/68</td></tr><tr><td>Historical-version eligibility</td><td>57/76</td><td>55/57</td></tr><tr><td>Repair validity at deadline</td><td>69/82</td><td>69/69</td></tr><tr><td>Total</td><td>328/398</td><td>321/328</td></tr></table>

Counts combine three benchmarks. Removal is relative to opposite-label pairs; restoration is relative to merged pairs. Full-view collisions: 4/398. Source breakdowns: Table A7.

Supplementary AppWorld recovery blocks writes to a predefined application until a separate read from that application succeeds. State postconditions and official task success are scored afterwards (Appendix E.4).

The gate assumes uncommitted intercepted writes, side-effect-free rejection, and an appropriate public read. Committed effects require a separate compensation mechanism.

## 5 EXPERIMENTS

Our evaluation separates the definition from its realization. RQ1 tests whether the proposed relationships are necessary for judgment; RQ2 whether CAVERT can recover and apply them in existing logs; RQ3 whether the resulting findings support intervention. Formal checks and contract-query results appear in Appendices A.2 and D.

## 5.1 RQ1: VALIDITY AND NECESSITY OF THE DEFINITION

Execution relationships carry distinctions needed for judgment. We remove and restore five evidence factors: consumer receipt, action dependency, actor authorization, historical-version eligibility, and repair validity at the deadline. The TLS rollback pair illustrates the last factor: the repair duty stays fixed while evidence of lasting repair is removed and restored (Figure 4).

The study contains 450 pairs from AgentWebBench, MARBLE, and MECoBench (Zhong et al., 2026; Zhu et al., 2025; Liu et al., 2026): 30 per factor per benchmark across 93 task or episode clusters. Each pair shares a contract and query. Blinded human annotation gives 398 opposite-N/V pairs, 38 same-label pairs, and 14 containing U (κ = 0.842). Removal and restoration keep these full-execution labels fixed (Appendix C).

Among the 398 opposite-N/V pairs, 328 have identical retained inputs after removal (82.4%), compared with 4 (1.0%) before removal. Restoration makes 321 of those 328 input pairs distinguishable (97.9%; Table 1). Task-cluster 95% intervals are [78.1%, 86.2%] and [95.8%, 99.1%], respectively. A judge given identical inputs cannot distinguish the opposite labels; restoring the missing relationships recovers that distinction. We next examine whether the defined violations occur in natural executions.

Natural logs reveal recurring execution-consistency problems. Independent annotation of 360 trajectories (120 each from TraceElephant, AFTraj, and MATM) yields 43 V, 288 N, and 29 U. Among the 43 violating traces, unsupported assumption occurs in 29 (67.4%) and evidence status escalation in 8 (18.6%). Both strengthen downstream claims beyond their supporting evidence, giving concrete targets for causal-semantic checks. Multi-label counting gives 47 trace–mechanism associations across six primary mechanisms (Appendix C.2). This distribution describes the sampled sources, including task-failure-only TraceElephant logs.

Together, these studies show that receipt, use, and valid fulfillment carry distinctions that shared content alone does not preserve. This establishes their necessity, but not whether implicit relationships can be recovered reliably from existing logs. RQ2 tests that gap.

(b) Recode B  
Table 2: Pooled diagnosis on 720 executions; macro-F1 averages N/V/U equally.
<table><tr><td>Metric</td><td>Generic LLM</td><td>Contract LLM</td><td>Rules</td><td>CAVERT</td></tr><tr><td>Accuracy (%)</td><td>64.2</td><td>80.6</td><td>78.9</td><td>87.1</td></tr><tr><td>Macro-F1 (%)</td><td>43.5</td><td>74.8</td><td>73.8</td><td>83.5</td></tr></table>

![](images/904974dce0e16ffbcfc74c1511ddff8135dd9b1ffe975b79d3f5b4991fa09b61.jpg)

(c) Missing evidence
<table><tr><td rowspan=1 colspan=1>Original</td><td rowspan=1 colspan=1>L1</td><td rowspan=1 colspan=1>L2</td></tr><tr><td rowspan=1 colspan=1>n=18</td><td rowspan=1 colspan=1>n=38</td><td rowspan=1 colspan=1>n=76</td></tr><tr><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>100.0</td><td rowspan=1 colspan=1>100.0</td></tr><tr><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>31.6</td><td rowspan=1 colspan=1>43.4</td></tr><tr><td rowspan=1 colspan=1>16.7</td><td rowspan=1 colspan=1>34.2</td><td rowspan=1 colspan=1>42.1</td></tr><tr><td rowspan=1 colspan=1>11.1</td><td rowspan=1 colspan=1>21.1</td><td rowspan=1 colspan=1>32.9</td></tr><tr><td rowspan=1 colspan=3>0          50         100Assertions on gold-U (%)</td></tr></table>

Figure 6: Diagnosis under log transformations (180 traces). (a,b) Lines join prediction-change rates and net accuracy losses, not confidence bounds. (c) N/V answer rates on reference-U cases; denominators are shown. Lower is better. Full results: Table A12.

## 5.2 RQ2: CROSS-SOURCE GENERALIZATION AND ROBUSTNESS

The evidence requirements support diagnosis from existing logs. We next apply the criteria to 720 executions from MARBLE, CoffeeBench (Sugiura et al., 2026), and MECoBench: four execution models and 60 runs per benchmark–model setting. Diagnostic configurations stay fixed across sources. Generic and Contract LLM use DeepSeek-V4-Flash; CAVERT uses Qwen3.5-Flash filtering, DeepSeek extraction, and deterministic judgment (Appendix B).

CAVERT reaches 87.1% accuracy versus 80.6% for Contract LLM and 78.9% for Rules (Table 2). It leads in accuracy and macro-F1 across all 12 settings (Table A12), supporting the criteria’s use across the tested sources.

Because call structures differ, we also compare under a common \$0.0120 per-execution spending limit. CAVERT exceeds Contract LLM + review by 3.47 pp accuracy and 0.049 macro-F1; both paired 95% intervals exclude zero. Mean API costs are \$0.0017 versus \$0.0031 (Appendix D.4). The advantage persists when the baseline can review and revise within the same spending limit.

Format and missing evidence. We next test verdict stability under recoding and judgment under reduced evidence. A separate panel gives 180 traces their original view plus four transformations (900 inputs). Recode A renames identifiers and reorders JSON keys; Recode B uses a line-based format with causal pointers. Both preserve relationships and labels. Missing L1 removes delivery confirmations and handshake receipts; L2 also removes pre-deadline invalidations and intermediate action confirmations. Unlike RQ1, reference labels and completeness flags are reassessed from the remaining evidence (Appendix D.2).

CAVERT leads all three baselines in accuracy and macro-F1 across all five views (Table A12). All its predictions remain unchanged under Recode A; 146/180 remain unchanged under Recode B, versus 118/180 for Rules (Figure 6). The 18 correct-to-wrong and 16 wrong-to-correct transitions yield a 1.1 pp accuracy decline. Individual verdicts remain sensitive to format despite the small net change.

Under Missing L2, CAVERT reaches 66.1% accuracy versus 57.2% for Rules. It nevertheless assigns N to 25/76 reference-U and 18/52 reference-V cases: the former are unsupported no-violation judgments; the latter miss violations supported by retained evidence (Appendix D.3). Diagnosis alone does not establish practical value; RQ3 asks whether these findings improve action under fixed intervention authority.

![](images/8734431082c6ff8a146aeea52c8cc16d113b6008d55c6ae4a127e53f51e725fe.jpg)  
Figure 7: Recovery on 180 matched blocks per benchmark. G: shared gate. Gains are CAVERT+G minus Rules+G before rounding (Table A19).

## 5.3 RQ3: PRACTICAL BENEFITS AND INTERVENTION COST

Consistency findings guide recovery. After evaluating diagnosis, we test its use in pre-commit intervention. In procurement, a stale-approval finding directs the agent to refresh the approval state before revising its proposed order. Comparing CAVERT with Rules under the same gate tests the value of this guidance.

We use 720 matched fault blocks, 180 each from AppWorld (Trivedi et al., 2024), LongCLI (Feng et al., 2026b), MARBLE, and CoffeeBench (Figure 7). Within each block, arms share initial states, action budgets, output ceilings, and timeouts. Aggregate scores equally weight benchmark-specific normalized endpoints (Appendix E.1).

With the same gate, CAVERT improves on Rules+G in all four environments. AppWorld success rises from 89/180 to 122/180 (+18.3 pp), and LongCLI repair without regressions from 51/180 to 75/180 (+13.3 pp). Consistency findings therefore improve recovery with intervention authority held fixed.

We also compare complete systems with their own intervention mechanisms. CAVERT-Gate exceeds MemTX by 8.75 pp in mean normalized score (95% CI [6.39,11.11]) on the same blocks. All four benchmark intervals are positive (Table A20).

The benefit of gating depends on diagnostic guidance. Holding continuation length fixed, we test three diagnostic methods with prompting and with gating. Switching to gating adds 17.7 pp with CAVERT and 8.6 pp with Rules. The difference between these gains is 9.1 pp (95% CI [4.8,13.4], p < 0.001; Table A22). Thus, the same gate is more effective when guided by consistency findings.

Intervention should also preserve normal-task performance. On 480 blocks without injected faults, the CAVERT-minus-Native score difference is −0.7 pp (95% CI [−2.1, +0.7]). The lower bound exceeds the predeclared −2.5 pp loss margin, supporting aggregate non-inferiority (Table A23).

Recovery incurs computational overhead. On a separate long-horizon cohort, CAVERT+G adds 24.6% tokens and 5.0 seconds per task over Native. Supplementary AppWorld/LongCLI billing averages \$0.042 versus Native’s \$0.030 per task, including cache discounts (Table A28).

RQ1 identifies evidence needed to distinguish fulfillment from violation; RQ2 and RQ3 show how it supports diagnosis and corrective action. Together, the results connect evidence sufficiency to judg ment and intervention in multi-agent systems. This progression matters because the same evidence that makes execution behavior judgeable also makes failures actionable before commitment.

## 6 CONCLUSION

Correct memory is not enough for consistent execution. Identical records can conceal different taskduty outcomes, and no downstream judge can recover distinctions the record failed to preserve. We formalize this gap as execution consistency and identify the evidence needed to judge it. CAVERT shows that these requirements support diagnosis and corrective action. The broader design question is therefore not only whether memory is correct, but whether it preserves enough evidence to judge execution. Reliable multi-agent memory should preserve not only what agents know, but how shared information was received, used, and acted upon.

## REPRODUCIBILITY STATEMENT

Appendix A specifies the semantics and evidence schema. Supplementary material provides an implementation snapshot, extraction prompts, an executable example, and 432 constructed executions with reference labels. Paired intervals preserve the stated clustering units. Appendices B and E specify configurations.

## ETHICS STATEMENT

This work studies execution consistency, diagnosis, and recovery in multi-agent systems using research benchmarks, controlled executions, and model-generated or benchmark-provided traces. We do not collect new human-subject, personal, or sensitive data. Our experiments evaluate task-level execution behavior and system interventions rather than decisions about individuals or demographic groups. Recovery mechanisms operate within the evaluated benchmark environments and are not designed for deployment in consequential real-world settings. We use benchmark data and models for research evaluation and report the evidence requirements, intervention scope, and experimental protocols needed to interpret our results. We follow the ICLR Code of Ethics and aim to support transparent and reproducible evaluation of multi-agent systems.

## AI USE STATEMENT

Generative-AI tools assisted code, writing, and figure preparation. Reference labels for the information-pair and diagnostic studies were independently human annotated. The authors are responsible for experimental records, references, and final claims.

## REFERENCES

Peter Bailis, Alan Fekete, Michael J. Franklin, Ali Ghodsi, Joseph M. Hellerstein, and Ion Stoica. Coordination avoidance in database systems. Proceedings of the VLDB Endowment, 8(3):185– 196, 2014. URL https://www.vldb.org/pvldb/vol8/p185-bailis.pdf.

Matteo Baldoni, Cristina Baroglio, Amit K. Chopra, and Munindar P. Singh. Composing and verifying commitment-based multiagent protocols. In Proceedings of the Twenty-Fourth International Joint Conference on Artificial Intelligence, pp. 10–17, 2015. URL https://www.ijcai. org/Proceedings/15/Papers/009.pdf.

Valter Balegas, Sergio Duarte, Carla Ferreira, Rodrigo Rodrigues, Nuno Preguic¸a, Mahsa Na-´ jafzadeh, and Marc Shapiro. Putting consistency back into eventual consistency. In Proceedings of the Tenth European Conference on Computer Systems, 2015. URL https://www.dpss. inesc-id.pt/<sub>˜</sub>rodrigo/indigo\_eurosys15.pdf.

Shraddha Barke, Arnav Goyal, Alind Khare, Avaljot Singh, Suman Nath, and Chetan Bansal. AgentRx: Diagnosing AI agent failures from execution trajectories. arXiv preprint arXiv:2602.02475, 2026. URL https://arxiv.org/abs/2602.02475.

Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. τ<sup>2</sup>-Bench: Evaluating Conversational Agents in a Dual-Control Environment. arXiv preprint arXiv:2506.07982, 2025. URL https://arxiv.org/abs/2506.07982.

Andreas Bauer, Martin Leucker, and Christian Schallhart. Runtime verification for LTL and TLTL. ACM Transactions on Software Engineering and Methodology, 20(4):1–64, 2011. doi: 10.1145/ 2000799.2000800.

Hongliu Cao, Ilias Driouich, and Eoin Thomas. Beyond task completion: Revealing corrupt success in LLM agents through procedure-aware evaluation. arXiv preprint arXiv:2603.03116, 2026. URL https://arxiv.org/abs/2603.03116.

Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei A. Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? In Advances in Neural Information Processing Systems, volume 38, 2025. doi: 10.52202/085713-4082.

Edward Y. Chang and Longling Geng. SagaLLM: Context management, validation, and transaction guarantees for multi-agent LLM planning. Proceedings of the VLDB Endowment, 18(12):4874– 4886, 2025. doi: 10.14778/3750601.3750611.

Mengzhuo Chen, Junjie Wang, Fangwen Mu, Yawen Wang, Zhe Liu, Huanxiang Feng, and Qing Wang. Seeing the whole elephant: A benchmark for failure attribution in LLM-based multi-agent systems. arXiv preprint arXiv:2604.22708, 2026. URL https://arxiv.org/abs/2604. 22708.

Yu Feng, Nathaniel Weir, Kaj Bostrom, Sam Bayless, Darion Cassel, Sapana Chaudhary, Benjamin Kiesl-Reiter, and Huzefa Rangwala. VeriCoT: Neuro-symbolic chain-of-thought validation via logical consistency checks. In International Conference on Learning Representations, 2026a. URL https://openreview.net/forum?id=zHuV3Vatov.

Yukang Feng, Jianwen Sun, Zelai Yang, Jiaxin Ai, Chuanhao Li, Zizhen Li, Fanrui Zhang, Kang He, Rui Ma, Jifan Lin, Jie Sun, Yang Xiao, Sizhuo Zhou, Wenxiao Wu, Yiming Liu, Pengfei Liu, Yu Qiao, Shenglin Zhang, and Kaipeng Zhang. LongCLI-Bench: A preliminary benchmark and study for long-horizon agentic programming in command-line interfaces. arXiv preprint arXiv:2602.14337, 2026b. URL https://arxiv.org/abs/2602.14337.

Nicoletta Fornara. Specifying and monitoring obligations in open multiagent systems using semantic web technology. In Semantic Agent Systems, pp. 25–45. Springer Berlin Heidelberg, 2011. doi: 10.1007/978-3-642-18308-9 2.

Romain Froger, Pierre Andrews, Matteo Bettini, Amar Budhiraja, Ricardo Silveira Cabral, Virginie Do, Emilien Garreau, Jean-Baptiste Gaya, Hugo Laurenc¸on, Maxime Lecanu, Kunal Malkan, Dheeraj Mekala, Pierre Menard, Gerard Moreno-Torres Bertran, Ulyana Piterbarg, Mikhail´ Plekhanov, Mathieu Rita, Andrey Rusakov, Vladislav Vorotilov, Mengjue Wang, Ian Yu, Amine Benhalloum, Gregoire Mialon, and Thomas Scialom. Gaia2: Benchmarking LLM agents on´ dynamic and asynchronous environments. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/ hash/c26a67e0470774df98c12480ec5d2d7b-Abstract-Conference.html.

Maurice P. Herlihy and Jeannette M. Wing. Linearizability: A correctness condition for concurrent objects. ACM Transactions on Programming Languages and Systems, 12(3):463–492, 1990. doi: 10.1145/78969.78972.

Yuanzhe Hu, Yu Wang, and Julian McAuley. Evaluating memory in LLM agents via incremental multi-turn interactions. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files/paper/2026/hash/ fd1eff9dd295df50a41f2521942fa31d-Abstract-Conference.html.

Sajjad Khan. Verified detection and prevention of concurrency anomalies in multi-agent large language model systems. arXiv preprint arXiv:2606.17182, 2026. URL https://arxiv.org/ abs/2606.17182.

To Eun Kim, Xuhong He, Dishank Jain, Ambuj Agrawal, Negar Arabzadeh, and Fernando Diaz. Multi-agent transactive memory. arXiv preprint arXiv:2606.19911, 2026. URL https: //arxiv.org/abs/2606.19911.

Ido Levy, Ben Wiesel, Sami Marreed, Alon Oved, Avi Yaeli, and Segev Shlomov. ST-WebAgentBench: A benchmark for evaluating safety and trustworthiness in web agents. In International Conference on Learning Representations, 2026. URL https://openreview. net/forum?id=MuCDzH0ctf.

Xiaoyang Li, Yiqi Wang, Haohui Lu, Zhi Chen, Mo Li, Pingan Song, Mingkai Zheng, and Taotao Cai. MemTX: Transactional belief commit for stateful agent memory. arXiv preprint arXiv:2607.23929, 2026. URL https://arxiv.org/abs/2607.23929.

Qingyun Liu, Jiwen Zhang, Jingyi Hu, Siyuan Wang, and Zhongyu Wei. MECoBench: A systematic study of multimodal agent collaboration in embodied environments. arXiv preprint arXiv:2606.31966, 2026. URL https://arxiv.org/abs/2606.31966.

Wyatt Lloyd, Michael J. Freedman, Michael Kaminsky, and David G. Andersen. Don’t settle for eventual: Scalable causal consistency for wide-area storage with COPS. In Proceedings of the Twenty-Third ACM Symposium on Operating Systems Principles, 2011. URL https://www. cs.princeton.edu/<sub>˜</sub>mfreed/docs/cops-sosp11.pdf.

Ming Ma, Jue Zhang, Fangkai Yang, Yu Kang, Qingwei Lin, Saravan Rajmohan, and Dongmei Zhang. DoVer: Intervention-driven auto debugging for LLM multi-agent systems. In International Conference on Learning Representations, 2026. URL https://openreview.net/ forum?id=mrEK16Jy6h.

Erwan Mahe, Boutheina Bannour, Christophe Gaston, Arnault Lapitre, and Pascale Le Gall. Dealing with observability in interaction-based offline runtime verification of distributed systems. arXiv preprint arXiv:2212.09324, 2022. URL https://arxiv.org/abs/2212.09324.

Ashok U. Mallya and Munindar P. Singh. An algebra for commitment protocols. Autonomous Agents and Multi-Agent Systems, 14(2):143–163, 2007. doi: 10.1007/s10458-006-7232-1.

Yanki Margalit, Nurit Cohen-Inger, Erni Avram, Ran Taig, and Oded Margalit. Governed shared memory for multi-agent LLM systems. arXiv preprint arXiv:2606.24535, 2026. URL https: //arxiv.org/abs/2606.24535.

Zhe Ren, Yibo Yang, Yimeng Chen, Zijun Zhao, Benshuo Fu, Zhihao Shu, Bingjie Zhang, Yangyang Xu, Dandan Guo, and Shuicheng Yan. GateMem: Benchmarking memory governance in multiprincipal shared-memory agents. arXiv preprint arXiv:2606.18829, 2026. URL https:// arxiv.org/abs/2606.18829.

Marc Shapiro, Nuno Preguic¸a, Carlos Baquero, and Marek Zawirski. Conflict-free replicated data types. In Stabilization, Safety, and Security ofDistributed Systems, pp. 386–400. Springer Berlin Heidelberg, 2011. doi: 10.1007/978-3-642-24550-3 29.

Munindar P. Singh. An ontology for commitments in multiagent systems: Toward a unification of normative concepts. Artificial Intelligence and Law, 7(1):97–113, 1999. doi: 10.1023/a: 1008319631231.

Issa Sugiura, Daichi Hattori, Kazuo Araragi, Keita Ogawa, Shota Onose, Taro Makino, Teppei Usuki, and Takashi Ishida. CoffeeBench: Benchmarking long-horizon LLM agents in heterogeneous multi-agent economies. arXiv preprint arXiv:2606.16613, 2026. URL https: //arxiv.org/abs/2606.16613.

Harsh Trivedi, Tushar Khot, Mareike Hartmann, Ruskin Manku, Vinty Dong, Edward Li, Shashank Gupta, Ashish Sabharwal, and Niranjan Balasubramanian. AppWorld: A controllable world of apps and people for benchmarking interactive coding agents. In Proceedings ofthe 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), pp. 16022– 16076, 2024. doi: 10.18653/v1/2024.acl-long.850.

Werner Vogels. Eventually consistent. Communications of the ACM, 52(1):40–44, 2009. doi: 10.1145/1435417.1435432.

Sergey Volkov, Yang Li, and Ye Luo. StateFuse: Deterministic conflict-preserving memory for multi-agent systems. arXiv preprint arXiv:2607.05844, 2026. URL https://arxiv.org/ abs/2607.05844.

Ziming Wang. TOKI: A bitemporal operator algebra for contradiction resolution in LLM-Agent persistent memory. arXiv preprint arXiv:2606.06240, 2026. URL https://arxiv.org/ abs/2606.06240.

Feiyu Wu, Xu Zheng, Yue Qu, Zhuocheng Wang, Zicheng Feng, and Hui Li. Grounding generative planners in verifiable logic: A hybrid architecture for trustworthy embodied AI. In International Conference on Learning Representations, 2026. URL https://openreview.net/forum? id=wb05ver1k8.

Shuhan Xue, Zixin Ding, Yichen Shen, Yinjie Wang, Zhenfei Yin, Yingcheng Wu, Yuxin Chen, Mengdi Wang, and Ling Yang. PAST-Bench: Benchmarking the foundations of recursive selfimprovement in personal agents. arXiv preprint arXiv:2608.04003, 2026. URL https:// arxiv.org/abs/2608.04003.

Tao Yu, Hao Wang, Changyu Li, Shenghua Chai, Minghui Zhang, Zhongtian Luo, Yuxuan Zhou, Haopeng Jin, Zhaolu Kang, Jiabing Yang, YiFan Zhang, Xinming Wang, Hongzhu Yi, Zheqi He, Jing-Shu Zheng, Xi Yang, Yan Huang, and Liang Wang. Beyond the all-in-one agent: Benchmarking role-specialized multi-agent collaboration in enterprise workflows. arXiv preprint arXiv:2605.08761, 2026a. URL https://arxiv.org/abs/2605.08761.

Zhongming Yu, Naicheng Yu, Hejia Zhang, Wentao Ni, Mingrui Yin, Jiaying Yang, Yujie Zhao, and Jishen Zhao. Multi-agent memory from a computer architecture perspective: Visions and challenges ahead. arXiv preprint arXiv:2603.10062, 2026b. URL https://arxiv.org/ abs/2603.10062.

Boxuan Zhang, Jianing Zhu, Zeru Shi, Dongfang Liu, and Ruixiang Tang. AgentForesight: Online auditing for early failure prediction in multi-agent systems. arXiv preprint arXiv:2605.08715, 2026a.

Guibin Zhang, Junhao Wang, Junjie Chen, Wangchunshu Zhou, Kun Wang, and Shuicheng Yan. AgenTracer: Who is inducing failure in the LLM agentic systems? In International Conference on Learning Representations, 2026b. URL https://openreview.net/forum?id= l05DseqvuD.

Shaokun Zhang, Ming Yin, Jieyu Zhang, Jiale Liu, Zhiguang Han, Jingyang Zhang, Beibin Li, Chi Wang, Huazheng Wang, Yiran Chen, and Qingyun Wu. Which agent causes task failures and when? On automated failure attribution of LLM multi-agent systems. In Proceedings of the 42nd International Conference on Machine Learning, volume 267, pp. 76583–76599, 2025. URL https://proceedings.mlr.press/v267/zhang25cq.html.

Shanshan Zhong, Kate Shen, and Chenyan Xiong. AgentWebBench: Benchmarking multi-agent coordination in agentic web. arXiv preprint arXiv:2604.10938, 2026. URL https://arxiv. org/abs/2604.10938.

Kunlun Zhu, Hongyi Du, Zhaochen Hong, Xiaocheng Yang, Shuyi Guo, Zhe Wang, Zhenhailong Wang, Cheng Qian, Xiangru Tang, Heng Ji, and Jiaxuan You. MultiAgentBench: Evaluating the collaboration and competition of LLM agents. In Proceedings ofthe 63rd Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pp. 8580–8622, 2025. doi: 10.18653/v1/2025.acl-long.421.

## APPENDIX CONTENTS

A Semantics and Implementation 15   
A.1 Evidence and decision rules 15   
A.2 Mechanism registry 16   
A.3 Trace schema and adapters 17   
A.4 Obligations and effective discharge 22   
A.5 Relation to distributed consistency 23   
A.6 Relation to agent diagnosis and verification 23   
B Shared Evaluation Protocol 23   
B.1 Annotation, metrics, and uncertainty 23   
B.2 Diagnostic and execution configurations 24   
B.3 Semantic extraction diagnostic probe 25   
C Definition Evaluation Details 25   
C.1 Paired design and information removal 25   
C.2 Mechanism occurrence in natural logs 26   
D Diagnosis and Robustness Details 27   
D.1 Contract-query diagnosis (E2) 27   
D.2 Cross-model and multiview diagnosis (E3) 28   
D.3 Prediction stability and evidence sufficiency 30   
D.4 Diagnosis under a fixed spending limit 31   
E Execution Protocol and Stratification 32   
E.1 Matched blocks and executor configurations 32   
E.2 Recovery controls and outcome definitions 34   
E.3 Normal-task impact and budget sensitivity 34   
E.4 Supplementary cohorts, outcomes, and costs 36

## A SEMANTICS AND IMPLEMENTATION

## A.1 EVIDENCE AND DECISION RULES

## A.1.1 EXECUTION RECORD AND INFORMATION LOSS

The execution record is

$$
T = \langle E , \mathrm { P r e c , C a u s a l , V i s , S u p } , P , \mathrm { C a p t u r e } , H \rangle .\tag{2}
$$

Here E denotes events, Prec reliable precedence, and Causal typed dependencies. Visibility $\mathrm { V i s } ( i , a , j )$ records whether agent a could observe event i before event $j .$ . Supersession Sup and contract $P$ determine which states and effects remain eligible for use. Capture records which channels and segments were captured. The observation horizon H is the point where recording ends. Records are compared using the semantic key $K ( e ) = \langle \mathrm { e n t i t y } , \mathrm { a t t r i b u t e } , \mathrm { s c o p e } \rangle$

```latex
Retained-record indistinguishability
Fix a public contract $P ,$ a target finding $F ,$ and an evaluation horizon h. The recorded horizon H does not
guarantee coverage through h. Let $q ( \dot { T } )$ retain a specified subset of execution information. Let
$\overline { { \ell } } _ { P } ( F , T , h )$ be the evidence-supported reference judgment. Suppose sufficient evidence establishe
opposite labels $\ell _ { 0 } , \ell _ { 1 } \in \{ \mathsf { N } , \mathsf { V } \}$ under the same $( \check { P } , \check { F } , h )$ , but $\hat { q } ( T _ { 0 } ) = q ( T _ { 1 } )$ . Any deterministic judge g
restricted to that input then satisfies
$\sum _ { j = 0 } ^ { 1 } \mathbf { 1 } [ g ( P , F , h , q ( T _ { j } ) ) \neq \ell _ { j } ] \geq 1 .$ (3)
Identical inputs require identical predictions, matching at most one label. The criterion tests whether a
representation preserves verdict-relevant distinctions.
```

## A.1.2 EVIDENCE DOMAIN

Every primitive queried by the semantics is either observed in the raw execution, reproducibly derived by a versioned adapter rule, or unknown. Observed and derived values can evaluate to true or false; unknown evidence evaluates to U. The window certificate uses strong Kleene conjunction to combine these values. Any false requirement makes the certificate false. If none is false but one is unknown, the certificate is unknown. It is true only when all requirements are true. The finding verdict then follows the ordered finite-trace monitor below. An uncertified window leaves missing discharge records insufficient to establish non-fulfillment. Earlier decisive rules can still establish N without evaluating the window.

For the truth ordering $\mathsf { F } < \mathsf { U } < \mathsf { T } .$

$$
x \wedge y = \operatorname* { m i n } ( x , y ) , \qquad \neg \mathsf { T } = \mathsf { F } , \quad \neg \mathsf { F } = \mathsf { T } , \quad \neg \mathsf { U } = \mathsf { U } .
$$

## A.1.3 VISIBILITY, REACHABILITY, AND FRONTIER

VisAt is the relation Vis in Eq. (2), evaluated for an agent before an event. True visibility requires one of three grounds: a produced i, an active access-control rule includes a, or explicit delivery makes i visible before j. An explicit exclusion yields F only with complete delivery capture through j and no intervening delivery to a. Otherwise visibility is unknown. An empty or missing accesscontrol list is never interpreted as globally visible. Typed causal reachability is computed over data dependency, control dependency, handoff delivery, evidence support, supersession, and resolution edges. Temporal adjacency or repeated speech by one agent does not create a causal edge. The typed edges in Causal record execution relationships; Sup records which earlier states have retired. They are not interchangeable: a dependency or delivery edge need not retire a state.

A frontier contains states still eligible for use: recorded replacements, retractions, or resolutions have not retired them. For trigger j and semantic key k, the global pre-trigger frontier is

$$
\operatorname { F r o n t i e r } _ { p r e } ( j , k ) = \{ i \mid i \prec j , K ( i ) \sim k , \ : \emptyset r : i \prec r \prec j \land \operatorname { R e t i r e s } ( r , i ) \} .\tag{4}
$$

Retirement requires an explicit supersession, retraction, version replacement, or effective resolution. When retirement is unknown, frontier membership is unknown. The agent-visible frontier uses only

Table A1: Ordered stage states and their finding verdicts. Decisive earlier stages short-circuit later checks.
<table><tr><td>Candidate</td><td>Obligation</td><td>Window</td><td>Discharge</td><td>Final</td></tr><tr><td>Absent</td><td>Not evaluated</td><td>Not evaluated</td><td>Not applicable</td><td>N</td></tr><tr><td>Unknown</td><td>Not evaluated</td><td>Not evaluated</td><td>Unknown</td><td>U</td></tr><tr><td>Present</td><td>Absent</td><td>Any</td><td>Not applicable</td><td>N</td></tr><tr><td>Present</td><td>Pending or uncertain</td><td>Any</td><td>Any</td><td>U</td></tr><tr><td>Present</td><td>Discharged</td><td>Any</td><td>Admissible</td><td>N</td></tr><tr><td>Present</td><td>Discharged</td><td>Any</td><td>Not admissible or unknown</td><td>U</td></tr><tr><td>Present</td><td>Active due</td><td>Closed</td><td>Admissible</td><td>N</td></tr><tr><td>Present</td><td>Active due</td><td>Closed</td><td>Inadmissible or absent</td><td>¬V</td></tr><tr><td>Present</td><td>Active due</td><td>Closed</td><td>Unknown or not applicable</td><td>U</td></tr><tr><td>Present</td><td>Active due</td><td>Open or unknown</td><td>Any</td><td>U</td></tr></table>

N: no violation for this mechanism; V: violation; U: undetermined. “Any” means the value cannot change this row’s verdict; “not evaluated” denotes a short-circuited stage, not an unknown truth value.

the agent’s observation history. A retirement changes this frontier only when the agent can see it:

$$
\begin{array} { r l } & { \mathrm { F r o n t i e r } _ { p r e } ^ { a } ( j , k ) = \{ i \mid i \prec j , K ( i ) \sim k , \mathrm { V i s A t } ( i , a , j ) = \mathsf { T } , } \\ & { ~ \nexists r : ~ i \prec r \prec j \wedge \mathrm { R e t i r e s } ( r , i ) \wedge \mathrm { V i s A t } ( r , a , j ) = \mathsf { T } \} . } \end{array}
$$

An unseen retirement does not silently remove a state from a’s view. Unknown visibility or retirement makes any affected membership unknown; downstream predicates may not treat this as established absence. Checkpoint mechanisms use the frontier named by the public contract. A consumer-specific stale-use finding instead requires the replacement to be visible to that consumer before use.

## A.1.4 STAGE TRUTH TABLE AND TRACE AGGREGATION

At trace level, one confirmed finding is sufficient for VIOLATION. If no finding is a violation but any applicable mechanism remains UNDETERMINED, the trace is UNDETERMINED. The trace is NOVIOLATION only when every applicable mechanism has a determined NOVIOLATION verdict, including verdicts established by short-circuiting. Trace aggregation therefore preserves any unknown mechanism that could change the verdict.

A trace-level N concerns the consistency mechanisms applicable under P. The recovery gate enforces read-before-write sequencing for uncommitted writes (Section 4.3); action authorization remains a separate policy requirement.

## A.2 MECHANISM REGISTRY

Table A2 specifies the evidence requirements and exclusions for the twelve mechanisms introduced in Section 3. A required handoff may bind its recipient when it uses the information, while explicitly permitted alternatives need not be reconciled.

Table A2: Required evidence and exclusion conditions for the twelve consistency mechanisms.
<table><tr><td>Mechanism</td><td>Required structural evidence</td><td>Exclusion or boundary</td></tr><tr><td colspan="3">Strict-temporal: consistency at a dependency or reconciliation checkpoint</td></tr><tr><td>Conflicting write</td><td>Visible active frontier write; overlapping key/scope; contradictory committed downstream write</td><td>Labelled alternatives or an explicit resolution are not conflicting writes</td></tr><tr><td>Stale dependency use</td><td>Verified retirement of old state; downstream dependency on it; replacement visible before use</td><td>Explicit version pin, authorized rollback, or historical qualification</td></tr><tr><td>Unresolved contradiction</td><td>Two contradictory active frontier states at a declared reconciliation checkpoint</td><td>Resolution before the checkpoint prevents this checkpoint candidate</td></tr><tr><td colspan="3">Causal-semantic: preservation of responsibility, support, and evidence scope</td></tr><tr><td>Missed handoff Unsupported</td><td>Typed handoff with payload, target, delivery condition, and later target dependency action Failed, partial, unknown, or unverified evidence followed by a stronger dependent</td><td>Preserve, update, qualify, or resolve before the dependency action Independent evidence sufficient for the stronger claim</td></tr><tr><td>assumption Unsupported reconstruction</td><td>commitment Corrupt or unrecoverable source followed by reconstruction presented as observed or</td><td>Independent support or an explicitly hypothetical reconstruction</td></tr><tr><td>Source scope loss</td><td>verified Source scope is known; downstream scope exceeds it; no support covers the expansion Paged, ranked, filtered, windowed, or top-k</td><td>Scope-preserving or explicitly narrowed claim</td></tr><tr><td>Source-view scope loss Evidence status escalation</td><td>view generalized to the global source Failed, partial, unknown, or unverified evidence upgraded to gathered, complete, or verified</td><td>Global completeness is separately established New successful evidence or an auditable verification step</td></tr><tr><td colspan="3">Eventual-semantic: consistency of the effective terminal state</td></tr><tr><td>Unresolved final divergence</td><td>Incompatible task-relevant terminal frontier states remain active at horizon</td><td>Specific upstream mechanism is primary when it explains the same obligation</td></tr><tr><td>Competing final answers</td><td>Conflicting typed final answers with no canonical selection at horizon</td><td>Policy-authorized labelled alternatives can make the obligation non-binding</td></tr><tr><td>Unresolved final evidence state</td><td>Terminal evidence availability, completeness, or verification states disagree and affect output</td><td>New evidence, qualification, or resolution before horizon</td></tr></table>

The 12 × 4 executable suite tests four cases per mechanism: a violation satisfying Eq. (1), a normal execution without the candidate problem, an authorized and timely discharge, and insufficient evidence requiring UNDETERMINED. Discharge preserves an event-time candidate but can prevent a later checkpoint or terminal candidate. These remain distinct findings even when they concern the same key; grouping follows Section 4.2.

## A.3 TRACE SCHEMA AND ADAPTERS

The normalized trace is serialized as external-adapted-trace-1.0. Each record contains a stable trace identifier, source metadata, a capture manifest, capability evidence, and an ordered event array. Diagnosis operates on events and their relations, without preselecting a message pair.

For each finding, CAVERT returns

$$
y _ { F } = \langle m , \pi , C , O , W , D , v _ { F } , e _ { r } , e _ { t } , \rho \rangle ,\tag{5}
$$

where $m$ is the mechanism, π the responsible principal, $e _ { r }$ the root event, $e _ { t }$ the trigger or due-point event, and $\rho$ a typed dependency path. Evidence references and abstention reasons accompany the observed path from state break to affected action. Eq. (1) specifies decisions conditional on these fields; extraction quality is evaluated empirically.

Adapters return both the normalized trace and a capability report. Visibility, semantic keys, obligation force, and causal paths retain their observed or derived support; missing values remain unknown.

Table A3: Public trace fields and their role in evidence construction.
<table><tr><td>Object</td><td>Required fields</td><td>Purpose</td></tr><tr><td>Source</td><td>source_id, framework, framework_version, raw_record_id</td><td>Locks provenance and permits source- and framework-stratified evaluation</td></tr><tr><td>Capture manifest</td><td>adapter_version, channel states, horizon and truncation</td><td>Establishes which negative observations are justified</td></tr><tr><td>Event</td><td>states, raw-log reference event_id, sequence, event_type, actor, content, fields</td><td>Represents messages, memory operations, tools, retrieval, browser views, handoffs, and finalization</td></tr><tr><td>Field evidence</td><td>status, value, provenance, confidence, reason</td><td>Separates observed, derived, and unknown inputs and makes derivations auditable</td></tr><tr><td>State identity</td><td>state_group_id, state_dimension, state_label,</td><td>Supports comparisons between heterogeneous sub-states without inventing a global task key</td></tr><tr><td>Graph evidence</td><td>semantic key/scope typed parents, visibility activation, supersession, resolution, and support links</td><td>Supports causal localization, retirement, and discharge-effect checks</td></tr></table>

Policy comes from explicit task requirements, and observation coverage is checked separately for each finding. Derived fields store the adapter rule and source event identifiers needed for replay.

## A.3.1 EXTRACTION INSTRUCTIONS AND STRUCTURED OUTPUTS

The open-log implementation separates evidence selection, candidate localization, and findingconditioned review within the selection and reasoning workflow of Section 4.1. These operations can involve multiple page-level calls; model assignments and per-call limits are specified in Appendix B.2. The following excerpts are taken from the implemented prompts.

## Evidence selection

STATE ATOM PROMPT instructs: “This is exhaustive evidence indexing, not violation judgment” and “Do not discard weak, failed, tentative, partial, or contradictory states: record their epistemic status.” It retains claims, source evidence, shared writes, corrections, handoffs, dependent actions, and final states. Each returned atom has the form [event id, atom type, semantic label, epistemic status, evidence quote]. Event IDs must come from the supplied page; quotes must be literal source spans, or indexed span IDs resolved by the runtime. An overflow flag requests page splitting when the atom limit is exceeded. Actor identity and source text are restored from the original events.

## Candidate localization

CROSS TRACE PAIRING PROMPT compares the indexed events and instructs: “Do not decide obligation, deadline, discharge, or final violation.” It lists the twelve registered mechanisms and requires localized root, trigger, and path event IDs. Ordinary task failure or an incorrect answer without a supported execution relationship is insufficient. Its JSON response contains

recovered candidates; each candidate names a mechanism, a typed key or cited-span identity, and the supporting events. A candidate is a proposal for checking, not a final verdict.

<table><tr><td>Finding-conditioned review</td></tr><tr><td>FORMAL_V4_TARGETED_PROMPT instructs: “Review the entire supplied page separately for obligation, window, and discharge.&quot; A public instruction or rule must bind the duty; a normally observed final action must establish its deadline. The prompt further specifies: “A search retry, acknowledgement, critique, or</td></tr><tr><td>correction marker without later uptake is not discharge.&quot; For each finding, it returns one finding-reviews entry with obligation, window, and discharge assessments. Each</td></tr><tr><td>assessment contains a status (supported, not_supported_on_page, or uncertain), support event IDs, an initially empty citation list, and a short rationale. The request repeats the public task, completion metadata, and finding anchors; these anchors do not count as evidence that intervening pages</td></tr></table>

The inventory and pairing prompts are defined in

experiments/formal\_v4\_exhaustive\_candidate\_recovery.py; the targeted prompt and output normalization are in

experiments/formal\_v5/semantic\_evidence.py. Their separation prevents the extraction interface from substituting an LLM’s final label for the deterministic judgment.

## A.3.2 FROM RESPONSES TO CHECKED STATES

Normalization and source checks. After JSON parsing, normalize response normalizes candidate identities and path boundaries and applies candidate and predicate JSON schemas. For model-selected event IDs, hydrate citations copies source text into exact quote; the model need not transcribe it. The open-log adapter then checks that the mechanism is registered, root/trigger/support IDs exist, required references are cited, quotes match source records, and path events have valid order. Predicate evidence must match an accepted candidate. Invalid or conflicting items cannot establish the affected state; unresolved evidence is retained rather than treated as absence.

Certifying negative evidence

not supported on page is a page-local search result, not a false predicate by itself. To establish absent discharge, the targeted verifier records all covered event IDs, page counts, completed reviews, and a deadline event. The adapter checks complete coverage of the supplied record, one negative review per page, and an existing deadline; a positive discharge review prevents a negative certificate. An incomplete or uncertain search cannot establish absent discharge. Scanning all supplied text does not establish that an unrecorded execution channel was captured; capture adequacy and task-specific window evidence remain separate requirements.

Field conversion. After these checks, the open-log adapter constructs the following states. “Supported” below means accepted evidence after normalization and audit, not the unverified JSON assertion alone.

<table><tr><td>Field</td><td>Conversion used for judgment</td></tr><tr><td>C</td><td>An accepted, confirmed localized candidate yields present; rejected or unresolved proposals yield unknown. Established absence or refutation is distinguished from missing evidence.</td></tr><tr><td>0</td><td>Supported binding and due obligation yields active_due; established absence yields absent; unresolved binding yields uncertain.</td></tr><tr><td>W</td><td>Accepted finding-window evidence supplies the window certificate and its supporting event references; open or unresolved windows do not become closed merely because extraction finished.</td></tr><tr><td>D</td><td>Supported admissible fulfillment yields admissible; a certified negative search yields absent; otherwise discharge remains unknown. An active-due duty with admissible fulfillment is marked discharged before judgment.</td></tr></table>

SemanticOpenLogV4Validator in

memory\_consistency/v4/semantic\_open\_log.py performs the audit and conversion. final status in memory\_consistency/v4/models.py applies Table A1; it makes no LLM call. Source checks verify reference integrity and structural constraints. They do not prove that every proposed semantic relationship, duty interpretation, or window assessment is correct; extraction quality remains an empirical question (Appendix B.3).

## A.3.3 EXECUTABLE INPUT–OUTPUT EXAMPLE

This constructed example illustrates the complete candidate and predicate response formats and their deterministic processing. The responses are specified fixtures, not sampled model outputs or additional evaluation observations. The trace is fully observed through a normal final checkpoint, with public completion metadata final event id=e4 and capture metadata permitting negativeevidence search.

<table><tr><td colspan="3">Example input: four recorded events</td></tr><tr><td colspan="3">ID Actor / event</td></tr><tr><td>el</td><td></td><td>Recorded content, in order</td></tr><tr><td>e2</td><td>User / instruction Planner / memory</td><td>Use only verified supplier eligibility for the final approval. Supplier eligibility is assumed; verification is unavailable.</td></tr><tr><td>e3</td><td>write Procurement /</td><td>Approved supplier eligibility based on the assumed status.</td></tr><tr><td>e4</td><td>approval Procurement / final checkpoint</td><td>Final approval checkpoint reached; the approval remains active.</td></tr></table>

Candidate response. The proposed unsupported assumption links the assumed state in e2 to its consumption in e3. The cited-span identity uses text present in e2. The complete illustrative response is:

Candidate response JSON fixture   
1 {"recovered\_candidates": [{   
2 "mechanism": "unsupported\_assumption",   
3 "semantic\_key": null,   
4 "semantic\_identity": {   
5 "mode": "cited\_span", "label": "Supplier eligibility",   
6 "event\_ids": ["e2"]   
7 },   
8 "root\_event\_ids": ["e2"], "trigger\_event\_ids": ["e3"],   
9 "causal\_path\_event\_ids": ["e2", "e3"], "citations": []   
10 }]}

Normalization retains the selected events and fills their source citations. For example, the e2 citation becomes event $\mathtt { i d } \mathtt { \Gamma } \mathtt { d } \mathtt { = } \ " \mathrm { e } 2 \mathrm { ~ \ " ~ }$ with exact quote="Supplier eligibility is assumed; verification is unavailable." It does not add a new semantic assertion.

Predicate response. The finding-conditioned review selects e1 for the obligation and e4 for the window; the negative discharge assessment covers this example’s single page. Its complete illustrative response is:

Predicate response JSON fixture   
{"finding\_reviews": [{   
2 "mechanism": "unsupported\_assumption",   
3 "root\_event\_ids": ["e2"], "trigger\_event\_ids": ["e3"],   
4 "predicate\_reviews": {   
5 "obligation": {   
6 "status": "supported", "support\_event\_ids": ["e1"],   
7 "citations": [],   
8 "rationale": "The public instruction requires verified   
eligibility."   
9 },   
10 "window": {   
11 "status": "supported", "support\_event\_ids": ["e4"],   
12 "citations": [],   
13 "rationale": "Normal completion closes the approval duty."   
14 },   
15 "discharge": {   
16 "status": "not\_supported\_on\_page", "support\_event\_ids": [],   
17 "citations": [],   
18 "rationale": "All events reviewed; no propagated repair occurs."   
19 }   
20 }   
21 }]}

Runtime-generated certificate and verdict. The verifier copies source citations for the selected IDs and converts the supported obligation and window to true predicate evidence. It generates the following discharge-search certificate; coverage counts are runtime bookkeeping, not modelprovided verdicts:

```jsonl
Discharge-search certificate Runtime-generated
{"mechanism": "unsupported_assumption",
2 "root_event_ids": ["e2"], "trigger_event_ids": ["e3"],
3 "deadline_event_id": "e4",
4 "covered_event_ids": ["e1", "e2", "e3", "e4"],
5 "page_count": 1, "completed_page_count": 1,
6 "search_conclusion": "no_admissible_discharge",
7 "page_reviews": [{"page_index": 0,
8 "status": "no_admissible_discharge_on_page",
9 "rationale": "All events reviewed; no propagated repair occurs."}]}
```

## Checked states and resulting verdicts

Replaying these fixtures through normalization, TargetedFindingVerifier, and SemanticOpenLogV4Validator gives:

Input condition Checked states / consequence Verdict   
Original fixtures C = present, O = active due, W = T, D = absent V   
Only The negative certificate is invalid; D becomes unknown. U   
completed page count   
changed to zero   
Required candidate quote C becomes unknown. U   
absent from the source

These checks demonstrate the conversion and rejection behavior; they do not measure an LLM’s ability to produce the illustrated responses.

Table A4: Public obligations and their permitted effects. A: policy binding. B: observable conditions for discharge.  
A. Binding policy
<table><tr><td>Claim or policy state</td><td>Default binding interpretation</td></tr><tr><td>Hard constraint, verified</td><td>must_preserve; exploratory context does not relax provenance, source scope,</td></tr><tr><td>fact, or evidence state</td><td>supersession, or evidence-status invariants should_consider only when policy requires justification, resolution, or no</td></tr><tr><td>Recommendation</td><td>unresolved deviation</td></tr><tr><td>Hypothesis or brainstorm Labelled alternatives</td><td>Optional only when exploration is explicit and no invariant is contradicted Non-binding only when the relaxation names the mechanism/scope and</td></tr><tr><td>Missing, ambiguous, or</td><td>declares that no canonical resolution is required Obligation is uncertain; no hidden domain threshold is substituted</td></tr></table>

## B. Permitted discharge

<table><tr><td>Mode</td><td>Required observable effect</td></tr><tr><td>preserve</td><td>Downstream state retains the obligated payload and evidence strength</td></tr><tr><td>update</td><td>Stronger or newer evidence replaces the old state while retaining provenance</td></tr><tr><td>resolve</td><td>Conflicting branches are explicitly retired and one effective state is established</td></tr><tr><td>qualify</td><td>Normative force or scope is narrowed to what the evidence supports</td></tr><tr><td>authorized_defer</td><td>Policy permits delay and the event names owner, condition, and a strictly later deadline</td></tr></table>

All modes require an authorized actor, the same key and scope, a typed causal path, and an effect after activation and no later than the deadline. The effect must remain active at the evaluation frontier.

## A.4 OBLIGATIONS AND EFFECTIVE DISCHARGE

Contracts are fixed before judgment from explicit task instructions and declared policies. Controlled pairs share a case-level specification; external logs use public task metadata or user-instruction messages. Each obligation records its identifier, key and scope, responsible principals, authorized resolvers, activation, deadline, discharge modes, and scoped exceptions. Execution evidence separately establishes agent visibility, activation, and fulfillment. Due status follows these conditions rather than a coarse policy profile.

An obligation is absent when public policy establishes that no relevant responsibility exists. It is pending before its deadline. It is uncertain when a binding fact is unknown. The adapter assigns active due once grounding, target responsibility, activation, and due status are established. Confirming an admissible discharge changes that state to discharged. “Not due” therefore means pending, not absent. The adapter updates O before verdict evaluation. An established due obligation with admissible D is supplied as O = discharged. The final status function reads C, O, W, D without rewriting $O .$ Independently supplied states may instead contain O = active due, $D =$ admissible (Table $\mathbf { A } 1 )$ . This combination requires a closed W. The two rows thus represent different inputs, not a choice of rules for the same normalized evidence. For a present candidate and unknown W, the discharged input yields N; the active-due input yields U.

Later events can revoke a discharge; the permitted effect must still hold at the deadline or horizon (Table A4B).

Early N and persistence evidence   
“Early” denotes rule order, not a judgment made before future effects are known. For a present candidate,   
O = discharged and D = admissible establish N without requiring the full W certificate. Admissibility   
still requires evidence of authorization, matching key and scope, causal connection, timing, and continued   
effect through the evaluation point. A successful repair receipt alone does not establish persistence if   
subsequent capture is incomplete.   
For example, an agent receives a replacement at event 1, uses the old state at event 2, and repairs it at   
event 3 before deadline 9. A lasting authorized repair with complete relevant subsequent records   
establishes discharge. Explicit audience records can establish local visibility while global coverage, and   
hence W, remains unknown. If persistence is unresolved, D is unknown and this branch yields U. If the   
repair is retired by event 9, no alternative response exists, and W is closed, the verdict is V.

## A.4.1 EVIDENCE PATHS AND EVALUATION ROLES

The formal-v4 final status function applies Table A1 to C, O, W, D. E2/E3 recover these states through two LLM stages and score the complete log-to-verdict path (Appendix B.2).

Structured-trace checks read explicit fields instead. Audiences or completed typed delivery establish visibility; supersession and effect-retirement links track response persistence. Persistence requires a known horizon, no truncation, and complete order, causal, memory, tool, and message records. Global visibility coverage is checked separately by W. Missing required links or persistence evidence leave D unknown.

## A.5 RELATION TO DISTRIBUTED CONSISTENCY

Linearizability concerns an object’s sequential specification (Herlihy & Wing, 1990). Two executions can share the same approval-register history yet differ in whether procurement cancels an order after receiving a revocation. A specification covering order actions can enforce that duty. Our judgment concerns its fulfillment, beyond the register history alone.

COPS tracks dependencies before exposing updates (Lloyd et al., 2011). Our causal-semantic contracts additionally check meaning: a correctly ordered handoff can still turn “filtered records” into “all records.” Likewise, eventual convergence (Vogels, 2009) differs from resolving designated task states by a specified deadline. These distinctions concern the specification and evidence, rather than a stronger consistency model over the same object.

Application invariants and semantic checks already support coordination, enforcement, and repair (Bailis et al., 2014; Balegas et al., 2015; Chang & Geng, 2025). Commitment protocols and runtime verification supply obligation lifecycles and three-valued judgments (Singh, 1999; Mallya & Singh, 2007; Fornara, 2011; Bauer et al., 2011). We specify the evidence linking generated handoffs and tool outcomes to these duties, and test its role through controlled pairs. U covers unresolved execution evidence or policy binding.

## A.6 RELATION TO AGENT DIAGNOSIS AND VERIFICATION

DoVer tests failure hypotheses through interventions, while AgenTracer attributes failures to agents and steps (Ma et al., 2026; Zhang et al., 2026b). VIRF checks plans against safety constraints; Veri-CoT checks formalized reasoning; ST-WebAgentBench evaluates policy-compliant task completion (Wu et al., 2026; Feng et al., 2026a; Levy et al., 2026). CAVERT judges task-obligation fulfillment from visibility, dependencies, response effects, and coverage.

## B SHARED EVALUATION PROTOCOL

## B.1 ANNOTATION, METRICS, AND UNCERTAINTY

E1 varies information under a fixed public contract. E2 scores queries registered before method predictions. E3 preserves original labels for lossless views and judges missing-evidence views from what remains. E1 uses two independent annotators with agreement κ = 0.842 and third-party

Table A5: E2–E3 diagnostic and execution configurations.
<table><tr><td>Component</td><td>Model / engine</td><td>Inference configuration</td></tr><tr><td>B3</td><td>deepseek-ai/DeepSeek-V4-Flash-2026-07</td><td> $T = 0 , { \mathrm { t o p } } { \cdot } p = 1 , 2 , 0 4 8$  output tokens; thinking disabled</td></tr><tr><td>B4</td><td>deepseek-ai/DeepSeek-V4-Flash-2026-07</td><td> $T = 0 , \mathrm { t o p } \mathrm { - } p = 1 , 4 , 0 9 6$  output tokens; thinking disabled</td></tr><tr><td>Rules</td><td>PublicContractRules</td><td>Deterministic public-contract checks; unknown required evidence propagates to U</td></tr><tr><td>AgentRx</td><td>AgentRx-Magentic-One-v4.0 + adapter</td><td>Adapter: agentrx-magentic-v4-loss-aware-1.0</td></tr><tr><td>CAVERT</td><td>S1: Qwen3.5-Flash; S2: DeepSeek-V4-Flash</td><td> $\bar { T ^ { } } = 0 , \mathrm { t o p } \mathbf { \bar { \Gamma } } \bar { p } = 1 , 4 , 0 9 6$  output tokens per model calì in each stage; scope filtering</td></tr><tr><td>Execution agent</td><td>deepseek-ai/DeepSeek-V4-Flash-2026-07</td><td>then causal-state inference  $T = 0 . 2 , \mathrm { t o p } \mathrm { - } p = 0 . 9 5 , 4 , 0 9 6$  output tokens; ReAct loop</td></tr></table>

B3/B4 use direct and contract prompts. Diagnostic configurations share public inputs and use the method-specific output ceilings above; CAVERT’s ceiling applies per call. These are limits rather than measured consumption. Core E4 executor settings appear in Appendix E.

adjudication of disagreements. The development sets contain 30 pairs for E1 and 90 traces for E2 and do not enter formal denominators.

Metric definitions   
Three-way macro-F1 averages the N, V, and U classwise F1 scores equally; zero-denominator classes   
receive zero.   
Decision coverage (Cov.) is the fraction of all units answered N or V, including reference-U units. It   
measures definite-answer frequency rather than log completeness.   
Binary violation F1 excludes reference U: a predicted U on V is a false negative, while a predicted U on   
N is not a false positive.

Task-cluster resampling keeps each task’s arms, queries, models, seeds, and views together. E1 uses 10,000 percentile-bootstrap replicates; all-zero or all-one cells yield degenerate empirical intervals.

## B.2 DIAGNOSTIC AND EXECUTION CONFIGURATIONS

B3/B4 both disable thinking, with different output ceilings (Table A5). Rules (also labeled Strong rules) uses PublicContractRules to evaluate public-contract evidence deterministically and propagate unknown required evidence to U. AgentRx uses a contract resolver because failure attribution differs from N/V/U judgment. CAVERT’s two LLM stages supply scope and causal-state reasoning; the final status function maps the recovered check states to N/V/U deterministically.

Stage interfaces. Given the log and contract, Qwen3.5-Flash selects relevant messages, memory operations, and tool records. DeepSeek-V4-Flash links this evidence to recipients, dependent actions, and effects. The adapter constructs T and finding-level C, O, W, D states, retaining supporting record references and leaving unresolved fields unknown. The validator then applies Table A1 and aggregates findings without further LLM calls. The 4,096-token output ceiling applies to each extraction call, not to the full two-stage process.

The execution-table labels DeepSeek, Gemini, Qwen, and Llama denote DeepSeek-V4-Flash, Gemini-2.5-Flash, Qwen3.8-27B-FP8, and Llama-3.3-70B-Instruct-FP8, respectively. These are execution models; Qwen3.5-Flash is used in extraction. The four-model diagnosis and supplementary recovery panels are distinct from Core E4’s two executor configurations, specified in Appendix E.

Table A6: System configurations and a reference-semantic-input probe (N = 432 traces; 144 task clusters).
<table><tr><td>Configuration</td><td>Evidence Source</td><td>Adjudication Rules</td><td>3-Way Acc</td><td>Error Count</td><td>Violation Prec / Rec / F1</td></tr><tr><td>1. Structural Baseline</td><td>Heuristic (No LLM)</td><td>Heuristic Rules</td><td>58.3%</td><td>180 errors</td><td>45.8% / 64.7% 1 0.537 (FP=117)</td></tr><tr><td>2. End-to-End CAVERT</td><td>2-stage LLM</td><td>Deterministic Rules</td><td>83.3%</td><td>72 errors</td><td>91.7% / 64.7% / 0.759 (FP=9) 100.0% /</td></tr><tr><td>3. Oracle + Rules</td><td>Reference semantics</td><td>Deterministic Rules</td><td>97.9%</td><td>9 errors</td><td>100.0% / 1.000 (FP=0)</td></tr></table>

The structural baseline changes both extraction and adjudication, making it a full-system comparison. CAVERT reduces FP from 117 to 9, while recall remains 64.7% (TP=99, FN=54). The non-deployable Oracle probe has 9 errors versus 72 end-to-end errors: a net reduction of 63 (87.5%). Its remaining errors are Gold N predicted as U; violation FP/FN are zero on this set. With rows and columns ordered V/N/U, the Oracle matrix is ((153, 0, 0), (0, 171, 9), (0, 0, 99)), giving macro-F1 0.977.

## B.3 SEMANTIC EXTRACTION DIAGNOSTIC PROBE

The Oracle probe tests evidence quality under fixed verdict rules on a separate calibration set of 432 traces from 144 task clusters (Table A6). It replaces CAVERT’s two-stage LLM extraction with reference semantic inputs. The structural baseline changes both extraction and adjudication.

Reference semantic inputs raise accuracy from 83.3% to 97.9% and reduce errors from 72 to 9 (87.5%); the remaining errors are N cases predicted U. RQ1 identifies relationships that preserve verdict distinctions; this probe shows the value of supplying them accurately. Together, they motivate recording these relationships during memory use, message delivery, and tool execution, so future checkers can read them directly. CAVERT provides the complementary log-based route for existing systems.

## C DEFINITION EVALUATION DETAILS

## C.1 PAIRED DESIGN AND INFORMATION REMOVAL

The three sources each contribute 150 pairs across five factors. The official query and public contract are the same within a pair. Consumer receipt and version eligibility matter only where the contract makes them relevant; an unreceived update is not by itself proof of a violation. A repair must remain admissible at the deadline, rather than merely have appeared earlier. Full, reduced, and selectively restored representations are canonicalized while preserving semantically relevant identity and ordering. Here, full denotes the normalized representation before ablation, not the complete execution record. Equality is tested on this representation; reference labels assess duty fulfillment in the underlying executions under the shared contract.

Reference judgments assess duty fulfillment using the pre-removal evidence and remain fixed across views. Each ablation removes the selected relationship; restoration adds back its verdict-relevant information from the full view. For example, active at deadline in Figure 4 summarizes whether the repair remains valid at the deadline after accounting for later events.

The 398 opposite-label pairs contain 4 full-view collisions, 328 reduced-view collisions, and 321 restored separations. Every reported factor cell satisfies $S \le C _ { \mathrm { m i n u s } } - C _ { \mathrm { f u l l } }$ . This bound assumes reduction is a deterministic function of the full representation and restoration introduces no information outside that full view.

Table A7 reports all benchmark–factor cells, each containing 30 pairs. The 38 same-label pairs and 14 pairs containing U remain in the collected sample but are excluded from collision denominators.

Table A7: Full sample accounting and representation collisions by information factor.
<table><tr><td>Information factor</td><td>D/A</td><td> $C _ { \mathrm { f u l l } } / D$ </td><td> $C _ { \mathrm { m i n u s } } / D$ </td><td> $S / C _ { \mathrm { m i n u s } }$ </td></tr><tr><td>AgentWebBench</td><td></td><td></td><td></td><td></td></tr><tr><td>Consumer receipt</td><td>27/30</td><td>0/27</td><td>23/27</td><td>23/23</td></tr><tr><td>Action dependency</td><td>26/30</td><td>0/26</td><td>21/26</td><td>21/21</td></tr><tr><td>Actor authorization</td><td>28/30</td><td>1/28</td><td>24/28</td><td>23/24</td></tr><tr><td>Historical-version eligibility</td><td>25/30</td><td>0/25</td><td>19/25</td><td>18/19</td></tr><tr><td>Repair validity at deadline</td><td>27/30</td><td>0/27</td><td>22/27</td><td>22/22</td></tr><tr><td>Subtotal</td><td>133/150</td><td>1/133</td><td>109/133</td><td>107/109</td></tr><tr><td>MARBLE</td><td></td><td></td><td></td><td></td></tr><tr><td>Consumer receipt</td><td>27/30</td><td>0/27</td><td>24/27</td><td>24/24</td></tr><tr><td>Action dependency</td><td>27/30</td><td>1/27</td><td>23/27</td><td>22/23</td></tr><tr><td>Actor authorization</td><td>26/30</td><td>1/26</td><td>22/26</td><td>21/22</td></tr><tr><td>Historical-version eligibility</td><td>26/30</td><td>0/26</td><td>20/26</td><td>19/20</td></tr><tr><td>Repair validity at deadline</td><td>28/30</td><td>0/28</td><td>24/28</td><td>24/24</td></tr><tr><td>Subtotal</td><td>134/150</td><td>2/134</td><td>113/134</td><td>110/113</td></tr><tr><td>MECoBench</td><td></td><td></td><td></td><td></td></tr><tr><td>Consumer receipt</td><td>26/30</td><td>0/26</td><td>22/26</td><td>22/22</td></tr><tr><td>Action dependency</td><td>26/30</td><td>1/26</td><td>21/26</td><td>20/21</td></tr><tr><td>Actor authorization</td><td>27/30</td><td>0/27</td><td>22/27</td><td>21/22</td></tr><tr><td>Historical-version eligibility</td><td>25/30</td><td>0/25</td><td>18/25</td><td>18/18</td></tr><tr><td>Repair validity at deadline</td><td>27/30</td><td>0/27</td><td>23/27</td><td>23/23</td></tr><tr><td>Subtotal</td><td>131/150</td><td>1/131</td><td>106/131</td><td>104/106</td></tr><tr><td>Total</td><td>398/450</td><td>4/398</td><td>328/398</td><td>321/328</td></tr></table>

A: collected pairs; D: opposite N/V labels; S: reduced collisions separated by restoring information. The 38 same-label and 14 U-containing pairs remain in sample accounting but not in D.

Table A8: Reference verdicts in the 360-trace natural-log panel.
<table><tr><td>Source</td><td>Sampling unit and stratum</td><td>Traces</td><td>V</td><td>N</td><td>U</td></tr><tr><td>TraceElephant</td><td>Released MAS task-failure traces</td><td>120</td><td>38</td><td>82</td><td>0</td></tr><tr><td>AFTraj</td><td>Safe / diagnosed natural-failure traces</td><td>120</td><td>2</td><td>105</td><td>13</td></tr><tr><td>MATM</td><td>One population run per sampled task</td><td>120</td><td>3</td><td>101</td><td>16</td></tr><tr><td>Total</td><td></td><td>360</td><td>43</td><td>288</td><td>29</td></tr></table>

V/N/U are consistency reference labels, not task-success labels or detector predictions. Each source has its own sampling frame.

## C.2 MECHANISM OCCURRENCE IN NATURAL LOGS

The natural-log annotation release, frozen on September 2, 2026, contains 360 traces: 120 each from TraceElephant (Chen et al., 2026), AFTraj (Zhang et al., 2026a), and MATM (Kim et al., 2026). Two annotators independently assess the twelve mechanisms without seeing CAVERT predictions; a third adjudicates disagreements. The separate 30-trace calibration set is excluded.

TraceElephant contributes 40 Captain-Agent and 80 Magentic-One task-failure traces. AFTraj contributes 90 source-labeled safe and 30 diagnosed natural-failure traces, sampled within domains. MATM contributes one randomly selected population run per sampled task: 40 ALFWorld and 80 WebArena tasks. Injected AFTraj traces, MATM expert demonstrations, and single-agent SWE-Agent traces are excluded. Source labels and task outcomes are hidden during annotation. The source strata define the descriptive sample; its proportions characterize these selected logs rathe than deployment-wide prevalence.

Table A9 counts a trace once for a mechanism when that mechanism’s final reference verdict is V. Secondary manifestations are excluded from these primary-mechanism counts. Four violating traces have two primary mechanisms, yielding 47 trace–mechanism associations across 43 violating traces. Percentages therefore use 43 as the denominator and need not sum to 100%.

Table A9: Primary violation mechanisms in natural logs.
<table><tr><td>Mechanism</td><td>TE</td><td>AF</td><td>MT</td><td>Total</td><td>% of 43 V</td></tr><tr><td>Strict-temporal</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Conflicting write</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Stale dependency use</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Unresolved contradiction</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Causal-semantic</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Missed handoff</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Unsupported assumption</td><td>24</td><td>2</td><td>3</td><td>29</td><td>67.4</td></tr><tr><td>Unsupported reconstruction</td><td>1</td><td>0</td><td>0</td><td>1</td><td>2.3</td></tr><tr><td>Source scope loss</td><td>3</td><td>0</td><td>0</td><td>3</td><td>7.0</td></tr><tr><td>Source-view scope loss</td><td>4</td><td>0</td><td>0</td><td>4</td><td>9.3</td></tr><tr><td>Evidence status escalation</td><td>8</td><td>0</td><td>0</td><td>8</td><td>18.6</td></tr><tr><td>Eventual-semantic</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Unresolved final divergence</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Competing final answers</td><td>2</td><td>0</td><td>0</td><td>2</td><td>4.7</td></tr><tr><td>Unresolved final evidence state</td><td>0</td><td>0</td><td>0</td><td>0</td><td>0.0</td></tr><tr><td>Trace-mechanism associations</td><td>42</td><td>2</td><td>3</td><td>47</td><td></td></tr></table>

TE: TraceElephant; AF: AFTraj; MT: MATM. Counts are distinct traces per primary mechanism with a V reference verdict. A trace can contribute to multiple rows. Secondary labels are excluded; zero means no primary V label in this sample.

Table A10: E2: Comparison over 930 contract queries across three benchmarks.
<table><tr><td>Method</td><td>MARBLE</td><td>CoffeeBench</td><td>MECoBench</td><td>Pooled</td><td>∆ [95% CI]</td><td></td></tr><tr><td>Always N</td><td>0.205</td><td>0.204</td><td>0.207</td><td>0.206</td><td>-0.545</td><td>[−0.578, -0.512]</td></tr><tr><td>Generic LLM</td><td>0.465</td><td>0.463</td><td>0.470</td><td>0.466</td><td>-0.285</td><td>−0.318, −0.252]</td></tr><tr><td>AgentRx + resolver</td><td>0.591</td><td>0.591</td><td>0.591</td><td>0.591</td><td>-0.160</td><td>[−0.191, -0.130]</td></tr><tr><td>Strong rules</td><td>0.632</td><td>0.631</td><td>0.622</td><td>0.629</td><td>-0.122</td><td>[-0.150, -0.094]</td></tr><tr><td>Contract LLM</td><td>0.663</td><td>0.660</td><td>0.657</td><td>0.660</td><td></td><td>−0.091 [−0.117, −0.064]</td></tr><tr><td>CAVERT</td><td>0.753</td><td>0.747</td><td>0.753</td><td>0.751</td><td>Reference</td><td></td></tr></table>

All score columns are three-way macro-F1; benchmark query counts are 348, 312, and 270. Pooled uses the combined matrix. Differences are method minus CAVERT pooled macro-F1, with paired 95% intervals. Accuracy, violation F1, and coverage are in the stratified appendix table. AgentRx includes a contract resolver.

Unsupported assumption is the most frequent primary mechanism: 24 TraceElephant, 2 AFTraj, and 3 MATM traces. Evidence status escalation is next, with 8 TraceElephant traces. Both concern downstream claims that exceed their supporting evidence. Unresolved final evidence state also appears as a secondary manifestation in 25 traces; its zero primary count reflects this distinction. The other zero entries denote no primary V label in this sample. Together, the annotations show that the definition identifies recurring execution problems in natural logs while distinguishing them from task failure alone.

## D DIAGNOSIS AND ROBUSTNESS DETAILS

## D.1 CONTRACT-QUERY DIAGNOSIS (E2)

The supplementary E2 panel evaluates diagnosis at the contract-query level: 930 preregistered queries on 360 executions, 120 each from MARBLE, CoffeeBench, and MECoBench. Query counts are 348, 312, and 270; pooled labels are 425 V, 415 N, and 90 U. Our AgentRx adaptation (Barke et al., 2026) uses a contract resolver to map failure attribution to N/V/U judgments.

Table A11: E2: Diagnostic performance by benchmark.
<table><tr><td>Method</td><td>Macro-F1</td><td>Binary V F1</td><td>Acc. (%)</td><td>Cov. (%)</td></tr><tr><td colspan="5">MARBLE (N = 348)</td></tr><tr><td>Always N</td><td>0.205</td><td>0.000</td><td>44.5</td><td>100.0</td></tr><tr><td>Generic LLM</td><td>0.465</td><td>0.749</td><td>66.4</td><td>100.0</td></tr><tr><td>AgentRx + resolver</td><td>0.591</td><td>0.630</td><td>64.9</td><td>90.5</td></tr><tr><td>Strong rules</td><td>0.632</td><td>0.714</td><td>70.1</td><td>87.1</td></tr><tr><td>Contract LLM</td><td>0.663</td><td>0.731</td><td>72.7</td><td>91.1</td></tr><tr><td>CAVERT</td><td>0.753</td><td>0.783</td><td>78.2</td><td>91.1</td></tr><tr><td colspan="5">CoffeeBench (N = 312)</td></tr><tr><td>Always N</td><td>0.204</td><td>0.000</td><td>44.2</td><td>100.0</td></tr><tr><td>Generic LLM</td><td>0.463</td><td>0.750</td><td>66.0</td><td>100.0</td></tr><tr><td>AgentRx + resolver</td><td>0.591</td><td>0.629</td><td>64.7</td><td>90.4</td></tr><tr><td>Strong rules</td><td>0.631</td><td>0.709</td><td>69.6</td><td>86.9</td></tr><tr><td>Contract LLM</td><td>0.660</td><td>0.727</td><td>72.1</td><td>90.7</td></tr><tr><td>CAVERT</td><td>0.747</td><td>0.770</td><td>77.2</td><td>90.4</td></tr><tr><td colspan="5">MECoBench (N = 270)</td></tr><tr><td>Always N</td><td>0.207</td><td>0.000</td><td>45.2</td><td>100.0</td></tr><tr><td>Generic LLM</td><td>0.470</td><td>0.752</td><td>67.4</td><td></td></tr><tr><td>AgentRx + resolver</td><td>0.591</td><td>0.620</td><td>65.2</td><td>100.0 90.7</td></tr><tr><td>Strong rules</td><td>0.622</td><td>0.715</td><td>70.4</td><td>88.1</td></tr><tr><td>Contract LLM</td><td>0.657</td><td>0.730</td><td>73.0</td><td>91.5</td></tr><tr><td>CAVERT</td><td>0.753</td><td>0.776</td><td>78.1</td><td>91.5</td></tr></table>

Macro-F1 averages N/V/U equally. Binary V F1 excludes reference-U cases; a predicted U on V is a miss.   
Cov. is the percentage of all queries answered N or V.

<table><tr><td colspan="17">E2: CAVERT confusion matrices</td></tr><tr><td colspan="11">Rows are reference labels and columns are predictions, both ordered V,N,U. CAVERT&#x27;s three benchmark matrices are</td></tr><tr><td>0</td><td></td><td></td><td></td><td></td><td>94</td><td>43</td><td>5</td><td></td><td>83</td><td>37</td><td>4</td></tr><tr><td></td><td>108</td><td>46</td><td>5</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>MMB =</td><td>9</td><td>142</td><td>4 2 12 22</td><td>McB =</td><td>8 0</td><td>126 11</td><td>4 21</td><td>MME =</td><td>7 0</td><td>112 8</td><td>3 16</td></tr></table>

CAVERT leads in macro-F1; the generic judge has higher violation recall. Table A11 reports perbenchmark metrics. Table A10 pools confusion matrices and reports method-minus-CAVERT differences with intervals from matched query predictions.

## D.2 CROSS-MODEL AND MULTIVIEW DIAGNOSIS (E3)

The multiview evaluation contains 180 traces, four methods, and five views (3,600 predictions). Accuracy, macro-F1, coverage, and prediction transitions use the same reference labels and predictions, matched by benchmark, executor, and trace within each view. AgentRx is evaluated in the query-level setting; the multiview panel compares CAVERT with Generic LLM, Contract LLM, and Rules.

Transformations and Evaluation Protocol. Original retains the supplied execution log. Recode A consistently renames event id and actor id, updates their references, and reverses the alphabetical order of JSON keys. Recode B flattens the JSON hierarchy into a line-based log format with explicit causal pointers. Both preserve the recorded relationships and original reference labels.

Missing L1 removes direct prerequisite-delivery confirmations (context deliver) and explicit handshake receipts, and sets channel complete to false. Missing L2 starts from L1 and also removes pre-deadline state-retirement events (effect retirement) and intermediate action confirmations. It additionally sets causal complete and complete through deadline to false. These flags describe lost coverage, rather than evidence that an unrecorded event did not occur. Each removal view receives a reference judgment based on its remaining evidence.

Table A12: Diagnosis across execution models and evidence views.
<table><tr><td>Benchmark</td><td>Executor</td><td></td><td colspan="2">n Generic LLM Contract LLM</td><td colspan="2"></td><td colspan="2">Rules</td><td colspan="2">CAVERT</td></tr><tr><td></td><td></td><td></td><td>Acc.</td><td></td><td>F1 Acc.</td><td></td><td>F1 Acc.</td><td></td><td>F1 Acc.</td><td>F1</td></tr><tr><td colspan="9">A. Cross-executor diagnosis: 720 executions</td><td></td><td></td></tr><tr><td>MARBLE</td><td>DeepSeek</td><td>60</td><td>63.3</td><td>0.435</td><td>80.0</td><td>0.742</td><td>78.3 0.731</td><td></td><td>86.7 0.828</td><td></td></tr><tr><td></td><td>Gemini</td><td>60</td><td>60.0</td><td>0.412</td><td>76.7</td><td>0.708</td><td>75.0</td><td>0.695</td><td>83.3</td><td>0.792</td></tr><tr><td></td><td>Qwen</td><td>60</td><td>61.7</td><td>0.421</td><td>78.3</td><td>0.725</td><td>76.7</td><td>0.714</td><td>85.0</td><td>0.810</td></tr><tr><td></td><td>Llama</td><td>60</td><td>66.7</td><td>0.468</td><td>83.3</td><td>0.785</td><td>81.7</td><td>0.768</td><td>90.0</td><td>0.865</td></tr><tr><td>CoffeeBench</td><td>DeepSeek</td><td></td><td>60 68.3</td><td>0.479</td><td>85.0</td><td>0.802</td><td>83.3</td><td>0.785</td><td>90.0</td><td>0.865</td></tr><tr><td></td><td>Gemini</td><td></td><td>6061.7</td><td>0.425</td><td>78.3</td><td>0.728</td><td>76.7</td><td>0.710</td><td>85.0</td><td>0.812</td></tr><tr><td></td><td>Qwen</td><td></td><td>6063.3</td><td>0.438</td><td>80.0</td><td>0.745</td><td>78.3</td><td>0.732</td><td>86.7</td><td>0.830</td></tr><tr><td></td><td>Llama</td><td>60</td><td>65.0</td><td>0.452</td><td>81.7</td><td>0.762</td><td>80.0</td><td>0.748</td><td>88.3</td><td>0.848</td></tr><tr><td>MECoBench</td><td>DeepSeek</td><td>60</td><td>66.7</td><td>0.462</td><td>81.7</td><td>0.765</td><td>80.0</td><td>0.745</td><td>88.3</td><td>0.846</td></tr><tr><td></td><td>Gemini</td><td>60</td><td>65.0</td><td>0.448</td><td>80.0</td><td>0.742</td><td>78.3</td><td>0.728</td><td>86.7</td><td>0.828</td></tr><tr><td></td><td>Qwen</td><td></td><td>6065.0</td><td>0.450</td><td>81.7</td><td>0.760</td><td>80.0</td><td>0.742</td><td>88.3</td><td>0.845</td></tr><tr><td></td><td>Llama</td><td>60</td><td>63.3</td><td>0.432</td><td>80.0</td><td>0.738</td><td>78.3</td><td>0.722</td><td>86.7</td><td>0.825</td></tr><tr><td colspan="2">All benchmarks / executors</td><td></td><td>72064.2</td><td>0.435</td><td>80.6</td><td>0.748</td><td>78.9</td><td>0.738</td><td>87.1</td><td>0.835</td></tr><tr><td colspan="9">B. Evidence transformations: 180 traces, five views</td><td></td></tr><tr><td>Original</td><td></td><td>18058.9</td><td></td><td>0.414</td><td>82.2</td><td>0.845</td><td>80.0</td><td>0.829</td><td>87.8</td><td>0.895</td></tr><tr><td>Recode A</td><td></td><td>180</td><td>56.7</td><td>0.399</td><td>80.6</td><td>0.833</td><td>80.0</td><td>0.829</td><td>87.8</td><td>0.895</td></tr><tr><td>Recode B</td><td></td><td>180</td><td>53.9</td><td>0.379</td><td>76.1</td><td>0.792</td><td>78.9</td><td>0.821</td><td>86.7</td><td>0.886</td></tr><tr><td>Missing L1</td><td></td><td>180</td><td>41.1</td><td>0.311</td><td>68.3</td><td>0.709</td><td>65.6</td><td>0.683</td><td>78.3</td><td>0.801</td></tr><tr><td>Missing L2</td><td></td><td>18023.9</td><td></td><td>0.217</td><td>56.1</td><td>0.568</td><td>57.2</td><td>0.578</td><td></td><td>66.1 0.661</td></tr></table>

Acc. is percent; F1 is three-class macro-F1. Upper: 720 executions across three benchmarks and four executors. Lower: a separate 180-trace set under five evidence views; all metrics use the same records.

The link to the verdict rules is finding-specific. Each obligation is checked over its relevant channels and time interval. A false required coverage condition prevents that finding from establishing V. An earlier N rule may still apply; otherwise the judgment is U (Table A1). At trace level, V requires at least one finding with a certified window and a violated duty, even if other findings remain unresolved.

Methods are scored on the same instances and reference labels within each view, using the configurations in Table A5. Cross-view agreement measures prediction stability; within-view accuracy measures agreement with that view’s reference judgments.

Table A13: E3: Joint prediction transitions from per-trace records.
<table><tr><td>View</td><td>Pred U</td><td>To U</td><td>From U</td><td>N/V flip</td><td>Same</td><td>Assert U</td></tr><tr><td colspan="7">Generic LLM</td></tr><tr><td>Original</td><td>0</td><td>0</td><td>0</td><td>0</td><td>180</td><td>18</td></tr><tr><td>Recode A</td><td>0</td><td>0</td><td>0</td><td>74</td><td>106</td><td>18</td></tr><tr><td>Recode B</td><td>0</td><td>0</td><td>0</td><td>73</td><td>107</td><td>18</td></tr><tr><td>Missing L1</td><td>0</td><td>0</td><td>0</td><td>78</td><td>102</td><td>38</td></tr><tr><td>Missing L2</td><td>0</td><td>0</td><td>0</td><td>70</td><td>110</td><td>76</td></tr><tr><td colspan="7">Contract LLM</td></tr><tr><td>Original</td><td>15</td><td>0</td><td>0</td><td>0</td><td>180</td><td>3</td></tr><tr><td>Recode A</td><td>15</td><td>3</td><td>3</td><td>51</td><td>123</td><td>3</td></tr><tr><td>Recode B</td><td>14</td><td>2</td><td>3</td><td>60</td><td>115</td><td>4</td></tr><tr><td>Missing L1</td><td>26</td><td>15</td><td>4</td><td>61</td><td>100</td><td>12</td></tr><tr><td>Missing L2</td><td>43</td><td>35</td><td>7</td><td>57</td><td>81</td><td>33</td></tr><tr><td colspan="7">Strong rules</td></tr><tr><td>Original</td><td>15</td><td>0</td><td>0</td><td>0</td><td>180</td><td>3</td></tr><tr><td>Recode A</td><td>15</td><td>0</td><td>0</td><td>0</td><td>180</td><td>3</td></tr><tr><td>Recode B</td><td>15</td><td>2</td><td>2</td><td>58</td><td>118</td><td>3</td></tr><tr><td>Missing L1</td><td>25</td><td>15</td><td>5</td><td>62</td><td>98</td><td>13</td></tr><tr><td>Missing L2</td><td>44</td><td>35</td><td>6</td><td>61</td><td>78</td><td>32</td></tr><tr><td colspan="7">CAVERT</td></tr><tr><td>Original</td><td>16</td><td>0</td><td>0</td><td>0</td><td>180</td><td>2</td></tr><tr><td>Recode A</td><td>16</td><td>0</td><td>0</td><td>0</td><td>180</td><td>2</td></tr><tr><td>Recode B</td><td>16</td><td>1</td><td>1</td><td>32</td><td>146</td><td>2</td></tr><tr><td>Missing L1</td><td>30</td><td>18</td><td>4</td><td>47</td><td>111</td><td>8</td></tr><tr><td>Missing L2</td><td>51</td><td>41</td><td>6</td><td>55</td><td>78</td><td>25</td></tr></table>

Transitions compare each view with Original on the same 180 traces. Pred U counts U outputs; To/From U counts N/V-to-U and U-to-N/V changes; N/V flip counts N-to-V or V-to-N changes; Same counts unchanged predictions. Assert U counts definite answers on that view’s reference-U cases (18 in lossless views, 38 in L1, 76 in L2).

Table A14: Prediction stability under recoding and judgments after evidence removal.
<table><tr><td>Pattern</td><td>CAVERT</td><td>Rules</td><td>CAVERT example</td></tr><tr><td colspan="4">Original to Recode B: paired predictions on 180 traces</td></tr><tr><td>Unchanged prediction</td><td>146/180</td><td>118/180</td><td></td></tr><tr><td>Correct to wrong</td><td>18/180</td><td>32/180</td><td>trace_marble_016</td></tr><tr><td>Wrong to correct</td><td>16/180</td><td>30/180</td><td>trace_marble_017</td></tr><tr><td colspan="4">Within Missing L2: prediction against the remaining-evidence reference</td></tr><tr><td>Reference U, predicted U</td><td>51/76</td><td>44/76</td><td></td></tr><tr><td>Reference U, predicted N</td><td>25/76</td><td>32/76</td><td>trace_coffeebench_047</td></tr><tr><td>Reference V, predicted N</td><td>18/52</td><td>23/52</td><td>trace_marble_004</td></tr><tr><td>Reference N, predicted V</td><td>18/52</td><td>22/52</td><td>trace_marble_054</td></tr></table>

Recode B retains Original’s reference labels. Missing L2 has 76 U, 52 V, and 52 N references; denominators are the corresponding class sizes. Its three error rows exhaust the observed L2 errors: 61 for CAVERT and 77 for Rules. Recoding transitions and L2 errors concern separate views of the same traces and are not added together.

## D.3 PREDICTION STABILITY AND EVIDENCE SUFFICIENCY

Missing L2 tests recognition of insufficient evidence and detection of violations supported by the retained records. Table A14 reports U-to-N and V-to-N errors alongside false violation predictions (N-to-V).

Table A15: Paired accuracy differences, CAVERT minus rules.
<table><tr><td>View</td><td></td><td>Difference (pp)</td><td>95% interval</td></tr><tr><td>Original</td><td></td><td>+7.8</td><td> $[ + 5 . 0 , + 1 0 . 6 ]$ </td></tr><tr><td>Recode A</td><td></td><td>+7.8</td><td> $[ + 5 . 0 , + 1 0 . 6 ]$ </td></tr><tr><td>Recode B</td><td></td><td>+7.8</td><td> $[ + 4 . 4 , + 1 1 . 7 ]$ </td></tr><tr><td>Missing L1</td><td></td><td>+12.8</td><td> $[ + 1 0 . 0 , + 1 5 . 6 ]$ </td></tr><tr><td>Missing L2</td><td></td><td>+8.9</td><td> $[ + 6 . 1 , + 1 1 . 7 ]$ </td></tr></table>

10,000 bootstrap draws stratified by benchmark, conditional on the recorded 30-cluster partition of traces. Methods and views share draws.

Missing L2: judgments from the retained evidence

Rows are reference labels and columns are predictions, both ordered V,N,U:

$$
M _ { \mathrm { C A V E R T } } = { \left( \begin{array} { l l l } { 3 4 } & { 1 8 } & { 0 } \\ { 1 8 } & { 3 4 } & { 0 } \\ { 0 } & { 2 5 } & { 5 1 } \end{array} \right) }
$$

$$
M _ { \mathrm { R u l e s } } = { \binom { 2 9 \quad 2 3 \quad 0 } { 2 2 \quad 3 0 \quad 0 } }
$$

CAVERT’s 25 U-to-N errors comprise five CoffeeBench and twenty MECoBench traces; its 18 Vto-N errors are all from MARBLE. Sixteen of these 18 violations were correctly predicted in the original view. For example, trace marble 004 retains reference V but changes from predicted V in L1 to N in L2. CAVERT correctly classifies 119/180 traces and Rules 103/180. These classspecific counts evaluate both evidence sufficiency and duty fulfillment.

## D.4 DIAGNOSIS UNDER A FIXED SPENDING LIMIT

Shared diagnostic inputs and spending limit

We compare CAVERT with Contract LLM + review on the same 720 E3 executions, public logs, and task contracts. Each benchmark contains four executors and 60 executions per executor. Contract LLM + review extends the single-round Contract LLM with an initial judgment, contract-by-contract review, and revision. Both methods have a per-execution diagnostic spending limit of B = \$0.0120, selected on an independent development set and fixed before test evaluation. Stage 1 uses Qwen3.5-Flash; Stage 2 and all Contract LLM calls use DeepSeek-V4-Flash. The limit covers the entire diagnostic process, including retries; CAVERT’s final verdict rules require no LLM call. The spending limit uses a fixed schedule per million input/output tokens: \$0.14/\$0.28 for DeepSeek-V4-Flash and \$0.05/\$0.15 for Qwen3.5-Flash.

Every execution remains in the evaluation denominator. Budget exhaustion or an invalid final response is recorded as a diagnostic failure, distinct from the semantic verdict U. A failure receives no accuracy credit and contributes a false negative for its reference class in N/V/U macro-F1. A valid U judgment instead denotes insufficient execution evidence.

CAVERT correctly diagnoses 627/720 executions, compared with 602/720 for Contract LLM + review (Table A16). Its accuracy advantage is +3.47 pp with a 95% interval of [+1.39, +5.56]; its macro-F1 advantage is +0.049 [+0.021, +0.076]. Both point estimates favor CAVERT in every benchmark (Table A17). The comparison extends the practical benefit of the criteria to a baseline with iterative contract review under the same spending limit. CAVERT averages 1.81 model calls per execution, versus 2.65 for the review baseline. Mean API cost is \$0.0017 for CAVERT and \$0.0031 for Contract LLM + review, with 95th-percentile costs of \$0.0026 and \$0.0042, respectively (Table A18). The diagnostic advantage thus persists with iterative review available to the baseline, while CAVERT uses fewer calls and lower API spending under the fixed price schedule.

For uncertainty estimates, 10,000 paired bootstrap replicates resample original task clusters within each benchmark. Each sampled cluster retains its executions, executor configurations, and both methods’ predictions. Accuracy and three-class macro-F1 are recomputed from pooled predictions in each replicate. The 95% intervals are the 2.5th and 97.5th percentiles of paired differences. Pooled macro-F1 is computed across executions; it is not the mean of benchmark macro-F1 scores.

Table A16: Diagnosis under a common per-execution spending limit of \$0.0120.
<table><tr><td>Benchmark</td><td>Method</td><td>Correct</td><td>Acc. (%)</td><td>Macro-F1</td><td>Calls</td></tr><tr><td>MARBLE</td><td>Contract LLM + review</td><td>198/240</td><td>82.50</td><td>0.778</td><td>2.64</td></tr><tr><td>MARBLE</td><td>CAVERT</td><td>207/240</td><td>86.25</td><td>0.824</td><td>1.80</td></tr><tr><td>CoffeeBench</td><td>Contract LLM + review</td><td>203/240</td><td>84.58</td><td>0.801</td><td>2.66</td></tr><tr><td>CoffeeBench</td><td>CAVERT</td><td>210/240</td><td>87.50</td><td>0.839</td><td>1.82</td></tr><tr><td>MECoBench</td><td>Contract LLM + review</td><td>201/240</td><td>83.75</td><td>0.779</td><td>2.65</td></tr><tr><td>MECoBench</td><td>CAVERT</td><td>210/240</td><td>87.50</td><td>0.841</td><td>1.81</td></tr><tr><td>Pooled</td><td>Contract LLM + review</td><td>602/720</td><td>83.61</td><td>0.786</td><td>2.65</td></tr><tr><td>Pooled</td><td>CAVERT</td><td>627/720</td><td>87.08</td><td>0.835</td><td>1.81</td></tr></table>

Each benchmark aggregates four executors × 60 executions. Calls are per-execution means over the complete diagnostic process, including format-repair attempts and retries. Both methods may finish below the spending limit. Single-round Contract LLM results are reported separately in Table 2.

Table A17: Paired performance differences under the common spending limit.
<table><tr><td>Scope</td><td>∆ Acc. (pp)</td><td>95% interval</td><td>∆ Macro-F1</td><td>95% interval</td></tr><tr><td>MARBLE</td><td>+3.75</td><td>[+0.42, +7.08]</td><td>+0.046</td><td>[+0.012, +0.080]</td></tr><tr><td>CoffeeBench</td><td>+2.92</td><td>[+0.00, +5.83]</td><td>+0.038</td><td>[+0.006, +0.070]</td></tr><tr><td>MECoBench</td><td>+3.75</td><td>[+0.83, +6.67]</td><td>+0.062</td><td>[+0.025, +0.099]</td></tr><tr><td>Pooled</td><td>+3.47</td><td>[+1.39, +5.56]</td><td>+0.049</td><td>[+0.021, +0.076]</td></tr></table>

Differences are CAVERT minus Contract LLM + review. Intervals use 10,000 paired task-cluster bootstrap replicates stratified by benchmark.

Table A18: API usage and cost across all 720 E3 executions.
<table><tr><td>Method</td><td>Diagnostic path</td><td>Calls</td><td>Mean cost</td><td>P95 cost</td></tr><tr><td>Generic LLM</td><td>Direct judgment</td><td>1.00</td><td>$0.0010</td><td>$0.0014</td></tr><tr><td>Rules</td><td>Deterministic checks</td><td>0.00</td><td>$0.0000</td><td>$0.0000</td></tr><tr><td>Contract LLM</td><td>Single-round contract judgment</td><td>1.00</td><td>$0.0014</td><td>$0.0021</td></tr><tr><td>Contract LLM + review</td><td>Initial judgment, review, revision</td><td>2.65</td><td>$0.0031</td><td>$0.0042</td></tr><tr><td>CAVERT</td><td>Scope filtering, causal-state inference, deterministic verdict</td><td>1.81</td><td>$0.0017</td><td>$0.0026</td></tr></table>

Calls and costs cover each execution’s complete diagnostic process, including format-repair attempts and billed failed calls or retries. Means use all 720 executions; P95 is the 95th percentile of per-execution total cost. USD costs use recorded token consumption and the fixed price schedule in the text. Per-call output ceilings are not consumption measurements. Rules incur no model API cost.

## E EXECUTION PROTOCOL AND STRATIFICATION

## E.1 MATCHED BLOCKS AND EXECUTOR CONFIGURATIONS

E4 and E5 share 720 fault blocks but score complete executions and truncated probes separately. E5 reports 6,480 base nine-arm probes, 720 fixed-step Rules+G sensitivity probes, and 576 probes from four length configurations on 144 edge-case blocks. Step-0 logs are shared, so E4 and E5 reported run counts are not simply added.

E4 assigns 90 blocks per benchmark to each executor configuration. Config-A uses DeepSeek-V4-Flash, temperature 0.2, top-p = 0.95, a 2,048-token output ceiling, a 30-action budget, and a 180-second timeout. Config-B uses Gemini-2.5-Flash, temperature 0, top-p = 1, a 4,096-token output ceiling, a 50-action budget, and a 300-second timeout. Their controllers are the standard and extended variants of cavert controller v4.2. The original allocation has six arms: Native, Retry, Direct+G, Rules+G, CAVERT-Prompt, and CAVERT-Gate. Supplementary MemTX evaluation adds one arm on the same 720 blocks, retaining each block’s executor configuration and resource ceilings. Table A19 and Figure 7 display the six intervention arms. The Native score of 27.8% in Table A21 belongs to E5’s fixed-length probes, not E4’s complete-execution endpoint.

Table A19: Matched complete-execution recovery with benchmark-specific endpoints.
<table><tr><td></td><td></td><td></td><td colspan="3">Baselines</td><td colspan="2">CAVERT</td></tr><tr><td>Benchmark</td><td>Endpoint</td><td></td><td></td><td>n/arm Retry Direct+G Rules+G MemTX Prompt Gate</td><td></td><td></td><td></td></tr><tr><td colspan="8">Matched complete executions: 180 blocks per benchmark</td></tr><tr><td>AppWorld Success %</td><td>180</td><td>31.1</td><td>38.9</td><td>49.4</td><td>55.0</td><td>43.3 67.8</td><td></td></tr><tr><td>LongCLI F2P+P2P %</td><td>180</td><td>12.2</td><td>18.9</td><td>28.3</td><td>32.8</td><td>22.2 41.7</td><td></td></tr><tr><td>MARBLE Score %</td><td>180</td><td>46.7</td><td>53.9</td><td>64.4</td><td>72.2</td><td></td><td>57.8 78.3</td></tr><tr><td>CoffeeBench Payout %</td><td>180</td><td>43.9</td><td>51.1</td><td>62.2</td><td>68.9</td><td></td><td>55.076.1</td></tr><tr><td>Equal-weight normalized mean</td><td></td><td>720 33.5</td><td>40.7</td><td>51.1</td><td>57.2</td><td></td><td>44.6 66.0</td></tr></table>

G: shared gate. All rows use two executor configurations, with 90 blocks each and 180 per arm (Appendix E). MemTX adds one arm on the same blocks. Payout is clipped normalization, not dollar income. The final row averages four normalized endpoint scores, not task-success probabilities across benchmarks. Paired differences appear in Table A20.

Table A20: Paired recovery score differences on the same 720 fault blocks.
<table><tr><td>Benchmark</td><td>MemTX - Rules+G</td><td>CAVERT-Gate – MemTX</td></tr><tr><td>AppWorld</td><td>+5.56 [1.11, 10.00]</td><td>+12.78 [7.78, 17.78]</td></tr><tr><td>LongCLI</td><td>+4.44 [−0.56, 9.44]</td><td>+8.89 [3.89, 13.89]</td></tr><tr><td>MARBLE</td><td>+7.78 [3.33, 12.22]</td><td>+6.11 [1.67, 10.56]</td></tr><tr><td>CoffeeBench</td><td>+6.67 [2.22, 11.11]</td><td>+7.22 [2.78, 11.67]</td></tr><tr><td>Equal-weight mean difference</td><td>+6.11 [3.89, 8.33]</td><td>+8.75 [6.39, 11.11]</td></tr></table>

Entries are differences in percentage points with 95% intervals, computed before rounding scores. The 10,000 percentile-bootstrap replicates resample matched blocks within each benchmark, keeping task, executor configuration, and seed paired across arms. Each aggregate replicate averages the four benchmark differences with equal weight. Intervals are pointwise.

Table A21: E5: Nine-arm suffix-probe results.
<table><tr><td>Arm</td><td>Macro-average score (%)</td></tr><tr><td>native</td><td>27.8</td></tr><tr><td>retry</td><td>33.5</td></tr><tr><td>generic prompt</td><td>36.9</td></tr><tr><td>rules prompt</td><td>42.5</td></tr><tr><td>direct prompt</td><td>43.1</td></tr><tr><td>cavert prompt</td><td>48.3</td></tr><tr><td>rules gate</td><td>51.1</td></tr><tr><td>direct gate</td><td>52.2</td></tr><tr><td>cavert gate</td><td>66.0</td></tr></table>

Matched executor limits refer to the action budget, model-output ceiling, and timeout assigned to a block. Diagnostic calls have their own per-call output ceilings (Table A5). Task-level token consumption, latency, and billed cost describe realized overhead and are reported separately in Table A28.

MemTX (Li et al., 2026) is a complete memory-system comparator, with belief commit, action gating, and cascading repair. Table A20 compares its recovery scores with Rules+G and CAVERT-Gate. For each of 10,000 percentile-bootstrap replicates, we resample matched blocks within each benchmark, preserving task, executor configuration, and seed across arms. The aggregate is the equal-weight mean of the four benchmark differences. CAVERT-Gate improves on MemTX in every benchmark, with an aggregate difference of +8.75 pp [6.39,11.11]. MemTX improves on Rules+G by +6.11 pp [3.89,8.33] overall; its LongCLI interval includes zero. These are comparisons of complete intervention configurations.

Table A22: Diagnosis–intervention interaction under fixed-length suffix evaluation.
<table><tr><td colspan="5">720 matched blocks; four-benchmark mean</td></tr><tr><td>Judge</td><td>Prompt (%)</td><td>Gate (%)</td><td>Gain (pp)</td><td>95% paired CI</td></tr><tr><td>Strong rules</td><td>42.5</td><td>51.1</td><td>+8.6</td><td>[5.2, 12.0]</td></tr><tr><td>Direct LLM</td><td>43.1</td><td>52.2</td><td>+9.1</td><td>[5.6, 12.6]</td></tr><tr><td>CAVERT</td><td>48.3</td><td>66.0</td><td>+17.7</td><td>[13.8, 21.6]</td></tr></table>

AppWorld, LongCLI, MARBLE, and CoffeeBench contribute 180 blocks each; judges share the gate. Gain is gate minus prompt. CAVERT versus rules interaction: +9.1 pp, 95% CI $[ 4 . 8 , 1 3 . 4 ] , p < 0 . { \dot { 0 } } 0 1$ . Scores average benchmark-specific normalized endpoints, not pooled success probabilities.

Table A23: Budget sensitivity and interference on normal tasks.
<table><tr><td colspan="4">A. Budget sensitivity: 240 fault blocks</td></tr><tr><td>Actions / tokens</td><td>Generic Prompt</td><td>Rules+G</td><td>CAVERT+G</td></tr><tr><td>15 / 50k</td><td>28.3</td><td>41.7</td><td>54.2</td></tr><tr><td>30 / 100k</td><td>36.9</td><td>51.1</td><td>66.0</td></tr><tr><td>50 / 200k</td><td>40.8</td><td>55.4</td><td>71.3</td></tr><tr><td></td><td></td><td></td><td></td></tr></table>

<table><tr><td colspan="5">B. Normal tasks: 480 uninjected blocks</td></tr><tr><td>Method</td><td>Score</td><td>∆(pp)</td><td>95% CI</td><td>Non-inferiority</td></tr><tr><td>Native</td><td>88.5</td><td>+0.0</td><td>Reference</td><td></td></tr><tr><td>Rules+G</td><td>86.1</td><td>-2.4</td><td>[−4.1, −0.7]</td><td>Not established</td></tr><tr><td>MemTX (mature)</td><td>86.7</td><td>-1.8</td><td>-3.4, −0.2]</td><td>Not established</td></tr><tr><td>MemTX (unmatured)</td><td>84.6</td><td>-3.9</td><td>-5.6, −2.1]</td><td>Not established</td></tr><tr><td>CAVERT+G</td><td>87.8</td><td>-0.7</td><td>[-2.1, +0.7]</td><td>Supported</td></tr></table>

Scores are percentages. A reuses a matched-fault subset; action/token values are ceilings. B compares interventions with Native on the same 480 blocks. Non-inferiority requires the 95% interval’s lower bound to exceed −2.5 pp (paired-block bootstrap; Appendix E.3).

## E.2 RECOVERY CONTROLS AND OUTCOME DEFINITIONS

Native, retry, and generic-prompt controls accompany rules, direct LLM, and CAVERT crossed with prompt/gate. Arms share initial prefixes, action interfaces, and executor limits within a block. The gate intercepts writes before commitment.

<table><tr><td>Recovery endpoints and aggregate score</td></tr><tr><td>AppWorld uses binary official success; LongCLI uses joint F2P/P2P pass; MARBLE uses a continuous task score. CoffeeBench uses the normalization</td></tr><tr><td>4 sCB = min(1, max(0, Payout/500)), § = 1-4 Sb. b=1</td></tr><tr><td>Payout is cumulative total earnings within a 30-environment-action scoring window, fixed across arms and executor configurations. The preregistered custom clipping threshold,  $\mathrm { P a y o u t } _ { \mathrm { t r u n c } } = \mathfrak { H } 5 0 0$  uses the reported 95th-percentile threshold of historical market scenarios. All arms share this normalization and zero floor. This windowed earnings score is reported separately from cumulative net income; the executor&#x27;s action ceiling controls the run budget rather than the scoring window.</td></tr></table>

## E.3 NORMAL-TASK IMPACT AND BUDGET SENSITIVITY

Table A24 reports normal-task scores for four 120-block cohorts. CAVERT meets the −2.5 pp noninferiority criterion for the 480-block aggregate; the MARBLE stratum’s lower bound equals the margin. On the same blocks, mature and unmatured MemTX score 86.7% and 84.6%, respectively.

Table A24: Normal-task scores and paired differences from Native, by benchmark.
<table><tr><td colspan="6">A. Scores (%); MemTX uses mature initialization</td></tr><tr><td>Cohort</td><td></td><td></td><td></td><td></td><td>Blocks Native Rules+G MemTX CAVERT+G</td></tr><tr><td>AppWorld</td><td>120</td><td>90.0</td><td>87.5</td><td>88.3</td><td>89.2</td></tr><tr><td>LongCLI</td><td>120</td><td>85.0</td><td>82.5</td><td>83.3</td><td>84.2</td></tr><tr><td>MARBLE</td><td>120</td><td>89.2</td><td>86.7</td><td>87.5</td><td>88.3</td></tr><tr><td>CoffeeBench</td><td>120</td><td>89.8</td><td>87.8</td><td>87.8</td><td>89.4</td></tr><tr><td>Equal-weight mean</td><td>480</td><td>88.5</td><td>86.1</td><td>86.7</td><td>87.8</td></tr><tr><td>Historical τ2</td><td>51</td><td>88.2</td><td>85.8</td><td>一</td><td>87.5</td></tr></table>

<table><tr><td colspan="5">B. Differences from Native (pp), with 95% intervals</td></tr><tr><td>Cohort</td><td></td><td></td><td>MemTX - Native CAVERT+G - Native</td><td></td></tr><tr><td>AppWorld</td><td></td><td>-1.7 [−3.3, +0.0]</td><td></td><td>−0.8 [−2.3, +0.7]</td></tr><tr><td>LongCLI</td><td></td><td>-1.7 [−3.4, +0.1]</td><td></td><td>−0.8 [−2.4, +0.8]</td></tr><tr><td>MARBLE</td><td></td><td>-1.7 [−3.4, +0.1]</td><td></td><td>−0.9 [−2.5, +0.7]</td></tr><tr><td>CoffeeBench</td><td></td><td>-2.0 [-3.5, −0.5]</td><td></td><td>−0.4 [−1.8, +1.0]</td></tr><tr><td>Equal-weight mean</td><td></td><td>-1.8 [-3.4, -0.2]</td><td></td><td>−0.7 [−2.1, +0.7]</td></tr><tr><td colspan="3">Historical  $\tau ^ { 2 }$ </td><td>1</td><td>−0.7 [−2.1, +0.7]</td></tr></table>

Four 120-block cohorts form the 480-block aggregate. The historical 51 are separate; no MemTX task score is reported for that cohort. Its permitted-action comparison appears in Table A28. Normal-task scores and action false-block rates have different denominators.

![](images/b815e3c6bf108aa8ca7b26ae826c7d0d24b3374babe18b4925a097c5117f307c.jpg)

![](images/dc7088abd47ae8db899ad6e6cabd720f1965ffd64a7a1034b33d14d2f74d4f88.jpg)  
Figure A1: Budget sensitivity (a, 240 fault blocks) and paired normal-task differences from Native (b, 480 blocks). Benchmark rows show CAVERT+G with 95% CIs. Dashed lines mark zero and the −2.5 pp margin. Protocol: Tables A23 and A24.

Each is compared with Native: −1.8 pp [−3.4, −0.2] and −3.9 pp [−5.6, −2.1]. Both intervals cross the non-inferiority margin.

The 51-block τ<sup>2</sup>-bench (Barres et al., 2025) action-audit cohort is separate. Of its 750 permitted actions, Rules+G blocks 43, MemTX 29, and CAVERT+G 12 (Table A28). These action-level rates are distinct from four-benchmark task scores.

For normal-task uncertainty, we resample the 480 matched blocks with replacement 10,000 times, retaining each block’s arm scores together. The 2.5th and 97.5th percentiles of the mean CAVERTminus-Native difference give [−2.1, +0.7] pp around −0.7 pp. Non-inferiority requires the lower bound to exceed the predeclared −2.5 pp margin.

Budget tiers jointly change action ceilings (15, 30, 50) and token ceilings (50k, 100k, 200k). They therefore measure combined budget sensitivity, not an isolated token effect.

Table A25: Recovery across execution models and supplementary cohorts.
<table><tr><td colspan="5"></td><td colspan="2">Baselines</td><td colspan="2">CAVERT</td></tr><tr><td>Benchmark</td><td>Executor</td><td>Endpoint</td><td>n/arm</td><td></td><td>Retry Direct+G Rules+G Prompt</td><td></td><td></td><td>Gate</td></tr><tr><td colspan="2">Cross-backbone recovery</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AppWorld</td><td>DeepSeek</td><td rowspan="5">Recovery %</td><td>120</td><td>25.8</td><td>34.2</td><td>51.7</td><td>36.7</td><td>68.3</td></tr><tr><td>Gemini</td><td></td><td>120</td><td>26.7</td><td>35.0</td><td>52.5</td><td>37.5</td><td>69.2</td></tr><tr><td>Qwen</td><td></td><td>120</td><td>20.8</td><td>29.2</td><td>45.8</td><td>30.8</td><td>61.7</td></tr><tr><td>Llama</td><td></td><td>120</td><td>22.5</td><td>30.8</td><td>47.5</td><td>32.5</td><td>64.2</td></tr><tr><td>LongCLI</td><td>DeepSeek F2P+P2P %</td><td>60</td><td>6.7</td><td>13.3</td><td>28.3</td><td>15.0</td><td>43.3</td></tr><tr><td rowspan="4"></td><td>Gemini</td><td></td><td>60</td><td>6.7</td><td>13.3</td><td>28.3</td><td>15.0</td><td>43.3</td></tr><tr><td>Qwen</td><td></td><td>60</td><td>5.0</td><td>11.7</td><td>25.0</td><td>13.3</td><td>38.3</td></tr><tr><td>Llama</td><td></td><td>60</td><td>6.7</td><td>13.3</td><td>26.7</td><td>15.0</td><td>40.0</td></tr><tr><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="8">Original and extended cohorts: native endpoints</td><td></td></tr><tr><td>AppWorld</td><td>4-model</td><td>Success %</td><td>480</td><td>18.33</td><td>25.00</td><td>40.83</td><td>27.50</td><td>54.17</td></tr><tr><td>LongCLI</td><td>4-model</td><td>F2P+P2P %</td><td>240</td><td>6.25</td><td>12.92</td><td>27.08</td><td>14.58</td><td>41.25</td></tr><tr><td>MARBLE</td><td>Llama</td><td>Score %</td><td>64</td><td>54.2</td><td>67.8</td><td>76.5</td><td>73.5</td><td>88.5</td></tr><tr><td>CoffeeBench</td><td>DeepSeek</td><td>Income (USD)</td><td>18</td><td>1,240</td><td>2,420</td><td>3,580</td><td>3,150</td><td>4,680</td></tr><tr><td>EntCollabBench DS/Gemini</td><td></td><td>Recovery %</td><td>24</td><td>33.3</td><td></td><td></td><td>54.2</td><td>79.2</td></tr><tr><td>PAST-Bench</td><td></td><td>Qwen/Llama Score [0,1]</td><td></td><td>90 0.583*</td><td></td><td></td><td></td><td>0.867*</td></tr><tr><td>EntCollab‡</td><td>DeepSeek</td><td>Workflow %</td><td>180</td><td>37.8</td><td>48.3</td><td>58.9</td><td>45.6</td><td>72.2</td></tr></table>

G: shared gate. AppWorld recovery flags and official success are distinct endpoints. \* PAST uses persistence/controller interfaces (Table A27). EntCollab (24 groups): DeepSeek-V4-Flash/Gemini-2.5-Flash; ‡: independent extension. Dollar income is unclipped. Cohort settings: Appendix E.4.

## E.4 SUPPLEMENTARY COHORTS, OUTCOMES, AND COSTS

The supplementary AppWorld cohort contains 40 task families, four models, and three seeds (480 blocks per arm). LongCLI contains 20 tasks, four models, and three seeds (240 blocks per arm). Both allow 15 interaction steps: 11 prefix steps and four recovery steps. This setting is separate from the budget sweep. AppWorld scores recovery sequencing, state postconditions, and official success separately: CAVERT+G obtains 316/480 recovery flags and 260/480 official successes. Paired resampling retains AppWorld families and repository-aware LongCLI groups with all models, seeds, and arms together. MARBLE uses task groups and CoffeeBench uses worlds. The CoffeeBench dollar-income panel reports unclipped income, separately from the normalized score panel.

Our EntCollabBench (Yu et al., 2026a) cohort has 24 recovery groups, six per fault family, within the original 36 semantic groups. Success requires official task passage and a final database state matching the reference. The 24 groups across three arms give 72 runs, spanning DeepSeek-V4- Flash and Gemini-2.5-Flash. Retry, prompt, and gate succeed on 8/24, 13/24, and 19/24 groups, respectively. The gate-minus-prompt difference is +25.0 pp, with 11 gate-only and five prompt-only successes. A two-sided exact McNemar test gives p = 0.2101; this small-cohort difference is not statistically significant at 0.05.

Our PAST-Bench (Xue et al., 2026) evaluation measures the fraction of subgoals passed over 90 episodes per arm (30 near, 60 far), five families, and three seeds. No persistence, generic persistence, structured records, and contract control score 0.425, 0.583, 0.742, and 0.867. State access and control authority vary jointly. We aggregate episodes and seeds within families, then resample the five families 20,000 times. Models are fixed within families.

The action panel reuses the 750 permitted actions rather than adding independent task blocks. Token increases use a common Native mean of 12,000 tokens per task. MemTX uses 14,220 tokens and 22.3 seconds per task; CAVERT+G uses 14,950 tokens and 23.5 seconds. Their AppWorld/LongCLI costs are \$0.044 and \$0.042 per task under DeepSeek-V4-Flash billing, including prompt-cache discounts. Token totals alone therefore do not determine billed cost.

Table A26: Five-arm supplementary recovery: native task outcomes and suffix counts.
<table><tr><td colspan="5">A. Native multi-agent complete tasks</td></tr><tr><td></td><td colspan="2">MARBLE: 64 tasks</td><td>CoffeeBench: 18 worlds</td><td></td></tr><tr><td>Recovery method</td><td>Task score ↑</td><td>Violations (%) ↓</td><td>Income (USD) ↑</td><td>Violations (%) ↓</td></tr><tr><td>Standard retry</td><td>0.542</td><td>31.6</td><td>1,240</td><td>42.3</td></tr><tr><td>Fresh-read recovery</td><td>0.615</td><td>25.2</td><td>1,850</td><td>33.5</td></tr><tr><td>Direct diagnosis + gate</td><td>0.678</td><td>18.4</td><td>2,420</td><td>24.8</td></tr><tr><td>CAVERT prompt-only</td><td>0.735</td><td>11.2</td><td>3,150</td><td>15.6</td></tr><tr><td>CAVERT + gate</td><td>0.885</td><td>1.8</td><td>4,680</td><td>2.3</td></tr></table>

B. Stateful recovery suffixes: counts

<table><tr><td rowspan="2">Recovery method</td><td colspan="3">AppWorld: 480 blocks/arm</td><td colspan="2">LongCLI: 240 blocks/arm</td></tr><tr><td>Recovery flag</td><td>Postcondition</td><td>Success</td><td>F2P+P2P</td><td>Regressions</td></tr><tr><td>Standard retry</td><td>115</td><td>104</td><td>88</td><td>15</td><td>4134</td></tr><tr><td>Fresh-read recovery</td><td>137</td><td>124</td><td>108</td><td>23</td><td></td></tr><tr><td>Direct diagnosis + gate</td><td>155</td><td>136</td><td>120</td><td>31</td><td>23</td></tr><tr><td>CAVERT prompt-only</td><td>165</td><td>152</td><td>132</td><td>35</td><td>18</td></tr><tr><td>CAVERT + gate</td><td>316</td><td>292</td><td>260</td><td>99</td><td>4</td></tr></table>

Direct uses the same gate; prompt-only uses CAVERT diagnosis without blocking. A: violations per assessed obligation. B: 40 AppWorld families / 20 LongCLI tasks × 4 models × 3 seeds. AppWorld scores recovery sequencing, state postconditions, and official success separately. F2P+P2P requires repair without test regressions; fewer regressions is better.

Table A27: Enterprise recovery and cross-session scores under their original intervention interfaces.
<table><tr><td>Panel / endpoint</td><td>Reference</td><td>Record</td><td>Controller</td><td> $\Delta _ { \mathrm { C - R } }$  [95% CI]</td></tr><tr><td>EntCollabBench:</td><td>0.333</td><td>0.542</td><td>0.792</td><td>+0.458 [0.250, 0.667]</td></tr><tr><td>recovery PAST-Bench: cross-session score</td><td>0.583</td><td>0.742</td><td>0.867</td><td>+0.284 [0.183, 0.385]</td></tr></table>

∆ is Controller minus Reference before rounding, in rate/score units. EntCollabBench columns: Retry, CAVERT prompt, CAVERT Gate (8/24, 13/24, 19/24). PAST-Bench columns: Generic Persistence, Root/Trigger Record, Contract Controller; 90 episodes per arm and 20,000 paired bootstrap draws over five families.

Table A28: Action false blocks, execution overhead, and monetary costs, with separate cohorts and units.
<table><tr><td colspan="5">A. Historical τ2-bench: 750 permitted actions</td></tr><tr><td colspan="3">Method False blocks / permitted actions</td><td>Rate</td><td>95% CI</td></tr><tr><td colspan="3">Native</td><td>0/750 0.0%</td><td></td></tr><tr><td colspan="2">Rules+G</td><td>43/750</td><td>5.7%</td><td>[4.2%, 7.5%]</td></tr><tr><td colspan="2">MemTX</td><td>29/750</td><td>3.9%</td><td>[2.6%, 5.4%]</td></tr><tr><td colspan="2">CAVERT+G</td><td>12/750</td><td>1.6%</td><td>[0.8%, 2.5%]</td></tr><tr><td colspan="5"></td></tr><tr><td colspan="5">B. Long-horizon execution: per-task means Latency (s) Added latency (s)</td></tr><tr><td colspan="5">Method Tokens Token increase</td></tr><tr><td>Native</td><td>Reference</td><td></td><td>18.5</td><td></td></tr><tr><td></td><td>12,000</td><td>+10.0%</td><td>19.8</td><td>Reference</td></tr><tr><td>Rules+G MemTX</td><td>13,200 14,220</td><td>+18.5%</td><td>22.3</td><td>+1.3</td></tr><tr><td>CAVERT+G</td><td>14,950</td><td>+24.6%</td><td>23.5</td><td>+3.8 +5.0</td></tr><tr><td colspan="5"></td></tr><tr><td colspan="5">C. AppWorld + LongCLI: DeepSeek-V4-Flash billing Unit</td></tr><tr><td>USD / task 0.035 0.044</td><td>Native 0.030</td><td>Rules+G</td><td>MemTX</td><td>CAVERT+G</td></tr></table>

A: separate from the 720 fault and 480 normal blocks; – denotes no interval. B: per-task means on a separate long-horizon cohort, relative to Native. C: AppWorld/LongCLI billed costs including prompt-cache discounts.

## E.5 LIMITATIONS

Our claims are limited to the evaluated contracts, evidence factors, and log transformations; naturallog frequencies characterize only the sampled sources. CAVERT recovers semantic relationships from existing logs rather than observing them natively, so diagnostic quality remains bounded by extraction quality. The recovery gate acts before commitment; repairing committed external effects requires compensation mechanisms beyond our current scope.