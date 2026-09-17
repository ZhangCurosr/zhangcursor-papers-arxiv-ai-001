# COMPOSITIONAL POLICY VIOLATIONS: WHEN STEP-LEVEL COMPLIANCE FAILS IN AGENTIC AI WORKFLOWS

Ashwini Kurady ashwini@runctrl.ai

Sri Sai Charith Grandhi charith@runctrl.ai

Rajesh Gupta rajesh@runctrl.ai

Sumit Mamoria sumit.mamoria@gmail.com

September 17, 2026

Abstract - Agentic workflows now make consequential decisions in regulated settings, and the governance placed around them is almost entirely step-scoped: input-output classifiers, per-turn rails, and span-level evaluators. The policies organizations actually hold, such as referral thresholds, authority limits, and review requirements, are properties of the whole execution rather than of any one step. This mismatch admits a failure mode we call a Compositional Policy Violation (CPV): every individual step passes its own check while the composed execution violates the governing policy. A predicate over a single step cannot evaluate a property that step does not determine, so no improvement in the accuracy of the step-scoped monitors detects this class. We define CPVs as the failure of step-level compliance to compose, and present a taxonomy of four types: Authority Creep, Threshold Laundering, Cumulative Sum Violation, and Context Collapse. We show that the correct repair for each class is dictated by where the guarded quantity mutates. We then introduce a provenance-aware runtime architecture that evaluates policies over complete execution traces, recomputing guarded quantities from raw provenance rather than the pipeline’s derived representation.

## 1 Introduction

Agentic workflows have moved into regulated decision making. American International Group (AIG), reporting on an early rollout built with Anthropic and Palantir, states that it compressed the timeline to review business by more than fivefold while raising data accuracy from 75% to over 90% [1]; Allianz [2] and The Baldwin Group [3] have announced comparable deployments across underwriting operations. These are not laboratory demonstrations but production pipelines in which sequences of specialized agents ingest requests, enrich information, perform domain-specific analyses, and route decisions, with human oversight retained at designated approval or review steps. As a request moves through such a pipeline, it accumulates permissions, context, and prior determinations at each hop. The decision that reaches the final routing step is a function of everything the request picked up along the way, not just the state of any single step.

Governance for agentic workflows however, has converged on step-level predicates. Input-output classifiers evaluate prompts and responses in isolation[4]. Programmable rails enforce constraints at turn level[5]. Evaluators and guardrails attach at the span level[6, 7]. Each mechanism asks the same question at the same grain: is this action permitted, given the state visible at this step?

The policies organizations actually hold are not of that shape. Referral thresholds, authority limits, and review requirements are properties of a trace (A trace is a complete ordered record of what a workflow did: every step, in sequence, with the state each step produced). Forcing them into step-level predicates creates an enforcement gap: a check that can reason only about the state in front of it cannot evaluate a property that depends on prior history or on what later steps will do. A workflow may therefore satisfy every individual check and still violate the policy it is subject to, with no component having failed.

We call this a Compositional Policy Violation (CPV). This is not a matter of imperfect calibration or insufficient monitor capability. It is structural: even perfectly accurate step-level monitors cannot detect this class, because the evidence needed to identify it exists only at the level of the composed execution.

We note at the outset that trace-level monitoring is not itself novel; runtime verification and process compliance supply those techniques. Our contribution is to identify a specific governance failure mode in agentic workflows, characterize its forms, and operationalize its detection.

The paper proceeds as follows. Section 2 situates our work relative to existing literature on guardrails, observability, evaluation, and human oversight. Section 3 defines Compositional Policy Violations (CPVs), differentiates them from conventional workflow defects, and introduces a four part taxonomy: Authority Creep, Threshold Laundering, Cumulative Sum Violation, and Context Collapse. Section 4 shows that the correct repair (relocating a gate, adding one, or extending what an existing gate can see) is dictated by where the guarded quantity mutates. Section 5 presents a provenance-aware runtime architecture for CPV detection. We close by discussing future research in Section 6 and offering concluding remarks in Section 7.

## 2 Related Work

Existing approaches to AI governance, evaluation, and oversight largely assume that policy compliance can be determined at the boundary of individual steps. Modern guardrail systems such as Llama Guard, NeMo Guardrails, Constitutional Classifiers, and AgentSpec enforce constraints over prompts, responses, turns, or individual actions [4, 5, 8, 9]. While effective for local violations, they do not evaluate whether a complete execution trajectory satisfies policies defined over the workflow.

Recent studies have shown that composition can defeat step-level safeguards in adversarial settings. Ahad et al. [10] demonstrate that orchestrated subtasks can individually pass multiple safety classifiers while the combined plan violates security constraints; similar compositional failures have been observed across sessions, accumulated memory, and agent trajectories [11, 12, 13]. However, these works assume an adversarial actor deliberately fragmenting behavior. In contrast, we study non-adversarial compositional failures where individually compliant actions, performed by independently scoped components, produce workflow-level policy violations through ordinary execution.

Observability and evaluation frameworks provide complementary capabilities but remain primarily step or outcomeoriented. Platforms such as Langfuse and LangSmith reconstruct execution traces through spans and runs but evaluate individual units rather than trace-level governance properties [6, 7]. Similarly, process and outcome supervision approaches [14] and agent failure taxonomies such as MAST [15] analyze intermediate reasoning or failed executions, whereas CPVs represent workflows that complete successfully while violating policies defined over the accumulated trajectory.

Finally, human oversight remains the common fallback for high-risk AI workflows, yet studies show that reviewers often over-rely on automated recommendations and that oversight effectiveness decreases as automation increases [16, 17]. These limitations motivate the need for runtime governance mechanisms that reason over complete execution histories. Our work addresses this gap through a provenance-aware architecture that combines policy evaluation, temporal reasoning, authority analysis, and sequence-level monitoring to detect compositional policy violations.

## 3 Taxonomy

We now define this failure mode precisely. A Compositional Policy Violation (CPV) is a workflow in which every individual step satisfies the policy applied to it, yet the composed execution violates the policy governing the workflow as a whole. Three conditions hold together: each step is evaluated against a policy scoped to that step alone; each step performs its function correctly under that scoped policy; and the composed sequence violates a policy defined at the workflow level. No specification is breached and no component contains a bug. The defect lies in how the step-level specifications compose, which are locally sound but globally insufficient.

This distinguishes a CPV from a conventional workflow defect. A race condition violates an invariant that assumed exclusive or ordered access to shared state, when concurrent execution interleaves that access unpredictably. A stale-state error reflects a step reading data past a validity window defined independently of any other step’s timing. A missing recomputation is an omitted design element. A CPV, by contrast, occurs in purely sequential, correctly designed workflows with every local check in place. The violation is invisible to those checks not because monitoring is absent, but because the property being violated is defined over the trace, and no single step determines it.

We classify CPVs by the aspect of execution that becomes unsafe through composition rather than by a shared mechanism (Figure 1). Authority Creep concerns who is entitled to decide, and arises when permissions accumulate across workflow steps. Threshold Laundering and Cumulative Sum Violation concern an accumulated quantity: in the first, a gate checks the quantity correctly but a later step carries it past the limit with nothing re-checking; in the second, no gate exists at the aggregate scope, and individually compliant actions accumulate past the limit. Context Collapse concerns the representation on which the decision is made, and arises when each hand-off accurately summarizes what it received, yet the resulting record drifts further from the original submission with every step, until the final gate evaluates a case that no longer resembles the one the policy was written to address. The four share the CPV structure rather than a common mechanism: every local check passes, the composed execution violates the governing policy, and the evidence needed to see the violation is absent from every step. They differ in which evidence is absent, and therefore in what a detector must retain. Section 4 takes up the repair each class demands.

![](images/76d57bafc8b0f714f5e81b52b1bde6204d900f3a5e74c9951938783f22b3f7cf.jpg)  
Figure 1: Taxonomy of compositional policy violations

Terminology We use the following terms consistently throughout.

• Predicate: a boolean condition over a state or a trace.

• Gate: a workflow step that evaluates a predicate and routes execution on the result. Gates are part of the workflow and can change its behavior.

• Monitor: an observer that evaluates a predicate without altering execution. Step-level when it is a function of the state at a single step; trace-level when it is a function of the execution history.

• Checkpoint: the practice of evaluating one step or one change in isolation, without reference to the trace it belongs to.

## 3.1 Authority Creep Violation

A compositional policy violation in which a sequence of individually authorized agent operations transforms the representation of a guarded quantity such that an authorityrouting gate(a rule that determines which principal - human tier, unit or escalation path is entitled to decide the case) evaluates to a different authority tier than the policy requires, even though the gate’s own evaluation is satisfied. The workflow never claims decision authority. It shapes the input to the rule that allocates decision authority.

Consider commercial insurance underwriting, shown in Figure 2. The governing policy is that any account involving three or more material exceptions must be escalated. The pipeline holds five components. Extraction pulls submission data and flags candidate guideline exceptions; its authority is narrow, surfacing facts and deciding nothing about who reviews anything. Downstream of it sit components that normalize fields, reconcile duplicate records, classify materiality, and resolve flags. Each was scoped and approved on its own terms, against a requirement that existed before the others did, and none confers any decision authority over routing.

![](images/9bcdff35a1de4be1cba2a90f19be2b092df35023dd7248131e5f98790c65f682.jpg)  
Figure 2: Authority Creep Violation in the sample under-writing pipeline. Expanding scope at each stage, illustrating how cumu lative scope expansion emerges through composition rather than explicit authorization at any single step.

Specifically: normalizing COPE fields standardizes property attributes; reconciling loss runs manages duplicate records; classifying discrepancy materiality judges data quality; resolving flags closes out items the agent itself raised. That decision belongs to the escalation router alone, a fixed workflow step the agent is never granted control over.

Suppose extraction surfaces four candidate exceptions on a submission: an expired certificate, a COPE mismatch reporting two locations as one address, an unclassified prior loss, and a discrepant square-footage figure. Four exceptions clears the escalation threshold on its own. Normalization then resolves the COPE mismatch, the two locations are the same property under a corrected address leaving three, still at threshold. Loss-run reconciliation matches the unclassified prior loss to a claim already captured elsewhere in the submission’s history and removes it as a duplicate, leaving two. Materiality classification determines the square-footage discrepancy falls within the policy’s five-percent tolerance and is immaterial, leaving one. The escalation router reads a final count of one against a threshold of three, and does not escalate.

The distinction that matters here is between fixing an error and making a judgment call. Fixing an error means there was one correct answer and the step found it. For example, two database rows that are literally the same record, duplicated by a retry. Removing one is not Authority Creep, since no discretion over the guarded quantity was exercised. Making a judgment call means there were two or more reasonable interpretations, and the step picked one, and picking it made the exception disappear. For example, classifying the square-footage discrepancy as immaterial is one such call: the discrepancy could reasonably have been treated as material, and treating it as immaterial is what removed it from the count. This is the mechanism of the violation: each of several components, scattered across the pipeline, removes one exception on its own judgment call; only in composition do those separate calls add up to a decision that none of them was authorized to make. By the time the router evaluates the final count, the question it exists to answer(does this account need to be escalated?) has already been answered upstream, by components never granted authority to answer it.

Why per-step monitoring is blind to it: Every transform stays inside its authorized scope and every step-level check passes. A monitor that sees one step and the policy scoped to it has no signal available, not a weak signal but none at all. The router itself is not defective either: it applies its threshold correctly to the number it was handed. What no step holds is the relationship between the number the router reads and the evidence that number was derived from, and that relationship is the only place the violation is visible.

Trigerring Conditions: An Authority Creep Violation is not an accident that can happen to any pipeline. It requires a specific arrangement of five conditions. Three of them describe the “Structural Preconditions” that makes the violation possible, and two Compositional Conditions that distinguish a genuine instance from ordinary misbehavior.

## Structural Preconditions

• First, the workflow must contain an authority-routing gate, as defined above — for example, escalation at three material exceptions, an SIU referral at two fraud indicators, or human-in-the-loop review above a risk band. Without such a rule there is no authority for the pipeline to creep past.

• Second, one or more components must hold representation authority over that quantity’s inputs: they are legitimately permitted to classify, score, merge, deduplicate, or suppress the very items the gate counts. This authority is granted for good reasons like noise reduction, precision, reviewer load.

• Third, as established above, the gate reads the pipeline’s transformed representation rather than recomputing from raw evidence under the policy’s own definitions.

Compositional Conditions These conditions should be enforced as hard validity checks on every generated trace.

• Local compliance must hold throughout, as established above. If any step is individually non-compliant, the trace is contaminated. It contains an ordinary local violation that a conventional monitor would catch, and it tells us nothing about compositional governance.

• No single transform may be sufficient: this condition concerns how the displacement is distributed across the pipeline: each transform, applied on its own to the raw evidence, must still route the case to the correct principal. This is also a no-culprit property because the identity of the responsible component depends on the execution order — determined by scheduling, not by anything the agent did. Consider two transforms in the opposite order. Consolidation first takes four exceptions to three, which still escalates; reclassification then takes three to two, which does not. Reverse them and reclassification is the innocent step while consolidation crosses the line. Same components, same scopes, same evidence, same outcome. Whichever transform runs second is the one that crosses the line, so the culprit is determined by scheduling rather than by anything any component did.

## 3.2 Threshold Laundering

Threshold Laundering is a Compositional Policy Violation in which a quantity is checked against a limit at one point in the workflow, and a later step carries that quantity past that limit. No step misapplies the rule. The gate compares the quantity against the limit correctly, but does so before the quantity has finished changing. The value that gets committed is not the value that was checked, and nothing looks again. The rule is still sitting in the manual, fully in force, and it never fires.

Consider loss-history referral in commercial underwriting, as described in Table 1. Original policy: any account whose prior losses over the last five years exceed \$250K must be referred to an underwriting manager before it can be quoted. Submissions of this kind rarely arrive as a single document. Loss runs, supplemental carrier statements, and claim-closure notices are processed one after another - so the account’s full loss picture builds up over the course of the workflow rather than being present at the start.

A loss run arrives first and is read as \$210K of prior losses. The referral gate checks this against the \$250K line, finds it under, and sends the account down the automatic-quote path. Its authority is narrow - it applies the rule to the number in front of it. It decides nothing about what arrives next. Later, a second document arrives carrying an additional \$75K closed claim, correctly dated inside the five-year window. The account now carries \$285K - over the line. But the referral gate has already run, and nothing re-checks the total. The account is quoted automatically, carrying \$285K of prior losses, with no manager referral.

![](images/8d2edfe69b6bd9c0c893ffa579a8c1455f014015ff96c390a09cbf82cf4cfc95.jpg)  
Figure 3: Threshold Laundering. The gate at s<sub>2</sub> applies the \$250K rule correctly to the value in front of it. A later step adds a \$75K in-window claim, so the committed value breaches the threshold, and no step re-evaluates the gate.

Every step did its job. The ingestion agent read the loss run correctly. The referral agent applied the \$250K rule correctly to the number it was handed. The supplemental intake agent parsed a valid, in-window claim. None exceeded its authority, misreported a value, or omitted a required check. The policy was still broken.

This is a compositional policy violation because two checks make the difference, and both must hold:

• Local compliance: Every component stayed inside its own scope. The extractor extracted, the referral agent applied its threshold to the number it was handed, the intake agent parsed a claim. Every local check passes. No monitor watching individual steps sees anything at all.

• No single culprit: The gate alone, on the number it saw, is correct: \$210K is under the line. The intake step alone is correct: it parsed a valid \$75K claim, and escalation is not its job. Only the two composed carry the account past the line with nobody watching.

Why per-step monitoring is blind to it. The gate holds the rule but sees only the earlier \$210K - the \$75K has not arrived yet. The intake step sees the \$75K but holds no referral rule. So the one step that has the rule cannot see the final total, and the one step that creates the final total does not have the rule. Nobody ever holds both at once, and the breach slips through the seam between them. This is why no amount of tuning, calibration, or capability added to the monitor that only sees one step can catch the violation: the thing that needs to be seen is not present at any single step.

One natural objection is that the workflow should simply run the gate last. For a single quantity this works. It does not generalise, and the reason is instructive. To re-check the account after the \$75K claim, the check has to know the running total from the earlier step and apply the referral rule that belongs to the gate. A check that reaches back across steps and applies another step’s rule is no longer a local check - it is a check over the whole account, which is exactly the trace-level check we argue for. And once a workflow gates several quantities that finalise at different steps - losses after intake, premium after rating, exposure after endorsements - there is no single position late enough to see all of them at once.

<table><tr><td>Step</td><td>Action</td><td>Local check Pi</td><td>State</td><td>Result</td></tr><tr><td> $s _ { 1 }$  Loss run intake</td><td>Read the submitted loss run and extract prior losses</td><td>Are losses correctly extracted and dated?</td><td>$210K</td><td>Pass</td></tr><tr><td> $s _ { 2 }$  Referral gate</td><td>Apply the referral policy</td><td> $q ( \sigma _ { 2 } ) \leq \ S 2 5 0 \bf { K } ?$ </td><td>$210K</td><td>Pass</td></tr><tr><td> $s _ { 3 }$  Supplemental intake</td><td>Process a later document carrying an additional $75K closed claim</td><td>Is the claim correctly parsed and inside the five-year lookback window?</td><td>$285K</td><td>Pass</td></tr><tr><td>Quote (commit)</td><td>Commit to the auto-quote path at  $\sigma _ { n }$ </td><td></td><td>$285K</td><td> $P ( W ) = \operatorname { f a i l }$ </td></tr></table>

Table 1: Illustrative threshold laundering workflow. Every step passes its local check, yet the committed state breaches the \$250K referral threshold because nothing re-evaluates the gate after s<sub>3</sub>.

The violation is also not a failure of any agent’s specification. Each agent met its specification exactly. The defect is a property of how the steps compose. It can only be repaired at that level: recompute the total on the committed account and re-ask the question before the account binds.

## 3.3 Cumulative Sum Violation

Threshold evasion through gate-bypass, as seen in Threshold Laundering, represents one compositional violation pattern. A related but distinct pattern emerges when no aggregation-level gate exists at all. A Cumulative Sum Violation (CSV) is a compositional policy violation in which individually compliant actions or signals collectively cross a policy threshold, such that the violation emerges only from their aggregate state and is not observable from any individual component in isolation.

For example, consider an AI purchasing agent authorized to buy office supplies on behalf of a company. The policy imposes two constraints: no individual purchase may exceed \$100 without approval, and total daily spending may not exceed \$250. Over the course of a day, the agent executes three purchases of \$90 each. Each purchase, when evaluated independently, satisfies the per-action constraint. No individual transaction violates the policy; the violation exists only in the composition of the transactions. A control mechanism that evaluates each action independently without maintaining aggregate state cannot detect this failure mode. This pattern has regulatory precedent in transaction structuring rules, where prohibited behavior is defined over a sequence of related transactions rather than any individual transaction in isolation [18].

![](images/b6fc0f9f3db10910c73433bc7cfacbd8c3674578e7ce771550f9b1ca25884c90.jpg)  
Figure 4: Cumulative Sum Violation: three compliant purchases aggregate to violate the daily spending threshold. Each purchase independently satisfies the per-action constraint $( \mathbb { S } 9 0 < \mathbb { S } 1 0 0 )$ but their composition exceeds the policy-level constraint (\$270 > \$250)

The accumulated effect need not be a repeated sum of the same quantity. It may be a composite over heterogeneous signals: a loan application whose income verification, requested amount, and debt-to-income ratio each clear their own limit, while the composite credit-risk score they feed exceeds the referral threshold that no component computes.

Relationship to Threshold Laundering: Both are threshold violations over accumulated state, but they differ in mechanism, in detection cost, and in repair.

Mechanism: In Threshold Laundering a gate for the guarded quantity exists and evaluates the policy correctly; the defect is its temporal position, since a later step revises the quantity it read. In a Cumulative Sum Violation, no gate at the aggregate scope exists at all; each component is gated at its own scope, and the composed quantity is never evaluated by anything.

Detection cost: Once the committed state is available, detecting Threshold Laundering requires one recomputation of the guarded quantity and one comparison. Detecting a Cumulative Sum Violation requires maintaining a running aggregate over the whole trace, and where the aggregate spans entities, sessions, or invocations, correctly associating events with the entity the policy is defined over. The first is a question of when the predicate is evaluated, the second, of what state must be carried to evaluate it at all.

Repair. Threshold Laundering is repaired by re-evaluating an existing predicate on the committed state. A Cumulative Sum Violation is repaired by introducing a predicate that no step currently holds. A workflow can therefore exhibit one without the other: a correctly aggregated quantity gated too early is Threshold Laundering with no Cumulative Sum Violation, and an unaggregated quantity gated only at the level of individual actions is a Cumulative Sum Violation with no Threshold Laundering.

## 3.4 Context Collapse Violation

Context Collapse is a compositional policy violation of a different shape. A decision is committed only after an authorised person reviews the file; every hand-off along the way is a faithful summary of what it received; and yet the file the reviewer sees supports a different decision from the one the original submission supports. No summary lies. Each drops only a little, and each drop is defensible on its own - but the drops accumulate down the chain, and by the end enough has been lost to flip the call. The review was real. It was performed on a file that no longer said what the submission said.

Consider referral review in commercial underwriting, shown in Table 2. The carrier’s policy is that any account flagged for referral must receive substantive underwriter review before binding. Submissions arrive as heterogeneous packages - broker email, ACORD forms, loss runs, inspection reports - that no downstream stage reads in full, so each stage condenses what it received for the next.

Intake records five facts that matter to the decision: an ambiguous class code carrying materially different rates, an unverified sprinkler certificate, a prior large loss marked under investigation, a coverage gap between prior carriers, and a broker note reporting pending litigation. Read whole, the submission supports referral. Enrichment merges the file into a consolidated narrative and drops the sprinkler certificate, since third-party data now supplies a protection class. Risk profiling condenses the file to a rated profile: the coverage gap and the pending litigation go, neither mapping to a rating factor, and the prior loss survives as a headline figure, \$180K, with the qualifier “under investigation” removed. Every word that remains is true and the number is exact, but a loss and a loss under investigation mean different things to an underwriter. The referral packet renders the underwriter-facing file, keeping the ambiguous class code. The underwriter reads a packet carrying one of the five facts and approves. The account binds on a decision of accept, when the submission supported refer.

Nowhere did a component fail. Intake extracted accurately, each summary was a faithful condensation of its input, and the underwriter read the packet carefully and reached the decision it supported. No fact was fabricated, no step skipped, no authority exceeded. The account still bound on the wrong decision.

This pattern, the fact kept and the qualifier that told you how to read it removed, is what Lee et al. [19] term decontextualisation; they find that under a fixed budget the share of such context-setting facts falls from 25% in the source to 9% after a single compression.

Why per-step monitoring is blind to it. The failure is invisible to per-step monitoring, and invisible by design. Ask what each summary’s checker can compare against. It can only compare its summary to the file it was handed - the previous summary. It cannot compare against the original submission, because the submission is gone by then; discarding it is the whole point of summarising. So every checker verifies faithfulness to the previous step, and no checker verifies faithfulness to the source. A chain of individually faithful hand-offs is exactly what the violation is made of, so no amount of accuracy added to the individual checkers can catch it. No step-level monitor holds both the source submission and the file the reviewer sees.

Threshold Laundering and Context Collapse have opposite temporal blind spots. In Threshold Laundering, an earlier gate cannot observe policy-relevant changes that occur later in the workflow. In Context Collapse, the final decision-maker cannot recover policy-relevant information that was present in the original submission but lost through intermediate transformations.

A natural objection is that the summaries are simply too aggressive - give them more room. But the loss does not fall away with a larger budget: decision flips barely move even as the summary budget grows [19]. Bigger summaries drift more gently, but they still flip the decision. It is also not a failure of the reviewer, and not simply overtrust in the machine. Over-trust in automated output is a disposition that training or incentives can address [17]; this is not that. Even a maximally skeptical, perfectly sharp underwriter reaches the same decision, because the facts they would need are not on the page. You cannot scrutinize what is no longer there.

Then why not require each summary to preserve the decision exactly, rather than just closely? Because that check is one nobody can run. To verify that a summary preserves the same decision-relevant meaning as its input, the monitor would need to determine what decision would be reached from the full pre-summary information and compare it with the decision reached from the summary alone. In effect, this requires re-evaluating the account twice at every hand-off. A workflow able to do that at each step would not need the downstream steps or the reviewer at all. What is checked in practice is cheaper and narrower - that a summary is factually accurate and its claims trace to its input - and none of that constrains how far the decision has drifted. This is why the only real remedy is to keep the original submission and re-derive the decision from it before binding, which is a check over the whole trace rather than any single step.

## 4 Repair Topology is Dictated by Violation Structure

The correct number and placement of policy gates is not a design choice. It is dictated by where the guarded quantity is mutated over the course of the workflow, and that differs by class.

<table><tr><td>Step</td><td>Action</td><td>Local check</td><td>Facts kept</td><td>Hop drift</td><td>Result</td></tr><tr><td>s1 Intake</td><td>Extract the submission package</td><td>Facts extracted accurately?</td><td>1-5</td><td></td><td>Pass</td></tr><tr><td> $s _ { 2 }$  Enrichment</td><td>Merge third-party data into a consolidated narrative</td><td>Materially faithful to input?</td><td>1,3,4,5</td><td>0.06</td><td>Pass</td></tr><tr><td>s3 Risk profiling</td><td>Condense to a rated risk profile</td><td>Materially faithful to input?</td><td>3,5</td><td>0.08</td><td>Pass</td></tr><tr><td> $s _ { 4 }$  Referral packet</td><td>Render the underwriter-facing file</td><td>Materially faithful to input?</td><td>1</td><td>0.07</td><td>Pass</td></tr><tr><td>S5 Review</td><td>Underwriter reads the packet and approves</td><td>Authorized person reviewed?</td><td>1</td><td></td><td>Pass</td></tr><tr><td>Bind</td><td>Commit on the underwriter&#x27;s approval</td><td></td><td>1</td><td>0.19</td><td>Fail</td></tr></table>

Table 2: Illustrative context collapse workflow. Each hop stays under the materiality tolerance, but the drift accumulates to 0.19 relative to the submission, flipping the decision from refer to accept.

Consider threshold laundering, shown in figure 5: risk is not a static property but a quantity that mutates across steps. A single gate at step 1 sees risk = 0.79, passes, and implicitly assumes that value will not change. Step 3 adds exposure the gate never observes, and the value crosses the threshold 0.80 in a place where nobody was looking. The gate enforced a point-in-time predicate, risk < 0.80 at step 1, when the governing policy specifies a invariant: risk < 0.80 on the committed state.

![](images/49e0616ee0479ca87ec889f0914a9430f779e38634f3ce400a10c8b60fe97672.jpg)  
Figure 5: Gate Placement for Threshold Laundering

The same principle governs the other three classes, though what mutates differs. In authority creep, what mutates is not a scalar but the representation of who is entitled to decide: the escalation router already sits at the only step where escalation authority is exercised, so there is no later position to move it to. What must be added is not a new position but backward reach. The router must be able to trace the count it fires on through the domain-scoped transforms that produced it, since the mutation happened upstream of any single gate rather than after one. In Cumulative Sum Violation(CSV), the mutation is a running total that accretes across independently-gated actions, and no gate exists at the scope where that total is tracked at all; placement is not wrong here so much as absent, and the fix is to instantiate a gate at the aggregate’s own scope rather than to relocate an existing one. In Context Collapse Violation, what mutates is the evidentiary record itself: each hand-off is a faithful summary of what it received, so the terminal review gate is correctly placed and correctly applied, but the file it evaluates is no longer the file the policy was written against. The mutation is in what the gate can see, not in the guarded quantity or in where the gate sits.

Each class therefore demands a different repair topology:

• Authority Creep: Trace backward once from the decision point.

• Threshold Laundering: Re-evaluate the existing predicate on the committed state rather than the value present when the gate first fired.

• Cumulative Sum Violation: Introduce a predicate at the aggregate’s own scope, since no step currently holds one.

• Context Collapse: Reconstruct against the original submission rather than the intermediate summaries.

This is why the detector cannot be a collection of singlepoint checkpoints. It must reconstruct full policy semantics and re-evaluate against the final committed state, which only the complete trace provides.

## 5 CPV Detection Architecture

The repair topologies established in Section 4 cannot be implemented by a single monolithic check. Each demands a different kind of infrastructure: a persistent, unabridged record of the execution to trace or monitor against, and the ability to recompute a guarded quantity from that record rather than trust an intermediate representation a prior step may have already laundered. The architecture therefore separates the concern of preserving the trace from the concern of evaluating policy against it. The detector comprises four core stages:

1. Provenance Ingestion. Collect and normalize runtime events into a canonical, timestamped trace.

2. State Reconstruction. Rebuild workflow state and policy-relevant history from raw provenance.

3. Policy Evaluation. Evaluate both step-level and workflow-level policies against reconstructed state.

4. Compositional Detection. Identify violations that emerge only at the sequence level and classify by CPV type.

Two design invariants govern all four:

• History-completeness. No component may render a violation verdict from a truncated or windowed view when the governing policies semantics require the full trajectory.

• Recount-from-raw-provenance. Composition gates must recompute guarded quantities from raw provenance under the policies own semantic definition, never from the pipeline’s transformed or derived representation.

![](images/65c5911626ea43007faff3228e228127de4b66eadbba7282af93f52e3e422c84.jpg)  
Figure 6: CPV Detector architecture. Four detection stages, governed throughout by two design invariants, with operational components that support governance workflows but do not contribute to detection logic

## 5.1 Stage 1: Provenance Ingestion and Normalization

• Workflow Trace Collector: Captures every agentrelevant event, including prompts, tool calls, approvals, memory writes, external API responses, file access, retries, and handoffs. It ingests raw runtime events, tool telemetry, orchestrator logs, and identity context and produces an immutable, ordered event log for each workflow instance with monotonic sequence numbers and timestamps.

• Event Normalizer: Converts heterogeneous runtime events into a unified workflow-trace schema while preserving raw provenance references. It consumes raw events from collectors and produces canonical events containing instance identifiers, sequence numbers, timestamps, actors, action types, typed parameters, and provenance links. Internally, it maps framework-specific artifacts into standardized event categories and enriches them with semantic attributes such as action class, resource class, sensitivity level, and purpose.

These two components establish the complete, auditable record required by history-completeness. Nothing is discarded; later stages see the full execution history.

## 5.2 Stage 2: State Reconstruction

• Governed State Store: Persists workflow state and the raw provenance ledger for each workflow instance, enabling recount-from-raw analysis. It ingests raw payloads from the Workflow Trace Collector, canonical events from the Event Normalizer, and derived annotations from downstream engines, providing queryable current and historical state with provenance retrieval by instance ID.

• Policy Knowledge Base (PKB): Stores machinereadable policies, control objectives, exceptions, and versioned policy definitions, serving as the authoritative source for policy semantics required by the recount principle.

These components implement the recount-from-rawprovenance principle: the State Store holds the raw facts, the PKB holds the policy semantics, and together they enable the detector to reconstruct what the policy actually requires at each decision point.

## 5.3 Stage 3: Policy Evaluation

• Policy Evaluation: Evaluates declarative policies, authority constraints, and separation-of-duty rules against the workflow trace and governed state. It produces policy-level pass/fail decisions with supporting facts. This engine answers two questions:

1. Do all individual steps satisfy their local policies? (step-level compliance)

2. Does the composed workflow satisfy trace-level policies? (workflow-level compliance)

The output is the raw material for compositional detection: a record of which steps are locally compliant, and which policies defined over the complete trace are violated.

## 5.4 Stage 4: Compositional Detection

• Sequence Analyzer: This layer analyzes complete workflow trajectories to identify compositional policy violations that emerge only across multiple steps. The Sequence Analyzer performs the core detection logic, and the check it runs follows the repair topology of each class:

– Authority Creep. Traces authority ownership backward from the decision gate to determine whether accumulated permissions govern every input the gate depends on.

– Threshold Laundering. Re-evaluates the guarded quantity on the committed state and compares it against the value present when the gate originally fired.

– Cumulative Sum Violation. Maintains the aggregate over the trace and compares it against the policy threshold no step holds.

– Context Collapse. Re-derives the decision from the retained submission and compares it against the decision the reviewed file supports.

• Violation Detector: Produces authoritative binary or graded violation verdicts for each workflow instance and maps detected issues to the CPV taxonomy with violation type, severity, and the responsible workflow segment. Confirmed findings are classified into known CPV categories, while unmatched patterns are flagged as candidates for further analysis.

## 5.5 Operational Components

The detector also includes components supporting deployment and governance workflows:

• Explainability Layer: Generates human-auditable justifications that trace each violation verdict to the responsible workflow events, raw provenance records, and policy definitions. This layer provides explanations to the Governance Dashboard and Audit Log Generator.

• Audit Log Generator: Produces tamper-evident, regulator-grade records of violation verdicts, explanations, and supporting evidence. The audit log serves as a durable sink for records from upstream components.

• Governance Dashboard & Alerting Pipeline: Provides portfolio- and instance-level visibility into CPV posture and routes actionable CPV findings to human reviewers and enforcement systems.

These are operational infrastructure supporting the core detection logic; they do not contribute to CPV identification itself.

The CPV Detector architecture operationalizes trace-level governance through four core detection stages: Provenance Ingestion normalizes runtime events into a canonical trace; State Reconstruction maintains complete history and raw provenance; Policy Evaluation checks both step-level and workflow-level policies; and Compositional Detection identifies violations that emerge only at the sequence level. This linear flow is anchored by two foundational design invariants: history-completeness ensures no violation verdict is rendered from a truncated view, and recountfrom-raw-provenance ensures all policy-relevant quantities are recomputed from original provenance, not intermediate pipeline representations. These invariants are not optional refinements - they are structural necessities. This is why the architecture cannot collapse into step-level checks; it must work at the workflow level, treating policy enforcement as a flow-level invariant rather than a post-hoc monitoring layer.

## 5.6 Detector Failure Modes

The architecture inherits limits from the trace it is given, and we state them explicitly.

Provenance loss: The detector can only recompute a guarded quantity from what the pipeline retained. If a transform deletes suppressed evidence rather than marking it, the pre-image is unrecoverable and the violation is undetectable in principle - not hard, but impossible, since no information remains in the trace from which the truth is derivable. Provenance retention is therefore a precondition rather than an implementation detail, and a pipeline that discards what it suppresses cannot be audited for Authority Creep regardless of what is added downstream.

Oracle dependence for subjective predicates: Recomputation is sound when the guarded quantity is objective. Many predicates are not: whether an exception is material applies a standard written in prose. The detector then requires an interpretation oracle, which becomes part of its trusted base. If that oracle and the pipeline component that produced the violation are both language models reasoning from similar priors, they may reach the same judgement; the detector sees no discrepancy and concurs. A recount is sound only where its oracle is independent of the pipeline’s and anchored to the policy text rather than to learned precedent.

Extraction error: Even with full provenance, recomputation from documents inherits extraction error: boundaryband dates misclassified against a lookback window, the same loss double-counted under distinct claim numbers, differing totals read from different documents. Trace-level evaluation removes a structural blindness; it does not remove measurement error.

Partial and asynchronous observability: The architecture assumes an ordered event log per workflow instance. Distributed execution, clock skew, out-of-order delivery, and external tools that do not emit telemetry all degrade this assumption, and runtime verification under incomplete or distributed observation is known to admit inconclusive verdicts [20, 21].

Entity resolution: Cumulative Sum Violations are defined over the aggregate belonging to an entity. Fragmented identities, aliases, or multiple accounts prevent construction of the correct aggregate, and no amount of trace retention repairs an aggregate assembled over the wrong partition.

Hidden state: Agent memory, retrieved context, and reasoning not surfaced as events lie outside the trace. Where policy-relevant state is carried in such channels, the reconstructed trajectory is incomplete and the detector’s verdict is correspondingly weaker.

Cost: Retaining raw provenance and recomputing guarded quantities at commit time imposes storage and latency costs that scale with trace length and with the number of gated quantities; for Context Collapse in particular, re-deriving the decision from the source submission approaches the cost of the un-automated decision itself.

## 6 Future Research

As governance systems scale to handle compositional violations, key challenges emerge around deployment, generalization, and trust. Operationalizing the detector introduces engineering considerations, provenance collection overhead, persistent event log storage, latency in real-time policy evaluation, and efficient state reconstruction in distributed workflows. These are important implementation concerns but are orthogonal to the core detection mechanism. Production deployments will need to make storagelatency-accuracy trade-offs specific to workflow volume and SLA requirements, leveraging tiered storage, batch vs. streaming policy evaluation, and incremental state reconstruction.

Beyond implementation, standardized frameworks for expressing governance logic and mapping it to workflow provenance remain absent. Whether violation mechanisms and detector architecture transfer to lending, hiring, moderation, and other high-stakes domains is unclear. Remediation strategies, accountability assignment, and defenses against adversarial obfuscation of audit trails raise practical and ethical questions beyond detection itself.

Future research should focus on: (i) standardized policy languages and semantic mappings between governance intent and workflow provenance, enabling transparent detection across domains; (ii) simulation testbeds and benchmarked datasets analogous to those in fairness and interpretability to lower barriers to entry and enable performance comparison; (iii) efficient trajectory analysis and real-time provenance capture to bring detection costs within operational feasibility; and (iv) domain-specific instantiation and validation in regulated workflows

## 7 Conclusion

We have identified compositional policy violations as a distinct and consequential class of governance failure: sequences where every individual step is locally policycompliant yet the composed trajectory violates organizational governance, regulatory obligations, or business intent. By formalizing four violation mechanisms, deriving the repair topology each one demands, and proposing a detector architecture that describes on how to reconstruct full policy semantics from raw provenance, we have established that point-in-time checkpoint enforcement is structurally insufficient. This work opens a research agenda in policy-aware AI systems design - one that treats policy enforcement not as an external checkpoint but as a standing property of the workflow, holding across the full trace rather than at isolated decision points. As agentic AI workflows and production AI systems grow in autonomy and composition, the ability to detect violations that emerge from sequences of individually compliant steps becomes not merely an engineering concern but a prerequisite for organizational accountability and regulatory compliance. Compositional violations are not edge cases; they are inherent to how autonomous agents compose decisions at scale, and detecting them is foundational to trustworthy deployment.

## References

[1] Anthropic. Claude for financial services. https: //www.anthropic.com/news/claude-for-fin ancial-services, 2025. Official announcement; contains AIG CEO statement.

[2] Allianz. Allianz and anthropic forge global partnership to advance responsible AI in insurance. https: //www.allianz.com/en/mediacenter/news/ media-releases/260109-allianz-and-anthr opic-forge-global-partnership.html, 2026. Official press release.

[3] The Baldwin Group. The baldwin group announces expanded enterprise relationship with anthropic. ht tps://ir.baldwin.com/news-releases/ne ws-release-details/baldwin-group-annou nces-expanded-enterprise-relationship/, 2026. Official press release.

[4] Hakan Inan, Kartikeya Upasani, Jianfeng Chi, Rashi Rungta, Krithika Iyer, Yuning Mao, Michael Tontchev, Qing Hu, Brian Fuller, Davide Testuggine, and Madian Khabsa. Llama guard: LLM-based inputoutput safeguard for human-AI conversations. arXiv preprint arXiv:2312.06674, 2023. Preprint.

[5] Traian Rebedea, Razvan Dinu, Makesh Narsimhan Sreedhar, Christopher Parisien, and Jonathan Cohen. NeMo Guardrails: A toolkit for controllable and safe LLM applications with programmable rails. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pages 431–445, 2023. Peer-reviewed.

[6] Langfuse. Observability data model. https://lang fuse.com/docs/observability/data-model, 2026. Official documentation; accessed 2026.

[7] LangChain. Observability concepts. https://docs .langchain.com/langsmith/observability -concepts, 2026. Official documentation; accessed 2026.

[8] Mrinank Sharma, Meg Tong, Jesse Mu, Jerry Wei, Jorrit Kruthoff, Scott Goodfriend, Euan Ong, Alwin Peng, Raj Agarwal, Cem Anil, and Ethan Perez. Constitutional classifiers: Defending against universal jailbreaks across thousands of hours of red teaming. arXiv preprint arXiv:2501.18837, 2025. Preprint (Anthropic).

[9] Haoyu Wang, Christopher M Poskitt, and Jun Sun. Agentspec: Customizable runtime enforcement for safe and reliable llm agents. arXiv preprint arXiv:2503.18666, 2025.

[10] Tanzim Ahad, Ismail Hossain, Md Jahangir Alam, Sai Puppala, Yoonpyo Lee, Syed Bahauddin Alam, and Sajedul Talukder. Semantic intent fragmentation: A single-shot compositional attack on multi-agent ai

pipelines. In Proceedings of the AAAI Symposium Series, volume 9, pages 229–237, 2026.

[11] Mahdi Azarafrooz. Cross-session threats in AI agents: Benchmark, evaluation, and algorithms. arXiv preprint arXiv:2604.21131, 2026. Preprint.

[12] Niloofar Mireshghallah, Neal Mangaokar, Narine Kokhlikyan, Arman Zharmagambetov, Manzil Zaheer, Saeed Mahloujifar, and Kamalika Chaudhuri. CIMemories: A compositional benchmark for contextual integrity of persistent memory in LLMs. In International Conference on Learning Representations (ICLR), 2026. Peer-reviewed.

[13] Aniruddha Dhodapkar and Fatima Pishori. Safety-Drift: Predicting when AI agents cross the line before they actually do. arXiv preprint arXiv:2603.27148, 2026. Preprint.

[14] Hunter Lightman, Vineet Kosaraju, Yura Burda, Harri Edwards, Bowen Baker, Teddy Lee, Jan Leike, John Schulman, Ilya Sutskever, and Karl Cobbe. Let’s verify step by step. In International Conference on Learning Representations (ICLR), 2024. Peerreviewed.

[15] Mert Cemri, Melissa Z. Pan, Shuyi Yang, Lakshya A. Agrawal, Bhavya Chopra, Rishabh Tiwari, Kurt Keutzer, Aditya Parameswaran, Dan Klein, Kannan Ramchandran, Matei Zaharia, Joseph E. Gonzalez, and Ion Stoica. Why do multi-agent LLM systems fail? In Advances in Neural Information Processing Systems (NeurIPS), 2025. Peer-reviewed.

[16] Ben Green. The flaws of policies requiring human oversight of government algorithms. Computer Law & Security Review, 45:105681, 2022. Peer-reviewed.

[17] Kate Goddard, Abdul Roudsari, and Jeremy C Wyatt. Automation bias: a systematic review of frequency, effect mediators, and mitigators. Journal of the American Medical Informatics Association, 19(1):121–127, 2012.

[18] United States Code. 31 u.s.c. § 5324: Structuring transactions to evade reporting requirements, 2024. Bank Secrecy Act provisions regarding transaction structuring.

[19] Hoyoung Lee, Suhwan Park, Seunghan Lee, Jun Seo, Jaehoon Lee, Sungdong Yoo, Minjae Kim, Cheol-Won Na, Zhangyang Wang, Zach Golkhou, et al. When summaries distort decisions: Information fidelity in llm-compressed financial analysis. arXiv preprint arXiv:2606.29251, 2026.

[20] David Basin, Felix Klaedtke, and Eugen Zalinescu.ˇ Failure-aware runtime verification of distributed systems. In Proceedings of the 35th IARCS Annual Conference on Foundations ofSoftware Technology and Theoretical Computer Science, 2015.

[21] Andreas Bauer, Martin Leucker, and Christian Schallhart. Runtime verification for ltl and tltl. ACM Transactions on Software Engineering and Methodology (TOSEM), 20(4):1–64, 2011.