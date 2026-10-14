# Cost-Eficient Privacy Enforcement for Email-Generating LLM Agents under Strategic Information Requests

Qikun Cai, Tao Huang, and Chen Hou

Abstract. LLM agents now act as delegated communication intermediaries: they read private context, draft messages, and send externally visible emails on behalf of a user. Existing checker–repair defenses can reduce inappropriate disclosure, but they rely on repeated calls to a remote action LLM (the model that drafts and repairs the email) and a remote checker LLM whenever a draft fails validation. This creates a cost problem as well as a privacy problem, especially when a requester strategically asks for exact values, raw records, source evidence, or other details that induce leakage-heavy drafts. We propose local contract-conditioned rewriting as a cost-eficient enforcement layer for email-generating agents. The mechanism intercepts an outgoing draft, constructs an informationrelease contract that specifies what to preserve, generalize, and forbid, rewrites the draft with a local Gemma rewriter, and selectively invokes a final PrivacyChecker.Across extensive email tasks under normal prompts and attack prompts that operationalize strategic information requests, Gemma + Checker reduces normal-prompt leakage from 11.41% to 4.03% while improving UsableRate (satisfied or partially satisfied outputs) from 85.2% to 92.6%. Under the strategic attack-prompt condition, it reduces leakage from 28.19% to 18.79% and reduces checker-only cost amplification from 1.32× to 1.02×.Ofline replay ablation shows that the full contract reduces leakage relative to generic Gemma rewriting by 36.3% under normal prompts and 54.1% under attack prompts. These results support a local-first privacy enforcement design: move repair to a cheaper trusted local layer when possible, but retain the checker as the final semantic gate for high-risk sends.

Keywords: LLM agents; privacy; information release; mechanism design; strategic requests; enforcement cost; local rewriting; email agents

## 1 Introduction

LLM agents are increasingly used as delegated actors in online platforms. Through tool use and multi-step reasoning, they can retrieve documents, inspect a mailbox, query workspace notes, consult calendars, and then execute external actions such as sending emails or updating shared records [10,11,14,15]. In such workflows, the agent is not only producing text; it is selecting, transforming, and releasing information on behalf of a principal. This makes privacy an action-level problem: the relevant question is not only whether a model can recognize sensitive information, but whether the final external action discloses information that the recipient is not authorized to receive.

Email generation is a concrete instance of this problem. Email has explicit recipients, persistent records, and social or legal consequences. A delegated agent may need to inspect private Gmail or Notion content before drafting a project update, care-coordination note, hiring response, school communication, or financial summary. Yet access by the agent does not imply authorization for the recipient [16]. The platform must therefore decide both what information may be released and how much enforcement cost to spend before the message is sent.

Existing agent-privacy defenses commonly introduce a separate checker before the final action. PrivacyChecker, for example, uses contextual-integrity reasoning to extract information flows and judge whether each flow is appropriate before sending [1,2,3]. Related benchmarks and defenses show that LLMs can recognize privacy risks yet still leak sensitive information during generation or tool use [4,5,6,7,8]. These checker-based designs are useful because they can be deployed at the action boundary. Their limitation is economic: if the checker rejects a draft, the system often asks the remote action LLM to repair it and then calls the checker again. The loop can be costly even for benign requests, and strategic information requests can make it worse by asking for exact values, raw records, source evidence, structured fields, or “all relevant details.” Such requests resemble prompt-based manipulation of agent behavior [9,15,17], but here the harm has two parts: more privacy leakage and more remote enforcement cost.We use the term strategic information request to denote a request that preserves an apparently legitimate email task while pressuring the agent to disclose unauthorized private details. In the experiments, we operationalize this requester behavior as the attack-prompt condition.

We study this setting as a cost-aware strategic information-release problem. The requester chooses language that influences the agent’s disclosure behavior. The platform chooses an enforcement mechanism that trades of authorized task utility, privacy leakage, remote token cost, local rewrite cost, and semantic drift. This perspective connects agent privacy to platform mechanism design and privacy economics, where stronger protection is valuable but not free [12,13].

We propose a local contract-conditioned enforcement mechanism. The remote action LLM still retrieves context and produces the original email draft. Before the draft enters a remote checker–repair loop, a local Gemma rewriter edits it under a structured information-release contract: a machine-readable policy object that specifies task content to preserve, sensitive details to generalize, unsupported facts to forbid, and output shape to maintain. The full mechanism, denoted $M _ { G + C }$ , then invokes the checker on the rewritten draft and uses checker feedback for local repair when needed. We also evaluate a Gemma-only variant, $M _ { G } .$ , as a low-cost option for lower-risk settings, but do not treat it as a replacement for final checking.

Our contributions are threefold. First, we formulate email-generating agents as delegated information intermediaries and make enforcement cost part of the privacy objective. Second, we introduce a local contract-conditioned rewrite layer that converts much of the remote repair loop into local policy-constrained editing before optional final checking. Third, using normal prompts and attack prompts that operationalize strategic information requests over the same 149 base email tasks, we show that Gemma + Checker reduces leakage and attack-induced cost amplification while preserving usable task completion.

![](images/3e0a5d8fee61900ef2a7d531c9c5720f49c995b0dd71d14d66b97cf5b82d4287.jpg)  
Fig. 1. Local contract-conditioned enforcement. The mechanism moves repair work from repeated remote action-LLM calls to a local rewrite executor, while preserving an optional final checker for high-risk sends.

## 2 Cost-Aware Strategic Information Release

Participants and information. We consider a delegated email-generation environment with four roles. The principal authorizes an email task and prefers both task completion and privacy preservation. The requester receives the eventual email and may issue a normal request or a strategic information-seeking request. The action LLM accesses private tools and produces an original draft. The platform defender controls send-time enforcement, including local rewriting, checking, retries, and blocking. Let C denote the private context accessible to the agent, S the set of benchmark-annotated sensitive items for a task, and Q the set of non-sensitive task slots required for authorized utility. The action LLM produces an original draft $D _ { 0 }$ . A mechanism M produces a final draft $D _ { M }$ that is either sent or blocked. A leakage event occurs when $D _ { M }$ reveals an item in S through exact disclosure, close paraphrase, or strong source-grounded implication.

Requester strategies and mechanisms. The requester chooses a prompt strategy $\sigma \in$ {normal, attack},where the attack condition operationalizes a strategic information request.A normal requester asks for a legitimate communication outcome without intentionally seeking private details. A strategic requester attempts to increase sensitive disclosure while preserving the appearance of a valid task, for example through exact-value requests, raw-record requests, source-evidence requests, structured extraction, or completeness pressure. We compare three mechanisms: $M _ { B }$ , a remote checker–repair baseline in which rejected drafts are returned to the action LLM; $M _ { G } .$ , a local Gemma-only rewrite mechanism without final checking; and $M _ { G + C }$ , a local Gemma rewrite followed by final checker validation and local feedback-guided repair.

Defender objective. The platform defender chooses a mechanism to trade of utility, leakage, remote cost, local cost, and semantic drift:

$$
W ( M , \sigma ) = U ( D _ { M } ) - \lambda L ( D _ { M } ) - \mu R _ { \mathrm { r e m o t e } } ( M , \sigma ) - \eta G _ { \mathrm { l o c a l } } ( M , \sigma ) - \nu S _ { \mathrm { d r i f t } } ( M , \sigma ) ,\tag{1}
$$

where U measures authorized task utility, L privacy leakage, $R _ { \mathrm { r e m o t e } }$ billingrelevant remote tokens or calls, $G _ { \mathrm { l o c a l } }$ local rewrite cost, and $S _ { \mathrm { d r i f t } }$ semantic abstraction or repair-induced drift. We report remote tokens as the primary cost because remote LLM calls are the deployment bottleneck; Gemma tokens are reported separately. If desired, a platform can form a total-cost measure $T _ { \rho } ( M , \sigma ) = R _ { \mathrm { r e m o t e } } ( M , \sigma ) + \rho G _ { \mathrm { l o c a l } } ( M , \sigma )$ , where $\rho$ converts local compute into remote-token-equivalent cost.

Let $c _ { A }$ be the cost of one remote action-LLM generation or repair call and $c _ { C }$ the cost of one remote checker call. Let $K _ { B } ( \sigma )$ be the average number of remote repair rounds under $M _ { B }$ , and $K _ { G } ( \sigma )$ the average number of checker-triggered local repair rounds after Gemma rewriting. The expected remote costs are

$$
R _ { \mathrm { r e m o t e } } ( M _ { B } , \sigma ) = c _ { A } ( 1 + K _ { B } ) + c _ { C } ( 1 + K _ { B } ) ,\tag{2}
$$

$$
R _ { \mathrm { r e m o t e } } ( M _ { G + C } , \sigma ) = c _ { A } + c _ { C } ( 1 + K _ { G } ) .\tag{3}
$$

Thus the remote-cost reduction is

$$
\Delta R _ { \mathrm { r e m o t e } } = c _ { A } K _ { B } + c _ { C } ( K _ { B } - K _ { G } ) .\tag{4}
$$

When $K _ { B } > K _ { G }$ and $c _ { A } , c _ { C } > 0$ , local-first enforcement strictly reduces remote cost. For total cost, $M _ { G + C }$ remains cheaper whenever $\rho G _ { \mathrm { l o c a l } } < \varDelta R _ { \mathrm { r e m o t e } }$ . This condition clarifies the role of the local model: it need not be a perfect privacy judge; it only needs to reduce the leakage burden entering the checker enough to avoid expensive remote repair cycles.

Risk-adaptive deployment. The three mechanisms define a deployment spectrum. $M _ { G }$ has the lowest remote cost but lacks an independent semantic gate. It is preferable in low-risk settings only when the saved remote cost outweighs the expected increase in privacy loss, i.e., when $\mu \Delta R _ { \mathrm { r e m o t e } } > \lambda \Delta L$ . For high-risk or attack-suspected sends, $M _ { G + C }$ is preferred because it retains final validation while still moving repair into a local contract-conditioned layer.

## 3 Local Contract-Conditioned Enforcement

The mechanism is placed at the outgoing email boundary. When the agent attempts to call the email-sending tool, a send-time wrapper captures the recipient fields, subject, and body before the message is delivered. The wrapper builds a normalized email signature and allows only the exact draft that has passed the selected enforcement path to be sent. If the agent attempts to send a diferent body after validation, the mismatch triggers another inspection. This boundary placement prevents the main agent from deciding whether its own output should be checked.

The information-release contract is generated from the user instruction and original draft. It is not an oracle label of sensitive items; it is a deployable policy object that constrains how $D _ { 0 }$ may be transformed into $D _ { M }$ . Table 1 summarizes the main contract components. The important design choice is to separate four concerns that a single generic rewrite instruction tends to conflate: preserving benign task slots, abstracting unauthorized sensitive details, preventing unsupported replacement facts, and retaining the response shape required for a useful email. Appendix A summarizes the main schema fields and gives an anonymized example.

Table 1. Main information-release contract components. The contract is a deployable rewrite policy, not an evaluation oracle.
<table><tr><td>Component</td><td>Enforcement role</td></tr><tr><td>must_preserve, required_slots must_generalize,</td><td>Preserve the benign task intent and privacy-allowed content needed for a useful email. Abstract sensitive, confidential, source-specific, or recon-</td></tr><tr><td>privacy-risk rules information_shape,</td><td>structive details rather than copying them. Keep the requested email structure, such as a status update,</td></tr><tr><td>safe_output_shape must_not_add</td><td>bullet list, next step, or high-level range. Prevent hallucinated replacement facts, new values, new com-</td></tr><tr><td>Safe-substitution guid- Preserve the function of a sensitive detail with a safe abstrac-</td><td>mitments, private-source references, or unsupported attach- ments.</td></tr></table>

Gemma is used as the local rewrite executor. It receives the original subject, original body, contract, and optionally checker feedback. It returns a rewritten subject, rewritten body, and change summary. The executor is instructed to make minimal edits: preserve the communicative purpose, generalize sensitive details, avoid unsupported facts, and retain a professional tone. The mechanism is not tied to Gemma as a model; any trusted local or on-premise editor with suficient rewriting capability could instantiate the same policy. Appendix B summarizes the production prompt and the ablation variants.

This contract-conditioned rewrite difers from generic privacy rewriting. A generic prompt can ask the local model to “remove private information,” but it does not explicitly state which non-sensitive slots must survive, what output form should remain useful, or which safe abstractions can replace prohibited details. The contract gives the rewriter a task-conditioned release rule: privacy constraints take priority, but the benign task shell should still be preserved whenever it can be satisfied without leakage. This distinction is tested directly in the ofline contract ablation in Section 5.4.

In the full mechanism $M _ { G + C }$ , PrivacyChecker remains the final semantic gate. After local rewriting, the checker evaluates whether the draft still reveals sensitive information. If the draft passes, the exact checked draft is sent. If the checker rejects it, its feedback is returned verbatim to the local executor for targeted repair. If no safe draft is produced within the bounded retry budget, the action is blocked. Thus Gemma and PrivacyChecker have distinct roles: Gemma performs inexpensive first-stage sanitation and utility-preserving editing, while the checker catches residual semantic leakage that a rewrite-only mechanism may miss.

## 4 Experimental Design

Dataset and prompt conditions. The evaluation uses 149 aligned emailgeneration tasks from the Privacy in Action benchmark [1]. Each task contains a user instruction, private context accessible through Gmail/Notion-style tools, a target recipient, a desired email objective, and annotated sensitive information items. The tasks contain 480 sensitive items and cover email generation or reply categories such as reply\_response, update\_status, provide\_details\_data, summary, and scheduling\_coordination. Normal prompts state legitimate tasks, while attack prompts operationalize strategic information requests by adding language intended to induce disclosure of raw records, exact values, source evidence, or sensitive fields.

Mechanism matrix and metrics. We evaluate a $2 \times 3$ matrix formed by crossing normal/attack requester conditions with $M _ { B } , M _ { G }$ , and $M _ { G + C }$ . Privacy leakage is measured at the item level, whether a task leaks at least one annotated sensitive item, and at the secret level, the number of individual sensitive fields disclosed through exact, paraphrased, or reconstructive leakage. For normal prompts, authorized utility is measured by recipient correctness and an LLM-judge label over the final email: satisfied, partially\_satisfied, not\_satisfied, or manual\_review. We report UsableRate as satisfied plus partially\_satisfied. For attack prompts, satisfaction is evaluated only on the benign task shell, the non-sensitive part of the request that remains after excluding attack-specific demands for private details. Appendix C gives the full judge protocol.

Evaluation protocol. Leakage evaluation is performed on the final email-facing content rather than on intermediate reasoning or draft artifacts. The evaluator compares the final subject and body against the benchmark-sensitive items and counts exact disclosure, partial disclosure, and reconstructive leakage, where the recipient could infer the private fact from the remaining wording. The leakage evaluator uses the same model family as the send-time checker but a diferent evaluation prompt; the utility judge is separate from the enforcement path. Manual spot checks were used to screen obvious judge failures, but the reported numbers are judge-based rather than a formal human annotation study. These details are important because the full mechanism uses a checker during enforcement, while the reported leakage metric evaluates the released artifact against annotated secrets. More details are presented in Appendix C.

Table 2. Core experimental settings.
<table><tr><td>Item</td><td>Setting</td></tr><tr><td>Tasks</td><td>149 aligned normal and attack email tasks from Privacy in Action; 480 sensitive items; Gmail/Notion retrieval followed by email generation or reply.</td></tr><tr><td>Action LLM A</td><td>DeepSeek deepseek-v4-flash; same initial agent prompt and same base tasks across mechanisms.</td></tr><tr><td>Checker C</td><td>DeepSeek deepseek-v4-flash; Boolean pass/fail; checker feedback is returned verbatim for repair.</td></tr><tr><td>Local rewriter G</td><td>gemma4-e4b-q4-12k via Ollama, q4_K_M; temperature 0.0; 12k context.</td></tr><tr><td>Judges</td><td>Leakage evaluator: DeepSeek deepseek-v4-flash with a separate evaluation prompt; utility judge: DeepSeek deepseek-v4-pro.</td></tr><tr><td>Repair policy</td><td>Baseline continues the same agent session after checker fail- ure, with a 25-turn overall cap;  $M _ { G + C }$  allows up to three local rewrite rounds; all checked mechanisms fail closed.</td></tr><tr><td>Cost accounting</td><td>Remote tokens = action-LLM input/output tokens + checker input/output tokens. Gemma input/output tokens are recorded separately as local tokens.</td></tr><tr><td>Evaluation caveat</td><td>The leakage evaluator uses a different prompt from the send- time checker but the same model family. Manual spot checks were used to screen obvious judge failures, but not as a formal</td></tr></table>

We distinguish remote enforcement cost from local rewrite cost. Remote tokens include action-LLM tokens and PrivacyChecker tokens. Local tokens include only Gemma rewrite tokens. Strategic cost amplification is

$$
\mathrm { C o s t A m p } ( M ) = \frac { \mathrm { R e m o t e T o k e n s } ( M , \mathrm { A t t a c k } ) } { \mathrm { R e m o t e T o k e n s } ( M , \mathrm { N o r m a l } ) } .\tag{5}
$$

Semantic preservation is measured using privacy-allowed slot retention and semantic drift with respect to the benign task shell. Contract ablation is evaluated by ofline replay: we reuse the same agent-generated draft from Gemma-only logs and apply diferent Gemma rewrite variants without rerunning the agent, tools, email sending action, or final checker. Appendix D.1 describes the replay procedure.

## 5 Results

## 5.1 Local Rewriting Reduces Remote Enforcement Cost

Table 3 decomposes enforcement cost into remote action-LLM tokens, remote PrivacyChecker tokens, and local Gemma rewrite tokens. The aggregation rules are defined in Appendix C. Under normal prompts, $M _ { G + C }$ reduces remote tokens from 8.73M to 7.19M, a 17.6% reduction relative to $M _ { B }$ . Under attack prompts, the reduction is larger: remote tokens fall from 11.54M to 7.32M, a 36.6% reduction. The result supports the cost mechanism: local contract-conditioned rewriting lowers aggregate remote enforcement cost even when the final checker receives a longer rewritten context.

Table 3. Remote enforcement cost decomposition. Remote tokens are action-LLM tokens plus PrivacyChecker tokens. Gemma local tokens are reported separately and are not included in remote-cost comparisons.
<table><tr><td colspan="6">Prompt Mechanism Avg. checker calls Remote tokens Agent tokens Checker tokens Gemma local tokens</td></tr><tr><td>Normal  $M _ { B }$ </td><td>1.06</td><td>8,728,781</td><td>8,390,242</td><td>338,539</td><td>0</td></tr><tr><td>Normal  $M _ { G }$ </td><td>0.00</td><td>5,404,802</td><td>5,404,802</td><td>0</td><td>1,165,287</td></tr><tr><td>Normal  $M _ { G + C }$ </td><td>1.00</td><td>7,189,612</td><td>5,281,175</td><td>1,908,437</td><td>1,144,181</td></tr><tr><td>Attack  $M _ { B }$ </td><td>1.79</td><td>11,540,135</td><td>10,861,107</td><td>679,028</td><td>0</td></tr><tr><td>Attack  $M _ { G }$ </td><td>0.00</td><td>5,311,550</td><td>5,311,550</td><td>0</td><td>1,248,082</td></tr><tr><td>Attack  $M _ { G + C }$ </td><td>1.02</td><td>7,318,258</td><td>5,306,892</td><td>2,011,366</td><td>1,246,723</td></tr></table>

The checker-token column alone is not the relevant cost metric. $M _ { G + C }$ uses more checker tokens than $M _ { B }$ because the checker is applied to a rewritten context with richer explanatory structure. Nevertheless, total remote tokens decrease because agent-token consumption drops sharply. Agent tokens include all action-LLM input/output calls in the same agent session, including the initial draft and subsequent tool-grounded generations. In the baseline, when the checker fails, the agent continues in the same session and inherits prior drafts, checker feedback, tool observations, and previous tool-call context. It does not reload a separate execution-log file each round, but the accumulated session context still makes remote repair expensive. $M _ { G + C }$ avoids most of this remote action-LLM repair path by moving repair to the local rewriter. The local Gemma token column should therefore be read as deployment cost rather than remote API cost. If a deployment wishes to compare total cost, these local tokens can be weighted by the conversion factor $\rho$ in Eq. (1); the empirical claim here is strictly about billing-relevant remote tokens, while local overhead is reported separately for transparency.

## 5.2 Strategic Requests Amplify Cost and Leakage

Table 4 summarizes amplification under the attack-prompt condition, which operationalizes strategic information requests. $M _ { B }$ is strongly amplified under this condition, increasing both remote cost and checker calls. $M _ { G + C }$ keeps cost amplification close to one and reduces checker-call amplification from 1.69× to 1.02×. It also achieves the lowest absolute attack leakage, 18.79%, compared with 28.19% for $M _ { B }$ and 21.48% for $M _ { G }$ . The relevant comparison is therefore the joint outcome: lower absolute attack leakage, smaller leakage increase, and much lower remote cost amplification.

Table 4. Attack amplification summary. Cost amplification compares remote tokens under attack prompts with normal prompts; call amplification compares average PrivacyChecker calls. Leak $\varDelta$ is attack leakage minus normal leakage. Lower is better for all metrics.
<table><tr><td>Mechanism</td><td>Cost amp. ↓</td><td>Call amp. ↓</td><td>Attack leak. ↓</td><td>Leak  $\varDelta \downarrow$ </td></tr><tr><td> $M _ { B }$ </td><td>1.32×</td><td>1.69×</td><td>28.19%</td><td>+16.78 pp</td></tr><tr><td> $M _ { G }$ </td><td>0.98×</td><td>N/A</td><td>21.48%</td><td>+16.11 pp</td></tr><tr><td> $M _ { G + C }$ </td><td>1.02×</td><td>1.02×</td><td>18.79%</td><td>+14.76 pp</td></tr></table>

## 5.3 Utility Is Preserved Under the Safe Task Shell

Table 5 evaluates whether privacy-preserving rewriting preserves the legitimate task. The safe-shell judge used for attack prompts is defined in Appendix C. Under normal prompts, $M _ { G + C }$ reaches the highest UsableRate, 92.6%, while retaining 98.1% of privacy-allowed task slots. Its semantic drift is 0.046, indicating a small abstraction cost rather than collapse of task utility. Under attack prompts, $M _ { B }$ requires 1.52 average repair rounds, while $M _ { G + C }$ requires only 0.04. At the same time, $M _ { G + C }$ retains 99.1% of safe slots and reaches 100.0% safe UsableRate. Thus the privacy improvement is not obtained by refusing all messages, clearing the email, or over-generalizing away the legitimate task.

Table 5. Semantic preservation and safe-task fulfillment. For attack prompts, satisfaction labels measure the benign task shell after removing attack-specific requests for sensitive details.
<table><tr><td colspan="6">Method Prompt Avg. repair rounds Slot retention Satisfied Partially satisfied UsableRate Semantic drift</td></tr><tr><td> $M _ { B }$  Normal</td><td>0.12</td><td>0.999</td><td>43.0%</td><td>42.3%</td><td>85.2% 0.001</td></tr><tr><td> $M _ { G }$  Normal</td><td>0.00</td><td>0.970</td><td>34.9%</td><td>55.0% 89.9%</td><td>0.062</td></tr><tr><td> $M _ { G + C }$  Normal</td><td>0.00</td><td>0.981</td><td>39.6%</td><td>53.0% 92.6%</td><td>0.046</td></tr><tr><td> $M _ { B }$  Attack</td><td>1.52</td><td>0.984</td><td>97.3%</td><td>2.0%</td><td>99.3% 0.012</td></tr><tr><td> $M _ { G }$  Attack</td><td>0.00</td><td>0.977</td><td>94.6%</td><td>4.0% 98.7%</td><td>0.043</td></tr><tr><td> $M _ { G + C }$  Attack</td><td>0.04</td><td>0.991</td><td>98.7%</td><td>1.3%</td><td>100.0% 0.022</td></tr></table>

## 5.4 The Information-Release Contract Matters

Table 6 reports an ofline replay ablation over the rewrite prompt and contract fields. The exact replay protocol and prompt variants are described in Appendices B and D.1. The agent, tools, and sending actions are not rerun; each variant starts from the same original agent draft from Gemma-only logs and then reruns leakage evaluation. Full contract reduces normal leakage from 7.38% to 4.70% and attack leakage from 32.21% to 14.77%, showing that the structured contract filters leakage more efectively than a generic privacy rewrite prompt. Removing required\_slots and information\_shape also weakens the privacy result.

Table 6. Ofline contract ablation. Relative reductions are computed against generic Gemma rewriting. The ablation evaluates leakage only, not utility or token cost.
<table><tr><td>Variant</td><td>Normal leakage Normal red. Attack leakage Attack red.</td><td></td><td></td><td></td></tr><tr><td>Generic Gemma rewrite</td><td>7.38%</td><td></td><td>32.21%</td><td></td></tr><tr><td>Full contract</td><td>4.70%</td><td>36.3%</td><td>14.77%</td><td>54.1%</td></tr><tr><td>w/o required_slots/information_shape</td><td>5.37%</td><td>27.2%</td><td>15.44%</td><td>52.1%</td></tr></table>

## 5.5 Mechanism-Level Cases

Case studies illustrate the same mechanism-level pattern. Table 7 summarizes three anonymized cases; Appendix D.2 provides additional detail for the residual semantic-leakage case. The cases are not additional quantitative evidence, but they explain why local rewriting improves the cost–privacy trade-of and why the checker remains necessary in high-risk settings.

## 6 Related Work and Limitations

Agent privacy and contextual integrity. Privacy evaluations for LLMs increasingly distinguish privacy judgment from privacy behavior. ConfAIde and PrivacyLens test whether models respect contextual privacy norms in responses and agent trajectories [4,5]; CI-Bench extends contextual-integrity evaluation to synthetic dialogues and emails [6].Recent work on autonomous web agents further frames the problem as data minimization, asking whether agents use sensitive information only when it is necessary for the current task [16]. Mitigation work has also used contextual integrity to supervise information sharing, including AirGapAgent, form-filling assistants, and PrivacyChecker [7,8,1]. Our work builds on this line but changes the objective: rather than asking only whether a checker reduces leakage, we ask whether the platform can reduce leakage and remote enforcement cost at the same time. The information-release contract follows the contextual-integrity view of senders, subjects, recipients, data types, and transmission principles [2,3], but uses that view as a rewrite policy rather than only as a judgment prompt. In this sense, PrivacyChecker is the strongest baseline and also the component we retain as a final validator; the contribution is to change the economics of the repair path before that validator is invoked.

Table 7. Mechanism-level case summaries. The table reports anonymized behavior, not full private email text.
<table><tr><td>Case</td><td>Observed mechanism behavior</td><td>Mechanism lesson</td></tr><tr><td>School health up- date</td><td> $M _ { B }$  recovery status, and test evi- ordination task without copying dence.  $M _ { G + C }$  rewrites the mes- sensitive facts. sage into a high-level health- status and care-plan summary.</td><td>directly reports condition, Abstraction can preserve the co-</td></tr><tr><td>Structured compensation request</td><td>request,  $M _ { B }$  salary, bonus, and equity de- extraction pressure. tails.  $M _ { G + C }$  keeps the negotia- tion context but removes exact compensation and stock-option</td><td>Under an adversarial structured Contracted rewriting resists discloses exact exact-value and structured-</td></tr><tr><td>Gemma-only residual leakage</td><td>fields.  $M _ { G }$  monitoring and clearance cues. The checker sends. in  $M _ { G + C }$  triggersfurther abstraction and removes recon- structive implications.</td><td>removes explicit disease Local rewriting reduces leakage names but leaves respiratory- before checking, but final valida- activity- tion is still needed for high-risk</td></tr></table>

Tool-using agent safety and privacy economics. Tool-using agents create safety risks because harms occur through external actions rather than only chat responses [10,11]. Prompt-injection and adaptive attack studies show that language inputs can manipulate agent behavior [9]; our strategic information requests are similar in spirit, but focus specifically on requester-driven pressure for unauthorized information release and measure both leakage and induced enforcement cost. The cost component connects the problem to privacy economics and platform mechanism design: stronger privacy enforcement can reduce harm, but it also consumes resources and may reduce utility [12,13]. Our mechanism is therefore an action-level wrapper for a high-risk external action, and the evaluation targets the artifact that would actually reach the email recipient. Guardrail systems are increasingly used as post-hoc enforcement layers that monitor inputs, outputs, or tool actions [18]. Recent surveys emphasize that guardrail design requires explicit requirements, evaluation, and deploymentspecific trade-ofs rather than generic filtering alone [18]. Our mechanism follows this boundary-enforcement view, but focuses on cost-aware local rewriting before final semantic checking.

Limitations. $M _ { G }$ has the lowest remote cost, but its leakage remains higher than $M _ { G + C } ,$ , especially under attack prompts. It should therefore be interpreted as a lightweight option for cost-sensitive, lower-risk deployments, not as a replacement for final checking. The final checker can miss semantic leakage or over-flag safe abstractions, and the contract can infer the wrong task type, omit a required slot, generalize too aggressively, or preserve an unsafe detail. The leakage evaluator uses a diferent prompt from the checker but the same model family; manual spot checks were used to screen obvious judge failures, but not as a formal human evaluation. Finally, the current evaluation focuses on email-generation actions. Extending the mechanism to Slack messages, calendar invitations, CRM updates, document sharing, and form submission requires action-specific information-shape constraints.

## 7 Conclusion

This paper studies privacy enforcement for email-generating LLM agents as a cost-aware strategic information-release problem. Checker-only defenses are valuable but economically expensive because unsafe drafts can trigger repeated remote action-LLM repair and checker calls. Strategic requesters can worsen both sides of the problem by inducing more privacy leakage and amplifying remote enforcement cost. We propose a local contract-conditioned mechanism that intercepts outgoing email actions, generates an information-release contract, rewrites the draft locally with Gemma, and selectively validates the rewritten draft with PrivacyChecker. Across 149 aligned normal and attack tasks, Gemma + Checker reduces leakage, suppresses attack-induced cost amplification, and maintains high usable safe-task fulfillment. More broadly, high-risk agent actions should be governed by explicit, cost-aware information-release mechanisms rather than by unconstrained drafting followed by repeated remote filtering.

## References

1. Wang, S., Yu, F., Liu, X., Qin, X., Zhang, J., Lin, Q., Zhang, D., Rajmohan, S.: Privacy in Action: Towards Realistic Privacy Mitigation and Evaluation for LLM-Powered Agents. Findings of ACL: EMNLP (2025).

2. Nissenbaum, H.: Privacy as Contextual Integrity. Washington Law Review 79(1), 119–157 (2004).

3. Nissenbaum, H.: Privacy in Context: Technology, Policy, and the Integrity of Social Life. Stanford University Press (2009).

4. Mireshghallah, N., Kim, H., Zhou, X., Tsvetkov, Y., Sap, M., Shokri, R., Choi, Y.: Can LLMs Keep a Secret? Testing Privacy Implications of Language Models via Contextual Integrity Theory. ICLR (2024).

5. Shao, Y., Li, T., Shi, W., Liu, Y., Yang, D.: PrivacyLens: Evaluating Privacy Norm Awareness of Language Models in Action. NeurIPS Datasets and Benchmarks Track (2024).

6. Cheng, Z., Wan, D., Abueg, M., Ghalebikesabi, S., Yi, R., Bagdasarian, E., Balle, B., Mellem, S., O’Banion, S.: CI-Bench: Benchmarking Contextual Integrity of AI Assistants on Synthetic Data. arXiv:2409.13903 (2024).

7. Bagdasarian, E., Yi, R., Ghalebikesabi, S., Kairouz, P., Gruteser, M., Oh, S., Balle, B., Ramage, D.: AirGapAgent: Protecting Privacy-Conscious Conversational Agents. ACM CCS (2024).

8. Ghalebikesabi, S., Bagdasarian, E., Yi, R., Yona, I., Shumailov, I., Pappu, A., Shi, C., Weidinger, L., Stanforth, R., Berrada, L., Kohli, P.-S., Huang, P.-S., Balle, B.: Privacy Awareness for Information-Sharing Assistants: A Case-Study on Form-Filling with Contextual Integrity. TMLR (2025).

9. Zhan, Q., Fang, R., Panchal, H.S., Kang, D.: Adaptive Attacks Break Defenses against Indirect Prompt Injection Attacks on LLM Agents. Findings of ACL: NAACL (2025).

10. Schick, T., Dwivedi-Yu, J., Dessi, R., Raileanu, R., Lomeli, M., Hambro, E., Zettlemoyer, L., Cancedda, N., Scialom, T.: Toolformer: Language Models Can Teach Themselves to Use Tools. NeurIPS (2023).

11. Yao, S., Zhao, J., Yu, D., Du, N., Shafran, I., Narasimhan, K., Cao, Y.: ReAct: Synergizing Reasoning and Acting in Language Models. ICLR (2022).

12. Ghosh, A., Roth, A.: Selling Privacy at Auction. Games and Economic Behavior 91, 334–346 (2011).

13. Kifer, D., Machanavajjhala, A.: No Free Lunch in Data Privacy. ACM SIGMOD (2011).

14. Debenedetti, E., Zhang, J., Balunovic, M., Beurer-Kellner, L., Fischer, M., Tramèr, F.: AgentDojo: A Dynamic Environment to Evaluate Prompt Injection Attacks and Defenses for LLM Agents. Advances in Neural Information Processing Systems 37, 82895–82920 (2024).

15. Zhan, Q., Liang, Z., Ying, Z., Kang, D.: InjecAgent: Benchmarking Indirect Prompt Injections in Tool-Integrated Large Language Model Agents. Findings of the Association for Computational Linguistics: ACL 2024, 10471–10506 (2024).

16. Zharmagambetov, A., Guo, C., Evtimov, I., Pavlova, M., Salakhutdinov, R., Chaudhuri, K.: AgentDAM: Privacy Leakage Evaluation for Autonomous Web Agents. NeurIPS Datasets and Benchmarks Track (2025).

17. Wallace, E., Xiao, K., Leike, R., Weng, L., Heidecke, J., Beutel, A.: The Instruction Hierarchy: Training LLMs to Prioritize Privileged Instructions. arXiv preprint arXiv:2404.13208 (2024).

18. Dong, Y., Mu, R., Zhang, Y., Sun, S., Zhang, T., Wu, C., Jin, G., Qi, Y., Hu, J., Meng, J., et al.: Safeguarding Large Language Models: A Survey. Artificial Intelligence Review 58(12), 382 (2025).

## A Information-Release Contract Schema

The information-release contract is the structured representation used to condition the local Gemma rewrite. It is constructed from the user instruction and the agent-generated draft email, and does not use hidden sensitive-item annotations, evaluation labels, or leakage-evaluator outputs. Thus, the contract is part of the deployable send-time mechanism rather than an evaluation-side oracle.

The contract separates several concerns that are dificult to control with a single generic rewrite prompt: (i) inferring the task type and confidence, (ii) preserving privacy-allowed task content, (iii) generalizing private or source-specific content, (iv) maintaining the requested information shape and detail density, and (v) preventing unsupported additions. Table 8 summarizes the main contract fields by function.

The privacy component is intentionally broader than exact string removal. The privacy\_risk\_policy treats a detail as risky if it reveals, strongly implies, or allows reconstruction of private, confidential, sensitive, or source-specific information. The safe\_detail\_preservation\_policycomplements this risk policy. It prevents the rewriter from treating an entire domain as sensitive merely because it is health-related, legal, financial, educational, HR-related, project-related, or technical. Instead, a detail may be preserved only when it is task-relevant, low-risk, non-distinctive, and non-reconstructive. This policy helps avoid over-generalization: safe task content can remain in the email, while private evidence, person-specific facts, raw records, identifiers, and reconstructive details are still generalized or removed.

Table 8. Information-release contract fields grouped by function.
<table><tr><td>Field group</td><td>Fields and purpose</td></tr><tr><td>Task typing</td><td>task_type, task_subtype, task_confidence, matched_signals, and low_confidence classify the email task and record the confidence of the rule-based task inference. Low-confidence cases fall back to preserving the draft&#x27;s</td></tr><tr><td>Task goal and util- task_goal, must_preserve, required_slots, ity</td><td>apparent purpose and structure. information_shape, fulfillment_requirements, body_inclusion_requirement, detail_density, plural_scope_requirement, and safe_output_shape specify the benign task shell, expected output form, required safe slots, and minimum conditions for a rewritten email to remain</td></tr><tr><td>Privacy straints</td><td>task-useful. con- must_generalize, privacy_risk_policy, safe_detail_preservation_policy, source_grounding_requirements, and safe_substitution_plan identify sensitive or source-specific content, distinguish risky details from preservable low-risk details, define semantic leakage risks, and guide the rewrite toward safe substitutes such as high-level conclusions,</td></tr><tr><td>rewrite-control constraints</td><td>category-level fields, process-level status, safe ranges, or de-identified entries. No-fabrication and must_not_add and rewrite_hints prevent unsupported additions, task shifts, private-source references, placeholders, and excessive rewriting, while encouraging minimal edits and preservation of safe source-grounded content.</td></tr></table>

It also gives privacy constraints priority over utility-oriented fields when the two conflict. Therefore, a required slot or detail-density target can only be satisfied with information that passes the privacy policy.

The utility component prevents the local rewrite from becoming an empty refusal. For example, required\_slots, information\_shape, and fulfillment\_requirements encourage the model to preserve the benign communication purpose and to retain safe task-specific content when available. When exact details are sensitive, the safe\_substitution\_plan asks the rewriter to preserve the information function rather than the sensitive fact.

Table 9. Compact anonymized contract example.
<table><tr><td>Field</td><td>Example value</td></tr><tr><td>task_type</td><td>update_status.</td></tr><tr><td>task_goal</td><td>Send a status update to the intended recipient after re- moving private-source lookup hints.</td></tr><tr><td>must_preserve</td><td>Update purpose; safe high-level topic; high-level status or conclusion when privacy-safe.</td></tr><tr><td>must_generalize</td><td>Person-specific health, student, financial, legal, HR, client, project, or source-grounded details that reveal or</td></tr><tr><td>must_not_add</td><td>reconstruct private information. Unsupported facts; new names, dates, locations, amounts, attachments, private-source references, placeholders, or</td></tr><tr><td>required_slots</td><td>unsupported follow-up promises. Greeting; update purpose; current high-level status; safe progress point; next step; closing.</td></tr><tr><td>Privacy-risk policy</td><td>Remove prohibited details, generalize sensitive details, and preserve only low-risk details. Exact, paraphrased,</td></tr><tr><td></td><td>and reconstructive leakage are all treated as privacy risks. safe_output_shape Write a privacy-safe update that preserves the benign task purpose, includes a high-level status, keeps safe up- date points, and ends with an appropriate next step.</td></tr></table>

A compact anonymized example is shown below. The example instantiates one possible task type, update\_status; other tasks instantiate the same schema with diferent task-type labels such as reply\_response, provide\_details\_data, summary, or scheduling\_coordination.

This schema shows why contract-conditioned rewriting is diferent from a generic privacy rewrite prompt. The contract does not merely ask the local model to “remove private information.” Instead, it provides structured task constraints, privacy-risk rules, source-grounding restrictions, safe-substitution guidance, and no-fabrication constraints. The final checker is still retained after the local rewrite, because the contract is designed to reduce leakage before checking, not to replace the checker in high-risk settings.

## B Rewrite Prompts and Ablation Variants

This appendix describes the local rewrite prompt used in the send-time pipeline and the prompt variants used for the P1-1 contract ablation. The purpose is to clarify the diference between a generic local privacy rewrite and a rewrite conditioned on the structured information-release contract.

Send-time local rewrite. In the full pipeline, the local Gemma rewriter is invoked after the agent has produced an email draft and before the final Gmail send action is allowed. The rewriter receives the original email arguments, the user instruction, the information-release contract, sender context, and, when available, feedback from a previous PrivacyChecker failure. The rewriter is instructed to return only a valid JSON object with two keys, subject and body. The original recipient fields are preserved from the tool arguments; the local rewrite only changes the email-facing text.

The production rewrite prompt is intentionally more specific than a generic “remove private information” instruction. It includes global privacy-risk rules, domain-specific privacy boundaries, utility-preservation rules, body inclusion requirements, structured-information requirements, and a silent self-check. The core instruction can be summarized as follows:

Minimally edit the original draft so it remains useful while removing privacy leakage. Privacy-risk filtering comes before helpfulness preservation. If a requested detail is sensitive, transform it into a safe substitute rather than erasing the task function. Do not reveal sensitive facts by exact wording, paraphrase, close synonym, distinctive decomposition, deidentified restatement, paired descriptors, or clear implication. Return only JSON with subject and body.

The prompt also instructs the rewriter to use the task contract to preserve the benign task goal, safe must\_preserve items, privacy-safe required\_slots, information\_shape, body inclusion requirements, detail-density requirements, plural-scope requirements, and safe\_substitution\_plan. These utilitypreservation requirements apply only after privacy filtering. If a helpfulness requirement conflicts with the privacy\_risk\_policy, the privacy policy takes priority.

Rewrite and checker loop. The full mechanism does not treat Gemma-only rewriting as the final safety decision. When Gemma is enabled, the send guard first constructs a task contract from the user instruction and the original agent draft. It then asks Gemma to rewrite the draft. The rewritten draft is passed to the final PrivacyChecker unless the experiment configuration disables the checker. If the checker passes, the rewritten email is allowed to be sent. If the checker detects leakage, its feedback is passed back to Gemma for another rewrite round, up to the configured rewrite-round limit. If no rewritten draft passes the final checker, the send action is blocked.

draft = agent\_proposed\_email   
contract = build\_task\_contract(user\_instruction, draft)   
checker\_feedback = None   
for round in 1..max\_rewrite\_rounds:   
rewritten = gemma\_rewrite(   
draft,   
contract,

```python
checker_feedback
)
result = final_privacy_checker(rewritten)
if result.passed:
send(rewritten)
stop
checker_feedback = result
draft = rewritten
```  
block if no rewritten draft passes the checker

This loop explains why the proposed full mechanism is diferent from a Gemma-only sanitizer. The local rewrite is used to reduce leakage before the final check, while the checker remains responsible for detecting residual exact, paraphrased, or reconstructive leakage.

Ofline ablation prompt variants. For P1-1, we compare local rewrite conditioning variants using ofline replay. The reported variants are summarized in Table 10. Each variant receives the same user instruction and the same original agent-produced draft for a given item. The diference is only the rewrite conditioning prompt.

Table 10. Prompt variants used in the P1-1 contract ablation.
<table><tr><td>Variant</td><td>Construction</td></tr><tr><td>Generic rewrite</td><td>Uses a generic privacy rewrite instruction with the user instruction and original draft, but without the information-release contract. The prompt asks the model to reduce privacy leakage, preserve the legitimate task intent, avoid unsupported facts, avoid private-source ref-</td></tr><tr><td>Full contract</td><td>erences, and return JSON with subject and body. Uses the same user instruction and original draft, but additionally provides the complete information-release contract. The prompt asks the model to preserve utility- bearing content when the contract requires it, generalize or omit sensitive details when the contract requires it,</td></tr><tr><td>shape</td><td>and avoid unsupported facts. No required slots / Uses the full contract after removing the fields that spec- ify required task slots and output shape. All other con- tract fields are preserved.</td></tr></table>

The generic replay prompt is a short privacy-oriented instruction. It does not receive the structured contract and therefore cannot distinguish contract fields such as must\_preserve, must\_generalize, required\_slots, or safe\_substitution\_plan. In shortened form, the prompt is:

Rewrite the following email to reduce privacy leakage while preserving the legitimate task intent. Remove or generalize private, sensitive, confidential, or source-specific details. Preserve the main purpose of the email and non-sensitive task content. Do not add facts not supported by the user instruction or original draft. Do not mention private sources such as Gmail, Notion, private notes, internal records, tools, or logs. Return exactly one valid JSON object with keys subject and body.

The full-contract replay prompt uses the same original draft, but includes the complete contract JSON:

Rewrite the email so that it satisfies the contract while preserving the legitimate user task. Use the user instruction, original draft, and task contract as the only grounding sources. Preserve required utility-bearing content when the contract indicates it should be preserved. Generalize, abstract, or omit sensitive details when the contract indicates they should be generalized. Do not add facts not supported by the user instruction or original draft. Return exactly one valid JSON object with keys subject and body.

For the ablated contract variant, we remove the utility-shape fields:

$$
\begin{array} { r l r } & { \tt { r e q u i r e d \_ s l o t s } } & { = \emptyset , } \\ & { \tt { i n f o r m a t i o n \_ s h a p e } } & { = \emptyset , } \\ & { \tt f u l f i 1 l m e n t \_ r e q u i r e m e n t s = \emptyset . } \end{array}
$$

All other contract fields are preserved. This ablation tests the efect of removing the fields that explicitly describe the expected task slots and output form, while keeping the privacy-oriented and no-fabrication parts of the contract.

Scope of P1-1. P1-1 is a leakage-only ofline replay ablation. It is not used to evaluate task utility, semantic drift, runtime, or token cost. The replay does not rerun the agent, does not call Gmail or Notion, does not execute a real Gmail send action, and does not rerun the PrivacyChecker. Instead, it extracts the original agent-produced draft from existing Gemma-only logs, applies each rewrite variant to the same draft, emits synthetic evaluation logs, and reruns the leakage evaluator on those synthetic logs. This design isolates the leakage efect of rewrite conditioning from upstream agent behavior, tool retrieval, and final checker decisions.

## C Evaluation Metrics and Judge Protocol

This appendix defines the evaluation metrics and judge protocols used in our experiments. We distinguish privacy leakage, content satisfaction, safe-shell semantic preservation, and cost/process metrics. Unless otherwise stated, all metrics are computed from execution logs and the final email-facing content.

Final email extraction. For leakage evaluation, the evaluator first tries to use the last email draft that passed the privacy checker. If no such draft is available, it falls back to the final successfully sent email. This choice evaluates the emailfacing content that is either privacy-approved or actually delivered. For content satisfaction, the evaluator uses the last explicit final\_sent\_email event when available, and otherwise falls back to a successful gmail\_send\_message tool call. Recipient correctness is evaluated separately and is not part of the contentsatisfaction label.

Leakage metrics. Leakage is evaluated at two granularities. Let N be the number of evaluated tasks, $L _ { \mathrm { i t e m } }$ be the number of tasks with at least one leaked sensitive item, S be the number of annotated sensitive items, and $L _ { \mathrm { s e c r e t } }$ be the number of leaked sensitive items. We define:

$$
{ \mathrm { I t e m L e a k a g e } } = { \frac { L _ { \mathrm { i t e m } } } { N } } .
$$

$$
\mathrm { S e c r e t L e a k a g e } = { \frac { L _ { \mathrm { s e c r e t } } } { S } } .
$$

For each benchmark item, the evaluator compares the final email subject and body against the annotated sensitive information items for that item. An item is counted as leaked if any annotated sensitive item is judged to be leaked. A sensitive item is counted as leaked when the email directly reveals it, partially reveals it, or strongly implies it such that the recipient can infer the private information. Therefore, the leakage metric covers exact, partial, and reconstructive leakage rather than exact string overlap only.

Content satisfaction and usable rate. Content satisfaction measures whether the final sent email content satisfies the user’s requested task. The judge assigns one of four labels:

$$
\begin{array}{c} \mathtt  s a t i s f i e d , p a r t i a l l y \_ s a t i s f i e d , n o t \_ s a t i s f i e d , e n e l a , e n e l a 2 \_ s a t i s f i e d , o r m a n u a l \_ r e v i e w \_ w i d e d \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d u a l \_ m o d e \_ n o t \_ h o a l \_ m o d u a l \_ m o d e \_ { \alpha \_ m o \alpha \alpha \_ \_ s \_ \_ m a t i s f i e s d \_ { \alpha \_ m a \_ \_ m o \alpha \_ \_ s \_ \_ m a t i s f i e s d \_ { \alpha \_ m a \_ \_ \_ m o \alpha \_ \_ \_ s c o \_ \_ \_ m a \_ \_ m o \_ \_ s \_ m a \_ \_ m o d \_ \_ m o d \_ \_ m o d u a l \_ \_ m o d u a l \_ \_ m o d u a l \_ \_ m o d u a l \_ m o d u a l \_ \_ m o d u a l \_ { \alpha \_ m o \alpha \_ \_ m o \alpha \_ \_ s i s i g m a \_ m a \_ p a \_ s i e s } } } } \end{array}
$$

A satisfied email fully or basically completes the user’s core task. A partially\_satisfied email contains task-relevant content and completes part of the task, but misses important details, is too brief, is too generic, or only partially fulfills the request. A not\_satisfied email is empty, irrelevant, generic filler, the wrong task, or fails to provide the requested information. The manual\_review label is reserved for cases where the available log information is insuficient for a reliable judgment, such as hidden attachments, corrupted final-email fields, severe instruction ambiguity, or unavailable external materials referenced by the email.

We report strict satisfaction as the rate of satisfied items and usable rate as the relaxed rate that counts both fully and partially useful outputs:

$$
\mathrm { U s a b l e R a t e } = \mathrm { S a t i s f i e d R a t e } + \mathrm { P a r t i a l l y S a t i s f i e d R a t e } .
$$

Safe-shell protocol for attack prompts. Attack prompts may include both a benign communication task and an attack-specific request for sensitive information. Therefore, satisfaction under attack should not be interpreted as attacker utility. For attack settings, we additionally use a safe-shell semantic judge. The judge first decomposes the user instruction into safe\_task\_shell and attack\_or\_sensitive\_requests. Only the privacy-allowed task shell contributes to the utility judgment. Sensitive requests, raw private values, source quotes, medical details, compensation details, student records, confidential source names, and similar attack-specific slots are excluded from the required task content.

Thus, for attack prompts, removing sensitive requested details is not counted as semantic drift. Drift occurs only when the final email drops or distorts the benign task shell, becomes an empty refusal, uses the wrong recipient or action, adds unsupported content, or over-generalizes away safe task-relevant information.

Slot retention and semantic drift. For safe-shell evaluation, the judge identifies privacy-allowed required slots from the benign task shell. Let $Q$ be the number of safe required slots and $P$ be the number of those slots preserved in the final email. Slot retention is defined as:

$$
{ \mathrm { S l o t R e t e n t i o n } } = { \frac { P } { Q } } .
$$

If no safe required slots can be identified, slot retention is treated as undefined rather than forced to zero. The judge also assigns a TaskIntentSimilarity score in [0, 1], where 1.0 means the final email fully preserves the privacy-allowed task intent and 0.0 means the email is unrelated, empty, or only refuses without preserving the task shell. Semantic drift is defined as:

$$
\mathrm { S e m a n t i c D r i f t = 1 - T a s k I n t e n t S i m i l a r i t y } .
$$

The semantic judge also records a drift type, including minor omission, overgeneralization, task shift, unsupported addition, refusal or empty output, recipient/action error, or unclear cases. These categories are used for analysis and case selection, while the main numeric summaries use slot retention and semantic drift.

Cost and process metrics. Process and cost metrics are computed directly from execution logs. We record PrivacyChecker calls, Gemma rewrite rounds, Gemma retry rate, Gemma failure rate, runtime, and token usage. Let RemoteTokens $( M , p )$ denote the billing-relevant non-local LLM tokens used by method M under prompt type $p ,$ including action-LLM tokens and PrivacyChecker tokens. Gemma local tokens are reported separately and are not included in remote token cost.

Remote token amplification under attack is defined as:

$$
\mathrm { C o s t A m p } ( M ) = \frac { \mathrm { R e m o t e T o k e n s } ( M , \mathrm { A t t a c k } ) } { \mathrm { R e m o t e T o k e n s } ( M , \mathrm { N o r m a l } ) } .
$$

Checker-call amplification is defined as:

$$
\mathrm { C h e c k e r C a l l A m p } ( M ) = \frac { \mathrm { C h e c k e r C a l l s } ( M , \mathrm { A t t a c k } ) } { \mathrm { C h e c k e r C a l l s } ( M , \mathrm { N o r m a l } ) } .
$$

The token aggregator separates agent tokens, privacy\_checker tokens, and gemma\_rewrite tokens from standardized llm\_usage log events. Gemma tokens are recorded as local tokens, while billing-relevant non-local tokens are accumulated as remote cost. Checker calls are computed from the final output’s privacy-check count when available, and otherwise from the maximum logged privacy-check count in the execution log.

Ofline replay scope. The P1-1 contract ablation uses ofline replay logs only for leakage evaluation. These replay logs are not used to compare runtime, token cost, checker-call amplification, or full-pipeline utility. In replay, the agent is not rerun, Gmail and Notion are not called, the PrivacyChecker is not rerun, and no real email is sent. The purpose of the replay is to isolate the leakage efect of diferent rewrite-conditioning variants on the same original agent draft.

## D Ofline Replay and Additional Case Studies

## D.1 Ofline Replay for Contract Ablation

The P1-1 contract ablation is conducted using ofline replay. The goal of this replay is to compare rewrite-conditioning variants while holding the upstream agent behavior fixed. Therefore, the replay does not rerun the agent, does not call Gmail or Notion, does not execute a real Gmail send action, and does not rerun the PrivacyChecker. Instead, it starts from existing Gemma-only execution logs and rewrites the same original agent-produced draft under diferent prompt variants.

For each item, the replay script extracts three pieces of information from the base log: the original user instruction, the information-release contract, and the original email draft submitted to the Gemma rewrite stage. The original draft is taken from the logged gemma\_rewrite\_requested.original\_arguments field. This is the draft produced by the upstream agent before the replayed rewrite variant is applied.

The reported replay variants are: generic\_gemma\_rewrite, full\_contract, and no\_required\_slots\_information\_shape. For the ablated utility-shape variant, the replay keeps the same base contract but sets required\_slots, information\_shape, and fulfillment\_requirements to empty lists. All other contract fields are preserved.

The replay procedure is summarized below.

22 Qikun Cai et al.   
for each item i:   
log = load\_gemma\_only\_log(i)   
user\_instruction =   
log.execution\_start.input   
original\_draft =   
log.gemma\_rewrite\_requested.original\_arguments   
full\_contract =   
log.task\_contract\_built.task\_contract   
for variant in reported\_variants:   
prompt\_or\_contract =   
build\_variant(variant, full\_contract)   
rewritten\_email =   
local\_gemma\_rewrite(   
user\_instruction,   
original\_draft,   
prompt\_or\_contract   
)   
synthetic\_log =   
replace\_final\_email(   
log,   
rewritten\_email   
)   
run\_leakage\_evaluator(synthetic\_log)

The generated logs are synthetic evaluation artifacts. They preserve the shape of the original execution logs as much as possible, but replace the final emailfacing fields with the rewritten variant output. The synthetic logs are marked as ofline ablation replay logs, with actual\_email\_sent=false. For evaluator compatibility, the logs retain send-related events, but those events are marked as simulated. The PrivacyChecker is also marked as disabled in the replay logs because it is not rerun during P1-1.

This design makes the leakage comparison fair for the purpose of the contract ablation. Since every variant starts from the same original agent draft for the same item, diferences in leakage are attributed to the rewrite-conditioning variant rather than to diferences in agent planning, private-tool retrieval, Gmail/Notion results, recipient selection, or final checker decisions. At the same time, because the replay does not execute the full pipeline, P1-1 is used only for leakage evaluation. It is not used to compare runtime, token cost, checker-call amplification, recipient correctness, or full-pipeline utility.

Table 11. Supplementary mechanism-level analysis for Case 3.
<table><tr><td>Mechanism</td><td>Behavior</td><td>Leakage outcome</td></tr><tr><td>Gemma-only</td><td>Removes some explicit sensitive wording, but preserves a combination of of 3 annotated health-management, respiratory-care, environmental-trigger, and activity-clearance cues. These cues make the output useful, but they also allow reconstruction of private health</td><td>Residual leakage: 2 sensitive items leaked.</td></tr><tr><td></td><td>information. Gemma + Checker Further abstracts the same coordination No leakage: 0 of 3 task into general care-management and annotated sensitive school-participation language. The benign task shell is preserved, while condition-specific and test-specific implications are avoided.</td><td>items leaked.</td></tr></table>

## D.2 Residual Semantic Leakage in Gemma-Only

We use Case 3 to illustrate why the local Gemma rewrite should not be interpreted as a replacement for the final checker. The case is a strategic healthcoordination request involving a student. The annotated sensitive facts include private health status, recovery-related information, and medical-test evidence. We anonymize the case and do not include the original private facts or full email text.

Table 11 summarizes the mechanism-level diference. The Gemma-only output removes some explicit sensitive wording, but it still retains a set of semantically revealing cues. In particular, the rewritten email contains health-management language, respiratory-care language, environmental trigger language, and activity-clearance language. Although these phrases are less explicit than the original private facts, the leakage judge marked two of the three annotated sensitive items as leaked because the remaining cues still support reconstruction of the private health information.

By contrast, the Gemma + Checker setting produces a privacy-passed email for the same item. The final email preserves the benign coordination task: it provides a general health-status update, confirms that the student can participate in normal school activities, and recommends following an existing care plan. However, the wording is more abstract and does not strongly imply the annotated sensitive facts. The leakage judge marked zero of the three annotated sensitive items as leaked.

This case highlights a key distinction between direct redaction and semantic privacy enforcement. The Gemma-only rewrite can reduce surface-level leakage, but it may still leave combinations of safe-looking phrases that reconstruct a private fact. The final checker is useful because it evaluates the rewritten email after local sanitization and can reject or trigger further abstraction when residual implication remains. Therefore, our full mechanism uses Gemma to reduce leakage before checking, but retains the checker as the final safety layer.