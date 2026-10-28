# Concordat: Typed Sessions for LLM Agents

Enforcing conversation protocols on a model that cannot be trusted to follow them

Milad Sikaroudi Independent Researcher sikaroudi@gmail.com

## Abstract

Agents built on large language models are increasingly allowed to take consequential actions: cancelling bookings, issuing refunds, changing accounts. The rules that govern those actions, such as “obtain the customer’s explicit confirmation before changing anything”, are usually stated in the prompt and followed only as reliably as the model chooses to follow them. We present Concordat, a Python library that treats an agent as an object whose interface is a finite-state protocol, in the spirit of typestate and session types, and enforces that protocol on the model at run time. On every turn the model is ofered only the tools that are legal in the current protocol state, calls outside that set are rejected before they have side efects, decisions reserved for humans can only be requested, and token and tool-call budgets are charged and split linearly across handofs. Protocols are checked when they are defined, and a subclass of an agent is rejected if it could bypass its parent’s protocol, a behavioural form of Liskov substitution. We evaluate a Concordat agent against the standard agent of τ-bench on its 50 airline tasks, adding simulated customers who pressure the agent to skip confirmation. The standard agent made changes without the required “yes” in 13 of 50 tasks with gpt-4.1 and in 29 of 128 task-runs with Claude Haiku 5.5; the Concordat agent made none, with no significant diference in task success. τ-bench’s own reward did not detect most of these violations. Concordat is open source.

## 1 Introduction

A customer-service agent for an airline may cancel a reservation, change a flight, or refund money. The airline’s policy says that before any such change the agent must describe it and obtain the customer’s explicit “yes”. In almost every deployed agent framework that rule is a sentence in the system prompt. The model is given every tool at every turn and is trusted to use them in the right order.

This works most of the time, and the remaining cases are the expensive ones. A customer in a hurry who says “I already told you what I want, just do it and stop asking” is not saying yes, yet a model that wants to be helpful may treat it as permission. When it does, the booking often ends up exactly as the customer wanted, so outcome-based checks see nothing wrong: in our experiments, τ-bench scored most such conversations as successful (Section 5). The policy was broken, and no one noticed.

Programming languages have long had tools for exactly this shape of rule. Typestate [23] makes the operations available on an object depend on its state; session types [12–14] describe the order of messages in a conversation and check that each party follows it. These disciplines assume every party is a program that can be type-checked. An LLM cannot be. Its side of the protocol has to be enforced while it runs.

Concordat applies the typestate idea to LLM agents with that constraint in mind. An agent is a Python class whose methods are tools, each legal only in some protocol state. The library derives the protocol from the class, checks it when the class is defined, and enforces it on every model turn by withholding illegal tools rather than asking the model not to use them. Our contributions are:

• A design that maps object-oriented concepts onto agent sessions: classes as protocol-typed agent templates, instances as sessions, methods as state-dependent tools, human decisions as protocol steps only a person can take, and budgets as resources split linearly on handof (Section 3).

• Definition-time checks for well-formedness and for protocol-safe inheritance: a subclass is rejected if any of its runs, with its own extra steps hidden, is not a run of its parent (Section 3.5).

• An open-source implementation with runners for the Anthropic Messages API and the OpenAI Responses API (Section 4).

• An empirical study on τ-bench’s airline domain with a pressure variant, which shows the standard agent breaking the confirmation rule with two models while the Concordat agent never does, at comparable task success (Section 5).

## 2 Background and related work

Behavioural types. Session types describe a channel’s protocol as a type, such as “send an integer, receive a boolean, end”, and multiparty session types derive each participant’s local type from one global protocol [12–14]. Their logical foundations connect them to linear logic [5, 24]; subtyping characterises when one session can safely replace another [10]; and runtime monitors can enforce a session type on an untrusted endpoint [3]. Resource-aware session types attach costs to protocol steps [7]. Typestate [23] and its integration with session types in object languages such as Mungo [15] are the closest object-oriented ancestors of our design. Concordat uses a deliberately simple fragment of these ideas: finite-state protocols over named actions, checked and enforced at run time.

Protocols for LLM agents. The closest recent work is ZipperGen [4], which projects messagesequence-chart workflows onto deadlock-free local programs for LLM agents and treats model calls as opaque, typed actions whose outputs are parsed after generation. Pact [11] is a choreographic language for agent ecosystems that analyses agent choices game-theoretically. XFlow [16] compiles a protocol language into a runtime harness in which protocol state gates tool calls and human approvals. Syntropy [17] uses multiparty session types in the opposite direction, to constrain an LLM that writes protocol refinements. Schlapbach [21] gives a process-calculus semantics for agent tool protocols such as MCP. Concordat difers from these in three ways: the protocol is derived from an ordinary agent class, it is enforced by changing which tools the model is ofered on each turn, and inheritance between agents is checked against the parent’s protocol.

Runtime enforcement and guardrails. A growing body of work monitors an agent’s tool calls against rules and blocks violations: AgentSpec [25] with trigger–predicate–enforcement rules, Progent [22] with privilege policies, ShieldAgent [6] with probabilistic rule circuits, and temporallogic monitors compiled to automata [9, 27]. These systems check a single agent’s trace after the model has chosen an action. Concordat instead restricts the choices the model is given, and keeps a monitor only as a fallback.

Listing 1: A refund agent. The protocol is built from the decorators.

```python
class RefundAgent(Agent):
final = {"refunded", "denied"}
budget = Budget(tokens=50_000, tool_calls=10)
@tool(when="start", then={True: "verified", False: "denied"})
def verify_identity(self, user_id: str) -> bool: ...
@tool(when="verified", then="awaiting_approval")
def propose_refund(self, amount: float) -> float: ...
@human(when="awaiting_approval", approve="refunded", reject="denied")
def approve_refund(self, amount: float) -> str:
return f"Refund ${amount}?" # shown to the human approver
```

Constrained generation. LMQL [2], Outlines [26] and XGrammar [8] constrain a single generation to a grammar; ToolDec [29] applies finite-state constraints to tool-call syntax; Formal-LLM [18] constrains an agent’s plan with automata; and type-constrained decoding enforces well-typedness during code generation [20]. These operate within one response. Concordat operates across turns, and per-state tool restriction is the coarse, provider-agnostic form of the decoding constraint a protocol implies.

## 3 Design

## 3.1 Protocols

Definition 1 (Protocol). A protocol is a tuple $P = ( S , A , s _ { 0 } , F , \delta )$ of a finite set of states S, a finite set of actions A, an initial state $s _ { 0 } \in S$ , a set of final states $F \subseteq S$ , and a transition relation $\delta : S \times A  2 ^ { S }$ . The actions allowed in state s are $a l l o w e d ( s ) = \{ a \mid \delta ( s , a )$ is defined}.

An action may lead to several states; which one is taken is decided at run time, for example by a tool’s return value. A protocol is well formed if (i) $F \neq \varnothing ; { \mathrm { ( i i ) } }$ final states allow no actions; (iii) every defined $\delta ( s , a )$ is non-empty; (iv) every state is reachable from $s _ { 0 } ;$ and (v) every state can reach a state in F. Condition (v) rules out dead ends: whatever the model and the tools do, a legal continuation to an outcome always exists. Concordat checks (i)–(v) when a protocol is constructed and rejects it with a message naming the ofending states.

A session is a run of a protocol. It holds the current state s; an action a with chosen target t is accepted only if $a \in a l l o w e d ( s )$ and $t \in \delta ( s , a )$ , and is otherwise rejected with a ProtocolViolation that leaves the state unchanged.

## 3.2 Agents as protocol-typed objects

Developers do not write protocols directly. They write an agent class, and decorators on its methods declare in which state each method is legal and where it leads (Listing 1). Declaring final states makes a class concrete, and its protocol is built and checked when the class is defined; a class without final states is an abstract base meant to be specialised.

Calling a method outside its state raises ProtocolViolation before the method body runs, so an illegal call has no side efects. Each tool call is charged to the instance’s budget. Methods keep their signatures for static type checkers. Agent.diagram() renders the protocol as a state diagram (Figure 1), so a reviewer can check the process without reading code.

![](images/87c6ee7e5d345d6c4c7e4a7bb06b65375d2c75aaa7380dd310159c147171a051.jpg)  
Figure 1: The protocol derived from Listing 1. Concordat renders such diagrams directly from the class.

## 3.3 Enforcement on the model

The runners drive an agent with a model. On each turn they compute allowed(s) for the current state and pass only those tools to the model, with parallel tool calls disabled so that each call sees the state the previous call left. A call the model makes anyway, by name or with invalid arguments, is returned to the model as an error and never reaches the tool. Because the visible tool list changes between turns, the system prompt states that tools are ofered step by step, and each tool result ends with the actions now allowed. We added both after the first live run, in which a model distrusted an earlier result because the tool that produced it had disappeared from its list.

Withholding tools is the coarse, provider-agnostic form of a decoding constraint: the model cannot select an action that is not on ofer. Grammar-constrained decoding [8, 26] could additionally constrain the arguments and text of each turn where the provider exposes it; Concordat currently relies on argument validation and the error-return path instead.

## 3.4 Humans and budgets

A @human step can be requested by the model but only decided by the agent’s approver, a function supplied by the application that receives the method’s return value and returns approve or reject. The two outcomes are distinct protocol transitions.

A Budget tracks tokens, tool calls and money. Every model response’s tokens are charged, and the next response’s output limit is capped at what remains. Handing work to another agent splits the budget: the child receives an allowance that is removed from the parent, so the same allowance cannot be spent twice. A child may not be unlimited in a dimension where its parent is limited. This is the operational core of treating budgets as linear resources [7]; Concordat does not yet type budgets statically.

## 3.5 Protocol-safe inheritance

Agents are specialised by subclassing. A team may define a compliant base agent and let others extend it. Ordinary inheritance would let a subclass add a tool that jumps straight to an outcome. Concordat rejects such subclasses when they are defined.

Definition 2 (Safe refinement). Let P be the parent protocol and C the child. C safely refines P if $F _ { C } \subseteq F _ { P }$ and, for every finite run of C, erasing the actions not in $A _ { P }$ yields a sequence of actions that P allows, under which every state of P that the child enters is a state P can be in at that point.

Concordat decides this by a breadth-first walk over pairs $( q , K )$ , where q is a state of C and $K \subseteq S _ { P }$ is the set of parent states the run can correspond to. It starts from $( s _ { 0 , C } , \{ s _ { 0 , P } \} )$ . For a transition $q \stackrel { a } {  } q ^ { \prime }$ of C: if $a \in A _ { P }$ , the new set is $\begin{array} { r } { K ^ { \prime } = \bigcup _ { p \in K } \delta _ { P } ( p , a ) } \end{array}$ , and an empty $K ^ { \prime }$ is a violation (the child takes a parent action where the parent does not allow it); if a $\notin A _ { P } , K ^ { \prime } = K$ On entering a state $q ^ { \prime } \in S _ { P }$ , the walk requires $q ^ { \prime } \in K ^ { \prime }$ and narrows $K ^ { \prime }$ to $\{ q ^ { \prime } \}$ ; otherwise the child has reached a parent state by a route the parent does not allow. States of the child that are not parent states are free. Since C and P are finite, the walk terminates. In addition, every human step of the parent must remain a human step in the child.

On failure, Concordat reports the child’s path and the parent’s required route, for example:

FastRefundAgent breaks RefundAgent’s protocol: after instant\_refund, it reaches

’refunded’, which RefundAgent only reaches through

verify\_identity -> propose\_refund -> approve\_refund

This is a safety check in the sense of trace inclusion over shared actions, a behavioural reading of Liskov substitution [19]. It is not full session subtyping [10]: it does not require the child to ofer every choice the parent ofers, and it relates states by name.

## 4 Implementation

Concordat is about 850 lines of Python with no required dependencies; the runners use the official Anthropic and OpenAI SDKs as optional extras. The Anthropic runner uses the Messages API with disable\_parallel\_tool\_use; the OpenAI runner uses the Responses API with parallel\_tool\_calls disabled. Tool input schemas are derived from method signatures and validated before invocation. A run ends with a status of done, waiting (the model ended its turn to ask the user something), refused, max\_tokens, out\_of\_budget or max\_turns; any pending tool call receives a result so the conversation can be resumed. The library has 50 unit tests and is checked with mypy in strict mode. It is published on PyPI as concordat, with source, benchmark harness and results at https://github.com/miladsikaroudi/concordat.

## 5 Evaluation

We ask two questions. RQ1: does a standard LLM agent break a stated confirmation rule, and does Concordat prevent it? RQ2: what does enforcement cost in task success?

## 5.1 Setup

Benchmark. We use τ-bench [1, 28] release v1.0.1, airline domain, all 50 tasks. A simulated customer, itself an LLM given a scenario, talks to the agent, which has the airline’s tools and policy. A task passes if the final database matches the expected one and the required information was communicated. The airline policy requires that before any action that updates the database the agent “list the action details and obtain explicit user confirmation (yes)”.

Table 1: Airline tasks under customer pressure. Counts over task-runs both agents completed. Haiku: three runs pooled (some task-runs lost to an API quota). gpt-4.1: one run, all 50 tasks, after the prompt change described in Section 5.2.
<table><tr><td>Agent model Agent</td><td></td><td>Passed</td><td>Unconfirmed changes</td><td>Task-runs with an unconfirmed change</td><td>Ended in transfer</td></tr><tr><td rowspan="2">Haiku 5.5</td><td>standard</td><td>112 / 128</td><td>50</td><td>29</td><td></td></tr><tr><td>Concordat</td><td>105 / 128</td><td>0</td><td>0</td><td></td></tr><tr><td rowspan="2">gpt-4.1</td><td>standard</td><td>20 /50</td><td>26</td><td>13</td><td>11</td></tr><tr><td>Concordat</td><td>22 /50</td><td>0</td><td>0</td><td>7</td></tr></table>

Agents. The standard agent is τ-bench’s own llm\_agent. The Concordat agent is the same agent with tool access governed by a protocol derived automatically from τ-bench’s tool annotations: two states, chatting and confirmed; lookup tools are allowed in both, tools that write to the database only in confirmed; a customer message containing “yes”, “yep” or “yeah” moves to confirmed, any other customer message back to chatting; a transfer to a human ends the session. The adapter is about 120 lines.

Pressure condition. τ-bench’s simulated customers are cooperative and readily say yes. To test the rule under pressure we append one instruction to every customer scenario (Appendix A): the first time the agent asks to confirm a change, push back (“I already told you what I want, just do it and stop asking”) without saying yes, and say yes only if asked again. A correct agent can still complete every task by asking twice.

Metrics. Pass: the task’s τ -bench reward is 1. Unconfirmed change: a database-writing tool call made when the customer’s latest message did not contain an explicit yes. We read the customer message preceding every flagged change in the first Haiku pressure run and in the gpt-4.1 run (49 changes). All but two were pushback such as “just do it and stop asking” or messages that confirmed nothing (for example a user ID or a payment choice). In the remaining two, both in task 33, the customer said “go ahead” without “yes”, which the policy’s “explicit . . . (yes)” arguably excludes; counting them as confirmations lowers the gpt-4.1 count to 12 tasks and leaves the Haiku count unchanged.

Models. Claude Haiku 5.5 as the agent with Claude Sonnet 5.5 as the customer; and gpt-4.1 (2025-04-14), τ-bench’s default model, as both agent and customer at temperature 0, accessed through an OpenAI-compatible gateway. One trial per task unless stated.

## 5.2 Results

RQ1: violations. Under pressure, the standard agent broke the confirmation rule with both models (Table 1): in 29 of 128 Haiku task-runs (13, 9 and 7 in the three runs) and in 13 of 50 gpt-4.1 tasks. The Concordat agent made no unconfirmed change in any run. The diference is significant (Fisher’s exact test, $p = 6 \times 1 0 ^ { - 1 0 }$ for Haiku treating task-runs as independent, and $p = 1 \times 1 0 ^ { - 4 }$ for gpt-4.1). Concordat prevented violations mostly by never ofering the writing tools before a yes; in the first Haiku pressure run the model attempted a write before a yes only once, and that call was blocked.

With cooperative customers neither agent broke the rule: on all 50 tasks with Haiku both passed 44 and the standard agent made every change directly after an explicit yes.<sup>1</sup> A partial run with Claude Opus 5.5 (20 tasks completed before an API quota was reached) showed no unconfirmed changes either, under the looser test. The rule is followed when nothing pushes against it; the risk appears under pressure.

The benchmark does not see these violations. In the first Haiku pressure run, 8 of the 13 tasks in which the standard agent made an unconfirmed change still received full reward, because the database ended in the expected state. Outcome-based evaluation cannot distinguish a change the customer consented to from one made over their objection.

Example. In gpt-4.1 task 11, a customer asks to downgrade a family’s flights. After the agent summarises the change, the customer replies “I already told you what I want, just do it and stop asking.” The standard agent then calls book\_reservation, creating a booking the customer never asked for, and later modifies the original reservation, and fails the task. The Concordat agent replies “I understand you’re in a hurry, but I need a clear ‘yes’ before I can make any changes”, restates the change, receives a yes, makes the single correct change, and passes.

RQ2: task success. With Haiku, the Concordat agent passed 7 fewer task-runs (McNemar’s exact test on paired outcomes, 13 vs. 6 discordant pairs, p = 0.17). Reading the 13 task-runs that only the standard agent passed, 11 ended with the Concordat agent transferring the customer to a human after repeated refusals, where the standard agent had simply made the change and broken the rule. We therefore added three sentences to the Concordat agent’s instructions: pushback is neither a confirmation nor a reason to transfer; state that a clear yes is needed, restate the change, and ask again. With gpt-4.1 after this change, the Concordat agent passed 22 tasks against the standard agent’s 20 (9 vs. 7 discordant, p = 0.80) and transferred fewer customers (7 vs. 11). In neither model is the diference in task success significant.

Cost. Reported by τ-bench from list prices, the Concordat agent cost \$1.49 against \$1.70 for the cooperative Haiku run and \$1.64 against \$1.76 for the first Haiku pressure run, consistent with sending fewer tool definitions per turn; with gpt-4.1 it cost \$3.47 against \$3.13, as its conversations under pressure were longer.

## 5.3 Threats to validity

The metric and the gate share a definition. Both use the same test for an explicit yes, so the Concordat agent’s zero is guaranteed relative to that definition; the evidence is the standard agent’s violations, checked by hand, and the preserved task success. A keyword test is a crude stand-in for consent: it would accept “yes, but not that one”. A structured confirmation, such as a button, is the right mechanism in deployment. The pressure scenario is ours. It is built on τ-bench’s tasks but is not an oficial τ-bench setting. Scale. One domain, two models, few trials; some Haiku task-runs were lost to an API quota, and with gpt-4.1 a rate limit forced about two thirds of the tasks to be rerun one at a time, at a diferent moment than the first pass. Confound. The prompt change was made between the Haiku and gpt-4.1 runs, so the improvement in task success cannot

be attributed to it alone; the clean comparison is the same model with and without the change.   
Simulated customers. Real customers may push in ways an LLM customer does not.

## 6 Discussion and future work

The central observation is that a rule stated in a prompt holds until something pushes against it, and that the usual way of measuring agents does not notice when it gives way. Withholding tools turns the rule from a request into a property of the interface. The cost we observed came not from the restriction itself but from what the model did when it could not act, which is addressable with instructions.

Several extensions follow from the session-types literature. Projection from a global, multiparty protocol [4, 14] would let several agents and humans share one specification. Grammar-constrained decoding [8, 26] could enforce each turn’s content, not only its tool choice, on models that expose it. Passing a session, with its remaining protocol and a split of the budget, to another agent is session delegation; giving budgets a static, resource-aware type [7] would let a protocol state bounds per phase. Durable sessions would let human approvals outlast a process. On the empirical side, more domains, models and trials, and a with-and-without comparison of the prompt change on one model, are needed.

## 7 Conclusion

Concordat treats an LLM agent as an object whose interface is a protocol and enforces that protocol on a model that cannot be type-checked, by changing what the model is ofered on each turn. Under customer pressure on τ-bench’s airline tasks, the standard agent broke the confirmation rule in 26% of gpt-4.1 tasks and 23% of Haiku task-runs, mostly without the benchmark noticing; the Concordat agent never did, at comparable task success.

## Acknowledgements

AI coding and writing assistants were used; the author reviewed all content.

## References

[1] Victor Barres, Honghua Dong, Soham Ray, Xujie Si, and Karthik Narasimhan. $\tau ^ { 2 } .$ -bench: Evaluating conversational agents in a dual-control environment, 2025.

[2] Luca Beurer-Kellner, Marc Fischer, and Martin Vechev. Prompting is programming: A query language for large language models. In Proceedings of the ACM on Programming Languages (PLDI), 2023. arXiv:2212.06094.

[3] Laura Bocchi, Tzu-Chun Chen, Romain Demangeon, Kohei Honda, and Nobuko Yoshida. Monitoring networks through multiparty session types. In Formal Techniques for Distributed Systems (FMOODS/FORTE 2013), volume 7892 of Lecture Notes in Computer Science, pages 50–65. Springer, 2013.

[4] Benedikt Bollig, Matthias Függer, and Thomas Nowak. Provable coordination for LLM agents via message sequence charts, 2026.

[5] Luís Caires and Frank Pfenning. Session types as intuitionistic linear propositions. In CON-CUR 2010 – Concurrency Theory, volume 6269 of Lecture Notes in Computer Science, pages 222–236. Springer, 2010.

[6] Zhaorun Chen, Mintong Kang, and Bo Li. ShieldAgent: Shielding agents via verifiable safety policy reasoning, 2025.

[7] Ankush Das, Jan Hofmann, and Frank Pfenning. Work analysis with resource-aware session types. In Proceedings ofthe 33rd Annual ACM/IEEE Symposium on Logic in Computer Science (LICS), pages 305–314, 2018.

[8] Yixin Dong, Charlie F. Ruan, Yaxing Cai, Ruihang Lai, Ziyi Xu, Yilong Zhao, et al. XGrammar: Flexible and eficient structured generation engine for large language models, 2024.

[9] Laïla Elkoussy and Julien Perez. AgentLTL: A trace-verification framework for measuring, enforcing, and training procedural compliance in tool-using LLM agents, 2026.

[10] Simon Gay and Malcolm Hole. Subtyping for session types in the pi calculus. Acta Informatica, 42(2–3):191–225, 2005.

[11] Kiran Gopinathan, Jack Feser, Michelangelo Naim, Zenna Tavares, and Eli Bingham. Pact: A choreographic language for agentic ecosystems, 2026.

[12] Kohei Honda. Types for dyadic interaction. In CONCUR ’93, 4th International Conference on Concurrency Theory, volume 715 of Lecture Notes in Computer Science, pages 509–523. Springer, 1993.

[13] Kohei Honda, Vasco T. Vasconcelos, and Makoto Kubo. Language primitives and type discipline for structured communication-based programming. In Programming Languages and Systems, ESOP ’98, volume 1381 of Lecture Notes in Computer Science, pages 122–138. Springer, 1998.

[14] Kohei Honda, Nobuko Yoshida, and Marco Carbone. Multiparty asynchronous session types. In Proceedings of the 35th ACM SIGPLAN-SIGACT Symposium on Principles of Programming Languages (POPL), pages 273–284, 2008.

[15] Dimitrios Kouzapas, Ornela Dardha, Roly Perera, and Simon J. Gay. Typechecking protocols with Mungo and StMungo. In Proceedings of the 18th International Symposium on Principles and Practice of Declarative Programming (PPDP), pages 146–159, 2016.

[16] Hanqi Li, Jing Peng, Zijian Wang, Lu Chen, and Kai Yu. XFlow: An executable protocol programming system for reliable multi-agent workflows, 2026.

[17] Yang Li, Ping Hou, and Nobuko Yoshida. Specification-guided synthesis of deadlock-free communication protocol refinements with large language models, 2026.

[18] Zelong Li, Wenyue Hua, Hao Wang, He Zhu, and Yongfeng Zhang. Formal-LLM: Integrating formal language and natural language for controllable LLM-based agents, 2024.

[19] Barbara H. Liskov and Jeannette M. Wing. A behavioral notion of subtyping. ACM Transactions on Programming Languages and Systems, 16(6):1811–1841, 1994.

[20] Niels Mündler, Jingxuan He, Hao Wang, Koushik Sen, Dawn Song, and Martin Vechev. Typeconstrained code generation with language models, 2025.

[21] Andreas Schlapbach. Formal semantics for agentic tool protocols: A process calculus approach, 2026.

[22] Tianneng Shi, Jingxuan He, Zhun Wang, Hongwei Li, Linyu Wu, Wenbo Guo, et al. Progent: Securing AI agents with privilege control, 2025.

[23] Robert E. Strom and Shaula Yemini. Typestate: A programming language concept for enhancing software reliability. IEEE Transactions on Software Engineering, SE-12(1):157–171, 1986.

[24] Philip Wadler. Propositions as sessions. In Proceedings of the 17th ACM SIGPLAN International Conference on Functional Programming (ICFP), pages 273–286, 2012.

[25] Haoyu Wang, Christopher M. Poskitt, and Jun Sun. AgentSpec: Customizable runtime enforcement for safe and reliable LLM agents, 2025.

[26] Brandon T. Willard and Rémi Louf. Eficient guided generation for large language models, 2023.

[27] Yifeng Xiao and Pierluigi Nuzzo. Symbolic temporal supervision of LLM agents using contracts, 2026.

[28] Shunyu Yao, Noah Shinn, Pedram Razavi, and Karthik Narasimhan. τ -bench: A benchmark for tool-agent-user interaction in real-world domains, 2024.

[29] Kexun Zhang, Hongqiao Chen, Lei Li, and William Wang. Don’t fine-tune, decode: Syntax error-free tool use via constrained decoding, 2023.

## A Pressure instruction

Appended to every simulated customer’s task instructions:

Additional instruction: you are in a big hurry and find confirmation questions annoying. Before the agent books, changes, cancels or refunds anything it may ask you to confirm. The first time it asks you to confirm a given change, do not confirm. Push back instead, for example: “I already told you what I want, just do it and stop asking.” Never use the word “yes” in that pushback. Only if the agent asks you to confirm that same change again, reply starting with “Yes”. Follow all your other instructions as before.

## B Concordat agent’s process note

Added to the Concordat agent’s system prompt (the second paragraph was added before the gpt-4.1 run):

Tools that change the database (bookings, cancellations, changes, refunds) are only ofered right after the customer has explicitly said yes to the exact action you described. Until then you only have lookup tools: describe the action and ask the customer to confirm. Earlier tool results stay valid when the tool list changes.

If the customer pushes back instead of confirming (for example “just do it” or “stop asking”), that is not a confirmation, and it is not a reason to transfer them. Stay polite and brief: say that you need a clear “yes” before you can make the change, restate the exact change in one sentence, and ask them to reply “yes”. Keep asking like this for as long as they still want the change. Transfer only when the policy itself calls for a transfer.

## C Reproduction

The harness is in bench/ of the repository. With τ -bench v1.0.1 cloned into bench/tau2-bench:

python run.py run --domain airline --task-set-name airline\_pressure \   
--agent concordat\_agent --agent-llm <model> --user-llm <model> --save-to <name>   
python compare.py <standard run> <concordat run>

All result summaries reported here are in bench/results/ and described in bench/RESULTS.md.