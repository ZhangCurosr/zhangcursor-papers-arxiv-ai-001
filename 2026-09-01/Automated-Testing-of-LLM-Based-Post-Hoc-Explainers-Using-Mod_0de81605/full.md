# Automated Testing of LLM-Based Post Hoc Explainers Using Model Checking as an Oracle

Dennis Gross<sup>1,2[0009−0001−5734−0538]</sup> and Helge Spieker<sup>2[0000−0003−2494−4279]</sup>

<sup>1</sup> Institut für Kommunikations- und Prüfungsforschung gGmbH 2 Simula Research Laboratory, Oslo, Norway Corresponding author: dennis@artigo.ai

Abstract. Large language models (LLMs) are used as post hoc explainers of sequential decision-making policies, producing natural-language explanations of why an action was chosen. However, LLMs often generate plausible but incorrect statements, and no existing approach systematically tests whether such explanations are faithful to the underlying environment. Two classic software testing challenges stand in the way: there is no oracle for the correctness of an explanation, and the test inputs, natural language queries about a policy’s behavior, lack the structure needed for systematic test case generation. We address both. Probabilistic model checking provides the test oracle, computing exact reference results against which LLM answers are graded automatically. A taxonomy of post hoc query categories structures the input space around the environment-level facts from which policy explanations are composed; test cases generated from it are prioritized by question-specific diagnostic dificulty scores. Across seven MDP environments, the testing separates three open-weight LLMs: a reasoning model passes 85% of test cases, a mid-size model 70%, and a 1B model falls below the random baseline, while prioritization surfaces significantly harder cases than random selection. Our results indicate how trustworthy LLM-generated explanations are in model-free settings, where the same LLMs are used but no oracle exists to verify them.

Keywords: Software testing · Large language models · Probabilistic model checking · Explainable reinforcement learning · Test oracle.

## 1 Introduction

Sequential decision-making tasks are at the core of applications in domains such as gaming [10], manufacturing [13], and healthcare [11]: an agent repeatedly observes the current state of its environment, selects an action, and thereby stochastically influences the successor state to reach a fixed objective. Such tasks are formally modeled as Markov decision processes (MDPs) [29]. We focus on MDPs with finitely many states and actions, and on memoryless policies, which base each decision solely on the current state rather than on the history of past states and actions. The consequences of a single decision unfold over subsequent steps. Policies are typically obtained automatically, e.g., via deep reinforcement learning (RL) [27], so the rationale behind an individual choice is often not accessible to the human user: the learned policy is a neural network whose internal computations are not human-interpretable. To address this, large language models (LLMs) have recently been employed as post hoc explainers that analyze a decision after the fact, rather than making the policy interpretable by design [6]: after the policy has made a decision, the LLM is prompted to produce a natural language explanation of why that action was chosen in the given state.

However, LLMs are known to produce plausible-sounding but incorrect statements [28], which raises a software testing problem: before such explanations can be trusted, the LLM-based explainer must itself be tested. Testing it is challenging for two reasons. First, there is no test oracle [4]: judging whether an explanation is correct requires knowing the true properties of the environment and policy. Second, the input space is unstructured: it is unclear which kinds of explanation queries exist and which concrete queries are worth posing. Consequently, while prior work has evaluated LLM-generated explanations [23,35] and even used them to improve policy performance [14], no existing approach systematically tests whether an LLM actually understands the underlying environment and produces explanations faithful to it.

In this paper, we present an automated approach for testing LLM-based post hoc explainers of sequential decision-making policies. We address the oracle problem via probabilistic model checking [3]: given a formal model of the environment and a property specified in a temporal logic such as PCTL [19], a model checker exhaustively analyzes the state space and computes, for instance, the exact probability of reaching a goal or violating a safety condition. These results are exact and carry formal guarantees, and thus serve as a reliable oracle against which the LLM’s answers are graded automatically. To structure the test input space, we introduce a taxonomy of post hoc query categories, which guides generating targeted test queries, for example, which actions in a state are optimal or whether a state is safety-critical, and shows, per category, where diferent LLMs succeed or fail. These queries target the environment-level facts, such as optimality and safety, from which any post hoc explanation of a policy’s decision is composed. Since not every state is equally informative for every query, we prioritize test inputs by a question-specific diagnostic score, such as the fraction of non-optimal actions, and select the highest-ranked states for testing. Our approach requires a formal model of the environment, but this restriction applies only to testing, not to its implications: the formal model provides a measurement instrument for the LLM’s reasoning capabilities. Test results obtained in this controlled setting thereby indicate how trustworthy LLM-generated explanations are in model-free settings, where the same LLMs are applied but no oracle exists to verify them.

## 2 Related Work

Our work touches four research areas: explainability for sequential decisionmaking, LLMs as post hoc explainers, testing and evaluating LLMs, and model checking for AI systems.

Explainability for sequential decision-making. A large body of work on explainable reinforcement learning (XRL) [34] generates post hoc explanations of learned policies, e.g., through policy summarization [30], saliency maps [33], and reward decomposition [25], and, most recently, by employing LLMs as post hoc explainers [6]. All these approaches focus on producing explanations; in contrast, we test whether such an explainer produces explanations that are faithful to the underlying environment.

LLMs as post hoc explainers. A growing line of work employs LLMs as post hoc explainers of learned policies, prompting them to produce natural language accounts of why an agent chose a given action [42,26,7,22,43]. Across this line of work, explanations are judged only against approximate references, such as human studies [45] and LLM-as-a-judge [5]. In contrast, we grade explanations against an exact ground truth from probabilistic model checking, organized along a taxonomy of post hoc query categories.

Testing and evaluating LLMs. Beyond explanations, a broad literature tests LLMs directly, ranging from behavioral test suites [31] and hallucination benchmarks [24,21] to benchmarks with verifiable ground truth, such as PlanBench [39], which grades LLM plans via automated validators; the lack of reliable test oracles for ML systems is a well-known challenge [44,4]. However, where an exact ground truth exists, it tests the LLM’s own task performance, not the faithfulness of its explanations of another system’s decisions. We use probabilistic model checking as a test oracle for exactly this purpose.

Model checking for AI systems. Probabilistic model checking has been used to verify learned policies against PCTL properties by model checking the policyinduced Markov chain [2,12,16], also in combination with explainability methods [17,1] and with LLMs, either generating counterfactuals for policy repair [15] or acting as the policy under verification [18]. In all these works, the object under verification is the policy. In contrast, we use model checking as a test oracle: its exact results serve as ground truth against which we grade the faithfulness of LLM-generated explanations.

## 3 Background

We introduce probabilistic systems, probabilistic model checking, LLMs, and the software testing concepts used throughout the paper, together with the notation used in later sections.

## 3.1 Probabilistic Systems

We model sequential decision-making (see Fig. 1) as a Markov decision process (MDP), a tuple $\mathcal { M } = ( S , s _ { 0 } , A c t , P , A P , L )$ where S is a finite set of states, $s _ { 0 } \in S$ the initial state, Act a finite set of actions, $P : S \times A c t \times S \to [ 0 , 1 ]$ a transition function with $\begin{array} { r } { \sum _ { s ^ { \prime } } P ( s , a , s ^ { \prime } ) = 1 } \end{array}$ for every enabled action a, AP a finite set of atomic propositions, and $L : S \to 2 ^ { A { \dot { P } } }$ a labeling function mapping each state to the set of atomic propositions that hold in it. We write $A c t ( s ) \subseteq .$ Act for the actions enabled in state s. In each state, the agent picks an action $^ { a , }$ and the environment draws the next state from the distribution $P ( s , a , \cdot )$ . For example, in a slippery gridworld, the action right may reach the intended tile with probability 0.8 and drift to a perpendicular neighbor otherwise. Labels identify the states of interest: the goal states are those labeled goal, i.e. {s : goal $\in \ { L ( s ) } \}$ , and the unsafe set $U \subseteq S$ is defined analogously via an unsafe label. A (deterministic, memoryless) policy $\pi : S $ Act chooses an action in each state; such policies sufice to attain the optimal reachability probabilities we consider. Fixing π resolves all choices and collapses the MDP into a induced discrete-time Markov chain (DTMC), whose behavior is fully determined by $P$ and π.

![](images/a881cec7dfce91d1aebf4f227df514a73f858c182f46ae9c811518edcfd2f54a.jpg)  
Fig. 1. A sequential decision-making system in which an agent interacts with an environment. The agent receives a state and a reward from the environment based on its previous action. The agent then uses this information to select the next action via the policy, which it sends to the environment.

## 3.2 Probabilistic Model Checking

Given a specification written in Probabilistic Computation Tree Logic (PCTL) [19], a model checker computes the probability that the specification holds. We use reachability properties of the form $\mathsf { P } _ { \mathrm { m a x } } { = } ? \left[ \mathsf { F } g o a l \right]$ , read as “what is the maximal probability of eventually (F) reaching a goal state?”; dually, $\mathsf { P } _ { \mathrm { m i n } } { = } ?$ [ F unsafe ] asks for the minimal such probability over all policies. For every state $s ,$ the checker computes the property result $V ( s )$ , the optimal probability of satisfying the property from $s ,$ and for every action $a \in A c t ( s )$ the action value $Q ( s , a )$ , the property result obtained by taking a in s and acting optimally afterwards; we write $Q _ { ( 1 ) } \geq Q _ { ( 2 ) } \geq . .$ . for the decreasing sort of the action values in a state. For the reachability properties $\mathsf { P } _ { \mathrm { m a x } } [ \mathsf { F } g o a l ]$ we consider, V and $Q$ coincide exactly with the reinforcement learning value and action-value functions under the reward structure granting reward 1 on entering a goal state (made absorbing) and 0 otherwise, undiscounted $( \gamma = 1 )$ , so the familiar $V / Q$ notation carries over. We further use the danger $D ( s ) = \mathsf { P } _ { \operatorname* { m i n } } [ \mathsf { F } U ]$ , the minimal reachability probability of the unsafe set from $s ,$ obtained by a second model-checking run. Tools such as Storm [20] compute these quantities eficiently over the reachable states.

## 3.3 Large Language Models

A large language model (LLM) is a neural network based on the transformer architecture [40]. It operates on tokens, i.e., elements of a finite vocabulary V into which a tokenizer segments natural-language text. Formally, an LLM defines a function $f _ { \theta } \colon \mathcal { V } ^ { * } \to \varDelta ( \nu )$ that maps a token sequence to a probability distribution over the next token. Text is generated autoregressively: given an input sequence, the model samples a next token from $f _ { \theta }$ , appends it to the sequence, and repeats until a designated stop token is produced; the resulting token sequence is decoded back into text.

We use an LLM as a post hoc explainer as follows. A prompt is a naturallanguage input text that describes the environment, a state s, and a query about the decision-making task in s. The prompt is tokenized and passed to the model, which generates a natural-language response. A parser then maps this response to a structured answer, such as a yes/no verdict or an action ranking.

## 3.4 Software Testing

Testing executes a system under test (SUT) on selected inputs, called test cases, and compares the observed behavior against the expected behavior. The mechanism that provides the expected behavior is the test oracle [4]; its absence or unreliability, known as the oracle problem [41], is a central challenge when testing systems whose correct output is not readily known, such as machine learning components [44]. Since executing all possible test cases is generally infeasible, test case prioritization orders them so that the most informative ones are executed first [32]. In our setting, the SUT is the LLM-based explainer; a test case pairs the (extracted) MDP and its natural-language description with a statelevel query; the model checker’s exact results serve as the test oracle; and our diagnostic state ranking instantiates test-case prioritization.

## 4 A Taxonomy of Post Hoc Test Queries

The model checker provides a test oracle for each state: the property result $V ( s )$ and the ranking of available actions by their values $Q ( s , a )$ . Any post hoc explanation of a concrete policy decomposes into state-level claims about the environment, for instance, that an action is suboptimal, that a state is unsafe, or that a decision leads to a dead end. Our queries test exactly these atomic claims against the optimal behavior in the environment: a model that fails them cannot ground faithful explanations of any policy acting in it, so passing them is a necessary condition for explanation faithfulness. Queries relative to a specific fixed policy π, whose induced DTMC the model checker analyzes in the same way, fit the same scheme. A test case is a query posed to the LLM under test; we obtain verdicts by automatically comparing its answers against the oracle and prioritize test cases so that the most diagnostic ones are executed first (Section 5). Queries vary along three dimensions: object (the property result or the action ranking), scope (one state or a subset), and mode (judging one object or comparing two). The examples below illustrate each dimension but are not exhaustive; any query that can be answered by the oracle can be added in the same way.

## 4.1 Derived Notions and Grading

Beyond $V , Q ,$ and D from the background, our queries use the following derived notions. The optimal action set $A ^ { \star } ( s ) = \{ a \in A c t ( s ) : Q ( s , a ) = \operatorname* { m a x } _ { a ^ { \prime } } Q ( s , a ^ { \prime } ) \}$ contains the maximizers of the action value in $s ,$ and the worst action set $A ^ { \circ } ( s )$ is defined dually as the set of minimizers. A state with $V ( s ) = 0$ is a dead end: the property can no longer be satisfied from it. A state s is a bottleneck if making s absorbing reduces the maximal goal-reachability probability from $s _ { 0 }$ to $0 ;$ this is decided exactly by re-checking the modified model. Finally, $C _ { B } ( s )$ denotes the betweenness centrality [9] of s in the $\mathrm { M D P ^ { \prime } s }$ underlying directed graph, which has an edge $s  s ^ { \prime }$ if $P ( s , a , s ^ { \prime } ) > 0$ for some $a . ~ C _ { B }$ serves only as a ranking heuristic, never as ground truth. Grading is exact: the probability answered for the Satisfy query is correct if it equals the model checker’s result $V ( s ) { \mathrm { : } }$ ; a named best (worst) action is correct if it lies in $A ^ { \star } ( s ) ~ ( A ^ { \circ } ( s ) )$ ; and in ranking queries, equal-valued actions may be ordered arbitrarily and are graded as either-order-correct.

## 4.2 Object

The object is the quantity a query reads, giving two families.

State queries Judgments about the state alone, independent of any action.

– Satisfy property? How likely the property is achieved from this state, answered as a probability estimate and graded against $V ( s )$

– Bottleneck? Whether every successful run must pass through this state, i.e. the goal becomes unreachable without it.

Preference queries Judgments about the actions at a given state.

– Best action? Which action is optimal, graded against $A ^ { \star } ( s )$

– Worst action? Which action is most damaging, graded against $A ^ { \circ } ( s )$

– Full ranking? How all actions order, graded against the sorted action values.

Note that a passing verdict on best-action test cases does not imply that the LLM, deployed as a policy over full trajectories, reaches the target states.

## 4.3 Scope

A local query reads the object at one state; a global query reads it over a subset $S ^ { \prime } \subseteq S$

– Bottleneck in the subset? Whether any state in $S ^ { \prime }$ is a bottleneck state.

– Dead ends in the subset? Whether $S ^ { \prime }$ contains a dead-end state, i.e. one with $V ( s ) = 0$

## 4.4 Mode

An individual query judges one object; a relational query compares two.

– Which state is more promising? Which of two states has the higher property result.

– Which state is safer? Which of two states has the lower danger D.

– Which state is the bottleneck? Given two states, which one every successful run must pass through.

## 5 Testing Approach

Our approach takes as input an MDP, the PCTL properties of interest (the main property and, where danger queries are used, the safety property defining D), the LLMs under test, a query category from Section 4 with a prompt template and a diagnostic scoring method, a test budget (how many test cases to execute), and a sample size (how many times each test case is repeated to account for LLM nondeterminism). It outputs per-category verdicts for each LLM. The pipeline has four stages.

Oracle construction. The MDP and the PCTL properties are model-checked, yielding for every reachable state the property result $V ( s )$ , the action values $Q ( s , a )$ , and, where required, the danger $D ( s )$ . From this we derive the expected answers the test cases need: the action ranking per state, the optimal policy, the dead ends $( V ( s ) = 0 )$ , and the bottleneck states, the latter by re-checking reachability with the candidate state made absorbing (Section 4.1). For the bestaction query, for example, this yields each state’s optimal action set $A ^ { \star } ( s )$ , the expected output against which the LLM’s answer is later compared.

Test case generation and prioritization. Each test case concerns a single state (state and preference queries), a pair of states (mode queries), or a subset of states (scope queries). The system generates candidate test cases of the right kind, scores each by a diagnostic dificulty $\delta ,$ and keeps the highest-δ (hardest) ones up to the test budget; the three binary categories (bottleneck, subsetbottleneck, subset-dead-end) first balance positive and negative cases and then rank within each class by δ. Diagnostic dificulty targets the cases most likely to be answered wrongly, via one of three notions: ambiguity (candidate values nearly tied, so the answer is barely determined), selectivity (a single correct choice among many distractors), and salience (a decoy that mimics the true structure). Using the notation of Sections 4 and 4.1, the dificulty is:

– Satisfy (ambiguity): $\begin{array} { r } { \delta = 1 - 2 | V ( s ) - \frac { 1 } { 2 } | } \end{array}$ , the property result closest to $\frac { 1 } { 2 }$

– Best action (selectivity): $\delta = 1 - | A ^ { \star } ( \bar { s } ) | / | A c t ( s ) |$ ; few actions are optimal, so the best must be singled out from many distractors.

– Worst action (selectivity): $\delta = 1 - | A ^ { \circ } ( s ) | / | A c t ( s ) |$ , symmetric to best action: few actions are worst.

– Full ranking (ambiguity): $\begin{array} { r } { \delta = 1 - \operatorname* { m i n } _ { i } \left( Q _ { ( i ) } - Q _ { ( i + 1 ) } \right) } \end{array}$ , the smallest adjacent gap in the true ordering; equal-valued actions are graded as either-ordercorrect (Section 4.1).

– More promising (ambiguity): $\delta = 1 - | V ( s _ { 1 } ) - V ( s _ { 2 } ) |$ , the two states are close in property result.

– Safer state (ambiguity): $\delta = 1 - | D ( s _ { 1 } ) - D ( s _ { 2 } ) |$ , the two states are close in danger.

– Bottleneck (salience): $\delta = C _ { B } ( s )$ ; the state lies on many paths, so it is the most bottleneck-like. The verdict itself is exact (Section 4.1), and $C _ { B }$ only ranks candidates.

– Which-is-bottleneck (salience): $\delta = C _ { B } ( s _ { d e c o y } )$ , a central non-bottleneck decoy paired with the true bottleneck.   
– Subset-bottleneck (salience): $\delta = \operatorname* { m a x } _ { s \in S ^ { \prime } } C _ { B } ( s )$ , the subset’s most bottleneck like member.

– Subset-dead-end (ambiguity, negatives): $\delta \ = \ 1 - \mathrm { m i n } _ { s \in S ^ { \prime } } V ( s )$ , a subset whose lowest value is near zero; positive subsets (containing a dead end, so mi $\displaystyle \mathrm { 1 } _ { s \in S ^ { \prime } } V ( s ) = 0 )$ tie at $\delta = 1$ and are sampled uniformly.

The 1 − gap forms above assume values in [0, 1], as for the reachability probabilities $V , D$ used throughout our experiments; more generally δ increases as the relevant gap shrinks, and for expected-reward properties the gaps are normalized by the value range so that $\delta \in [ 0 , 1 ]$

Test execution. Each selected test case is rendered into a prompt by the template, with its state, pair, or subset filled in, and sent to every LLM under test, repeated sample size times. The template is the only environment-specific part: it fixes how a state is described in words and what the LLM is asked. For the full-ranking query, it might read:

You are an agent in a 5×5 gridworld. Your goal is to reach the exit at (4, 4) while avoiding the trap at (2, 3).

Current position: {state}.

Available actions: {actions}.

Order all available actions from best to worst. Respond only with a JSON object of the form {"ranking": ["<best action>", ..., "<worst action>"]}.

At execution time, the placeholders are filled from the selected state, e.g. {state} $ ( 1 , 3 )$ and {actions} → up, down, left, right, and the LLM returns a structured reply, for instance: {"ranking": ["right", "down", "up", "left"]}.

Verdict. Each answer is parsed and compared automatically against the oracle, and verdicts are aggregated over repetitions and per LLM. The comparison depends on the category: an answered probability is compared for equality with $V ( s )$ , and a named best (worst) action is matched against $A ^ { \star } ( s ) ~ ( A ^ { \circ } ( s ) )$ , a predicted ranking is compared to the true ordering up to ties, a more-promising or safer-state answer is checked against the two property results, a claimed bottleneck is verified against the exact bottleneck set, and a subset verdict is checked against whether $S ^ { \prime }$ actually contains a bottleneck or dead end.

![](images/6c58ca26e3a4bc14ea63af6ce178664003f43d48f5bd6c5dbb0efdd91e8a983c.jpg)  
Fig. 2. Overview of our testing approach. Model checking the MDP against the PCTL property yields an exact test oracle (top). The query taxonomy structures the input space; test cases are prioritized by a diagnostic dificulty score δ derived from the oracle’s values and rendered into prompts for the LLM under test (bottom). Verdicts are obtained by automatically comparing the LLM’s answers against the oracle.

## 6 Evaluation

We evaluate our approach to answer three research questions.

RQ1 (Prioritization): Does the diagnostic ranking surface non-trivial cases? RQ2 (Discrimination): Do stronger LLMs pass more test cases than weaker ones?

RQ3 (Dificulty): Do query categories difer in dificulty?

## 6.1 Experimental Setup

Environments. We use seven MDPs, each paired with a reachability property whose model-checked result defines the oracle: Frozen Lake (a slippery 4 × 4 gridworld; reach the goal without falling into a hole; one bottleneck), Wolf– Goat–Cabbage (the river-crossing puzzle; move all items across without an unsafe pair; four bottlenecks), Water Jug (the classic water-measuring puzzle; reach the target volume within a move budget; no bottlenecks), Transporter (a pickupand-delivery task under stochastic movement; collect and deliver the item to its destination; no bottlenecks), Stock Market (a trading task with stochastic price movements; reach a target portfolio value without going bankrupt; no bottlenecks), Job Shop (a scheduling task with stochastic job durations; complete all jobs within a deadline; no bottlenecks), and Dam (a water-level control task with stochastic inflow; keep the reservoir within safe bounds while meeting demand; no bottlenecks). Environments without bottleneck states omit the three bottleneck categories (reported as “−”).

LLMs. We test three open-weight models of increasing scale served through Ollama: Gemma 3 1B [36] (small, run locally), Qwen3.5 [38] (a larger cloud model with a step-wise reasoning mode), and Gemma 4 31B [37](a larger cloud model). A uniform-Random answerer is the baseline; the optimal policy is the ceiling at 1 by construction.

Protocol. All models are decoded at temperature 0 (greedy), at which they are near-deterministic, so we use a sample size of 1. The test budget is 20 states per category, selected by the diagnostic dificulty δ of Section 5. Answers are constrained to structured JSON, which is parsed automatically. The oracle is computed once per environment with the Storm model checker [20] and reused across all models and categories. Scores are the fraction of passing test cases in [0, 1] (higher is better). We executed all experiments in a Docker container (16 GB RAM) on an AMD Ryzen 7 7735HS (16 threads) running Ubuntu 20.04.5 LTS. For model checking, we use Storm 1.12.0. The Ollama LLMs were hosted either on the same machine outside the Docker container or on the Ollama cloud and accessed via the REST API.

## 6.2 Results

We first validate the selection method itself (RQ1), then read of LLM performance from the table (RQ2), and finally rank the categories by dificulty (RQ3). Table 1 reports per-category scores for all LLMs and environments, and Figure 3 ranks the query categories by dificulty.

Efect of diagnostic prioritization (RQ1). We compare, per category cell, the score under diagnostic prioritization against the score under random state selection; a lower prioritized score means prioritization surfaced harder cases. Over all 220 comparable cells, prioritization yields the harder case in 75 cells (34.1%), the easier case in 61 (27.7%), and a tie in 84 (38.2%). A one-sided Wilcoxon signed-rank test rejects the null in favor of prioritized < random at p = 0.035.

Consistent with the mechanism, prioritization is not expected to depress every model’s score, and it does not. It lowers a score only when a model is genuinely sensitive to oracle dificulty (small value gaps, action ambiguity, boundary states). The Random baseline is insensitive by construction; its expected score is fixed at the chance level of the answer space regardless of which states are chosen, so its prioritized-versus-random cells split roughly evenly, and its gaps are sampling noise. The small Gemma 3 1B is insensitive for a diferent reason: as the model comparison below shows, it is already at floor on most categories, so it has little room to fall further on cases the oracle labels dificult. This answers RQ1: under diagnostic prioritization, we obtain more non-trivial test cases than with random test prioritization.

LLM performance (RQ2). Averaged over all environments, the reasoning model Qwen3.5 is strongest at 0.85, followed by Gemma 4 31B at 0.70, both clearly above the Random baseline at 0.51. Under diagnostic prioritization, the small Gemma 3 1B sits below random at 0.43. At 1B parameters, it does not merely guess but commits to wrong answers in categories where random guessing would score at chance. This deficit is specific to the prioritized test cases, which surface exactly these hard states. Under random selection, the same model rises to 0.55, essentially tied with random (0.54), so it is only exposed once the benchmark concentrates on diagnostic queries. This answers RQ2: stronger models pass more test cases than both weaker models and the random baseline.

Table 1. LLM performance across environments and query categories. State queries (satisfy property, bottleneck) and preference queries (best action, worst action, full ranking) vary along the object dimension; scope queries (bottleneck in subset, dead ends in subset) and mode queries (more promising, safer state, bottleneck) cover the scope and mode dimensions. Cells report the per-subcategory score in [0, 1]; higher is better. Each cell reports the diagnostic-prioritization score followed by the random-selection score (prioritized/random); the prioritized score is shown in bold where prioritization yields a harder (lower) score than random selection.
<table><tr><td></td><td></td><td colspan="2">State</td><td colspan="3">Preference</td><td colspan="2">Scope</td><td colspan="3">Mode</td><td></td></tr><tr><td>Environment</td><td>Model</td><td>Sat.</td><td>Bttl.</td><td>Best</td><td>Worst</td><td>Rank</td><td>Sub.B</td><td>Sub.D</td><td>Prom.</td><td>Safe</td><td>Bttl.</td><td>Avg.</td></tr><tr><td rowspan="4">Frozen Lake</td><td>Gemma 3 1B</td><td>0.65/0.65</td><td>0.13/0.13</td><td>0.27/0.27</td><td>0.40/0.40</td><td>0.50/0.50</td><td>0.45/0.45</td><td>0.50/0.50</td><td>0.12/0.38</td><td>0.50/0.50</td><td>0.67/0.20</td><td>0.42/0.40</td></tr><tr><td>Qwen3.5</td><td>0.74/0.77</td><td>1.00/1.00</td><td>0.55/0.73</td><td>0.70/0.90</td><td>0.93/0.85</td><td>0.95/1.00</td><td>0.95/1.00</td><td>0.88/0.88</td><td>0.50/0.50</td><td>1.00/1.00</td><td>0.82/0.86</td></tr><tr><td>Gemma 4 31B</td><td>0.83/0.83</td><td>0.93/0.93</td><td>0.55/0.55</td><td>0.90/0.90</td><td>0.87/0.77</td><td>0.55/0.50</td><td>1.00/0.90</td><td>0.75/0.88</td><td>0.33/0.33</td><td>0.93/0.93</td><td>0.76/0.75</td></tr><tr><td>Random</td><td>0.48/0.61</td><td>0.53/0.67</td><td>0.27/0.36</td><td>0.20/0.20</td><td>0.42/0.57</td><td>0.40/0.40</td><td>0.40/0.50</td><td>0.62/0.50</td><td>0.33/1.00</td><td>0.53/0.53</td><td>0.42/0.53</td></tr><tr><td rowspan="4">Wolf-Goat-Cabbage</td><td>Gemma 3 1B</td><td>0.61/0.61</td><td>0.31/0.31</td><td>0.78/0.78</td><td>0.20/0.20</td><td>0.57/0.57</td><td>0.50/0.50</td><td>0.50/0.50</td><td>1.00/0.00</td><td>0.00/1.00</td><td>0.35/0.55</td><td>0.48/0.50</td></tr><tr><td>Qwen3.5</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>0.90/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>0.99/1.00</td></tr><tr><td>Gemma 4 31B</td><td>0.64/0.64</td><td>0.85/0.69</td><td>1.00/1.00</td><td>0.40/0.40</td><td>0.87/0.87</td><td>0.70/0.80</td><td>0.50/0.50</td><td>1.00/0.00</td><td>0.00/1.00</td><td>0.95/0.95</td><td>0.69/0.69</td></tr><tr><td>Random</td><td>0.45/0.45</td><td>0.54/0.54</td><td>0.56/0.67</td><td>0.60/0.40</td><td>0.77/0.67</td><td>0.45/0.60</td><td>0.60/0.60</td><td>1.00/0.00</td><td>1.00/1.00</td><td>0.45/0.50</td><td>0.64/0.54</td></tr><tr><td rowspan="4">Water Jug</td><td>Gemma 3 1B</td><td>0.57/0.41</td><td>-1-</td><td>0.35/0.35</td><td>0.71/0.71</td><td>0.27/0.27</td><td>-1-</td><td>0.50/0.55</td><td>0.00/0.00</td><td>0.00/1.00</td><td>-1-</td><td>0.34/0.47</td></tr><tr><td>Qwen3.5</td><td>1.00/1.00</td><td>-/-</td><td>1.00/1.00</td><td>1.00/1.00</td><td>0.98/1.00</td><td>-1-</td><td>0.90/0.80</td><td>1.00/1.00</td><td>1.00/1.00</td><td>-1-</td><td>0.98/0.97</td></tr><tr><td>Gemma 4 31B</td><td>0.65/0.75</td><td>-1-</td><td>0.70/0.70</td><td>0.82/0.82</td><td>0.85/0.83</td><td>-1-</td><td>0.70/0.55</td><td>1.00/1.00</td><td>0.00/0.00</td><td>-1-</td><td>0.68/0.67</td></tr><tr><td>Random</td><td>0.48/0.41</td><td>-1-</td><td>0.40/0.45</td><td>0.65/0.71</td><td>0.73/0.64</td><td>-1-</td><td>0.40/0.50</td><td>0.00/1.00</td><td>1.00/0.00</td><td>-1-</td><td>0.52/0.53</td></tr><tr><td rowspan="4">Transporter</td><td>Gemma 3 1B</td><td>0.99/0.69</td><td>-1-</td><td>0.80/0.95</td><td>0.60/0.70</td><td>0.50/0.63</td><td>-1-</td><td>0.50/0.50</td><td>0.00/1.00</td><td>0.00/1.00</td><td>-1-</td><td>0.48/0.78</td></tr><tr><td>Qwen3.5</td><td>1.00/0.85</td><td>-1-</td><td>0.75/0.95</td><td>0.90/0.85</td><td>0.97/0.96</td><td>-1-</td><td>0.65/0.60</td><td>1.00/1.00</td><td>1.00/1.00</td><td>-1-</td><td>0.90/0.89</td></tr><tr><td>Gemma 4 31B</td><td>1.00/0.80</td><td>-1-</td><td>0.90/0.95</td><td>0.70/0.75</td><td>0.91/0.86</td><td>-1-</td><td>0.75/0.50</td><td>1.00/1.00</td><td>0.00/1.00</td><td>-1-</td><td>0.75/0.84</td></tr><tr><td>Random</td><td>0.46/0.40</td><td>-1-</td><td>0.20/0.90</td><td>0.25/0.50</td><td>0.73/0.72</td><td>-1-</td><td>0.35/0.50</td><td>1.00/1.00</td><td>1.00/0.00</td><td>-1-</td><td>0.57/0.57</td></tr><tr><td rowspan="4">Stock Market</td><td>Gemma 3 1B</td><td>0.87/0.87</td><td>-1-</td><td>1.00/1.00</td><td>0.00/0.00</td><td>0.00/0.00</td><td>-1-</td><td>0.50/0.50</td><td>0.00/1.00</td><td>0.00/1.00</td><td>-1-</td><td>0.34/0.62</td></tr><tr><td>Qwen3.5</td><td>0.53/0.80</td><td>-1-</td><td>1.00/1.00</td><td>1.00/1.00</td><td>1.00/1.00</td><td>-1-</td><td>0.80/0.65</td><td>1.00/1.00</td><td>1.00/1.00</td><td>-1-</td><td>0.90/0.92</td></tr><tr><td>Gemma 4 31B</td><td>0.72/0.87</td><td>-/-</td><td>0.80/1.00</td><td>0.95/0.75</td><td>0.40/0.50</td><td>-1-</td><td>0.60/0.70</td><td>1.00/1.00</td><td>1.00/1.00</td><td>-1-</td><td>0.78/0.83</td></tr><tr><td>Random</td><td>0.46/0.48</td><td>-1-</td><td>0.40/1.00</td><td>0.30/0.65</td><td>0.50/0.30</td><td>-1-</td><td>0.50/0.50</td><td>0.00/1.00</td><td>1.00/0.00</td><td>-1-</td><td>0.45/0.56</td></tr><tr><td rowspan="4">Job Shop</td><td>Gemma 3 1B</td><td>0.53/0.43</td><td>-1-</td><td>0.35/0.15</td><td>0.65/0.80</td><td>0.47/0.25</td><td>-1-</td><td>0.50/0.50</td><td>0.80/0.35</td><td>0.50/0.45</td><td>-1-</td><td>0.54/0.42</td></tr><tr><td>Qwen3.5</td><td>0.69/0.76</td><td>-1-</td><td>0.25/0.60</td><td>0.70/0.40</td><td>0.77/0.62</td><td>-1-</td><td>0.50/0.50</td><td>0.55/0.50</td><td>0.70/0.60</td><td>-1-</td><td>0.59/0.57</td></tr><tr><td>Gemma 4 31B</td><td>0.72/0.79</td><td>-/-</td><td>0.50/0.30</td><td>0.80/0.00</td><td>0.58/0.57</td><td>-1-</td><td>0.50/0.50</td><td>0.70/0.55</td><td>0.65/0.50</td><td>-1-</td><td>0.64/0.46</td></tr><tr><td>Random</td><td>0.75/0.54</td><td>-/-</td><td>0.30/0.50</td><td>0.30/0.60</td><td>0.60/0.55</td><td>-1-</td><td>0.65/0.50</td><td>0.65/0.40</td><td>0.45/0.40</td><td>-1-</td><td>0.53/0.50</td></tr><tr><td rowspan="4">Dam</td><td>Gemma 3 1B</td><td>0.51/0.33</td><td>-1-</td><td>0.00/0.75</td><td>0.10/0.65</td><td>0.92/0.75</td><td>-1-</td><td>0.50/0.50</td><td>0.55/0.75</td><td>0.45/0.75</td><td>-1-</td><td>0.43/0.64</td></tr><tr><td>Qwen3.5</td><td>0.73/0.87</td><td>-1-</td><td>0.95/1.00</td><td>0.85/0.85</td><td>0.97/0.96</td><td>-1-</td><td>0.50/0.50</td><td>0.70/0.60</td><td>0.45/0.55</td><td>-1-</td><td>0.74/0.76</td></tr><tr><td>Gemma 4 31B</td><td>0.63/0.91</td><td>-/-</td><td>0.90/1.00</td><td>0.25/0.55</td><td>0.97/0.92</td><td>-1-</td><td>0.55/0.65</td><td>0.60/0.70</td><td>0.50/0.45</td><td>-1-</td><td>0.63/0.74</td></tr><tr><td>Random</td><td>0.73/0.61</td><td>-1-</td><td>0.20/0.60</td><td>0.30/0.45</td><td>0.72/0.68</td><td>-1-</td><td>0.35/0.55</td><td>0.35/0.45</td><td>0.30/0.65</td><td>-/-</td><td>0.42/0.57</td></tr></table>

Category dificulty (RQ3). Figure 3 ranks the query categories by the mean score pooled over all models and both selection strategies, so the ranking reflects intrinsic task dificulty rather than an artifact of which states were sampled. The hardest categories are dead ends in subset (0.59) and worst action (0.60), followed by bottleneck in subset (0.63); the easiest are the binary bottleneck queries, at 0.72 for the relational which-is-bottleneck variant. This answers RQ3: the categories difer in dificulty.

## 7 Threats to Validity

Internal. Verdicts depend on parsing model output into structured answers; we mitigate this with JSON-constrained decoding. Greedy decoding with sample size 1 captures the modal, not average, behavior.

External. Conclusions rest on seven (exactly model-checkable) environments and three open-weight LLMs, and may not transfer to larger tasks or proprietary models. Templates are handwritten and environment-specific; rewording the same environment can shift absolute scores.

![](images/4b0d2db3b91859a7bad527068ecc51ec8067539e7f968586189d657be05217ab.jpg)  
Fig. 3. Query-category dificulty, averaged over both selection strategies (diagnostic prioritization and random). Lower is harder; bars are colored by taxonomy dimension.

Construct. We test faithfulness to optimal behavior; passingbest-action cases do not imply the LLM reaches the goal as a policy, nor that its wording is causally faithful to its own computation. The dificulty δ (and betweenness centrality in particular) only ranks candidates; the verdicts remain exact.

Conclusion. The prioritization efect is statistically significant over all 220 cells (p=0.035) but modest in magnitude.

## 8 Conclusion

We presented an automated approach for testing LLM-based post hoc explainers: probabilistic model checking is an exact test oracle, a taxonomy of query categories structures the input space, and diagnostic scores prioritize test cases. Across seven environments and three LLMs, the approach cleanly discriminates between models, localizes, per category, where each model can be trusted, and prioritizes significantly harder cases than random selection would. Since the same LLMs are deployed where no oracle exists, these results quantify a trust gap that would otherwise go unmeasured.

Future work includes using failed test cases to fine-tune explainers [8].

Acknowledgments. This work is funded by the European Union under grant agreement number 101091783 (MARS Project).

Disclosure of Interests. The authors declare that they have no competing financial interests.

## References

1. Ashok, P., Jackermeier, M., Křetínsk\`y, J., Weinhuber, C., Weininger, M., Yadav, M.: dtcontrol 2.0: explainable strategy representation via decision tree learning steered by experts. In: International Conference on Tools and Algorithms for the Construction and Analysis of Systems. pp. 326–345. Springer (2021)

2. Bacci, E., Parker, D.: Verified probabilistic policies for deep reinforcement learning. In: NASA Formal Methods Symposium. pp. 193–212. Springer (2022)

3. Baier, C., Katoen, J.: Principles of model checking. MIT Press (2008)

4. Barr, E.T., Harman, M., McMinn, P., Shahbaz, M., Yoo, S.: The oracle problem in software testing: A survey. IEEE transactions on software engineering 41(5) (2014)

5. Belouadah, A., Ruiz-Rodríguez, M.L., Kubler, S., Le Traon, Y.: Evaluating the efectiveness of llms for explainable deep reinforcement learning. Machine Learning with Applications p. 100795 (2025)

6. Bilal, A., Ebert, D., Lin, B.: Llms for explainable AI: A comprehensive survey. CoRR abs/2504.00125 (2025)

7. Cao, Y., Zhao, H., Cheng, Y., Shu, T., Chen, Y., Liu, G., Liang, G., Zhao, J., Yan, J., Li, Y.: Survey on large language model-enhanced reinforcement learning: Concept, taxonomy, and methods. IEEE Trans. Neural Networks Learn. Syst. (2025)

8. Chen, Y., Singh, C., Liu, X., Zuo, S., Yu, B., He, H., Gao, J.: Towards consistent natural-language explanations via explanation-consistency finetuning. In: Proceedings of the 31st International Conference on Computational Linguistics. pp. 7558– 7568 (2025)

9. Freeman, L.C.: A set of measures of centrality based on betweenness. Sociometry pp. 35–41 (1977)

10. Gross, D.: Turn-based multi-agent reinforcement learning model checking. In: ICAART (3). pp. 980–987. SCITEPRESS (2023)

11. Gross, D.: Formally verifying and explaining sepsis treatment policies with cool-mc. arXiv preprint arXiv:2602.14505 (2026)

12. Gross, D., Jansen, N., Junges, S., Pérez, G.A.: COOL-MC: A comprehensive tool for reinforcement learning and model checking. In: SETTA. Lecture Notes in Computer Science, vol. 13649, pp. 41–49. Springer (2022)

13. Gross, D., Schmidl, C., Jansen, N., Pérez, G.A.: Model checking for adversarial multi-agent reinforcement learning with reactive defense methods. In: Proceedings of the International Conference on Automated Planning and Scheduling. vol. 33, pp. 162–170 (2023)

14. Gross, D., Spieker, H.: Enhancing rl safety with counterfactual llm reasoning. In: IFIP International Conference on Testing Software and Systems. pp. 23–29. Springer (2024)

15. Gross, D., Spieker, H.: Enhancing RL safety with counterfactual LLM reasoning. In: ICTSS. Lecture Notes in Computer Science, vol. 15383, pp. 23–29. Springer (2024)

16. Gross, D., Spieker, H.: Probabilistic model checking of stochastic reinforcement learning policies. arXiv preprint arXiv:2403.18725 (2024)

17. Gross, D., Spieker, H.: PCTL model checking for temporal RL policy safety explanations. In: SAC. pp. 1514–1521. ACM (2025)

18. Gross, D., Spieker, H., Gotlieb, A.: Verifying memoryless sequential decisionmaking of large language models. CoRR abs/2510.06756 (2025)

19. Hansson, H., Jonsson, B.: A logic for reasoning about time and reliability. Formal aspects of computing 6(5), 512–535 (1994)

20. Hensel, C., Junges, S., Katoen, J., Quatmann, T., Volk, M.: The probabilistic model checker Storm. Int. J. Softw. Tools Technol. Transf. 24(4), 589–610 (2022)

21. Ji, Z., Lee, N., Frieske, R., Yu, T., Su, D., Xu, Y., Ishii, E., Bang, Y., Madotto, A., Fung, P.: Survey of hallucination in natural language generation. ACM Comput. Surv. 55(12), 248:1–248:38 (2023)

22. Kim, H., Chen, H., Li, C., Lee, J.M.: Talktoagent: A human-centric explanation of reinforcement learning agents with large language models. CoRR abs/2509.04809 (2025)

23. Kroeger, N., Ley, D., Krishna, S., Agarwal, C., Lakkaraju, H.: Are large language models post hoc explainers? CoRR abs/2310.05797 (2023)

24. Lin, S., Hilton, J., Evans, O.: Truthfulqa: Measuring how models mimic human falsehoods. In: Proceedings of the 60th annual meeting of the association for computational linguistics (volume 1: long papers). pp. 3214–3252 (2022)

25. Lin, Z., Zhao, L., Yang, D., Qin, T., Liu, T., Yang, G.: Distributional reward decomposition for reinforcement learning. In: NeurIPS. pp. 6212–6221 (2019)

26. Lu, W., Zhao, X., Magg, S., Gromniak, M., Li, M., Wermter, S.: A closer look at reward decomposition for high-level robotic explanations. In: ICDL. pp. 429–436. IEEE (2023)

27. Mnih, V., Kavukcuoglu, K., Silver, D., Rusu, A.A., Veness, J., Bellemare, M.G., Graves, A., Riedmiller, M.A., Fidjeland, A., Ostrovski, G., Petersen, S., Beattie, C., Sadik, A., Antonoglou, I., King, H., Kumaran, D., Wierstra, D., Legg, S., Hassabis, D.: Human-level control through deep reinforcement learning. Nat. 518(7540), 529– 533 (2015)

28. Mohsin, M.A., Umer, M., Bilal, A., Memon, Z., Qadir, M.I., Bhattacharya, S., Rizwan, H., Gorle, A.R., Kazmi, M.Z., Mohsin, A., Rafique, M.U., He, Z., Mehta, P., Jamshed, M.A., Ciofi, J.M.: On the fundamental limits of llms at scale. CoRR abs/2511.12869 (2025)

29. Puterman, M.L.: Markov Decision Processes: Discrete Stochastic Dynamic Programming. Wiley Series in Probability and Statistics, Wiley (1994)

30. Qian, Y., Nguyen, S., Chen, C., Zhou, Q., Zhao, L.: Interpret policies in deep reinforcement learning using SILVER with rl-guided labeling: A model-level approach to high-dimensional and multi-action environments. CoRR abs/2510.19244 (2025)

31. Ribeiro, M.T., Wu, T., Guestrin, C., Singh, S.: Beyond accuracy: Behavioral testing of nlp models with checklist. In: Proceedings of the 58th annual meeting of the association for computational linguistics. pp. 4902–4912 (2020)

32. Rothermel, G., Untch, R.H., Chu, C., Harrold, M.J.: Prioritizing test cases for regression testing. IEEE Transactions on software engineering 27(10), 929–948 (2001)

33. Samadi, A., Koufos, K., Debattista, K., Dianati, M.: SAFE-RL: saliency-aware counterfactual explainer for deep reinforcement learning policies. IEEE Robotics Autom. Lett. 9(11), 9994–10001 (2024)

34. Saulières, L.: A survey of explainable reinforcement learning: Targets, methods and needs. CoRR abs/2507.12599 (2025)

35. Suntharamoorthy, N., Betten, J.E., Gross, D., Valseth, E., Spieker, H.: Do llms explain or remember? evaluating language model explanations of reinforcement

learning failures. In: TRUST-AI: The Second European Workshop on Trustworthy AI. IJCAI/ECAI 2026 (2026)

36. Team, G.: Gemma 3 technical report (2025), https://arxiv.org/abs/2503.19786

37. Team, G.: Gemma 4 technical report (2026), https://arxiv.org/abs/2607.02770

38. Team, Q.: Qwen3.5: Accelerating productivity with native multimodal agents (February 2026), https://qwen.ai/blog?id=qwen3.5

39. Valmeekam, K., Marquez, M., Hernandez, A.O., Sreedharan, S., Kambhampati, S.: Planbench: An extensible benchmark for evaluating large language models on planning and reasoning about change. In: NeurIPS (2023)

40. Vaswani, A., Shazeer, N., Parmar, N., Uszkoreit, J., Jones, L., Gomez, A.N., Kaiser, L., Polosukhin, I.: Attention is all you need. In: NIPS. pp. 5998–6008 (2017)

41. Weyuker, E.J.: On testing non-testable programs. The Computer Journal 25(4), 465–470 (1982)

42. Xi-Jia, Z., Guo, Y., Chen, S., Stepputtis, S., Gombolay, M.C., Sycara, K.P., Campbell, J.: Model-agnostic policy explanations with large language models. CoRR abs/2504.05625 (2025)

43. Yang, X., Zeng, L., Dong, H., Yu, C., Wu, X., Yang, H., Wang, Y., Tambe, M., Wang, T.: Policy-to-language: Train llms to explain decisions with flow-matching generated rewards. CoRR abs/2502.12530 (2025)

44. Zhang, J.M., Harman, M., Ma, L., Liu, Y.: Machine learning testing: Survey, landscapes and horizons. IEEE Transactions on Software Engineering 48(1), 1– 36 (2020)

45. Zhang, X., Guo, Y., Stepputtis, S., Sycara, K., Campbell, J.: Explaining agent behavior with large language models. arXiv preprint arXiv:2309.10346 (2023)