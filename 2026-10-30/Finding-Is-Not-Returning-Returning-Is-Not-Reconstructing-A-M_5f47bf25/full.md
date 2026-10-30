# Finding Is Not Returning, Returning Is Not Reconstructing: A Multi-Dimensional Evaluation of Agent Social Re-entry

SHENG WAN<sup>∗</sup>, The Hong Kong University of Science and Technology, China and Southern University of Science and Technology, China

CONG MA<sup>∗</sup>, Southern University of Science and Technology, China and City University of Hong Kong, China

SICHENG DONG, Shenzhen MSU-BIT University, China

QIAN’ANG MAO, Nanjing University, China

CHENGHAN ZHANG, University of Technology Sydney, Australia

DANING HU, Southern University of Science and Technology, China

QIANG YANG<sup>†</sup>, The Hong Kong Polytechnic University, China

As Large Language Model (LLM) agents increasingly participate in persistent social environments, understanding their long-term social interaction capabilities becomes increasingly important. While existing research has advanced long-term memory and conversational context retrieval, less attention has been paid to whether agents can autonomously re-engage with previously inactive conversations. In particular, remembering a past interaction does not necessarily mean that an agent can identify when to revisit it, decide to return, or reconstruct the context needed for a coherent continuation. To investigate this gap, we introduce agent social re-entry as a multidimensional evaluation problem, distinguishing three complementary capabilities: Finding, Returning, and Reconstructing. We develop a shared-snapshot evaluation framework that measures these dimensions independently, allowing us to examine how conversational resurfacing and persistent memory relate to diferent aspects of social re-entry. Evaluations across six LLM configurations show that resurfacing substantially improves target conversation identification, while persistent memory is associated with higher rates of autonomous target-directed replying. However, neither improvement is accompanied by consistent gains in contextual reconstruction. These results highlight the distinction between accessing previous conversations, choosing to re-engage, and maintaining conversational continuity. Our framework provides a way to evaluate these capabilities separately rather than relying on memory retrieval or conversational response quality alone

CCS Concepts: • Computing methodologies → Natural language generation; • Human-centered computing → Laboratory experiments.

Additional Key Words and Phrases: LLM agents, social continuity, conversation re-entry, persistent memory, attention, behavioral evaluation

<sup>∗</sup>Both authors contributed equally to this research.   
<sup>†</sup>Corresponding author.

Authors’ Contact Information: Sheng Wan, swanae@connect.ust.hk, The Hong Kong University of Science and Technology, Hong Kong, China and Southern University of Science and Technology, Shenzhen, China; Cong Ma, Southern University of Science and Technology, Shenzhen, China and City University of Hong Kong, Hong Kong, China; Sicheng Dong, Shenzhen MSU-BIT University, Shenzhen, China; Qian’ang Mao, Nanjing University, Nanjing, China; Chenghan Zhang, University of Technology Sydney, Sydney, Australia; Daning Hu, Southern University of Science and Technology Shenzhen, China; Qiang Yang, profqiang.yang@polyu.edu.hk, The Hong Kong Polytechnic University, Hong Kong, China.

Permission to make digital or hard copies of all or part of this work for personal or classroom use is granted without fee provided that copies are not made or distributed for profit or commercial advantage and that copies bear this notice and the full citation on the first page. Copyrights for components of this work owned by others than the author(s) must be honored. Abstracting with credit is permitted. To copy otherwise, or republish, to post on servers or to redistribute to lists, requires prior specific permission and/or a fee. Request permissions from permissions@acm.org.

## ACM Reference Format:

Sheng Wan, Cong Ma, Sicheng Dong, Qian’ang Mao, Chenghan Zhang, Daning Hu, and Qiang Yang. 2026. Finding Is Not Returning, Returning Is Not Reconstructing: A Multi-Dimensional Evaluation of Agent Social Re-entry. ACM Trans. Intell. Syst. Technol. 1, 1 (October 2026), 17 pages. https://doi.org/10.1145/nnnnnnn.nnnnnnn

## 1 Introduction

In online communities, participants often revisit earlier conversations after hours or days of inactivity. This behavior, known as conversational re-entry, has been studied in human online interactions [15]. With LLM agents increasingly operating in persistent social environments [11, 14, 16], conversational re-entry also becomes relevant to autonomous agents. While prior research has investigated whether human participants will return to earlier discussions, an important question arises for LLM agents: Can they autonomously re-enter previous social conversations while maintaining contextual continuity?

Recent advances in long-context modeling, retrieval-augmented generation, and agent memory have improved the ability of language models to retain and access previous information [5, 7]. However, these advances primarily address the availability and use of historical information, whereas social re-entry also requires an agent to decide when and where to resume a previous interaction and to maintain continuity with that interaction. Consequently, evaluating retrieval accuracy or conversational memory alone is insuficient to determine whether an agent can successfully re-engage with previous social interactions.

We formulate agent social re-entry as a multi-dimensional evaluation problem involving three conceptually connected stages: Finding, Returning, and Reconstructing. Finding concerns identifying a relevant prior conversation. Returning concerns autonomously re-engaging with that conversation rather than merely recognizing it. Reconstructing concerns recovering the relevant conversational context to produce a coherent continuation. These stages form a natural conceptual progression, but success at one stage does not guarantee success at the next. An agent may find a previous conversation without returning to it, or return without successfully reconstructing its context. Related research on conversational grounding, interruption and resumption, and prospective memory provides useful perspectives on these challenges [2, 3, 9, 10]. Thus, Finding does not guarantee Returning, and Returning does not guarantee Reconstructing.

To systematically evaluate these stages, we propose a controlled evaluation framework for agent social re-entry. Rather than relying solely on end-to-end observations, the framework evaluates each stage through a separate experimental branch initialized from a common world-state snapshot. Finding is measured by selecting a target conversation among ten candidate threads, Returning by whether an agent produces an unprompted, target-directed reply, and Reconstructing by the contextual continuity of a prompted continuation. This design allows the stages to be evaluated separately without actions in one branch influencing measurements in another. We further investigate persistent memory and prioritized resurfacing through a 2 × 2 experimental design, supplemented by distractor-memory and oracle controls. The framework is evaluated across six service/model configurations under a common experimental protocol.

Our results show distinct patterns across the three stages. Prioritized resurfacing substantially improves Finding, while the persistent-memory configuration is associated with higher Returning rates. However, corresponding improvements in Reconstructing are not observed. These patterns are consistent across the six evaluated configurations, although the reconstruction diferences vary in magnitude and statistical significance. Together, these findings highlight the importance of evaluating Finding, Returning, and Reconstructing separately rather than treating them as a single measure of social re-entry.

Manuscript submitted to ACM

This work makes three main contributions. First, we formulate agent social re-entry as a multi-dimensional evaluation problem, distinguishing Finding, Returning, and Reconstructing as conceptually connected but separately measurable capabilities. Second, we propose a shared-snapshot evaluation framework that independently measures these dimensions and supports controlled interventions on persistent memory and conversational resurfacing. Third, through experiments across six service/model configurations, we find that successful Finding does not necessarily imply Returning, and increased Returning does not guarantee stronger Reconstructing. These findings highlight the distinction between the three dimensions and the importance of evaluating them separately.

## 2 Related Work

Conversational re-entry and social continuity. Conversational re-entry has been studied in human online communities, where the central task is to predict whether a participant will return to a conversation in which they previously participated. Wang et al. [15] investigate this problem by jointly modeling conversation context and user interaction history. Related research on conversational grounding emphasizes the establishment and maintenance of shared understanding [2], while studies of interruptions and prospective memory examine the challenges of resuming previous activities and acting on delayed intentions [3, 9, 10].

Recent studies have further examined conversational persistence and social interaction in agent-oriented environments. Eziz [4] analyzes reply dynamics and conversational persistence in Moltbook, highlighting the limited extent of sustained interactions. Zerhoudi et al. [18] investigate agent social behavior in Moltbook, reporting shallow conversational structures and infrequent returns by original post authors. These studies characterize observed interaction patterns in agent social networks, while our work evaluates agent social re-entry through three separately measurable capabilities: Finding, Returning, and Reconstructing.

Memory and context in LLM agents. Transformer-based language models generate responses conditioned on their input context [13], while retrieval-augmented generation incorporates externally retrieved information [5]. Persistent socialagent architectures further support the storage and retrieval of observations, reflections, and previous interactions [11]. However, access to additional context does not necessarily ensure its efective use, as demonstrated by the sensitivity of long-context models to the placement of relevant information [7]. Recent benchmarks such as LoCoMo [8] evaluate long-term conversational memory through question answering, event summarization, and dialogue generation. These approaches provide important foundations for accessing and using conversational history. Our work examines a complementary question: whether mechanisms that make earlier conversations accessible also afect an agent’s autonomous decision to return, and whether returning is accompanied by successful contextual reconstruction.

Evaluation ofautonomous LLM agents. Research on LLM agents has investigated reasoning, planning, and action, including the use of external tools [17]. Surveys identify memory, planning, and action as central components of autonomous-agent systems [14, 16], while general evaluation frameworks emphasize the importance of assessing multiple aspects of model behavior [6]. More specifically, SocialBench [1] evaluates the social interaction capabilities of role-playing agents at individual and group levels. ChronosBench [12] investigates proactive long-term task-oriented interaction, including intent-conditioned monitoring and event-triggered follow-up. These studies extend agent evaluation beyond conventional task execution, but address diferent settings from resuming previously inactive social conversations. Our work focuses on agent social re-entry as a multi-stage process, distinguishing the ability to identify a relevant prior conversation, autonomously return to it, and reconstruct its context. By evaluating these stages through separate branches initialized from a common world-state snapshot, we examine their responses to memory and resurfacing interventions without assuming that success in one stage establishes successful end-to-end re-entry.

![](images/3ed6dff3c0fceacfb917c5b6bff0b1f4882b7ed709823ad396c35142fc25ee74.jpg)  
Fig. 1. Overview of the agent social re-entry evaluation framework. (a) An initial interaction establishes a target conversation, followed by unrelated conversational activity during an interruption period. The resulting environment state is frozen before observation. (b) Three independent evaluation branches are initialized from copies of the same observation state to assess Finding, Returning, and Reconstructing under their respective task setings. The branches are evaluated separately rather than as a single sequential process.

## 3 Method

## 3.1 Problem Formulation

We study agent social re-entry, which refers to the process by which an LLM agent identifies and autonomously returns to a previously inactive social conversation. Successful contextual continuation further requires the agent to reconstruct the relevant conversational history and produce a response that maintains continuity with the earlier interaction. Unlike conventional conversational memory evaluations, which assess whether models can retrieve and use information from previous interactions through explicitly specified tasks [8], agent social re-entry additionally involves the decision to resume a previous interaction and the ability to continue it coherently.

Consider a social environment in which an agent � has previously participated in a conversation � with another participant. Following a period of inactivity, the agent encounters a set of available conversations $\mathcal { T } = \{ T _ { 1 } , \ldots , T _ { N } \}$ , where �<sup>∗</sup> ∈ T denotes the target conversation and the remaining threads represent competing conversational opportunities. The agent may have access to information about its earlier interactions, depending on the available conversationa context and memory configuration. The central question is whether the agent can recognize the relevant prior conversation, choose to return to it without an explicit target-directed instruction, and maintain contextual continuity when continuing the discussion.

We conceptualize this process through three stages: Finding, Returning, and Reconstructing. Finding concerns identifying the target conversation among available threads. Returning concerns autonomously re-engaging with the target conversation rather than merely recognizing it. Reconstructing concerns recovering the relevant conversational context and producing a coherent continuation. These stages form a natural conceptual progression: an agent must identify a conversation before deliberately returning to it, while a successful return does not necessarily imply that its continuation preserves the earlier context.

Importantly, the three stages describe distinct observable capabilities rather than independent internal cognitive mechanisms. Their conceptual relationship does not imply that they must be measured through a single sequential trajectory. Instead, we develop an evaluation framework that examines each stage separately, allowing us to distinguish target identification, autonomous returning, and contextual reconstruction without conflating their outcomes.

## 3.2 Social Re-entry Evaluation Framework

We introduce a controlled evaluation framework that represents agent social re-entry as an interrupted interaction followed by an opportunity for renewed engagement. As illustrated in Figure 1, the framework consists of two components: (i) a social re-entry scenario involving an initial interaction, an interruption period, and a shared observation state (Figure 1(a)); and (ii) three independent evaluation branches initialized from that state to assess Finding, Returning, and Reconstructing (Figure 1(b)).

Initial interaction and interruption. The environment first establishes an interaction between the agent and another participant in a target conversation �<sup>∗</sup>. This interaction provides the conversational history that the agent may later need to recognize and reconstruct. The original interaction is followed by unrelated social activity, during which the target conversation remains inactive. The interruption separates the initial interaction from the subsequent opportunity for re-entry and introduces competing conversational context

Shared observation state. Following the interruption, the environment reaches an observation state �, which contains the target conversation and other available conversations, as illustrated in Figure 1(a). The state is frozen immediately before observation and serves as the common starting point for the three evaluation branches. Importantly, the shared state does not imply identical inputs across branches; the information and conversational cues presented to the agent depend on the corresponding evaluation task and experimental conditions.

Independent evaluation branches. Rather than evaluating all three stages through a single sequence of agent actions, we initialize three independent branches from the same observation state:

• Finding branch: evaluates whether the agent can identify the target conversation when explicitly asked to select a relevant prior thread

• Returning branch: evaluates whether the agent spontaneously produces a target-directed reply while interacting with the social environment, without being explicitly instructed to return to the target conversation.

• Reconstructing branch: evaluates the contextual continuity of a response when the agent is explicitly prompted to continue the target conversation.

Each branch operates on an independent copy of the observation state. Consequently, actions performed in one branch do not alter the environment or conversational history used by another branch. This separation prevents an agent’s behavior in one evaluation task from influencing the measurements of the others.

The framework distinguishes a conceptual progression from an experimental sequence. Finding, Returning, and Reconstructing represent connected aspects of successful social re-entry, but their measured outcomes are obtained independently. Therefore, the framework evaluates stage-specific capabilities rather than directly estimating end-to-end social re-entry success.

## 3.3 Three Dimensions of Social Re-entry

As illustrated in Figure 1(b), we operationalize Finding, Returning, and Reconstructing through three independent evaluation tasks, each with a distinct output and evaluation outcome. Let $T ^ { * }$ denote the target conversation, $\mathcal { T }$ the set of candidate conversations, and � the shared observation state. The following definitions specify what each dimension measures.

Finding: target identification. Finding evaluates whether an agent can recognize the target conversation among available candidates. Given the candidate set $\mathcal { T } ,$ the agent is explicitly asked to select a conversation or abstain. Let $\hat { T }$ denote the selected conversation. The Finding outcome is defined as

$$
F = \mathbb { I } [ \hat { T } = T ^ { * } ] ,\tag{1}
$$

where I[·] is the indicator function. A successful Finding outcome indicates that the agent selects the target conversation. Because this task explicitly requests a selection, it measures target identification under instruction rather than the agent’s spontaneous decision to revisit the conversation.

Returning: autonomous re-engagement. Returning evaluates whether the agent independently chooses to resume the target conversation during an observation opportunity. Unlike Finding, the agent is not explicitly instructed to identify or reply to $T ^ { * }$ . Instead, it acts within the social environment, where multiple conversational opportunities are available.

Let � denote the agent’s observed action, and let target(�) and text(�) denote the destination thread and textual content of a reply action, respectively. We define the Returning outcome as

$$
R = \mathbb { I } \left[ { \mathrm { t y p e } } ( a ) = { \mathrm { r e p l y ~ } } \wedge { \mathrm { ~ t a r g e t } } ( a ) = T ^ { * } \ \wedge \ | { \mathrm { ~ t e x t } } ( a ) | > 0 \right] .\tag{2}
$$

A successful Returning outcome indicates that the agent produces a non-empty, target-directed reply without an explicit instruction to resume that particular conversation. This measure captures observable autonomous reengagement, but does not establish whether the response accurately reconstructs the earlier discussion.

Reconstructing: contextual continuity. Reconstructing evaluates whether an agent can produce a continuation that reflects the relevant history of the target conversation. In this task, the agent is explicitly prompted to continue $T ^ { * }$ allowing contextual reconstruction to be assessed independently of its spontaneous decision to return.

Let $H ^ { * }$ denote the relevant history of the target conversation and � the generated continuation. We define the Reconstructing outcome as

$$
Q = J ( y , H ^ { * } ) , \qquad Q \in \{ 0 , 1 , 2 , 3 \} ,\tag{3}
$$

where � denotes an LLM-based evaluation procedure that assigns an ordinal contextual continuity score. Higher scores indicate stronger continuity with the previous interaction, ranging from no meaningful continuity to a specific and natural continuation of the discussion.

Unlike Finding and Returning, which are measured as binary outcomes, Reconstructing is evaluated through an ordinal quality score. The three measures therefore capture diferent properties of agent behavior and should not be interpreted as interchangeable indicators of a single underlying capability.

Together, these measures characterize target identification, autonomous re-engagement, and contextual continuity as distinct observable outcomes

## 4 Experimental Design

## 4.1 Social Interaction Scenarios and Data Construction

We construct a benchmark consisting of 30 frozen synthetic conversation scenarios through Microsoft Copilot, shared across experimental conditions and evaluated models. The scenarios cover four conversational categories: Technical/Problem-Solving (8), Everyday/Lifestyle (8), Opinion/Discussion (7), and Entertainment/Interest (7).

Each stimulus follows the social re-entry scenario illustrated in Figure 1(a). An initial interaction (R1) establishes a target conversation between the agent and another participant. This is followed by five rounds of unrelated social activity, during which the target conversation remains inactive. The environment then reaches the observation state used for evaluation.

At observation, the environment contains ten candidate conversations, consisting of the original target conversation and nine distractor threads. The interruption length and number of candidate conversations are fixed across stimuli. All evaluated model calls use a temperature of zero.

The same frozen stimuli are reused across the experimental configurations, providing a common basis for comparing the three evaluation dimensions and the efects of diferent conditions.

## 4.2 Experimental Conditions and Controls

Our experimental design considers two factors: persistent social memory and conversational resurfacing. The memory factor compares two configurations, denoted as A0 and A2. A0 omits persistent social history, although the agent may still receive task-specific information about the target conversation. In particular, the reconstruction prompt retains the target title, topic, and latest reply. A2 additionally provides recent messages from previously posted threads and the full history of the target conversation.

The resurfacing factor compares B0 and B3. Under B0, no notification is provided, and the target conversation remains in its original position among the candidates. Under B3, the agent receives a notification concerning the target conversation, which is also promoted to the first position. Thus, B3 represents a combined resurfacing intervention involving both notification and candidate ordering.

These conditions are applied diferently across the three evaluation branches, as detailed in Section 4.3.

We include additional control conditions to examine the interpretation of the main experimental comparisons.

For Returning, the distractor-memory condition provides a memory block constructed from distractor threads under B0. This condition allows an exploratory comparison between the A0 baseline, the social history provided under A2 (including the target conversation), and a memory block containing unrelated conversational information.

The distractor-memory blocks are not strictly matched to A2. Their lengths difer from the corresponding A2 blocks by 15–22%, exceeding the intended 5% tolerance. They also difer in headers, thread composition, role and message distributions, and surrounding context. Accordingly, this condition is treated as an exploratory comparison rather than a controlled test isolating the contribution of target-relevant memory content.

For Reconstructing, we report an Oracle control that provides the full target-conversation context. This condition ofers an additional reference for interpreting continuity scores but is not treated as a calibrated performance ceiling.

## 4.3 Evaluation Protocol

Following the independent-branch design in Figure 1(b), we evaluate Finding, Returning, and Reconstructing through three separate tasks. Although the branches originate from the same observation state, the information and instructions presented to the agent difer according to the evaluation objective and experimental conditions.

Finding. The Finding branch evaluates target identification under B0 and B3, with the memory configuration fixed to A0. The agent is presented with candidate conversations and explicitly instructed to select a relevant prior conversation or abstain. A successful outcome occurs when the selected conversation matches the predefined target. This comparison examines whether prioritized resurfacing improves target selection when persistent social history is absent.

Returning. The Returning branch evaluates autonomous target-directed replying under four configurations: A0B0, A0B3, A2B0, and A2B3. The agent interacts with the observation environment without an explicit instruction to return to the target conversation. Its observed action is recorded and evaluated according to the Returning criterion defined in Section 3.3. A successful outcome requires a non-empty reply directed to the target conversation. By comparing the four configurations, we examine how persistent memory and resurfacing conditions relate to autonomous target-directed replying.

Reconstructing. The Reconstructing branch evaluates contextual continuity under A0 and A2. Unlike Finding and Returning, this branch does not present a candidate feed or notification. Instead, the agent is explicitly prompted to continue the target conversation using the information available under the assigned memory configuration. The generated continuation is evaluated using an LLM-based evaluation procedure with Copilot, which assigns an ordina contextual continuity score from 0 to 3. A score of 0 indicates a generic, unrelated, or context-unaware reply; 1 indicates topical acknowledgment without a specific prior detail; 2 indicates a continuation using at least one specific prior detail; and 3 indicates a natural extension that integrates specific history and appropriate conversational tone.

The three branches are executed independently. Consequently, an agent’s selection or reply in one branch does not afect the inputs or outcomes of another branch. The resulting measurements characterize the three evaluation dimensions separately rather than representing a single end-to-end execution of social re-entry.

## 4.4 Models, Implementation, and Statistical Analysis

We evaluate six service/model configurations: DeepSeek-V4-Flash, GLM-4.6V-FlashX, GLM-5.1-Cloud, MiniMax-M2.7- Cloud, Qwen3.5-Cloud, and GPT-5.5-OpenModel. All six configurations are evaluated using the same frozen stimuli, experimental protocol, and outcome definitions, ensuring consistent comparisons across models. We report model identifiers as provided by their respective services. Experimental outcomes are recorded by model, stimulus, evaluation branch, and condition.

All experimental conditions were designed around the same 30 frozen stimuli, which serve as the intended units for paired comparisons. Due to incomplete or invalid API calls, the number of available outcomes varies across models and experimental conditions. The condition-specific denominators used in the analysis are reported in Table 1, ranging from 20 to 30 per condition. Statistical comparisons are conducted using the available outcomes, with paired analyses restricted to stimuli for which both outcomes are available

Manuscript submitted to ACM

Table 1. Condition denominators by model. Values show the number of stimuli with available outcomes per condition. Missingness arises from incomplete or invalid API calls; Qwen3.5 experienced the highest rate of incomplete calls.
<table><tr><td>Model</td><td>A0B0</td><td>A0B3</td><td>A2B0</td><td>A2B3</td><td>Distractor</td><td>Disc B0</td><td>Disc B3</td><td>Recon A0</td><td>Recon A2</td><td>Oracle</td></tr><tr><td>DeepSeek</td><td>30</td><td>28</td><td>27</td><td>29</td><td>30</td><td>30</td><td>29</td><td>28</td><td>28</td><td>30</td></tr><tr><td>GLM-4.6V</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>29</td><td>30</td></tr><tr><td>GLM-5.1</td><td>30</td><td>28</td><td>27</td><td>28</td><td>29</td><td>29</td><td>27</td><td>29</td><td>28</td><td>30</td></tr><tr><td>MiniMax</td><td>26</td><td>26</td><td>26</td><td>26</td><td>28</td><td>28</td><td>27</td><td>26</td><td>26</td><td>30</td></tr><tr><td>Qwen3.5</td><td>20</td><td>20</td><td>23</td><td>24</td><td>23</td><td>23</td><td>20</td><td>22</td><td>20</td><td>23</td></tr><tr><td>GPT-5.5</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>30</td><td>28</td></tr></table>

We analyze the experimental outcomes using descriptive efect sizes and within-model statistical comparisons. Finding and Returning are evaluated as binary outcomes, while Reconstructing is evaluated using ordinal contextual continuity scores.

For Finding, we compare target identification under B0 and B3. For Returning, the main comparison examines A0B0 versus A2B0 to assess the relationship between persistent-memory configurations and autonomous target-directed replying without resurfacing. We also examine the remaining combinations of memory and resurfacing conditions. For Reconstructing, we compare the contextual continuity scores obtained under A0 and A2.

For binary outcomes, we use two-sided exact McNemar tests on paired stimuli and report Fisher exact tests as an independent-sample reference. Because each primary contrast is evaluated across six model configurations, we apply Holm–Bonferroni correction within each family of six tests. Efect sizes are reported as diferences in observed success rates, expressed in percentage points.

For Reconstructing, we report descriptive continuity scores and conduct exploratory paired �-tests and Wilcoxon signed-rank tests using stimulus-level scores, with Holm–Bonferroni correction across the six evaluated configurations Because contextual continuity is measured on an ordinal scale, the reported mean scores are interpreted descriptively.

Throughout the analysis, statistical comparisons are used to characterize observable diferences between experimental conditions. They do not, by themselves, establish the internal mechanisms responsible for an agent’s behavior. In particular, a non-significant diference is not interpreted as evidence of equivalence.

## 5 Experimental Results

We present the experimental results across six evaluated service/model configurations, following the three dimensions of agent social re-entry: Finding, Returning, and Reconstructing. We first examine how prioritized resurfacing afects target identification, followed by the efects of persistent-memory configurations on autonomous target-directed replying. We then report the exploratory distractor-memory comparison and contextual reconstruction results, before summarizing the patterns across models.

## 5.1 Finding: Efects of Conversational Resurfacing

We first examine whether prioritized resurfacing improves the agent’s ability to identify a previously inactive target conversation. Under B0, Finding success rates ranged from 13.3% to 53.3% across the six evaluated configurations. Under B3, all six configurations achieved 100% Finding success among the available evaluations. The corresponding risk diferences ranged from +46.7 to +86.7 percentage points (Table 2; Figure 2).

Table 2. Six-system summary of the $2 \times 2$ behavioral design and the two probe contrasts. Binary behavioral cells show successes/available outcomes; reconstruction cells show mean continuity with available �. Risk Diference (RD) is expressed in percentage points.
<table><tr><td rowspan="2">System</td><td colspan="4">Target-directed reply (2 × 2)</td><td colspan="4">Discovery</td><td colspan="4">Reconstruction</td></tr><tr><td>A0B0</td><td>A0B3</td><td>A2B0</td><td>A2B3</td><td></td><td>B0</td><td>B3</td><td>RD</td><td>A0</td><td></td><td></td><td>A2 A0-A2</td></tr><tr><td>DeepSeek-V4</td><td>0/30</td><td>2/28</td><td>12/27</td><td>15/29</td><td>4/30</td><td>29/29</td><td></td><td>+86.7</td><td>2.18 (28)</td><td></td><td>1.79 (28)</td><td>+0.39</td></tr><tr><td>GLM-4.6V</td><td>1/30</td><td>0/30</td><td>16/30</td><td>18/30</td><td>16/30</td><td></td><td>30/30</td><td>+46.7</td><td>2.00 (30)</td><td></td><td>1.69 (29)</td><td>+0.31</td></tr><tr><td>GLM-5.1</td><td>2/30</td><td>6/28</td><td>19/27</td><td>16/28</td><td>8/29</td><td></td><td>27/27</td><td>+72.4</td><td>2.24 (29)</td><td></td><td>1.89 (28)</td><td>+0.35</td></tr><tr><td>MiniMax-M2.7</td><td>2/26</td><td>4/26</td><td>11/26</td><td>13/26</td><td>11/28</td><td></td><td>27/27</td><td>+60.7</td><td>1.77 (26)</td><td></td><td>1.27 (26)</td><td>+0.50</td></tr><tr><td>Qwen3.5</td><td>0/20</td><td>1/20</td><td>10/23</td><td>10/24</td><td>4/23</td><td></td><td>20/20</td><td>+82.6</td><td>2.05 (22)</td><td></td><td>1.50 (20)</td><td>+0.55</td></tr><tr><td>GPT-5.5</td><td>0/30</td><td>5/30</td><td>9/30</td><td>9/30</td><td>13/30</td><td></td><td>30/30</td><td>+56.7</td><td>1.90 (30)</td><td></td><td>1.83 (30)</td><td>+0.07</td></tr></table>

Finding: Discovery Rate

![](images/d8228dfd7c63c6f74839306fd7e5f82b2d6ef0849df388c73072c180bfa68fb0.jpg)

![](images/9a4792c4954261a8425c8776b04417d6c645d06ea75c26e868462fbd4f62a31c.jpg)  
Fig. 2. Finding versus returning across six models. Left: Discovery rates under B0 (no cue) and B3 (cue). B3 reaches 100% in every system. Right: Target-directed reply rates under A0B0 (no memory) and A2B0 (memory). Memory is associated with large increases, but the absolute rates remain far below the 100% discovery ceiling under B3.

The improvements were statistically significant across all six configurations. Two-sided Fisher exact tests yielded �-values ranging from $1 . 6 8 \times 1 0 ^ { - 5 } \mathrm { t o } 7 . 8 4 \times 1 0 ^ { - 1 3 }$ . Exact McNemar tests on 20–30 common stimuli per configuration also showed significant diferences, with all six comparisons remaining significant after Holm–Bonferroni correction (adjusted �-values from $3 . 5 8 \times 1 0 ^ { - 7 }$ to $1 . 2 2 \times 1 0 ^ { - 4 } )$ . Across all matched pairs, no B0 success was paired with a B3 failure.

These results show that the B3 resurfacing intervention substantially increased target identification across the evaluated configurations. As B3 combines notification with first-position promotion, the observed improvement reflects the combined intervention rather than the independent contribution of either component.

However, successful target identification did not translate into target-directed replying at a comparable rate. Under A0B3, Returning success rates ranged from 0% to 21.4%, compared with 100% Finding success under B3. Specifically, the A0B3 reply rates were 7.1%, 0%, 21.4%, 15.4%, 5.0%, and 16.7% for DeepSeek, GLM-4.6V, GLM-5.1, MiniMax, Qwen, and GPT, respectively. This contrast illustrates the distinction between explicitly identifying a previous conversation and autonomously producing a reply to it. Since the two outcomes are measured through independent tasks, the comparison characterizes a diference between evaluation outcomes rather than a sequential failure within the same execution.

Manuscript submitted to ACM

Behavioral Re-entry: 2×2 Design Across Six Models  
![](images/c62d5f189fbc5c22252c5d3b0c70eaf5bde8c9b5b85a3cc242ea5c5f4dca2387.jpg)  
Fig. 3. Full 2 × 2 behavioral design across six models. Each group of four bars shows A0B0, A0B3, A2B0, and A2B3 target-directed reply rates. Memory (A2) shifts rates upward; resurfacing (B3) produces a smaller and more heterogeneous additional shift.

## 5.2 Returning: Efects of Persistent Memory

We next examine how persistent-memory configurations relate to autonomous target-directed replying. Without resurfacing, the A0B0 reply rates ranged from 0.0% to 7.7%, whereas the A2B0 rates ranged from 30.0% to 70.4% across the six evaluated configurations. The corresponding risk diferences were positive for every configuration, ranging from +30.0 to +63.7 percentage points (Table 2; Figure 2).

Two-sided Fisher exact tests indicated significant diferences for all six configurations $( p = . 0 0 8 7 3 \mathrm { t o } 8 . 5 3 \times 1 0 ^ { - 7 } )$ Exact McNemar tests on common stimuli likewise detected significant paired diferences. After Holm–Bonferroni correction, all six comparisons remained significant, with adjusted �-values of .0020 for DeepSeek, .0003 for GLM-4.6V, .0001 for GLM-5.1, .031 for MiniMax, .031 for Qwen, and .012 for GPT.

The full 2 × 2 results further show that the efects of resurfacing on target-directed replying were smaller and more heterogeneous than the A0-to-A2 diferences (Figure 3). Under A0B3, reply rates remained between 0% and 21.4%, while both A2 configurations generally produced substantially higher rates than their A0 counterparts.

Together, these findings indicate a consistent association between the A2 memory configuration and increased autonomous target-directed replying. These diferences characterize the observed association with the implemented memory configuration, rather than establishing a specific internal memory mechanism.

## 5.3 Exploratory Distractor-Memory Comparison

To further examine the Returning results, we compare the distractor-memory condition with the A0B0 baseline and the A2B0 configuration.

Distractor-Memory vs Real Memory: Placebo Comparison  
![](images/e223c9c808ab98cd6530e0828859368a2dcdee25d897d2a5d6f5c132aa0117e7.jpg)  
Fig. 4. Distractor-memory versus real memory. Here, real memory refers to the A2B0 configuration, which includes the target conversation history, whereas distractor-memory contains unrelated conversational history. The distractor-memory condition (orange) exceeds A0B0 (red) in every system but varies around A2B0 (blue), lying above it in four systems and below it in two.

Distractor-memory reply rates ranged from 33.3% to 60.7% and exceeded A0B0 rates in all six evaluated configurations. Exact McNemar comparisons between distractor memory and A0B0 were significant in every configuration, with all comparisons remaining significant after Holm–Bonferroni correction (adjusted �-values from .0007 to .008).

In contrast, comparisons between distractor memory and A2B0 showed no consistent direction (Figure 4). Distractormemory minus A2B0 risk diferences ranged from −18.6 percentage points for GLM-5.1 to +18.4 percentage points for MiniMax. Distractor-memory reply rates exceeded A2B0 in four configurations and were lower in two. Exact McNemar tests did not detect a significant distractor-memory versus A2B0 diference in any configuration after Holm correction (all adjusted � ≥ .18).

These findings show that the distractor-memory configuration was also associated with higher target-directed reply rates than A0B0. However, the absence ofstatistically significant diferences from A2B0 does not establish equivalence. In addition, the distractor-memory and A2 configurations were not strictly matched in length and structure. Consequently, this exploratory comparison does not establish whether memory-block presence or target-relevant memory content is responsible for the observed behavioral diferences.

## 5.4 Reconstructing: Contextual Continuity

We finally examine whether the A2 memory configuration improves the contextual continuity of prompted conversation continuation. Across all six evaluated configurations, the mean Reconstructing scores were lower under A2 than under A0 (Table 2).

When calculated on common scored stimuli, the paired mean diferences (A0 minus A2) were 0.46 for DeepSeek, 0.35 for GLM-4.6V, 0.26 for GLM-5.1, 0.48 for MiniMax, 0.55 for Qwen, and 0.07 for GPT. The corresponding marginal mean diferences reported in Table 2 vary slightly because the numbers of available observations difer across conditions.

Manuscript submitted to ACM

Reconstruction Score Distributions: A0 vs A2  
![](images/bdbbdc8af3a87233fa4507429bd66840815c4e0e564be153acdff9a4ea10754e.jpg)

![](images/e4cf6ab996b12e1f1f9ad4e7feee973aa99b846dbedb077abe78c672bfaf7ae5.jpg)

![](images/199d9d51bd157b8b94c05d26cc08acbd4bd43c3ad8bbd3ec5a85c9faf27ed243.jpg)

![](images/969204d5a395b35a74cd18f6a5a15610f0d285e0a51e4e19ae98f71c367bf672.jpg)

![](images/433703e84b945b46de00d19dfdc9862940ad2b950d45c3c4aa01d4b02f30d242.jpg)

![](images/c795c54bed8bc33ab7f0bfca0860de9b3fdac2dc43eaa8d57d3982288f4a3ea4.jpg)  
Fig. 5. Reconstruction score distributions (0–3 ordinal) under A0 (red) and A2 (blue) for each of the six models. Mean continuity scores are higher under A0 across all six configurations, with larger observed diferences for MiniMax and Qwen3.5.

Exploratory paired comparisons showed heterogeneous statistical support. Only Qwen3.5 retained significance after Holm–Bonferroni correction (adjusted � = .046). MiniMax showed the next strongest evidence (unadjusted $p = . 0 2 5 ;$ adjusted $p = . 1 2 6 )$ , while the remaining four configurations did not exhibit statistically significant diferences after correction.

The score distributions also show a directional shift toward higher continuity scores under A0, particularly for Mini-Max and Qwen3.5 (Figure 5). The percentages are calculated using the available outcomes for each model and condition, with denominators reported in Table 1. Nevertheless, the magnitude of the mean diferences varied substantially across configurations, and the GPT-5.5 diference was only 0.07 points.

The reconstruction controls provide additional context. Oracle mean scores ranged from 1.59 to 2.03 and did not consistently exceed the corresponding A0 scores. Thus, the Oracle condition did not establish a uniform empirical performance ceiling under the implemented evaluation settings.

Overall, the results do not provide consistent evidence that A2 improves contextual continuity. Although the observed mean diferences favor A0 across all six configurations, the statistical evidence does not support a uniformly negative efect of A2. Importantly, the higher target-directed reply rates associated with A2 in the Returning branch were not accompanied by corresponding improvements in the separately evaluated Reconstructing branch.

![](images/f93a46536d375c15ecca6a34bb869b223b8e3f66402b1a27f442630f4153be06.jpg)

![](images/966c8db6ecbdf0f4bee0202f8e564e1a7bb3ee8e81bbc8816a1f5b65eceb7009.jpg)  
Fig. 6. Risk diferences across six models. Left: A2B0 minus A0B0 for target-directed replies. Right: B3 minus B0 for discovery. Al efects are positive; magnitudes vary but no model reverses either direction.

## 5.5 Cross-Model Consistency

Across the six evaluated service/model configurations, the two primary binary outcomes exhibit consistent directiona patterns. The B3 resurfacing intervention increased Finding success rates relative to B0 in every configuration, with risk diferences ranging from +46.7 to +86.7 percentage points. Similarly, A2B0 produced higher target-directed reply rates than A0B0 in every configuration, with risk diferences ranging from +30.0 to +63.7 percentage points (Figure 6). Both sets of paired comparisons remained statistically significant after Holm–Bonferroni correction within their respective families of six tests.

The Reconstructing results show a diferent pattern. Although mean contextual continuity scores were lower under A2 in every evaluated configuration, the diferences varied in magnitude, and only one configuration retained statistical significance after correction. The distractor-memory comparison was also less consistent: it exceeded A0B0 in all six configurations but varied in direction relative to A2B0.

Across the evaluated systems, resurfacing consistently improved Finding, and the A2 memory configuration was associated with higher Returning rates. However, comparable improvements in Reconstructing were not observed. These results highlight the importance of evaluating Finding, Returning, and Reconstructing separately, rather than treating success in one dimension as evidence of success in another.

## 6 Discussion

## 6.1 Finding, Returning, and Reconstructing as Separate Evaluation Targets

Our results show that Finding, Returning, and Reconstructing capture distinct aspects of agent social re-entry. Although these dimensions are conceptually connected, improvements in one do not necessarily correspond to improvements in another. Evaluating them separately therefore provides a more complete picture of agent social re-entry.

The contrast between Finding and Returning is particularly informative. Under B3, all six evaluated configurations achieved 100% Finding success, yet target-directed replying remained much less frequent without persistent social Manuscript submitted to ACM

history. This suggests that making a previous conversation easy to identify does not necessarily lead an agent to return to it autonomously. Evaluations based only on explicit conversation selection may therefore overestimate an agent’s ability to re-engage with previous social interactions.

The contrast between Returning and Reconstructing reveals a diferent limitation. The A2 memory configuration was associated with substantially higher Returning rates across all six configurations, but did not produce consistent improvements in contextual continuity. Furthermore, the exploratory distractor-memory comparison suggests that higher Returning rates cannot be attributed solely to access to target-relevant history. Together, these findings indicate that providing additional conversational history may encourage autonomous re-engagement without necessarily improving the quality of the resulting continuation.

These findings support evaluating Finding, Returning, and Reconstructing as separate outcomes rather than using one as a proxy for another. Importantly, the three dimensions are evaluated through independent experimental branches. The observed diferences do not establish a causal sequence or imply distinct internal cognitive mechanisms.

## 6.2 Implications for Persistent Social Agents

Our findings suggest that persistent social agents need to address three related but distinct challenges: identifying relevant previous conversations, deciding when to re-engage, and maintaining contextual continuity after returning. Improvements in one area should not be assumed to resolve the others.

First, conversation resurfacing should help agents identify relevant earlier interactions within an evolving social environment. Our Finding results indicate that the implemented notification-and-promotion condition substantially improved target identification. However, the contrast with Returning suggests that making an interaction salient does not necessarily produce autonomous engagement. Efective agent design therefore requires attention not only to which conversations are surfaced, but also to how an agent determines whether a reply is appropriate.

Second, persistent memory should support more than the presence of historical information in the prompt. Persistent memory should not be evaluated solely by whether it increases target-directed replies; its relevance and contribution to subsequent conversational continuity must also be considered. Although the A2 configuration was associated with increased target-directed replying, the exploratory distractor-memory comparison suggests that the relationship between memory content and observable behavior is not straightforward. Memory blocks may difer in content, length, structure, and contextual cues, each of which could influence the resulting actions. The current experiments do not identify which of these properties drives the observed diferences. Future memory systems should therefore distinguish the retrieval of relevant social history from its presentation and use in action selection.

Third, contextual reconstruction warrants evaluation independently of both target discovery and replying. The absence ofan observed reconstruction benefit under A2 suggests that supplying additional history does not automatically produce a more coherent continuation. Persistent social agents may require context-selection and representation strategies that prioritize information necessary to resume a particular conversational thread, rather than simply increasing the amount of history supplied.

Taken together, these findings motivate evaluation frameworks that assess whether an agent can locate an earlier interaction, autonomously act on it when appropriate, and continue it with suficient contextual continuity. Reporting these outcomes separately can reveal limitations that would otherwise be obscured by a single social re-entry success measure.

## 6.3 Limitations and Future Work

The results suggest several directions for extending the evaluation of agent social re-entry.

One direction is to study more realistic and dynamic social environments. Our current evaluation uses controlled scenarios with fixed interruption lengths and candidate conversation sets, allowing consistent comparisons across models. Future evaluations could consider open-world settings with dynamically evolving conversations, varying interruption durations, and more diverse interaction patterns. Such settings would help examine how Finding, Returning, and Reconstructing perform under the complexity of long-term social interactions.

A second direction is to further investigate the roles of conversational resurfacing and persistent memory. Our current conditions combine multiple factors: B3 includes both notification and conversation promotion, while A2 changes the availability and presentation of conversational history. The exploratory distractor-memory comparison also involves diferences in memory length and structure. Future experiments could separately examine resurfacing strategies and compare more closely matched memory configurations to better understand how memory relevance, selection, and presentation relate to social re-entry.

A third direction is to investigate how Finding, Returning, and Reconstructing relate in end-to-end agent behavior. Our independent-branch design allows the three dimensions to be evaluated separately but does not measure their progression within a single continuous interaction. Future work could extend the framework to longitudinal evaluations that trace how agents identify previous conversations, autonomously return to them, and maintain contextual continuity over time.

## 7 Conclusion

In this paper, we investigate agent social re-entry and formulate it as a multi-dimensional evaluation problem. We propose a shared-snapshot evaluation framework consisting of three complementary dimensions: Finding, Returning, and Reconstructing. The framework evaluates these dimensions independently, allowing us to examine diferent aspects of social re-entry without treating them as a single success outcome.

Our experiments across six LLM service/model configurations reveal that successful Finding does not necessarily imply Returning, and increased Returning does not guarantee stronger Reconstructing. While conversational resurfacing improves Finding and persistent memory is associated with higher Returning rates, these improvements are not accompanied by corresponding gains in Reconstructing. These findings highlight the distinction between the three dimensions and the importance of evaluating them separately.

More broadly, our work extends the evaluation of persistent social agents beyond memory retrieval and isolated conversational responses. By distinguishing the ability to identify previous conversations, autonomously re-engage with them, and maintain contextual continuity, our framework provides a foundation for evaluating how agents sustain social interactions over time.

## References

[1] Hongzhan Chen, Hehong Chen, Ming Yan, Wenshen Xu, Gao Xing, Weizhou Shen, Xiaojun Quan, Chenliang Li, Ji Zhang, and Fei Huang. 2024. SocialBench: Sociality Evaluation of Role-Playing Conversational Agents. In Findings ofthe Association for Computational Linguistics: ACL 2024. Association for Computational Linguistics, Bangkok, Thailand, 2108–2126. doi:10.18653/v1/2024.findings-acl.125

[2] Herbert H. Clark and Susan E. Brennan. 1991. Grounding in Communication. In Perspectives on Socially Shared Cognition, Lauren B. Resnick, John M Levine, and Stephanie D. Teasley (Eds.). American Psychological Association, 127–149. doi:10.1037/10096-006

[3] Gilles O. Einstein and Mark A. McDaniel. 1990. Normal Aging and Prospective Memory. Journal ofExperimental Psychology: Learning, Memory, and Cognition 16, 4 (1990), 717–726. doi:10.1037/0278-7393.16.4.717

[4] Aysajan Eziz. 2026. Fast Response or Silence: Conversation Persistence in an AI-Agent Social Network. arXiv:2602.07667 [econ.EM] doi:10.48550 arXiv.2602.07667

[5] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Kuttler, Mike Lewis, Wen tau Yih, Tim Rocktaschel, Sebastian Riedel, and Douwe Kiela. 2020. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. In Advances in Neural Information Processing Systems, Vol. 33. 9459–9474. https://proceedings.neurips.cc/paper/2020/hash/6b493230205f780e1bc26945df7481e5- Abstract.html

[6] Percy Liang, Rishi Bommasani, Tony Lee, Dimitris Tsipras, Dilara Soylu, Michihiro Yasunaga, Yian Zhang, Deepak Narayanan, Yuhuai Wu, Ananya Kumar, et al. 2023. Holistic Evaluation of Language Models. Transactions on Machine Learning Research (2023). https://openreview.net/forum?id= iO4LZibEqW

[7] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the Middle: How Language Models Use Long Contexts. In Transactions ofthe Association for Computational Linguistics, Vol. 12. 157–173. doi:10.1162/tacl\_a\_00638

[8] Adyasha Maharana, Dong-Ho Lee, Sergey Tulyakov, Mohit Bansal, Francesco Barbieri, and Yuwei Fang. 2024. Evaluating Very Long-Term Conversational Memory of LLM Agents. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers). Association for Computational Linguistics, Bangkok, Thailand, 13851–13870. doi:10.18653/v1/2024.acl-long.747

[9] Gloria Mark, Daniela Gudith, and Ulrich Klocke. 2008. The Cost of Interrupted Work: More Speed and Stress. In Proceedings ofthe SIGCHI Conference on Human Factors in Computing Systems. Association for Computing Machinery, 107–110. doi:10.1145/1357054.1357072

[10] Daniel C. McFarlane and Kara A. Latorella. 2002. The Scope and Importance of Human Interruption in Human–Computer Interaction Design. Human–Computer Interaction 17, 1 (2002), 1–61. doi:10.1207/S15327051HCI1701\_1

[11] Joon Sung Park, Joseph C. O’Brien, Carrie J. Cai, Meredith Ringel Morris, Percy Liang, and Michael S. Bernstein. 2023. Generative Agents: Interactive Simulacra of Human Behavior. In Proceedings ofthe 36th Annual ACM Symposium on User Interface Software and Technology. Association for Computing Machinery. doi:10.1145/3586183.3606763

[12] Qinglong Shi, Donghai Wang, Hantao Zhou, Jiguo Li, Jun Xu, Jiuchong Gao, Jinghua Hao, and Renqing He. 2026. Long-term Task-oriented Agent: Proactive Long-term Intent Maintenance in Dynamic Environments. arXiv:2601.09382 [cs.AI] https://arxiv.org/abs/2601.09382

[13] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, and Illia Polosukhin. 2017. Attention Is All You Need. In Advances in Neural Information Processing Systems, Vol. 30. https://proceedings.neurips.cc/paper/7181-attention-is-all-you-need

[14] Lei Wang, Chen Ma, Xueyang Feng, Zeyu Zhang, Hao Yang, Jingsen Zhang, Zhiyuan Chen, Jiakai Tang, Xu Chen, Yankai Lin, et al. 2024. A Survey on Large Language Model Based Autonomous Agents. Frontiers ofComputer Science 18, 6 (2024), 186345. doi:10.1007/s11704-024-40231-1

[15] Lingzhi Wang, Xingshan Zeng, Huang Hu, Kam-Fai Wong, and Daxin Jiang. 2021. Re-entry Prediction for Online Conversations via Self-Supervised Learning. In Findings of the Association for Computational Linguistics: EMNLP 2021. 2127–2137. doi:10.18653/v1/2021.findings-emnlp.183

[16] Zhiheng Xi, Wenxiang Chen, Xin Guo, Wei He, Yiwen Ding, Boyang Hong, Ming Zhang, Junzhe Wang, Senjie Jin, Enyu Zhou, et al. 2023. The Rise and Potential of Large Language Model Based Agents: A Survey. arXiv preprint arXiv:2309.07864 (2023). doi:10.48550/arXiv.2309.07864

[17] Shunyu Yao, Jefrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik Narasimhan, and Yuan Cao. 2023. ReAct: Synergizing Reasoning and Acting in Language Models. In International Conference on Learning Representations. https://openreview.net/forum?id=WE\_vluYUL-X

[18] Saber Zerhoudi, Kanishka Ghosh Dastidar, Felix Klement, Artur Romazanov, Andreas Einwiller, Dang H. Dang, Michael Dinzinger, Michael Granitzer, Annette Hautli-Janisz, Stefan Katzenbeisser, Florian Lemmerich, and Jelena Mitrović. 2026. Form Without Function: Agent Social Behavior in the Moltbook Network. arXiv:2604.13052 [cs.SI] doi:10.48550/arXiv.2604.13052