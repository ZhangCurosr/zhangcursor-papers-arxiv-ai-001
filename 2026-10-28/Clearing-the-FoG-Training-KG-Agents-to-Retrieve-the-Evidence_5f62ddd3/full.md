# Clearing the FoG: Training KG Agents to Retrieve the Evidence Behind Their Answers

Zongyuan Sun<sup>1,\*</sup>, Shuai Zhao<sup>1,†</sup>, Ke Shu<sup>2,\*</sup>, Yingjun Du<sup>3</sup>, Ziyue Zhu<sup>1</sup>, Haoran Luo<sup>4,†</sup> <sup>1</sup>Beijing University of Posts and Telecommunications, Beijing, China <sup>2</sup>University of Helsinki, Helsinki, Finland <sup>3</sup>University of Amsterdam, Amsterdam, Netherlands <sup>4</sup>Nanyang Technological University, Singapore <sup>\*</sup>Equal contribution. <sup>†</sup>Corresponding authors.

## Abstract

Large language models (LLMs) can answer many factual questions correctly, but a correct answer is not necessarily a supported one. For an LLM agent, a knowledge graph (KG) plays two roles: as a source of answers, it supplies facts that the model lacks; as a source of evidence, it supplies the supporting triples that make an answer traceable and auditable. However, existing KG agents treat the KG only as a source of answers: their training data carries answers but not evidence, and their training rewards check the answer but not the retrieved evidence, so agents often answer correctly without retrieving the supporting triples. We propose FoG, a framework that trains KG agents to retrieve the evidence behind their answers. FoG consists of evidence-grounded synthesis, a KG interaction agent, and an evidencecoverage process reward (ECPR): the synthesis recursively hides explicit entities behind verifiable sub-questions, so that every question carries its supporting triples by construction; the agent records the triples returned by the KG; and ECPR credits each supporting triple to the turn that first retrieves it. Experiments on three KGQA benchmarks demonstrate that FoG enables 3B–4B open-source agents to outperform substantially larger baselines in answer accuracy and evidence coverage, with zero-shot cross-KG transfer.

## 1 Introduction

Large language models (LLMs) (Team, 2025a; GLM, 2025; Team, 2025b; Touvron et al., 2023; OpenAI, 2023) increasingly answer factual questions from parametric knowledge, but remain difficult to trust on questions that require complex factual reasoning (Mirzadeh et al., 2025; Hu et al., 2024; Jullien et al., 2025; Petroni et al., 2021). A correct answer is not necessarily a supported one: the model may answer from memory or surface cues while exposing little about why the answer holds (Huang et al., 2025; Alansari and Luqman, 2026; Nguyen et al., 2024), which is problematic in high-stakes settings such as healthcare and law (Cabello et al., 2024), where users need evidence that they can check (Sui et al., 2025; Agrawal et al., 2024). Current LLM answers thus lie in a fog: the answer is visible, yet the evidence behind it remains hidden.

![](images/01d68f394aa36c2457b6aa49c1bd0e8b83167d97521526513dde56253f02f170.jpg)  
Figure 1: Motivation of FoG. Answer-only training yields correct but unsupported answers, whereas FoG embeds supporting triples at synthesis and rewards their retrieval during RL training.

Knowledge graphs (KGs) (Pan et al., 2024) offer a natural way to clear this fog, since they store facts as verifiable triples. Existing KG-augmented methods follow two paradigms (Figure 2). Retrievalbased methods (Luo et al., 2024; Li et al., 2025; Luo et al., 2025) retrieve triples, paths, or subgraphs and answer from them. Agent-based methods (Sun et al., 2024; Chen et al., 2024; Lin et al., 2025; Jiang et al., 2025) interact with the KG through a multi-turn “think–act–observe” process, and recent work trains them with reinforcement learning (RL) (Lin et al., 2025; Chen et al., 2025;

![](images/90bd994cf46f924ded242c9e6c8f619730edcf1218d83ed36cc69b982ec2fde5.jpg)  
Figure 2: Comparison of KG-augmented reasoning paradigms. All three methods return the correct answer, but retrieval-based methods miss key evidence and agent-based methods answer after the first hop, whereas FoG retrieves every supporting triple of the answer.

## Sun et al., 2025, 2026a).

However, existing KG agents treat the KG only as a source of answers and overlook its role as a source of evidence. An LLM with no KG access can already answer many benchmark questions correctly, and even an RL-trained agent can answer correctly while retrieving only part of the supporting triples. We identify two causes. First, training data carries answers but not evidence: most KGQA datasets annotate answers and at most logical forms (Talmor and Berant, 2018; Yih et al., 2016; Gu et al., 2021), and recovering the supporting triples afterwards requires enumerating and verifying paths for every question. Second, training rewards check the answer but not the evidence: most rewards score only the final answer or whether the KG returns it (Lin et al., 2025; Song et al., 2026; Xu et al., 2026; Chen et al., 2025; Sun et al., 2025), so a trajectory that reaches the answer through part of the evidence earns as much as one that retrieves all of it. The process rewards that go further depend on an LLM judge or on gold paths recovered for every question (Wang and Yu, 2026a; Yan et al., 2026), both of which are costly. Both causes trace back to one bottleneck: evidence is expensive to recover once a question has been written.

To remove this bottleneck, we propose FoG (Figure 1), a framework that trains KG agents to retrieve the evidence behind their answers. Our key observation is that evidence is hard to recover after a question is written but can be embedded when the question is constructed and then rewarded during training. First, evidence-grounded synthesis recursively hides the explicit entities of a seed question behind verifiable sub-questions, so every question carries its supporting triples. Second, a KG interaction agent answers through multi-turn KG queries and records every triple that the KG returns. Lastly, during RL training, an evidence-coverage process reward (ECPR) gives a dense reward to each turn that first retrieves one of these supporting triples. Our contributions are as follows:

• We distinguish two roles of the KG and show that existing training data and rewards serve it only as a source of answers, not of evidence.

• We propose FoG, which embeds supporting triples into synthetic questions and rewards their retrieval through ECPR, so the dense reward comes from data synthesis with no LLM judge or extra annotation.

• On three benchmarks, 3B–4B FoG agents outperform larger baselines in accuracy and evidence coverage, and transfer zero-shot across KGs.

## 2 Related Work

Knowledge graph augmented reasoning. Unlike prompting strategies (Wei et al., 2022; Wang et al., 2023; Yao et al., 2023a; Jin et al., 2024), KGaugmented methods ground reasoning in verifiable triples and follow two paradigms. Retrieval-based methods (Luo et al., 2024; Li et al., 2025; Luo et al., 2025; Mavromatis and Karypis, 2025; He et al., 2024; Liu et al., 2026) condition the LLM on triples, paths, or subgraphs retrieved in a single step. Agent-based methods (Sun et al., 2024; Chen et al., 2024; Lin et al., 2025; Jiang et al., 2025) explore the KG iteratively: ToG and PoG prompt a frozen LLM, whereas recent work trains the agent with RL (Lin et al., 2025; Song et al., 2026; Chen et al., 2025; Sun et al., 2025; Wang and Yu, 2026b; Sun et al., 2026a). However, these agents use the KG only as a source of answers and can answer correctly without retrieving the supporting triples. In this work, we train agents to use it as a source of evidence.

Synthetic data for KGQA. Synthetic data frees KGQA from human annotation, since questions can be generated from the KG and verified by execution. It provides questions for agent training (Xu et al., 2026) and for benchmarking and retriever training (Zhang et al., 2025; Cattaneo et al., 2026). However, these pipelines treat it as extra examples to train or evaluate on. In this work, we treat it as a source of supervision: every synthesized question carries the evidence that defines its reward.

Rewards for KG agents. RL lets a KG agent learn its retrieval policy from interaction with the KG. Most work trains from outcome rewards on the final answer with rule-based terms for valid interaction (Lin et al., 2025; Song et al., 2026; Chen et al., 2025; Sun et al., 2025; Wang and Yu, 2026b; Sun et al., 2026a; Sun and Kazakov, 2026). Because outcome rewards are sparse, later work adds dense process rewards, obtained from an LLM judge (Wang and Yu, 2026a) or from gold paths and logical forms (Yan et al., 2026; Han et al., 2026; Sun et al., 2026b). However, both sources are costly: a judge must score every rollout, and gold references must be annotated or recovered for every question. In this work, the process reward comes from data synthesis: every synthesized question already carries its supporting triples, which provide a dense reward at no extra cost.

## 3 Method

In this section, we introduce FoG, as shown in Figure $^ { 3 , }$ including evidence-grounded synthesis, a KG interaction agent, and ECPR.

Definition 1 (KG question answering). A knowledge graph is $\mathcal { G } = ( \mathcal { E } , \mathcal { R } , \mathcal { T } )$ with entities $\mathcal { E } _ { : }$ relations $\mathcal { R }$ , and triples $\mathcal { T } \subseteq \mathcal { E } \times \mathcal { R } \times \mathcal { E }$ . A question $q$ has a logical form $\ell _ { q }$ and a gold answer $y _ { q } ^ { * } ~ \in ~ \mathrm { S P A R Q L } ( \ell _ { q } , T )$ , where SPARQL(ℓ, S) is the result of executing $\ell$ over a triple set ${ \mathcal { S } } .$ An agent answers q through a trajectory $\tau =$ $\big ( ( s _ { 1 } , a _ { 1 } ) , \dotsc , ( s _ { T } , a _ { T } ) \big )$ of states and actions and outputs a prediction $\hat { y } ,$ where $\mathcal { T } _ { t } \subseteq \mathcal { T }$ are the triples returned at step $t , \ T ( \tau ) \ = \ U _ { t } \mathcal { T } _ { t }$ all triples retrieved along $\tau ,$ , and $\mathcal { E } ( \tau )$ all returned entities.

Definition 2 (Supported answer). A triple set $\begin{array} { r l r l } { s } & { { } \subseteq { } } & { \tau } \end{array}$ supports the answer of $q$ if $y _ { q } ^ { * } \in$ $\operatorname { S P A R Q L } ( \ell _ { q } , S )$ . The answer of $\tau$ is correct if $\hat { y }$ matches $y _ { q } ^ { * } .$ , reached if $y _ { q } ^ { * } \in { \mathcal { E } } ( \tau )$ , and supported if $\mathcal T ( \tau )$ supports it. Unlike the first two levels, which use the KG as a source of answers, a supported answer uses it as a source of evidence. In FoG, each training question carries gold supporting triples $S _ { q } ^ { * }$ that support its answer.

## 3.1 Evidence-Grounded Synthesis

To obtain complex questions that carry their supporting evidence, we propose evidence-grounded synthesis that includes seed question construction, backward expansion, and quality control. All roles use the teacher Kimi-K2 (Team, 2025a).

Seed question construction. We sample a sizebounded subgraph $\mathcal { G } _ { \mathrm { s u b } }$ from Wikidata around a seed entity $v _ { \mathrm { s e e d } }$ of moderate degree. The teacher generates a 1-hop question whose answer is $v _ { \mathrm { s e e d } } \mathrm { : }$

$$
\left( Q ^ { ( 1 ) } , \ell ^ { ( 1 ) } , \mathcal { S } ^ { ( 1 ) } \right) \sim \pi _ { \mathrm { g e n } } \bigl ( v _ { \mathrm { s e e d } } , \mathcal { G } _ { \mathrm { s u b } } \bigr ) ,\tag{1}
$$

where $\pi _ { \mathrm { g e n } }$ is the teacher model, ${ Q } ^ { ( 1 ) }$ the seed question, $\ell ^ { ( 1 ) }$ its logical form, and ${ \mathcal { S } } ^ { ( 1 ) }$ its supporting triples. The seed is retained only if every triple in ${ \mathcal { S } } ^ { ( 1 ) }$ exists in $\mathcal { G } _ { \mathrm { s u b } }$ and $\ell ^ { ( 1 ) }$ executes over the global KG with $v _ { \mathrm { s e e d } }$ in its result set.

Backward expansion. Given a verified question $Q ^ { ( n ) }$ with logical form $\ell ^ { ( n ) }$ , answer $y \ = \ v _ { \mathrm { s e e d } } .$ and supporting triples $S ^ { ( n ) }$ , we replace an explicit non-answer entity u in $Q ^ { ( n ) }$ with a sub-question $\tilde { Q }$ whose answer is exactly u:

![](images/4069867c17e8e716b8f2b01e29bb1207970de3d0dda522e938aa3622a690512f.jpg)  
Figure 3: The FoG pipeline. Evidence-grounded synthesis produces complex questions with gold supporting triples, the KG interaction agent records the triples that it retrieves, and ECPR rewards the retrieval of the gold supporting triples.

$$
\begin{array} { r l } & { Q ^ { ( n + 1 ) } = Q ^ { ( n ) } [ u  \tilde { Q } ] , } \\ & { S ^ { ( n + 1 ) } = S ^ { ( n ) } \cup \tilde { S } , } \\ & { y \in \mathrm { S P A R Q L } ( \ell ^ { ( n + 1 ) } , S ^ { ( n + 1 ) } ) , } \end{array}\tag{2}
$$

where $\tilde { \cal S }$ are the supporting triples of ${ \tilde { Q } } .$ , drawn from the incoming neighborhood of $u ,$ and $\ell ^ { ( n + 1 ) }$ is the expanded logical form. Each expansion turns an explicit entity into a hidden intermediate variable, and the accumulated triples still support the answer. Thus $N \in \{ 0 , 1 , 2 \}$ expansions yield 1- to 3-hop questions, where $N = 0$ keeps the seed. Expansion uses a ReAct agent (Yao et al., 2023b) that retrieves the neighborhood of u, selects triples, writes ${ \tilde { Q } } .$ and verifies it; on failure, it reselects triples or the entity.

Proposition 1. Backward expansion preserves the answer and yields triples that support it.

Proof. We provide experimental results in Section 4.4 and proofs in Appendix A.1.

Quality control. Each synthesized question must further pass two symbolic checks, which keep the answer specific and every topic entity necessary. (a) Executability. Let $\mathcal { V } ^ { ( n ) } \stackrel {  } { = } \mathrm { S P A R Q L } ( \ell ^ { ( n ) } , \bar { T } )$ be the result set of the candidate question. We retain it only if the target answer is included and the result set is not overly broad, i.e., $y \in \mathcal { D } ^ { ( n ) }$ and $| \mathcal { V } ^ { ( n ) } | \le K _ { \mathrm { a n s } }$ for a maximum answer-set size $K _ { \mathrm { a n s } }$ (b) Non-redundancy. Let $E _ { q }$ be the topic entities of the question. We extract a shortest supporting path for each $e \in E _ { q }$ and discard the question if a proper subset yields an equally specific answer set:

$$
\begin{array} { c } { \mathcal { F } _ { \mathrm { r e d } } ( Q ^ { ( n ) } ) = \mathbb { I } \Bigl [ \exists Z \subsetneq E _ { q } : \mathcal { Y } ^ { ( n ) } \subseteq \mathcal { Y } _ { Z } ^ { ( n ) } , } \\ { \qquad | \mathcal { Y } _ { Z } ^ { ( n ) } | \leq | \mathcal { Y } ^ { ( n ) } | \Bigr ] , } \end{array}\tag{3}
$$

where ${ \mathcal { V } } _ { Z } ^ { ( n ) }$ is the answer set of the query built only from the paths of Z (Appendix B).

The triples accumulated for a final question q are its gold supporting triples $\boldsymbol { S } _ { \boldsymbol { q } } ^ { * }$ , and we report the distribution of query-graph types (Dutt et al., 2023) in Appendix C.2.

## 3.2 KG Interaction Agent

To make the evidence retrieved for an answer explicit, we develop a KG interaction agent that solves each question through a multi-turn think– act–observe loop with the action space

$$
\begin{array} { r } { \mathcal { A } = \{ \mathsf { t h i n k } , \mathsf { k g - q u e r y } , \mathsf { a n s w e r } \} , } \end{array}\tag{4}
$$

where think performs internal reasoning, kg-query interacts with the KG, and answer emits the final prediction. The kg-query action is instantiated with seven atomic operations that probe the relations of an entity, expand to its neighboring entities, filter an entity set, and combine entity sets (Appendix D). Every trajectory thus records the returned triples $\mathcal T ( \tau )$ , the only evidence that FoG rewards and evaluates.

## 3.3 Evidence-Coverage Process Reward

To train the agent to retrieve this evidence, we propose a training scheme with outcome and stepvalidity rewards, the evidence-coverage process reward, and turn-level policy optimization.

Outcome and step-validity rewards. We define the outcome reward of a trajectory as

$$
\begin{array} { r } { R _ { \mathrm { o u t } } ( \tau ) = \lambda _ { \mathrm { f l } } \cdot \mathrm { F } 1 ( \hat { y } , y _ { q } ^ { * } ) + \lambda _ { \mathrm { r e t } } \cdot \mathbb { I } [ y _ { q } ^ { * } \in \mathcal { E } ( \tau ) ] , } \end{array}\tag{5}
$$

where the first term rewards a correct answer and the second a reached answer. Neither term checks whether the answer is supported: a shortcut trajectory, which outputs and even reaches the gold answer while leaving some of the supporting triples $S _ { q } ^ { * }$ unretrieved, still earns the full outcome reward.

To stabilize graph interaction, each turn also earns a step-validity reward $r _ { t } ^ { \mathrm { s t e p } } = \lambda _ { \mathrm { s t e p } } v _ { t }$ , where $\ v _ { t } \ = \ 1$ if action $a _ { t }$ follows the required format and executes successfully, and $v _ { t } ~ = ~ 0$ otherwise. The trajectory-level term is the average $\begin{array} { r } { R _ { \mathrm { s t e p } } ( \tau ) = \frac { 1 } { T } \dot { \sum _ { t = 1 } ^ { T } } r _ { t } ^ { \mathrm { s t e p } } } \end{array}$

Evidence-coverage process reward. Unlike the terms above, which treat the KG only as a source of answers, the Evidence-Coverage Process Reward (ECPR) scores the trajectory against the gold supporting triples $S _ { q } ^ { * }$ provided by synthesis. We define the evidence coverage of a trajectory as

$$
\operatorname { C o v } ( \tau , S _ { q } ^ { * } ) = \frac { | \mathcal { T } ( \tau ) \cap S _ { q } ^ { * } | } { | S _ { q } ^ { * } | } ,\tag{6}
$$

where $\mathcal T ( \tau )$ are all triples retrieved along $\tau ,$ so a triple that the agent recalls from memory without retrieving it earns no reward. We call τ evidencecovered if Cov $( \tau , { \cal S } _ { q } ^ { * } ) = 1$ , and set $R _ { \mathrm { e c p r } } ( \tau ) =$ $\lambda _ { \mathrm { e c p r } } \cdot \operatorname { C o v } ( \tau , S _ { q } ^ { * } )$ . For dense credit assignment, first-match attribution assigns each gold triple to the step that first retrieves it:

$$
r _ { t } ^ { \mathrm { e c p r } } = \lambda _ { \mathrm { e c p r } } \cdot \frac { \left| ( \mathcal { T } _ { t } \cap \mathcal { S } _ { q } ^ { * } ) \setminus \bigcup _ { i < t } \mathcal { T } _ { i } \right| } { | \mathcal { S } _ { q } ^ { * } | } ,\tag{7}
$$

where $\begin{array} { r } { \sum _ { t } r _ { t } ^ { \mathrm { e c p r } } = R \mathrm { e c p r } ( \tau ) } \end{array}$ , so repeated retrieval of the same triple earns no additional reward. The composite reward of a trajectory is $R ( \tau )$ $R _ { \mathrm { o u t } } ( \tau ) + R _ { \mathrm { s t e p } } ( \tau ) + R _ { \mathrm { e c p r } } ( \tau )$ . We set $\lambda _ { \mathrm { f l } } = 0 . 7 $ $\lambda _ { \mathrm { { r e t } } } = 0 . 3 , \lambda _ { \mathrm { { s t e p } } } = 0 . 5 ,$ and $\lambda _ { \mathrm { e c p r } } = 1$

Proposition 2. ECPR is maximized only by supported answers and rewards each triple once.

Proof. We provide experimental results in Sections 4.3 and 4.6 and proofs in Appendix A.2.

Turn-level policy optimization. Following prior RL-trained KG agents (Song et al., 2026; Chen et al., 2025; Sun et al., 2025), the policy π is first initialized by supervised fine-tuning (SFT) on demonstration trajectories that a teacher model produces for the synthesized and training questions, minimizing the negative log-likelihood of the teacher actions. We then optimize it with Group Relative Policy Optimization (GRPO) (Shao et al., 2024) and, following KG-R1 (Lin et al., 2025), compute the advantage per turn. For each question $q ,$ the policy samples N trajectories $\{ \tau _ { i } \} _ { i = 1 } ^ { \tilde { N } }$ , and turn t of $\tau _ { i }$ receives the return

$$
G _ { i , t } = R _ { \mathrm { o u t } } ( \tau _ { i } ) + r _ { i , t } ^ { \mathrm { s t e p } } + r _ { i , t } ^ { \mathrm { e c p r } } ,\tag{8}
$$

i.e., the shared outcome reward plus the process rewards earned by this turn. The advantage $\hat { A } _ { i , t } =$ $( G _ { i , t } - \mu _ { q } ) / ( \sigma _ { q } + \delta )$ is normalized with the mean $\mu _ { q }$ and standard deviation $\sigma _ { q }$ of the returns over all turns of the N trajectories. A turn that first retrieves gold evidence therefore receives a higher advantage than the other turns. The policy then maximizes the clipped GRPO objective with these advantages, regularized toward the SFT model (Appendix H).

## 4 Experiments

This section presents the experimental setup, main results, and analysis. We answer the following research questions (RQs): RQ1: Does FoG outperform other methods in answer accuracy? RQ2: Does each component of FoG work? RQ3: Does evidence-grounded synthesis provide valid evidence and better training data? RQ4: Does the KG interaction agent retrieve the evidence behind its answers? RQ5: How does the design of ECPR affect evidence coverage? RQ6: Does FoG transfer to a KG unseen during training?

## 4.1 Experimental Setup

Datasets. We evaluate on three Wikidata-based KGQA benchmarks, WikiWebQuestions (Xu et al., 2023), QALD10-en (Usbeck et al., 2024), and GT-SQA (Cattaneo et al., 2026), and use WebQSP (Yih et al., 2016) over Freebase for cross-KG transfer. The synthetic questions do not overlap with the test sets (Appendices C.4 and E).

<table><tr><td rowspan="2">Category</td><td rowspan="2">Methods</td><td rowspan="2">Backbone</td><td colspan="2">WikiWebQuestions</td><td colspan="2">QALD10-en</td><td colspan="2">GTSQA</td></tr><tr><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td></tr><tr><td rowspan="7">LLM Reasoning</td><td>IO Prompt (Team, 2025b)</td><td>Qwen2.5-7B-Instruct</td><td>28.24</td><td>44.43</td><td>11.05</td><td>12.30</td><td>9.62</td><td>10.23</td></tr><tr><td>IO Prompt (OpenAI, 2023)</td><td>GPT-4o-mini</td><td>40.31</td><td>68.10</td><td>15.34</td><td>18.78</td><td>15.37</td><td>16.46</td></tr><tr><td>CoT (Wei et al., 2022)</td><td>GPT-4o-mini</td><td>41.66</td><td>58.42</td><td>22.68</td><td>26.81</td><td>18.87</td><td>20.12</td></tr><tr><td>Self-Consistency (Wang et al., 2023)</td><td>GPT-4o-mini</td><td>46.34</td><td>65.20</td><td>24.27</td><td>28.57</td><td>19.07</td><td>20.21</td></tr><tr><td>IO Prompt (DeepSeek-AI, 2024)</td><td>DeepSeek-V3</td><td>42.36</td><td>59.89</td><td>18.97</td><td>21.77</td><td>24.77</td><td>26.20</td></tr><tr><td>CoT (Wei et al., 2022)</td><td>DeepSeek-V3</td><td>45.91</td><td>64.60</td><td>19.02</td><td>25.28</td><td>27.66</td><td>29.41</td></tr><tr><td>Self-Consistency (Wang et al., 2023)</td><td>DeepSeek-V3</td><td>49.72</td><td>66.79</td><td>26.25</td><td>29.97</td><td>33.59</td><td>35.82</td></tr><tr><td rowspan="3">Retrieval-based</td><td>RoG (Luo et al., 2024)</td><td>Qwen2.5-7B-Instruct</td><td>53.27</td><td>68.71</td><td></td><td></td><td>41.10</td><td>42.48</td></tr><tr><td>SubGraphRAG-100 (Li et al., 2025)</td><td>Qwen2.5-7B-Instruct</td><td>56.84</td><td>72.36</td><td></td><td></td><td>44.27</td><td>45.93</td></tr><tr><td>GCR (Luo et al., 2025)</td><td>GPT-4o-mini</td><td>55.33</td><td>71.27</td><td></td><td></td><td>32.61</td><td>37.48</td></tr><tr><td rowspan="5">Agent-based</td><td>ToG (Sun et al., 2024)</td><td>GPT-4o-mini</td><td>58.55</td><td>72.75</td><td>30.63</td><td>32.18</td><td>41.08</td><td>42.36</td></tr><tr><td>PoG (Chen et al., 2024)</td><td>GPT-4o-mini</td><td>57.32</td><td>71.89</td><td>30.68</td><td>34.07</td><td>37.77</td><td>39.33</td></tr><tr><td>KG-R1 (Lin et al., 2025)</td><td>Qwen3-4B-Instruct</td><td>64.49</td><td>73.01</td><td>44.07</td><td>45.41</td><td>46.14</td><td>47.73</td></tr><tr><td>FoG-3B</td><td>Qwen2.5-3B-Instruct</td><td>66.28</td><td>75.43</td><td>45.63</td><td>48.92</td><td>48.25</td><td>49.20</td></tr><tr><td>FoG-4B</td><td>Qwen3-4B-Instruct</td><td>66.41</td><td>76.93</td><td>47.80</td><td>49.01</td><td>48.66</td><td>50.52</td></tr></table>

Table 1: Main results on three KGQA benchmarks. Best results are shown in bold, and second-best results are underlined. Our methods are highlighted in blue.
<table><tr><td rowspan="2">Variants</td><td colspan="4">WikiWebQuestions</td><td colspan="4">GTSQA</td></tr><tr><td>F1</td><td>Hit@1</td><td>RetQ</td><td>Cov</td><td>F1</td><td>Hit@1</td><td>RetQ</td><td>Cov</td></tr><tr><td>SFT only</td><td> $5 9 . 9 5 _ { \pm 0 . 1 8 }$ </td><td> $7 0 . 1 0 _ { \pm 0 . 2 4 }$ </td><td> $7 9 . 4 2 _ { \pm 0 . 3 1 }$ </td><td> $7 0 . 6 3 _ { \pm 0 . 2 9 }$ </td><td> $4 4 . 7 3 _ { \pm 0 . 2 2 }$ </td><td> $4 6 . 1 9 _ { \pm 0 . 2 7 }$ </td><td> $5 8 . 4 7 _ { \pm 0 . 3 3 }$ </td><td> $4 9 . 8 5 _ { \pm 0 . 3 0 }$ </td></tr><tr><td> $\mathrm { S F T } + \mathrm { R L } \mathrm { w } / { 0 } \ R _ { \mathrm { e c p r } }$ </td><td> $6 3 . 8 7 _ { \pm 0 . 4 4 }$ </td><td> $7 4 . 3 5 _ { \pm 0 . 5 3 }$ </td><td> $8 3 . 1 7 _ { \pm 0 . 4 9 }$ </td><td> $7 2 . 9 1 _ { \pm 0 . 5 6 }$ </td><td> $4 6 . 0 2 _ { \pm 0 . 4 9 }$ </td><td> $4 6 . 9 7 _ { \pm 0 . 5 6 }$ </td><td> $6 1 . 0 2 _ { \pm 0 . 5 2 }$ </td><td> $5 1 . 4 3 _ { \pm 0 . 5 8 }$ </td></tr><tr><td> $\mathrm { S F T } + \mathrm { R L } \mathrm { w } / { 0 } \ R _ { \mathrm { s t e p } }$ </td><td> $6 5 . 0 2 _ { \pm 0 . 3 3 }$ </td><td> $7 6 . 4 8 _ { \pm 0 . 4 0 }$ </td><td> $8 5 . 6 2 _ { \pm 0 . 3 8 }$ </td><td> $7 7 . 2 5 _ { \pm 0 . 4 2 }$ </td><td> $4 7 . 9 6 _ { \pm 0 . 3 9 }$ </td><td> $4 8 . 1 7 _ { \pm 0 . 4 7 }$ </td><td> $6 3 . 1 8 _ { \pm 0 . 4 4 }$ </td><td> $5 7 . 2 1 _ { \pm 0 . 4 6 }$ </td></tr><tr><td> $\mathrm { S F T } + \mathrm { R L } , R _ { \mathrm { e c p r } } {  } \mathrm { L L M } { \ - } \mathrm { a s } { \ - } \mathrm { j u d g e }$ </td><td> $6 4 . 9 5 _ { \pm 0 . 3 9 }$ </td><td> $7 5 . 4 1 _ { \pm 0 . 5 1 }$ </td><td> $8 4 . 0 3 _ { \pm 0 . 4 5 }$ </td><td> $7 4 . 1 2 _ { \pm 0 . 5 0 }$ </td><td> $4 6 . 8 8 _ { \pm 0 . 4 3 }$ </td><td> $4 8 . 2 5 _ { \pm 0 . 5 2 }$ </td><td> $6 2 . 2 4 _ { \pm 0 . 4 8 }$ </td><td> $5 3 . 3 6 _ { \pm 0 . 5 1 }$ </td></tr><tr><td> $\mathbf { S F T } + \mathbf { R L } \left( \mathbf { f u l l F o G } \right)$ </td><td> ${ \bf 6 6 . 4 1 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> ${ \bf 7 6 . 9 3 _ { \pm 0 . 3 8 } }$  </td><td> $\mathbf { 8 6 . 3 5 _ { \pm 0 . 3 5 } }$ </td><td> $7 8 . 3 4 _ { \pm 0 . 3 7 }$ </td><td> $\mathbf { 4 8 . 6 6 _ { \pm 0 . 3 4 } }$ </td><td> ${ \bf 5 0 . 5 2 } _ { \pm 0 . 4 1 }$  </td><td> ${ \bf 6 5 . 3 4 _ { \pm 0 . 3 9 } }$ </td><td> $5 8 . 6 4 _ { \pm 0 . 4 0 }$ </td></tr></table>

Table 2: Ablation of training stages and reward terms. The LLM-as-judge variant replaces $R _ { \mathrm { e c p r } }$ with a modelbased process score. Mean and std. over three seeds.

a single inference run per question (Appendix H).

Baselines. We compare FoG against LLM reasoning (IO Prompt, CoT (Wei et al., 2022), Self-Consistency (Wang et al., 2023)), retrieval-based (RoG (Luo et al., 2024), SubGraphRAG (Li et al., 2025), GCR (Luo et al., 2025)), and agent-based (ToG (Sun et al., 2024), PoG (Chen et al., 2024), KG-R1 (Lin et al., 2025)) methods. Retrieval-based methods require dataset-specific fine-tuning and are omitted from QALD10-en, whose training split has only 387 examples (Appendix G).

## 4.2 Main Results (RQ1)

Evaluation Metrics. We report Hit@1 and token-level F1 for answer accuracy. For evidence coverage, we report Cov, the fraction of gold supporting triples retrieved along the trajectory, and RetQ, the fraction of questions whose gold answer is returned by a KG query. GTSQA provides gold supporting triples natively, and we construct them for WikiWebQuestions with an LLM verifier that agrees with human annotators on 93.0% of 200 audited paths (Appendices F.1, F.2, and F.3).

As shown in Table 1, we compare FoG with all baselines in answer accuracy.

Implementation Details. We train FoG on Qwen2.5-3B-Instruct and Qwen3-4B-Instruct with LoRA-based SFT followed by GRPO, sampling N=4 trajectories of at most 10 KG turns per question for 450 steps. All methods are evaluated with

Comparison across methods. FoG achieves the best F1 and Hit@1 on all three benchmarks, exceeding the strongest baseline KG-R1 by up to +3.73 F1 and +3.92 Hit@1. As KG-R1 is also RL-trained on the same Qwen3-4B backbone, this indicates that training an agent to retrieve evidence also makes its answers more accurate.

Comparison across backbones. FoG-3B still exceeds the 4B KG-R1 on all three benchmarks, and FoG-4B surpasses GPT-4o-mini agents by at least +7.58 F1 and 7B retrieval-based methods by at least +4.39 F1. This suggests that how an agent is trained to retrieve matters more than its backbone size.

## 4.3 Ablation Study (RQ2)

As shown in Table 2, we ablate the training stages and reward terms on the two benchmarks with gold supporting triples, under the same backbone, SFT checkpoint, RL data, and rollout budget, so the variants differ only in the reward.

![](images/430f9f705ab37a5a578515f53790e2db51097c52d7d73517894d427e9ab13cf9.jpg)

![](images/37fe70077de3940fe827d3c4bcb1e8293490ddf434f91add975b2fc24de975cc.jpg)

![](images/f1b210f847217182103526c157b631fb91404b221f02b62c84c5d9a1d99eec04.jpg)

Figure 4: Effect of model scale. F1 and Hit@1 of backbones of different sizes after SFT on the same data.
<table><tr><td rowspan="2">Data Source</td><td colspan="2">WikiWebQuestions</td><td colspan="2">QALD10-en</td><td colspan="2">GTSQA</td></tr><tr><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td><td>F1</td><td>Hit@1</td></tr><tr><td>Train-Only</td><td>53.58</td><td>64.51</td><td>36.94</td><td>37.86</td><td>41.33</td><td>41.52</td></tr><tr><td>Synth-Only</td><td>57.84</td><td>68.17</td><td>39.52</td><td>40.36</td><td>43.21</td><td>44.58</td></tr><tr><td>Mixed</td><td>59.95</td><td>70.10</td><td>41.38</td><td>42.31</td><td>44.73</td><td>46.19</td></tr></table>

Table 3: Impact of synthetic training data.

Impact of RL. Full FoG improves over SFT only by +6.83 and +4.33 Hit@1, but RL without $R _ { \mathrm { e c p r } }$ raises Cov by only +2.28 and +1.58. This is because the outcome reward is paid in full once the answer is reached, whatever evidence is retrieved.

Impact of $R _ { \mathbf { e c p r } } .$ Removing $R _ { \mathrm { e c p r } }$ causes the largest drop among the reward terms, and Cov drops more (5.43 and 7.21) than Hit@1 (2.58 and 3.55). As both variants share the data and the SFT checkpoint, this confirms that the coverage gain comes from the reward itself.

Impact of $R _ { \mathbf { s t e p } } .$ Removing $R _ { \mathrm { s t e p } }$ lowers Hit@1 by 0.45 and 2.35 but Cov by only 1.09 and 1.43, indicating that step validity keeps the interaction executable but does not drive coverage.

ECPR vs. LLM-as-judge. An LLM-as-judge score of the reasoning process (Wang and Yu, 2026a) raises Cov by only 1.21 and 1.93 over no process supervision and stays 4.22 and 5.28 below ECPR. This suggests that judging how plausible the reasoning reads is a weaker signal than ECPR.

## 4.4 Evidence-Grounded Synthesis (RQ3)

As shown in Table 3 and Figure 4, we fix a budget of 6,588 SFT questions and vary the data source (Train-Only, Synth-Only, or a mix of both), and apply the same SFT data to three backbone scales.

Validity of embedded evidence. The supporting triples of every retained question reproduce its answer under SPARQL execution, and the redundancy check misses a shortcut in only 1.5% of 200 audited questions. This confirms that the embedded triples are a valid reward target without annotation.

<table><tr><td rowspan="2">Methods</td><td colspan="3">WikiWebQuestions</td><td colspan="3">GTSQA</td></tr><tr><td>Hit@1</td><td>RetQ</td><td>Cov</td><td>Hit@1</td><td>RetQ</td><td>Cov</td></tr><tr><td>PoG</td><td>71.89</td><td>75.66</td><td>69.85</td><td>39.33</td><td>35.70</td><td>25.38</td></tr><tr><td>ToG</td><td>72.75</td><td>73.53</td><td>68.79</td><td>42.36</td><td>50.99</td><td>37.34</td></tr><tr><td>KG-R1</td><td>73.01</td><td>82.28</td><td>74.50</td><td>47.73</td><td>61.41</td><td>54.05</td></tr><tr><td>FoG-4B</td><td>76.93</td><td>86.35</td><td>78.34</td><td>50.52</td><td>65.34</td><td>58.64</td></tr></table>

Table 4: Answer accuracy and evidence coverage of agent-based methods.

Synthetic vs. original data. Synth-Only outperforms Train-Only by +3.66, +2.50, and +3.06 Hit@1 with no in-domain question, indicating that questions built by backward expansion transfer to human-written benchmarks. Mixed performs best, so the two sources are complementary (Appendix C.3).

Performance across model scales. Hit@1 improves steadily from 3B to 7B (+12.89, +7.52, and +7.36), indicating that the synthetic data is not backbone-specific.

## 4.5 Analysis of KG Interaction Agent (RQ4)

As shown in Table 4 and Figure 5, we examine how much evidence agent-based methods retrieve.

Answers without evidence. On GTSQA, PoG attains 39.33 Hit@1 but reaches the gold answer on only 35.70% of questions, so at least 9.2% of its correct answers never appear in any KG observation, and the RL-trained KG-R1 still leaves 45.95% of the supporting triples unretrieved. Neither prompting nor outcome rewards thus make an agent use the KG as a source of evidence.

Evidence coverage. FoG attains the highest Cov and RetQ on both benchmarks (+3.84 and +4.59 Cov over KG-R1) together with the highest Hit@1, indicating that evidence coverage does not come at the cost of answer accuracy (Appendix I).

Performance under structural complexity. FoG leads KG-R1 on all 10 groups and is the best method on 9. Its margin over KG-R1 grows from +1.4 Hit@1 on (1) to +5.0 on (3) and +6.0 on the more branched (3)(1), questions that depend on more supporting triples. This suggests that the more evidence an answer requires, the more the agent benefits from rewards for retrieving it, although the margin narrows to +2.0 on (4).

![](images/2cda9a388007d07f191d941f9df840f5f1d1a342601b0b341a387ca903f9b634.jpg)

![](images/e4155cacd8e3be4d0ff822f90f19b057d4f028a68104a5290f9c44f48faadbe1.jpg)

![](images/41cd7d293146f8d166091d7e66c5d7286b51d2b3b9e4d2d2500dd31c51b4dde1.jpg)

Figure 5: Performance under structural complexity on GTSQA by reasoning hop and isomorphism type.  
![](images/4462ddce33987bff0b6b0949521a43e283b62f8621d1092019dc7999e307b7ba.jpg)

![](images/744359cb0d63773eeb80c9bea9c8ff9b1072b798afa166c2212f883b8baba3b5.jpg)

![](images/825337388d7f8840ff0d297a1605dc98cf4fa6732ab48fedc055467f74a0e55e.jpg)

![](images/c238c43992507ebd49c911c11b360ff066a515cbc620babbf5bc1726975c6004.jpg)

Figure 6: Training dynamics of FoG. Rewards rise while interaction turns and response length decline.
<table><tr><td>Credit assignment</td><td>Hit@1</td><td>Cov</td><td>Dup. rate</td></tr><tr><td>Trajectory-level</td><td>46.69</td><td>54.29</td><td>11.8%</td></tr><tr><td>Every-match</td><td>47.01</td><td>52.21</td><td>29.4%</td></tr><tr><td>First-match (ours)</td><td>50.52</td><td>58.64</td><td>8.6%</td></tr></table>

Table 5: Credit-assignment strategies for ECPR on GTSQA. Dup. rate is the fraction of KG queries that re-retrieve an already retrieved gold triple.

## 4.6 Analysis of ECPR (RQ5)

As shown in Table 5 and Figure 6, we compare first-match with trajectory-level attribution, which credits coverage to the final turn, and every-match attribution, which rewards every retrieval.

Comparison across attribution strategies. First-match achieves the best Hit@1 and Cov. Trajectory-level attribution loses 4.35 Cov, as turns that retrieve evidence earn the same return as the others. Every-match raises the duplicate-retrieval rate from 8.6% to 29.4% and falls below trajectorylevel in Cov, as it rewards re-querying known evidence. This indicates that dense credit must pay each triple once, or it rewards redundant retrieval rather than coverage.

Training dynamics. Rewards rise during RL while KG turns and response length decline, indicating that the agent learns a precise retrieval policy that covers more evidence with fewer queries.

<table><tr><td>Method</td><td>Train KG</td><td>Backbone</td><td>WebQSP train</td><td>WebQSP Hit@1</td></tr><tr><td>ToG (Sun et al., 2024)</td><td>None</td><td>GPT-4</td><td>No</td><td>82.6</td></tr><tr><td>RoG (Luo et al., 2024)</td><td>Freebase</td><td>LLaMA2-Chat-7B</td><td>Yes</td><td>85.7</td></tr><tr><td>KG-R1 (Lin et al., 2025)</td><td>Freebase</td><td>Qwen2.5-3B-Instruct</td><td>Yes</td><td>82.8</td></tr><tr><td>FoG SFT-only</td><td>Wikidata</td><td>Qwen3-4B-Instruct</td><td>No</td><td>71.6</td></tr><tr><td>FoG full</td><td>Wikidata</td><td>Qwen3-4B-Instruct</td><td>No</td><td>77.2</td></tr></table>

Table 6: Zero-shot cross-KG transfer to Freebase (WebQSP). Methods developed on Freebase are reference points, not matched baselines.

## 4.7 Cross-KG Transfer (RQ6)

As shown in Table 6, we evaluate the frozen FoG zero-shot on WebQSP over a Freebase backend.

Zero-shot transfer. FoG reaches 77.2 Hit@1 on an unseen KG, close to the methods developed on Freebase (82.6–85.7), and RL with ECPR contributes +5.6 Hit@1 over SFT only. This suggests that ECPR teaches the habit of retrieving evidence before answering, not the relations of one schema.

## 5 Conclusion

In this work, we present FoG, a framework that trains KG agents to retrieve the evidence behind their answers. By introducing evidence-grounded synthesis, a KG interaction agent, and an evidencecoverage process reward, FoG uses the KG as a source of evidence rather than only as a source of answers. Experiments on three KGQA benchmarks demonstrate consistent improvements in answer accuracy, evidence coverage, and cross-KG transfer.

## Limitations

KG completeness and freshness. FoG uses Wikidata triples as verifiable references for evidence-covered supervision, but real-world knowledge graphs may be incomplete, outdated, or noisy. When the required evidence is absent or incorrect in the graph, the agent may still fail despite following the intended evidence path. A natural extension is to equip the agent with a mechanism for detecting insufficient on-graph evidence and invoking external retrieval when needed.

Dependence on teacher models. Like other KG agents distilled from teacher trajectories (Chen et al., 2025; Sun et al., 2026a), FoG currently relies on an external teacher LLM (Kimi-K2) for question synthesis and trajectory distillation. Although symbolic checks filter invalid questions and evidence paths, the synthesis process still depends on the generation quality of external teachers. One promising direction is to replace teacher-driven synthesis with self-play, where agents generate, explore, and verify candidate questions through graph execution and symbolic filtering.

Long-horizon graph interaction. FoG caps graph interactions at 10 turns, which is sufficient for the 1–4-hop questions studied in our benchmarks. More open-ended KG reasoning may require longer evidence chains and broader graph exploration. Extending FoG to such settings requires balancing search depth against error accumulation, noisy observations, and sparser rewards.

## Ethical Considerations

All experiments use publicly available KGQA datasets derived from Wikidata (CC0), QALD10- en (MIT License), and WikiWebQuestions, GT-SQA, and WebQSP (publicly available for research). Backbone models Qwen2.5 and Qwen3 are released under Apache 2.0; GPT-4o-mini, DeepSeek-V3, and the teacher Kimi-K2 are accessed via official APIs under their terms of service. Our use of all existing artifacts is consistent with their intended research purposes. The datasets contain only public encyclopedic facts from structured KG triples, with no private or user-generated data. The only human involvement is the audit of constructed evidence paths (Appendix F.3), in which three graduate annotators judged whether a KG path supports its question. The annotators took part voluntarily, the task exposed them to no sensitive content, and no personal data was collected. This work involves no crowdsourcing and no user study. FoG is intended to make the answers of KG agents traceable to retrieved evidence. Since the KG may be outdated or incorrect, this evidence supports human verification rather than replaces it. We foresee no dual-use concerns.

## References

Garima Agrawal, Tharindu Kumarage, Zeyad Alghamdi, and Huan Liu. 2024. Mindful-rag: A study of points of failure in retrieval augmented generation. In 2nd International Conference on Foundation and Large Language Models, FLLM 2024, Dubai, United Arab Emirates, November 26-29, 2024, pages 607–611. IEEE.

Aisha Alansari and Hamzah Luqman. 2026. Large language models hallucination: A comprehensive survey. Comput. Sci. Rev., 61:100970.

Laura Cabello, Carmen Martin-Turrero, Uchenna Akujuobi, Anders Søgaard, and Carlos Bobed. 2024. MEG: medical knowledge-augmented large language models for question answering. CoRR, abs/2411.03883.

Alberto Cattaneo, Carlo Luschi, and Daniel Justus. 2026. Ground-truth subgraphs for better training and evaluation of knowledge graph augmented llms. Trans. Mach. Learn. Res., 2026.

Liyi Chen, Panrong Tong, Zhongming Jin, Ying Sun, Jieping Ye, and Hui Xiong. 2024. Plan-on-graph: Self-correcting adaptive planning of large language model on knowledge graphs. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Zhuo Chen, Fei Wang, Zixuan Li, Zhao Zhang, Weiwei Ding, Chuanguang Yang, YongJun Xu, Xiaolong Jin, and Jiafeng Guo. 2025. Knowcoder-a1: Incentivizing agentic reasoning capability with outcome supervision for KBQA. CoRR, abs/2510.25101.

DeepSeek-AI. 2024. Deepseek-v3 technical report. CoRR, abs/2412.19437.

Ritam Dutt, Sopan Khosla, Vinayshekhar Bannihatti Kumar, and Rashmi Gangadharaiah. 2023. Grailqa++: A challenging zero-shot benchmark for knowledge base question answering. In Proceedings ofthe 13th International Joint Conference on Natural Language Processing and the 3rd Conference ofthe Asia-Pacific Chapter of the Association for Computational Linguistics, IJCNLP 2023 -Volume 1: Long Papers, Nusa Dua, Bali, November 1 - 4, 2023, pages 897– 909. Association for Computational Linguistics.

GLM. 2025. GLM-4.5: agentic, reasoning, and coding (ARC) foundation models. CoRR, abs/2508.06471.

Yu Gu, Sue Kase, Michelle Vanni, Brian Sadler, Percy Liang, Xifeng Yan, and Yu Su. 2021. Beyond iid: three levels of generalization for question answering on knowledge bases. In Proceedings of the web conference 2021, pages 3477–3488.

Haoyu Han, Kai Guo, Harry Shomer, Yu Wang, Yucheng Chu, Hang Li, Li Ma, and Jiliang Tang. 2026. Reasoning by exploration: A unified approach to retrieval and generation over graphs. In Proceedings ofthe ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduled for April 13-17, 2026, rescheduledfor June 29 - July 3, 2026, pages 2408–2418. ACM.

Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, and Bryan Hooi. 2024. G-retriever: Retrieval-augmented generation for textual graph understanding and question answering. In Advances in Neural Information Processing Systems 37: Annual Conference on Neural Information Processing Systems 2024, NeurIPS 2024, Vancouver, BC, Canada, December 10 - 15, 2024.

Peng Hu, Changjiang Gao, Ruiqi Gao, Jiajun Chen, and Shujian Huang. 2024. Large language models are limited in out-of-context knowledge reasoning. In Findings of the Association for Computational Linguistics: EMNLP 2024, Miami, Florida, USA, November 12-16, 2024, volume EMNLP 2024 of Findings ofACL, pages 3144–3155. Association for Computational Linguistics.

Lei Huang, Weijiang Yu, Weitao Ma, Weihong Zhong, Zhangyin Feng, Haotian Wang, Qianglong Chen, Weihua Peng, Xiaocheng Feng, Bing Qin, and Ting Liu. 2025. A survey on hallucination in large language models: Principles, taxonomy, challenges, and open questions. ACM Trans. Inf. Syst., 43(2):42:1– 42:55.

Jinhao Jiang, Kun Zhou, Xin Zhao, Yang Song, Chen Zhu, Hengshu Zhu, and Ji-Rong Wen. 2025. Kgagent: An efficient autonomous agent framework for complex reasoning over knowledge graph. In Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers), ACL 2025, Vienna, Austria, July 27 - August 1, 2025, pages 9505–9523. Association for Computational Linguistics.

Bowen Jin, Chulin Xie, Jiawei Zhang, Kashob Kumar Roy, Yu Zhang, Zheng Li, Ruirui Li, Xianfeng Tang, Suhang Wang, Yu Meng, and Jiawei Han. 2024. Graph chain-of-thought: Augmenting large language models by reasoning on graphs. In Findings ofthe Associationfor Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting, August 11- 16, 2024, volume ACL 2024 of Findings of ACL, pages 163–184. Association for Computational Linguistics.

Maël Jullien, Marco Valentino, and André Freitas. 2025. The knowledge-reasoning dissociation: Fundamen-

tal limitations of llms in clinical natural language inference. CoRR, abs/2508.10777.

Mufei Li, Siqi Miao, and Pan Li. 2025. Simple is effective: The roles of graphs and large language models in knowledge-graph-based retrieval-augmented generation. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Junhong Lin, Shicheng Liu, Jinyeop Song, Song Wang, Julian Shun, and Yada Zhu. 2025. Efficient and transferable agentic knowledge graph rag via reinforcement learning. arXiv preprint arXiv:2509.26383.

Yu Liu, Xixun Lin, Yanmin Shang, Yangxi Li, Shi Wang, and Yanan Cao. 2026. Pathmind: A retrieveprioritize-reason framework for knowledge graph reasoning with large language models. In Fortieth AAAI Conference on Artificial Intelligence, Thirty-Eighth Conference on Innovative Applications of Artificial Intelligence, Sixteenth Symposium on Educational Advances in Artificial Intelligence, AAAI 2026, Singapore, January 20-27, 2026, pages 15386–15393. AAAI Press.

Linhao Luo, Yuan-Fang Li, Gholamreza Haffari, and Shirui Pan. 2024. Reasoning on graphs: Faithful and interpretable large language model reasoning. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. OpenReview.net.

Linhao Luo, Zicheng Zhao, Gholamreza Haffari, Yuan-Fang Li, Chen Gong, and Shirui Pan. 2025. Graphconstrained reasoning: Faithful reasoning on knowledge graphs with large language models. In Fortysecond International Conference on Machine Learning, ICML 2025, Vancouver, BC, Canada, July 13-19, 2025, volume 267 of Proceedings ofMachine Learning Research. PMLR / OpenReview.net.

Costas Mavromatis and George Karypis. 2025. GNN-RAG: graph neural retrieval for efficient large language model reasoning on knowledge graphs. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, volume ACL 2025 of Findings of ACL, pages 16682–16699. Association for Computational Linguistics.

Iman Mirzadeh, Keivan Alizadeh, Hooman Shahrokhi, Oncel Tuzel, Samy Bengio, and Mehrdad Farajtabar. 2025. Gsm-symbolic: Understanding the limitations of mathematical reasoning in large language models. In The Thirteenth International Conference on Learning Representations, ICLR 2025, Singapore, April 24-28, 2025. OpenReview.net.

Thi Nguyen, Linhao Luo, Fatemeh Shiri, Dinh Phung, Yuan-Fang Li, Thuy-Trang Vu, and Gholamreza Haffari. 2024. Direct evaluation of chain-of-thought in multi-hop reasoning with knowledge graphs. In Findings of the Association for Computational Linguistics, ACL 2024, Bangkok, Thailand and virtual meeting,

August 11-16, 2024, volume ACL 2024 of Findings of ACL, pages 2862–2883. Association for Computational Linguistics.

OpenAI. 2023. GPT-4 technical report. CoRR, abs/2303.08774.

Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, and Xindong Wu. 2024. Unifying large language models and knowledge graphs: A roadmap. IEEE Trans. Knowl. Data Eng., 36(7):3580–3599.

Fabio Petroni, Aleksandra Piktus, Angela Fan, Patrick Lewis, Majid Yazdani, Nicola De Cao, James Thorne, Yacine Jernite, Vladimir Karpukhin, Jean Maillard, Vassilis Plachouras, Tim Rocktäschel, and Sebastian Riedel. 2021. KILT: a benchmark for knowledge intensive language tasks. In Proceedings of the 2021 Conference of the North American Chapter of the Associationfor Computational Linguistics: Human Language Technologies, NAACL-HLT 2021, Online, June 6-11, 2021, pages 2523–2544. Association for Computational Linguistics.

Zhihong Shao, Peiyi Wang, Qihao Zhu, Runxin Xu, Junxiao Song, Mingchuan Zhang, Y. K. Li, Y. Wu, and Daya Guo. 2024. Deepseekmath: Pushing the limits of mathematical reasoning in open language models. CoRR, abs/2402.03300.

Guangming Sheng, Chi Zhang, Zilingfeng Ye, Xibin Wu, Wang Zhang, Ru Zhang, Yanghua Peng, Haibin Lin, and Chuan Wu. 2025. Hybridflow: A flexible and efficient RLHF framework. In Proceedings of the Twentieth European Conference on Computer Systems, EuroSys 2025, Rotterdam, The Netherlands, 30 March 2025 - 3 April 2025, pages 1279–1297. ACM.

Yanlin Song, Ben Liu, Víctor Gutiérrez-Basulto, Zhiwei Hu, Qianqian Xie, Min Peng, Sophia Ananiadou, and Jeff Z. Pan. 2026. Plan then retrieve: Reinforcement learning-guided complex reasoning over knowledge graphs. In Proceedings of the ACM Web Conference 2026, WWW 2026, Dubai, United Arab Emirates, originally scheduledfor April 13-17, 2026, rescheduled for June 29 - July 3, 2026, pages 3666–3676. ACM.

Yuan Sui, Yufei He, Nian Liu, Xiaoxin He, Kun Wang, and Bryan Hooi. 2025. Fidelis: Faithful reasoning in large language models for knowledge graph question answering. In Findings of the Association for Computational Linguistics, ACL 2025, Vienna, Austria, July 27 - August 1, 2025, volume ACL 2025 of Findings ofACL, pages 8315–8330. Association for Computational Linguistics.

Jia Ao Sun, Hao Yu, Fengran Mo, Zhan Su, Yuchen Hui, Bang Liu, and Jian-Yun Nie. 2026a. Searchon-graph-r1: Training large language models to search knowledge graphs with reinforcement learning. CoRR, abs/2607.18481.

Jiashuo Sun, Chengjin Xu, Lumingyuan Tang, Saizhuo Wang, Chen Lin, Yeyun Gong, Lionel M. Ni, Heung-Yeung Shum, and Jian Guo. 2024. Think-on-graph: Deep and responsible reasoning of large language model on knowledge graph. In The Twelfth International Conference on Learning Representations, ICLR 2024, Vienna, Austria, May 7-11, 2024. Open-Review.net.

Tianda Sun and Dimitar Kazakov. 2026. Peakthen-collapse and the four interface channels of knowledge-graph tool use. CoRR, abs/2605.26037.

Xin Sun, Zhongqi Chen, Xing Zheng, Qiang Liu, Shu Wu, Bowen Song, Zilei Wang, Weiqiang Wang, and Liang Wang. 2025. KBQA-R1: reinforcing large language models for knowledge base question answering. CoRR, abs/2512.10999.

Xin Sun, Jianan Xie, Zhongqi Chen, Qiang Liu, Shu Wu, Bowen Song, Weiqiang Wang, Zilei Wang, and Liang Wang. 2026b. GAPD: gold-action policy distillation for agentic reinforcement learning in knowledge base question answering. CoRR, abs/2605.29584.

Alon Talmor and Jonathan Berant. 2018. The web as a knowledge-base for answering complex questions. In Proceedings ofthe 2018 Conference ofthe North American Chapter of the Association for Computational Linguistics: Human Language Technologies, NAACL-HLT 2018, New Orleans, Louisiana, USA, June 1-6, 2018, Volume 1 (Long Papers), pages 641– 651. Association for Computational Linguistics.

Kimi Team. 2025a. Kimi K2: open agentic intelligence. CoRR, abs/2507.20534.

Qwen Team. 2025b. Qwen3 technical report. CoRR, abs/2505.09388.

Hugo Touvron, Thibaut Lavril, Gautier Izacard, Xavier Martinet, Marie-Anne Lachaux, Timothée Lacroix, Baptiste Rozière, Naman Goyal, Eric Hambro, Faisal Azhar, Aurélien Rodriguez, Armand Joulin, Edouard Grave, and Guillaume Lample. 2023. Llama: Open and efficient foundation language models. CoRR, abs/2302.13971.

Ricardo Usbeck, Xi Yan, Aleksandr Perevalov, Longquan Jiang, Julius Schulz, Angelie Kraft, Cedric Möller, Junbo Huang, Jan Reineke, Axel-Cyrille Ngonga Ngomo, Muhammad Saleem, and Andreas Both. 2024. QALD-10 - the 10th challenge on question answering over linked data: Shifting from dbpedia to wikidata as a KG for KGQA. Semantic Web, 15(6):2193–2207.

Shuai Wang and Yinan Yu. 2026a. Kg-hopper: Empowering compact open llms with knowledge graph reasoning via reinforcement learning. CoRR, abs/2603.21440.

Shuai Wang and Yinan Yu. 2026b. Kg-reasoner: A reinforced model for end-to-end multi-hop knowledge graph reasoning. CoRR, abs/2604.12487.

Xuezhi Wang, Jason Wei, Dale Schuurmans, Quoc V. Le, Ed H. Chi, Sharan Narang, Aakanksha Chowdhery, and Denny Zhou. 2023. Self-consistency improves chain of thought reasoning in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Jason Wei, Xuezhi Wang, Dale Schuurmans, Maarten Bosma, Brian Ichter, Fei Xia, Ed H. Chi, Quoc V. Le, and Denny Zhou. 2022. Chain-of-thought prompting elicits reasoning in large language models. In Advances in Neural Information Processing Systems 35: Annual Conference on Neural Information Processing Systems 2022, NeurIPS 2022, New Orleans, LA, USA, November 28 - December 9, 2022.

Shuwen Xu, Yao Xu, Jiaxiang Liu, Chenhao Yuan, Wenshuo Peng, Jun Zhao, and Kang Liu. 2026. Graphwalker: Agentic knowledge graph question answering via synthetic trajectory curriculum. CoRR, abs/2603.28533.

Silei Xu, Shicheng Liu, Theo Culhane, Elizaveta Pertseva, Meng-Hsi Wu, Sina J. Semnani, and Monica S. Lam. 2023. Fine-tuned llms know more, hallucinate less with few-shot sequence-to-sequence semantic parsing over wikidata. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, EMNLP 2023, Singapore, December 6-10, 2023, pages 5778–5791. Association for Computational Linguistics.

Shiqi Yan, Yubo Chen, Ruiqi Zhou, Zhengxi Yao, Shuai Chen, Tianyi Zhang, Shijie Zhang, Wei Qiang Zhang, Yongfeng Huang, Haixin Duan, and Yunqi Zhang. 2026. Explore-on-graph: Incentivizing autonomous exploration of large language models on knowledge graphs with path-refined reward modeling. In The Fourteenth International Conference on Learning Representations, ICLR 2026, Rio de Janeiro, Brazil, April 23-27, 2026. proceedings.iclr.cc.

Shunyu Yao, Dian Yu, Jeffrey Zhao, Izhak Shafran, Tom Griffiths, Yuan Cao, and Karthik Narasimhan. 2023a. Tree of thoughts: Deliberate problem solving with large language models. In Advances in Neural Information Processing Systems 36: Annual Conference on Neural Information Processing Systems 2023, NeurIPS 2023, New Orleans, LA, USA, December 10 - 16, 2023.

Shunyu Yao, Jeffrey Zhao, Dian Yu, Nan Du, Izhak Shafran, Karthik R. Narasimhan, and Yuan Cao. 2023b. React: Synergizing reasoning and acting in language models. In The Eleventh International Conference on Learning Representations, ICLR 2023, Kigali, Rwanda, May 1-5, 2023. OpenReview.net.

Wen-tau Yih, Matthew Richardson, Christopher Meek, Ming-Wei Chang, and Jina Suh. 2016. The value of semantic parse labeling for knowledge base question answering. In Proceedings ofthe 54th Annual Meeting of the Association for Computational Linguistics,

ACL 2016, August 7-12, 2016, Berlin, Germany, Volume 2: Short Papers. The Association for Computer Linguistics.

Liangliang Zhang, Zhuorui Jiang, Hongliang Chi, Haoyang Chen, Mohammed Elkoumy, Fali Wang, Qiong Wu, Zhengyi Zhou, Shirui Pan, Suhang Wang, and Yao Ma. 2025. Diagnosing and addressing pitfalls in KG-RAG datasets: Toward more reliable benchmarking. In Advances in Neural Information Processing Systems 38: Annual Conference on Neural Information Processing Systems 2025, NeurIPS 2025, San Diego, CA, USA, December 2-7, 2025 / Mexico City, Mexico, November 30 - December 5, 2025.

Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. 2024. Llamafactory: Unified efficient fine-tuning of 100+ language models. In Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), ACL 2024, Bangkok, Thailand, August 11-16, 2024, pages 400–410. Association for Computational Linguistics.

## A Proofs

Throughout the proofs, a logical form ℓ is a conjunctive query: a set of triple patterns over entities and variables with one answer variable, optionally with filters on bound values. For a triple set $s ,$ SPARQL(ℓ, S) is the set of values of the answer variable over all assignments that map every pattern of $\ell$ to a triple in $\boldsymbol { \mathcal { S } }$ and satisfy the filters. Such queries are monotone: ${ \mathcal { S } } \subseteq S ^ { \prime }$ implies $\mathrm { S P A R Q L } ( \ell , \mathcal { S } ) \ \subseteq \ \mathrm { S P A R Q L } ( \ell , \mathcal { S } ^ { \prime } )$ Following Definition 2, S supports $( \ell , y )$ if $y \in$ $\operatorname { S P A R Q L } ( \ell , S )$ , and a trajectory τ is evidencecovered if Cov $( \tau , { \cal S } _ { q } ^ { * } ) = 1$ , as defined in Equation 6.

## A.1 Proof of Proposition 1

Proposition 1. Backward expansion preserves the answer and yields triples that support it.

Proof. We show by induction that, for every $n \geq 1$ the triple set $S ^ { ( n ) } \subseteq \tau$ supports $( \ell ^ { ( n ) } , y )$

Base case. A seed is retained only if its supporting triples exist in the KG and instantiate $\ell ^ { ( 1 ) }$ with the answer variable bound to $y .$ Hence ${ \mathcal { S } } ^ { ( 1 ) }$ supports $( \ell ^ { ( 1 ) } , y )$ under some assignment $\mu .$

Inductive step. Assume that ${ \mathcal { S } } ^ { ( n ) }$ supports $( \ell ^ { ( n ) } , y )$ under an assignment $\mu .$ . An expansion selects an entity $u$ in $\ell ^ { ( n ) }$ and a verified sub-question with logical form $\tilde { \ell } ,$ answer variable $x ,$ and triples $\tilde { S } \subseteq \mathcal { T }$ that support $( \tilde { \ell } , u )$ under an assignment $\tilde { \mu }$ with $\tilde { \mu } ( x ) = u$ . With variables renamed apart, the expanded logical form is

$$
\ell ^ { ( n + 1 ) } = \ell ^ { ( n ) } [ u \mapsto x ] \cup \tilde { \ell } ,\tag{9}
$$

where $[ u \mapsto x ]$ replaces every occurrence of u with the variable x. Consider the assignment $\mu \cup \tilde { \mu }$ Every pattern of $\tilde { \ell }$ is mapped into ${ \tilde { S } } ,$ , and every pattern of $\ell ^ { ( n ) } [ u \mapsto x ]$ is mapped to the same triple of $S ^ { ( n ) }$ as under $\mu ,$ since $\tilde { \mu } ( x ) = u$ . The answer variable is still bound to $y ,$ which gives

$$
y \in \mathrm { S P A R Q L } \big ( \ell ^ { ( n + 1 ) } , S ^ { ( n + 1 ) } \big ) ,\tag{10}
$$

where $\begin{array} { r } { S ^ { ( n + 1 ) } = S ^ { ( n ) } \cup \tilde { S } , \mathrm { ~ s o ~ } S ^ { ( n + 1 ) } } \end{array}$ supports $( \ell ^ { ( n + 1 ) } , y )$ . Since $S ^ { ( n + 1 ) } \subseteq { \mathcal { T } } $ , monotonicity also gives $y \in \mathrm { S P A R Q L } ( \ell ^ { ( n + 1 ) } , \mathcal { T } )$ , so the answer is preserved over the full KG. Each expansion replaces one entity with a variable, so $N$ expansions hide N intermediate entities.

The proof guarantees that the accumulated triples are sufficient for the answer, not that they are the only triples that support it. In the special case where the sub-question has answers other than u, the result set of $\bar { \ell } ^ { ( n + 1 ) }$ may grow, but the executability check bounds it by $K _ { \mathrm { a n s } }$

In conclusion, every expansion keeps $y$ in the result set over the full KG, and the accumulated triples alone support the expanded logical form. Therefore, backward expansion preserves the answer and yields triples that support it. □

## A.2 Proof of Proposition 2

Proposition 2. ECPR is maximized only by supported answers and rewards each triple once.

Proof. Let $R = R _ { \mathrm { o u t } } + R _ { \mathrm { s t e p } } + R _ { \mathrm { e c p r } }$ be the composite reward with $\lambda _ { \mathrm { { e c p r } } } > 0 .$ , and let $S _ { q } ^ { * }$ be the supporting triples of a question $q ,$ which support $( \ell _ { q } , y _ { q } ^ { * } )$ by Proposition 1.

Supported maximizers. Since ${ \mathrm { F 1 } } \leq 1 , R _ { \mathrm { s t e p } } \leq$ $\lambda _ { \mathrm { s t e p } } .$ , and $\mathrm { C o v } ~ \leq ~ 1$ , the composite reward is bounded as

$$
R ( \tau ) \leq \lambda _ { \mathrm { f l } } + \lambda _ { \mathrm { r e t } } + \lambda _ { \mathrm { s t e p } } + \lambda _ { \mathrm { e c p r } } ,\tag{11}
$$

where equality holds if and only if every term attains its maximum, in particular only if $\mathrm { C o v } ( \tau , \mathcal { S } _ { q } ^ { * } ) = 1$ . Assume that an evidence-covered trajectory with a correct answer and only valid actions is feasible within the turn limit. The bound is then attained, so every trajectory $\tau$ that maximizes $R$ satisfies $S _ { q } ^ { * } \subseteq { \mathcal { T } } ( \tau )$ , and monotonicity gives

$$
y _ { q } ^ { * } \in \mathrm { S P A R Q L } ( \ell _ { q } , \mathcal { T } ( \tau ) ) ,\tag{12}
$$

which means that the answer of $\tau$ is supported.

For outcome rewards alone, $R _ { \mathrm { o u t } }$ in Equation $5$ depends on τ only through the prediction $\hat { y }$ and the returned entities $\mathcal { E } ( \tau )$ . A shortcut trajectory $\tau _ { s }$ that predicts and reaches $y _ { q } ^ { * }$ with Cov $( \tau _ { s } , S _ { q } ^ { * } ) <$ 1 therefore attains the maximum $\lambda _ { \mathrm { f } 1 } + \lambda _ { \mathrm { r e t } }$ of $R _ { \mathrm { o u t } }$ . Under the composite reward, it loses at least $\lambda _ { \mathrm { e c p r } } \bigl ( 1 - \mathrm { C o v } ( \tau _ { s } , \mathcal { S } _ { q } ^ { * } ) \bigr ) > 0$ against the bound and is no longer optimal.

One reward per triple. Let $\begin{array} { r } { U _ { t } = \left( \bigcup _ { i < t } \mathcal { T } _ { i } \right) \cap S _ { q } ^ { * } } \end{array}$ be the gold triples retrieved up to step t, with $U _ { 0 } =$ ∅. The step reward of ECPR is then

$$
r _ { t } ^ { \mathrm { e c p r } } = \lambda _ { \mathrm { e c p r } } \frac { | U _ { t } | - | U _ { t - 1 } | } { | S _ { q } ^ { * } | } ,\tag{13}
$$

where $U _ { t - 1 } \subseteq U _ { t }$ , so $r _ { t } ^ { \mathrm { e c p r } } \geq 0$ and the sum over steps telescopes to $\lambda _ { \mathrm { e c p r } } | U _ { T } | / | S _ { q } ^ { * } | = R _ { \mathrm { e c p r } } ( \tau )$ . If step t returns only triples that were already retrieved, then $U _ { t } = U _ { t - 1 }$ and $r _ { t } ^ { \mathrm { e c p r } } = 0$ . In contrast, every-match attribution pays $\lambda _ { \mathrm { e c p r } } | \mathcal { T } _ { t } \cap \mathcal { S } _ { q } ^ { * } | / | \mathcal { S } _ { q } ^ { * } |$ at each step. Repeating a query that returns m $\geq 1$ gold triples for T steps thus earns $\lambda _ { \mathrm { e c p r } } T m / | S _ { q } ^ { * } |$ which grows with $T$ while the coverage stays at $m / | S _ { q } ^ { * } |$ |.

The statement concerns the maximizers of the trajectory-level reward and does not guarantee that policy optimization reaches them. Covering $S _ { q } ^ { * }$ is sufficient but not necessary for a supported answer, since a question may admit other supporting sets. In the special case where no shortcut trajectory exists for the question, the maximizers of $R _ { \mathrm { { o u t } } }$ are already evidence-covered, so the two rewards share the same maximizers.

In conclusion, the composite reward attains its maximum only when the retrieved triples support the answer, and first-match attribution never pays a triple twice. Therefore, ECPR is maximized only by supported answers and rewards each triple once. □

## B FoG Algorithm Detail

Redundancy Check. To provide a clear overview of the non-redundancy check in Section 3.1, we present its procedure in the form of pseudocode. As shown in Algorithm 1, we first build a support graph from the supporting triples and extract the shortest path from each topic entity to the answer node. We then enumerate proper subsets of topic entities in increasing size and execute one SPARQL query per subset. If any subset yields an answer set no larger than that of the full query, the question admits a shortcut and is discarded.

Complexity Analysis. Given a question with $| E _ { q } |$ topic entities and $| S ^ { ( n ) }$ | supporting triples, extracting the shortest paths by BFS takes $O ( | E _ { q } | \cdot | S ^ { ( n ) } | )$ time. The subset enumeration issues at most 2<sup>|Eq|</sup> − 2 SPARQL queries and stops at the first redundant subset. Using only the shortest path per topic entity is an approximation, which we audit on 200 candidate questions against an exhaustive enumeration of all path combinations under the same hop limit. The shortest-path check misses a shortcut in 1.5% of the candidates, while issuing 5.1 rather than 48.6 SPARQL queries per question.

In summary, the redundancy check is dominated by SPARQL execution, and the shortest-path approximation reduces the number of queries by 9.5× while missing few shortcuts.

Algorithm 1: Redundancy Check   
Input: ${ \mathcal { S } } ^ { ( n ) } \colon$ supporting triples; $E _ { q } \mathrm { : }$ topic entities;   
${ \mathcal { V } } ^ { ( n ) }$ : full answer set.   
Output: Redundancy flag   
ISREDUNDANT ∈ {True, False}.   
Build an undirected graph G from ${ \mathcal { S } } ^ { ( n ) } ;$ label   
topic-entity nodes with their Wikidata IDs, the   
answer node with ?answer, and intermediates with   
fresh variables.   
for each $e \in E _ { q }$ do   
Compute shortest path $\mathcal { P } ( e )$ in G from e to the   
answer node.   
Derive SPARQL triple pattern $\phi ( e )$ from triples   
along $\mathcal { P } ( e )$   
end   
for $k = 1$ to $| E _ { q } | - 1$ do   
for each $Z \stackrel { \cdot \cdot } { \subseteq } E _ { q }$ with $| Z | = k$ do   
$Q z $   
SELECT;?answer;WHERE; $\{ \cup _ { e \in Z } \phi ( e ) \}$   
Execute $Q _ { Z }$ on the $\mathrm { K G } $ answer set ${ \mathcal { V } } _ { Z } ^ { ( n ) }$   
if $| \mathcal { V } _ { Z } ^ { ( n ) } | \leq | \mathcal { V } ^ { ( n ) } |$ then   
return TRUE   
end   
end   
end   
return FALSE

## C Synthetic Data Construction

## C.1 Topic Domains

As shown in Figure 7, the synthetic questions used for FoG training span seven named Wikidata domains. Entertainment is the largest (39.7%), reflecting the density of film, television, and music entities in Wikidata, followed by history (12.6%), society (11.4%), geography (9.6%), transportation (6.6%), sports (6.5%), and politics (5.7%); the remaining 7.9% fall into other domains.

## C.2 Query-Graph Topologies

To construct questions with diverse reasoning structures, we label each synthetic question with the isomorphism type of its query graph. As shown in Table 7, the 6,588 questions that pass all checks span nine chain and tree structures from 1 to 3 hops. All verified seeds enter backward expansion, and the 1-hop questions are seeds retained at depth one. Overall, 35.1%, 40.0%, and 24.9% of the questions have 1, 2, and 3 hops. The most frequent types are the three-branch star (1)(1)(1) (18.38%) and the 3-hop chain (3) (16.48%), and 67.6% of the questions have a multi-branch query graph rather than a single chain.

## C.3 Synthesis Cost and Retention

As shown in Table 8, we report the candidates retained, the rejection rate, and the teacher calls at each synthesis stage, all of which use the same teacher model, Kimi-K2 (Team, 2025a). Of 20,000 sampled seed entities, 14,400 yield a verified seed question, 8,784 candidates remain after backward expansion, and 6,588 questions pass all filters, an end-to-end rejection rate of 67.1%. The pipeline uses 115,184 teacher calls, about 17.5 per retained question, and backward expansion accounts for three quarters of them.

![](images/eb896aa6aba58aab63a4d31ad1de24167e8effc5d36e405dec9682d42839bbab.jpg)

Figure 7: Topic-domain distribution of the synthetic questions. The questions are drawn from major Wikidata domains.
<table><tr><td>Hops</td><td>Isomorphism Type</td><td>Questions</td><td>Proportion</td></tr><tr><td rowspan="3">1 hop</td><td>(1)</td><td>500</td><td>7.59%</td></tr><tr><td>(1)(1)</td><td>600</td><td>9.11%</td></tr><tr><td>(1)(1)(1)</td><td>1,211</td><td>18.38%</td></tr><tr><td rowspan="4">2 hop</td><td>(2)</td><td>550</td><td>8.35%</td></tr><tr><td>(2)(1)</td><td>853</td><td>12.95%</td></tr><tr><td>((1)(1))</td><td>700</td><td>10.63%</td></tr><tr><td>(2)(1)(1)</td><td>533</td><td>8.09%</td></tr><tr><td rowspan="2">3 hop</td><td>(3)</td><td>1,086</td><td>16.48%</td></tr><tr><td>((2)(1))</td><td>555</td><td>8.42%</td></tr><tr><td>Total</td><td>-</td><td>6,588</td><td>100.00%</td></tr></table>

Table 7: Topology distribution of the synthetic questions. Query-graph isomorphism types are grouped by hop count, the longest path from a topic entity to the answer.

## C.4 Data Leakage Check

To ensure that the synthetic questions do not overlap with the test sets, we encode each question with BGE-M3 and compute the cosine similarity between every test question and its most similar synthetic question. The similarity stays below 0.85 for all test questions. On QALD10-en and Wiki-WebQuestions, no test question exceeds 0.75, and on GTSQA only 6 questions (0.37%) fall in the 0.80–0.85 range.

<table><tr><td>Stage</td><td>Input</td><td>Retained</td><td>Rejected</td><td>Teacher calls</td></tr><tr><td>Seed-question generation</td><td>20,000</td><td>14,400</td><td>28.0%</td><td>20,000</td></tr><tr><td>Backward expansion</td><td>14,400</td><td>8,784</td><td>39.0%</td><td>86,400</td></tr><tr><td>Complex-question composition</td><td>8,784</td><td>6,588</td><td>25.0%</td><td>8,784</td></tr><tr><td>End-to-end</td><td>20,000</td><td>6,588</td><td>67.1%</td><td>115,184</td></tr></table>

Table 8: Synthesis cost and retention.

## D KG Interaction Operations

As shown in Table 9, the kg-query action in Section 3.2 is instantiated with seven operations in four categories: relation search, neighborhood expansion, constraint filtering, and set combination. The table lists the name, description, input parameters, and output of every operation.

## E Benchmark Datasets

The benchmarks used for FoG evaluation cover two knowledge graphs, with data sourced as follows:

WikiWebQuestions: Ported from WebQuestions and re-annotated with SPARQL queries over Wikidata (Xu et al., 2023), consisting of real-world natural language questions.

QALD10-en: Derived from the English subset of the tenth QALD challenge (Usbeck et al., 2024), the first edition to adopt Wikidata, covering 394 expert-written test questions with multi-hop and qualifier constraints.

GTSQA: Built over Wikidata with 30,144 training and 1,622 test questions (Cattaneo et al., 2026), each annotated with a gold answer subgraph and a query-graph isomorphism label.

WebQSP: Sourced from Freebase (Yih et al., 2016) and used only for the zero-shot cross-KG transfer experiment.

## F Evaluation Details

## F.1 Evaluation Metrics

We evaluate model performance using four complementary metrics that assess two aspects of KGQA: answer accuracy (Hit@1 and F1) and evidence coverage (Cov and RetQ).

(i) Hit@1. Hit@1 quantifies whether the predicted answer is correct. For each question, we check whether the normalized prediction matches any gold answer, and then average over all evalu-

<table><tr><td>Category</td><td>Function</td><td>Description</td><td>Input Parameters</td><td>Output</td></tr><tr><td>Relation search</td><td>find_outgoing_relations</td><td>Given an entity, return all outgoing rela- tions.</td><td>entity_name</td><td>List of outgoing relations</td></tr><tr><td></td><td>find_incoming_relations</td><td>Given an entity, return all incoming rela- tions.</td><td>entity_name</td><td>List of incoming relations</td></tr><tr><td>Neighborhood expansion</td><td>get_tail_entities</td><td>Follow the relation forward from head to tail. Follow the relation backward from tail</td><td>head_entity_name; relation_name tail_entity_name;</td><td>Set of tail entities Set of head enti-</td></tr><tr><td></td><td>get_head_entities</td><td>to head. Filter the provided entity_set and re-</td><td>relation_name</td><td>ties</td></tr><tr><td>Constraint filtering</td><td>get_entity_by_constraint</td><td>turn the entities satisfying the specified constraint.</td><td>entity_set; relation; operator; value</td><td>Entity subset</td></tr><tr><td></td><td></td><td>Return the intersection of the given list</td><td>entity_set_list</td><td>Intersection set</td></tr><tr><td></td><td>intersect</td><td>Return the union of the given list of en-</td><td></td><td></td></tr><tr><td>Set combination</td><td></td><td>of entity sets.</td><td></td><td></td></tr></table>

Table 9: The seven KG operations of the kg-query action. Each operation is listed with its description, input parameters, and output.

ated questions:

$$
\mathrm { H i t @ 1 } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { I } \big [ \exists y \in Y _ { i } ^ { * } :  \\  \mathrm { n o r m } ( \hat { y } _ { i } ) = \mathrm { n o r m } ( y ) \big ] ,\tag{14}
$$

where $\hat { y } _ { i }$ and $Y _ { i } ^ { * }$ denote the prediction and the gold answer set of the i-th question, norm(·) removes superficial differences such as case and punctuation, and M is the total number of evaluated questions in the benchmark.

(ii) Token-level F1. F1 quantifies the token-level overlap between the predicted and gold answers. For each question, we compute the harmonic mean of token precision and recall, and then average:

$$
\mathrm { F 1 } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { 2 \cdot | \mathrm { t o k e n s } ( \hat { y } _ { i } ) \cap \mathrm { t o k e n s } ( Y _ { i } ^ { \ast } ) | } { | \mathrm { t o k e n s } ( \hat { y } _ { i } ) | + | \mathrm { t o k e n s } ( Y _ { i } ^ { \ast } ) | } ,\tag{15}
$$

where tokens(·) normalizes each answer in a list, splits it on whitespace, and takes the union of the resulting tokens over the list.

(iii) Evidence Coverage (Cov). Cov quantifies how much of the gold supporting evidence the agent retrieves along its trajectory. For each question, we compute the fraction of gold supporting triples returned by its KG queries, and then average over all questions:

$$
\mathrm { C o v } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \frac { | T ( \tau _ { i } ) \cap S _ { i } ^ { * } | } { | S _ { i } ^ { * } | } ,\tag{16}
$$

where $\mathcal T ( \tau _ { i } )$ denotes the set of triples returned by all kg-query actions in the trajectory $\tau _ { i } .$ , and $\boldsymbol { S } _ { i } ^ { * }$ denotes the gold supporting triples of the i-th question. This metric reflects whether an answer is backed by the complete evidence path.

(iv) Retrieval Grounding (RetQ). RetQ quantifies whether the gold answer is retrieved from the KG rather than produced from parametric memory. For each question, we check whether any gold answer appears among the entities returned by its KG queries, and then average over all questions:

$$
\mathrm { R e t } \mathrm { Q } = \frac { 1 } { M } \sum _ { i = 1 } ^ { M } \mathbb { I } [ \exists y \in Y _ { i } ^ { * } , y \in \mathcal { E } ( \tau _ { i } ) ] ,\tag{17}
$$

where $\mathcal { E } ( \tau _ { i } )$ denotes the set of entities returned by all KG queries in the trajectory $\tau _ { i }$

## F.2 Gold Supporting Triples

The gold supporting triples used for evaluation are distinct from those produced by synthesis, which serve only as ECPR supervision during training. GTSQA provides gold answer subgraphs natively. For WikiWebQuestions, which has no such annotation, we follow EoG (Yan et al., 2026) and construct them in two stages. We first enumerate candidate paths from each topic entity to each answer entity by BFS under a maximum-hop constraint. An LLM verifier then judges whether each candidate path supports all constraints of the question, and the union of triples on the accepted paths forms the gold supporting triples.

## F.3 Human Validation of Constructed Paths

To validate the constructed paths independently of the LLM verifier, we sample 200 candidate paths from WikiWebQuestions, 100 accepted and 100 rejected by the verifier. Three graduate annotators independently judge whether each path supports all constraints of its question, without seeing the verifier decision. The annotators reach Fleiss $\kappa =$

<table><tr><td>Question</td><td>Candidate path</td><td>Reason for rejection</td></tr><tr><td>Who was the prime minister of Pakistan in 1999?</td><td>Links candidates only to the office “Prime Minister of Pakistan&quot;.</td><td>Does not establish that the candidate held the office in 1999.</td></tr><tr><td>What town did Justin Bieber grow up in?</td><td>Justin Bieber → Canada; Waterloo → Canada.</td><td>Sharing the same country does not estab- lish childhood residence.</td></tr></table>

Table 10: Representative rejected paths. The verifier checks whether a path covers the semantic constraints of the question, not only whether it reaches a plausible answer entity

0.91, and the human–verifier agreement is 93.0%, with a verifier precision of 0.94 and a recall of 0.92. Table 10 shows two representative rejected paths.

## G Baseline Details

We compare FoG against nine representative baselines that cover the LLM reasoning, retrieval-based, and agent-based paradigms:

IO Prompt (OpenAI, 2023; Team, 2025b; DeepSeek-AI, 2024) is a KG-free baseline where the LLM directly answers the question from its parametric knowledge. This serves as a lower bound for KG-augmented reasoning.

CoT (Wei et al., 2022) elicits step-by-step reasoning before the final answer, still without any access to the KG.

Self-Consistency (Wang et al., 2023) samples multiple CoT chains and aggregates their answers by majority voting.

RoG (Luo et al., 2024) generates relation paths as plans, retrieves reasoning paths along them, and answers from the retrieved paths.

SubGraphRAG (Li et al., 2025) retrieves a subgraph with a lightweight MLP that scores triples in parallel, and feeds it to an LLM reader.

GCR (Luo et al., 2025) encodes KG paths into a Trie index and constrains LLM decoding with the graph structure.

ToG (Sun et al., 2024) is a training-free agent that performs beam search over entities and relations through multi-turn KG interaction.

PoG (Chen et al., 2024) decomposes the question into sub-objectives and explores paths with memory updating and self-reflection.

KG-R1 (Lin et al., 2025) trains a single KG agent end-to-end with GRPO under outcome and turn-level action-validity rewards.

Adaptation to Wikidata. RoG, SubGraphRAG, and GCR were proposed on Freebase and rely on dataset-specific supervision, so they cannot be transferred to a new KG without retraining. We therefore retrain each of them on the training split of every Wikidata benchmark. KG-R1 was also proposed on Freebase. We retrain it on Wikidata with the same training data as FoG.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Fine-tuning method</td><td>LoRA</td></tr><tr><td>LoRA rank / alpha / dropout</td><td>16 / 32 / 0.05</td></tr><tr><td>Max sequence length</td><td>3,000</td></tr><tr><td>Training epochs</td><td>3</td></tr><tr><td>Per-device batch size</td><td>16</td></tr><tr><td>Gradient accumulation steps</td><td>8</td></tr><tr><td>Learning rate</td><td>5 × 10−5</td></tr><tr><td>LR scheduler</td><td>Cosine, warmup ratio 0.1</td></tr><tr><td>Weight decay</td><td>0.01</td></tr><tr><td>Gradient clipping</td><td>1.0</td></tr></table>

Table 11: Hyperparameters of supervised finetuning.

To ensure fairness, all methods, including FoG, are evaluated under identical conditions with a single inference run per question, and no multi-run aggregation is used. The backbone of each method is listed in Table 1.

## H Hyperparameter Settings

Supervised Fine-Tuning. For SFT, we adopt LLaMA-Factory (Zheng et al., 2024) and apply LoRA to all linear layers. The training trajectories are distilled from the same teacher model used in synthesis, Kimi-K2 (Team, 2025a), with the prompt in Table 19. Training runs on a single NVIDIA A100 GPU.

Reinforcement Learning. For RL, we adopt Verl (Sheng et al., 2025) and initialize the policy from the SFT checkpoint. Training runs on a single node with 4×NVIDIA A100 (40 GB) GPUs, using FSDP with parameter and optimizer CPU offloading, and rollouts are executed with vLLM. For each question, the agent samples N=4 trajectories with at most 10 KG interaction turns. KL regularization uses the k estimator with coefficient 0.03, anchored to the SFT model. Table 11 and Table 12 summarize the detailed hyperparameter configurations used throughout our experiments, and Table 13 lists the coefficients of the composite reward in Section 3.3.

<table><tr><td>Parameter</td><td>Value</td></tr><tr><td>Actor learning rate</td><td> $1 \times 1 0 ^ { - 6 }$ </td></tr><tr><td>KL loss coefficient</td><td>0.03 (k3 type)</td></tr><tr><td>GRPO rollout N</td><td>4</td></tr><tr><td>Max interaction turns per question</td><td>10</td></tr><tr><td>Train batch size</td><td>32</td></tr><tr><td>PPO mini-batch size</td><td>16</td></tr><tr><td>Max prompt length</td><td>4,096</td></tr><tr><td>Max response length per step</td><td>8,192</td></tr><tr><td>Max observation length</td><td>1,024</td></tr><tr><td>Rollout temperature</td><td>0.7</td></tr><tr><td>Total training steps</td><td>450</td></tr><tr><td>Hardware</td><td>4× NVIDIA A100 (40 GB)</td></tr></table>

Table 12: Hyperparameters of reinforcement learning.

Advantage Computation. The turn-level advantage in Section 3.3 is computed as follows. (i) Return. The return $G _ { i , t }$ in Equation 8 adds the immediate process rewards of turn t to the outcome reward of the trajectory. We use neither discounting nor a reward-to-go, so the evidence reward of a turn is credited to that turn only. (ii) Normalization. The mean $\mu _ { q }$ and standard deviation $\sigma _ { q }$ are computed over all (trajectory, turn) pairs of the N rollouts of the same question. Trajectories of different lengths need no padding, since each contributes as many returns as it has turns. (iii) Token mask. The advantage of a turn is broadcast to the tokens that the policy generates in that turn, i.e., its reasoning and its tool call or answer. Tool observations are part of the context but are masked out of the policy loss and the KL term. (iv) Objective. The policy maximizes

$$
\begin{array} { l } { \displaystyle \mathcal { I } ( \theta ) = \mathbb { E } _ { q , \{ \tau _ { i } \} } \left[ \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \frac { 1 } { | \tau _ { i } | } \sum _ { j = 1 } ^ { | \tau _ { i } | } \operatorname* { m i n } \left( \rho _ { i , j } \hat { A } _ { i , t ( j ) } , \right. \right. } \\ { \displaystyle \left. \left. \mathrm { c l i p } ( \rho _ { i , j } , 1 - \epsilon , 1 + \epsilon ) \hat { A } _ { i , t ( j ) } \right) \right] - \beta \mathbb { D } _ { \mathrm { K L } } ( \pi _ { \theta } \parallel \pi _ { \mathrm { r e f } } ) , } \end{array}\tag{18}
$$

where j ranges over the |τ<sub>i</sub>| policy-generated tokens, $t ( j )$ is the turn of token j, $\rho _ { i , j }$ is the probability ratio, and $\pi _ { \mathrm { r e f } }$ is the SFT model. (v) Difference from trajectory-level credit. When the coverage reward is credited only to the final turn, the turns that actually retrieved the evidence receive the same return as the turns that did not, so the update cannot distinguish them. With first-match attribution, two trajectories with the same coverage still produce different per-turn returns, and the advantage concentrates on the queries that first reach gold triples.

<table><tr><td>Reward term</td><td>Coefficient</td><td>Value</td></tr><tr><td>Answer F1  $( R _ { \mathrm { o u t } } )$ </td><td> $\lambda _ { \mathrm { f l } }$ </td><td>0.7</td></tr><tr><td>Answer retrieval  $( R _ { \mathrm { o u t } } )$ </td><td> $\lambda _ { \mathrm { { r e t } } }$ </td><td>0.3</td></tr><tr><td>Step validity  $( R _ { \mathrm { s t e p } } )$ </td><td> $\lambda _ { \mathrm { s t e p } }$ </td><td>0.5</td></tr><tr><td>Evidence coverage  $( R _ { \mathrm { e c p r } } )$ </td><td> $\lambda _ { \mathrm { e c p r } }$ </td><td>1.0</td></tr></table>

Table 13: Coefficients of the composite reward. Each coefficient weights one term of the reward in Section 3.3.

## I Case Study

To better understand how much of the supporting evidence different methods retrieve from the KG, we present a case study on the question: “Which airline flies the Embraer 195 and is based at the airport named after Margaret Kenyatta’sfather?” As shown in Tables 14, 15, and 16, this question requires resolving the father, the airport named after him, and the airline that is based there and flies the aircraft. All three methods predict the correct answer, but they differ in how much of the evidence chain they recover.

FoG recovers the complete gold evidence. It issues nine KG turns and grounds each hop in the triples returned by the previous one, so every link of the chain can be read from the trajectory.

In contrast, ToG performs four rounds of beam search but recovers only one relevant triple, missing the links for airport naming and airline hub, so its answer relies mainly on parametric knowledge. PoG answers after a single retrieval step, leaving an evidence chain that cannot be audited from the trajectory. This case highlights the strength of FoG in reaching the answer through evidence that is fully retrieved from the KG.

## J Prompts Used in FoG

## J.1 Seed Question Generation Prompt

As shown in Table 17, this prompt is designed for generating seed questions from a sampled subgraph. It guides the teacher to return a question, its logical form, and its supporting triples, which corresponds to $\pi _ { \mathrm { g e n } }$ in Equation 1.

## J.2 Backward Expansion Prompt

As shown in Table 18, this prompt is used by the backward-expansion agent. The agent is instructed to select an anchor entity, retrieve its neighborhood, generate a sub-question whose answer is the anchor, and verify the result, retrying on failure.

## J.3 Trajectory Distillation Prompt

As shown in Table 19, this prompt is used to distill the teacher trajectories for SFT. The teacher is instructed to explore the KG only through the provided tools, explain each call, and answer with entities that appear in the tool responses.

## J.4 Agent Prompt

As shown in Table 20, this prompt is used by the FoG agent in both training and evaluation. It guides the agent to analyze the question, call the KG operations, and answer from the returned triples.

Table 14: Case study of FoG. The trajectory recovers the complete gold evidence before answering.
<table><tr><td>Question: Which airline flies the Embraer 195 and is based at the airport named after Margaret Kenyatta&#x27;s father?</td></tr><tr><td>&lt;think&gt;I need to find an airline that satisfies two conditions: 1) flies the Embraer 195, and 2) is based at the airport named after Margaret Kenyatta&#x27;s father. Let me start by exploring the relationships around the first given entity to understand what information is stored about Margaret Kenyatta in the knowledge graph.&lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; find_outgoing_relations(entity_name=Margaret Kenyatta) &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Outgoing relations for entity Margaret Kenyatta: country of citizenship, father, given name, instance of, mother, occupation, sex or gendr...... &lt;/information&gt;</td></tr><tr><td>&lt;think&gt; The outgoing triple shows that Margaret Kenyatta has a &quot;father&quot; in the knowledge graph. Let me find out who her father is by querying the triple that directly connects Margaret Kenyatta to her father. &lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; get_tail_entities(head_entity_name=Margaret Kenyatta, relation_name=father) &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Entities outgoing via relation father from entity Margaret Kenyatta: Jomo Kenyat..... &lt;/information&gt; &lt;think&gt; Now I know Jomo Kenyatta is Margaret Kenyatta&#x27;s father. Next, I need to find the airport named after Jomo</td></tr><tr><td>Kenyatta. Let me check what relationships exist for Jomo Kenyatta in the knowledge graph to see if there is an airport named after him. &lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; find_incoming_relations(entity_name=Jomo Kenyatta) &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Incoming relations for entity Jomo Kenyatta: father,named after,spouse..... &lt;/information&gt; &lt;think&gt; I see that some entities have &quot;named after&quot; pointing to Jomo Kenyatta. Let me find out what entities are named</td></tr><tr><td>after Jomo Kenyatta in the knowledge graph. &lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; get_head_entities(tail_entity_name=Jomo Kenyatta, relation_name=named after) &lt;/kg-query&gt; &lt;information&gt; Entities incoming via relation named after from entity Jomo Kenyatta: Jomo Kenyatta International</td></tr><tr><td>Airport, Kenyatta International Convention Centre &lt;/information&gt;</td></tr><tr><td>&lt;think&gt; Great! I found that Jomo Kenyatta International Airport is named after Jomo Kenyatta. Now I need to find which airline is based at Jomo Kenyatta International Airport. Let me check what outgoing relationships exist for Jomo Kenyatta</td></tr><tr><td>International Airport to see if there is an airline bâsed there. &lt;/think&gt; &lt;kg-query&gt; find_outgoing_relations(entity_name=Jomo Kenyatta International Airport) &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Outgoing relations for entity Jomo Kenyatta International Airport: country, instance of, located in the</td></tr><tr><td>administrative territorial entity, named after...... &lt;/information&gt; &lt;think&gt; I don&#x27;t see a direct &quot;airline based at&quot; relationship listed for the airport. Let me check the incoming relationships</td></tr><tr><td>to see if any airlines have &quot;headquarters location&quot; or similar relationships pointing to this airport. &lt;/think&gt; &lt;kg-query&gt; find_incoming_relations(entity_name=Jomo Kenyatta International Airport). &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Incoming relations for entity Jomo Kenyatta International Airport: airline hub &lt;/information&gt;</td></tr><tr><td>&lt;think&gt; Perfect! I found that some airlines have &quot;airline hub&quot; pointing to Jomo Kenyatta International Airport. Let me find out which airlines have Jomo Kenyatta International Airport as their airline hub. &lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; get_head_entities(tail_entity_name=Jomo Kenyatta International Airport, relation_name=airline hub)</td></tr><tr><td>&lt;/kg-query&gt; &lt;information&gt; Entities incoming via relation airline hub from entity Jomo Kenyatta International Airport: Kenya</td></tr><tr><td>Airways &lt;/information&gt; &lt;think&gt; Now I have Kenya Airways as the airline based at Jomo Kenyatta International Airport. Finally, I need to verify</td></tr><tr><td>that Kenya Airways also flies the Embraer 195. Let me check what aircraft types Kenya Airways operates by looking at its outgoing relationships. &lt;/think&gt;</td></tr><tr><td>&lt;kg-query&gt; find_outgoing_relations(entity_name=Kenya Airways) &lt;/kg-query&gt; &lt;information&gt; Outgoing relations for entity Kenya Airways: airline alliance, airline hub, country, instance of, item</td></tr><tr><td>operated .&lt;/information&gt; &lt;think&gt; I see that Kenya Airways has &quot;item operated&quot; as an outgoing relationship. Let me find out what items (likely</td></tr><tr><td>aircraft) Kenya Airways operates. &lt;/think&gt; &lt;kg-query&gt; get_tail_entities(head_entity_name=Kenya Airways, relation_name=item operated) &lt;/kg-query&gt;</td></tr><tr><td>&lt;information&gt; Entities outgoing via relation item operated from entity Kenya Airways: Boeing 737 Classic,Boeing 737</td></tr><tr><td>Next Generation,Boeing 767,Boeing 777,Boeing 787 Dreamliner,Embraer 195 &lt;/information&gt;</td></tr><tr><td>&lt;think&gt; I now have enough grounded evidence to give the answer. &lt;/think&gt; &lt;answer&gt; [&quot;Kenya Airways&quot;] &lt;/answer&gt;</td></tr></table>

Table 15: Case study of ToG. The trajectory reaches the correct answer while covering only the first part of the gold evidence.  
Question: Which airline flies the Embraer 195 and is based at the airport named after Margaret Kenyatta’s father?   
Reasoning chains:   
Turn 1:   
‘Margaret Kenyatta -> father -> Jomo Kenyatta‘; ‘Margaret Kenyatta -> country of citizenship -> Kenya‘; ‘Embraer 195 ->   
subclass of -> E-Jet‘; ‘Margaret Kenyatta -> occupation -> diplomat‘; ‘Margaret Kenyatta -> occupation -> politician‘;   
Turn 2:   
‘Kenya -> named after -> Mount Kenya‘; ‘diplomat -> named after -> Diplomatstaden‘; ‘E-Jet -> manufacturer ->   
Embraer‘; ‘politician -> category combines topics -> Category:Dutch politicians‘; ‘Jomo Kenyatta -> country of   
citizenship -> Kenya‘;   
Turn 3:   
‘Mount Kenya -> country -> Kenya‘; ‘Embraer -> headquarters location -> São José dos Campos‘; ‘Category:Dutch   
politicians -> is a list of -> human‘; ‘Kenya -> named after -> Mount Kenya‘; ‘Diplomatstaden -> located in the   
administrative territorial entity -> Stockholm Municipality‘;   
Turn 4:   
‘Kenya -> named after -> Mount Kenya‘; ‘Mount Kenya -> country -> Kenya‘; ‘Kenya -> capital -> Nairobi‘; ‘São José   
dos Campos -> country -> Brazil‘; ‘São José dos Campos -> located in the administrative territorial entity -> São Paulo‘;   
<answer>First, the airline that flies the Embraer 195 is Kenya Airways. Second, the airport named after Margaret   
Kenyatta’s father is Jomo Kenyatta International Airport. The answer is {Kenya Airways}</answer>  
Table 16: Case study of PoG. The trajectory reaches the correct answer through a shortcut retrieval and misses the bridge evidence.

Table 17: Prompt for seed question generation.
<table><tr><td>Seed Question Generation Prompt</td></tr><tr><td>SYSTEM PROMPT</td></tr><tr><td>You are a question synthesis expert. You are deeply familiar with the structure of the Wikidata knowledge graph and specialize in constructing questions based on given triple information.</td></tr><tr><td>Core Concept Definitions</td></tr><tr><td>Three Entity Roles in this task:</td></tr><tr><td>• Constraint Entity (Con@). The starting point of the question; an entity the user already knows that must be explicitly mentioned. It anchors the context of the question.</td></tr><tr><td>• Intermediate Entity (Mid@). The intermediate node in the reasoning chain, obtained through the first reasoning step from the constraint entity. It connects two reasoning steps and must be inferred by the user, never stated</td></tr><tr><td>explicitly. • Answer Entity (Ans@). The endpoint of the reasoning chain and the final answer requested by the user.</td></tr><tr><td>Triple Selection Principles</td></tr><tr><td>• Subset selection. Select a subset of triples from the given set as the evidence basis. Prefer strong constraints that lead to clear, unambiguous answers.</td></tr><tr><td>• Avoid redundancy. If a triple already infers B from A, do not select another path that also infers B (no shortcuts allowed).</td></tr><tr><td>• Necessity. Every selected triple must contribute to the reasoning chain; no irrelevant triples are permitted.</td></tr><tr><td>Question Design Principles • Information sources. All information must come exclusively from the selected triples. No external knowledge</td></tr><tr><td>may be used; the question must be verifiable via a SPARQL query on Wikidata. • Reasoning forms. The question may adopt single-hop reasoning, multi-hop reasoning, intersection queries, union</td></tr><tr><td>queries, counting, or reverse reasoning. • Answer requirements. The answer must be derived entirely from selected triples and directly verifiable via a</td></tr><tr><td>SPARQL query.</td></tr><tr><td>• Path guidelines. A Path is an explanation of the reasoning process, expressed as a reasoning chain describing how the answer is derived.</td></tr><tr><td>SPARQL Query Requirements • The generated SPARQL must be syntactically correct and executable on the Wikidata Query Service (WDQS).</td></tr><tr><td>• Query results must accurately correspond to the answer of the question.</td></tr><tr><td>• For multi-hop questions, the SPARQL must clearly reflect intermediate variables in the reasoning process.</td></tr><tr><td></td></tr><tr><td>Entity Usage Rules in Questions √ Only constraint entities (question premises) may be explicitly mentioned.</td></tr><tr><td>√ The question must ask for the answer entity (final target).</td></tr><tr><td>× The specific name of any intermediate entity must never appear explicitly in the question.</td></tr><tr><td>× Intermediate entities must only be implicitly implied through relational wording.</td></tr><tr><td>× Avoid redundancy: if the answer can be uniquely determined with fewer triples or entities, the question is redundant and must be revised.</td></tr><tr><td>USER PROMPT</td></tr><tr><td>Relevant Triple Information:</td></tr><tr><td>#{Relevant triple information}</td></tr></table>

Table 18: Prompt for backward expansion.
<table><tr><td>Complex Question Generation Prompt</td></tr><tr><td>SYSTEM PROMPT</td></tr><tr><td>You are an expert agent specializing in Knowledge Graph Question Answering (KGQA) data synthesis. Select an Anchor Node from a given set of entities and generate a high-quality sub-question that can uniquely identify/locate</td></tr><tr><td>this anchor entity within the Knowledge Graph.</td></tr><tr><td>Standard Operating Procedure: Step 1 Select an entity as the Anchor Node and call retrieve_entity_neighborhood to fetch its neighboring</td></tr><tr><td>triples.</td></tr><tr><td>Step 2 Call select_triples_and_generate_subquestion to choose relevant triples and generate the sub- question.</td></tr><tr><td>Step 3 Call verify_subquestion to validate the accuracy and check for redundancy of the generated question.</td></tr><tr><td>Step 4 If validation passes, return the final result in &lt;answer&gt;{. . . }&lt;/answer&gt; format. If it fails, return to Step 2 and retry.</td></tr><tr><td>Strict Constraints:</td></tr><tr><td>Prohibited:</td></tr><tr><td>× No shortcuts. Do not skip any steps or generate the final answer directly.</td></tr><tr><td>× No tool-less answers. Never return &lt;answer&gt; without having successfully executed the required tool calls. × No preconceptions. Do not assume or hallucinate knowledge before receiving tool outputs.</td></tr><tr><td>×Redundancy rule. If verify_subquestion flags the question as “redundant,&quot; it means the anchor entity can be uniquely identified using fewer triples or fewer entities in the question. You must retry Step 2.</td></tr><tr><td>Required:</td></tr><tr><td>√ Sequential execution. Strictly follow the 1 → 2 → 3 → 4 workflow. √ Careful observation. Analyze the output of each tool before proceeding to the next step.</td></tr><tr><td>√Final approval. Only output the final result after verify_subquestion returns a success status.</td></tr><tr><td>USER PROMPT</td></tr><tr><td>Output Format:</td></tr><tr><td>The final answer must be returned as a JSON object wrapped within &lt;answer&gt;&lt;/answer&gt; tags:</td></tr><tr><td>JSON Format</td></tr><tr><td>If a question cannot be generated, return: &lt;answer&gt;do not generate question&lt;/answer&gt;</td></tr><tr><td>Input Entity Information:</td></tr><tr><td>Please select one entity from the list below to serve as the Anchor Node (the answer entity) and generate a sub-question for it:</td></tr></table>

Table 19: Prompt for trajectory distillation.  
![](images/2bd9bf29edf99894e68a5d75ed9a9272bd31d3fc7cb63863398c6791d36f573f.jpg)

Table 20: Prompt of the FoG agent in training and evaluation.  
FoG Training and Evaluation Prompt   
SYSTEM PROMPT   
You are a knowledge graph question answering assistant. You answer user questions by interacting with a knowledge   
graph. When a user provides a question and its associated entities, you call tools to query the knowledge graph   
accordingly. No matter how complex the question is, you will not give up until you believe an answer can be found   
based on the available information.   
• Comprehensive analysis. Combine external triple information obtained through tools to perform analysis and   
provide a complete answer.   
• Working principles. Clearly analyze the problem, select appropriate tools, invoke them to obtain information, and   
ensure the problem is solved efficiently.   
USER PROMPT   
<tools> tools definition </tools>   
The assistant operates in one or more cycles of (thinking → tool call → tool response), followed by a final (thinking   
→ answer) step. All intermediate processes are enclosed within their respective tags. Multiple thinking steps, tool   
calls, and responses may appear in sequence.   
Example Response:   
<think> thinking process here </think>   
<kg-query>{"name": "tool name", "arguments": {"param": value, ...}}</kg-query>   
<information> tool\_response here </information>   
<think> thinking process here </think>   
<kg-query> {"name": "another tool name", "arguments": {...}} </kg-query>   
<information> tool\_response here </information>   
... (more thinking, tool calls, and responses) ...   
<think> thinking process here </think>   
<answer> answer here </answer>   
User Input:   
Question: {question}   
Entity info {entity\_name}