# CERA-MoA: Co-Evolving Routing Mechanisms with Continually Learning LLM Agents

Jiaxuan Jiang<sup>1,</sup> <sup>3</sup>, Liyuan He<sup>2,</sup> <sup>3</sup>, Zhixuan Fang<sup>1,</sup> <sup>3,</sup>

<sup>1</sup>IIIS, Tsinghua University, Beijing, China

<sup>2</sup>School of Artificial Intelligence, Shanghai Jiao Tong University, Shanghai, China <sup>3</sup>Shanghai Qi Zhi Institute, Shanghai, China

## Abstract

Current Mixture-of-Agents (MoA) paradigms generally treat query routing and agent fine-tuning as separate processes, limiting their ability to respond to evolving agent capabilities. This disconnect prevents routing strategies from adapting to evolving agent capabilities during post-training and prevents agents from achieving synergistic data-driven specialization. To resolve this, we introduce CERA-MoA (Co-Evolving Router with continually learning Agents for Mixtureof-Agents), an iterative reinforcement learning framework where the dynamic router and independent agent policies co-evolve. We design a predictive familiarity estimator that leverages mid-layer hidden states to evaluate semantic competence among agents, avoiding the overhead of full rollouts. Based on these familiarity scores, a cumulative-threshold adaptive routing mechanism dynamically activates a tailored minimal agent subset, achieving a trade-of between task performance and eficiency. By proactively allocating targeted training samples to agents based on their evolving competence, CERA-MoA promotes capability diferentiation. Extensive experiments across various domains demonstrate that CERA-MoA outperforms state-of-the-art static-agent routing and fix-workflow fine-tuning baselines.

## 1 Introduction

Although large language models (LLMs) have demonstrated remarkable reasoning capabilities, standard posttraining methods frequently encounter generalization bottlenecks when dealing with increasingly complex and multifaceted problem domains (Hong et al. 2024; Ye et al. 2025). To transcend the limitations of single monolithic models, the Mixture-of-Agents (MoA) paradigm has emerged as a promising solution. By combining multiple agents, this paradigm aims to solve diverse tasks more efectively by leveraging the complementary strengths of diferent agents (Wang et al. 2025a). Such collaborative frameworks have the potential to substantially expand the performance boundaries of LLMs on complex tasks.

Despite this potential, the current development of multiagent systems primarily bifurcates into two directions. The first direction focuses on optimizing orchestration mechanisms among fixed-capability agents, which is typically achieved through debate frameworks to refine reasoning consensus (Chan et al. 2024; Estornell and Liu 2024; Yi et al. 2025; Hu et al. 2026; Fan, Yoon, and Ji 2026; Qiao et al. 2026), or through dynamic agent selection for eficient query allocation (Xia et al. 2024; Yue et al. 2025; Lee et al. 2026; Poon et al. 2026; Wang et al. 2026a; Xue et al. 2026a; Wang et al. 2026b). The second direction explores training and fine-tuning agents, but typically operates within predetermined multi-agent workflow architectures (Park et al. 2025; Motwani et al. 2025; Zhang et al. 2025; Xue et al. 2026b; Zhao et al. 2026; Wang et al. 2026d). Although these approaches efectively enhance collective performance, they inherently decouple the routing strategy from the continual learning dynamics of agents.

This decoupling exposes a critical research gap in contemporary multi-agent paradigms. First, the capabilities of individual agents continually evolve during the post-training phase. However, existing routing mechanisms are not designed to adapt to such capability shifts. Second, current post-training pipelines typically rely on manually partitioned datasets, lacking a dynamic sample allocation mechanism. Without routing specific training queries to agents based on their competence, the system fails to automatically induce distinct, targeted expertise. Consequently, the disconnect between routing strategies and the continual learning of agents prevents the system from achieving synergistic capability specialization, leaving individual agents acting as generalists rather than domain experts.

To bridge this critical disconnect, we formulate Co-Evolving Router with continually learning Agents for Mixture-of-Agents (CERA-MoA), an iterative closed-loop paradigm that integrates agent learning with dynamic routing optimization. Rather than treating routing and agent adaptation in isolation, CERA-MoA not only adapts to the progressively shifting capabilities of individual agents, but also dynamically allocates tailored training queries to explicitly induce skill specialization. At the core of our system is a predictive familiarity estimator, which directly extracts the intermediate hidden states of LLMs to quantify how well an input query aligns with each agent’s learned expertise. This design provides a dynamic evaluation of relative competence prior to text generation, reducing the heavy computational overhead of external evaluators or full rollout generations. Utilizing these familiarity scores, we further introduce a cumulativethreshold adaptive routing strategy. Rather than relying on a fixed top-k agent allocation, this mechanism dynamically selects a varying number of agents based on the estimated competence coverage of the agent population. By activating the smallest score-ranked subset of agents whose cumulative familiarity exceeds the threshold, our strategy achieves a fine-grained trade-of between task performance and computational cost.

The contributions of this work are summarized as follows:

• We introduce a sample-level Mixture-of-Agents framework that enables continual reinforcement learning of LLM agents through adaptive query routing. This coevolutionary system simultaneously adapts the routing mechanism to shifting agent capabilities and dynamically allocates training queries thereby encouraging distinct problem-solving specialization.

• We design a predictive familiarity estimator for eficient semantic-level competence evaluation, together with a cumulative-threshold adaptive routing mechanism to dynamically balance performance and computational cost based on the estimated competence.

• We conducted extensive experiments demonstrating that our framework outperforms static-agent routing and fixworkflow fine-tuning baselines. Empirical results highlight performance gains across diverse problem domains.

## 2 Related Work

Multi-agent systems (MAS) have emerged as a powerfu paradigm to extend the reasoning boundaries of large language models (LLMs) by leveraging collective intelligence (Zhao, Wang, and Peng 2024; Ye et al. 2025). To facilitate efective collaboration, standard MAS architectures typically organize models through structured interaction protocols, such as multi-agent debate (Liang et al. 2024; Estornell and Liu 2024), majority voting (Chen, Saha, and Bansa 2024; Taubenfeld et al. 2025), or mixtures of independent agents (Wang et al. 2024; Xie et al. 2025; Li et al. 2026). Unlike token-level Mixture-of-Experts (MoE) architectures that route internal hidden representations across specialized subnetworks (Oldfield et al. 2024; Lv et al. 2025; Zhuang et al. 2025), MAS operates at the sample and semantic level, requiring high-level coordination among autonomous models. To optimize the collective eficacy of these collaborative systems, current research methodologies generally bifurcate into two directions: orchestration optimization, which focuses on dynamic interaction protocols among fixed-capability agents, and agent fine-tuning, which actively trains individual agent policies within predefined workflows.

Orchestration Optimization for Multi-Agent Systems. To optimize collaboration structure, existing literature widely investigates dynamic orchestration and query routing. $\mathsf { A p - }$ proaches range from multi-arm bandits (Xia et al. 2024; Poon et al. 2026) and knapsacks within budgets (Wang et al. 2025b; Xue et al. 2026a) to agent diversity maximization (Xie et al. 2025), lightweight evaluation scorers (Yue et al. 2025; Wang et al. 2026a,b), and confidence-guided stepwise routing (Lee et al. 2026; Wang et al. 2026c). Recently, studies have also explored leveraging LLMs directly as self-orchestrators (Dang et al. 2026; Ke et al. 2026), topology graph generators (Zhang et al. 2026; Li et al. 2026), or meta-thinkers (Zhu et al. 2026), while methods like AgentDropout (Wang et al. 2025c) prune redundant communication nodes to reduce overhead. Despite efectively optimizing orchestration and reducing costs, these techniques generally assume that candidate models remain static during the orchestration training phase. Consequently, they lack an integrated mechanism to realign query allocation as individual models actively evolve and specialize. CERA-MoA bridges this gap by co-evolving a predictive familiarity estimator that captures dynamic relative competence alongside agent policy updates, ensuring dynamic adaptation to shifting agent expertise.

Agent Fine-tuning for Multi-Agent Systems. Beyond static interactions, recent work actively fine-tunes agents to enhance individual and collaborative reasoning using reinforcement learning, preference optimization, and supervised learning. The methods explore test-time self-verification (Lee et al. 2025), reasoning chain refinement (Puerto et al. 2025), self-reflection cycles (Zhao et al. 2025), and reflection interaction optimization (Yuan and Xie 2025). Within collaborative setups, fine-tuning is driven by rule-based verifiers (Park et al. 2025), tree-structured sampling (Motwani et al. 2025; Zhao et al. 2026), LLM-as-a-judge interactions (Xue et al. 2026b), and end-to-end multi-agent reinforcement learning (Wang et al. 2026d). However, existing fine-tuning pipelines predominantly optimize agent policies within predefined workflow architectures. Furthermore, they mainly rely on manually partitioned or uniform training datasets, which keep data allocation separate from real-time learning dynamics. Without capability-aware query routing during training, existing systems lack an explicit mechanism to guide agents toward complementary specialization, leaving agents to act as homogeneous generalists. CERA-MoA addresses this gap by integrating an iterative routing mechanism directly into the training loop, proactively allocating semantic queries to agents based on their evolving competence to explicitly foster domain specialization.

## 3 CERA-MoA

## 3.1 Overview

As illustrated in Figure 1, we formulate CERA-MoA (Co-Evolving Router with continually learning Agents for Mixture-of-Agents) as a collaborative framework $\begin{array} { r l } { \mathcal { M } } & { { } = } \end{array}$ $\left( \mathcal { D } _ { \psi } , \{ \mathcal { A } _ { \theta _ { i } } \} _ { i = 1 } ^ { N } , \mathcal { S } \right)$ , integrating a parameterized router $\mathcal { D } _ { \psi }$ a population of continually learning agent policies $\{ \mathcal { A } _ { \theta _ { i } } \} _ { i = 1 } ^ { N }$ and a voting-based aggregator S. Our primary design objective is to establish a closed-loop co-evolutionary process for the router and the agents: adaptive query allocation routes targeted training samples to specific agents to drive domain specialization, while the router synchronously updates its evaluation parameters to accurately track these continually shifting agent capabilities.

Training Phase. For an incoming query, the router $\mathcal { D } _ { \psi }$ computes a familiarity score that measures relative agent competence and allocates the sample to a tailored subset of agents. Selected agents independently generate completions and optimize their policies through reinforcement learning, driven by task-specific rewards. Concurrently, the router evaluates the relative advantages of agent performance to update its familiarity estimator. This iterative feedback loop naturally encourages specialization: agents receive problems aligned with their potential, evolving from homogeneous generalists into domain specialists, while the router synchronously tracks their real-time expertise.

![](images/ed536791ef352ae95f5ad6c379c07d7fb3bc829dfc2060fc4eeba749db31c117.jpg)  
Figure 1: Overview of CERA-MoA.

Inference Phase. During inference, $\mathcal { D } _ { \psi }$ routes the query to the minimal subset of competent agents based on learned familiarity scores. Each activated agent generates a single response. For tasks with deterministic solutions, the aggregator $\dot { s }$ executes familiarity-weighted majority voting:

$$
\hat { y } = \arg \operatorname* { m a x } _ { y } \sum _ { \mathrm { ~ \it ~ i ~ s . t . ~ a n s ( } a _ { i } ) = y } f _ { i } ( q ) ,\tag{1}
$$

selecting the final completion from the agent exhibiting the highest familiarity score $f _ { i } ( q )$ within the winning consensus group. For open-ended tasks, S directly outputs the completion of the most familiar agent.

## 3.2 Predictive Familiarity Estimator

To estimate agent-query compatibility prior to generation without relying on costly external verifiers, we propose a predictive familiarity estimator. It consists of a trainable predictor head $g _ { i }$ and a randomly initialized and permanently frozen target head $\bar { g } _ { i }$ for each agent i. The estimator projects query features extracted from the router’s frozen backbone into a reference space. Importantly, the backbone itself is not updated by the familiarity objective; only the predictor head is trained. Thus, the intermediate hidden states $h ( q )$ should be understood as fixed semantic features produced by the shared backbone, while what evolves during training is the learned mapping from these features to the familiarity score space. As agent policies co-evolve and the routed training distribution shifts, the predictor head adapts to the changing competence landscape over this fixed feature space.

Query q is first passed through the router’s base language model. Its semantic representation $h ( q )$ is then computed by concatenating intermediate hidden states from the model. Intermediate hidden states have been shown to capture semantic information complementary to final-layer representations, whose are more directly tied to next-token prediction (Skean et al. 2025). Let hidden (q) denote the hidden state at layer l of a total depth L. We define the semantic representation as

$$
h ( { q } ) = \mathrm { c o n c a t } \left( \mathrm { h i d d e n } _ { \lfloor \frac { L } { 2 } \rfloor } ( { q } ) \bigg | \bigg | \mathrm { h i d d e n } _ { \lfloor \frac { 3 L } { 4 } \rfloor } ( { q } ) \right)\tag{2}
$$

The semantic representation is projected into two distinct embeddings $c _ { i } ( q ) = g _ { i } ( h ( q ) )$ and $\bar { c } _ { i } \tilde { ( } q ) = \bar { g } _ { i } ( h ( q ) )$ ). The target heads remain frozen and are initialized orthogonally across diferent agents to provide a stable reference space while ensuring initial diversity among the agent population. The normalized Euclidean distance between these two projections is computed to quantify the head distance $d _ { i } ( q )$ according to

$$
d _ { i } ( q ) = \left. \frac { c _ { i } ( q ) } { \lVert c _ { i } ( q ) \rVert _ { 2 } } - \frac { \bar { c } _ { i } ( q ) } { \lVert \bar { c } _ { i } ( q ) \rVert _ { 2 } } \right. _ { 2 } .\tag{3}
$$

The familiarity score is then derived via exponential decay:

$$
f _ { i } ( q ) = \exp ( - \lambda d _ { i } ( q ) ) ,\tag{4}
$$

where $\lambda$ is a temperature hyperparameter controlling the sensitivity of the score to the projection distance. A smaller distance therefore yields a higher familiarity score. Rather than regressing an absolute reward, the estimator converts reward supervision into a relative push-and-pull signal around a fixed anchor. The frozen target head does not encode a semantic prototype or competence label; it simply provides a stable reference point. Positive relative advantages pull the predictor toward this anchor, while negative advantages push it away by a margin. In this way, the familiarity score becomes a reward-aligned geometric compatibility signal in a fixed reference space.

## 3.3 Cumulative-Threshold Adaptive Routing

To transcend fixed top-k routing constraints, we introduce a familiarity-aware, cumulative-threshold allocation mechanism that balances reasoning eficacy against multi-agent inference overhead by activating the minimal agent subset required to reach a target competence threshold.

During training, routing scores incorporate exploration bonuses alongside familiarity scores to prevent starvation, as well as historical entropy to encourage policy diversity:

$$
s _ { i } ( q ) = f _ { i } ( q ) + c _ { u c b } \sqrt { \frac { \log T } { n _ { i } + 1 } } + w _ { e n t } \overline { { H } } _ { i } ,\tag{5}
$$

where $T$ is the number of routed queries, $n _ { i }$ is the number of training samples allocated to agent $i ,$ and $\overline { { H } } _ { i }$ represents the agent’s historical mean generation entropy. Let (j) denote the index of the agent ranked j-th in descending order by the routing score $s _ { i } ( \boldsymbol q )$ . Crucially, while the routing score $s _ { i } ( \boldsymbol q )$ determines the priority of agent selection to foster exploration during training, the cutof for subset activation is strictly evaluated against familiarity scores. Thus, the router activates the smallest top-ranked subset of agents whose cumulative familiarity score $f _ { ( j ) } ( \boldsymbol { q } )$ satisfies:

$$
\operatorname* { m i n } _ { k } \sum _ { j = 1 } ^ { k } f _ { ( j ) } ( q ) \geq \tau ,\tag{6}
$$

where $\tau$ is the predefined cumulative threshold. If the total sum falls below τ , the query is allocated to the entire population. At inference time, the exploration terms are deactivated $( c _ { u c b } = w _ { e n t } = 0 )$ , strictly routing queries to the most competent specialists while preserving eficiency.

## 3.4 Joint Optimization of Router and Agents

The engine driving CERA-MoA is a closed-loop reinforcement learning protocol that bridges agent policy optimization with synchronous router updates. This training mechanism explicitly promotes capability diferentiation: agents that successfully solve a query acquire higher familiarity scores within that specific semantic space. Meanwhile, they become more likely to be allocated semantically similar queries in the future, establishing a learning loop that naturally induces deep domain specialization.

Agent Optimization. Selected agents store allocated queries in a last-in-first-out (LIFO) bufer. Upon forming a training batch, agent i samples G completions per query and updates its policy using Dynamic Sampling Policy Optimization (DAPO) (Yu et al. 2026) combined with sequencelevel importance sampling from Group Sequence Policy Optimization (GSPO) (Zheng et al. 2025).

Let $R \left( q , a _ { i , q } ^ { ( g ) } \right)$ denote the reward assigned to the g-th completion of question q by agent i. The normalized advantage for the g-th completion is

$$
\hat { A } _ { i , q } ^ { ( g ) } = \frac { R \left( q , a _ { i , q } ^ { ( g ) } \right) - \mu _ { i , q } } { \sigma _ { i , q } + 1 0 ^ { - 4 } } ,\tag{7}
$$

where $\mu _ { i , q }$ and $\sigma _ { i , q }$ are the empirical mean and standard deviation of the $G$ rewards.

For a completion $a _ { i , q } ^ { ( g ) } = ( y _ { 1 } , \dots , y _ { S } )$ , the sequence-level importance ratio is averaged over valid completion tokens:

$$
\rho _ { i , q } ^ { ( g ) } = \exp \left( \frac { 1 } { S } \sum _ { t = 1 } ^ { S } \log \frac { \pi _ { \theta _ { i } } ( y _ { t } \mid q , y _ { < t } ) } { \pi _ { \theta _ { i } ^ { \mathrm { o l d } } } ( y _ { t } \mid q , y _ { < t } ) } \right) .\tag{8}
$$

Then we define the clipped surrogate as

$$
s _ { i , q } ^ { ( g ) } = - \operatorname* { m i n } \left( \rho _ { i , q } ^ { ( g ) } \hat { A } _ { i , q } ^ { ( g ) } , \operatorname { c l i p } \left( \rho _ { i , q } ^ { ( g ) } , 1 - \epsilon , 1 + \epsilon \right) \hat { A } _ { i , q } ^ { ( g ) } \right) .\tag{9}
$$

The KL penalty is computed token-wise against the reference policy. Defining

$$
x _ { t } = \log \pi _ { \mathrm { r e f } } \left( y _ { t } \mid q , y _ { < t } \right) - \log \pi _ { \theta _ { i } } \left( y _ { t } \mid q , y _ { < t } \right) ,\tag{10}
$$

the token-level KL penalty then utilizes the estimator

$$
D _ { \mathrm { K L } , t } = \exp ( x _ { t } ) - x _ { t } - 1 .\tag{11}
$$

The loss for agent i is normalized by the number of active completion tokens in the accumulated training batch:

$$
\mathcal { L } _ { \mathrm { a g e n t } } ( \theta _ { i } ) = \frac { \sum _ { q , g , t } m _ { q , g , t } \left( s _ { i , q } ^ { ( g ) } + \beta D _ { \mathrm { K L } , t } \right) } { \sum _ { q , g , t } m _ { q , g , t } } ,\tag{12}
$$

where $m _ { q , g , t }$ masks padding tokens and $\beta$ acts as the regularization scaling hyperparameter.

Router Optimization. The router updates by evaluating the relative performance of participating agents. For each query $q ,$ let $\textstyle { \mathcal { S } } _ { q }$ denote the set of agents activated in the current optimization step. We calculate the mean reward for agent i over its sampled completions as

$$
r _ { i , q } = \frac { 1 } { G } \sum _ { g = 1 } ^ { G } R \left( q , a _ { i , q } ^ { \left( g \right) } \right) .\tag{13}
$$

<table><tr><td rowspan="2">Method</td><td colspan="3">Math</td><td colspan="3">Code</td><td colspan="2">Instruction</td><td>General</td><td rowspan="2">Avg.</td></tr><tr><td>GSM8k</td><td>MATH</td><td>DAPO-M</td><td>MBPP</td><td>Eurus</td><td>TACO</td><td>MAGPIE</td><td>RLVR-IF</td><td>BBH</td></tr><tr><td>Qwen3-4B</td><td>91.2</td><td>71.5</td><td>25.6</td><td>25.3</td><td>12.9</td><td>3.2</td><td>90.6</td><td>59.6</td><td>66.8</td><td>49.6</td></tr><tr><td>+ ICL-Router</td><td>91.6</td><td>75.4</td><td>30.0</td><td>25.7</td><td>12.7</td><td>3.0</td><td>90.2</td><td>59.8</td><td>67.6</td><td>50.7</td></tr><tr><td>+ LinUCB</td><td>92.0</td><td>71.6</td><td>25.2</td><td>23.7</td><td>13.0</td><td>3.3</td><td>91.2</td><td>59.4</td><td>70.6</td><td>50.0</td></tr><tr><td>+ RouteMoA</td><td>93.5</td><td>75.6</td><td>28.8</td><td>32.7</td><td>15.1</td><td>3.5</td><td>92.8</td><td>62.0</td><td>74.2</td><td>53.1</td></tr><tr><td>+ GSPO</td><td>91.4</td><td>76.1</td><td>35.6</td><td>59.9</td><td>27.3</td><td>12.5</td><td>91.0</td><td>64.2</td><td>76.6</td><td>59.4</td></tr><tr><td>+ MAPoRL</td><td>91.7</td><td>75.0</td><td>38.4</td><td>64.2</td><td>23.8</td><td>6.4</td><td>89.0</td><td>59.8</td><td>76.4</td><td>58.3</td></tr><tr><td>+ AT-GRPO</td><td>91.4</td><td>77.6</td><td>38.0</td><td>63.8</td><td>29.8</td><td>12.0</td><td>90.4</td><td>66.0</td><td>79.8</td><td>61.0</td></tr><tr><td>+ CERA-MoA (Ours)</td><td>93.5</td><td>79.6</td><td>41.8</td><td>60.3</td><td>31.3</td><td>13.4</td><td>92.8</td><td>72.4</td><td>83.4</td><td>63.2</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>62.9</td><td>34.6</td><td>10.4</td><td>41.2</td><td>4.9</td><td>0.8</td><td>66.4</td><td>41.0</td><td>39.8</td><td>33.6</td></tr><tr><td>+ ICL-Router</td><td>64.8</td><td>40.8</td><td>10.8</td><td>43.6</td><td>7.2</td><td>1.3</td><td>68.6</td><td>46.0</td><td>47.0</td><td>36.7</td></tr><tr><td>+ LinUCB</td><td>65.3</td><td>41.2</td><td>14.6</td><td>42.0</td><td>7.5</td><td>1.5</td><td>69.2</td><td>44.6</td><td>47.8</td><td>37.1</td></tr><tr><td>+ RouteMoA</td><td>79.2</td><td>47.3</td><td>16.2</td><td>49.0</td><td>9.7</td><td>1.3</td><td>79.0</td><td>50.8</td><td>57.4</td><td>43.3</td></tr><tr><td>+ GSPO</td><td>81.5</td><td>47.1</td><td>14.8</td><td>28.0</td><td>7.8</td><td>2.0</td><td>74.2</td><td>48.8</td><td>55.4</td><td>40.0</td></tr><tr><td>+ MAPoRL</td><td>81.3</td><td>49.1</td><td>14.8</td><td>40.9</td><td>6.8</td><td>1.1</td><td>73.6</td><td>43.2</td><td>57.8</td><td>41.0</td></tr><tr><td>+ AT-GRPO</td><td>80.8</td><td>48.8</td><td>16.2</td><td>25.3</td><td>7.9</td><td>2.5</td><td>74.2</td><td>51.6</td><td>59.4</td><td>40.7</td></tr><tr><td>+ CERA-MoA (Ours)</td><td>84.0</td><td>51.7</td><td>21.8</td><td>44.7</td><td>11.3</td><td>2.0</td><td>83.2</td><td>64.2</td><td>68.4</td><td>47.9</td></tr><tr><td>Phi-4-mini-Instruct</td><td>85.0</td><td>56.7</td><td>17.6</td><td>53.3</td><td>10.6</td><td>2.3</td><td>78.6</td><td>43.6</td><td>47.2</td><td>43.9</td></tr><tr><td>+ ICL-Router</td><td>87.6</td><td>62.6</td><td>22.8</td><td>52.5</td><td>11.5</td><td>2.9</td><td>81.0</td><td>43.4</td><td>54.2</td><td>46.5</td></tr><tr><td>+ LinUCB</td><td>86.9</td><td>63.0</td><td>20.6</td><td>51.4</td><td>14.9</td><td>3.0</td><td>82.0</td><td>43.4</td><td>54.2</td><td>46.6</td></tr><tr><td>+ RouteMoA</td><td>91.5</td><td>67.8</td><td>24.2</td><td>60.7</td><td>15.5</td><td>4.7</td><td>87.4</td><td>49.2</td><td>61.6</td><td>51.4</td></tr><tr><td>+ GSPO</td><td>91.0</td><td>65.4</td><td>20.6</td><td>56.4</td><td>22.7</td><td>8.3</td><td>87.2</td><td>56.2</td><td>73.4</td><td>53.5</td></tr><tr><td>+ MAPoRL</td><td>89.6</td><td>67.1</td><td>24.0</td><td>59.9</td><td>19.9</td><td>6.7</td><td>82.8</td><td>52.2</td><td>75.2</td><td>53.0</td></tr><tr><td>+ AT-GRPO</td><td>89.8</td><td>66.9</td><td>23.6</td><td>54.9</td><td>22.1</td><td>8.4</td><td>87.0</td><td>59.2</td><td>75.4</td><td>54.1</td></tr><tr><td>+ CERA-MoA (Ours)</td><td>91.2</td><td>68.6</td><td>26.8</td><td>59.5</td><td>22.2</td><td>7.7</td><td>87.6</td><td>61.0</td><td>76.0</td><td>55.6</td></tr><tr><td>Heterogeneous pool</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ ICL-Router</td><td>90.0</td><td>68.1</td><td>26.4</td><td>55.3</td><td>13.2</td><td>2.5</td><td>84.2</td><td>56.4</td><td>67.6</td><td>51.5</td></tr><tr><td>+ LinUCB</td><td>91.7</td><td>70.9</td><td>26.0</td><td>46.3</td><td>13.0</td><td>3.1</td><td>90.2</td><td>57.2</td><td>69.6</td><td>52.0</td></tr><tr><td>+ RouteMoA</td><td>93.1</td><td>68.7</td><td>25.2</td><td>63.0</td><td>16.9</td><td>4.0</td><td>90.4</td><td>56.6</td><td>75.2</td><td>54.8</td></tr><tr><td>+ MAPoRL</td><td>91.6</td><td>70.0</td><td>29.8</td><td>57.6</td><td>23.9</td><td>9.0</td><td>88.8</td><td>61.2</td><td>76.6</td><td>56.5</td></tr><tr><td>+ AT-GRPO</td><td>93.9</td><td>76.2</td><td>35.0</td><td>63.0</td><td>30.1</td><td>13.1</td><td>91.8</td><td>69.2</td><td>84.0</td><td>61.8</td></tr><tr><td>+ CERA-MoA (Ours)</td><td>93.1</td><td>81.5</td><td>44.2</td><td>59.5</td><td>23.8</td><td>12.5</td><td>92.8</td><td>75.4</td><td>83.8</td><td>63.0</td></tr></table>

Table 1: Comparison of CERA-MoA with baselines on in-distribution (ID) tasks across diferent base models. The best is in bold and the second-best is underlined.

The relative advantage of agent i is then computed by comparing its mean reward against the average performance across all participating agents:

$$
A _ { i , q } = r _ { i , q } - \frac { 1 } { \vert S _ { q } \vert } \sum _ { j \in S _ { q } } r _ { j , q } .\tag{14}
$$

For solo-activated agents, we set $A _ { i , q } = r _ { i , q }$ to ensure a steady increase in familiarity score. To dynamically align routing decisions with evolving agent competencies, the router loss optimizes the predictor head $g _ { i } \mathbf { \cdot }$

$$
\mathcal { L } _ { \mathrm { r o u t e r } } ( g _ { i } ) = \sum _ { q \in \mathcal { Q } } \left\{ \begin{array} { l l } { A _ { i , q } d _ { i } ( q ) , } & { \mathrm { i f ~ } A _ { i , q } \geq 0 ; } \\ { | A _ { i , q } | \operatorname* { m a x } ( 0 , m - d _ { i } ( q ) ) , } & { \mathrm { i f ~ } A _ { i , q } < 0 , } \end{array} \right.\tag{15}
$$

where m is the predefined separation margin. Minimizing this objective encourages high-performing agents to shrink their projection distance $d _ { i } ( \bar { \boldsymbol { q } } )$ , directly increasing their exponential familiarity score $f _ { i } ( q )$ , and vice versa.

The co-evolution framework is agnostic to how agents and the router are parameterized. In practice, CERA-MoA supports flexible configurations: sharing a single base backbone across agents via independent LoRA adapters (Hu et al. 2021) for storage and memory eficiency, or deploying heterogeneous base models to exploit diverse agent capabilities.

## 4 Experiments

The empirical evaluation is designed to answer the following research questions:

• RQ1: Does CERA-MoA outperform static agent routing baselines and fixed orchestration post-training paradigms?

• RQ2: How efectively does CERA-MoA adapt to heterogeneous base model pools?

• RQ3: Is the predictive familiarity estimator superior to direct reward estimation or multi-class classification approaches for query allocation?

• RQ4: Can the cumulative threshold adaptive routing effectively balance task performance and the number of activated agents?

• RQ5: How does CERA-MoA explicitly induce diferent capability specialization among agents?

## 4.1 Experimental Setup

Datasets. We evaluate across four domains: mathematica reasoning (GSM8k (Cobbe et al. 2021), Hendrycks MATH (Hendrycks et al. 2021), DAPO-MATH-17k (Yu et al. 2026)), code generation (MBPP (Austin et al. 2021), Eurus-2-Code (Yuan et al. 2024), TACO (Li et al. 2023)), instruction following (MAGPIE-IF (Xu et al. 2025), RLVR-IFEval (Lambert et al. 2024)), and general reasoning (BIG-bench Hard (Suzgun et al. 2023)). We train both the agents and the router using the training sets and report performance on their independently partitioned test sets. Out-of-distribution (OOD) generalization is evaluated on IFEval (Zhou et al. 2023), HumanEval (Chen et al. 2021), AGIEval (Zhong et al. 2023), ARC-c (Clark et al. 2018), LogicBench (Parmar et al. 2024), and OlympiadBench (He et al. 2024), without any additional fine-tuning.

Baselines. We compare CERA-MoA against static agent orchestration optimization and predetermined-workflow agent fine-tuning baselines. Orchestration optimization baselines include ICL-Router (Wang et al. 2026a), which constructs capability profiles within a reconstructed latent space; LinUCB (Poon et al. 2026), which formulates query routing as a contextual bandit problem to dynamically select agents; and RouteMoA (Wang et al. 2026b), which employs a contrastive-learning scorer to activate multiple expert models. Agent fine-tuning baselines include the single-agent reinforcement learning baseline GSPO (Zheng et al. 2025); MAPoRL (Park et al. 2025), which trains agents within a multi-agent debate framework using task and cross-agent correction rewards; and AT-GRPO (Zhao et al. 2026), which applies Group Relative Policy Optimization within a predefined tree-structured MoA architecture.

Implementation Settings. For the sharing base model setting, we configure a population of N = 4 agents, equipping each agent with an independent Low-Rank Adaptation (LoRA) module. For orchestration baselines, we simulate four experts by applying role-specific prompt instructions to the base model. This architecture allows us to deploy a 4- agent MoA system on a 4B parameter base model with a highly eficient aggregate footprint of only ∼4.4B parameters. For the heterogeneous model setting, we configure an ensemble of N = 3 agents by directly calling three distinct open-source base models: Qwen3-4B (Yang et al. 2025), Llama-3.2-3B-Instruct (Grattafiori et al. 2024) and Phi-4- mini-Instruct (Abouelenin et al. 2025). To maintain a fair evaluation across all baselines, we enforce identical generation hyper-parameters as well as task-specific reward functions throughout all training and inference runs as detailed in the Technical Appendix.

## 4.2 Main Results (RQ1)

Tables 1 and 2 compare CERA-MoA with orchestration optimization methods and fixed-workflow fine-tuning approaches for both in-distribution (ID) and out-of-distribution (OOD) tasks. CERA-MoA achieves the highest average performance across all three base models, demonstrating the broad applicability of our co-evolutionary framework across diferent model architectures and capacities.

<table><tr><td>Method</td><td>IFEval HumanEval AGIEval ARC-c LogicBench Olympiad Avg.</td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Qwen3-4B</td><td>79.3</td><td>36.4</td><td>64.0</td><td>91.7</td><td>17.0</td><td>29.7</td><td>53.0</td></tr><tr><td>+ ICL-Router</td><td>78.0</td><td>37.0</td><td>67.6</td><td>91.1</td><td>17.1</td><td>35.5</td><td>54.4</td></tr><tr><td>+ LinUCB</td><td>77.6</td><td>30.5</td><td>64.0</td><td>92.7</td><td>17.4</td><td>29.7</td><td>52.0</td></tr><tr><td>+ RouteMoA</td><td>83.0</td><td>44.2</td><td>67.5</td><td>94.4</td><td>25.7</td><td>32.5</td><td>57.9</td></tr><tr><td>+ GSPO</td><td>79.1</td><td>78.6</td><td>68.2</td><td>92.5</td><td>65.4</td><td>39.2</td><td>70.5</td></tr><tr><td>+ MAPoRL</td><td>78.7</td><td>85.1</td><td>66.4</td><td>93.3</td><td>65.5</td><td>34.2</td><td>70.5</td></tr><tr><td>+ AT-GRPO</td><td>78.6</td><td>84.4</td><td>69.1</td><td>93.2</td><td>64.8</td><td>38.9</td><td>71.5</td></tr><tr><td>+ CERA-MoA</td><td>83.0</td><td>82.5</td><td>71.3</td><td>92.7</td><td>65.6</td><td>41.6</td><td>72.8</td></tr><tr><td>Llama-3.2-3B-Instruct</td><td>62.3</td><td>48.7</td><td>29.7</td><td>71.6</td><td>37.2</td><td>8.2</td><td>43.0</td></tr><tr><td>+ ICL-Router</td><td>69.7</td><td>49.4</td><td>35.8</td><td>72.6</td><td>41.6</td><td>12.0</td><td>46.9</td></tr><tr><td>+ LinUCB</td><td>67.8</td><td>48.1</td><td>36.0</td><td>73.6</td><td>41.5</td><td>11.6</td><td>46.4</td></tr><tr><td>+ RouteMoA</td><td>69.7</td><td>61.7</td><td>43.4</td><td>79.6</td><td>49.3</td><td>13.7</td><td>52.9</td></tr><tr><td>+ GSPO</td><td>58.0</td><td>55.8</td><td>42.3</td><td>78.3</td><td>44.8</td><td>15.3</td><td>49.1</td></tr><tr><td>+ MAPoRL</td><td>53.0</td><td>51.9</td><td>41.8</td><td>79.2</td><td>44.9</td><td>13.7</td><td>47.4</td></tr><tr><td>+ AT-GRPO</td><td>71.0</td><td>50.0</td><td>41.6</td><td>78.7</td><td>49.3</td><td>15.6</td><td>51.0</td></tr><tr><td>+ CERA-MoA</td><td>72.1</td><td>62.3</td><td>46.2</td><td>80.2</td><td>44.9</td><td>17.4</td><td>53.9</td></tr><tr><td>Phi-4-mini-Instruct</td><td>69.9</td><td>57.8</td><td>50.5</td><td>71.2</td><td>18.0</td><td>24.3</td><td>48.6</td></tr><tr><td>+ ICL-Router</td><td>67.1</td><td>66.2</td><td>56.9</td><td>71.5</td><td>27.6</td><td>27.9</td><td>52.9</td></tr><tr><td>+ LinUCB</td><td>70.4</td><td>66.9</td><td>56.4</td><td>69.9</td><td>33.0</td><td>28.3</td><td>54.2</td></tr><tr><td>+ RouteMoA</td><td>70.2</td><td>79.9</td><td>59.9</td><td>82.8</td><td>27.1</td><td>29.8</td><td>58.3</td></tr><tr><td>+ GSPO</td><td>68.9</td><td>70.8</td><td>58.6</td><td>87.8</td><td>35.9</td><td>27.6</td><td>58.3</td></tr><tr><td>+ MAPoRL</td><td>60.6</td><td>80.5</td><td>58.8</td><td>88.1</td><td>34.1</td><td>28.3</td><td>58.4</td></tr><tr><td>+ AT-GRPO</td><td>73.9</td><td>74.0</td><td>59.6</td><td>88.4</td><td>35.3</td><td>29.0</td><td>60.0</td></tr><tr><td>+ CERA-MoA</td><td>74.7</td><td>77.3</td><td>60.7</td><td>89.6</td><td>42.6</td><td>29.8</td><td>62.5</td></tr><tr><td>Heterogeneous pool</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>+ ICL-Router</td><td>74.9</td><td>50.0</td><td>60.1</td><td>89.3</td><td>18.9</td><td>31.0</td><td>54.0</td></tr><tr><td>+ LinUCB</td><td>77.6</td><td>52.6</td><td>64.0</td><td>92.3</td><td>17.2</td><td>30.8</td><td>55.8</td></tr><tr><td>+ RouteMoA</td><td>78.6</td><td>53.9</td><td>61.4</td><td>92.4</td><td>26.1</td><td>30.7</td><td>57.2</td></tr><tr><td>+ MAPoRL</td><td>68.9</td><td>78.6</td><td>62.3</td><td>89.6</td><td>54.5</td><td>31.8</td><td>64.3</td></tr><tr><td>+ AT-GRPO</td><td>81.3</td><td>83.8</td><td>67.8</td><td>92.9</td><td>50.9</td><td>37.6</td><td>69.1</td></tr><tr><td>+ CERA-MoA</td><td>83.4</td><td>77.3</td><td>71.1</td><td>92.7</td><td>62.4</td><td>44.1</td><td>71.8</td></tr></table>

Table 2: Comparison of CERA-MoA with baselines on outof-distribution (OOD) tasks.

For in-distribution tasks, our framework achieves substantial average gains. This improvement highlights the benefit of dynamically allocating training queries based on evolving agent capabilities, which enables more targeted policy updates and complementary specialization. For out-ofdistribution benchmarks, CERA-MoA still achieves the highest average transfer scores. These results suggest that the learned capability diferentiation can transfer beyond the training distribution, allowing the agents to leverage their learned competencies on unseen tasks.

## 4.3 Adaptation to Heterogeneous Models (RQ2)

We evaluate CERA-MoA on a heterogeneous ensemble combining Qwen3-4B, Llama-3.2-3B, and Phi-4-mini. As shown in the bottom rows of Tables 1 and 2, CERA-MoA still outperforms all baselines, achieving the highest average scores for both ID tasks and OOD transfer.

Fixed-workflow paradigms often force diverse architectures into an unnatural consensus, which can dilute their unique pre-training strengths. In contrast, CERA-MoA leverages semantic familiarity scores to dynamically route each query to the most suitable backbone model. This capabilityaware alignment is particularly efective for complex reasoning. As a result, our framework achieves substantial performance gains over the strongest post-training baselines on exceptionally challenging reasoning datasets (e.g., DAPO-MATH, LogicBench and OlympiadBench). These findings suggest that CERA-MoA exhibits strong adaptability across diferent model architectures, efectively turning model diversity into a collaborative advantage.

<table><tr><td>Method / Configuration</td><td>ID Avg.</td><td></td><td>OOD Avg. Avg. Tokens</td></tr><tr><td>CERA-MoA (Adaptive Threshold)</td><td>63.2</td><td>72.8</td><td>367.77</td></tr><tr><td>Routing Metric (RQ3)</td><td></td><td></td><td></td></tr><tr><td>Direct Reward Regression</td><td>60.8</td><td>70.1</td><td></td></tr><tr><td>Multi-Class Classification</td><td>61.3</td><td>70.4</td><td></td></tr><tr><td>Allocation Strategy (RQ4)</td><td></td><td></td><td></td></tr><tr><td>CERA-MoA (Fixed Top-1)</td><td>61.4</td><td>70.1</td><td>325.70</td></tr><tr><td>CERA-MoA (Fixed Top-2)</td><td>63.3</td><td>72.5</td><td>666.37</td></tr></table>

Table 3: Ablation results on Qwen3-4B.

![](images/520de9ad704d1eaf8cfcfffa0a8f5e123b59c1d44d5f2f64d3b6e65631d1969a.jpg)

![](images/f201d629c9471c30537960ce80e908bc96e4565008f5f85520f1d46b973010b8.jpg)  
Figure 2: Comparison of routing metric dynamics.

## 4.4 Ablation on Familiarity Score (RQ3)

We compare our predictive familiarity estimator against two alternative routing mechanisms based on intermediate hidden states. As reported in Table 3, direct reward regression and reward-driven multi-class classification both fall noticeably short of CERA-MoA.

Figure 2 explains this gap by plotting familiarity scores and estimated rewards across training queries. While estimated rewards exhibit oscillations, our familiarity score evolves smoothly and converges steadily. Direct reward regression and reward-driven optimization fail because rewards are inherently unstable. As agent policies evolve during training, their performance on identical prompts constantly changes. In contrast, our predictive familiarity estimator translates these dynamic shifts into a geometric push-and-pull distance between two network heads. This mechanism naturally adapts to continual learning dynamics, ensuring stable sample allocation and deeper specialization.

## 4.5 Ablation on Adaptive Routing (RQ4)

We evaluate how our cumulative-threshold adaptive routing strategy balances reasoning accuracy against computational overhead. Table 3 compares our dynamic allocation mechanism against fixed Top-k activation strategies on Qwen3-4B.

Within our framework, enforcing a fixed Top-2 routing strategy yields strong reasoning accuracy but incurs high token costs. Conversely, a fixed Top-1 strategy reduces computational overhead, however, it sufers a notable performance drop because a single agent struggles to resolve complex queries alone. Our cumulative-threshold mechanism resolves this trade-of. The router generally activates multiple agents only for complex problems that exceed an individual agent’s expertise. Compared with fixed Top-2 routing, CERA-MoA reduces average generation tokens by approximately 45% while retaining comparable performance.

![](images/69daa0622a4605849f75833de9fefe4983943dcb0c03d5b78f7f479d132b3b47.jpg)  
Figure 3: Capability analysis of Qwen3-4B agents.

## 4.6 Capability Specialization Analysis (RQ5)

We investigate how our closed-loop co-evolution explicitly promotes distinct problem-solving specialization without human intervention or role assignment. We analyze the emergent behavioral profiles of the four Qwen3-4B agents.

As shown in Figure 3 (Left), the agent population spontaneously diferentiated into complementary specialists. Due to long chain-of-thought reasoning, Agent 2 was allocated the majority of queries as a primary solver. To supplement this foundational reasoning capacity, Agent 1 mastered concise math and logic, Agent 3 focused on coding, and Agent 4 specialized in general reasoning. Figure 3 (Right) explains this emergence through a t-SNE visualization of intermediate hidden states. Queries routed to the same agent form tightly clustered semantic neighborhoods. By consistently allocating similar prompts to the most receptive model during training, our predictive familiarity estimator drives deep policy diferentiation.

## 5 Conclusion

This paper introduces CERA-MoA, which co-evolves query routing with continually learning LLM agents. By evaluating agent competence via familiarity scores based on hiddenstate semantics and applying cumulative-threshold routing, CERA-MoA balances eficiency with reasoning accuracy while inducing domain specialization, significantly outperforming static routing and fixed-workflow training baselines.

For future work, extending our single-turn semantic routing to multi-turn interactions or multi-agent debate could better support iterative problem-solving. Additionally, upgrading our familiarity-weighted voting to advanced answer aggregation mechanisms, such as an adaptive meta-thinker, promises to further enhance collective reasoning.

## References

Abouelenin, A.; Ashfaq, A.; Atkinson, A.; Awadalla, H.; Bach, N.; Bao, J.; et al. 2025. Phi-4-mini technical report: Compact yet powerful multimodal language models via mixture-of-loras. arXiv preprint arXiv:2503.01743.

Austin, J.; Odena, A.; Nye, M.; Bosma, M.; Michalewski, H.; Dohan, D.; Jiang, E.; et al. 2021. Program synthesis with large language models. arXiv preprint arXiv:2108.07732.

Chan, C.-M.; Chen, W.; Su, Y.; Yu, J.; Xue, W.; Zhang, S.; Fu, J.; and Liu, Z. 2024. Chateval: Towards better llm-based evaluators through multi-agent debate. In ICLR, volume 2024, 9079–9093.

Chen, J.; Saha, S.; and Bansal, M. 2024. Reconcile: Roundtable conference improves reasoning via consensus among diverse llms. In Proc. ofACL, 7066–7085.

Chen, M.; Tworek, J.; Jun, H.; Yuan, Q.; de Oliveira Pinto, H. P.; Kaplan, J.; Edwards, H.; et al. 2021. Evaluating Large Language Models Trained on Code. arXiv:2107.03374.

Clark, P.; Cowhey, I.; Etzioni, O.; Khot, T.; Sabharwal, A.; Schoenick, C.; and Tafjord, O. 2018. Think you have Solved Question Answering? Try ARC, the AI2 Reasoning Challenge. arXiv:1803.05457.

Cobbe, K.; Kosaraju, V.; Bavarian, M.; Chen, M.; Jun, H.; Kaiser, L.; Plappert, M.; Tworek, J.; Hilton, J.; Nakano, R.; et al. 2021. Training Verifiers to Solve Math Word Problems. arXiv preprint arXiv:2110.14168.

Dang, Y.; Qian, C.; Luo, X.; Fan, J.; Xie, Z.; Shi, R.; Chen, W.; Yang, C.; Che, X.; et al. 2026. Multi-agent collaboration via evolving orchestration. NeurIPS, 38: 165025–165059.

Estornell, A.; and Liu, Y. 2024. Multi-LLM debate: Framework, principals, and interventions. NeurIPS, 37: 28938– 28964.

Fan, W.; Yoon, J.; and Ji, B. 2026. imad: Intelligent multiagent debate for eficient and accurate llm inference. In Proc. ofAAAI, volume 40, 29403–29411.

Grattafiori, A.; Dubey, A.; Jauhri, A.; Pandey, A.; Kadian, A.; Al-Dahle, A.; Letman, A.; Mathur, A.; et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

He, C.; Luo, R.; Bai, Y.; Hu, S.; Thai, Z. L.; Shen, J.; Hu, J.; et al. 2024. Olympiadbench: A challenging benchmark for promoting agi with olympiad-level bilingual multimodal scientific problems. arXiv preprint arXiv:2402.14008.

Hendrycks, D.; Burns, C.; Kadavath, S.; Arora, A.; Basart, S.; Tang, E.; Song, D.; and Steinhardt, J. 2021. Measuring mathematical problem solving with the math dataset. arXiv preprint arXiv:2103.03874.

Hong, S.; Zhuge, M.; Chen, J.; Zheng, X.; Cheng, Y.; Wang, J.; Zhang, C.; Yau, S.; Lin, Z.; Zhou, L.; et al. 2024. MetaGPT: Meta programming for a multi-agent collaborative framework. In ICLR, volume 2024, 23247–23275.

Hu, E. J.; Shen, Y.; Wallis, P.; Allen-Zhu, Z.; Li, Y.; Wang, S.; Wang, L.; and Chen, W. 2021. LoRA: Low-Rank Adaptation of Large Language Models. arXiv:2106.09685.

Hu, T.; Tan, Z.; Wang, S.; Qu, H.; and Chen, T. 2026. Multiagent debate for llm judges with adaptive stability detection. NeurIPS, 38: 46504–46540.

Ke, Z.; Ming, Y.; Xu, A.; Chin, R.; Nguyen, X.-P.; Jwalapuram, P.; et al. 2026. MAS-Orchestra: Understanding and Improving Multi-Agent Reasoning Through Holistic Orchestration and Controlled Benchmarks. arXiv:2601.14652.

Lambert, N.; Morrison, J.; Pyatkin, V.; Huang, S.; Ivison, H.; Brahman, F.; Miranda, L. J. V.; Liu, A.; Dziri, N.; Lyu, S.; et al. 2024. Tulu 3: Pushing frontiers in open language model post-training. arXiv preprint arXiv:2411.15124.

Lee, H.; Oh, S.; Kim, J.; Shin, J.; and Tack, J. 2025. Re-VISE: Learning to Refine at Test-Time via Intrinsic Self-Verification. In Proc. ofICML, 33616–33634. PMLR.

Lee, S.; Kim, D.; Koh, H.; Yang, N.; and Jung, K. 2026. Confidence-Guided Stepwise Model Routing for Cost-Eficient Reasoning. In Proc. ofAAAI, volume 40, 31483– 31491.

Li, R.; Fu, J.; Zhang, B.-W.; Huang, T.; Sun, Z.; Lyu, C.; Liu, G.; Jin, Z.; and Li, G. 2023. Taco: Topics in algorithmic code generation dataset. arXiv preprint arXiv:2312.14852.

Li, S.; Liu, Y.; Zheng, Y.; Li, M.; Nguyen, Q. V. H.; and Pan, S. 2026. OFA-MAS: One-for-all multi-agent system topology design based on mixture-of-experts graph generative models. In Proc. ofWWW, 1333–1344.

Liang, T.; He, Z.; Jiao, W.; Wang, X.; Wang, Y.; Wang, R.; Yang, Y.; Shi, S.; and Tu, Z. 2024. Encouraging divergent thinking in large language models through multi-agent debate. In Proc. ofEMNLP, 17889–17904.

Lv, A.; Ma, J.; Ma, Y.; and Qiao, S. 2025. Coupling Experts and Routers in Mixture-of-Experts via an Auxiliary Loss. arXiv:2512.23447.

Motwani, S. R.; Smith, C.; Das, R. J.; Rafailov, R.; Laptev, I.; Torr, P. H. S.; Pizzati, F.; Clark, R.; and de Witt, C. S. 2025. MALT: Improving Reasoning with Multi-Agent LLM Training. arXiv:2412.01928.

Oldfield, J.; Georgopoulos, M.; Chrysos, G. G.; Tzelepis, C.; Panagakis, Y.; Nicolaou, M. A.; Deng, J.; and Patras, I. 2024. Multilinear mixture of experts: Scalable expert specialization through factorization. NeurIPS, 37: 53022–53063.

Park, C.; Han, S.; Guo, X.; Ozdaglar, A. E.; Zhang, K.; and Kim, J.-K. 2025. Maporl: Multi-agent post-co-training for collaborative large language models with reinforcement learning. In Proc. ofACL, 30215–30248.

Parmar, M.; Patel, N.; Varshney, N.; Nakamura, M.; Luo, M.; et al. 2024. LogicBench: Towards Systematic Evaluation of Logical Reasoning Ability of Large Language Models. arXiv:2404.15522.

Poon, M.; Dai, X.; Liu, X.; Kong, F.; Lui, J. C.; and Zuo, J. 2026. Online multi-llm selection via contextual bandits under unstructured context evolution. In Proc. ofAAAI, volume 40, 24855–24863.

Puerto, H.; Chubakov, T.; Zhu, X.; Madabushi, H. T.; and Gurevych, I. 2025. Fine-Tuning on Diverse Reasoning Chains Drives Within-Inference CoT Refinement in LLMs. In Proc. ofACL, 3789–3808.

Qiao, D.; Chen, B.; Cai, F.; Chen, J.; Li, W.; Jiang, F.; Chen, Z.; Zha, H.; Zhang, T.; and Wang, B. 2026. Epistemic Gain, Aleatoric Cost: Uncertainty Decomposition in Multi-Agent Debate for Math Reasoning. arXiv:2603.01221.

Skean, O.; Arefin, M. R.; Zhao, D.; Patel, N. N.; Naghiyev, J.; Lecun, Y.; and Shwartz-Ziv, R. 2025. Layer by Layer: Uncovering Hidden Representations in Language Models. In Proc. of ICML, 55854–55875. PMLR.

Suzgun, M.; Scales, N.; Schärli, N.; Gehrmann, S.; Tay, Y.; Chung, H. W.; Chowdhery, A.; Le, Q.; Chi, E. H.; et al. 2023. Challenging big-bench tasks and whether chain-of-thought can solve them. In Findings ofACL, 13003–13051.

Taubenfeld, A.; Shefer, T.; Ofek, E.; Feder, A.; Goldstein, A.; Gekhman, Z.; and Yona, G. 2025. Confidence improves self-consistency in llms. In Findings ofACL, 20090–20111.

Wang, C.; Li, H.; Zhang, Y.; Chen, L.; Chen, J.; Jian, P.; Zhang, Q.; and Hu, S. 2026a. Icl-router: In-context learned model representations for llm routing. In Proc. of AAAI, volume 40, 33413–33421.

Wang, H.; Maia Polo, F.; Sun, Y.; Kundu, S.; Xing, E.; and Yurochkin, M. 2024. Fusing Models with Complementary Expertise. In ICLR, volume 2024, 45284–45306.

Wang, J.; Wang, J.; Athiwaratkun, B.; Zhang, C.; and Zou, J. Y. 2025a. Mixture-of-agents enhances large language model capabilities. In ICLR, volume 2025, 33944–33963.

Wang, J.; Wu, H.; You, Z.; Song, Y.; Wang, Y.; Shan, Z.; Li, Y.; Zhang, S.; Le, X.; Chen, C.; et al. 2026b. Route-MoA: Dynamic Routing without Pre-Inference Boosts Eficient Mixture-of-Agents. arXiv preprint arXiv:2601.18130.

Wang, J.; Zhao, S.; Liu, J.; Wang, H.; Li, W.; Qin, B.; and Liu, T. 2026c. Orchestrating Intelligence: Confidence-Aware Routing for Eficient Multi-Agent Collaboration across Multi-Scale Models. arXiv preprint arXiv:2601.04861.

Wang, X.; Liu, Y.; Cheng, W.; Zhao, X.; Chen, Z.; Yu, W.; et al. 2025b. Mixllm: Dynamic routing in mixed large language models. In Proc. ofNAACL, 10912–10922.

Wang, Z.; Liu, X.; Wang, L.; Shan, Z.; Wang, Y.; Song, Z.; and Zhang, M. 2026d. MASPO: Joint Prompt Optimization for LLM-based Multi-Agent Systems. arXiv:2605.06623.

Wang, Z.; Wang, Y.; Liu, X.; Ding, L.; Zhang, M.; Liu, J.; and Zhang, M. 2025c. Agentdropout: Dynamic agent elimination for token-eficient and high-performance llm-based multiagent collaboration. In Proc. ofACL, 24013–24035.

Xia, Y.; Kong, F.; Yu, T.; Guo, L.; Rossi, R. A.; Kim, S.; and Li, S. 2024. Which llm to play? convergence-aware online model selection with time-increasing bandits. In Proc. of WWW, 4059–4070.

Xie, Z.; Han, C.; Shi, J.; Cui, W.; Zhao, W. X.; Wu, X.; and Zhao, J. 2025. Rmoa: Optimizing mixture-of-agents through diversity maximization and residual compensation. In Findings ofACL, 6575–6602.

Xu, Z.; Jiang, F.; Niu, L.; Deng, Y.; Poovendran, R.; Choi, Y.; and Lin, B. Y. 2025. Magpie: Alignment data synthesis from scratch by prompting aligned llms with nothing. In ICLR, volume 2025, 76346–76382.

Xue, J.; Lou, Q.; Xing, J.; and Huang, H. 2026a. R2- Router: A New Paradigm for LLM Routing with Reasoning. arXiv:2602.02823.

Xue, X.; Zhou, Y.; Zhang, G.; Zhang, Z.; Li, Y.; Zhang, C.; Yin, Z.; Torr, P.; Ouyang, W.; and Bai, L. 2026b. CoMAS:

Co-Evolving Multi-Agent Systems via Interaction Rewards. arXiv:2510.08529.

Yang, A.; Li, A.; Yang, B.; Zhang, B.; Hui, B.; Zheng, B.; Yu, B.; Gao, C.; Huang, C.; Lv, C.; et al. 2025. Qwen3 technical report. arXiv preprint arXiv:2505.09388.

Ye, R.; Tang, S.; Ge, R.; Du, Y.; Yin, Z.; Chen, S.; and Shao, J. 2025. MAS-GPT: Training LLMs to build LLM-based multi-agent systems. arXiv preprint arXiv:2503.03686.

Yi, X.; Zhou, Z.; Cao, C.; Niu, Q.; Liu, T.; and Han, B. 2025. From Debate to Equilibrium: Belief-Driven Multi-Agent LLM Reasoning via Bayesian Nash Equilibrium. arXiv preprint arXiv:2506.08292.

Yu, Q.; Zhang, Z.; Zhu, R.; Yuan, Y.; Zuo, X.; Yue, Y.; Dai, W.; Fan, T.; Liu, G.; Liu, L.; et al. 2026. Dapo: An opensource llm reinforcement learning system at scale. NeurIPS, 38: 113222–113244.

Yuan, L.; Li, W.; Chen, H.; Cui, G.; Ding, N.; Zhang, K.; Zhou, B.; Liu, Z.; and Peng, H. 2024. Free process rewards without process labels. arXiv preprint arXiv:2412.01981.

Yuan, Y.; and Xie, T. 2025. Reinforce LLM Reasoning through Multi-Agent Reflection. In ICML, 73701–73731. PMLR.

Yue, Y.; Zhang, G.; Liu, B.; Wan, G.; Wang, K.; Cheng, D.; and Qi, Y. 2025. Masrouter: Learning to route llms for multi-agent systems. In Proc. of ACL, 15549–15572.

Zhang, G.; Yu, H.; Yang, K.; Wu, B.; Huang, F.; Li, Y.; and Yan, S. 2026. EvoRoute: Experience-Driven Self-Routing LLM Agent Systems. arXiv:2601.02695.

Zhang, K.; Liu, R.; Zhu, X.; Tian, K.; Zeng, S.; Jia, G.; Fan, Y.; Lv, X.; Zuo, Y.; Jiang, C.; et al. 2025. Marti: A framework for multi-agent llm systems reinforced training and inference. In The Fourteenth ICLR.

Zhao, X.; Kang, Z.; Feng, A.; Levine, S.; and Song, D. 2025. Learning to Reason without External Rewards. arXiv:2505.19590.

Zhao, X.; Wang, K.; and Peng, W. 2024. An electoral approach to diversify llm-based multi-agent collective decision-making. In Proc. ofEMNLP, 2712–2727.

Zhao, Y.; Hu, L.; Wang, Y.; Hou, M.; Zhang, H.; Ding, K.; and Zhao, J. 2026. Stronger-MAS: Multi-Agent Reinforcement Learning for Collaborative LLMs. arXiv:2510.11062.

Zheng, C.; Liu, S.; Li, M.; Chen, X.-H.; Yu, B.; Gao, C.; Dang, K.; Liu, Y.; Men, R.; Yang, A.; et al. 2025. Group Sequence Policy Optimization. arXiv:2507.18071.

Zhong, W.; Cui, R.; Guo, Y.; Liang, Y.; Lu, S.; Wang, Y.; et al. 2023. AGIEval: A Human-Centric Benchmark for Evaluating Foundation Models. arXiv:2304.06364.

Zhou, J.; Lu, T.; Mishra, S.; Brahma, S.; Basu, S.; et al. 2023. Instruction-following evaluation for large language models. arXiv preprint arXiv:2311.07911.

Zhu, J.; Chen, W.; Zhang, X.; Wu, Z.; and Dai, X. 2026. Recognize Your Orchestrator: An Entropy Dynamics Perspective for LLM Multi-Agent Systems. arXiv:2606.01351.

Zhuang, Y.; Shen, Y.; Bian, Y.; Su, Q.; Ji, S.; Shi, Y.; and Miao, F. 2025. LD-MoLE: Learnable Dynamic Routing for Mixture ofLoRA Experts. arXivpreprint arXiv:2509.25684.

## A Algorithm Pseudocode & Architecture Details

## A.1 Co-Evolutionary Training Procedure

Algorithm 1: CERA-MoA Co-Evolutionary Training   
Require: Training dataset D; agent policies $\{ \pi _ { \theta _ { i } } \} _ { i = 1 } ^ { N } ;$   
frozen target heads $\{ \bar { g } _ { i } \} _ { i = 1 } ^ { N } ;$ trainable predictor heads   
$\{ g _ { i } \} _ { i = 1 } ^ { N } ;$ cumulative threshold $\tau ;$ separation margin m   
1: Initialize an empty LIFO sample bufer $B _ { i }$ for each agent   
$i \in \{ 1 , \ldots , N \}$   
2: for each orchestration step do   
3: Sample a query batch $\dot { \mathcal { Q } } \subset \mathcal { D }$ and extract frozen se  
mantic representations $\{ h ( q ) : q \in \mathcal { Q } \}$   
4: for each query $q \in \mathcal { Q }$ do   
5: Compute raw familiarity scores $\{ f _ { i } ( q ) \} _ { i = 1 } ^ { N }$ via   
$\mathrm { E q . } \dot { 1 7 }$   
6: Calculate exploration routing scores $\{ s _ { i } ( q ) \} _ { i = 1 } ^ { N }$ via   
Eq. 18   
7: Sort agents in descending order by $s _ { i } ( \boldsymbol q )$   
8: Activate minimal prefix subset $S _ { q }$ satisfying   
$\begin{array} { r } { \sum _ { i \in S _ { q } } f _ { i } ( q ) \ge \tau } \end{array}$ (or activate all N agents if in  
feasible)   
9: Push tuple $( q , h ( q ) )$ into LIFO bufer $B _ { i }$ for every   
activated agent $i \in S _ { q }$   
10: end for   
11: for each agent $i$ with suficient samples in $B _ { i }$ do   
12: Pop a micro-batch $Q _ { i }$ from LIFO bufer $B _ { i }$   
13: Load adapter weights $\theta _ { i }$ and sample G trajectory   
rollouts per query $\bar { \boldsymbol { q } } \in Q _ { i }$   
14: Evaluate task rewards $\{ R ( q , a _ { i , q } ^ { ( g ) } ) \} _ { g = 1 } ^ { G }$ and update   
policy $\theta _ { i }$ via joint DAPO–GSPO optimization   
15: Record empirical mean reward $\begin{array} { r l } { \bar { r } _ { i , q } } & { { } = } \end{array}$   
$\begin{array} { r } { \frac { 1 } { G } \sum _ { g = 1 } ^ { G } R ( q , a _ { i , q } ^ { ( g ) } ) } \end{array}$ for synchronous router   
supervision   
16: end for   
17: for each agent i participating in routed queries do   
18: Compute relative advantage $\begin{array} { r c l } { { A _ { i , q } } } & { { = } } & { { \bar { r } _ { i , q } - } } \end{array}$   
$\begin{array} { r } { \frac { 1 } { \vert { \cal S } _ { q } \vert } \sum _ { j \in { \cal S } _ { q } } { \bar { r } } _ { j , q } \left( \mathrm { i f } \vert { \cal S } _ { q } \vert = 1 , \right. } \end{array}$ , set $A _ { i , q } = \bar { r } _ { i , q } )$   
19: Update predictor head $g _ { i }$ via Eq. 27, keeping $\bar { g } _ { i }$ and   
the backbone model frozen   
20: end for   
21: end for

We formally detail the end-to-end training methodology of CERA-MoA in Algorithm 1. Our co-evolutionary framework is inherently agnostic to the underlying parameterization of the agent population $\{ \pi _ { \boldsymbol { \theta } _ { i } } \} _ { i = 1 } ^ { N }$ , supporting two primary deployment paradigms. In the homogeneous setting, agents are instantiated as independent, lightweight Low-Rank Adaptation (LoRA) modules over a single shared backbone, achieving high parameter eficiency and low memory overhead during multi-agent rollouts. In the heterogeneous setting, each agent is powered by a distinct base model, allowing the router to actively exploit architectural diversity and complementary pre-training competencies. Crucially, query features are extracted from the frozen router backbone by concatenating the final non-padding hidden states from intermediate layers. Throughout the closed-loop optimization, only the agents policy parameters $\theta _ { i }$ and the router’s lightweight predictor heads $g _ { i }$ are updated.

Feature Extraction and Familiarity Estimation. For an incoming query $q ,$ the router tokenizes the prompt formatted for its specific task class and extracts the final non-padding hidden state from intermediate layers. Following our core findings, we concatenate the hidden states from layers $\lfloor L / 2 \rfloor$ and $\lfloor \bar { 3 } L / 4 \rfloor$ , yielding a rich semantic representation $h ( q ) \in$ $\mathbb { R } ^ { 2 H }$ for a backbone with hidden dimension H and total depth $L .$ . This mid-layer extraction captures deep syntactic and reasoning structures without overfitting to immediate next-token prediction dynamics.

To evaluate agent-query compatibility, each agent i is assigned a trainable predictor head $g _ { i }$ and a permanently frozen target head $\bar { g } _ { i }$ , both using an identical Multi-Layer Perceptron (MLP) architecture:

$$
\operatorname { L N } ( h ) \to \operatorname { L i n e a r } ( 2 H , 5 1 2 ) \to \operatorname { S i L U } \to \operatorname { L i n e a r } ( 5 1 2 , 2 5 6 ) .\tag{16}
$$

To establish a stable and diverse reference space across the population, the target heads $\{ \bar { g } _ { i } \} _ { i = 1 } ^ { N }$ are initialized orthogonally and remain strictly frozen. By projecting the representations onto a unit hypersphere, the semantic projection distance $d _ { i } ( q )$ and the resulting familiarity score $f _ { i } ( q )$ are formulated as:

$$
\begin{array} { r l } & { d _ { i } ( \boldsymbol { q } ) = \bigg \| \frac { g _ { i } \left( h ( \boldsymbol { q } ) \right) } { \left\| g _ { i } \left( h ( \boldsymbol { q } ) \right) \right\| _ { 2 } } - \frac { \bar { g } _ { i } \left( h ( \boldsymbol { q } ) \right) } { \left\| \bar { g } _ { i } \left( h ( \boldsymbol { q } ) \right) \right\| _ { 2 } } \bigg \| _ { 2 } , } \\ & { f _ { i } ( \boldsymbol { q } ) = \exp \left( - \lambda d _ { i } ( \boldsymbol { q } ) \right) , } \end{array}\tag{17}
$$

where $\lambda > 0$ is the temperature hyperparameter regulating distance sensitivity. Consequently, a smaller projection distance maps exponentially to a higher familiarity score, signaling stronger compatibility alignment.

Exploration-Augmented Query Routing. During coevolutionary training, query allocation must actively explore under-utilized agents to prevent policy starvation and encourage diversity, without compromising the actual competence required to solve the task. To balance these objectives, we make a strict distinction between agent prioritization and the subset activation threshold. To foster exploration, agents are ranked by an exploration-augmented routing score $s _ { i } ( \boldsymbol q )$

$$
s _ { i } ( q ) = f _ { i } ( q ) + c _ { \mathrm { u c b } } \sqrt { \frac { \log ( T ) } { n _ { i } + 1 } } + w _ { \mathrm { e n t } } \bar { H } _ { i } ,\tag{18}
$$

where $T$ denotes the total number of routed queries, $n _ { i }$ represents the historical training sample count allocated to agent $i ,$ and ${ \bar { H } } _ { i }$ is its running mean generation entropy. While the ranking order is dictated by $s _ { i } ( \boldsymbol q )$ , the activation criterion τ is evaluated strictly against the raw familiarity scores. Specifically, the router selects the minimal prefix subset $S _ { q }$ ordered by $s _ { i } ( \boldsymbol q )$ such that:

$$
\sum _ { i \in S _ { q } } f _ { i } ( q ) \geq \tau .\tag{19}
$$

If the entire population fails to reach $\tau ,$ all agents are activated. This critical design prevents exploration bonuses from inflating perceived competence coverage.

Agent Policy Optimization. Capability diferentiation is driven by each agent independently optimizing its reasoning policy using training samples dynamically allocated by the router. When an agent i accumulates a suficient microbatch $Q _ { i }$ in its LIFO bufer $B _ { i } .$ , it samples G distinct trajectory rollouts $\left\{ a _ { i , q } ^ { ( 1 ) } , \ldots , a _ { i , q } ^ { ( G ) } \right\}$ for each query $q \in Q _ { i }$ Let $R \left( q , a _ { i , q } ^ { ( g ) } \right)$ denote the scalar reward evaluated by the task-specific verifier.

To stabilize multi-sample reinforcement learning and mitigate reward scale variance across heterogeneous tasks, we employ Dynamic Sampling Policy Optimization (DAPO). Specifically, we compute the empirical mean $\mu _ { i , q }$ and standard deviation $\sigma _ { i , q }$ over the G sampled responses for query $q .$ The intra-agent normalized advantage $\hat { A } _ { i , q } ^ { ( g ) }$ for the g-th trajectory is formulated as:

$$
\hat { A } _ { i , q } ^ { ( g ) } = \frac { R \left( q , a _ { i , q } ^ { ( g ) } \right) - \mu _ { i , q } } { \sigma _ { i , q } + 1 0 ^ { - 4 } } .\tag{20}
$$

Integrating Group Sequence Policy Optimization (GSPO) importance sampling ratios efectively optimizes longhorizon chain-of-thought reasoning while mitigating tokenlevel variance. For a generated completion $\begin{array} { r l } { a _ { i , q } ^ { ( g ) } } & { { } = } \end{array}$ $( y _ { 1 } , . . . , y _ { S } )$ of length S, we define the sequence-level importance sampling ratio $\rho _ { i , q } ^ { ( g ) }$ averaged over all valid generation tokens:

$$
\rho _ { i , q } ^ { ( g ) } = \exp \left( \frac { 1 } { S } \sum _ { t = 1 } ^ { S } \log \frac { \pi _ { \theta _ { i } } \left( y _ { t } \mid q , y _ { < t } \right) } { \pi _ { \theta _ { i } ^ { \mathrm { o l d } } } \left( y _ { t } \mid q , y _ { < t } \right) } \right) .\tag{21}
$$

We then define the clipped surrogate objective $s _ { i , q } ^ { ( g ) }$ for the g-th completion as:

$$
s _ { i , q } ^ { ( g ) } = - \operatorname* { m i n } \left( \rho _ { i , q } ^ { ( g ) } \hat { A } _ { i , q } ^ { ( g ) } , \operatorname { c l i p } \left( \rho _ { i , q } ^ { ( g ) } , 1 - \epsilon , 1 + \epsilon \right) \hat { A } _ { i , q } ^ { ( g ) } \right) .\tag{22}
$$

where ϵ is the clipping threshold. To prevent policy degeneration and reward hacking, we apply a token-level Kullback–Leibler (KL) divergence penalty against a fixed reference policy $\pi _ { \mathrm { r e f } }$ . Defining the log-ratio at token step t as:

$$
x _ { t } = \log \pi _ { \mathrm { r e f } } ( y _ { t } \mid q , y _ { < t } ) - \log \pi _ { \theta _ { i } } ( y _ { t } \mid q , y _ { < t } ) ,\tag{23}
$$

the token-level KL penalty utilizes the estimator:

$$
D _ { \mathrm { K L } , t } = \exp ( x _ { t } ) - x _ { t } - 1 .\tag{24}
$$

Finally, the objective for agent i is normalized by the total number of active (non-padded) completion tokens across the accumulated micro-batch:

$$
\mathcal { L } _ { \mathrm { a g e n t } } ( \theta _ { i } ) = \frac { \sum _ { q \in Q _ { i } } \sum _ { g = 1 } ^ { G } \sum _ { t = 1 } ^ { S _ { g } } m _ { q , g , t } \left( s _ { i , q } ^ { ( g ) } + \beta D _ { \mathrm { K L } , t } \right) } { \sum _ { q \in Q _ { i } } \sum _ { g = 1 } ^ { G } \sum _ { t = 1 } ^ { S _ { g } } m _ { q , g , t } } .\tag{25}
$$

where $m _ { q , g , t } \in \{ 0 , 1 \}$ masks out padding tokens, and $\beta$ serves as the regularization scaling hyperparameter.

Router Optimization. As individual agent policies $\{ \pi _ { \boldsymbol { \theta } _ { i } } \} _ { i = 1 } ^ { N }$ evolve, their semantic competence boundaries shift. To maintain synchronization, the router is continually optimized using the feedback of the participating agents. Unlike the intra-agent advantage $\hat { A } _ { i , q } ^ { ( g ) }$ used for policy updates, the router evaluates a cross-agent relative advantage $A _ { i , q }$ that measures how agent i’s average performance compares against the collective consensus of the active subset $S _ { q } \colon$

$$
A _ { i , q } = \bar { r } _ { i , q } - \frac { 1 } { \left| S _ { q } \right| } \sum _ { j \in S _ { q } } \bar { r } _ { j , q } ,\tag{26}
$$

where $\begin{array} { r } { \bar { r } _ { i , q } = \frac { 1 } { G } \displaystyle \sum _ { g = 1 } ^ { G } R \left( q , a _ { i , q } ^ { ( g ) } \right) } \end{array}$ is the empirical mean reward of agent i. For solo-activated agents $( | S _ { q } | = 1 )$ , we set $A _ { i , q } = \bar { r } _ { i , q }$ to reward standalone mastery.

Geometrically, the router objective transforms these relative reward signals into a conditional push-and-pull optimization within the fixed reference space:

$$
\mathcal { L } _ { \mathrm { r o u t e r } } = \sum _ { i , q } \left\{ \begin{array} { l l } { A _ { i , q } d _ { i } ( q ) , } & { \mathrm { i f ~ } A _ { i , q } \geq 0 , } \\ { \vert A _ { i , q } \vert \operatorname* { m a x } ( 0 , m - d _ { i } ( q ) ) , } & { \mathrm { i f ~ } A _ { i , q } < 0 , } \end{array} \right.\tag{27}
$$

where $m > 0$ defines the separation margin. Minimizing Equation (27) pulls the predictor heads of outperforming agents $( A _ { i , q } \ge 0 )$ closer to their fixed anchors, thereby driving up their future familiarity scores for similar queries. Conversely, for underperforming agents $( A _ { i , q } ~ < ~ 0 )$ , the hinge mechanism pushes the predictor heads away from their anchors, but strictly bounds this repulsion to a maximum projection distance of m. This bounded separation is critical to prevent familiarity score collapse: without the margin m, unbounded push-away optimization on negative advantages would induce excessively large gradients, driving the projection distance to extremes and permanently collapsing the exponential familiarity score to zero. By capping the repulsion at distance $m ,$ our objective dynamically suppresses the activation probability for semantically similar queries while safeguarding the numerical and geometric stability of the router’s representation space. In practice, the value of m is set such that when the projection distances of all agents reach this boundary, their cumulative familiarity score falls just below the required activation threshold τ $( \mathrm { i . e . , ~ } \sum _ { i = 1 } ^ { N } \exp ( - \lambda m ) \le \tau )$ . This ensures that for universally unfamiliar or exceptionally dificult queries where no agent demonstrates competence, the cumulative threshold condition naturally fails, gracefully triggering the fallback strategy to activate the entire agent population.

## A.2 Inference and Consensus Aggregation

During test-time inference, all exploration dynamics are terminated $( c _ { \mathrm { u c b } } ~ = ~ w _ { \mathrm { e n t } } ~ = ~ 0 )$ . The router strictly applies the cumulative-threshold mechanism over learned familiarity scores $\{ f _ { i } ( q ) \} _ { i = 1 } ^ { N }$ , activating only the minimal subset of specialist agents required to satisfy τ. Each selected agent generates a single independent completion.

For final output synthesis, CERA-MoA deploys a domainaware aggregation protocol. For tasks with verifiable, deterministic solutions (e.g., mathematical reasoning or choice questions), the aggregator executes familiarity-weighted majority voting: candidate completions are grouped by their extracted final answers, and the winning consensus group is identified by summing the familiarity scores $f _ { i } ( q )$ of its contributing agents. The system then outputs the exact generation trajectory from the highest-familiarity agent within that winning group. For open-ended generation tasks where exact consensus is inapplicable, the aggregator directly outputs the response from the most familiar activated agent (arg $\operatorname* { m a x } _ { i \in S _ { q } } f _ { i } ( q ) )$ , maximizing domain alignment with minimal computational overhead.

## B Detailed Experimental Setup & Reproducibility

## B.1 Hardware and Software Infrastructure

All computational experiments are executed on a standardized high-performance computing cluster operating under the Ubuntu 22.04.5 LTS Linux distribution. The node is equipped with four NVIDIA H200 Tensor Core GPUs (each featuring 141 GB of HBM3e memory), dual Intel Xeon Platinum 8558 processors (yielding 96 physical CPU cores), and approximately 500 GB of host system memory. All neural network optimization and inference pipelines are implemented in Python under a CUDA-enabled PyTorch environment, utilizing mixed $\mathtt { b f l o a t 1 6 }$ precision to balance computational throughput and numerical stability. Scaling multi-

<table><tr><td>Library</td><td>Version / Role Description</td></tr><tr><td>Python</td><td>3.12</td></tr><tr><td>PyTorch</td><td>2.9.1</td></tr><tr><td>Transformers</td><td>4.57.6</td></tr><tr><td>PEFT</td><td>0.11.1</td></tr><tr><td>vLLM</td><td>0.16.0</td></tr><tr><td>Accelerate</td><td>1.13.0</td></tr><tr><td>Datasets</td><td>4.8.4</td></tr><tr><td>Safetensors</td><td>0.7.0</td></tr><tr><td>SymPy</td><td>1.14.0</td></tr><tr><td>NumPy</td><td>2.2.6</td></tr><tr><td>Pandas</td><td>2.2.2</td></tr></table>

Table 4: Major software libraries and dependencies imported by our training, reward evaluation, and verification frameworks. TRL components are sourced from our customized repository-local implementation to support groupnormalized DAPO and GSPO sequence-level importance sampling.

agent reinforcement learning within hardware memory budgets relies on implementing re-entrant gradient checkpointing during policy backward passes, substantially reducing activation memory consumption. Furthermore, we integrate memory-isolated, co-located vLLM engines to accelerate trajectory rollout generation during training. In the homogeneous setting, four independent LoRA adapter modules are co-hosted over a single shared LLM backbone, sharing fixed base model weights in GPU memory. In the heterogeneous setting, isolating each distinct base model-adapter pair within a dedicated worker process prevents CUDA context contention and memory fragmentation between the local training stack and the vLLM inference engine.

Regarding the training computational expenditure, advancing the co-evolutionary training by 1,000 optimization steps, which corresponds to processing 16,000 training queries and sampling $\bar { G } = 8$ trajectory rollouts per query, requires approximately one H200 GPU day.

Table 4 outlines the primary software libraries and framework dependencies required to execute our training, verification, and data-loading pipelines.

## B.2 Hyper-Parameter Configuration

Table 5 provides the comprehensive hyper-parameter schedule. Crucially, our co-evolutionary system decouples query throughput from fixed agent epoch cycles. In each orchestration round, the router samples a global query batch of size $\begin{array} { r } { B _ { \mathrm { g l o b a l } } = \frac { B _ { \mathrm { d e v i c e } } \times A } { G } = \frac { 1 6 \mathsf { \bar { \times } } 8 } { 8 } = \bar { 1 } 6 } \end{array}$ distinct prompts, where $B _ { \mathrm { d e v i c e } }$ denotes the per-device batch size, A is the gradient accumulation factor, and G represents the number of sampled completion trajectories per prompt. An individual agent policy $\theta _ { i }$ only executes a DAPO–GSPO optimization step when its local LIFO sample bufer $B _ { i }$ accumulates a complete micro-batch $( | Q _ { i } | = 1 6 )$ . Consequently, the training update frequency per agent is inherently adaptive: agents exhibiting higher familiarity and competence in specific semantic domains receive richer training allocation, naturally accelerating their capability specialization without requiring manual curriculum schedules.

## B.3 Prompt Design and Zero-Shot Task Instructions

We enforce a zero-shot, role-agnostic prompt design, rigorously evaluating whether CERA-MoA can induce genuine, data-driven domain specialization. Unlike conventional multi-agent orchestration baselines that rely heavily on prompt engineering—assigning explicit personas (e.g., "You are an expert mathematician...") or structured verification workflows—our system inputs contain only the raw user query. No overarching system prompts, role descriptors, or behavioral hints are injected into the context window. This design guarantees that the emergent capability diferentiation observed in our empirical analysis (Section 4.6) is strictly driven by our predictive familiarity routing and closed-loop reinforcement learning, rather than human-engineered role bias.

To prevent uncontrolled token inflation and maintain standardized evaluation across diverse base models, any proprietary or built-in reasoning modes $( \mathrm { e . g . }$ ., native internal "thinking" or hidden chain-of-thought routines) are disabled during generation; models rely entirely on standard autoregressive step-by-step reasoning. Prompts exceeding the configured input limit (800 tokens) are deterministically truncated from the left prior to routing and execution. Figure 4 presents the exact task-specific user formatting templates applied across all benchmark domains.

Hyper-Parameter Category Configured Value & Description   
Random seed 42   
Agent population size (N) $N = 4$ for shared-backbone settings; $N = 3$ for heterogeneous setting   
LoRA hyper-parameters Rank $r = 1 6 ,$ Alpha $\alpha = 3 2 ,$ Dropout probability 0.05   
LoRA target modules q\_proj, $\mathtt { k \_ p r o } \mathtt { j }$ , v\_proj, o\_proj, $\mathsf { g a t e \_ p r o j } .$ , up\_proj,   
down\_proj   
Learning rates Router predictor head learning rate: $1 \times 1 0 ^ { - 4 } ;$ ; Agent adapter learning rate: $3 \times 1 0 ^ { - 5 }$   
Agent Learning-rate schedule Cosine decay schedule with 600 warm-up steps; Minimum learning rate floor: $2 \times 1 0 ^ { - 5 }$   
Router routing parameters Temperature $\lambda = 2 . 0 .$ Threshold $\tau = 0 . { \dot { 7 } } .$ , Separation margin $m \stackrel { - } { = } 1 . 0$   
Exploration coeficients UCB bound $c _ { \mathrm { u c b } } = 2 . 0 $ ; Entropy weight $w _ { \mathrm { e n t } } = 2 . 0$ (shared) / 0.2 (heterogeneous)   
TRL optimization objective DAPO loss with group-normalized rewards and GSPO importance sampling ratios   
Trajectory rollout count (G) $G = 8$ independent completion rollouts per allocated training prompt   
Regularization KL divergence penalty weight $\beta = 1 \times \mathrm { \hat { 1 0 } ^ { - 4 } }$ ; Policy ratio clip margin $\epsilon = 0 . 2$   
Batching & accumulation Per-device batch size: 16 prompts; Gradient accumulation steps: $A = 8$   
Training duration & limits Maximum optimizer updates: 6,000 steps; Trainer steps per rollout generation: 2   
Optimizer specifics AdamW optimizer with $\beta _ { 1 } = 0 . 9 , \beta _ { 2 } = 0 . 9 5 ;$ Agent weight decay: $\mathrm { \bar { 1 } \times 1 0 ^ { - 3 } }$   
Gradient clipping Router predictor head gradient norm clipped at 1.0; Agent gradient norm clipped at 1.0   
Generation parameters Sampling temperature: 1.0; Top-p (nucleus): 0.95; Top-k: 50; Repetition penalty: 1.0   
Sequence length limits Maximum input prompt: 800 tokens; Maximum generated completion: 1,000 tokens  
Table 5: Primary hyper-parameter configurations for CERA-MoA training and inference.

## C Task-Specific Reward Formulation

Each rollout receives a task reward $r _ { \mathrm { b a s e } } \in [ 0 , 1 ]$ followed by a length penalty. Let ℓ denote the number of generated tokens, $\breve { L } _ { \mathrm { m a x } }$ the maximum completion length, $\gamma \in ( 0 , 1 )$ the fraction of $L _ { \mathrm { m a x } }$ at which penalization begins, and $\eta \geq 0$ the maximum penalty magnitude. The final reward is

$$
R = r _ { \mathrm { b a s e } } - \eta \cdot \sqrt { \frac { \ell - \gamma L _ { \operatorname* { m a x } } } { ( 1 - \gamma ) L _ { \operatorname* { m a x } } } } , \quad \gamma L _ { \operatorname* { m a x } } < \ell < L _ { \operatorname* { m a x } } ,\tag{28}
$$

Thus, the penalty is zero up to $\gamma L _ { \mathrm { m a x } }$ and grows linearly thereafter, reaching η at the completion limit. Accuracy reported at evaluation is only computed from whether $r _ { \mathrm { b a s e } } = 1$

Mathematical reasoning. For math problems, the evaluator searches for all occurrences of \boxed{} and returns the last complete balanced-brace expression, which avoids treating an intermediate derivation as the final answer. It then normalizes common presentation variants before symbolic comparison: surrounding delimiters and whitespace, units, thousands separators, reducible fractions, percentage markers, equivalent matrix delimiters, and multiple-choice notation. The verifier additionally handles set-style answers, ± expansions, ratios, and mathematical expressions represented in L<sup>A</sup>T X. It compares the normalized prediction and reference using exact checks followed by SymPy simplification and equivalent algebraic transformations; the process is isolated with a timeout to prevent malformed expressions from blocking training. The binary reward is

$$
r _ { \mathrm { b a s e } } ^ { \mathrm { m a t h } } = \mathbb { I } \{ \mathrm { M a t h E q u a l } ( \mathrm { E x t r a c t } ( a ) , y ^ { \star } ) \} .
$$

This procedure accepts mathematically equivalent expressions while retaining exact matching when symbolic parsing is unavailable.

Code generation. The evaluator extracts the last fenced Python block when present, otherwise the raw completion. Code is executed in an isolated subprocess under a reliability guard and a 10-second wall-clock limit covering compilation and all tests. Both stdin/stdout programs and function-call tasks are supported. With the enabled stepwise reward, the score is the pass fraction over the task’s tests:

$$
r _ { \mathrm { b a s e } } ^ { \mathrm { c o d e } } = \frac { 1 } { | \mathcal { T } | } \sum _ { ( x , y ) \in \mathcal { T } } \mathbb { I } \{ \mathrm { R u n } ( a , x ) = y \} .
$$

Compilation failures, runtime failures, timeouts, malformed outputs, and missing code receive zero reward. The implementation can instead use strict all-tests-pass reward.

Instruction following. Each IF item contains an instruction identifier list and its instantiated argument dictionaries. The oficial IFEval (Zhou et al. 2023) instruction registry reconstructs each verifier. With stepwise instruction reward enabled, the reward is the fraction of satisfied constraints:

$$
r _ { \mathrm { b a s e } } ^ { \mathrm { I F } } = \frac { 1 } { K } \sum _ { k = 1 } ^ { K } \mathbb { I } \{ { \mathrm { C h e c k } } _ { k } ( a ; q , \omega _ { k } ) = 1 \} .
$$

Examples include exact endings, required keywords, case restrictions, sentence or bullet counts, and markup constraints. If a registry entry is unavailable or the response is empty, the reward is zero. The strict alternative awards one only if every constraint is satisfied.

General reasoning. General reasoning items use the same final-answer extractor as math problems, followed by taskagnostic normalization: optional “Answer:” prefixes and enclosing parentheses are removed, one-letter choices are uppercased, and remaining whitespace is canonicalized and lowercased. The reward is exact match after normalization:

$$
r _ { \mathrm { b a s e } } ^ { \mathrm { g e n e r a l } } = \mathbb { I } \{ \mathrm { N o r m a l i z e } ( \operatorname { E x t r a c t } ( a ) ) = \mathrm { N o r m a l i z e } ( y ^ { \star } ) \} .
$$

![](images/955acc014b4eaf85abbbb99f01943eed8b9d5480128ddd0866e95a9711c7fff0.jpg)  
Figure 4: Zero-shot prompt templates used for diferent task families. The two code-generation blocks correspond to function-call and standard-input/output task interfaces, respectively. Variables {question} and {fn\_name} are instantiated from each dataset item. Instruction-following prompts are passed through unchanged so that their verifier-defined constraints are preserved.

## D Dataset Specifications

## D.1 Source Datasets

GSM8K. GSM8K (Cobbe et al. 2021) contains 7,473 training and 1,319 test grade-school mathematical word problems. Each item requires multi-step arithmetic reasoning from a natural-language question to a short numerical answer.

MATH. MATH (Hendrycks et al. 2021) has 7,500 training and 5,000 test competition-style problems spanning algebra, geometry, number theory, counting, and probability. Reference answers are represented in mathematical notation and require equivalence-aware rather than string-based verification.

DAPO-MATH-17K. DAPO-MATH-17K (Yu et al. 2026) is a verifiable mathematical reasoning collection. The processed version contains 13,616 training items and a 500-item test split, covering a broad range of symbolic and quantitative problem types with automatically checkable answers.

MBPP. The sanitized MBPP split (Austin et al. 2021) used in our pipeline contains 163 training and 257 test Python programming tasks. Problems specify a function-level intent and executable input–output examples, supporting direct functional correctness checks.

Eurus-2-Code. The processed Eurus-2-Code split (Yuan et al. 2024) contains 25,278 training and 1,024 test codegeneration tasks. Each task is associated with executable input–output tests, extending the code domain beyond short function synthesis.

TACO. TACO (Li et al. 2023) contains 24,702 training and 1,000 test programming problems. In addition to functioncall tasks, it includes standard-input/standard-output problems that evaluate complete programs against executable test cases.

MAGPIE-IF. The IFEval-like instruction-following source derived from MAGPIE (Xu et al. 2025) provides 551,465 training prompts and a 500-item test split. Its prompts encode constraints such as required content, output structure, length, casing, and ordering in a form that can be programmatically verified.

RLVR-IFEval. RLVR-IFEval (Lambert et al. 2024) is a rule-verifiable instruction-following source based on the

IFEval protocol, with 14,540 training and 500 test examples. Each instance associates a natural-language request with the instruction identifiers and arguments needed to reconstruct its verifiers.

BIG-bench Hard. BIG-bench Hard (BBH) (Suzgun et al. 2023) contains 6,011 training and 500 test instances in the processed split. It covers challenging general reasoning, including symbolic manipulation, logical deduction, temporal and spatial reasoning, language understanding, and multiplechoice tasks.

IFEval. IFEval (Zhou et al. 2023) is an instructionfollowing benchmark with 541 evaluation prompts and no training split. It measures compliance with individually and jointly specified constraints using deterministic verifiers rather than model-based judging.

HumanEval. The processed HumanEval benchmark (Chen et al. 2021) contains 154 test-only Python programming problems. Each problem specifies a function signature, a natural-language docstring, and unit tests designed to assess functional correctness.

AGIEval. The AGIEval evaluation file (Zhong et al. 2023) contains 1,000 test-only multiple-choice questions. It assesses general knowledge and reasoning across standardized examinations and does not contribute training instances to our method.

ARC-Challenge. ARC-Challenge (Clark et al. 2018) contains 1,172 test-only science questions in the processed evaluation split. Questions are multiple choice and target gradeschool scientific reasoning that typically requires combining facts or applying basic principles.

LogicBench. The LogicBench evaluation file (Parmar et al. 2024) contains 1,520 test-only instances. It focuses on formal and natural-language logical reasoning, including entailment, deduction, and structured constraint solving.

OlympiadBench. The OlympiadBench evaluation file (He et al. 2024) contains 707 test-only problems. It evaluates high-dificulty mathematical and scientific reasoning with questions drawn from olympiad-style settings.

## D.2 Construction of Training and In-Distribution Test Sets

The resulting training corpus contains 32,000 examples across four task families, deterministically sampled using a fixed random seed (seed = 42) to ensure full reproducibility: 11,200 mathematical reasoning examples (MATH: 5,000, DAPO-MATH-17k: 5,000, and GSM8K: 1,200), 9,600 code-generation examples (MBPP: 163, Eurus-2- Code: 7,000, and TACO: 2,437), 6,400 instruction-following examples (MAGPIE-IF: 1,400 and RLVR-IFEval: 5,000), and 4,800 general-reasoning examples (BBH: 4,800). All sources are converted into a unified JSON schema containing problem, class, and task-specific verifier metadata. Mathematical and general-reasoning instances provide true\_answer; CODE instances provide input–output test\_cases and, where applicable, fn\_name; IF instances provide instruction\_id\_list and the corresponding verifier arguments.

For final in-distribution (ID) evaluation, we use the heldout test splits of the nine source datasets used for training: GSM8K, MATH, and DAPO-MATH-17k for mathematical reasoning; MBPP, Eurus-2-Code, and TACO for code generation; MAGPIE-IF and RLVR-IFEval for instruction following; and BIG-bench Hard for general reasoning. These test splits are evaluated separately, and their per-dataset results constitute the ID results in our experiments.

## D.3 Out-of-Distribution Evaluation

We evaluate out-of-distribution (OOD) generalization on six benchmarks that are excluded from training: IFEval for instruction following, HumanEval for code generation, and AGIEval, ARC-Challenge, LogicBench, and Olympiad-Bench for general reasoning and mathematical/scientific reasoning. The OOD protocol retains the trained router, agent adapters, task-appropriate prompt construction, and answer aggregation rule, thereby testing transfer of the learned routing and specialization rather than adaptation to the held-out benchmarks.

## E Additional Experiments and Analysis

## E.1 Router Computational Overhead Analysis

A critical requirement for dynamic multi-agent orchestration is that the routing mechanism itself must not become a computational bottleneck. We quantify the eficiency of our predictive familiarity estimator by profiling the average latency per query during test-time inference. All measurements are executed on a single NVIDIA H200 GPU. As detailed in Table 6, we compare the router’s execution time against the time required for a single agent to generate a complete trajectory, both with and without the vLLM acceleration engine.

<table><tr><td>Execution Component</td><td>Batch Size</td><td>Avg. Latency (ms)</td></tr><tr><td>Predictive Router</td><td>1</td><td>11.87</td></tr><tr><td>Completion Generation (Native)</td><td>1</td><td>1350.32</td></tr><tr><td>Completion Generation (vLLM)</td><td>32</td><td>154.09</td></tr></table>

Table 6: Computational overhead profiling on a single NVIDIA H200 GPU.

The empirical results demonstrate that the CERA-MoA router imposes a negligible computational footprint, averaging merely 11.87 ms per query. This eficiency arises fundamentally from the architectural design: evaluating familiarity requires extracting intermediate hidden states via a single, partial forward pass. Unlike standard sequenceto-sequence evaluation mechanisms or external LLM-asa-judge verifiers, our router entirely bypasses the recursive, token-by-token autoregressive decoding phase. Consequently, the routing overhead is less than 1% of the native generation time, ensuring that the system’s eficiency gains from activating fewer agents are preserved in end-toend wall-clock latency.

## E.2 Co-Evolutionary Learning Dynamics

Tracing the internal training dynamics across the evolutionary timeline illuminates how CERA-MoA induces capability specialization without human intervention. Figures 5, 6, and 7 illustrate the progression of system-level rewards, individual agent update frequencies, and the corresponding familiarity scores, respectively.

![](images/e5fea0651548fb1b1323cdff0098c96613fd3caa72db47ec0a21f57f5357d536.jpg)  
Figure 5: System average reward progression.

![](images/f1df0fa6ad08fe0ef0847ff01dde767a5c422fee1bf23000915d6556f48f6845.jpg)  
Figure 6: Cumulative updates per agent.

![](images/e59a47d179e4c91a58ec18984301ccd4f27772dfcb411f965e08050c6a314d29.jpg)  
Figure 7: Familiarity score progression.

As shown in Figure 5, the aggregate system reward steadily increases and converges stably, validating the efectiveness of our DAPO–GSPO optimization protocol. Furthermore, Figures 6 and 7 reveal a clear hierarchical diferentiation in agent workloads. Agent 2 rapidly emerges as the primary reasoning engine, executing the highest total volume of updates. Agent 3 specializes strictly in code-generation topologies, maintaining the second-highest update frequency as it digests programming queries. Meanwhile, Agents 1 and 4 secure their niches by supplementing the population on instruction-following, general reasoning, and concise mathematical problems. Importantly, while their overall update frequencies are lower than the primary solvers, their familiarity scores still rise steadily within their specialized semantic clusters. This confirms that our exploration-augmented routing mechanism successfully prevents policy starvation, ensuring all agents continuously evolve and actively contribute to the overarching mixture.

E.3 Hyperparameter Robustness and Sensitivities Cumulative Threshold τ. The cumulative threshold τ serves as the primary control lever in CERA-MoA, balancing reasoning accuracy against computational cost. Using the default population trained under threshold 0.7, we conduct a test-time sweep from $\tau = 0 . 5$ to 0.9 (Table 7).
<table><tr><td>Test Threshold (τ)</td><td>ID Avg.</td><td>OOD Avg.</td><td>Avg. Tokens</td></tr><tr><td>0.5</td><td>62.9</td><td>71.9</td><td>329.06</td></tr><tr><td>0.6</td><td>63.0</td><td>72.5</td><td>342.22</td></tr><tr><td>0.7 (Default)</td><td>63.2</td><td>72.8</td><td>367.77</td></tr><tr><td>0.8</td><td>63.0</td><td>72.9</td><td>422.12</td></tr><tr><td>0.9</td><td>63.2</td><td>72.7</td><td>648.01</td></tr></table>

Table 7: Test-time sensitivity analysis for the default model (trained with $\tau = 0 . 7 )$ .

Lowering the test threshold aggressively curbs token expenditure but causes a drop in accuracy, as complex queries fail to trigger suficient multi-agent consensus. Conversely, elevating it to 0.8 or 0.9 triggers massive computational redundancy with little performance gain. Investigating training-time sensitivity involves re-training a population using a stricter threshold of $\tau = 0 . 8$ . Testing this population across varying thresholds (Table 8) reveals a substantial performance drop (peak ID: 61.2, OOD: 71.5). A higher training threshold forces over-activation, heavily diluting the targeted specialization feedback and preventing the emergence of sharp domain experts.

<table><tr><td>Test Threshold (τ)</td><td>ID Avg.</td><td>OOD Avg.</td><td>Avg. Tokens</td></tr><tr><td>0.5</td><td>60.9</td><td>70.9</td><td>353.84</td></tr><tr><td>0.6</td><td>61.2</td><td>71.5</td><td>371.32</td></tr><tr><td>0.7</td><td>60.8</td><td>71.4</td><td>405.36</td></tr><tr><td>0.8 (Train Setting)</td><td>61.2</td><td>71.4</td><td>485.88</td></tr><tr><td>0.9</td><td>61.2</td><td>71.5</td><td>676.93</td></tr></table>

Table 8: Evaluation of an alternative model trained with a sub-optimal threshold $( \tau = 0 . 8 )$

Familiarity Temperature λ. Examining the sensitivity of the exponential decay parameter λ reveals that decreasing the temperature to $\lambda = 1 . 0$ significantly degrades performance (ID: 55.3, OOD: 59.8, Avg. Tokens: 463.72) compared to the default λ = 2.0 (ID: 63.2, OOD: 72.8, Avg. Tokens: 367.77). As illustrated in Figure 8, tracking the average familiarity indicates that the scores under $\lambda = 1 . 0$ artificially inflate and reach the cumulative activation threshold $( \tau = 0 . 7 )$ far too early in the training process. Meeting the confidence bar so easily causes the router to prematurely halt the subset expansion, severely limiting the exploration-augmented allocation mechanism. This lack of early exploration prevents under-utilized agents from receiving suficient training queries across diverse semantic domains, ultimately hindering the emergence of well-diferentiated specialists.

Separation Margin m. Validating the necessity of the bounded repulsion margin m involves testing an increased boundary $( m ~ = ~ 2 . 0 )$ alongside a completely unbounded variant. Increasing the margin to $m \ = \ 2 . 0$ degrades the overall reasoning accuracy (ID: 59.2, OOD: 70.7, Avg. Tokens: 364.74). As depicted in Figure 9, an overly aggressive separation margin prevents the familiarity score of at least one agent from growing efectively. More critically, removing the margin (applying $A _ { i , q } d _ { i } ( q )$ for unbounded repulsion based on negative advantages) results in catastrophic instability. Figure 10 demonstrates that the familiarity scores of all agents fluctuate violently and fail to grow altogether under this unbounded configuration. These observations directly corroborate our geometrical derivation: without the protective margin m, unbounded negative gradients induce large shifts in the projection space, collapsing the familiarity representation.

![](images/465ad476e6a69697db307a28df64de42a379f459d51d00e7c54dfece0f0896e8.jpg)  
Figure 8: Familiarity score progression under $\lambda = 1 . 0 .$

![](images/395729d1bcd00f611736e71b38e9029a8e326ac2666c26e8f49a6a7329694a1c.jpg)  
Figure 9: Familiarity score progression under an increased margin (m = 2.0).

![](images/82e0714b40d7d27f497262a9b74acc23462d482d6c50dd19faa573d1ea492beb.jpg)  
Figure 10: Familiarity score progression without a protective separation margin.

## E.4 Ablation on Routing and Exploration Mechanisms

A core premise of CERA-MoA is that dynamically allocating data based on actual competence coverage is superior to fixed-size routing, and that active exploration is vital for avoiding policy starvation. We justify these architectural choices by ablating the adaptive threshold routing and the exploration bonuses (UCB and historical entropy) during training. The quantitative results of these ablations are summarized in Table 9.

<table><tr><td>Training Configuration</td><td> $\mathbf { I D \ A v g . }$ </td><td>OOD Avg.</td><td>Avg. Tokens</td></tr><tr><td>Fixed Top-3 Routing</td><td>63.1</td><td>72.3</td><td>817.97</td></tr><tr><td>No Exploration (w/o UCB &amp; Entropy)</td><td>57.1</td><td>63.9</td><td>436.60</td></tr><tr><td>Adaptive Threshold (Ours)</td><td>63.2</td><td>72.8</td><td>367.77</td></tr></table>

Table 9: Ablation of routing allocation and exploration mechanisms.

Adaptive Threshold vs. Fixed Top-K Training. Training an alternative population using a fixed Top-3 routing strategy (while retaining the threshold logic for test-time inference) yields comparable accuracy but severely inflates the token cost to 817.97 tokens per query. Tracing the internal metrics in Figure 11 explains this ineficiency: forcing the allocation of exactly three agents per query, regardless of the intrinsic dificulty of the prompt, causes the familiarity scores to oscillate violently rather than grow steadily. This instability prevents the system from accurately calibrating its confidence, ultimately failing to route queries eficiently during inference.

![](images/7d7e7daf457b383e761a2615c8562faf235c1574e676f6b1e5529b03fe858f18.jpg)  
Figure 11: Familiarity score progression under fixed Top-3 training.

Necessity of Exploration Mechanisms. Ablating the UCB exploration bonus and historical entropy regularization during training severely degrades system performance across all metrics (Table 9). Visualizing the routed hidden states via t-SNE (Figure 12) confirms that without active exploration, the router sufers from catastrophic mode collapse. Specifically, the vast majority of queries are blindly routed to a single dominant agent (Agent 3), leaving Agents 1 and 2 in complete policy starvation with minimal participation. This failure underscores the absolute necessity of forced exploration for cultivating a balanced, well-diferentiated agent population.

## E.5 Ablation on Hidden State Extraction

A cornerstone of our familiarity estimator is the reliance on mid-layer hidden states rather than the final-layer representations. We empirically justify this architectural design by ablating the feature extraction, forcing the router to predict familiarity strictly from the model’s final-layer outputs.

![](images/ce565e0b079a5aef29e1c25b894dc032425c9ff4d4ef58cdac3f53fa1cc9af08.jpg)

Figure 12: t-SNE visualization of routed hidden states without exploration mechanisms.
<table><tr><td>Feature Source</td><td>ID Avg.</td><td>OOD Avg.</td><td>Avg. Tokens</td></tr><tr><td>Final-layer States</td><td>61.1</td><td>71.3</td><td>438.12</td></tr><tr><td>Mid-layer States (Ours)</td><td>63.2</td><td>72.8</td><td>367.77</td></tr></table>

Table 10: Ablation of hidden states extraction depth.

As presented in Table 10, the final-layer variant sufers a notable degradation in both ID and OOD accuracy, accompanied by an unjustified surge in token generation. Figures 13, 14, and 15 visualize the internal routing metrics under the final-layer configuration, diagnosing this failure mode.

![](images/7c747900d67da3bb26f98e668a3d39bb54f12ee1b7c84cb145400daff0cceca3.jpg)  
Figure 13: Final-layer familiarity scores progression.

Figure 13 demonstrates that final-layer familiarity scores grow slower than their mid-layer counterparts. Consequently, meeting the identical threshold (τ = 0.7) during training forces the router to consistently activate a larger subset of agents (Figure 14), thereby diluting the targeted specialization feedback. Furthermore, the t-SNE visualization (Figure 15) reveals that final-layer representations exhibit entanglement, failing to form cleanly separable clusters for distinct agent expertise. We attribute this degradation to the phenomenon investigated by (Skean et al. 2025): while mid-layer hidden states preserve broad, structurally rich semantics, final-layer representations are heavily dominated by immediate, token-specific predictive distributions (autoregressive interference). By extracting features prior to this late-stage output bottleneck, our mid-layer router successfully preserves the representation separability required to map complex queries to the correct domain experts.

![](images/df08d72f700b02796af7e2e727edcc20950c40520612bd4cdcac58ab291c1e03.jpg)  
Figure 14: Average training agents activated per query.

![](images/706d352b275c590b49148986ed08fbc51a557ed4b7df8be2725e3351ddfcf0d8.jpg)  
Figure 15: t-SNE visualization of final-layer features.

## F Limitations

While CERA-MoA demonstrates strong adaptability and efficiency across various reasoning benchmarks, our current framework possesses certain analytical limitations.

First, the predictive familiarity estimator currently evaluates the semantic competence of agents based on the initial user prompt. Consequently, the routing mechanism is inherently optimized for single-turn interactions or fixedtrajectory generation. In long-horizon, multi-turn agentic workflows (e.g., iterative software development or extended multi-agent debate), the required expertise may shift dynamically as the conversational context grows. However, adapting CERA-MoA to these scenarios is structurally straightforward: it simply entails extending the familiarity evaluation to extract mid-layer hidden states from the complete, updated context window at each interaction step, rather than just the initial instruction. Exploring this continual, step-wise competence tracking remains an actionable and important avenue for future research.

Second, our current consensus aggregation protocol limits the collaborative potential of the agent population on open-ended generation tasks. While CERA-MoA successfully leverages familiarity-weighted majority voting for deterministic reasoning (e.g., mathematics), it defaults to outputting the single response from the most familiar agent for open-ended benchmarks, such as code synthesis in MBPP and HumanEval. Although this design choice reduces computational overhead, it inherently bypasses the advantages of multi-agent generation. In these generative scenarios, synthesizing diverse candidate solutions from multiple activated experts could potentially yield a more robust and optimal final output than relying on a single specialist. To address this, a promising extension is the integration of a generative aggregator, such as an LLM-based meta-thinker, to dynamically synthesize the completions ofthe selected expert subset. Importantly, this advanced answer aggregation approach is architecturally fully compatible with our current design.