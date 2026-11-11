# Broad-Persistent Advising in Interactive Reinforcement Learning via Latent State Encoding

Zhuoran Liang<sup>1</sup> Francisco Cruz<sup>1,2</sup>

<sup>1</sup>School of Computer Science and Engineering, UNSW Sydney, Australia <sup>2</sup>Escuela de Ingenier´ıa, Universidad Central de Chile, Santiago, Chile jackliangzhuoran@gmail.com, f.cruz@unsw.edu.au

## Abstract

Deep Interactive Reinforcement Learning (Deep IntRL) accelerates reinforcement learning training by incorporating external advice. However, conventional Deep IntRL uses each piece of advice only in the current step and discards it after a single interaction. Persistent Advising and Broad-Persistent Advising (BPA) extend this process by storing advice in persistent memory and reusing it in similar states through state generalisation based on K-means clustering. This research proposes Encoder BPA, which employs a latent encoder to retrieve and reuse advice. In this framework, a trained encoder is used to map a state sequence into a latent representation at every time step. Advice can be retrieved through latent comparison, together with a top-(k) aggregation and a consistency condition to improve reuse stability. Experimental results show that Encoder BPA improves learning eficiency and advice utilisation at an early stage of training compared with diferent Deep IntRL approaches, especially in complex and partially observable environments.

## 1 Introduction

The development of artificial intelligence has afected many industries, and technologies such as robotics and autonomous systems have become increasingly important. As a branch of AI techniques, reinforcement learning has attracted growing attention because it lets an agent learn, become familiar with, and work in an environment through interaction.

Reinforcement Learning (RL) describes a learning process in which an agent improves its decision-making through repeated interaction with an environment [Sutton and Barto, 2018; Plaat, 2022]. The agent can be understood as an autonomous system, such as a robot, a robotic arm, or a self-driving vehicle, while the environment refers to the surrounding space. As reinforcement learning has been applied to increasingly complex tasks, interactive reinforcement learning has emerged as an approach that incorporates external guidance into the learning process [Cruz and Igarashi, 2021; Millan et al., 2019]. This framework can be used in human-in-theloop reinforcement learning tasks where humans can intervene in agent decision-making. Building on interactive reinforcement learning, the Persistent Advising and Broad-Persistent Advising (BPA) approach uses persistent memory to store and reuse advice from the external trainer. Rather than treating each piece of advice as a one-time interaction, BPA allows the agent to retain useful guidance and apply it again when it encounters the same or similar situations [Nguyen et al., 2023; Bignold et al., 2023a].

This research is motivated by the limitations of the BPA approach. In the original BPA approach, states are mapped into a small number of clusters using Kmeans clustering and previous advice can be used when the agent encounters another state assigned to the same cluster [Nguyen et al., 2023]. Although this provides eficient state generalisation, it may be coarse for high dimensional robotic environments. For example, in the Domestic Robot Environment experiment of the original BPA study, each state is represented by a 64 × 64 RGB image, corresponding to 64 × 64 × 3 input dimensions. At the same time, the elbow method selects only four clusters in total for the K-means clustering [Nguyen et al., 2023]. A limited number of clusters may force dissimilar states into the same cluster. As a result, the agent may reuse the same advice for behaviourally different states. In addition, the K-means clustering may be afected by the curse of dimensionality in a highdimensional state space because of its distance measurement [Murphy, 2022].

To address these issues, this research proposes replacing the clustering-based state generalisation in the original BPA approach with a representation learning mechanism in which a neural network encoder learns latent representations from state histories and reuses previous advice via latent space advice retrieval. Based on the research gap identified above, this study examines whether replacing K-means clustering in the original BPA approach with a representation learning mechanism and latent space advice retrieval can improve advice reuse and training performance. The new framework, including representation learning and latent space advice retrieval, is called Encoder BPA.

This research hypothesises that the representation learning mechanism will capture more meaningful behavioural features than raw state clustering, and latent space advice retrieval will identify more relevant previous advice. As a result, the agent may reuse external advice more accurately and achieve improved training performance compared with the original K-means clustering BPA approach.

This research is expected to make several contributions. First, it proposes the Encoder BPA approach, providing an alternative advice reuse mechanism for persistent advising. Second, this research contributes by implementing persistent advising and BPA in continuous action space environments. Third, this research presents an experimental study of the proposed BPA encoder approach in partially observable reinforcement learning environments.

## 2 Related Work

## 2.1 Interactive Reinforcement Learning

Interactive Reinforcement Learning (IntRL) extends the traditional RL framework by allowing an external trainer to provide advice during the learning process [Cruz and Igarashi, 2021]. The trainer may be a human expert or another agent with prior knowledge of the task, which can guide the learner when assistance is available [Najar and Chetouani, 2021; Bignold et al., 2021]. The diference between traditional RL and IntRL is that in IntRL, advice can influence action selection, guide exploration, and may consequently accelerate learning.

Deep Interactive Reinforcement Learning (Deep IntRL) emerged from the integration of IntRL with Deep RL, allowing agents to learn from both environmental rewards and external feedback in high-dimensional or continuous state spaces [Moreira et al., 2020; Nguyen et al., 2023]. However, most existing Deep IntRL methods remain limited in that feedback is typically given only for the current state and is discarded after a single use. To address these limitations, recent Deep IntRL research has focused on making advice more reusable rather than treating it as one-of feedback [Bignold et al., 2023b; Nguyen et al., 2023].

## 2.2 Persistent Advising

Persistent advising extends Deep IntRL by allowing the agent to retain advice from a trainer and reuse it in later decision-making [Bignold et al., 2023b]. This improves the value of each interaction and reduces the need for the trainer to provide the same guidance repeatedly.

Persistent advising includes the Probabilistic Policy Reuse (PPR) mechanism, in which the agent gradually reduces its dependence on previous advice and increases its trust in newly acquired information. In this framework, the agent can learn from the newer trainer guidance through the training process [Bignold et al., 2023b].

State-based advice is a form of persistent advising in which the agent directly stores (state, action) pairs in a persistent memory after the trainer provides advice. When the same state is encountered again, the agent can reuse the stored advice instead of requesting new input [Bignold et al., 2023b]. This method is efective in discrete environments, where exact states are more likely to reappear. However, its main limitation is that states are not efectively generalised, which makes the method less suitable for large or continuous state spaces.

![](images/3b4044591dd1969ce97a9eb3b86fdf1d74322ba7086001965dd30988f35f18ab.jpg)  
Figure 1: The PPR mechanism. When external advice is available, the learning agent applies the advised action and stores the (state, action) pair in the persistent memory. When no external advice is available, the associated action may be reused if a matching state is found in the persistent memory.

Figure 1 illustrates how the state-based persistent memory is used in the learning process of Deep IntRL. In each time step, if there is external advice, the learning agent applies the advised action and stores the (state, action) pair in the persistent memory. When no external advice is available, the associated action may be reused if a matching state is found in the persistent memory.

## 2.3 Broad-Persistent Advising

Broad-Persistent Advising (BPA) builds on persistent advising by enabling advice to be generalised and reused across similar states during later decisionmaking [Nguyen et al., 2023]. This approach has two core components: persistent advising, which preserves advice for later reuse, and broad advising, which extends advice reuse from a repeated state to similar states.

In broad advising, BPA uses a clustering model to map a raw state to a cluster ID and stores advice as a (cluster id, action) pair rather than a (state, action) pair [Nguyen et al., 2023; Murphy, 2022]. This allows advice given in one state to be reused in other states that share similar features. As a result, broad advising increases the utilisation of persistent memory, making it more suitable for large and continuous state spaces.

In the original BPA approach, K-means clustering is used for generalisation with the elbow method, which helps determine the value of K. In Cart-Pole, the state consists of four continuous values. In contrast, the Domestic Robot environment uses a $6 4 \times 6 4 \times 3$ state representation, resulting in a high-dimensional input space, while the elbow method selects only four clusters. In this research, we develop a framework that uses a neural network encoder to replace the K-means algorithm in broad advising, where the learned latent representation is used for advice comparison [Nguyen et al., 2023].

## 2.4 Representation Learning

Representation learning allows a model to extract useful features from raw data and express them in more compact feature representations [Bengio et al., 2012]. Prior research generally suggests that efective representations can extract information from complex inputs and map them into a compact feature space that is better suited for downstream tasks, thereby improving learning eficiency [Srinivas et al., 2020; Stooke et al., 2021; Yarats et al., 2021]. In reinforcement learning, representation learning can be used to address the limitations of raw state observations. In formal terms, given an input $x \in \mathcal { X }$ , a representation model learns a mapping $f _ { \theta } : \mathcal { X }  \mathcal { Z }$ , where the latent representation is defined as $z = f _ { \boldsymbol { \theta } } ( x )$ [Echchahed and Castro, 2025]. A common way to compare two learned representations is cosine similarity. Given two sequence representations $z _ { i }$ and $z _ { j }$ , it is defined as:

$$
\cos ( z _ { i } , z _ { j } ) = { \frac { z _ { i } ^ { \top } z _ { j } } { \| z _ { i } \| \left\| z _ { j } \right\| } } .\tag{1}
$$

Cosine similarity compares the direction of two fixed dimensional embedding vectors rather than their magnitudes. This makes it suitable for measuring similarity of encoded state histories [Chen et al., 2020].

In this research, representation learning is used to replace the clustering-based state abstraction adopted in the original BPA. The proposed framework trains a latent encoder using state transition data. A persistent memory cell is then stored as a pair of latent and action $( z _ { i } ,$ , action), where $z _ { i }$ represents the encoded state history. During advice retrieval, the current latent representation can be compared with stored representations using cosine similarity.

## 3 Methodology

The BPA method enables advice to be reused across similar states. However, the broad advising component relies on K-means clustering, which may be limited in highdimensional environments. In the methodology section, we describe the Encoder BPA framework, which may enhance the original BPA approach by using representation learning and latent space advice retrieval.

## 3.1 Encoder Pre-training

In representation learning, we use a supervised learning method to train an encoder network to capture actiondiscriminative features. In this process, a teacher network continuously interacts with the environment to collect transition data. The collected data are then used to train an encoder together with a prediction head. At each time step t, the input to the encoder is a sequence of the most recent L states, where the state-history length is fixed at $L = 1 6$ across all experimental environments:

$$
h _ { t } = ( s _ { t - L + 1 } , s _ { t - L + 2 } , . ~ . ~ . , s _ { t } ) ,\tag{2}
$$

and the corresponding training target is the action $a _ { t } .$ The encoder $f _ { \theta }$ maps the input sequence $h _ { t }$ to a latent representation $z _ { t } .$ , defined as:

$$
z _ { t } = f _ { \theta } ( h _ { t } ) .\tag{3}
$$

A prediction head, implemented as a multilayer perceptron (MLP), takes $z _ { t }$ as input and predicts the action $\hat { a } _ { t } .$ . The encoder and prediction head are jointly trained using cross-entropy loss for discrete actions or mean squared error loss for continuous actions.

After the encoder pre-training stage, the prediction head is discarded, and the encoder parameters are frozen. The pre-trained encoder is then used to generate latent representations for downstream use in the advising framework.

## 3.2 Encoder BPA

The main diferences between the original BPA approach and the Encoder BPA approach are in the broad advising part. After the pre-training, the Encoder BPA approach maintains a fixed-length queue containing the most recent L states, $s _ { t - L + 1 } , \ldots , s _ { t }$ at each time step and uses the pre-trained encoder to obtain a latent representation $z _ { i }$ from the state history.

In the proposed framework, we retain the original persistent advising structure, which uses the PPR strategy to enable probabilistic advice reuse. Our framework also uses the original initial reuse probability of 80%. However, it adopts a lower decay rate in our Encoder BPA framework so that advice stored in persistent memory can be reused for a longer period. When external advice is available, the learning agent applies the advised action and stores the $( z _ { i } ,$ action) pair in the persistent memory. When no external advice is available, the associated action may be reused in latent space advice retrieval if a similar $z _ { i }$ is found in the persistent memory.

![](images/3db6f24b68b0fad216fe202ddb046782d5b17edbf03ac7f81d3c0b6bae325299.jpg)  
Figure 2: The PPR mechanism in Encoder BPA. In Encoder BPA, a trained encoder is used to map a state sequence into a latent representation at every time step. When external advice is available, the learning agent applies the advised action and stores the (latent, action) pair in the persistent memory. When no external advice is available, the associated action may be reused in latent space advice retrieval.

## 3.3 Latent Space Advice Retrieval

In Encoder BPA, advice retrieval includes three components: the similarity threshold τ , the top-(k) aggregation strategy, and the consistency threshold δ. Our framework first measures similarity in latent space between the current state-sequence representation and stored memories. Specifically, for each latent representation $z _ { i }$ at time step t, we filter out all similar previous latents using a cosine similarity threshold condition as follows:

$$
\sin ( z _ { c } , z _ { i } ) > \tau\tag{4}
$$

Both $z _ { c }$ and $z _ { i }$ are latent representations of state sequences with the same length. If the similarity between them exceeds the threshold $\tau ,$ the current situation is considered suficiently similar to a previously advised situation, and the corresponding stored action can be reused.

Then, the top-(k) aggregation is used to select similar previous latents from the memory set, and a majority vote or averaging strategy is used to form a candidate action [Mesto and Cruz, 2026]. The corresponding reuse probability can also be calculated by averaging the reuse rates of the selected memories.

To ensure that the retrieved action can be reused reliably, this study uses a consistency threshold δ to evaluate the action consistency of the memory set. In a discrete action space, majority voting is applied, and the consistency threshold specifies the minimum proportion of the majority action in top-(k) similar memories. In a continuous action space, the retrieved actions are averaged to obtain the candidate action. Their consistency is evaluated using the mean action distance, and the candidate action is reused only when the mean action distance is below the consistency threshold.

## 3.4 Experimental Design

In this research, the average reward over the most recent 100 episodes across 9 diferent random seeds is used to evaluate the performance of diferent IntRL approaches. A higher average reward indicates that the agent achieves better training performance. Another important metric is the number of advice reuses. This value increases only when the persistent memory successfully provides a piece of advice to reuse. Therefore, it directly reflects how often the persistent memory contributes to the action selection during training.

For model comparison, Deep Q-Network (DQN) is used as the reinforcement learning backbone in discrete action spaces, while Soft Actor-Critic (SAC) is used in continuous action spaces. In visualization, baseline rep resents pure RL, non-persistent represents IntRL, persis tent represents persistent advising, and broad-persistent represents the original BPA approach. Since the original BPA method relies on an ϵ-greedy strategy, this study follows the original BPA implementation for the DQN setting. In continuous action spaces, the PPR mechanism in BPA behaves similarly to persistent advising, allowing a fair comparison.

## 3.5 Experimental Environments Cart-Pole

Cart-Pole is a classic reinforcement learning control environment that aims to balance a pole by controlling a cart under the pole using forces in the left or right directions [Brockman et al., 2016]. The state is continuous and in 4 dimensions, and the action is discrete, representing the direction of the force. The agent can continuously obtain rewards by balancing the pole at each time step.

![](images/8b025664017afa17c263ce05e90fea41ae4f56be8b44015c64b22f03c721e978.jpg)  
Figure 3: The environments used in the experiments.

## Inverted Pendulum

The second experiment is in the Inverted Pendulum environment. This environment is similar to Cart-Pole in terms of rules, but its action space is continuous. The action space is one-dimensional and ranges from −3.0 to 3.0, where negative and positive values represent forces applied in opposite directions.

## MuJoCo Ant

The third experiment is in the MuJoCo Ant environment. In this environment, an agent learns to coordinate a four-legged robot to move forward eficiently while maintaining balance. The state is continuous and in 105 dimensions, including position, orientation, and joint information. The action is continuous and in 8 dimensions, controlling the torques applied to the legs of the agent. The reward encourages the ant to move forward eficiently while maintaining balance.

## POMDP Ant

The fourth experiment is conducted in a partially observable Ant environment. Unlike the original Ant environment, under the POMDP setting, the agent cannot observe the complete environment state and must make decisions based only on partial observations. To construct the POMDP settings, the state variables at indices 13 and 14 are masked and therefore unavailable to the learning agent. Indices 13 to 14 represent the x, y coordinate velocity of the torso. In addition, the agent of a POMDP environment learns from a sequence of recent observations rather than relying only on the current state. In this experiment, the observation history consists of the four most recent observations. This is distinct from the Encoder BPA history length L = 16, which is used as the input sequence to the latent encoder for advice retrieval.

In this experimental environment, the teacher network is trained using interactive reinforcement learning, in which the teacher network of the original Ant environment provides external advice. After training, the external advice is removed, and the teacher network interacts with the environment independently. An evaluation curve is then generated to assess the learned policy and verify the efectiveness of the advice provided by the teacher network.

## 3.6 Advisor Configuration

The advisor is controlled by two parameters: frequency and accuracy. The frequency represents how often the advisor gives advice. The accuracy parameter controls the reliability of the advice. There are two advisor settings used in this experiment, as follows:

• Advisor 1: frequency: 0.5, accuracy: 1.0

• Advisor 2: frequency: 0.5, accuracy: 0.9

By using these two advisor configurations, the experiments can compare both the efectiveness and robustness of diferent methods. In practice, the frequency of human advice varies across tasks. This study therefore uses a fixed advice frequency as a reference setting.

## 4 Results and Discussion

This chapter reports and analyses the experimental results obtained from the Cart-Pole, Inverted Pendulum, Ant, and POMDP Ant environments.

## 4.1 Teacher Network Performance

This section presents the training performance of the teacher networks across diferent environments. Figure 4a shows that the training curve of the teacher net work in the Cart-Pole environment gradually converges to an average reward of approximately 300, which represents high performance. Figure 4b shows that the training curve gradually converges to an average reward of approximately 300 and maintains high performance after 2000 episodes. Figure 4c shows that the training reward of the teacher network in the MuJoCo Ant environment gradually increases to approximately 3000 after 1000 episodes. In addition, Figure 4d shows that, with guidance from the Ant teacher network, the teacher agent in the POMDP Ant environment gradually converges to an average reward of approximately 2500 per episode. Figure 4e further demonstrates that when interacting with the environment independently without external advice, the trained teacher network in the POMDP Ant envi ronment maintains an average reward of around 1400 per episode.

![](images/a587bf7f5fd742add50e7da906eb3b33392a9f335d52f87f4e724d0eacea47fe.jpg)  
(a) Teacher network in the Cart-Pole environment. The DQN teacher network is trained for 5000 episodes and evaluated by the average reward.

![](images/cede309f62858e62577b6c33e395656188646aef19414211063a3fbe12a7aeea.jpg)

![](images/321bcf1a9ccccba48524211c3c4aac372876c3534e8e64fbb357ff3974f86f94.jpg)  
(c) Teacher network in the Ant environment. The SAC teacher network is trained for 5000 episodes and evaluated by the average reward.

(b) Teacher network in the Inverted Pendulum environment. The SAC teacher network is trained for 5000 episodes and evaluated by the average reward.  
![](images/2eee5552a79a9c5d4490929255791a25a21d5228ecb89ca8aac73f1ede534f1b.jpg)  
(d) Teacher network in the POMDP environment. In the POMDP experiment, the teacher network of the original Ant environment provides external advice to accelerate training.

![](images/984037e98fdcb397d98148c49810d08924c3f82e372ffb648258f80e1875f14e.jpg)  
(e) The evaluation curve of the teacher network in the POMDP environment. The teacher network is evaluated by using a deterministic policy with exploration disabled.  
Figure 4: The training curve of the teacher network in diferent environments.

## 4.2 K-means and Encoder

In the original BPA approach, the elbow method is used to select the K value for K-means clustering in both the Cart-Pole and the robotic environment. To maintain methodological consistency, this section presents results from the elbow method. Table 1 shows the K values applied to each environment.

Table 1: The results of the elbow method.
<table><tr><td rowspan=1 colspan=1>Parameter</td><td rowspan=1 colspan=1>Cart-Pole</td><td rowspan=1 colspan=1>Inv. Pend.</td><td rowspan=1 colspan=1>Ant</td><td rowspan=1 colspan=1>POMDP</td></tr><tr><td rowspan=1 colspan=1>K Value</td><td rowspan=1 colspan=1>3</td><td rowspan=1 colspan=1>4</td><td rowspan=1 colspan=1>2</td><td rowspan=1 colspan=1>4</td></tr></table>

To evaluate whether the learned latent representations capture action-discriminative features associated with the teacher policy, this part presents Spearman correlation results for the LSTM and Mamba encoders across diferent environments. Table 2 shows the Spearman correlation results in diferent environments.

The Spearman correlation results show a strong negative monotonic relationship between representation similarity and action distance. This means that when two state sequences have more similar latent representations, the distance between their corresponding teacher actions tends to be smaller. Therefore, the results support the reliability of using a similarity threshold to identify relevant previous advice for action selection.

Table 2: Spearman correlation between representation cosine similarity and action distance across diferent environments and encoders.
<table><tr><td rowspan=1 colspan=1>Encoder</td><td rowspan=1 colspan=1>Cart-Pole</td><td rowspan=1 colspan=1>Inv. Pend.</td><td rowspan=1 colspan=1>Ant</td><td rowspan=1 colspan=1>POMDP</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>-0.816</td><td rowspan=1 colspan=1>-0.986</td><td rowspan=1 colspan=1>-0.870</td><td rowspan=1 colspan=1>-0.886</td></tr><tr><td rowspan=1 colspan=1>Mamba</td><td rowspan=1 colspan=1>-0.819</td><td rowspan=1 colspan=1>-0.967</td><td rowspan=1 colspan=1>-0.867</td><td rowspan=1 colspan=1>-0.889</td></tr></table>

## 4.3 Cart-Pole

The Cart-Pole experiments evaluate whether the proposed Encoder BPA approach improves training performance by reusing stored advice in a discrete action environment. Based on the methodology, latent space advice retrieval includes a cosine similarity threshold, a top-(k) aggregation, and a consistency threshold. The main settings are as follows:

• Similarity threshold τ : 0.95

persistent non\_persistent baseline

![](images/500cea104deb8d120fa9f7afae7f1c8c5a37bf5e443d4eb2241ff8e9fa4bdef9.jpg)

(a) The Cart-Pole Experiments. This visualisation evaluates the performance of diferent IntRL approaches in the discrete action environment.  
![](images/b459045b08f666bd11e291cad4d4e8f197fa0db3693031aa5f0010503d5833f0.jpg)  
(c) The Ant experiments. This visualisation evaluates the performance in the complex control environment.

![](images/5a3edf7b9a29d80bcdfd29ee7461f6022c67fce1a9a48cf31a24a45d1af2012e.jpg)  
(b) The Inverted Pendulum experiments. This visualisation evaluates the performance of diferent IntRL approaches in the continuous action environment.

Ant Experiments in POMDP Settings  
![](images/13dd9db7e215d218f22adb1d6fdd16b728cc300c9a3b3c5296d78c393c8a24fb.jpg)  
(d) The POMDP Ant experiments. This visualisation evaluates the performance in the POMDP environment.  
Figure 5: Comparison of diferent interactive reinforcement learning approaches across diferent environments. The solid lines represent the average reward across multiple random seeds, while the shaded regions represent the corresponding standard deviation around the mean.

• Top-(k) aggregation: top-5 majority voting

• Consistency threshold δ: 0.5

For a discrete action space, the consistency threshold represents the minimum proportion of the majority action. Since Cart-Pole has only two actions, 0 and 1, the majority action must appear in at least 3 of the top-5 similar memories.

The training curve in Figure 5a shows that the Encoder BPA approach using an LSTM or Mamba encoder achieves the best early training performance. These results suggest that representation learning and latent space advice retrieval can efectively support the reuse of previous advice in a discrete action space.

## 4.4 Inverted Pendulum

The Inverted Pendulum experiments evaluate whether the proposed persistent advising approach improves training performance in a continuous-action environment. The advice retrieved in the Inverted Pendulum task is a continuous value. Therefore, the averaging strategy is used to select the candidate action.

• Similarity threshold τ : 0.995

• Top-(k) aggregation: top-5 averaging strategy

• Consistency threshold δ: 0.5

In a continuous action space, the consistency threshold is evaluated using the mean action distance.

As shown in Figure 5b, the training performance of the Encoder BPA increases in the early training stage. However, in later episodes, the Encoder BPA agent may reuse inaccurate or inappropriate advice through the averaging strategy, which can negatively afect its action selection and consequently reduce the rewards it receives.

## 4.5 MuJoCo Ant

The Ant experiments test the proposed approach in a more complex control environment. At each step, the agent must use information about body position, joint configuration, and velocity to coordinate several joints simultaneously.

• Similarity threshold τ: 0.97

• Top-(k) aggregation: top-5 averaging strategy

• Consistency threshold δ: 0.5

Since the Ant environment uses continuous state and action spaces, the averaging strategy and mean action distance are applied.

Figure 5c shows that the Encoder BPA approach achieves a greater performance improvement during the early stages of training than the other IntRL approaches in the Ant environment. This suggests that the proposed approach is efective in complex control environments with high-dimensional state and continuous action spaces.

## 4.6 POMDP Ant

The POMDP experiments evaluate whether the proposed approach remains efective when the agent cannot directly observe the complete state of the environment.

• Similarity threshold τ: 0.97

• Top-(k) aggregation: top-5 averaging strategy

• Consistency threshold δ: 0.5

The POMDP experiment uses the same advice retrieval configuration as the MuJoCo Ant environment.

Figure 5d shows that the Encoder BPA approach substantially improved early-stage training performance in the partially observable environment. By encoding the sequence of recent states into a latent representation, the encoder may capture useful temporal information that may not be fully available from a single observation or a short observation sequence. This may enable the Encoder BPA agent to retrieve relevant previous advice, resulting in better performance during the early stages of training.

## 4.7 Advice Reuse Frequency

This section compares the frequency of advice reuse in the persistent memory between diferent approaches, including state-based persistent advising, BPA, and Encoder BPA.

Table 3 shows that the Encoder BPA provides more opportunities to reuse previous advice in similar states compared to state-based persistent advising and the original BPA approach in most cases. In continuous state spaces, exact state revisits are dificult within a limited number of steps, resulting in an extremely low average reuse frequency for persistent advising.

Table 3: Average number of advice reuses across multiple experiments.
<table><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Cart-Pole</td><td rowspan=1 colspan=1>Inv. Pend.</td><td rowspan=1 colspan=1>Ant</td><td rowspan=1 colspan=1>POMDP</td></tr><tr><td rowspan=1 colspan=1>Persistent</td><td rowspan=1 colspan=1>18</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0</td><td rowspan=1 colspan=1>0</td></tr><tr><td rowspan=1 colspan=1>BPA</td><td rowspan=1 colspan=1>6032</td><td rowspan=1 colspan=1>5892</td><td rowspan=1 colspan=1>141952</td><td rowspan=1 colspan=1>73990</td></tr><tr><td rowspan=1 colspan=1>LSTM</td><td rowspan=1 colspan=1>28025</td><td rowspan=1 colspan=1>16553</td><td rowspan=1 colspan=1>119721</td><td rowspan=1 colspan=1>63288</td></tr><tr><td rowspan=1 colspan=1>Mamba</td><td rowspan=1 colspan=1>27737</td><td rowspan=1 colspan=1>17813</td><td rowspan=1 colspan=1>127132</td><td rowspan=1 colspan=1>99505</td></tr></table>

## 4.8 Supporting Experiments

This section provides supporting experiments to examine how diferent advisor configurations and threshold values afect the agent’s learning performance. This section uses the average reward across 9 diferent random seeds. The experimental results for diferent similarity thresholds are shown in Figure 6a. The results show that a lower similarity threshold provides more opportunities to reuse previous advice, leading to better performance during the early stages of training. However, in later episodes, a low threshold and the averaging strategy continue to incorporate advice that is no longer appropriate, ultimately preventing the agent from converging.

Secondly, the results obtained with diferent consistency thresholds are shown in Figure 6b. Reducing the consistency threshold from 0.5 to 0.3 substantially decreased the number of advice reuses, suggesting that many candidate actions were rejected under the stricter threshold. Based on the training curve, 0.5 achieved better training performance and was therefore selected for the Ant experiments.

In addition, the experimental results obtained using another advisor whose accuracy was reduced to 0.9 are shown in Figure 6c. The results show that the Encoder BPA approach remains sensitive to noisy advice, similar to other IntRL approaches. The encoder only enables previous advice to be retrieved and reused for similar states, but it does not directly evaluate or correct the quality of the advice.

## 4.9 Limitations

Some limitations of the current Encoder BPA approach are discussed below. Even when diferent encoders use the same network architecture, training procedure, and hyperparameter settings, their performance may still vary, resulting in diferences in the quality of advice retrieval.

Second, the latent space advice retrieval mechanism still needs further improvement. Although the method can accelerate training in the early stage, its performance may decrease later in continuous action environments. In addition, the performance of Encoder BPA is sensitive to the similarity and consistency thresholds. A small change in either threshold can cause a large change in the frequency of advice reuse. Therefore, more experiments may be needed to find suitable threshold values for each environment.

![](images/9219731756acb2e62793aec84211be555c3fcdc61dce664ddc1a75d21bb1301b.jpg)  
(a) Diferent similarity thresholds in the Ant environment. In latent space advice retrieval, the top-5 averaging strategy and the 0.5 consistency threshold remain the same.

![](images/e2f62af85da2a8fa14f346adf19d67480188d2e19147afd4a4621ebd0ff1ccb8.jpg)  
(b) Diferent consistency thresholds in the Ant environment. In latent space advice retrieval, the 0.97 similarity threshold and the top-5 averaging strategy remain the same.

![](images/a1fa6f0cf976fba45ec646445f67b7faba1a74e4e37109625e9adf63a13c1c02.jpg)  
(c) Diferent advisor in the Ant environment. The advisor is Advisor 2, as defined in the Experimental Design section, with an accuracy of 90%. The advice retrieval settings remain unchanged.  
Figure 6: Supporting experiments in the MuJoCo Ant environment.

Finally, Encoder BPA requires additional computation at each training step. Before retrieving previous advice, the encoder must process the current state sequence and generate its latent representation. Nevertheless, this computational cost may be acceptable in human-inthe-loop reinforcement learning scenarios where human trainers are unable to provide advice frequently.

## 4.10 Future Direction

Future work could explore contrastive learning to improve the stability of encoder training. The loss function could be designed to place state sequences associated with similar actions closer together in the latent space, while increasing the distance between latents associated with dissimilar actions [Khosla et al., 2020; Chuang et al., 2020]. This may produce a more structured latent space, improve advice retrieval across diferent types of encoders and training runs, and may reduce the sensitivity of the similarity and consistency thresholds.

The top-(k) aggregation in latent space advice retrieval can also be adjusted during agent training. Although combining multiple retrieved memories is useful in the early stage, mismatched advice may negatively affect the candidate action in a continuous action environment. Future work could gradually reduce the value of (k), or increase the decay rate of the persistent memory during training. These changes would reduce the influence of mismatched memories and allow the agent to rely more on its own policy as training progresses [Nguyen et al., 2023].

## 5 Conclusion

The experimental results show that Encoder BPA improves early-stage training performance. This improvement may come partly from the additional teacheraction supervision and partly from the learned representations for advice retrieval. The Cart-Pole, Inverted Pendulum, and Ant experiments showed that the proposed approach could improve early-stage learning across discrete action, continuous action, and complex control tasks. The greater improvement in POMDP Ant further suggests that the state-history encoding of the proposed BPA approach can make better use of teacher advice when the current observation does not contain complete state information. Nevertheless, the results also show that performance could be negatively afected when the averaging strategy adopts inappropriate advice in a continuous action space.

Future experiments can focus on applying contrastive learning to the Encoder pre-training process so that state sequences linked to similar teacher actions are represented closer together. Additionally, more experiments can be conducted using dynamic advice retrieval conditions. The agent can rely more on stored advice at the beginning and gradually depend more on its own policy.

## Acknowledgments

The experiments reported in this work were performed on the UNSW compute cluster Katana (DOI: 10.26190/669X-A286).

## References

[Bengio et al., 2012] Yoshua Bengio, Aaron Courville, and Pascal Vincent. Representation Learning: A Re view and New Perspectives, June 2012.

[Bignold et al., 2021] Adam Bignold, Francisco Cruz, Richard Dazeley, Peter Vamplew, and Cameron Foale. An Evaluation Methodology for Interactive Reinforcement Learning with Simulated Users. Biomimetics, 6(1):13, February 2021.

[Bignold et al., 2023a] Adam Bignold, Francisco Cruz, Richard Dazeley, Peter Vamplew, and Cameron Foale. Human Engagement Providing Evaluative and Informative Advice for Interactive Reinforcement Learning. Neural Computing and Applications, 35(25):18215–18230, September 2023. arXiv:2009.09575 [cs.AI].

[Bignold et al., 2023b] Adam Bignold, Francisco Cruz, Richard Dazeley, Peter Vamplew, and Cameron Foale. Persistent rule-based interactive reinforcement learning. Neural Computing and Applications, 35(32):23411–23428, November 2023.

[Brockman et al., 2016] Greg Brockman, Vicki Cheung, Ludwig Pettersson, Jonas Schneider, John Schulman, Jie Tang, and Wojciech Zaremba. OpenAI Gym, June 2016. arXiv:1606.01540 [cs.LG].

[Chen et al., 2020] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. A Simple Framework for Contrastive Learning of Visual Representations, July 2020. arXiv:2002.05709 [cs.LG].

[Chuang et al., 2020] Ching-Yao Chuang, Joshua Robinson, Yen-Chen Lin, Antonio Torralba, and Stefanie Jegelka. Debiased Contrastive Learning. In Advances in Neural Information Processing Systems, volume 33, pages 8765–8775. Curran Associates, Inc., 2020.

[Cruz and Igarashi, 2021] Christian Arzate Cruz and Takeo Igarashi. A Survey on Interactive Reinforcement Learning: Design Principles and Open Challenges, May 2021. arXiv:2105.12949 [cs].

[Echchahed and Castro, 2025] Ayoub Echchahed and Pablo Samuel Castro. A Survey of State Representation Learning for Deep Reinforcement Learning, June 2025. arXiv:2506.17518 [cs].

[Khosla et al., 2020] Prannay Khosla, Piotr Teterwak, Chen Wang, Aaron Sarna, Yonglong Tian, Phillip Isola, Aaron Maschinot, Ce Liu, and Dilip Krishnan.

Supervised Contrastive Learning. In Advances in Neural Information Processing Systems, volume 33, pages 18661–18673. Curran Associates, Inc., 2020.

[Mesto and Cruz, 2026] Maher Mesto and Francisco Cruz. The Consensus Paradox: When Low Disagreement Leads to Catastrophic Failure in Multi-teacher Reinforcement Learning. In Miaomiao Liu, Xin Yu, Chang Xu, and Yiliao Song, editors, AI 2025: Advances in Artificial Intelligence, pages 426–438, Singapore, 2026. Springer Nature.

[Millan et al., 2019] Cristian Millan, Bruno Fernandes, and Francisco Cruz. Human feedback in continuous actor-critic reinforcement learning. 27th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning, 2019.

[Moreira et al., 2020] Ithan Moreira, Javier Rivas, Francisco Cruz, Richard Dazeley, Angel Ayala, and Bruno Fernandes. Deep Reinforcement Learning with Interactive Feedback in a Human-Robot Environment. Applied Sciences, 10(16):5574, August 2020. arXiv:2007.03363 [cs].

[Murphy, 2022] Kevin P. Murphy. Probabilistic Machine Learning: An Introduction. MIT Press, Cambridge, UNITED STATES, 2022.

[Najar and Chetouani, 2021] Anis Najar and Mohamed Chetouani. Reinforcement Learning With Human Advice: A Survey. Frontiers in Robotics and AI, 8, June 2021.

[Nguyen et al., 2023] Hung Son Nguyen, Francisco Cruz, and Richard Dazeley. Towards a Broad-Persistent Advising Approach for Deep Interactive Reinforcement Learning in Robotic Environments. Sensors, 23(5), March 2023.

[Plaat, 2022] Aske Plaat. Deep Reinforcement Learning. Springer Nature, Singapore, 2022.

[Srinivas et al., 2020] Aravind Srinivas, Michael Laskin, and Pieter Abbeel. CURL: Contrastive Unsupervised Representations for Reinforcement Learning, September 2020. arXiv:2004.04136 [cs].

[Stooke et al., 2021] Adam Stooke, Kimin Lee, Pieter Abbeel, and Michael Laskin. Decoupling Representation Learning from Reinforcement Learning, May 2021. arXiv:2009.08319 [cs.LG].

[Sutton and Barto, 2018] Richard S. Sutton and Andrew G. Barto. Reinforcement Learning, Second Edition: An Introduction. MIT Press, Cambridge, UNITED STATES, 2018.

[Yarats et al., 2021] Denis Yarats, Rob Fergus, Alessandro Lazaric, and Lerrel Pinto. Reinforcement Learning with Prototypical Representations. In Proceedings of the 38th International Conference on Machine Learning, pages 11920–11931. PMLR, July 2021.