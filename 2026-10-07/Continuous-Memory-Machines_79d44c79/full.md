# Continuous Memory Machines

Ciaran Regan<sup>1,2∗</sup> Kai Arulkumaran<sup>2</sup> Luke Darlow<sup>2</sup> Stefania Druga<sup>2</sup> Sebastian Risi<sup>2</sup> Llion Jones<sup>2</sup>

<sup>1</sup>University of Tsukuba <sup>2</sup>Sakana AI

## Abstract

Recurrent neural networks typically compress information into a single vectorvalued recurrent state, forcing short-term computation and long-term retention to share the same representation. Past extensions alleviate this bottleneck by increasing the memory capacity or separating timescales, but lack the combination of rapid neuron-level processing and longer-term retention found in biology. To that end, we introduce the Continuous Memory Machine (CMM), a recurrent architecture with matrix-valued short- and long-term memory states serving distinct functional roles. Building on the Continuous Thought Machine (CTM), the CMM’s short-term memory tracks recent neural activity, with uniquely parameterized neuron-level models learning to use these activity patterns for computation. A persistent long-term memory stores information for later use, with a Transformer jointly updating both memory stores, providing an expressive bidirectional read– write mechanism such that each store can reorganize its own contents and both read from and write to the other. Across algorithmic, in-context learning, and recurrent reasoning tasks, the CMM outperforms a broad suite of baselines, exhibiting stronger generalization than prior memory-augmented networks while preserving the CTM’s interpretable attention patterns. Code is available at https://github. com/SakanaAI/continuous-memory-machines.

## 1 Introduction

Many complex problems require sequential information to be maintained and manipulated over multiple timescales. A model executing an algorithm, for instance, needs to simultaneously track rapidly changing inputs while managing information acquired earlier. Such tasks can be challenging for Transformers [1] which, despite their dominance in sequence processing, can struggle with state tracking [2]. Recurrent neural networks (RNNs), on the other hand, are naturally suited to such settings, updating a persistent hidden state over time. This state, however, typically compresses the entire sequence history into a single hidden vector [3, 4], with the same representation responsible for both short-term computation and long-term memory. This creates a bottleneck that limits performance on tasks that may benefit from both rapid computation and persistent storage [5].

Several lines of work have sought to address these limitations. Multiscale RNNs introduce hidden states that update with different rates [6, 7], allowing some components to respond to recent inputs while others retain information over longer horizons. Fast weight RNNs take an alternative approach, replacing vector-valued with matrix-valued hidden states, supporting associative storage [8–10]. Memory-augmented neural networks (MANNs) couple a controller, such as a long short-term memory [LSTM; 3], to an addressable matrix, providing an external store from which information can be retrieved and updated [5, 11]. Recurrent Transformers extend the effective context of standard Transformers by carrying hidden representations or memory tokens across successive input blocks [12,

![](images/2a1832d81cb7774eade4c422d4cdeed8a86fedf1cea85b71f5b53f0e8a63a1bb.jpg)  
Figure 1: The Continuous Memory Machine. At timestep t, the current observation $o ^ { t }$ and previous activated state $z ^ { t }$ are processed by the feedforward synapse model to produce a pre-activated state $a ^ { t }$ This state is appended to the sliding short-term memory (STM) $A ^ { t }$ $A ^ { t }$ and the long-term memory $B ^ { t }$ are jointly processed by a Transformer, producing updated memories $\tilde { A } ^ { t }$ and $\tilde { B } ^ { t } . \tilde { A } ^ { t }$ is passed through the neuron-level models to produce the next activated state $z ^ { t + 1 }$ . From the activated-state history, the model computes the synchronization $S ^ { t + 1 }$ and output $\boldsymbol { y } ^ { t + 1 } . ~ \boldsymbol { A } ^ { t } , ~ \tilde { \boldsymbol { B } } ^ { t } = \boldsymbol { B } ^ { t + 1 }$ , and $z ^ { t + 1 }$ are carried forward to the next iteration.

13]. These approaches expand capacity or separate timescales, but the combination of persistent memory with rich short-term neural dynamics remains relatively unexplored.

Biological neural networks present this as a possible solution, with memory organized across interact ing dynamical systems with distinct functional roles [14, 15]. The complementary learning systems framework argues that this separation is necessary, as a single memory store cannot simultaneously encode new information and retain accumulated knowledge stably without interference [14, 16]. Existing RNN architectures fail to capture this sufficiently. Multiscale RNNs retain vector-valued memories, fast weight RNNs increase the capacity of a single recurrent state, and MANNs pair an external matrix-valued memory with a limited controller state. Conversely, the recently-introduced Continuous Thought Machine [CTM; 17] demonstrates the benefits of rich short-term dynamics, directly leveraging recent neural activation patterns for computation, but lacks long-term storage, as we show in section 4. This motivates the question: “Can the short-term dynamics of the CTM be combined with the capabilities of memory-augmented architectures to create a more capable recurrent model?”

To that end, we introduce the Continuous Memory Machine (CMM), a recurrent architecture that explicitly separates short- and long-term memory into distinct matrix-valued recurrent states. The CMM combines the persistent memory mechanisms of MANNs with the CTM’s ability to exploit rich short-term neural dynamics for computation. Its short- and long-term memory states are processed by a shared Transformer, allowing each memory to update its own contents while also exchanging information with the other through learned bidirectional interactions. Across algorithmic, in-context learning, maze-solving, and image-classification tasks, the CMM achieves strong performance and improved generalization while preserving the CTM’s interpretable reasoning behavior. Analysis reveals the CMM learns task-dependent memory strategies, relying on persistent long-term memory for memory-intensive tasks while suppressing cross-memory interaction when it is unnecessary. In summary, our contributions are as follows:

• We introduce the Continuous Memory Machine, an RNN with distinct matrix-valued shortand long-term memories. A shared Transformer updates both, supporting within-memory computation and learned cross-memory interaction.

• We show the CMM achieves strong performance in algorithmic, few-shot learning and reasoning tasks, with improved generalization while preserving the interpretability of the CTM.

• We show that the CMM learns task-dependent memory strategies, using long-term memory for retention and manipulation while suppressing interactions when persistent storage is unnecessary.

## 2 Related Work

## 2.1 Recurrent Neural Networks

Recurrent neural networks [RNNs; 18–21] model temporal sequences by maintaining a hidden state that is updated over time. LSTMs introduced gating mechanisms to mitigate vanishing gradients, and, together with Gated Recurrent Units [GRUs; 4], have become the standard recurrent architectures for sequence modeling. However, these models compress the sequence history into a fixed-size hidden vector, limiting how much information can be retained [22]. Extensions address this limitation by modifying the structure of the hidden state or its update dynamics.

Multiscale RNNs aim to capture long-term dependencies through recurrent dynamics operating at different timescales [6]. By updating less frequently, slower recurrent states can preserve information over longer horizons while shortening the effective gradient path between temporally distant inputs. Early work by Schmidhuber [6, 23] proposed a hierarchy of RNN layers running at different temporal resolutions. Koutnik et al. [7] propose Clockwork RNNs, where the hidden layer is partitioned into separate modules, each processing inputs at a pre-specified temporal granularity. Follow-up work allowed multiscale RNNs to learn this granularity [24]. Fast-Slow RNNs split processing specifically into two interacting streams, with a slowly evolving higher-level recurrent module and faster lower-level modules, enabling long-term information to be retained while preserving sensitivity to recent inputs [25].

Fast weight RNNs expand the recurrent state of standard RNNs from a vector to a matrix, increasing its capacity for associative storage [26, 8, 27]. An early example is the Fast Weight Programmer [8], which writes key–value associations into a memory matrix and retrieves them using query vectors. This can be interpreted as linear attention [9], which has motivated a range of modern recurrent architectures [9, 28–30]. xLSTM, for example, introduces an LSTM variant with matrix-valued memory [31], while the Gated Delta Network [GDN; 10] combines gating with a delta-rule [32] update to selectively modify matrix-valued state.

MANNs augment RNNs with external matrix-valued memory stores [11, 33–36]. Neural Turing Machines [NTMs; 5] couple a neural controller with an addressable memory, analogous to the random access memory coupled to a CPU in a computer, and access it through content- and location-based attention. Differentiable Neural Computers [DNCs; 37] extend this with temporal and usage-based addressing, enabling the model to learn algorithms such as sorting through backpropagation. Although this improves retention, they can struggle with reasoning tasks as the stored memories do not interact with each other. To address this, Santoro et al. [38] introduced the Relational Memory Core (RMC), allowing memory slots to interact via self-attention. Building on the idea that attention itself can serve as a memory primitive, Nam and Seo [39] reformulate attention as a readable and writable memory, using it to construct MANNs (LSAM and NAM-TM) that matched or exceeded the DNC’s performance with simpler primitives.

Recurrent Transformers introduce memory mechanisms to the Transformer [1] that carry information across input blocks [40, 41]. Transformer-XL [12] caches hidden representations from previous blocks and makes them available to subsequent attention operations, extending the context. The Recurrent Memory Transformer [RMT; 13] instead introduces dedicated memory tokens whose updated representations are passed between blocks, allowing the model to learn how information is retained and retrieved.

## 2.2 Biological Memory

Biological memory is not organized as a single store but instead as a collection of dynamical systems operating at different timescales [15]. The complementary learning systems framework [14, 16] argues that this organization is computationally necessary, since a single memory store cannot simultaneously encode new experiences quickly and retain accumulated knowledge stably without one interfering with the other. Brains avoid this trade-off by pairing a fast hippocampal system and a slow neocortical system [14, 16], with a similar split appearing within working memory itself [42, 43]. Across both timescales, information is represented through coordinated neural dynamics [44, 45]. While prior work in artificial neural networks has exploited neural dynamics for computation [46, 17], their integration across distinct memory timescales remains underexplored.

## 3 Method

We introduce the Continuous Memory Machine (CMM), a recurrent architecture with distinct matrixvalued short- and long-term hidden states with different functional roles. The CMM builds on the CTM, draws on the memory mechanisms of MANNs, and, similar to multiscale RNNs, separates timescales to support distinct retention strategies. A Transformer jointly updates the two stores, providing an expressive mechanism for retrieving, writing, and reorganizing memories, both within and across stores. An overview of the architecture is shown in fig. 1. We review the CTM before introducing the CMM architecture.

## 3.1 Continuous Thought Machines

The CTM is a recurrent architecture whose latent representation is grounded in neural dynamics. It combines two key ideas. First, neuron-level temporal processing: each neuron independently processes a sliding window of its own activation history through a learned transformation, allowing individual neurons to discover distinct temporal features. Second, neural synchronization as the latent representation: pairwise temporal correlations between neuron activities serve as the model’s internal state, providing a smoothly evolving, high-dimensional representation. The CTM architecture is depicted in Appendix fig. 7a.

Concretely, the CTM processes information as follows. At each timestep t, the model receives the concatenation of its current activated state $z ^ { t } \in \mathbb { R } ^ { d }$ and an external observation $o ^ { t } \in \mathbb { R } ^ { d _ { \mathrm { o b s } } }$ , where d and $d _ { \mathrm { o b s } }$ denote the dimensionalities of the hidden and input vectors, respectively. A feedforward synapse network produces a pre-activated state $a ^ { t } \colon$

$$
a ^ { t } = f ( z ^ { t } , o ^ { t } ) \in \mathbb R ^ { d } ,\tag{1}
$$

where f is a learned feedforward network. This pre-activated state is appended to a sliding window of the $M _ { S }$ most recent pre-activations, forming a matrix $A ^ { t } \in \mathbb { R } ^ { d \times M _ { S } }$ . In this work, we refer to this matrix as the Short-Term Memory (STM) of the model. Each neuron i then independently maps its row of this history to the next activated state via a neuron-level model (NLM):

$$
z _ { i } ^ { t + 1 } = g _ { i } ( A _ { i } ^ { t } ) ,\tag{2}
$$

where $g _ { i }$ is a small feedforward network with parameters unique to each neuron i. These NLMs give each neuron a learned temporal response based on its historical activity, which was shown to produce more diverse activity patterns and lead to benefits in reasoning tasks [17]. The CTM’s latent representation is synchronization, computed as an exponentially weighted dot product of activated state histories:

$$
S _ { i j } ^ { t + 1 } \propto \sum _ { \tau = 1 } ^ { t + 1 } \mathrm { e } ^ { - r _ { i j } ( t + 1 - \tau ) } z _ { i } ^ { \tau } z _ { j } ^ { \tau } ,\tag{3}
$$

where $r _ { i j }$ is a learned decay parameter for each neuron pair, controlling the influence of past states. The output at each timestep is a mapping from the synchronization matrix, $y ^ { t + 1 } = W _ { \mathrm { o u t } } \mathrm { \bar { v e c } } ( S ^ { t + 1 } )$ . Since $\mathring { A } ^ { t }$ retains only a finite window of recent activity and $S ^ { t }$ compresses the activation history into a decaying summary, the CTM has no persistent mechanism for storing information over long horizons and, as we show empirically in section 4.1, this substantially limits its performance on tasks that require long-term retention.

## 3.2 Continuous Memory Machines

Building upon the CTM, we introduce an explicit matrix-valued long-term memory that operates alongside its sliding-window STM. The two memories are jointly updated by a Transformer, providing an expressive mechanism for writing, retrieval, and reorganization across stores.

Long-Term Memory. We define the Long-Term Memory (LTM) as a matrix $B ^ { t } \in \mathbb { R } ^ { d \times M _ { L } }$ , whose $M _ { L }$ columns are treated as tokens of dimension d. The LTM is initialized with a learned matrix $B ^ { 0 }$ and is subsequently updated at every timestep through interaction with the STM.

Joint Memory Update. At each timestep, the CMM first updates the STM as in the CTM. The synapse produces a pre-activated state $a ^ { t }$ from the previous activated state $z ^ { t }$ and the current observation $o ^ { \bar { t } }$ (eq. (1)), which is appended to the sliding window to form $A ^ { t } \in \mathbb { R } ^ { d \times M _ { S } }$ . Treating the

columns of $A ^ { t }$ and $B ^ { t - 1 }$ as tokens of dimension d, we concatenate a learnable sink token $s \in \mathbb { R } ^ { d }$ [47], the LTM, and the STM into a single sequence,

$$
X ^ { t } = \left[ s ; B ^ { t - 1 } ; A ^ { t } \right] \in \mathbb { R } ^ { d \times ( 1 + M _ { L } + M _ { S } ) } ,\tag{4}
$$

add learned positional embeddings, and pass $X ^ { t }$ through a stack of J Transformer blocks to obtain

$$
\tilde { X } ^ { t } = \mathrm { T r a n s f o r m e r B l o c k s } ( X ^ { t } ) = \left[ \tilde { s } ^ { t } ; \tilde { B } ^ { t } ; \tilde { A } ^ { t } \right] ,\tag{5}
$$

where $\tilde { B } ^ { t } \in \mathbb { R } ^ { d \times M _ { L } }$ and $\tilde { A } ^ { t } \in \mathbb { R } ^ { d \times M _ { S } }$ denote the updated long- and short-term memories.

Activated State and Recurrence. The updated STM $\tilde { A } ^ { t }$ is passed through the neuron-level models to produce the next activated state, as in the CTM (eq. (2)):

$$
z _ { i } ^ { t + 1 } = g _ { i } \left( \tilde { A } _ { i } ^ { t } \right) .\tag{6}
$$

The neuron-level models therefore operate on the STM after its interaction with the LTM, conditioning each neuron’s update on both its recent pre-activation history and information retrieved from longterm memory. The synchronization $S ^ { t + 1 }$ and output $y ^ { t + 1 }$ are then computed from the activated state histories as in eq. (3).

To advance to the next timestep, the updated LTM ${ \tilde { B } } ^ { t }$ is carried forward as the persistent state, while the updated STM $\tilde { A } ^ { t }$ is discarded after producing $z ^ { t + 1 }$ . The original STM $A ^ { t }$ is retained as the rolling window of recent pre-activations and updated with the next synapse output.

Relation to Prior Architectures. The CMM differs from prior MANNs in how memory is accessed and organized. Whereas the DNC [37] uses structured addressing and allocation mechanisms, the CMM learns memory interactions through self-attention, following the RMC [38]. Unlike the RMC, the CMM maintains distinct short- and long-term matrix-valued stores with different functional roles. The LTM retains and reorganizes information across timesteps, while the NLMs extract temporal features from the transformed STM histories to compute the next state. Their private parameters allow each neuron to learn a distinct response to recent activity, which has been shown to aid reasoning [17]. Recurrent Transformers such as RMT provide persistent memory but lack this explicit neuron-level temporal processing. Finally, the sink token enables flexible memory use by allowing the model to suppress attention to either store. As we show in section 4.5, the CMM learns different memory strategies across tasks, relying on persistent LTM for memory-intensive tasks while bypassing LTM when not needed.

## 4 Experiments

Our experiments address two questions. First, does the CMM improve memory manipulation and generalization over existing architectures? Second, does it improve the CTM’s reasoning while preserving its interpretable dynamics? We address these questions through algorithmic, maze-solving, in-context learning and image-classification tasks. Unless otherwise stated, we report the mean and standard deviation across three random seeds.

Baselines We compare the CMM against ten baselines chosen to isolate distinct design choices. To isolate the contribution of the long-term memory mechanism, we include the original CTM [17]. As standard RNN baselines we include a “vanilla” RNN and LSTM [3] with layer normalization [LayerNorm; 48]. For a multiscale RNN baseline we use the Fast-Slow LSTM [FS-LSTM; 25]. For a modern matrix-valued hidden state RNN we use the GDN [10], allowing for negative eigenvalues for improved state tracking capabilities [49]. For MANNs, we include the DNC [37] (superseding the NTM [5]), the RMC [38], and the more recent LSAM and NAM-TM [39]. For a recurrent Transformer we include the RMT [13]. In addition, we report a causal Transformer [1] as a non-recurrent reference. Unlike the bounded-memory recurrent baselines, it directly attends over the full context, giving it a strong advantage.

Ablations We additionally compare against three parameter-matched architectural ablations. CMM-LSTM replaces the CTM backbone with a LayerNorm LSTM while retaining the LTM and memoryupdate Transformer, testing the contribution of the CTM’s short-term dynamics. CTM-Ext. extends the STM window to match the CMM’s total memory slots. CMM-STM is a CMM with $M _ { L } = 0$ testing the CTM’s STM combined with a Transformer, without persistent LTM. Comparing CTM-Ext. and CMM-STM tests the benefit of the Transformer on the STM. Comparing CMM-STM and CMM tests the benefit of a distinct persistent LTM. The Appendix contains detailed architecture comparisons (section A.2), sweeps of LTM size $M _ { L }$ and Transformer depth J (section B.1) and lists all hyperparameters.

![](images/8c1fd28d57bffc80a254c88e913ce00981df1b4f1cdceafc9ededf7970d2164f.jpg)  
Figure 2: Loss for algorithmic tasks. The CMM demonstrates strong performance on all four tasks.

## 4.1 Memory Manipulation: Algorithmic Tasks

We evaluate on algorithmic tasks which have been problematic for standard RNNs [5]. Test losses are reported in fig. 2.

Copy. The copy task tests whether a model can store and recall a long sequence of arbitrary information [5]. The model receives 100 vectors of dimension D = 10 with entries drawn uniformly from [−1, 1] and must reproduce the sequence from memory. The loss is the mean squared error computed over the output values. The non-recurrent Transformer attains a near-zero loss, as expected given its direct access to the complete sequence. Among recurrent baselines, the CMM, DNC, and NAM-TM all achieve losses below $5 \times 1 0 ^ { - \frac { \cdot } { 2 } }$ , with LSAM close behind at $6 . 2 \times 1 0 ^ { - 2 }$ . The CTM, on the other hand, obtains $3 . 1 \times 1 0 ^ { - 1 }$ . This confirms that the CTM’s short activation window is insufficient for sequence-level recall, and that adding the LTM closes the gap to architectures explicitly designed for algorithmic manipulation.

Recall. While the copy task probes sequential storage and retrieval, recall tests a more complex form of memory access in which one data item points to another [5]. The model is shown a sequence of vectors, followed by a query equal to one of them, and must return the item that followed the query in the original sequence. The loss is the binary cross-entropy over the predicted item. The CMM, DNC, and LSAM all achieve test losses below $1 0 ^ { - 3 }$ , with the CMM reaching 99.95% accuracy. The non-recurrent Transformer similarly obtains a loss of $2 . 7 \times 1 0 ^ { - 4 }$ by directly attending over the complete sequence. By contrast, the CTM obtains a loss of $3 . 2 \bar { \times } 1 0 ^ { - 2 }$ , over 60 times that of the CMM, reflecting the limits of its short activation window for retrieval.

![](images/64d46fe5927bf2a9b0d2925fddd59ecdb60574782c88920c40b1806eaa1b5f54.jpg)  
Figure 3: The CMM generalizes best on sort.

Sort. This task tests whether the model can sort data, an important elementary algorithm [5]. The model receives $N = 2 0$ input items, each consisting of a scalar priority drawn uniformly from [−1, 1] and an associated $W ( = 8 )$ -bit binary value vector. After the input phase, the model must output the $K = 1 6$ value vectors corresponding to the K lowest-priority items, sorted in ascending order of priority. The loss is the binary cross-entropy computed over the $\dot { \boldsymbol { K } }$ output positions. The CMM attains the lowest test loss of all evaluated models at $4 . { \overset { \cdot } { 1 } } \times 1 0 ^ { - 3 }$ , improving over the DNC at $1 . 4 \times 1 0 ^ { - 2 }$ and the Transformer at $2 . 7 \times 1 0 ^ { - 2 }$ , while reducing the CTM’s loss by two orders of magnitude. This is notable given that sort requires both retrieval and reorganization.

Table 1: Maze-solving test accuracy.
<table><tr><td>ARCHITECTURE</td><td colspan="2">ACTION ACC. (%) MAZE ACC. (%)</td></tr><tr><td>TRANSFORMER</td><td>41.4±6.9</td><td>0.7±0.5</td></tr><tr><td>RNN</td><td>47.2±2.2</td><td>2.1±0.3</td></tr><tr><td>LSTM</td><td>50.6±10.1</td><td>2.7±1.7</td></tr><tr><td>FS-LSTM</td><td>54.7±9.8</td><td>3.3±1.9</td></tr><tr><td>DNC</td><td>82.7±2.6</td><td>13.8±2.5</td></tr><tr><td>RMC</td><td>23.7±2.7</td><td>0.0±0.0</td></tr><tr><td>NAM-TM</td><td>82.5±0.7</td><td>13.2±0.5</td></tr><tr><td>CTM</td><td>88.8±8.3</td><td>37.4±38.7</td></tr><tr><td>AB: CMM-LSTM</td><td>46.3±7.9</td><td>2.1±1.2</td></tr><tr><td>AB: CMM-STM</td><td>87.0±17.6</td><td> $4 3 . 8 { \pm } 4 2 . 5 $ </td></tr><tr><td>CMM (OURS)</td><td>96.3±3.8</td><td> ${ \bf 6 1 . 6 { \pm } 3 5 . 0 }$ </td></tr></table>

![](images/d477d0cfb6cb2767c898e73a13ab67a43b533616d1f0030bbb8348f033d9ec33.jpg)

![](images/d106598f60d42a9d9714ca8196c40cba226090eae784ed9d682f0c605483a206.jpg)  
Figure 4: Top: CMM attention traces a path through the 39 × 39 maze (crop). Bottom: Accuracy by maze path length. The CMM performs best on longer paths.

To test length generalization, we train on input lengths $N \in \{ 4 , \ldots , 1 6 \}$ with $W = 4$ and $K = 4 ,$ then evaluate up to $N = 5 0$ . The CMM, CTM-Ext., and DNC each use 26 memory elements, with $M _ { S } = 4$ and $M _ { L } = 2 2$ in the CMM, $M _ { S } = 2 6$ in CTM-Ext., and 26 external memory slots in the DNC. Results are shown in fig. 3. At the longest out-of-distribution length, the CMM achieves 72.1% accuracy, compared with the RMT’s 68.2% and the DNC’s near-chance 55.8%.

Parity. The model receives a sequence of random bits followed by a query token and predicts if the sequence is even or odd. We train models on sequence lengths 1–40 and test at length 40. All evaluated architectures except the Transformer and the RMT perform strongly on this task (fig. 2), due to the Transformer’s limited state-tracking capabilities [2, 50].

Across copy, recall, and sort, CMM outperforms CMM-LSTM, showing that the CTM backbone adds value beyond the shared LTM and memory-update Transformer. CTM-Ext. and CMM-STM perform well on copy and recall because their extended STM keeps relevant inputs directly accessible. On sort, which requires selective retrieval and reordering, no ablation surpasses CMM. CMM-STM improves over CTM-Ext. due to the additional Transformer, but it still trails CMM, supporting the benefit of jointly using persistent LTM and transient STM.

## 4.2 Recurrent Reasoning: Maze-Solving

We revisit the maze-solving task of Darlow et al. [17] to test whether the CMM improves the CTM’s recurrent reasoning capabilities while preserving its interpretable attention patterns. The model is given a $3 9 \times 3 9$ maze image with start and goal locations and must predict the sequence of actions connecting them. The model processes the maze over 50 internal steps, producing a full route prediction at each iteration. Following Darlow et al. [17], we omit positional embeddings from the maze image features, encouraging the model to learn an internal representation of the maze’s spatial structure. Experiments use a budget of 10M parameters, with full details provided in section D.7.

We report action and full maze accuracies of held-out mazes in table 1. The CMM outperforms all baselines<sup>2</sup>, including the CTM, increasing action accuracy from 88.8% to 96.3% and full maze accuracy from 37.4% to 61.6%. Although variability increases at the longest path lengths, and hence the greater uncertainty the full maze accuracy, the CMM performs markedly better at longer path lengths (fig. 4, bottom), with the strongest single run achieving 90.7% full maze accuracy, with per-seed results are provided in table 9. Its gains over the ablations further highlight the benefit of combining the CTM’s temporal processing with an expressive memory Transformer. Furthermore, the CMM preserves the CTM’s path-following attention behavior (fig. 4, top), with the CMM’s cross attention scanning a path through the maze while predicting the correct sequence of actions.

## 4.3 In-Context Learning: Few-Shot Regression

Few-shot learning requires a model to incorporate new examples while retaining previous information, making it a natural setting for evaluating short- and long-term memory [35]. We evaluate this capability using a few-shot regression task commonly used in meta-learning [51, 52]. In each episode a new target mapping $f : [ - 1 , 1 ] ^ { n } \to \mathbb { R }$ is sampled from a family of linear functions. The model is presented with K noisy support pairs $\left( \mathbf { x } _ { t } , f ( \mathbf { x } _ { t } ) + \epsilon _ { t } \right)$ with $\epsilon _ { t } \sim \mathcal { N } ( 0 , \sigma ^ { 2 } )$ , followed by query inputs whose outputs it must predict. Since the target function changes between episodes, the model cannot memorize any single mapping during training and must instead learn an in-context inference procedure. We set the input dimension to n = 12 and report results for $K \in \{ 1 0 , 2 0 \}$ . Full details are given in section D.5.

Table 2: Few-shot regression accuracy with K=10, 20 support examples. CMM achieves the highest accuracy in both settings.
<table><tr><td>ARCHITECTURE</td><td>K=20</td><td>K=10</td></tr><tr><td>TRANSFORMER</td><td>29.75±1.24</td><td>9.09±0.08</td></tr><tr><td>RNN</td><td>5.39±0.13</td><td>4.81±0.06</td></tr><tr><td>LSTM</td><td>9.72±0.42</td><td>6.40±0.05</td></tr><tr><td>FS-LSTM</td><td>11.02±0.10</td><td>6.53±0.07</td></tr><tr><td>GDN</td><td>10.65±0.94</td><td>6.68±0.06</td></tr><tr><td>DNC</td><td>9.59±0.29</td><td>6.34±0.06</td></tr><tr><td>RMC</td><td>19.53±7.93</td><td>7.94±0.30</td></tr><tr><td>LSAM</td><td>29.82±0.18</td><td>8.99±0.08</td></tr><tr><td>NAM-TM</td><td>27.65±0.11</td><td>9.34±0.10</td></tr><tr><td>RMT</td><td>30.46±0.41</td><td>9.81±0.08</td></tr><tr><td>CTM</td><td>8.78±0.11</td><td>4.85±0.07</td></tr><tr><td>AB: CMM-LSTM</td><td>19.95±8.75</td><td>6.38±0.05</td></tr><tr><td>AB: CTM-EXT.</td><td>8.00±0.07</td><td>5.84±0.02</td></tr><tr><td>AB: CMM-STM</td><td>13.03±0.61</td><td>7.28±0.12</td></tr><tr><td>CMM (OURS)</td><td>30.63±0.24</td><td>9.86±0.07</td></tr></table>

Test query accuracies across both support-set sizes are shown in table 2. A query prediction is correct when its absolute error is below 0.05 on the standardized target scale. The CMM

achieves the highest mean accuracy in both settings. At K=20, it improves accuracy over the CTM from 8.78% to 30.63% on linear functions. The CMM also achieves higher mean accuracies than all three architectural ablations, supporting the benefit of combining the CTM’s sliding-window STM with a distinct persistent LTM.

## 4.4 Image Classification: ImageNet-100

Following Darlow et al. [17], we use image classification to examine how models gather visual information over successive computation steps, rather than to pursue state-of-the-art classification accuracy. We trained models with 1M-parameter recurrent heads using the first two blocks of a frozen pre-trained ViT as the backbone [53], with patch features serving as keys and values for cross-attention from synchronization-based queries. Results are shown in table 3 and fig. 5. We find that the CMM is among the best performing architectures, while exhibiting interpretable attention patterns that shift between regions. Experimental details are provided in section D.6.

## 4.5 Analysis: Task-Dependent Memory Usage

To gain insight into how the CMM leverages its STM and LTM, we visualize the Transformer’s attention for various tasks, shown in fig. 6. On copy, STM queries attend predominantly to LTM. On sort, LTM attends to both stores during input, while STM increasingly attends to itself during output. On maze-solving tasks, STM queries attend entirely to the zero-valued sink, suppressing reads from both STM and LTM. This suggests that CMM’s gains arise from the expressivity of the memory Transformers’ feed-forward networks. On ImageNet-100, STM queries read from LTM but increasingly attend to the sink at later iterations. These patterns highlight the CMM’s flexibility to draw on both stores or emphasize one according to the task. By augmenting the CTM’s short-term temporal processing with a separately accessible persistent store, the CMM can leverage long-term memory when useful while retaining the ability to rely on short-term computation.

## 5 Discussion and Conclusion

The Continuous Memory Machine combines the expressive short-term processing of the CTM with the persistent long-term memory of MANNs. The CMM’s two matrix-valued recurrent states are jointly processed by a Transformer, acting as an expressive bidirectional read–write mechanism.

Table 3: ImageNet-100 accuracy. The CMM is among the top performing baselines.
<table><tr><td>ARCHITECTURE</td><td>AcC. (%)</td></tr><tr><td>TRANSFORMER</td><td>89.5±0.2</td></tr><tr><td>RNN</td><td>89.6±0.2</td></tr><tr><td>LSTM</td><td>89.7±0.1</td></tr><tr><td>FS-LSTM</td><td>89.5±0.1</td></tr><tr><td>GDN</td><td>63.3±41.3</td></tr><tr><td>DNC</td><td>89.7±0.2</td></tr><tr><td>RMC LSAM</td><td>75.9±9.0 59.2±49.0</td></tr><tr><td>NAM-TM</td><td>89.7±0.1</td></tr><tr><td>RMT</td><td>89.5±0.1</td></tr><tr><td>CTM</td><td>89.4±0.3</td></tr><tr><td>AB: CMM-LSTM</td><td>89.3±0.1</td></tr><tr><td>AB: CTM-EXT.</td><td>89.7±0.0</td></tr><tr><td>AB: CMM-STM</td><td>89.6±0.1</td></tr><tr><td></td><td></td></tr><tr><td>CMM (OURS)</td><td>89.7±0.2</td></tr></table>

![](images/21e691ab8463337fcd8016fa594f1d9838b11d557237bf749579d2d24bfe73ad.jpg)

![](images/c85e893ad0b318768b842b854f13da4208fa098a9a70117d98c5f7baeb3399b2.jpg)

![](images/9d79b9913f120307af54efb4cb1cc84fa55e7ca7ed05bdf744086f01777cef52.jpg)

![](images/ee8f395c8b4d484759f5564bb46919c9f429ecad29ed43c8e8f5b8380656c103.jpg)  
Figure 5: CMM attention exhibits the CTM’s “looking around” behavior [17].

![](images/9667bf03487419842f3bc4e902a5b10b5a0d3dfd3ab43b4e2f1a43f4527a0468.jpg)  
Figure 6: Memory attention across tasks. Each pair shows LTM queries (left) and STM queries (right), with attention to the sink, LTM, and STM tokens. CMM learns task-specific memory strategies.

The CMM outperforms a suite of baselines, with improved performance in algorithmic, in-context learning and maze-solving tasks. The CMM improves upon the generalization capabilities of prior MANNs, while maintaining the interpretable attention patterns of the CTM. Ablations support the benefits of utilizing the CTM’s short-term neuron-level processing, with the LSTM variant performing markedly worse throughout. Furthermore, analyzing the attention patterns of the CMM’s memory Transformer highlights the flexibility of the CMM, with the model relying on LTM for algorithmic tasks and in-context learning, yet bypassing it when not required.

The benefits of the CMM come with additional costs and limitations. First, the CMM introduces additional hyperparameters, namely the LTM size $M _ { L }$ and Transformer depth J. The joint memory update also increases computational and memory requirements, with training times and resource measurements reported in section C. A further limitation, shared with nonlinear RNNs more broadly, is that recurrence restricts training parallelism compared with linear recurrent architectures such as the GDN. Recent methods for accelerating nonlinear RNN training therefore offer a promising direction for future work [54]. Beyond accelerating the recurrence itself, future work could also improve the efficiency of the CMM’s memory update. In the current architecture, the memory Transformer is applied at every recurrent step, even when attention is routed primarily to the sink token and little cross-memory interaction is required. This suggests that updating the memory selectively, or only once every t steps, could reduce the computational cost without requiring a memory update at every iteration.

More broadly, the CMM serves as an example of how principles drawn from biological memory can inspire neural architectures with capabilities beyond those exhibited by existing recurrent models, combining flexible short-term computation with persistent storage to improve both algorithmic generalization and recurrent reasoning.

## AI use statement

In this work we used generative AI tools for polishing the writing. Furthermore, we used AI as a coding tool for building out some of the baseline architectures based on the original papers and their corresponding codebases. One author has tested the correctness of the code. For example, when building out the GDN baseline [10], we verified that the introduction of negative eigenvalues [49] allows the model to solve the parity task. We have not used AI for research ideation, the main writing of the text, performing the literature review, etc. We take responsibility for the final content of this work, including text, claims or artifacts produced with the aid of generative AI.

## Reproducibility statement

To ensure reproducibility of this work, we release all code at https://github.com/SakanaAI/ continuous-memory-machines. The repository contains everything needed to reproduce the main results, including the baselines, for all tasks. Instructions on how to run each task are described in the README.md of the codebase. A single NVIDIA H100 was used for each experiment in the paper, and is sufficient to reproduce the results.

## References

[1] Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017.

[2] Grégoire Delétang, Anian Ruoss, Jordi Grau-Moya, Tim Genewein, Li Kevin Wenliang, Elliot Catt, Chris Cundy, Marcus Hutter, Shane Legg, Joel Veness, et al. Neural networks and the chomsky hierarchy. arXiv preprint arXiv:2207.02098, 2022.

[3] Sepp Hochreiter and Jürgen Schmidhuber. Long short-term memory. Neural computation, 9(8): 1735–1780, 1997.

[4] Junyoung Chung, Caglar Gulcehre, KyungHyun Cho, and Yoshua Bengio. Empirical evaluation of gated recurrent neural networks on sequence modeling. arXiv preprint arXiv:1412.3555, 2014.

[5] Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014.

[6] J. Schmidhuber. Neural sequence chunkers. Technical Report FKI-148-91, Institut für Informatik, Technische Universität München, April 1991.

[7] Jan Koutnik, Klaus Greff, Faustino Gomez, and Juergen Schmidhuber. A clockwork rnn. In International conference on machine learning, pages 1863–1871. PMLR, 2014.

[8] Jürgen Schmidhuber. Learning to control fast-weight memories: An alternative to dynamic recurrent networks. Neural Computation, 4(1):131–139, 1992.

[9] Imanol Schlag, Kazuki Irie, and Jürgen Schmidhuber. Linear transformers are secretly fast weight programmers. In International conference on machine learning, pages 9355–9366. PMLR, 2021.

[10] Songlin Yang, Jan Kautz, and Ali Hatamizadeh. Gated delta networks: Improving mamba2 with delta rule. arXiv preprint arXiv:2412.06464, 2024.

[11] Jason Weston, Sumit Chopra, and Antoine Bordes. Memory networks. arXiv preprint arXiv:1410.3916, 2014.

[12] Zihang Dai, Zhilin Yang, Yiming Yang, Jaime G Carbonell, Quoc Le, and Ruslan Salakhutdinov. Transformer-xl: Attentive language models beyond a fixed-length context. In Proceedings ofthe 57th annual meeting ofthe associationfor computational linguistics, pages 2978–2988, 2019.

[13] Aydar Bulatov, Yury Kuratov, and Mikhail Burtsev. Recurrent memory transformer. Advances in neural information processing systems, 35:11079–11091, 2022.

[14] James L McClelland, Bruce L McNaughton, and Randall C O’Reilly. Why there are complementary learning systems in the hippocampus and neocortex: insights from the successes and failures of connectionist models of learning and memory. Psychological review, 102(3):419, 1995.

[15] Larry R Squire. Memory systems of the brain: a brief history and current perspective. Neurobiology oflearning and memory, 82(3):171–177, 2004.

[16] Dharshan Kumaran, Demis Hassabis, and James L McClelland. What learning systems do intelligent agents need? complementary learning systems theory updated. Trends in cognitive sciences, 20(7):512–534, 2016.

[17] Luke Darlow, Ciaran Regan, Sebastian Risi, Jeffrey Seely, and Llion Jones. Continuous thought machines. In Advances in Neural Information Processing Systems, 2025.

[18] Anthony J Robinson and Frank Fallside. The utility driven dynamic error propagation network, volume 11. University of Cambridge Department of Engineering Cambridge, 1987.

[19] Paul J Werbos. Generalization of backpropagation with application to a recurrent gas market model. Neural networks, 1(4):339–356, 1988.

[20] Ronald J. Williams. Complexity of exact gradient computation algorithms for recurrent neural networks. Technical Report NU-CCS-89-27, Northeastern University, College of Computer Science, Boston, 1989.

[21] Jeffrey L Elman. Finding structure in time. Cognitive science, 14(2):179–211, 1990.

[22] Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, Jürgen Schmidhuber, et al. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.

[23] Jürgen Schmidhuber. Learning complex, extended sequences using the principle of history compression. Neural computation, 4(2):234–242, 1992.

[24] Junyoung Chung, Sungjin Ahn, and Yoshua Bengio. Hierarchical multiscale recurrent neural networks. arXiv preprint arXiv:1609.01704, 2016.

[25] Asier Mujika, Florian Meier, and Angelika Steger. Fast-slow recurrent neural networks. Advances in Neural Information Processing Systems, 30, 2017.

[26] Geoffrey E Hinton and David C Plaut. Using fast weights to deblur old memories. In Proceedings of the ninth annual conference of the Cognitive Science Society, pages 177–186, 1987.

[27] Kazuki Irie and Samuel J Gershman. Fast weight programming and linear transformers: from machine learning to neurobiology. arXiv preprint arXiv:2508.08435, 2025.

[28] Albert Gu, Karan Goel, and Christopher Ré. Efficiently modeling long sequences with structured state spaces. arXiv preprint arXiv:2111.00396, 2021.

[29] Jimmy TH Smith, Andrew Warrington, and Scott W Linderman. Simplified state space layers for sequence modeling. arXiv preprint arXiv:2208.04933, 2022.

[30] Albert Gu and Tri Dao. Mamba: Linear-time sequence modeling with selective state spaces. arXiv preprint arXiv:2312.00752, 2023.

[31] Maximilian Beck, Korbinian Pöppel, Markus Spanring, Andreas Auer, Oleksandra Prudnikova, Michael Kopp, Günter Klambauer, Johannes Brandstetter, and Sepp Hochreiter. xlstm: Extended long short-term memory. Advances in Neural Information Processing Systems, 37:107547– 107603, 2024.

[32] Bernard Widrow and Marcian E Hoff. Adaptive switching circuits (in ire wescon convention record 1960). SPIE MILESTONE SERIES MS, 96:105–105, 1994.

[33] Sainbayar Sukhbaatar, Jason Weston, Rob Fergus, et al. End-to-end memory networks. Advances in neural information processing systems, 28, 2015.

[34] Armand Joulin and Tomas Mikolov. Inferring algorithmic patterns with stack-augmented recurrent nets. Advances in neural information processing systems, 28, 2015.

[35] Adam Santoro, Sergey Bartunov, Matthew Botvinick, Daan Wierstra, and Timothy Lillicrap. Meta-learning with memory-augmented neural networks. In International conference on machine learning, pages 1842–1850. PMLR, 2016.

[36] Yan Wu, Greg Wayne, Alex Graves, and Timothy Lillicrap. The kanerva machine: A generative distributed memory. arXiv preprint arXiv:1804.01756, 2018.

[37] Alex Graves, Greg Wayne, Malcolm Reynolds, Tim Harley, Ivo Danihelka, Agnieszka Grabska-Barwinska, Sergio Gómez Colmenarejo, Edward Grefenstette, Tiago Ramalho, John Agapiou,´ et al. Hybrid computing using a neural network with dynamic external memory. Nature, 538 (7626):471–476, 2016.

[38] Adam Santoro, Ryan Faulkner, David Raposo, Jack Rae, Mike Chrzanowski, Theophane Weber, Daan Wierstra, Oriol Vinyals, Razvan Pascanu, and Timothy Lillicrap. Relational recurrent neural networks. Advances in neural information processing systems, 31, 2018.

[39] Hyoungwook Nam and Seung Byum Seo. Neural attention memory. arXiv preprint arXiv:2302.09422, 2023.

[40] DeLesley Hutchins, Imanol Schlag, Yuhuai Wu, Ethan Dyer, and Behnam Neyshabur. Blockrecurrent transformers. Advances in neural information processing systems, 35:33248–33261, 2022.

[41] Aniket Didolkar, Kshitij Gupta, Anirudh Goyal, Nitesh Bharadwaj Gundavarapu, Alex M Lamb, Nan Rosemary Ke, and Yoshua Bengio. Temporal latent bottleneck: Synthesis of fast and slow processing mechanisms in sequence learning. Advances in Neural Information Processing Systems, 35:10505–10520, 2022.

[42] Mark G Stokes. ‘activity-silent’working memory in prefrontal cortex: a dynamic coding framework. Trends in cognitive sciences, 19(7):394–405, 2015.

[43] Joao Barbosa, Heike Stein, Rebecca L Martinez, Adrià Galan-Gadea, Sihai Li, Josep Dalmau, Kirsten CS Adam, Josep Valls-Solé, Christos Constantinidis, and Albert Compte. Interplay between persistent activity and activity-silent dynamics in the prefrontal cortex underlies serial biases in working memory. Nature neuroscience, 23(8):1016–1024, 2020.

[44] Wolf Singer. Neuronal synchrony: a versatile code for the definition of relations? Neuron, 24 (1):49–65, 1999.

[45] Pascal Fries. A mechanism for cognitive dynamics: neuronal communication through neuronal coherence. Trends in cognitive sciences, 9(10):474–480, 2005.

[46] Takeru Miyato, Sindy Löwe, Andreas Geiger, and Max Welling. Artificial kuramoto oscillatory neurons. In International Conference on Learning Representations, volume 2025, pages 44278– 44322, 2025.

[47] Guangxuan Xiao, Yuandong Tian, Beidi Chen, Song Han, and Mike Lewis. Efficient streaming language models with attention sinks. arXiv preprint arXiv:2309.17453, 2023.

[48] Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.

[49] Riccardo Grazzi, Julien Siems, Arber Zela, Jörg Franke, Frank Hutter, et al. Unlocking statetracking in linear rnns through negative eigenvalues. In International Conference on Learning Representations, volume 2025, pages 36565–36597, 2025.

[50] Michael Hahn. Theoretical limitations of self-attention in neural sequence models. Transactions ofthe Associationfor Computational Linguistics, 8:156–171, 2020.

[51] Chelsea Finn, Pieter Abbeel, and Sergey Levine. Model-agnostic meta-learning for fast adaptation of deep networks. In International conference on machine learning, pages 1126–1135. PMLR, 2017.

[52] Yu Duan, Zhongfan Jia, Qian Li, Yi Zhong, and Kaisheng Ma. Hebbian and gradient-based plasticity enables robust memory and rapid learning in rnns. arXiv preprint arXiv:2302.03235, 2023.

[53] Feng Wang, Sucheng Ren, Tiezheng Zhang, Predrag Neskovic, Anand Bhattad, Cihang Xie, and Alan Yuille. Vit-5: Vision transformers for the mid-2020s. arXiv preprint arXiv:2602.08071, 2026.

[54] Federico Danieli, Pau Rodriguez, Miguel Sarabia, Xavier Suau, and Luca Zappella. Pararnn: Unlocking parallel training of nonlinear rnns for large language models. In International Conference on Learning Representations, volume 2026, pages 53548–53579, 2026.

[55] Ruibin Xiong, Yunchang Yang, Di He, Kai Zheng, Shuxin Zheng, Chen Xing, Huishuai Zhang, Yanyan Lan, Liwei Wang, and Tieyan Liu. On layer normalization in the transformer architecture. In International conference on machine learning, pages 10524–10533. PMLR, 2020.

[56] Yann N Dauphin, Angela Fan, Michael Auli, and David Grangier. Language modeling with gated convolutional networks. In International conference on machine learning, pages 933–941. PMLR, 2017.

[57] Biao Zhang and Rico Sennrich. Root mean square layer normalization. Advances in neural information processing systems, 32, 2019.

[58] Diederik P Kingma and Jimmy Ba. Adam: A method for stochastic optimization. arXiv preprint arXiv:1412.6980, 2014.

[59] Anders Krogh and John Hertz. A simple weight decay can improve generalization. Advances in neural information processing systems, 4, 1991.

## A Architecture Details

## A.1 CMM Details

This section provides the full architectural specification of the CMM. For notational clarity in this section, we treat sequences as having tokens along the leading dimension, which is the transpose of the convention used in the main text.

Learnable Initialization. The LTM is initialized with a learned matrix $B ^ { 0 } \in \mathbb { R } ^ { M _ { L } \times d }$ , drawn from $\mathcal { N } ( 0 , 0 . 0 1 ^ { 2 } )$ ). Unlike the CTM, whose initial activated state and STM are both learned parameters, the CMM fixes both to zero buffers. The learned $B ^ { 0 }$ is the only trained initial state.

Sink Token. A learned vector $s \in \mathbb { R } ^ { d }$ , initialized to zero, is prepended to the concatenated sequence (eq. (4)). Following Xiao et al. [47], the sink absorbs uninformative attention mass, with its value projection zeroed at every block so that attention weight assigned to it contributes nothing to the output.

Positional Embeddings. A learned matrix $P \in \mathbb { R } ^ { ( 1 + M _ { L } + M _ { S } ) \times d }$ , initialized from $\mathcal { N } ( 0 , 0 . 0 2 ^ { 2 } )$ encodes the slot identity (sink, LTM position, or STM position) of each token. The same $\dot { P }$ is shared across all J Transformer blocks.

Transformer Block. Each of the J blocks is a pre-norm residual block [55] with multi-head self-attention and a feedforward network (FFN). Given an input sequence $\boldsymbol { X } ^ { \hat { } } \in \mathbb { R } ^ { S \times d }$ (where $S = 1 + M _ { L } + M _ { S } )$ and positional embeddings P, block j computes:

$$
{ \bar { X } } = \operatorname { L a y e r N o r m } ( X ) ,\tag{7}
$$

$$
Q = W _ { Q } ( \bar { X } + P ) , \quad K = W _ { K } ( \bar { X } + P ) , \quad V = W _ { V } ( \bar { X } ) ,\tag{8}
$$

$$
V _ { 0 }  \mathbf { 0 } ,\tag{9}
$$

$$
X ^ { \prime } = X + \gamma _ { j } \mathrm { D r o p o u t } \Big ( W _ { O } \mathrm { D r o p o u t } \Big ( \mathrm { s o f t m a x } \Big ( \frac { Q K ^ { \top } } { \sqrt { d _ { h } } } \Big ) \Big ) V \Big ) ,\tag{10}
$$

$$
X ^ { \prime \prime } = X ^ { \prime } + \gamma _ { j } \operatorname { D r o p o u t } \Bigl ( \mathrm { F F N } \bigl ( \mathrm { L a y e r N o r m } ( X ^ { \prime } ) \bigr ) \Bigr ) ,\tag{11}
$$

where $W _ { Q } , W _ { K } , W _ { V } , W _ { O } \in \mathbb { R } ^ { d \times d }$ are bias-free linear projections split into H heads, $d _ { h } = d / H$ is the per-head dimension, and

$$
\mathrm { F F N } ( x ) = W _ { 2 } \mathrm { S i L U } ( W _ { 1 } x ) , \quad W _ { 1 } , W _ { 2 } \in \mathbb { R } ^ { d \times d } ,\tag{12}
$$

also bias-free. Two design choices merit emphasis. First, the positional embeddings are added to the queries and keys but not to the values (eq. (8)), so that position influences which tokens attend to one another but does not alter the content that is read. Second, the value vector of the sink token is set to zero at every block $( \mathrm { e q . } ( 9 ) )$ , allowing the model to discard uninformative attention mass rather than distributing it across content-bearing tokens.

Residual Gate. For Maze and Parity, each block has its own unconstrained learned scalar $\gamma _ { j }$ initialized to zero and shared by its attention and FFN residual branches. Each block therefore starts as the identity map. For all other CMM experiments, $\gamma _ { j } = 1$ is fixed, giving ungated residual updates.

Output Processing. After the final Transformer block, the output sequence is sliced as in eq. (5). Layer normalization is applied to the updated LTM tokens before they are carried forward to the next timestep:

$$
\tilde { B } ^ { t } \gets \mathrm { L a y e r N o r m } ( \tilde { B } ^ { t } ) .\tag{13}
$$

The updated STM tokens are passed to the neuron-level models.

Neuron-Level Models. The NLMs follow the same SuperLinear-GLU structure as the CTM (eq. (2)), with the addition of a LayerNorm applied to the activated state after the GLU [56]:

$$
z ^ { t + 1 } = \mathrm { L a y e r N o r m } \Bigl ( \mathrm { G L U } \bigl ( W \cdot \tilde { A } ^ { t } \bigr ) \Bigr ) ,\tag{14}
$$

where $W \in \mathbb { R } ^ { M _ { S } \times 2 \times d }$ is the per-neuron weight tensor. The SuperLinear layer applies an independent linear map to each neuron’s $M _ { S }$ -length history, producing two outputs per neuron that are then halved by the GLU gate.

## A.2 Architectural Comparison

We provide a more detailed overview of the ablation described in section 4. Side-by-side diagrams of the full architectures are shown in fig. 7, and a closer comparison of the memory-update modules is shown in fig. 8. The CMM and CMM-LSTM share an identical LTM and memory-update Transformer; they differ only in how the short-term state is represented. The CMM passes the full sliding window of $M _ { S }$ recent pre-activations to the Transformer, whereas CMM-LSTM passes the LSTM’s single compressed hidden vector. Any performance gap between the two therefore reflects the contribution of the CTM’s distributed short-term representation and per-neuron processing, rather than the LTM or Transformer alone.

## A.3 Synchronization Normalization

The original CTM [17] normalizes the synchronization matrix by dividing by the square root of the sum of exponential weights:

$$
S _ { i j } ^ { t + 1 } = \frac { \sum _ { \tau = 1 } ^ { t + 1 } \mathrm { e } ^ { - r _ { i j } ( t - \tau ) } z _ { i } ^ { \tau } z _ { j } ^ { \tau } } { \left( \sum _ { \tau = 1 } ^ { t + 1 } \mathrm { e } ^ { - r _ { i j } ( t - \tau ) } \right) ^ { 1 / 2 } } .\tag{15}
$$

In our implementation, we remove the square root from the denominator. The square root causes the numerator and denominator to grow at different rates with t, so the magnitude of $S _ { i j } ^ { t }$ becomes implicitly tied to the number of internal ticks. Dividing by the linear sum of weights instead yields a properly normalized exponential moving average whose expected magnitude is invariant to $t ,$ allowing the model to generalize to thinking horizons beyond those seen during training.

We compute synchronization for d neuron pairs, sampled uniformly without replacement from the $d ^ { 2 }$ ordered pairs at initialization and then held fixed. After dividing by the linear sum of weights, we add $1 0 ^ { - 8 }$ to each component and apply RMSNorm [57] with learned elementwise scaling to obtain the d-dimensional synchronization representation.

![](images/2210f366fb17a16d2fa8c96af7f59a53026518b0f6b2a03db84ddcfec881f309.jpg)  
(a) Continuous Thought Machine (CTM).

![](images/ad013b4a1a84c4448832bcea2ae8271db133149e31d9f0b86282dfe49f880fb7.jpg)  
(b) Continuous Memory Machine (CMM).

![](images/fbada1767a326015c1a85e84e6412e557d305a931e20310402bd1eaf8d8ae860.jpg)  
(c) CMM-LSTM (LSTM substrate).  
Figure 7: Comparison of the Continuous Thought Machine (top), the Continuous Memory Machine (middle), and CMM-LSTM (bottom). The CTM maintains a sliding short-term memory (STM) of recent pre-activations, which is processed by independent neuron-level models, while the synchroniza tion module computes time-decayed outer products of activation histories that are projected to output logits. The CMM extends the CTM with a persistent long-term memory (LTM) that is jointly updated with the STM through the memory-update Transformer before the neuron-level models are applied. CMM-LSTM replaces the CTM backbone with a LayerNorm LSTM while retaining the same LTM and memory-update Transformer. Consequently, the Transformer receives a single hidden-state token instead of the CTM’s sliding window of $M _ { S }$ STM tokens, and there are no neuron-level models or synchronization representation. Comparing the CMM and CMM-LSTM isolates the contribution of the CTM’s distributed short-term representation and per-neuron processing.

## B Additional Ablations

## B.1 Memory capacity and update depth

We evaluate how Priority Sort performance changes with long-term memory capacity and memory update depth. For the capacity sweep, we reduce the STM length to $M _ { S } { = } 5$ to limit short-term storage, fix the memory update Transformer depth at $J { = } 2 ,$ and vary $\bar { M } _ { L } \in \{ 2 , 4 , 8 , 1 6 , 3 2 , 6 4 , 1 2 8 \}$ . For the depth sweep, we use the standard Priority Sort configuration with $\dot { M } _ { S } = 2 0$ and $M _ { L } { = } 1 2 8$ , and vary $J \bar { \in } \{ 1 , 2 , \bar { 4 } , 8 \}$ . All runs use the same 500K parameter budget and training setup as the main Priority Sort experiment. As shown in fig. 9, increasing LTM capacity broadly improves performance when STM is restricted, while two to four Transformer layers perform best in the depth sweep.

![](images/194366581772e8a461ee06ea4bf939477cc957d3e0ebda10156307e14364b534.jpg)  
(a) CMM memory update

![](images/de9da50b417ccc0712b6b854e43d2c085917ddc7f7954809bbc716ed5b0dc4cf.jpg)  
(b) CMM-LSTM memory update  
Figure 8: Memory-update modules for the CMM (a) and CMM-LSTM (b). Both apply a Transformer over a sequence consisting of a sink token, the LTM tokens, and STM tokens, with learned positional embeddings. The CMM (a) attends jointly over $M _ { S }$ STM tokens drawn from the sliding window of recent pre-activations, giving the Transformer a multi-token view of recent activity. CMM-LSTM (b) replaces the sliding window with the LSTM’s single hidden vector, so only one STM token is passed to the Transformer. Comparing the two isolates the effect of the CTM’s distributed short-term representation on the memory update.

## C Compute Resources

Experiments were run on a single NVIDIA H100 GPU. Mean training-loop wall times per CMM seed are listed in table 4. These include periodic evaluation and checkpointing, and exclude model setup, queueing, and downtime between resumed jobs. For Maze, we sum the original 4,500-epoch run and its 4,500-epoch continuation; both use 50 internal iterations, giving 9,000 epochs in total. Means are taken over three seeds, so total GPU-hours are three times the per-seed values, even when seeds run concurrently. Baseline architectures were trained under matched conditions. The full research project required additional compute beyond the reported experiments, including preliminary architecture exploration, hyperparameter tuning, and ablations that are not included in the final paper.

![](images/8eb3eb007ae18c86368124ec1c1a748b6cb204fc9d9d70a4e75fe16839b7a589.jpg)  
(a) LTM token sweep.

![](images/737f62cdda36012fcd2e706b92eb7d0bc542e3a107ed32d1d23d74a39e8d9995.jpg)  
(b) Transformer depth sweep.  
Figure 9: Additional CMM ablations on Priority Sort. Points show mean test binary cross-entropy. Error bars show one standard deviation over two seeds. Lower is better.

To characterize computational cost, we benchmark all architectures on batches of 64 complete Copy-100 sequences, each containing 100 input and 100 output steps. Hidden widths are selected independently under a 500K parameter budget, including a linear 12 → 64 input projection and a 10-dimensional output projection. Measurements use eager float32 execution on a single NVIDIA H100 GPU, with five warmup repetitions followed by twenty measured repetitions per timing. Each architecture runs in a fresh process. Forward timing disables autograd; forward+backward timing differentiates the sum of all outputs and excludes optimizer updates. Table 5 reports the mean latencies and memory use. The CMM’s joint STM–LTM update increases computation and activation memory relative to the RNN and LSTM.

Table 4: Mean training-loop wall time per CMM seed on a single NVIDIA H100 GPU.
<table><tr><td>Experiment</td><td>Epochs</td><td>Time per seed</td></tr><tr><td>Copy</td><td>1000</td><td>6.9 h</td></tr><tr><td>Associative Recall</td><td>1000</td><td>1.1 h</td></tr><tr><td>Sort</td><td>1000</td><td>3.0 h</td></tr><tr><td>Sort (length gen.)</td><td>3000</td><td>2.2 h</td></tr><tr><td>Parity</td><td>200</td><td>0.6 h</td></tr><tr><td>Few-Shot Regression</td><td>1000</td><td>1.3-1.8 h</td></tr><tr><td>ImageNet-100</td><td>25</td><td>0.9 h</td></tr><tr><td>Maze</td><td>9000</td><td>127.5 h</td></tr></table>

Table 5: GPU efficiency on complete Copy-100 sequences (batch size 64, sequence length 200). Times are amortized milliseconds per step, obtained by dividing full-sequence latency by 200; the Transformer processes the sequence in parallel and RMT uses ten-step segments. State is persistent memory per example (KiB); peak is allocated GPU memory for a complete forward+backward pass (MiB). The sequence Transformer has no recurrent state.
<table><tr><td>Architecture</td><td>Params</td><td>d Fwd/step</td><td></td><td>Fwd+Bwd/step</td><td>State</td><td>Peak</td></tr><tr><td>Transformer</td><td>479,278</td><td>100</td><td>0.011</td><td>0.032</td><td></td><td>639.4</td></tr><tr><td>RNN</td><td>499,170</td><td>668</td><td>0.066</td><td>0.246</td><td>2.61</td><td>201.4</td></tr><tr><td>LSTM</td><td>497,558</td><td>318</td><td>0.147</td><td>0.548</td><td>2.48</td><td>336.2</td></tr><tr><td>FS-LSTM</td><td>494,926</td><td>149</td><td>0.450</td><td>1.473</td><td>2.33</td><td>415.4</td></tr><tr><td>GDN</td><td>495,946</td><td>480</td><td>0.555</td><td>2.126</td><td>275.62</td><td>7,248.0</td></tr><tr><td>DNC</td><td>498,220</td><td>271</td><td>1.266</td><td>5.097</td><td>78.20</td><td>3,175.0</td></tr><tr><td>RMC</td><td>479,822</td><td>108</td><td>0.412</td><td>1.464</td><td>13.50</td><td>3,079.1</td></tr><tr><td>LSAM</td><td>496,122</td><td>372</td><td>0.219</td><td>0.976</td><td>136.59</td><td>3,644.1</td></tr><tr><td>NAM-TM</td><td>496,992</td><td>170</td><td>0.960</td><td>4.706</td><td>67.33</td><td>2,039.4</td></tr><tr><td>RMT</td><td>496,682</td><td>160</td><td>0.255</td><td>0.856</td><td>80.00</td><td>8,646.6</td></tr><tr><td>CTM</td><td>498,802</td><td>472</td><td>0.360</td><td>1.256</td><td>25.81</td><td>634.5</td></tr><tr><td>Ab: CMM-LSTM</td><td>491,054</td><td>204</td><td>0.529</td><td>1.793</td><td>27.09</td><td>4,587.7</td></tr><tr><td>Ab: CTM-Ext.</td><td>498,092</td><td>390</td><td>0.359</td><td>1.257</td><td>216.33</td><td>3,145.8</td></tr><tr><td>Ab: CMM-STM</td><td>495,862</td><td>212</td><td>0.850</td><td>2.755</td><td>117.59</td><td>21,325.3</td></tr><tr><td>CMM (Ours)</td><td>486,474</td><td>224</td><td>0.870</td><td>2.813</td><td>124.25</td><td>22,330.1</td></tr></table>

## D Experimental Details

Unless otherwise noted in a task-specific subsection, experiments use the AdamW optimizer [58] with a learning rate of $1 0 ^ { - 4 }$ and no learning rate scheduler. ImageNet-100 uses cosine annealing over its 25 training epochs. Weight decay [59] is set to 0 except for ImageNet-100 and Priority Sort length generalization, where it is 10<sup>−4</sup>. Models use parameter budgets of 500K by default, 1M for ImageNet-100 (excluding the frozen backbone), 10M for Maze, and 60K for Parity and

Priority Sort length generalization. For Maze and Parity, each CMM memory update Transformer block uses a learned scalar initialized to zero. The scalar scales both the attention and feed-forward residual branches. All other CMM experiments use ungated residual updates. Table 6 summarizes the remaining training hyperparameters for each task. Table 7 lists the CMM-specific architectural hyperparameters used for each task. Task-specific details follow in the subsequent sections. The error bars in figs. 2 and 3 show one standard deviation over three seeds.

Table 6: Per-task training hyperparameters.
<table><tr><td>Task</td><td>Loss</td><td>Num. Epochs</td><td>Batch Size</td><td>Grad. Clip</td></tr><tr><td>Copy</td><td>MSE</td><td>1000</td><td>128</td><td>5</td></tr><tr><td>Associative Recall</td><td>BCE</td><td>1000</td><td>32</td><td>50</td></tr><tr><td>Priority Sort</td><td>BCE</td><td>1000</td><td>32</td><td>50</td></tr><tr><td>Parity</td><td>BCE</td><td>200</td><td>64</td><td>5</td></tr><tr><td>Few-Shot Regression</td><td>MSE</td><td>1000</td><td>64</td><td>5</td></tr><tr><td>ImageNet-100</td><td>CE</td><td>25</td><td>64</td><td>1</td></tr><tr><td>Maze</td><td>Dual-tick  $\mathrm { C E } ^ { \dagger }$ </td><td>9000</td><td>1024</td><td>1</td></tr></table>

<sup>†</sup>Dual-tick selection with auto-curriculum masking over 5 action classes (see section D.7).

Table 7: CMM-specific hyperparameters for each task. $M _ { S } { \mathrm { : } }$ STM length (sliding window), $M _ { L } ;$ number of LTM tokens, J: Transformer layers in the memory update, $D \colon$ synapse depth.
<table><tr><td>Task</td><td> $M _ { S }$ </td><td> $M _ { L }$ </td><td> $J$ </td><td> $D$ </td></tr><tr><td>Copy</td><td>10</td><td>128</td><td>1</td><td>8</td></tr><tr><td>Associative Recall</td><td>10</td><td>32</td><td>1</td><td>8</td></tr><tr><td>Priority Sort</td><td>20</td><td>128</td><td>2</td><td>8</td></tr><tr><td>Parity</td><td>4</td><td>4</td><td>1</td><td>8</td></tr><tr><td>Few-Shot Regression</td><td>10</td><td>32</td><td>1</td><td>8</td></tr><tr><td>ImageNet-100</td><td>10</td><td>32</td><td>1</td><td>4</td></tr><tr><td>Maze</td><td>10</td><td>8</td><td>1</td><td>16</td></tr></table>

Transformer baseline. On Copy, Associative Recall, and Priority Sort, the Transformer uses four independently parameterized causal pre-normalization layers with hidden dimension 100, four attention heads, GELU feed-forward sublayers of dimension 384, fixed sinusoidal positional embeddings, a final layer normalization, and no dropout. The maximum context lengths are 256 for Copy, 32 for Associative Recall, and 64 for Priority Sort, in each case covering the complete input–output sequence. The respective models have 479,278, 478,554, and 479,012 trainable parameters and otherwise use the shared task-specific optimization settings in table 6. The smaller Transformer used for length generalization is specified in section D.3.

Recurrent Memory Transformer baseline. The RMT [13] uses two layers, four attention heads, relative positional attention, post-layer normalization, and ReLU feed-forward sublayers of width 2d. Memory tokens pass information between segments. We use the shared task-specific optimization settings and report final-epoch results over three seeds. Table 8 lists the resolved configurations. On ImageNet-100, each segment corresponds to one visual iteration, and accuracy is pooled over all 5,000 test images per seed.

Table 8: RMT configurations. Parameter counts exclude the frozen ImageNet backbone.
<table><tr><td>Task</td><td>d</td><td>Segment length</td><td>Memory tokens</td><td>Parameters</td></tr><tr><td>Copy</td><td>160</td><td>10</td><td>128</td><td>496,682</td></tr><tr><td>Associative Recall</td><td>160</td><td>10</td><td>32</td><td>480,358</td></tr><tr><td>Priority Sort</td><td>160</td><td>20</td><td>128</td><td>496,296</td></tr><tr><td>Priority Sort (length gen.)</td><td>52</td><td>4</td><td>22</td><td>54,752</td></tr><tr><td>Parity</td><td>56</td><td>4</td><td>4</td><td>59,569</td></tr><tr><td>Few-Shot Regression</td><td>160</td><td>10</td><td>32</td><td>496,801</td></tr><tr><td>ImageNet-100</td><td>152</td><td>1</td><td>32</td><td>997,876</td></tr></table>

## D.1 Copy Task

The model receives a sequence of $T { = } 1 0 0$ vectors of dimension $D { = } 1 0$ with entries drawn uniformly from $[ - 1 , 1 ]$ and must reproduce the sequence immediately after the input phase. The input at each step is $( r _ { t } , s _ { t } , q _ { t } )$ where $r _ { t }$ is the data vector (zero-padded outside the input phase), $s _ { t }$ is a binary sample flag active during the input phase, and $q _ { t }$ is a binary query flag active during the output phase. The total sequence length is $2 T \colon T$ input steps followed by $T$ query steps. A linear projection maps the $( D \small { + } 2 )$ -dimensional input to the model’s hidden dimension. The loss is mean squared error computed only over the query phase. The dataset consists of $6 { , } 4 0 0$ procedurally generated episodes per epoch.

## D.2 Associative Recall

Each trial presents N items, with N sampled uniformly from $\{ 2 , \ldots , 6 \}$ , each consisting of 3 binary vectors of 6 bits. The item sequence is zero-padded to 18 vector steps, followed by a delimiter, a threevector query item, a second delimiter, and three output slots. The query is sampled uniformly from the first $N - 1$ items, and the model must output its successor in the original sequence. Binary crossentropy is computed only over the three output vectors. The dataset consists of 6,400 procedurally generated episodes per epoch.

## D.3 Priority Sort

The model receives $N { = } 2 0$ input items, each consisting of a scalar priority drawn uniformly from $[ - 1 , 1 ]$ and an associated 8-bit binary value vector. After the input phase, the model must output the 16 value vectors corresponding to the 16 lowest-priority items, sorted in ascending order of priority. The loss is binary cross-entropy computed over the output positions. The dataset consists of $6 { , } 4 0 0$ procedurally generated episodes per epoch.

Length generalization. For fig. 3, we use $W = 4 , K = 4$ , and three training seeds per model. Each training batch contains a single input length, with lengths $N \in \{ 4 , \ldots , 1 6 \}$ balanced across the 13 batches of 64 episodes in each epoch. All models train for $3 { , } 0 0 0$ epochs under a 60K parameter budget, using AdamW with learning rate $1 0 ^ { - 3 }$ , weight decay $1 0 ^ { - 4 }$ , dropout 0.1, and gradient clipping at 50. We evaluate the final checkpoints on 832 freshly generated episodes at each $\mathbf { \bar { \boldsymbol { N } } } \in \{ 4 , 8 , \mathbf { \bar { 1 2 } } , \mathbf { \bar { 1 6 } } , 2 0 , 3 0 , 4 0 , 5 0 \}$ }, using the same evaluation examples across models and training seeds. Bit accuracy is computed over the four output value vectors; unbiased random bit predictions have expected accuracy 50%. The Transformer has two causal pre-normalization layers, hidden dimension $^ { 5 6 , }$ four attention heads, GELU feed-forward sublayers of dimension 128, fixed sinusoidal positional embeddings with a maximum sequence length of 2,048, and a final layer normalization. It has 59,068 trainable parameters. The RMT uses two layers with four attention heads, hidden dimension 52, ReLU feed-forward sublayers of width 104, four input steps per segment, and 22 recurrent memory tokens, for 54,752 trainable parameters.

## D.4 Parity

Each example contains independent uniformly sampled bits followed by a query token. The target is one if the number of set bits is odd and zero otherwise. Binary cross-entropy is computed only at the query step. All models use three seeds, a 60K parameter budget, zero dropout, and 200 training epochs with 8,192 generated examples per epoch. Training lengths vary from 1 to 40. We evaluate on 512 examples at each length in {40, 50, 100, 200, 500, 1000} and report the final epoch. The CMM uses $M _ { S } = 4$ and $M _ { L } = 4 ;$ the standard CTM uses $M _ { S } = 4$ , and CTM-Ext. uses $M _ { S } = 8$ to match the CMM’s total memory capacity.

## D.5 Few-Shot Regression

In each episode a new target linear function $f : [ - 1 , 1 ] ^ { n } \to \mathbb { R }$ with $n { = } 1 2$ is sampled. The model receives K support pairs $\bar { ( x _ { t } , f ( x _ { t } ) + \epsilon _ { t } ) }$ with $\epsilon _ { t } \sim \mathcal { N } ( 0 , 0 . 0 1 )$ sequentially, followed by 20 query inputs for which it must predict outputs. We evaluate with $K \in \{ 1 0 , 2 0 \}$ . Training minimizes MSE over the complete episode. Evaluation reports accuracy only on the 20 query predictions: the fraction satisfying $| \hat { y } _ { t } - y _ { t } | < 0 . 0 5$ , pooled over all test episodes. The clean targets $y _ { t }$ are standardized to zero mean and unit sample standard deviation within each episode before noise is added to the support-value inputs. Thus the threshold is 0.05 target standard deviations; support predictions do not enter the reported accuracy. At each step, the 12 input variables, observed-value channel, and support-mask bit are mapped to the model’s hidden dimension by a learned linear projection followed by a ReLU activation. The dataset consists of 10,000 episodes, with fresh functions sampled each epoch.

CTM-Ext. uses $\scriptstyle M _ { S } = 4 2$ , matching the CMM’s 10 STM slots plus 32 LTM tokens. CMM-STM uses the same 42-slot STM with one four-head memory-Transformer layer and no persistent LTM. Hidden width is selected independently under the shared 500K parameter budget, yielding 499,585 parameters for CTM-Ext. and 495,713 for CMM-STM, compared with 496,129 for the CMM. Both ablations use the shared optimization settings in table 6. We report the mean and one sample standard deviation of final epoch-1,000 test query accuracy across seeds 0, 1, and 2. Each seed is evaluated on 10,000 test episodes, giving 200,000 query predictions.

The Transformer baseline uses four independently parameterized causal pre-normalization layers with hidden dimension 104, four attention heads, and GELU feed-forward sublayers of dimension 384. It uses fixed sinusoidal positional embeddings, a maximum sequence length of 64, a final layer normalization, and no dropout. The causal mask restricts each position to the current and preceding items. A linear head maps each output representation to a scalar prediction. The model has 498,033 trainable parameters and otherwise uses the shared optimization settings in table 6.

## D.6 Image Classification (ImageNet-100)

We use the ImageNet-100 subset containing 100 classes with images resized to $2 2 4 \times 2 2 4$ . We extract $1 4 \times 1 4 = 1 9 6$ patch tokens of dimension 384 from the output of the penultimate Transformer block (block 11 of 12) of a frozen, pre-trained ViT-Small [53]. These patches serve as keys and values for cross-attention from synchronization-based queries, as in the original CTM [17]. The recurrent model processes the image over $I { = } 2 5$ internal iterations, producing a class prediction at each step. The loss is standard cross-entropy applied to the final prediction. The trainable model uses a 1M parameter budget, excluding the frozen backbone. We train for 25 epochs with cosine annealing from an initial learning rate of $1 \mathrm { { 0 } ^ { - 4 } }$ to zero, dropout of 0.2, and weight decay of $1 0 ^ { - 4 }$ . CTM-Ext. uses $M _ { S } = 4 2$ , matching the CMM’s 10 STM slots plus 32 LTM tokens, under the same 1M parameter budget. We report final-epoch accuracy pooled over all 5,000 test images per seed, with means and sample standard deviations over seeds 0, 1, and 2.

## D.7 Maze-Solving

We evaluate on the large maze dataset (39×39 grids) [17]. Each maze is encoded as a 3-channel image (walls, start, goal) and processed by a ResNet-34 backbone (first 2 blocks, no pre-training) producing $1 0 \times 1 0 = 1 0 0$ spatial tokens. The model recurrently processes these tokens over $I { = } 5 0$ internal iterations, outputting at each step a route of length 100 as a sequence of 5-class action predictions (up, down, left, right, wait).

The loss uses dual-tick selection: for each sample we compute the per-route-step cross-entropy at every internal tick, then average the loss at the minimum-CE tick and the most-certain tick (the tick with lowest mean output entropy over route steps). An auto-curriculum mask restricts the loss to the first c route steps beyond the model’s current longest fully-correct prefix, where c=25. This encourages the model to learn incrementally longer routes.

We target 9,000 epochs with an effective batch size of 1,024 and learning rate $1 0 ^ { - 3 }$ . NAM-TM uses microbatches of 256 with four gradient accumulation steps; the other reported architectures use batches of 1,024 without accumulation. All models use dropout 0.1. The CMM memory update Transformer uses one attention head. Table 1 reports the arithmetic mean and sample standard deviation across seeds 0, 1, and 2 of the final available current-model test accuracies, without EMA or selection of the best epoch during training. Predictions use each example’s most-certain interna iteration. Action accuracy measures the fraction of correct route actions; board (full-maze) accuracy requires all 100 actions, including wait padding, to match the target. The recorded test metrics average batch accuracies over the 5,000 test mazes.

Table 9 lists both test accuracies for each seed. NAM-TM and CMM-STM completed all three seeds at epoch 9,000. RMC seed 2 stopped after a gradient spike during epoch 5,076; we retain its last recorded test evaluation at epoch 5,075. Every other run included in the tables ends at epoch 9,000.

Incomplete baselines. RMT likewise exhibited unstable training and did not complete the 9,000- epoch schedule, so we omit its Maze results. For LSAM and GDN, the reference per-device batch size of 1,024 exceeded the available GPU memory. Smaller microbatches reduced memory usage, but matching the effective batch size required multiple sequential forward and backward passes per optimizer update. The resulting runtime made the full 9,000-epoch schedule impractical within our compute budget. We therefore omit LSAM and GDN from the Maze tables.

Table 9: Final available test action and board (full-maze) accuracies (%) for each training seed. Ending epochs are listed in the text; aggregate results appear in table 1.
<table><tr><td></td><td colspan="2">Seed 0</td><td colspan="2">Seed 1</td><td colspan="2">Seed 2</td></tr><tr><td>Architecture</td><td>Action</td><td>Board</td><td>Action</td><td>Board</td><td>Action</td><td>Board</td></tr><tr><td>Transformer</td><td>33.82</td><td>0.20</td><td>43.40</td><td>1.10</td><td>47.09</td><td>0.84</td></tr><tr><td>RNN</td><td>49.63</td><td>2.27</td><td>46.67</td><td>2.34</td><td>45.25</td><td>1.80</td></tr><tr><td>LSTM</td><td>39.67</td><td>1.06</td><td>52.66</td><td>2.51</td><td>59.46</td><td>4.47</td></tr><tr><td>FS-LSTM</td><td>59.53</td><td>4.10</td><td>61.23</td><td>4.58</td><td>43.48</td><td>1.08</td></tr><tr><td>DNC</td><td>81.44</td><td>12.75</td><td>81.00</td><td>12.04</td><td>85.76</td><td>16.70</td></tr><tr><td>RMC</td><td>25.26</td><td>0.00</td><td>25.10</td><td>0.00</td><td>20.59</td><td>0.00</td></tr><tr><td>NAM-TM</td><td>83.13</td><td>13.39</td><td>82.62</td><td>13.64</td><td>81.70</td><td>12.68</td></tr><tr><td>CTM</td><td>86.76</td><td>18.20</td><td>97.93</td><td>81.95</td><td>81.64</td><td>11.96</td></tr><tr><td>Ab: CMM-LSTM</td><td>46.64</td><td>1.57</td><td>54.04</td><td>3.52</td><td>38.27</td><td>1.24</td></tr><tr><td>Ab: CMM-STM</td><td>98.32</td><td>86.64</td><td>95.82</td><td>42.92</td><td>66.75</td><td>1.74</td></tr><tr><td>CMM (Ours)</td><td>97.90</td><td>71.28</td><td>91.93</td><td>22.84</td><td>99.08</td><td>90.73</td></tr></table>