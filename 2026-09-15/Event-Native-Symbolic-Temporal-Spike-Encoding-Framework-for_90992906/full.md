# Event-Native Symbolic-Temporal Spike Encoding Framework for Heterogeneous Cyber Streams

Dalton Diez , Peyton Andras , Max Shroyer , and James Ghawaly, Jr.

Louisiana State University

Baton Rouge, Louisiana, USA

{ddiez2, peyton.andras, mshroy1, jghawaly}@lsu.edu

Abstract—Spiking neural networks (SNNs) have shown promise for sparse, event-driven computation through stateful processing that is naturally compatible with low-power edge hardware. These properties align with cyber monitoring, where data arrives asynchronously, and malicious behavior often emerges through temporal patterns across event sequences. However, cyber streams are not composed solely of continuous numeric signals: their informative structure is also carried by categorical identifiers, irregular timing, and local behavioral context. Traditional rate- and population-based spike encodings are not naturally suited to these heterogeneous semantics, while conventional intrusion detection system (IDS) pipelines typically resolve the mismatch by converting raw events into flows, fixed aggregation windows, or dense tensors. Although useful for conventional classifiers, these transformations introduce bufering latency, obscure native temporal structure, and weaken the computational advantages of event-driven neuromorphic processing.

We introduce an event-native symbolic-temporal spike encoding framework that maps heterogeneous cyber events directly into sparse, spike-compatible inputs. By assigning explicit encoding roles to semantic identity, local frequency context, and inter-event timing, the framework preserves categorical semantics and temporal dynamics. We validate the approach on packet-level Network IDS and extend it to message-level CAN IDS, using both domains to evaluate whether the encoding exposes usable structure for recurrent SNNs operating directly on native event streams. Under edgeoriented, µCaspian-aligned hardware constraints, compact recurrent SNNs achieve strong anomaly detection performance, with an operational hybrid metric $\left( J _ { h y b r i d } \right)$ of 0.987 on Network IDS and 0.980 on CAN IDS. These results show that preserving native symbolic and temporal structure enables compact recurrent SNNs to perform efective attack detection using edgecompatible inference.

Index Terms—Spiking neural networks, event-native encoding, cyber intrusion detection, Controller Area Network, neuromorphic computing, asynchronous streams.

## I. Introduction

Spiking neural networks (SNNs) are naturally aligned with the asynchronous, event-driven characteristics of cybersecurity streams [1]. In domains like network monitoring or in-vehicle communication, data arrives as discrete events (packets or messages), and malicious activity often manifests as temporal patterns across sequences. However, despite this conceptual alignment, applying SNNs to native cyber event streams presents a major representational challenge.

The dificulty lies in the input interface. Unlike physical sensor signals, which are often regularly sampled and continuous-valued, cyber event fields are symbolic, heterogeneous, and irregularly timed. Attributes such as protocol type, destination port, service, and Controller Area Network (CAN) arbitration ID do not have natural numeric ordering or magnitude relationships. Treating a port number or protocol identifier as a continuous value therefore distorts its semantic meaning. Individual packets or messages are also weakly informative in isolation; their significance depends on local sequential context and relative timing across event arrivals.

To overcome this representation gap, conventional intrusion detection systems (IDS) rely on extensive preprocessing, such as flow construction, fixed-duration aggregation, or input aggregation windows [2]–[4]. While these methods extract useful numeric features, they transform the data from its native form, introducing latency and increasing resource overhead. Prior SNN-based IDS research typically inherits these limitations by using preengineered, static numeric feature vectors as the starting point for spike encoding [5]–[8], or by converting conventional deep neural networks into spiking equivalents via artificial-to-SNN pipelines [9]. This prevents end-toend event-native inference and shifts computation back toward preprocessing stages that reduce the latency and low-power advantages of sparse neuromorphic processing.

In this work, we propose an event-to-spike representation designed specifically for heterogeneous, asynchronous cyber streams. Our central claim is representational: cyber event streams can be made suitable for edge-based neuromorphic inference when each event is encoded according to its symbolic identity, local behavioral context, and interevent timing. Preserving this structure allows compact recurrent SNNs to accumulate evidence over time and classify attacks directly from native cyber streams. The contribution is not a new IDS model, but a symbolictemporal representation interface for mapping raw cyber events into sparse, spike-compatible channels at their exact moment of arrival. The encoder uses semantically defined bins for categorical attributes, online frequency counters for local context, and discretized inter-event intervals (∆t) for relative timing. This enables direct edge-compatible event-native classification without flow construction, fixedduration aggregation, or dense trafic tensors.

To demonstrate the generality of this representation, we instantiate it across two distinct domains: packetlevel Network IDS and message-level CAN IDS. We optimize the recurrent SNN topologies and parameters under edge-oriented µCaspian hardware constraints [10] using Evolutionary Optimization for Neuromorphic Systems (EONS) [11] as the optimization backend. We evaluate the representation with lightweight baselines on raw and encoded inputs, contextualizing the dificulty of eventnative detection and the value of the domain-informed encoded representation. We then show that SNNs can use this representation to retain context across native event arrivals and perform efective edge-compatible detection.

## II. Related Work

## A. SNN Input Encoding

Input encoding converts real-world data into spikes. Traditional encoding schemes (e.g., rate, temporal, and population coding) were designed for regularly sampled, continuous-valued signals where information is carried by physical magnitudes or precise timing [12], [13]. These approaches are poorly suited for cyber streams, where information is carried by symbolic attributes, irregular event spacing, and local temporal context rather than continuous magnitudes. For this reason, a direct rate-coded or population-coded baseline is not semantically well-defined for native packet or CAN fields. Ports, protocols, services, and arbitration IDs do not encode information through numeric magnitude, so mapping them directly to firing rates or population centers would impose artificial relationships that are not present in the underlying protocol semantics.

## B. Event Streams and Irregular Temporal Models

Outside neuromorphic computing, sequence models like DeepLog capture log sequences [14], while temporal point processes model irregular event arrivals [15]. Event-based vision pipelines also process asynchronous pixel streams using specialized spatial-temporal representations [16]. While these approaches capture temporal relationships, they rely on dense embeddings, self-attention, or large hidden states that are computationally expensive and incompatible with edge neuromorphic hardware.

## C. Machine Learning for Network IDS and CAN IDS

Conventional Network IDS and CAN IDS research heavily favors static, aggregated, or fixed-dimensional feature representations. Evaluations on network datasets typically use processed trafic features that summarize multiple packets over time [17]–[19]. Similarly, CAN IDS benchmarks frequently use fixed-dimensional representations of message-ID sequences or payload bit strings processed by dense neural networks [20], [21]. These approaches require bufering and aggregation, adding latency and preprocessing overhead that reduces their suitability for direct edge-based event processing.

## D. Spiking Neural Networks for Intrusion Detection

Prior spiking IDS implementations generally inherit these aggregation assumptions. Most spiking network intrusion detectors perform rate or temporal encoding on pre-extracted, high-level numeric features rather than raw packets [5]–[7]. For CAN security, researchers have proposed ANN-to-SNN conversion methods that map pretrained networks into spiking networks [9]. Because these networks operate on fixed numeric matrices, they do not process asynchronous cyber events directly and therefore cannot support event-native execution.

## E. Representation Gap

Despite domain alignments, there is a representational gap between asynchronous cyber streams and neuromorphic computation. Current SNN encoders cannot naturally represent the mixture of categorical symbols, irregular timings, and sequential contexts of packet and CAN trafic. Conversely, existing IDS pipelines recover context only after bufering or aggregation, discarding the timing and event boundaries that recurrent SNNs are designed to process. This motivates a representation layer that preserves cyber event semantics while enabling sparse recurrent SNN execution for direct, low-overhead edgebased monitoring.

## III. Symbolic-Temporal Encoding Framework

The core contribution of this work is a representation framework for asynchronous cyber streams that maps raw events into sparse, spike-compatible inputs at their moment of arrival. The encoder represents categorical cyber fields using operational semantics, maintains local context with online frequency counters, and captures timing explicitly, rather than implicitly through fixed-rate intervals.

## A. General Framework

Given an asynchronous event stream $\begin{array} { r l } { \mathcal { X } } & { { } = } \end{array}$ $\{ ( x _ { 1 } , t _ { 1 } ) , ( x _ { 2 } , t _ { 2 } ) , \ldots , ( x _ { T } , t _ { T } ) \}$ , each event $x _ { i }$ arriving at timestamp t<sub>i</sub> is mapped to a sparse binary spike vector $s _ { i } \in \{ 0 , 1 \} ^ { D }$ by concatenating K attribute groups with one temporal group:

$$
s _ { i } = \left[ g _ { 1 } ( x _ { i } ) \parallel g _ { 2 } ( x _ { i } ) \parallel \cdots \parallel g _ { K } ( x _ { i } ) \parallel h ( \Delta t _ { i } ) \right] \in \{ 0 , 1 \} ^ { D } ,\tag{1}
$$

where $\Delta t _ { i } = t _ { i } - t _ { i - 1 }$ is the inter-event arrival delay.

Each attribute group $g _ { k } ( x _ { i } )$ maps a specific categorical or numeric field to a one-hot subvector of size $B _ { k }$ . The temporal group $h ( \Delta t _ { i } )$ maps the inter-event delay to a one-hot subvector of size $B _ { \Delta t }$ . The total dimensionality of the input space is:

$$
D = \sum _ { k = 1 } ^ { K } B _ { k } + B _ { \Delta t } .\tag{2}
$$

Because each group activates exactly one bin per event, the resulting vector $s _ { i }$ contains exactly K+1 active elements, with the remaining $D - \left( K { + } 1 \right)$ elements set to zero. The encoding operation has a time complexity of $O ( K )$ per event, which is independent of the stream length. This mapping is illustrated in Fig. 1. This fully categorical formulation is used for Network IDS; Section VI extends it with rate-coded payload channels.

![](images/62b55702d2ffd21040deb99dda3e1a31d670afe43f2f981e6135f0e38b8db6c6.jpg)  
Fig. 1. Overview of the symbolic-temporal encoding pipeline, mapping a raw cyber event into a sparse, spike-compatible binary vector.

## B. Three Encoding Roles

To capture the varied semantics of cyber protocols, the feature groups are designed to fulfill three distinct representational roles:

1) Semantic Identity: Non-numeric categorical fields (such as protocols, services, and ports) are mapped into semantically defined bins based on domain context. This avoids assigning arbitrary continuous magnitudes to categorical values, preserving their symbolic meaning.

2) Frequency-Based Context: High-cardinality identifiers, such as IP addresses or CAN arbitration IDs, are mapped to local frequency bins. The encoder maintains an online FIFO window of the most recent L events and updates identifier counts as events arrive. The resulting count for the current identifier is discretized into a frequency bin, capturing local burst or periodic behavior without overfitting to specific identifier values.

3) Relative Timing: The inter-event delay $\Delta t _ { i }$ is mapped to logarithmically spaced bins, allowing the recurrent SNN to learn burst, periodic, and idletime dynamics directly from relative event spacing without fixed-duration accumulation windows or clock-aligned feature bins.

## C. Key Design Properties

The encoding framework satisfies four properties:

Event-Native: Events are processed individually upon arrival, avoiding flow reconstruction or dense traficsummary tensors.

Sparse: The resulting representation is highly sparse, activating only K + 1 channels out of $D .$

Domain-Informed: Bin boundaries are derived from standard network and communication specifications (e.g., IANA port allocations or Ethernet frame sizes).

Hardware-Compatible: The binary activations align directly with the input channels of neuromorphic hardware, minimizing encoding overhead.

## IV. Experimental Setup

All experiments use a common classification setting, training protocol, hardware-constrained simulator, and evaluation methodology. Data are partitioned at the run level: packets or messages from the same temporal run are never split across training, validation, and test sets. For the Combined datasets, stratification is performed by attack category while preserving run boundaries. This avoids event-level leakage across partitions and ensures that test performance is measured on held-out runs. Training runs are used for evolutionary fitness evaluation. Validation runs are monitored during evolution as a generalization check. Held-out test runs are used only for final performance reporting.

## A. Binary Classification Task

We formulate cyber intrusion detection as a per-event binary classification task. Given an event stream $x ,$ , the model must output a prediction $\hat { y } _ { i } ~ \in ~ \{ 0 , 1 \}$ upon the arrival of each event $x _ { i } .$ , where 1 indicates malicious and 0 indicates benign activity. The model must perform this classification using the current encoded event $s _ { i }$ and the context accumulated within the network’s recurrent state.

## B. SNN Optimization and Hardware-Constrained Simulation

We optimize recurrent SNN topology and parameters using Evolutionary Optimization for Neuromorphic Systems (EONS) [11]. EONS is a population-based, gradientfree optimizer that evolves both network structure and neuron/synapse parameters using tournament selection, crossover, and mutation. EONS serves as the optimization backend; the contribution of this work is the eventnative representation supplied to the recurrent SNN. For each dataset, we perform 200 independent EONS runs, each using 300 generations and a population size of 100 candidate networks, for a total of 6.0 million candidate evaluations per dataset.

All candidate networks are evaluated in the Caspian simulator configured to emulate the µCaspian neuromorphic processor [10]. Networks use Integrate-and-Fire neurons with 8-bit thresholds, 16-bit membrane potentials, and 4-bit axonal delays. To maintain edge relevance, evolution is constrained to networks with at most 256 neurons and 4,096 synapses.

## C. Rolling Decision Rule

To prevent transient spikes in output activity from causing excessive false alarms, we apply a rolling decision rule to the raw per-event predictions $\hat { y } _ { i }$ . An attack alarm

is raised if the number of malicious predictions within a rolling output alarm window of the last w events meets or exceeds a threshold θ:

$$
\hat { y } _ { t } ^ { a l a r m } = \left\{ \begin{array} { c c c } { 1 } & { \mathrm { i f } } & { \displaystyle \sum _ { i = t - w + 1 } ^ { t } \hat { y } _ { i } \geq \theta , } \\ { 0 } & { \mathrm { o t h e r w i s e . } } \end{array} \right.\tag{3}
$$

We fix $w = 5 0$ events across all experiments to suppress isolated false positives while maintaining low detection latency. The threshold θ is optimized on the training split. This rolling rule does not aggregate input features, construct flows, or build dense trafic-summary tensors.

## D. Baseline Models

For representation analysis, we train Logistic Regression (LR), Random Forest (RF), Decision Tree (DT), Gaussian Naive Bayes (GNB), and Recurrent Neural Network (RNN) classifiers. All baselines use the same train/validation/test partitions, threshold-selection procedure, and output rolling decision rule as the SNN. Raw features are represented using normalized continuous values, while encoded features use the sparse binary input vector.

## E. Evaluation Metrics

Cybersecurity evaluation must account for both whether attacks are detected and how frequently benign trafic produces false alarms. Prior NIDS work provides precedent for evaluating successful detection at the attack-instance level while measuring false positives at the packet level [22]. We follow this general distinction for event-stream intrusion detection. We define the Attack Detection Rate (ADR) as the proportion of labeled attack intervals containing at least one alarm, and $F P R _ { e v e n t }$ as the fraction of benign event positions for which the detector is in an alarm state. For Network IDS, an event is a packet; for CAN IDS, it is a CAN message. Because EONS requires a single fitness objective, we combine them as J<sub>hybrid</sub>.

$$
A D R = \frac { N _ { d e t e c t e d ~ a t t a c k s } } { N _ { a t t a c k s } } .\tag{4}
$$

$$
F P R _ { e v e n t } = \frac { N _ { b e n i g n ~ e v e n t s ~ i n ~ a l a r m } } { N _ { b e n i g n ~ e v e n t s } } .\tag{5}
$$

$$
J _ { h y b r i d } = A D R - F P R _ { e v e n t } .\tag{6}
$$

This metric is not conventional Youden’s $^ { J , }$ because ADR and $F P R _ { e v e n t }$ are measured at diferent resolutions. Instead, ${ J } _ { h y b r i d }$ is an operational IDS objective that rewards detection of attack instances while penalizing alarm activity on benign trafic. This provides a single fitness value for EONS that balances attack detection and false-positive behavior. We additionally report per-packet or per-message accuracy by comparing the detector alarm state with the ground-truth label at each event position.

## F. Confidence Intervals

We estimate 95% confidence intervals using 10,000 bootstrap samples of the test set. Clusters are grouped by source dataset and trafic class, then sampled with replacement within each group. For each sample, ADR and $F P R _ { e v e n t }$ are recomputed from the resampled attack detections and benign event positions, and $J _ { h y b r i d }$ is then recalculated. The confidence interval is given by the 2.5th and 97.5th percentiles of its bootstrap values. The same samples are used for all models to ensure paired comparisons.

## V. Network IDS Evaluation

We first evaluate the symbolic-temporal framework on packet-level network intrusion detection, where malicious behaviors must be identified from sequential packet arrivals.

## A. Dataset and Preprocessing

We utilize three public packet-level network datasets: TON-IoT [18], Bot-IoT [23], and TII-SSRC-23 [24], and construct a Combined dataset containing runs from all three sources (Table I). Raw packet captures are parsed using PyShark to extract packet-level attributes and ordered into temporal runs. The dataset is balanced at the run level (equal numbers of benign and attackcontaining runs), though individual event labels remain highly imbalanced to reflect realistic deployment. The runs are split 60/10/30 into train, validation, and test partitions, stratified by attack category.

Packet-level IDS is challenging because individual packets are often weakly informative in isolation, and malicious behavior is usually expressed through trafic patterns, feature distributions, and temporal context rather than a single separable packet attribute [2], [3]. This challenge is amplified by the irregular timing structure of network trafic [25] and the heterogeneity and imbalance of real IDS datasets [18], [23], [24]. These properties motivate an event-native representation that preserves per-packet semantics while allowing the recurrent SNN to accumulate evidence across arrivals.

TABLE I  
Network IDS Dataset Summary
<table><tr><td>Dataset</td><td>Runs</td><td>Packets</td><td>Attack %</td></tr><tr><td>TON-IoT [18]</td><td>1,346</td><td>57.4 M</td><td>5.26</td></tr><tr><td>Bot-IoT [23]</td><td>31,096</td><td>459.4 M</td><td>12.60</td></tr><tr><td>TII-SSRĆ-23 [24]</td><td>521</td><td>2.84 M</td><td>32.48</td></tr><tr><td>Combined</td><td>32,963</td><td>519.6 M</td><td>11.90</td></tr></table>

## B. Encoding Scheme

We map each packet to a 36-dimensional sparse spike vector composed of eight feature groups (Table II). IP frequency groups keep a local count of source and destination addresses over a rolling 20,000-packet window, capturing burst behavior without global address tracking. Port groups are binned according to IANA specifications [26], with destination ports having finer granularity to represent common service endpoints. Packet sizes are binned using Ethernet standard frames [27]. Inter-event times (∆t) are discretized into log-spaced bins to capture the heavy-tailed timing distributions typical of network trafic [25].

## C. Results and Baseline Comparison

The Network IDS evaluation assesses whether the symbolic-temporal encoding exposes useful structure from packet-level event streams and whether recurrent spiking state provides additional temporal decision capacity. We therefore report the best evolved SNN performance across datasets and compare raw and encoded inputs using controlled baselines on the Combined Network IDS dataset.

Table III reports the best evolved SNN on each Network IDS dataset. For each dataset, the reported network was selected as the highest-performing candidate on the training data and then evaluated on held-out test runs. Performance remains strong across all individual datasets and the Combined setting. The Combined result is particularly important because it merges trafic from multiple sources and attack distributions, making it a stronger test of whether the representation can support detection across heterogeneous packet streams.

These results are not intended as a direct state-of-theart comparison against flow-based or heavily engineered IDS pipelines, since published systems often use diferent preprocessing, feature construction, task definitions, and evaluation settings. For context, prior work reports 99.99% accuracy on TON-IoT using CNNs [17], 99.87% on Bot-IoT using hybrid LSTM-CNNs [19], and 100% on TII-SSRC-23 using XGBoost [24]. The relevance of Table III is instead that edge-aligned SNNs can achieve high detection directly from packet arrivals while avoiding flow construction and dense feature engineering.

Table IV isolates the role of the representation. Classical models trained on raw packet features perform poorly to moderately, indicating that the packet-level task is dificult when events are treated as static numeric records. When the same models are trained on the proposed encoded representation, performance improves for most classifiers. This shows that the encoder exposes classrelevant structure that is not as accessible in the raw numeric feature space. However, the encoded representation alone is not suficient for strong detection with static classifiers. To provide a non-spiking recurrentstate comparison, we additionally evaluate an RNN using the same encoded event sequences. The RNN achieves $J _ { h y b r i d } = 0 . 7 3 9$ , improving over the static Random Forest and Decision Tree models $( J _ { h y b r i d } = 0 . 7 0 9 )$ , while the recurrent SNN reaches $J _ { h y b r i d } = 0 . 9 8 7$ with a substantially lower $F P R _ { e v e n t }$ of 0.011, compared with 0.140 for the RNN.

![](images/6cdbe4a49d495e641daa638796d3877e21b06047b45719f2428fbcbd9958eb97.jpg)  
Fig. 2. Loss in ${ J } _ { h y b r i d }$ when individual symbolic feature groups are ablated, shown per dataset.

Detection latency is measured as the number of packets between attack onset and the first alarm within the labeled attack interval. On the Combined dataset, the best individual network achieves a median detection latency of 55 packets, with 25% of attacks detected within 22 packets. During high-rate packet bursts, this corresponds to short response delays, further supporting the realtime motivation for event-driven neuromorphic intrusion detection.

Together, these results indicate that a major obstacle to applying compact SNNs to cyber monitoring is representational: once native event semantics are preserved, sparse recurrent inference becomes a viable fit for asynchronous, edge-oriented cyber streams.

## D. Symbolic Feature-Group Ablation

To assess the relative importance of each symbolic attribute, we performed a structured ablation study during network training. For each dataset, we evolved 200 SNNs: 25 networks for each of the 7 conditions where a single symbolic feature group was removed, and 25 control networks trained with all features. All ablations preserve native inter-event timing during SNN propagation because elapsed time governs recurrent spiking dynamics and is part of the asynchronous computation model. The ablation, therefore, evaluates the contribution of each symbolic input group while keeping the event-time substrate fixed.

As shown in Fig. 2, packet size and protocol are the most critical symbolic features across all datasets. Ablating packet size leads to the largest degradation (mean $\Delta J _ { h y b r i d } = - 0 . 0 8 5 )$ , followed closely by protocol $( \Delta J _ { h y b r i d } ~ = ~ - 0 . 0 4 7 )$ . This matches the high volume of small, identical-protocol packet bursts characteristic of DoS and DDoS attacks. Conversely, IP frequency and port groups show negligible impact, suggesting that the networks learn general trafic behaviors rather than overfitting to specific addresses or ports.

TABLE II  
Symbolic Input Bin Definitions. Each network packet activates exactly one bin per feature group.
<table><tr><td>Bin</td><td>Definition</td><td>Bin</td><td>Definition</td><td>Bin</td><td>Definition</td></tr><tr><td>0</td><td>Src IP freq: Low (0–5k)</td><td>12</td><td>Dst port: Web (80, 443, 8080)</td><td>24</td><td>Service: HTTP/HTTPS</td></tr><tr><td>1</td><td>Src IP freq: Med-low (5k–10k)</td><td>13</td><td>Dst port: DNS (53, 5353)</td><td>25</td><td>Service: DNS</td></tr><tr><td>2</td><td>Src IP freq: Med-high (10k–15k)</td><td>14</td><td>Dst port: SSH (22)</td><td>26</td><td>Service: TLS/SSL</td></tr><tr><td>3</td><td>Src IP freq: High (15k–20k)</td><td>15</td><td>Dst port: Mail (25, 465, 587)</td><td>27</td><td>Service: Other</td></tr><tr><td>4</td><td>Dst IP freq: Low (0–5k)</td><td>16</td><td>Dst port: NTP (123)</td><td>28</td><td>Pkt size: Tiny (0–64 B)</td></tr><tr><td>5</td><td>Dst IP freq: Med-low (5k–10k)</td><td>17</td><td>Dst port: Other well-known</td><td>29</td><td>Pkt size: Small (65–512 B)</td></tr><tr><td>6</td><td>Dst IP freq: Med-high (10k–15k)</td><td>18</td><td>Dst port: Registered</td><td>30</td><td>Pkt size: Standard (513–1514 B)</td></tr><tr><td>7</td><td>Dst IP freq: High (15k–20k)</td><td>19</td><td>Dst port: Dynamic/Other</td><td>31</td><td>Pkt size: Large (&gt;1514 B)</td></tr><tr><td>8</td><td>Src port: Well-known (0–1023)</td><td>20</td><td>Protocol: TCP</td><td>32</td><td>∆t: Very fast (&lt;0.1 ms)</td></tr><tr><td>9</td><td>Src port: Registered</td><td>21</td><td>Protocol: UDP</td><td>33</td><td>∆t: Fast (0.1–1 ms)</td></tr><tr><td>10</td><td>Src port: Dynamic</td><td>22</td><td>Protocol: ICMP</td><td>34</td><td>∆t: Moderate (1-10 ms)</td></tr><tr><td>11</td><td>Src port: Other</td><td>23</td><td>Protocol: Other</td><td>35</td><td>∆t: Slow (≥10 ms)</td></tr></table>

TABLE III  
Best SNN Performance per Network IDS Dataset
<table><tr><td>Dataset</td><td> ${ J } _ { h y b r i d }$ </td><td>ADR</td><td> $F P R _ { e v e n t }$ </td><td>Packet Accuracy</td></tr><tr><td>TON-IoT</td><td>0.9929</td><td>0.9952</td><td>0.0023</td><td>0.9826</td></tr><tr><td>Bot-IoT</td><td>0.9908</td><td>0.9998</td><td>0.0090</td><td>0.8665</td></tr><tr><td>TII-SSRC-23</td><td>0.9996</td><td>1.0000</td><td>0.0004</td><td>0.7952</td></tr><tr><td>Combined</td><td>0.9873</td><td>0.9986</td><td>0.0113</td><td>0.8750</td></tr></table>

## VI. CAN IDS Evaluation

To test the transferability of our framework, we apply it to Controller Area Network (CAN) bus trafic, a structurally diferent asynchronous stream characterized by 11-bit arbitration IDs, short 8-byte payloads, and submillisecond inter-message spacing. CAN attacks typically inject spoofed frames to manipulate in-vehicle systems, demanding that a detector analyze both transmission timing and message payload sequences.

## A. Dataset and Preprocessing

We evaluate using two public CAN benchmarks: ROAD [21] and Car-Hacking [20], alongside a Combined CAN dataset containing runs from both (Table V). ROAD provides dynamometer captures with long contiguous attack segments, while Car-Hacking represents short, burst-like injection campaigns. For evaluation, adjacent attack messages separated by fewer than 50 benign frames are merged into single attack intervals, and runs are padded with benign context. Splits are 60/10/30 for train/validation/test.

## B. Encoding Scheme

We adapt the three representational roles to CAN semantics, resulting in a 26-dimensional hybrid input vector (Table VI).

The framework does not require every field to be symbolic. It assigns encoding roles according to field semantics: symbolic identifiers are represented categorically, timing is discretized as ∆t, and payload bytes retain magnitude because their values carry meaningful signal content. This preserves native per-event structure while applying numeric spike encoding only where numeric magnitude is semantically meaningful.

For CAN ID frequency, we count occurrences of each arbitration ID within a sliding 1,000-message window, binning the result into six categorical bins. Relative timing (∆t) is binned into four classes representing submillisecond ranges. For numeric payload content, we employ flip-flop rate coding, assigning two input channels per payload byte to represent its continuous value as positive and negative deviation rates.

## C. Results and Baseline Comparison

The CAN IDS evaluation serves the same purpose as the Network IDS evaluation: to test whether the proposed representation exposes useful structure from native event streams and whether recurrent spiking state adds temporal decision capacity beyond the encoded inputs alone. Unlike packet-level network trafic, CAN messages are shorter, more periodic, and dominated by arbitration ID, payload content, and sub-millisecond inter-message timing. This makes CAN a useful transfer test for the symbolictemporal framework.

Table VII reports the best evolved SNN for each CAN dataset. For each dataset, the reported network was selected as the highest-performing candidate on the training data and then evaluated on held-out test runs. The SNN achieves strong detection on both individual datasets and maintains high performance on the Combined CAN dataset, which merges short injection bursts with longer ROAD captures.

TABLE IV  
Baseline Comparison on the Combined Network IDS Dataset
<table><tr><td>Model</td><td>Representation</td><td> ${ J } _ { h y b r i d }$ </td><td> ${ J } _ { h y b r i d }$  95% CI</td><td>ADR</td><td> $F P R _ { e v e n t }$ </td><td>Packet Accuracy</td></tr><tr><td>LR</td><td>Raw</td><td>0.4367</td><td>[0.4150, 0.4584]</td><td>0.8729</td><td>0.4362</td><td>0.4698</td></tr><tr><td>LR</td><td>Encoded</td><td>0.6811</td><td>[0.6673, 0.6948]</td><td>0.9364</td><td>0.2554</td><td>0.6444</td></tr><tr><td>RF</td><td>Raw</td><td>0.7004</td><td>[0.6870, 0.7135]</td><td>0.9078</td><td>0.2074</td><td>0.5574</td></tr><tr><td>RF</td><td>Encoded</td><td>0.7089</td><td>[0.6960, 0.7218]</td><td>0.9607</td><td>0.2517</td><td>0.6709</td></tr><tr><td>DT</td><td>Raw</td><td>0.7068</td><td>[0.6934, 0.7200]</td><td>0.8937</td><td>0.1869</td><td>0.5428</td></tr><tr><td>DT</td><td>Encoded</td><td>0.7088</td><td>[0.6958, 0.7216]</td><td>0.9607</td><td>0.2519</td><td>0.6708</td></tr><tr><td>GNB</td><td>Raw</td><td>0.1098</td><td>[0.0982, 0.1213]</td><td>0.9972</td><td>0.8874</td><td>0.1520</td></tr><tr><td>GNB</td><td>Encoded</td><td>0.5163</td><td>[0.5034, 0.5293]</td><td>0.9457</td><td>0.4294</td><td>0.4348</td></tr><tr><td>RNN</td><td>Raw</td><td>0.4667</td><td>[0.4505, 0.4825]</td><td>0.9038</td><td>0.4371</td><td>0.4117</td></tr><tr><td>RNN</td><td>Encoded</td><td>0.7389</td><td>[0.7248, 0.7527]</td><td>0.8789</td><td>0.1401</td><td>0.8764</td></tr><tr><td>SNN</td><td>Encoded</td><td>0.9873</td><td>[0.9856, 0.9888]</td><td>0.9986</td><td>0.0113</td><td>0.8750</td></tr></table>

TABLE V  
CAN IDS Dataset Summary
<table><tr><td>Dataset</td><td>Runs</td><td>Messages</td><td>Attack %</td></tr><tr><td>ROAD</td><td>28</td><td>25.8 M</td><td>3.06</td></tr><tr><td>Car-Hacking</td><td>2,404</td><td>24.9 M</td><td>9.35</td></tr><tr><td>Combined CAN</td><td>2,432</td><td>50.8 M</td><td>6.15</td></tr></table>

TABLE VI  
CAN Input Encoding and Input Dimensions
<table><tr><td>Feature Group</td><td>Features</td><td>SNN In</td><td>Description</td></tr><tr><td>CAN ID freq</td><td>6</td><td>6</td><td>Sliding 1k window; one-hot: 0, 1–5, 6–10,</td></tr><tr><td>Payload bytes</td><td>8</td><td>16</td><td>11–20, 21–40, 41+ 8 bytes × 2 (flip-flop rate coding)</td></tr><tr><td>∆t</td><td>4</td><td>4</td><td> $< 0 . 5$  ms, 0.5–1 ms, 1–5 ms,  $\geq 5$  ms</td></tr><tr><td>Total</td><td>18</td><td>26</td><td></td></tr></table>

TABLE VII  
Best SNN Performance on CAN IDS Datasets
<table><tr><td>Dataset</td><td> ${ J } _ { h y b r i d }$ </td><td>ADR</td><td> $F P R _ { e v e n t }$ </td><td>Message Accuracy</td></tr><tr><td>Car-Hacking</td><td>0.9971</td><td>1.0000</td><td>0.0029</td><td>0.8418</td></tr><tr><td>ROAD</td><td>0.9850</td><td>1.0000</td><td>0.0150</td><td>0.9850</td></tr><tr><td>Combined</td><td>0.9803</td><td>1.0000</td><td>0.0197</td><td>0.9063</td></tr></table>

These results are not presented as a direct state-ofthe-art comparison against all CAN IDS methods, since prior systems often use diferent preprocessing, feature construction, and evaluation procedures. Existing CAN detectors commonly report accuracies above 99% [4], [9]; the distinction here is that the proposed approach performs detection directly from message arrivals using an edge-aligned SNN and the event-native encoded representation. The lower message-level accuracy on Car-Hacking should be interpreted in light of our operational objective. ADR counts an attack as detected when at least one alarm occurs within its labeled interval, so high ADR can coexist with lower message-level accuracy. We use ${ J } _ { h y b r i d }$ as our primary operational metric for comparing attack detection and benign alarm activity across ROAD, Car-Hacking, and the Combined CAN setting, while reporting message-level accuracy as a complementary measure.

Table VIII isolates the contribution of the representation on the Combined CAN dataset. Classical models trained on raw normalized features achieve limited performance. When trained on the proposed 26-dimensional hybrid encoding, performance improves for most baselines. This suggests that the encoding exposes useful structure from CAN ID frequency, payload behavior, and intermessage timing.

As in the Network IDS evaluation, the encoded representation improves the performance of static classifiers. A non-spiking RNN evaluated on the same encoded message sequences achieves a $J _ { h y b r i d } = 0 . 9 7 1 9$ (95% CI: [0.9579, 0.9830]), compared with $J _ { h y b r i d } = 0 . 9 8 0 3$ (95% CI: [0.9672, 0.9878]) for the SNN. These results indicate that temporal recurrence accounts for much of the improvement, while the SNN provides slightly improved performance in an event-driven, hardware-compatible form.

Detection latency is measured as the number of CAN messages between attack onset and the first alarm within the labeled attack interval. On the Combined CAN dataset, the best individual network achieves a median detection latency of 48 messages, with 25% of attacks detected within 35 messages. At typical CAN bus message rates, this corresponds to short detection delays suitable for real-time edge monitoring.

These results reproduce the central trend observed in the Network IDS study: when native event semantics are preserved in a meaningful representation, SNNs can use temporal state efectively while retaining the low-latency, event-driven computation needed for edge-oriented cyber monitoring.

TABLE VIII  
Baseline Comparison on the Combined CAN Dataset
<table><tr><td>Model</td><td>Representation</td><td> ${ J } _ { h y b r i d }$ </td><td> ${ J } _ { h y b r i d }$  95% CI</td><td>ADR</td><td> $F P R _ { e v e n t }$ </td><td>Message Accuracy</td></tr><tr><td>LR</td><td>Raw</td><td>0.3741</td><td>[0.3042, 0.4555]</td><td>0.6263</td><td>0.2522</td><td>0.7412</td></tr><tr><td>LR</td><td>Encoded</td><td>0.5140</td><td>[0.4196, 0.6235]</td><td>0.6824</td><td>0.1683</td><td>0.8236</td></tr><tr><td>RF</td><td>Raw</td><td>0.8107</td><td>[0.7221, 0.9183]</td><td>0.8772</td><td>0.0665</td><td>0.9304</td></tr><tr><td>RF</td><td>Encoded</td><td>0.8410</td><td>[0.7655, 0.9262]</td><td>0.8965</td><td>0.0555</td><td>0.9419</td></tr><tr><td>DT</td><td>Raw</td><td>0.8174</td><td>[0.7467, 0.9004]</td><td>0.9115</td><td>0.0940</td><td>0.9063</td></tr><tr><td>DT</td><td>Encoded</td><td>0.8485</td><td>[0.7910, 0.9167]</td><td>0.9185</td><td>0.0700</td><td>0.9294</td></tr><tr><td>GNB</td><td>Raw</td><td>0.3049</td><td>[0.2374, 0.3822]</td><td>0.3424</td><td>0.0375</td><td>0.9288</td></tr><tr><td>GNB</td><td>Encoded</td><td>0.2279</td><td>[0.1811, 0.2878]</td><td>0.2516</td><td>0.0237</td><td>0.9370</td></tr><tr><td>RNN</td><td>Raw</td><td>0.9712</td><td>[0.9601, 0.9829]</td><td>0.9918</td><td>0.0206</td><td>0.6341</td></tr><tr><td>RNN</td><td>Encoded</td><td>0.9719</td><td>[0.9579, 0.9830]</td><td>0.9726</td><td>0.0007</td><td>0.9660</td></tr><tr><td>SNN</td><td>Encoded</td><td>0.9803</td><td>[0.9672, 0.9878]</td><td>1.0000</td><td>0.0197</td><td>0.9063</td></tr></table>

## VII. Discussion

## A. Temporal Validation of the Hybrid Objective

A limitation of $J _ { h y b r i d }$ is that ADR is interval-based. Long attack intervals provide more opportunities for an alarm to overlap an attack by chance, allowing a temporally uninformative detector to achieve an inflated ADR. We therefore evaluate whether the observed performance reflects meaningful temporal alignment between SNN activity and attack behavior.

We apply a circular-shift control within each held-out run, randomly shifting the SNN output sequence to break its alignment with the attack labels while preserving its activity pattern. The rolling decision rule is then reapplied, and $J _ { h y b r i d } $ ADR, and $F P R _ { e v e n t }$ are recomputed over 10,000 shifts.

TABLE IX  
Observed and Temporally Shifted SNN Performance
<table><tr><td>Application</td><td>Evaluation</td><td> ${ J } _ { h y b r i d }$ </td><td>ADR</td><td> $F P R _ { e v e n t }$ </td></tr><tr><td>Network IDS</td><td>Observed</td><td>0.9873</td><td>0.9986</td><td>0.0113</td></tr><tr><td></td><td>Shifted Mean</td><td>0.7540</td><td>0.8609</td><td>0.1069</td></tr><tr><td>CAN IDS</td><td>Observed</td><td>0.9803</td><td>1.0000</td><td>0.0197</td></tr><tr><td></td><td>Shifted Mean</td><td>0.6490</td><td>0.7714</td><td>0.1224</td></tr></table>

Temporal shifting reduces mean $J _ { h y b r i d }$ by 0.2333 for Network IDS and 0.3313 for CAN IDS (Table IX), with the observed score exceeding all 10,000 shifted replicates in both domains $( p < 0 . 0 0 0 1 )$ ). However, shifted ADR remains relatively high, confirming that long attack intervals can still produce chance overlap. We therefore evaluate whether alarms occur promptly and remain aligned with attack activity after onset. First, fixed-horizon analysis measures the cumulative fraction of attacks for which a new alarm episode begins within the first K events. For Network IDS, this reaches 40.5%, 75.6%, and 83.1% within 50, 100, and 250 events, compared with 25.7%, 31.7%, and 48.7% under temporal shifting. For CAN IDS, the corresponding observed rates are 52.1%, 74.8%, and 90.1%, compared with 3.9%, 7.0%, and 13.7%. Median detection latency is also substantially lower for the observed outputs: 55 versus 263 events for Network IDS and 48 versus > 500 messages for CAN IDS. We additionally examine the instantaneous alarm state at specific positions relative to attack onset. This analysis shows whether alarm activity becomes concentrated after an attack begins. Network IDS increases from 11.0% of attacks in the alarm state at onset to 31.5% at event +49, while the shifted control remains near 16%. CAN IDS increases from 0.0% at onset to 52.1% at +49, while the shifted control remains near 37%. These results show that the high $J _ { h y b r i d }$ scores are not explained by long attack intervals alone. Observed outputs produce alarms substantially earlier than shifted controls, while new alarm episodes and alarm-state activity are strongly aligned with attack onset.

![](images/33d704ffd9333ea5c1b847e7ada38c2ee8f06471c9b9ceb7cc2086765e264597.jpg)  
Fig. 3. Evolved SNN topologies for Network IDS (top) and CAN IDS (bottom). Input neurons are blue, hidden are grey, and output are red.

## B. Topology

The evolved SNN topologies also highlight the deployment motivation for the proposed representation. The evolved Network IDS SNN (Fig. 3, top) uses only

46 neurons with dense recurrence, while still achieving high detection on the Combined dataset. This compact scale contrasts with typical deep IDS architectures, which rely on dense tensor operations and larger parameterized models. The input connections are concentrated around packet size (bins 28/30) and protocol (bins 20/21), which form recurrent hubs. This structural alignment with our ablation study suggests that the network routes information through the most discriminative encoding channels while suppressing brittle categorical indicators.

Similarly, for the CAN IDS network (Fig. 3, bottom), EONS produced a compact recurrent core using active inputs from ID-frequency bins, payload byte channels, and inter-arrival timing bins. Several inputs are pruned, leaving a small set of connections associated with highfrequency ID transmission and payload deviations. This compact recurrent structure is particularly relevant for CAN monitoring, where short inter-message intervals make detection latency critical: the small topology and event-driven dynamics allow temporal evidence to be processed as messages arrive, without waiting for large windows or dense feature construction.

The compact evolved topologies align with the µCaspian-oriented deployment constraints [10]. µCaspian is a small event-driven FPGA platform for neuromorphic edge applications. Hardware power and latency have also been evaluated on this platform in prior work: Ghawaly et al. [28] deployed compact EONS-evolved Integrateand-Fire SNNs on µCaspian and reported 2 mW power consumption with 20.2 ms inference latency. The resulting Network IDS and CAN IDS topologies fall within the same neuron, synapse, precision, and axonal-delay limits, indicating that the proposed event-native representation supports compact recurrent SNNs compatible with an established edge-oriented neuromorphic deployment path. Although the present work does not report physical hardware measurements for the cyber models, these constraints provide a hardware-aligned basis for future direct deployment from native packet and CAN event streams.

## VIII. Limitations

We acknowledge several limitations of this study. First, we do not claim that event-native spiking classifiers universally outperform conventional intrusion detection systems. Rather, our contribution is representational: we demonstrate that neuromorphic systems can perform eventnative monitoring when cyber semantics are preserved at the input interface. Second, the encoding framework is domain-informed and relies on manually defined bin boundaries. Future work should investigate methods for learning these boundaries directly from data. Third, the RNN and conventional classifiers were tuned through limited sweeps over key hyperparameters rather than exhaustive optimization. Future evaluations could expand this comparison through more extensive tuning and the inclusion of GRUs, LSTMs, and temporal transformers.

Additionally, $J _ { h y b r i d }$ is an operational optimization objective rather than a conventional sample-level classification metric; because ADR is interval-based, we supplement it with additional temporal analyses to test whether detections are genuinely aligned with attack activity. Finally, our evaluations use public benchmark datasets and simulated hardware constraints; therefore, they do not capture all deployment challenges.

## IX. Conclusion

This paper addresses the representational gap at the intersection of spiking neural networks and cybersecurity. We introduced a symbolic-temporal spike encoding framework that maps heterogeneous, asynchronous cyber events directly to sparse, spike-compatible vectors at arrival. By assigning distinct encoding roles to semantic identity, local frequency context, and inter-event timing, the framework preserves categorical semantics and temporal spacing without converting events to artificial numeric averages.

Across packet-level Network IDS and message-level CAN IDS, compact recurrent SNNs operating under edgealigned µCaspian constraints achieved strong detection $( J _ { h y b r i d } ~ \geq ~ 0 . 9 8 0 )$ . The proposed event-to-spike representation interface preserves the symbolic and temporal structure of cyber streams, enabling edge-aligned SNNs to perform direct, low-latency neuromorphic monitoring on native event streams.

## References

[1] W. Maass, “Networks of spiking neurons: the third generation of neural network models,” Neural networks, vol. 10, no. 9, pp. 1659–1671, 1997.

[2] A. Lakhina, M. Crovella, and C. Diot, “Mining anomalies using trafic feature distributions,” in Proceedings of the 2005 Conference on Applications, Technologies, Architectures, and Protocols for Computer Communications, ser. SIGCOMM ’05. New York, NY, USA: Association for Computing Machinery, 2005, pp. 217–228. [Online]. Available: https: //doi.org/10.1145/1080091.1080118

[3] R. Sommer and V. Paxson, “Outside the closed world: On using machine learning for network intrusion detection,” in 2010 IEEE Symposium on Security and Privacy. Oakland, CA, USA: IEEE, 2010, pp. 305–316.

[4] S. Rajapaksha, H. Kalutarage, M. O. Al-Kadri, A. Petrovski, G. Madzudzo, and M. Cheah, “AI-based intrusion detection systems for in-vehicle networks: A survey,” ACM Computing Surveys, vol. 55, no. 11, pp. 1–40, 2023.

[5] S. Zhou and X. Li, “Spiking neural networks with single-spike temporal-coded neurons for network intrusion detection,” in 2020 25th International Conference on Pattern Recognition (ICPR). Milan, Italy: IEEE, 2021, pp. 8148–8155.

[6] M. Živadinović and D. Simić, “Resource eficient internet-ofthings intrusion detection with spiking neural networks,” in Position Papers of the 19th Conference on Computer Science and Intelligence Systems (FedCSIS 2024), Belgrade, Serbia, Nov. 2024, pp. 73–78.

[7] P. S N, S. D, N. Shelke, A. Pimpalkar, D. K. Jang Bahadur Saini, and G. H. Kumar, “Event-driven intrusion detection systems using spiking neural networks for edge and iot security,” in 2025 5th International Conference on Soft Computing for Security Applications (ICSCSA), 2025, pp. 41–47.

[8] D. R. Follett, D. Townsend, P. L. Follett, G. D. Karpman, J. H. Naegle, R. A. Suppona, J. B. Aimone, and C. D. James, “Neuromorphic data microscope,” in Proceedings of the Neuromorphic Computing Symposium, ser. NCS ’17. New York, NY, USA: Association for Computing Machinery, 2017. [Online]. Available: https://doi.org/10.1145/3183584.3183617

[9] R. Islam, S. Alam, C. Yakopcic, N. Rahman, S. Khan, and T. Taha, “Unsupervised anomaly detection for automotive can bus on the intel loihi,” in 2024 International Joint Conference on Neural Networks (IJCNN). Yokohama, Japan: IEEE, 2024, pp. 1–8.

[10] J. P. Mitchell, C. D. Schuman, and T. E. Potok, “A small, low cost event-driven architecture for spiking neural networks on fpgas,” in International Conference on Neuromorphic Systems 2020, ser. ICONS 2020. New York, NY, USA: Association for Computing Machinery, 2020. [Online]. Available: https://doi.org/10.1145/3407197.3407216

[11] C. D. Schuman, J. P. Mitchell, R. M. Patton, T. E. Potok, and J. S. Plank, “Evolutionary optimization for neuromorphic systems,” in Proceedings of the 2020 Annual Neuro-Inspired Computational Elements Workshop. New York, NY, USA: ACM, 2020, pp. 1–9.

[12] W. Gerstner and W. M. Kistler, Spiking neuron models: Single neurons, populations, plasticity. Cambridge University Press, 2002.

[13] M. Bouvier, A. Valentian, T. Mesquida, F. Rummens, M. Reyboz, E. Vianello, and E. Beigne, “Spiking neural networks hardware implementations and challenges: A survey,” ACM Journal on Emerging Technologies in Computing Systems (JETC), vol. 15, pp. 1–35, 2019.

[14] M. Du, F. Li, G. Zheng, and V. Srikumar, “Deeplog: Anomaly detection and diagnosis from system logs through deep learning,” in Proceedings of the 2017 ACM SIGSAC Conference on Computer and Communications Security (CCS). ACM, 2017, pp. 1285–1298.

[15] H. Mei and J. Eisner, “The neural hawkes process: A neurally self-modulating multivariate point process,” in Advances in Neural Information Processing Systems (NeurIPS), vol. 30. Red Hook, NY: Curran Associates, Inc., 2017, pp. 6754–6764.

[16] G. Gallego, T. Delbrück, G. Orchard, C. Bartolozzi, B. Taba, A. Censi, S. Leutenegger, A. J. Davison, J. Conradt, K. Daniilidis, and D. Scaramuzza, “Event-based vision: A survey,” IEEE Transactions on Pattern Analysis and Machine Intelligence, vol. 44, no. 1, pp. 154–180, 2022.

[17] H. Asgharzadeh, A. Ghafari, M. Masdari, and F. S. Gharehchopogh, “An intrusion detection system on the internet of things using deep learning and multi-objective enhanced gorilla troops optimizer,” Journal of Bionic Engineering, vol. 21, no. 5, pp. 2658–2684, 2024.

[18] N. Moustafa, “A new distributed architecture for evaluating aibased security systems at the edge: Network ton\_iot datasets,” Sustainable Cities and Society, vol. 72, p. 102994, 2021.

[19] P. Sinha, D. Sahu, S. Prakash, T. Yang, R. S. Rathore, and V. Pandey, “A high performance hybrid lstm cnn secure architecture for iot environments using deep learning,” Scientific Reports, vol. 15, 03 2025.

[20] H. M. Song, J. Woo, and H. K. Kim, “In-vehicle network intrusion detection using deep convolutional neural network,” Vehicular Communications, vol. 21, p. 100198, 2020.

[21] M. E. Verma, R. A. Bridges, M. D. Iannacone, S. C. Hollifield, P. Moriano, S. C. Hespeler, B. Kay, and F. L. Combs, “A comprehensive guide to CAN IDS data and introduction of the ROAD dataset,” PLoS ONE, vol. 19, no. 1, p. e0296879, 2024.

[22] D. Bolzoni, B. Crispo, and S. Etalle, “ATLANTIDES: An architecture for alert verification in network intrusion detection systems,” in 21st Large Installation System Administration Conference (LISA 07). Dallas, TX: USENIX Association, Nov. 2007. [Online]. Available: https://www.usenix.org/confe rence/lisa-07/atlantides-architecture-alert-verification-netwo rk-intrusion-detection-systems

[23] N. Koroniotis, N. Moustafa, E. Sitnikova, and B. Turnbull, “Towards the development of realistic botnet dataset in the internet of things for network forensic analytics: Bot-iot dataset,” Future Generation Computer Systems, vol. 100, pp.

779–796, 2019. [Online]. Available: https://www.sciencedirect. com/science/article/pii/S0167739X18327687

[24] D. Herzalla, W. T. Lunardi, and M. Andreoni, “Tii-ssrc-23 dataset: Typological exploration of diverse trafic patterns for intrusion detection,” IEEE Access, vol. 11, pp. 118 577–118 594, 2023.

[25] V. Paxson and S. Floyd, “Wide area trafic: the failure of poisson modeling,” IEEE/ACM Transactions on Networking, vol. 3, no. 3, pp. 226–244, 1995.

[26] M. Cotton, L. Eggert, D. J. D. Touch, M. Westerlund, and S. Cheshire, “Internet Assigned Numbers Authority (IANA) Procedures for the Management of the Service Name and Transport Protocol Port Number Registry,” RFC 6335, Aug. 2011. [Online]. Available: https://www.rfc-editor.org/info/rfc 6335

[27] IEEE, “IEEE standard for ethernet,” IEEE Std 802.3-2022, 2022.

[28] J. Ghawaly, A. Nicholson, C. Schuman, D. Diez, A. Young, and B. Witherspoon, “Exploring spiking neural networks for binary classification in multivariate time series at the edge,” in 2025 International Joint Conference on Neural Networks (IJCNN). Rome, Italy: IEEE, Jun. 2025, pp. 1–10. [Online]. Available: http://dx.doi.org/10.1109/IJCNN64981.2025.11228041