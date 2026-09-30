# Dagger: Decoupling-based Model Stealing Attack against Graph Neural Networks

Ying Song

University of Pittsburgh

yis121@pitt.edu

Xiaowei Jia

Rutgers University

xiaowei.jia@rutgers.edu

Balaji Palanisamy

University of Pittsburgh

bpalan@pitt.edu

Abstract—As Graph Neural Networks (GNNs) are widely deployed as Machine Learning-as-a-Service (MLaaS) APIs, model stealing attacks have emerged as a critical security threat. By querying a victim model’s black-box API, an adversary can construct a functionally equivalent surrogate model, compromising proprietary intellectual property and downstream security. Existing GNN stealing attacks, however, rely on overly permissive assumptions, such as soft-label outputs, large query budgets, full-graph query access, and prior knowledge of victim backbones that rarely hold in real-world deployments. In this work, we formalize a strictly constrained black-box, hard-label and backbone-agnostic threat model for GNN stealing attacks under a tight query budget. Given these realistic restrictions, we identify four fundamental challenges: sparse local structures and isolated nodes that degrade victim label quality, insufficient supervision signals, systematic imbalance with incomplete class coverage, and backbone mismatch. To address these interlocking barriers, we propose Dagger, a novel two-phase decoupling-based attack framework. Specifically, in Phase 1, Dagger pre-trains a surrogate using decoupled information propagation to preserve structural context over sparse local subgraphs while handling isolated nodes, combined with manifold-level node mixup to synthesize continuous supervision signals and smooth decision boundaries. In Phase 2, Dagger freezes the encoder and fine-tunes the classifier head via class-balanced sampling paired with logit adjustment to rectify severe query imbalance without requiring extra victim queries. Extensive experiments across four benchmark graphs and four GNN backbones demonstrate that Dagger consistently outperforms state-of-the-art GNN stealing attacks, achieving up to 18.16% higher fidelity while only utilizing 12.23× fewer queries than the strongest baseline. Furthermore, Dagger consistently bypasses state-of-the-art query-monitoring and backdoor-based watermarking defenses, underscoring the urgent need for defense paradigms tailored to such realistic GNN stealing threats.

Keywords—Graph Neural Network, Model Stealing Attack

## I. INTRODUCTION

Graph Neural Networks (GNNs) have showcased remarkable performance across diverse real-world applications, ranging from drug discovery [2] and fraud detection [12] to recommendation systems [25]. Since training proprietary GNNs demands expensive data curation, computational resources and domain expertise, model providers increasingly commercialize their models via black-box Machine Learning-as-a-Service (MLaaS) APIs. However, their economic value renders GNNs attractive targets for model stealing attacks, where adversaries issue queries to extract functionally equivalent surrogate models, compromising proprietary intellectual property. Moreover, such stolen GNNs can serve as a springboard for subsequent adversarial attacks, such as backdoor injection and privacy inference attacks, thereby threatening broader system security.

Despite a growing body of work on model stealing attacks against GNNs [1], [5], [16], [18], [23], [24], [29], existing methods suffer from five critical limitations that severely hinder their practical applicability. (1) Over-reliance on rich information leakage: prior works assume APIs return full posterior probability or node embeddings [16], [18], [23], whereas model providers often restrict outputs to hard labels due to strict security and privacy policies in reality. (2) Topological over-privilege: current hard-label attacks still perform queries implicitly on the full graph or multi-hop neighborhoods beyond legitimate access [1], [5], [24], granting structural information that would be unavailable in a realistic black-box setting. (3) Large query budgets and detection risks: even data-free attacks that avoid using any real data still resort to synthetic data generation techniques, such as generative adversarial network (GAN)-based graph generation [29]. These approaches always incur high computational overhead, training instability, and statistically anomalous query patterns that can be readily detected by query-monitoring defenses [9]. (4) Victim backbone dependency: the vast majority of GNN stealing attacks heavily depend on specific victim architectures or assume prior knowledge of the target backbone, which rarely holds in realworld deployments. Under a true black-box setting, backbone misalignment may severely degrade surrogate performance due to divergent information propagation dynamics. (5) Evaluation inconsistency: while prior works claim to operate under restricted black-box threat models [23], [24], their open-source implementations implicitly grant the adversary access to the full graph during inference. This privilege inflates reported attack performance, suggesting that the practical severity of GNN stealing attacks under strict black-box constraints has been systematically underestimated and motivating a rigorous re-examination under a more faithful threat model.

To uncover the true vulnerabilities of GNNs under realistic MLaaS deployments, we formalize a strictly constrained blackbox, hard-label, and backbone-agnostic threat model for GNN stealing attacks under a tight query budget. Specifically, each query is strictly executed on the local subgraph induced by the adversary’s query node set, preventing access to the global topological context. Under this threat model, we identify four fundamental challenges that existing methods fail to address.

• Sparse local structures and isolated nodes: queries over induced subgraphs frequently collapse into sparse topologies or isolated nodes, particularly on sparse or smallscale graphs. Standard GNNs, which rely on multi-hop neighborhood aggregation, degrade significantly under such degenerative structures, producing unreliable victim predictions that corrupt the surrogate’s training signals.

• Insufficient supervision signals: Discrete hard labels discard inter-class confidence distributions, leaving the surrogate with insufficient supervision to learn smooth decision boundaries under a strict query budget. Unreliable victim predictions further propagate label noise into surrogate training, severely distorting the surrogate’s decision boundaries across class margins.

• Systematic imbalance and incomplete class coverage: Structural dominance in sparse subgraphs yields systematic biases, where majority classes receive a disproportionate share of labels while the minority remain severely under-represented or even uncovered by victim predictions.

• Victim Backbone Mismatch: The misalignment between the unknown victim and surrogate backbones further amplifies the aforementioned issues.

To tackle these interlocking challenges, we propose a novel decoupling-based model stealing attack against graph neural networks (Dagger). This framework consists of two phases, namely Decoupling-based Surrogate Pre-training and Imbalance-aware Classifier Tuning. In the first phase, motivated by APPNP [4] to decouple feature transformation from graph propagation, we pretrain a topology-adaptive surrogate using a decoupled architecture to preserve structural context over sparse local subgraphs while maintaining meaningful representations for isolated nodes. To compensate for hardlabel coarseness and incomplete class coverage, we perform manifold-level node mixup, which interpolates hidden representations across class boundaries. In the second phase, to rectify severe query-induced class imbalance, we freeze the surrogate encoder and only fine-tune the classifier head through class-balanced sampling paired with logit adjustment [14], providing complementary calibrations at both gradient and prediction levels to improve surrogate fidelity on minority classes without requiring extra victim queries.

We evaluate Dagger across four representative benchmark graph datasets and four victim GNN backbones, demonstrating both superior stealing effectiveness and query efficiency. Notably, with 12.23× fewer queries than the strongest baseline [24], Dagger still achieves up to 18.16% higher fidelity. With the same query budget, Dagger outperforms the rest baselines by up to 45.27% in fidelity, particularly on sparse and classimbalanced graphs. Furthermore, we assess Dagger against two state-of-the-art defense paradigms: a query-monitoring detector [9] and a backdoor-based watermarking mechanism [26]. Since Dagger issues queries via real local subgraphs, its query patterns exhibit no distribution-level anomalies. Additionally, the strict yet realistic constraints prevent watermark triggers from being transferred to the surrogate. The experimental validation confirms that Dagger consistently bypasses both defenses, underscoring the urgent need for defense mechanisms tailored to realistic GNN stealing threats.

## We summarize our contributions as follows:

• Realistic Threat Formalization: We systematically uncover five critical bottlenecks of existing stealing methods and formalize a strictly constrained black-box, hard-label, and backbone-agnostic threat model for GNN stealing attacks under a tight query budget.

• Novel Decoupled Framework: We propose Dagger, a two-phase decoupling-based attack framework that leverages decoupled information propagation to handle structural sparsity, applies manifold-level node mixup to enhance representation diversity and class coverage, and employs class-balanced sampling paired with logit adjustment to resolve severe query imbalance.

• Extensive Empirical Validation: We conduct comprehensive evaluations across diverse graph datasets, victim backbones, and defense mechanisms. The empirical results confirm Dagger’s superior effectiveness and efficiency, while demonstrating its resistance to both querymonitoring and watermarking defenses.

## II. BACKGROUND AND RELATED WORK

## A. Node Classification

Given an undirected attributed graph $\mathcal { G } = ( \nu , \mathcal { E } , X )$ , V denotes a node set with $| \nu |$ nodes and each node v is associated with a feature vector $X _ { v } \in \mathcal { R } ^ { 1 \times d }$ , where $d$ is the dimension of node features, $\mathcal { E }$ represents an edge set with |E| edges, GNNs $\Phi ( { \mathcal { G } } )$ aggregate each node $v \in \mathcal { V } \bar { \mathbf { \eta } } _ { \mathrm { s } }$ information from its local neighborhood $\mathcal { N } ( v )$ and further update its node embedding $H _ { v } ^ { l }$ at the l-th layer. Formally, this process can be expressed as:

$$
H _ { v } ^ { l } = U P D ^ { l } ( H _ { v } ^ { l - 1 } , A G G ^ { l - 1 } ( \{ H _ { u } ^ { l - 1 } : u \in \mathcal { N } ( v ) \} ) )\tag{1}
$$

where $H _ { v } ^ { 0 } = X _ { v } , \ l \ \in \ \{ 1 , \ldots , L \}$ with L denoting the total number of layers. $U P D$ and AGG are two arbitrary differentiable functions to design diverse GNN backbones. For node classification tasks, generally, $H _ { v } ^ { L }$ is fed into a linear classifier $f$ with a softmax function to obtain the final prediction $\hat { Y }$

## B. Model Stealing Attacks against GNNs

Existing model stealing attacks against GNNs can be broadly divided into two categories: query-based and datafree attacks. Query-based attacks assume the adversary can interactively query the victim GNN with partial or full publicly available graphs to receive model responses, while datafree attacks restrict the adversary’s access to any real data, she/he relies on sophisticated data synthesis and optimization techniques to ceaselessly query the victim with crafted graphs.

Query-based attacks. Based on the type of query response accessible to the adversary, existing attacks operate under either soft-label settings, where the victim returns full probability distributions over classes, or hard-label settings, where only the top-1 predicted labels are available. Shen et al. [18] propose the first systematic study of model stealing attacks against inductive GNNs, where the adversary queries the victim with shadow graphs from the same distribution as the training graph and exploits posterior probabilities, node embeddings, and t-SNE projections to train the surrogate. Their followup work [16] further introduces graph contrastive learning and spectral data augmentation to enhance attack performance. However, both methods require access to a substantial shadow dataset, i.e., 30% of the victim training graph, and assume the availability of rich intermediate outputs that are rarely accessible in real-world MLaaS deployments. CEGA [23] proposes an iterative node selection framework that adaptively queries nodes over multiple cycles using historical feedback.

While CEGA reduces query cost, its multi-cycle active querying paradigm requires global graph topology to compute PageRank-based centrality [15], relies on K-means over global node embeddings to measure diversity, and depends on softlabel outputs to estimate prediction uncertainty. Furthermore, its sequential and high-frequency query patterns can easily trigger system-level anomaly detectors [9].

Under the strict hard-label setting, DeFazio et al. [1] pioneer GNN model stealing attacks by introducing graph perturbation techniques on 2-hop subgraphs of each target node to synthesize queries. They empirically find that training the surrogate model with top-1 hard labels yields even higher fidelity than utilizing soft-label outputs. Wu et al. [24] further leverage diverse background knowledge, such as partial features or topologies, or full shadow graphs to synthesize missing information for inaccessible nodes. Guan et al. [5] additionally introduce an edge prediction module to mitigate noise propagation from incorrect predicted labels. However, their method still assumes the presence of relatively connected graph topologies and requires 25% of the training graph.

Data-free attacks. Zhuang et al. [29] train a GAN-style graph generator to synthesize query graphs, which are subsequently used to query the victim and train the surrogate. Despite being data-free, this GAN-style attack framework introduces substantial computational overhead, training instability and high detection risks, while still requiring a massive query budget through iterative generator-victim interactions.

Limitations of prior work. Despite significant progress, existing GNN stealing attacks share several critical bottlenecks that hamper their real-world applicability: (1) Data dependency: most methods assume access to large amounts of or even full victim training graphs even under soft-label settings; (2) Structural over-reliance: acquiring complete multi-hop subgraphs for each target node is often impractical, as it causes unintended topological and feature leakage that exceeds realistic API access permissions; (3) High query overhead and detection risks: query budgets remain prohibitively high for strict black-box deployments, where excessive queries not only incur significant monetary costs but also risk triggering querybased anomaly detection deployed by MLaaS providers; (4) Backbone Reliance: existing GNN stealing attacks commonly assume prior knowledge of the victim’s backbone, which rarely holds in realistic black-box deployments as such knowledge is proprietary and closely guarded by service providers; (5) Evaluation inconsistency: with few exceptions in inductive settings, open-source implementations of the remaining methods indicate that they implicitly query the victim with the full graph while only returning the query responses of target nodes, creating a fundamental discrepancy between claimed threat models and experimental evaluations. In contrast, our work operates under a strictly more constrained yet realistic threat model: black-box hard-label access only, queries restricted to the induced subgraphs of target nodes without full-graph access or complete subgraph information, and with a limited budget of only 5% of the victim training graph.

## III. THREAT MODEL AND EMPIRICAL STUDIES

In this section, we first formalize the threat model and problem statement, establishing a more restricted yet realistic attack setting than prior GNN stealing work. We then conduct three empirical studies to reveal the unique challenges arising from this restricted attack setting. Together, these challenges motivate the holistic design of our framework in Section IV.

## A. Threat Model

1) Attack Goals: In line with the standard taxonomy [8], [16], [18], we consider two adversarial objectives: theft and reconnaissance. A theft adversary aims to construct a surrogate model $\Phi _ { S }$ that achieves comparable task accuracy to the victim model $\Phi _ { V }$ , thereby compromising intellectual property and confidentiality. A reconnaissance adversary instead seeks to replicate the victim’s model behaviors. Ideally, $\Phi _ { S }$ should agree with Φ on any given input, including cases where both models make incorrect predictions. This alignment is quantified by fidelity, where a high-fidelity surrogate serves as a springboard for downstream exploitation, such as crafting transferable adversarial examples.

2) Attacker’s Knowledge and Capabilities: We consider a strict yet realistic black-box and hard-label attack setting. The adversary can interact with the victim solely through a query interface, such as a remotely accessible API, and receive hard-label responses. No inner parameters, backbone prior knowledge, architectural configurations, training procedures or intermediate representations are available.

Unlike prior work that explicitly or implicitly relies on querying full graphs, we strictly restrict query access to the target nodes and their induced subgraph. To align with standard settings that mirror real-world MLaaS deployments [1], [16], [18], [24], the adversary can submit k-hop local subgraphs of target nodes as query inputs. However, distinct from prior formulations where adversaries can acquire complete multihop subgraphs derived from the full victim training graph, our adversary can only construct query graphs within the induced subgraph. On sparse or small-scale graphs, such queries frequently collapse into sets of isolated nodes. This constraint not only prevents the adversary from exploiting global graph structure or cross-boundary neighborhood information that would be inaccessible in practice, but also reduces the risk of triggering rate limits or anomaly detectors [9].

The attack budget is strictly bounded by 5% of the victim training graph. We further prohibit data synthesis to expand queries, as synthetic samples lie outside the natural data manifold and are more vulnerable to distribution-aware defenses [9]. In contrast, querying only real training nodes renders queries indistinguishable from legitimate usage.

3) Attack Scenarios: We outline three representative realworld attack scenarios as follows:

(1) Commercial MLaaS Model Theft: A competitor company/institution queries a commercial GNN API to replicate its underlying functionality at minimal cost, bypassing proprietary data collection and intensive model training.

(2) Low-footprint Adversarial Transfer: An adversary first extracts a surrogate under a strict query budget to evade rate-limiting and anomaly defenses, then crafts transferable adversarial examples to attack the victim [18].

(3) Intellectual Property Audit: An auditor extracts a surrogate model under harsh constraints to assess model vulnerability or verify unauthorized IP infringement, providing critical insights for downstream defense mechanisms.

## B. Problem Statement

Based on the threat model described above, we now formalize our problem as follows.

Given $\mathcal { G } = ( \nu , \mathcal { E } , X )$ , a victim GNN $\Phi _ { V } : { \mathcal { G } } \to \mathbb { R } ^ { | \nu | \times C }$ is trained on $\mathcal { G }$ for the node classification task over $C$ classes and deployed as a black-box service. The adversary has access to the target node set $\mathcal { V } _ { q } \subseteq \mathcal { V }$ subject to a budget constraint:

$$
| \mathcal { V } _ { q } | \leq \beta \cdot | \mathcal { V } |\tag{2}
$$

where $\beta$ is the graph access ratio, we set $\beta = 5 \%$

Notably, the induced subgraph $\mathcal { G } _ { q }$ only covers inner connections among these target nodes $\gamma _ { q } ,$ and it can be largely unconnected with many isolated nodes:

$$
\mathcal G _ { \boldsymbol q } = \mathcal G [ \mathcal V _ { \boldsymbol q } ] = \big ( \mathcal V _ { \boldsymbol q } , \mathcal E \cap ( \mathcal V _ { \boldsymbol q } \times \mathcal V _ { \boldsymbol q } ) , X _ { \mathcal V _ { \boldsymbol q } } \big )\tag{3}
$$

For each queried node $v \in \mathcal { V } _ { q } ,$ the adversary submits its khop local subgraph within the induced subgraph, i.e., $\mathcal { G } _ { v } ^ { k } \subseteq \mathcal { G } _ { q }$ (with $k = 2$ throughout) as the query input, and the victim returns the hard-label prediction $\hat { y } _ { v } ^ { V }$ for the target node.

$$
\mathcal { G } _ { v } ^ { k } = \mathcal { G } _ { q } \Big [ \mathcal { N } _ { \mathcal { G } _ { q } } ^ { k } ( v ) \cup \{ v \} \Big ]\tag{4}
$$

$$
\hat { y } _ { v } ^ { V } = \arg \operatorname* { m a x } _ { c \in \mathcal { C } } \left[ \Phi _ { V } \mathopen { } \mathclose \bgroup \left( \mathcal { G } _ { v } ^ { k } \aftergroup \egroup \right) \right] _ { v , c }\tag{5}
$$

Given only the query-response pairs $\mathcal { D } = \{ ( \mathcal { G } _ { v } ^ { k } , \hat { y } _ { v } ^ { V } ) \} _ { v \in \mathcal { V } _ { q } }$ collected under the above constraints, the adversary seeks to train a surrogate model $\Phi _ { S }$ that simultaneously achieves theft and reconnaissance goals against ground-truth and predicted labels on unseen nodes $\mathcal { V } _ { n e w }$

$$
\mathsf { A c c } ( \Phi _ { S } ) = \frac { 1 } { | \mathcal V _ { n e w } | } \sum _ { v \in \mathcal V _ { n e w } } \mathbf { 1 } [ \hat { y } _ { v } ^ { S } = y _ { v } ]\tag{6}
$$

$$
\mathrm { F i d } ( \Phi _ { S } ) = \frac { 1 } { | \mathcal V _ { n e w } | } \sum _ { v \in \mathcal V _ { n e w } } \mathbf { 1 } [ \hat { y } _ { v } ^ { S } = \hat { y } _ { v } ^ { V } ]\tag{7}
$$

where $y _ { v }$ is the ground-truth label of node v. The inherent bottleneck is that the surrogate $\Phi _ { S }$ must generalize to unseen nodes, despite being trained only on hard labels derived from the potentially sparse and small-scale induced subgraph $\mathcal { G } _ { q } ^ { \mathrm { ~ ~ } } .$

## C. Empirical Studies

Before detailing our attack design, we conduct three empirical studies to investigate the unique challenges imposed by our strict threat model across four representative real-world graph datasets: Cora, PubMed [27], Amazon-Computer (Computer, hereinafter) and Physics [17]. Cora and PubMed are computer science and biomedical citation networks, respectively, where each node denotes a document with a bag-of-words representation, and each edge represents a citation link. Computer and Physics are built upon Amazon co-purchase and academic collaboration networks, where each node signifies a product or an author, each node feature vector encodes a product review or paper keywords, and each edge indicates co-purchases or co-authorship. Dataset statistics are summarized in Table I.

TABLE I: Statistics of the Graph Datasets
<table><tr><td>Dataset</td><td># of Nodes</td><td># of Edges</td><td># of Features</td><td># of Classes</td><td>Avg. Degree</td><td>Node Homophily (%)</td></tr><tr><td>Cora</td><td>2,708</td><td>10,556</td><td>1,433</td><td>7</td><td>3.90</td><td>82.52</td></tr><tr><td>PubMed</td><td>19,717</td><td>88,651</td><td>500</td><td>3</td><td>4.50</td><td>79.24</td></tr><tr><td>Computer</td><td>13,752</td><td>491,722</td><td>767</td><td>10</td><td>35.76</td><td>78.53</td></tr><tr><td>Physics</td><td>34,493</td><td>495,924</td><td>8,415</td><td>5</td><td>14.38</td><td>91.53</td></tr></table>

1) Label Quality Degradation: Under the threat model defined above, the adversary’s query access is doubly constrained: the query budget is strictly limited, and queries are fully restricted to the induced subgraph of the target nodes rather than the complete subgraph in the victim training graph. As a result, the compound effect of these two constraints gives rise to highly sparse subgraphs and even isolated nodes, particularly on sparse or small-scale graphs. This raises a natural question: can GNNs reliably perform inference on such degenerate graph structures? To investigate this, we evaluate GNN prediction quality on 5% randomly sampled target nodes under four topology conditions: isolated nodes (features only), 1-hop and 2-hop induced subgraphs, and the full graph, across all benchmark graphs and diverse GNN backbones.

![](images/c76493a5e2b2ff13856386860a1b657d71334724f71f4bf6cc3c998d3dd44219.jpg)  
Fig. 1: Label Quality Changes across Multi-hop Expansion

Empirical Results. Figure 1 shows that label accuracy generally degrades as the available topology decreases. On relatively sparse graphs such as Cora and PubMed, incrementally expanding topological information does not consistently improve label quality. In fact, 2-hop induced subgraphs occasionally yield lower label accuracy than isolated nodes, which indicates that most 1-hop and 2-hop neighbors within the induced subgraphs belong to different classes, thereby introducing noise that distorts victim predictions.

Takeaways. The above empirical results reveal two fundamental challenges imposed by the strict threat model to design GNN stealing attacks:

(1) C1: Poor Surrogate: the highly sparse induced subgraph with a large number of isolated nodes severely degrades standard GNN message passing mechanism, with few or no neighbors to aggregate, GNNs fail to learn informative node representations that should encapsulate both topological and feature context. This representation collapse fundamentally misaligns with the latent space of the victim model trained on the full graph, leaving the surrogate incapable of faithfully approximating the victim’s behaviors.

(2) C2: Insufficient Supervision: hard-label queries inherently discard inter-class similarity information encoded in soft probabilities, leaving the surrogate with insufficient training signals for generalization. To make matters worse, the victim predictions on isolated nodes and incomplete subgraphs are substantially less accurate than under full-graph inference, further compromising supervision reliability. Together, these factors render the query-response pairs both informationally encapsulated and noisy, posing a fundamental obstacle to surrogate model training.

2) Class Imbalance and Absence: Beyond degrading prediction quality, the sparse structures and isolated nodes identified above introduce a further complication: the resulting label distribution is highly imbalanced, and certain classes may be absent entirely from the queried nodes. We refer to this as the class imbalance and absence problem, and investigate its impact on GNN stealing performance below.

To quantify class imbalance, we define the imbalance ratio as the proportion of samples in the most frequent relative to the least frequent class over ground-truth or predicted labels.

Empirical Results. As illustrated in Figure 2 and 3, victim predictions on the induced subgraphs systematically amplify class imbalance relative to the ground-truth distribution, with imbalance ratios up to 45× on Cora and 70× on Computer when using GAT as the victim backbone, far exceeding the ground-truth baselines. In contrast, PubMed displays a relatively balanced prediction distribution, where victim imbalance ratios remain slightly below the ground-truth baseline across all backbones. While variations among victim backbones retain minor on Physics, they become substantial on other graphs with larger class space. More critically, when using APPNP as the victim backbone on Cora, class 6 is missing and class 1 only contains a single target node. Such extreme class imbalance and missing classes severely bias the surrogate model toward majority classes, leaving sparse or absent training signals for minority classes.

Takeaways. These empirical findings highlight another fundamental challenge–C3: Class Imbalance and Absence. Querying hard labels on sparse induced subgraphs inherently exacerbates class imbalance: structurally dominant classes capture a disproportionate share of target predictions, while minority classes remain severely under-represented or entirely absent from the query set. This systematic bias consequently skews the surrogate model’s classifier toward majority classes, further degrading stealing performance beyond what structural sparsity alone would cause.

![](images/1248a894ab027d327fe9d7968c29b78f7f0ec7a09c7f397ac780c4fa7c3f735e.jpg)

![](images/ef753a4ceaa7aec31cdbcdafda2e99e08ed04dff609195fea07ccc8cf616c12c.jpg)

![](images/a154fc6c04e4c8ea6f06a20885bae031917569b93c18517865743aeadbaf47ff.jpg)  
Fig. 2: Class Imbalance Ratio

![](images/e9dbfbffb426eabced03641a9ff96e70434ece20b27bc256e9c594222b0c88de.jpg)

3) Backbone Disagreement: Existing GNN model stealing attacks typically assume that the adversary masters the knowledge of victim backbone [1], [18], [23], [24]. However, this information is rarely available in practice due to commercial intellectual property protection and the black-box nature of MLaaS APIs. To understand the impact of victim backbone misalignment, we evaluate the pairwise prediction agreement across different victim backbones under our strict setting and present the results in Figure 4.

![](images/a53e5e157ffbd325e14f323dd2fd1c8bc87c0305ec8dfc85c85e9f0cbd1228e3.jpg)  
Fig. 3: Per-class Label Distribution

Empirical Results. We observe that the victim prediction agreement is substantially lower on sparse induced subgraphs than on the full graph, with off-diagonal entries as low as 0.51 on Computer for isolated nodes. This indicates that different backbones exhibit divergent behaviors as structural information decreases, making cross-backbone model stealing inherently more challenging under our restricted threat model. The progressive increase in agreement from isolated to full graphs confirms that topology serves as the primary driver of backbone divergence.

Takeaways. These observations underscore a key challenge– C4: Backbone Mismatch. Backbone mismatch further compounds the aforementioned challenges, as different backbones exhibit varying degrees of prediction degradation under the same induced subgraph queries. However, our threat model assumes a practical black-box adversary with no prior knowledge of the victim’s backbone, necessitating an attack framework that remains strictly backbone-agnostic.

## IV. ATTACK FRAMEWORK DESIGN

In this section, we propose Dagger, a novel decouplingbased attack framework tailored to the four challenges identified in Section III. Specifically, C1 (Poor Surrogate) demands that an effective surrogate must remain topology-adaptive, preserving rich semantic context for isolated nodes while extracting structural information across subgraphs of varying scale and density; C2 (Insufficient Supervision) requires maximizing the utility of collected query-response pairs while filtering or smoothing unreliable predictions to prevent error propagation; C3 (Class Imbalance and Absence) necessitates incorporating class-aware recalibration to mitigate systematic biases and ensure balanced classifier optimization across minority classes; and C4 (Backbone Mismatch) calls for the attack framework to remain strictly backbone-agnostic.

As illustrated in Figure 5, Dagger proceeds in two phases. In Phase 1, to tackle C1 and C2, we train a surrogate encoder via decoupling-based feature propagation and manifold-level mixup. This design mitigates the structural collapse of sparse induced subgraphs and enriches context-deficient node representations without over-relying on message passing. In Phase

![](images/65f8a1c42026c24bc51b8a752cdd7cce279a7bb68b2b11e968e7775aaffe3e1b.jpg)  
Fig. 4: Pairwise Prediction Agreement Changes across Multi-hop Expansion

![](images/5b2926cb509a465f0346f705b086130a994effc4ad0987dca7e3a04787821749.jpg)  
Fig. 5: The Overview of Dagger

2, to overcome C3, we freeze the pre-trained encoder and tune only the classifier head using class-balanced sampling paired with logit adjustment, rectifying predicted-label class imbalance and ensuring backbone-agnostic generalization. Notably, by decoupling the encoder from the classifier and avoiding any assumption about the victim’s backbone knowledge throughout both phases, Dagger inherently satisfies the backbone-agnostic requirement imposed by C4.

## A. Phase 1: Decoupling-based Surrogate Pretraining

1) Topology-adaptive Surrogate (C1): Due to the double constraints of query access under our threat model, the query subgraphs $\{ \bar { \mathcal { G } } _ { v } ^ { k } \} _ { v \in \mathcal { V } _ { \varsigma } }$ suffer from three compounding structural deficiencies. First, they can be highly sparse and frequently unconnected, particularly on sparse or small-scale graphs, causing standard message-passing GNNs trained on such degenerative structures to degrade towards Multi-Layer Perceptrons (MLPs). Second, nodes with originally high degrees may become isolated once their neighbors are excluded from the target node set, causing GNNs to lose the ability to extract information from their multi-hop neighbors. Third, even among connected nodes within $\{ \mathcal { G } _ { v } ^ { k } \} _ { v \in \mathcal { V } _ { q } }$ , their scope of information propagation is confined to a small local range, which makes it difficult to integrate broader structural context.

These structural deficiencies motivate a topology-adaptive surrogate that simultaneously handles both isolated nodes and highly-sparse local structures without over-relying on message passing. To this end, we adopt the decoupling design of APPNP [4]: predict and then propagate, separating feature transformation from graph propagation into two explicit steps. For each target node v, we first utilize an MLP to extract the initial representation $H _ { v } ^ { ( 0 ) }$ from raw features in local subgraphs $\mathcal { G } _ { v } ^ { k }$ , independent of its local topology. This prediction step obtains a meaningful representation for isolated nodes rather than collapsing to uninformative embeddings while laying the foundation for subsequent structural information propagation among connected ones.

$$
H _ { v } ^ { ( 0 ) } = \mathbf { M } \mathbf { L } \mathbf { P } ( X _ { \mathcal { G } _ { v } ^ { k } } )\tag{8}
$$

We then leverage Personalized PageRank (PPR) [4], [21] to propagate $H _ { v } ^ { ( 0 ) }$ across K steps, retaining an α fraction of the original representation at each step to adaptively incorporate local structural context while prevent over-smoothing:

$$
H _ { v } ^ { ( k ) } = ( 1 - \alpha ) \hat { A } _ { v } H _ { v } ^ { ( k - 1 ) } + \alpha H _ { v } ^ { ( 0 ) } , \quad k = 1 , \ldots , K\tag{9}
$$

where $\hat { A } _ { v }$ is the adjacency matrix of $\mathcal { G } _ { v } ^ { k }$ with self-loops. Mechanistically, $H _ { v } ^ { ( k ) }$ acts as an information origin while α serves as a restart probability to avoid diluting personalized representations over multiple hops, thereby effectively mitigating over-smoothing and enabling deep information propagation.

Standard message-passing GNNs implicitly rely on the node homophily assumption that connected nodes tend to share similar features and structural patterns [13]. To accommodate isolated nodes during PPR diffusion, we explicitly add selfloops to these nodes. Since a node exhibits maximum similarity with itself, the addition of self-loops transforms the diffusion mechanism into a self-reinforcing process that guarantees stable feature propagation and prevents representation collapse.

2) Knowledge Distillation Loss: Given the query-response pairs $\mathcal { D } = \{ ( \mathcal { G } _ { v } ^ { k } , \hat { y } _ { v } ^ { V } ) \} _ { v \in \mathcal { V } _ { q } }$ , we train the surrogate by minimizing the cross-entropy loss between its predictions on the target node v and the victim’s hard-label predictions:

$$
\mathcal { L } _ { \mathrm { d i s t i l l } } = - \frac { 1 } { | \mathcal { V } _ { q } | } \sum _ { v \in \mathcal { V } _ { q } } \log \frac { \exp \Bigl ( \left[ \Phi _ { S } ( \mathcal { G } _ { v } ^ { k } ) \right] _ { v , \hat { v } _ { v } ^ { V } } \Bigr ) } { \sum _ { c \in \mathcal { C } } \exp \Bigl ( \left[ \Phi _ { S } ( \mathcal { G } _ { v } ^ { k } ) \right] _ { v , c } \Bigr ) }\tag{10}
$$

where $\left[ \Phi _ { S } ( \mathcal { G } _ { v } ^ { k } ) \right] _ { \tau }$ denotes the logit vector of the center node v in the output of the surrogate $\Phi _ { S }$

3) Manifold Mixup Regularization (C2): A further challenge arises from incomplete class coverage: queries restricted to local subgraphs $\{ \mathcal { G } _ { v } ^ { k } \} _ { v \in \mathcal { V } _ { q } }$ may fail to elicit predictions for all |C| classes from the victim model, particularly on sparse or small-scale graphs with multiple classes. Beyond this, hard-label supervision discards inter-class relationships encoded in the victim’s soft-label predictions. Training on such sparse supervision forces the surrogate to overfit to the limited query space and learn spurious decision boundaries, severely hampering its generalization to unseen nodes.

To bridge these gaps, we apply manifold-level node mixup to synthesize continuous interpolated representations across class boundaries. We finally smooth the decision boundary in the manifold rather than the input space. Interpolating raw node features is ill-defined for graph-structured data: node features are typically sparse and discrete $( \mathrm { e . g . }$ , bag-of-words), rendering their linear combinations semantically meaningless, while input-space interpolation disregards the underlying non-Euclidean graph topology. In contrast, manifold interpolation directly operates on a continuous latent space that inherits the structural context encoded by PPR propagation and encourages the surrogate to learn more transferable representations.

We then formalize the manifold-level node mixup. For each query node $v _ { i }$ with predicted-label $\hat { y } _ { v _ { i } } ^ { V } = c _ { a }$ , we sample another query node $v _ { j }$ with a different predicted-label $\hat { y } _ { v _ { j } } ^ { V } = c _ { b }$ , where $( v _ { i } , c _ { a } )$ and $( v _ { j } , c _ { b } )$ both belong to the query-response pairs $\mathcal { D } = \{ ( \mathcal { G } _ { v } ^ { k } , \hat { y } _ { v } ^ { V } ) \} _ { v \in \mathcal { V } _ { q } } , ~ i \neq j , \bar { c _ { a } } \neq c _ { b } ,$ and interpolate their focal-node hidden representations $H _ { v _ { i } }$ and $H _ { v _ { j } }$

$$
\tilde { H } _ { v _ { i j } } = \lambda \cdot H _ { v _ { i } } + ( 1 - \lambda ) \cdot H _ { v _ { j } } , \quad \lambda \sim \mathrm { U n i f o r m } ( 0 , 1 )\tag{11}
$$

Unlike the feature-level node mixup [22] which employs Beta distributions to concentrate mixing ratios near the original samples, we adopt $\lambda \sim$ Uniform(0, 1) to uniformly explore the full interpolation spectrum between class boundaries. This setup particularly aligns with our attack setting: since node representations reside on a continuous semantic manifold rather than a discrete input space, dense and uniform sampling across (0, 1) synthesizes a richer variety of intermediate representations, providing more comprehensive coverage of the decision boundary between $c _ { a }$ and $c _ { b }$

The mixed representation $\tilde { H } _ { v _ { i i } }$ is trained against a paired soft label $\tilde { y } _ { i j } = \lambda \cdot { \bf e } _ { c _ { a } } + ( 1 - \lambda ) \cdot { \bf e } _ { c _ { b } }$ via KL divergence:

$$
\mathcal { L } _ { \mathrm { m i x } } = \frac { 1 } { \left| \mathcal { V } _ { q } \right| } \sum _ { v _ { i } \in \mathcal { V } _ { q } } D _ { \mathrm { K L } } \Big ( \tilde { y } _ { i j } \left| \right| \mathrm { s o f t m a x } \Big ( f _ { S } ( \tilde { H } _ { v _ { i j } } ) \Big ) \Big )\tag{12}
$$

where e is a one-hot vector and $f _ { S }$ denotes the linear classifier head of $\Phi _ { S }$ . The soft labels $\tilde { y } _ { i j }$ further complement inter-class relationships that are discarded by hard-label supervision.

4) Phase Objective: The final objective of Phase 1 is:

$$
\mathcal { L } _ { \mathrm { P h a s e \ 1 } } = \mathcal { L } _ { \mathrm { d i s t i l l } } + w _ { \mathrm { m i x } } \cdot \mathcal { L } _ { \mathrm { m i x } }\tag{13}
$$

where the coefficient $w _ { \mathrm { m i x } }$ controls the contribution of manifold-level node mixup.

## B. Phase 2: Imbalance-aware Classifier Tuning

After pretraining in Phase 1, the surrogate has learned a representation space that captures the victim’s functional behaviors. While manifold-level node mixup mitigates the class imbalance by implicitly augmenting minority class representations, it operates at the representation level and does not directly correct the skewed decision boundaries induced by majority-class dominance in the training distribution. As a result, the classifier head remains systematically biased towards majority-class nodes, leaving minority classes underrepresented and consequently eroding the overall fidelity of the surrogate. To rectify this bias, Phase 2 freezes the surrogate encoder $\operatorname { E n c } _ { S } ( \cdot )$ to preserve the learned representations and only fine-tunes the linear classifier head $f _ { S }$ with two complementary corrections: class-balanced sampling and logit adjustment [14].

1) Class-balanced Sampling (C3): To counteract the majority class dominance in the query-response pairs D, we oversample minority classes with replacement such that each class provides exactly the same number of nodes per epoch as the majority class. The sampling probability is defined as:

$$
p ( v ) = \frac { 1 } { | C | \cdot n _ { c } } , \quad v \in \mathcal { V } _ { q } ^ { ( c ) }\tag{14}
$$

where $n _ { c } = | \mathcal { V } _ { q } ^ { ( c ) } |$ is the number of query nodes for class $c .$

This sampling strategy balances per-class gradient contributions, preventing the classifier from collapsing to majorityclass predictions and ensuring that minority classes receive sufficient gradient signals throughout fine-tuning.

2) Logit Adjustment (C3): Class-balanced sampling equalizes per-class gradient contributions and stabilizes optimization. However, oversampling minority classes inherently alters the class prior from the skewed query distribution to a uniform one, inevitably introducing a distribution shift. To mitigate this discrepancy without resorting to complex optimization, we adopt a simple-yet-effective technique called logit adjustment [14] to calibrate the classifier logits proportional to log $\pi _ { c } .$

$$
\tilde { f } _ { S } ( H ) _ { c } = f _ { S } ( H ) _ { c } - \tau \cdot \log \pi _ { c }\tag{15}
$$

where $\pi _ { c } = n _ { c } / | \mathcal { V } _ { q } | , c \in C$ denotes the empirical class prior estimated from the victim-predicted label distribution, while $\tau > 0$ is a temperature parameter controlling the strength of the adjustment. This technique is theoretically grounded: Menon et al. [14] show that under the adjusted loss, the optimal classifier recovers the Bayes-optimal decision rule for the balanced distribution, effectively redistributing decision boundaries from majority to minority classes without requiring additional queries to the victim.

3) Phase Objective: The Phase 2 training objective is:

$$
\mathcal { L } _ { \mathrm { P h a s e } 2 } = - \frac { 1 } { | \mathcal { V } _ { q } | } \sum _ { v \in \mathcal { V } _ { q } } \log \frac { \exp \Bigl ( \tilde { f } _ { S } ( H _ { v } ^ { S } ) _ { \hat { y } _ { v } ^ { V } } \Bigr ) } { \sum _ { c = 1 } ^ { C } \exp \Bigl ( \tilde { f } _ { S } ( H _ { v } ^ { S } ) _ { c } \Bigr ) }\tag{16}
$$

where $H _ { v } ^ { S } { = } \mathrm { E n c } _ { S } ( v )$ denotes the frozen focal-node embedding from Phase 1. These two corrections are complementary by design: class-balanced sampling stabilizes training by equalizing per-class gradient contributions, while logit adjustment restores proper calibration during inference by explicitly compensating for the prior shift induced by rebalancing.

## V. EXPERIMENTS

In this section, we conduct empirical experiments to investigate the following key research questions:

RQ1: How effective and efficient is Dagger to launch model stealing attacks against GNNs?

RQ2: Can Dagger flexibly adapt to different GNN backbones? RQ3: How Dagger performs under diverse mismatched architecture configurations?

RQ4: What factors affect the attack performance of Dagger?

RQ5: How does each component contribute to Dagger?

RQ6: Can Dagger bypass diverse defense mechanisms?

## A. Setups

GNN Backbones We employ four GNN backbones: GCN [10], GAT [20], APPNP [4], and GraphSAGE [6], and report their benign performance across aforementioned four graphs in Table V in Appendix B.

Baselines. To quantify the contribution of real graph structural information to attack performance, we introduce a synthetic graph baseline (Rand.) that requires no access to real graph data. We generate a Stochastic Block Model (SBM) graph with |C| equal-sized blocks, intra-block edge probability $p _ { \mathrm { i n t r a } } = 0 . 3 ,$ and inter-block edge probability $p _ { \mathrm { i n t e r } } = 0 . 0 5 .$ , with the total number of nodes matching the 5% query budget of each dataset. Node features are constructed by randomly sampling rows from the real graph’s feature matrix with replacement and adding small Gaussian perturbations $\mathcal { N } ( 0 , 0 . 0 1 ^ { \frac { 1 } { 2 } } )$ , preserving the marginal feature distribution of the target dataset while decoupling the feature-topology correlations present in the real graph. We further compare Dagger with the state-of-the-art (SOTA) GNN stealing attacks, under the same access to 5% of the victim training graph, including 1) MEA-Attack0 [24]: to align with its setting, the adversary can obtain 2-hop neighbors of each target node. She/he then synthesizes the neighboring attributes via homophily-based neighbor expansion. 2) AdvMEA [1]: the adversary constructs fully-connected synthetic subgraphs by sampling node features from per-class empirical distributions from the whole graph. 3) CEGA [23]: the adversary employs an iterative active learning strategy that adaptively selects informative nodes based on structural centrality, uncertainty, and diversity across multiple query cycles. We additionally replace its soft-label setting with hard labels. 4) DFEAII-Real [29]: since the original data-free variant suffers from prediction collapse, we follow the benchmarking protocol of [28] and instead query the victim on real induced subgraphs under the black-box hard-label setting. Please note that except for DFEAII-Real, MEA-Attack0 requires substantially higher query budgets than Dagger, whereas AdvMEA and CEGA rely on access to global information that are unavailable in practice.

Implementations. Experiments are conducted on four Nvidia A100 GPUs. Please see Appendix A for more details.

Metrics. Except for accuracy and fidelity defined in Section III-B, we also consider 1) Macro F1 that evaluates all classes with equal weight, making it highly sensitive to performance degradation on minority classes; 2) Boundary fidelity that measures the prediction agreement between the surrogate and the victim on boundary nodes, defined as nodes where the victim’s margin between its top two predicted probabilities falls below a threshold δ = 0.2. We empirically select 0.2 as it achieves a better trade-off between the boundary space and number of samples located in this area. This metric captures the surrogate’s ability to replicate the victim’s decisions in the most challenging regions, where small perturbations can flip the predicted labels. 3) Number of queries, which is different from the number of accessible nodes. The adversary can adopt active learning or data synthesis and augmentation methods to reduce or expand the query node set.

## B. Attack Performance (RQ1 and RQ2)

TABLE II: The Performance of Dagger on Cora and Computer. Unit of Top 4 metrics: 1e-2. Arrow indicates the direction of better performance and the bold font denotes the ‘best’ results.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Baseline</td><td colspan="5">Metrics</td></tr><tr><td>ACC</td><td>FID</td><td>Macro F1</td><td>Boundary Fid.</td><td>Query</td></tr><tr><td rowspan="20">Cora</td><td rowspan="5">GCN</td><td>Rand</td><td>26.57±0.00 61.50±2.63</td><td>27.31±0.00</td><td>6.00±0.00 53.08±2.37</td><td>10.00±0.00 37.62±4.10</td><td>135 686</td></tr><tr><td>MEA</td><td>40.22±3.95</td><td>64.33±2.30</td><td>28.47±6.43</td><td>19.05±7.59</td><td>315</td></tr><tr><td>AdvMEA</td><td>28.66±5.15</td><td>42.19±5.77</td><td>19.32±3.13</td><td></td><td>133</td></tr><tr><td>CEGA</td><td></td><td>28.41±5.25</td><td></td><td>13.33±2.69</td><td></td></tr><tr><td>DFEAII-Real Dagger</td><td>29.64±3.87 71.83±0.46</td><td>29.64±4.11</td><td>13.87±3.51</td><td>10.48±0.67</td><td>135</td></tr><tr><td></td><td></td><td>73.68±0.17</td><td>59.22±1.20</td><td>32.86±1.17</td><td>135</td></tr><tr><td rowspan="5">GAT</td><td>Rand</td><td>28.78±3.13</td><td>29.52±3.65</td><td>8.43±3.31</td><td>11.11±3.93</td><td>135</td></tr><tr><td>MEA</td><td>59.66±2.44</td><td>62.85±3.22</td><td>51.99±2.97</td><td>31.94±1.96</td><td>686</td></tr><tr><td>AdvMEA</td><td>21.40±5.74</td><td>23.49±5.70</td><td>9.93±1.12</td><td>20.83±6.13</td><td>315</td></tr><tr><td>CEGA</td><td>36.29±7.36</td><td>38.25±8.45</td><td>21.88±8.14</td><td>22.22±9.67</td><td>133</td></tr><tr><td>DFEAII-Real</td><td>32.23±1.91</td><td>33.83±2.85</td><td>16.76±4.94</td><td>15.28±8.39</td><td>135</td></tr><tr><td rowspan="6">APPNP</td><td>Dagger</td><td>74.28±1.25</td><td>76.26±0.70</td><td>67.08±1.25</td><td>36.11±0.98</td><td>135</td></tr><tr><td>Rand</td><td>26.57±0.00</td><td>30.26±0.00</td><td>6.00±0.00</td><td>16.22±0.00</td><td>135</td></tr><tr><td>MEA</td><td>49.57±6.94</td><td>56.83±6.65</td><td>34.15±9.24</td><td>38.74±2.78</td><td>686</td></tr><tr><td>AdvMEA</td><td>27.31±1.09</td><td>31.00±1.09</td><td>7.64±1.47</td><td>15.77±0.64</td><td>315</td></tr><tr><td>CEGA</td><td>26.94±9.64</td><td>31.49±10.80</td><td>11.87±5.86</td><td>20.72±8.57</td><td>133</td></tr><tr><td>DFEAII-Real Dagger</td><td>31.73±5.52 67.40±0.92</td><td>35.18±6.51</td><td>15.93±3.29</td><td>18.47±2.30</td><td>135</td></tr><tr><td rowspan="6">GraphSAGE</td><td>Rand</td><td>26.57±0.00</td><td>74.29±1.06</td><td>51.69±0.74</td><td>41.89±1.91 17.14±0.00</td><td>135 135</td></tr><tr><td></td><td>64.58±1.83</td><td>26.20±0.00</td><td>6.00±0.00</td><td></td><td></td></tr><tr><td>MEA</td><td>31.24±8.44</td><td>67.28±2.72</td><td>57.30±2.48</td><td>42.86±4.67</td><td>686</td></tr><tr><td>AdvMEA</td><td>45.26±2.26</td><td>32.23±8.14</td><td>22.40±8.42</td><td>20.95±4.86</td><td>315</td></tr><tr><td>CEGA</td><td>50.68±3.31</td><td>45.26±2.05</td><td>38.87±5.55</td><td>26.67±11.74</td><td>133</td></tr><tr><td>DFEAII-Real Dagger</td><td>76.51±0.92</td><td>51.91±3.22</td><td>39.54±2.37</td><td>28.57±2.33</td><td>135</td></tr><tr><td rowspan="8">GCN</td><td></td><td>41.13±0.00</td><td>76.01±1.09 43.31±0.00</td><td>71.91±1.53 5.83±0.00</td><td>39.05±1.35 28.21±0.00</td><td>135 687</td></tr><tr><td>Rand MEA</td><td>76.24±1.61</td><td>82.66±1.81</td><td>47.60±3.91</td><td>47.86±5.37</td><td>8401</td></tr><tr><td>AdvMEA</td><td>62.52±4.39</td><td>67.61±5.36</td><td>43.50±6.25</td><td>42.74±7.72</td><td>480</td></tr><tr><td>CEGA</td><td>70.18±0.77</td><td>73.16±0.84</td><td>69.95±1.20</td><td>42.74±6.97</td><td>690</td></tr><tr><td>DFEAII-Real</td><td>77.96±1.89</td><td></td><td></td><td></td><td>687</td></tr><tr><td>Dagger</td><td>81.73±0.71</td><td>84.38±1.93</td><td>67.26±4.38</td><td>50.43±5.16 60.26±3.63</td><td></td></tr><tr><td>Rand</td><td>35.76±3.15</td><td>87.40±0.71 36.00±2.87</td><td>67.52±1.42</td><td></td><td>687</td></tr><tr><td>MEA</td><td>52.57±1.52</td><td>54.68±1.93</td><td>9.96±1.01 21.81±3.27</td><td>20.67±7.36</td><td>687</td></tr><tr><td rowspan="6">GAT Computer</td><td>AdvMEA</td><td></td><td></td><td></td><td>32.00±1.63</td><td>8401</td></tr><tr><td></td><td>33.48±11.20</td><td>34.81±11.42</td><td>10.05±2.12</td><td>23.33±7.36</td><td>480</td></tr><tr><td>CEGA</td><td>54.29±5.74</td><td>55.60±5.45</td><td>40.11±6.37</td><td>26.67±3.77</td><td>690</td></tr><tr><td>DFEAII-Real</td><td>50.12±3.72</td><td>52.42±3.89</td><td>18.11±5.63</td><td>33.33±0.94</td><td>687</td></tr><tr><td>Dagger</td><td>71.46±0.22</td><td>72.84±0.24</td><td>50.19±1.31</td><td>31.33±0.94</td><td>687</td></tr><tr><td>Rand</td><td>52.01±0.30</td><td>53.15±0.33</td><td>14.37±0.13</td><td>19.44±0.49</td><td>687</td></tr><tr><td rowspan="6">APPNP</td><td>MEA</td><td>74.61±1.21</td><td>79.75±1.22</td><td>47.81±9.98</td><td>43.06±1.30</td><td>8401</td></tr><tr><td>AdvMEA</td><td>55.79±5.41</td><td>57.19±5.59</td><td>45.71±8.80</td><td>29.17±3.07</td><td>480</td></tr><tr><td>CEGA</td><td>51.09±12.42</td><td>50.85±12.91</td><td>51.65±10.61</td><td>28.82±4.28</td><td>690</td></tr><tr><td>DFEAII-Real</td><td>64.17±5.32</td><td>67.71±6.62</td><td>33.85±1.23</td><td>38.54±3.71</td><td>687</td></tr><tr><td>Dagger</td><td>84.33±0.36</td><td>88.61±0.35</td><td>77.75±2.04</td><td>44.10±1.30</td><td>687</td></tr><tr><td>Rand</td><td>41.13±0.00</td><td>41.86±0.00</td><td>5.83±0.00</td><td>27.42±0.00</td><td>687</td></tr><tr><td rowspan="5">GraphSAGE</td><td>MEA</td><td>76.11±3.13</td><td></td><td>59.54±5.41</td><td></td><td></td></tr><tr><td></td><td></td><td>80.01±3.62</td><td></td><td>45.70±3.31</td><td>8401</td></tr><tr><td>AdvMEA</td><td>48.96±20.16</td><td>49.78±21.37</td><td>40.31±19.11</td><td>33.87±3.95</td><td>480</td></tr><tr><td>CEGA</td><td>75.24±3.81</td><td>76.84±3.78</td><td>72.62±5.15</td><td>38.71±4.75</td><td>690</td></tr><tr><td>DFEAII-Real</td><td>61.17±14.23</td><td>64.00±15.71</td><td>37.87±22.66</td><td>39.78±8.77</td><td>687</td></tr><tr><td></td><td>Dagger</td><td>80.62±0.33</td><td>86.36±0.48</td><td>72.36±1.21</td><td>47.31±2.01</td><td>687</td></tr></table>

1) Effectiveness: Due to space limitations, we put attack results of the rest two datasets in Appendix C. As shown in Table II and VI, Dagger consistently outperforms all baselines in nearly every metric under the most stringent threat model, demonstrating the effectiveness of our framework. It even surpasses or achieves comparable attack performance against MEA, which usually submits queries 3.87-12.23× than Dagger. In terms of graph datasets, when using a similar number of queries, Dagger largely improves accuracy and fidelity over the SOTA baselines by up to 43.17% and 45.27%, respectively. It excels in graph datasets with much larger class spaces, like Cora and Computer. As for victim backbones, there is no single backbone inherently weaker or stronger against Dagger. When using APPNP as the victim backbone and trained on Cora, the query nodes cannot cover all classes, yielding relatively lower but still effective attack performance. When using GAT as the victim backbone trained on Computer, the query nodes are extremely imbalanced, causing all the baselines and Dagger to degrade compared to other victim backbones. These two anomalies are also reflected in macro F1 scores. Regarding boundary performance, Dagger also achieves outstanding boundary fidelity across most settings, indicating that the surrogate not only replicates the victim’s confident predictions but also approximates its behaviors near decision boundaries. This is particularly relevant for downstream reconnaissance tasks such as crafting transferable adversarial examples [18].

2) Efficiency: A key advantage of Dagger lies in its query efficiency. Dagger strictly requires 5% nodes of the full graph and the number of queries is the exactly the same as the number of accessible nodes. MEA demands many more queries than all other baselines. Random generation and DFEAII-Real maintain the same query budgets as Dagger. The number of queries of AdvMEA and CEGA vary, AdvMEA possesses a larger query budget on Cora while a smaller budget than Dagger on the remaining graph datasets; CEGA owns the similar query budget to Dagger, it only increases queries compared to Dagger on Computer. Overall, Dagger is not only more attack-effective but also substantially more queryefficient. This efficiency is critical in practice, as a lower query footprint reduces the risk of detection by query-monitoring defenses [9].

## C. Impact Factors (RQ3 to RQ5)

1) GNN Architectures: Unlike existing stealing attacks, Dagger imposes no restrictions on requiring the same/similar GNN backbones or architecture configurations. Given that other graph datasets are relatively less prone to imbalance and class coverage issues while attaining compelling attack performance, we select Cora as the representative dataset for architecture sensitivity studies. In addition, GNNs typically employ shallow architectures to mitigate over-smoothing, a phenomenon where node representations become indistinguishable after multiple-round information propagation. Following standard settings [10], [20], [29], we set the number of layers as 2 or 3. As the computational complexity of GNN backbones scales linearly or quadratically with the hidden dimension and excessively large dimension might lead to overfitting problems especially on small-scale and sparse induced subgraphs obtained under limited query budgets, we evaluate the surrogate performance under compact hidden dimensions $d \in \{ 1 6 , \bar { 3 } 2 , 6 \bar { 4 } \}$

From Figure 6, Dagger remains stable to victim backbones and architecture configurations across all metrics, with performance variations remaining within approximately 1-4%, confirming that the framework does not rely on a carefully tuned surrogate capacity to achieve competitive results. Comparing 2-layer and 3-layer configurations at the same hidden dimension, deeper architectures consistently perform on par with or slightly below their 2-layer counterparts. This suggests that additional depth does not provide meaningful benefit under the 5% budget constraint. Increasing hidden dimension from 16 to 64 within the same depth also yields marginal changes, further indicating that the bottleneck lies in the quality and quantity of victim-predicted labels rather than model capacity. The default configuration [2, 16] achieves attack performance comparable to all larger architectures, even outperforming several specific victim backbones, such as GCN and APPNP.

These empirical findings indicate a desirable attack property in practice: the adversary does not need prior knowledge of victim backbones or capacities, and a minimal architecture suffices to launch powerful GNN stealing attacks.

2) Manifold: As shown in Figure 7, Dagger demonstrates strong robustness to the manifold coefficient $w _ { \mathrm { m i x } }$ across the range [0.1, 2.0], with performance fluctuations varying across graph datasets and victim backbones. On Physics and PubMed, performance curves are nearly flat across all victim backbones, indicating that the manifold regularization strength has negligible impact on larger and denser graphs with relatively smaller classes. On Cora, mild fluctuations are observed particularly in boundary fidelity, consistent with the fact that the induced subgraph is sparser and decision boundary regions are harder to approximate. Computer exhibits a moderate pattern, remaining largely stable with only a slight decline at larger coefficients. Consistent with previous findings, APPNP on Cora and GAT on Computer degrade due to class coverage and heavy imbalance issues. In all, there is no specific optimal manifold coefficient range, we accordingly tune this coefficient on different graph-backbone combinations.

3) Query Budget: Figure 8 demonstrates that the attack performance of Dagger consistently improves as the query budget increases from 1% to 20% across all datasets and victim backbones, confirming that more query nodes provide richer supervision for the surrogate. Notably, Dagger already achieves competitive performance at the 5% budget used in our main experiments, with diminishing returns beyond 10% on most datasets. On Physics and PubMed, 1% query budgets already yield competitive attack performance. Their accuracy, fidelity and F1 curves flatten around 5-10% and remain stable through 20%, suggesting that even a small fraction of training nodes is sufficient to capture the victim’s inner behaviors on these datasets. On Cora and Computer, the improvement from 1% to 5% is more pronounced, reflecting the greater difficulty introduced by sparser induced subgraphs and more severe class imbalance at lower budgets. The boundary fidelity curves are more unstable than the other metrics, particularly on Computer, PubMed and Physics, we attribute this higher variance to the small number of boundary nodes in these datasets, where the victim model achieves higher overall confidence, leaving fewer nodes near the decision boundary and making the metric more susceptible to statistical fluctuation. Nevertheless, a clear upward trend is observed on Cora as budget increases.

4) Balance and Its Impact Factors: Due to space limitations, details are provided in Appendix D.

5) Ablation Studies: To systematically assess the contribution of each component in Dagger, we conduct a series of ablation studies across graph datasets and victim backbones. The ablation results confirm the effectiveness of each component in Dagger, and the full model achieves the best or near-best performance in the majority of graph-backbone combinations.

In the Decoupling-based Surrogate Training phase, since the knowledge distillation loss is indispensable for model stealing attacks, we instead remove the manifold mixup regularization module (No Manifold). In the Imbalance-aware Classifier Tuning phase, where rebalancing is essential, we retain the class-balanced sampling and remove logit adjustment (No LA). We also evaluate a variant that retains only Phase 1 while omitting Phase 2 (No Tuning). The ablation results are presented in Table III and VII.

No Manifold consistently degrades performance across Cora and Computer, with the most noticeable drops observed in GAT and GraphSAGE. The degradation on PubMed and Physics is less pronounced, where the larger and relatively balanced query set, denser graph structure, and smaller class space already provide sufficient supervision signals, making the additional mixup regularization less critical.

![](images/45eeed975d35df625cf36f4e507f91b1d0fd649ac6e41fbdb6945d813f230bcd.jpg)

![](images/92676c76da4544ea84fbfd0f5ceb33313ec6e534b61a70af2eab21a85fad503b.jpg)  
Architecture Sensitivity  
Fig. 6: GNN Architecture Configuration Sensitivity

![](images/a48c925a1a6d73d0382da59e4c2e2f07cf0e73fef7388c37a867772abb567d44.jpg)

![](images/766ba4d13573d15459f3dfe391115addadfc8cfff1b4971f8b2b98c3d311eb32.jpg)

![](images/9deaf6c0005a4c921ee882cd13f01eef7c2d0b0a592b9d50643b1e613a3f9a9a.jpg)

Manifold Coefficient Sensitivity — Accuracy  
![](images/2b5e9300d0ff99d0674a4600d11a29c49dcef3276ba7c7ff4c0116831cfcfc9e.jpg)

![](images/0ea2401ccb629dfcf5498dbc89f78027e0ab9e11439372264c71bda284a62d44.jpg)

![](images/520eadfb2e8f8740a871bc087de3117d43e8a682b606b90efb311321e6c9ab0c.jpg)

![](images/642792c7643f61864771771e20889f207507cc3e886e742ab4dfb0284e8c0011.jpg)

Manifold Coefficient Sensitivity — Fidelity  
![](images/d0fa15244a24ecc5272f145eb8307204ac620d61238b36bc2a3c456a95cafce9.jpg)

![](images/e150753774284e081f652117fb080c94e54ffb49d2dca88c59239f6e3728f744.jpg)

![](images/98acd33ea759686c78419adc8c4395982e678160a2b0a47b666e00bcd2ce3558.jpg)

Manifold Coefficient Sensitivity — Macro F1  
![](images/b0106caab5bd28feae0cffbfe432bb20ec9aba7c2f736cf28fffac49c4ece78f.jpg)

![](images/7420e6041269bc9dc70d74726d603cc16c88c17d2c54db0b7cd3c7a5285cbd25.jpg)

![](images/dadb2cd4cce7a6368030bd9b244a26d3a7cdd517a16868d5c10e1e83b4dfd5ec.jpg)

![](images/2f1a6e72930fc33550a491254a6c69ef146fb648948ce2e4899929bd3e8f09c2.jpg)

Manifold Coefficient Sensitivity — Boundary Fid.  
![](images/a4a7df3cc67330fd18b4689c1624f5da3245abc00a1d238c1a87e32219028c1d.jpg)

![](images/233c13d316151a5926334a23eb45e50da96fae6ca5bea6e6d9a099c66b3fa8df.jpg)

![](images/e6ed18969d8d1e626f38babad1de08b37fcf17ff75d2aac1ee52fc2bd610bb55.jpg)  
Fig. 7: Manifold Coefficient Sensitivity

![](images/68e1c60e7b50cc498d413039b10d966df727a13fa3abe1121187d567b09887da.jpg)

![](images/250a3e1181d168451f2e60f28e70bc4bf74a7eac25bbe686d79c8ec3020bff55.jpg)  
Fig. 8: Query Budget Sensitivity

No LA leads to persistent performance drops, particularly on Cora and Computer, where class imbalance is most severe under the strict 5% budget. On PubMed and Physics, where the class distribution is more balanced and the victim exhibits higher confidence, the contribution of LA is smaller yet still measurable. Consistent with prior findings, when the query set distribution is balanced, enforcing LA may introduce overfitting and cause slight performance drops.

Similar to No LA, No Tuning still results in substantial performance degradation across all victim backbones on Cora and Computer while yielding a slight decline on PubMed and Physics. Under severe imbalance, No Tuning consistently performs worse than No LA, implying that class-balanced sampling plays a fundamental role in stabilizing classifier tuning under skewed query distributions.

## D. Defense (RQ6)

To evaluate Dagger’s robustness under SOTA defense mechanisms, we adopt two distinct defenses, namely, PRADA [9] and BackdoorWM [26]. PRADA monitors the statistical distribution of consecutive queries to identify deviations from a normal distribution, achieving a claimed perfect detection rate against model stealing attacks. Its core assumption is that legitimate queries follow a natural data distribution, whereas model stealing attacks generate statistically anomalous query patterns. BackdoorWM trains a watermarked GNN model to enable post-hoc black-box ownership verification. Specifically, for node classification tasks, a fixed trigger pattern is injected into a small subset of training nodes associated with a target label. Any surrogate inheriting the victim’s decision boundaries is expected to replicate this backdoor behavior on triggered inputs, facilitating near-perfect ownership verification.

TABLE III: Ablation Studies across Graph Datasets and GNN backbones
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Baseline</td><td colspan="4">Metrics</td></tr><tr><td>ACC</td><td>FID</td><td>Macro F1</td><td>Boundary Fid.</td></tr><tr><td rowspan="10">Cora</td><td rowspan="3">GCN</td><td>No manifold No LA</td><td>65.81±0.46</td><td>68.14±0.92</td><td>51.25±0.47</td><td>30.95±1.35</td></tr><tr><td></td><td>70.36±0.46</td><td>72.94±0.17</td><td>54.24±0.37</td><td>29.05±0.67</td></tr><tr><td>No Tuning</td><td>70.60±0.63</td><td>72.94±0.70</td><td>54.22±0.59</td><td>29.05±1.35</td></tr><tr><td rowspan="3">GAT</td><td>Dagger</td><td>71.83±0.46</td><td>73.68±0.17</td><td>59.22±1.20</td><td>32.86±1.17</td></tr><tr><td>No manifold</td><td>55.10±1.66</td><td>57.44±1.36</td><td>45.69±1.31</td><td>27.78±0.98</td></tr><tr><td>No LA</td><td>66.42±0.80</td><td>68.76±0.46</td><td>55.29±1.25</td><td>31.25±2.95</td></tr><tr><td rowspan="3"></td><td>No Tuning</td><td>60.76±2.22</td><td>62.85±2.63</td><td>51.34±2.48</td><td>30.56±1.96</td></tr><tr><td>Dagger</td><td>74.28±1.25</td><td>76.26±0.70</td><td>67.08±1.25</td><td>36.11±0.98</td></tr><tr><td>No manifold</td><td>63.96±0.35</td><td>71.59±0.30</td><td>48.09±0.54</td><td>40.09±2.55</td></tr><tr><td rowspan="3">APPNP</td><td>No LA</td><td>65.44±0.70</td><td>73.06±1.09</td><td>48.85±0.82</td><td>40.99±2.30</td></tr><tr><td>No Tuning</td><td>62.61±0.87 67.40±0.92</td><td>68.02±0.46</td><td>46.89±0.71</td><td>32.88±3.37</td></tr><tr><td>Dagger No manifold</td><td>62.85±0.17</td><td>74.29±1.06</td><td>51.69±0.74</td><td>41.89±1.91</td></tr><tr><td rowspan="4">GraphSAGE</td><td>No LA</td><td>70.85±0.30</td><td>62.36±0.30 69.50±0.17</td><td>52.12±0.04 61.77±0.61</td><td>25.71±0.00 32.38±3.56</td></tr><tr><td></td><td>64.70±1.71</td><td></td><td>53.07±1.66</td><td></td></tr><tr><td>No Tuning</td><td></td><td>63.59±1.66</td><td></td><td>28.57±2.33</td></tr><tr><td>Dagger</td><td>76.51±0.92</td><td>76.01±1.09</td><td>71.91±1.53</td><td>39.05±1.35</td></tr><tr><td rowspan="10">Computer</td><td rowspan="3">GCN</td><td>No manifold</td><td>80.35±0.36</td><td>85.51±0.55</td><td>62.33±1.67</td><td>53.85±2.77</td></tr><tr><td>No LA</td><td>80.86±0.72</td><td>86.63±0.77</td><td>65.38±0.81</td><td>55.13±3.63</td></tr><tr><td>No Tuning</td><td>80.21±0.36</td><td>85.59±0.50</td><td>64.43±1.48</td><td>57.26±3.36</td></tr><tr><td>Dagger</td><td>81.73±0.71</td><td>87.40±0.71</td><td>67.52±1.42</td><td></td><td>60.26±3.63</td></tr><tr><td rowspan="4">GAT</td><td>No manifold</td><td>64.97±0.27</td><td>66.72±0.54</td><td>34.80±0.74</td><td>31.33±0.94</td></tr><tr><td>No LA</td><td>67.30±0.33</td><td>68.48±0.69</td><td>39.62±0.52</td><td>30.00±1.63</td></tr><tr><td>No Tuning</td><td>65.60±0.27</td><td>67.42±0.38</td><td>35.61±1.02</td><td>32.67±0.94</td></tr><tr><td>Dagger</td><td>71.46±0.22</td><td>72.84±0.24</td><td>50.19±1.31</td><td>31.33±0.94</td></tr><tr><td rowspan="4">APPNP</td><td>No manifold</td><td>81.32±0.33</td><td>86.36±0.33</td><td>62.53±2.61</td><td>43.75±1.70</td></tr><tr><td>No LA</td><td>83.33±0.45</td><td>88.03±0.24</td><td>73.37±0.66</td><td></td></tr><tr><td>No Tuning</td><td>82.82±0.39</td><td>87.96±0.28</td><td>71.24±0.33</td><td>44.44±2.14</td></tr><tr><td>Dagger</td><td>84.33±0.36</td><td>88.61±0.35</td><td>77.75±2.04</td><td>43.06±0.98</td></tr><tr><td rowspan="4">GraphSAGE</td><td>No manifold</td><td>77.62±1.23</td><td>82.80±1.47</td><td>62.29±3.05</td><td>44.10±1.30 47.31±2.74</td></tr><tr><td>No LA</td><td>79.17±0.42</td><td>84.84±0.42</td><td>68.48±0.52</td><td>47.31±3.31</td></tr><tr><td>No Tuning</td><td>77.18±0.62</td><td>82.68±0.91</td><td>64.01±0.83</td><td>47.85±1.52</td></tr><tr><td>Dagger</td><td>80.62±0.33</td><td>86.36±0.48</td><td>72.36±1.21</td><td>47.31±2.01</td></tr></table>

We follow the default parameter setting in defense benchmarking [28] and report their defense results in Table IV. We observe that Dagger remains highly effective against both PRADA and BackdoorWM across all graph datasets and victim backbones, achieving attack performance comparable to the undefended setting. Under most graph-backbone combinations, Dagger evaluated against BackdoorWM (short for BackdoorWM, hereinafter) outperforms the one against PRADA (short for PRADA, hereinafter), which aligns with imbalance trends reflected by macro F1. Notably, on challenging graphs such as Cora and Computer, BackdoorWM even obtains better attack performance than the undefended setting, this variation can be attributed to subtle differences between the clean and the backdoor-injected victims, as the backdoor injection slightly alters the victim’s prediction distribution.

PRADA is ineffective. PRADA achieves 100% prediction fidelity relative to the original victim model, suggesting that it fails to trigger an alarm and treats every query as legitimate during initial model evaluation. Although PRADA eventually activates in the subsequent querying phase and corrupts a portion of the victim-predicted labels with Gaussian noise, Dagger’s attack performance remains comparable to the undefended setting. One plausible explanation is that the trigger occurs in the late query phase, Dagger has already accumulated sufficient clean supervision signals from early queries, and its manifold mixup confers inherent robustness against label noise by interpolating latent representations.

BackdoorWM provides negligible protection. As for BackdoorWM, fidelity drops to between 76.75% and 99.25% across datasets and backbones, indicating that the watermark is only partially inherited by the surrogate. More importantly, BackdoorWM yields slightly higher attack performance than or comparable to the undefended setting across all configurations, confirming that BackdoorWM fails to degrade the functional quality of the stolen surrogate model.

TABLE IV: Dagger’s Attack Performance against Defense Mechanisms across Graph Datasets and GNN Backbones
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Defense</td><td rowspan="2">Pred Acc</td><td rowspan="2">Pred Fid</td><td colspan="4">Metrics</td></tr><tr><td>ACC</td><td>FID</td><td>Macro F1</td><td>Boundary Fid.</td></tr><tr><td rowspan="6">Cora</td><td rowspan="2">GCN</td><td>PRADA</td><td>83.03</td><td>100.00</td><td>68.88±0.17</td><td>72.45±0.63</td><td>55.80±0.64</td><td>33.33±1.78</td></tr><tr><td>BackdoorWM</td><td>85.98</td><td>84.87</td><td>75.15±1.36</td><td>75.77±0.97</td><td>69.79±1.42</td><td>41.43±1.17</td></tr><tr><td rowspan="2">GAT</td><td>PRADA</td><td>80.81</td><td>100.00</td><td>72.69±1.38</td><td>74.91±0.60</td><td>65.61±2.25</td><td>40.97±2.60</td></tr><tr><td>BackdoorWM</td><td>85.98</td><td>84.87</td><td>74.66±0.97</td><td>76.01±1.68</td><td>68.96±1.08</td><td>41.67±5.10</td></tr><tr><td rowspan="2">APPNP</td><td>PRADA</td><td>80.44</td><td>100.00</td><td>68.02±0.46</td><td>73.55±0.35</td><td>53.95±0.63</td><td>45.95±1.10</td></tr><tr><td>BackdoorWM</td><td>88.19</td><td>76.01</td><td>77.98±0.46</td><td>76.63±0.63</td><td>73.28±0.70</td><td>49.10±1.69</td></tr><tr><td rowspan="6"></td><td rowspan="2">GraphSAGE</td><td>PRADA</td><td>87.45</td><td>100.00</td><td>69.13±0.35</td><td>69.74±0.80</td><td>61.92±0.16</td><td>26.67±1.35</td></tr><tr><td>BackdoorWM</td><td>89.30</td><td>92.25</td><td>72.82±0.46</td><td>72.69±0.52</td><td>69.38±0.51</td><td>39.05±1.35</td></tr><tr><td rowspan="2">GCN</td><td>PRADA</td><td>84.28</td><td>100.00</td><td>83.81±0.10</td><td>91.60±0.20</td><td>82.75±0.13</td><td>55.26±1.64</td></tr><tr><td>BackdoorWM</td><td>85.09</td><td>96.60</td><td>84.58±0.31</td><td>91.85±0.05</td><td>83.54±0.31</td><td>55.12±0.55</td></tr><tr><td rowspan="2">GAT</td><td>PRADA</td><td>84.84</td><td>100.00</td><td>83.62±0.18</td><td>92.00±0.29</td><td>82.53±0.20</td><td>54.92±1.47</td></tr><tr><td>BackdoorWM</td><td>85.04</td><td>97.77</td><td>83.92±0.25</td><td>92.22±0.35</td><td>82.87±0.17</td><td>56.82±0.65</td></tr><tr><td rowspan="4"></td><td rowspan="2">APPNP</td><td>PRADA</td><td>85.60</td><td>100.00</td><td>83.23±0.21</td><td>93.61±0.07</td><td>82.24±0.21</td><td>61.67±0.76</td></tr><tr><td>BackdoorWM</td><td>87.27</td><td>95.28</td><td>84.62±0.17</td><td>94.81±0.25</td><td>83.63±0.17</td><td>67.48±1.94</td></tr><tr><td>GraphSAGE PRADA</td><td>86.00</td><td>100.00</td><td>83.76±0.38</td><td>92.93±0.58</td><td>82.70±0.29</td><td></td><td>56.41±3.02</td></tr><tr><td></td><td>BackdoorWM</td><td>86.82</td><td>95.99</td><td>83.77±0.26</td><td>92.21±0.51</td><td>82.75±0.25</td><td>52.82±2.33</td></tr><tr><td rowspan="6">Computer</td><td rowspan="2">GCN</td><td>PRADA</td><td>89.97</td><td>100.00</td><td>81.61±0.66</td><td>87.48±0.81</td><td>66.88±1.77</td><td>59.40±3.02</td></tr><tr><td>BackdoorWM</td><td>89.03</td><td>95.20</td><td>80.21±0.73</td><td>86.07±0.39</td><td>71.69±1.46</td><td>53.85±1.81</td></tr><tr><td rowspan="2">GAT</td><td>PRADA</td><td>88.59</td><td>100.00</td><td>69.57±0.45</td><td>70.59±0.29</td><td>45.04±0.68</td><td>32.67±2.49</td></tr><tr><td>BackdoorWM</td><td>90.41</td><td>92.30</td><td>77.59±0.30</td><td>77.28±0.19</td><td>59.76±1.22</td><td>44.00±4.32</td></tr><tr><td rowspan="2">APPNP</td><td>PRADA</td><td>89.39</td><td>100.00</td><td>84.23±0.21</td><td>88.35±0.30</td><td>77.75±1.49</td><td>40.62±2.55</td></tr><tr><td>BackdoorWM</td><td>88.74</td><td>93.02</td><td>81.73±0.55</td><td>83.50±0.45</td><td>71.35±2.51</td><td>36.46±1.70</td></tr><tr><td rowspan="6"></td><td rowspan="2">GraphSAGE</td><td>PRADA</td><td>90.12</td><td>100.00</td><td>80.77±0.19</td><td>86.43±0.35</td><td>72.71±1.32</td><td>48.39±2.28</td></tr><tr><td>BackdoorWM</td><td>88.81</td><td>94.84</td><td>78.75±0.65</td><td>82.95±0.06</td><td>70.09±1.63</td><td>37.10±1.32</td></tr><tr><td rowspan="2">GCN</td><td>PRADA</td><td>96.26</td><td>100.00</td><td>95.60±0.17</td><td>97.18±0.03</td><td>94.07±0.27</td><td>56.30±1.05</td></tr><tr><td>BackdoorWM</td><td>95.91</td><td>97.86</td><td>95.20±0.07</td><td>96.77±0.05</td><td>93.58±0.07</td><td>50.37±2.10</td></tr><tr><td rowspan="2">GAT</td><td>PRADA</td><td>94.99</td><td>100.00</td><td>95.16±0.04</td><td>94.82±0.08</td><td>93.42±0.07</td><td>54.55±1.86</td></tr><tr><td>BackdoorWM</td><td>95.22</td><td>95.42</td><td>95.36±0.09</td><td>94.71±0.04</td><td>93.57±0.09</td><td>55.30±2.14</td></tr><tr><td rowspan="3">Physics</td><td>APPNP</td><td></td><td>100.00</td><td>95.66±0.08</td><td>97.92±0.07</td><td>94.13±0.10</td><td></td><td>56.86±2.77</td></tr><tr><td rowspan="2"></td><td>PRADA BackdoorWM</td><td>96.70 96.81</td><td>98.84</td><td>95.44±0.08</td><td>97.17±0.13</td><td>93.94±0.02</td><td>54.25±0.92</td></tr><tr><td>PRADA</td><td>95.83</td><td>100.00</td><td>95.65±0.09</td><td>96.45±0.05</td><td>94.18±0.12</td><td>45.74±2.90</td></tr><tr><td rowspan="2"></td><td>GraphSAGE BackdoorWM</td><td></td><td></td><td></td><td>95.36±0.04</td><td>96.65±0.10</td><td>93.65±0.06</td><td>44.96±2.19</td></tr><tr><td></td><td>95.71</td><td>99.25</td><td></td><td></td><td></td><td></td><td></td></tr></table>

Limitations and Future Work. First, our analysis reveals that extreme minority classes with near-zero class coverage represent a fundamental limitation, motivating the development of adaptive query strategies that allocate budget more effectively. Second, despite operating under a strict query budget, further reducing the number of queries through active learning methods remains a key priority. Third, while we focus on mainstream message-passing GNNs that implicitly assume homophily, tailoring model stealing attacks to heterogeneous graphs and GNNs represents a promising future direction. Finally, designing robust defenses against Dagger under such highly constrained threat model represents a critical open problem for the research community.

## VI. CONCLUSION

In this work, we propose Dagger, a novel model stealing attack against GNNs under a strictly constrained blackbox, hard-label, query-limited, and backbone-agnostic threat model. We identify four fundamental challenges specific to this setting: poor surrogate performance, insufficient supervision, class imbalance with class coverage issues, and backbone mismatch. Motivated by these systematic limitations, we design a two-phase decoupling-based attack framework. The first phase leverages decoupled information diffusion to incorporate isolated nodes and mitigate structural sparsity, and manifold-level node mixup to compensate for the incomplete and imbalanced supervision signals. The second phase applies a decoupled classifier fine-tuned with class-balanced sampling and logit adjustment to recover minority-class performance without requiring extra victim queries. Extensive experiments demonstrate that Dagger consistently outperforms state-of-theart baselines across diverse graph datasets and victim backbones. Moreover, our defense evaluations reveal that Dagger is inherently resistant to both query-monitoring and backdoor watermarking defenses, highlighting the urgent need for more robust GNN defense mechanisms.

## OPEN SCIENCE

To comply with the CFP’s Open Science guidelines, we release the source code along with implementation details and evaluation pipeline to reproduce our experiments and conduct the attacks. As the datasets used in our paper are all downloadable from Pytorch Geometric [3], we direct users to the downloading links and provide processing scripts in our artifacts. To preserve anonymity during review, these materials are hosted in an anonymous repository https://anonymous. 4open.science/r/Dagger-1D14, and will be migrated to a public GitHub repository upon acceptance. This artifact release is intended to facilitate more in-depth adversarial investigation on graph neural networks and to motivate future work on more robust and effective defense mechanisms.

## LLM USAGE CONSIDERATIONS

We use Claude only for editorial assistance in this manuscript, and all outputs were inspected by the authors to ensure accuracy and originality.

## ETHICAL CONSIDERATIONS

This study investigates the vulnerability of graph neural networks (GNNs) through the lens of model stealing attacks and the empirical results demonstrate GNNs are highly susceptible to such attacks even under a more restricted threat model than existing GNN stealing attacks.

We recognize the inherent dual-use risk in publishing attack methodologies. However, we contend that the benefits to the research community outweigh this risk. First, the threat model we explore reflects realistic deployments that existing defenses have not adequately addressed, and our findings underscore the practical severity of GNN stealing attacks and motivate the urgent development of corresponding defense mechanisms. Second, the primary beneficiaries of this work are defenders. By characterizing the attack surface, we enable model developers and deployment teams to proactively harden their systems before adversarial exploitation occurs in the wild. Third, as a red team, we conduct all experiments on local devices using publicly available graph datasets. The testing environment does not interact with real-world third-party systems, collect personal data, or involve human subjects. Finally, our public artifact release is scoped to the code and scripts necessary to reproduce the reported results. We do not release components that would trivially enable large-scale exploitation beyond the experimental setup described in this paper.

## REFERENCES

[1] D. DeFazio and A. Ramesh, “Adversarial model extraction on graph neural networks,” no. arXiv:1912.07721, Dec. 2019, arXiv:1912.07721 [cs, stat]. [Online]. Available: http://arxiv.org/abs/1912.07721

[2] Z. Fang, X. Zhang, A. Zhao, X. Li, H. Chen, and J. Li, “Recent developments in gnns for drug discovery,” 2025. [Online]. Available: https://arxiv.org/abs/2506.01302

[3] M. Fey and J. E. Lenssen, “Fast graph representation learning with PyTorch Geometric,” in ICLR Workshop on Representation Learning on Graphs and Manifolds, 2019.

[4] J. Gasteiger, A. Bojchevski, and S. Gunnemann, “Combining neural¨ networks with personalized pagerank for classification on graphs,” in International Conference on Learning Representations, 2019. [Online]. Available: https://openreview.net/forum?id=H1gL-2A9Ym

[5] F. Guan, T. Zhu, H. Tong, and W. Zhou, “A realistic model extraction attack against graph neural networks,” Know.-Based Syst., vol. 300, no. C, Nov. 2024. [Online]. Available: https: //doi.org/10.1016/j.knosys.2024.112144

[6] W. L. Hamilton, R. Ying, and J. Leskovec, “Inductive representation learning on large graphs,” in Proceedings of the 31st International Conference on Neural Information Processing Systems, ser. NIPS’17. Red Hook, NY, USA: Curran Associates Inc., 2017, p. 1025–1035.

[7] W. Hu, B. Liu, J. Gomes, M. Zitnik, P. Liang, V. Pande, and J. Leskovec, “Strategies for pre-training graph neural networks,” in International Conference on Learning Representations, 2020. [Online]. Available: https://openreview.net/forum?id=HJlWWJSFDH

[8] M. Jagielski, N. Carlini, D. Berthelot, A. Kurakin, and N. Papernot, “High accuracy and high fidelity extraction of neural networks,” in Proceedings of the 29th USENIX Conference on Security Symposium, ser. SEC’20. USA: USENIX Association, 2020.

[9] M. Juuti, S. Szyller, S. Marchal, and N. Asokan, “ PRADA: Protecting Against DNN Model Stealing Attacks ,” in 2019 IEEE European Symposium on Security and Privacy (EuroS&P). Los Alamitos, CA, USA: IEEE Computer Society, Jun. 2019, pp. 512– 527. [Online]. Available: https://doi.ieeecomputersociety.org/10.1109/ EuroSP.2019.00044

[10] T. N. Kipf and M. Welling, “Semi-supervised classification with graph convolutional networks,” in International Conference on Learning Representations, 2017. [Online]. Available: https://openreview.net/ forum?id=SJU4ayYgl

[11] C. Lin, G. J. Sun, K. C. Bulusu, J. R. Dry, and M. Hernandez, “Graph neural networks including sparse interpretability,” 2020. [Online]. Available: https://arxiv.org/abs/2007.00119

[12] Y. Liu, X. Ao, Z. Qin, J. Chi, J. Feng, H. Yang, and Q. He, “Pick and choose: a gnn-based imbalanced learning approach for fraud detection,” in Proceedings of the web conference 2021, 2021, pp. 3168–3177.

[13] S. Luan, C. Hua, M. Xu, Q. Lu, J. Zhu, X.-W. Chang, J. Fu, J. Leskovec, and D. Precup, “When do graph neural networks help with node classification? investigating the homophily principle on node distinguishability,” in Advances in Neural Information Processing Systems, A. Oh, T. Naumann, A. Globerson, K. Saenko, M. Hardt, and S. Levine, Eds., vol. 36. Curran Associates, Inc., 2023, pp. 28 748–28 760. [Online]. Available: https://proceedings.neurips.cc/paper files/paper/ 2023/file/5ba11de4c74548071899cf41dec078bf-Paper-Conference.pdf

[14] A. K. Menon, S. Jayasumana, A. S. Rawat, H. Jain, A. Veit, and S. Kumar, “Long-tail learning via logit adjustment,” in International Conference on Learning Representations, 2021. [Online]. Available: https://openreview.net/forum?id=37nvvqkCo5

[15] L. Page, S. Brin, R. Motwani, and T. Winograd, “The pagerank citation ranking : Bringing order to the web,” in The Web Conference, 1999. [Online]. Available: https://api.semanticscholar.org/CorpusID:1508503

[16] M. Podhajski, J. Dubinski, F. Boenisch, A. Dziedzic, A. P. Michalak,´ and Tomasz, “Efficient model-stealing attacks against inductive graph neural networks,” no. arXiv:2405.12295, Aug. 2024, arXiv:2405.12295 [cs]. [Online]. Available: http://arxiv.org/abs/2405.12295

[17] O. Shchur, M. Mumme, A. Bojchevski, and S. Gunnemann, “Pitfalls¨ of graph neural network evaluation,” 2019. [Online]. Available: https://arxiv.org/abs/1811.05868

[18] Y. Shen, X. He, Y. Han, and Y. Zhang, “Model stealing attacks against inductive graph neural networks,” in 2022 IEEE Symposium on Security and Privacy (SP), May 2022, p. 1175–1192. [Online]. Available: https://ieeexplore.ieee.org/abstract/document/9833607

[19] S. Suresh, V. Budde, J. Neville, P. Li, and J. Ma, “Breaking the limit of graph neural networks by improving the assortativity of graphs with local mixing patterns,” in Proceedings of the 27th ACM SIGKDD Conference on Knowledge Discovery & Data Mining, ser. KDD ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 1541–1551. [Online]. Available: https://doi.org/10.1145/3447548.3467373

[20] P. Velickoviˇ c, G. Cucurull, A. Casanova, A. Romero, P. Li´ o, and\` Y. Bengio, “Graph attention networks,” in International Conference on Learning Representations, 2018. [Online]. Available: https: //openreview.net/forum?id=rJXMpikCZ

[21] H. Wang, Z. Wei, J. Gan, S. Wang, and Z. Huang, “Personalized pagerank to a target node, revisited,” in Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining, ser. KDD ’20. New York, NY, USA: Association for Computing Machinery, 2020, p. 657–667. [Online]. Available: https://doi.org/10.1145/3394486.3403108

[22] Y. Wang, W. Wang, Y. Liang, Y. Cai, and B. Hooi, “Mixup for node and graph classification,” in Proceedings of the Web Conference 2021, ser. WWW ’21. New York, NY, USA: Association for Computing Machinery, 2021, p. 3663–3674. [Online]. Available: https://doi-org.pitt.idm.oclc.org/10.1145/3442381.3449796

[23] Z. Wang, M. Lin, B. Shen, K. Anderson, M. Liu, T. Cai, and Y. Dong, “CEGA: A cost-effective approach for graph-based model extraction and acquisition,” in Forty-second International Conference on Machine Learning, 2025. [Online]. Available: https: //openreview.net/forum?id=HnXElKZdEh

[24] B. Wu, V. Profile, X. Yang, V. Profile, S. Pan, V. Profile, X. Yuan, and V. Profile, “Model extraction attacks on graph neural networks,” Proceedings of the 2022 ACM on Asia Conference on Computer and Communications Security, p. 337–350, May 2022.

[25] S. Wu, F. Sun, W. Zhang, X. Xie, and B. Cui, “Graph neural networks in recommender systems: a survey,” ACM computing surveys, vol. 55, no. 5, pp. 1–37, 2022.

[26] J. Xu, S. Koffas, O. Ersoy, and S. Picek, “Watermarking graph neural networks based on backdoor attacks,” in 2023 IEEE 8th European Symposium on Security and Privacy (EuroS&P), 2023, pp. 1179–1197.

[27] Z. Yang, W. W. Cohen, and R. Salakhutdinov, “Revisiting semisupervised learning with graph embeddings,” in Proceedings of the 33rd International Conference on International Conference on Machine Learning - Volume 48, ser. ICML’16. JMLR.org, 2016, p. 40–48.

[28] K. Zhao, B. Shen, Y. Dai, S. Chakraborty, and Y. Dong, “Graphipbench: How hard is it to steal a graph neural network, and can we stop it?” 2026. [Online]. Available: https://arxiv.org/abs/2605.12827

[29] Y. Zhuang, C. Shi, M. Zhang, J. Chen, L. Lyu, P. Zhou, and L. Sun, “Unveiling the secrets without data: Can graph neural networks be exploited through Data-Free model extraction attacks?” 2024, p. 5251–5268. [Online]. Available: https://www.usenix.org/conference/ usenixsecurity24/presentation/zhuang

## APPENDIX

## A. Implementation

We follow the standard training/validation/test splits of 80%/10%/10% [7], [11], [19] across four graph datasets. As for victim GNN training, we adopt the learning rate as 0.001, weight decay as $5 e - 4$ and training epochs as 600. Adam is used as the optimizer. The hidden dimension is 16 and the number of layers is 2. In terms of GNN stealing attacks, we choose the learning rate from {0.01, 0.001}, weight decay as 5e − 4 and training epochs as 600. $K \in \{ 5 , 1 0 , 1 5 , 2 0 \}$ $\alpha \in \{ 0 . 1 , 0 . 1 5 , 0 . 2 , 0 . \hat { 2 } 5 , \hat { 0 } . 3 \}$ and $w _ { \mathrm { m i x } } \in \{ 0 . 1 , 0 . 5 , 1 , 1 . 5 \}$ for surrogate pre-training; We set the learning rate as 0.01, training epochs as 200 and $\tau { = } 0 . 5$ for imbalance-aware fine-tuning. Regarding baselines, we leverage the same training configurations for fair comparison and other customized parameters strictly follow the default setting in [28]. We further restrict node selection to training node sets for two reasons, first, test nodes are inaccessible to the adversary in realistic deployments; second, covering test nodes in the query set would introduce data leakage into the evaluation, artificially inflating the attack performance of the surrogate model. Notably, even under these restrictions, the baselines operate under more favorable conditions than Dagger as their victim-surrogate backbones are aligned while Dagger is backbone-agnostic. We finally report the average results of 3 trials. All experiments are conducted on a 64-bit machine with 4 Nvidia A100 GPUs.

## B. Benign GNN Performance

Table V reports the benign classification performance of each victim GNN backbone. All backbones achieve strong predictive performance across graph datasets, confirming that they represent valuable stealing targets. Among these, GraphSAGE generally outperforms other backbones on Cora, PubMed and Computer, while APPNP excels on Physics.

TABLE V: Benign Performance across GNN Backbones
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">GNN</td><td colspan="2">Metrics</td></tr><tr><td>Acc</td><td>F1</td></tr><tr><td rowspan="4">Cora</td><td>GCN</td><td>83.03</td><td>79.61</td></tr><tr><td>GAT</td><td>80.81</td><td>73.26</td></tr><tr><td>APPNP</td><td>80.44</td><td>67.30</td></tr><tr><td>GraphSAGE</td><td>87.45</td><td>85.48</td></tr><tr><td rowspan="3">PubMed</td><td>GCN GAT</td><td>84.28 84.84</td><td>83.18 83.63</td></tr><tr><td>APPNP</td><td>86.00</td><td>84.44</td></tr><tr><td>GraphSAGE</td><td>86.00</td><td>85.01</td></tr><tr><td rowspan="3">Computer</td><td>GCN</td><td>89.97</td><td>88.64</td></tr><tr><td>GAT</td><td>88.59</td><td>86.88</td></tr><tr><td>APPNP GraphSAGE</td><td>89.39 90.12</td><td>86.94 88.76</td></tr><tr><td rowspan="3">Physics</td><td>GCN</td><td>96.26</td><td>95.00</td></tr><tr><td>GAT APPNP</td><td>94.99</td><td>93.34 95.72</td></tr><tr><td>GraphSAGE</td><td>97.00 95.83</td><td>94.48</td></tr></table>

## C. Attack Performance

## D. Balance and Its Impact Factors

Balance To investigate the effect of class balance on attack performance, we construct four balance-level query node sets on Cora, the representative of the most imbalanced graphs. We subsample the full training nodes in the victim graph according to different target distributions. Specifically, let $n _ { c }$ denote the number of training nodes whose victim-predicted label is class $c ,$ and $| \mathcal { G } _ { t } |$ represents the number of training nodes. The four levels are defined by the following sampling proportions, $\begin{array} { r } { p _ { c } ^ { \mathrm { v e r y } } \propto \frac { n _ { c } } { | \mathcal { G } _ { t } | } ; p _ { c } ^ { \mathrm { i m b } } \propto \sqrt { \frac { n _ { c } } { | \mathcal { G } _ { t } | } } ; p _ { c } ^ { \mathrm { s l i g h t } } \propto \left( \frac { n _ { c } } { | \mathcal { G } _ { t } | } \right) ^ { 1 / 3 } ; } \end{array}$ and $\begin{array} { r } { p _ { c } ^ { \mathrm { b a l a n c e d } } = \frac { 1 } { | C | } } \end{array}$ . Each balance level is normalized to sum to one, and the largest-remainder allocation method is used to ensure a fixed query budget across all levels, where each class is first assigned its integer quota of query nodes and the remaining unallocated spots are then sequentially distributed to classes with the largest fractional remainders. The very imbalanced level preserves the heavily skewed distribution of victim predictions on the induced subgraph; imbalanced and slightly imbalanced levels progressively mitigate imbalance issues; and the balanced level enforces strict equality across classes. This design enables a controlled study of class balance effects while maintaining a fixed total query budget.

TABLE VI: The Performance of Dagger on PubMed and Physics. Unit: 1e-2. Arrow indicates the direction of better performance and the bold font denotes the ‘best’ results.
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Baseline</td><td colspan="5">Metrics</td></tr><tr><td>ACC</td><td>FID</td><td>Macro F1</td><td>Boundary Fid.</td><td>Query</td></tr><tr><td rowspan="20"></td><td rowspan="4">GCN</td><td>Rand</td><td>40.67±0.00</td><td>39.15±0.00</td><td>19.27±0.00</td><td>39.04±0.00</td><td>985</td></tr><tr><td>MEA</td><td>82.44±0.57</td><td>93.93±1.48</td><td>81.26±0.65</td><td>63.74±1.03</td><td>3810</td></tr><tr><td>AdvMEA</td><td>50.91±5.15</td><td>53.23±5.90</td><td>44.38±10.17</td><td>51.52±3.48</td><td>150</td></tr><tr><td>CEGA</td><td>50.68±1.21</td><td>54.36±1.18</td><td>47.83±2.72</td><td>42.54±0.62</td><td>984</td></tr><tr><td rowspan="6"></td><td>DFEAII-Real</td><td>49.15±2.47</td><td>52.18±2.35</td><td>42.51±3.44</td><td>39.04±1.29</td><td>985</td></tr><tr><td>Dagger</td><td>83.81±0.06</td><td>91.72±0.17</td><td>82.80±0.14</td><td>54.53±0.55</td><td>985</td></tr><tr><td>Rand</td><td>40.67±0.00</td><td>38.74±0.00</td><td>19.27±0.00</td><td>39.90±0.00</td><td>985</td></tr><tr><td>MEA</td><td>82.49±0.31 33.67±9.90</td><td>94.32±0.53</td><td>81.33±0.42</td><td>60.28±3.12</td><td>3810</td></tr><tr><td>AdvMEA CEGA</td><td>76.17±2.29</td><td>32.59±8.70 84.99±2.12</td><td>16.50±3.92 75.48±2.12</td><td>34.54±7.57 50.43±2.33</td><td>150</td></tr><tr><td>DFEAII-Real</td><td>78.06±0.77</td><td></td><td></td><td></td><td>984</td></tr><tr><td rowspan="8">PubMed APPNP</td><td></td><td></td><td>86.76±1.90</td><td>76.32±1.78</td><td>50.43±3.79</td><td>985</td></tr><tr><td>Dagger Rand</td><td>83.54±0.32 50.19±13.46</td><td>92.29±0.27</td><td>82.41±0.33</td><td>56.65±1.29</td><td>985</td></tr><tr><td>MEA</td><td>81.36±0.91</td><td>51.20±16.76</td><td>30.09215.30</td><td>45.48±4.20</td><td>985</td></tr><tr><td>AdvMEA</td><td>41.84±0.90</td><td>89.54±1.65 42.49±1.29</td><td>79.37±0.83</td><td>62.62±1.91</td><td>3810</td></tr><tr><td>CEGA</td><td>60.09±8.94</td><td></td><td>24.18±2.59</td><td>38.60±2.67</td><td>150</td></tr><tr><td>DFEAII-Real</td><td>64.37±4.60</td><td>62.90±9.64 67.44±6.38</td><td>59.45±8.48 54.58±9.96</td><td>41.70±2.01</td><td>984</td></tr><tr><td>Dagger</td><td>83.27±0.15</td><td>93.68±0.13</td><td>82.29±0.14</td><td>49.12±0.83</td><td>985</td></tr><tr><td>Rand</td><td>40.67±0.00</td><td>39.86±0.00</td><td>19.27±0.00</td><td>62.48±1.16</td><td>985</td></tr><tr><td rowspan="6">GraphSAGE</td><td>MEA</td><td>82.45±0.47</td><td></td><td></td><td>48.72±0.00</td><td>985</td></tr><tr><td></td><td>46.40±6.23</td><td>92.92±0.38</td><td>81.36±0.53</td><td>60.34±3.89</td><td>3810</td></tr><tr><td>AdvMEA</td><td></td><td>46.18±6.32</td><td>34.27±12.78</td><td>40.85±8.23</td><td>150</td></tr><tr><td>CEGA</td><td>81.80±1.56</td><td>83.87±1.15</td><td>81.64±1.39</td><td>49.91±3.03</td><td>984</td></tr><tr><td>DFEAII-Real</td><td>82.72±1.04</td><td>84.58±0.58</td><td>82.42±0.98</td><td>48.89±2.31</td><td>985</td></tr><tr><td>Dagger Rand</td><td>84.01±0.09</td><td>93.44±0.24</td><td>82.95±0.07</td><td>57.78±1.89</td><td>985</td></tr><tr><td rowspan="9">Physics</td><td>GCN</td><td>50.58±0.00</td><td>50.55±0.00</td><td>13.44±0.00</td><td>26.67±0.00</td><td>1724</td></tr><tr><td>MEA</td><td>95.34±0.11</td><td>97.64±0.21</td><td>93.64±0.15</td><td>60.00±4.80</td><td>16072</td></tr><tr><td>AdvMEA</td><td>91.61±0.33</td><td>93.19±0.51</td><td>88.04±0.54</td><td>53.33±4.80</td><td>250</td></tr><tr><td>CEGA</td><td>91.92±0.49</td><td>93.67±0.55</td><td>89.49±0.55</td><td>45.93±2.77</td><td>1720</td></tr><tr><td>DFEAII-Real</td><td>92.46±0.50</td><td>93.94±0.55</td><td>89.62±0.84</td><td>40.74±3.78</td><td>1724</td></tr><tr><td>Dagger</td><td>95.51±0.09</td><td>97.23±0.05</td><td>93.94±0.10</td><td>59.26±2.77</td><td>1724</td></tr><tr><td>Rand MEA</td><td>50.58±0.00</td><td>50.26±0.00</td><td>13.44±0.00</td><td>25.00±0.00</td><td>1724</td></tr><tr><td>AdvMEA</td><td>88.98±0.81</td><td>89.94±0.72</td><td>85.74±0.67</td><td>46.97±2.14</td><td>16072</td></tr><tr><td></td><td>79.46±0.46</td><td>80.01±0.45</td><td>72.60±1.46</td><td>42.42±4.29</td><td>250</td></tr><tr><td rowspan="5">GAT</td><td>CEGA</td><td>88.54±1.52</td><td>88.93±1.22</td><td>84.99±1.31</td><td>50.76±2.83</td><td>1720</td></tr><tr><td>DFEAII-Real</td><td>88.41±1.65</td><td>88.95±1.64</td><td>83.39±2.234</td><td>48.48±7.03</td><td>1724</td></tr><tr><td>Dagger</td><td>95.22±0.05</td><td>94.98±0.04</td><td>93.55±0.06</td><td>56.06±2.14</td><td>1724</td></tr><tr><td>Rand</td><td>69.76±0.29</td><td>70.97±0.36</td><td>50.14±0.16</td><td>36.60±3.33</td><td>1724</td></tr><tr><td>MEA</td><td>95.54±0.20</td><td>98.02±0.14</td><td>93.93±0.33</td><td>59.48±4.03</td><td>16072</td></tr><tr><td rowspan="6">APPNP</td><td>AdvMEA</td><td>92.81±0.46</td><td>94.93±0.52</td><td>89.86±0.86</td><td>60.78±2.77</td><td>250</td></tr><tr><td>CEGA</td><td>70.73±21.49</td><td>71.62±21.77</td><td>73.10±15.06</td><td>42.48±7.39</td><td>1720</td></tr><tr><td>DFEAII-Real</td><td>78.25±3.13</td><td>78.84±3.46</td><td>59.96±6.02</td><td>39.87±6.67</td><td>1724</td></tr><tr><td>Dagger</td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>Rand</td><td>95.46±0.09</td><td>97.60±0.09</td><td>93.86±0.12</td><td>54.90±1.60</td><td>1724</td></tr><tr><td>MEA</td><td>50.58±0.00</td><td>50.55±0.00</td><td>13.44±0.00</td><td>27.91±0.00</td><td>1724</td></tr><tr><td rowspan="5">GraphSAGE</td><td></td><td>94.71±0.18</td><td>96.13±0.26</td><td>92.79±0.25</td><td>47.29±2.90</td><td>16072</td></tr><tr><td>AdvMEA</td><td>91.35±0.24</td><td>92.41±0.18</td><td>87.83±0.31</td><td>49.61±2.19</td><td>250</td></tr><tr><td>CEGA</td><td>92.76±1.49</td><td>92.81±1.31</td><td>91.17±1.38</td><td>39.53±0.00</td><td>1720</td></tr><tr><td>DFEAII-Real</td><td>93.39±0.13</td><td>92.41±0.18</td><td>87.83±0.31</td><td>49.61±2.19</td><td>1724</td></tr><tr><td>Dagger</td><td>95.42±0.06</td><td>96.42±0.05</td><td>93.96±0.07</td><td>51.94±2.19</td><td>1724</td></tr></table>

Contrary to the intuition that perfectly balanced queries should always yield the best surrogate, gradually enforcing balance constraints does not consistently improve attack performance across all victim backbones, sometimes even degrading it. Under high imbalance, the gap between the settings with and without class-balanced sampling and logit adjustment is largest across victim backbones. As the balance level increases, the two curves converge, indicating that rebalancing provides marginal benefit as the query distribution becomes more uniform. Sometimes they might cause a slight performance degradation, one plausible explanation is that excess resampling and calibration enforcements might cause overfitting.

Impact Factors We then investigate this finding with three factors interlinked with balance, namely label accuracy, isolated ratio and average degree in Figure 9. As the balance level increases, these three factors exhibit victim-dependent and non-monotonic trends across balance levels, suggesting that the relationship between query balance and data quality is more nuanced than a simple trade-off.

Label accuracy shows inconsistent trends across victim backbones: it first increases then decreases for GCN, decreases then stably increases for GAT and APPNP, and steadily increases for GraphSAGE as the balance level rises. This suggests that the label quality effect of more balanced node selection is highly dependent on the victim’s prediction behaviors. For some backbones, selectively including minority-class nodes can eventually improve the overall label accuracy of the query set, while for others it introduces noise.

Isolated ratio is similarly non-monotonic: it decreases then rises for GCN, decreases then partially recovers for GAT, decreases monotonically for APPNP, and first increases then decreases for GraphSAGE. This indicates a complex interplay between query class balance and the structural sparsity of local subgraphs, where enforcing class balance does not guarantee structural integrity in the query set.

Average degree also exhibits victim-specific patterns: it rises then falls for GCN, rises with a slight dip then rises again for GAT, slightly decreases then sharply increases for APPNP, and remains stable before rising for GraphSAGE, which showcases that the structural richness of the query set does not vary predictably with query balance level.

## E. Ablation Studies

TABLE VII: Ablation Studies across Graph Datasets and GNN Backbones
<table><tr><td rowspan="2">Dataset</td><td rowspan="2">Backbone</td><td rowspan="2">Ablation</td><td colspan="4">Metrics</td></tr><tr><td>ACC</td><td>FID</td><td>Macro F1</td><td>Boundary Fid.</td></tr><tr><td rowspan="10">PubMed</td><td rowspan="3">GCN</td><td rowspan="3">No manifold No LA No Tuning</td><td>83.67±0.08</td><td>91.48±0.23</td><td>82.50±0.18</td><td>52.78±0.21</td></tr><tr><td>84.18±0.11</td><td>91.77±0.42</td><td>83.12±0.18</td><td>53.65±1.61</td></tr><tr><td>83.54±0.71</td><td>91.21±0.48</td><td>82.11±0.98</td><td>52.78±1.09</td></tr><tr><td rowspan="3">GAT</td><td>Dagger</td><td>83.81±0.06</td><td>91.72±0.17</td><td>82.80±0.14</td><td>54.53±0.55</td></tr><tr><td>No manifold</td><td>84.08±0.43</td><td>92.11±0.16</td><td>82.83±0.50</td><td>59.07±0.42</td></tr><tr><td>No LA</td><td>83.89±0.06</td><td>92.26±0,17</td><td>82.75±0.11</td><td>57.34±1.49</td></tr><tr><td rowspan="3"></td><td>No Tuning Dagger</td><td>83.81±0.19</td><td>92.33±0.53</td><td>82.60±0.17</td><td>60.45±2.48</td></tr><tr><td></td><td>83.54±0.32</td><td>92.29±0.27</td><td>82.41±0.33</td><td>56.65±1.29</td></tr><tr><td>No manifold No LA</td><td>83.54±0.23 83.42±0.17</td><td>94.46±0.49</td><td>82.56±0.17</td><td>63.97±1.75</td></tr><tr><td rowspan="3">APPNP</td><td>No Tuning</td><td>83.35±0.25</td><td>94.34±0.12 94.15±0.53</td><td>82.39±0.17 82.31±0.18</td><td>63.43±1.01 63.16±1.72</td></tr><tr><td>Dagger</td><td>83.27±0.15</td><td>93.68±0.13</td><td>82.29±0.14</td><td>62.48±1.16</td></tr><tr><td>No manifold</td><td>84.30±0.20</td><td>93.36±0.47</td><td>83.15±0.12</td><td>59.32±2.85</td></tr><tr><td rowspan="3">GraphSAGE</td><td>No LA</td><td>84.31±0.38</td><td>93.31±0.37</td><td>83.25±0.31</td><td>57.78±1.47</td></tr><tr><td>No Tuning</td><td>84.48±0.35</td><td>93.54±0.67</td><td>83.33±0.27</td><td>59.66±3.75</td></tr><tr><td>Dagger</td><td>84.01±0.09</td><td>93.44±0.24</td><td>82.95±0.07</td><td>57.78±1.89</td></tr><tr><td rowspan="9">Physics</td><td rowspan="3">GCN</td><td>No manifold</td><td>94.33±0.05</td><td>95.46±0.09</td><td>92.21±0.11</td><td>42.96±1.05</td></tr><tr><td>No LA</td><td>95.04±0.23</td><td>96.44±0.15</td><td>93.40±0.34</td><td>49.63±2.10</td></tr><tr><td>No Tuning</td><td>94.86±0.18</td><td>96.53±0.12</td><td>93.26±0.26</td><td>47.41±4.57</td></tr><tr><td rowspan="3">GAT</td><td>Dagger</td><td>95.51±0.09</td><td>97.23±0.05</td><td>93.94±0.10</td><td>59.26±2.77</td></tr><tr><td>No manifold</td><td>93.68±0.21</td><td>93.54±0.12</td><td>91.58±0.30</td><td>48.48±2.14</td></tr><tr><td>No LA</td><td>94.80±0.18</td><td>94.36±0.21</td><td>93.08±0.18</td><td>52.27±1.86</td></tr><tr><td rowspan="3"></td><td>No Tuning</td><td>94.96±0.05</td><td>94.53±0.01</td><td>93.30±0.09</td><td>51.52±2.14</td></tr><tr><td>Dagger</td><td>95.22±0.05</td><td>94.98±0.04</td><td>93.55±0.06</td><td>56.06±2.14</td></tr><tr><td>No manifold</td><td>94.87±0.00</td><td>96.68±0.04</td><td>92.92±0.03</td><td>54.90±0.00</td></tr><tr><td rowspan="3">APPNP</td><td>No LA No Tuning</td><td>95.70±0.12</td><td>97.89±0.11</td><td>94.28±0.18</td><td>58.17±0.92</td></tr><tr><td></td><td>95.77±0.06</td><td>97.82±0.03</td><td>94.39±0.04</td><td>58.17±2.45</td></tr><tr><td>Dagger No manifold</td><td>95.46±0.09</td><td>97.60±0.09</td><td>93.86±0.12</td><td>54.90±1.60</td></tr><tr><td rowspan="3">GraphSAGE</td><td>No LA</td><td>94.89±0.08 95.04±0.23</td><td>95.65±0.10</td><td>92.74±0.11</td><td>42.96±1.05</td></tr><tr><td>No Tuning</td><td>95.08±0.05</td><td>96.44±0.15 95.74±0.05</td><td>93.40±0.34 93.34±0.08</td><td>49.63±2.10</td></tr><tr><td>Dagger</td><td></td><td></td><td></td><td>44.96±1.10</td></tr><tr><td rowspan="3"></td><td></td><td>95.42±0.06</td><td>96.42±0.05</td><td>93.96±0.07</td><td>51.94±2.19</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr></table>

![](images/19183e06f683f4e56617fbc8f339a72db83ceff1f8bc0ccca1a650675db94421.jpg)

Balance Level Influence — Fidelity  
![](images/f00c8abcfbcedb83260066a331b219163f4b6770acc6c8ba2eb469bff3c40a45.jpg)

![](images/7f810d593e4b8dec12d288babd38cc639b2900fb8288a568071fd63ac75b911e.jpg)

![](images/f2218fcc4f880998dbf3629540b5def3dc8a7eda61aebe4b8ce1a701aea867a2.jpg)

![](images/360e9e67cf5b6531af4ea43d0b2d062e08ba2b03fe6ba29ab3bb44ff3c34969c.jpg)  
Balance Level Influence — Accuracy  
Balance Level Influence — Boundary Fid.

![](images/6fa62147bd0ab87162befa0dbe0fd3ac87efe45eede68c088b86a4803b87658f.jpg)

![](images/75175f92d1c783e10870a41e9ee5ac4e327ba9638ef14a22f49abd4bdc45c0d4.jpg)  
Fig. 9: Balance Level Sensitivity and Other Factors