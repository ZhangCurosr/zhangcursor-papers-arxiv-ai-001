# Multi-View Molecular Representation Learning with Hierarchical Graphs and Contextualized Fingerprints

Gwang-Hyeon Yun   
Yonsei University   
Wonju-si, Republic of Korea   
ghyun0130@yonsei.ac.kr   
Helen Chen   
University of Waterloo   
Waterloo, Canada   
helen.chen@uwaterloo.ca Jong-Hoon Park<sup>∗</sup> Yonsei University   
Wonju-si, Republic of Korea   
jonghoon\_park@yonsei.ac.kr   
Anita Layton   
University of Waterloo   
Waterloo, Canada   
anita.layton@uwaterloo.ca   
Bing Hu   
University of Waterloo   
Waterloo, Canada   
b25hu@uwaterloo.ca

Young-Rae Cho<sup>†</sup> Yonsei University - Mirae Campus Wonju-si, Republic of Korea youngcho@yonsei.ac.kr

## Abstract

Molecular property prediction requires representations that gen eralize from limited labeled data to structurally novel compounds. Existing molecular pretraining methods often rely on a single view: graph-based approaches model atom-bond topology but provide limited fragment-level supervision, whereas fingerprint descriptors encode chemical patterns but are typically used as fixed auxiliary features. We propose HiFi-Mol, a multi-view framework that separately pretrains a hierarchical graph encoder and a contextualized fingerprint encoder before downstream integration. The graph branch uses fragment-aware masking with multi-resolution supervision to capture substructure-aware representations, while the fingerprint branch tokenizes active entries from seven fingerprint families and applies masked language modeling to learn contextualized embeddings. During fine-tuning, HiFi-Mol combines projected multi-resolution graph features with fingerprint embeddings for downstream prediction. Evaluated on MoleculeNet benchmarks under the scafold split, HiFi-Mol achieves a 2.77% improvement in average ROC-AUC over the best baseline across eight classification tasks while maintaining competitive performance on three regression tasks. Further analyses reveal that fragment-aware masking improves graph representation quality, and classification results demonstrate dataset-dependent strengths of the individual graph and fingerprint variants, confirming that the two views provide complementary predictive signals.

## CCS Concepts

• Applied computing → Bioinformatics; • Computing methodologies → Learning latent representations.

## Keywords

Multi-View Learning, Molecular Representation Learning, Molecular Property Prediction, Hierarchical Graph, Contextualized Fingerprint

ACM Reference Format: Gwang-Hyeon Yun, Jong-Hoon Park, Bing Hu, Helen Chen, Anita Layton, and Young-Rae Cho. 2026. Multi-View Molecular Representation Learning with Hierarchical Graphs and Contextualized Fingerprints. In Proceedings of the 35th ACM International Conference on Information and Knowledge Management (CIKM ’26), November 7–11, 2026, Rome, Italy. ACM, New York, NY, USA, 9 pages. https://doi.org/10.1145/3799682.3840948

## 1 Introduction

Molecular property prediction [6, 32] is a central problem in earlystage drug discovery, where computational models are used to prioritize promising compounds and filter toxic or inactive candidates before costly experimental validation [25, 28, 33]. Despite significant progress, two key challenges remain. First, labeled molecular data are often scarce, forcing models to learn from limited supervision [21]. Second, such models must generalize to structurally novel molecules beyond the training distribution, since newly evaluated compounds are often absent from the training data [1, 22]. These challenges place strong demands on molecular representation learning: useful representations should encode not only atom-level topology but also higher-order chemical structures and functional patterns that govern molecular activity.

Graph neural networks (GNNs) have become a dominant approach for molecular representation learning because molecules naturally form atom-bond graphs [14, 16, 30]. However, atom-level graph representations mainly capture local relational topology through message passing over atoms and bonds, making it dificult to explicitly model chemically meaningful substructures such as functional groups, ring systems, and pharmacophoric scafolds [19]. While recent methods such as MGSSL [44] and HiMol [43] incorporate fragment-level nodes, they either decouple fragment identities from their constituent atom-bond patterns or lack multi-resolution supervision spanning atom, fragment, global, and geometric signals.

Molecular fingerprints provide another chemically informed view of molecular structure [2, 7, 40]. They encode predefined descriptors such as circular substructures, pharmacophoric patterns, and topological relations. Although many fingerprint descriptors are derived from molecular topology, their discrete and predefined encoding scheme can capture structural patterns that are complementary to learned graph representations [9]. However, existing molecular pretraining methods rarely treat these two views as learnable, multi-view components within a unified framework: fingerprints are typically incorporated as static auxiliary vectors and shallowly concatenated with graph embeddings, limiting their ability to model contextual descriptor relationships.

Motivated by these observations, we propose HiFi-Mol, a multiview molecular pretraining framework composed of a Hierarchical graph encoder and a contextualized Fingerprint encoder. Rather than enforcing a shared pretraining space, HiFi-Mol learns graph and fingerprint views with view-specific objectives and integrates them during downstream adaptation. On the graph side, fragmentaware masking corrupts chemically coherent fragments together with their constituent atoms and bonds, while multi-resolution supervision over atom, bond, fragment, global, and geometric signals guides the learning of hierarchical representations with fragmentlevel chemical semantics. In parallel, the fingerprint encoder represents active entries from multiple fingerprint families as descriptor tokens, which are processed together with accompanying SMILES tokens for joint masked language modeling [10]. This objective captures contextual dependencies among fingerprint descriptors while leveraging SMILES-derived molecular context, yielding fingerprint representations beyond static descriptor vectors. Finally, projected graph and fingerprint representations are concatenated and passed to a task-specific predictor for downstream property prediction.

Experiments on MoleculeNet [37] benchmarks show strong classification performance and competitive regression results. Singleview comparisons reveal dataset-dependent strengths of graph and fingerprint representations, suggesting that the two views provide complementary predictive signals.

The main contributions are summarized as follows:

• HiFi-Mol formulates graph topology and fingerprint descriptors as complementary molecular views with distinct inductive biases, and pretrains each encoder with view-specific objectives before downstream integration.

• Fragment-aware masking and multi-resolution supervision jointly improve hierarchical graph representations by encouraging coherent substructure recovery across molecular resolutions.

• Masked language modeling over joint SMILES–fingerprint token sequences learns contextualized descriptor representations that capture co-occurrence patterns beyond static fingerprint vectors.

## 2 Related Work

Molecular Graph Pretraining. Self-supervised molecular pretraining has been widely studied to alleviate label scarcity in molecular property prediction. Existing graph-based methods can be broadly categorized into generative objectives, such as AttrMask and ContextPred [16], which reconstruct masked node, edge, or contextual attributes, and contrastive objectives, such as GraphCL [42], JOAOv2 [41], and MolCLR [35], which maximize agreement between augmented molecular views. Hybrid methods combine generative and contrastive pretraining signals to capture richer molecular semantics. GraphMVP [24] and MoleculeSDE [23] incorporate geometric information by aligning 2D graph representations with 3D molecular signals, whereas Mole-BERT [38] combines masked atom prediction with contrastive learning over multiple molecular views. While efective, these approaches typically rely on atom-level or graph-level objectives, leaving explicit supervision of chemically meaningful substructures such as functional groups, ring systems, and pharmacophores relatively underexplored.

![](images/9d26e82578bc9181567df71897345b4f26efa4106c71664036fd055edf1473c5.jpg)  
Figure 1: Illustration of the hierarchical molecular graph and fragment-aware masking (FAM). (a) A molecule is decomposed into fragment-level units using BRICS and SSSR, forming a hierarchical graph with atom, fragment, and global nodes. (b) FAM samples fragment nodes and propagates the mask to their constituent atoms and incident bonds, encouraging reconstruction of coherent substructures.

Fragment-level Graph Pretraining. Recent studies have incorporated fragment- or motif-level information into molecular graph learning. MGSSL [44] performs motif-based generative pretraining by predicting motif labels and topological structure, while HiMol [43] constructs hierarchical molecular graphs with virtual motif and graph-level nodes. GraphFP [26] further aligns molecular and fragment representations through fragment-based contrastive learning. These methods demonstrate the value of modeling higherorder substructures, yet existing objectives often decouple fragment identities from the atom-bond patterns that define them.

Fingerprint Representation Learning and Multi-View Learning. Molecular fingerprints represent molecules as binary or count vectors encoding predefined substructural patterns, providing a complementary view of chemical structure [11, 29]. Prior methods such as FP-GNN [4] and DGCL [18] suggest that combining fingerprint features with graph representations can improve downstream molecular property prediction, but fingerprints are typically used as static auxiliary descriptors and fused shallowly with graph embeddings. Recent fingerprint language modeling approaches such as DELBERT [31] show that sparse fingerprint vectors can be tokenized into discrete sequences and pretrained with masked language modeling, producing contextualized descriptor representations beyond fixed fingerprint vectors.

## 3 Preliminaries

We formally introduce the two molecular views underlying HiFi-Mol and formulate the multi-view property prediction problem.

Definition 1 (Hierarchical Molecular Graph). Given a molecule �, its hierarchical molecular graph is defined as $\mathcal { G } = ( \mathcal { V } , \mathcal { E } )$ where

$$
\mathcal { V } = \mathcal { V } _ { \mathrm { a t o m } } \cup \mathcal { V } _ { \mathrm { f r a g } } \cup \{ v _ { \mathrm { g l o b a l } } \} ,\tag{1}
$$

$$
\mathcal { E } = \mathcal { E } _ { \mathrm { b o n d } } \cup \mathcal { E } _ { \mathrm { h i e r } } .\tag{2}
$$

$\mathcal { N } _ { \mathrm { a t o m } }$ consists of atom nodes, $\mathcal { V } _ { \mathrm { f r a g } }$ consists of fragment nodes corresponding to chemically coherent substructures, and $v _ { \mathrm { g l o b a l } }$ is a virtual global node representing the entire molecule. $\mathcal { E } _ { \mathrm { b o n d } }$ contains chemical bonds, and $\mathcal { E } _ { \mathrm { h i e r } }$ contains directed edges connecting atom, fragment, and global levels.

Definition 2 (Joint SMILES–Fingerprint Token Sequence). Let $\mathcal { R } _ { \mathrm { F P } } ~ = ~ \{ R _ { 1 } , . . . , R _ { K } \}$ denote a set of fingerprint families. For each active entry in family $R _ { k }$ , we construct a discrete token $t _ { p } =$ <fp\_name\_index\_value> encoding its fingerprint type, bit index, and count value. The sparse fingerprint token sequence is denoted by:

$$
X _ { \mathrm { F P } } = ( t _ { P } \mid t _ { P } \in \mathrm { A c t i v e } ( R _ { k } ) , \forall R _ { k } \in \mathcal { R } _ { \mathrm { F P } } ) .\tag{3}
$$

Given the SMILES token sequence $X _ { \mathrm { S M I } }$ , the joint token sequence is defined as:

$$
X _ { \mathrm { S F } } = [ X _ { \mathrm { S M I } } \parallel X _ { \mathrm { F P } } ] ,\tag{4}
$$

which serves as the input to the fingerprint encoder.

Problem 1 (Multi-View Molecular Property Prediction). Given a molecular dataset $\boldsymbol { \mathcal { D } } = \{ ( M _ { n } , y _ { n } ) \} _ { n = 1 } ^ { N } ,$ where $y _ { n }$ denotes the ground-truth property label, each molecule $M _ { n }$ is represented by a hierarchical graph $\mathcal { G } _ { n }$ and a joint token sequence $X _ { \mathrm { S F } , n }$ . The goal is to learn a prediction function $f _ { \mathrm { p r e d } } \colon$

$$
\hat { y } _ { n } = f _ { \mathrm { p r e d } } \big ( \Phi _ { \mathrm { G N N } } ( \mathcal { G } _ { n } ) \big | \big | \Phi _ { \mathrm { F P } } ( X _ { \mathrm { S F } , n } ) \big ) ,\tag{5}
$$

where $\Phi _ { \mathrm { G N N } }$ and $\Phi _ { \mathrm { F P } }$ denote the graph and fingerprint encoders, respectively, ∥ denotes feature concatenation, and ${ \hat { y } } _ { n }$ is the predicted molecular property.

## 4 Proposed Method

Figure 2 illustrates HiFi-Mol, a multi-view pretraining framework for molecular property prediction, consisting of three stages: hierarchical graph pretraining, contextualized fingerprint pretraining, and multi-view downstream integration.

## 4.1 Hierarchical Graph Pretraining

Hierarchical Graph Construction. As shown in Figure 1(a), fragment nodes $\mathcal { V } _ { \mathrm { f r a g } }$ are constructed via a two-stage decomposition. The BRICS algorithm [8] first cleaves retrosynthetically meaningful bonds to isolate initial fragments. SSSR-based ring refinement then further partitions coarse ring-containing fragments into the small est chemically coherent cyclic substructures. Each resulting frag ment is canonicalized into a SMILES string and mapped to a unique ID through a dictionary function $\phi : S _ { \mathrm { f r a g } }  \{ 0 , . . . , | \mathcal { D } _ { \mathrm { f r a g } } | - 1 \}$ where fragments occurring fewer than $\tau _ { \mathrm { m i n } } = 1 0$ times are assigned to an ⟨UNK⟩ token.

Fragment-aware Masking. As illustrated in Figure 1(b), fragmentaware masking (FAM) is applied to the hierarchical graph $\mathcal { G }$ as a corruption procedure. Unlike atom-level masking, which can often be inferred from local neighborhoods, FAM selects fragment nodes as masking units, requiring the encoder to recover coherent substructures rather than isolated graph elements. Formally, a subset of fragment nodes $\mathcal { M } _ { \mathrm { f r a g } } \subset \mathcal { V } _ { \mathrm { f r a g } }$ is sampled at ratio $r _ { f } = 0 . 1 5$ and designated as masked fragments. The corresponding atom mask is defined as

$$
\mathcal { M } _ { \mathrm { a t o m } } = \left\{ v _ { u } \in \mathcal { V } _ { \mathrm { a t o m } } \ | \ \exists v _ { s } \in \mathcal { M } _ { \mathrm { f r a g } } \ \mathrm { s . t . } \ v _ { u } \in \mathcal { A } ( s ) \right\} ,\tag{6}
$$

where $v _ { u }$ and $v _ { s }$ denote the atom and fragment nodes indexed by � and $s ,$ respectively, and ${ \mathcal { A } } ( s )$ is the set of atom nodes belonging to fragment �. If the induced atom masking ratio falls below $r _ { a } = $ 0.25, additional atoms are uniformly sampled to ensure suficient corruption. The set of masked bonds is defined as

$$
\begin{array} { r } { \mathcal { M } _ { \mathrm { b o n d } } = \{ ( v _ { u } , v _ { v } ) \in \mathcal { E } _ { \mathrm { b o n d } } \ | \ v _ { u } \in \mathcal { M } _ { \mathrm { a t o m } } \ \mathrm { o r } \ v _ { v } \in \mathcal { M } _ { \mathrm { a t o m } } \} , } \end{array}\tag{7}
$$

where $( v _ { u } , v _ { v } )$ denotes a chemical bond incident to at least one masked atom. This produces a masked graph ${ \tilde { g } } ,$ in which masked nodes $M _ { \mathrm { a t o m } } \cup M _ { \mathrm { f r a g } }$ and masked bonds $M _ { \mathrm { b o n d } }$ serve as reconstruction targets for the encoder.

Hierarchical Graph Encoding. Given the masked hierarchical graph ${ \tilde { g } } ,$ , node and edge features are represented as $h _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ and $e _ { i j } \in \mathbb { R } ^ { d }$ , respectively. Node features are derived from atom attributes, fragment dictionary identities, or global molecular context depending on the node type, with masked atom and fragment nodes replaced by a learnable mask embedding. Edge features incorporate bond attributes and hierarchical relation types, enabling information propagation over both chemical bonds and cross-resolution connections. Detailed feature initialization procedures are provided in Appendix A.

A shared GINE backbone [16] encodes $\tilde { \mathcal { G } }$ by performing message passing over both chemical and hierarchical edges. At each layer �, the representation of node $v _ { i }$ is updated as:

$$
h _ { i } ^ { ( l ) } = \mathrm { M L P } ^ { ( l ) } \left( \left( 1 + \epsilon ^ { ( l ) } \right) h _ { i } ^ { ( l - 1 ) } + \sum _ { j \in N ( i ) } \mathrm { R e L U } \left( h _ { j } ^ { ( l - 1 ) } + e _ { i j } \right) \right) ,\tag{8}
$$

where $N ( i )$ denotes the neighbors of $v _ { i }$ in $\tilde { \mathcal { G } } _ { : }$ , and $\epsilon ^ { ( l ) }$ is a learnable scalar that controls the contribution of the previous-layer representation of $v _ { i } .$ After � layers, the encoder produces final-layer node representations $\tilde { h } _ { i } ^ { ( L ) } \in \mathbb { R } ^ { d }$ , which capture multi-resolution chemical context across atom, fragment, and global levels.

Multi-resolution Pretraining Objectives. The final-layer representations $\{ \tilde { h } _ { i } ^ { ( L ) } \}$ are optimized with five objectives that supervise molecule-level consistency, masked substructure recovery, local atom-bond reconstruction, and geometric regularization.

For molecule-level supervision, we generate two independently masked views, $\tilde { g } ^ { ( 1 ) }$ and $\tilde { g } ^ { ( 2 ) }$ , for each molecule using FAM. Their global node embeddings are projected and $\ell _ { 2 } \cdot$ -normalized to obtain $\bar { z } ^ { ( 1 ) }$ and $z ^ { ( 2 ) }$ , which are then optimized with an InfoNCE loss [5]:

$$
\mathcal { L } _ { \mathrm { C L } } = - \mathbb { E } \left[ \log \frac { e ^ { \langle z ^ { ( 1 ) } , z ^ { ( 2 ) } \rangle / \tau } } { e ^ { \langle z ^ { ( 1 ) } , z ^ { ( 2 ) } \rangle / \tau } + \sum _ { z ^ { - } \in \mathcal { Z } ^ { - } } e ^ { \langle z ^ { ( 1 ) } , z ^ { - } \rangle / \tau } } \right] ,\tag{9}
$$

where $\tau = 0 . 1$ is the temperature parameter that scales the similarity logits, and $z ^ { . }$ <sup>−</sup> consists of in-batch negatives from diferent molecules.

For substructure-level supervision, masked fragment identities and atom types are recovered via linear classification heads over their respective masked node representations:

$$
\mathcal { L } _ { \mathrm { f r a g } } = - \sum _ { v _ { s } \in \mathcal { M } _ { \mathrm { f r a g } } } \log P \Big ( \phi ( s ) \mid \tilde { h } _ { s } ^ { ( L ) } \Big ) ,\tag{10}
$$

![](images/15d8453adc802a13e8c6a081274a7c1b4b3e014f2a353441dceacdc75d7ca96d.jpg)  
Figure 2: Overview of the HiFi-Mol framework. (a) Given a hierarchical molecular graph, fragment-aware masking produces two masked views that are passed through a shared 5-layer GINE encoder and optimized with global, fragment, atom, bond, and distance prediction objectives. (b) SMILES and multi-type fingerprint tokens are concatenated and processed by ChemFLM via masked language modeling. (c) For downstream prediction, the pretrained encoders provide multi-resolution graph features and contextualized fingerprint embeddings, which are projected, concatenated, and passed to a task-specific prediction head.

$$
\mathcal { L } _ { \mathrm { a t o m } } = - \sum _ { v _ { u } \in \mathcal { M } _ { \mathrm { a t o m } } } \log P \Bigl ( a _ { u } \mid \tilde { h } _ { u } ^ { ( L ) } \Bigr ) ,\tag{11}
$$

while bond types are predicted by a two-layer MLP over concatenated endpoint representations:

$$
\mathcal { L } _ { \mathrm { b o n d } } = - \sum _ { ( v _ { u } , v _ { v } ) \in \mathcal { M } _ { \mathrm { b o n d } } } \log P \Big ( c _ { u v } \mid \tilde { h } _ { u } ^ { ( L ) } , \tilde { h } _ { v } ^ { ( L ) } \Big ) .\tag{12}
$$

In these losses, $\phi ( s ) , a _ { u } ,$ and $c _ { u v }$ denote the ground-truth fragment dictionary ID, atom type of atom node $v _ { u } ,$ , and bond type between atom nodes $v _ { u }$ and $v _ { v } ,$ respectively.

For geometry-level supervision, a distance decoder $f _ { \mathrm { d i s t } }$ predicts the Euclidean distance between sampled atom pairs from their final representations:

$$
\mathcal { L } _ { \mathrm { d i s t } } = \sum _ { ( v _ { u } , v _ { v } ) \in \mathcal { P } } \left( f _ { \mathrm { d i s t } } ( \tilde { h } _ { u } ^ { ( L ) } , \tilde { h } _ { v } ^ { ( L ) } ) - \| _ { \mathcal { P } _ { u } } - p _ { v } \| _ { 2 } \right) ^ { 2 } ,\tag{13}
$$

where $\mathcal { P }$ denotes sampled atom pairs including bonded and randomly sampled non-bonded pairs, and $\boldsymbol { p _ { u } } , \boldsymbol { p _ { v } } \in \bar { \mathbb { R } } ^ { 3 }$ are the 3D coordinates available only during pretraining.

Uncertainty-based Loss Weighting. To balance the five pretraining objectives without manual tuning, we adopt uncertaintybased loss weighting [20]. Let $\sigma _ { t } ^ { 2 }$ denote a learnable variance parameter for objective �. The total graph pretraining loss is:

$$
\mathcal { L } _ { \mathrm { G } } = \sum _ { t \in \mathcal { T } } \left( \frac { 1 } { 2 \sigma _ { t } ^ { 2 } } \mathcal { L } _ { t } + \log \sigma _ { t } \right) ,\tag{14}
$$

where T denotes the set of pretraining objectives. The first term adaptively down-weights objectives with higher uncertainty, while the log-variance term prevents trivial growth of $\sigma _ { t }$

## 4.2 Contextualized Fingerprint Pretraining

To complement the graph-based structural view, we independently pretrain a fingerprint encoder $\Phi _ { \mathrm { F P } }$ by reframing molecular fingerprints as discrete token sequences. The fingerprint view is constructed from seven complementary fingerprint families capturing chemical patterns across multiple structural scales, including local circular environments (ECFP4, ECFP6), pharmacophoric features (FCFP4, FCFP6), long-range topological relations (ATOM-PAIR), torsional fragments (TOPTOR), and interpretable structural keys (MACCS). $\Phi _ { \mathrm { F P } }$ is initialized from ChemFLM, a ModernBERTbased fingerprint language model [31, 36], enabling contextualized descriptor modeling beyond fixed fingerprint vectors.

Given the joint token sequence $X _ { \mathrm { S F } }$ defined in Section 3, token and segment embeddings are summed to form the input representations:

$$
H _ { \mathrm { S F } } ^ { ( 0 ) } = E _ { \mathrm { t o k } } ( X _ { \mathrm { S F } } ) + E _ { \mathrm { s e g } } ( X _ { \mathrm { S F } } ) ,\tag{15}
$$

where $E _ { \mathrm { t o k } }$ maps discrete tokens to embeddings and $E _ { \mathrm { s e g } }$ distinguishes whether each token belongs to the SMILES segment or one of the fingerprint descriptor families. The initialized sequence is processed by stacked self-attention layers:

$$
\begin{array} { r } { H _ { \mathrm { S F } } ^ { ( l ) } = \Phi _ { \mathrm { F P } } ^ { ( l ) } \left( H _ { \mathrm { S F } } ^ { ( l - 1 ) } \right) , \quad l = 1 , \dots , L _ { \mathrm { F P } } , } \end{array}\tag{16}
$$

yielding final hidden states $H _ { \mathrm { S F } } = H _ { \mathrm { S F } } ^ { ( L _ { \mathrm { F P } } ) } \in \mathbb { R } ^ { | X _ { \mathrm { S F } } | \times d _ { \mathrm { F P } } }$

During pretraining, a masked sequence $\tilde { X } _ { \mathrm { S F } }$ is constructed by randomly replacing 15% of joint tokens with a mask token. Let $\tilde { H } _ { \mathrm { S F } } = \Phi _ { \mathrm { F P } } ( \tilde { X } _ { \mathrm { S F } } )$ denote the final hidden states obtained from the masked input. For each masked position $p \in { \mathcal { M } } _ { \mathrm { S F } }$ , an MLM head predicts the original token $x _ { p }$ from the corresponding contextualized representation $\tilde { H } _ { \mathrm { S F } , p }$ . The encoder is trained to reconstruct the original tokens at masked positions:

$$
\mathcal { L } _ { \mathrm { F P } } = - \sum _ { p \in \mathcal { M } _ { \mathrm { S F } } } \log P \left( x _ { p } \mid \tilde { X } _ { \mathrm { S F } } \right) ,\tag{17}
$$

where $x _ { p }$ denotes the original token at masked position $\mathcal { P } \cdot$ By maintaining $\Phi _ { \mathrm { F P } }$ as a standalone module separate from $\Phi _ { \mathrm { G N N } } ,$ HiFi-Mol preserves descriptor-level chemical semantics distinct from topology-driven graph representations.

## 4.3 Multi-View Downstream Integration

Since the graph and fingerprint encoders capture distinct chemical views, HiFi-Mol combines their representations only during downstream adaptation after view-specific pretraining. This design preserves the hierarchical structural information encoded across atom, fragment, and global levels, as well as the descriptor-level chemical knowledge captured by the fingerprint encoder.

Molecule-level representations are derived independently from each view. Given the hierarchical graph $^ { \mathcal { G } , }$ the graph encoder Φ<sub>GNN</sub> yields final-layer representations at three levels: atom-level $H _ { \mathrm { a t o m } } .$ fragment-level $H _ { \mathrm { f r a g } } ,$ and global-level $h _ { \mathrm { g l o b a l } }$ . A molecule-level atom summary $h _ { \mathrm { a t o m } }$ is obtained by mean pooling over $H _ { \mathrm { a t o m } } ,$ a fragment summary $h _ { \mathrm { f r a g \it } }$ by max pooling over $H _ { \mathrm { f r a g } } ,$ , and $h _ { \mathrm { g l o b a l } }$ is retained directly. For the fingerprint view, $\Phi _ { \mathrm { F P } }$ processes $X _ { \mathrm { S F } }$ without masking to produce final hidden states $H _ { \mathrm { S F } }$ . The molecule-level representation $h _ { \mathrm { F P } }$ is obtained by mean pooling over $H _ { \mathrm { S F } }$

These view-specific summaries are then projected into compati ble embedding spaces, concatenated, and passed to a task-specific prediction head:

$$
z _ { g } = f _ { \mathrm { G N N } } \big ( \left[ h _ { \mathrm { a t o m } } \right] \big | h _ { \mathrm { f r a g } } \big | \big | h _ { \mathrm { g l o b a l } } \big ] \big ) ,
$$

$$
z _ { \mathrm { f p } } = f _ { \mathrm { F P } } ( h _ { \mathrm { F P } } ) ,\tag{18}
$$

$$
{ \hat { y } } = f _ { \mathrm { p r e d } } \big ( \left[ { z } _ { g } \parallel { z } _ { \mathrm { f p } } \right] \big ) .\tag{19}
$$

(20)

In this implementation, �<sub>GNN</sub> and �<sub>FP</sub> are two-layer MLP projection heads with batch normalization, SiLU activation, and dropout,

Table 1: Dataset statistics and downstream hyperparameter settings for MoleculeNet benchmarks. All datasets use batch size 32, weight decay 0, and dropout 0.1.
<table><tr><td>Dataset</td><td>Avg. # Atoms</td><td>Avg. # Frags</td><td>LR</td><td>Epochs</td></tr><tr><td colspan="5">Classification benchmarks</td></tr><tr><td>BBBP</td><td>24.1</td><td>6.5</td><td> $1 0 ^ { - 4 }$ </td><td>20</td></tr><tr><td>Tox21</td><td>18.6</td><td>4.9</td><td> $1 0 ^ { - 5 }$ </td><td>100</td></tr><tr><td>ToxCast</td><td>18.8</td><td>5.1</td><td> $1 0 ^ { - 4 }$ </td><td>100</td></tr><tr><td>HIV</td><td>25.5</td><td>6.7</td><td> $1 0 ^ { - 5 }$ </td><td>20</td></tr><tr><td>MUV</td><td>24.2</td><td>7.1</td><td> $1 0 ^ { - 4 }$ </td><td>10</td></tr><tr><td>ClinTox</td><td>26.2</td><td>7.0</td><td> $1 0 ^ { - 5 }$ </td><td>100</td></tr><tr><td>BACE</td><td>34.1</td><td>9.4</td><td> $1 0 ^ { - 5 }$ </td><td>20</td></tr><tr><td>SIDER</td><td>33.6</td><td>9.0</td><td> $1 0 ^ { - 4 }$ </td><td>20</td></tr><tr><td colspan="5">Regression benchmarks</td></tr><tr><td>ESOL</td><td>13.3</td><td>3.4</td><td> $1 0 ^ { - 4 }$ </td><td>150</td></tr><tr><td>FreeSolv</td><td>8.7</td><td>2.2</td><td> $1 0 ^ { - 4 }$ </td><td>150</td></tr><tr><td>Lipophilicity</td><td>27.0</td><td>7.8</td><td> $1 0 ^ { - 4 }$ </td><td>100</td></tr></table>

producing 128-dimensional graph and fingerprint embeddings. Similarly, $f _ { \mathrm { p r e d } }$ uses the same MLP block to map the concatenated representation to the task output.

## 5 Experiments

## 5.1 Experimental Settings

Datasets and Metrics. The graph encoder is pretrained on PCQM4Mv2 [15], containing 3.3 million molecules with 3D coordinates used only for distance prediction. The fingerprint encoder is initialized from ChemFLM, pretrained on 4.4 million molecules from AIRCHECK [12], ChEMBL [13], and MOSES [27]. We evaluate on 11 MoleculeNet benchmarks under the scafold split [37]: eight classification datasets and three regression datasets, whose statistics are summarized in Table 1. Performance is measured by ROC-AUC for classification and RMSE for regression.

Baselines. We compare against representative molecular pretraining methods from four categories: generative SSL methods (Attr-Mask [16], ContextPred [16], GPT-GNN [17]), contrastive SSL methods (InfoGraph [34], GraphCL [42], JOAOv2 [41], MolCLR [35]), hybrid SSL methods (GraphMVP [24], MoleculeSDE [23], Mole-BERT [38]), and fragment-based SSL methods (MGSSL [44], Hi-Mol [43]). Baseline results are reproduced using oficially released implementations under the same scafold split protocol and metric settings.

Implementation Details. The graph encoder uses 5 GINE layers with hidden dimension 300, pretrained for 100 epochs with batch size 1,024, learning rate $1 0 ^ { - 4 } $ , AdamW optimizer, and a OneCycle scheduler with 10% warmup followed by cosine annealing. Fragment masking ratio is set to $r _ { f } = 0 . 1 5$ , with additional random atom masking to maintain a minimum atom masking ratio of $r _ { a } = 0 . 2 5$ The fingerprint encoder uses a ModernBERT large encoder with 12 layers and hidden size 1024, pretrained with learning rate $1 0 ^ { - 4 }$ AdamW optimizer, batch size 64, and a cosine annealing schedule with 10% warmup. Downstream hyperparameters are summarized

Table 2: ROC-AUC (↑) on MoleculeNet classification benchmarks under the scafold split (mean ± std over 3 seeds). Best in bold, second-best underlined. $\mathbf { H i F i - M o l } _ { \mathrm { G r a p h } }$ and $\mathbf { H i F i - M o l _ { F P } }$ denote graph-only and fingerprint-only variants, respectively.
<table><tr><td>Dataset</td><td>BBBP</td><td>Tox21</td><td>ToxCast</td><td>HIV</td><td>MUV</td><td>ClinTox</td><td>BACE</td><td>SIDER</td><td> $\operatorname { A v g . A U C }$ </td></tr><tr><td># Molecules # Tasks</td><td>2,039 1</td><td>7,831 12</td><td>8,577 617</td><td>41,127 1</td><td>93,087 17</td><td>1,477 2</td><td>1,513</td><td>1,427 27</td><td></td></tr><tr><td>AttrMask [16]</td><td> $6 9 . 2 5 { \scriptstyle \pm 1 . 8 5 }$ </td><td> $7 4 . 4 6 { \scriptstyle \pm 0 . 2 0 }$ </td><td></td><td></td><td></td><td></td><td>1</td><td></td><td>-</td></tr><tr><td>ContextPred [16]</td><td> $6 7 . 2 5 { \scriptstyle \pm 1 . 8 6 }$ </td><td> $7 3 . 7 3 { \scriptstyle \pm 0 . 4 4 }$ </td><td> $6 2 . 9 0 { \scriptstyle \pm 0 . 4 7 }$   $6 1 . 6 1 { \scriptstyle \pm 0 . 7 5 }$ </td><td> $7 5 . 4 9 { \scriptstyle \pm 0 . 8 6 }$   $7 4 . 2 6 { \scriptstyle \pm 1 . 5 6 }$ </td><td> $7 3 . 0 3 { \scriptstyle \pm 0 . 7 6 }$   $7 3 . 5 3 { \scriptstyle \pm 2 . 4 2 }$ </td><td> $7 1 . 2 8 { \scriptstyle \pm 8 . 2 4 }$   $7 6 . 0 3 { \scriptstyle \pm 3 . 6 7 }$ </td><td> $7 8 . 3 6 { \scriptstyle \pm 0 . 8 7 }$   $7 5 . 9 4 { \scriptstyle \pm 0 . 9 6 }$ </td><td> $5 8 . 0 9 { \scriptstyle \pm 0 . 5 3 }$   $5 9 . 2 0 { \scriptstyle \pm 0 . 4 4 }$ </td><td>70.36 70.19</td></tr><tr><td>GPT-GNN [17]</td><td> $6 1 . 3 3 { \scriptstyle \pm 1 . 5 9 }$ </td><td> $7 4 . 3 6 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $6 2 . 9 3 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $7 6 . 8 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $7 5 . 5 3 { \scriptstyle \pm 0 . 7 2 }$ </td><td> $5 4 . 4 7 { \scriptstyle \pm 1 . 5 1 }$ </td><td> $7 4 . 5 1 { \scriptstyle \pm 1 . 1 7 }$ </td><td> $5 7 . 1 3 { \scriptstyle \pm 1 . 7 4 }$ </td><td>67.14</td></tr><tr><td>InfoGraph [34]</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td></td><td> $6 8 . 5 4 { \scriptstyle \pm 1 . 3 9 }$ </td><td> $7 3 . 6 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $6 0 . 9 8 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $7 5 . 6 1 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $7 4 . 4 7 { \scriptstyle \pm 2 . 9 4 }$ </td><td> $6 9 . 0 2 { \scriptstyle \pm 4 . 6 5 }$ </td><td> $6 1 . 3 0 { \scriptstyle \pm 3 . 4 1 }$ </td><td> $5 8 . 2 0 { \scriptstyle \pm 1 . 1 0 }$ </td><td>67.72</td></tr><tr><td>GraphCL [42]</td><td> $6 9 . 6 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $7 5 . 1 7 { \scriptstyle \pm 0 . 2 3 }$ </td><td> $6 2 . 8 3 { \scriptstyle \pm 0 . 8 7 }$ </td><td> $7 5 . 7 2 { \scriptstyle \pm 1 . 2 6 }$ </td><td> $7 4 . 3 0 { \scriptstyle \pm 1 . 6 6 }$ </td><td> $7 1 . 5 6 { \scriptstyle \pm 8 . 0 8 }$ </td><td> $7 3 . 0 0 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $6 1 . 1 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td>70.42</td></tr><tr><td>JOAOv2 [41]</td><td> $6 9 . 8 3 { \scriptstyle \pm 0 . 7 9 }$ </td><td> $7 4 . 7 6 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $6 3 . 3 0 { \scriptstyle \pm 0 . 9 0 }$ </td><td> $7 4 . 2 7 { \scriptstyle \pm 2 . 3 1 }$ </td><td> $7 4 . 9 4 { \scriptstyle \pm 1 . 1 5 }$ </td><td> $6 6 . 4 4 { \scriptstyle \pm 1 . 2 5 }$ </td><td> $7 5 . 7 7 { \scriptstyle \pm 0 . 3 5 }$ </td><td> $6 0 . 6 4 { \scriptstyle \pm 1 . 0 1 }$ </td><td>69.99</td></tr><tr><td>MolCLR [35]</td><td> $\underline { { 7 2 . 1 6 { \pm } 0 . 3 4 } }$ </td><td> $7 3 . 6 3 { \scriptstyle \pm 0 . 2 8 }$ </td><td> $6 2 . 1 8 { \scriptstyle \pm 0 . 5 6 }$ </td><td> $7 7 . 6 4 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $7 6 . 5 5 { \scriptstyle \pm 1 . 0 6 }$ </td><td> $9 0 . 6 4 { \scriptstyle \pm 1 . 6 1 }$ </td><td> $\underline { { 8 2 . 1 3 \pm 1 . 0 6 } }$ </td><td> $5 7 . 8 3 { \scriptstyle \pm 0 . 7 1 }$ </td><td>74.10</td></tr><tr><td>GraphMVP [24]</td><td> $6 8 . 4 4 { \scriptstyle \pm 0 . 5 7 }$ </td><td> $7 4 . 2 6 { \scriptstyle \pm 1 . 0 0 }$ </td><td> $6 4 . 2 5 { \scriptstyle \pm 0 . 4 6 }$ </td><td> $7 5 . 7 4 { \scriptstyle \pm 1 . 9 1 }$ </td><td> $7 5 . 6 1 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $7 2 . 8 4 { \scriptstyle \pm 1 . 8 9 }$ </td><td> $8 0 . 8 4 { \scriptstyle \pm 2 . 7 1 }$ </td><td> $6 1 . 3 8 { \pm } 1 . 0 9$ </td><td>71.67</td></tr><tr><td>MoleculeSDE [23]</td><td> $7 0 . 6 8 { \scriptstyle \pm 0 . 9 7 }$ </td><td> $7 5 . 0 4 { \scriptstyle \pm 0 . 2 4 }$ </td><td> $6 2 . 6 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $7 6 . 8 9 { \scriptstyle \pm 0 . 5 0 }$ </td><td> $7 6 . 2 2 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $7 2 . 4 5 { \scriptstyle \pm 3 . 0 7 }$ </td><td> $8 0 . 7 4 { \scriptstyle \pm 0 . 5 9 }$ </td><td> $5 9 . 8 2 { \scriptstyle \pm 0 . 3 5 }$ </td><td>71.81</td></tr><tr><td>Mole-BERT [38]</td><td> $6 7 . 5 7 { \scriptstyle \pm 1 . 0 8 }$ </td><td> $7 5 . 4 5 { \scriptstyle \pm 0 . 5 1 }$ </td><td> $6 2 . 9 3 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $7 7 . 5 6 { \scriptstyle \pm 0 . 6 3 }$ </td><td> $7 6 . 6 9 { \scriptstyle \pm 0 . 8 8 }$ </td><td> $7 0 . 8 7 { \scriptstyle \pm 5 . 9 1 }$ </td><td> $7 8 . 5 7 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $5 8 . 2 1 { \scriptstyle \pm 0 . 3 8 }$ </td><td>70.98</td></tr><tr><td>MGSSL [44]</td><td> $6 8 . 0 6 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $7 5 . 6 5 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $6 2 . 7 6 { \scriptstyle \pm 0 . 2 6 }$ </td><td> $7 3 . 7 8 { \scriptstyle \pm 1 . 3 4 }$ </td><td> $7 4 . 4 5 { \scriptstyle \pm 1 . 4 2 }$ </td><td> $6 6 . 9 6 { \scriptstyle \pm 4 . 8 0 }$ </td><td> $\underline { { 8 2 . 1 3 \pm 1 . 6 4 } }$ </td><td> $5 8 . 0 4 { \scriptstyle \pm 1 . 2 6 }$ </td><td>70.23</td></tr><tr><td>HiMol [43]</td><td> $7 0 . 6 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $7 6 . 8 2 { \scriptstyle \pm 0 . 8 3 }$ </td><td> $\underline { { 6 5 . 3 7 \pm 0 . 3 5 } }$ </td><td> $7 3 . 0 4 { \scriptstyle \pm 1 . 8 2 }$ </td><td> $\underline { { 7 6 . 8 0 { \pm } 0 . 9 7 } }$ </td><td> $6 0 . 7 9 { \scriptstyle \pm 1 . 3 0 }$ </td><td> $8 0 . 3 1 { \scriptstyle \pm 0 . 6 8 }$ </td><td> $5 8 . 6 6 { \scriptstyle \pm 3 . 2 9 } $ </td><td>70.30</td></tr><tr><td> $\mathrm { H i F i - M o l _ { G r a p h } }$ </td><td> $7 0 . 6 8 { \scriptstyle \pm 1 . 5 3 }$ </td><td> $7 4 . 7 6 { \scriptstyle \pm 0 . 2 3 }$ </td><td> ${ \bf 6 6 . 0 5 { \scriptstyle \pm 0 . 2 7 } }$ </td><td> $7 5 . 4 9 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $7 4 . 8 2 { \scriptstyle \pm 0 . 6 1 }$ </td><td> $8 8 . 8 4 { \scriptstyle \pm 1 . 9 1 }$ </td><td> $\mathbf { 8 2 . 1 6 { \scriptstyle \pm 0 . 6 1 } }$ </td><td> $6 1 . 0 3 { \scriptstyle \pm 0 . 4 0 }$ </td><td>74.23</td></tr><tr><td> $_ \mathrm { H i F i - M o l _ { F P } }$ </td><td> $7 0 . 0 9 { \scriptstyle \pm 0 . 9 3 }$ </td><td> $7 4 . 0 3 { \scriptstyle \pm 0 . 3 9 }$ </td><td> $6 4 . 8 6 { \scriptstyle \pm 0 . 8 2 }$ </td><td> $7 8 . 2 0 { \scriptstyle \pm 0 . 6 0 }$ </td><td> $7 5 . 6 2 { \scriptstyle \pm 1 . 4 0 }$ </td><td> $\underline { { 9 1 . 6 8 \pm 6 . 3 8 } }$ </td><td> $8 0 . 4 9 { \scriptstyle \pm 0 . 9 0 }$ </td><td> ${ \bf 6 4 . 2 3 { \scriptstyle \pm 1 . 3 9 } }$ </td><td>74.90</td></tr><tr><td> $_ \mathrm { H i F i - M o l }$ </td><td> ${ 7 3 . 6 8 \pm 0 . 5 5 }$ </td><td> $7 4 . 5 6 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $6 5 . 2 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $7 7 . 1 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 7 . 2 9 { \scriptstyle \pm 2 . 3 6 }$ </td><td> $\mathbf { 9 6 . 8 3 { \scriptstyle \pm 0 . 9 5 } }$ </td><td> $8 1 . 8 4 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 2 . 6 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td>76.15</td></tr></table>

<table><tr><td>Dataset</td><td>ESOL</td><td>FreeSolv</td><td>Lipophilicity</td></tr><tr><td># Molecules # Tasks</td><td>1,128</td><td>642</td><td>4,200</td></tr><tr><td>MolCLR [35]</td><td>1</td><td>1</td><td>1</td></tr><tr><td>GraphMVP [24]</td><td>1.368±0.012</td><td> $2 . 9 4 0 { \scriptstyle \pm 0 . 0 9 9 }$ </td><td> $0 . 7 3 2 { \scriptstyle \pm 0 . 0 1 4 }$ </td></tr><tr><td>Mole-BERT [38]</td><td> $1 . 2 9 4 { \scriptstyle \pm 0 . 0 9 7 }$ </td><td> $2 . 7 4 1 { \scriptstyle \pm 0 . 2 6 6 }$ </td><td> $\underline { { 0 . 7 2 7 { \scriptstyle \pm 0 . 0 0 3 } } }$ </td></tr><tr><td>MGSSL [44]</td><td> $1 . 3 4 9 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $2 . 6 9 6 { \scriptstyle \pm 0 . 0 6 3 }$ </td><td> $0 . 7 6 6 { \scriptstyle \pm 0 . 0 0 7 }$ </td></tr><tr><td>HiMol [43]</td><td> $1 . 3 7 4 { \scriptstyle \pm 0 . 0 2 0 }$   $0 . 8 8 4 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $3 . 0 7 6 { \scriptstyle \pm 0 . 2 1 4 }$   $\mathbf { 2 . 0 4 2 { \scriptstyle \pm 0 . 0 6 0 } }$ </td><td> $0 . 7 7 9 { \scriptstyle \pm 0 . 0 0 5 }$  0.743±0.009</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td> $\mathrm { H i F i - M o l _ { G r a p h } }$ </td><td>0.861±0.051</td><td> $2 . 6 4 4 { \scriptstyle \pm 0 . 1 9 6 }$ </td><td>0.718±0.006</td></tr><tr><td> $_ \mathrm { H i F i - M o l _ { F P } }$ </td><td>0.968±0.019</td><td> $2 . 4 5 4 { \scriptstyle \pm 0 . 0 4 9 }$ </td><td>0.830±0.022</td></tr><tr><td>HiFi-Mol</td><td> $0 . 8 6 9 { \scriptstyle \pm 0 . 0 1 8 }$ </td><td> $2 . 3 5 8 { \scriptstyle \pm 0 . 0 8 0 }$ </td><td>0.743±0.001</td></tr></table>

Table 3: RMSE (↓) on MoleculeNet regression benchmarks under the scafold split. Best in bold, second-best underlined. $\mathbf { H i F i - M o l } _ { \mathrm { G r a p h } }$ and $\mathbf { H i F i - M o l _ { F P } }$ denote graph-only and fingerprint-only variants, respectively.

Classification. Table 2 reports ROC-AUC results on eight MoleculeNet classification benchmarks. HiFi-Mol achieves the highest average ROC-AUC of 76.15, corresponding to a 2.77% relative improvement over MolCLR, the strongest baseline. MolCLR remains competitive by learning globally consistent molecular representations, while GraphMVP and MoleculeSDE benefit from 2D-3D geometric alignment. Fragment-level methods such as MGSSL and in Table 1. All results are reported as mean and standard deviation over three random seeds. Graph pretraining is conducted on a single NVIDIA RTX A6000 GPU (∼50 hours), and fingerprint encoder pretraining on a single NVIDIA RTX 4090 GPU (∼500 hours).

## 5.2 Downstream Performance in MoleculeNet

HiMol show the value of higher-order substructure modeling, but their performance varies substantially across benchmarks.

To analyze the contribution of each molecular view, we compare HiFi-Mol with its graph-only and fingerprint-only variants. Across the three HiFi-Mol variants, at least one achieves state-ofthe-art performance on seven ofthe eight classification benchmarks. Within the HiFi-Mol family, the graph-only variant performs better on BBBP, Tox21, ToxCast, and BACE, while the fingerprint-only variant performs better on HIV, MUV, ClinTox, and SIDER. This dataset-dependent behavior suggests that graph and fingerprint representations provide complementary predictive signals, and their integration yields the best average performance.

Regression. Table 3 reports RMSE results on three MoleculeNet regression benchmarks. HiFi-Mol remains competitive, achieving the second-best result on ESOL and FreeSolv, while $\mathrm { H i F i - M o l _ { G r a p h } }$ performs best on ESOL and Lipophilicity. Notably, graph-based hierarchical models, including HiMol and $\mathrm { H i F i - M o l _ { G r a p h } } .$ achieve strong performance on ESOL and FreeSolv, suggesting that hierarchical structural representations can be beneficial for certain physicochemical regression tasks. The comparatively weaker gains from multi-view integration further suggest that some regression targets may already be suficiently modeled by hierarchical graph representations.

## 5.3 Ablation Study

Graph Pretraining Components. We analyze the contribution of each graph pretraining component through ablation on eight MoleculeNet classification benchmarks. As shown in Table 4, replacing fragment-aware masking with random masking reduces the average ROC-AUC by 0.91, and removing any individual objective consistently degrades performance, indicating that global contrastive consistency, fragment-level semantics, geometric regularization, and local atom-bond recovery provide complementary supervision signals. The largest degradation occurs upon removing the global contrastive objective $( \Delta \mathrm { A v g . } = - 1 . 2 3 )$ , followed by fragment identity prediction (−0.81), distance prediction $( - 0 . 7 0 ) ,$ and atom-bond recovery (−0.29), whose modest impact suggests that local bonding patterns are partially recoverable via indirect supervision from the remaining objectives.

Table 4: Ablation of graph pretraining components on MoleculeNet classification benchmarks under scafold split (best in bold). The w/o FAM variant replaces the masking strategy with random masking; each other variant removes one pretraining objective. ΔAvg. reports the change in average ROC-AUC relative to the full model.
<table><tr><td>Method</td><td>BBBP</td><td> $\mathrm { T o x } 2 1$ </td><td> $\mathrm { T o x C a s t }$ </td><td>HIV</td><td>MUV</td><td>ClinTox</td><td>BACE</td><td>SIDER</td><td> $\operatorname { A v g } .$ </td><td> $\Delta { \mathrm { A v g } } .$ </td></tr><tr><td>Full Model</td><td>73.68±0.55</td><td> $7 4 . 5 6 { \scriptstyle \pm 1 . 0 7 }$ </td><td> $6 5 . 2 7 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $7 7 . 1 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 7 . 2 9 { \scriptstyle \pm 2 . 3 6 }$ </td><td> $9 6 . 8 3 { \scriptstyle \pm 0 . 9 5 }$ </td><td> $8 1 . 8 4 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $6 2 . 6 4 { \scriptstyle \pm 0 . 7 0 }$ </td><td>76.15</td><td></td></tr><tr><td>w/o FAM</td><td> $7 1 . 1 9 { \scriptstyle \pm 1 . 8 8 }$ </td><td> $7 5 . 0 5 { \scriptstyle \pm 0 . 5 5 }$ </td><td> $6 6 . 1 6 { \scriptstyle \pm 0 . 1 1 }$ </td><td> $7 5 . 1 8 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $7 4 . 0 6 { \scriptstyle \pm 2 . 2 4 }$ </td><td> $9 4 . 4 5 { \scriptstyle \pm 1 . 6 0 }$ </td><td> $8 2 . 4 5 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $6 3 . 3 7 { \scriptstyle \pm 1 . 7 0 }$ </td><td>75.24</td><td>-0.91</td></tr><tr><td>w/o Fragment ID</td><td> $7 1 . 7 3 { \scriptstyle \pm 0 . 7 8 }$ </td><td> $7 4 . 6 7 { \scriptstyle \pm 0 . 6 9 }$ </td><td> $6 5 . 6 1 { \scriptstyle \pm 0 . 3 7 }$ </td><td> $7 4 . 6 3 { \scriptstyle \pm 0 . 8 0 }$ </td><td> $7 4 . 3 2 { \scriptstyle \pm 1 . 4 7 }$ </td><td> $9 7 . 4 6 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $8 0 . 5 5 { \scriptstyle \pm 1 . 8 5 }$ </td><td> ${ \bf 6 3 . 7 6 { \scriptstyle \pm 0 . 8 2 } }$ </td><td>75.34</td><td>-0.81</td></tr><tr><td>w/o Distance</td><td> $7 1 . 6 3 { \scriptstyle \pm 0 . 6 7 }$ </td><td> $7 5 . 2 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> ${ \bf 6 6 . 3 7 { \scriptstyle \pm 0 . 2 6 } }$ </td><td> $7 4 . 6 5 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 4 . 2 5 { \scriptstyle \pm 1 . 0 1 }$ </td><td> $\mathbf { 9 7 . 6 2 { \scriptstyle \pm 0 . 1 1 } }$ </td><td> $8 0 . 4 6 { \scriptstyle \pm 0 . 9 4 }$ </td><td> $6 3 . 3 7 { \scriptstyle \pm 0 . 6 6 }$ </td><td>75.45</td><td>-0.70</td></tr><tr><td> $\mathbf { w } / \mathbf { o }$  Global Contrastive</td><td> $7 1 . 6 0 { \scriptstyle \pm 0 . 4 2 }$ </td><td> $7 4 . 8 7 { \scriptstyle \pm 0 . 8 1 }$ </td><td> $6 5 . 7 6 { \scriptstyle \pm 0 . 3 3 }$ </td><td> $7 6 . 3 0 { \scriptstyle \pm 1 . 3 1 }$ </td><td> $7 3 . 8 1 { \scriptstyle \pm 1 . 5 4 }$ </td><td> $9 4 . 1 8 { \scriptstyle \pm 2 . 6 5 }$ </td><td> $7 9 . 3 1 { \scriptstyle \pm 0 . 3 0 }$ </td><td> $6 3 . 5 3 { \scriptstyle \pm 1 . 6 2 }$ </td><td>74.92</td><td>-1.23</td></tr><tr><td> $\mathbf { w } / \mathbf { o }$  Atom &amp; Bond</td><td> $7 1 . 0 1 { \scriptstyle \pm 1 . 9 9 }$ </td><td> $7 5 . 0 0 { \scriptstyle \pm 0 . 5 3 }$ </td><td> $6 6 . 0 0 { \scriptstyle \pm 0 . 2 9 }$ </td><td> $7 7 . 2 4 { \scriptstyle \pm 1 . 7 9 }$ </td><td> $7 5 . 5 6 { \scriptstyle \pm 1 . 6 7 }$ </td><td> $9 6 . 6 4 { \scriptstyle \pm 1 . 7 6 }$ </td><td> $\mathbf { 8 2 . 9 4 { \scriptstyle \pm 1 . 0 7 } }$ </td><td> $6 2 . 4 8 { \scriptstyle \pm 1 . 1 0 }$ </td><td>75.86</td><td>-0.29</td></tr></table>

Table 5: Efect of GNN backbone architectures within HiFi-Mol. Results report ROC-AUC under scafold split.
<table><tr><td>Backbone</td><td>BBBP</td><td>BACE</td><td>HIV</td><td>MUV</td></tr><tr><td>GIN</td><td> ${ 7 3 . 6 8 \pm 0 . 5 5 }$ </td><td> $8 1 . 8 4 { \scriptstyle \pm 1 . 3 6 }$ </td><td> $7 7 . 1 0 { \scriptstyle \pm 0 . 4 5 }$ </td><td> $7 7 . 2 9 { \scriptstyle \pm 2 . 3 6 }$ </td></tr><tr><td>GCN</td><td> $7 3 . 4 5 { \scriptstyle \pm 0 . 9 1 }$ </td><td> $8 0 . 8 2 { \scriptstyle \pm 0 . 2 2 }$ </td><td> $7 6 . 3 6 { \scriptstyle \pm 1 . 2 8 }$ </td><td> $7 5 . 6 8 { \scriptstyle \pm 0 . 9 2 }$ </td></tr><tr><td>GAT</td><td> $7 2 . 8 1 { \scriptstyle \pm 0 . 8 4 }$ </td><td> $7 8 . 4 2 { \scriptstyle \pm 2 . 1 9 }$ </td><td> $7 5 . 4 7 { \scriptstyle \pm 0 . 7 3 }$ </td><td> $7 5 . 4 9 { \scriptstyle \pm 0 . 7 9 }$ </td></tr><tr><td>GraphSAGE</td><td> $7 2 . 0 3 { \scriptstyle \pm 0 . 9 2 }$ </td><td> $\mathbf { 8 2 . 7 3 { \scriptstyle \pm 0 . 2 9 } }$ </td><td> $7 5 . 2 4 { \scriptstyle \pm 0 . 4 3 }$ </td><td> $7 4 . 9 3 { \scriptstyle \pm 0 . 7 3 }$ </td></tr></table>

## 5.4 Retrieval Analysis of Pretrained Graph Representations

Backbone Sensitivity. To evaluate the generality of HiFi-Mol across diferent GNN architectures, we experiment with four backbone variants on a representative subset of benchmarks spanning small datasets (BBBP, BACE) and large datasets (HIV, MUV). As shown in Table 5, GIN achieves the strongest performance, which we attribute to its expressive power under the Weisfeiler-Leman graph isomorphism test [39], a property particularly beneficial for distinguishing chemically distinct substructures during hierarchical pretraining. Importantly, all backbone variants remain competitive across both dataset scales, demonstrating that the hierarchical graph pretraining and fingerprint integration strategies are robust across GNN architectures with diferent aggregation characteristics. These results indicate that the proposed pretraining and integration strategies remain efective across multiple GNN backbones.

The efect of fragment-aware masking is assessed through a molecular retrieval task on the pretrained graph encoder, prior to fingerprint integration. For each query molecule, we extract its embedding from the pretrained graph encoder and retrieve the top-5 nearest neighbors by cosine similarity. As shown in Figure 3, the FAMpretrained encoder retrieves molecules with structurally similar scafolds to the query, with the 1st, 2nd, and 4th nearest neighbors sharing the ortho-disubstituted benzene core and maintaining considerably higher ECFP4 Tanimoto similarity. In contrast, the model trained without FAM retrieves chemically less coherent neighbors, including sugar derivatives and terpenoids, with substantially lower ECFP4 similarity despite high cosine similarity in the embedding space.

<table><tr><td rowspan=3 colspan=1>QueryMolecule</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Top 1</td><td rowspan=1 colspan=1>Top 2</td><td rowspan=1 colspan=1>Top 3</td><td rowspan=1 colspan=1>Top 4</td><td rowspan=1 colspan=1>Top 5</td></tr><tr><td rowspan=1 colspan=1>FulI del</td><td rowspan=1 colspan=1>1.000=0</td><td rowspan=1 colspan=1>0.562</td><td rowspan=1 colspan=1>0.193</td><td rowspan=1 colspan=1>0.4609</td><td rowspan=1 colspan=1>0.250</td></tr><tr><td rowspan=1 colspan=1>WOFAM</td><td rowspan=1 colspan=1>1.00002</td><td rowspan=1 colspan=1>0.072</td><td rowspan=1 colspan=1>0.215</td><td rowspan=1 colspan=1>0.145</td><td rowspan=1 colspan=1>0.075OH4</td></tr></table>

Figure 3: Top-5 retrieval results for a query molecule from the Tox21 dataset, retrieved by cosine similarity in the embedding space. Scores denote ECFP4 Tanimoto similarity to the query.

![](images/ce12fc092347716f6c3a9dc4281da484088edc3ce82a593959442ca9cae05e62.jpg)  
Figure 4: Quantitative comparison on Tox21 and ToxCast using Mean Tanimoto@5 and Scafold Match@5.

Retrieval quality is further quantified using Mean Tanimoto@5 and Scafold Match@5, which measure the average ECFP4 similarity and the fraction of retrieved molecules sharing the same Bemis-Murcko scafold [3] as the query, respectively. As illustrated in Figure 4, pretraining with FAM consistently improves both metrics across Tox21 and ToxCast, with Mean Tanimoto@5 increasing by +0.023 on both datasets and Scafold Match@5 improving by +0.034 and +0.030, respectively. Across both qualitative and quantitative evaluations, these results suggest that pretraining with FAM encourages the graph encoder to organize the embedding space along chemically meaningful structural axes, capturing scafold level relationships beyond local atom-level patterns.

## 6 Conclusions and Future Work

This paper presents HiFi-Mol, a multi-view molecular pretraining framework for molecular property prediction. HiFi-Mol separately pretrains hierarchical graph and contextualized fingerprint encoders before integrating them during downstream adaptation. Fragment-aware masking with multi-resolution supervision enables the graph encoder to learn substructure-aware representations, while masked language modeling over joint SMILES–fingerprin token sequences allows the fingerprint encoder to capture contextual descriptor relationships. Experiments on MoleculeNet benchmarks demonstrate the efectiveness of each design choice, with the dual-view integration achieving the best average performance across classification tasks. Nevertheless, 3D geometry is used only as pretraining supervision rather than downstream input, and the dual-view integration does not always yield gains over single-view variants on regression tasks. Future work will investigate scalable cross-view interaction mechanisms between hierarchical topology and descriptor semantics, as well as extensions to equivariant 3D molecular modeling.

## Acknowledgments

This research was supported by National Research Foundation of Korea (NRF) grant funded by the Ministry of Science and ICT (RS-2025-16067916), Basic Science Research Program through the NRF funded by the Ministry of Education (RS-2025-25432868), IITP grant funded by the Ministry ofScience and ICT through the Digital Columbus Project (RS-2025-02304331), and the ANCHOR program through the Gangwon ANCHOR Center funded by the Ministry of Education and the Gangwon State, Republic of Korea (2026- ANCHOR-10-006).

## Appendix

## A Graph Initialization Details

Given the masked hierarchical graph ${ \tilde { g } } ,$ each node $\upsilon _ { i } ~ \in ~ \mathcal { V }$ is initialized into a representation $\bar { h } _ { i } ^ { ( 0 ) } \in \mathbb { R } ^ { d }$ according to its node type. A learnable type embedding $E _ { \mathrm { t y p e } } \in \mathbb { R } ^ { 3 \times d }$ distinguishes atom, fragment, and global nodes. Masked atom and fragment nodes, $v _ { i } \in { \mathcal { M } } _ { \mathrm { a t o m } } \cup { \mathcal { M } } _ { \mathrm { f r a g } } ,$ are assigned a learnable mask embedding $h _ { \mathrm { m a s k } } \in \mathbb { R } ^ { d }$ in place of their original features.

For atom nodes $v _ { u } \in \mathcal { V } _ { \mathrm { a t o m } } .$ , the initial representation is obtained by summing embedding lookups over ten chemical atom features $\{ \dot { x _ { u } } ^ { ( k ) } \} _ { k = 1 } ^ { 1 0 }$ (Table 6):

$$
h _ { u } ^ { ( 0 ) } = \sum _ { k = 1 } ^ { 1 0 } E _ { \mathrm { n o d e } } ^ { ( k ) } \Big [ x _ { u } ^ { ( k ) } \Big ] + E _ { \mathrm { t y p e } } ( 0 ) .\tag{21}
$$

The ten atom attributes are atomic number, chirality, formal charge, hybridization, number of hydrogens, implicit valence, degree, aromaticity, ring membership, and number of radical electrons.

Table 6: Atom features used for node initialization. Atomic number 119 is reserved as a mask token.
<table><tr><td>Feature</td><td>Size</td></tr><tr><td>Atomic number</td><td>119</td></tr><tr><td>Chirality</td><td>4</td></tr><tr><td>Formal charge</td><td>11</td></tr><tr><td>Hybridization</td><td>7</td></tr><tr><td>Number of hydrogens</td><td>9</td></tr><tr><td>Implicit valence</td><td>7</td></tr><tr><td>Degree</td><td>11</td></tr><tr><td>Aromaticity</td><td>2</td></tr><tr><td>Ring membership</td><td>2</td></tr><tr><td></td><td></td></tr><tr><td>Radical electrons</td><td>5</td></tr></table>

For fragment nodes $v _ { s } \in \mathcal { V } _ { \mathrm { f r a g } } ,$ the initial representation is determined by the fragment dictionary identity � (�):

$$
h _ { s } ^ { ( 0 ) } = E _ { \mathrm { f r a g } } [ \phi ( s ) ] + E _ { \mathrm { t y p e } } ( 1 ) .\tag{22}
$$

The global node $v _ { \mathrm { g l o b a l } }$ is initialized by averaging the initial atom feature embeddings:

$$
h _ { \mathrm { g l o b a l } } ^ { ( 0 ) } = \frac { 1 } { | \mathcal { V } _ { \mathrm { a t o m } } | } \sum _ { v _ { u } \in \mathcal { V } _ { \mathrm { a t o m } } } \sum _ { k = 1 } ^ { 1 0 } E _ { \mathrm { n o d e } } ^ { ( k ) } [ x _ { u } ^ { ( k ) } ] + E _ { \mathrm { t y p e } } ( 2 ) .\tag{23}
$$

Each edge $( v _ { i } , v _ { j } )$ is initialized as the sum of four edge feature embeddings:

$$
e _ { i j } = \sum _ { r = 1 } ^ { 4 } E _ { \mathrm { e d g e } } ^ { ( r ) } [ b _ { i j } ^ { ( r ) } ] ,\tag{24}
$$

where $\{ b _ { i j } ^ { ( r ) } \} _ { r = 1 } ^ { 4 }$ denotes edge attributes for bond type, bond direction, stereo configuration, and conjugation. The bond type attribute is extended beyond physical bond types to include self-loops, mask tokens, and directed hierarchical relation types, including atom-tofragment, fragment-to-atom, fragment-to-fragment, fragment-toglobal, and global-to-fragment edges.

## GenAI Usage Disclosure

LLM was used solely for grammar checking and language polish ing of the manuscript. All scientific content, experimental design, analysis, and conclusions are entirely the work of the authors.

## References

[1] Evan Antoniuk, Shehtab Zaman, Tal Ben-Nun, Peggy Li, James Difenderfer, Busra Sahin, Obadiah Smolenski, Everett Grethel, Tim Hsu, Anna Hiszpanski, et al. 2026. Boom: benchmarking out-of-distribution molecular property predictions of machine learning models. Advances in Neural Information Processing Systems 38 (2026).

[2] Kenneth Atz, Francesca Grisoni, and Gisbert Schneider. 2021. Geometric deep learning on molecular representations. Nature Machine Intelligence 3, 12 (2021), 1023–1032.

[3] Guy W Bemis and Mark A Murcko. 1996. The properties of known drugs. 1. Molecular frameworks. Journal ofmedicinal chemistry 39, 15 (1996), 2887–2893.

[4] Hanxuan Cai, Huimin Zhang, Duancheng Zhao, Jingxing Wu, and Ling Wang. 2022. FP-GNN: a versatile deep learning architecture for enhanced molecular property prediction. Briefings in bioinformatics 23, 6 (2022), bbac408.

[5] Ting Chen, Simon Kornblith, Mohammad Norouzi, and Geofrey Hinton. 2020. A simple framework for contrastive learning of visual representations. In International conference on machine learning. PMLR, 1597–1607.

[6] Jae-Woo Chu, Jong-Hoon Park, and Young-Rae Cho. 2026. MEMOL: Mixture of experts for multimodal learning through multi-head attention to predict drug toxicity. Computer Methods and Programs in Biomedicine 273 (2026), 109088.

[7] Laurianne David, Amol Thakkar, Rocío Mercado, and Ola Engkvist. 2020. Molec ular representations in AI-driven drug discovery: a review and practical guide. Journal ofcheminformatics 12, 1 (2020), 56.

[8] Jorg Degen, Christof Wegscheid-Gerlach, Andrea Zaliani, and Matthias Rarey. 2008. On the art of compiling and using’drug-like’chemical fragment spaces. ChemMedChem 3, 10 (2008), 1503.

[9] Jianyuan Deng, Zhibo Yang, Hehe Wang, Iwao Ojima, Dimitris Samaras, and Fusheng Wang. 2023. A systematic study of key elements underlying molecular property prediction. Nature Communications 14, 1 (2023), 6395.

[10] Jacob Devlin, Ming-Wei Chang, Kenton Lee, and Kristina Toutanova. 2019. Bert: Pre-training of deep bidirectional transformers for language understanding. In Proceedings ofthe 2019 conference ofthe North American chapter ofthe association for computational linguistics: human language technologies, volume 1 (long and short papers). 4171–4186.

[11] David K Duvenaud, Dougal Maclaurin, Jorge Iparraguirre, Rafael Bombarell, Timothy Hirzel, Alán Aspuru-Guzik, and Ryan P Adams. 2015. Convolutional networks on graphs for learning molecular fingerprints. Advances in neural information processing systems 28 (2015).

[12] Aled M Edwards and Dafydd R Owen. 2025. Protein–ligand data at scale to support machine learning. Nature Reviews Chemistry 9, 9 (2025), 634–645.

[13] Anna Gaulton, LouisaJ Bellis, A Patricia Bento,Jon Chambers, Mark Davies, Anne Hersey, Yvonne Light, Shaun McGlinchey, David Michalovich, Bissan Al-Lazikani, et al. 2012. ChEMBL: a large-scale bioactivity database for drug discovery. Nucleic acids research 40, D1 (2012), D1100–D1107.

[14] Justin Gilmer, Samuel S Schoenholz, Patrick F Riley, Oriol Vinyals, and George E Dahl. 2017. Neural message passing for quantum chemistry. In International conference on machine learning. PMLR, 1263–1272.

[15] Weihua Hu, Matthias Fey, Hongyu Ren, Maho Nakata, Yuxiao Dong, and Jure Leskovec. 2021. Ogb-lsc: A large-scale challenge for machine learning on graphs. arXiv preprint arXiv:2103.09430 (2021).

[16] Weihua Hu, Bowen Liu, Joseph Gomes, Marinka Zitnik, Percy Liang, Vijay Pande, and Jure Leskovec. 2019. Strategies for pre-training graph neural networks. arXiv preprint arXiv:1905.12265 (2019).

[17] Ziniu Hu, Yuxiao Dong, Kuansan Wang, Kai-Wei Chang, and Yizhou Sun. 2020. Gpt-gnn: Generative pre-training of graph neural networks. In Proceedings of the 26th ACM SIGKDD international conference on knowledge discovery & data mining. 1857–1867.

[18] Xiuyu Jiang, Liqin Tan, and Qingsong Zou. 2024. DGCL: dual-graph neural networks contrastive learning for molecular property prediction. Briefings in Bioinformatics 25, 6 (2024), bbae474.

[19] Yinghui Jiang, Shuting Jin, Xurui Jin, Xianglu Xiao, Wenfan Wu, Xiangrong Liu, Qiang Zhang, Xiangxiang Zeng, Guang Yang, and Zhangming Niu. 2023. Pharmacophoric-constrained heterogeneous graph transformer model for molec ular property prediction. Communications Chemistry 6, 1 (2023), 60.

[20] Alex Kendall, Yarin Gal, and Roberto Cipolla. 2018. Multi-task learning using uncertainty to weigh losses for scene geometry and semantics. In Proceedings of the IEEE conference on computer vision and pattern recognition. 7482–7491.

[21] Ruifeng Li, Wei Liu, Xiangxin Zhou, Mingqian Li, Qiang Zhang, Hongyang Chen, and Xuemin Lin. 2025. Contextual Representation Anchor Network for Mitigating Selection Bias in Few-Shot Drug Discovery. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management. 1634–1642.

[22] Bowen Liu, Haoyang Li, Shuning Wang, Shuo Nie, and Shanghang Zhang. 2025. Subgraph aggregation for out-of-distribution generalization on graphs. In Proceedings ofthe AAAI Conference on Artificial Intelligence, Vol. 39. 18763–18771.

[23] Shengchao Liu, Weitao Du, Zhi-Ming Ma, Hongyu Guo, and Jian Tang. 2023. A group symmetric stochastic diferential equation model for molecule multimodal pretraining. In International Conference on Machine Learning. PMLR, 21497– 21526.

[24] Shengchao Liu, Hanchen Wang, Weiyang Liu, Joan Lasenby, Hongyu Guo, and Jian Tang. 2021. Pre-training molecular graph representation with 3d geometry. arXiv preprint arXiv:2110.07728 (2021).

[25] Ziyang Liu, Chaokun Wang, Shuwen Zheng, Cheng Wu, Hao Feng, Li Xu, Yue Zheng, Liang Rong, and Peng Li. 2025. Molecular Motif Learning as a pretraining objective for molecular property prediction. Nature Communications (2025).

[26] Kha-Dinh Luong and Ambuj K Singh. 2023. Fragment-based pretraining and finetuning on molecular graphs. Advances in Neural Information Processing Systems 36 (2023), 17584–17601.

[27] Daniil Polykovskiy, Alexander Zhebrak, Benjamin Sanchez-Lengeling, Sergey Golovanov, Oktai Tatanov, Stanislav Belyaev, Rauf Kurbanov, Aleksey Artamonov, Vladimir Aladinskiy, Mark Veselov, et al. 2020. Molecular sets (MOSES): a bench marking platform for molecular generation models. Frontiers in pharmacology 11 (2020), 565644.

[28] Jianbo Qiao, Junru Jin, Ding Wang, Saisai Teng, Junyu Zhang, Xuetong Yang, Yuhang Liu, Yu Wang, Lizhen Cui, Quan Zou, et al. 2025. A self-conformationaware pre-training framework for molecular property prediction with substruc ture interpretability. Nature Communications 16, 1 (2025), 4382.

[29] David Rogers and Mathew Hahn. 2010. Extended-connectivity fingerprints. Journal ofchemical information and modeling 50, 5 (2010), 742–754.

[30] Yu Rong, Yatao Bian, Tingyang Xu, Weiyang Xie, Ying Wei, Wenbing Huang, and Junzhou Huang. 2020. Self-supervised graph transformer on large-scale molecular data. Advances in neural information processing systems 33 (2020), 12559–12571.

[31] Arman Seyed-Ahmadi, Bing Xu Hu, Armin Geraili, Anita Layton, Helen Hong Chen, Shana O Kelley, and Bo Wang. 2026. DELBERT: Fingerprint Language Modeling For Generalizable Hit Discovery in DNA-Encoded Libraries. In ICLR 2026 Workshop on Machine Learning for Genomics Explorations.

[32] Hannes Stärk, Dominique Beaini, Gabriele Corso, Prudencio Tossou, Christian Dallago, Stephan Günnemann, and Pietro Liò. 2022. 3d infomax improves gnns for molecular property prediction. In International conference on machine learning. PMLR, 20479–20502.

[33] Jonathan M Stokes, Kevin Yang, Kyle Swanson, Wengong Jin, Andres Cubillos-Ruiz, Nina M Donghia, Craig R MacNair, Shawn French, Lindsey A Carfrae, Zohar Bloom-Ackermann, et al. 2020. A deep learning approach to antibiotic discovery. Cell 180, 4 (2020), 688–702.

[34] Fan-Yun Sun, Jordan Hofmann, Vikas Verma, and Jian Tang. 2019. Infograph: Un supervised and semi-supervised graph-level representation learning via mutual information maximization. arXiv preprint arXiv:1908.01000 (2019).

[35] Yuyang Wang, Jianren Wang, Zhonglin Cao, and Amir Barati Farimani. 2022. Molecular contrastive learning of representations via graph neural networks. Nature Machine Intelligence 4, 3 (2022), 279–287.

[36] Benjamin Warner, Antoine Chafin, Benjamin Clavié, Orion Weller, Oskar Hall ström, Said Taghadouini, Alexis Gallagher, Raja Biswas, Faisal Ladhak, Tom Aarsen, et al. 2025. Smarter, better, faster, longer: A modern bidirectional encoder for fast, memory eficient, and long context finetuning and inference. In Proceedings ofthe 63rd Annual Meeting ofthe Association for Computational Linguistics (Volume 1: Long Papers). 2526–2547.

[37] Zhenqin Wu, Bharath Ramsundar, Evan N Feinberg, Joseph Gomes, Caleb Geniesse, Aneesh S Pappu, Karl Leswing, and Vijay Pande. 2018. MoleculeNet: a benchmark for molecular machine learning. Chemical science 9, 2 (2018), 513–530.

[38] Jun Xia, Chengshuai Zhao, Bozhen Hu, Zhangyang Gao, Cheng Tan, Yue Liu, Siyuan Li, and Stan Z Li. 2023. Mole-bert: Rethinking pre-training graph neural networks for molecules. In The Eleventh International Conference on Learning Representations.

[39] Keyulu Xu, Weihua Hu, Jure Leskovec, and Stefanie Jegelka. 2018. How powerful are graph neural networks? arXiv preprint arXiv:1810.00826 (2018).

[40] Kevin Yang, Kyle Swanson, Wengong Jin, Connor Coley, Philipp Eiden, Hua Gao, Angel Guzman-Perez, Timothy Hopper, Brian Kelley, Miriam Mathea, et al. 2019. Analyzing learned molecular representations for property prediction. Journal of chemical information and modeling 59, 8 (2019), 3370–3388.

[41] Yuning You, Tianlong Chen, Yang Shen, and Zhangyang Wang. 2021. Graph contrastive learning automated. In International conference on machine learning. PMLR, 12121–12132.

[42] Yuning You, Tianlong Chen, Yongduo Sui, Ting Chen, Zhangyang Wang, and Yang Shen. 2020. Graph contrastive learning with augmentations. Advances in neural information processing systems 33 (2020), 5812–5823.

[43] Xuan Zang, Xianbing Zhao, and Buzhou Tang. 2023. Hierarchical molecular graph self-supervised learning for property prediction. Communications Chemistry 6, 1 (2023), 34.

[44] Zaixi Zhang, Qi Liu, Hao Wang, Chengqiang Lu, and Chee-Kong Lee. 2021. Motifbased graph self-supervised learning for molecular property prediction. Advances in Neural Information Processing Systems 34 (2021), 15870–15882.