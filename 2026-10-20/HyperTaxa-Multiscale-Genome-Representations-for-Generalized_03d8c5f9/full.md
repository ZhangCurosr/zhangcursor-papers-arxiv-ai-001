# HyperTaxa: Multiscale Genome Representations for Generalized Microbial Taxonomic Inference

Akash Rao CSSE Department Auburn University Auburn, AL, USA azr0187@auburn.edu

Akhilesh Ramanchandran Oklahoma Animal Disease Diagnostic Laboratory Oklahoma State University Stillwater, OK, USA rakhile@okstate.edu

Sushim Gupta Oklahoma Animal Disease Diagnostic Laboratory Oklahoma State University Stillwater, OK, USA sushim.gupta@okstate.edu

Sathyanarayanan N. Aakur   
CSSE Department   
Auburn University   
Auburn, AL, USA   
san0028@auburn.edu

Vishalini Ramnth CSSE Department Auburn University Auburn, AL, USA vlr0013@auburn.edu

Abstract—Reliable microbial taxonomic inference is challenged by the rapid growth of newly sequenced organisms and incomplete reference coverage. While alignment- and databasedriven tools remain essential for reference-backed identification, they are not designed for learned inference over species absent from genome-level training. We introduce HYPERTAXA, a taxonomy-aligned representation learning framework for openset microbial taxonomic inference. HYPERTAXA encodes whole genomes as multiscale hyperspectral k-mer transition fields and aligns these representations to a taxonomy-derived semantic space using a two-stage Node2Vec-to-MDS procedure that combines local graph structure with global hierarchical distance preservation. The resulting space supports zero-shot inference over species excluded from genome-level training but represented in the taxonomy, generalized zero-shot inference over seen and unseen species, and cascaded genus-to-species prediction. Across benchmark and large-scale disjoint-species evaluations, HYPERTAXA improves taxonomic prediction over pseudo-image and transformer-graph baselines, with the strongest gains at higher taxonomic ranks. Ablations further show that multiscale transition encoding and taxonomy-aware alignment are key contributors to generalization. These results position HYPERTAXA as a complementary taxonomy-aware representation and candidategeneration framework for microbial identification under incomplete reference coverage and open-set uncertainty.

Index Terms—Taxonomic inference, genome representation learning, hyperspectral encoding

## I. INTRODUCTION

Microbial taxonomic inference is fundamental to clinical diagnostics, public health, agriculture, environmental monitoring, and microbial systematics. As sequencing produces increasing numbers of newly assembled isolates and metagenome-assembled genomes, an important computational challenge is to reason about organisms that are rare, undercharacterized, or absent from the training distribution. We study this problem as open-set representation learning: given genome-derived sequence evidence, infer taxonomic identity while generalizing to species with no genome examples during training but whose taxonomic placement is known. Zero-shot (ZSL) and generalized zero-shot learning (GZSL) naturally frame this problem by representing unseen species through structured class-level information. Effective inference requires genome representations that capture multiscale sequence dependencies and an embedding geometry that reflects taxonomic organization rather than training-label frequency.

Existing approaches address related but distinct settings. Reference-based methods such as AAI/ANI [1], MyTaxa [2], EzBioCloud [3], TYGS [4], and GTDB-Tk [5] remain essential when close references are available, but are not learned zero-shot classifiers over species excluded from training. Supervised genomic models based on codon usage [6]– [9], recurrent networks [10]–[12], transformers [13], and metagenomic classification [14], [15] likewise remain constrained by observed labels. Species-delimitation [16]–[19] and phenotype-prediction methods [20]–[22] show that learned representations capture meaningful genomic structure. Mathematical genome mappings [23]–[25], graph-based representations and visualization [26]–[29], hyperdimensional encodings [30], [31], and image-based representations [32], [33] further motivate expressive genome representations. However, these methods do not explicitly align genome features with a taxonomic semantic space for ZSL/GZSL inference.

We introduce HYPERTAXA, a taxonomy-aligned framework for open-set microbial taxonomic inference. As shown in Figure 1, genomes are encoded as multiscale hyperspectral k-mer transition fields, with channels capturing sequence dependencies at different genomic strides. A convolutional encoder maps these tensors into a latent space aligned with taxonomy-derived embeddings. Node2Vec [34] captures local graph neighborhoods, while multidimensional scaling (MDS) over taxonomic distances imposes global hierarchical geometry. The resulting space supports ZSL over species unseen during genome-level training, GZSL over seen and unseen species, and cascaded genus-to-species inference. HYPER-TAXA is complementary to reference-based tools such as BLAST and GTDB-Tk, providing a learned taxonomy-aware representation and candidate-generation layer rather than replacing deterministic sequence comparison.

![](images/0ea65000fdd7f5fd3d90b0f74e07ae00e0c8e104b3ad6a89c77267f8161b7f7b.jpg)  
Fig. 1. Overview of HYPERTAXA. Multiscale k-mer transition fields encode genome structure, while the microbial taxonomy provides hierarchical and distance constraints. A genome encoder maps these representations into a latent space aligned with taxonomy-derived embeddings, enabling zero-shot and generalized zero-shot inference.

Our contributions are threefold: (1) we formulate microbial taxonomic inference as ZSL/GZSL over species excluded from genome-level training but represented in the taxonomy; (2) we introduce multiscale hyperspectral k-mer transition fields with Node2Vec-to-MDS taxonomy alignment; and (3) we evaluate the framework on the 97-species TEPI benchmark [32], a large-scale split with 1,110 training and 219 unseen species, and systematic representation and encoder ablations.

## II. METHODS

Problem Formulation. We formulate microbial taxonomic inference as open-set representation learning over a taxonomic hierarchy. Let $\mathcal { S } _ { \mathrm { s e e n } }$ denote species with genome examples during training and $S _ { \mathrm { u n s e e n } }$ species absent from genome-level training but present in the taxonomy graph. Given a query genome ${ \mathrm { G } } _ { q } ,$ the goal is to predict its identity at species, genus, family, and order ranks. ZSL evaluates only $S _ { \mathrm { u n s e e n } } .$ , while GZSL evaluates $S _ { \mathrm { s e e n } } \cup S _ { \mathrm { u n s e e n } } .$ . Each species v is represented by a taxonomy-derived embedding $\mathbf { e } _ { v } \in \mathbb { R } ^ { d }$ , including unseen species for which no training genomes are available. A genome encoder $f _ { \theta }$ maps the hyperspectral representation X to $\mathbf { h } ( \mathrm { G } ) = f _ { \theta } ( \mathrm { X } )$ , and prediction is performed by nearestneighbor matching to candidate taxon embeddings.

## A. Hyperspectral Genome Representation

To capture multiscale dependencies in whole bacterial genomes, we convert each genome into a hyperspectral kmer transition tensor. This representation summarizes how short nucleotide motifs transition into one another when the genome is sampled at different spatial scales, thereby encoding both local sequence composition and longer-range genomic organization. Unlike conventional pseudo-image representations or frequency-based k-mer histograms, the proposed representation preserves directional motif-transition structure across multiple genomic strides. Formally, let the genome be $\mathrm { ~ G ~ } = { \left( g _ { 1 } , \dots , g _ { L } \right) }$ with $g _ { i } \in \{ A , C , G , T \}$ , and fix a k-mer size k. Let $\mathcal { M } = \{ A , C , G , T \} ^ { k }$ denote the lexicographically ordered set of all k-mers, and let $\pi : { \mathcal { M } }  \{ { 1 , \dots , 4 ^ { k } } \}$ be an indexing map. For a stride s, we traverse the genome using hop size s, producing sampled positions $p _ { n } = 1 + ( n - 1 ) s .$ where $p _ { n } + k - 1 \leq L$ . The corresponding indexed k-mers are $u _ { n } ^ { ( s ) } = \pi \left( { \mathrm G } [ p _ { n } : p _ { n } + k - 1 ] \right)$ . Consecutive sampled kmers induce directed transitions $( u _ { n } ^ { ( s ) } \ \to \ u _ { n + 1 } ^ { ( s ) } )$ . For each stride s, these transitions are accumulated into a matrix $\mathrm { M } ^ { ( s ) } \in \mathbb { R } ^ { 4 ^ { k } \times 4 ^ { k } }$

$$
\mathrm { M } _ { i , j } ^ { \left( s \right) } = \sum _ { n = 1 } ^ { N _ { s } - 1 } \alpha _ { n } ^ { \left( s \right) } \Im \left( u _ { n } ^ { \left( s \right) } = i \right) \Im \left( u _ { n + 1 } ^ { \left( s \right) } = j \right) .\tag{1}
$$

![](images/70b0d70898151e9dc055853d6cde1718ae49756d8d49f8ce921d679e2bb7a649.jpg)

![](images/65331e77c137a866385e22269b8fecde9fefb3a8d3c1522eb97d2838d1c108b5.jpg)

![](images/78501d2d7f3e7c218f25fe333e798499fd9f7fe1c6defdec7f69326566711cb0.jpg)

![](images/db9b81a3f221fbd2a2db7fb1a848bc0586845918c745757b29f9f7ff23d7a9da.jpg)  
Fig. 2. Genome representation and taxonomy embedding geometry. a–b. TEPI pseudo-image and HYPERTAXA multiscale k-mer transition representation for B. contaminans. c–d. Node2Vec and MDS taxonomy-embedding similarities; MDS yields clearer taxonomic separation. \*contrast-enhanced.

where $\alpha _ { n } ^ { ( s ) }$ is a stride-specific transition weight. In practice, each channel is normalized as $\begin{array} { r } { \widetilde { \mathrm { M } } ^ { ( s ) } ~ = ~ \mathrm { M } ^ { \mathit { \bar { ( } s ) } } \Big / \sum _ { i , j } \mathrm { M } _ { i , j } ^ { ( s ) } } \end{array}$ Stacking transition matrices from C strides yields the hyperspectral genome tensor $\mathrm { X } = \big \lceil \widetilde { \mathrm { M } } ^ { ( s _ { 1 } ) } , \widetilde { \mathrm { M } } ^ { ( s _ { 2 } ) } , \dots , \widetilde { \mathrm { M } } ^ { ( s _ { C } ) } \big \rceil \overset { \cdot } { \in }$ $\mathbb { R } ^ { 4 ^ { k } \times 4 ^ { k } \times C }$ . For complete genomes, transition counts are accumulated by traversing the concatenated RefSeq sequence. Small strides emphasize local dependencies, including codon usage patterns and short-range compositional motifs, while larger strides capture longer-range structure such as compositional domains and genome-scale organization. Thus, each pixel $( i , j , c )$ measures the directional association between kmer i and k-mer j at stride $s _ { c } .$ Figure 2a compares this representation with the TEPI pseudo-image encoding, illustrating that hyperspectral transition fields produce structured multiscale motif patterns rather than sparse positional grids.

## B. Taxonomy-Aware Semantic Embedding Construction

Zero-shot inference requires a class-level embedding for each candidate species, including species with no training genomes. We therefore construct taxonomy-derived embeddings whose geometry reflects both local neighborhood structure and global hierarchical distance in the microbial taxonomy graph. Let V denote the set of species nodes and let $\mathbf { z } _ { v } \in \mathbb { R } ^ { d }$ denote the desired embedding of species $v \in \mathcal V$ . The embedding space should satisfy two complementary properties: nearby taxa in the taxonomy graph should have similar embeddings, and Euclidean distances between embeddings should preserve taxonomic distances. We express this objective as

$$
\mathcal { L } _ { \mathrm { t a x } } \big ( \mathbf { z } _ { v } \big ) = \lambda _ { \mathrm { l o c } } \mathcal { L } _ { \mathrm { l o c } } \big ( \mathbf { z } _ { v } \big ) + \lambda _ { \mathrm { g l o b } } \mathcal { L } _ { \mathrm { g l o b } } \big ( \mathbf { z } _ { v } \big ) ,\tag{2}
$$

where $\mathcal { L } _ { \mathrm { l o c } }$ encourages local graph consistency and $\mathcal { L } _ { \mathrm { g l o b } }$ encourages agreement with the shortest-path taxonomic distance $d _ { \mathrm { t a x } } ( u , v )$ . Rather than directly optimizing Eq. (2), we approximate its two terms using two complementary embeddings. First, we compute a Node2Vec embedding $\mathbf { e } _ { v } ^ { \mathrm { n 2 v } } \in \mathbb { R } ^ { d }$ for each species node using random walks on the taxonomy graph. This captures local co-occurrence and neighborhood structure. Second, we compute all-pairs shortest-path distances $d _ { \mathrm { t a x } } ( u , v )$ between species and apply classical multidimensional scaling (MDS) to obtain $\mathbf { e } _ { v } ^ { \mathrm { m d s } } \in \mathbb { R } ^ { d }$ such that $\left| \mathbf { e } _ { u } ^ { \mathrm { { m d s } } } - \mathbf { e } _ { v } ^ { \mathrm { { m d s } } } \right| _ { 2 }$ ≈ $d _ { \mathrm { t a x } } ( u , v )$ . Node2Vec provides local smoothness, while MDS imposes global hierarchical separation. Figure 2b visualizes the distinction between these two embedding spaces using dot-product similarity matrices. Node2Vec captures local taxonomic neighborhoods but can blur higher-level boundaries, whereas MDS produces sharper block structure aligned with genus- and family-level separation.

1) Genome Encoder and Alignment Objective: Given the hyperspectral tensor $\mathbf { X } ~ \in ~ \mathbb { R } ^ { 4 ^ { k } \times 4 ^ { k } \times C }$ , we learn a genome encoder $f _ { \theta }$ that maps X to a d-dimensional latent representation: $ { \mathbf { h } } ( \mathrm { G } ) { = } f _ { \theta } ( \mathrm { X } )$ . Because X is a multichannel image-like representation of motif-transition structure, we use a convolutional neural network (CNN) encoder. The CNN extracts local transition patterns and composes them into higher-level features through stacked convolution and pooling operations. This provides a natural inductive bias for hyperspectral genome tensors while avoiding the sequence-length scaling challenges associated with recurrent or transformer-based whole-genome encoders. For a genome G belonging to species $v ,$ the encoder is trained to align h(G) with the taxonomy-derived embedding of that species: $\mathcal { L } _ { \mathrm { a l i g n } } ~ = ~ \left| \mathbf { h } ( \mathrm { G } ) - \mathbf { e } _ { v } \right| _ { 2 } ^ { 2 } .$ Training proceeds in two stages. In Stage 1, ${ \bf e } _ { v } = { \bf e } _ { v } ^ { \mathrm { n 2 v } }$ , which initializes the encoder using locally smooth taxonomy neighborhoods. In Stage 2, $\mathbf { e } _ { v } ~ = ~ \mathbf { e } _ { v } ^ { \mathrm { m d s } }$ , which fine-tunes the encoder toward globally consistent hierarchical separation. This Node2Vecto-MDS alignment approximates the composite geometry in Eq. (2) while remaining trainable using genome-level supervision. We use a regression objective rather than a contrastive objective because microbial genome datasets are often sparse and imbalanced across species. Contrastive learning typically requires batches with multiple representatives per class and carefully sampled negatives, which may not be available for rare or newly sequenced organisms.

2) Inference: At inference time, a query genome $\mathrm { G } _ { q }$ is converted into its hyperspectral tensor $\mathrm { X } _ { q }$ and embedded as $\mathbf { h } ( \mathrm { G } _ { q } ) ~ = ~ f _ { \theta } ( \mathrm { X } _ { q } )$ . Taxonomic prediction is performed by nearest-neighbor matching between $\mathbf { h } ( \mathrm { G } _ { q } )$ and candidate taxon embeddings in the taxonomy-aligned latent space to support zero-shot and generalized zero-shot taxonomic prediction. Zero-shot inference. For unseen species, training genomes are unavailable, but the MDS-derived taxonomy embedding $\mathbf { e } _ { v } ^ { \mathrm { m d s } }$ can be computed from the taxonomy graph. The predicted species is $\begin{array} { r } { \hat { \boldsymbol { v } } = \arg \operatorname* { m i n } _ { \boldsymbol { v } \in S _ { \mathrm { u n s e e n } } } \left| \mathbf { h } ( \mathrm { G } _ { q } ) - \mathbf { e } _ { \boldsymbol { v } } ^ { \mathrm { m d s } } \right| _ { 2 } } \end{array}$ . This enables species-level prediction without observing any genome from the target species during training.

![](images/f1ffa03fed8fc34305e287645ef65a3dfe33e184e297232681dd31d20907768c.jpg)  
Fig. 3. Zero-shot Top-1 accuracy on the TEPI benchmark.

Generalized zero-shot inference. In the GZSL setting, the candidate set contains both seen and unseen species:

$$
\hat { v } = \arg \operatorname* { m i n } _ { v \in S _ { \mathrm { s e e n } } \cup S _ { \mathrm { u n s e e n } } } \left| { \bf h } ( \mathrm { G } _ { q } ) - { \bf e } _ { v } ^ { \mathrm { m d s } } \right| _ { 2 } .\tag{3}
$$

GZSL jointly evaluates seen and unseen species, requiring predictions over the combined candidate set.

Cascaded genus-to-species inference. We also evaluate a hierarchical inference strategy. First, the query is matched to genus-level embeddings, obtained by averaging the embeddings of species within each genus. Species prediction is then restricted to members of the top-ranked genus or genera, and the final label is assigned by nearest-neighbor matching within this reduced candidate set. This cascade reduces cross-genus confusion and captures the coarse-to-fine taxonomic structure.

## C. Experimental Setup

Datasets. We evaluated HYPERTAXA in three settings. First, we used the 97-species TEPI benchmark [32] under its original zero-shot protocol. Second, we constructed a large-scale dataset from NCBI RefSeq release 216 “Complete Genome” assemblies [35], treating each assembly as one sample and concatenating chromosomes and plasmids in RefSeq order. For ZSL/GZSL, we trained on 1,110 species and evaluated on 219 held-out species whose families were represented during training. Held-out species had no genome examples in training; their class embeddings were derived only from the taxonomy graph. Sparse training classes were balanced with noisy synthetic hyperspectral images, with no synthetic data used for evaluation. Third, for within-species generalization, we selected species with at least two complete genomes, using one for training and the remainder for testing, yielding 736 training and 414 test images across 184 species.

Implementation and Evaluation. We report Top-1 and Top-5 accuracy at species, genus, family, and order ranks. ZSL includes only held-out species, while GZSL evaluates seen and unseen species jointly. Cascaded inference predicts genus first and then restricts species prediction to that genus. All variants used the same splits and alignment protocol. We compared the proposed CNN with ResNet-50 [36] and ViT-B/16 [37], and evaluated three- versus five-channel hyperspectral inputs and single-stage Node2Vec- or MDS-aligned variants. All encoders produced $d { = } 1 2 8$ embeddings and were trained with SGD (learning rate $1 0 ^ { - 3 }$ , batch size 32). Unless otherwise specified, we used $k { = } 6 ,$ yielding $4 0 9 6 \times 4 0 9 6 \times C$ tensors, with $C { = } 5$ strides {5, 10, 15, 20, 25} and $\alpha _ { n } ^ { ( s ) } { = } 1 ;$ ; three-channel inputs were used only for ablation.

a. Zero-Shot (ZSL) Performance  
![](images/851577e825739466c14a6704b2b78ba7c23f0425d4f6c776686573b5d1257845.jpg)

b. Generalized Zero-Shot (GZSL) Performance  
![](images/7f293fb3878b604edb1f4713113925a32435d3797f53c00e5f802ef8eb4b541d.jpg)  
Fig. 4. ZSL and GZSL performance. a. HYPERTAXA outperforms MGNet/TEPI across all ranks in ZSL. b. In GZSL, MDS alignment yields the strongest performance, indicating improved hierarchical generalization.

## III. RESULTS

## A. Multiscale Representation Improves Zero-Shot Prediction

We first evaluated whether the proposed hyperspectral genome representation improves zero-shot taxonomic inference relative to prior genome pseudo-image and transformergraph baselines. On TEPI’s 97-species benchmark [32], HY-PERTAXA was trained and evaluated under the same held-out species protocol used by TEPI. As shown in Figure 2, the TEPI pseudo-image representation produces sparse position-derived patterns, whereas the HYPERTAXA image exhibits structured multiscale k-mer transition motifs. These differences translate into improved zero-shot accuracy across taxonomic ranks. HY-PERTAXA matches TEPI-WG at the species level with 11.7% Top-1 accuracy, while substantially improving genus, family, and order accuracy to 53.2%, 66.2%, and 75.3%, respectively. These gains indicate that multiscale transition fields provide a stronger substrate for learning taxonomically meaningful genome representations than pseudo-image or transformergraph alternatives. We also examined the taxonomy-derived embedding spaces used to guide alignment. Node2Vec embeddings preserve local graph neighborhoods, but their similarity structure shows diffuse higher-level separation. In contrast, MDS embeddings derived from taxonomy distances produce sharper block structure aligned with genus- and family-level hierarchy. This supports the two-stage alignment strategy used by HYPERTAXA: Node2Vec provides locally smooth initialization, while MDS imposes global taxonomic geometry.

a. Cascaded Prediction  
![](images/0884d542a66870cc01d41a3b5d3129f21e4f4100c8ce8acd685db5af508abbeb.jpg)

b. Hyperspectral Channel Ablation  
![](images/96a4da8680a72b4a82f14deb6c9bb5eaccb0affb6831cb47358b7d191c2dbea6.jpg)

c. Genome Encoder Ablation  
![](images/6678962215299953d48cb76e2388e9d78c99671c001cc2075ddbfdc00ad1ef09.jpg)  
Fig. 5. Hierarchical inference and ablations. a. Cascaded genus→species prediction improves accuracy. b. Increasing the number and range of hyperspectral k-mer transition channels improves zero-shot generalization. c. CNN+MDS provides the strongest zero-shot performance among encoder and alignment variants.

## B. Zero-Shot and Generalized Zero-Shot Evaluation

We next evaluated HYPERTAXA on a larger disjoint-species split with 1,110 training and 219 unseen test species. Because held-out species share higher-level taxa with training species, this setting evaluates generalization to unseen species and their hierarchical localization within known taxonomic lineages. Figure 4a shows that HYPERTAXA outperforms the MGNet/TEPI pseudo-image baseline across all ranks in the strict ZSL setting. At the species level, accuracy improves from approximately 2–3% for MGNet to 7–8% for HYPER-TAXA. The gains are larger at higher ranks, with the MDSaligned model reaching 49.0% family-level and 37.0% orderlevel Top-1 accuracy. Top-5 results follow the same trend, with HYPERTAXA placing the correct species in the shortlist for more than half of unseen-species queries. In the GZSL setting, where seen and unseen species are evaluated jointly, HYPERTAXA again improves over MGNet/TEPI (Figure 4b). The MDS-aligned model achieves approximately 34–35% species-level accuracy and 64–70% accuracy at genus, family, and order ranks.

## C. Hierarchical Inference and Ablation Studies

We evaluated whether explicit hierarchical inference improves prediction by applying a cascaded genus→species strategy. As shown in Figure 5a, restricting species-level prediction to candidates within the predicted genus improves performance over direct prediction. For the MDS-aligned model, species-level accuracy increases by approximately 5.1%, with consistent gains across higher ranks. This suggests that the learned embedding space preserves useful coarse-tofine taxonomic structure and that genus-level filtering reduces cross-genus confusion. Ablation studies further support the main design choices. Increasing the number of hyperspectral channels improves zero-shot performance across taxonomic ranks (Figure 5b). The five-channel configuration using strides 5, 10, 15, 20, 25 achieves the best overall performance, indicating that broader multiscale transition structure is important for open-set generalization. Encoder ablations show that the lightweight CNN paired with MDS alignment performs best in the ZSL setting (Figure 5c). Deeper ResNet and ViT backbones do not improve zero-shot performance, suggesting that strong local inductive bias and taxonomy-aligned supervision are more important than model capacity for this representation.

1) Qualitative Discussion: The results support taxonomyaware representation learning for open-set microbial taxonomic inference. HYPERTAXA combines multiscale k-mer transition representations with Node2Vec-to-MDS alignment, enabling inference over species absent from genome-level training but represented in the taxonomy graph. Gains are strongest at genus, family, and order ranks, while cascaded genus-to-species inference reduces cross-genus confusion and ablations indicate that representation and embedding geometry matter more than model capacity. HYPERTAXA is therefore complementary to tools such as BLAST, Kraken, GTDB-Tk, ANI/AAI, and TYGS rather than a replacement for deterministic sequence comparison. The current ZSL setting assumes unseen species are present in the taxonomy graph and share higher-level neighborhoods with training taxa, so truly unknown lineages may fall outside the learned space; experiments also focus on complete genomes rather than fragmented assemblies, MAGs, or raw-read mixtures.

## IV. CONCLUSION AND FUTURE WORK

We introduced HYPERTAXA, a taxonomy-aligned representation learning framework for open-set microbial taxonomic inference. By encoding genomes as multiscale hyperspectral k-mer transition fields and aligning them to a structured taxonomy-derived embedding space, HYPERTAXA supports zero-shot, generalized zero-shot, and cascaded hierarchical inference. Results across benchmark, large-scale disjoint-species, and ablation studies show that taxonomyaware genome representations improve inference for species excluded from genome-level training, particularly at higher taxonomic ranks. Future work will extend HYPERTAXA to stricter reference-masking and phylogenetic holdout settings, fragmented genomes, and raw-read metagenomic data. We also plan to evaluate diagnostic-style candidate retrieval and hybrid workflows with established profilers and genome-based classification tools, including their effects on search-space reduction, runtime, and failure modes.

## REFERENCES

[1] K. T. Konstantinidis and J. M. Tiedje, “Towards a genome-based taxonomy for prokaryotes,” Journal of Bacteriology, vol. 187, no. 18, pp. 6258–6264, 2005.

[2] C. Luo, D. H. Rodriguez-R, and K. V. Konstantinidis, “Mytaxa: an advanced taxonomic classifier for genomic and metagenomic sequences,” Nucleic Acids Research, vol. 42, no. 8, p. e73, 2014.

[3] S.-I. Na, Y. Kim, S.-H. Yoon, S.-G. Ha, I. Baek, and J. Chun, “Ezbiocloud: a taxonomically united database of 16s rrna gene sequences and whole-genome assemblies,” International Journal of Systematic and Evolutionary Microbiology, vol. 67, no. 5, pp. 1613–1617, 2017.

[4] J. P. Meier-Kolthoff and M. Goker, “Tygs is an automated high-¨ throughput platform for state-of-the-art genome-based taxonomy,” Nature Communications, vol. 10, no. 1, p. 2182, 2019.

[5] P.-A. Chaumeil, A. J. Mussig, P. Hugenholtz, and D. H. Parks, “Gtdbtk: a toolkit to classify genomes with the genome taxonomy database,” Bioinformatics, vol. 36, no. 6, pp. 1925–1927, 2020.

[6] B. B. Khomtchouk, “Codon usage bias levels predict taxonomic identity and genetic composition,” bioRxiv, 2020.

[7] J. Kim and coauthors, “Machine learning classifiers predict key genomic and evolutionary traits across the kingdoms of life,” Scientific Reports, vol. 13, p. 2078, 2023.

[8] S. a. Sen, “Deep learning-based codon optimization with large-scale contextual language models,” bioRxiv, 2023.

[9] A. Fallahpour, V. Gureghian, G. J. Filion, A. B. Lindner, and A. Pandi, “Codontransformer: a multispecies codon optimizer using context-aware neural networks,” Nature Communications, vol. 16, no. 1, p. 3205, 2025.

[10] Q. Liang, P. Bible, Y. Liu, B. Zou, and L. Wei, “Deepmicrobes: taxonomic classification for metagenomics with deep learning,” NAR Genomics and Bioinformatics, vol. 2, no. 1, p. lqaa009, 2020.

[11] Z. Shen, W. Bao, and D.-S. Huang, “Recurrent neural network for predicting transcription factor binding sites,” Scientific Reports, vol. 8, no. 1, p. 15270, 2018.

[12] J. Lanchantin, R. Singh, B. Wang, and Y. Qi, “Deep motif dashboard: Visualizing and understanding genomic sequences using deep neural networks,” in Pacific Symposium on Biocomputing (PSB), pp. 254–265, World Scientific, 2017.

[13] F. Mock, H. Ibrahim, P. Eugster, E. Seiler, A. Sczyrba, and D. B. Boeckaerts, “Taxonomic classification of dna sequences beyond sequence similarity using deep neural networks,” Proceedings of the National Academy of Sciences of the United States of America (PNAS), vol. 119, no. 36, p. e2122636119, 2022.

[14] A. Mathieu, M. Leclercq, M. Sanabria, J.-P. Flandrois, P. Peyret, M.- A. Aury, J. Weissenbach, P. Wincker, and C. Elie, “Machine learning and deep learning applications in metagenomic taxonomy and functional annotation,” Frontiers in Microbiology, vol. 13, p. 811495, 2022.

[15] H. Park, S. J. Lim, C. Nitikitpaiboon, L. R. Thompson, A. T. Brito, C. M. Duncombe, M. H. Parks, E. Liu, X. Zhang, X. Zhang, P. Wang, W. Wang, S. P. Parker, R. R. Ponce, J. C. Pereira, S. Janga, and E. E. Allen, “Investigation of machine learning algorithms for taxonomic classification of marine metagenomes,” Microbiology Spectrum, vol. 11, no. 5, pp. e05237–22, 2023.

[16] K. Karbstein, L. Kosters, L. Hoda ¨ c, M. Hofmann, E. H ˇ orandl,¨ S. Tomasello, N. Wagner, B. Emerson, D. Albach, S. Scheu, S. Bradler, J. de Vries, I. Irisarri, H. Li, P. S. Soltis, P. Mader, and J. W¨ aldchen,¨ “Species delimitation 4.0: integrative taxonomy meets artificial intelligence,” Trends in Ecology and Evolution, vol. 39, no. 8, pp. 771–784, 2024.

[17] M. A. Perez and coauthors, “Integrative taxonomy using traits´ and genomic data for species delimitation with deepid,” bioRxiv, p. 2025.03.22.644762, 2025.

[18] L. M. Kosters, K. Karbstein, M. Hofmann, L. Hoda¨ c, P. Mˇ ader, and¨ J. Waldchen, “Data fusion for integrative species identification using¨ deep learning,” Systematic Biology, p. syaf026, 2025.

[19] S. Singhal and coauthors, “A genomic perspective on species delimitation,” Annual Review of Ecology, Evolution, and Systematics, vol. 56, pp. 467–489, 2025.

[20] N. Koblitz, A. Reimer, E. Tetteh, F. Delogu, D. Ionescu, M. Neumann-Schaal, C. Ruckert, and J. Overmann, “Predicting bacterial phenotypic¨ traits through improved machine learning using high-quality, curated datasets,” Communications Biology, vol. 8, no. 1, p. 680, 2025.

[21] A. van Hilten, S. A. Kushner, J. E. Verhoeven, S. Kohler,¨ et al., “Gennet framework: interpretable deep learning for predicting phenotypes from genetic data,” Communications Biology, vol. 4, no. 1, p. 1214, 2021.

[22] S. Sehrawat, U. Basu, G. Li, C. J. Pozniak, P. Hucl, D. Spaner, S. M. R. Azam, and H. Rahman, “Predicting phenotypes from novel genomic markers using deep learning,” Bioinformatics Advances, vol. 3, no. 1, p. vbad028, 2023.

[23] F. Swidan, E. P. C. Rocha, M. Shmoish, and R. Y. Pinter, “An integrative method for accurate comparative genome mapping,” PLoS Computational Biology, vol. 2, no. 8, p. e75, 2006.

[24] A. M. Kaye and W. W. Wasserman, “The genome atlas: navigating a new era of reference genomes,” Trends in Genetics, vol. 37, no. 9, pp. 807– 818, 2021.

[25] P. Belokopytova and V. Fishman, “Predicting genome architecture: Challenges and solutions,” Frontiers in Genetics, vol. 11, p. 617202, 2021.

[26] G. Gonnella, N. Niehus, and S. Kurtz, “Gfaviz: flexible and interactive visualization of gfa sequence graphs,” Bioinformatics, vol. 35, no. 16, pp. 2853–2855, 2019.

[27] A. Parveen, S. Khurana, and A. Kumar, “Overview of genomic tools for circular visualization in the next-generation genomic sequencing era,” Current Genomics, vol. 20, no. 2, pp. 90–99, 2019.

[28] J. R. Grant, E. Enns, E. Marinier, A. Mandal, E. K. Herman, C.-Y. Chen, M. Graham, G. V. Domselaar, and P. Stothard, “Proksee: in-depth characterization and visualization of bacterial genomes,” Nucleic Acids Research, vol. 51, no. W1, pp. W484–W492, 2023.

[29] S. Narayanan, A. Ramachandran, S. N. Aakur, and A. Bagavathi, “Gradl: a framework for animal genome sequence classification with graph representations and deep learning,” in 2020 19th IEEE International Conference on Machine Learning and Applications (ICMLA), pp. 1297– 1303, IEEE, 2020.

[30] M. Stock, W. V. Criekinge, D. Boeckaerts, S. Taelman, M. V. Haeverbeke, P. Dewulf, and B. D. Baets, “Hyperdimensional computing: a fast, robust, and interpretable paradigm for biological data,” PLoS Computational Biology, vol. 20, no. 9, p. e1012426, 2024.

[31] Y. He, Y. Wang, H. Yu, H. Wu, M. Gong, Y. Liu, Y. Zhang, X. Wang, M. Sun, S. Li, Y. Qi, X. Li, Y. Li, J. Xu, Y. Bao, and B. Liu, “Hypergen: compact and efficient genome sketching using hyperdimensional computing,” Bioinformatics, vol. 40, no. 7, p. btae452, 2024.

[32] S. N. Aakur, V. R. Laguduva, P. Ramamurthy, and A. Ramachandran, “Tepi: Taxonomy-aware embedding and pseudo-imaging for scarcelylabeled zero-shot genome classification,” IEEE Journal of Biomedical and Health Informatics, vol. 28, no. 4, pp. 2385–2396, 2024.

[33] S. N. Aakur, S. Narayanan, V. Indla, A. Bagavathi, V. Laguduva Ramnath, and A. Ramachandran, “Mg-net: Leveraging pseudo-imaging for multi-modal metagenome analysis,” in International Conference on Medical Image Computing and Computer-Assisted Intervention, pp. 592–602, Springer, 2021.

[34] A. Grover and J. Leskovec, “node2vec: Scalable feature learning for networks,” in Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 855–864, 2016.

[35] T. Goldfarb, V. K. Kodali, S. Pujar, V. Brover, et al., “NCBI RefSeq: reference sequence standards through 25 years of curation and annotation,” Nucleic Acids Research, vol. 53, pp. D243–D257, Jan 2025.

[36] K. He, X. Zhang, S. Ren, and J. Sun, “Deep residual learning for image recognition,” in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 770–778, 2016.

[37] A. Dosovitskiy, “An image is worth 16x16 words: Transformers for image recognition at scale,” arXiv preprint arXiv:2010.11929, 2020.