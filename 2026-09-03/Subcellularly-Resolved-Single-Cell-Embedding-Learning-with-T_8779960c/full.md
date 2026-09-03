# Subcellularly Resolved Single-Cell Embedding Learning with Transcriptomic data, Protein Structure and Localization Information

Zhen Zhou<sup>1</sup>, Jiachen Li<sup>2</sup>, Yuan Liu<sup>1</sup>, Xiaoyong Pan<sup>1</sup>, and Hong-Bin Shen<sup>1,\*</sup>

<sup>1</sup>Institute of Image Processing and Pattern Recognition, Shanghai Jiao Tong University, and Key Laboratory of System Control and Information Processing, Ministry of Education of China, Shanghai, 200240, China

<sup>2</sup>Institute of Process Engineering, Chinese Academy of Sciences, Beijing, 100190, China <sup>\*</sup>Correspondence: hbshen@sjtu.edu.cn

## Abstract

Existing cell embedding methods predominantly rely on transcriptomic or proteomic measurements and represent each cell as a holistic entity, thereby overlooking the subcellular localization of individual molecules. Moreover, they rarely incorporate protein structural information, despite its fundamental role in determining molecular interactions and functions. In this work, we propose a multimodal framework for learning subcellularly resolved cell embeddings by jointly leveraging RNA expression profiles, protein sequence representations, and protein structural information. Specifically, we employ a cross-attention architecture to integrate transcriptomic, sequence, and structural modalities and model their interactions within distinct subcellular compartments. The resulting embeddings represent each cell through its fine-grained subcellular organization, capturing both molecular expression patterns and the functional properties of the associated proteins. By learning cell representations at subcellular resolution, our framework preserves spatially organized biological information while integrating complementary signals across multiple molecular levels. To the best of our knowledge, this is the first framework that produces subcellularly resolved cell embeddings by jointly incorporating transcriptomic information, protein sequence representations, and protein structural knowledge within a unified cross-modal learning paradigm.

## Keywords

Cell Embedding; Subcellular Representation Learning; Single-Cell Transcriptomics; Protein Structure; Protein Sequence; Multimodal Learning; Cross-Attention; Representation Learning

## Introduction

## 1 Introduction

Recent advances in single-cell RNA sequencing (scRNA-seq) technologies have enabled the systematic profiling of cellular states at unprecedented resolution, giving rise to large-scale cell atlases and foundation models for single-cell analysis. Inspired by the success of large language models, a series of single-cell foundation models have been proposed, including Geneformer<sup>1</sup>, $\mathsf { s c G P T } ^ { 2 }$ and scFoundation<sup>3</sup>. These models leverage transformer-based architectures to learn generalizable cellular representations from millions of transcriptomic profiles and have demonstrated strong performance across diverse downstream tasks such as cell type annotation, trajectory inference, and perturbation prediction.

Despite these advances, most existing cell embedding methods remain fundamentally transcriptomecentric. They represent each cell as a collection of gene expression values, implicitly assuming that mRNA abundance is sufficient to capture cellular identity. However, gene expression represents only an intermediate layer of the central dogma and does not directly reflect the functional execution of biological processes. As a result, current cell embeddings often fail to capture key molecular mechanisms underlying cellular behavior.

Proteins are the primary functional molecules in biological systems, directly executing cellular processes through enzymatic activity, molecular interactions, and signaling regulation<sup>4</sup>. Importantly, transcript abundance alone provides an incomplete description of cellular function, because protein activity is determined not only by expression level but also by three-dimensional structure. Proteins with similar expression patterns may exhibit distinct biological functions due to differences in folding topology, binding interfaces, and structural domains. Recent breakthroughs in protein structure prediction, such as AlphaFold<sup>5</sup> and RoseTTAFold<sup>6</sup>, have revealed that protein structures encode rich functional and evolutionary information that cannot be captured by transcriptomic data alone. Therefore, a biologically meaningful cellular representation should integrate both RNA expression and protein-level information, particularly structural properties that directly determine molecular function. However, existing single-cell foundation models rarely incorporate protein structure, limiting their ability to model cellular function in a mechanistically grounded manner.

In addition to molecular composition, cellular function is also strongly influenced by spatial organization. Eukaryotic cells are highly compartmentalized systems in which biomolecules are dynamically localized across distinct subcellular regions, including the nucleus, cytoplasm, mitochondria, and endoplasmic reticulum. Subcellular localization plays a critical role in regulating gene expression, RNA transport, protein trafficking, and signal transduction<sup>7</sup>. Moreover, aberrant localization of RNAs and proteins is closely associated with various diseases, highlighting the importance of spatial context in defining cellular states. Nevertheless, most existing cell embedding approaches treat cells as homogeneous entities and neglect intracellular spatial heterogeneity.

Recent advances in imaging technologies, spatial transcriptomics, and large-scale localization datasets have made it increasingly feasible to incorporate subcellular information into computational models<sup>8,9</sup>. Motivated by these developments, we argue that a comprehensive cellular representation should jointly model transcriptomic signals, protein abundance, protein structural properties, and subcellular localization.

In this work, we propose a cross-attention-based multimodal framework for subcellular cell representation learning. Our model integrates RNA expression matrices with protein features and explicitly incorporates protein structural information to capture functional molecular constraints. Furthermore, we introduce subcellular localization as an additional modality to encode intracellular spatial organization. By jointly modeling molecular abundance, structural biology, and spatial context, our framework learns biologically informed cellular embeddings that better reflect the complexity of cellular systems and enable more accurate downstream biological inference.

## Material & Methods

## Preprocessing and Pre-training

To construct a unified multimodal representation space for RNA and protein data, we first perform systematic preprocessing on both modalities to ensure consistency, quality, and biological interpretability.

The scRNA-seq data is preprocessed using the strategy widely adopted in previous studies<sup>10,11</sup>. After filtering out genes with fewer than 20 expressed counts, the top 3,000 highly variable genes are selected. The gene expression counts are then normalized and log transformed. In detail, each cell’s total counts are scaled to a fixed target sum (e.g., 10,000), as in $\begin{array} { r } { x _ { i j } ^ { n o r m } = \frac { x _ { i j } } { \sum _ { j } x _ { i j } } } \end{array}$ ∗ 10000. Here, $x _ { i j }$ denotes the expression value of gene $j$ in cell i. The log transformation, applied as $x _ { i j } ^ { l o g } = \log ( 1 + x _ { i j } ^ { n o r m } )$ , is widely used in the preprocessing of scRNA-seq data analysis methods to reduce the dynamic range and attenuate extreme values and preserve relative differences among lowly expressed genes<sup>10,12–15</sup>.

For protein data, we collect protein sequences and associated annotations from UniProt<sup>16</sup>. Each protein is represented not only by its amino acid sequence but also by structural and functional annotations when available. To capture structural information, we optionally incorporate embeddings derived from pretrained protein structure models or sequence-based encoders. This allows the model to encode both sequence-level and structure-aware representations of proteins.

To establish a consistent mapping between RNA and protein modalities, we leverage UniProt as a biological bridge. Specifically, gene symbols from the RNA modality are first mapped to their corresponding protein entries in UniProt. This mapping enables alignment between transcriptlevel measurements and protein-level entities, ensuring that each gene expression feature can be associated with its corresponding protein representation when available. Through this procedure, we construct a unified RNA–protein correspondence dictionary, which serves as the foundation for subsequent multimodal fusion.

After preprocessing and alignment, both RNA and protein features are projected into a shared embedding space, enabling direct interaction between modalities in downstream cross-attention modules.

## Protein Structure and RNA Foundation Embedding

To obtain biologically meaningful initial representations for both protein and RNA modalities, we leverage pretrained domain-specific models that encode structural and contextual information beyond raw sequence or expression signals.

For the protein modality, we first retrieve protein sequences and annotations from UniProt<sup>16</sup>. To incorporate structural information, each protein is further processed using FoldExplorer<sup>17</sup>, a structure-aware protein representation model designed for fast and accurate protein structure search via sequence-enhanced graph embedding . FoldExplorer integrates sequence and structural priors to generate embeddings that capture both evolutionary and conformational properties of proteins. Through this process, we obtain an initial protein embedding that encodes sequence identity, structural topology, and functional similarity in a unified representation space.

Beyond sequence and structure, protein function is also strongly influenced by its subcellular localization. To explicitly incorporate spatial cellular context, we further augment the FoldExplorerderived protein embeddings with subcellular localization priors derived from UniProt annotations. Specifically, we construct a seven-dimensional localization vector, where each dimension corresponds to the dominant probability of the protein being localized in one of the major cellular compartments, including cytoplasm, nucleus, plasma membrane, mitochondria, endoplasmic reticulum, Golgi apparatus, and endosome.

This subcellular prior is concatenated with the structural protein embedding, enabling the model to jointly capture molecular-level properties and spatial organizational context. By integrating sequence, structural, and localization information, the resulting protein representation provides a more comprehensive and biologically grounded embedding for downstream multimodal fusion.

For the RNA modality, we utilize scGPT, a generative pre-trained transformer model for singlecell transcriptomic data, to obtain contextualized gene and cell embeddings<sup>18</sup>. Specifically, gene expression matrices are first tokenized at the gene level, where each gene is associated with a unique gene identifier. These gene IDs, together with their corresponding expression values, are input into scGPT to generate latent representations that capture gene-gene dependencies and cell-state-specific transcriptional programs. This allows the RNA embedding to move beyond simple count-based representations and instead reflect higher-order regulatory relationships learned from large-scale single-cell datasets.

By aligning FoldExplorer-derived protein embeddings with scGPT-derived RNA embeddings, we construct modality-specific but biologically informed representations that serve as the foundation for subsequent cross-attention-based multimodal fusion.

## The Framework of RNA&Protein SubCell Embedding

We leverage the cross-attention model<sup>19</sup> as the backbone of our framework. The transcriptomic module takes two types of inputs, including a prior gene knowledge, and their corresponding expression values. In parallel, the proteomic module processes protein tokens as input.

These heterogeneous inputs are first projected into modality-specific embedding spaces through dedicated embedding layers and subsequently encoded using separate transformerbased architectures. Within the transcriptomic pathway, we adopt a masked-attention transformer architecture inspired by scGPT, which is designed to capture complex gene–gene and gene–cell dependencies from partially observed expression profiles.

The resulting RNA embeddings are then used to guide a cross-attention mechanism within the proteomic encoder, enabling RNA-informed modeling of protein abundance from corresponding cell surface protein tokens. Through this cross-modal interaction, transcriptomic signals provide contextual guidance for proteomic representation learning, facilitating biologically consistent alignment between RNA and protein modalities.

Finally, this RNA-guided proteomic reasoning yields a unified multi-omic joint embedding, which integrates complementary information from both modalities and supports a wide range of downstream analytical tasks.

## RNA&Protein Encoder Module

To address the non-sequential nature of single-cell transcriptomic data and to effectively leverage pretrained representations from scGPT, we adopt a masked-attention Transformer architecture to encode RNA embeddings. Given an input RNA embedding matrix

$$
\boldsymbol { h } ^ { ( i ) } \in \mathbb { R } ^ { M \times D } ,\tag{1}
$$

where M denotes the number of genes and D denotes the embedding dimension, we apply a multi-layer Transformer encoder to model gene–gene interactions.

The standard self-attention mechanism is defined as:

$$
\mathsf { A t t e n t i o n } ( Q , K , V ) = \mathsf { S o f t m a x } \left( \frac { Q K ^ { \top } } { \sqrt { d } } + A _ { \mathsf { m a s k } } \right) V ,\tag{2}
$$

where $Q , K , V \in \mathbb { R } ^ { M \times d }$ are query, key, and value matrices, and $A _ { \sf m a s k }$ is a task-specific attention mask.

The projection is computed as:

$$
Q = h W _ { Q } , \quad K = h W _ { K } , \quad V = h W _ { V } ,\tag{3}
$$

where $W _ { Q } , W _ { K } , W _ { V } \in \mathbb { R } ^ { D \times d }$ are learnable parameters.

To adapt to gene expression modeling, we introduce a masked attention<sup>20</sup> strategy that enables autoregressive-style prediction on non-sequential data. Specifically, we partition genes into observed (”known”) and masked (”unknown”) subsets, and define the attention mask as:

$$
A _ { \mathsf { m a s k } } ( i , j ) = { \left\{ \begin{array} { l l } { 0 , } & { { \mathsf { i f ~ } } { \mathsf { g e n e ~ } } j { \mathsf { ~ i s ~ o b s e r v a b l e ~ f o r ~ } } i } \\ { - \infty , } & { { \mathsf { o t h e r w i s e } } } \end{array} \right. }\tag{4}
$$

This prevents each query token from attending to future or unknown genes, enforcing a controlled generation process.

The RNA representation after the final Transformer layer is:

$$
h _ { n } ^ { ( i ) } = \mathsf { T r a n s f o r m e r } ( h ^ { ( i ) } ) \in \mathbb { R } ^ { M \times D } .\tag{5}
$$

To improve computational efficiency for large-scale gene panels (where M can reach tens of thousands), we adopt FlashAttention<sup>21</sup>, a memory-efficient implementation of self-attention that reduces quadratic memory complexity while maintaining exact attention computation.

For the protein modality, we employ a standard Transformer encoder initialized randomly to model intra-protein relationships. Given a protein embedding matrix:

$$
\boldsymbol { p } ^ { ( i ) } \in \mathbb { R } ^ { P \times D } ,\tag{6}
$$

where $P$ is the number of protein tokens, we first apply stacked self-attention layers to capture structural and contextual dependencies among protein tokens.

For the n-th layer, the self-attention operation is defined as:

$$
\mathsf { S e l f A t t } \mathsf { n } _ { n } ( p ) = \mathsf { S o f t m a x } \left( \frac { Q _ { p } K _ { p } ^ { \top } } { \sqrt { d } } \right) V _ { p } ,\tag{7}
$$

where:

$$
Q _ { p } = p W _ { Q } ^ { p } , \quad K _ { p } = p W _ { K } ^ { p } , \quad V _ { p } = p W _ { V } ^ { p } .\tag{8}
$$

Each Transformer block is followed by residual connections, layer normalization, and a positionwise feed-forward network, producing the final intra-protein representation:

$$
\tilde { p } ^ { ( i ) } \in \mathbb { R } ^ { P \times D } .\tag{9}
$$

To integrate transcriptomic context into protein representation learning, we introduce a crossattention mechanism that fuses RNA and protein modalities.

Specifically, RNA embeddings are used as queries and keys, while protein embeddings serve as values:

$$
Q _ { r } = h _ { n } W _ { Q } ^ { c } , \quad K _ { r } = h _ { n } W _ { K } ^ { c } , \quad V _ { p } = \tilde { p } W _ { V } ^ { c } .\tag{10}
$$

The cross-attention operation is then defined as:

$$
\mathsf { C r o s s A t t n } ( h _ { n } , \tilde { p } ) = \mathsf { S o f t m a x } \left( \frac { Q _ { r } K _ { p } ^ { \top } } { \sqrt { d } } \right) V _ { p } .\tag{11}
$$

This produces an RNA-conditioned protein representation:

$$
\boldsymbol { z } ^ { ( i ) } \in \mathbb { R } ^ { P \times d } .\tag{12}
$$

The output is then projected back to the original embedding dimension:

$$
\hat { z } ^ { ( i ) } = z ^ { ( i ) } W _ { O } + \tilde { p } ^ { ( i ) } ,\tag{13}
$$

followed by residual connection and layer normalization.

The resulting embedding:

$$
\hat { z } ^ { ( i ) } \in \mathbb { R } ^ { P \times D }\tag{14}
$$

encodes both transcriptomic context and proteomic structural information, and is used as the final multimodal representation for downstream tasks.

## Cell Representation Learning Via Cross-Modal Fusion

To construct a unified multi-modal cell representation, we design a hierarchical embedding framework that integrates protein-level structure modeling with RNA-guided cross-modal fusion.

For the protein modality, we prepend a special classification token ⟨cls⟩ to the sequence of protein tokens:

$$
\langle \mathrm { c l s } \rangle , P _ { 1 } , P _ { 2 } , \ldots , P _ { N } ,\tag{15}
$$

where N denotes the number of protein tokens. This design allows the model to aggregate global protein-level information into a single latent representation.

After passing through multiple layers of self-attention, the protein sequence is transformed into:

$$
\langle \mathrm { c l s } \rangle ^ { \prime } , P _ { 1 } ^ { \prime } , P _ { 2 } ^ { \prime } , \ldots , P _ { N } ^ { \prime } ,\tag{16}
$$

where $\langle \mathrm { c l s } \rangle ^ { \prime }$ serves as a global protein representation that integrates intra-protein structural and contextual dependencies learned via self-attention.

To further integrate transcriptomic information, we introduce a cross-attention-based fusion module. In this module, protein representations are used as queries, while RNA embeddings serve as keys and values, enabling RNA-guided modulation of protein representations:

$$
Q _ { p } = { \tilde { P } } W _ { Q } , \quad K _ { r } = H W _ { K } , \quad V _ { r } = H W _ { V } ,\tag{17}
$$

where $\tilde { P }$ denotes the protein representation after self-attention, and H denotes the RNA embedding.

The cross-attention operation is defined as Eq. (11). After fusion, we obtain a multi-modal representation:

$$
\tilde { P } _ { \mathfrak { f u s i o n } } \in \mathbb { R } ^ { ( N + 1 ) \times D } ,\tag{18}
$$

where each protein token is enriched with transcriptomic context while preserving structural information from the protein encoder.

Finally, we extract the hidden state corresponding to the ⟨cls⟩ token from the fusion block:

$$
z _ { \tt c e l l } = \tilde { P } _ { \sf t u s i o n } [ \langle \mathrm { c l s } \rangle ] ,\tag{19}
$$

which serves as the final multi-modal cell representation. This embedding simultaneously encodes:

• intra-protein structural organization (from self-attention),

• transcriptomic context (from cross-attention),

• and global cellular state (via ⟨cls⟩ pooling).

## Loss Function

The proposed model is pretrained using a composite objective function designed to learn biologically meaningful multimodal representations from single-cell data. The overall training objective integrates two complementary components: (1) a masked gene expression modeling task within the RNA modality; (2) cross-modal prediction of protein expression conditioned on transcriptomic features. Together, these objectives enable the model to capture both intra-modal gene regulatory structures and inter-modal transcript–protein relationships in a unified framework.

## Masked Gene Expression Modeling

To learn gene–gene dependencies from partially observed transcriptomic profiles, we adopt a masked gene expression (MGE) modeling objective, inspired by masked language modeling in Transformer-based architectures.

Given a cell i with gene expression vector $\boldsymbol { x } ^ { ( i ) } \in \mathbb { R } ^ { M }$ , we randomly mask a subset of genes $\mathcal { U } _ { \mathsf { u n k } } ^ { ( i ) } \subset \{ 1 , \dots , M \}$ during training.

For each masked gene j, a prediction head (MLP) maps its hidden representation $h _ { j } ^ { ( i ) }$ to the reconstructed expression value:

$$
\begin{array} { r } { \hat { x } _ { j } ^ { ( i ) } = \mathrm { M L P } ( h _ { j } ^ { ( i ) } ) . } \end{array}\tag{20}
$$

The MGE objective minimizes the mean squared error between predicted and observed expression values:

$$
\mathcal { L } _ { \sf M G E } = \frac { 1 } { | \mathcal { U } _ { \sf u n k } ^ { ( i ) } | } \sum _ { j \in \mathcal { U } _ { \sf u n k } ^ { ( i ) } } \left( x _ { j } ^ { ( i ) } - \hat { x } _ { j } ^ { ( i ) } \right) ^ { 2 } .\tag{21}
$$

Optimizing this objective encourages the model to learn gene–gene co-expression structure and latent regulatory dependencies from incomplete transcriptomic observations.

## Cross-Modal Protein Prediction Loss

To align transcriptomic and proteomic modalities, we introduce a cross-modal prediction objective that estimates protein expression from RNA-derived representations.

Let $\boldsymbol { p } ^ { ( i ) } \in \mathbb { R } ^ { P }$ denote the protein expression vector for cell i. Given the RNA embedding $\it { h ^ { ( i ) } }$ the model predicts protein abundance via a cross-attention-based fusion module followed by a regression head:

$$
\begin{array} { r } { \hat { p } ^ { ( i ) } = f _ { \theta } ( h ^ { ( i ) } , p ^ { ( i ) } ) , } \end{array}\tag{22}
$$

where $f _ { \theta } ( \cdot )$ denotes the multimodal fusion network.

The corresponding loss is defined as:

$$
\mathcal { L } _ { \mathtt { c r o s s } } = \| p ^ { ( i ) } - \hat { p } ^ { ( i ) } \| _ { 2 } ^ { 2 } .\tag{23}
$$

This objective enforces consistency between transcriptomic and proteomic representations and facilitates biologically grounded cross-modal alignment.

## Overall Objective

The final training objective is a weighted combination of the three losses:

$$
\mathcal { L } = \lambda _ { 1 } \mathcal { L } _ { \sf M G E } + \lambda _ { 2 } \mathcal { L } _ { \sf c r o s s }\tag{24}
$$

where $\lambda _ { 1 } = 0 . 6$ and $\lambda _ { 2 } = 0 . 4$ control the contributions of the masked gene expression modeling loss and the cross-modal prediction loss, respectively.

## Results

## Multimodal RNA–Protein representations improve cell type identification in PBMC datasets

To evaluate the effectiveness of our multimodal cell embedding framework for cell type identification, we applied our model to PBMC datasets<sup>22</sup> containing diverse immune cell populations. The learned embeddings were visualized using UMAP and compared with experimentally annotated cell identities (Fig.2a). The multimodal representations generated by our model successfully preserved the global organization of immune populations, including B cells, T cells, monocytes, natural killer (NK) cells, dendritic cells, and platelets. The predicted cell type distribution showed strong concordance with the original annotations, demonstrating that our model effectively captures biologically meaningful cellular structures.

The UMAP visualization further demonstrated that our model was capable of separating closely related immune subpopulations. In particular, distinct T cell states, including CD4 naive, CD4 TCM, CD4 TEM, CD8 naive, CD8 TCM, and CD8 TEM cells, were well resolved in the embedding space. These results indicate that integrating transcriptomic and protein-level information enables the model to capture subtle phenotypic differences that are difficult to distinguish using transcriptomic information alone.

To quantitatively evaluate classification performance, we calculated the confusion matrix across all immune cell subtypes (Fig.2b). The majority of cell populations showed strong diagonal enrichment, indicating high consistency between predicted and annotated labels. Major immune populations, including CD14 monocytes, CD16 monocytes, naive and memory B cells, NK cells, and T cell populations, achieved high classification accuracy with minimal misclassification. Moreover, the model demonstrated robust performance in distinguishing biologically similar cell states, highlighting the advantage of multimodal representation learning for fine-grained immune cell annotation.

We further benchmarked our approach against representative single-cell representation and integration methods, including the transcriptomic foundation model scGPT and the widely used conventional integration framework Seurat<sup>23</sup>, using four commonly used evaluation metrics: accuracy, precision, recall, and macro-F1 score (Fig. 2c). Cell-type labels were obtained from curated annotations provided by the original datasets and were used as reference labels for evaluation.

On the 10x Multiome PBMC dataset, our method achieved an accuracy of 0.962, precision of 0.961, recall of 0.937, and macro-F1 score of 0.947. Compared with scGPT, our approach achieved consistent improvements across most evaluation metrics. These improvements indicate that incorporating complementary protein-derived features and subcellular-scale information into transcriptomic representations provides additional biological signals for distinguishing closely related cell populations.

Furthermore, on the multi-tissue immune cell benchmark, our model achieved accuracy, precision, recall, and macro-F1 scores of 0.922, 0.898, 0.920, and 0.905, respectively. These results substantially outperformed Seurat, which obtained scores of 0.790, 0.848, 0.732, and 0.761, respectively. The improved recall and macro-F1 scores indicate that our multimodal representation provides better generalization capability across heterogeneous immune cell populations.

Overall, these results demonstrate that explicitly modeling RNA–protein interactions through cross-modal attention enables the construction of more informative cell embeddings, leading to accurate cell type identification and robust transferability across diverse single-cell datasets.

## Effect of Subcellular Cell Embedding on Cell Type Identification

To investigate the contribution of subcellular-scale information to cell representation learning, we performed an ablation analysis by comparing models with and without the incorporation of subcellular embeddings. Although the introduced subcellular representation consists of only seven dimensions, it provides complementary biological information beyond conventional transcriptomic features. As shown in the confusion matrix analysis (Fig.3b), incorporating subcellular embeddings improved the discrimination of several biologically related cell populations. In particular, the classification accuracy of CD4 TCM cells was enhanced, accompanied by a reduction in misclassification between CD4 TCM and cDC2 populations, suggesting that subcellular localization information contributes to resolving subtle cellular heterogeneity(Fig.4b). The improvement in CD4<sup>+</sup> TCM identification may be attributed to the incorporation of subcellular-aware features. CD4<sup>+</sup> TCM cells possess distinct intracellular states characterized by enhanced mitochondrial fitness and specialized metabolic programs required for long-term persistence and rapid immune recall responses.<sup>24</sup> Moreover, T cell activation and differentiation are accompanied by dynamic redistribution of signaling proteins among membrane, cytoplasmic, and nuclear compartments.<sup>25</sup> Therefore, integrating subcellular localization information provides complementary biological cues beyond transcript abundance, facilitating the resolution of subtle heterogeneity among immune cell populations.

Furthermore, the model incorporating subcellular-scale embeddings consistently achieved improved performance across multiple evaluation metrics compared with the model without subcellular information (Fig.3c). On the 10x Multiome PBMC dataset, the incorporation of subcellular embeddings improved the overall classification performance, increasing the accuracy from 0.92 to 0.94 and the macro-F1 score from 0.90 to 0.92. Similar improvements were observed on the multi-tissue immune cell dataset, demonstrating that subcellular representations provide robust and transferable biological signals across diverse cellular contexts.

These results indicate that even a compact seven-dimensional subcellular embedding can effectively complement transcriptomic information, enabling more informative cell representations and enhancing downstream cell type annotation.

## Cross-modal Representation Learning Enables Robust Batch Integration while Preserving Cellular Identity

Batch effects arising from donor variability, experimental conditions, and sequencing platforms represent a major challenge in single-cell analysis. Effective integration requires removing technical variations while preserving biologically meaningful differences between cell populations. However, existing integration approaches primarily rely on transcriptomic information, which may be insufficient for distinguishing biological signals from batch-associated fluctuations, particularly in heterogeneous and multi-modal datasets.

To evaluate the ability of our model to learn batch-robust cellular representations, we performed integration experiments across multiple heterogeneous single-cell datasets containing diverse biological backgrounds and technical conditions. Our framework jointly models RNA expression, protein abundance, and subcellular-scale information through a cross-modal attention mechanism, enabling the model to capture shared biological characteristics across datasets while reducing modality-specific and batch-specific variations.

We first evaluated batch integration performance on multi-batch scRNA-seq datasets collected from different donors. As summarized in Table 1, our method achieved consistently improved biological conservation compared with existing approaches. On the Human PBMCs Covid dataset, our model obtained the highest AvgBIO score (0.657), together with the highest $\mathsf { N M } \mathsf { I } _ { c e l l } \ ( 0 . 7 4 2 )$ and $\mathsf { A R l } _ { c e l l }$ (0.691), outperforming scGPT $( \mathsf { A v g B l O } = 0 . 6 2 6 )$ , TotalVI (0.607), and Scanpy (0.519). These results indicate that our learned representations better preserve cell-type-specific biological structures while effectively integrating cells across heterogeneous batches.

We further evaluated the integration capability on a more complex multi-modal benchmark consisting of TEA-seq, ECCITE-seq, and CITE-seq datasets. By jointly incorporating transcriptomic, proteomic, and subcellular information, our method achieved superior overall integration performance, with an AvgBIO score of 0.567 and an Overall score of 0.728. Compared with scGPT, our approach improved biological conservation (AvgBIO: 0.567 vs. 0.517), cell-type clustering consistency $( \mathsf { N M l } _ { c e l l } .$ 0.638 vs. 0.548; $\mathsf { A R l } _ { c e l l } .$ 0.408 vs. 0.339), and batch correction performance (AvgBATCH: 0.968 vs. 0.951). In particular, the improvements in Graph Connectivity (0.992 vs. 0.988) and $\mathsf { A S W } _ { b a t c h }$ (0.943 vs. 0.913) demonstrate that our model effectively aligns cells from different experimental conditions while maintaining biologically meaningful structures.

Visualization of the learned embeddings further confirmed these quantitative improvements (Fig.4). Compared with conventional integration approaches, our method generated a more coherent latent space in which cells from different batches were effectively aligned while retaining distinct immune cell identities. The UMAP visualization showed clear separation of major immune populations, including B cells, T cell subsets, monocyte populations, and NK cells, demonstrating that the learned representation preserves both global lineage structure and fine-grained cellular heterogeneity.

To further quantify biological preservation, we calculated the biological conservation score (AvgBIO) across different integration methods (Fig.4A). Our method achieved the highest Avg-BIO score (0.795), exceeding scGPT (0.723), Harmony (0.627), and Seurat (0.541). This improvement suggests that incorporating additional biological modalities provides complementary information beyond transcriptomic representations alone, enabling more accurate preservation of cellular identity during integration.

We further investigated the contribution of multimodal information to representation learning. The incorporation of protein features provided complementary phenotypic information beyond RNA expression, improving the resolution of closely related immune populations that exhibit overlapping transcriptional profiles. Moreover, the integration of subcellular-scale embeddings further enhanced representation robustness by introducing biologically informative constraints associated with cellular organization and molecular localization patterns. Unlike transcriptomic signals that may vary substantially across experimental batches, subcellular information provides an additional biological context that helps stabilize cell-state representations.

In particular, the improved representation quality enhanced the discrimination of closely related immune populations. For example, incorporation of subcellular localization information improved the identification of $\mathtt { C D 4 } ^ { + }$ TCM cells and reduced confusion with transcriptionally similar populations such as cDC2, suggesting that subcellular-aware representations capture additional intracellular organization and functional state information beyond gene expression patterns.

Together, these results demonstrate that cross-modal representation learning integrating RNA, protein, and subcellular-scale information enables robust batch correction while preserving biologically meaningful cellular variation across heterogeneous single-cell datasets.

Table 1: Benchmarking single-cell integration performance across heterogeneous datasets. Methods are evaluated from two complementary aspects: biological conservation and batch correction. Biological conservation is assessed using AvgBIO, which integrates normalized mutual information $( \mathsf { N M l } _ { c e l l } )$ , adjusted Rand index $( \mathsf { A R l } _ { c e l l } )$ , and average silhouette width $( \mathsf { A S W } _ { c e l l } )$ based on cell-type annotations. Batch correction performance is evaluated using AvgBATCH, combining $\mathsf { A S W } _ { b a t c h }$ and graph connectivity metrics. Higher values indicate better performance. The best-performing method in each dataset is highlighted in bold.
<table><tr><td rowspan="2"></td><td rowspan="2">Model</td><td colspan="4">Biological Conservation</td><td colspan="4">Batch Correction</td></tr><tr><td>AvgBIO</td><td> $\mathsf { N M l } _ { c e l l }$ </td><td> $\mathsf { A R l } _ { c e l l }$ </td><td> $\mathsf { A S W } _ { c e l l }$ </td><td>AvgBATCH</td><td> $\mathsf { A S W } _ { b a t c h }$ </td><td>GraphConn</td><td>Overall</td></tr><tr><td rowspan="4">Human PBMCs Covid</td><td>Our method (fine-tuned)</td><td>0.657</td><td>0.742</td><td>0.691</td><td>0.538</td><td></td><td></td><td></td><td></td></tr><tr><td>scGPT (fine-tuned)</td><td>0.626</td><td>0.710</td><td>0.608</td><td>0.545</td><td></td><td></td><td></td><td></td></tr><tr><td>TotaIVI</td><td>0.607</td><td>0.689</td><td>0.608</td><td>0.522</td><td></td><td></td><td></td><td></td></tr><tr><td>Scanpy</td><td>0.519</td><td>0.589</td><td>0.475</td><td>0.494</td><td></td><td></td><td></td><td></td></tr><tr><td rowspan="2">TEA-seq+ECCITE-seq+CITE-seq</td><td>Our method (fine-tuned)</td><td>0.567</td><td>0.638</td><td>0.408</td><td>0.657</td><td>0.968</td><td>0.943</td><td>0.992</td><td>0.728</td></tr><tr><td>scGPT (fine-tuned)</td><td>0.517</td><td>0.548</td><td>0.339</td><td>0.664</td><td>0.951</td><td>0.913</td><td>0.988</td><td>0.690</td></tr></table>

## Discussion

In this work, we present a multimodal framework for learning biologically informed single-cell representations by jointly integrating transcriptomic and proteomic information. Unlike existing single-cell foundation models that primarily rely on gene expression profiles, our approach explicitly incorporates protein-level structure and subcellular spatial context, enabling a more comprehensive characterization of cellular states.

Specifically, we propose a cross-attention-based architecture that aligns RNA and protein modalities within a unified representation space. On the protein side, we leverage structureaware embeddings derived from FoldExplorer, enriched with subcellular localization priors from UniProt annotations. On the RNA side, we adopt a pretrained Transformer model inspired by scGPT to capture gene regulatory dependencies from high-dimensional expression data. By bridging these two modalities through cross-attention mechanisms, our model learns biologically grounded interactions between transcriptomic programs and proteomic functions.

Furthermore, we design a CLS-token-based fusion strategy to obtain a compact yet expressive cell-level representation, which simultaneously encodes intra-protein structural information, transcriptomic context, and spatial cellular organization. The proposed framework is trained using a composite objective that combines masked gene modeling and cross-modal prediction, enabling both intra-modal structure learning and inter-modal alignment.

Despite these advances, several limitations remain. First, the current model relies on existing protein annotations and structure-derived embeddings, which may not fully capture dynamic conformational changes or context-specific protein behavior. Second, subcellular localization is represented as a coarse-grained prior, and more fine-grained spatial modeling could further improve biological fidelity. Third, the current framework focuses on static snapshots of cells and does not explicitly model temporal dynamics such as differentiation trajectories or perturbation responses.

In future work, we envision extending this framework in several directions. First, integrating dynamic protein structure representations and spatially resolved omics data could provide a more detailed view of intracellular organization. Second, incorporating perturbation-aware modeling would enable the simulation and prediction of cellular responses under genetic or pharmacological interventions, paving the way toward virtual cell modeling. Finally, scaling the framework to larger multi-modal datasets and more diverse biological conditions may further enhance its generalization ability and applicability in biomedical research.

## RESOURCE AVAILABILITY

## Lead contact

Requests for further information and resources should be directed to and will be fulfilled by the lead contact, Hongbin Shen (hbshen@sjtu.edu.cn).

## Data and code availability

The code to reproduce results, together with documentation and examples of usage, is available on GitHub at https://github.com/ZHOUZHEN2002/RNA Protein/tree/master

## ACKNOWLEDGMENTS

Acknowledgements This work was supported by the National Natural Science Foundation of China (No. 62573293, 62473257), and the Science and Technology Commission of Shanghai Municipality (No. 24ZR1435300, 24510714300).

## AUTHOR CONTRIBUTIONS

Author contributions: H.S. and Z.Z. designed research; Z.Z. performed research; J.L. and X.P. contributed new reagents/analytic tools; Z.Z. analyzed data; and Z.Z., Y.L., J.L. and H.S. wrote the paper.

## DECLARATION OF INTERESTS

The authors declare no competing interests.

## DECLARATION OF GENERATIVE AI AND AI-ASSISTED TECH-NOLOGIES

During the preparation of this manuscript, GPT-5 was used solely to assist with language editing and improve the clarity and readability of the text. All scientific content, data analysis, interpretation, and conclusions were developed and verified by the authors, who take full responsibility for the final manuscript.

# MAIN FIGURE TITLES AND LEGENDS

![](images/c081cdfbbd731aa15adb5dd336706b024ea4458c3f9a0bf65f93aeb3d9da21d3.jpg)

## Figure 1. Overview of the proposed multimodal framework for subcellular cell representation learning.

The model integrates protein structure-aware representations and RNA transcriptomic embeddings through a unified cross-attention architecture to generate biologically informed cell representations.

Left: Protein encoding module. Protein sequences and annotations are retrieved from UniProt and encoded using a structure-aware embedding model (FoldExplorer), which integrates sequence and structural priors to produce protein representations. Subcellular localization information is further incorporated as a 7-dimensional vector capturing major cellular compartments, including cytoplasm, nucleus, plasma membrane, mitochondria, endoplasmic reticulum, Golgi apparatus, and endosome. These features are fused and processed through a Transformer encoder with self-attention layers to obtain protein-level representations.

Right: RNA encoding module. Single-cell gene expression matrices are processed using a masked-attention Transformer inspired by scGPT. The model learns gene–gene dependencies via masked gene modeling, enabling robust reconstruction of missing gene expression values. FlashAttention is used to improve computational efficiency for large gene sets.

Middle: Cross-modal fusion. RNA embeddings are used to guide protein representation learning through a cross-attention mechanism, where transcriptomic signals act as contextual information for proteomic encoding. The fused representations are aggregated using a CLS token to generate the final multi-modal cell embedding, which captures transcriptomic, proteomic, structural, and spatial information in a unified latent space.

![](images/1f1ae00f48767cad97a7b5acbfbe477aa07b4da8e96ae6a16db7ea0c2f6b48b8.jpg)

a.  
![](images/1a8fc4ca7a4e14803005f932c2cb2dfd917c956a77dbb4ef1bd3d00dbb81286b.jpg)

![](images/ca9cd29dc0405fccbd4d1d40ec2759225f80a4a22b839227899a90f18a3a0e27.jpg)

![](images/f50f229446e6d10f70670c803ccf24e0fac16f6d97437a5ba71443ed2ef9c664.jpg)  
Figure 2. Cross-modal RNA–protein representation learning improves PBMC cell type identification.

(a) UMAP visualization of PBMC cells embedded by the proposed RNA–protein fusion model. The left panel shows manually annotated cell populations, whereas the right panel displays predicted cell identities based on the learned multimodal embeddings.

(b) Confusion matrix illustrating classification accuracy across immune cell subtypes, demonstrating the capability

of the model to distinguish closely related immune populations. (c) Comparison of cell annotation performance between our method, scGPT, and Seurat using four evaluation metrics (accuracy, precision, recall, and macro-F1) on a multi-tissue immune cell benchmark and the 10x Multiome PBMC dataset.

b.  
![](images/bcac3b4b4549f64b0c3c57bde579213fd256c907c72ef2e2142c1d8c978e3aae.jpg)

## Figure 3. Effect of subcellular-scale cell embedding on cell representation and cell type identification.

(a) UMAP visualization of PBMC cells based on the learned cell embeddings incorporating subcellular-scale information. Cells are colored according to manually annotated cell types, demonstrating the preservation of diverse immune cell populations and the ability of the proposed representation to capture biologically meaningful cellular heterogeneity.

(b) Confusion matrix of cell type classification performance using the subcellular-enhanced cell embeddings. The model achieves high classification accuracy across major immune populations and effectively resolves closely related cell types, indicating that subcellular information provides complementary signals for improving cell identity discrimination.

(c) Ablation analysis evaluating the contribution of subcellular-scale embeddings. The performance of models with and without the incorporation of subcellular embeddings is compared on the 10x Multiome PBMC dataset and the multi-tissue immune cell benchmark using four evaluation metrics, including accuracy, precision, recall, and macro-F1 score. Despite consisting of only seven dimensions, the compact subcellular representation consistently improves cell type annotation performance, demonstrating that subcellular-scale information provides additional biological context beyond transcriptomic features and enhances cell representation learning.

![](images/dfacb9aac27ef5605ec15c0d1260d2c34576c4694eeddf1c104495767754dda6.jpg)

![](images/6a1ce98429fa10cec18b908e50572433f83fb16fa09b1d3cacefcac6fc7ce7ad.jpg)

## Figure 4. UMAP visualization of the healthy PBMC dataset after integration using different methods.

(a) Quantitative comparison of biological conservation scores (AvgBIO) among different methods. Our method achieves the highest biological conservation score (AvgBIO = 0.795), outperforming scGPT (0.723), Harmony (0.627), and Seurat (0.541), indicating improved preservation of intrinsic biological structures during representation learning.

(b) UMAP visualization of cell embeddings generated by our method. Cells are colored according to curated immune cell identities, including B naive, CD4<sup>+</sup> Naive, CD4<sup>+</sup> TCM, CD8<sup>+</sup> Naive, CD8<sup>+</sup> TEM, CD14<sup>+</sup> Mono, CD16<sup>+</sup> Mono, and NK cells. The learned representation enables clear separation of major immune populations while preserving biologically meaningful relationships among related cell states. In particular, incorporating subcellular localization information improves the discrimination of CD4<sup>+</sup> TCM cells and reduces their confusion with closely related cDC2 populations, suggesting that subcellular-aware representations facilitate the resolution of subtle cellular heterogeneity.

(c–f) UMAP visualizations of the healthy PBMC dataset after integration using (c) Our method, (d) scGPT, (e) Seurat, and (f) Harmony. In each panel, the left UMAP is colored by batch (donor; P1–P8) to assess batch mixing, whereas the right UMAP shows the same cells colored by reference cell-type annotations to evaluate the preservation of biological identity. Effective integration is reflected by the mixing of donor labels within shared embedding regions while maintaining coherent separation of known immune cell populations. Compared with conventional integration methods, our approach provides improved alignment between the learned embedding structure and biological cell identities, achieving the highest AvgBIO score (0.795), compared with scGPT (0.723), Seurat (0.627), and Harmony (0.541). AvgBIO values are displayed above the cell-type legends.

## References

[1] Christina V Theodoris, Ling Xiao, Ashish Chopra, et al. Transfer learning enables predictions in network biology. Nature, 2023.

[2] Haotian Cui, Chloe Wang, Hassaan Maan, Kuan Pang, Fengning Luo, Nan Duan, and Bo Wang. scgpt: toward building a foundation model for single-cell multi-omics using generative ai. Nature methods, 21(8):1470–1480, 2024.

[3] Minsheng Hao, Jing Gong, Xin Zeng, Chiming Liu, Yucheng Guo, Xingyi Cheng, Taifeng Wang, Jianzhu Ma, Xuegong Zhang, and Le Song. Large-scale foundation model on singlecell transcriptomics. Nature methods, 21(8):1481–1491, 2024.

[4] Bruce Alberts et al. Molecular biology of the cell. Garland Science, 2017.

[5] John Jumper, Richard Evans, Alexander Pritzel, et al. Highly accurate protein structure prediction with alphafold. Nature, 596:583–589, 2021.

[6] Rohith Krishna, Jue Wang, Woody Ahern, Pascal Sturmfels, Preetham Venkatesh, Indrek Kalvet, Gyu Rie Lee, Felix S Morey-Burrows, Ivan Anishchenko, Ian R Humphreys, et al. Generalized biomolecular modeling and design with rosettafold all-atom. Science, 384 (6693):eadl2528, 2024.

[7] Hirofumi Kobayashi, Keith C Cheveralls, Manuel D Leonetti, and Loic A Royer. Selfsupervised deep learning encodes high-resolution features of protein subcellular localization. Nature methods, 19(8):995–1003, 2022.

[8] Anjali Rao, Dalia Barkley, Gustavo S Franc¸a, and Itai Yanai. Exploring tissue architecture using spatial transcriptomics. Nature, 596(7871):211–220, 2021.

[9] Yuexu Jiang, Duolin Wang, Weiwei Wang, and Dong Xu. Computational methods for protein localization prediction. 2021.

[10] Manu Setty, Vaidotas Kiseliovas, Jacob Levine, Adam Gayoso, Linas Mazutis, and Dana Pe’Er. Characterization of cell fate probabilities in single-cell data with palantir. Nature biotechnology, 37(4):451–460, 2019. URL: https://www.nature.com/articles/ s41587-019-0068-4 (accessed May 14, 2025).

[11] Volker Bergen, Marius Lange, Stefan Peidli, F Alexander Wolf, and Fabian J Theis. Generalizing rna velocity to transient cell states through dynamical modeling. Nature biotechnology, 38(12):1408–1414, 2020. URL: https://www.nature.com/articles/s41587-020-0591-3 (accessed May 14, 2025).

[12] Kelly Street, Davide Risso, Russell B Fletcher, Diya Das, John Ngai, Nir Yosef, Elizabeth Purdom, and Sandrine Dudoit. Slingshot: cell lineage and pseudotime inference for singlecell transcriptomics. BMC genomics, 19:1–16, 2018. URL: https://link.springer.com/ article/10.1186/s12864-018-4772-0 (accessed May 14, 2025).

[13] Cole Trapnell, Davide Cacchiarelli, and Xiaojie Qiu. Monocle: Cell counting, differential expression, and trajectory analysis for single-cell rna-seq experiments. Bioconductor. https://www. bioconductor. org/packages/release/bioc/html/monocle. html, 2017. URL: https://bioconductor.statistik.tu-dortmund.de/packages/3.5/bioc/ vignettes/monocle/inst/doc/monocle-vignette.pdf (accessed May 14, 2025).

[14] Shengyu Li, Pengzhi Zhang, Weiqing Chen, Lingqun Ye, Kristopher W Brannan, Nhat-Tu Le, Jun-ichi Abe, John P Cooke, and Guangyu Wang. A relay velocity model infers celldependent rna velocity. Nature biotechnology, 42(1):99–108, 2024. URL: https://www. nature.com/articles/s41587-023-01728-5 (accessed May 14, 2025).

[15] Jiachen Li, Xiaoyong Pan, Ye Yuan, and Hong-Bin Shen. Tfvelo: gene regulation inspired rna velocity estimation. Nature Communications, 15(1):1387, 2024. URL: https://www. nature.com/articles/s41467-024-45661-w (accessed May 14, 2025).

[16] UniProt Consortium. Uniprot: a hub for protein information. Nucleic acids research, 43(D1): D204–D212, 2015.

[17] Hong-Bin Shen, Yuan Liu, Ying Zhang, and Zhen Zhou. Foldexplorer: Fast and accurate protein structure search with sequence-enhanced graph embedding. 2024.

[18] Zhiyuan Chen et al. scgpt: toward building a foundation model for single-cell multi-omics using generative ai. bioRxiv, 2023.

[19] Hezheng Lin, Xing Cheng, Xiangyu Wu, and Dong Shen. Cat: Cross attention in vision transformer. In 2022 IEEE international conference on multimedia and expo (ICME), pages 1–6. IEEE, 2022.

[20] Bowen Cheng, Ishan Misra, Alexander G Schwing, Alexander Kirillov, and Rohit Girdhar. Masked-attention mask transformer for universal image segmentation. In Proceedings of the IEEE/CVF conference on computer vision and pattern recognition, pages 1290–1299, 2022.

[21] Tri Dao, Daniel Y Fu, Stefano Ermon, Atri Rudra, and Christopher Re. Flashattention: Fast ´ and memory-efficient exact attention with io-awareness. In Advances in neural information processing systems, 2022.

[22] Yuhan Hao, Stephanie Hao, Erica Andersen-Nissen, William M Mauck, Shiwei Zheng, Andrew Butler, Maddie J Lee, Aaron J Wilk, Charlotte Darby, Michael Zager, et al. Integrated analysis of multimodal single-cell data. Cell, 184(13):3573–3587, 2021.

[23] Alexander Gribov, Martin Sill, Sonja Luck, Frank R¨ ucker, Konstanze D¨ ohner, Lars Bullinger,¨ Axel Benner, and Antony Unwin. Seurat: visual analytics for the integrated analysis of microarray data. BMC medical genomics, 3(1):21, 2010.

[24] James R Rose, Bagdeser Akdogan-Ozdilek, Andrew R Rahmberg, Michael D Powell, Sakeenah L Hicks, Christopher D Scharer, and Jeremy M Boss. Distinct transcriptomic and epigenomic modalities underpin human memory t cell subsets and their activation potential. Communications Biology, 6(1):363, 2023.

[25] Rubin Narayan Joshi, Charlotte Stadler, Robert Lehmann, Janne Lehtio, Jesper Tegn¨ er,´ Angelika Schmidt, and Mattias Vesterlund. Tcellsubc: an atlas of the subcellular proteome of human t cells. Frontiers in immunology, 10:2708, 2019.