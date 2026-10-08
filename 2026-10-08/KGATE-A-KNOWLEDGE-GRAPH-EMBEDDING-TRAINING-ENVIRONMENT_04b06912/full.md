# KGATE : A KNOWLEDGE GRAPH EMBEDDING TRAINING ENVIRONMENT

Benjamin Loire<sup>1,2</sup>, Galadriel Brière<sup>1</sup>, Célia Brahimi<sup>1</sup>, Antoine Toffano<sup>3</sup>, and Anaïs Baudot<sup>1,4</sup>

<sup>1</sup>Aix Marseille Univ., INSERM, Marseille Medical Genetics, Systems Biomedicine Team, Marseille, France <sup>2</sup>Neurology Therapeutic Area, R&D Servier Paris-Saclay Institut, Gif-sur-Yvette, France <sup>3</sup>LIRMM, Univ. Montpellier, CNRS, Montpellier, France <sup>4</sup>CNRS, Marseille, France

## ABSTRACT

Knowledge graph embedding (KGE) models encode the entities and relations of a knowledge graph into a low-dimensional latent space, enabling tasks such as classification or link prediction. Most KGE models follow an autoencoder architecture, in which an encoder projects the knowledge graph into the latent space and a decoder reconstruct it. Combining both encoder and decoder components is increasingly needed, yet existing libraries rarely support complete autoencoders, are often unmaintained, rely on undocumented default hyperparameters, and produce results that cannot be compared across libraries. Here we present KGATE (Knowledge Graph Autoencoder Training Environment), a modular Python library built on PyTorch Geometric and TorchKGE. KGATE lets users assemble initializers, encoders, decoders, losses, negative samplers, and evaluation metrics as building blocks, or plug in their own block. KGATE includes a preprocessing procedure that controls data leakage, a builtin training pipeline, and reproducibility by design. Benchmarks against six existing KGE libraries show that KGATE training time is comparable with the fastest libraries while offering a broader set of features.

Availability: https://github.com/BAUDOTlab/KGATE

Keywords Knowledge Graph Embedding · Library · Graph Representation Learning · Reproductibility

## 1 Introduction

A Knowledge Graph (KG) is a data structure encoding entities as nodes and their interactions as edges, where the interaction type is explicitly stated. Formally, a KG is described as a set of triplets (head, relation, tail), where head and tail are the origin and destination nodes, and relation is the type of edge connecting them. This structured representation makes KGs particularly suited to represent heterogeneous domain knowledge. For example, biomedical knowledge graphs routinely encode tens of thousands of associations between genes, diseases, and drugs.

Given the scale of many KGs, their manipulation and interpretation is often computationally intractable. Knowledge Graph Embedding (KGE) addresses this by encoding entities and relations into a continuous, low-dimensional latent space. Once trained, a KGE model can be used to perform downstream tasks such as node or edge classification, or link prediction. The classes may correspond to any class label provided by the user, including multi-class biological labels or simply presence/absence in the KG. Link prediction, in contrast, identifies the most likely entity or relation to complete a triplet when given two of the three parts.

The most common architecture for a KGE model is the autoencoder: the encoder projects the graph nodes and edges into a latent space, and the decoder learns how to reconstruct the original graph in a self-supervised training loop. Encoders learn an embedding function that aggregates, for each node, information from its neighbors. This makes them inductive: the learned function applies to nodes unseen during training. Decoders score each triplet with a scoring function. Since each scoring function captures some topologies better than others, decoder performance varies with the relation of interest. Their main limitation is transductivity: they can only score triplets whose nodes were seen during training.

However, most KGE papers use only a decoder with a random initialization for the latent space and no encoder (Bordes et al., 2013; Boschin, 2020; Lin et al., 2015). On the other hand, papers introducing novel encoder architectures often use only DistMult as the decoder (Brody et al., 2021; Hamilton et al., 2017; Schlichtkrull et al., 2018). The rise in multimodal and ever-expanding KG datasets requires the inductive abilities of an encoder combined with the decoder specificity. Thus, modern KGE tasks call for an architecture able to implement a robust combination of encoders and decoders.

Several libraries including Pytorch Geometric (Fey et al., 2025), TorchKGE (Boschin, 2020), Ampligraph (Costabello et al., 2024), LibKGE (Broscheit et al., 2020), DGL-KE (Zheng et al., 2020) and PyKEEN (Ali et al., 2020) provide frameworks for KGE model training, with implementations of multiple KGE algorithms. Yet, except for PyKEEN and PyTorch Geometric, these libraries are mostly unmaintained. In addition, they are not interoperable, and the results obtained with one library cannot be compared with the results obtained with another (Supplementary materials section 1). Most of these libraries also propagate hardcoded and undocumented default hyperparameters, which might not be the optimal hyperparameters (Prieto & Garrido-Merchán, 2026).

In light of the currently limited KGE ecosystem, we propose here a new library, KGATE, for Knowledge Graph Autoencoder Training Environment. KGATE implements complete autoencoders while enforcing best practices and remaining accessible to newcomers. KGATE is built as a set of building blocks to be modular and adapt to any KGE application.

## 2 Implementation

KGATE, for Knowledge Graph Autoencoder Training Environment, is a modern KGE library built upon Pytorch Geometric (Fey et al., 2025) and TorchKGE (Boschin, 2020) while extending the capabilities of both libraries. KGATE enables fast prototyping and offers an extensive KGE API for advanced users. Its main characteristics are :

• A modular autoencoder interface that allows users to assemble building blocks (figure 1) or plug their own code with minimal friction;

• A preprocessing procedure that addresses data leakage biases identified in Brière et al. (2026);

• An off-the-shelf training pipeline;

• Extensive documentation for new KGE users;

• Fully exposed API to build new models;

• Reproducibility by design;

• Cross-library comparison of models;

## 2.1 Data Layer

In KGATE, all the data is contained in the KnowledgeGraph class which provides tensor indexing for all triplets, embeddings, mapping dictionaries, split masks, and a range of basic utility for simple graph exploration. As all the data is held in the KnowledgeGraph object, every other building block acts as a transformation function applied to this centralized data. KGATE comes with multiple builtin datasets, such as the standard KGs used in benchmark FB15k-237 (Toutanova & Chen, 2015) and WN18RR (Dettmers et al., 2018), as well as PrimeKG (Chandak et al., 2023), a biomedical knowledge graph.

KGATE has an extensive data preprocessing toolbox, mainly focused on controlling evaluation biases known as data leakage. Data leakage is described in machine learning as the inclusion in the training dataset of information that should be exclusive to the test dataset, resulting in skewed and inflated performances that do not reflect the real capabilities of a model. While this bias can have a major impact on the relevance of a model in real-life applications, it has been poorly explored in KGE. This issue was first identified in Akrami et al. (2020) and explored more in depth in Brière et al. (2026) (Supplementary materials section 2).

## 2.2 Model Layer

The model layer has four different blocks. First, the initializer manages the initial feature vectors for the nodes and edges. Currently, KGATE provides three types of initialization: A random initialization using a Xavier uniform function for each node and edge type. Initialization with user-supplied features such as text embeddings. Using a Node2Vec (Grover & Leskovec, 2016) model to generate topology-aware initial feature vectors.

![](images/d6b119a772957610b3dfdcb26028ab5935fde37053f5127e09fc99d221245640.jpg)  
Figure 1: KGATE building blocks. The numbers given for each building block are subject to change as the library evolves. The modular architecture allows for additional blocks to be included by users or future updates.

During training, the encoder learns to project these features into a low-dimensional latent space using the graph structure. Encoders are graph neural networks able to aggregate information from heterogeneous features. KGATE implements 3 encoders from PyTorch Geometric: GATv2 (Brody et al., 2021), RGCN (Schlichtkrull et al., 2018) and GraphSAGE (Hamilton et al., 2017). It is also possible to use no encoder at all, in which case the initial feature vector output of the initializer is directly forwarded as the decoder’s input.

After the encoder’s forward pass, the decoder attempts to reconstruct the original graph from the latent space by computing a score for each triplet according to its own scoring function. KGATE implements 11 decoders from the three main families of KGE models:

• Translational models: also called geometric KGE models, they score the triplets so that the translation of the head node according to the relation is close to the tail node. KGATE implements the following translational models: TransE (Bordes et al., 2013), TransH (Wang et al., 2014), TransR (Lin et al., 2015), TransD (Ji et al., 2015), TorusE (Ebisu & Ichise, 2018) and RotatE (Sun et al., 2018).

• Bilinear models: they use matrix factorization as the score function. KGATE implements the following bilinear models: RESCAL (Nickel et al., 2011), DistMult (Yang et al., 2014) and ComplEx (Trouillon et al., 2016).

• Convolutional models: they are deep learning algorithms using convolutional neural networks as the score function. KGATE implements ConvKB (Nguyen et al., 2018) and ConvE (Dettmers et al., 2018).

By default, each decoder’s configuration is identical to its original paper’s implementation, but users can freely override these defaults to explore the hyperparameter space.

In addition, KGATE provides multiple loss functions (margin ranking, binary cross-entropy and logistic losses), embedding regularizers and normalizers. Finally, KGATE supports two prediction methods: the regular rank-based prediction method or the set retrieval technique introduced in Li et al. (2024) a, which returns a set of true triplets.

## 2.3 Hyperparameter Layer

This layer holds all the configurable aspects of the training. KGATE can use any optimizer and learning rate scheduler provided by PyTorch. This layer encapsulates all training utilities, such as the embedding normalizer and the regularizer which are respectively applied before and after every gradient step, and the negative sampler. Indeed, KGE training relies on the generation of false (negative) triplets that the decoder should score lower than true triplets. KGATE provides four different negative samplers: Positional, Bernoulli, Uniform and Mixed. The choice of negative triplet generation method and the complexity of these false triplets is an essential hyperparameter.

## 2.4 Task Layer

A KGATE model is trained to perform one of three tasks: Node/Edge Classification, Triplet Classification or Link Prediction. Depending on the task, a range of metrics are available to evaluate the model’s performances. Link prediction metrics are: mean reciprocal rank (MRR), mean rank (MR), hit@k, median rank, MRR@k, score gap and relative rank; Classification tasks metrics consist of accuracy, precision, recall, specificity, F1, false negative and false positive rates. Helper objects allow for an easy extension of available metrics.

## 2.5 Modularity and Extensibility

While KGATE was created to lower the expertise threshold required to perform KGE analyses, it is also a suitable tool for experts looking for a reproducible autoencoder framework. Each block of a KGATE workflow can be customized or replaced entirely, and additional building blocks can be inserted. For instance, new tasks and metrics can be added directly to the main KGATE structure. Moreover, KG embeddings can be exported to be used externally.

## 3 Benchmarking KGATE in the KGE library ecosystem

We benchmarked the technical performance of KGATE against six other KGE libraries: PyTorch Geometric (Fey et al., 2025), PyKEEN (Ali et al., 2020), TorchKGE (Boschin, 2020), Ampligraph (Costabello et al., 2024), DGL-KE (Zheng et al., 2020) and LibKGE (Broscheit et al., 2020), on two standard KG datasets, FB15k-237 and WN18RR. We compared data loading, mean training epoch, evaluation durations, and evaluation metrics (supplementary materials section 3). On average, KGATE’s speed is comparable to the best-performing libraries and is orders of magnitude faster than PyKEEN, which is the most similar library in terms of utilities.

## 4 Conclusion

KGATE provides utilities absent from common KGE libraries, including a complete autoencoder architecture and a built-in data leakage control procedure, gathered in a fully modular structure. KGATE is released in version 1.0 with different extensions planned to integrate explainability and temporal KGE modules.

## 5 Code Availability

The code of KGATE is available at https://github.com/BAUDOTlab/KGATE

The code of the benchmark is available at https://github.com/BAUDOTlab/KGATE\_benchmark

## Acknowledgments

This work was supported by members of Servier R&D and in particular Philippe Moingeon, Nicolas Lévy and Céline Lefebvre at Servier.

## 6 Fundings

This work was primarily funded by Servier.

The aims of this study contribute to the ERDERA project, which has received funding from the European Union’s Horizon Europe research and innovation program under grant agreement N°101156595.

The project leading to this publication has received funding from the Excellence Initiative of Aix-Marseille Université - A\*Midex, a French “Investissements d’Avenir programme” AMX-21-IET-017.

## References

Akrami, F., Saeef, M. S., Zhang, Q., Hu, W., & Li, C. (2020). Realistic Re-evaluation of Knowledge Graph Completion Methods: An Experimental Study [Conference Name: SIGMOD/PODS ’20: International Conference on Management of Data ISBN: 9781450367356 Place: Portland OR USA Publisher: ACM]. Proceedings ofthe 2020 ACM SIGMOD International Conference on Management ofData, 1995–2010. https://doi.org/10.1145/ 3318464.3380599 [TLDR] This paper is the first systematic study with the main objective of assessing the true effectiveness of embedding models when the unrealistic triples are removed, and results show their poor accuracy renders link prediction a task without truly effective automated solution.

Ali, M., Berrendorf, M., Hoyt, C. T., Vermue, L., Sharifzadeh, S., Tresp, V., & Lehmann, J. (2020). PyKEEN 1.0: A Python Library for Training and Evaluating Knowledge Graph Embeddings. ArXiv. Retrieved October 2, 2026, from https://www.semanticscholar.org/paper/PyKEEN-1.0%3A-A-Python-Library-for-Training-and-Graph-Ali-Berrendorf/7c008e9f16a9457a630eae049a4bebdec9f608db [TLDR] PyKEEN 1.0 is re-designed and re-implemented, one of the first KGE libraries, in a community effort, and through the integration of Optuna extensive hyper-parameter optimization (HPO) functionalities are provided.

Bordes, A., Usunier, N., García-Durán, A., Weston, J., & Yakhnenko, O. (2013). Translating Embeddings for Modeling Multi-relational Data. Retrieved October 6, 2023, from https://www.semanticscholar.org/paper/Translating-Embeddings-for-Modeling-Data-Bordes-Usunier/2582ab7c70c9e7fcb84545944eba8f3a7f253248 [TLDR] TransE is proposed, a method which models relationships by interpreting them as translations operating on the low-dimensional embeddings of the entities, which proves to be powerful since extensive experiments show that TransE significantly outperforms state-of-the-art methods in link prediction on two knowledge bases.

Boschin, A. (2020). TorchKGE: Knowledge Graph Embedding in Python and PyTorch. ArXiv. Retrieved October 2, 2026, from https://www.semanticscholar.org/paper/TorchKGE%3A-Knowledge-Graph-Embedding-in-Python-and-Boschin/46b5198a535dfcaf1cc7d57d471ad9ec050e46cf [TLDR] TorchKGE is a Python module for knowledge graph (KG) embedding relying solely on PyTorch that features a KG data structure, simple model interfaces and modules for negative sampling and model evaluation.

Brière, G., Stosskopf, T., Loire, B., & Baudot, A. (2026). Benchmarking the impact of data leakage on the performance of knowledge graph embedding models for biomedical link prediction (J. Wren, Ed.). Bioinformatics, 42(8), btag608. https://doi.org/10.1093/bioinformatics/btag608 [TLDR] The impact of data leakage is assessed on KGE-based link prediction across three biomedical knowledge graphs, using decoder-only and GNN-based models, and current benchmarking practices may overestimate how well KGE models generalize to practical applications.

Brody, S., Alon, U., & Yahav, E. (2021). How Attentive are Graph Attention Networks? ArXiv. Retrieved January 2, 2025, from https://www.semanticscholar.org/paper/How-Attentive-are-Graph-Attention-Networks-Brody-Alon/ab30672c8c5e4787f6a5985f26a8f281f0db2fb8 [TLDR] It is shown that GAT computes a very limited kind of attention: the ranking of the attention scores is unconditioned on the query node, and a simple fix is introduced by modifying the order of operations and proposed GATv2: a dynamic graph attention variant that is strictly more expressive than GAT.

Broscheit, S., Ruffinelli, D., Kochsiek, A., Betz, P., & Gemulla, R. (2020). LibKGE - A knowledge graph embedding library for reproducible research. Proceedings of the 2020 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, 165–174. https://doi.org/10.18653/v1/2020.emnlp-demos.22 [TLDR] LibKGE is an open-source PyTorch-based library for training, hyperparameter optimization, and evaluation of knowledge graph embedding models for link prediction that reaches competitive to state-of-the-art performance for many models with a modest amount of automatic hyperparameters tuning.

Chandak, P., Huang, K., & Zitnik, M. (2023). Building a knowledge graph to enable precision medicine. Scientific Data, 10(1), 67. https://doi.org/10.1038/s41597-023-01960-3 [TLDR] PrimeKG is presented, a multimodal knowledge graph for precision medicine analyses that contains an abundance of ‘indications’, ‘contradictions’, and ‘off-label use’ drug-disease edges that lack in other knowledge graphs and can support AI analyses of how drugs affect disease-associated networks.

Costabello, L., Alberto, sumitpai, Janik, A., Tabacof, P., McGrath, R., Van, C. L., ACMCMC, Clauss, C., Alto, A., Tekin, D., McCarthy, N., Vandenbussche, P.-Y., & Aayam. (2024). Accenture/AmpliGraph: AmpliGraph 2.0.1 [Publisher: Zenodo]. https://doi.org/10.5281/zenodo.7707964

Dettmers, T., Minervini, P., Stenetorp, P., & Riedel, S. (2018). Convolutional 2D Knowledge Graph Embeddings [ISSN: 2374-3468, 2159-5399 Issue: 1 Journal Abbreviation: AAAI]. Proceedings ofthe AAAI Conference on Artificial Intelligence, 32. https://doi.org/10.1609/aaai.v32i1.11573

[TLDR] ConvE, a multi-layer convolutional network model for link prediction, is introduced, and it is found that ConvE achieves state-of-the-art Mean Reciprocal Rank across all datasets.

Ebisu, T., & Ichise, R. (2018). TorusE: Knowledge Graph Embedding on a Lie Group [ISSN: 2374-3468, 2159-5399 Issue: 1 Journal Abbreviation: AAAI]. Proceedings of the AAAI Conference on Artificial Intelligence, 32. https://doi.org/10.1609/aaai.v32i1.11538 [TLDR] A novel embedding model, TorusE, is proposed that outperforms other state-of-the-art approaches such as TransE, DistMult and ComplEx on a standard link prediction task and is scalable to large-size knowledge graphs and is faster than the original TransE.

Fey, M., Sunil, J., Nitta, A., Puri, R., Shah, M., Stojanovic, B., Bendias, R., Barghi, A., Kocijan, V., Zhang, Z., He, X.,ˇ Lenssen, J. E., & Leskovec, J. (2025). PyG 2.0: Scalable Learning on Real World Graphs [Publisher: arXiv Version Number: 2]. https://doi.org/10.48550/ARXIV.2507.16991 [TLDR] This paper presents Pyg 2.0 (and its subsequent minor versions), a comprehensive update that introduces substantial improvements in scalability and real-world application capabilities, including support for heterogeneous and temporal graphs, scalable feature/graph stores, and various optimizations.

Hamilton, W. L., Ying, Z., & Leskovec, J. (2017). Inductive Representation Learning on Large Graphs. Retrieved June 11, 2024, from https://www.semanticscholar.org/paper/Inductive-Representation-Learning-on-Large-Graphs-Hamilton-Ying/6b7d6e6416343b2a122f8416e69059ce919026ef

Ji, G., He, S., Xu, L., Liu, K., & Zhao, J. (2015). Knowledge Graph Embedding via Dynamic Mapping Matrix. Proceedings of the 53rd Annual Meeting of the Association for Computational Linguistics and the 7th International Joint Conference on Natural Language Processing (Volume 1: Long Papers), 687–696. https: //doi.org/10.3115/v1/P15-1067 [TLDR] A more fine-grained model named TransD, which is an improvement of TransR/CTransR, which not only considers the diversity of relations, but also entities, which makes it can be applied on large scale graphs.

Li, Z., Ao, Y., & He, J. (2024). SpherE: Expressive and interpretable knowledge graph embedding for set retrieval [Conference Name: SIGIR 2024: The 47th International ACM SIGIR Conference on Research and Development in Information Retrieval ISBN: 9798400704314 Place: Washington DC USA Publisher: ACM]. Proceedings of the 47th International ACM SIGIR Conference on Research and Development in Information Retrieval, 2629–2634. https://doi.org/10.1145/3626772.3657910

Lin, Y., Liu, Z., Sun, M., Liu, Y., & Zhu, X. (2015). Learning entity and relation embeddings for knowledge graph completion [ISSN: 2374-3468, 2159-5399 Issue: 1 Journal Abbreviation: AAAI]. Proceedings ofthe AAAI Conference on Artificial Intelligence, 29. https://doi.org/10.1609/aaai.v29i1.9491

Nguyen, D. Q., Nguyen, T. D., Nguyen, D. Q., & Phung, D. (2018). A Novel Embedding Model for Knowledge Base Completion Based on Convolutional Neural Network. Proceedings ofthe 2018 Conference ofthe North American Chapter ofthe Associationfor Computational Linguistics: Human Language Technologies, Volume 2 (Short Papers), 327–333. https://doi.org/10.18653/v1/N18-2053 [TLDR] The model ConvKB advances state-of-the-art models by employing a convolutional neural network, so that it can capture global relationships and transitional characteristics between entities and relations in knowledge bases.

Nickel, M., Tresp, V., & Kriegel, H. (2011). A Three-Way Model for Collective Learning on Multi-Relational Data. Retrieved June 11, 2024, from https://www.semanticscholar.org/paper/A-Three-Way-Model-for-Collective-Learning-on-Data-Nickel-Tresp/f6764d853a14b0c34df1d2283e76277aead40fde [TLDR] This work presents a novel approach to relational learning based on the factorization of a three-way tensor that is able to perform collective learning via the latent components of the model and provide an efficient algorithm to compute the factorizations.

Prieto, N. V., & Garrido-Merchán, E. C. (2026). Default Machine Learning Hyperparameters Do Not Provide Informative Initialization for Bayesian Optimization [Publisher: arXiv Version Number: 1]. https://doi.org/10.48550 ARXIV.2602.08774

for optimization, and recommend that practitioners treat hyperparameter tuning as an integral part of model development and favor principled, data-driven search strategies over heuristic reliance on library defaults.

Schlichtkrull, M., Kipf, T. N., Bloem, P., Van Den Berg, R., Titov, I., & Welling, M. (2018). Modeling Relational Data with Graph Convolutional Networks [Book Title: The Semantic Web Series Title: Lecture Notes in Computer Science]. In A. Gangemi, R. Navigli, M.-E. Vidal, P. Hitzler, R. Troncy, L. Hollink, A. Tordai, & M. Alam (Eds.). Springer International Publishing. https://doi.org/10.1007/978-3-319-93417-4\_38 [TLDR] It is shown that factorization models for link prediction such as DistMult can be significantly improved through the use of an R-GCN encoder model to accumulate evidence over multiple inference steps in the graph, demonstrating a large improvement of 29.8% on FB15k-237 over a decoder-only baseline.

Sun, Z., Deng, Z., Nie, J.-Y., & Tang, J. (2018). RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space. ArXiv. Retrieved June 11, 2024, from https : / / www . semanticscholar . org / paper / RotatE % 3A - Knowledge - Graph - Embedding - by - Relational - in - Sun - Deng / 8f096071a09701012c9c279aee2a88143a295935

[TLDR] Experimental results show that the proposed RotatE model is not only scalable, but also able to infer and model various relation patterns and significantly outperform existing state-of-the-art models for link prediction.

Toutanova, K., & Chen, D. (2015). Observed versus latent features for knowledge base and text inference. Proceedings ofthe 3rd Workshop on Continuous Vector Space Models and their Compositionality, 57–66. https://doi.org/10. 18653/v1/W15-4007

[TLDR] It is shown that the observed features model is most effective at capturing the information present for entity pairs with textual relations, and a combination of the two combines the strengths of both model types.

Trouillon, T., Welbl, J., Riedel, S., Gaussier, É., & Bouchard, G. (2016). Complex Embeddings for Simple Link Prediction. ArXiv. Retrieved June 11, 2024, from https : / / www. semanticscholar. org / paper / Complex - Embeddings-for-Simple-Link-Prediction-Trouillon-Welbl/2218e2e1df2c3adfb70e0def2e326a39928aacfc [TLDR] This work makes use of complex valued embeddings to solve the link prediction problem through latent factorization, and uses the Hermitian dot product, the complex counterpart of the standard dot product between real vectors.

Wang, Z., Zhang, J., Feng, J., & Chen, Z. (2014). Knowledge Graph Embedding by Translating on Hyperplanes [ISSN: 2374-3468, 2159-5399 Issue: 1 Journal Abbreviation: AAAI]. Proceedings ofthe AAAI Conference on Artificial Intelligence, 28. https://doi.org/10.1609/aaai.v28i1.8870

[TLDR] This paper proposes TransH which models a relation as a hyperplane together with a translation operation on it and can well preserve the above mapping properties of relations with almost the same model complexity of TransE.

Yang, B., Yih, W.-t., He, X., Gao, J., & Deng, L. (2014). Embedding entities and relations for learning and inference in knowledge bases. Retrieved June 11, 2024, from https://www.semanticscholar.org/paper/Embedding-Entitiesand-Relations-for-Learning-and-Yang-Yih/86412306b777ee35aba71d4795b02915cb8a04c3

Zheng, D., Song, X., Ma, C., Tan, Z., Ye, Z., Dong, J., Xiong, H., Zhang, Z., & Karypis, G. (2020). DGL-KE: Training Knowledge Graph Embeddings at Scale [Conference Name: SIGIR ’20: The 43rd International ACM SIGIR conference on research and development in Information Retrieval ISBN: 9781450380164 Place: Virtual Event China Publisher: ACM]. Proceedings of the 43rd International ACM SIGIR Conference on Research and Development in Information Retrieval, 739–748. https://doi.org/10.1145/3397271.3401172

[TLDR] DGL-KE introduces various novel optimizations that accelerate training on knowledge graphs with millions of nodes and billions of edges using multi-processing, multi-GPU, and distributed parallelism to increase data locality, reduce communication overhead, overlap computations with memory accesses, and achieve high operation efficiency.

# SUPPLEMENTARY MATERIALS - KGATE : A KNOWLEDGE GRAPH EMBEDDING TRAINING ENVIRONMENT

## Supplementary Section S1: Main differences between KGE libraries

Knowledge Graph Embedding (KGE) models, just like any machine learning models, rely on hyperparameters to guide their training. In order to train a reproducible model, these hyperparameters must be kept identical between training sessions. In theory, identical model hyperparameters should output the same results in the same conditions, regardless of the library running the code. However, the original publications of KGE models are sometimes ambiguous about these hyperparameters, or do not provide the source code. In turn, developers trying to implement these models are forced to interpret the author’s intent and make arbitrary decisions.

In addition, some KGE libraries present differences in model implementation. For example, in the original DistMult publication (Yang et al. 2014), the objective formula provided in the manuscript does not indicate embedding normalization, but reads “The entity vectors are renormalized to have unit length after each gradient step [...]. For the relation parameters, we used standard L2 regularization.” and the code is not provided by the authors. As a consequence, the DisMult model has been implemented differently in different libraries:

• TorchKGE performs L2 normalization on the entity vectors at the beginning of each epoch.

• PyTorch Geometric, Ampligraph, and DGL-KE, do not normalize nor regularize anything.

• PyKEEN performs L1 normalization on entity embeddings at the beginning of each epoch and regularizes the edge parameters after each gradient step.

These design choices regarding hyperparameters and implementation are often hardcoded, which makes it impossible for the end user to alter without editing the library’s code, and breaks cross-library reproducibility. The table below illustrates these variations on two standard KGE datasets with the following hyperparameters:

• Initialization: Random

• Encoder: None

• Decoder: TransE

• Embedding dimension: 256

• Loss: Margin ranking loss

• Margin: 0.5

• Negative Sampler (when tunable): Bernoulli

• Number of negatives per positive triplet: 5

• Batch size: 4096

• Epochs: 100

• Learning Rate: 0.001

• Optimizer: Adam

• Random seed: 42

![](images/3c100a3e9357da7d7f67f16a4628923f895b994d2adf2d20bb8bf3ffa20aa50f.jpg)

B  
![](images/f12ab8d76f12e187a2c7a29c368f145db00308b843e7c1bac0a70e452d30d6be.jpg)

C  
![](images/3a7a796aeea9ec8c130b292c249d8017579443f1ae83d4791714353d0c0fb16f.jpg)  
Supplementary Figure S1: Evolution of the embeddings of one gene node across 100 epochs using the training loops of five different libraries on the PrimeKG dataset. The embeddings were initialized once before being fed to each library as initial features to the nodes. Each run was performed on a single-threaded CPU to ensure determinism. A) Evolution of the two first components of a PCA performed on the embeddings across each epoch; B) Evolution of the L2 normalization of the embedding; C) Cosine similarity of the embedding at each epoch compared to the initial embedding.

<table><tr><td rowspan=1 colspan=1>Library</td><td rowspan=1 colspan=1>KGATE</td><td rowspan=1 colspan=1>TorchKGE</td><td rowspan=1 colspan=1>PyTorch Geometric</td><td rowspan=1 colspan=1>DGL-KE</td><td rowspan=1 colspan=1>Ampligraph</td><td rowspan=1 colspan=1>PyKEEN</td></tr><tr><td rowspan=1 colspan=1>KGATE</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1>0.169</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>0.284</td></tr><tr><td rowspan=1 colspan=1>TorchKGE</td><td rowspan=1 colspan=1>0.290</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.260</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.197</td></tr><tr><td rowspan=1 colspan=1>PyTorch Geometric</td><td rowspan=1 colspan=1>0.0169</td><td rowspan=1 colspan=1>0.260</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.114</td></tr><tr><td rowspan=1 colspan=1>DGL-KE</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>0.064</td></tr><tr><td rowspan=1 colspan=1>Ampligraph</td><td rowspan=1 colspan=1>0.107</td><td rowspan=1 colspan=1>0.077</td><td rowspan=1 colspan=1>0.058</td><td rowspan=1 colspan=1>0.053</td><td rowspan=1 colspan=1>1</td><td rowspan=1 colspan=1>0.070</td></tr><tr><td rowspan=1 colspan=1>PyKEEN</td><td rowspan=1 colspan=1>0.284</td><td rowspan=1 colspan=1>0.197</td><td rowspan=1 colspan=1>0.114</td><td rowspan=1 colspan=1>0.064</td><td rowspan=1 colspan=1>0.070</td><td rowspan=1 colspan=1>1</td></tr></table>

Supplementary Table S1: Mean Jaccard index of the similarity of the top 10 predictions for 10 different gene nodes sampled at random using the link prediction query “(gene, protein-protein, ?)”. Predictions are made once 100 epochs have elapsed, using the same model and the same hyperparameters when it is possible to tune them.

While all libraries are deterministic given the same seed and hyperparameters, the differences between their implementations and hyperparameters (e.g., normalization, loss and scoring functions) lead to a different output, which affects the reproducibility between libraries. These differences lead to very different MRRs and different predictions. This means that results may be driven more by the choice of the library rather than by model capabilities. From a reproducibility standpoint, this implies that model comparisons and benchmarks must be done using the exact same library. When that is not the case, a difference in results cannot be directly attributed to the underlying data or methodological approach.

To enable a fair comparison between libraries, KGATE provides cross-library comparison through configuration presets able to reproduce the results given by a different library.

## Supplementary Section S2: Details on the Data Leakage control procedure

When the train/test split of a knowledge graph is not properly controlled, the training set may contain triplets that are redundant with test triplets. Such redundancy biases the evaluation by artificially inflating the measured performance of KGE models, an issue known as data leakage (Akrami et al. (2020) and Brière et al. (2026)). To address it, KGATE implements a data leakage control procedure that detects redundant relations in the KG and removes the triplets inducing data leakage from the training set

Based on prior studies by Akrami et al. and Brière et al., the data leakage procedure implemented in KGATE is able to detect three types of relations that may cause data leakage issues in a KG: near-duplicate relations, near-reverse-duplicate relations, and Cartesian product relations.

In a biomedical KG, two relations types r1 and r2 may convey similar meanings, causing a certain proportion of their (head, tail) pairs to overlap. For example, “(protein)-binds to-(protein)” and “(protein)-interacts with-(protein)” are expected to link many of the same protein pairs, since binding is a form of interaction. Given two relations r1 and r2, when the overlap between their (head, tail) pairs in the untreated KG exceeds a user-defined threshold (often set to 80%, the default value in KGATE), they are considered as near-duplicate relations. To account for asymmetric redundancies, the overlap is computed in both directions (relative to r1 and relative to r2), and the user can specify a distinct threshold for each direction.

Similarly, reverse relations, such as “(drug)-inhibits-(protein)” and “(protein)-is inhibited by-(drug)” can be considered as near-reverse-duplicates as they convey reverse semantic meanings. Near-reverse-duplicate relations are detected with the same procedure to near-duplicate relations, but comparing the (head, tail) pairs of r1 with the (tail, head) pairs of r2.

The third type of redundant relations that can be detected with KGATE’s data leakage procedure are Cartesian product relations. Let us consider the set of unique heads and the set of unique tails connected by a relation r. The maximum number of distinct triplets that r could form between its set of heads and its set of tails is the product of the sizes of these two sets, corresponding to every head being connected to every tail. A relation r is considered a Cartesian product relation when the number of triplets it actually forms in the full KG exceeds a user-defined threshold of this maximum. In a biomedical KG, an example of such Cartesian product relation could be “(gene)-expressed in-(cell type)” for ubiquitously expressed genes (i.e. housekeeping genes).

After detecting near-duplicate relations, near-reverse-duplicate relations, and Cartesian product relations in the KG, KGATE performs the train/test split and proceeds to remove any leaked triplet from the training set. Specifically, for near-duplicate and near-reverse-duplicate relations, for any triplet placed in the validation or test set, its redundant counterparts are excluded from the training set. For Cartesian product relations, KGATE ensures that all triplets involving a given head (or tail) with the Cartesian relation r are placed in the same split.

While most KGE models handle exclusively directed relations, biomedical KGs often contain undirected relations (e.g., protein-protein interactions). KGATE can convert these undirected relations into directed ones while ensuring that this conversion does not introduce near-reverse-duplicate relations and data leakage in consequence. For any undirected relation r, KGATE creates a reverse counterpart r\_rev and adds, for every triple (a, r, b), the corresponding triple (b, r\_rev, a). Then, for every test triple (a, r, b), the reverse triple (b, r\_rev, a) is removed from the training set, and vice versa. Importantly, all triplets are kept in KGATE’s ground-truth. The ground-truth triplets are removed in the filtered evaluation, so that models are not penalized for ranking highly triplets that represent true facts but were removed from the training graph. For instance, since protein-protein interactions are undirected, a test triple (a, ppi, b) implies that (b, ppi, a), (a, ppi\_rev, b) and (b, ppi\_rev, a) are also true, and all of them are kept in the ground truth, even though they are removed from the training set.

Supplementary Section S3: Benchmarking KGATE and other libraries
<table><tr><td>Library</td><td>Encoders</td><td>Decoders</td><td>KGE Autoencoder</td><td>Negative samplers</td><td>Data Leakage Control</td><td>Actively Maintained</td></tr><tr><td>KGATE (ours)</td><td>3</td><td>10</td><td>Yes</td><td>4</td><td>Advanced</td><td>Yes</td></tr><tr><td>PyTorch Geometric</td><td>21</td><td>4</td><td>No</td><td>1</td><td>Basic</td><td>Yes</td></tr><tr><td>TorchKGE</td><td>0</td><td>11</td><td>No</td><td>3</td><td>No</td><td>No</td></tr><tr><td>PyKEEN</td><td>3</td><td>34</td><td>One</td><td>3</td><td>Basic</td><td>Yes</td></tr><tr><td>LibKGE</td><td>0</td><td>11</td><td>No</td><td>2</td><td>No</td><td>No</td></tr><tr><td>Ampligraph</td><td>0</td><td>4</td><td>No</td><td>1</td><td>No</td><td>No</td></tr></table>

Supplementary Table S2: Comparisons of KGATE and other KGE libraries utilities. The advanced data leakage of KGATE corresponds to the full procedure described in Brière et al, while Basic data leakage is a simple control of full duplicate edges. LibKGE appears to provide the necessary structure for an autoencoder; however no encoder is implemented. PyKEEN does implement a kind of autoencoder, with the implementation of specific message passing encoders like RGCN that take into account the different edge types. However, in PyKEEN, the generic message passing intended for user-driven customization treats the dataset as a homogeneous graph instead of a KG, failing to represent the complexity of the data.

<table><tr><td rowspan="13">FB15k-237</td><td>Library</td><td>Data loading (s)</td><td>Mean epoch (ms)</td><td>Test (s)</td><td>MRR</td><td rowspan="13">WN18RK</td></tr><tr><td>KGATE (ours)</td><td>3.09</td><td>0.92</td><td>4.57</td><td>0.2340</td></tr><tr><td>TorchKGE</td><td>3.92</td><td>0.21</td><td>5.51</td><td>0.2419</td></tr><tr><td>PyTorch Geometric</td><td>0.10</td><td>0.21</td><td>9.19</td><td>0.2381</td></tr><tr><td>PyKEEN</td><td>NA</td><td>7.11</td><td>3.89</td><td>0.0791</td></tr><tr><td>Ampligraph</td><td>1.73</td><td>0.22</td><td>44.01</td><td>0.1547</td></tr><tr><td>LibKGE</td><td>0.50</td><td>0.87</td><td>7.88</td><td>0.0592</td></tr><tr><td>DGL-KE</td><td>0.49</td><td>0.46</td><td>5.29</td><td>0.0870</td></tr><tr><td>KGE Library</td><td>Data loading (s)</td><td>Mean epoch (ms)</td><td>Test (s)</td><td>MRR</td></tr><tr><td>KGATE (ours)</td><td>1.01</td><td>0.44</td><td>2.93</td><td>0.0143</td></tr><tr><td>TorchKGE</td><td></td><td></td><td>2.29</td><td></td></tr><tr><td>PyTorch Geometric</td><td>1.30</td><td>0.11</td><td></td><td>0.0060</td></tr><tr><td></td><td>0.05</td><td>0.18</td><td>0.14</td><td>0.0001</td></tr><tr><td>PyKEEN</td><td>NA</td><td>2.41</td><td>2.15</td><td></td></tr><tr><td>Ampligraph</td><td>0.18</td><td>0.15</td><td>6.54</td><td>0.0967</td></tr><tr><td>LibKGE</td><td>0.46</td><td>0.42</td><td>4.15</td><td>0.0018</td></tr><tr><td>DGL-KE</td><td>NA</td><td>NA</td><td>NA</td><td>NA</td></tr></table>

Supplementary Table S3: Training duration and reported metric for different libraries trained with the TransE decoder on the FB15k-237 and WN18RR KG datasets using the same set of hyperparameters. For KGATE, the data loading duration was measured without using KGATE’s preprocessing pipeline. PyKEEN’s data loading duration is set to NA because it was not possible to separate it from the training step, however it was negligible compared to the duration of a training epoch. We did not manage to run DGL-KE on the WN18RR dataset.

KGATE has a similar or slightly longer training time than other libraries, but is 4 to 5 times faster than PyKEEN, which is the only library that provides a similar range of utilities. Except for PyKEEN, the duration differences in overall training time over a full training of 100 epochs are negligible.