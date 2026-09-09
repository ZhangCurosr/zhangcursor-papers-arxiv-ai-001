# OntologyBench: Can Dense Retrieval Satisfy Structured Biomedical Constraints?

Xiao Yu Cindy Zhang<sup>1,2</sup> Wyeth W. Wasserman<sup>2,3</sup> Jian Zhu<sup>4</sup>

<sup>1</sup>Graduate Program in Bioinformatics, The University of British Columbia <sup>2</sup>Centre for Molecular Medicine and Therapeutics, BC Children’s Hospital Research Institute <sup>3</sup>Department of Medical Genetics, The University of British Columbia <sup>4</sup>Department of Linguistics, The University of British Columbia Vancouver, BC, Canada

czhang@cmmt.ubc.ca wyeth@cmmt.ubc.ca jian.zhu@ubc.ca

## Abstract

We introduce OntologyBench, a tiered biomedical retrieval benchmark comprising 471,854 training and 125,744 evaluation query–document relevance pairs across concept grounding, relational retrieval, and compositional phenotype-based retrieval.

Although these tasks can be tractable using ontology-aware reference methods, across task tiers, embedding performance is generally lower on relational and compositional tasks than on concept-grounding tasks. Fine-tuning on ontology-derived supervision improves performance on several relational and compositional tasks, whereas the evaluated reranking and LLM-based candidate-scoring methods provide little or no end-to-end improvement. Errors frequently reflect diseases matching only subsets of the phenotype evidence.

These findings indicate that the evaluated embedding and reranking configurations do not reliably recover the compatibility encoded by the selected ontology relations and phenotype combinations and motivate retrieval systems that better integrate learned representations with structured biomedical knowledge.

## 1 Introduction

Dense retrieval models enable retrieval directly over natural language by mapping semantically related queries and documents into a shared embedding space. This supports retrieval beyond exact lexical overlap and reduces dependence on explicitly engineered symbolic methods, such as ontology-based semantic similarity scoring and graph traversal. This flexibility has made embedding retrieval attractive for biomedical and clinical applications, where concepts are often expressed using heterogeneous, incomplete, or non-canonical language.

However, many retrieval problems require satisfying multiple interdependent constraints rather than matching individual features independently. (Agarwal et al., 2026) Existing benchmarks primarily evaluate retrieval based on unstructured semantic similarity and provide limited insight into whether models can recover compatibility across multiple structured inputs. (Muennighoff et al., 2023)

To address this gap, we introduce Ontology-Bench, a retrieval benchmark constructed from expert-curated biomedical ontologies. The benchmark spans concept grounding, relational retrieval, and compositional phenotype-based retrieval. These task families provide complementary views of retrieval behaviour under different forms of ontology-encoded compatibility. Because they also differ in candidate spaces, relevance structures, and query ambiguity, cross-tier comparisons are interpreted descriptively rather than as isolated estimates of constraint complexity.

Within this framework, the fixed Tier 3 phenotype-triplet-to-disease task reveals a persistent limitation. The evaluated embedding models frequently rank diseases that match only a subset of the phenotype evidence, and their performance remains substantially below that of an ontologyaware baseline. The tested rerankers provide little or no end-to-end improvement across the complete query set. These findings indicate that the evaluated retrieval systems do not reliably recover the compatibility encoded by joint phenotype sets, although candidate omission and downstream ranking errors remain alternative sources of failure.

This work makes three contributions. First, we introduce a retrieval benchmark spanning concept grounding, ontology-relation retrieval, and phenotype-set retrieval to evaluate whether embedding models recover compatibility encoded by curated biomedical ontologies. Second, we characterize retrieval behaviour across grounding, relational, and compositional tasks, architectures, and model scales, identifying a persistent gap relative to an ontology-aware baseline on the fixed phenotypetriplet-to-disease task. Third, through two-stage evaluation, we distinguish candidate availability from the end-to-end performance of the reranked system. Across the complete query set, the evaluated rerankers provide little or no improvement over the first-stage retriever.

## 2 Related Work

Dense Retrieval Architectures. Dense retrieval represents queries and documents as vectors in a shared embedding space, enabling efficient largescale search (Reimers and Gurevych, 2019). Biencoder models support scalable retrieval via independent encoding, while late-interaction models such as ColBERT preserve token-level representations to improve fine-grained matching (Khattab and Zaharia, 2020; Chaffin and Sourty, 2025). Recent pretrained models further improve embedding quality for retrieval tasks (Zhang et al., 2025).

Embedding Benchmarks. Large-scale benchmarks such as BEIR (Thakur et al., 2021) and MTEB (Muennighoff et al., 2023) standardize evaluation across diverse retrieval and embedding tasks, primarily measuring semantic similarity in unstructured text. Biomedical retrieval datasets such as NFCorpus (Boteva et al., 2016) and TREC-COVID (Voorhees et al., 2021) similarly evaluate retrieval between clinical documents. More recent benchmarks explore retrieval requiring aggregation across multiple documents or reasoning steps, including multi-hop question answering (Yang et al., 2018) and reasoning-intensive retrieval benchmarks (Xiao et al., 2024; Su et al., 2024; Li et al., 2025). However, these benchmarks operate over unstructured text and do not isolate compositional compatibility.

Biomedical retrieval has long relied on symbolic methods that model relationships between diseases, phenotypes, and genes. Informationcontent–based semantic similarity methods, including Resnik similarity and phenotype set similarity approaches used in systems such as Phenomizer, leverage hierarchical biomedical structure for retrieval. (Resnik, 1999; Köhler et al., 2009; Hoehndorf et al., 2015; Vasilevsky et al., 2025) Graph-based retrieval methods similarly exploit relational paths between biomedical entities. Unlike embedding-based retrieval, these approaches enforce compatibility through curated symbolic structure rather than learning it implicitly through representation similarity, making them less flexible for heterogeneous natural-language inputs such as free-text clinical documentation.

Recent work has also explored ontologygrounded biomedical embeddings designed to improve interpretability. QIME (Tang et al., 2026) constructs interpretable medical text embeddings by representing documents through ontologygrounded natural-language questions derived from biomedical concepts. While such approaches improve semantic interpretability and retrieval effectiveness, they primarily evaluate embedding quality through semantic similarity, clustering, and retrieval performance rather than through controlled evaluation of compositional compatibility. In contrast, OntologyBench focuses on whether retrieval systems can jointly satisfy multiple structured biomedical constraints under controlled retrieval settings.

OntologyBench evaluates whether embeddingbased retrieval systems can recover the compatibility constraints enforced by symbolic biomedical retrieval methods. While embedding retrieval supports flexible natural-language matching, symbolic biomedical systems directly enforce compatibility through curated relational structure. OntologyBench organizes retrieval tasks across concept grounding, relational retrieval, and phenotypeset retrieval, enabling comparison of retrieval behaviour under different forms of ontology-defined relevance.

## 3 OntologyBench Dataset and Tasks

## 3.1 Knowledge Sources.

OntologyBench is designed as an evaluation framework spanning concept grounding, ontologyrelation retrieval, and phenotype-set retrieval. The task families differ in candidate spaces, relevance structures, and query ambiguity; cross-tier comparisons are therefore interpreted descriptively. The unified setting evaluates held-out directional pairs and phenotype combinations within a shared ontology vocabulary rather than generalization to entirely unseen biomedical concepts or relations.

![](images/42ec6980e1289184c5132d2eedca36ce4c1373d739fbad6a048893a52cd86678.jpg)  
Figure 1: OntologyBench overview illustrating the retrieval tiers from grounding to compositional retrieval.

We construct a structured retrieval space grounded in curated biomedical ontologies linking clinical findings, diseases, and genetic causes. Disease concepts are drawn from the Monarch Disease Ontology (MONDO) (Vasilevsky et al., 2025), which integrates disease terminology and mappings across resources including OMIM, SNOMED CT, and ICD. OntologyBench includes 6,611 canonical MONDO disease concepts represented by their preferred labels and available synonyms.

Phenotype concepts are drawn from the Human Phenotype Ontology (HPO) (Gargano et al., 2024), a controlled vocabulary describing observable clinical features (phenotypes). The dataset contains 8,253 phenotype concepts, each with associated textual synonyms used as query inputs.

Gene entities are constructed from HUGO Gene Nomenclature Committee(HGNC) gene names (Seal et al., 2026) combined with National Center for Biotechnology Information (NCBI) Gene metadata (O’Leary et al., 2024), resulting in 2,345 gene entries.

Together, these ontologies provide a structured biomedical knowledge space linking phenotypes, diseases, and genes, enabling evaluation of retrieval systems under structured biomedical constraints.

## 3.2 Task Formulation

OntologyBench is organized into three tiers spanning concept grounding, ontology relations, and compositional retrieval (Figure 1). This organization supports comparison across concept grounding, ontology-relation retrieval, and phenotype-set retrieval. Because the task families differ in candidate spaces, relevance multiplicity, and query ambiguity, cross-tier score differences are descriptive rather than isolated effects of increasing constraint complexity. Examples of each tier can be found in Appendix A.

Tier 1: Concept Grounding Tier 1 evaluates whether models can map surface forms (aliases) of genes (Gene), phenotypes (Phen), and disorders (Dis) to their canonical ontology definitions (Def) through single-hop retrieval. This tier includes three grounding tasks: Gene→Def, Phen→Def, and Dis→Def.

Tier 2: Relational Retrieval. This tier evaluates whether models capture pairwise biomedical relationships between diseases, phenotypes, and genes.

Disease ↔ Phenotype (Dis↔Phen). These tasks evaluate complementary retrieval patterns over the same curated many-to-many disease– phenotype relation. Disease-to-phenotype retrieval identifies characteristic features for constructing or reviewing disease profiles, whereas phenotype-todisease retrieval identifies diseases compatible with an observed finding and is generally more ambiguous because phenotypes are shared across diseases. Their differing query semantics, candidate spaces, and relevance-set structures motivate evaluation in both directions.

Phenotype→Gene (Phen→Gene). Given a phenotype query, the model retrieves genes associated with diseases exhibiting that phenotype.

Disease→Gene (Dis→Gene). Given a disease query, the model retrieves genes known to be molecularly associated with that disorder.

Tier 3: PhenTriplet→Dis (compositional retrieval). In clinical settings, diagnosis is often based on incomplete and partially observed phenotypes. Tier 3 models this under a controlled phenotype-based retrieval setting, evaluating compositional retrieval from limited evidence. Each query consists of three phenotypes drawn from a single disease’s phenotype set, ensuring internal consistency. Some phenotype subsets may include non-specific features; this is intentional, as such cases introduce ambiguity and test whether models can correctly aggregate weak signals rather than rely on a single highly discriminative feature.

The choice of three phenotypes reflects a tradeoff between ambiguity and determinism: single phenotypes correspond to many candidate diseases, while larger sets (≥ 4 phenotypes) sharply reduce the candidate space, making retrieval increasingly deterministic. As shown in Appendix B, phenotype triplets yield a non-trivial regime (∼ 5 relevant diseases on average), requiring models to aggregate multiple constraints rather than rely on single-feature matching.

While this construction does not model the full diagnostic process, which often needs to incorporate temporal progression, demographics, disease prevalence and other contextual evidence, it intentionally isolates the compositional retrieval problem by controlling for these additional factors. This enables evaluation of whether retrieval models can integrate multiple phenotype constraints under ambiguity.

## 3.3 Train–Evaluation Partition and Split-Integrity Audit

The unified multi-tier corpus comprises 471,854 training and 125,744 evaluation query–document relevance pairs. Task-specific partitions were constructed at the level of canonical directed query– target identifiers, and an audit of the training artifacts found zero within-task overlap of these directed pairs between training and evaluation. Because the tasks share an ontology vocabulary, concepts, targets, and underlying associations may recur across task formulations.

<table><tr><td rowspan="2">Task</td><td colspan="2">Train</td><td colspan="2">Test</td></tr><tr><td>#Queries</td><td></td><td>AvgPos  | #Queries</td><td>AvgPos</td></tr><tr><td colspan="5">Tier 1 (Concept Grounding)</td></tr><tr><td>Gene→Def Phen→Def</td><td>1,914 17,763</td><td>1.00 1.00</td><td>431 4,581</td><td>1.00 1.00</td></tr><tr><td>Dis→Def</td><td>39,644</td><td>1.00</td><td>9,205</td><td>1.00</td></tr><tr><td colspan="5">Tier 2 (Relational Retrieval)</td></tr><tr><td>Dis→Phen</td><td>6,647</td><td>19.81</td><td>6,291</td><td>5.63</td></tr><tr><td>Phen→Dis</td><td>7,988</td><td>16.62 1.00</td><td>5,229 1,507</td><td>6.57</td></tr><tr><td>Phen→Gene Dis→Gene</td><td>6,960 6,623</td><td>17.58</td><td>6,504</td><td>1.00 5.31</td></tr><tr><td></td><td></td><td></td><td></td><td></td></tr><tr><td colspan="5">Tier 3 (Compositional Retrieval)</td></tr><tr><td>PhenTriplet→Dis</td><td>5,642</td><td>4.39</td><td>2,537</td><td>2.26</td></tr></table>

Table 1: OntologyBench dataset statistics across tiers. AvgPos denotes the average number of positive documents per query. Tier-3 queries had an average of 4.39 relevant diseases in training and 2.26 in evaluation.

The two directions of Tier 2 Dis↔Phen were partitioned as separate tasks. Consequently, a disease– phenotype association evaluated in one direction could occur in training in the inverse direction. Reciprocal-task performance therefore evaluates whether shared relational supervision supports both retrieval formulations, rather than discovery of previously unseen disease–phenotype associations.

Tier 3 held out exact phenotype-triplet–disease relevance pairs and same-task positive disease targets. Most evaluation triplet queries were also absent from training; however, 146 of 2,537 queries (5.8%) occurred in training with different disease targets, and constituent phenotype–disease associations could appear in the relational training tasks. OntologyBench therefore evaluates concept grounding, transductive relational transfer, and compositional retrieval of held-out phenotype-set– disease compatibilities within a shared ontology vocabulary. It does not evaluate inductive generalization to entirely unseen entities or relation types.

## 3.4 Dataset Statistics

Table 1 summarizes the benchmark statistics across all retrieval tasks. AvgPos denotes the average number of ground-truth relevant (positive) target documents associated with each query. Additional dataset statistics are provided in Appendix B.

Tier 1 tasks are one-to-one concept grounding (AvgPos = 1), while Tier 2 ontology association tasks exhibit substantial ambiguity due to manyto-many biomedical relationships (e.g., Dis→Phen and Phen→Dis average 19.8 and 16.6 positives per query). In contrast, Tier 3 compositional queries correspond to a smaller relevant set (4.39 diseases on average), reflecting the narrowing effect of combining multiple phenotype constraints. Sequence length statistics are reported in Appendix C.

## 4 Method

## 4.1 Supervision Settings

To study how ontology supervision shapes embedding representations, we compare two training regimes using the same task-specific split artifacts.

Concept Grounding Supervision. Models are trained only on alias–definition grounding pairs that map surface biomedical expressions to canonical concepts. No ontology relations or compositional signals are provided during training. Performance on higher-tier tasks, therefore, measures whether relational structure can emerge from concept grounding alone.

Unified Ontology Supervision. Models are trained jointly on all ontology-derived tasks, including grounding, entity relations (e.g., phenotype–disease, disease–gene), and compositional phenotype queries.

## 4.2 Two-Stage Retrieval Evaluation

Performance degradation under compositional retrieval may arise from either missing candidates during retrieval or incorrect downstream ranking among retrieved candidates. To distinguish candidate-availability failure from downstreamranking performance, we evaluated the two stages separately. Candidate availability was measured over the complete query set.

First, an initial embedding retriever generates the top-50 candidates from the full corpus. At least one relevant disease is present among these candidates for 82.1% of queries (Hit@50 = 0.821), establishing the maximum proportion of queries for which downstream reranking can succeed. Here, Hit@k denotes the proportion of queries for which at least one relevant candidate appears among the top-k results. Second, we evaluate whether rerankers can place at least one relevant disease near the top of this fixed candidate set. Late-interaction reranking uses GTE-ModernColBERT with ColBERTstyle MaxSim scoring, whereas LLM-based candidate scoring directly evaluates phenotype–disease compatibility for each retrieved candidate (Appendix F).

Candidate-generation metrics identify queries for which reranking could not recover a relevant disease because none was available. End-to-end metrics retain all queries and therefore characterize the combined two-stage system.

## 4.3 Experimental Setup

Model Setup. We evaluate lexical, dense retrieval, reranking, and generative models. As a lexical baseline, we use BM25. Dense retrieval models include general-purpose embedding models such as all-MiniLM-L6-v2 (Reimers and Gurevych, 2019), Qwen3 embeddings (0.6B and 4B) (Zhang et al., 2025), and OpenAI text-embedding-3-large<sup>1</sup>, as well as biomedical embedding models including BioLORD-0.1B (Remy et al., 2024), MedEmbed-0.1B (Balachandran, 2024), and SapBERT-0.1B (Liu et al., 2021).

For late-interaction reranking, we evaluate GTE-ModernColBERT (Chaffin and Sourty, 2025) using ColBERT-style MaxSim scoring (Khattab and Zaharia, 2020). For LLM-based candidate scoring reranking, we use Qwen3-4B-Thinking-2507, Qwen3-4B-Instruct-2507 (Team, 2025), Qwen3.6- 27B (Qwen Team, 2026), Gemma4-31B (Google DeepMind, 2026), and OpenAI GPT5.4-mini<sup>2</sup>.

Ontology-Aware Reference Methods. We include ontology-aware reference methods to estimate retrieval performance when curated biomedical structure is explicitly available. These methods operate under a different information regime from text-only retrievers and are therefore interpreted as diagnostic references rather than directly comparable baselines.

For Dis→Phen, Phen→Dis, and Phen-Triplet→Dis, we use Phenomizer-style information-content similarity with set-level aggregation over HPO phenotypes. (Köhler et al., 2009) For Phen→Gene, we use graph traversal over phenotype–disease–gene paths in the train-only inductive graph. This graph contains 350 of the 447 unique Phen→Gene evaluation targets, and 1,484 of 1,507 queries can reach at least one candidate gene through an indirect path.

Note that we do not report an ontology-aware reference for Tier 1 concept grounding. These tasks require mapping textual aliases to canonical concept definitions, whereas ontology traversal requires an already grounded concept identifier. Resolving the alias through the ontology’s alias table would directly disclose the target concept and reduce the task to a tautological dictionary lookup. Also, Dis→Gene result was not reported because the same inductive graph contains none of the unique evaluation target genes. The absence of target-gene coverage prevents the method from assigning meaningful graph-derived scores. We exclude the transductive graph because it contains the held-out disease–gene edges and reduces retrieval to direct lookup.

Training Objective. Fine-tuned models use architecture-specific contrastive objectives. BioLORD-0.1B uses Cached Multiple Negatives Ranking Loss (CachedMNRL), while Qwen3- Embedding-0.6B uses InfoNCE. ModernColBERT uses the PyLate CachedContrastive objective with ColBERT-style MaxSim scoring. These objectives share the general softmax contrastive form

$$
\mathcal { L } ( q , d ^ { + } ) = - \log \frac { \exp ( \sin ( q , d ^ { + } ) ) } { \sum _ { d \in \mathcal { B } } \exp ( \sin ( q , d ) ) } ,\tag{1}
$$

where B denotes the contrastive candidate set. Similarity is computed using cosine similarity for the bi-encoder models and MaxSim aggregation for ModernColBERT. Architecture-specific loss and batching configurations are reported in Appendix D.

Training Setup. Within each architecture, the Tier 1 and unified-supervision configurations use the same epoch count and architecture-specific optimization settings. All fine-tuned configurations were trained for two epochs on one NVIDIA H100 MIG device. Complete batching, optimization, and run-accounting details are provided in Appendix D.

## 4.4 Evaluation Metrics

Retrieval performance is evaluated using normalized discounted cumulative gain (nDCG), mean reciprocal rank (MRR), and Hit@k, where $k \in$ {1, 5, 10}. We report nDCG@10 in the main paper and provide additional metrics and implementation details in Appendix E. Relevance is binary, and metrics are computed over the complete candidate set for each task and macro-averaged across queries. Although we use standard retrieval metrics, ontology structure is incorporated through task construction: queries, candidate sets, and relevance labels are derived from curated biomedical relations and compositional phenotype structures. The metrics therefore assess retrieval under ontologyconstrained relevance rather than ontology consistency directly.

## 5 Results

## 5.1 Ontology-Aware Retrieval Reveals a Gap in Compositional Compatibility

Ontology-aware reference methods demonstrate that OntologyBench retrieval tasks are highly solvable when explicit biomedical structure is directly accessible. As shown in Table 3, ontology-aware methods operating over curated hierarchical and relational structure achieve strong retrieval performance across phenotype-based and compositional tasks (e.g., PhenTriplet→Dis nDCG@10 = 0.739). These ontology-aware reference methods operate under fundamentally different assumptions from embedding-based retrieval models. Rather than inferring compatibility from textual representations, they directly exploit explicit ontology hierarchy or curated disease–gene connectivity. The ontologyaware reference methods are therefore not intended as direct competitors, but instead serve as diagnostic references establishing the level of retrieval performance achievable when structured biomedical compatibility is explicitly available.

The ontology-aware reference methods reveal a substantial performance gap relative to the evaluated embedding retrieval systems. While ontologyaware reference methods directly exploit curated biomedical relationships and hierarchical structure, embedding models must recover such compatibility implicitly through representation learning from unstructured text and retrieval supervision alone. The observed performance gap suggests that current embedding training objectives may not fully preserve ontology-defined compatibility.

## 5.2 Retrieval Performance Differs Across Grounding, Relational, and Compositional Tasks

Table 2 shows substantial variation across retrieval tasks. Several models achieve their highest scores on concept-grounding tasks, whereas scores are generally lower on relational and compositional tasks. These cross-tier differences are descriptive because candidate spaces, relevance structures, and query ambiguity also vary across tasks. On the fixed PhenTriplet→Dis task, however, the best evaluated embedding model achieves nDCG@10 of 0.313, compared with 0.739 for the ontology-aware baseline.

<table><tr><td></td><td colspan="8">nDCG@10</td></tr><tr><td></td><td colspan="3">Tier 1</td><td colspan="4">Tier 2</td><td>Tier 3</td></tr><tr><td>Model</td><td>Gene→Def</td><td>Phen→Def</td><td>Dis→Def</td><td>Dis→Phen</td><td>Phen→Dis</td><td>Phen→Gene</td><td>Dis→Gene</td><td>PhenTriplet→Dis</td></tr><tr><td>BM25</td><td>0.087</td><td>0.381</td><td>0.275</td><td>0.070</td><td>0.103</td><td>0.071</td><td>0.054</td><td>0.107</td></tr><tr><td>MiniLM-L6-v2</td><td>0.263</td><td>0.629</td><td>0.314</td><td>0.106</td><td>0.153</td><td>0.109</td><td>0.058</td><td>0.123</td></tr><tr><td>MedEmbed-0.1B</td><td>0.440</td><td>0.738</td><td>0.454</td><td>0.130</td><td>0.169</td><td>0.134</td><td>0.068</td><td>0.143</td></tr><tr><td>SapBERT-0.1B</td><td>0.670</td><td>0.818</td><td>0.476</td><td>0.088</td><td>0.159</td><td>0.113</td><td>0.069</td><td>0.131</td></tr><tr><td>BioLORD-0.1B</td><td>0.290</td><td>0.807</td><td>0.426</td><td>0.091</td><td>0.178</td><td>0.105</td><td>0.058</td><td>0.151</td></tr><tr><td>BioLORD-0.1B†</td><td>0.513</td><td>0.869</td><td>0.562</td><td>0.112</td><td>0.175</td><td>0.091</td><td>0.048</td><td>0.171</td></tr><tr><td>BioLORD-0.1B</td><td>0.455</td><td>0.668</td><td>0.532</td><td>0.160</td><td>0.166</td><td>0.191</td><td>0.089</td><td>0.244</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.325</td><td>0.605</td><td>0.247</td><td>0.089</td><td>0.151</td><td>0.106</td><td>0.063</td><td>0.109</td></tr><tr><td>Qwen3-Embed-0.6B÷</td><td>0.615</td><td>0.864</td><td>0.538</td><td>0.112</td><td>0.158</td><td>0.076</td><td>0.045</td><td>0.158</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.499</td><td>0.572</td><td>0.488</td><td>0.184</td><td>0.283</td><td>0.269</td><td>0.110</td><td>0.313</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.526</td><td>0.616</td><td>0.330</td><td>0.119</td><td>0.188</td><td>0.129</td><td>0.073</td><td>0.080</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.747</td><td>0.860</td><td>0.516</td><td>0.116</td><td>0.195</td><td>0.136</td><td>0.069</td><td>0.169</td></tr></table>

Table 2: Retrieval performance on OntologyBench, evaluated using nDCG@10 across eight retrieval tasks spanning three retrieval tiers from grounding to compositional retrieval. Fine-tuned models are highlighted in blue, where † indicates models fine-tuned on the Tier 1 training set only, and ‡ indicates models fine-tuned jointly on the training sets from all three tiers.

<table><tr><td>Task</td><td>Method</td><td>nDCG@10</td></tr><tr><td>Dis→Phen</td><td>Phenomizer-style</td><td>0.482</td></tr><tr><td>Phen→Dis</td><td>Phenomizer-style</td><td>0.788</td></tr><tr><td>PhenTriplet→Dis</td><td>Phenomizer-style</td><td>0.739</td></tr><tr><td>Phen→Gene</td><td>Graph traversal</td><td>0.694</td></tr></table>

Table 3: Performance of ontology-aware reference methods. Phenotype–disease and compositional tasks use Phenomizer-style ontology-based semantic similarity, while Phen→Gene uses graph traversal over phenotype– disease–gene paths.

## 5.3 Task-Dependent Effects of Domain Specification and Supervision

Domain specialization provides an advantage in some comparisons but not uniformly across relational tasks. On Phenotype-Triplet→Disease, BioLORD-0.1B achieves nDCG@10 = 0.151, exceeding the pretrained Qwen3-Embed-0.6B and Qwen3-Embed-4B scores of 0.109 and 0.080, respectively.

Unified multi-tier supervision produces substantial gains on several relational and compositional tasks. Relative to its pretrained counterpart, Qwen3-Embed-0.6B improves by +0.047 to +0.163 nDCG@10 across the four Tier 2 tasks and by +0.204 on Phenotype-Triplet→Disease. It also exceeds the strongest pretrained baseline by up to +0.133 across Tier 2 and by +0.144 on Phenotype-Triplet→Disease. Some model–task combinations decline after fine-tuning, however, indicating that the benefit is not universal. Within the evaluated configurations, task-aligned supervision can produce larger gains than increasing model size, but these results do not establish a general ordering between supervision and scale. Additional evaluation results are provided in Appendix G.

## 5.4 Late Interaction and Generative Reranking Provide Limited End-to-End Improvement

Late-interaction rerankers substantially underperform the first-stage retriever on Tier-3 compositional retrieval tasks. Tier 1 only fine-tuning produced little improvement in ModernColBERT performance, whereas unified multi-tier fine-tuning reduced its performance (Table 4). LLM-based candidate scoring performed better than the evaluated token-interaction rerankers but the strongest LLM scorers produced only small and inconsistent differences relative to the first-stage retriever. Qwen3.6- 27B increased MRR@10 from 0.297 to 0.306 and nDCG@10 from 0.313 to 0.315, while its Hit@10 was slightly lower. The strongest LLM scorers produced performance comparable to, rather than clearly exceeding, the retriever. GPT-5.4-mini produced the highest Hit@10, but its MRR@10 and nDCG@10 remained comparable to the retriever.

The first-stage retriever returned at least one relevant disease in its top-50 candidates for 2,083 of 2,537 queries (82.1%). For the remaining 454 queries (17.9%), reranking could not recover a relevant disease because no relevant candidate was available to reorder. These queries were retained in the evaluation and received zero retrieval credit, so the reported metrics reflect the performance of the complete two-stage system. Because paired uncertainty estimates were not calculated, the small numerical differences among rerankers should not be interpreted as evidence that one method was superior. Overall, none of the evaluated rerankers clearly improved on the first-stage retriever, indicating that reranking did not resolve the Tier-3 retrieval bottleneck.

<table><tr><td></td><td colspan="3">k = 10</td></tr><tr><td>Model</td><td>MRR</td><td>Hit</td><td>nDCG</td></tr><tr><td>Retriever</td><td></td><td></td><td>0.313</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.297</td><td>0.572</td><td></td></tr><tr><td>Token-Interaction Rerankers ModernColBERT</td><td>0.168</td><td>0.384</td><td>0.180</td></tr><tr><td></td><td></td><td></td><td></td></tr><tr><td>ModernColBERT† ModernColBERT</td><td>0.168</td><td>0.386</td><td>0.183</td></tr><tr><td></td><td>0.135</td><td>0.384</td><td>0.163</td></tr><tr><td>LLM Candidate Scoring</td><td></td><td></td><td></td></tr><tr><td>Qwen3-4B-Instruct-2507</td><td>0.262</td><td>0.525</td><td>0.275</td></tr><tr><td>Qwen3-4B-Thinking-2507</td><td>0.278</td><td>0.546</td><td>0.288</td></tr><tr><td>Qwen3.6-27B</td><td>0.306</td><td>0.571</td><td>0.315</td></tr><tr><td>Gemma4-31B</td><td>0.284</td><td>0.524</td><td>0.287</td></tr><tr><td>OpenAI (GPT5.4-mini)</td><td>0.301</td><td>0.574</td><td>0.312</td></tr></table>

Table 4: End-to-end performance on the Tier-3 PhenTriplet→Dis task $( N = 2 { , } 5 3 7 )$ . MRR@10, binary Hit@10, and nDCG@10 were calculated over the complete query set. The first-stage top-50 candidate sets contained at least one relevant disease for 2,083 queries (82.1%); the remaining 454 queries remained in the denominator and contributed zero retrieval credit. Candidate-scoring methods reordered retrieved candidates but did not introduce additional diseases.

## 5.5 Failure Modes in Compositional Retrieval

We perform a rule-based error analysis (Appendix H) of Tier-3 queries for which the embedding model does not rank the ground-truth disease first. The analysis distinguishes failures for which the ontology-aware reference retrieves the groundtruth disease within the top 10 from remaining failures. Failures cluster into four modes: (1) aggregation failure (58.5%), where the model retrieves diseases matching only a subset of the query phenotypes rather than the full phenotype combination; (2) generic phenotype bias (25.0%), where retrieval is driven by broad or non-specific phenotypes; (3) related-disease confusion (11.8%), where the predicted disease is phenotypically similar to the ground-truth disease; and (4) semantic drift (4.7%), where retrieved diseases show low phenotype overlap and low ontology similarity. Aggregation failures dominate, indicating that embedding models often match phenotypes independently rather than enforcing joint compatibility across the full phenotype set. Unified multi-tier supervision improves Tier 3 retrieval substantially, increasing Hit@10 from 0.170 to 0.572 and nDCG@10 from 0.109 to 0.313 (Appendix G Table 12). However, performance remains below the ontology-aware reference, indicating that compositional ranking errors remain common.

The evaluated embedding models frequently retrieved diseases compatible with only part of the phenotype set. This finding characterizes the evaluated systems and task construction; they do not establish that all embedding architectures are inherently limited to represent compositional biomedical information. This limitation is consistent with lower performance on relation types (e.g., phenotype→gene), indicating that compositional structure remains underrepresented in current representations.

## 6 Discussion

Retrieval systems are increasingly expected to recover compatibility across multiple structured constraints rather than simply retrieve semantically similar text. OntologyBench evaluates this distinction across concept-grounding, relational, and compositional retrieval tasks. Across the evaluated architectures, performance is generally lower on relational and compositional tasks than on several grounding tasks. These cross-task differences are descriptive because candidate spaces, relevance structures, and query ambiguity also vary. Within the fixed Tier-3 task, however, the evaluated embedding models remain substantially below the ontology-aware reference method, indicating that the evaluated text-based configurations do not recover ontology-defined relevance as effectively when multiple phenotype constraints must be combined.

Unified multi-tier supervision improves performance on several relational and compositional tasks, indicating that ontology-derived supervision can produce representations that better support these retrieval relationships. However, the remaining gap relative to ontology-aware reference methods cannot be attributed solely to the embedding objectives. The reference methods directly access curated ontology relations, whereas the embedding models operate over textual representations that may not contain all information used to define relevance. The results therefore motivate retrieval architectures that integrate textual representations with ontology knowledge rather than establishing that embedding objectives alone are responsible for the observed gap.

Our two-stage evaluation provides additional insight into where performance is lost. Under the evaluated configurations, token-interaction reranking underperforms the first-stage retriever, while the strongest LLM-based candidate scorer provides only a modest end-to-end improvement over the fixed retriever. Because these metrics include queries for which no relevant disease appears in the initial candidate set, they reflect both candidategeneration and downstream-ranking limitations. Within the rule-selected Tier-3 failure subset, the largest category consists of predictions matching only part of the phenotype combination. Together, these findings suggest that improving downstream scoring alone may be insufficient when relevant candidates are omitted or compatibility is not adequately represented during initial retrieval.

Although OntologyBench is constructed from biomedical ontologies, its evaluation framework may be applicable beyond biomedicine. Retrieval-augmented generation, agentic systems, knowledge-graph retrieval, and other knowledgeintensive NLP applications may require candidates satisfying multiple structured constraints rather than maximizing semantic similarity alone. By organizing tasks across grounding, relational, and compositional retrieval, OntologyBench provides a framework for examining retrieval behaviour under different forms of structured compatibility. We hope it supports future work on retrieval objectives and architectures that more effectively integrate representation learning with structured knowledge.

## 7 Conclusion

We introduced OntologyBench, a benchmark spanning concept grounding, ontology-relation retrieval, and phenotype-set retrieval. Scores were generally lower on relational and compositional tasks than on several grounding tasks. On the fixed PhenTriplet→Dis task, the evaluated embedding configurations remained below the ontology-aware reference methods, and the evaluated rerankers provided little or no end-to-end improvement over the first-stage retriever. Candidate omission contributed to the observed end-to-end gap.

These findings motivate retrieval systems that combine flexible textual representation with explicit structured knowledge; they do not establish a universal ordering between embedding and symbolic approaches.

## 8 Data and Code Availability

The OntologyBench benchmark is publicly available on Github<sup>3</sup> under the MIT license, with the benchmark dataset additionally distributed through Hugging Face<sup>4</sup> under the CC BY 4.0 License.

## Limitations

OntologyBench evaluates retrieval under controlled biomedical compatibility constraints and does not capture broader aspects of real-world diagnosis, such as temporal progression, causal mechanisms, incomplete observations, or patient-specific context.

Task-specific partitioning prevents overlap of exact query–document pairs and same-task target identifiers. However, biomedical concepts and underlying relations may recur across training tasks. The unified setting therefore evaluates transductive multi-task transfer within a shared ontology vocabulary rather than generalization to entirely unseen entities or relations. However, scores were generally lower on relational and compositional tasks than on several grounding tasks, although task characteristics differ and these cross-tier contrasts should not be interpreted as isolated effects of constraint complexity (Table 2).

Our experiments evaluate a representative range of retrieval architectures and model scales, but are not intended as an exhaustive leaderboard-style comparison of all emerging foundation systems. The primary goal of OntologyBench is instead to characterize retrieval behaviour across grounding, relational, and compositional retrieval tasks. Future work may investigate whether larger-scale or ontology-aware architectures mitigate the limitations identified here.

OntologyBench evaluates retrieval behaviour under structured biomedical constraints rather than directly measuring preservation of ontology geometry within embedding space. Standard retrieval metrics such as nDCG, MRR, and Hit@k therefore do not explicitly quantify hierarchical distance preservation or geometric alignment with ontology topology. Consequently, the lower scores observed on relational and compositional retrievals should be interpreted as evidence of limitations in compositional retrieval rather than a definitive measure of ontology structure preservation within learned representations.

Although ontology-aligned supervision improves relational and compositional retrieval, the current study does not fully isolate which components of the supervision signal contribute most strongly to these gains. Improvements may arise from a combination of structured relational supervision, task-specific retrieval supervision, and domain adaptation effects. Future work could further disentangle these factors through more controlled supervision ablations.

Finally, OntologyBench focuses on representation-level retrieval. While practical clinical systems may incorporate reranking, generative reasoning, and expert validation, our results suggest that substantial limitations already emerge at the retrieval stage. The benchmark should therefore be interpreted as a diagnostic framework for analyzing retrieval behaviour under structured biomedical constraints rather than a direct proxy for end-to-end clinical performance.

## Ethical Considerations

OntologyBench is constructed entirely from publicly available biomedical ontologies released under open licenses, including the HPO and MONDO (CC-BY 4.0) and HGNC and NCBI resources (public domain). These sources provide structured definitions of concepts and relations rather than patientlevel data. OntologyBench contains no personally identifiable information or sensitive health records.

## Statement of AI Use

Generative AI was used for language editing. All research design, methodology, experiments, analyses, results, and conclusions are the work of the authors and were verified by the authors.

## Acknowledgments

This research was enabled in part through the computational resources provided by Advanced Research Computing at the University of British Columbia and the Digital Research Alliance of Canada. This work was funded by the Nenad Blau IEMBase Endowment Fund of the MCF, Marin

County, CA, USA. This work was also supported by a Natural Sciences and Engineering Research Council of Canada (NSERC) Discovery Grant (RGPIN-2024-06783) awarded to WWW; a BC Children’s Hospital Research Institute (BCCHR) Doctoral Studentship awarded to XYCZ; and an NSERC Discovery Grant and a Canada Foundation for Innovation John R. Evans Leaders Fund (CFI-JELF) grant awarded to JZ.

## References

Rishita Agarwal, Himanshu Singhal, Peter Baile Chen, Manan Roy Choudhury, Dan Roth, and Vivek Gupta. 2026. Rear: Retrieve, expand and refine for effective multitable retrieval. In Proceedings of the 64th Annual Meeting ofthe Associationfor Computational Linguistics (Volume 1: Long Papers), pages 39360– 39374.

Abhinand Balachandran. 2024. Medembed: Medicalfocused embedding models.

Vera Boteva, Demian Gholipour, Artem Sokolov, and Stefan Riezler. 2016. A full-text learning to rank dataset for medical information retrieval. In European Conference on Information Retrieval, pages 716–722. Springer.

Antoine Chaffin and Raphaël Sourty. 2025. Pylate: Flexible training and retrieval for late interaction models. In Proceedings of the 34th ACM International Conference on Information and Knowledge Management, pages 6334–6339.

Michael A Gargano, Nicolas Matentzoglu, Ben Coleman, Eunice B Addo-Lartey, Anna V Anagnostopoulos, Joel Anderton, Paul Avillach, Anita M Bagley, Eduard Bakštein, James P Balhoff, et al. 2024. The human phenotype ontology in 2024: phenotypes around the world. Nucleic acids research, 52(D1):D1333–D1346.

Google DeepMind. 2026. gemma-4-31b. https:// huggingface.co/google/gemma-4-31B. Hugging Face model repository.

Robert Hoehndorf, Paul N Schofield, and Georgios V Gkoutos. 2015. The role of ontologies in biological and biomedical research: a functional perspective. Briefings in bioinformatics, 16(6):1069–1080.

Omar Khattab and Matei Zaharia. 2020. Colbert: Efficient and effective passage search via contextualized late interaction over bert. In Proceedings of the 43rd International ACM SIGIR conference on research and development in Information Retrieval, pages 39– 48.

Sebastian Köhler, Marcel H Schulz, Peter Krawitz, Sebastian Bauer, Sandra Dölken, Claus E Ott, Christine Mundlos, Denise Horn, Stefan Mundlos, and Peter N

Robinson. 2009. Clinical diagnostics in human genetics with semantic similarity searches in ontologies. The American Journal ofHuman Genetics, 85(4):457– 464.

Lei Li, Xiao Zhou, and Zheng Liu. 2025. R2med: A benchmark for reasoning-driven medical retrieval. arXiv preprint arXiv:2505.14558.

Fangyu Liu, Ehsan Shareghi, Zaiqiao Meng, Marco Basaldella, and Nigel Collier. 2021. Self-alignment pretraining for biomedical entity representations. In Proceedings of the 2021 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies, pages 4228–4238, Online. Association for Computational Linguistics.

Niklas Muennighoff, Nouamane Tazi, Loïc Magne, and Nils Reimers. 2023. Mteb: Massive text embedding benchmark. In Proceedings ofthe 17th Conference of the European Chapter of the Association for Computational Linguistics, pages 2014–2037.

Nuala A O’Leary, Eric Cox, J Bradley Holmes, W Ray Anderson, Robert Falk, Vichet Hem, Mirian TN Tsuchiya, Gregory D Schuler, Xuan Zhang, John Torcivia, et al. 2024. Exploring and retrieving sequence and metadata for species across the tree of life with ncbi datasets. Scientific data, 11(1):732.

Qwen Team. 2026. Qwen3.6-27B: Flagship-level coding in a 27B dense model.

Nils Reimers and Iryna Gurevych. 2019. Sentence-bert: Sentence embeddings using siamese bert-networks. arXiv preprint arXiv:1908.10084.

François Remy, Kris Demuynck, and Thomas Demeester. 2024. BioLORD-2023: semantic textual representations fusing large language models and clinical knowledge graph insights. Journal of the American Medical Informatics Association, page ocae029.

Philip Resnik. 1999. Semantic similarity in a taxonomy: An information-based measure and its application to problems of ambiguity in natural language. Journal ofartificial intelligence research, 11:95–130.

Ruth L Seal, Bryony Braschi, Kristian Gray, James McClay, Susan Tweedie, and Elspeth A Bruford. 2026. Genenames. org: the hgnc and pgnc resources in 2026. Nucleic Acids Research, 54(D1):D1098– D1107.

Hongjin Su, Howard Yen, Mengzhou Xia, Weijia Shi, Niklas Muennighoff, Han-yu Wang, Haisu Liu, Quan Shi, Zachary S Siegel, Michael Tang, et al. 2024. Bright: A realistic and challenging benchmark for reasoning-intensive retrieval. arXiv preprint arXiv:2407.12883.

Yixuan Tang, Zhenghong Lin, Yandong Sun, Wynne Hsu, Mong Li Lee, and Anthony KH Tung. 2026.

Qime: Constructing interpretable medical text embeddings via ontology-grounded questions. arXiv preprint arXiv:2603.01690.

Qwen Team. 2025. Qwen3 technical report.

Nandan Thakur, Nils Reimers, Andreas Rücklé, Abhishek Srivastava, and Iryna Gurevych. 2021. Beir: A heterogenous benchmark for zero-shot evaluation of information retrieval models. arXiv preprint arXiv:2104.08663.

Nicole A Vasilevsky, Sabrina Toro, Nicolas Matentzoglu, Joseph E Flack, Kathleen R Mullen, Harshad Hegde, Sarah Gehrke, Patricia L Whetzel, Yousif Shwetar, Nomi L Harris, et al. 2025. Mondo: Integrating disease terminology across communities. Genetics, page iyaf215.

Ellen Voorhees, Tasmeer Alam, Steven Bedrick, Dina Demner-Fushman, William R Hersh, Kyle Lo, Kirk Roberts, Ian Soboroff, and Lucy Lu Wang. 2021. Trec-covid: constructing a pandemic information retrieval test collection. In ACM SIGIR Forum, volume 54, pages 1–12. ACM New York, NY, USA.

Chenghao Xiao, G Thomas Hudson, and Noura Al Moubayed. 2024. Rar-b: Reasoning as retrieval benchmark. arXiv preprint arXiv:2404.06347.

Zhilin Yang, Peng Qi, Saizheng Zhang, Yoshua Bengio, William Cohen, Ruslan Salakhutdinov, and Christopher D Manning. 2018. Hotpotqa: A dataset for diverse, explainable multi-hop question answering. In Proceedings of the 2018 conference on empirical methods in natural language processing, pages 2369–2380.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, et al. 2025. Qwen3 embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176.

## A Dataset Examples

This appendix shows representative examples from the three tiers of OntologyBench tasks.

## A.1 Tier 1: Concept Grounding

Gene→Document Retrieval

doc\_id: GENE:2011 doc: Protein-coding gene located on chromosome 11q13.1 belonging to the microtubule affinity regulating kinase family. The encoded kinase regulates epithelial and neuronal cell polarity and microtubule stability.

## Phen→Definition Retrieval

doc\_id: HP:0006349   
doc: Congenital absence of one or more permanent teeth, including hypodontia, oligodontia, or complete tooth absence.

## MONDO→Definition Retrieval

doc\_id: MONDO:0014543 doc: Congenital myasthenic syndrome with glycosylation defect caused by mutations in the ALG2 gene.

## A.2 Tier 2: Ontology-Level Relations

## Disease→Phenotype

doc\_id: HP:0002154

doc: Elevated concentration of glycine in the blood.

## Phenotype→Disease

doc\_id: MONDO:0009479

doc: Johanson-Blizzard syndrome characterized by exocrine pancreatic insufficiency, nasal alae hypoplasia, hearing loss, growth retardation, and intellectual disability.

## Phenotype→Gene

doc\_id: GENE:80185

doc: TTI2 encodes a regulator of the DNA damage response and a component of the Triple T complex involved in cellular resistance to DNA damage stresses.

## Disease→Gene

doc\_id: GENE:889 doc: KRIT1 encodes protein involved in beta1-integrin-mediated cell proliferation and endothelial junction integrity. Mutations cause cerebral cavernous malformations.

## A.3 Tier 3: Compositional Phenotype-Based Disease Retrieval

## Multi-Phenotype→Disease

doc\_id: MONDO:0016394   
doc: Sporadic infantile bilateral striatal necrosis characterized by degeneration of the caudate nucleus, putamen, and globus pallidus, leading to developmental regression and movement disorders.

## Multi-Phenotype→Disease

doc\_id: MONDO:0011614   
doc: HMG-CoA synthase-2 deficiency, a rare disorder of ketone body metabolism presenting with vomiting, lethargy, hepatomegaly, and non-ketotic hypoglycemia.

## Multi-Phenotype→Disease

doc\_id: MONDO:0020502 doc: Yellow fever, a zoonotic viral disease that may progress from fever and systemic symptoms to hemorrhagic fever and multi-organ failure.

## B Additional Dataset Analysis

## Tier 1 Statistics

Disease identity grounding (MONDO) contains 6,611 canonical concepts associated with 48,854 alias queries (mean 7.39 aliases per concept; median 6; maximum 50).

Alias polysemy is limited: 1.0% of surface forms map to more than one concept. Gene identity grounding contains 2,345 gene concepts, each associated with a single canonical symbol. Phenotype grounding includes 8,253 phenotype concepts with extensive synonym coverage.

These statistics indicate that Tier 1 primarily evaluates lexical normalization under moderate alias diversity rather than heavy ambiguity.

## Phenotype Cardinality and Hypothesis Space Reduction

We fix Tier-3 phenotype cardinality to three to study the minimal non-trivial regime of compositional retrieval. Ontology association statistics indicate:

• Single phenotype: mean 328.9 compatible diseases

• Phenotype pairs: mean 26.7 compatible diseases

• Phenotype triplets: mean 5.0 compatible diseases

• Phenotype quadruplets: mean 1.95

• Phenotype quintuplets: mean 1.31

Triplets provide substantial hypothesis-space reduction while avoiding near-deterministic collapse observed at higher cardinalities.

## C Experimental Details

Sequence Length Distribution and Truncation

<table><tr><td>Model</td><td>Mean</td><td>P95</td><td>P99</td><td>Max</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>33.40</td><td>112</td><td>161</td><td>1053</td></tr><tr><td>BioLORD-0.1B</td><td>48.39</td><td>128</td><td>205</td><td>926</td></tr><tr><td>ColBERT</td><td>25.08</td><td>88</td><td>149</td><td>845</td></tr></table>

Table 5: Token length statistics of training instances under different tokenizers (no truncation applied). We report the mean length, 95th percentile (P95), 99th percentile (P99), and maximum sequence length.

Bi-encoder models used a maximum sequence length of 256 tokens to limit truncation under the heavy-tailed definition-length distribution. ModernColBERT retained its architecture-specific maximum lengths of 48 query tokens and 300 document tokens. Optimization, batching, hardware, and runaccounting details are provided in Appendix D.

## D Optimization and Training Setup

All fine-tuned configurations were trained for two epochs on one NVIDIA H100 MIG device with gradient accumulation of 1 and random seed 42. Reported fine-tuning results are from single training runs rather than averages across multiple seeds.

BioLORD-0.1B was fine-tuned with Cached Multiple Negatives Ranking Loss using a batch size of 256 and cached forward mini-batches of 128. Qwen3-Embedding-0.6B was fine-tuned with InfoNCE using a batch size of 256 packed records, sequence packing with a length of 256 tokens, and no cached-loss subdivision. ModernColBERT was fine-tuned with the PyLate CachedContrastive objective using a batch size of 256 and cached forward mini-batches of 32. Cross-device negative gathering was disabled for cached-loss configurations.

Bi-encoder models used a maximum sequence length of 256 tokens. ModernColBERT used maximum query and document lengths of 48 and 300 tokens, respectively. Learning rates were $2 \times 1 0 ^ { - 5 }$ for BioL $\mathrm { O R D - 0 . 1 B , 5 \times 1 0 ^ { - 5 } }$ for Qwen3-Embedding-0.6B, and $3 \times 1 0 ^ { - 6 }$ for ModernColBERT. Complete training configurations and saved trainer states are included in the released repository.

## E Evaluation Protocol

For each task, every query is evaluated by ranking all documents in the corresponding taskspecific corpus without heuristic filtering or ontology pruning. Metrics are computed independently for each task at $k \in \{ 1 , 5 , 1 0 \}$ and macro-averaged across all test queries.

Let $G _ { q }$ denote the non-empty set of relevant documents for query $q ,$ and let $d _ { q , i }$ denote the document at rank i after duplicate document identifiers have been removed. Relevance is binary:

$$
\operatorname { r e l } _ { q , i } = { \left\{ \begin{array} { l l } { 1 , } & { d _ { q , i } \in G _ { q } , } \\ { 0 , } & { d _ { q , i } \notin G _ { q } . } \end{array} \right. }\tag{2}
$$

Discounted cumulative gain at cutoff k is defined as

$$
\mathrm { D C G } _ { q } @ k = \sum _ { i = 1 } ^ { k } \frac { \mathrm { r e l } _ { q , i } } { \log _ { 2 } ( i + 1 ) } .\tag{3}
$$

For binary relevance, the ideal ranking contains min $\left( k , | G _ { q } | \right)$ relevant documents. Therefore,

$$
\mathrm { I D C G } _ { q } @ k = \sum _ { i = 1 } ^ { \operatorname* { m i n } ( k , | G _ { q } | ) } \frac { 1 } { \log _ { 2 } ( i + 1 ) } ,\tag{4}
$$

and normalized discounted cumulative gain is

$$
\mathrm { n D C G } _ { q } @ k = \frac { \mathrm { D C G } _ { q } @ k } { \mathrm { I D C G } _ { q } @ k } .\tag{5}
$$

Let $r _ { q }$ denote the rank of the first relevant document. The truncated reciprocal-rank and Hit@k metrics are

$$
\mathrm { R R } _ { q } @ k = \left\{ \begin{array} { l l } { 1 / r _ { q } , } & { r _ { q } \le k , } \\ { 0 , } & { r _ { q } > k , } \end{array} \right.\tag{6}
$$

$$
\mathrm { H i t } _ { q } @ k = \left\{ { 1 , } \quad r _ { q } \leq k , \right.\tag{7}
$$

For any query-level metric $m _ { q } @ k$ , the reported task-level score is the macro-average over the complete test-query set $Q$

$$
M @ k = \frac { 1 } { | Q | } \sum _ { q \in Q } m _ { q } @ k .\tag{8}
$$

Thus, MRR@k is the macro-average of $\mathrm { R R } _ { q } @ k$ while Hit@k is the proportion of test queries with at least one relevant document among the top-k retrieved results.

## F Prompt Used for Generative Candidate Scoring

For LLM-based candidate scoring, each candidate disease was evaluated independently using the following prompt template. The same semantic prompt was supplied through each model’s native chat template.

![](images/bf151faea42710ab3a0f6df435fdda03488f7a7f769b1a5dec81c3fa3bb3140e.jpg)

## G Additional Evaluation Metrics

<table><tr><td></td><td colspan="9">nDCG@k</td></tr><tr><td>Model</td><td colspan="3">Gene→Def</td><td colspan="3">Phen→Def</td><td colspan="3">Dis→Def</td></tr><tr><td></td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.079</td><td>0.083</td><td>0.087</td><td>0.265</td><td>0.361</td><td>0.381</td><td>0.200</td><td>0.262</td><td>0.275</td></tr><tr><td>MiniLM-L6-v2</td><td>0.158</td><td>0.244</td><td>0.263</td><td>0.500</td><td>0.609</td><td>0.629</td><td>0.224</td><td>0.296</td><td>0.314</td></tr><tr><td>BioLORD-0.1B‡</td><td>0.311</td><td>0.428</td><td>0.455</td><td>0.452</td><td>0.639</td><td>0.668</td><td>0.389</td><td>0.505</td><td>0.532</td></tr><tr><td>BioLORD-0.1B†</td><td>0.364</td><td>0.489</td><td>0.513</td><td>0.767</td><td>0.862</td><td>0.869</td><td>0.425</td><td>0.540</td><td>0.562</td></tr><tr><td>BioLORD-0.1B</td><td>0.167</td><td>0.268</td><td>0.290</td><td>0.693</td><td>0.795</td><td>0.807</td><td>0.309</td><td>0.402</td><td>0.426</td></tr><tr><td>MedEmbed-0.1B</td><td>0.320</td><td>0.424</td><td>0.440</td><td>0.600</td><td>0.722</td><td>0.738</td><td>0.350</td><td>0.433</td><td>0.454</td></tr><tr><td>SapBERT-0.1B</td><td>0.510</td><td>0.654</td><td>0.670</td><td>0.699</td><td>0.806</td><td>0.818</td><td>0.353</td><td>0.452</td><td>0.476</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.348</td><td>0.476</td><td>0.499</td><td>0.390</td><td>0.541</td><td>0.572</td><td>0.335</td><td>0.460</td><td>0.488</td></tr><tr><td>Qwen3-Embed-0.6B†</td><td>0.480</td><td>0.597</td><td>0.615</td><td>0.762</td><td>0.855</td><td>0.864</td><td>0.401</td><td>0.515</td><td>0.538</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.220</td><td>0.305</td><td>0.325</td><td>0.475</td><td>0.586</td><td>0.605</td><td>0.174</td><td>0.232</td><td>0.247</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.378</td><td>0.501</td><td>0.526</td><td>0.476</td><td>0.592</td><td>0.616</td><td>0.214</td><td>0.306</td><td>0.330</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.619</td><td>0.730</td><td>0.747</td><td>0.751</td><td>0.851</td><td>0.860</td><td>0.391</td><td>0.493</td><td>0.516</td></tr></table>

Table 6: Tier 1 ontology grounding tasks evaluated using nDCG@k.

<table><tr><td rowspan="3">Model</td><td colspan="9">MRR@k</td></tr><tr><td colspan="3">Gene→Def</td><td colspan="3">Phen→Def</td><td colspan="3">Dis→Def</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.079</td><td>0.082</td><td>0.083</td><td>0.265</td><td>0.333</td><td>0.341</td><td>0.200</td><td>0.244</td><td>0.250</td></tr><tr><td>MiniLM-L6-v2</td><td>0.158</td><td>0.219</td><td>0.227</td><td>0.500</td><td>0.579</td><td>0.587</td><td>0.224</td><td>0.275</td><td>0.282</td></tr><tr><td>BioLORD-0.1B</td><td>0.311</td><td>0.392</td><td>0.404</td><td>0.452</td><td>0.586</td><td>0.598</td><td>0.389</td><td>0.472</td><td>0.483</td></tr><tr><td>BioLORD-0.1B†</td><td>0.364</td><td>0.454</td><td>0.464</td><td>0.767</td><td>0.836</td><td>0.839</td><td>0.425</td><td>0.507</td><td>0.516</td></tr><tr><td>BioLORD-0.1B</td><td>0.167</td><td>0.237</td><td>0.246</td><td>0.693</td><td>0.766</td><td>0.771</td><td>0.309</td><td>0.375</td><td>0.385</td></tr><tr><td>MedEmbed-0.1B</td><td>0.320</td><td>0.395</td><td>0.402</td><td>0.600</td><td>0.688</td><td>0.695</td><td>0.350</td><td>0.409</td><td>0.418</td></tr><tr><td>SapBERT-0.1B</td><td>0.510</td><td>0.614</td><td>0.620</td><td>0.699</td><td>0.777</td><td>0.782</td><td>0.353</td><td>0.423</td><td>0.433</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.348</td><td>0.438</td><td>0.447</td><td>0.390</td><td>0.497</td><td>0.509</td><td>0.335</td><td>0.423</td><td>0.435</td></tr><tr><td>Qwen3-Embed-0.6B†</td><td>0.480</td><td>0.565</td><td>0.573</td><td>0.762</td><td>0.830</td><td>0.834</td><td>0.401</td><td>0.482</td><td>0.491</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.220</td><td>0.282</td><td>0.290</td><td>0.475</td><td>0.554</td><td>0.562</td><td>0.174</td><td>0.215</td><td>0.221</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.378</td><td>0.466</td><td>0.477</td><td>0.476</td><td>0.558</td><td>0.568</td><td>0.214</td><td>0.279</td><td>0.289</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.619</td><td>0.699</td><td>0.706</td><td>0.751</td><td>0.825</td><td>0.829</td><td>0.391</td><td>0.464</td><td>0.473</td></tr></table>

Table 7: Tier 1 ontology grounding tasks evaluated using MRR@k.

<table><tr><td rowspan="3">Model</td><td colspan="9">Hit@k</td></tr><tr><td colspan="3">Gene→Def</td><td colspan="3">Phen→Def</td><td colspan="3">Dis→Def</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.079</td><td>0.088</td><td>0.100</td><td>0.265</td><td>0.448</td><td>0.508</td><td>0.200</td><td>0.315</td><td>0.357</td></tr><tr><td>MiniLM-L6-v2</td><td>0.158</td><td>0.318</td><td>0.378</td><td>0.500</td><td>0.700</td><td>0.761</td><td>0.224</td><td>0.360</td><td>0.417</td></tr><tr><td>BioLORD-0.1B‡</td><td>0.311</td><td>0.534</td><td>0.619</td><td>0.452</td><td>0.797</td><td>0.886</td><td>0.389</td><td>0.605</td><td>0.686</td></tr><tr><td>BioLORD-0.1B†</td><td>0.364</td><td>0.592</td><td>0.666</td><td>0.767</td><td>0.937</td><td>0.958</td><td>0.425</td><td>0.639</td><td>0.706</td></tr><tr><td>BioLORD-0.1B</td><td>0.167</td><td>0.362</td><td>0.429</td><td>0.693</td><td>0.879</td><td>0.916</td><td>0.309</td><td>0.485</td><td>0.559</td></tr><tr><td>MedEmbed-0.1B</td><td>0.320</td><td>0.508</td><td>0.559</td><td>0.600</td><td>0.821</td><td>0.873</td><td>0.350</td><td>0.505</td><td>0.569</td></tr><tr><td>SapBERT-0.1B</td><td>0.510</td><td>0.775</td><td>0.821</td><td>0.699</td><td>0.892</td><td>0.929</td><td>0.353</td><td>0.538</td><td>0.612</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.348</td><td>0.592</td><td>0.664</td><td>0.390</td><td>0.676</td><td>0.771</td><td>0.335</td><td>0.569</td><td>0.656</td></tr><tr><td>Qwen3-Embed-0.6B†</td><td>0.480</td><td>0.689</td><td>0.745</td><td>0.762</td><td>0.929</td><td>0.955</td><td>0.401</td><td>0.615</td><td>0.684</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.220</td><td>0.374</td><td>0.436</td><td>0.475</td><td>0.681</td><td>0.741</td><td>0.174</td><td>0.283</td><td>0.329</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.378</td><td>0.606</td><td>0.680</td><td>0.476</td><td>0.692</td><td>0.768</td><td>0.214</td><td>0.386</td><td>0.461</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.619</td><td>0.824</td><td>0.876</td><td>0.751</td><td>0.928</td><td>0.956</td><td>0.391</td><td>0.582</td><td>0.652</td></tr></table>

Table 8: Tier 1 ontology grounding tasks evaluated using Hit@k.

<table><tr><td rowspan="3"></td><td colspan="10">nDCG@k</td></tr><tr><td colspan="3">Dis→Gene</td><td colspan="3">Dis→Phen</td><td colspan="3">Phen→Dis</td><td colspan="3">Phen→Gene</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.076</td><td>0.050</td><td>0.054</td><td>0.084</td><td>0.066</td><td>0.070</td><td>0.099</td><td>0.099</td><td>0.103</td><td>0.048</td><td>0.065</td><td>0.071</td></tr><tr><td>MiniLM-L6-v2</td><td>0.079</td><td>0.055</td><td>0.058</td><td>0.120</td><td>0.098</td><td>0.106</td><td>0.150</td><td>0.143</td><td>0.153</td><td>0.058</td><td>0.095</td><td>0.109</td></tr><tr><td>BioLORD-0.1B</td><td>0.112</td><td>0.083</td><td>0.089</td><td>0.158</td><td>0.142</td><td>0.160</td><td>0.034</td><td>0.119</td><td>0.166</td><td>0.104</td><td>0.165</td><td>0.191</td></tr><tr><td>BioLORD-0.1B†</td><td>0.061</td><td>0.043</td><td>0.048</td><td>0.130</td><td>0.104</td><td>0.112</td><td>0.151</td><td>0.159</td><td>0.175</td><td>0.054</td><td>0.078</td><td>0.091</td></tr><tr><td>BioLORD-0.1B</td><td>0.066</td><td>0.053</td><td>0.058</td><td>0.099</td><td>0.083</td><td>0.091</td><td>0.158</td><td>0.161</td><td>0.178</td><td>0.060</td><td>0.094</td><td>0.105</td></tr><tr><td>MedEmbed-0.1B</td><td>0.095</td><td>0.066</td><td>0.068</td><td>0.148</td><td>0.120</td><td>0.130</td><td>0.155</td><td>0.157</td><td>0.169</td><td>0.079</td><td>0.118</td><td>0.134</td></tr><tr><td>SapBERT-0.1B</td><td>0.096</td><td>0.066</td><td>0.069</td><td>0.104</td><td>0.080</td><td>0.088</td><td>0.142</td><td>0.148</td><td>0.159</td><td>0.068</td><td>0.101</td><td>0.113</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.137</td><td>0.102</td><td>0.110</td><td>0.185</td><td>0.165</td><td>0.184</td><td>0.224</td><td>0.254</td><td>0.283</td><td>0.143</td><td>0.236</td><td>0.269</td></tr><tr><td>Qwen3-Embed-0.6B÷</td><td>0.064</td><td>0.041</td><td>0.045</td><td>0.124</td><td>0.103</td><td>0.112</td><td>0.131</td><td>0.142</td><td>0.158</td><td>0.050</td><td>0.070</td><td>0.076</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.089</td><td>0.060</td><td>0.063</td><td>0.104</td><td>0.084</td><td>0.089</td><td>0.139</td><td>0.139</td><td>0.151</td><td>0.066</td><td>0.091</td><td>0.106</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.105</td><td>0.070</td><td>0.073</td><td>0.144</td><td>0.112</td><td>0.119</td><td>0.169</td><td>0.174</td><td>0.188</td><td>0.076</td><td>0.115</td><td>0.129</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.092</td><td>0.065</td><td>0.069</td><td>0.140</td><td>0.110</td><td>0.116</td><td>0.176</td><td>0.181</td><td>0.195</td><td>0.084</td><td>0.121</td><td>0.136</td></tr></table>

Table 9: Tier-2 ontology relation tasks evaluated using nDCG@k.

<table><tr><td></td><td colspan="10">MRR@k</td></tr><tr><td></td><td colspan="3">Dis→Gene</td><td colspan="3">Dis→Phen</td><td colspan="3">Phen→Dis</td><td colspan="3">Phen→Gene</td></tr><tr><td>Model</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.076</td><td>0.105</td><td>0.114</td><td>0.084</td><td>0.120</td><td>0.128</td><td>0.099</td><td>0.143</td><td>0.150</td><td>0.048</td><td>0.060</td><td>0.062</td></tr><tr><td>MiniLM-L6-v2</td><td>0.079</td><td>0.112</td><td>0.120</td><td>0.120</td><td>0.175</td><td>0.187</td><td>0.150</td><td>0.202</td><td>0.214</td><td>0.058</td><td>0.084</td><td>0.090</td></tr><tr><td>BioLORD-0.1B</td><td>0.112</td><td>0.164</td><td>0.177</td><td>0.158</td><td>0.249</td><td>0.269</td><td>0.034</td><td>0.125</td><td>0.150</td><td>0.104</td><td>0.146</td><td>0.156</td></tr><tr><td>BioLORD-0.1B†</td><td>0.061</td><td>0.090</td><td>0.099</td><td>0.130</td><td>0.189</td><td>0.202</td><td>0.151</td><td>0.219</td><td>0.232</td><td>0.054</td><td>0.070</td><td>0.075</td></tr><tr><td>BioLORD-0.1B</td><td>0.066</td><td>0.103</td><td>0.113</td><td>0.099</td><td>0.152</td><td>0.163</td><td>0.158</td><td>0.228</td><td>0.241</td><td>0.060</td><td>0.084</td><td>0.088</td></tr><tr><td>MedEmbed-0.1B</td><td>0.095</td><td>0.134</td><td>0.143</td><td>0.148</td><td>0.211</td><td>0.225</td><td>0.155</td><td>0.217</td><td>0.229</td><td>0.079</td><td>0.106</td><td>0.113</td></tr><tr><td>SapBERT-0.1B</td><td>0.096</td><td>0.131</td><td>0.141</td><td>0.104</td><td>0.149</td><td>0.160</td><td>0.142</td><td>0.206</td><td>0.217</td><td>0.068</td><td>0.091</td><td>0.096</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.137</td><td>0.195</td><td>0.210</td><td>0.185</td><td>0.287</td><td>0.306</td><td>0.224</td><td>0.324</td><td>0.341</td><td>0.143</td><td>0.208</td><td>0.222</td></tr><tr><td>Qwen3-Embed-0.6B†</td><td>0.064</td><td>0.089</td><td>0.097</td><td>0.124</td><td>0.182</td><td>0.195</td><td>0.131</td><td>0.193</td><td>0.206</td><td>0.050</td><td>0.064</td><td>0.066</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.089</td><td>0.123</td><td>0.132</td><td>0.104</td><td>0.149</td><td>0.158</td><td>0.139</td><td>0.198</td><td>0.210</td><td>0.066</td><td>0.083</td><td>0.089</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.105</td><td>0.142</td><td>0.152</td><td>0.144</td><td>0.204</td><td>0.215</td><td>0.169</td><td>0.238</td><td>0.251</td><td>0.076</td><td>0.103</td><td>0.109</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.092</td><td>0.132</td><td>0.141</td><td>0.140</td><td>0.198</td><td>0.209</td><td>0.176</td><td>0.247</td><td>0.259</td><td>0.084</td><td>0.110</td><td>0.116</td></tr></table>

Table 10: Tier-2 ontology relation tasks evaluated using MRR@k.

<table><tr><td></td><td colspan="10">Hit@k</td></tr><tr><td></td><td colspan="3">Dis→Gene</td><td colspan="3">Dis→Phen</td><td colspan="3">Phen→Dis</td><td colspan="3">Phen→Gene</td></tr><tr><td>Model</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.076</td><td>0.160</td><td>0.232</td><td>0.084</td><td>0.184</td><td>0.245</td><td>0.099</td><td>0.218</td><td>0.277</td><td>0.048</td><td>0.081</td><td>0.100</td></tr><tr><td>MiniLM-L6-v2</td><td>0.079</td><td>0.172</td><td>0.236</td><td>0.120</td><td>0.273</td><td>0.369</td><td>0.150</td><td>0.295</td><td>0.379</td><td>0.058</td><td>0.128</td><td>0.173</td></tr><tr><td>BioLORD-0.1B</td><td>0.112</td><td>0.258</td><td>0.349</td><td>0.158</td><td>0.419</td><td>0.570</td><td>0.034</td><td>0.342</td><td>0.524</td><td>0.104</td><td>0.225</td><td>0.306</td></tr><tr><td>BioLORD-0.1B†</td><td>0.061</td><td>0.145</td><td>0.213</td><td>0.130</td><td>0.298</td><td>0.397</td><td>0.151</td><td>0.341</td><td>0.435</td><td>0.054</td><td>0.104</td><td>0.143</td></tr><tr><td>BioLORD-0.1B</td><td>0.066</td><td>0.171</td><td>0.248</td><td>0.099</td><td>0.248</td><td>0.334</td><td>0.158</td><td>0.354</td><td>0.449</td><td>0.060</td><td>0.124</td><td>0.159</td></tr><tr><td>MedEmbed-0.1B</td><td>0.095</td><td>0.203</td><td>0.274</td><td>0.148</td><td>0.326</td><td>0.429</td><td>0.155</td><td>0.328</td><td>0.415</td><td>0.079</td><td>0.155</td><td>0.204</td></tr><tr><td>SapBERT-0.1B</td><td>0.096</td><td>0.197</td><td>0.272</td><td>0.104</td><td>0.232</td><td>0.314</td><td>0.142</td><td>0.319</td><td>0.401</td><td>0.068</td><td>0.131</td><td>0.169</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.137</td><td>0.303</td><td>0.410</td><td>0.185</td><td>0.470</td><td>0.611</td><td>0.224</td><td>0.498</td><td>0.622</td><td>0.143</td><td>0.319</td><td>0.422</td></tr><tr><td>Qwen3-Embed-0.6B+</td><td>0.064</td><td>0.140</td><td>0.202</td><td>0.124</td><td>0.286</td><td>0.378</td><td>0.131</td><td>0.302</td><td>0.398</td><td>0.050</td><td>0.088</td><td>0.106</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.089</td><td>0.186</td><td>0.255</td><td>0.104</td><td>0.228</td><td>0.298</td><td>0.139</td><td>0.305</td><td>0.391</td><td>0.066</td><td>0.117</td><td>0.163</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.105</td><td>0.211</td><td>0.288</td><td>0.144</td><td>0.308</td><td>0.395</td><td>0.169</td><td>0.360</td><td>0.456</td><td>0.076</td><td>0.149</td><td>0.192</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.092</td><td>0.204</td><td>0.277</td><td>0.140</td><td>0.302</td><td>0.387</td><td>0.176</td><td>0.372</td><td>0.466</td><td>0.084</td><td>0.167</td><td>0.202</td></tr></table>

Table 11: Tier-2 ontology relation tasks evaluated using Hit@k.

<table><tr><td rowspan="3">Model</td><td colspan="9">PhenTriplet→Dis</td></tr><tr><td colspan="3">nDCG@k</td><td colspan="3">MRR@k</td><td colspan="3">Hit@k</td></tr><tr><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td><td>@1</td><td>@5</td><td>@10</td></tr><tr><td>BM25</td><td>0.066</td><td>0.093</td><td>0.107</td><td>0.066</td><td>0.103</td><td>0.110</td><td>0.066</td><td>0.172</td><td>0.230</td></tr><tr><td>MiniLM-L6-v2</td><td>0.072</td><td>0.106</td><td>0.123</td><td>0.072</td><td>0.110</td><td>0.119</td><td>0.072</td><td>0.180</td><td>0.246</td></tr><tr><td>BioLORD-0.1B</td><td>0.085</td><td>0.196</td><td>0.244</td><td>0.085</td><td>0.185</td><td>0.207</td><td>0.085</td><td>0.368</td><td>0.527</td></tr><tr><td>BioLORD-0.1B†</td><td>0.096</td><td>0.146</td><td>0.171</td><td>0.096</td><td>0.151</td><td>0.163</td><td>0.096</td><td>0.247</td><td>0.340</td></tr><tr><td>BioLORD-0.1B</td><td>0.082</td><td>0.126</td><td>0.151</td><td>0.082</td><td>0.131</td><td>0.143</td><td>0.082</td><td>0.224</td><td>0.312</td></tr><tr><td>MedEmbed-0.1B</td><td>0.083</td><td>0.122</td><td>0.143</td><td>0.083</td><td>0.123</td><td>0.134</td><td>0.083</td><td>0.202</td><td>0.281</td></tr><tr><td>SapBERT-0.1B</td><td>0.076</td><td>0.111</td><td>0.131</td><td>0.076</td><td>0.116</td><td>0.127</td><td>0.076</td><td>0.194</td><td>0.272</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.183</td><td>0.275</td><td>0.313</td><td>0.183</td><td>0.281</td><td>0.297</td><td>0.183</td><td>0.449</td><td>0.572</td></tr><tr><td>Qwen3-Embed-0.6B†</td><td>0.083</td><td>0.134</td><td>0.158</td><td>0.083</td><td>0.135</td><td>0.147</td><td>0.083</td><td>0.231</td><td>0.321</td></tr><tr><td>Qwen3-Embed-0.6B</td><td>0.058</td><td>0.089</td><td>0.109</td><td>0.058</td><td>0.092</td><td>0.102</td><td>0.058</td><td>0.157</td><td>0.234</td></tr><tr><td>Qwen3-Embed-4B</td><td>0.028</td><td>0.060</td><td>0.080</td><td>0.028</td><td>0.061</td><td>0.071</td><td>0.028</td><td>0.123</td><td>0.202</td></tr><tr><td>OpenAI (text-embedding-3-large)</td><td>0.097</td><td>0.143</td><td>0.169</td><td>0.097</td><td>0.149</td><td>0.160</td><td>0.097</td><td>0.244</td><td>0.331</td></tr></table>

Table 12: Performance on the Tier-3 PhenTriplet→Dis retrieval task. We report nDCG, MRR, and Hit at k ∈ {1, 5, 10}. The † symbol identifies models fine-tuned using Tier 1 supervision, while ‡ identifies models jointly fine-tuned using supervision from all three tiers.

## H Error Analysis (Tier 3)

Failure-mode analysis. We analyzed Tier-3 queries for which the embedding model did not rank the ground-truth disease first. For each failure, we computed phenotype overlap between the query and the predicted disease $\mathrm { ( O v _ { p r e d } ) }$ , the rank of the ground-truth disease under the Phenomizer-style Resnik reference $( r _ { \mathrm { R e s n i k } } )$ , and mean query phenotype information content $( \overline { { \mathrm { I C } } } _ { q } )$ . Ontology recovery was defined as $r _ { \mathrm { R e s n i k } } \le 1 0 \colon$ the low-information threshold was $\overline { { \mathrm { I C } } } _ { q } \leq 5 . 9 4 9$ , the 25th percentile among embedding failures.

Failures were assigned sequentially to mutually exclusive categories: generic-bias if $\overline { { \mathrm { I C } } } _ { q } \leq 5 . 9 4 9 $ related-disease confusion if $\mathrm { { O v } _ { p r e d } } = 2$ and $r _ { \mathrm { R e s n i k } } \ \leq \ 1 0 ;$ aggregation failure if $\mathrm { O v } _ { \mathrm { p r e d } } \leq 1$ and $r _ { \mathrm { R e s n i k } } \leq 1 0 ;$ and semantic drift otherwise.

Interpretation. Tier-3 queries are phenotype triplets. Predictions matching two phenotypes are treated as near misses; those matching at most one are aggregation failures only when the ontology reference retrieves the true disease in the top 10, and semantic drift otherwise. Low-information is an operational query-specificity category, not a causal explanation.
<table><tr><td>Mode</td><td>QID</td><td>Phenotypes</td><td>Predicted disease (embed- ding model)</td><td>True disease</td><td>Phenomizer</td><td> $\overline { { \mathrm { ~ O v } _ { \mathrm { p r e d } } } }$ </td><td> $\overline { { \mathrm { \ O v } _ { \mathrm { t r u e } } } }$ </td></tr><tr><td>Aggregation failure</td><td>Q2599</td><td>Sloping forehead; 2-4 toe cutaneous syndactyly; Chiari type II malformation</td><td>Crossed polysyndactyly</td><td>Lathosterolosis</td><td>1</td><td>0</td><td>3</td></tr><tr><td>Aggregation failure</td><td>Q4636</td><td>Finger syndactyly; Ungual fi- broma; Glue ear</td><td>Crossed polysyndactyly</td><td>Choroidal atrophy-alopecia syndrome</td><td>1</td><td>1</td><td>3</td></tr><tr><td>Aggregation failure</td><td>Q5613</td><td>Long philtrum; Conical in- cisor; Hypophosphaturia</td><td>Hypophosphatemic rickets, autosomal recessive, 2</td><td>Global developmental delay- osteopenia-ectodermal de-</td><td>1</td><td>0</td><td>3</td></tr><tr><td>Related disease confusion</td><td>Q4072</td><td>Flat face; Sacral dimple; Prominent fingertip pads</td><td>PDE4D haploinsufficiency syndrome</td><td>fect syndrome Vulto-van Silfout-de Vries syndrome</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Related disease confusion</td><td>Q1943</td><td>Short philtrum; Prominent fingertip pads; Scoliosis</td><td>1q21.1 microdeletion syn- drome</td><td>Global developmental delay-visual anomalies- progressive cerebellar atrophy-truncal hypotonia</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Related disease confusion</td><td>Q2002</td><td>Midface retrusion; Scoliosis; Radial deviation of finger</td><td>Autosomal dominant Robi- now syndrome</td><td>syndrome Acrofacial dysostosis Nager type</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Generic bias</td><td>Q18</td><td>Intellectual disability; Tremor; Tapered finger</td><td>Developmental delay and seizures with or without movement abnormalities</td><td>Agenesis of the corpus callo- sum with peripheral neuropa-</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Generic bias</td><td>Q976</td><td>Abnormal cardiovascular system morphology; Long</td><td>1q21.1 microdeletion syn- drome</td><td>Doors syndrome</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Generic bias</td><td>Q730</td><td>philtrum; Frontal bossing Seizure; Splenomegaly; Pto- sis</td><td>Sialidosis type 2</td><td>Proteus syndrome, somatic</td><td>1</td><td>2</td><td>3</td></tr><tr><td>Semantic drift</td><td>Q6075</td><td>Ptosis; Paraplegia; Brain at- rophy</td><td>Primary lateral sclerosis</td><td>Oculopharyngodistal myopa- thy 1</td><td>12</td><td>0</td><td>3</td></tr><tr><td>Semantic drift</td><td>Q651</td><td>Brachydactyly; Limitation of joint mobility; Abnormal metaphysis morphology</td><td>Metaphyseal dysostosis- intellectual disability- conductive deafness</td><td>Thanatophoric dysplasia type 2</td><td>12</td><td>0</td><td>3</td></tr><tr><td>Semantic drift</td><td>Q2375</td><td>Generalized-onset seizure; Lower limb spasticity; Sta- tus epilepticus</td><td>syndrome Epileptic encephalopathy, early infantile, 6 (Dravet syndrome)</td><td>Multiple congenital anomalies-hypotonia- seizures syndrome 2</td><td>13</td><td>1</td><td>3</td></tr></table>

Table 13: Representative Tier-3 embedding failures grouped by rule-based error mode. The embedding model frequently retrieves partially compatible disorders despite full phenotype coverage by the ground-truth disease, while the Phenomizer baseline often resolves the same queries through ontology-based phenotype aggregation. QID denotes the query identifier. $r _ { \mathrm { P h e n o m i z e r } }$ denotes the ranking position assigned to the ground-truth disease by the ontology-aware Phenomizer baseline. $\mathrm { { O v } _ { \mathrm { { p r e d } } } }$ and $\mathrm { O v } _ { \mathrm { t r u e } }$ denote phenotype overlap counts between the query phenotype set and the predicted or ground-truth disease, respectively.