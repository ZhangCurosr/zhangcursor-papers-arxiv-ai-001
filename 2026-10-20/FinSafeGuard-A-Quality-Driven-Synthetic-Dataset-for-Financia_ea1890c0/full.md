# FinSafeGuard: A Quality-Driven Synthetic Dataset for Financial AI Safety

Joseph Thomas Thacil<sup>1</sup>, Reetu Raj Harsh<sup>1</sup>, Krati Saxena<sup>1</sup>, Bhaskarjit Sarmah<sup>1⋆</sup>, and Stefano Pasquali<sup>2</sup>

<sup>1</sup> Domyn, India

josephthomas.thachil@domyn.com, reeturaj.harsh@domyn.com, krati.saxena@domyn.com, bhaskarjit.sarmah@domyn.com <sup>2</sup> Domyn, USA stefano.pasquali@domyn.com

Abstract. Generating high-quality synthetic datasets at scale remains a fundamental challenge for training specialized AI systems in data-scarce domains. We present a comprehensive methodology that produces 14.3 million quality-filtered synthetic training examples from a structured taxonomy of 632,000 scenario specifications. Our four-stage pipeline addresses critical challenges in synthetic data quality, diversity, and realism through: (1) systematic pre-generation research establishing optimal generation parameters via diminishing returns analysis, (2) conditional autoevolution mechanisms that remediate low-diversity batches during generation, (3) a two-layer semantic deduplication architecture combining lexical and embedding-based methods, and (4) LLM-as-Judge quality evaluation filtering samples based on behavioral correctness and tuple fidelity. Empirical analysis across three domains (Banking, Insurance, Financial Services) demonstrates domain-specific deduplication patterns ranging from 9.9% to 40.6%, with output generation modes exhibiting highest redundancy. Our batch size optimization reveals mode-specific diversity saturation thresholds, enabling principled allocation strategies. The complete pipeline reduces 26.4M raw samples to 14.3M high-confidence examples through merge (20.9% reduction), semantic deduplication (18.7% reduction), and quality-based filtering (15.9% reduction), providing a reproducible framework for domain-specific synthetic data generation with rigorous quality control.

Keywords: Synthetic data generation · Quality evaluation · Diversity metrics · Semantic deduplication · LLM-as-Judge · Financial AI safety

## 1 Introduction

Large language models have demonstrated remarkable capabilities across diverse tasks, yet their deployment in specialized domains requires training data that captures domain-specific risks, edge cases, and regulatory constraints. Financial institutions face unique challenges: real-world safety incidents are sparse, sensitive customer data cannot be shared, and regulatory requirements demand comprehensive coverage of potential failure modes [3, 8].

Synthetic data generation ofers a promising solution, enabling creation of large-scale training datasets without privacy concerns or data scarcity limitations [5, 10]. However, naive scaling—simply generating millions of synthetic examples—produces datasets plagued by redundancy, low diversity, and quality inconsistencies that degrade downstream model performance [6, 11].

This work addresses three fundamental questions in large-scale synthetic data generation:

1. Batch Size Optimization: How many samples should be generated per scenario specification before diversity saturates and duplicates emerge?

2. Quality-Diversity Tradeof: How can we systematically measure and enforce both quality and diversity throughout the generation pipeline?

3. Scalable Deduplication: How can we eficiently detect and remove semantic duplicates across millions of generated samples?

## 1.1 Our Approach

We present a four-stage pipeline that produces 14.3 million quality-filtered synthetic examples for financial AI safety applications across Banking, Financial Services, and Insurance domains. Our methodology combines:

Taxonomy-Driven Generation. A comprehensive safety taxonomy expanded into 632,000 unique scenario tuples across 9 dimensions (domain, category, user sophistication, risk level, framing, channel, attack vectors, generation mode, safety label).

Pre-Generation Research. Systematic experiments establishing optimal data generation process (DGP) parameters, mode-specific batch sizes, and validated diversity metrics before full-scale generation.

Conditional Auto-Evolution. A chained mechanism that automatically detects low-diversity batches during generation and applies Evol-Instruct [10] operators to remediate them, ensuring consistent quality without post-hoc filtering.

Two-Layer Deduplication. Combining lexical (LSH + SimHash) and semantic (FAISS + MiniLM) methods to remove duplicates at sub-quadratic complexity, with domain-specific deduplication rates ranging from 9.9% to 40.6%.

LLM-as-Judge Quality Evaluation. A two-stage assessment framework evaluating behavioral correctness and tuple fidelity, filtering samples with low-confidence labels to produce the final 14.3M training corpus.

## 1.2 Contributions

1. Systematic Pre-Generation Research: Empirical analysis of batch size vs. diversity tradeofs, revealing mode-specific saturation thresholds (input: 20, output: 8, multi-turn: 3) through diminishing returns curves.

2. Validated Diversity Scoring: A 5-component diversity metric (uniqueness, length variety, pattern diversity, vocabulary richness, semantic spread) with Spearman correlations $\rho = 0 . 5 8 – 0 . 8 2$ against human judgments across 4,500 batches.

3. Conditional Auto-Evolution: Chained auto-evolve mechanism applying Evol-Instruct operators only to low-diversity batches (score < 70), reducing raw output by 20.9% while preserving high-quality samples.

4. Domain-Specific Deduplication Analysis: Detailed characterization of redundancy patterns showing Banking output mode exhibits 40.6% duplication (highest) while Financial Services input shows 9.9% (lowest).

5. Reproducible Pipeline: Complete four-stage methodology from 26.4M raw samples to 14.3M quality-filtered examples, with total reduction of 45.8% through principled quality control.

## 1.3 Paper Organization

Section 2 surveys related work. Section 3 describes the taxonomy structure. Section 4 presents pre-generation research. Section 5 details the data generation pipeline. Section 6 explains the two-layer deduplication architecture. Section 7 describes LLM-as-Judge evaluation. Section 8 discusses limitations. Section 9 concludes.

## 2 Related Work

Synthetic Data Generation. Recent work has demonstrated the efectiveness of synthetic data for training language models. WizardLM [10] introduced Evol-Instruct for instruction complexity enhancement. Textbooks Are All You Need [5] showed that high-quality synthetic textbooks can match real data. Phi-3 [2] and Orca 2 [7] scaled synthetic data generation with careful quality control. Our work extends these approaches with systematic pre-generation research and domain-specific deduplication.

Data Quality and Diversity. LIMA [11] demonstrated that 1,000 carefully curated examples outperform datasets orders of magnitude larger, emphasizing quality over quantity. Our diversity scoring system builds on these insights, providing quantitative metrics for batch quality assessment during generation.

Deduplication Methods. Lee et al. [6] showed that deduplication significantly improves language model quality. SemDeDup [1] introduced embedding-based semantic deduplication. Our two-layer architecture combines lexical and semantic methods, achieving sub-quadratic complexity through LSH and FAISS indexing.

AI Safety and Alignment. Constitutional AI [3] and RLHF [8] established frameworks for aligning language models with human values. Red-teaming approaches [4, 9] generate adversarial examples to identify model failures. Our work applies these principles to domain-specific safety in financial services.

## 3 Financial Safety Taxonomy

Our synthetic data generation is driven by a comprehensive financial safety taxonomy covering Banking, Financial Services, and Insurance (BFSI) domains. The taxonomy defines 15 risk categories spanning financial integrity, regulatory compliance, fraud prevention, and domain-specific vulnerabilities.

## 3.1 Taxonomy Structure

Each scenario is specified by a 9-dimensional tuple:

$$
Z = ( d , t , u , k , f , c h , a , m , y )\tag{1}
$$

where:

– d ∈ {Banking, Insurance, Financial Services}: Domain

– t: Taxonomy node (15 categories, 47 sub-categories)

u ∈ {Novice, Intermediate, Expert}: User sophistication

– k ∈ {1, 2, 3}: Risk severity level

– f ∈ {Direct, Indirect, Hypothetical}: Framing style

– ch ∈ {Chat, Email, Voice, Form}: Interaction channel

– a: Binary flags for 8 attack vectors (social engineering, jailbreak, etc.)

m ∈ {Input, Output, Multi-Turn}: Generation mode

– y ∈ {Safe, Unsafe}: Target safety label

This factorization yields approximately 632,000 unique tuples after filtering implausible combinations through a structural causal model encoding realistic dependencies.

## 3.2 Risk Categories

The 15 risk categories include: (1) Financial Integrity & Accuracy, (2) Regulatory Compliance, (3) Fraud & AML Evasion, (4) Unauthorized Trading Advice, (5) Insurance Fraud Coaching, (6) Data Privacy Violations, (7) Market Manipulation, (8) Misrepresentation of Products, (9) Discriminatory Practices, (10) Cybersecurity Vulnerabilities, (11) Conflicts of Interest, (12) Inadequate Risk Disclosure, (13) Churning & Excessive Trading, (14) Ponzi/Pyramid Schemes, and (15) Regulatory Arbitrage.

## 4 Pre-Generation Research

Before scaling to full dataset generation across 632K tuples, we conduct systematic research to establish optimal parameters for the data generation process (DGP), batch size per tuple, and diversity scoring thresholds.

## 4.1 Data Generation Process Design

We formalize synthetic data generation as a factorized generative model. Let $Z _ { \mathrm { s c e n a r i o } } = ( d , t , u , k , f , c h , a )$ denote the 7 scenario dimensions of the full tuple (Section 3), with generation mode m and target label y treated as conditioning parameters rather than latent variables. Each sample is produced by:

$$
Z _ { \mathrm { s c e n a r i o } } \sim P ( Z _ { \mathrm { s c e n a r i o } } )\tag{2}
$$

$$
x \sim P ( x \mid Z _ { \mathrm { s c e n a r i o } } , m )\tag{3}
$$

$$
( y , c , s ) \sim P ( y , c , s \mid x , Z _ { \mathrm { s c e n a r i o } } )\tag{4}
$$

where $x$ is the generated prompt/conversation, y is the safety label, $c$ is the taxonomy category, and s is the severity level. The LLM approximates $P ( x \mid$ $Z _ { \mathrm { s c e n a r i o } } , m \in$ , while labels are either specified by the tuple (for targeted generation) or judged post-hoc.

Temperature Sweep. We evaluate generation quality across temperature settings $T \in \{ 0 . 5 , 0 . 7 , 0 . 8 , 0 . 9 , 1 . 0 \}$ , measuring lexical diversity (Type-Token Ratio), semantic coherence, financial terminology accuracy, and schema compliance rate. Figure 1 shows that $T = 0 . 9$ provides optimal balance between creativity and coherence.

## 4.2 Batch Size Optimization

A batch is the set of outputs generated from the same fixed input tuple in a single run. The batch size N determines how many outputs we generate per tuple before moving to the next. This is critical because for any fixed input X, the LLM has finite conditional entropy $H ( Y | X )$ —it can only produce so many meaningfully diferent responses.

Theoretical Foundation. Define a diversity functional $D ( \cdot )$ measuring variety in a set of outputs. The optimal batch size $N ^ { * }$ is the smallest N such that:

$$
\mathbb { E } [ D ( N + 1 ) - D ( N ) ] \le \epsilon\tag{5}
$$

where $\epsilon$ is our tolerance threshold. Diversity growth is sublinear because the LLM’s conditional output space is bounded. We model this as:

$$
D ( n ) = D _ { \mathrm { m a x } } \cdot \left( 1 - e ^ { - \lambda n } \right)\tag{6}
$$

![](images/fcf76b78c9168b066d090d35f420c4e073e9e91735b57045f0c72adfd39eb7d5.jpg)  
Fig. 1. Impact of temperature on generation quality metrics. Higher temperatures increase lexical diversity but reduce schema compliance beyond $T > 0 . 9$ . The production pipeline uses $T = 0 . 9$ as the optimal balance.

where $D _ { \mathrm { m a x } }$ is the asymptotic diversity and λ is the mode-specific diversity growth rate. This idealized model captures the saturation behavior: diversity grows rapidly at small N then plateaus as the LLM’s conditional output space is exhausted. Higher-complexity modes saturate faster because generation constraints— coherence across turns, response-query consistency—naturally limit the space of meaningfully diferent outputs.

Experimental Setup. We sample 100 random tuples and generate outputs at $N \in \{ 5 , 1 0 , 1 5 , 2 0 , 5 0 , 1 0 0 \}$ samples per tuple. For each configuration, we measure per-batch diversity, within-tuple similarity, cross-tuple overlap, and generation eficiency.

Diminishing Returns Analysis. In practice, the empirical diversity curves (Figure 2) exhibit a sharper transition than the smooth saturation of Equation (5): diversity remains high $( > 9 0 )$ up to mode-specific thresholds, then drops of as near-duplicate rates spike—a clif-like behavior that the idealized exponential model understates. This motivates selecting batch sizes conservatively, just before the empirical drop-of. The marginal diversity loss follows:

$$
\mathrm { M a r g i n a l D i v e r s i t y } ( N ) = \frac { \partial D } { \partial N } \propto - N ^ { - \alpha } , \quad \alpha \approx 0 . 6\tag{7}
$$

Mode-Specific Batch Sizes. Based on these experiments, we select:

– Input mode: 20 samples/tuple (15 unsafe + 5 safe)

– Output mode: 8 samples/tuple (5 unsafe + 3 safe)

– Multi-turn mode: 3 samples/tuple (2 unsafe + 1 safe)

![](images/3bb2b019d964ab4f1ee986488df24d4f80c99401616d74d369691231a94061ce.jpg)

![](images/b4be0dc02fc53f6a3acfa3e33e75e97d7ec1da895f7f0ef96a78f9eb93ad2d52.jpg)  
Fig. 2. Batch diversity score as a function of batch size per tuple (log scale). Diversity remains high (> 90) up to each mode’s capacity threshold, then drops as the LLM exhausts its conditional output space. Multi-turn saturates earliest (N ≈ 5), output at $N \approx 1 2 ,$ and input at N ≈ 25. Selected batch sizes (stars) sit just before the empirical drop-of. Right panel shows marginal diversity loss, which spikes sharply at the transition point.

Table 1. Five-Component Diversity Scoring System (100 points)
<table><tr><td>Component</td><td>Points</td><td>Method</td><td>Detects</td></tr><tr><td>Uniqueness</td><td>0-30</td><td> $\mathrm { M D 5 } + \mathrm { S i m H a s h }$ </td><td>Exact and near-duplicates</td></tr><tr><td>Length Variety</td><td>0-20</td><td> $\mathrm { C V } = \sigma / \mu$ </td><td>Template-based generation</td></tr><tr><td>Pattern Diversity</td><td>0-20</td><td></td><td>Multi-position intro Repetitive openings</td></tr><tr><td>Vocabulary Richness</td><td>0-15</td><td></td><td>Type-Token Ratio Vocabulary collapse</td></tr><tr><td>Semantic Spread</td><td>0-15</td><td>Embedding distance Semantic similarity</td><td></td></tr></table>

## 4.3 Diversity Scoring Validation

We develop a 5-component diversity scoring system (100 points total) that evaluates batch quality during generation. Table 1 summarizes the components.

Validation Against Human Ratings. We validate these metrics against human diversity ratings on 4,500 batches. Table 2 reports Spearman rank correlations.

Figure 3 shows the distribution of diversity scores, with thresholds at 70 (triggers auto-evolve) and 80 (high-quality batches accepted without modification).

## 5 Data Generation Pipeline

Our pipeline transforms 632K taxonomy tuples into 14.3M quality-filtered samples through four stages: pre-generation research (establishing optimal parameters), conditional generation with auto-evolution (producing 26.4M raw samples, merged to 20.9M), two-layer semantic deduplication (reducing to 16.9M), and LLM-as-Judge quality filtering (reducing to 14.3M). The first two stages produce and refine samples; the latter two are filtering stages that remove duplicates and low-quality samples respectively.

Batch Diversity Score Distribution with Quality Thresholds  
Table 2. Diversity Metric Validation Against Human Ratings
<table><tr><td>Metric</td><td>Spearman ρ Weight</td><td></td></tr><tr><td>Uniqueness (MD5+SimHash)</td><td>0.82</td><td>30%</td></tr><tr><td>Length Variety (CV)</td><td>0.61</td><td>20%</td></tr><tr><td>Pattern Diversity</td><td>0.73</td><td>20%</td></tr><tr><td>Vocabulary Richness (TTR)</td><td>0.58</td><td>15%</td></tr><tr><td>Semantic Spread (Embeddings)</td><td>0.77</td><td>15%</td></tr></table>

![](images/5c1dc687b20f494c0a553befd20920545062183504442b7aae5d400e9e7fd489.jpg)  
Fig. 3. Distribution of diversity scores across 4,500 human-validated batches. Threshold at 70 triggers auto-evolve regeneration; threshold at 80 marks high-quality batches suitable for training without modification.

## 5.1 Sample Allocation Strategy

Given T taxonomy tuples, the total raw sample count is:

$$
N _ { \mathrm { r a w } } = T \times \sum _ { m \in \mathcal { M } } \left( n _ { \mathrm { u n s a f e } } ^ { ( m ) } + n _ { \mathrm { s a f e } } ^ { ( m ) } \right) = T \times ( 2 0 + 8 + 3 ) = 3 1 T\tag{8}
$$

With $T \approx 6 3 2 { , } 0 0 0$ unique tuples, this yields approximately 19.6M base samples. Including Evol-Instruct augmentation and auto-evolved variants, the raw SDG output totals 26,422,384 samples (Bank: 9,224,426; Insurance: 8,535,183; Financial Services: 8,662,775).

The allocation strategy is governed by three principles:

Complexity-Adjusted Allocation. Generation cost increases super-linearly with mode complexity. Input mode generates only user prompts; output mode generates prompt + response; multi-turn generates k coherent turns with quadratic context growth.

Diminishing Marginal Diversity. Beyond a threshold, additional samples yield diminishing diversity returns. Our batch size experiments confirm this: input mode plateaus at $n \approx 2 0$ , output at $n \approx 1 0$ , multi-turn at $n \approx 5$

Unsafe-Heavy Ratio. We generate more unsafe than safe samples per tuple (3:1 for input, 5:3 for output, 2:1 for multi-turn) because safety classifiers need rich coverage of violation patterns.

## 5.2 Auto-Evolve Process

A key innovation is the chained auto-evolve mechanism, which automatically detects and remediates low-diversity batches during generation.

Trigger Condition. After generating the initial batch for each (tuple, mode) pair, we compute the 5-component diversity score. If the score falls below 70, the batch is flagged for evolution-based retry:

$$
\mathrm { A u t o E v o l v e ( t u p l e , } m ) = \nVdash [ D ( \mathrm { b a t c h ) < 7 0 } ]\tag{9}
$$

Evolution Operators. Retry batches are regenerated using Evol-Instruct [10] evolution. The evolution applies operators including: add constraints, deepen reasoning, concretize abstractions, increase complexity, and complicate inputs.

Sample Replacement. Evolved samples are marked with an auto evolved replacement flag. During the merge step, records are grouped by (tuple id, mode, target label). For each group, if evolved replacements exist, they take priority over originals.

Merge Impact. The merge step reduces 26,422,384 raw samples to 20,909,187 (20.9% reduction, removing ∼5.5M records). This reduction comes from: (1) discarding original samples superseded by auto-evolved replacements, and (2) resolving overlap from parallel generation shards.

## 6 Two-Layer Deduplication

After merging, we perform global deduplication to remove semantically similar content generated from related taxonomy tuples. Prior work has demonstrated that deduplication significantly improves language model quality [1, 6].

## 6.1 Deduplication Architecture

Our pipeline implements two complementary layers:

Table 3. Deduplication Impact by Domain and Mode (FAISS, cosine ≥ 0.85)
<table><tr><td>Domain</td><td>Mode</td><td>Pre-Dedup Post-Dedup % Removed</td><td></td><td></td></tr><tr><td>Banking</td><td>Input</td><td>4,337,957</td><td>3,351,116</td><td>22.7%</td></tr><tr><td>Banking</td><td>Output</td><td>1,739,168</td><td>1,032,734</td><td>40.6%</td></tr><tr><td>Banking</td><td>Multi-Turn</td><td>1,083,563</td><td>822,789</td><td>24.1%</td></tr><tr><td>Insurance</td><td>Input</td><td>4,166,222</td><td>3,717,065</td><td>10.8%</td></tr><tr><td>Insurance</td><td>Output</td><td>1,666,469</td><td>1,284,466</td><td>22.9%</td></tr><tr><td>Insurance</td><td>Multi-Turn</td><td>1,037,367</td><td>883,066</td><td>14.9%</td></tr><tr><td>Financial Svcs Input</td><td></td><td>4,172,360</td><td>3,760,956</td><td>9.9%</td></tr><tr><td>Financial Svcs Output</td><td></td><td>1,668,864</td><td>1,252,687</td><td>24.9%</td></tr><tr><td>Financial Svcs Multi-Turn</td><td></td><td>1,037,217</td><td>889,736</td><td>14.2%</td></tr><tr><td colspan="4">Total</td><td>18.7%</td></tr></table>

Layer 1: LSH + SimHash (Lexical). Locality-Sensitive Hashing enables subquadratic lexical duplicate detection. Each sample’s text is converted to a 64-bit SimHash fingerprint, split into 8 bands, and bucketed. Candidate pairs with Hamming distance ≤ 5 are confirmed as duplicates. Complexity: O(N) for bucketing, O(|candidates|) for verification.

Layer 2: FAISS + MiniLM (Semantic). To catch paraphrases missed by lexical methods, we apply embedding-based semantic deduplication. Each sample is encoded into a 384-dimensional vector using all-MiniLM-L6-v2. We build a <sub>FAISS IndexIVFFlat with</sub> √<sub>N Voronoi cells for approximate nearest neighbor</sub> search. Pairs with cosine similarity ≥ 0.85 are flagged as semantic duplicates. Complexity: O(N) for embedding, O(N log N) for FAISS index construction and search.

Layer Interaction. The semantic layer (FAISS) subsumes the lexical layer: any text pair with near-identical surface form will necessarily have high embedding similarity. The semantic layer additionally catches paraphrases that lexical methods miss entirely, providing the bulk of the deduplication signal.

## 6.2 Deduplication Results

Table 3 presents the deduplication impact by domain and mode.

## 6.3 Domain and Mode Analysis

Output Mode Exhibits Highest Duplication. Output mode shows 22.9–40.6% deduplication across all domains, significantly higher than input (9.9–22.7%) or multi-turn (14.2–24.1%). This aligns with theoretical analysis: assistant responses to similar queries tend to converge on standard refusal patterns or compliance templates. Banking output exhibits the highest rate (40.6%), likely due to regulatory response templates.

Domain-Specific Patterns. Banking shows higher duplication than Insurance and Financial Services across all modes. Banking tuples share more overlapping terminology (account, transaction, transfer), leading to convergent outputs. Financial Services input shows the lowest duplication (9.9%), reflecting greater lexical diversity in financial services prompts.

Multi-Turn Moderate Duplication. Multi-turn conversations show 14.2–24.1% deduplication. The additional context from multiple turns creates more unique surface forms even when underlying intent overlaps, reducing duplicate detection rates.

Paraphrase Detection Examples. The semantic layer catches paraphrases that lexical methods miss:

“How can I transfer funds internationally?” ↔ “What’s the process for overseas wire transfers?”

– “Help me minimize my tax obligations” ↔ “What strategies reduce my tax burden legally?”

## 7 LLM-as-Judge Quality Evaluation

After deduplication, all 16.9M samples undergo LLM-as-Judge quality evaluation to assess behavioral correctness and tuple fidelity. Samples failing quality thresholds are filtered, reducing the dataset from 16.9M to the final 14.3M training corpus.

## 7.1 Two-Stage Evaluation Framework

Stage 1: Behavioral Assessment. The judge LLM evaluates whether the generated sample exhibits the intended safety behavior specified by the target label. For unsafe samples, this means checking if the content genuinely violates safety guidelines. For safe samples, this verifies appropriate handling of the scenario. The judge outputs a binary assessment (PASS/FAIL) with verbalized confidence (0–100).

Stage 2: Fidelity Checking. The judge evaluates whether the sample accurately reflects the constraints specified in the taxonomy tuple: correct domain, appropriate risk level, matching framing style, suitable for the specified channel, and consistent with user sophistication. This produces a fidelity score (0–5) with group-wise pass/fail indicators.

Confidence Delta. For each sample, we compute:

$$
\begin{array} { r } { \varDelta _ { \mathrm { c o n f } } = \mathrm { c o n f } _ { \mathrm { s t a g e 2 } } - \mathrm { c o n f } _ { \mathrm { s t a g e 1 } } } \end{array}\tag{10}
$$

Positive $\varDelta _ { \mathrm { c o n f } }$ indicates the fidelity check increased confidence in the behavioral assessment. Negative $\varDelta _ { \mathrm { c o n f } }$ suggests potential misalignment between content and tuple specification, flagging samples for downstream review.

## 7.2 Quality Labels for Downstream Filtering

The judge produces a confidence score and fidelity score for each sample. We apply empirically determined thresholds to filter the dataset into three regions:

– High-confidence training: Filter to samples with conf > 70 and $\varDelta _ { \mathrm { c o n f } } \geq 0$ – Balanced training: Include samples with $5 0 \leq \mathrm { c o n f } \leq 7 0$ and fidelity $\geq 3$ – Edge case focus: Target samples with conf $< 5 0$ for exclusion from training

This three-region approach reduces the post-deduplication dataset from 16.9M to 14.3M samples (16.5% reduction). The filtering disproportionately removes low-quality safe-labeled samples, shifting the label distribution slightly toward unsafe (75.3% vs. 72.8% pre-filtering).

## 8 Limitations and Future Work

Domain Scalability. Our methodology is tailored to financial AI safety. Extending to healthcare, legal, or government domains requires domain-specific taxonomies and validation of diversity metrics against domain expert ratings.

Multilingual Extension. The current pipeline generates English-only data. Global deployment requires multilingual generation with jurisdiction-specific regulatory constraints and cultural context.

Computational Cost Optimization. Generating 26.4M samples, computing 16.9M embeddings for deduplication, and running LLM-as-Judge evaluation on 16.9M samples is computationally expensive. Future work should explore adaptive batch sizing, hierarchical deduplication, and distilled judge models.

Alternative Deduplication Methods. Our FAISS-based approach uses fixed cosine similarity thresholds. Cluster-based methods, diversity-aware sampling, or learned duplicate classifiers may improve precision-recall tradeofs.

Quality Scoring Improvements. Single-judge evaluation may miss subtle quality issues. Multi-judge ensembles, calibrated confidence scoring, and learned fidelity models could enhance quality assessment.

Temporal Dynamics. Financial regulations evolve over time. Continuous learning systems that track regulatory changes and update the taxonomy accordingly would maintain dataset relevance.

Downstream Evaluation. While we validate diversity metrics against human ratings, comprehensive evaluation requires training models on this data and measuring downstream task performance, conducting ablation studies on pipeline components, and deploying in real-world financial AI systems.

Explainability. Current LLM-as-Judge assessments provide limited reasoning. Structured explanations, counterfactual examples, and audit trails would improve transparency for regulatory compliance.

Bias and Fairness. Synthetic data may inherit biases from the generation model. Systematic auditing for demographic bias, fairness constraints in generation, and alignment with regulatory fairness requirements are essential for production deployment.

## 9 Conclusion

We presented a comprehensive methodology for large-scale synthetic dataset generation, producing 14.3 million quality-filtered training examples from a structured taxonomy of 632,000 scenario specifications. Our four-stage pipeline addresses critical challenges in synthetic data quality, diversity, and realism through systematic pre-generation research, conditional auto-evolution, two-layer semantic deduplication, and LLM-as-Judge quality filtering.

Key contributions include: (1) empirical batch size optimization revealing mode-specific diversity saturation thresholds, (2) validated 5-component diversity scoring with strong correlation to human judgments, (3) conditional autoevolve mechanism reducing raw output by 20.9%, (4) domain-specific deduplication analysis showing 9.9–40.6% redundancy patterns, and (5) complete pipeline reducing 26.4M raw samples to 14.3M high-confidence examples through principled quality control (45.8% total reduction).

Our methodology provides a reproducible framework for domain-specific synthetic data generation with rigorous quality control, applicable to any specialized domain requiring large-scale training data with comprehensive coverage of edge cases and failure modes.

## References

1. Abbas, A., Tirumala, K., Simig, D., Ganguli, S., Morcos, A.S.: Semdedup: Dataeficient learning at web-scale through semantic deduplication. arXiv preprint arXiv:2303.09540 (2023)

2. Abdin, M., Jacobs, S.A., Awan, A.A., Aneja, J., Awadalla, A., Awadallah, H., Bach, N., Bahree, A., Bakhtiari, A., Behl, H., et al.: Phi-3 technical report: A highly capable language model locally on your phone. arXiv preprint arXiv:2404.14219 (2024)

3. Bai, Y., Kadavath, S., Kundu, S., Askell, A., Kernion, J., Jones, A., Chen, A., Goldie, A., Mirhoseini, A., McKinnon, C., et al.: Constitutional ai: Harmlessness from ai feedback. arXiv preprint arXiv:2212.08073 (2022)

4. Ganguli, D., Lovitt, L., Kernion, J., Askell, A., Bai, Y., Kadavath, S., Mann, B., Perez, E., Schiefer, N., Ndousse, K., et al.: Red teaming language models to reduce harms: Methods, scaling behaviors, and lessons learned. arXiv preprint arXiv:2209.07858 (2022)

5. Gunasekar, S., Zhang, Y., Aneja, J., Mendes, C.C.T., Del Giorno, A., Gopi, S., Javaheripi, M., Kaufmann, P., de Rosa, G., Saarikivi, O., et al.: Textbooks are all you need. arXiv preprint arXiv:2306.11644 (2023)

6. Lee, K., Ippolito, D., Nystrom, A., Zhang, C., Eck, D., Callison-Burch, C., Carlini, N.: Deduplicating training data makes language models better. arXiv preprint arXiv:2107.06499 (2022), aCL 2022

7. Mitra, A., Del Corro, L., Mahajan, S., Codas, A., Simoes, C., Agarwal, S., Chen, X., Razdaibiedina, A., Jones, E., Aggarwal, K., et al.: Orca 2: Teaching small language models how to reason. arXiv preprint arXiv:2311.11045 (2023)

8. Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., et al.: Training language models to follow instructions with human feedback. Advances in Neural Information Processing Systems 35, 27730–27744 (2022)

9. Perez, E., Huang, S., Song, F., Cai, T., Ring, R., Aslanides, J., Glaese, A., McAleese, N., Irving, G.: Red teaming language models with language models. arXiv preprint arXiv:2202.03286 (2022)

10. Xu, C., Sun, Q., Zheng, K., Geng, X., Zhao, P., Feng, J., Tao, C., Jiang, D.: Wizardlm: Empowering large language models to follow complex instructions. arXiv preprint arXiv:2304.12244 (2023), accepted at ICLR 2024

11. Zhou, C., Liu, P., Xu, P., Iyer, S., Sun, J., Mao, Y., Ma, X., Efrat, A., Yu, P., Yu, L., et al.: Lima: Less is more for alignment. arXiv preprint arXiv:2305.11206 (2023), neurIPS 2023