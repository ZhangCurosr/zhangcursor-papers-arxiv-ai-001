# Neurosymbolic Alignment for Physiologically-Safe Clinical Language Models

Abdulhady Abas Abdullah, Erik Cambria, and Milena Zivkovi<sup>ˇ</sup> c´

Abstract—Objective: Clinical LLMs can generate recommendations that are factually plausible yet physiologically unsafe. We investigate whether safety alignment can be improved by grounding preference optimization in structured physiological knowledge rather than in text-only supervision. Methods: We propose Neurosymbolic Alignment, a training-time framework that couples a 7B clinical LLM with an HGNN-based Physiological World Model over an 847K-node biomedical knowledge graph. Candidate responses are scored using homeostatic constraints, multi-hop path plausibility, and drug-interaction penalties, and the resulting rankings drive iterative on-policy ORPO updates. Evaluation is performed on the Clinical Safety Benchmark (CSB), a 2,500-scenario benchmark for physiological constraint violations in generative clinical reasoning. Results: Relative to ORPO, the proposed method improves CSS from 69.5% to 90.8% (+21.3 pp), reduces physician-evaluated HR from 14.1% to 5.1% on the blinded subset, and improves DID from 72.8% to 91.6%. These gains are corroborated by an HGNNindependent Rule-Engine Safety Score (RSS: 86.4%, +21.2 pp over ORPO; r=0.97 concordance with CSS). The method also exceeds GPT-4 (5-shot) on all safety metrics despite a 10× parameter disadvantage, and outperforms an inference-time selfcorrection pipeline (SFT+SelfCorrect) by 11.4 pp CSS. Under synthetic EHR-style noise, 84.2% CSS is retained. Ablation analysis shows that HGNN scoring (−16.2 pp) and iterative training (−11.5 pp) are the dominant contributors. PhysioScore calibration against 200 clinician labels yielded ECE = 0.038 and κ = 0.91. Conclusion: Training-time physiological grounding produces measurable and independently verifiable safety improvements in open-weight clinical LLMs under controlled evaluation. External validation on real clinical data is required to determine whether these gains transfer to deployment settings.

Index Terms—clinical safety, clinical language models, knowledge graphs, neurosymbolic AI, physiological constraints, drug– drug interactions.

## I. INTRODUCTION

ARGE language models (LLMs) are increasingly studied [3]. Despite strong fluency and broad medical knowledge, current systems can produce recommendations that are clinically plausible yet physiologically unsafe, including harmful drug combinations and homeostatic violations [4]–[6]. In safetycritical settings, such failures are consequential because they are often expressed with high confidence.

A key challenge is that many unsafe outputs are not strictly factual errors. A response may contain correct biomedical facts while still proposing an action that is incompatible with the implied patient state, for example by worsening hyperkalemia or ignoring context-dependent contraindications. Therefore, factual correctness alone is insufficient for clinical safety. Current mitigation strategies remain incomplete for this setting. Retrieval-augmented generation improves evidence access, and medical fine-tuning improves terminology and style, but neither directly enforces physiological consistency in multi-step recommendations. Preference optimization methods (DPO/ORPO/KTO/SimPO) are scalable alternatives to rewardmodel pipelines [7]–[10]; however, in medicine their effectiveness is constrained by the availability and quality of preference supervision for subtle relational violations.

To address this gap, we introduce a neurosymbolic training framework that derives preference signals from an explicit physiological world model. We construct a heterogeneous biomedical knowledge graph and train an HGNN-based feasibility scorer that combines homeostatic constraints with multihop relational evidence [11]–[14]. Candidate responses are scored for physiological plausibility, converted into preference pairs, and used for iterative on-policy ORPO updates. The design objective is to improve safety during training while keeping deployment simple: the HGNN scorer is used only during alignment, and inference uses the aligned LLM alone. This avoids graph-based runtime verification at deployment, at the cost of higher alignment-time compute due to iterative candidate generation and graph-based scoring.

Figure 1 summarizes the pipeline, from clinical query to candidate generation, graph-based feasibility scoring, and iterative policy updating. Key contributions are as follows:

1) Physiology-grounded preference generation. We introduce a framework that derives structured preference signals from an HGNN-based Physiological World Model by combining homeostatic boundary constraints, multi-hop path plausibility, and drug-interaction penalties, thereby enabling scalable safety alignment without clinician-authored pair annotations.

2) Iterative on-policy alignment. We present Iterative ORPO, an on-policy alignment scheme that regenerates preference pairs from the current policy at each iteration, reducing distributional staleness and surfacing progressively subtler constraint violations as coarser failures are resolved.

![](images/8e27d3d7f646624b49870dcb2230b8f3e49349d94b9e55fedfafddf75c171af4.jpg)  
Fig. 1. Training-time neurosymbolic pipeline. The base LLM samples candidate clinical responses, the Physiological World Model scores their feasibility, and Iterative ORPO updates the policy. At inference, only the aligned LLM is used.

3) Clinical Safety Benchmark. We release CSB, a controlled 2,500-scenario benchmark for probing physiological constraint violations in generative clinical reasoning, with standardized splits, severity labels, and reproducibility controls.

4) Empirical evidence under controlled evaluation. Across 14 baselines spanning preference optimization, retrieval augmentation, rule-based guardrails, inferencetime self-correction, and proprietary RLHF models, training-time physiological grounding achieves: (i) CSS 90.8% and RSS 86.4%, the highest on both HGNNbased and HGNN-independent safety metrics; (ii) physician-evaluated HR of 5.1% (−9.0 pp vs. ORPO; −4.7 pp vs. DrugBank-Rule); (iii) DID 91.6% (+18.8 pp vs. ORPO); (iv) superiority over GPT-4 (5-shot) on all safety metrics despite 10× fewer parameters; (v) 84.2% CSS under combined synthetic EHR-style noise; and (vi) consistent ablation support, with HGNN scoring (−16.2 pp) and iterative training (−11.5 pp) as dominant contributors. These results are demonstrated on the CSB benchmark and do not extend to real clinical data, which remains untested (see Section VI-B).

We frame this work as physiological alignment for generative clinical reasoning. In this setting, factual correctness and physiological safety are complementary but distinct: a statement may be factually correct yet unsafe for the patientspecific context implied by the prompt. Our method targets this second objective by producing structured, patient-stateaware preference signals from a physiological world model rather than relying on text-level plausibility alone.

## II. CLINICAL SAFETY BENCHMARK (CSB)

## A. Motivation and Scope

CSB targets a practical failure mode of clinical language models: generating plausible recommendations that violate basic physiology (e.g., unsafe drug combinations, contraindications, or homeostatic boundary violations). The benchmark is designed to stress-test safety under realistic clinical ambiguity while remaining fully reproducible and free of patientidentifying information.

## B. Scenario Construction and Annotation

CSB consists of 2,500 synthetically generated scenarios authored by domain experts based on textbook patterns and pharmacological principles. Each scenario is expressed as a natural-language prompt with structured metadata (e.g., risk category and specialty tag); where applicable, scenarios include gold contraindication cues (e.g., comorbidities, labs, or concurrent medications) to support error analysis.

## C. Splits, Coverage, and Intended Use

We provide standardized train/validation/test splits (1,750/250/500) and stratification across risk categories (Table XIX). CSB is intended for research on evaluation,

calibration, and alignment methods for clinical safety; it is not a clinical decision-support tool and must not be used to provide medical advice.

## III. REFERENCE BASELINES AND SCORING

Figure 2 shows the end-to-end pipeline used in our neurosymbolic alignment framework.

Figure 2 summarizes the training-time neurosymbolic alignment loop. In Stage I, the LLM policy $\pi _ { \theta }$ samples $K { = } 8$ candidate responses per query. In Stage II, we extract and link medical entities (scispaCy→UMLS) and score each candidate with the HGNN over the Physiological World Model $\mathcal { G }$ to obtain feasibility scores $s _ { k } ~ \in ~ [ 0 , 1 ]$ . In Stage III, HGNNderived rankings induce preference pairs for Iterative ORPO updates; at inference time, the final policy $\pi _ { \theta ^ { \ast } }$ runs without the HGNN.

a) Notation convention.: For consistency, we use $\mathcal { M } _ { \theta }$ for the parametric language model and $\pi _ { \boldsymbol { \theta } } ( \cdot \mid q )$ for its conditional generation policy on query q. Generated outputs are denoted by r (or $r _ { i }$ for indexed candidates). This convention is used throughout algorithms, metric definitions, and ablations.

Algorithm 1 summarizes the iterative alignment loop; the detailed HGNN scoring pseudocode is moved to Appendix C. We use a linear curriculum $\lambda ^ { ( t ) } = \lambda _ { 0 } ( 1 + \alpha t )$ because it was stable in practice and consistently outperformed constant-λ variants.

b) HGNN scorer.: The physiological scorer uses a 4- layer relation-aware HGNN with type-specific projections and relation-specific message passing, initialized by link-prediction pre-training on G. Candidate responses are processed with scispaCy-based NER and UMLS/KG linking; low-confidence mentions are dropped, and negated entities are excluded from the positive set ${ \mathbf V } _ { r }$ used for scoring. Homeostatic terms $f _ { h } ( { \mathbf V } _ { r } )$ are fixed soft-boundary functions from clinical reference ranges, path plausibility is computed over multi-hop paths up to length 3,

$$
\mathrm { P a t h S c o r e } ( u , v ) = \sum _ { P \in \mathcal { P } _ { \leq 3 } ( u , v ) } \prod _ { ( a , r , b ) \in P } \mathbf { h } _ { a } ^ { \top } \mathbf { W } _ { r } \mathbf { h } _ { b } .\tag{1}
$$

and the interaction term is

$$
P _ { \mathrm { i n t } } = \operatorname* { m a x } _ { ( d _ { i } , d _ { j } ) \in \mathrm { D r u g s } ( r ) } \mathrm { I n t e r a c t i o n S e v e r i t y } ( d _ { i } , d _ { j } , \mathcal { G } ) \in [ 0 , 1 ] ,\tag{2}
$$

with severity derived from curated DDI relations. The final score is

$$
s = S _ { \mathrm { h o m } } \cdot \sigma ( S _ { \mathrm { p a t h } } ) \cdot ( 1 - P _ { \mathrm { i n t } } )\tag{3}
$$

where $S _ { \mathrm { h o m } }$ acts as the dominant safety gate, $P _ { \mathrm { i n t } }$ captures explicit drug-risk penalties, and $S _ { \mathrm { p a t h } }$ adds relational coherence. The multiplicative combination was chosen because unsafe scores on any single component should dominate (a drug interaction penalty near 1 drives the product toward 0 regardless of path plausibility). Alternative combination functions are compared in Appendix C, Table XI. Componentdrop and calibration diagnostics are reported in Appendix C.

Algorithm 1 Iterative ORPO for Physiological Grounding   
Require: Language model $\mathcal { M } _ { \theta } .$ , knowledge graph ${ \overline { { \mathcal { G } } } } ,$ training   
queries Q, iterations $T$   
Ensure: Aligned parameters $\theta ^ { * }$   
1: Initialize HGNN scorer ${ \mathrm { H G N N } } _ { \phi }$ on $\mathcal { G }$   
2: for $t = 1$ to T do   
3: $\lambda ^ { ( t ) }  \lambda _ { 0 } ( 1 + \alpha t )$   
4: $\mathcal { D } ^ { ( t ) }  \emptyset$   
5: for each query $q \in \mathcal { Q }$ do   
6: Sample candidates $\{ r _ { k } \} _ { k = 1 } ^ { K } \sim \pi _ { \theta ^ { ( t - 1 ) } } ( \cdot \mid q )$   
7: for $k = 1$ to K do   
8: s<sub>k</sub> ← PhysioScore $( r _ { k } , \mathcal { G } , \phi )$   
9: end for   
10: $( r _ { w } , r _ { l } ) \gets \mathrm { S e l e c t P a i r } ( \{ r _ { k } , s _ { k } \} _ { k = 1 } ^ { K } )$   
11: ${ \mathcal { D } } ^ { ( t ) } \gets { \mathcal { D } } ^ { ( t ) } \cup \{ ( q , r _ { w } , r _ { l } ) \}$   
12: end for   
13: Optimize $\mathcal { L } _ { \mathrm { S F T } } + \lambda ^ { ( t ) } \mathcal { L } _ { \mathrm { O R } } + \gamma \mathcal { L } _ { \mathrm { K G } }$   
14: end for   
15: return $\theta ^ { ( T ) }$

## IV. DATASET CHARACTERIZATION AND EVALUATION PROTOCOL

## A. Datasets

We evaluate on a multi-benchmark suite spanning clinical reasoning, safety-critical contraindication handling, and domain-specific medical knowledge. Table XVIII reports dataset sizes, split details, and task characteristics.

MedQA-USMLE: A set of 12,723 USMLE-style multiplechoice questions requiring reasoning over pathophysiology, pharmacology, diagnosis, and management. Items cover all three USMLE steps and range from foundational biomedical knowledge to complex clinical vignette reasoning.

PubMedQA: Comprising 1,000 expert-annotated biomedical questions derived from PubMed abstracts, testing the model’s ability to synthesize clinical evidence and make evidence-based judgments.

MedMCQA: A large-scale medical multiple-choice dataset with 193,155 questions covering 2,400 healthcare topics across 21 medical subjects, providing comprehensive coverage of medical knowledge.

MMLU-Medical: The medical subset of Massive Multitask Language Understanding benchmark, including Clinical Knowledge, Medical Genetics, Anatomy, Professional Medicine, College Medicine, and College Biology sections.

Clinical Safety Benchmark (CSB): A novel benchmark we constructed containing 2,500 clinical scenarios specifically designed to test physiological constraint adherence. Table XIX details the CSB composition.

Reproducibility and Leakage Controls: We enforce four safeguards for train–test separation: (i) independent authoring of the CSB test set by clinicians without access to training artifacts, (ii) a temporal novelty split (post-cutoff approvals and unseen combinations), (iii) bounded lexical/semantic overlap via entity-level Jaccard filtering $( J ~ < ~ 0 . 4 0 )$ , and (iv) withholding 15% of KG drug entities (with incident edges removed) from alignment training. The temporal split is intentionally a stress test for novelty handling rather than a full simulation of routine clinical deployment, and should be interpreted accordingly.

![](images/a6f3ab50e15c879c2514de9934124010667209da62370183c2a522d7381ad37b.jpg)  
Fig. 2. Training-time neuro-symbolic alignment: the LLM samples candidates, the HGNN scores physiological feasibility on G, and Iterative ORPO updates the policy (HGNN not used at inference).

TABLE I  
GRAPH SOURCE STATISTICS
<table><tr><td>Source</td><td>Entities</td><td>Relations</td><td>Triples</td></tr><tr><td>UMLS Metathesaurus</td><td>4,200,000</td><td>54</td><td>12,847,392</td></tr><tr><td>DrugBank 5.1</td><td>14,315</td><td>12</td><td>1,432,847</td></tr><tr><td>Reactome</td><td>12,632</td><td>8</td><td>284,521</td></tr><tr><td>HPO</td><td>16,234</td><td>5</td><td>156,842</td></tr><tr><td>SIDER</td><td>1,430</td><td>3</td><td>139,756</td></tr><tr><td>PharmGKB</td><td>5,421</td><td>7</td><td>87,432</td></tr><tr><td>DisGeNET</td><td>24,166</td><td>4</td><td>628,685</td></tr><tr><td>STRING</td><td>19,566</td><td>1</td><td>11,759,454</td></tr><tr><td>Integrated Graph</td><td>847,392</td><td>12</td><td>3,241,567</td></tr></table>

Knowledge Graph Leakage Analysis: Entity overlap with G is expected for biomedical benchmarks and is not itself evidence of leakage. We therefore report a relational sensitivity check: MA is 76.1% when answer-critical relations are present in G versus 72.8% when they are absent (3.3% gap), suggesting gains are not explained by direct relation lookup alone. Version checks additionally confirm benchmark items predate the KG releases used in this study.

DDI-Corpus: Drug-Drug Interaction corpus containing 5,021 drug pairs annotated for interaction type (mechanism, effect, advise, int) from DrugBank and MedLine abstracts.

i2b2-2010: The i2b2/VA 2010 challenge dataset for clinical concept extraction and relation classification from discharge summaries.

## B. Knowledge Graph Construction

Our Physiological World Model integrates knowledge from multiple biomedical sources. Table I details the knowledge sources and their contributions.

The node type distribution in our heterogeneous graph is shown in Table II.

Figure 3 (Appendix) illustrates the schema of our heterogeneous knowledge graph with relation types.

TABLE II  
NODE TYPE DISTRIBUTION IN PHYSIOLOGICAL WORLD MODEL
<table><tr><td>Node Type</td><td>Count</td><td>Percentage</td></tr><tr><td>Drug</td><td>14,315</td><td>1.69%</td></tr><tr><td>Disease</td><td>24,166</td><td>2.85%</td></tr><tr><td>Symptom</td><td>7,823</td><td>0.92%</td></tr><tr><td>Organ/Anatomy</td><td>12,478</td><td>1.47%</td></tr><tr><td>Pathway</td><td>12,632</td><td>1.49%</td></tr><tr><td>Gene/Protein</td><td>19,566</td><td>2.31%</td></tr><tr><td>Biomarker</td><td>8,421</td><td>0.99%</td></tr><tr><td>Phenotype</td><td>16,234</td><td>1.92%</td></tr><tr><td>Procedure</td><td>45,892</td><td>5.42%</td></tr><tr><td>Clinical Finding</td><td>685,865</td><td>80.94%</td></tr><tr><td>Total</td><td>847,392</td><td>100%</td></tr></table>

TABLE III  
BASELINE METHOD SPECIFICATIONS
<table><tr><td>Method</td><td>Base Model</td><td>Training Data</td><td>KG</td><td>Pref. Opt.</td></tr><tr><td>SFT</td><td>Mistral-7B</td><td>MedQA + PubMed</td><td>x</td><td>x</td></tr><tr><td>SFT+RAG (lightweight)</td><td>Mistral-7B</td><td>MedQA + PubMed</td><td>Retrieval</td><td>x</td></tr><tr><td>SFT+DenseRAG</td><td>Mistral-7B</td><td>MedQA + PubMed</td><td>Dense+Rerank</td><td>x</td></tr><tr><td>DPO</td><td>Mistral-7B</td><td>Human Prefs (5K)</td><td>x</td><td>√</td></tr><tr><td>ORPO</td><td>Mistral-7B</td><td>Human Prefs (5K)</td><td>x</td><td>√</td></tr><tr><td>KG-LLM</td><td>Mistral-7B</td><td>MedQA + KG</td><td>Static</td><td>x</td></tr><tr><td>DrugBank-Rule</td><td>Mistral-7B</td><td>MedQA + PubMed</td><td>Rule-Based</td><td>Post-hoc</td></tr><tr><td>SFT-CSB</td><td>Mistral-7B</td><td>CSB Train (1,750)</td><td>x</td><td>x</td></tr><tr><td>MedAlpaca</td><td>LLaMA-7B</td><td>Medical Instruct</td><td>x</td><td>x</td></tr><tr><td>BioMistral</td><td>Mistral-7B</td><td>PubMed Pretrain</td><td>x</td><td>x</td></tr><tr><td>MEDITRON</td><td>LLaMA-70B</td><td>Medical Corpus</td><td>x</td><td>x</td></tr><tr><td>Med-PaLM 2</td><td>PaLM 2</td><td>Proprietary</td><td>x</td><td>RLHF</td></tr><tr><td>GPT-4</td><td>GPT-4</td><td>Proprietary</td><td>x</td><td>RLHF</td></tr><tr><td>GPT-4 (5-shot)</td><td>GPT-4</td><td>Proprietary</td><td>x</td><td>RLHF</td></tr><tr><td>SFT+SelfCorrect</td><td>Mistral-7B</td><td>MedQA + PubMed</td><td>Rule-Based</td><td>Post-hoc</td></tr><tr><td>Ours</td><td>Mistral-7B</td><td>Auto-Generated</td><td>Graph-guided</td><td>Iter-ORPO</td></tr></table>

## C. Baseline Methods

We compare against a comprehensive set of baseline approaches spanning supervised learning, preference optimization, knowledge-enhanced methods, and a dedicated clinical rule engine. Table III provides detailed baseline specifications. Our primary training experiments are centered on open-weight 7–8B models because Iterative ORPO requires repeated onpolicy sampling and full parameter access. Closed proprietary models are therefore used only as reference comparators at inference time.

SFT+DenseRAG uses a MedCPT retriever with crossencoder reranking to represent strong inference-time retrieval. DrugBank-Rule applies deterministic contraindication and homeostatic checks with up to three regeneration attempts, making it a favorable runtime-guardrail comparator. SFT-CSB fine-tunes directly on the 1,750 training scenarios to isolate the benefit of iterative preference optimization from simple exposure to safety-critical cases.

Two new baselines were added to address comparison gaps identified during review. GPT-4 (5-shot) evaluates GPT-4 with 5 safety-focused exemplars drawn from the CSB training set (selected to cover drug interactions, homeostatic violations, and polypharmacy scenarios), providing a stronger proprietary reference than strict zero-shot. SFT+SelfCorrect implements an inference-time self-correction pipeline: the SFT model generates a response, the DrugBank-Rule checker flags any contraindication or homeostatic alert, and if flagged, the model regenerates with the violation feedback appended to the prompt (up to 3 correction rounds). This approximates a tool-augmented agent that combines generation with external rule-engine verification at inference time.

Together these baselines cover preference-only, retrievalaugmented, static-KG, rule-based, self-correction, and proprietary RLHF alternatives.

Absent comparators. One important baseline family is not included: a matched-scale PPO/RLHF clinical system. At 7B parameters, PPO requires a separate reward model and value head, and no publicly available clinical reward model exists at comparable scale; training one from our 5K human-preference set would introduce confounds (rewardmodel quality) unrelated to our contribution. Med-PaLM 2 and GPT-4 use proprietary RLHF but differ in scale, data, and access, so they serve only as reference points. To partially address the comparison gap for tool-augmented agents, we added SFT+SelfCorrect (generate → DrugBank-Rule check → regenerate, up to 3 rounds) and GPT-4 (5-Shot) with safety-focused exemplars. SFT+SelfCorrect approximates a tool-augmented agent but lacks the iterative self-reflection and multi-tool orchestration of production agent systems; GPT-4 (5-Shot) is a stronger proprietary reference but still does not control for scale or training data. We therefore frame our claims relative to the included baselines rather than as absolute superiority, and note that future work should include PPO with a dedicated clinical reward model and full agent pipelines with multi-tool orchestration to complete the comparison landscape.

## D. Evaluation Metrics

We employ a comprehensive suite of evaluation metrics spanning accuracy, safety, and clinical utility. Table XXIII provides metric definitions.

Medical Accuracy (MA): Percentage of correct answers on medical QA benchmarks, computed as:

$$
\mathbf { M A } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { H } [ \hat { y } _ { i } = y _ { i } ]\tag{4}
$$

Clinical Safety Score (CSS): Measures the proportion of model responses that contain zero physiological constraint

violations:

$$
\mathrm { C S S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathcal { k } [ \mathrm { N o V i o l a t i o n } ( r _ { i } ) ]\tag{5}
$$

A response is classified as NoViolation when its HGNN feasibility score exceeds $\tau = 0 . 8 5$ , selected on a 200-response clinician-labeled validation set $( \kappa = 0 . 9 1 )$ . A 10,000-iteration bootstrap resampling of the 200 calibration labels yields a 95% CI for the optimal threshold of [0.83, 0.87]; CSS varies by ≤1.2 pp across this range (Table VII), indicating moderate sensitivity to the calibration sample. Expanding the calibration set beyond 200 labels is a priority for follow-up work. Calibration diagnostics and score-decile analysis are reported in Appendix C. Abstentions are not counted as safe because the task requires an actionable recommendation; in practice they occur in fewer than 0.3% of outputs.

Hallucination Rate (HR): Percentage of responses containing physiologically impossible statements, evaluated by a panel of 5 expert clinicians with inter-annotator agreement $\kappa ~ = ~ 0 . 8 4$ . Because physician review is expensive, HR is reported only for the six methods included in the blinded human-evaluation study.

Drug Interaction Detection (DID): F1 score for identifying contraindicated drug combinations:

$$
\mathrm { D I D - F 1 } = 2 \cdot \frac { \mathrm { P r e c i s i o n \cdot R e c a l l } } { \mathrm { P r e c i s i o n + R e c a l l } }\tag{6}
$$

Rule-Engine Safety Score (RSS): To provide an evaluation endpoint that is fully independent of the HGNN, we apply the DrugBank-Rule deterministic checker the same rule engine used by the DrugBank-Rule baseline as a post-hoc evaluator on all methods’ outputs. RSS measures the proportion of responses that trigger zero contraindication or homeostatic alerts under this checker:

$$
\mathrm { R S S } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \boldsymbol { k } \boldsymbol { \lvert \mathrm { R u l e C h e c k } ( r _ { i } ) = \ p a s s \rvert }\tag{7}
$$

RSS shares no parameters, training signal, or learned representations with the HGNN scorer. Its coverage is narrower than CSS (it checks only catalogued contraindications and predefined homeostatic thresholds, not learned multi-hop interactions), but it provides a structurally independent safety signal. We report RSS alongside CSS in Table IV and analyze their concordance in Section VI-B.

Physiological Consistency (PC): Average HGNNcomputed feasibility score across test responses:

$$
\mathrm { P C } = \frac { 1 } { N } \sum _ { i = 1 } ^ { N } \mathrm { H G N N } _ { \phi } ( \mathrm { E n t i t y E x t r a c t } ( r _ { i } ) , \mathcal { G } )\tag{8}
$$

Scorer-Decoupled Endpoints: Because CSS and PC are computed by the same HGNN scorer used to generate training preferences, reporting only HGNN-derived metrics would create a circularity risk: improvements might reflect optimization to the scorer rather than genuinely safer clinical reasoning. To mitigate this, we separately report three HGNN-independent safety endpoints: (i) RSS, computed by the DrugBank-Rule deterministic checker with no HGNN involvement, (ii) DID, computed from benchmark contraindication labels and rulebased interaction matching without any HGNN feasibility scores, and (iii) HR, adjudicated by blinded physicians who had no access to HGNN outputs. Consistent gains on these decoupled endpoints (RSS: +21.2 pp; DID: +18.8 pp; HR: −9.0 pp over ORPO) provide evidence that improvements extend beyond scorer optimization. The concordance between CSS and RSS (r = 0.97, κ = 0.82) further indicates that method rankings are robust to the choice of safety evaluator (Section VI-B).

PERFORMANCE COMPARISON ACROSS METHODS. HR IS REPORTED ONLY FOR THE BLINDED PHYSICIAN-EVALUATED SUBSET. RSS IS HGNN-INDEPENDENT (DRUGBANK-RULE DETERMINISTIC CHECKER).  
TABLE IV
<table><tr><td>Method</td><td>MA</td><td>CSS</td><td>RSS</td><td>HR↓</td><td>DID</td><td>PC</td></tr><tr><td>SFT</td><td>61.2</td><td>58.4</td><td>54.6</td><td>23.7</td><td>62.1</td><td>0.61</td></tr><tr><td>SFT+RAG</td><td>67.8</td><td>63.2</td><td>60.8</td><td></td><td>68.5</td><td>0.67</td></tr><tr><td>SFT+DenseRAG</td><td>70.8</td><td>78.3</td><td>74.2</td><td></td><td>81.4</td><td>0.78</td></tr><tr><td>DPO</td><td>69.4</td><td>67.8</td><td>63.4</td><td></td><td>71.3</td><td>0.69</td></tr><tr><td>ORPO</td><td>70.1</td><td>69.5</td><td>65.2</td><td>14.1</td><td>72.8</td><td>0.71</td></tr><tr><td>KG-LLM</td><td>68.9</td><td>72.4</td><td>68.1</td><td>12.8</td><td>76.2</td><td>0.74</td></tr><tr><td>DrugBank-Rule</td><td>68.3</td><td>76.1</td><td>82.4</td><td>9.8</td><td>80.4</td><td>0.77</td></tr><tr><td>SFT-CSB</td><td>66.7</td><td>71.8</td><td>67.3</td><td>11.4</td><td>74.9</td><td>0.73</td></tr><tr><td>SFT+SelfCorrect</td><td>68.8</td><td>79.4</td><td>83.7</td><td></td><td>82.6</td><td>0.78</td></tr><tr><td>GPT-4 (Zero-Shot)</td><td>78.4</td><td>74.2</td><td>71.6</td><td></td><td>79.1</td><td>0.76</td></tr><tr><td>GPT-4 (5-Shot)</td><td>79.1</td><td>80.6</td><td>78.2</td><td></td><td>84.3</td><td>0.80</td></tr><tr><td>Ours (Iter-1)</td><td>71.3</td><td>78.2</td><td>74.8</td><td></td><td>81.5</td><td>0.79</td></tr><tr><td>Ours (Iter-3)</td><td>73.8</td><td>86.4</td><td>82.1</td><td></td><td>87.3</td><td>0.85</td></tr><tr><td>Ours (Iter-5)</td><td>75.2</td><td>90.8</td><td>86.4</td><td>5.1</td><td>91.6</td><td>0.89</td></tr></table>

## E. Statistical Analysis

All trainable-model experiments use 5 random seeds; we report mean ± standard deviation across runs and assess significance with paired t-tests versus Ours. After Bonferroni correction for 35 primary comparisons, all reported primary comparisons remain significant $( p < 1 0 ^ { - 4 } )$ ). We report Cohen’s d as descriptive effect size and emphasize absolute metric deltas over null-hypothesis testing alone.

## F. Human Evaluation Protocol

For HR and clinical acceptability, 5 board-certified physicians evaluated 600 blinded responses (100 for each of 6 primary methods), with 3 annotators per response and interrater agreement $\kappa = 0 . 8 4$ . HR values in the main table are therefore reported only for these six methods. Annotators also assigned critical, moderate, and minor severity labels; the critical-hallucination reduction claim corresponds to 3.7% for ORPO versus 0.4% for Ours. For our 5.1% HR estimate, the 95% Wilson interval is [2.2%, 11.3%].

## V. TECHNICAL VALIDATION

## A. Main Results

Table IV presents the comparative performance across all evaluation metrics.

Compared with the standard RAG baseline (SFT+RAG), the final model improves CSS by 27.6 pp, DID by 23.1 pp, and MA by 7.4 pp, indicating that training-time physiological grounding adds value beyond retrieval alone.

Table IV shows that, relative to ORPO, our method improves MA by 5.1 pp, CSS by 21.3 pp, HR by 9.0 pp on the physician-evaluated subset, and DID by 18.8 pp. These gains are obtained from automatically generated HGNN-derived preferences rather than newly collected human preference labels.

The safety gains are not limited to HGNN-based endpoints: DID rises from 72.8% to 91.6%, HR falls from 14.1% to 5.1%, and the HGNN-independent Rule-Engine Safety Score (RSS) rises from 65.2% to 86.4% (+21.2 pp over ORPO). The model remains stronger than the deterministic DrugBank-Rule baseline on CSS and DID, and exceeds it on RSS (86.4% vs. 82.4%), indicating that training-time alignment internalizes safety patterns that go beyond what the rule engine alone catches at inference time.

Among the newly added baselines, SFT+SelfCorrect achieves 79.4% CSS and 83.7% RSS, demonstrating that inference-time self-correction with a rule engine partially closes the gap but does not match training-time alignment (CSS gap: 11.4 pp; RSS gap: 2.7 pp). GPT-4 (5-Shot) reaches 80.6% CSS and 78.2% RSS, a substantial improvement over strict zero-shot (CSS +6.4 pp), but our 7B model remains stronger on all safety metrics while GPT-4 retains an MA advantage (+3.9 pp). This comparison is more informative than the zero-shot reference, though it still does not control for GPT-4’s scale, data, or RLHF training.

The SFT-CSB control reaches 71.8% CSS, indicating that exposure to safety scenarios alone does not match iterative physiological preference optimization.

Residual Violations: Although CSS reaches 90.8%, 9.2% of responses still fall below the safety threshold; 5.3% are minor deviations (e.g., small dosing imprecision), whereas 3.9% are moderate-to-severe violations (≈20/500 test scenarios). The category breakdown below is reported over the unsafe subset and is non-exclusive because some failures involve multiple causes.

The dominant residual-error sources are sparse graph coverage (4.2%), multi-system interaction complexity (3.1%), and missing temporal physiology (2.8%); these three categories account for 70% of the unsafe responses. Detailed failure categories, severity distributions, and concrete clinical examples are reported in the appendix.

## B. Robustness and Novelty Generalization

a) Robustness stress tests.: We apply three perturbation families to the full CSB test set (500 scenarios) and measure safety degradation relative to clean-CSB performance. The perturbation families are: (i) noisy narratives section reordering, shorthand dosing notation, and omission of non-critical context; (ii) missing entities masking of labs, comorbidities, and co-medications at 30% drop rate; and (iii) ambiguous abbreviations replacing unambiguous drug and condition names with institution-typical shorthand requiring context-sensitive linking.

Under the combination of all three perturbations, our method retains 84.2% CSS, corresponding to a 6.6 pp degradation from clean-CSB performance. ORPO degrades by 6.9 pp from an already lower starting point, and SFT+DenseRAG degrades by 9.4 pp, indicating that retrieval-augmented baselines are more sensitive to entity-level noise than alignment-trained models. Ambiguous abbreviations produce the largest singlefamily degradation for all methods, consistent with entitylinking fragility being the dominant noise pathway.

b) Generalization under novelty splits.: We evaluate novelty along unseen drug-combination and unseen-diseasecontext splits. Novel combinations and sparse conditions remain meaningful residual-error sources, indicating that novelty handling is still a major frontier.

## C. Key Ablation Results

Ablation results support the central mechanism: removing HGNN scoring produces the largest degradation (−16.2 pp CSS), followed by removing iterative training (−11.5 pp). Among the PhysioScore components, $S _ { \mathrm { h o m } }$ contributes the most (−8.7 pp), followed by $P _ { \mathrm { i n t } } ( - 6 . 5 \mathrm { ~ \ p p } )$ and $S _ { \mathrm { p a t h } }$ (−3.9 pp). More detailed ablations and calibration diagnostics are reported in the appendix. Detailed term-drop and calibration tables are moved to Appendix C. In brief, $S _ { \mathrm { h o m } }$ is the dominant safety gate, $P _ { \mathrm { i n t } }$ is the key drug-specific term, and calibration against 200 clinician labels remains strong $( \mathrm { E C E } = 0 . 0 3 8 , \kappa = 0 . 9 1 \ \mathrm { a t } \ \tau = 0 . 8 5 )$

## D. Cross-Domain Generalization

Safety gains are not confined to a single specialty: across six reported domains, average CSS improves by 21.3 pp over ORPO, with the largest gains in geriatric and pediatric scenarios where polypharmacy and dose sensitivity are common.

## E. Iteration and Training Dynamics Summary

Performance improves monotonically over iterations (Table XIII) with diminishing returns after iteration 4, while training-loss trajectories (Fig. 5) show stable multi-objective convergence. We keep full per-component diagnostics in $\mathsf { A p - }$ pendix C.

Figure 4 (Appendix) visualizes the progression of key metrics across training iterations.

## F. Extended Technical Validation

Additional ablations and diagnostics are provided in $\mathsf { A p - }$ pendix C.

## G. Summary of Demonstrated Evidence

Table V consolidates the principal empirical findings and the evaluation method used for each claim, distinguishing HGNNbased metrics from independent endpoints.

Six of the twelve key results are measured by HGNNindependent or physician-adjudicated endpoints (RSS, DID, HR, PhysioScore–clinician $\kappa ,$ CSS–RSS concordance). The convergence of HGNN-based and independent metrics across all major comparisons provides triangulated evidence that the safety gains are not artifacts of scorer optimization. The remaining open question whether these controlled-benchmark results transfer to real clinical data is addressed in Section VI-B.

TABLE V  
SUMMARY OF KEY DEMONSTRATED RESULTS. METRIC INDEPENDENCE: I = FULLY HGNN-INDEPENDENT; H = HGNN-DERIVED; P = PHYSICIAN-ADJUDICATED.
<table><tr><td>Claim</td><td>Result</td><td>Ref.</td><td>Indep.</td></tr><tr><td>CSS over ORPO</td><td>+21.3 pp</td><td>Tab. IV</td><td>H</td></tr><tr><td>RSS over ORPO</td><td>+21.2 pp</td><td>Tab. IV</td><td>I</td></tr><tr><td>DID over ORPO</td><td>+18.8 pp</td><td>Tab. IV</td><td>I</td></tr><tr><td>HR over ORPO (physician)</td><td>-9.0 pp</td><td>Tab. IV</td><td>P</td></tr><tr><td>CSS vs. GPT-4 (5-shot)</td><td>+10.2 pp</td><td>Tab. IV</td><td>H</td></tr><tr><td>CSS vs. SFT+SelfCorrect</td><td>+11.4 pp</td><td>Tab. IV</td><td>H</td></tr><tr><td>CSS-RSS concordance</td><td>r=0.97</td><td>Sec. VI-B</td><td>I</td></tr><tr><td>CSS under combined noise</td><td>84.2%</td><td>Sec. V-C</td><td>H</td></tr><tr><td>Ablation: HGNN scoring</td><td>-16.2 pp</td><td>Sec. V-E</td><td>H</td></tr><tr><td>Ablation: iterative training</td><td>-11.5 pp</td><td>Sec. V-E</td><td>H</td></tr><tr><td>PhysioScore-clinician κ</td><td>0.91</td><td>Sec. IV-C</td><td>P</td></tr><tr><td>Cross-domain consistency</td><td>6/6 domains</td><td>Sec. V-F</td><td>H</td></tr></table>

## VI. DISCUSSION

## A. Implications for Medical AI Safety

Across CSB and auxiliary benchmarks, grounding preference optimization in an explicit physiological world model improves measured safety metrics while maintaining diagnostic utility. The central implication is that training-time constraint scoring can generate scalable preference data that targets physiologically invalid generations more directly than text-only supervision does.

## B. Clinical Integration and Limitations

Because the HGNN is used only during training, deployment does not require graph inference. This architectural simplification comes at the cost of higher alignment-time compute than SFT or one-pass preference optimization. The principal limitations are incomplete graph coverage, extraction and linking failures, and missing temporal physiology. In addition, CSB is synthetic and should therefore be interpreted as a controlled methodological benchmark rather than a realworld safety endpoint. The synthetic EHR-style noise stress tests provide only a proxy assessment of transfer and do not replace external validation on de-identified clinical notes or blinded clinician trust studies. High-stakes use therefore still requires layered safeguards and clinician oversight.

External validation gap. All results in this study are obtained on CSB, a controlled synthetic benchmark. No evaluation has been conducted on de-identified EHR notes, external clinical datasets (e.g., MIMIC-IV discharge summaries), or prospective clinical queries. The synthetic EHR-style noise stress tests (Section V-C) approximate some aspects of realworld input variability (abbreviations, missing labs, section reordering), but real clinical notes exhibit longer context, richer co-morbidity profiles, institution-specific templating, and copy-forward artifacts that CSB does not reproduce. We regard external validation ideally on a de-identified multi-site EHR corpus with blinded clinician adjudication as the single most important next step for establishing whether the safety gains measured on CSB transfer to practice. Until such validation is completed, the results should be interpreted as evidence of mechanism effectiveness on a controlled benchmark rather than as deployment-ready safety guarantees.

Comparison scope. The baseline set does not include a matched-scale PPO/RLHF clinical system or a tool-augmented agent (e.g., retrieval + rule-engine + self-correction pipeline). As detailed in Section IV-C, no publicly available clinical reward model exists at 7B scale, and tool-augmented agents address a complementary design point. The GPT-4 comparison is restricted to strict zero-shot inference without task-specific prompting or tool access, so it underestimates GPT-4’s potential when used with medical prompting strategies or plugins. These gaps mean that the reported improvements are demonstrated relative to the included baselines and should not be interpreted as claims of absolute state-of-the-art performance across all possible system configurations.

Scalability and computational cost. Iterative ORPO with K=8 sampling over 5 iterations requires 42.6 GPU-hours on 8×A100s (Table XVII), roughly 4× the cost of singlepass ORPO at 7B scale. This cost is driven primarily by candidate generation (70,000 forward passes), not HGNN scoring (2.0h total). Two practical factors partially mitigate this: (i) the one-time costs of KG construction and HGNN pre-training (11.0h) are amortized across model variants and re-alignment runs, and (ii) inference cost is unchanged because the HGNN is not used at deployment. However, the current framework does not support efficient continual updates: when the knowledge graph G is updated with new drug approvals or revised guidelines, the HGNN must be re-pretrained and the iterative alignment must be re-run from scratch, multiplying the cost. Efficient continual-alignment strategies such as warmstarting from the previous policy checkpoint, incremental HGNN updates on added subgraphs, or distilling the HGNN signal into a lightweight scorer would be necessary to make the framework practical for ongoing maintenance. Scaling beyond 7–8B parameters further compounds this cost (see Model Scale Analysis); at 70B, either reduced-K/reduced-iteration configurations or quantized generation would be needed to keep training cost manageable.

Knowledge-graph coverage and temporal staleness. The integrated KG is a static snapshot assembled from fixedversion sources (e.g., DrugBank 5.1, UMLS 2024AA). It therefore inherits two compounding limitations: coverage gaps in rapidly evolving domains and temporal staleness as new drug approvals, guideline revisions, and interaction reports accumulate after the snapshot date. Coverage gaps are the single largest source of residual failures, accounting for 4.2% of test-case errors; 67% of these occur in oncology, where therapeutic combinations and evidence change within months. Staleness is particularly concerning because the HGNN cannot flag an interaction it has never seen: a post-cutoff drug combination will be scored against an incomplete neighborhood, potentially assigning a falsely permissive PhysioScore. The current framework requires full HGNN re-pretraining and realignment when G is updated, making frequent refresh costly (≈11 h one-time plus 31.6 h marginal per model). Practical deployment therefore depends on efficient incremental update strategies, which we discuss in Section VI-C.

Inference-time safety gap. Removing the HGNN at inference eliminates graph-query latency and simplifies deployment, but it also means that the model has no runtime verifier to catch constraint violations that training did not anticipate. The 9.2% residual violation rate reported above shows that internalized constraints alone do not yet guarantee safety. For high-acuity clinical settings, a hybrid architecture trainingtime alignment plus a lightweight inference-time check would provide a second safety layer. We outline a concrete design for such a system in Section VI-C and regard it as the most important next step toward deployment-grade safety.

Evaluation circularity. A structural concern is that two of the five reported metrics CSS and PC rely on the same HGNN scorer whose signal drives training-time preference generation. To directly address this, we introduced an HGNNindependent metric (RSS) computed by the DrugBank-Rule deterministic checker, which shares no parameters or training signal with the HGNN. Across all methods in Table IV, CSS and RSS show a Pearson correlation of r = 0.97 and a Cohen’s $\kappa = 0 . 8 2$ (at their respective binary thresholds), indicating strong but imperfect agreement. Crucially, the method rankings are preserved: our Iter-5 model achieves the highest RSS (86.4%) among all methods, and the relative gains over baselines on RSS (+21.2 pp over ORPO, +4.0 pp over DrugBank-Rule) are directionally consistent with CSS gains, though smaller in magnitude because the rule engine covers fewer violation types. Additionally, DID (+18.8 pp) and physicianadjudicated HR (−9.0 pp) are also fully HGNN-independent and show consistent improvements. Together, three independent endpoints (RSS, DID, HR) corroborate the CSS-based conclusions. We offer additional mitigating observations: (i) the HGNN is frozen during all LLM alignment iterations and thus cannot co-adapt with the policy, (ii) PhysioScore was calibrated against 200 independent clinician labels with κ = 0.91 and ECE = 0.038, and (iii) the 4.4 pp gap between CSS (90.8%) and RSS (86.4%) for our method quantifies the portion of CSS gains that extends beyond rule-engineverifiable territory this gap is consistent across methods (≈4– 6 pp), suggesting it reflects genuine HGNN coverage breadth rather than scorer-specific inflation. Nonetheless, expanding clinician-only scoring beyond 600 responses and adopting external pharmacological classifiers remain priorities for followup work.

## C. Future Directions

Future work should prioritize temporal physiology, more robust entity grounding under negation and dosage ambiguity, continual updates of G, and external validation on real clin ical notes. To address residual evaluation circularity, scorerindependent evaluation protocols including external pharmacological rule engines (e.g., FDA FAERS-derived classifiers), independently trained safety models, and expanded blinded clinician adjudication beyond the current 600-response subset should be adopted as primary endpoints in follow-up studies. To strengthen comparative claims, future evaluations should include a matched-scale PPO/RLHF system with a dedicated clinical reward model [15] and tool-augmented agent baselines that combine retrieval with inference-time rule engines and self-correction loops. To improve scalability, efficient continual-alignment strategies such as warm-starting iterative ORPO from previous checkpoints when G is updated, incremental HGNN fine-tuning on added subgraphs rather than full re-pretraining, and distilling the HGNN preference signal into a smaller scorer for cheaper candidate ranking should be explored, along with empirical validation at 13B and 70B parameter scales.

Dynamic knowledge-graph maintenance. Because KG staleness is the upstream driver of the largest residual failure category, a principled update pipeline is essential for sustained safety. We identify three complementary strategies, ordered by increasing scope: (1) incremental subgraph patching new drug approvals and revised interaction alerts (e.g., from FDA Med-Watch or EMA DHPC notices) are ingested as delta subgraphs; existing HGNN weights are fine-tuned on the added edges while freezing parameters for unchanged regions, reducing re-pretraining cost to an estimated 15–20% of the full 8.6 h; (2) scheduled full refresh a quarterly or semi-annual rebuild of G from updated source releases, followed by full HGNN re-pretraining and warm-started re-alignment (the previous policy checkpoint seeds π <sub>(0)</sub>, which preliminary analysis suggests can halve convergence iterations from 5 to ≈2– 3); and (3) automated drift monitoring a background process that tracks the gap between the deployed KG snapshot and incoming pharmacovigilance feeds, triggering an update when the estimated coverage debt exceeds a predefined threshold (e.g., >50 new high-severity interactions not yet represented). Validating the cost–safety trade-offs of these strategies on a longitudinal benchmark is a priority for future work.

Smaller and efficient model variants. The current framework operates at 7–8B parameters. For resource-constrained deployment (e.g., on-premise hospital servers without multi-GPU infrastructure), two compression paths merit investigation: (1) alignment distillation, in which a smaller student model (1–3B) is trained on the aligned 7B teacher’s outputs using the same PhysioScore-ranked preferences, retaining much of the safety signal at reduced inference cost; and (2) post-alignment quantization (e.g., GPTQ 4-bit or AWQ), which preserves aligned behavior while halving memory and improving throughput. Preliminary profiling suggests that 4-bit quantization of the Iter-5 Mistral-7B checkpoint retains >99% of clean-input CSS, but systematic evaluation under noisy inputs and edge-case queries is needed before recommending quantized deployment. Exploring these efficient variants alongside the hybrid verification architecture would enable a spectrum of deployment configurations trading off compute, latency, and safety-guarantee strength.

Hybrid train-plus-verify architecture. The most direct way to close the inference-time safety gap is a selective deployment verifier that engages only when needed. We envision a two-tier design: (1) a lightweight confidence gate e.g., a distilled linear probe trained on the HGNN’s PhysioScore distribution runs on every output at <5 ms overhead and flags responses whose predicted PhysioScore falls below a tunable threshold τ; (2) flagged responses are routed to a full HGNN subgraph query (or an equivalent deterministic rule engine such as DrugBank-Rule) for verification and, if necessary, constrained re-generation. Because the aligned model already satisfies constraints for ≈91% of queries, the full verifier would be invoked on only the uncertain tail, keeping average latency close to the unverified baseline while providing a hard safety floor for the remaining cases. Investigating the calibration, latency, and safety trade-offs of this selectiveverification design is our highest-priority next step.

## VII. CONCLUSION

We introduced Neurosymbolic Alignment, a training-time framework that derives preference supervision from an HGNN-based Physiological World Model, and CSB, a 2,500- scenario benchmark for probing physiological constraint violations in clinical LLM outputs. Under controlled evaluation against 14 baselines, the framework achieves the highest scores on both HGNN-derived (CSS 90.8%) and HGNN-independent (RSS 86.4%, DID 91.6%) safety metrics, reduces physicianevaluated hallucination rate to 5.1%, and retains 84.2% CSS under synthetic noise. The gains are confirmed by three structurally independent evaluation endpoints (RSS, DID, HR) and supported by component-level ablation analysis. These results establish that structured physiological knowledge can generate effective preference supervision for clinical safety alignment at 7B scale.

Important limitations remain: evaluation is restricted to a synthetic benchmark, the HGNN contributes to two of six reported metrics, calibration uses 200 labels, and no real-world clinical validation has been conducted. External evaluation on de-identified EHR data, expanded clinician adjudication, and comparison with matched-scale PPO/RLHF systems are the necessary next steps toward translating these controlledbenchmark findings to clinical practice.

## ACKNOWLEDGMENT

We thank the clinical experts who supported dataset construction and evaluation, and the open-source communities behind the biomedical knowledge bases used to build the Physiological World Model.

## DATA RECORDS

CSB will be released as JSONL for train/valid/test, with prompt text, scenario category (Table XIX), and optional metadata (e.g., specialty and severity). Scripts are provided to reproduce metrics and analyses.

## DATA AVAILABILITY

Upon acceptance, CSB and the complete codebase will be available in an anonymized project repository. For reproducibility, the repository will provide: (1) the dataset in JSONL format, (2) scripts for constructing the integrated knowledge graph, (3) the pretrained HGNN weights, (4) the NER and entity-linking code, and (5) configuration files for all training experiments and baseline evaluations. Clinical-access and data-use requests may be sent to the corresponding author. Third-party resources, including UMLS and DrugBank, remain subject to their respective licenses.

## COMPLIANCE WITH ETHICAL STANDARDS

CSB contains only synthetically generated scenarios and no patient data. Physician evaluators consented to participate; IRB approval was not required.

## REFERENCES

[1] K. Singhal, T. Tu, J. Gottweis, R. Sayres, E. Wulczyn, L. Hou, K. Clark, S. Pfohl, H. Cole-Lewis, D. Neal et al., “Towards expert-level medical question answering with large language models,” Nature Medicine, vol. 30, pp. 943–957, 2024.

[2] A. J. Thirunavukarasu, D. S. W. Ting, K. Elangovan, L. Gutierrez, T. F. Tan, and D. S. W. Ting, “Large language models in medicine,” Nature Medicine, vol. 29, no. 8, pp. 1930–1940, 2023.

[3] Q. Lin, L. Xiao, B. Pu, K. Shi, Y. Du, J. Xu, K. He, S. Zhao, E. Cambria, S. Mishra et al., “A survey of LLM reasoning in healthcare and medicine: From individual modeling to collaborative agents,” arXiv preprint arXiv:2501.07362, 2026.

[4] H. Nori, N. King, S. M. McKinney, D. Carignan, and E. Horvitz, “Capabilities of GPT-4 on medical challenge problems,” arXiv preprint arXiv:2303.13375, 2023.

[5] M. Moor, Q. Huang, S. Wu, M. Yasunaga, Y. Dalmia, J. Leskovec, C. Zakka, E. P. Reis, and P. Rajpurkar, “Med-flamingo: a multimodal medical few-shot learner,” Machine Learning for Health, pp. 353–367, 2023.

[6] Z. Yang, J. Xu, W. Zhu, A. Wang, L. Zhang, and Y. Yang, “Mitigating hallucinations in medical large language models through knowledge grounding,” Journal of Biomedical Informatics, vol. 150, p. 104589, 2024.

[7] R. Rafailov, A. Sharma, E. Mitchell, C. D. Manning, S. Ermon, and C. Finn, “Direct preference optimization: Your language model is secretly a reward model,” Advances in Neural Information Processing Systems, vol. 36, pp. 53 728–53 741, 2024.

[8] J. Hong, N. Lee, and J. Thorne, “ORPO: Monolithic preference optimization without reference model,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing. Association for Computational Linguistics, 2024, pp. 11 170–11 183.

[9] K. Ethayarajh, W. Xu, D. Jurafsky, and D. Kiela, “Kto: Model alignment as prospect theoretic optimization,” arXiv preprint arXiv:2402.01306, 2024.

[10] Y. Meng, M. Xia, L. Wang, F. Sun, and Z. Liu, “Simpo: Simple preference optimization with a reference-free reward,” arXiv preprint arXiv:2405.14734, 2024.

[11] H. Chen, H. Yin, X. Sun, T. Chen, B. Gabrys, and K. Musial, “Heterogeneous graph neural networks for healthcare applications: A survey,” ACM Computing Surveys, vol. 56, no. 4, pp. 1–38, 2024.

[12] Y. Zhang, Z. Chen, Y. Liu, J. Wu, and P. S. Yu, “Neuro-symbolic integration for enhanced clinical reasoning in large language models,” Artificial Intelligence in Medicine, vol. 148, p. 102756, 2024.

[13] X. Wang, T. Zhang, Y. Lin, Z. Liu, and M. Sun, “Knowledge graph enhanced large language models for medical reasoning,” in Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics. Association for Computational Linguistics, 2024, pp. 5821– 5835.

[14] J. Wu, K. He, R. Mao, C. Li, and E. Cambria, “MEGACare: Knowledgeguided multi-view hypergraph predictive framework for healthcare,” Information Fusion, vol. 100, p. 101939, 2023.

[15] D. Perera, G. Habib, Q. Xu, D. J. Tan, K. He, E. Cambria, and M. Feng, “Beyond prediction: Reinforcement learning as the defining leap in healthcare AI,” arXiv preprint arXiv:2508.21101, 2025.

[16] T. Sun, J. He, X. Hong, and X. Qiu, “Head-to-tail: How knowledgeable are large language models (LLMs)? A.K.A. will LLMs replace knowledge graphs?” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, 2024, pp. 311–325.

[17] M. Neumann, D. King, I. Beltagy, and W. Ammar, “ScispaCy: Fast and robust models for biomedical natural language processing,” Journal of Biomedical Semantics, vol. 15, no. 1, pp. 1–14, 2024.

[18] J. Ji, M. Liu, J. Dai, X. Pan, C. Zhang, C. Bian, B. Chen, R. Sun, Y. Wang, and Y. Yang, “Beavertails: Towards improved safety alignment of LLM via a human-preference dataset,” Advances in Neural Information Processing Systems, vol. 36, 2024.

[19] W. Xiong, H. Dong, C. Ye, Z. Wang, H. Zhong, H. Ji, N. Jiang, and T. Zhang, “Iterative preference learning from human feedback: Bridging theory and practice for RLHF under KL-constraint,” in International Conference on Machine Learning. PMLR, 2024, pp. 54 561–54 582.

[20] T. Zhang, S. G. Patil, N. Jain, S. Shen, M. Zaharia, I. Stoica, and J. E. Gonzalez, “RAFT: Adapting language model to domain specific RAG,” in Proceedings of the 2024 Conference of the North American Chapter of the Association for Computational Linguistics, 2024, pp. 3865–3879.

[21] T. Schick, J. Dwivedi-Yu, R. Dess\`ı, R. Raileanu, M. Lomeli, E. Hambro, L. Zettlemoyer, N. Cancedda, and T. Scialom, “Toolformer: Language models can teach themselves to use tools,” Advances in Neural Information Processing Systems, vol. 36, pp. 68 539–68 551, 2024.

[22] P. Li, J. Rao, J. Yan, J. Zhang, and Z. Liu, “GraphCare: Enhancing healthcare predictions with personalized knowledge graphs,” Nature Machine Intelligence, vol. 6, pp. 218–231, 2024.

[23] D. Savage and R. Batista-Navarro, “Drug interaction prediction using knowledge graph embeddings and deep learning,” Journal ofBiomedical Informatics, vol. 152, p. 104631, 2024.

## APPENDIX A

## CHALLENGES OF NEUROSYMBOLIC ALIGNMENT IN CLINICAL LLMS (EXTENDED)

The gap between a neurosymbolic framework that is technically functional and one that is reliable in clinical practice remains substantial. Real clinical records combine incomplete narratives, measurement noise, polypharmacy involving multiple concurrent agents, and institution-specific shorthand that are not fully represented in any single ontology [1], [2]. In this setting, clinically important errors may arise from missed contraindications, superficially cautious but inappropriate dosing language, or plausible interaction chains supported by outdated graph edges [6]. The following subsections examine the main mechanisms underlying these failure modes.

## A. Ontology Friction and Coverage Debt

No clinical knowledge graph is fully current, and the consequences are especially pronounced in rapidly evolving domains. Oncology, immunotherapy, and rare-genotype pharmacology change faster than typical curation cycles, so the HGNN necessarily reasons over a graph snapshot that may already be outdated in clinically important regions. We refer to this problem as knowledge coverage debt.

A second limitation persists even when graph coverage is adequate. Many clinical concepts do not admit a stable single-identifier representation. For example, “renal failure” has different operational implications depending on acuity, assay standard, and whether the patient is moving toward or away from a threshold [13], [16]. Mapping such concepts to a single canonical node can remove distinctions that are relevant to safe medication recommendations. As a result, a reasoning path can be graph-consistent while still being clinically misleading.

## B. Entity Extraction as a Safety Bottleneck

In addition to ontology limitations, entity extraction is a frequent point of failure in neurosymbolic pipelines. Negated mentions for example, “the patient denies use of metformin”

shorthand dosing instructions, and institution-specific abbreviations must all be resolved before verification can proceed [17]. When extraction is incorrect, downstream components operate on a distorted representation of the clinical state. The constraint checker then evaluates a different scenario from the one described in the source note, without an internal mechanism for detecting that discrepancy.

Ambiguity further amplifies this problem. The abbreviation “ASA,” for example, may refer to aspirin in one context and to American Society of Anesthesiologists classification in another. Once an identifier is selected, the verifier will reason consistently from that representation. Graph consistency is therefore not equivalent to semantic correctness, and this divergence often originates at the entity-linking stage.

## C. Constraint Elicitation from Physiology to Logic

Beyond extraction, an additional challenge lies in translating physiological knowledge into rule-based representations. Encoding clinical knowledge as logic rules is challenging because many clinical thresholds are context-dependent conventions rather than fixed properties. A potassium threshold of 5.0 mEq/L can have different implications in a monitoring context versus a treatment decision context, or in a dehydrated patient versus a fluid-overloaded patient. Static rules necessarily abstract away these distinctions.

Defeasibility introduces an additional complication. Many constraints apply only when background conditions hold, such as adequate renal perfusion, medication adherence, or a recent dose within the expected window. These conditions are often absent from the prompt. A verifier that cannot observe them must either abstain more frequently, thereby reducing utility, or infer missing state, thereby introducing assumption risk. This trade-off affects model behavior in ways that may not be apparent in aggregate safety metrics.

## D. Scoring, Calibration, and Scalar Compression

Once constraints have been encoded, they must be summarized in a form suitable for optimization. Reducing a response’s clinical safety profile to a single score is operationally convenient, but it obscures important structure. For example, a score of 0.82 may correspond either to a minor imprecision in phrasing or to a moderate drug interaction that was partially recognized but insufficiently penalized. Preference optimization treats these cases similarly, which can bias training toward easily corrected signal rather than rare but clinically consequential errors.

Calibration across prompt distributions is a related concern [7], [8]. Scorers trained primarily on short, singlemedication queries may mis-rank complex polypharmacy narratives, in which co-occurrence patterns and interaction chains are substantially more difficult to resolve. Preference pairs derived from such scorers therefore reflect not only the intended safety signal but also the scorer’s distributional biases.

## E. Optimization Pathologies and Symbolic Shortcuts

These scoring limitations interact directly with the alignment objective. When a symbolic proxy defines the reward signal, the model is incentivized to optimize the proxy rather than the underlying clinical objective. In practice, this can produce several shortcut behaviors, including replacing specific drug names with therapeutic-class terminology, inserting conditional hedges that formally satisfy the constraint while preserving the original recommendation, and increasing use of generic deferral language. Each of these behaviors can improve measured CSS without materially improving recommendation safety.

On-policy pair regeneration directly addresses the distributional drift associated with static preference pairs, but it introduces a different challenge [7], [18], [19]. When the policy samples the pairs used for its own optimization, the scorer and policy may co-adapt in ways that reinforce shared biases. This risk motivates evaluation on independent heldout endpoints rather than relying solely on in-distribution performance metrics.

## F. Nonstationarity and Temporal Semantics

Even with iterative alignment, temporal reasoning remains a major limitation. Clinical risk evolves over time in ways that snapshot reasoning cannot adequately represent. Cumulative nephrotoxicity from aminoglycosides, washout windows for anticoagulants, and delayed serotonergic adverse effects all require the verifier to reason over prior clinical history, yet that information is typically absent from point-in-time prompts. Consequently, a recommendation that appears safe under current laboratory values may be unsafe when the recent trajectory of those values is considered.

This limitation reflects a structural mismatch between the clinical problem and the available representation. The patient’s state is a temporal process, whereas the prompt is only a partial observation of that process. Verifiers that do not model this distinction explicitly may produce recommendations that are locally plausible but temporally unsafe.

## G. Retrieval, Tools, and Responsibility Boundaries

Augmenting a language model with retrieval can reduce fabrication, but it also introduces distinct risks. Retrieved documents differ in provenance, recency, and population scope, and these attributes are not always explicit to either the reader or the model processing them [6], [20]. For example, a guideline published in 2019 and retrieved with high confidence may conflict with a 2024 update that is absent from the index. If retrieved content is treated as authoritative, outdated recommendations may be propagated and subsequently reinforced by downstream verification.

Tool use further complicates attribution of failure [21]. In a pipeline that includes retrieval, an external drug database, and a constraint verifier, an observed error may arise from model reasoning, an outdated API response, a query formulation problem, or incomplete database coverage. For this reason, evaluation in such settings requires end-to-end auditability rather than model-only benchmarking.

H. HGNN Expressivity, Oversmoothing, and Spurious Evidence

Heterogeneous graph neural networks capture relational diversity beyond what rigid rule-based systems can represent, but they also introduce characteristic failure modes. In dense subgraphs for example, those centered on common mechanisms such as CYP3A4 inhibition message passing aggregates information across many neighbors, which can attenuate rare but clinically severe contraindication signals before they reach the output layer [11]. Attention mechanisms introduced to mitigate this effect may instead prioritize frequent but clinically generic relations over the relations that are most relevant to the individual case.

Path-based plausibility scoring presents a related limitation. A multi-hop path that traverses hub nodes in the ontology that is, nodes with many connections because they are frequently annotated may receive a high plausibility score that reflects graph topology rather than mechanistic relevance. Distinguishing between topological prominence and clinically meaningful evidence generally requires domain knowledge that is not represented in the graph structure alone.

## I. Evaluation Gaps and Operational Risk

These modeling limitations also affect interpretation of empirical results. Controlled benchmark performance is necessary but insufficient evidence of deployment readiness. Clinical notes encountered in practice are typically longer, noisier, more ambiguous, and more structurally heterogeneous than benchmark items, and improved benchmark performance does not remove this discrepancy. CSB was designed to maximize reproducibility and interpretability, but those design choices require standardization trade-offs that reduce ecological validity for the most complex real-world cases.

Baseline coverage introduces an additional source of uncertainty [2]. Some recently published alternatives were not available for comparison at the time of evaluation, which limits the strength of comparative claims. Accordingly, the results should be interpreted as evidence that the proposed framework is effective under the evaluated conditions rather than as a definitive claim of superiority over all currently available methods.

Metric design also shapes the behaviors that are reinforced during training. A CSS that heavily penalizes unsafe actions may encourage abstention beyond what is clinically useful, whereas a metric that penalizes abstention too strongly may push the model toward unsupported specificity. Neither extreme is acceptable in practice, and selecting an appropriate balance requires continued empirical study rather than a onetime design decision.

## J. Personalization, Fairness, and Hidden Covariates

Personalizing the knowledge graph to individual patient records can improve specificity, but patient records encode more than clinical state. Socioeconomic patterns, institutional practice variation, and disparities in access to care may all be embedded in the data and subsequently learned if they correlate with outcomes [22]. As a result, a system that appears constraint-consistent may still reproduce inequitable patterns that are not detected by the safety scorer.

At the level of individual inference, an additional limitation is that important covariates are often unavailable. Genetic variation affecting drug metabolism, actual rather than prescribed adherence, and access to follow-up care are rarely documented in the prompt. The language model and the verifier therefore operate on the same incomplete patient representation, increasing the likelihood that their errors will be systematically correlated rather than independent.

## K. Practical Deployment and Uncertainty Management

Achieving low hallucination rates under controlled conditions is an important result, but it does not by itself establish readiness for deployment [6]. Practical deployment requires additional system-level safeguards, including acuitytiered triage so that high-stakes queries trigger deterministic contraindication checks, explicit uncertainty outputs that indicate when the system is operating near the limits of its reliability, and traceable evidence paths that support clinical review rather than opaque recommendations [23].

An important trade-off remains. Tighter constraints reduce the space of unsafe outputs, but they can also limit clinically useful nuance. Looser constraints preserve flexibility, but they leave open proxy-exploitation pathways associated with shortcut learning [7], [8], [12]. This trade-off cannot be removed by a single design choice and instead requires continued monitoring and iterative adjustment as the failure distribution becomes better characterized.

![](images/7e3f579edb720e56c828705ea67c5a752a194068b88e47efe13ba32e56592fff.jpg)  
Fig. 3. Schema of the heterogeneous physiological knowledge graph showing the six node types (T<sub>V</sub> ) and five canonical relation types $( \mathcal { T } _ { E } )$

![](images/e9b8caa1348a11b2cb4b970eae27d4b2c0062b1e79f598b8bd11ba7266feb919.jpg)  
Fig. 4. Performance progression across training iterations. CSS and DID show the steepest improvements in early iterations, while gains diminish after iteration 4. The inverted hallucination rate (100-HR) demonstrates consistent safety improvements throughout training.

## APPENDIX C

EXTENDED TECHNICAL VALIDATION (MOVED)

## A. Additional Analyses and Diagnostics

```latex
Algorithm 2 HGNN Physiological Feasibility Scoring
Require: Response r, knowledge graph G, HGNN parameters
ϕ
Ensure: Feasibility score $s \in [ 0 , 1 ]$
1: E ← EntityExtract(r)
2: $\mathbf { V } _ { r } \gets \mathbf { M a p T o G r a p h } ( \mathbf { E } , \mathcal { G } )$
3: Compute HGNN embeddings on the induced subgraph
4: $\begin{array} { r } { S _ { \mathrm { h o m } } \mathrm { ~  ~ } \frac { 1 } { | \mathcal { H } | } \sum _ { h \in \mathcal { H } } f _ { h } ( \mathbf { V } _ { r } ) } \end{array}$
5: $\begin{array} { r } { S _ { \mathrm { p a t h } }  \frac { 1 } { | \mathbf { V } _ { r } | ^ { 2 } } \sum _ { ( u , v ) \in \mathbf { V } _ { r } } } \end{array}$ PathScore $( u , v , { \mathcal { G } } )$
6: $P _ { \mathrm { i n t } }  \mathrm { m a x } _ { ( d _ { i } , d _ { j } ) } ^ { }$ <sub>∈Drugs(r)</sub> InteractionSeverity $( d _ { i } , d _ { j } , \mathcal { G } )$
7: $s  S _ { \mathrm { h o m } } \cdot \sigma ( S _ { \mathrm { p a t h } } ^ { ^ { \prime } } ) \cdot ( \bar { 1 } - P _ { \mathrm { i n t } } )$
8: return s
```

TABLE VI  
ABLATION STUDY RESULTS
<table><tr><td>Configuration</td><td>CSS</td><td>HR↓</td></tr><tr><td>Full Model</td><td>90.8</td><td>5.1</td></tr><tr><td>w/ Rule-based Scorer</td><td>77.4</td><td>10.2</td></tr><tr><td>w/ Simple KG Scorer</td><td>81.6</td><td>8.9</td></tr><tr><td>w/o Iterative Training</td><td>79.3</td><td>9.8</td></tr><tr><td>w/o HGNN Scoring</td><td>74.6</td><td>12.4</td></tr><tr><td>w/o Homeostatic Constraints</td><td>82.1</td><td>8.7</td></tr><tr><td>w/o  $\mathcal { L } _ { \mathrm { K G } }$ </td><td>85.4</td><td>6.9</td></tr><tr><td>w/o Curriculum Schedule</td><td>87.2</td><td>6.2</td></tr><tr><td>HGNN 2 Layers HGNN 4 Layers (default)</td><td>86.3</td><td>7.4</td></tr><tr><td>HGNN 6 Layers</td><td>90.8</td><td>5.1</td></tr><tr><td></td><td>91.1</td><td>5.0</td></tr><tr><td>Constant λ (no curriculum)</td><td>87.2</td><td>6.2</td></tr><tr><td>On-policy + constant λ</td><td>89.4</td><td>5.6</td></tr><tr><td>Candidate budget (K) and iteration (T) sensitivity</td><td></td><td></td></tr><tr><td>K = 4, T = 5</td><td>88.7</td><td>5.9</td></tr><tr><td>K = 8, T = 5 (default)</td><td>90.8</td><td>5.1</td></tr><tr><td> $K = 1 \dot { 6 } , T = 5$ </td><td>91.3</td><td>4.8</td></tr><tr><td> $K = 8 , { \overset { \cdot } { T } } = 3$ </td><td>87.9</td><td>6.8</td></tr><tr><td>K = 4, T = 3 (budget-efficient)</td><td>86.1</td><td>7.2</td></tr></table>

Reducing K from 8 to 4 costs 2.1 pp CSS but approximately halves generation time; doubling K to 16 yields only 0.5 pp, indicating diminishing returns. Reducing iterations from 5 to 3 costs 2.9 pp but saves 40% of iterative training time. The budget-efficient configuration $( K { = } 4 , \ T { = } 3 )$ achieves 86.1% CSS at roughly one-third the compute of the full configuration, which may be preferable when aligning larger models under resource constraints.

TABLE VII  
CSS THRESHOLD (τ ) SENSITIVITY. BOOTSTRAP 95% CI = [0.83, 0.87]; n=200 CALIBRATION LABELS.
<table><tr><td>Threshold τ</td><td>CSS (%)</td><td>HR↓ (%)</td><td>Note</td></tr><tr><td>0.80</td><td>92.0</td><td>6.4</td><td>More permissive; higher false-safe rate</td></tr><tr><td>0.83</td><td>91.4</td><td>5.6</td><td>Lower bound of 95% CI</td></tr><tr><td>0.85 (default)</td><td>90.8</td><td>5.1</td><td>Calibrated optimum</td></tr><tr><td>0.87</td><td>90.2</td><td>4.7</td><td>Upper bound of 95% CI</td></tr><tr><td>0.90</td><td>89.1</td><td>4.2</td><td>More conservative; higher abstention</td></tr></table>

TABLE VIII  
SENSITIVITY TO KNOWLEDGE-GRAPH QUALITY AND ENTITY-LINKING ERRORS
<table><tr><td>Condition</td><td>CSS (%)</td><td>HR↓ (%)</td></tr><tr><td>KG edge-deletion stress tests</td><td></td><td></td></tr><tr><td>Full KG (default)</td><td>90.8</td><td>5.1</td></tr><tr><td>10% random edge deletion</td><td>88.6</td><td>6.3</td></tr><tr><td>20% random edge deletion</td><td>85.9</td><td>7.8</td></tr><tr><td>30% random edge deletion</td><td>81.4</td><td>9.5</td></tr><tr><td>10% targeted deletion (high-severity edges)</td><td>86.1</td><td>7.4</td></tr><tr><td>Entity-linking error injection</td><td></td><td></td></tr><tr><td>Clean linking (default)</td><td>90.8</td><td>5.1</td></tr><tr><td>5% random entity mislinks</td><td>89.2</td><td>5.8</td></tr><tr><td>10% random entity mislinks</td><td>87.1</td><td>6.7</td></tr><tr><td>20% random entity mislinks</td><td>83.5</td><td>8.4</td></tr><tr><td>10% negation-blind errors</td><td>86.4</td><td>7.1</td></tr></table>

KG quality sensitivity (Table VIII, upper): Random deletion of 10% of KG edges degrades CSS by 2.2 pp, and 30% deletion by 9.4 pp, confirming that alignment quality depends substantially on graph completeness. Notably, targeted deletion of high-severity edges (top-10% by interaction severity score) at only 10% volume produces a 4.7 pp CSS drop, nearly matching the effect of 20% random deletion, which underscores the asymmetric importance of safety-critical edges.

Entity-linking sensitivity (Table VIII, lower): Injecting 5% random entity mislinks (swapping the linked UMLS CUI to a sibling concept) costs 1.6 pp CSS, while 20% mislinks cost 7.3 pp. Negation-blind errors where negated mentions (e.g., “denies metformin use”) are linked as affirmative are particularly damaging: 10% negation-blind errors produce a 4.4 pp drop, exceeding the effect of 10% random mislinks (3.7 pp), because they invert the clinical state representation. These results quantify entity linking as a critical upstream dependency and motivate robust NER as a prerequisite for safe deployment.

PHYSIOSCORE COMPONENT ABLATION: TERM-DROP RESULTS  
TABLE IX
<table><tr><td>Scorer Configuration</td><td>CSS</td><td>∆CSS</td><td>HR↓</td><td>∆HR</td><td>DID</td></tr><tr><td>Full PhysioScore  $( S _ { \mathrm { h o m } } \cdot \sigma ( S _ { \mathrm { p a t h } } ) \cdot ( 1 { - } P _ { \mathrm { i n t } } ) )$ </td><td>90.8</td><td></td><td>5.1</td><td></td><td>91.6</td></tr><tr><td>Drop  $\dot { S _ { \mathrm { h o m } } }$  (rule-based homeostasis)</td><td>82.1</td><td>-8.7</td><td>9.4</td><td>+4.3</td><td>88.2</td></tr><tr><td>Drop  $S _ { \mathrm { p a t h } }$  (graph path plausibility)</td><td>86.9</td><td>-3.9</td><td>7.2</td><td>+2.1</td><td>84.7</td></tr><tr><td>Drop  $\dot { P _ { \mathrm { i n t } } }$  (drug interaction penalty)</td><td>84.3</td><td>-6.5</td><td>10.8</td><td>+5.7</td><td>79.1</td></tr><tr><td>Drop  $S _ { \mathrm { h o m } } + \breve { P } _ { \mathrm { i n t } }$  (learned path only)</td><td>79.6</td><td>-11.2</td><td>12.3</td><td>+7.2</td><td>76.4</td></tr><tr><td>Drop  $S _ { \mathrm { p a t h } }$  (fix hom+int rules only)</td><td>84.3</td><td>-6.5</td><td>8.9</td><td>+3.8</td><td>83.1</td></tr></table>

TABLE X  
PHYSIOSCORE CALIBRATION VS. CLINICIAN SAFETY LABELS (200-RESPONSE HELD-OUT SET)
<table><tr><td>Score Decile</td><td>Clinician Safe Rate (%)</td><td>Mean PhysioScore</td></tr><tr><td>Decile 1 (lowest scores)</td><td>8.5</td><td>0.21</td></tr><tr><td>Decile 2</td><td>17.0</td><td>0.34</td></tr><tr><td>Decile 3</td><td>31.0</td><td>0.45</td></tr><tr><td>Decile 4</td><td>48.0</td><td>0.56</td></tr><tr><td>Decile 5</td><td>63.5</td><td>0.66</td></tr><tr><td>Decile 6</td><td>74.0</td><td>0.74</td></tr><tr><td>Decile 7</td><td>83.5</td><td>0.81</td></tr><tr><td>Decile 8</td><td>90.0</td><td>0.87</td></tr><tr><td>Decile 9</td><td>95.5</td><td>0.92</td></tr><tr><td>Decile 10 (highest scores)</td><td>98.0</td><td>0.97</td></tr><tr><td>Expected Calibration Error (ECE)</td><td colspan="2">0.038</td></tr><tr><td>Agreement at τ=0.85 (κ)</td><td colspan="2">0.91</td></tr></table>

TABLE XI  
PHYSIOSCORE COMBINATION FUNCTION SENSITIVITY
<table><tr><td>Combination Function</td><td>CSS</td><td>∆CSS</td><td>HR↓</td><td>DID</td></tr><tr><td> $S _ { \mathrm { h o m } }$  σ  $( S _ { \mathrm { p a t h } } )$  · (1-  $\cdot P _ { \mathrm { i n t } } )$  (default, multiplicative)</td><td>90.8</td><td></td><td>5.1</td><td>91.6</td></tr><tr><td> $\begin{array} { r } { \frac { 1 } { 3 } ( S _ { \mathrm { h o m } } + \dot { \sigma } ( S _ { \mathrm { p a t h } } ) + ( 1 - P _ { \mathrm { i n t } } ) ) } \end{array}$  (uniform additive)</td><td>86.2</td><td>-4.6</td><td>7.8</td><td>85.9</td></tr><tr><td> $\ddot { 0 . 5 } S _ { \mathrm { h o m } } + 0 . 3 \dot { \sigma } ( S _ { \mathrm { p a t h } } ) + 0 . 2 ( 1 - P _ { \mathrm { i n t } } )$  (weighted additive)</td><td>87.1</td><td>-3.7</td><td>7.3</td><td>86.4</td></tr><tr><td>min  $( S _ { \mathrm { h o m } } , \sigma ( S _ { \mathrm { p a t h } } ^ { \mathrm { ~ \scriptsize ~ { ~ a ~ } ~ } } ) , ( 1 - P _ { \mathrm { i n t } } ) )$  (worst-component)</td><td>88.4</td><td>-2.4</td><td>6.2</td><td>89.1</td></tr><tr><td> $S _ { \mathrm { h o m } } ^ { 0 . 5 } \cdot \sigma ( S _ { \mathrm { p a t h } } ) ^ { 0 . 3 } \cdot ( \mathrm { 1 } ^ { - } P _ { \mathrm { i n t } } ) ^ { 0 . 2 }$  (geometric weighted)</td><td>89.6</td><td>-1.2</td><td>5.6</td><td>90.2</td></tr></table>

The multiplicative default achieves the highest CSS (90.8%) because a near-zero score on any component e.g., a severe drug interaction $( P _ { \mathrm { i n t } } \approx 1 )$ drives the product toward zero, creating strong gradient signal for the preference pair. Additive variants dilute this effect: a dangerous interaction can be offset by high $S _ { \mathrm { h o m } }$ and $S _ { \mathrm { p a t h } }$ , weakening the safety signal. The worstcomponent (min) function is the second-best, consistent with the safety-gate intuition, but discards magnitude information that the product preserves.

TABLE XII  
STEEPNESS PARAMETER (λ) SENSITIVITY ON CSS
<table><tr><td>λ Perturbation</td><td>CSS</td><td>∆CSS</td><td>HR↓</td><td>∆HR</td><td>Note</td></tr><tr><td>Default λ values (Table 5)</td><td>90.8</td><td></td><td>5.1</td><td></td><td></td></tr><tr><td>All λ × 0.5 (halved steepness)</td><td>87.3</td><td>-3.5</td><td>7.4</td><td>+2.3</td><td>Softer penalties</td></tr><tr><td>All λ × 2.0 (doubled steepness)</td><td>89.2</td><td>-1.6</td><td>5.9</td><td>+0.8</td><td>Harsher but noisier</td></tr><tr><td>Uniform λ = 2.0 for all constraints</td><td>88.1</td><td>-2.7</td><td>6.8</td><td>+1.7</td><td>Ignores clinical risk profiles</td></tr><tr><td>λ from Gaussian noise (±20%)</td><td>89.9</td><td>-0.9</td><td>5.5</td><td>+0.4</td><td>Robust to moderate noise</td></tr></table>

CSS is moderately sensitive to λ choices: halving all steepness values degrades CSS by 3.5 pp because violations near boundaries are under-penalized, while doubling them causes a smaller 1.6 pp drop from over-penalization of borderlinenormal values. The constraint-specific defaults derived from clinical risk profiles outperform a uniform λ, justifying the per-constraint parameterization. Importantly, ±20% Gaussian perturbation of the default λ values degrades CSS by only

0.9 pp, indicating that the method is not brittle to moderate uncertainty in the steepness calibration.

TABLE XIII  
PERFORMANCE PROGRESSION ACROSS TRAINING ITERATIONS
<table><tr><td>Iteration</td><td>MA (%)</td><td>CSS (%)</td><td>HR (%)↓</td><td>DID (%)</td><td>PC</td></tr><tr><td>Iteration 1</td><td>71.3</td><td>78.2</td><td>10.4</td><td>81.5</td><td>0.79</td></tr><tr><td>Iteration 2</td><td>72.4</td><td>82.6</td><td>8.9</td><td>84.2</td><td>0.82</td></tr><tr><td>Iteration 3</td><td>73.8</td><td>86.4</td><td>7.2</td><td>87.3</td><td>0.85</td></tr><tr><td>Iteration 4</td><td>74.6</td><td>89.1</td><td>5.8</td><td>90.1</td><td>0.87</td></tr><tr><td>Iteration 5</td><td>75.2</td><td>90.8</td><td>5.1</td><td>91.6</td><td>0.89</td></tr><tr><td>∆ (Iter 1→5)</td><td>+3.9</td><td>+12.6</td><td>-5.3</td><td>+10.1</td><td>+0.10</td></tr></table>

TABLE XIV  
CROSS-DOMAIN GENERALIZATION RESULTS
<table><tr><td>Specialty</td><td>CSS (Ours)</td><td>CSS (ORPO)</td><td>∆</td><td>HR (Ours)</td><td>DID (Ours)</td></tr><tr><td>Cardiology</td><td>91.2</td><td>70.3</td><td>+20.9</td><td>4.8</td><td>92.4</td></tr><tr><td>Nephrology</td><td>89.7</td><td>68.1</td><td>+21.6</td><td>5.3</td><td>90.8</td></tr><tr><td>Oncology</td><td>87.4</td><td>65.8</td><td>+21.6</td><td>6.1</td><td>88.2</td></tr><tr><td>Pediatrics</td><td>86.9</td><td>64.2</td><td>+22.7</td><td>5.8</td><td>89.1</td></tr><tr><td>Psychiatry</td><td>84.3</td><td>67.5</td><td>+16.8</td><td>7.2</td><td>82.6</td></tr><tr><td>Emerg. Med.</td><td>88.6</td><td>66.4</td><td>+22.2</td><td>5.5</td><td>89.7</td></tr><tr><td>Geriatrics</td><td>87.1</td><td>63.8</td><td>+23.3</td><td>6.4</td><td>88.4</td></tr><tr><td>Average</td><td>87.9</td><td>66.6</td><td>+21.3</td><td>5.9</td><td>88.7</td></tr></table>

![](images/c7a0cf6e88a30f55109d623f773da4b4736b91300956e4a1b2d17371a29a6bd7.jpg)  
Fig. 5. Training loss convergence over 30K steps. Total loss decreases from 2.3 to 0.38, with KG loss showing the fastest convergence. The OR loss initially increases as preference learning intensifies, then stabilizes.

TABLE XV  
TRAINING STATISTICS ACROSS ITERATIONS
<table><tr><td>Metric</td><td>Iter 1</td><td>Iter 2</td><td>Iter 3</td><td>Iter 4</td><td>Iter 5</td></tr><tr><td>Total Loss</td><td>0.82</td><td>0.61</td><td>0.48</td><td>0.41</td><td>0.38</td></tr><tr><td>SFT Loss</td><td>0.95</td><td>0.78</td><td>0.68</td><td>0.62</td><td>0.60</td></tr><tr><td>KG Loss</td><td>0.28</td><td>0.18</td><td>0.14</td><td>0.12</td><td>0.11</td></tr><tr><td>OR Loss</td><td>0.35</td><td>0.32</td><td>0.30</td><td>0.29</td><td>0.28</td></tr><tr><td>Pref. Pairs Generated</td><td>1,750</td><td>1,750</td><td>1,750</td><td>1,750</td><td>1,750</td></tr><tr><td>Avg. Score Gap</td><td>0.42</td><td>0.38</td><td>0.31</td><td>0.26</td><td>0.22</td></tr><tr><td>GPU Hours</td><td>3.2</td><td>3.4</td><td>3.6</td><td>3.8</td><td>4.0</td></tr></table>

## IMPLEMENTATION DETAILS

We implement our framework using multiple open-source LLMs to ensure reproducibility and broad applicability. Table XVI summarizes all hyperparameters.

TABLE XVI  
HYPERPARAMETER CONFIGURATION
<table><tr><td>Component</td><td>Parameter</td><td>Value</td></tr><tr><td rowspan="6">LLM Training</td><td>Base Model</td><td>Mistral-7B-v0.3</td></tr><tr><td>Learning Rate</td><td>2 × 10−5</td></tr><tr><td>LR Schedule</td><td>Cosine Decay</td></tr><tr><td>Batch Size</td><td>32</td></tr><tr><td>Max Sequence Length</td><td>2048</td></tr><tr><td>Gradient Accumulation</td><td>4</td></tr><tr><td rowspan="4">LoRA</td><td>Rank (r)</td><td>64</td></tr><tr><td>Alpha (α)</td><td>128</td></tr><tr><td>Dropout</td><td>0.05</td></tr><tr><td>Target Modules</td><td>q_proj, v_proj, k_proj, o_proj</td></tr><tr><td rowspan="5">HGNN</td><td>Hidden Dimension</td><td>256</td></tr><tr><td>Number of Layers</td><td>4</td></tr><tr><td>Attention Heads</td><td>8</td></tr><tr><td>Dropout</td><td>0.1</td></tr><tr><td>Activation</td><td>LeakyReLU (0.2)</td></tr><tr><td rowspan="5">Iterative ORPO</td><td>Iterations (T)</td><td>5</td></tr><tr><td>Samples per Query (K)</td><td>8</td></tr><tr><td>Initial λ0</td><td>0.1</td></tr><tr><td>Curriculum α</td><td>0.2</td></tr><tr><td>KG Loss Weight (γ)</td><td>0.05</td></tr><tr><td rowspan="3">CTS Weights</td><td>ω1 (Physiological)</td><td>0.4</td></tr><tr><td>ω2 (Safety)</td><td>0.4</td></tr><tr><td>ω3 (Coherence)</td><td>0.2</td></tr></table>

Computational Resources: All experiments were conducted on 8× NVIDIA A100 80GB GPUs with mixedprecision training (bfloat16). We utilize the HuggingFace Transformers library with DeepSpeed ZeRO-3 optimization for distributed training. Table XVII details computational requirements.

TABLE XVII  
COMPUTATIONAL RESOURCE REQUIREMENTS
<table><tr><td>Stage</td><td>GPU Hours</td><td>Memory (GB)</td><td>Storage (GB)</td></tr><tr><td>KG Construction</td><td>2.4</td><td>128</td><td>45</td></tr><tr><td>HGNN Pre-training</td><td>8.6</td><td>320</td><td>12</td></tr><tr><td>LLM Fine-tuning (per iter.)</td><td>3.2</td><td>640</td><td>28</td></tr><tr><td>Response Generation</td><td>1.8</td><td>160</td><td>8</td></tr><tr><td>Preference Scoring</td><td>0.4</td><td>64</td><td>2</td></tr><tr><td>Total (5 iterations)</td><td>42.6</td><td></td><td>95</td></tr></table>

Reproducibility Details: For entity extraction, we use scispaCy v0.5.1 with the en\_core\_sci\_lg model (768- dimensional embeddings), with the scispacy-linker component configured to link entities to UMLS CUIs using a similarity threshold of 0.85 and a maximum of 5 candidate CUIs per entity. Drug node embeddings are initialized from Morgan fingerprints with radius 2 and 2,048-bit vector length, projected to the HGNN hidden dimension (d = 256) via a learned linear transformation. The CTS weights $\omega _ { 1 } ~ = ~ 0 . 4 .$ ω<sub>2</sub> = 0.4, ω<sub>3</sub> = 0.2 were selected via grid search over the validation set with step size 0.1, subject to the constraint $\omega _ { 1 } + \omega _ { 2 } + \omega _ { 3 } = 1$ . The “Memory (GB)” column in Table XVII reports aggregate memory across all 8 GPUs (i.e., 128 GB total = 16 GB per GPU for KG construction, which includes graph materialization and initial node embedding computation). Preference scoring at 0.4 GPU-hours per iteration evaluates K = 8 candidates for each of the 1,750 training queries and retains one winner–loser preference pair per query.

APPENDIX E  
ADDITIONAL TABLES  
TABLE XVIII  
DATASET STATISTICS AND CHARACTERISTICS
<table><tr><td>Dataset</td><td>Train</td><td>Valid</td><td>Test</td><td>Task Type</td></tr><tr><td>MedQA-USMLE</td><td>10,178</td><td>1,272</td><td>1,273</td><td>Multiple Choice</td></tr><tr><td>PubMedQA</td><td>450</td><td>50</td><td>500</td><td>Yes/No/Maybe</td></tr><tr><td>MedMCQA</td><td>182,822</td><td>4,183</td><td>6,150</td><td>Multiple Choice</td></tr><tr><td>MMLU-Medical</td><td>1,089</td><td>123</td><td>1,871</td><td>Multiple Choice</td></tr><tr><td>CSB (Ours)</td><td>1,750</td><td>250</td><td>500</td><td>Open Generation</td></tr><tr><td>DDI-Corpus</td><td>4,521</td><td>645</td><td>1,290</td><td>Interaction Detection</td></tr><tr><td>i2b2-2010</td><td>16,315</td><td>2,330</td><td>4,661</td><td>Relation Extraction</td></tr><tr><td>Total</td><td>217,125</td><td>8,853</td><td>16,245</td><td></td></tr></table>

TABLE XIX  
CLINICAL SAFETY BENCHMARK (CSB) COMPOSITION
<table><tr><td>Category</td><td>Scenarios</td><td>Percentage</td></tr><tr><td>Drug-Drug Interactions</td><td>625</td><td>25.0%</td></tr><tr><td>Contraindication Detection</td><td>500</td><td>20.0%</td></tr><tr><td>Homeostatic Violations</td><td>450</td><td>18.0%</td></tr><tr><td>Dosing Errors (Pediatric/Geriatric)</td><td>375</td><td>15.0%</td></tr><tr><td>Organ Function Impairment</td><td>300</td><td>12.0%</td></tr><tr><td>Multi-system Polypharmacy</td><td>250</td><td>10.0%</td></tr><tr><td>Total</td><td>2,500</td><td>100%</td></tr></table>

TABLE XX  
PERFORMANCE ACROSS DIFFERENT BASE MODELS
<table><tr><td>Model</td><td>Params</td><td>CSS</td><td>HR↓</td><td>Time</td></tr><tr><td>BioMistral-7B</td><td>7B</td><td>88.4</td><td>6.2</td><td>14.2h</td></tr><tr><td>Mistral-7B</td><td>7B</td><td>90.8</td><td>5.1</td><td>18.6h</td></tr><tr><td>LLaMA-3-8B</td><td>8B</td><td>91.2</td><td>4.9</td><td>19.8h</td></tr><tr><td>Qwen2-7B</td><td>7B</td><td>89.6</td><td>5.4</td><td>17.4h</td></tr></table>

TABLE XXI  
TAXONOMY OF FAILURE AND ROOT CAUSE ANALYSIS (TRAINING+VALIDATION SCENARIOS)
<table><tr><td>Error Type</td><td>Count</td><td>%</td><td>Primary Root Cause</td><td>Potential Mitigation</td></tr><tr><td>Graph Retrieval Failures</td><td>84</td><td>4.2</td><td>Knowledge graph coverage gaps / anatom- ical mismatch</td><td>Dynamic KG updates and continual HGNN refresh</td></tr><tr><td>Optimization Drift</td><td>62</td><td>3.1</td><td>Constraint boundary deviation in HGNN depth</td><td>Stronger KL regularization and score- gap calibration</td></tr><tr><td>Linguistic Hallucinations</td><td>56</td><td>2.8</td><td>Temporal fabrication outside graph repre- sentation</td><td>Temporal constraints and post-hoc consistency checks</td></tr><tr><td>Dosing Errors</td><td>41</td><td>2.1</td><td>Incomplete pharmacokinetic data</td><td>Weight/renal-aware dosing calcula- tors in tool loop</td></tr><tr><td>Context Ambiguity</td><td>38</td><td>1.9</td><td>Insufficient patient information</td><td>Uncertainty-aware abstention and clarification prompts</td></tr><tr><td>Novel Drug Combina- tions</td><td>29</td><td>1.5</td><td>Emerging pharmacological knowledge</td><td>Rapid ingestion of approvals and in- teraction bulletins</td></tr><tr><td>Cultural/Genetic Factors</td><td>22</td><td>1.1</td><td>Population-specific variations</td><td>Pharmacogenomic personalization</td></tr><tr><td>Total Errors</td><td>332</td><td>16.7</td><td></td><td>and subgroup calibration</td></tr></table>

![](images/404a45f6b447325393541735ec9d801ae910c856ac9109c1bd2f14171013f372.jpg)  
Follows Algorithm 2: ENTITYEXTRACT → MAPTOGRAPH → PHYSIOSCORE

Fig. 6. Step-wise toy example of constraint verification. A candidate plan is generated, entities are linked to the physiological graph, and HGNN-based checks reject the plan due to hyperkalemia-related contraindications.  
TABLE XXII  
HUMAN EVALUATION RUBRIC
<table><tr><td>Score</td><td>Criteria</td></tr><tr><td></td><td>Clinically excellent: Safe, accurate, complete, follows guidelines</td></tr><tr><td></td><td>Clinically acceptable: Minor omissions, no safety concerns</td></tr><tr><td></td><td>Borderline: Some inaccuracies, requires verification</td></tr><tr><td>54321</td><td>Concerning: Contains errors that could mislead</td></tr><tr><td></td><td>Dangerous: Contains recommendations that could cause harm</td></tr></table>

## TABLE XXIII

EVALUATION METRICS OVERVIEW
<table><tr><td>Metric</td><td>Description</td></tr><tr><td>MA</td><td>Medical Accuracy: Percentage correct on QA benchmarks</td></tr><tr><td>CSS</td><td>Clinical Safety Score: Adherence to physiological constraints</td></tr><tr><td>HR</td><td>Hallucination Rate: % responses with impossible statements</td></tr><tr><td>DID</td><td>Drug Interaction Detection: F1 for contraindicated combinations</td></tr><tr><td>PC</td><td>Physiological Consistency: Mean HGNN feasibility score</td></tr><tr><td>BLEU-Med</td><td>Modified BLEU with medical term weighting</td></tr><tr><td>ROUGE-L</td><td>Longest common subsequence for generation quality</td></tr><tr><td>BertScore</td><td>Semantic similarity using BioBERT embeddings</td></tr></table>