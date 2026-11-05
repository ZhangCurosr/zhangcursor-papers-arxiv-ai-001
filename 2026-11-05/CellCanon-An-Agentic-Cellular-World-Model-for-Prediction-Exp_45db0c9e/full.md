# CellCanon: An Agentic Cellular World Model for Prediction, Explanation, and Observation-Driven Updating

Zhaokang Liang<sup>∗</sup> Shuyang Zhuang<sup>∗</sup> Xiaoran Jiao Hao Chen<sup>†</sup> Zhejiang University

## Abstract

Biological discovery requires using molecular knowledge and new measurements to infer responses beyond the conditions already observed. We introduce Cell-Canon, the first agentic cellular world model to unify single-cell expression prediction, evidence-grounded explanation, and observation-driven updating. Its predictive core reuses condition-level biological evidence across gene-specific numerical readouts to guide single-cell expression generation. A shared world state links predictions, acquired measurements, and source-linked claims, supporting explanation and revision. On 339 held-out SciPlex3 conditions across three cell lines, CellCanon improves differential-expression AUPRC and Macro-F1 over the best baseline for each metric by 63.3% and 17.3%, respectively, and direction Macro-F1 by 7.2%. Relative to gene-wise reasoning on matched drug–cell cohorts, measured language-model token use falls by 99.77% for Predict and 98.41% for Explain. On the source-grounded benchmark, Explain correctly answers all answerable questions with supporting proofs and identifies all insufficient-evidence cases. In simulated experimental feedback, Update uses one measured source context per drug to exceed the best fine-tuned baselines in differential-expression AUPRC and Macro-F1 without changing model weights. Together, these capabilities support a cumulative account of cellular responses, with predictions and biological judgments that can be revised as evidence accumulates.

## 1 Introduction

Single-cell perturbation screens reveal drug-induced transcriptional changes but leave many compounds and cell contexts unmeasured [Srivatsan et al., 2020, Bunne et al., 2024]. A cellular world model must combine molecular knowledge and new measurements to predict unobserved responses while keeping biological judgments inspectable and revisable.

Numerical perturbation models efficiently predict expression profiles [Lotfollahi et al., 2023, Hetzel et al., 2022, Piran et al., 2024, Qi et al., 2024]. Gene-wise language reasoning combines molecular knowledge and retrieved examples to infer response labels [Wei et al., 2026]. Genes in one drug–cell condition share biological background but require distinct response estimates. We reuse this evidence interpretation across gene-specific numerical readouts, with distinct supervision for continuous response, differential expression (DE), and conditional direction (DIR).

We introduce CellCanon, the first agentic cellular world model to unify single-cell expression prediction, evidence-grounded explanation, and observation-driven updating. Encoded molecular evidence and a training-response atlas feed continuous multilayer perceptron (MLP) and classification readouts, whose gene-specific outputs guide single-cell generation (Figure 2). A shared world state links these predictions to source-linked claims and acquired measurements; observation-conditioned maps update response estimates (Figure 1).

![](images/cb4705a83b6301b83a509af23ac5eb4ce8da073bf9eea26736b263360115b6c0.jpg)  
Figure 1: An agent for prediction, explanation, and updating. Evidence and a training-response atlas condition prediction. The world state links responses, source-linked claims, and measurements. Evidence review supports explanations; Update transfers observed responses across cell contexts with fixed predictor weights.

Explain evaluates target actions and regulatory paths, providing proofs or identifying insufficient evidence. It distinguishes binding, signed regulation, and solvent-control RNA detection; claim dependencies allow revision when supporting records change.

Update uses a measured response to revise predictions for the same drug in other, still-unmeasured cell contexts. We evaluate this process by progressively revealing existing SciPlex3 measurements to simulate experimental feedback. Separate observation-state tasks assess how measurement availability changes support for biological claims.

Experiments demonstrate accurate prediction, source-grounded claim verification, and observationdriven updating with reduced token use. Ablations assess biological context and task-specific readouts. Code is available at https://github.com/ZhaokangLiang123/CellCanon.

## 2 Related Work

Predicting perturbation responses. Neural models predict perturbation responses [Lotfollahi et al., 2019, 2023, Hetzel et al., 2022, Piran et al., 2024, Qi et al., 2024]. Optimal transport models shifts between cell distributions [Bunne et al., 2023, Driessen et al., 2026]. LGR aggregates measured responses from LLM-selected neighboring drugs [Xiong et al., 2026]; MapPFN supports in-context genetic-perturbation prediction without gradient updates [Sextro et al., 2026]. Evaluations emphasize target and protocol choices [Ahlmann-Eltze et al., 2025, Viñas Torné et al., 2026].

![](images/492cff5b16e1c78ec42f6986867f079f240f8d3bc8d01531d1711529586e97ee.jpg)  
Figure 2: Shared evidence for single-cell prediction. a, Evidence compilation is shared across genes; continuous responses and DE/DIR probabilities guide generation from matched DMSO cells and doses. b, Text and structured features guide atlas retrieval and task readouts. Testing generated populations against DMSO gives final DE/DIR (Section B.4).

Agents for biological discovery. ChemCrow connects language models to chemistry tools; BioDiscoveryAgent designs genetic perturbation experiments [Bran et al., 2024, Roohani et al., 2025]. PerturbAgent organizes perturbation analysis and prediction, while VCWorld studies language-based virtual-cell reasoning [Pei et al., 2026, Wei et al., 2026]. PerTurboAgent updates phenotype predictions and prioritizes experiments from accumulated Perturb-seq observations [Hao et al., 2025]. BaiZe predicts expression and supports few-shot cross-species adaptation; BaiZe-Agent retrieves and explains saved results [Zeng et al., 2026]. CellCanon links prediction, sourcegrounded explanation, and measurement-driven revision through a shared world state.

Evidence-grounded biological explanation. GeneAgent uses domain databases and verification for gene-set interpretation [Wang et al., 2025]. Claim verification evaluates labels and rationales [Wadden et al., 2020]; citation benchmarks distinguish answer correctness from citation quality [Gao et al., 2023]. We evaluate judgments and supporting citations using biological evidence types, signed paths, and observation-driven changes. Molecular sources define reference answers and proofs.

## 3 Method

## 3.1 Cellular world model and evidence state

The shared world state connects three roles: language models interpret evidence, tools check citations and paths, and numerical models predict and update responses.

Let d denote a drug, c a cell context, and $q = ( d , c )$ a condition. Training drugs $\mathcal { D } _ { \mathrm { t 1 } }$ <sub>r</sub> supply the support atlas. Each query state contains a continuous response $\widehat { \mathbf { y } } _ { q } .$ , DE probabilities ${ \bf p } _ { q } ^ { \hat { D } E }$ , and conditional up-regulation probabilities $\mathbf { p } _ { q } ^ { \mathrm { u p } }$

$$
S _ { q } ^ { ( t ) } = \big ( \widehat { \mathbf { y } } _ { q } ^ { ( t ) } , \mathbf { p } _ { q } ^ { D E , ( t ) } , \mathbf { p } _ { q } ^ { \mathrm { u p } , ( t ) } , \mathcal { E } _ { q } , \mathcal { C } _ { q } ^ { ( t ) } , \mathcal { O } _ { t } \big ) ,\tag{1}
$$

where $\mathcal { E } _ { q }$ contains source-linked evidence, $\mathcal { C } _ { q } ^ { ( t ) }$ contains claims and their dependencies, and ${ \mathcal { O } } _ { t }$ records acquired measurements. The atlas supplies training responses; the observation record holds newly revealed measurements.

The knowledge base combines drug and gene descriptions, target and assay records, pathways, and signed regulatory relations, retaining each record’s source identifier and evidence type (Section B). Solvent-control mean expression and detection rates provide cell-specific target context. Explanations use partial evidence or complete candidate paths.

## 3.2 Predict: language-selected evidence meets an observation atlas

Biological evidence compilation. For each condition, the language model selects source-linked target evidence and qualifies its relevance and assay scope. The evidence packet combines drug knowledge and matched DMSO controls, distinguishing binding, functional action, background regulation, and control-cell expression.

Frozen text embeddings encode selected evidence and weighted target descriptions; structured features preserve evidence categories and quantitative context. Their concatenation $\mathbf { x } _ { q }$ undergoes training-fitted projection and normalization, ${ \bf z } _ { q } = { \cal P } _ { \mathrm { t r } } ( { \bf x } _ { q } )$ (Equation (6) and Section B). Source links are retained, and the representation is shared across genes.

Knowledge-conditioned retrieval. For each training drug i, the atlas stores knowledge features $\mathbf { z } _ { i c }$ and normalized continuous responses, DE labels, and up-regulated DE labels. Learned maps produce a query ${ \bf q } _ { d c } ,$ key $\mathbf { k } _ { i c } ,$ , and observation value ${ \bf v } _ { i c } .$ Retrieval combines learned compatibility with chemical, target, and mechanism-of-action similarities $s _ { r } ( d , i )$

$$
\ell _ { d i , c } = \frac { { \bf q } _ { d c } ^ { \top } { \bf k } _ { i c } } { \sqrt { H } } + \sum _ { r = 1 } ^ { 3 } \mathrm { s o f t p l u s } ( w _ { r } ) s _ { r } ( d , i ) , \qquad \alpha _ { d i , c } = \mathrm { s o f t m a x } _ { i \in A _ { d } } ( \ell _ { d i , c } ) ,\tag{2}
$$

$$
\mathbf h _ { d c } = \mathrm { L a y e r N o r m } \left( \mathbf q _ { d c } + \mathrm { s i g m o i d } ( W _ { g } \mathbf q _ { d c } + \mathbf b _ { g } ) \odot W _ { o } \sum _ { i \in \mathcal A _ { d } } \alpha _ { d i , c } \mathbf v _ { i c } \right) .\tag{3}
$$

A cell-specific map ${ \bf u } _ { d c } = \mathrm { M L P } _ { c } ( { \bf h } _ { d c } )$ produces the shared readout representation. Here $\mathbf { \mathcal { A } } _ { d }$ contains permitted training donors. Whole-drug masking excludes the query identity and its training fold during representation fitting; test-time donors come from the training atlas.

Continuous and classification readouts. A cell-specific MLP maps the shared representation to the continuous gene panel:

$$
\widehat { \mathbf { y } } _ { d c } = \pmb { \mu } _ { c } + s _ { y } \mathbf { M } \mathbf { L } \mathbf { P } _ { c } ^ { \mathrm { R N A } } ( \mathbf { u } _ { d c } ) ,\tag{4}
$$

where $\pmb { \mu } _ { c }$ and $s _ { y }$ are training response normalization statistics. We train the frontend and continuous head using mean squared error, then freeze the representation and fit independent regularized logistic heads:

$$
p _ { d c g } ^ { D E } = \mathrm { s i g m o i d } ( b _ { c g } ^ { D E } + \beta _ { c g } ^ { D E \top } \mathbf { u } _ { d c } ) , \qquad p _ { d c g } ^ { \mathsf { u p } } = \mathrm { s i g m o i d } ( b _ { c g } ^ { \mathsf { u p } } + \beta _ { c g } ^ { \mathsf { u p } \top } \mathbf { u } _ { d c } ) .\tag{5}
$$

Each gene has its own coefficients; direction supervision uses only training DE examples. A validation-fitted decoder estimates each condition’s DE count and selects genes with the highest DE probabilities; the direction head assigns up or down. Separately supervised magnitude, DE occurrence, and direction readouts jointly guide expression generation.

Single-cell generation. Matched DMSO cells, dose-dependent parameters, and the continuous and classification readouts generate single-cell expression (Figure 2). DE confidence and direction probabilities guide bounded response corrections and gene-detection rates using support-trained experts. Continuous metrics score generated expression; differential testing against DMSO yields population DE/DIR (Section B.4).

## 3.3 Explain: evidence-grounded reasoning and correction

Explain assesses source support for target action, control-cell expression, regulatory paths, and source statements, returning supported, contradicted, or insufficient evidence judgments with citations and natural-language rationales. Control-cell expression supplies cellular context; functional and binding records characterize drug–target interactions. Candidate drug–target–regulatory paths retain evidence types and edge signs. Tools enumerate unknown-sign assignments consistently across shared edges, and the language model selects eligible paths.

Review checks citations, biological interpretation, and path consistency, providing localized diagnostics for revision. Claims depend on background records, predictions, or measurements: new measurements can make claims answerable; source withdrawal can remove support. Section F specifies the source-question and observation-state tasks.

## 3.4 Update: incorporating newly measured responses

Update uses a newly measured drug response to revise predictions for the same drug in unmeasured cell contexts. The environment reveals a selected source condition’s response and DE/DIR labels.

A support-trained kernel ridge model combines measured source-response similarity with chemical and biological similarities to infer the target response (Equations (12) and (13)). The parent predictor and update maps remain fixed. One or two source-cell responses can be used; without a source measurement, the initial prediction is retained.

Support-trained classification corrections combine parent probabilities with source responses and source–target label relationships. A source-activity model adjusts the expected DE count, and updated probabilities rank the genes. Measurements and revised predictions enter successive states. Section G gives the transfer, correction, and count-model settings.

## 4 Experiments

## 4.1 Tasks, data, and evaluation

Drug-disjoint prediction. We evaluate 339 held-out SciPlex3 conditions in A549, K562, and MCF7 [Srivatsan et al., 2020], pooling 24-hour responses across doses. Continuous scores cover 2,000 genes; classification retains all reference DE positives and randomly samples up to 200 negatives with unadjusted $p > 0 . 1$ per condition (Section A). We use the DeepSeek-V4-Flash API for evidence compilation and frozen Qwen3-Embedding-0.6B embeddings [Zhang et al., 2025].

Comparators and metrics. We compare BioLORD, chemCPA, CMonge, PRnet, Squidiff [He et al., 2026], PerturbNet [Yu et al., 2025], STATE [Adduri et al., 2025], VCWorld, and Random. The first four use validation-fitted DE adapters and validation-selected decision rules. Squidiff, PerturbNet, and STATE generate cell populations, from which differential testing against matched DMSO yields DE and direction predictions (Section C). VCWorld uses DeepSeek-V4-Flash.

Continuous-response metrics include Top50 mean-expression R<sup>2</sup>, logFC Pearson, and ∆-Pearson for changes relative to DMSO; within-condition scores are averaged across test conditions. Top50/Top100 panels follow absolute reference Wilcoxon ranks. DE and DIR use AUPRC, AU-ROC, Macro-F1, accuracy, precision, and recall. DIR conditions on reference-DE queries with upregulation as positive. Joint unchanged/up/down evaluation includes detection and direction errors (Section D).

Source-grounded explanation. The benchmark contains 966 questions across 72 conditions, with answerability determined by molecular source records. Correctness requires a supported/contradicted judgment and complete proof, or abstention for insufficient evidence (Section F).

Observation-feedback protocol. We simulate experimental feedback by revealing existing Sci-Plex3 measurements under fixed acquisition schedules. Each of three partitions withholds one target cell context per test drug; the other two supply measurements. One observation reveals a drug–cell response at 24 hours across all four doses. CellCanon is evaluated with and without Update under identical schedules and budgets. Feedback scores use condition-level response and classification readouts, averaged across partitions. Six response models are fine-tuned on revealed source-cell populations at 48 observations and one source context per drug. Target conditions remain unobserved (Sections G and H.1).

## 4.2 Predicting expression and differential responses

CellCanon achieves DE AUPRC of 0.8067 and Macro-F1 of 0.8164, versus 0.4938 and 0.6961 for the strongest baselines (Table 1). Joint unchanged/up/down Macro-F1 reaches 0.7420 versus 0.5400. Its DE precision of 0.8370 and recall of 0.5739 indicate more precise but less sensitive predictions than PRnet. Shared evidence analysis supports this gene-specific accuracy with substantially lower language-model token use (Section 4.6).

CellCanon reaches logFC Pearson of 0.6944 and ∆-Pearson of 0.3254, versus 0.4306 and 0.2543 for chemCPA (Table 2 and Figure 3). Section D.2 reports additional response-focused comparisons.

The classification advantage extends across drugs (Figure 4a–c). Across 112 drugs, median DE and joint gains over PRnet are 0.1322 (95% CI 0.1074–0.1553) and 0.1456 (0.1243–0.1699), respectively; 10,000 paired drug-bootstrap resamples support robustness to test-drug composition. Joint median gains remain positive across all four response-strength strata. DIR gains concentrate in stronger responses, while the strongest-response DE interval crosses zero.

Table 1: Differential-response prediction on the 2,000-gene SciPlex3 panel. DE/joint use sampled test queries; DIR uses reference-DE queries with up-regulation as positive. Joint classes are unchanged/up/down. F1 is Macro-F1 unless specified; Acc. is accuracy. VCWorld and Random use binary ranking scores (Section C). Higher is better. In all tables, bold and underlining mark best and second-best scores.
<table><tr><td>Metric</td><td>BioLORD</td><td>chemCPA</td><td>CMonge</td><td>PRnet</td><td>Squidiff</td><td>PerturbNet</td><td>STATE</td><td>VCWorld</td><td>Random</td><td>CellCanon</td></tr><tr><td>DE</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AUPRC</td><td>0.2694</td><td>0.1345</td><td>0.3474</td><td>0.4058</td><td>0.0875</td><td>0.4870</td><td>0.4255</td><td>0.4938</td><td>0.3666</td><td>0.8067</td></tr><tr><td>AUROC</td><td>0.7734</td><td>0.4536</td><td>0.8140</td><td>0.8572</td><td>0.1279</td><td>0.8807</td><td>0.8371</td><td>0.6527</td><td>0.4998</td><td>0.9449</td></tr><tr><td>F1</td><td>0.6184</td><td>0.4422</td><td>0.6479</td><td>0.6961</td><td>0.1086</td><td>0.6732</td><td>0.6040</td><td>0.5500</td><td>0.4324</td><td>0.8164</td></tr><tr><td>Acc.</td><td>0.6886</td><td>0.4616</td><td>0.7402</td><td>0.7698</td><td>0.1144</td><td>0.8453</td><td>0.8275</td><td>0.6259</td><td>0.4996</td><td>0.9163</td></tr><tr><td>Precision</td><td>0.3125</td><td>0.2092</td><td>0.3432</td><td>0.3941</td><td>0.1056</td><td>0.5060</td><td>0.4093</td><td>0.2481</td><td>0.1555</td><td>0.8370</td></tr><tr><td>DIR</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>AUPRC</td><td>0.9042</td><td>0.7772</td><td>0.7671</td><td>0.8165</td><td>0.8222</td><td>0.7476</td><td>0.8652</td><td>0.6619</td><td>0.6558</td><td>0.9402</td></tr><tr><td>AUROC</td><td>0.8810</td><td>0.6996</td><td>0.7293</td><td>0.7589</td><td>0.7822</td><td>0.6744</td><td>0.8305</td><td>0.5459</td><td>0.4991</td><td>0.9255</td></tr><tr><td>F1</td><td>0.7903</td><td>0.6753</td><td>0.6699</td><td>0.7006</td><td>0.5852</td><td>0.6425</td><td>0.7418</td><td>0.5241</td><td>0.4982</td><td>0.8472</td></tr><tr><td>Acc.</td><td>0.7903</td><td>0.6756</td><td>0.6700</td><td>0.7006</td><td>0.6482</td><td>0.6461</td><td>0.7421</td><td>0.5316</td><td>0.4991</td><td>0.8480</td></tr><tr><td>Precision</td><td>0.8571</td><td>0.7556</td><td>0.7405</td><td>0.7582</td><td>0.6121</td><td>0.7589</td><td>0.7891</td><td>0.6104</td><td>0.5409</td><td>0.8686</td></tr><tr><td colspan="2">Joint response: unchanged/up/down</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td>F1</td><td>0.4900</td><td>0.3821</td><td>0.4686</td><td>0.5400</td><td>0.0944</td><td>0.4910</td><td>0.4885</td><td>0.3767</td><td>0.2881</td><td>0.7420</td></tr><tr><td>Acc.</td><td>0.6564</td><td>0.4210</td><td>0.6970</td><td>0.7293</td><td>0.0854</td><td>0.8254</td><td>0.8252</td><td>0.5777</td><td>0.4607</td><td>0.9104</td></tr><tr><td>Unchanged F1</td><td>0.7821</td><td>0.5461</td><td>0.8281</td><td>0.8458</td><td>0.0363</td><td>0.9103</td><td>0.9015</td><td>0.7348</td><td>0.6277</td><td>0.9518</td></tr><tr><td>Up F1</td><td>0.3614</td><td>0.4260</td><td>0.2582</td><td>0.3620</td><td>0.1277</td><td>0.1407</td><td>0.3171</td><td>0.1758</td><td>0.1258</td><td>0.6102</td></tr><tr><td>Down F1</td><td>0.3264</td><td>0.1742</td><td>0.3196</td><td>0.4121</td><td>0.1191</td><td>0.4220</td><td>0.2470</td><td>0.2196</td><td>0.1107</td><td>0.6640</td></tr></table>

![](images/6e3e0dd52372eed1a6a2db9adf83dbf1a6ec83c118e6ab2155b5071a558cf395.jpg)

![](images/311de346cb2980aa37973fd45e4407e5594d3642de1ed51071ffb80d6d4fc868.jpg)

![](images/cebe8677f58de04a174c731c5472790eba793e52a74995f0ffe807c87e658c13.jpg)  
Figure 3: Response-focused prediction. a, Per-condition logFC Pearson on 2,000 genes: boxes show quartiles, gray dashed lines medians, whiskers the furthest points within 1.5 interquartile ranges, dots outliers, and model symbols means. b, ∆-Pearson on all 2,000 genes. c, ∆-Spearman across gene panels. Delta subtracts matched DMSO means. Panels use 339 conditions (333 for PerturbNet); b–c show means. Top-k genes follow absolute reference Wilcoxon ranks.

Table 2: Continuous-response prediction. Scores average within-condition comparisons over evaluated test conditions. Expression $\mathtt { R } ^ { 2 }$ and ∆-Spearman use the top 50 and 100 reference-ranked genes, respectively; logFC Pearson and ∆-Pearson use all 2,000. Delta subtracts matched DMSO means. Higher is better.
<table><tr><td>Metric</td><td>BioLORD</td><td>chemCPA</td><td>CMonge</td><td>PRnet</td><td>Squidiff</td><td>PerturbNet</td><td>STATE</td><td>DMSO reference</td><td>CellCanon</td></tr><tr><td>Top50 R²</td><td>0.9197</td><td>0.9026</td><td>0.7359</td><td>0.9039</td><td>0.8590</td><td>0.4461</td><td>0.9063</td><td>0.8880</td><td>0.9145</td></tr><tr><td>logFC Pearson</td><td>0.2935</td><td>0.4306</td><td>0.2392</td><td>0.2061</td><td>0.2034</td><td>0.4268</td><td>0.3046</td><td></td><td>0.6944</td></tr><tr><td>∆-Pearson</td><td>0.2538</td><td>0.2543</td><td>0.1484</td><td>0.1528</td><td>0.0703</td><td>0.1500</td><td>0.2401</td><td></td><td>0.3254</td></tr><tr><td>Top100 ∆-Spearman</td><td>0.3199</td><td>0.3632</td><td>0.1822</td><td>0.2089</td><td>0.2396</td><td>0.2062</td><td>0.2809</td><td>一</td><td>0.3809</td></tr></table>

![](images/a0e222a53371bc2a12d1c4aacb8d9477308a1f9581863a7a246ef0ffb6442317.jpg)

![](images/968311e262f087bad9b8255920d007efd3d5bff7fff05a8410015fa9978b2598.jpg)

c  
![](images/0b63c793d297ef009252ad4881a61975bec9781cff4e3d98f90ba86ed9579ab6.jpg)

![](images/b5dcbbfb34760a3b321c04abfc420ed4f3e6fcda122bfca07a082d108e801a82.jpg)

e  
![](images/1bd830eb727764e2392278da4a1f669e899ffdc23fe5a74cf2eac675a0b6ceea.jpg)  
Figure 4: Classification robustness and observation feedback. a–c, Expression-derived Macro-F1 gains over fixed baselines (PRnet: DE/joint; BioLORD: DIR; Table 1): a, drug medians by responsestrength quartile, with 95% paired drug-bootstrap intervals; b, paired drug gains, medians and interquartile ranges; $\mathbf { c } ,$ win/tie/loss fractions. Scores pool three cell contexts and average equivalent identifiers (112 drugs; 108 for DIR; Section D.1). d,e, Updated condition-head DE AUPRC/Macro-F1: thin lines show three target-cell partitions, thick lines their mean, dashed lines the no-Update mean. Budgets 112/224 reveal one/two source contexts per canonical drug (Section G).

Beyond the 2,000-gene continuous head, CellCanon classifies 931 additional LINCS landmark genes using their support DE/DIR labels without their continuous response targets. Continuous generation for these genes uses a separate model fitted to their continuous support responses (Section J).

## 4.3 Updating unmeasured conditions from new observations

Six response models are fine-tuned on the revealed populations; CellCanon uses its fixed updater. With one source context per drug, CellCanon reaches DE AUPRC/Macro-F1 of 0.9077/0.8261, compared with 0.6279/0.7630 for the strongest fine-tuned comparator (Table 3). Joint unchanged/up/down Macro-F1 reaches 0.7431 versus 0.6116.

At 48 observations, CellCanon’s DE Macro-F1 rises from 0.8199 to 0.8393 (Figure 4e). Section H.1 and Table 20 report fine-tuning settings, intermediate budgets, and changes from each method’s initial prediction.

Table 3: Prediction after observing one source context per drug. Absolute scores average three source–target partitions over 2,000 genes, using Table 1 metrics. Fine-tuned comparators use differential tests on generated expression; CellCanon uses updated condition heads (Section H.1). Higher is better.
<table><tr><td>Metric</td><td>BioLORD</td><td>chemCPA</td><td>CMonge</td><td>PRnet</td><td>PerturbNet</td><td>STATE</td><td>CellCanon</td></tr><tr><td colspan="8">DE</td></tr><tr><td>AUPRC</td><td>0.1337</td><td>0.3188</td><td>0.1306</td><td>0.0948</td><td>0.6279</td><td>0.5535</td><td>0.9077</td></tr><tr><td>AUROC</td><td>0.3925</td><td>0.8069</td><td>0.3350</td><td>0.1467</td><td>0.9135</td><td>0.8663</td><td>0.9792</td></tr><tr><td>F1</td><td>0.2627</td><td>0.5786</td><td>0.1965</td><td>0.1241</td><td>0.7630</td><td>0.7003</td><td>0.8261</td></tr><tr><td>Acc.</td><td>0.2680</td><td>0.7199</td><td>0.1976</td><td>0.1338</td><td>0.8637</td><td>0.8555</td><td>0.9185</td></tr><tr><td>Precision</td><td>0.1236</td><td>0.2767</td><td>0.1023</td><td>0.1271</td><td>0.5846</td><td>0.5925</td><td>0.9021</td></tr><tr><td colspan="8">DIR</td></tr><tr><td>AUPRC</td><td>0.7850</td><td>0.6807</td><td>0.7836</td><td>0.8070</td><td>0.8305</td><td>0.9368</td><td>0.9313</td></tr><tr><td>AUROC</td><td>0.7425</td><td>0.6040</td><td>0.7269</td><td>0.7689</td><td>0.7607</td><td>0.9185</td><td>0.9167</td></tr><tr><td>F1</td><td>0.6691</td><td>0.5661</td><td>0.6608</td><td>0.6488</td><td>0.6889</td><td>0.8323</td><td>0.8453</td></tr><tr><td>Acc.</td><td>0.6709</td><td>0.5892</td><td>0.6693</td><td>0.6785</td><td>0.6937</td><td>0.8329</td><td>0.8460</td></tr><tr><td>Precision</td><td>0.7221</td><td>0.8084</td><td>0.6792</td><td>0.6530</td><td>0.8606</td><td>0.8750</td><td>0.8840</td></tr><tr><td colspan="8">Joint response: unchanged/up/down</td></tr><tr><td>F1</td><td>0.2400</td><td>0.4210</td><td>0.1859</td><td>0.1746</td><td>0.5983</td><td>0.6116</td><td>0.7431</td></tr><tr><td>Acc.</td><td>0.2460</td><td>0.6944</td><td>0.1769</td><td>0.1010</td><td>0.8365</td><td>0.8531</td><td>0.9100</td></tr><tr><td>Unchanged F1</td><td>0.3235</td><td>0.8224</td><td>0.2239</td><td>0.0327</td><td>0.9173</td><td>0.9158</td><td>0.9528</td></tr><tr><td>Up F1</td><td>0.1058</td><td>0.1092</td><td>0.1073</td><td>0.1364</td><td>0.2952</td><td>0.5220</td><td>0.6008</td></tr><tr><td>Down F1</td><td>0.2908</td><td>0.3312</td><td>0.2265</td><td>0.3548</td><td>0.5824</td><td>0.3969</td><td>0.6756</td></tr></table>

## 4.4 Evidence-linked explanations with explicit answerability

For TSA (Figure 5), functional records support HDAC1 inhibition and K562 controls establish HDAC1 expression. Reviewing unknown-sign alternatives changes two reverse-sign pathway judgments from contradicted to insufficient evidence. On the source-grounded benchmark, Explain correctly answers all answerable questions with complete proofs, abstains on all insufficient-evidence cases, and retrieves valid paths among the top three for all eligible conditions (Table 4). Observation-state and withdrawal tests track revisions as measurements arrive or sources are removed (Section F).

Table 4: Source-grounded explanation. Questions cover 72 conditions; path success uses the 63 eligible conditions.
<table><tr><td>Endpoint Correct / total</td></tr><tr><td>Label + complete proof</td></tr><tr><td>561 / 561 Correct abstention 405 / 405</td></tr><tr><td></td></tr><tr><td>Valid source path@3 63 / 63</td></tr></table>

## 4.5 Biological context and independent readouts

Three ablations assess biological information and task-specific supervision (Table 5). Removing biological annotations lowers DE Macro-F1 from 0.8077 to 0.7117. Removing measured DMSO expression and detection lowers ∆-Pearson from 0.3074 to 0.2810. Replacing independent classification heads with continuous-response scores lowers DE Macro-F1 to 0.5278 with validation-selected thresholds, supporting distinct supervision for response magnitude and DE occurrence. Paired drug-bootstrap intervals for these decreases exclude zero (Section E).

Table 5: Predict component ablations.  
Pre-generation readouts on 339 conditions and 2,000 genes. F1 is macro-averaged; the atlas is retained in the context comparison.
<table><tr><td>Biological annotations Full predictor Without annotations</td><td>DE F1 DIR F1 0.8077 0.8393 0.7117 0.7900</td></tr><tr><td>Measured context With DMSO features Without DMSO features</td><td>∆-Pearson ∆-Spearman 0.3074 0.2383 0.2810 0.2292</td></tr><tr><td>Task readouts</td><td>DE F1 DIR F1 0.8077</td></tr><tr><td>Independent heads</td><td>0.8393</td></tr><tr><td>Continuous + count</td><td>0.4538 0.8134</td></tr><tr><td>Continuous + threshold</td><td>0.5278 0.8134</td></tr><tr><td></td><td></td></tr></table>

![](images/aa344fd0dbf1a28a35f51fc986c44e48dc5042bd69412aeff9dea04aded9cd87.jpg)  
Figure 5: Explain and Update for Trichostatin A (TSA). HDAC1 evidence review changes two K562 pathway judgments from contradicted to insufficient evidence (purple loop). A separate Update experiment revises the K562 prediction from an A549 measurement. The blue loop indicates further observations (Section F.4).

## 4.6 Reusing evidence analysis reduces token use

Evidence analysis is shared across genes within a condition; explanation is invoked separately. Relative to VCWorld’s gene-wise prediction and chain-of-thought explanation on matched drug–cell cohorts, CellCanon reduces language-model token use by 99.77% for Predict across 339 conditions and 98.41% for Explain across 72 conditions. Numerical Update reuses compiled evidence and requires no additional language-model calls. Section I gives token totals and cohort definitions.

## 5 Discussion and Conclusion

CellCanon connects accurate single-cell response prediction, source-grounded explanation, and observation-driven updating in an agentic cellular world model. Shared evidence supports genespecific readouts and generation with substantially lower language-model token use. The world state links predictions to the evidence and observations needed to check and revise them. In simulated experimental feedback, measured responses improve predictions for unobserved cell contexts of the same drug.

## Acknowledgement

We thank Prof. Chunhua Shen for his close supervision and guidance throughout this work.

## References

Abhinav K. Adduri, Dhruv Gautam, Beatrice Bevilacqua, Alishba Imran, Rohan Shah, Mohsen Naghipourfar, Noam Teyssier, Rajesh Ilango, Sanjay Nagaraj, Mingze Dong, Chiara Ricci-Tam, Christopher Carpenter, Vishvak Subramanyam, Aidan Winters, Sravya Tirukkovular, Jeremy Sullivan, Brian S. Plosky, Basak Eraslan, Nicholas D. Youngblut, Jure Leskovec, Luke A. Gilbert, Silvana Konermann, Patrick D. Hsu, Alexander Dobin, Dave P. Burke, Hani Goodarzi, and Yusuf H. Roohani. Predicting cellular responses to perturbation across diverse contexts with State. bioRxiv, 2025. doi: 10.1101/2025.06.26.661135.

Constantin Ahlmann-Eltze, Wolfgang Huber, and Simon Anders. Deep-learning-based gene perturbation effect prediction does not yet outperform simple linear baselines. Nature Methods, 22: 1657–1661, 2025. doi: 10.1038/s41592-025-02772-6.

Andres M. Bran, Sam Cox, Oliver Schilter, Carlo Baldassari, Andrew D. White, and Philippe Schwaller. Augmenting large language models with chemistry tools. Nature Machine Intelligence, 6:525–535, 2024. doi: 10.1038/s42256-024-00832-8.

Charlotte Bunne, Stefan G. Stark, Gabriele Gut, Jacobo Sarabia del Castillo, Mitch Levesque, Kjong-Van Lehmann, Lucas Pelkmans, Andreas Krause, and Gunnar Rätsch. Learning single-cell perturbation responses using neural optimal transport. Nature Methods, 20:1759–1768, 2023. doi: 10.1038/s41592-023-01969-x.

Charlotte Bunne, Yusuf Roohani, Yanay Rosen, Ankit Gupta, Xikun Zhang, Marcel Roed, Theo Alexandrov, Mohammed AlQuraishi, Patricia Brennan, Daniel B. Burkhardt, et al. How to build the virtual cell with artificial intelligence: Priorities and opportunities. Cell, 187(25):7045–7063, 2024. doi: 10.1016/j.cell.2024.11.015.

Alice Driessen, Dhruva Abhijit Rajwade, Benedek Harsanyi, Marianna Rapsomaniki, and Jannis Born. Conditional Monge Gap enables generalizable single-cell perturbation modelling. Nature Machine Intelligence, 8:984–996, 2026. doi: 10.1038/s42256-026-01242-8.

Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling large language models to generate text with citations. In Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, 2023. doi: 10.18653/v1/2023.emnlp-main.398. URL https://aclanthology.org/2023.emnlp-main.398/.

Luz Garcia-Alonso, Christian H. Holland, Mahmoud M. Ibrahim, Denes Turei, and Julio Saez-Rodriguez. Benchmark and integration of resources for the estimation of human transcription factor activities. Genome Research, 29(8):1363–1375, 2019. doi: 10.1101/gr.240663.118.

Minsheng Hao, Yongju Lee, Hanchen Wang, Gabriele Scalia, and Aviv Regev. PerTurboAgent: An LLM-based agent for designing iterative Perturb-Seq experiments. In Proceedings of the 20th Machine Learning in Computational Biology meeting, volume 311 of Proceedings of Machine Learning Research, pages 44–64, 2025. URL https://proceedings.mlr.press/v311/ hao25b.html.

Siyu He, Yuefei Zhu, Daniel Naveed Tavakol, Haotian Ye, Yeh-Hsing Lao, Zixian Zhu, Cong Xu, Shradha Chauhan, Guy Garty, Raju Tomer, Gordana Vunjak-Novakovic, James Zou, Elham Azizi, and Kam W. Leong. Squidiff: predicting cellular development and responses to perturbations using a diffusion model. Nature Methods, 23:65–77, 2026. doi: 10.1038/s41592-025-02877-y.

Leon Hetzel, Simon Boehm, Niki Kilbertus, Stephan Günnemann, Mohammad Lotfollahi, and Fabian J. Theis. Predicting cellular responses to novel drug perturbations at a single-cell resolution. In Advances in Neural Information Processing Systems, volume 35, pages 26711– 26722, 2022. URL https://proceedings.neurips.cc/paper\_files/paper/2022/hash/ aa933b5abc1be30baece1d230ec575a7-Abstract-Conference.html.

Mohammad Lotfollahi, F. Alexander Wolf, and Fabian J. Theis. scGen predicts single-cell perturbation responses. Nature Methods, 16:715–721, 2019. doi: 10.1038/s41592-019-0494-8.

Mohammad Lotfollahi, Anna Klimovskaia Susmelj, Carlo De Donno, Leon Hetzel, Yuge Ji, Ignacio L. Ibarra, Sanjay R. Srivatsan, Mohsen Naghipourfar, Riza M. Daza, Beth Martin, Jay Shendure, José L. McFaline-Figueroa, Pierre Boyeau, F. Alexander Wolf, Nafissa Yakubova, Stephan Günnemann, Cole Trapnell, David Lopez-Paz, and Fabian J. Theis. Predicting cellular responses to complex perturbations in high-throughput screens. Molecular Systems Biology, 19:e11517, 2023. doi: 10.15252/msb.202211517.

Kanglu Pei, Sisi Qu, Philip H. S. Torr, Jonathan G. Hedley, and Christian Schroeder de Witt. PerturbAgent: An agentic AI system for analysis and prediction of genetic perturbations. In AAAI 2026 Workshop XAI4Science, 2026. URL https://openreview.net/pdf?id=FfHATDCkCQ.

Zoe Piran, Niv Cohen, Yedid Hoshen, and Mor Nitzan. Disentanglement of single-cell data with biolord. Nature Biotechnology, 42:1678–1683, 2024. doi: 10.1038/s41587-023-02079-x.

Xiaoning Qi, Lianhe Zhao, Chenyu Tian, Yueyue Li, Zhen-Lin Chen, Peipei Huo, Runsheng Chen, Xiaodong Liu, Baoping Wan, Shengyong Yang, and Yi Zhao. Predicting transcriptional responses to novel chemical perturbations using deep generative model for drug discovery. Nature Communications, 15:9256, 2024. doi: 10.1038/s41467-024-53457-1.

Eliot Ragueneau, Chuqiao Gong, Pierre Sinquin, Cristoffer Sevilla, Deidre Beavers, Alexander Grentner, Johannes Griss, Gregory F J Hogue, Nancy T Li, Lisa Matthews, Bruce May, Marija Milacic, Helia Mohammadi, Robert Petryszak, Karen Rothfels, Veronica Shamovsky, Ralf Stephan, Krishna Tiwari, Joel Weiser, Adam Wright, Marc Gillespie, Guanming Wu, Lincoln Stein, Henning Hermjakob, and Peter D’Eustachio. The Reactome knowledgebase 2026. Nucleic Acids Research, 54(D1):D673–D681, 2026. doi: 10.1093/nar/gkaf1223.

Yusuf Roohani, Andrew Lee, Qian Huang, Jian Vora, Zachary Steinhart, Kexin Huang, Alexander Marson, Percy Liang, and Jure Leskovec. BioDiscoveryAgent: An AI agent for designing genetic perturbation experiments. In International Conference on Learning Representations, 2025. URL https://proceedings.iclr.cc/paper\_files/paper/2025/hash/ 4252dc94531833029000f85dc5fac792-Abstract-Conference.html.

Marvin Sextro, Weronika Kłos, and Gabriel Dernbach. MapPFN: Learning causal perturbation maps in context. arXiv preprint arXiv:2601.21092, 2026. URL https://arxiv.org/abs/ 2601.21092.

Sanjay R. Srivatsan, José L. McFaline-Figueroa, Vijay Ramani, Lauren Saunders, Junyue Cao, Jonathan Packer, Hannah A. Pliner, Dana L. Jackson, Riza M. Daza, Lena Christiansen, Fan Zhang, Frank Steemers, Jay Shendure, and Cole Trapnell. Massively multiplex chemical transcriptomics at single-cell resolution. Science, 367:45–51, 2020. doi: 10.1126/science.aax6234.

Aravind Subramanian, Rajiv Narayan, Steven M. Corsello, David D. Peck, et al. A next generation connectivity map: L1000 platform and the first 1,000,000 profiles. Cell, 171(6):1437–1452.e17, 2017. doi: 10.1016/j.cell.2017.10.049. URL https://doi.org/10.1016/j.cell.2017.10. 049.

Dénes Türei, Alberto Valdeolivas, Lejla Gul, Nicolàs Palacio-Escat, Michal Klein, Olga Ivanova, Márton Ölbei, Attila Gábor, Fabian Theis, Dezso Módos, Tamás Korcsmáros, and Julio Saez-˝ Rodriguez. Integrated intra- and intercellular signaling knowledge for multicellular omics analysis. Molecular Systems Biology, 17(3):e9923, 2021. doi: 10.15252/msb.20209923.

Ramon Viñas Torné, Maciej Wiatrak, Zoe Piran, Shuyang Fan, Liangze Jiang, Sarah A. Teichmann, Mor Nitzan, and Maria Brbic. Systema: a framework for evaluating genetic perturbation response´ prediction beyond systematic variation. Nature Biotechnology, 44:1050–1059, 2026. doi: 10. 1038/s41587-025-02777-8.

David Wadden, Shanchuan Lin, Kyle Lo, Lucy Lu Wang, Madeleine van Zuylen, Arman Cohan, and Hannaneh Hajishirzi. Fact or fiction: Verifying scientific claims. In Proceedings ofthe 2020 Conference on Empirical Methods in Natural Language Processing, pages 7534–7550, 2020. doi: 10.18653/v1/2020.emnlp-main.609. URL https://aclanthology.org/2020.emnlp-main. 609/.

Zhizheng Wang, Qiao Jin, Chih-Hsuan Wei, Shubo Tian, Po-Ting Lai, Qingqing Zhu, Chi-Ping Day, Christina Ross, Robert Leaman, and Zhiyong Lu. GeneAgent: self-verification language agent for gene-set analysis using domain databases. Nature Methods, 22(8):1677– 1685, 2025. doi: 10.1038/s41592-025-02748-6. URL https://www.nature.com/articles/ s41592-025-02748-6.

Zhijian Wei, Runze Ma, Zichen Wang, Zhongmin Li, Shuotong Song, and Shuangjia Zheng. VC-World: A biological world model for virtual cell simulation. In International Conference on Learning Representations, 2026. URL https://proceedings.iclr.cc/paper\_files paper/2026/hash/767ff070c27c8954babe74eccf3fbe91-Abstract-Conference.html.

Betty Xiong, Jan-Christian Huetter, Gabriele Scalia, Tommaso Biancalani, and Sepideh Maleki. LLM-guided retrieval for prediction of molecular perturbation responses. arXiv preprint arXiv:2608.01734, 2026. URL https://arxiv.org/abs/2608.01734.

Hengshi Yu, Weizhou Qian, Yuxuan Song, and Joshua D. Welch. PerturbNet predicts single-cell responses to unseen chemical and genetic perturbations. Molecular Systems Biology, 2025. doi: 10.1038/s44320-025-00131-3.

Barbara Zdrazil, Eloy Felix, Fiona Hunter, Emma J Manners, James Blackshaw, Sybilla Corbett, Marleen de Veij, Harris Ioannidis, David Mendez Lopez, Juan F Mosquera, Maria Paula Magarinos, Nicolas Bosc, Ricardo Arcila, Tevfik Kizilören, Anna Gaulton, A Patrícia Bento, Melissa F Adasme, Peter Monecke, Gregory A Landrum, and Andrew R Leach. The ChEMBL database in 2023: a drug discovery platform spanning multiple bioactivity data types and time periods. Nucleic Acids Research, 52(D1):D1180–D1192, 2024. doi: 10.1093/nar/gkad1004.

Qingwu Zeng, Wenxiang Cai, Renchong Tian, Qi Wang, Dongzhan Zhou, Minting Pan, Hai Yang, Zhe Liu, Guan Ning Lin, and Zhe Wang. BaiZe: A multi-view dynamic framework for simulating and interpreting cellular responses across perturbation contexts. bioRxiv, 2026. doi: 10.64898/2026.07.15.738608. URL https://www.biorxiv.org/content/10.64898/2026. 07.15.738608v1.

Yanzhao Zhang, Mingxin Li, Dingkun Long, Xin Zhang, Huan Lin, Baosong Yang, Pengjun Xie, An Yang, Dayiheng Liu, Junyang Lin, Fei Huang, and Jingren Zhou. Qwen3 Embedding: Advancing text embedding and reranking through foundation models. arXiv preprint arXiv:2506.05176, 2025. URL https://arxiv.org/abs/2506.05176.

## A Data and prediction targets

The SciPlex3 evaluation uses A549, K562, and MCF7 at 24 hours, with the available dose conditions pooled into a drug–cell response. Each continuous target is a 2,000-gene log-fold-change vector relative to the corresponding control. The classification panel contains these genes and 931 additional genes with differential-expression and direction labels. The additional panel contribute classification supervision and atlas values, while the continuous output dimension remains 2,000.

DE labels compare pooled 24-hour treated cells with same-cell-line DMSO using Wilcoxon without tie correction, BH adjustment over 2,931 genes, adjusted $p < 0 . 0 5$ , and $| \log _ { 2 } \mathrm { F C } | > 0 . 2 5$

Drug identity resolution precedes support-atlas construction. The predictor uses 54 support drugs and 19 validation drugs. The test set has 113 recorded identifiers corresponding to 112 canonical identities, yielding 339 identifier–cell conditions. Canonically equivalent identities are excluded together when a training query or fold is masked. The 2,000-gene evaluation includes 678,000 condition–gene values. The sampled classification masks contain 55,598 queries in this panel and 24,630 additional queries, with 8,651 and 3,777 DE positives, respectively. For each drug–cell condition, the sampled endpoint retains all DE positives and up to 200 negatives with unadjusted $p > 0 . 1$ , sampled with seed 42. The 2,000-gene and additional-gene panels are scored separately.

The prediction target is the population response within each drug–cell condition. The shared representation combines drug knowledge and control-cell context with pooled-dose responses; the subsequent cell generator takes explicit dose inputs. All three cell lines are represented during training.

## B Predict implementation and training

## B.1 Evidence compilation and encoding

The biological knowledge base includes records derived from ChEMBL [Zdrazil et al., 2024], Reactome [Ragueneau et al., 2026], OmniPath [Türei et al., 2021], and DoRothEA [Garcia-Alonso et al., 2019]. Entity aliases connect the same compound or gene across sources.

Each condition is represented by drug descriptions, candidate targets, action or assay records, and cellular baseline features. The evidence compiler selects at most three targets and assigns each target cited evidence, relevance, and action scope. We use DeepSeek-V4-Flash through the deepseek-flash API with fixed language-model weights.

Qwen3-Embedding-0.6B [Zhang et al., 2025] supplies 1,024-dimensional frozen text embeddings. Text is divided into chunks of at most 768 tokens. The last valid token representation is normalized within each chunk; chunk representations are averaged with token-count weights and normalized again. The target-text aggregate weights target descriptions by relevance and control-cell detection. High, medium, low, and unknown relevance receive weights 1, 0.5, 0.1, and 0.25, multiplied by $0 . 2 5 + 0 . 7 5 \sqrt { \rho }$ , where $\rho$ is the target’s control detection rate; missing detection uses 0.5.

Structured features ${ \bf s } _ { q }$ are vectorized and normalized separately from the packet embedding $\mathbf { e } _ { q }$ and weighted target-description embedding $\mathbf { t } _ { q } .$ . After component normalization, the concatenation balances structured and semantic features:

$$
\mathbf { x } _ { q } = \left[ \mathbf { s } _ { q } / \sqrt { 2 } \ ; \ \mathbf { e } _ { q } / 2 \ ; \ \mathbf { t } _ { q } / 2 \right] , \qquad \mathbf { z } _ { q } = P _ { \mathrm { t r } } ( \mathbf { x } _ { q } ) \in \mathbb { R } ^ { 9 6 } .\tag{6}
$$

Truncated SVD and feature normalization are fitted using the support set. Selected targets are available for 279 of the 339 test conditions; missing evidence is encoded explicitly, and all conditions are evaluated. Citation lists are normalized to the evidence budget while retaining their source identifiers.

## B.2 Atlas frontend and output heads

The atlas value for a support condition concatenates normalized continuous response (2,000 entries), DE labels (2,931 entries), and the product of DE and up labels (2,931 entries). Its dimension is therefore 7,862. Linear query and key maps project knowledge inputs to 64 dimensions; a linear layer with GELU maps the observation values to the same width. The retrieval weights in Equation (2) include three nonnegative learned coefficients for the chemical, target, and mechanism-of-action similarity channels.

A sigmoid gate combines retrieved context with the query before layer normalization. Each cell context has a 64–64–32 MLP producing the latent representation. Its continuous head is a $3 2 \mathrm { - }$ 64–2,000 MLP with GELU and dropout 0.1. The final continuous layer is initialized to zero, so initialization corresponds to the training response mean. The response scale is a scalar learned from the support data through normalization, and the response mean is cell- and gene-specific.

Training minimizes MSE on normalized continuous responses with AdamW, learning rate 0.001, and gradient clipping at norm 5. The selected model uses hidden width 64, weight decay 0.01, and 25 epochs. We average predictions from three seeds. Five-fold drug masking restricts retrieval to observations outside the training query’s fold. Model fitting and atlas construction use support responses, and validation selects model settings and checkpoints.

After continuous training, the encoder is frozen and regularized logistic heads are fitted on its representations. For support size n, the DE prior is $\overline { { ( } } n _ { c g } ^ { D E } + 0 . 5 ) / ( n + 1 )$ ; the up prior is $( n _ { c g } ^ { \mathrm { u p } } + 0 . 5 ) / ( n _ { c g } ^ { D E } + 1 )$ Their logits initialize the offsets. The objective combines DE binary cross-entropy over all support coordinates and direction binary cross-entropy over support DE coordinates, with an $\ell _ { 2 }$ penalty on coefficients and intercepts. Head fitting uses 200 epochs; the validation-selected regularization values are 10 for DE and 100 for DIR. Classification training supplies gene-specific decision functions without changing the continuous head.

## B.3 Condition-level DE count decoding

The hard DE readout fits a ridge model to each validation condition’s true DE fraction, separately for the core and additional panels. Its inputs are the mean and standard deviation of DE probabilities; their 90th, 95th, and 99th percentiles; the fraction at least $0 . 5 ;$ and cell-context indicators. Grouped five-fold validation over drugs selects the ridge coefficient from {0.1, 1, 10, 100} by count-fraction error. At inference, the predicted fraction is bounded between one gene divided by panel size and 0.8. The rounded count determines how many highest-probability genes to select. Direction is called at up probability 0.5. All count-model parameters are fixed before test scoring.

## B.4 Probability-guided single-cell generation

Panel (a) of Figure 2 expands the shared condition state using drug identity, cell line, dose, time, matched DMSO expression rows, and stable sample identifiers. Classification probabilities guide the expression distribution; population DE/DIR is then recomputed from the generated cells. All expansion operations are numerical and reuse the compiled condition evidence.

Mean-response guidance. For each cell context and gene, support responses fit down- and upregulated mean-response experts $m ^ { - }$ and $m ^ { + }$ . We use the 54 support drugs and 5,627 core DE events, clip training logFC to [−8, 8], and shrink each mean toward its cell-context/direction mean with eight pseudo-observations. Let $p$ be the DE-head probability and u the up-head probability. The predicted-count threshold t is the midpoint between the smallest included and largest excluded probability, separately for the core and additional panels. Its relative decision weight and corrected core response are

$$
w = \frac { p ( 1 - t ) } { p ( 1 - t ) + t ( 1 - p ) } ,\tag{7}
$$

$$
r ^ { * } = r + \mathrm { c l i p } \big ( \lambda w | 2 u - 1 | \big [ ( 1 - u ) m ^ { - } + u m ^ { + } - r \big ] , - b , b \big ) .\tag{8}
$$

Here w is a relative confidence weight for the predicted gene set. The selected mean strength is $\lambda \ : = \ : 1$ , with the correction clipped to one logFC unit $( b \ : = \ : 1 )$ . An empty gene selection uses $t = \operatorname* { m a x } ( 0 . 5 , \operatorname* { m a x } p )$ ; an all-included panel uses half its minimum probability, with t bounded away from zero and one.

Detection guidance and cellular variation. Support-fitted down/up detection experts $q ^ { - } , q ^ { + }$ are cell- and gene-specific, pooling labeled support observations across doses with eight pseudoobservations of shrinkage. We mix them with the original detection target q<sub>0</sub>:

$$
q ^ { * } = ( 1 - \eta w ) q _ { 0 } + \eta w \bigl [ ( 1 - u ) q ^ { - } + u q ^ { + } \bigr ] , \qquad \eta = 1 .\tag{9}
$$

Table 6: Classification guidance connects cell generation to population decisions. All rows use the same 339 conditions and sampled queries from the 2,000-gene panel. The first row scores the shared condition heads; the other rows recompute DE/DIR from generated expression versus DMSO. F1 is Macro-F1.
<table><tr><td>Output</td><td>RMSE</td><td>Pearson</td><td>DE AUPRC</td><td>DE F1</td><td>DIR F1</td></tr><tr><td>Shared condition heads</td><td>4.4762</td><td>0.6943</td><td>0.8588</td><td>0.8077</td><td>0.8393</td></tr><tr><td>Control-conditioned expression</td><td>4.4770</td><td>0.6942</td><td>0.7982</td><td>0.6506</td><td>0.8203</td></tr><tr><td>Probability-guided expression</td><td>4.4755</td><td>0.6944</td><td>0.8067</td><td>0.8164</td><td>0.8472</td></tr></table>

The generator ranks matched control cells to realize gene-specific nonzero quotas, applies control amplitude and state modulation, and projects each generated cell to log1p CP10K. Amplitude factors are 0.25 for head-selected DE genes and 1 otherwise, with state strength 0.1. Dose-relative mean patterns are reanchored to the corrected pooled core response. Auxiliary support-fitted dose predictions supply the additional 931 genes’ means; detection guidance covers all 2,931 genes.

Selection and population readout. In this retrospective study, seven coupling settings are evaluated on 19 validation drugs, 57 pooled conditions, and 57,379 cells per setting. We select the highest expression-derived DE Macro-F1 subject to RMSE increasing by at most 1%, Pearson decreasing by at most 0.005, DIR Macro-F1 by at most 0.01, and DE AUPRC by at most 0.015 relative to the control-conditioned generator without probability guidance. The selected setting is then fixed for test generation.

Test generation covers 113 drug identifiers, three cell lines, four doses (10, 100, 1,000, and 10,000 nM), and 24 hours: 1,356 precise conditions and 341,202 cells, each with 2,931 genes. We pool generated cells across doses into 339 drug–cell populations and compare each against its same-cellline DMSO population. DE uses the reference Wilcoxon test without tie correction, BH adjustment over 2,931 genes, adjusted $p \ < \ 0 . 0 5$ , and $| \log _ { 2 } \mathrm { F C } | > 0 . 2 5$ . DE ranking uses $- \log _ { 1 0 } p .$ . DIR ranking and hard decisions use signed generated logFC. Thus, the scores in Table 1 come from expression testing, with classifier probabilities entering through generation.

Table 6 separates shared-head classification from expression-derived decisions. Probability guidance raises expression-derived DE/DIR Macro-F1 from 0.6506/0.8203 to 0.8164/0.8472, while logFC RMSE changes from 4.4770 to 4.4755. SciPlex3 supplies unpaired treated/control populations, so these classification endpoints describe population responses. Component ablations and Update evaluate the shared condition heads before this generation step.

## C External predictor configurations

The BioLORD, chemCPA, CMonge, and PRnet experiments use SciPlex3 with a 56-drug training split and seed 42. CellCanon uses 54 support drugs after canonical-identity exclusions and averages three seeds. Squidiff, PerturbNet, and STATE use the 54-drug support split, 19 validation drugs, the fixed 2,000-gene panel, and seed 42.

The BioLORD, chemCPA, CMonge, and PRnet models provide continuous response estimates. A validation-fitted logistic adapter using absolute predicted log fold change supplies their DE scores. For hard DE decisions, we enumerate validation probability thresholds and select the threshold with the highest Macro-F1 on the sampled validation queries, keeping expression predictions and adapter coefficients fixed. Selected thresholds are 0.038410 for BioLORD, 0.028758 for chemCPA, 0.041701 for CMonge, and 0.047620 for PRnet. BioLORD and CMonge use fitted direction scores. For chemCPA and PRnet, validation Macro-F1 selects between the fitted direction adapter and the native signed-logFC readout; both select signed $\mathrm { l o g F C , }$ , with positive values classified as up. This rule is fixed before test scoring. PRnet uses stochastic generation from its predicted expression distribution.

CMonge training and inference. The CMonge adaptation uses a two-layer 512-unit autoencoder with a 100-dimensional latent space. We train it for 100 complete epochs (71,600 updates), selecting the checkpoint by validation reconstruction MSE. A conditional transport map uses RDKit descriptors, log dose, and cell-line indicators and trains for 500,000 updates. Checkpoint selection every 10,000 updates uses mean-expression $\mathtt { R } ^ { 2 }$ over 228 validation drug–cell–dose conditions, with negative condition scores floored at zero for this selection criterion. The selected map is the 10,000-update checkpoint. At inference, matched DMSO cells are encoded, transported, decoded, and clipped to nonnegative expression; aggregation produces the pooled condition response. The test output contains 341,202 cells by 2,000 genes. All reported test $\mathrm { { \dot { R } } ^ { 2 } }$ values retain negative scores. On the sampled endpoint, CMonge’s DE precision/recall are 0.3432/0.7334 and accuracy is 0.7402; conditional DIR accuracy is 0.6700.

Squidiff training and inference. We use Squidiff’s drug-structure branch [He et al., 2026], conditioning on matched control expression and a 1,024-bit radius-2 feature-based Morgan fingerprint multiplied by $\log _ { 1 0 } ( 1 + \mathrm { d o s e } _ { \mathrm { n M } } )$ . The backbone has width 2,048 and three MLP blocks, with 1,000 diffusion timesteps. We apply VP output preconditioning: for $x _ { t } = a _ { t } x _ { 0 } + s _ { t } \epsilon$ , the network output v<sub>θ</sub> gives $\epsilon _ { \theta } = s _ { t } x _ { t } + a _ { t } v _ { \theta }$ , retaining the original noise-prediction MSE. Training uses 100,000 updates, batch size 128, learning rate $\mathrm { 3 \times 1 0 ^ { - 5 } } .$ , gradient clipping at 1, and EMA 0.999. We select the eligible live or EMA checkpoint with minimum validation population-expression MSE and generate with 50-step deterministic DDIM.

PerturbNet training and inference. PerturbNet [Yu et al., 2025] uses the pretrained 196- dimensional ChemicalVAE representation, a 10-dimensional scVI cell latent, and a conditional invertible network with 20 coupling blocks and hidden width 1,024. Molecular features, cell-line indicators, and log dose condition the flow. We train scVI for 700 epochs and the flow for 100 epochs, both with batch size 128; validation ELBO and population-expression MSE select epochs 695 and 80, respectively. Decoded counts use paired panel and full-library sizes sampled from training cells matched by cell line and dose, then transform to log1p CP10K. Molecular canonicalization and equivalent SMILES enumeration supply the chemical encoder inputs.

STATE training and inference. We train the STATE transition model [Adduri et al., 2025] from scratch with its eight-layer, 768-hidden-unit, 12-head bidirectional transformer and energy distribution loss. Its continuous perturbation input concatenates a 2,048-bit radius-2 Morgan fingerprint with normalized log dose. Matched DMSO cells form the input; each condition-homogeneous set contains 128 cells. Training uses batches of 96 sets, Adam at $\mathrm { { 1 0 ^ { - 4 } } }$ , and BF16 network computation with FP32 loss calculation. Early stopping ends training at epoch 540, and validation energy selects epoch 390.

For these three models, generated cells are pooled across doses within each drug–cell-line condition. DE uses the Wilcoxon test against matched DMSO with adjusted $p < 0 . 0 5$ and $| \log _ { 2 } \mathrm { F C } | > 0 . 2 5$ DE ranking uses $- \log _ { 1 0 } p ;$ direction ranking and hard decisions use signed generated logFC. Their continuous metrics use the generated population means.

For VCWorld, DeepSeek-V4-Flash receives condition evidence and returns separate DE and direction judgments with chain-of-thought rationales for each gene query. DE and up-regulation class indicators supply binary scores for AUPRC and AUROC, with tied scores retained. The same judgments supply the hard-decision metrics. CellCanon’s classification heads are trained on DE/DIR labels; the expression baselines use the response-to-label readouts described above.

Random reference. Random independently samples a Bernoulli(0.5) label for each query, using the same binary value as its ranking score and hard decision. DE uses all 55,598 sampled test queries; DIR uses the 8,651 reference-DE queries, independently of the random DE decisions. For each of 100 seeds (0–99), separate task-specific random streams draw uniform values and threshold them at 0.5. We compute each metric on the complete query set within each seed and report its mean across all seeds. Tied binary scores are retained. Table 7 gives the standard deviations across random draws. For joint evaluation, the DE decision first selects unchanged versus changed, and the independent direction decision splits changed predictions into up and down. The resulting class probabilities are 0.5, 0.25, and 0.25, respectively. Joint Macro-F1 is $0 . 2 8 8 1 \pm 0 . 0 0 1 8$ and accuracy is $0 . 4 6 0 7 \pm 0 . 0 0 1 9$ , where the standard deviations describe variation over the 100 random seeds. This reference produces classification outputs only.

Table 7: Random binary decisions. Mean ± standard deviation over 100 random seeds on the same test queries. These deviations describe variation across random draws.
<table><tr><td>Task</td><td>AUPRC</td><td>AUROC</td><td>Macro-F1</td></tr><tr><td>DE</td><td> $0 . 3 6 6 6 \pm 0 . 0 0 3 2$ </td><td> $0 . 4 9 9 8 \pm 0 . 0 0 3 0$ </td><td> $0 . 4 3 2 4 \pm 0 . 0 0 1 8$ </td></tr><tr><td>DIR</td><td> $0 . 6 5 5 8 \pm 0 . 0 0 4 0$ </td><td> $0 . 4 9 9 1 \pm 0 . 0 0 5 2$ </td><td> $0 . 4 9 8 2 \pm 0 . 0 0 5 2$ </td></tr></table>

Table 8: Continuous-response error. RMSE is computed on pooled logFC across 2,000 genes within each condition and averaged across evaluated test conditions. Lower is better; the corresponding logFC Pearson scores appear in Table 2.
<table><tr><td>Method</td><td>LogFC RMSE↓</td></tr><tr><td>BioLORD chemCPA</td><td>11.7617 9.1093</td></tr><tr><td>CMonge</td><td>15.1668</td></tr><tr><td>PRnet</td><td>15.6431</td></tr><tr><td>Squidiff</td><td>18.5063</td></tr><tr><td>PerturbNet</td><td>6.7180</td></tr><tr><td>STATE</td><td>10.2651</td></tr><tr><td>CellCanon</td><td>4.4755</td></tr></table>

## D Metric definitions and additional Predict results

Let $\mu _ { g }$ and $\hat { \mu } _ { g }$ denote observed and predicted mean expression, respectively, for gene g in a drug– cell condition. Mean expression is the pooled mean of log1p CP10K across cells. For the 50 genes with the largest absolute reference Wilcoxon statistics in each condition $( G = 5 0 )$ , the Predict comparison reports the coefficient of determination,

$$
R ^ { 2 } = 1 - \frac { \sum _ { g } ( \mu _ { g } - \hat { \mu } _ { g } ) ^ { 2 } } { \sum _ { g } ( \mu _ { g } - \bar { \mu } ) ^ { 2 } } , \qquad \bar { \mu } = G ^ { - 1 } \sum _ { g } \mu _ { g } .\tag{10}
$$

Delta Pearson is the Pearson correlation between $\Delta _ { g } = \mu _ { g } - \mu _ { q } ^ { \mathrm { D M S O } }$ and $\hat { \Delta } _ { g } = \hat { \mu } _ { g } - \mu _ { q } ^ { \mathrm { D M S O } }$ $\mathtt { R } ^ { 2 }$ uses the top 50 genes; delta Pearson uses all 2,000 evaluation genes unless a top-ranked subset is specified. Both metrics are calculated within each condition and macro-averaged across evaluated test conditions. Delta Spearman applies the same aggregation to rank correlations; correlations for constant vectors are undefined.

For the logFC targets $y _ { g }$ across all $G = 2 { , } 0 0 0$ evaluation genes, we additionally report conditionmacro RMSE, $\sqrt { G ^ { - 1 } \textstyle \sum _ { g } ( y _ { g } - \hat { y } _ { g } ) ^ { 2 } }$ , and within-condition Pearson correlation. Update MSE pools squared logFC errors within each hidden-cell partition and averages the three partition scores; its RMSE takes the square root within each partition before averaging.

DE treats differential expression as the positive class. Precision is $\mathrm { T P / ( T P + F P ) }$ , recall is $\mathrm { T P } / ( \mathrm { T P } + \mathrm { F N } )$ , and Macro-F1 averages class F1 scores. Accuracy is the fraction of query labels predicted correctly. AP, AUPRC, and AUROC are defined as

$$
\mathrm { A P } = \sum _ { k } ( R _ { k } - R _ { k - 1 } ) P _ { k } , \quad \mathrm { A U P R C } = \int _ { 0 } ^ { 1 } P ( R ) d R , \quad \mathrm { A U R O C } = \int _ { 0 } ^ { 1 } \mathrm { T P R } ( u ) d u ,\tag{11}
$$

where P and R denote precision and recall, and u denotes the false-positive rate. Ranking uses expression-test significance for final Predict DE, signed generated logFC for DIR, and head probabilities for component and Update experiments. Direction mAP averages the up- and down-class AP values on reference DE genes.

DIR metrics condition on the reference DE subset and classify up versus down. DIR AUPRC, precision, and recall treat up as positive. For joint prediction, a negative DE decision yields unchanged; a positive DE decision yields up or down according to the direction readout. We compute each class F1 from the pooled confusion matrix of sampled queries in the evaluated test conditions and average the three F1 scores. This evaluation uses the same DE thresholds and direction readouts as the separate tasks. Table 1 reports the joint scores and per-class F1 values.

Table 9: Delta response correlations relative to DMSO. Delta subtracts the same-cell control mean from both predicted and reference mean-log1p expression. Pearson and Spearman are computed across genes within each condition and averaged across evaluated test conditions. Both correlations are undefined for the constant zero-delta DMSO reference. Higher is better.
<table><tr><td>Method</td><td>All-gene ∆-Pearson</td><td>All-gene ∆-Spearman</td><td>Top100 ∆-Pearson</td><td>Top100 ∆-Spearman</td></tr><tr><td></td><td></td><td>0.1298</td><td>0.3618</td><td>0.3199</td></tr><tr><td>BioLORD chemCPA</td><td>0.2538 0.2543</td><td>0.1534</td><td>0.3888</td><td>0.3632</td></tr><tr><td>CMonge</td><td>0.1484</td><td>0.0900</td><td>0.1916</td><td>0.1822</td></tr><tr><td>PRnet</td><td>0.1528</td><td>0.0859</td><td>0.2357</td><td>0.2089</td></tr><tr><td>Squidiff</td><td>0.0703</td><td>0.0328</td><td>0.2719</td><td>0.2396</td></tr><tr><td>PerturbNet</td><td>0.1500</td><td>0.1085</td><td>0.2405</td><td>0.2062</td></tr><tr><td>STATE</td><td>0.2401</td><td>0.1814</td><td>0.3120</td><td>0.2809</td></tr><tr><td>DMSO reference</td><td>一</td><td>0.2412</td><td>0.3971</td><td>一 0.3809</td></tr></table>

## D.1 Drug-level classification robustness

Panels a–c of Figure 4 use the frozen prediction calls and sampled queries of Table 1. Within each recorded drug identifier, we pool queries across the three cell contexts and compute Macro-F1 with fixed class sets: two classes for DE and conditional DIR, and three for joint unchanged/up/down. Undefined class F1 is set to zero. DIR uses only reference-DE queries; four canonical drugs have none and are excluded from DIR analysis. Equivalent identifiers are scored separately and their scores averaged, giving equal weight to each of 112 canonical drugs (108 for DIR). Gains subtract a fixed comparator’s score from CellCanon’s. The comparator is the highest aggregate Macro-F1 baseline in Table 1: PRnet for DE and joint, and BioLORD for DIR. It is not reselected per drug.

Response strength is the reference Core2000 DE-gene count averaged across cell contexts and then equivalent identifiers. Four quartile strata retain tied strengths together at cutoffs of 1.3333, 5.8333, and 29.25 DE genes per context, giving 31/25/28/28 drugs (27 in the first DIR stratum). Across all eligible drugs and within each stratum, 10,000 resamples of canonical drugs with replacement give percentile 95% intervals for the median paired gain, conditional on the saved predictors. These describe variation across drugs rather than retraining uncertainty. Violin densities show the observed range, with individual scores, horizontal median bars, and vertical interquartile ranges; wins and losses require gains above $1 0 ^ { - 1 2 }$ or below $- 1 0 ^ { - 1 2 }$ , with intermediate differences treated as ties. Drug-level summaries weight drugs equally and therefore differ from the pooled-query main-table metric.

## D.2 Differential-gene and expression-delta evaluation

Delta correlations additionally cover all 2,000 genes and the top 20 and top 100 genes. Gene sets are selected for evaluation, with ties resolved by gene order. The top-k sets include the highest-ranked genes regardless of their significance threshold.

CellCanon, Squidiff, PerturbNet, and STATE mean expression is calculated from generated cells. BioLORD, chemCPA, CMonge, and PRnet means are recovered from predicted logFC vectors and the common DMSO reference using the inverse target transformation. We apply the Top50 $\mathtt { R } ^ { 2 }$ and gene-set-specific delta-correlation definitions above, using the same pooled-dose populations and reference labels. Table 2 reports the Top50 expression comparison.

Top50 expression $\mathtt { R } ^ { 2 }$ summarizes reconstruction on response-ranked genes: CellCanon reaches 0.9145, compared with 0.9197 for BioLORD and 0.8880 for the DMSO reference. Across all 2,000 genes, CellCanon’s delta Pearson/Spearman are 0.3254/0.2412. On the top-100 panel, it reaches 0.3971/0.3809, compared with 0.3888/0.3632 for chemCPA (Table 9).

Table 10: Predict input-component comparison. The common 54/19/113 drug-identifier split, 2,000-gene evaluation panel, and metric definitions match Section E. Text removal retains LLMselected structured fields. Context removal retains the response atlas. Ordinary training statistics and cell identity remain available.
<table><tr><td>Variant</td><td>RMSE↓</td><td>DE AP</td><td>DE F1</td><td>DIR mAP</td><td>DIR F1</td><td>Joint F1</td></tr><tr><td>Condition heads</td><td>4.4762</td><td>0.8588</td><td>0.8077</td><td>0.9135</td><td>0.8393</td><td>0.7242</td></tr><tr><td>Without biological knowledge</td><td>4.4699</td><td>0.7922</td><td>0.7117</td><td>0.8593</td><td>0.7900</td><td>0.5776</td></tr><tr><td>Without text embeddings</td><td>4.4688</td><td>0.8640</td><td>0.8066</td><td>0.9102</td><td>0.8370</td><td>0.7228</td></tr><tr><td>Without measured context</td><td>4.4781</td><td>0.8611</td><td>0.8118</td><td>0.9093</td><td>0.8360</td><td>0.7281</td></tr></table>

## E Predict component ablations

## E.1 Interventions and common training protocol

These ablations score the shared condition predictor before probability-guided generation. All variants use the 54-support/19-validation/113-test identifier split and the same 2,000-gene evaluation. Input interventions retrain the affected frontend and independent heads, select settings on validation drugs, and average three seeds. Readout-only interventions keep continuous predictions fixed and fit their decision rules on validation drugs. Validation selects hidden width, weight decay, and DE/DIR regularization using the same search protocol across variants. Knowledge projection and normalization use the 162 support conditions. The DE count decoder is refitted on the 57 validation conditions.

Biological annotations. The knowledge-removed variant sets all 96-dimensional query/key knowledge inputs to zero throughout training and removes target and mechanism-of-action priors. Removed information includes text embeddings, target/pathway/action annotations, LLM judgments, and target-conditioned DMSO features. Chemical similarity, support-response values, ordinary training statistics, and cell-specific modules remain. This intervention measures the contribution of the combined biological-annotation channel.

Text embeddings. The text-removed variant retains LLM target selection, relevance, scope, and structured fields, but removes evidence-text and target-description embeddings. Its structured vectorization and 96-dimensional projection are refitted on training conditions.

Measured cell context. Measured-context removal deletes target control-expression and detection fields before compiling evidence for every condition, excludes those fields and their missingness indicators from text and structured features, and removes detection-based target weights. The response atlas is retained. Query features and atlas knowledge keys use the context-free encoding; training response values remain available. Drug/gene annotations, cell identity, and ordinary training means and scales remain. Context-free compilation introduces generation variation alongside context removal.

Biological-annotation deletion produces the largest classification losses among these input interventions. Table 11 reports their ranking and hard-decision metrics.

Table 11: Classification operating points and ranking metrics for the ablations. DE uses 55,598 queries with 8,651 positives. DIR uses those 8,651 true-DE queries with 4,687 up-regulated positives. Its AP, PR area, precision, and recall treat up as positive; this AP is distinct from the twodirection mAP in Section E. Hard DE decisions use the validation-fitted count decoder.
<table><tr><td>Variant</td><td>Task</td><td>AP</td><td>PR area</td><td>AUROC</td><td>F1</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td rowspan="2">Condition heads</td><td>DE</td><td>0.8588</td><td>0.8588</td><td>0.9676</td><td>0.8077</td><td>0.9163</td><td>0.8876</td><td>0.5294</td></tr><tr><td>DIR</td><td>0.9292</td><td>0.9291</td><td>0.9156</td><td>0.8393</td><td>0.8398</td><td>0.8737</td><td>0.8233</td></tr><tr><td rowspan="2">Without biological knowledge</td><td>DE</td><td>0.7922</td><td>0.7921</td><td>0.9546</td><td>0.7117</td><td>0.8875</td><td>0.8396</td><td>0.3425</td></tr><tr><td>DIR</td><td>0.8905</td><td>0.8904</td><td>0.8695</td><td>0.7900</td><td>0.7900</td><td>0.8616</td><td>0.7295</td></tr><tr><td rowspan="2">Without text embeddings</td><td>DE</td><td>0.8640</td><td>0.8640</td><td>0.9676</td><td>0.8066</td><td>0.9162</td><td>0.8920</td><td>0.5251</td></tr><tr><td>DIR</td><td>0.9278</td><td>0.9278</td><td>0.9131</td><td>0.8370</td><td>0.8375</td><td>0.8714</td><td>0.8212</td></tr><tr><td rowspan="2">Without measured context</td><td>DE</td><td>0.8611</td><td>0.8610</td><td>0.9680</td><td>0.8118</td><td>0.9176</td><td>0.8861</td><td>0.5396</td></tr><tr><td>DIR</td><td>0.9263</td><td>0.9262</td><td>0.9124</td><td>0.8360</td><td>0.8364</td><td>0.8696</td><td>0.8212</td></tr></table>

## E.2 Measured cell context and response changes

With the response atlas retained, removing measured background reduces delta Pearson and Spearman on all 2,000 genes and the top-20/50/100 reference panels (Table 12). All-gene Pearson decreases by 0.0265 and Spearman by 0.0091; the top-100 decreases are 0.0241 and 0.0293. All eight paired difference intervals exclude zero. Direction mAP also decreases from 0.9135 to 0.9093. The classification operating points in Table 11 show that the benefit is endpoint-specific: DE F1 is 0.8118 without measured context versus 0.8077 for the complete model.

## E.3 Independent classification readouts

We keep the complete model’s continuous predictions fixed and replace the DE/DIR heads with bounded scores derived from the predicted log fold change $y \colon s ^ { D \dot { E } } = | y | / ( 1 + | y | )$ and $s ^ { \mathrm { u p } } =$ $1 / 2 + y / [ 2 ( 1 + | y | ) ]$ ]. These monotone transforms encode response magnitude and sign as ranking scores. The comparison covers the 2,000 genes with continuous outputs; the training atlas still contains its original DE/DIR observations.

The first readout variant refits the DE count decoder on validation predictions. A second variant replaces that decoder with fixed DE thresholds selected separately for each cell line to maximize sampled-validation Macro-F1. Both use validation-selected direction thresholds on true-DE queries, retaining score ties. All rows of Table 13 share identical continuous responses.

Independent heads achieve DE AUPRC 0.8588 versus 0.1429 for magnitude-based scores. Their DE Macro-F1 is 0.8077, compared with 0.4538 using the refitted count decoder and 0.5278 using fixed thresholds. Joint three-state Macro-F1 is 0.7242 versus 0.3029 and 0.4361, respectively. Direction remains more recoverable from the signed response, with Macro-F1 0.8134 versus 0.8393 for the independent direction head. These comparisons support task-specific classification readouts over direct magnitude/sign decisions from the same mean-response predictor.

Table 12: Measured cell context improves perturbation-response correlations. The contextremoved model retains the response atlas and cell-specific modules, while excluding measured control-expression and detection information from evidence compilation and encoding. Correlations use the same 339 conditions and reference gene panels as Section D.2. Differences are context-removed minus complete model; brackets give exploratory 95% intervals from 1,000 paired canonical-drug bootstrap draws.
<table><tr><td>Gene panel</td><td>Metric</td><td>Condition heads</td><td>No context</td><td>Difference [95% interval]</td></tr><tr><td>All 2,000</td><td>∆-Pearson</td><td>0.3074</td><td>0.2810</td><td>-0.0265 [-0.0410, -0.0130]</td></tr><tr><td>All 2,000</td><td>∆-Spearman</td><td>0.2383</td><td>0.2292</td><td>-0.0091 [-0.0144, -0.0039]</td></tr><tr><td>Top20</td><td>∆-Pearson</td><td>0.4264</td><td>0.4030</td><td>-0.0235 [-0.0423, -0.0062]</td></tr><tr><td>Top20</td><td>∆-Spearman</td><td>0.3894</td><td>0.3628</td><td>-0.0265 [-0.0471, -0.0081]</td></tr><tr><td>Top50</td><td>∆-Pearson</td><td>0.3961</td><td>0.3733</td><td>-0.0228 [-0.0370, -0.0093]</td></tr><tr><td>Top50</td><td>∆-Spearman</td><td>0.3700</td><td>0.3420</td><td>-0.0280 [-0.0433, -0.0133]</td></tr><tr><td>Top100</td><td>∆-Pearson</td><td>0.3694</td><td>0.3453</td><td>-0.0241 [-0.0374, -0.0113]</td></tr><tr><td>Top100</td><td>∆-Spearman</td><td>0.3510</td><td>0.3216</td><td>-0.0293 [-0.0440, -0.0150]</td></tr></table>

Table 13: Independent task readouts improve classification from the same continuous predictor. All rows share frozen 2,000-gene continuous outputs (RMSE 4.4762). DE AUPRC and DE/joint F1 use 55,598 sampled queries. DIR F1 uses 8,651 true-DE queries, and DIR mAP averages up/down AP on true-DE entries. All F1 scores are macro-averages. The two continuous-score rows use validation-refitted count decoding or validation-selected fixed DE thresholds, respectively.
<table><tr><td>Readout and DE decision rule</td><td>DE AUPRC</td><td>DE F1</td><td>DIR mAP</td><td>DIR F1</td><td>Joint F1</td></tr><tr><td>Independent heads + count decoder</td><td>0.8588</td><td>0.8077</td><td>0.9135</td><td>0.8393</td><td>0.7242</td></tr><tr><td>Continuous scores + count decoder</td><td>0.1429</td><td>0.4538</td><td>0.8901</td><td>0.8134</td><td>0.3029</td></tr><tr><td>Continuous scores + fixed thresholds</td><td>0.1429</td><td>0.5278</td><td>0.8901</td><td>0.8134</td><td>0.4361</td></tr></table>

## E.4 Paired uncertainty

We use 1,000 paired bootstrap draws over 112 canonical drug identities, keeping aliases and all associated cells and genes in the same block. Table 14 reports variant-minus-reference differences. The five principal classification intervals for annotation removal exclude zero. Context removal with the atlas retained reduces direction mAP, and replacing the independent readouts reduces DE AP, DE F1, and DIR F1; their reported intervals also exclude zero. Delta-context intervals appear alongside all four reference panels in Table 12. Text embeddings give a smaller direction-ranking effect. These exploratory intervals are conditional on the selected ensembles and use no multiplecomparison adjustment.

Table 14: Paired differences for the reported component effects. Differences are variant minus complete model. Percentile 95% intervals use 1,000 paired resamples of 112 canonical drug blocks, retaining all associated cells and genes. Intervals condition on the selected ensembles and are unadjusted for multiple comparisons; they describe sampling variation rather than retraining variation.
<table><tr><td>Comparison</td><td>Metric</td><td>Difference</td><td>Lower</td><td>Upper</td></tr><tr><td>Without biological knowledge</td><td>DE AP</td><td>-0.066687</td><td>-0.102294</td><td>-0.011355</td></tr><tr><td>Without biological knowledge</td><td>DE F1</td><td>-0.096053</td><td>-0.152370</td><td>-0.019405</td></tr><tr><td>Without biological knowledge</td><td>DIR mAP</td><td>-0.054199</td><td>-0.067440</td><td>-0.036245</td></tr><tr><td>Without biological knowledge</td><td>DIR F1</td><td>-0.049380</td><td>-0.068552</td><td>-0.030069</td></tr><tr><td>Without biological knowledge</td><td>Joint F1</td><td>-0.146592</td><td>-0.214128</td><td>-0.053563</td></tr><tr><td>Without text embeddings</td><td>DIR mAP</td><td>-0.003370</td><td>-0.006790</td><td>-0.000359</td></tr><tr><td>Without measured context</td><td>DIR mAP</td><td>-0.004185</td><td>-0.007109</td><td>-0.001334</td></tr><tr><td>Continuous scores + count</td><td>DE AP</td><td>-0.715933</td><td>-0.746663</td><td>-0.656455</td></tr><tr><td>Continuous scores + count</td><td>DE F1</td><td>-0.353891</td><td>-0.401736</td><td>-0.287063</td></tr><tr><td>Continuous scores + count</td><td>DIR F1</td><td>-0.025929</td><td>-0.042735</td><td>-0.012632</td></tr><tr><td>Continuous scores + threshold</td><td>DE F1</td><td>-0.279905</td><td>-0.317796</td><td>-0.243067</td></tr></table>

## E.5 Evidence-view sensitivity

The raw-source knowledge view gives DE AP 0.8661 and direction mAP 0.9065; LLM compilation gives 0.8588 and 0.9135, respectively. Compilation therefore improves direction ranking by 0.70 percentage points while trading off DE ranking and a small amount of continuous error. The two knowledge views differ in source volume.

Table 15: Raw-source and LLM-compiled knowledge views. Both use Qwen text encoding and the same numerical architecture. Raw-source encoding omits LLM selection and changes information volume.
<table><tr><td>Variant</td><td>RMSE↓</td><td>DE AP</td><td>DE F1</td><td>DIR mAP</td><td>DIR F1</td><td>Joint F1</td></tr><tr><td>Raw-source knowledge</td><td>4.4723</td><td>0.8661</td><td>0.8136</td><td>0.9065</td><td>0.8319</td><td>0.7339</td></tr><tr><td>LLM-compiled knowledge</td><td>4.4762</td><td>0.8588</td><td>0.8077</td><td>0.9135</td><td>0.8393</td><td>0.7242</td></tr></table>

## F Source-grounded explanation benchmark

## F.1 Reference construction and scoring

The evaluation cohort contains 24 drugs, 72 conditions, and 966 questions; the validation cohort contains six drugs and 225 questions. Each question is paired with its condition’s evidence, a supported/contradicted/insufficient label, and acceptable proof bundles constructed from source rules. References define 561 answerable and 405 insufficient-evidence questions. For an answerable question, joint correctness requires the correct label, at least one complete acceptable proof, and citations drawn from the supplied evidence. Other responses receive zero. Success on an insufficient-evidence question requires a valid output with an explicit insufficient-evidence judgment.

The answerable questions comprise 240 control-detection questions, 138 target-action scope questions, 126 background-path questions, and 57 source-statement questions. Across the 966 questions, there are 487 distinct question strings and 556 question-type/argument combinations; 126 questions share a string with the validation set. Question templates are reused with condition-specific evidence. The source rules also provide a deterministic comparator (Table 16). The benchmark evaluates structured judgments and cited proofs, and the agent supplies accompanying natural-language rationales.

The explanation procedure uses the DeepSeek Flash API (deepseek-flash) for source analysis, question answering, and localized review. Review feedback comprises source-scope, path, and formatting diagnostics. Error analysis on the evaluation panel informed refinement of the correction procedure. Reported scores evaluate the final pipeline; reference answers were not supplied to runtime correction.

## F.2 Paths and uncertain signs

A route is eligible when its endpoints, source records, and sign relationships satisfy the path criteria. Known edge signs are composed directly. Unknown signs are assigned jointly across paths so that a shared edge has one value within each completion. These alternative assignments distinguish conclusions that hold for every permitted sign from those that depend on an unresolved interaction.

The language model selects and ranks candidate paths. If eligibility review leads to a correction, the included paths are ordered by length and source identifier; otherwise the model’s ordering is retained. Among the 72 evaluation conditions, 63 contain eligible reference paths. On these 63 conditions, valid path@3, precision@3, and recall@3 are each 1.000. Proposition scoring includes all 72 conditions, including the nine with no eligible reference path.

## F.3 Observation-state reasoning and withdrawal

The observation-state task comprises 20 measurement–target records spanning 12 drugs and 13 unique conditions. Each record is presented before measurement, after measurement, and after withdrawal, producing 480 questions: 120 answerable and 360 insufficient. Gene identities and numerical predicates are matched to their source records, and the language model supplies a judgment, citation, and explanation. Withdrawal removes a record from the active evidence while retaining its history. The agent correctly answers all 120 answerable questions with complete proof and makes the correct insufficient-evidence judgment on all 360 insufficient questions.

An additional source-withdrawal challenge makes 16 previously answerable claims insufficient. The agent revises 14 claims correctly (87.5%). Both withdrawal tasks use source-defined reference judgments. These explanation experiments and the numerical transfer experiment in Section 3.4 are evaluated separately.

Table 16: Source-defined reference comparisons. Both label accuracy and label-plus-proof accuracy use all 561 answerable questions. Insufficiency uses all 405 questions. The symbolic reference applies the source rules directly.
<table><tr><td>Method</td><td>Label accuracy</td><td>Label + proof</td><td>Correct insufficiency</td></tr><tr><td>TF–IDF retrieval + rules</td><td>0.5989</td><td>0.5989</td><td>405 / 405</td></tr><tr><td>Complete-source symbolic reference</td><td>1.0000</td><td>1.0000</td><td>405 / 405</td></tr><tr><td>CellCanon</td><td>1.0000</td><td>1.0000</td><td>405 / 405</td></tr></table>

## F.4 TSA explanation and update case

Figure 5 presents Trichostatin A (TSA) in two experiments. The Explain experiment uses K562 at 24 hours with pooled doses. Its evidence includes functional HDAC1 inhibition, a binding record with an unspecified action sign, and an HDAC1 RNA detection fraction of 0.1530 in vehicle controls. Two background routes connect TSA to CYP24A1 through HDAC1: TSA–HDAC1–TP53– CYP24A1 has net sign +1, and TSA–HDAC1–AR–ESR1–CYP24A1 has net sign −1, including the inhibitory drug–target edge. These source-linked routes provide candidate mechanisms for the condition.

Localized review changes two reverse-sign propositions from contradicted to insufficient evidence because the binding record permits alternative sign assignments. The explanation retains the supporting citations and biological rationale.

The Update example measures TSA in A549 and predicts its response in unobserved K562. The source supplies a 2,000-gene continuous response and 2,931-gene DE/DIR labels. After 12 observations, the predicted K562 DE set changes from 153 to 146 of 2,931 genes. Aggregate prediction accuracy is evaluated in Section H.

## G Update fitting and observation protocol

## G.1 Training and cross-cell observations

Five-fold cross-fitting over the 54 support drugs produces parent residuals for updater training. Each outer fold refits the 96-dimensional knowledge projection, normalization, atlas representation, continuous MLP, and classification heads without that fold’s drug identities. Each MLP also uses internal whole-fold atlas masking. Cross-fitted predictions use three random seeds. The parent architecture and its 25-epoch schedule remain fixed. Nineteen validation drugs select updater settings before test evaluation.

Each of three test partitions assigns one hidden cell line to each test drug identity; the assignment can differ across drugs. Observations come from the other two cell lines, with target conditions and their canonical aliases excluded from acquisition. Each measured source condition supplies a 2,000- gene response vector and 2,931-gene DE/DIR labels. Budgets count distinct source conditions: 112 observations cover one source cell line for each canonical drug, and 224 cover both. Each source condition can be selected once, up to the observation budget.

The Update experiment scores the shared condition predictor’s continuous and classification heads before cell generation. The three target-cell partitions contain 19,311, 18,336, and 18,711 sampled DE queries, respectively, on the 2,000-gene panel. Budgets of 12, 24, and 48 follow fixed numerical acquisition schedules; budgets of 112 and 224 supply one or two source contexts per identity. At every budget, updated and initial predictions are paired on the same hidden targets. Acquisition schedules are held constant to measure the contribution of prediction updating under cross-cell observations.

## G.2 Continuous transfer equations

Let $\mathbf { r } _ { i }$ denote the standardized source-cell response of training drug i, with concatenation when two source cells are available. A product kernel combines similarity of measured responses and drug knowledge:

$$
K ( i , j ) = k _ { \mathrm { R N A } } ( { \bf r } _ { i } , { \bf r } _ { j } ) \left[ ( 1 - \eta ) + \eta k _ { \mathrm { d r u g } } ( i , j ) \right] .\tag{12}
$$

Table 17: Prediction after observation feedback. Each entry averages three partitions with 113 hidden drug–cell targets per partition. The 12/24/48 schedules select a subset of source conditions; 112/224 reveal one/two other cell contexts for each of the 112 canonical drugs. F1 denotes Macro-F1.
<table><tr><td>Observation budget</td><td>MSE↓</td><td>DE AP</td><td>DE AUROC</td><td>DE F1</td><td>DIR F1</td></tr><tr><td>No update</td><td>20.4360</td><td>0.8806</td><td>0.9710</td><td>0.8199</td><td>0.8402</td></tr><tr><td>12 observations</td><td>20.3889</td><td>0.8833</td><td>0.9715</td><td>0.8210</td><td>0.8419</td></tr><tr><td>24 observations</td><td>20.2662</td><td>0.8893</td><td>0.9723</td><td>0.8242</td><td>0.8410</td></tr><tr><td>48 observations</td><td>20.1772</td><td>0.8924</td><td>0.9726</td><td>0.8393</td><td>0.8425</td></tr><tr><td>112 observations</td><td>20.0427</td><td>0.9077</td><td>0.9792</td><td>0.8261</td><td>0.8453</td></tr><tr><td>224 observations</td><td>20.0447</td><td>0.9030</td><td>0.9777</td><td>0.8583</td><td>0.8445</td></tr></table>

Table 18: Update classification metrics: 2,000-gene evaluation panel. Sampled endpoint, averaged over three hidden-cell partitions. DIR AP, precision, and recall use up as the positive class. The zero row is the same model without Update.
<table><tr><td>Budget</td><td>Task</td><td>AP</td><td>AUROC</td><td>F1</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>0</td><td>DE</td><td>0.8806</td><td>0.9710</td><td>0.8199</td><td>0.9162</td><td>0.9013</td><td>0.5570</td></tr><tr><td>0</td><td>DIR</td><td>0.9323</td><td>0.9179</td><td>0.8402</td><td>0.8408</td><td>0.8845</td><td>0.8167</td></tr><tr><td>48</td><td>DE</td><td>0.8924</td><td>0.9726</td><td>0.8393</td><td>0.9230</td><td>0.8975</td><td>0.6064</td></tr><tr><td>48</td><td>DIR</td><td>0.9305</td><td>0.9157</td><td>0.8425</td><td>0.8432</td><td>0.8838</td><td>0.8224</td></tr><tr><td>112</td><td>DE</td><td>0.9077</td><td>0.9792</td><td>0.8261</td><td>0.9185</td><td>0.9021</td><td>0.5715</td></tr><tr><td>112</td><td>DIR</td><td>0.9313</td><td>0.9167</td><td>0.8453</td><td>0.8460</td><td>0.8840</td><td>0.8283</td></tr><tr><td>224</td><td>DE</td><td>0.9030</td><td>0.9777</td><td>0.8583</td><td>0.9305</td><td>0.8985</td><td>0.6553</td></tr><tr><td>224</td><td>DIR</td><td>0.9315</td><td>0.9168</td><td>0.8445</td><td>0.8451</td><td>0.8853</td><td>0.8248</td></tr></table>

Here $k _ { \mathrm { R N A } }$ is linear and $k _ { \mathrm { d r u g } }$ combines chemical and biological similarities. Kernel ridge regression learns the mapping to the target-cell response matrix $Y _ { c }$ . With centered kernel $K _ { 0 } ,$ , centered query kernel $\mathbf { k } _ { d , 0 } ,$ , and target mean $\bar { \mathbf { y } } _ { c } ,$ the transferred response is

$$
\begin{array} { r } { \widehat { \mathbf { y } } _ { d c } ^ { \mathrm { t r a n s f e r } } = \bar { \mathbf { y } } _ { c } + \mathbf { k } _ { d , 0 } ^ { \top } ( K _ { 0 } + \lambda I ) ^ { - 1 } \big ( Y _ { c } - \mathbf { 1 } \bar { \mathbf { y } } _ { c } ^ { \top } \big ) . } \end{array}\tag{13}
$$

The updater uses the transferred response when a source measurement is available and retains the parent prediction otherwise.

## G.3 Selected numerical update settings

Continuous transfer in Equation (12) uses $\eta = 0 . 5$ , ridge coefficient $\lambda = 1$ , and transfer strength 1. The transfer map is fitted on source and target responses from training drugs. At inference, the measured response supplies the query vector for updating the same drug in its hidden cell line.

For DE, the conditional-label likelihood updater uses smoothing 200 and strength 0.5. It is mixed equally with a cross-fitted residual update of strength 0.25. A response-feature tree with at most seven leaves contributes an additional correction of strength 0.1. For DIR, the likelihood contribution has selected strength zero; the residual contribution has strength 1 and mixture weight 0.25, with no additional response-feature tree correction. These settings are fixed across the reported budgets.

The response-count model standardizes support features and fits a ridge regressor with coefficient 10. Its source-activity count estimate contributes weight 0.25, and 1.25 times the parent’s selected DE count contributes weight 0.75. The updated probabilities determine the ranked gene set. Consequently, Update includes both response-conditioned probability changes and readout recalibration. The no-observation path returns the parent’s original outputs, including its hard gene selections.

The parent MLP, response-transfer map, and classification corrections remain fixed during test-time updating. Measured source responses enter as inference inputs, and the updated values are predictions for the hidden target cell line.

## H Additional Update results

Table 17 summarizes continuous and classification performance; Table 18 reports sampled DE and conditional direction on the 2,000-gene evaluation panel. Metrics are computed within each of the three partitions and then averaged. At 224 observations, DE recall increases by 9.84 percentage points with precision 0.8985. DIR Macro-F1 improves, while up-positive AP and AUROC decrease slightly. DE AP is highest at 112 observations and DE Macro-F1 at 224, showing different effects on ranking and hard decisions.

The mean partition RMSE is 4.5205 without Update, 4.4918 at 48 observations, and 4.4770 at 224.

## H.1 Adaptation to newly observed responses

This task asks whether observing a drug response in one cell context improves prediction in an unobserved context. Each partition assigns a hidden cell context to each drug, and measurements come from its other contexts. A measurement reveals the source population at 24 hours, including 10, 100, 1,000, and 10,000 nM doses. We evaluate no observations, 48 observed conditions, and one source context per drug. Source selections are fixed before adaptation; target responses are used for evaluation. Scores average the three partitions, pairing each method’s adapted and unadapted predictions on the same hidden targets.

Parameter adaptation. Each comparator starts from its selected trained weights and a new optimizer independently at each observation budget. Gradient updates use exclusively the newly observed treated populations, with matched DMSO controls. We select the learning rate from $\{ 1 0 ^ { - 5 } , 3 \times 1 0 ^ { - 5 } \}$ and the number of updates from {100, 400} using 19 validation drugs. Selection minimizes mean hidden-condition expression MSE over three validation partitions and observation budgets of 8 and 19. The selected settings are fixed for test adaptation (Table 19).

STATE updates its transition network with the energy distribution loss; PRnet updates its prediction network with Gaussian negative log-likelihood. chemCPA retains alternating reconstruction, dose-response, and adversarial optimization. PerturbNet updates its conditional flow with negative log-likelihood while holding scVI and chemical representations fixed. CMonge updates the conditional transport map with transport fitting and the Monge-gap penalty while keeping its autoencoder fixed. BioLORD updates the shared decoder and attribute networks together with nuisance codes for newly observed cells; existing cell codes remain fixed. STATE uses eight sets of 128 cells per update. CMonge uses sets of 128 cells from a single condition; the other models use mixed-condition batches of 128 cells. Optimization uses seed ${ \bar { 4 2 } } ;$ generation uses condition- and dose-specific random streams derived from seed 42.

Table 19: Validation-selected fine-tuning settings. Each setting minimizes hidden-condition expression MSE across the validation observation budgets and partitions.
<table><tr><td>Method</td><td>Learning rate</td><td>Updates</td></tr><tr><td>STATE</td><td>10−5</td><td>400</td></tr><tr><td>PRnet</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>400</td></tr><tr><td>chemCPA</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>400</td></tr><tr><td>PerturbNet</td><td> $1 0 ^ { - 5 }$ </td><td>400</td></tr><tr><td>CMonge</td><td> $3 \times 1 0 ^ { - 5 }$ </td><td>400</td></tr><tr><td>BioLORD</td><td> $1 0 ^ { - 5 }$ </td><td>100</td></tr></table>

Prediction and readout. Fine-tuned comparators generate the prescribed number of cells at each target dose in the 2,000-gene panel. Adapted and unadapted predictions use matched dose-specific cell counts and random draws. Responses are pooled across doses and tested against matched DMSO by Wilcoxon rank-sum, with Benjamini–Hochberg correction over the 2,000 predicted genes. DE calls require $q < 0 . 0 5$ and absolute log<sub>2</sub> fold change above 0.25; ranking uses $- \log _ { 1 0 } p .$ Direction uses signed generated logFC on reference-DE queries. PRnet Gaussian samples are clipped at zero per cell before pooling and testing. These population tests provide the comparator scores in this experiment, including their zero-observation reference.

CellCanon retains its condition-head readouts throughout observation feedback, with a 2,000-gene continuous source response and 2,931-gene source DE/DIR labels. Fine-tuning uses the observed single-cell expression in the 2,000-gene panel. All reported classification scores use the sampled 2,000-gene endpoint. Joint scores combine DE and direction into unchanged/up/down as in Section D; each metric is computed within a partition and then averaged across the three partitions.

Changes are measured within each method’s prediction pipeline (Table 20). Table 21 reports classification before and after observation; Table 22 reports generated-response metrics for the six finetuned models.

Table 20: Adaptation with one source condition per drug. Entries are absolute score changes after adaptation minus each method’s own unadapted baseline, averaged over three partitions on the 2,000-gene panel. F1 denotes Macro-F1; Acc. denotes accuracy. Protocol and absolute scores appear in Section H.1.
<table><tr><td>Method ∆ DE AUPRC ∆ DE F1</td></tr><tr><td>∆ DIR F1 ∆ DE Acc. Fine-tuning: generated-expression readouts</td></tr><tr><td>STATE +0.0872 +0.0808 +0.0716 +0.0297</td></tr><tr><td>PRnet +0.0001 +0.0065 +0.0552 +0.0076</td></tr><tr><td>chemCPA -0.0014 +0.0865 -0.0785 -0.0346</td></tr><tr><td>PerturbNet +0.1023 +0.0839 +0.0557 +0.0229</td></tr><tr><td>CMonge -0.0031 -0.0064 -0.0042 -0.0056</td></tr><tr><td>BioLORD -0.0011 -0.0912 -0.1143 -0.1306</td></tr><tr><td>Fixed updater: condition heads</td></tr></table>

Table 21: Classification after feedback adaptation. Scores are averaged over three source–target partitions on the 2,000-gene panel. None denotes the unadapted model; the other rows use 48 source conditions or one source condition per drug. DE uses sampled gene queries, and DIR uses reference-DE genes. F1 denotes Macro-F1 and Acc. denotes accuracy. Higher is better. Methods are ranked within each observation budget.
<table><tr><td>Method</td><td>Sources</td><td>DE AUPRC</td><td>DE F1</td><td>DE Acc.</td><td>DIR F1</td><td>DIR Acc.</td></tr><tr><td colspan="7">Fine-tuning: generated-expression readouts</td></tr><tr><td>STATE</td><td>None</td><td>0.4663 0.6195</td><td></td><td>0.8259</td><td>0.7606</td><td>0.7613</td></tr><tr><td></td><td>48</td><td>0.5314</td><td>0.6702</td><td>0.8434</td><td>0.7912</td><td>0.7930</td></tr><tr><td></td><td>One source/drug</td><td>0.5535</td><td>0.7003</td><td>0.8555</td><td>0.8323</td><td>0.8329</td></tr><tr><td>PRnet</td><td>None</td><td>0.0947</td><td>0.1176</td><td>0.1262</td><td>0.5936</td><td>0.6566</td></tr><tr><td></td><td>48</td><td>0.0945</td><td>0.1216</td><td>0.1310</td><td>0.5441</td><td>0.6282</td></tr><tr><td></td><td>One source/drug</td><td>0.0948</td><td>0.1241</td><td>0.1338</td><td>0.6488</td><td>0.6785</td></tr><tr><td>chemCPA</td><td>None</td><td>0.3202</td><td>0.4921</td><td>0.7545</td><td>0.6446</td><td>0.6449</td></tr><tr><td></td><td>48</td><td>0.3286</td><td>0.5335</td><td>0.7409</td><td>0.5870</td><td>0.5926</td></tr><tr><td></td><td>One source/drug</td><td>0.3188</td><td>0.5786</td><td>0.7199</td><td>0.5661</td><td>0.5892</td></tr><tr><td>PerturbNet None</td><td></td><td>0.5256</td><td>0.6791</td><td>0.8407</td><td>0.6331</td><td>0.6371</td></tr><tr><td></td><td>48</td><td>0.6463</td><td>0.7634</td><td>0.8744</td><td>0.6899</td><td>0.6930</td></tr><tr><td></td><td>One source/drug</td><td>0.6279</td><td>0.7630</td><td>0.8637</td><td>0.6889</td><td>0.6937</td></tr><tr><td>CMonge</td><td>None</td><td>0.1337</td><td>0.2030</td><td>0.2031</td><td>0.6649</td><td>0.6700</td></tr><tr><td></td><td>48</td><td>0.1322</td><td>0.1995</td><td>0.1999</td><td>0.6743</td><td>0.6809</td></tr><tr><td></td><td>One source/drug</td><td>0.1306</td><td>0.1965</td><td>0.1976</td><td>0.6608</td><td>0.6693</td></tr><tr><td>BioLORD</td><td>None</td><td>0.1348</td><td>0.3539</td><td>0.3986</td><td>0.7834</td><td>0.7860</td></tr><tr><td></td><td>48</td><td>0.1285</td><td>0.2920</td><td>0.3142</td><td>0.7365</td><td>0.7415</td></tr><tr><td></td><td>One source/drug</td><td>0.1337</td><td>0.2627</td><td>0.2680</td><td>0.6691</td><td>0.6709</td></tr><tr><td colspan="7">Fixed updater: condition heads</td></tr><tr><td>CellCanon None</td><td></td><td>0.8806 0.8199</td><td></td><td>0.9162</td><td>0.8402</td><td>0.8408</td></tr><tr><td></td><td>48</td><td>0.8924 0.8393</td><td></td><td>0.9230</td><td>0.8425</td><td>0.8432</td></tr><tr><td></td><td>One source/drug</td><td>0.9077</td><td>0.8261</td><td>0.9185</td><td>0.8453</td><td>0.8460</td></tr></table>

With one source context per drug, STATE improves DE ranking, DE and DIR decisions, and response-change Pearson correlation. PerturbNet improves classification while response-change Pearson decreases from 0.1531 to 0.1332. CMonge and BioLORD decline in the four classification metrics summarized in Table 20. The 48-observation results further show that increasing the amount of feedback need not improve every endpoint.

Table 22: Continuous prediction after fine-tuning. Scores are averaged over three source–target partitions. Top50 $R ^ { 2 }$ measures mean expression; logFC Pearson and ∆-Pearson use all 2,000 genes. $\Delta$ denotes expression changes from matched DMSO, and Top100 ∆-Spearman uses the reference Top100 gene panel. Source budgets follow Table 21. Higher is better. Methods are ranked within each observation budget.
<table><tr><td colspan="2"></td><td>Top50</td><td>logFC</td><td>All 2,000 ∆-Pearson</td><td>Top100</td></tr><tr><td>Method</td><td>Sources</td><td> $R ^ { 2 }$ </td><td>Pearson</td><td></td><td>∆-Spearman</td></tr><tr><td rowspan="3">STATE</td><td>None</td><td>0.8998</td><td>0.3045</td><td>0.2417</td><td>0.2800 0.3115</td></tr><tr><td>48</td><td>0.9146</td><td>0.3144</td><td>0.2758 0.3017</td><td></td></tr><tr><td>One source/drug</td><td>0.9230</td><td>0.3214 0.2386</td><td></td><td>0.3384 0.2149</td></tr><tr><td rowspan="3">PRnet</td><td>None 48</td><td>0.8475 0.8535</td><td>0.2394</td><td>0.0977 0.0941</td><td>0.2045</td></tr><tr><td>One source/drug</td><td>0.1368</td><td>0.2411</td><td>0.0724</td><td>0.1183</td></tr><tr><td></td><td>0.8948</td><td>0.4315</td><td>0.2436</td><td>0.3547</td></tr><tr><td rowspan="3">chemCPA</td><td>None 48</td><td>0.8668</td><td>0.4330</td><td>0.1675</td><td>0.2568</td></tr><tr><td>One source/drug</td><td>-1.6496</td><td>0.3859</td><td>0.0843</td><td>0.1605</td></tr><tr><td></td><td>0.4234</td><td>0.4255</td><td>0.1531</td><td>0.2162</td></tr><tr><td rowspan="3">PerturbNet None</td><td>48</td><td>0.4883</td><td>0.4377</td><td>0.1503</td><td>0.2106</td></tr><tr><td>One source/drug</td><td>0.2069</td><td>0.4185</td><td>0.1332</td><td>0.1948</td></tr><tr><td></td><td>0.7119</td><td></td><td></td><td></td></tr><tr><td rowspan="3">CMonge</td><td>None</td><td></td><td>0.2406</td><td>0.1383</td><td>0.1685</td></tr><tr><td>48</td><td>0.7828</td><td>0.2407</td><td>0.1351</td><td>0.1660</td></tr><tr><td>One source/drug</td><td>0.8401</td><td>0.2408</td><td>0.1331</td><td>0.1646</td></tr><tr><td rowspan="3">BioLORD</td><td>None</td><td>0.9145</td><td>0.2946</td><td>0.2428</td><td>0.3106</td></tr><tr><td>48</td><td>0.8952</td><td>0.2999</td><td>0.1858</td><td>0.2558</td></tr><tr><td>One source/drug</td><td>0.6793</td><td>0.2631</td><td>0.0580</td><td>0.0802</td></tr></table>

## I Language-model token use

Table 23: Token use on matched condition cohorts. VCWorld’s gene-wise prediction and chain-ofthought requests supply a shared reference; CellCanon Predict and Explain are compared separately. Counts sum recorded input and output tokens from API calls, including corrections. Savings are relative to the corresponding VCWorld cohort.
<table><tr><td>Method / stage</td><td>Input</td><td>Output</td><td>Total</td><td>Saved</td></tr><tr><td colspan="5">Predict: 339 conditions; 55,598 reference gene queries</td></tr><tr><td>VCWorld</td><td></td><td>310,614,134 109,667,313</td><td>420,281,447</td><td></td></tr><tr><td>CellCanon Predict</td><td>882,462</td><td>99,131</td><td>981,593</td><td>99.77%</td></tr><tr><td colspan="5">Explain: 72 conditions; 12,399 reference gene queries</td></tr><tr><td>VCWorld</td><td>69,060,037</td><td>24,334,813</td><td>93,394,850</td><td></td></tr><tr><td>CellCanon Explain</td><td>1,347,957</td><td>141,533</td><td>1,489,490</td><td>98.41%</td></tr></table>

We match conditions by drug identifier and cell line. The Predict comparison covers all 339 test conditions and 55,598 sampled gene queries from the 2,000-gene evaluation panel. The Explain comparison uses the 72 evaluated conditions and their 12,399 corresponding reference gene queries. VCWorld’s DE and direction requests include their chain-of-thought rationales; their combined token use supplies the reference for each separate CellCanon stage. Token savings are $1 0 0 ( 1 - T _ { \mathrm { s t a g e } } / T _ { \mathrm { r e f e r e n c e } } )$ percent.

Token totals include input and output tokens from each API call once, including successful correction calls. Calls without reported usage are excluded. Tables 23 and 24 reports the matched-cohort comparison and stage totals.

Explain’s 1,489,490 tokens cover source analysis, localized review, and path qualification for the 72- condition cohort. Observation-state experiments are evaluated separately. Numerical Update reuses compiled evidence and requires no additional language-model calls.

Evidence compilation is shared across genes. A fixed condition therefore incurs the same languagemodel token usage for continuous and classification predictions across the 2,000-gene evaluation

Table 24: Language-model usage. Token counts include successful correction requests. Input and output are totals; the last column gives total tokens per condition.
<table><tr><td>Stage</td><td>Conditions</td><td>Input</td><td>Output</td><td>Total</td><td>Per condition</td></tr><tr><td>Predict: test</td><td>339</td><td>882,462</td><td>99,131</td><td>981,593</td><td>2,896</td></tr><tr><td>Predict: all splits</td><td>558</td><td>1,403,786</td><td>155,684</td><td>1,559,470</td><td>2,795</td></tr><tr><td>Explain: evaluated cohort</td><td>72</td><td>1,347,957</td><td>141,533</td><td>1,489,490</td><td>20,687</td></tr></table>

panel. Text encoding, atlas retrieval, training, and numerical inference contribute additional computational cost.

## J Prediction beyond the continuous-head gene panel

The additional panel contains the 931 measured LINCS L1000 landmark genes [Subramanian et al., 2017] outside the 2,000 highly variable genes: all 978 landmarks map to measured SciPlex3 genes, and 47 overlap the primary panel. Selection uses gene identities and landmark membership. This panel tests response prediction for biologically defined genes beyond the continuous head’s varianceselected output set.

With support DE/DIR label supervision for these genes, the independent classification heads achieve AUPRC of 0.7674/0.9212 and Macro-F1 of 0.7908/0.8535 on 24,630 DE queries and 3,777 true-DE direction queries. Their continuous response targets are absent from the main regression loss. For expression generation, an auxiliary dose-response model is fitted to their continuous support responses (Section B.4); the generated Extra931 responses have logFC Pearson 0.5344 and ∆-Pearson of 0.2583 over 339 conditions. This separates classification with label supervision from continuous generation with response supervision. The combined 2,931-gene panel contains 80,228 sampled queries and 12,428 DE positives.

## J.1 Prediction and observation feedback

Table 25: Classification on the additional and combined gene panels. DIR uses the true DE subset; its AUPRC, precision, and recall treat up-regulation as the positive class. F1 is two-class Macro-F1. Additional: 24,630/3,777; combined: 80,228/12,428.
<table><tr><td>Panel</td><td>Task</td><td>AUPRC</td><td>AUROC</td><td>F1</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>Additional 931</td><td>DE</td><td>0.6842</td><td>0.8777</td><td>0.7466</td><td>0.8957</td><td>0.8082</td><td>0.4194</td></tr><tr><td>Additional 931</td><td>DIR</td><td>0.8879</td><td>0.9051</td><td>0.8030</td><td>0.8144</td><td>0.8714</td><td>0.6671</td></tr><tr><td>All 2,931</td><td>DE</td><td>0.7710</td><td>0.9280</td><td>0.7965</td><td>0.9100</td><td>0.8298</td><td>0.5270</td></tr><tr><td>All 2,931</td><td>DIR</td><td>0.9254</td><td>0.9186</td><td>0.8377</td><td>0.8378</td><td>0.8692</td><td>0.8012</td></tr></table>

Table 26: Update classification metrics: 931 additional genes. Sampled endpoint, averaged over three hidden-cell partitions. DIR AP, precision, and recall use up as the positive class. The zero row is the same model without Update.
<table><tr><td>Budget</td><td>Task</td><td>AP</td><td>AUROC</td><td>F1</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>0</td><td>DE</td><td>0.8179</td><td>0.9357</td><td>0.8136</td><td>0.9051</td><td>0.8581</td><td>0.5683</td></tr><tr><td>0</td><td>DIR</td><td>0.9157</td><td>0.9488</td><td>0.8599</td><td>0.8676</td><td>0.8194</td><td>0.8376</td></tr><tr><td>48</td><td>DE</td><td>0.8420</td><td>0.9392</td><td>0.8430</td><td>0.9166</td><td>0.8641</td><td>0.6421</td></tr><tr><td>48</td><td>DIR</td><td>0.9163</td><td>0.9490</td><td>0.8651</td><td>0.8724</td><td>0.8235</td><td>0.8472</td></tr><tr><td>112</td><td>DE</td><td>0.8618</td><td>0.9553</td><td>0.8182</td><td>0.9074</td><td>0.8685</td><td>0.5748</td></tr><tr><td>112</td><td>DIR</td><td>0.9160</td><td>0.9490</td><td>0.8644</td><td>0.8714</td><td>0.8210</td><td>0.8488</td></tr><tr><td>224</td><td>DE</td><td>0.8578</td><td>0.9527</td><td>0.8587</td><td>0.9240</td><td>0.8664</td><td>0.6823</td></tr><tr><td>224</td><td>DIR</td><td>0.9188</td><td>0.9504</td><td>0.8675</td><td>0.8745</td><td>0.8261</td><td>0.8506</td></tr></table>

Table 27: Update classification metrics: all 2,931 genes. Sampled endpoint, averaged over three hidden-cell partitions. DIR AP, precision, and recall use up as the positive class. The zero row is the same model without Update.
<table><tr><td>Budget</td><td>Task</td><td>AP</td><td>AUROC</td><td>F1</td><td>Accuracy</td><td>Precision</td><td>Recall</td></tr><tr><td>0</td><td>DE</td><td>0.8618</td><td>0.9615</td><td>0.8179</td><td>0.9127</td><td>0.8867</td><td>0.5606</td></tr><tr><td>0</td><td>DIR</td><td>0.9274</td><td>0.9284</td><td>0.8491</td><td>0.8495</td><td>0.8665</td><td>0.8219</td></tr><tr><td>48</td><td>DE</td><td>0.8774</td><td>0.9635</td><td>0.8405</td><td>0.9209</td><td>0.8859</td><td>0.6176</td></tr><tr><td>48</td><td>DIR</td><td>0.9264</td><td>0.9273</td><td>0.8523</td><td>0.8527</td><td>0.8671</td><td>0.8286</td></tr><tr><td>112</td><td>DE</td><td>0.8944</td><td>0.9729</td><td>0.8235</td><td>0.9150</td><td>0.8910</td><td>0.5723</td></tr><tr><td>112</td><td>DIR</td><td>0.9271</td><td>0.9282</td><td>0.8540</td><td>0.8543</td><td>0.8665</td><td>0.8335</td></tr><tr><td>224</td><td>DE</td><td>0.8895</td><td>0.9711</td><td>0.8585</td><td>0.9285</td><td>0.8876</td><td>0.6639</td></tr><tr><td>224</td><td>DIR</td><td>0.9279</td><td>0.9288</td><td>0.8543</td><td>0.8546</td><td>0.8689</td><td>0.8313</td></tr></table>

The three-state Macro-F1 is 0.7210 on the combined sampled panel. The prediction results use generated-expression testing; observation-feedback results use the shared condition heads.

## J.2 Sensitivity to the additional gene panel

Table 28: Sensitivity to the additional 931 classification genes. All scores use the same 2,000 core genes. The middle row retrains only core heads with the original frontend. The final row removes additional-gene DE/UP values from the atlas and retrains the frontend and heads; the atlas value dimension falls from 7,862 to 6,000. Continuous output remains 2,000 throughout. External biological knowledge and the original label definitions are retained.
<table><tr><td>Variant</td><td>RMSE↓</td><td>DE AP</td><td>DE F1</td><td>DIR mAP</td><td>DIR F1</td><td>Joint F1</td></tr><tr><td>Complete model</td><td>4.4762</td><td>0.8588</td><td>0.8077</td><td>0.9135</td><td>0.8393</td><td>0.7242</td></tr><tr><td>Full atlas, core-only heads</td><td>4.4762</td><td>0.8588</td><td>0.8077</td><td>0.9135</td><td>0.8390</td><td>0.7242</td></tr><tr><td>Core-only atlas and heads</td><td>4.4715</td><td>0.8584</td><td>0.8056</td><td>0.9102</td><td>0.8326</td><td>0.7216</td></tr></table>

Removing the additional 931 genes from atlas response values, classification supervision, and output preserves core DE AP within 0.0005 of the complete model, while reducing direction Macro-F1 by 0.0067. Retraining only core heads with the original frontend changes scores little. The core-only atlas also changes projection size and initialization: frontend-plus-continuous parameters fall from 939,027 to 819,859 per seed. Consequently this comparison measures the combined input-panel and architecture change. External knowledge may still mention genes outside the core panel, and DE labels retain their original multiple-testing definition.