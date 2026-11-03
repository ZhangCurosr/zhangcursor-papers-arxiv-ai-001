# Integrating Domain Adaptation and Retrieval-Augmented Generation for Saline-Alkali Land Reclamation Planning

Zuguang He, Congcong Zheng, Wanyue Chen, Wenya Zhou, Zaijin Yang, Long Long, Yucheng Zhang and Shuqin Gao<sup>∗</sup>

Institute of Computing Technology, Chinese Academy of Sciences Beijing 100190, China

hezuguang@ict.ac.cn, zhengcongcong@ict.ac.cn, chenwanyue@ict.ac.cn

zhouwenya@ict.ac.cn, yangzaijin@ict.ac.cn, longlong@ict.ac.cn, zhangyucheng@ict.ac.cn

## Abstract

Reclamation of saline-alkali land requires management plans that account for soil and water conditions, treatment evidence, and financial constraints. This study develops Wenyao, an agricultural decision-support system integrating domain adaptation with retrieval-augmented generation. A bilingual corpus of 8,664 research papers was processed using a 110-indicator schema to construct SA-Bench, comprising 4,860 paper-level planning instances. A QwQ-32B model was adapted through low-rank supervised fine-tuning and combined with case retrieval for budget-conditioned planning. Evidence extraction achieved an F1-score of 0.83 on 300 expertannotated papers. Without retrieval, Wenyao achieved 95.83% accuracy on 936 source-disjoint multiple-choice questions, compared with 87.93% for the strongest general-purpose baseline; removing explanatory supervision reduced accuracy to 89.42%. Matched-evidence comparisons across 69 region–budget cases supported complementary contributions from adaptation and retrieval. In a blind assessment by 31 experts, Wenyao with retrieval obtained the highest mean scores for objectives and projected outcomes, whereas GPT-5 scored highest for cost estimates. These findings support literature-informed reclamation planning with professional review, while ongoing field monitoring will assess soil and crop responses.

Keywords: salt-afected soils; reclamation planning; agricultural decision support; large language model; retrieval-augmented generation; domain adaptation

## 1 Introduction

Salt-afected soils constrain agricultural production through interacting efects on crop water uptake, ion balance, soil structure, and nutrient availability [1, 2]. Reclamation consequently requires more than lowering a single soil indicator: management must address the processes responsible for salinity or sodicity and support crop establishment under local water and resource constraints. The environmental processes underlying salinization vary across landscapes [3]. Global analyses describe spatial variation in historical salinity and sodicity and project changes in primary salinization under future climates [4, 5]. Reviews of reclamation in China emphasize the need to integrate soil improvement with productive land use [6, 7]. Selecting an appropriate combination of measures for a particular field remains an important agricultural decision problem.

The suitability of an intervention depends on the conditions under which it will be implemented. Fields with similar pH may difer in electrical conductivity (EC), exchangeable sodium percentage (ESP), texture, groundwater depth, drainage, and irrigation-water quality. Reviews of salt-afected soil management describe engineering, chemical, agronomic, biological, and combined approaches, with no single treatment suitable for all sites [8, 9]. For example, evidence from combined organicamendment and gypsum treatments must be interpreted in relation to the tested soil and experimental conditions [10]. Irrigation-water salinity and water management further constrain the transfer of recommendations between locations [11, 12]. A useful planning tool must therefore relate initial conditions to treatment mechanisms, implementation requirements, and the responses actually reported in comparable studies.

Recent amendment research further illustrates the need to retain treatment context. A metaanalysis of biochar applications relates responses to environmental conditions, management, and amendment characteristics [13], while work on biochar interlayers examines how placement afects evaporation and surface salt accumulation [14]. Studies of combined amendments also evaluate soil recovery together with crop performance and biological indicators [15, 16]. Such evidence is more useful for planning when the intervention is represented by its rate, placement, water regime, and observation period rather than its name alone.

Scientific papers contain much of this evidence, but the information is dispersed across experimental descriptions, treatment tables, crop measurements, and post-treatment soil analyses. Manual synthesis becomes demanding when several soil indicators and management options must be considered simultaneously. Financial constraints add a further requirement: a technically plausible intervention may exceed the available budget or depend on water, materials, and maintenance that are unavailable locally. Economic assessments of salt-induced degradation and restoration underscore the importance of costs and the time horizon of reclamation benefits [17]. Agricultural decision-support research highlights the importance of making interacting constraints and uncertainty explicit and presenting information in a form that agricultural users can inspect [18, 19]. For reclamation planning, the central information need is a traceable relationship among site conditions, measures, expenditure, and anticipated responses.

Large language models (LLMs) provide a flexible interface for organizing heterogeneous agricultural knowledge. Recent work has investigated crop-science instruction following, plant-protection question answering, agricultural knowledge graphs, and vision–language diagnosis [20–24]. A recent review places these developments within the broader use of foundation models for agricultural tasks [25]. However, performance on agricultural questions or image-based diagnosis does not establish the ability to assemble a consistent reclamation plan. This task requires the model to connect multiple site indicators with intervention mechanisms and to distinguish published treatment responses from expectations for a new site.

Retrieval-augmented generation (RAG) supplies external evidence at generation time [26]. In a closely related application, Chen et al. [27] developed a specialized RAG system for Yunnan Arabica cofee cultivation using hybrid retrieval and reranking. This work illustrates how a domain knowledge resource can support an agricultural information service. Saline-alkali land reclamation introduces a diferent planning requirement: the output must integrate measures and their implementation costs for a land scenario described by multiple indicators. It also raises a component-level question: when a general-purpose model and an adapted model receive the same retrieved evidence, does adaptation still improve the resulting plan?

This study develops Wenyao to examine that question. A bilingual corpus of 8,664 research papers is processed into SA-Bench, a resource of 4,860 paper-level planning instances organized using a 110-indicator schema. The adapted model links land-condition analysis, measures, and reported efects; case retrieval then supplies evidence for budget-conditioned planning. The objectives are to (i) assess the reliability of literature-based evidence extraction; (ii) quantify the contribution of domain adaptation and explanatory supervision to source-supported measure selection; and (iii) assess the complementary contributions of adaptation and retrieval to plan quality under matched evidence access. Extraction evaluation, source-disjoint multiple-choice testing, component comparisons, and blind expert assessment address these objectives.

## 2 Materials and Methods

## 2.1 System Overview and Planning Task

Wenyao links scientific evidence with the soil, hydrological, climatic, and management information available for a target site. The ofline workflow constructs SA-Bench and adapts a language model; the online workflow combines model-generated land analysis with retrieved cases and the user’s budget (Figure 1). We use Wenyao for the adapted model and Wenyao + RAG for the complete planning system.

![](images/a664665dbb8c4f89ede08f12b9de86f2acd4690973c6e2f9283bf5ba33aa3f2d.jpg)  
Figure 1: System workflow connecting literature-derived evidence, model adaptation, site-specific analysis, case retrieval, and budget-aware planning. Multiple-choice evaluation uses the adapted model without the online retrieval branch. Only the training partition of SA-Bench is used for supervised adaptation.

Given a site description x, the model generates

$$
y = ( a , m , e ) ,\tag{1}
$$

where a denotes land-condition analysis, m the recommended measures, and e the anticipated efects. The analysis connects observed site indicators with the proposed intervention mechanisms. For comprehensive planning, the user also supplies a budget B, and the system provides estimated

expenditure and implementation considerations. Published treatment responses supply the evidence for anticipated efects, while expert review connects the generated recommendation with field implementation.

## 2.2 Bilingual Literature and the Indicator Schema

The initial corpus contains 8,664 research papers collected through Google Scholar, Web of Science, and CNKI: 4,486 English papers (51.8%) and 4,178 Chinese papers (48.2%). It covers reclamation measures, amendment strategies, crop selection, field experiments, and case studies. English-language sources include international study locations, while the separately collected planning cases concern 23 regions in China.

The extraction schema defines 110 indicator entries in six groups (Table 1). In addition to initial soil conditions and interventions, the schema explicitly records crop growth and posttreatment responses, preserving the relationships needed for reclamation planning. The complete field catalogue is provided in Appendix A.2, and the operative extraction instructions are summarized in Appendix A.1.

Table 1: The six groups and 110 entries in the extraction schema. Counts refer to schema entries, including distinct pre- and post-treatment fields.
<table><tr><td>Group</td><td></td><td>Entries Representative content</td></tr><tr><td>Experimental information</td><td>5</td><td>Location, coordinates, experiment type, meteorological conditions, and duration.</td></tr><tr><td>Initial land indicators</td><td>35</td><td>EC, salt content, pH and measurement method, ESP, SAR, ions, nutrients, texture, water status, and groundwater.</td></tr><tr><td>Measures and costs</td><td>4</td><td>Measure or amendment, application rate, timing, and cost.</td></tr><tr><td>Crop growth</td><td>24</td><td>Crop and cultivar, salt tolerance, physiology, canopy and root traits, water use, and nutrient or salt content.</td></tr><tr><td>Post-treatment effects</td><td>37</td><td>Biomass, yield, and post-treatment soil, water, ion, and nutrient indicators.</td></tr><tr><td>Comprehensive benefits</td><td>5</td><td>Yield increase, salinity or alkalinity reduction, input-output ratio, soil carbon, and microbial responses.</td></tr><tr><td>Total</td><td>110</td><td>Conditions, treatments, crop responses, and outcomes.</td></tr></table>

## 2.3 Extraction, Normalization, and Expert Quality Checks

The extraction workflow comprises document normalization, context-preserving indexing, grouped retrieval, constrained extraction, and expert checking (Figure 2). MinerU [28] parses PDF text and tables. Post-processing removes redundant markup, normalizes tables into Markdown, and preserves the connection between table headers, treatment labels, and reported values.

Documents shorter than 30,000 tokenizer tokens supply their complete cleaned text as extraction context. Longer documents are divided at Markdown heading levels 1–3, retaining the headings to preserve section context. The workflow uses heading-defined sections rather than fixed-length overlapping chunks. It is implemented in LangChain 0.3.27; the component versions and configuration are reported in Appendix A.

BGE-M3 [29] produces normalized dense embeddings for a Chroma index. BM25 supports lexical matching [30]. Dense retrieval returns ten candidates, and The lexical and dense rankings are combined by weighted reciprocal rank fusion (RRF) [31]:

![](images/1ada179d6ee5f0d7830635f0b58d6b64500e62d8e12f0f1b39bc46c7f8bb1c29.jpg)  
Figure 2: Literature-to-evidence workflow. Documents shorter than 30,000 tokens supply full-text context; longer documents use heading-based hybrid retrieval and cross-encoder reranking. Six indicator groups guide structured extraction. Unit harmonization and expert checks support comparison with the original papers. The expert-checked sample contains 300 papers, and the retained resource contains 4,860 paper-level instances.

$$
R ( d , q ) = \sum _ { j \in \{ \mathrm { l e x , d e n s e } \} } \frac { w _ { j } \mathbf { 1 } [ d \in \mathcal { D } _ { j } ( q ) ] } { 6 0 + r _ { j } ( d , q ) } , \qquad w _ { \mathrm { l e x } } = w _ { \mathrm { d e n s e } } = 0 . 5 ,\tag{2}
$$

where $r _ { j }$ is the one-based rank in channel $j$ . The constant 60 is the default in LangChain 0.3.27, and repeated document contents are merged. Lexical and semantic retrieval provide complementary evidence coverage [32].

For long documents, each indicator group is queried separately because its evidence may occur in diferent sections or tables. A cross-encoder reranker, BAAI/bge-reranker-v2-m3, reranks the fused candidates and retains up to ten chunks per group:

$$
\mathcal { C } _ { g } = \mathrm { T o p } _ { 1 0 } ^ { \mathrm { c r o s s } } \left( \mathrm { R R F } _ { 0 . 5 , 0 . 5 } \left[ \mathrm { B M } 2 5 _ { k _ { B } } ( q _ { g } ) , \mathrm { D e n s e } _ { 1 0 } ( q _ { g } ) \right] \right) .\tag{3}
$$

The implementation uses the BM25 instance default $k _ { B } = 4$ . Software versions and the distinction between this extraction path and online case retrieval are documented in Appendix A.

The extraction experiments use DeepSeek-R1-Distill-Qwen-32B to produce JSON records from the group-specific fields and selected context. The prompt retains multiple treatment rows and associated explanations, fixes the order of output fields, and represents unreported values as Not provided. The six field groups and the extraction constraints are documented in Appendix A.1.

Extracted indicators are harmonized to common units and normalized representations before comparison with expert-checked reference values. Measurement methods and treatment context are retained with the values. Domain experts inspect a random sample of 300 papers. Numerical indicators are compared directly after unit and representation normalization, while textual indicators are judged for semantic equivalence by the experts. For each paper, precision is the fraction of extracted indicators judged correct, recall is the fraction of reference indicators recovered correctly, and F1 is their harmonic mean. These metrics are averaged across papers. Paper-level bootstrap resampling with replacement uses 1,000 replicates [33]; each replicate resamples the 300 annotated papers and recomputes the paper-averaged metrics. The resulting empirical distribution provides the reported 95% bootstrap intervals.

## 2.4 Paper-Level Instances and Source-Disjoint Partitioning

After curation, 4,860 source papers each contribute one planning instance and one corresponding MCQ. Each instance combines the initial land description with an explanatory analysis, supported measures, and reported efects. Cleaning and deduplication retain records containing more than 20 indicators. MCQs pair the source-supported measure with three alternatives drawn from other instances. Their answers are defined by the intervention supported in the corresponding source case. Document-derived questions provide a task-specific assessment of the recovered knowledge [34]; they do not establish that the keyed measure is the only agronomically feasible intervention.

Papers are randomly assigned to disjoint training and test partitions. No information derived from a test paper appears in the supervised training partition. The training and test partitions contain 3,924 and 936 instances, corresponding to 80.74% and 19.26% of the 4,860-paper resource, respectively.

Table 2: Paper-level corpus, training/test counts, and instance-length statistics. Each retained source paper contributes one instance.
<table><tr><td>Statistic</td><td>Value</td></tr><tr><td>Initial source papers</td><td>8,664</td></tr><tr><td>English / Chinese source papers</td><td>4,486 / 4,178</td></tr><tr><td>Retained papers / planning instances</td><td>4,860 / 4,860</td></tr><tr><td>Training instances / held-out MCQs</td><td>3,924 / 936</td></tr><tr><td>Input tokens, minimum / mean / maximum</td><td>66 / 453 / 1,605</td></tr><tr><td>Output tokens, minimum / mean / maximum</td><td>291 / 1,194 / 2,862</td></tr><tr><td>Total tokens, minimum / mean / maximum</td><td>385 / 1,647 / 4,020</td></tr></table>

The retained cases cover engineering, agronomic, chemical, and biological measures. Appendix B illustrates how site conditions, interventions, and reported responses are organized.

## 2.5 Domain Adaptation and Greedy Generation

Wenyao is initialized from QwQ-32B [35] and adapted through LoRA [36] in LLaMA-Factory [37]. The training mixture adds 2,000 general examples from alpaca-gpt4-data-zh and 94 identity examples to the source-disjoint SA-Bench training partition. Targets contain land analysis, measures, and efects. The supervised objective is

$$
{ \mathcal { L } } _ { \mathrm { S F T } } = - \sum _ { i } \sum _ { t } \log p _ { \theta , \phi } ( y _ { i , t } \mid x _ { i } , y _ { i , < t } ) ,\tag{4}
$$

where $\theta$ denotes the base model and $\phi$ the trainable LoRA parameters. The rationale ablation removes the land analysis and explanatory content while retaining the same scenarios, measures, and efects.

![](images/0f2def4bf1803eaa20e3953740b5ecde6b718863e099cf4d582bc160965a0e37.jpg)

![](images/328b2ff0e5e51f5000c0e6131e1f06484a6c8b8db7383211bb4844191b9c10a7.jpg)

C Land type  
![](images/f4a10316ad649e994c25e8e9e2b9e0a7bc2249d5ca275153d0b3f656c7f0aac1.jpg)

D Measure type  
![](images/57ac952af78e292fb932e21d73ed89015b195fec27dbe39b1ff414a19f39badf.jpg)  
Figure 3: Assigned SA-Bench labels across salinity severity, soil type, land type, and intervention category. Undetermined labels are not plotted, so the totals vary across dimensions. These are full-resource counts; held-out category counts appear in Appendix C.

Training uses two H20 GPUs with 96 GB each, 32 CPU processors, and 192 GB system memory. LoRA uses rank 8, alpha 16, dropout 0, and the attention projection modules q\_proj, k\_proj, v\_proj, and o\_proj. The remaining adaptation settings appear in Appendix $\operatorname { A } ;$ unspecified options use the framework defaults. Inference uses greedy generation: sampling is disabled, and API-based generation uses temperature zero. The MCQ evaluations use the same scenarios and answer options for all models without retrieved evidence.

## 2.6 Budget-Aware Retrieval-Augmented Planning

Wenyao first analyzes the target land and proposes candidate measures. Online retrieval concatenates the land request and initial response as its query. Each evidence-store document contains a case instruction and response, with its source-paper identity retained as metadata. The planner uses the same normalized BGE-M3 embeddings, Chroma dense retrieval with k = 10, BM25, and equal-weight RRF with constant 60. The online planning workflow directly uses the fused case list without the additional cross-encoder stage used for long-document extraction. Cost information is assembled for the retrieved measures before budget-conditioned generation. The final generation stage integrates the land information, initial analysis, retrieved evidence, and budget:

$$
\hat { y } _ { B } = G _ { \theta , \phi } \left( x , B , \hat { y } _ { 0 } , \mathcal { E } ( x , \hat { y } _ { 0 } ) \right) .\tag{5}
$$

Here, $\hat { y } _ { 0 }$ is the initial model output and E denotes retrieved evidence. The resulting plan organizes proposed operations, estimated expenditure, and anticipated efects for expert review.

For matched-evidence evaluation, Qwen3-Max receives the same retrieved cases as Wenyao. This separates the contribution of evidence access from the additional contribution of the adapted generator. The fused list is deduplicated by content and has no additional fixed top-n cutof in the planning implementation; its size depends on overlap between the two channels.

## 2.7 Evaluation Protocols

## 2.7.1 Model-Level Measure Selection

The comparison models are QwQ-32B, Qwen3-Max, DeepSeek-V3, DeepSeek-R1, GPT-5, and Gemini-2.5-Pro [38–40]. Each receives the same held-out scenario and four candidate measures under greedy decoding. Accuracy is the proportion of the 936 questions for which the selected option matches the source-supported answer. Mean wall-clock response time is recorded over the same task.

## 2.7.2 Automatic Plan Comparison over 69 Cases

The practical scenario pool covers 23 regions under three budgets: 1,000, 3,000, and 5,000 CNY mu<sup>−1</sup>. One mu is approximately 667 m<sup>2</sup>. This yields 69 region–budget cases for each system. DeepSeek-R1 and GPT-5 compare paired plans on precision, expertise, practicality, and innovation and select a preferred plan. Self-comparisons are omitted. Model-based judging provides a complementary assessment of generated outputs [41]; the expert assessments described below provide the domainspecific human evaluation.

For a system pair, the first system’s preference rate is 100W/69, where W is the number of cases in which the judge prefers it. Preference rates are displayed as percentages rounded to two decimal places. The percentages are calculated from the integer preference counts using the common denominator of 69. These are relative plan-preference rates rather than measurements of field efectiveness.

## 2.7.3 Expert Assessment and Score Aggregation

The primary human assessment selects 12 of the same 23 regions and includes all three budgets for each, giving 36 questionnaires drawn from the 69-case automatic pool. Each questionnaire contains four anonymized plans, in randomized order, from Wenyao + RAG, Qwen3-Max, DeepSeek-R1, and GPT-5. Thirty-one experts from 22 institutes participate, with a mean reported domain experience of 7.7 years. Five or six experts assess each questionnaire.

Experts assign 0–10 scores for rationality of objectives and measures (Objectives), accuracy of cost estimates (Cost), and validity of projected outcomes (Conclusions). Multiple ratings of the same questionnaire are averaged before system comparison:

$$
\bar { s } _ { q m c } = \frac { 1 } { K _ { q } } \sum _ { r = 1 } ^ { K _ { q } } { s _ { q r m c } } ,\tag{6}
$$

where q indexes a region–budget questionnaire, r an expert, m a model, and c a criterion. Individual expert ratings are retained separately for inter-rater reliability estimation.

A supplementary matched-evidence assessment compares Wenyao, Wenyao + RAG, Qwen3-Max, and Qwen3-Max + RAG using six experts, eight regions, and the same three budgets, giving 24 cases. It is reported separately from the primary assessment.

## 2.7.4 Statistical Analysis

Inter-rater reliability is evaluated from the individual expert ratings using a two-way random-efects, absolute-agreement ICC for average ratings, ICC(2,k), where k is the number of ratings in the corresponding expert panel [42]. For model comparisons, expert ratings are first averaged within each questionnaire, and the three budget-specific means are then averaged within a region. The region is the independent paired unit, giving 12 pairs in the primary assessment and eight in the supplementary assessment. Two-sided paired Wilcoxon signed-rank tests compare Wenyao + RAG with each baseline on each criterion. Holm adjustment [43] is applied to the three primary baseline contrasts within a criterion, with an adjusted significance threshold of 0.05.

Paired Cohen’s $d _ { z }$ is the mean of the regional score diferences divided by their standard deviation. Magnitudes of 0.2, 0.5, and 0.8 are used as conventional small, medium, and large interpretive references [44]; the standardization uses paired diferences.

## 3 Results

## 3.1 Evidence-Extraction Performance

On 300 expert-annotated papers, the full pipeline achieves precision 0.76, recall 0.92, and F1 0.83, with a 95% bootstrap interval of [0.80, 0.85] for F1 (Table 3). Evaluation compares normalized indicators after unit harmonization, using 1,000 bootstrap replicates.

Table 3: Evidence-extraction ablations on 300 expert-annotated papers. Intervals are the reported 95% bootstrap intervals for F1.
<table><tr><td>Configuration</td><td>Precision</td><td>Recall</td><td>F1</td><td>95% CI</td></tr><tr><td>Full pipeline</td><td>0.76</td><td>0.92</td><td>0.83</td><td>[0.80, 0.85]</td></tr><tr><td>pdfplumber only</td><td>0.60</td><td>0.62</td><td>0.61</td><td>[0.58, 0.63]</td></tr><tr><td>BM25 retrieval only</td><td>0.72</td><td>0.85</td><td>0.78</td><td>[0.75, 0.80]</td></tr><tr><td>Dense retrieval only</td><td>0.70</td><td>0.83</td><td>0.77</td><td>[0.74, 0.79]</td></tr><tr><td>Without multi-round batching</td><td>0.74</td><td>0.76</td><td>0.75</td><td>[0.73, 0.78]</td></tr><tr><td>Simplified extraction prompts</td><td>0.66</td><td>0.90</td><td>0.76</td><td>[0.75, 0.78]</td></tr></table>

Replacing the parsing and normalization stage with pdfplumber alone reduces F1 from 0.83 to 0.61. Hybrid retrieval outperforms either retrieval channel alone, while grouped retrieval improves recall by collecting evidence for indicators distributed across sections and tables. Structured prompts improve precision relative to simplified instructions. Together, these ablations support preserving document structure and aligning extraction with the indicator schema.

## 3.2 Measure Selection on Source-Disjoint Papers

Wenyao achieves 95.83% accuracy on the 936 held-out MCQs, exceeding DeepSeek-V3, the strongest general-purpose baseline, by 7.90 percentage points (87.93%). QwQ-32B attains 83.01%, giving a 12.82-point improvement after adaptation (Figure 4). All models receive identical scenarios and candidate options without retrieval.

Category-level results appear in Appendix C. Against DeepSeek-V3, Wenyao’s improvements are 1.50 points for engineering measures, 11.93 for agronomic measures, 10.73 for chemical measures, and 12.87 for biological measures. The larger gains in agronomic, chemical, and biological categories indicate improved recognition of the source-supported measures represented in SA-Bench. Wenyao’s observed mean response time is 1.95 s per question, compared with 14.29 s for QwQ-32B. Because these observations use the reported serving environments, they do not isolate a hardware-controlled eficiency advantage.

![](images/56edca1bb160ae3843332725014496c15625bcc16a758e66f1bc584f00207393.jpg)  
Figure 4: Multiple-choice accuracy and mean response time on 936 held-out questions without retrieval. Response times describe the reported serving environments; the time axis is logarithmic.

## 3.3 Rationale-Enhanced Supervision

Removing the land analysis and explanatory content from the training targets reduces overall accuracy from 95.83% to 89.42%, a decrease of 6.41 percentage points. Inputs, source cases, target measures, and efects remain fixed. The result supports the contribution of explanatory supervision to learning the relationships among site conditions, measures, and outcomes.

## 3.4 Retrieval and Adaptation in Comprehensive Planning

Automatic comparisons over all 69 region–budget cases favor Wenyao + RAG over the baseline models in the valid comparisons (Appendix D). Table 4 separates the efects of adaptation and evidence access by comparing Wenyao and Qwen3-Max with and without the same retrieved cases.

Adding retrieval to Wenyao receives preference rates of 66.67% from DeepSeek-R1 and 60.87% from GPT-5. Under matched evidence access, the adapted generator also retains a majority preference. The comparison with Qwen3-Max follows the same complementary pattern: retrieval strengthens the general-purpose generator, while the domain-adapted model provides further benefits in organizing a condition-specific plan.

## 3.5 Blind Expert Assessment

In the primary study, Wenyao + RAG receives the highest mean scores for Objectives (6.89) and Conclusions (7.60; Table 5). Scores are first averaged across the experts assigned to a questionnaire and then summarized across the evaluated cases.

Table 4: Preference rates (%) favoring the first system. Each judge evaluates the same 69 region–budget cases; values are 100W/69, rounded to two decimal places.
<table><tr><td>First system versus second system</td><td>DeepSeek-R1</td><td>GPT-5</td></tr><tr><td>Wenyao versus Qwen3-Max</td><td>65.22</td><td>62.32</td></tr><tr><td>Wenyao versus Qwen3-Max + RAG</td><td>55.07</td><td>52.17</td></tr><tr><td> $\mathrm { W e n y a o + R A G }$  versus Wenyao</td><td>66.67</td><td>60.87</td></tr><tr><td> $\mathrm { W e n y a o + R A G }$  versus Qwen3-Max</td><td>76.81</td><td>75.36</td></tr><tr><td> $\mathrm { W e n y a o + R A G }$  versus Qwen3-Max + RAG</td><td>60.87</td><td>57.97</td></tr><tr><td>Qwen3-Max + RAG versus Qwen3-Max</td><td>72.46</td><td>69.57</td></tr></table>

Table 5: Primary assessment by 31 experts from 22 institutes. The 36 questionnaires cover 12 regions sampled from the 23-region automatic evaluation pool, with three budgets per region. Each questionnaire has five or six expert ratings on a 0–10 scale.
<table><tr><td>System</td><td>Objectives</td><td>Cost</td><td>Conclusions</td></tr><tr><td>Wenyao + RAG</td><td>6.89</td><td>6.95</td><td>7.60</td></tr><tr><td>Qwen3-Max</td><td>6.80</td><td>6.98</td><td>6.80</td></tr><tr><td>DeepSeek-R1</td><td>6.55</td><td>7.00</td><td>7.10</td></tr><tr><td>GPT-5</td><td>6.79</td><td>7.13</td><td>7.43</td></tr></table>

Inter-rater ICC(2,k) estimates range from 0.68 to 0.95. The regional paired analysis identifies statistically significant system diferences under the Wilcoxon protocol with Holm adjustment $( p _ { \mathrm { a d j } } < 0 . 0 5 )$ , with paired efect magnitudes of approximately $| d _ { z } | = 0 . 8 .$ . These aggregate inferential summaries accompany the criterion-specific mean scores in Table 5 and should not be interpreted as evidence that every system–criterion contrast is significant. GPT-5 obtains the highest mean Cost score (7.13), whereas Wenyao + RAG scores 6.95 on this criterion.

The supplementary six-expert study evaluates both generators under matched evidence access (Table 6). Retrieval increases all three mean scores for each generator. Wenyao + RAG attains 7.16 for Objectives and 7.68 for Conclusions, compared with 6.96 and 7.38 for Qwen3-Max + RAG. The descriptive diferences support a contribution from adaptation to objectives and projected outcomes under common evidence access. Qwen3-Max + RAG receives the highest mean Cost score (7.02), compared with 6.92 for Wenyao + RAG.

Table 6: Supplementary matched-evidence assessment by six experts over eight regions and three budgets, giving 24 region–budget cases. Means are reported separately from the primary study.
<table><tr><td>System</td><td>Objectives</td><td>Cost</td><td>Conclusions</td></tr><tr><td>Wenyao</td><td>6.82</td><td>6.63</td><td>6.90</td></tr><tr><td>Wenyao + RAG</td><td>7.16</td><td>6.92</td><td>7.68</td></tr><tr><td>Qwen3-Max</td><td>6.67</td><td>6.94</td><td>6.86</td></tr><tr><td> $\mathrm { \Delta Q w e n 3 – M a x + R A G }$ </td><td>6.96</td><td>7.02</td><td>7.38</td></tr></table>

## 4 Discussion

## 4.1 Agronomic Interpretation of Evidence-Based Planning

The results indicate that domain adaptation and case retrieval can help organize the information needed for reclamation planning. The agricultural value lies in connecting site conditions with the mechanisms and implementation requirements of a proposed measure. The 110-indicator schema records initial conditions, treatment details, crop growth, and post-treatment responses in a common structure. This organization is consistent with the integrated view of salt-afected soil management presented in reclamation reviews [6–8].

Experimental studies of leaching with organic amendments and of gypsum–compost–water combinations show why water quality and treatment interactions must accompany the amendment description [45, 46]. This provides an agronomic rationale for retaining treatment and hydrological context in the evidence records.

The intervention-category results suggest that adaptation was particularly helpful for recognizing source-supported agronomic, chemical, and biological measures. These choices often depend on combinations of indicators rather than a single threshold. The rationale ablation further suggests that explanatory supervision helped the model learn relationships among conditions, measures, and reported efects. The gains concern source-supported measure selection and plan presentation.

## 4.2 Complementary Roles of Adaptation and Retrieval

The matched-evidence comparisons help distinguish the contribution of an adapted generator from the contribution of access to external cases. Retrieval improved plan preference and expert mean scores for both Wenyao and Qwen3-Max. Wenyao with retrieval retained higher mean scores for objectives and projected outcomes when the comparator received the same cases. These findings support the combination of domain-specific training and case-based evidence access for this planning task, without implying that adaptation replaces verification of the retrieved material.

The cofee-cultivation system of Chen et al. [27] provides a relevant comparison: both applications use a specialized evidence resource to support agricultural knowledge services. The present study examines planning across soil indicators and budget levels and compares adapted and generalpurpose generators under shared evidence access. Its extraction F1, measure-selection accuracy, and plan-preference rates address diferent evaluation targets from the cofee study’s citation-leve F1 and therefore should not be compared numerically. More generally, research on long-context language models shows that including relevant information does not guarantee reliable use of that information [47]. The extraction ablations here support retaining document structure and selecting context by indicator group.

## 4.3 Practical Implications for Agricultural Advisers

Wenyao is most appropriately used to prepare a plan for professional review. The output organizes land analysis, proposed operations, estimated expenditure, and anticipated responses, enabling an adviser to inspect the links between the request and the available evidence. This role aligns with agricultural decision-support research that emphasizes user interpretation and the treatment of interacting constraints [18, 19]. Relevant source conditions, measurement methods, treatment rates, and observation periods remain necessary when deciding whether a retrieved case is applicable.

Budget conditioning should also be distinguished from validated economic optimization. In the primary expert study, GPT-5 obtained the highest mean Cost score; in the supplementary matched-evidence study, Qwen3-Max with retrieval obtained the highest score on that criterion.

The present system consequently has stronger support for organizing objectives and projected responses than for improving cost accuracy. Local material prices, labor, transport, irrigation availability, and maintenance requirements must be checked before implementation. The three evaluated budgets represent planning constraints, and the reported scores do not measure realized expenditure, profitability, or long-term maintenance costs.

## 4.4 Limitations of the Evidence and Evaluation

Several limitations afect the interpretation of the results. First, extraction precision of 0.76 indicates that errors remain despite high recall. Unit harmonization cannot remove diferences in sampling depth, EC or pH measurement method, experimental duration, or treatment context. Retrieved values and the relationships among treatment rows require verification against the original paper. The evidence resource reflects the available published literature and its reporting practices rather than a prospectively balanced set of field trials.

Second, paper-level separation prevents supervised training on the held-out source papers but does not establish separation by site, region, study family, or foundation-model pretraining data. The MCQs assess recognition of the recorded intervention among alternatives drawn from other cases. A source-supported answer need not be the only agronomically feasible option. The 23-region planning pool and both expert assessments concern China; a bilingual source corpus alone does not demonstrate transfer to other agricultural regions.

Third, automated plan preferences and expert ratings assess proposed plans rather than observed crop or soil responses. Model judges can exhibit position, verbosity, and self-enhancement biases [41]. The human assessments provide complementary professional judgment, but their independent regional sample sizes are 12 and eight, respectively. The aggregate statistical summaries do not resolve every criterion-by-baseline contrast, and the small diference in the primary Objectives means should be interpreted alongside that limitation. Likewise, response times measured in diferent serving environments do not establish a controlled computational-eficiency advantage.

Finally, retrieval access does not ensure that every generated claim is supported by a source. Work on citation-based generation and RAG evaluation distinguishes evidence retrieval, answer relevance, and claim support [48, 49]. Those distinctions motivate future evaluation of source attribution and unsupported recommendations in addition to overall plan preference. The present study does not report a claim-level citation audit or a prospective agronomic outcome comparison.

## 4.5 Field Validation and Further Development

Following technical review by domain experts, field validation has been initiated, with soil and crop observations ongoing over the relevant growth cycles. No completed field-treatment outcome data are reported here. The next stage should relate the implemented measures to changes in soil conditions, crop establishment and yield, and actual expenditure, while recording the local conditions that afect transfer from a published case. The planned release of derived data, implementation code, indicator specifications, and evaluation questionnaires will support replication. Further development should prioritize evidence quality, current local cost information, and outcome-based validation before broader deployment.

## 5 Conclusions

Integrating domain adaptation with case retrieval provided a literature-informed approach to salinealkali land reclamation planning. The 110-indicator extraction workflow achieved F1 0.83 on 300 expert-annotated papers, and Wenyao achieved 95.83% source-supported measure-selection accuracy on 936 held-out questions without retrieval. Removing explanatory supervision reduced accuracy to 89.42%. Matched-evidence comparisons supported complementary contributions from adaptation and retrieval. In the expert assessments, the complete system obtained the highest mean scores for objectives and projected outcomes, while the strongest cost scores were obtained by comparator systems. These results support the use of Wenyao to organize reclamation evidence and prepare plans for professional review. Prospective validation of soil improvement, crop response, and realized costs remains necessary before judging agronomic or economic efectiveness.

## Declarations

Author contributions. The order of authorship reflects the authors’ relative contributions to this work. Conceptualization, S.G., Z.H. and C.Z.; software, Z.H.; investigation, Z.H. and C.Z.; validation, Z.H., C.Z. and W.C.; formal analysis, Z.H.; writing—original draft preparation, Z.H.; writing—review and editing, Z.H. and C.Z.; funding acquisition, S.G., Z.Y., W.Z., L.L. and Y.Z.

Funding. Supported by the Chinese Academy of Sciences.

Ethics statement. Institutional permission was obtained from the Institute of Computing Technology, Chinese Academy of Sciences, for the non-interventional expert-questionnaire assessment. Expert participation was voluntary and based on informed consent, and evaluation results are presented in aggregate form. The study involved literature-based computational analyses and professional assessment of model-generated reclamation plans; no clinical interventions or animal experiments were conducted.

Informed consent. Informed consent was obtained from all experts involved in the evaluation.

Data availability. The derived SA-Bench data, implementation code, and evaluation questionnaires supporting this study will be made publicly available in a future release. Released expert-evaluation records will be anonymized, and access to the original scientific papers remains subject to the respective publishers’ licenses and database terms.

Acknowledgments. We thank the domain experts who contributed to evidence checking and blind assessment of reclamation plans. Generative models were used for evidence extraction, instructiondata construction, plan generation, and complementary automatic evaluation as described in the Methods.

Conflicts of interest. The authors declare no conflicts of interest.

## Abbreviations

BM25 Best Matching 25

EC Electrical conductivity

ESP Exchangeable sodium percentage

ICC Intraclass correlation coeficient

LLM Large language model

LoRA Low-rank adaptation

MCQ Multiple-choice question

RAG Retrieval-augmented generation

SAR Sodium adsorption ratio

SFT Supervised fine-tuning

## A Implementation and Prompts

Table A1 records the adaptation, greedy decoding, and extraction-evaluation settings. Table A2 provides the retrieval and reranking implementation settings, with software versions recorded in the implementation specification.

Table A1: Adaptation, generation, and extraction-evaluation settings.
<table><tr><td>Component</td><td>Reported setting</td></tr><tr><td>Structured extraction model</td><td>DeepSeek-R1-Distill-Qwen-32B</td></tr><tr><td>Planning model initialization</td><td>QwQ-32B</td></tr><tr><td>Framework / adaptation</td><td>LLaMA-Factory; LoRA</td></tr><tr><td>Learning rate / optimizer</td><td>5 × 10−5 / AdamW</td></tr><tr><td>Epochs / LoRA rank</td><td>6/8</td></tr><tr><td>LoRA alpha / dropout</td><td>16 /  0</td></tr><tr><td>LoRA target modules Per-device batch /</td><td>q_proj, k_proj, v_proj, o_proj</td></tr><tr><td>accumulation Auxiliary general /</td><td>4/8</td></tr><tr><td>identity examples</td><td>2,000 / 94</td></tr><tr><td>Generation Extraction bootstrap</td><td>Greedy; do_sample=false; temperature 0 1,000</td></tr><tr><td>replicates</td><td></td></tr><tr><td>Hardware</td><td>2 H20 GPUs, 96 GB each; 32 CPUs; 192 GB RAM</td></tr></table>

## A.1 Structured Indicator Extraction Prompt

The following excerpt summarizes the operative constraints of the extraction specification. The field list and evidence context are populated separately for each of the six indicator groups. The complete field catalogue is reproduced below.

You are an agricultural expert. Please strictly extract relevant indicators for saline-alkali land remediation based on the following context.

Table A2: Retrieval and reranking settings. Library defaults correspond to the reported software versions. The extraction and planning paths have diferent final retrieval stages.
<table><tr><td>Component</td><td>Implementation setting</td></tr><tr><td>LangChain / community</td><td>0.3.27 / 0.3.29</td></tr><tr><td>Chroma / Hugging Face integrations</td><td>langchain_chroma 0.2.6; langchain_huggingface 0.3.1</td></tr><tr><td>Core / text splitters</td><td>0.3.76 / 0.3.11</td></tr><tr><td>Sentence Transformers / rank-bm25</td><td>5.0.0 / 0.2.2</td></tr><tr><td>Embedding model Full-text extraction</td><td>BAAI/bge-m3; CUDA; normalized embeddings Fewer than 30,000 tokenizer tokens</td></tr><tr><td>threshold Long-document splitting</td><td>Markdown headings 1-3; strip_headers=false</td></tr><tr><td>Dense retrieval</td><td>Chroma; search_kwargs={&quot;k&quot;:10}</td></tr><tr><td>BM25 lexical retrieval</td><td>Instance default kB = 4; default whitespace tokenization</td></tr><tr><td>Fusion</td><td>Weighted RRF; weights (0.5, 0.5); default constant c = 60; content-based</td></tr><tr><td>Extraction reranker</td><td>deduplication BAAI/bge-reranker-v2-m3; CrossEncoderReranker; explicit top_n=10</td></tr><tr><td>Other reranker model options</td><td>Pretrained-model and wrapper defaults</td></tr><tr><td>Online planning retrieval</td><td>Case-level RRF output; no subsequent cross-encoder or fixed final cutoff</td></tr></table>

Requirements: Extract information strictly from the context. Fabrication, inference, or supplementation is prohibited. For tabular data, completely extract all relevant values; retain all relevant rows. For indicators not explicitly mentioned, return “Not provided”. Retain explanatory context needed to understand an indicator. The number and order of output fields must match the given indicator list. Return JSON directly, without additional explanations or a code-fence prefix.

Indicator list to extract: {metrics}

Context: {context}

An illustrative output structure is shown below. The placeholders describe schema fields and are not additional experimental observations.

```json
{
"Location": "value explicitly stated in context",
"Soil pH": "value explicitly stated in context",
"Reclamation measures": "reported treatments",
"Reclamation effects": "reported outcomes",
"Unreported indicator": "Not provided"
}
```

## A.2 Complete Indicator Catalogue

The extraction schema contains 110 entries. The catalogue below reproduces their field names; the pre- and post-treatment prefixes are omitted within the corresponding groups for readability. The extraction constraints are summarized in Appendix A.1. Field names follow the extraction schema and should be interpreted with the measurement methods and treatment context reported in each source paper. Prompt examples illustrate the extraction format and are not additional experimental observations. Experimental information (5 entries). Location; Latitude and Longitude; Experiment $\mathrm { T y p e } ;$ Meteorological Conditions; Experiment Duration.

Initial land indicators (35 entries). Saline-Alkali Land Type; Electrical Conductivity (EC); Total Soluble Salt Content; pH; pH Measurement Method; Exchangeable Sodium Percentage (ESP); Sodium Adsorption Ratio (SAR):; Cation Exchange Capacity (CEC); Sodium Ion $\mathrm { ( N a ^ { + } ) }$ Content; Calcium Ion $( \mathrm { C a ^ { 2 + } } )$ Content; Magnesium Ion $( \mathrm { M g ^ { 2 + } } )$ Content; Chloride Ion (Cl<sup>−</sup>) Content; Sulfate Ion $\mathrm { ( S O _ { 4 } ^ { 2 - } ) }$ Content; Bicarbonate $\left( \mathrm { H C O _ { 3 } } ^ { - } \right)$ Content; Carbonate $\left( \mathrm { C O _ { 3 } } ^ { 2 - } \right)$ Content; Soil Organic Matter (SOM) Content; Soil Texture; Soil Bulk Density; Total Porosity; Soil Temperature; Calcium Carbonate $\left( \mathrm { C a C O _ { 3 } } \right)$ Content; Available Phosphorus (P) Content; Available Potassium (K) Content; Total Nitrogen (N) Content; Ammonium Nitrogen $\mathrm { ( N H _ { 4 } ^ { + } { - } N ) }$ Content; Nitrate Nitrogen $\left( \mathrm { N O _ { 3 } } ^ { - } \mathrm { - N } \right)$ Content; Trace Element Content; Field Capacity (FC); Permanent Wilting Point (PWP); Soil Water Content (SWC); Saturated Hydraulic Conductivity (Ks); Capillary Rise Height; Groundwater Depth; Irrigation Water EC; Annual Evaporation/Precipitation Ratio.

Measures and costs (4 entries). Measure/Amendment; Application Rate/Intensity; Duration/Timing;   
Cost.

Crop growth (24 entries). Experimental Crop Name(s); Seed Pre-treatment; Genotype/Cultivar; Trait; Photosynthetic Rate (Pn); Respiration Rate; Stomatal Conductance (gs); Transpiration Rate (Tr); Chlorophyll Content; Intrinsic Water Use Eficiency $\mathrm { ( P n / g s ; }$ labeled WUE in the extraction schema); Specific Leaf Area (SLA); Leaf Area; Plant Height; Root Length Density (RLD); Root Length; Root Surface Area; Root Volume; Root Diameter; Proline, Soluble Sugar Content; Specific Root Length (SRL); Root:Shoot Ratio, Root:Stem:Leaf Ratio; Crop Water Use Eficiency (CWUE); Crop Nutrient Content; Crop Salt Content.

Post-treatment efects (37 entries). Biomass; Yield; Saline-Alkali Land Type; Electrical Conductivity (EC); Total Soluble Salt Content; pH; pH Measurement Method; Exchangeable Sodium Percentage (ESP); Sodium Adsorption Ratio (SAR):; Cation Exchange Capacity (CEC); Sodium Ion $\mathrm { ( N a ^ { + } ) }$ Content; Calcium Ion $\mathrm { ( C a ^ { 2 + } ) }$ Content; Magnesium Ion $( \mathrm { M g ^ { 2 + } } )$ Content; Chloride Ion (Cl<sup>−</sup>) Content; Sulfate Ion $\mathrm { ( S O _ { 4 } } ^ { 2 - } \mathrm { ) }$ Content; Bicarbonate $\left( \mathrm { H C O _ { 3 } } ^ { - } \right)$ Content; Carbonate $\left( \mathrm { C O _ { 3 } } ^ { 2 - } \right)$ Content; Soil Organic Matter (SOM) Content; Soil Texture; Soil Bulk Density; Total Porosity; Soil Temperature; Calcium Carbonate $\left( \mathrm { C a C O _ { 3 } } \right)$ Content; Available Phosphorus (P) Content; Available Potassium (K) Content; Total Nitrogen (N) Content; Ammonium Nitrogen $\mathrm { ( N H _ { 4 } ^ { + } { - } N ) }$ Content; Nitrate Nitrogen $\left( \mathrm { N O _ { 3 } } ^ { - } \mathrm { - N } \right)$ Content; Trace Element Content; Field Capacity (FC); Permanent Wilting Point (PWP); Soil Water Content (SWC); Saturated Hydraulic Conductivity (Ks); Capillary Rise Height; Groundwater Depth; Irrigation Water EC; Annual Evaporation/Precipitation Ratio.

Comprehensive benefits (5 entries). Crop Yield Increase Rate; Soil Salinity/Alkalinity Reduction Rate;   
Amendment Input-Output Ratio; Soil Organic Carbon (SOC) Increment; Microbial Quantity and Diversity.

## A.3 Planning Instruction and Evaluation Rubric

The instruction in the training examples asks: “Please suggest suitable reclamation measures for this saline-alkali land and explain the reclamation efects.” The target is organized as Land Analysis, Reclamation Measures, and Reclamation Efects. At inference, a budget-aware request adds the user budget to the site description and asks for a technically feasible comprehensive plan with controlled costs.

Automatic plan assessment uses precision, expertise, practicality, and innovation. Judges compare both plans and provide an explanation before selecting a preference. Expert questionnaires use anonymized plans, a 0–10 scale, and space for written feedback. Repeated expert ratings of the same questionnaire are averaged for system comparison, while individual ratings are retained for ICC estimation.

## B Illustrative Evidence-to-Plan Instance

Table A3 presents selected fields from a SA-Bench source case, linking its initial conditions with the reported treatment and responses over the study duration.

Table A3: Selected fields from the Linze County evidence-to-plan example. Outcomes were reported for the source treatment and duration
<table><tr><td>Field</td><td>Source-case content</td></tr><tr><td>Location and soil</td><td>Xiaotun Township, Linze County, Gansu, China; Heihe River Basin; meadow solonchak.</td></tr><tr><td>Initial conditions</td><td>pH 8.45; salt content  $8 . 1 8 ~ \mathrm { g k g ^ { - 1 } }$  ; groundwater depth 1.50 m; bulk density 1.43  $\mathrm { g c m ^ { - 3 } } ;$  organic matter 8.50  $\mathrm { g k g ^ { - 1 } }$ </td></tr><tr><td>Climate</td><td>Mean annual precipitation 109 mm; mean annual evaporation 1,966 mm; mean  $7 . 8 ~ ^ { \circ } \mathrm { C } .$ </td></tr><tr><td>Reported</td><td>annual temperature Three years of alfalfa cultivation with manure, mineral fertilizer, and irrigation.</td></tr><tr><td>intervention</td><td>The full source instance specifies treatment quantities and timing.</td></tr><tr><td>Reported soil response</td><td>At 0–20 cm, pH decreases from 8.45 to 8.20; salt content decreases by 4.63  $\mathrm { g k g ^ { - 1 } }$  reported desalination rate 56.60%; bulk density decreases by 0.24  $\mathrm { g c m ^ { - 3 } }$ </td></tr><tr><td>Subsequent yield</td><td>Spring wheat  $6 . 5 7 \ : \mathrm { t h a } ^ { - 1 }$  and maize 9.84  $\tan \mathrm { { \alpha } } ^ { - 1 }$  after ploughing the alfalfa.</td></tr><tr><td>Evidence identity</td><td>Study on Saline Soil Resources in the Heihe River Basin and the Effects of Alfalfa on Soil Improvement, Fertility Enhancement, and Ecological Restoration (title</td></tr></table>

The source instance explains the intervention through plant cover, root efects on structure, organic matter inputs, and nutrient replenishment. The explanatory target preserves the reported study conditions and connects them with the intervention and its measured responses.

An illustrative SA-Bench MCQ describes the Yinbei area of the Ningxia Yellow River Irrigation District, with silty fine sand, poor permeability, shallow groundwater, and drainage-related salinization. The options are:

A. Subsurface pipe drainage combined with salt-tolerant plant cultivation.

B. Systematic improvement of drainage channels, including main ditches, branch ditches, and renovation of electric drainage stations.

C. Application of gypsum combined with deep plowing and soil turning.

D. Flood irrigation for salt leaching and straw mulching.

The designated source-supported answer is B, illustrating how the intervention is matched to the recorded drainage-related cause and the corresponding source case.

## C Category-Level Measure-Selection Results

Table A4 preserves the original category-level results. The overall denominator is 936 MCQs. Unassigned categories are excluded from the breakdown, and label coverage difers by dimension; category counts therefore should not be summed across dimensions or assumed to exhaust the overall set.

Table A4: Measure-selection accuracy (%) without retrieval. QwQ denotes QwQ-32B; Qwen3 denotes Qwen3-Max; V3 and R1 denote DeepSeek-V3 and DeepSeek-R1; Gemini denotes Gemini-2.5-Pro.
<table><tr><td>Dimension Category</td><td></td><td>n</td><td>Wenyao</td><td>QwQ</td><td>Qwen3</td><td>V3</td><td>R1</td><td>GPT-5</td><td>Gemini</td></tr><tr><td>Severity</td><td>Mild</td><td>171</td><td>93.57</td><td>74.85</td><td>82.46</td><td>81.29</td><td>80.12</td><td>78.36</td><td>80.12</td></tr><tr><td></td><td>Moderate</td><td>185</td><td>96.22</td><td>80.54</td><td>84.86</td><td>84.86</td><td>78.38</td><td>82.16</td><td>80.54</td></tr><tr><td></td><td>Severe</td><td>108</td><td>95.37</td><td>81.48</td><td>84.26</td><td>86.11</td><td>81.48</td><td>76.85</td><td>86.11</td></tr><tr><td>Soil</td><td>Chloride</td><td>212</td><td>96.23</td><td>85.85</td><td>91.04</td><td>90.09</td><td>86.32</td><td>88.68</td><td>86.32</td></tr><tr><td></td><td>Sulphate</td><td>48</td><td>93.75</td><td>70.83</td><td>81.25</td><td>81.25</td><td>75.00</td><td>75.00</td><td>83.33</td></tr><tr><td></td><td>Sodic</td><td>285</td><td>96.14</td><td>80.70</td><td>83.86</td><td>84.56</td><td>78.95</td><td>81.05</td><td>82.46</td></tr><tr><td>Land</td><td>Coastal</td><td>273</td><td>95.97</td><td>86.81</td><td>91.21</td><td>90.11</td><td>85.71</td><td>88.28</td><td>86.08</td></tr><tr><td></td><td>Inland</td><td>383</td><td>95.04</td><td>78.33</td><td>81.46</td><td>83.81</td><td>79.63</td><td>78.85</td><td>82.77</td></tr><tr><td></td><td>Secondary</td><td>230</td><td>96.52</td><td>84.78</td><td>89.13</td><td>90.87</td><td>86.96</td><td>87.39</td><td>85.22</td></tr><tr><td>Measure</td><td>Engineering</td><td>334</td><td>96.71</td><td>93.71</td><td>95.51</td><td>95.21</td><td>93.71</td><td>95.81</td><td>91.92</td></tr><tr><td></td><td>Agronomic</td><td>109</td><td>94.50</td><td>75.23</td><td>79.82</td><td>82.57</td><td>74.31</td><td>77.06</td><td>86.24</td></tr><tr><td></td><td>Chemical</td><td>261</td><td>95.02</td><td>77.39</td><td>84.67</td><td>84.29</td><td>80.84</td><td>78.93</td><td>81.23</td></tr><tr><td></td><td>Biological</td><td>202</td><td>96.53</td><td>76.24</td><td>78.71</td><td>83.66</td><td>76.24</td><td>74.75</td><td>77.23</td></tr><tr><td>Overall</td><td></td><td>936</td><td>95.83</td><td>83.01</td><td>86.86</td><td>87.93</td><td>83.97</td><td>84.19</td><td>84.72</td></tr></table>

## D Additional Automatic Plan Comparisons

Table A5 reports additional comparisons of Wenyao + RAG with the baseline systems. All rates use 69 region–budget comparisons per judge. The matched-evidence analysis appears in Table 4.

Table A5: Preference rates (%) favoring Wenyao + RAG across 69 region–budget comparisons. A dash denotes an omitted self-comparison.
<table><tr><td>Comparison model</td><td>DeepSeek-R1 judge</td><td>GPT-5 judge</td></tr><tr><td>QwQ-32B</td><td>79.71</td><td>76.81</td></tr><tr><td>Qwen3-Max</td><td>76.81</td><td>75.36</td></tr><tr><td>DeepSeek-V3</td><td>63.77</td><td>60.87</td></tr><tr><td>GPT-5</td><td>59.42</td><td></td></tr><tr><td>Gemini-2.5-Pro</td><td>60.87</td><td>75.36</td></tr><tr><td>DeepSeek-R1</td><td></td><td>65.22</td></tr></table>

## References

[1] Rana Munns and Mark Tester. Mechanisms of Salinity Tolerance. Annual Review of Plant Biology, 59:651–681, 2008. doi: 10.1146/annurev.arplant.59.032607.092911.

[2] Pichu Rengasamy. Soil processes afecting crop production in salt-afected soils. Functional Plant Biology, 37(7):613–620, 2010. doi: 10.1071/FP09249.

[3] Pichu Rengasamy. World salinization with emphasis on Australia. Journal of Experimental Botany, 57(5):1017–1023, 2006. doi: 10.1093/jxb/erj108.

[4] Amirhossein Hassani, Adisa Azapagic, and Nima Shokri. Predicting long-term dynamics of soil salinity and sodicity on a global scale. Proceedings of the National Academy of Sciences, 117 (52):33017–33027, 2020. doi: 10.1073/pnas.2013771117.

[5] Amirhossein Hassani, Adisa Azapagic, and Nima Shokri. Global predictions of primary soil salinization under changing climate in the 21st century. Nature Communications, 12:6663, 2021. doi: 10.1038/s41467-021-26907-3.

[6] Guangzhou Wang, Gang Ni, Gu Feng, Haley M. Burrill, Jianfang Li, Junling Zhang, and Fusuo Zhang. Saline-alkali soil reclamation and utilization in China: progress and prospects. Frontiers of Agricultural Science and Engineering, 11(2):216–228, 2024. doi: 10.15302/J-FASE-2024551.

[7] Wei Zhu, Shiguo Gu, Rui Jiang, Xin Zhang, and Ryusuke Hatano. Saline–Alkali Soil Reclamation Contributes to Soil Health Improvement in China. Agriculture, 14(8):1210, 2024. doi: 10.3390/ agriculture14081210.

[8] Mandana Shaygan and Thomas Baumgartl. Reclamation of salt-afected land: A review. Soil Systems, 6(3):61, 2022. doi: 10.3390/soilsystems6030061.

[9] Tibor Tóth. Review of sodic soil reclamation with a snapshot of current research activity. Chinese Geographical Science, 32(6):1099–1109, 2022. doi: 10.1007/s11769-022-1310-4.

[10] Demis Andrade Foronda and Gilles Colinet. Combined Application of Organic Amendments and Gypsum to Reclaim Saline–Alkali Soil. Agriculture, 12(7):1049, 2022. doi: 10.3390/agricu lture12071049.

[11] Manzoor Qadir and James D. Oster. Crop and irrigation management strategies for salinesodic soils and waters aimed at environmentally sustainable agriculture. Science of the Total Environment, 323(1–3):1–19, 2004. doi: 10.1016/j.scitotenv.2003.10.012.

[12] Negash Tessema, Dame Yadeta, Asfaw Kebede, and Gebiaw T. Ayele. Soil and Irrigation Water Salinity, and Its Consequences for Agriculture in Ethiopia: A Systematic Review. Agriculture, 13(1):109, 2023. doi: 10.3390/agriculture13010109.

[13] Liqiang Zhang, Baoyin Bate, Jinhu Cui, Yudi Feng, Jianning Yu, Zhengguo Cui, Hongyu Wang, and Qiuzhu Li. Biochar Input to Saline-Alkali Farmland Can Improve Soil Health and Crop Yield: A Meta-Analysis. Agriculture, 15(5):561, 2025. doi: 10.3390/agriculture15050561.

[14] Qiang Xu, Hongguang Liu, Mingsi Li, and Pengfei Li. The Presence of the Biochar Interlayer Efectively Inhibits Soil Water Evaporation and Salt Migration to the Soil Surface. Agriculture, 13(3):638, 2023. doi: 10.3390/agriculture13030638.

[15] María Alcívar, Andrés Zurita-Silva, Marco Sandoval, Cristina Muñoz, and Mauricio Schoebitz. Reclamation of Saline-Sodic Soils with Combined Amendments: Impact on Quinoa Performance and Biological Soil Quality. Sustainability, 10(9):3083, 2018. doi: 10.3390/su10093083.

[16] Salar Rezapour, Amin Nouri, Farrokh Asadzadeh, Mohsen Barin, Günay Erpul, Sindhu Jagadamma, and Ruijun Qin. Combining chemical and organic treatments enhances remediation performance and soil health in saline-sodic soils. Communications Earth & Environment, 4:285, 2023. doi: 10.1038/s43247-023-00948-6.

[17] Manzoor Qadir, Emmanuelle Quillérou, Vinay Nangia, Ghulam Murtaza, Murari Singh, Richard J. Thomas, Pay Drechsel, and Andrew D. Noble. Economics of salt-induced land degradation and restoration. Natural Resources Forum, 38(4):282–295, 2014. doi: 10.1111/1477-8947.12054.

[18] Martine J. Barons, Lael E. Walsh, Edward E. Salakpi, and Linda Nichols. A Decision Support System for Sustainable Agriculture and Food Loss Reduction under Uncertain Agricultural Policy Frameworks. Agriculture, 14(3):458, 2024. doi: 10.3390/agriculture14030458.

[19] Nyi-Nyi Htun, Diego Rojo, Jeroen Ooge, Robin De Croon, Aikaterini Kasimati, and Katrien Verbert. Developing Visual-Assisted Decision Support Systems across Diverse Agricultural Use Cases. Agriculture, 12(7):1027, 2022. doi: 10.3390/agriculture12071027.

[20] Hang Zhang, Jiawei Sun, Renqi Chen, Wei Liu, Zhonghang Yuan, Xinzhe Zheng, Zhefan Wang, Zhiyuan Yang, Hang Yan, Hansen Zhong, et al. Empowering and assessing the utility of large language models in crop science. Advances in Neural Information Processing Systems, 37: 52670–52722, 2024.

[21] Jie Xiong, Lingmin Pan, Yanjiao Liu, Lei Zhu, Lizhuo Zhang, and Siqiao Tan. Enhancing Plant Protection Knowledge with Large Language Models: A Fine-Tuned Question-Answering System Using LoRA. Applied Sciences, 15(7):3850, 2025.

[22] Xinyan Zhao, Baiyan Chen, Mengxue Ji, Xinyue Wang, Yuhan Yan, Jinming Zhang, Shiyingjie Liu, Muyang Ye, and Chunli Lv. Implementation of Large Language Models and Agricultural Knowledge Graphs for Eficient Plant Disease Detection. Agriculture, 14(8):1359, 2024. doi: 10.3390/agriculture14081359.

[23] Muhammad Awais, Ali Husain Salem Abdulla Alharthi, Amandeep Kumar, Hisham Cholakkal, and Rao Muhammad Anwer. AgroGPT: Eficient agricultural vision-language model with expert tuning. In 2025 IEEE/CVF Winter Conference on Applications of Computer Vision (WACV), pages 5687–5696. IEEE, 2025.

[24] Liqiong Wang, Teng Jin, Jinyu Yang, Ales Leonardis, Fangyi Wang, and Feng Zheng. Agri-LLaVA: Knowledge-infused large multimodal assistant on agricultural pests and diseases. arXiv preprint arXiv:2412.02158, 2024.

[25] Shuolei Yin, Yejing Xi, Xun Zhang, Chengnuo Sun, and Qirong Mao. Foundation Models in Agriculture: A Comprehensive Review. Agriculture, 15(8):847, 2025. doi: 10.3390/agriculture1 5080847.

[26] Patrick Lewis, Ethan Perez, Aleksandra Piktus, Fabio Petroni, Vladimir Karpukhin, Naman Goyal, Heinrich Küttler, Mike Lewis, Wen-tau Yih, Tim Rocktäschel, Sebastian Riedel, and Douwe Kiela. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. Advances in Neural Information Processing Systems, 33:9459–9474, 2020.

[27] Zheng Chen, Zihao Jiang, and Jianping Yang. A RAG-Augmented LLM for Yunnan Arabica Cofee Cultivation. Agriculture, 15(22):2381, 2025. doi: 10.3390/agriculture15222381.

[28] Bin Wang, Chao Xu, Xiaomeng Zhao, Linke Ouyang, Fan Wu, Zhiyuan Zhao, Rui Xu, Kaiwen Liu, Yuan Qu, Fukai Shang, et al. MinerU: An open-source solution for precise document content extraction. arXiv preprint arXiv:2409.18839, 2024.

[29] Jianlyu Chen, Shitao Xiao, Peitian Zhang, Kun Luo, Defu Lian, and Zheng Liu. M3- Embedding: Multi-Linguality, Multi-Functionality, Multi-Granularity Text Embeddings through Self-Knowledge Distillation. In Findings of the Association for Computational Linguistics: ACL 2024, pages 2318–2335, 2024. doi: 10.18653/v1/2024.findings-acl.137.

[30] Stephen Robertson and Hugo Zaragoza. The Probabilistic Relevance Framework: BM25 and Beyond. Foundations and Trends in Information Retrieval, 3(4):333–389, 2009. doi: 10.1561/1500000019.

[31] Gordon V. Cormack, Charles L. A. Clarke, and Stefan Büttcher. Reciprocal Rank Fusion Outperforms Condorcet and Individual Rank Learning Methods. In 32nd International ACM SIGIR Conference on Research and Development in Information Retrieval, pages 758–759, 2009. doi: 10.1145/1571941.1572114.

[32] Saar Kuzi, Mingyang Zhang, Cheng Li, Michael Bendersky, and Marc Najork. Leveraging semantic and lexical matching to improve the recall of document retrieval systems: A hybrid approach. arXiv preprint arXiv:2010.01195, 2020.

[33] Bradley Efron. Bootstrap Methods: Another Look at the Jackknife. The Annals of Statistics, 7(1):1–26, 1979. doi: 10.1214/aos/1176344552.

[34] Gauthier Guinet, Behrooz Omidvar-Tehrani, Anoop Deoras, and Laurent Callot. Automated Evaluation of Retrieval-Augmented Language Models with Task-Specific Exam Generation. In 41st International Conference on Machine Learning, volume 235 of Proceedings of Machine Learning Research, pages 16773–16801. PMLR, 2024.

[35] Qwen Team. QwQ-32B: Embracing the Power of Reinforcement Learning. Qwen technical blog, 2025.

[36] Edward J. Hu, Yelong Shen, Phillip Wallis, Zeyuan Allen-Zhu, Yuanzhi Li, Shean Wang, Lu Wang, and Weizhu Chen. LoRA: Low-Rank Adaptation of Large Language Models. In International Conference on Learning Representations, 2022.

[37] Yaowei Zheng, Richong Zhang, Junhao Zhang, Yanhan Ye, and Zheyan Luo. LlamaFactory: Unified Eficient Fine-Tuning of 100+ Language Models. In 62nd Annual Meeting of the Association for Computational Linguistics (Volume 3: System Demonstrations), pages 400–410, 2024. doi: 10.18653/v1/2024.acl-demos.38.

[38] Aixin Liu, Bei Feng, Bing Xue, Bingxuan Wang, Bochao Wu, Chengda Lu, Chenggang Zhao, Chengqi Deng, Chenyu Zhang, Chong Ruan, et al. DeepSeek-V3 Technical Report. arXiv preprint arXiv:2412.19437, 2024.

[39] Daya Guo, Dejian Yang, Haowei Zhang, Junxiao Song, Peiyi Wang, Qihao Zhu, Runxin Xu, Ruoyu Zhang, Shirong Ma, Xiao Bi, et al. DeepSeek-R1 Incentivizes Reasoning in LLMs through Reinforcement Learning. Nature, 645(8081):633–638, 2025. doi: 10.1038/s41586-025-09422-z.

[40] Gheorghe Comanici, Eric Bieber, Mike Schaekermann, Ice Pasupat, Noveen Sachdeva, Inderjit Dhillon, Marcel Blistein, Ori Ram, Dan Zhang, Evan Rosen, et al. Gemini 2.5: Pushing the frontier with advanced reasoning, multimodality, long context, and next generation agentic capabilities. arXiv preprint arXiv:2507.06261, 2025.

[41] Lianmin Zheng, Wei-Lin Chiang, Ying Sheng, Siyuan Zhuang, Zhanghao Wu, Yonghao Zhuang, Zi Lin, Zhuohan Li, Dacheng Li, Eric P. Xing, Hao Zhang, Joseph E. Gonzalez, and Ion Stoica. Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. In Advances in Neural Information Processing Systems, volume 36, pages 46595–46623, 2023. doi: 10.52202/075280-2 020.

[42] Terry K. Koo and Mae Y. Li. A Guideline of Selecting and Reporting Intraclass Correlation Coeficients for Reliability Research. Journal of Chiropractic Medicine, 15(2):155–163, 2016. doi: 10.1016/j.jcm.2016.02.012.

[43] Sture Holm. A Simple Sequentially Rejective Multiple Test Procedure. Scandinavian Journal of Statistics, 6(2):65–70, 1979.

[44] Jacob Cohen. Statistical Power Analysis for the Behavioral Sciences. Routledge, 2 edition, 1988. doi: 10.4324/9780203771587.

[45] Vijayasatya N. Chaganti, David M. Crohn, and Jirka Šimůnek. Leaching and reclamation of a biochar and compost amended saline-sodic soil with moderate SAR reclaimed water. Agricultural Water Management, 158:255–265, 2015. doi: 10.1016/j.agwat.2015.05.016.

[46] Parul Sundha, Nirmalendu Basak, Arvind Kumar Rai, Rajender Kumar Yadav, Parbodh Chander Sharma, and Dinesh Kumar Sharma. Can conjunctive use of gypsum, city waste composts and marginal quality water rehabilitate saline-sodic soils? Soil and Tillage Research, 200: 104608, 2020. doi: 10.1016/j.still.2020.104608.

[47] Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the Middle: How Language Models Use Long Contexts. Transactions of the Association for Computational Linguistics, 12:157–173, 2024. doi: 10.1162/tacl\_a\_00638.

[48] Tianyu Gao, Howard Yen, Jiatong Yu, and Danqi Chen. Enabling Large Language Models to Generate Text with Citations. In 2023 Conference on Empirical Methods in Natural Language Processing, pages 6465–6488, 2023. doi: 10.18653/v1/2023.emnlp-main.398.

[49] Shahul Es, Jithin James, Luis Espinosa Anke, and Steven Schockaert. RAGAs: Automated Evaluation of Retrieval Augmented Generation. In 18th Conference of the European Chapter of the Association for Computational Linguistics: System Demonstrations, pages 150–158, 2024. doi: 10.18653/v1/2024.eacl-demo.16.