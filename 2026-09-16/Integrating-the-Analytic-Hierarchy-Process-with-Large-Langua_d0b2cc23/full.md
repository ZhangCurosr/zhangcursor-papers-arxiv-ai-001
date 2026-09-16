# Integrating the Analytic Hierarchy Process with Large Language Models for Transparent Multi-Criteria Decision-Making

Han Zhiguang<sup>1</sup>, Farah Benamara<sup>2,3</sup> and Pascale Zaraté<sup>4</sup>

<sup>1</sup> CNRS@CREATE, Singapore, Singapore

<sup>2</sup> IRIT, Université de Toulouse, Toulouse, France

IRIT, Université Toulouse Capitole, Toulouse, France Corresponding author: farah.benamara@irit.fr

## Abstract

LLMs are increasingly employed in a wide range of decision-making tasks. However, the opacity of their internal reasoning makes it difficult to validate or interpret their outputs, and the need for interpretability becomes especially critical in high-stakes settings. This study examines the decision-making capabili ties of LLMs through the Analytic Hierarchy Process (AHP), a classical and widely used multicriteria decision-making framework. We construct a new annotated benchmark based on AHP and propose the first end-to-end approach that enables LLMs to perform the complete AHP workflow. Experiments in real-world decision problems in the legal and higher-education ranking domains show that our method significantly improves alignment with expert judgments.

## 1 Introduction

Multi-Criteria Decision Making (MCDM) provides systematic approaches for addressing complex problems that involve multiple, often conflicting objectives. Such problems are ubiquitous: policymakers must balance fairness against efficiency, healthcare professionals weigh treatment effectiveness against cost, and legal experts reconcile principles of justice with practical enforceability. Over decades, a rich set of MCDM techniques have been developed, including TOPSIS (Papathanasiou and Ploskas, 2018), PROMETHEE (Brans and De Smet, 2005), and ELECTRE (Govindan and Jepsen, 2016). Among these, we focus here on the Analytic Hierarchy Process (AHP) (Saaty, 1980; Forman and Gass, 2001; Munier et al., 2021) (see Simon (1960) and Opricovic and Tzeng (2007) for an overview).

AHP addresses decision complexity by decomposing a central goal into a hierarchy of criteria and alternatives (cf. Figure 1). Decision makers conduct pairwise comparisons to evaluate the relative importance of criteria and preferences among alternatives. From these judgments, AHP derives explicit priority weights, checks their consistency (i.e. ratio < 0.1), and synthesizes them into a final decision. This structured workflow makes AHP one of the most influential and widely applied methods in domains ranging from public policy (Ho, 2008), software selection (Labib, 2011), and healthcare (Badri, 2001). Its core strength lies in providing a transparent and auditable reasoning process.

![](images/df8523ab44f483c0b1812fed1b76377e2aa11bbd303c24e0238bf5aca6118ed4.jpg)  
Figure 1: Flowchart of MCDM and AHP. See Appendix A for theoretical foundations.

Recent research has sought to improve the scalability and robustness of AHP, including fuzzy AHP (Soltanpanah et al., 2018; Khosla and Singh, 2021) which extends the traditional framework to handle uncertainty in expert judgments, and trace-based approaches leveraging behavioral data to automate criteria weighting (Ho et al., 2014). However, these methods often depend on predefined knowledge bases or manual inputs, which limits their flexibility in real scenarios. In this paper, we explore to what extent LLM can be an alternative to automate the entire AHP reasoning process.

While the use of LLM in decision-support tools has been largely explored in the literature (Eigner and Händler, 2024; Sha et al., 2023; Papasotiriou et al., 2024; Sun et al., 2025), AHP-based approaches received less attention. Among the few attempts, existing work share three main limitations: (1) AHP prompting adopt simple prompting strategies, yet their findings remain inconclusive, some report strong performance from LLMs (Lu et al., 2024a; Svoboda and Lande, 2024), while others highlight inconsistent or unreliable results (Doe and Liu, 2023; Stelmach, 2025). This disparity suggests that conventional static prompting alone may fail to capture the complex reasoning AHP requires. (2) Comparisons between LLM and human experts are often constrained by the absence of standardized, publicly available datasets. Furthermore, many studies omit systematic consistency verification, making it difficult to evaluate the reliability and reproducibility of their findings. (3) Current evaluation practices also tend to overlook the intermediate reasoning steps within the AHP process. Except for Park et al. (2025), who manually analyzed these steps, most prior studies focus solely on final outcomes, without examining how decisions are formed throughout the pipeline. Our work addresses these gaps by proposing:

– LEGAL-AHP, a new expert-annotated benchmarkfollowing AHP. We target real-world reasoning in the legal domain, providing a basis for future research in the field (cf. Section 3).<sup>1</sup>

– The first end-to-end approach that operationalizes the entire AHP workflow with LLMs. We propose a set of strategies enabling interpretable decision-making through explicit AHP instructions, a single-agent AHP framework and a multi-agent architecture (cf. Section 4).

– A quantitative and qualitative evaluation of these strategies exploring both the internal stages of the pipeline and the final decisions, ensuring methodological robustness and transparency throughout (cf. Section 5). Our findings demonstrate improved alignment with expert judgment and more trustworthy explanations.

– An evaluation of the portability of our approach beyond the legal domain, targeting ranking tasks. Our results show that LLM-based AHP framework is applicable to various decision problems.

## 2 Related Work

## 2.1 LLMs as a Tool for Decision-Making

LLMs are increasingly explored as decisionsupport tools across domains. Eigner and Händler (2024) review key determinants of LLM-assisted decision-making, such as transparency, task complexity, and user trust. Beyond support, LLMs are also studied as autonomous decision-makers: for example, LanguageMPC (Sha et al., 2023) uses an LLM to reason about driving scenarios and translate them into control actions via model predictive control. At a broader scale, Sun et al. (2025) survey LLM-based multi-agent decision-making, highlighting challenges of coordination, communication, and human oversight. Complementing these efforts, Lu et al. (2024b) propose STRUX, a framework that improves transparency by extracting favorable and adverse evidence, weighting them, and generating structured explanations in high-stakes tasks like financial forecasting.

However, their application to MCDM remains limited: while capable of producing plausible outputs, LLMs lack the structured logic of established methods like AHP, and their reasoning processes are opaque and difficult to verify. As a result, their outputs often cannot meet the reliability and interpretability demands of multi-criteria settings.

Explainable AI (XAI) methods have attempted to mitigate this “black-box” problem, for example through saliency maps and post-hoc rationalizations (Ribeiro et al., 2016; Shapley, 1953). Yet these methods typically provide feature-level explanations and fail to capture how models weigh multiple criteria to reach a final outcome. In MCDM contexts, where process transparency and traceability are paramount, such explanations remain insufficient, especially in high-stakes environments where errors can have serious consequences.

## 2.2 LLMs-based AHP Decision-Making

Within this broader trend, researchers have explored integrating LLMs with structured decisionscience methods such as AHP. Lu et al. (2024a) introduced “AHP-Powered LLM Reasoning,” where the AHP hierarchy guides prompt construction to make evaluations of open-ended student responses more systematic. Results show that multiple criteria performs better compared to pairwise comparison without criteria.

Doe and Liu (2023) tested ChatGPT as a multicriteria decision-maker and found that although outputs appeared plausible, they often violated AHP’s consistency requirements. Similarly, Svoboda and Lande (2024) integrated GPT-4 with AHP for cybersecurity decision support, employing GPT-4 based “virtual experts” to perform pairwise comparisons and aggregate judgments, albeit only at partial workflow levels.

Most recently, Park et al. (2025) enhanced AHP modelling under uncertainty by fine-tuning LLMs on domain-specific documents to generate complete hierarchies and comparison matrices, showing alignment with expert judgments even under incomplete specifications.

In this paper, we continue these efforts by exploring this time LLMs reasoning capabilities in mimicking the full AHP pipeline while providing to the community the first benchmark dataset manually annotated following AHP in the legal domain.

## 3 The LEGAL-AHP Dataset

There is currently a lack of datasets specifically designed to evaluate MCDM problems, leading most existing studies to rely on multi-attribute datasets such as Cheng et al. (2024). For AHP, as no prior free dataset exists, we propose to leverage on available question answering (QA) benchmarks that we augment with expert annotations following an AHP-based annotation scheme enabling a direct comparison between human judgments and the outputs produced by LLMs. To this end, we rely on the LEGALBENCH benchmark (Holzenberger et al., 2023) which provides a diverse set of carefully curated problems that closely reflect real-world legal reasoning. Its domain specificity and structured tasks make it a natural choice for evaluating complex decision-making in legal contexts.

## 3.1 Dataset Selection

LEGALBENCH is a collaboratively constructed benchmark, comprising 162 subsets or tasks that span six distinct categories of legal reasoning in English. Following a thorough inspection, we sample five subsets, each containing series of QA for a total of 525 pairs: 45 from Abercrombie (9 series of 5 QA), 135 Judicial\_Ethics (15 series of 9 QA), 90 Common\_Law (15 series of 6 QA), 120 Decision\_Section (15 series of 8 QA), and 135 Privacy\_Policy (15 series of 9 QA). These subsets are selected for two main reasons: (1) They cover diverse areas of law, ensuring a variety of legal contexts; and (2) They feature distinct decisionmaking formats (e.g., binary Yes/No judgments and multiple-choice style questions), allowing us to assess the decision-making performance across different task structures. A brief description of each subset is provided in Appendix B.1.

## 3.2 Annotation Procedure

The selected subsets were reframed as decisionmaking problems, with multiple-choice options serving as the alternatives. For each subset, ten independent groups of law Master students with relevant expertise were invited to provide annotations. Each group, composed of 2 members, was randomly given two subsets to enable stronger crossvalidation.

We designed an AHP annotation guidelines where for each QA pair, annotators defined a list of 3 to 7 criteria along with their definitions, constructed pairwise comparison matrices, and computed the weights for each criterion based on these matrices. See Appendix B.3 for a detailed description of the guidelines together with a complete example.

A total of 525 questions were annotated in two steps. First a training phase where for each subset, groups were trained on the first questions of each series for a total of 450 QA (see Appendix B.2 for detail). Given the complexity of the task, the training step was very deep guided by an expert in MCDM. Then students were asked to annotate the last questions of each subset (series) separately, which corresponds to a total of 75 pairs. Within a group, annotations are made by consensus among its members.

## 3.3 Qualitative Evaluation

Consistency check. We conducted a consistency check to evaluate the reliability of the criteria and weights provided by the annotators (see Appendix A, Consistency section). A group is retained only when the consistency ratio associated to its annotations satisfied CR < 0.1 or the list of the annotated criteria is greater than two (since pairwise comparisons and the corresponding CR could not be computed). Among the 20 annotations from the groups, 12 were removed due to invalid or inconsistent criteria ensuring that only coherent and reliable annotations were included in the analysis. Table 2 reports the CR values for the remaining groups across subsets with CR < 0.1, which corresponds to a total of 60 reliable QA pairs in the final Legal-AHP benchmark.

Agreements among criteria. In addition, we manually identified the criteria that were common across groups under the same subset to arrive at a gold list. To this end, we rely on the definitions provided during annotations and performed a semantic analysis by grouping similar criteria. When the definitions were not aligned, a generic criteria was found. For example, for the subset Judicial\_Ethics, one criteria was described as follows by different groups: transparency, professional neutrality, conflict of interest. The last one was chosen to encompass all the others. The final criteria list is summarized in Table 1.

<table><tr><td rowspan=1 colspan=1>Criteria</td><td rowspan=1 colspan=1>Abercrombie</td><td rowspan=1 colspan=1>Common_Law</td><td rowspan=1 colspan=1>Decision_Section</td><td rowspan=1 colspan=1>Judicial_Ethics</td><td rowspan=1 colspan=1>Privacy_Policy</td></tr><tr><td rowspan=1 colspan=1>Criterion 1</td><td rowspan=1 colspan=1>Type product/service</td><td rowspan=1 colspan=1>Tangibility</td><td rowspan=1 colspan=1>References  topassed cases</td><td rowspan=1 colspan=1>Impartiality</td><td rowspan=1 colspan=1>Topic precision</td></tr><tr><td rowspan=1 colspan=1>Criterion 2</td><td rowspan=1 colspan=1>Link productmark</td><td rowspan=1 colspan=1>Services</td><td rowspan=1 colspan=1>Affirmation</td><td rowspan=1 colspan=1>Compliancewith law</td><td rowspan=1 colspan=1>Lexical  scopeconsistency</td></tr><tr><td rowspan=1 colspan=1>Criterion 3</td><td rowspan=1 colspan=1>Originality</td><td rowspan=1 colspan=1>Real estate</td><td rowspan=1 colspan=1>Law references</td><td rowspan=1 colspan=1>Conflict of inter-est</td><td rowspan=1 colspan=1></td></tr></table>

Table 1: Gold criteria per subset in LEGAL-AHP.

<table><tr><td>Subset</td><td>Groups</td><td>CR&lt; 0.1</td></tr><tr><td>Abercrombie</td><td>Group_1 0.0439 Group_2 0.0432 Group_3 0.0473</td><td rowspan="3"></td></tr><tr><td>Judicial_Ethics</td><td>Group_4 0.0348 Group_6 0.0465</td></tr><tr><td>Common_Law</td><td>Group_2 0.0332 Group_3 0.0334</td></tr><tr><td></td><td></td><td rowspan="2"></td></tr><tr><td></td><td>Decision_Section Group_4 0.0692</td></tr></table>

Table 2: Consistency Ratios (CR) of annotation groups across subsets.

Agreements on final decisions. After verifying the consistency ratio, we further evaluated the intergroup agreement for each subset based on the final annotation decisions. For each QA item, we measured the Agreement Ratio (AR), defined as the proportion of groups selecting the same (majorityvoted) label. Most QA items showed high agreement $( \mathrm { A R } > = 0 . 6 )$ , indicating strong consensus across groups, while only a few exhibited moderate variation (AR < 0.5). This provides a concrete measure of consensus across annotator groups, complementing the internal consistency check.

We additionally compared the inter-group agreement ratios with those derived from the LegalBench gold standard. As shown in Table 3, the average agreement reached 80%, demonstrating strong alignment between group-level judgments and the gold-standard annotations. The Decision section, however, exhibited a lower agreement ratio, likely because its multiple-choice format made it more difficult for annotators to reach consensus.

<table><tr><td>Subset</td><td>Agreement Ratio (%)</td></tr><tr><td>Abercrombie</td><td>86.7</td></tr><tr><td>Judicial_Ethics</td><td>93.3</td></tr><tr><td>Common_Law</td><td>93.3</td></tr><tr><td>Decision_Section</td><td>60.0</td></tr></table>

Table 3: Agreement ratios between gold LEGALBENCH and gold LEGAL-AHP.

## 4 Methodology

## 4.1 AHP-based Strategies

Let a decision task be defined by a natural language question and a set of candidate alternatives $\mathcal { A } = \{ a _ { 1 } , a _ { 2 } , . . . , a _ { n } \}$ . The goal is to identify the best alternative $a ^ { * }$ using a set of the criteria ${ \mathcal { C } } = \{ c _ { 1 } , c _ { 2 } , \ldots , c _ { m } \}$ and associated importance weights $\mathbf { w } = [ w _ { 1 } , \dots , w _ { m } ] ;$

$$
a ^ { * } = \arg \operatorname* { m a x } _ { a _ { i } \in \mathcal { A } } \sum _ { j = 1 } ^ { m } w _ { j } \cdot s ( a _ { i } , c _ { j } )
$$

where $s ( a _ { i } , c _ { j } )$ is the score of alternative $a _ { i }$ with respect to criterion $c _ { j }$ . We designed three methods to arrive at the decision $a ^ { * }$ . Their corresponding prompts are illustrated in Appendix E.

(1) AHP Instruction. Unlike prior work that employed only partial elements of AHP such as hierarchical structuring of prompts (Lu et al., 2024a) or localized pairwise comparisons within specific decision stages (Svoboda and Lande, 2024; Doe and Liu, 2023) our instruction set requires the LLM to execute the entire AHP decision-making process. The model is guided through defining the problem, identifying key criteria, estimating relative importance via structured pairwise comparisons, and synthesizing results into a transparent and interpretable ranking of alternatives, as shown in Table 16.

(2) Single-Agent AHP Framework. We further refine the AHP instruction by implementing it as a structured three-step pipeline, executed through multiple turns of interaction, as shown in Table 17. Specifically, (i) the model first generates a set of key decision criteria from the given question and alternatives; (ii) it then performs pairwise comparisons to derive relative weights for these criteria; and (iii) finally, it computes and ranks the alternatives to identify the optimal choice.

Each stage is conditioned on the outputs of the previous one, maintaining a continuous dialogue history that captures the evolving reasoning process. During weighting and aggregation, the assistant explicitly refers back to its earlier responses, ensuring internal consistency and interpretability of the final recommendation. This multi-turn prompting design enhances the transparency andexplainability of the model’s decision-making compared to the single-turn AHP instruction.

(3) Multi-Agent AHP Framework. We propose a multi-agent AHP framework that enables structured, interpretable, and context-aware decisionmaking with LLMs. As illustrated in Figure 2 (Appendix A) and the corresponding prompts in Tables 18 and 19 (Appendix E), the framework consists of two key components, which collaboratively decompose the AHP workflow and simulate expert-level reasoning.

(a) Information Block. It summarizes contextual relations between the decision problem and alternatives. The Retrieval/Summary agent generates a structured summary I that contextualizes how each alternative connects to the decision intent:

$$
I = \mathsf { S u m m a r i z e C o n t e x t } ( \mathbf { Q } , \mathcal { A } )
$$

Rather than retrieving external sources, this agent performs in-situ inference using only the provided inputs, enabling robust performance even in constrained or sensitive domains.

(b) AHP Block. Through pairwise comparisons, it derives priority weights for criteria and alternatives, and aggregates them into a final ranking (see Appendix A for details). It is implemented as a multi-agent module composed of three specialized agents:

– Agent 1: Criteria Generation. Given the decision question, alternatives, and the summary I produced by the Information Block, this agent extracts a set of the criteria C that reflect the dimensions relevant for comparing the alternatives:

$$
\mathcal { C } = \mathsf { G e n e r a t e C r i t e r i a } ( \mathsf { Q } , \mathcal { A } , I )
$$

Each criterion $c _ { j }$ is phrased in natural language (e.g., “distinctiveness”, “functionality”) and de-

signed to support transparent downstream evaluation.

– Agent 2: Pairwise Weights. To assess the relative importance of each criterion, this agent constructs a pairwise comparison matrix $M \in \mathbb { R } ^ { m \times m }$

$$
M _ { i j } = \mathsf { L L M C o m p a r e } ( c _ { i } , c _ { j } \mid \mathbf { Q } , \boldsymbol { \mathcal { A } } , I )
$$

$$
w _ { j } = \frac { 1 } { m } \sum _ { i = 1 } ^ { m } \frac { M _ { i j } } { \sum _ { k = 1 } ^ { m } M _ { i k } }
$$

– Agent 3: Decision Making. The agent computes the final score of each alternative using the weighted criteria:

$$
\mathsf { S c o r e } ( a _ { i } ) = \sum _ { j = 1 } ^ { m } w _ { j } \cdot s ( a _ { i } , c _ { j } )
$$

It selects the top-ranked alternative and produces a natural language explanation reflecting how each criterion contributed to the decision:

$$
a ^ { * } = \arg \operatorname* { m a x } _ { a _ { i } } { \mathsf { S c o r e } ( a _ { i } ) }
$$

Explain(a<sup>∗</sup>) = LLMExplain(a<sup>∗</sup>, C, w, Q)

In this third strategy, agents can either be:

• Manually constructed (Multi-Agent (Manual)) where the roles, responsibilities, and prompt templates of each agent are explicitly defined based on AHP to execute key stages of the AHP workflow, including information organization, criterion generation, weight estimation, and decision aggregation.

• Automatically generated (Multi-Agent (Auto)). Here the agent architecture is no longer statically specified. Instead, a centrally coordinated meta-agent dynamically decomposes the decision task and automatically instantiates a set of specialized agents according to the structure and complexity of the problem. Each generated agent is assigned a task-specific role, enabling collaborative execution of the AHP reasoning process. We used the CaptainAgent framework (Song et al., 2025) to implement this second setting.

Unlike the manually constructed agents, generated agents are created in a model-driven manner, allowing adaptive organization and coordination during inference. Despite this difference in agent generation, both settings share the same decision objective, criterion weighting scheme, and aggregation formulation, ensuring a fair and controlled comparison between the two settings.

## 4.2 Experimental Settings

We evaluate our framework on four representative LLMs, including three open-source models, Llama-3-8B-Instruct (Grattafiori et al., 2024), Qwen2.5-7B-Instruct (Cloud, 2024), and Mistral-7B-Instruct-v0.2 (Jiang and et al., 2023),as well as the proprietary GPT-4o-mini (OpenAI et al., 2023).<sup>2</sup>

Each model is prompted with one of our AHPbased strategies and compared against a Non-AHP baseline using standard few-shot prompting from LegalBench. All strategies are provided with identical decision inputs and the same set of 5–8 few-shot examples to ensure fair comparison.

For the automatically generated multi-agent setting, we focus on the strongest open-source model (Qwen2.5-7B-Instruct) and the proprietary GPT-4o-mini to further investigate the effectiveness of AHP in a collaborative setting.

All models are evaluated with identical prompts and generation parameters. We fix the temperature to 0.7 and report accuracy averaged over three runs. Additional results, including macro-F1 scores, are reported in Appendix 11.

## 5 Results

We conduct experiments to answer four research questions in the following subsections respectively: (RQ1) How does Non-AHP and AHP-based strategies perform when compared to the original LEGALBENCH where legal problems are framed as a QA task? (RQ2) How effective and faithful are the intermediate decision criteria generated by our AHP prompting when compared to LEGAL-AHP gold criteria? (RQ3) How does the AHP strategies output compare with humanfinal decisions as given by LEGAL-AHP? (RQ4) Do AHP-based strategies generalize to other decision problems, demonstrating portability beyond the legal domain?

## 5.1 LLMs vs. LEGALBENCH Decisions

Non-AHP vs. AHP-based Methods. As shown in Table 4, the Non-AHP baseline provides strong and stable performance across models, with overall accuracies ranging from 68.3% to 73.3%. Among AHP-based methods, performance varies substantially by model. Notably, GPT-4o-mini consistently benefits from AHP prompting: AHP-Instruction improves accuracy from 73.3% to

78.3%, and the automatically generated Multi-Agent setting further raises performance to 81.7%, the best result observed. In contrast, Mistral and LLAMA3 experience notable performance degradation under AHP-Instruction and Single-Agent prompting, indicating limited robustness to structured decision prompts. Qwen2.5 remains comparatively stable, maintaining near-baseline accuracy under Single-Agent and Multi-Agent settings.

Comparison within AHP-based Methods. Table 11 illustrates the relative behavior of different AHP-based variants across models. AHP-Instruction shows the highest potential gains but primarily benefits stronger models, most notably GPT-4o-mini. Single-Agent prompting exhibits mixed and model-dependent effects, often degrading weaker models while remaining viable for Qwen2.5 and GPT-4o-mini. Multi-Agent prompting further amplifies this divergence: the manually constructed variant yields limited or inconsistent improvements, whereas the automatically generated Multi-Agent configuration provides the most stable and consistent gains for GPT-4o-mini. Overall, these results indicate that the effectiveness of AHP-based methods depends critically on both model capacity and the level of agent coordination.

## 5.2 Evaluation of Generated Decision Criteria

Motivated by the idea that structured processes such as AHP can guide models toward more meaningful and transparent dimensions, (RQ2) examines whether decision criteria generated by different methods align with human reasoning.

As references, we summarize the representative criteria identified by annotators for each subset (Table 1). Using an LLM-as-a-judge strategy (Qwen3- 30B-A3B), the LLM extracts criteria from each method’s output, compares them with the human set, and selects the closest match. Table 5 shows how often each method is selected as the bestaligned option across different subsets and models.

The Single-Agent stands out as the most frequently chosen approach (e.g., 47 for Mistral, 53 for LLaMA3, 40 for GPT-4o-mini), suggesting that its broad coverage makes it more likely to capture human-relevant dimensions. In contrast, the AHP Instruction is selected less often (e.g., 13 for GPT-4o-mini). This indicates that in one-shot settings models find it difficult to follow multi-step AHP instructions consistently, leading to less stability in criteria generation, even though they tend to achieve higher accuracy than other methods. The AHP Multi-Agent shows competitive results, particularly with Qwen2.5 (38), highlighting that dividing roles among agents can improve alignment when the base model is capable of effective coordination.

<table><tr><td>Dataset</td><td>Baseline</td><td>Mistral</td><td>LLAMA3</td><td>Qwen2.5-7B</td><td>GPT-40-mini</td></tr><tr><td rowspan="5">Overall</td><td>Non-AHP</td><td>68.3%</td><td>68.3%</td><td>71.7%</td><td>73.3%</td></tr><tr><td>AHP-Instruction</td><td>66.7%</td><td>53.3%</td><td>63.3%</td><td>78.3%</td></tr><tr><td>Single-Agent</td><td>43.3%</td><td>43.3%</td><td>66.7%</td><td>70.0%</td></tr><tr><td>Multi-Agent (Manual)</td><td>36.7%</td><td>66.7%</td><td>68.3%</td><td>61.7%</td></tr><tr><td>Multi-Agent (Auto)</td><td>一</td><td>一</td><td>65.0%</td><td>81.7%</td></tr></table>

Table 4: Overall model accuracy when evaluated on LEGAL-AHP (RQ1). The best score is shown in bold. Detailed scores per subset are presented in Table 11.
<table><tr><td>Methods</td><td>Mistral</td><td>LLAMA3</td><td>Qwen2.5</td><td>GPT-4o-mini</td></tr><tr><td>AHP-Instruction</td><td>9</td><td>8</td><td>5</td><td>13</td></tr><tr><td>Single-Agent</td><td>47</td><td>53</td><td>32</td><td>40</td></tr><tr><td>Multi-Agent (Manual)</td><td>19</td><td>14</td><td>38</td><td>22</td></tr></table>

Table 5: Number of instances (out of 75) where each method’s generated decision criteria were judged as most aligned with human reasoning (RQ2).

## 5.3 LLMs vs. LEGAL-AHP Decisions

As discussed in (RQ2), Single-Agent prompting produces decision criteria that are more closely aligned with human annotations, whereas AHP-Instruction can be less stable when following complex multi-step instructions. Motivated by this observation, we further evaluate how these prompting strategies translate into final decision accuracy when compared against human-annotated LEGAL-AHP labels. The results are summarized in Table 6.

Compared with the automatic evaluation in RQ1, several consistent trends emerge. GPT-4o-mini achieves the strongest overall performance under AHP-Instruction, reaching 80.0% accuracy and improving over its automatic evaluation result (76.0%), which indicates closer alignment with human judgments. Qwen2.5 exhibits relatively stable performance across prompting strategies. In contrast, Mistral and LLAMA3 show limited performance gains under all evaluated settings. Based on these observations, we do not further evaluate the Multi-Agent (Auto) framework on Mistral and LLAMA3, and instead focus on Qwen2.5 and GPT-4o-mini as representative open-source and proprietary models, respectively, for the multi-agent experiments.

Overall, the comparison between (RQ1) and (RQ3) suggests that although Single-Agent prompting aligns better with human reasoning at the criteria level (RQ2), AHP-based strategies yield more stable improvements in final decision accuracy, particularly for stronger models.

## 5.4 Portability to Other Decision Problems

To examine whether AHP-based strategies generalize beyond the legal domain, we evaluate our framework on two real-world ranking tasks: the QS World University Rankings and the U.S. News Best Global Universities Rankings. For each benchmark, we restrict the candidate set to the top-10 universities from the official rankings and ask the model to re-rank them under identical conditions.

Given this fixed candidate pool, we compare a zero-shot baseline (Non-AHP), which directly outputs a ranking, with an AHP-based strategy. Motivated by its strong performance in the legal domain, we adopt the automatically generated multi-agent AHP framework (Multi-Agent Auto), where the model explicitly generates decision criteria, assigns relative importance via pairwise comparisons, and aggregates weighted scores to produce the final ranking.

Tables 12 and 13 in Appendix D report Kendall’s Tau between model-generated rankings and the official reference rankings. On the QS benchmark, Multi-Agent (Auto) improves ranking correlation for GPT-4o-mini, while Non-AHP remains slightly stronger for Qwen2.5-7B-Instruct. On the U.S. News benchmark, Multi-Agent (Auto) yields a substantial improvement for Qwen2.5-7B-Instruct, whereas GPT-4o-mini performs better under the

<table><tr><td>Method</td><td>Mistral</td><td>LLAMA3</td><td>Qwen2.5</td><td>GPT-40-mini</td></tr><tr><td>AHP-Instruction</td><td>61.3%</td><td>48.0%</td><td>69.3%</td><td>80.0%</td></tr><tr><td>Single-Agent</td><td>38.7%</td><td>47.0%</td><td>64.0%</td><td>61.0%</td></tr><tr><td>Multi-Agent (Auto)</td><td></td><td>一</td><td>65.3%</td><td>69.3%</td></tr></table>

Table 6: Accuracy of different models evaluated against human-annotated LEGAL-AHP decisions (RQ3).

Non-AHP baseline. We further analyze whether the generated criteria align with the ones used by the official ranking. Tables 14 and 15 show that AHP-based rankings produce more stable top-tier ordering, particularly among highly ranked universities, whereas zero-shot rankings exhibit larger fluctuations. These results suggest that structured AHP reasoning improves robustness and generalization in real-world ranking tasks. Overall, these results indicate that AHP-based reasoning exhibits meaningful portability to non-legal ranking tasks, enabling LLMs to perform structured multi-criteria decision-making beyond the legal domain.

## 6 Discussion

The integration of structured decision-making frameworks into LLM prompting substantially improves interpretability and alignment with expert judgments. However, these benefits are not uniform across models. Stronger LLMs, such as GPT-4omini and Qwen2.5, are better able to internalize and exploit explicit reasoning scaffolds, whereas weaker models struggle to reliably follow multistep analytical procedures. These observations suggest that the effectiveness of structured prompting is closely tied to the underlying reasoning capacity and instruction-following stability of the model.

An important finding is that LLMs can partially automate the Analytic Hierarchy Process. Across experimental settings, models are capable of generating meaningful intermediate criteria and producing interpretable final decisions, but their reliability decreases as the depth and coordination requirements of reasoning increase. Different AHP realizations exhibit complementary strengths: singleagent formulations align more closely with human reasoning at the criterion level, while AHP-guided aggregation yields more stable improvements in final decision accuracy. Together, these results indicate that effective automation depends not only on prompt design, but also on how reasoning steps are decomposed, ordered, and coordinated within the decision pipeline.

Beyond the legal domain, the proposed framework demonstrates meaningful generalization. As shown in (RQ4), the same AHP-based reasoning pipeline transfers effectively to non-legal decision problems, such as real-world ranking tasks, without relying on domain-specific knowledge. Such portability is enabled by the modular structure of the framework, which decouples criterion generation, weighting, and aggregation, allowing the same reasoning structure to be reused across domains.

Finally, systematic differences emerge across task formats. Binary decision problems are generally easier for LLMs, whereas multiple-choice and comparative settings benefit most from explicit structure and multi-criteria aggregation. Taken together, these findings suggest that while structured decision frameworks can significantly enhance transparency and reliability, achieving fully robust and interpretable reasoning automation in complex, high-stakes, and cross-domain settings remains an open challenge.

## 7 Conclusion

This paper proposes an agent-based framework that integrates the Analytic Hierarchy Process (AHP) with large language models to support transparent and interpretable multi-criteria decision-making. By structuring reasoning into criterion generation, weighting, and aggregation, the framework transforms opaque model outputs into auditable decision processes. Experiments on the LEGAL-AHP benchmark demonstrate that AHP-based prompting enables LLMs to better reflect expert decision logic at the level of intermediate reasoning. This is particularity salient for the multi-agent strategy where the resulting decisions are substantially more interpretable and more closely aligned with human analytical reasoning. We further show that the framework generalizes beyond the legal domain to real-world ranking tasks without relying on domain-specific knowledge.

Future work will explore improved agent coordination, symbolic consistency verification, and extensions to broader high-stakes domains where transparency and accountability are critical.

## Ethical Considerations

The data used to build LEGAL-AHP are composed of question–answer pairs taken from publicly available datasets accessible to the research community, which do not contain any abusive content or privacy issues.

Regarding the annotation campaign, the annotators were Master’s students in law, and their participation was part of their academic training. The dataset will be anonymized prior to its release to ensure privacy protection.

It is important to note that decision-making in the legal domain is highly context dependent. The same law or decree may lead to different judgments depending on the specific circumstances of each case. In such situations, even a decision tree cannot capture the underlying reasoning, making full automation of legal reasoning impractical. Therefore, our aim is not to develop large language models that replace legal experts in complex settings, but rather to assist them as tools that help reduce cognitive load, improve analytical efficiency, and support fair and transparent decision-making.

## Limitations

This study has several limitations.First, the effectiveness of AHP-based strategies depends on the reasoning capability of the underlying language models and the stability of multi-agent coordination. As observed in our experiments, models with stronger instruction-following and reasoning abilities benefit more consistently from structured AHP prompting, whereas weaker models may struggle with multi-step or multi-agent reasoning. Future work should investigate adaptive coordination and communication mechanisms to improve robustness across different model capacities.

Second, the annotation process relies on expertdefined criteria and pairwise comparisons, which inevitably introduce a degree of subjectivity. Although consistency checks were applied to filter unreliable annotations, differences in annotators expertise and interpretation can still affect the resulting criteria and weights. Exploring hybrid human–model annotation pipelines or partially automated criterion generation may help mitigate this limitation while preserving annotation quality.

Finally, the LEGAL-AHP dataset is relatively small due to the complexity of AHP-based annotation. Constructing the dataset required multiple expert annotators to define consistent criteria, perform pairwise comparisons, and provide detailed reasoning justifications. This process is inherently time-consuming and cognitively demanding, limiting scalability but also making the dataset a highquality benchmark for studying structured decisionmaking and interpretability.

## References

Masood A Badri. 2001. A combined ahp-gp model for quality control systems. Omega, 29(5):387–406.

Jean-Pierre Brans and Yves De Smet. 2005. Promethee methods. In Multiple criteria decision analysis: state of the art surveys, pages 187–219. Springer.

Daixuan Cheng, Shaohan Huang, and Furu Wei. 2024. Adapting large language models via reading comprehension. In The Twelfth International Conference on Learning Representations.

Alibaba Cloud. 2024. Qwen2.5 technical report. ArXiv preprint arXiv:2412.15115.

Jane Doe and Han Liu. 2023. Can chatgpt serve as a multi-criteria decision maker? Proceedings of the 2023 Conference on Decision Science.

Eva Eigner and Thorsten Händler. 2024. Determinants of llm-assisted decision-making. arXiv preprint arXiv:2402.17385.

Ernest H Forman and Saul I Gass. 2001. The analytic hierarchy process—an exposition. Operations Research, 49(4):469–486.

Kannan Govindan and Martin Brandt Jepsen. 2016. Electre: A comprehensive literature review on methodologies and applications. European Journal ofOperational Research, 250(1):1–29.

Aaron Grattafiori, Abhimanyu Dubey, and et al. 2024. The llama 3 herd of models. arXiv preprint arXiv:2407.21783.

Hoang Nam Ho, Mourad Rabah, Samuel Nowakowski, and Pascal Estraillier. 2014. A process for tracebased criteria weighting in multiple criteria decision making. Journal of Software.

William Ho. 2008. An integrated analytic hierarchy process and its applications to service selection. European Journal of Operational Research, 186(1):211– 228.

Nils Holzenberger, Cheng Zheng, Tanmay Guha, and 1 others. 2023. Legalbench: A collaboratively built benchmark for measuring legal reasoning in large language models. arXiv preprint arXiv:2308.11462.

Albert Q. Jiang and et al. 2023. Mistral 7b. arXiv preprint arXiv:2310.06825.

Ankit Khosla and Ram Kumar Singh. 2021. A fuzzy ahp approach for prioritizing indicators of sustainable manufacturing in the indian context. Journal of Cleaner Production, 281:124715.

Ashraf W Labib. 2011. A decision analysis model for maintenance policy selection using ahp. International Journal ofProduction Economics, 132(2):174– 182.

Xiaotian Lu, Jiyi Li, Koh Takeuchi, and Hisashi Kashima. 2024a. Ahp-powered llm reasoning for multi-criteria evaluation of open-ended responses. Findings ofEMNLP.

Yiming Lu, Yebowen Hu, Hassan Foroosh, Wei Jin, and Fei Liu. 2024b. Strux: An llm for decisionmaking with structured explanations. arXiv preprint arXiv:2410.12583.

Nolberto Munier, Eloy Hontoria, and 1 others. 2021. Uses and limitations of the ahp method.

OpenAI and 1 others. 2023. Gpt-4 technical report. arXiv preprint arXiv:2303.08774.

Serafim Opricovic and Gwo-Hshiung Tzeng. 2007. Extended vikor method in comparison with outranking methods. European journal ofoperational research, 178(2):514–529.

Kassiani Papasotiriou, Srijan Sood, Shayleen Reynolds, and Tucker Balch. 2024. Ai in investment analysis: Llms for equity stock ratings. In Proceedings of the 5th ACM International Conference on AI in Finance, ICAIF ’24, page 419–427, New York, NY, USA. Association for Computing Machinery.

Jason Papathanasiou and Nikolaos Ploskas. 2018. Topsis. In Multiple criteria decision aid: Methods, examples and python implementations, pages 1–30. Springer.

Haeun Park, Hyunjoo Oh, Feng Gao, and Ohbyung Kwon. 2025. Enhancing analytic hierarchy process modelling under uncertainty with fine-tuning llm. Expert Systems, 42(6):e70051.

Marco Tulio Ribeiro, Sameer Singh, and Carlos Guestrin. 2016. why should i trust you?¨ ¨: Explaining the predictions of any classifier. In Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pages 1135–1144.

Thomas L Saaty. 1980. The Analytic Hierarchy Process. McGraw-Hill.

Thomas L. Saaty. 2005. The analytic hierarchy and analytic network processes for the measurement of intangible criteria and for decision-making. In José Figueira, Salvatore Greco, and Matthias Ehrgott, editors, Multiple Criteria Decision Analysis: State of the Art Surveys, International Series in Operations Research & Management Science, pages 345–405. Springer, New York, NY.

Yicheng Sha, Yifan Zhang, Rui Wang, Yu Guo, Yue Zhang, and Xiaodong He. 2023. Languagempc: Large language models as decision makers for autonomous driving. arXiv preprint arXiv:2310.03026.

Lloyd S Shapley. 1953. A value for n-person games. Contributions to the Theory of Games, 2:307–317.

Herbert A Simon. 1960. The new science of management decision.

Hossein Soltanpanah, Morteza Yazdani, and Ghulam Kabir. 2018. Application of ahp and fuzzy ahp for project selection under uncertainty: A case study. Technological and Economic Development of Economy, 24(2):674–691.

Linxin Song, Jiale Liu, Jieyu Zhang, Shaokun Zhang, Ao Luo, Shijian Wang, Qingyun Wu, and Chi Wang. 2025. Adaptive in-conversation team building for language model agents. Preprint, arXiv:2405.19425.

Alexander Stelmach. 2025. Evaluating ai judgements: A case study of llms in product development.

Chuanneng Sun, Songjun Huang, and Dario Pompili. 2025. Llm-based multi-agent decision-making: Challenges and future directions. IEEE Robotics and Automation Letters.

Igor Svoboda and Dmytro Lande. 2024. Enhancing multi-criteria decision analysis with ai: Integrating analytic hierarchy process and gpt-4. arXiv preprint arXiv:2402.07404.

## A Analytic Hierarchy Process (AHP)

According to its founder (Saaty, 1980), the Analytic Hierarchy Process (AHP) is built on three key principles: (i) hierarchical structuring of the decision problem, (ii) priority derivation through pairwise comparisons, and (iii) logical consistency checks. In decision-support contexts such as student accommodation allocation (Saaty, 2005), AHP is applied to derive the weights of the criteria before applying other multi-criteria ranking methods.

## Hierarchical structure

The problem is decomposed into three levels: (i) the overall goal at the top, (ii) the criteria in the middle, and (iii) the alternatives at the bottom.

## Pairwise comparisons

The relative importance of each criterion is elicited using Saaty’s fundamental scale (Table 7). From these judgments, a comparison matrix $A = [ a _ { i j } ]$ is constructed, where

$$
a _ { i j } = \frac { w _ { i } } { w _ { j } } , \quad i , j = 1 , \dots , n ,
$$

and $w _ { i }$ denotes the weight of criterion i.

![](images/3facce034a136ae1a38e0cca91231f8f414a5b38976a2c088718d38ac365999b.jpg)

Figure 2: Pipeline of our multi-agent decision-making framework illustrated here on a legal classification task from the LEGAL-AHP subset aiming to determine the trademark category of the term $\mathbf { \ddot { \boldsymbol { s } } } \boldsymbol { a } l t \mathbf { \dot { \boldsymbol { \mathbf { \rho } } } }$
<table><tr><td>Intensity</td><td>Definition</td><td>Explanation</td></tr><tr><td>1</td><td>Equal importance</td><td>Neither alternative is preferred</td></tr><tr><td>3</td><td>Weak importance</td><td>One alternative is slightly preferred</td></tr><tr><td>5</td><td>Strong importance</td><td>Clear preference of one over the other</td></tr><tr><td>7</td><td>Very strong</td><td>Strongly favored alternative</td></tr><tr><td>9</td><td>Extreme</td><td>Extremely more important</td></tr><tr><td>2,4,6,8</td><td>Intermediate</td><td>Compromise values between the above</td></tr></table>

Table 7: Saaty’s fundamental scale (Saaty, 1980).

## Normalization and weight derivation

The comparison matrix is normalized to obtain a matrix $B = [ b _ { i j } ]$ :

$$
b _ { i j } = \frac { a _ { i j } } { \sum _ { j = 1 } ^ { n } a _ { i j } } .
$$

The weight of each criterion i is then computed as the row average:

$$
W _ { i } = \sum _ { j = 1 } ^ { n } b _ { i j } .
$$

## Consistency check

To ensure judgments are logically consistent, the following quantities are calculated:

$$
c _ { i } = \sum _ { j = 1 } ^ { n } a _ { i j } w _ { j } ,\tag{1}
$$

$$
\lambda _ { \operatorname* { m a x } } = \frac { 1 } { n } \sum _ { i = 1 } ^ { n } \frac { c _ { i } } { w _ { i } } ,\tag{2}
$$

$$
C I = { \frac { \lambda _ { \operatorname* { m a x } } - n } { n - 1 } } ,\tag{3}
$$

$$
C R = { \frac { C I } { R I } } ,\tag{4}
$$

where n is the number of criteria and RI is the random index (Table 8). A pairwise comparison matrix is considered consistent if $C R \leq 0 . 1$

## Manual Multi-Agent Construction

Figure 2 provides an overview of how the Analytic Hierarchy Process (AHP) is instantiated within our manually constructed multi-agent framework. The figure illustrates the end-to-end decision pipeline on a representative legal classification task from the LEGAL-AHP dataset, where the objective is to determine the trademark category of the term $" S a l t "$

As depicted, the framework is organized into two core components. The Information Block first processes the decision question and candidate alternatives to extract and summarize task-relevant contextual information. This structured context is then forwarded to the AHP Block, which decomposes the decision-making process into criterion generation, pairwise weighting, and final aggregation. Each stage is handled by a dedicated agent with a predefined role, enabling a clear separation of responsibilities across the workflow. gb Overall, the figure offers a high-level visualization of how information analysis and AHP-style structured reasoning are integrated in the manual multi-agent setting. It complements the detailed prompt templates and algorithmic formulations introduced in subsequent sections, and highlights the transparency and modularity of the proposed design.

<table><tr><td>n</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>RI</td><td>0</td><td>0</td><td>0.58</td><td>0.90</td><td>1.12</td><td>1.24</td><td>1.32</td><td>1.41</td><td>1.45</td><td>1.49</td></tr></table>

Table 8: Random Index Values (RI) for different n (Saaty, 1980).

## B Details of the Annotation Procedure

## B.1 Selected Subsets

We select five subsets from LEGALBENCH, each representing a distinct legal domain and decisionmaking format, as shown in Table 9:

• The Abercrombie task assesses trademark distinctiveness by classifying a mark as Generic, Descriptive, Suggestive, Arbitrary, or Fanciful.

• The Judicial\_Ethics task evaluates whether certain actions by New York State judges are ethically permissible, based on official advisory opinions.

• The Common\_Law task determines whether a given contract is governed by the Uniform Commercial Code (UCC) or by common law, depending on the contract’s subject matter.

• The Decision\_Section task classifies paragraphs from judicial decisions into one of seven functional roles: Facts, Procedural History, Issue, Rule, Analysis, Conclusion, or Decree.

• Finally, the Privacy\_Policy task asks whether a given privacy policy excerpt is relevant to answering a specific question, framed as a binary classification problem.

## B.2 Training Phase

We detail below how QA pairs in each subset were selected for the training phase:

• Abercrombie: Training on the 4 first questions of each series. Total number of training questions=32.

• Common\_Law: Training on the 5 first questions of each series. Total number of training questions=75.

• Decision\_Section: Training on the 7 firsAccording to its foundet questions of each series. Total number of training questions=105.

• Judicial\_Ethics: Training on the 8 first questions of each series. Total number of training questions=120.

• Privacy\_Policy: Training on the 8 first questions of each series. Total number of training questions=120.

## B.3 AHP-Based Annotation Guidelines

This appendix describes the procedure that annotators followed when applying the Analytic Hierarchy Process (AHP) to each question. In this setting, the goal is the question itself, the alternatives are the answer options, and the best alternative is the correct answer. The annotators carried out the following steps:

1. Identify the criteria: Define a set of the criteria that are relevant for determining the best answer. These criteria should capture different aspects of what makes an answer correct or appropriate (e.g., legal relevance, clarity, completeness).

2. Create pairwise comparison matrices: Construct pairwise comparison matrices to evaluate (a) the relative importance of the criteria and (b) the relative performance of the alternatives under each criterion. Use the standard AHP 1–9 scale, where 1 indicates equal importance and 9 indicates extreme importance. Use reciprocal values (e.g., 1/3, 1/5) when one item is less important than another.

3. Derive weights: Normalize each pairwise comparison matrix and average the rows to compute the priority weights. This produces (i) the weights for each criterion and (ii) the weights for each alternative under each criterion.

4. Evaluate alternatives: Combine the weights of criteria and alternatives to calculate an overall score for each alternative. The alternative with the highest overall score is considered the best answer to the question.

These steps were provided to all student annotators to ensure consistency and rigor in the annotation process.

<table><tr><td rowspan=1 colspan=1>Subtask</td><td rowspan=1 colspan=1>Example Question</td><td rowspan=1 colspan=1>Answer</td></tr><tr><td rowspan=1 colspan=1>Abercrombie</td><td rowspan=1 colspan=1>The mark &quot;Salt&quot; for packages of sodium chloride. What is the type of mark?</td><td rowspan=1 colspan=1>Generic</td></tr><tr><td rowspan=1 colspan=1>Judicial_Ethics</td><td rowspan=1 colspan=1>Is a judge required to disclose a former law clerk&#x27;s employment relationship when theformer law clerk appears in an adversarial role within one year after his/her employmentwith the judge ended?</td><td rowspan=1 colspan=1>Yes</td></tr><tr><td rowspan=1 colspan=1>Common_Law</td><td rowspan=1 colspan=1>Contract: Grace and Hudson form an agreement for the sale of a microwave. Is thiscontract governed by the UCC or the common law?</td><td rowspan=1 colspan=1>UCC</td></tr><tr><td rowspan=1 colspan=1>Decision_Section</td><td rowspan=1 colspan=1>On appeal, Khochinsky challenges the district court&#x27;s dismissal under the FSIA as wellas the court&#x27;s vacatur of the default.</td><td rowspan=1 colspan=1>Issue</td></tr><tr><td rowspan=1 colspan=1>Privacy_Policy</td><td rowspan=1 colspan=1>Clause: Personal information such as purchase history shall not be further processed orused for any commercial purposes.Question: is there a way to opt out of data sharing</td><td rowspan=1 colspan=1>Relevant</td></tr></table>

Table 9: Example question–answer pairs for the selected LEGALBENCH subsets.

## B.4 Illustrative Example

Table 10 illustrates an example of the AHP-based annotation process, as applied by a group of annotators to the subset Decision\_Section.

## C Detailed Results in Terms of F1-scores

Table 11 presents the detailed F1-scores across datasets and model configurations. Overall, the Non-AHP baseline remains the most stable across all models, with scores ranging from 62.9% to 70.1%. In contrast, AHP-Instruction and Single-Agent strategies show greater variance, often enhancing stronger models while degrading weaker ones. GPT-4o-mini achieves the best overall performance, reaching 78.8% under AHP and 74.5% with the Single-Agent setup, both exceeding its baseline of 69.5%. Qwen2.5 also performs consistently well, attaining perfect F1 (100.0%) on Common Law and high scores on other tasks. Conversely, Mistral and LLAMA3 exhibit instability under AHP (37.7% and 37.0%, respectively) but moderate recovery in Multi-Agent settings, such as LLAMA3’s 73.2% on Judicial Ethics. Datasetwise, GPT-4o-mini leads on Abercrombie (71.3%) and Judicial Ethics (86.6%), while both GPT-4omini and Qwen2.5 reach 100.0% on Common Law. On Decision Section, GPT-4o-mini achieves its highest F1 (76.2%) with the Single-Agent method, and Qwen2.5 peaks under Multi-Agent (73.3%). Overall, Non-AHP offers the most reliable baseline, whereas AHP and Single-Agent strategies can significantly boost advanced models like GPT-4omini and Qwen2.5 but tend to destabilize smaller models such as Mistral and LLAMA3.

## D Portability to Ranking Decision Problems

Tables 12 and 13 report Kendall’s Tau scores on the QS World University Rankings and the U.S. News

Best Global Universities Rankings. We evaluate both a zero-shot Non-AHP baseline and the proposed Auto Multi-Agent AHP framework under an identical top-10 candidate setting.

Overall, AHP-based structured reasoning leads to more consistent rankings when the underlying model can reliably follow multi-step instructions. On the QS benchmark, Auto Multi-Agent AHP improves performance for GPT-4o-mini, while showing comparable results to the Non-AHP baseline for Qwen2.5. On the U.S. News benchmark, the AHP framework substantially boosts performance for Qwen2.5, indicating that explicit criteria decomposition and aggregation can effectively reduce ranking noise for weaker models.

Position-level comparisons in Tables 14 and 15 further show that AHP-based rankings produce more stable top-tier ordering, particularly among highly ranked universities.

## E Prompts

Tables 16, 17, and 18 present the prompt templates used in our experiments. All prompts are constructed within the Analytic Hierarchy Process (AHP) framework, which structures decisionmaking into interpretable steps of problem definition, criteria formulation, pairwise comparison, and final aggregation.

AHP-Instruction. The AHP-Instruction prompt, shown in Table 16, defines a structured decisionmaking process based on the AHP. It guides the model to define the decision problem, identify 3–5 key criteria, perform pairwise comparisons using the AHP 1–9 scale with reciprocals for less important factors, and generate an interpretable ranking of alternatives.

Multi-Agent Setting. The multi-agent framework realizes the AHP reasoning pipeline by decomposing the decision process into specialized agents. In the manual setting (Table 18), the workflow is divided into four predefined roles: a Retrieval/Summary Agent for contextual analysis, a

<table><tr><td rowspan=1 colspan=1>Criteria</td><td rowspan=1 colspan=1>Consistency of lexical scope</td><td rowspan=1 colspan=1>Level of precision</td><td rowspan=1 colspan=1>Subject</td></tr><tr><td rowspan=1 colspan=1>Consistency of lexical scope</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.20</td><td rowspan=1 colspan=1>0.14</td></tr><tr><td rowspan=1 colspan=1>Level of precision</td><td rowspan=1 colspan=1>5.00</td><td rowspan=1 colspan=1>1.00</td><td rowspan=1 colspan=1>0.20</td></tr><tr><td rowspan=1 colspan=1>Subject</td><td rowspan=1 colspan=1>7.00</td><td rowspan=1 colspan=1>5.00</td><td rowspan=1 colspan=1>1.00</td></tr></table>

Table 10: Criteria comparison matrix as given by annotators from one group for the subset Decision\_Section.

<table><tr><td>Dataset</td><td>Baseline</td><td>Mistral</td><td>LLAMA3</td><td>Qwen2.5-7B</td><td>GPT-4o-mini</td></tr><tr><td rowspan="5">Abercrombie</td><td>Non-AHP</td><td>53.3%</td><td>53.3%</td><td>53.3%</td><td>73.3%</td></tr><tr><td>AHP-Instruction</td><td>46.7%</td><td>40.0%</td><td>33.3%</td><td>60.0%</td></tr><tr><td>Single-Agent</td><td>40.0%</td><td>13.3%</td><td>46.7%</td><td>60.0%</td></tr><tr><td>Multi-Agent(Manual)</td><td>26.7%</td><td>40.0%</td><td>40.0%</td><td>40.0%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>40.0%</td><td>66.7%</td></tr><tr><td rowspan="5">Judicial_Ethics</td><td>Non-AHP</td><td>86.7%</td><td>73.3%</td><td>66.7%</td><td>73.3%</td></tr><tr><td>AHP-Instruction</td><td>73.3%</td><td>73.3%</td><td>60.0%</td><td>86.7%</td></tr><tr><td>Single-Agent</td><td>40.0%</td><td>60.0%</td><td>60.0%</td><td>66.7%</td></tr><tr><td>Multi-Agent(Manual)</td><td>20.0%</td><td>73.3%</td><td>73.3%</td><td>73.3%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>60.0%</td><td>80.0%</td></tr><tr><td rowspan="5">Common_Law</td><td>Non-AHP</td><td>80.0%</td><td>80.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>AHP-Instruction</td><td>86.7%</td><td>60.0%</td><td>86.7%</td><td>93.3%</td></tr><tr><td>Single-Agent</td><td>53.3%</td><td>86.7%</td><td>100.0%</td><td>86.7%</td></tr><tr><td>Multi-Agent(Manual)</td><td>80.0%</td><td>86.7%</td><td>93.3%</td><td>93.3%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>93.3%</td><td>100.0%</td></tr><tr><td rowspan="5">Decision_Section</td><td>Non-AHP</td><td>53.3%</td><td>66.7%</td><td>66.7%</td><td>46.7%</td></tr><tr><td>AHP-Instruction</td><td>60.0%</td><td>40.0%</td><td>73.3%</td><td>73.3%</td></tr><tr><td>Single-Agent</td><td>40.0%</td><td>13.3%</td><td>60.0%</td><td>66.7%</td></tr><tr><td>Multi-Agent(Manual)</td><td>20.0%</td><td>46.7%</td><td>66.7%</td><td>40.0%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>66.7%</td><td>80.0%</td></tr><tr><td rowspan="5">Abercrombie</td><td>Non-AHP</td><td>51.8%</td><td>44.5%</td><td>45.3%</td><td>71.3%</td></tr><tr><td>AHP-Instruction</td><td>37.7%</td><td>33.2%</td><td>40.4%</td><td>64.8%</td></tr><tr><td>Single-Agent</td><td>39.8% 26.1%</td><td>13.0%</td><td>46.6%</td><td>58.7%</td></tr><tr><td>Multi-Agent(Manual)</td><td></td><td>48.4%</td><td>29.3%</td><td>38.5%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>40.0%</td><td>66.7%</td></tr><tr><td rowspan="5">Judicial_Ethics</td><td>Non-AHP</td><td>86.6%</td><td>70.0%</td><td>64.1%</td><td>60.0%</td></tr><tr><td>AHP-Instruction</td><td>77.8%</td><td>60.0%</td><td>58.3%</td><td>86.6%</td></tr><tr><td>Single-Agent</td><td>39.9%</td><td>58.3%</td><td>59.8%</td><td>76.9%</td></tr><tr><td>Multi-Agent(Manual)</td><td>20.0%</td><td>73.2%</td><td>79.7%</td><td>73.2%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>60.0%</td><td>80.0%</td></tr><tr><td rowspan="5">Common_Law</td><td>Non-AHP</td><td>80.0%</td><td>80.0%</td><td>100.0%</td><td>100.0%</td></tr><tr><td>AHP-Instruction</td><td>85.0%</td><td>25.0%</td><td>86.6%</td><td>89.9%</td></tr><tr><td>Single-Agent</td><td>34.8%</td><td>85.0%</td><td>100.0%</td><td>86.1%</td></tr><tr><td>Multi-Agent(Manual)</td><td>59.3%</td><td>82.9%</td><td>92.8%</td><td>92.1%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>93.3%</td><td>100.0%</td></tr><tr><td rowspan="6">Decision_Section</td><td>Non-AHP</td><td>48.2%</td><td>57.0%</td><td>70.8%</td><td>46.5%</td></tr><tr><td>AHP-Instruction</td><td>40.2%</td><td>30.1%</td><td>71.3%</td><td>73.9%</td></tr><tr><td>Single-Agent</td><td>29.5%</td><td>16.7%</td><td>45.7%</td><td>76.2%</td></tr><tr><td>Multi-Agent(Manual)</td><td>22.9%</td><td>45.3%</td><td>73.3%</td><td>25.9%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>56.2%</td><td>55.0%</td></tr><tr><td>Non-AHP</td><td>66.7%</td><td>62.9%</td><td>70.1%</td><td>69.5%</td></tr><tr><td rowspan="5">Overall</td><td>AHP-Instruction</td><td></td><td></td><td></td><td></td></tr><tr><td></td><td>60.2%</td><td>37.0%</td><td>64.2%</td><td>78.8%</td></tr><tr><td>Single-Agent</td><td>36.0%</td><td>43.3%</td><td>63.0%</td><td>74.5%</td></tr><tr><td>Multi-Agent(Manual)</td><td>32.1%</td><td>62.5%</td><td>68.8%</td><td>57.4%</td></tr><tr><td>Multi-Agent(Auto)</td><td></td><td></td><td>52.1%</td><td>72.7%</td></tr></table>

Table 11: Comparison of model performance across datasets. The upper block reports accuracy, while the lower block reports F1-score. The best score in each setting is shown in bold.

Criteria Generation Agent for identifying evaluation criteria, a Pairwise Weights Agent for estimating criterion importance via AHP pairwise comparisons, and a Decision-Making Agent for aggregating weighted scores and selecting the final alternative. This explicit division of labor improves modularity and makes intermediate reasoning steps transparent and interpretable.

<table><tr><td>Method</td><td>Qwen2.5</td><td>GPT-4o-mini</td></tr><tr><td>Non-AHP</td><td>24.4%</td><td>37.8%</td></tr><tr><td>Multi-Agent (Auto)</td><td>20.0%</td><td>42.2%</td></tr></table>

Table 12: Kendall’s Tau on QS World University Rankings
<table><tr><td>Method</td><td>Qwen2.5</td><td>GPT-4o-mini</td></tr><tr><td>Non-AHP</td><td>37.8%</td><td>68.9%</td></tr><tr><td>Multi-Agent (Auto)</td><td>60.0%</td><td>60.0%</td></tr></table>

Table 13: Kendall’s Tau on U.S. News Best Global Universities Rankings.

In contrast, the automatic multi-agent setting (Table 19) does not rely on predefined agent roles. Instead, a coordinating Captain Agent autonomously organizes and executes the same AHP reasoning pipeline, including problem formulation, criterion identification, weighting, and aggregation. Despite differences in agent instantiation, both settings follow the same underlying AHP formulation, enabling a controlled comparison between manually designed and automatically coordinated multi-agent reasoning.

<table><tr><td>Ranking</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>QS Ranking</td><td>MIT</td><td>Imperial</td><td>Stanford</td><td>Oxford</td><td>Harvard</td><td>Cambridge</td><td>ETH</td><td>NUS</td><td>UCL</td><td>Caltech</td></tr><tr><td>Zero-Shot (4o-mini)</td><td>MIT</td><td>Stanford</td><td>Harvard</td><td>Caltech</td><td>Cambridge</td><td>Oxford</td><td>ETH</td><td>Imperial</td><td>UCL</td><td>NUS</td></tr><tr><td>Auto Multi-Agent(4o-mini)</td><td>MIT</td><td>Cambridge</td><td>Stanford</td><td>Harvard</td><td>Oxford</td><td>Imperial</td><td>Caltech</td><td>NUS</td><td>UCL</td><td>ETH</td></tr><tr><td>Zero-Shot (Qwen-2.5)</td><td>Harvard</td><td>MIT</td><td>Cambridge</td><td>Stanford</td><td>Caltech</td><td>Oxford</td><td>Imperial</td><td>UCL</td><td>NUS</td><td>ETH</td></tr><tr><td>Auto Multi-Agent(Qwen-2.5)</td><td>MIT</td><td>Harvard</td><td>Caltech</td><td>Cambridge</td><td>Stanford</td><td>Oxford</td><td>NUS</td><td>Imperial</td><td>UCL</td><td>ETH</td></tr></table>

Table 14: QS Ranking Comparsion

<table><tr><td>Ranking</td><td>1</td><td>2</td><td>3</td><td>4</td><td>5</td><td>6</td><td>7</td><td>8</td><td>9</td><td>10</td></tr><tr><td>USNEWS Ranking</td><td>Harvard</td><td>MIT</td><td>Stanford</td><td>Oxford</td><td>Cambridge</td><td>UC Berkeley</td><td>UCL</td><td>Washington</td><td>Yale</td><td>Columbia</td></tr><tr><td>Zero-Shot (4o-mini)</td><td>MIT</td><td>Harvard</td><td>Stanford</td><td>Cambridge</td><td>Oxford</td><td>Yale</td><td>UCL</td><td>UC Berkeley</td><td>Columbia</td><td>Washington</td></tr><tr><td>Auto Multi-Agent(4o-mini)</td><td>MIT</td><td>Harvard</td><td>Stanford</td><td>Yale</td><td>Cambridge</td><td>Oxford</td><td>UCL</td><td>UC Berkeley</td><td>Columbia</td><td>Washington</td></tr><tr><td>Zero-Shot (Qwen-2.5)</td><td>Cambridge</td><td>Oxford</td><td>Harvard</td><td>MIT</td><td>Stanford</td><td>Yale</td><td>Columbia</td><td>UC Berkeley</td><td>Washington</td><td>UCL</td></tr><tr><td>Auto Multi-Agent(Qwen-2.5)</td><td>MIT</td><td>Harvard</td><td>Stanford</td><td>Yale</td><td>UC Berkeley</td><td>Oxford</td><td>UCL</td><td>Washington</td><td>Columbia</td><td>Cambridge</td></tr></table>

Table 15: QS Ranking Comparsion

<table><tr><td>System Prompt</td></tr><tr><td>You are an expert assistant in AHP decision-making. Follow a step-by-step structure.</td></tr><tr><td>1. Identify the decision problem clearly.</td></tr><tr><td>2. Identify 3–5 key criteria that influence the alternative decision.</td></tr><tr><td>3. Use AHP pairwise comparison (1–9 scale, reciprocals for less important) to weight criteria and score alternatives.</td></tr><tr><td>4. Output the ranking or prioritization in a clear, interpretable format.</td></tr></table>

<table><tr><td>Table 16: AHP-Instruction Prompt.</td></tr><tr><td>System Prompt</td></tr><tr><td>You are an expert assistant in AHP decision-making. Follow a step-by-step structure.</td></tr><tr><td>Step 1</td></tr><tr><td>Question and Alternatives: {question}</td></tr><tr><td>Step 1: Identify 3–5 key criteria that influence the alternative decision.</td></tr><tr><td>Step 2</td></tr><tr><td>Question and Alternatives: {question} Step 2: Using pairwise comparison, assign weights to the criteria and score the alternatives under each criterion using the AHP</td></tr><tr><td>1–9 scale. Use reciprocal values (1/x) for less important items.</td></tr><tr><td>Step 3</td></tr><tr><td>Question and Alternatives: {question} Step 3: Calculate the final scores for each alternative based on the alternative, question and weighted criteria, and select the best</td></tr></table>

Table 17: Single-Agent decomposition prompt.
<table><tr><td>Retrieval/Summary Agent</td></tr><tr><td>You are an expert assistant in planning analysis. You will be given a question along with a set of alternatives and examples. Your task is to analyze how each alternative relates to the question.</td></tr><tr><td>Criteria Generation Agent You are an expert in AHP decision-making.</td></tr><tr><td>You will be given a structure that includes a question, a set of alternative answers and anaysis.</td></tr><tr><td>Your task:</td></tr><tr><td>- Analyze the question and the alternatives. - Generate a concise set of evaluation criteria for AHP (3–5 criteria).</td></tr><tr><td>- Briefly define each criterion.</td></tr><tr><td>Pairwise Weights Agent</td></tr><tr><td>You are an expert in AHP decision-making.</td></tr><tr><td>You will be given a question with alternatives and a criteria list. Using AHP, do pairwise comparisons with the Saaty 1–9 scale (use reciprocals for the inverse).</td></tr><tr><td>Decision Making Agent</td></tr><tr><td>You are an expert in AHP decision-making.</td></tr><tr><td>You will be given a question with alternatives and a criteria weight list. Identify the best alternative and provide a brief explanation for your choice.</td></tr></table>

Table 18: Manual multi-agent prompt with predefined agent roles.

<table><tr><td>Captain Agent</td></tr><tr><td>You are an expert assistant in AHP decision-making.</td></tr><tr><td>Follow a step-by-step structure:</td></tr><tr><td>1. Identify the decision problem clearly.</td></tr><tr><td>2. Identify 3-5 key criteria that influence the alternative decision.</td></tr><tr><td>3. Use AHP pairwise comparison (1–9 scale, reciprocals for less important) to weight criteria and score alternatives.</td></tr><tr><td>4. Output the ranking or prioritization in a clear, interpretable format.</td></tr></table>

Table 19: Automatic Multi-Agent prompt (Captain Agent).