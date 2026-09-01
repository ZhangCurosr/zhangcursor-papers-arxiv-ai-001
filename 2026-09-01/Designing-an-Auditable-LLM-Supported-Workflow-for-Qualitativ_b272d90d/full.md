# Designing an Auditable LLM-Supported Workflow for Qualitative Thematic Analysis

Nadia Jul Jeldtoft<sup>1</sup> Tariq Yousef<sup>2</sup>

<sup>1</sup>Department of Culture and Language <sup>2</sup>Department of Mathematics and Computer Science University of Southern Denmark najel@sdu.dk yousef@imada.sdu.dk

## Abstract

Large Language Models (LLMs) offer new possibilities for scaling qualitative analysis, but existing applications often provide limited methodological transparency regarding how qualitative methods are translated into compu tational procedures. This paper presents an au ditable and privacy-preserving computational operationalization of inductive and latent The matic Analysis (TA). This paper first derives five design principles from the methodologi cal requirements of TA and the conditions in troduced by LLM-based inference: preserv ing interpretative context, maintaining trace able relationships between empirical material and analytical outputs, representing analytical constructs and reasoning explicitly, constrain ing LLM inference to interpretative tasks, and enabling privacy-preserving local deployment. Second, it presents a proof-of-concept for a two-phase workflow that operationalizes these principles by combining interpretative LLM inference with deterministic procedural con trol to generate codes, analytical justifications, themes, and theme descriptions while preserving explicit links to the source material. Third, it proposes an evaluation framework combining structural comparison with human-led TA and independent expert assessment of analyt ical quality. The evaluation is conducted on semi-structured Danish interview transcripts. and the results shows that the workflow pro duces code-level outputs with coverage broadly comparable to human annotations and highly rated analytical justifications, while generating a more compressed thematic structure charac terized by fewer and broader themes. The find ings demonstrate the feasibility of auditable LLM-supported TA through a modular work flow designed to scale to larger datasets, accom modate different LLMs, and support transfer across research domains, with domain adap tation primarily requiring adjustments to the prompting strategy.

## 1 Introduction

Large Language Models (LLMs) have expanded the possibilities for analyzing large collections of unstructured textual data. Their ability to process textual context, identify patterns and generate human-interpretable analytical outputs has led to their increasing application in qualitative research (Barros et al., 2025; Davison et al., 2024). However, qualitative analysis is defined not only by its analytical outputs but by the methodological principles through which analytical outputs are generated. Across qualitative research traditions, these principles include contextual interpretation and iterative engagement with empirical material(Kvale and Brinkmann, 2015; Hastrup, 2004; Denzin and Lincoln, 2011).

THEMATIC ANALYSIS (TA) (Braun and Clarke, 2006; Boyatzis, 1998) is a foundational method for the analysis of qualitative data across disciplines. It provides a systematic, yet flexible framework for identifying and interpreting patterns of meaning in diverse research contexts (Kvale and Brinkmann, 2015; Denzin and Lincoln, 2011). TA can be understood as an interpretative process that, in part, resembles unsupervised learning tasks such as pattern detection and clustering in unlabeled text data (Tan et al., 2019). However, unlike purely computational approaches, TA is guided by meaning and depends on the analyst’s theoretical and epistemological perspective. In practice, TA involves iteratively assigning codes to meaningful features of the data and grouping these into higher-level themes while maintaining explicit links between empirical data, codes and themes. Consequently, computational operationalizations of TA must support not only code and theme generation, but also contextual interpretation, analytical reasoning and traceable relationships across the analytical process.

Existing LLM-supported approaches to TA primarily demonstrate the analytical capabilities of the models themselves, while providing limited methodological accounts of how qualitative analytical methods are operationalized computationally. The challenge is therefore not whether LLMs can generate codes and themes, but how LLM inference can be utilized in computational workflows that preserve the methodological principles of qualitative analysis. Without careful workflow design, the use of LLMs risks obscuring rather than supporting the interpretative analytical process central to qualitative analysis (Beltoft and Galke, 2025; Davison et al., 2024).

This paper presents a a proof of concept for an auditable computational operationalization of TA that translates the method into a structured computational workflow using an LLM for analysis. In this context, auditability refers to the ability to systematically document and critically assess how analytical outputs are generated, supported by transparency across intermediate analytical operations and traceability between outputs and their empirical source material (Kroll, 2021; Liao and Vaughan, 2024; Mora-Cantallops et al., 2021).

The workflow separates interpretative inference from deterministic procedural computation. It embeds methodological quality criteria from TA directly into the computational design, and preserves explicit links between data, codes and themes throughout the analytical process in a privacypreserving manner (Cavoukian and Chibba, 2018). The proposed workflow is evaluated through structural comparison with expert-generated TA and independent expert assessment on a corpus of semistructured Danish interview transcripts.

This paper makes three contributions: First, it proposes five design principles for the computational operationalization of TA grounded in the methodological principles of qualitative research. Second, it introduces an auditable workflow architecture that combines constrained LLM inference with deterministic programmatic procedures to preserve analytical transparency, traceability, and privacy throughout the analytical process. Third, it proposes an evaluation framework for assessing whether computational operationalizations of TA preserve central methodological quality criteria through complementary structural comparison and expert assessment

## 2 Related work

Although recent studies demonstrate the functional feasibility of LLMs for supporting TA, they provide limited methodological descriptions of the computational architecture, analytical representations and workflow design decisions that translate core principles from qualitative methodology into computational procedures. This is reflected in limited descriptions of prompt design, data representations, coding procedures and theme construction, making it difficult to assess how analytical outputs are generated and how methodological principles are preserved throughout the workflow (Perkins and Roe, 2024; Ornelas et al., 2025).

More recent studies (De Paoli, 2024; Montes et al., 2025) have begun to introduce multi-stage workflows and methodologically informed prompting strategies. Nevertheless, key workflow design decisions remain underspecified, including how analytical units are represented, how contextual information is managed across inferential stages, how intermediate analytical products are linked, and how analytical traceability is maintained throughout the workflow.

A further limitation is that many existing solutions rely on proprietary cloud-based models and API infrastructures. Besides limiting methodological transparency, this raises challenges for governance and the analysis of sensitive qualitative data.

Taken together, existing work demonstrates that LLMs can support core analytical tasks in TA. However, comparatively little attention has been given to the computational operationalization of the method itself. This leaves a need for workflow architectures that explicitly embed the methodological principles of TA into the computational design while preserving analytical traceability, transparency and data governance.

## 3 Background Theory

In this work, we adopt the framework for TA proposed by Braun and Clarke (2006). TA provides a flexible methodological framework in which themes are developed through iterative interpretation of meaningful patterns in the data. Because TA depend on the analytical focus and epistemological perspective of the researcher, there is no single "correct" thematic representation independent of the analytical context and research focus. A central characteristic of TA is that pattern identification is guided by meaning rather than formal similarity.

Consequently, the identification and grouping of codes into themes depend on contextual interpretation and the focus of the analyst rather than predefined computational rules. In this work, we adopt an inductive and latent approach in which codes and themes are iteratively derived from underlying patterns of meaning in the data (Braun and Clarke, 2006).

The workflow for conducting TA consists of two overall analytical phases. The first phase (from text to codes) involves systematic coding of the dataset, where interview segments are assigned one or more codes. The second phase (from codes to themes) involves constructing broader themes by identifying patterns of shared meaning across codes.

Braun and Clarke (2006) outline a set of quality criteria for conducting TA that extend beyond the analytical outputs themselves. Based on these quality criteria, we derive four methodological dimensions of TA: the analytical process, the representation of analytical constructs, the traceability of analytical relationships, and the characteristics of analytically valid outputs. Table 1 summarizes these methodological dimensions.

These quality criteria and methodological dimensions serve different functions in the computational operationalization of TA. The dimensions of analytical process, representation, and traceability specify central characteristics that must be preserved in the computational design, while the characteristics of analytically valid outputs provide criteria for evaluating the resulting analysis. These methodological considerations must furthermore be combined with the opportunities and limitations introduced by LLM-based analysis, including probabilistic inference and the governance of sensitive qualitative data.

## 4 Five design requirements for adapting LLMs for TA

LLMs possess a dual capability that makes them particularly relevant for computational TA. Through their training on large-scale textual corpora, they are capable not only of generating text, but also of identifying and organizing patterns of meaning in unstructured textual data (Hao et al., 2023; Floridi, 2023; Naveed et al., 2024) However, their application is not methodologically neutral. Successfully operationalizing TA therefore requires more than applying an LLM to qualitative data: it requires explicit computational design decisions that preserve the methodological characteristics of TA while accommodating the opportunities and limitations of LLM-based inference. A computational operationalization must therefore account both for the methodological characteristics of TA and for the conditions introduced by LLM-based inference. Based on these two sources, we derive five design principles that guide the architecture of the proposed workflow.

## 4.1 Principle 1: Preserve interpretative context

TA relies on contextual interpretation, where codes are assigned to meaningful patterns within their local context (Braun and Clarke, 2006). Consequently, a computational operationalization of TA requires analytical units that preserve sufficient contextual information to support meaningful interpretation.

When utilizing an LLM for analytical purposes, this principle introduces an additional computational consideration. Analytical units must be sufficiently large to provide the contextual information required for interpretation, yet sufficiently small to support stable inference and consistent code generation within the model’s context window (Liu et al., 2024). Data segmentation therefore represents a methodological design decision rather than a preprocessing step, as the structure of the analytical units directly shapes the analytical outputs.

## 4.2 Principle 2: Preserve traceable relationships between empirical data and analytical outputs

The methodological dimension of traceability requires that analytical relationships remain explicit throughout the analytical process. In TA, this entails maintaining transparent links from empirical data to codes and from codes to themes. In a computational operationalization, codes, themes, and their associated justifications and descriptions must therefore be linked in a manner that enables the complete analytical path from themes, through codes, and back to the original data to be reconstructed in a deterministic manner.

## 4.3 Principle 3: Represent analytical constructs and reasoning explicitly

The methodological dimension of analytical representation requires that analytical constructs remain explicit and interpretable throughout the analytical process. In TA, this entails representing codes and themes through clear labels and descriptions, while making the analytical rationale for coding and theme construction transparent and grounded in the empirical material. When an LLM is used for analysis, these analytical constructs and their associated rationales must therefore be expressed as inspectable outputs rather than remaining implicit within model inference.

Table 1: Methodological dimensions underlying the quality criteria of Thematic Analysis, adapted from Braun & Clarke
<table><tr><td>TA quality criterion</td><td>Methodological dimension</td></tr><tr><td>Systematic and thorough coding across the entire dataset</td><td>Analytical process</td></tr><tr><td>Themes organize multiple related codes into broader interpretative patterns</td><td>Analytical process</td></tr><tr><td>Code and theme labels must provide clear and meaningful representations</td><td>Analytical representation</td></tr><tr><td>Theme descriptions should describe the content and meaning of the theme</td><td>Analytical representation</td></tr><tr><td>Transparent links between empirical data, codes and themes</td><td>Traceability</td></tr><tr><td>Generated codes and themes should constitute analytically meaningful interpreta- tions aligned with the analytical focus</td><td>Analytical outputs</td></tr></table>

## 4.4 Principle 4: Constrain inference

The methodological dimension of the analytical process requires contextual judgement when identifying and abstracting patterns of meaning. LLMs are well suited to support such tasks because they can process textual context and generate humaninterpretable analytical outputs. However, these outputs are produced through probabilistic inference rather than deterministic procedures. In a computational operationalization of TA, LLM inference should therefore be limited to analytical tasks that require contextual interpretation, while procedural operations that do not require interpretation should be handled deterministically. This establishes afunctional separation oflabor between interpretative tasks and the procedural tasks of computational control.

## 4.5 Principle 5: Preserve privacy though local deployment

Qualitative interview data often contain sensitive information, making privacy preservation a methodological requirement. Cloud-based infrastructures may pose challenges in relation to data governance and transparency. Consequently, computational operationalizations of TA should be designed to support privacy-preserving deployment (Cavoukian and Chibba, 2018). This requires deployment strategies that support local inference using openweight language models without exposing sensitive data to external services.

## 5 Data

The data for the development of the workflow stems from the research project The Family Revolution (FREI) (Bjerre et al., 2022). The project investigates micro-level experiences of family life and social change in Denmark between the 1950s and 1970s. The interviews provide a suitable case for developing and evaluating computational support for TA because they consist of rich, contextdependent qualitative narratives that require interpretative analysis.

The interviews were collected from 2021 to 2025 by Danish high school students. All interviews were transcribed with automated transcription software, followed by manual quality control and anonymization. A sample of 100 interview transcripts was selected from the larger dataset (n=570) using a stratified sampling strategy. First, interviews were stratified by length (long, medium, short) to reflect the distribution of the full dataset. Within each length category, interviews were then selected to achieve an approximately balanced gender distribution. Overall, this sample is intended to preserve key structural characteristics of the full dataset, including variation in interview length, narrative structure and thematic content. The sample dataset was randomly divided into a development and a test set at the interview level to separate model development from evaluation. After excluding one interview due to poor transcription quality, the final dataset consisted of 99 transcripts, where 5 transcripts were assigned to the test set and the remainder of 94 transcripts to the development set.

To satisfy principle 1 (Preserve interpretable context), interview transcripts were segmented into analytical units consisting of two consecutive question-answer pairs (See Figures 1 and 2). The choice of two consecutive question-answer pairs as the analytical unit represents a relevant level of segmentation for conducting TA. Smaller units risk loss of interpretative context and excessive overlap between generated codes, while larger units increase the risk of the LLM “losing” context by ignoring or overlooking parts of the input text when segments become too long. Initial experiments with different segment sizes suggested that longer input segments were associated with less stable outputs. This likely reflects a trade-off between prompt length , contextual complexity, and output generation within the model context window. Although this relationship was not systematically evaluated, the selected segmentation strategy provides a sufficient basis for the development of the proof-of-concept. Exploration of optimal segment size remains a relevant area for future development and testing.

## 6 Methodology

A functional division of labor structures the workflow by aligning computational roles with the methodological requirements of TA.

Specifically, the central interpretative tasks of generating and representing codes and themes, including the analytical reasoning underlying the analysis of assignment of codes and themes, are delegated to the LLM through domain-specific prompting strategies. The procedural task of ensuring systematic and transparent links between data extracts, codes, and themes are handled programmatically through deterministic processing and validation procedures.

This separation between interpretative inference and procedural control enables the workflow to preserve key methodological requirements of TA while supporting computational traceability and reproducibility. Generated codes and themes remain explicitly linked to their originating data, while all analytical and procedural steps are systematically represented within the workflow (Figure 3).

A smaller, open-weight model Mistral 7B v0.3 Instruct (Mistral AI, 2024) is used for the interpretative tasks. Recent benchmarking suggests that closely related Mistral models provide functional Danish-language capabilities, although with lower performance compared to larger proprietary models (Vejlgaard Holm et al., 2025) and aligns with Principle 5: Preserve privacy though local deploy-

ment.

Inference was configured deterministically to reduce stochastic variation. Because the workflow consists of analytical tasks with varying output requirements, generation length is constrained to between 40 and 800 new tokens depending on the task.

Figure 3 illustrates the different analytical representations which are generated throughout the workflow across Phase 1 and Phase 2: codes and accompanying justifications, themes and accompanying theme descriptions.

## 6.1 Phase 1: From text to codes

Each interview segment is processed by the LLM using a three-part, domain-specific prompting strategy inspired by Self-Refine (Madaan et al., 2023) and which emphazises that each code must be accompanied by an explicit justification (Principle 3). This also reflects the many-to-many relationship between text and codes where a single segment may be assigned multiple codes and similar codes may recur across different segments. Output generation is constrained through predefined Pydantic schemas (Pydantic, 2024). These are implemented both during inference using Outlines (dottxt-ai, 2023) and again during parsing and validation to ensure structurally consistent outputs. This step implements principles 1 and 4 by both defining relevant constraining inference. The resulting codes and justifications are subsequently normalized to enforce standardized representations before being integrated into a structured dataframe which link outputs to their corresponding interview segments and identifiers in alignment with Principle 2. Table 2 presents an example of outputs from Phase 1.

## 6.2 Phase 2: From codes to themes

Codes generated in Phase 1 are used to construct higher-level themes, reflecting a many-to-one relationship between codes and themes, where multiple codes are grouped in a single theme. An initial thematic map is first generated by prompting the LLM on a random batch of 12 codes and corresponding justifications for analytical context<sup>1</sup>. The remaining codes, along with their associated justifications and interview segments for context, are then processed sequentially. Using a methodological and domain-specific analytical prompting strategy, the LLM assigns each code to an existing theme or mark it as non-applicable.

![](images/c44fa29b87690476e6e354a56124a280868ae192057fe8991def9923606a5ece.jpg)  
Figure 1: Distribution of segments per interview (n = 99)

Unassigned codes are continuously accumulated in a separate pool. When this pool reaches 20 codes, the thematic map is updated by prompting the LLM on a new random subset of 12 codes, allowing for iterative refinement and expansion of the thematic structure. During these updates, previously generated themes are provided to the model as contextual guidance, and the model is instructed to introduce new themes only when the current data contains patterns not already represented in the existing thematic structure. This enables iterative expansion of the thematic map while preserving localized inference at the assignment level. The refinement of the thematic map and assigning codes to themes continues until all codes are either assigned or incorporated through map updates.

The results of Phase 2 are themes, descriptions of themes as well as justifications explaining the reasoning for why each code is assigned to its theme, again in alignment with design principles 2-4. Codes that remain unassigned are retained in the pool for documentation, indicating that they may have limited support in the data. As for Phase 1, outputs are constrained, validated, and integrated into a structured dataframe linking themes to their identifiers, thereby ensuring traceability between codes, themes, and interview segments.

Finally, the workflow generates an inverse “codes-in-themes” mapping documenting code-totheme assignments together with their justifications. Here justifications are provided which explain the analytical reasoning for why codes are assigned to themes. An example of Phase 2 outputs is shown

![](images/ed25ad235a8e6361a78c96cd71235eca14edd8a9eb33f783af2fb757622f4f48.jpg)  
Figure 2: Log-transformed number of tokens per segment (n = 99)

in Table 3.

Across both phases, the design translates inductive TA into an auditable computational workflow while maintaining core TA requirements such as context sensitivity, data traceability, and coherent representation of codes and themes.

The design thus implements TA in alignment with the central epistemology of the method and as a proof-of-concept. Furthermore, the proofof-concept supports scalability and transferability across datasets and research contexts, as adaptation primarily involves adjusting the domain-specific prompting strategy to reflect the analytical focus and characteristics of the data. The design in effect also integrates a Privacy By Design approach (Cavoukian and Chibba, 2018) as local deployment is handled within controlled environments where no data is exposed to external systems.

## 7 Evaluation framework

The proposed design implemements the methodological principles of TA outlines above. However, adherence to the design does not in itself establish whether the resulting codes and themes constitute analytically meaningful interpretations. The central quality criterion of TA - the characteristics of analytically valid outputs (Table 1) - therefore must be addressed through evaluation.

Because inductive TA is inherently interpretative, multiple analytical outputs of the same empirical material may be methodologically valid. Consequently, conventional metrics based on accuracy, semantic similarity or inter-annotator agreement are insufficient as stand-alone measures of analytical quality (Beltoft and Galke, 2025; James, 2026)

![](images/3bc1cca3f0f68622992a458f5f94b4753ae620c4af4a5cb3236518d4423be659.jpg)  
Figure 3: Overview of the proposed pipeline. Transcribed interviews are segmented into analytical units(two Q–A pairs). Phase 1 generates segment-level codes with justifications, while Phase 2 aggregates codes into themes via an iterative many-to-one mapping. Outputs are validated and structured to preserve links between segments, codes, themes and justifications.

We therefore propose an evaluation framework that assesses the computational operationalization from two complementary perspectives: (i) the structural characteristics of the generated analysis and (ii) the analytical quality of its outputs. The first perspective examines how the workflow organizes the empirical material into codes and themes and identifies systematic tendencies relative to human-led analyses. The second perspective examines whether the generated outputs are analytically meaningful in relation to the empirical material and research focus and satisfy output-level quality criteria for TA.

Accordingly, the framework combines a structural comparison with an independent expert assessment. Human-led analyses are used as methodological reference points rather than as a single correct representation of the empirical material.

## 7.1 Annotated datasets for evaluation

To support the evaluation, annotations were produced from the test dataset by three researchers experienced in qualitative analysis. The three annotators worked independently and followed the same methodological procedure.

One annotator completed a full thematic analysis of the entire test set (n=291 segments), providing a reference for workflow-level evaluation across the two phases. Two additional annotators independently coded a subset of interview segments (n=5) which provided an additional reference only at the coding level.

![](images/fecf9d6f69881b49b4191b4d2421f17eba821a53bd23bea05e1d8a05461e426c.jpg)  
Figure 4: Overview of the evaluation data. The LLM and Human Annotator 1 analyzed the full dataset across both Phase 1 and Phase 2 (Evaluation Dataset A), while Human Annotators 2 and 3 independently annotated a smaller subset for the Phase 1 evaluation only (Evaluation Dataset B).

Together, the two datasets support complementary forms of evaluation. Evaluation Dataset A provides a reference for assessing the complete computational workflow across both coding and theme generation, whereas Evaluation Dataset B supports focused comparison of code-level outputs across multiple human annotators, see Figure 4.

Table 2: Example of annotated interview segments showing the relationship between text excerpts, assigned codes, and analytical justifications. For readability, the interview segments are here translated to English, although the LLM generated the codes based on the original Danish text.
<table><tr><td>Interview Excerpt</td><td>Codes</td><td>Justification</td></tr><tr><td>Q: what other jobs have you had? A: Yes, but then [the informant&#x27;s husband] and I — Granddad and I —we moved in together and moved to Kolding. And I got a job as a brewery worker at a place called [brewery], which no longer exists. I worked there for.. . four years, I think, and earned 10 kroner an hour. That was what you earned —10 kroner an hour. Then we got a collective agreement and our wages went up by 2 kroner. That was an incredible amount of money. Q: Mm, that was a lot compared with 10 kroner. So... A: Yes, yes. So you should be grateful that you&#x27;re able to earn money today. Yes. But, well—and I was actually very happy about that too, and then—yes. A good life. A wonderful life. We got an apartment and then—well, I don&#x27;t know, is this where the questions about the children and things like that come in?</td><td>&#x27;WORK HISTORY&#x27;, &#x27;ECONOMIC CHANGE&#x27;, &#x27;POSITIVE LIFE EXPERIENCE&#x27;</td><td>i) Discusses the informant&#x27; s work history and employment in a brewery in Kolding. ii) Mentions the significant increase in wages from DKK (Danish Krone) over time, indicating economic changes in the society during the time period under study. iii) Expresses a positive sentiment about their life, including having a good life and a happy family.</td></tr><tr><td>Q: Yes. The next question is about whether you took maternity leave or parental leave when you had your children. A: There was no such thing back then. Q: There was no such thing?</td><td>&#x27;NO FORMAL PARENTAL LEAVE&#x27;, &#x27;DURATION OF PARENTAL LEAVE&#x27; A: I think we had three months when we could stay</td><td>i) The informant implies that there was no formal parental leave policy in place during their time as they mention only having three months to be at home with their children without any specific policy being named. ii) The informant specifies a duration of 3 months for parental leave.</td></tr></table>

## 7.2 Expert assessment for quality of codes

To assess the analytical quality of workflow outputs, an expert assessment was conducted using Evaluation Dataset B together with the corresponding outputs generated by the workflow on the same interview excerpts. The assessment was carried out by two independent researchers experienced in qualitative methodology and familiar with the analytical focus of the FREI project.

Outputs were evaluated independently using a purpose-built survey in which annotator outputs were presented in randomized and blinded order to reduce evaluation bias. The evaluation assessed two complementary quality dimensions using fivepoint Likert scales: (i) code coverage, referring to the extent to which assigned codes captured the meaning of the interview excerpt, and (ii) justification quality, referring to the clarity and adequacy of the analytical reasoning underlying each assigned code.

Beyond its application to the present workflow, the evaluation framework offers a transferable approach for assessing future computational operationalizations of qualitative analytical methods.

Given the relatively small datasets, the present analysis is exploratory and relies primarily on descriptive statistics rather than significance testing. The findings should therefore be interpreted as indicative rather than conclusive. However, the framework can be extended to larger datasets and more comprehensive statistical analyses.

## 8 Results

First, the structural comparison characterizes how the workflow organizes the empirical material into codes and themes relative to human-led analyses, thereby identifying systematic tendencies in the generated analysis. Second, the independent expert assessment examines whether the generated codes and their associated justifications constitute analytically meaningful and well-grounded interpretations of the empirical material.

## 8.1 Structural comparison

Structural comparisons of the five randomly selected interview segments from evaluation dataset

Table 3: Example themes produced during the code-to-theme stage of the thematic analysis pipeline, showing theme descriptions, representative code labels, and the number of codes assigned to each theme.
<table><tr><td>Theme label</td><td>Theme description</td><td>Code label</td><td></td><td># Codes</td></tr><tr><td>Struggles in Education</td><td>The participants’ experiences reveal a shared struggle with education, often characterized by challenges, repetition, and negative interactions with teachers.</td><td>&#x27;ECONOMIC &#x27;TRADITIONAL HEALTHCARE&#x27;, ..</td><td>HARDSHIP&#x27;, LIFESTYLE&#x27;, &#x27;LIMITED ACCESS TO CHILD</td><td>10</td></tr><tr><td>Impact of Feminist Move- ments</td><td>references to significant feminist movements in- dicate a profound impact on their lives, particu- PARTICIPATION IN FEMINIST larly in terms of women&#x27;s liberation and chang-</td><td colspan="3">The informants&#x27; personal experiences and direct &#x27;GENDER ROLE CHANGES&#x27;, 125 &#x27;SUPPORTIVE ATTITUDE&#x27;, &#x27;NON</td></tr></table>

B shows that the LLM in Phase 1 produces a broadly comparable number of codes per segment relative to human annotators. As shown in Table 4, the average number of generated codes per segment falls within the range observed across human annotators, although the LLM generally produces slightly fewer codes overall. Across human annotators, variation in the number of codes per segment was also observed. Given that both the LLM and human annotators were instructed to generate up to three codes per segment, some degree of similarity in coding volume is expected. Nevertheless, the results suggest that the workflow operationalizes Phase 1 at a level broadly comparable to human annotators.

A comparison of average number of code ratios per segment between the LLM workflow and the single annotator in evaluation dataset A shows that the workflow produces an average of 2.33 codes per segment, compared to 1.85 codes per segment for the human annotator. In contrast to the comparison with evaluation dataset B, this indicates that the LLM workflow performs coding in Phase 1 in a slightly more inclusive manner by assigning a larger number of codes per interview segment.

However, differences become more apparent when examining code diversity, i.e. how many codes are unique and reused respectively. In TA, code diversity may provide insight into the degree of analytical differentiation and granularity applied during coding, as more diverse code generation can reflect increased sensitivity to variation and nuance within the data.

Table 5 show that the LLM generates fewer unique code labels overall compared to the human annotators, although patterns of code reuse vary across both human and LLM-generated outputs. Given the relatively small absolute differences, the results overall suggest broadly comparable coding behavior in terms of code diversity across human annotators and the workflow.

The lower number of unique codes produced by the LLM is particularly noteworthy. This result is interesting given the methodological setup of the workflow, in which the LLM processes each interview segment independently without memory of prior coding decisions. In contrast, human annotators code segments with awareness of previously generated codes and may therefore be expected to reuse existing labels more consistently across segments. From this perspective, it could be expected that the LLM would generate a larger number of unique codes due to the absence of cumulative coding awareness across the dataset. The fact that this is not observed may suggest that the computational workflow is able to meet the quality criteria for "systematic and thorough coding across the entire dataset" (See Table 1) despite the localized inference structure of Phase 1.

Another plausible explanation for this pattern is the effect of the domain-specific prompting strategy of Phase 1. Given the otherwise high capacity for linguistic variation for the LLM , it is notable that the model does not produce a substantially larger number of unique codes compared to the human annotators.

The final structural analysis focuses on how the workflow i Phase 2 transforms codes into broader thematic structures relative to human-led TA.

Table 6 compares overall structural properties of the thematic organization generated by the workflow and the human annotator (evaluation dataset

Table 4: Number of codes generated by the proposed LLM-based pipeline and three human annotators across five randomly selected interview segments (unit) from the test set (n= 5 segments)
<table><tr><td>Segment id</td><td>Human 1</td><td>LLM</td><td>Human 2</td><td>Human 3</td></tr><tr><td>901</td><td>1</td><td>1</td><td>2</td><td>2</td></tr><tr><td>2333</td><td>3</td><td>1</td><td>1</td><td>2</td></tr><tr><td>4580</td><td>2</td><td>1</td><td>1</td><td>2</td></tr><tr><td>7490</td><td>1</td><td>2</td><td>2</td><td>2</td></tr><tr><td>7524</td><td>2</td><td>1</td><td>2</td><td>2</td></tr><tr><td>Mean</td><td>1.8</td><td>1.2</td><td>1.6</td><td>2.0</td></tr></table>

Table 5: Comparison of unique, reused, and total code counts across human annotators and the LLM $( n = 5$ segments).
<table><tr><td>Annotator</td><td>Unique codes</td><td>Reused codes</td><td>Total codes</td></tr><tr><td>Human 1</td><td>9</td><td>0</td><td>9</td></tr><tr><td>LLM</td><td>4</td><td>2</td><td>6</td></tr><tr><td>Human 2</td><td>8</td><td>0</td><td>8</td></tr><tr><td>Human 3</td><td>8</td><td>2</td><td>10</td></tr></table>

A). The results show substantial structural differences between the thematic composition produced by the workflow and the human annotator. Compared to the human TA, the LLM generates fewer but substantially larger themes in terms of code concentration per theme. This is reflected in both the mean and median number of codes per theme, as well as in the considerably larger maximum theme size produced by the workflow.

These findings suggest that the workflow operationalizes Phase 2 through a more compressed thematic structure, where higher number of codes are aggregated into broader themes. In contrast, the human annotator produces a more fine-grained thematic composition distributed across a larger number of smaller themes.

Figure 5 further illustrates differences in how codes are distributed across themes in the humanand LLM-generated thematic compositions. The workflow exhibits a more concentrated thematic composition, where a relatively small number of themes contain a large proportion of the assigned codes. In contrast, the human-generated thematic composition distributes codes more evenly across a larger number of medium-sized themes.

These differences may partly reflect characteristics of the Phase 2 workflow design, in which codes are iteratively assigned to an evolving thematic map. Because codes are continuously evaluated against already established themes, the workflow may favor incorporation into existing themes rather than assigning codes into new themes. In contrast, human analysts ideally develop themes with broader contextual awareness of previously generated themes, analytical decisions, and the dataset as a whole. This may support a fine-grained, and as such distributed, thematic composition. Although this relationship was not systematically evaluated, the concentration of codes within fewer LLMgenerated themes may suggest that the Phase 2 design promotes early thematic consolidation, where initially established themes increasingly shape subsequent code-to-theme assignments.

![](images/2c367a95a7e3f1719a6e2c13ea91aa587de56d197df7a307189a7c8eeda8e514.jpg)  
Figure 5: Comparison of ranked theme sizes in humanand LLM-generated thematic structures. Distribution of theme sizes measured by the number of assigned codes, ranked from largest to smallest theme (n=291 segments).

Table 6: Comparison of theme compositions generated by a human annotator and by the proposed LLM-based pipeline on the test set. Two human-generated themes were merged due to a singular/plural variation in the theme label. The theme description was identical for both labels due to human error (n=291 segments)
<table><tr><td>Metric</td><td>Human</td><td>LLM</td></tr><tr><td>Total number of themes</td><td>19</td><td>10</td></tr><tr><td>Mean codes per theme</td><td>20.79</td><td>46.7</td></tr><tr><td>Median codes per theme</td><td>18</td><td>29</td></tr><tr><td>Largest theme (no. of codes)</td><td>54</td><td>148</td></tr><tr><td>Smallest theme (no. of codes)</td><td>4</td><td>2</td></tr></table>

## 8.2 Expert assessment of quality of codes and justifications

The expert assessment indicates that the proposed workflow generates code-level outputs comparable to those produced by human annotators, while the accompanying analytical justifications receive consistently higher expert ratings.

Figure 6 (left) shows mean code coverage scores between human- and LLM-generated outputs across the five interview segments. Overall, the LLM achieves coverage scores comparable to or higher than the human annotators across several interview segments, although performance varies across cases. The mean coverage score for the LLM is slightly higher overall (3.40) compared to the human annotators (3.13).

The results further show substantial variation between interview segments. In some cases, such as segments 4580 and 7490, the workflow receives noticeably higher coverage scores, whereas the human annotators receive higher scores in segments 901 and 7524.

Figure 6 (right) zooms in on the scores at the individual annotator level. Here additional variation becomes apparent. The plot shows that mean coverage scores vary substantially across human annotators, with the LLM score falling within the broader range observed across the human evaluations. Notably, the difference between the LLM and individual human annotators is comparable to the variation observed between the human annotators themselves.

These findings suggest that the workflow preserves sufficient contextual information to support meaningful interpretation while enabling the LLM to identify analytically relevant aspects of the empirical material. Despite relying on localized inference where the LLM does not have access to it previous coding decisions, the workflow produces code-level outputs that are broadly comparable to those of human annotators.

Turning to the quality of the justifications, Figure 7 (right) shows that the workflow generally receives higher justification quality scores across the evaluated interview segments, and in several cases by a substantial margin. Compared to the human annotators, the LLM produces more highly rated analytical justifications in four out of the five interview segments, resulting in a higher overall mean justification score (3.80 compared to 2.73 for the human annotators).

The differences are particularly pronounced in segments 2333 and 4580, where the workflow receives the maximum score, while the humangenerated justifications remain closer to the overall human mean. Compared to the code coverage evaluations, the observed pattern is also more consistent across interview segments. This suggests that expert evaluations of justification quality are less sensitive to variation between individual segments.

Figure 7 (right) examines the evaluations at the annotator level. The results show that the workflow achieves a higher overall mean justification quality score than all three human annotators. At the same time, substantial variation is also observed across the human annotators themselves, with mean scores ranging from approximately 2.5 to 3.3. Compared to the code coverage quality evaluations, the workflow therefore appears to perform more consistently above the range of the human annotators with regard to justification quality.

These findings in relation to justification quality suggest that requiring the workflow to generate explicit analytical justifications does not merely improve transparency, but also produces representations of analytical reasoning that experts evaluate as more interpretable than those produced through human annotation.

Although the evaluation data is limited in scope, the results from the expert evaluations of code coverage and justification quality suggest that the workflow produces outputs at a level comparable to human-led TA and in some cases receives higher evaluation scores, particularly in the explicit articulation of analytical reasoning.

![](images/2082d1cc0bed23abca0f819199120c7de484ef0c81806023406f7b291efb1e6d.jpg)

![](images/6399a693bf5fd136e3790d7bde56f5eaa5c49f5b8ea053c5f82c2c8079254d00.jpg)  
Figure 6: Comparison of code coverage scores. Left: mean code coverage scores across the five interview segments. Right: overall mean code coverage scores for the human annotators and the proposed LLM. Scores are averaged across the two evaluators (n = 33).

![](images/01b6b8ce60fda9c13ad48d4bf986f7dfab1ef5618417cc0f348492cefffbcad6.jpg)

![](images/a644902b7a8dbd831e46b75868f6ea393aec0885b73c8615f47693955d5259ca.jpg)  
Figure 7: Comparison of justification quality scores. Left: mean justification quality scores across the five interview segments. Right: overall mean justification quality scores for the human annotators and the proposed LLM. Scores are averaged across the two evaluators (n = 28).

## 9 Conclusion

This paper has presented an auditable computational operationalization of inductive and latent thematic analysis. Rather than treating TA as a sequence of generic code- and theme-generation tasks, the proposed approach translates central methodological requirements into five explicit design principles: preserving interpretative context, maintaining traceable relationships between empirical material and analytical outputs, representing analytical constructs and reasoning explicitly, constraining LLM inference to interpretative tasks, and supporting privacy-preserving local deployment.

These principles are operationalized through a functional separation between interpretative LLM inference and deterministic procedural control. While the LLM generates codes, themes, and analytical justifications, programmatic procedures structure, validate, and preserve the relationships between interview segments, codes, and themes. Auditability therefore arises from the workflow architecture as a whole rather than from the transparency of the language model itself.

The exploratory evaluation provides preliminary support for the feasibility of this approach. At the coding level, the workflow produced outputs with expert-rated coverage broadly comparable to that of human annotators. Its analytical justifications received higher mean ratings, indicating that requiring explicit representations of analytical reasoning may strengthen the transparency and interpretability of LLM-supported qualitative analysis. The structural comparison nevertheless showed that the workflow generated fewer and substantially broader themes than the human analysis. This suggests that the sequential Phase 2 procedure favors early thematic consolidation and demonstrates that computational workflow design actively shapes the resulting analytical structure.

These findings should be interpreted cautiously given the limited evaluation sample, the use of a single small open-weight model, and the reliance on one human analysis for the full workflow comparison. Moreover, the analytical quality of the generated themes was not independently assessed. Future work should therefore evaluate the workflow across larger and more diverse datasets, models, annotators, and expert evaluators, while developing Phase 2 procedures that support broader thematic reconsideration and refinement. Overall, the study demonstrates that LLM-supported TA can be computationally operationalized in a methodologically explicit, traceable, and privacy-preserving manner, while also showing that auditability requires systematic attention to the design of the entire analytical workflow

## References

Cauã Ferreira Barros, Bruna Borges Azevedo, Valdemar Vicente Graciano Neto, Mohamad Kassab, Marcos Kalinowski, Hugo Alexandre D. Nascimento, and Michelle C. G. S. P. Bandeira. 2025. Large language model for qualitative research: A systematic mapping study. In 2025 IEEE/ACM International Workshop on Methodological Issues with Empirical Studies in Software Engineering (WSESE), pages 48–55. IEEE.

Stine Beltoft and Lukas Galke. 2025. Not everything that counts can be counted: A case for safe qualitative ai. arXiv preprint arXiv:2511.09325.

Cecilie Bjerre, Mette Fentz Haastrup, and Klaus Petersen. 2022. Citizen science in the humanities: Implementing the collaborative history model (CHM) in the classroom. In Proceedings of Engaging Citizen Science Conference 2022, page 067, Aarhus University, Denmark. Sissa Medialab.

Richard E. Boyatzis. 1998. Transforming qualitative information: Thematic analysis and code development. Sage Publications, Inc, Thousand Oaks, CA, US. Pages: xvi, 184.

Virginia Braun and Victoria Clarke. 2006. Using thematic analysis in psychology. Qualitative Research in Psychology, 3(2):77–101.

Ann Cavoukian and Michelle Chibba. 2018. Start with privacy by design in all big data applications. In S. Srinivasan, editor, Guide to Big Data Applications, pages 29–48. Springer International Publishing.

Robert M. Davison, Hameed Chughtai, Petter Nielsen, Marco Marabelli, Federico Iannacci, Marjolein Van Offenbeek, Monideepa Tarafdar, Manuel Trenz, Angsana A. Techatassanasoontorn, Antonio Díaz Andrade, and Niki Panteli. 2024. The ethics of using generative AI for qualitative data analysis. Information Systems Journal, 34(5):1433–1439.

Stefano De Paoli. 2024. Performing an inductive thematic analysis of semi-structured interviews with a large language model: An exploration and provocation on the limits of the approach. Social Science Computer Review, 42(4):997–1019.

Norman K Denzin and Yvonna S Lincoln. 2011. The Sage handbook ofqualitative research. sage.

dottxt-ai. 2023. Outlines. Accessed: 2026-05-22.

Luciano Floridi. 2023. AI as Agency Without Intelligence: on ChatGPT, Large Language Models, and Other Generative Models. Philos. Technol., 36(1):15, s13347–023–00621–y.

Shibo Hao, Yi Gu, Haodi Ma, Joshua Jiahua Hong, Zhen Wang, Daisy Zhe Wang, and Zhiting Hu. 2023. Reasoning with language models is planning with world models. arXiv preprint arXiv:2305.14992.

Kirsten Hastrup. 2004. Getting it right: Knowledge and evidence in anthropology. Anthropological Theory, 4(4):455–472.

Joseph James. 2026. Counting on consensus: Selecting the right inter-annotator agreement metric for nlp annotation and evaluation.

Joshua A. Kroll. 2021. Outlining traceability: A principle for operationalizing accountability in computing systems. In Proceedings of the 2021 ACM Conference on Fairness, Accountability, and Transparency, pages 758–771. ACM.

Steinar Kvale and Svend Brinkmann. 2015. InterViews: Learning the Craft ofQualitative Research Interviewing, 3 edition. SAGE, Los Angeles.

Q. Vera Liao and Jennifer Wortman Vaughan. 2024. Ai transparency in the age of llms: A human-centered research roadmap. Harvard Data Science Review, Special Issue 5.

Nelson F. Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. 2024. Lost in the middle: How language models use long contexts. Transactions of the Association for Computational Linguistics, 12:157–173.

Aman Madaan, Niket Tandon, Prakhar Gupta, Skyler Hallinan, Luyu Gao, Sarah Wiegreffe, Uri Alon, Nouha Dziri, Shrimai Prabhumoye, Yiming Yang, Shashank Gupta, Bodhisattwa Prasad Majumder, Katherine Hermann, Sean Welleck, Amir Yazdanbakhsh, and Peter Clark. 2023. Self-refine: Iterative refinement with self-feedback. arXiv preprint arXiv:2303.17651.

Mistral AI. 2024. Mistral-7b-instruct-v0.3. https://huggingface.co/mistralai/ Mistral-7B-Instruct-v0.3. Accessed: 2026-05- 22.

Cristina Martinez Montes, Robert Feldt, Cristina Miguel Martos, Sofia Ouhbi, Shweta Premanandan, and Daniel Graziotin. 2025. Large language models in thematic analysis: Prompt engineering, evaluation, and guidelines for qualitative software engineering research. arXiv preprint arXiv:2510.18456.

Marçal Mora-Cantallops, Salvador Sánchez-Alonso, Elena García-Barriocanal, and Miguel-Angel Sicilia. 2021. Traceability for trustworthy AI: A review of models and tools. BDCC, 5(2):20.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. 2024. A comprehensive overview of large language models. arXiv preprint arXiv:2307.06435.

Tatiane Ornelas, Allysson Allex Araújo, Júlia Araújo, Marina Araújo, Bianca Trinkenreich, and Marcos Kalinowski. 2025. Llm-assisted thematic analysis: Opportunities, limitations, and recommendations. arXiv preprint arXiv:2511.14528.

Mike Perkins and Jasper Roe. 2024. The use of generative ai in qualitative analysis: Inductive thematic analysis with chatgpt. Journal of Applied Learning & Teaching, 7(1):390–395.

Pydantic. 2024. Pydantic validation documentation. Accessed: 2026-05-22.

Pang-Ning Tan, Michael Steinbach, Vipin Kumar, and Anuj Kartpatne. 2019. Introduction to Data Mining, 2. ed. Pearson.

Søren Vejlgaard Holm, Lars Kai Hansen, and Martin Carsten Nielsen. 2025. Danoliteracy of Generative Large Language Models. In Proceedings of the Joint 25th Nordic Conference on Computational Linguistics and 11th Baltic Conference on Human Language Technologies (NoDaLiDa/Baltic-HLT 2025), pages 785–800, Tallinn, Estonia. University of Tartu Library.