# CONTEXT OPERATIONS TO ARCHITECTURE MODELLING OUTPUT FROM LARGE LANGUAGE MODELS AND EVALUATION CRITERIA FOR THEIR USE IN SYSTEMS ENGINEERING DESIGN

## A PREPRINT

Vinicius Kaster Marini<sup>∗</sup>   
Department of Mechanical Engineering Centre of Technology Federal University of Santa Maria   
Santa Maria/RS, BR 97105-900, Brazil vinicius.marini@ufsm.br

Petter Krus Section for Fluid Power and Mechatronics Department of Management and Engineering Linköping University SE 581 83, Linköping, Sweden petter.krus@liu.se

September 10, 2026

## ABSTRACT

The development of generative artificial intelligence resources enables opportunities of speeding up systems and engineering design work. This contribution introduces a framework of formal operations for assembling context in LLM-based engineering design. This framework involves the assembly of modular context units, including policy prompts, reference units with persistence, and user questions with prompt vectoring. This approach enables the systematic structuring of interactions with generative models. A formal method for evaluating modelling-as-code LLM outputs is also presented, which enables the evaluation of compliance to intent from LLM answers and thereby asses the support from LLMs for systems architecture modelling.

Keywords Large language models · Systems engineering · Context operations · Prompt engineering · Generative AI

## 1 Introduction

The complexity of automation in mobility systems draws to human-machine interfaces in operations as a matter of concern for systems engineering (Cummings [2004]). The implementation of artificial intelligence (AI) to augment control in novel vehicles has implications that extend back to the design process. The realization of gains from AI-enabled and connected smart devices entails added complexity and intricacy in the development of system designs as multi-domain system stacks (Törngren and Grogan [2018], Grogan [2021]).

This contribution aims to demonstrate that human-machine collaboration through the system design process can be designed in a way that leverages generative artificial intelligence (GenAI) capabilities. Among many approaches of AI, tools with these capabilities make use of large language models (LLMs) (Brown et al. [2020], Vaswani et al. [2017]), which are pre-trained over a very large corpus of data over the internet to yield conversational abilities in answering questions. These tools have attracted the attention of system and design engineers (Krus [2024], Johns et al. [2024]) for the potential support to early design, a context where other design automation techniques – including those based on machine learning as well as LLMs – fell short.

However, their use in engineering and design is subject to challenges regarding their probabilistic approach to content (Teubner et al. [2023]). While synthesis of assurance arguments makes a potential use case for LLMs upon the amount of paperwork involved, a research report by NASA analyses early explorations with LLMs and highlights their lack of matter-of-factness (Graydon and Lehman [2025]). Hence, the use of LLM-based tools in engineering and design requires careful review of the LLM outcomes towards design work products (Pradas-Gomez et al. [2024]).

## Then, how to improve the accuracy of GenAI to design intent towards system architecture modelling?

This contribution introduces the theme of LLMs in systems design and engineering within the use case of system architecture modelling, with developing from awareness to the state-of-the-art in LLMs (Marini et al. [2025]) towards the systematization of context input for that purpose. Related work involves the following use cases: assistance to systems modelling; assistance in system tools; example&rule assistance; complementary knowledge; assisted safety-driven methods; assisted reliability&safety methods; exploration of dependencies; and HMI for design assistance.

\*\*Assistance to systems modelling\*\*: Explorations on using LLMs with systems models demonstrate the early use of developing language models along system modelling frameworks. Cámara et al. [2023] experiment with prompting at the language models with focus on a single system modelling task, supported by model templates intended to provide exemplars. The integration of modelling frameworks through exemplars helps at extracting useful information to proceed a with a significant part of model-building.

\*\*Assistance in system tools\*\*: LLMs with chatbox tools can work within model-based systems engineering (MBSE) environments such as reported by DeHart [2024] and Johns et al. [2024], where the use of LLMs benefits from capturing the modelling framework in the MBSE environment and thereby enables the generation of system models from concept to architecture. This is also the focus in Timperley et al. [2025], who enable prompting at coding frameworks interface between the LLM tool and the modelling environment. Here, they add a design element ontology which enables their solution to provide significant support over the synthesis of design specifications.

\*\*Example&rule assistance\*\*: Krus [2024] explores the use of LLMs within aircraft concept design with support of structured templates and domain-specific rules to generate system configurations onto prompting the LLM to compose and generate system models with considering these inputs. This approach evolves from early contributions by using a preliminary domain question refined to a systematic prompt with topic structure, aiming to convey design intent and composition rules on objects and their mutual relations in the intended model.

\*\*Complementary knowledge\*\*: Balu et al. [2025] prompt LLMs to generate safety requirements; as they recognize the limitations by LLMs within their own pre-training, they elicit the aid of databases for the language model. They sample LLM responses at safety-focused prompts and figure the performance of agent-based RAG to generate better accurate responses against design intent. This is also the case with Hanke et al. [2025], who make use of RAG database support towards parsing unstructured content in design repositories towards structured content that can be leveraged onto system models.

\*\*Safety-driven methods\*\*: The potential of LLMs on the generative synthesis of design information has not gone unnoticed by the safety and reliability community. Nouri et al. [2024] make use of LLMs to generate safety requirements for automotive applications, with proposing a pipeline of prompts designed to automate a hazard analysis and risk assessment (HARA) procedure. Another approach is proposed by El Hassani et al. [2024, 2025], who focus the processing of relationships in failure modes and effects analysis (FMEA) process with support from product-related data, through crafted prompts, RAG and model fine-tuning.

\*\*Assistance to reliability\*\* Qi et al. [2025] make use of LLMs in an elaborate approach to performing systemstheoretical process analysis (STPA) by experimenting with prompt compositions and communication patterns between engineers and LLMs with different degrees of automation including stepwise review. The use of meta-structures to be supported by LLMs, is a characteristic in the application by Chen et al. [2025], where trustworthiness derivation trees (TDT) convey hierarchical dependencies between safety claims – generated by LLMs and curated of purpose-designed user interface – help the synthesis of assurance cases.

\*\*Exploration of dependencies\*\*: Other approach for using LLMs in systems design and engineering is the exploration of dependencies within process models. Lipizzi [2025] looks to capture dependencies between information concepts through the synthesis of triplets and their vectoring to explore the design space and synthesize it into sentence-based network graphs. Another way to look at information dependencies examines the use of LLMs to generating design structure matrices (DSMs) representing the design space through connections between design objects (Koh [2025]).

\*\*HMI for design assistance\*\*: Counter to the perception that the designer is to be automated out of the process, Marini et al. [2025] used concept maps to be parsed/splitted to enable sequenced prompt chains on mission design information for aircraft design. Krus [2025] explores ways of working with LLMs in different modes of operation: firstly, the direct generation of models from prompting; then, aircraft architecture models generated through on-the-fly generated application code that instantiates LLM-generated modules; then, through the embedding of LLM API calls in systems design applications.

These use case propositions demonstrate the diversity of situations where LLM-supported GenAI tools could help systems design. At the same time, this contribution aims to further develop knowledge and practice on their use by providing an a formal structured approach to LLM operation, and an example on how it works.

## 2 Overview of LLM-based engineering approaches

The use cases presented in the introduction establish the field of operation towards positioning this contribution is positioned to support systems engineering and design. Figure 1 shows a morphological matrix characterizing the use cases (rows) and the workflow mechanisms (columns) of the approaches identified in the literature.

The field overview from Figure 1 displays use cases that demonstrate the performance of LLMs in delivering outcomes within intended semantic and grammar approaches, often alongside sentence-based outputs. Here, we can see different characteristics of the use cases.

## 2.1 Use case design

\*\*Intent\*\*: The intent of using the LLM in each use case regards the proposition of the outcome towards the engineering design process. Design \*requirements\* state properties that the technical system under development shall meet or comply with over its lifecycle. System \*architecture\* assembles system characteristics/elements and their relations onto models conveying properties of the system. \*X-ability\* definitions regard the systematic processing of requirements and architecture to identify actual performance attributes of the technical system.

\*\*Context\*\*: The activity intent involves context in which it addresses product characteristics and missions within various applications. The use of LLMs in engineering design can address \*Product\* development, when the focus of the approach is to develop a physical system that will be produced for use as part of a given operating context; or, it can address \*Mission\* design, when the focus is to develop the actual operation with physical systems being used within it, which means any physical system will be custom-produced for use within its context.

\*\*Meta-model\*\*: The semantic meta-structure expressing the context information is seen to provide LLMs a route to achieving the intent towards certain engineering context. The formalization of design content for delegating knowledge processing to LLMs works through \*Modelling language\* related to the grammar and \*Procedure model\* related to the design representation. Then, \*Boilerplating\* uses clause/sentence structures to help automate requirement semantics and \*Concept maps\* provide visual representation of how elements and relationships are arranged.

## 2.2 Context workloading

\*\*Method\*\*: Because LLMs are language tools based on large-scale datasets, the approach to requesting the knowledge processing task bears significant influence on its output. The primary method in use by LLM approaches is by standard \*Instructions to chatbox\*, leveraging various prompting techniques in single questions or in conversations comprising a sequence of questions. There is the possibility of prompting with \*Code to chatbox\* as main conveyor of context in association to short requests. Then, the use of \*Custom application\* involves a purpose-specific application where LLM prompts are embedded within its workings and actions are performed by API calls.

\*\*Exemplar\*\*: The use of exemplars is widely recognized as supportive to provide LLMs a better context in regards to the objectives of compliance with processing intent. Users can leverage the methods with exemplars. While \*Zero-shot\* involves no exemplars besides the core question, \*One-shot\* and \*Few-shot\* can involve one or more exemplars, respectively, which convey relevant grammar, semantics, and situations that are relevant to the query. The use of \*RAG\* also helps guiding LLM responses to make answers that are closer to query intent, by having the queried LLM to draw on a domain-specific vector database.

\*\*Prompt\*\*: This regards how the demonstrated use cases frame user input to LLMs. \*Instruction\*-based prompts from the query provide general guidance to how the LLM shall process language to yield its output. Then, \*Formulae\* and \*Rules\* offer more structured control - through the relations between elements are still probabilistic - to get the LLM to yield in compliance to certain occurrence relationship from its training data. There are use cases with \*Templates\*, which show to be useful in providing detailed guidance about expected output formats and characteristics, and \*Chained\* instructions allow for sequenced responses from the buildup of context.

## 2.3 Use implementation

\*\*Tool/environment\*\*: These provide means for the user to interface with requesting information from the LLM, and understanding the outcome of its answer. Most use cases rely on \*Chatbox\* outputs which can be standalone developer-issued tools, or can interoperate with application environments such as \*MBSE tools\*. While standalone or \*Standard\* chatboxes provide incomplete and semi-compliant results, the \*Custom\* integration within applications through API-based routines helps with modelling the input contexts with favourable result to LLM outcomes.

<table><tr><td rowspan=1 colspan=2>兴</td><td rowspan=1 colspan=4>米</td><td rowspan=1 colspan=1>是</td><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1></td></tr><tr><td rowspan=1 colspan=1></td><td rowspan=1 colspan=1>Intent</td><td rowspan=1 colspan=1>Context</td><td rowspan=1 colspan=1>Meta-model</td><td rowspan=1 colspan=1>Method</td><td rowspan=1 colspan=1>Exemplar</td><td rowspan=1 colspan=1>Prompt</td><td rowspan=1 colspan=1>Tool/env</td><td rowspan=1 colspan=1>Output</td></tr><tr><td rowspan=1 colspan=1>Asistnceto lling</td><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Product ormission</td><td rowspan=1 colspan=1>Modellinglanguage</td><td rowspan=1 colspan=1>Code tochatbox</td><td rowspan=1 colspan=1>Zero-shot,One-shot</td><td rowspan=1 colspan=1>Instruction</td><td rowspan=1 colspan=1>Diagramcodeparser</td><td rowspan=1 colspan=1>Model*</td></tr><tr><td rowspan=1 colspan=1>AsicinMBS ols</td><td rowspan=1 colspan=1>RequirementArchitecture</td><td rowspan=1 colspan=1>Product ormission</td><td rowspan=1 colspan=1>Modellinglanguage</td><td rowspan=1 colspan=1>Code to in-applicationchatbox</td><td rowspan=1 colspan=1>One-shot</td><td rowspan=1 colspan=1>Instruction+&amp; formulae,chained</td><td rowspan=1 colspan=1>Chatbox inMBSE tool</td><td rowspan=1 colspan=1>Models*</td></tr><tr><td rowspan=1 colspan=1>E-Xaule-asince</td><td rowspan=1 colspan=1>Architecture</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>Modellinglanguage</td><td rowspan=1 colspan=1>Instructionsand code tochatbox</td><td rowspan=1 colspan=1>Few-shot</td><td rowspan=1 colspan=1>Instruction++rules andtemplates,composed</td><td rowspan=1 colspan=1>Diagramcodeparser</td><td rowspan=1 colspan=1>Model*</td></tr><tr><td rowspan=1 colspan=1>Kknowdecoment</td><td rowspan=1 colspan=1>RequirementX-ability</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>BoilerplatingModellinglanguage</td><td rowspan=1 colspan=1>Instructionsand code tochatbox</td><td rowspan=1 colspan=1>RAG</td><td rowspan=1 colspan=1>Instruction +</td><td rowspan=1 colspan=1>StandardchatboxChatbox inMBSE tool</td><td rowspan=1 colspan=1>Sentences&amp; tables*</td></tr><tr><td rowspan=1 colspan=1>Satfrdrvenmetthods</td><td rowspan=1 colspan=1>RequirementX-ability</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>BoilerplatingProceduremodel</td><td rowspan=1 colspan=1>Instructionsand code tochatbox</td><td rowspan=1 colspan=1>One-shotRAG</td><td rowspan=1 colspan=1>Instruction+,chainedInstruction+&amp; formulae</td><td rowspan=1 colspan=1>StandardchatboxCustomchatbox</td><td rowspan=1 colspan=1>Sentences&amp; tables*Spreadsheet</td></tr><tr><td rowspan=1 colspan=1>Asisncetorliity</td><td rowspan=1 colspan=1>ArchitectureX-ability</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>ProceduremodelModellinglanguage</td><td rowspan=1 colspan=1>Instructionsto chatboxCustomapplication</td><td rowspan=1 colspan=1>One-shotZero-shot</td><td rowspan=1 colspan=1>Instruction+,chainedEmbeddedinstructions</td><td rowspan=1 colspan=1>StandardchatboxCustomapplication</td><td rowspan=1 colspan=1>Model*</td></tr><tr><td rowspan=1 colspan=1>Explt ofdepies</td><td rowspan=1 colspan=1>ArchitectureX-ability</td><td rowspan=1 colspan=1>Product</td><td rowspan=1 colspan=1>ProceduremodelModellinglanguage</td><td rowspan=1 colspan=1>Instructionsto chatboxCustomapplication</td><td rowspan=1 colspan=1>One-shotZero-shot</td><td rowspan=1 colspan=1>Instruction+,chainedEmbeddedinstructions</td><td rowspan=1 colspan=1>StandardchatboxCustomapplication</td><td rowspan=1 colspan=1>Model*</td></tr><tr><td rowspan=1 colspan=1>Hi n ienasnce</td><td rowspan=1 colspan=1>RequirementArchitecture</td><td rowspan=1 colspan=1>MissionProduct</td><td rowspan=1 colspan=1>ConceptmapModellinglanguage</td><td rowspan=1 colspan=1>Instructionsto chatboxInstructionsand code</td><td rowspan=1 colspan=1>Zero-shotFew-shot</td><td rowspan=1 colspan=1>Instruction,chainedInstruction++composed</td><td rowspan=1 colspan=1>StandardchatboxDiagramcode</td><td rowspan=1 colspan=1>Sentences&amp; tables*Model*</td></tr></table>

Figure 1: Field of operation and positioning of this contribution

\*\*Output\*\*: The type of output determines the work product delivered by the LLM query workflow. Most use cases engage onto the generation of \*Models\*, which require an underlying grammar the LLM will attend to under its processing; this is possible because developers usually train their LLMs with including language grammars with a large variety of examples. Then, \*Sentences\*, \*Tables\*, and \*Spreadsheets\* involve direct application of natural or coded languages to their processing.

## 3 Degraded attention and mitigating factors

Hence, the use of LLMs depends on adding task-related to obtain better work product quality. However, the transformer attention mechanism (Vaswani et al. [2017]) cannot equally attend to all tokens in a context workload. This means that forcing the LLM to process multiple sources through long sequences may drive the attention mechanism to lose focus (Liu et al. [2024]), thereby reducing the effectiveness in yielding a sufficiently accurate response to the main query. The lost in the middle problem is in display by Figure 2 regarding the context workload length.

![](images/fde653c97e413a3f35e027bb492db54e90ea7f412065d2f07ecd0e539c6a7b95.jpg)

![](images/817f1e86364e68198ef364acfad89bd1b156872b9af665a970dc4eb04aa9999e.jpg)

![](images/b06f73fb052afa23e3a8fe105cdb531b6358fc4cb0bcb43616ca194b954c08b7.jpg)  
Figure 2: Reliance upon model pre-training for LLM performance.

Liu et al. [2024] verify the issue becomes more significant in proportion to the length of the assembled context $C _ { Q } = \{ t k _ { 1 } . . t k _ { n } \}$ , considering the actual context capacity of the model - tokens around $t k _ { 1 }$ and $t k _ { n }$ by both ends of the context workload are more likely to be attended to than those tokens in the middle of the context workload $C _ { Q }$ This triggers the need to understand the factors influencing the attention mechanism and how to mitigate them, which motivates use to approach the topic of degraded attention and mitigating factors in the next sections.

## 3.1 Degraded attention factors

Language processing literature diagnoses a few factors for transformer-based models to lose performance about processing the context workload. While there is no reference concerning how these factors play out in systems design and engineering tasks, any technique that uses LLMs as processing resource is liable to these:

\*\*Workload length to context capacity\*\*: the attention mechanism of a specific model has a limit on the number of tokens $\{ t k _ { 1 } . . t \bar { k } _ { n } \}$ in the context workload $C _ { Q }$ it can process, a context capacity determined by the pre-training context intake length (Chen et al. [2023]). Besides this predetermined overall limit, Gupte et al. [2025] figure a common characteristic among experimented models, a degradation in recall probability $P [ t \bar { k _ { i } } \in A _ { M } ]$ as the workload length increases over certain proportion to the total context capacity of a given model.

\*\*Token distance from main query\*\*: besides the fact that LLMs lose focus on the ’middle’ of the context workload $C _ { Q }$ , Zhang et al. [2024] also demonstrate context loss (low $P [ t k _ { i } \in A _ { M } ] )$ on tokens $t k _ { i }$ whose position is far from the main query. LLMs run on positional embeddings with higher weights for tokens close to the main query, and lower ones for those farther away from it (Naveed et al. [2025]). If the main query is located close to the middle, then the attention mechanism will mostly depend on the information by the ends of the workload (Zhang et al. [2026]).

\*\*Ambiguity across context units\*\*: The presence of ambiguous information across different context units along the workload $C _ { Q }$ can lead to confusion and misinterpretation by the LLM (Zhang et al. [2026]). This ambiguity, especially when it involves multiple context units around the middle of the workload, can arise from overlapping or conflicting information and further reduce $P [ t k _ { i } \in A _ { M } ]$ because the model has difficulty in determining the desired response. This is also the case when context units are not clearly defined about elements that relate it to the main query.

\*\*Evidence complexity\*\*: Zhang et al. [2024] found that the complexity of the evidence at hand within the context workload $C _ { Q }$ also plays a role in the degradation of context processing $\mathbf { \bar { \mathit { P } } } [ t k _ { i } \in A _ { M } ]$ towards the answer. Complex context require the so-called ’multi-hop’ connection between units across different positions $t k _ { i }$ , which is a known challenge for LLMs (Baker et al. [2024]). This characteristics means complex evidences require more effort at reasoning about specifics within a main query with respect to the context, which can be challenging for LLMs to maintain and process effectively.

\*\*Order-to-structure incoherence\*\*: The structural incoherence of the context can also lead to degradation in the performance of LLMs. This occurs when the context units are not organized in a coherent manner regarding the work product intent (Li et al. [2025]). This is explained by the behaviour of the attention mechanism, which processes the context workload through sequence-dependent concatenation $C _ { Q ( i ) } = \left\{ t k _ { 1 } . . t k _ { n } \right\} = \left| \left| { } _ { i = 1 } ^ { n } \ t k _ { i } \right. \right.$ . Here, a context workload with units that are misordered on their use to the intended answer can lead to misinterpretation by the attention mechanism, and thereby degrade the response generated by the LLM.

## 3.2 Mitigating degraded attention

The sources presented in this paper about LLM use cases demonstrate that when context units present information that is complementary and related to the main query. While there are factors inducing attention degradation, others make the effect to mitigate such mechanism, such as:

\*\*Placement about the context ends\*\*: The attention mechanism of LLMs over the context workload $\begin{array} { r } { C _ { Q } = | | _ { i = 1 } ^ { n } \ t k _ { i } } \end{array}$ is subject to the following tendencies about how the LLM reads it (Liu et al. [2024]): primacy, for the tokens in the beginning $t k _ { i  1 }$ of the context workload determine the processing of all subsequent tokens; and recency, for the tokens in its end $t k _ { i \to n }$ point out at the direction to which the model shall attend to. Zhang et al. [2024] also point out at the placement of duplicates of relevant context by the ends of the workload to avoid losing that information as reference.

\*\*Relevance to the main query\*\*: The relevance of context units to the main query is a critical factor in mitigating attention degradation. Li et al. [2025] demonstrate that when a higher relationship to the main query from within the pretraining data, atracts attention to specific context. Zhang et al. [2024] experiment with placing relevant context units closer to the main query, which they find helps the attention mechanism to focus on those units when looking to improve the quality of the response from the LLM.

\*\*Role-based context positioning\*\*: this approach involves the strategic placement of context units based on their roles in relation to the main query, with taking advantage of the primacy and recency tendencies upon the sequential processing of the context workload (Liu et al. [2024]). Guo et al. [2024] demonstrate that the placement of context units based on their roles - policy and rules in $t k _ { i  1 }$ by the beginning, taking advantage of primacy; then exemplars in $t k _ { i }$ →n by the end of the workload, taking advantage of recency - is beneficial to the quality of the answer.

\*\*Attention-trigger context tagging\*\*: this approach involves the use of tags or markers to highlight important context units. Attention triggers to context units can involve styling or semantic cues to indicate the importance of a context unit, including its relation to the main query. These can help the attention mechanism to focus on context units about their relationship to the main query, and improve the capability of the LLM to process that particular tagged unit as reference for the answer (Zhang et al. [2026]).

\*\*Reduction of context workload\*\*: Reducing the number of context units in the workload can help mitigate the effects of attention degradation Gupte et al. [2025], because it reduces the competitive demand for the attention capacity by the model. A less diverse context mitigates degraded attention on both context length and multi-hop requirement effects (Baker et al. [2024]) because a less diverse context workload entails less complexity in terms of the number of context units and the relationships between them.

## 4 Context assembly levels in LLM use cases

A study performed by Graydon and Lehman [2025] on specific use cases of LLMs in safety engineering finds resulting specifications come incomplete and inaccurate. This led our interest in the factors affecting the quality of the work product $A _ { M }$ delivered by the LLMs with basis in the context workload $C _ { Q }$ . Authors such as Cámara et al. [2023], Crabb and Jones [2024] tried the use of LLMs with simple queries to elicit responses from the LLM, and found that examples counting on the query alone yielded insufficient quality regarding the work product delivered by the LLM.

Pradas-Gomez et al. [2024] and Timperley et al. [2025], among others, demonstrate the role of context assembly into building work productas at reasonable quality, which is seen to reduce the amount of rework required to make it compliant to design intent. As section 3 explores influences to the attention mechanism of LLMs in generic terms, the following sections explore context composition for LLMs in systems design and engineering.

## 4.1 Context composition levels

As section 2 shows, use cases involve specifics to utilising and assembling supportive context towards the LLM. Besides the overview of working pinciples from Figure 1, these specifics are outlined along the following characteristics:

\*\*Conveyor formats\*\*: Role, guideline, requirement and directive definitions can involve text prose or specificallycontrolled prose when working with safety- and traceability-critical information (El Hassani et al. [2025]); short database records with cross-related information such as requirements, benchmarking and traceable safety information take benefit of csv-, markdown- and other table formats (Geissler et al. [2024]); and, visual information requires a multi-modal or vision-capable model that can interpret the visual information and relate it to the main query.

\*\*Grammar formats\*\*: These can include formal or informal structures. Natural language syntax is mostly used to convey directives and guidelines, yet can sometimes be used to convey requirements and exemplars (Dehn et al. [2025]). Formal grammars such as markup (XML, YAML, JSON, etc.) and programming languages (Python, Rust, C++, etc.) in codeblocks (Pradas-Gomez et al. [2024], Krus [2026]), and programming languages, which convey the relationships between functions and their parameters in a program.

\*\*Application adapters\*\*: These adapters enable the assembly of context and the communication to the LLM and back. Most MBSE use cases leverage application-specific APIs to enable the assembly of context units within the modelling environment and then thits communication to the LLM and back (Dehn et al. [2025], Timperley et al. [2025]); another way of assembling context workloads is the embedding of custom code and interfaces in applications through internal routines, which enable LLM calls within their own working environment (Chen et al. [2025]).

\*\*Downstream interpreters\*\*: In this approach, the forwarding of the context workload to the LLM is carried out by the means of LLM-provider APIs, which get the LLM to generate work products from the exemplars in the same modelling language (Krus [2025]). The use of modelling-as-code interpreters can include a single foundation exemplar or a set of exemplars (Krus [2024]) that convey the relationships between elements and their properties in the intended work product.

Conveyor and grammar formats enable machine-readable context workload so that the LLM can process it, along application adapters and downstream interpreters that enable LLM outputs to be actionable for a modelling environment.

## 4.2 Context composition strategies

Some use cases obtain acceptable work products from modelling-as-code context workloads, which enable modelbuilding by downstream interpreters, or by using application adapters within MBSE modelling environments. Both approaches share the following strategies:

\*\*Unit composition:\*\*: The use cases display a diversity of strategies: counting on the pretraining context of the specific LLM (Cámara et al. [2023]); the cumulation of downstream question-answer turns at the LLM for in-context learning (Von Heissen et al. [2024]); the use of formal grammar snippets as exemplars supportive to the model-building task (Krus [2024]) along with the option to alternate interaction between context workloads for LLM and generated deterministic model-builing code (Krus [2026]).

\*\*Unit order\*\*: The placement of references by along the main query (Guo et al. [2024]) is driven by the primacy and recency biases. Leveraging the tendencies related to context memory and attention degradation (Liu et al. [2024]), an ordered approach to context assembly involves setting rules, precedents, findings and other contextual information following a logical and methodic sequence; examples such as Dehn et al. [2025] and Krus [2025] demonstrate the use of ordered context units - from rules to examples -to improve the quality of the work product delivered by the LLM.

\*\*Modelling-as-code\*\*: This technique involves sequencing the context workload with starting directives that precede µ-template codeblock elements (Krus [2025]), and by allowing the modular composition/intake of separate context units (Marini et al. [2025]), both techniques making use of models-as-code around the main query. The use of code-based modelling languages is also enabled in examples like those from Timperley et al. [2025] and Dehn et al. [2025] that compose the context to LLMs within modelling environments by the means of application adapters.

\*\*Context memory\*\*: The use of context memory involves persistent context units being reused across multiple turns. This can be achieved through the use of RAG databases (Balu et al. [2025]) or by storing context units in a structured format that can be easily retrieved and reused (Hanke et al. [2025]). This helps to reduce the amount of context workload required for each query, thereby mitigating attention degradation and improving the quality of the work product delivered by the LLM.

The effectiveness of context composition is determined by how directly each element of the workload convey information to the LLM, and by how well the context units scaffold the LLM onto processing the work product.

## 5 Model generation parameters

This contribution is part of ongoing development on the use of LLMs as generative AI resource (GenAI) in systems engineering. The understanding of human-machine collaboration with GenAI makes the context, whereas the proposition of knowledge, strategy and applications to leverage the use of GenAI provide directive viewpoint. This study departs from fundamentals in the introduction to this paper and in previous contributions of our own (Krus [2024], Marini and Krus [2025]). Then, we aim at a formal approach to context operations in LLM-based systems engineering and design.

## 5.1 Choice of LLM function

The approach to context operations in this paper first involves formalizing the choice of LLM function and its parametrisation, to configure the LLM emdpoint that will receive the context workload. Then, the context workload assembly is formalized with modedls that present the context units and the operations that enable their composition onto the context workload.

To understand the context intake functionality of large language models (LLMs), we formalize its workings as a function class $' L L M _ { X } ^ { \prime }$ , where each specific model plays the role of a function; $' X ^ { \prime }$ may be replaced by any particular LLM at the discretion of the user. Here, we use equation clauses to present the functionality and parameters of LLMs, and denote all references to them as $L L M _ { X }$

Equation 1 approaches LLMs in representing the ability of the professional user to choose one particular model among several available:

$$
\begin{array} { r } { L L M _ { X } = \vee \{ M _ { G P T _ { m } } , M _ { C l a u d e _ { m } } , M _ { G e m i n i _ { m } } , . . . , } \\ { M _ { D e e p s e e k _ { m } } , M _ { Q w e n _ { m } } , M _ { K i m i _ { m } } , . . . , } \\ { M _ { M i s t r a l _ { m } } , N _ { C o m m a n d _ { m } } , . . . \} } \end{array}\tag{1}
$$

The first engagement is the choice of operator – $. L L M _ { X }$ function –, which is done with the application by choosing a given individual model operator to answer an intended request. Here, M refers to each being a model, and m refers to one of several versions in a specific model lineup. One can then understand M as an instance of $L L M _ { X }$ which can perform advanced language processing operations, including design and engineering tasks, where $L L M _ { X }$ involves the processing of any chosen M at answering to a request by the user.

## 5.2 Parameters of LLM function

Considering the model choice from those in display, each option for $L L M _ { X }$ from equation 1 has parameters that can be worked upon. The combination of such parameters in $L L M _ { X }$ can define how the context will be processed. Our contribution considers the following parameters to $L L M _ { X }$

$$
M _ { l m } : \_ m o d e l \_ s i z e \_
$$

the size of $L L M _ { X }$ on the amonut of parameters in the probability matrix from training,

$D _ { t r }$ : \_dataset\_size\_

the diversity of sources into the calculation of $M _ { l m }$ on which $L L M _ { X }$ is trained,

$T _ { l l m }$ : \_temperature\_

the index to how random the $L L M _ { X }$ will predict content in its response to a request, and,

– $C _ { M ( i ) _ { l e n } }$ : \_context\_length\_

the intake capability of $L L M _ { X }$ to receive a context length towards a request,

$A _ { M ( i ) _ { l e n } }$ : \_answer\_length\_

the answer length capability of $L L M _ { X }$ to yield content in response to a request,

$C _ { Q ( i ) _ { l e n } } :$ \_context\_workload\_

the length of the context assembled towards the call to $L L M _ { X }$

Then, $L L M _ { X }$ can be expressed by Equation 2 as a function of the parameters each call relays to the model. The parameters of main interest to our operations are: the context length capacity $C _ { M ( i ) _ { l e n } }$ , the answer length capacity $A _ { M ( i ) _ { l e n } } ,$ and the assembled context workload $C _ { Q ( i ) _ { l e n } } ;$ the first two are properties of $L L M _ { X }$ and the last one is determined from the context operations by the user.

$$
\begin{array} { r l } { L L M _ { X } = f \big ( \overbrace { M _ { l m } , ~ D _ { t r } } ^ { \mathrm { m o d e l ~ c h o i c e } } , ~ \overbrace { t o p _ { p } , t o p _ { k } , T _ { l l m } , \ldots } ^ { \mathrm { t u n i n g ~ p a r a m c t e r s } } , } & { } \\ { \underbrace { C _ { M } ( i ) _ { l e n } } _ { \mathrm { c o n t e x t ~ c a p a c i t y } } , ~ \underbrace { A _ { M ( i ) _ { l e n } } } _ { \mathrm { a n s w e r t r a p ~ d u e t h ~ c o n t e r ~ i n i t a k e } } \big ) } & { \bigg | \quad C _ { M ( i ) _ { l e n } } \geq \{ C _ { Q ( i ) _ { l e n } } + A _ { M ( i ) _ { l e n } } \} } \end{array}\tag{2}
$$

Each API library has a specific syntax for calling the function and determining its parameters. Specific language models - instances of $\dot { L L M _ { X } } \dot { - \mathrm { ~ v ~ } }$ ill also have specific parameter setting requirements for $T _ { l l m } , t o p _ { p }$ and $t o p _ { k }$ . On our main interest, settings to both $C _ { M ( i ) _ { l e r } }$ and $A _ { M ( i ) _ { l e n } }$ are also model- and provider- specific: within the condition set above, some models set limits at both whereas others allow any proportion between them.

## 6 Context operations

We have engaged onto formalizing the context assembly operations that enable a complete LLM call. Here, we use equation clauses to present context units as to their role onto supporting the LLM’s context intake. Our intent is to demonstrate the context-building components towards the enhanced chatbox. Before proceeding with the example, the following context operations are introduced:

## 6.1 Generic context formulation

Context assembly can work through assembling several context units around the main query, whose sequenced combination is intended for ingestion by $L L M _ { X }$ as a single context package. The operator will assemble the context units as available so that $L L \bar { M } _ { X }$ will perform its internal processing towards the intent. Then, the overall context workload $C _ { Q ( i ) }$ assembly for the $\because \overrightarrow { \overrightarrow { \mathbfit { i } } \overrightarrow { \mathbfit { e } } }$ call will include the following components:

$C _ { Q ( i ) }$ : \_context\_workload\_

the aggregate operated context for an $L L M _ { X }$ function call.

$C _ { u p ( k , i ) }$ − \_upstream\_context\_ in section 6.2.1:

the workload part that is added before the main query to calling $L L M _ { X }$ at each question.

$Q _ { P ( i ) } ~ -$ \_core\_question\_ from section 6.2.3

the question statement as elected by the user to call $L L M _ { X }$

$C _ { d n ( m , i ) }$ − \_downstream\_context\_ in section 6.2.2:

the workload part included after the main query towards $L L M _ { X }$ , and,

$A _ { M ( i ) }$ : \_answer\_yield\_

the answer yield from $L L M _ { X }$ to yield content in response to a request,

The context workload components for $C _ { Q ( i ) }$ will be assembled before $\ – \ C _ { u p ( i ) }$ - and after - $C _ { d n ( i ) }$ - the main query, for calling the $L L M _ { X }$ function to the intent expressed in the $Q _ { P ( i ) }$ query. The context workload $\ C _ { Q ( i ) }$ to the $L { \dot { L } } M _ { X } ^ { - }$ function results from assembling context units onto a message; the model internals in $L L M _ { X }$ will process the workload $C _ { Q ( i ) }$ onto an answer $A _ { M ( i ) }$ . Then, equation 3 displays the formulation of each single call to $L L M _ { X }$ with the individual terms to the call.

![](images/1f169e3d9d7db0f9554fa6a04b22b20ddf577df4889b5dc4187846d8228fd41c.jpg)

$$
A _ { M ( i ) } = L L M _ { X } \{ C _ { Q ( i ) } \}\tag{3}
$$

Here, $C _ { u p ( i ) }$ is placed first for the upstream context component to set the initial context for the call. Then, the core question unit $Q _ { P ( i ) }$ will express the intent of the context workload. The $C _ { d n ( i ) }$ component will convey imported reference units to provide complementary reference information. Whereas equation 3 only displays context unit grouping per position, there is the opportunity of allowing the intake of several references per context group.

The user will first compose/import the components he expects to forward to $L L M _ { X } ;$ ; here, he can add a single component either upstream or downstream of the query, or can add a composition of context units positioned around the main query. Equation 4 represent the context assembly. Here, $' k ^ { \prime }$ represents the number of upstream context units and $' m ^ { \prime }$ represents the number of downstream context units relative to the main query $Q _ { P ( i ) }$

$$
C _ { u p ( i ) } = \Big | \Big | _ { k = 1 } ^ { K } C _ { u p ( k , i ) } \qquad \Big | \qquad C _ { d n ( i ) } = \Big | \Big | _ { m = 1 } ^ { M } C _ { d n ( m , i ) } \qquad \Big | \quad k , m \in { \mathbb N }\tag{4}
$$

$$
\begin{array} { r l } { C _ { u p ( k , i ) } \ \Big | \Big | \ Q _ { P ( i ) } \ \Big | \Big | \ C _ { d n ( m , i ) } = } & { { } C _ { Q ( i ) } } \end{array}
$$

Then, the user can import or compose several context units to be positioned before and after the main query, and the application will assemble them into a single context workload $\bar { C _ { Q ( i ) } }$ to be sent to $L L M _ { X }$ as in display by Figure $^ { 3 , }$ which displays the context assembly process. The model will process the workload $C _ { Q ( i ) }$ onto an answer $A _ { M ( i ) }$ from its internals.

![](images/72fd5b3b1ccdfe69d0e15509ee943487ab1ff92547a4edcdaee51957d21ec3da.jpg)  
Figure 3: Generic assembly operations of ancillary context units for a single $L L M _ { X }$ call.

The assembly of context units from string variables results from the concatenation of individual strings into a single string variable that will carry the full context workload to $L L M _ { X }$ . This concatenation of context units is order-dependent upon the alignment between grammar structures in the input and language-processing capabilities by generativce resources such as LLMs. However, the attention mechanism within LLMs has a characteristic of reading the unified context workload from the beginning and from the end of the string, with a tendence to losing attention to the content by the middle of the string. The lost in the middle problem as identified by Liu et al. [2024] .

## 6.2 Role-focused units onto single-call

Role-focused units contain tokenized content attending to distinct purposes in relation to the main query, whose sequenced combination is intended for ingestion by $L L \bar { M } _ { X }$ as a single context package. The operator will assemble the context units as available so that $L L \bar { M _ { X } }$ will perform its internal processing towards the intent. Then, the overall context workload $C _ { Q ( i ) }$ assembly for the $\because \overrightarrow { \overrightarrow { \mathbfit { i } } \overrightarrow { \mathbfit { e } } }$ call will include the following components:

$C _ { Q ( i ) }$ : \_context\_workload\_

the aggregate operated context for an $L L M _ { X }$ function call.

$R _ { P ( i ) }$ − \_policy\_rulework\_ in section 6.2.1:

the resulting rulework prompt with directives to calling $L L M _ { X }$ at each question.

$P _ { P ( i ) }$ − \_context\_reference\_ in section 6.2.2:

a context unit to recur at every call within $\ ' _ { r } \ast$ calls of the context memory $C _ { m e m } .$ , and,

$Q _ { P ( i ) } ~ - ~ \_ { c o r e \_ q u e s t i o n _ { - } }$ from section 6.2.3

the question statement as elected by the user to call $L L M _ { X }$

$A _ { M ( i ) }$ : \_answer\_yield\_

the answer yield from $L L M _ { X }$ to yield content in response to a request,

The context workload components for $C _ { Q ( i ) }$ shall follow certain order in order to maximise the compliance by the $L L M _ { X }$ function to the intent of the query. The context workload $C _ { Q ( i ) }$ to the $L L M _ { X }$ function results from assembling these context modules onto a message to the model. The model will process the workload $C _ { Q ( i ) }$ onto an answer $A _ { M ( i ) }$ from its internals. Then, Equation 5 displays the formulation of each single call to $L L M _ { X }$

![](images/e558cc45f582b9ef6b9dc4289bbd4a6b123eb3c7a0e52dfead7db574cbef734e.jpg)

$$
A _ { M ( i ) } = L L M _ { X } \{ C _ { Q ( i ) } \}\tag{5}
$$

![](images/bc94f5c45352df83f8c9f4ae30a1eaecba70873e71f3a68c94d07510f8c51743.jpg)  
Figure 4: Role-focused assembly operations to context units for a single $L L M _ { X }$ call.

Here, $R _ { P ( i ) }$ is placed first for the policy component to set rules over the whole of the call. The $P _ { P ( i ) }$ component will convey imported reference units to provide complementary guidelines towards the answer. Then, the core question unit $Q _ { P ( i ) }$ can include exemplars such as µ-Templates (Krus [2026]) and specifics besides the actual question, and then take advantage of recency (Liu et al. [2024]) to maximise the influence of intent-related statements in the query.

## 6.2.1 Rule and policy operations

One can set policy context units to help steer the workings of $L L M _ { X }$ function to yield an answer to a closer approximation to the query intent. These context units will work as rules of engagement which will drive the $L L M _ { X }$ function to process the core question along the given context of composed and imported units. These context units can be individually composed by the user at anytime regarding its intent towards operating the $L L M _ { X }$ function. Then, the professional user can compose the following string units:

$R _ { P ( i ) }$ : \_policy\_rulework\_

the rulework prompt with directives and guidelines to calling $L L M _ { X }$

– $G _ { s p }$ : \_global\_prompt\_

a recurring prompt to every call that provides directives of engagement,

$$
\ - \ B _ { s p ( i ) } : \_ b o u n d a r y \_ p r o m p t _ { - }
$$

a context memory prompt with complementary guidelines, and,

$$
R _ { P ( n , i ) } : _ { \substack { \scriptscriptstyle { \it p o l i c y \_ c o n t e x t \_ } } }
$$

additional policy context units for a single call to $L L M _ { X }$

The user will first compose/import the $R _ { P }$ components he expects to forward to $L L M _ { X }$ ; here, he can add policy components with n policy context units towards setting the policy component $R _ { P } . \mathrm { \Delta A }$ context operation regarding policy assembles the $R _ { P ( i ) }$ component with several units $R _ { P ( n , i ) }$ , and sends it to $L L M _ { X }$ at the request of the user. The user can compose/import the global prompt $G _ { s p }$ to request the LLM to follow a set of rules, without adding a boundary prompt. Here, the call to $\mathsf { \bar { L } L } M _ { X }$ takes place by the (I) message line, and $G _ { s p }$ will be forwarded at all times.

$$
\underbrace { G _ { s p } } _ { \mathrm { g l o b a l } } \in R _ { P ( i ) } \qquad \underbrace { B _ { s p ( i ) } } _ { \mathrm { b o u n d a r y } } \in R _ { P ( i ) }\tag{6}
$$

The user can also compose/import the boundary prompt $B _ { s p }$ as complementary guidance to $L L M _ { X }$ alongside $G _ { s p }$ to the $R _ { P ( i ) }$ policy component. Here, the call to $L L M _ { X }$ takes place by the (II) message line, and $\boldsymbol { B _ { s p } }$ context component can be forwarded along the $G _ { s p }$ global prompt. Here, $R _ { P ( i ) }$ can carry the single global prompt, or both global and boundary prompts, and can include further units $R _ { P ( n , i ) }$ intended for a similar role.

These complement each other regarding the order in which they are assembled: $G _ { s p }$ conveys role, background, attitude and style directives for $L L M _ { X }$ , whereas $B _ { s p ( i ) }$ forwards guidelines to how $L L M _ { X } ^ { ' }$ shall assemble the answer such as work process, topic structure, stylesheet and generic codeblock format as applicable. Then, the user can compose/import several $' _ { n ^ { \prime } }$ policy units $R _ { P ( n , i ) }$ to complement the policy component $R _ { P ( i ) }$ for a single call to $L L M _ { X }$

## 6.2.2 Imported references

One can import reference context units to provide basis for the workings of $L L M _ { X }$ function, to guide the process at a closer approximation to the intended reasoning mechanism for the query. These context units will work as reference basis to support the reasoning process by the $L L M _ { X }$ function to process the core question. In an ordered context workload, the references are intended for placement between the policy units and the question units. Then, the professional user can compose the following string units:

• P<sub>P(i)</sub> − \_reference\_component\_:

the reference component added to the context intake towards $L L M _ { X }$

$P _ { p ( i ) } -$ \_single\_reference\_ :

an individual reference unit that will be directly added to the context,

$$
\_ P P _ { \mathit { P } ( p , i ) } - \_ s e \nu e r a l \_ r e f e r e n c e s \_ :
$$

several $' p ^ { \prime }$ reference units that will be assembled together and added to the context,

The user will first import the context unit(s) he expects to forward to $L L M _ { X }$ ; here, he can add a single $P _ { p ( i ) }$ context unit, or can add several $P _ { P ( p , i ) }$ reference units to the context workload. Here, $P _ { P ( i ) } \left( I \right)$ considers the import of a single reference unit $P _ { p ( i ) } .$ whereas $\dot { P } _ { P ( i ) } \left( I I \right)$ considers the import of several reference units $P _ { P ( p , i ) }$ . The assembled context is then sent to $L L M _ { X }$

$$
\begin{array} { r } { P _ { P ( i ) } = \underbrace { P _ { p ( i ) } } _ { \mathrm { s i n g l e } } ~ \left| ~ \begin{array} { l } { P _ { P ( i ) } = \left| \right| _ { p = 1 } ^ { P } \underbrace { P _ { P ( p , i ) } } _ { \mathrm { r e f . ~ u n i t } } } \end{array} \right| \forall ~ p \in \mathbb { N } } \end{array}\tag{7}
$$

These context units can be individually composed/imported by the user at anytime before the call. Here, an imported context unit $P _ { p ( i ) }$ serves as reference towards the call $^ { \prime } i ^ { \prime }$ . to $\dot { L } L M _ { X }$ , according to the equation 7 above. At the same time, the reference component can include several $' p ^ { \prime }$ context units towards the call $^ \prime _ { i ^ { \prime } }$ , from the first to the final $' P _ { t h } ^ { \prime }$ reference, each making a context unit $P _ { P ( p , i ) }$ to $L L M _ { X }$

The user makes all considerations of purpose and ordering about importing the reference units, one or multiple at a time. The effects of the formulation of the reference component $P _ { P ( i ) }$ are only constrained by the limit within the LLM context window as defined by the selected $L L M _ { X }$ function; the context window limit affects the functionality considering references the answer $A _ { M ( i ) }$ , with effects explained in the section 3 of this paper.

## 6.2.3 Core question operations

The context assembly to $L L M _ { X }$ calls involves the crafting of a question for each request, in the form of a query statement that is intended to trigger attention by the model. This is a key component to steer the internals of the selected $L L M _ { X }$ function, as it defines the object of inquiry and thus the focus of the process. These context units, composed through the process in the Figure 5 will work as reference basis to support the reasoning process by the $L L M _ { X }$ function to process the core question.

Then, the professional user can compose the following string units:

$Q _ { P ( i ) }$ : \_core\_question\_

the question statement as elected by the user to call $L L M _ { X }$

– $Q _ { p ( i ) }$ : \_query\_statement\_

the query statement written by the user towards its intent for $L L M _ { X }$

$O _ { v ( i ) }$ : \_vectoring\_operator\_

a relationship operator clause to steer $Q _ { c ( i ) }$ onto specifics,

$Q _ { v ( m , i ) }$ : \_prompt\_vector\_

an aspect clause that adds a specific to $Q _ { c ( i ) }$ that $L L M _ { X }$ shall process,

![](images/dbe4ea2fbb30cae9afc635c12331280dde2ee9540db2915cff1e5cb6e7bd244a.jpg)  
Figure 5: Assembly operations to core question $Q _ { P ( i ) }$ for a single $L L M _ { X }$ call.

The user will first compose the question he expects to forward to $L L M _ { X } ;$ ; here, he can add the query statement $Q _ { p ( i ) }$ alone, or can set a prompt vectoring operation. with a vectoring operator $O _ { v ( i ) }$ with $' m ^ { \prime }$ prompt vectors to request $L L M _ { X }$ to answer the query statement in several specific aspects, even with the ability to associate $Q _ { x ( i , m , k _ { m } ) }$ exemplars to each vector. Once being set about the core question $Q _ { P ( i ) }$ before calling $L L M _ { X }$ , the user can proceed to perform the query.

The user needs to compose the core question $Q _ { P ( i ) }$ towards calling $L L M _ { X }$ . Here, the user can compose a single query statement $Q _ { c ( i ) }$ as in display by Equation 8. For that purpose, the query statement will contain guidewords such as ’what’, ’where’, ’how’, or elaborated requests such as with including ’please explain’ or ’I need to know’ clauses. These words play the role of attention-triggers telling $L L M _ { X }$ to focus on the specific aspect of the query statement, and to provide a response that is compliant with the intent of the user.

$$
Q _ { P ( i ) } = \underbrace { Q _ { c ( i ) } } _ { \mathrm { q u e r y } } ~ | \begin{array} { l } { { } } \\ { { } } \end{array} | ~ Q _ { P ( i ) } = | \begin{array} { l } { { } | _ { q = 1 } ^ { Q } \underbrace { Q _ { c ( i ) } } _ { \mathrm { q u e r y } } ~ | | ~ \{ ~ \underbrace { O _ { v ( i ) } } _ { \mathrm { o p e r a t o r } } ~ | | ~ \underbrace { Q _ { v ( q , i ) } } _ { \mathrm { v e c t o r s } } ~ | } _ { \mathrm { e r e c t o r s } } ~ | ~ \forall ~ q \in \mathbb { N } \end{array}\tag{8}
$$

At the same time, the user may figure the query statement can be enhanced by prompt vectors. Here, the question $Q _ { P ( i ) }$ as shown by equation 8 aggregates the clauses for $Q _ { c ( i ) }$ , and then the $O _ { v ( i ) }$ , repeated times to each $Q _ { v ( i , m ) }$ elements, and then $Q _ { P ( i ) }$ will ask the $\bar { L } L M _ { X }$ function to provide an aggregate answer considering all vectors according to Equation $\mathbf { 8 , }$ , under guidance by the rulework first provided in $R _ { P ( i ) }$ and with reference to imported context units within $P _ { P ( i ) }$

The prompt vectoring formulation in Equation 8 applies to the assembly of a single request to $L L M _ { X }$ function, and enables comprehensive responses upon the capability of individual $L L M _ { X }$ functions. The individual answer provided by $L L M _ { X }$ will address all single specifics as defined in prompt vectors, because the query statement $Q _ { c ( i ) }$ and the operator $O _ { v ( i ) }$ are replicated at all times along each prompt vector in the core question component.

## 6.3 Context assembly operations

The assembly of context towards engineering tasks, from system design context definition to model-building and implementation, requires flexibility and modularity in designing and handling systems information as context input for use with large language models, through performing the context operations formalized in section 6. Table 1 lists the role-focused context units in the context assembly process as defined in the previous section, and the intent for each.

This setting of different context units with basis on roles enables the assembly of a context workload $C _ { Q }$ to be forwarded to the $L L M _ { X }$ function with mind to a certain intent towards a work product that shall be embodied by the means of the answer yield $A _ { M }$ . However, the effectiveness of the context workload $C _ { Q }$ depends on the assembly of the context units, which requires understanding the dependencies between them.

The reason for this understanding lies in the dependency relations between context units as expressed by the context operation definitions from section 6: the joining and assembly of context unities takes place by means of concatenation; the transformer mechanism within large laguage models Vaswani et al. [2017] is such that the order of the context units in the workload matters, and the dependencies between them are relevant to the answer yield $A _ { M }$

This means dependencies between context units must respected when assembling the context workload $C _ { Q }$ Table 2 presents a Design Structure Matrix (DSM) capturing dependency relations between context units. The context assembly takes place before the call to $L L M _ { X }$ , from which the internals of the model generate the answer.

The DSM displays the correspondence and the resulting order of context units, from the upstream policy units, through the imported references placed in between, and the core question units downstream. Three principles steer this ordering sequence: (i) the order of context units in the workload matters due to the transformer mechanism; (ii) the dependencies between context units are relevant to the answer yield $A _ { M } ;$ and (iii) the memory of $L L M _ { X }$ ends up prioritizing the beginning and the end of the context workload.

A person in the role of systems design engineer curates and composes/imports the relevant context units, thus assembling the context workload, and then requests $L L M _ { X }$ . The chosen LLM will process the context workload $C _ { Q }$ with the parameters set in section 5.2 and yield its answer $A _ { M }$ under directives and guidelines set within the policy rulework module $R _ { P }$ , with basis on the $P _ { P }$ imported references, and supported by the exemplars set within $Q _ { P }$

Table 1: Context modules and units used in the assembly of the context workload forwarded to the LLM.
<table><tr><td></td><td>Context unit</td><td>Operation</td><td>Intent</td></tr><tr><td> ${ \pmb R } _ { P }$   $ G _ { s p }$ </td><td>Policy rulework Global prompt</td><td>section 6.2.1</td><td>Defines engineering role, epistemic rules,</td></tr><tr><td> $ B _ { s p }$ </td><td>Boundary prompt</td><td></td><td>and non-hallucination constraints Defines system scope to its</td></tr><tr><td></td><td></td><td></td><td>intended application and characteristics.</td></tr><tr><td> $P _ { P }$   $ P _ { P }$ </td><td>Reference priors Full References</td><td>section 6.2.2</td><td>Provides references with descriptions</td></tr><tr><td> $\to P _ { p }$ </td><td>Other references</td><td></td><td>and state-of-the-art technology. Extracts condensed system and design</td></tr><tr><td> $Q _ { P }$ </td><td>Core question</td><td>section 6.2.3</td><td>constraints from imported sources.</td></tr><tr><td> $ Q _ { p }$ </td><td>Query statement</td><td></td><td>Sets primary architectural inquiry.</td></tr><tr><td> $ O _ { v } ^ { - }$ </td><td>Vectoring operator</td><td></td><td>Field: Sets relationships between partial aspects and the main inquiry.</td></tr><tr><td> $ Q _ { v }$ </td><td>Prompt Vectors</td><td></td><td>Field(s): Defines specific aspects that</td></tr><tr><td> $C _ { Q } \Rightarrow$ </td><td>Context workload</td><td>section 6.3</td><td>the task shall consider. Context workload assembled</td></tr><tr><td> $L L M _ { X }$ </td><td>Large language</td><td></td><td>from context operations.</td></tr><tr><td></td><td>model</td><td>section 5.2</td><td></td></tr><tr><td> $\Rightarrow A _ { M }$ </td><td>Answer yield</td><td></td><td>LLM-generated content that considers the</td></tr></table>

Table 2: DSM Representing context assembly, $L L M _ { X }$ event horizon and answer yield with memory.  
LEGEND:

<table><tr><td></td><td> $G _ { s p }$ </td><td> $B _ { s p }$ </td><td> $R _ { P }$ </td><td> $P _ { r }$ </td><td> $P _ { r }$   $P _ { P }$ </td><td></td><td> $Q _ { c }$ </td><td> $O _ { v }$ </td><td> $Q _ { v }$ </td><td> $Q _ { P }$ </td><td> $C _ { Q }$ </td><td> $L L M _ { X }$   $A _ { M }$ </td></tr><tr><td> $G _ { s p }$ </td><td>一</td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $B _ { s p }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $R _ { P }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $P _ { r }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $P _ { r }$ </td><td></td><td></td><td></td><td>●</td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $P _ { P }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $Q _ { p }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td>一</td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $O _ { v }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $Q _ { v }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $Q _ { P }$ </td><td></td><td></td><td></td><td></td><td></td><td>●</td><td>●</td><td></td><td></td><td></td><td>1</td><td></td></tr><tr><td> $C _ { Q }$ </td><td>→</td><td>→</td><td>二</td><td>→</td><td>→</td><td>ⅡI</td><td>→ →</td><td>→</td><td>II</td><td>一</td><td>一</td><td></td></tr><tr><td> $L L M _ { X }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td></tr><tr><td> $A _ { M }$ </td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td></td><td>→ ●</td><td>一</td><td>(●)</td></tr></table>

• Row items succeeding column items. → Concatenation of items onto modules. || Operations on context modules

## 7 Case study

This section demonstrates the outputs of system configuration requests enabled by the context operations formally defined in section 6. This study focuses the assembly of context units to support the processing of $L L M _ { X }$ functions at building component architecture models of complex systems. To demonstrate how context operations work onto supporting these functions, this paper involves a modelling case regarding the component architecture of a hybrid SAR UAV, which is hereby denoted as Suthern-Cross as in display by Table 3.

Table 3: Context operations demonstration assignments regarding systems characteristics
<table><tr><td></td><td>Heavy SAR UAV - Suthern-Cross</td></tr><tr><td>Assignment</td><td>Operation: Search-and-rescue operations to lift and recover victims from flash flood to medical care.</td></tr><tr><td>Key requirements</td><td>Station-keeping on hover, limited downwash over victims, low noise footprint, aircraft regulatory compliance.</td></tr><tr><td>Key constraints</td><td>Heavy rainfall at location, aggressive wind and gusts, timespan to victim acquisition, energy and power budget.</td></tr><tr><td>LLM Assignment</td><td>Help functional decomposition and architecture design under professional supervision towards defining system/concept architecture.</td></tr></table>

![](images/f05082374dda6de6572b6ff15b09182051ed0d18f3c0e424faddc220af17f015.jpg)  
Picture source: https://youtu.be/QVyptRhdDgA?si=ygtpMefwQXCHfO54&t=84, accessed on 2026-06-05.

Table 3 includes a depiction of an as-current Search-and-Rescue operation over flooded area with an helicopter - the SAR UAV is expected to have similar capability of recovering victims. The architecture modelling task is defined with the goal of creating a functional and operational system, starting from a system scope of required functionality. The focus of this case study is the leveraging of the modelling-as-code paradigm, where the modelling language is used to generate a model from the answer yield $A _ { M }$ provided by $L L M _ { X }$

## 7.1 Context workload treatments

The context treatments involve selective use of context units onto requesting $L L M _ { X }$ for the generation of the intended work product. The context units within the treatments follow the definitions set in section $\begin{array} { r } { 6 ; } \end{array}$ the role of individual context units towards the assembly of input to $L L M _ { X }$ is considered upon composing and importing information content, as in display by Table 4.

Table 4: Context workload treatments.
<table><tr><td> $G _ { s p }$ </td><td>Global</td><td> $\boldsymbol { B _ { s p } }$  Boundary</td><td>Pp Reference</td><td> $Q _ { c }$  Query</td><td> $O _ { v } | | Q _ { \imath }$  Vectors</td></tr><tr><td>Content</td><td>Role directives, Validation</td><td>Modelling scope, Generated mission scenario, principles, rules, benchmarking requirements</td><td>analysis.</td><td>Work product request, and requirements.</td><td>µ-Template subsystem exemplars.</td></tr><tr><td>T1</td><td></td><td></td><td></td><td>● ●&gt; 127 Tk</td><td></td></tr><tr><td>T2</td><td>•++&gt;</td><td>●++&gt;</td><td></td><td>● ● &gt; 2485 Tk</td><td></td></tr><tr><td>T3</td><td>•++&gt;</td><td>•++&gt;</td><td></td><td>●&gt;</td><td>•++&gt; 7645 Tk</td></tr><tr><td>T4</td><td>•++&gt;</td><td>•++&gt;</td><td>•++&gt;</td><td>● &gt;</td><td>•++&gt; 20555 Tk</td></tr><tr><td>Input</td><td>Typed/pasted prose &amp; topics, saveable.</td><td>Typed/pasted prose &amp; topics, saveable.</td><td>Pasted source text import or DOCX, PDF</td><td>Typed/pasted or imported prose &amp; topics</td><td>Paste into or import</td></tr></table>

The treatments are designed to be cumulative, where each treatment adds more context units to the workload<sup>2</sup> for $L L M _ { X }$ to process. The cumulation of treatments is such that the first treatment T1 only includes the query statement $\textstyle Q _ { c } ;$ the second treatment T2 adds the global and boundary prompts $G _ { s p }$ and $B _ { s p } ;$ the third treatment T3 adds the prompt vectors $O _ { v }$ and $Q _ { v } \mathrm { ; }$ ; and the fourth treatment T4 adds the reference priors $\bar { P } _ { P }$

The treatments are designed to evaluate the effects of different combinations of context units on the answer yield $A _ { M }$ from $L L M _ { X }$ , which are assembled with basis on the context operations defined in section 6. For that purpose, the experiment is based on examples of using plantUML as modelling language, considering LLM-aided modelling technique, such as by Cámara et al. [2023] and Krus [2024].

## 7.2 Model use treatments

Besides the four context treatments, the experiment also includes five $L L M _ { X }$ model treatments. The model assortment for this case study is designed to evaluate the effects of model characteristics on the answer yield $A _ { M }$ from $L L M _ { X }$ receiving the same context workload. The LLM assortment is designed to evaluate the effects of model characteristics on the answer yield $A _ { M }$ from $L L M _ { X }$ receiving the same context workload.

The LLM assortment was defined with mind to considering processing resource: one model running on local $\mathrm { G P U } ^ { 3 }$ two models from the cloud<sup>4</sup> that can be run on high-end local desktop GPUs to 48Gb RAM, and two frontier models to be run from the cloud. Table 5 displays the specifications, their descriptions and their units. The API interface forwards the assembled context workloads to the selected $L L M _ { X }$ models.

Table 5: Model structure metrics - characterization of models.
<table><tr><td rowspan="5"></td><td>M1</td><td>M2</td><td>M3</td><td>M4</td><td>M5</td></tr><tr><td>Qwen3 8b</td><td>ChatGPT OSS-20b</td><td>Nemotron3 super-120b-a12b K2.5</td><td>Kimi</td><td>Claude Sonnet4.6</td></tr><tr><td>Yang et al. [2025]</td><td>OpenAI [2025]</td><td>NVIDIA [2026]</td><td>Ollama [2026]</td><td>Anthropic [2026]</td></tr><tr><td>Local/GPU</td><td>Cloud/Nvidia</td><td>Cloud/Nvidia</td><td>Cloud/Ollama</td><td>Cloud/Anthropic</td></tr><tr><td>Resource Architecture</td><td>Distilled</td><td>MoE</td><td>MoE</td><td>MoE</td><td>Undisclosed</td></tr><tr><td>[Param.]</td><td> $8 \times 1 0 ^ { 9 }$ </td><td> $2 \times 1 0 ^ { 1 0 }$ </td><td> $1 , 2 \times 1 0 ^ { 1 1 }$ </td><td> $1 \times 1 0 ^ { 1 2 }$ </td><td></td></tr><tr><td>Context [Tk]</td><td> $3 , 2 7 \times 1 0 ^ { 4 }$ </td><td> $1 , 2 8 \times 1 0 ^ { 5 }$ </td><td> $1 , 0 0 \times 1 0 ^ { 6 }$ </td><td> $2 , 5 6 \times 1 0 ^ { 5 }$ </td><td> $1 , 0 0 \times 1 0 ^ { 6 }$ </td></tr><tr><td>Answer [Tk]</td><td> $4 , 0 9 \times 1 0 ^ { 3 }$ </td><td> $3 , 2 7 \times 1 0 ^ { 4 }$ </td><td> $6 , 4 0 \times 1 0 ^ { 4 }$ </td><td> $6 , 4 0 \times 1 0 ^ { 4 }$ </td><td> $6 , 4 0 \times 1 0 ^ { 4 }$ </td></tr></table>

The following characteristics affect modelling performance: (i) local hosting is limited to the memory room in the local machine; (ii) a higher scale of parameter count usually enables more complex and targeted processing; (iii) a longer context length capacity means the maximum number of tokens that can be processed by an individual LLM; and (iv) the answer length limit is less about capability, and more about policy - especially with higher-end models.

## 7.3 Modelling corpus requirements

The case study proceeds with the assumption that the modelling response can provide a work product the interpreter is able to render. Context workload components involve the same content set by context composition treatment regardless of the model called, so the resulting models can be compared across treatments and $L L M _ { X }$ models to evaluate the role of context operations and model scale.

In the modelling-as-code approach, the context workload conveys to $L L M _ { X }$ a request for a system model intended to represent the intended system architecture with sufficient detail and within a configuration that complies with required functionality. To enable the proper validation of the model outputs in the corpus of generated models across treatments, two sets of modelling requirements are defined:

a) plantUML syntax convention; and

b) architecture requirements defined in Table 6.

Considering the context workloads from Table 4 submitted to the $L L M _ { X }$ models from Table 5 include requests for plantUML cobeblocks as part of the answers $A _ { M }$ , the evaluation of the capability by the models to process context workloads takes place through the manual checking and validation of the codeblocks inside the answer files. For that purpose, the model outputs in conversation codeblocks are exported onto plantUML model code files, to be verified and corrected individually in the following steps:

1. Verify the model $m _ { 0 } \in A _ { M }$ in the answer codeblock from $L L M _ { X }$ with, duplicating it onto a $m _ { 1 } = m _ { 0 }$ plantUML model file for syntax review with the interpreter;

2. verify the $m _ { 1 }$ model file for syntax errors and architecture requirements, iterate corrections until the model is fully correct, then save the corrected model $m _ { 2 } \neq m _ { 1 }$ to a model file with Corr1 suffix;

3. Verify the corrections made to m on architectural requirements, amend it to comply with the requirements and save the corrected model $m _ { 3 } \neq m _ { 2 }$ to a model file with Corr2 suffix.

Then, the evaluation of the resulting corpus involves the review and validation of model-as-code units with metrics conveying quality requirements from Table 6.

Table 6: Modelling requirements for system architecture models.
<table><tr><td></td><td colspan="5">plantUML syntax rules</td></tr><tr><td>Model syntax</td><td>Unique component name_def definitions in entity syntaxes: [component_name], or component &quot;Name&quot;, or port &quot;PT&quot; under with unique as alias. component name_def.</td><td>Unique port name_def definitions in entity syntaxes: port PT within element,</td><td>Single-pair flows with cp1 - cp2:src-tgt source-sink definition and matching port links with correct arrows to</td><td>Flow direction signs and [format] params between dashes and no stray component or port name_defs</td><td>Traceable, consistent component and port name_def assignment throughout component entity definitions to</td></tr><tr><td colspan="6">Component architecture requirements Model Single energy source Power take-off and</td></tr><tr><td>elements</td><td>with fuel specification and power plant with controls, power lines, drivetrain and end- effect components.</td><td>distribution with heat exchange components, end-effects linked to structural elements</td><td>Safety and payload systems including parachute, hoist, winch, cabling, load cell and control with</td><td>Flight control systems including sensory and processing components, along connectivity and controls to ancillary &amp; end-effect actuators.</td><td>Flight support and, and onboard mission controls and data processing with communication systems and protocols.</td></tr><tr><td>Model ports/flows</td><td>Component name_def shall have one or more individual name_def ports within or under name_def statement.</td><td>name_def flow links to same name_def ports in single component at source and in single component at sink.</td><td>There is no stray name_def component and no port/flow name_def mismatch by either flow end.</td><td>Flow linetype setting within brackets in flow definitions between single pair of source and sink components.</td><td>Single flow connections between connected pairs of name_def ports by source and sink name_def components.</td></tr><tr><td>Model compliance</td><td>Power system energy flows from single source to countable end-effect components and outputs.</td><td>Control system flows from sensor elements through controller elements and actuator</td><td>One-way source-to-sink Control architecture flow chains for control and power systems with acting components to no circular paths across</td><td>across signal flows by ensure functionality and control modes.</td><td>Specific quantification of components about energy sources and end-effect assemblies</td></tr></table>

The inclusion of exported plantUML model files enables the verification of the outputs regarding the requirements above, on the following order: raw → syntax → architecture. A model is considered correct if it complies with the syntax and architecture requirements, and it is considered incorrect if it does not comply with either of them. The variety of models from Table 5 will determine different capabilities regarding the satisfaction of the requirements, which means each pair context-model may yield different levels of compliance with the requirements.

## 7.4 Modelling answer verification

LLM limitations can produce a representation with insufficient detail and lacking compliance to system requirements. Sometimes, the feeding system model codeblocks to the plantUML interpreter fails to render the model correctly because of syntax errors. Moreover, incorrect model representations manifest in differences between raw models from $L L M _ { X }$ output and corrected models to syntax. In this context, the metrics reflect findings on the architectural model about whether it misses any required element or whether it does not comply to rules. The metrics proposed for this case study come in display by Table 7.

Table 7: Model metrics considered for architecture validation.
<table><tr><td>Generative modelling outputs</td><td colspan="5"></td></tr><tr><td>Model answer</td><td>Answer length [tk]  $A _ { M _ { l e n } }$   $\in A _ { M }$ </td><td>Context utilization Answer time  $[ \mathrm { t k } ] C _ { L _ { l e n } }$   $/ [ \mathrm { t k } ] C _ { M _ { l e n } }$ </td><td>[min:s] tA  $C _ { Q }  A _ { M }$ </td><td>Model length [lin]  $m _ { l e n }$   $\in A _ { M }$ </td><td>Model type  $[ \mathrm { t y p e } ] m _ { t y p e }$  ∈ plantUML</td></tr><tr><td>Model</td><td colspan="5">plantUML syntax rules Model versions Element syntax</td></tr><tr><td>syntax</td><td>[n]  $m _ { x } \ \mathrm { o f f } \ A _ { M }$ </td><td>[n]  $e _ { e r r } \notin [ \mathrm { t y p e } ]$  ∈ plantUML</td><td>Flow syntax [n]  $f _ { e r r } \notin [ \mathrm { t y p e } ]$  ∈ plantUML</td><td>Port syntax [n]  $p _ { e r r } \notin [ \mathrm { t y p e } ]$  ∈ plantUML</td><td> $[ n ] e _ { a } = e _ { b }$   $\ge 2 x \in A _ { M }$ </td></tr><tr><td colspan="5">Component architecture requirements</td></tr><tr><td>Model elements</td><td>Element count [n]  $e _ { x } \in A _ { M }$ </td><td>Element miss [n]  $e _ { x } \in D _ { C }$   $\notin A _ { M }$ </td><td>Stray element  $[ \boldsymbol { \mathrm { n } } ] e _ { x } \in D _ { C }$   $\nexists f _ { x _ { i n } } , f _ { x _ { o u t } }$ </td><td>Module miss  $[ \boldsymbol { \mathrm { n } } ] g _ { x } \in D _ { C }$   $\notin A _ { M }$ </td><td>No end-effect [n]  $z _ { x } \in D _ { C }$   $\notin A _ { M }$ </td></tr><tr><td>Model ports/flows</td><td>Flow count  $[ { \boldsymbol { \mathrm { n } } } ] \in A _ { M }$ </td><td>Flow miss [n]  $f _ { x } \in D _ { C }$   $\notin A _ { M }$ </td><td>Port dangle  $[ \boldsymbol { \mathrm { n } } ] p _ { x } \in \mathbf { \bar { \cal D } } _ { C }$   $\nexists f _ { x _ { i n } } , f _ { x _ { o u t } }$ </td><td>Port count  $[ \boldsymbol { \mathrm { n } } ] \in A _ { M }$ </td><td>Port miss [n]  $p _ { x } \in D _ { C }$   $\notin A _ { M }$ </td></tr><tr><td>Model compliance</td><td></td><td>Input misses [n]  $f _ { x _ { o u t } } \neq p _ { x _ { i n } }$ </td><td>Chain misses [n]  $\{ e _ { x }  e _ { y }  e _ { z } \}$ </td><td>Output misses [n]</td></tr></table>

The model answer metrics in the first row quantify characteristics of the answer provided by $L L M _ { X }$ to certain context workload treatment. The other metrics are designed to quantify the characteristics of the model outputs from $L L M _ { X }$ regarding syntax and architecture requirements. These include $A _ { M }$ processing statistics by $L L M _ { X }$ , then all model verification criteria - syntax and architecture - apply at the raw model codeblock $m _ { 0 }$ exported to the model file $m _ { 1 }$

Equation 9 expresses the relationship between the context workload $C _ { Q }$ and the answer yield $A _ { M }$ from $L L M _ { X }$ , where the answer length $A _ { M _ { l e n } }$ is less than or equal to the context length $\dot { C } _ { Q \iota _ { e n } }$ , and the model length $m _ { l e n }$ is less than or equal to the answer length $A _ { M _ { l e n } }$

$$
\begin{array} { r }  \underbrace { L L M _ { X } } _ { C _ { M _ { l e n } } \geq } \ \underbrace { \{ C _ { Q ( i ) } \} = A _ { M ( i ) } } _ { C _ { Q _ { l e n } } + A _ { M _ { l e n } } = C _ { L _ { l e n } } } \quad | \begin{array} { l } { \underbrace { m _ { 0 } \in \ A _ { M ( i ) } } _ { m _ { l e n } < A _ { M _ { l e n } } } \quad | \begin{array} { l } { \underbrace { m _ { 1 } = m _ { 0 } } } \\ { m _ { l e n } < A _ { M _ { l e n } } } \end{array}  } \end{array} \end{array}\tag{9}
$$

Once a $m _ { 1 }$ model file is available, its verification is carried out on the following basis: the syntax of the model is checked for errors, and then the architecture of the model is evaluated for compliance with the requirements. This is done with the checking the following assertions on the model: the assertion (I) in Equation 10 regards the existence of syntax errors in the model $m _ { 1 }$ regarding element, flow and port definitions, and the second condition regards the existence of duplicate name\_def definitions for elements.

$$
\begin{array} { r } { \mathrm { I } { \colon \quad m _ { e r r } } = \{ \underbrace { \exists \left[ e _ { e r r } , ~ f _ { e r r } , ~ p _ { e r r } \right] \notin ~ \mathrm { p l a n t U M L } } _ { \mathrm { s y n t a x ~ e r r o r s } } \lor \underbrace { \forall ~ [ a , b ] \to e _ { a } = e _ { b } } _ { \mathrm { d u p l i c a t e ~ n a m e s } / \mathrm { a l i a s e s } } \} \in ~ m _ { 1 } } \end{array}\tag{10}
$$

Then, the assertion (II) in the first line of Equation 11 regards the architecture requirements, with the first condition being about any missing element, flow, port, module or end-effect in the model and the second condition being about any stray component or dangling port in the model. Then, the assertion (III) in the second line of Equation 11 is true if it does not comply with architecture requirements, which include any input and output mismatches in the model, or any chain of elements that is not present in the model.

$$
\begin{array} { r } { \mathrm { I I } { : \quad m _ { m i s s } = \{ \underbrace { \oint _ { \mathbb { \Lambda } } \left[ e _ { x } , \ f _ { x } , \ p _ { x } , \ g _ { x } , \ z _ { x } \right] \ \in \ D _ { C } } _ { \mathrm { c l e m e n t , \ f l o w , \ p o r t , \ m o d u l e ~ o r ~ e n d - f f e c t m i s s e s } } \} } \ \underbrace { \left[ e _ { x } , \ p _ { x } \right] \ \sharp \left[ f _ { x _ { i n } } \wedge f _ { x _ { o u t } } \right] } _ { \mathrm { s t r a y ~ c o m p o n e n t , \ f i a n g l i n g ~ p o r t } } \ \in \ m _ { x } } \end{array}\tag{11}
$$

$$
\begin{array} { r } { \mathrm { I I I } { : } \quad m _ { s h o r t } = \{ \underbrace { \exists [ f _ { x _ { o u t } } \neq p _ { x _ { i n } } \lor p _ { x _ { o u t } } \neq f _ { x _ { i n } } ] } _ { \mathrm { I n p u t a d o u p u t ~ m i s m a t c h e s } } \lor \underbrace { \exists [ e _ { x }  e _ { y }  e _ { z } ] \notin D _ { C } } _ { \mathrm { C h a i n m i s s c s } } \} \in m _ { x } } \end{array}
$$

The verification of architectural metrics is preconditioned by correct model syntax.This means that the model-building attempts from context workload may display different inconsistencies with requirements - different context workloads entail levels of specification to modelling-as-code LLM outputs regarding those criteria. Then, Equation 12 describes the conditions under which additional model files are needed.

$$
\mathrm { I } { : } \quad \mathrm { { C o r r } } 1 = \left\{ I \Rightarrow \exists m _ { 2 } \neq m _ { 1 } \right\} \wedge \ \mathrm { { C o r r } } 2 = \left\{ \left[ I \wedge \left( I I \vee I I I \right) \right] \Rightarrow \exists m _ { 3 } \neq m _ { 2 } \right\}
$$

$$
\mathrm { I I } \colon \ \mathrm { C o r r } 1 = \ \left\{ \left[ \lnot I \wedge \left( I I \vee I I I \right) \right] \Rightarrow \exists m _ { 2 } \neq m _ { 1 } , \sharp m _ { 3 } \right\}\tag{12}
$$

$$
\mathrm { I I I } { : } \quad \vec { \mathbb { H } } \left( \mathrm { C o r r 1 } , \ \mathrm { C o r r 2 } \right) : \ \left\{ \left[ \neg I \wedge ( \neg I I \wedge \neg I I I ) \right] \Rightarrow \vec { \mathbb { H } } m _ { 3 } , \ m _ { 2 } \right\}
$$

The first condition in Equation 12 applies to the case where the model $m _ { 1 }$ has syntax errors, which requires a new model file $m _ { 2 } \neq m _ { 1 }$ to be created with corrections to syntax and architecture. Here, the need to make additional model files m and m shall involve syntax and architectural refinements to satisfy the requirements; the corrections are ladder-staged ifrom $m _ { 1 }$ to $m _ { 2 }$ and then from $m _ { 2 }$ to $m _ { 3 } \neq m _ { 2 }$ . When $m _ { 1 }$ has correct syntax, then it undergoes architectural review of $m _ { 2 }$ to determine the need for $m _ { 3 }$

The second condition in Equation 12 applies to the case where the model $m _ { 1 }$ has correct syntax, but it does not comply with architecture requirements, which requires a new model file $m _ { 2 } \neq m _ { 1 }$ to be created with corrections to architecture; no further model instances will be created. The third condition applies to the case where the model $m _ { 1 }$ has correct syntax and complies with architecture requirements, which means no additional model files are required, and the architectural review can be carried out directly to the first model off the answer codeblock.

## 8 Results

The results of the case study are presented in this section, with a focus on the modelling outputs from the generative modelling approach supported by modular context assembly. The analysis includes selected examples of model responses by LLMs, as well as a comparative analysis of model length and element counts across different treatments and models. The generative modelling approach supported by modular context assembly has been experimented on 20 parallel model runs along the workload treatments from section 7.1 and the models called in section 7.2. With including the corrections needed, then the modelling corpus expands to 54 - fifty-four - model instances.

## 8.1 Modelling examples

To demonstrate the results from the case study, this section first presents selected examples of model responses by LLMs. The purpose of characterizing these examples is to illustrate the effectiveness of the generative modelling approach, and its characteristics in proportion to the context workload. This section should present and describe the resulting models from the generative modelling approach with regard to the context workload treatments, and the LLMs called.

Figure 6 displays the model produced by (M5) Claude Sonnet 4.6 for the first treatment (T1) with no ancillaries to the question. This is a class model with system packages. This model has 38 high-level components with specifications in class properties (+) and 46 flows mostly expressing overall interdependence between elements. While possibly displaying a design specification of a heavy a SAR UAV, it deviates from the intent of a component architecture.

![](images/5ea5a7f4ef7252202849a00df4c85cc7df06432118cbe0d0be5cd0566b8c866c.jpg)  
Figure 6: Model requested from Claude Sonnet 4.6 for T1 context.

While the first treatment presents the question alone and therefore mostly counts on the weights from model pretraining to process the question intent, the quality of the first model is not sufficient; it does not include ports in components, neither it does specify component flows between them, then it is not a component architecture model and does not comply with the requirements in Table 6.

Then, the second treatment (T2) adds policy elements besides the question to the context workload, which is expected to improve the quality of the model output. For that purpose, the T2 treatment includes a policy statement with declaring role and directives by a $G _ { s p }$ global policy component and supportive modelling guidelines by the means of a $B _ { s p }$ boundary prompt component. An example model out from the T2 treatment is displayed in Figure 7, which is a model produced by (M4) Nemotron3-super for the second treatment.

![](images/da5b23e51d511abf5e9ba1577d62c377299d8b962ca52d962212e484ec3b886e.jpg)  
Figure 7: Model requested from Nemotron3-super for T2 context.

This model has 28 individual elements and 32 flows; power transmission and distribution elements appear with two sets of soft ports, combining lower-level components. The raw model did not render correctly due to syntax errors; Figure 7 displays the corrected Corr1 model. Ports are represented as nested components, and flows are expressed as single declarations between port pairs. There are a few dangling ports, such as QM\_in within the Airframe subsystem, and some combined components such as the generator, yet most of the model complies with the requirements in Table 6.

To support that purpose, treatments (T3) and (T4) involve the use of µ-Templates Krus [2024] as exemplars embedded in prompt vectors, with examples in figure 8a, 8b and 8c. These are aggregated to the query statement as prompt vectors following equation 8 to steer the processing of the LLM. The µ-Templates carry modelling-as-code snippets of the system components, ports and flows - plantUML in this case - to provide a reference so that the LLM shall produce a model with similar characteristics.

![](images/1d70ee763e29e25ffdb8f6456e697867bc904f4137967f2992081c3dbbab7711.jpg)  
Figure 8: System model errors from $L L M _ { X }$ answer products.

The use of µ-Templates is a key aspect of the tretments T3 and T4 in our experimental approach. To present an idea on how their use support the modelling process, Figure 9 displays the raw model produced by ChatGPT-OSS-20b for the third treatment. This T3 treatment involved the policy and prompt vector ancillaries (T3). This model has 28 individual elements and 32 flows; power transmission and distribution elements appear with two sets of soft ports – combining lower level components. Soft port declarations are expressed within the component statements, and flows are expressed as single declarations between port pairs.

![](images/37fd047a32711a6fd9ec4dc6cdd6e477b553e2d4f560042c17c0209f71c296f1.jpg)  
Figure 9: Model requested from GPT-OSS-20b for T3 context.

Then, Figure 10 displays a corrected model with basis on the one above. This corrective treatment involves layout modifications to the powertrain components, whereas the hoist system, electronics and communication systems remained the same, including the port-combination components, and the single flow declarations between port pairs. This means the resulting model is acceptable regarding its compliance to the requirements in Table 6, and it is considered a valid component architecture model, worth considering for check and approval in due engineering review process.

![](images/41f03c5d0ec90c73de586c77cc167a2e094d00abe8a85316399a5e762223c327.jpg)  
Figure 10: Model corrected from that of GPT-OSS-20b for T3 context.

The modelling results from the T3 treatment show that the use of µ-Templates as exemplars embedded in prompt vectors can improve the quality of the model outputs from LLMs. The models produced by LLMs with the T3 treatment have more complete and accurate representations of the system components, ports, and flows, and they comply better with the requirements in Table 6. Ultimately, the T4 treatment yields best results by a marginal difference with support of extra references; their effectiveness is somehow limited by the LLMs’ ability to process and integrate the additional information, considering the increased context utilization and the effect from the bias to missing the middle.

## 8.2 Modelling analysis

This section is intended to present the results of the modelling experiment over the whole corpus and discuss its outcome with basis on the metrics in Table 7. The analysis is carried out with respect to the context workload treatments and the LLMs called, by the means of a python tool that implements the assertions in Equations 10, 11 and 12. The tool is able to parse the model files and check for syntax errors, missing elements, flows, ports, modules, end-effects, stray components, dangling ports, input/output mismatches, and chain misses. The tool also counts the number of elements, flows, ports, modules, and end-effects in the model files.

Figure 11 displays the result of model length from processing context workload. Here, policy (T2) has mostly increased the modelling output from $L L M _ { X }$ with exception of the top frontier model (M5). The introduction of µ-Templates as prompt vectors into the question (T3) reduced the model length; the addition of reference (T4) did not have so significant effect, which means the added references were not so effective in steering the model output.

![](images/483c6053bce19d684a9062c508167124eb9263d91b3c359ed5a3a9d8a9c8ba7b.jpg)  
Figure 11: Model length as result of processing context workloads.

Then, an increase in element count from figure 12a with the introduction of policy (T2) and prompt vectors (T3) corresponds to a decrease in sub-element count from figure 12b in these treatments. Sub-elements are counted when system component statements include several component names from the vocablulary in a single entity. Then, the counts did not change significantly from treatment (T3) to treatment (T4) showing the major effect of µ-Templates in steering the representation of system architecture.

![](images/00701925eb5c03ff46f5855a546311868d157a0473e4bcf43006b48a6b1625ec.jpg)  
(a) Element count by workload treatment.

![](images/31f1aa42969ec439ea60246463a32c924fd091091847306ad2f713fe5e1de809.jpg)  
(b) Subelement by workload treatment.  
Figure 12: System model errors from $L L M _ { X }$ answer products.

The results from the plots above are detailed in pairs regarding context workload treatments T1 & T2 in Table 8 and treatments T3 & T4 in Table 9. The differences in these two sets of treatments lie in the context workload; first, that T1 and T2 treatments do not include µ-Templates, while T3 and T4 treatments do; consequently, the context workload length is larger for T3 and T4 treatments as result of incorporating several µ-Templates as exemplars in the prompt vectors.

Table 8 displays the metrics for evaluating the answer yields from $L L M _ { X }$ on a workload that includes the single question preceded by the policy components, global and boundary prompt with different context workloads and models. Processing time differs sharply between GPU-ran and cloud-processed models, because model-processing is contrained to local GPU capacity whereas cloud-based models have access to more extensive computational resources. Then, the workload expands from 99 to 1164 tokens, thereby setting a higher load to each model’s context capacity.

Model type: Cmp – component diagram; <sup>\*</sup>Cmp – packaged components (components grouped across multiple subsystem packages); Cls – class diagram.   
Element type: Sub – subsystem element; Unit – unit element; Spec – named property/rating; Port – connection in code; Port<sup>\*</sup> – connection in element.   
Flow type: Fu – function; Ct – containment; Dv – deliverable; Fw – port-matching flow; Fu-Ct – mixed function/containment; n/a – unlabeled flows.   
Counts: length in code lines; Element/Subelement/Flow counts as [:n].   
Audit.

Table 8: Metrics for evaluating LLM<sub>X</sub> answer yields on single question, treatments T1 and T2.
<table><tr><td rowspan="2">T1 &amp; T2 Context workloads</td><td colspan="2">Qwen3 8b</td><td colspan="2">ChatGPT OSS-20b</td><td colspan="2">Nemotron3 super-120b-a12b</td><td colspan="2">Kimi K2.5</td><td colspan="2">Claude Sonnet4.6</td></tr><tr><td>T1</td><td>T2</td><td>T1</td><td>T2</td><td>T1</td><td>T2</td><td>T1</td><td>T2</td><td>T1</td><td>T2</td></tr><tr><td>Answer [Tk]</td><td>1207</td><td>2038</td><td>1039</td><td>1718</td><td>1241</td><td>1790</td><td>1600</td><td>3080</td><td>4039</td><td>5185</td></tr><tr><td>Utilization [%]</td><td>4,07</td><td>13,81</td><td>3,56</td><td>12,83</td><td>2,09</td><td>6,52</td><td>0,67</td><td>2,17</td><td>0,42</td><td>0,77</td></tr><tr><td>Time [min: sec]</td><td>6:19</td><td>12:42</td><td>0:12</td><td>0:41</td><td>1:31</td><td>2:34</td><td>0:14</td><td>1:01</td><td>1:37</td><td>1:48</td></tr><tr><td>Model type/ Length [lines]</td><td>Cmp 89</td><td>Cmp 150</td><td>*Cmp 141</td><td>Cmp 157</td><td>*Cmp 81</td><td>Cmp 180</td><td>*Cmp 122</td><td>*Cmp 270</td><td>Cls 403</td><td>*Cmp 374</td></tr><tr><td>Element type/</td><td>Sub</td><td>Sub</td><td>Unit</td><td>Sub</td><td>Sub</td><td>Unit</td><td>Sub</td><td>Sub</td><td>Sub</td><td>Unit</td></tr><tr><td>count [:n] Flow type/</td><td>12 Fu-Ct</td><td>12 n/a</td><td>39 n/a</td><td>13 Fw-Fu</td><td>32 Fu-Ct</td><td>21 Fw-Fu</td><td>35 Fu-Ct</td><td>44 Fw-Fu</td><td>48 Fu-Ct</td><td>50 Fw-Fu</td></tr><tr><td>count [:n] Subelements/</td><td>22</td><td>43</td><td>32</td><td>32 Port</td><td>24 Spec</td><td>40 Port</td><td>30 Spec</td><td>41</td><td>46</td><td>101 Port*</td></tr><tr><td>count [:n]</td><td>=</td><td></td><td></td><td>38</td><td>24</td><td>58</td><td>29</td><td>Spec 36</td><td>Spec 145</td><td>170</td></tr><tr><td>Elements/ Ports/</td><td>12</td><td>12</td><td>39</td><td>13</td><td>32</td><td>21</td><td>35</td><td>44</td><td>48</td><td>50</td></tr><tr><td>Flows [:n]</td><td>0 22</td><td>0 43</td><td>0 32</td><td>38 32</td><td>0 24</td><td>58 40</td><td>0 30</td><td>0 41</td><td>0 46</td><td>170</td></tr><tr><td>Syntax</td><td>none</td><td>none</td><td>none</td><td>Fl: 16</td><td>none</td><td>Fl: 40</td><td>none</td><td>F1: 30</td><td>none</td><td>101 none</td></tr><tr><td>misses [Types/:n]</td><td></td><td></td><td></td><td></td><td></td><td>Pt: 2</td><td></td><td></td><td></td><td></td></tr><tr><td>Element misses [Types/:n]</td><td>23</td><td>22</td><td>14</td><td>13</td><td>10</td><td>16</td><td>12</td><td>15</td><td>7</td><td>12</td></tr><tr><td>Port/flow mismatches [Types/:n]</td><td>Ph: 1</td><td>none</td><td>Orph: 3</td><td>Dang: 38</td><td>Orph: 1</td><td>Dir: 7</td><td>Orph: 2 Ph: 1</td><td>Orph: 2</td><td>Orph: 1</td><td>Dang: 9</td></tr></table>

Legend.  
Syntax misses are render-blocking errors (those requiring a syntax-only Corr1 pass): Fl – flow statement, Pt – port statement, El – declaration.  
Element misses counted strictly vs the SAR reference checklist (per fine-grained element, of 32).  
Mismatches: Orph – declared element, no flow; Dang – declared, unwired; Dir – direction/port error; Ph – flow to/from undeclared id.

Answer size varies substantially, from 466 tokens (Qwen3 8B) to 1,984 tokens (Claude Sonnet 4.6), indicating different expansion behavior even under comparable prompting conditions. Model-output structure becomes richer in larger responses, as seen in line counts (31 to 195) and flow counts (12 to 56), suggesting greater elaboration of system interactions. Processing time also differs sharply, with local/smaller models taking longer in this setup (up to 12:42) while larger cloud models respond faster (down to 0:14).

Besides computing hardware itself, the cloud processing time is determined by the scale of processing resource and the bandwidth of the cloud connection. Regarding the answer output, model-output structure becomes richer in larger responses, as seen in line counts (31 to 195) and flow counts (12 to 56), suggesting greater elaboration of system interactions.

Overall, the results suggest that higher-capacity models improve output depth and structural detail, while all models remain lightly loaded in context usage for these treatments. This means that the answers depend more of the model pretraining than of the context workload, and thereby the model outputs are more diverse in structure and content.

Table 9 displays the metrics for evaluating the answer yields from LLM<sub>X</sub> on a workload that includes the single question preceded by the policy components, global and boundary prompt with different context workloads and models; µ-templates are included in prompt vectors along the context workload in both T3 and T4, whereas T4 also includes additional context from references to previous flash-flood events and SAR resources involved.

The workload expands to 7645 tokens and then to 20555 tokens, thereby setting a significantly higher utilization of each model’s context capacity by the incoming context workload.

Table 9: Metrics for evaluating LLM<sub>X</sub> answer yields on single question, treatments T3 and T4.
<table><tr><td rowspan="2">T3 &amp; T4 Context workloads</td><td colspan="2">Qwen3 8b</td><td colspan="2">ChatGPT OSS-20b</td><td colspan="2">Nemotron3 super-120b-a12b</td><td colspan="2">Kimi K2.5</td><td colspan="2">Claude Sonnet4.6</td></tr><tr><td>T3</td><td>T4</td><td>T3</td><td>T4</td><td>T3</td><td>T4</td><td>T3</td><td>T4</td><td>T3</td><td>T4</td></tr><tr><td>Answer [Tk]</td><td>1285</td><td>1245</td><td>1808</td><td>2095</td><td>1997</td><td>2425</td><td>3189</td><td>2964</td><td>3897</td><td>3379</td></tr><tr><td>Utilization [%]</td><td>27,25</td><td>66,53</td><td>28,85</td><td>69,12</td><td>14,71</td><td>35,06</td><td>4,23</td><td>9,19</td><td>1,15</td><td>2,39</td></tr><tr><td>Time [min: sec]</td><td>8:57</td><td>8:42</td><td>0:15</td><td>0:26</td><td>0:20</td><td>2:49</td><td>0:25</td><td>0:23</td><td>1:39</td><td>1:30</td></tr><tr><td>Model type/ Length [lines]</td><td>Cmp 60</td><td>*Cmp 107</td><td>*Cmp 118</td><td>*Cmp 140</td><td>*Cmp 水 132</td><td>*Cmp 183</td><td>*Cmp 240</td><td>*Cmp 191</td><td>Cmp 262</td><td>*Cmp 217</td></tr><tr><td>Element type/ count [:n]</td><td>Unit 24</td><td>Unit 33</td><td>Unit 34</td><td>Unit 35</td><td>Unit 51</td><td>Unit 58</td><td>Unit 73</td><td>Unit 66</td><td>Unit 71</td><td>Unit 66</td></tr><tr><td>Flow type/ count [:n]</td><td>Fw-Fu 21</td><td>Fw-Fu 36</td><td>Fw-Fu 32</td><td>Fw-Fu 49</td><td>Fw-Fu 70</td><td>Fw-Fu 70</td><td>Fw-Fu 109</td><td>Fw-Fu 82</td><td>Fw-Fu 108</td><td>Fw-Fu 74</td></tr><tr><td>Subelements/ count [:n]</td><td>Port 23</td><td>Port 29</td><td>Port 28</td><td>Port 30</td><td>Port 46</td><td>Port 53</td><td>Port 66</td><td>Port 59</td><td>Port 68</td><td>Port 55</td></tr><tr><td>Syntax misses [Types/:n]</td><td>none</td><td>none</td><td>Fl: 8</td><td>Pre: 1</td><td>none</td><td>none</td><td>none</td><td>Fl: 2</td><td>none</td><td>none</td></tr><tr><td>Element misses [Types/:n]</td><td>16</td><td>13</td><td>13</td><td>9</td><td>2</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td></tr><tr><td>Mismatches [Types/:n]</td><td>none</td><td>none</td><td>none</td><td>none</td><td>Ph: 3 Dup: 8</td><td>none</td><td>none</td><td>none</td><td>none</td><td>none</td></tr></table>

Legend.  
Model type: Cmp – component diagram; <sup>\*</sup>Cmp – packaged components (components grouped across multiple subsystem packages); Cls – class diagram. Element type: Sub – subsystem element; Unit – unit element; Spec – named property/rating; Port – connection in code; Port<sup>\*</sup> – connection in element. Flow type: Fu – function; Ct – containment; Dv – deliverable; Fw – port-matching flow; Fu-Ct – mixed function/containment; n/a – unlabeled flows. Counts: length in code lines; Element/Subelement/Flow counts as [:n]. Audit.  
Syntax misses are render-blocking errors (those requiring a syntax-only Corr1 pass): Fl – flow statement, Pt – port statement, El – declaration. Element misses counted strictly vs the SAR reference checklist (per fine-grained element, of 32).  
Mismatches: Orph – declared element, no flow; Dang – declared, unwired; Dir – direction/port error; Ph – flow to/from undeclared id.

Processing time also differs sharply, with local/smaller models taking longer in this setup (up to 08:21) while larger cloud models respond faster (down to 0:30). The trends between local and cloud models are consistent with the overall performance patterns observed, yet the presence of examples and references seemed to enable better performance to timing of local models.

Answer size is somehow correlated with the parameter complexity of the models in use, with Qwen3-8b yielding 1300 tokens and the frontier models, Kimi and Claude yielding 3000 tokens.Model-output structure becomes richer in larger responses, as seen in line counts (31 to 195) and flow counts (21 to 109), suggesting greater elaboration of system interactions and interconnections in larger models - the interdependencies between component functions may appear more pronounced from larger models.

Overall, the results suggest that higher-capacity models improve output depth and structural detail, while all models remain lightly loaded in context usage for this experiment. At the same time, fromtier models manage to yield more complete and accurate representations of the system components, ports, and flows, and the results from the Table show that they perform better at leveraging context workload onto complying with the requirements in Table 6.

The most significant gain in model quality is observed when µ-Templates are used in conjunction with policy and prompt vector ancillaries (T3), as compared to the baseline treatments (T1 and T2). Then, the lost-in-the-middle effect limits the effectiveness of the additional references in T4, as the model’s attention will concentrate on the policy by the beginning of the workload and on the µ-Templates along the core question by the end of the workload.

There are significant effects from treatments regarding element misses and port/flow mismatches from $L L M _ { X }$ yields. Element misses were consistently high on smaller LLMs, and policy and guidelines along the question (T2) made component misses to increase. Then, µ-Templates as prompt vectors practically annulled component misses in (T3) and (T4) treatments.

## 8.3 Discussion of results

This paper presented modelling samples towards architecting a heavy SAR UAV system, a set of requirements and associated assertions on which the models are evaluated, and results from a quantitative analysis over the generated models. This case study experiment involved our running 20 queries spanning context workloads and selected LLMs that were requested to generate system architecture plantUML codeblock models.

The overall modelling answer corpus evolved from 20 treatment answer results to 55 individual model files, by counting the answer files plus the spawned models from the answer codeblocks upon the corrections required by the eval assertions. The analysis part involved setting up modelling metrics to verify the output of the LLMs; here, we focus element count and syntax requirements for the verification.

![](images/5eea4208d19ff667083242add90519ddb84c80674a48a3ba7ebda04bd3cf8469.jpg)  
Figure 13: AI-generated Heavy SAR UAV rendering (source: Gemini).

The whole dataset reveals the effect of the primacy and recency, characteristics of information units within a context workload to an LLM that were revealed by Liu et al. [2024] upon finding out on the limitations of LLMs under high utilization of their context length capacity. Here, smaller models such as Qwen3-8b (M1) and GPT-OSS-20b (M2), both with maximum 32767 Tk of context capacity, were significantly utilized – especially in treatments T3 and T4 with including µ-Template exemplars.

These effects have something to do with context units that precede the question having a divergence effect, and context units that are within the question, which actually succeed the query statement – that carries the actual request and thereby the modelling intent – and make both convergence and enforcement effects to expected patterns in both coding and modelling. This means that the effects of primacy and recency as discussed by Liu et al. [2024] manifest themselves when the context workload in single question becomes significant towards context length capacity.

While the performance of frontier models such as Kimi K2.5 (M4) and Claude Sonnet 4.6 (M5) in modelling architectures with support of µ-Templates – and that of similar-scale models such as Deepseek and ChatGPT-5.x from previous experience of ours in Marini and Krus [2025] – bears no remarks upon their higher scale and longer context capacities, µ-templates have a convergence effect to a desired model setup regarding its format, structure and compliance. In models such as Qwen3, GPT-OSS and Nemotron, the effects of context operations deserve attention.

Their performance comes not without their responses coming briefly, either with subcomponents declared within component blocks – which deviate from the purpose of architecting – or with omitting components entirely yet complying with code syntax and modelling conventions. Here, one expects a topology of components, effect flows and ports that supports component-level specification and designing with mind to later integration tasks. At the same time, such responses can be fixed to completeness, accuracy and compliance with a few minutes of juggling model code.

One significant matter of attention to be considered is the effect of different units onto the workload being forwarded to the LLMs. The workload units are the policy, the prompt vectors, the references and the question itself. The policy units are intended to set up a context for the modelling request, and they are expected to be processed by the LLMs as a primacy effect. The prompt vectors are intended to set up a context for the modelling request, and they are expected to be processed by the LLMs as a recency effect.

Here, references in the \*\*middle\*\* of the context workload are expected to be processed by the LLMs as a convergence effect, yet their effectiveness is limited by the lost-in-the-middle effect. Then, this means that policy and modelling directives, as well as examples, are the key elements on which the LLMs focus their attention to produce a modelling result. Nevertheless, in-context learning could be a powerful technique regarding the build-up of reference for modelling requests, whose usefulness goes down to sorting out the token economics over searching through a knowledge database

The use of evaluation assertions to verify the modelling results from LLMs is a key element onto enabling the usefulness of a modelling corpus onto the development of a learning database for systems modelling with generative artificial intelligence. Along with context handling, evaluations enable setting convergence criteria for modelling loops and ultimately enable the potential for the use of hybrid deterministic-probabilistic modelling approaches to systems design, such as proposed by Krus [2025] when requesting a deterministic model builder algorithm from a context prompt.

Understanding the capabilities of LLMs of a scale spectrum from giga to tera-scale enables proceeding to investigate and take advantage of novel context handling techniques to improve the capability of artificial intelligence onto systems modelling to design intent.

## 9 Conclusions

This contribution has presented a framework of context operations on which to prompt at large language models. Being implemented in a chatbox applicaiton as that shown by Marini and Krus [2025], this framework was successfully operated towards a case study of aircraft design, here being a heavy SAR UAV. This case study contributes with evolving from modelling examples towards the experimentation with modelling requests, and the verification of their modelling results through measuring their architectural properties.

The evaluation of modelling quality by large language models from the synthesis of modular context workloads enabled us to discuss on the modelling capability of LLMs under a single query. The modelling results presented from the experiment are on par with current understanding about the workings of LLMs to system design. At the same time, these modelling results enable the development of a learning database on how to support systems modelling with generative artificial intelligence.

Future work involves expanding the modular workload approach to conversation threading and agentic modelling, which in turn requires the expansion of the learning base to fine-tune model counting and validation rules.

## 10 Acknowledgments

This study has been carried out with the support of the CNPq-CISB grant no. 200944/2024-0 of the Brazilian National Council for Scientific and Technological Development (CNPq) and the Swedish-Brazilian Centre for Innovation, through collaborative research work carried out by the authors at the Federal University of Santa Maria (UFSM) in Brazil and at the Linköping University (LiU) in Sweden.

The authors manifest their gratitude in advance to reviewers for their valuable comments and suggestions, which help improve the quality of this paper.

## References

Mary L. Cummings. Automation bias in intelligent time critical decision support systems. In : 1st AIAA Intelligent Systems Technical Conference. Chicago, IL: American Institute of Aeronautics and Astronautics, 2004. doi:10.2514/6.2004-6313.

Martin Törngren and Paul T. Grogan. How to deal with the complexity of future cyber-physical systems? Designs, 2(4): 40 pp., 2018. ISSN 2411-9660. doi:10.3390/designs2040040.

Paul T Grogan. Perception of Complexity in Engineering Design. Systems Engineering, 24(4):221–233, 2021. ISSN 1520-6858. doi:10.1002/sys.21574.

Tom Brown, Benjamin Mann, Nick Ryder, Melanie Subbiah, Jared D Kaplan, Prafulla Dhariwal, Arvind Neelakantan, Pranav Shyam, Girish Sastry, Amanda Askell, et al. Language models are few-shot learners. Advances in neural information processing systems, 33:1877–1901, 2020. URL https://proceedings.neurips.cc/paper\_files/ paper/2020/hash/1457c0d6bfcb4967418bfb8ac142f64a-Abstract.html.

Ashish Vaswani, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez, Łukasz Kaiser, and Illia Polosukhin. Attention is all you need. Advances in neural information processing systems, 30, 2017. URL https://proceedings.neurips.cc/paper\_files/paper/2017/hash/ 3f5ee243547dee91fbd053c1c4a845aa-Abstract.html.

Petter Krus. Large language model in aircraft system design. In : 34th Congress of the International Council ofthe Aeronautical Sciences, ICAS 2024, 2024. URL https://www.icas.org/icas\_archive/icas2024/data/ papers/icas2024\_0514\_paper.pdf.

Brian Johns, Kristina Carroll, Casey Medina, et al. AI systems modeling enhancer (AI-SME): Initial investigations into a chatgpt-enabled MBSE modeling assistant. INCOSE International Symposium, 34(1):1149–1168, 2024. ISSN 2334-5837. doi:10.1002/iis2.13201.

Timm Teubner, Christoph M Flath, Christof Weinhardt, Wil Van Der Aalst, and Oliver Hinz. Welcome to the era of ChatGPT et al. Business & information systems engineering, 65(2):95–101, 2023. ISSN 1867-0202. doi:10.1007/s12599-023-00795-x.

Mallory S. Graydon and Sarah M. Lehman. Examining proposed uses of LLMs to produce or assess assurance arguments. Technical memorandum, NASA/TM-2025-0001849, National Aeronautics and Space Administration, March 2025. URL https://ntrs.nasa.gov/api/citations/20250001849/downloads/NASA-TM-20250001849.pdf.

Alejandro Pradas-Gomez, Petter Krus, Massimo Panarotto, and Ola Isaksson. Large language models in complex system design. In : DESIGN 2024 International design conference, volume 4, pages pp. 2197–2206. Cavtat, Croatia: Design Society, 2024. doi:10.1017/pds.2024.222.

Vinicius K. Marini, Jens Alfredson, and Petter Krus. Context of collaborative human-machine systems architecture design for enhanced functionality awareness and balanced command and control authority. In Proceedings ofthe 12th Swedish Aersopace Technology Congress - FT2025. Stockholm, Sweden: Swedish Society for Aeronautics and Astronautics (FTF), 2025. doi:10.3384/ecp215.1192.

Javier Cámara, Javier Troya, Lola Burgueño, and Antonio Vallecillo. On the assessment of generative AI in modeling tasks: an experience report with ChatGPT and UML. Software and Systems Modeling, 22(3):781–793, June 2023. ISSN 1619-1374. doi:10.1007/s10270-023-01105-5.

John K. DeHart. Leveraging large language models for direct interaction with SysML v2. INCOSE International Symposium, 34(1):2168–2185, 2024. ISSN 2334-5837. doi:10.1002/iis2.13262.

Louis Richard Timperley, Lucy Berthoud, Chris Snider, and Theo Tryfonas. Assessment of large language models for use in generative design of model based spacecraft system architectures. Journal ofEngineering Design, 36(4): 550–570, 2025. ISSN 0954-4828. doi:10.1080/09544828.2025.2453401.

Balahari Vignesh Balu, Florian Geissler, Francesco Carella, et al. Towards automated safety requirements derivation using agent-based RAG. Proceedings of the AAAI Symposium Series, 5(1):299–307, May 2025. doi:10.1609/aaaiss.v5i1.35605.

Fabian Hanke, Isaac Mpidi Bita, Oliver von Heißen, Weller Julian, Hovemann Aschot, and Dumitrescu Roman. AI-augmented systems engineering: conceptual application of retrieval-augmented generation for model-based systems engineering graph. Proceedings ofthe Design Society, 5:439–448, 2025. doi:10.1017/pds.2025.10058.

Ali Nouri, Beatriz Cabrero-Daniel, Fredrik Törner, et al. Engineering safety requirements for autonomous driving with large language models. In 2024 IEEE 32nd International Requirements Engineering Conference (RE), pages 218–228. IEEE, 2024. doi:10.1109/RE59067.2024.00029.

Ibtissam El Hassani, Tawfik Masrour, Nouhan Kourouma, Damien Motte, and Jože Tavcar. Integrating large languageˇ models for improved failure mode and effects analysis (fmea): a framework and case study. Proceedings of the Design Society, 4:2019–2028, 2024. doi:10.1017/pds.2024.204.

Ibtissam El Hassani, Tawfik Masrour, Nouhan Kourouma, and Jože Tavcar. AI-driven FMEA: integration of largeˇ language models for faster and more accurate risk analysis. Design Science, 11:e10, 2025. ISSN 2053-4701. doi:10.1017/dsj.2025.7.

Yi Qi, Xingyu Zhao, Siddartha Khastgir, and Xiaowei Huang. Safety analysis in the era of large language models: A case study of STPA using ChatGPT. Machine Learning with Applications, 19:no. 100622, 2025. ISSN 2666-8270. doi:10.1016/j.mlwa.2025.100622.

Zezhong Chen, Yuxin Deng, and Wenjie Du. Trusta: Reasoning about assurance cases with formal methods and large language models. Science of Computer Programming, 244:103288, 2025. ISSN 1872-7964. doi:10.1016/j.scico.2025.103288.

Carlo Lipizzi. From Text to Structure: Extracting and Validating Complex System Representations Using Large Language Models. In DS 141: Proceedings of the 27th International DSM Conference (DSM 2025), pages 145–153. Hoboken, NJ, USA: the Design Society, 2025. URL https://www.designsociety.org/publication/48684/from\_text\_to\_structure\_extracting\_and\_ validating\_complex\_system\_representations\_using\_large\_language\_models.

Edwin Koh. Retrieving Asymmetrical Indirect Links Through Large Language Models. In DS 141: Proceedings of the 27th International DSM Conference (DSM 2025), pages 5–8. Hoboken, NJ, USA: the Design Society, 2025. URL https://www.designsociety.org/download-publication/48688/retrieving\_ asymmetrical\_indirect\_links\_through\_large\_language\_models.

Petter Krus. Augmenting aerospace system design using large language models. In Proceedings ofthe 12th Swedish Aersopace Technology Congress - FT2025. Stockholm, Sweden: Swedish Society for Aeronautics and Astronautics (FTF), 2025. doi:10.3384/ecp215.1197.

Nelson F Liu, Kevin Lin, John Hewitt, Ashwin Paranjape, Michele Bevilacqua, Fabio Petroni, and Percy Liang. Lost in the middle: How language models use long contexts. Transactions ofthe associationfor computational linguistics, 12:157–173, 2024. doi:10.1162/tacl\_a\_00638.

Shouyuan Chen, Sherman Wong, Liangjian Chen, and Yuandong Tian. Extending context window of large language models via positional interpolation. ArXiv: 2306.15595, 2023. doi:10.48550/arXiv.2306.15595.

Mihir Gupte, Eshan Dixit, Muhammad Tayyab, and Arun Adiththan. What works for ’lost-in-the-middle’ in llms? a study on gm-extract and mitigations. ArXiv: 2511.13900, 2025. doi:10.48550/arXiv.2511.13900.

Zhenyu Zhang, Runjin Chen, Shiwei Liu, Zhewei Yao, Olatunji Ruwase, Beidi Chen, Xiaoxia Wu, and Zhangyang Wang. Found in the middle: How language models use long contexts better via plug-and-play positional encoding. ArXiV: 2403.04797, 2024. doi:10.48550/arXiv.2403.04797.

Humza Naveed, Asad Ullah Khan, Shi Qiu, Muhammad Saqib, Saeed Anwar, Muhammad Usman, Naveed Akhtar, Nick Barnes, and Ajmal Mian. A comprehensive overview of large language models. ACM Transactions on Intelligent Systems and Technology, 16(5):1–72, 2025. doi:10.1145/3744746.

Chuyifei Zhang, Hongyu Cui, Xiaowen Huang, and Jitao Sang. Positional failures in long-context llms: A blind spot in reasoning benchmarks. ArXiv: 2605.23170, 2026. doi:10.48550/arXiv.2605.23170.

George Arthur Baker, Ankush Raut, Sagi Shaier, Lawrence E Hunter, and Katharina von der Wense. Lost in the middle, and in-between: Enhancing language models’ ability to reason over long contexts in multi-hop qa. ArXiV: 2412.10079, 2024. doi:10.48550/arXiv.2412.10079. URL https://arxiv.org/abs/2412.10079.

Warren Li, Yiqian Wang, Zihan Wang, and Jingbo Shang. Order matters: Rethinking prompt construction in in-context learning. ArXiv: 2511.09700, 2025. doi:10.48550/arXiv.2511.09700.

Qi Guo, Leiyu Wang, Yidong Wang, Wei Ye, and Shikun Zhang. What makes a good order of examples in in-context learning. In Lun-Wei Ku, Andre Martins, and Vivek Srikumar, editors, Findings ofthe Associationfor Computational Linguistics: ACL 2024, pages 14892–14904, Bangkok, Thailand, August 2024. Association for Computational Linguistics. doi:10.18653/v1/2024.findings-acl.884.

Erin Crabb and Matthew T. Jones. Accelerating model-based systems engineering by harnessing generative AI. In 2024 19th Annual System of Systems Engineering Conference (SoSE), pages 110–115, 2024. doi:10.1109/SOSE62659.2024.10620975.

Florian Geissler, Karsten Roscher, and Mario Trapp. Concept-guided LLM agents for Human-AI safety codesign. Proceedings of the AAAI Symposium Series, 3(1):100–104, May 2024. doi:10.1609/aaaiss.v3i1.31188.

Simon Dehn, Simon Schnürer, Georg Jacobs, and Gregor Höpfner. Generating sysml v2 models from natural language requirements using large language models. In 2025 IEEE International Symposium on Systems Engineering (ISSE), pages 1–7. IEEE, 2025. doi:10.1109/ISSE65546.2025.11369988.

Petter Krus. Using large language models for fluid power system design. JFPS International Journal of Fluid Power System, 19(2):74–79, 2026. doi:10.5739/jfpsij.19.74.

Oliver Von Heissen, Fabian Hanke, Isaac Mpidi Bita, et al. Toward intelligent generation of system architectures. Proceedings ofNordDesign 2024, pages 504–513, 2024. URL https://www.designsociety.org/publication/ 47646/toward\_intelligent\_generation\_of\_system\_architectures.

Vinicius Kaster Marini and Petter Krus. Synthesizing aircraft system specifications with document-managed Large Language Model outputs from one-shot system inquiry prompt chain. In 10th CEAS Aerospace Europe Conference, 28th AIDAA International Congress, 2025. doi:10.21741/9781644904251-99.

An Yang, Anfeng Li, Baosong Yang, Beichen Zhang, Binyuan Hui, Bo Zheng, Bowen Yu, Chang Gao, Chengen Huang, et al. Qwen3 technical report. ArXiv:2505.09388, 2025. doi:10.48550/arXiv.2505.09388.

OpenAI. gpt-oss-120b & gpt-oss-20b model card. ArXiv: 2508.10925, 2025. doi:10.48550/arXiv.2508.10925.

NVIDIA. Nemotron 3 super: Open, efficient mixture-of-experts hybrid mamba-transformer model for agentic reasoning. ArXiv: 2604.12374, 2026. doi:10.48550/arXiv.2604.12374. URL https://arxiv.org/abs/2604.12374.

Ollama. Kimi k2.5 model card. Ollama model card, 2026. URL https://ollama.com/library/kimi-k2.5. Accessed: 2026-06-04.

Anthropic. Introducing Claude Sonnet 4.6. Anthropic Product Announcement, February 2026. URL https://www. anthropic.com/news/claude-sonnet-4-6.