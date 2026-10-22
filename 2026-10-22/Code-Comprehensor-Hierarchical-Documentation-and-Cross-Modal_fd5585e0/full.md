# Code Comprehensor: Hierarchical Documentation and Cross-Modal Retrieval for Large Repositories

Sujoy Roychowdhury<sup>∗</sup>, Joseph H. R. Isaac<sup>†</sup>, Jiangning Gao<sup>‡</sup>, Martin Klitte<sup>§</sup> and Diarmuid Corcoran<sup>¶</sup> Ericsson R&D

Email: {<sup>∗</sup>sujoy.roychowdhury, <sup>†</sup>joseph.isaac, <sup>‡</sup>jiangning.gao, <sup>§</sup>martin.klitte, <sup>¶</sup>diarmuid.corcoran}@ericsson.com

Abstract—Understanding large, complex, and rapidly evolving code bases remains a core challenge for software engineering at scale, hindering onboarding, maintenance, refactoring, and impact analysis. Providing this information to a Large-Language Model (LLM) is another hurdle, as the LLM must understand not only the required code method but also its associated methods in the grand scheme of the repository. We propose Code Comprehensor, a hybrid retrieval–reasoning framework combining (i) a code dependency callgraph, (ii) CodeScribe, which uses code-aware LLMs to generate hierarchical documentation across repository call graphs, and (iii) CodeContrast, a CLIP-style dualencoder model trained on CodeScribe output to learn aligned text-code embeddings. CodeContrast enables fast, semantically precise retrieval of relevant code-context neighborhoods, which serve as grounded evidence for an MCP-based LLM orchestrator answering domain-specific queries. Evaluation on OpenCV and a proprietary 5M-LOC repository shows that CodeScribederived supervision significantly improves cross-modal retrieval in text-to-code accuracy over baseline approaches. Richer textual supervision (hierarchical documentation, parent/sibling context, and LLM-generated synthetic questions) consistently enhances performance across both repositories. LLM-as-a-judge evaluation using the popular RAGAS of the Code Comprehensor output yields above 90% faithfulness scores.

Index Terms—Large Language Models (LLMs), Code comprehension, Semantic code retrieval, Contrastive learning, CLIPstyle encoders, Software repository analysis, Program analysis, Hybrid retrieval-reasoning frameworks

## I. INTRODUCTION

Enterprises typically have large code repositories that have been developed and maintained over the years. The ability to document and understand such complex repositories has been a challenge [1], [2]. Code complexity creates dependencies on senior engineers’ expertise to understand and navigate such repositories, and introduces inefficiencies in the Software Development Life Cycle (SDLC) [3]. The growth of Large Language Models (LLM) in recent years and their ability to understand and explain code introduce the possibility of making this process more efficient [4]. This growth will lead to greater productivity in developing new features and maintaining and debugging existing ones.

However, the ability to use LLMs to document and understand large repositories in a reliable, replicable fashion requires careful design and consideration. Despite the large context window of many modern LLMs [5], code repositories often will consume a much larger number of tokens. For example, [6] gives Lines of Code (LOC) for some public repositories, and token sizes will be much larger. Proprietary code repositories are often large in terms of LOC. For example, one of the repositories mentioned in this paper is larger than most listed in [6]. More importantly, just because an LLM can process a large number of tokens does not mean an LLM should access the entire repository for every question. In addition to cost implications, large context sizes are known to give poorer results [7], [8].

Additionally, code repositories often contain identical or very similar files, methods, and variables across different parts of the code base, and using probabilistic machinery like an LLM is unlikely to be reliable or replicable. In view of this, it is important to design systems that enable LLMs to navigate code repositories more consistently and deterministically. Regarding this issue, some studies discuss using the inherent structure of code, as captured in the call graph, to drive documentation. Work by [9] discusses using a static-analysis-based call tree to generate documentation for large-scale repositories recursively. A similar approach to generate initial and incremental documentation has been followed by other works [10]. A hierarchical engine for document generation and querying has also been studied [11]. Recently, Google released a repository to document code [12]; however, to the best of our knowledge, the system’s design has not been disclosed. The common aspect of these repositories is the use of either the Abstract Syntax Tree (AST) [13] or the call-graph. However, if one wants to build more than just static documentation, a broader and more comprehensive approach is needed. Previous work [14], [15] has looked at cross-modal contrastive learning for code search; however, without deep domain-level documentation mapping methods to domain and code structure, the effectiveness of such cross-modal retrieval on large, complex repositories is limited.

## A. Proposed Solution

We present Code Comprehensor, an end-to-end system that combines four main components: (i) CodeGraph, a call graph generated via static analysis and exposed through API endpoints for traversal and chunking; (ii) CodeScribe, a hierarchical documentation engine that recursively processes directory-structured code, supports multiple documentation chapters and custom prompts per directory, and outputs PlantUML diagrams; (iii) a custom data processing module that parses CodeScribe output into relational records linking methods to parent/sibling context; and (iv) CodeContrast, a CLIPbased dual-encoder contrastive model using InfoNCE loss for asymmetric text-to-code and code-to-text retrieval with multipositive learning support.

![](images/7d1d8ee4b906372a12747eaefd0667558b008ce86ad1c36d726038211d005c08.jpg)  
Fig. 1: Overview of the Code Comprehensor architecture. CodeGraph is created through static analysis and consumed by CodeScribe to generate structured technical descriptions. These descriptions are used by the contrastive model for text and code retrieval. An agentic layer built on the Model Context Protocol coordinates CodeGraph traversal, CodeScribe based retrieval, and contrastive model queries for interactive use.

CodeScribe operates on a repository directory structure with configurable chapter-based perspectives (e.g., code description, algorithms, data flow). For each start method (in-degree zero in the method subgraph), CodeScribe retrieves the Depth-First-Search (DFS) traversal tree via CodeGraph, chunks it if necessary using post-order traversal to preserve hierarchy, and generates documentation. CodeScribe supports both full and incremental (git-commit-driven) documentation runs.

CodeContrast leverages the structured training data from CodeScribe parsing to train a retrieval model. We adopt CLIP’s dual-encoder architecture for efficient pre-computed embeddings and asymmetric modalities. The InfoNCE objective learns well-separated embedding spaces and extends naturally to multi-positive scenarios.

Finally, a Model Context Protocol (MCP) based orchestration layer integrates CodeGraph, CodeScribe, and Code-Contrast into an agentic system accessible via an Integrated Development Environment (IDE). This Master Agent invokes specialized agents (RAG over CodeScribe output, CodeGraph traversal, CodeContrast retrieval) to answer domain-specific queries on code repositories.

The core contributions of this paper are as follows:

• A post-order chunking algorithm that enables LLM documentation of arbitrarily large call trees while preserving structural hierarchy.

• A multi-granularity training data construction pipeline that leverages parent-sibling relationships from documentation structure and LLM-generated synthetic questions for enhanced retrieval performance.

• Code Comprehensor: an integrated end-to-end system combining CodeGraph, CodeScribe, and CodeContrast components for large-scale repository understanding and

retrieval.

• Empirical evidence that hierarchical documentationderived supervision improves cross-modal code retrieval by at least 9% Top-1 accuracy over baseline approaches.

The remainder of the paper is structured as follows. We briefly summarize previous work in Section II. We discuss the System Architecture in Section III, the detailed methodology in Section IV, and present results in Section V. We conclude with a discussion, threats to validity, and future works in Sections VI, VII, and VIII, respectively.

## II. RELATED WORK

## A. Code Search and Retrieval

Code search and retrieval has been an active area of research for understanding large repositories. Recent work has focused on neural approaches combining text and code embeddings. Sun et al. [16] presented context-aware code translation for code search, while Chai et al. [17] explored cross-domain deep code search using meta-learning techniques. Mathew and Stolee [18] investigated cross-language code search, and Mao et al. [19] proposed self-supervised query reformulation for improved retrieval. More recently, Wang et al. [20] examined fusion of multiple code searchers, and Shi et al. [21] introduced CoCoSoDa for effective contrastive learning on code search tasks. Large-scale contrastive pre-training for code retrieval has been explored by Li et al. [22], who developed CodeRetriever. Comprehensive surveys on code search have been provided by Di Grazia and Pradel [23] and Sun et al. [24].

## B. Repository-Level Documentation Generation

Documentation generation at repository scale has emerged as a critical need for understanding complex systems. Luo et al. [25] introduced RepoAgent, which combines static analysis with LLM-driven documentation. Hoang et al. [26] developed CodeWiki, a framework for generating hierarchical documentation. Sounthiraraj et al. [27] proposed a hierarchical engine for structured document generation and querying.

## C. Contrastive Learning for Code

Contrastive learning has proven effective for learning aligned embeddings of code and text. Neelakantan et al. [28] introduced text and code embeddings using contrastive methods. Li et al. [22] leveraged contrastive learning in CodeRetriever for large-scale pre-training. Shi et al. [21] proposed CoCoSoDa with a momentum encoder design.

## D. Differentiation from Prior Work

Code Comprehensor extends prior work in three key ways. First, unlike code search methods that operate on isolated code snippets (e.g., CodeRetriever, CoCoSoDa), our system first generates hierarchical documentation that captures intermethod relationships through call graph analysis and multiperspective documentation generation, then trains a contrastive model on this enriched representation. Prior work on code retrieval has focused primarily on Python and Java; CodeRetriever and CoCoSoDa lack publicly available implementations for C/C++ repositories, limiting direct empirical comparison on the systems programming domain where our work is evaluated. Second, unlike RepoAgent, which focuses solely on documentation generation, our system uses the generated documentation as a structured training signal for a retrieval model, enabling efficient inference-time retrieval without requiring LLM calls per query. Third, we employ a post-order chunking algorithm to handle arbitrarily large call trees while preserving structural hierarchy, enabling scalable documentation and retrieval on repositories exceeding 5M lines of code.

## III. SYSTEM ARCHITECTURE

Figure 1 shows the overall system architecture for Code Comprehensor. Git repositories are first cloned into our system, and the CodeGraph is generated using static analysis. This CodeGraph is then consumed in CodeScribe, which uses two additional external information sources: prompts and configuration files. These are discussed in detail in Section IV-B. The output of CodeScribe, called “Synthesized Technical Descriptions” (SynTD), is consumed by CodeContrast. CodeScribe uses recursive synthesis (shown as “S” in Figure 1) indicated by the self-loop. To enable an end user (or an external system/agent) to interact with the documentation, multiple custom agents are built. The end user interacts with the Master Agent (MA) via an MCP gateway. The MA is an LLM-based orchestrator that calls multiple specialized agents. The CodeScribe Agent is an RAG-based system on the SynTD. There is a CodeContrast agent that can invoke API calls on the trained contrastive model. The CodeGraph agent can perform custom traversals on the CodeGraph via an exposed API.

Given a natural-language question from the user (an Engineer in this example), the MA retrieves relevant code blocks from the CodeContrast agent using Text→Code-based retrieval. The MA retrieves the hierarchy of the relevant methods from the CodeGraph agent and obtains their documentation using the CodeScribe Agent. Finally, the MA composes a natural-language response to the end user and adds relevant diagrams and code blocks to provide a more comprehensive answer. The end user can interact further with the MA with follow-up questions. The Engineer can also ask questions, such as explanations of specific files, methods, or classes, and their relevance to the overall system. In such a case, MA uses the Code→Text retrieval from the CM agent. It also identifies the method or file hierarchy from the CodeGraph agent and the documentation from the CodeScribe Agent. The MA decides which type of retrieval to invoke from the CodeContrast agent. This system allows for seamless end-to-end navigation and interpretation of the repository.

## IV. METHODOLOGY

## A. CodeGraph Creation

To create the CodeGraph, a static analysis of the code repository is first performed. Static analysis identifies all code files and their corresponding methods. Dependencies are resolved via compile commands or, if unavailable, with reasonable heuristics. Node types are files and methods, and in object-oriented languages, other node types, such as classes, are added. Comments written by the developer for the methods are also extracted as attributes for the method nodes in Code-Graph, along with the method signatures. For methods without a developer-written comment, an LLM is prompted to generate one. The actual method body is also available as an attribute on the method. Since the same method can appear in multiple files, nodes are named “method\_name:::file\_name” where “:::” is chosen as the separator since it is unlikely to appear elsewhere. In object-oriented languages, nodes are named “method\_name:::file\_name::::line number” to handle operator overloading. Method names include the namespace and class name, where relevant, as fully qualified names. The Tree-Sitter library [29] is used for parsing, though other approaches can be used. The CodeGraph generation, of course, needs (programming) language-specific parsing. The Code-Graph structure is generic, and the rest of the processing in the entire system is language-agnostic.

The CodeGraph is saved as a Python Networkx [30] library object. This CodeGraph is also exposed through a FastAPI service [31]. Several endpoints are exposed via the FastAPI service. Most are standard endpoints for graph traversal, e.g., Breadth-First-Search (BFS) and Depth-First-Search (DFS), along with the “invokes” edge in both forward and reverse directions. The resultant CodeGraph and the endpoints are used to generate the documents using CodeScribe.

## B. CodeScribe

CodeScribe documentation is driven by the repository’s directory structure because that is the natural way developers navigate it. The output of CodeScribe is Markdown files in a directory structure that replicates the code repository’s directories. CodeScribe has configuration files that allow one to exclude irrelevant directories and include only file name extensions that are present. Every directory can have one or more perspectives to document the code, called “chapters” (e.g., code description, algorithms, data flow, interfaces). For each chapter, a custom prompt is generated, with flexibility to customize per directory. CodeScribe can also consume configuration information, such as abbreviation lists for specialized domains.

The system prompts are designed with three key principles: (1) LLM output is constrained to generate documents at hierarchical levels of abstraction, summarizing at appropriate semantic boundaries; (2) quoted code segments are marked with metadata (method name, file path, language) to enable downstream parsing and tracing back to source; (3) consistency across documentation perspectives ensures coherent multi-chapter views of the same code units. These design choices are critical for downstream parsing and contrastive model training.

When processing any directory, CodeScribe invokes Code-Graph to find all files in the directory and the methods they contain. Next, among the methods it identifies, start methods are those with an in-degree of zero in the methodonly subgraph (a subgraph consisting only of methods). These methods are invoked by the external system (outside the scope of the current module graph). For each start method, the DFS traversal tree is retrieved via an endpoint on the CodeGraph API. The DFS traversal endpoint generates a key for each element in the DFS walk, and this information is included in the LLM’s context. The context also includes the method definitions in the DFS tree. Thus, there is one Markdown output file per chapter and per start method. The documents for each start method are then combined for a summarized document at the directory level. Optionally, via a configuration file, summarizations for directory hierarchies are also done.

However, in large repositories, it is common to find trees that exceed the allowable context window, especially since method definitions are included. It is important to note that context sizes include output tokens, and there must be a reasonable allowance left for the LLM to generate sufficiently detailed documentation. This requires that the DFS walk trees be properly chunked. Hence, we build a chunking process that allows customizable chunk sizes. To handle such large trees, we implement an algorithm that uses recursive postorder traversal to process the DFS tree. For each node, we first process all children recursively, collecting any remaining chunks. We then check if the current node plus all child remainders fit within the primary context limit. If so, we merge them. If not, we finalize each child remainder as an independent chunk and keep only the current node. Finally, if the resulting chunk exceeds the primary limit plus a buffer, we split it into overflow chunks and return the remaining tail.

The chunks are presented to the LLM as context, and the output of the previous chunk is added to future chunks in the context in a recursive generation process. The output of intermediate chunks is temporary and not included in the final documentation. Our algorithm ensures that when the LLM processes a parent caller, it possesses a high-level functional summary of all callee implementations, even if the source code of those children is not included in the current active window. This rolling summarization effectively compresses the global DFS walk state, allowing the model to generate accurate parent-level documentation without having to guess the behavior of its dependencies.

![](images/344a9658896a7eeba77878ed1095ffc6df4bcae05c9a1e74c077df155f510753.jpg)  
Fig. 2: Incremental processing to identify all impacted root methods

CodeScribe can also process incrementals based on git commit IDs. This process can be done by identifying the last processed git commit and identifying files that have changed. Further to this, the methods in these impacted files are identified and checked against the previous version of the graph to see if they have changed. If they have, then the root/start methods of the trees they belong to are identified. This method is shown in Figure 2. Once this is done, CodeScribe asks the LLM to modify only the start methods and any summarization documents generated for this. This process allows incremental processing at any frequency, from periodic intervals such as weekly to integration with Continuous Integration/Continuous Development (CI/CD) pipelines. CodeScribe’s output is then parsed into a more palatable data format for CodeContrast.

## C. Parsing CodeScribe Data

As mentioned before, CodeScribe generates Markdown output that mimics the repository’s directory structure. The documents contain a mixture of the following:

• Hierarchical section headers denoting algorithms and module structure.

• Inline code blocks annotated with method names, file paths, and programming language identifiers.

• PlantUML diagram specifications embedded within fenced code blocks.

• Free-form textual descriptions providing domain context, algorithmic rationale, and inter-component relationships. To utilize this data for training CodeContrast, the Markdown files undergo several processes:

1) Markdown-to-JSON Conversion: A custom recursive parser was implemented to convert Markdown documents into nested JSON structures. The headers and subheaders of the Markdown file become the keys of the JSON file. Non-header content is aggregated under an arbitrary key at the appropriate nesting level. This approach preserves the hierarchical organization of the original documentation while enabling programmatic traversal.

2) Code Block and Text Separation: Each leaf node in the resulting JSON tree is processed to separate the code blocks from the text segments. A regex-based extractor identifies fenced code blocks (delimited by triple backticks) and captures both the language identifier and the source code. The remaining text, after code block removal, is retained as a description. The extraction procedure traverses the JSON tree recursively, assigning a dot-separated depth identifier (e.g., 1.2.3) to each node. This depth encoding enables the subsequent identification of parent-child and sibling relationships among extracted entries, which is critical for constructing the relational features of the final dataset.

3) Code Block Classification: Each extracted code block is classified into one of three categories based on its content:

• Code: Blocks containing annotated source code with explicit method name and file path metadata (extracted via inline comments from CodeScribe).

• UML: Blocks identified as PlantUML specifications, detected by the presence of @startuml directives or the PlantUML language identifier.

• Pseudocode: Code blocks lacking method name and file path annotations, treated as pseudocode or algorithmic sketches.

4) Relational Feature Construction: For each code entry, the pipeline constructs a rich set of relational features by using the depth-encoded tree structure from CodeScribe:

• Parent context: Textual descriptions from the immediate parent node, providing high-level context.

• Sibling context: Method names, comments, file paths, and code of entries sharing the same parent, capturing co-located functionality.

• Module-level context: Aggregated commentary from text-only nodes that lack direct child code blocks, serving as overarching module documentation.

These relational features are designed to provide the contrastive learning model with multi-granularity contextual signals linking code to its surrounding documentation. The parsed records are merged with the call graph data created earlier, using the same composite key of the form “method\_name:::file\_path”

## D. Synthetic Data Generation

Large repositories often have sparse, inconsistent, or outdated data (e.g., missing comments, legacy documentation, or domain-specific abbreviations). Moreover, real developer queries often take the form of questions rather than direct text descriptions. Therefore, to improve coverage and better match downstream retrieval and reasoning workloads, synthetic questions generated from the parsed records are augmented, with code included only as a grounding aid.

1) Design Principles: Synthetic questions are constructed with two explicit objectives:

• Intent diversity: cover common developer intents such as purpose/behavior, parameter usage, edge cases, and preconditions, error handling, and interactions within a local module context.

• Retrieval grounding: include repository-specific terminology (parameter names, types, macros, struct/enum identifiers, and field names) to improve retrievability, while explicitly discouraging “keyword dumping”.

In our pipeline, the semantic intent of each question is primarily driven by the available documentation signal. At the same time, the method signature and a bounded code snippet are used as auxiliary sources for terminology anchors.

2) Generation Inputs: Each synthetic question set is generated for a single code entry produced by Section IV-C.

For a given entry, the generation payload consists of:

• Documentation signal: one consolidated text slice corresponding to a selected comment level or hierarchical comment fields when available.

• Code signal: the method signature and a bounded code snippet (provided for terminology only, with explicit instructions not to copy verbatim). We provide a signature with the symbol name removed.

The generator, therefore, receives a comment-driven description of behavior, augmented with a limited amount of codelocal context.

3) Terminology Anchor Extraction: To encourage grounded and retrievable questions, we extract candidate anchors from the entry:

• Behavior anchors (BEHAVIOR\_ANCHORS): action-level intent tokens (verbs) that constrain question semantics.

• Parameter anchors (PARAM\_TERMS): parameter names and salient type tokens parsed from the function signature.

• Context anchors (CONTEXT\_TERMS): repository- and domain-specific terminology mined from the documentation slice and the bounded code context.

• Key terms (KEY\_TERMS): a capped union of parameter and context anchors, used as a lexical grounding constraint while limiting the total number of enforced tokens to avoid keyword stuffing.

4) Prompting Strategy: Comment-Driven Intent with Anchored Questions: We use a code-capable instructionfollowing LLM to generate synthetic questions.<sup>1</sup>

For each entry, the prompt instructs the generator to produce developer-style questions with the following rules:

• Reflect the described behavior and constraints from the documentation signal;

• includes at least two terminology anchors (preferably one from PARAM\_TERMS and one from CONTEXT\_TERMS when available), and at least three verbatim hits from KEY\_TERMS;

<sup>1</sup>We use [GEN\_MODEL] with decoding parameters [TEMP], [TOP\_P], and [MAX\_TOKENS].

```c
Reference context(do NOT copy reference code verbatim into the questions):
Function signature (function name removed): bfloat16x8_t (float <sub>*</sub> src)
Comment: Loads 8 float values from memory and converts them to bfloat16 ...
Code snippet (terminology only; do NOT copy):
bfloat16x8_t vld1q_as_bf16(const float src){
float32x4_t s0 = vld1q_f32(src), s1 = vld1q_f32(src + 4);
return vcombine_bf16(vcvt_bf16_f32(s0), vcvt_bf16_f32(s1));}
BEHAVIOR_ANCHORS (choose >=1 per question): convert, pack, register
PARAM_TERMS (use in questions): src, float
CONTEXT_TERMS (use in questions): NEON, ARM, SIMD, memory, bfloat16, vector, ...
KEY_TERMS (use in questions): src, float, NEON, ARM, SIMD, memory, bfloat16, vector,
Task:
Generate EXACTLY 3 DISTINCT natural-language questions.
Hard constraints for each question:
MUST include at least ONE BEHAVIOR_ANCHOR (verbatim).
MUST include at least ONE term from PARAM_TERMS and at least ONE term from
CONTEXT_TERMS.
MUST include at least 3 items from KEY_TERMS (verbatim).
MUST end with a question mark ’?’.
MUST NOT mention any function name, file path, any id, or file:line identifiers.
OUTPUT QUESTIONS: [
"How can I convert eight consecutive float values from memory into a bfloat16
NEON vector using ARM SIMD instructions that maintains high precision bits?",
"What ARM NEON approach should I use to pack multiple float values from src
memory into a single bfloat16 vector through vectorized type conversion?",
"How do I register-load float data from memory and convert them to bfloat16
format using NEON SIMD instructions for efficient floating point processing?"]
```

Fig. 3: Example prompt for synthetic question generation and corresponding generated questions in OpenCV. Some words/phrases are in bold here for clarity.  
![](images/c83ec1c711839cd20fe5b28f0ea879687352d930c51d7e32c5f150af44d893d5.jpg)  
Fig. 4: Overview of the CodeContrast workflow, covering data preprocessing, distributed training, along with the testing and inference processes.

• Include at least one BEHAVIOR\_ANCHORS (e.g., validate, initialize, convert, decode) to reduce purely tokendriven outputs, while discouraging keyword dumping;

• Enforce interrogative form (each question ends with ‘?’);

• Avoid copying code verbatim and avoid dataset-specific identifiers (e.g., <path>:<line> strings).

This design yields questions whose intent is commentdriven, while code-derived anchors improve grounding and retrieval utility. Listing 3 shows an example prompt used for synthetic question generation. We instruct the model to produce exactly three developer-style questions for each method.

5) Post-processing: We apply lightweight filtering to improve precision and reduce noise:

• Format and alignment checks: enforce strict JSON output and validate echoed tags; discard malformed outputs.

• Deduplication: remove exact and near-duplicate questions within the same ID.

• Informativeness: discard questions that do not contain the required anchors or fall below a minimum length.

To assess retrieval generalization beyond trivial lexical shortcuts, we additionally evaluate a post-processed variant where function-name tokens are removed from questions.

## E. CodeContrast Architecture

CodeContrast is a custom dual-encoder contrastive learning system [32] that aligns natural-language documentation embeddings with source-code embeddings. The system is designed for a code retrieval task: given a textual query (e.g., documentation or comments), retrieve the most relevant code snippet from a corpus, and vice versa. It uses PyTorch’s Fully Sharded Data Parallel (FSDP) [33] for distributed multi-GPU training, supports both SentenceTransformer (SBERT) [34] and normal BERT Transformer [35] backbones, and includes a full pipeline from data preparation through training, embedding store creation, and evaluation. The full architecture is shown in Figure 4.

1) Data Preparation Pipeline: This module serves as the initial stage of the data-processing pipeline. It loads a JSON input file that is created from Section IV-C, and applies validation and filtering logic to produce structured entries containing code, documentation, unique identifiers, and query types. This induces a one-to-many supervision structure: one code unit corresponds to many valid textual descriptions:

• a single code unit (method signature and bounded code snippet), and

• multiple text views (raw documentation, hierarchical context, and synthetic questions), each tagged by a query\_type.

During this step, the system also flags entries that may cause issues, such as those with insufficient documentation or code that is too long to process effectively. Once everything is cleaned and organized, the module splits the data into training, validation, and test sets and then saves the final outputs.

2) Core Model Architecture: The embedding system supports several backbone options. In one setup, we use SBERT encoders, which produce sentence embeddings directly. In another, we use Transformer models such as BERT, where we create fixed-size representations either by averaging the contextual token embeddings or by taking the final token embedding. To make sure these different backbones produce vectors of the same size, we optionally add a projection layer that maps everything to a shared dimension d. All embeddings are then $\ell _ { 2 } \cdot$ -normalized onto the unit hypersphere, enabling cosine similarity to be computed by dot product.

A dual-encoder framework integrates these embedding models in a unified contrastive learning setup. Each encoder processes either a text query or a code snippet, and both are trained jointly using a similarity objective that is adjusted by a temperature parameter. Because training runs across several devices, we gather embeddings from all GPUs to form a single global similarity matrix. This lets the model compare every item in the batch to all others, increasing the number of negative examples and effectively creating a much larger batch. In practice, this leads to more stable training and stronger contrastive representations.

3) Differentiable Global-Batch Training under FSDP: We use the Fully Sharded Data Parallel (FSDP) paradigm for training - this runs across multiple devices, so we bring all embeddings and labels together into one global batch before computing the similarity matrix. This gives the model a much larger set of negatives and keeps the contrastive comparisons consistent across GPUs. To make this work, we use a differentiable all-gather operation that allows gradients to flow through the gathered tensors, which keeps the entire contrastive objective fully trainable even in a distributed setup. For safety, any rank that produces an empty batch still participates in the gather using placeholder tensors. This avoids synchronization issues and prevents the training process from stalling.

4) Multi-positive Symmetric Contrastive Objective: Multiple text views may describe the same underlying code unit, and these code-text pairs may appear multiple times within a batch, depending on sampling. Therefore, positives using identifier labels are defined. Let $\ell _ { i }$ denote the class label derived from $\mathrm { i } \mathrm { d } _ { i }$ . For each text anchor $t _ { i } ,$ , the set of positive code indices is $P ( i ) = \{ j \mid \ell _ { j } = \ell _ { i } \}$ . The Text→Code loss uses a multipositive InfoNCE objective with log-sum-exp aggregation:

$$
\mathcal { L } ^ { t  c } = - \frac { 1 } { B } \sum _ { i = 1 } ^ { B } \log ( \frac { \sum _ { j \in P ( i ) } \exp ( \mathbf { S } _ { i j } ) } { \sum _ { j = 1 } ^ { B } \exp ( \mathbf { S } _ { i j } ) } ) .\tag{1}
$$

Similarly, for Code→Text we swap roles and apply the same objective to S<sup>⊤</sup>: $\mathbf { S } ^ { \top }$

$$
\mathcal { L } ^ { c  t } = - \frac { 1 } { B } \sum _ { j = 1 } ^ { B } \log ( \frac { \sum _ { i : \ell _ { i } = \ell _ { j } } \exp ( \mathbf { S } _ { i j } ) } { \sum _ { i = 1 } ^ { B } \exp ( \mathbf { S } _ { i j } ) } ) .\tag{2}
$$

We optimize a symmetric objective:

$$
\mathscr { L } = \frac { 1 } { 2 } ( \mathscr { L } ^ { t  c } + \mathscr { L } ^ { c  t } ) .\tag{3}
$$

5) Evaluation Metrics: We evaluate retrieval performance in both directions using metrics derived from the cross-modal similarity matrix S . For each query, we define the top matches as the candidates with the highest matching similarity scores, and report Top-k accuracy. Recall@K measures how many of the relevant items are successfully retrieved within the topK results out of the total number of relevant items. We also report direction-specific metrics by evaluating both S (Text→Code) and $\mathbf { S } ^ { \top }$ (Code→Text).

## V. RESULTS

## A. Test Methodology

The objective of the testing phase is to assess how effectively the model retrieves the correct code function when provided with a natural language query (Text→Code), and how well it identifies the appropriate textual description when given a code snippet (Code→Text). The evaluation uses a held-out test set consisting of methods paired with their corresponding text–code samples. Each sample includes a function along with all associated textual descriptions, such as raw docstrings, brief comments, self-comments, and synthetic questions. None of the test methods or their synthetic variants are included in the training data, which ensures fair and unbiased assessment. Two types of tests are conducted. The first is the non-synthetic test, which pairs each code sample with its original human-authored (or LLM-authored when absent) textual descriptions. The second is the synthetic test, which augments each original pair with LLM-generated questions derived from the same function, as described in Section IV-D.

![](images/6f1333de2c0034bbfd27b606a7438cdb3cdd3a5d17b4af5c2ded31c624c4b5d1.jpg)  
Fig. 5: Sample diagrammatic representation of sub-modules in OpenCV Calib3d module. This module provides 3D calibration features and has a number of important submodules.

In addition to testing for the accuracy of CodeContrast, we also do an evaluation for testing the output of the Code Comprehensor system by using LLM-as-a-judge for the output of the MA. As this is a subjective evaluation, we use the RAGAS framework [36]. In particular, we use the Answer Correctness (AC) and Faithfulness (FF) metrics of this framework; earlier studies have shown that these have the best correlation with ground truth [37]. This evaluation is done on a subset of the synthetic questions; the generated answer from the MA is compared with the ground-truth answer used at the time the synthetic questions were generated.

## B. Dataset

We evaluate our results on two repositories. One of them (called Repo-A) is a proprietary C-language repository in a large telecommunications organization, having over 5 million Lines of Code (LOC) in “.c” and “.h” files only. In addition, to ensure replicability, we run our experiments using OpenCV<sup>2</sup>. OpenCV has about 1.2 million LOC across its ”.c”, ”.h”, and ”.cpp” files. We use Anthropic Sonnet 4.5 as the LLM for CodeScribe and Anthropic Sonnet 4.0 for all other generation tasks (missing comments, synthetic questions).

## C. CodeScribe output on OpenCV

CodeScribe output quality can be verified subjectively. Subject Matter Experts (SMEs) have verified the results of the proprietary repository and have expressed satisfaction with the output. For OpenCV, the authors did a manual comparison themselves, as they are quite conversant with the library.

We compare the output with CodeWiki [12], which is publicly available for OpenCV. For example, in the section on 3D calibration, one of the important features available in OpenCV, CodeScribe provides a modular representation of its important components and discussions on each of them separately. This is shown in Figure 5, which is much more informative than the corresponding module<sup>3</sup> in CodeWiki.

The documentation is largely correct and useful, but somewhat verbose. This documentation may be very helpful for an agent to consume; in some cases, it can be challenging for humans to browse. However, the Master Agent can interactively expand or summarize the documentation, along with additional context from the other agents. Additionally, CodeScribe generates component diagrams that provide a systematic understanding of the repository’s contents.

## D. Computational Efficiency and Costs

The practical deployment of Code Comprehensor requires an understanding of resource requirements. CodeScribe generation for OpenCV (1.2M LOC) required approximately 48 hours and \$3,200 USD in LLM API costs using Sonnet 4.5. For Repo-A (5M LOC), the generation required approximately 8 days and a few thousand USD. CodeContrast training on 4 NVIDIA A100 GPUs converged in 15 epochs over approximately 24 hours. Inference is significantly more efficient: CodeContrast retrieval requires only a forward pass through the text encoder followed by cosine similarity computation against pre-computed code embeddings, achieving <50 ms latency per query. The full Master Agent pipeline (text-tocode retrieval + context assembly + LLM response generation) averages 3-5 seconds per query, depending on the result set size and the complexity of the response synthesis. These costs are one-time for initial documentation generation and model training; incremental updates (triggered by code changes) cost significantly less per month.

## E. Configurations

To study how different levels of natural language supervision influence cross-modal retrieval, we define three dataset configurations, D1 through D3. Each configuration adds an additional layer of textual descriptions to the same code elements. This gradual buildup allows us to clearly observe how various comment types and question-answer styles affect the model’s ability to generalize.

1) D1 — Method + Docstring (Baseline Model): This configuration uses only the method’s source code paired with its corresponding docstring or brief functional description. Docstrings provide a concise, developer-written explanation of what the method does and form the foundational text modality for contrastive learning. They represent the simplest and most common documentation layer in code repositories.

2) D2 — Method + Docstring + Self-Comments: D2 augments D1 by adding self-comments, which were extracted from the CodeScribe data (Refer Section IV-C). These comments offer finer-grained insights into the how and why of the

TABLE I: Top-K retrieval results across four model configurations. Top-K columns report text→code accuracy and code→text accuracy; Recall@K (c→t) reports recall for the code→text task. N denotes the number of test samples for results in those rows.
<table><tr><td rowspan="2">Repository</td><td rowspan="2">Configuration</td><td rowspan="2">Test Type</td><td colspan="4">TopK (text→code) %</td><td colspan="4">TopK (code→text) %</td><td colspan="4">Recall@K (c→t) %</td></tr><tr><td>Top1</td><td>Top3</td><td>Top5</td><td>Top10</td><td>Top1c</td><td>Top3c</td><td>Top5c</td><td>Top10c</td><td>R@1</td><td>R@3</td><td>R@5</td><td>R@10</td></tr><tr><td rowspan="4">OpenCV</td><td>D1</td><td>Non-synthetic test</td><td>92.68</td><td>97.23</td><td>97.94</td><td>98.36</td><td>91.99</td><td>97.62</td><td>98.04</td><td>98.51</td><td>91.04</td><td>97.41</td><td>97.94</td><td>98.50</td></tr><tr><td>D2</td><td>N = 2813</td><td>94.45</td><td>98.19</td><td>98.54</td><td>99.15</td><td>94.66</td><td>97.97</td><td>98.50</td><td>99.00</td><td>93.42</td><td>97.75</td><td>98.41</td><td>98.98</td></tr><tr><td>D2</td><td>Synthetic</td><td>66.05</td><td>80.94</td><td>85.23</td><td>89.37</td><td>96.53</td><td>99.28</td><td>99.50</td><td>99.61</td><td>95.17</td><td>99.02</td><td>99.43</td><td>99.59</td></tr><tr><td>D3</td><td>N = 12658</td><td>80.64</td><td>93.29</td><td>95.83</td><td>97.61</td><td>93.67</td><td>98.07</td><td>98.68</td><td>98.82</td><td>92.24</td><td>97.82</td><td>98.57</td><td>98.80</td></tr><tr><td rowspan="4">Repo-A</td><td>D1</td><td>Non-synthetic</td><td>74.82</td><td>89.92</td><td>93.31</td><td>95.15</td><td>74.21</td><td>89.12</td><td>92.47</td><td>95.21</td><td>67.91</td><td>86.34</td><td>91.40</td><td>95.07</td></tr><tr><td>D2</td><td>N = 3800</td><td>79.54</td><td>92.72</td><td>94.82</td><td>96.23</td><td>79.28</td><td>92.65</td><td>94.85</td><td>96.36</td><td>72.42</td><td>90.02</td><td>93.95</td><td>96.29</td></tr><tr><td>D2</td><td>Synthetic</td><td>67.46</td><td>84.21</td><td>88.16</td><td>91.69</td><td>80.41</td><td>92.44</td><td>94.75</td><td>96.44</td><td>74.37</td><td>90.01</td><td>93.93</td><td>96.32</td></tr><tr><td>D3</td><td>N = 23480</td><td>76.18</td><td>91.20</td><td>93.91</td><td>96.01</td><td>81.24</td><td>93.56</td><td>96.05</td><td>97.43</td><td>75.43</td><td>91.10</td><td>95.13</td><td>97.34</td></tr></table>

## Query-Type Definitions:

• self\_comment: The CodeScribe-generated description of the method itself (self-level documentation).

• doc\_brief\_merged: The original developer-written docstring/brief comment merged with any existing documentation.

• qa\_doc\_gen: Synthetic questions generated from CodeScribe-produced documentation.

• qa\_doc\_raw: Synthetic questions generated from the raw/original developer documentation.

• qa\_self: Synthetic questions generated from the self-comment level documentation.

• qa\_parent: Synthetic questions generated from parent-level context documentation.

• qa\_sibling: Synthetic questions generated from sibling-level context documentation.

TABLE II: Per–query-type Top-K (text→code) accuracy comparing D2 vs D3 for the OpenCV Repository.
<table><tr><td rowspan="2">Query Type</td><td rowspan="2">Number of Samples (Test)</td><td colspan="4">D2 (TopK) %</td><td colspan="4">D3 (TopK) %</td></tr><tr><td>Top1</td><td>Top3</td><td>Top5</td><td>Top10</td><td>Top1</td><td>Top3</td><td>Top5</td><td>Top10</td></tr><tr><td>self_comment</td><td>209</td><td>65.55</td><td>80.38</td><td>84.69</td><td>88.04</td><td>65.55</td><td>84.69</td><td>88.99</td><td>93.30</td></tr><tr><td>doc_brief_merged</td><td>2804</td><td>96.43</td><td>99.14</td><td>99.36</td><td>99.54</td><td>93.94</td><td>98.18</td><td>98.64</td><td>98.82</td></tr><tr><td>qa_doc_gen</td><td>8049</td><td>57.39</td><td>76.43</td><td>82.02</td><td>87.25</td><td>78.38</td><td>93.14</td><td>96.15</td><td>98.05</td></tr><tr><td>qa_doc_raw</td><td>351</td><td>78.06</td><td>94.02</td><td>94.87</td><td>97.72</td><td>83.19</td><td>97.72</td><td>99.43</td><td>99.43</td></tr><tr><td>qa_self</td><td>418</td><td>62.20</td><td>78.47</td><td>83.25</td><td>90.67</td><td>76.79</td><td>94.02</td><td>96.89</td><td>98.56</td></tr><tr><td>qa_parent</td><td>144</td><td>54.17</td><td>70.14</td><td>75.69</td><td>79.17</td><td>69.44</td><td>88.19</td><td>91.67</td><td>93.75</td></tr><tr><td>qa_sibling</td><td>402</td><td>44.78</td><td>60.45</td><td>67.16</td><td>74.38</td><td>63.93</td><td>78.86</td><td>84.83</td><td>90.80</td></tr></table>

code’s behavior. The model is then trained on text that will have self-comments added to the text corpus.

3) D3 — D2 + Parent–Sibling Comments and QA (Structured Context Expansion): D3 introduces higher-level relational documentation by adding parent and sibling comments extracted from the CodeScribe output. This stage also includes QA derived from these comment types, enabling the model to learn abstractions that span across method boundaries and reflect multi-method interactions. D3, therefore, represents the richest supervision setting, capturing both local method semantics and global contextual semantics.

TABLE III: Mean scores (%) for evaluation of master agent using RAGAS metrics. AC: Answer Correctness
<table><tr><td>Metric</td><td>Repo-A</td><td>OpenCV</td></tr><tr><td>No. samples</td><td>4986</td><td>4971</td></tr><tr><td>AC: Precision</td><td>72.0</td><td>87.5</td></tr><tr><td>AC: Recall</td><td>58.3</td><td>72.1</td></tr><tr><td>AC: F1</td><td>60.0</td><td>76.4</td></tr><tr><td>Faithfulness</td><td>94.6</td><td>93.4</td></tr></table>

## F. Evaluation of Master agent

Table III presents the agentic evaluation results for the MA using RAGAS metrics. The high FF scores indicate strong grounding of the generated responses in the retrieved context. The comparatively lower AC scores are influenced by the limited number of claims in both the ground-truth and generated responses, which results in a small denominator during metric computation and increases score sensitivity to individual mismatches.

## VI. DISCUSSION

The results across both repositories demonstrate the nuanced benefits of incorporating CodeScribe-generated information into the CodeContrast pipeline. In Table I, the Top-1 accuracy overall for OpenCV is 66.05% for D2 and is 80.64% for D3, showing an improvement of 14.59%. Similarly, for Repo-A, we get 8.72% improvement (67.6% for D2 and 76.18% for D3). In Table II, for non-synthetic tests on human-authored descriptions, D2 achieves superior performance: on OpenCV’s doc\_brief\_merged category, D2 achieves Top-1 accuracy of 96.43% while D3 achieves 93.94%. This indicates that adding parent/sibling context and synthetic QA can reduce performance when evaluating against developer-authored comments.

However, for synthetic test queries, D3 demonstrates consistent improvements over D2 across all Top-K values for complex query categories (qa\_doc\_gen, qa\_doc\_raw, qa\_self, qa\_parent, qa\_sibling). This improvement compensates for the drop in accuracy in doc\_brief\_merged discussed earlier. To ensure that function names in synthetic questions do not lead to data leakage, we remove function-name tokens from questions - the results in Table II are using these post-processed questions.

The larger architecture explains these complementary benefits. CodeScribe introduces a structured way of summarizing code at different levels of abstraction through call graph traversal, recursive document synthesis, and chapter-based documentation prompts. The parsing pipeline converts these hierarchical documents into structured records that preserve parent and sibling relationships. These relational signals help the contrastive model learn contextual relationships between methods. D2 excels when queries closely match humanauthored documentation, while D3 provides benefits for highvariability or indirectly-specified queries that require broader semantic reasoning.

In practice, the expected query distribution should inform configuration selection: D2 is recommended for applications where queries directly correspond to method documentation (e.g., documentation search), while D3 is better suited for diverse developer queries requiring understanding of broader system context. The practical deployment should measure performance on the target query distribution.

## VII. THREATS TO VALIDITY

Internal Validity: Our evaluation is limited to two repositories, both in the systems programming domain (C and C++). LLM non-determinism and the use of different model versions (Sonnet 4.5 vs 4.0) across pipeline stages may introduce variability. We used Sonnet 4.5 for CodeScribe to ensure high-quality reasoning over large code contexts, and Sonnet 4.0 for simpler generation tasks (missing comments, synthetic questions) to balance quality and cost; this choice may introduce distributional differences in the training data.

Additionally, the train/test split is performed at the method level, but parent and sibling context fields in training entries may reference code from held-out test methods. To mitigate this: (1) sibling context includes only method names and brief comments, not full code bodies; (2) the code encoder sees only the target method’s body, not surrounding context; (3) removing cross-references to test methods from training context fields changes Top-1 accuracy by less than 1% on OpenCV, confirming minimal impact.

External Validity: We cannot claim language-agnostic generalization without evaluation on Java, Python, Go, Rust, and other languages. While the methodology (CodeGraph, CodeScribe, CodeContrast) is language-agnostic by design, empirical validation is limited to C/C++ systems programming. This is a significant limitation acknowledged in the Conclusion.

Construct Validity: We evaluate the retrieval component in isolation using Top-K and Recall metrics as proxies for “code comprehension.” The final agentic evaluation is conducted using the RAGAS framework, specifically two metrics: FF and AC.

Conclusion Validity: The D3 configuration simultaneously adds both parent/sibling context and synthetic QA. Without an intermediate configuration isolating each component, we cannot definitively attribute improvements solely to either source without additional ablation studies. The lack of finetuned models for agentic evaluation of code-based question answering may affect the RAGAS evaluation results.

## VIII. CONCLUSION AND FUTURE WORKS

In this work, we presented Code Comprehensor, a unified retrieval–reasoning framework for understanding large and complex software repositories. The system integrates hierarchical documentation generation via CodeScribe and a dual-encoder CLIP model to enable precise, scalable code comprehension. Our approach systematically combines call-graph traversal, structured multi-level documentation, and a rich text–code alignment pipeline to produce a repository-scale knowledge representation that can be consumed by an agentic orchestration layer. Evaluations on two large repositories, including OpenCV and one proprietary industrial codebase, show complementary benefits across different query distributions: D2 improves retrieval for natural human-authored descriptions (96.43% Top-1 on OpenCV), while D3 provides additional gains for high-variability synthetic query scenarios (80.64% Top-1 on synthetic text→code), with the choice of configuration depending on the expected deployment query distribution. The Code Comprehensor evaluation shows FF scores above 90%, indicating the quality of the MA-generated response.

Our experimental results are for one C (proprietary) repository and one C++ repository (OpenCV), both in the systems programming domain. See Section VII for a detailed discussion of validity threats. An experimental evaluation across a broader set of languages (Java, Python, Go, etc.) and application domains would be important for validating generalization. Given that CodeScribe relies on LLM generation, niche languages that are less well-represented in LLM training data may pose challenges.

Future work will expand multimodal capabilities by integrating structural graphs, sequence diagrams, and visual artifacts. This could better capture relationships and control flows that are difficult to express in text, improving retrieval relevance. Additionally, using the CodeGraph as an extra modality would allow the retrieval to be guided by the proximity of nodes in the graph and not just semantic retrieval. Such a process would help improve the answer correctness of the master agent because relevant methods would be retrieved by the CodeGraph agent. Also, synthetic question generation can be refined through improved prompting, richer intent modeling, curriculum-based generation, and adversarial techniques to enhance generalization.

## REFERENCES

[1] T. Cerny, J. Svacina, D. Das, V. Bushong, M. Bures, P. Tisnovsky, K. Frajtak, D. Shin, and J. Huang, “On code analysis opportunities and challenges for enterprise systems and microservices,” IEEE access, vol. 8, pp. 159449–159470, 2020.

[2] J. Sayago-Heredia, R. Perez-Castillo, and M. Piattini, “A systematic´ mapping study on analysis of code repositories,” Informatica, vol. 32, no. 3, pp. 619–660, 2021.

[3] M. Hammad, H. Hijazi, M. Hammad, and A. F. Otoom, “Mining expertise of developers from software repositories,” International Journal of Computer Applications in Technology, vol. 62, no. 3, pp. 227–239, 2020.

[4] S. MacNeil, A. Tran, A. Hellas, J. Kim, S. Sarsa, P. Denny, S. Bernstein, and J. Leinonen, “Experiences from using code explanations generated by large language models in a web software development e-book,” in Proceedings of the 54th ACM Technical Symposium on Computer Science Education V. 1, pp. 931–937, 2023.

[5] J. Kowalski, “Long-context benchmarks leaderboard: Mrcr, ruler, and longbench v2.” https://awesomeagents.ai/leaderboards/ long-context-benchmarks-leaderboard/, February 2026. Awesome Agents — AI models benchmarks and leaderboards.

[6] S. Gammon, “Statistics on the world’s largest github monorepos.” https://stacey-gammon.github.io/repo-stats/, 2026. GitHub Pages site with aggregated repository size and LOC statistics.

[7] Y. Du, M. Tian, S. Ronanki, S. Rongali, S. Bodapati, A. Galstyan, A. Wells, R. Schwartz, E. A. Huerta, and H. Peng, “Context length alone hurts llm performance despite perfect retrieval,” arXiv preprint arXiv:2510.05381, 2025.

[8] H. Firooz, M. Sanjabi, W. Jiang, and X. Zhai, “Lost-in-distance: Impact of contextual proximity on llm performance in graph tasks,” arXiv preprint arXiv:2410.01985, 2024.

[9] A. N. Hoang, M. Le-Anh, B. Le, and N. D. Bui, “Codewiki: Evaluating ai’s ability to generate holistic documentation for large-scale codebases,” arXiv preprint arXiv:2510.24428, 2025.

[10] Q. Luo, Y. Ye, S. Liang, Z. Zhang, Y. Qin, Y. Lu, Y. Wu, X. Cong, Y. Lin, Y. Zhang, et al., “Repoagent: An llm-powered open-source framework for repository-level code documentation generation,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 436–464, 2024.

[11] D. Sounthiraraj, J. Hancock, Y. Kortam, A. Javvaji, P. Singh, and S. Shankar, “Code-craft: Hierarchical graph-based code summarization for enhanced context retrieval,” arXiv preprint arXiv:2504.08975, 2025.

[12] F. Hurley, P. Rodriguez, R. Marques, and O. Shams, “Introducing code wiki: Accelerating your code understanding.” https://developers.googleblog.com/ introducing-code-wiki-accelerating-your-code-understanding/, November 2025. Google Developers Blog announcement of Code Wiki, an AI-powered code documentation and exploration platform.

[13] J. Zhang, X. Wang, H. Zhang, H. Sun, K. Wang, and X. Liu, “A novel neural source code representation based on abstract syntax tree,” in 2019 IEEE/ACM 41st International Conference on Software Engineering (ICSE), pp. 783–794, IEEE, 2019.

[14] Z. Shi, Y. Xiong, X. Zhang, Y. Zhang, S. Li, and Y. Zhu, “Crossmodal contrastive learning for code search,” in 2022 IEEE International Conference on Software Maintenance and Evolution (ICSME), pp. 94– 105, IEEE, 2022.

[15] H. Li, X. Zhou, L. A. Tuan, and C. Miao, “Rethinking negative pairs in code search,” in Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing, pp. 12760–12774, 2023.

[16] W. Sun, C. Fang, Y. Chen, G. Tao, T. Han, and Q. Zhang, “Code search based on context-aware code translation,” in Proceedings of the 44th International Conference on Software Engineering (ICSE), pp. 388–400, ACM, 2022.

[17] Y. Chai, H. Zhang, B. Shen, and X. Gu, “Cross-domain deep code search with meta learning,” in Proceedings of the 44th International Conference on Software Engineering (ICSE), pp. 487–498, ACM, 2022.

[18] G. Mathew and K. T. Stolee, “Cross-language code search using static and dynamic analyses,” in Proceedings of the 29th ACM Joint Meeting on European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE), pp. 205–217, ACM, 2021.

[19] Y. Mao, C. Wan, Y. Jiang, and X. Gu, “Self-supervised query reformulation for code search,” in Proceedings of the 31st ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering (ESEC/FSE), pp. 22–34, ACM, 2023.

[20] S. Wang, M. Geng, B. Lin, Z. Sun, M. Wen, Y. Liu, L. Li, T. F. Bissyande, and X. Mao, “Fusing code searchers,” ´ IEEE Transactions on Software Engineering, vol. 50, no. 7, pp. 1852–1866, 2024.

[21] E. Shi, Y. Wang, L. Du, J. Chen, S. Han, H. Zhang, D. Zhang, and H. Sun, “Cocosoda: Effective contrastive learning for code search,” in 2023 IEEE/ACM 45th International Conference on Software Engineering (ICSE), pp. 2198–2210, IEEE, 2023.

[22] X. Li, Y. Gong, Y. Shen, X. Qiu, H. Zhang, B. Yao, W. Qi, D. Jiang, W. Chen, and N. Duan, “Coderetriever: A large scale contrastive pretraining method for code search,” in Proceedings of the 2022 Conference on Empirical Methods in Natural Language Processing, pp. 2898–2910, Association for Computational Linguistics, 2022.

[23] L. Di Grazia and M. Pradel, “Code search: A survey of techniques for finding code,” ACM Computing Surveys, vol. 55, no. 11, 2023.

[24] W. Sun, C. Fang, Y. Ge, Y. Hu, Y. Chen, Q. Zhang, X. Ge, Y. Liu, and Z. Chen, “A survey of source code search: A 3-dimensional perspective,” ACM Transactions on Software Engineering and Methodology, vol. 33, no. 6, 2024.

[25] Q. Luo, Y. Ye, S. Liang, Z. Zhang, Y. Qin, Y. Lu, Y. Wu, X. Cong, Y. Lin, Y. Zhang, et al., “Repoagent: An llm-powered open-source framework for repository-level code documentation generation,” in Proceedings of the 2024 Conference on Empirical Methods in Natural Language Processing: System Demonstrations, pp. 436–464, 2024.

[26] A. N. Hoang, M. Le-Anh, B. Le, and N. D. Bui, “Codewiki: Evaluating ai’s ability to generate holistic documentation for large-scale codebases,” arXiv preprint arXiv:2510.24428, 2025.

[27] D. Sounthiraraj, J. Hancock, Y. Kortam, A. Javvaji, P. Singh, and S. Shankar, “Code-craft: Hierarchical graph-based code summarization for enhanced context retrieval,” arXiv preprint arXiv:2504.08975, 2025.

[28] A. Neelakantan, T. Xu, R. Puri, A. Radford, J. M. Han, J. Tworek, Q. Yuan, N. Tezak, J. W. Kim, C. Hallacy, et al., “Text and code embeddings by contrastive pre-training,” 2022.

[29] T.-S. Contributors, “Tree-sitter: Incremental parsing system.” https:// tree-sitter.github.io/tree-sitter/index.html, 2026. Official documentation for the Tree-Sitter incremental parsing library.

[30] N. Developers, “Networkx — python package for the creation, manipulation, and study of complex networks.” https://networkx.org/en/, 2026. Official documentation and reference for the NetworkX library.

[31] S. Ram´ırez, “Fastapi – fast, high-performance, web framework for building apis with python.” https://fastapi.tiangolo.com/, 2026. Official documentation for the FastAPI framework.

[32] Y. Bai, X. Wu, T. Wei, and L. He, “A dual-encoder contrastive learning model for knowledge tracing,” Entropy, vol. 27, no. 7, p. 685, 2025.

[33] Y. Zhao, A. Gu, R. Varma, L. Luo, C.-C. Huang, M. Xu, L. Wright, H. Shojanazeri, M. Ott, S. Shleifer, A. Desmaison, C. Balioglu, P. Damania, B. Nguyen, G. Chauhan, Y. Hao, A. Mathews, and S. Li, “Pytorch fsdp: Experiences on scaling fully sharded data parallel,” Proc. VLDB Endow., vol. 16, p. 3848–3860, Aug. 2023.

[34] N. Reimers and I. Gurevych, “Sentence-bert: Sentence embeddings using siamese bert-networks,” in Proceedings of the 2019 conference on empirical methods in natural language processing and the 9th international joint conference on natural language processing (EMNLP-IJCNLP), pp. 3982–3992, 2019.

[35] J. Devlin, M.-W. Chang, K. Lee, and K. Toutanova, “Bert: Pre-training of deep bidirectional transformers for language understanding,” in Proceedings of the 2019 conference of the North American chapter of the association for computational linguistics: human language technologies, volume 1 (long and short papers), pp. 4171–4186, 2019.

[36] S. Es, J. James, L. E. Anke, and S. Schockaert, “Ragas: Automated evaluation of retrieval augmented generation,” in Proceedings ofthe 18th conference of the european chapter of the association for computational linguistics: system demonstrations, pp. 150–158, 2024.

[37] S. Roychowdhury, S. Soman, H. Ranjani, N. Gunda, V. Chhabra, and S. Bala, “Evaluation of RAG metrics for question answering in the telecom domain,” arXiv preprint arXiv:2407.12873, 2024.